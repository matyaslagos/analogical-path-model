from enum import Enum, auto
from collections import defaultdict, namedtuple
from copy import deepcopy
from pprint import pp

# Setup functions

# Import a txt file (containing one sentence per line) as a list whose each
# element is a list of words corresponding to a line in the txt file:
def txt2list(filename):
    """Import a txt list of sentences as a list of tuples of words.

    Argument:
        - filename (string): e.g. 'grimm_corpus_no_commas.txt'

    Returns:
        - list (of tuples of strings): e.g.
          [('my', 'name', 'is', 'jolán'), ('i', 'am', 'cool'), ..., ('bye',)]
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return [tuple(line.strip().split()) for line in lines]

def corpus_setup():
    return txt2list('grimm_full_no_commas.txt')

def distrtrie_setup(corpus):
    ddy = DistrTrie()
    for sentence in corpus:
        ddy.insert_distr(sentence)
    return ddy


# Trie class for recording distribution information about corpus



class FreqNode:
    def __init__(self, label):
        self.label = label
        self.children = {}
        self.count = 0
        self.context_count = 0
    
    def get_or_make_branch(self, tuple_of_strings):
        current_node = self
        for word in tuple_of_strings:
            current_node.context_count += 1
            current_node = current_node.get_or_make_child(word)
        current_node.count += 1
        return current_node
    
    def get_or_make_child(self, child_label):
        if child_label not in self.children:
            child_type = type(self)
            self.children[child_label] = child_type(child_label)
        return self.children[child_label]

class FreqTrie:
    def __init__(self):
        self.root = FreqNode('~')

class DistrNode(FreqNode):
    def __init__(self, label):
        super().__init__(label)
        self.finder = FreqTrie() # Left-to-right left contexts

class DistrTrie:
    def __init__(self):
        self.root = DistrNode('~')
     
    # Record distribution information about a sentence
    def insert_distr(self, sentence):
        """Record all contexts and fillers of `sentence` into trie.
        
        Arguments:
            - sentence (tuple of strings): e.g. ('the', 'king', 'was', 'here')
        
        Effect:
            For each prefix--suffix pair of `sentence`, records the suffix as
            a branch, and for each word in this branch, records all suffixes
            of the prefix as branches.
            The resulting trie structure can be used to look up shared fillers
            of two contexts or shared contexts of two fillers.
            In the former case:
            - main branches act as fillers and right contexts, and
            - finder branches act as left contexts.
            In the latter case:
            - main branches act as right contexts, and
            - finder branches act as left contexts and fillers.
        """
        context_pairs = ((sentence[:i], sentence[i:]) for i in range(len(sentence)))
        for left_context, right_context in context_pairs:
            left_context_suffixes = [left_context[j:] for j in range(len(left_context))]
            current_node = self.root
            for word in right_context:
                current_node = current_node.get_or_make_child(word)
                current_node.count += 1
                current_node.context_count += 1
                # Record all suffixes of left-to-right context
                finder_node = current_node.finder.root
                for left_context_suffix in left_context_suffixes:
                    finder_node.get_or_make_branch(left_context_suffix)
    
    # Yield each shared filler of two contexts
    def shared_fillers(self, context1, context2):
        """Yield each shared filler of `context1` and `context2`.
        
        Arguments:
            - context1 (string): e.g. 'a _ garden'
            - context2 (string): e.g. 'this _ lake'
        
        Returns:
            - generator (of strings): e.g. ('beautiful', 'very nice', ...)
        """
        context_node1 = self.get_context_node(context1)
        context_node2 = self.get_context_node(context2)
        return self.shared_branches(context_node1, context_node2)
    
    # Return the filler finder trie of a context
    def get_context_node(self, context):
        """Return the filler finder trie of `context`.
        
        Argument:
            - context (string): e.g. 'a _ king'
        
        Returns:
            - DistrNode: the node that is the root of the trie of the fillers
              that occurred in `context`
        """
        left_context, right_context = map(lambda x: x.strip().split(), context.split('_'))
        context_node = self.root
        current_node = self.root
        # Go to node of right context
        for i, word in enumerate(right_context):
            try:
                current_node = current_node.children[word]
                context_node = current_node.finder.root
            except KeyError:
                failed_part = '_ ' + ' '.join(right_context[:i+1]) + ' ...'
                raise KeyError(
                    f'Context \"{context}\" not found (failed at \"{failed_part}\")'
                    )
        # Within the filler finder of right context, go to node of left context
        for i, word in enumerate(left_context):
            try:
                context_node = context_node.children[word]
            except KeyError:
                failed_part = ' '.join(left_context[:i+1]) + ' ... _ ' + ' '.join(right_context)
                raise KeyError(
                    f'Context \"{context}\" not found (failed at \"{failed_part}\")'
                    )
        return context_node
    
    # Recursively yield each shared filler of two context nodes
    def shared_branches(self, distr_node1, distr_node2, path=[]):
        """Yield each shared branch of `distr_node1` and `distr_node2`.
        
        Arguments:
            - distr_node1 (DistrNode): root of a subtrie
            - distr_node2 (DistrNode): root of another subtrie
        
        Yields:
            - string: branch that is shared by the two input subtries
        """
        for child in distr_node1.children:
            if child in distr_node2.children:
                new_path = path + [child]
                child_node1 = distr_node1.children[child]
                child_node2 = distr_node2.children[child]
                if child_node1.count > 0 and child_node2.count > 0:
                    freq1 = child_node1.count
                    freq2 = child_node2.count
                    form = ' '.join(new_path)
                    yield (form, freq1, freq2)
                yield from self.shared_branches(child_node1, child_node2, new_path)
        
    # Yield each shared context of two fillers
    def shared_contexts(self, filler1, filler2):
        """Yield each shared context of `filler1` and `filler2`.
        
        Arguments:
            - filler1 (string): e.g. 'the king'
            - filler2 (string): e.g. 'this garden'
        
        Returns:
            - generator (of strings): e.g. ('visited _ today', 'i saw _', ...)
        """
        filler_node1 = self.get_filler_node(filler1)
        filler_node2 = self.get_filler_node(filler2)
        return self.shared_contexts_aux(filler_node1, filler_node2)
    
    # Return the node of a filler
    def get_filler_node(self, filler):
        """Return the context finder node of `filler`.
        
        Argument:
            - filler (string): e.g. 'the nice king'
        
        Returns:
            - DistrNode: the main trie node of `filler`
        """
        filler_words_list = filler.split()
        filler_node = self.root
        for i, word in enumerate(filler_words_list):
            try:
                filler_node = filler_node.children[word]
            except KeyError:
                failed_part = ' '.join(filler_list[:i+1]) + ' ...'
                raise KeyError(
                    f'Filler \"{filler}\" not found (failed at \"{failed_part}\")'
                )
        return filler_node
    
    # Recursively yield each shared context of two filler nodes
    def shared_contexts_aux(self, filler_node1, filler_node2, shared_right_context=[]):
        """Yield each shared context of `filler_node1` and `filler_node2`.
        
        Arguments:
            - filler_node1 (DistrNode): node where children are considered to
              be trie of right contexts and .finder trie is considered to
              be trie of left contexts
            - filler_node1 (DistrNode): as filler_node1
        
        Yields:
            - string: shared context of fillers, e.g. 'visited _ today'
        """
        # Find all shared left contexts within the current shared right context
        left_contexts1 = filler_node1.finder.root
        left_contexts2 = filler_node2.finder.root
        shared_left_context_infos = self.shared_branches(left_contexts1, left_contexts2)
        for shared_left_context_info in shared_left_context_infos:
            shared_left_context, context_freq1, context_freq2 = shared_left_context_info
            shared_context = (
                shared_left_context
                + ' _ '
                + ' '.join(shared_right_context)
            ).strip()
            yield (shared_context, context_freq1, context_freq2)
        # Recursive traversal of each shared child of the fillers, to cover all
        # shared right contexts
        for child in filler_node1.children:
            if child in filler_node2.children:
                new_shared_right_context = shared_right_context + [child]
                child_node1 = filler_node1.children[child]
                child_node2 = filler_node2.children[child]
                # Yield newly found shared right context by itself
                shared_context = '_ ' + ' '.join(new_shared_right_context)
                context_freq1 = child_node1.count
                context_freq2 = child_node2.count
                yield (shared_context, context_freq1, context_freq2)
                # Recursive call on new shared right context and new child
                # nodes, to find the shared left contexts within this new right
                # context
                yield from self.shared_contexts_aux(
                    child_node1,
                    child_node2,
                    new_shared_right_context
                )
    
    def get_fillers(self, context):
        context_node = self.get_context_node(context)
        return self.get_fillers_aux(context_node, path=[])
    
    def get_fillers_aux(self, context_node, path):
        for child in context_node.children:
            new_path = path + [child]
            child_node = context_node.children[child]
            if child_node.count > 0:
                filler = ' '.join(new_path)
                freq = child_node.count
                yield (filler, freq)
            yield from self.get_fillers_aux(child_node, new_path)
    
    def analogical_paths(self, context, filler):
        anl_path_infos = []
        freq_dict = {}
        freq_dict[context] = self.get_context_node(context).context_count
        # Loop over all fillers of context to find analogical fillers
        anl_fillers = self.get_fillers(context)
        for anl_filler, first_step_freq in anl_fillers:
            freq_dict[anl_filler] = self.get_filler_node(anl_filler).count
            freq_dict[(context, anl_filler)] = first_step_freq
            # Calculate weight of moving from context to analogical filler
            first_step_weight = freq_dict[(context, anl_filler)] / freq_dict[context]
            # Loop over all shared contexts of analogical filler and filler
            # to find analogical contexts
            anl_contexts = self.shared_contexts(anl_filler, filler)
            for anl_context, second_step_freq, third_step_freq in anl_contexts:
                freq_dict[anl_context] = self.get_context_node(anl_context).context_count
                freq_dict[(anl_context, anl_filler)] = second_step_freq
                freq_dict[(anl_context, filler)] = third_step_freq
                # Calculate weight of moving from analogical filler to
                # analogical context and then from analogical context to filler
                second_step_weight = freq_dict[(anl_context, anl_filler)] / freq_dict[anl_filler]
                third_step_weight = freq_dict[(anl_context, filler)] / freq_dict[anl_context]
                # Calculate and record full weight of analogical path
                anl_path_weight = first_step_weight * second_step_weight * third_step_weight
                anl_path_info = ((anl_context, anl_filler), anl_path_weight)
                anl_path_infos.append(anl_path_info)
        return sorted(anl_path_infos, key=lambda x: x[1], reverse=True)