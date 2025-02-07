from itertools import product
from collections import defaultdict

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
    
    # Yield each shared filler of two contexts TODO: not str, tup
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
    
    # Return the filler finder trie of a context TODO: not str, tup
    def get_context_node(self, context):
        """TODO: not str, tup. Return the filler finder trie of `context`.
        
        Argument:
            - context (string): e.g. 'a _ king'
        
        Returns:
            - DistrNode: the node that is the root of the trie of the fillers
              that occurred in `context`
        """
        #left_context, right_context = map(lambda x: x.strip().split(), context.split('_'))
        slot_index = context.index('_')
        left_context, right_context = context[:slot_index], context[slot_index + 1:]
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
                    form = tuple(new_path)
                    yield (form, freq1, freq2)
                yield from self.shared_branches(child_node1, child_node2, new_path)
        
    # Yield each shared context of two fillers TODO: not str, tup
    def shared_contexts(self, filler1, filler2, max_length=float('inf')):
        """Yield each shared context of `filler1` and `filler2`.
        
        Arguments:
            - filler1 (string): e.g. 'the king'
            - filler2 (string): e.g. 'this garden'
        
        Returns:
            - generator (of strings): e.g. ('visited _ today', 'i saw _', ...)
        """
        filler_node1 = self.get_filler_node(filler1)
        filler_node2 = self.get_filler_node(filler2)
        return self.shared_contexts_aux(filler_node1, filler_node2, max_length)
    
    # Return the node of a filler TODO: not str, tup
    def get_filler_node(self, filler):
        """TODO: not str, tup. Return the context finder node of `filler`.
        
        Argument:
            - filler (string): e.g. 'the nice king'
        
        Returns:
            - DistrNode: the main trie node of `filler`
        """
        #filler = filler.split()
        filler_node = self.root
        for i, word in enumerate(filler):
            try:
                filler_node = filler_node.children[word]
            except KeyError:
                failed_part = ' '.join(filler[:i+1]) + ' ...'
                raise KeyError(
                    f'Filler \"{filler}\" not found (failed at \"{failed_part}\")'
                )
        return filler_node
    
    # Recursively yield each shared context of two filler nodes
    def shared_contexts_aux(self, filler_node1, filler_node2, max_length=float('inf'), shared_right_context=[]):
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
            if len(shared_left_context) + len(shared_right_context) > max_length:
                return
            shared_context = (
                shared_left_context
                + ('_',)
                + tuple(shared_right_context)
            )
            yield (shared_context, context_freq1, context_freq2)
        # Recursive traversal of each shared child of the fillers, to cover all
        # shared right contexts
        if len(shared_right_context) >= max_length:
            return
        for child in filler_node1.children:
            if child in filler_node2.children:
                new_shared_right_context = shared_right_context + [child]
                child_node1 = filler_node1.children[child]
                child_node2 = filler_node2.children[child]
                # Yield newly found shared right context by itself
                shared_context = ('_',) + tuple(new_shared_right_context)
                context_freq1 = child_node1.count
                context_freq2 = child_node2.count
                yield (shared_context, context_freq1, context_freq2)
                # Recursive call on new shared right context and new child
                # nodes, to find the shared left contexts within this new right
                # context
                yield from self.shared_contexts_aux(
                    child_node1,
                    child_node2,
                    max_length,
                    new_shared_right_context
                )
    
    # From here on: contexts and fillers are tup
    def get_fillers(self, context, max_length=float('inf')):
        context_node = self.get_context_node(context)
        return self.get_branches(context_node, max_length)
    
    def get_branches(self, current_node, max_length=float('inf'), path=[]):
        if len(path) >= max_length:
            return
        for child in current_node.children:
            new_path = path + [child]
            child_node = current_node.children[child]
            if child_node.count > 0:
                branch = tuple(new_path)
                freq = child_node.count
                yield (branch, freq)
            yield from self.get_branches(child_node, max_length, new_path)
    
    def get_contexts(self, filler, max_length=float('inf')):
        filler_node = self.get_filler_node(filler)
        return self.get_contexts_aux(filler_node, max_length)
    
    def get_contexts_aux(self, filler_node, max_length=float('inf'), right_context=[]):
        # Find all shared left contexts within the current shared right context
        left_context_node = filler_node.finder.root
        left_context_infos = self.get_branches(left_context_node)
        for left_context_info in left_context_infos:
            left_context, freq = left_context_info
            if len(left_context) + len(right_context) > max_length:
                return
            context = (
                left_context
                + ('_',)
                + tuple(right_context)
            )
            yield (context, freq)
        # Recursive traversal of each shared child of the fillers, to cover all
        # shared right contexts
        if len(right_context) >= max_length:
            return
        for child in filler_node.children:
            new_right_context = right_context + [child]
            child_node = filler_node.children[child]
            # Yield newly found shared right context by itself
            context = ('_',) + tuple(new_right_context)
            freq = child_node.count
            yield (context, freq)
            # Recursive call on new shared right context and new child
            # nodes, to find the shared left contexts within this new right
            # context
            yield from self.get_contexts_aux(
                child_node,
                max_length,
                new_right_context
            )
    
    def anl_paths(self, context, filler):
        anl_path_infos = []
        org_ctxt_freq = self.get_context_node(context).context_count
        anl_fillers = self.get_fillers(context, len(filler))
        for anl_filler, org_ctxt_anl_fllr_freq in anl_fillers:
            anl_fllr_freq = self.get_filler_node(anl_filler).count
            # Calculate probability of moving from original context to
            # analogical filler
            org_ctxt_anl_fllr_prob = org_ctxt_anl_fllr_freq / org_ctxt_freq
            # Loop over all shared contexts of analogical filler and filler
            # to find analogical contexts
            anl_contexts = self.shared_contexts(anl_filler, filler)
            for anl_context, anl_ctxt_anl_fllr_freq, anl_ctxt_org_fllr_freq in anl_contexts:
                anl_ctxt_freq = self.get_context_node(anl_context).context_count
                # Calculate weight of moving from analogical filler to
                # analogical context and then from analogical context to filler
                anl_ctxt_anl_fllr_prob = anl_ctxt_anl_fllr_freq / anl_fllr_freq
                anl_ctxt_org_fllr_prob = anl_ctxt_org_fllr_freq / anl_ctxt_freq
                # Calculate and record full weight of analogical path
                anl_path_prob = (
                      org_ctxt_anl_fllr_prob
                    * anl_ctxt_anl_fllr_prob
                    * anl_ctxt_org_fllr_prob
                )
                anl_path_info = ((anl_context, anl_filler), anl_path_prob)
                anl_path_infos.append(anl_path_info)
        filler_dict = {}
        for path, score in anl_path_infos:
            if path[1] in filler_dict:
                filler_dict[path[1]] += score
            else:
                filler_dict[path[1]] = score
        return sorted(filler_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def get_indirect_fillers(self, right_context, max_length=float('inf')):
        context_node = self.get_context_node(right_context)
        return self.get_indirect_fillers_aux(context_node, max_length, path=[])

    def get_indirect_fillers_aux(self, context_node, max_length, path):
        if len(path) >= max_length:
            return
        for child in context_node.children:
            new_path = path + [child]
            child_node = context_node.children[child]
            if child_node.context_count > 0:
                filler = tuple(new_path)
                freq = child_node.context_count
                yield (filler, freq)
            yield from self.get_indirect_fillers_aux(child_node, max_length, new_path)
    
    def rec_anls(self, gram, lookup_dy=None):
        if lookup_dy is None:
            lookup_dy = {}
        # End of recursion
        if len(gram) == 1:
            return [(gram, 1)]
        # Check dynamic lookup table
        if gram in lookup_dy:
            return lookup_dy[gram]
        # Gram is context
        if '_' in gram:
            slot_index = gram.index('_')
            left_context, right_context = gram[:slot_index], gram[slot_index + 1:]
            if slot_index in {0, len(gram) - 1}:
                context = left_context + right_context
                anl_contexts = self.rec_anls(context, lookup_dy)[context]
                is_left = int(slot_index == 0)
                context_format = lambda x: is_left * ('_',) + x + (1 - is_left) * ('_',)
                anl_grams = [
                    (context_format(anl_context), score)
                    for anl_context, score in anl_contexts
                ]
                lookup_dy[gram] = anl_grams
                return lookup_dy
            else:
                anl_left_contexts = self.rec_anls(left_context, lookup_dy)[left_context]
                anl_right_contexts = self.rec_anls(right_context, lookup_dy)[right_context]
                anl_context_pairs = product(anl_left_contexts, anl_right_contexts)
                anl_contexts = defaultdict(float)
                for anl_left_context, anl_right_context in anl_context_pairs:
                    subst_contexts = self.indir_anl_paths(anl_left_context, anl_right_context)
                    for subst_context, score in subst_contexts:
                        anl_contexts[subst_context] += score
                anl_grams = sorted(anl_contexts.items(), key=lambda x: x[1], reverse=True)[:5]
                lookup_dy[gram] = anl_grams
                return lookup_dy
        
        # Find analogies using each context-filler split
        context_filler_pairs = (
            ((sentence[:i], sentence[j:]), sentence[i:j])
            for i in range(len(sentence) + 1)
            for j in range(i + 1, len(sentence) + int(i > 0))
        )
        anl_grams = defaultdict(float)
        for context, filler in context_filler_pairs:
            anl_context_infos = self.rec_anls(context, lookup_dy)[context]
            anl_fillers_infos = self.rec_anls(filler, lookup_dy)[filler]
            for anl_context_info, anl_filler_info in product(anl_context_infos, anl_fillers_infos):
                anl_context, anl_context_score = anl_context_info
                anl_filler, anl_filler_score = anl_filler_info
                subst_filler_infos = self.anl_paths(anl_context, anl_filler)
                for subst_filler_info in subst_filler_infos:
                    subst_filler, subst_filler_score = subst_filler_info
                    slot_index = anl_context.index('_')
                    anl_gram = (
                          anl_context[:slot_index]
                        + subst_filler
                        + anl_context[slot_index+1:]
                    )
                    anl_grams[anl_gram] += subst_filler_score * anl_context_score * anl_filler_score
        anl_grams = sorted(anl_grams.items(), key=lambda x: x[1], reverse=True)[:5]
        lookup_dy[gram] = anl_grams
        return lookup_dy

    def indir_anl_paths(self, left_context, right_context):
        path_infos = defaultdict(float)
        indir_left_context = left_context + ('_',)
        indir_right_context = ('_',) + right_context
        indir_lr_paths = self.anl_paths(indir_left_context, right_context)
        indir_rl_paths = self.anl_paths(indir_right_context, left_context)
        indir_lr_prob = sum(score for path, score in indir_lr_paths)
        indir_rl_prob = sum(score for path, score in indir_rl_paths)
        for lr_path, score in indir_lr_paths:
            try:
                ctxt_freq = self.get_context_node(indir_left_context + lr_path).context_count
                left_freq = self.get_context_node(indir_left_context).context_count
                right_freq = self.get_filler_node(lr_path).count
                rel_freq = ctxt_freq / right_freq
                path_infos[indir_left_context + lr_path] += score * indir_lr_prob * rel_freq
            except:
                continue
        for rl_path, score in indir_rl_paths:
            try:
                ctxt_freq = self.get_context_node(rl_path + indir_right_context).context_count
                right_freq = self.get_context_node(indir_right_context).context_count
                left_freq = self.get_filler_node(rl_path).count
                rel_freq = ctxt_freq / left_freq
                path_infos[rl_path + indir_right_context] += score * indir_rl_prob * rel_freq
            except:
                continue
        for lr_path_info, rl_path_info in product(indir_lr_paths[:10], indir_rl_paths[:10]):
            lr_path, lr_score = lr_path_info
            rl_path, rl_score = rl_path_info
            anl_context = rl_path + ('_',) + lr_path
            try:
                ctxt_freq = self.get_context_node(anl_context).context_count
                score = min(lr_score, rl_score)
                indir_prob = min(indir_lr_prob, indir_rl_prob)
                left_freq = self.get_filler_node(rl_path).count
                right_freq = self.get_filler_node(lr_path).count
                rel_freq = ctxt_freq / (left_freq * right_freq)
                path_infos[anl_context] += score * indir_prob * rel_freq
            except:
                continue
        return sorted(path_infos.items(), key=lambda x: x[1], reverse=True)[:5]