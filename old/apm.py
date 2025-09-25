from itertools import product
from collections import defaultdict
from string import punctuation
import math
import csv

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

def txt2wordlist(filename):
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    clean_word = lambda s: s.strip(punctuation + ' ').lower()
    return [
        ('#',) + tuple(word) + ('/#',)
        for line in lines
        for word in map(clean_word, line.split())
        if word.isalpha()
    ]

def csv2list(filename):
    with open(filename, newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file, delimiter=';')
        csv_as_list = [('<s>',) + tuple(row[0]) + ('</s>',) for row in reader]
    return csv_as_list

def ngrams(corpus, n):
    ngram_corpus = []
    for sentence in corpus:
        endmarked_sentence = ('<s>',) * (n-1) + sentence + ('</s>',)
        ngrams_of_sentence = zip(*(endmarked_sentence[i:] for i in range(n)))
        for ngram in ngrams_of_sentence:
            ngram_corpus.append(ngram)
    return ngram_corpus

def corpus_setup():
    return ngrams(txt2list('grimm_full_with_commas.txt'), 7)

def distrtrie_setup(corpus):
    ddy = DistrTrie()
    for sentence in corpus:
        ddy.insert_distr(sentence)
    return ddy

def distrtrie_setup_counted(corpus_with_freqs):
    ddy = DistrTrie()
    for gram, freq in corpus_with_freqs:
        ddy.insert_distr_ngram_with_freq(gram, freq)
    return ddy

# Trie class for recording distribution information about corpus

class FreqNode:
    def __init__(self, label):
        self.label = label
        self.children = {}
        self.count = 0
        self.context_count = 0
    
    def get_or_make_branch(self, tuple_of_strings, freq=1):
        current_node = self
        for word in tuple_of_strings:
            current_node.context_count += freq
            current_node = current_node.get_or_make_child(word)
        current_node.count += freq
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
    
    # Record distribution information about a sentence
    def insert_distr_ngram(self, ngram):
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
        pref_suff_pairs = ((ngram[:i], ngram[i:]) for i in range(len(ngram)))
        for prefix, suffix in pref_suff_pairs:
            suffs_of_pref = (prefix[i:] for i in range(len(prefix)))
            current_node = self.root.get_or_make_branch(suffix)
            for suff_of_pref in suffs_of_pref:
                finder_node = current_node.finder.root
                finder_node.get_or_make_branch(suff_of_pref)
    
    # Record distribution information about a sentence
    def insert_distr_ngram_with_freq(self, ngram, freq):
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
        pref_suff_pairs = ((ngram[:i], ngram[i:]) for i in range(len(ngram)))
        for prefix, suffix in pref_suff_pairs:
            suffs_of_pref = (prefix[i:] for i in range(len(prefix)))
            current_node = self.root.get_or_make_branch(suffix, freq)
            for suff_of_pref in suffs_of_pref:
                finder_node = current_node.finder.root
                finder_node.get_or_make_branch(suff_of_pref, freq)
    
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
                return None
                """
                raise KeyError(
                    f'Context \"{context}\" not found (failed at \"{failed_part}\")'
                    )
                """
        # Within the filler finder of right context, go to node of left context
        for i, word in enumerate(left_context):
            try:
                context_node = context_node.children[word]
            except KeyError:
                failed_part = ' '.join(left_context[:i+1]) + ' ... _ ' + ' '.join(right_context)
                return None
                """
                raise KeyError(
                    f'Context \"{context}\" not found (failed at \"{failed_part}\")'
                    )
                """
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
                return None
                """
                raise KeyError(
                    f'Filler \"{filler}\" not found (failed at \"{failed_part}\")'
                )
                """
        return filler_node
    
    # Recursively yield each shared context of two filler nodes
    def shared_contexts_aux(self, filler_node1, filler_node2, max_length=float('inf'), shared_right_context=[], found_left_contexts=set()):
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
        return self.get_filler_branches(context_node, max_length)
    
    def get_filler_branches(self, current_node, max_length=float('inf'), path=[]):
        if len(path) >= max_length:
            return
        for child in current_node.children:
            new_path = path + [child]
            child_node = current_node.children[child]
            if child_node.count > 0:
                branch = tuple(new_path)
                freq = child_node.count
                yield (branch, freq)
            yield from self.get_filler_branches(child_node, max_length, new_path)

    def get_context_branches(self, current_node, max_length=float('inf'), path=[]):
        if len(path) >= max_length:
            return
        for child in current_node.children:
            new_path = path + [child]
            child_node = current_node.children[child]
            if child_node.context_count > 0:
                branch = tuple(new_path)
                freq = child_node.context_count
                yield (branch, freq)
            yield from self.get_context_branches(child_node, max_length, new_path)

    def get_contexts(self, filler, max_length=float('inf')):
        filler_node = self.get_filler_node(filler)
        return self.get_contexts_aux(filler_node, max_length)
    
    def get_contexts_aux(self, filler_node, max_length=float('inf'), right_context=[]):
        # Find all shared left contexts within the current shared right context
        left_context_node = filler_node.finder.root
        left_context_infos = self.get_context_branches(left_context_node)
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
            freq = child_node.context_count
            yield (context, freq)
            # Recursive call on new shared right context and new child
            # nodes, to find the shared left contexts within this new right
            # context
            yield from self.get_contexts_aux(
                child_node,
                max_length,
                new_right_context
            )
    
    def rec_anls(self, gram, lookup_dy=None):
        if lookup_dy is None:
            lookup_dy = {}
        # Check dynamic lookup table
        if gram in lookup_dy:
            return lookup_dy
        # End of recursion
        if len(gram) == 1:
            lookup_dy[gram] = [(gram, 1)]
            return lookup_dy
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
                for anl_left_context_info, anl_right_context_info in anl_context_pairs:
                    anl_left_context, anl_left_context_score = anl_left_context_info
                    anl_right_context, anl_right_context_score = anl_right_context_info
                    subst_contexts = self.indir_anl_paths(anl_left_context, anl_right_context)
                    subst_score = sum(x[1] for x in subst_contexts)
                    for subst_context, score in subst_contexts:
                        anl_contexts[subst_context] += score * anl_left_context_score * anl_right_context_score * subst_score
                anl_grams = sorted(anl_contexts.items(), key=lambda x: x[1], reverse=True)[:10]
                lookup_dy[gram] = anl_grams
                return lookup_dy
        
        # Find analogies using each context-filler split
        context_filler_pairs = [
            (gram[:i] + ('_',), gram[i:])
            for i in range(1,len(gram))
        ] + [
            (('_',) + gram[i:], gram[:i])
            for i in range(1,len(gram))
        ]
        
        anl_grams = defaultdict(float)
        for context, filler in context_filler_pairs:
            anl_context_infos = self.rec_anls(context, lookup_dy)[context]
            anl_fillers_infos = self.rec_anls(filler, lookup_dy)[filler]
            for anl_context_info, anl_filler_info in product(anl_context_infos, anl_fillers_infos):
                anl_context, anl_context_score = anl_context_info
                anl_filler, anl_filler_score = anl_filler_info
                subst_filler_infos = self.anl_paths(anl_context, anl_filler)
                subst_score = sum(x[1] for x in subst_filler_infos)
                for subst_filler_info in subst_filler_infos:
                    subst_filler, subst_filler_score = subst_filler_info
                    slot_index = anl_context.index('_')
                    anl_gram = (
                          anl_context[:slot_index]
                        + subst_filler
                        + anl_context[slot_index+1:]
                    )
                    anl_grams[anl_gram] += subst_filler_score * anl_context_score * anl_filler_score * subst_score
        anl_grams = sorted(anl_grams.items(), key=lambda x: x[1], reverse=True)[:10]
        lookup_dy[gram] = anl_grams
        return lookup_dy
    
    def anl_paths(self, context, filler):
        anl_path_infos = {}
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
                if anl_filler in anl_path_infos:
                    pass
                else:
                    anl_path_infos[anl_filler] = [0,0]
            
                if context[0] == '_':
                    if anl_context[0] != '_':
                        anl_path_infos[anl_filler][1] += anl_path_prob
                    else:
                        anl_path_infos[anl_filler][0] += anl_path_prob
                elif context[-1] == '_':
                    if anl_context[-1] != '_':
                        anl_path_infos[anl_filler][0] += anl_path_prob
                    else:
                        anl_path_infos[anl_filler][1] += anl_path_prob
        return sorted(
            [(key, value[0] * value[1]) for key, value in anl_path_infos.items() if value[0] * value[1] > 0],
            key=lambda x: x[1],
            reverse=True
        )

def rec_anls_func(self, gram, lookup_dy=None):
    if lookup_dy is None:
        lookup_dy = {}
    # Check dynamic lookup table
    if gram in lookup_dy:
        return lookup_dy
    # End of recursion
    if len(gram) == 2 and '_' not in gram:
        lookup_dy[gram] = [((gram, (gram[0], gram[0])), 1)]
        return lookup_dy
    # Gram is context
    if '_' in gram:
        slot_index = gram.index('_')
        left_context, right_context = gram[:slot_index], gram[slot_index + 1:]
        if slot_index in {0, len(gram) - 1}:
            context = left_context + right_context
            anl_contexts = rec_anls_func(self, context, lookup_dy)[context][:20]
            is_left = int(slot_index == 0)
            context_format = lambda x: is_left * ('_',) + x + (1 - is_left) * ('_',)
            # NEW
            anl_grams = [
                ((context_format(anl_context[0]), anl_context[1]), score)
                for anl_context, score in anl_contexts
            ]
            lookup_dy[gram] = anl_grams
            return lookup_dy
        else:
            anl_left_contexts = rec_anls_func(self, left_context, lookup_dy)[left_context][:20]
            anl_right_contexts = rec_anls_func(self, right_context, lookup_dy)[right_context][:20]
            anl_context_pairs = product(anl_left_contexts, anl_right_contexts)
            anl_contexts = defaultdict(float)
            for anl_left_context_info, anl_right_context_info in anl_context_pairs:
                anl_left_context, anl_left_context_score = anl_left_context_info
                anl_left_gram, anl_left_tree = anl_left_context
                anl_right_context, anl_right_context_score = anl_right_context_info
                subst_contexts = self.indir_anl_paths(anl_left_context, anl_right_context)
                subst_score = sum(x[1] for x in subst_contexts)
                for subst_context, score in subst_contexts:
                    anl_contexts[subst_context] += score * anl_left_context_score * anl_right_context_score * subst_score
            anl_grams = sorted(anl_contexts.items(), key=lambda x: x[1], reverse=True)[:10]
            lookup_dy[gram] = anl_grams
            return lookup_dy
    
    # Find analogies using each context-filler split
    context_filler_pairs = [
        (gram[:i] + ('_',), gram[i:])
        for i in range(1,len(gram))
    ] + [
        (('_',) + gram[i:], gram[:i])
        for i in range(1,len(gram))
    ]
    anl_grams = defaultdict(float)
    for context, filler in context_filler_pairs:
        try:
            anl_context_infos = rec_anls_func(self, context, lookup_dy)[context][:20]
            anl_fillers_infos = rec_anls_func(self, filler, lookup_dy)[filler][:20]
        except:
            continue
        for anl_context_info, anl_filler_info in product(anl_context_infos, anl_fillers_infos):
            anl_context, anl_context_score = anl_context_info
            # NEW
            anl_context_gram, anl_context_trees = anl_context
            anl_filler, anl_filler_score = anl_filler_info
            # NEW
            anl_filler_gram, anl_filler_trees = anl_filler
            # NEW
            subst_filler_infos = anl_paths_func(self, anl_context_gram, anl_filler_gram)
            subst_score = sum(x[1] for x in subst_filler_infos)
            slot_index = anl_context_gram.index('_')
            for subst_filler_info in subst_filler_infos:
                subst_filler, subst_filler_score = subst_filler_info
                # NEW
                anl_gram = (
                      anl_context_gram[:slot_index]
                    + subst_filler
                    + anl_context_gram[slot_index+1:]
                )
                # NEW
                anl_context_slotless_str = ' '.join(anl_context_gram[:slot_index] + anl_context_gram[slot_index+1:])
                subst_filler_str = ' '.join(subst_filler)
                if slot_index == 0:
                    anl_tree = ((subst_filler_str, anl_filler_trees[0]), (anl_context_slotless_str, anl_context_trees[0]))
                    tree_struct = (anl_filler_trees[1], anl_context_trees[1])
                else:
                    anl_tree = ((anl_context_slotless_str, anl_context_trees[0]), (subst_filler_str, anl_filler_trees[0]))
                    tree_struct = (anl_context_trees[1], anl_filler_trees[1])
                anl_form = (anl_gram, (anl_tree, tree_struct))
                anl_grams[anl_form] += subst_filler_score * min(anl_context_score, anl_filler_score) * subst_score
    anl_grams = sorted(anl_grams.items(), key=lambda x: x[1], reverse=True)
    lookup_dy[gram] = anl_grams
    return lookup_dy

def rec_anls_func_morph(self, gram, lookup_dy=None):
    if lookup_dy is None:
        lookup_dy = {}
    # Check dynamic lookup table
    if gram in lookup_dy:
        return lookup_dy
    # End of recursion
    if len(gram) == 1:
        lookup_dy[gram] = [(gram, 1)]
        return lookup_dy
    # Find analogies using each context-filler split
    context_filler_pairs = [
        (('_',) + gram[i:], gram[:i])
        for i in range(1, len(gram))
    ]
    anl_grams = defaultdict(float)
    for context, filler in context_filler_pairs:
        try:
            anl_fillers_infos = rec_anls_func_morph(self, filler, lookup_dy)[filler][:20]
        except:
            continue
        for anl_filler_info in anl_fillers_infos:
            anl_filler, anl_filler_score = anl_filler_info
            subst_filler_infos = anl_paths_func_morph(self, ('#',) + context + ('/#',), anl_filler)
            for subst_filler_info in subst_filler_infos:
                subst_filler, subst_filler_score = subst_filler_info
                anl_gram = (subst_filler + context[1:])
                subst_filler_str = ' '.join(subst_filler)
                anl_grams[anl_gram] += subst_filler_score * anl_filler_score
    anl_grams = sorted(anl_grams.items(), key=lambda x: x[1], reverse=True)
    lookup_dy[gram] = anl_grams
    return lookup_dy


def anl_paths_func(self, context, filler):
    anl_path_infos = {}
    org_ctxt_freq = self.get_context_node(context).context_count
    anl_fillers = sorted(list(self.get_fillers(context)), key=lambda x: x[1], reverse=True)[:10]
    for anl_filler, org_ctxt_anl_fllr_freq in anl_fillers:
        if '<s>' in anl_filler or len(anl_filler) == 1:
            continue
        anl_fllr_freq = self.get_filler_node(anl_filler).count
        # Calculate probability of moving from original context to
        # analogical filler
        org_ctxt_anl_fllr_prob = org_ctxt_anl_fllr_freq / org_ctxt_freq
        if anl_filler in anl_path_infos:
            pass
        else:
            anl_path_infos[anl_filler] = [0, 0, org_ctxt_anl_fllr_prob]
        # Loop over all shared contexts of analogical filler and filler
        # to find analogical contexts
        anl_contexts = sorted(list(self.shared_contexts(anl_filler, filler)), key=lambda x: min(x[1], x[2]), reverse=True)[:10]
        for anl_context, anl_ctxt_anl_fllr_freq, anl_ctxt_org_fllr_freq in anl_contexts:
            anl_ctxt_freq = self.get_context_node(anl_context).context_count
            # Calculate weight of moving from analogical filler to
            # analogical context and then from analogical context to filler
            if 0 in {anl_fllr_freq, anl_ctxt_freq}:
                continue
            anl_ctxt_anl_fllr_prob = anl_ctxt_anl_fllr_freq / anl_fllr_freq
            anl_ctxt_org_fllr_prob = anl_ctxt_org_fllr_freq / anl_ctxt_freq
            # Calculate and record full weight of analogical path
            anl_path_prob = (
                  anl_ctxt_anl_fllr_prob
                * anl_ctxt_org_fllr_prob
            )
            if context[0] == '_':
                if anl_context[0] != '_':
                    anl_path_infos[anl_filler][1] += anl_path_prob
                elif anl_context[0] == '_':
                    anl_path_infos[anl_filler][0] += anl_path_prob
            elif context[-1] == '_':
                if anl_context[-1] != '_':
                    anl_path_infos[anl_filler][0] += anl_path_prob
                elif anl_context[-1] == '_':
                    anl_path_infos[anl_filler][1] += anl_path_prob
    return sorted(
        [(key, (value[0] * value[1] * value[2])) for key, value in anl_path_infos.items() if (value[0] * value[1]) > 0],
        key=lambda x: x[1],
        reverse=True
    )

def anl_paths_func_morph(self, context, filler):
    anl_path_infos = {}
    context_node = self.get_context_node(context)
    if context_node is None:
        return [(tuple(), 0)]
    else:
        org_ctxt_freq = context_node.context_count
    anl_fillers = self.get_fillers(context)
    for anl_filler, org_ctxt_anl_fllr_freq in anl_fillers:
        filler_node = self.get_filler_node(filler)
        if filler_node is None:
            return [(tuple(), 0)]
        else:
            org_fllr_freq = filler_node.count
        anl_filler_node = self.get_filler_node(anl_filler)
        if anl_filler_node is None:
            return [(tuple(), 0)]
        else:
            anl_fllr_freq = anl_filler_node.count
        # Calculate probability of moving from original context to
        # analogical filler
        org_ctxt_anl_fllr_prob = org_ctxt_anl_fllr_freq / anl_fllr_freq
        if anl_filler in anl_path_infos:
            pass
        else:
            anl_path_infos[anl_filler] = 0
        # Loop over all shared contexts of analogical filler and filler
        # to find analogical contexts
        anl_contexts = self.shared_contexts(anl_filler, filler)
        for anl_context, anl_ctxt_anl_fllr_freq, anl_ctxt_org_fllr_freq in anl_contexts:
            anl_ctxt_freq = self.get_context_node(anl_context).context_count
            # Calculate weight of moving from analogical filler to
            # analogical context and then from analogical context to filler
            if 0 in {anl_fllr_freq, anl_ctxt_freq}:
                continue
            anl_ctxt_anl_fllr_prob = anl_ctxt_anl_fllr_freq / anl_ctxt_freq
            anl_ctxt_org_fllr_prob = anl_ctxt_org_fllr_freq / org_fllr_freq
            # Calculate and record full weight of analogical path
            anl_path_prob = (
                  org_ctxt_anl_fllr_prob
                * anl_ctxt_anl_fllr_prob
                * anl_ctxt_org_fllr_prob
            )
            anl_path_infos[anl_filler] += anl_path_prob
    return sorted(anl_path_infos.items(),
        key=lambda x: x[1],
        reverse=True
    )

def anls_by_gram(anl_list):
    anl_subst_dict = defaultdict(dict)
    for anl_item in anl_list:
        anl_subst, anl_tree, anl_score = anl_item[0][0], anl_item[0][1][0], anl_item[1]
        anl_subst_dict[anl_subst][anl_tree] = anl_score
    ordered_anls = sorted(anl_subst_dict.items(), key=lambda x: sum(x[1].values()), reverse=True)
    ordered_trees = []
    for anl_subst, anl_trees_dict in ordered_anls:
        ordered_trees.append((anl_subst, sorted(anl_trees_dict.items(), key=lambda x: x[1], reverse=True)))
    return ordered_trees

def anls_by_struct(anl_list):
    anl_subst_dict = defaultdict(dict)
    for anl_item in anl_list:
        #anl_subst, tree_struct, anl_score = anl_item[0][0], anl_item[0][1][1], anl_item[1]
        anl_subst, tree_struct, anl_score = anl_item[0][0], anl_item[0][1], anl_item[1]
        anl_subst_dict[tree_struct][anl_subst] = anl_score
    ordered_anls = sorted(anl_subst_dict.items(), key=lambda x: sum(x[1].values()), reverse=True)
    ordered_trees = []
    for tree_struct, anl_substs_dict in ordered_anls:
        ordered_trees.append((tree_struct, sorted(anl_substs_dict.items(), key=lambda x: x[1], reverse=True)))
    return ordered_trees

def anls_by_anl(anl_list):
    anls_dict = defaultdict(float)
    for anl_form, score in anl_list:
        anls_dict[anl_form[0]] += score
    return sorted(anls_dict.items(), key=lambda x: x[1], reverse=True)