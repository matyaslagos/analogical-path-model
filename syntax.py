#!/usr/bin/env python3
from collections import defaultdict, Counter
from itertools import product
from operator import itemgetter
from heapq import nlargest
import math

#-----------------#
# Setup functions #
#-----------------#

# Import some text as corpus
def txt_to_list(filename):
    """Import a txt list of sentences as a list of tuples of words.

    Argument:
        filename (string): e.g. 'corpus.txt', the name of a txt file with one sentence
        per line

    Returns:
        list of tuples of strings: each sentence is an endmarked tuple of strings,
        e.g. ('<', 'this', 'is', 'good', '>')
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return [('<',) + tuple(line.strip().split()) + ('>',) for line in lines]

# Make frequency trie out of corpus
def freqtrie_setup(corpus):
    """Make a frequency trie data structure from corpus.

    Argument:
        corpus (list of iterables, or dict of iterables and their frequencies)

    Returns:
        freq_trie: trie data structure of corpus frequencies
    """
    freq_trie = FreqTrie()
    # If corpus is a dict of sequences and their frequencies
    if isinstance(corpus, dict):
        for sequence_data, frequency in corpus.items():
            # If keys include paradigm cell tags
            if len(sequence_data) == 2:
                sequence, tags = sequence_data
                freq_trie._insert(sequence, frequency, tags)
            # Elif keys are just sequences
            else:
                freq_trie._insert(sequence_data, frequency)
    # Elif corpus is just an iterator of sequences
    else:
        for sequence in corpus:
            freq_trie._insert(sequence)
    return freq_trie

#-----------------------------------------#
# FreqNode and FreqTrie class definitions #
#-----------------------------------------#

class FreqNode:
    def __init__(self):
        self.children = {}
        self.freq = 0
        self.tags = None

    def _increment_or_make_branch(self, sequence, freq=1):
        """Increment the frequency of token_tuple or make a new branch for it.
        """
        current_node = self
        for token in sequence:
            current_node = current_node._get_or_make_child(token)
            current_node.freq += freq

    def _get_or_make_child(self, token):
        """Return the child called token or make a new child called token.
        """
        if token not in self.children:
            self.children[token] = FreqNode()
        return self.children[token]

class FreqTrie:
    def __init__(self):
        self.fw_root = FreqNode()
        self.bw_root = FreqNode()

    def _insert(self, sequence, freq=1, tags=None):
        """Record distributions of prefixes and suffixes of sequence.

        Arguments:
            - sequence (iterable of strings): e.g. ['<', 'this', 'is', 'good', '>']
            - freq (int): how many occurrences of sequence should be recorded

        Effect:
            - For each prefix--suffix split of sequence, record the occurrences of
              prefix and suffix. (Prefix is reversed to make shared-neighbor search more
              efficient.)
        """
        # Add token frequency mass of sequence to root nodes (to record corpus size)
        token_freq_mass = len(sequence) * freq
        self.fw_root.freq += token_freq_mass
        self.bw_root.freq += token_freq_mass
        # Record each suffix in fw trie and each reversed prefix in bw trie
        prefix_suffix_pairs = (
            (sequence[:i], sequence[i:])
            for i in range(len(sequence) + 1)
        )
        for prefix, suffix in prefix_suffix_pairs:
            self.fw_root._increment_or_make_branch(suffix, freq)
            self.bw_root._increment_or_make_branch(reversed(prefix), freq)
        # Record cell features for full sequence
        if tags is not None:
            self.sequence_node(sequence).tags = tags

    def sequence_node(self, sequence, direction='fw'):
        """Return the node that represents sequence.

        Argument:
            sequence (tuple of strings): of the form ('this', 'is', '_') or
            ('_', 'is', 'good'), with '_' indicating the empty slot. If no
            slot is indicated, defaults to ('this', 'is', '_')

        Returns:
            FreqNode representing sequence.
        """
        # If left context, look up token sequence in forward trie
        if direction == 'fw':
            current_node = self.fw_root
        # If right context, look up reversed token sequence in backward trie
        else:
            current_node = self.bw_root
            sequence = reversed(sequence)
        # General lookup
        for token in sequence:
            try:
                current_node = current_node.children[token]
            except KeyError:
                return None
        return current_node

    def freq(self, sequence=''):
        """Return the frequency of sequence.
        """
        seq_node = self.sequence_node(sequence)
        return seq_node.freq if seq_node else 0

    def tags(self, sequence):
        seq_node = self.sequence_node(sequence)
        return seq_node.tags if seq_node else frozenset()

    def right_neighbors(self, sequence, max_length=float('inf'),
                        min_length=0, only_completions=False):
        return self._neighbors(sequence, 'fw',
                               max_length, min_length, only_completions)

    def left_neighbors(self, sequence, max_length=float('inf'),
                       min_length=0, only_completions=False):
        return self._neighbors(sequence, 'bw',
                               max_length, min_length, only_completions)

    def _neighbors(self, sequence, direction='fw',
                   max_length=float('inf'), min_length=0, only_completions=False):
        """Return generator of each neighbor of sequence with their joint frequency.
        """
        seq_node = self.sequence_node(sequence, direction)
        if not seq_node:
            return iter(())
        return self._neighbors_aux(seq_node, direction, max_length, min_length,
                                   only_completions, path=[])

    def _neighbors_aux(self, seq_node, direction,
                       max_length, min_length, only_completions, path):
        """Yield each neighbor of sequence with their joint frequency.
        """
        if len(path) >= max_length:
            return
        for child in seq_node.children:
            new_path = path + [child] if direction == 'fw' else [child] + path
            child_node = seq_node.children[child]
            freq = child_node.freq
            if len(new_path) >= min_length:
                if (not only_completions) or (child in {'<','>'}):
                    yield (tuple(new_path), freq)
            yield from self._neighbors_aux(child_node, direction,
                                           max_length, min_length,
                                           only_completions, new_path)

    def shared_right_neighbors(self, sequence_1, sequence_2, max_length=float('inf'),
                               min_length=0, only_completions=False):
        return self._shared_neighbors(sequence_1, sequence_2, 'fw',
                                      max_length, min_length, only_completions)

    def shared_left_neighbors(self, sequence_1, sequence_2, max_length=float('inf'),
                              min_length=0, only_completions=False):
        return self._shared_neighbors(sequence_1, sequence_2, 'bw',
                                      max_length, min_length, only_completions)

    def _shared_neighbors(self, sequence_1, sequence_2, direction='fw',
                          max_length=float('inf'), min_length=0, only_completions=False):
        """Return generator of shared neighbors of sequence_1 and sequence_2.

        Arguments:
            sequence_1 (tuple of strings): e.g. ('_', 'is', 'good')
            sequence_2 (tuple of strings): e.g. ('_', 'was', 'here')

        Returns:
            generator of (filler, freq_1, freq_2) tuples:
                if e.g. the tuple (('this', '_'), 23, 10) is yielded, then:
                - 'this' occurred before 'is good' 23 times, and
                - 'this' occurred before 'was here' 10 times.
        """
        seq_node_1 = self.sequence_node(sequence_1, direction)
        seq_node_2 = self.sequence_node(sequence_2, direction)
        if not seq_node_1 or not seq_node_2:
            return iter(())
        return self._shared_neighbors_aux(seq_node_1, seq_node_2, direction,
                                          max_length, min_length, only_completions, path=[])

    def _shared_neighbors_aux(self, seq_node_1, seq_node_2, direction,
                              max_length, min_length, only_completions, path):
        """Yield shared neighbors of seq_node_1 and seq_node_2.
        """
        if len(path) >= max_length:
            return
        for child in seq_node_1.children:
            if child in seq_node_2.children:
                new_path = path + [child] if direction == 'fw' else [child] + path
                child_node_1 = seq_node_1.children[child]
                child_node_2 = seq_node_2.children[child]
                freq_1 = child_node_1.freq
                freq_2 = child_node_2.freq
                if len(new_path) >= min_length:
                    if (not only_completions) or (child in {'<','>'}):
                        yield (tuple(new_path), freq_1, freq_2)
                yield from self._shared_neighbors_aux(child_node_1, child_node_2,
                                                      direction, max_length, min_length,
                                                      only_completions, new_path)

    def right_analogies(self, sequence, max_length=float('inf')):
        """Return dict of analogical bases of sequence based on right contexts.
        """
        return self._analogies(sequence, 'fw', max_length)

    def left_analogies(self, sequence, max_length=float('inf')):
        """Return dict of analogical bases of sequence based on left contexts.
        """
        return self._analogies(sequence, 'bw', max_length)

    def _analogies(self, sequence, direction, max_length=float('inf')):
        """Return analogies of target sequence in the given direction.

        A source is an analogy of a target with score s if source can be
        substituted by target with degree of certainty s.
        """
        target_freq = self.freq(sequence)
        base_dict = defaultdict(float)
        contexts = self._neighbors(sequence, direction)
        for context, context_target_freq in contexts:
            context_freq = self.freq(context)
            # Probability of going from context to target
            context_target_prob = context_target_freq / context_freq
            other_direction = 'bw' if direction == 'fw' else 'fw'
            bases = self._neighbors(context, other_direction, max_length)
            for base, context_base_freq in bases:
                base_freq = self.freq(base)
                # Probability of going from source to context
                context_base_prob = context_base_freq / base_freq
                base_dict[base] += combine_path_scores(context_target_prob, context_base_prob)
        return base_dict

#-----------------------------#
# Analogical parser functions #
#-----------------------------#

def combine_path_scores(source_to_context_prob, context_to_target_prob):
    """Combine the conditional probabilities an analogical path.
    """
    return min(source_to_context_prob, context_to_target_prob)

def combine_side_scores(left_subst_score, right_subst_score):
    """Combine the left- and right-substitutabilities of a phrase by another phrase.
    """
    return min(left_subst_score, right_subst_score)

def combine_split_scores(prefix_subst_score, suffix_subst_score):
    """Combine the substitutabilities of phrases p1, p2 in a split (p1, p2).
    
    Suppose we are analysing the split ('my cat', 'was chasing a mouse'). Then if
        - prefix_subst_score is the degree to which 'the dog' is substitutable
          by 'my cat', and
        - suffix_subst_score is the degree to which 'is sleeping' is substitutable
          by 'was chasing a mouse',
    then this function returns the degree to which ('the dog', 'is sleeping') is
    substitutable by ('my cat', 'was chasing a mouse').
    """
    return min(prefix_subst_score, suffix_subst_score)

def bilateral_analogies(model, sequence):
    left_anl_scores = model.left_analogies(sequence, len(sequence))
    right_anl_scores = model.right_analogies(sequence, len(sequence))
    bilateral_anls = left_anl_scores.keys() & right_anl_scores.keys()
    bilateral_anl_scores = {}
    for anl in bilateral_anls:
        bilateral_anl_scores[anl] = min(left_anl_scores[anl], right_anl_scores[anl])
    return nlargest(50, bilateral_anl_scores.items(), key=itemgetter(1))

def bigram_analogies(model, bigram):
    s1, s2 = ((word,) for word in bigram)
    s1_anls = bilateral_analogies(model, s1)
    s2_anls = bilateral_analogies(model, s2)
    anls = {}
    for s1_anl, s1_score in s1_anls:
        for s2_anl, s2_score in s2_anls:
            n = model.freq(s1_anl + s2_anl)
            if n:
                anls[s1_anl + s2_anl] = (s1_score * s2_score) * (n ** (1 / 4))
    return nlargest(50, anls.items(), key=itemgetter(1))

def bigram_to_unigrams(model, bigram):
    bigrams_to_mix = bigram_analogies(model, bigram)
    return mix_and_reduce(model, bigrams_to_mix, 1)

def mix_and_reduce(model, xgrams, analogy_length):
    bw_weighted_contexts = defaultdict(float)
    fw_weighted_contexts = defaultdict(float)
    for anl_bigram, weight in xgrams:
        for bw_context, freq in model.left_neighbors(anl_bigram):
            bw_weighted_contexts[bw_context] += freq
        for fw_context, freq in model.right_neighbors(anl_bigram):
            fw_weighted_contexts[fw_context] += freq
    bw_target_freq = sum(bw_weighted_contexts.values())
    bw_anl_contexts = defaultdict(lambda: defaultdict(float))
    bw_anls = defaultdict(float)
    for bw_context, context_target_freq in bw_weighted_contexts.items():
        context_freq = model.freq(bw_context)
        potential_analogies = model.right_neighbors(bw_context, max_length=analogy_length)
        for source, context_source_freq in potential_analogies:
            source_freq = model.freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            score = min(source_to_context_prob, context_to_target_prob)
            bw_anls[source] += score
            bw_anl_contexts[source][bw_context] += score
    fw_target_freq = sum(fw_weighted_contexts.values())
    fw_anl_contexts = defaultdict(lambda: defaultdict(float))
    fw_anls = defaultdict(float)
    for fw_context, context_target_freq in fw_weighted_contexts.items():
        context_freq = model.freq(fw_context)
        potential_analogies = model.left_neighbors(fw_context, max_length=analogy_length)
        for source, context_source_freq in potential_analogies:
            source_freq = model.freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            score = min(source_to_context_prob, context_to_target_prob)
            fw_anls[source] += score
            fw_anl_contexts[source][fw_context] += score
    anls = {}
    context_dict = {}
    for anl in bw_anls.keys() & fw_anls.keys():
        anls[anl] = min(bw_anls[anl], fw_anls[anl])
        left_contexts = nlargest(10, bw_anl_contexts[anl].items(), key=itemgetter(1))
        right_contexts = nlargest(10, fw_anl_contexts[anl].items(), key=itemgetter(1))
        context_dict[anl] = {'left': left_contexts, 'right': right_contexts}
    return nlargest(50, anls.items(), key=itemgetter(1))

def mix_and_reduce_simple(model, sequences, analogy_length):
    # Count freq of mix in each context (sum of freqs of sequences in this context)
    mixed_left_contexts = defaultdict(float)
    mixed_right_contexts = defaultdict(float)
    for sequence, _ in sequences:
        for left_context, freq in model.left_neighbors(sequence):
            mixed_left_contexts[left_context] += freq
        for right_context, freq in model.right_neighbors(sequence):
            mixed_right_contexts[right_context] += freq
    # Compute left analogies for mix
    left_anl_contexts = defaultdict(lambda: defaultdict(float))
    left_anls = defaultdict(float)
    for left_context, context_target_freq in left_contexts.items():
        context_freq = model.freq(left_context)
        sources = model.right_neighbors(left_context, max_length=analogy_length)
        for source, context_source_freq in sources:
            source_freq = model.freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            score = combine_path_scores(source_to_context_prob, context_to_target_prob)
            left_anls[source] += score
            left_anl_contexts[source][left_context] += score
    # Compute right analogies for mix
    right_anl_contexts = defaultdict(lambda: defaultdict(float))
    right_anls = defaultdict(float)
    for right_context, context_target_freq in right_contexts.items():
        context_freq = model.freq(right_context)
        sources = model.left_neighbors(right_context, max_length=analogy_length)
        for source, context_source_freq in sources:
            source_freq = model.freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            score = combine_path_scores(source_to_context_prob, context_to_target_prob)
            right_anls[source] += score
            right_anl_contexts[source][right_context] += score
    anls = {}
    context_dict = {}
    for anl in left_anls.keys() & right_anls.keys():
        anls[anl] = combine_side_scores(left_anls[anl], right_anls[anl])
        best_left_contexts = nlargest(10, left_anl_contexts[anl].items(), key=itemgetter(1))
        best_right_contexts = nlargest(10, right_anl_contexts[anl].items(), key=itemgetter(1))
        context_dict[anl] = {'left': best_left_contexts, 'right': best_right_contexts}
    return nlargest(50, anls.items(), key=itemgetter(1))
    

def recursive_analogies(model, sequence, lookup_dict=None, return_lookup_dict=False):
    # Initialise dynamic lookup dict
    if lookup_dict is None:
        lookup_dict = {}
    # Dynamic lookup
    if sequence in lookup_dict:
        return lookup_dict[sequence]
    # Base case
    elif len(sequence) == 1:
        anls = [(sequence, 1)]
        lookup_dict[sequence] = anls
        return anls
    # Recursive case
    else:
        binary_splits = ((sequence[:i], sequence[i:]) for i in range(1, len(sequence)))
        anl_seq_scores = defaultdict(float)
        for prefix, suffix in binary_splits:
            # Recursive calls on constituents
            rec_prefixes = recursive_analogies(model, prefix, lookup_dict)
            rec_suffixes = recursive_analogies(model, suffix, lookup_dict)
            # Find analogical sequences for prefix and suffix analogies
            anl_sequences = pair_analogies(model, prefix, suffix, rec_prefixes, rec_suffixes)
            for anl_sequence, score in anl_sequences:
                anl_seq_scores[anl_sequence] += score
        best_anls = nlargest(50, anl_seq_scores.items(), key=itemgetter(1))
        lookup_dict[sequence] = best_anls
        if return_lookup_dict:
            return lookup_dict
        else:
            return best_anls

def pair_analogies(model, prefix, suffix, rec_prefixes, rec_suffixes):
    prefix_anls = defaultdict(float)
    for rec_prefix, weight in rec_prefixes[:10]:
        anls = bilateral_analogies(model, rec_prefix)
        for anl, score in anls:
            prefix_anls[anl] += score * weight
    sorted_prefix_anls = nlargest(50, prefix_anls.items(), key=itemgetter(1))
    suffix_anls = defaultdict(float)
    for rec_suffix, weight in rec_suffixes[:10]:
        anls = bilateral_analogies(model, rec_suffix)
        for anl, score in anls:
            suffix_anls[anl] += score * weight
    sorted_suffix_anls = nlargest(50, suffix_anls.items(), key=itemgetter(1))
    anls = defaultdict(float)
    for prefix_info, suffix_info in product(sorted_prefix_anls, sorted_suffix_anls):
        prefix_anl, prefix_score = prefix_info
        suffix_anl, suffix_score = suffix_info
        if model.freq(prefix_anl + suffix_anl):
            score = (prefix_score * suffix_score)
            anls[prefix_anl + suffix_anl] += score
    return nlargest(50, anls.items(), key=itemgetter(1))