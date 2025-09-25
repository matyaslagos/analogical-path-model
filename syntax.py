#!/usr/bin/env python3
from collections import defaultdict
from collections import Counter
from itertools import product
from string import punctuation
from pprint import pp
import math
import csv
import custom_io

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
                if (not only_completions) or (child in '<>'):
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
        """Return generator of shared fillers of sequence_1 and sequence_2 up to max_length.

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
        """Yield each shared filler of context_node_1 and context_node_2 up to max_length.
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
                    if (not only_completions) or (child in '<>'):
                        yield (tuple(new_path), freq_1, freq_2)
                yield from self._shared_neighbors_aux(child_node_1, child_node_2,
                                                      direction, max_length, min_length,
                                                      only_completions, new_path)

    def right_analogical_bases(self, sequence, max_length=float('inf')):
        """Return dict of analogical bases of sequence based on right contexts.
        """
        return self._analogical_bases(sequence, 'fw', max_length)

    def left_analogical_bases(self, sequence, max_length=float('inf')):
        """Return dict of analogical bases of sequence based on left contexts.
        """
        return self._analogical_bases(sequence, 'bw', max_length)

    def _analogical_bases(self, sequence, direction, max_length=float('inf')):
        """Return dict of analogical bases of target sequence given direction.

        Sequence s1 is an analogical base of target sequence s2 to degree r if
        we can substitute s1 by s2 in arbitrary contexts with certainty r.
        """
        target_freq = self.freq(sequence)
        base_dict = defaultdict(float)
        contexts = self._neighbors(sequence, direction, 2)
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
                base_dict[base] += min(context_target_prob, context_base_prob)
        return base_dict

def bilateral_analogical_bases(self, sequence, max_length=float('inf')):
    left_base_dict = self.left_analogical_bases(sequence, max_length)
    right_base_dict = self.right_analogical_bases(sequence, max_length)
    bilateral_bases = left_base_dict.keys() & right_base_dict.keys()
    bilateral_base_dict = {}
    for base in bilateral_bases:
        bilateral_base_dict[base] = min(left_base_dict[base], right_base_dict[base])
    return sorted(bilateral_base_dict.items(), key=lambda x: x[1], reverse=True)

def node_efficiency(sequence_node):
    if not sequence_node.children.values():
        return 1
    total_freq = sequence_node.freq
    distr = []
    for child_node in sequence_node.children.values():
        distr.append(child_node.freq / total_freq)
    entropy = sum(prob * math.log(1 / prob, 2) for prob in distr)
    if entropy == 0:
        return 0
    max_entropy = math.log(len(sequence_node.children), 2)
    return entropy / max_entropy

def prob_neighbors_aux(sequence_node, parent_freq, prob_mass=1, path=[]):
    node_freq = sequence_node.freq
    if path:
        efficiency = node_efficiency(sequence_node)
        prob = efficiency * prob_mass * (node_freq / parent_freq)
        yield (path, prob)
    else:
        efficiency = 0
    for child_token, child_node in sequence_node.children.items():
        new_path = path + [child_token]
        child_prob_mass = (1 - efficiency) * prob_mass * (node_freq / parent_freq)
        yield from prob_neighbors_aux(child_node, node_freq, child_prob_mass, new_path)

def weighted_condprob(model, context, sequence):
    context_node = model.sequence_node(context)
    rem_prob_mass = 1
    curr_prob_mass = 0
    parent_freq = context_node.freq
    for token in sequence:
        child_node = context_node.children[token]
        child_freq = child_node.freq
        efficiency = min(max(0.5, node_efficiency(child_node)), 0.9)
        curr_prob_mass = rem_prob_mass * efficiency * (child_freq / parent_freq)
        rem_prob_mass = rem_prob_mass * (1 - efficiency) * (child_freq / parent_freq)
        parent_freq = child_freq
        context_node = child_node
    return curr_prob_mass

def most_prob_neighbors(model, context):
    neighbors = []
    for neighbor, freq in model.right_neighbors(context):
        p = weighted_condprob(model, context, neighbor)
        neighbors.append((neighbor, p, freq))
    return sorted(neighbors, key=lambda x: x[1], reverse=True)

def bigram_anls(self, bigram):
    s1, s2 = tuple(bigram.split()[:1]), tuple(bigram.split()[1:])
    s1_bases = bilateral_analogical_bases(self, s1, len(s1))[:50]
    s2_bases = bilateral_analogical_bases(self, s2, len(s2))[:50]
    anls = {}
    for s1_anl, s1_score in s1_bases:
        for s2_anl, s2_score in s2_bases:
            if self.freq(s1_anl + s2_anl):
                anls[s1_anl + s2_anl] = (s1_score * s2_score)
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def mixed_anl_bases(model, targets, max_length):
    trie = weighted_trie(model, targets)
    left_bases = defaultdict(float)
    for context, context_target_freq in trie.left_neighbors(tuple()):
        context_freq = model.freq(context)
        for base, context_base_freq in model.right_neighbors(context, max_length=max_length):
            base_freq = model.freq(base)
            context_base_prob = context_freq / base_freq
            context_target_prob = context_target_freq / context_base_freq
            left_bases[base] += min(context_base_prob, context_target_prob)
    right_bases = defaultdict(float)
    for context, context_target_freq in trie.right_neighbors(tuple()):
        context_freq = model.freq(context)
        for base, context_base_freq in model.left_neighbors(context, max_length=max_length):
            base_freq = model.freq(base)
            context_base_prob = context_base_freq / base_freq
            context_target_prob = context_target_freq / context_freq
            right_bases[base] += min(context_base_prob, context_target_prob)
    anl_bases = {}
    for base in left_bases.keys() & right_bases.keys():
        anl_bases[base] = min(left_bases[base], right_bases[base])
    return sorted(anl_bases.items(), key=lambda x: x[1], reverse=True)

def weighted_trie(model, sequences):
    trie = FreqTrie()
    for sequence, weight in sequences:
        for left_context, freq in model.left_neighbors(sequence, only_completions=True):
            trie.bw_root._increment_or_make_branch(reversed(sequence), weight * freq)
        for right_context, freq in model.right_neighbors(sequence, only_completions=True):
            trie.fw_root._increment_or_make_branch(sequence, weight * freq)
    return trie

def mixed_paired_anls(model, prefixes, prefix_length, suffixes, suffix_length):
    prefix_bases = mixed_anl_bases(model, prefixes[:50], prefix_length)[:100]
    suffix_bases = mixed_anl_bases(model, suffixes[:50], suffix_length)[:100]
    combined_bases = product(prefix_bases, suffix_bases)
    anls = {}
    for prefix_info, suffix_info in combined_bases:
        anl_prefix, prefix_score = prefix_info
        anl_suffix, suffix_score = suffix_info
        if model.freq(anl_prefix + anl_suffix):
            anls[anl_prefix + anl_suffix] = min(prefix_score, suffix_score)
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def paired_anls2(model, prefixes, prefix_length, suffixes, suffix_length):
    prefix_bases = defaultdict(float)
    for prefix, score in prefixes[:50]:
        for anl_base, anl_score in bilateral_analogical_bases(model, prefix):
            prefix_bases[anl_base] += score * anl_score
    suffix_bases = defaultdict(float)
    for suffix, score in suffixes[:50]:
        for anl_base, anl_score in bilateral_analogical_bases(model, suffix):
            suffix_bases[anl_base] += score * anl_score
    best_prefix_bases = custom_io.sorted_items(prefix_bases)[:50]
    best_suffix_bases = custom_io.sorted_items(suffix_bases)[:50]
    combined_bases = product(best_prefix_bases, best_suffix_bases)
    anls = {}
    for prefix_info, suffix_info in combined_bases:
        anl_prefix, prefix_score = prefix_info
        anl_suffix, suffix_score = suffix_info
        freq = model.freq(anl_prefix + anl_suffix)
        if freq:
            anls[anl_prefix + anl_suffix] = min(prefix_score, suffix_score) * math.sqrt(freq)
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def bigram_to_unigrams(self, bigram):
    comp_bigrams = bigram_anls(self, bigram)[:20] # :200
    bw_weighted_contexts = defaultdict(float)
    fw_weighted_contexts = defaultdict(float)
    for anl_bigram, weight in comp_bigrams:
        for bw_context, freq in self.left_neighbors(anl_bigram):
            bw_weighted_contexts[bw_context] += freq * weight
        for fw_context, freq in self.right_neighbors(anl_bigram):
            fw_weighted_contexts[fw_context] += freq * weight
    bw_target_freq = sum(bw_weighted_contexts.values())
    bw_anl_contexts = defaultdict(lambda: defaultdict(float))
    bw_anls = defaultdict(float)
    for bw_context, context_target_freq in bw_weighted_contexts.items():
        context_freq = self.freq(bw_context)
        for source, context_source_freq in self.right_neighbors(bw_context, max_length=1):
            source_freq = self.freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            score = some_mean(source_to_context_prob, context_to_target_prob)
            bw_anls[source] += score
            bw_anl_contexts[source][bw_context] += score
    fw_target_freq = sum(fw_weighted_contexts.values())
    fw_anl_contexts = defaultdict(lambda: defaultdict(float))
    fw_anls = defaultdict(float)
    for fw_context, context_target_freq in fw_weighted_contexts.items():
        context_freq = self.freq(fw_context)
        for source, context_source_freq in self.left_neighbors(fw_context, max_length=1):
            source_freq = self.freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            score = some_mean(source_to_context_prob, context_to_target_prob)
            fw_anls[source] += score
            fw_anl_contexts[source][fw_context] += score
    anls = {}
    context_dict = {}
    for anl in bw_anls.keys() & fw_anls.keys():
        anls[anl] = min(bw_anls[anl], fw_anls[anl])
        left_contexts = sorted(bw_anl_contexts[anl].items(), key=lambda x: x[1], reverse=True)[:10]
        right_contexts = sorted(fw_anl_contexts[anl].items(), key=lambda x: x[1], reverse=True)[:10]
        context_dict[anl] = {'left': left_contexts, 'right': right_contexts}
    return (sorted(anls.items(), key=lambda x: x[1], reverse=True), context_dict)

def some_mean(n, m):
    return math.sqrt(n * m)

def anl_bases(self, target_sequences):
    anls = defaultdict(float)
    for target_sequence, weight in target_sequences:
        target_freq = self.freq(target_sequence)
        left_anls = defaultdict(float)
        left_contexts = self.left_neighbors(target_sequence)
        for context, context_target_freq in left_contexts:
            context_freq = self.freq(context) * weight
            bases = self.right_neighbors(context, max_length=len(target_sequence))
            for base, context_base_freq in bases:
                base_freq = self.freq(base)
                context_base_prob = context_base_freq / base_freq
                context_target_prob = context_target_freq / context_freq
                left_anls[base] += some_mean(context_base_prob, context_target_prob)
        right_anls = defaultdict(float)
        right_contexts = self.right_neighbors(target_sequence)
        for context, context_target_freq in right_contexts:
            context_freq = self.freq(context) * weight
            bases = self.left_neighbors(context, max_length=len(target_sequence))
            for base, context_base_freq in bases:
                base_freq = self.freq(base)
                context_base_prob = context_base_freq / base_freq
                context_target_prob = context_target_freq / context_freq
                right_anls[base] += some_mean(context_base_prob, context_target_prob)
        for base in left_anls.keys() & right_anls.keys():
            anls[base] += min(left_anls[base], right_anls[base])
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def rec_anls(model, sequence, lookup_dict=None, return_lookup_dict=False):
    print('Checking sequence:', ' '.join(sequence))
    if lookup_dict is None:
        lookup_dict = {}
    if sequence in lookup_dict:
        print('<- dynamic lookup ->')
        return lookup_dict[sequence]
    elif len(sequence) == 1:
        anls = [(sequence, 1)]#bilateral_analogical_bases(model, sequence, len(sequence))[:10]
        lookup_dict[sequence] = anls
        return anls
    else:
        print(sequence)
        binary_splits = ((sequence[:i], sequence[i:]) for i in range(1, len(sequence)))
        anl_base_dict = defaultdict(float)
        for prefix, suffix in binary_splits:
            print('Checking split:', (' '.join(prefix),  ' '.join(suffix)))
            # Recursive calls
            rec_prefixes = rec_anls(model, prefix, lookup_dict)
            rec_suffixes = rec_anls(model, suffix, lookup_dict)
            # Find analogical sequences
            anl_sequences = paired_anls(model, prefix, suffix, rec_prefixes, rec_suffixes)
            for anl_sequence, score in anl_sequences:
                anl_base_dict[anl_sequence] += score
        sorted_anls = sorted(anl_base_dict.items(), key=lambda x: x[1], reverse=True)
        lookup_dict[sequence] = sorted_anls
        if return_lookup_dict:
            return lookup_dict
        else:
            return sorted_anls

def paired_anls(model, prefix, suffix, rec_prefixes, rec_suffixes):
    prefix_anls = defaultdict(float)
    for rec_prefix, weight in rec_prefixes[:10]:
        anls = bilateral_analogical_bases(model, rec_prefix, len(prefix))
        for anl, score in anls:
            prefix_anls[anl] += score * weight
    sorted_prefix_anls = sorted(prefix_anls.items(), key=lambda x: x[1], reverse=True)[:50]
    suffix_anls = defaultdict(float)
    for rec_suffix, weight in rec_suffixes[:10]:
        anls = bilateral_analogical_bases(model, rec_suffix, len(suffix))
        for anl, score in anls:
            suffix_anls[anl] += score * weight
    sorted_suffix_anls = sorted(suffix_anls.items(), key=lambda x: x[1], reverse=True)[:50]
    anls = defaultdict(float)
    for prefix_info, suffix_info in product(sorted_prefix_anls, sorted_suffix_anls):
        prefix_anl, prefix_score = prefix_info
        suffix_anl, suffix_score = suffix_info
        if model.freq(prefix_anl + suffix_anl):
            score = (prefix_score * suffix_score)
            anls[prefix_anl + suffix_anl] += score
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def mixed_trie(model, sequences):
    trie = FreqTrie()
    for sequence, weight in sequences:
        for neighbor, freq in model.left_neighbors(sequence, only_completions=True):
            trie.bw_root._increment_or_make_branch(reversed(neighbor), weight * freq)
        for neighbor, freq in model.right_neighbors(sequence, only_completions=True):
            trie.fw_root._increment_or_make_branch(neighbor, weight * freq)
    return trie

def anl_subst(model, base_sequence, base_freq, sequence_mix_trie):
    fw_seq_node = model.sequence_node(base_sequence, 'fw')
    bw_seq_node = model.sequence_node(base_sequence, 'bw')
    fw_seq_mix_node = sequence_mix_trie.fw_root
    bw_seq_mix_node = sequence_mix_trie.bw_root
    fw_subst_score = 0
    shared_right_neighbors = model._shared_neighbors_aux(fw_seq_node, fw_seq_mix_node, 'fw',
                                                         float('inf'), 0, False, [])
    for context, context_base_freq, context_mix_seq_freq in shared_right_neighbors:
        context_freq = model.freq(context)
        context_base_prob = context_base_freq / base_freq
        context_mix_seq_prob = context_mix_seq_freq / context_freq
        score = some_mean(context_base_prob, context_mix_seq_prob)
        fw_subst_score += score
    bw_subst_score = 0
    shared_left_neighbors = model._shared_neighbors_aux(bw_seq_node, bw_seq_mix_node, 'bw',
                                                        float('inf'), 0, False, [])
    for context, context_base_freq, context_mix_seq_freq in shared_left_neighbors:
        context_freq = model.freq(context)
        context_base_prob = context_base_freq / base_freq
        context_mix_seq_prob = context_mix_seq_freq / context_freq
        score = some_mean(context_base_prob, context_mix_seq_prob)
        bw_subst_score += score
    return min(fw_subst_score, bw_subst_score)


# ---------- #
# Morphology #
# ---------- #

#> Setup <#

def csv_to_wordfreqdict(filename):
    """Import filename as a dict of words (tuples of characters) with int values.
    """
    with open(filename, newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        clean_word = lambda s: ('<',) + tuple(s.strip(punctuation).lower()) + ('>',)
        return {clean_word(row['key']): int(row['value']) for row in reader}

# Find anl_prefs for pref by examining their right distributions
def outside_morph_anls(self, pref, suff):
    anl_dict = defaultdict(float)
    # If suffix is word-ending, we pretend we haven't seen it
    is_unseen = lambda x: False
    """
    is_unseen = (
        (lambda x: x[:len(suff) - 1] == tuple(suff)[:-1]) if '>' in suff
        else (lambda x: False)
    )
    """
    pref_freq = self.freq(pref)
    anl_prefs = self.left_neighbors(suff, only_completions=True)
    for anl_pref, anl_pref_suff_freq in anl_prefs:
        if anl_pref == pref:
            continue
        anl_pref_cell = self.tags(anl_pref + ('>',))
        anl_word_cell = self.tags(anl_pref + tuple(suff.strip('>')) + ('>',))
        if anl_pref_cell and not anl_word_cell:
            continue
        anl_pref_freq = self.freq(anl_pref)
        shared_contexts = self.shared_right_neighbors(pref, anl_pref, min_length=3)
        for context, context_pref_freq, context_anl_pref_freq in shared_contexts:
            if is_unseen(context):
                continue
            context_freq = self.freq(context)
            # Calculate probs of pref--context and anl_pref--context paths
            anl_pref_prob = context_anl_pref_freq / anl_pref_freq
            pref_prob = context_pref_freq / pref_freq
            score = min(anl_pref_prob, pref_prob)
            key = (anl_pref, anl_pref_cell, anl_word_cell)
            anl_dict[key] += score
    return custom_io.dict_to_list(anl_dict)

def rec_morph_anls(self, word, lookup_dict=None, return_lookup_dict=False):
    if lookup_dict is None:
        lookup_dict = {}
    if word == '<>':
        return (frozenset(), [(('<',), 1)])
    elif word in lookup_dict:
        return lookup_dict[word]['info']
    else:
        word = custom_io.hun_encode(word)
        pref_suff_pairs = ((word[:i], word[i:]) for i in range(1, len(word.strip('>'))))
        anl_words = defaultdict(float)
        tag_scores = defaultdict(float)
        split_analysis = {}
        for pref, suff in pref_suff_pairs:
            split_dict = defaultdict(lambda: defaultdict(float))
            triple_dict = defaultdict(lambda: defaultdict(float))
            # Recursive call
            pref_cell, anl_prefs = rec_morph_anls(self, pref + '>', lookup_dict)
            anl_prefs = anl_prefs.copy()[:20]# + [(pref, 1)]
            # Find outside analogies
            # TODO: integrate into recursive analogy finding below
            outside_anl_bases = outside_morph_anls(self, pref, suff)[:10]
            for anl_base, score in outside_anl_bases:
                anl_pref, anl_pref_cell, anl_word_cell = anl_base
                anl_triple = (anl_pref_cell, anl_word_cell, pref_cell)
                anl_word = anl_pref + tuple(suff)
                triple_dict[anl_triple][anl_word] += score
                anl_words[anl_word] += score
            # Find recursive (inside-indirect) analogies
            for anl_pref, anl_pref_score in anl_prefs:
                inside_anl_bases = outside_morph_anls(self, anl_pref, suff)[:10]
                for anl_base, score in inside_anl_bases:
                    anl_pref, anl_pref_cell, anl_word_cell = anl_base
                    anl_triple = (anl_pref_cell, anl_word_cell, pref_cell)
                    anl_word = anl_pref + tuple(suff)
                    triple_dict[anl_triple][anl_word] += math.sqrt(score * anl_pref_score)
                    anl_words[anl_word] += math.sqrt(score * anl_pref_score)
            tag_dict = defaultdict(dict)
            for anl_triple, triple_words in triple_dict.items():
                tag = tulip(*anl_triple)
                a, b, c = anl_triple
                tupled_anl_triple = (('anl_pref', tuple(a)),
                                     ('anl_word', tuple(b)),
                                     ('pref', tuple(c)))
                tag_dict[tag][tupled_anl_triple] = triple_words
                tag_scores[tag] += sum(triple_words.values())
            split_analysis[(pref, suff)] = tag_dict
        best_tag = (sorted(tag_scores.keys(), key=tag_scores.get, reverse=True) + [frozenset()])[0]
        anl_word_list = custom_io.dict_to_list(anl_words)
        lookup_dict[word] = {'info': (best_tag, anl_word_list),
                             'analysis': split_analysis}
        if not return_lookup_dict:
            return (best_tag, anl_word_list)
        else:
            return lookup_dict

def tulip(a, b, c):
    """Return the "tulip" of sets a, b, and c.

    The "tulip" of sets a, b, and c is the set (b - a) | (c - a) | (a & b & c),
    a solution to the four-part analogy problem [a : b :: c : ?].

    Arguments:
        - a, b, c (sets)

    Returns:
        - (b - a) | (c - a) | (a & b & c)
    """
    return (b - a) | (c - a) | (a & b & c)

def analyser(analysis_dict):
    split_scores = {}
    split_best_tags = {}
    for split, tag_dict in analysis_dict.items():
        tag_scores = defaultdict(float)
        tag_best_anl_triples = {}
        for tag, triple_dict in tag_dict.items():
            anl_triple_scores = defaultdict(float)
            anl_triple_best_words = {}
            for anl_triple, word_scores in triple_dict.items():
                anl_triple_scores[anl_triple] = sum(word_scores.values())
                best_words = list(map(lambda x: custom_io.hun_decode(''.join(x)),
                                      sorted(word_scores.keys(),
                                             key=word_scores.get,
                                             reverse=True)[:5]))
                anl_triple_best_words[anl_triple] = best_words
            tag_scores[tuple(tag)] = sum(anl_triple_scores.values())
            best_anl_triples = sorted(anl_triple_best_words.items(),
                                      key=lambda x: anl_triple_scores[x[0]],
                                      reverse=True)[:5]
            tag_best_anl_triples[tuple(tag)] = best_anl_triples
        split_scores[split] = sum(tag_scores.values())
        best_tags = sorted(tag_best_anl_triples.items(),
                           key=lambda x: tag_scores[x[0]],
                           reverse=True)[:5]
        split_best_tags[split] = best_tags
    best_splits = sorted(split_best_tags.items(),
                         key=lambda x: split_scores[x[0]],
                         reverse=True)
    return best_splits

def naive_tag_finder(self, word):
    current_node = self.bw_root
    suffix = ''
    for token in reversed(word):
        if token in current_node.children:
            suffix = token + suffix
            current_node = current_node.children[token]
        else:
            break
    print(suffix)
    if suffix == word:
        return self.tags(word)
    tags = Counter()
    for prefix, freq in self.left_neighbors(suffix, only_completions=True):
        tags[self.tags(''.join(prefix) + suffix)] += freq
    return sorted(tags.keys(), key=tags.get, reverse=True)[0]
    