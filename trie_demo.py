#!/usr/bin/env python3
from collections import defaultdict
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
        """Return analogical bases of sequence based on right contexts.
        """
        return self._analogical_bases(sequence, 'fw', max_length)

    def left_analogical_bases(self, sequence, max_length=float('inf')):
        """Return analogical bases of sequence based on left contexts.
        """
        return self._analogical_bases(sequence, 'bw', max_length)
    
    def _analogical_bases(self, sequence, direction, max_length=float('inf')):
        """Return dict of analogical bases of target sequence.

        Sequence s1 is an analogical base of target sequence s2 to degree r if
        we can substitute s1 by s2 in arbitrary contexts with certainty r.
        """
        target_freq = self.freq(sequence)
        base_dict = defaultdict(float)
        contexts = self.neighbors(sequence, direction)
        for context, context_target_freq in contexts:
            context_freq = self.freq(context)
            # Probability of going from context to target
            context_target_prob = context_target_freq / context_freq
            other_direction = 'bw' if direction == 'fw' else 'fw'
            bases = self.neighbors(context, other_direction, max_length)
            for base, context_base_freq in bases:
                base_freq = self.freq(base)
                # Probability of going from source to context
                context_base_prob = context_base_freq / base_freq
                base_dict[base] += min(context_target_prob, context_base_prob)
        return base_dict

def bilateral_analogical_bases(self, sequence):
    left_base_dict = self.left_analogical_bases(sequence, len(sequence))
    right_base_dict = self.right_analogical_bases(sequence, len(sequence))
    bilateral_bases = left_base_dict.keys() & right_base_dict.keys()
    bilateral_base_dict = {}
    for base in bilateral_bases:
        bilateral_base_dict[base] = min(left_base_dict[base], right_base_dict[base])
    return sorted(bilateral_base_dict.items(), key=lambda x: x[1], reverse=True)

def bigram_anls(self, bigram):
    s1, s2 = tuple(bigram.split()[:1]), tuple(bigram.split()[1:])
    s1_bases = bilateral_analogical_bases(self, s1)[:50]
    s2_bases = bilateral_analogical_bases(self, s2)[:50]
    anls = {}
    for s1_anl, s1_score in s1_anls:
        for s2_anl, s2_score in s2_anls:
            if self.freq(s1_anl + s2_anl):
                anls[s1_anl + s2_anl] = min(s1_score, s2_score)
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def bigram_to_unigrams(self, bigram):
    comp_bigrams = bigram_anls(self, bigram)[:200]
    left, right = tuple(bigram.split()[:1]), tuple(bigram.split()[1:])
    bw_weighted_contexts = defaultdict(float)
    fw_weighted_contexts = defaultdict(float)
    for anl_bigram, weight in comp_bigrams:
        for bw_context, freq in self.get_fillers(('_',) + anl_bigram):
            bw_weighted_contexts[bw_context] += freq * weight
        for fw_context, freq in self.get_fillers(anl_bigram + ('_',)):
            fw_weighted_contexts[fw_context] += freq * weight
    bw_target_freq = sum(bw_weighted_contexts.values())
    bw_anls = defaultdict(float)
    for bw_context, context_target_freq in bw_weighted_contexts.items():
        context_freq = self.get_freq(bw_context)
        for source, context_source_freq in self.get_fillers(bw_context, max_length=2):
            source_freq = self.get_freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            bw_anls[source[1:]] += path_mean(source_to_context_prob, context_to_target_prob)
    fw_target_freq = sum(fw_weighted_contexts.values())
    fw_anls = defaultdict(float)
    for fw_context, context_target_freq in fw_weighted_contexts.items():
        context_freq = self.get_freq(fw_context)
        for source, context_source_freq in self.get_fillers(fw_context, max_length=2):
            source_freq = self.get_freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            fw_anls[source[:-1]] += path_mean(source_to_context_prob, context_to_target_prob)
    anls = {}
    for anl in bw_anls:
        if anl in fw_anls:
            anls[anl] = anl_mean(bw_anls[anl], fw_anls[anl])
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)


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
    is_unseen = (
        (lambda x: x[:len(suff) - 1] == tuple(suff)[:-1]) if '>' in suff
        else (lambda x: False)
    )
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

def rec_morph_anls(self, word, lookup_dict=None):
    if lookup_dict is None:
        lookup_dict = {}
    if word == '<':
        return (frozenset(), [(('<',), 1)])
    elif word in lookup_dict:
        return lookup_dict[word]
    else:
        word = custom_io.hun_encode(word)
        pref_suff_pairs = ((word[:i], word[i:]) for i in range(1, len(word.strip('>'))))
        anl_words = defaultdict(float)
        word_cells = defaultdict(float)
        for pref, suff in pref_suff_pairs:
            # Recursive call
            pref_cell, anl_prefs = rec_morph_anls(self, pref, lookup_dict)
            anl_prefs = anl_prefs.copy()[:20]# + [(pref, 1)]
            # Find outside analogies
            # TODO: integrate into recursive analogy finding below
            outside_anl_bases = outside_morph_anls(self, pref, suff)[:10]
            for anl_base, score in outside_anl_bases:
                anl_pref, anl_pref_cell, anl_word_cell = anl_base
                anl_word = anl_pref + tuple(suff)
                anl_words[anl_word] += score
                word_cell = tulip(anl_pref_cell, anl_word_cell, pref_cell)
                word_cells[word_cell] += score
            # Find recursive (inside-indirect) analogies
            for anl_pref, anl_pref_score in anl_prefs:
                inside_anl_bases = outside_morph_anls(self, anl_pref, suff)[:10]
                for anl_base, score in inside_anl_bases:
                    anl_pref, anl_pref_cell, anl_word_cell = anl_base
                    anl_word = anl_pref + tuple(suff)
                    anl_words[anl_word] += math.sqrt(score * anl_pref_score)
                    word_cell = tulip(anl_pref_cell, anl_word_cell, pref_cell)
                    word_cells[word_cell] += math.sqrt(score * anl_pref_score)
        best_word_cell = sorted(word_cells.keys(), key=word_cells.get, reverse=True)[0]
        anl_word_list = custom_io.dict_to_list(anl_words)
        lookup_dict[word] = (best_word_cell, anl_word_list)
        return (best_word_cell, anl_word_list)

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
