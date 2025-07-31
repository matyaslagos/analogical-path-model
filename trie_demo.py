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
            # If keys include paradigm cell features
            if len(sequence_data) == 2:
                sequence, cell_features = sequence_data
                freq_trie.insert(sequence, frequency, cell_features)
            # Elif keys are just sequences
            else:
                freq_trie.insert(sequence_data, frequency)
    # Elif corpus is just an iterator of sequences
    else:
        for sequence in corpus:
            freq_trie.insert(sequence)
    return freq_trie

def setup():
    corpus = txt_to_list('norvig_corpus.txt')
    return freq_trie_setup(corpus)

#-----------------------------------------#
# FreqNode and FreqTrie class definitions #
#-----------------------------------------#

class FreqNode:
    def __init__(self):
        self.children = {}
        self.freq = 0
        self.cell = None
    
    def _increment_or_make_branch(self, sequence, count=1):
        """Increment the frequency of token_tuple or make a new branch for it.
        """
        current_node = self
        for token in sequence:
            current_node = current_node._get_or_make_child(token)
            current_node.freq += count
    
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
    
    def insert(self, sequence, count=1, cell_features=None):
        """Record distributions of prefixes and suffixes of sequence.
        
        Arguments:
            sequence (tuple of strings): e.g. ('<', 'this', 'is', 'good', '>')
            count (int): how many occurrences of sequence should be recorded
        
        Effect:
            For each prefix--suffix split of sequence, record the occurrences of
            prefix and suffix. (Prefix is reversed to make shared-neighbor search more
            efficient.)
        """
        # Add token frequency mass of sequence to root nodes
        token_freq_mass = len(sequence) * count
        self.fw_root.freq += token_freq_mass
        self.bw_root.freq += token_freq_mass
        # Record each suffix in fw trie and each prefix in bw trie
        prefix_suffix_pairs = (
            (sequence[:i], sequence[i:])
            for i in range(len(sequence) + 1)
        )
        for prefix, suffix in prefix_suffix_pairs:
            self.fw_root._increment_or_make_branch(suffix, count)
            self.bw_root._increment_or_make_branch(reversed(prefix), count)
        # Record cell features for full sequence
        if cell_features is not None:
            self.sequence_node(sequence).cell = cell_features
    
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
    
    def cell(self, sequence):
        seq_node = self.sequence_node(sequence)
        return seq_node.cell if seq_node else frozenset()
    
    def right_neighbors(self, sequence, max_length=float('inf'),
        min_length=0, only_completions=False):
        return self.neighbors(sequence, 'fw',
                              max_length, min_length, only_completions)

    def left_neighbors(self, sequence, max_length=float('inf'),
        min_length=0, only_completions=False):
        return self.neighbors(sequence, 'bw',
                              max_length, min_length, only_completions)
    
    def neighbors(self, sequence, direction='fw',
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
            if not only_completions and len(new_path) >= min_length:
                yield (tuple(new_path), freq)
            else:
                if (new_path[0] == '<' or new_path[-1] == '>') and len(new_path) >= min_length:
                    yield (tuple(new_path), freq)
            yield from self._neighbors_aux(child_node, direction,
                                           max_length, min_length,
                                           only_completions, new_path)
    
    def shared_right_neighbors(self, sequence_1, sequence_2, max_length=float('inf'),
        min_length=0, only_completions=False):
        return self.shared_neighbors(sequence_1, sequence_2, 'fw',
                                     max_length, min_length, only_completions)
    
    def shared_left_neighbors(self, sequence_1, sequence_2, max_length=float('inf'),
        min_length=0, only_completions=False):
        return self.shared_neighbors(sequence_1, sequence_2, 'bw',
                                     max_length, min_length, only_completions)
    
    def shared_neighbors(self, sequence_1, sequence_2, direction='fw',
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
                if (not only_completions) and len(new_path) >= min_length:
                    yield (tuple(new_path), freq_1, freq_2)
                else:
                    if (new_path[0] == '<' or new_path[-1] == '>') and len(new_path) >= min_length:
                        yield (tuple(new_path), freq_1, freq_2)
                yield from self._shared_neighbors_aux(child_node_1, child_node_2,
                                                      direction, max_length, min_length,
                                                      only_completions, new_path)

def path_mean(n, m):
    return min(n, m)

def subst_mean(n, m):
    return min(n, m)

def anl_mean(n, m):
    return min(n, m)

def anl_substs(self, source_context_string):
    right_source = rc(self, source_context_string)
    right_source_freq = self.get_freq(right_source)
    left_fillers = self.get_fillers(right_source)
    right_subst_dict = defaultdict(float)
    for left_filler, fw_step_freq in left_fillers:
        left_filler_freq = self.get_freq(left_filler)
        right_substs = self.get_fillers(left_filler, len(right_source))
        for right_subst, bw_step_freq in right_substs:
            right_subst_freq = self.get_freq(right_subst)
            fw_prob = fw_step_freq / (left_filler_freq)
            bw_prob = bw_step_freq / (right_subst_freq)
            right_subst_dict[right_subst] += path_mean(fw_prob, bw_prob)
    left_source = lc(self, source_context_string)
    left_source_freq = self.get_freq(left_source)
    left_subst_dict = defaultdict(float)
    for right_subst, right_subst_score in right_subst_dict.items():
        left_subst = right_subst[1:] + ('_',)
        left_subst_freq = self.get_freq(left_subst)
        right_fillers = self.get_shared_fillers(left_subst, left_source)
        for right_filler, bw_step_freq, fw_step_freq in right_fillers:
            right_filler_freq = self.get_freq(right_filler)
            fw_prob = fw_step_freq / (right_filler_freq)
            bw_prob = bw_step_freq / (left_subst_freq)
            left_subst_dict[left_subst] += path_mean(fw_prob, bw_prob)
    subst_dict = {}
    for left_subst, left_subst_score in left_subst_dict.items():
        subst_dict[left_subst[:-1]] = subst_mean(left_subst_score, right_subst_dict[('_',) + left_subst[:-1]])
    return sorted(subst_dict.items(), key=lambda x: x[1], reverse=True)

def analogies(self, word):
    word_freq = self.get_freq(lc(self, word))
    best_anls = anl_substs(self, word)[1:11]
    anl_list = [
        (anl[0], self.get_freq(anl[0] + ('_',)))
        for anl in best_anls
    ]
    max_len = max(len(anl[0]) for anl, freq in anl_list)
    print(f'Ten best analogies for "{word}" ({word_freq}):')
    for anl, freq in anl_list:
        space = ' ' * (max_len - len(anl[0]) + 1)
        print(f'- "{anl[0]}"{space}({freq})')

def min_anls(self, target):
    """Target is undirected, i.e. ('apple',)."""
    fw_target, bw_target = target + ('_',), ('_',) + target
    fw_anls = defaultdict(float)
    for anl, context, score in min_anls_dir(self, fw_target):
        fw_anls[anl] += score
    bw_anls = defaultdict(float)
    for anl, context, score in min_anls_dir(self, bw_target):
        bw_anls[anl] += score
    anls = {}
    for anl in fw_anls:
        if anl in bw_anls:
            anls[anl] = subst_mean(fw_anls[anl], bw_anls[anl])
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def min_anls_dir(self, target):
    """Target is directed, i.e. ('apple', '_') or ('_', 'apple')."""
    target_freq = self.get_freq(target)
    contexts = self.get_fillers(target)
    for context, context_target_freq in contexts:
        context_freq = self.get_freq(context)
        sources = self.get_fillers(context, len(target))
        for source, context_source_freq in sources:
            source_freq = self.get_freq(source)
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            source = source[:-1] if source.index('_') else source[1:]
            yield (source, context, path_mean(source_to_context_prob, context_to_target_prob))

def anl_paths_dir(self, target):
    """Target is directed, i.e. ('apple', '_') or ('_', 'apple').
    """
    target_freq = self.get_freq(target)
    contexts = self.get_fillers(target)
    for context, context_target_freq in contexts:
        context_freq = self.get_freq(context)
        sources = self.get_fillers(context, len(target))
        for source, context_source_freq in sources:
            source_freq = self.get_freq(source)
            src_to_cxt_prob = context_source_freq / source_freq
            cxt_to_trg_prob = context_target_freq / context_freq
            source = source[:-1] if source.index('_') else source[1:]
            yield (source, context, src_to_cxt_prob, cxt_to_trg_prob)

def bigram_anls(self, bigram):
    s1, s2 = tuple(bigram.split()[:1]), tuple(bigram.split()[1:])
    s1_anls = min_anls(self, s1)[:50]
    s2_anls = min_anls(self, s2)[:50]
    anls = {}
    for s1_anl, s1_score in s1_anls:
        for s2_anl, s2_score in s2_anls:
            if self.get_freq(s1_anl + s2_anl + ('_',)) > 0:
                anls[s1_anl + s2_anl] = anl_mean(s1_score, s2_score)
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

#> Analogy-finding <#

def morph_anls(self, target, unseen=lambda x: False, encode=False):
    """Target is signed string, i.e. '<kalap' or 'jaim>'.
    """
    if encode:
        target = custom_io.hun_encode(target)
    fw_anls = defaultdict(float)
    for anl, context, score in morph_anls_dir(self, target, 'fw', unseen):
        if encode:
            anl = custom_io.hun_decode(''.join(anl))
        fw_anls[anl] += score
    bw_anls = defaultdict(float)
    for anl, context, score in morph_anls_dir(self, target, 'bw', unseen):
        if encode:
            anl = custom_io.hun_decode(''.join(anl))
        bw_anls[anl] += score
    anls = {}
    if '<' in target:
        anls = fw_anls
    elif '>' in target:
        anls = bw_anls
    else:
        for fw_anl in fw_anls:
            if fw_anl in bw_anls:
                anls[fw_anl] = subst_mean(fw_anls[fw_anl], bw_anls[fw_anl])
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def morph_anls_dir(self, target, direction, unseen):
    """Target is directed.
    """
    target_freq = self.freq(target)
    contexts = self.neighbors(target, direction, max_length=5, min_length=3)
    other_direction = 'fw' if direction == 'bw' else 'bw'
    for context, context_target_freq in contexts:
        if unseen(context):
            continue
        context_freq = self.freq(context)
        if '<' in target or '>' in target:
            sources = self.neighbors(context, other_direction, only_completions=True)
        else:
            sources = self.neighbors(context, other_direction, only_completions=True)
        for source, context_source_freq in sources:
            source_freq = self.freq(source)
            if source_freq == 0:
                continue
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / target_freq
            yield (source, context, path_mean(source_to_context_prob, context_to_target_prob))

def morph_anl_fixed_c2(self, c1, c2):
    anls = {}
    startswith_c2 = lambda x: x[:len(c2) - 1] == tuple(c2)[:-1]
    try:
        c1_anls = morph_anls(self, c1, unseen=startswith_c2)
    except:
        return []
    for c1_anl, c1_score in c1_anls:
        if c1_anl == tuple(c1):
            continue
        full_word = c1_anl + tuple(c2)
        if self.freq(full_word) > 0:
            anls[custom_io.hun_decode(''.join(full_word))] = c1_score
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def morph_anls_iter(self, word, encode=False):
    if encode:
        word = custom_io.hun_encode(word)
    total_score = 0
    anls_by_split = {}
    anl_words = defaultdict(float)
    constituent_pairs = ((word[:i], word[i:]) for i in range(1, len(word)))
    for c1, c2 in constituent_pairs:
        dec_c1, dec_c2 = custom_io.hun_decode(c1), custom_io.hun_decode(c2)
        anl_data = morph_anl_fixed_c2(self, '<' + c1, c2 + '>')
        split_score = sum(anl[1] for anl in anl_data)
        anls_by_split[(dec_c1, dec_c2)] = (split_score, anl_data[:10])
        total_score += split_score
        for anl_prefix, score in anl_data:
            anl_words[custom_io.hun_decode(anl_prefix)] += score
    anl_words = sorted(anl_words.items(), key=lambda x: x[1], reverse=True)
    return (anls_by_split, anl_words, total_score)

#> New iter algorithm <#

def morph_anls_ending(self, prefix, suffix):
    unseen = lambda x: x[:len(suffix) - 1] == tuple(suffix)[:-1]
    prefix_freq = self.freq(prefix)
    anl_prefixes = self.left_neighbors(suffix, only_completions=True)
    anl_prefix_dict = defaultdict(float)
    for anl_prefix, anl_prefix_suffix_freq in anl_prefixes:
        anl_prefix_freq = self.freq(anl_prefix)
        shared_contexts = self.shared_right_neighbors(prefix, anl_prefix,
                                                      min_length=2, only_completions=True)
        for context, context_prefix_freq, context_anl_prefix_freq in shared_contexts:
            if unseen(context):
                continue
            context_freq = self.freq(context)
            path_out = context_anl_prefix_freq / anl_prefix_freq
            path_in = context_prefix_freq / prefix_freq
            anl_prefix_dict[anl_prefix + tuple(suffix)] += path_mean(path_out, path_in)
    return sorted(anl_prefix_dict.items(), key=lambda x: x[1], reverse=True)

def morph_anls_ending_iter(self, word):
    prefix_suffix_pairs = ((word[:i], word[i:]) for i in range(1, len(word)))
    anl_words_dict = defaultdict(float)
    for prefix, suffix in prefix_suffix_pairs:
        if self.freq('<' + prefix) == 0 or self.freq(suffix + '>') == 0:
            continue
        anls = morph_anls_ending(self, '<' + prefix, suffix + '>')
        score = sum(x[1] for x in anls)
        for anl, score in anls:
            anl_words_dict[anl] += score
    return dict_format(anl_words_dict)

def dict_format(dy):
    sorted_dy = sorted(dy.items(), key=lambda x: x[1], reverse=True)
    return list(map(lambda x: (custom_io.hun_decode(''.join(x[0])), x[1]), sorted_dy))

# Find anl_prefs for pref by examining their right distributions
def outside_morph_anls(self, pref, suff, find_cells=True):
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
        anl_pref_cell = self.cell(anl_pref + ('>',))
        anl_word_cell = self.cell(anl_pref + tuple(suff.strip('>')) + ('>',))
        if not anl_word_cell:
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
            if find_cells:
                key = (anl_pref, anl_pref_cell, anl_word_cell)
            else:
                key = anl_pref
            anl_dict[key] += score
    return custom_io.dict_to_list(anl_dict)

def rec_morph_anls(self, word, lookup_dict={}):
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
            anl_prefs = anl_prefs.copy()[:20]
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

def tulip(set_a, set_b, set_c):
    center = set_a.intersection(set_b).intersection(set_c)
    left = set_b.difference(set_a)
    right = set_c.difference(set_a)
    return left.union(center).union(right)
