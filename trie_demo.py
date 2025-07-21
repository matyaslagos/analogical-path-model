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
    if isinstance(corpus, dict):
        for seq, freq in corpus.items():
            freq_trie.insert(seq, freq)
    else:
        for seq in corpus:
            freq_trie.insert(seq)
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
    
    def increment_or_make_branch(self, token_tuple, count=1):
        """Increment the frequency of token_tuple or make a new branch for it.
        """
        current_node = self
        for token in token_tuple:
            current_node = current_node.get_or_make_child(token)
            current_node.freq += count
    
    def get_or_make_child(self, token):
        """Return the child called token or make a new child called token.
        """
        if token not in self.children:
            self.children[token] = FreqNode()
        return self.children[token]

class FreqTrie:
    def __init__(self):
        self.fw_root = FreqNode()
        self.bw_root = FreqNode()
    
    # Record distribution information about a sentence
    def insert(self, sentence, count=1):
        """Record all contexts and fillers of sentence into trie.
        
        Argument:
            sentence (tuple of strings): e.g. ('<', 'this', 'is', 'good', '>')
        
        Effect:
            For each prefix--suffix split of sentence, record the occurrences of
            prefix and suffix. (Prefix is reversed to make shared-filler search more
            efficient.)
        """
        prefix_suffix_pairs = (
            (sentence[:i], sentence[i:])
            for i in range(len(sentence) + 1)
        )
        self.fw_root.freq += len(sentence) * count
        for prefix, suffix in prefix_suffix_pairs:
            self.fw_root.increment_or_make_branch(suffix, count)
            self.bw_root.increment_or_make_branch(reversed(prefix), count)
    
    def get_context_node(self, context):
        """Return the node corresponding to a context.
        
        Argument:
            context (tuple of strings): of the form ('this', 'is', '_') or
            ('_', 'is', 'good'), with '_' indicating the empty slot. If no
            slot is indicated, defaults to ('this', 'is', '_')
        
        Returns:
            FreqNode corresponding to context.
        """
        # If left context, look up token sequence in forward trie
        if context[-1] == '_':
            current_node = self.fw_root
            token_sequence = context[:-1]
        # If right context, look up reversed token sequence in backward trie
        elif context[0] == '_':
            current_node = self.bw_root
            token_sequence = reversed(context[1:])
        else:
            current_node = self.fw_root
            token_sequence = context
        # General lookup
        for token in token_sequence:
            try:
                current_node = current_node.children[token]
            except KeyError:
                return None
        return current_node
    
    def get_freq(self, context):
        """Return the frequency of context.
        """
        context_node = self.get_context_node(context)
        return context_node.freq if context_node else 0
    
    def get_fillers(self, context, max_length=float('inf'), min_length=0, only_completions=False):
        """Return generator of fillers of context up to max_length.
        """
        context_node = self.get_context_node(context)
        # Set direction: "fw" if slot is after context, "bw" if slot is before context
        direction = 'fw' if context[-1] == '_' else 'bw'
        return self.get_fillers_aux(context_node, direction, max_length, min_length, only_completions, path=['_'])
    
    def get_fillers_aux(self, context_node, direction, max_length, min_length, only_completions, path):
        """Yield each filler of context_node up to max_length.
        """
        if len(path) >= max_length:
            return
        for child in context_node.children:
            new_path = path + [child] if direction == 'fw' else [child] + path
            child_node = context_node.children[child]
            freq = child_node.freq
            if not only_completions and len(new_path) >= min_length:
                yield (tuple(new_path), freq)
            else:
                if new_path[0] == '<' or new_path[-1] == '>' and len(new_path) >= min_length:
                    yield (tuple(new_path), freq)
            yield from self.get_fillers_aux(child_node, direction, max_length, min_length, only_completions, new_path)
    
    def get_shared_fillers(self, context_1, context_2, max_length=float('inf'), only_completions=False):
        """Return generator of shared fillers of context_1 and context_2 up to max_length.
        
        Arguments:
            context_1 (tuple of strings): e.g. ('_', 'is', 'good')
            context_2 (tuple of strings): e.g. ('_', 'was', 'here')
        
        Returns:
            generator of (filler, freq_1, freq_2) tuples:
                if e.g. the tuple (('this', '_'), 23, 10) is yielded, then:
                - 'this' occurred before 'is good' 23 times, and
                - 'this' occurred before 'was here' 10 times.
        """
        context_node_1 = self.get_context_node(context_1)
        context_node_2 = self.get_context_node(context_2)
        direction = 'fw' if context_1[-1] == '_' else 'bw'
        return self.get_shared_fillers_aux(context_node_1, context_node_2, direction,
                                           max_length, only_completions, path=['_'])
  
    # Recursively yield each shared filler of two context nodes
    def get_shared_fillers_aux(self, context_node_1, context_node_2, direction,
                               max_length, only_completions, path):
        """Yield each shared filler of context_node_1 and context_node_2 up to max_length.
        """
        if len(path) >= max_length:
            return
        for child in context_node_1.children:
            if child in context_node_2.children:
                new_path = path + [child] if direction == 'fw' else [child] + path
                child_node_1 = context_node_1.children[child]
                child_node_2 = context_node_2.children[child]
                freq_1 = child_node_1.freq
                freq_2 = child_node_2.freq
                if not only_completions:
                    yield (tuple(new_path), freq_1, freq_2)
                else:
                    if new_path[0] == '<' or new_path[-1] == '>':
                        yield (tuple(new_path), freq_1, freq_2)
                yield from self.get_shared_fillers_aux(child_node_1, child_node_2, direction, max_length, only_completions, new_path)

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

def mixed_anls(self, weighted_phraselist):
    """Finds analogies for the mixed distribution of weighted_phraselist.
    
    Arguments:
        weighted_phraselist (list): [(tuple, float)], where tuple is a tuple
            of strings, e.g. ('our', 'teacher'), and float is its weight, i.e.
            its substitutability score by some target phrase
    
    Returns:
        ...
    """
    # Aggregate probabilities using left contexts
    left_halves = ((('_',) + phrase, weight) for phrase, weight in weighted_phraselist)
    left_src_to_cxt = defaultdict(lambda: {})
    left_cxt_to_trg = defaultdict(float)
    for target, weight in left_halves:
        anl_paths = anl_paths_dir(self, target)
        for source, context, src_to_cxt_prob, cxt_to_trg_prob in anl_paths:
            left_src_to_cxt[source][context] = src_to_cxt_prob
            left_cxt_to_trg[context] += cxt_to_trg_prob
    # Aggregate probabilities using right contexts
    right_halves = ((phrase + ('_',), weight) for phrase, weight in weighted_phraselist)
    right_src_to_cxt = defaultdict(lambda: {})
    right_cxt_to_trg = defaultdict(float)
    for target, weight in right_halves:
        anl_paths = anl_paths_dir(self, target)
        for source, context, src_to_cxt_prob, cxt_to_trg_prob in anl_paths:
            right_src_to_cxt[source][context] = src_to_cxt_prob
            right_cxt_to_trg[context] += cxt_to_trg_prob
    # Compute path means for left contexts
    left_anls = defaultdict(float)
    for source in left_src_to_cxt:
        for context in left_src_to_cxt[source]:
            left_anls[source] += path_mean(
                left_src_to_cxt[source][context],
                left_cxt_to_trg[context]
            )
    # Compute path means for right contexts
    right_anls = defaultdict(float)
    for source in right_src_to_cxt:
        for context in right_src_to_cxt[source]:
            right_anls[source] += path_mean(
                right_src_to_cxt[source][context],
                right_cxt_to_trg[context]
            )
    # Calculate bilateral substitutability
    anls = {}
    for anl in left_anls:
        if anl in right_anls:
            anls[anl] = subst_mean(left_anls[anl], right_anls[anl])
    
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

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
    fw_target = tuple(target) + ('_',)
    fw_anls = defaultdict(float)
    for anl, context, score in morph_anls_dir(self, fw_target, unseen):
        if encode:
            anl = custom_io.hun_decode(''.join(anl))
        fw_anls[anl] += score
    bw_target = ('_',) + tuple(target)
    bw_anls = defaultdict(float)
    for anl, context, score in morph_anls_dir(self, bw_target, unseen):
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

def morph_anls_dir(self, target, unseen):
    """Target is directed, i.e. ('apple', '_') or ('_', 'apple').
    """
    target_freq = self.get_freq(target)
    contexts = self.get_fillers(target, max_length=5, min_length=3)
    for context, context_target_freq in contexts:
        if unseen(context):
            continue
        context_freq = self.get_freq(context)
        if '<' in target or '>' in target:
            sources = self.get_fillers(context, only_completions=True)
        else:
            sources = self.get_fillers(context, max_length=5)
        for source, context_source_freq in sources:
            source_freq = self.get_freq(source)
            if source_freq == 0:
                continue
            source_to_context_prob = context_source_freq / source_freq
            context_to_target_prob = context_target_freq / context_freq
            source = source[:-1] if source.index('_') else source[1:]
            yield (source, context, path_mean(source_to_context_prob, context_to_target_prob))

def morph_anl_fixed_c2(self, c1, c2):
    anls = {}
    startswith_c2 = lambda x: x[1:len(c2)] == tuple(c2)[:-1]
    try:
        c1_anls = morph_anls(self, c1, unseen=startswith_c2)
    except:
        return [('-', 0)]
    for c1_anl, c1_score in c1_anls:
        if c1_anl == tuple(c1):
            continue
        if self.get_freq(c1_anl + tuple(c2)) > 0:
            anls[custom_io.hun_decode(''.join(c1_anl)[1:])] = c1_score
    return sorted(anls.items(), key=lambda x: x[1], reverse=True)

def morph_anls_iter(self, word, encode=False):
    if encode:
        word = custom_io.hun_encode(word)
    total_score = 0
    anls_by_split = {}
    anl_words = defaultdict(float)
    constituent_pairs = ((word[:i], word[i:]) for i in range(1, len(word) + 1))
    for c1, c2 in constituent_pairs:
        dec_c1, dec_c2 = custom_io.hun_decode(c1), custom_io.hun_decode(c2)
        anl_data = morph_anl_fixed_c2(self, '<' + c1, c2 + '>')
        split_score = sum(anl[1] for anl in anl_data)
        anls_by_split[(dec_c1, dec_c2)] = (split_score, anl_data[:10])
        total_score += split_score
        for anl_prefix, score in anl_data:
            anl_words[custom_io.hun_decode(anl_prefix + ''.join(c2))] += score
    anl_words = sorted(anl_words.items(), key=lambda x: x[1], reverse=True)
    pp((anls_by_split, anl_words[:10], total_score))

def relevant_endparts(self, word, suffix):
    suffix_node = self.get_context_node(('_',) + tuple(suffix))
    for char in tuple(word):
        # calculate how much char influences the probability of suffix
        pass

def rec_morph_anls(self, word, lookup_dict={}):
    if word == ('<',):
        return [(('<',), 1)]
    elif word in lookup_dict:
        return lookup_dict[word]
    else:
        pref_suff_pairs = ((word[:i], word[i:]) for i in range(1, len(word)))
        anl_words = defaultdict(float)
        for pref, suff in pref_suff_pairs:
            # Find outer analogies if prefix is attested
            anl_prefs = []
            # Ensure that we haven't seen full word
            if '>' in suff:
                unseen_ending = lambda x: x[1:len(suff)] == tuple(suff)[:-1]
            else:
                unseen_ending = lambda x: False
            # Find outer analogies
            if self.get_freq(pref + ('_',)) > 0:
                anl_prefs += morph_anls(self, pref, unseen=unseen_ending)
            # Find recursive (inner) analogies
            inner_anl_prefs = rec_morph_anls(self, pref, lookup_dict)
            # Get those analogical prefixes that occurred before suffix
            for inner_anl_pref, score in inner_anl_prefs:
                anl_inner_anl_prefs = morph_anls(self, inner_anl_pref)
                anl_prefs += anl_inner_anl_prefs
            for anl_pref, score in anl_prefs:
                if self.get_freq(anl_pref + suff) > 0:
                    anl_words[anl_pref + suff] += score
        anl_word_list = sorted(anl_words.items(), key=lambda x: x[1], reverse=True)
        lookup_dict[word] = anl_word_list
        print('hi')
        return anl_word_list
