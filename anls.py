#!/usr/bin/env python3
from collections import defaultdict
from itertools import product
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
def freq_trie_setup(corpus):
    """Make a frequency trie data structure from corpus.
    
    Argument:
        corpus (list of tuples of strings)
    
    Returns:
        FreqTrie data structure, representing distribution information about corpus
    """
    freq_trie = FreqTrie()
    for sentence in corpus:
        freq_trie.insert(sentence)
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
    
    def increment_or_make_branch(self, token_tuple):
        """Increment the frequency of token_tuple or make a new branch for it."""
        current_node = self
        for token in token_tuple:
            current_node = current_node.get_or_make_child(token)
            current_node.freq += 1
    
    def get_or_make_child(self, token):
        """Return the child called token or make a new child called token."""
        if token not in self.children:
            self.children[token] = FreqNode()
        return self.children[token]

class FreqTrie:
    def __init__(self):
        self.fw_root = FreqNode()
        self.bw_root = FreqNode()
    
    # Record distribution information about a sentence
    def insert(self, sentence):
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
        for prefix, suffix in prefix_suffix_pairs:
            self.fw_root.increment_or_make_branch(suffix)
            self.bw_root.increment_or_make_branch(reversed(prefix))
    
    def get_context_node(self, context):
        """Return the node corresponding to a context.
        
        Argument:
            context (tuple of strings): of the form ('this', 'is', '_') or
            ('_', 'is', 'good'), with '_' indicating the empty slot.
        
        Returns:
            FreqNode corresponding to context.
        """
        # If left context, look up token sequence in forward trie
        if context[-1] == '_':
            current_node = self.fw_root
            token_sequence = context[:-1]
        # If right context, look up reversed token sequence in backward trie
        else:
            current_node = self.bw_root
            token_sequence = reversed(context[1:])
        # General lookup
        for token in token_sequence:
            try:
                current_node = current_node.children[token]
            except KeyError:
                formatted_context = ' '.join(context)
                raise KeyError(f'Context "{formatted_context}" not found.')
        return current_node
    
    def get_freq(self, context):
        """Return the frequency of context."""
        return self.get_context_node(context).freq
    
    def get_fillers(self, context, max_length=float('inf')):
        """Return generator of fillers of context up to max_length."""
        context_node = self.get_context_node(context)
        # Set direction: "fw" if slot is after context, "bw" if slot is before context
        direction = 'fw' if context[-1] == '_' else 'bw'
        return self.get_fillers_aux(context_node, direction, max_length)
    
    def get_fillers_aux(self, context_node, direction, max_length=float('inf'), path=None):
        """Yield each filler of context_node up to max_length."""
        if path is None:
            path = ['_']
        if len(path) >= max_length:
            return
        for child in context_node.children:
            new_path = path + [child] if direction == 'fw' else [child] + path
            child_node = context_node.children[child]
            freq = child_node.freq
            yield (tuple(new_path), freq)
            yield from self.get_fillers_aux(child_node, direction, max_length, new_path)
    
    def get_shared_fillers(self, context_1, context_2, max_length=float('inf')):
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
        return self.get_shared_fillers_aux(context_node_1, context_node_2, direction, max_length)
  
    # Recursively yield each shared filler of two context nodes
    def get_shared_fillers_aux(self, context_node_1, context_node_2, direction, max_length=float('inf'), path=None):
        """Yield each shared filler of context_node_1 and context_node_2 up to max_length.
        """
        if path is None:
            path = ['_']
        if len(path) >= max_length:
            return
        for child in context_node_1.children:
            if child in context_node_2.children:
                new_path = path + [child] if direction == 'fw' else [child] + path
                child_node_1 = context_node_1.children[child]
                child_node_2 = context_node_2.children[child]
                freq_1 = child_node_1.freq
                freq_2 = child_node_2.freq
                yield (tuple(new_path), freq_1, freq_2)
                yield from self.get_shared_fillers_aux(child_node_1, child_node_2, direction, max_length, new_path)

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

def context_tuples_string(self, left_context, right_context):
    return ' '.join(left_context[:-1]) + ' + ' + ' '.join(right_context[1:])

def context_tuples_merge(self, left_context, right_context):
    return left_context[:-1] + right_context[1:]

def anl_paths(self, source_tuple, target_tuple):
    source_freq = self.get_freq(source_tuple)
    target_freq = self.get_freq(target_tuple)
    yielded_paths = set()
    anl_targets = self.get_fillers(source_tuple, len(target_tuple))
    for anl_target_tuple, source_anl_target_freq in anl_targets:
        anl_target_freq = self.get_freq(anl_target_tuple)
        if (source_tuple, anl_target_tuple) not in yielded_paths:
            path_data = {
                'source_subst': source_tuple,
                'target_subst': anl_target_tuple,
                'joint_freq': source_anl_target_freq,
                'source_freq': source_freq,
                'target_freq': anl_target_freq
            }
            yield path_data
            yielded_paths.add((source_tuple, anl_target_tuple))
        anl_sources = self.get_shared_fillers(anl_target_tuple, target_tuple, len(source_tuple))
        for anl_source_tuple, anl_source_anl_target_freq, anl_source_target_freq in anl_sources:
            anl_source_freq = self.get_freq(anl_source_tuple)
            if (anl_source_tuple, anl_target_tuple) not in yielded_paths:
                path_data = {
                    'source_subst': anl_source_tuple,
                    'target_subst': anl_target_tuple,
                    'joint_freq': anl_source_anl_target_freq,
                    'source_freq': anl_source_freq,
                    'target_freq': anl_target_freq
                }
                yield path_data
                yielded_paths.add((anl_source_tuple, anl_target_tuple))
            if (anl_source_tuple, target_tuple) not in yielded_paths:
                path_data = {
                    'source_subst': anl_source_tuple,
                    'target_subst': target_tuple,
                    'joint_freq': anl_source_target_freq,
                    'source_freq': anl_source_freq,
                    'target_freq': target_freq
                }
                yield path_data
                yielded_paths.add((anl_source_tuple, target_tuple))
            

def rc(self, context_string):
    return ('_',) + tuple(context_string.split())

def lc(self, context_string):
    return tuple(context_string.split()) + ('_',)

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

def cos_substs(self, source_context_string):
    right_source = rc(self, source_context_string)
    right_source_freq = self.get_freq(right_source)
    left_fillers = list(self.get_fillers(right_source, 2))
    left_source_magn = math.sqrt(sum((joint_freq / right_source_freq) ** 2 for filler, joint_freq in left_fillers))
    right_subst_dict = defaultdict(float)
    right_subst_magns = {}
    for left_filler, fw_step_freq in left_fillers:
        left_filler_freq = self.get_freq(left_filler)
        right_substs = self.get_fillers(left_filler, len(right_source))
        for right_subst, bw_step_freq in right_substs:
            right_subst_freq = self.get_freq(right_subst)
            right_subst_fillers = self.get_fillers(right_subst, 2)
            if right_subst not in right_subst_magns:
                right_subst_magns[right_subst] = math.sqrt(sum((joint_freq / right_subst_freq) ** 2 for filler, joint_freq in right_subst_fillers))
            right_subst_magn = right_subst_magns[right_subst]
            source_prob = fw_step_freq / right_source_freq
            subst_prob = bw_step_freq / right_subst_freq
            right_subst_dict[right_subst] += (source_prob * subst_prob) / (left_source_magn * right_subst_magn)
    left_source = lc(self, source_context_string)
    left_source_freq = self.get_freq(left_source)
    right_fillers = list(self.get_fillers(left_source, 2))
    right_source_magn = math.sqrt(sum((joint_freq / left_source_freq) ** 2 for filler, joint_freq in right_fillers))
    left_subst_dict = defaultdict(float)
    left_subst_magns = {}
    for right_filler, fw_step_freq in right_fillers:
        right_filler_freq = self.get_freq(right_filler)
        left_substs = self.get_fillers(right_filler, len(left_source))
        for left_subst, bw_step_freq in left_substs:
            left_subst_freq = self.get_freq(left_subst)
            left_subst_fillers = self.get_fillers(left_subst, 2)
            if left_subst not in left_subst_magns:
                left_subst_magns[left_subst] = math.sqrt(sum((joint_freq / left_subst_freq) ** 2 for filler, joint_freq in left_subst_fillers))
            left_subst_magn = left_subst_magns[left_subst]
            left_subst_magn = left_subst_magns[left_subst]
            source_prob = fw_step_freq / left_source_freq
            subst_prob = bw_step_freq / left_subst_freq
            left_subst_dict[left_subst] += (source_prob * subst_prob) / (right_source_magn * left_subst_magn)
    subst_dict = {}
    for left_subst, left_subst_score in left_subst_dict.items():
        subst_dict[left_subst[:-1]] = min(left_subst_score, right_subst_dict[('_',) + left_subst[:-1]])
    """
    for right_subst, right_subst_score in right_subst_dict.items():
        subst_dict[right_subst[1:]] = (right_subst_score + left_subst_dict[left_subst[1:] + ('_',)])
    """
    return sorted(subst_dict.items(), key=lambda x: x[1], reverse=True)

def anl_phrases(self, source, target):
    anl_phrase_set = set()
    anl_source_dict = {}
    anl_target_dict = {}
    source_context = source + ('_',)
    target_context = ('_',) + target
    anl_path_list = anl_paths(self, source_context, target_context)
    for path_data in anl_path_list:
        anl_source = path_data['source_subst'][:-1]
        anl_target = path_data['target_subst'][1:]
        anl_phrase_set.add((anl_source, anl_target))
        if anl_source not in anl_source_dict:
            score = subst_score(self, anl_source, source)
            anl_source_dict[anl_source] = score
        if anl_target not in anl_target_dict:
            score = subst_score(self, anl_target, target)
            anl_target_dict[anl_target] = score
    return sorted(list(((anl_source, anl_target), min(anl_source_dict[anl_source], anl_target_dict[anl_target])) for anl_source, anl_target in anl_phrase_set), key=lambda x: x[1], reverse=True)

def indir_anls(self, source, target):
    anl_phrase_dict = defaultdict(float)
    alt_sources = anl_substs(self, source)[:5]
    alt_targets = anl_substs(self, target)[:5]
    for alt_source_data, alt_target_data in product(alt_sources, alt_targets):
        alt_source, alt_source_score = alt_source_data
        alt_target, alt_target_score = alt_target_data
        score = math.sqrt(alt_source_score * alt_target_score)
        anl_phrase_list = anl_phrases(self, alt_source, alt_target)
        for anl_phrase, anl_phrase_score in anl_phrase_list:
            anl_phrase_dict[anl_phrase] += min(anl_phrase_score, score)
    return sorted(anl_phrase_dict.items(), key=lambda x: x[1], reverse=True)

def subst_score(self, subst_tuple, orig_tuple):
    left_subst_score = 0
    subst_left, orig_left = subst_tuple + ('_',), orig_tuple + ('_',)
    subst_freq = self.get_freq(subst_left)
    shared_contexts = self.get_shared_fillers(subst_left, orig_left)
    for shared_context, subst_context_freq, orig_context_freq in shared_contexts:
        context_freq = self.get_freq(shared_context)
        context_given_subst = subst_context_freq / subst_freq
        orig_given_context = orig_context_freq / context_freq
        left_subst_score += path_mean(context_given_subst, orig_given_context)
    right_subst_score = 0
    subst_right, orig_right = ('_',) + subst_tuple, ('_',) + orig_tuple
    shared_contexts = self.get_shared_fillers(subst_right, orig_right)
    for shared_context, subst_context_freq, orig_context_freq in shared_contexts:
        context_freq = self.get_freq(shared_context)
        context_given_subst = subst_context_freq / subst_freq
        orig_given_context = orig_context_freq / context_freq
        right_subst_score += path_mean(context_given_subst, orig_given_context)
    return subst_mean(left_subst_score, right_subst_score)