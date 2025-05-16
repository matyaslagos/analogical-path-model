#!/usr/bin/env python3

from collections import defaultdict

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
                raise KeyError(f'Context "{' '.join(context)}" not found.')
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

def context_tuples_merge(self, left_context, right_context):
    return ' '.join(left_context[:-1]) + ' + ' + ' '.join(right_context[1:])

def anl_paths(self, left_context_string, right_context_string):
    left_context = tuple(left_context_string.split()) + ('_',)
    ext_left_context = ('_',) + tuple(left_context_string.split())
    left_context_freq = self.get_freq(left_context)
    right_context = ('_',) + tuple(right_context_string.split())
    ext_right_context = tuple(right_context_string.split()) + ('_',)
    right_context_freq = self.get_freq(right_context)
    left_context_node = self.get_context_node(left_context)
    right_context_node = self.get_context_node(right_context)
    anl_right_contexts = self.get_fillers(left_context, len(right_context))
    ext_probs = {}
    for anl_right_context, lc_arc_freq in anl_right_contexts:
        anl_left_contexts = list(self.get_shared_fillers(anl_right_context, right_context, len(left_context)))
        anl_right_context_freq = self.get_freq(anl_right_context)
        if bool(anl_left_contexts) and anl_right_context not in ext_probs:
            ext_anl_right_context = anl_right_context[1:] + ('_',)
            ext_right_fillers = self.get_shared_fillers(ext_anl_right_context, ext_right_context, len(left_context))
            ext_right_prob = 0
            for ext_right_filler, earc_erf_freq, erc_erf_freq in ext_right_fillers:
                ext_right_filler_freq = self.get_freq(ext_right_filler)
                ext_right_prob += (earc_erf_freq * erc_erf_freq) / (anl_right_context_freq * ext_right_filler_freq)
            ext_probs[anl_right_context] = ext_right_prob
        for anl_left_context, alc_arc_freq, alc_rc_freq in anl_left_contexts:
            anl_left_context_freq = self.get_freq(anl_left_context)
            if anl_left_context not in ext_probs:
                ext_anl_left_context = ('_',) + anl_left_context[:-1]
                ext_left_fillers = self.get_shared_fillers(ext_anl_left_context, ext_left_context, len(right_context))
                ext_left_prob = 0
                for ext_left_filler, ealc_elf_freq, elc_elf_freq in ext_left_fillers:
                    ext_left_filler_freq = self.get_freq(ext_left_filler)
                    ext_left_prob += (ealc_elf_freq * elc_elf_freq) / (anl_left_context_freq * ext_left_filler_freq)
                ext_probs[anl_left_context] = ext_left_prob
            anl_path_data = {
                'middle_edge': (anl_left_context, anl_right_context),
                'middle_edge_freq': alc_arc_freq,
                'top_edge': (left_context, anl_right_context),
                'top_edge_freq': lc_arc_freq,
                'bottom_edge': (anl_left_context, right_context),
                'bottom_edge_freq': alc_rc_freq,
                'left_freq': left_context_freq,
                'right_freq': right_context_freq,
                'left_anl_freq': anl_left_context_freq,
                'right_anl_freq': anl_right_context_freq,
                'ext_left_prob': ext_probs[anl_left_context],
                'ext_right_prob': ext_probs[anl_right_context]
            }
            yield anl_path_data

def anl_substs(self, left_context_string, right_context_string):
    anl_paths_data = anl_paths(self, left_context_string, right_context_string)
    anl_subst_dict = {}
    for ap_data in anl_paths_data:
        left_subst, right_subst = ap_data['middle_edge']
        anl_subst_gram = context_tuples_merge(self, left_subst, right_subst)
        top_edge_prob = ap_data['top_edge_freq']# / ap_data['left_freq']
        middle_edge_prob = ap_data['middle_edge_freq'] / ap_data['right_anl_freq']
        bottom_edge_prob = ap_data['bottom_edge_freq'] / ap_data['left_anl_freq']
        anl_subst_dict[anl_subst_gram] = top_edge_prob * middle_edge_prob * bottom_edge_prob * (ap_data['ext_left_prob'] * ap_data['ext_right_prob'])
    return sorted(anl_subst_dict.items(), key=lambda x: x[1], reverse=True)

def rc(self, context_string):
    return ('_',) + tuple(context_string.split())

def lc(self, context_string):
    return tuple(context_string.split()) + ('_',)

def subst(self, source_context_string, target_context_string):
    right_source, right_target = rc(self, source_context_string), rc(self, target_context_string)
    left_source, left_target = lc(self, source_context_string), lc(self, target_context_string)
    right_fillers = self.get_shared_fillers(left_source, left_target)
    source_freq = self.get_freq(left_source)
    left_subst_prob = 0
    for right_filler, source_filler_freq, target_filler_freq in right_fillers:
        filler_freq = self.get_freq(right_filler)
        fw_step = source_filler_freq / left_source_freq
        bw_step = target_filler_freq / filler_freq
        left_subst_prob += fw_step * bw_step
    left_fillers = self.get_shared_fillers(right_source, right_target)
    right_subst_prob = 0
    pass

def anl_repls(self, source_context_string):
    
    right_source = rc(self, source_context_string)
    
    left_fillers = self.get_fillers(right_source)
    right_subst_dict = defaultdict(float)
    for left_filler, fw_step_freq in left_fillers:
        left_filler_freq = self.get_freq(left_filler)
        right_substs = self.get_fillers(left_filler, len(right_source))
        for right_subst, bw_step_freq in right_substs:
            right_subst_freq = self.get_freq(right_subst)
            fw_prob = fw_step_freq / left_filler_freq
            bw_prob = bw_step_freq / right_subst_freq
            right_subst_dict[right_subst] += min(fw_prob, bw_prob)
    
    left_source = lc(self, source_context_string)
    
    right_fillers = self.get_fillers(left_source)
    left_subst_dict = defaultdict(float)
    for right_filler, fw_step_freq in right_fillers:
        right_filler_freq = self.get_freq(right_filler)
        left_substs = self.get_fillers(right_filler, len(left_source))
        for left_subst, bw_step_freq in left_substs:
            left_subst_freq = self.get_freq(left_subst)
            fw_prob = fw_step_freq / right_filler_freq
            bw_prob = bw_step_freq / left_subst_freq
            left_subst_dict[left_subst] += min(fw_prob, bw_prob)
    subst_dict = {}
    for right_subst, right_prob in right_subst_dict.items():
        if right_subst[1:] + ('_',) in left_subst_dict:
            subst_dict[right_subst[1:]] = min(right_prob, left_subst_dict[right_subst[1:] + ('_',)])
    return sorted(subst_dict.items(), key=lambda x: x[1], reverse=True)

def min_anls(self, source_string, target_string):
    pass