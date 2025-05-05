#!/usr/bin/env python3

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