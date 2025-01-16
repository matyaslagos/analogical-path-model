from enum import Enum, auto
from collections import defaultdict, namedtuple
from copy import deepcopy
from pprint import pp

# Function implementation

# Import a txt file (containing one sentence per line) as a list whose each
# element is a list of words corresponding to a line in the txt file:
def txt2list(filename):
    """Import a txt list of sentences as a list of lists of words.

    Argument:
        - filename (string), e.g.: 'grimm_corpus.txt'

    Returns:
        - list (of lists of strings), e.g.:
          [['my', 'name', 'is', 'jolán'], ['i', 'am', 'cool'], ..., ['bye']]
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return [tuple(line.strip().split()) for line in lines]

def corpus_setup():
    return txt2list('grimm_full_commas.txt')

class FreqNode:
    def __init__(self, label):
        self.label = label
        self.children = {}
        self.count = 0
        self.context_count = 0
    
    def get_or_make_branch(self, tuple_of_strings, count_each_word=False):
        current_node = self
        for word in tuple_of_strings:
            current_node = current_node.get_or_make_child(word)
            if count_each_word:
                current_node.count += 1
            current_node.context_count += 1
        if not count_each_word:
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
        self.filler_finder = FreqTrie() # Left-to-right left contexts
        self.context_finder = FreqTrie() # Right-to-left left contexts

class DistrTrie:
    def __init__(self):
        self.root = DistrNode('~')
    
    def insert_distr(self, sentence):
        context_pairs = ((sentence[:i], sentence[i:]) for i in range(len(sentence)))
        for left_context, right_context in context_pairs:
            current_node = self.root
            for word in right_context:
                current_node = current_node.get_or_make_child(word)
                current_node.count += 1
                # Record right-to-left left context
                context_finder_node = current_node.context_finder.root
                context_finder_node.get_or_make_branch(tuple(reversed(left_context)), count_each_word=True)
                # Record all suffixes of left-to-right context
                filler_finder_node = current_node.filler_finder.root
                left_context_suffixes = (left_context[j:] for j in range(len(left_context)))
                for left_context_suffix in left_context_suffixes:
                    filler_finder_node.get_or_make_branch(left_context_suffix)