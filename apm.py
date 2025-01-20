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
        - filename (string): e.g. 'grimm_corpus.txt'

    Returns:
        - list (of list of strings): e.g.
          [['my', 'name', 'is', 'jolán'], ['i', 'am', 'cool'], ..., ['bye']]
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return [tuple(line.strip().split()) for line in lines]

def corpus_setup():
    return txt2list('grimm_full_commas.txt')

def distrtrie_setup(corpus):
    ddy = DistrTrie()
    for sentence in corpus:
        ddy.insert_distr(sentence)
    return ddy

class FreqNode:
    def __init__(self, label):
        self.label = label
        self.children = {}
        self.count = 0
        self.context_count = 0
    
    def get_or_make_branch(self, tuple_of_strings, count_each_word=False):
        current_node = self
        for word in tuple_of_strings:
            current_node.context_count += 1
            current_node = current_node.get_or_make_child(word)
            if count_each_word:
                current_node.count += 1
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
    
    # Record distribution information about a sentence
    def insert_distr(self, sentence):
        context_pairs = ((sentence[:i], sentence[i:]) for i in range(len(sentence)))
        for left_context, right_context in context_pairs:
            current_node = self.root
            for word in right_context:
                current_node = current_node.get_or_make_child(word)
                current_node.count += 1
                """
                # Record right-to-left left context
                context_finder_node = current_node.context_finder.root
                context_finder_node.get_or_make_branch(tuple(reversed(left_context)), count_each_word=True)
                """
                # Record all suffixes of left-to-right context
                filler_finder_node = current_node.filler_finder.root
                left_context_suffixes = (left_context[j:] for j in range(len(left_context)))
                for left_context_suffix in left_context_suffixes:
                    filler_finder_node.get_or_make_branch(left_context_suffix)
    
    # Yield all shared fillers of two contexts
    def shared_fillers(self, context1, context2):
        """Yield all shared fillers of `context1` and `context2`.
        
        Arguments:
            - context1 (string): e.g. 'a _ garden'
            - context2 (string): e.g. 'this _ lake'
        
        Returns:
            - generator (of strings): e.g. ('beautiful', 'very nice', ...)
        """
        context_node1 = self.get_context_node(context1)
        context_node2 = self.get_context_node(context2)
        return self.shared_fillers_aux(context_node1, context_node2)
    
    # Return the filler finder trie of a context
    def get_context_node(self, context):
        """Return the filler finder trie of `context`.
        
        Argument:
            - context (tuple of strings): e.g. ('a', '_', 'king')
        
        Returns:
            - DistrNode: the node that is the root of the trie of the fillers
              that occurred in `context`
        """
        left_context, right_context = map(lambda x: x.strip().split(), context.split('_'))
        context_node = self.root
        for i, word in enumerate(right_context):
            try:
                context_node = context_node.children[word]
            except KeyError:
                failed_part = '_ ' + ' '.join(right_context[:i+1]) + ' ...'
                raise KeyError(
                    f'Context \"{context}\" not found (failed at \"{failed_part}\")'
                    )
        context_node = context_node.filler_finder.root
        for i, word in enumerate(left_context):
            try:
                context_node = context_node.children[word]
            except KeyError:
                failed_part = ' '.join(left_context[:i+1]) + ' ... _ ' + ' '.join(right_context)
                raise KeyError(
                    f'Context \"{context}\" not found (failed at \"{failed_part}\")'
                    )
        return context_node
    
    # Recursively find all shared fillers from two context nodes
    def shared_fillers_aux(self, context_node1, context_node2, path=[]):
        """Yield all shared fillers of `context_node1` and `context_node2`.
        
        Arguments:
            - context_node1 (DistrNode): filler trie of a context
            - context_node2 (DistrNode): filler trie of another context
        
        Yields:
            - string: filler that is shared by the two input contexts
        """
        for child in context_node1.children:
            if child in context_node2.children:
                new_path = path + [child]
                child_node1 = context_node1.children[child]
                child_node2 = context_node2.children[child]
                if child_node1.count * child_node2.count > 0:
                    yield ' '.join(new_path)
                yield from self.shared_fillers_aux(child_node1, child_node2, new_path)
        
    # Generator of all shared contexts of two fillers
    def shared_contexts(self, filler1, filler2):
        pass