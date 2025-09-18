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

def setup():
    morph_triples = custom_io.sztaki_tsv_noun_tag_wordform_lemma_import()
    tag_tries = defaultdict(lambda: TagTrie())
    lemmas = defaultdict(set)
    for tag, word_form, lemma in morph_triples:
        tag_tries[tag]._insert(word_form, lemma)
        lemmas[lemma].add(tag)
    return (dict(tag_tries), dict(lemmas))

#-----------------------------#
# MorphModel class definition #
#-----------------------------#

class MorphModel():
    def __init__(self):
        self.tagtries = defaultdict(lambda: TagTrie())
        self.lemmas = defaultdict(set)

    def setup(self):
        morph_triples = custom_io.sztaki_tsv_noun_tag_wordform_lemma_import()
        for tag, word_form, lemma in morph_triples:
            self.tagtries[tag]._insert(word_form, lemma)
            self.lemmas[lemma].add(tag)

def anl_bases(self, lemma, target_tag):
    starting_tags = self.lemmas[lemma]
    shared_tag_dict = defaultdict(set)
    candidate_lemmas = self.tagtries[target_tag].root.lemmas.keys()
    for candidate_lemma in candidate_lemmas:
        for shared_tag in starting_tags & self.lemmas[candidate_lemma]:
            shared_tag_dict[shared_tag].add(candidate_lemma)
    return dict(shared_tag_dict)

def most_similar_bases(self, lemma, target_tag):
    tag_dict = {}
    bases = anl_bases(self, lemma, target_tag)
    for tag, anl_lemmas in bases.items():
        anl_lemma_dict = {}
        tag_trie = self.tagtries[tag]
        for anl_lemma in anl_lemmas:
            common_suffix = ''
            current_node = tag_trie.root
            while anl_lemma in current_node.lemmas:
                common_suffix = current_node.label + common_suffix
                current_node = current_node.lemmas[lemma]
            anl_lemma_dict[anl_lemma] = common_suffix
        tag_dict[tag] = anl_lemma_dict
    return tag_dict

#--------------------------#
# TagTrie class definition #
#--------------------------#

class TagNode:
    def __init__(self, char):
        self.children = {}
        self.freq = 0
        self.lemmas = {}
        self.label = char
    
    def _increment_or_make_branch(self, word_form, lemma):
        """Record branch of word_form and lemma.
        """
        current_node = self
        current_node.freq += 1
        for char in reversed(word_form):
            current_node = current_node._get_or_make_child(char, lemma)
            current_node.freq += 1
    
    def _get_or_make_child(self, char, lemma):
        """Return child labeled by char or make new child.
        """
        if char not in self.children:
            self.children[char] = TagNode(char)
        if lemma not in self.lemmas:
            self.lemmas[lemma] = self.children[char]
        return self.children[char]

class TagTrie:
    def __init__(self):
        self.root = TagNode('')

    def _insert(self, word_form, lemma):
        self.root._increment_or_make_branch(word_form, lemma)
