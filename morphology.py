from collections import defaultdict
from collections import Counter
import custom_io as cio

#-----------------------------#
# MorphModel class definition #
#-----------------------------#

class MorphModel():
    def __init__(self):
        self.tagtries = defaultdict(lambda: TagTrie())
        self.lemmas = defaultdict(Counter)

    def setup(self):
        morph_triples = cio.sztaki_tsv_noun_tag_wordform_lemma_import()
        for tag, word_form, lemma in morph_triples:
            self.tagtries[tag]._insert(word_form, lemma)
            self.lemmas[lemma][tag] += 1

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

#---------------------------#
# Analogy-finding functions #
#---------------------------#

def anl_bases(self, lemma, target_tag):
    encoded_lemma = cio.hun_encode(lemma)
    starting_tags = self.lemmas[encoded_lemma]
    shared_tag_dict = defaultdict(set)
    candidate_lemmas = self.tagtries[target_tag].root.lemmas.keys()
    for candidate_lemma in candidate_lemmas:
        for shared_tag in starting_tags & self.lemmas[candidate_lemma]:
            shared_tag_dict[shared_tag].add(candidate_lemma)
    return dict(shared_tag_dict)

def produce_word(model, lemma, target_tag):
    try:
        return (most_similar_bases(model, lemma, target_tag)[0][0])
    except:
        return ''

def produce_word_list(model, lemma, target_tag):
    try:
        return (most_similar_bases(model, lemma, target_tag))
    except:
        return ''

def most_similar_bases(self, lemma, target_tag):
    encoded_lemma = cio.hun_encode(lemma)
    tag_dict = {}
    bases = anl_bases(self, encoded_lemma, target_tag)
    new_wordforms = defaultdict(float)
    for tag, anl_lemmas in bases.items():
        transform_dict = defaultdict(Counter)
        tag_trie = self.tagtries[tag]
        lemma_wordform = wordform(self, encoded_lemma, tag)
        for anl_lemma in anl_lemmas:
            # Record all common suffixes for tagged wordforms of lemma and anl_lemma
            common_suffixes = set()
            common_suffix = ''
            anl_lemma_node = None
            current_node = tag_trie.root
            while anl_lemma in current_node.lemmas and encoded_lemma in current_node.lemmas:
                common_suffix = current_node.label + common_suffix
                common_suffixes.add(common_suffix)
                anl_lemma_node = current_node
                current_node = current_node.lemmas[encoded_lemma]
            # Find tagged wordform of anl_lemma
            anl_lemma_wordform = common_suffix
            while anl_lemma in anl_lemma_node.lemmas:
                anl_lemma_node = anl_lemma_node.lemmas[anl_lemma]
                anl_lemma_wordform = anl_lemma_node.label + anl_lemma_wordform
            # Find wordform of anl_lemma for target tag
            target_anl_lemma_wordform = wordform(self, anl_lemma, target_tag)
            # For each common suffix, find and record transformation pair
            for original_suffix in common_suffixes:
                anl_prefix = anl_lemma_wordform.removesuffix(original_suffix)
                if not target_anl_lemma_wordform.startswith(anl_prefix):
                    continue
                new_suffix = target_anl_lemma_wordform.removeprefix(anl_prefix)
                transform_dict[original_suffix][new_suffix] += 1
        for original_suffix, new_suffixes in transform_dict.items():
            stem = lemma_wordform.removesuffix(original_suffix)
            # ... calculate Simpson diversity for new_suffixes ...
            new_suffix_count = len(new_suffixes)
            simpson_index = sum((n / new_suffix_count) ** 2 for n in new_suffixes.values())
            for new_suffix, freq in new_suffixes.items():
                new_wordform = stem + new_suffix
                weight = (freq / new_suffix_count)# * simpson_index
                new_wordforms[new_wordform] += weight
    return sorted(new_wordforms.items(), key=lambda x: x[1], reverse=True)

def wordform(model, lemma, target_tag):
    word = ''
    tag_trie = model.tagtries[target_tag]
    current_node = tag_trie.root
    while lemma in current_node.lemmas:
        word = current_node.label + word
        current_node = current_node.lemmas[lemma]
    word = current_node.label + word
    return word

def testing(model, test_corpus):
    novel_results = {True: set(), False: set()}
    for lemma, tag, word_form in test_corpus:
        lemmafreq = sum(model.lemmas[lemma].values())
        is_novel = tag not in model.lemmas[lemma]
        if (lemma not in model.lemmas) or (not is_novel):
            continue
        produced_word = produce_word(model, lemma, tag)
        guess = produced_word == word_form
        novel_results[guess].add((word_form, produced_word, lemmafreq, tuple(tag)))
    return novel_results