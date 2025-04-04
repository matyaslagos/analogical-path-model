from itertools import product
from collections import defaultdict
from string import punctuation
import random
import math
import csv

# Setup functions

def txt2wordlist(filename):
    """Import filename as a list of words (tuples of characters).
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    clean_word = lambda s: s.strip(punctuation).lower()
    return [
        ('<',) + tuple(word) + ('>',)
        for line in lines
        for word in map(clean_word, line.split())
        if word.isalpha()
    ]

def csv2wordfreqdict(filename):
    """Import filename as a dict of words (tuples of characters) with int values.
    """
    with open(filename, newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        clean_word = lambda s: s.strip(punctuation).lower()
        return {
            ('<',) + tuple(clean_word(row['key'])) + ('>',): int(row['value'])
            for row in reader}

def train_test_split(corpus):
    corpus_copy = corpus[:]
    random.shuffle(corpus_copy)
    split_point = int(0.9 * len(corpus_copy))
    return corpus_copy[:split_point], corpus_copy[split_point:]

def corpus_setup():
    return txt2wordlist('sztaki_corpus.txt')

def distrtrie_setup(sequence_list):
    ddy = FreqTrie()
    for sequence in sequence_list:
        ddy.insert_distr(sequence)
    return ddy

def distrtrie_setup_freq(sequence_freq_dict):
    ddy = FreqTrie()
    most_freq_items = sorted(
        sequence_freq_dict.items(),
        key=lambda x: x[1],
        reverse=True
        )[:50000]
    for sequence, freq in most_freq_items:
        ddy.insert_distr(sequence, freq)
    return ddy

def lc(word):
    return ('<',) + tuple(word) + ('_',)

def rc(word):
    return ('_',) + tuple(word) + ('>',)

# Trie class for recording distribution information about corpus

class FreqNode:
    def __init__(self):
        self.children = {}
        self.count = 0
        self.context_count = 0
    
    def get_or_make_branch(self, iterator_of_strings, freq=1):
        current_node = self
        for token in iterator_of_strings:
            current_node = current_node.get_or_make_child(token)
            current_node.context_count += freq
        current_node.count += freq
        return current_node
    
    def get_or_make_child(self, child_label):
        if child_label not in self.children:
            self.children[child_label] = FreqNode()
        return self.children[child_label]

class FreqTrie:
    def __init__(self):
        self.fw_root = FreqNode()
        self.bw_root = FreqNode()
    
    # Record distribution information about a sentence
    def insert_distr(self, sentence, freq=1):
        """Record all contexts and fillers of `sentence` into trie.
        
        Arguments:
            - sentence (tuple of strings): e.g. ('the', 'king', 'was', 'here')
        
        Effect:
            For each prefix--suffix pair of `sentence`, records the suffix as
            a branch, and for each word in this branch, records all suffixes
            of the prefix as branches.
            The resulting trie structure can be used to look up shared fillers
            of two contexts or shared contexts of two fillers.
            In the former case:
            - main branches act as fillers and right contexts, and
            - finder branches act as left contexts.
            In the latter case:
            - main branches act as right contexts, and
            - finder branches act as left contexts and fillers.
        """
        pref_suff_pairs = (
            (sentence[:i], sentence[i:])
            for i in range(len(sentence) + 1)
        )
        for prefix, suffix in pref_suff_pairs:
            self.fw_root.get_or_make_branch(suffix, freq)
            self.bw_root.get_or_make_branch(reversed(prefix), freq)
    
    def get_context_node(self, context):
        if context[-1] == '_':
            current_node = self.fw_root
            context_iterator = context[:-1]
        else:
            current_node = self.bw_root
            context_iterator = reversed(context[1:])
        for word in context_iterator:
            try:
                current_node = current_node.children[word]
            except KeyError:
                return
        return current_node
    
    # Yield each shared filler of two contexts TODO: not str, tup
    def shared_fillers(self, context1, context2):
        """Yield each shared filler of `context1` and `context2`.
        
        Arguments:
            - context1 (string): e.g. 'a _ garden'
            - context2 (string): e.g. 'this _ lake'
        
        Returns:
            - generator (of strings): e.g. ('beautiful', 'very nice', ...)
        """
        context_node1 = self.get_context_node(context1)
        context_node2 = self.get_context_node(context2)
        direction = context1.index('_')
        return self.get_shared_branches(context_node1, context_node2, direction)
  
    # Recursively yield each shared filler of two context nodes
    def get_shared_branches(self, distr_node1, distr_node2, direction, path=[]):
        """Yield each shared branch of `distr_node1` and `distr_node2`.
    
        Arguments:
            - distr_node1 (DistrNode): root of a subtrie
            - distr_node2 (DistrNode): root of another subtrie
    
        Yields:
            - string: branch that is shared by the two input subtries
        """
        for child in distr_node1.children:
            if child in distr_node2.children:
                new_path = path + [child]
                child_node1 = distr_node1.children[child]
                child_node2 = distr_node2.children[child]
                if child_node1.count > 0 and child_node2.count > 0:
                    freq1 = child_node1.count
                    freq2 = child_node2.count
                    form = tuple(new_path) if direction else tuple(reversed(new_path))
                    yield (form, freq1, freq2)
                yield from self.get_shared_branches(child_node1, child_node2, direction, new_path)
    
    # From here on: contexts and fillers are tup
    def get_fillers(self, context, max_length=float('inf')):
        context_node = self.get_context_node(context)
        direction = context.index('_')
        return self.get_branches(context_node, direction, max_length)
    
    def get_branches(self, current_node, direction, max_length=float('inf'), path=[]):
        if len(path) >= max_length:
            return
        for child in current_node.children:
            new_path = path + [child]
            child_node = current_node.children[child]
            if child_node.count > 0:
                branch = tuple(new_path) if direction else tuple(reversed(new_path))
                freq = child_node.count
                yield (branch, freq)
            yield from self.get_branches(child_node, direction, max_length, new_path)

def get_freq(self, context):
    if context[-1] == '_':
        current_node = self.bw_root
        context_iterator = reversed(context[:-1])
    else:
        current_node = self.fw_root
        context_iterator = context[1:]
    for word in context_iterator:
        try:
            current_node = current_node.children[word]
        except KeyError:
            return 0
    return current_node.count

def get_fillers_func(self, context, max_length=float('inf')):
    context_node = self.get_context_node(context)
    direction = context.index('_')
    return get_branches_func(self, context_node, direction, max_length)

def get_branches_func(self, current_node, direction, max_length=float('inf'), path=[]):
    if len(path) >= max_length:
        return
    for child in current_node.children:
        new_path = path + [child]
        child_node = current_node.children[child]
        if child_node.count > 0:
            branch = tuple(new_path) if direction else tuple(reversed(new_path))
            branch = ('_',) + branch if direction else branch + ('_',)
            freq = child_node.count
            if branch not in {('_', '>'), ('<', '_')}:
                yield (branch, freq)
        yield from get_branches_func(self, child_node, direction, max_length, new_path)

# Yield each shared filler of two contexts TODO: not str, tup
def get_shared_fillers_func(self, context1, context2):
    """Yield each shared filler of `context1` and `context2`.
    
    Arguments:
        - context1 (string): e.g. 'a _ garden'
        - context2 (string): e.g. 'this _ lake'
    
    Returns:
        - generator (of strings): e.g. ('beautiful', 'very nice', ...)
    """
    context_node1 = self.get_context_node(context1)
    context_node2 = self.get_context_node(context2)
    direction = context1.index('_')
    return get_shared_branches_func(self, context_node1, context_node2, direction)

# Recursively yield each shared filler of two context nodes
def get_shared_branches_func(self, distr_node1, distr_node2, direction, path=[]):
    """Yield each shared branch of `distr_node1` and `distr_node2`.

    Arguments:
        - distr_node1 (DistrNode): root of a subtrie
        - distr_node2 (DistrNode): root of another subtrie

    Yields:
        - string: branch that is shared by the two input subtries
    """
    for child in distr_node1.children:
        if child in distr_node2.children:
            new_path = path + [child]
            child_node1 = distr_node1.children[child]
            child_node2 = distr_node2.children[child]
            if child_node1.count > 0 and child_node2.count > 0:
                freq1 = child_node1.count
                freq2 = child_node2.count
                form = tuple(new_path) if direction else tuple(reversed(new_path))
                form = ('_',) + form if direction else form + ('_',)
                yield (form, freq1, freq2)
            yield from get_shared_branches_func(self, child_node1, child_node2, direction, new_path)

def anl_contexts_func(self, context, filler):
    anl_path_infos = defaultdict(float)
    for anl_context, anl_context_filler_freq in get_fillers_func(self, filler):
        if anl_context == context:
            continue
        anl_context_pred, anl_context_data = context_pred_func(self, anl_context, context, filler)
        anl_context_freq = get_freq(self, anl_context)
        filler_cond_prob = anl_context_filler_freq / anl_context_freq
        anl_prob = anl_context_pred * filler_cond_prob
        anl_path_infos[''.join(anl_context)] += anl_prob
    return sorted(
        anl_path_infos.items(),
        key=lambda x: x[1],
        reverse=True
    )

def anl_substs(self, left, right):
    anl_bridge_dict = {}
    left_self_subst = 0
    right_self_subst = 0
    left_freq = get_freq(self, left)
    right_freq = get_freq(self, right)
    left_substs_dict = defaultdict(float)
    right_substs_dict = defaultdict(float)
    subst_grams_dict = {}
    # Calculate P(left || left)
    anl_rights = list(get_fillers_func(self, left))
    for anl_right, left_anl_right_freq in anl_rights:
        anl_right_freq = get_freq(self, anl_right)
        fw_prob = left_anl_right_freq / left_freq
        bw_prob = left_anl_right_freq / anl_right_freq
        left_self_subst += fw_prob * bw_prob
    left_substs_dict[left] = left_self_subst
    # Calculate P(right || right)
    anl_lefts = get_fillers_func(self, right)
    for anl_left, anl_left_right_freq in anl_lefts:
        anl_left_freq = get_freq(self, anl_left)
        fw_prob = anl_left_right_freq / right_freq
        bw_prob = anl_left_right_freq / anl_left_freq
        right_self_subst += fw_prob * bw_prob
    right_substs_dict[right] = right_self_subst
    # Calculate P(right || anl_right) and P(left || anl_left)
    # for each (anl_left, anl_right) bridge where anl_dir != dir
    for anl_right, left_anl_right_freq in anl_rights:
        anl_right_freq = get_freq(self, anl_right)
        top_gram_fw_prob = left_anl_right_freq / left_freq
        top_gram_bw_prob = left_anl_right_freq / anl_right_freq
        anl_lefts = get_shared_fillers_func(self, anl_right, right)
        for anl_left, anl_left_anl_right_freq, anl_left_right in anl_lefts:
            anl_left_freq = get_freq(self, anl_left)
            middle_gram_fw_prob = anl_left_anl_right_freq / anl_left_freq
            middle_gram_bw_prob = anl_left_anl_right_freq / anl_right_freq
            bottom_gram_fw_prob = anl_left_right_freq / left_freq
            bottom_gram_bw_prob = anl_left_right_freq / anl_right_freq
            left_entropy = entropy_func(self, anl_left)
            right_entropy = entropy_func(self, anl_right)
            anl_left_prob = middle_gram_fw_prob * top_gram_bw_prob * left_entropy
            anl_right_prob = middle_gram_bw_prob * bottom_gram_fw_prob * right_entropy
            left_substs_dict[anl_left] += anl_left_prob
            right_substs_dict[anl_right] += anl_right_prob
            gramify = lambda x: lambda y: context_filler_tuple(self, x, y)
            if (left, anl_right) not in anl_bridge_dict:
                top_gram = gramify(left)(anl_right)
                anl_bridge_dict[(left, anl_right)] = get_freq(self, top_gram)
            middle_gram = gramify(anl_left)(anl_right)
            bottom_gram = gramify(anl_left)(right)
            anl_bridge_dict[(anl_left, anl_right)] = get_freq(self, middle_gram)
            anl_bridge_dict[(anl_left, right)] = get_freq(self, bottom_gram)
    # Calculate support for each available analogical gram
    for anl_left, anl_right in anl_bridge_dict:
        anl_subst = left_substs_dict[anl_left] * right_substs_dict[anl_right]
        anl_support = anl_subst * anl_bridge_dict[(anl_left, anl_right)]
        anl_gram = context_filler_merge(self, anl_left, anl_right)
        subst_grams_dict[anl_gram] = anl_support
    return sorted(subst_grams_dict.items(), key=lambda x: x[1], reverse=True)

def subst_contexts_func(self, context, filler):
    anl_context_dict = defaultdict(float)
    context_freq = get_freq(self, context)
    for anl_filler, context_anl_filler_freq in get_fillers_func(self, context):
        anl_filler_freq = get_freq(self, anl_filler)
        if anl_filler == filler:
            pass
        filler_subst_prob, anl_context_data = context_subst_func(self, anl_filler, filler)
        anl_filler_cond_prob = context_anl_filler_freq / context_freq
        for anl_context, anl_context_subst_prob in anl_context_data:
            entropy_weight = entropy_func(self, anl_filler) * entropy_func(self, anl_context)
            anl_prob = anl_filler_cond_prob * anl_context_subst_prob * entropy_weight
            anl_word = context_filler_merge(self, anl_context, anl_filler)
            anl_context_dict[anl_word] += anl_prob
    return sorted(
        anl_context_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

def anl_words_func(self, context, filler):
    anl_path_infos = defaultdict(float)
    for anl_context, anl_context_filler_freq in get_fillers_func(self, filler):
        if anl_context == context:
            continue
        anl_context_pred, anl_filler_data = context_pred_func(self, anl_context, context, filler)
        anl_context_freq = get_freq(self, anl_context)
        filler_cond_prob = anl_context_filler_freq / anl_context_freq
        anl_prob = anl_context_pred * filler_cond_prob
        for anl_filler, anl_filler_value in anl_filler_data:
            gram = context_filler_merge(self, anl_context, anl_filler)
            anl_path_infos[gram] += anl_filler_value * filler_cond_prob * len(anl_filler_data)
    return sorted(
        anl_path_infos.items(),
        key=lambda x: x[1],
        reverse=True
    )

def context_filler_tuple(self, context, filler):
    if context.index('_'):
        return context[:-1] + filler[1:] + ('_',)
    else:
        return filler[:-1] + context[1:] + ('_',)

def context_filler_merge(self, context, filler):
    if context.index('_'):
        return ''.join(context[:-1]) + ' + ' + ''.join(filler[1:])
    else:
        return ''.join(filler[:-1]) + ' + ' + ''.join(context[1:])

def typefreq_func(self, context):
    return len(list(self.get_fillers(context)))

def entropy_func(self, context):
    entropy = 0
    context_node = self.get_context_node(context)
    for child in context_node.children:
        child_node = context_node.children[child]
        cond_prob = child_node.context_count / context_node.context_count
        entropy += cond_prob * math.log(1 / cond_prob, 2)
    return entropy

def context_subst_func(self, anl_context, context, filler=None):
    pred_dict = defaultdict(float)
    anl_cxt_freq = get_freq(self, anl_context)
    shared_fillers = get_shared_fillers_func(self, anl_context, context)
    for shared_filler, anl_cxt_flr_freq, org_cxt_flr_freq in shared_fillers:
        if filler == shared_filler:
            continue
        flr_freq = get_freq(self, shared_filler)
        anl_cxt_cond_prob = anl_cxt_flr_freq / anl_cxt_freq
        org_cxt_cond_prob = org_cxt_flr_freq / flr_freq
        pred_dict[shared_filler] += anl_cxt_cond_prob * org_cxt_cond_prob
    return sum(pred_dict.values()), pred_dict.items()
        

def predictors_func(self, context):
    direction = context.index('_')
    predictor_dict = defaultdict(lambda: defaultdict(float))
    context_freq = get_freq(self, context)
    fillers = self.get_fillers(context)
    for filler, context_filler_freq in fillers:
        filler = ('_',) + filler if direction else filler + ('_',)
        filler_freq = get_freq(self, filler)
        # Calculate probability of moving from original context to
        # analogical filler
        context_filler_prob = context_filler_freq / filler_freq
        # Loop over all shared contexts of analogical filler and filler
        # to find analogical contexts
        anl_contexts = self.get_fillers(filler)
        for anl_context, anl_context_filler_freq in anl_contexts:
            anl_context = anl_context + ('_',) if direction else ('_',) + anl_context
            anl_context_freq = get_freq(self, anl_context)
            # Calculate weight of moving from analogical filler to
            # analogical context and then from analogical context to filler
            if 0 in {filler_freq, anl_context_freq}:
                continue
            anl_context_filler_prob = anl_context_filler_freq / anl_context_freq
            anl_path_prob = context_filler_prob * anl_context_filler_prob
            predictor_dict[''.join(anl_context)][''.join(filler)] += anl_path_prob
    predictor_list = []
    for anl_context in predictor_dict:
        predictor_list.append(
            (
            anl_context,
            sorted(predictor_dict[anl_context].items(), key=lambda x: x[1], reverse=True)
            )
        )
    return sorted(
        predictor_list,
        key=lambda x: sum(y[1] * len(x[1]) for y in x[1]),
        reverse=True
    )

def iter_anls(self, word):
    context_filler_pairs = (
        (('_',) + word[i:] + ('>',), ('<',) + word[:i])
        for i in range(1, len(word))
    )
    anl_dict = {}
    for context, filler in context_filler_pairs:
        pass