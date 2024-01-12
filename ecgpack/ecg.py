# =====
# The analogical path model for computing well-formedness analyses for n-grams
# using recursive distributional analogies.
# =====

import math
import random
from itertools import product
import numpy as np
from numpy.linalg import norm
#from alive_progress import alive_bar; import time

# --------------------- #
# Setting up the corpus #
# --------------------- #

# Import a txt file (containing one sentence per line) as a list whose each
# element is a list of words corresponding to a line in the txt file:
def txt2list(filename):
    """Import a txt list of sentences as a list of lists of words.
    
    Keyword arguments:
    filename -- name of a txt file, containing one normalised sentence per line,
        with no sentence-ending periods
    
    Returns:
    list -- list of lists of words, e.g.
        [['my', 'name', 'is', 'mary'], ['i', 'am', 'cool'], ..., ['bye']]
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return [line.strip().split() for line in lines]

# Randomly sort a list of sentences into 90% training and 10% testing data:
def train_test_split(corpus):
    """Randomly separate `sentences` into 90pct training and 10pct testing data.
    
    Randomly separates `sentences` into 90 percent training data and 10 percent
    testing data, such that the testing data is filtered to only contain 
    sentences whose words all occur in the training data (so actual testing data
    may be smaller than 10 percent).
    
    Keyword arguments:
    sentences -- list of lists of words (output of `txt_to_list()`)
    
    Returns:
    tuple -- (train, test), where `train` is randomly selected 90pct of
        `sentences`, and `test` is remaining 10pct filtered s.t. none of its
        sentences contain words that aren't in `train`
    """
    sentences = corpus.copy()
    random.shuffle(sentences)
    n = round(len(sentences) * 0.9)
    train = sentences[:n]
    vocab = {word for sentence in train for word in sentence}
    test = [sentence for sentence in sentences[n:]
                     if set(sentence).issubset(vocab)]
    return (train, test)

def tenfold_splits(corpus):
    sentences = corpus.copy()
    random.shuffle(sentences)
    e = round(len(sentences) * 0.1)
    trains = []
    tests = []
    for i in range(10):
        train = sentences[:i*e] + sentences[(i+1)*e:]
        tr_vocab = {word for sentence in train for word in sentence}
        test = [sentence for sentence in sentences[i*e:(i+1)*e]
                         if set(sentence).issubset(tr_vocab)]
        trains.append(train)
        tests.append(test)
    return (trains, tests)

def tenfold_splits_50(corpus):
    sentences = corpus.copy()
    random.shuffle(sentences)
    sentences = sentences[:len(sentences)//2]
    return tenfold_splits(sentences)

def tenfold_splits_25(corpus):
    sentences = corpus.copy()
    random.shuffle(sentences)
    sentences = sentences[:len(sentences)//4]
    return tenfold_splits(sentences)

# ----------------------------- #
# General analogical path model #
# ----------------------------- #

# -----
# Training the general analogical path model
# -----

# Build n-gram analogical path model from training data
# (train is a list of lists of words, n is order of recorded ngrams):
def ap_model(train, n):
    # Compute context dictionary along with total n-gram frequencies
    cdy, freqs = gctxt(train, n)
    pdy = {}
    # For each context and goal, calculate joint, forw., and backw. prob.s
    # (i.e. P(context goal), P(goal | context _ ), P(context | _ goal)):
    for context in cdy['f']:
        # Get frequency of context (for forwards tr. prob.):
        curr_cxt = cdy['f'][context]
        cxt_freq = sum(curr_cxt[gl] for gl in curr_cxt if len(gl) == 1)
        for goal in curr_cxt:
            # Get frequency of goal (for backwards tr. prob.):
            curr_goal = cdy['b'][goal]
            goal_freq = sum(curr_goal[ct] for ct in curr_goal if len(ct) == 1)
            # Count cooccurrence of context and goal, record tr. prob.s:
            count = curr_cxt[goal]
            joint = count / freqs[len(context) + len(goal)]
            forw = count / cxt_freq
            backw = count / goal_freq
            pdy[(context, goal)] = {'j': joint, 'f': forw, 'b': backw}
    return (cdy, pdy)

# [aux for ap_model()] Create n-gram context dictionary of train:
def gctxt(train, n):
    freqs = {i: 0 for i in range(2, n+1)}
    cdy = {'f': {}, 'b': {}}
    for sentence in train:
        grams = []
        for i in range(2, n+1):
            curr = ngrams(sentence, i)
            freqs[i] += len(curr)
            grams += curr
        # Record cooccurrence data in cdy with forwards and backward contexts:
        for gram in grams:
            # First n-1 words are forward context, nth word is goal:
            try:
                cdy['f'][gram[:-1]][gram[-1:]] += 1
            except:
                try:
                    cdy['f'][gram[:-1]][gram[-1:]] = 1
                except:
                    cdy['f'][gram[:-1]] = {gram[-1:]: 1}
            # First word is forward context, last n-1 words are goal
            # (except for bigrams, b/c would override previous):
            if len(gram) > 2:
                try:
                    cdy['f'][gram[:1]][gram[1:]] += 1
                except:
                    try:
                        cdy['f'][gram[:1]][gram[1:]] = 1
                    except:
                        cdy['f'][gram[:1]] = {gram[1:]: 1}
            # Last n-1 words are backward context, first word is goal:
            try:
                cdy['b'][gram[1:]][gram[:1]] += 1
            except:
                try:
                    cdy['b'][gram[1:]][gram[:1]] = 1
                except:
                    cdy['b'][gram[1:]] = {gram[:1]: 1}
            # Last word is backward context, first n-1 words are goal
            # (except for bigrams, b/c would override previous):
            if len(gram) > 2:
                try:
                    cdy['b'][gram[-1:]][gram[:-1]] += 1
                except:
                    try:
                        cdy['b'][gram[-1:]][gram[:-1]] = 1
                    except:
                        cdy['b'][gram[-1:]] = {gram[:-1]: 1}
    return (cdy, freqs)

# [aux for gctxt()] Return list of n-grams in sentence:
def ngrams(sentence, n):
    sentence = ['<s>'] * (n-1) + sentence + ['</s>'] * (n-1)
    return list(zip(*(sentence[i:len(sentence)-n+i+1] for i in range(n))))

# -----
# Finding analogical paths for a pair of n-grams
# -----

# Find analogical paths for a duplet (a pair of n-grams) in model:
def anl_paths(duplet, model):
    d1, d2 = duplet
    cdy, pdy = model
    # Get attested duplets:
    attested = pdy.keys()
    # Get 50 most frequent forw. neighbours of d1 and backw. neighbours of d2:
    d1_forw = (x for x in cdy['f'][d1].keys() if len(x) <= len(d2))
    d2_backw = (x for x in cdy['b'][d2].keys() if len(x) <= len(d1))
    d1_forw = sorted(d1_forw, reverse=True,
                     key=lambda x: sum(cdy['b'][x].values()))[:50]
    d2_backw = sorted(d2_backw, reverse=True,
                      key=lambda x: sum(cdy['f'][x].values()))[:50]
    # Get potential paths between d1 and d2, and then attested paths:
    potential_paths = product(set(d2_backw), set(d1_forw))
    paths = set(attested).intersection(set(potential_paths))
    # Compute weight of each path and record as analogies:
    analogies = []
    for p1, p2 in paths:
        gen_swap = pdy[(p1, p2)]['j'] * pdy[(d1, p2)]['b'] * pdy[(p1, d2)]['f']
        cos = cosine_sim((d1, d2), (p1, p2), model)
        analogy = ((p1, p2), gen_swap * cos)
        analogies.append(analogy)
    analogies.sort(reverse=True, key=lambda x: x[1])
    return (sum(analogy[1] for analogy in analogies), analogies)

# [aux for anl_paths()] Return the product of the cosine similarities
# d1~p1 and d2~p2 (where duplet is (d1, d2), path is (p1, p2)):
def cosine_sim(duplet, path, model):
    # TODO: update docstring, consider different probs for unigrams and bigrams
    """Return the product of the cosine similarities for path for duplet.
    
    Computes the word-class similarities of `s` and `a`, and of `t` and `b`;
    the former is given by `sum_x: P(x|s_)*P(x|a_)` and the latter by
    `sum_y: P(y|_t)*P(y|_b)`. Returns the product of these two values.
    
    Keyword arguments:
    s, t, a, b -- strings, such that we're investigating the analogical path
        from `s` to `t` through `b` and `a`
    model -- `(pdy, cdy)` where `pdy` is a probability dictionary (output of
        `prob_dy()`) and `cdy` is a context dictionary (output of `ctxt_dy()`)
    
    Returns:
    left_sim * right_sim -- where `left_sim` is the word-class similarity of
        `s` and `a`, given by `sum_x: P(x|s_)*P(x|a_)`, and `right_sim` is the
        word-class similarity of `t` and `b`, given by `sum_y: P(y|_t)*P(y|_b)`
    """
    d1, d2 = duplet
    p1, p2 = path
    cdy, pdy = model
    
    # Appropriate-length common forw. neighbours of d1 and p1:
    d1_forw = set(x for x in cdy['f'][d1] if len(x) <= len(p2))
    p1_forw = set(x for x in cdy['f'][p1] if len(x) <= len(p2))
    forw_common = d1_forw.intersection(p1_forw)
    # Arrays of P(x|d1_) and P(x|p1_) for all common x:
    d1_comm = np.array([pdy[(d1, x)]['f'] for x in forw_common])
    p1_comm = np.array([pdy[(p1, x)]['f'] for x in forw_common])
    # Arrays of P(d1,x) and P(p1,y) for all attested (d1, x) and (p1, y):
    d1_full = np.array([pdy[(d1, x)]['j'] for x in d1_forw])
    p1_full = np.array([pdy[(p1, x)]['j'] for x in p1_forw])
    # Forward transitional cosine similarity of d1 and p1:
    left_sim = np.dot(d1_comm, p1_comm) / (norm(d1_full) * norm(p1_full))
    
    # Appropriate-length common backw. neighbours of d2 and p2:
    d2_backw = set(x for x in cdy['b'][d2] if len(x) <= len(p1))
    p2_backw = set(x for x in cdy['b'][p2] if len(x) <= len(p1))
    backw_common = d2_backw.intersection(p2_backw)
    # Arrays of P(x|_d2) and P(x|_p2) for all common x:
    d2_comm = np.array([pdy[(x, d2)]['b'] for x in backw_common])
    p2_comm = np.array([pdy[(x, p2)]['b'] for x in backw_common])
    # Arrays of P(x,d2) and P(y,p2) for all attested (x, d2) and (y, p2):
    d2_full = np.array([pdy[(x, d2)]['j'] for x in d2_backw])
    p2_full = np.array([pdy[(x, p2)]['j'] for x in p2_backw])
    # Backward transitional cosine similarity of d2 and p2:
    right_sim = np.dot(d2_comm, p2_comm) / (norm(d2_full) * norm(p2_full))
    
    return left_sim * right_sim

# -----
# Computing the analogical likelihood of an n-gram
# -----

# Recursively compute the analogical likelihood of ngram in model:
def ranl_sc(ngram, model, dyn={}):
    if len(ngram) == 1:
        # Base case is unigram, here the best analogical path is itself:
        return ([(ngram, ngram)], 1, dyn)
    else:
        # Check dynamic lookup dict and use it if possible:
        if ngram in dyn:
            # Return analogies, score, and unchanged lookup dict:
            return (dyn[ngram][0], dyn[ngram][1], dyn)
        else:
            # Collect all possible binary source-target splits (which I call
            # "duplets") of ngram;
            # a binary source-target split of ('hello', 'there', 'friend') is
            # e.g. (('hello',), ('there', 'friend')), in which case we're
            # looking for paths from ('hello',) to ('there', 'friend'):
            duplets = ((ngram[:i], ngram[i:]) for i in range(1, len(ngram)))
            paths = []
            score = 0
            for d1, d2 in duplets:
                # Recursively analyse source (d1) and target (d2) separately,
                # and record analyses (a1 and a2, resp.) in dynamic lookup dict:
                a1, a2 = (ranl_sc(d1, model, dyn)[:2],
                          ranl_sc(d2, model, dyn)[:2])
                dyn[d1], dyn[d2] = a1, a2
                # Find pairwise analogies between the members of a1 and a2:
                duplet_paths, duplet_score = pairw_anl(a1, a2, model)
                # Annotate each path with the current duplet (elements are:
                # given the duplet (d1, d2), we replaced d1 with path[1][0]
                # and d2 with path[1][1], and found the analogical path
                # path[0][0] with score path[0][1]):
                annotated_paths = [(path[0], path[1], (d1, d2))
                                   for path in duplet_paths]
                # Add the annotated paths and the score of this duplet to the
                # total paths and total score of the n-gram:
                paths += annotated_paths
                score += duplet_score
            # Sort the analogical paths according to their score (best first):
            paths.sort(reverse=True, key=lambda x: x[0][1])
            # Return the five best annotated paths, the total score of the
            # n-gram, and the dynamic lookup dict:
            return ([(path[0][0], path[1:]) for path in paths[:5]], score, dyn)

# [aux for ranl_sc()] Pairwise compute analogical paths between two sets of
# ngrams a1 and a2:
def pairw_anl(a1, a2, model):
    # Generate analogical source-target pairs:
    anl_pairs = product((x[0] for x in a1[0]), (x[0] for x in a2[0]))
    duplet_score = 0
    duplet_paths = []
    for pair in anl_pairs:
        b1 = tupleify(pair[0])
        b2 = tupleify(pair[1])
        # For trigram chunking:
        if len(b1) > 2 or len(b2) > 2:
            continue
        # Elems of analogies are ((p1, p2), score):
        curr_score, analogies = anl_paths((b1, b2), model)
        # Annotate analogies with current source-target duplet:
        annotated_analogies = [(anl, (b1, b2)) for anl in analogies]
        duplet_paths += annotated_analogies
        duplet_score += curr_score
    duplet_score *= math.sqrt(a1[1]*a2[1])
    return (duplet_paths, duplet_score)

# [aux for pairw_anl()] Flatten a multiplet into an ngram,
# e.g. (('hello',), ('there', 'friend')) becomes ('hello', 'there', 'friend'):
def tupleify(multiplet):
    if len(multiplet) == 1:
        return multiplet
    else:
        elements = ()
        for tup in multiplet:
            for element in tup:
                elements = elements + (element,)
        return elements

# -----
# [in dev] Finding the duplets most supported by an analogical path
# -----

# Returns list of d2 from duplets (d1, d2) most supported by (p1, singlet):
def right_support(singlet, model):
    cdy, pdy = model
    supp = []
    p2 = (singlet[0][0], singlet[1][0])
    top5_p2 = sorted(cdy['b'][p2], reverse=True, key=lambda x: pdy[(x, p2)]['f'])[:5]
    for p1 in top5_p2:
        path = (p1, p2)
        joint = pdy[path]['j']
        for d1 in top5_p2:
            backw = pdy[(d1, p2)]['b']
            for d2 in cdy['f'][p1]:
                duplet = (d1, d2)
                forw = pdy[(p1, d2)]['f']
                cos = cosine_sim(duplet, path, model)
                score = joint * backw * forw * cos
                supp.append((duplet, score))
    supp.sort(reverse=True, key=lambda x: x[1])
    uniques = []
    for path in supp:
        if path[0][1] not in uniques:
            uniques.append(path[0][1])
    return uniques

# Returns list of d1 from duplets (d1, d2) most supported by (singlet, p2):
def left_support(singlet, model):
    cdy, pdy = model
    supp = []
    p1 = (singlet[0][0], singlet[1][0])
    top5_p1 = sorted(cdy['f'][p1], reverse=True, key=lambda x: pdy[(p1, x)]['b'])[:5]
    for p2 in top5_p1:
        path = (p1, p2)
        joint = pdy[path]['j']
        for d2 in top5_p1:
            forw = pdy[(p1, d2)]['f']
            for d1 in cdy['b'][p2]:
                duplet = (d1, d2)
                backw = pdy[(d1, p2)]['b']
                cos = cosine_sim(duplet, path, model)
                score = joint * backw * forw * cos
                supp.append((duplet, score))
    supp.sort(reverse=True, key=lambda x: x[1])
    uniques = []
    for path in supp:
        if path[0][0] not in uniques:
            uniques.append(path[0][0])
    return uniques

# -----
# [in dev] Different ways of computing analogical likelihood
# -----

# Score is the score of the best analysis:
def ranl_b(ngram, model, dyn={}):
    if len(ngram) == 1:
        return ([(ngram, ngram)], 1, dyn)
    else:
        # Check dynamic lookup table:
        if ngram in dyn:
            # Return (analogies, score, lookup table):
            return (dyn[ngram][0], dyn[ngram][1], dyn)
            """
            # Handle sentence starting ngrams (should outdent):
            elif ngram[0] == '<s>':
                d1, d2 = (ngram[:1], ngram[1:])
                a1, a2 = (ranl_sc(d1, model, dyn)[:2], ranl_sc(d2, model, dyn)[:2])
                dyn[d1], dyn[d2] = a1, a2
                duplet_paths, duplet_score = pairw_anl(a1, a2, model)
                paths = [(x, (d1, d2)) for x in duplet_paths]
                score = duplet_score
                paths.sort(reverse=True, key=lambda x: x[0][1])
                return ([(path[0][0], path[1]) for path in paths[:5]], score, dyn)
            # Handle sentence ending ngrams:
            elif ngram[-1] == '</s>':
                d1, d2 = (ngram[:-1], ngram[-1:])
                a1, a2 = (ranl_sc(d1, model, dyn)[:2], ranl_sc(d2, model, dyn)[:2])
                dyn[d1], dyn[d2] = a1, a2
                duplet_paths, duplet_score = pairw_anl(a1, a2, model)
                paths = [(x, (d1, d2)) for x in duplet_paths]
                score = duplet_score
                paths.sort(reverse=True, key=lambda x: x[0][1])
                return ([(path[0][0], path[1]) for path in paths[:5]], score, dyn)
            """
        # Handle mid-sentence ngrams:
        else:
            # Make all possible binary source-target splits of ngram (a binary
            # source-target split of ('hello', 'there', 'friend') is e.g.
            # (('hello',), ('there', 'friend')), in which case we're looking
            # for paths from ('hello',) to ('there', 'friend')):
            duplets = ((ngram[:i], ngram[i:]) for i in range(1, len(ngram)))
            path_groups = []
            for d1, d2 in duplets:
                # Recursively analyse source and target and record in table:
                a1, a2 = (ranl_b(d1, model, dyn)[:2], ranl_b(d2, model, dyn)[:2])
                dyn[d1], dyn[d2] = a1, a2
                duplet_paths, duplet_score = pairw_anl(a1, a2, model)
                annotated_paths = [(x, (d1, d2)) for x in duplet_paths]
                path_groups.append((annotated_paths, duplet_score))
            path_groups.sort(reverse=True, key=lambda x: x[1])
            paths = sorted(path_groups[0][0], reverse=True, key=lambda x: x[0][1])
            score = path_groups[0][1]
            return ([(path[0][0], path[1]) for path in paths[:5]], score, dyn)

# Maybe doesn't work:
def ranl_parse(sentence, model, n):
    grams = ngrams(sentence, n)[n-2:-(n-2)]
    probs = []
    zeros = []
    for gram in grams:
        print('Checking', gram)
        try:
            probs.append(math.log(ranl_sc(gram, model)[1], 10))
        except ValueError:
            zeros.append(gram)
    p = sum(probs) if zeros == [] else 0
    return (p, zeros)

# Doesn't work:
def ranl(ngram, model, dyn={}):
    score = 0
    """Recursively find 5 best analogical paths for ngram.
    
    Keyword arguments:
    ngram -- an n-tuple of strings
    model -- the analogical path model 
    dyn -- the current dynamic lookup table
    
    Returns:
    (analogies, dict) -- where `analogies` are the five best analogical paths
        for ngram, and `dict` is the dynamic lookup table of the best analogies
        for parts of the ngram
    """
    if len(ngram) == 1:
        return (([(ngram, ngram)], 0), dyn)
    else:
        if ngram in dyn:
            return (dyn[ngram], dyn)
        else:
            duplets = ((ngram[:i], ngram[i:]) for i in range(1, len(ngram)))
            paths = []
            for d1, d2 in duplets:
                a1, a2 = (ranl(d1, model, dyn)[0], ranl(d2, model, dyn)[0])
                dyn[d1], dyn[d2] = a1, a2
                base_pairs = product([x[0] for x in a1], [x[0] for x in a2])
                for pair in base_pairs:
                    b1 = tupleify(pair[0])
                    b2 = tupleify(pair[1])
                    sc, analogies = anl_paths((b1, b2), model)
                    # Record current parse along with best analogical paths
                    # (an element in analogies is (path, score)):
                    paths += [(x, (d1, d2)) for x in analogies]
                    score += sc
            paths.sort(reverse=True, key=lambda x: x[0][1])
            return (([(path[0][0], path[1]) for path in paths[:5]], score), dyn)

# ---------------------------- #
# Bigram analogical path model #
# ---------------------------- #

# Create transitional and joint probability dictionary for each bigram in train
# (train is a list of sentences, a sentence is a list of words):
def ap2_model(train):
    # TODO: add comments to lines
    """Return bigram probability dictionary from context dictionary.
    
    Computes the "probability dictionary" of each word in `cdy` ("context
    dictionary", output of `ctxt_dy()`), containing the joint, backward
    transitional, and forward transitional probabilities of each bigram based
    on the co-occurrence data in `cdy`.
    
    > E.g. if `pdy` is a probability dictionary, then `pdy[('the', 'king')]` is
    `{'joint': 0.1, 'left': 0.6, 'right': 0.002}` if the joint probability of
    `('the', 'king')` is  0.1, its backward transitional probability is 0.6,
    and its forward transitional probability is 0.002.
    
    Keyword arguments:
    cdy -- "context dictionary" containing co-occurrence data for words,
        output of `ctxt_dy()`
    
    Returns:
    dict -- `pdy`, a "probability dictionary" containing the joint, backward
        transitional, and forward transitional probability of each bigram based
        on the co-occurrence data in `cdy`
    """
    cdy = ctxt_dy(train)
    # TODO: record total_freq with ctxt_dy() so that no need to count again
    total_freq = sum(sum(cdy[key]['right'].values()) for key in cdy)
    pdy = {}
    for w1 in cdy:
        pdy[('_', w1)] = sum(cdy[w1]['left'].values()) / total_freq
        pdy[(w1, '_')] = sum(cdy[w1]['right'].values()) / total_freq
        for w2 in cdy[w1]['right']:
            count = cdy[w1]['right'][w2]
            joint = count / total_freq
            left = count / sum(cdy[w2]['left'].values())
            right = count / sum(cdy[w1]['right'].values())
            pdy[(w1, w2)] = {'joint': joint, 'left': left, 'right': right}
    # Compute anl. path weights through each attested bigram (a,b):
    anl_probs = {}
    for (a,b) in pdy:
        if '_' in (a,b):
            anl_probs[(a,b)] = pdy[(a,b)]
        else:
            for s in cdy[b]['left']:
                for t in cdy[a]['right']:
                    try:
                        anl_probs[(s,t)] += pdy[(a,b)]['joint']   \
                                            * pdy[(s,b)]['left']  \
                                            * pdy[(a,t)]['right']
                    except:
                        anl_probs[(s,t)]  = pdy[(a,b)]['joint']   \
                                            * pdy[(s,b)]['left']  \
                                            * pdy[(a,t)]['right']
    return (anl_probs, cdy, pdy)

# [aux for above] Create context dictionary for each word in training data:
def ctxt_dy(train):
    # TODO: add comments for lines
    """Return bigram context dictionary from training data.
    
    Computes the "context dictionary" of each word in `train`. This dictionary
    maps each word to its left and right context dictionaries: the left context
    dictionary consists of pairs (left neighbor, freq), where `left neighbor`
    is a word that occurs in `train` directly before the word, and `freq` is
    the number of times these two words co-occur in this order; the right
    context dictionary is the same but for right neighbors.
    > E.g. if `cdy` is a context dictionary, then cdy['king']['left']['the']
    is 14 if the bigram ('the', 'king') occurs 14 times in the sentences
    of `train`.
    
    Keyword arguments:
    train -- list of lists of words (output of `txt_to_list()` or of
        `list_to_train_test()[0]`)
    
    Returns:
    dict -- `cdy`, where cdy['king']['left']['the'] is 14 if the bigram
        ('the', 'king') occurs 14 times in the sentences of `train`, and
        cdy['king']['right']['was'] is 18 if the bigram ('king', 'was') occurs
        18 times in the sentences of `train`
    """
    vocab = {word for sentence in train for word in sentence}
    vocab = vocab.union({'<s>', '</s>'})
    cdy = {word: {'left':{}, 'right':{}} for word in vocab}
    for sentence in train:
        padded = ['<s>'] + sentence + ['</s>']
        bigrams = zip(padded[:-1], padded[1:])
        for first, second in bigrams:
            try:
                cdy[first]['right'][second] += 1
            except KeyError:
                cdy[first]['right'][second] = 1
            try:
                cdy[second]['left'][first] += 1
            except KeyError:
                cdy[second]['left'][first] = 1
    return cdy

# Compute the analogical likelihood of a bigram (s,t) according to model
# (model is the output of `ap2_model()`):
def ap2_paths(s, t, model):
    """Compute an analogical parse of the bigram `(s, t)` using `model`.
    
    Returns the analogical likelihood of the bigram (s, t), and all
    analogical paths from `s` to `t`, based on `model` (output of `ap_train()`),
    which consists of a "probability dictionary" and a "context dictionary".
    
    Keyword arguments:
    s -- a string
    t -- another string
    model -- (pdy, cdy), output of `ap_train()`, where `pdy` is a "probability
        dictionary" and `cdy` is a "context dictionary"
    
    Returns:
    tuple -- (anl_lhood, anl_paths), where `anl_lhood` is the analogical
        likelihood of the bigram (s, t), and `anl_paths` is the list of
        analogical paths from `s` to `t`, ordered from most to least valuable
    """
    anl_probs, cdy, pdy = model
    attested = set(pdy)
    links = attested.intersection(set(product(cdy[t]['left'], cdy[s]['right'])))
    paths = []
    for a, b in links:
        p = pdy[(a, b)]['joint'] * pdy[(s, b)]['left'] * pdy[(a, t)]['right']
        paths.append([a + ' & ' + b, p])
    paths.sort(key=lambda x: x[1], reverse=True)
    return (sum(path[1] for path in paths), paths)

# Compute the analogical log-probability of a sentence
# (sentence is a list of words, model is the output of ap2_model()):
def ap2_score(sentence, model):
    """Return the analogical log-probability of sentence according to model.
    
    Computes the analogical log-probability of sentence by adding together
    the analogical log-probabilities of the bigrams that occur in it. The
    probabilities of the bigrams (s,t) interpolate the following:
    - mle transitional probability of (s,t),
    - analogical-path transitional probability of (s,t), and
    - mle unigram probability of (_,t).
    
    Keyword arguments:
    sentence -- a list of strings
    model    -- (anl_probs, cdy, pdy), output of ap2_model()
    
    Returns:
    float -- logarithm of the probability of sentence according to model
    """
    bigrams = zip(['<s>'] + sentence, sentence + ['</s>'])
    anl_probs, cdy, pdy = model
    p = 0
    for s, t in bigrams:
        # (1) Get mle transitional probability of (s,t):
        try:
            mle_st = pdy[(s,t)]['right']
        except:
            mle_st = 0
        # (2a) Get anl-path probability of (s,t):
        try:
            ap_st = anl_probs[(s,t)]
        except:
            ap_st = 0
        # (2b) Get anl-path probability of (s,_) (same as its mle probability):
        ap_s = pdy[(s,'_')]
        # (2c) Get anl-path transitional probability of (s,t):
        anl_st = ap_st / ap_s
        # (3) Get mle probability of (_,t):
        mle_t = pdy[('_',t)]
        # Interpolate (1) mle_st, (2) anl_st, and (3) mle_t:
        intp_st = (0.595 * mle_st) + (0.4 * anl_st) + (0.005 * mle_t)
        p += math.log(intp_st, 2)
    return p

def perplexity(test, model):
    prob = 0
    probs = []
    for sentence in test:
        prob += ap2_score(sentence, model)
    pp = 2 ** (-prob / sum(len(sentence) for sentence in test))
    return pp

def tenfold_perplexity(tenfold_splits):
    trains, tests = tenfold_splits
    pps = []
    for i, train in enumerate(trains):
        model = ap2_model(train)
        pps.append(perplexity(tests[i], model))
    return pps

def ap2_prob(sentence, model):
    return 2 ** ap2_score(sentence, model)

def interp_model(train):
    # TODO: add comments to lines
    """Return bigram probability dictionary from context dictionary.
    
    Computes the "probability dictionary" of each word in `cdy` ("context
    dictionary", output of `ctxt_dy()`), containing the joint, backward
    transitional, and forward transitional probabilities of each bigram based
    on the co-occurrence data in `cdy`.
    
    > E.g. if `pdy` is a probability dictionary, then `pdy[('the', 'king')]` is
    `{'joint': 0.1, 'left': 0.6, 'right': 0.002}` if the joint probability of
    `('the', 'king')` is  0.1, its backward transitional probability is 0.6,
    and its forward transitional probability is 0.002.
    
    Keyword arguments:
    cdy -- "context dictionary" containing co-occurrence data for words,
        output of `ctxt_dy()`
    
    Returns:
    dict -- `pdy`, a "probability dictionary" containing the joint, backward
        transitional, and forward transitional probability of each bigram based
        on the co-occurrence data in `cdy`
    """
    cdy = ctxt_dy(train)
    # TODO: record total_freq with ctxt_dy() so that no need to count again
    total_freq = sum(sum(cdy[key]['right'].values()) for key in cdy)
    pdy = {}
    for w1 in cdy:
        pdy[('_', w1)] = sum(cdy[w1]['left'].values()) / total_freq
        pdy[(w1, '_')] = sum(cdy[w1]['right'].values()) / total_freq
        for w2 in cdy[w1]['right']:
            count = cdy[w1]['right'][w2]
            joint = count / total_freq
            left = count / sum(cdy[w2]['left'].values())
            right = count / sum(cdy[w1]['right'].values())
            pdy[(w1, w2)] = {'joint': joint, 'left': left, 'right': right}
    return (cdy, pdy)

def interp_score(sentence, model):
    bigrams = zip(['<s>'] + sentence, sentence + ['</s>'])
    cdy, pdy = model
    p = 0
    for s, t in bigrams:
        try:
            trp_st = pdy[(s,t)]['right']
        except:
            trp_st = 0
        # Get maximum likelihood probability of t:
        p_t = pdy[('_',t)]
        # Get smoothed forw. trans. anl. prob. of st
        # by interpolating trp_st and p_t:
        smp_st = 0.75 * trp_st + 0.25 * p_t
        p += math.log(smp_st, 2)
    return p

def interp_prob(sentence, model):
    return 2 ** interp_score(sentence, model)

# ---- #
# Misc #
# ---- #

# -----
# Finding analogical paths for trigrams in particular
# -----

# Finds smoothed analogical paths for trigram (s, r, t):
def ap3(trigram, model):
    # Parsing as (s, r)-->(t):
    left_score, left_analogies, left_paths = left_anl3(trigram, model)
    # Parsing as (s)-->(r, t):
    right_score, right_analogies, right_paths = right_anl3(trigram, model)
    # Summing analogies and scores:
    analogies = left_analogies + right_analogies
    analogies.sort(reverse=True, key=lambda x: x[0][1])
    return (right_score + left_score, analogies, left_paths, right_paths)

# Calculates analogical likelihood of sentence using model:
def ap3_score(sentence, model):
    trigrams = ngrams(sentence, 3)
    probs = []
    zeros = []
    first, last = (trigrams[0], trigrams[-1])
    # Calculating ('<s>', '<s>', 'first_word') without analogical paths for
    # (('<s>'), ('<s>',)) (so only "(<s>) –-> (<s> first_word)"):
    try:
        probs.append(math.log(right_anl3(first, model)[0], 10))
    except ValueError:
        zeros.append(first)
    # Calculating non-border trigrams normally:
    for trigram in trigrams[1:-1]:
        try:
            probs.append(math.log(ap3(trigram, model)[0], 10))
        except ValueError:
            zeros.append(trigram)
    # Calculating ('last_word', '</s>', '</s>') without analogical paths for
    # (('</s>'), ('</s>',)) (so only "(last_word </s>) –-> (</s>)"):
    try:
        probs.append(math.log(left_anl3(last, model)[0], 10))
    except ValueError:
        zeros.append(last)
    
    p = sum(probs) if zeros == [] else 0
    return (p, zeros)

# Finds left analogies for trigram:
def left_anl3(trigram, model):
    t1, t2, t3 = trigram
    # Compute analogical paths for ((t1, t2), t3):
    d1 = ((t1,), (t2,))
    d2 = (t3,)
    # Compute top 5 analogical paths for (t1, t2):
    left_paths = [x[0] for x in anl_paths(d1, model)[1]][:5]
    left_score = 0
    left_analogies = []
    with alive_bar(len(left_paths)) as bar:
        for path in left_paths:
            curr_anl = anl_paths(((path[0][0], path[1][0]), d2), model)
            left_score += curr_anl[0]
            left_analogies += [(x, tts(d1) + ' = ' + tts(path))
                               for x in curr_anl[1]]
            bar()
    return (left_score, left_analogies, left_paths)

# Finds right analogies for trigram:
def right_anl3(trigram, model):
    t1, t2, t3 = trigram
    # Compute analogical paths for (t1, (t2, t3)):
    d1 = (t1,)
    d2 = ((t2,), (t3,))
    # Compute top 5 analogical paths for (t2, t3):
    right_paths = [x[0] for x in anl_paths(d2, model)[1]][:5]
    right_score = 0
    right_analogies = []
    with alive_bar(len(right_paths)) as bar:
        for path in right_paths:
            curr_anl = anl_paths((d1, (path[0][0], path[1][0])), model)
            right_score += curr_anl[0]
            right_analogies += [(x, tts(d2) + ' = ' + tts(path))
                                for x in curr_anl[1]]
            bar()
    return (right_score, right_analogies, right_paths)

# -----
# Bigram paths with information theoretic weghting (worse than above)
# -----

def aps_inf(s, t, model, forw_losses, backw_losses):
    cdy, pdy = model
    attested = set(pdy)
    links = attested.intersection(set(product(cdy[t]['left'], cdy[s]['right'])))
    paths = []
    for a, b in links:
        #print(a,b)
        ab = pdy[(a,b)]['joint']
        ab_sb = prod_AB_sB(s, a, b, model)
        sb_st = prod_sB_st(t, a, b, backw_losses[(a,b)], model)
        at_st = prod_At_st(s, a, b, forw_losses[(a,b)], model)
        ab_at = prod_AB_At(t, a, b, model)
        score = ab * (ab_sb * sb_st + ab_at * at_st)
        paths.append([a + ' & ' + b, score])
    paths.sort(key=lambda x: x[1], reverse=True)
    return (sum(path[1] for path in paths), paths)

def ap_score_inf(sentence, model, forw_losses, backw_losses):
    """Return the analogical likelihood of a sentence according to `model`.
    
    Computes the analogical likelihood of `sentence` by multiplying together
    the analogical likelihoods of the bigrams that occur in it.
    
    Keyword arguments:
    sentence -- a string containing words separated by spaces, with no
        sentence-ending periods, e.g. 'my name is mary'
    model -- (pdy, cdy), output of `ap_train()`, where `pdy` is a "probability
        dictionary" and `cdy` is a "context dictionary"
    
    Returns:
    tuple -- (p, zeros), where `p` is the analogical likelihood of `sentence`
        according to `model`, and `zeros` is the list of those bigrams in
        `sentence` whose analogical likelihood is zero
    """
    bigrams = zip(['<s>'] + sentence, sentence + ['</s>'])
    cdy, pdy = model
    p = 1
    zeros = []
    # Compute and multiply P([_t] | [s_]) for all bigrams (s, t):
    for s, t in bigrams:
        try:
            p_st = aps_inf(s, t, model, forw_losses, backw_losses)[0] 
        except ValueError:
            zeros.append((first, second))
            p = 0
        
        p_s = 0                                                   # P(S -> s_)
        for a in cdy:
            for b in cdy[a]['right']:
                if s in cdy[b]['left']:
                    ab = pdy[(a,b)]['joint']
                    ab_sb = prod_AB_sB(s, a, b, model)
                    at_st = prod_At_st(s, a, b, forw_losses[(a,b)], model)
                    p_s += ab * (ab_sb + at_st)

        p *= p_st / p_s                          # p *= P(S -> _t | S -> s_)
    
    return (p, zeros)

# -----
# Getting statistics for finding analogical paths for bigrams
# -----

# Compute word-class similarities s~a and t~b and multiply them:
def wc_sim(s, t, a, b, model):
    # TODO: add comments to lines
    """Return the product of the word-class similarities `s`~`a` and `t`~`b`.
    
    Computes the word-class similarities of `s` and `a`, and of `t` and `b`;
    the former is given by `sum_x: P(x|s_)*P(x|a_)` and the latter by
    `sum_y: P(y|_t)*P(y|_b)`. Returns the product of these two values.
    
    Keyword arguments:
    s, t, a, b -- strings, such that we're investigating the analogical path
        from `s` to `t` through `b` and `a`
    model -- `(pdy, cdy)` where `pdy` is a probability dictionary (output of
        `prob_dy()`) and `cdy` is a context dictionary (output of `ctxt_dy()`)
    
    Returns:
    left_sim * right_sim -- where `left_sim` is the word-class similarity of
        `s` and `a`, given by `sum_x: P(x|s_)*P(x|a_)`, and `right_sim` is the
        word-class similarity of `t` and `b`, given by `sum_y: P(y|_t)*P(y|_b)`
    """
    cdy, pdy = model
    left_sim = 0
    for x in min(cdy[s]['right'], cdy[a]['right'], key=len):
        try:
            left_sim += pdy[(s, x)]['right'] * pdy[(a, x)]['right']
        except KeyError:
            continue
    right_sim = 0
    for y in min(cdy[t]['left'], cdy[b]['left'], key=len):
        try:
            right_sim += pdy[(y, t)]['left'] * pdy[(y, b)]['left']
        except KeyError:
            continue
    return left_sim * right_sim

# Compute restricted Kullback-Leibler divergence, i.e. how badly q_distr
# approximates p_distr within the support of q_distr
# (p_distr and q_distr are both tuples of floats):
def rkld(p_distr, q_distr):
    # Restrict p_distr to support of q_distr:
    p_restr = tuple(n for (i, n) in enumerate(p_distr) if q_distr[i] > 0)
    # Normalise p_restr:
    try:
        p = tuple(n / sum(p_restr) for n in p_restr)
    # If p_distr and q_distr are disjoint:
    except:
        return float('inf')
    # Restrict q_distr to support of q_distr:
    q = tuple(n for n in q_distr if n > 0)
    # Compute how badly q approximates p:
    d = sum(p[i] * math.log(p[i]/q[i], 2) for i, n in enumerate(p) if n > 0)
    return d

# Loss + waste:

def loss_waste_forw(pw, qw, model):
    cdy, pdy = model
    p = cdy[pw]['right']
    q = cdy[qw]['right']
    loss =  sum(pdy[(pw,t)]['right']
                * math.log(pdy[(pw,t)]['right']/pdy[(qw,t)]['right'])
                for t in q if t in p)
    waste = sum(pdy[(qw,t)]['right']
                * math.log(1/pdy[(qw,t)]['right'])
                for t in q if t not in p)
    return loss + waste

def loss_waste_backw(pw, qw, model):
    cdy, pdy = model
    p = cdy[pw]['left']
    q = cdy[qw]['left']
    loss =  sum(pdy[(s,pw)]['left']
                * math.log(pdy[(s,pw)]['left']/pdy[(s,qw)]['left'])
                for s in q if s in p)
    waste = sum(pdy[(s,qw)]['left']
                * math.log(1/pdy[(s,qw)]['left'])
                for s in q if s not in p)
    return loss + waste

def loss_AB_sB(a, b, model):
    cdy, pdy = model
    b_supp = cdy[b]['left']
    losses = {s: 1/(2 ** loss_waste_backw(s, a, model)) for s in b_supp}
    total_loss = sum(losses.values())
    return (losses, total_loss)

def loss_AB_At(a, b, model):
    cdy, pdy = model
    a_supp = cdy[a]['right']
    losses = {t: 1/(2 ** loss_waste_forw(t, b, model)) for t in a_supp}
    total_loss = sum(losses.values())
    return (losses, total_loss)

def prod_AB_sB(s, a, b, model):
    cdy, pdy = model
    return pdy[(s,b)]['left']

def prod_AB_At(t, a, b, model):
    cdy, pdy = model
    return pdy[(a,t)]['right']

def prod_sB_st(t, a, b, loss, model):
    losses, total_loss = loss
    cdy, pdy = model
    return losses[t] / total_loss

def prod_At_st(s, a, b, loss, model):
    losses, total_loss = loss
    cdy, pdy = model
    return losses[s] / total_loss

# ----------------- #
# String conversion #
# ----------------- #

# String to tuple -- 'we will run' to ('we', 'will', 'run'):
def s2t(string):
    return tuple(string.split())

# String to duplet -- 'we ; will run' to (('we',), ('will', 'run')):
def s2d(string):
    d1, d2 = string.split(' ; ')
    return (tuple(d1.split()), tuple(d2.split()))

# String to list -- 'we will run' to ['we', 'will', 'run']:
def s2l(string):
    return list(string.split())

# Tuple to string -- ('we', 'will', 'run') to 'we will run':
def t2s(tuple):
    s = ''
    for part in tuple:
        for elem in part:
            s += str(elem) + ' '
    return s[:-1]

def shuf(lst):
    lst_copy = lst[:]
    random.shuffle(lst_copy)
    return lst_copy