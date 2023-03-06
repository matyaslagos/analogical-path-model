# -----
# Computing well-formedness analyses for bigrams and sentences,
# based on distributional analogies, considering immediate contexts
# -----

import csv
import math
import random
from itertools import product
import time

from ecgpack import text_import as ti
from ecgpack import train_test as tt
from ecgpack import n_gram

def ctxt_dy(train):
    """
    Returns bigram context dictionary from training data.
    --- train: [['the', 'evil', 'spirit', 'urged', 'her'], [...], ..., [...]]
    --> dict with words as keys and their left & right context dicts as values,
        with frequencies given in each context dict for each context word
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

def prob_dy(cdy):
    """
    Returns dictionary with bigram probabilities and left & right conditional
    bigram probabilities from bigram context dictionary (output of ctxt_dy()).
    --- cdy: bigram context dictionary (output of ctxt_dy())
    --> [0] dict with pairs of words as keys and their joint & left cond.
            & right cond. probabilities as values
        [1] cdy
    """
    total_freq = sum(sum(cdy[key]['right'].values()) for key in cdy)
    bigram_vocab = ((w1, w2) for w1 in cdy for w2 in cdy)
    pdy = {}
    for w1, w2 in bigram_vocab:
        try:
            count = cdy[w1]['right'][w2]
        except KeyError:
            #prob_dict[(w1, w2)] = {'joint': 0, 'left': 0, 'right': 0}
            continue
        joint = count / total_freq
        left = count / sum(cdy[w2]['left'].values())
        right = count / sum(cdy[w1]['right'].values())
        pdy[(w1, w2)] = {'joint': joint, 'left': left, 'right': right}
    return (pdy, cdy)

# Much faster than prob() on frequent s and t.
def anl(s, t, dys):
    """
    Returns the analogical probability of the bigram (s, t), and all
    analogical paths from s to t, based on a probability dictionary and a
    context dictionary.
    --- s, t: words
    --- dys: (pdy, cdy) where pdy is a probability dict (output of prob_dy())
        and cdy is a context dict (output of ctxt_dy())
    --> [0] analogical probability of the bigram (s, t)
        [1] all analogical paths from s to t, each repr. by a link and a weight
    """
    pdy, cdy = dys
    bigrams = set(pdy)
    links = bigrams.intersection(set(product(cdy[t]['left'], cdy[s]['right'])))
    paths = []
    for a, b in links:
        gs = pdy[(a, b)]['joint'] * pdy[(s, b)]['left'] * pdy[(a, t)]['right']
        wc = wc_sim(s, t, a, b, dys)
        paths.append([a + ' & ' + b, gs * wc])
    paths.sort(key=lambda x: x[1], reverse=True)
    return (sum(path[1] for path in paths), paths)

def wc_sim(s, t, a, b, dys):
    """
    Returns the word-class similarity score of s & a times that of t & b.
    --- s, t, a, b: words
    --- dys: (pdy, cdy) where pdy is a probability dict (output of prob_dy())
        and cdy is a context dict (output of ctxt_dy())
    --> value of: [sum_x: P(x|s_)P(x|a_)] * [sum_y: P(y|_t)P(y|_b)]
    """
    pdy, cdy = dys
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

# -----
# Past functions
# -----

def prob(s, t, dy):
    total_freq = sum(sum(dy[key]['right'].values()) for key in dy)
    
    if s not in dy or t not in dy:
        return [[], 0]
    
    anl_paths = set()
    
    after_s = dy[s]['right']
    before_t = dy[t]['left']
    
    s_right_links = set(after_s.keys())
    t_left_links = set(before_t.keys())
    for b in s_right_links:
        before_b = dy[b]['left']
        
        s_right_left_links = set(before_b.keys())
        for a in t_left_links.intersection(s_right_left_links):
            after_a = dy[a]['right']
            
            # Simulating a generative language model.
            # Bigram frequencies of the edges of the path:
            s_b_freq = after_s[b]
            a_b_freq = after_a[b]
            a_t_freq = after_a[t]
            bigr_freqs = s_b_freq * a_b_freq * a_t_freq
            # Unigram frequencies of left and right link words:
            a_freq = sum(after_a.values())
            b_freq = sum(before_b.values())
            # Generate link edge as collocation and swap for 'a' and 'b':
            gen_and_swap = bigr_freqs / (total_freq * a_freq * b_freq)
            
            # Checking if the quasi-equivalent pairs have similar
            # distributions.
            # - Counting intersection and union method:
            
            # (Needs to be updated for s and t input names & a and b link names.)
            # Number of common right or left neighbours:
            s_a_common = sum(min(after_s[key], after_a[key])
                             for key in after_s if key in after_a)
            t_b_common = sum(min(before_t[key], before_b[key])
                             for key in before_t if key in before_b)
            # Number of total neighbours:
            s_a_total = sum(after_s.values()) + a_freq
            t_b_total = sum(before_t.values()) + b_freq
            # Multiplying the ratios of common vs. total neighbours:
            wc_similarity = (s_a_common / s_a_total) * (t_b_common / t_b_total)
            
            # - Conditional probability method:
            """
            s_freq = sum(after_s.values())
            t_freq = sum(before_t.values())
            
            left_sim = 0
            for x in after_s:
                try:
                    left_sim += (after_s[x] / s_freq) * (after_a[x] / a_freq)
                except:
                    pass
            
            right_sim = 0
            for y in before_t:
                try:
                    right_sim += (before_t[y] / t_freq) * (before_b[y] / b_freq)
                except:
                    pass
            
            wc_similarity = left_sim * right_sim
            """
            # Computing the analogical weight of this path:
            anl_weight = (gen_and_swap * wc_similarity) / (a_freq * b_freq)
            
            anl_paths.add((a + ' & ' + b, anl_weight,
                           (s + '/' + a) + ' ' + b,
                           a + ' ' + (b + '/' + t)))
    
    paths = sorted(list(anl_paths), reverse=True, key=lambda x: x[1])
    return [paths, sum([path[1] for path in paths])]

def anl_prob(bigram, dy):
    result = prob(bigram.split()[0], bigram.split()[1], dy)
    top_ten = [(row[0], tt.om(row[1])) for row in result[0][:10]]
    bottom_ten = [(row[0], tt.om(row[1])) for row in result[0][-10:]]
    every_path = [(row[0], tt.om(row[1])) for row in result[0]]
    print('\nScore:', tt.om(result[1]))
    print('\nTop ten:')
    for row in top_ten:
        print(row)
    print('\nBottom ten:')
    for row in bottom_ten:
        print(row)
    """
    print('\nAll paths:')
    for row in every_path:
        print(row)
    """

def parse(sentence, dy):
    bigrams = zip(['<s>'] + sentence, sentence + ['</s>'])
    probs = []
    zeros = []
    for first, second in bigrams:
        try:
            probs.append(math.log(prob(first, second, dy)[1], 10))
        except ValueError:
            zeros.append((first, second))
    p = sum(probs) if zeros == [] else 0
    return (p, zeros)

def shfl(list):
    r = list.copy()
    random.shuffle(r)
    return r

def xanl_paths(a, b, dy):
    '''
    Computes distributional analogical evidence for well-formedness of
    an ordered pair of strings, taking into account immediate left and
    right context words.
    
    Parameters:
    - a (str): first word
    - b (str): second word
    - dy (dict): context dictionary of training set, output of
      context_dy(phrases)
    
    Returns:
    - [evidence, value] where evidence is the list of analogical links found
      between a and b, and value is sum of weights of 5 best links.
    
    '''
    # All the words that have been right contexts of a:    
    a_rights = right_ctxs(a, dy)
    # All the words that have been left contexts of b:
    b_lefts = left_ctxs(b, dy)
    
    # Finding (+1, -1, +1) paths from a to b:
    links = set()
    for r_link in a_rights:
        for l_link in b_lefts.intersection(left_ctxs(r_link, dy)):
            # Type diversities of links:
            a_td = len(a_rights)
            b_td = len(b_lefts)
            r_link_td = len(left_ctxs(r_link, dy))
            l_link_td = len(right_ctxs(l_link, dy))
            # Value of analogical path
            weight = 1 / (a_td * b_td * r_link_td * l_link_td) * 10**6
            # Adding path to set of linking paths
            links.add((l_link + ' & ' + r_link, weight,
                       (a + '/' + l_link) + ' ' + r_link,
                       l_link + ' ' + (r_link + '/' + b)))
    
    l = sorted(list(links), reverse = True, key = lambda x: x[1])
    # Try with sum of weights
    return [l, prod([x[1] for x in l[:5]])]

def show_anl(a: str, b: str, dy: dict):
    """ Prints the output of anl_paths(a, b, dy). """
    out = xanl_paths(a, b, dy)
    
    l = out[0]
    s = list(map(lambda x: (x[0], sci_ntn(x[1]), x[2], x[3]), l))
    evidence = '\n'.join([str(link) for link in s[:5]])
    print('Score is:', sci_ntn(out[1]),
          '\nBased on:\n' + evidence)

def overl_gen(a: str, b: str, dy: dict):
    '''
    Computes distributional analogical evidence for well-formedness of
    an ordered pair of strings.
    
    Parameters:
    - a (str): first word
    - b (str): second word
    - dy (dict): context dictionary of training set, output of
      context_dy(phrases)
    
    Returns:
    - [evidence, weight] where `evidence` is the list of analogical links found
      between `a` and `b`, and `weight` is sum of weights of 5 best links.
    
    '''
    # All the words that have been suffixes of a:    
    a_suffs = [x[2][0] for x in dy[a]]
    # All the words that have been prefixes of b:
    b_prefs = [x[0][-1] for x in dy[b]]
    
    # Sum of frequencies of the words in dy
    t = 0
    for i in dy:
        t = t + len(dy[i])
    
    # Word pairs: (i <- prefix of both w and b, w <- suffix of a)
    links = set()
    # Only (+1, -1, +1) paths:
    for w in set(a_suffs):
        aw_freq = a_suffs.count(w)
        w_prefs = prefs(1, w, dy)
        for i in set(b_prefs).intersection(set(w_prefs)):
            # Frequencies of temporary bigrams
            iw_freq = w_prefs.count(i)
            ib_freq = b_prefs.count(i)
            gen_by_switch = (aw_freq * iw_freq * ib_freq) / \
                            (len(dy[i]) * len(dy[w]) )
            weight = gen_by_switch
            links.add((i + ' & ' + w, weight,
                       a + '/' + i + ' ' + w, i + ' ' + w + '/' + b))

    l = sorted(list(links), reverse = True, key = lambda x: x[1])
    #return [list(map(lambda x: (x[0], sci_ntn(x[1]), x[2], x[3]), l)),
            #sci_ntn(prod([x[1] for x in l[:5]]))]
    return [l, sum([x[1] for x in l])]
    # list(pref_links.union(suff_links))

def prod(l):
    if l == []:
        return 0
    elif len(l) == 1:
        return l[0]
    else:
        return l[0] * prod(l[1:])

def sci_ntn(n):
    return format(n, '.2E')

def xan_parse(sentence: list, dy: dict):
    c = 1
    for i in range(len(sentence)-1):
        c = c * xanl_paths(sentence[i], sentence[i+1], dy)[1]
    return sci_ntn(c)

def right_ctxs(w, dy):
    return set([key[2][0] for key in dy[w]])

def left_ctxs(w, dy):
    return set([key[0][-1] for key in dy[w]])