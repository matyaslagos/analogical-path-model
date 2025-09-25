"""
Using NLTK to construct bigram language models with (a) add-half smoothing,
(b) Good-Turing smoothing, and (c) Kneser-Ney smoothing.
"""

import csv
import math
import random
import nltk

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Lidstone
from nltk.lm import KneserNeyInterpolated
from nltk.probability import *

from ecgpack import text_import as ti
from ecgpack import train_test as tt

def lid_fit(train):
    """ Trains a lidstone-smoothed bigram model on a training set,
    adding gamma = 0.5 to each bigram count. """
    tr, voc = padded_everygram_pipeline(2, train)
    lid = Lidstone(0.5, 2)
    lid.fit(tr, voc)
    return lid

def kn_fit(train):
    """ Trains a kneser-ney-smoothed bigram model on a training set,
    with absolute discount = 0.75. """
    tr, voc = padded_everygram_pipeline(2, train)
    kn = KneserNeyInterpolated(2, discount=0.75)
    kn.fit(tr, voc)
    return kn

# Testing for the Lidstone and Kneser-Ney language models:
def lm_test(sentence, lm):
    """ Tests a bigram model on a list of words. """
    bigrams = zip(['<s>'] + sentence, sentence + ['</s>'])
    return math.prod([lm.score(second, [first]) for first, second in bigrams])


# Simple Good-Turing implementation.

# Bigrams from training data:
def bigrams_sgt(train):
    return [(first, second) for sentence in train
            for (first, second) in zip(['<s>'] + sentence, sentence + ['</s>'])]

# Bigram frequency dictionary:
def bigr_dict(bigrams):
    vocab = {'<s>', '</s>'}.union(set(first for first, second in bigrams))
    bigram_freqs = {word: {} for word in vocab}
    for first, second in bigrams:
        try:
            bigram_freqs[first][second] += 1
        except KeyError:
            bigram_freqs[first][second] = 1
    return bigram_freqs

# Normalising discount by unigram probability products of unseen bigrams:
def norm_factor(total_tokens, bigram_freqs):
    redist = 0
    for first in bigram_freqs:
        p_first = sum(bigram_freqs[first].values()) / total_tokens
        rem_prob = p_first
        for second in bigram_freqs[first]:
            p_second = sum(bigram_freqs[second].values()) / total_tokens
            rem_prob -= p_first * p_second
        redist += rem_prob
    return redist

def gt_fit(train):
    bigrams = bigrams_sgt(train)
    bigram_freqs = bigr_dict(bigrams)
    total_tokens = sum(len(sentence) for sentence in train)
    fd = FreqDist(bigrams)
    sgt = SimpleGoodTuringProbDist(fd)
    p_zero_prime = norm_factor(total_tokens, bigram_freqs)
    p_zero = sgt.discount()
    bigram_freqs['</s>']['#'] = len(train)
    return (sgt, p_zero/p_zero_prime, bigram_freqs, total_tokens)

def gt_test(sentence, sgt_params):
    sgt, alpha, bigram_freqs, total = sgt_params
    bigrams = zip(['<s>'] + sentence, sentence + ['</s>'])
    prob = 1
    for first, second in bigrams:
        p_first = sum(bigram_freqs[first].values()) / total
        if second in bigram_freqs[first]:
            prob *= sgt.prob((first, second)) / p_first
        else:
            p_second = sum(bigram_freqs[second].values()) / total
            prob *= alpha * p_second
    return prob