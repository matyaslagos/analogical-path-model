# Bigram analogical path model

from ap import txt2list
from collections import defaultdict
import random

def train_test(corpus):
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
    train = [sentence for sentence in sentences[:n]]
    vocab = {word for sentence in train for word in sentence}
    test = [tuple(sentence) for sentence in sentences[n:]
                            if set(sentence).issubset(vocab)]
    return (train, test)

def ngrams_list(sentence, n):
    beg = ['<s{}>'.format(i) for i in range(1,n)]
    end = ['</s{}>'.format(i) for i in range(1,n)]
    sentence = beg + sentence + end
    return list(zip(*(sentence[i:len(sentence)-n+i+1] for i in range(n))))

def ctxt_dy2(train):
    cdy = {'fw': defaultdict(lambda: defaultdict(int)),
           'bw': defaultdict(lambda: defaultdict(int))}
    for sentence in train:
        bigrams = ngrams_list(sentence, 2)
        for bigram in bigrams:
            cdy['fw'][bigram[:1]][bigram[1:]] += 1
            cdy['bw'][bigram[1:]][bigram[:1]] += 1
    # Convert cdy from defaultdict to dict (dr is direction i.e. 'fw' or 'bw')
    cdy = {dr: {ctxt: dict(cdy[dr][ctxt]) for ctxt in cdy[dr]} for dr in cdy}
    return cdy

def prob_dy2(train):
    cdy = ctxt_dy2(train)
    total_freq = sum(len(sentence) + 1 for sentence in train)
    pdy = defaultdict(lambda: defaultdict(dict))
    for ctxt in cdy['fw']:
        ctxt_freq = sum(cdy['fw'][ctxt].values())
        #pdy[ctxt]['jt'] = ctxt_freq / total_freq
        for goal in cdy['fw'][ctxt]:
            bigram_freq = cdy['fw'][ctxt][goal]
            goal_freq = sum(cdy['bw'][goal].values())
            probs = {'fw': bigram_freq / ctxt_freq,
                     'bw': bigram_freq / goal_freq,
                     'jt': bigram_freq / total_freq}
            pdy[ctxt][goal] = probs
    pdy = {ctxt: {goal: pdy[ctxt][goal] for goal in pdy[ctxt]} for ctxt in pdy}
    return (cdy, pdy)

def aps2(bigram, model):
    cdy, pdy = model
    ctxt, goal = (bigram.split()[0],), (bigram.split()[1],)
    # Z-shaped paths:
    lr_words = defaultdict(float)
    rl_words = defaultdict(float)
    z_paths = []
    for anl_goal in cdy['fw'][ctxt]:
        for anl_ctxt in cdy['bw'][anl_goal]:
            if goal in pdy[anl_ctxt]: # bw jt fw
                anl_prob =   pdy[ctxt][anl_goal]['fw']     \
                           * pdy[anl_ctxt][anl_goal]['bw'] \
                           * pdy[anl_ctxt][goal]['fw']
                lr_words[anl_ctxt] += anl_prob
                rl_words[anl_goal] += anl_prob
                z_paths.append((anl_ctxt, anl_goal))
    # Analogical context words:
    ll_words = defaultdict(float)
    for anl_envr in cdy['bw'][ctxt]:
        for anl_ctxt in cdy['fw'][anl_envr]:
            if '</s>' not in anl_ctxt and goal in pdy[anl_ctxt]:
                anl_prob =   pdy[anl_envr][ctxt]['bw']     \
                           * pdy[anl_envr][anl_ctxt]['fw'] \
                           * pdy[anl_ctxt][goal]['fw']
                ll_words[anl_ctxt] += anl_prob
    # Analogical goal words:
    rr_words = defaultdict(float)
    for anl_goal in cdy['fw'][ctxt]:
        if '</s>' not in anl_goal:
            for anl_envr in cdy['fw'][anl_goal]:
                if goal in cdy['bw'][anl_envr]:
                    anl_prob =   pdy[goal][anl_envr]['fw']     \
                               * pdy[anl_goal][anl_envr]['bw'] \
                               * pdy[ctxt][anl_goal]['bw']
                    rr_words[anl_goal] += anl_prob
    # Best paths:
    substs = defaultdict(float)
    cross_substs = defaultdict(float)
    for l_word, r_word in z_paths:
        l_prob = sum(pdy[l_word][word]['jt'] for word in pdy[l_word])
        r_prob = sum(pdy[word][r_word]['jt'] for word in pdy
                                             if r_word in pdy[word])
        substs[l_word + r_word] +=   (lr_words[l_word] * ll_words[l_word]
                                                       / l_prob) \
                                   * (rl_words[r_word] * rr_words[r_word]
                                                       / r_prob)
        substs[ctxt + r_word]   += rl_words[r_word] * rr_words[r_word] / r_prob
        substs[l_word + goal]   += lr_words[l_word] * ll_words[l_word] / l_prob
        cross_substs[l_word + r_word] +=   (lr_words[l_word] * ll_words[l_word]
                                                             / l_prob) \
                                         * (rl_words[r_word] * rr_words[r_word]
                                                             / r_prob)
    substs = [x for x in substs.items() if x[1] > 0]
    substs.sort(key=lambda x: x[1], reverse=True)
    cross_substs = [x for x in cross_substs.items() if x[1] > 0]
    cross_substs.sort(key=lambda x: x[1], reverse=True)
    return (substs, cross_substs)