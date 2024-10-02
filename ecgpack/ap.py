# =====
# The analogical path model for computing well-formedness analyses for n-grams
# using recursive distributional analogies.
# =====

import math
import random
from itertools import product, chain
from collections import defaultdict
from math import log2, exp2

# --------------------- #
# Setting up the corpus #
# --------------------- #

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
    return [line.strip().split() for line in lines]

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
    train = [tuple(sentence) for sentence in sentences[:n]]
    vocab = {word for sentence in train for word in sentence}
    test = [tuple(sentence) for sentence in sentences[n:]
                            if set(sentence).issubset(vocab)]
    return (train, test)

# ----------------------------------- #
# New bigram analogical path model(s) #
# ----------------------------------- #

def ctxt_dy2(train):
    cdy = {'fw': defaultdict(lambda: defaultdict(int)),
           'bw': defaultdict(lambda: defaultdict(int))}
    for sentence in train:
        bigrams = ngrams_list(sentence, 2)
        for bigram in bigrams:
            cdy['fw'][bigram[:1]][bigram[1:]] += 1
            cdy['bw'][bigram[1:]][bigram[:1]] += 1
    # Convert cdy from defaultdict to dict (dr is direction i.e. 'fw' or 'bw')
    return {dr: {ctxt: dict(cdy[dr][ctxt]) for ctxt in cdy[dr]} for dr in cdy}

def prob_dy2(train):
    cdy = ctxt_dy2(train)
    total_freq = sum(len(sentence) + 1 for sentence in train)
    pdy = defaultdict(lambda: defaultdict(float))
    for ctxt in cdy['fw']:
        ctxt_freq = sum(cdy['fw'][ctxt].values())
        pdy[ctxt]['jt'] = ctxt_freq / total_freq
        for goal in cdy['fw'][ctxt]:
            bigram = ctxt + goal
            bigram_freq = cdy['fw'][ctxt][goal]
            goal_freq = sum(cdy['bw'][goal].values())

            pdy[bigram]['fw'] = bigram_freq / ctxt_freq
            pdy[bigram]['bw'] = bigram_freq / goal_freq
            pdy[bigram]['jt'] = bigram_freq / total_freq
    return (cdy, {bigram: dict(pdy[bigram]) for bigram in pdy})

def in_model(train):
    total_freq = sum(len(sentence) + 1 for sentence in train)
    cdy, pdy = prob_dy2(train)
    ady = defaultdict(float)
    for bridge in pdy:
        anl_ctxt, anl_goal = bridge[:1], bridge[1:]
        for ctxt in cdy['bw'][anl_goal]:
            ctxt_prob = sum(cdy['fw'][ctxt].values()) / total_freq
            for goal in cdy['fw'][anl_ctxt]:
                anl_prob = pdy[ctxt + anl_goal]['bw'] \
                           * pdy[bridge]['jt']        \
                           * pdy[anl_ctxt + goal]['fw']
                ady[ctxt + goal] += anl_prob / ctxt_prob
    return (ady, cdy, pdy)

def fw_model(train):
    total_freq = sum(len(sentence) + 1 for sentence in train)
    cdy, pdy = prob_dy2(train)
    ady = defaultdict(float)
    for bridge in pdy:
        anl_ctxt, anl_goal = bridge[:1], bridge[1:]
        if anl_ctxt == ('<s>',):
            continue
        for ctxt in cdy['bw'][anl_ctxt]:
            ctxt_prob = sum(cdy['fw'][ctxt].values()) / total_freq
            for goal in cdy['bw'][anl_goal]:
                if goal == ('<s>',):
                    anl_prob = pdy[ctxt + anl_ctxt]['bw'] \
                               * pdy[bridge]['jt']        \
                               * pdy[goal + anl_goal]['bw']
                    ady[ctxt + ('</s>',)] += anl_prob / ctxt_prob
                    continue
                anl_prob = pdy[ctxt + anl_ctxt]['bw'] \
                           * pdy[bridge]['jt']        \
                           * pdy[goal + anl_goal]['bw']
                ady[ctxt + goal] += anl_prob / ctxt_prob
    return (ady, cdy, pdy)

def anl_paths2(bigram, cdy, pdy):
    ctxt, goal = (bigram.split()[0],), (bigram.split()[1],)
    analogies = []
    for anl_goal in cdy['fw'][ctxt]:
        for anl_ctxt in cdy['bw'][anl_goal]:
            try:
                prob = pdy[anl_ctxt + goal]['fw']       \
                       * pdy[anl_ctxt + anl_goal]['jt'] \
                       * pdy[ctxt + anl_goal]['bw']
                envr_prob = 0
                for bw_envr in cdy['bw'][ctxt]:
                    for fw_envr in cdy['fw'][goal]:
                        try:
                            envr_prob += pdy[bw_envr + anl_ctxt]['bw']   \
                                         * pdy[bw_envr + ctxt]['jt']     \
                                         * pdy[anl_goal + fw_envr]['fw'] \
                                         * pdy[goal + fw_envr]['jt']
                        except:
                            continue
                analogies.append((anl_ctxt + anl_goal, prob * envr_prob))
            except:
                continue
    analogies.sort(key=lambda x: x[1], reverse=True)
    return analogies

def interp_prob(model, sentence, total_freq):
    ady, cdy, pdy = model
    bigrams = ngrams_list(sentence, 2)
    prob = 1
    for bigram in bigrams:
        try:
            cond_prob = pdy[bigram]['fw']
        except:
            cond_prob = 0
        prob *= cond_prob * 0.595   \
                + ady[bigram] * 0.4 \
                + (sum(cdy['bw'][bigram[1:]].values()) / total_freq) * 0.005
    return prob

def perplexity(test, model, train):
    total_freq = sum(len(sentence) + 1 for sentence in train)
    rate = 1 / sum(len(sentence) + 1 for sentence in test)
    cross_entropy = sum(log2(1 / interp_prob(model, sentence, total_freq))
                        for sentence in test)
    return exp2(cross_entropy * rate)

# --------------------------------- #
# New trigram analogical path model #
# --------------------------------- #

def ngrams_list(sentence, n):
    sentence = ['<s>'] * (n-1) + sentence + ['</s>']
    return list(zip(*(sentence[i:len(sentence)-n+i+1] for i in range(n))))

def ctxt_dy(train):
    cdy = {'fw': defaultdict(lambda: defaultdict(int)),
           'bw': defaultdict(lambda: defaultdict(int)),
           'md': defaultdict(lambda: defaultdict(int))}
    for sentence in train:
        trigrams = ngrams_list(sentence, 3)
        for trigram in trigrams:
            for i in range(3):
                for j in range((i == 0) + 1):
                    cdy['fw'][trigram[i:-1-j]][trigram[-1-j:]]   += 1
                    cdy['bw'][trigram[i+1+j:]][trigram[i:i+1+j]] += 1
            cdy['md'][trigram[:1] + trigram[-1:]][trigram[1:2]] += 1
            skipgram = trigram[:1] + ('*',) + trigram[-1:]
            for j in range(2):
                cdy['fw'][skipgram[:-1-j]][skipgram[-1-j:]] += 1
                cdy['bw'][skipgram[1+j:]][skipgram[:1+j]]   += 1
    return {key: dict(cdy[key]) for key in cdy}

def prob_dy(train):
    cdy = ctxt_dy(train)
    total_freq = sum(len(sentence) + 1 for sentence in train)
    pdy = {'fw': defaultdict(lambda: defaultdict(float)),
           'bw': defaultdict(lambda: defaultdict(float)),
           'md': defaultdict(lambda: defaultdict(float)),
           'jt': defaultdict(float)}
    for dr in cdy:
        for ctxt in cdy[dr]:
            ctxt_freq = sum(cdy[dr][ctxt].values())
            for goal in cdy[dr][ctxt]:
                pdy[dr][goal][ctxt] = cdy[dr][ctxt][goal] / ctxt_freq
                if dr == 'fw':
                    pdy['jt'][ctxt + goal] = cdy[dr][ctxt][goal] / total_freq
    return (cdy, {key: dict(pdy[key]) for key in pdy})

def anl_paths(ctxt_str, goal_str, cdy, pdy):
    ctxt, goal = tuple(ctxt_str.split()), tuple(goal_str.split())
    

def external_anls(ctxt, goal, cdy):
    bigrams = set(x[1:] for x in cdy if '*' not in x and '_' not in x)
    bw_borders = set(key for key in cdy[('_',) + ctxt]
                     if len(key) > 0 and '*' not in key[1:])
    fw_borders = set(key for key in cdy[goal + ('_',)]
                     if len(key) > 0 and '*' not in key[:1])
    anl_ctxts = set(key for bw_b in bw_borders for key in cdy[bw_b + ('_',)]
                    if len(key) > 0 and '*' not in key)
    anl_goals = set(key for fw_b in fw_borders for key in cdy[('_',) + fw_b]
                    if len(key) > 0 and '*' not in key)
    bridges = set(bridge for bridge in product(anl_ctxts, anl_goals)
                  if bridge[0] + bridge[1] in bigrams)
    analogies = defaultdict(float)
    return bridges
    for bw_border, fw_border in product(bw_borders, fw_borders):
        #print(bw_border, fw_border)
        bw_repl = emp_cond_prob(ctxt[:1], bw_border + ('_',), cdy)
        fw_repl = emp_cond_prob(goal[1:], ('_',) + fw_border, cdy)
        for anl_ctxt, anl_goal in bridges:
            if anl_ctxt + ('*',) in cdy[bw_border + ('_',)] \
            and ('*',) + anl_goal in cdy[('_',) + fw_border]:
                br_prob = emp_cond_prob(anl_ctxt + anl_goal, ('*', '_',), cdy)
                fw_link = emp_cond_prob(bw_border, ('_',) + anl_ctxt, cdy)
                bw_link = emp_cond_prob(fw_border, anl_goal + ('_',), cdy)
                prob = bw_repl * bw_link * br_prob * fw_link * fw_repl
                analogies[(anl_ctxt, anl_goal)] += prob
    analogies = sorted(analogies.items(), key=lambda x: x[1], reverse=True)
    return analogies


# -----
# Training the analogical trigram model
# -----

# [Aux to ctxt_dy()] Return list of n-grams in sentence:

"""
# [Aux of ap3_model()] Computes context dictionary of train:
def ctxt_dy(train):
    size = 0
    cdy = {'forw': {}, 'backw': {}}
    for sentence in train:
        size += len(sentence)
        trigrams = ngrams(sentence, 3)
        # Record cooccurrence data in cdy with forward and backward contexts:
        ctxt_dy_update(cdy, ('<s>',), ('<s>',))
        for gram in trigrams:
            # Bigram model: (ctxt,) -> (goal,):
            ctxt_dy_update(cdy, gram[-2:-1], gram[-1:])
            # Trigram model: (ctxt,) –> (goal1, goal2) and
            #                (ctxt1, ctxt2) -> (goal,):
            ctxt_dy_update(cdy, gram[:-2], gram[-2:])
            ctxt_dy_update(cdy, gram[:-1], gram[-1:])
    return (cdy, size)


# [Aux of ctxt_dy()] Updates context dictionary:
def ctxt_dy_update(cdy, context, goal):
    # Forward contexts:
    try:
        cdy['forw'][context][goal] += 1
    except:
        try:
            cdy['forw'][context][goal] = 1
        except:
            cdy['forw'][context] = {goal: 1}
    # Backward contexts:
    try:
        cdy['backw'][goal][context] += 1
    except:
        try:
            cdy['backw'][goal][context] = 1
        except:
            cdy['backw'][goal] = {context: 1}

# Computes trigram analogical model (records cooccurrence data and cond. probs):
def ap3_model(train):
    # Compute context dictionary and restrict it to bigrams:
    cdy, size = ctxt_dy(train)
    # Compute probability dictionary (prob_dy() ):
    pdy = prob_dy(cdy, size)
    return (cdy, pdy)

# [Aux of ap3_model()] Computes probability dictionary
def prob_dy(cdy, size):
    pdy = {}
    # Forward contexts unigram and conditional probabilities:
    for w1 in cdy['forw']:
        pdy[(w1, '_')] = sum(cdy['forw'][w1].values()) / size
        for w2 in cdy['forw'][w1]:
            count = cdy['forw'][w1][w2]
            joint = count / size
            backw = count / sum(cdy['backw'][w2].values())
            forw  = count / sum(cdy['forw'][w1].values())
            pdy[(w1, w2)] = {'joint': joint, 'backw': backw, 'forw': forw}
    # Backward contexts unigram probabilities:
    for w2 in cdy['backw']:
        pdy[('_', w2)] = sum(cdy['backw'][w2].values()) / size
    return pdy

# [Aux of ap3_model()] Checks if current path is among top n paths in ady:
def anl_dy_update(anl_dy, bigram, path, anl_prob, n):
    # If (s,t) already has some analogical paths,
    # update its analogical probability and the top n paths if necessary:
    try:
        anl_data = anl_dy[bigram]
        anl_data['p_anl'] += anl_prob
        paths = anl_data['paths']
        # If (s,t) already has at least n analogical paths,
        # check if new path is among top n and include it if yes:
        try:
            if paths[n-1][1] < anl_prob:
                paths[n-1] = (path, anl_prob)
                paths.sort(key=lambda item: item[1], reverse=True)
        # If (s,t) doesn't have at least n analogical paths,
        # just add the new path and sort the paths:
        except:
            paths.append((path, anl_prob))
            paths.sort(key=lambda item: item[1], reverse=True)
    # If (s,t) doesn't have any analogical paths yet, just record the new path:
    except:
        anl_dy[bigram] = {'p_anl': anl_prob,
                          'paths': [(path, anl_prob)]}

# -----
# Recursively computing the best analogies, obtaining trees as epiphenomena
# -----

# Works with ap3_internal_anls() and ap3_external_anls():
def ap3_drec_anl(gram, model, dr='brdg', dyn={'brdg': {}, 'fw': {}, 'bw': {}}):
    # Recursion ends when we reach single words or sentence markers:
    if len(gram) == 1 or gram in (('<s>', '<s>'), ('</s>', '</s>')):
        return {'prob': 1,
                'tree': gram[0],
                'analogies': {gram: 1}}
    # Dynamic lookup:
    elif gram in dyn[dr]:
        return dyn[dr][gram]
    # Carrying out the analysis:
    else:
        # Split n-gram into all possible (ctxt, goal) duplets:
        duplets  = [(gram[:i], gram[i:]) for i in range(1, len(gram))]
        # Compute analysis for each possible duplet:
        parse_dy = {duplet: {'prob': 0,
                             'tree': (),
                             'analogies': {}}
                    for duplet in duplets}
        for duplet in duplets:
            ctxt, goal = duplet
            # Recursively analyse ctxt and goal, looking forward resp. backward:
            rec_ctxt, rec_goal = ap3_drec_anl(ctxt, model, 'fw', dyn), \
                                 ap3_drec_anl(goal, model, 'bw', dyn)
            # Retrieve top 5 analogical substitutes for ctxt and goal:
            anl_ctxts = rec_ctxt['analogies'].keys()
            anl_goals = rec_goal['analogies'].keys()
            # Compute all combinations of best analogical substitutes:
            anl_duplets = list(product(anl_ctxts, anl_goals))
            # Compute analogies and probability for each analogical substitute
            # in the appropriate direction (forw. or backw.):
            anl_dy = {}
            duplet_prob = 0
            for anl_duplet in anl_duplets:
                anl_ctxt, anl_goal = anl_duplet
                # (1) How good are the analogical substitutes?
                # Weighting for: Was anl_ctxt a good analogy for ctxt?
                anl_ctxt_wt = rec_ctxt['analogies'][anl_ctxt]
                # Weighting for: Was anl_goal a good analogy for goal?
                anl_goal_wt = rec_goal['analogies'][anl_goal]
                # (2) Finding internal analogies for analogical substitutes.
                # If internal analogies are already computed, just get them:
                if anl_duplet in dyn['brdg']:
                    anl_bridges = dyn['brdg'][anl_duplet]
                # If internals aren't computed, compute and save them:
                else:
                    anl_bridges = ap3_internal_anls(anl_duplet, model)
                    dyn['brdg'][anl_duplet] = anl_bridges
                # (3) Finding external analogies (if necessary) and recording
                #     the analogical data.
                # If we're at the top n-gram or looking at markers, record
                # just the internal analogies
                # (dr is 'brdg' iff we're at the top n-gram):
                if (dr == 'brdg')                               \
                or (dr == 'bw' and anl_ctxt == ('<s>', '<s>'))  \
                or (dr == 'fw' and anl_goal == ('</s>', '</s>')):
                    for bridge in anl_bridges:
                        anl_bigram = bridge[0][0] + bridge[0][1]
                        score = bridge[1] * anl_ctxt_wt * anl_goal_wt
                        try: # if anl_bigram is already in anl_dy:
                            anl_dy[anl_bigram]['substs'] += [anl_duplet]
                            anl_dy[anl_bigram]['score']  += score
                        except KeyError:
                            anl_dy[anl_bigram] = {'substs': [anl_duplet],
                                                   'score': score}
                        duplet_prob += score
                # If we're in normal case, then compute external analogies
                # and record the values weighted by them:
                else:
                    anl_links = ap3_external_anls(anl_duplet, model, dr)
                    # Set directionality index for connecting links with bridge:
                    d, opp_d = ((dr == 'bw'), (dr != 'bw'))
                    for bridge in anl_bridges:
                        anl_bigram = bridge[0][0] + bridge[0][1]
                        internal_score = bridge[1] * anl_ctxt_wt * anl_goal_wt
                        external_score = sum(link[1] for link in anl_links
                                             if link[0][d] == bridge[0][opp_d])
                        score = internal_score * external_score
                        try: # if anl_bigram is already in anl_dy:
                            anl_dy[anl_bigram]['substs'] += [anl_duplet]
                            anl_dy[anl_bigram]['score']  += score
                        except KeyError:
                            anl_dy[anl_bigram] = {'substs': [anl_duplet],
                                                   'score': score}
                        duplet_prob += score
            # Summarise current tree's analogical data:
            parse_dy[duplet]['prob'] = duplet_prob        \
                                       * rec_ctxt['prob'] \
                                       * rec_goal['prob']
            parse_dy[duplet]['analogies'] = anl_dy
            parse_dy[duplet]['tree'] = (rec_ctxt['tree'], rec_goal['tree'])
        # Select best parse for n-gram and get its analogical data:
        best_parse = sorted((parse_dy[duplet] for duplet in parse_dy),
                            key=lambda x: x['prob'], reverse=True)[0]
        prob = best_parse['prob']
        tree = best_parse['tree']
        anl_dy = best_parse['analogies']
        anl_list = [(path, anl_dy[path]['score']) for path in anl_dy]
        # Get top 5 analogies by score as a dictionary of (path, score) pairs:
        analogies = dict(sorted(anl_list, key=lambda x: x[1], reverse=True)[:5])
        dyn[dr][gram] = {'prob': prob,
                         'tree': tree,
                         'analogies': analogies}
        return dyn[dr][gram]

# [^Aux of ap3_drec_anls()]
def ap3_internal_anls(duplet, model):
    ctxt, goal = duplet
    cdy, pdy = model
    anl_ctxts = (x for x in cdy['backw'][goal].keys() if len(x) == 1)
    anl_goals = (x for x in cdy['forw'][ctxt].keys() if len(x) == 1)
    # Get attested bigrams:
    attested = (x for x in pdy.keys() if len(x[0]) == 1 and len(x[1]) == 1)

    potential_paths = product(set(anl_ctxts), set(anl_goals))
    anl_paths = set(attested).intersection(set(potential_paths))

    analogies = []
    for anl_ctxt, anl_goal in anl_paths:
        anl_prob = pdy[(anl_ctxt, anl_goal)]['joint'] \
                   * pdy[(ctxt, anl_goal)]['backw']   \
                   * pdy[(anl_ctxt, goal)]['forw']
        analogies.append(((anl_ctxt, anl_goal), anl_prob))
    analogies.sort(key=lambda x: x[1], reverse=True)
    return analogies

# [^Aux of ap3_drec_anls()]
def ap3_external_anls(duplet, model, dr):
    ctxt, goal = duplet
    cdy, pdy = model
    d, opp_d = ('backw', 'forw') if dr == 'bw' else ('forw', 'backw')
    # Get attested bigrams:
    att_bigrams = [x for x in pdy.keys() if len(x[0]) == 1 and len(x[1]) == 1]
    # Get contexts and goals for possible analogical bridges:
    anl_ctxts = (x for x in cdy[d][ctxt] if len(x) == 1 and x != ('</s>',))
    anl_goals = (x for x in cdy[d][goal] if len(x) == 1 and x != ('<s>',))
    # Get analogical links:
    poss_links = product(anl_ctxts, anl_goals)
    anl_links = set(att_bigrams).intersection(set(poss_links))
    # Compute full analogies:
    anl_dy = {}
    for anl_link in anl_links:
        anl_ctxt, anl_goal = anl_link
        # Set the connecting duplet:
        if dr == 'fw':
            con_duplet = (goal, anl_goal)
        else:
            con_duplet = (anl_ctxt, ctxt)
        # Calculate weight of analogical link:
        try:
            link_weight = pdy[anl_link][d] \
                          * pdy[con_duplet][opp_d]
        except KeyError:
            print('Duplet is:', duplet)
            print('Direction is:', dr, d)
            print('Analogical link is:', anl_link)
            print('Connecting duplet is:', con_duplet)
            return '---'
        # Record in analogy dictionary:
        try:
            anl_dy[anl_link] += link_weight
        except:
            anl_dy[anl_link]  = link_weight
    return sorted(anl_dy.items(), key=lambda x: x[1], reverse=True)

def duplet_backoff(duplet, model):
    ctxt, goal = duplet
    cdy, pdy = model
    skip_goals = set(chain(*(cdy['forw'][skip].keys() for skip in cdy['forw']
                                                      if skip[0] == ctxt[0])))
    skip_ctxts = set(chain(*(cdy['backw'][skip].keys() for skip in cdy['backw']
                                                      if skip[1:] == goal[1:])))
    anl_goals = [sgoal for sgoal in skip_goals
                       if sgoal in cdy['forw'][ctxt[1:]]]
    anl_ctxts = [sctxt for sctxt in skip_ctxts
                       if sctxt in cdy['backw'][goal[:1]]]
    analogies = []
    for anl_pair in product(anl_ctxts, anl_goals):
        try:
            bridge = pdy[anl_pair]['joint']
        except KeyError:
            continue
        anl_ctxt, anl_goal = anl_pair
        depart = pdy[(ctxt[:1], ctxt[1:])]['joint'] \
                 * sum(pdy[(bo_ctxt, anl_goal)]['forw']
                       for bo_ctxt in cdy['forw']
                       if len(bo_ctxt) == 2
                          and bo_ctxt[0] == ctxt[0]
                          and anl_goal in cdy['forw'][bo_ctxt]) \
                 * pdy[(ctxt[1:], anl_goal)]['forw']
        print(anl_goal)
        depart = depart / pdy[('_', anl_goal)]
        arrive = pdy[(goal[:1], goal[1:])]['joint'] \
                 * sum(pdy[(anl_ctxt, bo_goal)]['backw']
                       for bo_goal in cdy['backw']
                       if len(bo_goal) == 2
                          and bo_goal[1] == goal[1]
                          and anl_ctxt in cdy['backw'][bo_goal]) \
                 * pdy[(anl_ctxt, goal[:1])]['backw']
        arrive = arrive / pdy[(anl_ctxt, '_')]
        score = depart * bridge * arrive
        analogies.append(((anl_ctxt, anl_goal), (score)))
    analogies.sort(key=lambda x: x[1], reverse=True)
    return analogies

# (end of latest version)

# -----
# Using skip-grams
# -----

def skip_ctxt_dy(train):
    size = 0
    cdy = {'fw': defaultdict(lambda: defaultdict(int)),
           'bw': defaultdict(lambda: defaultdict(int))}
    for sentence in train:
        size += len(sentence) + 1 # end-of-sentence marker
        trigrams = ngrams(sentence, 3)
        # Record cooccurrence data in cdy with forward and backward contexts:
        for trigram in trigrams:
            # Bigram model:          (ctxt2) -> (goal,):
            skip_ctxt_dy_update(cdy, trigram[1:])
            # Trigram model:  (ctxt1, ctxt2) -> (goal,):
            skip_ctxt_dy_update(cdy, trigram)
            # Skipgram model: (ctxt1, _____) -> (goal,):
            skip_ctxt_dy_update(cdy, trigram[:1] + ('_',) + trigram[-1:])
    return (cdy, size)

def skip_ctxt_dy_update(cdy, gram):
    cdy['fw'][gram[1:2]][gram[2:]] += 1

    # Forward contexts:
    cdy['fw'][gram[:i]][gram[i:]] += 1
    # Backward contexts:
    cdy['bw'][gram[i:]][gram[:i]] += 1

def skip_prob_dy(cdy, size):
    pdy = {'lin': {}, 'emb': {}}
    for ctxt in cdy['fw']:
        for goal in cdy['fw'][ctxt]:
            count = cdy['fw'][ctxt][goal]
            joint = count / size
            backw = count / sum(cdy['bw'][goal].values())
            forw  = count / sum(cdy['fw'][ctxt].values())
            pdy['lin'][(ctxt, goal)] = {'jt': joint, 'bw': backw, 'fw': forw}
    for ctxt in cdy['']:
        pass
    return (cdy, pdy)

def skip_model(train):
    return skip_prob_dy(*skip_ctxt_dy(train))

def skiplet_analogies(duplet, model):
    cdy, pdy = model
    ctxt, goal = duplet
    c1, c2, g1, g2 = ctxt[:1], ctxt[1:], goal[:1], goal[1:]
    # Obtaining analogical bridges
    alt_goals = (w[:1] for w in cdy['fw'][c2] if '_' not in w)
    alt_ctxts = (w[-1:] for w in cdy['bw'][g1] if '_' not in w)
    anl_brdgs = set(product(alt_ctxts, alt_goals)).intersection(set(pdy['ln']))
    # Obtaining analogical environments
    anl_depts = (w[-1:] for w in cdy['bw'][c1] if '_' not in w)
    anl_arrvs = (w[:1] for w in cdy['fw'][g2] if '_' not in w)
    anl_envrs = set(product(anl_depts, anl_arrvs)).intersection(set(pdy['eb']))
    analogies = []
    for anl_brdg in anl_brdgs:
        anl_ctxt, anl_goal = anl_brdg
        # Bridge probability
        brdg_prob = pdy[anl_brdg]['jt']         \
                    * pdy[(c2, anl_goal)]['bw'] \
                    * pdy[(anl_ctxt, g1)]['fw']
        # Backward probability ("analogical departures")
        skp_ctxt = ('_',) + anl_ctxt
        anl_depts = set(w[-1:] for w in cdy['bw'][skp_ctxt]
                               if '_' not in w and w in cdy['bw'][c1])
        dept_prob = sum(pdy[(anl_dept, c1)]['bw']         # orig: 'fw'
                        * pdy[(anl_dept, skp_ctxt)]['bw'] # orig: 'bw'
                        for anl_dept in anl_depts)
        # Forward probability ("analogical arrivals")
        skp_goal = anl_goal + ('_',)
        anl_arrvs = set(w[:1] for w in cdy['fw'][skp_goal]
                              if '_' not in w and w in cdy['fw'][g2])
        arrv_prob = sum(pdy[(g2, anl_arrv)]['fw']         # orig: 'bw'
                        * pdy[(skp_goal, anl_arrv)]['fw'] # orig: 'fw'
                        for anl_arrv in anl_arrvs)
        # Analogical probability (of current bridge)
        anl_prob = brdg_prob * dept_prob * arrv_prob
        if anl_prob > 0:
            analogies.append((anl_brdg, anl_prob))
    analogies.sort(key=lambda x: x[1], reverse=True)
    return analogies

"""
