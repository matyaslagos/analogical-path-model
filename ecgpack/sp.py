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

# ------------- #
# Trigram model #
# ------------- #

def ctxt_dy3(train):
    cdy = {'fw': defaultdict(lambda: defaultdict(int)),
           'bw': defaultdict(lambda: defaultdict(int))}
    for sentence in train:
        trigrams = ngrams_list(sentence, 3)
        for trigram in trigrams:
            # Record ['the' -> 'king was'] and ['the king' -> 'was']
            for i in range(2):
                cdy['fw'][trigram[:i+1]][trigram[i+1:]] += 1
                cdy['bw'][trigram[i+1:]][trigram[:i+1]] += 1
            # Record ['king' -> 'was']:
            cdy['fw'][trigram[1:2]][trigram[2:]] += 1
            cdy['bw'][trigram[2:]][trigram[1:2]] += 1
    # Convert cdy from defaultdict to dict (dr is direction i.e. 'fw' or 'bw')
    cdy = {dr: {ctxt: dict(cdy[dr][ctxt]) for ctxt in cdy[dr]} for dr in cdy}
    return cdy

def ctxt_dy3(train):
    cdy = {'fw': defaultdict(lambda: defaultdict(int)),
           'bw': defaultdict(lambda: defaultdict(int))}
    for sentence in train:
        trigrams = ngrams_list(sentence, 3)
        for trigram in trigrams:
            # Record ['the' -> 'king was'] and ['the king' -> 'was']
            for i in range(2):
                cdy['fw'][trigram[:i+1]][trigram[i+1:]] += 1
                cdy['bw'][trigram[i+1:]][trigram[:i+1]] += 1
            # Record ['king' -> 'was']:
            cdy['fw'][trigram[1:2]][trigram[2:]] += 1
            cdy['bw'][trigram[2:]][trigram[1:2]] += 1
    # Convert cdy from defaultdict to dict (dr is direction i.e. 'fw' or 'bw')
    cdy = {dr: {ctxt: dict(cdy[dr][ctxt]) for ctxt in cdy[dr]} for dr in cdy}
    return cdy

def prob_dy3(train):
    total_freq = sum(len(sentence) + 1 for sentence in train)
    cdy = ctxt_dy3(train)
    pdy = defaultdict(lambda: defaultdict(dict))
    for ctxt in cdy['fw']:
        ctxt_freq = sum(cdy['fw'][ctxt][goal] for goal in cdy['fw'][ctxt]
                                              if len(goal) == 1)
        for goal in cdy['fw'][ctxt]:
            joint_freq = cdy['fw'][ctxt][goal]
            goal_freq = sum(cdy['bw'][goal][ctxt] for ctxt in cdy['bw'][goal]
                                                  if len(ctxt) == 1)
            probs = {'fw': joint_freq / ctxt_freq,
                     'bw': joint_freq / goal_freq,
                     'jt': joint_freq / total_freq}
            pdy[ctxt][goal] = probs
    pdy = {ctxt: {goal: pdy[ctxt][goal] for goal in pdy[ctxt]} for ctxt in pdy}
    return (cdy, pdy)

def aps3(ctxt, goal, model):
    '''
    # If input is duplet_string, e.g. 'the ; old king'
    ctxt, goal = str2dpl(duplet_string)
    '''
    cdy, pdy = model
    # Middle analogies
    lr_items = defaultdict(float)
    rl_items = defaultdict(float)
    middle_paths = set()
    for anl_goal in cdy['fw'][ctxt]:
        if len(cdy['bw'][anl_goal]) < len(cdy['bw'][goal]):
            for anl_ctxt in cdy['bw'][anl_goal]:
                if goal in cdy['fw'][anl_ctxt]:
                    anl_prob =   pdy[ctxt][anl_goal]['fw']     \
                               * pdy[anl_ctxt][anl_goal]['bw'] \
                               * pdy[anl_ctxt][goal]['fw']
                    lr_items[anl_ctxt] += anl_prob
                    rl_items[anl_goal] += anl_prob
                    middle_paths.add((anl_ctxt, anl_goal))
        else:
            for anl_ctxt in cdy['bw'][goal]:
                if anl_goal in cdy['fw'][anl_ctxt]:
                    anl_prob =   pdy[ctxt][anl_goal]['fw']     \
                               * pdy[anl_ctxt][anl_goal]['bw'] \
                               * pdy[anl_ctxt][goal]['fw']
                    lr_items[anl_ctxt] += anl_prob
                    rl_items[anl_goal] += anl_prob
                    middle_paths.add((anl_ctxt, anl_goal))
    # Left analogies
    ll_items = defaultdict(float)
    for anl_envr in cdy['bw'][ctxt]:
        if len(cdy['fw'][anl_envr]) < len(cdy['bw'][goal]):
            for anl_ctxt in cdy['fw'][anl_envr]:
                if '</s>' in anl_ctxt:
                    continue
                if goal in cdy['fw'][anl_ctxt]:
                    anl_prob =   pdy[anl_envr][ctxt]['bw']     \
                               * pdy[anl_envr][anl_ctxt]['fw'] \
                               * pdy[anl_ctxt][goal]['fw']
                    ll_items[anl_ctxt] += anl_prob
        else:
            for anl_ctxt in cdy['bw'][goal]:
                if '<s>' in anl_ctxt:
                    continue
                if anl_envr in cdy['bw'][anl_ctxt]:
                    anl_prob =   pdy[anl_envr][ctxt]['bw']     \
                               * pdy[anl_envr][anl_ctxt]['fw'] \
                               * pdy[anl_ctxt][goal]['fw']
                    ll_items[anl_ctxt] += anl_prob
    # Right analogies
    rr_items = defaultdict(float)
    for anl_goal in cdy['fw'][ctxt]:
        if '</s>' in anl_goal:
            continue
        if len(cdy['fw'][anl_goal]) < len(cdy['fw'][goal]):
            for anl_envr in cdy['fw'][anl_goal]:
                if goal in cdy['bw'][anl_envr]:
                    anl_prob =   pdy[ctxt][anl_goal]['fw']     \
                               * pdy[anl_goal][anl_envr]['fw'] \
                               * pdy[goal][anl_envr]['bw']
                    rr_items[anl_goal] += anl_prob
        else:
            for anl_envr in cdy['fw'][goal]:
                if anl_goal in cdy['bw'][anl_envr]:
                    anl_prob =   pdy[ctxt][anl_goal]['fw']     \
                               * pdy[anl_goal][anl_envr]['fw'] \
                               * pdy[goal][anl_envr]['bw']
                    rr_items[anl_goal] += anl_prob
    # Best analogies
    substs = defaultdict(float)
    for l_item, r_item in middle_paths:
        # Exclude long paths:
        if len(l_item + r_item) > 2:
            continue
        l_prob = (sum(cdy['fw'][l_item].values()) / 2)
        r_prob = (sum(cdy['bw'][r_item].values()) / 2)
        substs[l_item + r_item] +=   (lr_items[l_item] * ll_items[l_item]) \
                                   * (rl_items[r_item] * rr_items[r_item]) #\
                                   #/ (l_prob * r_prob)
        if len(ctxt) == 1:
            substs[ctxt + r_item]   += (rr_items[r_item] / r_prob) * rl_items[r_item]
        if len(goal) == 1:
            substs[l_item + goal]   += (ll_items[l_item] / l_prob) * lr_items[l_item]
    substs = [x for x in substs.items() if x[1] > 0]
    substs.sort(key=lambda x: x[1], reverse=True)
    return substs

def trig_ps(trigram_string, model):
    trigram = tuple(trigram_string.split())
    cdy, pdy = model
    splits = [(trigram[:i+1], trigram[i+1:] ) for i in range(2)]
    anl_paths = defaultdict(float)
    anl_dy = defaultdict(lambda: defaultdict(float))
    for ctxt, goal in splits:
        if len(ctxt) > 1:
            rec_ctxts = aps3(ctxt[:1], ctxt[1:], model)
        else:
            rec_ctxts = [(ctxt, 1)]
        if len(goal) > 1:
            rec_goals = aps3(goal[:1], goal[1:], model)
        else:
            rec_goals = [(goal, 1)]
        for ctxt_sub, ctxt_score in rec_ctxts:
            for goal_sub, goal_score in rec_goals:
                paths = aps3(ctxt_sub, goal_sub, model)
                for path, path_score in paths:
                    anl_score = path_score * ctxt_score * goal_score
                    anl_paths[path] += anl_score
                    anl_dy[(ctxt, goal)][(ctxt_sub, goal_sub)] += anl_score
    path_dy = sorted(list(anl_paths.items()), key=lambda x: x[1], reverse=True)
    return path_dy

def rec_parse3(gram, model, anl_dy=None):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    try:
        return (anl_dy, anl_dy[gram])
    except KeyError:
        pass
    # End recursion when we reach unigrams
    if len(gram) == 1:
        anl_dy[gram] = [((gram, gram), 1)]
        return (anl_dy, [((gram, gram), 1)])
    # Recursive step
    splits = ((gram[:i], gram[i:]) for i in range(1,len(gram)))
    split_anls = {}
    anl_path_dy = defaultdict(float)
    for ctxt, goal in splits:
        split_anls[(ctxt, goal)] = defaultdict(float)
        # Recursive calls
        ctxt_subs = rec_parse3(ctxt, model, anl_dy)[1]
        goal_subs = rec_parse3(goal, model, anl_dy)[1]
        for ctxt_sub, ctxt_score in ctxt_subs:
            for goal_sub, goal_score in goal_subs:
                paths = aps3(ctxt_sub[0], goal_sub[0], model)[:5]
                for path, path_score in paths:
                    score = path_score #* ctxt_score * goal_score
                    split_anls[(ctxt, goal)][path] += score
                    anl_path_dy[(path, (ctxt, goal))] += score
    anls = sorted(list(anl_path_dy.items()), key=lambda x: x[1], reverse=True)[:5]
    anl_dy[gram] = anls
    return (anl_dy, anls)

def str2dpl(duplet_string):
    duplet_list = duplet_string.split()
    return (tuple(duplet_list[:duplet_list.index(';')]),
            tuple(duplet_list[duplet_list.index(';')+1:]))

def ctxt_dy(corpus, n):
    ngrams = [x for sentence in corpus for x in ngrams_list(sentence, n)]
    ngrams_rev = [tuple(reversed(ngram)) for ngram in ngrams]
    cdy = {}
    cdy['fw'] = ctxt_dy_aux(ngrams, n)
    cdy['bw'] = ctxt_dy_aux(ngrams_rev, n)
    return cdy

def ctxt_dy_aux(ngrams, n):
    # End recursion at unigrams, just return dict with counts
    if n == 1:
        cdy = {ngram[0]: {'#': ngrams.count(ngram)} for ngram in set(ngrams)}
        cdy['#'] = sum(cdy[word]['#'] for word in cdy)
        return cdy
    # Sort tails of ngrams according to head word
    cdy_lists = defaultdict(list)
    for ngram in ngrams:
        cdy_lists[ngram[0]].append(ngram[1:])
    # Recursively compute context dict for each head word
    cdy = {word: ctxt_dy_aux(cdy_lists[word], n-1) for word in cdy_lists}
    cdy['#'] = sum(cdy[word]['#'] for word in cdy)
    return cdy

def fw_count(ngram, cdy):
    rem_dy = cdy['fw']
    for word in ngram:
        try:
            rem_dy = rem_dy[word]
        except KeyError:
            return 0
    return rem_dy['#']

def bw_count(ngram, cdy):
    rem_dy = cdy['bw']
    for word in reversed(ngram):
        try:
            rem_dy = rem_dy[word]
        except KeyError:
            return 0
    return rem_dy['#']

def fw_lookup(ngram, cdy):
    rem_dy = cdy['fw']
    for word in ngram:
        try:
            rem_dy = rem_dy[word]
        except KeyError:
            print('Could not forw. find', ngram)
    return rem_dy

def bw_lookup(ngram, cdy):
    rem_dy = cdy['bw']
    for word in reversed(ngram):
        try:
            rem_dy = rem_dy[word]
        except KeyError:
            print('Could not backw. find', ngram)
    return rem_dy

def fw_nbs(ngram, cdy, n=float('inf')):
    return fw_nbs_aux(fw_lookup(ngram, cdy), n)

def fw_nbs_aux(rem_dy, n, path=()):
    if len(rem_dy) == 1 or len(path) >= n:
        return []
    nbs = []
    for word in rem_dy:
        if word != '#':
            if word[-1] == '>':
                nbs.append(path + (word,))
                continue
            path += (word,)
            nbs.append(path)
            nbs += fw_nbs_aux(rem_dy[word], n, path)
            path = path[:-1]
    return nbs

def bw_nbs(ngram, cdy, n=float('inf')):
    return bw_nbs_aux(bw_lookup(ngram, cdy), n)

def bw_nbs_aux(rem_dy, n, path=()):
    if len(rem_dy) == 1 or len(path) >= n:
        return []
    nbs = []
    for word in rem_dy:
        if word != '#':
            if word[-1] == '>':
                nbs.append(tuple(reversed(path + (word,))))
                continue
            path += (word,)
            nbs.append(tuple(reversed(path)))
            nbs += bw_nbs_aux(rem_dy[word], n, path)
            path = path[:-1]
    return nbs

def cmn_fw_nbs(ngram1, ngram2, cdy, n=float('inf')):
    return cmn_fw_nbs_aux(fw_lookup(ngram1, cdy), fw_lookup(ngram2, cdy), n)

def cmn_fw_nbs_aux(rem_dy1, rem_dy2, n, path=()):
    if len(rem_dy1) == 1 or len(rem_dy2) == 1 or len(path) >= n:
        return []
    rem_dys = (rem_dy1, rem_dy2)
    index = len(rem_dy1) <= len(rem_dy2)
    nbs = []
    for word in rem_dys[1-index]:
        if word != '#' and word in rem_dys[index]:
            if word[-1] == '>':
                nbs.append(path + (word,))
                continue
            path += (word,)
            nbs.append(path)
            nbs += cmn_fw_nbs_aux(rem_dy1[word], rem_dy2[word], n, path)
            path = path[:-1]
    return nbs

def cmn_bw_nbs(ngram1, ngram2, cdy, n=float('inf')):
    return cmn_bw_nbs_aux(bw_lookup(ngram1, cdy), bw_lookup(ngram2, cdy), n)

def cmn_bw_nbs_aux(rem_dy1, rem_dy2, n, path=()):
    if len(rem_dy1) == 1 or len(rem_dy2) == 1 or len(path) >= n:
        return []
    rem_dys = (rem_dy1, rem_dy2)
    index = len(rem_dy1) <= len(rem_dy2)
    nbs = []
    for word in rem_dys[1-index]:
        if word != '#' and word in rem_dys[index]:
            if word[-1] == '>':
                nbs.append(tuple(reversed(path + (word,))))
                continue
            path += (word,)
            nbs.append(tuple(reversed(path)))
            nbs += cmn_bw_nbs_aux(rem_dy1[word], rem_dy2[word], n, path)
            path = path[:-1]
    return nbs

def aps(ctxt, goal, cdy, n=float('inf')):
    rl_grams = defaultdict(float)
    lr_grams = defaultdict(float)
    md_pairs = defaultdict(float)
    for anl_goal in fw_nbs(ctxt, cdy, len(goal)):
        for anl_ctxt in cmn_bw_nbs(anl_goal, goal, cdy, len(ctxt)):
            prob =   prob_fw(ctxt, anl_goal, cdy)     \
                   * prob_bw(anl_ctxt, anl_goal, cdy) \
                   * prob_fw(anl_ctxt, goal, cdy)
            md_pairs[(anl_ctxt, anl_goal)] += prob
            rl_grams[anl_goal] += prob
            lr_grams[anl_ctxt] += prob
    rr_grams = defaultdict(float)
    for anl_goal in rl_grams:
        if goal[-1][:2] == '</':
            rr_grams[anl_goal] = 1
            continue
        if anl_goal[-1][:2] == '</':
            continue
        for rgt_envr in cmn_fw_nbs(anl_goal, goal, cdy):
            prob =   prob_fw(ctxt, anl_goal, cdy)     \
                   * prob_fw(anl_goal, rgt_envr, cdy) \
                   * prob_bw(goal, rgt_envr, cdy)
            rr_grams[anl_goal] += prob
    ll_grams = defaultdict(float)
    for anl_ctxt in lr_grams:
        if ctxt[0][:2] == '<s':
            ll_grams[anl_ctxt] = 1
            continue
        if anl_ctxt[0][:2] == '<s':
            continue
        for lft_envr in cmn_bw_nbs(ctxt, anl_ctxt, cdy):
            prob =   prob_bw(lft_envr, ctxt, cdy)     \
                   * prob_fw(lft_envr, anl_ctxt, cdy) \
                   * prob_fw(anl_ctxt, goal, cdy)
            ll_grams[anl_ctxt] += prob
    anls = defaultdict(float)
    for anl_ctxt, anl_goal in md_pairs:
        r_prob = rl_grams[anl_goal] * rr_grams[anl_goal]
        l_prob = lr_grams[anl_ctxt] * ll_grams[anl_ctxt]
        j_prob = r_prob * l_prob# / prob_jt(anl_ctxt, anl_goal, cdy)
        if len(ctxt + anl_goal)     < n:
            anls[(ctxt + anl_goal)]     += r_prob
        if len(anl_ctxt + goal)     < n:
            anls[(anl_ctxt + goal)]     += l_prob
        if len(anl_ctxt + anl_goal) < n:
            anls[(anl_ctxt + anl_goal)] += j_prob 
    return sorted(list(anls.items()), key=lambda x: x[1], reverse=True)

def rec_parse(gram, cdy, anl_dy=None, n=float('inf')):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    try:
        return (anl_dy, anl_dy[gram])
    except KeyError:
        pass
    # End recursion when we reach unigrams
    if len(gram) == 1:
        anl_dy[gram] = [((gram, gram), 1)]
        return (anl_dy, [((gram, gram), 1)])
    # Recursive step
    splits = ((gram[:i], gram[i:]) for i in range(1,len(gram)))
    split_anls = {}
    anl_path_dy = defaultdict(float)
    for ctxt, goal in splits:
        split_anls[(ctxt, goal)] = defaultdict(float)
        # Recursive calls
        ctxt_subs = rec_parse(ctxt, cdy, anl_dy, n)[1]
        goal_subs = rec_parse(goal, cdy, anl_dy, n)[1]
        for ctxt_sub, ctxt_score in ctxt_subs:
            for goal_sub, goal_score in goal_subs:
                paths = aps(ctxt_sub[0], goal_sub[0], cdy, n)[:10]
                for path, path_score in paths:
                    score = path_score * ctxt_score * goal_score
                    split_anls[(ctxt, goal)][path] += score
                    anl_path_dy[(path, (ctxt, goal))] += score
    anls = sorted(list(anl_path_dy.items()), key=lambda x: x[1], reverse=True)[:10]
    anl_dy[gram] = anls
    return (anl_dy, anls)

def prob_fw(ngram1, ngram2, cdy):
    return fw_count(ngram1 + ngram2, cdy) / fw_count(ngram1, cdy)
    
def prob_bw(ngram1, ngram2, cdy):
    return bw_count(ngram1 + ngram2, cdy) / bw_count(ngram2, cdy)

def prob_jt(ngram1, ngram2, cdy):
    return fw_count(ngram1 + ngram2, cdy) / fw_count((), cdy)

def rec_parse2(gram, cdy, anl_dy=None, n=float('inf')):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    try:
        return (anl_dy, anl_dy[gram][0])
    except KeyError:
        pass
    # End recursion when we reach unigrams
    if len(gram) == 1:
        anl_dy[gram] = ([((gram, gram), 1)], [((gram, gram), 1)])
        return (anl_dy, [((gram, gram), 1)])
    # Recursive step
    splits = ((gram[:i], gram[i:]) for i in range(1,len(gram)))
    split_anls = defaultdict(lambda: defaultdict(float))
    anl_path_dy = defaultdict(float)
    for ctxt, goal in splits:
        # Recursive calls
        ctxt_subs = rec_parse2(ctxt, cdy, anl_dy, n)[1]
        goal_subs = rec_parse2(goal, cdy, anl_dy, n)[1]
        for ctxt_sub, ctxt_score in ctxt_subs:
            for goal_sub, goal_score in goal_subs:
                paths = aps(ctxt_sub[0], goal_sub[0], cdy, n)[:10]
                for path, path_score in paths:
                    score = path_score * ctxt_score * goal_score
                    split_anls[(ctxt_sub[1], goal_sub[1])][(path,
                                              (ctxt_sub[0], goal_sub[0]))] += score
                    anl_path_dy[(path,
                                 (ctxt_sub[1], goal_sub[1]),
                                 (ctxt_sub[0], goal_sub[0]))] += score
    '''
    best_split = sorted([split_anls[key] for key in split_anls],
                        key=lambda x: sum(x.values()), reverse=True)[0]
    '''
    
    best_anls = sorted(list(anl_path_dy.items()), key=lambda x: x[1], reverse=True)[:10]
    all_anls  = split_anls
    anl_dy[gram] = (best_anls, all_anls)
    return (anl_dy, best_anls)

def ppt(tree, coords=(), ones_out=[]):
    if isinstance(tree, str):
        print(bars(coords, ones_out) + tree)
    else:
        ppt(tree[0], coords + (0,), ones_out)
        ppt(tree[1], coords + (1,), ones_out + [len(coords) + 1])

def bars(coords, ones_out):
    bar_tup = ()
    stop = False
    for i, coord in enumerate(reversed(coords)):
        if coord == 0 and not stop:
            if i != 0:
                bar_tup += ('┌',)
            else:
                bar_tup += ('┌ ',)
        elif not stop:
            stop = True
            if i != 0:
                bar_tup += ('└',)
            else:
                bar_tup += ('└ ',)
        elif (len(coords) - i) in ones_out:
            bar_tup += (' ',)
        else:
            bar_tup += ('│',)
    return ''.join(reversed(bar_tup))