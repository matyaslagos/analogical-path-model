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

def envr_dy(corpus, n):
    ngrams = [x for sentence in corpus for x in ngrams_list(sentence, n)]
    edy = defaultdict(dict)
    ody = defaultdict(set)
    for ngram in ngrams:
        while len(ngram) >= 3:
            try:
                edy[ngram[1:-1]][ngram[:1]][ngram[-1:]] += 1
            except:
                edy[ngram[1:-1]][ngram[:1]] = {ngram[-1:]: 1}
            ody[(ngram[0], ngram[-1])].add(ngram[1:-1])
            ngram = ngram[:-1]
    return (edy, ody)

def cmn_envrs(refr, targ, edy):
    envrs = set()
    for bw_ctxt in edy[refr]:
        if bw_ctxt in edy[targ]:
            for fw_ctxt in edy[refr][bw_ctxt]:
                if fw_ctxt in edy[targ][bw_ctxt]:
                    envrs.add((bw_ctxt[0], fw_ctxt[0]))
    return envrs

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
            continue
    return rem_dy

def bw_lookup(ngram, cdy):
    rem_dy = cdy['bw']
    for word in reversed(ngram):
        try:
            rem_dy = rem_dy[word]
        except KeyError:
            continue
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

def cmn_ctxts(ngram1, ngram2, cdy):
    ctxts = []
    for bw_nb in cmn_bw_nbs(ngram1, ngram2, cdy):
        try:
            for fw_nb in cmn_fw_nbs(bw_nb + ngram1, bw_nb + ngram2, cdy):
                ctxts.append((bw_nb, fw_nb))
        except:
            continue
    return ctxts

def prob_fw(ngram1, ngram2, cdy):
    return fw_count(ngram1 + ngram2, cdy) / fw_count(ngram1, cdy)
    
def prob_bw(ngram1, ngram2, cdy):
    return bw_count(ngram1 + ngram2, cdy) / bw_count(ngram2, cdy)

def prob_jt(ngram1, ngram2, cdy):
    return fw_count(ngram1 + ngram2, cdy) / fw_count((), cdy)

def nb_gen(srce, trgt, cdy):
    bw_score = 0
    for bwnb in cmn_bw_nbs(srce, trgt, cdy):
        try:
            bw_score += prob_bw(bwnb, srce, cdy) * prob_fw(bwnb, trgt, cdy)
        except:
            continue
    fw_score = 0
    for fwnb in cmn_fw_nbs(srce, trgt, cdy):
        try:
            fw_score += prob_fw(srce, fwnb, cdy) * prob_bw(trgt, fwnb, cdy)
        except:
            continue
    return bw_score * fw_score
    

def aps(ctxt, goal, cdy, n=float('inf')):
    rl_grams = defaultdict(float)
    lr_grams = defaultdict(float)
    md_pairs = set()
    for anl_goal in fw_nbs(ctxt, cdy, len(goal)):
        for anl_ctxt in cmn_bw_nbs(anl_goal, goal, cdy, len(ctxt)):
            rl_prob =   prob_bw(anl_ctxt, anl_goal, cdy) \
                      * prob_fw(anl_ctxt,     goal, cdy)
            lr_prob =   prob_fw(anl_ctxt, anl_goal, cdy) \
                      * prob_bw(    ctxt, anl_goal, cdy)
            md_pairs.add((anl_ctxt, anl_goal))
            rl_grams[anl_goal] += rl_prob
            lr_grams[anl_ctxt] += lr_prob
    rr_grams = defaultdict(float)
    for anl_goal in rl_grams:
        if goal[-1][:2] == '</':
            rr_grams[anl_goal] = 1
            continue
        if anl_goal[-1][:2] == '</':
            continue
        for rgt_envr in cmn_fw_nbs(anl_goal, goal, cdy):
            rr_prob =   prob_fw(anl_goal, rgt_envr, cdy) \
                      * prob_bw(    goal, rgt_envr, cdy)
            rr_grams[anl_goal] += rr_prob
    ll_grams = defaultdict(float)
    for anl_ctxt in lr_grams:
        if ctxt[0][:2] == '<s':
            ll_grams[anl_ctxt] = 1
            continue
        if anl_ctxt[0][:2] == '<s':
            continue
        for lft_envr in cmn_bw_nbs(ctxt, anl_ctxt, cdy):
            ll_prob =   prob_fw(lft_envr,     ctxt, cdy) \
                      * prob_bw(lft_envr, anl_ctxt, cdy)
            ll_grams[anl_ctxt] += ll_prob
    anls = defaultdict(float)
    for anl_ctxt, anl_goal in md_pairs:
        r_prob = rr_grams[anl_goal] * rl_grams[anl_goal]
        l_prob = ll_grams[anl_ctxt] * lr_grams[anl_ctxt]
        j_prob = r_prob * l_prob
        if len(ctxt + anl_goal)     < n:
            anls[(ctxt + anl_goal)]     += r_prob
        if len(anl_ctxt + goal)     < n:
            anls[(anl_ctxt + goal)]     += l_prob
        if len(anl_ctxt + anl_goal) < n:
            anls[(anl_ctxt + anl_goal)] += j_prob
    return sorted(list(anls.items()), key=lambda x: x[1], reverse=True)

def aps2(ctxt, goal, cdy, n=float('inf')):
    l_grams = defaultdict(float)
    r_grams = defaultdict(float)
    m_set = set()
    l_set = set()
    r_set = set()
    for anl_goal in fw_nbs(ctxt, cdy, len(goal)):
        for anl_ctxt in cmn_bw_nbs(anl_goal, goal, cdy, len(ctxt)):
            m_set.add((anl_ctxt, anl_goal))
            l_set.add(anl_ctxt)
            r_set.add(anl_goal)
    for l_gram in l_set:
        l_grams[l_gram] = nb_gen(l_gram, ctxt, cdy)
    for r_gram in r_set:
        r_grams[r_gram] = nb_gen(r_gram, goal, cdy)
    anls = defaultdict(float)
    for anl_ctxt, anl_goal in m_set:
        l_prob = l_grams[anl_ctxt]# * prob_jt(anl_ctxt, goal, cdy)
        r_prob = r_grams[anl_goal]# * prob_jt(ctxt, anl_goal, cdy)
        j_prob = r_prob * l_prob# * prob_jt(anl_ctxt, anl_goal, cdy)
        if len(ctxt + anl_goal)     < n:
            anls[(ctxt + anl_goal)]     += r_prob
        if len(anl_ctxt + goal)     < n:
            anls[(anl_ctxt + goal)]     += l_prob
        if len(anl_ctxt + anl_goal) < n:
            anls[(anl_ctxt + anl_goal)] += j_prob
    return sorted(list(anls.items()), key=lambda x: x[1], reverse=True)

def aps2_split(ctxt, goal, cdy, n=float('inf')):
    l_grams = defaultdict(float)
    r_grams = defaultdict(float)
    m_set = set()
    l_set = set()
    r_set = set()
    for anl_goal in fw_nbs(ctxt, cdy, len(goal)):
        for anl_ctxt in cmn_bw_nbs(anl_goal, goal, cdy, len(ctxt)):
            m_set.add((anl_ctxt, anl_goal))
            l_set.add(anl_ctxt)
            r_set.add(anl_goal)
    for l_gram in l_set:
        l_grams[l_gram] = nb_gen(l_gram, ctxt, cdy)
    for r_gram in r_set:
        r_grams[r_gram] = nb_gen(r_gram, goal, cdy)
    anls = defaultdict(float)
    for anl_ctxt, anl_goal in m_set:
        l_prob = l_grams[anl_ctxt]# * prob_jt(anl_ctxt, goal, cdy)
        r_prob = r_grams[anl_goal]# * prob_jt(ctxt, anl_goal, cdy)
        j_prob = r_prob * l_prob# * prob_jt(anl_ctxt, anl_goal, cdy)
        if len(ctxt + anl_goal)     < n:
            anls[(ctxt, anl_goal)]     += r_prob
        if len(anl_ctxt + goal)     < n:
            anls[(anl_ctxt, goal)]     += l_prob
        if len(anl_ctxt + anl_goal) < n:
            anls[(anl_ctxt, anl_goal)] += j_prob
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


def rec_parse2(gram, cdy, anl_dy=None, n=float('inf')):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    if gram in anl_dy:
        return anl_dy
    # End recursion when we reach unigrams
    if len(gram) == 1:
        anls = [{'path': gram, 'score': 1, 'split': gram[0], 'subst': gram[0]}]
        anl_dy[gram] = anls
        return anl_dy
    # Recursive step
    splits = ((gram[:i], gram[i:]) for i in range(1,len(gram)))
    anls = []
    for ctxt, goal in splits:
        split_dy = defaultdict(float)
        # Recursive calls
        rec_ctxts = rec_parse2(ctxt, cdy, anl_dy, n)[ctxt][:30]
        rec_goals = rec_parse2(goal, cdy, anl_dy, n)[goal][:30]
        for rec_ctxt in rec_ctxts:
            for rec_goal in rec_goals:
                paths = aps(rec_ctxt['path'], rec_goal['path'], cdy, n)[:30]
                for path, path_score in paths:
                    score = path_score * rec_ctxt['score'] * rec_goal['score']
                    split = (rec_ctxt['split'], rec_goal['split'])
                    subst = (rec_ctxt['path'], rec_goal['path'])
                    anls.append({'path': path, 'score': score,
                                 'split': split, 'subst': subst})
    anls.sort(reverse=True, key=lambda x: x['score'])
    anl_dy[gram] = anls
    return anl_dy

def rec_parse3(gram, cdy, anl_dy=None, n=float('inf')):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    if gram in anl_dy:
        return anl_dy
    # End recursion when we reach unigrams
    if len(gram) == 1:
        anls = [{'path': gram, 'score': 1, 'split': gram[0]}]
        anl_dy[gram] = anls
        return anl_dy
    # Recursive step
    splits = ((gram[:i], gram[i:]) for i in range(1,len(gram)))
    anls = []
    for ctxt, goal in splits:
        split_dy = defaultdict(float)
        # Recursive calls
        rec_ctxts = rec_parse3(ctxt, cdy, anl_dy, n)[ctxt][:30]
        rec_goals = rec_parse3(goal, cdy, anl_dy, n)[goal][:30]
        for rec_ctxt in rec_ctxts:
            for rec_goal in rec_goals:
                paths = aps(rec_ctxt['path'], rec_goal['path'], cdy, n)[:30]
                for path, path_score in paths:
                    score = path_score * rec_ctxt['score'] * rec_goal['score']
                    split_dy[path] += score
        split_score = sum(split_dy.values())
        for path, score in split_dy.items():
            anls.append({'path': path, 'score': score,
                         'split': (ctxt, goal)})
    anls.sort(reverse=True, key=lambda x: x['score'])
    anl_dy[gram] = anls
    return anl_dy

def rec_parse_split(gram, cdy, anl_dy=None, n=float('inf')):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    if gram in anl_dy:
        return anl_dy
    # End recursion when we reach unigrams
    if len(gram) == 1:
        anls = [{'path': gram, 'score': 1, 'tree': gram[0], 'tree type': gram[0]}]
        anl_dy[gram] = anls
        return anl_dy
    # Recursive step
    splits = ((gram[:i], gram[i:]) for i in range(1,len(gram)))
    anls = []
    for ctxt, goal in splits:
        split_dy = defaultdict(float)
        # Recursive calls
        rec_ctxts = rec_parse_split(ctxt, cdy, anl_dy, n)[ctxt]
        rec_goals = rec_parse_split(goal, cdy, anl_dy, n)[goal]
        for rec_ctxt in rec_ctxts:
            for rec_goal in rec_goals:
                tree_type = (rec_ctxt['tree type'], rec_goal['tree type'])
                abts = []
                paths = aps2_split(rec_ctxt['path'], rec_goal['path'], cdy, n)[:30]
                for path, path_score in paths:
                    score = path_score * rec_ctxt['score'] * rec_goal['score']
                    tree = ((' '.join(path[0]), rec_ctxt['tree']),
                            (' '.join(path[1]), rec_goal['tree']))
                    abts.append({'path': path[0] + path[1], 'score': score,
                                 'tree': tree, 'tree type': tree_type})
                abts.sort(reverse=True, key=lambda x: x['score'])
                anls.append({'tree type': abts})
    anls = sorted(anls, reverse=True,
                  key=lambda x: sum(item['score'] for item in x['tree type']))[0]['tree type'][:10]
    anls.sort(reverse=True, key=lambda x: x['score'])
    anl_dy[gram] = anls
    return anl_dy

def rec_parse_tree(tree, cdy, anl_dy=None, n=float('inf')):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    if tree in anl_dy:
        return anl_dy
    # End recursion when we reach unigrams
    if len(tree) == 1:
        anls = [{'path': tree, 'score': 1, 'split': tree[0], 'subst': tree[0]}]
        anl_dy[tree] = anls
        return anl_dy
    # Recursive step
    ctxt, goal = tree[0], tree[1]
    anls = []
    # Recursive calls
    rec_ctxts = rec_parse_tree(ctxt, cdy, anl_dy, n)[ctxt][:10]
    rec_goals = rec_parse_tree(goal, cdy, anl_dy, n)[goal][:10]
    for rec_ctxt in rec_ctxts:
        for rec_goal in rec_goals:
            paths = aps(rec_ctxt['path'], rec_goal['path'], cdy, n)[:20]
            for path, path_score in paths:
                score = path_score * rec_ctxt['score'] * rec_goal['score']
                split = (rec_ctxt['split'], rec_goal['split'])
                subst = (rec_ctxt['path'], rec_goal['path'])
                anls.append({'path': path, 'score': score,
                             'split': split, 'subst': subst})
    anls.sort(reverse=True, key=lambda x: x['score'])
    anl_dy[tree] = anls
    return anl_dy

def rec_parse_tree2(tree, cdy, anl_dy=None, n=float('inf')):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    if tree in anl_dy:
        return anl_dy
    # End recursion when we reach unigrams
    if len(tree) == 1:
        anls = [{'path': tree, 'score': 1}]
        anl_dy[tree] = anls
        return anl_dy
    # Recursive step
    ctxt, goal = tree[0], tree[1]
    anls = defaultdict(float)
    # Recursive calls
    rec_ctxts = rec_parse_tree2(ctxt, cdy, anl_dy, n)[ctxt][:20]
    rec_goals = rec_parse_tree2(goal, cdy, anl_dy, n)[goal][:20]
    for rec_ctxt in rec_ctxts:
        for rec_goal in rec_goals:
            paths = aps(rec_ctxt['path'], rec_goal['path'], cdy, n)
            for path, path_score in paths:
                score = path_score * rec_ctxt['score'] * rec_goal['score']
                anls[path] += score
    anls = [{'path': path, 'score': anls[path]} for path in anls]
    anls.sort(reverse=True, key=lambda x: x['score'])
    anl_dy[tree] = anls
    return anl_dy

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

def ppt_annot(tree, coords=(), ones_out=[], annots=[], annot_lengths=[-2]):
    if isinstance(tree, str):
        print(bars_annot(coords, ones_out, annots, annot_lengths) + '╶─ ' + tree)
    else:
        ppt_annot(tree[0][1], coords + (0,), ones_out,
                  annots + [tree[0][0]], annot_lengths + [len(tree[0][0])])
        ppt_annot(tree[1][1], coords + (1,), ones_out + [len(coords) + 1],
                  annots + [tree[1][0]], annot_lengths + [len(tree[1][0])])

def bars_annot(coords, ones_out, annots, annot_lengths):
    bar_tup = ()
    stop = False
    for i, coord in enumerate(reversed(coords)):
        annot_length = annot_lengths[:len(coords)][-(i+1)] + 2
        annot = annots[-(i+1)]
        annot_form = '(' + annot + ')'
        if coord == 0 and not stop:
            if i != 0:
                bar_tup += (annot_form, '┌╴')
            else:
                bar_tup += (annot_form, '┌╴')
        elif not stop:
            stop = True
            if i != 0:
                bar_tup += (annot_form, '└╴', annot_length * ' ')
            else:
                bar_tup += (annot_form, '└╴', annot_length * ' ')
        elif (len(coords) - i) in ones_out:
            bar_tup += ('  ', annot_length * ' ')
        else:
            bar_tup += ('│ ', annot_length * ' ')
    return ''.join(reversed(bar_tup))

def sum_paths(path_dict):
    sums_dy = defaultdict(float)
    for item in path_dict:
        #sums_dy[item['subst'][0] + item['subst'][1]]
        #sums_dy[item['path']]
        sums_dy[item['path']] += item['score']
    return sorted(list(sums_dy.items()), key=lambda x: x[1], reverse=True)