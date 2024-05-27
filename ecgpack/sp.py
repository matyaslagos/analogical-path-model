from ap import txt2list, train_test, ngrams_list
from collections import defaultdict

def ctxt_dy(train):
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

def prob_dy(train):
    cdy = ctxt_dy(train)
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

def aps(bigram, model):
    cdy, pdy = model
    ctxt, goal = (bigram.split()[0],), (bigram.split()[1],)
    anl_paths = []
    for anl_goal in cdy['fw'][ctxt]:
        for anl_ctxt in cdy['bw'][anl_goal]:
            if goal in pdy[anl_ctxt]:
                anl_prob = pdy[ctxt][anl_goal]['bw']       \
                           * pdy[anl_ctxt][anl_goal]['jt'] \
                           * pdy[anl_ctxt][goal]['fw']
                anl_paths.append((anl_ctxt + anl_goal, anl_prob))
    anl_paths.sort(key=lambda x: x[1], reverse=True)
    return anl_paths

def test_aps(bigram, model):
    # Z-shaped paths: joint or steps
    cdy, pdy = model
    ctxt, goal = (bigram.split()[0],), (bigram.split()[1],)
    anl_paths = []
    for anl_goal in cdy['fw'][ctxt]:
        for anl_ctxt in cdy['bw'][anl_goal]:
            if goal in pdy[anl_ctxt]:
                
                anl_prob = pdy[ctxt][anl_goal]['jt']       \
                           * pdy[anl_ctxt][anl_goal]['jt'] \
                           * pdy[anl_ctxt][goal]['jt']
                '''
                anl_prob = pdy[ctxt][anl_goal]['bw']       \
                           * pdy[anl_ctxt][anl_goal]['jt'] \
                           * pdy[anl_ctxt][goal]['fw']
                '''
                anl_paths.append((anl_ctxt + anl_goal, anl_prob))
    anl_paths.sort(key=lambda x: x[1], reverse=True)
    return anl_paths

def test_aps2(bigram, model):
    # Left environments: joint or steps
    cdy, pdy = model
    ctxt, goal = (bigram.split()[0],), (bigram.split()[1],)
    anl_paths = defaultdict(float)
    for anl_envr in cdy['bw'][ctxt]:
        for anl_ctxt in cdy['fw'][anl_envr]:
            if anl_ctxt != ('</s>',) and goal in pdy[anl_ctxt]:
                
                anl_prob = pdy[anl_envr][ctxt]['fw']       \
                           * pdy[anl_envr][anl_ctxt]['jt'] \
                           * pdy[anl_ctxt][goal]['fw']
                '''
                anl_prob = pdy[anl_envr][ctxt]['jt']       \
                           * pdy[anl_envr][anl_ctxt]['jt'] \
                           * pdy[anl_ctxt][goal]['jt']
                '''
                anl_paths[anl_ctxt + goal] += anl_prob
    anl_paths = list(anl_paths.items())
    anl_paths.sort(key=lambda x: x[1], reverse=True)
    return anl_paths

def test_aps3(bigram, model):
    # Right environments: joint or steps
    cdy, pdy = model
    ctxt, goal = (bigram.split()[0],), (bigram.split()[1],)
    anl_paths = defaultdict(float)
    for anl_goal in cdy['fw'][ctxt]:
        if anl_goal != ('</s>',):
            for anl_envr in cdy['fw'][anl_goal]:
                if goal in cdy['bw'][anl_envr]:
                    
                    anl_prob = pdy[ctxt][anl_goal]['bw']       \
                               * pdy[anl_goal][anl_envr]['jt'] \
                               * pdy[goal][anl_envr]['bw']
                    '''
                    anl_prob = pdy[ctxt][anl_goal]['jt']       \
                               * pdy[anl_goal][anl_envr]['jt'] \
                               * pdy[goal][anl_envr]['jt']
                    '''
                    anl_paths[ctxt + anl_goal] += anl_prob
    anl_paths = list(anl_paths.items())
    anl_paths.sort(key=lambda x: x[1], reverse=True)
    return anl_paths