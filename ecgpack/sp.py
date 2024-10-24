from ap import txt2list
from collections import defaultdict, Counter
import random
import math
import itertools
from pprint import pp

# -----
# Model building algorithms
# -----

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
    train = [list(sentence) for sentence in sentences[:n]]
    vocab = {word for sentence in train for word in sentence}
    test = [tuple(sentence) for sentence in sentences[n:]
                            if set(sentence).issubset(vocab)]
    return (train, test)

def ngrams_list(sentence, n):
    beg = ['<s{}>'.format(i) for i in range(1, n)]
    end = ['</s{}>'.format(i) for i in range(1, n)]
    sentence = beg + sentence + end
    ends = [tuple(['<s0>'] + beg), tuple(end + ['</s{}>'.format(n)])]
    return list(zip(*(sentence[i:len(sentence)-n+i+1] for i in range(n)))) + ends

def nesting_dict():
    return defaultdict(nesting_dict)

def slot_ctxt_dy(corpus):
    grams = [tuple(['<s>'] + sentence + ['</s>']) for sentence in corpus]
    #cdy = slot_ctxt_dy_aux_stars_ctxts(grams)
    gdy = slot_ctxt_dy_aux_stars_goals(grams)
    return gdy

def slot_ctxt_dy_aux(grams):
    # Initialise context dictionary
    ctxt_dict = nesting_dict()
    # Add new dictionary structure for each gram
    for gram in grams:
        seen_slot_grams = set()
        for slot_gram, filler in slot_grams(gram):
            fill = False
            traversing_dict = ctxt_dict
            path = ()
            for head in slot_gram:
                path += (head,)
                # Mark that we've seen a slot and filler should be recorded
                if head == '_':
                    fill = True
                if head not in traversing_dict:
                    traversing_dict[head] = {'$': nesting_dict()}
                    traversing_dict[head]['$']['#'] = 0
                # If first time seeing this prefix in this gram, add count
                if path not in seen_slot_grams:
                    traversing_dict[head]['$']['#'] += 1
                    seen_slot_grams.add(path)
                # Add filler
                if fill:
                    # Record filler in nested dict for easier common-
                    # neighbour search
                    filler_dict = traversing_dict[head]['$']
                    for i, filler_head in enumerate(filler):
                        if i == len(filler) - 1:
                            try:
                                filler_dict[filler_head]['#'] += 1
                            except TypeError:
                                filler_dict[filler_head]['#'] = 1
                        filler_dict = filler_dict[filler_head]
                traversing_dict = traversing_dict[head]
    return ctxt_dict

def slot_ctxt_dy_aux_stars_ctxts(grams):
    # Initialise context dictionary
    ctxt_dict = nesting_dict()
    # Add new dictionary structure for each gram
    for gram in grams:
        seen_slot_grams = set()
        for slot_gram, fillers in slot_plus_grams_ctxts(gram):
            fill = False
            traversing_dict = ctxt_dict
            path = ()
            for head in slot_gram:
                path += (head,)
                # Mark that we've seen a slot and filler should be recorded
                if head == '_':
                    fill = True
                if head not in traversing_dict:
                    traversing_dict[head] = nesting_dict()
                    traversing_dict[head]['#'] = 0
                # If first time seeing this prefix in this gram, add count
                # and filler
                if path not in seen_slot_grams:
                    traversing_dict[head]['#'] += 1
                    seen_slot_grams.add(path)
                if fill:
                    if '$' not in traversing_dict[head]:
                        traversing_dict[head]['$'] = nesting_dict()
                    # Record filler in nested dict for easier common-
                    # neighbour search
                    filler_dict = traversing_dict[head]['$']
                    for filler in fillers:
                        for i, filler_head in enumerate(filler):
                            if i == len(filler) - 1:
                                try:
                                    filler_dict[filler_head]['#'] += 1
                                except TypeError:
                                    filler_dict[filler_head]['#'] = 1
                            filler_dict = filler_dict[filler_head]
                        filler_dict = traversing_dict[head]['$']
                traversing_dict = traversing_dict[head]
    return ctxt_dict

def slot_ctxt_dy_aux_stars_goals(grams):
    # Initialise context dictionary
    ctxt_dict = nesting_dict()
    # Add new dictionary structure for each gram
    for gram in grams:
        prefix_indices = set()
        for i, j, filler in slot_plus_grams_goals(gram):
            traversing_dict = ctxt_dict
            for item in filler:
                traversing_dict = traversing_dict[item]
            try:
                traversing_dict['#'] += 1
            except TypeError:
                traversing_dict['#'] = 1
            traversing_dict = traversing_dict['$']
            # Record filler in nested dict for easier common-
            # neighbour search
            context_suffix = gram[j:]
            context_prefixes = (gram[k:i] for k in range(i))
            # For each tail of the context prefix, record context:
            for k, context_prefix in enumerate(context_prefixes):
                filler_dict = traversing_dict
                for item in context_prefix:
                    filler_dict = filler_dict[item]
                # Add place-holder for filler if not yet accounted for:
                filler_dict = filler_dict['_']
                if (k, i, filler) not in prefix_indices:
                    try:
                        filler_dict['#'] += 1
                    except TypeError:
                        filler_dict['#'] = 1
                    prefix_indices.add((k, i, filler))
                # Add rest of context:
                for item in context_suffix:
                    filler_dict = filler_dict[item]
                    try:
                        filler_dict['#'] += 1
                    except TypeError:
                        filler_dict['#'] = 1
    return ctxt_dict

def slot_plus_grams_ctxts(gram):
    slot_dict = defaultdict(list)
    for i in range(len(gram)):
        for j in range(i + 1, min(len(gram) + 1, len(gram) + i)):
            slot_gram, filler = gram[:i] + ('_',) + gram[j:], gram[i:j]
            if i > 0 and j < len(gram):
                fillers = [filler]
                stop = i
            else:
                if i == 0:
                    fillers = tails(filler)
                    stop = i + 1
                if j == len(gram):
                    fillers = inits(filler)
                    stop = i
            for k in range(stop):
                slot_subgram = slot_gram[k:]
                slot_dict[slot_subgram] += fillers
                if slot_subgram[:1] == ('_',):
                    for l in range(2, len(slot_subgram)):
                        plus_gram = ('_', '+') + slot_subgram[l:]
                        slot_dict[plus_gram] += fillers
                if slot_subgram[-1:] == ('_',):
                    for l in range(1, len(slot_subgram) - 1):
                        plus_gram = slot_subgram[:l] + ('+', '_')
                        slot_dict[plus_gram] += fillers
    for slot_gram, fillers in slot_dict.items():
        yield (slot_gram, fillers)

def old(gram):
    slot_dict = defaultdict(lambda: defaultdict(list))
    for i in range(len(gram)):
        for j in range(i + 1, min(len(gram) + 1, len(gram) + i)):
            slot_gram, filler = gram[:i] + ('_',) + gram[j:], gram[i:j]
            if i > 0:
                fillers = [filler]
                stop = i
            else:
                fillers = tails(filler)
                stop = i + 1
            for k in range(stop):
                slot_subgram = slot_gram[k:]
                for subfiller in fillers:
                    slot_dict[subfiller][slot_gram].append(slot_subgram)
                    for l in range(len(subfiller)):
                        for m in range(l + 1, min(len(subfiller) + 1, len(subfiller) + l)):
                            plus_filler = subfiller[:l] + ('+',) + subfiller[m:]
                            slot_dict[plus_filler][slot_gram].append(slot_subgram)
    for filler, slot_gram_dict in slot_dict.items():
        for subgrams in slot_gram_dict.values():
            yield (filler, subgrams)

def slot_plus_grams_goals(gram):
    for i in range(len(gram)):
        for j in range(i + 1, min(len(gram) + 1, len(gram) + i)):
            filler, prefix, suffix = gram[i:j], gram[:i], gram[j:]
            for m in range(i):
                yield (i, j, filler)
                for k in range(len(filler)):
                    for l in range(k + 1, min(len(filler) + 1, len(filler) + k)):
                        yield (i, j, filler[:k] + ('+',) + filler[l:])


def slot_grams(gram):
    for i in range(len(gram)):
        for j in range(i + 1, len(gram) + 1):
            yield (gram[:i] + ('_',) + gram[j:], gram[i:j])

def slot_grams_stars_ctxts(gram):
    star_grams = defaultdict(list)
    for i in range(len(gram)):
        for j in range(i+1, min(len(gram)+1, i+len(gram))):
            slot_gram, fillers = gram[:i] + ('_',) + gram[j:], [gram[i:j]]
            if slot_gram[:1] == ('_',):
                for k in range(2, len(slot_gram)):
                    star_grams[('_',) + ('+',) + slot_gram[k:]] += fillers
            if slot_gram[-1:] == ('_',):
                for k in range(1,len(slot_gram)-1):
                    filler_inits = [filler[:l] for filler in fillers for l in range(1,len(filler)+1)]
                    star_grams[slot_gram[:k] + ('+',) + ('_',)] += filler_inits
            yield (slot_gram, fillers)
    for star_gram, fillers in star_grams.items():
        yield (star_gram, fillers)

def slot_grams_stars_goals(gram):
    for i in range(len(gram)):
        for j in range(i + 1, len(gram) + 1):
            filler, slot_gram = gram[i:j], gram[:i] + ('_',) + gram[j:]
            yield (filler, slot_gram)
            for k in range(len(filler)):
                for l in range(k + 1, len(filler) + min(1, k)):
                    star_filler = filler[:k] + ('+',) + filler[l:]
                    yield (star_filler, slot_gram)
            

def tails(sentence):
    return [tuple(sentence[i:]) for i in range(len(sentence))]

def inits(sentence):
    return [tuple(sentence[:len(sentence)-i]) for i in range(len(sentence))]

def arb_ctxt_dy(corpus):
    grams = [tail for sentence in corpus for tail in tails(['<s>'] + sentence + ['</s>'])]
    grams_rev = [tuple(reversed(init)) for sentence in corpus for init in inits(['<s>'] + sentence + ['</s>'])]
    cdy = {}
    cdy['fw'] = arb_ctxt_dy_aux(grams)
    cdy['bw'] = arb_ctxt_dy_aux(grams_rev)
    return cdy

def arb_ctxt_dy_aux(grams):
    # End recursion at unigrams, just return dict with counts
    if set(grams) == {()}:
        return {'#': len(grams)}
    # Sort tails of ngrams according to head word
    tail_dict = defaultdict(list)
    for gram in grams:
        tail_dict[gram[:1]].append(gram[1:])
    # Recursively compute context dict for each head word
    cdy = {head: arb_ctxt_dy_aux(tails) for head, tails in tail_dict.items()}
    cdy['#'] = len(grams)
    return cdy

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

# -----
# Neighbour searching algorithms
# -----

def gram_lookup(gram, dy):
    rem_dy = dy
    for head in gram:
        if head in rem_dy:
            rem_dy = rem_dy[head]
        else:
            raise KeyError(str(gram) + ' not found in context dictionary.')
    return rem_dy['$']

def retrieve_fillers(slot_gram, cdy):
    pass

def cmn_fillers(slot_gram1, slot_gram2, cdy):
    return cmn_fillers_aux(gram_lookup(slot_gram1, cdy), gram_lookup(slot_gram2, cdy))

def cmn_fillers_aux(rem_dy1, rem_dy2, path=()):
    filling_words1, filling_words2 = list(rem_dy1.keys()), list(rem_dy2.keys())
    if filling_words1 == ['#'] or filling_words2 == ['#']:
        return []
    fillers = []
    cmn_filling_words = filter(lambda x: x != '#' and x in filling_words2, filling_words1)
    for word in cmn_filling_words:
        path += (word,)
        if '#' in rem_dy1[word].keys() and '#' in rem_dy2[word].keys():
            fillers.append(path)
        fillers += cmn_fillers_aux(rem_dy1[word], rem_dy2[word], path)
        path = path[:-1]
    return fillers

def cmn_contexts(filler1, filler2, gdy):
    return cmn_contexts_aux(gram_lookup(filler1, gdy), gram_lookup(filler2, gdy))

def cmn_contexts_aux(rem_dy1, rem_dy2, path=(), fill=False):
    context_words1, context_words2 = list(rem_dy1.keys()), list(rem_dy2.keys())
    if context_words1 == ['#'] or context_words2 == ['#']:
        return []
    contexts = []
    cmn_context_words = filter(lambda x: x != '#' and x in context_words2, context_words1)
    for word in cmn_context_words:
        path += (word,)
        old_fill = fill
        if word == '_':
            fill = True
        if fill and '#' in rem_dy1[word].keys() and '#' in rem_dy2[word].keys() and len(path) > 1:
            contexts.append(path)
        contexts += cmn_contexts_aux(rem_dy1[word], rem_dy2[word], path, fill)
        path = path[:-1]
        fill = old_fill
    return contexts

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
            return {}
    return rem_dy

def bw_lookup(ngram, cdy):
    rem_dy = cdy['bw']
    for word in reversed(ngram):
        try:
            rem_dy = rem_dy[word]
        except KeyError:
            return {}
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
    return math.sqrt(bw_score * fw_score)

def fw_gen_prob(refr, targ, cdy):
    fw_prob = 0
    for fwnb in cmn_fw_nbs(refr, targ, cdy):
        fw_num = fw_count(refr + fwnb, cdy) * fw_count(targ + fwnb, cdy)
        fw_den = fw_count(refr, cdy) * bw_count(fwnb, cdy)
        fw_prob += fw_num / fw_den
    return fw_prob

def bw_gen_prob(refr, targ, cdy):
    bw_prob = 0
    for bwnb in cmn_bw_nbs(refr, targ, cdy):
        bw_num = bw_count(bwnb + refr, cdy) * bw_count(bwnb + targ, cdy)
        bw_den = bw_count(refr, cdy) * fw_count(bwnb, cdy)
        bw_prob += bw_num / bw_den
    return bw_prob

def gen_prob(refr, targ, cdy):
    return bw_gen_prob(refr, targ, cdy) * fw_gen_prob(refr, targ, cdy)

def gen_sims(targ, cdy):
    fw_sims = defaultdict(float)
    for fwnb in fw_nbs(targ, cdy):
        for refr in bw_nbs(fwnb, cdy, len(targ)):
            fw_sims[refr] += prob_bw(targ, fwnb, cdy) * prob_fw(refr, fwnb, cdy)
    bw_sims = defaultdict(float)
    for bwnb in bw_nbs(targ, cdy):
        for refr in fw_nbs(bwnb, cdy, len(targ)):
            bw_sims[refr] += prob_fw(bwnb, targ, cdy) * prob_bw(bwnb, refr, cdy)
    sims = {}
    for fw_sim in fw_sims:
        sims[fw_sim] = min(fw_sims[fw_sim], bw_sims[fw_sim])
    return sorted(list(sims.items()), key=lambda x: x[1], reverse=True)
    

def fw_gen_prob_sep(refr, targ, cdy):
    fw_num = 0
    fw_den = 0
    for fwnb in cmn_fw_nbs(refr, targ, cdy):
        fw_num += fw_count(refr + fwnb, cdy) * fw_count(targ + fwnb, cdy)
        fw_den += fw_count(refr + fwnb, cdy) * bw_count(fwnb, cdy)
    return fw_num / fw_den
    
def bw_gen_prob_sep(refr, targ, cdy):
    bw_num = 0
    bw_den = 0
    for bwnb in cmn_bw_nbs(refr, targ, cdy):
        bw_num += bw_count(bwnb + refr, cdy) * bw_count(bwnb + targ, cdy)
        bw_den += bw_count(bwnb + refr, cdy) * fw_count(bwnb, cdy)
    return bw_num / bw_den

def rel_grams(ctxt, goal, cdy):
    paths = set()
    for anl_goal in fw_nbs(ctxt, cdy, len(goal)):
        for anl_ctxt in cmn_bw_nbs(anl_goal, goal, cdy, len(ctxt)):
            paths.add((anl_ctxt, anl_goal))
            paths.add((    ctxt, anl_goal))
            paths.add((anl_ctxt,     goal))
    relts = {}
    for anl_ctxt, anl_goal in paths:
        try:
            relts[(anl_ctxt, anl_goal)] = min(gen_prob(anl_ctxt, ctxt, cdy),
                                              gen_prob(anl_goal, goal, cdy))
        except:
            continue
    return sorted(list(relts.items()), key=lambda x: x[1], reverse=True)

# -----
# Analogical path finding algorithms
# -----

def paths_split(ctxt, goal, cdy, n=float('inf')):
    m_pairs = set()
    ac_items = {ctxt}
    ag_items = {goal}
    for anl_goal in fw_nbs(ctxt, cdy, len(goal)):
        for anl_ctxt in cmn_bw_nbs(anl_goal, goal, cdy, len(ctxt)):
            ac_items.add(anl_ctxt)
            ag_items.add(anl_goal)
            m_pairs.add((anl_ctxt, anl_goal))
            m_pairs.add((ctxt, anl_goal))
            m_pairs.add((anl_ctxt, goal))
    fw_acs = defaultdict(float)
    bw_acs = defaultdict(float)
    for anl_ctxt in ac_items:
        try:
            fw_acs[anl_ctxt] += fw_gen_prob(anl_ctxt, ctxt, cdy)
        except:
            pass
        try:
            bw_acs[anl_ctxt] += bw_gen_prob(anl_ctxt, ctxt, cdy)
        except:
            pass
    fw_ags = defaultdict(float)
    bw_ags = defaultdict(float)
    for anl_goal in ag_items:
        try:
            fw_ags[anl_goal] += fw_gen_prob(anl_goal, goal, cdy)
        except:
            pass
        try:
            bw_ags[anl_goal] += bw_gen_prob(anl_goal, goal, cdy)
        except:
            pass
    anls = []
    for anl_ctxt, anl_goal in m_pairs:
        ac_prob = (min(fw_acs[anl_ctxt], bw_acs[anl_ctxt]))
        ag_prob = (min(fw_ags[anl_goal], bw_ags[anl_goal]))
        anls.append(((anl_ctxt, anl_goal), min(ac_prob, ag_prob)))
    return sorted(anls, key=lambda x: x[1], reverse=True)

def mx_paths(ctxts, goals, cdy):
    m_pairs = set()
    ac_items = set()
    ag_items = set()
    for ctxt in ctxts:
        ac_items.add(ctxt['item'])
        for goal in goals:
            ag_items.add(goal['item'])
            for anl_goal in fw_nbs(ctxt['item'], cdy, len(goal['item'])):
                for anl_ctxt in cmn_bw_nbs(anl_goal, goal['item'], cdy, len(ctxt['item'])):
                    ac_items.add(anl_ctxt)
                    ag_items.add(anl_goal)
                    m_pairs.add((anl_ctxt, anl_goal))
                    #'''
                    m_pairs.add((ctxt['item'], anl_goal))
                    m_pairs.add((anl_ctxt, goal['item']))
                    #'''
    fw_acs = defaultdict(float)
    bw_acs = defaultdict(float)
    for ctxt in ctxts:
        for anl_ctxt in ac_items:
            try:
                fw_acs[anl_ctxt] += ctxt['weight'] * fw_gen_prob(anl_ctxt, ctxt['item'], cdy)
            except:
                pass
            try:
                bw_acs[anl_ctxt] += ctxt['weight'] * bw_gen_prob(anl_ctxt, ctxt['item'], cdy)
            except:
                pass
    fw_ags = defaultdict(float)
    bw_ags = defaultdict(float)
    for goal in goals:
        for anl_goal in ag_items:
            try:
                fw_ags[anl_goal] += goal['weight'] * fw_gen_prob(anl_goal, goal['item'], cdy)
            except:
                pass
            try:
                bw_ags[anl_goal] += goal['weight'] * bw_gen_prob(anl_goal, goal['item'], cdy)
            except:
                pass
    anls = []
    for anl_ctxt, anl_goal in m_pairs:
        ac_prob = (min(fw_acs[anl_ctxt], bw_acs[anl_ctxt]))# * prob_jt(anl_ctxt, (), cdy)
        ag_prob = (min(fw_ags[anl_goal], bw_ags[anl_goal]))# * prob_jt(anl_goal, (), cdy)
        anls.append(((anl_ctxt, anl_goal), min(ac_prob, ag_prob)))
    return sorted(anls, key=lambda x: x[1], reverse=True)

def gen_words(gram, cdy):
    fw_gens = defaultdict(float)
    for fwnb in fw_nbs(gram, cdy):
        for bwnb in bw_nbs(fwnb, cdy, len(gram)):
            fw_gens[bwnb] += prob_bw(gram, fwnb, cdy) * prob_fw(bwnb, fwnb, cdy)
    bw_gens = defaultdict(float)
    for bwnb in bw_nbs(gram, cdy):
        for fwnb in fw_nbs(bwnb, cdy, len(gram)):
            bw_gens[fwnb] += prob_fw(bwnb, gram, cdy) * prob_bw(bwnb, fwnb, cdy)
    gens = []
    for word in fw_gens:
        gens.append((word, fw_gens[word] * bw_gens[word] * prob_jt(word, (), cdy)))
    return sorted(gens, key=lambda x: x[1], reverse=True)

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

# -----
# Recursive parsing algorithms
# -----

def rec_parse_split(gram, cdy, anl_dy=None, n=float('inf')):
    # Use empty dict as default value for argument `anl_dy`
    if anl_dy == None:
        anl_dy = {}
    # Attempt dynamic lookup
    if gram in anl_dy:
        return anl_dy
    # End recursion when we reach unigrams
    if len(gram) == 1:
        sims = gen_sims(gram, cdy)[:10]
        #'''
        anls = []
        for sim, score in sims:
            anl = {'path': sim, 'score': score, 'tree': sim[0], 'tree type': gram[0]}
            anls.append(anl)
        #'''
        #anls = [{'path': gram, 'score': 1, 'tree': gram[0], 'tree type': gram[0]}]
        anl_dy[gram] = anls
        return anl_dy
    # Recursive step
    splits = ((gram[:i], gram[i:]) for i in range(1,len(gram)))
    anls = []
    tt_dict = {}
    for ctxt, goal in splits:
        # Recursive calls
        rec_ctxts = rec_parse_split(ctxt, cdy, anl_dy, n)[ctxt][:10]
        rec_goals = rec_parse_split(goal, cdy, anl_dy, n)[goal][:10]
        for rec_ctxt in rec_ctxts:
            for rec_goal in rec_goals:
                tree_type = (rec_ctxt['tree type'], rec_goal['tree type'])
                tt_dict[tree_type] = 0
                paths = paths_split(rec_ctxt['path'], rec_goal['path'], cdy, n)[:30]
                for path, path_score in paths:
                    score = path_score * rec_ctxt['score'] * rec_goal['score']
                    tt_dict[tree_type] += score
                    tree = ((' '.join(path[0]), rec_ctxt['tree']),
                            (' '.join(path[1]), rec_goal['tree']))
                    anls.append({'path': path[0] + path[1], 'score': score,
                                 'tree': tree, 'tree type': tree_type})
    for anl in anls:
        anl['score'] = anl['score']# * tt_dict[anl['tree type']]
    anls.sort(reverse=True, key=lambda x: x['score'])
    anl_dy[gram] = anls
    return anl_dy

# -----
# Functions for visualising analyses
# -----

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