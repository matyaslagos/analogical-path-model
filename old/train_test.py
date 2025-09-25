# -----
# Training and testing
# -----

import random
from math import log
from math import floor

from ecgpack import ecg
#from ecgpack import n_gram as ng
from ecgpack import text_import as ti
from ecgpack import tk

def zeros_count(total):
    r = []
    for row in total:
        es = row[6]
        if not isinstance(es, tuple):
            r.append(row)
        else:
            if es[0] == 0:
                if es[1] == 0:
                    r.append(row[:6] + [0])
                else:
                    r.append(row[:6] + [om(es[1])])
            else:
                r.append(row[:6] + [-om(es[0])])
    return r

def zeros_zeros(total):
    r = []
    for row in total:
        if not isinstance(row[6], tuple):
            r.append(row)
        else:
            r.append(row[:6] + [0])
    return r

def means(results):
    g_results = [row for row in results if not isinstance(row[6], tuple)]
    mean_list = []
    for i in range(3, 7):
        mean = sum(row[i] for row in g_results) / len(g_results)
        mean_list.append(mean)
    return [mean_list, len(results) - len(g_results)]

def kn_mass(test_data, cdy, kn):
    result_list = []
    for sentence in test_data:
        good = sentence.copy()
        bad = sentence.copy()
        random.shuffle(bad)
        
        kn_result = kn_comp(good, bad, kn)
        ecg_result = ecg_comp(good, bad, cdy)
        
        result_list.append([' '.join(good), ' '.join(bad), len(good),
                            kn_result, ecg_result])
    return result_list

def test_comp(test, lid, gt, kn, cdy):
    result_list = []
    for sentence in test:
        good = sentence.copy()
        bad = sentence.copy()
        random.shuffle(bad)
        
        lid_result = lid_comp(good, bad, lid)
        gt_result = gt_comp(good, bad, gt)
        kn_result = kn_comp(good, bad, kn)
        ecg_result = ecg_comp(good, bad, cdy)
        
        result_list.append([' '.join(good), ' '.join(bad), len(good),
                            lid_result, gt_result, kn_result, ecg_result])
    return result_list

def lid_comp(s1, s2, lid):
    return om_diff(tk.lm_test(s1, lid), tk.lm_test(s2, lid))

def kn_comp(s1, s2, kn):
    return om_diff(tk.lm_test(s1, kn), tk.lm_test(s2, kn))

def gt_comp(s1, s2, gt):
    return om_diff(tk.gt_test(s1, gt), tk.gt_test(s2, gt))

def ecg_comp(s1, s2, cdy):
    p1 = ecg.parse(s1, cdy)
    p2 = ecg.parse(s2, cdy)
    if p1[1] == [] and p2[1] == []:
        return p1[0] - p2[0]
    else:
        return (p1, p2)

def om(float):
    return floor(log(float, 10))

def om_diff(float1, float2):
    return om(float1) - om(float2)

def train_test(s:list):
    '''
    Selects 90 percent of utterances as training set and 10 percent as
    test set, such that there are no unknown words in the test set.
    
    Parameters:
    - text (list): list of lists of words, output of
      phrases_listed(text)
    
    Returns:
    - [train, test] where `train` is randomly selected 90% of utterances in
      text, and `test` is remaining 10% of utterances
    
    '''
    l = s.copy()
    random.shuffle(l)
    n = round(len(l) * 0.9)
    train = l[:n]
    vocab = {word for sentence in train for word in sentence}
    test = [sentence for sentence in l[n:] if set(sentence).issubset(vocab)]
    return [train, test]

def remove_unknowns(s:list, dy:dict):
    ''' 
    Removes those sentences from s that contain unknown words (words not in dy).
    
    '''
    r = []
    for sentence in s:
        ok = 1
        for word in sentence:
            if word not in dy:
                ok = 0
                break
        if ok == 1:
            r.append(sentence)
    return r

def tt_prep(filename:str):
    spl = train_test(ti.phrases_listed(ti.imp_txt(filename)))
    dy = ti.context_dy(spl[0])
    return [dy, spl[0], remove_unknowns(spl[1], dy)]

def inside_shuffle(s:list):
    inside = s[1:-1][:]
    random.shuffle(inside)
    return ['<'] + inside + ['>']

def str_shuf(string):
    sl = string.split()
    random.shuffle(sl)
    return ' '.join(sl)

def test(sentence:str, ecg_dy:dict, lapl_dy:dict):
    inp = ['<'] + sentence.split() + ['>']
    shuff = inside_shuffle(inp)
    ecg1 = ecg.an_parse(inp, ecg_dy)
    ecg2 = ecg.an_parse(shuff, ecg_dy)
    lapl1 = ng.pr_parse(inp, lapl_dy)
    lapl2 = ng.pr_parse(shuff, lapl_dy)
    r = 'Test sentence: ' + sentence + \
        '\nShuf sentence: ' + ' '.join(shuff[1:-1]) + \
        '\nAnl scores: ' + str(ecg1) + ', ' + str(ecg2) + \
        '\nLap scores: '  + str(lapl1) + ', ' + str(lapl2)
    print(r)
    #return r