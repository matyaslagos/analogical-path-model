from collections import defaultdict
import random
import math
import itertools
import csv

def csv2list(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = [tuple(row) for row in reader]
    return data

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
    return [tuple(line.strip()) for line in lines]

def splits(data):
    return [((item[0][:i], item[0][i:]), int(item[1])) for item in data for i in range(0,len(item[0])+1)]

def ctxt_dy(data):
    fw = defaultdict(lambda: defaultdict(int))
    bw = defaultdict(lambda: defaultdict(int))
    for split, freq in splits(data):
        ctxt, goal = split
        fw[ctxt][goal] += freq
        bw[goal][ctxt] += freq
    fw = {key: dict(fw[key]) for key in fw}
    bw = {key: dict(bw[key]) for key in bw}
    return {'fw': fw, 'bw': bw}

def new_words(cdy):
    nw = defaultdict(float)
    for ctxt in cdy['fw']:
        ctxt_freq = sum(cdy['fw'][ctxt].values())
        for goal in cdy['fw'][ctxt]:
            goal_freq = sum(cdy['bw'][goal].values())
            prob_jt = cdy['fw'][ctxt][goal]
            for new_ctxt in cdy['bw'][goal]:
                prob_bw = cdy['bw'][goal][new_ctxt] / goal_freq
                for new_goal in cdy['fw'][ctxt]:
                    prob_fw = cdy['fw'][ctxt][new_goal] / ctxt_freq
                    if new_goal not in cdy['fw'][new_ctxt]:
                        nw[new_ctxt + new_goal] += prob_jt * prob_fw * prob_bw
    return sorted(list(nw.items()), key=lambda x: x[1], reverse=True)

def paths(word, cdy):
    splits = [(word[:i], word[i:]) for i in range(0,len(word))]
    anls = defaultdict(float)
    for ctxt, goal in splits:
        if goal not in cdy['bw'] or ctxt not in cdy['fw']: continue
        for anl_ctxt in cdy['bw'][goal]:
            anl_ctxt_freq = sum(cdy['fw'][anl_ctxt].values())
            for anl_goal in cdy['fw'][ctxt]:
                anl_goal_freq = sum(cdy['bw'][anl_goal].values())
                if anl_goal in cdy['fw'][anl_ctxt]:
                    prob_bw = cdy['bw'][anl_goal][ctxt] / anl_goal_freq
                    prob_jt = cdy['fw'][anl_ctxt][anl_goal]
                    prob_fw = cdy['fw'][anl_ctxt][goal] / anl_ctxt_freq
                    anls[((ctxt, goal), (anl_ctxt, anl_goal))] += prob_bw * prob_jt * prob_fw
    return sorted(list(anls.items()), key=lambda x: x[1], reverse=True)