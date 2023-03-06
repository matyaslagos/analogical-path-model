# -----
# Preparing a text
# -----

import csv

def imp_txt(filename, sep='.'):
    """
    Imports normalised text as list of lists of words, according to NLTK
    convention.
    
    Parameters:
    - filename (str): name of a .txt file
    - sep (str): sentence-separating symbol
    
    Returns:
    - list: list of lists of words in file
    
    """
    # encoding parameter can be removed, only used here to remove
    # '\ufeff' from beginning
    with open(filename, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    #r = ''
    #for l in lines:
    #    r = r + l.replace('\n', ' ')
    return [sentence.strip().split() for sentence in lines]

def imp_csv(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        rows = f.readlines()
    return [row.split(',') for row in rows]

def exp_sent_txt(list, filename):
    """ Exports a list of lists of words as a txt file, with sublists
    separated by newlines. """
    with open(filename, 'w') as file:
        for sentence in list:
            file.write(' '.join(sentence) + '\n')

def exp_list_txt(list, filename):
    with open(filename, 'w') as file:
        for row in list:
            file.write(','.join(map(str, row)) + '\n')

def words(l):
    """ Returns list of words from l, a list of grams. """
    ws = set()
    
    for gram in l:
        for w in gram:
            ws.add(w)
    
    return list(ws)

# Writes two csvs from edgelist
# - el is output of bigr_edgelist(dy)
def two_to_csv(el):
    with open('vertices.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in el[0]:
            writer.writerow(i)
    with open('edges.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in el[1]:
            writer.writerow(i)

def to_csv(file, name):
    with open(name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in file:
            writer.writerow(i)

# Gets top k to l most frequent words from list of words (if k and l are not given,
# then gets all of words in order of frequency).
def fr_words(ws, k=0, l=0):
    if l == 0:
        l = len(ws)
    d = {}
    
    for w in ws:
        if w in d:
            d[w] = d[w] + 1
        else:
            d[w] = 1
    
    freqs = [(key, d[key]) for key in d]
    
    return sorted(freqs, key = lambda x : x[1], reverse = True)[k:l+1]

# Gets all contexts for each word into a dictionary
# - text is output of phrases_listed(text)
def context_dy(phrases):
    dy = {}
    
    for p in phrases:
        for i in range(len(p)):
            if p[i] in dy:
                dy[p[i]].append([p[:i], [p[i]], p[i+1:]])
            else:
                dy[p[i]] = [[p[:i], [p[i]], p[i+1:]]]
    
    return dy
                
# Makes a list of grams, where a gram is a list of consecutive strings enclosed
# by '<' and '>', e.g. ['<', 'how', 'are', 'you', '>'].
# - text is a list of strings, output of imp_txt(filename)
def phrases_listed(text):

    phrases = []
    current = []
    
    for w in text:
        if w == '>':
            current.append('>')
            phrases.append(current)
            current = []
        else:
            current.append(w)

    return phrases

def prep(t):
    return context_dy(phrases_listed(imp_txt(t)))

def imp_phrases(t):
    return phrases_listed(imp_txt(t))