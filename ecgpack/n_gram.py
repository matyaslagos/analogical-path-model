# -----
# Existing n-gram models
# -----

from ecgpack import text_import as ti
from ecgpack import ecg

def bgrs_of_sent(l:list):
	''' Returns list of bigrams from l, a list of words. '''
	r = []
	for i in range(len(l)-1):
		r.append(l[i:i+2])
	return r

def bigrams(l:list):
	''' Returns bigram dictionary with frequency counts from l, a list
		of sentences (output of phrases_listed(text)). '''
	dy = {}
	
	for sentence in l:
		for bigram in bgrs_of_sent(sentence):
			if bigram[0] in dy:
				if bigram[1] in dy[bigram[0]]:
					dy[bigram[0]][bigram[1]] = \
					dy[bigram[0]][bigram[1]] + 1
				else:
					dy[bigram[0]][bigram[1]] = 1
			else:
				dy[bigram[0]] = {}
				dy[bigram[0]][bigram[1]] = 1

	return dy

def bigrams_prob(l:list):
	''' Returns bigram dictionary with relative frequencies from l, a list
		of sentences. '''
	# Gets simple bigram dictionary
	dy = bigrams(l)
	
	# Sets values to relative frequencies
	n = 0
	for j in dy:
		for k in dy[j]:
			n = n + 1

	for j in dy:
		for k in dy[j]:
			dy[j][k] = dy[j][k] / n
	
	return dy

def bigrams_lapl(l:list):
	''' Returns bigram dictionary with laplace-smoothed relative frequencies
		from l, a list of sentences. '''
	# Gets simple bigram dictionary
	dy = bigrams(l)
	
	# Adds one to each count
	words = ti.words(l)
	for w in words:
		if w != '>':
			for x in words:
				if x != '<':
					if x in dy[w]:
						dy[w][x] = dy[w][x] + 0.5
					else:
						dy[w][x] = 0.5
	
	# Sets values to laplace-smoothed relative frequencies
	n = 0
	for j in dy:
		for k in dy[j]:
			n = n + 1

	for j in dy:
		for k in dy[j]:
			dy[j][k] = dy[j][k] / n
	
	return dy

def cond_prob(b:list, dy:dict):
	both = dy[b[0]][b[1]]
	first = sum([dy[b[0]][k] for k in dy[b[0]]])
	return both / first

def pr_parse(s:list, dy:dict):
	r = 1
	for i in range(len(s)-1):
		try:
			r = r * cond_prob(s[i:i+2], dy)
		except:
			r = 0
			break
	return ecg.sci_ntn(r)