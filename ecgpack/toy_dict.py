
# -----
# Toy dictionary for (+1)-path search algorithm
# -----

# Toy bigram dictionary
toy_words = ['to', 'eat', 'drink', 'sleep', 'see', 'some', 'like', 'an', 'him',
			 'for', 'the', 'apple', 'dog', 'in', 'big', 'chair']

toy_dict = {}

for i in toy_words:
	toy_dict[(i, 1)]  = {}
	for j in toy_words:
		toy_dict[(i,1)][j]  = 0

toy_dict[('to', 1)]['eat'] = 1
toy_dict[('to', 1)]['drink'] = 1
toy_dict[('to', 1)]['sleep'] = 1
toy_dict[('like', 1)]['him'] = 1
toy_dict[('like', 1)]['an'] = 1
toy_dict[('eat', 1)]['an'] = 1
toy_dict[('drink', 1)]['some'] = 1
toy_dict[('see', 1)]['some'] = 1
toy_dict[('him', 1)]['for'] = 1
toy_dict[('an', 1)]['apple'] = 1
toy_dict[('the', 1)]['apple'] = 1
toy_dict[('the', 1)]['big'] = 1
toy_dict[('the', 1)]['dog'] = 1
toy_dict[('for', 1)]['the'] = 1
toy_dict[('dog', 1)]['in'] = 1
toy_dict[('sleep', 1)]['in'] = 1
toy_dict[('big', 1)]['chair'] = 1

# In an adjacency matrix, replaces frequencies relative frequencies.
def d_probs(dy):
	c = 0
	for i in dy:
		for j in dy[i]:
			c = c + dy[i][j]
	for i in dy:
		for j in dy[i]:
			dy[i][j] = dy[i][j] / c
	return dy

toy_dict = d_probs(toy_dict)

# Sets the values of the backward ("negative") edges based on the positive edges
def set_negs(dy):
	r = dict(dy)
	for pos_key in dy:
		for w in dy[pos_key]:
			neg_key = (w, -1)
			if neg_key in r:
				r[neg_key][pos_key[0]] = r[pos_key][w]
			else:
				r[neg_key] = {}
				r[neg_key][pos_key[0]] = r[pos_key][w]
	return r

toy_dict = set_negs(toy_dict)


# -----
# Looking for plus-one paths in graphs to make analogies
# -----

# Simple bigram dictionary from normalised text (only periods, newlines ignored).
# - text is a list of words, output of imp_txt()
def simp_bigrams(text):
	dy = {}
	words = set()
	
	for i in range(len(text)-1):
		b1 = text[i]
		b2 = text[i+1]
		words.add(b1)
		if b1 in dy:
			if b2 in dy[b1]:
				dy[b1][b2] = dy[b1][b2] + 1
			else:
				dy[b1][b2] = 1
		else:
			dy[b1] = {}
			dy[b1][b2] = 1
	
	dy = d_probs(dy)
	
	return dy

# Creates edgelist from bigram dictionary, for use in R graph visualisation
# - dy is output of simp_bigrams()
def bigr_edgelist(dy):
	vertices = [['id','label']]
	edges = [['from','to','weight']]
	d_list = list(dy)
	k_list = {}
	for i in range(len(d_list)):
		vertices.append(['v' + str(i), d_list[i]])
		k_list[d_list[i]] = 'v' + str(i)
	for i in dy:
		for j in dy[i]:
			edges.append([k_list[i], k_list[j], str(dy[i][j])])
	
	return [vertices, edges]

# Indexed bigram dictionary from normalised text (only periods, newlines ignored).
# - text is a list of words, output of imp_txt()
def bigrams_ind(text):
	dy = {}
	words = set()
	
	for i in range(len(text)-1):
		b1 = text[i]
		b2 = text[i+1]
		words.add(b1)
		if (b1, 1) in dy:
			if b2 in dy[(b1, 1)]:
				dy[(b1, 1)][b2] = dy[(b1, 1)][b2] + 1
			else:
				dy[(b1, 1)][b2] = 1
		else:
			dy[(b1, 1)] = {}
			dy[(b1, 1)][b2] = 1
	
	dy = set_negs(d_probs(dy))
	
	return dy

# Gets consecutive words based on distributional analogy in k steps.
def get_conseqs(dy, w, k, c=0):
	if k == 0:
		return []
	l = []
	# Selects those words i that have followed w,
	# and if going from w to i makes sets the path index to +1,
	# then i is added to the possible next words.
	# Additionally, for any i that has followed w, we call the
	# function on i while decrementing k and incrementing the path index.
	for i in dy[(w, 1)]:
		if dy[(w,1)][i] == 1:
			if c+1 == 1:
				l.append(i)
				l = l + get_conseqs(dy, i, k-1, c+1)
			else:
				l = l + get_conseqs(dy, i, k-1, c+1)
	# Similarly for all words i that have preceded w.
	for i in dy[(w, -1)]:
		if dy[(w,-1)][i] == 1:
			if c-1 == 1:
				l.append(i)
				l = l + get_conseqs(dy, i, k-1, c-1)
			else:
				l = l + get_conseqs(dy, i, k-1, c-1)
	return list(set(l))

# Gets consecutive words based on distributional analogy in k steps,
# but also includes their probabilities, which is just the product of the relative
# frequencies of the attested cooccurrences of the words in the paths.
# And doesn't allow forth-back-forth moves (by storing prev).
# - dy is a bigram dictionary, output of bigrams()
def get_conseqs_prob(dy, w, k, c=0, p=1, prev=''):
	if k == 0:
		return []
	l = []
	# Selects those words i that have followed w,
	# and if going from w to i makes sets the path index to +1,
	# then i is added to the possible next words.
	# Additionally, for any i that has followed w, we call the
	# function on i while decrementing k and incrementing the path index.
	for i in dy[(w, 1)]:
		if dy[(w,1)][i] > 0 and i != prev:
			if c+1 == 1:
				l.append((k, i, p * dy[(w,1)][i]))
				l = l + get_conseqs_prob(dy, i, k-1, c+1, p * dy[(w,1)][i], w)
			else:
				l = l + get_conseqs_prob(dy, i, k-1, c+1, p * dy[(w,1)][i], w)
	# Similarly for all words i that have preceded w.
	for i in dy[(w, -1)]:
		if dy[(w,-1)][i] > 0 and i != prev:
			if c-1 == 1:
				l.append((k, i, p * dy[(w,-1)][i]))
				l = l + get_conseqs_prob(dy, i, k-1, c-1, p * dy[(w,-1)][i], w)
			else:
				l = l + get_conseqs_prob(dy, i, k-1, c-1, p * dy[(w,-1)][i], w)
	return list(set(l))

