# Usage guide for [`syntax_model.py`](https://github.com/matyaslagos/analogical-path-models/blob/main/syntax_model.py)

## Setup and usage of some functions

Setting up the model using [Peter Norvig's Gutenberg corpus](https://github.com/matyaslagos/analogical-path-models/blob/main/corpora/norvig_corpus.txt) (note that due to the inefficiency of my data structure implementation, the trained model uses about 4GB of memory):
```python
import syntax_model as syn
model = syn.FreqTrie()
model.setup() # ~30 secs, needs file with path 'corpora/norvig_corpus.txt'
```

What are the 10 best analogies for the word `idea`?
```python
from pprint import pp # pretty printing
a = syn.bilateral_analogies(model, ('idea',))
pp(a[:10])
```
What are the 10 best bigram analogies for the bigram `green sofa`?
```python
a = syn.bigram_analogies(model, ('green', 'sofa'))
pp(a[:10])
```
What are the 10 best unigram analogies for the bigram `her book`?
```python
a = syn.bigram_to_unigrams(model, ('her', 'book'))
pp(a[:10])
```

## Data structure overview

The `FreqTrie` data structure looks like the image below. The image illustrates the `_insert()` method inserting some prefixes and suffixes of the sequence `the quick fox ran towards me`. The black part of the trie already exists, and is now updated with all prefixes of `the quick fox` in `bw_root` and all suffixes of `ran towards me` in `fw_root` (when called on a sequence like `the quick fox ran towards me`, `_insert()` inserts all of its prefixes into `bw_root` and all of its suffixes into `fw_root`):
- green edges with nodes are added (represented by the `dict`-type `.children` attribute of `FreqNode`), and
- frequencies are updated (represented by the `int`-type `.freq` attribute of `FreqNode`).

In the `fw_root` trie, each node's `.freq` attribute represents the frequency of the word sequence with which we can reach that node; in the `bw_root` trie, each node's `.freq` attribute represents the frequency of the _reverse_ of the word sequence with which we can reach that node. (E.g. the `.freq` of the top left node `the` is `1`, meaning that the frequency of the _reverse_ of `fox quick the`, i.e. the frequency of `the quick fox`, is `1`.)

<img width="799" height="477" alt="Screenshot 2025-10-05 at 17 49 45" src="https://github.com/user-attachments/assets/92d3998f-4fcf-4762-a3e3-495947dd33dc" />
