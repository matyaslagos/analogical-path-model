# Usage guide for `syntax_model.py`

## Setup and data structure overview

```python
import syntax_model as syn
model = syn.FreqTrie()
model.setup() # ~30 secs, needs file with path 'corpora/norvig_corpus.txt'
```
The `FreqTrie` data structure looks like the image below. The image illustrates the `_insert()` method inserting a particular _(prefix, suffix)_ split of the sequence `the quick fox ran towards me`. The black part of the trie already exists, and is now updated with prefix `the quick fox` and suffix `ran towards me`:
- green edges with nodes are added (represented by the `dict`-type `.children` attribute of `FreqNode`), and
- frequencies are updated (represented by the `int`-type `.freq` attribute of `FreqNode`).

<img width="879" height="565" alt="Screenshot 2025-10-05 at 17 15 19" src="https://github.com/user-attachments/assets/832fb9b8-343a-4d8d-a950-31dcd906110a" />

## Example usage

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
