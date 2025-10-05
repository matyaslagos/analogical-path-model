# Usage guide for `syntax_model.py`
Setup:
```python
from pprint import pp # pretty printing
import syntax_model as syn
model = syn.FreqTrie()
model.setup() # ~30 secs, needs file with path 'corpora/norvig_corpus.txt'
```
What are the 10 best analogies for the word `idea`?
```python
a = syn.bilateral_analogies(model, ('idea',))
pp(a[:10)
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
