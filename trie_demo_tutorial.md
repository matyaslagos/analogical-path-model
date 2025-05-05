Import the Grimm corpus as a list of tuples:
```
import trie_demo
corpus = trie_demo.txt_to_list('grimm_corpus.txt')
```
Create the frequency trie data structure:
```
dt = trie_demo.freq_trie_setup(corpus)
```
Find the fillers that occur after "the fox":
```
fillers = list(dt.get_fillers(('the', 'fox', '_'))
```
Find the fillers that occur before "the fox":
```
fillers = list(dt.get_fillers(('_', 'the', 'fox'))
```
Find the fillers that occur both after "king" and after "queen":
```
shared_fillers = list(dt.get_shared_fillers(('king', '_'), ('queen', '_'))
```
