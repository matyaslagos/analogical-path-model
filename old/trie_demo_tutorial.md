Import the demo module and the Grimm corpus as a list of tuples:
```
import trie_demo
corpus = trie_demo.txt_to_list('grimm_corpus.txt')
```
Create the frequency trie data structure:
```
dt = trie_demo.freq_trie_setup(corpus)
```
_Data structure looks like this, e.g. after reading `the king arrived today` (forward root is for looking up what occurs on the right of some context, backward root is for looking up what occurs on the left of some context):_

![Frequency trie illustration](https://live.staticflickr.com/65535/54498802384_ac26b8f83c.jpg)

Find the fillers that occur on the right of `the fox`:
```
fillers = list(dt.get_fillers(('the', 'fox', '_'))
```
_For example, `(('_', 'said'), 7)` will be in `fillers`, meaning that `said` occurs 7 times on the right of `the fox`._

Find the fillers that occur on the left of `the fox`:
```
fillers = list(dt.get_fillers(('_', 'the', 'fox'))
```
Find the fillers that occur on the right of both `king` and `queen`:
```
shared_fillers = list(dt.get_shared_fillers(('king', '_'), ('queen', '_'))
```
_For example, `(('_', 'had'), 13, 6)` will be in `shared_fillers`, meaning that `had` occurs 13 times on the right of `king` and 6 times on the right of `queen`._
