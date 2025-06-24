Setup
```
import trie_demo
corpus = trie_demo.txt_to_list('norvig_corpus.txt')
dt = trie_demo.freq_trie_setup(corpus)
```
Mik a `green shirt` bigram legjobb bigram-analógiái sorrendezve?
```
anls = trie_demo.bigram_anls(dt, 'green shirt')
```
(Ez csak bigramokra működik.)
