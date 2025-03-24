## Setup

Sztaki korpusz importálása:
```
corpus = mph.txt2wordlist('sztaki_corpus.txt')
training_set, test_set = corpus[:1000000], corpus[1000000:]
```

Disztribúciós szótár létrehozása (kb. 20 másodperc alatt lefut):
```
ddy = mph.distrtrie_setup(training_set)
```

## Tesztelés

Mik a legerősebb analógiás kontextus-források ha az `_ok`
kontextusból keresünk utat a `nap` fillerbe?
```
from pprint import pp

anls = mph.anl_contexts_func(ddy, mph.rc('ok'), mph.lc('nap'))
pp(anls[:10])    # 10 legerősebb kontextus-forrás szépen printelve
```
(`rc` = "right context", `lc` = "left context")

Mik a legerősebb analógiás szó-források ha az `_ok`
kontextusból keresünk utat a `nap` fillerbe?
```
anls = mph.anl_words_func(ddy, mph.rc('ok'), mph.lc('nap'))
```

Mennyire jól jósolja az `_ot` kontextus viselkedése az
`_on` kontextus viselkedését?
```
pred_prob, common_fillers_dict = mph.context_pred_func(ddy, mph.rc('ot'), mph.rc('on'))
```

Melyik kontextusok jósolják a legjobban az `_ait` kontextus viselkedését?
```
pred_list = mph.predictors_func(ddy, mph.rc('ait'))
```
