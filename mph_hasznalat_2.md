
## Setup

Főneves korpusz importálása:
```
corpus = mph.csv2wordfreqdict('fonev_corpus.csv')
```

Disztribúciós szótár létrehozása (kb. 5 másodperc alatt lefut):
```
ddy = mph.distrtrie_setup_freq(corpus)
```

## Tesztelés

Mik a legerősebb analógiák a `kalap + jaink` összetételre?
```
from pprint import pp

anls = mph.anl_substs(ddy, mph.lc('kalap'), mph.rc('jaink'))
pp(anls[:10])    # 10 legerősebb analógia szépen printelve
```
Itt a legerősebb analógia az lesz hogy `nap + ja`. Ezekből számolódik ki az erőssége:
- `gyakoriság`: a `napja` szó gyakorisága.
- `bal-csere`, `jobb-csere`: P(`kalap` || `nap`), P(`jai` || `ja`). Azaz: a `nap` mennyire cserélhető a `kalap`-ra és a `ja` mennyire cserélhető a `jai`-ra.
- `bal-entrópia`, `jobb-entrópia`: a `nap` után előforduló betűk eloszlásának entrópiája és a `ja` előtt előforduló betűk eloszlásának entrópiája.

Ezeket összeszorozzuk: a `nap + ja` analógia erőssége az hogy `gyakoriság * bal-csere * jobb-csere * bal-entrópia * jobb-entrópia`.
