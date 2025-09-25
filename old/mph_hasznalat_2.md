
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
pp(anls[:10])    # 10 legerősebb analógia
```
Itt a második legerősebb analógia az lesz hogy `csont + juk`. (A legerősebb az lesz hogy `kalap + juk`, de ezt nem jó bemutatónak használni.) Most csak azt nézi az algoritmus hogy a `csont`-ot mennyire jól lehet a `kalap`-ra cserélni és a `juk`-ot mennyire jól lehet a `jaink`-ra cserélni. És a csere jósága nem más mint a "közös entrópia": megnézzük hogy pl. a `csont` és a `kalap` után mik fordultak elő, ebből csinálunk egy eloszlást (ezt majd elmondom hogy hogyan), és ennek az eloszlásnak az entrópiája a csere jósága. A teljes analógia erősségét úgy kapjuk meg hogy a bal és a jobb cserék jóságát összeszorozzuk. Szóval itt entrópiákat szorzunk össze, ami nagyon csúnya, de az eredmények elég jók szerintem ahhoz hogy megpróbáljuk szépre megcsinálni.

Mik a legjobb felbontásai a `tornyok` szónak, és ezekre a felbontásokra mi a 10-10 legerősebb analógia?
```
anl_splits = mph.iter_anls(ddy, 'tornyok')
pp(anl_splits)
```
Ezzel egy listát kapunk aminek ilyen 3-tuple-ök az elemei: `("felbontás" (str), "felbontás jósága" (float), "10 legjobb analógia erre a felbontásra" (list))`.
