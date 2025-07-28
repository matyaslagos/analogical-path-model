import csv
import re
from collections import Counter
from string import ascii_lowercase
from pprint import pp

#> Importing SZTAKI .tsv corpus <#

def sztaki_tsv_nouns_import():
    """Import nouns from a SZTAKI cleaned corpus as frequency dict.

    Arguments:
        None, defaults to a particular corpus.

    Returns:
        freqs (Counter): dict of nouns and their frequencies
    """
    freqs = Counter()
    with open('corpora/sztaki_corpus_2017_2018_0001_clean.tsv', newline='') as f:
        reader = csv.reader(
            (row for row in f if row.strip() and not row.startswith('#')), delimiter='\t'
        )
        next(reader, None)
        is_hun_char = lambda x: x.lower() in ascii_lowercase + 'áéíóúöőüű'
        is_hun_string = lambda x: all(map(is_hun_char, x))
        for row in reader:
            if len(row) >= 4 and row[3].startswith('[/N]') and is_hun_string(row[0]):
                word_form = '<' + hun_encode(row[0].lower()) + '>'
                cell_features = cell_feature_set(row[3])
                freqs[(word_form, cell_features)] += 1
    return freqs

#> Encoding Hungarian multiletter sounds <#

hun_encodings = {
    'ccs': '11',
    'ddz': '22',
    'ggy': '33',
    'nny': '44',
    'ssz': '55',
    'tty': '66',
    'zzs': '77',
    'dzs': '8',
    'lly': 'jj',
    'cs': '1',
    'dz': '2',
    'gy': '3',
    'ny': '4',
    'sz': '5',
    'ty': '6',
    'zs': '7',
    'ly': 'j'
}
encode_pattern = re.compile(
    '|'.join(re.escape(k) for k in sorted(hun_encodings, key=len, reverse=True))
)

def hun_encode(word):
    return encode_pattern.sub(lambda x: hun_encodings[x[0]], word)

hun_decodings = {value: key for key, value in hun_encodings.items() if 'j' not in value}

decode_pattern = re.compile(
    '|'.join(re.escape(k) for k in sorted(hun_decodings, key=len, reverse=True))
)

def hun_decode(word):
    return decode_pattern.sub(lambda x: hun_decodings[x[0]], word)

#> Prettyprinting dictionaries <#

def dict_to_list(dy, mapping=lambda x: x):
    mapped_items = ((mapping(key), value) for key, value in dy.items())
    return sorted(mapped_items, key=lambda x: x[1], reverse=True)

def dict_to_list_hun_decode(dy):
    return dict_sort(dy, mapping=lambda x: hun_decode(''.join(x)))

def custom_pp(list):
    pp([(hun_decode(''.join(x[0])), x[1]) for x in list])

#> Parsing paradigm cell features <#

def cell_feature_set(xpostag):
    return frozenset(re.split(r'\]\[|\.', xpostag[4:].strip('[]')))
