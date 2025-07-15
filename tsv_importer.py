import csv
from collections import Counter

def tsv_import():
    freqs = Counter()
    with open('corpora/sztaki_corpus_2017_2018_0001_clean.tsv', newline='') as f:
        reader = csv.reader(
            (row for row in f if row.strip() and not row.startswith('#')), delimiter='\t'
        )
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and row[3].startswith('[/N]') and row[0].isalpha():
                freqs['<' + row[0].lower() + '>'] += 1
    return freqs
