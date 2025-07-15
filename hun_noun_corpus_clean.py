import csv
from string import punctuation

with open('hun_noun_corpus.csv', newline='', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    clean_word = lambda s: s.strip(punctuation).lower()
    word_freq_dict = {clean_word(row['key']): int(row['value']) for row in reader}

normalized_word_freq_dict = {}

long_replacements = {
    'ccs': '11',
    'ddz': '22',
    'ggy': '33',
    'nny': '44',
    'ssz': '55',
    'tty': '66',
    'zzs': '77',
    'dzs': '8',
    'lly': 'jj'
}

short_replacements = {
    'cs': '1',
    'dz': '2',
    'gy': '3',
    'ny': '4',
    'sz': '5',
    'ty': '6',
    'zs': '7',
    'ly': 'j'
}

for word, freq in word_freq_dict.items():
    for old, new in long_replacements.items():
        word = word.replace(old, new)
    for old, new in short_replacements.items():
        word = word.replace(old, new)
    normalized_word_freq_dict[word] = freq

with open('normalized_hun_noun_corpus.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['key', 'value'])  # header
    for k, v in normalized_word_freq_dict.items():
        writer.writerow([k, v])
