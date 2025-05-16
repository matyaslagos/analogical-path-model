import nltk
import re
from nltk.tokenize import sent_tokenize

with open('norvig_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'\n+', ' ', text)
text = re.sub(r'\s+', ' ', text).strip()

sentences = sent_tokenize(text)

cleaned_sentences = []

for sentence in sentences:
    sentence = sentence.lower()
    sentence = re.sub(r'[\'"*=_]', r'', sentence)
    sentence = re.sub(r'([;,]|--|\.\.\.)', r'\n', sentence)
    sentence = re.sub(r'\n +', r'\n', sentence)
    sentence = re.sub(r' +\n', r'\n', sentence)
    sentence = re.sub(r'([^\s]),', r'\1 ,', sentence)
    sentence = re.sub(r'[.!?]+$', r'', sentence)
    cleaned_sentences.append(sentence)

with open('norvig_result.txt', 'w', encoding='utf-8') as f:
    for sentence in cleaned_sentences:
        f.write(sentence.strip() + '\n')