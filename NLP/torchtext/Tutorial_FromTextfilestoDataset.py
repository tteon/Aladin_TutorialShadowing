# data from https://nlp.stanford.edu/projects/nmt/
import spacy
import pandas as pd
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split

english_txt = open('data/train2.en.txt', encoding='utf8').read().split('\n')
german_txt = open('data/train.de.txt', encoding='utf8').read().split('\n')

raw_data = {'English': [line for line in english_txt[1:1000]],
            'German': [line for line in german_txt[1:1000]]}

df = pd.DataFrame(raw_data, columns=['English', 'German'])

train, test = train_test_split(df, test_size=0.2)

# gonna output the format 'json'
train.to_json('train.json', orient='records', lines=True)
test.to_json('test.json', orient='records', lines=True)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
'''
import sys
sys.exit()
'''
spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de_core_news_sm')

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_ger, lower=True)

fields = {'English': ('eng', english), 'German' : ('ger', german)}

train_data, test_data = TabularDataset.splits(
    path='',
    train='train.json',
    test='test.json',
    format='json',
    fields=fields
)

english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=32,
    device='cuda'
)

for batch in train_iterator:
    print(batch)