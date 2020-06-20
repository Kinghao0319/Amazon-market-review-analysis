import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
from sklearn.ensemble import RandomForestClassifier
#合理思路：先导入无标签数据生成word2vec向量，后导入有标签的形成分类器
from sklearn.feature_extraction.text import CountVectorizer


def load_dataset(name, nrows=None):
    datasets = {
        "unlabeled_train": "unlabeledTrainData.tsv",
        "labeled_train": "labeledTrainData.tsv",
        "test": "testData.tsv"
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join("..", "data", datasets[name])
    df = pd.read_csv(data_file, sep="\t", escapechar="\\", nrows=nrows)
    return df

eng_stopwords = {}.fromkeys([ line.rstrip() for line in open('../data/stopwords.txt')])
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)

df = load_dataset('unlabeled_train')
print('Number of unlabeledTrainData.csv reviews: {}'.format(len(df)))
df['clean_review'] = df.review.apply(clean_text)
review_part = df['clean_review']
review_part.shape
import warnings
warnings.filterwarnings("ignore")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def split_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]#清洗
    return sentences
sentences = sum(review_part.apply(split_sentences), [])
print('{} reviews -> {} sentences'.format(len(review_part), len(sentences)))
sentences_list = []
for line in sentences:
    sentences_list.append(nltk.word_tokenize(line))
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
model_name = 'Word_Vector.model'
from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences_list, workers=num_workers, size=num_features, min_count = min_word_count, window = context)
model.init_sims(replace=True)
model.save(os.path.join('..', 'models', model_name))
