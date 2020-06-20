import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
from sklearn.ensemble import RandomForestClassifier
from gensim.models.word2vec import Word2Vec
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
model_name='Word_Vector.model'
model = Word2Vec.load(os.path.join('..', 'models', model_name))
df=load_dataset("labeled_train")
df['clean_review'] = df.review.apply(clean_text)
vectorizer = CountVectorizer(max_features = 5000) #对所有关键词的term frequency(tf)进行降序排序，只取前5000个做为关键词集
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()#fit_transform转换成文档词频矩阵,toarray转成数组

forest = RandomForestClassifier(n_estimators=100)#随机森林分类
forest = forest.fit(train_data_features, df.sentiment)#开始数据训练

datafile = os.path.join('..', 'data', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t',escapechar='\\')

print('Number of reviews: {}'.format(len(df)))
df['clean_review'] = df.review.apply(clean_text)
print(df.head())

test_data_features = vectorizer.transform(df.clean_review).toarray()
test_data_features.shape

result = forest.predict(test_data_features)  # 预测结果
output = pd.DataFrame({'id': df.id, 'sentiment': result})
print(output.head())

output.to_csv(os.path.join('..', 'data', 'output1.csv'), index=False)

import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

