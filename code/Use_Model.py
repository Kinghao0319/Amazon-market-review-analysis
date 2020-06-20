import os

from gensim.models import Word2Vec


model_name = 'Word_Vector.model'
model = Word2Vec.load(os.path.join('..', 'models', model_name))
#
# print(model.most_similar("enthusiastic"))
# print(model.similarity("enthusiastic","product"))
print(model.wv["disappointed"])