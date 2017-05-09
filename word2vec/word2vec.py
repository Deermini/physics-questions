# -*- coding: cp936 -*-
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

sentences = LineSentence('result.txt')
model = word2vec.Word2Vec(sentences, size=50, window=6, min_count=3, workers=4)
model.save("model.txt")
#
model=word2vec.Word2Vec.load('model.txt')

print(model.most_similar("滑动摩擦力"))
print(model.similarity('小明', '打点计时器'))
# print(model.score(["在  轴  上方  有  磁感应强度  大小  为".split()]))
print("动摩擦因数")


