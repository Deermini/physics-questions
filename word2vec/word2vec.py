# -*- coding: cp936 -*-
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

sentences = LineSentence('result.txt')
model = word2vec.Word2Vec(sentences, size=50, window=6, min_count=3, workers=4)
model.save("model.txt")
#
model=word2vec.Word2Vec.load('model.txt')

print(model.most_similar("����Ħ����"))
print(model.similarity('С��', '����ʱ��'))
# print(model.score(["��  ��  �Ϸ�  ��  �Ÿ�Ӧǿ��  ��С  Ϊ".split()]))
print("��Ħ������")


