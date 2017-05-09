#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim

sentences=[]
f=open("E:/test/questions-words.txt")
for line in f.readlines():
    line=line.split()
    sentences.append(line)

#print(sentences)
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
print(model.similarity('woman', 'man'))
print(model.most_similar(positive=['man', 'king'], negative=['woman'],topn=5))


