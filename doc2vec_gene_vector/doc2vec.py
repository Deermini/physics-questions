#!usr/bin/env python
# -*- coding:utf-8 -*-

from gensim.models import doc2vec
import codecs
from numpy import *


def y_label(filename):
    f8=codecs.open(filename,'r',encoding="utf-8")
    label_names=[]
    label_dict={}
    for li in f8.readlines():
        li=li.split()
        for j in li:
            if j not in label_names:
                label_names.append(j)
    #print(label_names)
    for i,la in enumerate(label_names):
        label_dict[la]=i

    # print(label_dict)
    # label_=(sorted(label_dict.items(),key=lambda x:x[1]))
    # print(label_)

    y=zeros((12901,len(label_names)))
    f9=codecs.open(filename,'r',encoding="utf-8")
    i=0
    for li in f9.readlines():
        li=li.split()
        for j in li:
            y[i,label_dict[j]]=1
        i+=1
    return y


sentences = doc2vec.TaggedLineDocument("result.txt")

model = doc2vec.Doc2Vec(sentences, size=280, window=5, min_count=1, workers=8,iter=168)
# model.build_vocab(sentences)
model.train(sentences)

filename="label_level01.txt"
corpus = model.docvecs
y=y_label(filename)

vector=[]
for i,vec in enumerate(corpus):
    if i<12901:
        vector.append(vec)
    else:
        break

final_vector=hstack((vector,y))
f=open("data.arff",'w')
for li in final_vector:
    for li2 in li:
        f.write(str(li2)+',')
    f.write("\n")
