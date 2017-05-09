# -*- coding: cp936 -*-

from gensim import corpora, similarities,models
import codecs
from sklearn import grid_search
from sklearn import cross_validation
from numpy import *
import sklearn
from gensim.models.hdpmodel import HdpModel


corpora_documents = []
label_level=[]
filename="label_level02.txt"
f=codecs.open("finalresult1.txt",'r',encoding="utf-8").readlines()
f2=codecs.open(filename,'r',encoding="utf-8").readlines()
#y2,label_dict,n_x=y_label(filename)
count1=12901
kfolds=10
kf = cross_validation.KFold(count1, n_folds=kfolds)
for li in f:
    li=li.split()
    corpora_documents.append(li)
for la in f2:
    la=la.split()
    label_level.append(la)
corpora_documents=array(corpora_documents)
label_level=array(label_level)

#生成字典和向量语料
dictionary = corpora.Dictionary(corpora_documents)
#dictionary.save('dictionary.dict')
corpus = [dictionary.doc2bow(text) for text in corpora_documents]
tfidf=models.TfidfModel(corpus)
corpus_tfidf=tfidf[corpus]

hdp=HdpModel(corpus_tfidf,id2word=dictionary)
corpus_hdp=hdp[corpus_tfidf]
index=similarities.MatrixSimilarity(corpus_hdp)

print(hdp.print_topics(num_topics=20, num_words=10))







