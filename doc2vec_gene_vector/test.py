#!usr/bin/env python
# -*- coding:utf-8 -*-

from gensim.models import doc2vec
import codecs
from numpy import *
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.ensemble.rakeld import RakelD
from skmultilearn.ensemble.rakelo import RakelO
import sklearn.metrics
#from skmultilearn.dataset import Dataset
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from scipy.sparse import hstack, coo_matrix,vstack,lil_matrix,csr_matrix,csc_matrix


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

model = doc2vec.Doc2Vec(sentences, size=180, window=5, min_count=1, workers=8,iter=168)
# model.build_vocab(sentences)
model.train(sentences)

filename="label_level01.txt"
corpus = model.docvecs
y=y_label(filename)

X=[]
for i,vec in enumerate(corpus):
    if i<12901:
        X.append(vec)
    else:
        break

X=array(X)
k=[];k1=[]
k2=[];k3=[]
count=12901
kfolds=10
kf = cross_validation.KFold(count, n_folds=kfolds)
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RakelD(LabelPowerset(RandomForestClassifier()),labelset_size=3,require_dense=[True, True])
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    y_test=csr_matrix(y_test)
    pre=csr_matrix(predictions)
    for li in range(6,12,1):
        print(pre[li,:])
        print()
        print(y_test[li,:])
        print( "--------------------------")

    hammingloss= sklearn.metrics.hamming_loss(y_test, predictions)
    jaccard= sklearn.metrics.jaccard_similarity_score(y_test, predictions)
    f1score= sklearn.metrics.f1_score(y_test, predictions,average='micro')
    zerooneloss= sklearn.metrics.zero_one_loss(y_test, predictions)
    print("hammingloss,jaccard,f1score,zerooneloss:", hammingloss, jaccard, f1score, zerooneloss)
    k.append(hammingloss)
    k1.append(jaccard)
    k2.append(zerooneloss)
    k3.append(f1score)


print("hamming_loss mean:", array(k).mean())
print("hamming_loss var:", array(k).var())

print("jaccard mean:",array(k1).mean())
print("jaccard var:", array(k1).var())

print("f1_score mean:", array(k3).mean())
print("f1_score var:", array(k3).var())

print("zero_one_loss mean:", array(k2).mean())
print("zero_one_loss var:", array(k2).var())







