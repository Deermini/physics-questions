# -*- coding: cp936 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
import codecs
from numpy import *


corpora_documents = []
label_level=[]
f=codecs.open("finalresult1.txt",'r',encoding="utf-8").readlines()
f2=codecs.open("label_level01.txt",'r',encoding="utf-8").readlines()

count=12901
kfolds=10
kf = cross_validation.KFold(count, n_folds=kfolds)

for li in f:
    # li=li.split()
    corpora_documents.append(li)
for la in f2:
    la=la.split()
    label_level.append(la)

corpora_documents=array(corpora_documents)
label_level=array(label_level)
print(label_level)

sum_count=[]
for train_index, test_index in kf:
    print(train_index, test_index)
    X_train, X_test = corpora_documents[train_index], corpora_documents[test_index]
    y_train, y_test = label_level[train_index], label_level[test_index]

    tfidf_vec=TfidfVectorizer()
    X_tfidf_train=tfidf_vec.fit_transform(X_train)
    X_tfidf_test=tfidf_vec.transform(X_test)
    print("X_tfidf_train:",X_tfidf_train.shape)


