# -*- coding: cp936 -*-

from gensim import corpora, similarities,models
import codecs
from sklearn import grid_search
from sklearn import cross_validation
from numpy import *
import sklearn
from sklearn.cross_validation import train_test_split


def classify(k4=None):
    corpora_documents = []
    label_level=[]
    filename="RSXTags.txt"
    f=codecs.open("RSXTiMu.txt",'r',encoding="utf-8").readlines()
    f2=codecs.open(filename,'r',encoding="utf-8").readlines()
    #y2,label_dict,n_x=y_label(filename)
    count1=9502
    kfolds=10
    kf = cross_validation.KFold(count1, n_folds=kfolds,shuffle=False,random_state=33)
    for li in f:
        li=li.split()
        corpora_documents.append(li)
    for la in f2:
        la=la.split()
        label_level.append(la)
    corpora_documents=array(corpora_documents)
    label_level=array(label_level)
    sum_count=[]
    k=[];k1=[]
    k2=[];k3=[]
    for train_index, test_index in kf:
        #print(train_index, test_index)
        X_train, X_test = corpora_documents[train_index], corpora_documents[test_index]
        y_train, y_test = label_level[train_index], label_level[test_index]

        #X_train, X_test,y_train, y_test=train_test_split(corpora_documents,label_level,test_size=0.1,random_state=33)
        #生成字典和向量语料
        dictionary = corpora.Dictionary(X_train)
        #dictionary.save('dictionary.dict')
        corpus = [dictionary.doc2bow(text) for text in X_train]
        tfidf=models.TfidfModel(corpus)
        corpus_tfidf=tfidf[corpus]

        Lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=500)
        corpus_lsi=Lsi[corpus_tfidf]
        index=similarities.MatrixSimilarity(corpus_lsi)

        # Lda=models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=500)
        # corpus_lda=Lda[corpus_tfidf]
        # index=similarities.MatrixSimilarity(corpus_lda)

        count=0
        y_prediction=[]
        for i,li in enumerate(X_test):
            #li=li.split()
            test_corpus_1 = dictionary.doc2bow(li)
            test_corpus_tfidf=tfidf[test_corpus_1]
            test_corpus_Lsi=Lsi[test_corpus_tfidf]
            sims=index[test_corpus_Lsi]
            sort_sims=sorted(enumerate(sims),key=lambda x:-x[1])
            #print(sort_sims[:k4])
            predictions={}
            for m,n in sort_sims[:k4]:
                for key in y_train[m]:
                    if key not in predictions.keys():
                        predictions[key]=1
                    else:
                        predictions[key]+=1
            dict_sorted=sorted(predictions.items(),key=lambda x:x[1],reverse=True)
            y_prediction.append(dict_sorted[0][0])
            # print("True_label:",y_test[i][0])
            # print("prediction:",dict_sorted[0][0])
            if y_test[i][0]==dict_sorted[0][0]:
                count+=1
        print(1.0*count/len(y_test))

classify(k4=1)











