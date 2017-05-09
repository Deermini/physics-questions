# -*- coding: cp936 -*-
from gensim import corpora, similarities,models
import codecs
from sklearn import grid_search
from sklearn import cross_validation
from numpy import *

def classify(k=None):
    corpora_documents = []
    label_level=[]
    f=codecs.open("finalresult1.txt",'r',encoding="utf-8").readlines()
    f2=codecs.open("label_level03.txt",'r',encoding="utf-8").readlines()
    count=12901
    kfolds=10
    kf = cross_validation.KFold(count, n_folds=kfolds)
    for li in f:
        li=li.split()
        corpora_documents.append(li)
    for la in f2:
        la=la.split()
        label_level.append(la)
    corpora_documents=array(corpora_documents)
    label_level=array(label_level)
    sum_count=[]
    for train_index, test_index in kf:
        print(train_index, test_index)
        X_train, X_test = corpora_documents[train_index], corpora_documents[test_index]
        y_train, y_test = label_level[train_index], label_level[test_index]


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

        #test_data=[]
        count=0
        for i,li in enumerate(X_test):
            #li=li.split()
            test_corpus_1 = dictionary.doc2bow(li)
            test_corpus_tfidf=tfidf[test_corpus_1]
            test_corpus_Lsi=Lsi[test_corpus_tfidf]
            sims=index[test_corpus_Lsi]
            sort_sims=sorted(enumerate(sims),key=lambda x:-x[1])
            print(sort_sims[:k])
            predictions={}
            for m,n in sort_sims[:k]:
                for key in y_train[m]:
                    if key not in predictions.keys():
                        predictions[key]=1
                    else:
                        predictions[key]+=1
            true_label=y_test[i]
            prediction=[]
            for key in predictions.keys():
                if predictions[key]>=k/2:
                    prediction.append(key)
            true_label.sort()
            prediction.sort()
            print("true label:",true_label)
            if len(prediction)==0:
                dict_sorted=sorted(predictions.items(),key=lambda x:x[1],reverse=True)
                prediction.append(dict_sorted[0][0])
                if true_label==prediction:
                    count+=1
                    print(1)
            elif true_label==prediction:
                count+=1
                print(1)
            else:print(0)
            print("predict",prediction)
            print(predictions)
            print("#####################################")

        print("count:",count)
        sum_count.append(count)
    print("sum_count:",array(sum_count).mean())
classify(k=5)





