# -*- coding: cp936 -*-

from gensim import corpora, similarities,models
import codecs
from sklearn import grid_search
from sklearn import cross_validation
from numpy import *
import sklearn

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
    return y,label_dict,len(label_names)

# y=y_label()
# print(y)


def classify(k4=None):
    corpora_documents = []
    label_level=[]
    filename="label_level04.txt"
    f=codecs.open("finalresult1.txt",'r',encoding="utf-8").readlines()
    f2=codecs.open(filename,'r',encoding="utf-8").readlines()
    y2,label_dict,n_x=y_label(filename)
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
    sum_count=[]
    k=[];k1=[]
    k2=[];k3=[]
    for train_index, test_index in kf:
        #print(train_index, test_index)
        X_train, X_test = corpora_documents[train_index], corpora_documents[test_index]
        y_train, y_test = label_level[train_index], label_level[test_index]
        y2_train, y2_test = y2[train_index],y2[test_index]

        #生成字典和向量语料
        dictionary = corpora.Dictionary(X_train)
        #dictionary.save('dictionary.dict')
        corpus = [dictionary.doc2bow(text) for text in X_train]
        tfidf=models.TfidfModel(corpus)
        corpus_tfidf=tfidf[corpus]

        # Lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=500)
        # corpus_lsi=Lsi[corpus_tfidf]
        # index=similarities.MatrixSimilarity(corpus_lsi)

        Lda=models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=500)
        corpus_lda=Lda[corpus_tfidf]
        index=similarities.MatrixSimilarity(corpus_lda)

        #test_data=[]
        count=0
        y_prediction=zeros((len(test_index),n_x))
        for i,li in enumerate(X_test):
            #li=li.split()
            test_corpus_1 = dictionary.doc2bow(li)
            test_corpus_tfidf=tfidf[test_corpus_1]
            test_corpus_Lsi=Lda[test_corpus_tfidf]
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
            true_label=y_test[i]
            prediction=[]
            for key in predictions.keys():
                if predictions[key]>=k4/2:
                    prediction.append(key)
            true_label.sort()
            prediction.sort()
            # print("true label:",true_label)
            if len(prediction)==0:
                dict_sorted=sorted(predictions.items(),key=lambda x:x[1],reverse=True)
                prediction.append(dict_sorted[0][0])
                if true_label==prediction:
                    count+=1
                    #print(1)
            elif true_label==prediction:
                count+=1
                #print(1)
            else:
                false=0
                #print(0)
            # print("predict",prediction)
            # print(predictions)
            for label in prediction:
                # print('label_dict[label]:',label_dict[label])
                # print(i,len(y_prediction[i]))
                y_prediction[i,label_dict[label]]=1
            #print("#####################################")
        #print("count:",count)
        sum_count.append(count)
        hammingloss= sklearn.metrics.hamming_loss(y2_test, y_prediction)
        jaccard= sklearn.metrics.jaccard_similarity_score(y2_test, y_prediction)
        f1score= sklearn.metrics.f1_score(y2_test, y_prediction,average='micro')
        zerooneloss= sklearn.metrics.zero_one_loss(y2_test, y_prediction)
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

    #print("sum_count:",array(sum_count).mean())


classify(k4=5)











