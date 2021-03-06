# -*- coding: cp936 -*-
from gensim import corpora, similarities
import codecs
from sklearn import grid_search



def classify(k=None):
    corpora_documents = []
    f=codecs.open("result.txt",'r',encoding="utf-8").readlines()
    f2=codecs.open("label_level1.txt",'r',encoding="utf-8").readlines()
    for li in f[:12106]:
        li=li.split()
        corpora_documents.append(li)

    #生成字典和向量语料
    dictionary = corpora.Dictionary(corpora_documents)
    #dictionary.save('dictionary.dict')
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    index1 = similarities.Similarity('-Similarity-index', corpus, num_features=25000)
    #index1.save('documents.index')

    #dictionary=corpora.Dictionary.load('dictionary.dict')
    #index1= similarities.Similarity.load('documents.index')
    #test_data=[]
    count=0
    for i,li in enumerate(f[12106:]):
        li=li.split()
        test_corpus_1 = dictionary.doc2bow(li)
        #k=11
        index1.num_best = k
        index=index1[test_corpus_1]
        print(index)
        predictions={}
        for m,n in index:
            for key in f2[m].split():
                if key not in predictions.keys():
                    predictions[key]=1
                else:
                    predictions[key]+=1
        true_label=f2[12106+i].split()
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

classify(k=5)

# parameters={"k":[3,4,5,6,7]}
# clf=grid_search.GridSearchCV(classify(),parameters)
# print(clf.best_params_,clf.best_score_)




