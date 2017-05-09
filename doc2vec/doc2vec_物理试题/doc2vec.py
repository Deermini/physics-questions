#!usr/bin/env python
# -*- coding:utf-8 -*-

from gensim.models import doc2vec
from gensim.models import Doc2Vec
import codecs

k=5
sentences = doc2vec.TaggedLineDocument("result.txt")
sentences2=codecs.open("result2.txt",'r',encoding="utf-8").readlines()
f1=codecs.open("label_level01.txt",'r',encoding="utf-8").readlines()
f2=codecs.open("label_level01_1.txt",'r',encoding="utf-8").readlines()


model = doc2vec.Doc2Vec(sentences, size=280, window=5, min_count=1, workers=8,iter=180)
#model =Doc2Vec(sentences, size=200, window=5, min_count=5, workers=8,iter=60)
# model.build_vocab(sentences)
model.train(sentences)
#print('#########', model.vector_size)

count=0
for i,li in enumerate(sentences2):
    list = li.split()
    inferred_vector = model.infer_vector(list)
    #print(inferred_vector)
    index = model.docvecs.most_similar([inferred_vector], topn=5)
    print(index)

    true_label=f2[i].split()
    true_label.sort()
    print("true label:",true_label)

    predictions={}
    for m,n in index:
        for key in f1[m].split():
            if key not in predictions.keys():
                predictions[key]=1
            else:
                predictions[key]+=1

    prediction=[]
    for key in predictions.keys():
        if predictions[key]>=k/2:
            prediction.append(key)

    prediction.sort()

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

