# -*- coding: cp936 -*-

from gensim import corpora, models,similarities
import logging
import codecs

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
documents=[]
f=codecs.open("result.txt",'r',encoding="utf-8").readlines()
for li in f:
    li=li.split()
    documents.append(li)

dictionary=corpora.Dictionary(documents)
# print(dictionary)
# print(dictionary.token2id)

corpus=[dictionary.doc2bow(text) for text in documents]
#print(corpus[:4])

tfidf=models.TfidfModel(corpus)
corpus_tfidf=tfidf[corpus]
# for doc in corpus_tfidf:
#     print(doc)

Lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=30)
#print(Lsi.print_topics(30))

corpus_lsi=Lsi[corpus_tfidf]
# for doc in corpus_lsi:
#     print(doc)

Lda=models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=30)
corpus_lda=Lda[corpus_tfidf]
# for doc in corpus_lda:
#     print(doc)

index=similarities.MatrixSimilarity(corpus_lsi)

query=documents[0]
query_bow=dictionary.doc2bow(query)
print(query_bow)
query_tfidf=tfidf[query_bow]

query_lsi=Lsi[query_tfidf]
sims=index[query_lsi]
#print(list(enumerate(sims)))

sort_sims=sorted(enumerate(sims),key=lambda x:-x[1])
print(sort_sims)


