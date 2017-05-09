#!usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk,re
from gensim.models import word2vec

def news_to_sentences(news):
    news_text=BeautifulSoup(news).get_text()
    tokenizer=nltk.data.load("tokenizers/punkt/english.pickle")
    raw_sentences=tokenizer.tokenize(news_text)
    print(raw_sentences)
    print("##################################################")
    sentences=[]
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]',' ',sent.lower().strip()).split())
    return sentences

news=fetch_20newsgroups(subset="all")
X,y=news.data,news.target
sentences=[]
for x in X:
    sentences+=news_to_sentences(x)
#print(sentences)
num_features=300
min_word_count=20
num_workers=8
context=5
downsampling=1e-3
model=word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)
model.init_sims(replace=True)
print(model.most_similar("morning"))
