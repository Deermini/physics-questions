#!usr/bin/env python
# -*- coding:utf-8 -*-

import gensim
import os
import collections
import smart_open
import random
from numpy import *
from gensim.models import doc2vec

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus("result.txt"))
test_corpus = list(read_corpus("result2.txt", tokens_only=True))
#print(train_corpus[2])

model=gensim.models.doc2vec.Doc2Vec(size=180, min_count=1, iter=55)
model.build_vocab(train_corpus)
print(model.train(train_corpus))

doc_id=6
infered_vector=model.infer_vector(test_corpus[doc_id])
sims=model.docvecs.most_similar([infered_vector],topn=5)
print(sims)
