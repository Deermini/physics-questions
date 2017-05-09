#!usr/bin/env python
# -*- coding:utf-8 -*-

from gensim.models import doc2vec
import numpy as np


sentences = doc2vec.TaggedLineDocument("result.txt")

model = doc2vec.Doc2Vec(sentences, size=20, window=3, min_count=3, workers=4,iter=60)
# model.build_vocab(sentences)
model.train(sentences)
print('#########', model.vector_size)

corpus = model.docvecs
np.save("d2v.corpus.arff", corpus)
