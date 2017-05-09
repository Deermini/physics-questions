#!usr/bin/env python
# -*- coding:utf-8 -*-

import gensim
import os
import collections
import smart_open
import random
from numpy import *

# Set file names for train and test data
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

# print(train_corpus[:2])
# print(test_corpus[:2])

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)
print(model.train(train_corpus))

# test_vector=model.infer_vector(['only', 'you', 'can', 'prevent', 'forrest', 'fires'])
#print(test_vector)

# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)
#     second_ranks.append(sims[1])

    # print("Document",(doc_id),train_corpus[doc_id].words,"\n")
    # for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    #     print(label,sims[0],train_corpus[sims[index][0]].words)
    # break
    # print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    # for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    #     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


# doc_id = random.randint(0, len(train_corpus))
# print('Train Document',doc_id,train_corpus[doc_id].words,"\n")
# sim_id = second_ranks[doc_id]
# print('Similar Document',sim_id,train_corpus[sim_id[0]].words)


#doc_id=random.randint(0,len(test_corpus))
doc_id=6
infered_vector=model.infer_vector(test_corpus[doc_id])
sims=model.docvecs.most_similar([infered_vector],topn=len(test_corpus))

print("test Document",(doc_id),train_corpus[doc_id].words,"\n")
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(label,sims[index],train_corpus[sims[index][0]].words,"\n")