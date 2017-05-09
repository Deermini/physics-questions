# -*- coding: cp936 -*-

from gensim.models.wrappers import fasttext

model = fasttext.FastText.train('fasttext', corpus_file='finalresult.txt')

