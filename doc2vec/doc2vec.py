#!usr/bin/env python
# -*- coding:utf-8 -*-

from gensim.models import doc2vec

sentences = doc2vec.TaggedLineDocument("result.txt")

model = doc2vec.Doc2Vec(sentences, size=20, window=3, min_count=3, workers=4,iter=60)
# model.build_vocab(sentences)
# model.train(sentences)
print('#########', model.vector_size)

#str = '如图  甲  所示  电路  测定  小灯泡  功率  被测小灯泡  额定电压  为  电阻  左右  实验室  有  如下  器材   电源  电压  恒为  电流表  电压表  开关  滑动变阻器   各  一个  导线若干'
str = '电压表  满偏  时  通过  该表  的  电流  是  半  偏时  通过  该表  电流  的  两倍  某  同学  利用  这一  事实  测量  电压表  的  内阻  半偏法  实验室  提供  的  器材  如下  待测电压表  量程 内阻  约  为  电阻箱  最大阻值  为   滑动变阻器  最大阻值   额定电流  电源电动势   内阻  不计   开关  个  导线若干'
#str ="在  测  额定电压  为   小灯泡  的  电功率  的  实验  中 阳阳  同学  已  连接  好  如图所示  的  部分  电路 "
list = str.split()
print(list)
inferred_vector = model.infer_vector(list)
print(inferred_vector)
sims = model.docvecs.most_similar([inferred_vector], topn=50)
print(sims)

print(model.most_similar("摩擦力"))
