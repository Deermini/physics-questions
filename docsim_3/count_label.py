# -*- coding: cp936 -*-

import codecs

f=codecs.open("label.txt",'r',encoding="utf-8")
label_count=[]
for li in f.readlines():
    li=li.split()
    for j in li:
        if j not in label_count:
            label_count.append(j)

print(label_count)
print(len(label_count))


#================================#
"""为标签建立一个数字索引，以便可以用多标签的评价指标来对其进行度量"""
# import codecs
# from numpy import *
#
#
# f=codecs.open("label_level01.txt",'r',encoding="utf-8")
# label_names=[]
# label_dict={}
# for li in f.readlines():
#     li=li.split()
#     for j in li:
#         if j not in label_names:
#             label_names.append(j)
# #print(label_names)
# for i,la in enumerate(label_names):
#     label_dict[la]=i
#
# print(label_dict)
# label_=(sorted(label_dict.items(),key=lambda x:x[1]))
# print(label_)
#
# y=zeros((12901,len(label_names)))
# f1=codecs.open("label_level01.txt",'r',encoding="utf-8")
# i=0
# for li in f1.readlines():
#     li=li.split()
#     for j in li:
#         y[i,label_dict[j]]=1
#     i+=1
#
# print(y)