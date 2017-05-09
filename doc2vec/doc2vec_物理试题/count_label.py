# -*- coding: cp936 -*-

import codecs

f=codecs.open("label_level03.txt",'r',encoding="utf-8")
label_count=[]
for li in f.readlines():
    li=li.split()
    for j in li:
        if j not in label_count:
            label_count.append(j)

print(label_count)
print(len(label_count))





