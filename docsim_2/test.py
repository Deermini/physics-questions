# -*- coding: cp936 -*-
import codecs

f=codecs.open("label.txt",encoding="utf-8")
for i in f.readlines():
    print(i.split())