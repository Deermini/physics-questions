# -*- coding: cp936 -*-

import codecs
import re

f=codecs.open("tag_label.txt",'r',encoding="utf-8")
f1=codecs.open("label.txt",'w',encoding="utf-8")
for li in f.readlines():
    li2=re.sub("TAG_IDS:---- ","",li)
    li3=re.sub("TAG_SUB_IDS:---- ","",li2)
    f1.write(li3)