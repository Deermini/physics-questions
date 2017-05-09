# -*- coding: cp936 -*-
import codecs

f=codecs.open("label.txt",encoding="utf-8")
f2=codecs.open("label_level04.txt",'w',encoding="utf-8")
for li in f.readlines():
    label_level2=[]
    li=li.split()
    for i in li:
        f1=codecs.open("tags.txt",'r',encoding="utf-8")
        for li1 in f1.readlines():
            li1=li1.split()
            if i==li1[0]:
                if "4" in li1:
                    index=li1.index("4")
                    label_level2.append(li1[index-1])
                # elif "4" in li1:
                #     index=li1.index("4")
                #     label_level2.append(li1[index-1])
                elif "3" in li1:
                    index=li1.index("3")
                    label_level2.append(li1[index-1])
                elif "2" in li1:
                    index=li1.index("2")
                    label_level2.append(li1[index-1])
                else:
                    index=li1.index("1")
                    label_level2.append(li1[index-1])

    for j in set(label_level2):
        f2.write(j+" ")
    f2.write("\n")
