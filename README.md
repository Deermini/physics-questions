# physics-questions
需安装的库:
--------------------
            anaconda
            gensim
            scikit-multilearn
            
项目描述：对高中物理试题进行多标记分类，因为每道物理试题都同时对应多个标签，因此是一个多标签分类的问题。

采用的方法：
--------------------------------------------
1、采用了scikit-multilearn中的一些多标记分类的方法，如BR、LP、rakelo及rakeld等等。
2、利用主题模型LDA、LSA找出最近的样例，然后再根据最近样例标签也应该类似的原理推断出标签。
3、利用word2vec、doc2vec把文本转换成向量的形式对文本进行分类。
4、结合tensorflow利用深度学习的一些方法对数据进行探索。

