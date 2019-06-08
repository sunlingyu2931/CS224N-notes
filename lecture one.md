Stanford cs224n

https://web.stanford.edu/class/cs224n/index.html#coursework 这里是slides 和课程的schedule之类的信息链接

Lecture one




# 一：知识储备

在lecture 1 的学习过程中我觉得用到了很多不同方面的知识，以下是一些我原有的知识储备，但学习过程中也发现有些知识有所遗忘，于是又进行了查缺补漏，但最后确实是收获满满。

1. Bayesian (贝叶斯) 中的 posterior distribution (后验分布)和likelihood (似然函数) 它们各自的含义，联系与区别。

2. one-hot 独热编码
3. Softmax founction
4. Time series 





# 二：word2vec

两个模型：

1. Skip-gram model 跳字模型

   

   首先，要区分场景，如词典中中心词Wc, 背景词Wo

   然后根据time series很好理解，可以把  **likelihood** 写出来，两边取log，记得取log之后以前想乘的就会变成相加。T 是将它们平均。负号是 **minimize cost function**。

   

   我们要始终记得 Skip-gram 这个模型是一种中心词去predict背景词，p(Wo|Wc) 。

   

   p(Wo|Wc)  **传入softmax函数** 。对 **objective function求偏导**。



2. Continuous bag-of-words连续词袋模型

 



# 三：总结

第一节课内容其实并不多，老头也没有口音，很好懂，就是有很多知识的遗忘，再推导过程中暂停很多，还好最后都搞明白了。

这节课并不难，大家加油。


各位胖友如果发现有错或者迷惑的地方请comment在下面谢谢！！


