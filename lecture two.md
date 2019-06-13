Stanford cs224n

https://web.stanford.edu/class/cs224n/index.html#coursework 这里是slides 和课程的schedule之类的信息链接

Lecture two



## 一：知识储备

logistic regression 逻辑回归

通常用来做二分类问题而且可以给出相应的概率。

i.e. 垃圾邮件分类0/1，银行是否通过用户的信用卡申请0/1.

但也可以拓展到多分类问题，此处略过。。



首先逻辑回归问题中，我们已知每个样本的位置（二维坐标轴中,x1为横轴，x2为纵轴），还有它正确的标签。此时我们的目的是找到一个决策边界把两个类别分开。 为了达到这个目的可以用二元一次方程来表达这条分界线 z= w0 + w1x1 +w2x2 。（当然还有非线性的可以用其他函数来表达）然后通过sigmod函数，可以获得一个0-1之间的数字。

我们知道cost function（这里不推了，有图的话很明显）。然后cost 对 w求导。

利用gradient descent不断更新w，逼近最优解，可能很难，所以我们在写code对时候要设置一个最大的iteration的次数。



##  二：optimizer 的选择

1. Batch Gradient Descent

2. Stochastic Gradient Descent

3. Mini Batch Gradient Descent

   

   

   从code上可以看出SGD 与 BGD 比起来花费更小。

    

   

   BGD :

   ```
   for i in range(nb_epochs):
     params_grad = evaluate_gradient(loss_function, data, params)
     params = params - learning_rate * params_grad
   ```

   

   

   SGD:

   ```
   for i in range(nb_epochs):
     np.random.shuffle(data)
     for example in data:
       params_grad = evaluate_gradient(loss_function, example, params)
       params = params - learning_rate * params_grad
   ```

而且learning rate也要小一点。





BGD：一个epoch中 **所有** 都需要参与。



SGD：一个epoch不需要所有的都参与，window为1，然后不断的换center word。



MBGD：一个epoch不需要所有的都参与，我们设置一个bunch，每次只考虑这个bunch的范围，一般选bunch的size是32或者64的，然后不断的变化center word。



MBGD的advantage：

最后会得到很多bunch的结果进行average，这样会减少noise。

 对GPU也好。



## 三：回顾

两种模型：skip-gram model 

​                    continuous bag-of-words



两种提高训练效率方法：softmax

​                                          negative sampling







negative sampling 负采样，除了正样本都是负样本， 所以可以看作一个二分类问题。

i.e. 正确的是：今天去打篮球   

​      错误的话是：今天去打人

这时我们可以指定一个正样本和负样本，正样本是篮球，其他的是负样本。但是负样本是非常多的。





首先构建负样本时，选择出现频数高的。 





N-Gram: unigram, bigram, trigram 

















