# Text Classification with CNNs in PyTorch
主要是留下一个基于CNN的文本分类模型，方便以后快速测试等等；模型结构参考Paper^[1]。代码思路则是参考^[2]。这里自己重新
手撸一遍。顺便顺理一遍一维卷积。

## 网络结构
[Convolutional Neural Networks for Sentence Classification](Convolutional Neural Networks for Sentence Classification)

## 一个构造一维卷积的输入与输出的列子
```
1维卷积对于处理时间序列数据有重要意义，具体讲解如下：

给定一个数据集，数据维度为3000行7列; 3000个样本，7个特征，5个类别，利用一维卷积进行分类

首先对数据进行处理，将其转换为3维数据，3000×时间步长×特征数，从而使得数据的格式能被keras接受。这里取时间步长10

model.add(Conv1D(100,2))

添加第一个一维卷积层，100个卷积核，卷积核大小为2，10－2+1=9，输出数据9行100列，channel为7

model.add(Conv1D(100,2))

添加第一个一维卷积层，100个卷积核，卷积核大小为2，9－2+1=8，输出数据8行100列，channel为7

model.add(Maxpooling1D(3,2))池化核大小为3，步长为2，(8-3＋1)/2=3,输出数据3行100列，channel为7

(疑问:假设池化核为2，步长为2，则(8-2+1)/2=3.5，这时维度有待于keras上实验)

注意:若model.add(Maxpooling1D(2))，则池化核大小为2，步长也为2。

model.add(Conv1D(160,2))

添加第一个一维卷积层，160个卷积核，卷积核大小为2，3－2+1=2，输出数据2行160列，channel为7

此时卷积核的权重矩阵为100行160列，

2行100列的矩阵与100行160列的矩阵相乘，即可得到2行160列的矩阵
```

## reference
[1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
[2] [Text Classification with CNN in PyTorch](https://towardsdatascience.com/text-classification-with-cnns-in-pytorch-1113df31e79f)
[3] [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)