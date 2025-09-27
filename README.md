# 深度学习
## 〇、预备
### 损失函数：
1. softmax函数
2. 交叉熵函数
## 一、多层感知机MLP
多层感知机:多层感知机由多层神经元组成，最简单的深度网络。
### 感知机
只能做二分类问题（只能产生线性分割面？）。
- 决策定理：
 ![alt text](assets/README/image.png)
### 多层感知机
#### 激活函数：
 - sigmoid函数
  ![alt text](assets/README/image-1.png)
 - tanh函数
![alt text](assets/README/image-2.png)
- RELU函数
 ![alt text](assets/README/image-3.png)

#### K-则交叉验证
 ![alt text](assets/README/image-4.png)

#### 丢弃法：
 动机：增加模型鲁棒性
 做法：在层之间加入噪声
 无偏差加入扰动
 ![alt text](assets/README/image-5.png)
 加在全连接层的输出上，并且可以理解为一个正则项，只在训练的时候使用

#### 模型初始化
<!-- Xavier初始化（没看懂），后面再来看  -->

## 二、卷积神经网路
### 卷积层
- 卷积层对位置敏感
网络大小 nh*hw
卷积核 kh*kw
输出 （nh-kh+1）*（nw-kw+1）
填充 ph，pw
输出（nh-kh+ph+1）*（nw-kw+pw+1）
![alt text](assets/README/image-6.png)
![alt text](assets/README/image-7.png)
### 池化层
如果size为2并且步长为2，则会高宽减半
![alt text](assets/README/image-8.png)

## 三、线代卷积神经网路
### 经典卷积网络-LeNet
卷积神经网路
### AlexNet
更大更深的卷积神经网络
### VGG
将卷积和池化层集成成块来进行复制，打造更深的网络
### NiN
1. 卷积层的参数
ci*co*k²（对每一输入通道对应的每一个输出通道都有一个对应的卷积核）
2. 全连接层的参数
很多
为了解决全连接层参数很多的问题，提出NiN块
- NiN块
一个卷积层和两个1x1的卷积层
![alt text](assets/README/image-9.png)
### GoogLeNet
- inception块
![alt text](assets/README/image-10.png)
### 批量归一化
![alt text](assets/README/image-11.png)

### Resnet
lr, num_epochs, batch_size = 0.05, 10, 256
![alt text](assets/README/image-12.png)
增加epochs到三十次
![alt text](assets/README/resnet_l0p1_e30_b128.png)


## 三、Transformer
![alt text](assets/README/image-13.png)
### Multi-Head Attention 多头注意力
