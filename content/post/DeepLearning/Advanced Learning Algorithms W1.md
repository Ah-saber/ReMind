# 神经元与大脑

神经网络早期研究动机是模拟人类或者动物的大脑，尽管到现在和一般认知里大脑的运作有很大的差别，依旧是以其为重要的指导之一

- 开始于1950
- 在 1980 - 1990 年有一定应用
- 2005年出现深度学习

## 神经元

![Pasted image 20240304074541](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240304074541.png)
大脑的神经元示意图如上，可以概括为接受输入信号，再将处理后的信号传输给下一个神经元的模型
![Pasted image 20240304074800|200](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240304074800.png)
仅仅抽象出一个不够，一般需要多个同样的神经元来做相同的处理，这些神经元构成一层处理层

事实上，大脑的机制人类并不清楚，上述的模拟不一定正确，这说明对于使用尚未完善的生物学构建深度学习算法并不是合理的道路，现实中往往使用工程的思想构建算法

为什么今年深度学习突飞猛进？
![Pasted image 20240304075558](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240304075558.png)
数据量剧增，对于传统AI算法来说（逻辑回归，线性回归等），性能上受数据影响小，无法再有明显提升的，但是，深度学习却深受影响，加上英伟达GPU在深度学习上的成功，推动了其进展


## Demand Prediction

以此为例理解神经网络

假设预测一件衣服的需求
![Pasted image 20240304084417](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240304084417.png)
- 分析有哪些方面会影响衣服的需求量
- 分析这些方面取决于什么特征
- 构建层次图
其中，可以将每个计算抽象成一个神经元，同层次的神经元抽象成一**层（layer）** ,其输出再计算出最终结果
简单的抽象为三层，输入层（input layer) 隐藏层（hidden layer）和输出层（outlayer）
	隐藏层因为在数据集中其输入值与输出值的正确数值是未知的，因此称为隐藏
一般的，神经元将读取全部特征，但是并非每个特征都对神经元有效，通过 **参数** 来对特征进行选择，深度学习不需要决定那些特征对隐藏层的那些输出有作用，而是通过算法自己决定

>[!概念]
>**激活** _activation_
>将神经元的输出称为这个神经元的激活
>
>多层感知器 _multilayter perceptron_
>就只是多层隐藏层的神经网络
>





## Example Recognizing Images

![Pasted image 20240304090502](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240304090502.png)
当应用某个人脸识别深度学习算法时，将隐藏层的输出可视化，可以看到如图所示的
- 第一个输出说明在找某些边缘线条
- 第二个再找某些脸部器官
- 第三个在找粗略的人脸

这并不是设置好的，而是通过算法自己确定的查找特征和输出结果

## Neural network Layer

称输入层为第0层，隐藏层从第1层开始计数，一般用$X^{[n]}$来表示某数与第几层有关（应用在公式和输出，输入不包括），每层有输入（除0层）有输出，传递到下一层
一般计数神经网络的层数不包括输入层


**激活函数**
假设使用sigmoid函数计算某层的输出值，此时sigmoid函数也称为激活函数就 (activation function)


## Inference: making predictions(forward propagation)

前向传播

传播每层输出的激活（每层的输出）给下一层，最终计算出结果，称为前向传播

```Python
# TensorFlow

x = np.array([[200.0, 17.0]])
layer_1 = Dense(unit = 3, activation = 'sigmoid')
a1 = layer_1(x)

layer_2 = Dense(unit = 1, activation = 'sigmoid')
a2 = layer_2(a1)
```

**Dense 稠密层**
又称全连接层，是神经网络的一种
特点是两层神经元之间全连接，通过权重矩阵和偏置向量求得z，再通过非线性激活函数进行非线性转换后输出结果

## Data in TensorFlow

首先明确，现阶段已经认识大概三种深度学习框架：
- _Google - TensorFlow_
- _Facebook等 - PyTorch
- _baidu - Paddle_

Deep Learning使用的框架是Google TensorFlow，其中一些数据结构类型定义不同，一些语法有区别

一些Python代码概念

```Python
# 两个[]代表二维
x = np.array([[200.0, 17.0]])
tf.Tensor([[0.2 0.7 0.3]]), shape = (1,3), dtype=float32  #定义张量，矩阵大小，数据类型

"""
张量就是矩阵，是TensorFLow定义的一种利于矩阵计算和存储的数据结构
"""

#相互转换
a1.numpy();  # 将a1张量转化为numPy数组
```

对于TensorFlow，是以tensor（张量）数据结构展开的深度学习框架：粗浅的认为，张量与张量之间可以形成工作流，高效的组合深度学习的各个层级。

```Python
# TensorFlow example
layer_1 = Dense(units=3, activation = "sigmoid")
layer_2 = Dense(units = 1, activation = "sigmoid")
model = Sequential([layer_1, layer_2])
```

```Python
# 定于dense层函数，计算返回结果
def dense(a_in , W, b)
	units = W.shape[1] # 取第一行的元素的维度
	a_out= np.zeros(units)
	for j in range(units):
		w = W[:, j]
		z = np.dot(w, a_in) + b[j]
		a_out[j] = g(z)
	return a_out
```

利用矩阵做运算是基础，大多的隐藏层都是函数实现，基本上是依托某些原理的矩阵运算，一般将layer合写成一个函数


# AGI deeper and Matrix deeper

AI有两种认识
- ANI  狭义AI
	作用在特定场合，例如自动驾驶，自动农场之类的AI
- AGI  广义AI
	大众广泛认识的AI，能做所有人类能做到事

ANI在如今已经有很大的进展，但是AGI还没有，着一部分是因为现今的大脑研究没有进展，人类不知道神经元究竟是怎么运作的，要用什么样的算法实现

事实证明神经元是存在某种算法可以实现对不同的输入的处理，研究这种算法是AGI进展的一大方向

## Matrix

```Python
A = np.array([[1,-1,0,1],
			 [2,-1,0,2]])
AT = A.T #转置

W = np.array([[3,5,7,9],
			[4,6,8,0]])

Z = np.matmul(AT,W) #矩阵乘法
# Z = AT @ W
```