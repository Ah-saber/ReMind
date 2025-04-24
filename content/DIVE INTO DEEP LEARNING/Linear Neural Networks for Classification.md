# Softmax

深度学习只有软聚类，经过Softmax Regression之后得到各个概率值

$$ 
\hat{y} = \text{softmax}(o)
\quad where \,\,\, \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)} 
$$

由于softmax保留了参数之间的顺序，也保留了原本的概率顺序，所以不需要再次计算哪个类别的概率更大

>The idea of a softmax dates back to Gibbs ([1902](https://d2l.ai/chapter_references/zreferences.html#id81 "Gibbs, J. W. (1902). Elementary Principles of Statistical Mhanics. Scribner's.")), who adapted ideas from physics. Dating even further back, Boltzmann, the father of modern statistical physics, used this trick to model a distribution over energy states in gas molecules. In particular, he discovered that the prevalence of a state of energy in a thermodynamic ensemble, such as the molecules in a gas, is proportional to exp⁡(−E/kT). Here, E is the energy of a state, T is the temperature, and k is the Boltzmann constant. When statisticians talk about increasing or decreasing the “temperature” of a statistical system, they refer to changing T in order to favor lower or higher energy states. Following Gibbs’ idea, energy equates to error. Energy-based models ([Ranzato _et al._, 2007](https://d2l.ai/chapter_references/zreferences.html#id228 "Ranzato, M.-A., Boureau, Y.-L., Chopra, S., & LeCun, Y. (2007). A unified energy-based framework for unsupervised learning. Artificial Intelligence and Statistics (pp. 371–379).")) use this point of view when describing problems in deep learning.


## 损失函数

softmax将输出映射到概率空间，但还需要一种方法来优化映射的准确性，在现代方法中一般采用的是**对数似然(Log-Likelihood)**

使用负对数将问题转换成最小化问题

$$
-\log P(Y \mid X) = \sum_{i=1}^{n} -\log P(y^{(i)} \mid x^{(i)}) = \sum_{i=1}^{n} l(y^{(i)}, \hat{y}^{(i)})
$$

这个式子简化后的结果称为**交叉熵损失(cross-entropy loss)**

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^{q} y_j \log \hat{y}_j
$$

其中$\hat{y}_j$是预测的概率，$y_j$是01值，只对正确的类别为1，其他都是0，对其求导可以得到：

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^{q} \exp(o_k)} - y_j = \text{softmax}(\mathbf{o})_j - y_j
$$

发现其实导数是 模型通过softmax进行分配的概率与实际发生的情况的差异，这与回归的损失函数求导结果十分相(观察值与估计值之间的差异)，这使得梯度计算变得容易


### 信息论


在信息论的解释下，其实熵是知道真实概率的人所经历的惊讶程度(surprisal)，交叉熵是观察者在看到实际的概率P生成数据时，基于主观概率Q的预期惊讶程度

> 详见：[4.1. 软最大化回归 — 深度学习入门 1.0.3 文档 --- 4.1. Softmax Regression — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_linear-classification/softmax-regression.html)


# Image Classification

## Fhasion-MNIST 

这是一个17年发布的数据集，其包含10种类别，分辨率在28 * 28，每个类别训练集有6000张，测试机有1000张

很多主流框架提供了其处理方法

```python
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

其中`tansforms.Compose([transforms.Resize(resize), transforms.ToTensor()])`方法是构建了一个图像transform流水线，之后使用时图像会经过这个转换流程，可以独立定义

```python
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

这个函数可以从类别下标转换成对应的文字类别

```python
def get_dataloader(self, trian):
	data = self.train if train else self.val
	return torch.utils.data.DataLoader(data, self.batch_size, shuffle=trian, num_workers=self.num_workers)
```

获取一个minibatch的工具函数

```python
tic = time.time()
for X, y in data.trian_dataloader():
	continue
f'{time.time() - tic:.2f} sec'
```

输出一个batch的读取时间，只要其小于图像处理时间就足够好了

```python
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
	X, y = batch
	if not labels:
		labels = self.text_labels(y)
	d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

可视化，输出一行8列的图像结果进行查看

```python
import matplotlib.pyplot as plt
 def visualize(batch, nrows=1, ncols=8, labels=[]):
 X, y = batch
 X = X.squeeze(1) #去掉单通道维度，从[N, 1, H, W] 变为 [N, H, W]
 if not labels:
	 labels = [f"label {int(i)}" for i in y]
 fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
 axes = axes.flatten()
 for i in range(len(axes)):
	 if i < len(X):
		 axes[i].imshow(X[i])
		 axes[i].set_title(labels[i], fontsize=10)
		 axes[i].axis('off')  # 关闭坐标轴
	 else:
		 axes[i].axis('off') # 如果图像数量少于子图数量，隐藏多余的子图
	 plt.tight_layout()
	 plt.show()
	 
batch = next(iter(data.val_dataloader()))
visulize(batch, nrows=2, ncols=5)
```

使用matplotlib实现方法

# Base Classification Model

基于之前的代码，先尝试构建一个基础类

## Classifier 类

增加`validation_step`方法，每次报告上一批次的损失和分类准确度，为每个num_val_batches批次绘制一个更新，方便验证分析数据的损失和准确率，尽管最后一个批次可能较少(不一定能整除），但不做特殊处理

```python
class Classifier(nn.Module):
	def validation_step(self, batch):
		Y_hat = self(*batch[:-1])
		self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
		self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

其中`*batch[:-1]` ，* 号是解包的含义，将原本作为列表的`batch[:-1]`解包成一个个元素，`[:-1]` 是取出出最后一个元素的所有元素（最后一个元素一般是标签），`self()` 其实是因为`Module` 实现了内置函数，`__call__` ，调用`self`其实就是调用`self.__call__` ，进行前向传播（forward函数），所以返回了预测的Y值


## 报告精度

```python
def accuracy(self, Y_hat, Y, averaged=True):
	Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
	preds = Y_hat.argmax(axis=1).type(Y.dtype)
	compare = (preds == Y.reshape(-1)).type(torch.float32)
	return compare.mean() if averaged else compare
```

其中，
- `Y_hat.rashape((-1, Y_hat.shape[-1]))`的含义是取出Y_hat形状的最后一维（也就是类别数）作为第二维来reshape Y_hat，`-1` 代表的是自动计算第一维的大小
- 另外，代码假设预测的概率值在Y_hat的第二维，对其取最大值并进行类型转换，因为后续的 == 对类别敏感
- 最后转换成float32数值


# Softmax Regression Implementation from Scratch


## Softmax实现

实现只是为了理解原理，实际上还有很多处理的缺陷，使用时还是用框架实现的softmax比较好

```python
def softmax(X):
	X_exp = torch.exp(X)
	partition = X_exp.sum(1, keepdim=True)
	return X_exp / partition
```

### Model

这里模型与之前无二，初始化W为高斯分布，b为0

forward之前先将图片展平，因为是线性分类，所以先展成一维，而且列维度要和权重W行维度一致

```python
def forward(self, X):
	X = X.reshape((-1, self.W.shape[0]))
	return softmax(torch.matmul(X, self.W) + self.b)
```

预测结束之后可以通过下方的代码查看预测错误

```python
wrong = preds.type(y.dtype) != y
X,y,preds - X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
	data.text_labels(y), data.text_labels(preds)
)] # 定义两个标签，一个是正确的结果，一个是错误的预测

data.visualize([X, y], labels=labels)
```


# Concise Implementation of Softmax Regression

## 定义模型

```python
class SoftmaxRegression(nn.Classifier):
	def __init__(self, num_output, lr):
		super()__init__()
		self.save_hyperparameters()
		self.net = nn.Sequential(nn,Flatten(), nn.LazyLinear(num_output)
	
	def forward(self, x):
		return self.net(x)
```

## Softmax 

这里提到两个softmax实现会遇到的问题，一个是指数溢出和下溢，因为通过`exp()`来计算softmax，如果指数很大，那很可能会超float32的精度，如果指数很小，则会得到非常大的负数，发生下溢。

实际的解决方法是在指数中减去最大的指数值

$$
\hat{y}_j = \frac{\exp o_j}{\sum_k \exp o_k} = \frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp(o_k - \bar{o}) \exp \bar{o}} = \frac{\exp(o_j - \bar{o})}{\sum_k \exp(o_k - \bar{o})}.
$$


这样一来，分子会保持在1以内，分母则在$[1-m]$之间（m为指数的数量），只有当分子为0时，在后续的取对数运算中会发生下溢。这就是另一个NaN问题。

解决方法是将softmax与交叉熵结合，将中间的过程结果（而不是计算之后的输出）传递给交叉熵，让取对数操作对`exp`进行，化简之后得到


$$
 \log \hat{y}_j = \log \left( \frac{\exp(o_j - \bar{o})}{\sum_k \exp(o_k - \bar{o})} \right) = o_j - \bar{o} - \log \sum_k \exp(o_k - \bar{o}).
 $$

这种方法称为“LogSumExp trick"，传递的中间结果称为logits，在交叉熵损失中一次计算softmax与对数结果。




# Generalization in Classification

## The Test Set

对测试集的评估其实属于对基础种群的一种统计估计，因为假设测试集是随机均匀采样，因此，泛化问题其实是一个均值估计问题

**中心极限定理**， 从均值$\mu$ 标准差$\sigma$ 分布中抽取n随机样本，当n的数量趋于无穷大时，样本均值趋于以真实均值为中心，标准差为$\sigma/\sqrt{n}$ 的正态分布。

从这个角度来看，随着实例数量的增加，测试集的误差以$O(1/\sqrt{n})$的速度接近真实误差，也就是如果要精度增到 $m$ 倍，就得有 $m^2$ 倍的测试集

由于随机变量$1(f(X) !=Y)$ 只能取值0和1，因此是伯努利随机变量，特征是一个参数，表示取值1的概率，这里随机变量参数实际上是真是错误率。

伯努利分布的方差取决于其参数

$$Var(Z)=p(1-p)$$

对于这种情况下，也就是参数为真实错误率的情况下，参数最大为1，最小为0，所以方差最大时错误率为0.5，标准差为$\sqrt{Var(Z)/n}$ 。

这说明，当想让测试误差近似于总体误差，使得标准差在正负0.01之间有效，那么应该收集2500个样本，如果要有95%的把握，则需要10000个样本

上述其实只给出了一个渐进关系，由于随机变量是有界的，可以通过Hoeffding不等式来获得有效的有限样本界限

对于独立同分布的随机变量 $Z_1, Z_2, \dots, Z_n$，若每个 $Z_i \in [a,b]$（有界），则Hoeffding不等式给出样本均值与期望值的偏差概率上界：

$$
P\left( \left| \frac{1}{n} \sum_{i=1}^n Z_i - \mathbb{E}[Z] \right| \geq t \right) \leq 2 \exp\left( -\frac{2n t^2}{(b-a)^2} \right)
$$

在分类错误率问题中：
- $Z_i \in \{0,1\}$（伯努利变量，故 $a=0, b=1$）
- $\mathbb{E}[Z] = e(f)$（真实错误率）
- 样本均值 $\hat{e}(f) = \frac{1}{n} \sum_{i=1}^n Z_i$
- $t$ 就是误差容忍值，表示已P()的概率认为误差小于$t$

代入参数后，Hoeffding不等式简化为：

$$
P\left( |\hat{e}(f) - e(f)| \geq t \right) \leq 2 \exp(-2n t^2) \tag{4.6.3}
$$

根据这个式子可以计算出满足大多的置信度时需要多少样本

## Test Set Reuse


在收集测试集时，一般考虑的是当前分类器的精度需求，当想采用多个分类器时，每个分类器有一定概率出错（例如95%），当多个分类器一起训练时，就无法确定这次训练的有效性，因为涉及到多个假设等

同时存在自适应过拟合问题，因为针对某个数据集的精度提高，已经修改了多次分类器，这可能引入测试集中的信息，导致测试集不能完全视为真正的测试集

## VC维

为了衡量模型类的复杂性，解决泛化能力问题，解释为什么、何时基于经验数据训练的模型可以泛化到没见过的数据，用于分析经验误差与总体误差之间的差距。

VC维其实是衡量一个模型类别的表达能力，具体来说，其表示模型能“完美记忆”的最大样本量，例如直线分类器的VC维是3，3点内的数据总能被一条直线分开，而4个点不行

引入VC维，可以将泛化误差的概率上界定义为：

$$P(R_{pop}(f)- R_{emp}(f)  \leq \alpha) \geq 1 - \delta$$
其中：
- $R_{\text{pop}}(f)$：模型在真实分布上的误差（总体风险）
- $R_{\text{emp}}(f)$：模型在训练集上的误差（经验风险）
- $\alpha$：泛化误差的**最大允许差值**（如要求泛化误差不超过经验误差+2%）
- $\delta$：违反该条件的概率（如设定为5%）
- **VC维度（$d_{\text{VC}}$）和样本量（$n$）的影响**：

  $$
  \alpha \geq c \sqrt{\frac{d_{\text{VC}} - \log \delta}{n}}
  $$
  - $c$ 是与损失函数范围相关的常数

通过VC维可以指导统计概率模型样本量，但是对于深度来说并不适用

