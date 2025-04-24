---
date: 2025.3.8
tags:
  - Deep
  - ML
  - torch
  - Python
---
# Linear Regression

## 基础模型

线性回归就是要训练一个模型
$$y = \omega_1x_1 + ... \omega_ix_i + b$$
具体来说，线性回归是做一种仿射变换（affine transformation），通过加权求和做特征线性变换，结合偏置进行平移

对于多个数据，模型总结成矩阵形式
$$y_{hat} = Xw+b$$
其中X是n * d 的矩阵，含义是多个多维数据

## 损失函数

一般来说是欧氏距离，即：

$$
L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} l^{(i)}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{2} \left( \mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)} \right)^2 
$$

这个公式有一个解析解，因为每个x都是线性独立的，所以矩阵X满秩，$\omega$ 有唯一解。

$$ 
\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} 
$$

### Minibatch

虽然某些问题会有唯一解，但是其限制十分严格，基本上来说大多数模型都没有唯一解，所以应该有除了解析解之外的其他求解。

随机梯度下降法（stochasic gradient descent **SVG**） 是其中一个方法，缺点是在计算和统计上，一个问题是处理器在处理乘法，加法运算要更快；矩阵-向量的乘法比向量-向量的操作也快一个数量级，这意味着处理一个批次要快于处理一个样本。另外，在批量归一化时需要同时查看多个观察值才能更好的工作。

所以选择采用 **小批次随机梯度下降法**（Minibatch Stochastic Gradient Descent) ，每次取一个minibatch（2的幂次）进行梯度下降更新，具体为：

$$
(w, b) \leftarrow (w, b) - \frac{\eta}{|B|} \sum_{i \in B_s} \partial_{(w, b)} l^{(i)} (w, b)
$$
基本上，找到迭代次数结束，我们都找不到一个最优解，因为随机性的存在，即使确实有封闭解，算法一般无法在有限步骤找到这个解。

### 为什么用这个损失

1. 对错误的惩罚足够大
2. 其实损失函数实际上是在优化$E[(Y-c)^2]$ ，对于损失函数中的定义，在噪声满足高斯分布的情况下，损失函数的最优解是$E[Y]$ ，在输入x的条件下，最优解是$E[Y|X]$，也就是实际上是优化参数，使得模型预测值逼近条件期望。
	>如何理解？
	>实际上如果噪声是高斯分布，那么原来的损失函数右半部分可以看成是高斯分布的指数部分，最大化高斯分布就是找到期望值，对于损失函数（高斯分布的对数+取反）的最小值也就是期望值，也就是优化参数，对输入x取到y的期望值
	>
	>$$
	p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
	$$

# Object-Oriented Design

## Models

常用模块定义

```Python
class Module(nn.Module, d2l.HyperParameters):
	def __init__(self, plot_train_epoch=2, plot_vaild_eopch=1):
		super()__init__()
		self.save_hyperparameters()
		self.board = ProgressBord()
	
	def loss(self, y_hat, X):
		raise NotImplenetedError
	
	def forward(self, X):
		assert hasatter(self, 'net'), "Neural Network is defined"
		return self.net(X)
		
	def trainging_step(self, batch):
		l = self.loss(self(*batch[:-1]), batch[-1])
		self.plot('loss', l, train=True)
		return l
	
	def validation_step(self, batch):
		l = self.loss(self(*batch[:-1]), batch[-1])
		self.plot('loss', l, train=False)
	
	def configure_optimizers(self):
		raise NotImplementedError
	
```

# Synthetic Regression Data

## Pytorch内部迭代器

定义Dataset其实是为了使用Pytorch的内部迭代器，加快数据的存取，访问速度

```python
def get_tensorloader(self, tensors, train, indices=slice(0, None)):# slice(0, None)是选择全部的意思
	tensors = tuple(a[indices] for a in tensors)
	detaset = torch.utils.data.TensorDataset(*tensors)
	return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

def get_dataloader(self, train):
	i = slice(0, self.num_train) if train else silce(self.num_train, None)
	return self.get_tensorloader((self.X, self.y), train, i)
```


定义内部迭代器数据加载器

```python
X, y = next(iter(data.train_dataloader()))
print(....)
```

`next`迭代器的方法，获取数据的下一批次，有特征张量X和标签张量y构成。

`iter` 将数据加载器转化为一个迭代器


# Linear Regression Implementation from Scratch

## Defining the Model

```python
class linearRregession(nn.Module):
	def __init__(self, num_inputs, lr, sigma=0.01……)：
		super().__init__()
		self.save_hyperparameters()
		self.w = torch.normal(0, sigma, (num_inputs, 1), requries_grad=True)
		self.b = torch.zeros(1, requires_grad=True)
		# 初始化参数

	def forward(self, x):
		return torch.matmul(X, self.w) + self.b
		# 定义前推公式

	def loss(self, y_hat, y):
		l = (y_hat - y) ** 2 / 2
		return l.mean()
		# 定义损失
	
	def configure_optimizers(self):
		return SGD([self.w, self.b], self.lr)
		# 定义优化器
```


### SGD实现

```python
class SGD(d2l.HyperParameters):
	def __init__(self, params, lr):
		self.sace_hyperparameters()
	
	# 定义更新过程
	def step(self):
		for param in self.params:
			params -= self.lr * param.grad

	# 定义梯度清零
	def zero_grad(self):
		for param in self.params:
			if para.grad is not None:
				param.grad_zero_()
```

## Defining Trainer

```python
def prepare_batch(self, batch):
	return batch

def fit_epoch(self):
	self.model.train()
	for batch in self.train_dataloader:
		loss = self.model.traing_step(self, prepare_batch(batch))
		self.optim.zero_grad()
		with torch.no_grad(): # 暂时不知道作用
			if self.gradient_clip_val > 0:
				self.clip_gradients(self.gradient_clip_val, self.model)
			self.optim_step()
		self.train_batch_idx += 1 # 手动更新
	if self.val_dataloader is None:
		return
	self.model.eval()
	for batch in self.val_dataloader:
		with torch.no_grad():
			self.model,validation_step(self.prepare_batch(batch))
		self.val_batch_idx += 1
```

## torch 实现

### 模型定义


```python
class LinearRegression(d2l.Module):
  def __init__(self, lr):
    super().__init__()
    self.save_hyperparameters()
    self.net = nn.LazyLinear(1)
    self.net.weight.data.normal_(0, 0.01)
    self.net.bias.data.fill_(0)

def forward(self, X):
  return self.net(X)
```

### 损失函数

```python
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
  fn = nn.MSELoss();
  return fn(y_hat, y)
```

### 优化器

```python
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
  return torch.optim.SGD(self.parameters(), self.lr)
```

### 训练

```python
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

### 查看参数

```python
def get_w_b(self):
	rreturn self.net.weight.data, self.net.bias.data
```


# Weight Decay

## 权重衰减与正则化

权重衰减是最常见的正则化技术，通过限制参数的取值来缓解过拟合，其直觉大概是**在所有可能的函数中，对所有输入赋予0参数权重的函数某种意义上是最简单的**，也就是通过函数参数与零的距离来衡量函数的复杂性

那么问题在于如何衡量参数与零的距离？

其中常见的方法是利用某些范数来衡量，要减少这种衡量下参数与0的距离，可以将这个范数加到损失中进行最小化优化，结合优化的好处是：**当权重向量过大时，优化会更侧重于优化权重，而不是一味的减少标签损失**

然而，添加这个损失依旧需要能对其与原本的标签损失的重要性进行权衡，这需要引入一个超参数 $\lambda$ 

即完整损失为：

$$
L(w, b) + \frac{\lambda}{p}||w||^p
$$

另外要注意的是，对b的正则化在不同实现中可能不同，通常不做正则化

## 一个示例

定义权重衰减正则化项为 $||w||^2$，这里使用平方范数是因为计算更方便，这样可以去掉平方根，使得导数更容易计算

另外，$l_1$正则化和$l_2$正则化都十分常见，前者称为lasso回归，后者称为岭回归，前者注重将权重集中在少数几个特征上，后者则是更加均匀的分配权重

最终应用$l_2$正则化下，梯度更新（对w求导）为：

$$
w \leftarrow (1-\eta \lambda)\mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i=1}^{\mathcal{B}}\mathbf{x}_i(\mathbf{w}^T\mathbf{x}_i+b-\mathbf{y}_i)
$$

$l_2$损失定义为

```python
def l2_panalty(w):
	return (w ** 2).sum() / 2
```


在Pytorch中可以定义为：

```python
def configure_optimizers(self):
	return torch.optim.SGD([
		{'params':self.net.weight, 'weight_decay': self.wd},
		{'params':self.net.bias}], lr = self.lr)
```

其中，默认是$l_2$范数，`wd`是正则化参数 $\lambda$


