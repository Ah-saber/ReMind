---
date: 2025.2.25
tags:
  - Deep
  - ML
  - torch
  - Python
---
# 基础张量操作

### Concatenate

```python
x = torch.arange(12, dtype=torch.float32).reshape(3,4)
x = torch.tensor([1, 3, 4, 5], [1, 3, 4, 5], [4, 3, 2, 1])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
# 按第一维拼接，就是行数增加，按第二维拼接，列数增加
```

### Broadcasting

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a + b
```

会逐个元素彼此运算，拼成结合二者维度的新张量

```result
tensor([[0, 1],
        [1, 2],
        [2, 3]])
```

要注意的是只有相同维度，缺省维度，单个维度，可以使用broadcasting，其他均不可

### Memory

Python的内存分配机制会让`Y=X+Y`的Y属于两个不同的内存地址

```python
Z = torch.zero_like(Y) #init Z
before = id(Z)
Z[:] = X + Y #Y[:] = X + Y
id(Z) == before
```

通过 `Z[:]` 解决问题，指明使用同一变量

```result
Ture
```

### Conversion

```python
# 转化numpy
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B) # numpy  tensor

a = torch.tensor([3])
a.item() # 将维度为1的tensor转换成python的标量
```

要注意的是torch和numpy共享一个地址，如果两个变量指示一个数据这两种不同格式，修改其中一个的数据，另一个也会更改

# pandas操作

> Comm-separated values 即CSV文件，含义是逗号分割的文件

### 读取

```python
data = pd.read_csv(data_file)
print(data)
```

## 选择

```python
name_column = data['Name', 'City']
name_column = data.loc[:, 'Name']
```

##  bed bugs 处理

> bed bugs是指NaN的数值

### 文本分类值

```python
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

其中`get_dummies`是将分类数据拆分为独热编码的函数，pandas会默认将文字和空白补充为NaN，使用函数后可以将这些单独列出一列（包括空白的NaN）

```result
   NumRooms  RoofType_Slate  RoofType_nan
0       NaN           False          True
1       2.0           False          True
2       4.0            True         False
3       NaN           False          True
```

### 数字空缺值


```python
inputs = inputs.fillna(inputs.mean())
```

## tensor转换

原始值是数值，先转成numpy形式再转成对应的tensor

```python
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
```


# Linear Algebra

> 使用order（阶）来代表维度，使用dimension（维）来代表维度的长度

## Vector

```python
x = torch.arange(3)
x = torch.rand(10)
len(x)
x.shape
```

## Matrices

```python
A.T # 转置
```

Hadamard 积，也就是 `A * B `，逐元素相乘

## Tensor

```python
a = 2
X = torch.arange(24).reshape(3, 4, 2)
a + X, (a * X).shape # 含义是每个元素的运算
X.sum() # 结果是直接全部元素相加
```

## 降维

```python
A.sum(axis=0).shape
```

sum其实是对tensor降维，可以选定第几维进行降维，降维之后变成一个和的标量，但不会影响tensor本身

```python
A.mean() == A.sum() / A.numel() # True
A.mean(axis=0)
```

### 非降维Sum

有时候需要保持维度来支持后续运算，例如每行标准化

```python
sumA = A.sum(axis=1, keepdims=True)
reduceA = A.sum(axis=1)
sum_A.shape   reduceA.shape
```

```result
(torch.size([2, 1])  torch.size([2]))
```

```python
A.cumsum(axis=0), A
```

这是按行（列）加，每行每列的元素等于其前面元素的累加和

```result
tensor([[0., 1., 2.],
        [3., 5., 7.]])
tensor([[0., 1., 2.],
        [3., 4., 5.]])
```

### Dot Products

```python
y = torch.ones(3, dtype = torch.float32)
torch.dot(x, y)
```

逐元素相乘再相加 与 `torch.sum(x * y)` 等价

矩阵与向量的乘法。 m * n 的矩阵与 n * 1的向量点乘可以看作是n到m的投影变换，也描述了前一层输出确定时，计算网络每层输出的关键（？）

```python
toch.mv(A, x), A@x
```

mv是matrix与vector，@支持矩阵与矩阵，和矩阵与向量的运算

```python
torch.mm(A, B), A@B
```

# Norm

## 性质

- 满足 $||ax|| = |a|\,||x||$ 
- 满足三角不等式 $||x+y|| \leq ||x||+||y||$
- 非0，除非向量为0

- 欧式距离 $l_2$ （平方和开根号）

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

- lasso $l_1$ （绝对值之和）

```python
torch.abs(u).sum()
```

- 更高的范式 $l_p$  
	$$||x||_p = (\sum_{i=1}^n|x_i|^p)^{1/p}$$
	上述范数都满足这个范数

- Frobenius Norm （矩阵各元素平方和的平方根）
```python
torch.norm(torch.ones((4, 9)))
```

- 谱范数，描述了矩阵运算Xv 与 原本的向量v相距多长


# Automatic Differentitaion

## 基础计算

```python
x = torcb.arange(4.0)
x # (0., 1., 2., 3.)
```

现代深度学习框架会自动计算梯度，对每个连续函数传递数据时，框架会构建一个计算图，跟踪每个值并依赖于其他值，为了计算导数，自动微分程序通过链式法则在图中反向工作称为**反向传播(backpropagation)**

使用梯度计算，可以

```python
x = torch.arange(4.0, require_grad=True)
x.require_grad(True)
x.grad  # 查看梯度
```

实现y是x的函数

```python
y = 2 * torch.dot(x, x)
y.backward()
x.grad
```

```result
tensor([ 0.,  4.,  8., 12.])
```

一般的增加关于x的函数时，程序会自动将其梯度结果加在上次计算的梯度上（即存储在一个地方，两数相加）

```python
x.grad_zero_() # 梯度重置
y = x.sum()
y.backward()
x.grad
```

```result
tensor([1., 1., 1., 1.])
```

注意向量函数的对向量的导数是向量

## 计算相关

### 向量计算

其实向量对向量求导的结果应该是雅可比矩阵（Jacobian），但是深度学习一般更在乎y的每个分量对完整向量x的梯度之和，得到一个于x形状相同的向量

深度学习在解释非标量张量的梯度是存在差异，Pytorch在非标量上调用backward会出错，除非设置好求梯度对象如何缩减（降维）成标量。具体来说，需要有一个 **v** 来实现对 $v^T\partial_xy$而不是 $\partial_xy$

```python
y = x * x
y.backward(gradient=torch.ones(len(x)))
x.grad
```

结合例子来看，实际上gradient是起到一个权重作用，对于向量y，其提供一个加权和的降维方法，将y改成标量，再对x逐一求导，得到向量结果

### Detaching 

当计算时，可能会有这样一种情况：
- $z = x * y \quad y = x * x$ 
- 但是只想要关注 z 对 x 的梯度结果，而 y 不过是一个辅助的过程函数

那么需要将y排除在反向传播之外，这需要建立一个新变量u ，保证其来源被抹去，只有数值，使用u来代替y，就可以保证其梯度不会流向x

```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```result
tensor([True, True, True, True])
```

### 其他

另外，自动求导对分段函数也适用，对自定的python计算程序，利用Python控制流（条件，循环，调用等）就可以计算结果的梯度