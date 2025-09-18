---
tags:
  - CV
  - CS
---
# Key Issue

当前的卷积方法最后一层都是全连接层，并不适用于密集预测任务（例如语义分割），因为只有一维，丢失了空间信息

在采样过程中，以往使用的shift-and-stitch方法计算量巨大

# Motivation

- **原理合适**：实际上全连接层可以看作一种感受野为全图的卷积
- **输入灵活**：之前由于使用全连接层，输入需要满足全连接层的维度设置，使用裁剪方法输出，如果改成全卷积，输入可以是整图
- **计算效率高**：以往的方法中，需要分不同patch来计算，划分后反而引入一些重叠部分，这些重叠会被反复计算、训练，而整图输入的话则可以解决这些冗余，在训练、梯度计算，反向更新等方面都保证计算高效
- **密集任务适用**：换成全卷积，网络实现端到端，对密集预测任务十分适用，且具有空间信息。

![[Pasted image 20250905223819.png]]

# Method

## Shift-and-stitch is filter rarefaction

在现有方法中，平移-拼接用来保证下采样方法可以看到所有的像素，保证特征的学习，但是计算量十分巨大，若滤波器大小为$f$，那么会计算$f^2$次

论文提出用步长为1的卷积来代替下采样滤波器，保证看到所有的像素，同时实现高分辨率，但是这样会引入一些之前采样时看不到的像素和计算，于是提出**稀疏滤波器**

$$
f'_{ij} =
\begin{cases}
    f_{i/s, j/s} & \text{if } s \text{ divides both } i \text{ and } j; \\
    0 & \text{otherwise},
\end{cases}
$$

用0填充来扩大滤波器，保持稀疏，即增大感受野，看到所有像素，又保证不引入多余计算

## Patchwise training is loss sampling

使用整图的输入好处很多，但是实际上失去了patch方法的随机性，难以表示不同的分布

如果随机选择预测图上的一些像素来计算损失，或者使用dropout方法，则可以保证这种随机性，说明全卷积方法确实可以完全代替之前的方法

## skip-connect

直接上采样的结果很差，模糊不清，因此提出：**浅层网络有更高的分辨率，可以用浅层的输出来丰富采样结果**

![[Pasted image 20250905230256.png]]


例如：
- pool4的输出可以和pool5的输出相结合（pool4要先过1卷积改变维度），再进行16x上采样，搭建FCN016s
- 把上述的输出再和pool3结合，再8x上采样，构建FCN-8s

论文中提到提升有限，就只做到这里

![[Pasted image 20250905230719.png]]

## Result

![[Pasted image 20250905230930.png]]

![[Pasted image 20250905230941.png]]

最终在时间和精度上都大满贯

# Conclusion

FCN提出全卷积方法，首次将图像分类的基础模型引向密集预测任务，提出完善方法并论证了全卷积整图训练可以减少计算量，不丢失以往方法的重要特性，同时提出空洞卷积（稀疏滤波）和跳跃连接等，首次构建端对端模型，达到SOTA

# Refer

【代码实现】[pochih/FCN-pytorch: 🚘 Easiest Fully Convolutional Networks](https://github.com/pochih/FCN-pytorch)

【笔记参考】[Fully Convolutional Networks 論文閱讀 | by 李謦伊 | 謦伊的閱讀筆記 | Medium](https://medium.com/ching-i/fully-convolutional-networks-%E8%AB%96%E6%96%87%E9%96%B1%E8%AE%80-246aa68ce4ad)

