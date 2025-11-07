---
tags:
  - CS
  - CV
  - CVPR
---
# Key Issue

高质量的红外数据集十分缺乏，且红外图像面临着较为严重的动态退化问题

# Motivation

生成方法，在这里是对抗学习是一种解决数据稀缺的思路，通过少量数据，自己生成噪声（退化）再对抗着做增强，可以达到大量数据的训练效果

其次，这种方法还可以加不同的噪声来学习到对不同的退化的增强能力

# Method

对抗框架

$$
\begin{aligned}
&\min_{\omega} \mathcal{L}(\mathcal{N}_{\text{E}}(\hat{\mathbf{x}}; \omega), \mathbf{y}), \\
&\text{s.t.} \begin{cases}
\hat{\mathbf{x}} = \mathcal{N}_{\text{G}}(\mathbf{x}; \theta^\star), \\
\theta^\star = \mathop{\arg\max}_{\theta} \mathcal{L}(\mathcal{N}_{\text{E}}(\mathcal{N}_{\text{G}}(\mathbf{x}; \theta); \omega), \mathbf{y}),
\end{cases}
\end{aligned}
$$

最小化增强结果与真实结果的差距，生成器相反

![[Pasted image 20251016120929.png]]

其中，生成器是通过增加退化来实现的，采用参数化或非网络的方法增加某些退化，文中强调了三种，条纹噪声（红外成像传感器非均匀，响应不一致），低分辨率，低对比率

但是这样的话无法训练，所以增加一个参数a来做学习参数，作为退化三者的权重，参与微分

$$
\hat{\mathbf{x}}^{i+1} = \sum_{j=1}^{N} a_{ij} \mathbf{D}_j(\hat{\mathbf{x}}^i), \hat{\mathbf{x}}^0 = \mathbf{x},
$$

### Dual Interaction Network

一个Scale transform module，做不同尺度的特征学习

一个Spiking-guided separation module，将退化特征分离，参考SNN，将空间特征转化成脉冲序列，解耦退化与原图，便于增强部分的优化学习


# Conclusion

方法在各种退化中都有部分进步，在多种退化的测试中取得较大改进，同时适用于下游任务

这是一个基于对抗学习的增强方案，优势是这种方案可以仅使用少量数据，同时处理多种退化，其引入的SSM也将部分先验引入作为增强

但是似乎不是优解，每个块都单独存在，结合多种退化的方法很巧但是不够好，相比扩散，反而显得不够灵活
