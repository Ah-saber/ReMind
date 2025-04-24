---
tags:
  - "#CV"
  - "#Generate"
  - "#DM"
date: 2025.4.14
---
# Before Learning

除了DM之外还有许多的生成模型，GAN、VAE和Flow-based模型，它们在生成高质量样本方面取得巨大成功，每种模型都有其自身的局限性。
- GAN模型对抗性训练的性质和潜在的训练不稳定性，生成多样性不足
- VAE依赖于代理损失(surrogate loss)
- 流模型必须使用专门的架构来构建可逆变换。

而扩散模型基于非平衡热力学知识(non-equilibrium thermodynamics)，定义一个马尔科夫链(Markov chain)，每一步慢慢的增加随机噪声，然后通过反向扩散过程来构造期望的采样分布

相比 VAE 和 Flow models，扩散模型学习的步骤确定，潜在变量维度更高（与原始数据相同）。

![[Pasted image 20250418205208.png]]
# What are DMs

现有的扩散模型基于数个前人的伟大工作，扩散概率模型(diffusion probabilistic model) [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585) ，噪声-条件评分网络(noise-conditioned score network) [NCSN; Yang & Ermom, 2019](https://arxiv.org/abs/1907.05600) ，去噪扩散概率模型(denoising diffusion probabilistic model[DDPM; Ho et al. 2020](https://arxiv.org/abs/2006.11239)

## Forward Diffusion Process

## 公式推导

给定一个从真实分布中采样的出来的样本点，每次对其增加一个微小的高斯噪声，持续 T 步，最后生成一个 $x_1 …… x_T$  的序列，

要从高斯分布里采样出来一个$z$，其公式如下：
$$
\frac{z - \mu}{\sigma} \sim \mathcal{N}(0, I)
$$

做一点变换（重采样）：

$$
z = \mu + \sigma \cdot \varepsilon \quad \varepsilon \sim \mathcal{N}(0, I)
$$

> [!why Doing Such a Change]
> 因为直接从高斯分布里采样$z$的话不可微分，无法求导优化，所以希望能换一种可微分的方式，在VAE中提出，用一个固定噪声$\varepsilon$来代替原分布，化简得到重采样式子

而对于每次加噪，公式是：

$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \varepsilon_{t-1}$$

其中，$\sqrt{1-\beta_t} x_{t-1}$ 是均值，$\sqrt{\beta_t}$ 是方差，所以在这个式子下，$x_t$ 会逐渐失去其可辨识性，会越来越接近高斯分布（参照上面的式子）

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, (\beta_t) I)$$

对其进行化简，令$\alpha_t = 1 - \beta_t \quad \text{和} \quad \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$ 可得

$$
\begin{align}  \mathbf{x}_t &= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} \quad &;\text{where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \cdots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\ 
&= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\bar{\boldsymbol{\epsilon}}_{t-2} \quad &;\text{where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \cdots \\ &= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\ q(\mathbf{x}_t|\mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \end{align}
$$

>在高斯分布中，对于 $\mathcal{N}(0, \sigma^2_1 \mathbf{I})$ + $\mathcal{N}(0, \sigma^2_2 \mathbf{I})$ ，可以得到 $\mathcal{N}(0, (\sigma^2_1+\sigma^2_2) \mathbf{I})$


每次增加的噪声不一定一致，直观来看，对于原始图片，微量的噪声可以造成很大的影响，对于中间过程，微量噪声基本看不出来，如果这样来看，当样本变得嘈杂时，可以承受更大的步长。

## 随机梯度朗之万动力学

朗之万动力学(Langevin dynamics) 是一个物理概念，用于对分子系统做统计建模，结合随机梯度下降(stochastic gradient descent)构成 随机梯度下降朗之万动力学[Welling & Teh 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) 

提供了一种采样方法，使用Markov chain 更新中的梯度$\nabla_{\mathbf{x}} \log p(\mathbf{x})$ 来生成概率密度 $p(x)$ 的样本

$$\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t, \quad \text{where } \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

对比标准的 SGD 方法，随机梯度朗之万方法将高斯噪声结合进参数更新中，避免陷入局部最优解

>[!关于朗之万动力学]
>是一种考虑到系统热噪声下用来模拟分子运动的动力学公式，在机器学习中用来模拟从概率分布中抽取样本的过程
>![[朗之万优化示意图.gif |400]]
>直观来看，其实这是在确定性上增加不确定性，沿梯度方向优化的同时增加噪声，让结果在高峰随机游走，避免局部最优
>[【概率方法】朗之万动力学 Langevin Dynamics-CSDN博客](https://blog.csdn.net/qq_18846849/article/details/134909343)


