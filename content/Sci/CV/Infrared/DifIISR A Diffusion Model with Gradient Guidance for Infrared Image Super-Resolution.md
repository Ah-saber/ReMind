---
tags:
  - CV
  - CS
  - CVPR
  - Super-Resolution
  - DM
---
# Key Issue

当前方法缺失对红外成像这一模态特性的考虑，例如高频，热谱分布等，同时缺少机器感知需求，强调视觉感知，缺失了纹理细节等


# Motivation

扩散模型在超分方向极具优势，尤其在梯度引导这一方法出来后，说明可以通过梯度修改来增强、控制某些特性，可以以此来引入一些红外成像上的特性或对某些特征的考虑

# Method

采用DDPM扩散方法，DDIM的非马尔科夫过程

在梯度中注入先验的过程可以转变成

$$
\nabla_{x_t} \log p(x_t \mid g) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(g \mid x_t)
$$

将问题变成求解 $\log p(g \mid x_t)$ 

进一步近似可以得到（对数最大可以看成欧几里得最小）

$$
\begin{aligned}
\nabla_{x_t} \log p(g \mid x_t) &\approx \nabla_{x_t} \log p(g \mid \hat{x}_0(x_t)) \\
&\approx - \rho \nabla_{x_t} \|g - \mathcal{M}(\hat{x}_0(x_t))\|_2^2,
\end{aligned}
$$

即梯度变成

$$
\begin{aligned}
\epsilon'_\phi &= \epsilon_\phi(x_t, t) + \rho\sqrt{1 - \alpha_t} \nabla_{x_t} \|g - \mathcal{M}(\hat{x}_0(x_t))\|_2^2 \\
&= \epsilon_\phi(x_t, t) + \rho\sqrt{1 - \alpha_t} \nabla \mathcal{L}_g,
\end{aligned}
$$

由此引入先验

![[Pasted image 20251014123452.png]]



### Visual Optimization

引入频率分布的学习

$$
\mathcal{L}_{\text{visual}} = \left( \overbrace{N\left(\log(1 + |\hat{\mathbf{I}}_{HR}^{\text{shift}}|)\right)}^{M_{HR}^{\text{norm}}} - \overbrace{N\left(\log(1 + |\hat{\mathbf{I}}_{SR}^{\text{shift}}|)\right)}^{M_{SR}^{\text{norm}}} \right)^2
$$

其中通过正则化缩小强度，强调频率分布

### Perceptual Optimization

使用VGG来的得到更多纹理，边界学习，使用SAM来得到更多的语义信息学习

$$
\mathcal{L}_{\text{perceptual}} = \overbrace{\|\phi_l(\mathbf{I}_{HR}) - \phi_l(\mathbf{I}_{SR})\|_2^2}^{\mathcal{L}_{\text{VGG}}} + \overbrace{\|\mathbf{S}_{HR} - \mathbf{S}_{SR}\|_2^2}^{\mathcal{L}_{\text{seg}}}
$$

## Result

在无参考指标中提升显著，在有参考中有所增强，在单图对比中有所提高

在下游任务表现最佳，说明了纹理细节，语义等的有效学习

# Conclusion

生成模型在红外超分的使用，方法简单，提升不小