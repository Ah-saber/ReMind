---
tags:
  - Generate
  - Survey
  - CV
  - CS
---
# Attention-based

这种方法通过注入特征图，针对模型Unet/DiT做设计，将所需的特征注入在某些关键层，以此保证生成的一致性，或者实现任务修改

- **Stable Flow: Vital Layers for Training-Free Image Editing**  [[2411.14430] Stable Flow: Vital Layers for Training-Free Image Editing](https://arxiv.org/abs/2411.14430)
	- 分析了DiT的层级重要性，实际上应用了MasaCtrl中的注意力机制，但是用在DiT中
	- 同时提出给逆向的起始噪声乘某个系数来拜托正向过程非高斯的OOD情况
- **MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent  Image Synthesis and Editing** [[2304.08465] MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://arxiv.org/abs/2304.08465)
	- 注意力注入，为保持一致性，将原图像的K，V注入到编辑过程的注意力中，调节适当的时间步注入得到保持一致且能完成编辑的效果
- **PROMPT-TO-PROMPT IMAGE EDITING WITH CROSS-ATTENTION CONTROL** [[2208.01626] Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)
	- 注意力注入的鼻祖，提出图像像素与文本token之间的关系，以及修改注意力实现图像编辑的路线

# Plug and Play Method

## Forward-Backward Splitting

FBS，前向后向分裂优化，这是分裂优化逻辑，其核心问题是期望最大化后验概率估计（MAP问题，也就是已有一张坏图，求原本这张图的好图最可能是长什么样的，以此来实现图像修复），通过贝叶斯公式得到两项

$$
\begin{aligned}
    \hat{x} &= \underset{x}{\arg\max} \log p(x|y) \quad \text{(1)} \\
    &= \underset{x}{\arg\max} \left[ \underbrace{\log p(y|x)}_{\text{数据项: 这叫图像观测数据}} + \underbrace{\log p(x)}_{\text{先验项: 这叫图像先验}} \right]
\end{aligned}
$$

其中数据项用于保证与原图的相似性，可以提供优化损失，即任务目标函数的反向过程，（让反向后的图像与原图接近），这一步将噪声认为是错误的；先验项用于保证图像的保真性（例如满足扩散生成的中间图，满足语义特征等），在这一步才去噪

这种方法只需要前向传播，不需要反向


- **PNP-FLOW: PLUG-AND-PLAY IMAGE RESTORATION  WITH FLOW MATCHING** [[2410.02423] PnP-Flow: Plug-and-Play Image Restoration with Flow Matching](https://arxiv.org/abs/2410.02423)
	- 提出使用Flow Matching的PnP方法，论证OT构造的直线耦合是最合适的路径，定义数据项满足任务，进行三步优化：
		1. 梯度步，和FBS方法一致，先用数据项做梯度的优化，不需要反向，因为没有计算
		2. 插值步，梯度引导之后的噪声图不一定还在Flow的路径上，所以进行一步插值引导
		3. 去噪步，用FLow做去噪器，也就是先验项


## Diffusion Posterior Sampling

旨在用梯度指导来最大化后验概率分布，也就是利用条件y来做生成采样指导，同时最大化这个值来保证生成的结果满足目标

后验概率可以分解为两项：

$$
\underbrace{\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y})}_{\text{我们要的采样方向}} = \underbrace{\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)}_{\text{无条件得分 (U-Net)}} + \underbrace{\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\mathbf{x}_t)}_{\text{引导项 (Guidance)}}
$$

和FBS相比，FBS是逐项计算，类似于ADMM，先算一项，再算另一项，二者相加，而DPS是用梯度指导，将loss（目标损失集成到引导项中，计算其梯度和扩散模型梯度，二者相加来指导x的生成）

通过设计Loss，和对噪声图的修改范式，依据Loss来对采样过程的噪声图进行修改，实现免训练的生成图像修改，需要后向传播计算梯度，一般通过一定假设得到噪声图对应的清晰图，再用清晰图与Loss计算损失，反向传播得到梯度，进行梯度引导

- **Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation** [Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation](https://proceedings.mlr.press/v202/song23k.html)
	- 假设噪声图满足高斯分布，认为DPS的设计中没有考虑到这个高斯分布应有的方差，所以采用蒙特卡洛多次采样的方式来期望得到更好的噪声图预测，并依据设计的loss对噪声图进行修改
	- 对原理做优化
- **Self-Supervised Selective-Guided Diffusion Model for Old-Photo Face Restoration** [[2510.12114] Self-Supervised Selective-Guided Diffusion Model for Old-Photo Face Restoration](https://arxiv.org/abs/2510.12114)
	- 使用扩散的弱引导生成的半成品图片和其他模型，提取出关键的修复细节
	- 通过不同时间步注入不同的特征引导，来生成较好的修复效果，并逐一解决各种问题
	- 对应用做优化
	- 需要一个较强的扩散基础模型，需要高质量的数据


# Model-based

这里归类非体系或者难以用本文使用的分类归类的方法

## Zero-shot Model

构建模型，或者训练方法，实现一个Zero-shot的框架，满足图像编辑的需求

- **Zero-shot Image Editing with Reference Imitation** [[2406.07547] Zero-shot Image Editing with Reference Imitation](https://arxiv.org/abs/2406.07547)
	- 跨模型进行注意力学习，用视频帧相互预测，补全的形式做掩码预测，使模型学会自动从参考图像中选择需要的部分来补全掩码
	- 使用两个Unet，将参考图像的K，V给源图像的Unet（拼接），做交叉注意力，提供参考信息

## Sample Editing

这种方法通过修改采样过程，对任务做映射或分解，提供任务需求的损失和实现，是十分具有创新性的工作

- **FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models** [[2412.08629] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](https://arxiv.org/abs/2412.08629)
	- 直接提出inversion-free的分布转变方法
	- 用蒙特卡洛模拟等，先构建源分布与目标分布的关系，引导流轨迹，将结果引导到目标分布，且降本增效，保持了良好的一致性
- **ZERO-SHOT IMAGE RESTORATION USING  DENOISING DIFFUSION NULL-SPACE MODEL** [[2212.00490] Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model](https://arxiv.org/abs/2212.00490)
	- 将图像修复转变成对零空间（null-space）的学习，认为设计构造A变换（依据需要什么修复）在采样过程中实现修复
	- 分析修复过程，提出采样策略
- **Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided  Diffusion Models**  [[2305.16807] Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models](https://arxiv.org/abs/2305.16807)
	- **Null-text inversion** [[2211.09794] Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://arxiv.org/abs/2211.09794)中提出对CFG中的空条件进行优化，已解决图像重建/生成与源图的一致性问题，但是这种优化需要对每一个时间步的潜在表示做MSE损失计算等，计算成本高、慢
	- 推导出实际上如果期望重建结果一致性高，就是需要正向（t-1）、逆向过程（t）的噪声一致，在假设连续的时间步的向量场近似相等下，发现其实优化的空条件就是原条件C
	- 在重建时用原条件来代替空条件（就是没使用CFG），在编辑时使用Prompt-to-Prompt的方式来做，使用编辑条件
	- **针对某个问题对CFG过程做改善，实际是修改了CFG来达到目的**