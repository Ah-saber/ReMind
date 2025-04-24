# Background

文生图模型虽然效果不错，但是难以做到设计恰当的prompts来生成完全符合用户需求的图片，达不到细粒度的编辑调整

#  Problem


受限于DM的弱关联性，操作上远远不如DragGAN，以往的方法大多数适用于GAN，而DM与GAN的原理和效果相去甚远，没有可编辑的潜在空间？

“Due to the lack of a concise and editable latent space,” (Mou 等, 2023, p. 1)

要找到一种适用于DM的图像细粒度编辑方法

# 现有的方案

“T2I diffusion models via correspondence between text and image features.” (Mou 等, 2023, p. 1)

基于文生图扩散模型，加强文本与图像的相关性

“Recently, self-guidance Epstein et al. (2023) proposes a differentiable approach that employs crossattention maps between text and image to locate and calculate the size of objects within images.” (Mou 等, 2023, p. 1)

微分交叉注意力图，定位和放缩图像内物体

但是只有文本和图像的相关性其实很弱，还是十分依赖于prompts，多目标情况下也难以控制文本对某个具体目标的关联

dragdiffusion使用LORA？

# 作者的方法

## 主要创新

- 文本-图像关联弱，图像之间关联强，那就使用图像特征之间的关联，可以实现图像编辑
    
- 提出能量函数来做梯度指导，通过视觉一致性的交叉注意力机制保证修改前后的图像一致性
    

## 主要方法

1. 首先通过DDIM Inversion 得到一个强相关性的特征，存入Memory Bank，对有refer的情况还要存下refer的特征表示
    
2. 构建一个energy function来做梯度指导，引导图像重构到与其的采样结果
    
    1. UNet denoiser 提取当前时间步下的特征F，再得到MB中的中间特征F_m
        
    2. 实现拖拽功能，需要两个mask来指明起点和终点
        
    3. energy function用于限制两点之间的F的相似性，用余弦相似性
        
    4. 再构建一个全局相似性的约束公式
        
    5. 二者结合取倒数构建最小化方程
        
    6. 希望未编辑的部分原来一致，构建一个m_s为主的内容一致性方程
        
    7. 二者加权组合成损失函数，有些任务额外增加另一个优化损
        
    8. 构建conditional score function？
        
3. 实验中发现只需要在前n个时间步加上引导即可
    
4. 实验发现对第二层和第三层做梯度引导可以结合底层和高层的视觉特征
    

# 实验

StaleDiffusion v1.5

几个模型的对比

结论是：DragGAN有最最高的精度，但是由于过高的相关性限制，对背景没有太好的约束，其余情况下本论文的方法基本最好

- 提到DragGAN 有无对齐情况的结果，如果不对齐，则结果不准，如果对齐则背景，或姿态出错
    
- 消融实验去掉了 DDIM，内容一致性函数，cross-attention函数，结果都不好
    

# 附录

更多的展示和对比null-text，DragGAN，DragD

在大幅的变化下，DragGAN精度下降，

注意到论文保持了更好的现实性

还有一个调查问卷

# 创新关键

- 组件式的损失函数设置，一个通用的大框架满足各种应用需求，即插即用，随时更改
    
- 无需训练和模块
    
- 解决DragGAN做不到的一致性问题
    
    - 多尺度特征引导
        
    - 全局，内容相似性梯度指导
        
    - MB的设置，初始一致性DDTM，交叉注意力
        

适用于多种编辑方式

- 为什么用能量函数
    
    - 梯度指导
        
    - 多用于可控生成任务
        
    - score-based DM设计逻辑
        

# 实验

# 不足和启发

- 分析方法（逻辑与局限）
    
- 对比Drag系列模型
    
- 分析相关性


# 疑问与解答

## 什么是能量函数，其公式的含义

来源于 “Diffusion models beat gans on image synthesis” (Mou 等, 2023, p. 10)

其用于分类器based的梯度引导中，定义为分类器对条件y的预测概率

用来加到条件生成梯度中，引导生成结果

本文的能量函数是自己的定义，同余弦相似度的倒数来构造，避免了分类器的训练，可对不同需求修改不同的引导来实现任务目标

个人认为是最大创新

## 什么是DDIM inversion

DDIM是扩散模型采样方法，通过确定性反向过程加速图像生成，DDIM inversion是逆向操作，旨在讲帧数图像编码到潜在空间中的初始状态$z_t$

>[!什么是确定性？]
>不同于正向过程的随机噪声，反向过程是通过确定性步骤实现的
>
>  $$z_{t-1} = \sqrt{\alpha_{t-1}} \cdot \underbrace{\frac{z_t - \sqrt{1-\alpha_t} \epsilon_\theta(z_t, t)}{\sqrt{\alpha_t}}}_{\text{预测的 } z_0} + \sqrt{1-\alpha_{t-1}} \cdot \epsilon_\theta(z_t, t)$$
>  
>  DDIM Inversion 逆向使用该公式，从$z_0$ 逐步推导到 $z_T$

文中先生成$z_0$再解码回原图，存入Memory Bank中，应用于后续的梯度指导中


## 指导如何加入能量函数

score function的含义是指向概率密度增长最快的方向，分为有条件和无条件，在梯度指导中其实就是修改有条件score function来指导概率变化方向，生成不同的图片
