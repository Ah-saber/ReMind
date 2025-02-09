# Intro

多视角学习/多视图学习，陶大成提出的一个研究方向，在实际应用中，对同一个事物从多个途径或不同角度进行描述，构成多个视图

多视图学习就是**引入一个函数模型化一个特定的视角，并利用相同输入的冗余视角联合优化所有函数，提高学习效果**

# Algorithm

对多个不同特征集表示的数据进行机器学习

遵循两条准则：
- consensus principle
	共识准则，多个不同视角来自一个对象，理应具有一致性
- complementary priciple
	互补准则，每个视图都足以完成特定的知识发现任务，而不同视图通常得存在互补的信息，利用互补信息充分描述对象，可以提供更深入的见解

陶大成将现有算法分成三类
- Co-training 协同学习
- Multiple Kernel Learning 多核学习
- Subspace Learning 子空间学习

## Co-training

在未标记数据的两个视图下，轮流训练达到彼此的一致性最大化

假设每个数据可以从不同角度分类，不同角度训练出不同的分类器，彼此互补，提高精度

不同角度训练不同的分类器，再用这些分类器对无标签数据进行分类，挑选可信的标签与数据继续训练，直到达到目标

假设存在前提：
- 充分性
- 兼容性
- 条件独立性
	难以满足

## Multiple Kernel Learning

[[Kernel Function]]在多视图学习的应用，因为核函数本来就是在不同分布下的数据，对应不同的视图，通过将核线性的或非线性的组合，可以得到统一的核，以此提高目标效果

关键问题在于找到合适的核优化组合算法

## Subspace learning

每个样本看成是高维空间分布一个点，每个视角所有样本的分布构成一个样本空间，子空间学习认为这些样本空间存在一个公共子空间，各个视角的各个样本在这个公共子空间都有一个投影，称为**表示** 

子空间学习目标就是寻找这个公共子空间，让各样本的表示具有更好的性质或保持原始分布的特性，因为子空间的维度会有低于原始空间的情况，子空间学习和多视角降维几乎相同

因为维度的差异，跨视角的度量是其面临的主要难题


# Acknowledgement


[多视图学习 (Multi-View Learning)-CSDN博客](https://blog.csdn.net/xq151750111/article/details/123346744)


# 或许有用的几个方向

- 多数据
- 深度学习
- 更多视图
- 不完全视图
- 高斯过程
- 贝叶斯框架

贝叶斯框架知识补全
CCA等算法