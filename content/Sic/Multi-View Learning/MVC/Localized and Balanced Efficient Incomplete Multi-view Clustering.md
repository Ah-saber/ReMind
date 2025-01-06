#Jie_Wen  #IMVC 

_Mark  -AS     23:03  11.17  2024_
# Object Function
## Method

![{19E6565F-EBB4-49C5-B269-333F7F4405CA}|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/%7B19E6565F-EBB4-49C5-B269-333F7F4405CA%7D.png)


基于**非负矩阵分解**，参考 **OPIMVC**  和 **GRPMVC**

提出一个包含 软标签 一步聚类，最近邻图结构学习，视图自适应加权和一步聚类矩阵正则化避免无意义解（平衡项）的 集成模型 **LBIMVC**

## Focus

以往聚类通过 K-means 的弊端，实现在模型中直接聚类

解决OPIMVC中一步聚类的额外约束问题，保证聚类矩阵优化的合理性

引用W矩阵，将**最近邻图结构学习**和不完全多视图共识表示学习集中在一个框架里

# Innovation

-  实现软标签聚类
- 平衡项的提出与Lasso回归求解
- W矩阵结合两个框架

# Idea

- 最近邻学习参考
	- HeatKernel 构造最近邻矩阵
- 一步聚类参考

为什么不在做视图融合的时候再做一步聚类呢？

