# 问题

- 如何利用上下文
- 为什么序列化
# Abstract

提出Faster R-CNN结构将检测和重识别并行，使得提出的行人选择框是由Region Proposals Network框选的低质框，而不是检测时产生的高质量结果

同时提出COntext Bipartite Graph Matching(GBGM)算法，将上下文信息用于行人匹配上

# Intro

网络提取到的低质结果对于行人的检测分类影响不大，但是对要求严格的重识别任务影响较大

目前的端对端模型都是并行结构，而两步的模型则计算量大

![Pasted image 20240831230452|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240831230452.png)

使用Faster R-CNN头作为增强的RPN，提供BBox，然后用baseline的head来提取这些BBox的判别特征，并在重识别前用[[非极大抑制|非最大抑制(NMS)]]删除冗余BBox

将查询的行人图片与库中的行人图片看成顶点，相互连接构成二分图，边权使用行人搜索网络计算的相似度，再使用K-M算法优化最大权重匹配，匹配的彼此认为top-1预测结果

最后指明在两个数据集上都取得不错的结果

# Conclusion

重述




