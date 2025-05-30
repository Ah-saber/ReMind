# 概念

要找什么样的特征

特征要保持数据的奇异性，也就是更能表示数据的分布，那么要求这个特征向量在数据空间中的方差越大越好

就这样找多个相互垂直的特征，就可以将向量压缩到目标维度


## Lagrange multiplier

利用Lagrange multiplier求解这个x，要求方差最大

计算可以得到，求解第一个x时，定义 $W \alpha = x$ ，发现这个式子是x的协方差的特征值，W是特这个向量，要找最大，即W是最大的特征向量，$\alpha$是特征值

第二个X要求满足与第一相互垂直，增加限制进行结算

因为协方差是对称的，所以可以选第二大的特征值，以此类推

![[Pasted image 20240802112533.png|400]]


参考：[ML Lecture 13: Unsupervised Learning - Linear Methods (youtube.com)](https://www.youtube.com/watch?v=iwh5o_M4BNU)

