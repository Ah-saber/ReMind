# 步骤

1. 载入数据集
	- 定义初始化函数
	- 定义按索引获取函数
	- 定义返回尺寸函数
2. 定义model
	- 初始化函数
		- 调用父类构造函数
		- 定义`self.layers=nn.Sequential()`模型结构
		- 定义前面传播函数
3. 定义特征获取函数
	- 选择特征或默认全部特征
4. Trianer函数
	- 定义损失函数criterion
	- 定义优化器optimizer
	- tensorboard
	- 训练epochs循环，内嵌每个批次训练循环
		- 同时内嵌验证循环
	- 输出数据，保存最佳，提前停止判断
5. 定义超参数
6. 数据可视化（其实可以在前面）
7. 数据划分
8. 训练
9. 测试


# 图像

## 感受野与卷积尺寸计算

### 感受野

- 前向公式
	![Pasted image 20240720111342|300](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240720111342.png)
- 后向公式
	![Pasted image 20240720111355|300](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240720111355.png)
	

