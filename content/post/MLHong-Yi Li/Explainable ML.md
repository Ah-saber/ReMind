# Why

机器学习的模型在某些领域不能只当作黑箱，其功能必须存在可解释性
- 银行
- 医疗
- 法律

理解其应用并得到答案的理由

# How 

如何才是Explainable ML

真正的可解释性AI并不一定是完全能够解释其全部的运行原理，可以存在一些未解，但是需要为其判断或结果提供理由

# Explainable ML

## Local Explanation

 对于具体的某一个任务做出解释：
 
 >为什么这是一只猫

具体而言，对于一张图片或文字，哪一个特征（像素）重要，会影响到模型做出判断

### 方法

### **判断理由**

- mask
	挡住某些部位，尝试是否影响机器做出判断
- 增加x
	对某个特征增加x，在增加下计算loss，如果对loss影响大说明该特征重要，通过loss对x的偏微分，可以得到重要性图像Saliency Map
	![Pasted image 20240804082729|400](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240804082729.png)
	
	**得到更好的Saliency Map:** 
	- SmoothGrad 
		增加杂讯做数据增强，将各种杂讯的Saliency Map合并起来作为最终结果
		
	然而只看特征的导数不一定看得出某个特征是否重要，可能该特征处于loss的平滑区，梯度小，但是其在非平滑区梯度十分大

### **过程解释**：

- Visualization
	直接将特征向量取出观察，媒介段理解其作用（降维）
	
	看attention结果
- Probing
	探针 ，训练模型当作探针（例如分类器）将过程结果试着转化为某种信息，如果转化的正确率高，说明这个过程结果中这种信息的含量高
	
	或者合成模型，尝试从某个过程还原另一个模型的输入结果，如果此时实现了该模型的任务，例如去掉这个输入的某种性质，还原之后这种性质确实消失了
	
- **Global Explanation**
	对于某一个概括性的任务做出解释：猫是什么样的

## Global Explanation

### 计算Global Explanation

在训练好的模型，例如CNN中，对一个feature map中的任意一个矩阵元素 $a_{ij}$ ，训练一个图片X来让这些元素 和 最大

$$X* = arg max_x \sum_i\sum_j a_{ij}(gradient ascent)$$
这样一来就知道这个特征图代表的模型想找的图片是什么

也可以直接研究模型整体的期待输入

训练一个X，期待模型对某一个分类结果的输入可能性越大越好，但是这样直接做会因为存在（adversarial attack）得到杂讯

增加限制，例如图片本身限制：没有噪点；例如目标图片限制：长得像数字；

或者使用生成模型，接到分类模型之前，使用生成模型生成X，X生成y，训练y越大越好

![Pasted image 20240806105745](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240806105745.png)

### Outlook

简单模型模仿复杂模型

![Pasted image 20240806110033](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240806110033.png)

但是简单模型不一定能够模仿复杂模型，所以提出 **Local Interpertable Model-Agnostic Expalnations (LIME)**


