Life Long Learning 实际上是指机器可以一直不断地学习，包括学习多个技能，也包括不断更新自己在某方面的能力

## Problem

学习任务一，再学任务二（不同分布或类似的问题），发现任务二学习到了但是任务一忘记了；机器无法依次学习不同的任务，同时学习却都可以学会

这称为**Catastophic Forgetting**，灾难性遗忘

因此在进行Life Long learning 的尝试的时候，通常以**Multi-task training**的结果作为上界(upper bound)

**Evaluation**  如何评估LLL的性能？可以直接对最后的个任务精度做加和衡量精度；也可以将最后的结果与每个任务刚学完的成果做差值，衡量遗忘水平；也有通过纵向对比衡量在学习各个任务后，某个任务的精度变化
![Pasted image 20240816092722|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816092722.png)

# Solution

## Why Forgetting

在任务一上训练，使得loss降低，此时会将参数变化到使得任务一的loss低的位置；再以此对任务二进行训练，此时参数的值变化到任务二loss低的位置，由于二者的loss分布不同，在任务二loss小并不意味着在任务一loss小，因此发生Forgetting

![Pasted image 20240816093813|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816093813.png)

>蓝色表示loss低，白色loss高

## Selective Synaptic Plasticity

选择性突触可塑性

是**Regularization-base approach**

既然出现遗忘的原因是loss的分布不同，有无可能将参数训练到两者loss都满足的情况？

一个想法是尝试训练任务一中不重要的参数

![Pasted image 20240816094413|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816094413.png)

修改Loss function，增加一项表示新参数与旧参数之间的距离，加权b表示该参数对任务一的重要性

这可能有两个问题
- Catastrophic Forgetting
	如果b的权重过小，每个参数都大幅改变，则无法解决遗忘问题
- Intransigence
	如果每个参数都过大，不改变，则不明智，无法学习新的任务

### B的设计

权重b无法直接用训练得到，因为要让Loss最小，权重值最小即可

因此要认为参与，尝试变动某些参数，如果对Loss影响大，则权重b大，否则权重b小

![Pasted image 20240816095612|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816095612.png)

### Gradient Episodic Memory

梯度情景记忆

update 梯度来修改更新方向，找到一个与原来的梯度内积结果大于0，且与原来任务二的最佳梯度尽可能接近的方向


![Pasted image 20240816100327|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816100327.png)

但是需要计算前面任务的梯度，也就是需要存储一些数据，这不符合Life Long Learning的宗旨，尽管只要一点数据

## Additional Neural Resource Allocation

额外的神经元分配

每个任务有各自的神经元，新的任务会接收旧任务每一层的输出与新的输入，训练额外的神经元

![Pasted image 20240816100946|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816100946.png)

相反的思路称为PackNet

![Pasted image 20240816101051|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816101051.png)

## Memory Reply

**Generating Data** 训练一个生成器，负责生成之前的数据，或许类似于回忆，以此将两种数据送给模型做训练，使模型学会两种任务

![Pasted image 20240816101301|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816101301.png)

## Curriculum Learning

遗忘其实与顺序有关，如果任务学习的顺序改变，有些遗忘就不会发生

![Pasted image 20240816101821|500](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240816101821.png)

研究学习顺序的，称为curriculum learning

