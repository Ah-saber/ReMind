# 计算

## E步

E步计算X属于某个分布的期望，其中X为数据点，Z表示X属于某个分布的标签，Z=k即表示X属于k这个分布

在 EM 算法的 E 步（期望步）中，**Z 属于某个分布的概率**通过计算 **后验概率** 来体现。具体来说，每个数据点 $X$ 都可能由不同的分布生成，而这些分布对应于不同的混合成分（例如在高斯混合模型中，每个高斯分布就是一个成分）。

### 如何在计算中体现：

1. **模型参数和数据点  $X$**：
   - 通过观察数据 $X$和当前的模型参数 $\theta^{\text{old}}$（包括每个成分的参数，如均值、协方差矩阵、混合系数等），你可以计算每个 $X$ 属于不同成分的**似然函数**$p(X \mid Z=k, \theta_k^{\text{old}})$ ，即在第 $k$ 个成分下生成 $X$ 的概率。

2. **Z 的后验概率**：
   - 后验概率 $p(Z=k \mid X, \theta^{\text{old}})$ 是通过**贝叶斯定理**计算得到的：
   
   $$
   p(Z=k \mid X, \theta^{\text{old}}) = \frac{p(X \mid Z=k, \theta_k^{\text{old}}) \cdot \pi_k^{\text{old}}}{\sum_{j=1}^K p(X \mid Z=j, \theta_j^{\text{old}}) \cdot \pi_j^{\text{old}}}
   $$
   
   - $p(X \mid Z=k, \theta_k^{\text{old}})$ 是 $X$ 在第$k$ 个成分下的**似然**，描述了第 $k$ 个成分生成数据 $X$ 的可能性。
   - $\pi_k^{\text{old}}$ 是第 $k$ 个成分的**先验概率**，表示从第 $k$ 个成分生成数据的初始可能性。
   - 分母部分是所有成分的加权似然，用来归一化使得后验概率的总和为 1。

3. **解释**：
   - 对每一个数据点 X，我们根据当前模型参数 $\theta^{\text{old}}$，计算其在每个成分下的似然，然后用这些似然和每个成分的先验概率来计算数据点$X$ 属于每个成分的后验概率（即 $Z=k$ 的概率）。
   - 这个后验概率 $p(Z=k \mid X, \theta^{\text{old}})$ 表示给定当前参数估计下，数据点 $X$ 属于第 $k$ 个分布的概率。

在这样的基础上，E步所求的期望值为每个X所属分布的表示Z，求对数似然函数，再以Z=k的概率作为其权值进行加权求和

$$
Q( \mathbf{\theta}, \mathbf{\theta}^{\text{old}} ) = \sum_{\mathbf{Z}} p( \mathbf{Z} \mid \mathbf{X}, \mathbf{\theta}^{\text{old}} ) \ln p( \mathbf{X}, \mathbf{Z} \mid \mathbf{\theta} )
$$

## M步

M步就是通过最大化期望Q来更新 $\theta$ 

公式为 

$$ 
\theta \, = arg\, maxQ(\theta, \theta^{old})
$$

找到一个参数最大化期望Q，更新参数 $\theta$ 
