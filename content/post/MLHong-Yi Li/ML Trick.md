#  时间序列

## Conformer and Self-attention Pooling

针对语音和文本等长序列的优化Transformer和自注意力全局池化

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(SelfAttentionPooling, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            pooled_x: (batch_size, d_model)
        """
        # Compute attention scores
        attention_scores = self.attention_layer(x)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)

        # Compute attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Compute weighted sum
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), x)  # (batch_size, 1, d_model) 批量矩阵乘法运算
        pooled_x = weighted_sum.squeeze(1)  # (batch_size, d_model)

        return pooled_x


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, conv_kernel_size=31):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.mhsa = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ffn2 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Apply the first feed-forward module
        x = x + self.ffn1(x)
        x = self.layer_norm(x)

        # Apply the multi-head self-attention module
        attn_out, _ = self.mhsa(x, x, x)
        x = x + attn_out
        x = self.layer_norm(x)

        # Apply the convolution module
        conv_out = self.conv_module(x.transpose(1, 2)).transpose(1, 2)
        x = x + conv_out
        x = self.layer_norm(x)

        # Apply the second feed-forward module
        x = x + self.ffn2(x)
        x = self.layer_norm(x)

        return x

class Classifier(nn.Module):
    def __init__(self, d_model=160, n_spks=600, dropout=0.1, num_layers=2, nhead=8):
        super(Classifier, self).__init__()
        self.prenet = nn.Linear(40, d_model)

        # Conformer layers
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(d_model, nhead, dim_feedforward=256, dropout=dropout) for _ in range(num_layers)]
        )

        self.self_attention_pooling = SelfAttentionPooling(d_model)
        
        # Prediction layer
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        Args:
            mels: (batch size, length, 40)
        Returns:
            out: (batch size, n_spks)
        """
        out = self.prenet(mels)
        out = out.permute(1, 0, 2)  # (length, batch size, d_model)

        for layer in self.conformer_layers:
            out = layer(out)

        out = out.transpose(0, 1)  # (batch size, length, d_model)
        
        stats = self.self_attention_pooling(out)
        
        #stats = out.mean(dim=1)  # mean pooling
        
        out = self.pred_layer(stats)
        return out
```



# 分类任务

## AM-softmax

**加强分类区分度，在数据不平衡，类别相似度高情况下使用**

**Additive Margin Softmax (AM-Softmax)** 是一种用于深度神经网络分类任务的损失函数，它是在标准的 Softmax 损失函数基础上引入了一个加性边距（Additive Margin），以提高模型对不同类别的区分能力。这种方法常用于人脸识别、语音识别等任务。

### 1. **背景**

在分类任务中，常用的损失函数是 Softmax 结合交叉熵（Cross-Entropy Loss），其工作方式是通过最大化正确类别的预测概率来训练模型。然而，在某些任务中，如人脸识别或语音识别，我们希望模型不仅仅能够正确分类，还能够拉大不同类别之间的距离，使得模型的判别能力更强。

### 2. **AM-Softmax 的定义**

AM-Softmax 对标准 Softmax 损失函数进行了改进，它引入了一个固定的加性边距 $m$ 来拉开类别之间的距离。其核心思想是通过在 Softmax 中引入一个固定的负边距，使得模型在判别不同类别时需要达到更高的信心才能将一个样本分类为正确的类别。

AM-Softmax 的公式如下：
$$
L = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\cos(\theta_{y_i}) - m)}}{e^{s(\cos(\theta_{y_i}) - m)} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}}
$$

其中：
- $\theta_{y_i}$ 表示样本 $i$ 对应的正确类别 $y_i$ 的角度。
- $s$ 是一个缩放因子，用于控制 logits 的大小。
- $m$ 是加性边距，通过这个边距来拉开类别之间的决策边界。

### 3. **工作原理**

在标准的 Softmax 中，每个类别的 logits 是通过计算特征向量与类别权重向量的点积（或余弦相似度）来获得的。在 AM-Softmax 中，首先计算余弦相似度$\cos(\theta_j)$，然后在正确类别的 logits 中减去一个固定的边距 $m$。这样，模型需要更高的相似度（更高的置信度）才能将样本归入正确类别，从而增强了类别之间的可分性。

### 4. **优点**

- **更强的区分能力**：AM-Softmax 通过在正确类别上引入负边距，显著增强了不同类别之间的可分性，提高了模型的判别能力。
- **控制更严格的分类标准**：由于引入了边距，模型需要在更高的置信度下才能正确分类，这有助于减少误分类。

### 5. **应用场景**

AM-Softmax 常用于以下场景：
- **人脸识别**：在识别人脸时，AM-Softmax 可以有效地将不同人的特征空间拉开，使得识别精度提高。
- **语音识别**：在说话人识别任务中，AM-Softmax 可以增强对不同说话人之间特征的区分能力。

### 6. **代码示例**

AM-Softmax 通常作为损失函数的一部分，在实际使用中可能需要手动实现，下面是一个简单的实现示例：

```python
import torch
import torch.nn.functional as F

def am_softmax_loss(logits, labels, s=30.0, m=0.35):
    # Compute the cosine similarity
    cos_theta = logits
    # Additive margin
    target_logits = cos_theta - m
    # Apply scale
    logits = s * cos_theta
    target_logits = s * target_logits

    # Create mask for correct class
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
    
    # Apply mask
    logits = one_hot * target_logits + (1.0 - one_hot) * logits

    # Cross entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss

# Usage example
logits = torch.randn(32, 10)  # Example logits (32 samples, 10 classes)
labels = torch.randint(0, 10, (32,))  # Example labels
loss = am_softmax_loss(logits, labels)
print(loss)
```

### 总结

Additive Margin Softmax (AM-Softmax) 是一种改进的损失函数，通过在 Softmax 中引入加性边距来提高模型对不同类别的区分能力，常用于人脸识别、语音识别等任务。在实际应用中，AM-Softmax 有助于增强分类模型的判别力，使得模型在处理高相似性类别时表现更加稳健。

在 `am_softmax_loss` 函数中，参数 `s` 和 `m` 是超参数，用于控制 Additive Margin Softmax (AM-Softmax) 的行为。如何确定它们的值通常需要结合具体的任务进行实验和调优。下面是关于这两个参数的一些指导原则和常见的做法：

### 1. **参数 `s`（缩放因子）**

- **作用**: `s` 是一个缩放因子，用于放大 logits 的值，从而使得模型的输出更加分离。较大的 `s` 会使得类别之间的差异更加显著，增强分类器的判别能力。
  
- **选择建议**:
  - **经验值**: 通常 `s` 取值在 10 到 30 之间。例如，`s=30.0` 是一个常见的默认值，在许多研究中表现良好。
  - **任务相关**: 如果任务的类别数较多，或者需要更高的区分度，可以尝试更大的 `s` 值；如果类别之间的差异较大，较小的 `s` 可能已经足够。

- **调整策略**:
  - **增加 `s`**: 可以增加 `s` 以进一步拉开类别之间的距离，但可能导致训练变得更不稳定，需要与学习率等其他超参数共同调优。
  - **减小 `s`**: 如果发现训练过程中的梯度较大，模型不稳定，可能需要适当减小 `s`。

### 2. **参数 `m`（加性边距）**

- **作用**: `m` 是加性边距，用于强制拉开正确类别与其他类别之间的间隔。通过减去 `m`，AM-Softmax 要求模型在分类时必须达到更高的置信度才能正确分类，从而提高了模型的判别能力。

- **选择建议**:
  - **经验值**: `m` 的值通常设置为 0.2 到 0.5 之间。例如，`m=0.35` 是一个常见的默认值。
  - **任务相关**: 对于类别相似度较高的任务，可能需要更大的 `m`；而对于类别间差异较大的任务，较小的 `m` 值可能更合适。

- **调整策略**:
  - **增加 `m`**: 可以增加 `m` 来进一步增强模型的区分能力，但过大的 `m` 可能导致模型在训练初期无法收敛。
  - **减小 `m`**: 如果发现模型在训练中难以收敛，或者损失值下降缓慢，可以尝试减小 `m`。

### 3. **实验调优**

- **网格搜索**: 通过网格搜索来尝试不同的 `s` 和 `m` 组合，从中找到最优的参数配置。
- **验证集调优**: 在验证集上监控模型的性能，如准确率、F1-score 等指标，来调节 `s` 和 `m`。
- **逐步调整**: 可以先固定一个参数（如 `s`），调整另一个参数（如 `m`），然后反过来调整，以简化调优过程。

### 总结

- **`s`**: 控制分类的分离度，常见值在 10 到 30 之间，建议先从 `s=30` 开始。
- **`m`**: 控制类别之间的加性边距，常见值在 0.2 到 0.5 之间，建议从 `m=0.35` 开始。

最终的参数选择应基于你的具体任务和数据，通过实验验证不同参数组合的效果。


# HW5

机器翻译基础

