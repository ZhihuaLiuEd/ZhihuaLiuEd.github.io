<head>
    <script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>
</head>

# 测试

## Introduction

1.  $\boldsymbol{O}(T^2D)$，其中$T$是时间序列的长度，$D$是隐变量的维度。对于长序列时序预测（Long Sequence Time-series Forecasting，也被称为LSTF问题），这种计算复杂度可能会非常高。为了解决这个问题，Informer使用了一个叫做ProbSparse attention的新自注意力机制，它的时间空间复杂度为$O(T \log T)$。
2. 堆叠功能层时的内存瓶颈：当我们堆叠$N$个编码/解码层时，原始的Transformer对内存的使用复杂度为$O(NT^2)$，这限制了模型对于长时序列的容量。Informer使用了一个蒸馏的操作，将层间的输入尺寸减小为原来的一半。通过这种操作，Informer将内存的使用减小至$O(N \cdot T \log T)$。

$$
Attention(Q,L,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
其中$Q \in \mathbb{R}^{L_Q \times d}$, $K \in \mathbb{R}^{L_K \times d}$, $V \in \mathbb{R}^{L_V \times d}$。注意在实际应用中，Query和Key的输入序列长度在自注意力计算中通常是相等的，即$L_Q = L_K = T$，其中$T$是时间序列的长度。因此，$QK^T$的乘积计算复杂度为$O(T^2 \cdot d)$。在ProbSparse attention中，我们的目标时构建一个新的$Q_{reduced}$矩阵，定义为：
$$
\text{ProbSparseAttention}(Q,K,V) = \text{softmax}(\frac{Q_{reduced}K^T}{\sqrt{d_k}})V
$$