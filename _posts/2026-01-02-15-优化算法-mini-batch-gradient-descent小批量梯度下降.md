---
title: "15 优化算法 —— Mini-batch Gradient Descent（小批量梯度下降）"
date: 2026-01-02T10:33:44Z
last_modified_at: 2026-01-02T10:34:03Z
categories:
  - 改进深度神经网络
math: true
---

### 一、背景与动机

　　深度学习通常在大规模数据集上训练神经网络。若使用**全批量梯度下降（Batch Gradient Descent）** ，每次参数更新需遍历全部 $m$ 个训练样本：

$$
\theta := \theta - \alpha \nabla_\theta J(\theta; X, Y)
$$

　　其中 $X \in \mathbb{R}^{n_x \times m}$，$Y \in \mathbb{R}^{1 \times m}$。当 $m$ 极大（如 $5 \times 10^6$）时，单次梯度计算代价高昂，训练效率低下。

　　为加速训练，引入 **Mini-batch Gradient Descent**：在处理完一小部分数据后即更新参数，实现更频繁的迭代。

---

### 二、Mini-batch 的划分与符号约定

　　将训练集 $\{X, Y\}$ 划分为 $T = \frac{m}{b}$ 个 mini-batches，其中 $b$ 为 mini-batch 大小（如 $b = 1000$）。

- 第 $t$ 个 mini-batch 的输入和标签记为：

  $$
  X^{\{t\}} \in \mathbb{R}^{n_x \times b}, \quad Y^{\{t\}} \in \mathbb{R}^{1 \times b}
  $$
- 符号区分：

  - $x^{(i)}$：第 $i$ 个训练样本（圆括号）
  - $W^{[l]}$：第 $l$ 层的权重（方括号）
  - $X^{\{t\}}$：第 $t$ 个 mini-batch（花括号）

---

### 三、Mini-batch Gradient Descent 算法步骤

　　对每个 epoch（完整遍历训练集一次），执行以下流程：

1. **前向传播**（在第 $t$ 个 mini-batch 上）：

   $$
   \begin{aligned}
   Z^{[1]} &= W^{[1]} X^{\{t\}} + b^{[1]} \\
   A^{[1]} &= g^{[1]}(Z^{[1]}) \\
   &\vdots \\
   A^{[L]} &= g^{[L]}(Z^{[L]})
   \end{aligned}
   $$
2. **计算 mini-batch 损失**：

   $$
   J^{\{t\}} = \frac{1}{b} \sum_{i=1}^{b} \mathcal{L}\left( \hat{y}^{(i)}, y^{(i)} \right) + \frac{\lambda}{2b} \sum_{l=1}^{L} \|W^{[l]}\|_F^2
   $$

   其中 $\hat{y}^{(i)}$ 和 $y^{(i)}$ 来自 $X^{\{t\}}$ 与 $Y^{\{t\}}$，$\|\cdot\|_F$ 表示 Frobenius 范数。
3. **反向传播**：基于 $(X^{\{t\}}, Y^{\{t\}})$ 计算梯度 $\nabla_{W^{[l]}} J^{\{t\}}$ 和 $\nabla_{b^{[l]}} J^{\{t\}}$。
4. **参数更新**：

   $$
   \begin{aligned}
   W^{[l]} &:= W^{[l]} - \alpha \cdot \nabla_{W^{[l]}} J^{\{t\}} \\
   b^{[l]} &:= b^{[l]} - \alpha \cdot \nabla_{b^{[l]}} J^{\{t\}}
   \end{aligned}
   $$

　　重复上述过程 $t = 1, 2, \dots, T$，完成一个 epoch。每个 epoch 包含 $T = m/b$ 次参数更新。

---

### 四、与其他方法的对比

|方法|每次更新所用样本数|向量化效率|更新频率|收敛行为|
| ---------------------| --------------------| ----------------------| ------------| --------------|
|Batch GD|$m$（全集）|高（但 $m$ 过大时受限）|1 次/epoch|平滑但极慢|
|Stochastic GD (SGD)|1|低（难以向量化）|$m$ 次/epoch|噪声大，震荡|
|**Mini-batch GD**|$b$（如 64–1024）|**高（GPU 友好）**|$m/b$ 次/epoch|**稳定且高效** ✅|

---

### 五、关键概念澄清

- **Epoch**：模型遍历整个训练集一次。
- **Iteration / Step**：一次基于单个 mini-batch 的参数更新。
- **Mini-batch size** **$b$**：典型值为 2 的幂（如 64, 128, 256, 512, 1024），以匹配硬件内存对齐和并行计算优化。

---

### 六、总结

1. Mini-batch Gradient Descent 通过**分块处理数据**，在保持向量化效率的同时显著提升训练速度。
2. 每个 mini-batch 内仍使用**完整的前向/反向传播流程**，仅作用于子集数据。
3. 相比 Batch GD，它在**一个 epoch 内完成多次参数更新**，加速收敛。
4. 它是现代深度学习框架（如 TensorFlow、PyTorch）中**默认采用的优化策略基础**。
