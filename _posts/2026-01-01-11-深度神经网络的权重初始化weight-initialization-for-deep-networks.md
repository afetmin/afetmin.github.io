---
title: "11 深度神经网络的权重初始化（Weight Initialization for Deep Networks）"
date: 2026-01-01T20:42:48Z
last_modified_at: 2026-01-01T20:45:04Z
categories:
  - 改进深度神经网络
math: true
---

## 🧠 课程核心目标

　　解决深度神经网络训练中的 **梯度消失（vanishing gradients）**  和 **梯度爆炸（exploding gradients）**  问题。  
虽然权重初始化不能完全消除该问题，但**合理的初始化策略能显著缓解**，从而提升训练稳定性和收敛速度。

---

## 🔍 1. 单神经元示例分析

　　考虑一个单神经元模型：

- 输入：$x_1, x_2, \dots, x_n$（共 $n$ 个特征）
- 权重：$w_1, w_2, \dots, w_n$
- 偏置设为 0（简化分析）：$b = 0$
- 线性组合：

  $$
  z = \sum_{i=1}^{n} w_i x_i = \mathbf{w}^\top \mathbf{x}
  $$
- 激活输出：$a = g(z)$

### ⚠️ 问题

　　若 $w_i$ 初始化过大 → $z$ 过大 → 激活函数饱和（如 sigmoid/tanh）→ 梯度接近 0（**梯度消失**）  
若 $w_i$ 初始化过小 → $z$ 接近 0 → 梯度极小 → 学习缓慢  
若层数很深，误差会逐层放大或缩小 → **梯度爆炸/消失**

### ✅ 直觉解决方案

- $z$ 是 $n$ 项之和，因此每项 $w_i x_i$ 应“小一点”，避免总和过大或过小。
- 若输入 $x_i$ 满足：$\mathbb{E}[x_i] \approx 0$，$\mathrm{Var}(x_i) \approx 1$
- 则希望：$\mathrm{Var}(z) \approx 1$

　　由于：

$$
\mathrm{Var}(z) = \mathrm{Var}\left( \sum_{i=1}^n w_i x_i \right) = \sum_{i=1}^n \mathrm{Var}(w_i x_i) = \sum_{i=1}^n \mathrm{Var}(w_i) \cdot \mathrm{Var}(x_i)
$$

　　若 $w_i \sim \mathcal{N}(0, \sigma^2)$，且 $\mathrm{Var}(x_i) = 1$，则：

$$
\mathrm{Var}(z) = n \cdot \sigma^2
$$

　　令 $\mathrm{Var}(z) = 1$ ⇒ $\sigma^2 = \frac{1}{n}$

---

## 📐 2. 推广到深度网络

　　对于第 $l$ 层（layer $l$）：

- 输入来自前一层的激活值：$\mathbf{a}^{[l-1]} \in \mathbb{R}^{n^{[l-1]}}$
- 权重矩阵：$\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$
- 初始化建议：

  $$
  \mathbf{W}^{[l]} = \texttt{np.random.randn}(n^{[l]}, n^{[l-1]}) \times \sqrt{\frac{k}{n^{[l-1]}}}
  $$

  其中 $k$ 取决于激活函数类型。

---

## 🧪 3. 不同激活函数对应的初始化策略

### (1) ReLU 激活函数（最常用）

- 推荐使用 **He 初始化（He Initialization）**

  $$
  \mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right)
  $$

  即：

  $$
  \mathbf{W}^{[l]} = \texttt{np.random.randn}(\cdots) \times \sqrt{\frac{2}{n^{[l-1]}}}
  $$

> 💡 理由：ReLU 会使一半神经元输出为 0，有效方差减半，因此需将方差加倍以补偿。

---

### (2) Tanh / Sigmoid 激活函数

- 推荐使用 **Xavier 初始化（Glorot Initialization）**

  $$
  \mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{1}{n^{[l-1]}}\right)
  $$

  即：

  $$
  \mathbf{W}^{[l]} = \texttt{np.random.randn}(\cdots) \times \sqrt{\frac{1}{n^{[l-1]}}}
  $$

> 📚 由 Xavier Glorot & Yoshua Bengio (2010) 提出，适用于对称激活函数（如 tanh）。

---

### (3) 其他变体（较少用）

- Yoshua Bengio 团队提出更复杂的初始化（如考虑前后层维度平均）：

  $$
  \mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]} + n^{[l]}}\right)
  $$

  但在实践中，**He 初始化（ReLU）和 Xavier 初始化（tanh）已足够有效**。

---

## ⚙️ 4. 实践建议与超参数调优

- **默认选择**：

  - 使用 ReLU → 用 He 初始化（$\sqrt{2 / n^{[l-1]}}$）
  - 使用 Tanh → 用 Xavier 初始化（$\sqrt{1 / n^{[l-1]}}$）
- **可调超参数**：

  - 可引入缩放因子 $\alpha$：

    $$
    \mathbf{W}^{[l]} = \texttt{np.random.randn}(\cdots) \times \alpha \cdot \sqrt{\frac{2}{n^{[l-1]}}}
    $$
  - 将 $\alpha$ 作为超参数进行微调（但通常影响较小）
- **优先级**：权重初始化重要，但**不如学习率、网络结构、批量大小等关键**，属于“第二梯队”调参项。

---

## ✅ 总结：为什么有效？

　　合理初始化使每一层的 **输入和输出保持相似的尺度（scale）** ，从而：

- 防止 $z^{[l]}$ 过大或过小
- 保持激活值在非饱和区
- 使梯度在反向传播中既不爆炸也不消失
- **显著提升深度网络的可训练性**

---

## 📘 公式速查表

|激活函数|初始化方差|初始化代码（Python）|
| ----------| ------------| ----------------------|
|ReLU|$\displaystyle \frac{2}{n^{[l-1]}}$|`np.random.randn(...) * np.sqrt(2 / n_prev)`|
|Tanh|$\displaystyle \frac{1}{n^{[l-1]}}$|`np.random.randn(...) * np.sqrt(1 / n_prev)`|

　　其中 $n^{[l-1]}$ 表示第 $l-1$ 层的神经元数量（即当前层的输入维度）。
