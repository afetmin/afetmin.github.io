---
title: "16 向量化逻辑回归（Vectorizing Logistic Regression）"
date: 2025-12-08T19:50:52Z
last_modified_at: 2025-12-08T20:00:57Z
categories:
  - 神经网络与深度学习
math: true
---

### 🎯 核心目标

　　在不使用任何显式 `for` 循环的前提下，**一次性对整个训练集**（共 $M$ 个样本）进行前向传播（forward propagation）和激活值计算，从而大幅提升计算效率。这是深度学习中**向量化**（vectorization）的核心思想。

---

## 1️⃣ 传统方法 vs 向量化方法

### ❌ 传统方法（低效）

　　对于每个训练样本 $x^{(i)} \in \mathbb{R}^{n_x}$，依次计算：

$$
z^{(i)} = w^T x^{(i)} + b \\
a^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}
$$

　　需要循环 $M$ 次 → 时间复杂度高，无法利用现代 CPU/GPU 的并行能力。

### ✅ 向量化方法（高效）

　　将所有训练样本**堆叠成矩阵**，一次性完成全部计算。

---

## 2️⃣ 数据表示：从向量到矩阵

- 单个输入样本：$x^{(i)} \in \mathbb{R}^{n_x}$
- 将 $M$ 个样本横向堆叠成 **输入矩阵** $X$：

  $$
  X = \begin{bmatrix}
  \vert & \vert & & \vert \\
  x^{(1)} & x^{(2)} & \cdots & x^{(M)} \\
  \vert & \vert & & \vert
  \end{bmatrix} \in \mathbb{R}^{n_x \times M}
  $$

> 💡 注意：每一列是一个样本（column-major），这是深度学习中的标准约定。

---

## 3️⃣ 向量化前向传播

### 步骤 1：计算所有 $z^{(i)}$ → 得到向量 $Z$

　　我们希望一次性计算：

$$
Z = \begin{bmatrix} z^{(1)} & z^{(2)} & \cdots & z^{(M)} \end{bmatrix} \in \mathbb{R}^{1 \times M}
$$

　　利用矩阵运算：

$$
Z = w^T X + b
$$

　　其中：

- $w \in \mathbb{R}^{n_x}$ 是权重向量
- $w^T X \in \mathbb{R}^{1 \times M}$：每一列是 $w^T x^{(i)}$
- $b \in \mathbb{R}$ 是标量偏置

> 🔍 **广播机制**（Broadcasting）  
> 虽然 $b$ 是标量，但在 NumPy 中执行 `w.T @ X + b` 时，Python 会自动将 $b$ 广播为形状 $(1, M)$ 的行向量 $[b, b, \dots, b]$，然后逐元素相加。

### 步骤 2：计算所有激活值 $a^{(i)}$ → 得到向量 $A$

　　定义 sigmoid 函数作用于整个矩阵（**向量化 sigmoid**）：

$$
A = \sigma(Z) = \frac{1}{1 + e^{-Z}} \quad \text{（逐元素应用）}
$$

　　结果：

$$
A = \begin{bmatrix} a^{(1)} & a^{(2)} & \cdots & a^{(M)} \end{bmatrix} \in \mathbb{R}^{1 \times M}
$$

---

## 4️⃣ Python / NumPy 实现（一行代码！）

```python
import numpy as np

# 假设：
# X.shape = (n_x, M)
# w.shape = (n_x, 1) 或 (n_x,) —— 但通常用 (n_x, 1)
# b is a scalar

Z = np.dot(w.T, X) + b          # shape: (1, M)
A = 1 / (1 + np.exp(-Z))        # vectorized sigmoid, shape: (1, M)
```

> ✅ 这两行代码就完成了对整个训练集的前向传播，**无任何 for 循环**！

---

## 5️⃣ 关键优势总结

|项目|传统方法|向量化方法|
| ------------| --------------------| --------------------|
|循环次数|$M$ 次|0 次|
|计算效率|低（串行）|高（并行）|
|代码简洁性|冗长|极简（1~2 行）|
|可扩展性|难以扩展到深层网络|天然适用于神经网络|

---

## 6️⃣ 后续预告：向量化反向传播

　　课程提到，**反向传播**（backward propagation）同样可以向量化：

- 一次性计算所有样本的梯度 $\frac{\partial \mathcal{L}}{\partial w}$ 和 $\frac{\partial \mathcal{L}}{\partial b}$
- 利用 $dZ = A - Y$（其中 $Y$ 是真实标签矩阵）
- 最终得到：

  $$
  dw = \frac{1}{M} X (A - Y)^T \\
  db = \frac{1}{M} \sum_{i=1}^M (a^{(i)} - y^{(i)})
  $$

> 这部分将在下一节详细讲解。

---

## 📌 总结要点（Key Takeaways）

1. **向量化是深度学习高效计算的基石**。
2. 将训练样本按列堆叠成矩阵 $X \in \mathbb{R}^{n_x \times M}$。
3. 前向传播可写为：

    $$
    Z = w^T X + b,\quad A = \sigma(Z)
    $$
4. 利用 NumPy 的**广播机制**和**向量化函数**，避免显式循环。
5. 此方法可无缝推广到多层神经网络。

---

　　如果你正在学习深度学习编程，强烈建议你在 Jupyter Notebook 中亲手实现这个向量化逻辑回归，并与 for 循环版本对比速度（可用 `%timeit`），你会直观感受到向量化的威力！
