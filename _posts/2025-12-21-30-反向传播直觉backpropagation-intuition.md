---
title: "30 反向传播直觉（Backpropagation Intuition）"
date: 2025-12-21T20:00:15Z
last_modified_at: 2025-12-21T20:00:53Z
categories:
  - 神经网络与深度学习
math: true
---

## 一、背景：从逻辑回归说起

　　在逻辑回归中，我们有如下前向传播流程：

$$
\begin{aligned}
z &= \mathbf{w}^\top \mathbf{x} + b \\
a &= \sigma(z) = \frac{1}{1 + e^{-z}} \\
\mathcal{L}(a, y) &= -y \log a - (1 - y) \log(1 - a)
\end{aligned}
$$

### 反向传播步骤（单样本）：

1. **对** **$a$** **求导（损失对激活值的梯度）** ：

   $$
   da = \frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1 - y}{1 - a}
   $$
2. **对** **$z$** **求导（利用链式法则）** ：

   $$
   dz = \frac{\partial \mathcal{L}}{\partial z} = da \cdot \sigma'(z) = a - y
   $$

   > 注：因为 $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = a(1 - a)$，代入后可简化为 $dz = a - y$。
   >
3. **对参数求导**：

   $$
   \begin{aligned}
   d\mathbf{w} &= dz \cdot \mathbf{x} \\
   db &= dz
   \end{aligned}
   $$

---

## 二、推广到两层神经网络（一个隐藏层）

　　网络结构：输入层 → 隐藏层（激活函数 $g_1$）→ 输出层（sigmoid）

### 前向传播（单样本）：

$$
\begin{aligned}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} \\
\mathbf{a}^{[1]} &= g_1(\mathbf{z}^{[1]}) \\
z^{[2]} &= \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]} \\
a^{[2]} &= \sigma(z^{[2]}) \\
\mathcal{L}(a^{[2]}, y) &= -y \log a^{[2]} - (1 - y) \log(1 - a^{[2]})
\end{aligned}
$$

> 注：上标 $[l]$ 表示第 $l$ 层；$\mathbf{a}^{[1]}$ 是隐藏层输出（向量），$a^{[2]}$ 是标量（二分类输出）。

---

### 反向传播（单样本）——6个核心公式

1. **输出层误差**：

   $$
   dz^{[2]} = a^{[2]} - y
   $$
2. **输出层参数梯度**：

   $$
   \begin{aligned}
   d\mathbf{W}^{[2]} &= dz^{[2]} \cdot (\mathbf{a}^{[1]})^\top \\
   db^{[2]} &= dz^{[2]}
   \end{aligned}
   $$
3. **隐藏层误差（关键！链式法则 + 激活函数导数）** ：

   $$
   d\mathbf{z}^{[1]} = (\mathbf{W}^{[2]})^\top dz^{[2]} \odot g_1'(\mathbf{z}^{[1]})
   $$

   > 其中 $\odot$ 表示 **逐元素乘积（Hadamard product）** 。
   >
4. **隐藏层参数梯度**：

   $$
   \begin{aligned}
   d\mathbf{W}^{[1]} &= d\mathbf{z}^{[1]} \cdot \mathbf{x}^\top \\
   db^{[1]} &= d\mathbf{z}^{[1]}
   \end{aligned}
   $$

---

### 维度验证（重要！避免 bug）

　　设：

- 输入维度：$n_x = n^{[0]}$
- 隐藏层单元数：$n^{[1]}$
- 输出层单元数：$n^{[2]} = 1$（二分类）

　　则：

|变量|维度|
| ------| ------|
|$\mathbf{x}$|$(n_x, 1)$|
|$\mathbf{W}^{[1]}$|$(n^{[1]}, n_x)$|
|$\mathbf{z}^{[1]}, \mathbf{a}^{[1]}, d\mathbf{z}^{[1]}$|$(n^{[1]}, 1)$|
|$\mathbf{W}^{[2]}$|$(1, n^{[1]})$|
|$z^{[2]}, a^{[2]}, dz^{[2]}$|$(1, 1)$|

　　验证 $d\mathbf{z}^{[1]}$ 的维度：

$$
(\mathbf{W}^{[2]})^\top \in \mathbb{R}^{n^{[1]} \times 1},\quad dz^{[2]} \in \mathbb{R}^{1 \times 1} \Rightarrow (\mathbf{W}^{[2]})^\top dz^{[2]} \in \mathbb{R}^{n^{[1]} \times 1}
$$

　　再与 $g_1'(\mathbf{z}^{[1]}) \in \mathbb{R}^{n^{[1]} \times 1}$ 逐元素相乘，结果维度一致 ✅

> **提示**：实现时务必检查所有矩阵运算的维度是否匹配，这是调试反向传播最有效的方法之一。

---

## 三、向量化实现（批量训练，m 个样本）

　　将 m 个样本堆叠成矩阵：

- $\mathbf{X} = [\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(m)}] \in \mathbb{R}^{n_x \times m}$
- $\mathbf{A}^{[1]} = [\mathbf{a}^{[1]\!(1)}, \dots, \mathbf{a}^{[1]\!(m)}] \in \mathbb{R}^{n^{[1]} \times m}$
- $\mathbf{Y} = [y^{(1)}, \dots, y^{(m)}] \in \mathbb{R}^{1 \times m}$

### 向量化反向传播公式：

1. **输出层**：

   $$
   \begin{aligned}
   d\mathbf{Z}^{[2]} &= \mathbf{A}^{[2]} - \mathbf{Y} \quad &\in \mathbb{R}^{1 \times m} \\
   d\mathbf{W}^{[2]} &= \frac{1}{m} d\mathbf{Z}^{[2]} (\mathbf{A}^{[1]})^\top \quad &\in \mathbb{R}^{1 \times n^{[1]}} \\
   d\mathbf{b}^{[2]} &= \frac{1}{m} \sum_{i=1}^m d\mathbf{Z}^{[2]}_{:,i} = \frac{1}{m} \text{np.sum}(d\mathbf{Z}^{[2]}, \text{axis}=1, \text{keepdims=True})
   \end{aligned}
   $$
2. **隐藏层**：

   $$
   \begin{aligned}
   d\mathbf{Z}^{[1]} &= (\mathbf{W}^{[2]})^\top d\mathbf{Z}^{[2]} \odot g_1'(\mathbf{Z}^{[1]}) \quad &\in \mathbb{R}^{n^{[1]} \times m} \\
   d\mathbf{W}^{[1]} &= \frac{1}{m} d\mathbf{Z}^{[1]} \mathbf{X}^\top \quad &\in \mathbb{R}^{n^{[1]} \times n_x} \\
   d\mathbf{b}^{[1]} &= \frac{1}{m} \text{np.sum}(d\mathbf{Z}^{[1]}, \text{axis}=1, \text{keepdims=True})
   \end{aligned}
   $$

> 注意：由于代价函数是平均损失 $J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}^{(i)}$，因此所有梯度都要乘以 $\frac{1}{m}$。

---

## 四、关键直觉与工程建议

1. **反向传播本质**：利用计算图 + 链式法则，从损失函数反向逐层计算梯度。
2.  **“塌缩”技巧**：实践中常将 $da \to dz$ 合并为一步（如 $dz^{[2]} = a^{[2]} - y$），避免显式计算 $da$。
3. **维度一致性原则**：任何变量与其梯度维度相同（如 $\mathbf{W}$ 与 $d\mathbf{W}$ 同维）。
4. **向量化是标准做法**：现代深度学习框架（如 TensorFlow/PyTorch）默认批量处理，上述向量化公式是底层实现基础。
5. **数学难度高但不必恐惧**：即使不完全掌握矩阵微积分，只要理解计算流程和维度规则，也能正确实现。

---

## 五、后续预告

> **权重初始化很重要！不能全零初始化**  
> 下一节将解释为何必须使用**随机初始化**（如 Xavier / He 初始化），否则神经网络无法有效训练（对称性问题）。

---

　　✅ **总结**：本节通过逻辑回归过渡到两层神经网络，清晰展示了反向传播的推导逻辑、核心公式、维度匹配和向量化实现。掌握这些内容，你就具备了手动实现神经网络训练算法的能力。

　　如需进一步推导（如 sigmoid 导数、链式法则展开），可结合矩阵微积分深入学习，但对大多数实践者而言，理解上述直觉已足够强大。
