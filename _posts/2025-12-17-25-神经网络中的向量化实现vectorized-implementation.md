---
title: "25 神经网络中的向量化实现（Vectorized Implementation）"
date: 2025-12-17T19:45:05Z
last_modified_at: 2025-12-17T19:46:11Z
categories:
  - 神经网络与深度学习
math: true
---

### 一、背景与动机

　　在训练神经网络时，如果我们对每个训练样本单独进行前向传播（forward propagation），会使用如下形式的循环：

```python
for i in range(m):
    z1[i] = W1 @ x[i] + b1
    a1[i] = g(z1[i])
    ...
```

　　但这种逐样本处理的方式效率低下。**向量化（vectorization）**  的目标是：**一次性对所有** **$m$** **个训练样本进行前向传播计算**，从而大幅提升计算效率（尤其在 GPU 上）。

---

### 二、数据组织方式

- 单个输入样本：$\mathbf{x}^{(i)} \in \mathbb{R}^{n_x}$（列向量）
- 所有训练样本堆叠成矩阵：

  $$
  \mathbf{X} = 
  \begin{bmatrix}
  \vert & \vert & & \vert \\
  \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)} \\
  \vert & \vert & & \vert
  \end{bmatrix}
  \in \mathbb{R}^{n_x \times m}
  $$

  > 注意：**每个样本是矩阵的一列**（这是吴恩达课程的标准约定）。
  >

---

### 三、单隐藏层神经网络的前向传播（非向量化 vs 向量化）

#### 1. 非向量化（逐样本）形式：

　　对第 $i$ 个样本：

$$
\begin{aligned}
\mathbf{z}^{[1]\!(i)} &= \mathbf{W}^{[1]} \mathbf{x}^{(i)} + \mathbf{b}^{[1]} \\
\mathbf{a}^{[1]\!(i)} &= g^{[1]}(\mathbf{z}^{[1]\!(i)}) \\
z^{[2]\!(i)} &= \mathbf{w}^{[2]T} \mathbf{a}^{[1]\!(i)} + b^{[2]} \\
a^{[2]\!(i)} &= g^{[2]}(z^{[2]\!(i)})
\end{aligned}
$$

#### 2. 向量化（批量）形式：

　　将所有样本同时处理：

$$
\begin{aligned}
\mathbf{Z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{X} + \mathbf{b}^{[1]} \\
\mathbf{A}^{[1]} &= g^{[1]}(\mathbf{Z}^{[1]}) \\
\mathbf{Z}^{[2]} &= \mathbf{w}^{[2]T} \mathbf{A}^{[1]} + b^{[2]} \\
\mathbf{A}^{[2]} &= g^{[2]}(\mathbf{Z}^{[2]})
\end{aligned}
$$

　　其中：

- $\mathbf{Z}^{[1]} \in \mathbb{R}^{n^{[1]} \times m}$
- $\mathbf{A}^{[1]} \in \mathbb{R}^{n^{[1]} \times m}$
- $\mathbf{Z}^{[2]} \in \mathbb{R}^{1 \times m}$
- $\mathbf{A}^{[2]} \in \mathbb{R}^{1 \times m}$

> ✅ 关键点：**输出矩阵的每一列对应一个样本的计算结果**。

---

### 四、为什么向量化是正确的？——矩阵乘法的解释

　　考虑权重矩阵 $\mathbf{W}^{[1]} \in \mathbb{R}^{n^{[1]} \times n_x}$，输入矩阵 $\mathbf{X} = [\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(m)}]$。

　　根据矩阵乘法规则：

$$
\mathbf{W}^{[1]} \mathbf{X} = 
\left[
\mathbf{W}^{[1]} \mathbf{x}^{(1)},
\mathbf{W}^{[1]} \mathbf{x}^{(2)},
\dots,
\mathbf{W}^{[1]} \mathbf{x}^{(m)}
\right]
= 
\left[
\mathbf{z}^{[1]\!(1)},
\mathbf{z}^{[1]\!(2)},
\dots,
\mathbf{z}^{[1]\!(m)}
\right]
= \mathbf{Z}^{[1]}
$$

> 这正是我们想要的：**一次矩阵乘法自动完成所有样本的线性变换**。

#### 关于偏置项 $\mathbf{b}^{[1]}$ 的广播（Broadcasting）

- $\mathbf{b}^{[1]} \in \mathbb{R}^{n^{[1]} \times 1}$
- 在 NumPy / Python 中，`W1 @ X + b1` 会自动将 $\mathbf{b}^{[1]}$ **广播（broadcast）到每一列**，即：

  $$
  \mathbf{Z}^{[1]} = \mathbf{W}^{[1]} \mathbf{X} + \underbrace{\begin{bmatrix} \mathbf{b}^{[1]} & \mathbf{b}^{[1]} & \cdots & \mathbf{b}^{[1]} \end{bmatrix}}_{m \text{ 次重复}}
  $$

　　因此，即使包含偏置项，向量化依然成立。

---

### 五、通用模式与深层网络的扩展

　　课程指出，前向传播具有**递归对称结构**：

　　令 $\mathbf{A}^{[0]} = \mathbf{X}$（输入层激活值），则对任意第 $l$ 层：

$$
\begin{aligned}
\mathbf{Z}^{[l]} &= \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{A}^{[l]} &= g^{[l]}(\mathbf{Z}^{[l]})
\end{aligned}
$$

> 🔁 这种“线性变换 + 激活函数”的模式在每一层重复，使得**向量化可自然推广到任意深度的神经网络**。

---

### 六、小结（Recap）

|操作|非向量化（循环）|向量化（矩阵）|
| ----------| ------------------| ----------------|
|输入|$\mathbf{x}^{(i)}$|$\mathbf{X} \in \mathbb{R}^{n_x \times m}$|
|线性输出|$\mathbf{z}^{[1]\!(i)} = \mathbf{W}^{[1]} \mathbf{x}^{(i)} + \mathbf{b}^{[1]}$|$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]} \mathbf{X} + \mathbf{b}^{[1]}$|
|激活输出|$\mathbf{a}^{[1]\!(i)} = g(\mathbf{z}^{[1]\!(i)})$|$\mathbf{A}^{[1]} = g(\mathbf{Z}^{[1]})$|

　　✅ **优势**：

- 无需显式 for 循环
- 利用高度优化的 BLAS 库（如 cuBLAS）
- 代码简洁，运行速度快

---

### 七、后续内容预告

　　本节使用 **Sigmoid 激活函数**作为示例，但吴恩达指出：**Sigmoid 并非最优选择**。下一节将介绍其他更有效的激活函数，如：

- ReLU（Rectified Linear Unit）：$g(z) = \max(0, z)$
- tanh
- Leaky ReLU 等

　　这些函数能缓解梯度消失问题，加速训练。
