---
title: "28 神经网络中激活函数的导数（Derivatives of Activation Functions）"
date: 2025-12-21T19:43:20Z
last_modified_at: 2025-12-21T19:43:34Z
categories:
  - 神经网络与深度学习
math: true
---

## 🧠 课程主题：

　　在实现**反向传播（Backpropagation）** 时，我们需要计算激活函数对输入 $z$ 的导数（即斜率）。本课程介绍了三种常用激活函数（Sigmoid、Tanh、ReLU 及 Leaky ReLU）的导数形式及其在实际编程中的高效计算方式。

---

### 1. Sigmoid 激活函数

#### 定义：

$$
a = g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

#### 导数（Derivative）：

$$
g'(z) = \frac{d}{dz} g(z) = g(z)(1 - g(z)) = a(1 - a)
$$

#### 特性验证：

- 当 $z \to +\infty$，$g(z) \to 1$，导数 $\to 0$
- 当 $z \to -\infty$，$g(z) \to 0$，导数 $\to 0$
- 当 $z = 0$，$g(z) = 0.5$，导数 $= 0.25$

> ✅ **实用技巧**：若前向传播中已计算出 $a = g(z)$，则可直接用 $a(1 - a)$ 快速计算导数，无需重复计算指数函数。

---

### 2. Tanh（双曲正切）激活函数

#### 定义：

$$
a = g(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

　　取值范围：$(-1, 1)$

#### 导数：

$$
g'(z) = \frac{d}{dz} \tanh(z) = 1 - \tanh^2(z) = 1 - a^2
$$

#### 特性验证：

- 当 $z \to +\infty$，$a \to 1$，导数 $\to 0$
- 当 $z \to -\infty$，$a \to -1$，导数 $\to 0$
- 当 $z = 0$，$a = 0$，导数 $= 1$

> ✅ **实用技巧**：同样，若已知 $a = \tanh(z)$，导数可直接用 $1 - a^2$ 计算，效率高。

---

### 3. ReLU（Rectified Linear Unit）激活函数

#### 定义：

$$
g(z) = \max(0, z)
$$

#### 导数（分段定义）：

$$
g'(z) =
\begin{cases}
0, & z < 0 \\
1, & z > 0 \\
\text{未定义（或任意取 0 或 1）}, & z = 0
\end{cases}
$$

> 💡 **工程实践**：在代码实现中，通常将 $z = 0$ 时的导数设为 1（或 0），因为浮点数精确等于 0 的概率极低，不影响训练效果。  
> 数学上，此时使用的是**次梯度（sub-gradient）** ，但梯度下降仍能正常工作。

---

### 4. Leaky ReLU 激活函数

#### 定义（以斜率 0.01 为例）：

$$
g(z) = \max(0.01z, z)
$$

#### 导数：

$$
g'(z) =
\begin{cases}
0.01, & z < 0 \\
1, & z > 0 \\
\text{未定义（实践中可任选 0.01 或 1）}, & z = 0
\end{cases}
$$

> ✅ Leaky ReLU 解决了 ReLU 的“神经元死亡”问题，在负区间保留微小梯度。

---

## 🔑 总结要点

|激活函数|表达式 $a = g(z)$|导数 $g'(z)$|实现建议|
| ------------| ---------| -------| --------------------------------|
|Sigmoid|$\frac{1}{1 + e^{-z}}$|$a(1 - a)$|避免用于深层网络（梯度消失）|
|Tanh|$\tanh(z)$|$1 - a^2$|输出均值为 0，优于 Sigmoid|
|ReLU|$\max(0, z)$|$\mathbb{I}_{z > 0}$|默认首选，计算快，缓解梯度消失|
|Leaky ReLU|$\max(\alpha z, z)$（如 $\alpha=0.01$）|$\begin{cases} \alpha & z<0 \\ 1 & z>0 \end{cases}$|改进版 ReLU，避免神经元死亡|

> 其中 $\mathbb{I}_{\text{condition}}$ 是指示函数（满足条件为 1，否则为 0）。

---

## 📌 为什么导数如此重要？

　　在反向传播中，每一层的误差信号需通过链式法则传递：

$$
\frac{\partial \mathcal{L}}{\partial z^{[l]}} = \frac{\partial \mathcal{L}}{\partial a^{[l]}} \cdot g'(z^{[l]})
$$

　　其中 $g'(z^{[l]})$ 就是激活函数的导数。若导数接近 0（如 Sigmoid/Tanh 在饱和区），会导致**梯度消失**，使深层网络难以训练。

　　因此，**选择合适的激活函数及其导数形式，是构建高效神经网络的关键一步**。
