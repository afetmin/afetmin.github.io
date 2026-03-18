---
title: "12 逻辑回归的梯度下降（Gradient Descent for Logistic Regression）"
date: 2025-12-08T07:57:27Z
last_modified_at: 2025-12-08T07:58:25Z
categories:
  - 神经网络与深度学习
math: true
---

### 1. 逻辑回归模型回顾

　　对于一个样本 $(x, y)$，其中：

- 输入特征向量：$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \in \mathbb{R}^2$（为简化，假设只有两个特征）
- 参数：权重 $\mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}$，偏置 $b \in \mathbb{R}$
- 真实标签：$y \in \{0, 1\}$

　　逻辑回归的前向传播过程如下：

$$
\begin{aligned}
z &= \mathbf{w}^\top \mathbf{x} + b = w_1 x_1 + w_2 x_2 + b \\
\hat{y} = a &= \sigma(z) = \frac{1}{1 + e^{-z}} \\
\mathcal{L}(a, y) &= -y \log a - (1 - y) \log(1 - a)
\end{aligned}
$$

　　其中：

- $a$ 是模型预测的概率输出（即 $\hat{y}$）
- $\mathcal{L}(a, y)$ 是**单个样本的损失函数（Loss）** ，也称为**交叉熵损失（Cross-Entropy Loss）**

---

### 2. 反向传播：计算梯度（使用计算图）

　　目标：通过反向传播计算损失 $\mathcal{L}$ 对参数 $\mathbf{w}$ 和 $b$ 的偏导数，用于梯度下降更新。

#### 步骤 1：计算 $\frac{\partial \mathcal{L}}{\partial a}$

$$
\frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1 - y}{1 - a}
$$

　　记作：`da = -y/a + (1 - y)/(1 - a)`

---

#### 步骤 2：计算 $\frac{\partial \mathcal{L}}{\partial z}$

　　利用链式法则：

$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z}
$$

　　由于 $a = \sigma(z)$，其导数为：

$$
\frac{\partial a}{\partial z} = a(1 - a)
$$

　　因此：

$$
\frac{\partial \mathcal{L}}{\partial z} = \left( -\frac{y}{a} + \frac{1 - y}{1 - a} \right) \cdot a(1 - a) = a - y
$$

　　记作：`dz = a - y`

> ✅ 这是一个非常简洁且重要的结果：**dz = a - y**

---

#### 步骤 3：计算参数梯度

　　根据 $z = w_1 x_1 + w_2 x_2 + b$，可得：

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial w_1} &= \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w_1} = dz \cdot x_1 \\
\frac{\partial \mathcal{L}}{\partial w_2} &= dz \cdot x_2 \\
\frac{\partial \mathcal{L}}{\partial b} &= dz \cdot 1 = dz
\end{aligned}
$$

　　记作：

- `dw1 = x1 * dz`
- `dw2 = x2 * dz`
- `db = dz`

---

### 3. 梯度下降更新规则（单个样本）

　　设学习率为 $\alpha$，则参数更新为：

$$
\begin{aligned}
w_1 &:= w_1 - \alpha \cdot dw_1 = w_1 - \alpha \cdot x_1 (a - y) \\
w_2 &:= w_2 - \alpha \cdot x_2 (a - y) \\
b &:= b - \alpha \cdot (a - y)
\end{aligned}
$$

　　这是一次**基于单个训练样本**的梯度下降步骤。

---

### 4. 扩展到整个训练集（预告）

　　在实际训练中，我们有 $m$ 个样本 $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$。  
此时需计算**总损失（Cost Function）** ：

$$
J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})
$$

　　对应的梯度为各样本梯度的平均值：

$$
\begin{aligned}
d\mathbf{w} &= \frac{1}{m} \sum_{i=1}^m \mathbf{x}^{(i)} (a^{(i)} - y^{(i)}) \\
db &= \frac{1}{m} \sum_{i=1}^m (a^{(i)} - y^{(i)})
\end{aligned}
$$

　　然后进行批量更新：

$$
\begin{aligned}
\mathbf{w} &:= \mathbf{w} - \alpha \cdot d\mathbf{w} \\
b &:= b - \alpha \cdot db
\end{aligned}
$$

> 🔜 这部分内容将在下一节课中详细讲解（向量化与批量梯度下降）。

---

### 5. 关键公式总结（必须记住）

|量|表达式|
| --------------| --------|
|预测值|$a = \sigma(\mathbf{w}^\top \mathbf{x} + b)$|
|单样本损失|$\mathcal{L}(a, y) = -y \log a - (1 - y)\log(1 - a)$|
|关键中间梯度|$dz = a - y$|
|权重梯度|$d\mathbf{w} = \mathbf{x} \cdot dz$|
|偏置梯度|$db = dz$|

---

### 6. 为什么用计算图？

　　虽然对逻辑回归而言，直接求导更简单，但吴恩达使用**计算图（Computation Graph）**  是为了：

- 建立反向传播（Backpropagation）的直观理解
- 为后续**深度神经网络**的梯度计算打下基础
- 强调**链式法则（Chain Rule）**  在自动微分中的核心作用

---

　　✅ **学习建议**：

- 动手推导一次 $\frac{\partial \mathcal{L}}{\partial z} = a - y$，加深理解
- 尝试用 Python 实现单样本和批量的逻辑回归梯度下降
- 注意区分 **Loss（单样本）**  和 **Cost（全体样本平均）**
