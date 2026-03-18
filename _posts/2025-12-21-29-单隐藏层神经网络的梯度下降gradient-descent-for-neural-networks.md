---
title: "29 单隐藏层神经网络的梯度下降（Gradient Descent for Neural Networks）"
date: 2025-12-21T19:51:05Z
last_modified_at: 2025-12-21T19:51:21Z
categories:
  - 神经网络与深度学习
math: true
---

## 🧠 课程主题：

　　本课程讲解如何对具有**一个隐藏层**的神经网络实现**梯度下降**（Gradient Descent），核心在于**前向传播**（Forward Propagation）和**反向传播**（Backpropagation）的计算流程。

---

## 🔢 网络结构与参数维度

　　假设：

- 输入特征维度：$n^{(0)} = n_x$
- 隐藏层单元数：$n^{(1)} = n_1$
- 输出层单元数（二分类）：$n^{(2)} = n_2 = 1$

　　则参数维度如下：

|参数|维度|
| ------| ------|
|$W^{[1]}$|$(n_1, n_x)$|
|$b^{[1]}$|$(n_1, 1)$|
|$W^{[2]}$|$(n_2, n_1) = (1, n_1)$|
|$b^{[2]}$|$(n_2, 1) = (1, 1)$|

> 注：上标 $[l]$ 表示第 $l$ 层，下标表示维度。

---

## 📉 损失函数（Cost Function）

　　对于**二分类任务**，使用与逻辑回归相同的损失函数：

$$
\mathcal{L}^{(i)} = -y^{(i)} \log(\hat{y}^{(i)}) - (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})
$$

　　整体成本函数（对 $m$ 个样本取平均）：

$$
J(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}^{(i)}
$$

　　其中 $\hat{y}^{(i)} = a^{[2]\!(i)}$ 是神经网络的输出。

---

## ➡️ 前向传播（Forward Propagation）

　　对整个训练集进行**向量化**前向传播（$X \in \mathbb{R}^{n_x \times m}$）：

$$
\begin{aligned}
Z^{[1]} &= W^{[1]} X + b^{[1]} \quad &\text{维度: } (n_1, m) \\
A^{[1]} &= g^{[1]}(Z^{[1]}) \quad &\text{激活函数作用于每个元素} \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \quad &\text{维度: } (1, m) \\
A^{[2]} &= g^{[2]}(Z^{[2]}) = \sigma(Z^{[2]}) \quad &\text{sigmoid 激活（二分类）}
\end{aligned}
$$

　　其中：

- $g^{[1]}$：隐藏层激活函数（如 tanh、ReLU）
- $g^{[2]} = \sigma(z) = \frac{1}{1 + e^{-z}}$

---

## ⬅️ 反向传播（Backpropagation）

　　目标：计算损失函数 $J$ 对各参数的偏导数，用于梯度下降更新。

### 第一步：输出层误差（Layer 2）

$$
dZ^{[2]} = A^{[2]} - Y \quad \text{（Y 是真实标签矩阵，维度 }(1, m)\text{）}
$$

> 这个简洁形式仅在使用 sigmoid 激活 + 交叉熵损失时成立。

　　接着计算输出层参数梯度：

$$
\begin{aligned}
dW^{[2]} &= \frac{1}{m} dZ^{[2]} (A^{[1]})^T \quad &\text{维度: } (1, n_1) \\
db^{[2]} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[2](i)} = \frac{1}{m} \text{np.sum}(dZ^{[2]}, \text{axis}=1, \text{keepdims}=\text{True}) \quad &\text{维度: } (1, 1)
\end{aligned}
$$

### 第二步：隐藏层误差（Layer 1）

　　先计算隐藏层的“误差信号”：

$$
dZ^{[1]} = (W^{[2]})^T dZ^{[2]} \odot g^{[1]\prime}(Z^{[1]})
$$

　　其中：

- $\odot$ 表示**逐元素相乘**（Hadamard product）
- $g^{[1]\prime}(Z^{[1]})$ 是隐藏层激活函数的导数（例如 tanh 的导数为 $1 - \tanh^2(z)$）

　　然后计算隐藏层参数梯度：

$$
\begin{aligned}
dW^{[1]} &= \frac{1}{m} dZ^{[1]} X^T \quad &\text{维度: } (n_1, n_x) \\
db^{[1]} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[1](i)} = \frac{1}{m} \text{np.sum}(dZ^{[1]}, \text{axis}=1, \text{keepdims}=\text{True}) \quad &\text{维度: } (n_1, 1)
\end{aligned}
$$

> ✅ **重要提示**：使用 `keepdims=True` 可避免 NumPy 生成秩为 1 的数组（如 `(n,)`），而保持为列向量 `(n, 1)`，防止后续矩阵运算出错。

---

## 🔁 梯度下降更新规则

　　使用学习率 $\alpha$ 更新所有参数：

$$
\begin{aligned}
W^{[1]} &:= W^{[1]} - \alpha \cdot dW^{[1]} \\
b^{[1]} &:= b^{[1]} - \alpha \cdot db^{[1]} \\
W^{[2]} &:= W^{[2]} - \alpha \cdot dW^{[2]} \\
b^{[2]} &:= b^{[2]} - \alpha \cdot db^{[2]}
\end{aligned}
$$

　　重复上述过程（前向 → 反向 → 更新）若干轮（epochs），直到损失收敛。

---

## ⚠️ 关键注意事项

1. **参数初始化不能全为零**  
   若 $W^{[1]} = 0$，所有隐藏单元将对称更新，无法学习不同特征。应使用**随机小值初始化**（如 `np.random.randn * 0.01`）。
2. **向量化是高效训练的关键**  
   所有计算均对整个批量（batch）向量化处理，避免 for 循环。
3. **可不深究微积分也能实现**  
   吴恩达强调：即使不完全理解反向传播的链式法则推导，只要正确实现上述公式，就能成功训练网络。但理解原理有助于调试和创新。

---

## 📌 总结流程（算法伪代码）

```python
# 初始化参数（随机）
W1 = np.random.randn(n1, nx) * 0.01
b1 = np.zeros((n1, 1))
W2 = np.random.randn(1, n1) * 0.01
b2 = np.zeros((1, 1))

for epoch in range(num_epochs):
    # 前向传播
    Z1 = W1 @ X + b1
    A1 = g1(Z1)          # 如 tanh
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)     # 输出预测

    # 计算损失
    cost = compute_cost(A2, Y)

    # 反向传播
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = (W2.T @ dZ2) * g1_prime(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    # 梯度下降更新
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
```

---

## 📘 后续建议

- 下一节（可选）将深入讲解**反向传播公式的微积分推导**（链式法则）。
- 掌握此基础后，可扩展到**多隐藏层**、**多分类**（Softmax）、**不同激活函数**等更复杂场景。
