---
title: "17 向量化逻辑回归的梯度计算（Vectorizing Logistic Regression's Gradient Computation）"
date: 2025-12-15T20:19:28Z
last_modified_at: 2025-12-15T20:29:48Z
categories:
  - 神经网络与深度学习
math: true
---

### 一、背景回顾

　　在上一节中，我们已经学会了如何通过**向量化**一次性计算整个训练集上的预测值：

- 对于 $m$ 个训练样本，输入矩阵为：

  $$
  X = \begin{bmatrix} 
  | & | & & | \\
  x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\
  | & | & & |
  \end{bmatrix} \in \mathbb{R}^{n \times m}
  $$

  其中每个 $x^{(i)} \in \mathbb{R}^n$ 是一个样本的特征向量。
- 参数向量 $w \in \mathbb{R}^n$，偏置标量 $b \in \mathbb{R}$。
- 线性输出（logits）为：

  $$
  Z = w^\top X + b \quad \text{（在 NumPy 中自动广播）}
  $$

  注意：这里 $b$ 是标量，但 NumPy 会将其广播到长度为 $m$ 的向量。
- 激活（预测概率）为：

  $$
  A = \sigma(Z) = \frac{1}{1 + e^{-Z}} \in \mathbb{R}^{1 \times m}
  $$
- 真实标签为：

  $$
  Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix} \in \mathbb{R}^{1 \times m}
  $$

---

### 二、目标：向量化梯度计算

　　逻辑回归的损失函数是**交叉熵损失**，对单个样本 $(x^{(i)}, y^{(i)})$ 的梯度为：

$$
\begin{aligned}
dz^{(i)} &= a^{(i)} - y^{(i)} \\
dw^{(i)} &= x^{(i)} dz^{(i)} \\
db^{(i)} &= dz^{(i)}
\end{aligned}
$$

　　传统实现需要对 $i = 1$ 到 $m$ 循环累加。现在我们要**完全向量化**，去掉所有显式 for 循环。

---

### 三、关键步骤：定义向量化的中间变量

#### 1. 定义误差向量 $dZ$

　　将所有样本的 $dz^{(i)}$ 拼成一行向量：

$$
dZ = A - Y = \begin{bmatrix} a^{(1)} - y^{(1)} & a^{(2)} - y^{(2)} & \cdots & a^{(m)} - y^{(m)} \end{bmatrix} \in \mathbb{R}^{1 \times m}
$$

> ✅ **一行代码实现**（NumPy）：
>
> ```python
> dZ = A - Y
> ```

#### 2. 向量化计算 $db$

　　偏置梯度是所有 $dz^{(i)}$ 的平均值：

$$
db = \frac{1}{m} \sum_{i=1}^m dz^{(i)} = \frac{1}{m} \cdot \text{np.sum}(dZ)
$$

> ✅ **一行代码**：
>
> ```python
> db = (1 / m) * np.sum(dZ)
> ```

#### 3. 向量化计算 $dw$

　　权重梯度为：

$$
dw = \frac{1}{m} \sum_{i=1}^m x^{(i)} dz^{(i)}
$$

　　注意到 $X \in \mathbb{R}^{n \times m}$，而 $dZ^\top \in \mathbb{R}^{m \times 1}$，因此：

$$
dw = \frac{1}{m} X \cdot dZ^\top \in \mathbb{R}^{n \times 1}
$$

> ✅ **一行代码**：
>
> ```python
> dw = (1 / m) * np.dot(X, dZ.T)
> ```

> 🔍 **为什么成立？**
>
> 矩阵乘法展开：
>
> $$
> X \cdot dZ^\top = x^{(1)} dz^{(1)} + x^{(2)} dz^{(2)} + \cdots + x^{(m)} dz^{(m)}
> $$
>
> 正好是 $dw$ 的累加形式。

---

### 四、完整向量化逻辑回归的一次梯度下降迭代

```python
# 前向传播
Z = np.dot(w.T, X) + b        # shape: (1, m)
A = sigmoid(Z)                # shape: (1, m)

# 反向传播（梯度计算）
dZ = A - Y                    # shape: (1, m)
dw = (1 / m) * np.dot(X, dZ.T)  # shape: (n, 1)
db = (1 / m) * np.sum(dZ)     # scalar

# 参数更新
w = w - alpha * dw
b = b - alpha * db
```

> ✅ **无任何 for 循环！**  仅需矩阵运算即可完成整个训练集的前向+反向传播。

---

### 五、注意事项

- **外层循环无法避免**：虽然单次梯度下降可完全向量化，但若要运行 $T$ 次迭代（如 1000 次），仍需一个外层 for 循环遍历迭代次数。
- **广播机制（Broadcasting）** ：在 `Z = w.T @ X + b` 中，标量 `b` 被自动广播到 `(1, m)` 形状，这是 NumPy 的强大特性，将在下一节详细讲解。

---

### 六、总结

|步骤|非向量化（低效）|向量化（高效）|
| ----------| ------------------| ----------------|
|前向传播|循环计算每个 $a^{(i)}$|一次矩阵运算 $A = \sigma(w^\top X + b)$|
|梯度 $dZ$|循环计算 $a^{(i)} - y^{(i)}$|一行：`dZ = A - Y`|
|梯度 $dw$|循环累加 $x^{(i)} dz^{(i)}$|一行：`dw = (1/m) * X @ dZ.T`|
|梯度 $db$|循环累加 $dz^{(i)}$|一行：`db = (1/m) * sum(dZ)`|

> 💡 **核心思想**：用**矩阵运算代替 for 循环**，大幅提升计算效率，尤其在 GPU 上效果显著。
