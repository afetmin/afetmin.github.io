---
title: "13 在 m 个样本上执行梯度下降（Gradient Descent on m Examples）"
date: 2025-12-08T08:14:43Z
last_modified_at: 2025-12-08T08:15:38Z
categories:
  - 神经网络与深度学习
math: true
---

## 🎯 课程目标

　　本节课程的核心目标是：

> **如何将单样本的梯度下降扩展到整个训练集（m 个样本）上，并为后续向量化（Vectorization）打下基础。**

---

## 📌 1. 逻辑回归回顾

　　我们考虑一个二分类问题，使用 **逻辑回归（Logistic Regression）**  模型：

- 输入特征：$\mathbf{x}^{(i)} \in \mathbb{R}^n$（第 $i$ 个样本）
- 真实标签：$y^{(i)} \in \{0, 1\}$
- 模型参数：权重 $\mathbf{w} \in \mathbb{R}^n$，偏置 $b \in \mathbb{R}$

　　模型预测为：

$$
z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b,\quad a^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}
$$

---

## 📌 2. 成本函数（Cost Function）

　　对 **m 个训练样本**，整体成本函数 $J(\mathbf{w}, b)$ 定义为 **平均交叉熵损失**：

$$
J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log a^{(i)} + (1 - y^{(i)}) \log (1 - a^{(i)}) \right]
$$

> 注意：这是 **单个样本损失** **$\mathcal{L}^{(i)}$** **的平均值**。

---

## 📌 3. 单样本梯度（回顾）

　　对于单个样本 $(\mathbf{x}^{(i)}, y^{(i)})$，我们已知其梯度为：

$$
dz^{(i)} = a^{(i)} - y^{(i)}
$$

$$
d\mathbf{w}^{(i)} = \mathbf{x}^{(i)} dz^{(i)}, \quad db^{(i)} = dz^{(i)}
$$

　　其中 $d\mathbf{w}^{(i)} \in \mathbb{R}^n$ 是该样本对权重的梯度贡献。

---

## 📌 4. 整体梯度（m 个样本）

　　由于成本函数是 **平均损失**，其梯度也是 **各样本梯度的平均**：

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \sum_{i=1}^{m} d\mathbf{w}^{(i)} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{x}^{(i)} (a^{(i)} - y^{(i)})
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (a^{(i)} - y^{(i)})
$$

> 这就是我们要用于梯度下降的 **总梯度**。

---

## 📌 5. 梯度下降算法（朴素实现，含 for 循环）

　　以下是 **一次梯度下降迭代** 的完整步骤（假设特征数 $n = 2$，便于理解）：

### 初始化

```text
J = 0
dw1 = 0, dw2 = 0, db = 0
```

### 遍历所有 m 个样本（第一个 for 循环）

　　对每个 $i = 1$ 到 $m$：

1. 计算线性输出：

    $$
    z^{(i)} = w_1 x_1^{(i)} + w_2 x_2^{(i)} + b
    $$
2. 计算激活（预测）：

    $$
    a^{(i)} = \sigma(z^{(i)})
    $$
3. 累加成本函数：

    $$
    J \mathrel{+}= -\left[ y^{(i)} \log a^{(i)} + (1 - y^{(i)}) \log (1 - a^{(i)}) \right]
    $$
4. 计算局部梯度：

    $$
    dz^{(i)} = a^{(i)} - y^{(i)}
    $$
5. 累加参数梯度：

    $$
    dw_1 \mathrel{+}= x_1^{(i)} \cdot dz^{(i)} \\
    dw_2 \mathrel{+}= x_2^{(i)} \cdot dz^{(i)} \\
    db \mathrel{+}= dz^{(i)}
    $$

> 如果有 $n$ 个特征，则需对 $j = 1$ 到 $n$ 执行 `dw_j += x_j^{(i)} * dz^{(i)}` —— 这就是 **第二个 for 循环**。

### 计算平均梯度

$$
dw_1 = \frac{dw_1}{m},\quad dw_2 = \frac{dw_2}{m},\quad db = \frac{db}{m},\quad J = \frac{J}{m}
$$

### 更新参数（学习率 $\alpha$）

$$
w_1 := w_1 - \alpha \cdot dw_1 \\
w_2 := w_2 - \alpha \cdot dw_2 \\
b := b - \alpha \cdot db
$$

> 此过程需重复多次（多个 epoch）以完成训练。

---

## ⚠️ 6. 问题：双重 for 循环效率低

　　上述实现存在两个显式 for 循环：

1. **外层循环**：遍历 $m$ 个样本
2. **内层循环**：遍历 $n$ 个特征（如 `dw1, dw2, ..., dwn`）

　　在深度学习中，当 $m$ 和 $n$ 都很大时（如百万级样本、千维特征），这种写法 **速度极慢**。

---

## ✅ 7. 解决方案：向量化（Vectorization）

> **向量化 = 用矩阵/向量运算代替 for 循环**

　　核心思想：

- 将所有样本堆叠成矩阵：$\mathbf{X} \in \mathbb{R}^{n \times m}$
- 所有标签组成向量：$\mathbf{Y} \in \mathbb{R}^{1 \times m}$
- 一次性计算所有 $z^{(i)}, a^{(i)}, dz^{(i)}$ 等

　　例如：

$$
\mathbf{Z} = \mathbf{w}^\top \mathbf{X} + b \quad (\text{形状: } 1 \times m) \\
\mathbf{A} = \sigma(\mathbf{Z}) \\
d\mathbf{Z} = \mathbf{A} - \mathbf{Y} \\
d\mathbf{w} = \frac{1}{m} \mathbf{X} d\mathbf{Z}^\top \quad (\text{形状: } n \times 1) \\
db = \frac{1}{m} \sum_{i=1}^{m} dZ^{(i)}
$$

> 向量化后，**无需任何 for 循环**，且可利用 GPU 并行加速。

---

## 🔚 总结

|内容|说明|
| ------| ----------------------------------|
|**成本函数**|$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log a^{(i)} + (1-y^{(i)}) \log(1-a^{(i)})]$|
|**单样本梯度**|$dz^{(i)} = a^{(i)} - y^{(i)}$, $d\mathbf{w}^{(i)} = \mathbf{x}^{(i)} dz^{(i)}$|
|**整体梯度**|对所有样本求平均|
|**朴素实现**|需要两个 for 循环（样本 + 特征）|
|**高效实现**|使用向量化，完全消除 for 循环|
|**下一步**|学习 **向量化技术**，实现无循环的高效梯度下降|
