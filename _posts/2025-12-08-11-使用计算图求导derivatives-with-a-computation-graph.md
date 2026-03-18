---
title: "11 使用计算图求导（Derivatives with a Computation Graph）"
date: 2025-12-08T07:48:53Z
last_modified_at: 2025-12-08T07:49:35Z
categories:
  - 神经网络与深度学习
math: true
---

## 🧠 课程核心思想

　　在深度学习中，**反向传播（Backpropagation）**  是通过**计算图（Computation Graph）**  高效计算损失函数对各参数偏导数的核心机制。  
本节通过一个具体例子，展示了如何利用**链式法则（Chain Rule）**  沿着计算图**从右向左（backward pass）**  逐层计算梯度。

---

## 📐 示例函数与计算图结构

　　考虑如下函数：

$$
J = 3v, \quad \text{其中} \quad v = a + u, \quad u = b \cdot c
$$

　　给定初始值：

- $a = 5$
- $b = 3$
- $c = 2$
- $u = b \cdot c = 6$
- $v = a + u = 11$
- $J = 3v = 33$

　　计算图结构（从左到右为前向传播）：

```
b ──┐
    X → u ──┐
c ──┘       +
            → v → ×3 → J
a ──────────┘
```

---

## 🔁 反向传播：从右向左计算导数

　　目标：计算 $J$ 对各个中间变量和输入变量的偏导数：

- $\frac{\partial J}{\partial v}$
- $\frac{\partial J}{\partial a}$
- $\frac{\partial J}{\partial u}$
- $\frac{\partial J}{\partial b}$
- $\frac{\partial J}{\partial c}$

---

### 1. 计算 $\frac{\partial J}{\partial v}$

$$
J = 3v \quad \Rightarrow \quad \frac{\partial J}{\partial v} = 3
$$

　　记作代码中的变量：`dv = 3`

---

### 2. 计算 $\frac{\partial J}{\partial a}$

　　由于 $v = a + u$，所以：

$$
\frac{\partial v}{\partial a} = 1
$$

　　应用链式法则：

$$
\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v} \cdot \frac{\partial v}{\partial a} = 3 \cdot 1 = 3
$$

　　记作：`da = 3`

---

### 3. 计算 $\frac{\partial J}{\partial u}$

　　同理，$v = a + u \Rightarrow \frac{\partial v}{\partial u} = 1$

$$
\frac{\partial J}{\partial u} = \frac{\partial J}{\partial v} \cdot \frac{\partial v}{\partial u} = 3 \cdot 1 = 3
$$

　　记作：`du = 3`

---

### 4. 计算 $\frac{\partial J}{\partial b}$

　　由于 $u = b \cdot c$，所以：

$$
\frac{\partial u}{\partial b} = c = 2
$$

　　再用链式法则：

$$
\frac{\partial J}{\partial b} = \frac{\partial J}{\partial u} \cdot \frac{\partial u}{\partial b} = 3 \cdot 2 = 6
$$

　　记作：`db = 6`

---

### 5. 计算 $\frac{\partial J}{\partial c}$

　　同理：

$$
\frac{\partial u}{\partial c} = b = 3
$$

$$
\frac{\partial J}{\partial c} = \frac{\partial J}{\partial u} \cdot \frac{\partial u}{\partial c} = 3 \cdot 3 = 9
$$

　　记作：`dc = 9`

---

## 💻 编程中的简化记号

　　在实际代码（如 Python）中，为了简洁，通常**省略对最终输出** **$J$** **的显式书写**，而直接用变量名表示其对某中间量的偏导：

|数学符号|代码变量名|含义|
| ----------| ------------| -------------|
|$\frac{\partial J}{\partial v}$|`dv`|$J$ 对 $v$ 的偏导|
|$\frac{\partial J}{\partial a}$|`da`|$J$ 对 $a$ 的偏导|
|$\frac{\partial J}{\partial u}$|`du`|$J$ 对 $u$ 的偏导|
|$\frac{\partial J}{\partial b}$|`db`|$J$ 对 $b$ 的偏导|
|$\frac{\partial J}{\partial c}$|`dc`|$J$ 对 $c$ 的偏导|

> ✅ 这种记法假设所有导数都是相对于**同一个最终输出** **$J$**（即损失函数），因此无需重复写 `dJ_d...`。

---

## 🔗 链式法则（Chain Rule）回顾

　　若变量依赖关系为：

$$
J \leftarrow v \leftarrow u \leftarrow b
$$

　　则：

$$
\frac{\partial J}{\partial b} = \frac{\partial J}{\partial v} \cdot \frac{\partial v}{\partial u} \cdot \frac{\partial u}{\partial b}
$$

　　但在本例中，因 $v$ 直接依赖于 $u$，而 $J$ 直接依赖于 $v$，可分步计算：

$$
\frac{\partial J}{\partial b} = \underbrace{\frac{\partial J}{\partial u}}_{=3} \cdot \underbrace{\frac{\partial u}{\partial b}}_{=2} = 6
$$

---

## 🔄 前向 vs 反向传播

|步骤|方向|目的|计算内容|
| ---------------------------| ----------| -----------| ----------|
|前向传播（Forward Pass）|左 → 右|计算输出 $J$|$a, b, c \rightarrow u \rightarrow v \rightarrow J$|
|反向传播（Backward Pass）|右 ← 左|计算梯度|$\frac{\partial J}{\partial v} \rightarrow \frac{\partial J}{\partial a}, \frac{\partial J}{\partial u} \rightarrow \frac{\partial J}{\partial b}, \frac{\partial J}{\partial c}$|

> ⚡ **效率关键**：反向传播避免重复计算，利用已算出的上游梯度（如 `dv`）高效推导下游梯度（如 `da`, `du`）。

---

## ✅ 关键结论

1. **计算图是理解和实现自动微分的基础工具**。
2. **反向传播的本质是链式法则的系统化应用**。
3. **编程中采用简写记号（如** **`dv`** **,**  **`da`** **）极大提升代码可读性与简洁性**。
4. **所有梯度最终服务于优化目标（如最小化损失函数** **$J$** **）** 。

---

## 🔜 下一步预告

　　正如视频末尾所提，下一节将把这一机制应用到**逻辑回归（Logistic Regression）**  中，展示如何在真实模型中计算梯度并更新参数。
