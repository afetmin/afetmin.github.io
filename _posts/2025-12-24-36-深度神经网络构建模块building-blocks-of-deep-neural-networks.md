---
title: "36 深度神经网络构建模块（Building Blocks of Deep Neural Networks）"
date: 2025-12-24T19:06:45Z
last_modified_at: 2025-12-24T19:08:28Z
categories:
  - 神经网络与深度学习
math: true
---

## 一、核心思想

　　深度神经网络的训练依赖于两个关键过程：

- **前向传播（Forward Propagation）** ：从输入到输出逐层计算激活值。
- **反向传播（Backward Propagation）** ：从输出误差出发，逐层计算梯度，用于参数更新。

　　每一层都可视为一个“计算单元”，包含独立的前向函数和反向函数，并通过**缓存（cache）** 传递中间变量（尤其是 $z^{[l]}$）以支持反向传播。

---

## 二、单层前向传播（Forward Function for Layer $l$）

　　给定第 $l$ 层的输入激活 $a^{[l-1]}$，参数 $W^{[l]}$ 和 $b^{[l]}$，计算如下：

$$
\begin{aligned}
z^{[l]} &= W^{[l]} a^{[l-1]} + b^{[l]} \\
a^{[l]} &= g^{[l]}(z^{[l]})
\end{aligned}
$$

　　其中：

- $g^{[l]}(\cdot)$ 是第 $l$ 层的激活函数（如 ReLU、sigmoid 等）。
- **缓存内容**：为后续反向传播，需保存 $z^{[l]}$（有时也包括 $W^{[l]}, b^{[l]}$）。

> ✅ **关键点**：前向传播的输出是 $a^{[l]}$，同时缓存 $z^{[l]}$ 供反向使用。

---

## 三、单层反向传播（Backward Function for Layer $l$）

　　反向传播的目标是：已知损失函数对当前层激活的梯度 $\frac{\partial \mathcal{L}}{\partial a^{[l]}} = da^{[l]}$，计算：

- 对上一层激活的梯度：$da^{[l-1]}$
- 对本层参数的梯度：$dW^{[l]}, db^{[l]}$

### 计算流程（基于链式法则）：

1. **计算** **$dz^{[l]}$**：

   $$
   dz^{[l]} = da^{[l]} \odot g'^{[l]}(z^{[l]})
   $$

   其中 $\odot$ 表示逐元素相乘（Hadamard product），$g'$ 是激活函数的导数。
2. **计算参数梯度**：

   $$
   \begin{aligned}
   dW^{[l]} &= dz^{[l]} (a^{[l-1]})^\top \\
   db^{[l]} &= \sum_{i=1}^{m} dz^{[l](i)} \quad \text{（对 batch 中所有样本求和）}
   \end{aligned}
   $$
3. **计算传给上一层的梯度**：

   $$
   da^{[l-1]} = (W^{[l]})^\top dz^{[l]}
   $$

> ✅ **关键点**：反向传播需要缓存中的 $z^{[l]}$ 来计算 $g'(z^{[l]})$；实践中常将 $W^{[l]}, b^{[l]}$ 也存入 cache，方便实现。

---

## 四、完整网络的前向与反向流程

### 前向传播（从输入到输出）：

- 输入：$a^{[0]} = X$
- 逐层计算：

  $$
  a^{[1]} \rightarrow a^{[2]} \rightarrow \cdots \rightarrow a^{[L]} = \hat{y}
  $$
- 每层缓存 $z^{[l]}$（及可能的 $W^{[l]}, b^{[l]}$）

### 反向传播（从输出回传梯度）：

- 从损失函数开始，计算 $da^{[L]} = \frac{\partial \mathcal{L}}{\partial \hat{y}}$
- 逐层反向计算：

  $$
  da^{[L]} \rightarrow da^{[L-1]} \rightarrow \cdots \rightarrow da^{[1]}
  $$
- 同时得到每层的 $dW^{[l]}, db^{[l]}$

> ⚠️ 注意：$da^{[0]}$（即对输入 $X$ 的梯度）通常不需要，因为输入不是可训练参数。

---

## 五、参数更新（梯度下降）

　　完成一次前向+反向后，使用梯度下降更新所有参数：

$$
\begin{aligned}
W^{[l]} &:= W^{[l]} - \alpha \cdot dW^{[l]} \\
b^{[l]} &:= b^{[l]} - \alpha \cdot db^{[l]}
\end{aligned}
$$

　　其中 $\alpha$ 是学习率。

---

## 六、实现细节（编程提示）

- **Cache 的作用**：不仅存储 $z^{[l]}$，还常包含 $W^{[l]}, b^{[l]}$，以便在反向函数中直接使用，避免重复传参。
- **模块化设计**：将每层的前向/反向封装为独立函数，便于构建任意深度的网络。
- **向量化**：所有运算应对整个 mini-batch 向量化处理（$m$ 个样本并行计算）。

---

## 七、总结图示（逻辑结构）

```
前向传播：
a⁰ → [Layer 1: W¹,b¹] → a¹ → [Layer 2: W²,b²] → a² → ... → aᴸ = ŷ
          ↑ cache z¹             ↑ cache z²

反向传播：
daᴸ → [Layer L: dzᴸ, dWᴸ, dbᴸ] → daᴸ⁻¹ → ... → da¹
```

---

## 八、学习建议

- 动手实现一个两层或 L 层神经网络，亲自编写 `linear_forward`, `linear_activation_forward`, `linear_backward` 等函数。
- 理解 cache 在连接前向与反向中的桥梁作用。
- 掌握如何从数学推导过渡到代码实现（尤其注意维度匹配）。
