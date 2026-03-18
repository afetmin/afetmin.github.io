---
title: "09 GRU（门控循环单元，Gated Recurrent Unit）"
date: 2026-03-08T11:11:21+08:00
last_modified_at: 2026-03-08T11:11:34+08:00
categories:
  - 循环序列模型
math: true
---

### 1. 核心动机

　　传统 RNN 在处理长序列时，难以保留早期的信息（如句子开头的主语单复数），导致梯度消失。GRU 通过引入**门控机制（Gating Mechanism）** ，让网络学会决定何时更新记忆、何时保留旧记忆，从而有效缓解这一问题。

### 2. 关键变量与符号

- $a^{\langle t \rangle}$：时间步 $t$ 的激活值（输出）。
- $c^{\langle t \rangle}$：记忆细胞（Cell）的值。在简化版 GRU 中，$c^{\langle t \rangle} = a^{\langle t \rangle}$。
- $\Gamma_u$：**更新门（Update Gate）** ，取值范围 $[0, 1]$。决定多大程度上用新候选值替换旧记忆。
- $\tilde{c}^{\langle t \rangle}$：**候选记忆值（Candidate Value）** ，表示如果完全更新时的新记忆内容。
- $\Gamma_r$：**重置门（Reset Gate）** （完整 GRU 中引入），决定过去的记忆对当前候选值计算的相关性。

---

### 3. GRU 的计算公式

#### A. 简化版 GRU (Simplified GRU)

　　吴恩达课程首先介绍了一个简化版本，主要包含**更新门**和**候选值**。

1. **计算候选记忆值 (****$\tilde{c}^{\langle t \rangle}$** **)** ：  
   使用 $\tanh$ 激活函数计算新的候选状态。

   $$
   \tilde{c}^{\langle t \rangle} = \tanh\left( W_c \left[ c^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_c \right)
   $$
2. **计算更新门 (****$\Gamma_u$** **)** ：  
   使用 $\sigma$ (sigmoid) 激活函数，输出接近 0 或 1 的值。

   $$
   \Gamma_u = \sigma\left( W_u \left[ c^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_u \right)
   $$
3. **更新记忆细胞 (****$c^{\langle t \rangle}$** **)** ：  
   这是 GRU 的核心公式。利用更新门加权组合“旧记忆”和“新候选值”。

   $$
   c^{\langle t \rangle} = \Gamma_u * \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) * c^{\langle t-1 \rangle}
   $$

   - 当 $\Gamma_u \approx 1$ 时：$c^{\langle t \rangle} \approx \tilde{c}^{\langle t \rangle}$（更新为新值）。
   - 当 $\Gamma_u \approx 0$ 时：$c^{\langle t \rangle} \approx c^{\langle t-1 \rangle}$（保持旧值，解决梯度消失）。
4. **输出激活值**：

   $$
   a^{\langle t \rangle} = c^{\langle t \rangle}
   $$

> **直观理解**：想象 $\Gamma_u$ 是一个开关。如果看到 "The cat"，门将打开（$\Gamma_u \approx 1$），记录“单数”信息；在随后的长句中，门保持关闭（$\Gamma_u \approx 0$），一直保留该信息直到句尾需要判断 "was/were"。

---

#### B. 完整版 GRU (Full GRU)

　　在实际应用和研究中，通常使用包含**重置门（Reset Gate,**  **$\Gamma_r$** **）** 的完整版本。重置门用于控制过去的状态 $c^{\langle t-1 \rangle}$ 对计算**候选值**的影响程度。

1. **计算重置门 (****$\Gamma_r$** **)** ：

   $$
   \Gamma_r = \sigma\left( W_r \left[ c^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_r \right)
   $$
2. **计算候选记忆值 (****$\tilde{c}^{\langle t \rangle}$** **)** ：  
   注意此处 $c^{\langle t-1 \rangle}$ 被乘以了 $\Gamma_r$（元素对应乘积）。如果 $\Gamma_r \approx 0$，则忽略过去的记忆，仿佛这是序列的第一个时间步。

   $$
   \tilde{c}^{\langle t \rangle} = \tanh\left( W_c \left[ \Gamma_r * c^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_c \right)
   $$
3. **计算更新门 (****$\Gamma_u$** **)** ：  
   （同简化版）

   $$
   \Gamma_u = \sigma\left( W_u \left[ c^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_u \right)
   $$
4. **更新记忆细胞 (****$c^{\langle t \rangle}$** **)** ：  
   （同简化版）

   $$
   c^{\langle t \rangle} = \Gamma_u * \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) * c^{\langle t-1 \rangle}
   $$
5. **输出激活值**：

   $$
   a^{\langle t \rangle} = c^{\langle t \rangle}
   $$

---

### 4. 为什么 GRU 能缓解梯度消失？

- **恒等映射路径**：当更新门 $\Gamma_u \approx 0$ 时，公式变为 $c^{\langle t \rangle} \approx c^{\langle t-1 \rangle}$。这意味着梯度可以直接通过时间步反向传播而不发生衰减（类似于高速公路）。
- **长期依赖**：这使得网络能够记住很久之前的信息（如句子开头的 "cat"），即使中间隔了很多单词，也能在句尾正确预测 "was"。

### 5. 实现细节

- **向量运算**：上述公式中的 $c, \tilde{c}, \Gamma_u, \Gamma_r$ 都是向量（例如 100 维）。乘法 $*$ 表示**元素对应乘积（Element-wise Product）** 。
- **灵活性**：网络可以学习在不同的维度上更新不同的信息。例如，某些维度记忆“单复数”，其他维度记忆“食物话题”。
- **符号差异**：学术界符号不统一，有时用 $h^{\langle t \rangle}$ 代替 $a^{\langle t \rangle}$ 或 $c^{\langle t \rangle}$，用 $z$ 代替 $\Gamma_u$，用 $r$ 代替 $\Gamma_r$。但核心逻辑一致。

### 6. 总结

　　GRU 是 LSTM 的一种变体，结构更简单（参数更少），但在许多任务上表现同样出色。它通过**更新门**决定保留多少旧记忆，通过**重置门**（完整版）决定忽略多少旧记忆来计算新候选值，从而高效地捕捉长距离依赖关系。
