---
title: "23 学习率衰减（Learning Rate Decay）"
date: 2026-01-05T19:59:50Z
last_modified_at: 2026-01-05T20:01:43Z
categories:
  - 改进深度神经网络
math: true
---

### 一、为什么要使用学习率衰减？

　　在使用 **Mini-batch 梯度下降法** 时：

- Mini-batch 的样本数量较小（如 64 或 128），导致梯度估计存在 **噪声（noise）** ；
- 若使用 **固定学习率** $\alpha$，优化过程会在最优值附近 **持续震荡**，无法精确收敛；
- 如果 **随训练进程逐渐减小学习率**，初期快速下降，后期精细调整，可使参数更稳定地收敛到最小值附近。

> ✅ **核心思想**：  
> 初期用较大的学习率快速逼近最优区域，后期用较小的学习率精细微调，减少震荡。

---

### 二、什么是“一代”（Epoch）？

- **1 个 epoch** = 完整遍历一次整个训练集；
- 若将训练集划分为多个 mini-batch，则一个 epoch 包含若干次参数更新；
- 学习率衰减通常以 **epoch 数**（$\text{epoch\_num}$）为自变量进行调整。

---

### 三、常用的学习率衰减策略（公式汇总）

#### 1. **分式衰减（Inverse Time Decay）**

$$
\alpha = \frac{1}{1 + \text{decay\_rate} \cdot \text{epoch\_num}} \cdot \alpha_0
$$

- $\alpha_0$：初始学习率；
- $\text{decay\_rate}$：衰减率（超参数）；
- 随 epoch 增加，学习率单调递减。

> 📌 示例：若 $\alpha_0 = 0.2$，$\text{decay\_rate} = 1$，则：
>
> - epoch 1: $\alpha = \frac{1}{1+1} \cdot 0.2 = 0.1$
> - epoch 2: $\alpha = \frac{1}{1+2} \cdot 0.2 \approx 0.067$

---

#### 2. **指数衰减（Exponential Decay）**

$$
\alpha = \alpha_0 \cdot \beta^{\text{epoch\_num}}
$$

- 其中 $0 < \beta < 1$（如 $\beta = 0.95$）；
- 学习率呈指数级下降，初期下降快，后期趋缓。

---

#### 3. **平方根衰减（Square Root Decay）**

　　有两种常见形式：

- 基于 epoch：

  $$
  \alpha = \frac{k}{\sqrt{\text{epoch\_num}}} \cdot \alpha_0
  $$
- 基于迭代步数 $t$（即 mini-batch 的更新次数）：

  $$
  \alpha = \frac{k}{\sqrt{t}} \cdot \alpha_0
  $$
- $k$ 为常数超参数；
- 衰减速度比指数衰减慢，适用于某些理论保证的优化场景（如凸优化）。

---

#### 4. **阶梯式衰减（Step Decay / Discrete Staircase）**

- 在预设的 epoch 节点（如每 10 个 epoch）将学习率 **乘以一个因子**（如 0.5）；
- 例如：

  $$
  \alpha = 
  \begin{cases}
    \alpha_0, & \text{if epoch} < 10 \\
    0.5 \alpha_0, & \text{if } 10 \leq \text{epoch} < 20 \\
    0.25 \alpha_0, & \text{if } 20 \leq \text{epoch} < 30 \\
    \dots
  \end{cases}
  $$
- 实践中非常常用，尤其在计算机视觉任务中。

---

#### 5. **手动衰减（Manual Decay）**

- 在长时间训练中（如数天），人工观察损失曲线，**手动调低学习率**；
- 适用于单模型精细调优，但**不可扩展**，不适合自动化或大规模实验。

---

### 四、实践建议

- 学习率衰减 **不是首要尝试的技巧**；
- 优先 **固定学习率 + 良好调参**（如通过验证集选择 $\alpha_0$）往往已有显著效果；
- 衰减策略和超参数（$\alpha_0$, decay_rate, $\beta$, $k$ 等）需通过 **超参数调优**（如网格搜索、随机搜索、贝叶斯优化）确定；
- 下一节将介绍 **系统化的超参数调优方法**。

---

### 五、关键总结

|要点|说明|
| ------| ----------------------------------------------------|
|**动机**|减少 mini-batch 噪声引起的震荡，提升收敛稳定性|
|**核心机制**|初期大步快跑，后期小步微调|
|**常用策略**|分式衰减、指数衰减、平方根衰减、阶梯衰减、手动调整|
|**超参数**|$\alpha_0$（初始学习率）、decay_rate、$\beta$、$k$ 等|
|**工程建议**|先固定学习率调好基础模型，再考虑引入衰减|
