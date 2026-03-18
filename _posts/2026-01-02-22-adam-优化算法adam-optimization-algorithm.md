---
title: "22 Adam 优化算法（Adam Optimization Algorithm）"
date: 2026-01-02T11:16:03Z
last_modified_at: 2026-01-04T09:59:20Z
categories:
  - 改进深度神经网络
math: true
---

### ✅ 简介

　　在深度学习的发展历程中，许多研究人员提出了新的优化算法，但大多数仅在特定问题上表现良好，缺乏泛化能力。因此，社区对新优化算法持谨慎态度。

　　而 **Adam（Adaptive Moment Estimation）**  是少数被广泛验证、适用于多种神经网络架构的有效优化算法之一。它结合了：

- **动量法（Momentum）**
- **RMSProp**

　　从而实现了快速且稳定的训练效果。

> 🔍 推荐理由：Adam 被广泛使用，性能稳定，是训练神经网络时的首选优化器之一。

---

## 🧠 Adam 算法原理

　　Adam 是一种自适应学习率的优化算法，其核心思想是：

> 同时计算梯度的一阶矩（均值）和二阶矩（未中心化的方差），并利用这两个估计来动态调整每个参数的学习率。

---

### 🔁 算法流程（伪代码 + 公式）

#### 初始化

```text
Vdw = 0      # 动量项（一阶矩）
Sdw = 0      # RMSProp 项（二阶矩）
Vdb = 0
Sdb = 0
```

#### 第 T 次迭代步骤

1. **计算当前 mini-batch 的梯度**

   $$
   dw, db \leftarrow \frac{\partial J}{\partial w}, \frac{\partial J}{\partial b}
   $$
2. **更新动量项（一阶矩估计）——类似 Momentum**

   $$
   v_{dw} = \beta_1 v_{dw} + (1 - \beta_1) dw
   $$

   $$
   v_{db} = \beta_1 v_{db} + (1 - \beta_1) db
   $$
3. **更新 RMSProp 项（二阶矩估计）**

   $$
   s_{dw} = \beta_2 s_{dw} + (1 - \beta_2) dw^2
   $$

   $$
   s_{db} = \beta_2 s_{db} + (1 - \beta_2) db^2
   $$

   > ⚠️ 注意：`dw²` 表示逐元素平方（element-wise squaring）
   >
4. **偏差校正（Bias Correction）**   
   由于初始时刻 `v` 和 `s` 都为 0，导致前几轮估计有偏，需进行校正：

   $$
   v_{dw}^{\text{corrected}} = \frac{v_{dw}}{1 - \beta_1^T}
   $$

   $$
   v_{db}^{\text{corrected}} = \frac{v_{db}}{1 - \beta_1^T}
   $$

   $$
   s_{dw}^{\text{corrected}} = \frac{s_{dw}}{1 - \beta_2^T}
   $$

   $$
   s_{db}^{\text{corrected}} = \frac{s_{db}}{1 - \beta_2^T}
   $$
5. **参数更新**

   $$
   w := w - \alpha \cdot \frac{v_{dw}^{\text{corrected}}}{\sqrt{s_{dw}^{\text{corrected}}} + \epsilon}
   $$

   $$
   b := b - \alpha \cdot \frac{v_{db}^{\text{corrected}}}{\sqrt{s_{db}^{\text{corrected}}} + \epsilon}
   $$

---

## 🛠️ 超参数设置建议

|超参数|建议值|说明|
| --------| ------------------| ------------------------------|
|$\alpha$|需调参（如 $10^{-3}$ ~ $10^{-1}$）|学习率，最关键超参数|
|$\beta_1$|0.9|一阶矩衰减率（动量系数）|
|$\beta_2$|0.999|二阶矩衰减率（RMSProp 系数）|
|$\epsilon$|$10^{-8}$|小常数，防止除零|

> ✅ 实践建议：
>
> - $\beta_1$、$\beta_2$、$\epsilon$ 通常使用默认值。
> - 只需调整 $\alpha$ 即可获得良好效果。
> - 很少有人手动调节 $\beta_1$、$\beta_2$ 或 $\epsilon$。

---

## 📚 名称来源：Adam = Adaptive Moment Estimation

- **第一阶矩（First Moment）** ：由 $\beta_1$ 计算，表示梯度的指数加权平均 → 近似均值。
- **第二阶矩（Second Moment）** ：由 $\beta_2$ 计算，表示梯度平方的指数加权平均 → 近似方差。

　　因此：

$$
\text{Adam} = \text{Adaptive Moment Estimation}
$$

> 💡 注：虽然名字叫 Adam，但与研究者 Adam Coates 无关（只是巧合）。

---

## 🔄 总结对比表

|方法|是否包含动量|是否自适应学习率|是否需要调参多|
| ----------| --------------| ------------------| ------------------|
|SGD|❌|❌|✅（需调 α）|
|Momentum|✅|❌|✅|
|RMSProp|❌|✅|✅|
|**Adam**|✅|✅|❌（基本用默认）|

> ✅ Adam = Momentum + RMSProp 的融合体，兼具两者优点。

---

## 📌 关键优势

1. **收敛快**：结合动量加速方向，同时抑制震荡。
2. **鲁棒性强**：适用于不同架构的神经网络。
3. **超参数少**：只需调一个学习率即可。
4. **无需手动归一化**：自动调整各参数的学习速率。

---

## ✅ 附加知识扩展

### 为什么需要偏差校正？

- 初始阶段：$v_{dw} = 0$，但真实梯度不为零 → 估计偏低。
- 经过若干步后，随着 $T$ 增大，$\beta_1^T \to 0$，校正因子趋于 1。
- 校正使早期更新更准确。

### 何时不用 Adam？

- 如果你追求极致性能（如训练超大规模模型），可以考虑 **AdamW**、**Lion**、**AdaFactor** 等变种。
- 对于简单线性模型，SGD 有时更优。

---

## 🧾 最终总结（一句话）

> **Adam 是一种结合动量和自适应学习率的高效优化算法，通过估计梯度的一阶和二阶矩实现快速收敛，是现代深度学习中最常用、最可靠的优化器之一。**

---

　　✅ 推荐你在实践中优先尝试 Adam，配合合理的学习率调度（后续视频将讲“学习率衰减”），可显著提升训练效率。
