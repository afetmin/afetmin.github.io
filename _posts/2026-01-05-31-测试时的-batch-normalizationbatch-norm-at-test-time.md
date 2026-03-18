---
title: "31 测试时的 Batch Normalization（Batch Norm at Test Time）"
date: 2026-01-05T20:27:05Z
last_modified_at: 2026-01-05T20:27:15Z
categories:
  - 改进深度神经网络
math: true
---

### 一、背景与动机

- **训练阶段**：Batch Norm 对每个 mini-batch 中的激活值 $z^{(i)}$ 进行归一化，依赖该 batch 的均值 $\mu$ 和方差 $\sigma^2$。
- **测试阶段问题**：

  - 测试时通常**逐样本推理**（batch size = 1）；
  - 单个样本无法计算有意义的均值和方差；
  - 因此不能直接使用训练时的 batch-wise 统计量。

> ✅ **核心问题**：如何在测试时获得用于归一化的 $\mu$ 和 $\sigma^2$？

---

### 二、训练阶段的 Batch Norm 公式回顾

　　对于某一层 $l$，给定一个 mini-batch（含 $m$ 个样本），对每个样本 $i$：

1. **计算 mini-batch 均值**：

   $$
   \mu^{[l]} = \frac{1}{m} \sum_{i=1}^{m} z^{(i)[l]}
   $$
2. **计算 mini-batch 方差**：

   $$
   \sigma^{2[l]} = \frac{1}{m} \sum_{i=1}^{m} \left( z^{(i)[l]} - \mu^{[l]} \right)^2
   $$
3. **归一化**：

   $$
   z_{\text{norm}}^{(i)[l]} = \frac{z^{(i)[l]} - \mu^{[l]}}{\sqrt{\sigma^{2[l]} + \varepsilon}}
   $$

   其中 $\varepsilon > 0$ 是数值稳定项（如 $10^{-8}$）。
4. **缩放和平移（可学习参数）** ：

   $$
   \tilde{z}^{(i)[l]} = \gamma^{[l]} z_{\text{norm}}^{(i)[l]} + \beta^{[l]}
   $$

   其中 $\gamma, \beta$ 是可训练参数。

---

### 三、测试阶段的关键调整

#### ❗ 核心思想：

> 在测试时，**不再使用当前 batch 的统计量**，而是使用**训练过程中累积的全局估计值** $\hat{\mu}^{[l]}$ 和 $\hat{\sigma}^{2[l]}$。

#### ✅ 解决方案：**指数加权平均（Exponentially Weighted Average）**

　　在训练过程中，对每一层 $l$，持续更新两个全局统计量：

- **均值的移动平均**：

  $$
  \hat{\mu}^{[l]} := \alpha \cdot \hat{\mu}^{[l]} + (1 - \alpha) \cdot \mu^{[l]}_{\text{current batch}}
  $$
- **方差的移动平均**：

  $$
  \hat{\sigma}^{2[l]} := \alpha \cdot \hat{\sigma}^{2[l]} + (1 - \alpha) \cdot \sigma^{2[l]}_{\text{current batch}}
  $$

　　其中：

- $\alpha \in [0, 1)$ 是衰减率（decay rate），通常取 **0.9 或 0.99**；
- 初始值 $\hat{\mu}^{[l]} = 0$, $\hat{\sigma}^{2[l]} = 1$；
- 这些值在**训练过程中不断更新**，并在**测试时固定使用**。

> 💡 这也被称为 **running mean** 和 **running variance**。

---

### 四、测试阶段的前向传播公式

　　给定一个测试样本的激活值 $z^{[l]}$，使用训练阶段累积的全局统计量：

1. **归一化**：

   $$
   z_{\text{norm}}^{[l]} = \frac{z^{[l]} - \hat{\mu}^{[l]}}{\sqrt{\hat{\sigma}^{2[l]} + \varepsilon}}
   $$
2. **缩放和平移**（使用训练好的 $\gamma, \beta$）：

   $$
   \tilde{z}^{[l]} = \gamma^{[l]} z_{\text{norm}}^{[l]} + \beta^{[l]}
   $$

> ✅ 此过程**不依赖 batch**，可对单个样本进行推理。

---

### 五、其他可能的估算方式（补充）

　　虽然指数加权平均是主流方法，但理论上也可用以下方式估算 $\mu$ 和 $\sigma^2$：

- 在训练结束后，**在整个训练集上**前向传播一次，计算每层激活值的**真实均值和方差**；
- 但在实践中，由于效率和内存限制，**指数加权平均更常用**。

> 🔧 大多数深度学习框架（如 PyTorch、TensorFlow）默认使用 running mean/variance，并自动在训练/测试模式下切换行为。

---

### 六、关键总结（重点提炼）

|阶段|使用的 $\mu, \sigma^2$|是否可学习参数|是否依赖 batch|
| ------| --------------------------------------------| ----------------| ------------------|
|**训练**|当前 mini-batch 的统计量|$\gamma, \beta$ 可学习|是|
|**测试**|全局 running mean/variance（指数加权平均）|使用训练好的 $\gamma, \beta$|否（支持单样本）|

- **目的**：使网络在测试时行为稳定，不受 batch size 影响；
- **优势**：提升模型泛化能力，允许更深网络训练；
- **实现注意**：必须区分 `train mode` 和 `eval mode`，否则测试结果会错误！

---

### 七、实践建议

- 在代码中务必调用 `.train()` / `.eval()`（PyTorch）或设置 `training=False`（TF/Keras）；
- 不要手动实现 running statistics，除非必要——框架已优化；
- $\varepsilon$ 通常设为 $10^{-5}$ 或 $10^{-8}$，防止除零错误；
- Batch Norm 的效果对 batch size 敏感，小 batch 时可考虑 **Group Norm** 或 **Layer Norm** 替代。
