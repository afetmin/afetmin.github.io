---
title: "05 带步长的卷积（Strided Convolutions）"
date: 2026-01-18T12:15:18Z
last_modified_at: 2026-01-18T13:24:09Z
categories:
  - 卷积神经网络
math: true
---

# 05 带步长的卷积（Strided Convolutions）

### 一、基本概念回顾

在标准卷积中，我们使用一个 **滤波器（filter）** （也称卷积核）在输入图像上滑动，逐点进行**逐元素相乘再求和**，生成输出特征图（feature map）。

- 输入图像尺寸：$n \times n$
- 滤波器尺寸：$f \times f$
- Padding（填充）：$p$
- Stride（步长）：$s$

> 💡 **注意**：在深度学习中，所谓的“卷积”实际上是指**互相关（cross-correlation）** ，即**不翻转滤波器**。尽管数学或信号处理中“卷积”包含翻转操作，但深度学习惯例省略此步骤，仍称其为“卷积”。

---

### 二、带步长的卷积（Strided Convolution）

#### 1. 步长（Stride）定义

- **步长** **$s$** 表示滤波器每次滑动的像素数。
- 当 $s = 1$：标准卷积，逐像素滑动。
- 当 $s = 2$：每次跳过 1 个像素，滑动 2 个位置。

#### 2. 输出尺寸计算公式

当对一个 $n \times n$ 的输入图像，使用 $f \times f$ 的滤波器，padding 为 $p$，stride 为 $s$ 时，输出特征图的尺寸为：

$$
\left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor \times \left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor
$$

> ✅ **关键点**：
>
> - 若结果不是整数，**向下取整（floor）** 。
> - 这是因为只有当滤波器**完全落在（填充后的）输入区域内**时，才进行一次有效卷积运算。
> - 向下取整确保了不会对部分超出边界的区域进行计算。

#### 3. 示例验证

- 输入：$7 \times 7$
- 滤波器：$3 \times 3$
- Padding：$p = 0$
- Stride：$s = 2$

代入公式：

$$
\left\lfloor \frac{7 + 2 \cdot 0 - 3}{2} + 1 \right\rfloor = \left\lfloor \frac{4}{2} + 1 \right\rfloor = \left\lfloor 2 + 1 \right\rfloor = 3
$$

→ 输出尺寸为 $3 \times 3$，与课程示例一致。

---

### 三、边界情况处理

- 如果 $\frac{n + 2p - f}{s} + 1$ 不是整数，则**丢弃**无法完整覆盖滤波器的边缘区域。
- 这是深度学习框架（如 TensorFlow、PyTorch）中的**默认行为**（除非特别指定“same” padding 等策略）。

---

### 四、关于“卷积” vs “互相关”的技术说明

|领域|操作|是否翻转滤波器|
| -----------------| ---------------------| --------------------------|
|数学 / 信号处理|卷积（Convolution）|✅ 是（先水平+垂直翻转）|
|深度学习|通常称为“卷积”|❌ 否（实际是互相关）|

- **为什么深度学习不用翻转？**

  - 翻转对特征提取无实质帮助；
  - 学习过程中滤波器权重是**可训练的**，无论是否翻转，网络都能学到等效的模式；
  - 省略翻转可简化实现，提升效率。

> 📌 **结论**：在 CNN 中，“卷积” = **互相关（无翻转）** ，这是行业标准约定。

---

### 五、关键公式汇总（KaTeX 兼容）

1. **输出尺寸公式（方形输入）** ：

   $$
   n_{\text{out}} = \left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor
   $$
2. **通用形式（矩形输入** **$n_H \times n_W$** **）** ：

   $$
   H_{\text{out}} = \left\lfloor \frac{n_H + 2p - f}{s} + 1 \right\rfloor, \quad
   W_{\text{out}} = \left\lfloor \frac{n_W + 2p - f}{s} + 1 \right\rfloor
   $$
3. **向下取整符号**：

   $$
   \lfloor x \rfloor = \text{floor}(x)
   $$

---

### 六、学习建议

- **动手验证**：尝试不同 $n, f, p, s$ 组合，手动计算输出尺寸，加深理解。
- **编程实践**：在 PyTorch 或 TensorFlow 中使用 `Conv2d` 层，调整 `stride` 和 `padding` 参数，观察输出形状变化。
- **注意维度匹配**：在构建深层网络时，合理设计 stride 和 padding 可控制特征图分辨率，影响模型感受与计算量。

---

### 七、预告（课程下一部分）

下一节将讲解 **对立体（volume）进行卷积**，即处理 **多通道输入（如 RGB 图像）**  和 **多滤波器输出**，这是构建真实 CNN 的核心。

---

✅ **总结一句话**：

> **带步长的卷积通过控制滑动步幅** **$s$** **来调节输出分辨率，其尺寸由公式** **$\left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor$** **决定；深度学习中的“卷积”实为互相关，无需翻转滤波器。**
