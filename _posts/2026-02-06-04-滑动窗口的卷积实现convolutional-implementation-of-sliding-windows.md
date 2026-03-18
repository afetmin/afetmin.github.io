---
title: "04 滑动窗口的卷积实现（Convolutional Implementation of Sliding Windows）"
date: 2026-02-06T16:09:41Z
last_modified_at: 2026-02-06T16:14:11Z
categories:
  - 卷积神经网络
math: true
---

# 04 滑动窗口的卷积实现（Convolutional Implementation of Sliding Windows）

## 一、背景：传统滑动窗口方法的问题

在目标检测任务中，传统滑动窗口方法的做法是：

- 将输入图像划分为多个固定大小（如 $14 \times 14$）的子区域；
- 对每个子区域单独送入一个训练好的分类 CNN；
- 输出该区域是否包含目标（如行人、汽车等）及其类别概率。

### 缺点：

- **计算冗余严重**：相邻窗口之间存在大量重叠区域，导致重复前向传播；
- **效率低下**：对每个窗口独立运行整个网络，时间复杂度高。

---

## 二、核心思想：将全连接层转换为卷积层

为了提升效率，关键在于 **将全连接层（Fully Connected Layer）替换为卷积层（Convolutional Layer）** ，从而使得整个网络变为“全卷积网络”（Fully Convolutional Network, FCN），支持任意尺寸输入并一次性输出所有窗口的预测结果。

### 转换过程详解（以示例网络为例）：

原始网络结构（用于 $14 \times 14 \times 3$ 输入）：

1. 卷积层：$5 \times 5$ 卷积核 ×16 → 输出 $10 \times 10 \times 16$
2. 最大池化：$2 \times 2$，步长 2 → 输出 $5 \times 5 \times 16$
3. 全连接层：400 个神经元
4. 全连接层 + Softmax：输出 4 类概率（如行人、汽车、摩托车、背景）

#### ✅ 转换步骤：

1. **第一全连接层 → 卷积层**

   - 原输入：$5 \times 5 \times 16$
   - 使用 **$5 \times 5 \times 16$** **的卷积核**，共 400 个
   - 输出：$1 \times 1 \times 400$
   - 数学上等价于全连接：每个输出单元 = $\sum_{i,j,k} w_{ijk}^{(l)} \cdot a_{ijk}^{(l-1)} + b^{(l)}$
2. **第二全连接层 →**  **$1 \times 1$** **卷积层**

   - 输入：$1 \times 1 \times 400$
   - 使用 **$1 \times 1 \times 400$** **卷积核**，共 400 个
   - 输出：$1 \times 1 \times 400$
3. **Softmax 层 →**  **$1 \times 1$** **卷积 + Softmax**

   - 使用 4 个 $1 \times 1 \times 400$ 卷积核
   - 输出：$1 \times 1 \times 4$，即 4 个类别的概率分布

> 📌 **关键结论**：全连接层本质上是空间维度为 $1 \times 1$ 的卷积操作。将其显式写成卷积形式后，网络可接受任意尺寸输入。

---

## 三、滑动窗口的卷积实现（高效版本）

### 场景设定：

- 原训练图像尺寸：$14 \times 14 \times 3$
- 测试图像尺寸：$16 \times 16 \times 3$
- 滑动窗口步长：2 像素 → 共有 $2 \times 2 = 4$ 个窗口位置

### 传统方法：

- 分别裁剪 4 个 $14 \times 14$ 区域，各自前向传播 → 4 次独立计算

### 卷积实现方法：

- **将整张** **$16 \times 16 \times 3$** **图像一次性输入全卷积网络**
- 网络自动在内部完成“滑动窗口”的等效计算
- 输出为 $2 \times 2 \times 4$ 的张量，每个 $(i,j)$ 位置对应原图中第 $(i,j)$ 个窗口的 4 类概率

#### 输出解释：

设输出张量为 $\mathbf{Y} \in \mathbb{R}^{2 \times 2 \times 4}$，则：

- $\mathbf{Y}[0,0,:]$：左上角 $14 \times 14$ 区域的预测
- $\mathbf{Y}[0,1,:]$：右上角区域预测
- $\mathbf{Y}[1,0,:]$：左下角区域预测
- $\mathbf{Y}[1,1,:]$：右下角区域预测

> ✅ **优势**：共享卷积特征计算，避免重复；只需一次前向传播即可获得所有窗口结果。

---

## 四、一般化公式推导

设输入图像尺寸为 $H \times W \times C$，滑动窗口大小为 $h \times w$，步长为 $s$，则：

- 滑动窗口数量（输出网格大小）为：

  $$
  H_{\text{out}} = \left\lfloor \frac{H - h}{s} \right\rfloor + 1,\quad
  W_{\text{out}} = \left\lfloor \frac{W - w}{s} \right\rfloor + 1
  $$

当使用全卷积网络时，若网络最后的卷积层感受野等于 $h \times w$，且总下采样倍率为 $r$（由池化和卷积步长决定），则：

- 输入尺寸 $H \times W$ → 输出尺寸 $\frac{H}{r} \times \frac{W}{r}$
- 若 $r = s$ 且 $h = r \cdot k$（$k$ 为某整数），则输出网格自然对应滑动窗口位置

> 💡 在示例中，最大池化步长为 2，总下采样率 $r=2$，故 $16 \to 8$，但因窗口为 $14$，实际有效输出为 $2 \times 2$（因 $16 - 14 = 2$，步长 2 → 2 个位置）。

---

## 五、算法流程总结

1. **训练阶段**：

   - 使用固定尺寸图像（如 $14 \times 14$）训练 CNN 分类器
   - 将全连接层替换为等效卷积层，构建全卷积网络（FCN）
2. **推理阶段**：

   - 输入任意尺寸图像（如 $28 \times 28$）
   - 网络输出一个 **密集预测图**（dense prediction map），如 $8 \times 8 \times 4$
   - 每个空间位置 $(i,j)$ 对应原图中以 $(i \cdot s, j \cdot s)$ 为中心的窗口的分类结果
3. **优点**：

   - 计算高效（共享特征）
   - 支持任意输入尺寸
   - 适用于密集预测任务（如目标检测、语义分割）
4. **局限性**：

   - 边界框位置由滑动窗口网格决定，**定位精度受限于步长** **$s$**
   - 无法生成任意形状或尺度的边界框（需后续引入如 **Region Proposal** 或 **YOLO/SSD** 等方法改进）

---

## 六、参考文献与延伸

- **OverFeat (2013)** ：首次系统提出用全卷积网络实现滑动窗口检测

  > Sermanet, P., et al. *OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks.*  arXiv:1312.6229
  >
- **后续发展**：

  - R-CNN 系列引入 Region Proposal 解决定位不准问题
  - YOLO、SSD 直接回归边界框，摆脱滑动窗口范式

---

## 七、关键公式汇总

1. **卷积输出尺寸（无填充）** ：

   $$
   H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - K}{S} \right\rfloor + 1
   $$

   其中 $K$ 为卷积核大小，$S$ 为步长。
2. **全连接 → 卷积等价性**：

   $$
   z^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} 
   \quad \Leftrightarrow \quad
   Z^{(l)} = \text{Conv}\left(A^{(l-1)},\ \mathcal{K}^{(l)}\right) + b^{(l)}
   $$

   当 $\mathcal{K}^{(l)}$ 的空间尺寸等于 $A^{(l-1)}$ 的空间尺寸时，输出为 $1 \times 1$。
3. **滑动窗口数量**：

   $$
   N_{\text{windows}} = \left( \left\lfloor \frac{W - w}{s} \right\rfloor + 1 \right) \times \left( \left\lfloor \frac{H - h}{s} \right\rfloor + 1 \right)
   $$

---

通过本节学习，你应掌握：**如何将分类 CNN 改造为支持高效滑动窗口检测的全卷积网络**，这是现代目标检测（如 Faster R-CNN 的 RPN、YOLO 的早期版本）的重要基础。
