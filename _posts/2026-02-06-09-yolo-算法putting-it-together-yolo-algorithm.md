---
title: "09 YOLO 算法（Putting it together： YOLO algorithm）"
date: 2026-02-06T16:26:40Z
last_modified_at: 2026-02-06T16:29:45Z
categories:
  - 卷积神经网络
math: true
---

# 09 YOLO 算法（Putting it together: YOLO algorithm）

## 🧠 一、YOLO 算法整体思想

YOLO（You Only Look Once）是一种**单阶段（one-stage）目标检测算法**，其核心思想是：

> 将目标检测问题转化为**回归问题**：给定一张图像，通过一个卷积神经网络直接预测出所有目标的边界框（bounding box）及其类别概率。

YOLO 将输入图像划分为 $S \times S$ 的网格（grid cells），每个网格负责预测若干个边界框（通常借助 anchor boxes），并判断这些框是否包含目标以及属于哪一类。

---

## 📦 二、训练集构造（Ground Truth Labeling）

### 1. 类别设定

- 假设要检测 $C = 3$ 类目标：行人、汽车、摩托车。
- **不显式定义“背景”类**，而是通过置信度 $\mathbb{1}_{\text{obj}}$ 判断是否有目标。

### 2. Anchor Boxes

- 使用 $B = 2$ 个预定义的 anchor boxes（具有不同宽高比）。
- 每个 anchor box 对应一个预测分支。

### 3. 输出张量维度

对每个网格单元，输出一个向量：

$$
\underbrace{[p_c, b_x, b_y, b_h, b_w, c_1, c_2, c_3]}_{8\text{ 维}} \quad \text{（每个 anchor box）}
$$

其中：

- $p_c = \mathbb{1}_{\text{obj}}$：该 anchor box 是否包含目标（0 或 1）；
- $(b_x, b_y)$：边界框中心相对于当前网格左上角的偏移（归一化到 [0,1]）；
- $(b_h, b_w)$：边界框高度和宽度（相对于整张图像的比例）；
- $c_1, c_2, c_3$：one-hot 编码的类别概率（仅当 $p_c=1$ 时有效）。

> ⚠️ 注意：若某网格无目标，则对应 anchor box 的 $p_c = 0$，其余值为 “don’t care”，训练时不参与损失计算。

### 4. 整体输出形状

- 网格大小：$S \times S = 3 \times 3$（教学示例）；
- 每个网格有 $B = 2$ 个 anchor boxes；
- 每个 anchor 输出 8 维向量；
- 总输出张量：

  $$
  S \times S \times B \times (5 + C) = 3 \times 3 \times 2 \times 8 = 3 \times 3 \times 16
  $$

> ✅ 实际应用中常用 $S = 19$，$B = 5$，则输出为 $19 \times 19 \times 5 \times 8 = 19 \times 19 \times 40$。

---

## 🔍 三、训练过程关键细节

- **目标分配规则**：对每个真实边界框（ground truth box），计算其与所有 anchor boxes 的 **IoU（交并比）** ，将其分配给 **IoU 最大的那个 anchor box**。
- 只有被分配的 anchor box 的 $p_c = 1$，其余 anchor boxes 在该网格中 $p_c = 0$。
- 类别标签 $c_i$ 仅在 $p_c = 1$ 时有意义。

---

## 🤖 四、推理（预测）过程

输入图像 → 卷积网络 → 输出 $S \times S \times B \times (5 + C)$ 张量。

对每个网格中的每个 anchor box，网络输出：

- 一个置信度分数 $p_c$（实际实现中常为 sigmoid 输出，表示存在目标的概率）；
- 边界框坐标 $(b_x, b_y, b_h, b_w)$；
- 类别概率分布（通常为 softmax 或 sigmoid 输出）。

> 💡 注意：即使某网格无目标，网络仍会输出具体数值（不能输出“don’t care”），但因 $p_c \approx 0$，这些预测会被后续过滤。

---

## 🧹 五、后处理：非极大值抑制（Non-Max Suppression, NMS）

由于每个网格可能输出多个重叠的预测框，需进行 NMS 去除冗余。

### 步骤如下：

1. **过滤低置信度预测**：  
   设定阈值 $\tau$（如 0.6），丢弃所有满足

   $$
   p_c < \tau
   $$

   的预测框。
2. **按类别分别处理**：  
   对每个类别 $c \in \{1, 2, 3\}$（行人、汽车、摩托车）：

   - 提取所有预测为该类别的边界框；
   - 计算每个框的 **类别置信度得分**：

     $$
     \text{score}_c = p_c \cdot P(\text{class}=c)
     $$
   - 对该类别的所有框执行 NMS：

     - 选择得分最高的框；
     - 删除与其 IoU > 阈值（如 0.5）的其他框；
     - 重复直至无剩余框。
3. **输出最终检测结果**：保留所有通过 NMS 的边界框及其类别。

> ✅ **关键点**：YOLO 对每个类别独立运行 NMS，避免不同类别之间的误抑制。

---

## 📌 六、YOLO 核心优势总结

|特性|说明|
| ------| ----------------------------------------------------|
|**端到端训练**|单一网络同时学习定位与分类|
|**实时性高**|只需一次前向传播即可完成检测|
|**全局上下文感知**|不像滑动窗口或 R-CNN 系列，YOLO 利用整图信息做预测|
|**结构简洁**|易于部署，适合嵌入式或移动端|

---

## 📘 七、关键公式汇总（KaTeX 兼容）

1. **输出向量结构（每个 anchor box）** ：

   $$
   \left[ p_c,\ b_x,\ b_y,\ b_h,\ b_w,\ c_1,\ c_2,\ \dots,\ c_C \right] \in \mathbb{R}^{5 + C}
   $$
2. **整体输出张量**：

   $$
   \mathbf{Y} \in \mathbb{R}^{S \times S \times B \times (5 + C)}
   $$
3. **类别置信度得分（用于 NMS）** ：

   $$
   \text{score}_c = p_c \cdot P(c \mid \text{obj})
   $$
4. **IoU（交并比）** ：

   $$
   \text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
   $$

---

## ✅ 学习建议

- 动手实现 YOLO 的简化版（如 $3\times3$ 网格 + 2 anchors）有助于理解 label 构造与 loss 设计；
- 注意区分 **训练时的 ground truth 构造** 与 **推理时的预测解码**；
- NMS 是目标检测通用技术，不仅用于 YOLO，也用于 Faster R-CNN、SSD 等。
