---
title: "05 Bounding Box预测（YOLO算法基础）"
date: 2026-02-06T16:14:13Z
last_modified_at: 2026-02-06T16:17:54Z
categories:
  - 卷积神经网络
math: true
---

# 05 Bounding Box预测（YOLO算法基础）

## 📘 课程主题：YOLO 算法基础 —— 单次检测实现精准边界框预测

### 一、背景与动机

- **滑动窗口法的问题**：

  - 使用离散窗口位置进行分类，无法输出**任意形状/位置**的边界框。
  - 边界框受限于窗口大小和步长，**精度低**。
  - 无法处理非正方形目标（如横向延伸的汽车）。
- **目标**：设计一种能**直接回归精确边界框坐标**的端到端卷积神经网络。

---

### 二、YOLO 核心思想（You Only Look Once）

> YOLO 将目标检测视为**单次回归问题**：输入整张图像，直接输出所有边界框及其类别。

#### 1. 网格划分（Grid Cell Division）

- 将输入图像划分为 $S \times S$ 的网格（例如 $3\times3$ 或更常用的 $19\times19$）。
- 每个网格负责**预测中心点落在该网格内的目标**。

#### 2. 目标分配规则

- 对每个真实目标（ground truth object）：

  - 计算其**边界框中心点** $(x_{\text{center}}, y_{\text{center}})$。
  - 将该目标**分配给包含其中心点的唯一网格单元**。
  - **即使目标跨越多个网格，也只由一个网格负责预测**。

> ✅ 优点：避免重复检测；简化标签设计。  
> ⚠️ 缺点：若多个目标中心落在同一网格（尤其在粗网格下），会漏检（后续版本通过 anchor boxes 改进）。

---

### 三、输出表示：每个网格的预测向量

对每个网格单元，输出一个 **8 维向量**：

$$
\mathbf{y} = 
\begin{bmatrix}
p_c \\
b_x \\
b_y \\
b_h \\
b_w \\
c_1 \\
c_2 \\
c_3
\end{bmatrix}
$$

其中：

|符号|含义|取值说明|
| ------| --------------------------| ----------------------------------------|
|$p_c$|是否包含目标|$p_c = 1$ 表示有目标，$0$ 表示无目标|
|$b_x, b_y$|边界框中心相对于**当前网格左上角**的偏移|归一化到 $[0, 1]$|
|$b_h, b_w$|边界框高、宽相对于**整个图像高度/宽度**的比例|可大于 1（若目标比网格大）|
|$c_1, c_2, c_3$|类别概率（one-hot 编码）|如：行人、汽车、摩托车；背景不单独编码|

> 🔍 注意：若某网格无目标，则 $p_c = 0$，其余值为 “don’t care”（训练时可忽略）。

#### 示例：

- 若某网格包含一辆汽车（类别2），则标签为：

  $$
  \mathbf{y} = 
  \begin{bmatrix}
  1 \\
  b_x \\
  b_y \\
  b_h \\
  b_w \\
  0 \\
  1 \\
  0
  \end{bmatrix}
  $$

---

### 四、网络架构与训练

- **输入**：$100 \times 100 \times 3$ 图像（仅为教学示例，实际更大）。
- **主干网络**：标准 CNN（含卷积层、池化层等）。
- **输出层**：通过卷积/全连接层，最终输出尺寸为 $S \times S \times 8$。

  - 例如 $3\times3$ 网格 → 输出 $3 \times 3 \times 8$ 张量。
  - 实际常用 $19\times19$ → 输出 $19 \times 19 \times 8$。
- **训练方式**：

  - 构建对应的目标标签张量 $\mathbf{Y} \in \mathbb{R}^{S \times S \times 8}$。
  - 使用**均方误差（MSE）损失函数**（原始 YOLO v1）：

    $$
    \mathcal{L} = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
    + \lambda_{\text{coord}} \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
    + \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} (C_i - \hat{C}_i)^2 \\
    + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{noobj}} (C_i - \hat{C}_i)^2 \\
    + \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
    $$

    > 注：此处为原始 YOLO v1 损失函数概要，视频未展开，但理解输出结构是关键。
    >

---

### 五、边界框参数化细节（重点！）

对分配到某网格的目标，其边界框参数定义如下：

- 设当前网格左上角坐标为 $(x_{\text{grid}}, y_{\text{grid}})$，网格宽高为 $w_{\text{cell}}, h_{\text{cell}}$。
- 则：

  $$
  \begin{aligned}
  b_x &= \frac{x_{\text{center}} - x_{\text{grid}}}{w_{\text{cell}}} \in [0, 1] \\
  b_y &= \frac{y_{\text{center}} - y_{\text{grid}}}{h_{\text{cell}}} \in [0, 1] \\
  b_w &= \frac{w_{\text{box}}}{W_{\text{image}}} \\
  b_h &= \frac{h_{\text{box}}}{H_{\text{image}}}
  \end{aligned}
  $$

> ✅ 优势：$b_x, b_y$ 被约束在 $[0,1]$，保证中心点在本网格内。  
> ⚠️ $b_w, b_h$ 可 >1，允许目标跨越多个网格。

> 💡 高级技巧（论文中使用）：
>
> - 用 **sigmoid 函数** 约束 $b_x, b_y \in (0,1)$。
> - 用 **指数变换** 确保 $b_w, b_h > 0$，如：$\hat{b}_w = e^{t_w}, \hat{b}_h = e^{t_h}$。

---

### 六、YOLO 的核心优势

1. **端到端训练**：直接从像素到边界框+类别。
2. **全局推理**：一次性看完整张图，避免滑动窗口的局部性。
3. **高效卷积实现**：

   - 不需在每个网格独立运行分类器（如 $19\times19=361$ 次）。
   - 所有网格共享卷积特征，**计算高效**。
4. **实时性**：原始 YOLO 可达 45 FPS，适合实时应用。

---

### 七、局限性（本节提及）

- **每个网格只能预测一个目标** → 多目标重叠时漏检。
- **小目标检测弱**（尤其在粗网格下）。
- **定位精度不如两阶段方法**（如 R-CNN 系列）——但速度远胜。

> ✅ 后续改进：YOLO v2 引入 **Anchor Boxes** 解决多目标问题；YOLO v3/v4/v5 进一步优化。

---

### 八、学习建议

- **精读论文**：Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.

  - ⚠️ 提醒：论文较难，建议结合代码（如 Darknet）理解。
- **动手实践**：尝试用 PyTorch/TensorFlow 实现简化版 YOLO（$7\times7$ 网格，单类别）。
- **对比学习**：与 Faster R-CNN、SSD 等方法比较“单阶段 vs 两阶段”检测范式。

---

## ✅ 总结一句话：

> **YOLO 通过将图像划分为网格，并让每个网格直接回归其负责目标的边界框和类别，实现了高效、端到端、实时的目标检测。**
