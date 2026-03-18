---
title: "10 候选区域（Region Proposals, Optional）"
date: 2026-02-06T16:29:54Z
last_modified_at: 2026-02-06T16:31:35Z
categories:
  - 卷积神经网络
math: true
---

# 10 候选区域（Region Proposals, Optional）

## 🧠 一、核心思想：从滑动窗口到候选区域

### 1. 滑动窗口法的问题

- **传统方法**：在图像上使用固定大小的滑动窗口，对每个窗口运行分类器（如 CNN），判断是否包含目标（如车辆、行人等）。
- **主要缺点**：

  - **计算冗余**：大量窗口覆盖的是“空背景”区域（如天空、道路），无实际目标，却仍需运行昂贵的 CNN 推理。
  - **效率低下**：即使使用卷积实现滑动窗口（见 3.4 节），仍需处理全图所有位置。

> 💡 目标：**只在“可能包含目标”的区域运行分类器**，减少无效计算。

---

## 🔍 二、R-CNN：Region-based CNN（基于区域的卷积网络）

### 1. 基本流程（Girshick et al., CVPR 2014）

1. **生成候选区域（Region Proposals）**   
   使用**图像分割算法**（如 Selective Search）生成约 2000 个可能包含目标的区域（称为“色块”或“超像素聚类”）。
2. **对每个候选区域**：

   - 提取对应图像块；
   - 调整为固定尺寸（如 227×227）；
   - 输入预训练 CNN（如 AlexNet）提取特征；
   - 用 SVM 分类器判断类别；
   - 用回归器微调边界框坐标。

### 2. 优势

- **聚焦有效区域**：仅在 ~2000 个候选区域上运行 CNN，而非全图滑动窗口（可能数十万窗口）。
- **支持多尺度与多形状**：候选区域可为任意高宽比（瘦高检测行人，宽扁检测车辆）。
- **边界框精修**：输出的边界框 $\hat{B} = (x, y, w, h)$ 通过回归优化，比原始分割边界更精确。

> ✅ 公式（边界框回归）：  
> 设真实框为 $B^* = (x^*, y^*, w^*, h^*)$，候选框为 $P = (x_p, y_p, w_p, h_p)$，则回归目标为：
>
> $$
> t_x = \frac{x^* - x_p}{w_p}, \quad
> t_y = \frac{y^* - y_p}{h_p}, \quad
> t_w = \log\left(\frac{w^*}{w_p}\right), \quad
> t_h = \log\left(\frac{h^*}{h_p}\right)
> $$
>
> 网络学习映射 $P \mapsto (t_x, t_y, t_w, t_h)$。

### 3. 缺点

- **速度极慢**：每个候选区域独立前向传播 CNN，无法共享计算。
- **多阶段训练**：CNN 微调、SVM 训练、边界框回归需分步进行。

---

## ⚡ 三、Fast R-CNN（2015）

### 改进思路

- **共享卷积特征**：对整张图像只运行一次 CNN，得到 feature map。
- **RoI Pooling（Region of Interest Pooling）** ：

  - 将每个候选区域映射到 feature map 上；
  - 通过空间金字塔池化（或固定网格池化）将其转为固定长度特征向量。

### 流程

1. 输入整图 → CNN → feature map；
2. 在 feature map 上投影候选区域；
3. RoI Pooling → 全连接层 → **同时输出**：

   - 类别概率 $p(c | R)$（含背景类）；
   - 边界框偏移 $(t_x, t_y, t_w, t_h)$。

> ✅ 优点：
>
> - **端到端训练**；
> - **速度大幅提升**（因共享卷积计算）。

> ❌ 仍存在的问题：**候选区域生成（如 Selective Search）非常慢**，且非神经网络实现。

---

## 🚀 四、Faster R-CNN（Ren et al., NIPS 2015）

### 核心创新：**RPN（Region Proposal Network）**

- 用一个轻量级 CNN 替代传统分割算法，**直接从 feature map 生成候选区域**。
- RPN 与检测网络共享底层卷积特征。

### RPN 工作原理

1. 在 feature map 上滑动小窗口（如 3×3）；
2. 对每个位置，预测 $k$ 个**锚框（anchors）** （预设不同尺度/长宽比）；
3. 输出：

   - 每个 anchor 是否包含目标（二分类：object / not object）；
   - 对应的边界框偏移量。

> ✅ 锚框示例（常见设置）：
>
> $$
> \text{Scales: } \{128^2, 256^2, 512^2\}, \quad
> \text{Ratios: } \{1:1, 1:2, 2:1\}
> \Rightarrow k = 9 \text{ anchors per location}
> $$

### 整体架构

- **共享 backbone**（如 VGG、ResNet）；
- **RPN 分支**：生成 proposals；
- **Fast R-CNN 分支**：对 proposals 进行分类与精修。

> ✅ 优势：
>
> - **完全端到端**；
> - **实时性显著提升**（但仍慢于 YOLO）。

---

## 🤔 五、候选区域 vs. 单阶段检测器（如 YOLO）

|方法|阶段数|速度|精度|特点|
| --------------| --------| ------| ------| ------------------------------------|
|R-CNN|两阶段|慢|高|精确但效率低|
|Fast R-CNN|两阶段|中|高|共享特征|
|Faster R-CNN|两阶段|较快|高|RPN 自动生成 proposals|
|YOLO / SSD|**单阶段**|**快**|中高|“You Only Look Once”，端到端回归|

> 🎯 吴恩达观点（个人看法）：
>
> - 候选区域是**重要历史贡献**，影响深远；
> - 但**单阶段检测器**（如 YOLO）代表更高效、简洁的未来方向；
> - 建议了解 R-CNN 系列，因其仍在学术/工业界广泛使用。

---

## 📚 六、关键参考文献

1. **R-CNN**:  
   Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014).  
   *Rich feature hierarchies for accurate object detection and semantic segmentation*.  
   **CVPR**.
2. **Fast R-CNN**:  
   Girshick, R. (2015).  
   *Fast R-CNN*.  
   **ICCV**.
3. **Faster R-CNN**:  
   Ren, S., He, K., Girshick, R., & Sun, J. (2015).  
   *Faster R-CNN: Towards real-time object detection with region proposal networks*.  
   **NeurIPS**.

---

## ✅ 总结要点（速记）

- **候选区域**：减少无效计算，聚焦潜在目标区域。
- **R-CNN**：先提 proposals，再逐个分类 + 回归 → 慢。
- **Fast R-CNN**：整图 CNN + RoI Pooling → 共享特征，加速。
- **Faster R-CNN**：引入 RPN，用 CNN 生成 proposals → 端到端两阶段检测。
- **趋势**：两阶段（高精度） vs. 单阶段（高速度），YOLO 代表后者。
