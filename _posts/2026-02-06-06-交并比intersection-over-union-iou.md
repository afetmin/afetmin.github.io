---
title: "06 交并比（Intersection over Union, IoU）"
date: 2026-02-06T16:18:28Z
last_modified_at: 2026-02-06T16:21:16Z
categories:
  - 卷积神经网络
math: true
---

# 06 交并比（Intersection over Union, IoU）

### 一、IoU 的定义与作用

在**目标检测（Object Detection）** 任务中，不仅要识别出图像中是否存在某个物体，还需要**精确定位其位置**，通常用**边界框（Bounding Box）** 表示。为了衡量预测边界框与真实边界框（Ground Truth）之间的**定位精度**，引入了 **交并比（IoU）**  这一指标。

> **IoU 衡量的是两个边界框重叠程度的相对大小**，是评估目标检测算法性能的关键指标之一。

---

### 二、IoU 的数学定义

设：

- $B_{\text{pred}}$：模型预测的边界框
- $B_{\text{gt}}$：真实（Ground Truth）边界框

则 IoU 定义为：

$$
\text{IoU} = \frac{\text{Area of Intersection}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area of Union}(B_{\text{pred}} \cup B_{\text{gt}})}
$$

其中：

- **交集（Intersection）** ：两个边界框重叠区域的面积
- **并集（Union）** ：两个边界框覆盖的总面积（即各自面积之和减去交集）

> ✅ IoU 的取值范围为：

$$
0 \leq \text{IoU} \leq 1
$$

- 当两个框**完全不重叠**时，IoU = 0；
- 当两个框**完全重合**时，IoU = 1。

---

### 三、IoU 的阈值判断标准

在目标检测中，通常设定一个 **IoU 阈值** 来判断一次检测是否“成功”：

- **通用标准**：若

  $$
  \text{IoU} \geq 0.5
  $$

  则认为该预测框为**正确定位**（True Positive）。
- **更严格场景**（如高精度要求）：可将阈值提高至 0.6、0.7 甚至更高。
- **极少情况**会使用低于 0.5 的阈值。

> ⚠️ 注意：0.5 是经验性约定，**并无严格的理论依据**，可根据具体任务调整。

---

### 四、IoU 的应用场景

1. **评估检测性能**：作为 mAP（mean Average Precision）等指标的基础组成部分。
2. **非极大值抑制（Non-Maximum Suppression, NMS）** ：在后续视频中，IoU 用于**去除冗余检测框**——保留置信度最高的框，剔除与其 IoU 超过阈值的其他重叠框。
3. **损失函数设计**：某些现代检测器（如 YOLOv3+、GIoU、DIoU、CIoU）将 IoU 或其变体直接嵌入损失函数，以优化定位精度。

---

### 五、常见误区提醒

- **IoU ≠ “I owe you”（我欠你）** ：虽然缩写相同，但此处的 **IoU = Intersection over Union**，纯属术语巧合，注意区分。

---

### 六、小结（Key Takeaways）

|项目|内容|
| ------| --------------------------------|
|**目的**|评估预测框与真实框的定位准确性|
|**公式**|$\text{IoU} = \frac{\text{Area of Intersection}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area of Union}(B_{\text{pred}} \cup B_{\text{gt}})}$|
|**取值范围**|$[0, 1]$|
|**常用阈值**|0.5（可调）|
|**核心用途**|检测评价、NMS、损失函数设计|

---

掌握 IoU 是理解现代目标检测系统（如 YOLO、SSD、Faster R-CNN）性能评估和后处理机制的基础。建议结合后续“非极大值抑制（NMS）”内容一起学习，效果更佳。
