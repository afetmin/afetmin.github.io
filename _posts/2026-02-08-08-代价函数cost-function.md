---
title: "08 代价函数（Cost function）"
date: 2026-02-08T12:00:53Z
last_modified_at: 2026-02-08T12:01:31Z
categories:
  - 卷积神经网络
math: true
---

# 08 代价函数（Cost function）

### 一、问题定义

给定：

- 一张**内容图像** $C$
- 一张**风格图像** $S$

目标：生成一张新图像 $G$，使其：

- **内容**上接近 $C$
- **风格**上接近 $S$

这就是**神经风格迁移（Neural Style Transfer）** 的核心任务。

---

### 二、整体策略：优化生成图像 $G$

我们不训练一个神经网络去“生成”图像，而是：

- **固定预训练的 CNN（如 VGG-19）**
- **将生成图像** **$G$** **本身作为可学习参数**
- 通过**梯度下降**直接优化 $G$ 的像素值，以最小化一个精心设计的**总代价函数**

> 初始时，$G$ 可以是随机噪声（例如白噪声图像），尺寸任意（如 $100 \times 100 \times 3$ 或 $500 \times 500 \times 3$）。

---

### 三、总代价函数（Total Cost Function）

总代价函数由两部分组成：

$$
J(G) = \alpha \cdot J_{\text{content}}(C, G) + \beta \cdot J_{\text{style}}(S, G)
$$

其中：

- $J_{\text{content}}(C, G)$：**内容代价函数**，衡量 $G$ 与 $C$ 在内容上的相似度
- $J_{\text{style}}(S, G)$：**风格代价函数**，衡量 $G$ 与 $S$ 在风格上的相似度
- $\alpha > 0$、$\beta > 0$：**超参数**，控制内容与风格的相对权重

> 💡 虽然理论上可以用一个比例参数（如 $\lambda = \beta / \alpha$），但原始论文 [Gatys et al., 2015] 使用两个独立超参数，课程遵循此设定。

---

### 四、优化过程

使用**梯度下降法**更新生成图像 $G$：

$$
G := G - \eta \cdot \frac{\partial J(G)}{\partial G}
$$

其中：

- $\eta$ 是学习率
- 梯度 $\frac{\partial J(G)}{\partial G}$ 通过反向传播计算（利用预训练 CNN 的中间层激活值）

> 注意：**CNN 权重固定不变**，只更新输入图像 $G$ 的像素值。

---

### 五、算法流程概览

1. **输入**：内容图 $C$，风格图 $S$
2. **初始化**：生成图 $G \leftarrow \text{随机噪声}$
3. **重复以下步骤直至收敛**：

   - 前向传播 $C, S, G$ 通过预训练 CNN（如 VGG-19）
   - 计算 $J_{\text{content}}(C, G)$ 和 $J_{\text{style}}(S, G)$
   - 计算总代价 $J(G)$
   - 反向传播计算 $\frac{\partial J(G)}{\partial G}$
   - 更新 $G$ 的像素值
4. **输出**：风格迁移后的图像 $G$

---

### 六、关键参考文献

本方法基于以下开创性论文：

> **Leon A. Gatys, Alexander S. Ecker, Matthias Bethge**  
> *A Neural Algorithm of Artistic Style*  
> arXiv:1508.06576 [cs.CV], 2015  
> 🔗 https://arxiv.org/abs/1508.06576

该论文首次提出使用 CNN 的**中间层特征**分别建模“内容”与“风格”，并成功实现高质量艺术风格迁移。

---

### 七、后续内容预告

- 如何具体定义 $J_{\text{content}}$？ → 使用某一层（如 conv4_2）的激活值的平方误差
- 如何定义 $J_{\text{style}}$？ → 使用**Gram 矩阵（Gram Matrix）**  衡量特征通道间的相关性，捕捉纹理与风格

---

✅ **总结一句话**：  
神经风格迁移通过**联合优化内容相似性与风格相似性**的代价函数，直接在像素空间中迭代生成兼具指定内容与风格的新图像，其核心思想建立在深度 CNN 的层次化表征能力之上。
