---
title: "07 梯度下降（Gradient Descent）"
date: 2025-12-08T07:34:27Z
last_modified_at: 2025-12-08T07:38:23Z
categories:
  - 神经网络与深度学习
math: true
---

### 1. 背景回顾：逻辑回归与目标

　　在逻辑回归中，我们希望学习参数 **权重向量** **$\mathbf{w}$** 和 **偏置** **$b$**，使得模型对训练数据的预测尽可能准确。

- 对于第 $i$ 个样本，模型输出为：

  $$
  \hat{y}^{(i)} = \sigma(\mathbf{w}^\top \mathbf{x}^{(i)} + b)
  $$

  其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 sigmoid 激活函数。
- 单个样本的 **损失函数（Loss Function）**  定义为：

  $$
  \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) = -\left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]
  $$
- 整个训练集（共 $m$ 个样本）的 **成本函数（Cost Function）**  为平均损失：

  $$
  J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
  $$

> ✅ **目标**：找到 $\mathbf{w}$ 和 $b$，使得 $J(\mathbf{w}, b)$ 最小。

---

### 2. 成本函数的性质：凸性（Convexity）

- 逻辑回归所使用的上述成本函数 $J(\mathbf{w}, b)$ 是一个 **凸函数（convex function）** 。
- 凸函数的图像像一个“碗”，**只有一个全局最小值**，没有局部极小值。
- 这一性质保证了：**无论从哪里初始化参数，梯度下降最终都会收敛到全局最优解**（或非常接近）。

> 💡 因此，逻辑回归通常将 $\mathbf{w} = \mathbf{0}$, $b = 0$ 作为初始值（无需随机初始化）。

---

### 3. 梯度下降算法原理

　　梯度下降是一种迭代优化算法，通过不断沿 **成本函数下降最快的方向** 更新参数，逐步逼近最小值。

#### 3.1 直观理解

- 在参数空间 $(\mathbf{w}, b)$ 中，$J(\mathbf{w}, b)$ 构成一个曲面。
- 梯度下降从某个初始点出发，每一步都朝 **最陡下降方向** 移动。
- 步长由 **学习率（learning rate）****$\alpha$** 控制。

#### 3.2 一维简化示例（仅含 $w$）

　　假设 $J(w)$ 是单变量函数，则梯度下降更新规则为：

$$
w := w - \alpha \cdot \frac{dJ(w)}{dw}
$$

- 若导数 $\frac{dJ}{dw} > 0$，则 $w$ 减小（向左移动）；
- 若导数 $\frac{dJ}{dw} < 0$，则 $w$ 增大（向右移动）；
- 最终趋向最小值点。

> 🔍 导数 $\frac{dJ}{dw}$ 表示函数在该点的**斜率**，决定了下降方向。

---

### 4. 逻辑回归中的梯度下降（多参数情况）

　　由于 $J(\mathbf{w}, b)$ 同时依赖 $\mathbf{w}$ 和 $b$，需对每个参数分别求偏导并更新：

#### 参数更新规则：

$$
\begin{aligned}
\mathbf{w} &:= \mathbf{w} - \alpha \cdot \frac{\partial J(\mathbf{w}, b)}{\partial \mathbf{w}} \\
b &:= b - \alpha \cdot \frac{\partial J(\mathbf{w}, b)}{\partial b}
\end{aligned}
$$

> ⚠️ 注意：因为 $J$ 是多变量函数，使用 **偏导数（partial derivative）**  符号 $\partial$，而非普通导数 $d$。  
> 但在实际编程和直觉理解中，可将其视为“对某个参数的斜率”。

#### 编程实现约定：

- 在代码中，通常用变量名表示导数：

  - `dw` 表示 $\frac{\partial J}{\partial \mathbf{w}}$
  - `db` 表示 $\frac{\partial J}{\partial b}$
- 更新语句写作：

  ```python
  w = w - alpha * dw
  b = b - alpha * db
  ```

---

### 5. 关于导数与微积分的说明

- 即使你**不熟悉微积分**，也可以有效使用神经网络。
- 课程后续会提供**直观的导数理解**（如“斜率”、“变化率”），无需深入数学推导。
- 核心思想：**导数告诉我们当前参数应往哪个方向调整，才能更快降低成本**。

---

### 6. 总结：梯度下降的核心步骤

1. **初始化**：设 $\mathbf{w} = \mathbf{0}$, $b = 0$（逻辑回归常用）。
2. **重复以下步骤直到收敛**：

    - 计算成本函数 $J(\mathbf{w}, b)$
    - 计算偏导数 $\frac{\partial J}{\partial \mathbf{w}}$ 和 $\frac{\partial J}{\partial b}$
    - 更新参数：

      $$
      \mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \frac{\partial J}{\partial \mathbf{w}}, \quad
      b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}
      $$
3. **输出**：学到的最优参数 $\mathbf{w}^*, b^*$

---

### 附：符号说明

|符号|含义|
| ------| --------------------------|
|$\mathbf{x}^{(i)}$|第 $i$ 个输入特征向量|
|$y^{(i)}$|第 $i$ 个真实标签（0 或 1）|
|$\hat{y}^{(i)}$|模型预测输出|
|$J(\mathbf{w}, b)$|成本函数（整体误差）|
|$\alpha$|学习率（控制步长）|
|$\frac{\partial J}{\partial \mathbf{w}}$|成本对权重的偏导（梯度）|
|$\frac{\partial J}{\partial b}$|成本对偏置的偏导|

---

　　✅ **关键洞见**：  
梯度下降不是魔法，而是一个**基于局部斜率信息进行迭代修正**的过程。只要成本函数是凸的（如逻辑回归），它就能可靠地找到最优解。
