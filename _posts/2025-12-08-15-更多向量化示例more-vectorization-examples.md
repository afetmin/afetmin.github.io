---
title: "15 更多向量化示例（More Vectorization Examples）"
date: 2025-12-08T19:47:48Z
last_modified_at: 2025-12-08T19:48:06Z
categories:
  - 神经网络与深度学习
math: true
---

### 一、核心思想：避免显式 for 循环

> **黄金法则（Rule of Thumb）** ：  
> 在编写神经网络或回归模型时，**尽可能避免使用显式的 for 循环**。  
> 虽然有时无法完全消除循环，但只要能用 NumPy 等库的**内置向量化函数**替代，代码速度通常会显著提升。

　　原因：

- Python 的 for 循环在解释执行下效率低；
- NumPy 底层由 C/C++ 实现，支持 SIMD（单指令多数据）并行计算；
- 向量化代码更简洁、可读性更强。

---

### 二、向量化示例 1：矩阵-向量乘法

#### ❌ 非向量化实现（双重 for 循环）

　　设 $\mathbf{A} \in \mathbb{R}^{n \times m}$，$\mathbf{v} \in \mathbb{R}^{m}$，目标是计算：

$$
\mathbf{u} = \mathbf{A} \mathbf{v}, \quad \text{其中} \quad u_i = \sum_{j=1}^{m} A_{ij} v_j
$$

　　Python 伪代码：

```python
u = np.zeros((n, 1))
for i in range(n):
    for j in range(m):
        u[i] += A[i][j] * v[j]
```

#### ✅ 向量化实现（一行代码）

```python
u = np.dot(A, v)  # 或 A @ v
```

- 消除了两层嵌套循环；
- 利用 BLAS（基础线性代数子程序）高效计算。

---

### 三、向量化示例 2：逐元素函数（Element-wise Functions）

　　假设已有向量 $\mathbf{v} = [v_1, v_2, \dots, v_n]^\top$，想计算：

$$
\mathbf{u} = \left[ e^{v_1}, e^{v_2}, \dots, e^{v_n} \right]^\top
$$

#### ❌ 非向量化实现

```python
u = np.zeros_like(v)
for i in range(len(v)):
    u[i] = np.exp(v[i])
```

#### ✅ 向量化实现

```python
u = np.exp(v)
```

#### 其他常用 NumPy 向量化函数：

|操作|NumPy 函数|说明|
| -------------------| ------------| ---------------------|
|对数|`np.log(v)`|逐元素 $\log(v_i)$|
|绝对值|`np.abs(v)`|逐元素 $|
|最大值（与0比较）|`np.maximum(v, 0)`|ReLU 激活函数|
|平方|`v ** 2`|逐元素 $v_i^2$|
|倒数|`1 / v`|逐元素 $1/v_i$（注意除零）|

> 💡 **提示**：当你想写 for 循环时，先查 NumPy 文档——很可能已有对应函数！

---

### 四、应用：逻辑回归梯度下降的向量化

　　回顾逻辑回归的梯度计算（训练集大小为 $m$，特征数为 $n_x$）：

　　原始非向量化版本包含**两个 for 循环**：

1. 外层：遍历每个训练样本 $i = 1, \dots, m$
2. 内层：遍历每个特征 $j = 1, \dots, n_x$，更新 $dw_j$

#### ❌ 原始代码结构（简化版）

```python
dw1 = 0; dw2 = 0  # 假设 nx=2
for i in range(m):
    dz = ...  # 计算 dz(i)
    dw1 += x1[i] * dz
    dw2 += x2[i] * dz
dw1 /= m; dw2 /= m
```

　　→ 若 $n_x$ 很大，需写 $n_x$ 个变量和循环！

#### ✅ 第一步向量化：将 dw 变成向量

```python
dw = np.zeros((nx, 1))  # 初始化为向量
for i in range(m):
    dz_i = ...          # 标量
    x_i = X[:, i:i+1]   # 第 i 个样本，形状 (nx, 1)
    dw += x_i * dz_i    # 向量加法，自动广播
dw /= m
```

　　✅ **效果**：

- 消除了内层对特征的 for 循环；
- 保留外层对样本的循环（$m$ 次）；
- 代码更通用（适用于任意 $n_x$）。

---

### 五、展望：完全向量化（下一讲预告）

> 即使外层对训练样本的循环，也可以被消除！

　　通过将整个训练集 $\mathbf{X} \in \mathbb{R}^{n_x \times m}$ 和标签 $\mathbf{Y} \in \mathbb{R}^{1 \times m}$ 一次性输入，利用矩阵运算，**完全不用任何 for 循环**即可完成前向传播与梯度计算。

　　例如：

- 预测值：$\mathbf{A} = \sigma(\mathbf{W}^\top \mathbf{X} + b)$
- 损失梯度：$d\mathbf{W} = \frac{1}{m} \mathbf{X} (\mathbf{A} - \mathbf{Y})^\top$

　　这将在下一节课中详细展开。

---

### ✅ 总结要点

|主题|关键点|
| ------| ----------------------------------------------|
|**向量化优势**|更快、更简洁、更接近数学表达|
|**避免 for 循环**|尤其是嵌套循环；优先使用 NumPy 内置函数|
|**矩阵乘法**|用 `np.dot` 或 `@` 替代双重循环|
|**逐元素操作**|`np.exp`, `np.log`, `**`, `/` 等天然支持向量|
|**逻辑回归优化**|将参数 $dw$ 从标量列表变为向量，消除特征维度循环|
|**终极目标**|完全向量化：一次处理整个训练集，零 for 循环|
