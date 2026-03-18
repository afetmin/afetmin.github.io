---
title: "11 一维到三维推广（1D and 3D generalizations of models）"
date: 2026-02-08T12:14:38Z
last_modified_at: 2026-02-08T12:21:15Z
categories:
  - 卷积神经网络
math: true
---

# 11 一维到三维推广（1D and 3D generalizations of models）

### 一、核心思想

卷积神经网络（ConvNets）最初为处理 **2D 图像** 而设计，但其核心机制——**局部感受野 + 权重共享 + 平移不变性**——具有高度通用性，可自然推广至 **1D 时间序列** 和 **3D 体数据（如医学影像、视频）** 。

> **关键洞察**：卷积操作的本质是 **滑动窗口特征检测器**，维度只是输入/滤波器形状的扩展。

---

### 二、2D 卷积回顾（基准）

- **输入**：$H_{\text{in}} \times W_{\text{in}} \times C_{\text{in}}$
- **滤波器（核）** ：$K_h \times K_w \times C_{\text{in}}$
- **输出尺寸（无 padding，stride=1）** ：

  $$
  H_{\text{out}} = H_{\text{in}} - K_h + 1,\quad
  W_{\text{out}} = W_{\text{in}} - K_w + 1
  $$
- 若使用 $N$ 个滤波器，则输出为：

  $$
  (H_{\text{out}} \times W_{\text{out}} \times N)
  $$

> 例：$14 \times 14 \times 3$ 输入 + $5 \times 5 \times 3$ 滤波器 ×16 → 输出 $10 \times 10 \times 16$

---

### 三、1D 卷积（适用于时间序列）

#### 应用场景

- 心电图（EKG/ECG）
- 音频信号
- 金融时间序列
- 文本嵌入序列（虽 RNN/Transformer 更主流，但 1D CNN 仍有效）

#### 数学形式

- **输入**：$L_{\text{in}} \times C_{\text{in}}$（长度 × 通道数）
- **滤波器**：$K \times C_{\text{in}}$（1D 核长 $K$）
- **输出长度**：

  $$
  L_{\text{out}} = L_{\text{in}} - K + 1
  $$
- 使用 $N$ 个滤波器 → 输出：$L_{\text{out}} \times N$

> 例：
>
> - 输入 EKG：$14 \times 1$
> - 滤波器：$5 \times 1$ ×16
> - 输出：$10 \times 16$  
>   下一层：输入 $10 \times 16$，用 $5 \times 16$ 滤波器 ×32 → 输出 $6 \times 32$

#### 优势 vs 局限

- ✅ 计算高效、并行性强、可捕获局部模式（如心跳尖峰）
- ❌ 对长程依赖建模弱于 RNN/Transformer（将在序列模型课程中对比）

---

### 四、3D 卷积（适用于体数据）

#### 应用场景

- CT / MRI 医学影像（3D 体素数据）
- 视频分析（$x, y, t$ 三维：空间 + 时间）
- 科学计算中的 3D 场数据

#### 数学形式

- **输入**：$D_{\text{in}} \times H_{\text{in}} \times W_{\text{in}} \times C_{\text{in}}$
- **滤波器**：$K_d \times K_h \times K_w \times C_{\text{in}}$
- **输出尺寸（无 padding，stride=1）** ：

  $$
  \begin{aligned}
  D_{\text{out}} &= D_{\text{in}} - K_d + 1 \\
  H_{\text{out}} &= H_{\text{in}} - K_h + 1 \\
  W_{\text{out}} &= W_{\text{in}} - K_w + 1
  \end{aligned}
  $$
- 使用 $N$ 个滤波器 → 输出：$D_{\text{out}} \times H_{\text{out}} \times W_{\text{out}} \times N$

> 例：
>
> - CT 输入：$14 \times 14 \times 14 \times 1$
> - 滤波器：$5 \times 5 \times 5 \times 1$ ×16
> - 输出：$10 \times 10 \times 10 \times 16$  
>   下一层：$5 \times 5 \times 5 \times 16$ 滤波器 ×32 → 输出 $6 \times 6 \times 6 \times 32$

#### 特点

- 捕获 **空间+时间（或深度）联合特征**
- 参数量巨大（$K^3$ 增长），需谨慎设计网络深度

---

### 五、统一视角：卷积的维度泛化

|维度|输入形状|滤波器形状|输出形状|典型应用|
| ------| ----------| ------------| ----------| -----------|
|1D|$L \times C_{\text{in}}$|$K \times C_{\text{in}}$|$(L-K+1) \times N$|EKG、音频|
|2D|$H \times W \times C_{\text{in}}$|$K_h \times K_w \times C_{\text{in}}$|$(H-K_h+1)(W-K_w+1) \times N$|图像识别|
|3D|$D \times H \times W \times C_{\text{in}}$|$K_d \times K_h \times K_w \times C_{\text{in}}$|$(D-K_d+1)(H-K_h+1)(W-K_w+1) \times N$|CT、视频|

> **共同规则**：
>
> - 滤波器在对应维度上滑动
> - 通道数必须匹配（$C_{\text{in}}^{\text{filter}} = C_{\text{in}}^{\text{input}}$）
> - 每个滤波器生成一个特征图（feature map），$N$ 个滤波器 → $N$ 通道输出

---

### 六、课程启示与延伸

1. **卷积的普适性**：CNN 不仅限于图像，而是 **任意结构化网格数据** 的通用工具。
2. **选择依据**：

   - 1D：快速提取局部时序模式
   - 2D：图像标准方案
   - 3D：需建模体数据内部结构（如器官、动作时空演化）
3. **后续课程预告**：序列模型（RNN、LSTM、GRU、Transformer）更适合处理 **长程依赖**，但 1D CNN 在某些任务（如语音关键词 spotting）中仍具竞争力。

---

### ✅ 总结公式速查

- **通用输出尺寸（stride=1, no padding）** ：

  $$
  \text{Output size} = \text{Input size} - \text{Kernel size} + 1
  $$
- **多通道卷积输出**：

  $$
  \text{Output} = \text{Conv}_{d}(\mathbf{X}; \mathbf{W}_1, \dots, \mathbf{W}_N) \in \mathbb{R}^{S_1 \times \cdots \times S_d \times N}
  $$

  其中 $d=1,2,3$，$S_i$ 为第 $i$ 维输出长度。
