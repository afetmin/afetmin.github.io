---
title: "11双向循环神经网络（Bidirectional RNN, Bi-RNN）"
date: 2026-03-08T11:36:16+08:00
last_modified_at: 2026-03-08T11:36:27+08:00
categories:
  - 循环序列模型
math: true
---

### 1. 动机与背景

　　在标准的单向 RNN（包括标准 RNN、GRU 或 LSTM 单元）中，序列中某一点 $t$ 的预测 $\hat{y}^{\langle t \rangle}$ 仅依赖于该点之前的输入信息 $(x^{\langle 1 \rangle}, \dots, x^{\langle t \rangle})$。

- **局限性**：在某些任务中（如命名实体识别），仅靠前文无法准确判断当前词的含义。

  - **例子**：句子 "He said Teddy Roosevelt..." 中，要判断 "Teddy" 是否为人名的一部分，仅看 "He said Teddy" 是不够的（可能是 "Teddy bear"）。必须看到后面的 "Roosevelt" 才能确定。
- **目标**：构建一个模型，使得在序列的任意位置 $t$，不仅能获取过去的信息，还能获取未来的信息。

### 2. 双向 RNN 的结构原理

　　双向 RNN 通过引入两个独立的隐藏层序列来解决上述问题：

1. **前向传播层（Forward Pass）** ：

   - 使用带有向右箭头 $\rightarrow$ 的循环单元（可以是 RNN、GRU 或 LSTM）。
   - 按时间顺序从 $t=1$ 到 $t=T_x$ 计算激活值 $\overrightarrow{a}^{\langle t \rangle}$。
   - 依赖关系：$\overrightarrow{a}^{\langle t \rangle}$ 依赖于 $\overrightarrow{a}^{\langle t-1 \rangle}$ 和 $x^{\langle t \rangle}$。
2. **反向传播层（Backward Pass）** ：

   - 使用带有向左箭头 $\leftarrow$ 的循环单元。
   - 按时间逆序从 $t=T_x$ 到 $t=1$ 计算激活值 $\overleftarrow{a}^{\langle t \rangle}$。
   - 依赖关系：$\overleftarrow{a}^{\langle t \rangle}$ 依赖于 $\overleftarrow{a}^{\langle t+1 \rangle}$ 和 $x^{\langle t \rangle}$。
   - *注意*：这里的“反向”指的是时间步上的逆序计算，属于前向传播过程的一部分，而非训练时的反向传播算法（Backpropagation through time）。
3. **输出层计算**：

   - 在任意时间步 $t$，网络的预测输出 $\hat{y}^{\langle t \rangle}$ 同时结合了前向激活值和反向激活值。
   - 计算公式通常为：

     $$
     \hat{y}^{\langle t \rangle} = g\left( W_y \left[ \overrightarrow{a}^{\langle t \rangle}, \overleftarrow{a}^{\langle t \rangle} \right] + b_y \right)
     $$

     其中 $[\cdot, \cdot]$ 表示向量拼接，$g$ 为激活函数（如 softmax）。

### 3. 信息流动路径

　　以预测时间步 $t=3$ 为例：

- **过去信息**：从 $x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, x^{\langle 3 \rangle}$ 流经 $\overrightarrow{a}^{\langle 1 \rangle} \to \overrightarrow{a}^{\langle 2 \rangle} \to \overrightarrow{a}^{\langle 3 \rangle}$，最终汇入输出。
- **未来信息**：从 $x^{\langle 4 \rangle}, \dots, x^{\langle T_x \rangle}$ 流经 $\overleftarrow{a}^{\langle T_x \rangle} \to \dots \to \overleftarrow{a}^{\langle 4 \rangle} \to \overleftarrow{a}^{\langle 3 \rangle}$，最终汇入输出。
- 这使得模型在处理中间位置的单词时，拥有完整的上下文信息（Context）。

### 4. 关键特性与优缺点

#### 优点

- **上下文感知能力强**：能够利用整个序列的信息进行局部预测，显著提升了如命名实体识别（NER）、词性标注等 NLP 任务的性能。
- **通用性**：基本单元可以是标准 RNN、GRU 或 LSTM。在实际应用中，**双向 LSTM (Bi-LSTM)**  是最常用的组合，效果通常最好。

#### 缺点

- **需要完整序列**：必须等到整个输入序列 $(x^{\langle 1 \rangle}, \dots, x^{\langle T_x \rangle})$ 全部接收完毕后，才能开始计算反向层的激活值并进行预测。
- **实时性差**：对于在线语音识别等需要低延迟的场景（即边听边译），标准的双向 RNN 不适用，因为不能等待用户说完整个句子再处理。这类场景通常需要更复杂的流式模型或单向模型。

### 5. 应用场景建议

- **适用**：处理完整的文本句子，如机器翻译的编码器部分、情感分析、命名实体识别等，其中整个输入序列在预测前是可用的。
- **不适用**：对延迟敏感的实时流数据处理（除非采用特殊的截断或缓存策略）。

### 6. 总结公式

　　双向 RNN 的核心计算逻辑可以概括为：

$$
\begin{aligned}
\overrightarrow{a}^{\langle t \rangle} &= g_1\left( W_{\rightarrow} \overrightarrow{a}^{\langle t-1 \rangle} + W_{x\rightarrow} x^{\langle t \rangle} + b_{\rightarrow} \right) \\
\overleftarrow{a}^{\langle t \rangle} &= g_1\left( W_{\leftarrow} \overleftarrow{a}^{\langle t+1 \rangle} + W_{x\leftarrow} x^{\langle t \rangle} + b_{\leftarrow} \right) \\
\hat{y}^{\langle t \rangle} &= g_2\left( W_y \left[ \overrightarrow{a}^{\langle t \rangle}, \overleftarrow{a}^{\langle t \rangle} \right] + b_y \right)
\end{aligned}
$$

　　其中 $t = 1, \dots, T_x$。
