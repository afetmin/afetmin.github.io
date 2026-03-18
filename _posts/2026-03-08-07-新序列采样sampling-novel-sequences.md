---
title: "07 新序列采样（Sampling novel sequences）"
date: 2026-03-08T10:44:07+08:00
last_modified_at: 2026-03-08T11:02:26+08:00
categories:
  - 循环序列模型
math: true
---

### 1. 新序列采样的基本流程

　　采样的目的是从模型学习到的概率分布 $P(y^{\langle t \rangle} | y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle})$ 中生成新的单词或字符序列。

- **初始化**：

  - 在第一个时间步 $t=1$，通常输入全零向量 $\mathbf{x}^{\langle 1 \rangle} = \mathbf{0}$ 或特定的开始标识（如 `<BOS>`）。
  - 网络通过 Softmax 层输出第一个词的概率分布 $\hat{\mathbf{y}}^{\langle 1 \rangle}$。
  - 根据该概率分布进行随机采样（例如使用 `np.random.choice`），得到第一个生成的词 $y^{\langle 1 \rangle}$。
- **迭代生成**：

  - 将上一步采样得到的词 $y^{\langle t-1 \rangle}$ 转换为 One-hot 编码（或嵌入向量），作为下一个时间步的输入 $\mathbf{x}^{\langle t \rangle}$。
  - 即：$\mathbf{x}^{\langle t \rangle} = y^{\langle t-1 \rangle}$。
  - 网络计算当前时间步的输出概率分布 $\hat{\mathbf{y}}^{\langle t \rangle}$，并再次采样得到 $y^{\langle t \rangle}$。
  - 重复此过程，直到满足停止条件。
- **停止条件**：

  1. **EOS 标识**：如果字典中包含句子结束标识（`<EOS>` 或 `UNK` 在此处特指结尾），当采样到该标识时停止。
  2. **固定长度**：如果没有 EOS 标识，可以设定一个最大时间步 $T_{max}$（如 20 或 100），达到该长度后强制停止。
- **处理未知标识（UNK）** ：

  - 若不希望生成 `<UNK>`，可采用**拒绝采样（Rejection Sampling）** 策略：一旦采样到 `<UNK>`，则在该步重新采样，直到得到一个非 `<UNK>` 的词为止。

### 2. 两种建模范式的对比

#### A. 基于词汇的模型 (Word-level RNN)

- **定义**：字典由完整的单词组成（如 "cat", "average"）。
- **优点**：

  - 序列较短（英语句子通常 10-20 个词），更容易捕捉长距离依赖关系。
  - 计算成本相对较低，训练速度较快。
- **缺点**：

  - 存在**未知词问题**：如果测试集中出现字典外的词（如专有名词 "Mau"），只能标记为 `<UNK>`，丢失信息。
  - 字典规模通常很大。

#### B. 基于字符的模型 (Character-level RNN)

- **定义**：字典由单个字符组成（如 'a'-'z', 'A'-'Z', '0'-'9', 空格等）。
- **优点**：

  - **无未知词问题**：任何字符串（包括 "Mau"）都可以由字符组合表示，概率非零。
  - 适用于处理大量未知文本、专有名词或特定领域术语。
- **缺点**：

  - **序列极长**：一个句子可能包含数百个字符，导致序列长度大幅增加。
  - **长距离依赖难捕捉**：由于序列变长，前面的字符对后面字符的影响更难传递。
  - **计算成本高**：训练需要更多的计算资源和时间。

### 3. 应用趋势与展望

- 目前主流的自然语言处理（NLP）任务多采用**基于词汇**的模型，因其在效率和长依赖捕捉上的平衡。
- 随着计算能力的提升，**基于字符**的模型在特定场景（如处理生僻词、多语言混合、无需分词的场景）中的应用正在增加。
- **后续挑战**：基础 RNN 在处理长序列时面临**梯度消失（Vanishing Gradient）** 问题，这限制了其捕捉长期依赖的能力。接下来的内容将引入 **GRU (Gated Recurrent Unit)**  和 **LSTM (Long Short-Term Memory)**  来解决这一问题。

### 4. 采样示例

　　笔记中展示了基于字符的模型在不同语料库训练后的生成效果：

- **新闻语料**：生成的文本具有新闻风格，但语法可能不完全通顺（例："Concussion epidemic, to be examined..."）。
- **莎士比亚语料**：生成的文本模仿了莎士比亚的句式结构和用词风格（例："The mortal moon hath her eclipse in love..."）。

---

　　**核心公式总结：**

1. **时间步** **$t$** **的输入**：

   $$
   \mathbf{x}^{\langle t \rangle} = y^{\langle t-1 \rangle}
   $$

   （其中 $y^{\langle 0 \rangle}$ 通常为 $\mathbf{0}$ 或 `<BOS>`）
2. **概率分布计算**：

   $$
   \hat{\mathbf{y}}^{\langle t \rangle} = \text{softmax}(\mathbf{W}_y \mathbf{a}^{\langle t \rangle} + \mathbf{b}_y)
   $$
3. **采样过程**：

   $$
   y^{\langle t \rangle} \sim \text{Categorical}(\hat{\mathbf{y}}^{\langle t \rangle})
   $$
