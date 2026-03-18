---
title: "08 注意力模型（Attention Model）"
date: 2026-03-17T08:07:44+08:00
last_modified_at: 2026-03-17T08:07:57+08:00
categories:
  - 序列模型与注意力机制
math: true
---

## 这一节在讲什么

　　这一节把上一节的直觉正式写成公式，说明注意力模型到底怎么算。

## 第一步：先用双向 RNN 编码输入

　　假设输入句子长度是 $T_x$。

　　对每个输入位置 $t'$，双向 RNN 会得到一个表示向量：

$$
a^{\langle t' \rangle}
$$

　　你可以把它理解成：

> 第 $t'$ 个词在结合了左边和右边上下文之后的“上下文特征”。

　　所以输入句子不是被压成一个固定向量，而是保留成一整串特征：

$$
a^{\langle 1 \rangle}, a^{\langle 2 \rangle}, \dots, a^{\langle T_x \rangle}
$$

## 第二步：生成第 $t$ 个输出词时，先算注意力权重

　　注意力权重写成：

$$
\alpha^{\langle t, t' \rangle}
$$

　　它表示：

> 生成第 $t$ 个输出词时，要对第 $t'$ 个输入位置放多少注意力。

　　这些权重满足：

$$
\alpha^{\langle t, t' \rangle} \ge 0
$$

　　并且对固定的 $t$ 来说：

$$
\sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} = 1
$$

　　也就是说，它们是一组概率分布。

## 第三步：用加权求和得到上下文向量

　　第 $t$ 个时间步的上下文向量定义为：

$$
C^{\langle t \rangle} = \sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}
$$

　　这就是注意力的核心。

　　含义是：

- 如果某个输入位置很重要，它的权重就大
- 它对应的特征就会对 $C^{\langle t \rangle}$ 贡献更大

　　所以 $C^{\langle t \rangle}$ 就是“当前输出步骤最该参考的输入摘要”。

## 第四步：这个上下文再送进解码器

　　解码器在第 $t$ 步生成输出时，会综合：

- 上一步的隐藏状态 $s^{\langle t-1 \rangle}$
- 当前上下文向量 $C^{\langle t \rangle}$
- 之前已生成的词

　　然后输出当前词 $y^{\langle t \rangle}$。

## 关键问题：注意力权重怎么得到

　　课程里先定义一个未归一化分数：

$$
e^{\langle t, t' \rangle} = \mathrm{score}(s^{\langle t-1 \rangle}, a^{\langle t' \rangle})
$$

　　这个 `score` 可以由一个小型神经网络来学习。

　　直观理解：

- 你当前已经翻到哪里了，由 $s^{\langle t-1 \rangle}$ 体现
- 输入第 $t'$ 个位置包含什么信息，由 $a^{\langle t' \rangle}$ 体现
- 小网络判断“当前该不该重点看这里”

　　然后再用 softmax 归一化：

$$
\alpha^{\langle t, t' \rangle}
=
\frac{\exp(e^{\langle t, t' \rangle})}{\sum_{k=1}^{T_x} \exp(e^{\langle t, k \rangle})}
$$

　　这样就保证所有权重非负且总和为 1。

## 把整个流程串起来

　　对于每个输出时间步 $t$：

1. 看上一时刻状态 $s^{\langle t-1 \rangle}$
2. 对所有输入位置 $t'$ 算匹配分数 $e^{\langle t, t' \rangle}$
3. 经 softmax 得到注意力权重 $\alpha^{\langle t, t' \rangle}$
4. 加权求和得到上下文向量 $C^{\langle t \rangle}$
5. 用解码器生成当前输出词

## 用课程里的翻译例子体会

　　输入：

$$
\text{Jane visite l'Afrique en septembre}
$$

　　输出：

$$
\text{Jane visits Africa in September}
$$

　　当生成 `Africa` 时，模型通常会让对应 `l'Afrique` 的权重更大。

　　也就是说，注意力矩阵会学出一种近似“对齐关系”：

- `Jane` 对 `Jane`
- `visits` 对 `visite`
- `Africa` 对 `l'Afrique`
- `September` 对 `septembre`

## 课程提到的应用扩展

　　这套想法不只用于机器翻译，还能用于：

- 图像描述：生成某个词时只看图像中相关区域
- 日期标准化：把各种日期写法统一成标准格式

　　例如：

- 输入：`July 20 1969`
- 输出：`1969-07-20`

　　注意力会帮助模型在输出年份、月份、日期时去看输入里对应的位置。

## 关于计算代价怎么理解

　　注意力会对“每个输出位置”和“每个输入位置”的组合都打分，所以计算量会明显上升。

　　直观上说：

- 输出长度越长
- 输入长度越长
- 要计算的对齐关系就越多

　　但在句子长度不太夸张时，这个代价通常是值得的。

## 这一节最该记住的公式

### 注意力分数

$$
e^{\langle t, t' \rangle} = \mathrm{score}(s^{\langle t-1 \rangle}, a^{\langle t' \rangle})
$$

### 注意力权重

$$
\alpha^{\langle t, t' \rangle}
=
\frac{\exp(e^{\langle t, t' \rangle})}{\sum_{k=1}^{T_x} \exp(e^{\langle t, k \rangle})}
$$

### 上下文向量

$$
C^{\langle t \rangle} = \sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}
$$

## 这一节最该记住的要点

### 要点 1：输入不再压成一个固定向量

　　而是保留整串编码特征。

### 要点 2：每个输出步都会重新计算注意力分布

　　所以关注点是动态变化的。

### 要点 3：上下文向量是加权和

　　它是当前输出步需要参考的信息摘要。

### 要点 4：注意力权重由小神经网络学习出来

　　不是人工指定的。

## 这一节一句话总结

　　注意力模型的数学本质，就是先对每个输入位置编码，再在每个输出时间步上计算一组注意力权重，用加权和形成当前上下文，最后据此生成输出词。
