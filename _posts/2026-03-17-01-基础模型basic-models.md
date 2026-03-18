---
title: "01 基础模型（Basic Models）"
date: 2026-03-17T08:03:41+08:00
last_modified_at: 2026-03-17T08:03:52+08:00
categories:
  - 序列模型与注意力机制
math: true
---

## 这一节在讲什么

　　这一节讲的是最基础的 `seq2seq`，也就是“输入一个序列，输出另一个序列”。最典型的例子就是：

- 输入法语句子：`Jane visite l'Afrique en septembre.`
- 输出英语句子：`Jane is visiting Africa in September.`

　　课程想说明的是：只要你能把输入句子先“压缩理解”成一个向量，再让另一个网络根据这个向量一个词一个词地往外写，就能做机器翻译。

## 先把问题想简单

　　假设输入句子有 $T_x$ 个词，输出句子有 $T_y$ 个词。

　　我们把输入记成：

$$
x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \dots, x^{\langle T_x \rangle}
$$

　　把输出记成：

$$
y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, \dots, y^{\langle T_y \rangle}
$$

　　这里的意思非常朴素：

- $x^{\langle 1 \rangle}$ 是输入句子的第 1 个词
- $x^{\langle 2 \rangle}$ 是第 2 个词
- $y^{\langle 1 \rangle}$ 是输出句子的第 1 个词
- 以此类推

## 模型分成两半：编码器 + 解码器

### 1. 编码器做什么

　　编码器（Encoder）通常是一个 RNN，也可以是 GRU 或 LSTM。

　　它会把输入句子一个词一个词读进去：

$$
x^{\langle 1 \rangle} \rightarrow x^{\langle 2 \rangle} \rightarrow \cdots \rightarrow x^{\langle T_x \rangle}
$$

　　读完后，得到一个向量 $c$，你可以把它理解为：

> “这整个句子的浓缩理解结果”。

　　写成概念式就是：

$$
c = \mathrm{Encoder}(x^{\langle 1 \rangle}, \dots, x^{\langle T_x \rangle})
$$

### 2. 解码器做什么

　　解码器（Decoder）接收这个向量 $c$，然后开始一个词一个词往外生成翻译：

$$
y^{\langle 1 \rangle} \rightarrow y^{\langle 2 \rangle} \rightarrow \cdots \rightarrow y^{\langle T_y \rangle}
$$

　　每生成一个词，都会把这个词再喂回去，帮助生成下一个词。一直到输出句子结束标记 `<EOS>`，就停止。

## 小白怎么理解这件事

　　你可以把它想成两个人配合：

- 第一个人只负责“听完法语句子，记住意思”
- 第二个人只负责“根据记住的意思，重新说成英语”

　　第一个人就是编码器。  
第二个人就是解码器。

## 课程里的核心例子

### 例子 1：机器翻译

　　输入：

$$
\text{Jane visite l'Afrique en septembre.}
$$

　　输出：

$$
\text{Jane is visiting Africa in September.}
$$

　　这里最重要的不是某一个词，而是“整个句子的语义关系”被编码器先抓住了，然后解码器再按英语习惯表达出来。

### 例子 2：图像描述

　　课程还讲了一个非常像的任务：

- 输入不是一句话，而是一张图
- 输出是一句描述图像的话

　　例如一张猫坐在椅子上的图，输出：

$$
\text{A cat is sitting on a chair.}
$$

　　做法是：

1. 先用 CNN（课程里举的是预训练 `AlexNet`）把图片编码成一个特征向量
2. 再把这个向量交给 RNN
3. RNN 再一个词一个词生成图像描述

　　也就是说：

- 机器翻译是 `sequence -> sequence`
- 图像描述是 `image -> sequence`

　　它们骨架很像，都是“先编码，再逐步生成”。

## 为什么这个模型有效

　　因为很多任务都可以抽象成：

- 输入端先提取信息
- 输出端按顺序生成结果

　　机器翻译、图像描述、语音转文本，本质上都属于这种模式。

## 这一节最该记住的要点

### 要点 1：`seq2seq` 是两段式结构

$$
\text{输入序列} \xrightarrow{\text{Encoder}} c \xrightarrow{\text{Decoder}} \text{输出序列}
$$

### 要点 2：编码器负责“压缩理解”

　　它把一长串输入，压成一个向量 $c$。

### 要点 3：解码器负责“逐词生成”

　　它不是一次生成整句，而是每一步生成一个词。

### 要点 4：图像描述和机器翻译是同一思路

　　区别只是输入类型不同：

- 机器翻译输入是词序列
- 图像描述输入是图像特征向量

## 这一节一句话总结

　　最基础的 `seq2seq` 模型，就是先用编码器把输入整体理解成一个向量，再用解码器根据这个向量一步一步生成输出；机器翻译和图像描述都可以用这套思路。
