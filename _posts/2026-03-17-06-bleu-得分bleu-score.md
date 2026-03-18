---
title: "06 BLEU 得分（Bleu Score）"
date: 2026-03-17T08:06:48+08:00
last_modified_at: 2026-03-17T08:07:22+08:00
categories:
  - 序列模型与注意力机制
math: true
---

## 这一节在讲什么

　　机器翻译有一个麻烦：

> 同一句法语，可能有好几种都正确的英文翻译。

　　所以不能像分类任务那样，只看“是否完全等于标准答案”。

　　BLEU 就是为了解决这个问题而设计的自动评价指标。

## 为什么机器翻译不好评估

　　课程里的例子：

　　法语句子：

$$
\text{Le chat est sur le tapis}
$$

　　参考翻译 1：

$$
\text{The cat is on the mat}
$$

　　参考翻译 2：

$$
\text{There is a cat on the mat}
$$

　　这两句都很好。

　　所以如果机器输出的不是其中某一句的逐字复制，也不能立刻说它错了。

## 一个错误但直觉的想法：只看词是否出现过

　　假设机器输出：

$$
\text{the the the the the the the}
$$

　　如果你只看“这些词是否出现在参考里”，那每个 `the` 都确实出现在参考中。

　　于是会得到一个很离谱的高分。

　　这显然不合理。

## 所以 BLEU 用“截断计数”

　　做法是：

- 某个词在机器输出里出现了很多次
- 但它最多只能按参考翻译中出现的最大次数计分

　　上面的例子里，`the` 在参考中最多出现 2 次，
所以机器虽然输出了 7 次 `the`，最多也只记 2 次有效。

　　因此改良后的一元精确率（unigram precision）是：

$$
\frac{2}{7}
$$

　　而不是 $\frac{7}{7}$。

## 还要看词组，而不只是单词

　　BLEU 不只看单词，还看连续词组，也叫 `n-gram`。

- 1 个词叫 unigram
- 2 个词连着叫 bigram
- 3 个词连着叫 trigram
- 以此类推

　　因为只看单词不够。

　　比如：

$$
\text{The cat the cat on the mat}
$$

　　这里虽然很多单词都对，但词序明显不自然。

　　如果看 bigram，就能发现：

- `the cat` 还行
- `cat the` 就很奇怪

　　这比只看单词更像“真正的翻译质量”。

## 课程里的 bigram 例子

　　机器输出：

$$
\text{The cat the cat on the mat}
$$

　　它的 bigram 有：

- `the cat`
- `cat the`
- `the cat`
- `cat on`
- `on the`
- `the mat`

　　对每个 bigram，也做“截断计数”。

　　课程里最后算出的改良 bigram 精确率是：

$$
\frac{4}{6} = \frac{2}{3}
$$

## 一般形式怎么写

### 改良后一元精确率

$$
P_1 = \frac{\sum_{\text{unigram} \in \hat y} \mathrm{Count}_{\mathrm{clip}}(\text{unigram})}{\sum_{\text{unigram} \in \hat y} \mathrm{Count}(\text{unigram})}
$$

### 改良后 $n$ 元精确率

$$
P_n = \frac{\sum_{\text{n-gram} \in \hat y} \mathrm{Count}_{\mathrm{clip}}(\text{n-gram})}{\sum_{\text{n-gram} \in \hat y} \mathrm{Count}(\text{n-gram})}
$$

　　其中 $\hat y$ 是机器生成的翻译。

## 最终 BLEU 怎么合成

　　课程的意思是：

- 先算 $P_1, P_2, P_3, P_4$
- 再把它们组合起来
- 还要乘一个简短惩罚因子 `BP`

　　标准写法通常是：

$$
\mathrm{BLEU} = \mathrm{BP} \cdot \exp\left(\frac{1}{4}\sum_{n=1}^{4} \log P_n\right)
$$

　　这里的 `BP` 是 brevity penalty，简短惩罚。

## 为什么需要简短惩罚 BP

　　如果系统只输出特别短的句子，反而可能更容易拿到高精确率。

　　比如它只输出几个非常常见、肯定在参考里出现的词，分数可能不低。

　　但这种系统显然没真正翻译好。

　　所以 BLEU 会对“过短输出”额外惩罚。

## 小白可以怎么理解 BLEU

　　它不是在问：

> “你和标准答案一模一样吗？”

　　它在问：

> “你生成的句子，在用词和局部短语上，和一个或多个人工参考翻译有多像？”

　　所以 BLEU 是“近似自动批改机器翻译”的办法。

## 这一节最该记住的要点

### 要点 1：机器翻译经常不止一个正确答案

　　所以不能只用完全匹配来评价。

### 要点 2：BLEU 核心看的是 n-gram 匹配程度

　　不只看单词，还看连续词组。

### 要点 3：要用截断计数

　　防止输出 `the the the the...` 这种垃圾句子骗高分。

### 要点 4：要加简短惩罚

　　防止系统总输出过短句子。

### 要点 5：BLEU 是很重要的单一数值指标

　　它让不同模型之间能快速比较。

## 这一节一句话总结

　　BLEU 通过比较机器输出和多个参考翻译在 unigram、bigram 等局部片段上的匹配程度，并加入截断计数和简短惩罚，给机器翻译提供了一个非常实用的自动评估分数。
