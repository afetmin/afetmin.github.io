---
title: "03 词嵌入的特性（Properties of Word Embeddings）"
date: 2026-03-17T07:56:00+08:00
last_modified_at: 2026-03-17T07:56:13+08:00
categories:
  - 自然语言处理与词嵌入
math: true
---

## 这一节在讲什么

　　这节课讲词嵌入一个非常有名的性质：

> 向量之间的差，能编码某些语义关系。

　　最著名的例子就是类比推理：

$$
\text{man : woman :: king : queen}
$$

## 为什么这很神奇

　　如果词向量真的学得好，那么：

- `man -> woman` 这条变化方向
- `king -> queen` 这条变化方向

　　应该很相似。

　　用向量写，就是：

$$
e_{\text{man}} - e_{\text{woman}} \approx e_{\text{king}} - e_{\text{queen}}
$$

　　或者等价地写成：

$$
e_{\text{queen}} \approx e_{\text{king}} - e_{\text{man}} + e_{\text{woman}}
$$

## 类比推理怎么做成算法

　　课程里的核心公式是：

$$
\arg\max_w \; \mathrm{sim}\left(e_w, \; e_{\text{king}} - e_{\text{man}} + e_{\text{woman}}\right)
$$

　　意思是：

1. 先算出目标向量

   $$
   v = e_{\text{king}} - e_{\text{man}} + e_{\text{woman}}
   $$
2. 再在词表里找一个词 $w$
3. 让它的词向量 $e_w$ 最接近这个 $v$

　　如果训练得好，找到的词就是 `queen`。

## 余弦相似度为什么常用

　　课程里重点讲了相似度函数，最常见的是余弦相似度：

$$
\mathrm{sim}(u,v)=\frac{u^T v}{\lVert u \rVert_2 \lVert v \rVert_2}
$$

　　它其实就是两个向量夹角的余弦：

$$
\mathrm{sim}(u,v)=\cos(\theta)
$$

　　直觉上：

- 夹角越小，越相似
- 夹角接近 0 度，相似度接近 1
- 夹角接近 90 度，相似度接近 0
- 方向完全相反，相似度接近 -1

## 课程里的更多类比例子

　　除了 `man : woman :: king : queen`，课程还提到很多典型关系：

- `boy : girl`
- `Ottawa : Canada :: Nairobi : Kenya`
- `big : bigger :: tall : taller`
- `Yen : Japan :: Ruble : Russia`

　　这说明词嵌入学到的不是单个词本身，而是词与词之间的关系结构。

## 这里要小心一个误区

　　课程也提醒了一点：

　　t-SNE 图里你看到的点和箭头只是为了可视化，  
二维图中的“平行四边形”关系不一定完全保真。

　　真正的类比关系发生在原始高维空间里，而不是 t-SNE 压缩后的 2D 图里。

## 这个性质是不是百分百可靠

　　不是。

　　课程里也说了，这类类比任务的准确率大约可能在 30% 到 75% 左右，具体依赖训练细节和数据。

　　所以这不是魔法，也不是完美规则。

　　但它非常有启发性，因为它说明：

> 词向量空间里真的会出现可解释的语义方向。

## 小白怎么理解“向量差表示关系”

　　你可以把词向量想成地图上的点。

- `man -> woman` 是往“更女性化”的方向走一步
- `king` 如果也沿着这个方向走一步，就会走到 `queen`

　　所以这里重要的不是点本身，而是“从一个词走到另一个词的方向”。

## 这一节最该记住的公式

### 类比关系

$$
e_{\text{man}} - e_{\text{woman}} \approx e_{\text{king}} - e_{\text{queen}}
$$

### 等价写法

$$
e_{\text{queen}} \approx e_{\text{king}} - e_{\text{man}} + e_{\text{woman}}
$$

### 搜索最像的词

$$
\arg\max_w \; \mathrm{sim}\left(e_w, \; e_{\text{king}} - e_{\text{man}} + e_{\text{woman}}\right)
$$

### 余弦相似度

$$
\mathrm{sim}(u,v)=\frac{u^T v}{\lVert u \rVert_2 \lVert v \rVert_2}
$$

## 这一节最该记住的要点

### 要点 1：词嵌入不仅表示词，还能表示词间关系

　　这是它特别强的地方。

### 要点 2：很多语义关系会在向量空间中表现为相似方向

　　这就是类比推理的基础。

### 要点 3：余弦相似度是衡量向量相似性的常用方法

　　它关心的是方向，而不只是长度。

### 要点 4：t-SNE 图只是可视化辅助

　　真正的几何结构在高维空间里。

## 这一节一句话总结

　　词嵌入最迷人的地方在于，词与词之间的语义关系也会在向量空间里表现成稳定方向，因此像 `king - man + woman ≈ queen` 这样的类比推理才会成为可能。
