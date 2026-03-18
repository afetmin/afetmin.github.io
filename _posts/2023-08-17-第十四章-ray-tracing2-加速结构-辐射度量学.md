---
title: "第十四章：Ray tracing2 加速结构、辐射度量学"
date: 2023-08-17 19:51:18
categories:
  - 图形学
---

Ray Tracing 2

- 加速结构
- 辐射度量学

题外话：

GTC（GPU Technology Conference）：

- DLSS（Deep Learning Super Sampling）2.0 [https://zhuanlan.zhihu.com/p/116211994](https://zhuanlan.zhihu.com/p/116211994)

  - 光栅化生成一个 1080p 的图，把它拉大成一张 4K 的图，结果不损失太多性能，同时看上去依然清晰
- RTXGI（全局光照）[https://developer.nvidia.com/rtxgi](https://developer.nvidia.com/rtxgi)

回顾上节课：

- 为什么要做光线追踪
- Whitted-style ray tracing（递归的光线追踪，光线弹射到多个地方，在每个交点计算着色和阴影）
- 光线和物体求交

  - 光线和隐式表面求交
  - 光线和三角形求交
- AABB 包围盒

  - 理解包围盒：坐标轴上三对相对的平板
  - 光线和 AABB 盒求交

本节课：

- AABB 盒怎么加速光线追踪？

  - 均匀的网格（Uniform grids）
  - 空间划分 （spatial partitions）
- 辐射度量学

#### AABB 加速光线追踪

**均匀的网格：Uniform Spatial Partitions（Grids）**

先让光线和盒子求交，再进一步和盒子里的物体求交

预处理：

1.先找到场景里的一个包围盒

2.把盒子分成一堆格子

3.标记与物体表面相交的格子

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204558-karsvo5.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204602-xjyn2a4.png)

怎么判断光线继续往后传播时要和哪个盒子求交，不能每个格子都求一遍。有一个简单的思路是：如果光线往右上传播，就只看当前格子右边和上面的格子，判断光线与哪个格子有交点，再把光线移动一格。光栅化一条线:[https://zhuanlan.zhihu.com/p/20213658](https://zhuanlan.zhihu.com/p/20213658)。

加速效果：<br />一个格子（格子很稀疏），基本没有加速效果

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204609-qqu5bmx.png)

格子太密，要做很多次光线和格子的求交，效率很低

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204615-miu4wws.png)

需要找一个平衡

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204619-otjwusd.png)

当几何物体在场景中分布比较均匀时，格子加速的效果比较好

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204624-l67m3mz.png)

物体在场景中分布不均匀的时候，加速的效果不好

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204632-ohvjn97.png)

#### **空间划分 Spatial Partitions**

三种空间划分的结构：<br />Oct-Tree：8 叉树，把空间分成 8 个小方块，小方块继续分割，当格子里是空的，或物体足够少，就停止分割（平面是 4 叉树，即分成几份和维度有关系）

KD-Tree：2 叉树，每次一个格子只砍一刀（水平竖直沿着坐标轴交替划分，基本保证划分的空间是均匀的）

BSP-Tree：选一个方向砍一刀

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204649-cee1sju.png)

##### KD-Tree 预处理，建立加速结构

把空间划分为二叉树：

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204658-s0bi29g.png)

数据结构：

中间节点需要存储：

- 当前节点沿着哪个坐标轴划分
- 划分在哪个位置（不一定划分在中间）
- 2 个子节点

叶子节点需要存储：

* 和格子相交的几何物体
* ![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204705-wgytx09.png)

判断光线和当前节点是不是有交点，如果有交点，继续判断和当前节点的子节点是不是有交点，一直到叶子节点，如果光线和叶子节点有交点，就求光线和叶子节点里面所有物体的交点；如果光线和当前节点没有交点就不需要继续往下找。<br />如果是求最近的交点就一边找一边记录最近的。

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204714-42teely.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204718-bz5cned.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204722-s8wc8ov.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204726-qsnpo7h.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204736-aedp5yz.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204740-rxxh5k1.png)

KD-Tree 的问题：

1.不好判断几何对象（三角形）和盒子是否有交集

2.1 个对象可能和不同的盒子都有交集，即同一个物体会被多个叶子节点存储

##### **Object Partitions &amp; Bounding Volumn Hierarchy（BVH）**

划分物体：把一个盒子里的三角形分为两部分，把两部分的三角形再重新求包围盒，然后每个包围盒继续划分，直到一个包围盒里包含的节点数够少就停止划分。

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204752-vsj137t.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204802-dw9fna8.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204806-e07iour.png)![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204812-1n2czdl.png)

BVH 的好处是：一个物体只可能出现在一个盒子里，且无需求三角形和包围盒的交点，避免了 KD-Tree 的问题。

但 BVH 也有一个问题：BVH 对空间的划分不是很严格的划分开，BoundingBox 可以相交，所以需要在划分几何形体的时候尽量减少重叠。

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204818-9pfar63.png)

总结构造 BVH 加速结构的过程：

1.找到一个包围盒

2.递归地把包围盒中的物体拆成两部分

3.重新计算包围盒

4.当包围盒的物体足够少的时候停止递归

5.把物体信息存储在叶子节点里

**怎么做节点的划分？**

* 选一个维度
* 方式 1：每次选一个最长的轴把节点分成两半，使得节点最后分布比较均匀
* 方式 2：取中间的三角形的位置把节点分为两半，是的分割后节点的三角形数量差不多，让这个树形结构两边保持平衡（深度小，平均搜索次数小）

  * 取中间的三角形涉及到排序：所有三角形取重心，沿一个轴排个序，找到中间的那个三角形。（也可以不用排序找中位数，快速选择算法，可以达到 O(n)的时间复杂度）
* 如果场景是动态的，三角形数量会变化，就需要每次变化都重新算一下 BVH。

  ![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204830-voj0qcl.png)

BVH 的数据结构：
中间节点存储：

- 包围盒
- 子节点的指针

叶子节点存储：

* 包围盒
* 实际的物体
* ![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204841-zo7ev8y.png)

BVH 算法伪代码：

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204847-q8eypig.png)

空间划分和物体划分的区别：

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204852-d3um7bh.png)

以上是 Whitted-style 光线追踪的内容。

#### 辐射度量学 Basic radiometry

为什么要学习辐射度量学？

之间的 Blinn-Phong 模型里提到光的强度 I 是怎么得到的，他的物理意义是什么？

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204902-rtdzgu2.png)

Whitted-style 光线追踪给的结果是正确的吗？

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204923-ev1qs1a.png)

辐射度量学给了一种精准定义光照的物理量的方法。

辐射度量学学习的内容：
为如何描述光照定义了一系列的方法和单位

给光定义了各种空间中的属性（仍然是基于几何光学，认为光线沿直线传播）：

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204928-25ee893.png)

- Radiant flux，intensity，irradiance，radiance

Radiant Energy and Flux

Radiant energy 是电磁辐射的能量，单位是焦耳，用符号 Q 表示。

Radiant Flux 是单位时间的能量。

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204935-zhh77mo.png)

另一种理解：单位时间通过感光平面的光子的数量

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204941-3lo9p1g.png)

光源辐射的能量：radiant Intensity，定义了方向性和能量相关的概念

物体表面接受到了多少能量：Irradiance

光线传播中的能量怎么度量：Radiance

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204947-ch8k18p.png)

**Radiant Intensity**

单位立体角上点光源辐射出的单位能量

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204952-czaovps.png)

立体角：
角度：弧长/半径

立体角：角度在三维的延伸。锥体对应的面积/求面的面积。

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817204957-l8s8bw3.png)

单位立体角（微分立体角）

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817205002-ikor5vv.png)

对单位立体角做积分可以得到整个球面

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817205007-7f67wjp.png)

在辐射度量里面通常用 ω 表示方向，ω 可以用 θ 和 φ 来定义位置，并通过 sinθdθdφ 算出单位立体角。

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817205012-k9khbxq.png)

对一个点光源，radiant intensity 是单位立体角的能量，把所有方向上的单位立体角的 intensity 积分，就可以得到它的 power，反之任何一个方向的 Intensity 就是 power/4π。

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817205018-foe19s0.png)

小知识：现代 LED 灯上标注的瓦数不是真实的，而是对应于白炽灯的瓦数，LED 实际瓦数更低。

![image](/assets/img/posts/第十四章-ray-tracing2-加速结构-辐射度量学/image-20230817205022-d5s39qt.png)
