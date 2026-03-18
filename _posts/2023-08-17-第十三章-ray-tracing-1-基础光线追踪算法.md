---
title: "第十三章：Ray tracing 1 基础光线追踪算法"
date: 2023-08-17 19:51:18
categories:
  - 图形学
---

### Ray Tracing（光线追踪）

光栅化的问题

不能很好地处理全局效果，比如：

1.软阴影

2.光线多次弹射（毛玻璃反射、间接反射）

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204242-i9pwn8h.png)

光栅化速度快，但是质量低；光线追踪准确但是速度慢。所以一般光栅化用于实时、光线追踪用于离线。

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204247-44w81g6.png)

### 基础的光线追踪算法

光线的三种假设：

1. 光沿直线传播（实际并不是
2. 光线相交时不会碰撞（实际并不是
3. 光线从光源传播到眼睛（根据光的可逆性，光线以不变的光路从眼睛传播到光源

   ![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204252-rq5ma55.png)

#### 光线投射

从摄像机到成像平面中的一个像素连成一条射线，这个射线会和一个物体相交。再把这个交点和光源做连线，判定他是不是在阴影内。

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204310-a4f3y6w.png)

假设眼睛是一个点（针孔摄像机），同时光源也是点光源。光线打到场景中的物体会发生完美的折射和反射。

考虑光线和场景中的物体最近的交点，交点再与光源做连线，如果没有遮挡说明没有阴影。求出交点后再对交点做着色。

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204315-s1c6ju9.png)![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204318-s2voec6.png)

#### recursive Whitted-styled ray tracing

光线可以不断弹射。每个弹射点可画出一条 shadow ray，计算是不是在阴影里，光源弹射后能照到的点的着色值最后都加到一个像素里面去。

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204323-6as49bq.png)

求光线和物体表面的交点

光线的定义：

点光源：起点 + 方向

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204331-pl418gg.png)

判断光线与球体是否相交：

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204335-tljx4px.png)

点即在光线上，又在球上。求解一元二次方程：

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204341-vxbltr2.png)

对于一般面

隐式表示：列出 surface 的隐式表达式，带入求解：

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204347-seawpb6.png)

显式表示：求光线与三角面的交点

一个封闭的图形，从图形内部发出一条射线，与图形的边的交点数量一定是奇数

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204352-8swig07.png)

简单做法：一个一个三角形求解是否与射线相交

先光线与平面求交，再判断一个点是不是在三角形内部

用一个法线和一个平面上的点可以确定一个平面

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204358-d9252z6.png)

Möller Trumbore Algorithm

快速求交点（重心坐标表示平面上的点）

解线性方程组

b1 和 b2 必须是非负的

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204403-w9iwkuw.png)

加速光线和表面求交

如果每个三角形都求一次交点很花时间

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204408-5knulr8.png)

用包围盒进行加速<br />如果一个光线碰不到包围盒，那就不可能碰到包围和里的物体

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204414-yjgij4p.png)

三个对面形成的交集

通常用 AABB 包围盒（和坐标轴平行的包围盒）

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204421-i2ydbcm.png)

2D 求交：

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204426-nic5gz7.png)

3D 求交：
3D 下有三组对面，光线要进入所有面才算进入盒子，只要出了一个面就认为出了盒子

3D 对三组对面分别计算进入的最大时间和出来的最小时间

如果出来时间小于进入时间，说明光线在盒子里待了一段时间

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204432-5woyi5w.png)

离开时间小于 0，说明盒子在光线背后

如果进入时间小于 0，说明起点在盒子里

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204437-seo5rd8.png)

用 AABB 包围盒更好计算

![image](/assets/img/posts/第十三章-ray-tracing-1-基础光线追踪算法/image-20230817204441-g7rxglt.png)
