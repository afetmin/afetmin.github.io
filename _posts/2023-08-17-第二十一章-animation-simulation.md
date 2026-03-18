---
title: "第二十一章：Animation、Simulation"
date: 2023-08-17 19:51:18
categories:
  - 图形学
---

Animation/Simulation

计算机动画介绍：

- 历史
- 基本制作方法：关键帧
- 物理模拟方法
- 运动学
- rigging（绑定动画）

动画最初的定义“让东西动起来”：

- 一种交流的工具，展示动起来的东西
- 美学

对于建模或者几何的拓展

- 把 3D 模型延伸到时间的纬度

动画怎么制作

- 把很多图按顺序，按一定速度去播放（人眼有视觉暂留的效果）

虚拟现实为了让人们带头戴设备不晕，对帧率要求非常高（2 只眼睛必须达到 90fps）![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212848-injp93u.png)

动画的发展![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212854-ndj2f72.png)

电影技术的历史<br />最早电影技术用于科学研究（比如生物学研究）![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212906-dnxd3g5.png)

第一部手绘 Feature-Length 动画![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212910-465iyii.png)

第一部数字动画![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212914-ac10gz7.png)

早期计算机动画![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212918-zlqd2p3.png)

电脑生成的恐龙![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212922-zod4uuh.png)

玩具总动员（里程碑式的，整个电影都是用 CG 生成的，用光栅化的效果生成阴影）![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212927-kn2z84p.png)

关键帧

关键帧能定义动画的总的走向

10 年前（食破天惊）![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212932-tfl3o4n.png)

近年（冰雪奇缘 2）![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212936-zyspnsj.png)

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212943-ctpth6m.png)关键帧插值![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212947-ncw50fv.png)

线性插值不够连续，所以去要用到一些样条去做到更光滑的插值![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212953-s28xvi5.png)

#### 物理模拟

牛顿第二定律：F=ma，只要知道一些初始条件，比如物体的初始位置，受到的力，可以动态算出物体的位置。

物体仿真

建立正确的受力模型![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817212958-iai570w.png)

布料的模拟（模拟的不对会发生穿模的现象，比如衣服穿透了人的身体）![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213003-6bmut2k.png)

##### 流体：

分为两步：1.模拟水是怎么运动的，拿到水的形状、位置。2.把结果拿去渲染![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213008-k73xjhh.png)

##### 质点弹簧系统模拟绳子的运动![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213013-egmcdn8.png)

##### 模拟头发的运动

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213018-4fqjavn.png)

用点符号表示求导

一块布可以用很多的质点弹簧进行模拟![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213028-2371bm8.png)

简单的弹簧

理想的弹簧：a 受到的弹簧把 a 拉向 b 的力 fa-\>b = ks(b-a)，b 受到的力与 a 相反力的大小和 a、b 间的距离成正比（胡克定律），ks 是弹性系数![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213031-1gwdpty.png)

被拉伸的弹簧

a 受到的力等于弹簧质点间的距离绝对值减去弹簧原本长度，再乘以 a 到 b 的方向的单位向量。问题是：弹窗不会停下来。![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213037-ql0mn1f.png)

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213043-0lmyjru.png)质点在运动过程中的能量损失，可以用摩擦系数来计算。表现起来就像是质点运动的时候有一个拉力，使得质点的速度慢下来。![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213049-w5efpt7.png)

但也存在一个问题，这样做摩擦系数他会同时作用于两个质点，让所有的运动都慢下来（比如弹簧从高处落下来降到地上速度也会降低），但我们想要的是质点之间的关系。

我们把弹簧的摩擦系数与质点之间的速度差做乘积，并在物体运动方向做投影，此时阻力仅在质点之间作用，不会影响整体。![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213053-s2s9qf2.png)

弹簧结构

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213057-4mq04gb.png)

模拟一块布的时候，会遇到一些问题：

1.这个结构无法承受对角线方向的剪力

2.这个结构无法抵抗平面外的弯曲（比如把一张纸对折时，基本对折处的节点不会受到周围节点对抗的力，但是对于布而言，显然折叠时，折叠处的节点会受到周围节点的拉力）![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213108-ntv3bsg.png)

这个时候可以往剪力作用的方向的垂直方向加一个约束，此时可以对抗一个方向上的剪力，但还是无法对抗平面外的弯曲。![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213112-ez1kdm2.png)

再往另一个方向加一个约束，此时可以对抗两个方向的剪力，但还是无法对抗平面外弯曲![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213119-srnkhm1.png)

在节点间跳连，来对抗平面外的弯曲，同时这个连接应该是一种较弱的起辅助作用的连接![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213128-gkyneu3.png)

最后，可以得到一个比较好的对布的模拟：![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213132-aw10l3w.png)

除了质点弹窗系统外，还可以用有限元的方法进行布的运动的模拟，运动从一个位置传递到另一个位置。

#### 粒子系统

有一些运动可以用大量粒子来模拟，比如说雾、沙子，在游戏中有广泛的应用。<br />粒子系统需要计算每个粒子受到的力，优点是易于理解和实现，可以用少量粒子来体现速度，更多的粒子来体现复杂度。缺点是有一些场景需要大量的粒子，比如说模拟流体，同时一个粒子会受到四周其他粒子的力的作用，这些计算需要用到加速结构。![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213138-hjeok4e.png)

粒子系统动画制作的步骤：
对于每一帧的运动：
1.创造一些粒子

2.计算每个粒子受到的作用力

3.更新粒子的位置和速度

4.移除死掉的粒子

5.渲染

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213144-k4aaya5.png)

粒子系统的受力：

1.引力和斥力

2.摩擦力

3.碰撞

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213150-xbzpva9.png)

粒子间的引力：![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213158-n6qmgmm.png)

模拟星云：![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213202-uuhi1so.png)

模拟流体：![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213206-p3x72r3.png)

粒子系统就是对一个群里中的个体的运动进行计算

模拟鸟群，把每只鸟作为一个粒子，鸟的运动遵循一些基本规则：

1.每个鸟会受到周围鸟的吸引

2.每个个体之间有一段距离（斥力）

3.所有鸟基本都朝着一个方向运动![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213211-i8kp8gb.png)

模拟分子运动![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213215-wqpkbdp.png)

#### 正向运动学

骨骼的表示：
1.拓扑结构（找到每个骨骼相连的骨骼）

2.关节处的几何关系

3.树状结构：

关节的连接类型

1.一维的铰连接

2.二维的球关节

3.平移关节![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213223-7wbmkrp.png)

2d 模拟一个两段的手臂：先算出第一个关节的旋转角，再算出第二个关节的旋转角![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213229-uxjwf4f.png)

动画可以表示为时间和角度的函数![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213233-tqx7asr.png)

正向运动学的优缺点：
优点：

运动的控制很方便，实现方式也很直接

缺点：物体表示和艺术家们想要的动画效果不一致，花费太多时间![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213237-g7l0pby.png)

#### 逆向运动学

从运动的顶点推测中间的位置![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213240-xxz7wwx.png)

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213244-ili860d.png)逆向运动学下的模拟的手臂运动的求解![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213249-0jafn1c.png)

逆向运动学的困难点：

可能有多个解或者没有解![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213253-gxqn0qo.png)

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213257-c1k6gsk.png)![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213311-t2k0d1w.png)因此会用到一些优化算法（比如梯度优化算法）![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213318-arlr0bx.png)

逆向运动学的模拟 demo![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213324-hsyrhfv.png)

#### Rigging

就像牵线木偶一样，捕捉所有特征的变化，但制作成本很大，同时需要艺术和技术上的训练：![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213330-yrtkw9l.png)

Blend Shapes

对两个面做插值，计算从一个面到另一个面变化的中间态。![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213334-px0wl9y.png)

#### 运动捕捉

记录真实世界的运动，在大量数据中提取某一个动作。![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213338-x6pa6de.png)

运用最多的还是光学的

运动捕捉的优缺点：
优点：可以快速捕捉大量真实数据，效果的真实性很高

缺点：成本很大，捕捉的运动效果无法满足艺术家的需求（动画可以需要更夸张的运动）

运动捕捉设备：光学的、电磁波的、机械的![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213343-fh06ewa.png)

运用最多的还是光学的![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213347-nargvr7.png)

![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213352-tauyql9.png)![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213357-pfjw1bf.png)面部的动画模拟![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213402-afz91t3.png)

当模拟的动画过于接近真实人类时，会有恐怖谷效应

面部表情捕捉![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213406-ncqy94v.png)

电影的生产线![image](/assets/img/posts/第二十一章-animation-simulation/image-20230817213411-b3rw3wd.png)

下一节课：
怎么通过力、物理理论模拟真实的运动
