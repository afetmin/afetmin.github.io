---
title: "better-scroll结合vue遇到图片引起的问题"
date: 2020-11-30 10:04:39
categories:
  - vue
---

## 使用better-scroll遇到的问题
原来项目中只使用了better-scroll来做轮播图和滑动组件，但是头部和标签栏是固定的，只滑动下面部分，于是设置 scroll-warpper样式如下：
```css
.scroll-wrapper {
  position: absolute;
  top: 80px;
  left: 0;
  right: 0;
  bottom: 0;
}
```
但是想要实现下部组件与上部标签的联动，仅仅靠better-scroll就不够了，于是项目中按需引入了vant的的标签组件，来实现移动端的左右滑动（不得不说，封装好的东西就是香）。紧接着就产生了问题，

 - 一是：下部组件无法滑动，
 - 二是：better-scroll滑不到底部，只能滑一部分。刷新页面后可以滑动了

分析问题：
 - 查看dom发现vant的content__wrap没有高度，而scroll-warpper和内容高度一致。这是由于content__wrap的定位是relative，而scroll-wrapper是相对于content__wrap而定位的，并不是相对整个页面定位，所有内容是被里面撑开的。
 - 页面中有大量图片，当dom加载完成后图片并没有加载完成，scroll已经初始化了，所以滑不到底部。刷新页面可以滑动是因为图片已经被浏览器缓存了，刷新直接拿到图片不需要重新请求。

解决问题：
  - 第一个问题是定位的原因，那我们把wrapper的高度写死就好了，
  但是由于头部和标签是自适应的，高度是不确定的，就没法写死了。使用动态获取设置wrapper的style：
```js
setScrollWrapperHeight() {
      this.$refs.scroll.$el.style.height =
        window.innerHeight - this.topHeight + 'px'
    },
```
 - 由于多个页面都需要这个配置，放到mixin中，在mounted方法中调用此方法。
解决wrapper高度问题。
 - 第二个问题是图片加载时content的高度设置错误问题，可以通过img的load方法判断图片是否加载完成。
 子组件：
 ```js
imgAllLoad() {
      const mulitImg = document.querySelectorAll('.suggest-item-img')
      const promiseAll = []
      const imgTotal = mulitImg.length
      for (let i = 0; i < imgTotal; i++) {
        promiseAll[i] = new Promise((resolve, reject) => {
          mulitImg[i].onload = function () {
            // 第i张加载完成
            resolve(mulitImg[i])
          }
        })
      }
      return promiseAll
    },
 ```
在子组件mounted中：派发一个imgLoad方法
```js
Promise.all(this.imgAllLoad()).then(() => {
      this.$emit('imgLoad')
    })
```
在父组件监听此方法，并刷新scroll组件
```js
methods: {
    imgLoad() {
      this.$refs.scroll.refresh()
    }
  },
```
这样体验就不错了。
