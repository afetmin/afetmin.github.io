---
title: "React memo lazy"
date: 2021-09-09 09:20:46
categories:
  - React
---

## React.memo

react.memo 有第二个参数，可以通过第二个参数来自定义组件的渲染时机。

```jsx
React.memo(Component, compare)
```

React.memo 接受两个参数，第一个参数 Component 原始组件本身，第二个参数 compare 是一个函数，可以根据一次更新中 props 是否相同决定原始组件是否重新渲染。

memo 的几个特点是：

- React.memo: 第二个参数 返回 true 组件不渲染 ， 返回 false 组件重新渲染。
  和 shouldComponentUpdate 相反，shouldComponentUpdate : 返回 true 组件渲染 ， 返回 false 组件不渲染。

- memo 当二个参数 compare 不存在时，会用浅比较原则处理 props ，相当于仅比较 props 版本的 pureComponent 。
- memo 同样适合类组件和函数组件。
  关于第二个参数你可以这样写：

```jsx
const compare = (prev, next) =>
  prev.number === next.number ||
  (prev.number !== next.number && next.number > 1)
```

这里有两个意思：

- number 改变时 组件渲染
- number 小于 1 时 组件渲染

## 关于 rander

### 有没有必要在乎组件不必要渲染。

在正常情况下，无须过分在乎 React 没有必要的渲染，执行 render 不等于真正的浏览器渲染视图，render 阶段执行是在 js 当中，js 中运行代码远快于浏览器的 Rendering 和 Painting 的，更何况 React 还提供了 diff 算法等手段，去复用真实 DOM 。

### 什么时候需要注意渲染节流。

但是对于以下情况，值得开发者注意，需要采用渲染节流：

- 第一种情况数据可视化的模块组件（展示了大量的数据），这种情况比较小心因为一次更新，可能伴随大量的 diff ，数据量越大也就越浪费性能，所以对于数据展示模块组件，有必要采取 memo ， shouldComponentUpdate 等方案控制自身组件渲染。
- 第二种情况含有大量表单的页面，React 一般会采用受控组件的模式去管理表单数据层，表单数据层完全托管于 props 或是 state ，而用户操作表单往往是频繁的，需要频繁改变数据层，所以很有可能让整个页面组件高频率 render 。
- 第三种情况就是越是靠近 app root 根组件越值得注意，根组件渲染会波及到整个组件树重新 render ，子组件 render ，一是浪费性能，二是可能执行 useEffect ，componentWillReceiveProps 等钩子，造成意想不到的情况发生。

## 使用 Suspense 和 React.lazy 模拟实现异步加载组件

### 实现效果

- 异步请求数据，请求完数据挂载组件，请求过程中展示 loading 效果。
- 全程组件只渲染一次

### 编写

```jsx
function AsyncComp(Comp, asyncFn) {
  const AsyncCompPromise = () =>
    new Promise(async (resolve) => {
      const data = await asyncFn()
      resolve({
        default: (props) => <Comp value={data} {...props} />,
      })
    })
  return React.lazy(AsyncCompPromise)
}
const fetchData = () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        name: "Blum",
        msg: "a lot of data",
      })
    }, 1000)
  })
}
function Test({ value, gender }) {
  const { name, msg } = value
  return (
    <div>
      <div>my name is {name}</div>
      <div>this is {msg}</div>
      <div>my gender is {gender}</div>
    </div>
  )
}
function App() {
  const LzayTest = AsyncComp(Test, fetchData)
  return (
    <Suspense fallback={<div>loading...</div>}>
      <LzayTest gender='男' />
    </Suspense>
  )
}
```

React.lazy 要求函数需要返回一个 Promise。大致流程是，react.lazy 第一次渲染时会向外抛个异常 Promise，这个异常会被 Suspense 捕获到，Suspense 处理 Promise，Promise 执行成功后 Susponse 发起渲染，返回真正需要渲染的组件。
