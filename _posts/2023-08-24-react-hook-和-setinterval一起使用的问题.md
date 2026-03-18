---
title: "React Hook 和 setInterval一起使用的问题"
date: 2023-08-24 06:50:05
categories:
  - React
---

如果仅仅是用一个变量来保存tiner的ID，会导致无法清除timerID<br />如下：

```javascript
const TimerCount = () => {
  let [count,setCount] = useState(0)
  let timer
  const handleStart =() => {
    timer = setInterval(() => {
      setCount(count++)
    },1000)
  }
  const handleEnd = () => {
    clearInterval(timer)
  }
  return (
    <>
    <button onClick={handleStart}>start</button>
    <button onClick={handleEnd}>end</button>
    <div>{count}</div>
    </>
  )
}
```

### 原因：

timer的id是一个number类型，每次在页面展示count时，都会 触发重新渲染，timer会被重新赋值，这是一个基本类型的值，需要清除的timer值已经不是原来的值了，虽然打印出来的是一样的值。

### 解决方案

用useRef，在整个渲染周期固定一个值

```javascript
const TimerCount = () => {
  let [count,setCount] = useState(0)
  let timer = useRef<any>(null)
  const handleStart =() => {
    timer.current = setInterval(() => {
      setCount(count++)
    },1000)
  }
  const handleEnd = () => {
    clearInterval(timer.current)
  }
  return (
    <>
    <button onClick={handleStart}>start</button>
    <button onClick={handleEnd}>end</button>
    <div>{count}</div>
    </>
  )
}
```

这时timer是一个引用类型，里面的current属性保存了timer的id，并且整个渲染周期不变。使用ref访问DOM节点也是一样的原理
