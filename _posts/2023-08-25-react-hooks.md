---
title: "React Hooks"
date: 2023-08-25 19:43:00
categories:
  - React
---

## useState

返回一个 state，和一个更新 state 的函数

### 函数式更新

如果新的 state 需要通过先前的 state 计算得出，可以传递一个函数给 setState，如下：

```javascript
setState((prev) => {
  return prev + 1;
});
```

### 惰性初始 state

如果初始 state 需要通过复杂计算获得，则可以传入一个函数，在函数中计算并返回初始的 state，此函数只在**初始渲染**时被调用：

```javascript
const [state, setState] = useState(() => {
  const initialState = someExpensiveComputation(props);
  return initialState;
});
```

**第一个常见的使用场景是当创建初始 state 很昂贵时：**

```javascript
function Table(props) {
  // ⚠️ createRows() 每次渲染都会被调用
  const [rows, setRows] = useState(createRows(props.count));
  // ...
}
function Table(props) {
  // ✅ createRows() 只会被调用一次
  const [rows, setRows] = useState(() => createRows(props.count));
  // ...
}
```

## useEffect

使用 `useEffect` 完成副作用操作。赋值给 `useEffect` 的函数会在**组件渲染到屏幕之后**执行。<br />如果想执行只运行一次的 effect（仅在组件挂载和卸载时执行），可以传递一个空数组（`[]`）作为第二个参数。这就告诉 React 你的 effect 不依赖于 props 或 state 中的任何值，所以它永远都不需要重复执行。

## useReducer

[`useState`]() 的替代方案。它接收一个形如 `(state, action) => newState` 的 reducer，并返回当前的 state 以及与其配套的 `dispatch` 方法。在某些场景下，`useReducer` 会比 `useState` 更适用，例如 state 逻辑较复杂且包含多个子值，或者下一个 state 依赖于之前的 state 等。

## useCallback

```javascript
const memoizedCallback = useCallback(() => {
  doSomething(a, b);
}, [a, b]);
```

返回一个 [memoized](https://en.wikipedia.org/wiki/Memoization) 回调函数。该回调函数仅在某个依赖项改变时才会更新。

## useMemo

```javascript
const memoizedValue = useMemo(() => computeExpensiveValue(a, b), [a, b]);
```

返回一个 [memoized](https://en.wikipedia.org/wiki/Memoization) 值。把“创建”函数和依赖项数组作为参数传入 `useMemo`，它仅会在某个依赖项改变时才重新计算 memoized 值。这种优化有助于避免在每次渲染时都进行高开销的计算。
