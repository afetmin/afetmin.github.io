---
title: "React element，children，parents and re-renders"
date: 2023-08-22 19:52:46
categories:
  - React
tags:
  - 进阶
---

### children as a render function

```jsx
const MovingComponent = ({ children }) => {
  ...
  return (
    <div ...// callbacks same as before
      >
      // children as render function with some data
      // data doesn't depend on the changed state!
      {children({ data: 'something' })}
    </div>
  );
};

const SomeOutsideComponent = () => {
  return (
    <MovingComponent>
      // ChildComponent re-renders when state in MovingComponent changes!
      // even if it doesn't use the data that is passed from it
      {() => <ChildComponent />}
    </MovingComponent>
  )
}
```

### React.memo behavior

#### memo parent

```jsx
// wrapping MovingComponent in memo to prevent it from re-rendering
const MovingComponentMemo = React.memo(MovingComponent);

const SomeOutsideComponent = () => {
  // trigger re-renders here with state
  const [state, setState] = useState();

  return (
    <MovingComponentMemo>
      <!-- ChildComponent will still re-render when SomeOutsideComponent re-renders -->
      <ChildComponent />
    </MovingComponentMemo>
  )
}
```

#### memo child

```jsx
// wrapping ChildComponent in memo to prevent it from re-rendering
const ChildComponentMemo = React.memo(ChildComponent);

const SomeOutsideComponent = () => {
  // trigger re-renders here with state
  const [state, setState] = useState();

  return (
    <MovingComponent>
      <!-- ChildComponent won't re-render, even if the parent is not memoized -->
      <ChildComponentMemo />
    </MovingComponent>
  )
}
```

### useCallback hook behavior

```jsx
const SomeOutsideComponent = () => {
  // trigger re-renders here with state
  const [state, setState] = useState();

  // trying to prevent ChildComponent from re-rendering by memoising render function. Won't work!
  const child = useCallback(() => <ChildComponent />, []);

  return (
    <MovingComponent>
      <!-- Memoized render function. Didn't help with re-renders though -->
      {child}
    </MovingComponent>
  )
}
```

## Why?

### 什么是react 的child？

```jsx
const Parent = ({ children }) => {
  return <>{children}</>;
};

<Parent>
  <Child />
</Parent>;
```

仅仅是一个 prop<br />我们可以将组件作为 elements， functions 或者 Components

```jsx
// as prop
<Parent children={() => <Child />} />

// "normal" syntax
<Parent>
  {() => <Child />}
</Parent>

// implementation
const Parent = ({ children }) => {
  return <>{children()}</>
}

// even this
<Parent children={Child} />;

const Parent = ({ children: Child }) => {
  return <>{<Child />}</>;
};
```

### 什么是 React element？

```jsx
const child = <Child />
```

这里 <Child /> 仅仅是一个 React.createElement 语法糖，它返回一个对象，这个对象是你想要在屏幕上看到的元素的描述

```jsx
const Parent = () => {
  // will just sit there idly
  const child = <Child />;

  return <div />;
};

// same as
const Parent = () => {
  // exactly the same as <Child />
  const child = React.createElement(Child, null, null);

  return <div />;
};
```

```jsx
const Parent = () => {
  // render of Child will be triggered when Parent re-renders
  // since it's included in the return
  const child = <Child />;

  return <div>{child}</div>;
};
```

### 更新元素

元素是不可变的对象。更新元素并触发其相应组件重新渲染的唯一方法是重新创建对象本身。这正是重新渲染期间发生的事情

```jsx
const Parent = () => {
  // child definition object will be re-created.
  // so Child component will be re-rendered when Parent re-renders
  const child = <Child />;

  return <div>{child}</div>;
};
```

而且仅仅会重新创建和更新存在的组件，我们可以使用 React.memo 或者 useMemo 记忆它：

```jsx
const ChildMemo = React.memo(Child);

const Parent = () => {
  const child = <ChildMemo />;

  return <div>{child}</div>;
};
```

```jsx
const Parent = () => {
  const child = useMemo(() => <Child />, []);

  return <div>{child}</div>;
};
```

定义对象不会被重新创建，React 会认为它不需要更新，Child 的重新渲染也不会发生。

### 我们能够知道

1. 当我们编写const child = <Child />时，我们只是在创建一个Element，即组件定义，而不是渲染它。这个定义是一个**不可变的对象**。
2. 此定义中的组件仅在它最终出现在实际渲染树中时才会被渲染。对于功能组件，它是您实际从组件中返回它的时候。
3. 重新创建定义对象会触发对应组件的重新渲染。

#### 为什么作为prop传递的组件不会重新渲染？

```jsx
const MovingComponent = ({ children }) => {
  // this will trigger re-render
  const [state, setState] = useState();
  return (
    <div
      // ...
      style={{ left: state.x, top: state.y }}
    >
      <!-- those won't re-render because of the state change -->
      {children}
    </div>
  );
};

const SomeOutsideComponent = () => {
  return (
    <MovingComponent>
      <ChildComponent />
    </MovingComponent>
  )
}
```

这是因为 <ChildComponent /> 是在 SomeOutsideComponent 中创建的， MovingComponent 重新渲染时它的props并没有改变，child 不会重新创建，也不会重新渲染。

#### 为什么children 作为一个render function 时会重新渲染呢？

```jsx
const MovingComponent = ({ children }) => {
  // this will trigger re-render
  const [state, setState] = useState();
  return (
    <div ///...
    >
      <!-- those will re-render because of the state change -->
      {children()}
    </div>
  );
};

const SomeOutsideComponent = () => {
  return (
    <MovingComponent>
      {() => <ChildComponent />}
    </MovingComponent>
  )
}
```

在这种情况下， children 是一个函数， Element 是这个函数调用的结果，每次调用MovingComponent 时，都会重新创建函数返回定义对象 <ChildComponent />，从而触发重新渲染。

#### 为什么使用 React.memo 包裹父组件会重新渲染？而包裹子组件 （子组件）不会重新渲染呢？

```jsx
// wrapping MovingComponent in memo to prevent it from re-rendering
const MovingComponentMemo = React.memo(MovingComponent);

const SomeOutsideComponent = () => {
  // trigger re-renders here with state
  const [state, setState] = useState();

  return (
    <MovingComponentMemo>
      <!-- ChildComponent will re-render when SomeOutsideComponent re-renders -->
      <ChildComponent />
    </MovingComponentMemo>
  )
}

// same as
const SomeOutsideComponent = () => {
  // ...
  return <MovingComponentMemo children={<ChildComponent />} />;
};
```

触发重新渲染时，被记忆的父组件会做 prop 检查，查看是否有prop 被改变了，因为 <ChildComponent /> 被重新创建了，所以prop改变了，子组件就重新渲染了。

```jsx
// wrapping ChildComponent in memo to prevent it from re-rendering
const ChildComponentMemo = React.memo(ChildComponent);

const SomeOutsideComponent = () => {
  // trigger re-renders here with state
  const [state, setState] = useState();

  return (
    <MovingComponent>
      <!-- ChildComponent won't be re-rendered anymore -->
      <ChildComponentMemo />
    </MovingComponent>
  )
}
```

子组件被记忆了， MovingComponent 触发重新渲染时，子组件没有改变，被跳过，不会重新渲染。

#### 为什么将 children 作为函数传递时，记忆这个函数不起作用呢？

```jsx
const SomeOutsideComponent = () => {
  // trigger re-renders here with state
  const [state, setState] = useState();

  // this memoization doesn't prevent re-renders of ChildComponent
  const child = useCallback(() => <ChildComponent />, []);

  return <MovingComponent>{child}</MovingComponent>;
};
// same as
const SomeOutsideComponent = () => {
  // trigger re-renders here with state
  const [state, setState] = useState();

  // this memoization doesn't prevent re-renders of ChildComponent
  const child = useCallback(() => <ChildComponent />, []);

  return <MovingComponent children={child} />;
};
```

这里记忆的仅仅是这个函数，它的返回值并没有被记忆，所以每次都会重新创建一个新的对象，导致子组件重新渲染。<br />有两种方式可以解决：

1. React.memo 包裹 MovingComponent
2. 删掉 useCallback， 使用 React.memo 包裹 ChildComponent
