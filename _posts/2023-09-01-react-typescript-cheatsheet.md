---
title: "React typescript cheatsheet"
date: 2023-09-01 08:12:56
categories:
  - React
---

[React Typescript Cheetsheet](https://react-typescript-cheatsheet.netlify.app/docs/basic/getting-started/basic_type_example)

### JSX.Element vs React.ReactNode

- jsx.element -> React.createElement 的返回值
- React.ReactNode ->  组件返回值的集合 

### interface or type?

使用 Interface 直到你需要 Type

- 在编写库或第 3 方环境类型定义时，始终 interface 用于公共 API 的定义，因为这允许消费者在缺少某些定义时通过声明合并来扩展它们。
- 在你的 React 组件 Props 和 State 使用 Type，以保持一致性并且因为它受到更多限制。

### custom hooks

```jsx
import { useState } from "react";

export function useLoading() {
  const [isLoading, setState] = useState(false);
  const load = (aPromise: Promise<any>) => {
    setState(true);
    return aPromise.finally(() => setState(false));
  };
  return [isLoading, load] as const; // infers [boolean, typeof load] instead of (boolean | typeof load)[]
}
```

当你解构时，你会在解构的位置获取正确的类型<br />或者编写一个自动类型推断的函数

```jsx
// 这个函数仅仅为了让TS自动推断参数类型
function tuplify<T extends any[]>(...elements: T) {
  return elements;
}

function useArray() {
  const numberValue = useRef(3).current;
  const functionValue = useRef(() => {}).current;
  return [numberValue, functionValue]; // type is (number | (() => void))[]
}

function useTuple() {
  const numberValue = useRef(3).current;
  const functionValue = useRef(() => {}).current;
  return tuplify(numberValue, functionValue); // type is [number, () => void]
}
```

### class Component

```jsx
type MyProps = {
  // using `interface` is also ok
  message: string,
};
type MyState = {
  count: number, // like this
};
class App extends React.Component<MyProps, MyState> {
  state: MyState = {
    // optional second annotation for better type inference
    count: 0,
  };
  render() {
    return (
      <div>
        {this.props.message} {this.state.count}
      </div>
    );
  }
}
```

**为什么要注释两次 state？**<br />对类属性进行注释不是必要的，但它允许在访问 this.state 和初始化状态时更好地进行类型推断。<br />注释以两种不同的方式工作，第二个泛型类型参数将允许 this.setState() 正常工作，因为该方法来自基类，但 state 在组件内部初始化时可能会覆盖基类的实现，因此您必须确保告诉编译器你实际上并没有做任何不同的事情。<br />**不需要添加 readonly**

```jsx
type MyProps = {
  readonly message: string;
};
type MyState = {
  readonly count: number;
};
```

React.Component<P,S> 已经默认标记它们为不可变。
