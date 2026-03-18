---
title: "React Props Children 混合插槽"
date: 2021-09-09 09:08:43
categories:
  - React
---

## 当多种 children 一起输入时

```jsx
function Container(props) {
  const defaultProps = {
    name: "alen",
    msg: "hello",
  }

  return props.children.map((item) => {
    // 通过判断 clone 混入props
    if (React.isValidElement(item)) {
      return React.cloneElement(item, { ...defaultProps })
    } else if (typeof item === "function") {
      return item(defaultProps)
    } else {
      return null
    }
  })
}

function Child(props) {
  return <div>{props.name + props.msg} </div>
}

function App() {
  return (
    <Container>
      // 多种方式混入
      <Child />
      {(props) => <Child {...props} />}
    </Container>
  )
}

render(<App />, document.getElementById("root"))
```
