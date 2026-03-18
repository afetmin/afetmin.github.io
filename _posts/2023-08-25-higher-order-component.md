---
title: "Higher-order component"
date: 2023-08-25 07:31:33
categories:
  - React
---

## 高阶组件可以做什么？

### 增强回调和生命周期

1. 增强回调

```jsx
type Base = { onClick: () => void };
export const withLoggingOnClickWithProps = <TProps extends Base>(Component: ComponentType<TProps>) => {
  // our returned component will now have additional logText prop
  return (props: TProps & { logText: string }) => {
    const onClick = () => {
      // accessing it here, as any other props
      console.log('Log on click: ', props.logText);
      props.onClick();
    };

    return <Component {...props} onClick={onClick} />;
  };
};

const Page = () => {
  return (
    <ButtonWithLoggingOnClickWithProps onClick={onClickCallback} logText="this is Page button">
      Click me
    </ButtonWithLoggingOnClickWithProps>
  );
};
```

2. 在挂载时发送数据而不是点击时

```jsx
export const withLoggingOnMount = <TProps extends unknown>(Component: ComponentType<TProps>) => {
  return (props: TProps) => {
    // no more overriding onClick, just adding normal useEffect
    useEffect(() => {
      console.log('log on mount');
    }, []);

    // just passing props intact
    return <Component {...props} />;
  };
};
```

### 拦截 DOM 事件

```jsx
useEffect(() => {
  const keyPressListener = (event) => {
    // do stuff
  };

  window.addEventListener('keypress', keyPressListener);

  return () => window.removeEventListener('keypress', keyPressListener);
}, []);

export const Modal = ({ onClose }: ModalProps) => {
  const onKeyPress = (event) => event.stopPropagation();

  return <div onKeyPress={onKeyPress}>...// dialog code</div>;
};

// same as

export const withSupressKeyPress = <TProps extends unknown>(Component: ComponentType<TProps>) => {
  return (props: TProps) => {
    const onKeyPress = (event) => {
      event.stopPropagation();
    };

    return (
      <div onKeyPress={onKeyPress}>
        <Component {...props} />
      </div>
    );
  };
};
```

### 通用 React 上下文选择器

```jsx
// mini-redux
export const withContextSelector = <TProps extends unknown, TValue extends unknown>（
  Component: ComponentType<TProps & Record<string, TValue>>,
  selectors: Record<string, (data: Context) => TValue>,
）: Component<Record<string, TValue>> => {
  // memoising component generally for every prop
  const MemoisedComponent = React.memo(Component) as ComponentType<Record<string, TValue>>;
  return (props: TProps & Record<string, TValue>) => {
    // extracting everything from context
    const data = useFormContext();

    // mapping keys that are coming from "selectors" argument
    // to data from context
    const contextProps = Object.keys(selectors).reduce((acc, key) => {
      acc[key] = selectors[key](data);

      return acc;
    }, {});

    // spreading all props to the memoised component
    return <MemoisedComponent {...props} {...contextProps} />;
  };

}
```

```jsx
// props are injected by the higher order component below
const CountriesWithFormId = ({
  formId,
  countryName,
}: {
  formId: string,
  countryName: string,
}) => {
  console.log("Countries with selector re-render");
  return (
    <div>
      <h3>List of countries for form: {formId}</h3>
      Selected country: {countryName}
      <ul>
        <li>Australia</li>
        <li>USA</li>
      </ul>
    </div>
  );
};

// mapping props to selector functions
const CountriesWithFormIdSelector = withContextSelector(CountriesWithFormId, {
  formId: (data) => data.id,
  countryName: (data) => data.country,
});
```

### 总结

1. 使用附加功能增强回调和 React 生命周期事件，例如发送日志记录或分析事件
2. 拦截 DOM 事件，例如在打开模式对话框时阻止全局键盘快捷键
3. 提取一段上下文而不导致组件中不必要的重新渲染
