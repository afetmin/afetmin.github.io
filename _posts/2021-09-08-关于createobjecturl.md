---
title: "createObjectURL"
date: 2021-09-08 09:08:43
categories:
  - js
---

## createObjectURL 是什么

> URL.createObjectURL() 静态方法会创建一个 DOMString，其中包含一个表示参数中给出的对象的 URL。这个 URL 的生命周期和创建它的窗口中的 document 绑定。这个新的 URL 对象表示指定的 File 对象或 Blob 对象。 — MDN
<!--more-->

## Blob URL/Object URL
Blob URL/Object URL 是⼀种伪协议，允许 Blob 和 File 对象⽤作图像，下载⼆进制数据链接等的 URL
源。在浏览器中，我们使⽤ URL.createObjectURL ⽅法来创建 Blob URL，该⽅法接收⼀个 Blob 对
象，并为其创建⼀个唯⼀的 URL，其形式为 `blob:<origin>/<uuid>` ，对应的示例如下：

```js
blob:https://example.org/40a5fb5a-d56d-4a33-b4e2-0acf6a8e5f641
```

浏览器内部为每个通过 URL.createObjectURL ⽣成的 URL 存储了⼀个 URL → Blob 映射。因此，此
类 URL 较短，但可以访问 Blob 。⽣成的 URL 仅在当前⽂档打开的状态下才有效。它允许引⽤
`<img> 、 <a>` 中的 Blob ，但如果你访问的 Blob URL 不再存在，则会从浏览器中收到 404 错误。

上述的 Blob URL 看似很不错，但实际上它也有副作⽤。虽然存储了 URL → Blob 的映射，但 Blob 本
身仍驻留在内存中，浏览器⽆法释放它。映射在⽂档卸载时⾃动清除，因此 Blob 对象随后被释放。但
是，如果应⽤程序寿命很⻓，那不会很快发⽣。因此，如果我们创建⼀个 Blob URL，即使不再需要该
Blob，它也会存在内存中。

针对这个问题，我们可以调⽤ URL.revokeObjectURL(url) ⽅法，从内部映射中删除引⽤，从⽽允许
删除 Blob（如果没有其他引⽤），并释放内存。

## createObjectURL 可以用来做什么

比如显示上传的预览图

```html
<body>
  <input
    type="file"
    multiple
    id="fileInput"
    onchange="previewFiles(this.files)"
  />
  <ul id="preview"></ul>
  <script>
    const inputEl = document.getElementById('fileInput') const ulEle =
    document.getElementById('preview') function previewFiles(files){" "}
    {Array.from(files).forEach((file) => {
      const li = document.createElement("li")
      ulEle.appendChild(li)
      const img = document.createElement("img")
      img.src = URL.createObjectURL(file)
      img.height = 100
      img.width = 100
      img.onload = function () {
        // 当图片加载完成之后对象URL就不再需要了
        URL.revokeObjectURL(this.src)
      }
      li.appendChild(img)
    })}
  </script>
</body>
```

来看看效果：

![效果](/assets/img/posts/关于createobjecturl/1.png)

> 在每次调用  `createObjectURL()`  方法时，都会创建一个新的 URL 对象，即使你已经用相同的对象作为参数创建过。当不再需要这些 URL 对象时，每个对象必须通过调用  [URL.revokeObjectURL()](https://developer.mozilla.org/zh-CN/docs/Web/API/URL/revokeObjectURL) 方法来释放。

浏览器在 document 卸载的时候，会自动释放它们，但是为了获得最佳性能和内存使用状况，你应该在安全的时机主动释放掉它们。

## FileReader 的 readAsDataURL

> readAsDataURL 方法会读取指定的 Blob 或 File 对象。读取操作完成的时候，readyState 会变成已完成 DONE，并触发 loadend 事件，同时 result 属性将包含一个 data:URL 格式的字符串（base64 编码）以表示所读取文件的内容。

还是上一个例子

```js
function useFileReaderPreivewFiles(files) {
  Array.from(files).forEach((file) => {
    const reader = new FileReader()
    const li = document.createElement("li")
    ulEle.appendChild(li)
    reader.onload = function (e) {
      const img = new Image()
      img.height = 100
      img.width = 100
      img.src = e.target.result
      li.appendChild(img)
    }
    reader.readAsDataURL(file)
  })
}
```

同样的效果:

![效果图](/assets/img/posts/关于createobjecturl/2.png)

来一张终极图：

![data_url.jpg](/assets/img/posts/关于createobjecturl/3.jpg)
