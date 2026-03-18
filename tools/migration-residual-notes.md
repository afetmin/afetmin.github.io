# Migration Residual Notes

## Bulk Migration Summary

- Migrated posts: `241`
- Unresolved assets: `0`
- Unresolved Hexo `post_link`: `0`

## Manual Review Items

### Posts With Local Video

- `现代浏览器架构`

### Posts With iframe Embeds

- `复现 musicforprogramming.net 的音乐可视化效果`
- `揭秘微信“快捷登录”：为什么你不用扫码就能直接登录？`
- `逆天!纯CSS实现获取窗口大小`

These posts were migrated successfully, but iframe rendering should still be checked in the browser.

## Trial-Migration Duplicates To Delete Later

The earlier manual trial created four English-slug files. The bulk migration also generated title-based files for the same source posts. Keep one version of each pair and delete the other later.

- `2023-03-05-frontend-performance-optimization.md`
- `2023-03-05-前端性能优化方法.md`
- `2023-08-19-homogeneous-coordinates.md`
- `2023-08-19-齐次坐标到底是什么.md`
- `2023-08-29-modern-browser-architecture.md`
- `2023-08-29-现代浏览器架构.md`
- `2025-12-01-why-deep-learning-takes-off-now.md`
- `2025-12-01-03-为什么深度学习现在才真正起飞.md`

Recommendation:

- Keep the English-slug versions if you want cleaner URLs.
- Delete the Chinese-slug duplicates after a final spot check.
