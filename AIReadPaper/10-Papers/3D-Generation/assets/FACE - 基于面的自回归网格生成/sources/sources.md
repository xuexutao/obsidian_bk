# FACE 关键图与官方资料溯源

## 官方源材料

| 资料 | 来源 | 用途 | 获取日期 |
| --- | --- | --- | --- |
| 论文 PDF（v2, 10 页） | https://arxiv.org/abs/2603.01515 | 主参考 | 2026-08-25 |
| arXiv 页面 | https://arxiv.org/abs/2603.01515 | 元数据 / 版本 | 2026-08-25 |

## 关键图（图编号 → 裁切 PNG → 原文位置）

| 论文图号 | 描述 | 本地文件 | PDF 页码 | 裁切框（相对页面） |
| --- | --- | --- | --- | --- |
| Fig.1 | teaser：从点云重建的高保真网格 | `figures/01-teaser.png` | p1 | (0.096, 0.294, 0.904, 0.601) |
| Fig.2 | 端到端管线（VecSet 编码器 + 自回归 Face Decoder） | `figures/02-pipeline.png` | p3 | (0.04, 0.08, 0.96, 0.86) |
| Fig.3 | image-to-mesh 管线（DiT + Face Decoder） | `figures/03-image-to-mesh.png` | p5 | (0.06, 0.10, 0.94, 0.72) |
| Fig.4 | Toys4K 重建定性对比 | `figures/04-reconstruction.png` | p7 | (0.174, 0.092, 0.821, 0.454) |
| Fig.5 | image-conditioned 生成定性对比（vs EdgeRunner） | `figures/05-image-conditioned.png` | p8 | (0.105, 0.091, 0.821, 0.50) |

## 笔记中引用约定

- 所有图均使用 `assets/figures/<name>.png` 相对路径
- 在笔记正文中通过 `![描述](assets/figures/<name>.png)` 引用
- 裁切框由 `scan_figures.py` + 论文文本位置推断；Fig.2/Fig.3 为矢量图，`scan_figures.py` 无法检测，改为整页渲染后按图注位置裁切
- Fig.6（base vs large 对比）未单独裁切，正文以文字描述对应结论
