# MeshGPT 关键图与官方资料溯源

## 官方源材料

| 资料 | 来源 | 用途 | 获取日期 |
| --- | --- | --- | --- |
| 论文 PDF（v1, 16 页） | https://arxiv.org/abs/2311.15475 | 主参考 | 2026-08-25 |
| 项目页 | https://nihalsid.github.io/mesh-gpt/ | Demo / 视频 | 2026-08-25 |
| 官方代码 | https://github.com/audi/MeshGPT | 实现细节 / 复现 | 2026-08-25 |

## 关键图（图编号 → 裁切 PNG → 原文位置）

| 论文图号 | 描述 | 本地文件 | PDF 页码 | 裁切框（相对页面） |
| --- | --- | --- | --- | --- |
| Fig.1 | teaser：自回归采样生成三角网格的效果总览 | `figures/01-teaser.png` | p1 | (0.082, 0.223, 0.891, 0.384) |
| Fig.3 | 词汇学习管线（图卷积编码器 → RQ → 1D ResNet 解码） | `figures/02-encoder-decoder.png` | p3 | (0.08, 0.075, 0.92, 0.34) |
| Fig.5 | GPT 式 transformer 自回归生成 token 序列 | `figures/03-transformer.png` | p4 | (0.101, 0.52, 0.891, 0.816) |
| Fig.6 | Chair/Table 定性对比（紧凑锐利 vs 过三角化） | `figures/04-qualitative.png` | p6 | (0.082, 0.091, 0.891, 0.689) |
| Fig.14 | 编码器-解码器网络细节（SAGEConv + 1D ResNet-34） | `figures/05-encoder-decoder-detail.png` | p15 | (0.082, 0.42, 0.891, 0.662) |

## 笔记中引用约定

- 所有图均使用 `assets/figures/<name>.png` 相对路径
- 在笔记正文中通过 `![描述](assets/figures/<name>.png)` 引用
- 裁切框由 `scan_figures.py` + 论文文本位置推断，对矢量绘制图以 `extract_figure.py` 整页渲染后裁切
- Fig.4（ResNet 解码器离散 vs 连续预测）、Fig.7（Bench/Lamp 对比）、Fig.10（消融定性）、Fig.16（per-vertex 量化直觉）未单独裁切，正文以文字表格呈现对应结论
