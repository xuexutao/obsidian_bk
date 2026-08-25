# LATTICE 关键图与官方资料溯源

## 官方源材料

| 资料 | 来源 | 用途 | 获取日期 |
| --- | --- | --- | --- |
| 论文 PDF（v1, 14 页） | https://arxiv.org/abs/2512.03052 | 主参考 | 2026-08-25 |
| 项目页 | https://lattice3d.github.io | Demo / 视频 | 2026-08-25 |

## 关键图（图编号 → 裁切 PNG → 原文位置）

| 论文图号 | 描述 | 本地文件 | PDF 页码 | 裁切框（相对页面） |
| --- | --- | --- | --- | --- |
| Fig.1 | 单图 3D 生成 teaser（多视角 3D 资产） | `figures/01-teaser.png` | p1 | (0.096, 0.271, 0.904, 0.619) |
| Fig.4 | 三种 latent 表征 + 三种 query 类型对比 | `figures/04-representations.png` | p4 | (0.521, 0.092, 0.903, 0.356) |
| Fig.5 | LATTICE 整体管线 + VoxSet VAE + VoxSet DiT 架构 | `figures/05-architecture.png` | p5 | (0.096, 0.120, 0.892, 0.380) |
| Fig.8 | 与 5 个 SOTA 开源方法视觉对比 | `figures/08-comparison.png` | p7 | (0.096, 0.091, 0.904, 0.677) |
| Fig.11 | 用户研究胜率（vs 4 个商业模型） | `figures/11-user-study.png` | p8 | (0.519, 0.420, 0.904, 0.760) |

## 笔记中引用约定

- 所有图均使用 `assets/figures/<name>.png` 相对路径
- 在笔记正文中通过 `![描述](assets/figures/<name>.png)` 引用
- 裁切框由 `scan_figures.py` + 论文文本位置推断，对矢量绘制图以 `extract_figure.py` 整页渲染后裁切
