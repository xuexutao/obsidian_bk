# TripoSG 关键图与官方资料溯源

## 官方源材料

| 资料 | 来源 | 用途 | 获取日期 |
| --- | --- | --- | --- |
| 论文 PDF（v1, 22 页） | https://arxiv.org/abs/2502.06608 | 主参考 | 2026-08-25 |
| 项目页 | https://www.tripo3d.ai/work/triposg | Demo / 视频 | 2026-08-25 |

## 关键图（图编号 → 裁切 PNG → 原文位置）

| 论文图号 | 描述 | 本地文件 | PDF 页码 | 裁切框（相对页面） |
| --- | --- | --- | --- | --- |
| Fig.1 | 输入图 + 4 个生成结果 | `figures/01-teaser.png` | p1 | (0.091, 0.260, 0.885, 0.605) |
| Fig.3 | 整流流架构 + 块内结构 + MoE | `figures/03-architecture.png` | p5 | (0.05, 0.04, 0.95, 0.52) |
| Fig.4 | VAE 编码器-解码器 + 表面法向引导 | `figures/04-vae-architecture.png` | p7 | (0.235, 0.085, 0.55, 0.42) |
| Fig.5 | Occupancy vs SDF 重建质量对比 | `figures/05-occupancy-vs-sdf.png` | p7 | (0.55, 0.085, 0.95, 0.38) |
| Fig.6 | 数据构建系统 4 阶段管线 | `figures/06-data-building-system.png` | p9 | (0.05, 0.04, 0.95, 0.55) |
| Fig.7 | 与 6 个 SOTA baseline 对比 | `figures/07-comparison-sota.png` | p11 | (0.114, 0.085, 0.885, 0.680) |

## 笔记中引用约定

- 所有图均使用 `assets/figures/<name>.png` 相对路径
- 在笔记正文中通过 `![描述](assets/figures/<name>.png)` 引用
- 裁切框由 `scan_figures.py` + 论文文本位置推断，对矢量绘制图（如 Fig.3/Fig.4/Fig.6）以 `extract_figure.py` 整页渲染后裁切
