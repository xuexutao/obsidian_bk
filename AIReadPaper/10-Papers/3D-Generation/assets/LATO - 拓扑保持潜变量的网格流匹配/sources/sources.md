# LATO 关键图与官方资料溯源

## 官方源材料

| 资料 | 来源 | 用途 | 获取日期 |
| --- | --- | --- | --- |
| 论文 PDF（v1, 16 页） | https://arxiv.org/abs/2603.06357 | 主参考 | 2026-08-25 |
| arXiv 页面 | https://arxiv.org/abs/2603.06357 | 元数据 / 版本 | 2026-08-25 |

## 关键图（图编号 → 裁切 PNG → 原文位置）

| 论文图号 | 描述 | 本地文件 | PDF 页码 | 裁切框（相对页面） |
| --- | --- | --- | --- | --- |
| Fig.1 | teaser：稀疏体素结构合成 + T-Voxels 渐进去噪，网格逐步成形 | `figures/01-teaser.png` | p1 | (0.091, 0.238, 0.885, 0.548) |
| Fig.2 | LATO vs 既有范式（vecset/voxel 隐式 vs 逐面显式 vs T-Voxels） | `figures/02-paradigm.png` | p2 | (0.08, 0.07, 0.92, 0.45) |
| Fig.3 | LATO 管线总览（VDF → 稀疏体素 VAE → T-Voxels → 分层细分 + connection head） | `figures/03-pipeline.png` | p4 | (0.091, 0.085, 0.885, 0.389) |
| Fig.4 | 几何条件生成对比（vs BPT/DeepMesh/FastMesh/MeshSilkSong/MeshAnyV2） | `figures/04-generation.png` | p6 | (0.091, 0.100, 0.885, 0.531) |
| Fig.5 | 与隐式基础模型（Tripo3D/TRELLIS/Hunyuan3D/CLAY）的拓扑对比 | `figures/05-topology.png` | p6 | (0.091, 0.55, 0.885, 0.85) |

## 笔记中引用约定

- 所有图均使用 `assets/figures/<name>.png` 相对路径
- 在笔记正文中通过 `![描述](assets/figures/<name>.png)` 引用
- 裁切框由 `scan_figures.py` + 论文文本位置推断；Fig.2/Fig.3 为矢量示意图，以 `extract_figure.py` 整页渲染后裁切
- Fig.7（推理时间曲线）、Fig.8（城市级生成）、Fig.9（VDF/T-Voxel 消融）、Fig.10（顶点数条件）未单独裁切，正文以文字/表格呈现对应结论
