# 来源记录 — π³: Permutation-Equivariant Visual Geometry Learning

本文件记录撰写 `π³ - 置换等变视觉几何学习.md` 精读笔记所用到的原始材料与图片来源，便于溯源与复现。

## 官方材料

| 材料 | 说明 | 本地路径 | 获取日期 |
| --- | --- | --- | --- |
| 论文 PDF | 官方 arXiv 版全文（ICLR 2026 camera-ready） | `sources/paper.pdf` | 2026-08-26 |
| 原始链接 | arXiv: 2507.13347 | https://arxiv.org/abs/2507.13347 | 2026-08-26 |

## 关键图来源（figures/）

| 文件名 | 对应论文图 | 页码 | 用途 | 裁剪方式 |
| --- | --- | --- | --- | --- |
| `fig1_teaser.png` | Figure 1（teaser） | Page 1 | 说明"取消固定参考帧、置换等变"的核心主张 | 按页面图片块坐标裁剪（scan_figures.py 定位 + extract_figure.py，ZOOM=2.4） |
| `fig3_architecture.png` | Figure 3（架构对比 Type A / Type B / π³） | Page 4 | 说明 π³ 如何通过去除参考帧实现置换等变 | 同上 |
| `fig5_reconstruction.png` | Figure 5（多视图重建定性对比） | Page 8 | 展示重建质量对比 | 同上 |

## 说明

- 全文文本由 PyMuPDF 提取，用于方法、公式、实验数据（Table 1–7）与结论的精读。
- 所有实验数值均直接取自论文 Table 1–7，未做二次加工。
- 图片均从官方 PDF 整页按图像块坐标完整裁剪，未使用 PDF 内嵌碎片。
