# FramePack 来源记录

- 获取日期：2026-10-15
- arXiv：<https://arxiv.org/abs/2504.12626>
- 官方 PDF：<https://arxiv.org/pdf/2504.12626> → `paper.pdf`（最新 v3，2025-10-14）
- 官方源码：<https://arxiv.org/e-print/2504.12626> → `source.tar`，解包于 `tex/`
- 代码：<https://github.com/lllyasviel/FramePack>

| 本地文件 | 原始文件 | 论文位置 | 获取方式 | 用途 |
|---|---|---|---|---|
| `../figures/01-packing-method.png` | `tex/imgs/pack2.pdf` | Figure 1，PDF p.4 | arXiv 源码原始矢量图转 216 DPI PNG | 时间/特征排序与 packing schedules |
| `../figures/02-drift-prevention.png` | `tex/imgs/sampling_v2.pdf` | Figure 2，PDF p.5 | arXiv 源码原始矢量图转 216 DPI PNG | 端点、反向、规划、离散历史机制 |
| `../figures/03-ablation-results.png` | `paper.pdf` | Table 1，PDF p.8 | 官方 PDF 按表格边界 216 DPI 完整裁取 | 关键消融与 ELO/漂移结果 |

## 说明

- Figure 1/2 直接来自官方 v3 源码矢量图。
- Table 1 在源码中以 LaTeX 表格存在，不是独立图片，因此从官方 PDF 完整裁取；数值另与 `framepack_arxiv.tex` 逐项复核。
