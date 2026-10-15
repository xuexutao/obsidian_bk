# VMem 来源记录

- 获取日期：2026-10-15
- arXiv：<https://arxiv.org/abs/2506.18903>
- 官方 PDF：<https://arxiv.org/pdf/2506.18903> → `paper.pdf`（最新 v3）
- 官方源码：<https://arxiv.org/e-print/2506.18903> → `source.tar`，解包于 `tex/`
- 项目页：<https://v-mem.github.io>

| 本地文件 | 原始文件 | 论文位置 | 获取方式 | 用途 |
|---|---|---|---|---|
| `../figures/01-teaser.png` | `tex/imgs/vmem_teaser.pdf` | Figure 1，PDF p.1 | arXiv 源码原始矢量图转 216 DPI PNG | 展示有/无记忆的长轨迹一致性 |
| `../figures/02-method.png` | `tex/imgs/vmem_method.pdf` | Figure 2，PDF p.3 | arXiv 源码原始矢量图转 216 DPI PNG | 完整读写与生成闭环 |
| `../figures/03-memory-index.png` | `tex/imgs/vmem_spatial_memory.pdf` | Figure 4，PDF p.5 | arXiv 源码原始矢量图转 216 DPI PNG | Surfel 索引读写细节 |
| `../figures/04-results.png` | `tex/imgs/vmem_long-term_comparison_figure.pdf` | Figure 6，PDF p.8 | arXiv 源码原始矢量图转 216 DPI PNG | ≥400 帧 cycle trajectory 定性比较 |

## 说明

- 四张图均为官方 arXiv v3 源码中的完整矢量原图。
- 定量数字复核自 `tex/tables/long-term-benchmark.tex`、`cycle_result.tex`、`tank_results.tex`、`ablation.tex`。
