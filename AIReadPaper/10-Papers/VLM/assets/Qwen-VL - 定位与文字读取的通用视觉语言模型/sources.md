# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2308.12966 | 全文，arXiv v3 | 2026-10-15 | 精读主来源 |
| `figures/01-capabilities.jpg` | 官方论文源码图 | https://arxiv.org/e-print/2308.12966 | Figure 2, p.2，源码 `images/demo2.jpg` | 2026-10-15 | 能力与定性结果 |
| `figures/02-training-pipeline.png` | 官方论文源码矢量图转 PNG | https://arxiv.org/e-print/2308.12966 | Figure 3, p.4，源码 `images/qwenvl.pdf` | 2026-10-15 | 三阶段训练流程 |
| `figures/03-benchmark-radar.png` | 官方论文源码矢量图转 PNG | https://arxiv.org/e-print/2308.12966 | Figure 1, p.1，源码 `images/radar1.pdf` | 2026-10-15 | 总体 benchmark 对比 |
| `figures/04-query-ablation.png` | 官方论文源码矢量图转 PNG | https://arxiv.org/e-print/2308.12966 | Figure 7, p.22，源码 `images/figure_q.pdf` | 2026-10-15 | learnable query 消融 |

## 提取与核验说明

- PDF 直接下载自 arXiv 官方地址，未使用发布博客替代论文。
- 图片优先从 arXiv 官方 LaTeX source 提取；PDF 矢量图使用 `extract_source_figures.py` 以高清 PNG 导出并裁除白边。
- 4 张最终图片均已逐图预览，内容完整，未使用 PDF 内嵌碎片。
- 论文未提供独立 supplementary 文件；附录已包含在 `paper.pdf` 中。
