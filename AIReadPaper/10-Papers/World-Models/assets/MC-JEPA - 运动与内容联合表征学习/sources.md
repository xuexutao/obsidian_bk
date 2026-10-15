# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2307.12698 | 全文，共 20 页 | 2026-10-15 | 精读主来源 |
| `sources/arxiv-source.tar.gz` | arXiv 官方源码包 | https://arxiv.org/e-print/2307.12698 | 全部 TeX 与原始图 | 2026-10-15 | 公式、表格、原图溯源 |
| `sources/arxiv-source/` | 官方源码展开目录 | https://arxiv.org/e-print/2307.12698 | `iclr2024_conference.tex`、`appendix.tex`、`figures/` | 2026-10-15 | 核对正文、附录与实验细节 |
| `figures/01-teaser.png` | 官方原图渲染 | https://arxiv.org/e-print/2307.12698 | Figure 1，论文第 2 页；源码 `figures/mc_jepa_teaser.pdf` | 2026-10-15 | 解释运动/内容双任务动机 |
| `figures/02-method-pipeline.png` | 官方原图渲染 | https://arxiv.org/e-print/2307.12698 | Figure 2，论文第 3 页；源码 `figures/mc_jepa_arch.pdf` | 2026-10-15 | 解释完整训练 pipeline |
| `figures/03-optical-flow-results.png` | 官方原图 | https://arxiv.org/e-print/2307.12698 | Figure 3，论文第 7 页；源码 `figures/ultime_flow_viz.png` | 2026-10-15 | 光流定性对比 |

## 在线核验

- arXiv 论文页：https://arxiv.org/abs/2307.12698
- ICLR 2024 OpenReview 投稿页：https://openreview.net/forum?id=9XdLlbxZCC
- 录用状态核验：OpenReview 记录为 submitted to ICLR 2024，公开决策数据为 Reject；截至 2026-10-15，未找到正式会议录用证据。因此笔记将其记为 arXiv 预印本，而非已录用会议论文。
- 官方代码：论文与 arXiv 页面未给出可核验的官方代码仓库；`code` 保留为 `null`。

> 图片均直接取自 arXiv 官方源码中的完整 PDF/PNG 图件，并以 300 DPI 渲染；已逐张目视核验，不是从论文 PDF 抽出的零散位图。