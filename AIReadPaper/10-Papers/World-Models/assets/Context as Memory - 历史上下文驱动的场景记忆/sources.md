# Context as Memory 来源记录

- 获取日期：2026-10-15
- arXiv：<https://arxiv.org/abs/2506.03141>
- 官方 PDF：<https://arxiv.org/pdf/2506.03141> → `paper.pdf`（最新 v2）
- 官方源码：<https://arxiv.org/e-print/2506.03141> → `source.tar`，解包于 `tex/`
- 项目页：<https://context-as-memory.github.io/>

| 本地文件 | 原始文件 | 论文位置 | 获取方式 | 用途 |
|---|---|---|---|---|
| `../figures/01-teaser.png` | `tex/figures/teaser.pdf` | Figure 1，PDF p.1 | arXiv 源码原始矢量图转 216 DPI PNG | 展示长轨迹重访时的场景一致性 |
| `../figures/02-method.png` | `tex/figures/framework.pdf` | Figure 3，PDF p.4 | arXiv 源码原始矢量图转 216 DPI PNG | 解释 Context-as-Memory 与 FOV 检索 |
| `../figures/03-results.png` | `tex/figures/comparison.pdf` | Figure 5，PDF p.6 | arXiv 源码原始矢量图转 216 DPI PNG | 定性比较 GT/历史重访一致性 |

## 说明

- 三张图均来自官方 arXiv v2 源码，不是网页缩略图或拼接碎片。
- 正文中的表格数字复核自 `tex/fig_tab_texs/tab_comp.tex`、`tab_ablation_context.tex`、`tab_ablation_retrieval.tex`。
