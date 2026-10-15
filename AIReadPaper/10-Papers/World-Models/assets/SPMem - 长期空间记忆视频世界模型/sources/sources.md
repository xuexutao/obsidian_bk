# SPMem 来源记录

- 获取日期：2026-10-15
- arXiv：<https://arxiv.org/abs/2506.05284>
- 官方 PDF：<https://arxiv.org/pdf/2506.05284> → `paper.pdf`（v1）
- 官方源码：<https://arxiv.org/e-print/2506.05284> → `source.tar`，解包于 `tex/`
- 项目页：<https://spmem.github.io/>

| 本地文件 | 原始文件 | 论文位置 | 获取方式 | 用途 |
|---|---|---|---|---|
| `../figures/01-teaser.png` | `tex/figures/teaser.pdf` | Figure 1，PDF p.1 | arXiv 源码原始矢量图转 216 DPI PNG | 展示三类记忆与长时动态生成 |
| `../figures/02-method.png` | `tex/figures/pipeline.pdf` | Figure 2，PDF p.4 | arXiv 源码原始矢量图转 216 DPI PNG | 系统与 Condition DiT 架构 |
| `../figures/03-results.png` | `tex/figures/qualitative.pdf` | Figure 4，PDF p.7 | arXiv 源码原始矢量图转 216 DPI PNG | 相机、重访、动作定性比较 |
| `../figures/04-ablation.png` | `tex/figures/ablation.pdf` | Figure 5，PDF p.8 | arXiv 源码原始矢量图转 216 DPI PNG | working/episodic memory 消融 |
| `../figures/05-failure.png` | `tex/figures/failurecase.pdf` | Figure 6，PDF p.9 | arXiv 源码原始矢量图转 216 DPI PNG | 快速大幅相机运动失败案例 |

## 说明

- 五张图均来自官方 arXiv 源码中的完整单页矢量图。
- 定量数字复核自 `tex/tables/main_metrics.tex`、`vbench.tex`、`net_abl.tex`。
