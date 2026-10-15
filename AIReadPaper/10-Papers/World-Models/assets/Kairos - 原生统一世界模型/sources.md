# Kairos — 材料溯源表

> 论文：Kairos: A Regret-Aware Native World-Action Model Stack for Physical AI
> arXiv：2606.16533 ｜ 项目页：https://github.com/kairos-agi/kairos ｜ 获取日期：2026-08-31

| 本地文件 | 材料类型 | 原始 URL | Figure / 页码 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | https://arxiv.org/pdf/2606.16533 | 全文（正文 + 附录，约 119 页） | 精读主来源 |
| `figures/01-teaser-motivation.png` | 论文 Figure（`fig:motivation`） | https://arxiv.org/abs/2606.16533 | §1 引言，第 1 页 | 研究背景章节 |
| `figures/02-framework.png` | 论文 Figure（`fig:kairos_3_0_framework`） | https://arxiv.org/abs/2606.16533 | §1 引言 | 核心方法章节 |
| `figures/03-architecture.png` | 论文 Figure（`fig:kairos_dit`，DiT block） | https://arxiv.org/abs/2606.16533 | §Model | 核心方法章节 |
| `figures/04-data-pyramid.png` | 论文 Figure（`fig:data_pyramid`） | https://arxiv.org/abs/2606.16533 | §Pretraining | 核心方法章节 |
| `figures/05-benchmark-wmbench.png` | 论文 Figure（`fig:vs` 子图，WorldModelBench-Robot） | https://arxiv.org/abs/2606.16533 | §1 引言 / §Evaluation | 实验章节 |
| `figures/06-benchmark-libero.png` | 论文 Figure（`fig:vs` 子图，LIBERO-Plus） | https://arxiv.org/abs/2606.16533 | §1 引言 / §Evaluation | 实验章节 |
| `figures/07-shift-curves.png` | 论文 Figure（`fig:kairos-shift-curves-stages`） | https://arxiv.org/abs/2606.16533 | §Pretraining | 训练细节章节 |
| LaTeX 源码包 | `export.arxiv.org/src/2606.16533` | https://export.arxiv.org/src/2606.16533 | 全部 section 源码 | 公式/图注/数字交叉验证 |

## 说明

- 图均从论文 LaTeX 源码包中的矢量图（`.pdf` / `.png`）转成高清 PNG，保证非 PDF 内嵌碎片。
- 页码说明：Kairos 技术报告为动态分页（119 页含附录），正文图未逐一标注固定页码，故溯源表以 **Figure label + 章节** 为准；仅有明确位于首页的 motivation 图标注第 1 页。
- 关键实验数字均已与 LaTeX 源码逐条核对（WorldModelBench-Robot 9.30、RoboTwin 2.0 平均 96.1、LIBERO-Plus 89.0/90.8、效率表 23.5GB/2.3PFlops/43s 等）。
