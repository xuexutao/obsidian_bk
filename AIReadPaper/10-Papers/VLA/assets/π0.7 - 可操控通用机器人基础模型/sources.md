# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2604.15483> | 全文，arXiv:2604.15483v2 | 2026-10-20 | 精读主来源；prompt、架构、训练、实验、消融与附录核对 |
| `figures/01-teaser.png` | 数据与能力总览图 | <https://arxiv.org/pdf/2604.15483> | Figure 1，p.1 | 2026-10-20 | 说明多源数据、diverse prompting 与开箱能力 |
| `figures/02-architecture.png` | 模型架构图 | <https://arxiv.org/pdf/2604.15483> | Figure 2，p.4 | 2026-10-20 | 说明 5B VLA、高层策略、BAGEL world model、metadata 与 subgoals |
| `figures/03-cross-embodiment.png` | 跨本体结果图 | <https://arxiv.org/pdf/2604.15483> | Figure 12，p.12 | 2026-10-20 | 比较 π0.5/π0.6/π0.7/π0.7(GC) 的多任务跨本体迁移 |

## Additional official source

- 项目页：<https://pi.website/pi07>（论文首页给出的官方链接；用于项目介绍和视频入口，笔记中的数字以 PDF 为准）。

## Notes

- 论文正式标题：*π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities*。
- DOI：<https://doi.org/10.48550/arXiv.2604.15483>。
- 本附件目录未提供 supplementary material、独立代码仓库或公开权重。
- 论文多项结果以柱状图展示但未印逐项数字；笔记只对正文明确给出的数值（如 UR5e shirt folding 的 85.6% progress / 80% success、人类 90.9% / 80.6%）作精确引用。
