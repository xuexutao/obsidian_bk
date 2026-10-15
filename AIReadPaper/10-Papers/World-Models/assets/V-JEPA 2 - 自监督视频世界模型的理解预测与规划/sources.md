# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2506.09985> | 全文（含附录） | 2026-09-03 | 精读主来源 |
| `figures/01-overview.png` | Figure 1：V-JEPA 2 Overview（1M 小时互联网视频 + 1M 图像 → 视频预训练 → 语言对齐 / Attentive Probe / 动作条件后训练） | arXiv:2506.09985 | 第 2 页 | 2026-09-03 | 总体框架 |
| `figures/02-multistage-training.png` | Figure 2：Multistage training（左：V-JEPA 2 mask denoising；右：V-JEPA 2-AC 动作条件自回归预测） | arXiv:2506.09985 | 第 4 页 | 2026-09-03 | 核心方法（两阶段训练） |
| `figures/03-scaling-ingredients.png` | Figure 3：Scaling Ingredients 柱状图（VJEPA 1 baseline 84.2 → +Data → +Model → +Long Training → +Higher Resolution 88.2） | arXiv:2506.09985 | 第 5 页 | 2026-09-03 | 关键实验（scaling 消融） |
| <https://github.com/facebookresearch/vjepa2> | 官方代码仓库（V-JEPA 2 + V-JEPA 2-AC） | — | — | 2026-09-03 | 复现参考 |
| <https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks> | 官方博客 | — | — | 2026-09-03 | 实验背景与对比 |