# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2412.12095 | 全文，arXiv v2 | 2026-09-04 | 精读主来源 |
| `figures/01-dual-factorization.png` | 双重因子分解总览 | https://arxiv.org/pdf/2412.12095 | Figure 1, p.1 | 2026-09-04 | 研究动机与统一视角 |
| `figures/02-zero-shot-editing.png` | 生成与零样本编辑结果 | https://arxiv.org/pdf/2412.12095 | Figure 2, p.2 | 定性结果与编辑能力 |
| `figures/03-dit-vs-causalfusion.png` | 架构对比图 | https://arxiv.org/pdf/2412.12095 | Figure 3, p.4 | 核心方法与注意力机制 |
| `figures/04-task-difficulty.png` | 任务难度分析 | https://arxiv.org/pdf/2412.12095 | Figure 4, p.5 | AR step 采样与损失分析 |
| `figures/05-multimodal-generation.png` | 多模态生成结果 | https://arxiv.org/pdf/2412.12095 | Figure 5, p.7 | 文生图与图像描述实验 |

## 官方在线资源

- arXiv 论文页：https://arxiv.org/abs/2412.12095
- arXiv HTML：https://arxiv.org/html/2412.12095v2
- 官方代码：https://github.com/causalfusion/causalfusion
- 预训练 CausalFusion-XL：官方仓库 README 中提供 Google Drive 下载链接
- 复现依赖：官方仓库要求 PyTorch 2.5.1、torchvision 0.20.1、transformers 4.46.2，并依赖 MAR 的预训练 VAE

## 获取说明

- 论文没有单独的 Supplementary Material 文件；附录 A–D 已包含在 21 页官方 PDF 中。
- 关键图均从官方 PDF 按完整图表区域裁剪，保留 Figure 编号与图注；未使用 PDF 内嵌的零散图片碎片。
- 官方仓库公开了训练、采样代码和 CausalFusion-XL 权重，但 README 未注明许可证，也未给出完整 800 epoch 系统级训练的硬件时长或总 GPU-hours。
