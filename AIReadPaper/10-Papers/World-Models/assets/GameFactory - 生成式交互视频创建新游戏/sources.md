# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v4） | https://arxiv.org/pdf/2501.08325 | 全文，含附录 | 2026-09-04 | 精读主来源 |
| `figures/01-teaser.png` | 开放域动作控制结果 | https://arxiv.org/pdf/2501.08325 | Figure 1, p.1 | 2026-09-04 | 任务动机与总体能力 |
| `figures/02-action-module.png` | 动作注入架构图 | https://arxiv.org/pdf/2501.08325 | Figure 3, p.4 | 2026-09-04 | 键盘/鼠标动作控制机制 |
| `figures/03-multiphase-training.png` | 多阶段解耦训练图 | https://arxiv.org/pdf/2501.08325 | Figure 6, p.6 | 2026-09-04 | 风格—动作解耦与开放域泛化 |
| `figures/04-control-and-generalization-ablation.png` | 控制与泛化消融 | https://arxiv.org/pdf/2501.08325 | Table 2–3, Figure 7–8, p.7 | 2026-09-04 | 动作注入与训练策略消融 |
| `figures/05-dataset-and-long-video-results.png` | 数据偏差与长视频结果 | https://arxiv.org/pdf/2501.08325 | Table 4–6, Figure 9–10, p.8 | 2026-09-04 | 数据集、长视频与失败行为分析 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2501.08325
- 官方项目页：https://yujiwen.github.io/gamefactory/
- 官方代码入口（当前重定向至 KlingAIResearch）：https://github.com/KwaiVGI/GameFactory
- GF-Minecraft 数据集入口：https://huggingface.co/datasets/KwaiVGI/GameFactory-Dataset

## 获取与缺口说明

- arXiv v4 已把数据、动作空间和补充实验并入论文附录，未发现独立 supplementary PDF，因此未保存 `supplementary.pdf`。
- 官方 GitHub 仓库截至获取日只公开 README、图片、`detection.py` 和示例视频等少量文件，未见论文所述 1B 模型的完整训练/推理代码、模型权重或多阶段训练配置；因此本文可依据论文复述与估算复现成本，但不能端到端复现。
- 论文使用内部 1B 文生视频扩散模型，基础模型及其开放域预训练数据未公开，这是最主要的可复现性缺口。
- 二手博客 https://asdfo123.github.io/blog/world-model/ 仅作为线索，没有作为正文定量结论的证据来源。
