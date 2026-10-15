# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v2） | https://arxiv.org/pdf/2602.22208 | 全文，含附录 | 2026-09-04 | 精读主来源 |
| `figures/01-teaser.png` | 双玩家联合生成样例 | https://arxiv.org/pdf/2602.22208 | Figure 1, p.3 | 2026-09-04 | 任务定义与多视角一致性 |
| `figures/02-solaris-engine.png` | 多人数据采集系统 | https://arxiv.org/pdf/2602.22208 | Figure 2, p.5 | 2026-09-04 | 控制器、相机、服务器与技能库 |
| `figures/03-multiplayer-dit.png` | 多人 DiT 模块 | https://arxiv.org/pdf/2602.22208 | Figure 5, p.8 | 2026-09-04 | 玩家共享注意力与条件注入 |
| `figures/04-checkpointed-self-forcing.png` | Checkpointed Self Forcing | https://arxiv.org/pdf/2602.22208 | Figure 7, p.11 | 2026-09-04 | 内存优化、mask 与 KV 反传 |
| `figures/05-qualitative-comparison.png` | 长序列基线对比 | https://arxiv.org/pdf/2602.22208 | Figure 9, p.13 | 2026-09-04 | 224 帧稳定性与失败案例 |
| `figures/06-quantitative-and-ablation.png` | 定量结果与 Self-Forcing 消融 | https://arxiv.org/pdf/2602.22208 | Table 2–3, p.15 | 2026-09-04 | 五项多人能力与消融 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2602.22208
- 官方项目页：https://solaris-wm.github.io/
- 模型代码：https://github.com/solaris-wm/solaris
- 数据引擎代码：https://github.com/solaris-wm/solaris-engine
- 模型集合：https://huggingface.co/collections/nyu-visionx/solaris-models
- 数据集合：https://huggingface.co/collections/nyu-visionx/solaris-data
- 训练数据：https://huggingface.co/datasets/nyu-visionx/solaris-training-dataset
- 评测数据：https://huggingface.co/datasets/nyu-visionx/solaris-eval-datasets

## 获取与缺口说明

- arXiv v2 已含系统、数据、训练、评测和 Self Forcing 附录，未发现独立 supplementary PDF。
- 官方 JAX 代码公开了四阶段训练和 GPU/TPU 推理；GPU 推理要求每卡至少 48 GB，训练仅官方支持至少 95 GB 显存的 TPU v5p。单玩家阶段使用 v5p-128，其余训练使用 v5p-64，完整训练成本极高。
- 训练数据、评测数据、模型权重与 SolarisEngine 均有官方入口，开源完整度明显高于前两篇；但论文没有披露总 TPU 小时、模型参数量、输出分辨率和推理 FPS。
- 当前模型仅训练并评测两名玩家；“可扩展到任意玩家数”是架构层面的陈述，尚无超过两人的实验证据。
- 二手博客 https://asdfo123.github.io/blog/world-model/ 仅作线索，正文结论均来自官方论文、项目页、代码或模型/数据仓库。
