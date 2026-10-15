# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v4） | https://arxiv.org/pdf/2508.13009 | 全文 | 2026-09-04 | 精读主来源 |
| `figures/01-teaser.png` | 实时交互生成总览 | https://arxiv.org/pdf/2508.13009 | Figure 1, p.2 | 2026-09-04 | 能力与场景范围 |
| `figures/02-architecture.png` | 模型架构与动作模块 | https://arxiv.org/pdf/2508.13009 | Figure 8, p.9 | 2026-09-04 | 输入、DiT 与动作注入 |
| `figures/03-self-forcing.png` | Self-Forcing 蒸馏流程 | https://arxiv.org/pdf/2508.13009 | Figure 10, p.10 | 2026-09-04 | 因果学生与分布匹配 |
| `figures/04-minecraft-results.png` | Minecraft 基线对比 | https://arxiv.org/pdf/2508.13009 | Figure 11, Table 1, p.11 | 2026-09-04 | 定性与主要定量结果 |
| `figures/05-long-video-speed-ablation.png` | 长视频与速度消融 | https://arxiv.org/pdf/2508.13009 | Figure 13, Table 3, p.13 | 2026-09-04 | 1350 帧与 25 FPS 证据 |
| `figures/06-kv-cache-and-failure-cases.png` | KV cache 消融与失败案例 | https://arxiv.org/pdf/2508.13009 | Figure 16–17, p.15 | 2026-09-04 | 长序列退化与 OOD 限制 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2508.13009
- 官方项目页：https://matrix-game-v2.github.io/
- 官方代码（Matrix-Game-2 子目录）：https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2
- 官方模型权重：https://huggingface.co/Skywork/Matrix-Game-2.0
- GameWorld Score 评测代码：https://github.com/SkyworkAI/Matrix-Game/tree/main/GameWorldScore

## 获取与缺口说明

- 技术报告未提供独立 supplementary PDF，补充信息主要在正文与代码仓库中。
- 官方仓库公开了推理与流式推理代码、配置和三个场景权重；README 给出的最低环境是 Linux、24 GB NVIDIA GPU、64 GB RAM，并依赖 Apex 与 FlashAttention。论文的 25 FPS 是单张 H100 上的吞吐，不等同于 24 GB GPU 可达到相同速度。
- 仓库未提供论文所述 Unreal Engine/GTA5 全量数据生产系统、完整训练脚本与约 800/1200 小时训练数据，因此可以复现推理，无法严格复现数据生成和训练过程。
- 论文摘要称约 1200 小时，但实验主训练集写作约 800 小时（153h Minecraft + 615h Unreal，另含 85h Sekai；GTA/TempleRun 为额外微调）。正文中已区分不同统计口径。
- 二手博客 https://asdfo123.github.io/blog/world-model/ 仅作线索，所有公式与数字以官方论文、项目页和代码仓库为准。
