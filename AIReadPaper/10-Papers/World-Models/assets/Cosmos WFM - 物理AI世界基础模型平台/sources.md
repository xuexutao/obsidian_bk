# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2501.03575 | 全文，75 页 | 2026-10-16 | 精读主来源 |
| `figures/01-overview.png` | Cosmos 平台能力总览 | https://arxiv.org/pdf/2501.03575 | Figure 1, p.2 | 2026-10-16 | 说明预训练与三类后训练应用 |
| `figures/02-tokenizer-architecture.png` | Cosmos Tokenizer 架构 | https://arxiv.org/pdf/2501.03575 | Figure 9, p.14 | 2026-10-16 | 解释因果 wavelet 编解码器与时空压缩 |
| `figures/03-diffusion-wfm-architecture.png` | Diffusion WFM 架构 | https://arxiv.org/pdf/2501.03575 | Figure 11, p.20 | 2026-10-16 | 解释连续 latent、DiT、T5 条件与去噪目标 |
| `figures/04-robot-human-evaluation.png` | 机器人指令视频预测人评 | https://arxiv.org/pdf/2501.03575 | Figure 24, p.46 | 2026-10-16 | 机器人后训练定量证据 |

## 官方在线来源

- arXiv 摘要与版本页：https://arxiv.org/abs/2501.03575
- NVIDIA Research 论文页：https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai
- Cosmos-Predict1 官方代码：https://github.com/nvidia-cosmos/cosmos-predict1
- 官方模型集合：https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7
- ShotBench：https://github.com/NVlabs/ShotBench
- TokenBench：https://github.com/NVlabs/TokenBench

## 核验说明

- 四张图片均从本地官方 PDF 按原始 Figure 页面完整裁剪，保留图号或图注，并已用 `Read` 逐张视觉核验。
- 论文主体依据 PDF；官方研究页用于核对发布日期与开放说明；代码仓库用于核对当前模型名称、权重、许可证和仓库迁移状态。
- 论文明确披露完整平台训练使用约 10,000 张 H100、持续三个月；未披露每个模型的独立 GPU-hours，因此复现成本只能分层估算，不能给出精确金额。
