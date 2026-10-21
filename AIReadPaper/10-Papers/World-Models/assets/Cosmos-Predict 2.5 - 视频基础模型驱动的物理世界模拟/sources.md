# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2511.00062 | 全文，44 页 | 2026-10-16 | 精读主来源 |
| `figures/01-curation-pipeline.png` | 视频策展流水线 | https://arxiv.org/pdf/2511.00062 | Figure 1, p.5 | 2026-10-16 | 解释 35M 小时原始视频到 200M clip 的治理流程 |
| `figures/02-unified-architecture.png` | Predict2.5 统一架构 | https://arxiv.org/pdf/2511.00062 | Figure 2, p.10 | 2026-10-16 | 解释 Reason1 文本编码与统一 T2W/I2W/V2W DiT |
| `figures/03-rl-human-vote.png` | RL 后训练人评 | https://arxiv.org/pdf/2511.00062 | Figure 5, p.13 | 2026-10-16 | 解释 RL 后训练的偏好收益 |
| `figures/04-long-video-error.png` | 长视频误差累积 | https://arxiv.org/pdf/2511.00062 | Figure 10, p.21 | 2026-10-16 | 比较 Transfer2.5 与 Transfer1 的长程稳定性 |

## 官方在线来源

- arXiv 摘要与版本页：https://arxiv.org/abs/2511.00062
- Cosmos-Predict2.5 官方代码：https://github.com/nvidia-cosmos/cosmos-predict2.5
- Cosmos-Transfer2.5 官方代码：https://github.com/nvidia-cosmos/cosmos-transfer2.5
- NVIDIA Cosmos 产品入口：https://www.nvidia.com/en-us/ai/cosmos/

## 核验说明

- 四张图片均从本地官方 PDF 按原始 Figure 页面完整裁剪，保留图号/图注，并已用 `Read` 逐张视觉核验；RL 人评图第一次裁剪截断图注，已重新裁剪并再次核验。
- 论文 PDF 是架构、公式、数据、实验与消融的主来源；官方仓库用于核对检查点、许可、推理/后训练入口及“已转入有限维护、推荐迁移 Cosmos 3”的当前状态。
- 训练效率表披露 4096 张 H100 下的 MFU，但没有给出总训练步数对应的完整 GPU-hours；复现成本不做虚构金额估算。
