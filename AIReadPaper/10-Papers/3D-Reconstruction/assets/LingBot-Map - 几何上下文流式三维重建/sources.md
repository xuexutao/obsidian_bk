# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/abs/2604.14141 / https://raw.githubusercontent.com/Robbyant/lingbot-map/master/lingbot-map_paper.pdf | 全文，v3 | 2026-10-26 | 精读主来源 |
| `figures/01-overview-results.png` | 论文概览图 | https://arxiv.org/abs/2604.14141 | Figure 1, p.1 | 2026-10-26 | 总体性能概览 |
| `figures/02-gca-attention-mask.png` | 注意力模式图 | https://arxiv.org/abs/2604.14141 | Figure 3, p.6 | 2026-10-26 | 解释 GCA mask 与复杂度 |
| `figures/03-method-pipeline.png` | 方法架构图 | https://arxiv.org/abs/2604.14141 | Figure 4, p.7 | 2026-10-26 | 核心 pipeline |
| `figures/04-oxford-results.png` | 定量结果表 | https://arxiv.org/abs/2604.14141 | Table 2-3, p.13 | 2026-10-26 | Oxford Spires 位姿与长序列结果 |
| `figures/05-pointcloud-comparison.png` | 定性点云对比 | https://arxiv.org/abs/2604.14141 | Figure 6, p.16 | 2026-10-26 | 重建质量与漂移分析 |
| `figures/06-ablation.png` | 消融表 | https://arxiv.org/abs/2604.14141 | Table 7, p.18 | 2026-10-26 | Anchor/Memory/RoPE/relative loss 贡献 |

## 在线材料

| 材料 | URL | 获取日期 | 用途 |
|---|---|---|---|
| 官方项目页 | https://technology.robbyant.com/lingbot-map | 2026-10-26 | 方法概览、Demo 与官方入口 |
| 官方代码仓库 | https://github.com/Robbyant/lingbot-map | 2026-10-26 | 代码结构、推理和工程限制 |
| 官方模型卡 | https://huggingface.co/robbyant/lingbot-map | 2026-10-26 | checkpoint 与运行说明 |
| 官方仓库 commit | `b9dd1b1de535e6d36e777fb42b8433c781bfd783` | 2026-10-26 | 代码核查版本 |

## 说明

- 图像均从官方 PDF 对应页面按 3× 分辨率完整裁剪，保留原图/表及图注，未使用 PDF 内嵌碎片。
- 官方仓库历史修复过 FlashInfer 与 SDPA 的 KV-cache bug，复现实验应固定 commit，并优先使用最新稳定代码。
- 项目 README 的超长视频演示、关键帧与窗口参数属于工程实现信息，不应替代论文主实验数字。
