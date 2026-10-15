# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2604.08995 | 全文，v2 | 2026-09-04 | 精读主来源 |
| `figures/01-teaser.png` | 官方论文原图 | https://arxiv.org/abs/2604.08995 | Figure 1 | 2026-09-04 | 展示长程记忆、动作控制与分钟级生成 |
| `figures/02-overview.png` | 官方论文原图 | https://arxiv.org/abs/2604.08995 | Figure 2 | 2026-09-04 | 数据—训练—部署总体框架 |
| `figures/03-memory-model.png` | 官方论文原图 | https://arxiv.org/abs/2604.08995 | Figure 4 | 2026-09-04 | 误差缓冲、长期记忆与统一注意力 |
| `figures/04-dmd-distillation.png` | 官方论文原图 | https://arxiv.org/abs/2604.08995 | Figure 6 | 2026-09-04 | 多 segment 自回归 DMD 蒸馏 |

## 官方链接

- arXiv：https://arxiv.org/abs/2604.08995
- arXiv HTML：https://arxiv.org/html/2604.08995v2
- 项目页：https://matrix-game-v3.github.io/
- 团队仓库（博客所引）：https://github.com/SkyworkAI/Matrix-Game

## 备注

- 图片从 arXiv 官方源码包中的 `asset/*.pdf` 转为高清 PNG，并逐张检查完整性。
- 论文未提供独立 Limitations 章节；正文中明确的系统瓶颈与报告证据边界已在笔记中分开标注。
- 论文声称的 720p@40 FPS 来自 8 张 GPU 运行 DiT、1 张 GPU 运行 VAE 的异步 8+1 GPU 配置，不应理解为单卡性能。
