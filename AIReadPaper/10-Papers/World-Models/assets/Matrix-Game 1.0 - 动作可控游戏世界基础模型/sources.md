# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2506.18701 | 全文，arXiv v1，26 页 | 2026-10-16 | 精读主来源 |
| `figures/01-data-pipeline.png` | arXiv LaTeX 源码原图 | https://export.arxiv.org/e-print/2506.18701 | Figure 3，PDF p.6，`asset/data.pdf` | 2026-10-16 | 无标签数据过滤流程 |
| `figures/02-overview.png` | arXiv LaTeX 源码原图 | https://export.arxiv.org/e-print/2506.18701 | Figure 4，PDF p.8，`asset/Pipeline2.pdf` | 2026-10-16 | image-to-world 总体流程 |
| `figures/03-architecture.png` | arXiv LaTeX 源码原图 | https://export.arxiv.org/e-print/2506.18701 | Figure 5，PDF p.9，`asset/architecture_try.pdf` | 2026-10-16 | 自回归与模型架构 |
| `figures/04-failure-cases.png` | arXiv LaTeX 源码原图拼接 | https://export.arxiv.org/e-print/2506.18701 | Figure 11，PDF p.18，`asset/failure_case1_new.pdf` + `failure_case2_new.pdf` | 2026-10-16 | 边缘场景与物理失败案例 |

## 在线来源

- arXiv 摘要页：https://arxiv.org/abs/2506.18701
- arXiv HTML：https://arxiv.org/html/2506.18701
- 官方项目页：https://matrix-game-homepage.github.io/
- Matrix-Game 系列仓库：https://github.com/SkyworkAI/Matrix-Game
- 1.0 代码目录：https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-1
- 模型权重：https://huggingface.co/Skywork/Matrix-Game

## 开源与缺口核验

- 1.0 目录公开了推理代码、双卡/并行推理脚本、配置、模型实现和 GameWorld Score；仓库声明 MIT License。
- 官方权重已公开；README 要求 NVIDIA A100/H100，单次 65 帧推理需至少 80 GB 显存。
- 未找到完整训练脚本、2,700/1,200 小时训练数据下载、MineRL/Unreal 数据生产全流水线或论文训练集快照，因而不能严格复现从数据到 17B 权重的训练过程。
- 论文没有正式内部组件消融表；本文不把 baseline 对比误称为消融。
