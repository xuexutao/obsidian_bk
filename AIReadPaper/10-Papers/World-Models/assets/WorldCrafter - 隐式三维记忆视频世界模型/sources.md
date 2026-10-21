# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2609.24984 | 全文，arXiv v1 | 2026-10-20 | 精读主来源 |
| `figures/01-teaser.png` | 论文关键图 | https://arxiv.org/pdf/2609.24984 | Figure 1，PDF p.1 | 2026-10-20 | 能力、动机与任务概览 |
| `figures/02-method-pipeline.png` | 论文方法图 | https://arxiv.org/pdf/2609.24984 | Figure 2，PDF p.3 | 2026-10-20 | 总体 pipeline、记忆写入与读取 |
| `figures/03-static-revisit.png` | 静态场景定性结果 | https://arxiv.org/pdf/2609.24984 | Figure 3，PDF p.6 | 2026-10-20 | 静态闭环重访与点云一致性 |
| `figures/04-dynamic-revisit.png` | 动态场景定性结果 | https://arxiv.org/pdf/2609.24984 | Figure 4，PDF p.7 | 2026-10-20 | 动态主体与长时重访对比 |
| `figures/05-memory-ablation.png` | 消融与效率图 | https://arxiv.org/pdf/2609.24984 | Table 4、Figure 5，PDF p.8 | 2026-10-20 | 记忆设计、联合训练和延迟分析 |

## 官方在线材料

- arXiv 页面：https://arxiv.org/abs/2609.24984
- 项目页：https://drexubery.github.io/WorldCrafter
- 官方代码：https://github.com/TencentARC/WorldCrafter
- Base 权重：https://huggingface.co/TencentARC/WorldCrafter-Base
- Fast 权重：https://huggingface.co/TencentARC/WorldCrafter-Fast
- 在线演示：https://huggingface.co/spaces/Drexubery/worldcrafter-demo
- 演示视频：https://www.youtube.com/watch?v=sg09ftQOl0E

## 取材说明

- 笔记中的方法、公式、训练配置和实验数字以 arXiv v1 官方 PDF 为准。
- 项目页和官方仓库只用于补充公开推理接口、环境配置、相机动作 DSL、权重与许可证信息。
- 五张本地图均从官方 PDF 的完整页面区域裁剪，保留图中标注和原始图注；未使用 PDF 内嵌的碎片位图。
- 截至 2026-10-20，论文未提供独立 Supplementary Material；附录信息也未单独公开。
