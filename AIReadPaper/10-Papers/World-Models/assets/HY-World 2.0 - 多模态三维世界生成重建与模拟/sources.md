# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v1） | https://arxiv.org/pdf/2604.14268 | 全文，44 页 | 2026-10-16 | 精读主来源 |
| `figures/01-teaser.png` | arXiv 源码原图 | https://arxiv.org/e-print/2604.14268 | Figure 1，PDF p.3 | 2026-10-16 | 生成、重建、模拟能力概览 |
| `figures/02-overview.png` | arXiv 源码原图 | https://arxiv.org/e-print/2604.14268 | Figure 2，PDF p.5 | 2026-10-16 | 四阶段总体架构 |
| `figures/03-worldstereo-pipeline.png` | arXiv 源码原图 | https://arxiv.org/e-print/2604.14268 | Figure 7，PDF p.10 | 2026-10-16 | WorldStereo 2.0 双记忆与相机控制 |
| `figures/04-worldmirror-architecture.png` | arXiv 源码原图 | https://arxiv.org/e-print/2604.14268 | Figure 12，PDF p.15 | 2026-10-16 | WorldMirror 2.0 多模态重建架构 |
| `figures/05-trajectory-ablation.png` | arXiv 源码原图 | https://arxiv.org/e-print/2604.14268 | Figure 16，PDF p.27 | 2026-10-16 | WorldNav 轨迹规划消融 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2604.14268
- 项目页：https://3d-models.hunyuan.tencent.com/world/
- 官方代码：https://github.com/Tencent-Hunyuan/HY-World-2.0
- 官方模型目录：https://huggingface.co/tencent/HY-World-2.0

## 核验说明

- PDF 为 arXiv v1，提交于 2026-04-15。
- 五张图片均直接从 arXiv 官方 LaTeX 源码包的完整 PDF 图转换为 PNG，并于 2026-10-16 使用 Read 逐张检查；未发现裁边、缺坐标/图例或零散子图问题。
- 论文称“release all model weights, code, and technical details”。当前仓库可明确核验 WorldMirror 2.0、HY-Pano 2.0、WorldStereo 2.0 与完整 world-generation 流程的主要入口；但 WorldNav 依赖的 VLM/独立权重、HY-Pano-2-Qwen 的基础模型、mesh 导出操作链及若干第三方权重仍需按仓库文档逐项确认，不能把论文表述理解为无外部依赖的单仓开箱复现。
