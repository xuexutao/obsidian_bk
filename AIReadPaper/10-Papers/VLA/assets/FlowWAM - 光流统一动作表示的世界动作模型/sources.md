# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2607.13017 | 全文，v1，27 页（含附录 A–F） | 2026-10-25 | 精读主来源、公式、表格与附录 |
| `figures/01-teaser.png` | 动机总览图 | https://arxiv.org/pdf/2607.13017 | Figure 1, p.2 | 2026-10-25 | 研究背景：三类动作表示对比与双模式 |
| `figures/02-method-pipeline.png` | 方法架构图 | https://arxiv.org/pdf/2607.13017 | Figure 2, p.4 | 2026-10-25 | 核心方法：双流 DiT、Action Expert、两阶段训练与两种推理模式 |
| `figures/03-real-world-success.png` | 真机结果图 | https://arxiv.org/pdf/2607.13017 | Figure 3, p.8 | 2026-10-25 | 实验：Franka / ARX 七任务成功率 |
| `figures/04-ablation-flow-decodability.png` | 消融与可解码性图 | https://arxiv.org/pdf/2607.13017 | Figure 4, p.9 | 2026-10-25 | 实验：策略/世界模型消融、flow 误差与成功率相关性 |
| `figures/05-qualitative-two-modes.png` | 定性结果图 | https://arxiv.org/pdf/2607.13017 | Figure 5, p.9 | 2026-10-25 | 实验：策略模式与世界模型模式可视化 |

## 其它官方来源

| 材料 | URL | 获取日期 | 用途 |
|---|---|---|---|
| 项目页 | https://flow-wam.github.io/ | 2026-10-25 | 结果汇总、引用信息 |
| RoboTwin 策略代码 | https://github.com/YixiangChen515/FlowWAM | 2026-10-25 | 训练/推理流程、robot-only 渲染、`FLOW_MODE=full_scene` 选项 |
| WorldArena 世界模型代码 | https://github.com/YixiangChen515/FlowWAM_WorldArena | 2026-10-25 | robot-only renderer、SeedVR2-3B refiner（论文未提及）、训练数据布局 |
| 模型权重与模型卡 | https://huggingface.co/YixiangChen/FlowWAM | 2026-10-25 | 解码配置（num_frames 33 / num_video_frames 9 / action_snr_shift 5.0 等） |
| RoboTwin 训练数据 | https://huggingface.co/datasets/YixiangChen/FlowWAM_RoboTwin | 2026-10-25 | 含 robot-only 视角的 RoboTwin 2.0 数据 |

## 说明

- 论文无单独 Supplementary Material，附录已包含在 arXiv PDF 中。
- 所有图片均从官方 PDF 按图注坐标以 300 DPI 完整裁剪，非内嵌碎片。
