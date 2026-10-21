# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方技术报告 PDF | <https://arxiv.org/pdf/2507.15493> | 全文，20 页 | 2026-10-21 | 精读主来源 |
| `figures/01-teaser.jpeg` | 官方源码原图 | <https://arxiv.org/e-print/2507.15493> | Figure 1，p.2，`images/teaser.jpeg` | 2026-10-21 | 三类训练数据与三类能力总览 |
| `figures/02-model-architecture.jpg` | 官方源码原图 | <https://arxiv.org/e-print/2507.15493> | Figure 3，p.4，`images/model_arch.jpg` | 2026-10-21 | MoT、VLM 与 Action DiT 架构 |
| `figures/03-training-data.jpg` | 官方源码原图 | <https://arxiv.org/e-print/2507.15493> | Figure 4，p.6，`images/data.jpg` | 2026-10-21 | 机器人、人类、视觉语言数据 |
| `figures/04-pick-place-results.jpg` | 官方源码原图 | <https://arxiv.org/e-print/2507.15493> | Figure 7，p.9，`images/ppa_exp_result.jpg` | 2026-10-21 | OOD 与 few-shot 实验 |
| `figures/05-cloth-results.jpg` | 官方源码原图 | <https://arxiv.org/e-print/2507.15493> | Figure 10，p.13，`images/cloth_exp_result.jpg` | 2026-10-21 | 灵巧布料操控结果与失败阶段 |
| 项目页 | 官方项目页 | <https://seed.bytedance.com/GR3> | 论文首页 | 2026-10-21 | 视频与系统展示 |

## 提取与核验说明

- arXiv 源码包中的 GR-3 关键图原始即 JPEG/JPG，直接保留官方原图，避免二次有损编码；先用 `extract_source_figures.py --list` 建立候选清单，再据 `includegraphics` 与图注映射选图。
- 五张图均已逐张目视检查，标题、柱状图图例、阶段标记和边缘完整；正文全部引用。
- 技术报告未披露完整训练算力、VL 数据总量、机器人轨迹总规模和推理延迟，笔记中均明确标为“未披露”。
