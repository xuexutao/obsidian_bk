# Sources

## 官方入口

- arXiv 页面：https://arxiv.org/abs/2602.02276
- 官方 PDF（v2，2026-08-07）：https://arxiv.org/pdf/2602.02276
- arXiv TeX source：https://arxiv.org/e-print/2602.02276
- arXiv DOI：https://doi.org/10.48550/arXiv.2602.02276
- 获取日期：2026-10-15

## 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2602.02276 | v2 全文，31 页 | 2026-10-15 | 精读主来源、公式和数值复核 |
| `figures/01-main-results.png` | 原文总览图 | arXiv TeX source 中 `figures/k25-main-result.png` | Figure 1，p.2 | 2026-10-15 | 主要结果与 baseline 对比 |
| `figures/02-agent-swarm.png` | 原文方法图 | arXiv TeX source 中 `figures/multi-agent-rl-system.png` | Figure 3，Agent Swarm 架构 | 2026-10-15 | Orchestrator、冻结子智能体与并行任务分解 |
| `figures/03-vision-rl-curves.png` | 原文训练曲线 | arXiv TeX source 中 `figures/k25_visionzerorl_curves.pdf`，300 DPI 渲染 | Figure 2，Zero-Vision SFT 起点上的视觉 RL 曲线 | 2026-10-15 | 联合视觉 RL 与扩展趋势 |
| `figures/04-early-fusion-ablation.png` | 原文消融图 | arXiv TeX source 中 `figures/vision-joint.pdf`，300 DPI 渲染 | Figure 9（附录），早期视觉融合消融 | 2026-10-15 | 视觉注入时机与视觉/文本比例分析 |

所有图片均直接来自官方 arXiv source，并已逐张本地预览检查；未使用 PDF 内嵌碎片。TeX source 仅作为取材来源，未在最终附件目录重复保留。

## 公开支持材料

- 论文称已发布 post-trained Kimi K2.5 checkpoint；具体模型入口以 arXiv 页面及 Kimi/Moonshot 官方发布为准。论文正文未给出一个可稳定核验的独立代码仓库 URL，因此 frontmatter 中 `code` 保留为 `null`。
- Kimi K2 基座和 Kimi-VL 前作仅用于理解版本关系，本文结论及数字均以 K2.5 v2 官方 PDF 为主。

## 版本说明

本笔记读取 arXiv v2。v1 提交于 2026-02-02，v2 修订于 2026-08-07；正文中的模型、评测与 Agent Swarm 结论以 v2 为准。