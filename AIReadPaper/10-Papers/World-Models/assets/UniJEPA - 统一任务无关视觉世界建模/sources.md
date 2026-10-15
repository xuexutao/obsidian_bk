# Sources

> 本目录对应 **An Lanji 等人的视觉世界模型论文** *UniJEPA: A Unified Joint-Embedding Predictive Architecture for Task-Agnostic Visual World Modeling*（arXiv:2608.07409），不是同名的机器人策略工作。

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2608.07409 | 全文，v1 | 2026-10-15 | 精读主来源 |
| `figures/01-motivation.png` | 动机图 | https://arxiv.org/pdf/2608.07409 | Figure 1，PDF 第 2 页 | 2026-10-15 | 解释任务统一动机 |
| `figures/02-unified-architecture.png` | 方法架构图 | https://arxiv.org/pdf/2608.07409 | Figure 2，PDF 第 6 页 | 2026-10-15 | 解释共享 encoder、predictor 与 Gaussian regularizer |
| `figures/03-training-planning-pipeline.png` | 训练与规划流程图 | https://arxiv.org/pdf/2608.07409 | Figure 3，PDF 第 7 页 | 2026-10-15 | 解释预训练、动作条件后训练与 MPC |
| `figures/04-planning-by-task.png` | 规划结果图 | https://arxiv.org/pdf/2608.07409 | Figure 4，PDF 第 7 页 | 2026-10-15 | 展示三类任务的规划成功率 |
| `figures/05-regularizer-sensitivity.png` | 正则敏感性图 | https://arxiv.org/pdf/2608.07409 | Figure 7，PDF 第 12 页 | 2026-10-15 | 解释 Gaussian regularizer 权重的折中 |

## 官方页面核验

- arXiv 摘要页：https://arxiv.org/abs/2608.07409
- arXiv HTML（v1）：https://arxiv.org/html/2608.07409v1
- arXiv 提交记录：v1，2026-08-07。
- arXiv `Comments` 字段：`10 pages, 7 figures; Accepted by ICML2026`。本记录只转述官方页面，不以第三方网页作为会议状态依据。
- arXiv-issued DOI：`10.48550/arXiv.2608.07409`。
- 未在官方论文页或正文中找到项目主页、代码仓库、模型权重或独立 Supplementary Material 链接。
