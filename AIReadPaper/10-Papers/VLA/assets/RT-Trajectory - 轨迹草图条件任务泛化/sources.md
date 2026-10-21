# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2311.01977> | 全文，23 页 | 2026-10-21 | 精读主来源 |
| `figures/01-overview.png` | 官方源码图 | <https://arxiv.org/abs/2311.01977> | Figure 1，p.2；源码 `figures/Architecture_v3.png` | 2026-10-21 | 训练标签、推理草图来源与 RT-1 骨干 |
| `figures/02-trajectory-representation.png` | 官方源码图 | <https://arxiv.org/abs/2311.01977> | Figure 3，p.4；源码 `figures/Trajectory_Representation_Visualization.png` | 2026-10-21 | 2D/2.5D、时间与抓放标记 |
| `figures/03-unseen-task-results.png` | 官方源码图 | <https://arxiv.org/abs/2311.01977> | Figure 5，p.7；源码 `figures/quantitative-unseen-tasks-reordered.pdf` | 2026-10-21 | 七个未见任务定量结果 |
| `figures/04-emergent-capabilities.png` | 官方源码图 | <https://arxiv.org/abs/2311.01977> | Figure 8，p.8；源码 `figures/emergent_capabilities_v2.pdf` | 2026-10-21 | 新运动、新家具与新场景 rollout |

## 提取与核验记录

- 使用 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv 官方源码包提取，最大宽度 2400 px。
- 四张图已逐张预览；轨迹颜色、抓放标记、图例、坐标和所有 rollout 行完整，无边缘裁切。
- 结果表数字同时与 Appendix Table 4 核对，避免仅凭柱状图估读。
