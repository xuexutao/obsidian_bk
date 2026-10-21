# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2310.08864> | 全文，12 页 | 2026-10-21 | 精读主来源 |
| `figures/01-dataset-overview.png` | 官方源码图 | <https://arxiv.org/abs/2310.08864> | Figure 1，p.1；源码 `figures/data_vis_v3.jpg` | 2026-10-21 | OXE 数据总览 |
| `figures/02-dataset-analysis.png` | 官方源码图 | <https://arxiv.org/abs/2310.08864> | Figure 2，p.3；源码 `figures/rtx_data_analysis.pdf` | 2026-10-21 | 本体、场景、轨迹、技能和物体分布 |
| `figures/03-rtx-architectures.png` | 官方源码图 | <https://arxiv.org/abs/2310.08864> | Figure 3，p.4；源码 `figures/rt12_teaser_model_vertical.pdf` | 2026-10-21 | RT-1-X 与 RT-2-X 架构及动作接口 |
| `figures/04-cross-embodiment-results.png` | 官方源码图 | <https://arxiv.org/abs/2310.08864> | Figure 4，p.5；源码 `figures/rtx_barplot_10.png` | 2026-10-21 | 五个小数据域正迁移结果 |

## 提取与核验记录

- 使用 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv 2310.08864 官方源码提取，最大宽度 2400 px。
- 已逐张预览：四张图文字、图例、坐标和边界完整；未从 PDF 抽取零散子图。
- 数据仓库统计与模型实际 9 本体训练混合已在笔记中分开描述，避免把 22 本体数据总览误写成全部参与模型训练。
