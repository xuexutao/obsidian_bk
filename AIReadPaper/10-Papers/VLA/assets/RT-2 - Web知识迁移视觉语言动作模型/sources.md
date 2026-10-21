# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2307.15818> | 全文，26 页 | 2026-10-21 | 精读主来源；核对方法、公式、实验、附录与局限 |
| `figures/01-overview.png` | 官方源码图 | <https://arxiv.org/abs/2307.15818> | Figure 1，p.2；源码 `figures/rt2_teaser.pdf` | 2026-10-21 | VLA 统一训练与部署流程 |
| `figures/02-generalization-results.png` | 官方源码图 | <https://arxiv.org/abs/2307.15818> | Figure 4，p.8；源码 `figures/rt2_overall_wide_dm.png` | 2026-10-21 | seen 与未见分布主结果 |
| `figures/03-emergent-capabilities.png` | 官方源码图 | <https://arxiv.org/abs/2307.15818> | Figure 6a，p.10；源码 `figures/rt2_emergent_dm.png` | 2026-10-21 | 符号、推理与人物识别结果 |
| `figures/04-size-training-ablation.png` | 官方源码图 | <https://arxiv.org/abs/2307.15818> | Figure 6b，p.10；源码 `figures/rt2_ablations_dm.png` | 2026-10-21 | 模型规模与训练策略消融 |

## 提取与核验记录

- 图片使用 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv 官方源码包提取，统一限制最大宽度 2400 px。
- 已逐张预览核对：四张图均完整包含坐标、图例、标签和边界，无裁切、拉伸或拼接缺失。
- 论文事实以本地 `sources/paper.pdf` 为准；项目页只用于补充视频入口，不以二手文章替代原文。
