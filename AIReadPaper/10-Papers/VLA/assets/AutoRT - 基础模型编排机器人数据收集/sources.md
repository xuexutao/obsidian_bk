# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2401.12963> | 全文，26 页 | 2026-10-21 | 精读主来源 |
| `figures/01-system-overview.png` | 官方源码图 | <https://arxiv.org/abs/2401.12963> | Figure 1，p.4；源码 `images/AutoRT_Graph_v5.pdf` | 2026-10-21 | 探索、任务生成、affordance 与采集闭环 |
| `figures/02-deployment-scale.png` | 官方源码图 | <https://arxiv.org/abs/2401.12963> | 部署统计；源码 `images/AutoRTHoursByPolicy.png` | 2026-10-21 | 分策略、按日采集时长 |
| `figures/03-visual-diversity.png` | 官方源码图 | <https://arxiv.org/abs/2401.12963> | Figure 5 组成图，p.8；源码 `images/clip_cdf_v3.png` | 2026-10-21 | 视觉多样性 CDF |
| `figures/04-safety-scenes.png` | 官方源码图 | <https://arxiv.org/abs/2401.12963> | Section 5.3 对抗安全场景；源码 `images/safetyscene.png` | 2026-10-21 | 宪法与过滤安全评测示例 |

## 提取与核验记录

- 使用 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv 官方源码提取，最大宽度 2400 px。
- 四张图片逐张预览：系统流程、曲线轴、CDF 标注和安全场景均完整清晰。
- `04-safety-scenes.png` 是论文官方源码中的单个对抗场景原图，不是从 PDF 截取的组合图；用途与来源已在正文和本清单明确。
