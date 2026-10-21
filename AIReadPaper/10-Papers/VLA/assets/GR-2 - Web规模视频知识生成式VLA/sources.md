# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2410.06158> | 全文，21 页 | 2026-10-21 | 精读主来源 |
| `figures/01-overview.png` | 官方源码原图 | <https://arxiv.org/e-print/2410.06158> | Figure 1，p.2，`figure/overview.pdf` | 2026-10-21 | 视频语言预训练到 VLA 微调总览 |
| `figures/02-pretraining-data.png` | 官方源码原图 | <https://arxiv.org/e-print/2410.06158> | Figure 2，p.3，`figure/pt_dataset-min.pdf` | 2026-10-21 | 38M 视频数据组成 |
| `figures/03-multitask-results.png` | 官方源码原图 | <https://arxiv.org/e-print/2410.06158> | Figure 6，p.7，`figure/multi_task_success_rate.png` | 2026-10-21 | 多任务与泛化定量对比 |
| `figures/04-scaling.png` | 官方源码原图 | <https://arxiv.org/e-print/2410.06158> | Figure 11，p.10，`figure/scale.pdf` | 2026-10-21 | 模型规模消融 |
| 项目页 | 官方项目页 | <https://gr2-manipulation.github.io> | 论文首页 | 2026-10-21 | 视频与补充展示 |

## 提取与核验说明

- PDF 矢量图使用 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv 源码包提取并转为限宽 2400 px 的 PNG；Figure 6 原始即 PNG，直接复制官方源码文件。
- 四张图均已逐张目视检查，标题、坐标、图例和边缘完整；正文全部引用。
- 论文文本由本地官方 PDF 逐页解析；显式数字以正文为准，只有柱状图而无精确表值的项目不伪造小数。
