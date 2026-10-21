# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2312.01990> | 全文，8 页 | 2026-10-21 | 精读主来源 |
| `figures/01-overview.png` | 官方源码图 | <https://arxiv.org/abs/2312.01990> | Figure 1，p.1；源码 `img/intro-vis-compressed.pdf` | 2026-10-21 | RT-2 与 PCT 两类机器人策略概览 |
| `figures/02-attention-navigation.png` | 官方源码图 | <https://arxiv.org/abs/2312.01990> | Figure 2，p.2；源码 `img/clip10-compressed.pdf` | 2026-10-21 | softmax、简单线性、随机映射和 SARA 对比 |
| `figures/03-pct-uptraining.png` | 官方源码图 | <https://arxiv.org/abs/2312.01990> | Figure 4，p.4；源码 `img/fig-uptraining-final.pdf` | 2026-10-21 | PCT up-training 收敛 |
| `figures/04-sara-rt2-architecture.png` | 官方源码图 | <https://arxiv.org/abs/2312.01990> | Figure 7，p.6；源码 `img/PaLI-3.pdf` | 2026-10-21 | SARA-ViT、多帧与 fuser 结构 |

## 提取与核验记录

- 使用 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv 官方源码提取，最大宽度 2400 px。
- 四张图均逐张预览；架构标签、导航 patch、训练曲线和边界完整。
- Table I 的成功率与正文延迟数字直接从 PDF 文本核对，图像仅作辅助说明。
