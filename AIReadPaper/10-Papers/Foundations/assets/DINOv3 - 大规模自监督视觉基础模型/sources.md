# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2508.10104 | 全文，arXiv:2508.10104 | 2026-10-15 | 精读主来源、公式与实验数据核对 |
| `figures/01-ssl-and-dense-features.png` | 论文关键图 | https://arxiv.org/pdf/2508.10104 | Figure 1, p.2 | 2026-10-15 | 说明 SSL 的定位与 DINOv3 稠密特征优势 |
| `figures/02-dense-feature-degradation.png` | 论文关键图 | https://arxiv.org/pdf/2508.10104 | Figure 6, p.11 | 2026-10-15 | 展示长训练导致 patch 局部一致性退化 |
| `figures/03-gram-anchoring-results.png` | 论文关键图 | https://arxiv.org/pdf/2508.10104 | Figure 8, p.13 | 2026-10-15 | 展示 Gram anchoring 对稠密与全局任务的恢复效果 |
| `figures/04-gram-anchoring-recovery.png` | 论文关键图 | https://arxiv.org/pdf/2508.10104 | Figure 10, p.14 | 2026-10-15 | 定性比较 Gram anchoring 前后的局部相似度图 |

## 来源说明

- 论文主页：https://arxiv.org/abs/2508.10104
- 所有本地图片均从官方 PDF 对应页面以完整图表区域裁剪，保留原始 Figure 编号和图注，不使用 PDF 内嵌位图碎片。
- 本笔记中的模型、训练、结果和复现成本数字均以本地 `sources/paper.pdf` 为准。
