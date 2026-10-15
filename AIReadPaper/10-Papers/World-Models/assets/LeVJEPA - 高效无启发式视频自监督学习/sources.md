# Sources

> 论文版本：arXiv:2608.27395v1，2026-08-27。arXiv 页面截至获取日未标注正式会议/期刊发表信息，本文按预印本记录。索引使用短名 LeVJEPA；官方标题为 *LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics*。

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v1） | <https://arxiv.org/pdf/2608.27395v1> | 全文，12 页 | 2026-10-15 | 精读主来源 |
| `sources/source.tar.gz` | arXiv 官方 TeX 源码包（v1） | <https://arxiv.org/e-print/2608.27395v1> | 全文源码与矢量原图 | 2026-10-15 | 核对公式、表格并提取原图 |
| `figures/01-method-pipeline.png` | Figure 1 矢量原图转 300 DPI PNG | arXiv v1 源码：`figures/LeVJEPA_main.pdf` | Figure 1，PDF p.1 | 2026-10-15 | 完整训练 pipeline 与 SIGReg 机制 |
| `figures/02-compute-scaling.png` | Figure 2 矢量原图转 300 DPI PNG | arXiv v1 源码：`figures/imagenet_scaling.pdf` | Figure 2，PDF p.2 | 2026-10-15 | 相同 epoch 下准确率—计算量比较 |
| `figures/03-dense-feature-pca.png` | Figure 3 矢量原图转 300 DPI PNG | arXiv v1 源码：`figures/pca_comparison.pdf` | Figure 3，PDF p.3 | 2026-10-15 | 无 patch 监督时的稠密特征涌现 |
| — | arXiv 摘要与版本历史 | <https://arxiv.org/abs/2608.27395> | v1 提交历史 | 2026-10-15 | 核对标题、作者、版本日期和预印本状态 |

## 完整性核验

- 3 张图片均直接取自 arXiv v1 官方源码中的单页矢量 Figure 文件，并以 300 DPI 渲染；不是 PDF 图片碎片。
- `01-method-pipeline.png`、`02-compute-scaling.png`、`03-dense-feature-pca.png` 已分别使用 Read 预览，图像边界、坐标轴、标签和内容完整清晰。
- 未发现独立公开的 Supplementary PDF；SIGReg 与训练实现附录已包含在 12 页官方 PDF 中。
