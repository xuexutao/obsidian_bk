# Sources

> 论文版本：arXiv:2603.14482v3，2026-06-11。arXiv 页面截至获取日未标注正式会议/期刊发表信息，本文按预印本记录。

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v3） | <https://arxiv.org/pdf/2603.14482v3> | 全文，37 页 | 2026-10-15 | 精读主来源 |
| `sources/source.tar.gz` | arXiv 官方 TeX 源码包（v3） | <https://arxiv.org/e-print/2603.14482v3> | 全文源码与原始图 | 2026-10-15 | 核对公式、表格并提取原图 |
| `figures/01-dense-feature-teaser.png` | Figure 1 原图 | arXiv v3 源码：`sections/teaser_screenshot_5dice_onlyimg_OLD.png` | Figure 1，PDF p.1 | 2026-10-15 | 展示 V-JEPA 2 与 2.1 的稠密特征差异 |
| `figures/02-method-pipeline.jpg` | Figure 4 原图 | arXiv v3 源码：`diagrams/architecture_vjepa2_1.jpg` | Figure 4，PDF p.5 | 2026-10-15 | 完整方法 pipeline |
| `figures/03-component-ablation.png` | Figure 5 矢量原图转 300 DPI PNG | arXiv v3 源码：`diagrams/v2_to_25.pdf` | Figure 5，PDF p.6 | 2026-10-15 | 训练组件累计消融 |
| `figures/04-robot-grasping.png` | Figure 8 矢量原图转 300 DPI PNG | arXiv v3 源码：`assets/robot_execution.pdf` | Figure 8，PDF p.13 | 2026-10-15 | 机器人抓取与能量地形 |
| — | arXiv 摘要与版本历史 | <https://arxiv.org/abs/2603.14482> | v1/v2/v3 提交历史 | 2026-10-15 | 核对标题、作者、版本日期和预印本状态 |

## 完整性核验

- 4 张图片均从 arXiv v3 官方源码包中的完整 Figure 源文件提取，而非 PDF 内嵌碎片。
- `01-dense-feature-teaser.png`、`02-method-pipeline.jpg`、`03-component-ablation.png`、`04-robot-grasping.png` 已分别使用 Read 预览，图像边界、图例、坐标轴和文字完整清晰。
- 未发现独立公开的 Supplementary PDF；论文附录已包含在 37 页官方 PDF 中。
