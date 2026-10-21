# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v4） | https://arxiv.org/pdf/2511.13719 | 全文，39 页 | 2026-10-16 | 精读主来源 |
| `figures/01-teaser.png` | 官方 LaTeX 原图转 PNG | https://arxiv.org/e-print/2511.13719 | Figure 1，`figures/images/teaser_camera_ready.pdf` | 2026-10-16 | 八项结果与 0–8M scaling |
| `figures/02-data-taxonomy.png` | 官方 LaTeX 原图转 PNG | https://arxiv.org/e-print/2511.13719 | Figure 2，`figures/images/data_vis_v3.pdf` | 2026-10-16 | 数据来源、taxonomy 与任务样例 |
| `figures/03-capability-transfer.png` | 官方 LaTeX 原图转 PNG | https://arxiv.org/e-print/2511.13719 | Figure 3，`figures/images/spill.pdf` | 2026-10-16 | 跨数据集能力迁移证据 |

## 外部链接

- arXiv 页面：https://arxiv.org/abs/2511.13719
- 官方代码：https://github.com/OpenSenseNova/SenseNova-SI
- 模型集合：https://huggingface.co/collections/sensenova/sensenova-si

## 完整性核验

- PDF 由 arXiv 官方 `pdf/2511.13719` 下载，版本对应 v4。
- 三张图片均来自官方 e-print 源码包，不是 PDF 内嵌碎片。
- 三张图片已用 Read 逐张预览：scaling 曲线、Sankey taxonomy、能力迁移样例均完整可读，无裁切。
- 附录包含在主 PDF 中；arXiv 页面未提供独立 supplementary PDF。
