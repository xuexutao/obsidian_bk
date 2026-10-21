# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v2） | https://arxiv.org/pdf/2507.21809 | 全文，22 页 | 2026-10-16 | 精读主来源 |
| `figures/01-teaser.png` | arXiv 源码原图 | https://arxiv.org/e-print/2507.21809 | 封面 teaser，PDF p.1 | 2026-10-16 | 展示生成世界的视觉范围 |
| `figures/02-method-pipeline.png` | arXiv 源码原图 | https://arxiv.org/e-print/2507.21809 | Figure 2，PDF p.5 | 2026-10-16 | 总体方法：全景代理、分层、重建 |
| `figures/03-data-pipeline.png` | arXiv 源码原图 | https://arxiv.org/e-print/2507.21809 | Figure 3，PDF pp.5–6 | 2026-10-16 | 全景数据清洗与人工复核 |
| `figures/04-image-to-world-comparison.png` | arXiv 源码原图 | https://arxiv.org/e-print/2507.21809 | Figure 12，PDF p.15 | 2026-10-16 | 图生世界定性对比 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2507.21809
- 项目页：https://3d-models.hunyuan.tencent.com/world/
- 官方代码：https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
- 官方模型：https://huggingface.co/tencent/HunyuanWorld-1

## 核验说明

- PDF 为 arXiv v2，提交于 2025-07-29，修订于 2025-08-13。
- 四张图片均直接从 arXiv 官方 LaTeX 源码包中的完整 PDF 图转换为 PNG，并于 2026-10-16 使用 Read 逐张检查；未发现裁边、缺图例或拼接碎片。
- 当前官方仓库可确认已有推理代码与 PanoDiT / PanoInpaint 权重；README 未披露最低显存、完整训练代码、总训练成本和严格端到端耗时。Lite / Voyager 的发布状态不能只凭“Open-Source Plan”标题推断。
