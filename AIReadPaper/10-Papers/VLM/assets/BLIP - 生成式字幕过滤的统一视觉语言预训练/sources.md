# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2201.12086 | 全文，arXiv v2，12 页 | 2026-10-16 | 精读主来源、表格与附录核对 |
| `figures/04-teaser.png` | arXiv TeX 源码原图转 PNG | https://arxiv.org/e-print/2201.12086 | `imgs/teaser.pdf`；Figure 1，PDF 第 1 页 | 2026-10-16 | 说明 Captioner/Filter 的核心动机 |
| `figures/01-med-architecture.png` | arXiv TeX 源码原图转 PNG | https://arxiv.org/e-print/2201.12086 | `imgs/MED.pdf`；Figure 2，PDF 第 2 页 | 2026-10-16 | MED 三种模式及 ITC/ITM/LM 架构 |
| `figures/02-capfilt-pipeline.png` | arXiv TeX 源码原图转 PNG | https://arxiv.org/e-print/2201.12086 | `imgs/BootCap.pdf`；Figure 3，PDF 第 4 页 | 2026-10-16 | CapFilt 数据自举流水线 |
| `figures/03-caption-filter-examples.png` | arXiv TeX 源码原图转 PNG | https://arxiv.org/e-print/2201.12086 | `imgs/examples.pdf`；Figure 4，PDF 第 5 页 | 2026-10-16 | 过滤效果、误拒与失败边界分析 |

## 官方在线来源

| 来源 | URL | 核对内容 |
|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2201.12086 | 标题、作者、版本日期、摘要、PDF 与代码入口 |
| arXiv TeX 源码包 | https://arxiv.org/e-print/2201.12086 | 方法、实验、附录原文及完整矢量图 |
| 官方代码仓库 | https://github.com/salesforce/BLIP | PyTorch 实现、配置、预训练/微调脚本、权重、CapFilt JSON 数据 |
| 后续工作 BLIP-2 | https://arxiv.org/abs/2301.12597 | 核对从 MED/CapFilt 到冻结底座与 Q-Former 的技术演化 |

## 图像验证记录

四张图片均直接来自 arXiv TeX 源码包中的 `imgs/*.pdf`，使用 `research-field-survey/scripts/extract_source_figures.py` 的 PDF 转 PNG 逻辑转换，并于 2026-10-16 逐张使用 `Read` 检查。Figure 1、2、3、4 均保留完整边界、文字、图例和子图，没有使用从论文 PDF 抽出的零散内嵌位图。