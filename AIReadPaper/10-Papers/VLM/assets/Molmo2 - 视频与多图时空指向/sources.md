# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（含附录） | https://arxiv.org/pdf/2601.10611 | 全文，arXiv v4 | 2026-10-16 | 精读主来源 |
| `sources/source.md` | PDF 文本抽取 | https://arxiv.org/abs/2601.10611 | 全文 | 2026-10-16 | 全文检索与数字复核 |
| `figures/01-data-and-capabilities.png` | 数据与能力总览 | https://arxiv.org/pdf/2601.10611 | Figure 1, p.2 | 2026-10-16 | 九套数据构建流程与输出能力 |
| `figures/02-model-architecture.png` | 模型架构图 | https://arxiv.org/pdf/2601.10611 | Figure 2, p.6 | 2026-10-16 | Encoder–connector–LLM 与 point 输出 |
| `figures/03-packing-attention-mask.png` | Attention mask 图 | https://arxiv.org/pdf/2601.10611 | Figure 3, p.6 | 2026-10-16 | 双向视觉注意力、message tree 与 packing |
| `figures/04-counting-pointing-results.png` | Grounding 主结果表 | https://arxiv.org/pdf/2601.10611 | Table 3, p.9 | 2026-10-16 | 视频计数与时空 pointing baseline 对比 |
| `figures/05-model-and-grounding-ablations.png` | 消融表 | https://arxiv.org/pdf/2601.10611 | Tables 8–9, p.13 | 2026-10-16 | 双向注意力、time token、pooling、point-then-count 与数据配比 |
| `figures/06-pointing-qualitative.png` | 定性结果 | https://arxiv.org/pdf/2601.10611 | Figure 35, p.57 | 2026-10-16 | 多实例 pointing、视频 QA 与视觉伪影定位 |
| `figures/07-failure-cases.png` | 失败案例 | https://arxiv.org/pdf/2601.10611 | Figure 36, p.58 | 2026-10-16 | pointing 假阳性与 tracking 漏检 |

## 公开实现与数据

- 代码仓库：https://github.com/allenai/molmo2
- arXiv 摘要页：https://arxiv.org/abs/2601.10611
- 论文首页列出的模型、数据与在线演示入口可从官方代码仓库和论文链接继续访问。

## 说明

- `sources/source.pdf` 是既有取材缓存；本次复用并复制为规范路径 `sources/paper.pdf`，同时从本地 PDF 提取 `sources/source.md` 便于复核。
- 正文引用的图片均从 `sources/paper.pdf` 对应页面以 300 DPI 完整裁剪，保留原始图号/表号与图注，且已逐图检查完整性和可读性。
