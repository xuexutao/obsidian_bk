# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | Read 验证 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2405.09818 | arXiv:2405.09818v2，27 页 | 2026-10-16 | 已核对标题、页数与正文 |
| `source.md` | PDF 文本抽取 | 同上 | 全文与附录 | 2026-10-16 | 已分段读取 tokenizer、稳定训练、数据、推理与实验 |
| `figures/01-early-fusion-architecture.png` | 300 DPI 完整图 | 官方 PDF | Figure 1，PDF p.2 | 2026-10-16 | 已预览；预训练/生成两侧与图注完整 |
| `figures/02-stability-ablation.png` | 300 DPI 完整图 | 官方 PDF | Figure 5，PDF p.6 | 2026-10-16 | 已预览；三组曲线、坐标、图例与图注完整 |
| `figures/03-mixed-modal-human-eval.png` | 300 DPI 完整图 | 官方 PDF | Figure 9，PDF p.12 | 2026-10-16 | 已预览；绝对/相对人评图与图注完整 |
| `meta.json` | 抓取元信息 | arXiv 摘要页 | 标题、作者、摘要、页图索引 | 2026-10-16 | 已读取 |

## 官方入口

- 摘要页：https://arxiv.org/abs/2405.09818
- PDF：https://arxiv.org/pdf/2405.09818

> 完整性说明：三张图均从官方 PDF 对应页面按图框裁剪，不使用内嵌图片碎片；已逐张通过 Read 视觉复核。原始整页渲染与嵌入图片保留在 `images/`。
