# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | Read 验证 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2408.11039 | arXiv:2408.11039v1，23 页 | 2026-10-16 | 已核对标题、页数与正文 |
| `source.md` | PDF 文本抽取 | 同上 | 全文与附录 | 2026-10-16 | 已分段读取 VAE、双损失、attention、采样、数据与消融 |
| `figures/01-unified-transformer.png` | 300 DPI 完整图 | 官方 PDF | Figure 1，PDF p.2 | 2026-10-16 | 已预览；混合序列、BOI/EOI、输出与图注完整 |
| `figures/02-vae-patch-architecture.png` | 300 DPI 完整图 | 官方 PDF | Figure 3，PDF p.5 | 2026-10-16 | 已预览；VAE、加噪、线性/U-Net 路径与图注完整 |
| `figures/03-intra-image-attention.png` | 300 DPI 完整图 | 官方 PDF | Figure 4，PDF p.5 | 2026-10-16 | 已预览；块状 attention mask 与图注完整 |
| `figures/04-scaling-comparison.png` | 300 DPI 完整图 | 官方 PDF | Figure 5，PDF p.9 | 2026-10-16 | 已预览；六项 scaling 曲线、坐标、图例与图注完整 |
| `meta.json` | 抓取元信息 | arXiv 摘要页 | 标题、作者、摘要、页图索引 | 2026-10-16 | 已读取 |

## 官方入口

- 摘要页：https://arxiv.org/abs/2408.11039
- PDF：https://arxiv.org/pdf/2408.11039

> 完整性说明：四张图均从官方 PDF 对应页面按图框裁剪，不使用内嵌图片碎片；已逐张通过 Read 视觉复核。原始整页渲染与嵌入图片保留在 `images/`。
