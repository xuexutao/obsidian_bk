# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | Read 验证 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2312.17172 | arXiv:2312.17172v1，38 页 | 2026-10-16 | 已核对标题、页数与正文 |
| `source.md` | PDF 文本抽取 | 同上 | 全文与附录 | 2026-10-16 | 已分段读取方法、训练、实验与附录 |
| `figures/01-capability-overview.png` | 300 DPI 完整图 | 官方 PDF | Figure 1，PDF p.1 | 2026-10-16 | 已预览；图框、任务示例与图注完整 |
| `figures/02-architecture.png` | 300 DPI 完整图 | 官方 PDF | Figure 2，PDF p.4 | 2026-10-16 | 已预览；输入编码器、共享主干、三类输出与图注完整 |
| `figures/03-dynamic-masking.png` | 300 DPI 完整图 | 官方 PDF | Figure 5，PDF p.5 | 2026-10-16 | 已预览；三种 mask 范式、箭头、图例与图注完整 |
| `meta.json` | 抓取元信息 | arXiv 摘要页 | 标题、作者、摘要、页图索引 | 2026-10-16 | 已读取 |

## 官方入口

- 摘要页：https://arxiv.org/abs/2312.17172
- PDF：https://arxiv.org/pdf/2312.17172

> 完整性说明：三张图均从官方 PDF 对应页面按图框裁剪，不使用 PDF 内嵌碎片；已逐张通过 Read 视觉复核。原始整页渲染与嵌入图片保留在 `images/`，未清理个人 Vault 中的支持材料。
