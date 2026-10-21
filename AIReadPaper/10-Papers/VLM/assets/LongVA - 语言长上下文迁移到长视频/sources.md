# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（含附录） | https://arxiv.org/pdf/2406.16852 | 全文，arXiv v2 | 2026-10-16 | 精读主来源 |
| `sources/source.md` | PDF 文本抽取 | https://arxiv.org/abs/2406.16852 | 全文 | 2026-10-16 | 全文检索与数字复核 |
| `figures/01-long-context-transfer.png` | 方法动机图 | https://arxiv.org/pdf/2406.16852 | Figure 1, p.4 | 2026-10-16 | 解释语言到视觉的长上下文迁移 |
| `figures/02-unires-encoding.png` | 编码架构图 | https://arxiv.org/pdf/2406.16852 | Figure 2, p.5 | 2026-10-16 | UniRes 图像/视频统一编码与 token shape |
| `figures/03-v-niah-design.png` | Benchmark 设计图 | https://arxiv.org/pdf/2406.16852 | Figure 3, p.6 | 2026-10-16 | 解释视觉“大海捞针”评测 |
| `figures/04-v-niah-results.png` | 长上下文结果图 | https://arxiv.org/pdf/2406.16852 | Figure 4, p.6 | 2026-10-16 | 2000–3000 帧检索与消融 |
| `figures/05-video-mme-results.png` | 主结果表 | https://arxiv.org/pdf/2406.16852 | Table 4, p.7 | 2026-10-16 | Video-MME baseline 与帧数消融 |

## 公开实现

- 代码仓库：https://github.com/EvolvingLMMs-Lab/LongVA
- arXiv 摘要页：https://arxiv.org/abs/2406.16852

## 说明

- `sources/source.pdf` 与 `sources/images/` 是既有取材缓存；本次复用原 PDF，并复制为规范路径 `sources/paper.pdf`。
- 正文引用的图片均从 `sources/paper.pdf` 对应页面以 300 DPI 完整裁剪，保留原始图号/表号与图注，且已逐图检查完整性和可读性。
