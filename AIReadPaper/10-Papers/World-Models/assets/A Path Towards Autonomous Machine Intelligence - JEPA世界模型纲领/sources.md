# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF（Version 0.9.2） | [OpenReview 官方页面](https://openreview.net/forum?id=BZ5a1r-kVsf)；[官方 PDF 端点](https://openreview.net/pdf?id=BZ5a1r-kVsf) | 全文，62 页 | 2026-10-15 | 精读主来源 |
| `sources/paper.txt` | PDF 文本抽取 | 由本地 `sources/paper.pdf` 使用 PyMuPDF 抽取 | 全文 | 2026-10-15 | 全文检索与逐段核对 |
| `figures/01-system-architecture.png` | 总体系统架构图 | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | Figure 2, p.6 | 2026-10-15 | 解释自主智能体模块与信息流 |
| `figures/02-ssl-energy-model.png` | 自监督学习与能量模型图 | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | Figure 8, p.18 | 2026-10-15 | 解释兼容性、能量函数与多模态预测 |
| `figures/03-hierarchical-jepa.png` | H-JEPA 层级架构图 | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | Figure 15, p.30 | 2026-10-15 | 解释多层抽象与多时间尺度预测 |
| `figures/04-hierarchical-planning.png` | 层级规划图 | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | Figure 16, p.32 | 2026-10-15 | 解释高层动作、子目标与低层控制 |

## 获取说明

OpenReview 页面明确给出标题、作者、发布日期和官方 PDF 端点。本次直接访问官方 PDF 端点时服务器返回 HTTP 403，因此 `sources/paper.pdf` 从 Internet Archive 条目 `a-path-towards-autonomous-machine-intelligence` 的 **original** 文件镜像下载；该条目记录的原始来源即上述 OpenReview 页面。下载文件 SHA-1 为 `775f42ed458b8c5b0f2094ea4ff5b64c557b1a34`，与 Internet Archive 元数据中 original PDF 的 SHA-1 一致。PDF 首页标明 `Version 0.9.2, 2022-06-27`，共 62 页。

四张图片均直接从该 PDF 指定 Figure 所在页以 300 DPI 裁剪，保留完整图形与原始图注，并已逐张目视核验。本文是 position paper，没有配套实验结果图；因此按论文自身重点选取总体架构、能量模型、H-JEPA 层级预测与层级规划图，而不以无关图片凑数。

- OpenReview 页面：https://openreview.net/forum?id=BZ5a1r-kVsf
- OpenReview 官方 PDF：https://openreview.net/pdf?id=BZ5a1r-kVsf
- Internet Archive original 文件条目：https://archive.org/details/a-path-towards-autonomous-machine-intelligence
