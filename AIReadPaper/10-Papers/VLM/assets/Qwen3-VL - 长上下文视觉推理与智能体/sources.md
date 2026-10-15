# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2511.21631 | 全文，arXiv v2 | 2026-10-15 | 精读主来源、架构/训练/RL/实验/消融核验 |
| `figures/01-benchmark-overview.png` | 官方源码原图 | https://arxiv.org/e-print/2511.21631 | 首页概览图，p.1；源码 `figures/qwen3vl_head.png` | 2026-10-15 | Instruct/Thinking 的视觉与文本总体能力概览 |
| `figures/02-architecture.jpg` | 官方源码原图 | https://arxiv.org/e-print/2511.21631 | Figure 1，p.3；源码 `figures/qwen3vl_arc.jpg` | 2026-10-15 | Dense/MoE、Interleaved-MRoPE、DeepStack、文本时间戳 |
| `figures/03-multilingual-ocr.png` | 官方源码原图 | https://arxiv.org/e-print/2511.21631 | Figure 2，p.17；源码 `figures/multi_lan_ocr_support.png` | 2026-10-15 | 39 种语言 OCR 覆盖与内部评测 |
| `figures/04-needle-haystack.png` | 官方源码矢量图转 PNG | https://arxiv.org/e-print/2511.21631 | Figure 3，p.25；源码 `figures/heatmap_comparison_qwen-vl_needle_in_a_haystack_performance.pdf` | 2026-10-15 | 256K 原生上下文与 1M YaRN 外推的长视频检索 |

## 官方链接

- 论文页：https://arxiv.org/abs/2511.21631
- PDF：https://arxiv.org/pdf/2511.21631
- arXiv LaTeX 源码：https://arxiv.org/e-print/2511.21631
- 官方代码：https://github.com/QwenLM/Qwen3-VL

## 取材说明

正文的 Interleaved-MRoPE、DeepStack、显式时间戳、256K 课程、Dense/MoE、Instruct/Thinking、强到弱蒸馏、SAPO RL、定量结果、消融、局限与 future work 均由本地官方 PDF 核验。四张图片均取自 arXiv 官方源码包并逐图预览；Figure 3 的矢量 PDF 转为 PNG，首页概览原图含透明通道，为避免 Obsidian 深浅色背景导致图例不可读，仅将透明背景无损合成为白底，未改动图表内容。论文没有独立 supplementary PDF，附录包含在主 PDF 中。
