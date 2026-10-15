# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2502.13923 | 全文，arXiv v1 | 2026-10-15 | 精读主来源、训练与实验数字核验 |
| `figures/01-benchmark-overview.jpg` | 官方源码原图 | https://arxiv.org/e-print/2502.13923 | 首页概览图，p.1；源码 `figures/head.jpg` | 2026-10-15 | 多模态与纯文本总体能力对照 |
| `figures/02-architecture.jpeg` | 官方源码原图 | https://arxiv.org/e-print/2502.13923 | Figure 1，p.3；源码 `figures/qwen2.5vl_arc.jpeg` | 2026-10-15 | 原生 ViT、window attention、动态 FPS 与绝对时间 MRoPE |
| `figures/03-training-stages.png` | 官方 PDF 完整裁图 | https://arxiv.org/pdf/2502.13923 | Table 2，p.8 | 2026-10-15 | 1.5T/2T/0.6T 三阶段预训练与上下文长度 |
| `figures/04-ocr-document-results.png` | 官方 PDF 完整裁图 | https://arxiv.org/pdf/2502.13923 | Table 5，p.13 | 2026-10-15 | 文档、OCR、图表与综合解析定量结果 |
| `figures/05-gui-agent-results.png` | 官方 PDF 完整裁图 | https://arxiv.org/pdf/2502.13923 | Table 9，p.15 | 2026-10-15 | GUI grounding、Android 与 OSWorld Agent 结果 |

## 官方链接

- 论文页：https://arxiv.org/abs/2502.13923
- PDF：https://arxiv.org/pdf/2502.13923
- arXiv LaTeX 源码：https://arxiv.org/e-print/2502.13923
- 官方代码：https://github.com/QwenLM/Qwen2.5-VL

## 取材说明

正文只使用官方技术报告核验架构、4.1T token 训练、约 2M SFT 数据、benchmark、局限和原文明确的 ongoing challenge。Figure 1 与首页概览取自 arXiv 官方源码；三张表格因源码未提供独立图文件，由官方 PDF 按完整表格范围裁取。五张图片均已逐图预览，确认表头、数值、图例和边缘完整。论文没有独立 supplementary PDF，也没有正式 ablation 或 future-work 章节；笔记对此如实标注，未以博客补齐或编造组件效果。
