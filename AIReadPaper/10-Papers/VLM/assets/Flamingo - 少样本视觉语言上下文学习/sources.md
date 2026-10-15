# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2204.14198 | 全文 | 2026-10-15 | 精读主来源 |
| `figures/01-results-overview.png` | 主要结果原图 | https://arxiv.org/abs/2204.14198 | Figure 2, p.3 | 2026-10-15 | 解释参数/shot 扩展与相对 SOTA |
| `figures/02-architecture.png` | 总体架构原图 | https://arxiv.org/abs/2204.14198 | Figure 3, p.4 | 2026-10-15 | 解释冻结底座与门控跨注意力 |
| `figures/03-perceiver-resampler.png` | 模块结构与伪代码原图 | https://arxiv.org/abs/2204.14198 | Figure 5, p.23 | 2026-10-15 | 解释可变视觉网格到固定 token |
| `figures/04-failure-cases.png` | 失败案例原图 | https://arxiv.org/abs/2204.14198 | Appendix failure-cases figure, p.42 | 2026-10-15 | 分析幻觉与无依据猜测 |

## 官方在线来源

- arXiv 摘要与版本记录：https://arxiv.org/abs/2204.14198
- 获取日期：2026-10-15
- 官方训练代码、80B 权重与核心训练数据：论文未公开；未以第三方复现替代官方来源。

## 提取说明

四张图片均由 arXiv 官方 LaTeX 源码包中的完整矢量 PDF 原图转换为 PNG。已逐张预览确认图例、文字、边界和多子图完整清晰；未使用零散内嵌图片。