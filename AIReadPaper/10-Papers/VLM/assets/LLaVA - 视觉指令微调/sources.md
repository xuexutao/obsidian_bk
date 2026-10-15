# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2304.08485 | 全文 | 2026-10-15 | 精读主来源 |
| `figures/01-data-generation.png` | 指令数据生成示例（PDF 完整裁剪） | https://arxiv.org/abs/2304.08485 | Table 1, p.3 | 2026-10-15 | 解释字幕/框到三类指令数据 |
| `figures/02-architecture.png` | 网络架构原图 | https://arxiv.org/abs/2304.08485 | Figure 1, p.4 | 2026-10-15 | 解释 CLIP、线性投影与 Vicuna |
| `figures/03-ablation-results.png` | 结果与数据消融（PDF 完整裁剪） | https://arxiv.org/abs/2304.08485 | Table 4–5, p.7 | 2026-10-15 | 比较训练数据组合及基线 |
| `figures/04-qualitative-result.png` | 定性结果原图 | https://arxiv.org/abs/2304.08485 | Figure 2, p.16 | 2026-10-15 | 展示视觉草图到 HTML/JS 的开放任务 |

## 官方在线来源

- arXiv 摘要与版本记录：https://arxiv.org/abs/2304.08485
- LLaVA 官方项目页：https://llava-vl.github.io/
- LLaVA 官方代码：https://github.com/haotian-liu/LLaVA
- 获取日期：2026-10-15

## 提取说明

`02-architecture.png` 与 `04-qualitative-result.png` 来自 arXiv 官方 LaTeX 源码包中的完整矢量 PDF 原图；`01-data-generation.png` 与 `03-ablation-results.png` 由官方 PDF 按完整表格及图注精确裁剪。四张图均已逐张预览，确认无切边、文字可读、表格列与图注完整；未使用零散内嵌碎片。