# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2311.03079 | 全文 | 2026-10-15 | 精读主来源 |
| `figures/01-demo.png` | 多任务成功案例原图 | https://arxiv.org/abs/2311.03079 | Figure 2, p.2 | 2026-10-15 | 展示 OCR、推理、grounding 与坐标输出 |
| `figures/02-language-degradation.png` | 语言能力退化曲线原图 | https://arxiv.org/abs/2311.03079 | Figure 3, p.3 | 2026-10-15 | 解释冻结语言路径的动机 |
| `figures/03-architecture.png` | 视觉专家架构原图 | https://arxiv.org/abs/2311.03079 | Figure 4, p.3 | 2026-10-15 | 解释逐层视觉 QKV/FFN 与文本路径 |
| `figures/04-benchmark-overview.png` | 多基准雷达图原图 | https://arxiv.org/abs/2311.03079 | Figure 1, p.1 | 2026-10-15 | 展示任务覆盖与相对基线表现 |

## 官方在线来源

- arXiv 摘要与版本记录：https://arxiv.org/abs/2311.03079
- THUDM 官方代码与权重：https://github.com/THUDM/CogVLM
- 获取日期：2026-10-15

## 提取说明

`01-demo.png` 与 `04-benchmark-overview.png` 由 arXiv 官方 LaTeX 源码包中的完整矢量 PDF 原图转换；`02-language-degradation.png` 与 `03-architecture.png` 直接复制自官方源码包的完整 PNG。四张图均已逐张预览，确认图例、坐标、文字、多面板与边界完整清晰；未使用 PDF 内嵌碎片。