# Long-WAM 材料溯源

| 材料 | 原始来源 | Figure / 页码 | 获取日期 | 用途 |
| --- | --- | --- | --- | --- |
| `figures/fig1_overview.png` | https://arxiv.org/pdf/2610.10528v1 | Figure 1 / Page 1 | 2026-10-24 | 总览：AR 视频预训练 → causal-to-causal 适配 → 上下文伸缩 → 高效部署 |
| `figures/fig2_attention.png` | https://arxiv.org/pdf/2610.10528v1 | Figure 2 / Page 3 | 2026-10-24 | 四种注意力模式与视频-动作推理策略（IDM / CoD） |
| `figures/fig6_context.png` | https://arxiv.org/pdf/2610.10528v1 | Figure 6 / Page 8 | 2026-10-24 | 三种视频初始化（Wan2.2 / LongLive-2.0 / LongLive2.0-Robot）的上下文伸缩曲线 |
| `figures/fig7_dynamic.png` | https://arxiv.org/pdf/2610.10528v1 | Figure 7 / Page 10 | 2026-10-24 | Unitree G1 传送带动态抓取与动态叠杯真机结果 |
| `sources/Long-WAM-arXiv2610.10528v1.pdf` | https://arxiv.org/pdf/2610.10528v1 | 全文 v1（28 页，2026-10-07 提交，cs.RO） | 2026-10-24 | 原始论文存档 |

- 裁剪方式：`research-field-survey/scripts/extract_figure.py`（pymupdf 按相对框渲染）；Fig 2 / Fig 6 为矢量绘制图，先用 `get_drawings()` 包围盒定位再裁剪，非 PDF 内嵌碎片。
- 边缘校验：渲染后检查四条边缘的非白像素占比，Fig 1/2/7 为 0，Fig 6 底部 0.039（图例末行贴边），均未截断主体内容。
- 项目页 / 代码：论文首页给出的 Code 与 Project Page 链接（NVIDIA，© 2026 NVIDIA）。
