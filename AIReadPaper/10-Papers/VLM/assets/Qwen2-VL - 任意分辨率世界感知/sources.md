# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2409.12191 | 全文，arXiv v2 | 2026-10-15 | 精读主来源、方法与实验数字核验 |
| `figures/01-any-resolution.jpg` | 官方源码原图 | https://arxiv.org/e-print/2409.12191 | Figure 2，p.4；源码 `images/qwen2_vl.png` | 2026-10-15 | 动态分辨率、变长视觉 token 与统一图像视频架构 |
| `figures/02-mrope.png` | 官方源码原图 | https://arxiv.org/e-print/2409.12191 | Figure 3，p.5；源码 `images/mrope.png` | 2026-10-15 | M-RoPE 的时间/高度/宽度位置分解 |
| `figures/03-dynamic-resolution-ablation.png` | 官方源码矢量图转 PNG | https://arxiv.org/e-print/2409.12191 | Figure 4，p.14；源码 `images/peng_fig.pdf` | 2026-10-15 | `min_pixels` 与感知、OCR、推理性能消融 |
| `figures/04-length-extrapolation.png` | 官方源码矢量图转 PNG | https://arxiv.org/e-print/2409.12191 | Figure 5，p.15；源码 `images/peng_fig2.pdf` | 2026-10-15 | M-RoPE 在 16K 训练长度之外的推理外推 |

## 官方链接

- 论文页：https://arxiv.org/abs/2409.12191
- PDF：https://arxiv.org/pdf/2409.12191
- arXiv LaTeX 源码：https://arxiv.org/e-print/2409.12191
- 官方代码：https://github.com/QwenLM/Qwen2-VL

## 取材说明

正文中的架构、训练 token、benchmark、消融、局限和 future work 均以本地 `sources/paper.pdf` 为准。四张图片来自 arXiv 官方源码包中的完整 Figure 原图，未使用博客或二手解读；均已逐图预览，确认标题、坐标轴、图例与边缘完整。论文未提供独立 supplementary PDF，附录已包含在主 PDF 中。
