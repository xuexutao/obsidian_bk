# Sources

## 官方入口

- arXiv 页面：https://arxiv.org/abs/2408.04840
- 官方 PDF：https://arxiv.org/pdf/2408.04840
- arXiv TeX source：https://arxiv.org/e-print/2408.04840
- 官方代码仓库：https://github.com/X-PLUG/mPLUG-Owl
- arXiv DOI：https://doi.org/10.48550/arXiv.2408.04840
- 获取日期：2026-10-16

## 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | arXiv PDF | 全文 | 方法、训练、结果与消融复核 |
| `figures/01-capabilities.png` | 完整原图，300 DPI 渲染 | arXiv TeX `images/title.pdf` | Figure 1 | 效率及短中长多图能力 |
| `figures/02-architecture.png` | 完整原图，300 DPI 渲染 | arXiv TeX `images/owl3.pdf` | Figure 2 | 独立视觉序列与稀疏 HATB |
| `figures/03-hyper-attention.png` | 完整原图，300 DPI 渲染 | arXiv TeX `images/hatb.pdf` | Figure 3 | Hyper Attention、门控与 cross mask |
| `figures/04-distractor-resistance.png` | 官方完整位图 | arXiv TeX `images/ir.png` | Figure 4 | 1–400 图抗干扰曲线 |

四张图均来自官方 arXiv source，未使用 PDF 内嵌碎片；已逐张本地读取，确认边界、图例、坐标和子图完整。

## 版本与口径说明

- 正文称 Qwen2 的 HATB 层为 `[0, 9, 17, 25]`，消融表写为 `[1, 9, 17, 25]`；可能是 0/1-based 编号差异，笔记未擅自统一。
- 论文阶段表未把 vision encoder 列为 Stage 2/3 可训练模块，正文也将细粒度短板归因于未微调视觉 encoder，因此笔记按“视觉 encoder 冻结”解读。
