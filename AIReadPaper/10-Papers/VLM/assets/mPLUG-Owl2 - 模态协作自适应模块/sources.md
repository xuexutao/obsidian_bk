# Sources

## 官方入口

- arXiv 页面：https://arxiv.org/abs/2311.04257
- 官方 PDF：https://arxiv.org/pdf/2311.04257
- arXiv TeX source：https://arxiv.org/e-print/2311.04257
- 官方代码、模型与 Demo：https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2
- arXiv DOI：https://doi.org/10.48550/arXiv.2311.04257
- 获取日期：2026-10-16

## 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | arXiv PDF | 全文 | 方法、训练、结果与公式复核 |
| `figures/01-overview.png` | 完整原图，300 DPI 渲染 | arXiv TeX `figures/fig_merge.pdf` | Figure 1 | 总体结果与模态协作动机 |
| `figures/02-architecture-mam.png` | 完整原图，300 DPI 渲染 | arXiv TeX `figures/fig2.pdf` | Figure 2 | 架构、MAM、冻结策略 |
| `figures/03-attention-map.png` | 完整原图，300 DPI 渲染 | arXiv TeX `figures/visu_map_draw2.pdf` | Figure 4 | 分层 attention 可视化 |
| `figures/04-unrelated-modality.png` | 完整原图，300 DPI 渲染 | arXiv TeX `figures/anti-fact.pdf` | Figure 5 | 无关图像干扰案例 |

四张图均来自官方 arXiv source，未使用 PDF 内嵌碎片；已逐张本地读取，确认边界、文字与子图完整。

## 复核备注

- 论文正文称 Figure 5 中 MAM 版本正确列出彩虹七色，但原图输出含 `white` 且缺 `red`。精读笔记保留该文图矛盾。
- MAM 公式正文未显式写 causal mask；笔记将其标记为 decoder-only 生成所需约束，而不把未披露的多图 mask 细节写成论文事实。
