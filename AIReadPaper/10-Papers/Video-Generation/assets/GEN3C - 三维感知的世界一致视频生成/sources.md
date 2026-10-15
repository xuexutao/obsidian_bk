# GEN3C 来源清单

获取日期统一为 **2026-10-15**。图片均直接取自 arXiv 官方源码包并完整渲染，未从 PDF 页面截图、未裁切边缘。

| 本地文件 | 官方 URL | 源码内文件 | Figure / Page | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | https://arxiv.org/pdf/2503.03751 | — | 全文，16 页 | 官方论文全文与数字、公式溯源 |
| `sources/arxiv-source.tar` | https://arxiv.org/e-print/2503.03751 | — | 官方 TeX 源码包 | 高清原图、表格与公式溯源 |
| `figures/fig1_teaser.png` | https://arxiv.org/e-print/2503.03751 | `fig/teaser/teaser_fig_1114_3.pdf` | Figure 1 / Page 1 | 总览输入形态、长视频、Dolly Zoom 与驾驶编辑能力 |
| `figures/fig3_method.png` | https://arxiv.org/e-print/2503.03751 | `fig/method_compatible.pdf` | Figure 3 / Page 3 | 时空 3D cache、渲染、掩码与扩散条件注入主流程 |
| `figures/fig5_single_view_results.png` | https://arxiv.org/e-print/2503.03751 | `fig/single_view_img.pdf` | Figure 5 / Page 5 | 单图新视角合成与 CameraCtrl、NVS-Solver 定性比较 |
| `figures/fig9_fusion_ablation.png` | https://arxiv.org/e-print/2503.03751 | `fig/buffer_vs_point_crop.pdf` | Figure 10 / Page 8 | 隐式特征池化对显式点云融合的消融比较 |

> 注：源码文件名 `buffer_vs_point_crop.pdf` 在论文定稿中对应 Figure 10；笔记沿用语义化本地名 `fig9_fusion_ablation.png`，不把文件名中的数字当作论文 Figure 编号。
