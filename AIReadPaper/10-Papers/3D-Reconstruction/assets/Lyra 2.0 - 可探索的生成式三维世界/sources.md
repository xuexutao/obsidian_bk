# Lyra 2.0 来源清单

获取日期统一为 **2026-10-15**。图片优先取自 arXiv 官方源码包；PDF 图按完整画布渲染，PNG 图保持原始分辨率，均已逐张检查无切边。

| 本地文件 | 官方 URL | 源码内文件 | Figure / Page | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | https://arxiv.org/pdf/2604.13036 | — | 全文，25 页 | 官方论文、附录、公式与实验数值溯源 |
| `sources/arxiv-source.tar` | https://arxiv.org/e-print/2604.13036 | — | 官方 TeX 源码包 | 高清原图、表格和实现细节溯源 |
| `figures/fig1_teaser.png` | https://arxiv.org/e-print/2604.13036 | `figures/teaser_v3.pdf` | Figure 1 / Page 1 | 长程探索、回望、约 90 米场景与仿真应用总览 |
| `figures/fig2_method.png` | https://arxiv.org/e-print/2604.13036 | `figures/method_v3.pdf` | Figure 2 / Page 6 | retrieve–generate–update、空间记忆与稠密对应注入 |
| `figures/fig5_video_results.png` | https://arxiv.org/e-print/2604.13036 | `figures/comparison_grid_all.png` | Figure 3 / Page 10 | 约第 800 帧的长视频定性比较 |
| `figures/fig7_ablation.png` | https://arxiv.org/e-print/2604.13036 | `figures/comparison_grid_ablations.png` | Figure 6 / Page 14 | 全局点云、显式对应融合、FramePack、自增强消融 |

> 注：本地文件名中的数字用于阅读排序；论文正式编号以 “Figure / Page” 列为准。
