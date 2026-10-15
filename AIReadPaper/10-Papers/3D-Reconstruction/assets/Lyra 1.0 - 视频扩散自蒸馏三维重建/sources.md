# Lyra 1.0 来源清单

获取日期统一为 **2026-10-15**。图片优先取自 arXiv 官方源码包；PDF 图按完整画布渲染，PNG 图保持原始分辨率，均已逐张检查无切边。

| 本地文件 | 官方 URL | 源码内文件 | Figure / Page | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | https://arxiv.org/pdf/2509.19296 | — | 全文，29 页 | 官方论文、附录与数值溯源 |
| `sources/arxiv-source.tar` | https://arxiv.org/e-print/2509.19296 | — | 官方 TeX 源码包 | 高清原图、表格、训练细节与公式溯源 |
| `figures/fig1_teaser.png` | https://arxiv.org/e-print/2509.19296 | `imgs/teaser.pdf` | Figure 1 / Page 1 | 单图到静态 3DGS、单视频到动态 4DGS 能力总览 |
| `figures/fig3_method.png` | https://arxiv.org/e-print/2509.19296 | `imgs/method.pdf` | Figure 4 / Page 5 | RGB 教师—3DGS 学生自蒸馏与解码器结构 |
| `figures/fig4_main_results.png` | https://arxiv.org/e-print/2509.19296 | `imgs/main_results.png` | Figure 6 / Page 8 | 多类场景的单图到 3DGS 新视角结果 |
| `figures/fig8_depth_ablation.png` | https://arxiv.org/e-print/2509.19296 | `imgs/ablation_depth.pdf` | Figure 11 / Page 26 | 深度损失抑制扁平几何的消融 |

> 注：本地文件名中的数字用于阅读排序；论文正式编号以 “Figure / Page” 列为准。
