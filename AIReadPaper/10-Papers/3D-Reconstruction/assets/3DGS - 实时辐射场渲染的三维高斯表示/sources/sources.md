# 3DGS 材料来源

- 论文：*3D Gaussian Splatting for Real-Time Radiance Field Rendering*
- arXiv：2308.04079
- 记录 / 获取日期：2026-10-15
- 正文笔记：`../../../3DGS - 实时辐射场渲染的三维高斯表示.md`

## 官方链接

| 类型 | URL | 本地文件 / 用途 | 获取日期 |
|---|---|---|---|
| arXiv 摘要 | https://arxiv.org/abs/2308.04079 | 元信息、摘要、作者、期刊信息 | 2026-10-15 |
| arXiv 官方 PDF | https://arxiv.org/pdf/2308.04079 | `paper.pdf` | 2026-10-15 |
| arXiv 官方源码 | https://arxiv.org/e-print/2308.04079 | `arxiv-source.tar` | 2026-10-15 |
| 官方项目页 | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ | 项目说明、数据、结果与代码入口 | 2026-10-15 |
| 官方代码 | https://github.com/graphdeco-inria/gaussian-splatting | 环境、训练、渲染、viewer 与复现说明 | 2026-10-15 |
| ACM DOI | https://doi.org/10.1145/3592433 | SIGGRAPH/TOG 正式出版记录 | 2026-10-15 |

## 图源与页码

以下图片均优先从 `arxiv-source.tar` 中的官方矢量 PDF 提取，以 300 DPI 转为 PNG；每张均于 2026-10-15 使用 Read 逐张检查，确认图像完整、清晰、无裁边。

| 本地图片 | 内容 | 论文 Figure | PDF Page | 官方源码文件 | 来源 URL | 验证日期 |
|---|---|---:|---:|---|---|---|
| `../figures/fig1_teaser.png` | Bicycle 上的速度、训练时间与质量对比 | Figure 1 | Page 1 | `figures/teaser/teaser_02.pdf` | https://arxiv.org/e-print/2308.04079 | 2026-10-15 |
| `../figures/fig2_overview.png` | SfM 初始化、3D Gaussian、投影、density control 与 tile rasterizer 总览 | Figure 2 | Page 5 | `figures/overview/overview_01.pdf` | https://arxiv.org/e-print/2308.04079 | 2026-10-15 |
| `../figures/fig3_density_control.png` | under-reconstruction 的 clone 与 over-reconstruction 的 split | Figure 4 | Page 6 | `figures/density_control/density_control_01.pdf` | https://arxiv.org/e-print/2308.04079 | 2026-10-15 |
| `../figures/fig4_densification_ablation.png` | No-Split、No-Clone、Full 密度控制消融 | Figure 8 | Page 10 | `figures/ablations/densification.pdf` | https://arxiv.org/e-print/2308.04079 | 2026-10-15 |
| `../figures/fig5_anisotropy_ablation.png` | Full 与 Isotropic Gaussian 的定性消融 | Figure 10 | Page 11 | `figures/ablations/isotropy_vs_anisotropy2.pdf` | https://arxiv.org/e-print/2308.04079 | 2026-10-15 |

## 本地归档

- `paper.pdf`：arXiv 官方 PDF，14 页。
- `arxiv-source.tar`：arXiv v1 官方 TeX 源码与原始图包。
- `../figures/`：从官方源码提取并逐张验证的 5 张笔记用图。

## 口径说明

- 论文原始实时目标是 1080p ≥30 FPS；项目页当前展示的 ≥100 FPS 是更强的结果概括，笔记以论文表格和原始 PDF 为主要数字来源。
- 官方代码仓库当前主分支含论文发布后的训练加速、深度正则、曝光补偿、抗锯齿等更新；这些不作为 SIGGRAPH 2023 原始 baseline。
- 论文正文未完整披露所有 pruning/size 阈值与 optimizer learning rate，严格复现需进一步锁定官方代码 commit 和参数文件。
