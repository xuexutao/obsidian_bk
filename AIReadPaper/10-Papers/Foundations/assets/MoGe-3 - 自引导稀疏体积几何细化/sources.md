# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2607.17967v2 | 全文，arXiv v2 | 2026-10-21 | 精读主来源 |
| `figures/01-teaser.png` | 官方源码原图 | https://arxiv.org/abs/2607.17967 | Figure 1 | 2026-10-21 | 细结构三维几何对比 |
| `figures/02-method-pipeline.png` | 官方源码原图 | https://arxiv.org/abs/2607.17967 | Figure 2 | 2026-10-21 | Base + Self-Guided Sparse 3D Refinement |
| `figures/03-qualitative-comparison.png` | 官方源码原图 | https://arxiv.org/abs/2607.17967 | Figure 3 | 2026-10-21 | point map / disparity 定性比较 |

## 官方在线材料

- arXiv：https://arxiv.org/abs/2607.17967
- 项目页：https://qft-333.github.io/moge3page/
- 系列代码：https://github.com/microsoft/MoGe
- 模型权重：https://huggingface.co/Ruicheng/moge-3-vitl

## 说明

- 笔记以 2026-07-21 的 arXiv v2 官方论文为准；截至 2026-10-21，arXiv 页面未标会议录用信息。
- 官方仓库说明 MoGe-3 依赖 FlexGEMM/Triton，macOS 无原生 wheel；论文正文只将其抽象为 sparse 3D convolution backend，笔记在复现部分区分论文方法与代码工程依赖。
- 三张关键图均由 arXiv 官方 TeX source 中的矢量 PDF 渲染为 PNG。
