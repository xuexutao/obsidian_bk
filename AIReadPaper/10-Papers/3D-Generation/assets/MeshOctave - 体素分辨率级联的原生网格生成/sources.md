# MeshOctave 精读素材溯源

- 论文：MeshOctave: Vertex Split-and-Rewire Cascades for Native Mesh Generation
- arXiv:2609.38985v1 [cs.CV]，2026-09-30 提交
- 作者：Junkai Lin et al.（HUST / Meshy AI / TUM）
- 获取日期：2026-10-22
- 获取方式：arXiv 官方 PDF（https://arxiv.org/pdf/2609.38985v1），scan_figures 定位 caption bbox + extract_figure.py 按相对坐标裁剪，全部为官方原图裁剪、无内嵌碎片

## 图片清单

| 文件 | Figure/Page | 原始 URL | 用途 |
|---|---|---|---|
| 01-teaser-cascading-resolution.png | Figure 1 / p.1 | https://arxiv.org/pdf/2609.38985v1 | 级联分辨率生成总览（teaser） |
| 02-face-centric-split-and-rewire.png | Figure 2 / p.4 | 同上 | split-and-rewire 表示：8-bit 占据 + 8×8 连接矩阵 |
| 03-scale-conditioned-diffusion.png | Figure 3 / p.5 | 同上 | 模型架构：hourglass + 离散扩散 + 双头解码 |
| 04-qualitative-comparison.png | Figure 4 / p.8 | 同上 | 与 AR / flow 基线定性对比 |
| 05-generation-process.png | Figure 6 / p.8 | 同上 | 尺度间生成与尺度内去噪可视化 |
| 06-lod-baseline-comparison.png | Figure 5 / p.8 | 同上 | 与 VertexRegen / ARMesh 的 LOD 对比 |
| 07-ablation-table-qualitative.png | Table 3 + Figure 7 / p.9 | 同上 | 消融定量与定性 |
| 08-mesh-subdivision.png | Figure 8 / p.9 | 同上 | 网格细分应用 |
| 09-intra-scale-denoising.png | Figure 9 / p.16 | 同上 | 尺度内去噪逐步过程 |
| 10-generation-across-resolutions.png | Figure 10 / p.17 | 同上 | 跨分辨率生成轨迹（LOD） |
| 11-inference-speed.png | Figure 11 / p.17 | 同上 | 推理耗时 vs 面数曲线 |

## 其他来源

- 项目页：https://maymhappy.github.io/MeshOctave/
- alphaXiv 中文摘要（辅助理解，未直接引用）：https://www.alphaxiv.org/zh/abs/2609.38985
