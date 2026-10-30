# Hunyuan3D 2.5 材料来源

- 论文：https://arxiv.org/abs/2506.16504
- 官方体验：https://3d.hunyuan.tencent.com
- 报告所列 GitHub：https://github.com/Tencent/Hunyuan3D-2
- 阅读日期：2026-10-30
- 本地论文：`sources/paper.pdf`

## 开源核查

截至 2026-10-30，未找到 Hunyuan3D 2.5 专属 GitHub 仓库、Hugging Face 模型仓库或 10B LATTICE / 2.5 PBR 权重。报告链接的 `Tencent/Hunyuan3D-2` 仓库只在 README 中登记 2.5 技术报告，未包含可识别的 2.5 配置、权重或推理入口。因此本笔记不编造 2.5 代码级张量流。

独立论文 [[LATTICE - 大规模高保真三维生成表征]] 对 VoxSet、Voxel Queries、3D RoPE 与 scaling 做了更完整方法披露，并有其独立代码语境；它可解释 2.5 几何主干的技术来源，但不能等同于 2.5 端到端系统代码。

## 图片

均从官方 PDF 以 300 DPI 裁剪：

- `figures/01-pipeline.png`：Figure 3 / p.3
- `figures/02-shape-features.png`：Figure 4 / p.3
- `figures/03-pbr-framework.png`：Figure 5 / p.4
- `figures/04-shape-comparison.png`：Figure 6 / p.6
- `figures/05-texture-comparison.png`：Figure 7 / p.7
- `figures/06-user-study.png`：Figure 8 / p.9
