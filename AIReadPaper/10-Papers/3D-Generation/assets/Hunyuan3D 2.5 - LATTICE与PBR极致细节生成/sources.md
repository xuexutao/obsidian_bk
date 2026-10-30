# Hunyuan3D 2.5 材料来源

- 论文：https://arxiv.org/abs/2506.16504
- 官方体验：https://3d.hunyuan.tencent.com
- 报告所列 GitHub：https://github.com/Tencent/Hunyuan3D-2
- 阅读日期：2026-10-30
- 本地论文：`sources/paper.pdf`

## 开源核查

截至 2026-10-30，未找到 Hunyuan3D 2.5 专属 GitHub 仓库、Hugging Face 模型仓库或 10B LATTICE / 2.5 PBR 权重。报告链接的 `Tencent-Hunyuan/Hunyuan3D-2` 仓库只在 README 中登记 2.5 技术报告，未包含可识别的 2.5 配置、权重或推理入口。

- LATTICE 官方项目页：https://lattice3d.github.io/
- LATTICE 官方仓库：https://github.com/wasahaiah/LATTICE
- 核验 commit：`8355d55dd6afed553bf596387d0e1ca37f29d782`（2025-12-04）
- 仓库内容：仅 `README.md` 与 `assets/`
- TODO：仍列出 `Code and Model`

独立论文 [[LATTICE - 大规模高保真三维生成表征]] 对 VoxSet、Voxel Queries、3D RoPE 与 scaling 做了更完整方法披露；它可解释 2.5 几何主干的技术来源，但其最高 4.5B 研究配置不能等同于 2.5 的 10B 端到端产品模型。笔记只据此写算法级 symbolic shape，并明确标记证据边界，不编造 2.5 代码入口与固定配置。

## 图片

均从官方 PDF 以 300 DPI 裁剪：

- `figures/01-pipeline.png`：Figure 3 / p.3
- `figures/02-shape-features.png`：Figure 4 / p.3
- `figures/03-pbr-framework.png`：Figure 5 / p.4
- `figures/04-shape-comparison.png`：Figure 6 / p.6
- `figures/05-texture-comparison.png`：Figure 7 / p.7
- `figures/06-user-study.png`：Figure 8 / p.9
