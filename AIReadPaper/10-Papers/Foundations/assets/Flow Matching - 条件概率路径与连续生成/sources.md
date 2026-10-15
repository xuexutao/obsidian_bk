# Flow Matching 附件来源与核验记录

## 1. 论文原文

- 论文：Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. **Flow Matching for Generative Modeling**.
- arXiv ID：`2210.02747`
- 官方摘要页：https://arxiv.org/abs/2210.02747
- 官方 PDF：https://arxiv.org/pdf/2210.02747
- 官方 TeX 源包：https://arxiv.org/e-print/2210.02747
- 本地 PDF：`sources/paper.pdf`
- 版本核验：PDF 首页标明 `arXiv:2210.02747v2 [cs.LG] 8 Feb 2023`，共 28 页。
- PDF SHA-256：`5eeb39ba516396924aba4787452f9d0abdee88467a4d0c264d9f66cad0c5ee14`
- 核验日期：2026-10-15

## 2. 图片来源与逐张核验

所有图片均来自上述官方 arXiv v2 TeX 源包。为便于 Obsidian 阅读，只对官方面板做等比例缩放、拼接和文字标签补齐；没有生成、重绘或改动图中实验数据。

| 本地文件 | 对应原论文 | 官方 TeX 素材 | 核验结果 | SHA-256 |
|---|---|---|---|---|
| `fig2_vector_fields.png` | Figure 2：Diffusion conditional score 与 OT conditional vector field | `figures/plots/2d_vf_score_dif_{0,1,2,3}.pdf`；`figures/plots/2d_vf_flow_match_ot_{0,1,2,3}.pdf` | 1500×780，8 个时刻面板均完整；箭头、目标点、源点和背景等高线清晰；与 PDF 第 6 页 Figure 2 的上下两行内容一致，无边缘裁切。 | `c1f8a2d98c7dfafac7779d012a5f016ef335ed1e85c49f863e6d0a43aea008d9` |
| `fig3_ot_vs_diffusion_trajectories.png` | Figure 3：Diffusion 与 OT 条件轨迹 | `figures/2d_traj/2d_traj_diff.png`；`figures/2d_traj/2d_traj_ot.png` | 1440×800，两幅源图完整；Diffusion 的弯曲、过冲/回折与 OT 的直线轨迹清楚可辨；与 PDF 第 6 页 Figure 3 一致。 | `22f0f73c3c74cb3e40146633ade0df52621c26f2e6952e4edd69ad3acdd09950` |
| `fig6_imagenet64_sample_paths.png` | Figure 6：ImageNet-64 同一初始噪声的三种采样轨迹 | `figures/imagenet64/imagenet64_sm_difv2.png`；`figures/imagenet64/imagenet64_fm_difv2.png`；`figures/imagenet64/imagenet64_fm_otv2.png` | 1780×1589，三种方法各 3 条轨迹、全部时间帧和终点图像完整；可见 OT 路径较早显现图像结构，Diffusion 路径后段才快速去噪；与 PDF 第 8 页 Figure 6 一致。 | `38358336c1120f8119455c063cd94ad39104b4335d1e44ef76b4bc5c3fc76d23` |

## 3. 后续工作关系的核对来源

以下只用于笔记中“后续影响”小节，不能当作 Lipman et al. 原论文的实验或结论：

- Rectified Flow：Xingchao Liu, Chengyue Gong, Qiang Liu. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**. arXiv:2209.03003. https://arxiv.org/abs/2209.03003
- SiT：Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, Saining Xie. **SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers**. arXiv:2401.08740. https://arxiv.org/abs/2401.08740

## 4. 边界说明

- “OT path”在原论文中首先是**每个条件高斯端点之间**的 Wasserstein-2 displacement interpolation。作者明确提醒：条件流最优不代表混合后的边际向量场就是全局最优传输解。
- Rectified Flow 与本论文为同期、相关但独立的工作；SiT 是 2024 年后续工作。笔记已将它们与原论文贡献分开标注。
