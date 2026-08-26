# Fast3R - 来源材料清单

> 与 `../Fast3R - 千图单次前馈重建.md` 配套使用。所有附件、图、PDF 均位于本目录的同级路径。

## 1. 原始 PDF

| 文件              | 路径                                                                  | 来源                                                                                                                | 获取日期       | 用途            |
| --------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------- | ------------- |
| 论文全文 PDF        | `assets/Fast3R - 千图单次前馈重建/sources/paper.pdf`（24 MB / 16 页）     | [arXiv 2501.13928](https://arxiv.org/abs/2501.13928)                                                                | 2026-08-26 | 全文精读、方法/实验对照 |
| 官方代码            | https://github.com/facebookresearch/fast3r                           | Meta FAIR 官方仓库                                                                                                    | —          | 模型 / 复现参考     |

## 2. 关键配图（figures/）

| 文件名                            | 原图位置                | 中文说明                                                                                                                                                                                                                                | 用于笔记段落            |
| ------------------------------ | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `figures/fig1_overview.png`        | Figure 1, Page 1    | Overview teaser：从 1000+ 无序、无位姿图像出发，单次前向输出稠密 pointmap。                                                                                                                                                                                  | §1 研究背景          |
| `figures/fig2_architecture.png`    | Figure 2, Page 3    | Model Architecture：所有 N 张图的 patch token 一起做 all-to-all attention（Fusion Transformer），每 token 输出 local pointmap + global pointmap + 双 confidence。                                                                              | §2.2 网络结构         |
| `figures/fig3_qualitative.png`     | Figure 3, Page 4    | Qualitative Output：Fast3R 在多个真实场景（室内、室外、物体级）上的重建结果，含相机位姿框与稠密点云。                                                                                                                                                                | §3.4 与 Spann3R 对比  |

> 所有图均通过 `~/.workbuddy/skills/research-field-survey/scripts/extract_figure.py` 从官方 PDF 直接裁剪，未做任何重绘。

## 3. 笔记结构与正文映射

- §1 研究背景：pairwise 范式的 $O(N^2)$ 瓶颈与「一次前向」的工程野心；
- §2 核心方法：Fusion Transformer + all-to-all attention、dual pointmap、position interpolation、FlashAttention + ZeRO-2 + tensor parallelism；
- §3 实验与结果：CO3Dv2 / RealEstate10K 位姿（1000 视图 AUC@30 ≈ 99%）、DTU 重建、scaling 行为、Lighthouse 大场景、BA 互补、消融；
- §4 局限与评价：自述局限（显存、动态、位置插值极限）+ 个人评价（pairwise → all-to-all 的范式价值）+ 复现建议。

## 4. 与同库其他笔记的衔接

- 直接接 [[DUSt3R - 前馈式几何重建奠基]]（pairwise 基座）；
- 平行 [[MASt3R - 匹配与重建统一]]、[[Spann3R - 空间记忆增量式重建]]、[[MonST3R - 运动场景几何估计]]、[[VGGT - 前馈式通用三维重建]]、[[π³ - 置换等变视觉几何学习]]；
- 综述视角见 [[视觉基础模块综述 - Survey]]（DUSt3R / MASt3R 被列为代表）。

---

*维护者：金悦。配套笔记：[[Fast3R - 千图单次前馈重建]]。最后更新：2026-08-26。*