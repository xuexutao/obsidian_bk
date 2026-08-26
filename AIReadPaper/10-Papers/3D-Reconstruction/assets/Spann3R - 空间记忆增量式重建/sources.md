# Spann3R - 来源材料清单

> 与 `../Spann3R - 空间记忆增量式重建.md` 配套使用。所有附件、图、PDF 均位于本目录的同级路径。

## 1. 原始 PDF

| 文件              | 路径                                                                      | 来源                                                                                                                | 获取日期       | 用途            |
| --------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------- | ------------- |
| 论文全文 PDF        | `assets/Spann3R - 空间记忆增量式重建/sources/paper.pdf`（52 MB / 14 页）       | [arXiv 2408.16061](https://arxiv.org/abs/2408.16061)                                                                | 2026-08-26 | 全文精读、方法/实验对照 |
| 官方代码 / 预训练权重     | https://github.com/HengyiWang/spann3r                                   | UCL 官方仓库                                                                                                            | —          | 模型 / 复现参考     |

## 2. 关键配图（figures/）

| 文件名                          | 原图位置               | 中文说明                                                                                                                                                            | 用于笔记段落            |
| ---------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `figures/fig1_overview.png`      | Figure 1, Page 1   | Overview teaser：有序或无序图像输入，无需相机参数，增量地输出每帧在统一全局坐标系下的 pointmap。                                                                                       | §2.1 总体设计         |
| `figures/fig2_architecture.png`  | Figure 3, Page 3   | 架构图：ViT encoder + 两个 interleaved decoder（沿用 DUSt3R），reference decoder 处理 memory readout + 当前帧 visual feature，target decoder 输出 pointmap + confidence。 | §2.2 网络结构         |
| `figures/fig3_memory.png`       | Figure 4, Page 4   | Spatial Memory Overview：query feature、注意力直方图（含 $5 \times 10^{-4}$ 阈值）、long-term memory、working memory、Fused feature。                            | §2.3 Spatial Memory |

> 所有图均通过 `~/.workbuddy/skills/research-field-survey/scripts/extract_figure.py` 从官方 PDF 直接裁剪，未做任何重绘。

## 3. 笔记结构与正文映射

- §1 研究背景：DUSt3R 的「全局对齐」痛点 + XMem 的 memory 范式可借鉴；
- §2 核心方法：target/reference decoder 角色解耦、Spatial Memory (working + long-term)、Curriculum 训练；
- §3 实验与结果：7Scenes / NRGBD / DTU 重建精度、>50 FPS 在线速度、消融；
- §4 局限与评价：自述局限（静态、阈值固定、顺序依赖）+ 个人评价（去掉 BA 的范式价值）+ 复现建议。

## 4. 与同库其他笔记的衔接

- 直接接 [[DUSt3R - 前馈式几何重建奠基]]（骨架基座）；
- 平行扩展 [[MASt3R - 匹配与重建统一]]（加匹配头）、[[MonST3R - 运动场景几何估计]]（动态场景）、[[Fast3R - 千图单次前馈重建]]（多视图 all-to-all）、[[VGGT - 前馈式通用三维重建]]（多任务）；
- 综述视角见 [[视觉基础模块综述 - Survey]]（DUSt3R / MASt3R 被列为代表）。

---

*维护者：金悦。配套笔记：[[Spann3R - 空间记忆增量式重建]]。最后更新：2026-08-26。*