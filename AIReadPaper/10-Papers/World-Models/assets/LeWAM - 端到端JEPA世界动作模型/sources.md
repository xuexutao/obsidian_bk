# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v1，33 页，含附录） | https://arxiv.org/pdf/2610.12818v1 | 全文 | 2026-10-16 | 精读主来源；SHA-256：`059942527366ac96c2ff413cc9b0c9b9cd7382be4d7a0c2ba01ba13756e734ce` |
| `sources/arxiv-source.tar` | 官方 arXiv TeX 源归档 | https://export.arxiv.org/e-print/2610.12818v1 | 全部 TeX、表格与原始图 | 2026-10-16 | 公式、表格、图注和实验数字核验 |
| `sources/tex/` | 官方 TeX 源解包副本 | https://export.arxiv.org/e-print/2610.12818v1 | `main.tex`、`sections/`、`tables/`、`figures/` | 2026-10-16 | 逐段精读及关键图原始矢量文件提取 |
| `figures/01-training-overview.png` | 总体训练架构图 | https://arxiv.org/abs/2610.12818v1 | Figure 1 上半部，PDF p.1；TeX `figures/lewam_figure1_python_package/figures/lewam_figure1_python.pdf` | 2026-10-16 | 解释 shared encoder、SIGReg、MoT、latent/action 双目标 |
| `figures/02-positioning.png` | 方法定位图 | https://arxiv.org/abs/2610.12818v1 | Figure 2，PDF p.2；TeX `figures/lewam_three_consequences.pdf` | 2026-10-16 | 对照 LeWM、V-JEPA 2、SIGReg/DINOv2、视频 WAM 与动作搜索 |
| `figures/03-planning-mechanism.png` | 规划与执行机制图 | https://arxiv.org/abs/2610.12818v1 | Figure 4，PDF p.5；TeX `figures/lewam_planning_python/output/lewam_planning_autoregressive.pdf` | 2026-10-16 | 解释 action proposal、latent rollout、gradient refinement 与 best-of-N |
| `figures/04-main-results.png` | 十环境主要实验图 | https://arxiv.org/abs/2610.12818v1 | Figure 6，PDF p.7；TeX `figures/sr_short_horizon.pdf` | 2026-10-16 | 比较 LeWAM、LeWM、DINO-WM、GCBC 的视觉规划成功率 |
| `figures/05-ablations.png` | 关键消融与行为克隆图 | https://arxiv.org/abs/2610.12818v1 | Figure 8，PDF p.8；TeX `figures/model_ablations/model_ablations.pdf` | 2026-10-16 | SIGReg、视觉编码器、flow vs. MSE、Diffusion Policy 消融/对比 |

## 在线来源

- arXiv 摘要页（v1）：https://arxiv.org/abs/2610.12818v1
- 项目页：https://le-wam.github.io/
- 官方代码：https://github.com/MinghaoFu/lewam
- Flow-JEPA 官方 arXiv 页面（用于相关工作对照；LeWAM 正文未引用该工作）：https://arxiv.org/abs/2608.29029v3

## 提取与核验说明

- 五张图片均直接由官方 arXiv TeX 源中的完整矢量 PDF 以 300 DPI 渲染，没有使用 PDF 内嵌位图碎片。
- 已逐张目视检查：边缘、标题、坐标轴、图例、文本与流程箭头完整，清晰度可满足 Obsidian 放大阅读。
- Figure 1 在论文中由训练架构与结果总览上下拼接；本地 `01-training-overview.png` 保留其训练架构上半部，结果证据由 Figure 6、Figure 8 及正文表格单独呈现。
- arXiv v1 提交时间为 2026-10-08 23:58:36 UTC。PDF 已含 Supplementary/Appendix，因此没有单独的 supplementary PDF。
