# WALL-SS 材料来源与溯源

## 1. 官方论文

- 标题：WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression
- arXiv：https://arxiv.org/abs/2608.26239
- arXiv HTML：https://arxiv.org/html/2608.26239
- 官方 PDF：https://arxiv.org/pdf/2608.26239
- 本地归档：`sources/paper.pdf`
- arXiv 版本：v1，2026-08-26；PDF 首页标注 Date: August 28, 2026
- 读取日期：2026-10-29

## 2. 官方项目与代码核查

- 官方项目页：https://x2robot.com/en/pages/ss
- 官方 GitHub：https://github.com/X-Square-Robot/wall-ss
- 核查 commit：`04951f54db010ba289409a210fb8cb31ab7ea6f7`
- commit 日期：2026-08-28T13:04:12+08:00
- 本地保留：`sources/official-repo-README.md`、`sources/LICENSE`
- 许可证：仓库含 MIT License，但 README 明确写的是“Code will be released under the MIT License”。
- 截至 2026-10-29 的开源状态：仓库只有 README、论文 PDF、项目展示图和社区二维码；README TODO 中“Release the training and inference code”仍未完成。未发现训练代码、推理代码、配置、模型权重、数据处理脚本或 Hugging Face 模型仓库。因此本笔记不虚构代码级张量流，仅按论文公开信息分析。

## 3. 本地关键图

| 文件 | 原始出处 | Figure / 页码 | 用途 |
|---|---|---|---|
| `figures/01-overview.jpg` | 官方 GitHub `assets/teaser.jpg` | 对应论文 Figure 1，PDF p.2 | 总览：异构数据、next-scale AR、有界记忆、闭环 rollout |
| `figures/02-framework.jpg` | 官方 GitHub `assets/framework.jpg` | 对应论文 Figure 3，PDF p.7 | 完整方法框架 |
| `figures/03-closed-loop-calibration.jpg` | 官方 GitHub `assets/closed-loop-calibration.jpg` | 对应论文 Figure 12，PDF p.20 | 任务级校准与策略 checkpoint 排序 |
| `figures/04-counterfactual-actions.png` | 从官方 PDF 300 DPI 裁剪 | Figure 7，PDF p.16 | 同初始状态、不同动作的反事实未来 |
| `figures/05-long-horizon-ablation.png` | 从官方 PDF 300 DPI 裁剪 | Figure 9，PDF p.18 | time-scale memory 与 dream forcing 消融 |
| `figures/06-sim-real-outcomes.png` | 从官方 PDF 300 DPI 裁剪 | Figure 13 + Table 3，PDF p.21 | 600 对 sim-real episode 一致性与误差分布 |
| `figures/07-real-robot-results.png` | 从官方 PDF 300 DPI 裁剪 | Figure 14，PDF p.22 | 真机 Task Progress 对比 |

## 4. 主要事实对应出处

- 因果交错的观察—动作序列、动作条件 next-scale prediction：论文 §3.1–3.2，PDF pp.6–7。
- time–scale KV memory、固定 anchor、last-use eviction：论文 §3.3，PDF pp.7–8。
- scale-wise dream forcing：论文 Eq. (4)–(5)，PDF p.8。
- flow-matching action expert：论文 §3.4、Eq. (6)，PDF p.8。
- on-policy visual alignment、组相对更新、KL 与 AR replay：论文 §3.5、Eq. (7)–(9)，PDF pp.8–9。
- 数据组成与 10 FPS 管线：论文 §4、Figure 4，PDF pp.9–12。
- 训练与推理阶段：论文 §5，PDF pp.13–14。
- 视频、动作、长程、RL、sim-real 与真机实验：论文 §6，PDF pp.14–22。

## 5. 公开信息缺口

论文与仓库均未披露足以独立复现的模型参数量、Transformer 层数/宽度、视觉 token 尺度表、VAE latent 尺寸、动作/记忆尺度集合、KV memory budget、优化器、学习率、batch size、训练步数、训练硬件和采样参数。任何未给出的形状或超参都不应从 InfinityStar 擅自外推为 WALL-SS 的实现事实。
