# Sources

> 主题：`Video Distillation：经典主线与2026前沿 - Survey`  
> 获取/核对日期：2026-11-03  
> 原则：论文主张、指标与局限优先依据 arXiv / CVF Open Access / 作者项目页；既有关键图复用自同一 Obsidian 库的官方论文裁图，并保留原始 Figure 溯源。

## 1. 综述图片

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `figures/00-evolution-map.svg` | 综述自绘路线图 | 本综述根据下列官方论文整理 | 自绘 | 总体技术演进 |
| `figures/01-dmd-method-overview.png` | 官方论文图复用 | https://arxiv.org/abs/2311.18828 | DMD Figure 2 | 解释 DMD 三网络与 score difference |
| `figures/02-diffusion-forcing-overview.png` | 官方论文图复用 | https://arxiv.org/abs/2407.01392 | Diffusion Forcing Figure 2 | 解释 token-wise 独立噪声 |
| `figures/03-causvid-asymmetric-dmd.png` | 官方论文图复用 | https://arxiv.org/abs/2412.07772 | CausVid method figure | 解释 asymmetric video DMD |
| `figures/04-self-forcing-paradigms.png` | 官方论文图复用 | https://arxiv.org/abs/2506.08009 | Self Forcing Figure 1 | 比较 Teacher/Diffusion/Self Forcing |
| `figures/05-causal-forcing-injectivity.png` | 官方论文图复用 | https://arxiv.org/abs/2602.02214 | Causal Forcing Figure 3 | 解释 frame-level injectivity |
| `figures/06-duomatching-overview.png` | 官方论文图复用 | https://arxiv.org/abs/2610.03543 | DuoMatching Figure 3 | 解释 joint-marginal、LatentBridge、LVS |
| `figures/07-six-gaps-matrix.svg` | 综述自绘框架图 | 本综述根据下列官方论文整理 | 自绘 | 统一“六类鸿沟”分类 |

## 2. 新增官方论文 PDF

| 本地文件 | 论文 | 原始链接 | 用途 |
|---|---|---|---|
| `sources/causal-forcing-plus-plus-2605.15141.pdf` | Causal Forcing++ | https://arxiv.org/pdf/2605.15141 | causal CD、成本与指标核对 |
| `sources/sparse-forcing-2604.21221.pdf` | Sparse Forcing | https://arxiv.org/pdf/2604.21221 | 稀疏注意力、PBSA、长视频指标 |
| `sources/causal-rcm-2606.25473.pdf` | Causal-rCM | https://arxiv.org/pdf/2606.25473 | TF-CM + SF-DMD、JVP kernel |
| `sources/one-forcing-2605.23458.pdf` | One-Forcing | https://arxiv.org/pdf/2605.23458 | 一步 DMD+GAN、曲率分析 |
| `sources/reward-forcing-2512.04678.pdf` | Reward Forcing | https://arxiv.org/pdf/2512.04678 | EMA-Sink、Re-DMD |
| `sources/data-forcing-2606.18478.pdf` | Data-Forcing Distillation | https://arxiv.org/pdf/2606.18478 | 真实样本 teacher-score 与多样性 |
| `sources/self-gradient-forcing-2607.20368.pdf` | Self Gradient Forcing | https://arxiv.org/pdf/2607.20368 | historical context-gradient gap |
| `sources/sgf-plus-2610.10429.pdf` | SGF+ | https://arxiv.org/pdf/2610.10429 | writer/denoiser 梯度冲突与解耦 |
| `sources/dmad-2610.02188.pdf` | DMAD | https://arxiv.org/pdf/2610.02188 | 密度比分类替代 fake score critic |
| `sources/leta-2610.34092.pdf` | Leta | https://arxiv.org/pdf/2610.34092 | ScoreDelta 与 FutureCredit |
| `sources/sf2-2610.35594.pdf` | Self Flow Forcing (SF²) | https://arxiv.org/pdf/2610.35594 | 内部表征预测正则 |
| `sources/ptd-2610.11498.pdf` | Parametric Trajectory Distillation | https://arxiv.org/pdf/2610.11498 | DMD 之外的轨迹参数化路线 |

> 上表 12 份 PDF 均已下载完成并通过 `%PDF-` 文件头校验；每个 URL 均为 arXiv 官方 PDF，不使用二手页面替代原文。

## 3. 官方项目页与代码

| 工作 | 项目/代码 | 核对内容 |
|---|---|---|
| DMD | https://tianweiy.github.io/dmd/ | 方法定位、速度与图像侧基线 |
| Diffusion Forcing | https://boyuan.space/diffusion-forcing/ | 训练范式与长 rollout 演示 |
| CausVid | https://causvid.github.io/ | asymmetric DMD、流式推理 |
| Self Forcing | https://self-forcing.github.io/ | self-rollout 与 rolling KV |
| Self-Forcing++ | https://self-forcing-plus-plus.github.io/ | 分钟级生成、Visual Stability、局限 |
| Causal Forcing | https://thu-ml.github.io/CausalForcing.github.io/ | frame-level injectivity 与受控对比 |
| Causal Forcing++ | https://github.com/thu-ml/Causal-Forcing | 代码、checkpoint 与 minWM 入口 |
| Sparse Forcing | https://boxunxu.top/SparseForcing | 持久块、PBSA、长视频演示 |
| Causal-rCM | https://github.com/NVlabs/rcm | 开放训练配方与基础设施 |
| One-Forcing | https://github.com/Aurora-edu/One-Forcing | 一步模型与代码 |
| Reward Forcing | https://reward-forcing.github.io/ | 23.1 FPS、长视频与消融 |
| Data-Forcing | https://csy2077.github.io/dfd-project-page/ | 代码、模型和数据发布说明 |
| SGF | https://zhuang2002.github.io/SelfGradientForcing/ | 两遍训练、长视频示例 |
| SGF+ | https://zihan-su.github.io/self-gradient-forcing-plus | 24 小时演示与资源数据 |
| DMAD | https://yzmblog.github.io/projects/DMAD/ | 图像/视频/音视频结果 |
| DuoMatching | https://johnzhan2023.github.io/DuoMatching/ | joint-marginal 目标与成对视频 |
| Leta | https://scdzx.github.io/Leta/ | ScoreDelta/FutureCredit 可视化 |
| PTD | https://alan-lanfeng.github.io/PTD/ | 轨迹曲率与 MiniMax-H3 结果 |

## 4. 口径说明

1. 不同论文使用 Wan2.1-1.3B / Wan2.1-14B / MiniMax-H3-33B、frame-wise / chunk-wise、不同 prompt 与 VBench 复现，绝对分数不可直接排名。
2. Causal Forcing++ 的 1/2-step 设置中，首个 latent frame 仍使用 4 steps；论文延迟不含 VAE。
3. One-Forcing 主结果使用 First-Frame Enhancement，首 block 4 steps、后续 block 1 step。
4. Sparse Forcing 摘要和主表使用了不同配置口径；正文报告主表可复算值，并明确标注差异。
5. SGF+ 的 24 小时结果属于项目页持续 rollout 演示；正式定量集中在 60 秒与 240 秒。
6. Reward Forcing 的奖励强度可显著提升 Dynamic，但会损害 Quality；不将最高 Dynamic 视为无条件更优。
7. DuoMatching 的主要收益是视觉质量和语义，动态人评基本持平。
