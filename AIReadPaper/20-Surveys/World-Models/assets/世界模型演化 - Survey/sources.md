# 世界模型演化 - Survey · 材料溯源

> 本文件记录综述正文引用的九篇论文的官方来源、关键图出处、下载日期与用途。图片正文用相对路径引用，见 `figures/`；官方论文 PDF 见 `sources/`。

## 九篇论文官方来源

| # | 论文 | 原始链接 | 获取日期 | 用途 |
|---|---|---|---|---|
| 01 | PlaNet: Learning Latent Dynamics for Planning from Pixels | https://arxiv.org/abs/1811.04551 | 2026-08-25 | 综述 §2 |
| 02 | When to Trust Your Model (MBPO) | https://arxiv.org/abs/1906.08253 | 2026-08-25 | 综述 §3 |
| 03 | Objective Mismatch in Model-based RL | https://arxiv.org/abs/2002.04523 | 2026-08-25 | 综述 §4 |
| 04 | Planning to Explore via Self-Supervised World Models | https://arxiv.org/abs/2005.05960 | 2026-08-25 | 综述 §5 |
| 05 | TD-MPC: Temporal Difference Learning for MPC | https://arxiv.org/abs/2203.04955 | 2026-08-25 | 综述 §6 |
| 06 | Transformers are Sample-Efficient World Models (IRIS) | https://arxiv.org/abs/2209.00588 | 2026-08-25 | 综述 §7 |
| 07 | LINGO-2: Driving with Natural Language | https://wayve.ai/thinking/lingo-2-driving-with-language/ | 2026-08-25 | 综述 §8.1（官方博客，无 arXiv） |
| 08 | GAIA-2: Pushing the Boundaries of Video Generative Models | https://wayve.ai/?p=8694 | 2026-08-25 | 综述 §8.2（官方博客，无 arXiv） |
| 09 | Cosmos World Foundation Model Platform for Physical AI | https://arxiv.org/abs/2501.03575 | 2026-08-25 | 综述 §9 |

## 关键图出处（正文引用的 01–09 图）

| 图 | 内容 | 来源（Figure/页面） |
|---|---|---|
| 01-planet-rssm-architecture.png | PlaNet RSSM 结构（确定性/随机路径） | PlaNet 论文 Fig.1 附近 |
| 02-mbpo-model-error.png | 模型误差随 rollout 增长、回报见顶示意 | MBPO 论文 Fig.1 附近 |
| 03-obj-mismatch-illustration.png | objective mismatch 概念示意 | Objective Mismatch 论文 Fig.1 附近 |
| 04-plan2explore-paradigm.png | 探索→通用世界模型→下游适配范式 | Plan2Explore 论文 Fig.1 附近 |
| 05-tdmpc-overview.png | TD-MPC 方法总览 | TD-MPC 论文 Fig.1/2 附近 |
| 06-iris-imagination.png | 世界模型想象 + policy 学习示意 | IRIS 论文 Fig.1 附近 |
| 07-cosmos-wfm-results.png | Cosmos WFM 视频生成结果 | Cosmos 论文结果页 |
| 08-lingo2-architecture.png | LINGO-2 视觉 token → 语言模型架构 | Wayve LINGO-2 官方博客 |
| 09-gaia2-cover.png | GAIA-2 多摄像头可控驾驶视频生成示意 | Wayve GAIA-2 官方博客 |

> 说明：图片从官方 PDF 页面完整裁剪或取自官方博客封面；`figures/` 下另保留各论文的原始裁剪图（`planet_p*.png`、`mbpo_p*.png` 等），供后续核对。LINGO-2 / GAIA-2 无 arXiv PDF，图取自 Wayve 官方博客。

## 官方论文 PDF（`sources/`）

| 文件 | 对应论文 |
|---|---|
| planet.pdf | PlaNet |
| mbpo.pdf | MBPO |
| objective_mismatch.pdf | Objective Mismatch |
| plan2explore.pdf | Plan2Explore |
| tdmpc.pdf | TD-MPC |
| iris.pdf | IRIS |
| cosmos.pdf | Cosmos |

（LINGO-2 / GAIA-2 为 Wayve 官方博客，无 PDF。）
