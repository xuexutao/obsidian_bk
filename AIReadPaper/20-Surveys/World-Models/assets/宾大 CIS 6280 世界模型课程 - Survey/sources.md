# 来源清单 · 宾大 CIS 6280 世界模型课程

获取日期：2026-09-23

## 一手材料（官方）

| 本地文件 | 原始 URL | 用途 |
|---|---|---|
| `sources/lecture-01-world-models-overview.pdf` | https://www.cis.upenn.edu/~cis6280/lectures/lecture-01-world-models-overview.pdf | L1 概述（纯图版，文本层为空，内容由交互式 HTML 页面与中文拆解交叉核对） |
| `sources/lecture-02-world-models-history-foundations-probabilistic-formulation.pdf` | https://www.cis.upenn.edu/~cis6280/lectures/lecture-02-world-models-history-foundations-probabilistic-formulation.pdf | L2 历史谱系 + 概率表述（81 页） |
| `sources/lecture-03-environments-simulators-rollouts.pdf` | https://www.cis.upenn.edu/~cis6280/lectures/lecture-03-environments-simulators-rollouts.pdf | L3 环境 / 模拟器 / Rollout（64 页） |
| `sources/lecture-04-state-space-models.pdf` | https://www.cis.upenn.edu/~cis6280/lectures/lecture-04-state-space-models.pdf | L4 状态空间模型与卡尔曼滤波（56 页） |
| `sources/lecture-04-lgssm-handwritten-notes.pdf` | https://www.cis.upenn.edu/~cis6280/lectures/lecture-04-lgssm-handwritten-notes.pdf | L4 LGSSM 手写推导补充（3 页） |
| `sources/lecture-05-representation-learning-i.pdf` | https://www.cis.upenn.edu/~cis6280/lectures/lecture-05-representation-learning-i.pdf | L5 自监督表示学习 I（59 页） |
| `sources/lecture-06-representation-learning-ii.pdf` | https://www.cis.upenn.edu/~cis6280/lectures/lecture-06-representation-learning-ii.pdf | L6 自监督表示学习 II（73 页） |
| `sources/lecture-07-generative-models-i.pdf` | https://www.cis.upenn.edu/~cis6280/lectures/lecture-07-generative-models-i.pdf | L7 潜变量与对抗模型（52 页） |

课程主页（日程、资源、Coursework、Logistics 全量）：https://www.cis.upenn.edu/~cis6280/
讲师主页：https://jiataogu.me/ ｜ 课程镜像页：https://jiataogu.me/cis6280-world-models/
课程日历：https://www.cis.upenn.edu/~cis6280/cis6280-fall-2026.ics
官方课程列表：https://courses.upenn.edu/?details&srcdb=202630&crn=90075

**注**：L8 起（2026-09-22 之后）官方尚未发布讲义 PDF，本综述对 L8–L23 的描述基于课程官网公布的官方日程标题与摘要，已在正文中明确标注为「尚未发布讲义」。

## 图片来源（figures/）

所有图均由官方 PDF 以 200 DPI 整页渲染得到（幻灯片本身即为整页图文排版，非论文式插图，故整页渲染是完整呈现方式）。

| 文件 | 来源 | 页码 |
|---|---|---|
| `fig01-history-roadmap.png` | L2 讲义 | p.5 |
| `fig02-encoder-predictor-output.png` | L2 讲义 | p.70 |
| `fig03-same-history-different-futures.png` | L2 讲义 | p.60 |
| `fig04-inside-env-step.png` | L3 讲义 | p.12 |
| `fig05-contact-location-rotation.png` | L3 讲义 | p.37 |
| `fig06-predict-correct-recursion.png` | L4 讲义 | p.22 |
| `fig07-single-gaussian-merges-modes.png` | L4 讲义 | p.52 |
| `fig08-repr-l1-title.png` | L5 讲义 | p.1 |
| `fig09-repr-l2-title.png` | L6 讲义 | p.1 |
| `fig10-three-learning-signals.png` | L7 讲义 | p.3 |
| `fig11-world-models-2018-vmc.png` | L7 讲义 | p.42 |
| `fig12-three-collapse-problems.png` | L7 讲义 | p.51 |

## 二手中文解读（仅用于交叉核对与补充背景）

### 知乎 · quiescencent「2026.9 宾大世界模型 World Model CIS 6280 课程」系列

以课件截图 + 个人理解为主，具身视角强，对物理引擎与卡尔曼推导补充尤多。

| 篇次 | 覆盖讲次 | URL |
|---|---|---|
| 1/N | L1–L3（基础、历史、仿真器） | https://zhuanlan.zhihu.com/p/2083702068351714513 |
| 2/N | L4（状态空间模型） | https://zhuanlan.zhihu.com/p/2084785325193294899 |
| 3/N | L5–L6（表征学习） | https://zhuanlan.zhihu.com/p/2085341287616602994 |

### 觉醒学院「世界模型入门到精通」专题（8 篇逐讲拆解）

专题总览：https://www.jxxy.net/ai/paths/world-models-mastery

| 篇次 | 标题 | URL |
|---|---|---|
| 1.1 | 全球第一门世界模型研究生课：23 讲完整地图 | https://www.jxxy.net/ai/paths/world-models-mastery/world-models-mastery-01-01-overview/ |
| 1.2 | 第一讲 · 世界模型概述 | https://www.jxxy.net/ai/paths/world-models-mastery/world-models-mastery-01-02-lecture01/ |
| 2.1 | 第二讲 · 八十年简史 | https://www.jxxy.net/ai/paths/world-models-mastery/world-models-mastery-02-01-lecture02-history/ |
| 2.2 | 第二讲 · 概率表述 p(Y\|h,a) | https://www.jxxy.net/ai/paths/world-models-mastery/world-models-mastery-02-02-lecture02-formulation/ |
| 3.1 | 第三讲 · 环境、模拟器与 Rollout | https://www.jxxy.net/ai/paths/world-models-mastery/world-models-mastery-03-01-lecture03/ |
| 4.1 | 第四讲 · 状态空间模型 | https://www.jxxy.net/ai/paths/world-models-mastery/world-models-mastery-04-01-lecture04/ |
| 5.1 | 第五讲 · 表示学习 I | https://www.jxxy.net/ai/paths/world-models-mastery/world-models-mastery-05-01-lecture05/ |
| 5.2 | 第六讲 · 表示学习 II | https://www.jxxy.net/ai/paths/world-models-mastery/world-models-mastery-06-01-lecture06/ |

### 其他网络总结

- AGI Hunt：https://agihunt.info/p/1a093ef83e5cc70183325c7d847 （课程发布速报）
- MKLab：https://mklab.gr/showthread.php?tid=2013 （英文课程大纲概述）

## 课程官方推荐资源（Resources 板块，共 26 条）

**Essays & perspectives (6)**
- World Models — Ha & Schmidhuber: https://worldmodels.github.io/
- A Path Towards Autonomous Machine Intelligence — Yann LeCun: https://openreview.net/forum?id=BZ5a1r-kVsf
- From Words to Worlds — Fei-Fei Li: https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence
- A Functional Taxonomy of World Models — World Labs: https://www.worldlabs.ai/blog/taxonomy-of-world-models
- Agents That Imagine and Plan — Google DeepMind: https://deepmind.google/blog/agents-that-imagine-and-plan/
- The Quest for a Common Model of the Intelligent Decision Maker — Sutton: https://incompleteideas.net/papers/RLDM22-quest-common-model.pdf

**Tutorials & collections (6)**
- From Video Generation to World Model (CVPR 2025): https://world-model-tutorial.github.io/
- World Modeling Workshop (Mila 2026): https://world-model-mila.github.io/
- Awesome World Models: https://github.com/JiahuaDong/Awesome-World-Models
- From World Models to World Action Models: https://clearlab-sustech.github.io/WorldModelSurvey/
- World Models (CMU Generative AI): https://www.cs.cmu.edu/~mgormley/courses/10423/slides/lecture26-world-models.pdf
- The World of World Modeling (Stanford CS234): https://web.stanford.edu/class/cs234/slides/ShaneGuCS234_2026.pdf

**Talks & seminars (6)**
- Ilya Sutskever × Jensen Huang (NVIDIA GTC 2023): https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52092/
- The Sensorimotor Road to AI — Jitendra Malik (UC Berkeley 2023): https://news.berkeley.edu/2023/03/24/berkeley-talks-jitendra-malik/
- With Spatial Intelligence, AI Will Understand the Real World — Fei-Fei Li (TED 2024): https://www.ted.com/talks/fei_fei_li_with_spatial_intelligence_ai_will_understand_the_real_world
- Fireside Chat with Yann LeCun (RAISE 2026) / AFOSR 2024 / IHES 2023: YouTube（EvSe0ktD95k / OKkEdTchsiE / iDpPFAXmcZc）

**Systems & demos (8)**
- DreamerV3: https://danijar.com/project/dreamerv3/
- V-JEPA 2 (Meta AI): https://ai.meta.com/research/vjepa/
- Genie (Google DeepMind): https://deepmind.google/models/genie/
- Marble (World Labs): https://www.worldlabs.ai/blog/marble-world-model
- GAIA-4 (Wayve): https://wayve.ai/thinking/gaia-4/
- AIDO Cell Simulator (GenBio AI): https://genbio.ai/aido-cell-simulator/
- DayDreamer: https://danijar.com/project/daydreamer/
- DINO-WM (Zhou et al., ICML 2025): https://proceedings.mlr.press/v267/zhou25t.html

**逐讲延伸阅读（课程 schedule 内嵌）**
- L2 · MBRL History: https://loqmansamani.github.io/articles/model_based_rl/index.html
- L3 · Gymnasium Basic Usage: https://gymnasium.farama.org/introduction/basic_usage/
- L3 · 课程 Colab「Build the PushT World」: https://colab.research.google.com/drive/1EuxvdBaevrKno4vWL3hPNBTLNRgpgY1v
- L4 · Kalman Filter Tracking (dynamax): https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/kf_tracking.html
- L5 · Contrastive Representation Learning (Lilian Weng): https://lilianweng.github.io/posts/2021-05-31-contrastive/
- L6 · Deep Dive into Yann LeCun's JEPA: https://rohitbandaru.github.io/blog/JEPA-Deep-Dive/
- L7 · From Autoencoder to Beta-VAE (Lilian Weng): https://lilianweng.github.io/posts/2018-08-12-vae/
- L7 · Tutorial on Variational Autoencoders (Doersch): https://arxiv.org/abs/1606.05908
