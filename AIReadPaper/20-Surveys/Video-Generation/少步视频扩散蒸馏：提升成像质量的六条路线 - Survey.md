---
type: survey
status: done
domain: Video-Generation
paper: null
year: 2026
arxiv: null
doi: null
source: null
project: null
code: null
tags: [Video-Generation, Survey, Distillation, DMD, Few-Step, DuoMatching, Image-Teacher, Reward, Adversarial]
created: 2026-10-23
updated: 2026-10-23
---

# 少步视频扩散蒸馏：提升成像质量的六条路线

> **结论先看：** 把多步视频扩散压到 1–4 步这件事已经基本解决，真正没解决的是**压完之后画面变糊、细节丢失、prompt 不听话**。所有"提升少步蒸馏后成像质量"的工作，本质都在回答同一个问题：**除了"让学生模仿教师的输出分布"这条监督之外，还能从哪里再榨出一份质量信号？**
>
> 答案目前有六条路：**A 改轨迹（一致性/流映射）、B 改散度（分布匹配的数学形式）、C 加判别器（对抗补高频）、D 加奖励（对齐人类偏好）、E 换教师（借跨模态强先验）、F 改系统（量化与调度，不提上限但决定能否落地）。**
>
> **[[DuoMatching - 联合边缘分布匹配的少步视频生成]] 站在 E 类的第一个成熟样本上**——它不引入新的数学，而是引入一个**更强的图像教师**和一座**潜空间桥**（LatentBridge），把"帧级该长什么样"这份本来被 joint matching 忽略的监督显式补回去。
>
> **总体评级：★★★★☆（4/5）**——这个切面本身没有单点突破，但一旦按"质量信号从哪来"重新切，原本看起来互相打架的十几篇工作会立刻归位，而且能直接反过来用：**先看自己少步模型的症状，再决定该借哪一路的信号。**

## 基本信息

**报告主题**：少步（1–4 步）视频扩散蒸馏中"成像质量"的提升路线谱系，兼论 DuoMatching 的分类归属与近邻工作

**覆盖范围**：2022.06（Progressive Distillation）— 2026.10（DMAD / DuoMatching / Data-Forcing / Stream-R1）

**与既有两篇综述的分工**：

| 笔记 | 主轴 | 回答的问题 |
|---|---|---|
| [[因果视频蒸馏路线 - Survey]] | 纵向一条线 | 怎么把双向扩散蒸馏成因果自回归模型（慢 → 快、非流式 → 流式） |
| [[AR扩散与因果视频生成：五篇方法对比 - Survey]] | 横向五篇对比 | 因果生成的几种训练/推理范式怎么取舍 |
| **本篇** | **横向一个切面** | **已经少步了，画面质量还能从哪些地方补回来** |

**核心论文**（按本文六条路线归类，粗体为已有精读笔记）：

- **A 轨迹/一致性**：Progressive Distillation、Consistency Models / sCM / **rCM**、MeanFlow、PDD、TMD、AnyFlow
- **B 分数分布匹配**：**[[DMD - 分布匹配蒸馏一步图像生成]]**、**[[DMD2 - 改进的分布匹配蒸馏快速图像合成]]**、**[[SiD - 分数恒等蒸馏一步生成]]**、f-distill、SGMD、SenseFlow、DP-DMD、PhasedDMD、Uni-Instruct、FSF-DMD
- **C 对抗/判别**：ADD（SDXL-Turbo）、ADM / DMDX、DMAD、D2O-F
- **D 奖励/偏好**：Reward Forcing（Re-DMD）、Reward Lightning（LRM + HPD）、Stream-R1、AMD、DanceGRPO / Flow-GRPO
- **E 跨模态/多教师**：**[[DuoMatching - 联合边缘分布匹配的少步视频生成]]**、MILD
- **F 系统/工程**：TurboDiffusion、[[LongLive-2.0 - NVFP4长视频生成训练与推理系统]]

**教师/基座**：[[Wan2.1 - 开源大规模视频生成基础模型]]、[[Goku - 联合图像视频整流流基础模型]]

**撰写日期**：2026-10-23

**报告定位**：以「质量信号来源」为唯一分类轴，把少步视频蒸馏的方法版图切成六条路，先给 DuoMatching 精确定位，再给出「症状 → 该借哪路信号」的可操作对照表。

---

## 1. 先定位：DuoMatching 站在哪个格子里

### 1.1 三个正交的分类轴

判定一篇"提升少步质量"的工作属于哪一类，用三个轴就够了：

| 轴 | 取值 | DuoMatching 的位置 |
|---|---|---|
| **改的是哪一层** | 训练目标 / 模型架构 / 采样器 / 后处理 | **训练目标**（只动 loss） |
| **质量信号来自谁** | 同一位教师 / 额外教师 / 奖励模型 / 真实数据 | **额外教师**（图像教师 Qwen-Image） |
| **适配哪种部署形态** | 双向整段 / 因果流式 / 通用 | **通用**（因果与双向两种模式都报了数） |

三条轴交叉出来的落点非常干净：**「改训练目标 × 引入额外教师 × 两种形态通用」**。这也解释了它为什么推理零开销——第一轴就决定了它不可能增加推理成本。

### 1.2 一句话归类

> **DuoMatching = 分布匹配蒸馏（DMD）家族里的「多教师 / 跨模态监督」分支。**

更细地说，它不是在 DMD 的数学形式上做文章，而是在**监督源**上做文章：原本 DMD 只有一个监督源（视频教师给出的真实分布 score），它拆成两个——视频教师管**联合分布**（帧与帧怎么配合），图像教师管**边缘分布**（单帧长什么样）。

![](assets/少步视频扩散蒸馏：提升成像质量的六条路线%20-%20Survey/figures/01-duomatching-teaser-quality-gap.png)

> 这张 teaser 直接展示了本文要解决的"质量缺口"：同样是因果少步生成，Causal CD Init / Causal Forcing++ 与 DuoMatching 在 Temporal / Visual / Semantic 三个标记上的差异集中在**视觉与语义**两栏，左下角的 Q-Align 曲线给出量化差距。注意一个关键细节——论文特别指出**即便在首帧**（还没有任何 rollout 漂移可言）差距依然存在，这说明缺口不是"误差累积"造成的，而是**联合目标本身就没有约束单帧质量**。这是整篇工作的立论基点。来源：DuoMatching Figure 1（arXiv:2610.03543, p.2）。

### 1.3 DMD 家族内部的四个改进方向

把 DMD 当作基座，后续所有改进可以按"动了哪一处"分成四支，DuoMatching 属于第三支：

| 改进方向 | 动的东西 | 代表工作 | 主要收益 |
|---|---|---|---|
| **改散度形式** | 把 reverse KL 换成别的散度 | f-distill（Jeffreys）、SGMD（Fisher 匹配）、SiD、DP-DMD | 缓解 mode-seeking / 过饱和 |
| **改估计器** | fake score 怎么估、能不能不估 | DMAD（判别器 logit 代替）、FSF-DMD（从 flow map 推导替身） | 省显存、训练更稳 |
| **改/加监督源** | 谁提供"真实分布"的回答 | **DuoMatching（图像教师）**、AMD（reward proxy）、Reward Forcing（奖励加权） | **补上教师没覆盖的维度** |
| **加对抗信号** | 在 score 差之外再加判别损失 | DMD2、DMAD、ADM | 补高频、去模糊 |

这四支并不互斥（DMAD 同时占了"改估计器"和"加对抗"两格），但**"加监督源"这一支是 2026 年才明显成型的**，也是目前涨点最实在的一支——因为它引入的是**教师本身不具备的信息**，而其他三支都是在既有信息上换算法。

---

## 2. 六条路线纵览

### 2.1 A 类：轨迹与一致性——约束"怎么走"

**核心思想**：不去管输出分布，而是要求学生在概率流 ODE 轨迹上的落点与教师一致。Progressive Distillation 是最早的形态（每次把步数砍半，学生再当下一轮教师）；Consistency Models 要求同一条轨迹上任意点映射到同一个 $x_0$；MeanFlow / Flow Map 把这个约束写成连续的；rCM（regularized Consistency Model）是第一个把它 scale 到大型文生图/文生视频模型的工作；TMD（Transition Matching Distillation）则把匹配对象从"点"扩展到"子区间上的转移概率"，因此能处理大规模视频模型。

**对成像质量的作用机制**：主要保**结构正确性**——少步下最容易出现的残影、结构错乱、几何崩坏，靠轨迹约束压住。它是"打底"的一类。

**代价与局限**：轨迹约束本身是 mode-covering 的，容易**偏平滑、缺高频**；而且论文里反复提到一个现象——轨迹类方法 scale 到大模型时质量下降明显。所以它通常不是单独用，而是和 C/D 类叠加（rCM 就是"consistency + score regularization"）。

### 2.2 B 类：分数分布匹配——约束"落在哪"

**核心思想**：直接最小化学生输出分布与真实分布的散度，梯度写成两个 score 之差：$\nabla_\theta\mathcal{L}\propto\mathbb{E}\big[(\nabla_x\log p_{\text{fake}}-\nabla_x\log p_{\text{real}})\nabla_\theta G_\theta\big]$。DMD 用 reverse KL，DMD2 加判别器缓解 mode-seeking，f-distill 把它推广到积分 $f$-散度，SGMD 用 Fisher 型匹配，SiD 从 score identity 推导出等价目标，DP-DMD / PhasedDMD 把监督分散到不同步或不同噪声区间。

**对成像质量的作用机制**：保**整体观感与多样性**，比 A 类更不容易糊。视觉保真度的主力来自这一类。

**代价与局限**：reverse KL 天生 mode-seeking → **过饱和、多样性塌缩**；fake score 需要一个持续 refit 的辅助网络 → **显存与训练复杂度高**；而且所有 B 类方法都受限于"教师看到的维度"——教师是视频模型，它就只知道视频层面的好坏，不知道单帧的纹理该多细。这正是 DuoMatching 的切入点。

### 2.3 C 类：对抗与判别——补高频

**核心思想**：加一个判别器区分真实/生成（或教师/学生），用对抗损失补回 score 匹配丢失的高频细节。ADD（SDXL-Turbo 背后的技术）是这一类的开山之作；ADM / DMDX 用隐空间或像素级 GAN 直接对 score 做分布匹配；DMAD（2026.10）走得更远——它**完全不估 score**，用两个判别头（一个对真实数据、一个对教师样本）的 logit 直接恢复出分布匹配梯度，一举解决了 B 类"fake score 网络太贵"的问题。

**对成像质量的作用机制**：专治**糊**。频域上判别器的梯度天然偏向中高频，正好补上 score 匹配平滑掉的纹理。

**代价与局限**：GAN 训练不稳定是老问题；判别器容易带来 artifacts；DMAD 的聪明之处在于把"判别器"和"分布匹配"统一成了一个东西，但代价是要同时平衡两个 head 在不同噪声水平的权重。DMAD 在 Wan2.1-14B 上做到 VBench 85.15，是目前公开最好的四步视频学生之一。

### 2.4 D 类：奖励与偏好对齐——对标人类偏好

**核心思想**：VBench/人类偏好关心的很多东西（动态是否合理、构图是否舒服、有没有"闪回"）**根本不在教师分布里**——教师自己也会犯。于是引入一个外部奖励模型，用它的打分去**加权或改写** DMD 梯度：

- **Reward Forcing（Re-DMD）**：用运动质量奖励给样本加权，把分布匹配梯度偏向高动态样本，专治注意力 sink 导致的"运动幅度衰减、画面僵住/闪回首帧"；
- **Reward Lightning（LRM + HPD）**：先在**潜空间**训一个奖励模型（不 decode 回像素），再让判别与偏好对齐**共用同一个 backbone**，从根上消除"对齐"与"加速"两个目标的梯度冲突；
- **Stream-R1**：用**逐 rollout 可靠性**（外部视频奖励分）+ **逐区域困惑度**给 DMD loss 重加权——可信的 rollout 学得重、模型薄弱的区域学得重；
- **AMD**：用 reward proxy 适配分布匹配。

**对成像质量的作用机制**：补的是**"教师也不知道什么更好"的那部分**。这是 2026 年涨点最猛的一类，且几乎都满足"只改训练 loss、推理零开销"。

**代价与局限**：奖励模型本身的偏置会被放大（Stream-R1 明确承认这点）；偏好分容易 reward hacking；而且 VBench 上的领先幅度往往很薄（Stream-R1 84.40 vs 教师 84.26），真正的证据在长视频人工评测上。

### 2.5 E 类：跨模态与多教师——借外部强先验（DuoMatching 所在）

**核心思想**：视频教师的分布只覆盖了"帧之间怎么配合"，**单帧的画质先验其实图像模型更懂**——图像模型训练数据更大更干净、美学与文字渲染能力更强。那就把图像模型也拉进来当第二位教师，专门管"每一帧长什么样"。

![](assets/少步视频扩散蒸馏：提升成像质量的六条路线%20-%20Survey/figures/03-joint-vs-marginal-signal.png)

> 这两张热力图是 E 类立论最直观的证据：同一个生成帧上，**Joint Matching Signal 是散乱的**（因为是条件 score，单帧改动若与邻帧冲突就被抵消），而 **Marginal Matching Signal 呈现清晰连贯的空间结构**。换句话说——联合监督本身会**惩罚**"把某一帧变清晰"这种改进，因为清晰的邻帧和模糊的邻帧放一起反而不一致。这解释了为什么只加 joint 监督时模型学不会纹理。来源：DuoMatching Figure 2（arXiv:2610.03543, p.2）。

**两篇代表工作**：

| | DuoMatching（CUHK MMLab + 字节 + HKUST） | MILD（HKU，arXiv:2609.34371） |
|---|---|---|
| 任务 | 少步蒸馏（4 步学生 ← 多步视频教师） | On-policy 蒸馏（不改步数，专精某项能力） |
| 图像教师 | Qwen-Image（默认），FLUX / SDXL 通用 | SD3.5-Medium 等专用图像专家 |
| 桥接方式 | **LatentBridge**：10.75M 参数 FiLM 残差块，输入 $(z^{l-1}, z^l, i)$ 三元组 | **可学习线性连接器**，对齐 latent state 与预测更新量 |
| 防时序破坏 | Latent Variation Sampling（按 latent 变化最大的 K−1 处切段采样）+ ω 权重 | 把图像引导的修正**约束在学生原始预测附近** + 光流运动奖励 |
| 训练成本 | 桥 15 分钟（8×80GB）+ 1000 DMD 步 | 图像教师 OPD：25.80 GiB / 1.83s per query（视频教师为 47.76 GiB / 4.59s） |
| 涨什么 | 视觉质量、语义对齐（动态持平） | 美学、OCR/文字渲染，且保住原有动态 |

两篇的结论高度一致：**图像教师是"免费"的质量来源**，而且查询成本只有视频教师的 40%。差别在于 DuoMatching 要的是"帧级边缘分布"，MILD 要的是"图像专家的专项能力"；两者都必须在两个异质潜空间之间搭桥，这是 E 类的**共同技术瓶颈**。

![](assets/少步视频扩散蒸馏：提升成像质量的六条路线%20-%20Survey/figures/02-duomatching-overview.png)

> E 类工作的完整形态：(a) 目标层面把"联合 + 边缘"两个分布都纳入匹配；(b) 工程层面用 LatentBridge 把视频 VAE 的时间压缩 latent 翻译成图像 VAE 的单帧 latent（这是让边缘监督可行的前提，直接 decode-encode 会 OOM）；(c) 采样层面用 Latent Variation Sampling 把有限的图像教师预算花在变化最大的时刻。三者缺一不可——**只有 (a) 是想法，(b)(c) 才是它能不能跑起来**。来源：DuoMatching Figure 3（arXiv:2610.03543, p.4）。

### 2.6 F 类：系统与工程——不提上限，但决定能否落地

TurboDiffusion（rCM 步蒸馏 + attention/权重量化）、[[LongLive-2.0 - NVFP4长视频生成训练与推理系统]]（NVFP4 低精度训练与推理）、KV cache 调度、attention sink 压缩（EMA-Sink）——这些**不提升质量上限**，但决定"少步模型能不能真的实时跑"。实践上 F 类常与 A–E 任一类叠加：Reinforcement 侧的 EMA-Sink 就是 Reward Forcing 能跑到 23.1 FPS 的前提。

---

## 3. DuoMatching 的近邻：九篇对标

按"像在哪一层"排序：

| 工作 | 相似层级 | 具体关系 |
|---|---|---|
| **Causal Forcing++** | 直接基线 | DuoMatching 从它的 checkpoint 起步、只加 1000 步 DMD；论文全称里的 "Causal Forcing++" 就是对照组 |
| **MILD** | **思路同构** | 图像教师 + 潜空间连接器，见 2.5 表；差别在任务（蒸馏 vs OPD）与防时序破坏的手段 |
| **Reward Forcing** | 同层（改 DMD 梯度） | 都是在 DMD 梯度上做文章，但 Reward Forcing 用**运动奖励**加权（保动态），DuoMatching 用**图像教师**加权（保画质）——互补，不冲突 |
| **Reward Lightning** | 同层（潜空间外部评分） | 都在潜空间里引入外部评分而不 decode 回像素；它用奖励模型，DuoMatching 用生成模型 |
| **Stream-R1** | 同层（训练期重加权） | 都是"只改训练 loss、推理零开销"；它重加权样本，DuoMatching 加新信号 |
| **DMAD** | 同族（DMD 改进） | 同为解决 DMD 的痛点，但 DMAD 解决"fake score 太贵"，DuoMatching 解决"教师覆盖不全"——可叠加 |
| **f-distill / SGMD** | 同族（改散度） | 改的是数学形式，DuoMatching 改的是监督源；理论上可叠加（联合-边缘目标也可以用别的散度） |
| **Data-Forcing Distillation** | 同目标（修少步退化） | 它补的是**多样性与保真度**，靠把真实数据回灌进蒸馏；DuoMatching 补的是**帧级画质**，靠外部教师 |
| **Self-Forcing / [[Self-Forcing++ - 分钟级高质量视频生成]] / [[CausVid - 从慢双向到快自回归视频扩散蒸馏]]** | 同一条主线 | 同属因果视频蒸馏路线，解决的是"训练/推理鸿沟"，与本文的质量问题是上下游关系 |

**最该精读的两篇**：想理解 DuoMatching 的**想法来源**读 [[因果视频蒸馏路线 - Survey]]；想理解它的**工程同构**读 MILD。

---

## 4. 反过来用：症状 → 该借哪路信号

这是本篇最实用的一张表。先诊断自己少步模型的症状，再选路线：

| 症状 | 根因 | 该看哪类 | 首选代表 |
|---|---|---|---|
| 画面糊、缺纹理高频 | score 匹配天然平滑 | **C 对抗** / **E 图像教师** | DMAD、DuoMatching、ADD |
| prompt 细节丢失（颜色/数量/文字渲染） | joint 目标压制单帧改进 | **E 跨模态教师** | DuoMatching、MILD |
| 运动幅度衰减、画面僵住/闪回首帧 | attention sink + 误差累积 | **D 奖励（动态项）** / F EMA-Sink | Reward Forcing、Stream-R1 |
| 过饱和、多样性塌缩 | reverse KL 的 mode-seeking | **B 改散度** / 真实数据回灌 | f-distill（Jeffreys）、Data-Forcing、DP-DMD |
| 结构错乱、残影、几何崩 | 步数太少导致轨迹跟丢 | **A 一致性先打底** | rCM、TMD、MeanFlow |
| 想同时对齐人类偏好又不想掉速 | 对齐与加速梯度冲突 | **D 同源设计** | Reward Lightning（LRM + HPD 共用 backbone） |
| 视频/图像潜空间不兼容导致外部教师用不上 | 时间压缩 + 通道不匹配 | **E 的桥接模块** | LatentBridge、MILD connector |
| 压到 1 步还要能看 | 单步承载太重 | **A + C** | ADD、rCM、DMAD |

**组合建议**：当前最强的少步视频学生基本都是**A/B 打底 + C 或 D 或 E 之一**。DuoMatching 与 DMAD、Reward Forcing 三者理论上正交（分别补画质、省估计器、保动态），叠加是自然的下一步。

---

## 5. 个人判断与开放问题

1. **教师的上限就是学生的上限，除非换监督源。** B 类所有工作（改散度、改估计器）都只是在**同一个信息源**上换算法，天花板被教师锁死；真正突破天花板的是 D（奖励）和 E（跨模态教师）——它们引入了教师不知道的信息。这是我认为 2026 年 D/E 两类明显比 B 类涨点更实在的根本原因。

2. **潜空间桥接会成为标准件。** 凡是"借外部模型监督"的方案，第一道关卡都是潜空间不兼容（视频 VAE 的 4× 时间压缩 + 通道数不同）。LatentBridge 和 MILD 的线性连接器已经证明这件事可以用 10M 级参数、十几分钟训完解决。我预期会出现**通用的开源 latent 翻译器**，就像当年 CLIP 的投影层一样被复用。

3. **动态与画质仍未同时解决。** DuoMatching 补偿了画质但动态持平；Reward Forcing 保住了动态但画质靠别处。这两条线在同一个目标里还是互斥的——Reward Lightning 的"同源"设计是目前最有希望的解法。

4. **评估已经接近饱和。** VBench 上 84.40 vs 84.26 这类差距基本在噪声内，论文们纷纷转向 60 秒长视频 + 50 条视频的人工评测，样本量太小、结论偏软。这个领域急需一个**针对少步退化的专项评测**（专门测高频、测长时间一致性、测 prompt 细节命中）。

5. **复现门槛被低估。** 双教师 + 双 fake 估计器 + 桥接模块，8×80GB 起步，还要先有一个 Causal Forcing++ checkpoint。这意味着 E/D 类工作的复现成本远高于 A/B 类，也解释了为什么论文之间很难做严格对照。

---

## 6. 附件与溯源

| 文件 | 内容 | 来源 |
|---|---|---|
| `figures/01-duomatching-teaser-quality-gap.png` | 少步学生的质量缺口（含首帧对比与 Q-Align 曲线） | DuoMatching Figure 1，arXiv:2610.03543 p.2 |
| `figures/02-duomatching-overview.png` | 联合-边缘分布匹配 + LatentBridge + Latent Variation Sampling 三组件总览 | DuoMatching Figure 3，arXiv:2610.03543 p.4 |
| `figures/03-joint-vs-marginal-signal.png` | Joint vs Marginal matching signal 的空间结构差异 | DuoMatching Figure 2，arXiv:2610.03543 p.2 |

完整 URL、获取日期与用途见 `assets/少步视频扩散蒸馏：提升成像质量的六条路线 - Survey/sources.md`。

## 7. 相关笔记

- 纵向主线：[[因果视频蒸馏路线 - Survey]]、[[AR扩散与因果视频生成：五篇方法对比 - Survey]]
- 领域全景：[[视频生成模型技术谱系与2026现状 - Survey]]、[[视频生成模型发展与架构演进 - Survey]]
- E 类主角精读：[[DuoMatching - 联合边缘分布匹配的少步视频生成]]
- 主线论文：[[CausVid - 从慢双向到快自回归视频扩散蒸馏]]、[[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]]、[[Self-Forcing++ - 分钟级高质量视频生成]]、[[OmniForcing - 实时音视频联合流式生成]]、[[Causal Forcing - 正确的自回归扩散蒸馏]]
- 基座与系统：[[Wan2.1 - 开源大规模视频生成基础模型]]、[[LongLive-2.0 - NVFP4长视频生成训练与推理系统]]、[[Goku - 联合图像视频整流流基础模型]]
