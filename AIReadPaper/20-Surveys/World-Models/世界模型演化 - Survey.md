# 世界模型演化 - Survey：从 RSSM 到世界基础模型

> **结论先看：** 这篇综述把博主「忍nin」World Model 日更系列里逐天拆解的一组经典论文，串成一条从 2018 到 2025 的演化主线。它不是某篇单论文的精读，而是一次「问题演化」梳理：世界模型（World Model）如何在七年里，从一个「给深度强化学习提样本效率的技巧」，一步步演变成「物理 AI 时代的基础设施级生成模型」。
>
> 九个节点分别是：**PlaNet**（2018，RSSM 潜空间规划）、**MBPO**（2019，模型何时可信）、**Objective Mismatch**（2020，可信模型未必可控）、**Plan2Explore**（2020，把探索变成规划）、**TD-MPC**（2022，时序差分 + MPC 合流）、**IRIS**（2023，Transformer 取代 RNN）、**LINGO-2**（2024，语言可解释的闭环驾驶）、**GAIA-2**（2025，多摄像头视频生成）、**Cosmos**（2025，世界基础模型平台）。
>
> 把它读成一条线，最重要的三个观察是：
>
> 1. **表示一直在换，但「潜空间预测」这个核心承诺没变。** 从 RSSM 的连续潜状态，到 IRIS 的离散 token，再到 Cosmos 的视频 tokenizer，承载世界知识的表示从「低维、确定性」一路走向「高维、可生成」，但「先编码、再在表征里推演未来」这件事始终是主轴。
> 2. **目标函数经历了三次「再对齐」。** PlaNet 用变分下界对齐「重建」，TD-MPC 用 TD-learning 对齐「回报」，Plan2Explore 用 disagreement 对齐「探索」——每一次都是对「到底该让模型优化什么」的重新回答。
> 3. **出口从游戏台一路走到实车与平台。** PlaNet/Plan2Explore/TD-MPC 在 DMControl 游戏台，IRIS 在 Atari，LINGO-2/GAIA-2 上了公开道路，Cosmos 则把世界模型打包成可 post-training 的基础设施。用途从「RL 内部模拟器」改写为「合成数据 / 边缘场景生成器」。

**总体评级：★★★★★（5/5）**——作为「世界模型方向的地图册 + 演化史」，这组论文的价值在于它把一条最核心的技术血脉讲透了；扣分点只在它是一条「回溯式」脉络，不覆盖 JEPA、视频扩散世界模型等 2024 之后的平行主线（那些留给 [[视觉世界模型 - Survey]] 去展开）。

## 1. 研究背景与演化主线

### 1.1 为什么要串这九篇

博主「忍nin」的 World Model 日更系列，本质上是一次「按时间顺序重读世界模型经典」的连载。它挑出来的不是同一个年代、同一个阵营的工作，而是跨越 7 年、分属不同目标函数与不同应用域的一组节点。单独读每一篇，看到的是孤立的方法；把它们按发表顺序排开，才会浮现出那条真正的演化线索——**世界模型内部一直存在几条相互拉扯的张力，每一篇经典都在解决其中一条，又为下一篇留下新的问题。**

这正是写这篇综述的动机：不是复述九篇论文的摘要，而是回答「这九篇放在一起，到底讲了一个什么故事」。

### 1.2 一条七年时间线

| 年份 | 工作 | 表示 | 目标函数 | 出口 / 应用 |
|---|---|---|---|---|
| 2018 | PlaNet | RSSM 连续潜状态 | 变分下界（重建） | DMControl 游戏台 |
| 2019 | MBPO | 集成神经网络 | 似然 + 误差界 | MuJoCo 连续控制 |
| 2020 | Objective Mismatch | —（分析框架） | one-step 似然 vs reward | 诊断工具 |
| 2020 | Plan2Explore | Dreamer 系潜状态 | 自监督 disagreement | DMControl 探索 |
| 2022 | TD-MPC | task-oriented 潜状态 | TD-learning（回报） | DMControl + Meta-World |
| 2023 | IRIS | 离散 token（VQ） | next-token 似然 | Atari 100k |
| 2024 | LINGO-2 | 视觉 token + 语言 | 自回归联合似然 | 公开道路闭环驾驶 |
| 2025 | GAIA-2 | 视频 tokenizer + 潜扩散 | 扩散去噪 | 自动驾驶合成数据 |
| 2025 | Cosmos | 连续/离散 tokenizer | 扩散 + 自回归 | 物理 AI 基础平台 |

### 1.3 本文的四条叙事主线

后续各节会围绕这四条线交替推进，这也是理解整条脉络的钥匙：

- **表示线**：连续潜状态 → 离散 token → 大规模视频 tokenizer；
- **信任线**：模型误差从何而来（MBPO）、误差最小化为何不等于控制最优化（Objective Mismatch）；
- **目标线**：重建（PlaNet）→ 控制（TD-MPC）→ 探索（Plan2Explore）→ 平台（Cosmos）；
- **出口线**：游戏台 → 实车 → 基础设施。

## 2. PlaNet（2018）：把像素压缩进潜空间

**论文：** Learning Latent Dynamics for Planning from Pixels（Hafner et al.，arXiv:1811.04551）

PlaNet 是整个「从像素学世界模型」路线的共同祖先。在它之前，model-based RL 大多要么学一个「输出下一帧像素」的模型（误差巨大、难以规划），要么学一个「只给价值/策略」的黑盒。PlaNet 的贡献是把这两者拆开：**只学一个紧凑的潜空间动力学模型，规划完全在这个潜空间里做。**

### 2.1 RSSM：确定性记忆 + 随机潜状态

PlaNet 的核心是 **Recurrent State-Space Model（RSSM）**。它把世界状态拆成两条路径：

- **确定性路径** $h_t$：由 RNN 维持的长期记忆，负责「确定性、可预测」的那部分演化；
- **随机路径** $s_t$：概率潜状态，负责「多模态未来」——同一历史观测可能对应多种未来。

两者关系可以写成：

$$h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$$

$$s_t \sim p(s_t \mid h_t)$$

观测与回报则由潜状态解出：$p(o_t \mid s_t)$、$p(r_t \mid s_t)$。

这个「确定性记忆 + 随机潜状态」的二分，几乎成了后来所有 world model 的模板：Dreamer 系、TD-MPC、以及大量后续工作都在复用或改造它。它解决了纯 RNN 的一个老问题——纯确定性状态在随机环境下无法表达多种可能未来，而纯随机状态又难以记忆长期依赖。

### 2.2 latent overshooting：把梯度连到多步未来

单步预测训练出来的模型，做多步 rollout 时误差会累积放大。PlaNet 提出 **latent overshooting**：一个多步变分下界，让模型在训练时就直接预测未来 $d$ 步（$d=0\dots H$），梯度从多步未来反传回来，迫使潜状态学会「为长期预测服务」而不是「只为了拟合下一帧」。

这一步在概念上非常重要：它第一次把「训练目标」和「规划视界」对齐了——你打算用它规划多少步，就让它学着预测多少步。

### 2.3 在潜空间里规划

给定学好的潜动力学，PlaNet 用 **Cross-Entropy Method（CEM）** 在潜空间里做短视界轨迹优化：采样一批动作序列，用潜模型 rollout 出未来，选 reward 最高的一批重采样迭代，执行第一个动作。因为规划发生在低维潜空间而非像素空间，单次 rollout 极快，可以在线做 MPC 式规划。

### 2.4 结果与意义

PlaNet 在 DMControl 的 6 个连续控制任务上，用 **远少于 model-free 方法的样本** 达到了可比或更好的表现（论文强调约 **5000% 的样本效率提升**，且在单个 GPU 上数小时即可训完）。更重要的是它证明了「潜空间规划」这条路线可行——不需要重建像素，也能做好控制。

下图是 PlaNet 的 RSSM 结构示意，确定性/随机两条路径的分工是理解后续所有工作的起点：

![PlaNet RSSM 架构](assets/世界模型演化%20-%20Survey/figures/01-planet-rssm-architecture.png)

*（图：PlaNet 的 RSSM 结构，确定性路径 $h$ 与随机路径 $s$ 分工。来源：arXiv:1811.04551 论文配图）*

**这一节的遗留问题（留给下一篇）：** PlaNet 证明了「潜空间规划可行」，但没有回答「这个潜模型到底能不能信、误差会在什么时候爆炸」。这正是 MBPO 要接的问题。

## 3. MBPO（2019）：模型何时可信

**论文：** When to Trust Your Model: Model-Based Policy Optimization（Janner et al.，arXiv:1906.08253）

MBPO 是 model-based RL 里第一份「把『模型误差』当成一个可分析、可度量、可控制的对象」的工作。它回答的核心问题是：**模型误差在 rollout 时是如何放大的，以及我们该如何在「用模型」和「用真实环境」之间做取舍。**

### 3.1 误差累积与分布漂移

MBPO 把 model-based RL 的误差拆成两部分，并给出一个有上界的分析：

- **泛化误差**：模型在训练分布之外必然变差，而 rollout 越深，模型离训练分布越远；
- **策略漂移（policy shift）**：随着 rollout 变长，模型隐式生成的「策略访问分布」与真实分布之间的 KL 散度会以指数级增长。

合起来，rollout 长度为 $k$ 时的累计误差可以近似上界为 $O(\epsilon_m \cdot k^2)$ 量级（随深度二次增长），这也是「长 rollout 不可靠」的形式化来源。结论很直白：**模型只在你「信任半径」内可靠，一旦 rollout 过长，误差会盖过收益。**

下图是 MBPO 的经典示意：模型误差随 rollout 长度增长，而策略性能在某个「最优 rollout 长度」处见顶——太短浪费样本，太长被误差反噬。

![MBPO 模型误差与 rollout 长度](assets/世界模型演化%20-%20Survey/figures/02-mbpo-model-error.png)

*（图：模型误差随 rollout 长度增长、策略回报在某中间长度见顶的示意。来源：arXiv:1906.08253 论文配图）*

### 3.2 branched rollouts 与模型集成

基于上面的分析，MBPO 给出两个工程手段：

- **短视界 branched rollouts**：从真实状态出发，只往前 rollout $k$ 步就回到真实环境采样，而不是无限往前推。$k$ 是一个要调的「信任长度」。
- **模型集成（ensemble）**：训练一组概率神经网络模型，rollout 时随机选一个，用集成不确定性对冲单模型偏差。

这样，MBPO 拿到了一个「中间地带」：比纯 model-based 更稳（因为 rollout 短、有集成），比纯 model-free 更省样本（因为大部分「经验」来自廉价模型）。

### 3.3 结果与意义

MBPO 在 Hopper、Walker2d、HalfCheetah、Ant、Humanoid 等 MuJoCo 连续控制任务上，用比 SAC 少一个数量级的样本达到持平或更好的回报。它的意义不在「刷榜」，而在**把「何时信任模型」从一个经验直觉变成了一个可分析、可优化的对象**，成为后续 STEVE、PPME 等一系列混合路线的基准对照。

**这一节的遗留问题（留给下一篇）：** MBPO 假设「把模型误差最小化」就自然对控制有利。但 Objective Mismatch 会指出——**这个假设本身就是错的。**

## 4. Objective Mismatch（2020）：可信的模型未必可控

**论文：** Objective Mismatch in Model-based Reinforcement Learning（Lambert et al.，L4DC 2020，arXiv:2002.04523）

这篇是九篇里唯一的「分析性/批判性」工作，但它对整个领域的影响不亚于任何一篇新模型。它指出 model-based RL 框架里一个根本性的、被长期忽视的缺陷：**目标错配（objective mismatch）。**

### 4.1 one-step 似然与 reward 控制的错配

标准 MBRL 里，动力学模型是用 **one-step 似然**（最小化下一帧预测误差）训练的；但它真正被使用的方式，是在 rollout 里为**下游 reward / 控制**服务。这两件事经常不相关：

- 模型擅长预测「平均意义下最可能」的下一状态；
- 但控制真正关心的是「哪个状态对策略价值最敏感」的那部分差异。

当这两者错位时，就会出现「模型在全局指标上越来越准，但用它做的控制却越来越差」的反直觉现象。

### 4.2 反直觉现象：更准的模型反而更差

论文用受控实验证明了这一点：在某些环境里，**一个「全局预测误差更小」的动力学模型，做出来的策略回报反而更低**。原因在于，one-step 似然会把模型容量平均分配到「对控制无关紧要」的状态维度上，而把「对控制关键」的维度挤占掉。

下图是 objective mismatch 的核心示意：训练目标（拟合下一帧）与使用目标（最大化回报）指向不同的方向，中间的「mismatch」就是误差的来源。

![Objective Mismatch 示意](assets/世界模型演化%20-%20Survey/figures/03-obj-mismatch-illustration.png)

*（图：objective mismatch 的概念示意——训练目标与使用目标不一致。来源：arXiv:2002.04523 论文配图）*

### 4.3 缓解与后续影响

论文给出一个初步缓解方案：对训练样本做**重加权**（更关注对控制重要的区域）。但它的真正价值在于**把 mismatch 命名为一个必须正视的基础问题**——此后任何 MBRL 工作都不能再默认「学好动力学 = 做好控制」。这一思想也直接或间接影响了 TD-MPC 的思路：后者干脆把目标函数直接对齐到回报，绕开了这个错配。

## 5. Plan2Explore（2020）：把探索变成规划

**论文：** Planning to Explore via Self-Supervised World Models（Sekar et al.，ICML 2020，arXiv:2005.05960）

Plan2Explore 回答的是一个不同的问题：**在没有奖励信号时，怎么训练一个「任务无关」的世界模型，让它之后能快速适配任意下游任务？**

### 5.1 从 novelty 到规划

传统自监督探索的思路是「事后判断新不新」：agent 走一步，再看刚到达的观测有没有见过。Plan2Explore 把这件事**倒过来**——它在世界模型里**规划去追求「未来的新颖性」**：维护一个对下一状态预测有 disagreement 的集成，然后让 agent 规划出一条「能最大化未来 disagreement」的动作序列，主动去往它最不了解的区域。

这本质上是用 PlaNet/Dreamer 那套「潜空间规划」的机械结构，去执行一个「探索」目标——**探索不再是探索策略的事，而是规划目标的事。**

下图是 Plan2Explore 的范式总览：一次无奖励的探索，得到一个 global 世界模型，再零样本/少样本适配到多个下游任务。

![Plan2Explore 范式](assets/世界模型演化%20-%20Survey/figures/04-plan2explore-paradigm.png)

*（图：Plan2Explore 的「探索 → 通用世界模型 → 下游适配」范式。来源：arXiv:2005.05960 论文配图）*

### 5.2 任务无关的世界模型

关键洞察是：探索阶段**不接触任何任务奖励**，只靠自监督 disagreement 信号学出一个 global world model。这个模型是「任务无关」的——它对世界本身的动力学理解，不绑定任何具体奖励函数。

### 5.3 zero/few-shot 适配

一次无奖励探索得到的 global world model，可以被零样本或少量样本适配到 standing、walking、running、flipping 等多个 DMControl 下游任务。这是世界模型从「任务专用」走向「任务无关基础模型」的关键一步——它的思想（一个通用世界模型 + 轻量下游适配）在 Cosmos 时代被放大成了「world foundation model」。

## 6. TD-MPC（2022）：时序差分与 MPC 的合流

**论文：** Temporal Difference Learning for Model Predictive Control（Hansen et al.，ICML 2022，arXiv:2203.04955）

TD-MPC 是 Dreamer 系与 model-predictive control 的一次「优雅杂交」。它要解决的核心矛盾是：**PlaNet/Dreamer 的潜模型学得好，但目标函数（重建）与控制目标（回报）不一致；而 model-free 的价值函数对齐了回报，却没有「想未来」的能力。**

### 6.1 task-oriented latent dynamics

TD-MPC 学一个 **task-oriented 潜动力学模型（TOLD）**：潜状态不再是「通用世界状态」，而是**针对当前任务优化过的**潜表示。它同时学一个终端价值函数（Q-function），用来做长视界 return 估计。

规划时：用 CEM 在短视界做**局部轨迹优化**（靠潜模型 rollout），再叠加**价值函数对终端状态的 return 估计**（靠 TD-learning 学到的 Q），两者合起来完成「短视界精度 + 长视界引导」。

### 6.2 单一损失函数

TD-MPC 的一个标志性设计是：把潜动力学、价值函数、策略、回报预测全部塞进**一个统一的损失函数**里（TD-MPC objective），通过 TD-learning 联合训练。这正好回应了 Objective Mismatch 的批评——它不再「先学动力学、再外挂控制」，而是**让表示学习直接服务于控制目标**。

下图是 TD-MPC 的方法总览，展示潜状态、价值函数与 MPC 规划如何协同：

![TD-MPC 方法总览](assets/世界模型演化%20-%20Survey/figures/05-tdmpc-overview.png)

*（图：TD-MPC 的 task-oriented 潜动力学 + 价值函数 + MPC 协同示意。来源：arXiv:2203.04955 论文配图）*

### 6.3 结果与意义

TD-MPC 在 DMControl 与 Meta-World 上同时拿到了 state 输入和 image 输入的 SOTA，是当年连续控制最亮眼的世界模型。其后续 TD-MPC2 进一步扩到 80+ 任务，用单一算法覆盖多环境。它的意义在于：**证明了「把目标函数对齐到回报」是比「把重建做准」更正确的方向。**

**这一节的遗留问题（留给下一篇）：** TD-MPC 仍然是 RNN 系（连续潜状态）。但 2022 之后，序列建模的皇冠正在从 RNN 移向 Transformer。IRIS 就是把这个皇冠戴到世界模型头上的那篇。

## 7. IRIS（2023）：Transformer 取代 RNN

**论文：** Transformers are Sample-Efficient World Models（Micheli et al.，ICLR 2023，arXiv:2209.00588）

IRIS 做了一件「范式切换」的事：把「在想象中训练」的世界模型，直接从 RNN 语言重写成了 **Transformer 语言**。

### 7.1 离散 token 化

IRIS 用一个**离散自编码器（VQ-VAE 风格）** 把图像压缩成一串离散 token（一个小的「图像词汇表」）。这一步把连续像素问题转成了离散序列问题——从而能直接套用语言建模的那套成熟 Transformer 基础设施。

### 7.2 在想象中训练

世界模型本身是一个 **GPT 风格的自回归 Transformer**：给定过去的 token 序列和动作，预测下一个 token。policy 和 critic 则**完全在 latent token 轨迹上学习**，从不接触真实像素。整个训练流程是：

1. 用离散自编码器把真实观测 token 化；
2. 用 Transformer 世界模型在 token 空间里 rollout 出「想象轨迹」；
3. policy/critic 在这些想象轨迹上训练。

下图是 IRIS 的「想象中训练」示意：世界模型（Transformer）生成 latent 轨迹，policy 在其上学习。

![IRIS 想象中训练](assets/世界模型演化%20-%20Survey/figures/06-iris-imagination.png)

*（图：IRIS 的「世界模型在 latent 空间想象 + policy 在想象中学习」示意。来源：arXiv:2209.00588 论文配图）*

### 7.3 Atari 100k 样本效率

在 Atari 100k 基准（每局仅 **10 万环境步、约 2 小时真实交互**）上，IRIS 超越了同设置下的 DreamerV2、EfficientZero、MuZero 等，并在 26 款游戏中的 **10 款** 超过人类水平。这个结果的意义在于：**它证明了 Transformer 世界模型不仅可行，而且在极低样本下更优**——把世界模型范式从 RNN 正式推向了 Transformer（后来 DreamerV3 也转向了类似的离散/符号化预测）。

## 8. 走向真实世界：LINGO-2 与 GAIA-2（2024-2025）

从这一节开始，世界模型走出游戏台，进入 Wayve 的自动驾驶实车场景。这两篇都不是 arXiv 论文，而是 Wayve 官方技术博客，但它们代表世界模型「用途」的一次根本改写。

### 8.1 LINGO-2：语言可解释的闭环驾驶世界模型

**来源：** Wayve 官方博客《LINGO-2: Driving with Natural Language》（2024-10）

LINGO-2 是首个在**公开道路上测试的闭环 vision-language-action 驾驶模型（VLAM）**。它的架构思路与 IRIS 一脉相承：Wayve 的视觉模型把连续摄像头图像编码成 token，一个自回归语言模型同时预测两样东西——**驾驶轨迹（动作 token）** 和 **自然语言解释（"前方有行人，我正在减速"）**。

这让世界模型第一次获得了两个此前没有的性质：

- **可解释**：模型能「说出」它在做什么、为什么这么做；
- **可控**：能用自然语言指令实时调整驾驶行为（"靠边停车"、"右转"）。

下图是 LINGO-2 的架构示意：视觉 token 输入，语言模型同时输出轨迹与解释。

![LINGO-2 架构](assets/世界模型演化%20-%20Survey/figures/08-lingo2-architecture.png)

*（图：LINGO-2 的视觉 token → 自回归语言模型 → 轨迹 + 语言解释。来源：Wayve LINGO-2 官方博客）*

LINGO-2 的意义在于：**把「语言」接进了世界模型的输入输出接口**，让世界模型从「纯内部模拟器」变成「可以被自然语言控制、也能用自然语言解释」的系统。这为后来的 VLAM / VLA 路线铺了路。

### 8.2 GAIA-2：多摄像头视频生成世界模型

**来源：** Wayve 官方博客《GAIA-2: Pushing the Boundaries of Video Generative Models for Safer Assisted and Automated Driving》（2025-03）

GAIA-2 是 GAIA-1 之后的第二代**视频生成世界模型**。它用「视频 tokenizer + 潜扩散」生成**多摄像头时空一致**的驾驶视频，并支持细粒度可控生成：

- 自车动作（ego action）；
- 动态主体（车辆、行人）；
- 环境因素（天气、光照、时间）；
- 道路属性（车道线、红绿灯）。

下图是 GAIA-2 的封面/效果示意，展示多摄像头一致的可控驾驶视频生成。

![GAIA-2 封面](assets/世界模型演化%20-%20Survey/figures/09-gaia2-cover.png)

*（图：GAIA-2 多摄像头时空一致的可控驾驶视频生成示意。来源：Wayve GAIA-2 官方博客）*

GAIA-2 的意义在于：**它把世界模型的「用途」从 RL 内部模拟器，改写为自动驾驶系统的「合成数据 + 边缘场景生成器」。** 世界模型不再只是「帮策略做规划」，而是直接产出训练数据、生成极端场景来压测下游系统。这个定位的转变，直接预告了 Cosmos 的「平台化」野心。

## 9. Cosmos（2025）：世界基础模型平台化

**论文：** Cosmos World Foundation Model Platform for Physical AI（NVIDIA，arXiv:2501.03575）

Cosmos 是这条演化线的「终章」——它把「世界模型」从一个 RL 子领域，正式抬升为**物理 AI 时代的基础设施级生成模型平台**。九篇里它体量最大，也最「工业」。

### 9.1 数据治理与 Tokenizer

Cosmos 首先把「数据」当作第一性问题：用 **Ray 流水线** 治理 **2000 万小时** 的视频数据，做清洗、去重、标注。配套的 **Cosmos Tokenizer** 提供**连续**和**离散**两种形态，连续 tokenizer 保细节（适合扩散），离散 tokenizer 保压缩（适合自回归）——这正好呼应了前面「连续潜 vs 离散 token」那条表示线的合流。

### 9.2 扩散与自回归 WFM 家族

Cosmos 同时提供两类世界基础模型（WFM）：

- **扩散 WFM**：7B / 14B 参数，latent diffusion 路线；
- **自回归 WFM**：4B / 12B 参数，next-token 路线。

下图是 Cosmos WFM 的视频生成结果示意，展示其在具身场景下的可生成能力：

![Cosmos WFM 结果](assets/世界模型演化%20-%20Survey/figures/07-cosmos-wfm-results.png)

*（图：Cosmos 世界基础模型的视频生成结果示意。来源：arXiv:2501.03575 论文配图）*

### 9.3 具身 post-training

Cosmos 把前面八篇积累的所有「下游用法」统一成了 **post-training**：在预训练 WFM 之上，针对 **Camera Control、Robotic Manipulation、Autonomous Driving** 三个具身任务做后训练。这本质上把 Plan2Explore「任务无关世界模型 + 下游适配」的思想放大到了产业尺度。

## 10. 演化脉络总结：三条贯穿的张力

把九篇放回时间线，能看到三条贯穿始终的张力，每一条都在驱动「下一篇」出现：

### 10.1 表示线：连续潜 → 离散 token → 大规模生成

- PlaNet 的 RSSM 是**连续、低维、确定性主导**的潜状态；
- IRIS 用 VQ 把它变成**离散 token**，接上 Transformer 基础设施；
- Cosmos 的 tokenizer 同时提供连续/离散，回到**大规模视频生成**。

这条线说明：**「用什么样的表示承载世界知识」一直在变，但「先编码、再在表征里推演」这个承诺始终没变。** 表示的每次升级，都是为了接入更强的基础设施（Transformer、扩散、视频语料）。

### 10.2 目标线：重建 → 控制 → 探索 → 平台

- PlaNet 用**变分下界对齐重建**；
- Objective Mismatch 指出重建 ≠ 控制，TD-MPC 用 **TD-learning 对齐回报**；
- Plan2Explore 用 **disagreement 对齐探索**；
- Cosmos 用**预训练 + post-training 对齐「平台」**。

这条线说明：世界模型的核心争议从来不是「要不要潜空间」，而是「**到底该让模型优化什么**」。每一次目标函数的再对齐，都回应了前一篇留下的 mismatch。

### 10.3 出口线：游戏台 → 实车 → 基础设施

- PlaNet / Plan2Explore / TD-MPC 在 **DMControl 游戏台**；
- IRIS 在 **Atari**；
- LINGO-2 / GAIA-2 上了**公开道路**；
- Cosmos 变成**物理 AI 基础设施**。

这条线说明：世界模型的「用途」发生了质变——从「给 RL 提样本效率的内部模拟器」，到「自动驾驶的合成数据生成器」，再到「通用具身智能的基础模型」。用途的改写，反过来又定义了「什么才算好世界模型」。

## 11. 局限与评价

1. **这是一条「回溯式」主线，不覆盖 2024 之后的平行世界。** 九篇里没有 JEPA（embedding 预测）这条线，也没有 2024 之后爆炸式增长的视频扩散世界模型（Sora、Genie 2/3、Veo 等）。它们与「潜空间规划」是平行的另一条血脉，需要 [[视觉世界模型 - Survey]] 来补全地图。
2. **「世界模型」一词在九篇里指代不同。** PlaNet 的世界模型是「潜空间动力学」，GAIA-2/Cosmos 的世界模型是「视频生成器」。串线时要注意：它们共享「预测未来」的表层，但目标函数、表示、用途差异巨大——这正是第 10 节三条张力存在的原因。
3. **早期工作（2018-2022）的游戏台结论，不能直接外推到实车/具身。** DMControl / Atari 是低维、无视觉杂波、奖励密集的环境；实车与真实机器人则有视觉噪声、稀疏奖励、安全约束等全新挑战。LINGO-2/GAIA-2 的价值在于把战场搬过去，但它们本身是工业博客，缺少可复现的实验协议。
4. **客观度问题。** Cosmos 是 NVIDIA 的工业平台，LINGO-2/GAIA-2 是 Wayve 的官方博客，三者都带有明显的产品宣发属性，其「效果」缺乏独立第三方复现。

## 12. 个人判断与下一步

### 12.1 我对这条脉络的判断

把九篇串起来，我最想强调的一个判断是：**世界模型的「潜空间预测」承诺是 7 年不变的，但「为什么预测」的答案变了三次。** 早期是「为了省样本做规划」，中期是「为了对齐控制/探索目标」，晚期是「为了生成合成数据与边缘场景」。谁抓住了「用途」的变迁，谁就抓住了这条线的主线。

另一个判断：**Objective Mismatch 是这九篇里最被低估的一篇。** 它没有提出新模型，但它点破的「训练目标 ≠ 使用目标」问题，在今天的大模型时代（预训练目标 vs 下游对齐）依然成立，甚至更尖锐。读这一篇，比读十篇刷榜模型更能建立正确的判断框架。

### 12.2 建议后续跟踪的分叉

- **JEPA / 非重建路线**：本 vault 已有 [[LeWorldModel - 像素端到端稳定JEPA世界模型]]、[[LeJEPA - 世界模型线性可识别性理论]]，是「拒绝像素重建」这条线的理论+实践锚点，与本文的「重建/生成」主线正好互补。
- **世界动作模型（WAM）**：[[Dyna-2 - 世界动作模型百万小时缩放定律]]、[[Fast-WAM - 无测试时未来想象的世界动作模型]]、[[DreamDojo - 基于大规模人类视频的通用机器人世界模型]] 是「视频世界模型 + 动作」的最新工业脉络，可视为 GAIA-2/Cosmos 路线的直接延续。
- **显式几何世界模型**：[[GaussianDream - 机器人操控的前馈三维高斯世界模型]] 代表「世界状态 = 几何而非视频」的另一支，与本文全部九篇的「潜/像素」表示形成对照。
- **系统化版图**：[[视觉世界模型 - Survey]] 是 2026 年的总纲式综述，可把本文的「演化史」放进它的「Vision Encoding → Knowledge Learning → Controllable Simulation」框架里做二次定位。

### 12.3 一句话总结

这九篇串起来，讲的是同一件事的七次进化：**一个「在表征里预测未来」的朴素想法，如何在「表示、信任、目标、出口」四条张力的拉扯下，从 2018 年的游戏台潜空间规划，一路演变成 2025 年的物理 AI 基础设施。** 理解这条线，你就理解了世界模型领域一半的骨架；剩下的一半，在 JEPA 与视频扩散这两条平行血脉里。

## 参考与来源

九篇论文的官方来源与关键图出处，详见 `assets/世界模型演化 - Survey/sources.md`。各篇 arXiv/官方链接也已在 [[Paper-Index]] 的「世界模型」域逐条登记。
