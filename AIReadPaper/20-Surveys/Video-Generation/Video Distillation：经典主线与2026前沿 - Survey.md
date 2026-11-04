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
tags:
  - Video-Generation
  - Survey
  - Distillation
  - DMD
  - Diffusion-Forcing
  - Causal-Distillation
  - Streaming-Video
  - Few-Step
  - DuoMatching
created: 2026-11-03
updated: 2026-11-03
---

# Video Distillation：从 DMD 与 Forcing 范式到 2026 前沿

> **结论先看：** 截图把 DMD、Diffusion Forcing、CausVid、Self Forcing、Causal Forcing 与 Sparse Forcing 连成“视频蒸馏组件”，方向是对的，但它们并不处在同一层。**DMD 是分布蒸馏目标，Diffusion Forcing 是因果序列训练范式，CausVid / Causal Forcing / Causal Forcing++ 主要解决双向教师到因果少步学生的初始化与结构转换，Self Forcing / Self-Forcing++ 解决 self-rollout 的训练—推理对齐和长 horizon，Sparse Forcing 则解决注意力与 KV-cache 系统瓶颈。**
>
> **截至 2026 年 11 月，video distillation 的主矛盾已经不再只是“50 步怎么压到 4 步”**。真正决定结果的，是六类鸿沟能否同时闭合：采样步数、师生可见信息、历史分布、教师质量上限、跨 chunk 信用分配、以及长视频的注意力/KV 系统成本。最新方法 DuoMatching、DMAD、Leta、SF²、SGF+ 等，分别从多教师、密度比估计、未来信用、隐藏表征和角色解耦切入。
>
> **总体评级：★★★★★（5/5）**。这是 2024–2026 年视频生成里最明确的一条方法学收敛路线：少步扩散正在和 decoder-only 式因果生成、on-policy 训练、长记忆系统以及多教师质量监督合流。

## 基本信息

**报告主题**：视频扩散/流模型的少步蒸馏与因果自回归生成，覆盖基础目标、初始化、self-rollout、长视频训练、DMD 改进和系统优化。

**覆盖时间**：2023.11（DMD）— 2026.10（DMAD、DuoMatching、SGF+、Leta、SF²）。

**截图文章核查结果**：

| 截图中的条目 | 已有独立精读 | 说明 |
|---|---|---|
| DMD | **有**：[[DMD - 分布匹配蒸馏一步图像生成]]；另有 [[DMD2 - 改进的分布匹配蒸馏快速图像合成]]、[[SiD - 分数恒等蒸馏一步生成]] | 图像侧上游工具箱，不是专门的视频方法 |
| Diffusion Forcing | **有**：[[Diffusion Forcing - 下一Token预测与全序列扩散]] | 训练范式，不是教师—学生蒸馏损失 |
| CausVid | **有**：[[CausVid - 从慢双向到快自回归视频扩散蒸馏]] | 双向教师→因果学生的路线起点 |
| Self Forcing | **有**：[[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]] | 训练时执行学生自己的 rollout |
| Self Forcing++ | **有**：[[Self-Forcing++ - 分钟级高质量视频生成]] | 把 self-rollout 与 DMD 扩展到分钟级 |
| “Casual Forcing” | **应为 Causal Forcing；已有**：[[Causal Forcing - 正确的自回归扩散蒸馏]] | 修复双向 ODE teacher 与因果 student 的目标错配 |
| Causal Forcing++ | **此前没有独立精读**；本综述新增专节 | 用 causal CD 取代昂贵的完整 causal ODE 轨迹库 |
| Sparse Forcing | **此前没有独立精读**；本综述新增专节 | 原生可训练稀疏注意力与持久 KV 记忆 |
| DuoMatching | **有**：[[DuoMatching - 联合边缘分布匹配的少步视频生成]] | 2026.10 的 joint-marginal 双教师路线 |

**与既有综述的分工**：

- [[因果视频蒸馏路线 - Survey]]：聚焦 DMD→CausVid→Self-Forcing 的纵向主线，写于 Causal Forcing++、Sparse Forcing、DuoMatching 等出现之前。
- [[AR扩散与因果视频生成：五篇方法对比 - Survey]]：重点比较因果建模范式。
- [[少步视频扩散蒸馏：提升成像质量的六条路线 - Survey]]：以“质量信号从哪里来”为轴，重点落在 DuoMatching。
- **本篇**：把三篇既有综述合并到一个 2026-11 快照中，按“六类鸿沟”统一解释经典主线与最新工作。

![](assets/Video%20Distillation：经典主线与2026前沿%20-%20Survey/figures/00-evolution-map.svg)

> **图 1：Video Distillation 技术演进图（综述自绘）。** 上游有三套不同工具：DMD 管输出分布，Diffusion Forcing 管因果时序的加噪训练，一致性/轨迹蒸馏管 flow map。CausVid 把它们汇入因果视频蒸馏；Self Forcing 与 Causal Forcing 分别修历史分布和初始化结构；2026 年的新工作再向质量监督、信用分配和稀疏系统三个方向展开。来源：本综述根据各论文官方方法自绘。

---

## 1. 先把概念分层：并非所有 Forcing 都是蒸馏损失

### 1.1 五个技术层次

| 层次 | 问题 | 代表工作 | 典型输出 |
|---|---|---|---|
| **蒸馏目标** | 学生分布向哪里更新 | DMD/DMD2/SiD、DMAD、DuoMatching、DFD | score difference、密度比、奖励/边缘监督 |
| **序列训练范式** | 每个时序 token 如何加噪与可见 | Diffusion Forcing、Teacher Forcing | 因果 diffusion teacher |
| **少步初始化** | 1–4 步学生怎样先学会近似 flow map | ODE distillation、causal ODE、causal CD、TF-CM、PTD | 可工作的少步 checkpoint |
| **on-policy 训练** | 学生是否在自己的历史上训练 | Self Forcing、Self-Forcing++、Causal-rCM Stage 3 | 缓解 exposure bias |
| **长视频系统** | KV 如何保存、梯度如何跨块、推理如何实时 | Sparse Forcing、SGF、SGF+、EMA-Sink | 固定/稀疏 KV、长期一致性、吞吐 |

把这些层混为一谈，会产生两个常见误区。第一，认为 Diffusion Forcing 与 DMD 是竞争 loss；其实一个决定训练输入的噪声结构，一个决定蒸馏时学生分布的更新方向。第二，认为 Causal Forcing 推翻了 Self Forcing；实际 Causal Forcing 只修正了 ODE 初始化教师的因果结构，最后仍沿用 Self Forcing 的 asymmetric DMD。

### 1.2 六类鸿沟是更稳定的坐标系

![](assets/Video%20Distillation：经典主线与2026前沿%20-%20Survey/figures/07-six-gaps-matrix.svg)

> **图 2：视频蒸馏的六类鸿沟（综述自绘）。** 2026 年的论文之所以名字相似、机制却差异很大，是因为它们修的是不同层：One-Forcing 修一步质量，Causal Forcing 修师生可见信息，Self Forcing 修历史分布，DuoMatching 修教师质量信号，Leta/SGF 修未来信用，Sparse Forcing 修系统记忆。来源：本综述自绘。

---

## 2. 基础目标：DMD 把“蒸馏”从轨迹回归改成分布匹配

### 2.1 DMD 的核心主张

[[DMD - 分布匹配蒸馏一步图像生成]] 的原始任务是把多步图像扩散模型压成一步生成器。其关键转变不是“用更少的 ODE 步”，而是<span style="color: red">**不要求学生复现教师的具体采样轨迹，只要求学生输出分布接近目标分布**</span>。令学生生成器为 $G_\theta(z)$、目标分布为 $p_{\mathrm{real}}$、当前学生分布为 $p_\theta$，DMD 近似优化 reverse KL：

$$D_{\mathrm{KL}}(p_\theta\|p_{\mathrm{real}}).$$

其生成器梯度可写成加噪空间中两类 score 的差：

$$\nabla_\theta\mathcal L_{\mathrm{DMD}}\propto\mathbb E\left[w(t)\left(s_{\mathrm{fake}}(x_t,t)-s_{\mathrm{real}}(x_t,t)\right)\frac{\partial G_\theta}{\partial\theta}\right].$$

这里 $s_{\mathrm{real}}$ 来自冻结的预训练扩散教师，$s_{\mathrm{fake}}$ 由一个持续在线拟合学生生成分布的辅助 diffusion critic 提供。直觉是：real score 把样本拉向教师高密度区域，fake score 防止所有样本挤进同一模式。原始 DMD 再叠加 LPIPS 回归损失稳定结构。

![](assets/Video%20Distillation：经典主线与2026前沿%20-%20Survey/figures/01-dmd-method-overview.png)

> **图 3：DMD 方法总览。** 生成器、冻结的 real score 模型和动态 fake score 模型形成三网络闭环；最重要的不是某个具体网络，而是“真实分数减生成分数”的更新原则。来源：DMD Figure 2，arXiv:2311.18828；复用自已有精读附件。

### 2.2 DMD 为什么能迁移到视频，又为什么会出新问题

DMD 对样本维度没有本质限制，因此图像 $x$ 可以换成整段视频 latent $X=(x^1,\ldots,x^T)$。但视频引入了三类额外结构：

1. 学生可能是因果 AR，教师却是全序列双向；
2. 学生每个新 chunk 都条件于自己之前生成的历史，误差会累积；
3. joint video score 优先考虑帧间共同合理性，不必然给单帧纹理最强监督。

这三点分别催生 CausVid、Self Forcing 和 DuoMatching。DMD 自己的 reverse-KL mode-seeking、fake score 估计误差和训练成本，则在 2026 年催生 DFD、DMAD、Leta 等改进。

### 2.3 DMD2 与 SiD 提供的上游方法论

[[DMD2 - 改进的分布匹配蒸馏快速图像合成]] 去掉昂贵的回归配对，以 fake critic:generator 约 5:1 的 TTUR 稳住训练，再加真实数据 GAN 和多步 backward simulation，使学生可反超教师。[[SiD - 分数恒等蒸馏一步生成]] 则用 Fisher 散度与 score identity 构造数据无关目标，避免显式 GAN，但需反传 score gradient。视频侧后来反复出现的三个主题——**动态 critic 是否准确、真实数据能否突破教师上限、训练状态是否覆盖推理状态**——都已在 DMD2/SiD 中露出原型。

---

## 3. 序列范式：Diffusion Forcing 不是蒸馏 loss，而是连续程度的 mask

### 3.1 核心主张

[[Diffusion Forcing - 下一Token预测与全序列扩散]] 给同一序列中的每个 token 独立采样噪声等级。对第 $i$ 个 token，可见条件是当前与过去的 noisy tokens；当历史 token 噪声为零时，它退化成 teacher forcing；当未来 token 是纯噪声时，它接近 next-token prediction；当整段共享相近噪声时，它又类似全序列 diffusion。

![](assets/Video%20Distillation：经典主线与2026前沿%20-%20Survey/figures/02-diffusion-forcing-overview.png)

> **图 4：Diffusion Forcing 总览。** 噪声强度被解释为连续的 mask：零噪声 token 已知，纯噪声 token 未知，中间噪声 token 部分已知；因果模型因而可以在同一目标中处理预测、恢复、规划与反馈控制。来源：Diffusion Forcing Figure 2，arXiv:2407.01392；复用自已有精读附件。

### 3.2 它与视频蒸馏的真实关系

Diffusion Forcing 提供的是**如何训练一个多步因果 diffusion teacher/student**的办法，自己并不需要教师模型。它对后续路线的作用有两层：

- 架构层：证明“时间因果 + token-wise diffusion”可以统一，给 CausVid、世界模型和流式生成提供建模底座；
- 初始化层：Causal Forcing 后来指出，若最终少步学生按干净历史自回归生成，拿独立加噪历史训练出的 DF teacher 未必是最匹配的 flow-map teacher，Teacher Forcing 反而更适合构造 causal ODE/CD 初始化。

这不是说 Diffusion Forcing 错了，而是说**训练覆盖不确定历史**与**为 clean-prefix 少步 AR 学生提供单值流映射**是两个不同目标。

---

## 4. CausVid：双向教师到因果少步学生的起点

### 4.1 非对称蒸馏的主张

[[CausVid - 从慢双向到快自回归视频扩散蒸馏]] 首次系统建立双向视频 diffusion teacher 到因果 AR student 的两阶段流程：先用教师 PF-ODE 轨迹做初始化，再用 asymmetric DMD 对齐分布。学生采用块内双向、块间因果的注意力，因此可以按 chunk 生成并复用 KV cache；教师仍看完整视频，以更强全局建模能力提供 real score。

![](assets/Video%20Distillation：经典主线与2026前沿%20-%20Survey/figures/03-causvid-asymmetric-dmd.png)

> **图 5：CausVid 的 asymmetric distillation。** 双向 teacher 与因果 student 的架构不同，但 DMD 只要求输出分布匹配，因此这种非对称在分布监督层面成立。来源：CausVid 方法图，arXiv:2412.07772；复用自已有精读附件。

CausVid 的历史地位不在某一个 VBench 数字，而在两个结论。第一，**双向教师与因果学生不必同构**；第二，视频少步蒸馏可与 decoder-only 式 KV-cache 推理结合。它把 DMD 从“图像一步生成”真正变成“流式视频生成”的核心工具。

### 4.2 留下的两个缺口

CausVid 训练时仍主要使用真实/加噪历史，而推理时历史来自学生自己，因此 exposure bias 尚未闭合。更隐蔽的问题是，它的 ODE 初始化直接回归双向教师整段视频轨迹；学生看不到未来帧，这个逐帧回归目标未必是单值函数。前者由 Self Forcing 解决，后者由 Causal Forcing 解决。

---

## 5. 两条关键修正线：Self Forcing 与 Causal Forcing

### 5.1 Self Forcing：让学生在训练时就吃自己的错误

[[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]] 把历史条件分布写得很清楚：teacher forcing 使用 $p_{\mathrm{data}}$ 历史，Diffusion Forcing/CausVid 使用独立加噪的数据历史，而真实推理使用 $p_\theta$——学生自己生成、带误差且会累积的历史。若训练从未见过 $p_\theta$，长序列推理必然出现 distribution shift。

Self Forcing 因此在训练时执行与推理相同的自回归 rollout，再对完整学生视频使用 DMD、SiD 或 GAN。它是一种**loss-agnostic 的 on-policy 训练范式**：关键不是发明一个新 loss，而是改变 loss 作用的样本分布。

![](assets/Video%20Distillation：经典主线与2026前沿%20-%20Survey/figures/04-self-forcing-paradigms.png)

> **图 6：Teacher / Diffusion / Self Forcing 的历史分布差异。** Self Forcing 唯一让训练上下文真正来自学生自己的 rollout，因此最直接地处理 exposure bias；代价是串行生成、KV 缓存和梯度截断带来的训练复杂度。来源：Self Forcing Figure 1，arXiv:2506.08009；复用自已有精读附件。

Self Forcing 的主要贡献可概括为：**“训练时自回归”成为因果视频蒸馏的标配。** 但为控制显存，历史 KV 通常 detach，未来损失不能训练早期历史应该怎样写入 K/V；这个更深的信用分配缺口直到 SGF 才被显式提出。

### 5.2 Causal Forcing++：用 causal CD 取代昂贵轨迹库

在讲 Causal Forcing++ 之前，必须先回到 [[Causal Forcing - 正确的自回归扩散蒸馏]]。Causal Forcing 指出，双向教师第 $i$ 帧的 ODE 输出依赖未来帧。若学生只看到当前噪声帧和过去历史，同一学生输入可能对应多个教师目标；MSE 的最优解就变成条件均值：

$$G_\theta^*(x_t^i,x_t^{<i},t)=\mathbb E[x_0^i\mid x_t^i,x_t^{<i},t],$$

导致模糊和运动平均。论文把“逐帧 flow map 对因果学生必须是单值的”称为 frame-level injectivity。解决办法是先训练一个 AR diffusion teacher，再由它生成 causal ODE 轨迹；最终 Stage 3 仍使用 Self Forcing 的 asymmetric DMD。

![](assets/Video%20Distillation：经典主线与2026前沿%20-%20Survey/figures/05-causal-forcing-injectivity.png)

> **图 7：Causal Forcing 的 frame-level injectivity。** 双向 teacher→双向 student 在整段视频上可逆；AR teacher→AR student 在相同因果前缀下逐帧单值；双向 teacher→AR student 则因不可见未来而产生一对多监督。来源：Causal Forcing Figure 3，arXiv:2602.02214；复用自已有精读附件。

**Causal Forcing++（arXiv:2605.15141）进一步解决的不是理论方向，而是成本。** causal ODE 需要离线生成并保存完整 PF-ODE 轨迹。causal CD 改为从真实视频前向加噪到 $x_t$，让 AR teacher 在线走一次相邻 ODE step得到 $\hat x_{t-\Delta t}$，再约束 student 与 EMA student 对两个相邻状态预测相同 endpoint。它用局部一致性传播学习同一个 AR-conditional flow map，无需完整轨迹库。

关键结果：

- 2-step frame-wise CF++：VBench Total **84.14**、Quality **84.89**、VisionReward **6.661**；
- 相比 4-step chunk-wise Causal Forcing，首帧延迟约减半；
- Stage 2 从约 **11,600 A800 GPU·hours + 1,900 GiB** 轨迹存储降至约 **2,900 GPU·hours + 0 轨迹存储**，约 4× 成本下降；
- 但 1/2-step 模型的首个 latent frame 仍用 4 步，论文速度不含 VAE，不应表述成全链路“纯两步实时”。

Causal Forcing++ 的主张可以浓缩为一句：**先把 teacher 的可见信息结构改对，再用 consistency 只学局部相邻关系，最后仍靠 on-policy DMD 精修最终分布。**

### 5.3 Self-Forcing++：短教师如何纠正分钟级 rollout

[[Self-Forcing++ - 分钟级高质量视频生成]] 处理另一个正交问题：训练窗口只有约 5 秒，推理却要 1–4 分钟。它先用学生 rolling KV cache 自生成长视频，再从长 rollout 中抽取连续短窗口；对窗口中的 clean latent 做 backward noise initialization，最后用短时双向教师执行 extended DMD。教师不需要生成整段长视频，只负责局部纠偏。

论文报告从 5 秒教师扩展到 100 秒训练目标，并展示 4 分 15 秒生成；但它没有真正长期世界记忆，遮挡后物体身份、布局回访和多事件剧情仍会漂移。换言之，Self-Forcing++ 扩展了“可维持的生成时长”，没有解决“持久世界状态”。

---

## 6. 系统分支：Sparse Forcing 并不改 DMD，而是改长期注意力

### 6.1 Sparse Forcing：把蒸馏与 KV 记忆稀疏化共同训练

**Sparse Forcing（arXiv:2604.21221）** 从注意力可视化观察到两种稳定结构：少数视觉块长期作为持久语义锚点，近期窗口则呈局部、动态、块稀疏模式。它因此维护两类记忆：

- **Persistent memory $\mathcal P$**：固定容量，保留 sink block 与动态选出的长期关键块；
- **Local window $\mathcal L$**：保留近期连续上下文，但只让 query 访问动态 Top-K 局部块。

Persistent Block-Sparse Attention（PBSA）kernel 把训练反传、稀疏注意力与缓存更新统一到专用 GPU kernel。重点是**训练时就使用目标稀疏结构**，而不是蒸馏完成后再做推理剪枝；否则模型可能发生 semantic rewrite。

论文在 Wan2.1-T2V-1.3B、4-step Self-Forcing/DMD 底座上报告：

- 5 秒：VBench 83.88→**84.14**，17.0→**19.9 FPS**，峰值 KV cache 约降 42%；
- 1 分钟：主表一组设置 78.93→**81.96**，13.9→**18.0 FPS**；摘要使用另一配置口径，写作时不应混用；
- 完整 1 分钟 FP16 KV cache 示例约 44.9 GB，说明“少步”并没有自动解决记忆膨胀。

Sparse Forcing 与 DMD 的关系是：**DMD 负责学生生成什么分布，Sparse Forcing 负责学生用哪些历史 token 来生成。** 它与 Causal-rCM、SGF、DuoMatching 理论上可组合，但当前论文没有完成所有组合验证。

### 6.2 Reward Forcing 的 EMA-Sink：另一种固定记忆思路

Reward Forcing 除了奖励蒸馏，还提出 EMA-Sink：初始 sink 不再永远静态，而是把滑窗中被淘汰的 token 以指数移动平均融合进去，使固定容量的 sink 同时携带初始锚点和近期动态。它针对的是“模型过度依赖首帧→运动衰减或闪回首帧”；Sparse Forcing 更一般地学习哪些空间时间块值得持久保留。两者都说明**KV 管理已经成为视频蒸馏的一部分，而非独立部署优化**。

---

## 7. 2026 的目标组合：一步、奖励与真实数据

### 7.1 Causal-rCM：TF-CM 打底，SF-DMD 精修

**Causal-rCM（arXiv:2606.25473）** 将蒸馏统一为前向与反向两段：Teacher-Forcing consistency model 提供 mode-covering、稳定的少步 causal 初始化；Self-Forcing DMD 再在学生自己的 rollout 上做 reverse-KL 精修。其连续时间 TF-sCM / MeanFlow 依赖 custom-mask FlashAttention-2 JVP kernel，约 1–2k iterations 达到离散 TF-dCM 10k iterations 的水平。

代表结果是 frame-wise 1/2-step Wan2.1-1.3B 均达到 VBench **84.63**；chunk-wise 2-step noisy-context 达 **25.6 FPS**。但 reverse-KL 仍对初始化敏感，frame-wise 4-step 可能出现单向 camera drift，历史 KV 也仍 detach。因此 Causal-rCM 解决的是**初始化目标与最终分布优化的互补**，不是完整长程信用分配。

### 7.2 One-Forcing：DMD 加 GAN 补一步高频

**One-Forcing（arXiv:2605.23458）** 针对一步极端设置指出：视频教师 ODE 在高噪声端曲率高度集中，一致性/轨迹学生一步跨越时容易丢运动；DMD 又更擅长低频整体分布，生成帧可能模糊。它给 DMD 加 noised-latent GAN，直接用真实视频 latent 提供高频真假信号，并发现 frame-wise AR 比 chunk-wise 更稳定地训练判别目标。

一步 frame-wise VBench **83.76**，但主配置采用 First-Frame Enhancement：首 block 仍 4 步，之后 block 1 步。因此更准确的说法是“**稳态逐帧一步 AR**”，而不是整段每个 block 从第一帧起都严格单 NFE。

### 7.3 Reward Forcing：把 DMD 倾斜到高运动区域

**Reward Forcing（arXiv:2512.04678，CVPR 2026）** 让 VideoAlign 的 motion reward 以 $\exp(r/\beta)$ 加权 DMD 更新，使学生分布向高运动奖励区域倾斜。它补的不是“教师没有真实感”，而是 vanilla DMD 对高动态与低动态样本同权，后者只要画面清晰就不容易被纠正。

4-step、1.3B 模型报告 VBench **84.13**、单 H100 **23.1 FPS**。消融中去掉 Re-DMD，Dynamic 64.06→43.75；但过强 reward 会使 Dynamic 飙升的同时 Quality 从 80+ 跌到 68.26。结论非常明确：**奖励不是无条件增益，而是在动态、稳定和画质之间重定向分布质量。**

### 7.4 Data-Forcing：用真实样本校正 reverse-KL 的模式塌缩

**Data-Forcing Distillation（DFD，arXiv:2606.18478）** 对 DMD 做了一处看似简单但意义很大的修改：fake score 仍在学生样本上算，real teacher score 改在同条件真实样本 $x\sim p_{\mathrm{real}}$ 上算。这样更新会把学生拉向遗漏模式、推离真实数据中不存在的过饱和模式。

DFD 只需对成熟 DMD2 checkpoint 微调 100–300 步，推理结构不变；但直接从教师初始化约 1400 iterations 仍失败，说明它是后处理修正而不是通用初始化。其“真实数据池”实际主要来自 Wan2.1-14B 合成并筛选的视频，仍是高质量数据代理，不应写成完全独立于教师的真实互联网视频。

---

## 8. 2026 的长程训练：从看见自身历史到训练历史如何写入

### 8.1 SGF：让未来损失训练历史 KV 写入

Self Forcing 虽让学生在自身历史上训练，但历史 KV 为了省显存通常 detach。未来 chunk 的 loss 能训练“如何读取过去”，不能训练“过去的 clean latent 应该怎样编码成未来有用的 K/V”。**Self Gradient Forcing（SGF，arXiv:2607.20368）** 将其称为 historical context-gradient gap。

SGF 用两遍训练恢复这条路径：第一遍无梯度执行真实自回归 rollout并记录退出步；第二遍以 stop-gradient clean latent 重算 context K/V 和 future-to-context attention，让未来 DMD loss 回传进 KV writer，但不反传穿过完整串行采样轨迹。相对 Self Forcing，其显存约 +10.1%、每 5 步训练耗时约 +12.7%，而直接可微串行 KV 会 OOM。

SGF 只用 5 秒训练窗口，却在 60/240 秒上显著改善身份、背景和布局一致性。它不等价于 Sparse Forcing：Sparse Forcing 选择“保存什么”，SGF 训练“保存内容时参数应怎样写”。

### 8.2 SGF+：把 context writer 与 denoiser 分开

**SGF+（arXiv:2610.10429）** 进一步发现，去噪当前帧和把当前帧写成未来上下文是两个角色，其共享参数梯度在所有测试配对上系统性负对齐：Attention 与 FFN 平均夹角约 104.2°/106.3°。因此它复制角色参数，用 $\mathcal C_{\theta_c}$ 负责 clean-context writing，用 $\mathcal D_{\theta_d}$ 负责 noisy-token denoising，两者仍通过因果注意力交互、共同由原始生成 loss 优化。

代价是生成器约 1.4B→2.8B，推理显存增加约 3.11 GB；但单 token 只经过其角色网络，81 帧推理时间几乎不变。论文展示 5 秒训练后持续生成 24 小时，但严格定量仍只到 60/240 秒，因此“24 小时不崩”应视为项目页案例，不是完整 benchmark 结论。

---

## 9. 2026 最新 DMD 改进：估计器、多教师、未来信用与表征

### 9.1 DMAD：直接学习密度比，移除 fake score diffusion

**DMAD（arXiv:2610.02188）** 从最优二分类器 logit 满足 $h^*(x)=\log p_{\mathrm{target}}(x)-\log p_\theta(x)$ 出发，指出其输入梯度就是 DMD 所需的 score difference。它用共享 backbone 上两个判别 head 分别区分“真实数据 vs 学生”和“教师样本 vs 学生”，直接训练 log-density ratio，不再维护完整 fake-score diffusion critic。

这和 One-Forcing 的“DMD 外再加 GAN”不同：**在 DMAD 中，对抗分类本身就是分布匹配估计器。** 它在 Wan2.1-T2V-14B 四步达到 VBench **85.15**；MiniMax-H3-33B 四步相对 DMD2/rCM 人评总体偏好 79.1%/84.6%。受控训练中每更新周期 233.3 秒 vs DMD2 818.1 秒，显著降低训练成本；但理论等价依赖最优判别器，现实中仍只是近似。

### 9.2 DuoMatching：视频 joint + 图像 marginal

[[DuoMatching - 联合边缘分布匹配的少步视频生成]] 的主张是：视频教师适合约束全部帧的联合分布，图像教师更擅长单帧纹理与语义边缘分布。目标变为：

$$J(\theta)=D_{\mathrm{KL}}(Q_\theta\|P_v)+\omega D_{\mathrm{KL}}(M Q_\theta\|P_i).$$

LatentBridge 解决视频 VAE 时间压缩 latent 与图像 VAE latent 的不兼容，Latent Variation Sampling 把有限帧级监督分配到变化大的时间段。

![](assets/Video%20Distillation：经典主线与2026前沿%20-%20Survey/figures/06-duomatching-overview.png)

> **图 8：DuoMatching 三组件。** joint-marginal 目标决定“视频教师管时序、图像教师管单帧”；LatentBridge 让异质潜空间可传梯度；LVS 防止有限图像监督只落在静止段。来源：DuoMatching Figure 3，arXiv:2610.03543；复用自已有精读附件。

其 frame-wise 2-step VBench **83.53**，对应 Causal Forcing++ 基线 80.67；人评整体偏好对多基线超过 80%，但 Temporal & Motion 基本只打平。边缘权重过大时 Dynamic 会明显下降。这说明 DuoMatching 的边界很清楚：**它买的是成像与语义，不是更强运动。**

### 9.3 Leta：直接估计 ScoreDelta，并摊销 FutureCredit

**Leta（arXiv:2610.34092）** 认为强 causal CD 初始化后的学生已经接近教师，此时 DMD 先估两个大 score 再相减，真正的小 correction 容易被估计误差淹没。ScoreDelta 直接回归教师—学生 denoising residual 差；FutureCredit 则把下一 chunk 教师残差经 VJP 映射回当前 chunk，再用 estimator 摊销这个跨块信号。

标准 generator:estimator=1:4 时，DMD→Leta 的 VBench 83.6→**84.5**、VideoAlign 1.692→**2.348**。它与 SGF 都在补跨 chunk 信用，但路径不同：SGF 重算 context K/V 让真实未来 loss 回传，Leta 用下一 chunk teacher VJP 训练一个有偏但便宜的 FutureCredit 估计器。

### 9.4 SF²：从输出分布扩展到内部表征

**Self Flow Forcing（SF²，arXiv:2610.35594）** 指出 DMD 只约束输出空间，极少步学生内部未必学到稳定时空表征。它让更高噪声的学生预测低噪声 EMA teacher 的深层 hidden representations，并用浅层 student→深层 teacher 的 projector/cosine loss形成信息不对称。

SF² 不改 DMD 输出目标，也不需要外部特征模型，因此与 DMAD、DuoMatching、DFD 理论上更正交。论文四步因果模型 VBench **84.15**、VisionReward **7.523**，训练多一次 EMA teacher forward，约占完整周期 6.2%，推理吞吐/延迟与基线相同。当前证据主要在四步 Wan/Hunyuan 配置，一步和超长视频尚未充分验证。

### 9.5 PTD：DMD 之外的轨迹参数化路线

**Parametric Trajectory Distillation（PTD，arXiv:2610.11498）** 提醒我们：2026 并非所有进展都收敛到 DMD。传统轨迹蒸馏要求学生复现高噪声处弯曲的教师转移，少步容量不足时容易平均。PTD 让学生用多项式参数化每个 hop 的曲率，并沿学生自己的预测路径查询教师局部速度；曲率 head 只在训练中存在，推理删除。

Wan2.1-14B 四步 VBench **84.57**，比 PDD 83.67 更强；MiniMax-H3 上相对 LightX2V Turbo 盲测偏好 63.4%。轨迹路线仍保留 seed-to-sample 对应和更好的 mode coverage，适合把多样性放在首位的场景；代价是训练期每 hop 需要多次教师查询。

---

## 10. 统一对比：每篇论文到底主张什么

| 工作 | 核心主张 | 改动层 | 最直接解决的问题 | 推理额外开销 |
|---|---|---|---|---|
| DMD | 匹配分布，不模仿轨迹 | 蒸馏目标 | 多步→一步/少步 | 无额外 critic |
| Diffusion Forcing | token-wise 独立噪声统一 AR 与 diffusion | 训练范式 | 因果序列建模 | 取决于模型 |
| CausVid | 双向教师可以分布蒸馏因果学生 | 架构+蒸馏流水线 | 非流式→流式 | KV cache 反而加速 |
| Self Forcing | 训练时执行真实 student rollout | 样本分布 | exposure bias | 无 |
| Causal Forcing | 轨迹 teacher 必须与 student 因果结构一致 | 初始化 | 非单值 ODE 目标 | 无 |
| Causal Forcing++ | 用 causal CD 学同一 flow map、不要完整轨迹库 | 初始化 | 初始化成本与 1–2 步 | 无 |
| Self-Forcing++ | 长 rollout 抽窗，短教师局部纠错 | 长时域训练 | 分钟级漂移 | 无 |
| Sparse Forcing | 原生可训练持久/局部稀疏记忆 | 注意力系统 | KV 与长视频吞吐 | 减少开销 |
| Causal-rCM | TF-CM mode-covering + SF-DMD mode-seeking | 目标组合 | 稳定少步初始化与最终质量 | 无 |
| One-Forcing | DMD + GAN 补一步高频 | 目标组合 | 一步模糊 | 无 |
| Reward Forcing | reward-weighted DMD + EMA-Sink | 质量+系统 | 动态衰减、首帧复制 | 无 |
| DFD | 在真实样本上求 teacher score | DMD 数据侧 | 多样性塌缩、过饱和 | 无 |
| DMAD | 分类器 log-density ratio 取代 fake score diffusion | 估计器 | 训练成本与 critic 误差 | 无 |
| DuoMatching | 视频 joint + 图像 marginal | 多教师 | 单帧纹理、语义 | 无 |
| Leta | 直接估 correction + 下一 chunk future credit | 估计器/信用 | 小残差难估、跨块梯度断裂 | 无 |
| SGF / SGF+ | 恢复 KV 写入梯度，再解耦 writer/denoiser | 梯度/参数 | 长程身份与布局漂移 | SGF 无；SGF+ 增显存 |
| SF² | 输出分布之外对齐内部表征 | 表征正则 | 少步下细节与时序特征退化 | 无 |
| PTD | 学学生容量可承受的参数化轨迹 | 轨迹蒸馏 | 曲率过大与多样性 | 无 |

---

## 11. 关键指标快照与读数边界

| 工作/设置 | 报告指标 | 需要保留的口径限制 |
|---|---:|---|
| Self Forcing chunk 4-step | VBench 84.31，17.0 FPS | 单 H100；论文内口径 |
| Causal Forcing chunk 4-step | VBench 84.04，17.0 FPS | 相对 Self Forcing 主要赢高运动与偏好指标 |
| Causal Forcing++ frame 2-step | VBench 84.14 | 首 latent frame 4 步；延迟不含 VAE |
| Causal-rCM frame 1/2-step | VBench 84.63 | 使用自身评测/部署配置，勿与其他论文无条件排名 |
| One-Forcing frame 1-step | VBench 83.76 | 首 block 4-step FFE |
| Reward Forcing 4-step | VBench 84.13，23.1 FPS | 奖励与画质存在权衡 |
| Sparse Forcing 1-minute | 主表一组 81.96，18.0 FPS | 摘要和主表存在不同配置口径 |
| DMAD Wan 14B 4-step | VBench 85.15 | 14B 与 1.3B 基线不可直接比较 |
| DuoMatching frame 2-step | VBench 83.53 | 主要涨画质/语义，动态持平 |
| Leta 4-step | VBench 84.5，VideoAlign 2.348 | 受控实验；FutureCredit 为有偏近似 |
| SF² causal 4-step | VBench 84.15，VisionReward 7.523 | 四步验证为主 |
| PTD Wan 14B 4-step | VBench 84.57 | 轨迹路线，训练教师查询更多 |

VBench 跨论文比较很危险：基础模型大小、prompt、生成模式、是否计 VAE、frame-wise/chunk-wise、首帧步数与作者复现协议都不同。**综述应该比较“主张是否被同论文受控消融支持”，而不是把所有 Total 分数排成榜单。**

---

## 12. 失败模式与工程选型

### 12.1 按症状选方法

| 你的少步模型症状 | 最可能的根因 | 优先阅读/尝试 |
|---|---|---|
| 一步视频结构或运动消失 | 高噪声轨迹曲率过大、DMD 高频不足 | One-Forcing、Causal-rCM、PTD |
| 输出模糊但时序稳定 | joint/output-only 监督不足 | DuoMatching、SF²、One-Forcing |
| 过饱和、多样性塌缩 | reverse-KL mode seeking、critic 误差 | DFD、DMAD、PTD |
| 动态衰减、闪回首帧 | static sink + 同权 DMD | Reward Forcing |
| 第一个 chunk 还行，后面越来越漂 | 训练历史不等于推理历史 | Self Forcing、Self-Forcing++ |
| 短视频好、分钟级身份与布局漂移 | 历史 KV 写入无未来梯度 | SGF、SGF+ |
| 长视频显存爆炸、FPS 下降 | KV 全保存、attention 近似稠密 | Sparse Forcing |
| causal student 初始化模糊 | 双向 ODE teacher 依赖未来 | Causal Forcing / ++ |
| DMD 训练太贵、fake critic 跟不上 | score difference 间接估计 | DMAD、Leta |

### 12.2 一个合理的 2026 工程栈

如果目标是实时流式、1–2 步、长视频且允许高训练成本，一个更合理的组合不是押注单篇论文，而是：

1. 训练一个结构匹配的 AR teacher（Teacher Forcing 或适当 causal training）；
2. 用 causal CD / TF-sCM 初始化少步学生；
3. 用 Self-Forcing DMD 在自身 rollout 上精修；
4. 根据症状叠加 DuoMatching（画质/语义）、Reward/DFD（动态/多样性）、SF²（表征）之一；
5. 用 Leta 或 SGF 恢复跨 chunk 信用；
6. 用 Sparse Forcing / EMA-Sink 控制长期 KV；
7. 用 Self-Forcing++ 式长 rollout 训练覆盖部署 horizon。

这套栈目前仍是研究议程而非已有完整系统。各组件可能发生冲突：例如 DuoMatching 的边缘监督与 Reward Forcing 的动态偏好、SGF+ 的双参数显存与 Sparse Forcing 的缓存节省、DMAD 的真实数据 head 与 data-free 配方之间都需要重新调权重和系统预算。

---

## 13. 局限与批判性评价

### 13.1 这个领域最大的证据问题是评测不统一

许多论文只在作者自己的 Wan2.1 复现、内部 prompt 和 VBench 协议上对比；0.1–0.3 的 Total 差异可能小于评测噪声。分钟级与 24 小时案例往往依赖项目页视频和小样本人评，缺少统一长视频 benchmark。Visual Stability、VideoAlign、VisionReward 又各有偏置。未来需要将**短时画质、长时身份、动态丰富度、物理一致性、prompt 跟随和推理延迟**分开测，而不是压成单一总分。

### 13.2 “实时”常常不是端到端实时

部分速度只算 diffusion backbone，不含 VAE；部分首帧仍多步；部分依赖 H100/A800 与定制 kernel；部分 frame-wise 低首帧延迟但总 FPS 不高。因此报告中必须同时写清 NFE、首帧策略、GPU、是否计 VAE、分辨率和帧数。

### 13.3 更长并不等于有世界记忆

Self-Forcing++、SGF+ 和 Sparse Forcing 可以延长稳定 rollout，但都不能严格保证遮挡后物体身份、重访场景几何或交互物理状态。它们主要是在像素/latent 历史上减缓漂移，不是显式对象/3D/事件 memory。实时世界模型仍需要将蒸馏与可检索持久状态结合。

### 13.4 reverse-KL 仍是主线的结构性风险

DMD 的 mode-seeking 提供锐利和效率，也带来多样性塌缩、过饱和和初始化敏感。2026 年的 DFD、DMAD、Reward Forcing、DuoMatching 实际都在给 reverse-KL 加外部修正。这意味着未来很可能从“哪个 DMD 变体最好”转向**可自适应的混合散度与多源 score/reward 组合**。

---

## 14. 个人判断：Video Distillation 正在变成“生成器编译器”

最早的蒸馏只是把 50 步压到 4 步；现在它越来越像一个生成器编译过程：

- **源模型**：全序列、双向、多步、质量高；
- **目标运行时**：因果、流式、1–4 步、固定/稀疏 KV、消费级或单卡低延迟；
- **编译约束**：保留输出分布、让训练覆盖部署状态、修复可见信息错配、管理长期记忆、引入外部质量信号。

从这个视角看，DMD 是“分布级链接器”，causal CD/TF-CM 是“少步初始化编译”，Self Forcing 是“profile-guided on-policy 优化”，Sparse Forcing 是“运行时内存布局”，DuoMatching/Reward/DFD/SF² 是“质量后端”，SGF/Leta 是“跨时间自动微分”。这比把它们统称为几个 Forcing 技巧更接近当前真实技术结构。

我认为接下来最重要的四个方向是：

1. **统一散度与信用**：让 forward mode-covering、reverse mode-seeking、奖励和真实数据反馈按噪声/时间/语义区域自适应组合；
2. **持久状态蒸馏**：教师不仅教像素分布，还教对象、几何和事件状态怎样写入外部 memory；
3. **端到端部署指标**：从 NFE 转向包含 VAE、cache、kernel、首帧、持续 FPS 和功耗的完整口径；
4. **可组合验证**：系统研究 DMAD/Leta/SF²/DuoMatching/Sparse Forcing 是否互补，而不是每篇只在自己的底座上赢一个局部基线。

---

## 15. 材料来源与已有精读

### 15.1 已有精读主链

- [[DMD - 分布匹配蒸馏一步图像生成]]
- [[DMD2 - 改进的分布匹配蒸馏快速图像合成]]
- [[SiD - 分数恒等蒸馏一步生成]]
- [[Diffusion Forcing - 下一Token预测与全序列扩散]]
- [[CausVid - 从慢双向到快自回归视频扩散蒸馏]]
- [[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]]
- [[Self-Forcing++ - 分钟级高质量视频生成]]
- [[Causal Forcing - 正确的自回归扩散蒸馏]]
- [[DuoMatching - 联合边缘分布匹配的少步视频生成]]

### 15.2 本综述新增重点官方来源

| 工作 | 官方论文 | 项目/代码 |
|---|---|---|
| Causal Forcing++ | https://arxiv.org/abs/2605.15141 | https://github.com/thu-ml/Causal-Forcing |
| Sparse Forcing | https://arxiv.org/abs/2604.21221 | https://boxunxu.top/SparseForcing |
| Causal-rCM | https://arxiv.org/abs/2606.25473 | https://github.com/NVlabs/rcm |
| One-Forcing | https://arxiv.org/abs/2605.23458 | https://github.com/Aurora-edu/One-Forcing |
| Reward Forcing | https://arxiv.org/abs/2512.04678 | https://reward-forcing.github.io/ |
| Data-Forcing | https://arxiv.org/abs/2606.18478 | https://csy2077.github.io/dfd-project-page/ |
| SGF | https://arxiv.org/abs/2607.20368 | https://zhuang2002.github.io/SelfGradientForcing/ |
| SGF+ | https://arxiv.org/abs/2610.10429 | https://zihan-su.github.io/self-gradient-forcing-plus |
| DMAD | https://arxiv.org/abs/2610.02188 | https://yzmblog.github.io/projects/DMAD/ |
| Leta | https://arxiv.org/abs/2610.34092 | https://scdzx.github.io/Leta/ |
| SF² | https://arxiv.org/abs/2610.35594 | 论文项目页链接见 arXiv |
| PTD | https://arxiv.org/abs/2610.11498 | https://alan-lanfeng.github.io/PTD/ |

本地官方 PDF、复用的官方关键图、自绘路线图及逐项溯源记录见：`assets/Video Distillation：经典主线与2026前沿 - Survey/`。
