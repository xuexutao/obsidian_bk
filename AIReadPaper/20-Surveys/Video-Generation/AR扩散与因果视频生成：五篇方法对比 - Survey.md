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
  - Diffusion-Forcing
  - Causal-Diffusion
  - Autoregressive
  - Distillation
  - DMD
  - Survey
created: 2026-10-15
updated: 2026-10-15
---

# AR 扩散与因果视频生成：Diffusion Forcing、CausalFusion、CausVid、Self Forcing 与 Causal Forcing

## 基本信息

- **覆盖工作**：Diffusion Forcing（2024-07）、CausalFusion（2024-12）、CausVid（2024-12）、Self Forcing（2025-06）、Causal Forcing（2026-02）
- **主题**：如何把扩散模型的连续生成质量，与自回归模型的因果分解、流式生成和可变长度能力结合起来
- **结论先看**：这五篇不能简单排成一条直线。前两篇是**建模范式**，后三篇是**视频蒸馏工程路线**。最清楚的结构是：
  - **Diffusion Forcing**：让每个时间 token 拥有独立噪声级别，统一 next-token 与 full-sequence diffusion；
  - **CausalFusion**：沿“token 序列 × diffusion 噪声”双轴分解，主要验证于图像与图文多模态；
  - **CausVid → Self Forcing → Causal Forcing**：把慢速双向视频扩散模型蒸馏成少步因果生成器，并依次修复**速度/结构错位、训练—推理分布错位、教师—学生信息结构错位**。

> **一句话总判断**：五篇的共同问题是“怎样让连续视觉 token 既能扩散去噪，又能沿时间或 token 顺序因果生成”；真正的演进不在于名字都含 forcing/causal，而在于它们分别选择了不同的**概率分解、训练上下文和教师结构**。

---

## 1. 先把关系摆正：不是五篇连续升级

### 1.1 两条技术路线

第一条是**原生建模路线**：模型从训练开始就学习 AR 与 diffusion 的统一分解。

- [[Diffusion Forcing - 下一Token预测与全序列扩散]]：对每个时间 token 独立加噪，训练模型处理任意可靠度的历史和未来；
- [[CausalFusion - 因果扩散Transformer生成建模]]：每个 AR step 选择一组 token，再在组内执行 diffusion，形成二维因子分解。

第二条是**视频蒸馏路线**：不从头训练一个统一模型，而是把已有的高质量双向视频扩散模型改造成少步、因果、流式学生。

- [[CausVid - 从慢双向到快自回归视频扩散蒸馏]]：证明双向教师可以通过 ODE 初始化与非对称 DMD，蒸馏为 4 步因果学生；
- [[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]]：训练时直接运行学生自身的 AR rollout，使后训练分布与推理一致；
- [[Causal Forcing - 正确的自回归扩散蒸馏]]：指出双向教师的 ODE 轨迹不是 AR 学生的合法逐帧回归目标，改用 AR teacher 做 causal ODE 初始化。

### 1.2 作者关系也支持这一区分

CausVid 与 Self Forcing 共享 Xun Huang、Eli Shechtman，属于同一视频生成研究脉络；Causal Forcing 的 Guande He 也是 Self Forcing 作者，因此它是对该路线的直接理论修正。Diffusion Forcing 与 CausalFusion 的作者团队则与后三篇没有重叠，更多是平行提出的统一生成范式。

CausalFusion 还需单独强调：**它主要是 ImageNet 图像生成与图文联合建模论文，不是视频生成论文**。它能启发视频，但不能直接用其 ImageNet FID 证明流式视频能力。

---

## 2. 五篇共同的核心逻辑

### 2.1 AR 与 diffusion 解决的是两个不同维度

自回归模型沿序列轴分解：

$$p(x_{1:N})=\prod_{i=1}^{N}p(x_i\mid x_{<i}).$$

它擅长因果生成、可变长度、流式输出和 KV cache，但训练时若总读真实历史、推理时却读取自身历史，就会产生 exposure bias。

扩散模型沿噪声轴分解：

$$p_\theta(x_{t-1}\mid x_t),\qquad t:T\rightarrow0.$$

它擅长连续高维信号和多模态分布，却常用双向注意力一次性处理整段视频，采样慢，也无法在未来信息尚不存在时流式生成。

五篇都在做同一件事：**让一个连续 token 的条件分布由 diffusion/flow 建模，同时让 token 或视频块之间遵循某种 causal/AR 结构。** 差异只在于：这种结构是在原始训练目标里直接学出来，还是从双向教师蒸馏出来。

### 2.2 三种容易混淆的“错位”

这五篇最值得记住的不是五套模型名，而是三种不同错位：

1. **历史可靠度错位**：模型应当知道历史 token 是真实观测、部分噪声还是完全未知。Diffusion Forcing 用独立 per-token noise 建模这一点。
2. **训练—推理轨迹错位**：训练时读取真实/教师历史，推理时读取自身生成历史。Self Forcing 用 self-rollout 处理这一点。
3. **教师—学生信息结构错位**：双向教师的逐帧目标依赖未来，但 AR 学生看不到未来。Causal Forcing 用 AR teacher 的 causal ODE 轨迹处理这一点。

CausVid 首先把工程系统跑通，后两篇则分别修正它留下的第 2、3 类错位。

---

## 3. Diffusion Forcing：把噪声当作连续 mask

### 3.1 核心逻辑

标准 full-sequence diffusion 给整段序列使用同一噪声级别；Diffusion Forcing 则给每个时间 token 独立采样 $k_t$：

$$x_t^{k_t}=\sqrt{\bar\alpha_{k_t}}x_t+\sqrt{1-\bar\alpha_{k_t}}\epsilon_t.$$

当 $k_t=0$ 时，token 完全可见；当 $k_t=K$ 时，它近似完全被 mask；中间状态则表示部分可靠。因果模型被训练去噪所有不同 $k_t$ 的组合，因此同一模型能处理干净历史、带噪历史、缺失历史和不同不确定度的未来。

![](../../10-Papers/Video-Generation/assets/Diffusion%20Forcing%20-%20下一Token预测与全序列扩散/figures/02-method-overview.png)

> 图 1：Diffusion Forcing 的关键不是“给下一帧加噪”，而是让整条时间轴上每个 token 拥有独立噪声级别；Teacher Forcing 和 full-sequence diffusion 都可视作其调度空间的特殊切片。来源：论文 Figure 2。

### 3.2 最重要的贡献

**最重要贡献是重新定义了序列扩散的训练坐标系：时间位置与扩散噪声不再绑定。** 由此可以设计二维采样日程，让近未来逐渐确定、远未来保持高不确定性，并在滚动生成中显式表达“先前预测并非绝对真值”。

它的价值偏基础：统一了 next-token、full-sequence diffusion、任意子序列补全、规划 guidance 与长程 rollout。它并不是现代高分辨率文生视频的直接工程方案；原论文视频实验主要是 64×64/128×128 游戏环境，骨干也以卷积 RNN 为主。

### 3.3 与后续工作的关键差别

Diffusion Forcing 的上下文虽然可被加噪，但仍主要来自**真实数据帧**。它扩大了训练状态覆盖范围，却没有保证训练上下文等于模型实际 rollout 时的自生成错误分布。因此，Self Forcing 的批评不是“DF 没见过噪声”，而是“DF 见到的是被人工加噪的真帧，不是模型自己产生的系统性错误”。

---

## 4. CausalFusion：沿 token 轴和噪声轴双重因子分解

### 4.1 核心逻辑

CausalFusion 将图像 token 分成 $S$ 个 AR 组 $\kappa_1,\ldots,\kappa_S$。第 $s$ 步以此前已经生成的干净 token 为上下文，对当前 token 组执行 diffusion：

$$p_\theta(x_{t-1,\kappa_s}\mid x_{t,\kappa_s},x_{0,\kappa_{1:s-1}}).$$

$S=1$ 时接近标准 DiT：所有图像 token 一起扩散；$S=L$ 时接近逐 token AR；中间值则是 next-token(s) diffusion。

![](../../10-Papers/Foundations/assets/CausalFusion%20-%20因果扩散Transformer生成建模/figures/01-dual-factorization.png)

> 图 2：横轴表示 token/AR step，纵轴表示 diffusion noise step。CausalFusion 在二维平面上移动：先选择一组 token，再把该组从噪声去噪为干净 token，随后进入下一组。来源：论文 Figure 1。

### 4.2 最重要的贡献

**最重要贡献是把 AR 与 diffusion 都解释为概率分布的因子分解，而非两个必须拼接的模块。** 这使文本 next-token loss、连续图像 diffusion loss 和 decoder-only causal attention 可以进入一个统一架构。

工程上更有价值的发现是：AR step 越多，后期 token 可见上下文越丰富，任务会退化为局部插值；训练 loss 更低却不代表生成更强。作者因此使用：

- 指数衰减的 AR-step 采样，增加困难的小 $S$ 任务；
- 随机 token 顺序，减少局部邻域捷径；
- 早期 AR step 加权，让上下文最少的步骤得到更多梯度。

### 4.3 证据边界

论文最强 ImageNet 结果仍使用 $S_{eval}=1$、DDPM 250 步；纯 AR 或多 AR-step 模式的 FID 明显更差。因此它充分证明了“双重因子分解可以作为训练归纳偏置”，但尚未证明多 AR-step 是成熟的高效推理方式，更没有直接验证视频流式生成。

---

## 5. CausVid：把双向视频扩散蒸馏成少步因果生成器

### 5.1 核心逻辑

CausVid 的目标很工程化：高质量视频 DiT 是双向、50 步、必须整段生成；如何得到一个 4 步、块间因果、支持 KV cache 的学生？

它采用两阶段训练：

1. **ODE 初始化**：从双向教师的 PF-ODE 轨迹采样中间状态，以 MSE 让因果学生先学会基本 flow map；
2. **Asymmetric DMD**：冻结双向 teacher 作为真实数据 score，在线训练 fake score 跟踪学生分布，用二者之差更新因果学生。

![](../../10-Papers/Video-Generation/assets/CausVid%20-%20从慢双向到快自回归视频扩散蒸馏/figures/03-distillation-method.png)

> 图 3：上半部分是双向教师 ODE 轨迹初始化，下半部分是双向 teacher → 因果 student 的非对称 DMD。学生采用块内双向、块间因果注意力。来源：论文 Figure 6。

### 5.2 最重要的贡献

**最重要贡献是第一次把“强双向教师监督弱信息结构的因果学生”做成了可用的视频生成系统。** DMD 只要求最终学生分布接近高质量视频分布，不要求教师与学生的注意力结构相同，因此可以保留 teacher 的全局质量，又让 student 以 4 步方式流式输出。

论文报告约 1.3 秒初始延迟、9.4 FPS，并支持动态 prompt、图生视频和流式 video-to-video。它证明“高质量、少步、因果视频”不是互斥目标。

### 5.3 留下的问题

CausVid 的 ODE 初始化和 DMD 训练仍没有严格模拟最终 AR 推理：训练状态主要由教师轨迹或在真实视频上加噪得到，推理却要依赖学生自己的历史。后续 Self Forcing 抓住了 exposure bias；Causal Forcing 又进一步指出，双向教师的 ODE 目标本身对 AR 学生并非良定义。

---

## 6. Self Forcing：训练时就让模型承受自己的错误

### 6.1 核心逻辑

Self Forcing 在训练阶段直接执行和推理一致的 AR rollout：当前帧/块生成完成后，把它的 KV 写入 cache，下一块以这些**自身生成历史**为条件继续去噪。完整视频生成后，再用 DMD、SiD 或 GAN 做视频级分布匹配。

![](../../10-Papers/Video-Generation/assets/Self-Forcing%20-%20弥合自回归视频扩散的训练推理鸿沟/figures/01-training-paradigms.png)

> 图 4：Teacher Forcing 使用干净真实历史，Diffusion Forcing 使用不同噪声级别的真实历史；Self Forcing 则在训练时直接使用模型此前生成的历史，因此训练过程最接近真实推理。来源：论文 Figure 1。

### 6.2 最重要的贡献

**最重要贡献不是某个新 loss，而是把自回归视频扩散后训练变成 on-policy 式 self-rollout。** 模型不再只学习“给定正确历史如何生成下一块”，而是直接优化“沿自己的错误轨迹走完整段视频后，联合分布是否仍像真实视频”。

为了让这件事可训练，论文加入：

- 4 步 few-step diffusion，缩短展开链；
- 只对每帧最后一次去噪保留梯度；
- 对跨帧 KV cache 截断梯度；
- rolling KV cache，把长视频复杂度降到 $O(TL)$；
- DMD、SiD、GAN 三种分布匹配目标，证明收益来自训练范式而非单一损失。

Self Forcing 在相同 1.3B 规模下达到 17 FPS、约 0.69 秒首块延迟，并显著优于 CausVid 的颜色漂移和过饱和失败模式。

### 6.3 它没有解决什么

Self Forcing 修复的是**后训练阶段的 exposure bias**，但其 ODE 初始化仍来自双向教师。换句话说，后半程 on-policy，不代表前半程的回归目标就一定合理。这正是 Causal Forcing 的切入点。

---

## 7. Causal Forcing：先让 ODE 教师与学生拥有相同信息

### 7.1 核心逻辑

双向教师的 PF-ODE 对完整视频 $x_t^{1:N}$ 是确定映射，但第 $i$ 帧的输出可能依赖未来噪声帧 $x_t^{>i}$。AR 学生只看到当前帧与过去前缀，因此相同的学生输入可能对应多个不同的双向教师目标。MSE 的最优解就会退化为条件均值：

$$G_\theta^*(x_t^i,x_t^{<i},t)=\mathbb E[x_0^i\mid x_t^i,x_t^{<i},t],$$

从而平均掉未来不确定性，产生模糊、弱动态或时序不一致。

![](../../10-Papers/Video-Generation/assets/Causal%20Forcing%20-%20正确的自回归扩散蒸馏/figures/02-injectivity-principle.png)

> 图 5：双向 teacher → 双向 student 只需 video-level injectivity；AR student 则需要 frame-level injectivity。双向 teacher 的当前帧目标会随不可见未来改变，因此不是 AR student 的单值回归目标。来源：论文 Figure 3。

Causal Forcing 的三阶段流程是：

1. 用 **Teacher Forcing** 训练一个多步 AR diffusion teacher；
2. 用该 AR teacher 生成 causal PF-ODE trajectory，初始化少步 AR student；
3. 保留 Self Forcing 的 self-rollout + asymmetric DMD，用强双向 score teacher 拉高最终分布质量。

### 7.2 最重要的贡献

**最重要贡献是提出 frame-level injectivity，并把“教师选择”与“损失类型”联系起来。**

- 用 MSE/ODE 回归学习确定 flow map 时，teacher 与 student 必须拥有相同可见信息，所以要用 AR teacher；
- 用 reverse-KL/DMD 匹配最终生成分布时，teacher 可以是更强的双向模型，因为学生不需要逐点复制其未来信息。

这比单纯再加一个 loss 更有普适性：任何“非因果 teacher → 因果 student”的序列蒸馏，都应先检查监督目标是否依赖学生看不到的信息。

### 7.3 和 Self Forcing 的关系

Causal Forcing 不是否定 Self Forcing，而是**保留 Self Forcing 的 Stage 3，替换其 ODE 初始化**。在相同 17 FPS、0.69 秒延迟下，它相对 Self Forcing 把 Dynamic Degree 从 57 提升到 68、VisionReward 从 5.820 提升到 6.326、Instruction Following 从 48 提升到 56。

---

## 8. 横向对比表

| 工作 | 定位 | 训练上下文来自哪里 | 是否需要 teacher | 核心机制 | 主要修复的问题 | 最重要贡献 | 主要边界 |
|---|---|---|---|---|---|---|---|
| **Diffusion Forcing** | 通用序列建模范式 | 不同噪声级别的真实 token | 否 | 独立 per-token noise + 二维采样调度 | 历史可靠度与未来不确定性 | 统一 next-token 与 full-sequence diffusion | 原论文不是大规模视频 DiT；仍非 self-generated context |
| **CausalFusion** | 图像/多模态统一架构 | 先前干净 token + 当前 noisy token group | 否 | token 组 × diffusion step 双因子分解 | AR 与 diffusion 的结构统一 | decoder-only next-token(s) diffusion | 最强结果仍在 $S=1$；无视频流式证据 |
| **CausVid** | 实时因果视频蒸馏 | 双向 teacher ODE / 加噪训练状态 | 是，双向视频扩散 | ODE init + asymmetric DMD + block-causal attention | 多步双向模型太慢、不可流式 | 把双向视频 teacher 蒸馏成 4 步因果 student | exposure bias；ODE teacher/student 信息不匹配 |
| **Self Forcing** | 因果视频后训练范式 | 学生自身 AR rollout | 是，双向 score teacher | self-rollout + video-level DMD/SiD/GAN + rolling KV | 训练—推理分布错位 | on-policy 式自回归视频扩散训练 | ODE 初始化仍可不合法；长度外推仍退化 |
| **Causal Forcing** | 因果视频蒸馏理论修正 | AR teacher 的 causal ODE + student self-rollout | AR ODE teacher + 双向 score teacher | TF AR teacher → causal ODE → Self-Forcing DMD | teacher/student 信息结构错位 | frame-level injectivity 与分阶段教师原则 | 三阶段更复杂；未直接解决长视频 |

---

## 9. 最关键的演进：从“覆盖状态”到“匹配轨迹”再到“匹配信息结构”

### 9.1 第一步：Diffusion Forcing 扩大训练状态覆盖

训练时让模型看到干净、带噪和未知 token 的任意组合，回答的是：**历史并不总是完全可靠，模型能否把可靠度一起建模？**

### 9.2 第二步：Self Forcing 让训练轨迹等于推理轨迹

人工加噪的真实帧仍不等于模型自己的错误。Self Forcing 回答的是：**模型能否在训练时就运行自己的推理循环，并对最终联合分布负责？**

### 9.3 第三步：Causal Forcing 让监督信息等于学生可见信息

即使训练过程 self-rollout，若初始化目标来自一个看见未来的 teacher，学生输入与 target 仍可能不是单值关系。Causal Forcing 回答的是：**teacher 产生监督时，是否只能使用 student 将来实际可见的信息？**

这三步不是相互否定，而是逐层收紧条件：

$$\text{状态覆盖}\rightarrow\text{轨迹对齐}\rightarrow\text{信息结构对齐}.$$

CausVid 则位于三者之间，提供了让整个路线成立的工程底座：block-causal attention、few-step generation、KV cache 与 asymmetric DMD。

---

## 10. 哪篇贡献最重要

### 10.1 从“概念影响力”看：Diffusion Forcing

它给出了最通用的统一视角：噪声就是连续 mask，每个 token 可以有自己的不确定度。该思想可迁移到视频、规划、机器人、时间序列和世界模型。其贡献不依赖某个具体 teacher 或视频 backbone。

### 10.2 从“把实时视频做成”看：CausVid

CausVid 的贡献是证明双向强 teacher 可以转成少步因果 student，而且质量、吞吐和动态 prompt 能同时成立。没有这一步，后面的 exposure bias 和 causal ODE 讨论都缺少现实系统载体。

### 10.3 从“训练范式”看：Self Forcing

Self Forcing 把自回归视频扩散后训练改成 self-rollout，使训练目标真正作用于推理分布。这个思想最容易成为后续实时世界模型、交互视频和连续动作生成的通用 post-training 配方。

### 10.4 从“理论纠错”看：Causal Forcing

Causal Forcing 最有价值的不是 VBench 多 0.30，而是说明：**逐点回归 teacher flow 时，teacher 不能使用 student 看不到的信息。** 这是一个可能迁移到语言、音频、机器人轨迹乃至 VLA 蒸馏的原则。

### 10.5 CausalFusion 的位置

CausalFusion 的贡献更偏统一生成架构和训练难度设计。它与视频蒸馏路线的连接点是“组内 diffusion、组间 causal”，但当前证据不足以把它视为 CausVid/Self Forcing 的直接前驱或替代方案。

---

## 11. 最终总结

如果只记五句话：

1. **Diffusion Forcing**：给每个 token 独立噪声级别，让模型学会在任意历史可靠度下生成；
2. **CausalFusion**：每个 AR step 生成一组 token，组内 diffusion、组间 causal，把 AR 与 diffusion 放入同一二维因子分解；
3. **CausVid**：用 ODE 初始化 + 双向 teacher 的 asymmetric DMD，把 50 步双向视频模型变成 4 步因果流式模型；
4. **Self Forcing**：训练时直接按推理方式 self-rollout，再匹配整段视频分布，修复 exposure bias；
5. **Causal Forcing**：ODE 初始化必须由 AR teacher 提供，避免双向 teacher 的未来信息让 AR student 学成条件均值。

我的整体判断是：**五篇最值得抽象出来的统一公式不是某个 loss，而是“连续 token 的 diffusion 条件分布 + token/chunk 之间的 causal factorization”。真正决定系统是否稳定的，是训练上下文、推理轨迹和教师可见信息三者能否一致。**

---

## 12. 材料来源

| 工作 | 官方来源 | 本地精读 |
|---|---|---|
| Diffusion Forcing | https://arxiv.org/abs/2407.01392 | [[Diffusion Forcing - 下一Token预测与全序列扩散]] |
| CausalFusion | https://arxiv.org/abs/2412.12095 | [[CausalFusion - 因果扩散Transformer生成建模]] |
| CausVid | https://arxiv.org/abs/2412.07772 | [[CausVid - 从慢双向到快自回归视频扩散蒸馏]] |
| Self Forcing | https://arxiv.org/abs/2506.08009 | [[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]] |
| Causal Forcing | https://arxiv.org/abs/2602.02214 | [[Causal Forcing - 正确的自回归扩散蒸馏]] |

本文引用的五张图片均来自上述论文的本地精读附件，Figure 编号、PDF 页码与原始 URL 详见各论文 `sources.md`。
