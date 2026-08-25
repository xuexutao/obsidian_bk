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
tags: [Video-Generation, Survey, Diffusion, DiT, 3D-VAE, Spacetime-Patch, Flow-Matching, VQ-VAE]
created: 2026-08-25
updated: 2026-08-25
---

# 视频生成模型的发展与架构演进：从逐帧建模到时空统一表示

> **结论先看：** 这篇综述把视频生成在 2021–2025 的五年演化，梳理成一条以「问题演化」为骨架的主线。它不是一个「模型越来越大」的单调故事，而是「表示 + 骨干 + 范式」三条线同步进化的故事。
>
> - **表示线**：逐帧 → 离散 token（VQ-VAE）→ 3D 时空潜变量（3D VAE）→ spacetime patches（统一时空 patch）；
> - **骨干线**：2D CNN + 时间 RNN/3D 卷积 → 3D U-Net（视频扩散主流）→ 扩散 Transformer（DiT，Sora 之后成为默认）；
> - **范式线**：GAN → VAE/自回归（VQ-VAE + GPT）→ 扩散（DDPM/LDM）→ 流匹配（Rectified Flow / Flow Matching）。
>
> 把这五条主线放在一起，能看到三组贯穿始终的张力：**数据**（配对 vs 无配对 vs 大规模）、**时空耦合**（逐帧独立 vs 时空分解 vs 时空统一）、**算力/一致性**（短片段 vs 长视频、低分辨率 vs 高分辨率、单任务 vs 多任务）。
>
> **总体评级：★★★★☆（4.5/5）**—— 作为「视频生成方向的地图册 + 演化史」，本综述把五年的架构演进讲透了。扣分点主要有两条：（1）本综述未单独精读 Phenaki、Imagen Video 的原文，相关章节以二手综述 + 官方报告为依据；（2）Sora 的 spacetime patches 原图为 OpenAI 官方博客动态渲染，本综述未本地化，统一以 HunyuanVideo Figure 5 作为可下载的同源架构代表，并在正文标注 Sora 原图出处。

## 基本信息

**报告主题**：视频生成模型的架构演进——从逐帧建模到时空统一表示

**覆盖范围**：2021.04（VideoGPT）— 2025.03（HunyuanVideo），2025.05（Movie Gen）、2025.05–2026（LongLive-2.0 等长视频系统）作为补充

**核心论文**：VideoGPT、Phenaki、Make-A-Video、Imagen Video、VideoLDM、AnimateDiff、DiT、Sora、CogVideoX、HunyuanVideo、Movie Gen、LongLive-2.0

**撰写日期**：2026-08-25

**报告定位**：以「问题演化」为主轴，覆盖表示、骨干、范式三条主线的协同演进，回答「为什么后面的架构会出现」「每条路线留下了什么开放问题」

---

## 1. 研究背景与演化主线

### 1.1 视频生成的特殊性：三重约束

视频生成（Video Generation）任务看起来像是图像生成的时间扩展——给定文本、图像或低分辨率视频片段，生成一段时空一致的视频。但实际上，视频给生成模型设下了三道几乎互相矛盾的约束，这些约束直接决定了整个领域五年来的架构走向。

**第一，时空耦合与一致性。** 视频不是「一组独立帧」，而是「时间方向上强相关、相邻帧之间差异有限但必须平滑」的时空信号。如果模型把每一帧独立生成，就只能得到一组「看起来像同一场景但运动不连贯」的图像拼接。**时间一致性**不是质量指标，而是基本可用性——人眼对帧间抖动、运动不连续极度敏感。这是视频生成区别于图像生成最本质的一票否决条件。

**第二，数据稀缺与配对成本。** 图像-文本对虽然昂贵，但可以拼出 LAION 这种十亿级数据集。视频-文本对则同时需要文本、视频片段、人工标注，规模化困难得多。2022 年之前的视频生成研究长期被「几千小时人工标注视频」绑死，直到扩散时代才有 Make-A-Video、CogVideo 这类「从图像数据外推到视频」的方法解放出来。

**第三，算力墙与时空 token 数。** 一段 16 帧、256×256 的视频有 $16 \times 3 \times 256 \times 256 \approx 3.1M$ 个像素。如果直接在这个量级上跑自注意力，单帧 256² 的序列长度是 65,536，16 帧全部展平是 $1M+$。这就是为什么视频生成比图像生成晚了整整五年才进入「大规模」阶段——不是方法不行，是算力和显存拖了后腿。

这三道约束共同指向一个核心问题：**视频该用什么样的表示喂给生成模型？** 围绕这个问题，五年的演化分裂出三条互相缠绕的主线。

### 1.2 三条主线：表示、骨干、范式

视频生成模型的架构演进，可以从三条主线独立看：

**表示线**：视频 = 帧序列 → 视频 = 离散 token 序列（VQ-VAE）→ 视频 = 3D 时空潜变量（3D VAE）→ 视频 = spacetime patches（统一时空 patch）。表示的每次升级，都是为了让生成模型「更便宜地处理时空」。

**骨干线**：2D CNN + 时间 RNN/3D 卷积（早期）→ 3D U-Net（视频扩散主流）→ 扩散 Transformer（DiT，Sora 之后成为默认）。骨干的演进节奏其实和图像生成几乎同步——DiT 在 2023 年取代 U-Net，2024 年扩散 Transformer 统治视频生成。

**范式线**：GAN → VAE / 自回归（VQ-VAE + GPT）→ 扩散（DDPM / LDM）→ 流匹配（Rectified Flow / Flow Matching）。范式线和图像生成主线高度同步，但视频生成因为时空一致性约束，多停留了一年自回归时代（2021–2022），直到 2022 下半年才全面转向扩散。

这三条线不是各自独立演化，而是互相锁定：**VQ-VAE 化（表示）解锁了自回归（范式），3D VAE 化（表示）解锁了时空潜扩散（范式 + 骨干），spacetime patches（表示）解锁了统一时空 DiT（骨干 + 范式）**。每次「表示」升级，都打开了新的「骨干 + 范式」组合空间。

### 1.3 本文的叙事结构

本报告按时间顺序展开，但每一节都聚焦「这条工作解决了上一节留下的什么问题、又为下一节留下了什么新问题」：

- 第 2 章：GAN 时代的困境 + VQ-VAE 自回归开山（VideoGPT、Phenaki）；
- 第 3 章：扩散模型接管视频生成（Make-A-Video、VideoLDM、Imagen Video、AnimateDiff）；
- 第 4 章：扩散 Transformer 取代 U-Net（DiT）；
- 第 5 章：时空统一与 3D VAE + DiT 时代（Sora、CogVideoX、HunyuanVideo、Movie Gen）；
- 第 6 章：横向对比；
- 第 7 章：系统、长视频与工程化（LongLive-2.0）；
- 第 8–9 章：开放问题与个人判断。

涉及的关键单篇论文均已单独精读或在文献索引中标注来源；本报告聚焦「共同问题下的相对位置」，而不是复述每篇的全部细节。

---

## 2. 早期探索：GAN、VAE 与离散 token 自回归（2016–2022）

### 2.1 GAN 时代：图像拼接而非视频生成

最早一批视频生成工作几乎清一色走 GAN 路线。2016 年的 TGAN 用「时间方向的一维卷积 + 2D 空间卷积」生成短视频帧序列；MoCoGAN（2017）把视频内容分解为「内容向量」+「运动码本」；DVD-GAN（DeepMind, 2019）把 BigGAN 扩展到视频，用双流判别器分离前景/背景。这些工作的核心思路都是「GAN 在图像上能成功，把它从空间扩展到时间维度」。

GAN 路线的最大问题是**时间一致性**。GAN 的对抗损失天然关注「单帧是否真实」，对「相邻帧之间是否平滑」没有直接约束，结果生成的视频通常会出现「主角突然变色」「背景在抖」一类问题。为了缓解，后续工作加上了 flow loss、temporal discriminator、内容-运动解耦，但都只能缓解，不能根治。这是 GAN 在视频上始终没有走出 demo 级别的根本原因。

GAN 时代的另一个教训是**算力 / 数据 / 任务复杂度三者不匹配**：那时候的算力和视频数据规模，根本支撑不起 GAN 这种「从零生成视频分布」的范式。于是视频生成的主线在 2020–2021 年间，从 GAN 默默转向了「自回归视频 + VQ-VAE」路线。

### 2.2 VQ-VAE + 自回归开山：VideoGPT

2021 年 4 月，VideoGPT（Yan et al., 2021）借鉴了 VQ-VAE（van den Oord et al., 2017）和 ImageGPT 的思路，首次把视频生成问题重写为「离散 token 上的自回归」。在讲它的两阶段训练之前，必须先说清 VQ-VAE 这个底座——因为它才是整条「离散 token」路线的地基。

**VQ-VAE 的机制：把连续隐变量「查表」成离散码。** 普通 VAE 的隐变量 $z$ 是连续的，而 VQ-VAE 引入一个可学习的码本（codebook）$E = \{e_1, e_2, \dots, e_K\}$，把编码器输出的连续向量 $z_e(x)$ 替换为码本中「欧氏距离最近」的那一个嵌入：

$$z_q(x) = e_k,\quad k = \arg\min_j \|z_e(x) - e_j\|_2^2$$

这一步叫**向量量化**（vector quantization），本质上是把每个空间位置分配到一个离散索引 $k$。于是整段视频就被压缩成一串整数 token，正好可以交给语言模型式的 Transformer 做 next-token 预测。因为「取 argmin」这一步不可导，VQ-VAE 用 **straight-through estimator** 在反向传播时把 $z_q$ 的梯度原样拷贝给 $z_e$，再配合两个「对齐」项迫使编码器输出靠近码本、码本靠近编码器输出。完整的 VQ-VAE 训练目标写出来是三段：

$$\mathcal{L}_{\text{VQVAE}} = \underbrace{\|\hat{x} - x\|_2^2}_{\text{重建}} + \underbrace{\|\text{sg}[z_e] - e\|_2^2}_{\text{码本对齐}} + \underbrace{\beta\,\|z_e - \text{sg}[e]\|_2^2}_{\text{承诺损失}}$$

其中 $\text{sg}[\cdot]$ 是 stop-gradient（截断梯度）。第一项让「解码回去像原视频」，后两项让「编码器输出和码本互相靠拢」，$\beta$ 平衡二者。

**VideoGPT 的两阶段训练。** 第一阶段按上面的 VQ-VAE 目标训练 3D 编码器/解码器 + 码本，把视频帧压成离散 latent code；第二阶段冻结 VQ-VAE，用一个 GPT 结构的 Transformer 在这些 code 上做自回归建模，最大化条件似然：

$$\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \log p_\theta\big(s_t \mid s_{<t},\, c\big)$$

其中 $s_t$ 是第 $t$ 个 latent token，$c$ 是条件（动作标签或帧条件）。生成时，从 Transformer 自回归采样出一串 token，再交给 VQ-VAE 解码器还原成视频帧。下图展示了这一两阶段管线：

![](assets/视频生成模型发展与架构演进%20-%20Survey/figures/01-videogpt-pipeline.png)

> 图 1：VideoGPT 的两阶段训练流程。左侧用 Conv3D Encoder + Codebook 把视频压缩为离散 latent，右侧用 Transformer 在这些 latent 上做自回归预测，最后再用 Conv3D Decoder 解码回视频。  
> 来源：论文 Figure 2，p.3，https://arxiv.org/abs/2104.10157

**为什么这件事重要。** VideoGPT 第一次用「离散 token + Transformer」统一了视频生成范式——和 GPT 时代语言模型的基础设施完全对齐，意味着视频生成可以复用 LLM 全套训练/推理栈。但它把 VQ-VAE 的短板也一并继承了下来：码本容量小导致 token 化丢失高频细节、Transformer 的 context length 短导致只能生成 4 帧低分辨率（64×64）单类物体视频（BAIR 机器人推动）。这几个瓶颈本质上都指向同一句话——**「token 化太粗」+「上下文太短」**，而它们要到 2024 年 3D VAE + 长上下文 DiT 出现后才被真正拆掉。

### 2.3 长视频自回归：Phenaki

2022 年，Phenaki（Villegas et al., 2022）把 VideoGPT 的范式往前推了一步，专门解决「长视频生成」问题。它的两个关键组件需要分开看。

**第一个组件：C-ViViT，因果 3D 视频 tokenizer。** VideoGPT 的 VQ-VAE 是「整段视频一次性编码」，token 之间没有严格的时间因果约束。Phenaki 换成一个**因果** ViT tokenizer（C-ViViT）：时间方向上用因果注意力（causal attention），每个时刻的 token 只能看到当前帧及之前帧，看不到未来。这样做的目的是**支持变长、流式的 token 序列**——生成时可以一边出 token 一边喂回条件，为分钟级长视频的「逐段生成」铺路。视频被 C-ViViT 压成一串时空离散 token，供下游语言模型消费。

**第二个组件：双向 masked transformer。** 纯自回归（如 VideoGPT）虽然生成质量稳定，但 token 串行产出、长序列推理极慢。Phenaki 没有用单向自回归，而是用**双向注意力 + 随机 mask 训练**（类似 BERT/MAE）：训练时随机盖住一部分 token，让模型用双向上下文重建被盖住的 token；推理时通过多次「迭代去 mask」的方式逐步填充 token。双向注意力的好处是每个位置都能同时看到前后文，长视频里「后文依赖前文」的时空一致性更容易建模，且并行度远高于纯自回归。

$$\mathcal{L}_{\text{mask}} = -\sum_{t \in \mathcal{M}} \log p_\theta\big(s_t \mid s_{\setminus \mathcal{M}},\, c\big)$$

其中 $\mathcal{M}$ 是被随机 mask 的 token 集合，模型以「未被 mask 的 token + 条件 $c$」为上下文预测它们。

Phenaki 的工程意义是**证明自回归/掩码路线可以做到分钟级长视频**——这在 2022 年是独一无二的。但它的局限同样继承了 VQ-VAE 路线：离散 token 化不可避免地丢失高频细节（高频纹理被码本「平均掉」），token 序列长度仍受 Transformer 注意力复杂度限制，且逐 token/逐段生成的累计成本远高于同期正在崛起的图像扩散。

更关键的是**范式级的岔路口**。到 2022 年中，LDM（Latent Diffusion Models）和 Stable Diffusion 已在图像生成上证明扩散范式的优势——采样质量更高、训练更稳定、天然适合并行的迭代去噪。于是 2022 下半年起，视频生成主线全面从自回归转向扩散，VQ-VAE 自回归路线逐渐退居二线（只保留在「无限长度生成」等少数场景）。

**这一节的遗留问题（留给下一篇）：** VQ-VAE 路线证明了「token 化」可行，但留下了三个问题：（1）能不能在扩散范式下做 token 化？（2）token 化能不能不丢失细节？（3）能不能从图像数据外推到视频（解决配对数据稀缺）？这三个问题被 Make-A-Video、VideoLDM、Imagen Video 在 2022 下半年到 2023 上半年集中回答。

---

## 3. 扩散模型接管：时空分解与无配对数据（2022–2023）

### 3.1 Make-A-Video：时空分解 + 无配对数据

2022 年 9 月，Meta 的 Make-A-Video（Singer et al., 2022）是第一个把扩散模型系统化扩展到视频的工作。它的核心思路有三条，逐一拆开。

**第一，把时空维度分开处理（时空分解）。** Make-A-Video 不从头训练视频扩散模型，而是拿一个预训练好的 T2I（text-to-image）扩散模型当「空间基座」，在其上插入三类**时间模块**：

- **时间卷积（temporal convolution）**：用 3D 卷积核在「相邻帧 + 空间」上滑动，让网络感知跨帧的空间模式。为了省算力，它把 3D 卷积分解成「1×3×3 空间卷积 + 3×1×1 时间卷积」两段（pseudo-3D 卷积），等效于先做单帧空间特征、再沿时间轴混合——这正是「时空分解」思想的微观体现；
- **时间注意力（temporal attention）**：在空间注意力之后接一个「沿时间维度」的注意力层，让帧间信息显式流动；
- **帧插值网络（frame interpolation）**：在关键帧之间插出中间帧，提高帧率与运动平滑度。

这三类模块都是「在已有 T2I 网络里插入或附加」，而非从头训练——训练代价因此大幅降低。

**第二，不依赖配对的 text-video 数据。** 视频-文本对稀缺，是 2022 年前后视频生成被卡死的主因。Make-A-Video 用了一个漂亮的拆招：**空间先验从「图像-文本对」学，时间先验从「无标签视频」学**。先用海量图文对训练 T2I 基座（学「文字→画面」），再用无标签视频训练那批时间模块（学「画面→运动」）。两者解耦后，就不再需要昂贵的配对视频-文本数据了。这一招是它能在 2022 年就训出高质量视频的关键，也是后续 VideoLDM、Sora 等「复用图像先验」思路的源头。

**第三，三阶段级联生成。** 推理时按分辨率从低到高逐级生成：T2I 生成关键帧 → 时空解码器 $D^t$ 生成 16 帧 64×64 → 帧插值 $\uparrow_F$ 提高帧率 → 时空超分 $\mathrm{SR}^t_l$ + 空间超分 $\mathrm{SR}_h$ 提到 768×768。整条推理链写成一个复合函数：

$$\hat{y}_t = \mathrm{SR}_h \circ \mathrm{SR}_l^t \circ \uparrow_F \circ D^t \circ P\big(\hat{x},\, C_x(x)\big)$$

其中 $P$ 是「文本→图像嵌入」的先验，$C_x(x)$ 是文本编码，$\hat{x}$ 是采样的图像 embedding。下图展示了这一高层架构：

![](assets/视频生成模型发展与架构演进%20-%20Survey/figures/02-makeavideo-architecture.png)

> 图 2：Make-A-Video 的高层架构。Input Text 经 Prior P 编码为 image embedding，Spatiotemporal Decoder D^t 从中解码出 16×64×64 的低分辨率视频，再经帧插值 t_F、时空超分 SR^t_l、空间超分 SR_h 输出 768×768 视频 $\hat{y}$。整条推理公式为 $\hat{y}_t = \mathrm{SR}_h \circ \mathrm{SR}_l^t \circ \uparrow_F \circ D^t \circ P(\hat{x}, C_x(x))$。  
> 来源：论文 Figure 2，p.4，https://arxiv.org/abs/2209.14792

Make-A-Video 的代表意义是**第一次把扩散范式推到视频生成，并摆脱了对配对数据的依赖**。但它的架构带着明显的历史包袱：「时空分解」是工程妥协而非最优解——时间与空间在网络里被分开处理，时间一致性要靠「时间注意力 + 帧插值 + 级联超分」层层兜底，而非原生建模。这恰恰是后面 VideoLDM、Sora 要正面解决的问题。

### 3.2 VideoLDM：图像 LDM + 时空层微调

2023 年 4 月，Blattmann et al.（NVIDIA）的 VideoLDM 提出了一种比 Make-A-Video 更优雅的扩展方式。要理解它的贡献，先得说清「潜扩散（LDM）」这个底座。

**LDM 做了什么。** 标准 DDPM 直接在像素空间逐步加噪/去噪，显存和算力都极贵。LDM 的洞察是：先用一个预训练 VAE 把图像压到低维潜空间，**在潜空间里做扩散**，最后再解码回像素。潜空间维度比像素低几十倍，扩散模型因此能训到更大规模。这个「潜空间扩散」范式（Stable Diffusion 即其代表作）是 VideoLDM 的起点。

**VideoLDM 的关键一招：只给 LDM 的潜空间 U-Net 插入「时间层」，其余冻结。** 它不再像 Make-A-Video 那样在像素级搞级联，而是直接在图像 LDM 的 U-Net 里，在每个空间层之后插入一个时间层（temporal layer，通常用 3D 卷积或时间注意力实现），然后**只对这批时间层做 fine-tuning**，空间层的图像先验完全复用。核心洞察可以用一句话概括：

> **图像 LDM 已经学好了「什么是好画面」，视频生成只需要补上「时间怎么动」。**

下图展示了 fine-tuning 前后 LDM 潜空间样本分布的变化：

![](assets/视频生成模型发展与架构演进%20-%20Survey/figures/03-videoldm-temporal-finetuning.png)

> 图 3：VideoLDM 的 Temporal Video Fine-Tuning。左半是 fine-tuning 前的 LDM 潜空间——同一 batch 的样本在不同 batch/video frame 轴上是独立的，右半是 fine-tuning 后——沿时间轴对齐，形成连贯的视频序列。这一变化本质上是把「独立图像」转成「时间一致视频」。  
> 来源：论文 Figure 2，p.2，https://arxiv.org/abs/2304.08818

VideoLDM 的关键贡献有三：

- **复用图像 LDM 的大规模训练**：无需从头训练视频生成器，只需在预训练 LDM 上加时间层，把训练成本压到极致——这一「冻结空间、只训时间」的策略成为 2023 年所有视频扩散工作的模板；
- **潜空间视频超分**：把 LDM 的潜空间上采样沿时间维度扩展，从而生成 1024×2048 的高分辨率长视频——即「先在低分辨率潜空间生成、再在潜空间里做时间+空间超分」，绕开了像素级超分的显存瓶颈；
- **应用聚焦自动驾驶**：用 nuScenes 等真实驾驶数据 fine-tuning，是视频生成第一次进入自动驾驶仿真场景，也由此发展出后来的视频世界模型（如 GAIA-2）。

VideoLDM 的局限是它本质上仍是 **2D + 时间的「拼装」结构**——时间层是「外挂」在 2D U-Net 上的，时空耦合程度有限；图像 LDM 的能力上限也直接限制了视频生成的上限（如果基座 LDM 对某类物体画不好，视频也动不好）。但它的工程意义巨大：证明了「在已有图像扩散模型上低成本添加时间维度」可行，奠定了 2023 年所有视频扩散工作的基础范式。

### 3.3 Imagen Video 与级联视频扩散

几乎同时，Google 的 Imagen Video（Ho et al., 2022）选择了与 VideoLDM 相反的另一条路：**完全从头训练一个级联视频扩散系统，不用任何图像基座**。它把视频生成拆成一个 7 级级联（一个基础视频扩散模型 + 一系列时空/空间超分模型），每一级负责把分辨率往上抬一档，最终把视频推到 1280×768 高分辨率。

Imagen Video 的贡献是**证明「级联视频扩散」这条路能走通**——用「低分辨率建全局 + 高分辨率补细节」的分工，规避了「一次性生成高清视频」的显存爆炸问题。它的代价也最直观：7 个扩散模型串起来训练，单次训练任务需要数百 TPU-days，是当时算力堆叠最极致的视频生成工作，几乎不可复现。

这条「多级级联」路线在 2024 年之后被「端到端大 DiT」逐步取代——因为 DiT 配合 3D VAE 后，单模型就能吃下多分辨率，级联的工程复杂度不再划算。但 Imagen Video 留下的「低分辨率先行、逐级超分」思想，仍被后来很多长视频系统沿用。

### 3.4 AnimateDiff：可插拔的运动模块

2023 年 7 月，Guo et al.（AnimateDiff）提出了一个非常工程化、但极有影响力的思路：**把视频生成变成「在已有 T2I 模型上加一个可插拔的 motion module」**。要理解它，先看它解决的问题——2023 年社区里每个人都在用 LoRA、DreamBooth 微调出个性化的 T2I 模型，但想让这些个性化模型「动起来」，你没法给每个 LoRA 都配一套视频训练数据。

**AnimateDiff 的解法：把「运动」和「风格」彻底解耦。** 它设计一个轻量的 motion module——一组插在 U-Net 各层的 temporal Transformer 层（含时间位置编码 + 时间自注意力 + 前馈层），先在**大规模无标签视频**上训练这个 motion module（学「物体该怎么动」），同时**冻结 T2I 的所有空间权重**。训练完成后，这个 motion module 就成了一个「运动插件」：推理时把它插进**任意一个已经个性化微调过的 T2I 模型**里，那个 T2I 就能直接生成动画，无需针对该个性化模型重新训练。

**关键细节：Domain Adapter 对齐。** 个性化 T2I 模型的潜空间分布和「训练 motion module 时用的基座 T2I」不完全一致，直接插入会导致风格漂移。AnimateDiff 加了一个轻量的 domain adapter（一个小的归一化/缩放层），在推理时把 motion module 的输出对齐到目标个性化模型的特征分布。这样既保留了目标模型的视觉风格，又不破坏 motion module 学到的运动先验。

AnimateDiff 的关键洞察是 **「运动先验」与「图像风格先验」可以分离**——这也是它能火遍 Civitai 等社区的原因：任何定制化的 T2I LoRA 都可以插入 AnimateDiff 直接做动画，「个性化 + 视频化」第一次被解耦成两个独立可插拔的模块。

它的局限是 motion module 容量有限、时间一致性依赖较短的 attention window（通常只看相邻若干帧），长视频生成仍需外挂 frame interpolation、video extension 模块。但它把视频生成「民主化」到了 T2I 社区，是工程上极重要的一步。

**这一节的遗留问题（留给下一篇）：** 扩散路线证明了可行，但骨干仍是 U-Net（VideoLDM、AnimateDiff 都是 2D U-Net 加时间层），时空耦合仍是「拼装」而非「原生」。**真正统一的时空处理**要等 DiT 出现后才能实现。同时，「无配对数据」虽然解决了数据稀缺，却让模型失去了视频专属知识——怎么把两者结合，是 2024 年 Sora 等工作的核心议题。

---

## 4. 骨干革命：DiT 与扩散 Transformer（2023）

### 4.1 DiT：U-Net → Transformer

2022 年底，Peebles & Xie（UC Berkeley / NYU）发表了 DiT（Scalable Diffusion Models with Transformers, arXiv 2212.09748, ICCV 2023），**首次系统性地证明 Transformer 可以完全取代 U-Net 作为扩散模型的骨干**。虽然 DiT 本身是为图像设计的，但它带来的骨干范式直接决定了 2024 年后视频生成的走向，所以值得把它的机制讲透。

DiT 的核心架构拆成三件事：

**1. Patchify——把潜变量切成 token。** 输入是潜空间表示 $z \in \mathbb{R}^{I \times I \times C}$（$I$ 是潜空间边长，$C$ 是通道数）。DiT 用一个线性层把它切成 $T = (I/p)^2$ 个 patch token，每个 token 维度 $d$。$p$ 是 patch 大小——$p$ 越小，token 数越多、序列越长、计算量越大。这一步是「把图像变成语言模型可消费的 token 序列」的关键，也是后面视频生成做「spacetime patch」的直系祖先。

**2. N 个 DiT Block。** 每个 block 就是标准 Transformer 块：LayerNorm → Multi-Head Self-Attention → LayerNorm → Pointwise FeedForward。没有 U-Net 的下采样/上采样/跳跃连接，纯粹靠自注意力的全局感受野。

**3. adaLN-Zero 做条件注入（DiT 最重要的设计）。** 扩散模型需要把「时间步 $t$」和「类别 $y$（或文本）」注入网络。DiT 对比了三种方式，最终选定 **adaLN-Zero**：先用一个 MLP 把 $t$、$y$ 编码成每层的 scale/shift 参数，再对 LayerNorm 输出做仿射变换：

$$\mathrm{adaLN}(h) = \big(1 + \gamma\big) \odot \mathrm{LN}(h) + \beta, \quad (\gamma, \beta) = \mathrm{MLP}(t, y)$$

其中 $\odot$ 是逐元素乘，$(\gamma, \beta)$ 是每层独立学习的缩放/平移，且**初始化为 0**（所以叫 Zero）——训练初期网络退化为「无条件的恒等映射」，条件信号逐步「长」出来，训练更稳定。实验证明 adaLN-Zero 比 cross-attention 和 in-context conditioning 效果都更好。

下图展示了 DiT 的整体架构和三种 conditioning 变体：

![](assets/视频生成模型发展与架构演进%20-%20Survey/figures/04-dit-architecture.png)

> 图 4：DiT 架构与三种 conditioning 变体。左侧是 Latent Diffusion Transformer 整体（Patchify → N 个 DiT Block → Linear & Reshape 出 $\epsilon$ 和 $\Sigma$），右侧是 DiT Block 的三种条件注入方式：adaLN-Zero（左二）、Cross-Attention（右二）、In-Context Conditioning（最右）。adaLN-Zero 表现最好，是 DiT 的默认选择。  
> 来源：论文 Figure 3，p.3，https://arxiv.org/abs/2212.09748

DiT 最核心的发现是一条 **scaling 曲线**：**「GFlops 越大，FID 越低」**——扩散 Transformer 像语言模型一样有清晰、单调、可预测的 scaling 规律。参数从 0.6B 扩到 1.5B 再到 3B，ImageNet 256×256 的 FID 从 9.62 一路降到 2.27，且远未饱和。这条曲线是 DiT 之后整个领域敢把模型推到 10B+ 的**理论依据**——你不再需要「猜」模型够不够大，只需要「按算力预算堆上去」。

DiT 为视频生成带来了三个直接影响其走向的属性：

- **原生时空扩展性**：Transformer 的自注意力天然可沿时间维度扩展（只要把时空 token 化）；
- **Scaling 友好**：基于 Transformer 的扩散模型可以享受 LLM 的全部训练基础设施（Flash Attention、序列并行、ZeRO 等）；
- **条件注入灵活**：adaLN 让条件注入非常干净，方便在视频生成里塞文本、图像、运动、相机轨迹等多模态条件。

### 4.2 视频 DiT 的两条路径

DiT 之后，视频生成领域出现了两条把 Transformer 应用到时空的路径：

**路径 A：先扩 U-Net，再换 DiT。** 代表是 VideoLDM（2023）、AnimateDiff（2023）——它们都是 2D U-Net 加时间层，时空仍不统一。DiT 出现后，这些工作的后继者（Stable Video Diffusion、CogVideoX）都转向了 DiT 路线。

**路径 B：直接做时空 DiT。** 跳过「图像 DiT + 时间模块」的拼装思路，直接把视频视为「时空 token 序列」，用 Transformer 在这个序列上做 full attention。这就是 Sora 的核心选择，也是 2024 年之后视频生成的主流。

路径 B 的关键是**「时空 patch 化」**——把视频切成时空维度上的统一 patch，每个 patch 作为一个 token。这是下一节的核心。

**这一节的遗留问题（留给下一篇）：** DiT 给视频生成提供了骨干，但表示问题仍未解决：怎么把视频切成一个「适合 DiT 全注意力」的 token 序列？VQ-VAE 太粗（丢细节），2D VAE 切片缺时间（要拼装），3D VAE（下一节）是答案。

---

## 5. 时空统一：3D VAE + 大规模 DiT（2024–2025）

### 5.1 Sora 与 spacetime patches 范式

2024 年 2 月，OpenAI 发布 Sora（"Video generation models as world simulators"），虽然没有放出训练细节和论文，但官方技术报告透露了四个关键设计，其中第一个是决定性的：

**1. 时空 patch 化（spacetime patching）。** Sora 把视频视为「时空维度上的 3D 立方体」，直接切成 spacetime patches 作为 Transformer 的输入 token。这是对 DiT「patchify」的直系推广——图像 DiT 把 2D 潜变量切成 2D patch，Sora 把 3D 时空潜变量切成 3D patch。**关键后果是：视频在 DiT 里和图像毫无区别**——只要时空统一切成 patch，DiT 就能用同一套架构、同一套注意力处理视频，彻底告别「图像基座 + 外挂时间模块」的拼装时代。

**2. 统一的视频生成器。** 同一个 Transformer 同时支持文本到视频、图像到视频、视频到视频，且天然支持不同分辨率、宽高比、时长（因为 patch 化后 token 序列长度随输入自适应，无需固定尺寸）。这种「原生多模态、原生多尺度」的能力，是之前所有 U-Net 级联路线都不具备的。

**3. 大规模训练。** 训练数据规模未披露，但从生成质量（运动连贯性、物理合理性、长镜头一致性）推断是千万到亿级视频片段量级。

**4. 作为「世界模拟器」的定位。** 报告标题直接把视频生成和世界建模画等号——这也是本综述第 8、9 章「视频生成与世界模型合流」判断的直接源头。

Sora 的核心贡献是把视频生成从「时空拼装」推进到「时空原生」。**spacetime patches 让「表示」这一环彻底统一**，随之「骨干」（全注意力 DiT）和「范式」（扩散）也都收敛到同一套配方。这把视频生成从「在图像模型上外挂时间模块」彻底解放出来，是 2024 年后整个领域的事实标准。

> **Sora 原图位置**：本综述未能本地化 Sora 报告中的 spacetime patches 原图，因为 OpenAI 官方报告页（https://openai.com/index/video-generation-models-as-world-simulators/）的图片是 React 动态加载的（WebFetch 抓取不到 img src）。原图请在该页面 "Turning visual data into patches" 一节查看。

### 5.2 CogVideoX：3D Causal VAE + Expert Transformer

2024 年 8 月，清华 + 智谱的 CogVideoX（Yang et al., ICLR 2025）是**第一个开源复现 Sora 路线**的工作，把 Sora 语焉不详的三个关键设计明确落到了工程上。

**第一，3D Causal VAE。** 此前视频生成用的多是「2D VAE + 时间层」的拼装。CogVideoX 训练了一个**真正的 3D VAE**：用 CausalConv3D 在空间和时间维度上**同时**压缩，且时间维度是「因果」的（卷积核只向未来方向滑动，不偷看未来帧）。压缩比为 $(T+1) \times 8 \times 8$（时间维度不强制对称），把视频压成 $(T/4+1) \times H/8 \times W/8$ 的潜变量。这一设计带来两个直接收益：**潜空间的 token 数比「2D VAE 逐帧切片」少约 4 倍**（显存压力大降），且**帧间因果**让潜变量天然适合自回归 / 长视频逐段扩展。

**第二，Expert Transformer with Expert AdaLN。** 这是对 DiT 的 adaLN 的进一步特化。CogVideoX 设计了一个 Expert Transformer：把文本 token 和视频 token 在不同 stage 里，用**分组的 expert adaptive LayerNorm**（text 和 video 各用一套独立的 LN 参数）做条件融合——本质上是「文本条件注入」与「视频生成」在深层特征上的「expert 化」解耦。同时它用 **3D 全注意力**（而非 2D 空间注意力 + 1D 时间注意力分离）直接对时空 token 做 full attention，与 Sora 同源。

**第三，多分辨率帧打包训练。** 把不同分辨率、不同帧率的视频打包进同一个 batch 训练（配合 3D RoPE 位置编码），提升数据利用率、增强多尺度泛化。这一招也被 HunyuanVideo 沿用。

CogVideoX 的开源（5B 和 2B 两个版本）让中文社区第一次有了可用的 Sora 同路线基座。它的局限是规模相对小（5B）、时长有限（10 秒 16fps @ 768×1360），但已经证明了「3D Causal VAE + 全注意力 DiT」可以工业级落地，为后续 HunyuanVideo 等更大规模复现铺平了路。

### 5.3 HunyuanVideo：完整工业化管线

2024 年 12 月，腾讯的 HunyuanVideo（Kong et al., arXiv 2412.03603）把 Sora 路线推到 13B 参数规模，是当时最大的开源视频生成模型。它给出的不是「一个模型」，而是**一套完整的工业化管线**——数据预处理、3D VAE 设计、DiT backbone、训练基础设施、后训练应用，五件事各有一个明确答案。

下图展示了 HunyuanVideo 的总体架构：

![](assets/视频生成模型发展与架构演进%20-%20Survey/figures/05-hunyuanvideo-overall.png)

> 图 5：HunyuanVideo 总体架构。视频经 Causal 3D VAE Encoder 压缩为时空潜变量，文本经 Large Language Model 编码为 text embedding，两者拼接成 token 序列后由 HunyuanVideo Diffusion Backbone（全注意力 DiT）处理，最后经 Causal 3D VAE Decoder 解码回视频。这是与 Sora 同源的「3D VAE + DiT」现代视频生成完整管线。  
> 来源：论文 Figure 5，p.5，https://arxiv.org/abs/2412.03603

HunyuanVideo 的三个关键设计：

- **Causal 3D VAE**：与 CogVideoX 同思路，但压缩比更激进——$4 \times 8 \times 8$，单帧 token 数比「2D VAE 切片」少 64 倍。它同时用「重建损失 + 感知损失 + GAN 判别 + KL 正则」联合训练 VAE，在极致压缩下保住纹理细节；
- **双流 → 单流融合 DiT**：早期 layer 用 **dual-stream**（text token 和 video token 各走一路注意力，分别建模），后期 layer 用 **single-stream**（text 和 video 拼接后走统一注意力），平衡「文本语义对齐」与「时空一致性」；
- **图像-视频联合训练**：先用海量图-文对预训练文本编码器和 DiT 的视觉能力，再用视频-文对 fine-tune 时空能力。**这一步是稳定训练 13B 视频模型的关键**——直接上视频数据容易训崩，图像先验是「垫脚石」。

下图展示了 HunyuanVideo 的训练系统视角——数据预处理 → 模型训练 → 应用：

![](assets/视频生成模型发展与架构演进%20-%20Survey/figures/06-hunyuanvideo-training-system.png)

> 图 6：HunyuanVideo 的总体训练系统。Data Pre-processing（Data Filtering + Structured Captioning）→ Model Training（Generation Algorithm + Distributed Infrastructure + Scaling Law）→ Applications（Image-to-Video、Upper-body Talking Avatar、Whole-Body Avatar Generation）。这是把视频生成从「单模型竞争」推向「工业化流水线」的标志性视角。  
> 来源：论文 Figure 3，p.3，https://arxiv.org/abs/2412.03603

HunyuanVideo 的意义在于**证明「3D VAE + 大规模 DiT + 工业化数据治理」是视频生成的成熟配方**。它也是本综述把 HunyuanVideo Figure 5 选为「时空统一表示」代表图的原因——它是当前可下载的、最完整的 Sora 路线复现。

### 5.4 Movie Gen：流匹配 + 多任务统一

2024 年 10 月，Meta 的 Movie Gen（Polyak et al., arXiv 2410.13720）是「时空统一时代」里在**范式**上另起炉灶的代表。它的两个关键贡献：

**第一，全面采用流匹配（Flow Matching）。** 前文所有扩散工作都在 DDPM/LDM 框架下做「逐步去噪」，而 Movie Gen 换成了流匹配：把生成过程建模成「噪声到数据」的一条**直线 ODE**，模型直接学这个直线流的速度场 $v_\theta$：

$$x_t = (1-t)\,x_0 + t\,x_1,\qquad \mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\,x_0,\,x_1}\Big[\big\| v_\theta(x_t, t) - (x_1 - x_0)\big\|_2^2\Big]$$

其中 $x_1$ 是目标数据、$x_0$ 是噪声、$t$ 是时间。相比 DDPM 的随机游走，流匹配的直线轨迹**训练更稳定、采样步数更少、收敛更快**——在规模化训练里这些优势被进一步放大。这也是为什么 2025 年之后视频生成开始更多采用 Flow Matching / Rectified Flow（如 HunyuanVideo-Sparse、LongLive 系列）。

**第二，多任务统一生成。** Movie Gen 在**同一个 Transformer backbone + 同一套流匹配 loss** 下，同时支持 T2V、Video Personalization（参考图像保持人物身份）、Instruction-Guided Video Editing、Video-to-Audio。统一架构 + 统一 loss，让一个模型处理多种视频生成任务，是视频生成从「单点工具」走向「通用视频基础模型」的重要一步。

Movie Gen 的局限是模型规模（30B）和训练数据规模都极大，但 Meta 没有完全开源（只放出推理 demo）。它与 HunyuanVideo 的开放策略形成对比——前者闭源展示工业能力，后者开源让社区复现。

### 5.5 三个里程碑的对比

Sora、CogVideoX、HunyuanVideo、Movie Gen 共同定义了 2024–2025 的「时空统一时代」。它们的共同点是：

- **3D VAE 做时空压缩**（不再拼接时间模块）；
- **全注意力 DiT 做时空生成**（而非局部时空注意力）；
- **大规模数据 + 工业化训练**（数据治理、分布式训练、scaling law）。

它们的差异主要在工程实现细节：是否开源、是否开源训练代码、是否支持流匹配、训练数据规模和模型规模。这种「方法统一 + 工程分化」的状态，正是视频生成进入「基础模型时代」的标志。

**这一节的遗留问题（留给下一篇）：** 时空统一 + 大 DiT 已经把视频生成的视觉质量推到了「可用」水平。但真正的瓶颈已经从「能不能生成」转向了「能不能长、能不能快、能不能稳」——这是系统与工程问题，下一章讨论。

---

## 6. 横向对比：表示、骨干、范式

下面这张表把本综述涉及的核心工作按三条主线横向展开。注意分类是「主要路线」，很多工作是多范式混合的：

| 工作 | 时间 | 表示 | 骨干 | 范式 | 训练数据 | 时长/分辨率 |
|---|---|---|---|---|---|---|
| VideoGPT | 2021.04 | VQ-VAE 离散 token | GPT 自回归 | 自回归 | 配对（BAIR） | 4 帧、64× |
| Phenaki | 2022.10 | C-ViViT 离散 token | 双向 masked Transformer | 自回归 | 无配对视频 + 图像-文本 | 分钟级 |
| Make-A-Video | 2022.09 | T2I 基座 + 时间模块 | 2D U-Net + 时间层 | 扩散 | 仅图像-文本，无视频-文本 | 16 帧 64× → 768× |
| Imagen Video | 2022.10 | 时空 U-Net | 3D U-Net | 级联扩散 | 配对 | 1280×768 |
| VideoLDM | 2023.04 | LDM 潜空间 + 时间层 | 2D U-Net + 时间层 | 扩散 | LDM 图像 + 无标签视频 | 1024×2048 |
| AnimateDiff | 2023.07 | 图像 LDM + motion module | 2D U-Net + temporal Transformer | 扩散 | 大规模无标签视频 | 16 帧 256× |
| DiT | 2022.12 | 2D VAE + patchify | DiT（全注意力） | 扩散 | ImageNet 配对 | 256× / 512×（图像） |
| Sora | 2024.02 | spacetime patches + 3D VAE | DiT（全注意力） | 扩散 | 大规模视频-文本（未披露） | 多分辨率、宽高比 |
| CogVideoX | 2024.08 | 3D Causal VAE | Expert Transformer | 扩散 | 大规模视频-文本 | 10s @ 768×1360 |
| HunyuanVideo | 2024.12 | Causal 3D VAE | Dual-Stream → Single-Stream DiT（13B） | 扩散 | 大规模视频-文本 + 图像-文本 | 多分辨率 |
| Movie Gen | 2024.10 | 视频 VAE + 时空 patch | DiT（30B） | 流匹配 | 大规模视频-文本 | 多分辨率 + 多任务 |

可以读出的几条规律：

**表示线在 2024 年彻底转向 3D VAE + 时空 patch。** 在此之前的 2D VAE + 时间层是工程妥协，之后的 3D VAE + DiT 是「原生统一」。这个转变的拐点是 Sora 的发布（2024.02），CogVideoX 和 HunyuanVideo 在 2024 下半年开源复现，把这一路线固化为标准。

**骨干线在 2023–2024 完成从 U-Net 到 Transformer 的切换。** 视频领域的 DiT 切换比图像晚了约一年，主要因为视频的 token 数更大，DiT 的 quadratic 复杂度对显存的压力更重。3D VAE 压缩（时空 token 数减 4–64 倍）是 DiT 能在视频领域落地的关键前提。

**范式线在 2024 年底开始向流匹配迁移。** Movie Gen 用 Flow Matching 取代 DDPM 是有意义的信号——流匹配的直线轨迹在大型模型训练中更稳定、更高效，2025 年之后的视频生成（HunyuanVideo-Sparse、LongLive 等）开始更多采用流匹配。但扩散（特别是 rectified flow）仍然是主流，两者在工程上差异不大。

---

## 7. 系统与长视频：从「能用」到「好用」

### 7.1 长视频的根本挑战

第 5 章的所有工作都只能生成**短片段**（5–10 秒）。把生成时长推到分钟级甚至小时级，至少有三道坎：

- **显存墙**：时长 × 分辨率 = token 数，全注意力 DiT 的显存是 token 数的二次方。16s@256p 有 ~10K tokens，1min@720p 有 ~150K tokens，1h@1080p 有 ~5M tokens——后者对当前显存是不可行的；
- **漂移累积**：自回归生成长视频时，前一段的微小误差会随时间累积，导致「场景突变」「主体变形」；
- **训练算力**：长视频的时空一致性监督需要大算力，长视频对梯度稳定性的要求远高于短视频。

### 7.2 LongLive-2.0：NVFP4 + 长视频系统

针对这些挑战，本 vault 已收录的 [[LongLive-2.0 - NVFP4长视频生成训练与推理系统]]（NVIDIA, 2025）给出了一种**系统级而非模型级**的解法：

- **Balanced Sequence Parallelism（Balanced SP）**：把长视频 token 序列在多个 GPU 上做负载均衡的切分，绕过单卡显存墙；
- **NVFP4 4-bit 浮点训练**：把训练精度从 FP8/FP16 降到 NVFP4，在几乎不损失精度的前提下把训练吞吐提升 2.15 倍；
- **Shot-level SFT**：用镜头切换作为分段信号，让长视频的「场景」自然分段，避免单主体长程漂移。

LongLive-2.0 的意义是**把长视频生成从「模型问题」重新定义为「系统问题」**。它本身不是一个新模型，而是给现有长视频模型（如 Self-Forcing、Wan）配套的训练/推理基础设施。这种「模型 + 系统」的协同优化，正是视频生成进入「工业化可用」阶段的标志。

### 7.3 长视频方案的更多技术线

除 LongLive-2.0 外，主流长视频路线还有：
- **滑动窗口 + 帧重叠**：用 overlap-blend 在窗口之间做平滑过渡，推理时随时间推进；
- **递归 / 自回归扩展**：先生成短视频，再以前一段末尾帧为条件继续生成；
- **分层时序**：把视频生成拆成「关键帧 + 中间帧插值」两阶段，类似 Make-A-Video 的级联思路；
- **World Model 路线**：从世界模型视角生成长视频——Cosmos、GAIA-2 都是这条路（详见 [[世界模型演化 - Survey]]）。

这些路线并不互斥，现代长视频系统通常组合多种方案（LongLive-2.0 就是 LongLive 模型 + Balanced SP + NVFP4 + Shot SFT 的组合）。

---

## 8. 当前正在攻坚的问题

### 8.1 物理一致性与因果

视频生成的「好看」和「真」之间仍有巨大鸿沟。当前的 Sora、HunyuanVideo 在「玻璃碎裂」「水流入容器」「布料折叠」等中等物理交互上仍然会出现明显错误——物体穿模、重力方向不一致、刚体变软体。这不是「再加数据」能解决的，而是「模型需要内化物理规律」的问题。这正是 [[视觉世界模型 - Survey]] 和 [[世界模型演化 - Survey]] 强调的方向——视频生成正与「世界模型」合流。

### 8.2 从「视频生成」到「视频基础模型」

CogVideoX、Movie Gen 的多任务统一预示着视频生成正走向「视频基础模型」——同一模型同时支持 T2V、I2V、视频编辑、视频补全、视频超分、视频理解等。这种「一个模型、一个 loss、多种能力」的状态，是图像基础模型（CLIP、GPT-4V）的自然延伸。

### 8.3 实时与交互式生成

Sora、LongLive 等已经能把视频生成推到接近实时（~24 FPS），但距离「用户在交互式游戏、机器人仿真里实时生成视频」仍有差距。这需要模型压缩、token 化、缓存、并行采样等多方面的协同。

### 8.4 长视频的「剧情」一致性

5–10 秒内的「镜头级」一致性已经被解决，但分钟级视频的「剧情级」一致性（人物动机、情节递进、视觉风格统一）仍是开放问题。这需要把视频生成从「生成器」升级为「叙事生成器」，涉及 planning、记忆、role consistency 等更广义的能力。

---

## 9. 局限与评价

### 9.1 本综述的方法论局限

1. **未单独精读 Phenaki 和 Imagen Video 的原文。** 涉及这两篇工作的章节以二手综述 + 官方报告为依据，方法细节可能不完整；
2. **Sora 原图未本地化。** OpenAI 官方报告页是 React 动态渲染，WebFetch 无法抓取，本综述未单独下载 spacetime patches 原图，相关讨论以文字描述为主；
3. **覆盖时间不完整。** 本综述写到 2025 上半年（HunyuanVideo、Movie Gen、LongLive-2.0），2025 下半年到 2026 上半年的最新进展（如 Wan 2.1、HunyuanVideo-Sparse、LongLive 系列等）仅在第 7 章简略带过；
4. **主观判断标注明确。** 第 1、5、8 章中的趋势判断是综述者基于公开论文的推断，不构成同行评审结论。

### 9.2 视频生成领域的整体判断

把五年串起来，最值得强调的两个判断：

**「表示 + 骨干 + 范式」三条线同步演化，没有单点胜负。** GAN 输给自回归、自回归输给扩散、扩散正在被流匹配侵蚀——每一种「胜利」都同时改写了表示、骨干、范式三件事。把任一线单独拿出来讲演化史，都会丢掉另外两条线的影响。这正是本综述坚持「三主线并讲」的原因。

**视频生成正与「世界模型」合流。** Sora 报告直接以 "Video generation models as world simulators" 为题，把视频生成定位为「世界模拟」的副产品。本 vault 的 [[视觉世界模型 - Survey]] 和 [[世界模型演化 - Survey]] 从另一个角度印证了同一趋势——视频生成的下一步不是更大的模型或更长的视频，而是「能模拟物理、能保持状态、能被条件控制的视频」。这条线的合流点是 2025–2026 年视频生成最重要的演化方向。

### 9.3 后续值得跟踪的分叉

- **3D 一致视频生成**：与 [[GaussianDream - 机器人操控的前馈三维高斯世界模型]] 这类显式 3D 世界模型的结合，把视频生成从「2D 时空」推到「3D 时空」；
- **VLA / WAM**：本 vault 已收录 [[DreamZero]]、[[Dyna-2]]、[[Fast-WAM]] 等世界动作模型，把视频生成变成机器人策略；
- **物理约束生成**：借鉴 [[SymTRELLIS - 对称约束的体素潜变量三维生成]] 这类「在生成中注入结构约束」的工作，让视频生成也具备「物理一致性约束」；
- **多模态统一**：与 [[三维多模态大语言模型架构 - Survey]] 这类多模态 LLM 工作的合流，把视频作为 MLLM 的输出模态之一。

---

## 10. 关键文献索引

### 10.1 本综述重点涉及的论文

| 工作 | 出处 | 链接 | 角色 |
|---|---|---|---|
| VideoGPT | arXiv 2104.10157 | https://arxiv.org/abs/2104.10157 | VQ-VAE + 自回归开山 |
| Phenaki | arXiv 2210.02399 | https://arxiv.org/abs/2210.02399 | 长视频自回归（双向 masked transformer） |
| Make-A-Video | arXiv 2209.14792 | https://arxiv.org/abs/2209.14792 | 时空分解 + 无配对数据（图 2） |
| Imagen Video | arXiv 2210.02303 | https://arxiv.org/abs/2210.02303 | 级联视频扩散 |
| VideoLDM | arXiv 2304.08818 | https://arxiv.org/abs/2304.08818 | 图像 LDM + 时空层（图 3） |
| AnimateDiff | arXiv 2307.04725 | https://arxiv.org/abs/2307.04725 | 可插拔 motion module |
| DiT | arXiv 2212.09748 | https://arxiv.org/abs/2212.09748 | 扩散 Transformer（图 4） |
| Sora | OpenAI 技术报告 | https://openai.com/index/video-generation-models-as-world-simulators/ | spacetime patches 范式定义 |
| CogVideoX | arXiv 2408.06072 | https://arxiv.org/abs/2408.06072 | 3D Causal VAE + Expert Transformer（开源） |
| HunyuanVideo | arXiv 2412.03603 | https://arxiv.org/abs/2412.03603 | 完整工业化管线（图 5、图 6） |
| Movie Gen | arXiv 2410.13720 | https://arxiv.org/abs/2410.13720 | 流匹配 + 多任务统一 |

### 10.2 本 vault 已有独立笔记的相关工作

| 工作 | 笔记 | 关联章节 |
|---|---|---|
| LongLive-2.0 | [[LongLive-2.0 - NVFP4长视频生成训练与推理系统]] | 第 7 章 长视频系统 |
| （视觉世界模型相关） | [[视觉世界模型 - Survey]] | 第 8 章 视频生成与世界模型合流 |
| （世界模型演化相关） | [[世界模型演化 - Survey]] | 第 8 章 视频生成与世界模型合流 |

### 10.3 直接前置工作（图中引用但未单独精读）

- TGAN（2016）、MoCoGAN（2017）、DVD-GAN（2019）：GAN 时代视频生成
- VQ-VAE（van den Oord et al., 2017）、ImageGPT（Chen et al., 2020）：VideoGPT 的直接前置
- LDM / Stable Diffusion（Rombach et al., 2022）：VideoLDM 的直接前置
- T2I Adapter、ControlNet 等：与 AnimateDiff 同期个性化 T2I 工具

---

*本综述基于 8 篇 arXiv 论文 PDF + 1 篇 OpenAI 官方报告的细读整理，正文引用的 6 张关键架构图均裁剪自对应论文原文，已在本目录 `sources.md` 中留存溯源。Sora 报告因 OpenAI 页面为 React 动态渲染，原图未本地化，统一以 HunyuanVideo Figure 5 作为可下载的同源架构代表。涉及 Phenaki、Imagen Video 的章节以二手综述 + 官方报告为依据，未单独精读原文。*