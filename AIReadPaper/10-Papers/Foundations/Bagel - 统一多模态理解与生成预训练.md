---
type: paper-note
status: draft
domain: Foundations
paper: Emerging Properties in Unified Multimodal Pretraining
year: 2025
arxiv: 2505.14683
doi: null
source: https://arxiv.org/abs/2505.14683
project: https://seed.bytedance.com/bagel
code: https://github.com/bytedance-seed/BAGEL
tags: [Foundations, MoT, Multimodal, Unified-Understanding-Generation, Interleaved-Pretraining]
created: 2026-08-27
updated: 2026-08-27
---

# BAGEL：统一多模态理解与生成的涌现式预训练

## 基本信息

- **作者**：Chaorui Deng*、Deyao Zhu*、Kunchang Li*、Chenhui Gou*、Feng Li*、Zeyu Wang、Shu Zhong、Weihao Yu、Xiaonan Nie、Ziang Song、Guang Shi（通讯）、Haoqi Fan（项目负责人）等（* 共同一作）
- **机构**：ByteDance Seed，联合中国科学院深圳先进技术研究院、莫纳什大学、香港科技大学、UC Santa Cruz
- **发表**：arXiv 2505.14683（v3，2025-07-27）
- **项目页**：<https://seed.bytedance.com/bagel>
- **代码**：<https://github.com/bytedance-seed/BAGEL>
- **权重**：<https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT>
- **原文 PDF**：<https://arxiv.org/pdf/2505.14683>

## 一句话结论

BAGEL（Bottleneck-free, All-Generative and Emergent Learner）用「无瓶颈 MoT 双专家架构 + 双编码器（SigLIP2 ViT 语义 + FLUX VAE 像素）+ Next-Group-of-Token 预测（理解走 CE、生成走 Rectified Flow 速度预测）」在数万亿交错多模态 token 上做 decoder-only 预训练，7B 激活/14B 总参数即拿下 MME 2388、MMBench 85.0、MMMU 55.3、MM-Vet 67.2、MathVista 73.1、GenEval 0.88，并涌现出经典图像编辑、自由形式智能编辑（IntelligentBench 44.9，Self-CoT 55.3）、未来帧预测、3D 操控与世界导航等能力，成为理解-生成统一模型的开源标杆，也是 BagelVLA 与 π0.7 的生成底座。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

统一多模态理解与生成（unified multimodal understanding and generation）在 GPT-4o、Gemini 2.0 等闭源系统上已经展示出惊人的能力，但学术界与开源社区始终存在一个「gap」：开源统一模型的生成质量和推理能力远落后于闭源系统，而闭源系统的核心技术路线秘而不宣。BAGEL 要回答的核心问题是：**这个 gap 的根源是什么？** 论文的答案是——**数据，尤其是「精心构造的多模态交错数据」（multimodal interleaved data）的规模与质量**。以往的统一模型（Show-o、Janus、Chameleon、Emu3 等）大多只在标准的图文对数据（image-text paired data）上训练，而这类数据天然缺乏跨模态、跨时间、跨上下文的长程交互信号。BAGEL 的出发点是把文本、图像、视频、网页四种来源融合成数万亿 token 的交错序列，观察统一预训练规模扩大后模型是否会「涌现」出理解与生成之外的更高级能力。

### 1.2 统一模型的三种设计路线与瓶颈

论文把现有统一模型的设计空间归纳为三条路线，并逐一点评其瓶颈：

- **Quantized AR（离散自回归）**：用离散视觉 tokenizer（如 VQGAN、FSQ）把图像转成离散 token，统一走 Next-Token-Prediction。优点是能直接复用 LLM 基础设施、实现简单；缺点是视觉生成质量经验上劣于扩散模型，且自回归的串行特性导致推理延迟高。
- **External Diffuser（外挂扩散器）**：把预训练的 LLM/VLM 通过轻量可训练 adapter 连到扩散模块（如 SEED-X、DreamLLM、MetaQuery），LLM 自回归生成一组 latent token 作为「语义条件」，再交给扩散模块生成图像。优点是收敛快、数据需求小、benchmark 上可能不错；**致命缺点是「瓶颈」**——LLM 的整个上下文被压缩成少量 latent token，理解与生成模块之间存在显式信息瓶颈，在长上下文多模态推理中会有大量信息丢失，与「大基础模型 scaling」的哲学相悖。
- **Integrated Transformer（一体化 Transformer）**：把自回归 Transformer 和扩散 Transformer 融合进同一个模型，共享架构、无缝切换两种范式（如 Transfusion、LlamaFusion、JanusFlow）。训练算力需求远高于 External Diffuser，但换来的是**全程无瓶颈的上下文交互**，理解与生成模块之间无损互动，更适合大规模 scaling。

BAGEL 的选择是第三条：无瓶颈的一体化 Transformer。论文的核心论断是，只有无瓶颈设计才能让「数据规模扩大」的信号不被架构约束所掩盖，从而催生涌现能力。

### 1.3 本文核心贡献

1. 提出 BAGEL——一个 **7B 激活/14B 总参数**的 decoder-only 统一模型，在数万亿交错多模态 token 上预训练，理解与生成 natively 统一；
2. 设计**无瓶颈 MoT（Mixture-of-Transformer-Experts）架构**：两个全尺寸 Transformer 专家（理解专家处理文本 + ViT token，生成专家处理 VAE token）通过**共享多模态自注意力**在每一层交互，不存在任何 connector 瓶颈；
3. 采用**双编码器**：SigLIP2 ViT 负责语义理解、FLUX VAE 负责像素级生成，二者各自专注最擅长的事；
4. 建立一套**可扩展的交错数据获取协议**（视频 + 网页 + reasoning-augmented），并系统揭示**涌现性质**：理解/生成 → 基础编辑 → 复杂智能编辑的能力按阶段涌现，智能编辑在 3T token 后才显著跃升；
5. 开源代码与权重，成为后续 BagelVLA、π0.7 等工作的生成底座。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

BAGEL 不是为单一任务训练的模型，而是一个统一的多模态「接口」——同样的模型既可以做理解（VQA、OCR、图表、数学），也可以做生成（文生图、多图、视频片段），还可以做两者交叉的编辑、操纵与世界建模。因此它的输入输出是异构的：

- **输入**：交错的多模态序列，由文本 token、ViT 语义 token（来自 SigLIP2 编码的图像）与 VAE 像素 token（来自 FLUX VAE 编码的图像/视频）按模态结构交错排列；生成任务还附带 diffusion timestep 条件。
- **输出**：理解任务输出下一段文本 token（语言回答）；生成任务输出一组视觉 token（去噪后的 VAE latent，经 VAE decoder 解码为图像/视频帧）。
- **核心假设**：理解与生成虽然优化目标部分解耦，但可以在同一个 Transformer 内通过共享自注意力共同探索参数空间；只要无瓶颈、数据足够多且交错，二者不仅不冲突，反而会「协同」出跨模态组合推理能力。

### 2.2 关键符号与目标函数

BAGEL 把训练目标统一表述为「下一组 token 预测」（Next-Group-of-Token Prediction）——对语言 token 走经典的 Next-Token-Prediction，对视觉 token 走 Rectified Flow。两部分损失线性加权：

$$\mathcal{L} = \lambda_{\text{CE}}\,\mathcal{L}_{\text{CE}} + \lambda_{\text{MSE}}\,\mathcal{L}_{\text{MSE}}$$

**理解目标（交叉熵，Next-Token-Prediction）**：对文本 token 序列，最大化下一 token 的对数似然：

$$\mathcal{L}_{\text{CE}} = - \frac{1}{N}\sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i})$$

**生成目标（Rectified Flow / velocity prediction）**：对视觉 token，遵循 Rectified Flow [19, 41, 45] 的业界最佳实践，学习从噪声到数据的直线路径上的速度场。设干净 latent 为 $\mathbf{z}_0$，噪声为 $\boldsymbol{\epsilon}$，插值潜变量为 $\mathbf{z}_t = (1-t)\,\mathbf{z}_0 + t\,\boldsymbol{\epsilon}$，目标速度 $\mathbf{v} = \boldsymbol{\epsilon} - \mathbf{z}_0$：

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{\mathbf{z}_0, \boldsymbol{\epsilon}, t}\left\| \mathbf{v}_\theta(\mathbf{z}_t, t, c) - (\boldsymbol{\epsilon} - \mathbf{z}_0) \right\|^2$$

其中 $c$ 是文本/ViT/clean-VAE 等条件，$t$ 是 diffusion timestep（以 timestep embedding 直接加到 VAE token 的初始 hidden states 上，而非 DiT 惯用的 AdaLN）。论文最终取 $\lambda_{\text{CE}} : \lambda_{\text{MSE}} = 0.25 : 1$，即生成损失的权重是理解损失的 4 倍，这与「生成数据应采样更多」的经验结论一致（见第 4.4 节）。

---

## 3. 核心方法

### 3.1 总体框架

BAGEL 的骨架是一个 decoder-only Transformer，初始化自 **Qwen2.5 LLM**（沿用其 RMSNorm、SwiGLU、RoPE、GQA，并在每个注意力块加入 QK-Norm 以稳定训练）。在这个骨架上做两件事：**（1）挂两个视觉编码器**——SigLIP2 ViT（理解）与 FLUX VAE（生成）；**（2）把 Transformer 扩展成双专家 MoT**——理解专家处理文本 + ViT token，生成专家处理 VAE token，二者在每一层共享多模态自注意力。整个过程无任何 connector 瓶颈，理解与生成的上下文全程无损交互。

![](assets/Bagel%20-%20统一多模态理解与生成预训练/figures/02-architecture.png)

> **图 2：BAGEL 的 MoT 双专家架构。** 左上是文本 tokenizer，右上两个编码器分别输出 ViT 语义 token（理解）与 VAE 像素 token（生成）；中间每个 Transformer block 内含两个专家（Und. Expert 与 Gen. Expert），各自有独立的 QKV 与 FFN，但共享同一个「Multi-modal Self Attention」；底部两种输出头分别对应语言回答（Next Token Prediction / CE）与图像/多图/视频片段（Velocity Prediction / MSE）。这张图点出了 BAGEL 设计的核心——**解耦的是参数，共享的是注意力**。
> 来源：论文 Figure 2，第 4 页，<https://arxiv.org/abs/2505.14683>

### 3.2 双编码器：SigLIP2 ViT + FLUX VAE

BAGEL 用两个编码器分别负责「语义」与「像素」，各司其职：

- **理解编码器（语义）**：采用 **SigLIP2-so400m/14** 作为 ViT 初始化，原始固定 384 分辨率。先插值位置编码把最大输入尺寸扩到 $980 \times 980$，再集成 **NaViT** 以支持任意原生宽高比；ViT token 经一个两层 MLP connector 对齐到 LLM 的 hidden 维度。这样理解侧可以吃高分辨率、原生比例的图，充分保留 OCR、图表、定位等细粒度信息。
- **生成编码器（像素）**：用 **FLUX 预训练 VAE** 在像素空间与 latent 空间之间往返。latent 下采样倍率为 8、通道数 16，随后经一个 $2\times 2$ patch embedding 进一步压缩空间尺寸并对齐 hidden 维度。VAE 全程**冻结**，只作为不可训练的编解码器。

两种 token（ViT 与 VAE）在进入 LLM 前都加 2D 位置编码。论文强调这个双编码器分工的价值：ViT 提供**高层语义上下文**（对编辑、智能编辑尤其关键，见第 5.1 节「去掉 ViT 导致 Intelligent Edit 掉 16%」的消融），VAE 提供**低层像素细节**（对高保真生成关键）。

### 3.3 MoT 双专家架构（Dense vs MoE vs MoT）

这是 BAGEL 最重要的架构决策。论文在 1.5B Qwen2.5 LLM 上做了严格受控实验，唯一变量是 Transformer 变体：

- **Dense**：单一稠密 Transformer，文本/视觉 token 全走同一套参数；
- **MoE**：只把每个 block 里的 FFN 复制一份作为「生成专家」的初始化（参数只增 FFN）；
- **MoT**：把 Qwen2.5 LLM 的**全部可训练参数复制一份**，得到全尺寸的生成专家（参数翻倍）。

三种变体都采用**硬路由**：新复制的生成专家**只处理 VAE token**，原始参数（理解专家）**只处理文本 + ViT token**（沿用 Qwen-VL 系列的策略）。关键点是：**三者总参数量虽不同（MoE/MoT 约为 Dense 的两倍），但训练与推理的 FLOPs 完全一致**——因为硬路由下每个 token 只激活一个专家。

结论（Figure 3）非常清晰：**MoT 在生成（MSE）和理解（CE）两个损失上都最优**，差距在生成任务上最明显——MSE 平滑单调下降，MoT 不仅收敛最快、终值也最低；CE 因交错异构数据而波动更大，但 MoT 整体仍最好。论文的解读是：理解与生成两个目标会把模型推向**参数空间的不同区域**，只有给它们分配独立的容量（full-size 专家）才能缓解模态间优化目标相互竞争的问题；只复制 FFN 的 MoE 容量不够，效果介于两者之间。

### 3.4 广义因果注意力（Generalized Causal Attention）

统一模型的最大工程难点在于：交错序列里既有自回归的文本（要因果掩码），又有扩散的视觉（要双向/组内全注意力），还要处理「生成图 A → 再生成图 B」的跨图条件依赖。BAGEL 设计了**广义因果注意力**来统一处理：

对每个样本，token 先被划分成多个连续 split，每个 split 只含单一模态（text / ViT / VAE）；**某 split 的 token 可以 attend 到前面所有 split 的 token**；split 内部，文本走因果注意力、视觉走双向注意力。针对生成样本里的每张图，准备三组视觉 token：

- **Noised VAE tokens**：加了扩散噪声的 latent，只用于 Rectified-Flow 训练，MSE loss 只算在这组上；
- **Clean VAE tokens**：无噪声的原始 latent，作为后续图像/文本 token 的生成条件；
- **ViT tokens**：SigLIP2 编码的语义 token，用于统一交错理解/生成数据的输入格式，且经验上能提升交错生成质量。

规则是：后续的图像或文本 token 可以 attend 到前面图像的 clean VAE 与 ViT token，但**不能 attend 到 noised VAE token**。对交错多图生成，采用 **diffusion forcing** 策略——不同图加独立噪声，每张图以前面图的「噪声表征」为条件；同时随机把连续图分组、组内 full attention、组内噪声级别相同，以增强生成一致性。

工程上，论文用 **PyTorch FlexAttention** 实现这套自定义掩码，比朴素 scaled-dot-product attention 快约 2 倍。推理时，广义因果结构允许只缓存 clean VAE 与 ViT token 的 KV；一张图生成完毕后，上下文中对应的 noised VAE token 被替换为其 clean 版本，从而实现多模态解码加速。

### 3.5 推理与 Classifier-Free Guidance

推理阶段沿用扩散模型的 classifier-free guidance（CFG）。为在交错推理中实现 CFG，训练时随机 drop 三类条件 token：**text drop 概率 0.1、ViT drop 概率 0.5、clean VAE drop 概率 0.1**。ViT 的 drop 概率最高（0.5），呼应了「ViT 是可选的语义增强而非必需条件」这一观察——但恰恰是这层语义在复杂编辑任务里价值巨大（见 5.1）。推理时生成的视觉 token 由 VAE decoder 解码回像素，语言 token 由 LLM 头解码为回答。

---

## 4. 数据集与实验设置

### 4.1 数据组成与规模

「数据定义了大基础模型的知识边界」是本文的方法论基石。BAGEL 的训练语料横跨语言、图像、视频、网页四种模态，规模与构成总结在 Table 1：

| 数据来源 | 数据量（M） | Token 数（T） |
|---|---|---|
| 纯文本 | 400 | 0.4 |
| 图文对·理解 | 500 | 0.5 |
| 图文对·生成 | 1600 | 2.6 |
| 交错·理解 | 100 | 0.5 |
| 交错·生成：视频 | 45 | 0.7 |
| 交错·生成：网页 | 20 | 0.4 |

其中「交错多模态数据」是 BAGEL 区别于所有前作的关键（表中灰色高亮部分）。值得注意的是，纯文本只占 0.4T，图文对生成占 2.6T 的大头，而交错视频 + 交错网页合计 1.1T——正是这 1.1T 的交错数据，催生了后续的涌现能力。

### 4.2 交错数据的构建协议

**（a）从视频构建交错数据**。视频是「最大的自然模拟器」，天然提供像素级、概念级、时序与物理连续性信号，对图像编辑、导航、3D 操控尤其有效。数据源包括公开在线视频 + 两个开源集（Koala36M 的指令/交互内容、MVImgNet2.0 的多视角空间理解）。流程是：先按 T2V 流水线做镜头切分、去黑边/去 logo/去字幕、质量过滤、CLIP 相似度去重；然后用一个**蒸馏自大 VLM 的轻量 captioner**（基于 Qwen2.5-VL-7B 微调）生成相邻帧之间的变化描述——captions 限长 30 token 以抑制幻觉；每段 clip 平均采样 4 帧、为每对相邻帧生成 caption，最终得到 **4500 万条时序对齐的交错序列**。

**（b）从网页构建交错数据**。网页承载复杂的真实多模态结构（图解百科、分步教程等）。BAGEL 基于 **OmniCorpus**（从 Common Crawl 预处理的 100 亿级图文交错语料），借鉴 DeepSeekMath 做两阶段主题选择：LLM 标注少量样本 → 训练 fastText 分类器做大规模粗筛 → 再用 LLM（Qwen2.5-14B）细过滤，叠加 UI 剔除、分辨率、清晰度、文本密度（OCR 超 100 token 的文档剔除）、相关性、文档修剪、图像数量（保留 3–8 图）等规则过滤。之后用 **caption-first 策略**：每张图前插入 Qwen2.5-VL-7B 生成的概念性描述作为「脚手架」，超 300 token 的图间文本用 LLM 摘要器重写，最终得到 **2000 万条结构化交错网页文档**。

**（c）Reasoning-Augmented 数据**。受 O1/DeepSeek-R1 启发，构造 **50 万条带推理轨迹的生成样本**，覆盖 T2I、自由形式操纵、概念编辑几类。例如 T2I 用 Qwen2.5-72B 生成「query → guidance + 详细 prompt」配对、再交给 FLUX.1-dev 生成目标图；自由形式操纵用 R1 的 reasoning trace 为 source/target 图对构造指令；概念编辑走三阶段 VLM 流水线（选对 → 出题 → 质量评估 → 生成 grounded 解释）。这套数据让模型学会「先想再画」，是 Self-CoT 能显著涨点的直接来源。

### 4.3 训练策略：四阶段 Recipe

BAGEL 采用多阶段训练，动态混合上述数据（Table 3）：

- **Alignment（对齐）**：只训 MLP connector，把 SigLIP2 ViT 对齐到 Qwen2.5 LLM，视觉/语言主干冻结；只喂图文对做 captioning，图 resize 到固定 378×378；5K steps、4.9B token、LR 1e-3（cosine）。
- **Pre-training（PT）**：给 LLM 加 QK-Norm，除 VAE 外全部参数可训练；**2.5T token**，含文本/图文对/多模态对话/网页交错/视频交错；理解与生成都用原生分辨率（生成短边 256~512、理解短边 224~980）；200K steps、LR 1e-4（constant）、CE:MSE = 0.25:1、EMA 0.9999、max context 16K、timestep shift 1.0。
- **Continued Training（CT）**：提升视觉分辨率（生成 512~1024、理解 378~980），**策略性提高交错数据采样比例**以强化跨模态推理（此时理解/生成核心能力已趋稳定）；**约 2.6T token**；100K steps、max context 40K、timestep shift 4.0。
- **Supervised Fine-tuning（SFT）**：生成侧从图文对 + 交错生成集构造高质量子集，理解侧从 LLaVA-OV 与 Mammoth-VL 指令集过滤子集；**72.7B token**、15K steps、LR 2.5e-5、EMA 0.995。

全阶段用 AdamW（$\beta_1=0.9, \beta_2=0.95$，$\epsilon=1\times 10^{-15}$ 以抑制 loss spike），PT/CT/SFT 都用**常数学习率**以便无需重启即可扩展数据规模；每个 rank 的序列被 pack 到窄长度区间（PT 32K–36K、CT/SFT 40K–45K）以保证负载均衡。

### 4.4 关键超参消融：采样比例与学习率

统一预训练有两个必须小心调的超参——数据采样比例与学习率。论文在 1.5B Qwen2.5 上做了受控实验：

- **采样比例（Figure 5）**：把生成:理解从 1:1 提到 4:1，MSE loss 持续下降（4g1u 最低，绝对降幅约 0.4%，对 rectified-flow 模型是相当大的提升）；CE loss 则无一致规律（最大差距 0.07，对下游 benchmark 影响可忽略）。结论：**生成数据应显著多于理解数据**，最终采用约 4:1。
- **学习率（Figure 6）**：两个目标对 LR 的反应相反——LR 大 → MSE 收敛快；LR 小 → CE 更优。论文的调和方式是给两个目标分配独立的 loss 权重（即 0.25:1），而非给两组参数分 LR。

### 4.5 评价指标与 Benchmark

- **理解**：MME、MMBench (1.0-EN)、MM-Vet、MMMU、MathVista、MMVP 六个基准，覆盖感知/认知/多模态推理，对 SOTA 排序有强区分力；
- **文生图**：GenEval（物体/计数/颜色/位置/属性对齐）+ WISE（世界知识语义评估，分文化/时间/空间/生物/物理/化学六类）；
- **图像编辑**：GEdit-Bench（真实用户请求，GPT-4.1 自动打分，分 EN/CN 两套）；
- **智能编辑**：论文自建 **IntelligentBench**——350 例（问题图 + 问题文本 + 参考答案图），用 GPT-4o 对完整四元组打分（0–2 分/题），考察请求满足、视觉一致性、知识驱动创造性，最终归一化到百分制，专测「需要复杂多模态推理与任务组合」的自由形式操纵。

---

## 5. 实验结果

### 5.1 涌现性质：理解/生成 → 基础编辑 → 智能编辑

这是全文最核心的发现。论文给「涌现」下了一个聚焦定义：**一种能力若在早期训练阶段不存在、而在后期预训练中出现，即为涌现**（即 phase transition，无法从 loss 曲线外推）。为验证，作者在历史 checkpoint 上系统性评测，并以「达到峰值性能 85% 所需的 seen token 数」作为涌现时点指标（Figure 7）：

- **理解**在 **0.18T** token 即达到 85%（饱和最早）；
- **生成**（GenEval）在 **0.68T** 达到 85%；
- **基础编辑**（GEdit）需 **2.64T**（收敛更慢，因为它同时依赖理解与生成）；
- **智能编辑**（IntelligentBench）需 **3.61T**，且呈现「初期低、3T 后显著跃升（15→45，翻三倍）」的典型涌现曲线——3T 前模型倾向于「原样复制输入图」这种不理解任务时的 fallback 策略，3.5T 后才开始真正推理。

关键消融：**去掉 ViT token 对 GEdit-Bench 影响很小，但使 Intelligent Edit 掉 16%**——证明视觉语义推理（ViT 提供的语义上下文）是复杂编辑能力的来源，而非像素细节。

![](assets/Bagel%20-%20统一多模态理解与生成预训练/figures/03-emerging.png)

> **图 8/9：不同训练 token 量模型的定性对比。** 上排 Figure 8 是 T2I 与经典编辑——生成质量在 1.5T 前已很强，高分辨率（3.0T 后）带来小幅提升，「hello/BAGEL」的**正确文字渲染**则在 1.5T–4.5T 之间才涌现。下排 Figure 9 是智能编辑——3.5T 之前模型基本照抄输入（fallback），3.5T 之后开始产出语义正确、符合指令的新内容（如把针织项目补全、把角色画成惊讶表情）。这张图是「涌现不是 loss 曲线能预测的」最直观的视觉证据。
> 来源：论文 Figure 8/9，第 14–15 页，<https://arxiv.org/abs/2505.14683>

### 5.2 图像理解

BAGEL 7B MoT 在六个理解基准上全面领先同规模统一模型，并超越多数专用理解模型（Table 4）：MME-S 2388、MMBench 85.0、MMMU 55.3、MM-Vet 67.2、MathVista 73.1、MMVP 69.3。相对同为统一模型的 Janus-Pro-7B，MMMU 与 MM-Vet 分别高出 14.3 与 17.1 分；相比专用理解标杆 Qwen2.5-VL-7B（MMMU 58.6、MM-Vet 67.1），BAGEL 在多数指标上不落下风。论文据此论证：MoT 设计有效缓解了理解与生成的任务冲突，统一模型不必牺牲理解能力。

### 5.3 图像生成（T2I 与 WISE）

- **GenEval**（Table 5）：BAGEL 达 **0.88**（用 LLM rewriter），不用 rewriter 也有 0.82——超越专用生成模型 FLUX.1-dev（0.82）、SD3-Medium（0.74），也超越统一模型 Janus-Pro-7B（0.80）与 MetaQuery-XL（0.80）。生成质量可对标一流专用文生图模型。
- **WISE**（Table 6）：BAGEL 0.52，**加 Self-CoT 后跃升至 0.70**，超过此前所有开源模型（前 SOTA MetaQuery-XL 0.55），仅次于 GPT-4o（0.80）。这说明世界知识驱动的文生图能靠「先想后画」大幅增强。
- 定性上（Figure 10），BAGEL 原生支持中英文 prompt 与任意宽高比，图像质量显著高于 Janus-Pro-7B 与 SD3-medium。

### 5.4 图像编辑与智能编辑

- **GEdit-Bench**（Table 7）：BAGEL 在 EN 集上 G_SC 7.36 / G_PQ 6.83 / G_O 6.52，与最强专用编辑模型 Step1X-Edit（7.09/6.76/6.70）相当，并超过 Gemini 2.0；CN 集同样领先。
- **IntelligentBench**（Table 8）：BAGEL **44.9**，远超开源 Step1X-Edit（14.9）30 分；**加 Self-CoT 后 55.3**，逼近 Gemini 2.0（57.6）、低于 GPT-4o（78.9）。这 30 分的差距最能说明「智能编辑」是统一模型 + 交错预训练独有的能力，专用编辑模型因缺乏多模态推理而几乎完全失败。
- **思考模式**（7.4 节）：显式 CoT 不仅提升 IntelligentBench（44.9→55.3），还在 WISE（0.52→0.70）、RISEBench（6.1→11.9）、KRIS-Bench（56.21→60.18）上一致增益，证明 reasoning-augmented 数据让模型习得了可泛化的「先推理后生成」。

### 5.5 世界建模与失败案例

**世界建模（7.5 节）**：通过在训练 recipe 中提高视频与导航数据比例做微调，BAGEL 展现出世界导航、旋转、多帧生成能力——能按指令生成动态数量的图像（前进/左看/右看/相机旋转 9°），并用 ParticleSfM 标注相机轨迹构造导航数据。更值得注意的是其泛化性：**只训了真实街景导航，却能无缝迁移到水墨画、卡通、游戏等领域**。

**失败案例（Figure 17）**：特殊 IP 生成、复杂文字渲染、复杂人体姿态、多实例同时生成仍是当代文生图系统（含 BAGEL、Gemini 2.0）的共性难题；编辑侧的物体位置交换、同时修改大量实例也普遍失败。GPT-4o 在这些场景下最稳定。论文给出的改进方向是：增加含文字的图像、扩大模型容量、或在后训练阶段引入 RLHF。

---

## 6. 与相关工作的关系

- **vs Quantized AR（Janus / Janus-Pro / Chameleon / Emu3）**：同属统一模型，但走离散 token + 自回归路线；BAGEL 用 Rectified Flow 的连续 latent，生成质量（GenEval 0.88 vs Janus-Pro 0.80）与推理速度都占优，且 Janus 系列把理解与生成编码器解耦（这与 BAGEL 的双编码器同构），但 Janus 仍在 transformer 内部共用一套稠密参数，缺乏 BAGEL 的专家解耦。
- **vs External Diffuser（SEED-X / DreamLLM / MetaQuery）**：这类方法靠 LLM 生成少量 latent 条件再外挂扩散器，收敛快但存在显式信息瓶颈；MetaQuery-XL 甚至直接冻结 Qwen2.5-VL 主干，可塑性受限。BAGEL 用无瓶颈一体化设计换取长上下文推理能力，这是 IntelligentBench 上 44.9 vs Step1X-Edit 14.9 巨大差距的架构根源。
- **vs Integrated Transformer（Transfusion / LlamaFusion / JanusFlow）**：BAGEL 与它们共享「一体化无瓶颈」哲学，但把 Dense 参数升级为 **MoT 双专家**——这是 BAGEL 在生成损失上对 Dense/MoE 都更优的关键（Figure 3），也是 LlamaFusion 等单一稠密 Transformer 未能验证的。
- **vs 专用模型（Qwen2.5-VL / InternVL2.5 理解，FLUX / SD3 生成）**：BAGEL 在理解上对标甚至超越专用 VLM，在生成上对标专用 T2I 模型，是「统一不牺牲单项」的代表。
- **下游作用：BagelVLA 与 π0.7 的生成底座**。BAGEL 的价值不止于自身 benchmark。它是 **BagelVLA 的骨干初始化来源**——BagelVLA 的 LLM 专家 7B + 生成专家 7B + FLUX VAE 这一组合正直接继承自 BAGEL 的 MoT 双专家结构与双编码器；同时 **π0.7 的轻量世界模型也基于 BAGEL**。理解 BAGEL，才能理解这两个后续机器人/世界模型工作的生成底座从何而来、为何选这种「理解专家 + 生成专家共享注意力」的拆分方式。

---

## 7. 局限与批判性评价

1. **涌现的证据链仍有软肋**：论文的「涌现」主要靠历史 checkpoint 评测 + 定性图对比，且 85% 时点指标（0.18T/0.68T/2.64T/3.61T）受 checkpoint 采样粒度影响；作者也承认评估指标未必线性反映真实能力，「spurious emergence」虽不太可能但无法完全排除。缺一套更严格的 scaling law 拟合（如 Dyna-2 那种）。
2. **算力与数据门槛极高**：数万亿 token 的多模态交错预训练 + 四阶段 recipe，是绝大多数实验室无法复现的「重工业」工程，开源权重固然好，但论文的增量贡献更多在「工程与数据协议」而非可轻量验证的算法思想。
3. **世界建模/视频能力被低估式呈现**：7.5 节的世界导航、旋转、多帧生成只有定性展示，没有定量 benchmark，且是「额外微调提高视频/导航比例」后的结果，与主预训练的涌现能力边界不够清晰。
4. **推理成本的隐性代价**：Self-CoT 能大幅涨点（IntelligentBench 44.9→55.3），但也意味着复杂任务需要在生成前产出长 reasoning trace，端到端延迟与 token 成本显著上升，论文未量化这部分开销。
5. **IntelligentBench 的评测主体是 GPT-4o**：自建基准 + 闭源 GPT-4o 打分，存在「裁判即天花板」的偏差（GPT-4o 自己拿 78.9 分），且 350 例的规模偏小，统计显著性存疑。
6. **与闭源仍有距离**：IntelligentBench（55.3 vs GPT-4o 78.9）、WISE（0.70 vs 0.80）、失败案例中 GPT-4o 的一致性，都说明 BAGEL 虽坐稳开源第一梯队，但距 GPT-4o 的复杂推理与生成一致性仍有明确 gap。
7. **可解释性缺失**：为什么「生成数据要 4 倍于理解数据」、为什么 MoT 优于 MoE，论文只给了经验曲线，未触及机制层面的解释，这为后续理论工作留了空间。

---

## 8. 复现与实践建议

- **权重与代码**：BAGEL-7B-MoT 权重在 HuggingFace（`ByteDance-Seed/BAGEL-7B-MoT`），代码在 `bytedance-seed/BAGEL`；这是最现实的「复现」路径——直接加载推理，而非重训。
- **若要做小规模验证**：论文的消融给了明确的缩水路线——用 **1.5B Qwen2.5** 做架构/超参探索（Dense vs MoE vs MoT、采样比例、学习率都在 1.5B 上做的），保持生成:理解 = 4:1、CE:MSE = 0.25:1、常数学习率。
- **架构实现要点**：
  - MoT 硬路由下**每个 token 只激活一个专家**，训练/推理 FLOPs 与 Dense 相同，但总参数量翻倍——显存主要瓶颈在参数存储而非计算；
  - 广义因果注意力用 **PyTorch FlexAttention** 实现，比朴素 attention 快约 2×，是必须的工程优化；
  - 每张生成图要准备 **noised VAE / clean VAE / ViT 三组 token**，MSE 只算在 noised VAE 上；
  - VAE **必须冻结**；ViT 在理解与编辑中价值大（去掉掉 16%），但 CFG 训练时 ViT drop 概率要设 0.5。
- **数据构建**：交错数据是灵魂——视频侧走「蒸馏 captioner + 帧间变化描述（限 30 token）」、网页侧走「OmniCorpus + 两阶段主题选择 + caption-first」，reasoning-augmented 用 R1 trace 造「先想后画」样本；纯文本少放（0.05 采样比例即可）。
- **常见踩坑**：统一训练中 CE 与 MSE 对学习率反应相反，不要试图用单 LR 同时优化好两者，应靠 loss 权重（0.25:1）+ 常数 LR 调和；分辨率提升时记得把 diffusion timestep shift 从 1.0 调到 4.0。

---

## 9. 个人启发与后续问题

1. **「无瓶颈」是 scaling 的前提而非锦上添花**：BAGEL 把 External Diffuser 的瓶颈问题点破——信息瓶颈会直接封死涌现的上限。这解释了为什么 MetaQuery 类方法 benchmark 不差却做不出智能编辑。对后续任何「统一模型」设计，这是第一条要过的架构关。
2. **交错数据才是统一模型的护城河**：图文对数据 2.6T，交错数据仅 1.1T，但恰恰是这 1.1T 催生了编辑与智能编辑。它提示我们：**数据的「结构」（interleaving 与 reasoning trace）比「总量」更关键**——这与 LLM 预训练中「数据质量 > 数量」的判断一脉相承，只是换到了多模态。
3. **涌现能力可以「有意识地」设计出来**：BAGEL 不是被动等涌现，而是用 reasoning-augmented 数据主动「喂」出了 Self-CoT 能力，再用自建 IntelligentBench 把它度量出来。这是一种「先定义能力 → 构造数据 → 验证涌现」的主动范式，比事后惊呼「涌现了」更工程化。
4. **作为 BagelVLA/π0.7 底座的意义**：BAGEL 的「理解专家 7B + 生成专家 7B + FLUX VAE + 共享注意力」这一拆分，正是 BagelVLA 的初始化来源；π0.7 的轻量世界模型也基于 BAGEL。这意味着 BAGEL 的涌现能力（未来帧预测、世界导航、3D 操控）可以直接被下游机器人策略继承——**一个多模态基础模型的「生成/世界建模」能力，正在成为 VLA 的新一代底座**，这与 VPP「预测性视觉表征」的思路殊途同归（VPP 是让视频扩散模型当编码器，BAGEL 是让统一模型直接当世界模型）。
5. **未解问题**：（i）涌现的 3T 阈值是架构决定的还是数据决定的？换 Dense 或 MoE 是否也能在更大数据下涌现？（ii）Self-CoT 的推理开销能否被蒸馏进隐式前向（不输出 trace 也能等价推理）？（iii）IntelligentBench 是否可用开源裁判（而非 GPT-4o）复现，以消除「裁判天花板」？（iv）BAGEL 的世界建模能力如何定量化，进而支撑 BagelVLA/π0.7 的闭环机器人策略？

---

## 10. 材料来源

### Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2505.14683> | 全文（37 页 + 附录） | 2026-08-27 | 精读主来源 |
| `figures/01-showcase.png` | Figure 1：BAGEL 多功能展示（理解/生成/编辑/未来帧预测/3D/世界导航） | arXiv:2505.14683 | 第 2–3 页 | 2026-08-27 | 概念总览 |
| `figures/02-architecture.png` | Figure 2：MoT 双专家架构（理解/生成专家 + 共享注意力） | arXiv:2505.14683 | 第 4 页 | 2026-08-27 | 核心方法 |
| `figures/03-emerging.png` | Figure 8/9：不同训练 token 量模型对比（scaling/涌现） | arXiv:2505.14683 | 第 14 页 | 2026-08-27 | 涌现性质 |
| 项目页 <https://seed.bytedance.com/bagel> | 项目主页/补充材料 | — | — | 2026-08-27 | 模型展示 |
| 代码 <https://github.com/bytedance-seed/BAGEL> | 官方实现 | — | — | 2026-08-27 | 复现参考 |
| 权重 <https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT> | 模型 checkpoint | — | — | 2026-08-27 | 直接加载推理 |
