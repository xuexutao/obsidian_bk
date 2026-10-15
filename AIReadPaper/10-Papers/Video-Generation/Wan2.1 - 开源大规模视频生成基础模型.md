---
type: paper-note
status: draft
domain: Video-Generation
paper: Wan: Open and Advanced Large-Scale Video Generative Models
year: 2025
arxiv: 2503.20314
doi: null
source: https://arxiv.org/abs/2503.20314
project: null
code: https://github.com/Wan-Video/Wan2.1
tags:
  - Video-Generation
  - Diffusion-Transformer
  - Flow-Matching
  - Open-Source
  - Foundation-Model
  - Wan
  - VAE
created: 2026-08-27
updated: 2026-10-15
---

# Wan2.1：开源大规模视频生成基础模型

## 基本信息

- **作者**：Wan Team（通义万相）
- **机构**：阿里巴巴
- **发表**：arXiv 2503.20314v1（2025-03-27），v2（2025-04-19）
- **项目/权重**：<https://github.com/Wan-Video/Wan2.1>
- **原文 PDF**：<https://arxiv.org/pdf/2503.20314>
- **模型规模**：1.3B（小）/ 14B（旗舰），均支持 480p T2V 与 I2V
- **训练数据规模**：数十亿图像 + 视频，约 O(1) 万亿 tokens
- **核心依赖**：Diffusion Transformer（DiT）+ Flow Matching + 时空因果 VAE（Wan-VAE）+ umT5 文本编码器

## 一句话结论

Wan2.1 是阿里通义万相在 2025 年 3 月开源的**视频基础模型套件**，它把「时空因果 VAE（Wan-VAE，127M）+ 全注意力视频 DiT（1.3B/14B）+ Flow Matching 训练 + 工业化数据治理 + 大规模分布式训练 + Wan-Bench 自建评测」拼成一条完整配方，在 Wan-Bench 和 VBench 上以开源姿态同时超过 Sora、Runway Gen-3、Hunyuan、Mochi、CogVideoX 等闭源/开源竞品，并把 1.3B 模型压到 8.19GB VRAM 即可运行，成为 2025 年视频生成「可复现、可商用、可二次开发」的公共底座。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

Sora 把视频生成带进公众视野之后，开源社区迅速跟进：HunyuanVideo、Mochi、CogVideoX 等工作把视频基础模型的代码和权重先后公开，让开源-闭源差距迅速缩小。但到 2025 年初，这股开源浪潮仍存在三个明显短板：第一，**性能**——闭源商业模型在画面质量、运动幅度、指令跟随上仍普遍领先开源一个身位；第二，**能力**——多数开源基座只能做通用 T2V，缺少 I2V、指令编辑、长视频等下游任务的一体化支持；第三，**效率**——主流 5B~13B 模型对显存要求很高，创意团队很难本地跑起来。Wan2.1 的出发点就是同时解决这三点。

### 1.2 现有方法的瓶颈

论文把此前的不足归纳为三方面：

- **架构选择分歧**：HunyuanVideo 用 Full Attention DiT，Mochi 用非对称 3D 注意力，CogVideoX 偏 3D 变分卷积 + Expert Transformer。每家在「时空注意力怎么排」上各做各的，缺乏可复现的工程模板；
- **数据治理标准缺失**：图像社区已经有 LAION-5B 美学过滤、OCR 过滤、CLIP 相似度过滤等成熟管线，视频社区一直没有等价物——WebVid-10M 偏小且水印严重，InternVid 只做粗筛；
- **训练策略不透明**：大部分工作只公开最终模型，**预训练数据如何清洗、训练怎样分阶段、并行与内存如何优化**这些决定性的工程细节几乎不写，学术界无法复现也无处下手。

### 1.3 本文核心贡献

1. **完整开源的工业级视频基础模型套件**：1.3B（轻量）和 14B（旗舰）两个版本，覆盖 T2V、I2V、指令编辑（V2V）、个性化、长视频等最多 8 个下游任务，**首次支持在视频中同时生成中英文视觉文本**；
2. **Wan-VAE**：一个 127M 参数的时空因果 3D VAE，4×8×8 压缩比，支持任意长视频的 chunked 编码/解码（feature cache 机制），重建质量与效率超过 HunyuanVideo、CogVideoX、SVD、Mochi、Open Sora Plan、Step Video；
3. **可复现的训练工程**：Flow Matching 训练目标、图像 256px 预训练 + 渐进式图像-视频联合训练、umT5 文本编码、2D Context Parallelism（Ulysses+Ring）+ FSDP、8-bit FlashAttention、Diffusion Cache 等一整套训练/推理栈，**全部开源**；
4. **完整的数据治理与 caption 管线**：四步数据清洗（基础维度 / 视觉质量 / 运动质量 / 视觉文本）+ 内部 dense caption 模型，覆盖「数据从哪儿来、怎么标、怎么过滤」全流程；
5. **Wan-Bench 自建评测**：14 个维度的细粒度评测（人类偏好加权），配合 1,035 条 prompt，覆盖大运动、物理合理性、相机控制、ID 一致性等 VBench 难以量化的能力。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与任务族

Wan2.1 不是单一任务模型，而是一套视频生成基础基座。论文展示的下游任务族包括：

- **文生视频 T2V**：输入文本提示 $\mathbf{c}_{\text{text}}$，输出时序一致的视频 $\hat{\mathbf{x}}_0 \in \mathbb{R}^{T \times H \times W \times 3}$；1.3B 与 14B 两种规模均支持 480p 推理；
- **图生视频 I2V**：输入一张静止条件帧 $\mathbf{x}_{\text{cond}}$ + 文本提示，输出以其为开头的视频；
- **指令引导视频编辑 V2V**：基于 VACE 框架，输入视频条件单元 $\mathcal{V} = [\mathbf{T}; \mathbf{F}; \mathbf{M}]$（文本 + 帧序列 + 掩码序列），支持局部重绘、扩展、外绘等编辑；
- **个性化生成**：输入参考人/物图像，输出保留身份的动态视频；
- **中英文视觉文本生成**：在视频画面内生成可读的中英文文字（这是 Wan2.1 的差异化卖点）；
- **长视频与实时生成**：通过 chunked 编码和 feature cache 支持任意长视频，配合 DiT streaming 输出可实时生成。

### 2.2 关键符号与扩散目标

Wan2.1 采用 Flow Matching 框架（Lipman et al., 2022; Esser et al., 2024）作为扩散目标的统一形式。给定干净潜变量 $\mathbf{x}_1$、随机噪声 $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，从 logit-normal 分布采样时间步 $t \in [0, 1]$，中间潜变量定义为线性插值（Rectified Flow）：

$$\mathbf{x}_t = t\,\mathbf{x}_1 + (1-t)\,\mathbf{x}_0, \quad t \sim p_{\text{logit-normal}}$$

对应的 ground truth 速度为：

$$\mathbf{v}_t = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0$$

模型 $u(\mathbf{x}_t, \mathbf{c}_{\text{text}}, t;\theta)$ 预测速度，训练目标为 MSE：

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0,\mathbf{x}_1,\mathbf{c}_{\text{text}},t}\left\| u(\mathbf{x}_t, \mathbf{c}_{\text{text}}, t;\theta) - \mathbf{v}_t \right\|^2$$

其中 $\mathbf{c}_{\text{text}}$ 是 umT5 文本编码器输出的 512 token 序列，$\theta$ 是 DiT 权重。推理时用 ODE 求解器从 $\mathbf{x}_0$ 积分到 $\mathbf{x}_1$，约 50 步采样（可结合 CFG 与 diffusion cache 加速）。

### 2.3 潜空间压缩

视频先经 Wan-VAE 编码到潜空间。对一段 $(1+T)$ 帧的视频，Wan-VAE 输出 $\mathbf{x} \in \mathbb{R}^{(1+T/4) \times H/8 \times W/8}$（默认 4×8×8 压缩比，1 帧条件 + T/4 潜帧）。DiT 再把 $\mathbf{x}$ patchify 成 token 序列：使用 kernel=(1,2,2) 的 3D 卷积后展平，序列长度 $L = (1+T/4) \times H/16 \times W/16$，潜维度 $D$。

---

## 3. 核心方法

### 3.1 整体架构

Wan2.1 的整体 pipeline 由四部分组成：**Wan-VAE（时空因果 VAE）+ 视频 DiT（Full Attention 视频 Transformer）+ umT5 文本编码器 + Flow Matching 训练目标**。视频通过 Wan-Encoder 压缩到潜空间，文本通过 umT5 编码为 token 序列，二者在 DiT 内通过 cross-attention 融合，DiT 输出速度场经 ODE 求解后由 Wan-Decoder 解码回像素空间。整体 pipeline 如图 9 所示。

![](assets/Wan2.1%20-%20开源大规模视频生成基础模型/figures/03-pipeline.png)

> **图 1：Wan2.1 整体架构（论文 Figure 9）。** 输入视频 $V \in \mathbb{R}^{(1+T)\times H\times W\times 3}$（条件帧 + T 帧视频）经 Wan-Encoder 压缩到潜空间 $\mathbf{x} \in \mathbb{R}^{1+T/4\times H/8\times W/8}$；DiT 在 Flow Matching 框架下以 timestep $t$ 和 umT5 文本嵌入（"A panda is holding up a cardboard that says 'Wan2.1'"）为条件，对加噪潜变量去噪；最后 Wan-Decoder 重建回视频。**图注**：架构极简但「Full Attention + Cross-Attention + Flow Matching + 因果 VAE」的组合是 Wan 与 SVD（UNet 路线）最大的架构分野。
> 来源：论文 Figure 9，第 13 页，<https://arxiv.org/abs/2503.20314>

DiT block 的内部结构（Figure 10）展示三个关键设计：

- **Full spatio-temporal self-attention**：在 3D token 序列上做完整自注意力，而不是 UNet 那种空间+时间分离注意力。这是 HunyuanVideo 验证过的「简单即正确」路线；
- **Cross-attention 嵌入文本**：与 Stable Diffusion 一致，但 token 序列更长（umT5 输出 512 token），需要 DiT 在长序列下保持稳定；
- **共享 MLP 预测 adaLN 参数**：用一个共享的 MLP（Linear+SiLU）从 time embedding 预测 6 个调制参数（scale、shift、gate 各两个），但每个 block 拥有独立的 bias。**这一设计比 PixelArt / PixArt-α 的「AdaLN-single」更彻底地把参数压缩到极致，最终在 1.3B 模型上比非共享版省约 25% 参数，并在同等参数下取得更低的训练 loss**（见 §5.4 消融）。

![](assets/Wan2.1%20-%20开源大规模视频生成基础模型/figures/05-dit-block.png)

> **图 2：Wan 视频 DiT block 结构（论文 Figure 10）。** V-Tokens（视频 token）经过 Layer Norm → Self-Attention → Layer Norm → Cross-Attention（T-Tokens 注入）→ Layer Norm → FFN；右侧 Timestep 通过 MLP 预测 adaLN 的 6 个调制参数并广播到所有 block。**关键设计点**：所有 block 共享同一个 timestep MLP，**每个 block 只学独立的 bias**，这是 Wan 在 1.3B 量级做到比非共享结构省 25% 参数且性能不退化的核心 trick。
> 来源：论文 Figure 10，第 14 页，<https://arxiv.org/abs/2503.20314>

### 3.2 Wan-VAE：因果时空 VAE

视频 VAE 是 Wan 能在 8GB 显存跑 1.3B 模型的关键，因为视频潜空间大小直接决定 DiT 的序列长度。Wan-VAE 的设计目标可以概括为四个：**(a) 强重建质量**、**(b) 强时序因果性**（支持任意长视频 streaming 编码/解码）、**(c) 小模型**（127M）、**(d) 高效推理**（feature cache 机制）。

#### 3.2.1 模型设计

- 沿用 MagViT-v2 的 3D 因果卷积 + 注意力结构，把 GroupNorm 全部替换为 RMSNorm，以保留时序因果性（GroupNorm 会跨帧归一化，破坏因果）；
- 空间上采样层通道减半 → 推理显存下降 33%；
- 精心调节 base channels 数后，最终模型只有 **127M 参数**，远小于 HunyuanVideo VAE 246M、CogVideoX VAE 182M、Mochi 215M、Open Sora Plan 239M、CVVAE 460M、Step Video 499M；
- 默认压缩比 **4×8×8**（1 帧条件 + T/4 潜帧，H/8 × W/8 空间压缩）；可灵活配置为 4×16×16 等组合。

#### 3.2.2 三阶段训练

1. **Stage 1：2D 图像 VAE**：用同样的结构先在图像上训练一个 2D 空间 VAE，作为空间压缩先验；
2. **Stage 2：3D 因果 VAE 从图像 inflate**：把 2D 权重膨胀为 3D 权重，在 128×128 / 5 帧低分辨率短视频上快速收敛；训练损失 = L1 重建 (3) + KL (3e-6) + LPIPS 感知损失 (3)；
3. **Stage 3：高质量视频 + GAN 微调**：在不同分辨率和帧数的高质量视频上微调，**额外引入 3D 判别器的 GAN loss**，进一步提升细节与时序一致性。

#### 3.2.3 Feature cache：任意长视频 streaming

Wan-VAE 的因果卷积支持 chunked 编码/解码，这是它能处理任意长视频的关键。视频按 1+T/4 拆成 chunk，每个 chunk 内做一次编码/解码，跨 chunk 通过维护 frame-level feature cache 保证时序连续：

- **(a) 默认 causal conv（kernel=3）**：维护前 2 帧的 feature 作为 cache，首 chunk 用 zero padding 初始化，后续 chunk 复用上一 chunk 最后两帧；
- **(b) 2× temporal downsample（stride=2）**：维护前 1 帧的 feature cache，保证下采样后的序列长度严格符合倍数关系。

这套 feature cache 让 Wan-VAE 可以处理无限长视频，且 chunk 边界处不会出现不连续。

#### 3.2.4 重建质量与效率

论文在 720×720 / 25 帧的设置下对比主流视频 VAE（Figure 7）：Wan-VAE 在 PSNR 维度达到 38.5+，**和 HunyuanVideo VAE 持平**，显著优于 CogVideoX（~36）、SVD（~33）、Mochi（~30）、Open Sora Plan（~30）、Step Video（~31）。在效率维度（frame/second latency），Wan-VAE 凭借 127M 的小体量也明显领先。

![](assets/Wan2.1%20-%20开源大规模视频生成基础模型/figures/02-vae-recon.png)

> **图 3：Wan-VAE 重建质量可视化（论文 Figure 8）。** 4 行场景：纹理（动物毛）、面部（漫威美队）、文本（《楚辞》中文）、高运动（汽车 + 字幕屏）。每行从左到右：GT + zoom-in + 多种对比方法的重建 + Wan 重建（蓝框）。**关键观察**：在面部嘴唇/眼睛周围、文本笔画结构、高运动场景的运动锐度三个最容易被 VAE 丢细节的场景下，Wan-VAE 重建（蓝框，最右一列）几乎保留 GT 的高频细节，而其他方法在红圈标注处出现明显模糊或畸变。这是 Wan-VAE 在 127M 小体量下 PSNR 仍领先的根本原因。
> 来源：论文 Figure 8，第 12 页，<https://arxiv.org/abs/2503.20314>

### 3.3 视频 DiT 细节

- **Patchify**：kernel=(1,2,2) 的 3D 卷积 + 展平 → 序列长 $L = (1+T/4)\times H/16 \times W/16$；
- **Transformer block**：Layer Norm → Self-Attention → Layer Norm → Cross-Attention → Layer Norm → FFN，N× 块堆叠；
- **文本嵌入**：用 **umT5（5.3B 参数）** 作为文本编码器，输出 512 token 序列。论文在消融中比较了 umT5、Qwen2.5-7B-Instruct、GLM-4-9B、Qwen-VL-7B-Instruct 四种文本编码器，umT5 在中英双语能力、长上下文组合性、收敛速度上均最优（详见 §5.4 消融）；
- **Timestep 调制**：共享的 MLP（Linear+SiLU）从 $t$ 预测 6 个 adaLN 参数（scale/shift/gate × 2），每个 block 拥有独立 bias；这一设计在 1.3B 模型上比 non-shared adaLN 省约 25% 参数，且在同等 1.5B 参数下比 non-shared 1.7B 训练 loss 略低；
- **分类器无关引导（CFG）**：推理时用 $\tilde{\mathbf{v}}(\mathbf{x}_t, \mathbf{c}) = \mathbf{v}(\mathbf{x}_t, \varnothing) + w(\mathbf{v}(\mathbf{x}_t, \mathbf{c}) - \mathbf{v}(\mathbf{x}_t, \varnothing))$，$w$ 通常 5~7。

### 3.4 Flow Matching 训练目标

Wan2.1 选用 **Rectified Flow（RF）**（Esser et al., 2024）而不是经典 DDPM，理由是：RF 用 ODE 描述连续时间生成过程，**数值稳定性更好、与 likelihood 目标等价**，且天然支持 logit-normal 时间步采样，让训练在低/高噪声区间都保持梯度平衡。公式见 §2.2。

### 3.5 训练策略：图像预训练 + 图视频联合

#### 3.5.1 图像预训练阶段

直接用 256px 文本-图像对做 T2I 预训练，目的是在进入视频联合训练前就让 DiT 学会**跨模态语义对齐和几何结构保真**。这一步的动机来自两个观察：（1）高分辨率长视频序列长度极大（1280×720 × 81 帧潜空间 token 超过 256K），直接联合训练会严重拖慢吞吐量；（2）GPU 显存被迫使用小 batch，**梯度方差大、训练不稳定**。先在 256px 图像上预训练可以让模型在进入视频阶段前已具备稳定表征。

#### 3.5.2 图像-视频联合训练（分辨率渐进课程）

完成 256px 图像预训练后，Wan2.1 用**三阶段分辨率渐进课程**做图视频联合训练：

- **阶段 1**：256px 图像 + 192px（16fps）5 秒短视频联合训练；
- **阶段 2**：分辨率提升（论文未在主体内披露具体数值，但强调「空间分辨率逐步升级」），同时延长视频时长；
- **阶段 3**：进一步提升分辨率与视频时长至目标设定。

这种「先低清预训练 + 渐进升清」的课程设计是 SDXL、SD3、HunyuanVideo 等大规模生成模型的通用范式，目的是让模型在不同分辨率 / 时长上都能保持稳定收敛。

### 3.6 文本编码器：umT5

Wan2.1 选用 mT5 的多语言升级版 **umT5** 作为文本编码器，论文在 §4.7.2 给出了详细的消融（详见 §5.4）。核心结论：

- **umT5 > Qwen2.5-7B-Instruct > GLM-4-9B**：尽管后两者是更强的 LLM，但 decoder-only 的 causal attention 不适合 diffusion 模型的 cross-attention；umT5 的双向 attention 在 T2I/T2V 任务上表现更好；
- umT5 收敛更快、显存更省（5.3B vs 7B+）；
- 与 Qwen-VL-7B-Instruct（多模态 LLM）相比，umT5 在 FID 上略胜或持平（43.01 vs 42.91 at 15k steps），但 umT5 显存小得多。

### 3.7 大规模分布式训练

Wan2.1 在阿里云集群上训练 14B 模型，依赖三套并行与内存优化技术：

#### 3.7.1 Workload 分析

Wan 的计算瓶颈在 attention：自注意力的计算量随序列长度 $s$ 平方增长，但显存随 $s$ 线性增长。这意味着在长序列场景下，计算量最终会超过 activation offloading 的 PCIe 传输时间，**优化重点是 attention 通信而非权重通信**。

#### 3.7.2 2D Context Parallelism（Ulysses + Ring Attention）

Wan 2D CP 是一种 hybrid 方案：

- **内层 Ulysses（sequence parallelism）**：在 attention head 维度切分 token，每张卡只算部分 head 的 QKV。优点是通信量小，缺点是跨机通信延迟高；
- **外层 Ring Attention**：在序列维度切分 token，环形传递 K/V。优点是可处理超长序列，缺点是当切分粒度大时需要较大的 block size；
- **2D 组合**：Ulysses 做内层减少跨机通信，Ring Attention 做外层支持长序列。论文报告在 256K 序列长度、16 GPU（2 机）配置下，**2D CP 通信开销从 Ulysses 的 10%+ 降至 1% 以下**。

#### 3.7.3 FSDP + DP

- DiT 参数 / 梯度 / 优化器状态超过单卡容量（1M token、batch=1 时 activation 达 8TB），用 FSDP 做参数分片；
- 文本编码器也用 FSDP（需要 20GB+ GPU 内存）；
- 最外层是 DP（数据并行），例如 128 GPU 配置：Ulysses=8, Ring=2, FSDP=32, DP=4，全局 batch = 8b。

#### 3.7.4 Activation offloading + Gradient Checkpointing

长序列场景下 CPU 内存也会耗尽，Wan 组合使用：

- **Activation offloading**：将 DiT 层 activation 卸载到 CPU 内存，计算与 PCIe 传输 overlap，1~3 层 DiT 的计算时间足以隐藏 1 层 offload 的传输时间；
- **Gradient checkpointing（GC）**：对显存/计算比高的层优先用 GC，offloading 与 GC 形成互补。

#### 3.7.5 集群可靠性

训练周期长达数周，依赖阿里云 PAI 平台的**智能调度 + 慢机检测 + 自愈**能力：训练启动前隔离故障节点，训练中故障节点自动隔离并恢复任务，训练自动从 checkpoint 续训。

### 3.8 推理优化

#### 3.8.1 并行策略

推理时 14B 模型用 FSDP 做参数分片（避免单卡显存溢出），并沿用 2D Context Parallel 做长序列切分。**2D CP + FSDP 几乎线性加速**（Figure 12）。

#### 3.8.2 Diffusion Cache

Wan 在 §4.4.2 分析了两个推理时的相似性：

- **Attention 相似性**：同一个 DiT block 内，不同采样步的 attention 输出高度相似；
- **CFG 相似性**：采样后期 conditional 与 unconditional 输出一致性变高。

基于这两点，Wan 用 attention cache（每隔 N 步算一次 attention、其余步复用）和 CFG cache（每隔 N 步算一次 unconditional DiT、其余步复用 conditional 输出）**几乎无损失地减少约 1/3~1/2 推理 FLOPs**。

#### 3.8.3 8-bit FlashAttention

Wan 在 NVIDIA H20 GPU 上自研了 8-bit FlashAttention，针对 attention 数值稳定性做了两点改进：

- **混合 Int8/FP8 量化**：$S = QK^\top$ 用 Int8，$O = PV$ 用 FP8（参考 SageAttention 的数据分布分析）；
- **FP32 跨 block 累积**：原生 FP8 WGMMA 用 14-bit accumulator，长序列会溢出；Wan 用 FP8 WGMMA 做 block 内 PV 累积，跨 block 用 CUDA core 的 FP32 register 累积（参考 DeepSeek-V3 的 FP8 GEMM）。

最终 8-bit FlashAttention 在 H20 上达到 **95% MFU**，相对原版 FP8 FA3 推理效率提升 **1.27× 以上**。

### 3.9 Prompt 对齐

为了让用户输入的简短 prompt 能像训练时的 dense caption 那样工作，Wan 做了两点 prompt 对齐：

- **多 caption 风格**：每张图/视频配长短不一、风格多样（正式 / 非正式 / 诗意等）的多份 caption，覆盖不同用户的输入风格；
- **Prompt Rewrite**：用一个 LLM 把用户简短 prompt 改写成训练时 dense caption 的分布，Table 1 展示了 Viking warrior、motorboat、Potato King 三个 case 的改写效果。

---

## 4. 数据集与训练配置

### 4.1 数据治理管线

Wan2.1 的数据治理是全文工程量最大的部分，分为**预训练数据**和**后训练数据**两套管线，目标是「**从噪声分布中过滤出可工业训练的高质量、多样化、合规的视频+图像库**」。

#### 4.1.1 预训练数据：四步清洗

**Step 1：基础维度过滤（fundamental dimensions）**

- **OCR 文本检测**：剔除含过多文字的图/视频（避免训练数据被字幕主导）；
- **美学分数（LAION-5B aesthetic classifier）**：剔除低美学图；
- **NSFW 分数**：内部安全模型过滤不当内容；
- **水印 / logo 检测**：检测到则 crop 掉；
- **黑边检测**：启发式方法自动裁剪；
- **过曝检测**：内部 expert classifier；
- **合成图检测**：**即使 < 10% 的合成图污染也会显著降低模型质量**，所以专门训了一个 expert classifier 过滤；
- **模糊检测**：内部模型给 blur score 排序；
- **时长 & 分辨率约束**：视频 ≥ 4 秒；不同训练阶段用不同分辨率阈值。

**这套预处理合计过滤掉约 50% 原始数据**。

**Step 2：视觉质量筛选（visual quality）**

- 用 embedding 把数据聚成 **100 个 cluster**，按 cluster 采样人工评分（1-5 分）→ 训练 expert scoring model → 全量打分。
- **好处**：cluster 内采样避免长尾分布下小但重要的子类被一并过滤掉。

**Step 3：运动质量筛选（motion quality）**

按运动质量把视频分 6 档：

- **Optimal motion**：运动布局/视角/幅度显著，干净流畅 → 训练主力；
- **Medium-quality motion**：有运动但有遮挡/多主体 → 保留作为多样性补充；
- **Static videos**：聊天/访谈类，运动少但画质高 → 单独识别、降采样比；
- **Camera-driven motion**：航拍类相机主导运动 → 单独处理；
- 其余两档：被剔除或特殊用途。

**Step 4：视觉文本数据**

Wan2.1 强调中英文视觉文本能力，专门收集含中英文文字的视频（招牌、字幕、街头广告等），并在 caption 里标注文字内容。

#### 4.1.2 Post-training Data

预训练完成后用**少量高质量 prompt-视频对**做监督微调，caption 与训练时的 dense caption 同分布，目标是让模型对齐用户实际 prompt 风格。

### 4.2 Dense Video Caption

DALL-E 3 已经证明，**用 dense caption 训练可显著提升 prompt following 能力**。Wan 训练了一个内部 caption 模型生成 dense caption：

- **开放源数据**：caption 数据集（多种风格/长度）+ 视觉问答数据集（动作、计数、OCR 等）+ 纯文本指令数据（控制 caption 风格）；
- **内部数据**：
  - **名人 / 地标 / 影视角色**：用 LLM 收集名字，TEAM（CLIP 风格模型，对中文名人识别最好）从图像-文本库检索相关图，再用关键词匹配二次过滤；
  - **物体计数**：检索含 "one/two/three..." 的图，用 LLM 提 (category, count) 对，再用 Grounding DINO 计数验证；
  - **OCR 增强**：用 OCR 提取文字作为 prior，让 caption 模型基于 OCR 结果生成含准确文字的描述；
  - **相机角度与运动**：先标注一批视频的相机角度/运动，再训练一个 expert 模型去标注更多视频，**因为 GPT-4o 和 Gemini Pro 在这方面都很弱**；
  - **细粒度类别**（动物/植物/车辆）：数百万张图；
  - **关系理解**（左右上下）：复用现成 object detection 数据集；
  - **Re-caption / 编辑指令 / 群图描述 / 人工标注** 等多种子任务。

最终 caption 模型能输出长/中/短、正式/非正式/诗意等**多种风格**的 dense caption，覆盖用户的各种输入风格。

### 4.3 训练配置

- **架构**：1.3B 与 14B 两种 DiT；Wan-VAE 127M；umT5 5.3B 文本编码器；
- **扩散目标**：Rectified Flow + logit-normal 时间步采样；
- **优化器**：AdamW（论文未披露具体超参，与主流视频 DiT 训练一致）；
- **总训练数据**：数十亿图像 + 视频，约 O(1) 万亿 tokens；
- **硬件**：阿里云 PAI 集群，14B 模型训练依赖 128+ GPU；
- **精度**：混合精度训练，attention 用 8-bit FlashAttention（FP8 训练与推理）；
- **长视频支持**：Wan-VAE feature cache + 2D Context Parallelism 支持任意长视频训练与推理。

---

## 5. 实验结果

### 5.1 Wan-Bench 自建评测

#### 5.1.1 评测维度

Wan-Bench 包含 **14 个评测维度**（Figure 13），分为 3 大类：

- **动态质量（Dynamic Quality）**：Large Motion Generation、Physical Plausibility、Smoothness、Camera Control、Action Instruction Following；
- **图像质量（Image Quality）**：Comprehensive Image Quality、Scene Generation Quality、Pixel-level Stability、Stylization Ability、Human Artifacts；
- **指令跟随（Instruction Follow）**：Single Object、Multiple Objects、Spatial Positions、ID Consistency。

每个维度的 prompt 数量不同（Figure 14），从 30（相机控制）到 180（大运动生成）不等。

#### 5.1.2 Wan-Bench 加权总分（Table 2）

| 模型 | Large Motion | Physical Plausibility | ID Consistency | Camera Control | Stylization | Weighted Score |
|---|---|---|---|---|---|---|
| **Wan 14B** | 0.415 | **0.939** | **0.946** | **0.527** | 0.328 | **0.724** |
| Wan 1.3B | 0.468 | 0.912 | 0.938 | 0.483 | 0.430 | 0.689 |
| Sora | 0.482 | 0.933 | 0.925 | 0.380 | 0.606 | 0.700 |
| CN-TopA | 0.284 | 0.759 | 0.936 | 0.529 | 0.346 | 0.693 |
| CN-TopB | 0.413 | 0.898 | 0.935 | 0.406 | 0.386 | 0.690 |
| Hunyuan | 0.420 | 0.728 | 0.930 | 0.605 | 0.403 | 0.673 |
| Mochi | 0.405 | 0.836 | 0.940 | 0.465 | 0.623 | 0.639 |

**关键发现**：

- **Wan 14B 加权总分 0.724 全面领先**：超过 Sora（0.700）、Hunyuan（0.673）、Mochi（0.639）等闭源/开源竞品；
- **物理合理性与 ID 一致性最强**：0.939 / 0.946 高居榜首，说明视频 DiT + Flow Matching 配合大规模数据对运动物理与人/物身份保持的建模非常有效；
- **1.3B 也有 0.689**：与 CN-TopB 持平，超过 Mochi（0.639），证明 8GB 显存可跑的轻量模型依然竞争力强；
- **Sora 在大运动、风格化能力上仍占优**（0.482 / 0.606），但综合被 Wan 反超。

#### 5.1.3 人类偏好评估

在 700+ 任务、20+ 标注者、4 维评估（alignment / image / motion / overall）的 pairwise 测试中，Wan 14B 全面胜出：

| 对手 | Visual Quality 胜率 | Motion Quality 胜率 | Matching 胜率 | Overall 胜率 |
|---|---|---|---|---|
| CN-TopA | 30.6% | 16.1% | 46.0% | 44.0% |
| CN-TopB | 15.9% | 9.7% | 57.9% | 44.0% |
| CN-TopC | 27.8% | 14.9% | 56.7% | 48.9% |
| Runway Gen-3 | 48.1% | 40.3% | 69.1% | **67.6%** |

**Wan 14B vs Runway Gen-3 的 Overall 胜率 67.6%** 是最值得关注的数字——Runway Gen-3 是当时最被认可的闭源商业模型之一，Wan 在总投票中以 67.6% vs 32.4% 胜出。

![](assets/Wan2.1%20-%20开源大规模视频生成基础模型/figures/00-teaser-bench.png)

> **图 4：Wan 与 SOTA 模型对比（论文 Figure 1）。** **左**：Wan-Bench 加权 Score 条形图——Wan2.1-14B 0.72 第一，超过 Sora 0.70、CN-TopA 0.69、CN-TopB 0.69、Hunyuan 0.67、Mochi 0.64。**右**：人类偏好 Win Rate 堆叠条——对 Runway Gen-3 Wan 胜率 69%（Loss 15%, Draw 8%）；对 CN-TopC 胜率 82%。**解读**：Wan 的优势不在单一维度，而在加权总分与人类偏好的综合表现，尤其对最强闭源竞品 Runway 也有 30+ 个百分点的胜率优势，这是开源视频基础模型当时少见的成绩。
> 来源：论文 Figure 1，第 1 页，<https://arxiv.org/abs/2503.20314>

### 5.2 VBench 公开榜单

VBench（16 维、视频生成最权威的公开评测）：

| 模型 | Quality | Semantic | **Total** |
|---|---|---|---|
| **Wan 14B** | **86.67%** | **84.44%** | **86.22%** |
| Sora | 85.51% | 79.35% | 84.28% |
| Hailuo | 84.85% | 77.65% | 83.41% |
| HunyuanVideo | 85.09% | 75.82% | 83.24% |
| Runway Gen-3 | 84.11% | 75.17% | 82.32% |
| CogVideoX1.5-5B | 82.78% | 79.76% | 82.17% |
| Kling 1.0 | 83.39% | 75.68% | 81.85% |
| **Wan 1.3B** | 84.92% | 80.10% | **83.96%** |

**Wan 14B 在 Quality、Semantic、Total 三项全部第一**，86.22% 的总分超过 Sora 84.28%、Hailuo 83.41%。**Wan 1.3B 也有 83.96%**，超过 Kling 1.0、CogVideoX1.5-5B 与 HunyuanVideo，与 Sora 仅差 0.32%——**对一款 8GB 显存可跑的轻量模型，这是惊艳的成绩**。

### 5.3 定性结果

![](assets/Wan2.1%20-%20开源大规模视频生成基础模型/figures/01-samples.png)

> **图 5：Wan 采样示例（论文 Figure 2）。** 涵盖大运动场景（雪貂微距特写、电影级运镜泪滴女子、复古漫画风 "WAN" 爆炸字）、多人物（舞会、玩偶场景）、**中英文视觉文本**（婚礼 "囍" 汉字书法红纸、情侣海边剪影）。**关键看点**：Wan 是当时**首个能在视频中同时稳定生成中英文视觉文字**的开源模型，这对广告/字幕/短剧等下游场景有直接商业价值。
> 来源：论文 Figure 2，第 4 页，<https://arxiv.org/abs/2503.20314>

> **补充定性结果（论文 Figure 15）：Wan-T2V 多场景结果。** 8 个 prompt × 4 帧，涵盖 80 年代怪物屋喜剧、复古公路自行车赛、电影感水中出浴、专业摩托车越野、专业跳水、草地飘逸长发、父子骑行的"囍"字气球头套、电影级 "WAN" 爆炸火焰文字。**核心观察**：Wan 在长 prompt 复杂场景（多人物+多动作+相机运动+物理交互）下仍能保持时序一致与指令跟随；视觉文本生成稳定（"囍"气球、爆炸 "WAN"），这是其他开源视频模型当时的普遍弱点。
> 来源：论文 Figure 15，第 23 页，<https://arxiv.org/abs/2503.20314>

### 5.4 消融实验

#### 5.4.1 adaLN 共享 vs 模型深度

论文对比 4 种 adaLN 配置（1.3B = 30 层全共享为基线；其余配置调整共享策略 / 深度）：

- **Full-shared-adaLN-1.3B**（30 层全共享）：基线；
- **Half-shared-adaLN-1.5B**（前 15 层共享，后 15 层非共享）：1.5B；
- **Full-shared-adaLN-1.5B**（35 层全共享）：1.5B；
- **Non-shared-adaLN-1.7B**（30 层全非共享）：1.7B。

训练 200K 步测 L2 loss：

![](assets/Wan2.1%20-%20开源大规模视频生成基础模型/figures/04-ablation.png)

> **图 7：adaLN 配置训练 loss 曲线（论文 Figure 16）。** Full-shared-adaLN-1.3B（蓝）loss 略高；Full-shared-adaLN-1.5B（橙，35 层）稳定最低；Half-shared-adaLN-1.5B（绿）与 Non-shared-adaLN-1.7B（红）居中。**结论**：**把 adaLN 参数全共享、把节省的预算用于加深网络深度，比用预算堆 adaLN 参数量更有效**。这是 Wan 1.3B 在 30 层做到「参数更少但性能更好」的核心 trick，也为 14B 模型选择「40+ 层全共享 adaLN」提供了依据。
> 来源：论文 Figure 16，第 25 页，<https://arxiv.org/abs/2503.20314>

#### 5.4.2 文本编码器消融

对比 umT5、Qwen2.5-7B-Instruct、GLM-4-9B：

- 训练 loss 曲线（图 17）显示 **umT5 收敛更快、最终 loss 最低**；
- 即使按 HunyuanVideo 的「bidirectional token refiner」方案把 Qwen / GLM 加上 refiner adapter，**umT5 仍最优**；
- 与多模态 LLM Qwen-VL-7B-Instruct 对比（Table 6，15K 步 FID）：

| 文本编码器 | FID (↓) |
|---|---|
| umT5 | 43.01 |
| Qwen-VL-7B (last layer) | 43.72 |
| Qwen-VL-7B (second-to-last) | 42.91 |

**umT5 与 Qwen-VL-7B 第二层特征 FID 几乎持平，但 umT5 显存小得多**。

#### 5.4.3 VAE 重建损失函数消融

对比标准 VAE（重建损失）vs VAE-D（用 diffusion loss 代替重建损失），150K 步 T2I 训练测 FID：

| 训练步数 | VAE | VAE-D |
|---|---|---|
| 10K | 42.60 | 44.21 |
| 15K | **40.55** | 41.16 |

**VAE 在 10K / 15K 步 FID 均低于 VAE-D**，说明直接监督重建比 diffusion 目标监督更适合视频 VAE。

---

## 6. 扩展应用

### 6.1 图生视频（I2V）

Wan-I2V 在 T2V 基座上微调，引入**条件图 + 掩码**作为额外输入：

- 把条件图 $\mathbf{I} \in \mathbb{R}^{C \times 1 \times H \times W}$ 沿时间轴 zero-pad 成 $\mathbf{I}_c \in \mathbb{R}^{C \times T \times H \times W}$，经 Wan-VAE 压缩为条件潜变量 $\mathbf{z}_c$；
- 引入二值掩码 $\mathbf{M} \in \{0, 1\}^{1 \times T \times h \times w}$（1 = 保留帧，0 = 待生成），按 Wan-VAE 的时间 stride 重新排列；
- 噪声潜变量 $\mathbf{z}_t$、条件潜变量 $\mathbf{z}_c$、重排掩码 $\mathbf{m}$ 沿 channel 维拼接，输入 DiT 训练。

效果（Figure 19）显示 Wan-I2V 能稳定捕捉真实场景的动力学并扩展为动态视频。

### 6.2 VACE：统一可控生成与编辑框架

Wan2.1 还提出 **VACE（Video Condition Unit）**框架，把多种可控生成任务（reference-to-video、video editing、inpainting、outpainting、long video re-rendering）统一到一个输入范式：

$$\mathcal{V} = [\mathbf{T}; \mathbf{F}; \mathbf{M}]$$

- $\mathbf{T}$：文本提示；
- $\mathbf{F} = \{\mathbf{u}_1, \mathbf{u}_2, ..., \mathbf{u}_n\}$：context 视频帧序列；
- $\mathbf{M} = \{\mathbf{m}_1, \mathbf{m}_2, ..., \mathbf{m}_n\}$：对应掩码（1 = 编辑/参考，0 = 保留/生成）。

支持两种训练模式：
- **Full Fine-tuning**：把 VCU 作为输入，全量微调 Wan；
- **Context Adapter Tuning**：以 Res-Tuning 方式加 Context Adapter，不改 base model 权重，扩展 base 模型功能。

这一框架在长视频 re-rendering、compositional editing 等场景下显著降低了迁移成本。

### 6.3 其他应用

论文提到 Wan 套件还支持：
- **个性化生成（zero-shot personal video）**：参考人/物图像 → 保留身份的动态视频；
- **音频生成**：与视频 DiT 联合训练音频解码器；
- **实时生成**：通过 streaming 输出支持实时视频生成。

这些应用的具体架构与训练细节在 Wan2.1 论文中相对简略；后续 Wan2.2 已沿去噪阶段 MoE、高压缩统一 TI2V、S2V 与 Animate 等方向补齐部分能力，详见 §7.1。

---

## 7. 与相关工作的关系

- **vs Sora（OpenAI, 2024）**：Sora 是闭源 DiT 路线视频模型的代表作，Wan 是首个以开源姿态在 Wan-Bench / VBench / 人类偏好上同时反超 Sora 的工作，且工程细节完全公开；
- **vs HunyuanVideo（Kong et al., 2024）**：Hunyuan 是同期腾讯开源的全注意力视频 DiT，同样用 umT5 文本编码。Wan 的差异化在 Wan-VAE（127M vs 246M）、8-bit FlashAttention、Wan-Bench 自建评测、视觉文本生成能力；
- **vs Mochi（CGenmoTeam, 2024）**：Mochi 用非对称 3D attention 节省显存，Wan 用 2D CP 走 full attention 路线；Mochi 1.5B 参数量偏小，质量落后 Wan 较多；
- **vs CogVideoX（Yang et al., 2025b）**：CogVideoX 用 3D 变分卷积 + Expert Transformer 路线，Wan 用主流 DiT + Flow Matching 路线；Wan 14B 在 VBench 全面超过 CogVideoX1.5-5B；
- **vs SVD（Blattmann et al., 2023）**：SVD 是 UNet + EDM 路线的代表作，Wan 是 DiT + Flow Matching 路线的代表作。SVD 早一年开源，是「视频基础模型」概念的奠基者；Wan 是 2025 年「可商用级视频基础模型」的天花板；
- **vs 闭源商业模型（Kling、Hailuo、Runway Gen-3、Vidu）**：Wan 14B 在 Wan-Bench 加权总分与人类偏好的综合表现上**全面领先**这些闭源竞品，且 Wan 1.3B 以 8GB 显存可跑的特性独占消费级赛道。

### 7.1 Wan2.1 → Wan2.2：从单一 DiT 到分段专家与统一高压缩模型

#### 7.1.1 发布与论文归属核实

Wan2.2 没有另起一篇新的 arXiv 论文：官方 [Wan2.2 仓库](https://github.com/Wan-Video/Wan2.2)与官方发布页仍要求引用 **《Wan: Open and Advanced Large-Scale Video Generative Models》、arXiv:2503.20314**。但需要精确区分「官方引用沿用」与「arXiv 正文已更新」：截至 **2026-10-15**，arXiv 条目本身仍只有 2025-03-26 的 v1 与 2025-04-19 的 v2，摘要和代码链接仍指向 Wan2.1；Wan2.2 的技术增量主要由官方仓库与发布页披露，而不是一版新的 arXiv 正文。

#### 7.1.2 架构与产品形态差异

| 维度 | Wan2.1 | Wan2.2 |
|---|---|---|
| 主力去噪器 | 1.3B / 14B 单一 dense DiT，整条去噪轨迹共享同一组参数 | **T2V-A14B / I2V-A14B 双专家 MoE**：高噪专家负责早期整体布局与运动结构，低噪专家负责后期纹理与细节；每个专家约 14B、总计约 27B，但每一步只激活约 14B |
| 专家切换 | 无 | 按去噪时间步对应的 **SNR 阈值**切换，高噪阶段结束后转入低噪专家；官方消融显示完整双专家版本优于只替换其中一个专家的混合版本 |
| 轻量部署 | 1.3B，Wan-VAE 为 4×8×8 压缩 | **TI2V-5B dense 模型**，使用 Wan2.2-VAE 的 **T×H×W = 4×16×16** 高压缩；叠加 patchify 后 token 级总压缩达到 4×32×32 |
| 任务组织 | T2V、I2V 主要以不同模型/适配方式交付 | TI2V-5B 在**同一模型内原生统一 T2V 与 I2V**，支持 720p@24fps，面向消费级 GPU 部署 |
| 数据与美学 | 大规模通用图像/视频治理 | 在 Wan2.1 基础上增加 65.6% 图像与 83.2% 视频，并加入光照、构图、对比度、色调等细粒度美学标注 |

这两个分支解决的是不同问题：**A14B 用噪声区间专门化扩大有效容量，TI2V-5B 用更强潜空间压缩与任务统一降低部署成本**。因此 Wan2.2 不是简单把 Wan2.1-14B 加宽，而是同时沿「质量上限」和「统一高效推理」两条线升级。

### 7.2 Wan 作为后续系统骨干时，究竟被改造了什么

#### 7.2.1 Lyra 2.0：相机条件 + 空间记忆 + 防漂移的生成式重建

Lyra 2.0 以 **Wan2.1-14B DiT** 为视频骨干，但把短视频生成器改造成可长期探索的 3D 世界生成器：

- **相机控制**：把最近帧按目标相机位姿和估计深度 forward warp，编码后与去噪 latent 沿通道拼接；同时把 6D Plücker ray embedding 加到每层 self-attention 的 Q/K（不改 V），补偿大视角变化时 warp 无覆盖的问题；
- **防空间遗忘**：不融合一个会持续积累误差的全局点云，而为每帧维护独立 3D cache；按目标视角的可见覆盖检索 5 个历史帧作为 spatial slots，并将 warp 后的 canonical coordinate + depth 注入 Q/K，让几何只负责「找回哪一帧、对应到哪里」，外观仍由 Wan 先验合成；
- **防时间漂移与提速**：用 FramePack 按远近压缩时间历史，以 self-augmented history 让模型学习纠正自身退化输出，再用 DMD 将 35 步采样蒸馏到 4 步；生成的长视频最终经微调后的 Depth Anything V3 前馈重建为 3DGS，并可提取 mesh。

也就是说，Lyra 2.0 保留 Wan 的生成先验，新增的是**显式相机接口、可检索的 3D 空间记忆、自回归防漂移训练和视频到 3D 的重建后端**。

#### 7.2.2 Self-Forcing：把全局双向 Wan 改造成实时因果流式生成器

Self-Forcing 使用 **Wan2.1-T2V-1.3B** 初始化，不重做 VAE 或文本编码器，而是改写 DiT 的时间组织与训练分布：

- 将全时空双向注意力改为**帧级/块级因果注意力**，块内仍可双向、块间只能看历史；主配置每次自回归生成 3 个 latent frames；
- 把多步全视频采样蒸馏成**每个块 4 步去噪**，并为每层、每个去噪步分别维护 KV cache，历史块不再重复计算；rolling KV cache 只保留首块锚点与最近窗口，使长视频的显存和单块计算量有界；
- 真正的创新在训练侧：不是用 ground-truth 历史做 teacher forcing，而是让模型执行与测试一致的自回归 rollout、读取自己产生的历史与 KV cache，再以 DMD、SiD 或 GAN 这类分布目标训练；随机梯度截断只对抽中的块/去噪后缀反传，避免保存整条 rollout。

因此它对 Wan 的改造可概括为：**双向整段去噪 → 因果分块去噪，真实历史条件 → 自生成历史条件，多步离线生成 → 4 步 KV-cache 流式生成**。

#### 7.2.3 Yume-1.5：在 Wan2.2-TI2V-5B 上加入无限历史压缩与交互控制

Yume-1.5 实际采用的是 **Wan2.2-TI2V-5B**（而非 Wan2.1-14B）作为联合 T2V/I2V 基础模型，保留 Wan 的 mask 融合条件与 Rectified Flow，再增加三组面向交互世界的接口：

- **文本解耦**：把 caption 拆成 Event Description 与 Action Description 分别过 T5 后拼接；有限的键鼠动作词表可预编码缓存，事件文本只在初始/事件触发时计算；
- **TSCM 长历史**：时空分支对越久远的历史用越激进的可变 Patchify，送入标准注意力；通道分支把历史压到 96 通道后用 linear attention 融合。它不用 Self-Forcing 原版的 KV cache，而以双路压缩保留更长历史；
- **自回归与蒸馏**：chunk-wise 生成并把自身输出回灌为历史，以 Self-Forcing 式 distribution matching 把模型压到 4 步；训练时逐 step 交替 T2V / I2V，使同一模型同时支持 text-to-world、image-to-world、世界编辑和键鼠探索。

这里 Wan2.2 提供的是统一高压缩视频先验，Yume-1.5 新增的是**动作/事件语义接口、TSCM 无限上下文和实时自回归训练**。

#### 7.2.4 FantasyWorld：冻结 Wan2.1，在后半段并联显式几何分支

FantasyWorld 以 **Wan2.1** 为冻结的 imagination prior，不把视频模型整体微调成 3D 网络，而是做轻量双分支扩展：

- 复用前 **16 个冻结 Wan blocks** 作为 Preconditioning Blocks，先把高噪 latent 部分去噪到含几何线索的特征，再送入几何分支；
- 从 block 16 之后，在剩余 **24 个 blocks** 旁并联 Geometry-Consistent Branch，并在每层后插入门控双向 cross-attention：Geometry→Video 约束多视角一致性，Video→Geometry 用生成先验补全遮挡区域；
- 定制 **3D DPT heads**，从几何分支较深层特征解码 depth、point map 与 camera pose，并做与 WanVAE 对齐的 4× 时间上采样；
- 两阶段训练先做 latent bridging（冻结 Wan、只训练几何分支），再只微调双向 cross-attention 与 camera adapter，以 diffusion loss + geometry loss 协同优化，核心 Wan backbone 始终冻结。

它代表了另一种复用方式：不是把 Wan 因果化或压缩化，而是**把冻结的视频想象先验与可训练的显式 3D 感知并联，并让两者逐层双向校正**。

---

## 8. 局限与批判性评价

1. **训练成本对学术界不友好**：14B 模型 + 数十亿视频 + 万亿 tokens 的预训练，即使有完整开源代码与策略，**任何实验室都难以从零复现**。论文的核心贡献在「完整公开工程配方」而非「可复现的训练预算」；
2. **数据管线不公开细节**：论文描述了四步清洗与 dense caption 流程，但**没有公开具体阈值、打分模型权重、过滤比例**，学术界无法严格复现「数据治理具体贡献多少」；
3. **Wan-Bench 评测的独立性**：Wan-Bench 是阿里自建评测，prompt 设计与维度权重可能存在「为自家模型定制」的风险；论文虽然在 VBench 上也拿到第一，但 Wan-Bench 的加权总分计算细节、是否进行多次随机化评测，需要在公开 leaderboard 上进一步验证；
4. **视频长度仍受限于算力**：尽管 Wan-VAE 支持任意长视频 feature cache，**但 DiT 序列长度的平方级 attention 复杂度仍是硬约束**。Wan2.1 演示的样例大多是 5 秒短片，与 Sora 的 60 秒、Kling 1.6 的 10 秒相比不算长；
5. **物理一致性与因果推理仍不够强**：Wan 在 Wan-Bench Physical Plausibility 拿到 0.939 已经是第一，但**复杂物理交互（液体飞溅、布料、刚体碰撞）仍会出现明显错误**，距离「物理准确的视频生成」还有差距；
6. **视觉文本生成虽领先但仍非完美**：图 6 的 "囍" 气球与 "WAN" 爆炸字都稳定，但更复杂的中文段落、英文长句仍有字符级错误；
7. **8-bit FlashAttention 数值稳定性挑战**：FP8/Int8 混合量化 + FP32 跨 block 累积虽解决了 H20 上的稳定性问题，**但迁移到其他 GPU（如 A100、H100）需要重新调优**，论文未给出跨硬件的可移植性证据；
8. **2D Context Parallelism 的硬件耦合**：Ulysses+Ring 的 2D CP 在 NVLink + InfiniBand 的阿里云集群上表现优异，**但对一般研究组的单台 8 卡机器甚至单机多卡不友好**。

---

## 9. 复现与实践建议

- **官方资源**：<https://github.com/Wan-Video/Wan2.1> 提供 1.3B / 14B 的 T2V 与 I2V 完整权重与推理代码，支持消费级 GPU（1.3B 8.19GB VRAM）和高端 GPU（14B 80GB+）；
- **不要从零训练基座**：1.3B / 14B 模型直接下载权重 inference 即可，**不要尝试复现预训练**；
- **微调成本**：Wan-I2V / VACE / 个性化等下游任务都是少量 GPU + 短时间即可适配（论文未明确给出每任务的微调 step，但参考同类工作的 1~3 天 8×A100 量级）；
- **关键超参与技巧**：
  - 推理时用 **Diffusion Cache**（attention cache + CFG cache）几乎无损失提速 1.3~1.5×；
  - 8-bit FlashAttention 在 H20 上拿 95% MFU，A100/H100 上需要降回 BF16；
  - Flow Matching 推理步数可调到 30~50 步（论文默认 50）；
  - Prompt Rewrite：长 prompt 改写后效果显著优于原 prompt；
- **VAE 使用**：Wan-VAE 单独开源，可作为通用视频压缩 VAE 用于其他 DiT 训练；
- **数据治理（如自建视频集）**：四步清洗（基础维度 / 视觉质量 / 运动质量 / 视觉文本）+ dense caption（内部 caption 模型 + 开放源数据 + OCR 增强），按 cluster 采样打分；
- **常见踩坑**：
  - 不要把时间层与空间层分离 attention（除非有显存硬约束），full spatio-temporal attention 在 Wan 与 HunyuanVideo 上都被验证为最优；
  - 不要用 decoder-only LLM（Qwen、GLM）直接当文本编码器，**必须加 bidirectional refiner 或换 umT5**；
  - 不要在长序列训练时只依赖 gradient checkpointing，**activation offloading + 2D CP** 才是 Wan 的关键；
  - 高分辨率阶段务必做 **noise schedule 调整**（类似 SDXL 的 noise shift），Wan 论文虽未在主体强调，但视频 DiT 高清阶段训练不稳是普遍现象。

---

## 10. 在机器人世界模型中的应用

Wan 系列在本库多篇笔记中被引用：

- **Fast-WAM**：用 Wan2.2-5B 视频 DiT 做一次前向编码，得到隐式世界表征后直接生成动作，推理仅 190ms；
- **Dyna-2**：以 Wan 系视频模型为底座做百万小时缩放；
- 与 SVD、Cosmos 并列，成为「**视频基础模型 → 世界动作模型**」这条路线的主要视频骨干来源。

Wan 的开源 + 强运动建模 + 8GB 显存可跑特性，使其在 VPP（视频预测策略）、DreamZero、Ctrl-World 等机器人世界模型路线中成为 SVD 之后的事实标准之一。**理解 Wan2.1 是理解「视频基础模型如何被机器人世界模型复用」的必经入口**。

---

## 11. 个人启发与后续问题

1. **「Full Attention + 共享 adaLN + 因果 VAE + Flow Matching」是 2025 年视频 DiT 的事实标准范式**：Sora 之后的所有开源 DiT（HunyuanVideo、Wan、CogVideoX、Mochi）虽然在细节上千差万别，但都收敛到这套范式上。Wan2.1 把这条范式的**工程细节全部开源**，对学术界是一份珍贵的模板；
2. **「小 VAE + 强 DiT」比「大 VAE + 小 DiT」更划算**：Wan-VAE 127M 的极小体量 + 8×8×4 的高压缩比，让 DiT 的序列长度大幅下降，**这意味着 1.3B 模型的显存占用能压到 8GB**。这条经验对 3D DiT、音频 DiT 等其他模态也适用；
3. **「shared adaLN + 加深网络深度」是参数分配的最优解**：消融里 1.5B 共享 adaLN（35 层）的 loss 始终低于 1.7B 非共享（30 层），这说明对生成模型来说，**深度比宽度更重要**——这与 Transformer scaling law 的最新结论（DeepSeek-V3）一致；
4. **「Wan-Bench + VBench + 人类偏好」三层评测是工业化视频生成的新标准**：单一 benchmark 容易被「为评测训练」攻破，Wan 给出 1,035 条 prompt 自建评测 + 1,000+ prompt 人类偏好 + 1,500+ prompt VBench 的三层验证，**这是工业化视频生成最严肃的评测范式**；
5. **「中英文视觉文本」是 Wan 差异化的真正卖点**：在广告、字幕、短剧等下游场景，「能在视频里稳定生成准确的中英文文字」是其他开源视频模型当时的普遍弱点。Wan 通过 OCR 增强的 dense caption 训练 + 视觉文本专项数据收集，把这个能力直接产品化；
6. **未解问题**：
   - 长视频生成（>1 分钟）如何突破 DiT 序列长度平方级复杂度？Wan 的 feature cache 只解决 VAE 端，DiT 端仍受限于 256K token 上限；
   - 物理一致性、因果推理、3D 理解能否从「视频先验」直接涌现而不靠后训练？Wan 在 ID Consistency 0.946 / Physical Plausibility 0.939 已经很高，但与「完美物理引擎」仍有差距；
   - 8-bit FlashAttention 的跨硬件可移植性如何？DeepSeek-V3 的 FP8 GEMM 经验能否直接复用？
   - Wan2.2 已给出一种折中：专家内部保留 Full Attention，但按高/低噪声阶段切换 14B 专家；后续还会继续沿这种时间步 MoE 扩展，还是转向 token 级稀疏 attention？
   - 视频 DiT 作为世界模型时，「动作生成」与「视频生成」能否统一在一个 DiT 里？Fast-WAM 给出了一种 MoT 混合方案，但训练成本仍是问题。

---

## 12. 材料来源

### 12.1 Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/Wan2.1.pdf` | 论文 PDF（60 页） | <https://arxiv.org/pdf/2503.20314> | 全文 | 2026-08-29 | 精读主来源 |
| `figures/00-teaser-bench.png` | Figure 1：Wan-Bench Score + Human Preference Win Rate | arXiv:2503.20314 | 第 1 页 | 2026-08-29 | 整体对比与人类偏好 |
| `figures/01-samples.png` | Figure 2：Wan 采样示例（中英文视觉文本、大运动、多人物） | arXiv:2503.20314 | 第 4 页 | 2026-08-29 | 质量展示与视觉文本能力 |
| `figures/02-vae-recon.png` | Figure 8：Wan-VAE 重建质量可视化（纹理/面部/文本/高运动） | arXiv:2503.20314 | 第 12-13 页 | 2026-08-29 | VAE 方法关键图 |
| `figures/03-pipeline.png` | Figure 9：Wan 整体架构 pipeline | arXiv:2503.20314 | 第 13 页 | 2026-08-29 | 整体架构图 |
| `figures/04-ablation.png` | Figure 16：adaLN 共享 vs 深度 消融训练 loss 曲线 | arXiv:2503.20314 | 第 25 页 | 2026-08-29 | 消融实验 |
| `figures/05-dit-block.png` | Figure 10：DiT block 内部结构（共享 adaLN + Cross-Attn + FFN） | arXiv:2503.20314 | 第 14 页 | 2026-08-29 | DiT 块详细结构 |
| <https://github.com/Wan-Video/Wan2.1> | 官方代码与权重 | — | — | 2026-08-29 | 复现参考 |
| <https://github.com/Wan-Video/Wan2.2> | Wan2.2 官方代码、权重与技术说明 | arXiv:2503.20314（官方沿用引用） | README | 2026-10-15 | Wan2.1→2.2 差异核实 |
| <https://arxiv.org/abs/2604.13036> | Lyra 2.0 论文 | arXiv:2604.13036 | 方法与附录 A.1 | 2026-10-15 | Wan2.1-14B 的空间记忆/防漂移改造 |
| <https://arxiv.org/abs/2506.08009> | Self-Forcing 论文 | arXiv:2506.08009 | 方法与实验设置 | 2026-10-15 | Wan2.1-1.3B 的因果流式改造 |
| <https://arxiv.org/abs/2512.22096> | Yume-1.5 论文 | arXiv:2512.22096 | §4 方法 | 2026-10-15 | Wan2.2-TI2V-5B 的 TSCM/交互改造 |
| <https://arxiv.org/abs/2509.21657> | FantasyWorld 论文 | arXiv:2509.21657 | §3 方法 | 2026-10-15 | Wan2.1 的视频-几何双分支改造 |

### 12.2 引用本笔记的相关笔记

- [[Fast-WAM - 无测试时未来想象的世界动作模型]] — 用 Wan2.2-5B 做一次前向世界表征
- [[Spann3R - 空间记忆增量式重建]] — 视频基础模型用于世界模型的同期工作
- [[Stable Video Diffusion - 大规模潜视频扩散模型]] — 视频基础模型路线的上一代代表作
