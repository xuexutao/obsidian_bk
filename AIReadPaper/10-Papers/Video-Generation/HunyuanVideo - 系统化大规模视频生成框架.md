---
type: paper-note
status: complete
domain: Video-Generation
paper: "HunyuanVideo: A Systematic Framework For Large Video Generative Models"
year: 2024
arxiv: 2412.03603
doi: null
source: https://arxiv.org/abs/2412.03603
project: https://hunyuan.tencent.com/video/en
code: https://github.com/Tencent-Hunyuan/HunyuanVideo
code_commit: 4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e
tags:
  - Video-Generation
  - Diffusion-Transformer
  - Flow-Matching
  - MMDiT
  - Causal-3D-VAE
  - Scaling-Law
  - Open-Source
  - HunyuanVideo
created: 2026-11-05
updated: 2026-11-05
---

# HunyuanVideo：系统化大规模视频生成框架

## 基本信息

- **论文**：*HunyuanVideo: A Systematic Framework For Large Video Generative Models*
- **作者/机构**：Tencent Hunyuan Foundation Model Team
- **版本边界**：本文以官方 **arXiv:2412.03603 v6** 为主论文。v6 已纳入早期 I2V、Avatar 等应用章节，但核心基础模型仍是 2024 年发布的 13B HunyuanVideo；后来的 HunyuanVideo 1.5 是独立技术报告 arXiv:2511.18870，不能把其 SSTA、8.3B、16×16 空间 VAE 或实验结果倒推给初代。
- **官方代码**：<https://github.com/Tencent-Hunyuan/HunyuanVideo>，本文固定到 commit [`4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e`](https://github.com/Tencent-Hunyuan/HunyuanVideo/tree/4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e)。论文中的旧链接 <https://github.com/Tencent/HunyuanVideo> 指向同一项目。
- **官方模型卡**：<https://huggingface.co/tencent/HunyuanVideo>，核对 revision `6204ad6aea1a77ff5aba337c88278bb9500eb37d`。
- **开源范围**：公开模型定义、T2V 权重、VAE、推理/采样、FP8 与 xDiT 并行推理；**没有公开**互联网规模基座预训练的 dataloader、过滤器权重、AngelPTM 训练循环与完整优化配置。
- **核心配置**：13B 级 HunyuanVideo Diffusion Transformer，20 个 dual-stream blocks + 40 个 single-stream blocks，hidden size 3072，24 heads，head dim 128，FFN 12288；Causal 3D VAE 的时间/空间压缩为 4×8×8、latent channel 为 16。
- **官方默认输出档位**：推荐 720×1280、129 帧；另支持 540p 档。官方 README 报告 batch=1 时峰值显存约 60GB（720×1280×129）或 45GB（544×960×129），建议 80GB GPU。

## 一句话结论

HunyuanVideo 的真正价值不只是“开源一个 13B 视频模型”，而是把**数据分层过滤与结构化重标注、图像到视频的 scaling law、Causal 3D VAE、dual-to-single-stream 全注意力 MMDiT、MLLM+CLIP 文本条件、Flow Matching、渐进式图视频联合训练、CFG 蒸馏和 5D 并行基础设施**组织成一条可解释的大模型工程链；它证明了视频生成质量的提升来自系统协同，而非单独扩大 DiT，但公开代码只能严格复现推理，不能严格复现论文训练。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

2024 年的视频生成出现了明显的“能力闭源化”：Sora、Runway Gen-3、Luma 等商业系统在画面质量、运动与复杂提示词理解上领先，而公开模型往往规模更小、训练配方不完整。HunyuanVideo 要解决的不是一个局部模块问题，而是：**如何在公开权重和代码的前提下，把视频基础模型扩到工业级规模，并让数据、架构、训练课程与系统并行共同支撑这一规模。**

论文指出，朴素地同时增加参数、数据和计算并不高效。视频比图像多一个时间轴，token 数和 full attention 成本急剧上升；高质量视频又远少于图像。如果没有模型/数据配比、渐进课程和基础设施，13B 只是不可训练的数字。作者因此先在图像模型上拟合 compute-optimal scaling law，再把各尺寸图像 checkpoint 迁移到视频模型，报告这种策略可把达到目标性能所需的计算资源降低至随机扩展方案的约五分之一。

### 1.2 核心贡献

第一，论文把数据处理写成分层课程的一部分，而不是一次性清洗：从 256p 到 720p，每升一级就提高美学、清晰度、运动与去水印阈值，最后用约 1M 条人工复核视频做高性能微调。第二，模型采用 4×8×8 的 Causal 3D VAE 和 20+40 层 dual-stream→single-stream MMDiT，以统一 full attention 同时覆盖图像和视频。第三，文本端不只用传统 T5，而以 decoder-only MLLM 提取细粒度语义，再用双向 token refiner 修复因果注意力不适合扩散条件的问题，并以 CLIP pooled feature 提供全局语义。第四，论文用图像→视频两阶段 scaling law 选择 13B，而不是凭经验定规模。第五，Flow Matching、progressive image-video joint training、time-step shift、CFG distillation、5D parallelism 与自动容错形成完整训练/推理工程。

---

## 2. 任务定义、表示与目标函数

### 2.1 输入输出

初代基础模型的主任务是文本生成视频。给定提示词 $c$，目标是生成视频 $x\in\mathbb{R}^{B\times3\times F\times H\times W}$。公开 T2V 默认使用 $B=1,F=129,H=720,W=1280$。图像被视为单帧视频，因此同一 DiT 也能参与 T2I 预训练。

视频不是直接在像素空间去噪。Causal 3D VAE 把 $4n+1$ 帧压成 $n+1$ 个 latent frames：

$$F_z=1+\left\lfloor\frac{F-1}{4}\right\rfloor,\quad H_z=\frac{H}{8},\quad W_z=\frac{W}{8}$$

因此 129×720×1280 的视频对应 latent：

$$z\in\mathbb{R}^{1\times16\times33\times90\times160}$$

再经 kernel/stride 为 $(1,2,2)$ 的 3D patch embedding，视觉 token 数为：

$$L_v=33\times45\times80=118{,}800$$

### 2.2 Flow Matching

论文从干净 latent $x_1$ 与高斯噪声 $x_0\sim\mathcal{N}(0,I)$ 构造直线路径：

$$x_t=(1-t)x_0+tx_1,\quad t\sim p_{\mathrm{logit-normal}}$$

真实速度场为：

$$u_t=\frac{dx_t}{dt}=x_1-x_0$$

网络 $v_\theta(x_t,t,c)$ 用 MSE 拟合速度：

$$\mathcal{L}_{\mathrm{gen}}=\mathbb{E}_{t,x_0,x_1}\left[\left\|v_\theta(x_t,t,c)-u_t\right\|_2^2\right]$$

推理从 $x_0$ 出发，用一阶 Euler ODE solver 沿估计速度积分到 $x_1$。公开实现默认 50 步，并对时间变量做 shift。论文与代码都强调：Flow Matching 只是目标函数，最终质量还高度依赖时间采样、shift、guidance 和训练课程。

### 2.3 CFG 与 embedded guidance

经典 CFG 组合条件/无条件速度：

$$\hat v=v_{\varnothing}+s(v_c-v_{\varnothing})$$

它需要每一步计算两个 batch。论文将组合后的 CFG 输出蒸馏进一个结构相同的 student，并把 guidance scale 作为额外条件；训练时 scale 从 1 到 8 随机采样，论文报告约 1.9× 加速。公开默认 checkpoint `HYVideo-T/2-cfgdistill` 正是这种模型：`cfg_scale=1.0`，但 `embedded_cfg_scale=6.0`。因此“默认只跑一次 DiT”不等于没有 guidance，而是 guidance 已成为网络输入。代码仍保留经典 CFG；当 `cfg_scale>1` 时会把 latent 和正负文本 batch 拼成 2 倍再执行公式。

---

## 3. 核心方法

### 3.1 总体架构

![](assets/HunyuanVideo%20-%20系统化大规模视频生成框架/figures/01-overall-architecture.png)

> **图 1（论文 Figure 5）：HunyuanVideo 端到端架构。** 视频经 Causal 3D VAE 编码成时空 latent，加噪后作为 DiT 视觉流；文本经大语言模型形成条件流；Diffusion Backbone 预测速度/去噪 latent，再由同一 VAE 解码。图中最关键的不是串联顺序，而是“像素视频→高压缩 latent→统一图视频 Transformer”把 720p、129 帧问题转成可分布式训练的 118,800-token 序列。  
> 来源：论文 Figure 5，PDF 文件第 5 页，<https://arxiv.org/abs/2412.03603>；官方原图见 `sources.md`。

### 3.2 Causal 3D VAE

![](assets/HunyuanVideo%20-%20系统化大规模视频生成框架/figures/02-causal-3d-vae.png)

> **图 2（论文 Figure 6）：Causal 3D VAE。** 输入 $(T+1)\times H\times W$ 被压到 $(T/4+1)\times H/8\times W/8$，再对称解码。首帧单独保留使 $4n+1$ 帧与 $n+1$ latent frames 对齐；16 个 latent channels 在压缩率与重建能力之间折中。  
> 来源：论文 Figure 6，PDF 文件第 6 页，<https://arxiv.org/abs/2412.03603>；官方原图见 `sources.md`。

#### 3.2.1 输入输出与因果性

VAE 输入 `[B,3,F,H,W]`，输出 posterior 的均值/方差参数，再采样 16 通道 latent。`CausalConv3d` 只使用当前及历史帧，避免未来像素泄漏到较早 latent。空间做三次 2× 下采样，时间做两次 2× 下采样，因此比例为 4×8×8。对默认视频，像素 RGB 标量约为 3.54 亿，而 latent 标量为 760.32 万，标量数量约缩小 46.6×；随后 patchify 再把每个 1×2×2 latent patch 变成一个 token。

#### 3.2.2 为什么这样设计

4× 时间压缩保留了足够运动带宽，8× 空间压缩则延续成熟 latent diffusion 的质量/效率平衡。过强时间压缩容易把快速动作平均掉，过弱压缩又使 token 数难以承受。因果卷积还让任意长度与时序切块更自然。论文没有用预训练 2D VAE 初始化，而是从零联合训练图像/视频 VAE，使时空卷积从一开始就适配两种模态。

#### 3.2.3 损失与训练

图像/视频按 1:4 混合，完整损失为：

$$\mathcal{L}_{\mathrm{VAE}}=\mathcal{L}_1+0.1\mathcal{L}_{\mathrm{LPIPS}}+0.05\mathcal{L}_{\mathrm{adv}}+10^{-6}\mathcal{L}_{\mathrm{KL}}$$

训练从低分辨率短视频渐进到高分辨率长视频；采样间隔在 1～8 中随机选，以提高高速运动重建。推理以时空重叠 tile 分块，重叠区线性融合；为了避免“训练整图、推理切块”造成接缝，末期训练随机开启/关闭 tiling。公开 config 给出通道 `[128,256,512,512]`、每块 2 层、latent 16、`sample_size=256`、`sample_tsize=64`、`scaling_factor=0.476986`，默认推理开启 tiling。

#### 3.2.4 重建实验

论文 Table 1 在 ImageNet 256×256 上报告 PSNR 33.14，在 MCL-JCV 33×360×640 上报告 35.39，均高于表中 FLUX/OpenSora/CogVideoX/Cosmos-VAE 对照。Figure 7 的文字、小脸和复杂纹理也更完整。需要注意：PSNR 不能直接代表生成质量；VAE 优势主要说明给定真实视频时压缩损失较小，不能单独证明 DiT 的语义或运动建模更强。

### 3.3 Dual-stream→Single-stream MMDiT

![](assets/HunyuanVideo%20-%20系统化大规模视频生成框架/figures/03-diffusion-backbone.png)

> **图 3（论文 Figure 8）：HunyuanVideo Diffusion Backbone。** 前 20 个 dual-stream blocks 为视觉/文本保留独立 LayerNorm、调制与 MLP，但在一次 joint attention 中让 Q/K/V 跨模态交互；后 40 个 single-stream blocks 把 118,800 个视觉 token 与至多 256 个文本 token 直接拼接，统一更新。右侧结构也显示 3D RoPE、QK-Norm、门控残差和并行 Attention/MLP。  
> 来源：论文 Figure 8，PDF 文件第 7 页，<https://arxiv.org/abs/2412.03603>；官方原图见 `sources.md`。

#### 3.3.1 双流阶段

视觉流和文本流形状分别为 `[B,L_v,3072]`、`[B,L_t,3072]`。每个流有独立的 adaLN-style shift/scale/gate、QKV 与 MLP，所以视觉可以学习时空统计，文本可以保持语言结构；但 attention 前把两侧 Q/K/V 沿序列拼接，使视觉 query 能看文本 key/value，文本也能反向吸收视觉/噪声状态。attention 输出再拆回两个流，各自残差更新。

#### 3.3.2 单流阶段

20 层后把两流拼成 `[B,L_v+L_t,3072]`，进入 40 个 single-stream blocks。每块先用一个线性层同时产生 QKV 和 MLP 分支，attention 与激活后的 MLP 输出再拼接，经第二个线性层和 gate 回写。这种并行形式来自 SD3/FLUX 一类 MMDiT：前段保护模态特性，后段让语义和视觉彻底融合。论文表述为 dual-stream 占 1/3、single-stream 占 2/3，公开配置恰为 20/40。

#### 3.3.3 Full attention 的收益与代价

作者选择统一时空 full attention，而不是空间 attention+时间 attention 分解，理由有三点：经验质量更好；图像是单帧视频，架构无需分叉；可以直接复用 LLM 的 FlashAttention、sequence/context parallel 等系统优化。代价是二次复杂度。默认 720p 的视觉序列 $L_v=118{,}800$，单个 head 的 dense score 数约 141 亿，24 heads 若显式保存 BF16 score 将超过 670GB/层。实际 Flash/Fused Attention 不物化完整矩阵，并用 context parallel 切序列，但算术量仍存在，这解释了为什么初代推理仍需高端 GPU。

### 3.4 3D RoPE

每个 head 为 128 维，论文与代码把它分成 `(d_t,d_h,d_w)=(16,56,56)`。对视觉 token 坐标 $(t,h,w)$，分别生成三个旋转频率并作用于 Q/K 对应通道，最后拼回：

$$\operatorname{RoPE}_{3D}(q)=\left[R_t(q_{1:16});R_h(q_{17:72});R_w(q_{73:128})\right]$$

时间维只分 16 维，空间各 56 维，反映默认视频 token 在空间上的变化量更大。RoPE 不加到文本 token；single-stream 中代码先分离视觉与文本 Q/K，只对视觉片段旋转，再拼回。公开默认 `rope_theta=256`。

### 3.5 MLLM、CLIP 与双向 Token Refiner

传统 T5 是双向 encoder，适合作为扩散条件，但多模态指令理解和细节推理有限。HunyuanVideo 选择视觉指令微调过的 decoder-only MLLM，认为它更擅长对象细节、关系和系统指令；问题是 causal attention 使前部 token 看不到后文。两层 `SingleTokenRefiner` 因此做双向 self-attention，把 MLLM `[B,256,4096]` 投影/修正成 `[B,256,3072]`。

Refiner 不只是静态 adapter。它把 timestep embedding 与 masked mean text context 相加作为条件，通过零初始化 gate 调制两层 self-attention 与 MLP，因此文本表示会随去噪阶段变化。与此同时，CLIP-L/14 的 pooled feature `[B,768]` 经 MLP 投到 `[B,3072]`，与 timestep embedding、可选 embedded guidance embedding 相加，作为所有 DiT block 的全局调制向量。

**论文与开源的关键边界**：论文称可配置多种 MLLM，并使用内部 Hunyuan MLLM；官方 checkpoint 下载说明明确表示当时未发布 HunyuanMLLM，社区版本推荐拆出 `llava-llama-3-8b-v1_1-transformers` 的语言模型作为 `text_encoder`。所以“论文训练所用文本编码器”与“公开可运行默认文本编码器”不应视为完全同一权重。

### 3.6 时间步 shift 与采样

公开 scheduler 默认 flow shift 为 7。论文将原始 $t$ 映射为：

$$t'=\frac{st}{1+(s-1)t}$$

$s>1$ 会把采样预算向高噪声早期倾斜。50 步取 $s=7$；低于 20 步时论文建议增大到约 17。Figure 11 显示在 10 步时，该策略比 linear-quadratic scheduler 更能维持视觉质量。但这部分主要是定性比较，论文没有给出完整的多随机种子统计或感知指标。

---

## 4. Scaling 策略与渐进训练

### 4.1 图像 scaling law

![](assets/HunyuanVideo%20-%20系统化大规模视频生成框架/figures/04-scaling-laws.png)

> **图 4（论文 Figure 10）：DiT-T2X scaling laws。** 上排在 92M～6.6B 图像模型上拟合 loss envelope、compute→参数和 compute→token；下排从各尺寸 compute-optimal 图像 checkpoint 初始化视频模型，再拟合视频规律。13B 与 740B/928B token 标记是模型选型的外推点，不等于论文披露了实际唯一数据集条目数或最终真实训练 FLOPs。  
> 来源：论文 Figure 10，PDF 文件第 9 页，<https://arxiv.org/abs/2412.03603>；本图由官方 PDF 300 DPI 整页渲染后完整裁剪。

作者先构造 92M～6.6B 的 DiT-T2X(I) 家族，在同一 256p 数据和超参下用 DDPM/v-prediction训练，拟合：

$$N_{\mathrm{opt}}=a_1C^{b_1},\quad D_{\mathrm{opt}}=a_2C^{b_2}$$

图像模型得到 $a_1=5.48\times10^{-4},b_1=0.5634,a_2=0.324,b_2=0.4325$；其中 $N,D$ 以 billion 计，$C$ 以 Peta FLOPs 计。方法的核心是对不同模型 loss 曲线取下包络：给定计算量，不一定模型越大越好，过大模型会因为 token 不足而训练不充分。

### 4.2 视频 scaling law

每个尺寸选择其图像 envelope 上的最优 checkpoint，再启动 T2X(V) 实验。视频拟合得到 $a_1=0.0189,b_1=0.3618,a_2=0.0108,b_2=0.6289$。综合训练消耗与推理成本，作者选 13B。论文也明确说，这些 token 估计只对应图像和视频各自的第一阶段；从低分辨率到高分辨率的 progressive scaling 尚未建立规律。

因此这组实验应读成“13B 的选型依据”，而不是“13B 已被严格证明全局最优”。loss 是训练目标代理，不等于人类偏好；拟合范围最大只有 6.6B，13B 是外推；高分辨率阶段、数据质量和后训练均未进入公式。

### 4.3 Progressive image-video joint training

训练先做两阶段图像预训练。Stage 1 用 256p 多宽高比图像学习低频概念和文本对齐；Stage 2 混合 256p/512p buckets，避免只在 512p 微调导致 256p 能力灾难性遗忘，并为不同 scale 使用动态 micro-batch。

视频阶段继续采用多宽高比、多时长 bucket，每个 bucket 按 token 数设最大 batch。课程依次为：

1. **低分辨率短视频**：先学文本→内容和短动作一致性；
2. **低分辨率长视频**：再学长时动态、场景变化与时空一致性；
3. **高分辨率长视频**：最后补细节，同时维持复杂运动。

每一阶段都以不同比例混入图像，既利用更丰富的世界知识，也防止视频分布使图像语义退化。论文概括为从 256p/65 帧渐进到最高 960p，并在主要高质量设置展示 720p/129 帧；各阶段的确切 step、全局 batch、学习率、图像/视频比例没有完整披露。

---

## 5. 数据过滤、重标注与训练基础设施

### 5.1 分层数据过滤

原始视频先用 PySceneDetect 切单镜头；以清晰帧作为 clip 起点；内部 VideoCLIP embedding 同时用于余弦去重和约 10K concept centroids 的重采样/均衡。之后按维度过滤：DOVER 评估审美与技术质量，内部模型判清晰度，光流估计运动速度，PySceneDetect+TransNet V2 检查镜头边界，OCR 删除文字过多样本并定位裁字幕，YOLOX-like 检测器处理水印、边框、logo 与遮挡/敏感区域。

真正关键是**阈值随训练阶段升级**。256p、360p、540p、720p 数据集逐级收紧，每一级会去掉上一阶段约 1/2～1/5。最终 SFT 约 1M clips，由人工从颜色、光照、主体突出、构图，以及速度、动作完整度、运动模糊等维度复核。论文没有披露原始视频总小时数、每一级绝对条目数、具体阈值或过滤器权重，不能从 1M SFT 反推预训练规模。

图像也复用除运动外的大部分过滤器：第一阶段是 billions 级图文对，第二阶段是 hundreds-of-millions 级更高质量图像。Figure 10 中的 740B/928B 是 scaling law 的 token 外推，不是“训练集有 740B 张图/928B 个视频”。

### 5.2 Structured recaption

内部 VLM 为全部图像/视频生成 JSON caption，包含短描述、dense description、背景、风格、shot type、光照、氛围，以及来源/质量等 metadata。dense description 特别把镜头运动、转场和主体动作合并到叙事。训练时随机 dropout、排列组合这些字段，构造不同长度和格式，避免模型只适应一种冗长模板。

另一个相机分类器识别 14 类运镜：zoom in/out、pan 四向、tilt 四向、around left/right、static、handheld。高置信标签被写入 caption，使相机控制从偶然相关变成显式文本条件。推理前再用 Hunyuan-Large 做 prompt rewrite，把用户短句改成训练分布；self-revision 比较原始与改写提示，LoRA 版本则用高质量 rewrite pairs 蒸馏这一流程。

### 5.3 分布式训练

论文在 Tencent AngelPTM 上采用“5D”组合：TP 切矩阵参数；SP 切序列并减少 LayerNorm/Dropout 重复；CP 用 Ring Attention 支撑长上下文；DP 横向扩数据；DP 内的 ZeroCache 分片参数、梯度与优化器状态。长序列 attention 使用 FusedAttention；activation 通过 recomputation 换显存，并以 layer-based offload 移到 host memory。

跨机通信使用 Tencent XingMai 网络，任务由 Angel 平台调度。自动容错检测硬件错误、替换健康节点并拉起任务，论文报告训练稳定性 99.5%。但 GPU 型号/总卡数、训练墙钟时间、电力、云成本与碳排放均未披露，所以无法给出可信的“从零训练预算”。

---

## 6. 实验与结果解读

### 6.1 专业人评

主对比使用 1,533 条代表性 prompts，每个模型每条只推理一次，避免 cherry-picking；60 名专业评估者从 text alignment、motion quality、visual quality 评价。Table 3 报告：HunyuanVideo 为 61.8% / 66.5% / 95.7%，overall 41.3%，排名第一；CNTopA 的 alignment 62.6% 略高，CNTopB 与 Gen-3 的 visual quality 97.7%/97.5% 也高于 HunyuanVideo，但 HunyuanVideo 的 motion quality 66.5% 最强，使综合排名领先。

这一结果支持的严格结论是：**在作者的人评协议和当时五个闭源基线下，HunyuanVideo 综合偏好第一，主要优势是运动。**它不能直接证明在公开 benchmark 上全面领先，因为三款中国模型被匿名，API/Web 版本不可长期复核，评分定义与置信区间未完整公开，也没有对随机种子方差进行报告。

### 6.2 VAE、Scaling 与采样消融

论文最扎实的模块实验有三类：VAE PSNR 与可视化证明压缩器不成为明显瓶颈；Figure 10 的多尺寸 loss envelope 支撑 13B 选型；Figure 11 比较低步数 scheduler，支撑 time-step shift。相反，主论文没有逐项关闭 MLLM、token refiner、3D RoPE、dual/single 比例或数据过滤器的完整量化消融。这使系统整体有效，但每个模块的边际贡献难以精确归因。

### 6.3 定性能力

论文展示多主体关系、快速车辆/体育运动、组合概念、顺序动作规划、场景文字与自动 scene cut。最有说服力的是高运动样例与复杂提示词：它们与人评中 motion 优势一致。文字生成只展示短英文词组，不能据此推断长文本或复杂中文排版已经可靠。

### 6.4 失败案例与未报告项

论文没有独立 failure-case figure，也未给失败率。根据架构和公开要求，可以把复现风险分为四类，但以下是**工程判断而非论文量化结论**：长序列 full attention 导致高显存/高时延；复杂物理与多主体交互仍可能身份漂移或拓扑错误；VAE 时空压缩可能损失细小高速目标；prompt rewrite 可能提高美学却改写用户语义。严格的失败分布、内容安全误拒/漏检、训练数据版权构成和不同语言公平性均未披露。

---

## 7. 代码级实现

### 7.1 固定版本与论文模块→代码入口

本文只读核对官方仓库 commit `4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e`。主要映射如下：

| 论文模块 | 公开代码入口 | 可验证内容 |
|---|---|---|
| CLI/默认参数 | `sample_video.py`, `hyvideo/config.py` | 720p/129f、50 steps、flow shift、embedded CFG、精度/offload |
| 推理编排 | `hyvideo/inference.py` | 加载 DiT/VAE/双文本编码器、3D RoPE、xDiT 序列并行 |
| 去噪循环 | `hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py` | prompt encode、噪声 shape、CFG、embedded guidance、scheduler、VAE decode |
| Flow scheduler | `hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py` | shift、reverse、Euler solver |
| MMDiT | `hyvideo/modules/models.py` | 20 dual + 40 single、3072、24 heads、QK RMSNorm、patch/unpatch |
| Token Refiner | `hyvideo/modules/token_refiner.py` | 2 层双向 self-attention、timestep/context-aware gates |
| 3D RoPE | `hyvideo/modules/posemb_layers.py` | `(16,56,56)` 时/高/宽通道分配 |
| Causal 3D VAE | `hyvideo/vae/autoencoder_kl_causal_3d.py`, `hyvideo/vae/vae.py` | 4×8×8、16 channels、tiling、posterior 与 decode |
| MLLM/CLIP | `hyvideo/text_encoder/__init__.py`, `ckpts/README.md` | LLaVA-Llama3 8B 社区替代、CLIP-L/14、hidden-state 选层 |

### 7.2 默认 HunyuanVideo-T2V 张量总表

以下采用官方推荐的 720×1280、129 帧、batch=1、默认 guidance-distilled checkpoint。`L_t=256` 是裁掉 95 个 system instruction tokens 后的上限；实际短提示的有效 token 由 mask 决定。

| 阶段 | 具体张量形状 | 说明 |
|---|---:|---|
| 像素输出目标 | `[1,3,129,720,1280]` | 129 必须为 $4n+1$ |
| 初始高斯 latent | `[1,16,33,90,160]` | `33=(129-1)/4+1`，空间各除 8 |
| PatchEmbed 后网格 | `[1,3072,33,45,80]` | Conv3d kernel=stride=`(1,2,2)` |
| 视觉 token | `[1,118800,3072]` | $33\times45\times80$ |
| MLLM 输入 hidden | `[1,256,4096]` | 默认取倒数第 3 组 hidden state（`skip_layer=2`） |
| Token Refiner 输出 | `[1,256,3072]` | 两层双向 refiner |
| CLIP pooled | `[1,768]` | CLIP-Large 全局文本向量 |
| timestep / CLIP / guidance vec | 各 `[1,3072]` | 三者相加作为 block 调制；guidance 输入代码乘 1000 |
| dual-stream block ×20 | img `[1,118800,3072]`; txt `[1,256,3072]` | joint attention 内部总长度 119,056，输出再拆流 |
| single-stream 输入 ×40 | `[1,119056,3072]` | 文本与视觉直接拼接 |
| final visual token | `[1,118800,64]` | $16\times1\times2\times2=64$/patch |
| unpatchify / 速度预测 | `[1,16,33,90,160]` | scheduler 每步保持 shape |
| VAE decode | `[1,3,129,720,1280]` | latent 除 scaling factor 后 tiled decode |

若显式设置 `cfg_scale>1`，latent、MLLM 与 CLIP 条件在去噪时 batch 维变成 2；默认 `cfg_scale=1` 配合 `embedded_cfg_scale=6`，则保持 batch=1。

### 7.3 逐模块四段拆解

#### 7.3.1 Causal 3D VAE

**输入输出**：`[B,3,129,720,1280]↔[B,16,33,90,160]`。**设计原因**：先把时空 token 压到可训练规模，同时用因果卷积与 $4n+1$ 对齐保留首帧。**关键超参**：latent 16、time ratio 4、space ratio 8、block channels 128/256/512/512、scaling factor 0.476986、tile overlap 0.25。**代码与论文差异**：论文披露训练损失与随机 tiling 微调；公开仓库只有结构、权重加载与推理 tiling，没有 VAE 训练 loop。

#### 7.3.2 MLLM+CLIP+Token Refiner

**输入输出**：提示词先套视频 system template，tokenizer 最大长度 351，裁掉前 95 个 instruction tokens 后得到至多 `[B,256,4096]`；Refiner 输出 `[B,256,3072]`；CLIP 输出 `[B,768]` pooled vector。**设计原因**：MLLM负责细节/推理，CLIP 提供稳定全局语义，双向 refiner 修复 causal encoding。**关键超参**：`text_len=256`、`hidden_state_skip_layer=2`、refiner depth=2、CLIP length 77。**代码与论文差异**：内部 HunyuanMLLM 未随初代仓库公开，模型卡推荐 LLaVA-Llama3-8B 替代；不能假设两者表示完全等价。

#### 7.3.3 Dual-/Single-stream MMDiT

**输入输出**：视觉 118,800 tokens、文本最多 256 tokens，20 层保持双流形状，40 层处理拼接序列，最后只取视觉部分恢复 16-channel latent。**设计原因**：前段避免模态统计互相干扰，后段最大化语义-视觉融合；full attention 统一图像/视频并复用 LLM kernel。**关键超参**：hidden 3072、FFN 12288、24 heads、head dim 128、QK RMSNorm、patch `(1,2,2)`、RoPE `(16,56,56)`。**代码与论文差异**：Figure 8 是概念图；代码显示 dual block 虽有独立 QKV/MLP，却在每一层已把 Q/K/V 拼成 joint attention，并非前 20 层完全不跨模态。

#### 7.3.4 Flow、CFG 与 decode

**输入输出**：scheduler 持有 `[B,16,33,90,160]`，每步 DiT 输出同形速度，Euler 更新后 VAE 解码。**设计原因**：Flow Matching 提供直线路径，time shift 把有限步数放在高噪声结构阶段；embedded guidance 避免两次前向。**关键超参**：50 steps、shift 7、Euler、embedded scale 6；低于 20 步论文建议 shift 约 17。**代码与论文差异**：公开默认是 CFG-distilled 权重；论文也描述 teacher CFG。若不了解这一点而同时把 `cfg_scale` 和 `embedded_cfg_scale` 都调高，容易过度引导。

### 7.4 计算账、显存与复现建议

- **权重与峰值**：官方 BF16/FP16 DiT checkpoint 文件约 25.6GB，FP8 文件约 13.2GB；FP8 官方称可节省约 10GB GPU memory。端到端峰值还包含 8B 级文本编码器、VAE、激活与 CUDA workspace，所以官方仍报告 720p 约 60GB、540p 约 45GB。
- **token 与 attention**：720p/129f 产生 118,800 视觉 tokens。dense attention 的 score 规模是 $O(L^2)$；必须使用 Flash/Fused Attention、VAE tiling，并在多卡时用 Ulysses/Ring context parallel。FP8 减的是权重显存，不会消除 full attention 算术量。
- **最低风险复现**：先按官方 544×960×129、50 steps、embedded scale 6、shift 7、CPU offload 验证；再升 720p。保持 BF16 DiT、FP16 文本编码器/VAE；显存不足优先降低分辨率，不要先缩帧到非 $4n+1$。
- **训练复现边界**：从零预训练不可由公开仓库严格复现。缺少绝对数据规模、过滤阈值、全局 batch、学习率曲线、GPU 总数、训练时长和 AngelPTM 代码。现实路线应是使用官方权重做 LoRA/下游适配，而非宣称复刻基座。
- **可检查的正确性**：固定 commit/revision、记录 prompt rewrite 是否启用、seed、分辨率、帧数、shift、step、guidance 类型和精度；对同一 prompt 至少多 seed 评估，避免把单个好样本当模型结论。

---

## 8. 生态演进：HunyuanVideo → I2V/Custom/Avatar → HunyuanVideo 1.5

### 8.1 初代基础模型内生的应用接口

主论文 v6 已描述三个与基座直接相关的方向。I2V 把参考图 latent 作为第一个 latent frame 并令其 timestep 为 0（token replace），其余帧正常加噪；参考图再经 MLLM 产生 semantic image tokens，与视频 tokens 做 full attention。Avatar 原型把参考图 VAE latent 沿时间复制并与噪声 latent 沿通道拼接，再以额外 cross-attention 注入音频/表情、以 pose latent 注入姿态。它们说明 13B 基座的主要扩展方式是“在 latent/token 接口增加条件”，而非重写生成目标。

### 8.2 HunyuanVideo-I2V、HunyuanCustom 与 HunyuanVideo-Avatar

- **HunyuanVideo-I2V（2025-03-06）**：把论文原型产品化，保留 token replace 与 MLLM semantic image tokens，支持最高 720p/129 帧，并公开特效 LoRA。官方给出的 720p 单卡峰值约 60GB。它是基础模型的条件扩展，不是 HunyuanVideo 1.5。
- **HunyuanCustom（2025-05-08）**：在 HunyuanVideo 上加入 LLaVA text-image fusion、沿时间强化 ID 的图像模块、AudioNet 空间 cross-attention 和 conditional video latent/patch feature alignment，面向单/多主体、音频驱动与视频驱动编辑。
- **HunyuanVideo-Avatar（2025-05-28）**：面向音频驱动角色动画，引入 character image injection、Audio Emotion Module 与 face-aware audio adapter，以 mask 隔离多角色的音频注入。它继承初代 MM-DiT，但评测目标变成身份、口型、情绪和多角色控制。

这三者都应视为“初代 HunyuanVideo 先验+条件适配器/专用微调”的生态，而不是基础 T2V 论文的额外实验成绩。

### 8.3 HunyuanVideo 1.5：独立后继版本

HunyuanVideo 1.5（arXiv:2511.18870）是 2025 年 11 月发布的独立 8.3B 模型。本文只做边界对照，代码固定到 commit `60783e704160023913bee78f0b47036d393d4dfa`：

| 维度 | 初代 HunyuanVideo | HunyuanVideo 1.5 |
|---|---|---|
| 主干 | 13B；20 dual + 40 single；full attention | 8.3B；54 dual-stream blocks；引入 SSTA |
| VAE | 4×8×8，16 latent channels | 4×16×16，32 latent channels |
| 文本/视觉条件 | MLLM + CLIP-L + bidirectional refiner | Qwen2.5-VL + Glyph-ByT5；I2V 另用 SigLIP |
| 任务组织 | 基础权重以 T2V 为主，I2V 后续独立扩展 | 同一框架联合 T2I/T2V/I2V |
| 训练 | 图像预训练→渐进图视频联合→高质量微调 | 更明确的多阶段预训练/CT/SFT/RLHF，使用 Muon |
| 高分辨率 | 基座直接支持 720p | 基座+独立 latent VSR 到 1080p |
| 稀疏注意力 | 无，依赖 full attention 与系统并行 | SSTA：局部 sliding tiles+动态 top-k blocks |

1.5 报告披露了 10M+ 原始视频小时、约 800M 过滤后 segments、5B 图像、13.6GB offload 峰值等数字；**这些数字只属于 1.5，不能用于填补初代未披露项。**同理，SSTA、Glyph-ByT5、Muon、VSR 和 8.3B 结果也都不是初代论文贡献。

---

## 9. 局限与批判性评价

1. **开放推理不等于开放训练。**论文最有价值的是数据与基础设施配方，但最不可复现的恰是数据、过滤器和 AngelPTM 训练；公开代码无法验证 5× 节省或 99.5% 稳定性的实现细节。
2. **评测依赖私有人评。**1,533 prompts、60 名专业评估者比少量 demo 严肃，但匿名竞品、缺少置信区间和公开自动 benchmark 限制了可复核性。
3. **模块归因不足。**缺少 MLLM vs T5、refiner on/off、dual/single 深度比例、过滤器边际贡献等完整消融；“系统有效”成立，“每个模块都必要”证据不足。
4. **full attention 是质量选择，也是部署瓶颈。**它让架构统一，却把 720p/129f 推理推到 60GB 级。后继 1.5 转向更高 VAE 压缩和 SSTA，恰好说明初代效率问题真实存在。
5. **数据透明度有限。**初代未披露训练视频总小时数、来源构成、具体许可比例和过滤误差；GDPR 合规声明不能替代可审计的数据卡。
6. **失败与安全评测不足。**论文没有系统报告物理错误、身份漂移、文字错误、偏见或内容安全的失败率。

---

## 10. 复现与实践建议

若目标是本地生成，先按官方权重跑 guidance-distilled T2V，不要混用 I2V/Avatar/Custom checkpoint。单卡 80GB 用 720p；显存更小时先降到 544×960，再启用 CPU offload、VAE tiling 和 FP8。保持 129 或其他 $4n+1$ 帧；默认 50 steps、shift 7、embedded scale 6 是稳定起点。低步数加速时同步增大 shift，而不是只砍 step。

若目标是微调，应明确任务接口：普通风格/主体适配优先 LoRA；首帧控制使用 I2V；多模态身份定制使用 HunyuanCustom；音频角色使用 Avatar。初代 T2V full attention 的训练成本不适合普通实验室从零复制。所有对比都应固定 commit、权重 revision、prompt rewrite 模式和 seed，并至少报告生成失败率、峰值显存与单样本时延。

---

## 11. 个人启发与后续问题

HunyuanVideo 最值得迁移的经验是：**先建立可扩展的数据/系统，再讨论大模型架构。**3D VAE 决定 token 预算，scaling law 决定参数/数据预算，progressive curriculum 决定收敛路径，MMDiT 决定条件融合，CFG distillation 决定部署成本；任何一环不足都会抵消“13B”的收益。

另一个启发是文本编码器并非越像 LLM 越好。decoder-only MLLM 的知识和推理有价值，但扩散条件需要双向、逐 token 的完整语义，因此还要 refiner 修正其因果性。后续值得回答的问题包括：3D RoPE 的 16/56/56 分配是否最优；dual→single 的切换深度能否由 token 动态决定；数据质量、模型规模和高分辨率课程能否放进同一 scaling law；稀疏 attention 能否在不依赖蒸馏的情况下保留初代 full attention 质量。

---

## 12. 材料来源

完整来源、Figure/Page、固定 commit/revision 与获取日期见：

- [assets/HunyuanVideo - 系统化大规模视频生成框架/sources.md](assets/HunyuanVideo%20-%20系统化大规模视频生成框架/sources.md)
- 主论文本地文件：`assets/HunyuanVideo - 系统化大规模视频生成框架/sources/paper.pdf`
- 1.5 技术报告本地文件：`assets/HunyuanVideo - 系统化大规模视频生成框架/sources/hunyuanvideo-1.5.pdf`

> 版本声明：除 §8.3 外，方法、实验与结论均以 HunyuanVideo 初代 arXiv:2412.03603 v6 和初代官方代码为准；§8.3 的所有 SSTA/8.3B/高压缩 VAE/1.5 数据与结果均明确隔离。