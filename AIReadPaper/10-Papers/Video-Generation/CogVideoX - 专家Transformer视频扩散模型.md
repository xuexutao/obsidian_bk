---
type: paper-note
status: complete
domain: Video-Generation
paper: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"
year: 2025
arxiv: 2408.06072
doi: 10.48550/arXiv.2408.06072
source: https://arxiv.org/abs/2408.06072v3
project: https://github.com/zai-org/CogVideo
code: https://github.com/zai-org/CogVideo
code_commit: 7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2
tags:
  - Video-Generation
  - Diffusion-Transformer
  - CogVideoX
  - Expert-Transformer
  - 3D-Causal-VAE
  - Full-Attention
  - Open-Source
created: 2026-11-05
updated: 2026-11-05
---

# CogVideoX：专家 Transformer 视频扩散模型

## 基本信息

- **作者**：Zhuoyi Yang、Jiayan Teng、Wendi Zheng、Ming Ding 等
- **机构**：清华大学知识工程实验室（KEG）与智谱 AI 团队
- **发表**：arXiv:2408.06072v3；ICLR 2025 接收
- **主论文版本**：v3，2025-03-26；本文所有“论文结论”以该版本为准
- **官方论文**：<https://arxiv.org/abs/2408.06072v3>
- **官方代码**：<https://github.com/zai-org/CogVideo>（旧 `THUDM/CogVideo` 已重定向）
- **代码固定提交**：[`7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2`](https://github.com/zai-org/CogVideo/tree/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2)
- **官方模型**：CogVideoX-2B、CogVideoX-5B、CogVideoX-5B-I2V；后续发布 CogVideoX1.5-5B 与 1.5-5B-I2V
- **核心组件**：3D causal VAE、T5-v1.1-XXL、Expert Transformer / Expert AdaLN、文本—视频 3D full attention、3D RoPE、v-prediction + zero-SNR、渐进式多分辨率训练
- **数据规模**：过滤后约 3500 万个平均 6 秒的单镜头视频片段，另混合约 20 亿张图像；论文未披露原始数据总量、去重细节和总训练算力

## 一句话结论

CogVideoX 的关键不只是“把 DiT 搬到视频上”，而是把视频先压缩为 $8\times8\times4$ 的时空潜变量，再把 T5 文本 token 与视频 patch token 放进同一次全注意力中，并用模态专属 Expert AdaLN 消除文本/视觉统计差异；配合渐进式长视频训练、Multi-Resolution Frame Pack、显式均匀时间步采样和密集视频 caption，它在 5B 规模上把开源文生视频推进到可生成大运动、长叙事和多主体交互的阶段，但代价是 3D full attention 的高昂二次复杂度，以及论文对训练成本、采样细节和最终失败案例披露不足。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

CogVideoX 要解决的不是单帧“看起来像照片”，而是文本条件下的视频联合建模：给定文本 $c$，从随机噪声生成视频 $x_0\in\mathbb{R}^{F\times H\times W\times3}$，要求主体身份、动作因果、镜头运动和背景在时间上连续，同时覆盖长文本中的对象、数量、属性与事件顺序。

2024 年前后的开源视频生成模型常见三个瓶颈。第一，像素或弱压缩潜空间导致 token 数过大，模型只能生成短、低分辨率片段。第二，把空间注意力和时间注意力拆开虽然省算力，却让跨帧大位移只能间接传递信息：下一帧移动后的头部 patch 不能直接寻找上一帧原位置的头部。第三，训练视频常有字幕、水印、剪辑、静态演讲和贫乏 caption；模型即使有容量，也难学到“文本中的动作如何随时间发生”。

### 1.2 核心思路

CogVideoX 把问题分成三层处理：

1. **表示层**：3D causal VAE 同时压缩空间与时间，把 $F\times H\times W$ 压到约 $(F-1)/4+1\times H/8\times W/8$；
2. **建模层**：将 T5 文本序列和视频 patch 序列拼接，由 Expert Transformer 做统一 3D full attention，但文本与视觉各自使用 Expert AdaLN；
3. **训练层**：用渐进式分辨率/时长课程、Multi-Resolution Frame Pack、Explicit Uniform Sampling 与高质量密集 caption，分别处理算力利用率、扩散时间步方差和语义监督质量。

### 1.3 主要贡献与证据边界

论文的实质贡献可归纳为：

- 一个 $8\times8\times4$、16 latent channels 的 3D causal VAE；在 WebVid 验证集 256×256、17 帧设置下取得 Flickering 85.5、PSNR 29.1，优于论文比较的 Open-Sora 与 Open-Sora-Plan VAE；
- 一个统一文本—视频序列的 Expert Transformer，以模态专属 AdaLN 替代 MMDiT 式双 Transformer，并以 3D full attention 直接建立跨时空关系；
- 一条从 256×384、6 秒逐步扩展到 768×1360、10 秒的训练课程，以及 mixed-duration frame packing；
- 约 3500 万过滤视频、20 亿图像和 CogVLM2-Caption 数据管线；
- 开源 2B/5B T2V、5B I2V、VAE 与 caption 模型。

需要提前划清边界：**CogVideoX1.5 是后续官方发布，不是主论文的一组消融。** 1.5 的 1360×768、16 fps、81/161 帧、时间 patch size 2 等信息来自仓库发布说明和模型配置，不能写成论文 Table 8 或 Figure 10 的实验结论。

![](assets/CogVideoX%20-%20专家Transformer视频扩散模型/figures/01-teaser.jpg)

> **图 1：长时、大运动生成示例。** 上半部分中“山石被闪电击碎、孙悟空跃出”的事件横跨多个阶段，考察的是动作因果、主体显现和大位移；下半部分“秃头男子戴上彩色假发”考察物体—人体交互和身份保持。它直观体现论文要突破的并非静态画质，而是跨帧事件连续性。  
> 来源：论文 Figure 1，PDF 首页附近；官方 arXiv HTML 原图，<https://arxiv.org/html/2408.06072v3>。

---

## 2. 任务定义、符号与扩散目标

### 2.1 T2V 输入输出

文生视频输入为英文文本提示 $c$。T5 encoder 输出文本表示：

$$z_{text}=E_{T5}(c)\in\mathbb{R}^{B\times L_t\times4096}.$$

官方 5B diffusers 配置将 $L_t$ 上限设为 226。视频经 VAE 编码：

$$z_0=E_{VAE}(x_0)\in\mathbb{R}^{B\times C_z\times F_z\times H_z\times W_z},$$

其中 $C_z=16$，空间压缩率为 8，时间压缩率为 4。对于首帧因果保留的实现，可用下式理解时间长度：

$$F_z=1+\left\lfloor\frac{F-1}{4}\right\rfloor.$$

这使 49 帧视频变为 13 个 latent frames，81 帧变为 21 个 latent frames。

### 2.2 前向加噪与 v-prediction

官方 scheduler 配置使用 1000 个训练时间步、scaled-linear beta、zero terminal SNR 和 `v_prediction`。标准前向过程为：

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).$$

速度目标为：

$$v_t=\sqrt{\bar\alpha_t}\epsilon-\sqrt{1-\bar\alpha_t}z_0.$$

模型预测 $\hat v_\theta(z_t,t,c)$，以加权均方误差训练。当前官方 diffusers 微调代码先调用 scheduler 的 `get_velocity` 将网络输出转换为 latent 预测，再按 $1/(1-\bar\alpha_t)$ 加权重建 latent：

$$\mathcal{L}=\mathbb{E}\left[\frac{1}{1-\bar\alpha_t}\left\|\hat z_0-z_0\right\|_2^2\right].$$

这段公开微调实现可验证 `v_prediction` 与权重形式，但不应反推为论文基座预训练的全部实现；主论文只明确说明 v-prediction、zero SNR 和 LDM noise schedule，未公开完整预训练 loop。

### 2.3 CFG 与采样

推理以 classifier-free guidance 合并正、负条件预测：

$$\hat v_{cfg}=\hat v_{uncond}+s\left(\hat v_{cond}-\hat v_{uncond}\right).$$

当前官方 CLI 默认 50 步、guidance scale 6，并将 scheduler 切为 `CogVideoXDPMScheduler`、`timestep_spacing="trailing"`、`use_dynamic_cfg=True`；代码注释建议 2B 用 DDIM，5B/I2V 用 DPM。论文正文没有披露 CFG scale、负提示、具体求解器等，因此这些值属于**当前开源推理配方**，不是论文消融。

---

## 3. 核心方法

### 3.1 总体架构

CogVideoX 的主链路是：像素视频经 3D causal VAE 变为 latent；latent 被 patchify 为 $z_{vision}$；T5-v1.1-XXL 把文本变为 $z_{text}$；两种 token 拼接后进入 $N$ 层 Expert Transformer；网络输出再 unpatchify 为 latent 速度/去噪预测，最后由 VAE decoder 还原视频。

![](assets/CogVideoX%20-%20专家Transformer视频扩散模型/figures/02-overall-architecture.png)

> **图 2：CogVideoX 总体架构与 Expert Transformer block。** 图底部是文本与视频两条编码支路；进入 block 后，两种 token 共享 3D full attention 与 FFN，因此可深度交换信息，但左、右两侧分别由 Text Expert AdaLN 和 Vision Expert AdaLN 产生 scale、shift、gate。换言之，“expert”不是稀疏 MoE 路由，而是**按模态分开的归一化/调制专家**。  
> 来源：论文 Figure 3，第 2 节（PDF 前部方法页）；官方 arXiv HTML 原图，<https://arxiv.org/html/2408.06072v3>。

### 3.2 3D causal VAE

#### 3.2.1 输入输出

VAE 输入为 $[B,3,F,H,W]$，输出 posterior 的 16 通道 latent。默认空间压缩 $8\times8$，时间压缩 $4\times$；Decoder 做逆变换。Encoder/Decoder 由多级 ResNet blocks 组成，前三个尺度中穿插空间和时间下/上采样，最深层只做空间采样，并用 KL regularizer 约束潜空间。

#### 3.2.2 为什么是因果卷积

时间卷积把 padding 放在序列开头，使第 $t$ 个位置只依赖当前和过去帧。因果性有两个价值：一是首帧可单独编码，适合图像与视频联合训练；二是长视频可沿时间切分给不同 rank。若 temporal kernel 为 $k$，每个 rank 只需把尾部 $k-1$ 个 feature 发送给下一 rank，而不是交换整个序列。

#### 3.2.3 关键超参与训练

论文 VAE 先在 256×256、17 帧、8/16 fps 数据上训练，再用 context parallel 微调到 161 帧。损失包含加权 L1、LPIPS、KL，并在数千步后加入 3D discriminator 的 GAN loss；具体损失权重、加入 GAN 的精确步数和判别器结构未披露。论文发现 VAE 不含 attention，因此对更高空间分辨率可直接泛化；但对更多帧泛化不足，161 帧微调是必要步骤。

VAE 消融如下：

| 变体 | 压缩率 | latent channels | Flickering ↓ | PSNR ↑ |
|---|---:|---:|---:|---:|
| SDXL 2D VAE baseline | 8×8×1 | 4 | 93.2 | 28.4 |
| A | 8×8×4 | 8 | 87.6 | 27.2 |
| **B（论文预训练采用）** | **8×8×4** | **16** | **86.3** | **28.7** |
| C | 8×8×4 | 32 | 87.7 | **30.5** |
| D | 8×8×8 | 32 | 87.8 | 29.0 |
| E | 16×16×8 | 128 | 87.3 | 27.9 |

Variant C 的 PSNR 更高，但主模型选择 B。论文没有给出完整取舍；更小 latent channel 会显著降低后续 DiT 的输入投影和存储成本，是合理解释，但属于工程判断而非作者明说。更重要的负结果是：16×16×8 即使有 128 channels 也难稳定收敛，说明“压得越狠越好”并不成立。

![](assets/CogVideoX%20-%20专家Transformer视频扩散模型/figures/03-3d-causal-vae.jpg)

> **图 3：3D causal VAE 与时间 context parallel。** 左图展示空间/时间交错压缩到 $(f+1)\times h\times w$ latent；右图展示 rank 0 用首帧 padding，后续 rank 接收前一 rank 的尾部 $k-1$ 帧作为 causal padding。它解释了模型为何既能做时间压缩，也能把长视频编码分散到多卡。  
> 来源：论文 Figure 4，第 2.1 节（PDF 方法部分）；官方 arXiv HTML 原图，<https://arxiv.org/html/2408.06072v3>。

#### 3.2.4 代码与论文差异

公开 diffusers VAE 配置是 `AutoencoderKLCogVideoX`，`block_out_channels=[128,256,256,512]`、每块 3 层、GroupNorm 32 组、`scaling_factor=0.7`。5B 与 1.5 都是 16 latent channels、4×8×8 压缩；1.5 额外设置 `invert_scale_latents=true`。论文的方法描述强调因果卷积和 context parallel，当前主分支则通过外部 diffusers 类执行 VAE，并以 slicing/tiling/CPU offload 为部署重点；不能把 diffusers 的 tiling 当作论文训练时的 context parallel。

### 3.3 Patchify 与 token 数

#### 3.3.1 一般形式

设 VAE latent 为 $T\times H\times W\times C$，空间 patch size 为 $p$、时间 patch size 为 $q$，则视觉 token 数为：

$$L_v=\frac{T}{q}\frac{H}{p}\frac{W}{p}.$$

当 $q>1$ 且 latent 时间长度不能整除 $q$ 时，论文提出复制序列开头的第一帧。当前 1.5 微调代码正是计算 `ncopy = latent.shape[2] % patch_size_t`，再把首帧复制到序列前端。

#### 3.3.2 5B 与 1.5 的关键差异

初代 CogVideoX-5B 的模型配置只有空间 `patch_size=2`，没有 temporal patch；49 帧经 VAE 得 13 latent frames，patch grid 是 $13\times30\times45$，共有 17,550 个视觉 token。

CogVideoX1.5-5B 新增 `patch_size_t=2`。81 帧的名义 VAE 长度是 21 latent frames；训练时微调代码在序列开头复制首帧补为 22，T2V 推理时 pipeline 则直接按补齐后的 22 帧创建噪声 latent，去噪结束后从开头裁掉额外 1 帧，再把 21 帧交给 VAE decode。按时间 2、空间 2×2 patchify 后，grid 是 $11\times48\times85$，共有 44,880 个视觉 token。若没有时间 patch，token 会翻倍；因此 1.5 能提高分辨率和帧率，一个关键补偿就是额外的时间 patchification。

### 3.4 T5 文本编码器

论文正文只写 T5，官方模型配置把它具体化为 **T5-v1.1-XXL encoder**：24 层、64 heads、hidden 4096、FFN 10240、gated-GELU，词表 32,128。文本最多 226 token，输出 $[B,226,4096]$，由 patch embedding 模块映射到 Transformer hidden 3072。

采用 encoder-only T5 的直接好处是双向文本上下文适合描述对象、属性、动作和镜头语言；代价是 T5-XXL 本身很大。当前微调框架冻结 VAE 和文本编码器，并预计算视频 latent 与 prompt embedding，避免每一步重复执行二者。

模型卡把 1.5 的 prompt limit 写为 224，而 transformer config 仍是 226。本文在张量表中以机器可读配置的 226 为准，并把 224 视为模型卡/配置口径差异，不能擅自消除。

### 3.5 Expert Transformer 与 Expert AdaLN

#### 3.5.1 输入输出

Text tokens 与 vision tokens 在序列维拼接：

$$h=[h_{text};h_{vision}]\in\mathbb{R}^{B\times(L_t+L_v)\times D}.$$

5B 的 $D=48\times64=3072$，共 42 层。每层对整个联合序列执行 attention 和 FFN，最后只保留视觉 token，经输出归一化、线性层和 unpatchify 得到 16 通道预测。

#### 3.5.2 设计原因

文本与视觉数值分布不同：文本是离散 token 的语言表征，视频是连续潜变量 patch。若完全共享 AdaLN，时间步调制必须同时适配两种统计；若像 MMDiT 那样使用两套完整 Transformer，参数和计算显著增加。Expert AdaLN 选择中间方案：attention/FFN 共享，但文本与视觉各有 scale、shift、gate，既允许联合注意力，又保留模态专属校准。

#### 3.5.3 关键超参

2B 为 30 层、32 heads、hidden 1920，论文附录位置编码列为 sinusoidal；5B 为 42 层、48 heads、hidden 3072，使用 RoPE。两者 time embedding size 256（当前 diffusers 投影后的 `time_embed_dim` 为 512）、Adam $\beta=(0.9,0.95)$、$\epsilon=10^{-8}$、weight decay $10^{-4}$、cosine decay、gradient clipping 1.0、BF16（5B）。

#### 3.5.4 代码与论文差异

当前仓库不再在主分支内实现 SAT 的 block 源码，而是从 diffusers 加载 `CogVideoXTransformer3DModel`。因此可从配置和训练调用验证 hidden、层数、heads、RoPE、输入输出通道和 timestep 条件，但不能把现行 diffusers 类的所有内部细节称作论文 SAT 原实现。权重转换脚本明确把 SAT 权重映射到 diffusers，并为 I2V 把 `in_channels` 从 16 改为 32；这是两套实现的桥，而不是两套模型可以混用配置。

### 3.6 文本—视频 3D full attention

#### 3.6.1 输入输出

联合序列的 Q/K/V 具有 $[B,h,L,d_h]$ 形状，其中 $h=48$、$d_h=64$、$L=L_t+L_v$。注意力为：

$$\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.$$

“3D”并非使用三维卷积式局部窗口，而是视频 token 携带 $(t,y,x)$ 位置后，在整个时空序列上做 full attention；文本 token 也在同一次 attention 中参与。

#### 3.6.2 设计原因

2D spatial + 1D temporal attention 要经过中间 patch 才能传播跨帧大位移信息。full attention 允许任意两帧、任意位置直接建立关系，对快速运动、遮挡后重现和镜头运动更自然。论文消融显示 2D+1D 在早期 FVD 更差、训练更易 collapse，5B 时不稳定更明显。

#### 3.6.3 计算代价

full attention 的 score 矩阵复杂度为 $O(L^2)$。论文在 H800 上测得单次 DiT forward：

| 输入规格 | 2D+1D | 3D full attention |
|---|---:|---:|
| 256×384×6s | 0.38 s | 0.41 s |
| 480×720×6s | 1.26 s | 2.11 s |
| 768×1360×5s | 4.17 s | 9.60 s |

最高规格下 full attention 已是分解注意力的 2.3 倍。这是 CogVideoX 的核心能力来源，也是扩展长视频最直接的瓶颈。

#### 3.6.4 工程实现

论文借助 FlashAttention 和并行化控制中间显存。按默认 1.5 形状，视觉 token 44,880，加 226 文本 token 后 $L=45,106$；若显式保存单头 BF16 attention score，单层单头约需 $45,106^2\times2\approx4.07$ GB，48 heads 更不可行。FlashAttention 通过分块在线 softmax 避免物化完整 $L\times L$ 矩阵，但 FLOPs 的二次增长仍存在。

### 3.7 3D RoPE 与位置外推

每个视频 token 有 $(x,y,t)$ 坐标。论文将 head channel 的 $3/8$、$3/8$、$2/8$ 分给 x、y、t，分别施加 1D RoPE 后拼接。对更高分辨率，作者比较：

- **interpolation**：把固定位置范围缩放到新 grid，较保留全局布局，却可能模糊局部；
- **extrapolation**：扩大位置表并按实际 grid 截取，保持相邻 token 的相对间隔，局部更清晰，但训练初期会出现多个小而重复的对象。

CogVideoX 选择 extrapolation。需注意：论文附录明确 2B 使用 sinusoidal、5B 使用 RoPE，所以不能写“所有 CogVideoX 都采用 3D RoPE”。当前 5B 与 1.5 transformer config 均设 `use_rotary_positional_embeddings=true`。

### 3.8 T2V 与 I2V 条件差异

T2V 的 Transformer 输入是 16 通道 noisy video latent，条件文本通过联合序列进入 attention。I2V 从 T2V 微调而来，额外把参考图像编码成 16 通道 latent，沿**通道维**与 noisy video latent 拼接，故 I2V Transformer 输入通道为 32，而输出仍为 16。

训练时参考图像通常取目标视频第一帧。为缩小“训练首帧”和“推理独立真实图像”的分布差，公开 I2V 微调代码给图像加 log-normal 强度噪声：

$$\log\sigma_{img}\sim\mathcal{N}(-3,0.5^2),\qquad \tilde I=I+\sigma_{img}\epsilon.$$

图像只占第一个 latent frame，其余时间位置补零；随后与 noisy video latent 拼接。1.5-I2V 还通过 `ofs_embed_dim=512` 注入 offset 条件。初代 5B-I2V 使用 RoPE 加 learnable positional embedding，而 1.5-I2V 使用 RoPE；这些均来自发布配置，不是主论文消融。

### 3.9 Progressive Training

高分辨率长视频直接训练会让序列、激活和 attention FLOPs 同时爆炸，因此论文采用四阶段课程：

| 阶段 | 最大分辨率 | 最大时长 | Batch size | 论文列出的 sequence length | Steps |
|---|---:|---:|---:|---:|---:|
| Stage 1 | 256×384 | 6 s | 2000 | 25k | 400k |
| Stage 2 | 480×720 | 6 s | 1000 | 75k | 220k |
| Stage 3 | 768×1360 | 10 s | 250 | 700k | 120k |
| Stage 4 高质量微调 | 768×1360 | 10 s | 100 | 700k | 10k |

课程先学习低频布局与语义，再增加空间细节和时间跨度。正文把分辨率概括为约 256→512→768，而表格第二阶段是 480×720；本文以表格精确值为准。最后使用过滤数据中质量最高的 20% 微调，字幕和水印减少、视觉质量略升，但作者同时观察到语义能力略降，说明美学筛选会牺牲概念覆盖。

### 3.10 Multi-Resolution Frame Pack

固定帧数训练会丢弃短视频、截断长视频，并造成图像 1 帧与视频几十帧之间巨大的 token 数差异。Multi-Resolution Frame Pack 将不同时长、分辨率和宽高比样本组合成总形状相近的 pack，让同一 batch 更充分地利用 token 预算；3D RoPE 负责为每个样本提供正确时空坐标。

论文没有公开 pack 构造伪代码、样本间 attention mask、padding 比例或每 pack 样本数，因此不能声称已能从公开仓库严格复现这一基座预训练技巧。当前 diffusers 微调数据集是固定训练分辨率加 bucket sampler，与论文的大规模 Frame Pack 概念相关但不等价。

### 3.11 Explicit Uniform Sampling

普通数据并行中，每个 rank 独立从 $[1,T]$ 均匀采样时间步；期望虽均匀，有限 global batch 仍可能集中在某些噪声区，而不同 $t$ 的 loss 尺度差异会造成曲线抖动。若有 $n$ 个 data-parallel ranks，CogVideoX 把 $[1,T]$ 分成 $n$ 段，让第 $i$ 个 rank 只在第 $i$ 段内均匀采样，从全局 batch 看显式覆盖整个噪声区间。

40k 步验证 loss：

| Timestep | 100 | 300 | 500 | 700 | 900 |
|---|---:|---:|---:|---:|---:|
| 常规独立均匀采样 | 0.222 | 0.130 | 0.119 | 0.133 | 0.161 |
| **Explicit Uniform Sampling** | **0.216** | **0.126** | **0.116** | **0.129** | **0.157** |

五个时间步都改善，幅度不大但一致；更重要的是训练曲线更稳定。它不改变单样本目标分布，而是降低有限 global batch 下的 stratified sampling 方差。

---

## 4. 数据工程与 caption

### 4.1 视频过滤

论文从原始视频切出 single-shot clips，人工标注 20,000 段视频，并基于 Video-LLaMA 训练六类过滤器：Editing、Lack of Motion Connectivity/Static、Low Quality、Lecture、Text Dominated、Noisy Screenshots；同时使用 optical-flow score 和图像 aesthetic score，并在训练中动态调阈值。

过滤器测试结果：

| 分类器 | TP | FP | TN | FN | Accuracy |
|---|---:|---:|---:|---:|---:|
| Editing | 0.81 | 0.02 | 0.09 | 0.08 | 0.91 |
| Static | 0.48 | 0.04 | 0.44 | 0.04 | 0.92 |
| Lecture | 0.52 | 0.00 | 0.47 | 0.01 | 0.99 |
| Text | 0.60 | 0.03 | 0.36 | 0.02 | 0.96 |
| Screenshot | 0.61 | 0.01 | 0.37 | 0.01 | 0.98 |
| Low Quality | 0.80 | 0.02 | 0.09 | 0.09 | 0.89 |

最终得到约 3500 万段、平均 6 秒的单镜头视频。论文没有给出各过滤器阈值、每类删除比例、原始视频总量、版权与去重策略，这些是复现数据分布的主要缺口。

### 4.2 图像混合训练

额外使用 LAION-5B 和 COYO-700M 中经 aesthetic score 过滤的约 20 亿图像，把图像视为单帧视频。图像提供广泛概念和高质量静态外观，视频提供运动监督。时间 patch $q>1$ 时复制首帧，使单图也能形成合法 patch，避免另设完全独立的图像网络。

### 4.3 从帧 caption 到 CogVLM2-Caption

第一代 dense caption 流程是：Panda-70M 模型先给短视频描述；每 2 秒抽帧；CogVLM 给每帧生成密集图像描述；GPT-4 汇总为最终视频 caption。团队收集 50,000 条 GPT-4 汇总结果，微调 LLaMA2 来替代昂贵的 GPT-4 批量推理。

随后又训练端到端 CogVLM2-Caption：以 CogVLM2-Video 和 Llama 3 为基础，直接描述对象、动作及其时间变化。这个模型降低 caption 成本，也减少“先逐帧、再文本汇总”丢失时序关系的问题。论文未披露 CogVLM2-Caption 参数量、训练步数、caption 长度分布或定量 caption 指标。

### 4.4 Prompt upsampling

训练 caption 比真实用户 prompt 更长、更具体。推理时官方建议用 LLM 把短提示扩写为英语长描述；I2V 可用 GPT-4V 或 CogVLM 同时理解参考图。它不是可有可无的文案优化，而是减小训练—推理文本分布差异的重要步骤。若直接输入短中文提示，既有语言分布偏移，又缺少训练 caption 中的动作、镜头、材质和时间信息。

---

## 5. 实验结果与消融

### 5.1 VAE 重建

在 WebVid validation、256×256、17 帧上：

| VAE | Flickering ↓ | PSNR ↑ |
|---|---:|---:|
| Open-Sora | 92.4 | 28.5 |
| Open-Sora-Plan | 90.2 | 27.6 |
| **CogVideoX VAE** | **85.5** | **29.1** |

两项同时领先，说明 3D 建模不仅提高逐帧重建，也降低时序闪烁。不过该评测规模和测试视频长度有限，不能直接推出它对 161 帧、768×1360 仍保持同等优势。

### 5.2 自动视频指标

论文 Table 3 的关键结果：

| 模型 | Human Action | Scene | Dynamic Degree | Multiple Objects | Appearance Style | Dynamic Quality | GPT4o-MTScore |
|---|---:|---:|---:|---:|---:|---:|---:|
| VideoCrafter-2.0 | 95.0 | 55.29 | 42.50 | 40.66 | **25.13** | 43.6 | 2.68 |
| OpenSora V1.2 | 85.8 | 42.47 | 47.22 | 58.41 | 23.89 | 63.7 | 2.52 |
| CogVideoX-2B | 96.6 | 55.35 | **66.39** | 57.68 | 24.37 | 57.7 | 3.09 |
| **CogVideoX-5B** | **96.8** | 55.44 | 62.22 | **70.95** | 24.44 | **69.5** | **3.36** |

5B 的优势集中在人体动作、多物体、动态质量和文本—视频综合评分；Dynamic Degree 则是 2B 更高。论文文字称“5B 在七项中五项最佳”，但逐项读表只能确认 5B 单模型 4/7 最佳；若把 2B 与 5B 合称 CogVideoX 系列，才是 5/7。这一处应以表格为准。

### 5.3 人工评估

100 个手工 prompt 上，论文与 2024 年 7 月版本 Kling 对比：

| 模型 | Sensory Quality | Instruction Following | Physics Simulation | Cover Quality | Total Score |
|---|---:|---:|---:|---:|---:|
| Kling | 0.638 | 0.367 | 0.561 | 0.668 | 2.17 |
| **CogVideoX-5B** | **0.722** | **0.495** | **0.667** | **0.712** | **2.74** |

四个分项均领先，但 Total Score 与四项简单相加不一致：Kling 分项和 2.234，5B 分项和 2.596。附录又称总分范围 0–5，而正文只列四项；论文没有说明第五项或聚合公式。因此可以引用“分项领先”，不应自行解释总分算法或显著性。

### 5.4 架构消融

![](assets/CogVideoX%20-%20专家Transformer视频扩散模型/figures/04-ablation.png)

> **图 4：Expert AdaLN、3D attention 与 Explicit Uniform Sampling 消融。** (a)(d) 比较 Expert AdaLN、无 Expert AdaLN 与两种 MMDiT；(b) 显示 2D+1D attention 在约 20k 步后 FVD 爆炸，而 3D full attention 稳定下降；(c) 显式均匀采样使 FVD 曲线后期更低、更平稳。该图支持三个设计组合使用，但不能从图中恢复未标注的精确柱值。  
> 来源：论文 Figure 8，第 4.1 节（PDF 实验部分）；官方 arXiv HTML 原图，<https://arxiv.org/html/2408.06072v3>。

**Expert AdaLN。** 论文比较同参数量 MMDiT1、同层数但约两倍参数的 MMDiT2、Expert AdaLN 和无 Expert AdaLN。Expert AdaLN 在 FVD、CLIP4Clip 和训练 loss 上更优，说明“共享 attention/FFN + 模态专属调制”比完全共享归一化更稳，也比双塔更节省参数。

**3D full attention。** 2D+1D 方案早期 FVD 更高，并在更大模型上更容易 collapse；full attention 的代价由 Table 8 的单步耗时清晰量化。因此结论不是“3D 始终免费更好”，而是“用更多计算换训练稳定性和大运动建模”。

**3D RoPE。** 3D RoPE 的 loss 收敛快于 sinusoidal absolute position embedding；但论文附录没有提供每点数值。2B 最终仍采用 sinusoidal，5B 才采用 RoPE，说明该消融结论也受到模型规模和具体版本约束。

**Explicit Uniform Sampling。** 在五个固定 timestep 上验证 loss 全部下降，并降低曲线波动，是论文中数值最完整、也最容易复现的训练技巧之一。

### 5.5 推理效率

论文 H800、50 steps：

| 模型与规格 | 时间 | 显存 |
|---|---:|---:|
| 2B，480×720，6 s | 49 s | 18 GB |
| 5B，480×720，6 s | 113 s | 26 GB |
| 2B，768×1360，5 s | 220 s | 53 GB |
| 5B，768×1360，5 s | 500 s | 76 GB |

论文没有报告 768×1360、10 秒的推理时间/显存。当前仓库模型卡的 5B SAT 26 GB、1.5 SAT 76 GB 与上述量级相符；diffusers 所称 5 GB/10 GB 起依赖 sequential CPU offload、VAE slicing/tiling，意味着大量权重不常驻 GPU，不能与 SAT 全驻留显存直接比较。

### 5.6 失败案例与未覆盖能力

论文没有系统展示最终生成模型的失败样例，也没有统计手部、文字、计数、身份交换、物理错误、长视频漂移或 I2V 首帧保持失败率。可确认的负结果只有：高压缩 VAE 难收敛；VAE 对更多帧不能零样本泛化；高质量微调略损语义；3D full attention 显著增算力；RoPE 外推在早期出现重复小对象。

因此，对“CogVideoX 已解决复杂物理、长叙事和 10 秒身份一致性”的表述应谨慎。论文证明的是相对基线和选定指标上的改善，不是这些问题被彻底解决。

---

## 6. 计算账、显存与扩展瓶颈

### 6.1 默认 token 规模

| 模型 | 像素视频 | VAE latent `[F,C,H,W]` | patch grid | 视觉 tokens | 加文本后的序列 |
|---|---:|---:|---:|---:|---:|
| CogVideoX-5B | 49×480×720 | `[13,16,60,90]` | `[13,30,45]` | 17,550 | 17,776 |
| CogVideoX1.5-5B | 81×768×1360 | `[21,16,96,170]`；复制首帧后 22 | `[11,48,85]` | 44,880 | 45,106 |

1.5 的 token 数约为初代 2.56 倍，attention score FLOPs 约按 $2.56^2\approx6.58$ 倍增长；时间 patch size 2 已在抵消这一增长，否则视觉 token 将接近 9 万。

### 6.2 参数与显存

5B Transformer 实际在模型卡中显示约 6B params；BF16 仅权重约 12 GB，再加 T5-XXL、VAE、临时张量和 KV/activation，SAT 推理报告 26 GB。1.5 架构宽度和层数未增，但序列更长、激活更大，SAT 报告约 76 GB。diffusers 的 5/10 GB“最低显存”依赖 CPU offload，不代表总内存或速度更低。

训练若使用 AdamW，单个可训练参数通常需权重、梯度和两个优化器状态，朴素混合精度可轻易达到每参数 12–16 bytes；5B 级全量微调不可能在单张消费卡常规完成。官方当前 diffusers 指南给出：5B LoRA rank 128、49×480×720 约 24 GB；1.5 LoRA、81×768×1360 约 35 GB；1.5 SFT 加 ZeRO/offload 仍约 40–56 GB/卡。数字依赖软件版本和数据形状，只应作为官方测试点。

### 6.3 FLOPs 直觉

每层 full attention 的 QK 与 AV 主项约为：

$$\operatorname{FLOPs}_{attn}\approx4L^2D.$$

1.5 默认 $L=45,106,D=3072$，仅该主项每层约 $2.50\times10^{13}$ 次运算，42 层约 $1.05\times10^{15}$，还未计 QKV/输出投影、FFN、CFG 双分支与 50 个采样步。FlashAttention 减少显存读写，但不会消除 $L^2$ 算术量，这解释了模型卡中 1.5 单 H100 生成 5 秒视频仍约 550 秒。

---

## 7. 模型演进：CogVideo → CogVideoX → CogVideoX1.5

### 7.1 CogVideo（ICLR 2023）

旧版 CogVideo 来自 arXiv:2205.15868，2022 年公开、ICLR 2023 发表。它是继承 CogView2 文生图模型的 **9B 自回归 Transformer**，用 multi-frame-rate hierarchical training 缓解文本—视频数据稀缺和从零训练成本。核心范式是离散 token 的层级自回归生成，与 2024 年 CogVideoX 的 latent diffusion Transformer 不同。

### 7.2 CogVideoX-2B / 5B / 5B-I2V

2024-08-06 发布 2B 与 3D causal VAE；08-27 发布 5B；09-19 发布 5B-I2V 和 CogVLM2-Caption。初代开源规格为 720×480、49 帧、8 fps、约 6 秒。2B 是 30 层、1920 hidden、sinusoidal；5B 是 42 层、3072 hidden、3D RoPE。I2V 把图像 latent 与 noisy video latent 沿通道拼接，并针对独立图像加噪。

这一步的本质变化是：**9B 自回归像素/离散 token 视频模型 → 2B/5B latent diffusion Transformer**；计算集中在压缩 latent 上，并以统一文本—视频 attention 改善语义对齐。

### 7.3 CogVideoX1.5

1.5 于 2024-11-08 由官方仓库发布，11-15 补齐 diffusers 版本。T2V 默认 1360×768、16 fps，支持 81 帧约 5 秒和 161 帧约 10 秒；I2V 支持短边 768、长边 768–1360 且长边可被 16 整除。配置新增 temporal `patch_size_t=2`，使 81 帧高分辨率的 token 数仍可控制；VAE 设置 `invert_scale_latents=true`。

**论文归属必须谨慎**：1.5 没有独立论文，官方模型卡仍引用 arXiv:2408.06072。论文训练计划确实讨论 10 秒、768×1360，但 1.5 的具体默认帧数、16 fps、patch size 2、显存和推理速度来自发布说明与 config，不是论文消融。不得把“1.5 比 5B 更好”写成论文 Table 3 的结论。

---

## 8. 局限与批判性评价

1. **算力扩展依旧受 $O(L^2)$ 限制。** 3D full attention 对大运动有效，但默认 1.5 已有 45k 联合 token；扩展到更长、更高分辨率需要时域 patch、并行或稀疏化，而不能只加 GPU。
2. **训练成本不可复现。** 论文给出 stages、batch 和 steps，却未披露 GPU 型号/数量、训练时长、总 FLOPs、优化器学习率峰值和完整数据配比。学术团队能复现微调，难以复现基座。
3. **Frame Pack 细节不足。** 这是高利用率 mixed-duration training 的关键，但 mask、packing 算法和 padding 统计缺失。
4. **数据可追溯性有限。** 3500 万视频与 20 亿图像的过滤框架清楚，原始来源比例、许可、去重和安全过滤却没有完整披露。
5. **评测存在口径问题。** Table 3 “5B 五项最佳”与逐项数字不一致；人工评估 Total Score 无法由四项相加得到，标注人数、一致性和显著性也未给出。
6. **失败案例披露不足。** 论文没有系统展示最终模型错误，因此定性图更适合说明“能做到什么”，不适合估计失败概率。
7. **T2V 与 I2V 的工程路径分裂。** I2V 需要 32 通道独立 checkpoint；初代与 1.5 的位置条件、offset 和尺寸约束也不同，不能随意互换权重。
8. **开源实现边界复杂。** SAT 是原生研究实现，diffusers 是部署/微调实现；当前 main 主要调用外部 diffusers。若不固定 commit、模型 revision 和 diffusers 版本，很容易把两套张量顺序与配置混写。

---

## 9. 复现与实践建议

### 9.1 推理

- 初次验证优先用 CogVideoX-5B、49×480×720、BF16、50 steps、CFG 6；显存不足时启用 sequential CPU offload、VAE slicing 与 tiling，但要接受速度下降。
- CogVideoX1.5-5B 用 81×768×1360；161 帧是 10 秒上限，但算力与内存明显更高。模型卡报告 5 秒单 H100 约 550 秒。
- 2B 用 FP16，5B/1.5 用 BF16；不要因显卡支持而默认把 5B 改为 FP16，官方明确推荐训练精度。
- Prompt 使用英语长描述，明确主体、动作阶段、场景、镜头、光照和风格；短提示先做 prompt upsampling。
- 固定 seed、scheduler、steps、CFG、diffusers 版本和输出 fps。仓库 CLI 默认 fps=16，但初代 5B 模型原生是 8 fps，调用初代模型时应显式改回 8，避免只改变播放速度造成误判。

### 9.2 微调

- 小数据个性化优先 LoRA，不从零训练；官方 5B LoRA rank 128 测试约 24 GB，1.5 约 35 GB。
- 预先将数据 crop+resize 到目标比例。官方脚本会直接 resize 不匹配样本，可能拉伸主体。
- T2V 数据为 `prompts.txt + videos/ + videos.txt`；I2V 可显式给 `images/`，缺失时取视频第一帧。
- 预计算的 latent 与 prompt embedding 会缓存到磁盘；修改原视频或 prompt 后要清理相应缓存，否则可能继续训练旧内容。
- I2V 需使用 I2V checkpoint；T2V 的 16 通道输入不能直接接收 32 通道 noisy-video + image-latent。

### 9.3 不可复现项

截至 2026-11-05，公开资料仍不足以严格复现论文基座的数据采集、Frame Pack、完整 progressive pretraining、Explicit Uniform Sampling 在原始 SAT 训练栈中的实现，以及总训练成本。最现实的复现目标是：固定官方权重做推理、LoRA/SFT 和模块级验证，而非从零复刻 5B 预训练。

---

## 10. 代码级实现

### 10.1 版本边界

本节固定官方仓库 commit [`7a1af715`](https://github.com/zai-org/CogVideo/tree/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2)。模型配置固定到 CogVideoX-5B revision [`8fc5b281`](https://huggingface.co/zai-org/CogVideoX-5b/tree/8fc5b281006c82b82d34fd2543d2f0ebb4e7e321) 与 CogVideoX1.5-5B revision [`fdc5267c`](https://huggingface.co/zai-org/CogVideoX1.5-5B/tree/fdc5267c90b5c06492985b966e43aae984e189e0)。

当前 main 是 **diffusers 调用、微调与权重转换层**，并不内嵌 `CogVideoXTransformer3DModel`/`AutoencoderKLCogVideoX` 全部实现；SAT 原生研究栈应从官方 SAT 权重/说明进入。以下张量顺序按 diffusers：视频 latent 在 VAE 阶段为 `[B,C,F,H,W]`，送 Transformer 前为 `[B,F,C,H,W]`。

### 10.2 论文模块 → 代码入口

| 论文模块 | 当前官方代码/配置入口 | 可验证内容 |
|---|---|---|
| 推理 pipeline | `inference/cli_demo.py` | T2V/I2V/V2V pipeline、分辨率、50 steps、CFG 6、DPM、offload、输出 fps |
| 3D causal VAE | HF `vae/config.json`；diffusers `AutoencoderKLCogVideoX` | 16 channels、4×8×8、scaling 0.7、block channels、1.5 invert scale |
| T5 | HF `text_encoder/config.json` | T5-v1.1-XXL、24 层、4096 hidden、64 heads、10240 FFN |
| Patchify / Transformer | HF `transformer/config.json` | 42 层、48×64、patch 2、1.5 temporal patch 2、RoPE |
| T2V 训练目标 | `finetune/models/cogvideox_t2v/lora_trainer.py` | VAE sample、随机 timestep、add_noise、v-prediction、加权 loss、首帧复制 |
| I2V 条件 | `finetune/models/cogvideox_i2v/lora_trainer.py` | 图像加噪、VAE 编码、时间补零、通道拼接、offset embedding |
| SAT→diffusers | `tools/convert_weight_sat2hf.py` | 16/32 输入通道、1.0/1.5 patch 配置、权重名映射 |
| 微调资源 | `finetune/README.md` | LoRA/SFT 的显存测试点与数据格式 |

### 10.3 默认 CogVideoX-5B 全链路张量

以 $B=1$、49×480×720、226 文本 token 为例：

| 阶段 | 张量形状 | 说明 |
|---|---:|---|
| 像素视频 | `[1,3,49,480,720]` | 8 fps，约 6 秒 |
| VAE posterior/sample | `[1,16,13,60,90]` | 时间 4×、空间 8×压缩 |
| Transformer 输入顺序 | `[1,13,16,60,90]` | diffusers 将 frame 提到 channel 前 |
| 空间 patch | `[1,13,30,45,64]` | 每 patch `16×2×2=64` |
| 视觉 token | `[1,17550,3072]` | `13×30×45` |
| T5 token | `[1,226,4096]` | 冻结 T5-v1.1-XXL |
| 投影后文本 | `[1,226,3072]` | 与视觉 hidden 对齐 |
| 联合序列 | `[1,17776,3072]` | 42 层、48 heads、head dim 64 |
| 输出 patch | `[1,17550,64]` | 只取视觉部分 |
| unpatchify | `[1,13,16,60,90]` | 预测 v/去噪结果 |
| VAE decode | `[1,3,49,480,720]` | 导出 8 fps |

### 10.4 默认 CogVideoX1.5-5B 全链路张量

以 $B=1$、81×768×1360、226 文本 token 为例：

| 阶段 | 张量形状 | 说明 |
|---|---:|---|
| 目标像素视频 | `[1,3,81,768,1360]` | 16 fps，约 5 秒 |
| 名义 VAE latent | `[1,16,21,96,170]` | 输出端 81 帧对应 21 latent frames |
| 补齐后的去噪 latent | `[1,22,16,96,170]` | T2V 推理直接创建 22 帧噪声；训练编码后复制首帧 |
| 时空 patch | `[1,11,48,85,128]` | 每 patch `2×16×2×2=128` |
| 视觉 token | `[1,44880,3072]` | `11×48×85` |
| T5 token | `[1,226,4096]` | config 上限 226 |
| 联合序列 | `[1,45106,3072]` | 42× Expert Transformer |
| 输出 patch | `[1,44880,128]` | unpatchify 回 22 个 latent frames |
| decode 前裁剪 | `[1,21,16,96,170]` | pipeline 从时间开头裁掉额外 1 帧 |
| VAE decode | `[1,3,81,768,1360]` | 导出 16 fps |

### 10.5 逐模块四段拆解

#### 10.5.1 VAE

**输入输出**：`[B,3,F,H,W] → [B,16,F_z,H/8,W/8] → [B,3,F,H,W]`。  
**设计原因**：视频 DiT 的成本由 token 数主导，时间压缩与空间压缩同样重要；因果卷积允许首帧锚定与时间切分。  
**关键超参**：16 latent channels、temporal ratio 4、spatial ratio 8、`scaling_factor=0.7`、channels `[128,256,256,512]`。  
**代码与论文差异**：论文训练含 LPIPS/KL/GAN/context parallel；当前主分支主要消费预训练 VAE，部署优化是 slicing/tiling，未开放论文 VAE 完整训练配方。

#### 10.5.2 文本编码器

**输入输出**：token ids `[B,226] → [B,226,4096] → [B,226,3072]`。  
**设计原因**：T5 encoder 的双向上下文适合密集描述；4096 维语义再投影到视频 Transformer hidden。  
**关键超参**：T5-v1.1-XXL、24 layers、64 heads、FFN 10240、词表 32128；训练中冻结。  
**代码与论文差异**：论文只写 T5，型号由公开权重 config 才能确定；模型卡 1.5 写 224 token，config 为 226。

#### 10.5.3 Patch embedding

**输入输出**：5B 将 `[13,16,60,90]` 变为 17,550 个 3072 维 token；1.5 将补齐后的 `[22,16,96,170]` 以 2×2×2 patch 变为 44,880 token。  
**设计原因**：空间 patch 降低 token；1.5 再用 temporal patch 抵消高帧率、高分辨率造成的序列增长。  
**关键超参**：`patch_size=2`；1.5 `patch_size_t=2, patch_bias=false`；初代 `patch_size_t=None, patch_bias=true`。  
**代码与论文差异**：论文给一般 $p,q$ 公式但未在正文列出版本具体值；转换脚本与 HF config 补全了这些数值。

#### 10.5.4 Expert Transformer / AdaLN

**输入输出**：`[B,L,3072] → [B,L,3072]`，42 层；最终只对视觉段输出 patch。  
**设计原因**：共享 attention/FFN 让文本和视频深度融合，模态专属 AdaLN 解决统计差异，以较少参数替代双塔 MMDiT。  
**关键超参**：48 heads、head dim 64、GELU-approximate、dropout 0、norm eps $10^{-5}$、attention bias true。  
**代码与论文差异**：当前 main 通过外部 diffusers 类实现；论文图中的 expert 是 AdaLN expert，不是 token routing MoE。不要用“每 token 选择专家”解释。

#### 10.5.5 3D RoPE 与 full attention

**输入输出**：视觉 Q/K 的每个 head 为 64 维，并按时空坐标旋转；attention 覆盖 226 文本与全部视觉 token。  
**设计原因**：直接连通跨帧大位移；相对位置更适应多分辨率，外推保持局部间隔。  
**关键超参**：5B/1.5 RoPE；论文通道分配 x:y:t = 3:3:2；当前 fine-tune 用 `get_3d_rotary_pos_embed(..., grid_type="slice")`。  
**代码与论文差异**：2B 是 sinusoidal；初代 I2V 还叠加 learnable position。不能把 3D RoPE 统一套到所有版本。

#### 10.5.6 Diffusion、CFG 与 scheduler

**输入输出**：同形 latent 从 $z_T$ 迭代到 $z_0$，每步网络输出 16 通道预测。  
**设计原因**：v-prediction 配合 zero-SNR 改善高噪/低噪端训练；CFG 提升文本对齐。  
**关键超参**：1000 train timesteps、scaled-linear beta 0.00085→0.012、zero-SNR、50 inference steps、CLI CFG 6、DPM trailing。  
**代码与论文差异**：论文没有公开 sampler/CFG；DPM、dynamic CFG 和具体默认值来自当前 CLI。微调代码每进程独立 `torch.randint`，未实现论文 Explicit Uniform Sampling，因此不能据此称公开微调脚本复现了基座采样策略。

#### 10.5.7 I2V 条件

**输入输出**：`noisy video latent [B,F,16,H,W]` 与 `image latent [B,F,16,H,W]` 沿 channel 拼成 32 通道；输出 16 通道。  
**设计原因**：像素参考经同一 VAE 后与生成 latent 对齐；图像加噪减小“训练首帧 vs 推理照片”偏移。  
**关键超参**：$\log\sigma\sim\mathcal{N}(-3,0.5^2)$；1.5-I2V `ofs=2.0`、`ofs_embed_dim=512`。  
**代码与论文差异**：公开细节比论文更具体；但初代与 1.5-I2V 的 position/offset 不同，且 T2V 权重输入通道不兼容。

### 10.6 计算与复现建议

- **只做推理**：使用 diffusers pipeline，先固定模型 revision；显存足够时关闭 sequential offload 可明显提速。
- **结构研究**：应使用 SAT 权重/代码路径，不要在 diffusers wrapper 上误判论文原始 block；转换时检查 1.0/1.5、T2V/I2V 四个维度。
- **LoRA**：优先锁定 Q/K/V/out 与 FFN 层，不训练冻结 T5/VAE；1.5 长序列需要 gradient checkpointing、ZeRO 或更小 batch。
- **验证论文技巧**：Expert AdaLN、3D vs 2D+1D attention、Explicit Uniform Sampling 都需要基座级训练对照；公开微调脚本不足以公平复现实验。
- **避免形状错误**：初代 49 帧→13 latent frames；1.5 的 81 帧名义上→21，但训练时需在开头复制到 22，T2V 推理时则直接创建 22 帧噪声并在 decode 前从开头裁回 21。若直接用 $F/4$ 或忽略这一步时间补齐，会造成 RoPE grid 与 unpatchify 错位。

---

## 11. 个人启发与后续问题

1. CogVideoX 最值得复用的思想是“**共享计算主干、分离模态归一化**”：它比双塔省参数，又比完全共享更稳定，可推广到视频—音频、视频—动作等联合 DiT。
2. VAE 压缩与 patchify 必须一起算。1.5 没改 5B 的层数和宽度，却靠时间 patch 2 承担更高帧率；模型升级不一定等于参数增大，也可能是 token 预算重分配。
3. 数据 caption 是生成质量的一半。逐帧 caption + LLM 汇总能启动规模化监督，而 CogVLM2-Caption 将时序理解重新放回视觉模型，避免纯文本汇总丢失动作顺序。
4. Explicit Uniform Sampling 本质是对扩散时间步做分层采样，是低成本、可迁移到任何大规模 diffusion 训练的方差降低方法。
5. 未解问题包括：如何在保留大运动能力时把 full attention 从 $O(L^2)$ 降到近线性；如何公开可复现的 Frame Pack；如何评价 10 秒视频中的身份漂移与事件顺序；以及高质量筛选造成的语义退化能否通过分层重采样而非最终硬微调解决。

---

## 12. 材料来源

完整本地材料清单、图片原始 URL、模型 revision 和获取日期见：

- [assets/CogVideoX - 专家Transformer视频扩散模型/sources.md](assets/CogVideoX%20-%20专家Transformer视频扩散模型/sources.md)
- 官方论文：<https://arxiv.org/abs/2408.06072v3>
- 官方代码固定提交：<https://github.com/zai-org/CogVideo/tree/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2>
