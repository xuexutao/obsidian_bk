---
type: paper-note
status: done
domain: World-Models
paper: Cosmos World Foundation Model Platform for Physical AI
year: 2025
arxiv: 2501.03575
source: https://arxiv.org/abs/2501.03575
project: https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai
code: https://github.com/nvidia-cosmos/cosmos-predict1
tags:
  - World-Models
  - Physical-AI
  - Video-Generation
  - Diffusion
  - Autoregressive
  - Tokenizer
created: 2026-10-16
updated: 2026-10-16
---

# Cosmos WFM：物理 AI 世界基础模型平台

## 基本信息

- **正式名称**：Cosmos World Foundation Model Platform for Physical AI
- **机构**：NVIDIA
- **发布时间**：NVIDIA Research 页面为 2025-01-06；arXiv 2501.03575
- **核心资产**：视频策展流水线、连续/离散 Cosmos Tokenizer、扩散与自回归两族 WFM、后训练样例、Guardrail
- **开放边界**：代码采用 Apache 2.0；模型采用 NVIDIA Open Model License；当前 Predict1 仓库已进入有限维护，官方建议新项目迁移到 Cosmos 3
- **重要性评估**：★★★★★。它的里程碑意义不是某一项视频指标，而是第一次把“视频数据治理—tokenizer—两种生成范式—领域后训练—安全层”组织成可开放复用的 Physical AI 平台；但论文也坦率承认，它仍不是可靠的物理模拟器。

## 一句话结论

Cosmos 1.0 把世界模型从单一视频生成器升级成基础设施：从约 2000 万小时视频中构建训练语料，以因果连续/离散 tokenizer 分别支撑扩散 DiT 与自回归 GPT，再用相机位姿、机器人动作和多视角轨迹做后训练；它证明大规模视频预训练能显著改善 3D 一致性和下游可控生成，却也用物理基准暴露出对象永存、接触动力学、重力与指令一致性仍不可靠。

---

## 1. 研究背景与问题定义

### 1.1 Physical AI 为什么需要 WFM

论文把 Physical AI 拆成两种“数字孪生”：策略模型是智能体自身的数字孪生，世界模型是外部环境的数字孪生。若直接在真实机器人或车辆上采集试错，成本高、风险大、覆盖长尾事件困难；因此作者希望先预训练一个通用世界基础模型 $\mathcal{W}$，再用目标领域数据后训练成定制世界模型。

形式上，给定截至时刻 $t$ 的视觉历史 $x_{0:t}$ 与扰动/控制 $c_t$，模型预测下一世界状态：

$$\hat{x}_{t+1}=\mathcal{W}(x_{0:t},c_t)$$

本文把世界状态具体化为 RGB 视频，把 $c_t$ 扩展为文本、相机位姿、机器人动作、车辆轨迹等条件。这里的“世界模型”仍是**视觉观测生成模型**，而不是显式状态空间或严格物理求解器。

### 1.2 论文要解决的四个系统问题

1. 如何从互联网和专有数据中治理数千万小时视频，使训练数据尽量符合连续物理演化而非剪辑逻辑；
2. 如何把高分辨率长视频压缩成可训练 token，同时兼顾扩散所需的连续 latent 与 GPT 所需的离散 token；
3. 如何同时探索视觉质量更强的扩散路线和可复用 LLM 推理栈的自回归路线；
4. 如何证明通用预训练能迁移到相机控制、机器人操作和自动驾驶，而不是只生成“看起来像真的”视频。

![](assets/Cosmos%20WFM%20-%20物理AI世界基础模型平台/figures/01-overview.png)

> **图 1：Cosmos 1.0 平台能力总览。** 上半部分并列扩散 WFM 与自回归 WFM，下半部分展示相机控制、机器人操作和自动驾驶三类后训练。它也直接揭示 1.0 的本质：通用视频底座加多个领域专用接口，而非一个统一理解—生成—动作模型。  
> 来源：论文 Figure 1，p.2。

### 1.3 主要贡献

- 构建覆盖约 **2000 万小时**、720p–4K 视频的工业级策展系统，得到约 $10^8$ 个预训练 clip 和 $10^7$ 个高质量微调 clip；
- 提出统一图像/视频、时间因果的 Cosmos Tokenizer，同时输出连续 latent 与 FSQ 离散 token；
- 发布扩散 WFM（7B/14B）与自回归 WFM（4B/12B 基座、5B/13B 文本条件版）；
- 设计 3D 一致性与物理对齐评测，并给出相机、机器人、驾驶的后训练实例；
- 开源模型、代码、Tokenizer、ShotBench/TokenBench 与 Guardrail。

---

## 2. 任务定义与输入输出

### 2.1 两族 WFM 的 I/O

| 模型族 | 输入 | 输出 | 生成目标 | 主要 tokenizer |
|---|---|---|---|---|
| Diffusion Text2World | 文本、FPS | 完整视频 | 连续 latent 去噪 | CV $8\times8\times8$ |
| Diffusion Video2World | 文本、FPS、1 或多帧历史 | 未来视频 | 条件连续 latent 去噪 | CV $8\times8\times8$ |
| AR base | 第一帧/历史离散 token | 后续视频 token | next-token prediction | DV $8\times16\times16$ |
| AR Video2World | 历史离散 token、文本 | 后续视频 token | 文本条件 next-token prediction | DV $8\times16\times16$ |
| 后训练模型 | 加入相机、动作、轨迹或多视角条件 | 可控未来视频 | 延续所属基座目标 | 按基座选择 |

Diffusion 路线直接输出高质量连续 latent；AR 路线以更激进的离散压缩换取 GPT 式序列建模与 KV cache/推测解码能力，再以 diffusion decoder 修复模糊和伪影。

### 2.2 Cosmos Tokenizer 的输入输出

输入视频为：

$$x_{0:T}\in\mathbb{R}^{(1+T)\times H\times W\times3}$$

编码结果为：

$$z_{0:T'}\in\mathbb{R}^{(1+T')\times H'\times W'\times C}$$

重建过程：

$$\hat{x}_{0:T}=\mathcal{D}(\mathcal{E}(x_{0:T}))$$

第一个时间 token 对应第一帧，因此 $T=0$ 时同一网络自然退化为图像 tokenizer；$T>0$ 时处理视频。这一“首帧单独、后续因果分组”的定义后来仍是 Cosmos 系列把图像视为单帧视频的基础。

---

## 3. 核心方法

### 3.1 数据策展：先让训练数据遵守“镜头内物理连续”

流水线依次执行 splitting、filtering、annotation、deduplication、sharding：

- **镜头切分**：TransNetV2/AutoShot 检测 shot boundary；少于 2 秒的片段丢弃，超过 60 秒继续切分；
- **统一转码**：`h264_nvenc` + NVDEC/NVENC；PyNvVideoCodec 方案吞吐 0.3702 videos/s，相比 H100 CPU `libx264` 的 0.0574 videos/s 约快 6.5 倍；
- **过滤**：运动、DOVER 感知质量、审美分、文字覆盖、动画/游戏/抽象视觉类型；最低 15% 感知质量被剔除；
- **标注**：内部 VILA-13B 对 8 帧采样生成平均 97 词 caption；TRT-LLM FP8 batch 16 达 1.96 clips/s，约为 PyTorch FP16 的 9.3 倍；
- **去重**：InternVideo2 embedding、10,000 个聚类、簇内语义去重，约移除 30% 训练数据；
- **分片**：按分辨率、长宽比和视频长度生成 WebDataset，由 Ray 流水线在异构集群上调度。

最终原始数据约 2000 万小时，预训练片段约 1 亿，微调片段约 1000 万，单片 2–60 秒。全平台训练使用约 **10,000 张 H100、持续三个月**。

### 3.2 Cosmos Tokenizer：连续与离散双轨

![](assets/Cosmos%20WFM%20-%20物理AI世界基础模型平台/figures/02-tokenizer-architecture.png)

> **图 2：因果 Cosmos Tokenizer。** 左侧用 3D Haar wavelet 把连续帧分组并保持仅依赖过去；右侧 encoder/decoder 由因果残差块、因果下/上采样和因果时空注意力组成。完整图与图注已保留。  
> 来源：论文 Figure 9，p.14。

Tokenizer 先做两级 wavelet transform，在 $x,y,t$ 方向各下采样，再进入因果 encoder-decoder。3D 卷积分解为 $1\times k\times k$ 空间卷积和 $k\times1\times1$ 时间卷积；时间侧左填充 $k-1$，防止未来信息泄漏。网络使用 Swish 与 LayerNorm，并在最后层加入可覆盖全历史的因果时空注意力。

两种 latent 形式：

- **连续 tokenizer**：普通 AE，latent 维度 16，无 VAE KL；服务 diffusion WFM；
- **离散 tokenizer**：FSQ，latent 维度 6，levels 为 $(8,8,8,5,5,5)$，词表大小 64,000，无 VQ commitment loss；服务 AR WFM。

训练分两阶段。第一阶段为 L1 与 VGG 感知损失：

$$\mathcal{L}_1=\|\hat{x}_{0:T}-x_{0:T}\|_1$$

$$\mathcal{L}_{\mathrm{Perceptual}}=\frac{1}{L}\sum_{l=1}^{L}\sum_t\alpha_l\|\mathrm{VGG}_l(\hat{x}_t)-\mathrm{VGG}_l(x_t)\|_1$$

第二阶段加入双向光流、Gram matrix 与对抗损失，提高时序平滑和锐度。Tokenizer 支持图像 $8\times8/16\times16$、视频 $4\times8\times8$、$8\times8\times8$、$8\times16\times16$ 等压缩率。

### 3.3 Diffusion WFM：连续 latent 上的 EDM/DiT

![](assets/Cosmos%20WFM%20-%20物理AI世界基础模型平台/figures/03-diffusion-wfm-architecture.png)

> **图 3：Cosmos-Predict1 Diffusion WFM。** 视频先经 CV $8\times8\times8$ tokenizer 压缩，加高斯噪声后做 $1\times2\times2$ 3D patchify；DiT 重复 self-attention、T5 cross-attention 与 MLP，时间步通过 AdaLN 的 scale/shift/gate 注入。  
> 来源：论文 Figure 11，p.20。

扩散模型在 continuous latent 上使用 EDM 风格 denoising score matching：

$$\mathcal{L}(D_\theta,\sigma)=\mathbb{E}_{x_0,n}\left[\|D_\theta(x_0+n;\sigma)-x_0\|_2^2\right]$$

总损失带噪声级不确定性自适应：

$$\mathcal{L}(D_\theta)=\mathbb{E}_\sigma\left[\frac{\lambda(\sigma)}{e^{u(\sigma)}}\mathcal{L}(D_\theta,\sigma)+u(\sigma)\right]$$

$$\lambda(\sigma)=\frac{\sigma^2+\sigma_{\mathrm{data}}^2}{(\sigma\sigma_{\mathrm{data}})^2}$$

关键结构包括：

- 3D patchify：$(p_t,p_h,p_w)=(1,2,2)$；
- T5-XXL 文本 embedding，通过 cross-attention 注入，长度 pad 到 512；
- FPS-aware 3D factorized RoPE + 可学习绝对位置 embedding；
- self/cross attention 均做 QK RMSNorm，抑制 attention entropy collapse；
- AdaLN-LoRA 将 7B 模型从原本约 11B 参数压到 7B，论文称指标保持相当。

Video2World 把条件帧和待生成帧沿时间拼接，并在通道维加入条件 mask。条件帧加轻微噪声，损失仅作用于未来帧；训练时随机改变条件帧数，因此推理可接受单图或多帧视频。

### 3.4 Autoregressive WFM：离散视频 next-token prediction

AR 模型是 Llama3 风格 GPT，从零训练。4B/12B 基座只做视频 token 续写，5B/13B 版本新增 T5 cross-attention 以接受文本。训练课程为：

1. 第一帧条件，17 帧上下文，预测 16 帧；
2. 用 YaRN 将上下文扩为 34 帧；
3. 加文本 cross-attention，联合图像/视频训练；
4. 用高质量图文/视频数据做 30,000 次 cooldown，学习率衰减到 0。

所有 AR 模型固定在 $640\times1024$ 训练；离散序列长度 12,800，词表 64,000。Medusa 通过多头并行猜测后续 token，再用拒绝采样校验。9 个 Medusa heads 是吞吐与前向次数的最佳折中：4B 吞吐 444.95→894.67 tokens/s；5B 为 303.61→982.77 tokens/s。低分辨率适配后，4B 在 8×H100 上达 10.08 FPS。

AR 高压缩输出较模糊，因此作者训练一个 diffusion decoder：以 DV $8\times16\times16$ token 为条件，生成 CV $8\times8\times8$ latent，再由连续 tokenizer 解码 RGB。它提高锐度，但增加额外扩散推理成本。

### 3.5 后训练：条件接口仍是“外挂式”

- **相机控制**：用 Plücker ray $r=(d,c\times d)\in\mathbb{R}^6$ 与 latent 拼接，在 DL3DV-10K 后训练；
- **机器人指令**：T5 指令通过 cross-attention 注入；
- **机器人动作**：7 维 gripper 坐标动作经 MLP 投影；扩散版加到 timestep embedding，AR 版走 cross-attention；
- **驾驶**：六视角共同 self-attention；每个视角有独立 view embedding 与 view-specific text cross-attention；轨迹为 64 个 3D 点、0.1 秒间隔。

这一阶段的关键边界是：1.0 还没有统一的动作 tokenizer，也没有统一理解塔。新增条件通常需要新 embedder、新 cross-attention 或 timestep 注入，并分别后训练专用模型。

---

## 4. 数据与训练设置

### 4.1 数据构成

视频类别分布覆盖自然动力学 20%、手部/物体操作 16%、空间导航 16%、驾驶 11%、人体运动 10%、第一人称 8%、动态相机 8%、合成渲染 4% 等。后训练数据包括：

- 相机控制：DL3DV-10K；
- 机器人指令：约 200 小时 Cosmos-1X，约 12,000 个 1–9 秒 episode；
- 动作预测：Bridge 约 20,000 episodes，$320\times256$、5 FPS、7 维动作；
- 驾驶：内部 RDS 约 360 万个 20 秒六视角 clip，约 20,000 小时。

### 4.2 扩散训练课程

| 阶段 | 分辨率 | 帧数 | 序列长度 | Context Parallel |
|---|---:|---:|---:|---:|
| 低分辨率预训练 | $640\times512$ | 57 | 10,240 | 2 |
| 高分辨率预训练 | $1280\times704$ | 121 | 56,320 | 8 |
| 高质量微调 | $1280\times704$ | 121 | 56,320 | 8 |

使用 FSDP、context parallelism 和选择性 activation checkpoint。14B 模型参数/梯度/优化器约占 280GB，高分辨率 activation 约 310GB，说明即使只复现单个大模型也已超出普通实验室资源。

---

## 5. 实验结果

### 5.1 Tokenizer baseline

代表性结果：

| 设置 | Cosmos | 主要 baseline | 结果 |
|---|---|---|---|
| CV $4\times8\times8$，DAVIS | Tokenize1 360p | CogVideoX VAE | PSNR 35.85 vs 29.29；rFVD 10.06 vs 19.58 |
| DV $4\times8\times8$，TokenBench | Tokenize1 360p | VideoGPT | PSNR 35.74 vs 33.66；rFVD 19.25 vs 13.85 |
| CV 720p 逐帧耗时 | Cosmos-0.1 | CogVideoX | 34.8ms vs 414ms |
| CI $8\times8$ 1024² | Cosmos-0.1 | FLUX VAE | 62.7ms vs 242ms |

连续 tokenizer 的质量/速度优势明显；离散 tokenizer 的 PSNR 较好，但 rFVD 并非全面领先，说明“更锐”不等于视频分布更接近真实。

### 5.2 基座的 3D 一致性

在 RealEstate10K 500 个静态场景上：

| 模型 | Sampson ↓ | 位姿估计成功率 ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| VideoLDM | 0.841 | 4.4% | 26.23 | 0.783 | 0.135 |
| Cosmos Diffusion 7B Text2World | **0.355** | 62.6% | 33.02 | 0.939 | 0.070 |
| Cosmos Diffusion 7B Video2World | 0.473 | **68.4%** | 30.66 | 0.929 | 0.085 |
| Cosmos AR 4B | 0.433 | 35.6% | 32.56 | 0.933 | 0.090 |
| 真实视频 | 0.431 | 56.4% | 35.38 | 0.962 | 0.054 |

Cosmos 显著优于 VideoLDM，但某些指标超过真实视频并不意味着“比真实更物理”：静态场景、筛选成功样本和几何工具偏好都可能影响数值。

### 5.3 物理对齐：更多上下文比更大模型更重要

在 8 类 PhysX/Isaac Sim 场景、800 个 1080p 视频上，Diffusion 7B Video2World 从 1 帧条件增加到 9 帧后，PSNR 17.34→21.06、SSIM 0.538→0.691、IoU 0.332→0.592。14B 并未稳定优于 7B；论文明确总结：所有模型仍会出现物体突然出现/消失、形变、违反重力和不合理运动学。

### 5.4 后训练 baseline

| 任务 | Cosmos | Baseline | 关键结果 |
|---|---|---|---|
| 相机控制 | Diffusion 7B CameraCond | CamCo | 位姿成功率 82.0% vs 43.0%；FVD 120.49 vs 433.24 |
| 机器人指令视频 | Diffusion 7B | VideoLDM-Instruction | 总体人评偏好 78.3% vs 13.0% |
| Bridge 动作预测 | Diffusion 7B | IRASim-Action | PSNR 21.14 vs 19.13；FVD 190 vs 593 |
| 驾驶六视角 | Diffusion 7B MultiView | VideoLDM-MultiView | FVD 210.23 vs 884.46；CSE 2.11 vs 6.48 |

![](assets/Cosmos%20WFM%20-%20物理AI世界基础模型平台/figures/04-robot-human-evaluation.png)

> **图 4：Cosmos-1X 机器人指令视频预测人评。** 7B 扩散模型在 instruction following、object permanence、verity 和 overall 四项均大幅胜过同数据微调的 VideoLDM；5B AR 版本仍有优势，但明显弱于 7B 扩散版。  
> 来源：论文 Figure 24，p.46。

### 5.5 消融与反直觉结论

- **Medusa heads**：更多 head 减少前向次数，但 12 头吞吐略低于 9 头，说明推测更长不等于 wall-clock 更快；
- **模型大小与物理性**：大模型视觉更锐，但 14B 在物理对齐上没有稳定优于 7B；物理规律不是简单参数 scaling 自动涌现；
- **上下文长度**：9 帧条件显著优于单帧，尤其扩散模型；真实速度/加速度需要多帧才能辨识；
- **AR vs Diffusion**：扩散模型在 3D 一致性和机器人视频质量上更强；AR 的优势主要是实时推理栈与可扩展序列接口，而不是当时的最终视觉质量；
- **AR failure rate**：单图条件下 4B/5B/12B/13B 的失败率分别为 15%/7%/2%/3%，9 帧条件下降至 1%/2%/1%/0%。

---

## 6. 从 1.0 到 2.5 与 3：哪些设计被保留，哪些被推翻

| 维度 | Cosmos 1.0 | 后续变化 |
|---|---|---|
| Tokenizer | 自研 continuous AE + discrete FSQ 双轨 | 2.5 改用 WAN2.1 因果 VAE；3 使用 Wan2.2 视频 VAE，连续生成 token 成为主线 |
| 生成目标 | 扩散 EDM 与离散 AR 两套独立 WFM | 2.5 收敛到 flow matching 单一视频 DiT；3 保留语言 AR，但视觉/音频/动作统一用 rectified flow |
| 条件方式 | T5 cross-attention；相机/动作/轨迹各自外挂 | 2.5 用条件帧+mask+frame replacement 统一 T2W/I2W/V2W；3 用统一 token 序列表达所有任务 |
| 理解与生成 | 无独立理解模型，T5 只提供文本条件 | 2.5 用 Cosmos-Reason1 作文本 encoder；3 将 Reasoner 与 Generator 变成同一 MoT 的两塔 |
| 动作 | 后训练时新增 7 维 action embedder | 3 设计跨具身统一动作表示，并原生支持 FD/ID/Policy |

我的判断：1.0 的真正价值是**定义了平台层问题并把数据与 tokenizer 做实**；它的双范式和大量专用后训练头更像一次全面探索。2.5 选择了视频 flow-matching 主线，3 才把“AR 理解 + diffusion 生成 + action”重新统一。

---

## 7. 局限与批判性评价

### 7.1 论文明确承认的局限

- 缺少对象永存，物体可能凭空出现/消失或变形；
- 接触丰富动力学、重力、光照交互、流体动力学不可靠；
- 指令遵循存在不一致；
- 人评物理真实性主观，且未必与下游策略指标相关；
- 3D/物理评测只覆盖有限设置，论文明确把更全面评估留给 future work；
- 相机、机器人、驾驶模型只是 sample application，不是生产系统。

### 7.2 批判性判断

1. **“世界模型”与“视频生成器”仍高度重合。** 它没有给出长期闭环规划成功率，也未证明生成误差对控制决策的界；
2. **物理 scaling 未成立。** 参数变大改善清晰度，却不稳定改善物理对齐；这说明数据中“看起来合理”的统计规律不足以替代干预数据与物理约束；
3. **后训练条件接口碎片化。** 相机、动作、轨迹、多视角各有不同模块，复用的是视觉主干，不是统一控制语义；
4. **评测存在代理指标风险。** Sampson error、PSNR、FVD 只能覆盖几何或视觉相似度，无法证明可安全用于策略评估；
5. **工业级开放仍不等于可复现。** 权重和代码可用，但原始训练语料、10k H100 预算与内部数据不可重建。

### 7.3 Future work

作者明确提出：自动化多模态 evaluator、用物理模拟器做可复现交互评测、改进物理数据过滤与模型设计、探索 AR/扩散混合结构，以及验证 WFM 在策略评估、初始化、RL、MPC 和 Sim2Real 中的真实收益。

---

## 8. 复现成本与实践建议

### 8.1 成本分层

- **完整预训练**：不可由普通实验室复现；论文披露约 10,000 H100、三个月，且依赖 2000 万小时视频与内部数据治理；
- **Tokenizer 复现**：比 WFM 可行，但高分辨率长视频、光流/感知/对抗损失仍需要多卡；
- **后训练**：最现实。7B Video2World 在定制相机/机器人数据上微调仍需多节点，但 LoRA/低分辨率版本可显著降低门槛；
- **推理**：AR 4B 单卡约需 29GB VRAM，5B 约 59GB；扩散 7B/14B 的长视频推理通常需要多卡或 offload。

### 8.2 建议的最小复现路线

1. 直接使用开放的 continuous tokenizer 与 7B Video2World checkpoint；
2. 先在单一短时域复现 Video2World，而不是从 Text2World 起步；
3. 对动作任务保留条件帧与动作时间对齐，优先复现 Bridge 的 7 维 action-conditioned 设置；
4. 同时报告视觉指标、对象永存、动作遵循和闭环任务指标，避免只用 FVD；
5. 若追求新项目而非历史复现，优先评估 Cosmos 3，Predict1 官方仓库已转为有限维护。

---

## 9. 个人启发与后续问题

- 世界模型的第一道门槛不是更大的 Transformer，而是**能否把剪辑视频变成近似物理连续的数据**；
- continuous/discrete 双 tokenizer 是 1.0 最具时代特征的设计，随后两代逐渐放弃“视觉离散 AR 为主线”，转向连续流模型；
- 多帧条件显著提高物理预测，说明动作和动力学评测必须控制可观测性，否则单帧任务把不可辨识性误算成模型错误；
- 真正未解的问题仍是：视频预测质量何时能可靠地转化为策略排序、规划收益和安全边界？1.0 只搭好了平台，没有完成这条因果链。

---

## 10. 材料来源

完整材料清单见：`assets/Cosmos WFM - 物理AI世界基础模型平台/sources.md`。
