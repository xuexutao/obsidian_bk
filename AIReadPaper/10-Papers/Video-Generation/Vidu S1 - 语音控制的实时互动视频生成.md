---
type: paper-note
status: done
domain: Video-Generation
paper: Vidu S1
year: 2026
arxiv: 2607.03118
source: https://arxiv.org/abs/2607.03118
project: https://vidu.com/vidu-stream
code: null
tags: [Video-Generation, RealTime, StreamingDiffusion, Avatar, Distillation]
---

# Vidu S1：语音控制的实时互动视频生成

## 基本信息

- **论文正式名称**：Vidu S1: A Real-Time Interactive Video Generation Model
- **作者与机构**：Jintao Zhang、Kai Jiang、Jintao Chen 等 24 人（清华大学 × 生数科技 Shengshu Technology；顾问 Jun Zhu、Jianfei Chen、Fan Bao、Zhijie Deng）
- **arXiv**：2607.03118（v1 2026-07-03，v2 2026-07-21）
- **项目页 / Demo**：https://vidu.com/vidu-stream
- **代码**：未开源（技术报告 + 在线 Demo + API）
- **关键规格**：540p @ 42 FPS（3 步蒸馏），RTX 5090 消费级 GPU，无限时长流式生成
- **前置技术**：TurboDiffusion（推理加速）、TurboServe（流式服务）
- **关联笔记**：[[Diffusion Forcing - 下一Token预测与全序列扩散]]、[[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]]、[[Causal Forcing - 正确的自回归扩散蒸馏]]、[[OmniForcing - 实时音视频联合流式生成]]、[[LongLive-2.0 - NVFP4长视频生成训练与推理系统]]

## 一句话结论

Vidu S1 把「实时互动视频生成」当成**模型 + 数据 + 推理系统的联合设计问题**：语音作为显式的未来内容控制信号，训练走「双向 video-audio teacher → 因果 teacher（Teacher Forcing + Diffusion Forcing 混合）→ DMD + PCM 少步蒸馏」三阶段，推理用「滑动窗口 + sink 参考上下文 + RoPE Repositioning + TwinCache」实现与序列长度无关的恒定每步延迟，再叠 SageAttention/per-block W8A8/CUDA Graph/Ulysses 并行的系统栈，最终在消费级 GPU 上做到 540p/42 FPS 的语音可控无限时长数字人，并在 HDTF 三项指标与自建 Vidu-StreamBench 人类偏好上全面领先商业系统。

---

## 1. 研究背景与问题定义

### 1.1 为什么实时互动生成是一个独立问题

论文开篇用一个非常直白的需求模型区分两类生成：设单用户对实时互动内容的平均需求为 $\alpha\in[0,1]$、对离线内容为 $\beta\in[0,1]$，用户数为 $N$，离线视频平均被观看 $m$ 次，则实时需求总量按 $\alpha\times N$ 增长，而离线需求按 $\beta\times N/m$ 被观看次数稀释。只要 $\alpha\approx\beta$ 且 $m>100$，实时互动生成的实际需求就远大于离线生成。这个论证虽是简化模型，但定调了全文立场：**直播、游戏、陪伴、数字人对话这类「内容必须立即响应用户」的场景，不是离线生成范式能覆盖的**。Sora/Veo/Wan/Seedance 这类离线一次性生成的模型，无论质量多高，用户在等待期间都无法干预。

### 1.2 现有自回归方法的四个缺口

论文把限制归纳为四条，也正好对应它的四项贡献：

1. **不支持真交互**：CausVid、Self-Forcing 等方法把离线生成改成自回归，但生成过程中用户无法插入指令——只是「更快」，不是「可交互」。
2. **语音不是显式控制信号**：多数方法只做音频驱动的口型同步，用户无法用自然口语指令直接改变未来内容（如「举起左手」）。
3. **长时生成漂移崩溃**：流式自回归中，前段误差作为条件喂给后段，小误差滚雪球成模糊、漂移乃至视觉崩坏，多数模型只能维持有限时长。
4. **推理基础设施缺位**：实时不仅是建模问题，还需要高效注意力内核、量化、调度与服务系统；没有这些，模型再好也无法上线。

### 1.3 能力总览

![[assets/Vidu S1 - 语音控制的实时互动视频生成/figures/fig1_overview.png]]
*Figure 1（p.1）：Vidu S1 三大能力。① 语音引导的未来视频生成——90 秒流中随时用语音切动作（挥手→比心→竖拇指→抬腿→坐下）；② 无限时长高效推理与服务——540p/42FPS、低成本 GPU、TurboDiffusion + TurboServe；③ 多样自定义角色——真人、宠物、动漫均可作为参考形象。来源：官方 PDF Figure 1。*

---

## 2. 数据准备：六道工序把杂乱视频变成可训练的互动片段

### 2.1 收集与预筛

训练语料分两类来源，各有分工：**直播/说话头视频**负责教表情、肢体动作和唇同步等细粒度特征；**影视高质量素材**负责跨镜头角度、场景与风格的泛化一致性。全部数据先去重，再按帧率、分辨率、音视频完整性和音视频同步性做预筛。

### 2.2 切分与联合过滤

视频沿镜头边界切成 3–60 秒的单镜头片段，切点被约束不得落在语音中间。过滤环节最有意思的观察是：**只用专家模型有明显局限**——人脸检测器泛化不到夸张的 2D 动画角色，基于单帧的专家模型又缺乏全局信息。因此引入 Omni 模型（Qwen3-Omni、Gemini）对完整视频做全局语义理解，在剪辑、主体、动作、情绪、面部、语音、场景、镜头、色调等维度打标签，与专家模型组成「全局 + 局部」联合过滤系统。具体过滤维度包括：主体唯一且占比合理、画面清洁（滤水印/字幕/广告）、视觉质量（美学与技术评分）、内容安全、镜头稳定性（保留静态或慢速运动镜头以降低漂移风险）、交互性（主体须有明确动作）。

![[assets/Vidu S1 - 语音控制的实时互动视频生成/figures/fig2_data_pipeline.png]]
*Figure 2（p.3）：数据过滤流水线。Raw Videos → Prefiltering（低帧率/分辨率、音视频不完整、音画失步）→ 单镜头 Clipping → Subject Filtering → Other Filtering（清晰度/内容安全/镜头稳定性/交互性）→ Diarization（Onscreen/Overlap/Trainable 三类）→ Caption + Embedding。来源：官方 PDF Figure 2。*

### 2.3 语音分离（Diarization）

高保真唇同步要求训练时「看到的表演」与「听到的语音」严格对应。流水线用 VAD + 活跃说话人检测（ASD）标注每段语音的时间戳和说话人，再按说话人与画面主体的关系分三类：**onscreen**（说话人就是画面里的人）、**offscreen**（不匹配）、**overlap**（多人重叠，直接丢弃）。一个实战细节：分离模型在唱歌或强背景音乐场景下不稳定，常把人声错分到音乐轨或引入合成音色伪影，于是加了启发式规则——说话人正在发声但语音能量占比过低的片段直接丢弃。

### 2.4 双粒度标注

Caption 分两层：**全片段级** caption 提供全局语义锚点；**语音感知的块级** caption 与时间区间对齐，为流式生成提供细粒度条件。为减少跨模态幻觉，采用**双路径解耦**——视觉属性只从视频帧推断、声学属性只从音轨推断，两路互不污染。

---

## 3. 训练：三阶段从双向先验到少步自回归

### 3.1 记号

设第 $i$ 帧的干净视频、音频表示为 $\boldsymbol{v}_0^i,\boldsymbol{a}_0^i$，定义视频-音频联合状态（沿模态维拼接）：

$$\boldsymbol{x}_0^i=[\boldsymbol{v}_0^i;\boldsymbol{a}_0^i]$$

统一条件接口 $\boldsymbol{c}$ 包含语音、文本提示和参考图像，条件序列为 $\boldsymbol{c}^{1:\infty}$；$t_j$ 为扩散时间步（$t_0=0$ 为干净状态），$\hat{\cdot}$ 表示模型预测量，$f_\theta^{\mathrm{bi}}$ 与 $f_\theta$ 分别为双向与因果去噪模型。**video 和 audio 作为一个联合状态同时生成**，这是与只生成视频、音频另接 TTS 的方案的根本区别。

### 3.2 Stage 1：双向 Teacher

在全序列上训练双向 video-audio teacher，建立高质量生成先验：

$$\hat{\boldsymbol{x}}_0^{1:N}=f_\theta^{\mathrm{bi}}(\boldsymbol{x}_{t_j}^{1:N},t_j,\boldsymbol{c}^{1:N})$$

$$\mathcal{L}_{\mathrm{bi}}=\mathbb{E}_{(\boldsymbol{x}_0^{1:N},\boldsymbol{c}^{1:N}),t_j}\left[\left\|f_\theta^{\mathrm{bi}}(\boldsymbol{x}_{t_j}^{1:N},t_j,\boldsymbol{c}^{1:N})-\boldsymbol{x}_0^{1:N}\right\|_2^2\right]$$

### 3.3 Stage 2：因果 Teacher（Teacher Forcing + Diffusion Forcing 混合）

从双向 teacher 初始化，对 video-audio token 施加因果注意力掩码，第 $i$ 帧只能访问因果条件上下文和有效历史前缀：

$$\hat{\boldsymbol{x}}_0^i=f_\theta(\boldsymbol{x}_{t_j}^i,t_j,\boldsymbol{c}^{\le i},\boldsymbol{x}_{\tau_j}^{<i},\tau_j)$$

缩小训练-推理差距的关键是对每个训练样本从 $\mathrm{Bernoulli}(p)$ 抽样选择两种模式：

- **Teacher Forcing**：历史前缀是干净真值 $\boldsymbol{x}_0^{<i}$（$\tau_j=0$），提供稳定的条件同步与运动一致性监督；
- **Diffusion Forcing**：历史前缀是带噪状态 $\boldsymbol{x}_{\tau_j}^{<i}$（$\tau_j>0$），让模型学会在「不完美前缀」下继续生成——这正是推理时自身 rollout 的真实分布。

$$\mathcal{L}_{\mathrm{causal}}=\mathbb{E}_{(\boldsymbol{x}_0^{1:N},\boldsymbol{c}^{1:N}),i,t_j,\tau_j}\left[\left\|f_\theta(\boldsymbol{x}_{t_j}^i,t_j,\boldsymbol{c}^{\le i},\boldsymbol{x}_{\tau_j}^{<i},\tau_j)-\boldsymbol{x}_0^i\right\|_2^2\right]$$

这一混合策略与 [[Diffusion Forcing - 下一Token预测与全序列扩散]] 的思想一脉相承，也与 [[Causal Forcing - 正确的自回归扩散蒸馏]] 的「正确混合训练」立场一致：干净历史和带噪历史各有监督价值，缺了后者推理时必然分布失配。

### 3.4 Stage 3：DMD 蒸馏 + PCM 正则

多步自回归生成太慢，用 DMD（Distribution Matching Distillation）把自回归生成压缩到几步。DMD 通过匹配生成分布与数据分布的 score 差异来优化生成器：

$$\nabla_\theta\mathcal{L}_{\mathrm{DMD}}=\mathbb{E}_{z,t}\left[w(t)\left(\nabla_{\boldsymbol{x}_t^{1:N}}\log p_{\mathrm{data}}^t(\boldsymbol{x}_t^{1:N})-\nabla_{\boldsymbol{x}_t^{1:N}}\log p_\theta^t(\boldsymbol{x}_t^{1:N})\right)^{\!\top}\frac{d\boldsymbol{x}_t^{1:N}}{d\theta}\right]$$

论文的一个重要实战观察：**只优化 DMD 常出现模式崩溃**——相机漂移、内容退化、时间不一致。因此叠加 PCM（Phased Consistency Models）一致性目标做正则，距离函数用感知特征距离，其中 $f_{\theta^-}$ 为 EMA 学生网络：

$$\mathcal{L}_{\mathrm{PCM}}=\mathbb{E}\left[\lambda(t_n)\,d_{\mathrm{PERC}}\!\left(f_\theta(\boldsymbol{x}_{t_{n+1}}^{1:N},t_{n+1}),f_{\theta^-}(\tilde{\boldsymbol{x}}_{t_n}^{1:N},t_n)\right)\right]$$

训练时以 DMD 与 PCM 的加权组合联合优化。这与 [[CausVid - 从慢双向到快自回归视频扩散蒸馏]] 验证过的「DMD 需要正则防崩」的经验相互印证。

---

## 4. 推理：滑动窗口、双缓存与系统栈

### 4.1 流式推理

第 $i$ 步自回归解码时，注意力被限制在固定长度滑窗内，窗口由三部分组成：

1. **持久参考上下文**：用户首帧 latent + 首个生成状态，只构建一次、全程固定——角色相当于 LLM 的 attention sink（[[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]] 一系的 sink frame 设计）；
2. **滑窗内缓存的历史状态**；
3. **当前正在去噪的状态**。

固定窗口满足了因果自回归约束，同时让**每步推理延迟与已生成序列长度无关**——这是「无限时长」在工程上成立的前提。

- **RoPE Repositioning**：历史 key-value 在施加 RoPE 之前缓存；滑窗推进时按更新后的相对位置再施加。避免重算历史特征，同时保证所有可见位置都落在训练时的位置范围内，跨窗口位置一致。
- **TwinCache**：为每个历史状态维护**带噪缓存**与**干净缓存**两份互补表示。中间去噪步骤用带噪缓存 $\hat{\boldsymbol{x}}_{\tau_j}^{<i}$——残余噪声保留粗粒度时间动态、抑制高频伪影跨段累积，相当于时间传播的隐式低通先验；最后一步去噪换用干净缓存 $\hat{\boldsymbol{x}}_0^{<i}$，恢复细节并维持身份一致性。本质是把「时间传播」与「外观精修」解耦成两个阶段的缓存调度。

### 4.2 推理基础设施（TurboDiffusion + TurboServe 路线）

- **注意力加速**：SageAttention 系列 + SpargeAttention + Sparse-Linear Attention（SLA），多数是同一团队（Jintao Zhang 等）的工作，形成自洽的加速组合。
- **线性层量化**：实测 per-tensor 与 per-channel 量化都容易被离群值撑大量化区间导致画质劣化，因此自研 **per-block W8A8** CUDA GEMM 算子，细粒度 scaling 限制离群值影响。
- **Kernel Fusion**：把 RMSNorm 与逐元素操作等细粒度算子融合为 Triton/CUDA 内核，减少 launch 开销与中间张量的全局内存读写。
- **CUDA Graph**：流式生成的执行模式高度重复、单算子耗时短，主机端 launch 开销成为瓶颈；把固定结构子图捕获成 CUDA Graph 重放，多次 kernel launch 合并为单次图启动。
- **多 GPU 并行**：Ulysses 式上下文并行，attention 前后用 all-to-all 在序列并行与头并行布局间重分区，并优化通信实现。

---

## 5. 实验与结果

### 5.1 评测设置

- **Vidu-StreamBench（自建）**：500 个样本，每个含动作指令 + 参考首帧 + 音频片段，覆盖多样动作命令、参考图风格、说话人属性、情绪与场景；与 HeyGen、LemonSlice、Kling Avatar 2.0 做 pairwise A/B 偏好测试。
- **HDTF（公开）**：对比商业系统与开源模型（Wan2.2-S2V-14B、LiveAvatar、OmniAvatar-1.3B、Hallo3、StableAvatar-1.3B），指标为 CSIM（身份保持 ↑）、Sync-D（音画同步 ↓）、DOVER（感知质量 ↑）。
- **效率**：RTX 5090 上实测。

### 5.2 HDTF 定量结果

| 模型 | 指令跟随 | 实时 | 分辨率 | FPS | CSIM ↑ | Sync-D ↓ | DOVER ↑ |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| OmniAvatar | ✗ | ✗ | 480p | – | 0.8062 | 9.242 | 0.5476 |
| StableAvatar-1.3B | ✗ | ✗ | 480p | – | 0.8358 | 11.18 | 0.5560 |
| Hallo3 | ✗ | ✗ | 480×720 | – | 0.7698 | 8.660 | 0.5313 |
| Wan2.2-S2V-14B | ✗ | ✗ | 480p/720p | – | 0.7936 | 8.255 | 0.5510 |
| LiveAvatar | ✗ | ✓ | – | – | 0.8127 | 8.447 | 0.5639 |
| LemonSlice | ✗ | ✓ | 368×560 | – | 0.8407 | 7.921 | 0.5196 |
| HeyGen | ✗ | ✓ | – | 25 | 0.9191 | 8.037 | 0.4864 |
| Kling Avatar 2.0 | ✓ | ✗ | – | – | 0.8688 | 8.158 | 0.5406 |
| **Vidu S1** | **✓** | **✓** | **540p** | **42** | **0.9192** | **7.8470** | **0.5660** |

核心信息：Vidu S1 是表中**唯一同时支持指令跟随与实时生成**的方法，且三项指标全部最优。HeyGen 的 CSIM 几乎打平（0.9191 vs 0.9192）但 DOVER 明显更差（0.4864 vs 0.5660）；Kling Avatar 2.0 能跟指令但不能实时。

### 5.3 人类偏好

![[assets/Vidu S1 - 语音控制的实时互动视频生成/figures/fig3_preference.png]]
*Figure 3（p.9）：Vidu-StreamBench A/B 偏好。总体偏好上 Vidu S1 对 HeyGen 56% vs 28%、对 LemonSlice 46% vs 30%、对 Kling Avatar 2.0 48% vs 30%；**主体可控性（Subject Controllability）对 HeyGen 与 LemonSlice 达 100% 偏好**——所有评分者在显式动作指令任务上都选了 Vidu S1。来源：官方 PDF Figure 3。*

![[assets/Vidu S1 - 语音控制的实时互动视频生成/figures/fig4_qualitative.png]]
*Figure 4（p.9）：与 Kling Avatar 2.0 的定性对比。Prompt 1「抬头并摆出沉思姿态」中 Kling 未抬头且有背景伪影（红框）；Prompt 2「竖大拇指」中 Kling 手势未正确执行，Vidu S1 均正确完成（绿框）。来源：官方 PDF Figure 4。*

### 5.4 实时能力

3 步蒸馏配置下，RTX 5090 平均吞吐 42 FPS，超过 30 FPS 实时播放阈值，且是在保持身份一致性、音画同步与感知质量的前提下达成。

---

## 6. 局限与评价

### 6.1 边界

- **场景窄**：实验聚焦单人单镜头数字人；多人交互、大场景未涉及。
- **内部基准主导**：Vidu-StreamBench 是论文自建，外部可复现性依赖其开放程度；缺少逐模块消融表，TwinCache / RoPE Repositioning / 各系统优化各自贡献多少无法精确量化。
- **数据管线复杂**：依赖 diarization、Omni 模型标注等重工序，唱歌/强背景音乐场景仍需启发式补丁；训练数据要求 3–60 秒高质量单镜头片段。
- **部署门槛**：多 GPU Ulysses 并行 + 定制 CUDA 算子 + CUDA Graph，复现工程成本高。
- **代码未开源**：训练与推理栈均不可获得，只能通过在线 Demo 与 API 验证。

### 6.2 个人判断

这篇报告的价值不在单点算法创新——三阶段训练（双向 teacher → 因果 teacher 混合 forcing → DMD+PCM 蒸馏）与滑动窗口 + sink 的组合，在 Self-Forcing / Causal Forcing / LongLive 一系工作里都能找到原型（论文自己也大量引用了这条谱线）。它的增量在于三件事：**语音作为显式未来控制信号进入联合 video-audio 状态**；**TwinCache 的带噪/干净双缓存调度**是对「时间传播 vs 外观精修」矛盾的一个干净解法；以及把整条链路（数据、模型、内核、服务）作为系统问题一起交付，并在消费级 GPU 上给出可验证的 42 FPS。对做实时流式生成的人来说，它更像一份「把现有技术拼成可上线系统」的工程蓝图，而不是方法论文；消融缺失让组件级贡献难以归因，这是读它时需要保留的折扣。

### 6.3 复现成本

- 模型权重与训练/推理代码均未开源，无法本地复现；
- 可通过 vidu.com/vidu-stream 在线 Demo 或 platform.vidu.com 的 API 定性验证交互能力；
- 若要复现同等能力，需自建数据管线（diarization + Omni 标注）+ 三阶段训练 + 完整推理栈，工程量约等于一个中等规模的系统团队。
