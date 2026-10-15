---
type: paper-note
status: done
domain: Video-Generation
paper: Vidu S2
year: 2026
arxiv: 2609.11638
source: https://arxiv.org/abs/2609.11638
project: https://vidu.com/vidu-stream
code: null
tags: [Video-Generation, RealTime, StreamingDiffusion, Avatar, VideoEditing, SpatialVideo, Distillation]
---

# Vidu S2：可编辑、可空间化的实时流式视频生成

## 基本信息

- **论文正式名称**：Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation
- **作者与机构**：Jintao Zhang、Kai Jiang、Jintao Chen 等 35 人（清华大学 × 生数科技；顾问 Zhijie Deng、Fan Bao、Jianfei Chen、Jun Zhu；一作团队与 Vidu S1 高度重合）
- **arXiv**：2609.11638（v1 2026-09-10，发布五天后登顶 HF Daily Papers）
- **项目页 / Demo**：https://vidu.com/vidu-stream；GitHub 文档仓 https://github.com/shengshu-ai/Vidu-S（README + API 索引，无权重/训练代码）
- **关键规格**：720p @ 25–42 FPS（S1 为 540p/42），动态参考图随时可更新，支持跳舞等大幅度动作，实时视频流编辑，实时空间（立体）视频
- **前置阅读**：[[Vidu S1 - 语音控制的实时互动视频生成]]、[[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]]、[[Diffusion Forcing - 下一Token预测与全序列扩散]]、[[OmniForcing - 实时音视频联合流式生成]]

## 一句话结论

Vidu S2 把 S1 的「语音可控数字人」扩展成**双子系统 + 一条空间视频支线**：S2-Avatar 用 Self-Replay Forcing（对学生自生成轨迹重加噪后做梯度使能的因果重放）解决流式误差累积的监督信号问题，用单步潜空间 Refiner + 非对称噪声缓存把实时分辨率推到 720p，用 VLM agentic system 让动态参考图（换装/拿物体/换背景）在流中随时生效；S2-Editing 用 frame-aligned attention（目标帧只读同时刻源帧）在保持动作时序不变的前提下对流式输入做风格化/换装/换人/换背景；两者都能转换成同步左右目视图输出给 VR 头显。公开基准上 StreamAV-Bench 全指标第一、Sparkle-Bench/OpenVE/RefVIE/ViViD 全面领先，但所有结果均为自报、无消融、未开源。

---

## 1. 研究背景：从 S1 到 S2 的四个缺口

![[assets/Vidu S2 - 可编辑可空间化的实时流式视频生成/figures/fig1_overview.png]]
*Figure 1（p.1）：Vidu S2 总览。上半为 S2-Avatar：语音指令驱动的 720p 实时数字人 + 风格/服装/背景/物体的实时编辑与参考图交互；下半为 S2-Editing 的四类编辑与 Real-Time Spatial Video 输出。来源：官方 PDF Figure 1。*

论文沿用 S1 的需求论证（实时需求 $\alpha\times N$ vs 离线需求 $\beta\times N/m$），然后把 S1 的边界列得很诚实：分辨率止步 540p；角色参考一旦开播即固定；跳舞等大幅度身体动作难以跟随；不支持编辑输入视频流。S2 的四个方向正对着这四个缺口。

其中方法上最核心的是 **Self-Replay Forcing（SRF）**。论文对 [[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]] 的批评是：Self-Forcing 虽然让每段条件来自模型自生成的 chunk（on-policy，方向正确），但 (1) 自生成历史以**干净**形式喂入而非带噪，与 Diffusion Forcing 的噪声调度不一致；(2) 自生成历史被 **detach**，梯度不流经历史，数据效率与训练质量受限。SRF 的解法见 §2.3。

---

## 2. Vidu S2-Avatar：720p 实时交互数字人

### 2.1 数据准备：在 S1 流水线上加三个算子 + 重做 caption

![[assets/Vidu S2 - 可编辑可空间化的实时流式视频生成/figures/fig2_data_pipeline.png]]
*Figure 2（p.4）：双数据流水线。左：S2-Avatar 继承 S1 五阶段（Clipping/Filtering/Speech/Captioning/Embedding），新增切点检测、高清晰度筛选、背景稳定三个算子，数据源扩展 solo dance 与 2D/3D 动画；右：S2-Editing 从 80 万条精选视频切出 4×20 万条互斥子集构造四类编辑对，风格对由「法线条件 V2V 重建模型」两阶段生成。来源：官方 PDF Figure 2。*

保留 S1 的六维过滤体系，四个关键改动：

1. **切点检测升级**：vlog/开箱类视频大量隐藏剪辑，单纯提高检测灵敏度会误报，于是对候选切点附近抽帧交给 VLM 二次判定，把误拒率压到 2% 以下。
2. **高清晰度筛选**：名义分辨率不可信——重压缩的 1080p 可能比良好编码的 720p 更糊。按分辨率/帧率硬阈值 → codec/像素格式/位深/码率的相互关系评估 → 专家模型评技术质量/纹理细节/边缘锐度/压缩伪影 → 加权分聚合筛选。这个「按实测清晰度而非名义分辨率选数据」的细节是 720p 能立住的数据侧前提。
3. **背景稳定算子**：舞蹈视频常有环绕/推拉/变焦/跟踪镜头，直接丢掉会损失动作多样性，保留又会破坏背景一致性学习。观察是多数舞蹈视频的相机平移有限、视差小，背景运动可近似为跨帧几何变换——于是前景分割掩膜后，从背景区域估计相机变换，几何校正 + 裁剪得到稳定背景，再做二次主体在框过滤。
4. **时间有序密集 caption**：S1 的结构化字段式 caption（主体/环境/镜头各一个字段）表达不了交互、时序演化与同步。S2 改为**按事件发生顺序**的密集描述——每个事件带时间边界（如「2.5s–3.8s 完成某动作及其后果」）；块级 caption 也改为按事件边界切分而非固定时长对齐语音，粒度更细、动作切换时延更低。标注管线为层级多 agent 设计（专家模型做检测/ASR 打事实地基 → 识别/描述/校验/过滤各由专门 agent 负责）。

### 2.2 模型与训练

**模型**：音频-视觉联合 Diffusion Transformer，参考图条件生成。参考图 $\boldsymbol{r}$ 与条件序列 $\boldsymbol{c}^{1:N}$ 下联合预测前 $N$ 段的干净 video-audio latent：

$$\hat{\boldsymbol{x}}_0^{1:N}=\boldsymbol{f}_\theta^{\mathrm{avatar}}(\boldsymbol{x}_t^{1:N},t,\boldsymbol{r},\boldsymbol{c}^{1:N})$$

**双向 I2V + R2V 联合训练**：同一个双向模型同时承担 image-to-video（$\boldsymbol{r}$ 为目标视频首帧）与 reference-to-video（$\boldsymbol{r}$ 为参考形象图）。与常规双向模型整段一个 prompt 不同，这里**每段用自己的条件 $\boldsymbol{c}^i$（该段 caption）监督**，分段条件监督显著提升指令跟随。

**混合 Teacher/Diffusion Forcing**：与 S1 Stage 2 相同——block-wise 因果掩码，Teacher Forcing（干净历史）与 Diffusion Forcing（带噪历史）按预定义概率混合，获得初始流式能力。

### 2.3 Self-Replay Forcing（SRF）：本文方法核心

SRF 是一个**保持跨块梯度流的 on-policy DMD**。流程：

1. 学生按推理流程做**长自回归 rollout**（Self-Forcing 式），生成的块与 KV cache 全部 detach——不保留 rollout 的完整计算图，避免显存爆炸；
2. 取学生自生成的整条轨迹，按 Diffusion Forcing 的方式**对每段独立重加噪**；
3. 在单次**梯度使能的因果 pass** 中重放整条轨迹：重放段之间彼此连通于同一计算图，因此**后面段的 loss 可以传到前面段**；rollout 遗留的外部历史保持固定。

$$\mathcal{L}_{\mathrm{SRF}}=\mathcal{L}_{\mathrm{DMD}}\!\left(\boldsymbol{f}_\theta^{\mathrm{causal}}(\boldsymbol{x}_{\boldsymbol{t}}^{1:N},t,\boldsymbol{r},\boldsymbol{c}^{1:N})\right)+\mathcal{L}_{\mathrm{perc}}\!\left(\boldsymbol{f}_\theta^{\mathrm{causal}}(\boldsymbol{x}_{\boldsymbol{t}}^{1:N},t,\boldsymbol{r},\boldsymbol{c}^{1:N})\right)$$

DMD 监督作用于重放段，感知损失防模式崩溃（沿用 S1 的 PCM 思路）；训练全程保留 sink blocks、滑动窗口与带噪 KV cache，保证训练与推理的注意力结构一致。

**理解要点**：Self-Forcing 只把「当前段」接进计算图、历史作为常数条件；SRF 把「整条轨迹」重放进计算图，等于把误差累积本身变成可优化的对象——模型在训练中就学会「接住自己的输出并修正」，而不是跟着自己漂。代价是每个训练迭代要跑一次完整 rollout + 一次因果重放，计算量明显更高（论文未给出训练开销数字）。

### 2.4 偏好优化

两个阶段各自叠加 RL 类目标：双向阶段用 diffusion DPO 提升视觉保真、表情丰富度、动作自然度与音画同步，给后续因果适配一个更强的 teacher；流式阶段用 Streaming NFT（DiffusionNFT）在按 SRF 原则构造的自生成轨迹上做基于奖励的优化，使优化状态对齐推理时的自回归分布。

### 2.5 单步超分 Refiner 与非对称缓存

backbone 在低分辨率 latent 上快速跑，Refiner 一步把分辨率抬到 720p。给定第 $i$ 段低分辨率 latent，先潜空间上采样再加固定精炼时间步 $t_r$ 的噪声，预测高分辨率 latent：

$$\boldsymbol{z}_{t_r}^i=(1-t_r)\,\mathcal{U}(\hat{\boldsymbol{x}}_{0,\mathrm{LR}}^i)+t_r\,\boldsymbol{\epsilon}^i,\qquad\boldsymbol{\epsilon}^i\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$$

缓存设计直接从 S1 的 TwinCache 推广成**非对称噪声水平**：backbone 看高噪缓存 $\hat{\boldsymbol{x}}_{\tau_{\mathrm{B}}}^{<i}$（$\tau_{\mathrm{B}}$ 大，粗时间结构、对高频伪影不敏感），Refiner 看低噪高分辨率缓存 $\hat{\boldsymbol{x}}_{\tau_{\mathrm{R}},\mathrm{HR}}^{<i}$（$\tau_{\mathrm{B}}>\tau_{\mathrm{R}}$，保外观与身份线索）。时间传播与空间精修彻底分家。

### 2.6 推理系统

与 S1 同栈：分层混合注意力（敏感层保精度、不敏感层用激进方法，在 Sage/Sparge/SLA 中逐层选）、per-block W8A8 GEMM、Triton/CUDA 融合 + CUDA Graph、Ulysses 上下文并行（新增：通信张量也量化以降传输量）。S2 的增量是**模块间调度**——VAE 编码器、backbone、Refiner、VAE 解码器在共享时间线上复用 GPU 组：backbone/refiner 空闲时让 VAE 模块顶上，减少静态预留造成的 GPU 闲置。

### 2.7 VLM Agentic System

![[assets/Vidu S2 - 可编辑可空间化的实时流式视频生成/figures/fig3_agentic.png]]
*Figure 3（p.8）：VLM agent 读取用户指令（语音转写）与图像 → 生成 prompt 驱动 S2-Avatar → 按时间序检查生成帧反馈给后续 prompt。来源：官方 PDF Figure 3。*

动态参考图要真正可用，光靠生成模型不够，论文用一层 VLM agent 补齐可靠性：

- **Prompt 生成**：分别描述身份外观、表情/视线/姿态/动作、手持物；用户没要求改变的细节显式保留（如「拿起杯子然后微笑」——微笑 prompt 里继续指定「仍持杯」）；
- **视觉反馈**：VLM 按时间序检查动作是否完成/部分完成/偏离，完成的判断会触发「描述结果姿态而不重复动作」的 prompt，不完整的触发修正指令；
- **参考图分类**：agent 先判断参考图是手持物/背景/服装，再组合成场景转移或一键换装 prompt；
- **配饰穿脱**：prompt 同时描述动作与穿脱后的持续状态（摘下→拿着→戴回，后续 prompt 继续标注「戴着」）。

这层设计的本质是把「状态记忆」从生成模型卸载给 agent——扩散模型每段独立生成，跨段状态一致性由 agent 的显式 prompt 维持。

---

## 3. Vidu S2-Editing：实时视频流编辑

### 3.1 数据：80 万视频 + 法线条件构造风格对

从 S2-Avatar 数据管线过滤结果中再施加更严标准并平衡标签分布，得 80 万条高质量视频，切出**四个互斥的 20 万条子集**分别对应风格迁移、主体替换、背景替换、虚拟试穿。

风格迁移数据的构造最巧妙：训练一个「法线视频 + 参考图 → 视频」的条件模型——用 NormalCrafter 从原视频估计表面法线序列，模型学习在法线给定的空间结构与运动下重建原视频；生成训练对时把参考图换成风格化图、法线序列保持不变，模型就输出「原视频的空间结构与运动 + 参考图的风格」。该模型可泛化到 50+ 种风格（赛璐璐、赛博朋克、莫奈、素描、工笔画等）。其他任务用类似思路微调任务专用编辑模型，并评估 Bernini、SCAIL-2、Wan2.1 VACE、SAMA-14B、CoinVE-Edit 等开源编辑模型补充数据——观察是没有单一模型在所有任务上都最好，因此做**后过滤 + 比较选择**。

Caption 层面有两个防泄漏细节：caption 标注编辑任务类型与参考图相关属性，并显式要求保留源视频中不编辑的内容；**生成 caption 时不把源视频喂给 VLM**——caption 只描述编辑目标与保留约束、不描述源内容，防止源信息通过文本条件泄漏。

### 3.2 方法：frame-aligned attention

DiT 编辑模型，源视频 $\boldsymbol{s}^{1:N}$、参考图 $\boldsymbol{r}$、逐段编辑指令 $\boldsymbol{c}^i$ 条件下预测目标 latent：

$$\hat{\boldsymbol{x}}_0^{1:N}=\boldsymbol{f}_\theta^{\mathrm{edit}}(\boldsymbol{x}_t^{1:N},t,\boldsymbol{s}^{1:N},\boldsymbol{r},\boldsymbol{c}^{1:N})$$

核心设计是**帧对齐注意力**：

- 生成的目标帧之间正常双向交互；
- 但每个目标帧**只读同时刻的源帧**，禁止源与目标的跨时刻自由混合——保证编辑后视频与输入的动作和时序完全一致；
- 参考图 token 对所有目标帧可见——新外观贯穿全片；
- 源视频与参考图作为条件 token 拼接，各 token 流用各自的 RoPE 编码表示时空位置。

这个设计同时保证了双向训练的表达能力与后续因果适配的兼容性。因果流式训练完全复用 Avatar 的配方：block-wise 因果掩码 → 混合 forcing → SRF（detach rollout + 重加噪 + 可微重放 + DMD 监督）。流式推理时每帧源视频与它产出的目标帧一起被消费、**不进缓存**——编辑延迟与用户动作保持同步。

---

## 4. 实时空间（Spatial）视频

![[assets/Vidu S2 - 可编辑可空间化的实时流式视频生成/figures/fig13_spatial.png]]
*Figure 13（p.21）：Spatial Video 示例。左：S2-Avatar 生成的左右目同步数字人视图；右：S2-Editing 对单目流做梵高风格化后转换的立体视图。来源：官方 PDF Figure 13。*

- **生成管线**：S2-Avatar 先生成单目 chunk → 逐帧深度估计并映射为水平视差 → 以单目帧为中心视图向左右反向 warp 出双眼视图 → 轻量处理缓解深度不连续伪影、填补 disocclusion 孔洞、时序稳定深度 → 同步为流式 chunk。
- **编辑管线**：两条工作流——单目输入先编辑再转换；立体输入把左右目水平拼接成一条宽流，一次编辑后拆回。编辑时同样支持风格/试穿/换人/换背景，输入甚至可以是头显 passthrough 的实时画面。
- **论文自认的差距**：头显下空间视频占据大视场、需要更高分辨率；passthrough 模式对端到端延迟极敏感（头部运动与画面失步会眩晕）；当前只是「固定视角空间视频」，远景是全景空间视频 + 实时生成的 metaverse。这一节定位是可行性探索，不是成熟能力。

---

## 5. 实验与结果

### 5.1 公开基准：数字人生成（StreamAV-Bench）

与 13 个开源/商业系统对比（PixVerse R1、HappyOyster、OmniForcing、Odyssey-2、Self-Forcing、LongLive、SWIFT、IAMFlow、SoulX-FlashTalk、AptAvatar、Live Avatar、AvatarForcing、Hallo-Live）。关键行如下（完整表含 9 指标）：

| 模型 | VA ↑ | VQ ↑ | PQ ↑ | AQ ↑ | AVAlign ↑ | AVSync ↓ | AIF ↑ | SC ↑ | BC ↑ |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Self-Forcing | .585 | 2.753 | 6.453 | 2.623 | .256 | .919 | 2.810 | .981 | .969 |
| LongLive | .605 | 2.768 | 6.459 | 2.696 | .272 | .969 | 2.814 | .986 | .973 |
| SWIFT | .605 | 2.735 | 6.483 | 2.680 | .270 | .970 | 2.887 | .986 | .973 |
| Odyssey-2 | .531 | 2.820 | 6.426 | 2.659 | .260 | .946 | 2.778 | .976 | .966 |
| HappyOyster | .529 | 2.785 | 6.817 | 3.157 | .206 | 1.044 | 2.692 | .904 | .934 |
| Live Avatar | .661 | 3.295 | 7.133 | 3.079 | .116 | 1.145 | 2.745 | .997 | .989 |
| **Vidu S2-Avatar** | **.687** | **3.370** | **7.138** | **3.286** | **.353** | **.617** | **2.985** | **.998** | **.993** |

S2-Avatar 在全部报告指标上第一，且增益分布在互补维度上而非单点：AVSync 误差 0.617（第二名 Self-Forcing 的 0.919）说明音画时间同步显著更紧，AVAlign 0.353 反映语义耦合更强，SC/BC 接近天花板说明长 rollout 的身份与场景稳定。

### 5.2 公开基准：视频编辑

| 基准 | 关键结果 |
| --- | --- |
| Sparkle-Bench | Overall **3.74** 全场第一（最强流式基线 Decart-Lucy2.5 为 3.67）；Ins. 4.00 / FgIn. 3.98 / FgMo. 4.00 |
| OpenVE | GS **4.71**、BC **4.14**、Overall **4.42** 第一 |
| RefVIE | Overall **3.78**，超流式基线 Decart-Lucy2.5（3.71） |
| Joint OpenVE–RefVIE | Overall **4.26**，超最强离线基线 Bernini-R 14B（3.92）**0.34** |
| ViViD 虚拟试穿（无配对） | VFIDI **9.9515**，大幅优于 CatV2TON（19.51）与 ViViD（21.80） |

一个流式模型在编辑质量上压过离线大模型（Bernini-R 14B），是这组数字里最有分量的结论。

### 5.3 人类偏好（内部 GSB）

![[assets/Vidu S2 - 可编辑可空间化的实时流式视频生成/figures/fig6_gsb_avatar.png]]
*Figure 6（p.17）：GSB 对比。上：S2-Avatar vs Runway Character GWM-1 / PixVerse Image Avatar / HeyGen——总体质量偏好率 85.7%（vs Runway）与 100%（vs PixVerse、HeyGen）；表情质量对三家几乎全胜；语义遵从 71.4%–100%。下：S2-Editing vs Decart-Lucy2.5 / XMax-X2.0，各维度均为更高偏好份额。来源：官方 PDF Figure 6。*

![[assets/Vidu S2 - 可编辑可空间化的实时流式视频生成/figures/fig8_qualitative.png]]
*Figure 8（p.18）：数字人定性对比。Vidu S2-Avatar 在化妆品/手部交互（女）与发型/眼色/服饰（男）上保持身份与细粒度几何稳定；PixVerse 出现面部身份与身体比例漂移，Runway 在头发、眉毛、手指上失真。来源：官方 PDF Figure 8。*

时长分层评分（Figure 5）补充了关键维度：10s→90s 各维度评分曲线几乎不掉，一致性与情绪表达在 90 秒仍维持高位——这与 SRF 针对「误差跨段累积」的设计动机互为印证。内部编辑 GSB 中时序一致性均分：S2-Editing 3.56 vs Decart-Lucy2.5 2.59 vs XMax-X2.0 1.67。

---

## 6. 局限与评价

### 6.1 边界

- **全部自报结果**：无论公开基准还是 GSB 都由官方评测，无独立复现；GSB 评分人数（20 名训练评估者）与样本量（编辑 150 对）不大，置信区间未报告。
- **无逐模块消融**：SRF 相对 Self-Forcing 式基线的增益、非对称缓存、Refiner、agentic system 各自贡献多少，论文没有给出拆解——SRF 作为方法核心恰恰最需要这个消融。
- **SRF 训练成本未披露**：每迭代一次 rollout + 一次重放，训练开销显著高于普通蒸馏，论文未给数字。
- **Spatial 部分是探索**：分辨率与延迟对头显场景仍不足，作者自己承认是 gap；warp 合成立体在遮挡区域的伪影也只是「轻量处理」。
- **未开源**：仅 README 文档仓 + 商业 API；TurboDiffusion/TurboServe 的实现也未随 GitHub 发布。
- **场景假设**：Avatar 仍是单主体数字人；Editing 依赖源帧对齐，源流本身的运动即编辑结果的运动，无法凭空加动作。

### 6.2 与 S1 的关系：一条清晰的系统演进线

| 维度 | Vidu S1 | Vidu S2 |
| --- | --- | --- |
| 分辨率/FPS | 540p @ 42 FPS | 720p @ 25–42 FPS |
| 长时稳定 | 混合 forcing + TwinCache | + Self-Replay Forcing（跨块梯度）+ Streaming NFT |
| 参考图 | 开播即固定 | 流中任意时刻动态更新 + VLM agent |
| 动作范围 | 说话头为主 | + 跳舞等大幅度动作（数据扩展 + 背景稳定） |
| 编辑 | 不支持 | S2-Editing 流式实时编辑 |
| 输出形态 | 单目 | + 空间视频（左右目同步） |
| 训练监督 | DMD + PCM | on-policy SRF（DMD + 感知正则）+ DPO + NFT |

### 6.3 个人判断

S2 的方法学增量集中在 **SRF**：它把 Self-Forcing「条件来自自身输出」的 on-policy 思想推进到「整条轨迹重加噪后可微重放」，让误差累积直接进入优化目标，逻辑上是 Diffusion Forcing 噪声调度与 Self-Forcing rollout 的自然合流，也是这条技术路线（[[Diffusion Forcing - 下一Token预测与全序列扩散]] → [[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]] → Causal Forcing → SRF）里目前最完整的训练-推理一致性方案。工程侧的两个设计同样值得记：**按实测清晰度而非名义分辨率选数据**，以及**编辑任务的帧对齐注意力**（用「目标帧只读同时刻源帧」一条约束同时买到时序一致与因果适配性）。保留意见：全部结果自报、核心方法无消融、权重不开源，SRF 的真实增益幅度与训练成本要等独立复现；编辑能力的「保留源运动」既是特性也是天花板。作为系列产品报告，它给出的最重要的信号是：实时流式视频生成已经从「能不能实时」进入「实时之上还要求可编辑、可空间化」的能力叠加阶段。

### 6.4 复现成本

- 权重与训练/推理代码未开源；训练侧 SRF 依赖大规模高质量数据管线（法线估计 V2V、多 agent 标注、背景稳定等），独立复现基本不可行；
- 在线 Demo（vidu.com/vidu-stream）与 API（platform.vidu.com/vidu-stream/doc）可验证 S2-Avatar / S2-Editing 的交互与编辑能力；
- 推理栈组件（SageAttention、SpargeAttention、SLA、per-block W8A8）多有独立开源，但完整系统集成未公开。
