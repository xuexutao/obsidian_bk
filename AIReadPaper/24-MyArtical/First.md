# 视频生成模型：技术谱系与 2026 现状

---

## 基本信息

|   |   |
|---|---|
|项目|内容|
|主题|视频生成模型的技术分支谱系与 2026 年现状全景|
|定位|与本 vault 已有两篇互补：[[视频生成模型发展与架构演进 - Survey]]（纵向历史：表示/骨干/范式三主线）与 [[因果视频蒸馏路线 - Survey]]（专项：双向扩散→因果自回归蒸馏）分别覆盖历史与蒸馏，本篇做**横向分支谱系 + 2026-10 快照**|
|时间|2026-10-10|
|主要参考|Wan 技术报告 [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)、HunyuanVideo 1.5 技术报告 [arXiv:2511.18870](https://arxiv.org/abs/2511.18870)、Sparse VideoGen [arXiv:2502.01776](https://arxiv.org/abs/2502.01776)、Radial Attention [arXiv:2506.19852](https://arxiv.org/abs/2506.19852)、FramePack [arXiv:2504.12626](https://arxiv.org/abs/2504.12626)、DanceGRPO [arXiv:2505.07818](https://arxiv.org/abs/2505.07818)、Controllable Video Generation: A Survey [arXiv:2507.16869](https://arxiv.org/abs/2507.16869)|
|附件|`assets/视频生成模型技术谱系与2026现状 - Survey/figures/`（8 张官方图）、`sources/`（PDF 与溯源）|

---

## 1. 本篇的分工：三篇综述拼一张地图

视频生成这个领域单篇综述已经装不下了。本 vault 里已有的两篇是这样分工的：

- **架构演进篇**回答「怎么走到今天」：逐帧建模 → VQ-VAE 自回归 → 时空分解扩散 → DiT 革命 → 3D VAE + 大规模 DiT，止于 2025 的 Sora/CogVideoX/HunyuanVideo/Movie Gen 五个里程碑；
- **因果蒸馏篇**回答「怎么变快」：DMD 分布匹配 → CausVid 非对称蒸馏 → Self-Forcing 弥合训练推理鸿沟 → Self-Forcing++ 分钟级 → OmniForcing 音视频实时；
- **本篇**回答「现在长什么样、往哪分叉」：把 2026 年 10 月的格局按**基座、可控性、效率、长视频、后训练、世界模型接口**六个分支摊开，每个分支给出代表工作与关键数字，并附一张闭源/开源现状快照表。

三篇拼起来才是完整地图：历史（从哪来）→ 专项（一条最深的主线）→ 谱系（全景分支）。

---

## 2. 2026 年基座模型格局快照

先给结论：**视频生成已经进入「没有单一赢家」的阶段**。第三方盲测竞技场（Artificial Analysis Text-to-Video Arena，2026 年中快照）上 Seedance 2.0 以 Elo 1213 排第一，Veo 3.1（1100）与 Kling 3.0 Omni（1097）随后，开源阵营由 Wan 2.5 与 LTX-2 领跑。各家旗舰每 6–10 周出一个大版本，任何「当前最强」都只是快照。

### 2.1 闭源旗舰

|                  |          |                                                                |            |                                      |
| ---------------- | -------- | -------------------------------------------------------------- | ---------- | ------------------------------------ |
| 模型               | 团队       | 突出能力                                                           | 原生音频       | 备注                                   |
| Seedance 2.0/2.5 | 字节       | Arena 第一；多镜头叙事；@Image 引用绑定（最多 4 张参考图绑定到具体角色/产品）；2.5 起单镜头最长 30s | ✅          | 闭源                                   |
| Veo 3.1          | Google   | 电影感基线；提示遵循与物理；2026-01 起 4K                                     | ✅（最早普及者之一） | 闭源                                   |
| Kling 2.6/3.0    | 快手       | 2.6 首次同步音视频；运动笔刷、首末帧控制成熟；3.0（2026-02）4K、15s、6 镜头               | ✅          | 闭源                                   |
| Sora 2           | OpenAI   | 25s 单镜头连贯性、物理与对象持久性最强档                                         | ✅          | **2026-03 宣布退场，API 于 2026-09-24 停服** |
| Runway Gen-4.5   | Runway   | 多镜头约 1 分钟；参考驱动一致性                                              | ✅          | 闭源                                   |
| Hailuo 2.3       | MiniMax  | 真人微表情、身体运动                                                     | ❌（截至 2.3）  | 闭源                                   |
| Genie 3          | DeepMind | 实时可交互世界（720p@24fps），严格说是世界模型而非普通视频生成                           | —          | 见第 8 节                               |

**Sora 2 退场是 2026 年最重要的生态事件**：它证明了对闭源单点的依赖是结构性风险，API 说停就停，下游产品必须迁移。这直接抬升了开源基座的战略价值。

### 2.2 开源基座

|   |   |   |   |
|---|---|---|---|
|模型|参数|关键设计|许可|
|Wan 2.2（阿里通义，2025-07）|A14B：27B 总/14B 激活 + TI2V-5B|首个开源 MoE 视频扩散；Wan2.2-VAE 16×16×4 高压缩；4090 可跑 720p@24fps|Apache 2.0|
|Wan 2.5（2025-10）|未全开源|原生音视频、1080p；开源线质量标杆|部分闭源|
|HunyuanVideo 1.5（腾讯混元，2025-11）|8.3B DiT|SSTA 稀疏注意力；Qwen2.5-VL + ByT5 文本编码；14GB 显存可跑；两阶段出 1080p|社区许可|
|LTX-2（Lightricks）|22B/19B|原生 4K、高帧率、A/V 同步、Fast→Pro→Ultra 分层|开放权重|
|SkyReels-V2/V3（昆仑万维）|14B/19B|Diffusion Forcing 长视频；多参考图、音频驱动|开放|
|Mochi 1 / CogVideoX / Step-Video / Open-Sora|1B–30B|早期开源生态，现多作研究底座|Apache 2.0 为主|

开源阵营的真实格局比营销窄：**真正同时满足「权重开放 + 质量第一梯队 + 许可可商用」的，2026 年 10 月基本就是 Wan 系（Apache 2.0）与 HunyuanVideo 系两条线**，LTX-2 在 4K/高帧率生态位上补充。

---

## 3. 基座架构：收敛与三条分岔

![Wan 2.1 DiT 结构](figures/wan_dit_arch.png)

**图：Wan 2.1 的 DiT 设计**（来源：[arXiv:2503.20314](https://arxiv.org/abs/2503.20314) HTML 版配图 dit_arch）。DiT block 内以 3D 全注意力处理时空 patch，文本经 T5 类编码器以 cross-attention 注入——这是 2025 年后开源旗舰的标准形态。

到 2026 年，所有第一梯队开源基座已经**收敛到同一套骨架**：

1. **3D causal VAE** 做时空压缩（空间 8–16×、时间 4×），latent 空间做扩散；
2. **DiT 骨干 + 流匹配**（rectified flow）目标，取代 DDPM 噪声预测；
3. **3D full attention** 统一建模时空依赖（取代早期 U-Net 时代的时空分离注意力）;
4. **VLM 级文本编码器**（Qwen2.5-VL 类）替代纯 T5，配 ByT5 字形编码保文字渲染；
5. **渐进式 curriculum**：低分辨率短时长 → 高分辨率长时长 → 高质量微调。

但在这套共同骨架之上，出现了三条值得区分的分岔：

**分岔一：MoE 化（Wan 2.2）。** 把去噪过程按噪声水平拆给两个专家——高噪专家管布局与运动规划，低噪专家管细节精修。切换点由信噪比 SNR 决定：SNR 降到一半以下即从高噪专家切到低噪专家。总参数 27B、每步只激活 14B，容量翻倍而单步算力不变。这是把 LLM 的 MoE 思想搬到「时间步维度」而非「token 维度」——专家按去噪阶段分工，是视频扩散特有的 MoE 形态。

**分岔二：小钢炮 + 稀疏注意力（HunyuanVideo 1.5）。** 反向路线：不堆参数，用 8.3B DiT 配自研 SSTA（Selective and Sliding Tile Attention）动态剪掉冗余时空 token，10s 720p 端到端比 FlashAttention-3 快 1.87×，14GB 消费卡可跑，再用少步超分网络拉到 1080p。它证明「小模型 + 系统优化」可以顶住 27B 旗舰。

**分岔三：高压缩 VAE（Wan 2.2-VAE）。** 压缩比 4×16×16（总 64×，加 patchify 达 4×32×32），让 5B 密集模型在单张 4090 上 9 分钟出 5s 720p@24fps。VAE 压缩率正在成为和参数量同量级的工程杠杆。

![HunyuanVideo 1.5 总览](figures/hunyuanvideo_overview.png)

**图：HunyuanVideo 1.5 全管线**（来源：[arXiv:2511.18870](https://arxiv.org/abs/2511.18870) HTML 版配图 hunyuanvideo_overview）。从数据 curated、Qwen2.5-VL + ByT5 双文本编码、8.3B DiT（SSTA）到视频超分两阶段出 1080p 的完整工业化管线，可当作 2025 年末开源基座的标准样板对照阅读。

---

## 4. 可控性：条件信号图谱与相机控制子线

![可控视频生成的条件信号分类](figures/ctrl_survey_taxonomy.png)

**图：可控视频生成的条件信号总览**（来源：Controllable Video Generation: A Survey，[arXiv:2507.16869](https://arxiv.org/abs/2507.16869) HTML 版配图 visual_0706）。Depth/Subject/Image/BBox/Identity/Style/Flow/Camera/Voice/Trajectory/Sketch/Landmark/BEV/Pose/Text-Rendering/Motion 共 16 类条件。值得注意：BEV-Guided 一列已经是自动驾驶场景，条件信号与研究社区正在向具身靠拢。

按这篇综述的划分，可控性方法沿三条轴组织：**控制什么**（外观/运动/相机/音频/结构）、**怎么注入**（cross-attention 适配器、token 拼接、ControlNet 式旁路、guidance 干预）、**训练还是免训练**。2026 年的趋势是**条件信号正在从外挂适配器走向基座内置**。

![Wan 2.1 相机控制模块](figures/wan_framework.png)

**图：Wan 2.1 内置的相机控制**（来源：[arXiv:2503.20314](https://arxiv.org/abs/2503.20314) HTML 版配图 framework）。相机外参序列经 Plücker embedding 编码，由 Camera Encoder（ResBlock 栈）生成特征，经 Camera Adapter 以 scale/shift 方式注入 DiT 的 self-attention 特征——条件注入从「外挂 ControlNet」变成「基座原生模块」。

相机控制是其中与三维重建交叉最深的一条子线，演化脉络清晰：

- **MotionCtrl → CameraCtrl**（2024）：专用相机编码器注入 6DoF 轨迹，首次把「相机」变成显式控制信号；
- **ViewCrafter**（2024）：相机条件 + 3D 先验做新视角合成，是「视频生成当 NVS 用」的早期尝试；
- **ReCamMaster**（快手 Kling，ICCV 2025 Oral & Best Paper Finalist，[arXiv:2503.11647](https://arxiv.org/abs/2503.11647)）：换机位重摄给定视频（V2V），核心是帧级 video conditioning + UE5 合成的 136K 多机位同步数据（13.6K 场景、122K 轨迹）弥合 sim-to-real 域差；
- **GEN3C**（NVIDIA）：干脆用 3D 点云缓存做条件（见 [[Lyra系列 - 生成式三维场景重建]] 对 GEN3C 的定位），相机控制精度来自显式 3D 而非学出来的 embedding。

一条规律：**控制信号越几何化（深度/位姿/点云），精度越高、生成自由度越低；越语义化（文本/参考图），自由度越高、精度越低**。2026 年的产品形态（Kling 运动笔刷、Seedance @Image 绑定、Veo 的文字运镜）本质是在这条谱上给不同用户群选不同的工作点。

---

## 5. 效率：稀疏注意力成为主战场

蒸馏线（CausVid/Self-Forcing/OmniForcing，步数压缩）已由 [[因果视频蒸馏路线 - Survey]] 专文覆盖；系统侧的另一条主线是**稀疏注意力**，针对的正是 3D full attention 占推理 80%+ 的瓶颈。

![SVG 与 Radial Attention 对比](figures/radial_idea.png)

**图：Sparse VideoGen 与 Radial Attention 的对比**（来源：[arXiv:2506.19852](https://arxiv.org/abs/2506.19852) HTML 版配图 idea）。(a) SVG 逐头在线判别「空间头/时间头」再分别套掩码——推理快，但 profiling 在训练和长视频外推时会误判；(b) Radial Attention 用统一的 O(n log n) 静态掩码同时覆盖训练与推理，右侧三个勾表示推理、训练、长视频适配三项全能。

这条线两步走：

- **Sparse VideoGen**（Berkeley/MIT，ICML 2025，[arXiv:2502.01776](https://arxiv.org/abs/2502.01776)）：训练自由，发现注意力头天然分成空间头（只看帧内）与时间头（只看跨帧），在线 profiling 分类后各自套稀疏掩码 + 定制 kernel，CogVideoX/HunyuanVideo 上端到端 2.28×/2.33× 加速，PSNR≈29 几乎无损。
- **Radial Attention**（MIT Han Lab，[arXiv:2506.19852](https://arxiv.org/abs/2506.19852)）：提出**时空能量衰减**（Spatiotemporal Energy Decay）——post-softmax 注意力随空间/时间距离指数衰减，类比波的传播。据此把两类头统一为一个静态径向掩码，窗口随时序距离指数收缩，复杂度 O(n log n)。关键增量是**训练也可加速**：4× 长视频 LoRA 适配成本降 4.4×，推理 3.7×。

而 HunyuanVideo 1.5 的 SSTA 说明这条线已经**从社区优化内化进官方架构**——2026 年新发布的基座默认带稀疏注意力，不再是可以后挂的 trick。与之并行的还有：缓存复用（TeaCache、Δ-DiT）、量化（ViditQ、NVFP4，见 [[LongLive-2.0 - NVFP4长视频生成训练与推理系统]]）、序列/上下文并行。效率优化已从单点 trick 变成**训练-架构-系统协同**的系统工程。

---

## 6. 长视频：遗忘与漂移的根本两难

![FramePack 三种采样顺序](figures/framepack_sampling.png)

**图：FramePack 的抗漂移采样设计**（来源：[arXiv:2504.12626](https://arxiv.org/abs/2504.12626) HTML 版配图 sampling）。(a) 朴素逐段生成误差单向累积；(b) 先定端点帧再填中间，打断因果链；(c) 倒序生成，每段都向已知高质量帧靠拢。斜纹块为正在生成的段。

FramePack（Lvmin Zhang & Maneesh Agrawala，Stanford）把长视频自回归的根本矛盾说透了：**遗忘与漂移是一对此消彼长的矛盾**——增强记忆（编码更多历史帧）会放大误差传播；切断误差传播（重噪/掩码历史）又削弱记忆。它的两个贡献：

1. **FramePack 结构**：按重要性（时间邻近度/特征相似度）逐帧压缩历史上下文，使 transformer 上下文长度收敛到固定上界——**不管视频多长，计算瓶颈等同图像扩散**，13B 模型 6GB 显存的笔记本能跑千帧级推理；
2. **抗漂移采样**：端点先行 + 倒序生成（上图 b/c），并在训练中把历史转成离散 token 缩小训练/推理分布差。

放到谱系里看，长视频在 2025–2026 年分成三派：

- **分段拼接派**（StreamingT2V、anchor 帧）：实现简单但接缝与剧情漂移难解，基本退出旗舰；
- **因果流式派**：Diffusion Forcing 定义框架 → CausVid/Self-Forcing 系蒸馏提速 → Self-Forcing++ 分钟级 → OmniForcing 音视频实时（全部见 [[因果视频蒸馏路线 - Survey]]）；
- **上下文打包派**：FramePack 代表，不改变扩散目标而改变**给模型看什么**——历史被压缩而非截断。

SkyReels-V2（Diffusion Forcing 工程化）、Infinite-World（千帧 HPMC 记忆压缩，见生成式世界模型综述）与 FramePack 属于后两派的代表。**「历史如何进入上下文」是长视频的下一个架构竞赛**，这与三维重建里 Spann3R 的空间记忆、Lyra 2.0 的几何路由是同一个问题的两种解法。

---

## 7. 后训练：RL 对齐进场

**DanceGRPO**（字节 Seed + 港大，[arXiv:2505.07818](https://arxiv.org/abs/2505.07818)）是把 GRPO（LLM 时代验证过的组相对策略优化）系统性搬进视觉生成的第一个统一框架：跨两种生成范式（扩散 + 流匹配）、三个任务（T2I/T2V/I2V）、四个基础模型（Stable Diffusion、HunyuanVideo、FLUX、SkyReel-I2V）、五种奖励模型（美学、文图对齐、视频运动质量、二元反馈等），基准上最高提升 181%。

三个技术要点：

1. **兼容 ODE 采样**：此前 DDPO/DPOK 类方法与流匹配的 ODE 采样不兼容、大 prompt 集下不稳定，DanceGRPO 利用 GRPO 的组内相对优势机制解决了稳定性；
2. **Best-of-N 推理扩展**：RL 后的策略更会「沿去噪轨迹走」，配合 Best-of-N 采样可继续涨点——训练侧与推理侧的对齐收益可以叠加；
3. **稀疏二元反馈可学**：不依赖稠密打分，点赞/点踩级别的信号也能用，这对真实产品数据闭环很关键。

配套的奖励模型（VideoReward、VisionReward、VideoAlign）与评测基准（VBench、Wan-Bench 2.0、TailOR 物理理解）一起，标志着视频生成从「纯预训练 Scaling」进入「**预训练 + RLHF 双轮**」阶段——这正是 LLM 2023 年走过的路。值得注意的差别：视频的奖励信号维度更多（运动质量、物理合理性、时序一致性），奖励 hacking 风险更高，目前还没有出现 LLM 级别的成熟 RL 基础设施。

---

## 8. 视频生成 → 世界模型与 3D：三种角色

这是与三维重建交接的一节，视频模型在具身与 3D 里扮演三种角色：

**角色一：数据引擎。** 用文本/条件生成带相机轨迹的训练视频，喂给重建、NVS、策略学习。Veo/Wan 类模型的相机可控性 + 物理合理性直接决定数据质量。

**角色二：交互式世界本体。** Genie 3（DeepMind，2026-02 向 Ultra 用户开放）自回归逐帧生成可导航世界：720p@24fps 实时、约一分钟记忆一致性、支持 promptable world events（中途改天气）。Waymo 已用它微调出 Waymo World Model 生成极端驾驶场景。NVIDIA Cosmos 是另一极：Predict2.5（视频生成）+ Transfer2.5（把 Omniverse 物理仿真渲染成真实感视频）+ Reason（物理常识推理），2026-02 的 Cosmos Policy 直接后训练出操作策略（真实任务完成率 +12.5%）。

**角色三：策略先验/世界动作模型。** DreamZero/DreamDojo（NVIDIA，2026）：用 14B 视频扩散骨干**单次前向同时预测未来世界状态与机器人动作**，10–20 分钟示教视频即可跨本体迁移，泛化超标准 VLA 基线两倍以上。这里视频模型不再是数据生成器，而直接是策略的表征底座——VLA 与世界模型正在合流成 World Action Model。

**与三维重建线的接口**在本 vault 已有专文：GEN3C 的点云缓存（视频生成←3D 条件）、Voyager 的 RGB-D 联合生成、Lyra 的自蒸馏出 3DGS、WorldMirror 的先验 token 注入——见 [[Lyra系列 - 生成式三维场景重建]] 与 [[生成式世界模型路线的前馈三维重建 - Survey]]。一条贯穿性判断：**视频模型管「看起来对」，几何模型管「真的是对」**，两侧分工在 2026 年已经定型；而 Cosmos 与 Genie 3 的对照（物理锚定 vs 像素涌现）说明「世界模型的物理正确性」还没有共识解法——sim-to-real 的瓶颈是物理不是像素。

---

## 9. 判断与局限

**判断一：基座架构已收敛，创新点外移。** 3D causal VAE + DiT + 流匹配 + 3D full attention + VLM 编码器这套骨架没有人再推翻，竞争转移到 MoE 化、稀疏注意力、VAE 压缩率、数据引擎与 RL 后训练——2026 年的论文创新集中在「骨架之外」。

**判断二：闭源单点风险已被市场定价。** Sora 2 API 停服事件后，「可自持的开源基座」从省钱选项变成架构决策。Wan（Apache 2.0）与 HunyuanVideo 的双雄格局对产业是健康冗余。

**判断三：效率与长视频是同一枚硬币。** 稀疏注意力（Radial/SSTA）同时服务推理加速与长视频外推；FramePack 的上下文压缩与 Spann3R 的空间记忆在数学上是同构思想——「有限预算下保留什么」。跨领域迁移值得做。

**判断四：物理正确性是下一个分水岭。** VBench 的美学/一致性分数各家已接近，TailOR 类物理基准的差距仍然巨大；Genie 3 的「行人倒着走」幻觉与 Cosmos 选择物理引擎锚定，代表两种赌注。

**局限（本篇的方法论）**：闭源模型（Seedance/Veo/Kling 3.0）无技术报告，能力描述依赖第三方竞技场与厂商博客，属于间接证据；版本迭代 6–10 周一次，本篇数字是 2026-10-10 快照；RL 后训练一节只有 DanceGRPO 深入，VideoReward/VisionReward 未单独精读；音频驱动数字人（OmniAvatar 等）与本篇未展开的多语言/多模态指令跟随留待后续。

---

## 10. 关键文献索引

|   |   |   |
|---|---|---|
|工作|arXiv|本篇角色|
|Wan: Open and Advanced Large-Scale Video Generative Models|[2503.20314](https://arxiv.org/abs/2503.20314)|开源基座样板：DiT/VAE/相机控制/数据管线|
|HunyuanVideo 1.5 Technical Report|[2511.18870](https://arxiv.org/abs/2511.18870)|小钢炮路线：SSTA 稀疏注意力|
|Sparse VideoGen|[2502.01776](https://arxiv.org/abs/2502.01776)|稀疏注意力第一步：空间/时间头分类|
|Radial Attention|[2506.19852](https://arxiv.org/abs/2506.19852)|时空能量衰减 → O(n log n) 统一掩码|
|FramePack|[2504.12626](https://arxiv.org/abs/2504.12626)|长视频：遗忘-漂移两难与上下文打包|
|DanceGRPO|[2505.07818](https://arxiv.org/abs/2505.07818)|RL 后训练统一框架|
|Controllable Video Generation: A Survey|[2507.16869](https://arxiv.org/abs/2507.16869)|可控性条件信号分类总纲|
|ReCamMaster|[2503.11647](https://arxiv.org/abs/2503.11647)|相机控制 V2V 重摄（ICCV 2025 Oral）|
|因果蒸馏系列（CausVid/Self-Forcing++/OmniForcing）|见 [[因果视频蒸馏路线 - Survey]]|蒸馏专项已另文覆盖|
|LongLive-2.0|见 [[LongLive-2.0 - NVFP4长视频生成训练与推理系统]]|低精度训练/推理系统|

### 本 vault 相关笔记

- [[视频生成模型发展与架构演进 - Survey]]（历史主线）
- [[因果视频蒸馏路线 - Survey]]（蒸馏专项）
- [[Lyra系列 - 生成式三维场景重建]]（视频先验 → 3D）
- [[生成式世界模型路线的前馈三维重建 - Survey]]（世界模型另一侧）
