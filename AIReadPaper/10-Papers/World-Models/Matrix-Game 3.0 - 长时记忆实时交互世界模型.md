---
type: paper-note
status: complete
domain: World-Models
paper: "Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory"
year: 2026
arxiv: 2604.08995
doi: null
source: https://arxiv.org/abs/2604.08995
project: https://matrix-game-v3.github.io/
code: https://github.com/SkyworkAI/Matrix-Game
tags:
  - World-Model
  - Interactive-Video
  - Long-Term-Memory
  - DMD
  - Real-Time-Generation
  - Action-Control
created: 2026-09-04
updated: 2026-09-04
---

# Matrix-Game 3.0：带长时记忆的实时流式交互世界模型

## 基本信息

- **完整标题**：Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory
- **作者**：Zile Wang、Zexiang Liu、Jiaxing Li、Kaichen Huang、Baixin Xu 等，Skywork AI
- **发表**：arXiv 2604.08995，v1 于 2026-04-10 提交，v2 于 2026-04-13 修订
- **基础模型**：Wan2.2-TI2V-5B；另扩展到 MoE-28B（2×14B）
- **目标能力**：720p、最高 40 FPS、分钟级交互生成、相机感知长时记忆
- **项目页**：<https://matrix-game-v3.github.io/>

## 一句话结论

Matrix-Game 3.0 不再只追求“下一段视频生成得快”，而是把交互式世界模型的工程矛盾拆成**误差自纠正、长时记忆检索、训练—推理对齐蒸馏和系统级加速**四个问题：5B 双向 DiT 用 error buffer 学会适应自身历史误差，用相机视锥检索把远期场景重新注入统一注意力，再以多 segment self-rollout 的 DMD 蒸馏对齐真实流式推理，最终在 8+1 GPU 配置下达到 720p@40 FPS。它是一个完整系统方案，但长时一致性和动作准确性仍以定性结果为主，严格 benchmark 证据明显弱于吞吐证据。

---

## 1. 研究背景与问题定义

### 1.1 从“实时片段”到“持续存在的世界”

Matrix-Game 2.0、HY-Gamecraft-2 等工作已经证明少步扩散能够实时生成动作可控的视频，但模型通常只有局部历史：玩家离开一个街角再回来，场景几何、物体位置和纹理可能被重新生成。交互视频由此暴露出一个比短视频质量更根本的问题——**世界是否持续存在**。

论文将目标明确为三项同时成立：

1. **实时性**：用户输入动作后必须立即生成下一段视频；
2. **高分辨率**：720p 级画面，不能靠极低分辨率换速度；
3. **长时一致性**：分钟级探索后重访旧区域，模型仍能恢复相同场景与物体状态。

这三项彼此牵制：更长上下文会使注意力和检索越来越贵；few-step 蒸馏会放大历史误差；高分辨率又会把瓶颈从 DiT 转移到 VAE 解码。

![](assets/Matrix-Game%203.0%20-%20长时记忆实时交互世界模型/figures/01-teaser.png)

> **图 1：Matrix-Game 3.0 的目标能力。** 上方展示动作条件下的多样生成，左侧把 spatial memory 与 temporal frames 作为世界状态，底部长序列展示约一分钟内的连续场景演化。图的价值不在单帧美观，而在于把“实时、可控、记得住”放进同一系统目标。
> 来源：论文 Figure 1，<https://arxiv.org/abs/2604.08995>

### 1.2 论文核心贡献

论文的贡献不是一个孤立模块，而是一条数据到部署的闭环：

- 构建 Unreal Engine 合成、AAA 游戏自动采集、真实视频增强组成的工业级数据引擎；
- 用 **error buffer** 把模型自身预测残差重新注入训练历史，使基础模型学会在不完美上下文中自纠正；
- 以相机位姿与视锥重叠检索长期记忆，并在统一 self-attention 中联合建模长期记忆、短期历史与当前目标；
- 提出多 segment 自生成 DMD 蒸馏，让训练条件与真实流式推理一致；
- 通过 INT8、MG-LightVAE、GPU 检索和异步多 GPU 流水线实现 720p@40 FPS。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与状态

训练数据被统一为 Video–Pose–Action–Prompt 四元组。对每个时刻 $t$，Unreal-Gen 同步记录：

$$\mathcal{D}_t=(\mathbf I_t,\mathbf p_t,\mathbf r_t,\mathbf c_t,\boldsymbol\theta_t,\mathbf a_t),$$

其中 $\mathbf I_t$ 是 RGB 帧，$(\mathbf p_t,\mathbf r_t)$ 是玩家位置与朝向，$(\mathbf c_t,\boldsymbol\theta_t)$ 是 6-DoF 相机位姿，$\mathbf a_t$ 是键盘、鼠标与游戏动作。

推理时的主要输入包括：

- 第一帧或 reference latent；
- 离散键盘动作与连续鼠标动作；
- 最近若干 past-frame latents；
- 依据相机位姿检索的长期 memory latents；
- 当前 segment 的随机噪声 latent。

输出是动作条件下的连续 720p 视频流。典型训练窗口由 5 个 memory latents、4 个 past-frame latents 和 10 个当前 noisy latents 构成；首个 segment 没有历史，退化为 image-to-video。

### 2.2 评价目标

论文实际关注四种质量：

1. 动作是否控制正确；
2. 局部运动与画面是否连续；
3. 重访旧视角时，几何、物体状态和外观是否一致；
4. 系统能否在目标硬件上持续实时运行。

需要注意，论文对第 4 点给出了较完整的量化消融，而第 1–3 点主要依赖视频样例和定性图，缺少统一的动作成功率与长时记忆指标。

---

## 3. 数据系统

### 3.1 Unreal-Gen 合成数据

团队在 Unreal Engine 5 中构建超过 1,000 个定制场景，利用 Nanite 与 Lumen 生成高质量画面，并在同一个 engine tick 内同步采集图像、玩家状态、相机位姿与动作，避免外部录屏最常见的时间错位。NavMesh 与强化学习 Agent 自动探索，第一/第三人称视角并存，组合式角色资产可产生超过 $10^8$ 种外观组合。

### 3.2 AAA 游戏与真实世界视频

第二路数据来自 GTA V、Red Dead Redemption 2、Cyberpunk 2077、Hogwarts Legacy 等 AAA 游戏的自动录制。系统把游戏进程、控制 Agent、录制协调和数据输出解耦，论文声称动作与画面对齐准确率超过 99%。

第三路是真实视频，包括 DL3DV-10K、RealEstate10K、OmniWorld-CityWalk、SpatialVid-HD，并用 ViPE 统一估计相机轨迹。这部分数据主要补足真实世界外观、运动与场景泛化，而不是提供精确键鼠监督。

### 3.3 数据过滤与证据边界

数据经过文本、运动、轨迹、速度与感知质量过滤，约剔除原始数据的 20%；记忆增强基础模型最终使用约 4.8M clips。论文没有充分披露每类数据的占比、许可证边界、训练总 GPU 时和总 token/帧数，这些都是外部复现的主要障碍。

![](assets/Matrix-Game%203.0%20-%20长时记忆实时交互世界模型/figures/02-overview.png)

> **图 2：系统总览。** 左侧数据引擎产生视频—动作—相机监督，中间训练带 error buffer 与 memory 的 DiT，右侧再通过少步采样、量化和剪枝部署为 5B 720p@40 FPS 系统。它强调的是协同设计，而非单个模型结构的“魔法”。
> 来源：论文 Figure 2，<https://arxiv.org/abs/2604.08995>

---

## 4. 核心方法

### 4.1 误差感知基础模型：让模型见过自己的错误

普通 teacher forcing 用干净真值历史预测未来，实际推理却只能看到模型自己的生成结果。随着 segment 数增加，小误差会被再次作为条件，最终形成滚雪球式漂移。

Matrix-Game 3.0 维护一个误差缓冲区 $\mathcal E$。周期性地让当前模型生成样本，并计算预测 latent 与真值 latent 的残差：

$$\delta=\hat{x}^{i}-x^{i}.$$

训练时从缓冲区采样残差，以强度 $\gamma$ 注入历史条件：

$$\tilde{x}^{i}=x^{i}+\gamma\delta.$$

于是模型优化的不再是“给定完美历史时预测下一段”，而是“历史已带有自己常见误差时如何修正并继续生成”。这与 Diffusion Forcing 的随机噪声扰动不同：error buffer 使用的是**模型真实产生的结构化残差**，更接近推理分布。

### 4.2 相机感知长期记忆

长期记忆首先是一个检索问题。模型根据当前相机位姿、视场与历史视锥的几何重叠，从不断增长的 memory pool 中取回最相关帧。相对于“最近 N 帧”，这种方式能在玩家折返时找到很久以前但空间上相邻的画面。

取回后，模型把三类 latent 拼在同一 self-attention 序列中：

- **memory latents**：远期场景、物体状态与视觉身份；
- **past-frame latents**：短期运动连续性；
- **current noisy latents**：当前要去噪的 segment。

相对 Plücker 几何编码描述记忆视角与当前视角的关系，原始 frame index 写入 temporal RoPE；不同注意力头对 RoPE base 做扰动，以降低远距离周期混叠。推理中还可以保留首个 latent 作为 persistent sink，稳定整体风格。

![](assets/Matrix-Game%203.0%20-%20长时记忆实时交互世界模型/figures/03-memory-model.png)

> **图 4：记忆增强基础模型。** Memory Frame、Past Frame 与 Noised Current Frame 共同进入同一个 Diffusion Transformer；左侧 error buffer 同时对长期与短期历史注入模型残差。统一注意力避免了单独记忆分支的接口割裂，但代价是序列和检索成本随时间增长。
> 来源：论文 Figure 4，§3.2，<https://arxiv.org/abs/2604.08995>

### 4.3 动作注入

论文将离散键盘信号与连续鼠标信号分开处理：键盘动作通过专门 cross-attention 注入，鼠标控制融入 self-attention 路径。这样做符合两类信号的统计特性：键盘是稀疏的符号事件，鼠标是连续位移/旋转。动作条件贯穿基础模型与蒸馏模型，避免“基础模型会控制、加速后控制失效”。

### 4.4 训练—推理对齐的多 segment DMD

单窗口双向学生面临一个难题：若只在一个窗口内训练，无法得到真实流式推理中“由学生自己生成、带偏差但相对干净”的历史；若使用 ground truth，则再次产生 exposure bias。

Matrix-Game 3.0 让学生连续 rollout 多个 segments。第 $i$ 段的历史来自第 $i-1$ 段学生输出，长期记忆也在线写入并检索。随机选择最后一个 segment $k$ 进行分布匹配，teacher 与 critic 比较学生分布并提供 DMD 梯度。其核心目标可理解为最小化学生分布 $p_\theta$ 对教师分布 $p_{\text{teacher}}$ 的 reverse-KL，梯度由真实/教师分数与 fake/critic 分数之差近似：

$$\nabla_\theta D_{\mathrm{KL}}(p_\theta\Vert p_{\text{teacher}})\approx\mathbb E\left[(s_{\mathrm{fake}}(x_t,t)-s_{\mathrm{teacher}}(x_t,t))\frac{\partial x_t}{\partial\theta}\right].$$

训练前 600 steps 用单 segment cold start，随后 2,400 steps 随机展开 1–6 个 segments。teacher、critic、student 都从 memory-augmented base model 初始化，降低异构师生映射的不稳定性。

![](assets/Matrix-Game%203.0%20-%20长时记忆实时交互世界模型/figures/04-dmd-distillation.png)

> **图 6：多 segment 自生成蒸馏。** 学生迭代生成并更新 memory pool，最后一个 segment 才送入教师与 critic 做分布匹配。训练时的历史由学生自己产生，因此比 ground-truth history 更接近真实部署。
> 来源：论文 Figure 6，§3.3，<https://arxiv.org/abs/2604.08995>

### 4.5 系统级实时加速

达到 40 FPS 不是单纯依赖 few-step diffusion，而是三类工程优化叠加：

- **INT8 DiT**：主要量化 attention projection，FFN、VAE 和 text encoder 保持较高精度；
- **MG-LightVAE**：对 decoder hidden dimensions 做 50% 或 75% 剪枝，配合 `torch.compile`；
- **GPU 相机记忆检索**：以空间采样近似视锥重叠，替代 CPU 上昂贵的精确体积相交；
- **异步并行**：8 张 GPU 跑 DiT，1 张 GPU 跑 VAE 解码。

这说明论文的“实时”是**系统属性**。few-step 模型只是必要条件，检索和解码若未同步优化，仍会成为瓶颈。

---

## 5. 实验结果

### 5.1 主要结果与比较边界

| 指标 | Matrix-Game 3.0 | 说明 |
|---|---:|---|
| 分辨率 | 720p | 5B 实时模型 |
| 最大吞吐 | 约 40 FPS | 异步 8+1 GPU |
| 连续生成 | 分钟级 | 主要以重访样例定性验证 |
| 基础模型 | 5B | Wan2.2-TI2V-5B 初始化 |
| 扩展模型 | MoE-28B / 2×14B | 质量提升以定性展示为主 |
| 记忆训练数据 | 约 4.8M clips | 5 memory + 4 past + 10 current |

论文相关工作提到 Genie 3 约 720p/24 FPS，Matrix-Game 2.0 与 HY-Gamecraft-2 已可实时，Lingbot-World 有更长上下文。但这些不是在相同硬件和数据上的受控比较，因此不能简单据此宣布“速度或质量全面领先”。

### 5.2 加速组件消融

论文 Table 1 在 75% VAE 剪枝配置下逐项移除组件：

| 配置 | FPS | 相比完整系统损失 |
|---|---:|---:|
| 完整系统 | 约 40.00 | — |
| 移除 INT8 | 27.38 | -12.62 |
| 移除 MG-LightVAE | 25.79 | -14.21 |
| 移除 GPU retrieval | 6.60 | -33.40 |

最反直觉也最重要的结论是：**记忆检索而不是 DiT 本身，可能成为长时实时系统的最大瓶颈**。没有 GPU 检索，吞吐从约 40 FPS 跌到 6.60 FPS。

### 5.3 VAE 剪枝质量—速度权衡

| VAE | PSNR ↑ | SSIM ↑ | 编码+解码时间 ↓ | 仅解码 ↓ |
|---|---:|---:|---:|---:|
| Wan2.2 VAE | 33.79 | 0.99 | 0.99 s | 0.76 s |
| MG-LightVAE 50% | 31.84 | 0.99 | 0.52 s | 0.30 s |
| MG-LightVAE 75% | 31.14 | 0.99 | 0.35 s | 0.13 s |

50% 剪枝约带来 2.6× decoder 加速，75% 剪枝约 5.2×，但 PSNR 分别下降 1.95 dB 和 2.65 dB。SSIM 四舍五入后都为 0.99，不能据此断言视觉差异可以忽略；高频纹理和小物体正是交互世界最容易积累误差的部分。

### 5.4 长时记忆与动作控制

论文通过“前半程离开、后半程反向动作返回”的视频展示场景重访：建筑、道路、物体布局和纹理能够在离开视野后恢复；蒸馏模型也保留部分记忆能力。28B 模型展示了更多户外、驾驶、骑马、夜景等场景。

但论文没有提供长时几何误差、重访匹配率、动作成功率、相机轨迹误差、FVD/VBench 等统一指标，也没有逐项消融 error buffer、Plücker 编码、sink latent、RoPE 扰动和 memory 数量。因此目前最可靠的量化结论是**系统吞吐与 VAE 权衡**，而不是长期世界一致性已经被充分测量。

---

## 6. 与前序工作的关系

### 6.1 相对 Matrix-Game 2.0

2.0 的中心是 Self-Forcing 式少步实时流式生成；3.0 把核心瓶颈转向“长期记忆与错误累积”。结构上也从 causal chunk/KV history 转向较大的双向 segment + overlap/history + 显式 memory retrieval。其变化并非放弃自回归，而是把自回归单位从小 chunk 扩展为 segment，并用外部 memory 补充跨段信息。

### 6.2 相对 Self-Forcing

Self-Forcing 的关键是训练时让学生看到自己的 rollout。Matrix-Game 3.0 继承这一思想，但适配到双向 segment：跨多个 segments 生成历史和 memory，再在最后一段做 DMD。它解决的是“单个双向窗口无法同时充当当前预测和自生成历史”的结构问题。

### 6.3 相对检索式世界模型

传统 long-context attention 把所有历史都保留在上下文中，成本随时间增长；检索式记忆只取空间相关帧，成本更可控。Matrix-Game 3.0 的关键选择是**以相机几何定义相关性**，而不是仅依赖语义 embedding。这更适合重访场景，但也高度依赖可靠相机位姿。

---

## 7. 局限、失败案例与批判性评价

### 7.1 论文明确承认的瓶颈

- DiT 优化后，VAE decoder 成为高分辨率流式生成的主要瓶颈；
- CPU 精确视锥重叠准确但昂贵，候选池会随时间增长；
- 75% VAE 剪枝以更明显的重建保真损失换速度；
- 最高 40 FPS 依赖 8+1 GPU，单节点 7+1 配置更慢；
- 未来仍需更高效架构、更长序列与更先进记忆机制。

### 7.2 证据缺口

论文没有独立 Limitations 章节。结合实验可以看到：

1. **长时记忆缺少指标**：分钟级重访主要靠视觉样例，尚不足以回答物体状态是否严格保持；
2. **动作控制缺少任务成功率**：键鼠输入和画面运动看起来一致，不等于细粒度控制可靠；
3. **关键模块缺少逐项消融**：error buffer、相机检索、几何编码与多 segment DMD 的独立贡献不清楚；
4. **硬件门槛极高**：8+1 GPU 的 40 FPS 更接近服务器级演示，而非普遍可部署能力；
5. **相机位姿依赖**：真实互联网视频中的姿态估计误差会污染记忆检索，论文未系统量化；
6. **世界状态仍是图像记忆**：保存历史 latent 并不能保证因果状态、任务规则与不可见变量被正确维护。

### 7.3 个人判断

Matrix-Game 3.0 最有价值的不是“40 FPS”这个宣传数字，而是揭示了交互世界模型的**瓶颈迁移**：当扩散步数被蒸馏、DiT 被量化后，检索和 VAE 会成为主瓶颈；当短段生成变稳定后，重访一致性和历史状态会成为主要矛盾。它提供了一条可部署系统路线，但还不能证明模型已经形成稳定的、可推理的“世界状态”。

---

## 8. 复现与实践建议

### 8.1 资源评估

- **数据**：需要带同步动作和相机位姿的大规模视频；普通公开视频不能直接替代；
- **基础模型**：Wan2.2-TI2V-5B 级 DiT，另有 28B 扩展；
- **训练**：基础模型、记忆后训练、teacher/critic/student DMD 均需大量显存；
- **部署**：论文最高性能采用 8+1 GPU；消费级单卡难以复现；
- **工程**：需维护在线 memory pool、相机检索、异步 DiT/VAE 流水线、INT8 kernel 与剪枝 VAE。

### 8.2 推荐复现顺序

1. 先在低分辨率单场景上复现 error buffer，验证长 rollout 是否少漂移；
2. 加入相机位姿检索，以 loop-closure/重访任务建立定量指标；
3. 再做多 segment DMD，比较 ground-truth history、single-segment 与 self-rollout；
4. 最后再做量化、VAE 剪枝和 GPU retrieval，避免系统优化掩盖模型问题。

### 8.3 应补充的评测

建议至少加入：动作响应准确率、重访图像/几何一致性、物体状态保持率、不同时间跨度的 memory recall、相机位姿噪声鲁棒性、单卡/多卡延迟与显存、检索池规模随时间增长的复杂度曲线。

---

## 9. 后续问题

1. 长期记忆是否应从“图像 latent 检索”升级为带对象和状态转移的结构化记忆？
2. 相机位姿未知或漂移时，能否联合学习可微 loop closure？
3. 多 segment DMD 能否与 Causal Forcing 的可注入性条件结合，减少条件期望式模糊？
4. 40 FPS 若降到单卡部署，应该优先牺牲分辨率、segment 长度还是记忆数量？
5. 如何设计一个同时衡量动作控制、物体永久性和重访几何的统一 benchmark？

---

## 10. 材料来源

- 论文页：<https://arxiv.org/abs/2604.08995>
- 官方 PDF：<https://arxiv.org/pdf/2604.08995>
- arXiv HTML：<https://arxiv.org/html/2604.08995v2>
- 项目页：<https://matrix-game-v3.github.io/>
- 团队仓库：<https://github.com/SkyworkAI/Matrix-Game>
- 本地来源清单：[[assets/Matrix-Game 3.0 - 长时记忆实时交互世界模型/sources]]
