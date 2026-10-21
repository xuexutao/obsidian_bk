---
type: paper-note
status: complete
domain: World-Models
paper: "Matrix-Game: Interactive World Foundation Model"
year: 2025
arxiv: "2506.18701"
doi: null
source: https://arxiv.org/abs/2506.18701
project: https://matrix-game-homepage.github.io/
code: https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-1
tags:
  - World-Models
  - Interactive-Video
  - Action-Control
  - Minecraft
  - Autoregressive-Generation
created: 2026-10-16
updated: 2026-10-16
---

# Matrix-Game 1.0：动作可控游戏世界基础模型

## 基本信息

- **作者**：Yifan Zhang、Chunli Peng、Boyang Wang、Puyi Wang、Qingcheng Zhu、Fei Kang、Biao Jiang、Zedong Gao、Eric Li、Yang Liu、Yahui Zhou；Skywork AI
- **版本**：arXiv:2506.18701v1，2025-06-23
- **模型**：17B 参数，基于 HunyuanVideo I2V 初始化的 MMDiT，720p、16 FPS 数据口径
- **任务**：给定单张参考图、最近运动帧与逐帧键鼠动作，生成动作可控的 Minecraft/Unreal 游戏视频
- **官方资源**：[论文](https://arxiv.org/abs/2506.18701) · [项目页](https://matrix-game-homepage.github.io/) · [代码](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-1) · [权重](https://huggingface.co/Skywork/Matrix-Game)

## 一句话结论

Matrix-Game 1.0 的关键贡献是用“2,700 小时无标签世界预训练 + 约 1,200 小时动作标注训练”把 17B 图生视频模型改造成能响应键盘与鼠标的游戏世界生成器，并用 GameWorld Score 将画质、时间质量、控制准确率和物理一致性放入同一评测；但它仍以 50 步、分段自回归离线生成，只有 5 帧局部历史、没有自生成分布对齐或长期记忆，因此后续 2.0 必须解决实时流式与误差累积，3.0 又进一步解决重访与世界持续存在。

---

## 1. 研究背景与问题定义

### 1.1 从“生成游戏视频”到“动作改变未来”

普通 image-to-video 模型接受参考图和文本后一次性生成固定片段，用户无法在生成过程中改变方向、视角或行为。交互世界模型的基本要求则是条件分布必须显式依赖动作：同一初始画面下，按前进、后退、跳跃或转动相机应产生不同且合理的后续观察。

Matrix-Game 1.0 将问题限定在第一人称游戏视频：不恢复 mesh，也不运行显式游戏引擎，而是直接学习从视觉历史与动作到未来像素的映射。这使规模化训练成为可能，却也意味着碰撞、物体持久性和游戏规则都隐式存在于生成模型参数中，无法像传统引擎那样查询或严格验证。

### 1.2 三个核心难点

1. **动作监督稀缺**：互联网游戏视频规模大，但缺少逐帧键鼠记录；脚本采集则容易动作单一、场景失衡。
2. **视觉质量与控制耦合**：大视频模型擅长生成好看的片段，却可能忽略低维动作条件；控制模块必须与时空 latent 对齐。
3. **长视频误差累积**：固定长度扩散片段只能靠上一段末帧继续生成，一旦末帧出现伪影，错误会被后续片段放大。

### 1.3 论文贡献

- 构建 Matrix-Game-MC：由 6,000 小时原始视频筛成 2,700 小时无标签 720p 数据和 870 小时高质量子集，并以 MineRL/VPT 与 Unreal 仿真构建超过 1,000 小时动作标注数据。
- 提出纯视觉 image-to-world 路线：去掉文本分支，以单张参考图、运动上下文和动作控制 MMDiT。
- 将连续鼠标与离散键盘分别注入 latent/self-attention 和 cross-attention，支持细粒度动作控制。
- 提出 GameWorld Score，统一评估视觉、时间、动作和物理一致性。

---

## 2. 任务定义与输入输出

### 2.1 条件与输出

第 $n$ 个视频片段可概括为

$$V_n=G_\theta(I_{ref},M_{n-1},A_n,\epsilon_n),$$

其中 $I_{ref}$ 是固定参考图，$M_{n-1}$ 是上一片段最后 $k=5$ 帧，$A_n$ 是当前片段逐帧键鼠动作，$\epsilon_n$ 是 Gaussian noise。输出 $V_n$ 是 720p 固定长度视频片段；连续视频通过重复调用 $G_\theta$ 拼接得到。

动作空间包括六种离散键盘动作——forward/backward/left/right/jump/attack——以及由 pitch/yaw 变化描述的连续鼠标相机动作。论文评测时将相机运动离散成八个方向及 empty 类。

### 2.2 任务边界

模型输出的是视频而非可执行状态：没有显式位置、速度、碰撞体、库存、地图或对象槽。Unreal 数据虽包含位置、速度、朝向和交互成功状态，论文的生成模型仍主要以视频 latent 和动作作为条件。因此“理解物理”在本文中实质上是通过像素/几何一致性代理指标衡量，而不是证明学得了可查询的物理模拟器。

---

## 3. 数据：先学世界外观，再学动作响应

### 3.1 无标签数据过滤

![](assets/Matrix-Game%201.0%20-%20动作可控游戏世界基础模型/figures/01-data-pipeline.png)

> **Figure 3：无标签数据过滤。** 6,000 小时原始 gameplay 经过质量与美学过滤得到 2,700 小时 720p clips，再经菜单、字幕、人脸、运动与相机运动过滤得到 870 小时高质量子集。该图清楚显示 870 小时不是独立新增数据，而是 2,700 小时中的精选子集。  
> 来源：论文 Figure 3，PDF p.6，arXiv LaTeX 源码 `asset/data.pdf`。

原始 MineDojo 视频先由 TransNet V2 切镜、FFmpeg 转码，每段首尾各丢弃 4 帧；随后使用 DOVER、LAION aesthetic、IDM、CRAFT、DeepFace、GMFlow 和相机角速度过滤器。过滤不仅追求画质，也主动移除静止、急转与界面状态，使训练分布更容易建模。

这带来一个重要偏差：模型在稳定相机与活跃 gameplay 上学得更好，却可能对真实玩家突然转头、停顿、菜单切换和极端速度更脆弱。后续失败案例中的边缘场景退化，与这种“为了稳定而收窄训练分布”的选择一致。

### 3.2 动作标注数据

动作数据来自两条互补管线：

- **MineRL exploration agent**：课程式 VPT agent 以 16 Hz 探索 Minecraft，导出逐帧键盘与鼠标动作；每帧 yaw/pitch 被限制在 $15^\circ$ 内，并禁用视锥触发的 chunk loading 以避免地形突然弹出。
- **Unreal procedural simulation**：在城市、沙漠、森林等可编程环境中同步提供动作、gaze vector、位置、速度、朝向和方块交互成败，得到噪声更低的监督。

论文有三种相近口径：摘要写“超过 1,000 小时高质量动作标注”，数据章节写 33 帧训练集超过 1,026 小时、65 帧平衡集超过 1,200 小时，训练章节写 1,200 小时动作标注 33 帧数据。最稳妥的理解是数据在不同训练阶段继续扩充，论文没有解释 1,026 与 1,200 小时之间的精确版本关系。

### 3.3 两阶段训练为何必要

Stage 1 不加动作模块，先在 2,700 小时视频上学习空间布局、对象动态和视觉先验，再在 870 小时高质量子集上精炼。Stage 2 才加入动作控制模块，以约 1,200 小时标注数据训练 33 帧设置，随后在覆盖 8 个重点 biome 的 65 帧平衡集上继续训练。

这是一种标签效率设计：昂贵的动作标注只负责“动作—结果”映射，世界外观和一般动态尽量从无标签视频中吸收。遗憾的是论文没有消融“只用标注数据”或“去掉 Stage 1”，因此两阶段收益虽合理，却没有被受控实验定量隔离。

---

## 4. 核心方法

### 4.1 Image-to-world 总体框架

![](assets/Matrix-Game%201.0%20-%20动作可控游戏世界基础模型/figures/02-overview.png)

> **Figure 4：image-to-world 总体流程。** 视频经 3D Causal VAE 压到时空 latent，参考图经 visual encoder 形成图像 token，MMDiT 联合噪声、参考图与交互动作预测视频 latent，再由 VAE 解码。  
> 来源：论文 Figure 4，PDF p.8，arXiv LaTeX 源码 `asset/Pipeline2.pdf`。

3D Causal VAE 的空间压缩率为 8、时间压缩率为 4。主干从 HunyuanVideo I2V 初始化，但将文本分支替换为图像分支；作者的动机是减少语言先验对空间理解的干扰，使模型更多依赖视觉与物理线索。该设计没有“保留文本分支”的消融，因此应视为论文的建模主张，而非被实验完全证明的因果结论。

训练使用 rectified flow。若干净 latent 为 $x_0$、噪声为 $x_1\sim\mathcal N(0,I)$，线性路径可写为

$$x_t=(1-t)x_0+tx_1,\qquad v^*(x_t,t)=x_1-x_0,$$

典型 flow-matching 目标为

$$\mathcal L_{RF}=\mathbb E_{x_0,x_1,t}\left[\lVert v_\theta(x_t,t\mid I_{ref},M,A)-v^*(x_t,t)\rVert_2^2\right].$$

论文只明确说明采用 rectified flow loss，未展开显示公式；上式用于解释其标准机制，不应误读为论文给出的独有损失。

### 4.2 分段自回归与运动上下文

![](assets/Matrix-Game%201.0%20-%20动作可控游戏世界基础模型/figures/03-architecture.png)

> **Figure 5：自回归与模型架构。** 左侧显示上一片段末帧、当前噪声和参考图共同生成下一片段；右侧显示 motion latent 与 noisy latent 按通道拼接并附加 binary mask，图像 token 则进入 MMDiT 双流/单流 blocks。  
> 来源：论文 Figure 5，PDF p.9，arXiv LaTeX 源码 `asset/architecture_try.pdf`。

上一片段最后 5 帧经 VAE 编码，与当前 noisy latent 在通道维拼接；binary mask 标记哪些位置有有效运动信息。patch embedding 后，视频 token 与参考图 token 一起进入 MMDiT。模型还沿用 HunyuanVideo I2V 的 token replacement trick 稳定首帧/参考图约束。

为缓解推理时历史不完美而训练时历史过于干净的分布差异，作者做了两种扰动：

- 以 0.2 概率给 motion frames 与 reference image 加 Gaussian noise；
- 以 0.25 概率把 motion-frame latent 替换为 zero latent，实施 motion CFG dropout。

这些策略提高模型对受损局部历史的鲁棒性，但训练仍主要使用真实视频历史，不等同于 2.0 的 Self-Forcing：1.0 并没有在完整 rollout 上让学生持续条件于自己的生成结果。

### 4.3 异构动作注入

键盘是离散符号，嵌入后通过 cross-attention 注入；鼠标是连续角度，先经过 MLP 与 temporal self-attention，再与输入 latent 拼接。由于 3D VAE 将时间压缩 4 倍，动作序列通过 group operation trick 与 latent token 对齐。训练时以 0.1 概率丢弃动作条件，使 CFG 能在推理时增强动作响应。

异构注入是本文最可迁移的结构经验：离散动作适合作为可查询条件，连续动作适合进入时序表征。后续 Matrix-Game 2.0/3.0 延续了键盘 cross-attention 与鼠标 latent/self-attention 的分工，说明 1.0 真正确立的是系列的“动作接口”。

### 4.4 训练与推理配置

| 项目 | 论文设置 |
|---|---|
| 参数量 | 17B |
| 初始化 | HunyuanVideo I2V |
| 分辨率 | 720p |
| 训练 FPS | 16 |
| 每 GPU batch | 1 |
| 数值/并行 | bf16，FSDP |
| 学习率 | $5\times10^{-5}$ |
| motion frames | 5 |
| 推理 | 50 steps，CFG 6，flow shift 15 |

论文未披露 GPU 数、训练总步数、optimizer/betas、总 GPU-hour 与 wall-clock。这使方法可以概念复现，但无法核对训练预算或完整重训成本。

---

## 5. GameWorld Score：世界模型评测的第一次系统化

### 5.1 四个维度、八项指标

1. **视觉质量**：MUSIQ image quality 与 LAION aesthetic。
2. **时间质量**：相邻帧 CLIP cosine similarity；插帧网络重建误差代理 motion smoothness。
3. **动作可控性**：在 1,962 小时 Minecraft 上训练的 IDM 反推动作，与输入动作比较。
4. **物理规则理解**：DROID-SLAM 共视像素重投影误差衡量 object consistency；对称往返相机轨迹的配对帧误差衡量 scenario consistency。

相邻帧语义一致性可写为

$$TC=\frac{1}{T-1}\sum_{t=1}^{T-1}\cos\left(f_{CLIP}(I_t),f_{CLIP}(I_{t+1})\right).$$

Keyboard Accuracy 将 forward/back/empty、left/right/empty、attack/empty、jump/empty 分为四组并平均 precision：

$$KAcc=\frac14\sum_{g=1}^{4}\operatorname{Precision}_g.$$

### 5.2 指标的歧义与边界

论文正文说明 object reprojection error 与 scenario MSE 越低越好，但主表将 Object/Scenario Consistency 标为 $\uparrow$ 并报告归一化分数。论文没有说明误差如何转换为一致性分数，这是 benchmark 复现时必须检查的实现细节。

此外，高 temporal consistency 可能奖励“几乎不动”，IDM accuracy 依赖代理模型在生成域上的可靠性，DROID-SLAM 在严重生成伪影下也可能失效。GameWorld Score 比只看 FVD 更接近交互需求，但仍不是游戏任务成功率或可规划性评测。

---

## 6. 实验结果

### 6.1 总体定量

| 模型 | IQ↑ | Aesthetic↑ | Temporal↑ | Smooth↑ | Keyboard↑ | Mouse↑ | Object↑ | Scenario↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OASIS | 0.65 | 0.48 | 0.94 | **0.98** | 0.77 | 0.56 | 0.56 | 0.86 |
| MineWorld | 0.69 | 0.47 | 0.95 | **0.98** | 0.86 | 0.64 | 0.51 | 0.92 |
| **Matrix-Game** | **0.72** | **0.49** | **0.97** | **0.98** | **0.95** | **0.95** | **0.76** | **0.93** |

最显著提升在鼠标控制（0.95 对 0.56/0.64）和对象一致性（0.76 对 0.56/0.51），说明动作标注规模与连续鼠标注入确实对应论文的核心目标。视觉指标增益较小，证明工作重点并非单纯追求更漂亮的视频。

双盲人工评测中，Matrix-Game 的胜率为 Overall 96.30%、Controllability 93.76%、Visual Quality 98.23%、Temporal Consistency 89.56%。论文未披露样本数、annotator 数、置信区间和统计显著性，因而这些极高胜率适合支持“偏好明显”，不宜当作精确总体效应量。

### 6.2 逐动作控制

Matrix-Game 的 forward/backward/left/right/jump/attack precision 分别为 0.99/0.91/0.92/0.96/0.88/0.95；八个相机方向均不低于 0.89。论文正文称 jump 为 95%，但 Table 3 中 jump 是 0.88、attack 才是 0.95，属于原文内部不一致，本文以表格为准。

### 6.3 场景与长视频

模型在 beach、desert、forest、hills、icy、mushroom、plains、river 八类场景的雷达图中整体优于 OASIS 与 MineWorld。长视频实验展示三个连续动作片段能维持边界平滑，但没有报告序列总长度上的指标曲线，也没有把误差随 segment 数增长量化。因此实验证明的是“局部片段可连续扩展”，不是无限时域稳定。

### 6.4 消融证据缺口

论文没有正式内部消融表。以下关键问题均未被逐项验证：

- 2,700 小时无标签预训练贡献多大；
- 870 小时精炼是否优于直接继续用全量数据；
- 5 个 motion frames 是否最优；
- 0.2 噪声、0.25 motion dropout、0.1 action dropout 各自贡献；
- 33→65 帧课程、Minecraft/Unreal 混合的独立效果；
- 键盘 cross-attention 与鼠标 concat 是否优于统一注入。

因此本文最强证据是“完整系统优于公开 baseline”，而不是各设计均已被严格证实。OASIS/MineWorld 比较是 baseline 实验，不应误称为消融。

---

## 7. 失败案例与局限

![](assets/Matrix-Game%201.0%20-%20动作可控游戏世界基础模型/figures/04-failure-cases.png)

> **Figure 11：失败案例。** 上排在罕见/欠覆盖场景中逐步失去时间与空间一致性；下排角色直接穿过树叶，暴露碰撞和材质交互并未被可靠建模。  
> 来源：论文 Figure 11，PDF p.18，arXiv LaTeX 源码 `failure_case1_new.pdf` 与 `failure_case2_new.pdf`。

### 7.1 作者明确承认的限制

- 罕见或复杂场景因数据覆盖不足而出现控制与空间一致性退化；
- 碰撞、支撑、材质和地形穿越等物理规则仍不可靠；
- 5 帧 motion context 只能维持局部连续，长时误差仍会逐段传播；
- 当前动作空间只有六类键盘动作和有限鼠标范围；
- Minecraft 是简化沙盒，尚未验证复杂写实游戏与多智能体环境。

Future Work 对应为持续扩充数据与 continual training、增加 physics-aware supervision、使用更长上下文或 memory mechanism、扩展连续动作空间，并迁移到 Black Myth: Wukong、赛车与 CS:GO 等环境。

### 7.2 我的判断

第一，1.0 的 17B/50-step 配置解决了“能不能控制”，没有解决“能不能及时响应”。单次 65 帧推理要求至少 80 GB A100/H100，无法构成低延迟交互循环。第二，给历史加随机噪声不是让模型见到自身结构化错误；曝光偏差仍在。第三，5 帧运动上下文既不保存远期位置，也不支持重访，所谓 world foundation model 更准确地说是动作条件视频生成基础模型。

---

## 8. 系列迭代：为什么 2.0 与 3.0 必然出现

### 8.1 1.0 → 2.0：动作控制之后，瓶颈转为实时流式

[[Matrix-Game 2.0 - 实时流式交互世界模型|Matrix-Game 2.0]] 保留 1.0 确立的键鼠动作接口，但将 17B/50-step 双向片段生成改为 1.8B/3-step 因果模型，加入固定滚动 KV cache、ODE 初始化与 Self-Forcing/DMD，在单张 H100 上报告约 25 FPS。

这不是简单压缩模型：1.0 训练看真实历史、推理看生成历史，结构化伪影会累积；2.0 让学生在自己的 rollout 上训练，直接处理 exposure bias。瓶颈由“动作是否可控”转成“动作到画面是否足够快、生成历史是否可用”。

### 8.2 2.0 → 3.0：实时之后，瓶颈转为长期记忆

2.0 的固定 KV window 让每步成本近似常数，却主动遗忘窗口外世界；更大的 cache 还可能保存更多伪影。[[Matrix-Game 3.0 - 长时记忆实时交互世界模型|Matrix-Game 3.0]] 因此加入相机视锥检索的长期 memory、短期 past frames、persistent sink 和 error buffer，并以多 segment self-rollout DMD 对齐流式推理。

系列演化可概括为：

| 版本 | 首要问题 | 核心解法 | 新暴露瓶颈 |
|---|---|---|---|
| 1.0 | 动作能否控制生成世界 | 大规模两阶段数据 + 异构键鼠注入 | 50 步、非实时、真实/生成历史错配 |
| 2.0 | 能否实时流式持续生成 | 因果化、KV cache、Self-Forcing、3 步 DMD | 窗口外遗忘、回访不一致 |
| 3.0 | 世界能否长期持续存在 | 几何检索记忆、error buffer、多段蒸馏 | 检索/VAE 成本、状态仍主要是视觉记忆 |

关键不是前一代问题被“彻底解决”，而是工程可行后主要矛盾发生转移：动作控制使交互成为可能，实时流式使交互可用，长期记忆才使生成视频接近持续存在的世界。

---

## 9. 开源与复现

### 9.1 已公开

官方 1.0 目录提供模型实现、推理脚本、双卡/并行推理、配置、GameWorld Score 和模型权重，仓库声明 MIT License。README 要求 A100/H100，生成单个 65 帧视频至少 80 GB VRAM。

### 9.2 未公开或披露不足

未找到完整训练脚本、训练集下载、MineRL/Unreal 全数据生产流水线、确切数据快照、GPU 数与训练时长。因此可以复现官方权重推理和部分评测，但无法仅依赖公开材料严格复现 17B 模型训练。

### 9.3 建议复现实验

1. 先在固定参考图上复现六类键盘与八方向相机动作，检查 IDM 指标是否与官方一致。
2. 使用 no-op/相反动作/往返动作建立反事实测试，避免仅看精选视频。
3. 按 segment 记录 IQ、控制准确率和回环误差，画出随时间退化曲线。
4. 对比真实历史、随机加噪历史和模型自生成历史，直接测量 1.0 的 exposure bias。
5. 若训练资源有限，优先复现动作模块，而非尝试从零重训 17B 基础模型。

---

## 10. 材料来源

本地官方 PDF、4 张经预览核验的 arXiv 源码关键图和开源状态记录见 [sources.md](assets/Matrix-Game%201.0%20-%20动作可控游戏世界基础模型/sources.md)。
