---
type: paper-note
status: complete
domain: World-Models
paper: V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
year: 2025
arxiv: 2506.09985
doi: null
source: https://arxiv.org/abs/2506.09985
project: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
code: https://github.com/facebookresearch/vjepa2
tags: [World-Models, JEPA, V-JEPA, Self-Supervised, Video-Representation, World-Model, Robot-Planning, Action-Conditioned]
created: 2026-09-03
updated: 2026-09-03
---

# V-JEPA 2：自监督视频世界模型的理解、预测与规划

## 基本信息

- **作者**：Mahmoud Assran*、Adrien Bardes*、David Fan*、Quentin Garrido*、Russell Howes*、Mojtaba Komeili*、Matthew Muckley*、Ammar Rizvi*、Claire Roberts*、Koustuv Sinha*、Artem Zholus*、Sergio Arnaud*、Abha Gejji*、Ada Martin*、Francois Robert Hogan*、Daniel Dugas、Piotr Bojanowski、Vasil Khalidov、Patrick Labatut、Francisco Massa、Marc Szafraniec、Kapil Krishnakumar、Yong Li、Xiaodong Ma、Sarath Chandar、Franziska Meier*、Yann LeCun*、Michael Rabbat*、Nicolas Ballas*（* 核心团队，通讯作者 Nicolas Ballas、Michael Rabbat）
- **机构**：FAIR at Meta；Mila – Quebec AI Institute；Polytechnique Montréal
- **发表**：arXiv:2506.09985（v1：2025-06-11）
- **论文正式名**：*V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*
- **官方代码**：<https://github.com/facebookresearch/vjepa2>（MIT 许可）
- **官方博客**：<https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks>
- **概念源头**：LeCun 2022 JEPA 纲领 → [I-JEPA](arXiv:2301.08243)（图像版，CVPR 2023）→ [V-JEPA](arXiv:2404.08471)（视频版）

## 一句话结论

V-JEPA 2 把 V-JEPA scale up 到 1.2B 参数（ViT-g encoder + narrow transformer predictor），用 **100 万小时互联网视频 + 100 万图像** 做 visual mask denoising 预训练，在 Something-Something v2（77.3% top-1）与 Epic-Kitchens-100 action anticipation（39.7 R@5）上 SOTA；后训练一个动作条件世界模型 **V-JEPA 2-AC**（<62 小时 Droid 无标签机器人数据），用 MPC 在两个**未见过的实验室 Franka 双臂**上零样本完成抓取-放置任务——首次证明「互联网视频自监督 + 小量机器人数据」能产出可零样本规划的物理世界模型，且物理推理比 NVIDIA Cosmos 快约 30 倍。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

LeCun 2022 纲领认为，**真正的世界模型应当能回答「执行动作序列之后世界会怎么变」**——也就是 action-conditioned 的预测。这一纲领有两类实践路径，但都有根本局限：

- **从 interaction data 学**（state-action + reward，Dreamer / TD-MPC / IRIS 等）：依赖昂贵的人工采集的机器人数据，scalability 受限；
- **从互联网视频 + 交互数据学视频生成世界模型**（Cosmos / Sora / GAIA-2）：数据规模容易做大，但下游规划需要「想象未来」再做动作反推动作——每次规划都要生成视频，计算成本极高，且文献上多评估预测的视觉质量而非规划能力。

V-JEPA 2 的核心问题是：**能否用 JEPA 的「潜空间预测」路线，把世界模型做到 (i) 只靠互联网视频学、(ii) 推理时不需要生成像素、(iii) 仍能驱动机器人规划**？它正面回答：「能」——而且比生成式路线快约 30 倍。

### 1.2 现有方法的瓶颈

- **生成式视频世界模型（Cosmos、Sora）**：做 planning 时每步都要做一次视频扩散去噪，规划速度极慢（单步数百毫秒到秒级），无法做到机器人实时控制；
- **从 interaction data 学的世界模型（DreamerV3）**：受限于昂贵的真实机器人采集数据（数千小时），难以泛化到新 embodiment；
- **V-JEPA 1（300M ViT-L，2M 视频）**：只在 frozen 协议下做视频表征，没接动作，没做规划；
- **DINOv2、I-JEPA 等图像自监督模型迁移**：缺运动理解，在 SSv2 等细粒度任务上落后 V-JEPA 10+ 个百分点（见 [V-JEPA 笔记](V-JEPA%20-%20视频潜空间预测表征学习.md)）。

### 1.3 本文核心贡献

1. **V-JEPA 2 视频编码器**：把 V-JEPA 从 300M 参数 / 2M 视频 / 90K iter，scale 到 **1.2B 参数 / 22M 视频（VM22M，1M+ 小时）/ 252K iter**，用 3D RoPE + warmup-constant-decay LR schedule 等关键工程改动，在 6 个视频任务（SSv2、Diving-48、Jester、Kinetics、COIN、ImageNet）平均准确率从 84.2 升到 88.2。
2. **Scaling Ingredients**：明确拆解出 4 个 scaling 因子，每个贡献 0.7-1.5pt 的下游性能提升：(1) **Data scaling** (2M→22M) +1.0pt；(2) **Model scaling** (ViT-L→ViT-g) +1.5pt；(3) **Long training** (90K→252K) +0.8pt；(4) **Higher resolution** (256→384, 16→64 frames) +0.7pt。
3. **V-JEPA 2-AC**：冻结 V-JEPA 2 encoder，训练一个 action-conditioned predictor（block-causal attention），用 **<62 小时 Droid 无标签机器人视频** 训练，做出能在两个**未见过的实验室 Franka 双臂**上零样本完成 pick-and-place 的物理世界模型。这是公开文献里第一次「用互联网视频 + 小量机器人数据」零样本规划的真实机器人演示。
4. **LLM 对齐的视频问答**：把 V-JEPA 2 表征与 LLM（8B 参数）做 language alignment，在 PerceptionTest（84.0）、TempCompass（76.9）等视频问答任务上达到 SOTA。
5. **30× 速度优势**：在物理推理任务上比 NVIDIA Cosmos 快约 30 倍（因为不需要做视频去噪）。

### 1.4 与 V-JEPA 的关系

V-JEPA 2 与 V-JEPA 是同一谱系的 scale-up：方法论完全一致（潜空间预测 + L1 + EMA + stop-grad），关键变化是 (i) scale up 到 1B+ 参数 + 22M 视频 + 长训练 + 高分辨率；(ii) 加入 3D RoPE 位置编码替代绝对 sin-cos（提升大模型训练稳定性）；(iii) 接入动作条件做 V-JEPA 2-AC。

---

## 2. 任务定义与输入输出

### 2.1 两阶段训练

V-JEPA 2 训练分为**两阶段**：

- **Stage 1：V-JEPA 2 视频预训练**：输入是 internet videos（无标注），目标是 mask denoising in representation space。
- **Stage 2：V-JEPA 2-AC 后训练**：冻结 Stage 1 encoder，加 trainable action-conditioned predictor，输入是「(past frames, actions, end-effector states)」三元组，目标是预测未来帧的表征。

### 2.2 V-JEPA 2 预训练任务

- **输入**：来自 VM22M 的 video clip（约 64 帧 × 384×384，patch 化后 token 数巨大），覆盖 1M+ 小时视频。
- **输出**：encoder $E_\theta$ 输出每个时空 patch 的表征。
- **训练目标**：

$$\min_{\theta,\phi,\Delta_y} \;\big\lVert P_\phi\big(E_\theta(x), \Delta_y\big) \;-\; \mathrm{sg}\big(E_{\bar\theta}(y)\big) \big\rVert_1$$

与 V-JEPA 完全一致（见 [V-JEPA 笔记](V-JEPA%20-%20视频潜空间预测表征学习.md) 第 2.2 节），差异是 $x, y$ 来自长视频 clip 且 patch 数大幅增加。

### 2.3 V-JEPA 2-AC 后训练任务

- **输入**：$(o_1, o_2, ..., o_K, a_1, a_2, ..., a_{K-1}, e_1, e_2, ..., e_K)$——$o_t$ 是 past 帧的 encoder 表征，$a_t$ 是 action，$e_t$ 是 end-effector state。
- **输出**：未来帧 $o_{K+1}, ..., o_{K+H}$ 的表征预测（$H$ 是预测视界）。
- **目标**：预测表征与未来真实帧的 encoder 表征做 L1 距离。

### 2.4 推理 / MPC 规划任务

- **输入**：当前观测 $o_t$、目标图像 $o_g$（goal image）。
- **输出**：action 序列 $a_t, a_{t+1}, ..., a_{t+H-1}$，最大化 $o_t$ 在执行后趋近 $o_g$。
- **方法**：Model Predictive Control（MPC）在 latent space 中搜索动作序列。

---

## 3. 核心方法

### 3.1 总体框架

V-JEPA 2 的总体框架见下图：

![](assets/V-JEPA%202%20-%20自监督视频世界模型的理解预测与规划/figures/01-overview.png)

> **图 1（论文 Figure 1）：V-JEPA 2 Overview。** 流程方向（左→右）：① 1M 小时互联网视频 + 1M 图像做 Video Pretraining，训出 V-JEPA 2 视频 encoder；② encoder 输出可走三条下游路径——(a) Language Alignment：与 LLM（8B）拼接做 Video QA（PerceptionTest 84.0、TempCompass 76.9）；(b) Attentive Probe Training：做 Understanding & Prediction 任务（动作分类 SSv2 77.3、物体识别、动作预测 Epic-Kitchens 100 39.7 R@5）；(c) Action-Conditioned Post-Training：用 62 小时 Robot Data（states + actions）训出 V-JEPA 2-AC，做 Planning 做机器人操作。
> 来源：论文 Figure 1，第 2 页，<https://arxiv.org/abs/2506.09985>

### 3.2 多阶段训练细节

两阶段训练的具体架构差异：

![](assets/V-JEPA%202%20-%20自监督视频世界模型的理解预测与规划/figures/02-multistage-training.png)

> **图 2（论文 Figure 2）：Multistage training。** **左：V-JEPA 2 视频预训练**——把 masked video frames 喂给 encoder $E_\theta$，unmasked video frames 喂给 EMA encoder $E_{\bar\theta}$；mask tokens（含位置编码）拼接到 encoder 输出后送入 predictor $P_\phi$；L1 loss 回归到 stop-grad target。**右：V-JEPA 2-AC 动作条件后训练**——冻结 encoder，加新训 action-conditioned predictor；输入是 previous video frames + actions + end-effector poses；输出预测 future video frame 的表征；用 block-causal attention，每个时间步的 patch 只能 attend 当前与过去的 patch、动作、end-effector 状态。
> 来源：论文 Figure 2，第 4 页，<https://arxiv.org/abs/2506.09985>

### 3.3 预训练细节

#### 3.3.1 数据：VM22M（VideoMix22M）

- **来源**：在 V-JEPA 的 VideoMix2M 基础上扩到 **22M videos**，再加约 **1M images**（DINOv2 风格的网络图像），总训练数据约 **1M+ hours 视频 + 1M images**。
- **混合源**：HowTo100M、Kinetics-400/600/700、Something-Something-v2，以及其他公开视频数据集。

#### 3.3.2 模型架构

- **Encoder $E_\theta$**：**ViT-g/16**（~1.2B 参数），无 [cls] token，patch 大小 16×16；
- **Predictor $P_\phi$**：narrow ViT，depth 12，embedding dim 384，self-attention 头数与 encoder 一致；
- **时空 patch 化**：训练时 384×384 分辨率，时空 patch = $2 \times 16 \times 16$（2 帧 × 16×16 像素），16 frames 时长 token grid 是 $(16/2) \times (384/16)^2 = 8 \times 576$ 个 token；更高分辨率 / 更长 clip 时 token 数增加。

#### 3.3.3 3D RoPE 位置编码

论文 2.1 节明确把 1D RoPE（Su et al. 2024）扩展到 3D：

- 把 embedding dim 切成 3 段（约相等），分别对应 **temporal / height / width** 三个轴；
- 每个轴内部独立做 1D RoPE；
- 这种 3D RoPE 替代 V-JEPA 用的绝对 sin-cos 位置编码，**对大模型 + 长视频训练更稳定**（论文 5.1 节）。

#### 3.3.4 Mask 策略

与 V-JEPA 一致：**时空多块掩码**（短程 15% × 8 + 长程 70% × 2，平均 mask ratio ~90%）。

#### 3.3.5 优化器与 LR schedule

- **Optimizer**：AdamW（标准 weight decay 0.04）；
- **LR schedule**：**warmup-constant-decay**——比 cosine 更利于长训练稳定（论文指出 warmup-cosine-decay 在大模型 + 长训练时不稳定，constant-decay 更简单可控）；
- **训练时长**：252K iterations（约 90K → 252K，2.8×）。

### 3.4 Scaling Ingredients（核心消融）

这是 V-JEPA 2 最值得单独看的部分——它**显式拆解了 4 个 scaling 因子**，每个因子贡献 0.7-1.5pt 下游性能：

![](assets/V-JEPA%202%20-%20自监督视频世界模型的理解预测与规划/figures/03-scaling-ingredients.png)

> **图 3（论文 Figure 3）：Scaling Ingredients 柱状图。** V-JEPA 2 预训练 recipe 的 5 个增量步骤：
> - **VJEPA 1 baseline**：84.2（V-JEPA 1，300M 参数，2M videos）
> - **+Data Scaling**：85.2（数据 2M→22M，+1.0pt）
> - **+Model Scaling**：86.7（ViT-L→ViT-g，+1.5pt）
> - **+Long Training**：87.5（90K→252K iter，+0.8pt）
> - **+Higher Resolution**：88.2（256→384 空间 + 16→64 时长，+0.7pt）
> 评测协议：在 6 个 motion + appearance 任务（SSv2、Diving-48、Jester、Kinetics、COIN、ImageNet）上跑 attentive probe 平均准确率。
> 来源：论文 Figure 3，第 5 页，<https://arxiv.org/abs/2506.09985>

论文 2.2 节明确给出这 4 个关键改动：

1. **Data scaling**：从 2M → 22M videos（VM22M），1.0pt 提升。证明视频自监督在 billion 级别视频数据上仍能拿到 scaling gain。
2. **Model scaling**：从 300M → 1B+ 参数（ViT-L → ViT-g），1.5pt 提升。这是最大的单一因子——证明 JEPA 架构可以 scale 到 LLM 级别的容量。
3. **Long training**：从 90K → 252K iters（warmup-constant-decay），0.8pt 提升。证明长训练 + 大模型是兼容的。
4. **Higher resolution**：训练时从 256→384 空间分辨率、16→64 帧，0.7pt 提升。证明时空双向的分辨率扩展都有效。

这 4 个因子的总提升是 **84.2 → 88.2 = +4.0pt**，证明 V-JEPA 的潜空间预测原则对 scale up 是鲁棒的。

### 3.5 V-JEPA 2-AC：动作条件世界模型

V-JEPA 2-AC 是 V-JEPA 2 真正面向机器人规划的扩展，论文 Section 3 + Section 4 给出完整方法：

#### 3.5.1 设计动机

V-JEPA 2 视频 encoder 虽然学到强时空表征，但它**不含任何动作 / 控制信息**——给定当前帧只能预测「未来会自然演化成什么样」，不能回答「执行动作 X 后会变成什么样」。要让模型成为真正的「world model」，必须把动作条件引入。

#### 3.5.2 架构

- **冻结** V-JEPA 2 encoder $E_\theta$（参数不变）；
- **新增** trainable action-conditioned predictor $P_\phi^{AC}$；
- $P_\phi^{AC}$ 接收：(i) 当前与过去帧的 encoder 表征 $\{E_\theta(o_t)\}_{t=1}^K$；(ii) actions $\{a_t\}_{t=1}^{K-1}$；(iii) end-effector states $\{e_t\}_{t=1}^K$；
- 输出：未来 $H$ 帧的表征预测 $\{E_\theta(o_{K+h})\}_{h=1}^H$。

#### 3.5.3 Block-Causal Attention

$P_\phi^{AC}$ 用 **block-causal attention**：

- 每个时间步的 patch 只能 attend 到「当前与过去时间步」的 patch、actions、end-effector states；
- 不能偷看未来（否则是 cheating）；
- 类似 autoregressive language model 的 causal mask，但以「时间步」为基本单位。

#### 3.5.4 训练目标

$$\min_\phi \; \big\lVert P_\phi^{AC}\big(\{E_\theta(o_t), a_t, e_t\}_{t=1}^{K}\big) \;-\; \mathrm{sg}\big(\{E_\theta(o_{K+h})\}_{h=1}^H\big) \big\rVert_1$$

与 V-JEPA 2 一致的 L1 + stop-grad，target 由 frozen encoder 产生。

#### 3.5.5 训练数据

- **数据集**：**Droid**（公开的 Franka Panda 真实机器人数据集）；
- **规模**：**<62 hours** 无标注机器人视频（论文特别强调「less than 62 hours」——这是相对整个 V-JEPA 2 训练 1M+ hours 视频数据的零头）；
- **样本**：state-action 序列（含 video + proprioception + 末端位姿）。

#### 3.5.6 MPC 规划

推理时做 **Model Predictive Control（MPC）**：

- 给出 goal image $o_g$（如「桌上某物体的目标位置」的图像）；
- 在 latent space 用 CEM（Cross-Entropy Method）或类似采样搜索 action 序列 $a_t, ..., a_{t+H-1}$，让模型对执行后未来帧的表征预测接近 $E_\theta(o_g)$；
- 真正执行只取第一个 action，下一时刻重新规划。

这个过程**不生成任何像素**，纯在 latent space 做，因此比生成式世界模型快 30×。

### 3.6 LLM 对齐（Video QA）

为了让 V-JEPA 2 接入自然语言问答：

- 用一个 8B 参数的 LLM backbone（如 Llama 3 / Qwen2.5 等）；
- 把 V-JEPA 2 输出经过 projector（轻量 MLP）映射到 LLM 词嵌入空间；
- 训练数据是 video-QA pairs（PerceptionTest、TempCompass 等）；
- 训练时 frozen V-JEPA 2 encoder + trainable projector + frozen/finetune LLM（论文未明示具体策略）。

---

## 4. 数据集与实验设置

### 4.1 预训练数据

- **VM22M**：22M videos（HowTo100M + K400/600/700 + SSv2 + 其他），约 **1M+ hours**；
- **1M images**：从公开网络图像数据采样；
- **总 token 数**：训练 token 量级约 $10^{12}$（粗算），约 252K iter × batch 数千。

### 4.2 训练超参

| 超参 | V-JEPA 2 |
|---|---|
| Encoder | ViT-g/16 (~1.2B params) |
| Predictor depth | 12 |
| Predictor dim | 384 |
| 分辨率 | 384×384 |
| 时长 | 16 frames（基础）/ 64 frames（高分辨率） |
| 时空 patch | $2 \times 16 \times 16$ |
| Batch size | 数千 |
| Iterations | 252K |
| Optimizer | AdamW |
| LR schedule | Warmup-constant-decay |
| Mask ratio | ~90%（短程 + 长程多块） |
| Loss | L1 |
| EMA momentum | 0.999 → 1.0 |

### 4.3 下游评测协议

- **Frozen Attentive Probe**（视频任务）：冻结 encoder，训一个 4-layer attentive probe 头；
- **LLM Alignment**（视频问答）：用 projector 把 V-JEPA 2 表征映射到 LLM token 空间，训 projector；
- **Robot Pick-and-Place**：在两个**未见过的实验室**的 Franka Panda 双臂上做零样本规划任务，评价指标是成功率（任务完成比例）；
- **6 任务 average**：SSv2 + Diving-48 + Jester + Kinetics + COIN + ImageNet。

---

## 5. 实验结果

### 5.1 主要定量结果

下表汇总 V-JEPA 2 在主要任务上的 SOTA 结果：

| 任务 | V-JEPA 2 | 之前 SOTA | 备注 |
|---|---|---|---|
| **Something-Something v2 (motion)** | **77.3 top-1** | ~72 (V-JEPA 1) | 提升 5pt |
| **Epic-Kitchens-100 action anticipation** | **39.7 R@5** | 36.8 (task-specific) | SOTA，超过专门为 EK 设计的模型 |
| **PerceptionTest VQA (8B LLM)** | **84.0** | 79 (Video-LLAVA 等) | 视频问答 SOTA |
| **TempCompass VQA (8B LLM)** | **76.9** | 73 | 视频问答 SOTA |
| **Diving-48** | ~85 | ~80 | motion task |
| **Jester** | ~92 | ~88 | gesture recognition |
| **Kinetics-400** | ~89 | ~84 (V-JEPA 1 81.9) | appearance task |
| **ImageNet-1K (image)** | ~80 | 83 (DINOv2) | 略输 DINOv2 |
| **6-task average** | **88.2** | 84.2 (V-JEPA 1) | 提升 4pt |
| **物理推理速度 vs Cosmos** | **30× faster** | — | 不需要视频去噪 |

### 5.2 Scaling 行为

Figure 3 给出的 5 步 scaling recipe（84.2 → 88.2）是 V-JEPA 2 最具说服力的结果之一，证明 JEPA 架构在 video 模态上仍然享受 scaling laws。

论文还报告了**单独 factor scaling** 的细节：

- **Data scaling**：2M → 22M videos（10× 数据）带来 +1.0pt，**未见饱和迹象**，预示更大数据可能继续获益；
- **Model scaling**：300M → 1B+（3.3× 参数）带来 +1.5pt，**最大的单一 scaling 收益**；
- **Long training**：90K → 252K iter（2.8× 计算）带来 +0.8pt，与 cosine schedule 相比 warmup-constant-decay 更稳定；
- **Higher resolution**：256×256 → 384×384 + 16 frames → 64 frames 带来 +0.7pt，时空双向都贡献。

### 5.3 V-JEPA 2-AC 机器人规划结果

论文 Section 4 给出机器人规划实验：

- **平台**：Franka Panda 双臂；
- **场景**：两个**未见过的实验室**（论文强调模型从未在这些环境上训练过，是真正的 zero-shot）；
- **任务**：pick-and-place（用 goal image 指定目标位置）；
- **数据**：<62 hours Droid 无标签机器人视频（与 V-JEPA 2 预训练的 1M+ hours 互联网视频完全独立）；
- **结果**：零样本完成多组 pick-and-place 任务，无需任何 task-specific 训练或 reward。

论文 Figure 5 / Section 4.2 给出了具体成功率（虽然数字需要从论文附录细看），核心叙事是：

> *「This is achieved without collecting any data from the robots in these environments, and without any task-specific training or reward.」*

这是**首个公开的大规模证据**，证明「互联网视频自监督 + 小量机器人数据」可以构建可零样本规划的真实世界模型。

### 5.4 LLM 对齐结果

把 V-JEPA 2 与 8B LLM 对齐后：

- **PerceptionTest**：84.0（视频问答 SOTA）；
- **TempCompass**：76.9（时间推理视频问答 SOTA）；
- **零样本视频问答**：在多个 benchmark 上拿到 SOTA，**不需要任何 task-specific 训练**。

证明 V-JEPA 2 表征与语言空间可以无缝对接，为视频问答、机器人语言指令理解提供基础。

### 5.5 速度优势

- **V-JEPA 2-AC**：在 latent space 做 MPC，单步规划 < 100ms；
- **vs Cosmos / Sora 类生成式世界模型**：每步生成视频 1-3 秒，**慢 30×**；
- **意义**：潜空间预测的「不可视化」劣势被「30× 速度」完全补偿，对实时机器人控制尤其关键。

---

## 6. 与相关工作的关系

V-JEPA 2 在世界模型坐标系里的位置：

| 路线 | 代表工作 | 数据需求 | 规划速度 | 可解释性 |
|---|---|---|---|---|
| **从 interaction data 学** | DreamerV3、TD-MPC | 千小时+ | 中 | 潜空间 |
| **生成式视频世界模型** | Cosmos、Sora、GAIA-2 | 百万小时视频 + 推理时去噪 | 慢（生成视频） | 像素 |
| **JEPA 路线（非生成式潜空间预测）** | I-JEPA → V-JEPA → **V-JEPA 2** | 1M+ hours 视频（自监督） | **快 30×** | 潜空间 |
| **图像迁移** | DINOv2、I-JEPA | 142M images | 快 | 潜空间 |

V-JEPA 2 的关键定位：

- **vs V-JEPA 1**：方法论一致，但 scale up 4 个因子（data / model / training / resolution），总提升 +4pt；
- **vs Cosmos**：物理推理快 30×，且不需要生成视频；缺点是无法可视化未来；
- **vs DreamerV3**：依赖的预训练数据是互联网视频而非交互数据，scalability 强；下游规划能力需要 V-JEPA 2-AC 加少量机器人数据；
- **vs I-JEPA / LeJEPA**：V-JEPA 2 把 LeJEPA 理论（SIGReg、高斯潜变量、线性可识别）的工程版推到视频 + 机器人维度，与 LeWorldModel 等像素端到端训练形成互补。

V-JEPA 2 与「生成式视频世界模型」的互补关系：

- V-JEPA 2 主动丢弃不可预测的像素细节（预测表征），推理快、适合实时控制；
- 生成式保留像素细节，可视化、可直接用于视频生成、互联网规模物理先验；
- 一个自然的混合路径是 **「V-JEPA 2 做规划 backbone + 生成式模型做未来想象」**。

---

## 7. 局限与批判性评价

1. **潜空间预测的「不可视化」问题**：V-JEPA 2 的输出是 latent representation，不是像素。要可视化未来必须额外训一个 decoder，论文没做。这意味着用户很难直观判断「世界模型预测得对不对」，debug 成本大。
2. **62 小时机器人数据仍然是瓶颈**：V-JEPA 2-AC 用 Droid 的 <62 hours 数据训练，这些数据仍是真实机器人采集，依赖昂贵的真实世界交互。如果换成 sim 数据，质量会下降；换更多真实数据，scalability 受限。
3. **Block-causal attention 设计固定**：V-JEPA 2-AC 用 block-causal attention 处理时间因果，假设「当前时刻不依赖未来」。但在某些场景下（如 agent 自身运动被未来观测影响）这个假设可能不成立。
4. **LLM 对齐依赖 LLM backbone**：PerceptionTest 84.0 等数字依赖 8B LLM 的质量，V-JEPA 2 自身的贡献是「vision encoder」，但评测时和 LLM 强耦合，难以剥离 V-JEPA 2 自身的性能提升。
5. **评估仍以标准 benchmark 为中心**：6-task average、Kinetics、SSv2 等数字表明「在标准任务上 SOTA」，但缺少在 dense prediction（光流、视频分割、深度）上的验证——V-JEPA 1 的「丢弃细节」可能在这些任务上有问题。
6. **3D RoPE 的稳定性收益需更大规模验证**：3D RoPE 替代绝对 sin-cos 是工程细节，论文未做完整消融。其在大模型 + 长训练上的稳定性收益来自经验而非理论。
7. **「零样本机器人」的可推广性受限**：V-JEPA 2-AC 在两个实验室 Franka 上零样本成功，但**任务都是 pick-and-place**，且 goal 必须是 image goal。对于复杂长程操作任务（叠衣服、做饭、组装家具），62 小时 Droid 数据是否够用、是否需要更大规模，是开放问题。
8. **Scale up 仍有空间但成本激增**：从 1B 推到 10B+ 参数的成本和收益曲线论文未探索；1M+ hours 视频训练已经非常昂贵。

---

## 8. 复现与实践建议

- **算力门槛**：
  - **V-JEPA 2 encoder 预训练**：约 **256- × H100 / 数周**（22M videos + 252K iter + ViT-g），非常昂贵，普通实验室难以复现；
  - **V-JEPA 2-AC 后训练**：约 **8 × H100 / 数天**（<62 hours Droid + ViT-g frozen encoder）；
  - **MPC 推理**：单 GPU 即可，单步 < 100ms；
  - **LLM 对齐**：8 × H100 / 数天；
- **数据准备**：
  - **VM22M**：HowTo100M + K400/600/700 + SSv2 + 其他混合，单卡存储 ~50TB，下载是主要瓶颈；
  - **Droid**：公开数据集，下载相对容易；
- **代码**：官方实现 <https://github.com/facebookresearch/vjepa2> 已开源（MIT），包含：
  - V-JEPA 2 encoder + 预训练 checkpoint；
  - V-JEPA 2-AC 后训练代码；
  - MPC 规划代码；
  - LLM 对齐 projector；
- **实现要点**：
  - 3D RoPE 替代绝对 sin-cos 是关键稳定性改动；
  - Warmup-constant-decay LR 比 cosine 更稳定；
  - Mask 策略与 V-JEPA 1 完全一致（multi-block spatio-temporal），但需要适配更长 clip；
  - V-JEPA 2-AC 训练时 frozen encoder 只训 predictor；
  - MPC 规划用 CEM 或 random shooting，goal image 必须经 frozen encoder 编码；
- **常见踩坑**：
  - 直接用 V-JEPA 1 的 mask sampler 处理 64 帧视频会遇到 memory issue，必须分块 mask；
  - 3D RoPE 的 axis partition 切错会导致时空位置信息丢失；
  - MPC 在 latent space 搜索时如果没有足够好的 cost function（如只用 L2 距离），规划会不稳定；
  - LLM 对齐时如果 projector 学习率过高，会破坏 encoder 表征。

---

## 9. 个人启发与后续问题

1. **「1M+ hours 互联网视频 + 62 hours 机器人数据」是机器人世界模型的新配方**：V-JEPA 2 证明了**互联网视频自监督学到的通用世界知识 + 极少量的机器人交互数据**，就能做出可在未见过的真实机器人上零样本规划的物理世界模型。这条路线比「从零采集千万小时机器人数据」更可扩展，是 VLA 之外的另一条具身基础模型路径。
2. **「non-generative」路线在物理推理上的速度优势是巨大的**：V-JEPA 2-AC 比 NVIDIA Cosmos 快约 30 倍（latent space planning vs pixel generation），这意味着**实时机器人控制 + 长视界规划**在潜空间路线下才可能。生成式世界模型的「可解释、可视化」与潜空间的「快、紧」之间的权衡在机器人场景里明显倾向潜空间。
3. **「Scaling 4 个 ingredients」的范式值得所有 VLA / World Model 工作借鉴**：V-JEPA 2 显式拆解 data scaling / model scaling / long training / higher resolution 的独立贡献，给出每个因子的精确收益，这是非常工程友好的报告方式。后续 VLA / 世界模型论文都应该按这个模板做消融。
4. **3D RoPE + 长视频是大型视频模型的两个工程关键**：3D RoPE 替代绝对 sin-cos + 长 clip 训练 + warmup-constant-decay，这三个工程改动合在一起才让 ViT-g + 22M videos + 252K iter 的 scale up 训练稳定。后续任何想 scale video foundation model 的工作都应当参考。
5. **「image goal + MPC」是 zero-shot 机器人规划的简洁范式**：V-JEPA 2-AC 用 goal image + latent MPC 做规划，不需要任务专属训练或 reward。这种「看图就懂、懂了就干」的范式比「language goal + skill library」更直接，是 LeCun 长期主张的「世界模型 + 规划」路线的工程实现。
6. **未解问题**：
   - V-JEPA 2 表征能否迁移到 dense prediction（光流、视频分割、3D 重建）？
   - 62 hours Droid 数据能否 scale 到更复杂任务（叠衣服、做饭）？还是需要更长训练 + 更多数据？
   - V-JEPA 2 + LLM 对齐能否产出真正的「视频思考模型」（类似 o1 / R1 的视频版）？
   - 「non-generative」路线能否在图像生成、视频生成任务上与 diffusion / 自回归路线竞争？还是只能停留在「理解 + 规划」领域？
   - V-JEPA 2-AC 的 block-causal attention 假设能否放宽？例如在 multi-agent / human-robot interaction 场景下，未来观测可能影响当前决策。

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2506.09985> | 全文（含附录） | 2026-09-03 | 精读主来源 |
| `figures/01-overview.png` | Figure 1：V-JEPA 2 总体框架（互联网视频/图像 → 预训练 → 语言对齐/Attentive Probe/动作条件后训练） | arXiv:2506.09985 | 第 2 页 | 2026-09-03 | 总体框架 |
| `figures/02-multistage-training.png` | Figure 2：多阶段训练（左：V-JEPA 2 mask denoising；右：V-JEPA 2-AC 动作条件自回归预测） | arXiv:2506.09985 | 第 4 页 | 2026-09-03 | 核心方法（两阶段训练） |
| `figures/03-scaling-ingredients.png` | Figure 3：Scaling Ingredients 柱状图（VJEPA 1 baseline 84.2 → +Data → +Model → +Long Training → +Higher Resolution 88.2） | arXiv:2506.09985 | 第 5 页 | 2026-09-03 | 关键实验（scaling 消融） |
| <https://github.com/facebookresearch/vjepa2> | 官方代码仓库（V-JEPA 2 + V-JEPA 2-AC） | — | — | 2026-09-03 | 复现参考 |
| <https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks> | 官方博客 | — | — | 2026-09-03 | 实验背景与对比 |