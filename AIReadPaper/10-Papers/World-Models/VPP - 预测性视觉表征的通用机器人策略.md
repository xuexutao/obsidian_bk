---
type: paper-note
status: draft
domain: World-Models
paper: Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
year: 2025
arxiv: 2412.14803
doi: null
source: https://arxiv.org/abs/2412.14803
project: https://video-prediction-policy.github.io/
code: https://github.com/roboterax/video-prediction-policy
tags:
  - World-Models
  - VLA
  - World-Action-Model
  - Video-Diffusion
  - Inverse-Dynamics
  - Closed-Loop-Policy
created: 2026-08-27
updated: 2026-08-27
---

# VPP：预测性视觉表征驱动的通用机器人策略

## 基本信息

- **作者**：Yucheng Hu*、Yanjiang Guo*、Pengchao Wang、Xiaoyu Chen、Yen-Jen Wang、Jianke Zhang、Koushil Sreenath、Chaochao Lu、Jianyu Chen（* 共同一作）
- **机构**：清华大学交叉信息研究院（IIIS）、上海人工智能实验室、上海期智研究院、星动纪元（RobotEra）、UC Berkeley
- **发表**：ICML 2025 Spotlight（arXiv 2412.14803v2，2025-05-04）
- **项目页**：<https://video-prediction-policy.github.io/>
- **代码**：<https://github.com/roboterax/video-prediction-policy>
- **原文 PDF**：<https://arxiv.org/pdf/2412.14803>

## 一句话结论

VPP 把 Stable Video Diffusion（SVD, 1.5B）的内部表征作为"预测性视觉表征"（同时编码当前帧和未来帧），用一步去噪代替完整视频预测，再用 Video Former + 扩散策略学习隐式逆动力学模型；该设计在 CALVIN ABC→D 上以 Avg. Len 4.33（相对 SOTA +18.6%）和真机灵巧手任务成功率 +31.6% 双指标全面超越之前的世界动作模型代表工作 GR-1、Susie、UniPi、MDT 等。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

构建能跨任务、跨本体的通用机器人策略，需要回答两个核心子问题：（1）怎么把当前观测里的视觉信息编码成下游策略能用的视觉表征；（2）怎么把视频生成模型的"理解物理世界"能力迁移到动作决策。这两个问题在 2024 年的机器人研究里一直是分开优化的：前者用对比学习、单图重建或图像-文本对比（如 R3M、VC-1、Voltron、VIP）学静态视觉表征；后者把视频扩散模型直接当视频预测器，再串一个逆动力学模型（UniPi、SuSIE、GR-1）来提取动作。VPP 的出发点是把两者合二为一：直接把视频扩散模型的内部表征当作下游策略的视觉输入，而不再需要先做完整的视频预测。

### 1.2 现有方法的瓶颈

按照论文与石麻笔记 EP06 专访的总结（胡钰承、郭彦江口径），2024 年之前的工作存在三个层面的不足：

- **视觉编码器无法预测未来**：R3M、VC-1、Voltron 等都基于单图重建或两图对比训练，其表征只覆盖"现在"，下游策略必须自己外推未来，这一步在数据有限时几乎不可能做好；
- **完整视频去噪太慢，无法闭环**：UniPi、SuSIE 把 VDM 当成视频预测器使用，但 10–50 步的去噪流程把控制频率压到开环级别，UniPi 因此退化为先预测整条视频再反推动作的离线流程，推理延迟难以满足实时机器人控制；
- **现有视频-动作联合模型质量不够**：GR-1 用自回归 transformer 同时生成下一帧和动作，但每步只生成一帧，预测质量落后于扩散模型，且没有利用预训练视频基础模型。

VPP 把这三件事一起改：把 SVD 的预训练权重继续训练到机器人操作分布、用一步前向替代完整去噪、引入 Video Former 把时序表征压成固定 token。

### 1.3 本文核心贡献

1. 提出"预测性视觉表征（Predictive Visual Representations）"这一概念——VDM 内部张量的 `(T, H, W)` 结构天然显式表达 1 个当前步和 (T−1) 个预测未来步；
2. 设计两阶段训练：第一阶段把通用视频基础模型 SVD 微调成面向机器人操作的 Text-guided Video Prediction (TVP) 模型，第二阶段在 TVP 的一次前向表征上学习隐式逆动力学；
3. 提出 Video Former 模块，用 learnable tokens + 时空 attention 把表征压缩到固定长度的 token；
4. 在仿真（CALVIN ABC→D、MetaWorld 50 任务）和真机（Franka Panda、12-DoF XHand 灵巧手含 4 个工具使用任务）上同时验证，平均任务完成长度相对 SOTA 提升 18.6%、灵巧手任务成功率提升 31.6%。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：当前多视角观测 $\mathbf{s}_0 = \{\mathbf{s}_0^{\text{static}}, \mathbf{s}_0^{\text{wrist}}\}$（第三视角 + 腕部相机），以及语言指令 $\mathbf{l}_{\text{emb}}$；
- **输出**：未来动作序列 $\mathbf{a}_0 = (\bar{\mathbf{a}}_i, \bar{\mathbf{a}}_{i+1}, \ldots, \bar{\mathbf{a}}_{i+m})$（action chunking，默认 10 步）；
- **核心假设**：只要 TVP 模型能预测出足够清晰的未来，逆动力学模型只需"在预测未来中跟踪机器人手臂的位置"即可生成正确动作；这个假设把视频生成模型的泛化能力直接迁移到策略层。

### 2.2 关键符号和目标函数

扩散目标（Stage 1，TVP 微调）：

$$\mathcal{L}_D = \mathbb{E}_{\mathbf{x}_0 \sim \mathcal{D}, \boldsymbol{\epsilon}, t}\left\| V_\theta(\mathbf{x}_t, \mathbf{l}_{\text{emb}}, \mathbf{s}_0) - \mathbf{x}_0 \right\|^2$$

其中 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$，是视频序列上加噪后的潜变量；$\mathbf{s}_0$ 作为初始帧条件与 $\mathbf{x}_t$ 在通道维拼接。多数据集加权损失：

$$\mathcal{L}_{\text{video}} = \lambda_H \mathcal{L}_{\mathcal{D}_H} + \lambda_R \mathcal{L}_{\mathcal{D}_R} + \lambda_C \mathcal{L}_{\mathcal{D}_C}$$

其中 $\mathcal{D}_H$ 是互联网人类操作数据、$\mathcal{D}_R$ 是互联网机器人数据、$\mathcal{D}_C$ 是自采数据，$\lambda$ 按采样比例加权（详见第 4.1 节）。

Stage 2 的扩散策略目标：

$$\mathcal{L}_{\text{diff}}(\psi; \mathcal{A}) = \mathbb{E}_{\mathbf{a}_0, \boldsymbol{\epsilon}, k}\left\| D_\psi(\mathbf{a}_k, \mathbf{l}_{\text{emb}}, \mathbf{Q}'') - \mathbf{a}_0 \right\|^2$$

其中 $\mathbf{Q}''$ 是 Video Former 聚合出的固定 token 序列，$D_\psi$ 是 DiT 扩散策略。

---

## 3. 核心方法

### 3.1 总体框架

VPP 是两阶段 pipeline（Figure 2）。Stage 1 把 SVD (1.5B 参数，1.5B 的 transformer U-Net) 改造成 16×256×256 的机器人操作视频预测器；Stage 2 冻结 TVP，把它当视觉编码器，用 1 次前向提取 up-sampling 层特征，Video Former 把时空表征压缩成 token，DiT 扩散策略基于这些 token 生成动作。控制频率可达 7–10 Hz，整条推理 <160 ms（NVIDIA RTX 4090），实现了闭环控制。

![](assets/VPP%20-%20预测性视觉表征的通用机器人策略/figures/01-teaser.png)

> **图 1：预测性视觉表征 vs 传统视觉表征。** 上半：传统视觉编码器（对比学习 / 单图重建）输出 (H, W) 张量，只能表达"现在"。下半：视频扩散模型的内部表征 shape 是 (T, H, W)，显式包含 1 个当前帧 + (T−1) 个预测未来帧。这是 VPP 的"为什么做"——把视频模型当作本身就具备未来预测能力的视觉编码器，而不再额外训一个静态编码器。
> 来源：论文 Figure 1，第 1 页，<https://arxiv.org/abs/2412.14803>

![](assets/VPP%20-%20预测性视觉表征的通用机器人策略/figures/02-method-pipeline.png)

> **图 2：VPP 两阶段训练框架。** 左半 Stage 1：用互联网人类/机器人 + 自采机器人数据 fine-tune SVD，加入 CLIP 语言特征 cross-attention，得到 TVP（manipulation-focused）。右半 Stage 2：把 TVP 冻结、只做一次前向，提取 up-sampling 层特征并插值到统一分辨率，concat 后送入 Video Former（learnable tokens → spatial-attn → temporal-attn → FFN），再经 DiT 扩散策略生成动作 chunk。
> 来源：论文 Figure 2，第 4 页，<https://arxiv.org/abs/2412.14803>

### 3.2 关键模块一：Text-guided Video Prediction (TVP) 模型

论文在 Stage 1 选用开源的 Stable Video Diffusion（SVD, Blattmann et al., 2023a）作为基础，原因有三：它是当时最大的开源视频扩散模型（1.5B 参数）、原生基于 transformer U-Net、训练数据量大且具备物理世界先验。为了让它适配机器人操控，做了三个调整：

1. **加入语言条件**：原 SVD 只用初始帧 $\mathbf{s}_0$ 作条件。VPP 增加 CLIP 语言特征 $\mathbf{l}_{\text{emb}}$，通过 cross-attention 层注入，让模型可以根据指令预测不同未来；
2. **调整输出分辨率**：把视频序列统一到 $16 \times 256 \times 256$（16 帧 / 256×256），平衡训练效率与下游可用性；
3. **替换数据集，保留其他组件**：保留 SVD 的 UNet、时间位置编码等核心结构，只替换训练数据为人类操作 + 机器人操作混合集。

数据集分三类，按采样比例 $\lambda$ 加权：
- **$\mathcal{D}_H$（互联网人类操作）**：Something-Something-v2，191,642 条轨迹，$\lambda_H = 0.30$；
- **$\mathcal{D}_R$（互联网机器人操作）**：RT-1（87,212）、Bridge（23,377）、BC-Z（43,264）、Taco-Play（3,603）、Jaco-Play（1,085），$\lambda_R$ 共 0.40；
- **$\mathcal{D}_C$（下游任务视频 + 自采）**：Calvin-ABC（18,033）、MetaWorld（2,500）、Panda Arm（2,000）、Dexterous Hand（2,476），$\lambda_C$ 共 0.30。

合计约 37.5 万条轨迹，覆盖人类第一视角操作、单臂工业操作、桌面抓放、长程语言条件任务等多个分布。Stage 1 训练用 8 块 A100 约 2–3 天。

### 3.3 关键模块二：TVP 作为 Vision Encoder（One-step Forward）

这是 VPP 最具巧思的工程决策。论文做了一个关键观察：**VDM 的去噪过程存在明显的"信息密度分布不均"——大尺度动态信息（机器人手臂方向、目标物体的接近方式）在去噪前几步就已基本确定，后期去噪只是修补纹理细节**。基于此，他们把 TVP 当视觉编码器使用：

$$\mathbf{L}_m = V_\theta(\mathbf{x}_{t'}, \mathbf{l}_{\text{emb}}, \mathbf{s}_0)^{(m)}, \quad \mathbf{L}_m \in \mathbb{R}^{T \times C_m \times W_m \times H_m}$$

其中 $t'$ 通常是最大噪声时刻（即从纯噪声开始），$m$ 是 up-sampling 层索引。论文验证了上采样层（而非下采样层）能产出更有效的表征（Xiang et al., 2023 的发现）。多个层的特征被线性插值到统一分辨率 $W_p \times H_p$ 后沿通道维 concat，得到最终预测性视觉表征：

$$\mathbf{F}_p = \text{concat}(\mathbf{L}'_0, \mathbf{L}'_1, \ldots, \mathbf{L}'_m, \text{dim}=1) \in \mathbb{R}^{T \times (\sum_m C_m) \times W_p \times H_p}$$

多视角（第三视角 + 腕部相机）独立预测后记为 $\mathbf{F}_p^{\text{static}}$、$\mathbf{F}_p^{\text{wrist}}$。

### 3.4 关键模块三：Video Former

TVP 表征是高维（$T \times C \times H \times W$）的时空序列，直接喂给策略会面临维度爆炸和难以跨视角融合的问题。论文设计了 Video Former 把多视角多帧表征压成固定 token：

$$\mathbf{Q}' = \{\text{Spat-Attn}(\mathbf{Q}[i], (\mathbf{F}_p^{\text{static}}[i], \mathbf{F}_p^{\text{wrist}}[i]))\}_{i=0}^{T}$$

$$\mathbf{Q}'' = \text{FFN}(\text{Temp-Attn}(\mathbf{Q}'))$$

- **Learnable tokens $\mathbf{Q}$**：shape 为 $T \times L$，是固定数量的可学习查询；
- **Spatial-Attention**：对每一帧 i，用 $\mathbf{Q}[i]$ 对该帧的第三视角 + 腕部相机表征做注意力，融合两个视角的空间信息；
- **Temporal-Attention**：在 $T$ 个帧之间做注意力，让每个 token 都能感知整段时间历史；
- **FFN**：标准 transformer 前馈。

输出 $\mathbf{Q}''$ 是 $T \times L \times d$ 的 token 序列，作为 DiT 扩散策略的 condition。

### 3.5 训练目标与损失函数

Stage 2 的训练标准扩散策略损失（第 2.2 节）。作者沿用 Diffusion Policy 的 action chunking（默认 10 步），与 DiT block 配合：

$$\mathbf{a}_k = \sqrt{\bar{\beta}_k}\mathbf{a}_0 + \sqrt{1-\bar{\beta}_k}\boldsymbol{\epsilon}$$

$D_\psi$ 通过 cross-attention 层把 $\mathbf{Q}''$ 注入 DiT block；最终从 $\mathbf{a}_k$ 去噪得到完整动作序列。Stage 2 训练在 4 块 A100 上约 6–12 小时。

### 3.6 推理流程与复杂度

推理时一次完整流程：
1. **编码**：当前观测 $\mathbf{s}_0$（多视角）→ TVP 一次前向 → $\mathbf{F}_p$（up-sample 层特征）→ Video Former → $\mathbf{Q}''$；
2. **动作生成**：扩散策略以 $\mathbf{Q}''$ 为条件，10 步去噪得到 10 步动作 chunk；
3. **执行**：执行 chunk 中前若干步，下一时刻重新编码。

整体推理时延 <160 ms（RTX 4090），控制频率 7–10 Hz；相比 UniPi（需要先完整去噪生成整条视频）、SuSIE（需先生成 goal image）快 1 个数量级以上。这是 VPP 能做闭环控制的关键。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

仿真：
- **CALVIN ABC→D**：训练在 ABC 环境（静态 + 腕部相机，4 个子任务），零样本评测在未见过的 D 环境执行 5 个连续子任务，指标是 Avg. Len（连续完成的子任务数）。这是检验长程泛化的标准基准。
- **MetaWorld**：Sawyer 机械臂 50 个任务，按难易分 Easy（28）/ Middle（11）/ Hard（11）三档。每任务 50 条轨迹，由 Oracle 策略采集。

真机：
- **Franka Panda**：6 类任务（pick / place / press / route / drawer），30+ 子任务，2,000 条轨迹，space mouse 遥操作；
- **Xarm + 12-DoF XHand 灵巧手**：13 类任务（含 cup-upright、stack、pass、unplug、pouring、tool-use 等），100+ 子任务，2,476 条轨迹，用 Vision Pro 捕捉手部关节后重定向到灵巧手。

训练 TVP 时还加入 Something-Something-v2、RT-1、Bridge、BC-Z、Taco-Play、Jaco-Play 等公开数据集，按采样比例加权。

### 4.2 Baseline 与评价指标

仿真对比 RT-1、Diffusion Policy、Robo-Flamingo、UniPi、MDT、SuSIE、GR-1、Robo-Uniview、Vidman 9 个基线；真机对比 Diffusion Policy、SuSIE、GR-1（最强基线）。指标：仿真用 Avg. Len（CALVIN）和分类成功率（MetaWorld）；真机用任务成功率，按 seen / unseen / tool-use 分组。

### 4.3 实现细节

- TVP 基础模型：SVD 1.5B，输出 16×256×256；Stage 1 训练 8×A100 约 2–3 天；
- Stage 2：Video Former 6 层、DiT 4 层 encoder + 4 层 decoder；batch size 在 64–128 之间；学习率 1e-4 或 5e-5；12–40 个 epoch；
- 推理：NVIDIA RTX 4090，<160 ms 一次前向，10 步扩散采样；
- action chunk：10 步；
- 评估：CALVIN 每条链跑 1000 次（报告子任务完成率）、MetaWorld 每任务跑 50 次、真机 Franka Panda 200+ rollout / XHand 500+ rollout。

---

## 5. 实验结果

### 5.1 主要定量结果

**CALVIN ABC→D（Table 1）**：VPP Avg. Len = 4.33，相对之前 SOTA（GR-1 3.06、Vidman 3.42、RoboUniview 3.65）提升 26.5%–41.5%，与论文标题宣称的"18.6% 相对改进"对应的是相对 GR-1 的提升口径。值得注意的是：**用 10% 训练数据时 VPP 仍达到 Avg. Len 3.25，超过全量数据训练的 GR-1、MDT、Uni-Pi 等所有基线**——这体现了预测性视觉表征对数据量的需求远低于直接动作学习。

**MetaWorld 50 任务（Table 2）**：平均成功率 0.682，比 GR-1（0.574）高 10.8 个百分点；Hard 任务（11 个）0.526 vs GR-1 0.451，差距同样显著。

**真机 Franka Panda（Table 5）**：seen 0.856 vs GR-1 0.519 / Susie 0.563 / DP 0.425；unseen 0.737 vs GR-1 0.388。

**真机 XHand 灵巧手（Table 5 + Table 7）**：seen 0.749 vs GR-1 0.319；unseen 0.605 vs GR-1 0.159；**4 个工具使用任务（spoon / hammer / drill / pipette）平均 0.68 vs GR-1 0.15**——这是最体现泛化能力的指标，因为训练集里完全不包含这些工具。

### 5.2 定性结果

![](assets/VPP%20-%20预测性视觉表征的通用机器人策略/figures/03-one-step-representation.png)

> **图 3：一步去噪可视化（Figure 4）。** 三行分别是 GT（绿框）、30 步去噪预测（红框）、1 步直接预测（蓝框）。可以看到：30 步去噪预测在细节上接近 GT（纹理、形状），而 1 步预测虽然细节模糊，但仍能反映**机器手臂的运动方向、目标物体的位移、夹爪的开合**等关键动态信息。这是 VPP"用一次前向即可"的视觉证据。
> 来源：论文 Figure 4，第 7 页，<https://arxiv.org/abs/2412.14803>

![](assets/VPP%20-%20预测性视觉表征的通用机器人策略/figures/04-realmworld.png)

> **图 4：真机平台与任务示例（Figure 5）。** 包含两类硬件：Franka Panda + 腕部相机（30+ 桌面任务）和 Xarm + 12-DoF XHand 灵巧手（100+ 任务，含 4 个工具使用）。任务覆盖 picking、placing、stacking、pouring、tool-use 等。这张图说明 VPP 不仅在仿真里成立，在工业级和灵巧手级真机上也能保持领先。
> 来源：论文 Figure 5，第 8 页，<https://arxiv.org/abs/2412.14803>

Figure 6（unseen 任务）显示：VPP 在网球 / 可口可乐 / 勺子（这些物体在训练集里从未出现过）的任务上，视频预测模型仍能给出合理的未来轨迹，实际执行轨迹与预测高度吻合。论文把这个泛化机制归结为两点：（1）视频模型基于互联网规模预训练，对未见物体也能做出正确视觉预测；（2）底层策略只需在预测未来中跟踪机器人自身运动，不需要去关注新物体/新背景。

### 5.3 消融实验

**表征对比（Table 3，CALVIN ABC→D Avg. Len）**：

| 视觉编码器 | 预训练任务 | Avg. Len |
|---|---|---|
| VDM (Ours) | 视频生成 | **4.33** |
| Stable-VAE | VAE 重建 | 2.58 |
| VC-1 | MAE 重建 | 1.23 |
| Voltron | MAE + 语言 | 1.54 |

把视频生成模型换成单图 VAE/MAE 编码器，Avg. Len 下降 40%–72%——这是"预测性视觉表征"最直接的证据：能预测未来的表征，对长程操控任务的价值远高于只能编码现在的表征。

**视频预训练 + 互联网数据（Table 4）**：

| 配置 | Avg. Len | 延迟 |
|---|---|---|
| VPP 完整 | 4.33 | ~140 ms |
| 去互联网数据 | 3.97 | ~140 ms |
| 去 Calvin video | 3.31 | ~140 ms |
| 去 SVD 预训练 | **1.63** | ~140 ms |

去掉 SVD 预训练权重后从零训练 TVP，Avg. Len 直接腰斩到 1.63——证明 SVD 预训练的物理世界先验是 VPP 泛化能力的核心来源。

**Video Former 与 Feature Aggregation（Table 4）**：

| 配置 | Avg. Len | 延迟 |
|---|---|---|
| 完整 VPP | 4.33 | ~140 ms |
| 去 Video Former | 3.86 | **~450 ms** |
| 去 Feature Aggregation | 3.60 | ~140 ms |

Video Former 不只提升精度，还把延迟从 450 ms 压到 140 ms——它把高维时空表征压缩到固定 token，使下游策略可以做并行 attention 而不需要逐帧处理。

**特征层选择（Table 10）**：Layer-3 / 6 / 9 / 12 单独使用时 Avg. Len 分别是 3.72 / 3.88 / 4.29 / 4.05，全层 concat 后 4.33——深层 + 多层聚合最优。

**去噪步数（Table 11）**：1 步 / 10 步 / 20 步 / 30 步 Avg. Len 分别是 4.21 / 4.33 / 4.25 / —，2 步反而略低于 1 步（4.19 vs 4.33），论文假设 1 步已足够产生有效轨迹预测，2 步把推理时间翻倍、控制频率减半，权衡之下选 1 步。

**单视角消融（Table 12）**：只用第三视角时 Task 5 成功率 3.58，超过使用双视角+深度的 3D Diffuser Actor——这说明视频扩散模型本身的视觉信息密度已经很高，不一定需要深度或多视角补充。

### 5.4 泛化、效率与失败案例

- **泛化**：VPP 在训练集中**完全没出现过的物体**（网球、可口可乐瓶、勺子、锤子、电钻、移液器）上仍能生成合理动作，证明视频模型的预训练先验在 VPP pipeline 中得到了很好保留；
- **效率**：单次推理 <160 ms（4090），控制频率 7–10 Hz，达到商业部署标准；之前 UniPi 类方法需要先整条视频去噪，控制频率只能到 Hz 以下；
- **失败案例**：论文未单独列出失败 case 的图片或定量统计，但承认灵巧手工具使用任务中"pipette"（移液器）成功率仅 0.4，明显低于 spoon 0.9；推测原因是精细工具使用涉及复杂多指协作，1.5B 视频模型的预测能力不够支撑。

---

## 6. 与相关工作的关系

按论文与石麻笔记 EP06 专访口径梳理：

- **vs SuSIE / UniPi**：同属"未来预测驱动策略"路线，但 SuSIE/UniPi 把视频模型当"视频预测器"使用（先生成完整未来再反推动作），VPP 把视频模型当"视觉编码器"使用（只用其内部表征）。VPP 的关键优势是 1 步前向的推理速度 + 闭环控制能力。
- **vs GR-1**：GR-1 也是 video + action 联合训练，但用自回归 transformer，每步只生成一帧，没利用视频基础模型预训练。VPP 用 SVD 预训练 + 一步去噪，在数据效率、生成质量、控制频率三个维度全面领先。
- **vs MDT**：MDT 把 MAE loss + 扩散 transformer 用于策略学习，仍然依赖传统视觉编码器（VC-1）；VPP 直接用 VDM 内部表征，效果更好（4.33 vs MDT 1.55）。
- **vs Diffusion Policy**：纯扩散策略，没有利用视频先验；CALVIN 上 Avg. Len 0.56，远低于 VPP 4.33。
- **vs Robo-Uniview**：3D-aware 视觉编码器 + 3D occupancy loss 辅助策略；VPP 用预测性视觉表征在 ABC→D 上超越（4.33 vs 3.65），且不需要 3D 标注。
- **后续延伸**：VPP 的两步走思想被同一团队的 BagelVLA（arXiv 2602.09849）扩展为"理解 + 预测 + 动作"三位一体的统一模型；同期 Dyna-2（百万小时人类视频缩放定律）、Fast-WAM（推理时跳过未来想象）等工作都验证了"预测性视觉表征"的有效性。

---

## 7. 局限与批判性评价

1. **像素级预测的固有代价**：虽然 1 步去噪避免了多步去噪的时间成本，但 VPP 仍然需要 1.5B 参数的 SVD 作为 backbone，显存和算力门槛高于纯静态编码器方案（如 VC-1 1.23 也能跑，只是精度差）；
2. **1 步表征的纹理是模糊的**：Figure 3 蓝框显示一步预测在纹理层面远不如 30 步去噪（红框），论文把它定位为"足够用于动作决策"——但如果下游任务需要更精确的空间感知（如细小螺丝拧紧），一步表征可能不够；
3. **互联网数据的分布偏移**：Something-Something-v2 是人类第一视角操作，与桌面 Franka / 12-DoF 灵巧手的视觉分布差异较大。论文靠采样比例加权缓解，但没有专门做域适应；
4. **数据集规模仍小**：37.5 万条轨迹对视频扩散模型来说是很小的训练量（对比 Cosmos 20M 小时）；TVP 的物理世界先验主要来自 SVD 预训练而非继续训练；
5. **依赖 SVD 开源权重**：闭源视频模型（如 Sora）无法直接复用此路径，限制了方法的通用性；
6. **工具使用泛化的上限**：XHand 工具任务平均 0.68，但 drill 仅 0.4、pipette 仅 0.4，说明对长工具、复杂多指协作仍有显著瓶颈；
7. **没有真正做"未来不确定性"建模**：1 步去噪是一个点估计，没有像联合模型（PAD、Cosmos Policy）那样输出多种未来可能性，因此在多模态操作任务上可能受限。

---

## 8. 复现与实践建议

- **算力门槛**：Stage 1 TVP 微调需要 8×A100 2–3 天，Stage 2 策略训练需要 4×A100 6–12 小时；推理单卡 RTX 4090 即可（<160 ms）；
- **数据准备**：需要 Something-Something-v2（人类操作）+ RT-1 / Bridge / BC-Z 等机器人数据集 + 自己的下游任务数据；Calvin-ABC 与 MetaWorld 是公开 benchmark，可直接下载；
- **代码可获得性**：GitHub `roboterax/video-prediction-policy` 已开源（含 TVP 训练、策略训练、推理脚本）；
- **实现要点**：
  - SVD 替换条件时记得加 CLIP 文本编码器；
  - Video Former 的 learnable token 数量 L 是关键超参（论文 14–28）；
  - 多视角预测要分别走 TVP，再在 Video Former 内融合；
  - 推理时 TVP 必须冻结，避免一次前向破坏表征；
- **常见踩坑**：用 1.5B SVD 微调时 batch size 不要太大（论文 4），否则会 OOM；下游策略 batch size 可放大到 64–128。

---

## 9. 个人启发与后续问题

1. **"视频模型当 encoder"的范式已经被多篇后续工作印证**：Fast-WAM 直接验证了"推理时不需要未来想象、训练期联合视频建模才是 WAM 真正的价值来源"；Dyna-2 把人类视频规模推到 100 万小时并画出了可预测的 scaling law。VPP 是这条线的起点之一。
2. **"动作学习是视频模型的副产品" vs "视频模型是动作学习的辅助"**：VPP 把视频模型当 encoder 用，本质上是后者；同期 PAD（郭彦江，NeurIPS 2024）等工作坚持前者（联合预测 video + action），两种路线在仿真和真机上各有胜负。本文的实证表明在数据有限时 VPP 路线占优，但 pi0.7 等工作表明数据足够时纯 VLA 也能 work。
3. **一步去噪 vs 多步去噪 vs 跳过未来想象**：本文用 1 步去噪（<160 ms），UniPi 用多步去噪（开环），Fast-WAM 跳过未来想象（仅 190 ms）——三种推理范式都已被验证有效，但训练目标都是"联合视频建模"。这暗示**视频模型的核心价值在训练阶段而非推理阶段**，是一个值得继续追问的方向。
4. **统一建模的下一步**：BagelVLA 把 VPP 的"预测性视觉表征"扩展为"subtask + keyframe"两段式生成，并在 VLM 理解能力的加持下取得更复杂任务的能力；Motus 进一步把 VLM 和 video model 共享 MoT 结构——VPP 是这条统一建模路线的"最小可用版本"。
5. **未解问题**：（i）一步去噪的视觉表征能否继续压缩到 latent/token？（ii）互联网人类数据与机器人数据的最佳混合比例是多少？（iii）多视角 / 多本体联合训练能否进一步把泛化推到一个新台阶？

---

## 10. 材料来源

### Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2412.14803> | 全文（18 页 + 附录） | 2026-08-27 | 精读主来源 |
| `figures/01-teaser.png` | Figure 1：预测性视觉表征 vs 传统视觉表征 | arXiv:2412.14803 | 第 1 页 | 2026-08-27 | 概念对比 |
| `figures/02-method-pipeline.png` | Figure 2：VPP 两阶段训练框架 | arXiv:2412.14803 | 第 4 页 | 2026-08-27 | 核心方法 |
| `figures/03-one-step-representation.png` | Figure 4：一步去噪可视化 | arXiv:2412.14803 | 第 7 页 | 2026-08-27 | 关键定性结果 |
| `figures/04-realmworld.png` | Figure 5：真机平台与任务示例 | arXiv:2412.14803 | 第 8 页 | 2026-08-27 | 真机实验 |
| 项目页 <https://video-prediction-policy.github.io/> | 视频/补充材料 | — | — | 2026-08-27 | 实验视频 |
| 代码 <https://github.com/roboterax/video-prediction-policy> | 官方实现 | — | — | 2026-08-27 | 复现参考 |