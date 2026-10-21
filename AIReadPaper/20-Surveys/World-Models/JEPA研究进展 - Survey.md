---
type: survey
status: complete
domain: World-Models
topic: Joint-Embedding Predictive Architecture
coverage: 2022-06 to 2026-10-16
created: 2026-10-15
updated: 2026-10-16
tags:
  - World-Models
  - JEPA
  - Self-Supervised-Learning
  - Latent-Dynamics
  - Robot-Planning
  - SIGReg
---

# JEPA 研究进展：从表征预测到可规划世界模型（2022—2026）

## 基本信息

- **时间范围**：2022 年 6 月—2026 年 10 月 16 日
- **主线论文**：LeCun 2022 纲领、Slow Features、I-JEPA、MC-JEPA、Image World Models、V-JEPA、DINO-WM、V-JEPA 2、LeJEPA、LeWorldModel、When Does LeJEPA Learn a World Model?、V-JEPA 2.1、LeVJEPA、UniJEPA、Flow-JEPA、AD-E2E-JEPA、LeWAM
- **最新分支**：LeWAM、VL-JEPA、SD-JEPA、M-JEPA 等
- **核心判断**：JEPA 已经从“预测 embedding 的视觉自监督目标”演化成一组世界模型设计原则，但目前还不是单一统一架构。真正被反复验证的是三件事：潜空间预测能避免像素级不可预测细节的负担；可规划 latent 必须同时满足可预测、非坍塌和任务相关；高效规划需要在表示粒度、动力学形式和搜索预算之间做明确权衡。

---

## 1. JEPA 到底是什么

### 1.1 不预测像素，而预测“可预测的世界”

JEPA（Joint-Embedding Predictive Architecture）的核心不是某一种 Transformer，而是一条学习原则：把 context $x$ 和 target $y$分别编码到潜空间，用 predictor 从 context embedding 预测 target embedding，而不是直接生成 target 的像素、波形或 token：

$$s_x=E_x(x),\qquad s_y=E_y(y),\qquad \hat s_y=P(s_x,z)$$

$$\mathcal L_{\mathrm{pred}}=D(\hat s_y,s_y)$$

其中$z$可以是目标位置、时间间隔、光度变换或动作。目标 encoder 允许把不可预测、任务无关的细节从$s_y$里丢掉，因此 predictor 不必复原草叶纹理、压缩噪声或语言表面措辞，只需保留支撑理解和行动的结构。

这与像素生成路线的差别不是“有没有 decoder”这么简单。生成式世界模型希望得到可视未来，优势是细节完整、可人工检查；JEPA 希望得到紧凑可操作状态，优势是训练和规划更快。两条路线解决的是不同问题，未来更可能组合，而不是互相消灭。

![](../../10-Papers/World-Models/assets/A%20Path%20Towards%20Autonomous%20Machine%20Intelligence%20-%20JEPA世界模型纲领/figures/02-ssl-energy-model.png)

> **图 1：LeCun 2022 对生成式、联合嵌入与 JEPA 的能量模型划分。** JEPA 不要求从潜变量完整解码观测，而是让 context 与 target 的兼容性通过低能量/低预测误差表达。来源：《A Path Towards Autonomous Machine Intelligence》原文图。

### 1.2 三种常被混称为 JEPA 的系统

为了避免谱系混乱，本文把相关工作分成三类：

1. **Canonical JEPA**：encoder 与预测目标共同训练，且核心目标是相关视图/时空目标的 latent prediction，例如 I-JEPA、V-JEPA、V-JEPA 2。
2. **JEPA 混合体**：latent prediction 与像素重建、光流、VICReg 或其他任务联合，例如 MC-JEPA。它们验证某个结构，但不能把结果完全归因于纯 JEPA 目标。
3. **JEPA-style world model**：冻结外部预训练 encoder，只学习动作条件 latent dynamics，例如 DINO-WM。它们没有用预测任务塑造 encoder，却遵循“不重建像素、在表征空间预测和规划”的 JEPA 哲学。

LeJEPA 以后又出现第四种形态：用 SIGReg 显式约束表征分布，允许单 encoder、无 EMA/stop-gradient 的端到端训练。LeWorldModel、UniJEPA 和 Flow-JEPA 属于这条“Gaussian-regularized JEPA world model”分支。

### 1.3 一个好 JEPA 必须同时过三关

JEPA 不是只要预测误差低就成功。至少要同时满足：

- **可预测性**：context 和动作足以预测 target latent；
- **非坍塌性**：不能把所有观测编码为同一点或低秩子空间；
- **任务相关性**：不能只保留最慢、最容易预测却与行动无关的背景。

2022—2026 的发展史，本质上就是这三关之间的瓶颈转移：先证明潜空间预测可行，再发现慢特征捷径；然后靠掩码和 teacher 提升语义；接着把视频表征接到动作规划；最后用 SIGReg 重做 anti-collapse，并继续处理稠密信息、鲁棒 rollout 与规划效率。

---

## 2. 技术时间线：问题如何一步步转移

| 时间 | 工作 | 解决的问题 | 新暴露的瓶颈 |
|---|---|---|---|
| 2022-06 | A Path Towards Autonomous Machine Intelligence | 提出 JEPA/H-JEPA、能量推理、层级规划的总蓝图 | 无端到端实证，anti-collapse 与训练方法未确定 |
| 2022-11 | Slow Features | 证明 JEPA 会优先编码易预测的慢变量 | 非坍塌不等于任务相关，背景捷径仍可能占满表示 |
| 2023-01 | I-JEPA | 大块 masked latent prediction 首次在图像上成熟落地 | mask 工程敏感；没有时间、动作与规划 |
| 2023-07 | MC-JEPA | 联合运动、内容和光流，探索动态图像表征 | 混合损失复杂，难判断 JEPA 单独贡献 |
| 2024-03 | Image World Models | 把已知光度变换作为“动作”，研究等变性与 predictor | 仍是人工图像变换，不是真实物理动力学 |
| 2024-04 | V-JEPA | 纯 feature prediction 扩展到视频，兼顾运动与外观 | 主要是双向掩码补全，无动作条件和因果规划 |
| 2024-11 | DINO-WM | 冻结 DINOv2 patch 特征上学习动力学并零样本规划 | 依赖外部 encoder；密集 patch rollout 成本高 |
| 2025-06 | V-JEPA 2 / 2-AC | 规模化视频预训练，并用少量机器人数据做潜空间 MPC | 稠密特征不足、目标图像规划范围窄、成本仍高 |
| 2025-11 | LeJEPA | 用 SIGReg 显式解决统计坍塌，去掉 teacher/stop-grad 依赖 | 高斯健康不保证任务相关或状态可识别 |
| 2026-03 | LeWorldModel | 从像素端到端训练轻量动作条件 JEPA，单卡可复现 | 长时规划、复杂 3D 和低本征维适配有限 |
| 2026-03 | V-JEPA 2.1 | 修复 V-JEPA 2 稠密/时空定位能力 | dense loss 与全局分类存在冲突 |
| 2026-05 | When Does LeJEPA Learn a World Model? | 给出高斯潜变量下线性可识别与规划等价条件 | 真实策略轨迹常非高斯、非平稳 |
| 2026-08 | UniJEPA / LeVJEPA / Flow-JEPA | 统一图像—视频任务、降低视频计算、提升噪声鲁棒性 | 真实机器人、长时程、严格机制拆分仍不足 |
| 2026-09 | AD-E2E-JEPA / M-JEPA | 把规划 patch 压缩到可用规模；优化多 mask 训练执行 | 安全目标、oracle goal、跨节点系统效率仍未闭环 |
| 2026-10 | LeWAM | 用共享 encoder 联合训练未来 latent 预测与 action flow matching，让单模型兼容直接策略执行和基于世界模型的候选动作评估 | 仍需核验复杂真实机器人任务、长时闭环误差、规划与直接策略的机制拆分及对基线算力的公平比较 |

---

## 3. 第一阶段：从自主智能纲领到慢特征警告（2022）

### 3.1 2022 纲领：JEPA 是认知架构的一部分

LeCun 的 position paper 并不是 I-JEPA 的算法说明，而是一整套系统：Perception 抽取状态，World Model 在动作条件下预测，Cost/Critic 评价轨迹，Actor 搜索动作，Configurator 按任务调制各模块，短期记忆保存真实与想象状态。JEPA 是世界模型的训练接口，H-JEPA 则负责跨时间尺度的抽象预测。

![](../../10-Papers/World-Models/assets/A%20Path%20Towards%20Autonomous%20Machine%20Intelligence%20-%20JEPA世界模型纲领/figures/01-system-architecture.png)

> **图 2：自主智能系统总架构。** 关键不是单个视觉 encoder，而是“状态表示—结果预测—代价评价—动作搜索”的闭环。I-JEPA 只实现了其中的表征学习子问题，V-JEPA 2-AC 和 LeWorldModel 才开始接近动作条件世界模型。来源：该论文 Figure 2。

H-JEPA 的洞见至今仍未真正完成：短时低层预测保留细节，长时高层预测主动抽象；规划从高层子目标逐层细化，而不是在每个时间步穷举像素未来。

![](../../10-Papers/World-Models/assets/A%20Path%20Towards%20Autonomous%20Machine%20Intelligence%20-%20JEPA世界模型纲领/figures/03-hierarchical-jepa.png)

> **图 3：H-JEPA 的多时间尺度预测。** 越往高层，时间跨度更长、表征更抽象，允许忽略长时不可预测细节。2026 年的主流 JEPA 仍多为单尺度短 horizon，这张图更像未完成的研究议程。

### 3.2 Slow Features：JEPA 会学“好预测的错东西”

《Joint Embedding Predictive Architectures Focus on Slow Features》用玩具轨迹揭示一个基础问题：预测目标天然偏好时间上稳定的因素。当噪声逐帧变化时，JEPA 会忽略它；但当噪声在整条轨迹里固定，模型反而会把轨迹身份和背景当成最容易预测的信号。

![](../../10-Papers/World-Models/assets/Slow%20Features%20-%20JEPA的慢特征偏置/figures/03-noise-performance.png)

> **图 4：慢噪声会占据 JEPA 表征。** 动态噪声不可预测，模型倾向过滤；轨迹固定噪声具有极强可预测性，即便与任务无关也会被编码。来源：Slow Features 论文关键结果图。

这篇工作给后续研究留下一个一直没有完全解决的分界：**anti-collapse 只保证表示有变化，不保证变化来自正确因素。** I-JEPA 用强 mask 降低局部捷径，LeJEPA 用高斯分布避免统计退化，但 neither 自动阻止 slow nuisance。动作、对象、因果干预和任务进度等结构先验仍然需要进入预测任务。

---

## 4. 第二阶段：图像 JEPA 从概念变成可扩展算法（2023）

### 4.1 I-JEPA：大目标块 + 分散上下文

I-JEPA 是第一个成熟的 canonical JEPA。完整图像通过 EMA target encoder 得到 patch 表征，再采样多个较大 target block；context encoder 只读去掉 target 区域后的上下文；窄 predictor 根据 context 和位置 token 预测目标表征。损失是预测 latent 与 target latent 的 L2 距离。

![](../../10-Papers/World-Models/assets/JEPA%20-%20联合嵌入预测架构/figures/02-method.png)

> **图 5：I-JEPA 三件套。** Context encoder 学可见区域，EMA target encoder 产生稳定目标，predictor 在位置条件下补全目标 embedding。target mask 作用在 encoder 输出而非输入，是语义和效率的关键。来源：I-JEPA Figure 3。

I-JEPA 最有价值的结果不是单一准确率，而是消融中非常陡峭的 mask 规律：多块 target 显著优于随机 patch、单大块和象限式目标；目标要足够大以承载语义，上下文要足够广但不能包含答案。像素空间预测与表征空间预测在 1% ImageNet 评估上存在约 26 个点差距，是“预测什么比 decoder 多强更重要”的直接证据。

它的局限也很明确：无增强并不等于无工程，mask 尺度、数量、context 范围高度敏感；EMA 与 stop-gradient 仍是经验性 anti-collapse；静态图像内补全不能证明世界动力学或规划能力。

### 4.2 MC-JEPA：运动与内容的混合过渡方案

MC-JEPA 用共享 ConvNeXt 同时学习内容与光流，联合 VICReg 表征目标、PWC 风格多尺度光流、warping 与重建约束。它在 DAVIS 等任务展示运动与内容可共享 backbone，却不是“纯 JEPA”：性能来自多目标共同作用，光流路径仍依赖像素 warping。

![](../../10-Papers/World-Models/assets/MC-JEPA%20-%20运动与内容联合表征学习/figures/02-method-pipeline.png)

> **图 6：MC-JEPA 混合 pipeline。** 内容分支、运动分支、光流金字塔和重建/一致性目标被联合训练。它是从静态 I-JEPA 走向视频的桥梁，但不能把所有收益归因于 latent prediction。

其最重要的历史作用，是证明“动态理解不能只把图像 SSL 逐帧应用”。但复杂联合损失也暴露出新问题：如果要验证 feature prediction 本身是否成立，需要像 V-JEPA 那样去掉光流标签、像素重建和额外预训练。

### 4.3 Image World Models：把变换显式作为动作

Image World Models（IWM）把 brightness、contrast、saturation、hue 等图像变换参数输入 predictor，要求预测变换后图像的 embedding。它把“数据增强”从 encoder 应忽略的扰动，改写成 predictor 应解释的 action，因而可以研究表示到底学 invariance 还是 equivariance。

![](../../10-Papers/World-Models/assets/Image%20World%20Models%20-%20动作条件图像表征预测/figures/01-method-pipeline.png)

> **图 7：IWM 的动作条件图像预测。** 已知变换参数调制 predictor，encoder 保留可由变换解释的结构；这为后来把机器人动作、相对位姿和轨迹作为 condition 提供了概念接口。

IWM 的结论是 predictor 容量、任务难度和 condition 信息共同决定表征抽象度。但“世界”仅是人工光度变换，不能代表接触、遮挡、物体持久性和多模态未来。它是动作条件 JEPA 的可控实验台，而不是完整物理世界模型。

---

## 5. 第三阶段：视频 feature prediction 成为独立路线（2024—2025）

### 5.1 V-JEPA：同时学外观与运动

V-JEPA 把 2D patch 升级为时空 tubelet，以约 90% 高掩码率的多块时空 mask 迫使模型跨帧推断；损失改为更稳定的 L1，target encoder 仍由 EMA 更新。它不用像素重建、对比负样本、图像 encoder 或文本监督，仅靠 feature prediction 在 K400、SSv2 和 ImageNet frozen 评估中兼顾外观与运动。

![](../../10-Papers/World-Models/assets/V-JEPA%20-%20视频潜空间预测表征学习/figures/03-method-pipeline.png)

> **图 8：V-JEPA 训练 pipeline。** 时空 token 经高比例 mask 后进入 context encoder，predictor 补全 masked latent，EMA target encoder 提供目标。视频冗余更强，所以 mask 必须跨完整时间维度，避免模型靠邻帧插值作弊。来源：V-JEPA Figure 3。

V-JEPA 证明了“video feature prediction as a stand-alone objective”，但它仍更接近视频自监督表征，不是可交互世界模型：输入 clip 短、训练目标为双向 masked denoising，没有明确动作，也没有严格因果 rollout 与 planner。

### 5.2 DINO-WM：冻结视觉坐标系，专注动力学

DINO-WM 做了一个重要拆分：不再让环境小数据同时塑造 encoder 和 dynamics，而是冻结 DINOv2 patch 特征，只训练动作条件 transition。测试时把当前图和目标图编码成 latent，用 CEM/MPC 搜索动作，使预测终点接近目标 latent。

![](../../10-Papers/World-Models/assets/DINO-WM%20-%20预训练视觉特征上的零样本规划/figures/02-method-pipeline.png)

> **图 9：DINO-WM 的零样本视觉目标规划。** “Zero-shot”主要指新目标无需重训 policy，不代表对新环境动力学完全零样本。冻结 encoder 彻底消除 encoder collapse，却把表示上限锁在 DINOv2。

DINO-WM 在 PointMaze、Push-T、绳索和颗粒环境中说明 patch 特征比单 CLS 更适合空间规划，也证明重建 decoder 不是规划必要条件。代价是 dense token rollout 昂贵，且 planner 会利用模型误差；它还需要离线 action labels，环境覆盖不足时 latent MSE 无法保证真实成功。

### 5.3 V-JEPA 2：从视频理解跨到动作条件规划

V-JEPA 2 把模型扩到约 1.2B 参数、22M 视频和超过 100 万小时视觉数据，并拆解 data、model、training length 和 spatiotemporal resolution 四个 scaling ingredient。其 action-conditioned 变体 V-JEPA 2-AC 冻结大规模视频 encoder，用少于 62 小时 DROID 机器人数据训练 block-causal predictor，再以 goal image 做 latent MPC。

![](../../10-Papers/World-Models/assets/V-JEPA%202%20-%20自监督视频世界模型的理解预测与规划/figures/02-multistage-training.png)

> **图 10：V-JEPA 2 两阶段训练。** 第一阶段从互联网视频学 action-free visual state；第二阶段只用少量机器人 state-action 数据把动作后果接入 predictor。该配方把“大规模观察学习 + 小规模交互适配”变成 JEPA 机器人路线的核心范式。

它的重要性有两层：一是证实 JEPA 可以享受视频规模化收益；二是首次把互联网视频表征与真实机器人零样本 goal-image planning 连成闭环。局限是机器人任务集中于 reaching、grasping、pick-and-place；规划目标需要目标图像；latent 不可直接可视化；对长时任务、复杂接触与跨 embodiment 的证据仍有限。

---

## 6. 第四阶段：从“用技巧防坍塌”到显式高斯表征（2025—2026）

### 6.1 LeJEPA：SIGReg 重做 anti-collapse

LeJEPA 将 JEPA 简化为跨视图对齐与 Sketched Isotropic Gaussian Regularization。SIGReg 将 batch embedding 沿随机方向投影，并用 Epps–Pulley 特征函数检验让一维分布接近标准高斯：

$$\mathcal L=\lambda\mathcal L_{\mathrm{SIGReg}}+(1-\lambda)\mathcal L_{\mathrm{align}}$$

![](../../10-Papers/World-Models/assets/LeJEPA%20-%20无启发式高斯正则自监督学习/figures/01-sigreg-pipeline.png)

> **图 11：SIGReg。** 与 VICReg 只匹配二阶矩不同，随机切片上的完整一维分布检验可发现协方差正常但联合分布仍退化的情况。有限切片和有限 batch 仍是近似，论文给出$O(1/B)$偏差。

论文在 10+ 数据集、60+ 架构与最高 1.8B ViT 上展示无需 EMA teacher、stop-gradient 和 predictor 也可稳定训练。更重要的是，它让训练损失与下游准确率强相关，为无标签模型选择提供信号。

但“高斯最优”不能被简化为“世界是高斯的”。LeJEPA 证明的是给定 probe 风险和约束下的表征几何；SIGReg 防止统计坍塌，却不保证 latent 只编码任务相关因素。

### 6.2 LeWorldModel：最小可复现的端到端动作世界模型

LeWorldModel 把 LeJEPA 的 SIGReg 接到动作条件 next-embedding prediction：

$$\mathcal L=\|P(z_t,a_t)-z_{t+1}\|_2^2+\lambda\mathcal L_{\mathrm{SIGReg}}$$

它用约 15M 参数、ViT-Tiny encoder 和 transformer predictor，从像素端到端训练，不依赖 frozen foundation encoder、EMA、stop-gradient、reward 或重建损失；再用 CEM 在 latent space 搜索动作。

![](../../10-Papers/World-Models/assets/LeWorldModel%20-%20像素端到端稳定JEPA世界模型/lewm-plan.png)

> **图 12：LeWorldModel 的 latent planning。** 当前图与目标图进入同一 encoder，候选动作经 predictor rollout，终点 latent 距离作为 CEM 代价。它把 JEPA 世界模型做成单卡数小时可训练的强基线。

在 Push-T、Reacher 等任务中，LeWM 以远低于 DINO-WM 的规划成本获得竞争结果，并通过物理量 probing 与 violation-of-expectation 说明 latent 对瞬移比颜色变化更敏感。它也暴露 SIGReg 的边界：低本征维 Two-Room 不占优，复杂 3D 旋转弱于 foundation feature，规划 horizon 仍短。

### 6.3 When Does LeJEPA Learn a World Model?：理论闭环与现实缺口

该论文进一步问：何时 LeJEPA embedding 真能恢复真实世界状态？在真实 latent $z\sim\mathcal N(0,I)$、正样本满足 OU 式平稳加性转移

$$z'=\rho z+\sqrt{1-\rho^2}\eta$$

且 alignment 与 Gaussianization 达到总体最优时，论文证明组合映射$h=f\circ g$只能是$h(z)=Qz$，$Q$为正交矩阵。若规划代价旋转不变，latent planning 与真实状态规划等价。

这是很强的充分条件，也是很强的限制。真实 Reacher 策略轨迹并不满足理想高斯/平稳条件，线性恢复明显下降；数据 stride 过小信号太弱，过大又暴露多峰转移。真正可迁移的结论是：**规划正确性依赖 latent 几何，而 latent 几何依赖数据采样分布。**

---

## 7. 2026 最新进展：瓶颈从“能不能学”转向“学得密不密、快不快、稳不稳”

### 7.1 V-JEPA 2.1：补回稠密时空信息

V-JEPA 2 擅长全局动作与语义，却不一定保留逐 patch、逐时刻可定位特征。V-JEPA 2.1 加入 Dense Predictive Loss、跨中间层 Deep Self-Supervision，并统一图像/视频 tokenizer。它在 NYUv2、Ego4D、EPIC-KITCHENS 等稠密与时序任务上提升，真实机器人 grasping 比 V-JEPA 2-AC 提高约 20 个百分点。

![](../../10-Papers/World-Models/assets/V-JEPA%202.1%20-%20视频稠密特征自监督学习/figures/02-method-pipeline.jpg)

> **图 13：V-JEPA 2.1。** 新目标不只预测 masked target，还给可见 context 和中间层提供密集监督。消融显示 context loss 可增强稠密能力却损害全局分类，deep self-supervision 用来缓和这组冲突。

这标志着 JEPA 评测从“线性分类是否强”转向“局部对应、深度、跟踪和操作是否保留空间结构”。

### 7.2 LeVJEPA：把 SIGReg 带到视频效率问题

LeVJEPA 用单 encoder、跨视图不变性与 SIGReg 取代 V-JEPA 的 EMA teacher 和受限 predictor，并通过随机 token 丢弃降低视频预训练计算。论文报告以约 5.6—20.8 倍更少计算匹配或超过 V-JEPA 2 的部分结果。

![](../../10-Papers/World-Models/assets/LeVJEPA%20-%20高效无启发式视频自监督学习/figures/02-compute-scaling.png)

> **图 14：LeVJEPA 的计算—性能曲线。** 它把“视频 JEPA 必须依赖 teacher-predictor masked prediction”变成可检验问题；但高稀疏率会损害运动对应，且大规模、稠密任务和真实规划尚未充分验证。

### 7.3 UniJEPA：统一光度、时间与动作条件世界建模

UniJEPA 试图把 IWM 的 photometric prediction、视频 temporal prediction 与动作条件后训练放进共享 encoder/predictor/latent。预训练端到端使用 next-embedding prediction 与 Gaussian regularizer，不依赖 EMA 或 frozen encoder；后训练后使用 latent MPC。

![](../../10-Papers/World-Models/assets/UniJEPA%20-%20统一任务无关视觉世界建模/figures/02-unified-architecture.png)

> **图 15：UniJEPA 统一架构。** 同一个 predictor 以光度变换、时间或真实动作作为 condition，试图在共享潜空间同时学习 invariance 与 equivariance。论文结果支持统一 recipe，但训练数据、模型细节和真实机器人验证仍不充分。

### 7.4 Flow-JEPA：用整轨迹 flow matching 对抗递归误差

LeWM 逐步回归未来，早期误差会被反复喂回。Flow-JEPA 改为给定当前 latent 与未来动作序列，从高斯源通过条件向量场联合生成整段未来 latent trajectory：

$$Z_\tau=(1-\tau)Z_0+\tau Z,\qquad\mathcal L_{\mathrm{FM}}=\|v_\theta(Z_\tau,\tau,z_t,A)-(Z-Z_0)\|_2^2$$

![](../../10-Papers/World-Models/assets/Flow-JEPA%20-%20流匹配的鲁棒JEPA潜动力学/figures/02-method.png)

> **图 16：Flow-JEPA。** 训练覆盖从随机轨迹到真实未来轨迹的插值邻域，可能学会把受扰 latent 拉回有效流形；整轨迹生成也消除一步递归链。平均 clean 成功率 86%→92%，局部视觉噪声 67%→86%。

最关键的保留意见是：论文同时改变“自回归一步”与“确定性回归”两件事，未用联合轨迹 MSE 等中间基线拆分贡献；8 次向量场调用使单规划约慢 3.8 倍。

### 7.5 SD-JEPA：显式分离任务进度与内容

Subspace-Decomposed JEPA（arXiv:2605.31111）把 latent 拆成低维 progression 子空间和高维 content 子空间，前者用 cosine-margin triplet loss，后者用 LeWM 的 SIGReg。论文报告仅约 4.2% 维度可解释 72%—95% 的回合内任务进度，并可用角度变化定位语义事件。

这条分支直接回应 slow-features 与 planning cost 的缺口：不是让所有 latent 维度同时服务所有目的，而是给“任务进行到哪里”一个专用坐标。局限是控制环境少、短 horizon，progression 维度与跨 episode 标定仍依赖环境。

### 7.6 AD-E2E-JEPA：把 dense patch planning 压到驾驶可评测规模

AD-E2E-JEPA 冻结 DINOv3 ViT-L，用 SIGReg 约束可学习 projector，把每帧$16\times32\times1024$压到$4\times8\times256$，再以相对位姿轨迹条件预测未来。256 个候选、8 帧 rollout 在 A100 上约 0.8 秒，相比 dense DINO-WM/JEPA-WM 快约两个数量级。

![](../../10-Papers/World-Models/assets/AD-E2E-JEPA%20-%20端到端自动驾驶潜空间规划/figures/02-method-pipeline.png)

> **图 17：AD-E2E-JEPA。** Projector 将 patch 数压缩 16 倍、维度压缩 4 倍；候选轨迹词表取代连续 CEM。其 goal 是数据中的真实未来图像，因此这是隔离 world-model 质量的 oracle 诊断，不是实际导航系统可直接获得的目标。

它还揭示“latent 终点更准”不等于“驾驶更安全”：规划代价若没有碰撞、道路边界和舒适性约束，模型可能在视觉潜空间接近目标却违反安全规则。8192 候选可进一步提分，但耗时约 18.2 秒，不能把 0.8 秒口径外推到最佳结果。

### 7.7 M-JEPA 与 VL-JEPA：系统优化和跨模态接口

M-JEPA（arXiv:2609.22674）不改学习目标，而是共享 mask-independent 计算、融合 token routing，并让 target encoder 只处理目标 token 并集；在 A100、2—10 masks 条件下最高约 1.7 倍端到端训练加速。它说明 JEPA 的新瓶颈已经进入执行系统：多 mask 的冗余前向、内存搬运和同步可以比损失本身更贵。

VL-JEPA（arXiv:2512.10942）把 vision-language 训练从自回归 token 生成改为目标文本 embedding 预测，需要显式文本时才调用轻量 decoder。论文报告在受控比较中可训练参数减少 50%，选择性解码次数降低 2.85 倍，并支持分类、检索和判别式 VQA。其边界是复杂开放生成、工具调用和长链语言推理尚未证明；它更像“语义预测接口”而不是完整替代生成式 VLM。

### 7.8 LeWAM：把世界预测、动作分布和规划收进同一端到端模型

[[LeWAM - 端到端JEPA世界动作模型]]沿着 LeWorldModel 的 shared encoder、next-latent prediction 与 SIGReg 继续前进，但不再把策略或求解器视作训练后的外接模块。它在同一个 Mixture-of-Transformers 中联合优化两条接口：clean actions 条件下预测未来 latent，noisy action chunks 条件下用 flow matching 学习多模态动作分布。动作监督因此能直接塑造视觉表征，模型也同时获得 consequence model 与 behavior model。

![](../../10-Papers/World-Models/assets/LeWAM%20-%20端到端JEPA世界动作模型/figures/01-training-overview.png)

> **图 18：LeWAM 的联合训练。** shared encoder 同时服务历史、目标与未来帧；MoT 的状态分支预测下一 latent，动作分支回归 action flow velocity，SIGReg 约束潜空间为各向同性高斯。关键变化是 world model 和 policy 不再分阶段训练。来源：LeWAM Figure 1。

部署时，LeWAM 可以直接把 action flow 解码为策略，也可以先从学到的动作分布提出多条 action chunks，再用冻结的 latent dynamics 做梯度细化与 best-of-N 选择。相比 LeWM 从宽高斯出发的 CEM，这使候选动作更接近数据支持，尤其适合接触模式切换明显的任务。论文报告四项接触丰富任务平均成功率由 28.6% 提高到 89.7%，100-step goal offset 的长时任务由 10.9% 提高到 46.2%，规划吞吐从 1.62 提升到 52.50 plans/s，约 32.3 倍。

这些数字不能被解读为专用策略已被淘汰：真实机器人三任务中，LeWAM 证明了端到端世界—动作联合建模的可行性与计算效率，但平均成功率没有超过 Diffusion Policy。更需要后续拆分的是：收益究竟来自 action flow 的多模态建模、动作监督对 encoder 的反向塑形、proposal initialization，还是 gradient refinement；同时还需在更长闭环、跨任务数据和安全约束下验证世界模型是否会被 planner 利用其误差。

---

## 8. Anti-collapse 机制的四次演化

| 阶段 | 机制 | 优点 | 根本限制 |
|---|---|---|---|
| 对比/负样本 | 拉近正样本、推开负样本 | 直观、判别性强 | 负样本与 batch 成本，可能编码实例身份 |
| EMA teacher + stop-grad | I-JEPA/V-JEPA 的非对称目标 | 大规模实践稳定 | 平凡解没有被数学排除，依赖 schedule 与架构 |
| 方差—协方差 | VICReg、MC-JEPA | 防常数与低秩坍塌 | 只看二阶矩，挡不住 cluster/nonlinear collapse |
| SIGReg 高斯化 | LeJEPA、LeWM、Flow/SD/UniJEPA | 随机一维完整分布匹配，单一主要权重 | 有限 batch/切片；高斯健康不保证任务相关或低本征维适配 |

一个关键结论是：**坍塌问题正在从“表示有没有方差”升级为“表示的几何是否适合预测与规划”。** 后续评测不能只报告 loss、rank 和 linear probe，还要报告真实状态可读性、动作可达性、rollout 稳定性以及 planner 是否会 exploit model error。

---

## 9. 主要路线横向对比

| 路线 | 代表工作 | Encoder 是否共同训练 | 预测对象 | 动作条件 | 可规划 | 最大优势 | 最大局限 |
|---|---|---:|---|---:|---:|---|---|
| 图像 masked JEPA | I-JEPA | 是 | masked patch latent | 否 | 否 | 语义强、训练高效 | mask 敏感、无时间 |
| 混合运动内容 | MC-JEPA | 是 | 内容 latent + 光流/像素约束 | 间接 | 否 | 联合运动与语义 | 多目标混杂 |
| 图像变换世界模型 | IWM | 是 | 变换后 latent | 光度参数 | 有限 | 研究等变性 | 非真实动力学 |
| 视频 masked JEPA | V-JEPA/2/2.1 | 是 | masked spatiotemporal latent | 2-AC 后训练 | 是 | 规模化视频理解 | 预训练本身多为非因果补全 |
| 冻结特征 WM | DINO-WM、AD-E2E-JEPA | 否 | 预训练 patch latent | 是 | 是 | 稳定、空间特征强 | 表征受外部模型限制，rollout 贵 |
| SIGReg 端到端 WM | LeWM、UniJEPA | 是 | next latent | 是 | 是 | recipe 简、单卡可复现 | 低维/复杂 3D/长时程有限 |
| 随机轨迹动力学 | Flow-JEPA | 是/可热启动 | 整段 latent trajectory | 是 | 是 | 噪声鲁棒、减递归误差 | 多次 NFE，机制未拆分 |
| 端到端世界—动作模型 | LeWAM | 是 | next latent + action flow | 是，联合训练 | 直接策略 + 规划 | 动作监督塑造表征，proposal 搜索快 | 真实机器人未超专用策略，组件贡献待拆分 |
| 分解潜空间 | SD-JEPA | 是 | progression + content | 是 | 是 | 可解释任务进度 | 环境依赖、规模小 |
| 视觉语言 JEPA | VL-JEPA | 是 | 文本语义 embedding | 问句/视频 | 非物理规划 | 减少 token 解码 | 开放生成与推理不足 |

---

## 10. 评测体系如何变化

### 10.1 从分类分数到“世界模型是否真的可用”

I-JEPA 时代主要看 ImageNet linear probe、few-shot 和迁移；V-JEPA 加入 K400/SSv2，区分 appearance 与 motion；V-JEPA 2.1 加入深度、跟踪和稠密时空定位；动作世界模型则必须报告 success rate、latent rollout error、planning latency 和真实状态 probe。

### 10.2 2026 年应至少报告的六组指标

1. **统计健康**：embedding 方差、effective rank、投影正态性与 collapse rate；
2. **状态可读性**：位置、姿态、接触、速度等线性/非线性 probe；
3. **动力学**：one-step 与 multi-step latent error，teacher-forcing 与 free rollout 差距；
4. **规划**：成功率、候选数、horizon、重规划频率、planner wall-clock；
5. **鲁棒性**：背景、光照、遮挡、任务物体噪声、动力学偏移和 OOD goal；
6. **安全与代价一致性**：latent 距离是否与碰撞、道路、舒适性或约束满足一致。

只报告 frozen classification 已不足以支持“世界模型”结论；只报告预测误差也不足，因为更低 MSE 可能来自信息压缩或慢特征捷径。

---

## 11. 当前仍未解决的核心问题

### 11.1 Anti-collapse 之后是 anti-shortcut

SIGReg 解决“分布退化”，却没有规定表示应该装什么。慢背景、相机 ID、场景纹理都可能形成健康高斯分布。需要对象级 masking、动作干预、跨轨迹负样本、因果因素或任务进度约束来排除这些捷径。

### 11.2 真实数据远离高斯与平稳假设

可识别性理论依赖高斯 latent 与平稳加性转移；真实机器人数据由策略选择，状态分布多峰、受目标和安全规则截断。如何通过重采样、局部 chart、mixture prior 或 normalizing flow 把理论推广到真实轨迹，是理论主线的下一步。

### 11.3 多模态未来如何表示

L1/L2 latent regression 倾向条件均值，Flow-JEPA 引入随机 trajectory 是第一步，但尚未回答不同未来 mode 怎样和动作、不可观测变量对应。JEPA 原始纲领中的 latent variable 与能量推理仍未成为主流实现。

### 11.4 潜空间距离不等于任务代价

Goal embedding 的欧氏距离简单高效，却可能忽略碰撞、拓扑不可达和安全边界。AD-E2E-JEPA 已清楚显示几何误差、命中率和安全指标可能不一致。未来需要可校准的 cost probe、约束模型或分层 subgoal，而不是只把$\|z-z_g\|^2$当万能目标。

### 11.5 长时层级规划仍未兑现 H-JEPA

现有工作多为 5—8 步或数秒 horizon，主要靠 MPC 频繁重规划抑制漂移。真正的分钟级任务需要高层 progression/subgoal、低层连续控制和跨尺度记忆；H-JEPA 的训练方式、层间目标与 uncertainty propagation 仍是空白。

### 11.6 统一模型还是专用子空间

UniJEPA 强调共享潜空间，SD-JEPA 强调功能分解；V-JEPA 2.1 发现 dense loss 与 global classification 有冲突。这些结果共同提示，未来不一定是“一个 homogeneous latent 解决全部任务”，更可能是共享底座上分出 geometry、content、progression、controllable state 和 language semantics 子空间。

### 11.7 训练和推理系统成本

M-JEPA 说明 mask routing 与 target-side 冗余是训练瓶颈；AD-E2E-JEPA 说明 patch 数、维度和候选数共同决定规划速度；Flow-JEPA 则引入 NFE。公平比较必须统一硬件、候选数、horizon、CEM 迭代和重规划次数，不能只报单次 forward。

---

## 12. 事实判断与趋势判断

### 12.1 已较为确立的事实

- 在图像和视频自监督中，预测学习到的表征通常比重建原始像素更利于 frozen/low-shot 迁移；
- EMA teacher 不是唯一可行 anti-collapse 方案，SIGReg 可在多架构上稳定替代；
- 预训练 patch 特征可以支持无 reward、无 task-specific policy 的视觉目标规划；
- 大规模 action-free 视频预训练可被少量 action-labeled 机器人数据转化为规划模型；
- dense feature、latent trajectory robustness 和 planning compression 已成为独立研究问题。

### 12.2 仍属于推测或早期证据的判断

- JEPA 会取代生成式世界模型：证据不足，两者更可能互补；
- 高斯 latent 是真实世界的普遍结构：理论并未如此声称；
- 互联网视频加少量机器人数据足以解决通用操作：目前只在有限任务和平台验证；
- 单一共享 latent 可以统一理解、语言、几何与控制：冲突已在 dense/global 任务中出现；
- latent planning 天然安全：AD-E2E-JEPA 反而说明必须显式建模安全代价。

### 12.3 我的总体判断

JEPA 最重要的贡献，不是提供了一个“反 LLM/反生成”的品牌，而是把世界模型设计问题重新拆成三层：**观测应压成什么状态、状态应怎样预测、预测怎样转成行动。** 2023 年解决了第一层的可行性，2024—2025 年把第二层扩到视频和动作，2026 年开始真正处理第三层所需的几何、鲁棒性、延迟和代价一致性。

当前最值得投入的路线不是继续只刷视频分类，而是把 LeJEPA 的统计健康、V-JEPA 2.1 的稠密时空状态、Flow-JEPA 的多步随机动力学、LeWAM 的世界—动作联合训练、SD-JEPA 的进度子空间和显式安全 cost 组合到统一评测中。LeWAM 已经证明动作监督可以从外接控制器变成塑造 latent 的训练信号，但它还没有回答多步随机世界、跨任务策略与安全约束能否在同一模型中稳定共存。若这些模块不能同时成立，JEPA 仍只是优秀表征学习器；若能成立，它才会成为 LeCun 2022 所说的可规划世界模型。

---

## 13. 推荐阅读顺序

### 13.1 第一轮：建立直觉

1. [[A Path Towards Autonomous Machine Intelligence - JEPA世界模型纲领]]：理解完整认知架构，不要从 I-JEPA 反推 JEPA 全貌；
2. [[JEPA - 联合嵌入预测架构]]：掌握 context/target/predictor、mask 与 latent prediction；
3. [[V-JEPA - 视频潜空间预测表征学习]]：看 feature prediction 如何跨到时空；
4. [[V-JEPA 2 - 自监督视频世界模型的理解预测与规划]]：理解“大规模观察 + 少量交互”的两阶段配方。

### 13.2 第二轮：理解失败模式与理论

5. [[Slow Features - JEPA的慢特征偏置]]：理解可预测性为什么会学错；
6. [[LeJEPA - 无启发式高斯正则自监督学习]]：掌握 SIGReg 与 anti-collapse；
7. [[LeJEPA - 世界模型线性可识别性理论]]：理解何时 latent 几何才可用于规划；
8. [[LeWorldModel - 像素端到端稳定JEPA世界模型]]：把理论 recipe 放进动作世界模型。

### 13.3 第三轮：看 2026 的真实瓶颈

9. [[V-JEPA 2.1 - 视频稠密特征自监督学习]]：全局语义与稠密空间的冲突；
10. [[LeVJEPA - 高效无启发式视频自监督学习]]：训练效率与单 encoder；
11. [[Flow-JEPA - 流匹配的鲁棒JEPA潜动力学]]：多步误差与分布鲁棒性；
12. [[UniJEPA - 统一任务无关视觉世界建模]]：统一任务接口；
13. [[AD-E2E-JEPA - 端到端自动驾驶潜空间规划]]：规划压缩、oracle goal 与安全边界；
14. [[LeWAM - 端到端JEPA世界动作模型]]：看 action flow、shared encoder 与 latent dynamics 如何端到端联合，并比较直接执行和规划两种部署模式；
15. [[DINO-WM - 预训练视觉特征上的零样本规划]]：作为 frozen-feature 强基线对照。

---

## 14. 文献索引

### 14.1 核心主线

- LeCun, 2022. *A Path Towards Autonomous Machine Intelligence*. OpenReview.
- Sobal et al., 2022. *Joint Embedding Predictive Architectures Focus on Slow Features*. arXiv:2211.10831.
- Assran et al., 2023. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243, CVPR 2023.
- Bardes et al., 2023. *MC-JEPA*. arXiv:2307.12698.
- Garrido et al., 2024. *Learning and Leveraging World Models in Visual Representation Learning*. arXiv:2403.00504.
- Bardes et al., 2024. *Revisiting Feature Prediction for Learning Visual Representations from Video*. arXiv:2404.08471.
- Zhou et al., 2024/2025. *DINO-WM*. arXiv:2411.04983, ICML 2025.
- Assran et al., 2025. *V-JEPA 2*. arXiv:2506.09985.
- Balestriero & LeCun, 2025. *LeJEPA*. arXiv:2511.08544.

### 14.2 2026 理论、世界模型与系统

- Mur-Labadia et al., 2026. *V-JEPA 2.1*. arXiv:2603.14482.
- Maes et al., 2026. *LeWorldModel*. arXiv:2603.19312.
- Klindt, LeCun & Balestriero, 2026. *When Does LeJEPA Learn a World Model?* arXiv:2605.26379.
- Thil et al., 2026. *Subspace-Decomposed JEPAs*. arXiv:2605.31111.
- Lanji et al., 2026. *UniJEPA*. arXiv:2608.07409.
- Balestriero et al., 2026. *LeVJEPA*. arXiv:2608.27395.
- Huo et al., 2026. *Flow-JEPA*. arXiv:2608.29029.
- Sanim et al., 2026. *Mask-Aware Execution for Efficient JEPA Training*. arXiv:2609.22674.
- Zhu et al., 2026. *AD-E2E-JEPA*. arXiv:2609.34085.
- Fu et al., 2026. *LeWAM: End-to-End World and Action Modeling with JEPAs*. arXiv:2610.12818.
- Meta AI, 2025/2026. *VL-JEPA*. arXiv:2512.10942.
