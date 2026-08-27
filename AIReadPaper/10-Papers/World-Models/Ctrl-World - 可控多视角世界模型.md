---
type: paper-note
status: draft
domain: World-Models
paper: "CTRL-WORLD: A Controllable Generative World Model for Robot Manipulation"
year: 2026
arxiv: 2510.10125
doi: null
source: https://arxiv.org/abs/2510.10125
project: https://ctrl-world.github.io
code: null
tags:
  - World-Models
  - VLA
  - World-Action-Model
  - Multi-View
  - Memory-Retrieval
  - Policy-Evaluation
  - Synthetic-Data
created: 2026-08-27
updated: 2026-08-27
---

# Ctrl-World：可控多视角生成式世界模型

## 基本信息

- **作者**：Yanjiang Guo*、Lucy Xiaoyang Shi*、Jianyu Chen、Chelsea Finn（* 共同一作）
- **机构**：斯坦福大学、清华大学交叉信息研究院（IIIS）
- **发表**：ICLR 2026（arXiv 2510.10125v3，2026-03-01）
- **项目页**：<https://ctrl-world.github.io>
- **补充视频**：<https://sites.google.com/view/ctrl-world>
- **原文 PDF**：<https://arxiv.org/pdf/2510.10125>

## 一句话结论

Ctrl-World 把预训练视频扩散模型 Stable Video Diffusion（SVD 1.5B）改造成一个**可控的多视角生成式世界模型**，专为「策略闭环 rollout（policy-in-the-loop）」设计：通过联合预测第三视角 + 腕部视角、帧级动作条件（笛卡尔空间位姿、厘米级精度）与位姿条件记忆检索，在 DROID（95k 轨迹、564 场景）上训练后，能对新场景新机位零样本生成超过 20 秒的时空一致轨迹。它第一次打通了「世界模型做策略评估 + 策略改进」的完整闭环：在想象空间里准确复现策略的真实指令跟随排名（相关性 y=0.87x），并用合成的成功轨迹做监督微调，把 π0.5 的成功率从 38.7% 提升到 83.4%（+44.7%）。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

通用机器人策略已经能执行大量操控技能，但在「不熟悉物体 + 新指令」下评估与改进仍然困难。严谨评估需要大量真实世界 rollout（跨任务、跨环境反复重复才能达到统计显著性），系统性改进则需要带专家标注的纠正数据——两者都慢、贵、难扩展。世界模型提供了一个可扩展的替代方案：让策略在**想象空间**里 rollout。但关键挑战是构建一个能支持「与通用策略做多步交互」的可控世界模型——它必须支持多视角预测、细粒度动作控制、以及长程一致交互，而此前的工作都做不到。

### 1.2 现有方法的瓶颈

此前 action-conditioned world model 大多停留在**被动视频预测**设定，不足以主动交互先进通用策略。论文指出三个关键局限：

- **只模拟单视角**：仅模拟一个第三人称视角会导致严重部分可观测性，进而产生幻觉（例如物体在没有物理接触前就"瞬移"进夹爪）；单视角输入也与许多要求「第三视角 + 腕部视角」同时输入的现代 VLA 策略不兼容；
- **缺乏细粒度控制**：无法捕捉高频动作的因果效应；
- **长程时间一致性差**：长时间视频生成中难以保持一致。

### 1.3 本文核心贡献

1. 提出 Ctrl-World，一个可控多视角生成式世界模型，把预训练的被动视频生成器转化为策略兼容的交互式模拟器；
2. 三大关键组件：**多视角联合预测**（含腕部视角，显著减少接触密集交互的幻觉）、**帧级动作条件**（让视觉动态与高频控制信号严格对齐）、**位姿条件记忆检索**（稀疏历史帧 + 位姿投影，稳定长程 rollout）；
3. 展示世界模型的两个新用途：**策略评估**（在想象空间里复现真实策略排名）与**策略改进**（合成成功轨迹做 SFT，把 π0.5 提升 +44.7%）。

---

## 2. 任务定义与输入输出

### 2.1 问题形式化

通用策略 $\pi$ 把多视角观测 + 语言指令映射为动作序列。观测 $o_t = [I_t^1,\dots,I_t^n,q_t]$ 含 $n$ 个相机视角与机器人位姿 $q_t$，给定指令 $l$，策略输出 $H$ 步动作 chunk：

$$a_{t+1}, a_{t+2}, \dots, a_{t+H} \sim \pi(\cdot \mid o_t, l)$$

目标是训练一个世界模型 $W$，预测执行动作 chunk $A_t=[a_{t+1},\dots,a_{t+H}]$ 中每一步后的未来多视角观测：

$$o_{t+1}, \dots, o_{t+H} \sim W(\cdot \mid o_t, A_t)$$

最终预测 $o_{t+H}$ 再送回策略 $\pi$ 生成下一个动作 chunk $A_{t+H}\sim\pi(\cdot\mid o_{t+H},l)$。策略与世界模型**自回归交互**，即可在想象空间内完成长程 rollout。

### 2.2 输入、输出与假设

- **输入**：多视角观测 $o_t$（3 个相机视角 + 机器人位姿）、历史帧、动作 chunk（转成笛卡尔空间位姿）、语言指令 $l$；
- **输出**：未来 $H$ 步多视角观测 $o_{t+1:t+H}$；
- **核心假设**：只要世界模型能「可靠、紧密地跟随动作输入」且「长程保持一致」，就能作为策略的交互式模拟器，支持评估与改进。

---

## 3. 核心方法

### 3.1 总体框架

Ctrl-World 以 SVD（1.5B）为骨干，只新增一个动作投影 MLP（其余参数沿用预训练），再用扩散损失微调。三大适配（Figure 2）针对性地解决「多视角」「可控性」「长程一致」三个问题。

![](assets/Ctrl-World%20-%20可控多视角世界模型/figures/01-teaser.png)

> **图 1：框架总览（Figure 1）。** Ctrl-World 专为通用策略的 policy-in-the-loop rollout 设计：生成多视角联合预测（含腕部视角）、通过帧级条件实现细粒度动作控制、借助位姿条件记忆检索维持长程动力学一致。这三者支撑了两种应用——在想象空间里准确评估策略（与真实 rollout 对齐），以及用合成轨迹定向改进策略（+44.7%）。
> 来源：论文 Figure 1，第 2 页，<https://arxiv.org/abs/2510.10125>

### 3.2 多视角联合预测

现代 VLA（π0.5、HybridVLA 等）常依赖多个第三人称相机提供全局上下文 + 腕部相机提供精确交互。为匹配这种输入格式，世界模型必须在每一步生成跨视角空间一致的预测。沿用前馈 transformer 可扩展地捕捉多视角空间关系（Wang et al., 2025），论文把 $N$ 张输入图像（每张 $H\times W$ token）沿 token 维拼接，**联合预测所有视角** $o_{t:t+H}$。实验发现多视角联合预测既提升一致性，又显著减少幻觉。

### 3.3 位姿条件记忆检索（Memory Retrieval）

世界模型的预测误差会在长程 rollout 中累积，导致漂移与不连贯。为缓解，论文把**历史帧**加入上下文——为避免上下文过长，以步长 $m$ 采样 $k$ 个历史帧，使模型预测 $o_{t+1:t+H}\sim W(\cdot\mid o_{t-km},\dots,o_t,l)$。此外，把对应的机械臂位姿 $[q_{t-km},\dots,q_t]$ 通过 spatial transformer 内的**帧级 cross-attention** 嵌入到历史帧 $[o_{t-km},\dots,o_t]$ 中——让模型用「位姿」去识别过去的相关帧，把未来预测锚定到相似的历史状态。

### 3.4 帧级动作条件（Frame-level Action Conditioning）

预训练视频模型只以文本和图像为条件，控制精度有限。为实现完全可控，模型额外以策略输出的动作序列 $[a_{t+1:t+H}]$ 为条件；并把每条动作序列转成**笛卡尔空间机械臂位姿** $[a'_{t+1:t+H}]$，与历史位姿 $[q_{t-km},\dots,q_t]$ 拼接。在 spatial transformer 内做帧级 cross-attention（Zhu et al., 2024; He et al., 2025），让每帧的视觉 token attend 到其关联的位姿嵌入——历史帧对应 $[q_{t-km},\dots,q_t]$，未来帧对应 $[a'_{t+1:t+H}]$。

![](assets/Ctrl-World%20-%20可控多视角世界模型/figures/02-method-pipeline.png)

> **图 2：Ctrl-World 架构（Figure 2）。** 以预训练视频扩散模型初始化，经三大适配改造成可控、时间一致的世界模型：(1) 多视角输入 + 联合预测；(2) 记忆检索机制——在上下文中加入稀疏历史帧、通过帧级 cross-attention 把位姿投影进每帧，把预测重新锚定到相似历史状态；(3) 帧级动作条件——把高频动作与视觉动态对齐。图中展示了 spatial transformer（B×P,T,C）与 temporal transformer（B×T,P,C）的 token 组织，以及 history poses + action chunk poses、CLIP 语义 token、稀疏历史、加噪未来的拼接结构。
> 来源：论文 Figure 2，第 4 页，<https://arxiv.org/abs/2510.10125>

### 3.5 训练目标与实现

初始化自 SVD 1.5B，仅新初始化一个 3 层 MLP（把 7 维笛卡尔动作投影到 1024 维 latent）。图像先经空间下采样 8×8 的 VAE 编码，$k=7$ 个历史帧各自加独立随机噪声以增强鲁棒性；动作条件窗口 1 秒（15 步），转成笛卡尔空间后时间下采样到 5 步再输入。每帧 3 个 192×320 视角编码成 $24\times40$ latent，总输入 token 形如 $B\times(7+5)\times(3\times24\times40)$。扩散训练目标为：

$$\mathcal{L} = \mathbb{E}_{x_0,\epsilon,t'}\left\| \hat{x}_0(x_{t'}, t', c) - x_0 \right\|^2$$

其中 $x_0=o_{t+1:t+H}$ 为预测目标，加噪得 $x_{t'}=\sqrt{\alpha_{t'}}x_0+\sqrt{1-\alpha_{t'}}\epsilon$；模型输入为历史 token 与加噪未来的拼接 $[o_{t-km},\dots,o_{t-1},o_t,x_{t'}]$，条件 $c=[q_{t-km},\dots,q_t,a'_{t+1:t+H},o_{t-km},\dots,o_t]$。训练用 2×8 H100、总 batch 64、学习率 $1\times10^{-5}$、100k 步，约 2–3 天。

### 3.6 策略评估与改进流程

- **策略评估**：给定初始观测 $o_0$ 与指令 $l$，策略 $\pi$ 与世界模型 $W$ 交互生成合成轨迹 $\tau$；每条轨迹按人工偏好标注成功/失败（用 VLM 做 reward model 留作未来工作）；
- **策略改进（Algorithm 1）**：固定初始观测与指令时策略行为高度确定（总抓同一个物体），因此引入**结构化扰动**鼓励多样性——(i) 改写指令（VLA 策略可被不同指令引导出不同行为），(ii) 在世界模型内把机械臂重置到随机初始状态。从下游任务集 $[l_0,\dots,l_M]$ 收集合成 rollout，按人工偏好筛选成功轨迹入合成集 $D_s$，最后用 $L_\theta=\mathbb{E}_{o_t,a_{t:t+H}\sim D_s}\|\pi_\theta(o_t,l)-a_{t:t+H}\|^2$ 微调策略。

---

## 4. 数据集与实验设置

### 4.1 平台与数据集

- **DROID 平台**：Franka Panda 机械臂 + Robotiq 夹爪，含 1 个腕部相机 + 2 个随机摆放的第三人称相机；
- **DROID 数据集**：95,599 条轨迹、564 个场景，含约 76k 成功 + 19k 失败轨迹。多样动作与失败数据的覆盖，是训练「可模拟多种未来」的可控世界模型的关键。

### 4.2 训练细节

模型联合预测 3 个相机（各 192×320）；以 7 帧历史为条件（帧间隔 1–2 秒）；以未来 15 个动作（对应 DROID 里 1 秒动作 chunk）为条件。交互时若策略输出不足 15 步则用 dummy 动作补齐、只取有效动作的预测。训练 2×8 H100、batch 64、约 2–3 天。

### 4.3 Baseline 与评价指标

对比两个先前的 action-conditioned 世界模型：WPE（World-model-based Policy Evaluation）与 IRASim。由于它们只预测单视角，公平起见另训单视角版 Ctrl-World-third-view。评价：留 2% 轨迹作验证集，随机采样 256 个 10 秒视频片段，自回归预测 10 轮（每轮 15 步 = 1 秒），用计算指标（PSNR、SSIM）与模型指标（LPIPS、FID、FVD）对比。

---

## 5. 实验结果

### 5.1 世界模型质量（Table 1）

| 方法 | 视角 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|---|---|---|---|---|---|
| WPE-Single-View | third | 20.33 | 0.772 | 0.131 | 156.4 |
| IRASim-Single-View | third | 21.36 | 0.774 | 0.117 | 138.1 |
| Ctrl-World-Single-View | third | 21.27 | 0.793 | 0.110 | 127.5 |
| **Ctrl-World（多视角）** | third | **23.56** | **0.828** | **0.091** | **97.4** |

Ctrl-World 单视角版已超越 WPE / IRASim，多视角联合预测进一步把 PSNR 提升到 23.56、FVD 降到 97.4。定性上（Figure 3），单视角方法 WPE、IRASim、Ctrl-World-third-view 都无法移动绿色毛巾或抓住红色碗（幻觉），而 Ctrl-World 通过联合预测腕部视角（提供接触事件与物体状态变化的细粒度信息）精确建模了机器人与物体的交互。

![](assets/Ctrl-World%20-%20可控多视角世界模型/figures/03-long-horizon-rollouts.png)

> **图 3：长程 rollout 定性结果（Figure 3）。** 对比 ground truth、IRASim 与 Ctrl-World。此前模型依赖单视角预测，遭遇部分可观测与幻觉（如无法移动绿毛巾或抓红碗）；Ctrl-World 联合预测第三视角 + 腕部视角，产生与 ground truth 对齐的精确未来轨迹。
> 来源：论文 Figure 3，第 6 页，<https://arxiv.org/abs/2510.10125>

### 5.2 可控性与一致性

**可控性（Figure 4）**：不同动作序列（相差仅几厘米，如「Z 轴 -6cm」「Z 轴 +6cm」）能产生不同的 rollout，达到厘米级精度。论文把可控性归因于两点：DROID 数据集的稠密动作空间覆盖 + 多视角预测与帧级动作条件。注意力可视化显示，预测 $t=4s$ 帧时模型强烈关注 $t=0s$ 处位姿相似的帧，验证了记忆检索的有效性。

**一致性（Figure 5）**：腕部相机视场在单条轨迹内剧烈变化，是长程一致预测的难点。Ctrl-World 同时利用「其他视角的信息」与「历史帧」生成一致的腕部视角预测（绿框 = 从其他视角推断，红框 = 从记忆检索）。

**消融（Table 2）**：去掉 memory、frame-level cond 或 joint pred 均导致指标下降——尤其腕部视角，去掉 frame-level cond 后 PSNR 从 19.18 掉到 15.69、FVD 从 127.1 恶化到 179.1；去掉 joint pred 后 PSNR 掉到 15.94。

![](assets/Ctrl-World%20-%20可控多视角世界模型/figures/04-controllability.png)

> **图 4：可控性与消融（Figure 4）。** 不同动作序列在 Ctrl-World 里产生厘米级精度的不同 rollout。去掉 memory 导致预测模糊（蓝框），去掉帧级位姿条件降低控制精度（紫框）。左侧注意力可视化显示预测 $t=4s$ 帧时强烈关注 $t=0s$ 同位姿帧，证明记忆检索有效。每个动作 chunk 用自然语言表达（如「Z 轴 -6cm」）。
> 来源：论文 Figure 4，第 7 页，<https://arxiv.org/abs/2510.10125>

### 5.3 世界模型做策略评估

论文自建 DROID 平台并随机摆放两个第三人称相机，评估三个公开策略 π0、π0-FAST、π0.5 在 7 类任务（Pick-and-Place、Fold-Towel、Drawer、Wipe-Table、Close-Laptop、Pull-tissue、Stack）上的表现，真实与想象 rollout 用相同初始观测。

**核心结论**：策略的高层**指令跟随**行为在世界模型里与真实世界高度相关（线性拟合 $y=0.87x-0.04$），但**执行成功率**被系统性低估（$y=0.81x-0.11$）。缺口来自复杂物理动力学（碰撞、滑动、旋转）的建模不精确（如 Figure 6 中关笔记本的交互不准确），以及「通用策略在真实世界失败后会反复重试、世界模型有时没捕捉到」。论文预期采集更多 in-domain 策略 rollout 数据能缩小此差距。

![](assets/Ctrl-World%20-%20可控多视角世界模型/figures/05-policy-eval-correlation.png)

> **图 5：真实与想象 rollout 的相关性（Figure 7）。** 世界模型可靠捕捉指令跟随行为（$y=0.87x-0.04$），但倾向于低估执行成功率（$y=0.81x-0.11$）。三个策略（π0、π0-fast、π0.5）在 7 类任务上、真实 vs 世界模型的指令跟随率与成功率散点 + 回归线清晰展示了这一结论。
> 来源：论文 Figure 7，第 8 页，<https://arxiv.org/abs/2510.10125>

### 5.4 世界模型做策略改进（核心结果）

以 π0.5 为基座策略，按 Algorithm 1 通过「改写指令（调 LLM API 改写，如 "place glove in box" → "pick up the cloth and put it inside the box"）」或「重置机械臂初始状态」鼓励多样性。每任务生成 400 条轨迹，保留 25–50 条成功轨迹（人工偏好判断），微调 2k 步（4 H100）。

**结果（Figure 9）**：π0.5 预训练模型在陌生物体与新指令上成功率低，后训练把下游任务成功率从 **38.7% 提升到 83.4%（+44.7%）**。四类任务中：Spatial Understanding 0.29→0.88、Shape Understanding 0.44→0.91、Towel-Folding-Direction 0.57→0.80、Novel Objects 0.25→0.75（详见 Table 4–7，其中 Spatial 分方向的最难「Right Top」从 0.05 提升到 0.90）。

![](assets/Ctrl-World%20-%20可控多视角世界模型/figures/06-policy-improvement.png)

> **图 6：策略改进（Figure 8、9）。** 上排为后训练任务示例（空间理解、叠毛巾方向、形状理解、新物体），下排为世界模型内生成的合成轨迹。世界模型能生成成功与失败两种 rollout，保留成功轨迹做策略微调。柱状图（Figure 9）显示合成数据后训练把平均成功率从 0.39 提升到 0.83，即 +44.7%。
> 来源：论文 Figure 8、9，第 9 页，<https://arxiv.org/abs/2510.10125>

---

## 6. 与相关工作的关系

按论文 Related Works 口径梳理：

- **vs 视频生成模型用于机器人（WPE、IRASim、Genie Envisioner、AdaWorld、Unified World Models）**：多数是 action-conditioned 视频预测或「视频生成 + 合成动作标签」，但只做单视角、被动预测，不足以交互先进策略。Ctrl-World 用「多视角 + 帧级动作 + 记忆检索」补足，第一次实现与 π0 系 SOTA 策略的 policy-in-the-loop 交互；
- **vs 视频模型直接当策略（UniPi、VPP、GR-1 等）**：这条线把视频模型当策略骨干解码动作；Ctrl-World 反过来把视频模型当**模拟器**，用于评估与改进策略，方向互补；
- **vs 未来预测 co-training（PAD、UP-VLA、Unified World Models 等）**：把未来预测目标并入策略 co-training；Ctrl-World 单独训世界模型，不混入策略；
- **vs 传统 action-conditioned 世界模型（Dreamer、TD-MPC、IRIS、Daydreamer）**：这些多为任务专用、低维或图像状态空间的模型；Ctrl-World 是通用、多任务、多视角的扩散世界模型；
- **与 VPP / BagelVLA 同属「清华 IIIS + 陈建宇组」脉络**：郭彦江是共同一作，且 Ctrl-World 与 VPP（同为 SVD 骨干）共享「把预训练视频扩散模型迁移到机器人」的技术底座，但 VPP 用视频模型做策略、Ctrl-World 用视频模型做模拟器，是同一底座的两个互补用途。

---

## 7. 局限与批判性评价

1. **低层执行建模不精确**：碰撞、物体滑动、旋转等复杂物理动力学建模不准确（如关笔记本交互不精确），导致执行成功率被低估（$y=0.81x-0.11$）——这是「世界模型评估」可信度的最大短板；
2. **无法捕捉策略的「重试」行为**：真实世界里策略失败后会反复重试，世界模型有时没捕捉到这一行为模式，进一步拉大真实 vs 想象的执行差距；
3. **依赖 DROID 数据分布**：尽管包含 19k 失败轨迹，仍有很多失败模式在数据分布之外，泛化到全新失败模式的能力有限；
4. **对初始观测敏感、长程推理会失败**：论文自承模型在精确交互与长程推理任务上会失败，性能对初始观测敏感；
5. **改进范围有限**：实验聚焦「指令跟随」改进，模型精度不足以改进「已见指令的低层成功率」；迭代式 rollout + 微调是留待未来的方向；
6. **奖励信号依赖人工**：轨迹成功/失败靠人工偏好判断（论文也承认可用 VLM reward model 自动化但未实现），规模化成本高；
7. **单平台验证**：仅在 DROID（Franka Panda）上验证，跨本体、跨任务的泛化尚未证明。

---

## 8. 复现与实践建议

- **算力门槛**：训练 2×8 H100、batch 64、100k 步、约 2–3 天；策略微调 4 H100、2k 步。推理需 SVD 1.5B + VAE，显存适中；
- **底座依赖**：初始化自 SVD 1.5B 预训练 checkpoint，唯一新组件是 3 层 MLP（7 维笛卡尔动作 → 1024 维 latent）；
- **数据准备**：DROID 数据集（95k 轨迹）公开可下载；需把关节速度策略输出转成笛卡尔空间位姿（论文训了一个两层 MLP adapter + Franka Panda FK 做关节→笛卡尔桥接）；
- **实现要点**：
  - $k=7$ 历史帧、帧间隔 1–2 秒、动作窗口 1 秒（15 步，下采样到 5 步）；
  - 历史帧各自加独立随机噪声增强鲁棒性；
  - 帧级 cross-attention 是「可控性 + 记忆检索」的公共实现，务必在 spatial transformer 内做；
- **常见踩坑**：单视角会严重幻觉（物体瞬移进夹爪），务必联合预测腕部视角；与策略交互时动作 chunk 不足 15 步需补 dummy 动作、只取有效步预测。

---

## 9. 个人启发与后续问题

1. **世界模型从「数据引擎」升级为「评估 + 改进」闭环**：此前世界模型多用于合成训练数据（DreamGen、Genie Envisioner），Ctrl-World 第一次系统展示「策略评估」（排名对齐）+「策略改进」（合成 SFT）两个用途，且都给出量化证据（y=0.87x、+44.7%）。这为「廉价、可扩展的策略迭代」提供了可复用的范式；
2. **「指令跟随」比「执行成功」更容易被世界模型捕捉**：高层语义行为高度相关、低层物理执行被低估，这一发现点出了世界模型评估的**可信边界**——它适合做语义层面的策略筛选，不适合做精细执行的成功判定；
3. **多视角（尤其腕部视角）是接触密集交互的关键**：腕部视角的细粒度信息对「物体状态变化」建模至关重要，去掉后 FVD 从 127 恶化到 158——这印证了「观测完备性决定世界模型上限」；
4. **位姿条件记忆检索是可迁移的设计**：把「位姿」当作检索 key 去锚定相似历史帧，这一机制与 BagelVLA 的 RFG（当前帧先验）、VPP 的一次前向共享「结构化先验降低预测难度」的直觉，值得在其它长程世界模型里复用；
5. **未解问题**：（i）能否用 VLM reward model 自动化成功判定以规模化？（ii）跨本体、跨场景的世界模型何时成立？（iii）迭代式 rollout + 微调（用世界模型自身生成的数据反复改进）能否形成 self-improvement 闭环？（iv）低层物理执行建模的 gap 如何靠更强视频骨干（Genie 3、Cosmos）缩小？

---

## 10. 材料来源

### Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2510.10125> | 全文（18 页 + 附录） | 2026-08-27 | 精读主来源 |
| `figures/01-teaser.png` | Figure 1：框架总览 | arXiv:2510.10125 | 第 2 页 | 2026-08-27 | 概念对比 |
| `figures/02-method-pipeline.png` | Figure 2：Ctrl-World 架构 | arXiv:2510.10125 | 第 4 页 | 2026-08-27 | 核心方法 |
| `figures/03-long-horizon-rollouts.png` | Figure 3：长程 rollout 定性结果 | arXiv:2510.10125 | 第 6 页 | 2026-08-27 | 世界模型质量 |
| `figures/04-controllability.png` | Figure 4：可控性与消融 | arXiv:2510.10125 | 第 7 页 | 2026-08-27 | 可控性/记忆检索 |
| `figures/05-policy-eval-correlation.png` | Figure 7：真实 vs 想象相关性 | arXiv:2510.10125 | 第 8 页 | 2026-08-27 | 策略评估 |
| `figures/06-policy-improvement.png` | Figure 8/9：策略改进 | arXiv:2510.10125 | 第 9 页 | 2026-08-27 | 策略改进 |
| 项目页 <https://ctrl-world.github.io> | 视频/补充材料 | — | — | 2026-08-27 | 实验视频 |
