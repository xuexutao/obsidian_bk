---
type: paper-note
status: draft
domain: VLA
paper: X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model
year: 2025
arxiv: 2510.10274
source: https://arxiv.org/abs/2510.10274
project: https://thu-air-dream.github.io/X-VLA/
code: null
tags:
  - VLA
  - Soft-Prompt
  - Flow-Matching
  - Cross-Embodiment
  - Tsinghua-AIR
  - Shanghai-AI-Lab
created: 2026-08-31
updated: 2026-08-31
---

# X-VLA：软提示驱动的可扩展跨具身视觉-语言-动作模型

## 基本信息

- **作者**：Jinliang Zheng、Jianxiong Li、Zhihao Wang、Dongxiu Liu、Xirui Kang、Yuchun Feng、Yinan Zheng、Jiayin Zou、Yilun Chen、Jia Zeng、Ya-Qin Zhang、Jiangmiao Pang、Jingjing Liu、Tai Wang、Xianyuan Zhan（通讯：taiwang.me@gmail.com、zhanxianyuan@air.tsinghua.edu.cn）
- **机构**：AIR 清华大学（Institute for AI Industry Research, Tsinghua University）、上海人工智能实验室（Shanghai AI Lab）、北京大学
- **发表**：arXiv 2510.10274（v1，2025-10-11 提交，2025-10-14 修订），33 页技术报告
- **项目页**：<https://thu-air-dream.github.io/X-VLA/>
- **代码**：论文未公布官方仓库（项目页提供 demo 视频与可视化）
- **原文 PDF**：<https://arxiv.org/pdf/2510.10274>
- **License**：CC BY-NC-ND 4.0

## 一句话结论

X-VLA 用一套**「Soft Prompt + 标准 Transformer encoder 堆叠」**的极简 flow-matching VLA 架构，把跨具身异构性（异构动作空间、相机配置、任务分布）交给**每个数据源一组可学习的软提示**去吸收：预训练阶段联合训练 backbone 与 soft prompts，新本体适配时先 prompt warm-up 再联合微调，最终用 0.9B 参数在 6 个仿真基准（单臂、双臂、自动驾驶）+ 3 个真机平台 + 灵巧叠衣任务上同时刷 SOTA，且 1% 参数（9M LoRA）即可在 LIBERO 达到 93% / Simpler-WidowX 54%，与 π0（3B / 94.2% / 55.7%）持平。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

通用 VLA 模型的成功高度依赖**大规模、跨具身、异构**的机器人数据集预训练（O'Neill 2024、Lin 2025、π0 训练数据覆盖 7 种本体）。然而这恰恰也是它最大的瓶颈：不同机器人在硬件配置、动作空间、相机视角、任务分布上存在严重的**异构性（heterogeneity）**。这种异构性不仅体现在「末端执行器 6-DoF vs 双臂 14-DoF」这种动作空间差异上，还体现在固定相机 vs 腕部相机的视觉域差异、人类遥操作 vs 自主采集的策略差异上。直接把这些数据喂给一个共享的 VLA，会带来**分布漂移与语义错位**（Wang 2024c、Liu 2025b、Zheng 2025），导致预训练不稳定、跨本体迁移效果差。

X-VLA 要回答的核心问题是：**如何在「不强迫所有数据共享一个动作表示」的前提下，让一个模型既能从异构数据中学到通用模式，又能在新本体上快速适配？**

### 1.2 现有方法的瓶颈

论文把已有方法按「怎么处理异构性」分成三族（Fig 2），并明确指出每族的局限：

- **(a) 不做处理（None）**：直接把异构数据混在一起训。表征互相干扰，跨本体迁移差；
- **(b) HPT-style 投影（Heterogeneous Pre-trained Transformers 风格）**：在骨干中间插入一组**按硬件类型区分的投影层**（Input / Output Projection Library）。这相当于为每个本体训练独立的 IO adapter，结构复杂、参数量大、且与 VLM 内部表征耦合不好，训练不稳定；
- **(c) 语言提示（Language prompts）**：用硬件配置的自然语言描述作为 prompt 注入。依赖 VLM 的语言理解能力来"翻译"硬件配置，但 VLM 内部表示本身不适合作为精确硬件描述的锚点，在实践中**可扩展性与稳定性都差**（Reuss 2025、Driess 2025 也指出过类似问题）。

X-VLA 提出 **(d) Soft prompts**：为**每个数据源**学一组**随机初始化、端到端学习**的 embedding（`Φ ∈ R^{k}`），注入到动作生成的早期阶段；它不需要手工模板，参数极少（仅占全部参数的 0.04%），且独立于 VLM 内部表征，训练稳定。

### 1.3 本文核心贡献

1. **Soft Prompt 机制**：把「处理跨具身异构性」这一任务从"加结构 / 加 adapter"换成"加一组可学习 embedding"，参数开销极小（每个数据源一组 prompt，总量仅 0.04%），但能稳定吸收异构性；
2. **极简的 Soft-Prompted Transformer 架构**：抛弃「VLM 骨干 + 复杂动作头」的堆叠范式，改用**预训练 VLM 视觉编码器（Florence-Large）+ 共享 wrist 视骨干 + proprio/action 早融合线性层 + 标准 Transformer encoder 堆叠**。整模型 0.9B，全部由 24 层标准 self-attention block 组成；
3. **完整工程化训练 recipe**：统一动作表示（EEF xyz + Rotate6D + 离散 gripper）、意图抽象（30 个 anchor point / 4 秒）、balanced 采样、prompt warm-up + joint adaptation 两阶段适配；
4. **全面 SOTA**：0.9B 模型在 6 个仿真基准（Simpler、LIBERO、Calvin、RoboTwin-2.0、VLABench、NAVSIM）+ 3 个真机平台（BridgeData-v2 协议）+ Soft-Fold 灵巧叠衣任务（1,200 演示、~100% 成功率、33 次/小时）上同时刷 SOTA；1% 参数（9M LoRA）PEFT 即可逼近全参数 π0。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：多视角图像 $I = \{img_i\}$、语言指令 $L$、本体感受状态 $R_t$（关节位置 / 末端执行器位姿）。图像分主视图（fixed-view，由预训练 VLM 编码）与腕部视图（wrist-view，由共享 ViT 编码）；
- **输出**：未来 30 个 anchor point 构成的动作 chunk $\{a_n\}_{n=t}^{t+29}$，覆盖 4 秒轨迹；
- **核心假设**：异构性可以被「每个数据源一组 soft prompt」完美吸收；backbone 自身保持"embodiment-agnostic"，遇到新本体时只需要换 prompt + 少量微调即可适配。

### 2.2 关键符号与目标函数

Soft prompt 记为 $\Phi_H = \{p_h\}_{h \in H}$，其中 $H$ 是训练数据集集合，每个 $p_h \in R^{k}$ 是该数据源专属的可学习 embedding。Soft prompt 在动作生成早期被注入到 transformer 中。

backbone $\pi_\theta$ 的训练采用 **conditional flow matching**（Lipman 2023 / Rectified Flow, Liu 2023）的行为克隆版本。给定观测 $o_n$ 与目标动作 $a_1$，从噪声 $a_0 \sim \mathcal{N}(0, I)$ 出发构造线性插值 $a_s = s a_1 + (1 - s) a_0$，让 $\pi_\theta$ 回归速度场 $(a_1 - a_0)$：

$$
\mathcal{L}_{\text{FM-BC}}(\theta) = \mathbb{E}_{s,\, a_0,\, a_1} \left\| v_\theta(a_s, s, o_n, \Phi_H) - (a_1 - a_0) \right\|^2
$$

注意：flow matching 输出的是**连续动作向量**，不经过离散化。动作的连续化（位置、旋转）与离散化（gripper 的开/合）分开监督：position/rotation 用 MSE，gripper state 用 BCE。推理时从 $a_0 \sim \mathcal{N}(0,I)$ 出发用 ODE 积分（Euler）推出 $a_1$。Soft prompt 与 backbone 在 Phase I 预训练阶段联合优化。

---

## 3. 核心方法

### 3.1 总体框架

X-VLA 的总体框架（Figure 1 / 详细架构见下方）可以概括为「**Soft Prompt Library 查询 + 异构编码 + 标准 Transformer 堆叠**」。给定一个数据源 ID $h \in H$，先从 Soft Prompt Library 中查询对应的 prompt embedding $p_h$，再把多视角图像、语言、proprio/flow-time 与 soft prompt 一起送入 24 层标准 self-attention block，最后用一个线性头输出连续动作 chunk。**没有 VLM 端到端微调、没有复杂的动作头，只有"软提示 + 堆 transformer"**——这是 X-VLA 与 π0、OpenVLA、GR00T N1 等"大 VLM + 复杂头"路线最大的区别。

![](assets/X-VLA%20-%20软提示跨具身视觉语言动作模型/figures/01-teaser.png)

> **图 1：X-VLA 总体框架（teaser）。** 用 per-dataset soft prompt 吸收跨具身异构性，叠 N 层 soft-prompted self-attention block，配合预训练 VLM 视觉编码器，即可构建一个简洁可扩展的跨具身 VLA。
> 来源：论文 Figure 1，第 1 页，<https://arxiv.org/abs/2510.10274>

### 3.2 Soft Prompt Library

Soft Prompt Library 是一组「**按数据源 ID 索引的可学习 embedding 字典**」$\Phi_H = \{p_h\}$，每个 $p_h \in R^k$ 对应一个数据源（数据集 / 机器人配置）。它的关键设计：

- **不依赖手工模板**（区别于 language prompt）：$p_h$ 随机初始化，端到端优化；
- **注入到动作生成的早期阶段**：与 control tokens（noise action chunk + proprio + flow time 投影得到的 token）一起进入 transformer 之前就被加上，从而让 backbone 在第一层就感知到"当前数据来自哪个具身"；
- **参数量极小**：所有 prompt 加起来仅占全部参数的 **0.04%**；
- **可作为适配新本体的入口**：Phase II 适配只需新增一组 $p_{new}$，backbone 冻结。

### 3.3 详细模型架构

详细架构图清晰地展示了 X-VLA 的前向路径（Figure 2 / §4 详细架构，第 7 页）：

- **High-dimensional observation stream**：
  - 语言 + 主视图（fixed-view）由 **预训练 VLM 编码器**（X-VLA 用 **Florence-Large**）联合处理；
  - 腕部视图（wrist-view）由**共享 ViT 视觉编码器**单独处理（"Shared ViT"）。设计上把腕部视与语言解耦，是因为腕部相机"高噪声、快变化"，更适合"提供精细操作线索"而非"参与高层语义推理"；
  - 经过 Pretrained VLM 后的语言 / 主视图 token 与经过 Vision Encoder 的腕部 token 通过 **Shared Linear Projection** 统一投影到同一 token 空间，与 soft prompt 拼接为 **Multi-modal Tokens**；
- **Low-dimensional proprio–action stream**：
  - 噪声动作 chunk + proprio state + flow time embedding 经 **Repeat & Concat** 拼接为一个序列，经 **Input Projection** 投影为 **Control Tokens**；
- **Backbone**：Soft Prompt + Multi-modal Tokens + Control Tokens 一起进入 **Standard Self-attention Transformer Block × N**（N=24，hidden=1024），即 24 层 vanilla self-attention；
- **Output**：最后一层隐藏态经 **Output Projection** 投影回动作空间，得到 **Action Chunk**（未来 30 个 anchor point）。

![](assets/X-VLA%20-%20软提示跨具身视觉语言动作模型/figures/02-architecture.png)

> **图 2：X-VLA 详细模型架构。** 上半部分是 Soft Prompt Library：按 Dataset id 查询得到 Soft Prompt（外加 Input/Output Linear Projection Library，三者共占全部参数 0.04%）。下半部分是 Model Forward Path：Pretrained VLM 处理 Lang+Main view，共享 ViT 处理 Other views，noise action chunk + Proprio + Flow t 经 Input Projection 变成 Control Tokens，soft prompt + multi-modal tokens + control tokens 进入 N 层标准 self-attention block，再经 Output Projection 输出 Action Chunk。
> 来源：论文 §4.1 详细模型架构图，第 7 页，<https://arxiv.org/abs/2510.10274>

### 3.4 训练流程：两阶段 + 关键工程技巧

X-VLA 的训练分为两个阶段，并配合若干关键的工程化技巧：

**Phase I：预训练（Cross-embodiment Pretraining）**。
在 290K episodes、7 个数据源（Droid、Robomind、Agibot 等）、覆盖 5 种机械臂（单臂到双臂）的异构混合数据上，联合训练 backbone $\pi_\theta$ 与所有 soft prompts $P_H$：

$$
\min_{\theta, P_H} \; \mathbb{E}_{h \sim H} \left[ \mathcal{L}_{\text{FM-BC}}(\pi_\theta; P_H, h) \right]
$$

预训练结束后，backbone 成为"embodiment-agnostic 通用策略"，能跨异构数据做 zero-shot 推断。

**Phase II：新本体域适配（Domain Adaptation）**。
给定新硬件配置 $h_{new}$，分两步：

1. **Prompt warm-up**：新引入一组 $p_{new} \in R^k$，先用较大的学习率优化它，**backbone 冻结**。这一步让 $p_{new}$ 先"投影到预训练特征空间"，得到一个好的起点；
2. **Joint policy adaptation**：再同时优化 backbone 与 $p_{new}$，对新本体做端到端微调。

**关键工程技巧**：
- **Custom learning rate**：soft prompt 与视觉-语言模块用**比 backbone 主干更低的学习率**，避免预训练表征被灾难性漂移破坏（与 Reuss 2025 / Driess 2025 的经验一致）；
- **Aligned action representation**：所有本体的动作统一为 **EEF 末端执行器位姿**——Cartesian xyz 位置 + **Rotate6D**（Zhou 2019）旋转表示（避免欧拉角 / 四元数的不连续性）+ 离散 gripper 状态。位置/旋转 MSE，gripper BCE；
- **Intention abstraction（时间下采样）**：低层动作轨迹太细且含人为噪声，不适合高层意图建模。X-VLA 把它**下采样为 30 个 anchor point，覆盖 4 秒**，从而学到"接下来 4 秒打算做什么"的高层意图，而不是"每步的精确位姿"；
- **Balanced data sampling**：不仅在数据源间 shuffle，还在每个数据源的轨迹间 shuffle，避免"dominant 域吃掉所有 batch"，让预训练收敛更平滑。

### 3.5 推理流程

一次推理：当前观测 $(I, L, R_t)$ → 软提示按数据源 ID 查询 → Florence-Large 编码语言+主视图 + 共享 ViT 编码腕部视图 → proprio + 噪声动作 + flow time 拼成 control tokens → 与 soft prompt、multi-modal tokens 一起送入 24 层 self-attention block → 输出 30 个 anchor point 动作 chunk → 按控制频率执行。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

预训练数据共 **290K episodes**，来自 7 个数据源 / 5 种机械臂：

| 数据源 | 占比 | 机械臂 | 频率 |
|---|---|---|---|
| AGIBOT | 48.8% | 多种 | — |
| Franka（主）| 15.8% | Franka | 15 Hz |
| Franka（次）| 15.8% | Franka | 15 Hz |
| Franka（第三）| 6.7% | Franka | 15 Hz |
| Dual-Franka | 0.8% | 双 Franka | — |
| UR-5 | 8.7% | UR-5 | 30 Hz |
| Agilex | 3.7% | Agilex 双臂 | — |

动作统一为 EEF xyz + Rotate6D + 离散 gripper。X-VLA 还另外采集了 **Soft-Fold 叠衣数据集**（1,200 演示，双臂 Agilex 平台）用于验证灵巧操作能力，并承诺开源。

### 4.2 Baseline 与评价指标

论文给出目前最完整的 VLA 横向对比（Table 2），覆盖 0.1B–9B 的 18 个基线：

- **离散 token VLA**：OpenVLA-7B、OpenVLA-OFT-7B、TraceVLA-7B、ThinkAct-7B、UniVLA-9B；
- **连续 flow / 扩散 VLA / WAM**：π0-3B、π0+FAST-3B、GR00T-N1-3B、SmolVLA-2B、Octo-0.1B；
- **小规模/专项模型**：LBP-0.2B、MoDE-0.4B、SuSIE-1B、GHIL-Glue-1B、SpatialVLA-4B、FPC-VLA-7B、MemoryVLA-7B、DD-VLA-7B、Seer-0.3B、UniAct-0.5B、FLOWER-1B；
- **仿真基准（6 个）**：Simpler（Google Robot 视觉匹配/聚合 + WidowX 视觉匹配）、LIBERO（Spatial/Object/Goal/Long/Avg）、Calvin（ABC→D Avg. Len）、RoboTwin-2.0（Easy/Hard）、VLABench（Avg. PS / PDMS）、NAVSIM（自动驾驶）；
- **真机基准（3 平台）**：BridgeData-v2 协议的 5 个任务、Agilex Soft-Fold 灵巧叠衣、AIRBOT PEFT；
- **指标**：仿真基准用 success rate（Calvin 用 Avg. Len），真机用 success rate 或 folds/hour。

### 4.3 实现细节

- **视觉编码器**：Florence-Large（主视图+语言），共享 ViT（腕部视图）；
- **backbone**：24 层标准 Transformer encoder，hidden 1024，参数 ~0.9B；
- **soft prompt**：每个数据源一组，注入到第一层之前；
- **动作**：30 个 anchor point / 4 秒，EEF xyz + Rotate6D + 离散 gripper；
- **Phase I 预训练**：在 290K episodes 上联合训练 backbone + 全部 prompt；
- **Phase II 适配**：先 prompt warm-up，再 backbone + prompt 联合微调；
- **PEFT**：LoRA（rank 选 1% 比例），约 9M 可训练参数；
- **scaling 实验**：模型规模包含 768/1024 hidden、depth 6/12/24，prompt 长度 5/10/20/40/60，数据源 1/3/5/7，data size 1k/5k/10k/20k/25k。

---

## 5. 实验结果

### 5.1 Scaling 实验（Figure 5）

论文沿**模型规模、数据源多样性、数据规模**三个轴做 scaling 实验。**预测误差（ℓ1 error）**与**下游适配性能**之间存在强相关（R² = -0.925）。X-VLA-0.9B（hidden 1024，depth 24，7 个数据源）在三个轴上**都没有饱和趋势**，暗示继续扩大 backbone、数据多样性、数据规模有望进一步提升。Prompt 长度 20 之后收益边际递减。

![](assets/X-VLA%20-%20软提示跨具身视觉语言动作模型/figures/03-scaling.png)

> **图 5：X-VLA 的三轴 scaling 行为。** 随模型规模（hidden×depth）、prompt 长度、数据源数量、数据规模增加，预测误差单调下降且 R²=-0.925，最佳配置（1024+depth24）仍未饱和。
> 来源：论文 Figure 5，第 8 页，<https://arxiv.org/abs/2510.10274>

### 5.2 仿真基准全面 SOTA（Table 2）

X-VLA-0.9B 在 6 个仿真基准上**5 个建立新 SOTA**，且以 0.9B 的体量与 9B（UniVLA）/ 7B（OpenVLA、MemoryVLA、TraceVLA）/ 4B（SpatialVLA）级别对手正面比拼不落下风，部分指标甚至大幅领先（Simpler-WidowX 95.8 vs SOTA 71.9，RoboTwin-Easy 70.0 vs 46.4）：

| 基准 | 子项 | X-VLA-0.9B | 此前 SOTA | 结果 |
|---|---|---|---|---|
| Simpler | VM / VA / WidowX | 80.4 / 75.7 / **95.8** | 78.0 / 72.7 / 71.9 | 2 胜 1 胜 / 巨大领先 |
| LIBERO | Spatial / Object / Goal / Long / Avg | 98.2 / 98.6 / 97.8 / **97.6** / **98.1** | 98.4 / 98.8 / 97.9 / 94.5 / 97.1 | 2 胜，1 平，2 微负 |
| Calvin | ABC→D | 4.43 | 4.53 (FLOWER) | 略负（-0.1）|
| RoboTwin-2.0 | Easy / Hard | **70.0** / **39.0** | 46.4 / 16.4 | 巨大领先（+23.6 / +22.6）|
| VLABench | Avg. PS / PDMS | **51.1** / **87.3** | 39.7 / 81.7 | 双胜（+11.4 / +5.6）|
| NAVSIM | — | — | — | （未报告）|

论文摘要级说法："on several benchmarks, e.g., Simpler-WidowX (96%), Libero (98%), and Calvin-1st stage"，强调**多数子基准 90% 以上**。尤其 Simpler-WidowX 95.8 vs 此前 SOTA 71.9（差距 23.9 pt），是最醒目的单项。

### 5.3 真实机器人：BridgeData-v2 + Soft-Fold 叠衣

X-VLA-0.9B 在 3 个真机平台上做验证：

- **简单操作（Simple Manipulation, WidowX）**、**灵巧操作（Dex Manipulation, Agilex）**、**快速适配（PEFT, AIRBOT）**，5 个任务**全部超过 baseline**（Figure 7）；
- **Soft-Fold 灵巧叠衣**：1,200 演示，~100% 成功率，**33 次/小时**——平均一折不到 2 分钟。论文将其与闭源 π0-folding（数据量未披露）做对比，X-VLA throughput 与其相当；公平对照实验中，finetune π0-base 与从头训 ACT 都**达不到 X-VLA-0.9B 的 throughput**，验证了 X-VLA 的灵巧操作能力。

![](assets/X-VLA%20-%20软提示跨具身视觉语言动作模型/figures/05-real-world.png)

> **图 7：X-VLA 真机评估结果。** 在 BridgeData-v2 协议下的 5 个任务、Agilex 灵巧叠衣、AIRBOT PEFT 等 3 个真机平台均超过 baseline。
> 来源：论文 Figure 7，第 10 页，<https://arxiv.org/abs/2510.10274>

### 5.4 PEFT：1% 参数逼近 π0

X-VLA-0.9B 的 backbone 编码 embodiment-agnostic 特征，PEFT 只需要**约 9M 可训练参数（约 1%）**，即可在 Libero 与 Simpler-WidowX 上分别达到 **93% / 54%**——**与 π0 全参数（3B）微调的 94.2% / 55.7% 相当**（Table 3）。这意味着 X-VLA backbone + 软提示的设计确实把 embodiment-agnostic 与 embodiment-specific 干净地解耦了。

Table 3 详细对比（X-VLA-0.9B-LoRA-9M vs π0-3B-full）：
- Libero-Spatial 95.4 vs 96.8
- Libero-Object 96.6 vs 98.8
- Libero-Goal 96.0 vs 95.8
- Libero-Long 84.2 vs 85.2
- Simpler-WidowX 54.2 vs 55.7

四类 Libero 子基准上 1 胜 3 微负，Simpler-WidowX 接近（-1.5 pt）。

### 5.5 Soft Prompt 的可解释性（T-SNE，Figure 8）

对 7 个数据源训练后学到的 soft prompt 做 T-SNE 可视化：

- **硬件配置相同的 prompt 自然聚在一起**：Franka、UR-5、Agilex、AGIBOT 各成一簇；
- **Droid 中两套 Franka（left view / right view）相互交织**，说明 soft prompt 不是"机械按数据源 ID 划分"，而是**真正学到了 embodiment 语义**——两套 Franka 只在主视图指派上不同，应当属于同一 embodiment 类别。

这是 soft prompt 设计最重要的可解释性证据：它不靠"硬模板 / 语言硬描述"，仍能学到有意义的语义聚类。

![](assets/X-VLA%20-%20软提示跨具身视觉语言动作模型/figures/04-tsne.png)

> **图 8：7 个数据源 soft prompt 的 T-SNE 可视化。** 硬件配置相同的 prompt 自然成簇，Franka 的 left/right view 相互交织，说明 soft prompt 学到的是 embodiment 语义而非机械划分。
> 来源：论文 Figure 8，第 11 页，<https://arxiv.org/abs/2510.10274>

### 5.6 消融与失败案例

- **Soft Prompt vs 其他 prompt（Figure 4 + §3）**：none / HPT-style / language prompt / soft prompt 四组对照，soft prompt 唯一能在 Simpler-WidowX 跨本体适配上同时拿到**高 AD 成功率 + 低 validation error**；HPT 训练不稳定，language prompt 依赖 VLM 表示质量，soft prompt 是最稳的选择；
- **失败案例**：论文未单列逐个失败 case 统计，但 Limitations 一节承认：X-VLA 在**训练中完全没见过的新硬件形态**上仍需要 Phase II 适配（即 plug-and-play 部署尚未实现）；NVSIM 自动驾驶子基准未报告结果，可能因为该基准与机器人本体差异过大。

---

## 6. 与相关工作的关系

- **vs π0 / π0-FAST / GR00T N1 / OpenVLA（VLM-骨干 VLA）**：X-VLA 不像这些工作那样把整个 VLM 端到端微调，而是**只借用 VLM 的视觉编码器（Florence-Large）+ 自己堆标准 Transformer**。这让 X-VLA 可以在 0.9B 体量上同时与 3B（π0、GR00T N1）和 7B（OpenVLA、MemoryVLA）正面竞争；
- **vs Discrete Diffusion / MMaDA-VLA（离散扩散原生 VLA）**：X-VLA 走的是**连续 flow matching + 标准 Transformer encoder** 路线，强调"用 prompt 吸收异构性"；MMaDA-VLA 走的是"统一离散 token 空间 + 掩码扩散"路线，强调"统一表示"。两条路线在 Simpler-WidowX、RoboTwin-Easy 等指标上接近，但在 Libero 各类子基准上 X-VLA 更稳；
- **vs UniAct / HPT-style（heterogeneous pretrained transformers）**：X-VLA 的 soft prompt 思路与 HPT 共享"为每个数据源学专属参数"的灵魂，但 X-VLA 用 prompt 替代了"中间加 adapter"，参数量从 adapter-level（百万级 × N 本体）降到 0.04% 级，且训练稳定性显著更好；
- **vs FLOWER / Seer（小规模 flow-matching VLA）**：X-VLA 与 FLOWER 同属 flow matching 路线，但 FLOWER 强调"在 Simpler、Calvin 上小模型打平大模型"，X-VLA 强调"用 soft prompt 把跨具身异构性 + 真机灵巧操作 + PEFT 全部包圆"，覆盖广得多。

---

## 7. 局限与批判性评价

1. **规模仍小（0.9B）**：相比 7B / 9B 级对手，X-VLA-0.9B 仍小一个量级。论文承认"进一步扩大 backbone 与训练数据有望继续提升"（R²=-0.925 暗示 scaling 趋势还在），但受算力与高质量机器人数据稀缺所限；
2. **数据异质性上限未达**：尽管 AGIBOT 占比 48.8%，但 Dual-Franka 仅 0.8%，**双臂 + 高自由度灵巧手数据仍稀缺**。Soft-Fold 1,200 演示靠自采，规模化昂贵；
3. **动作监督的固有局限**：flow matching 的监督信号只有"低维动作标签"，无法表达高层推理、意图、多步依赖。论文在 Limitations §N 中明确指出，未来需要"3D 空间推理线索、物理动力学、中间 subgoal 标注"等更丰富监督，或自监督目标补充；
4. **新本体仍需 Phase II 适配**：soft prompt + 9M LoRA PEFT 让适配很轻，但**完全 plug-and-play 部署**（不收集任何演示、零微调）尚未实现。当演示数据稀缺 / 昂贵 / 危险时，这一前提会显著限制 X-VLA 的实用范围；
5. **动作空间仍依赖 EEF 表示**：所有本体都被压到 EEF xyz + Rotate6D + gripper。**本体特有的本体感觉**（关节构型、力矩信号）被丢弃，理论上会限制接触丰富、力控重要的任务；
6. **NVSIM 自动驾驶基准未报告结果**：X-VLA 论文把自己定位为"覆盖单臂、双臂、自动驾驶"的通用模型，但 Table 2 中 NVSIM 一栏对 X-VLA 留空（虽然 LBP、MoDE、SuSIE 等对手报告了 NVSIM 分数），这一点需要后续工作补齐或解释；
7. **代码与可复现性**：论文未公布官方代码与模型权重（项目页只有 demo 视频）。在 OpenVLA、π0、GR00T N1 均开源的当下，**不开源**会显著影响 X-VLA 论文的社区影响力。

---

## 8. 复现与实践建议

- **算力门槛**：Phase I 预训练 290K episodes × 7 数据源，需要大规模 GPU 集群。论文未披露具体 GPU 数与训练时长，但根据模型规模（0.9B）推测至少 64×A100 / H100 量级。**个人 / 小团队几乎不可能从头复现**；
- **PEFT 复现门槛低**：在 X-VLA-0.9B 公开权重上做 LoRA PEFT（9M 参数），单卡 RTX 4090 / A6000 即可尝试——但**前提是 X-VLA 官方开源权重**，目前还无；
- **数据准备**：跨本体数据的关键是「统一 EEF 表示 + Rotate6D 旋转 + 离散 gripper」三件套。每个数据源需要预先做坐标变换 / 重标注 / 时序对齐，工作量不小；
- **实现要点**：
  - 软提示要按「数据源 ID」索引，不能用自然语言硬模板；
  - 视觉编码器**只冻结微调是有损的**——论文用 custom LR 让 vision-language 模块用更小学习率，而不是完全冻结；
  - 时间下采样（30 anchor / 4s）这一招对高层意图建模帮助很大，但**下游任务若需要短时序精细控制，需要降低下采样率**；
  - balanced sampling 的"轨迹间 shuffle"很关键，不要用普通的 round-robin domain sampler；
- **常见踩坑**：
  - soft prompt 不能用 language prompt 替代（VLM 内部表示不精确）；
  - HPT-style adapter 在跨本体数据上**容易训崩**，不要照搬 HPT 的 recipe；
  - EEF 旋转用欧拉角 / 四元数会在不连续点附近产生 gradient spike，**必须用 Rotate6D**。

---

## 9. 个人启发与后续问题

1. **「吸收异构性」有比 adapter 更便宜的方法**：X-VLA 用 0.04% 参数的 soft prompt 就把跨具身异构性问题压住了。这暗示"为每个域 / 数据源学一组专属 embedding"是一个被低估的设计模式，可能在多任务 LLM finetune、多域推荐等场景里同样有效；
2. **「标准 Transformer + 极小额外参数」是 VLA 的一个被忽视的甜点**：相比 π0、GR00T N1 动辄 3B 全量微调，X-VLA 用 0.9B + 9M LoRA 就能逼近 π0，**这可能是 VLA 走向"低成本后训练"的关键一步**；
3. **T-SNE 显示 soft prompt 学到 embodiment 语义而非机械划分**：这是软提示设计最让人安心的证据——它不靠人工规则、不靠语言描述，仍能学到正确语义聚类。提示我们：**可学习 embedding + 简单约束 > 人工设计模板**；
4. **动作表示的"第二次革命"仍未定论**：X-VLA 走连续 flow matching + 极简 encoder；MMaDA-VLA 走离散扩散 + 统一 token 空间。两条路线目前在 Simpler-WidowX、RoboTwin-Easy 等指标上接近，但 X-VLA 在 Libero 子基准上更稳（98.1 vs 95.4）。**短期内"连续 + 离散"会并行 SOTA**，长期谁更优取决于：(i) 数据规模继续扩大的边际收益，(ii) 真机灵巧任务的稳定性（flow matching 的 ODE 积分在长程上是否漂移）；
5. **"embodiment-agnostic backbone + embodiment-specific prompt"是 VLA 的清晰解耦范式**：X-VLA 的 Phase II（prompt warm-up + joint adaptation）让"加新本体"成本极低，**未来 VLA 的生态可能像 LLM 一样——一个共享 backbone + 各种领域 prompt / adapter 库**；
6. **未解问题**：
   - (i) 软提示能否扩展到"任务级 prompt"（让同一个 backbone 用不同 prompt 切换任务）？
   - (ii) 0.9B 的"小模型"上限在哪里？继续扩到 3B / 7B 是否会与 π0 / OpenVLA-7B 拉出更大差距？
   - (iii) Soft Prompt 与 LoRA 的协同：能不能把软提示也用低秩矩阵参数化，进一步压缩参数量？
   - (iv) NVSIM 自动驾驶基准上的空白是否意味着 X-VLA 的"通用 VLA"定位尚有边界？

---

## 10. 材料来源

### Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2510.10274> | 全文（正文 33 页 + 附录） | 2026-08-31 | 精读主来源 |
| `figures/01-teaser.png` | Figure 1：X-VLA teaser / 总体框架 | arXiv:2510.10274 | 第 1 页 | 2026-08-31 | 核心框架章节 |
| `figures/02-architecture.png` | §4.1 详细模型架构（Soft Prompt Library + 异构编码 + Standard Self-attention Transformer Block） | arXiv:2510.10274 | 第 7 页 | 2026-08-31 | 核心方法章节 |
| `figures/03-scaling.png` | Figure 5：Scaling behavior across model size / prompt length / data sources / data size（R²=-0.925） | arXiv:2510.10274 | 第 8 页 | 2026-08-31 | Scaling 实验章节 |
| `figures/04-tsne.png` | Figure 8：T-SNE visualization of soft prompts on 7 data sources | arXiv:2510.10274 | 第 11 页 | 2026-08-31 | 可解释性章节 |
| `figures/05-real-world.png` | Figure 7：Real-world evaluation on 3 platforms (BridgeData-v2, Agilex Soft-Fold, AIRBOT PEFT) | arXiv:2510.10274 | 第 10 页 | 2026-08-31 | 真机实验章节 |
| 项目页 <https://thu-air-dream.github.io/X-VLA/> | Demo 视频与可视化 | — | — | 2026-08-31 | 补充材料 |
| arXiv HTML 版 <https://arxiv.org/html/2510.10274> | 论文 HTML 渲染（便于引用） | — | — | 2026-08-31 | 写作交叉验证 |
