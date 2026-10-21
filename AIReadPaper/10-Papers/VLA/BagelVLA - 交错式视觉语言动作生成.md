---
type: paper-note
status: draft
domain: VLA
paper: "BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation"
year: 2026
arxiv: 2602.09849
doi: null
source: https://arxiv.org/abs/2602.09849
project: https://cladernyjorn.github.io/BagelVLA.github.io
code: null
tags:
  - VLA
  - World-Models
  - World-Action-Model
  - Mixture-of-Transformers
  - Flow-Matching
  - Keyframe-Prediction
  - Interleaved-Planning
created: 2026-08-27
updated: 2026-08-27
---

# BagelVLA：交错式视觉-语言-动作生成驱动的长程操作

## 基本信息

- **作者**：Yucheng Hu*、Jianke Zhang*、Yuanfei Luo*、Yanjiang Guo、Xiaoyu Chen、Xinshu Sun、Kun Feng、Qingzhou Lu、Sheng Chen、Yangang Zhang、Wei Li、Jianyu Chen（* 共同一作，Wei Li 与 Jianyu Chen 为共同通讯）
- **机构**：清华大学交叉信息研究院（IIIS）、字节跳动 Seed
- **发表**：arXiv 2602.09849v2（2026-02-11）
- **项目页**：<https://cladernyjorn.github.io/BagelVLA.github.io>
- **通讯邮箱**：liwei.85@bytedance.com、jianyuchen@tsinghua.edu.cn
- **原文 PDF**：<https://arxiv.org/pdf/2602.09849>

## 一句话结论

BagelVLA 把「语言规划（识别当前子任务 subtask）→ 视觉预测（预测未来关键帧 keyframe）→ 动作生成」三条链在单一 Mixture-of-Transformers（MoT）架构里交错地串起来，初始化自字节的统一理解-生成模型 Bagel（Qwen2.5-7B 底座 + SigLIP2 + FLUX VAE），并引入 Residual Flow Guidance（RFG）——用「当前帧」作噪声先验、单步去噪即可提取预测性视觉特征，把视觉预见的推理代价压到最小。它在 CALVIN ABC→D 达到 Avg. Len 4.41、RoboTwin Clean 75.26 / Randomized 20.87 全面超越 π0 / RDT / UP-VLA / VPP，真机双机械臂长程规划任务成功率达 73.3%、规划准确率近 90%，是「统一理解-生成-动作」路线在 VPP 之后的最新一步。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

一个真正通用的机器人策略需要同时具备三种能力：**理解该做什么**（根据指令做语义推理）、**预测接下来会发生什么**（预见物理后果）、**执行必要的动作**（精确低层控制）。但 2025 年以前的 Vision-Language-Action（VLA）模型普遍把这三件事当作互相割裂的模块：一类方法（如 Robix、Gemini Robotics 的 planning 分支）只做高层语言规划、缺乏视觉预测；另一类方法（如 UniPi、GR-2、Cosmos Policy）专注视觉预测、却缺乏复杂任务所需的逻辑推理。BagelVLA 的出发点是：把「推理、预测、控制」显式地交错进同一个 transformer 序列里，让模型在每一步都同时「用文字想下一步」、再「用像素想象结果」、最后「落地成动作」。

### 1.2 现有方法的瓶颈

按论文 Related Works 的梳理，此前路线存在三层不足：

- **语言规划与视觉预测割裂**：RT-2、OpenVLA 用离散动作 token 直接映射「视觉语言表征 → 动作」，表达能力受限于离散化；Octo、3D Diffuser Actor、π0 用连续扩散表示，但三者都忽略了「VLM 预训练 → VLA 微调」之间的对齐鸿沟，微调后视觉-语言能力退化；
- **纯视觉预测路线缺乏 VLM 主干**：VPP、Cosmos Policy 用视频生成模型当策略，像素预测虽然容易和机器人观测对齐，但没有专门的 VLM 主干，指令跟随（尤其需要复杂推理的任务）表现差；
- **统一理解-生成模型没被真正用到具身**：Bagel、Chameleon、LMFusion、Show-o 等统一多模态模型虽涌现出多模态推理能力，但 Villa-X、F1、UniUgp 等把它们接进具身时，只是「外挂」action expert，没有显式建模「语言 ↔ 视觉」的交错推理，丢掉了逐步骤多模态 chain-of-thought 的能力——而这对长程任务是关键。

### 1.3 本文核心贡献

1. 提出 BagelVLA，一个把语言规划、视觉预测、动作生成统一进单一架构的 VLA，显式建模「语言 → 视觉动态」的转移，增强长程任务的推理与控制；
2. 引入 **Residual Flow Guidance（RFG）**：以当前观测为结构化先验、单步去噪预测「到下一个关键帧的残差变化」，用极低延迟提取预测性视觉特征；
3. 在仿真（CALVIN、RoboTwin 2.0）与真机（AgileX 双机械臂，9 项基本技能 + 2 类长程规划任务）上全面超越基线，尤其在多阶段推理任务上优势显著。

---

## 2. 任务定义与输入输出

### 2.1 交错规划（Interleaved Planning）的形式化

传统语言条件操控学的是直接映射 $p_\theta(a_t \mid v_t, L)$，把全局指令 $L$ 压成一个黑盒条件。这对长程任务不够——一条全局指令（如「按红→黄→蓝→绿的顺序叠积木」）隐式蕴含一串有先后依赖的子阶段。

BagelVLA 把问题重写为**交错规划**：给定全局指令 $L$ 和当前观测 $v_t$，显式建模「当前子任务 $l_t$、未来结果（关键帧）$v_{t+k}$、动作 $a_t$」的联合分布，并按操控的逻辑依赖做因子分解：

$$p_\theta(a_t, v_{t+k}, l_t \mid v_t, L) = p_\theta(l_t \mid v_t, L) \cdot p_\theta(v_{t+k} \mid v_t, L, l_t) \cdot p_\theta(a_t \mid v_t, L, l_t, v_{t+k})$$

训练目标即最大化上述因子化似然：

$$J = -\left( \mathcal{L}_l + \mathcal{L}_v + \mathcal{L}_a \right) = \max_\theta \mathbb{E}_{\mathcal{D}}\log\left[ p_\theta(l_t|v_t,L)\,p_\theta(v_{t+k}|v_t,L,l_t)\,p_\theta(a_t|v_t,L,l_t,v_{t+k}) \right]$$

三个因子对应三步：① 语言规划——先识别全局指令中的即时文本目标 $l_t$；② 视觉预测——以 $l_t$ 为条件，模型充当世界模型预测物理结果 $v_{t+k}$；③ 动作生成——在文本规划 + 视觉预测的双重 grounding 下生成动作 $a_t$。

### 2.2 输入、输出与假设

- **输入**：当前观测 $v_t$（图像 + proprioception）、全局语言指令 $L$；
- **输出**：子任务文本 $l_t$、未来关键帧 $v_{t+k}$、动作 chunk $a_t$；
- **核心假设**：只要显式耦合「语言规划 + 视觉预测」，动作生成的精度与长程指令跟随能力就能显著提升——后文的消融（去掉 textual-planning / keyframe-forecasting 分别掉点）直接验证了这一假设。

---

## 3. 核心方法

### 3.1 总体框架

BagelVLA 采用 **Mixture-of-Transformers（MoT）** 架构，用三个独立 transformer 分别专责语言、视觉、动作三种模态，经 self-attention 互联。理解专家与生成专家**初始化自 Bagel**（字节的统一 MoT 理解-生成模型），之上再挂一个更小的 action 专家。区别于此前的 MoT-VLA（Villa-X、F1、UniUgp），BagelVLA 的语言与视觉组件都有稳健的预训练初始化，并采用全新的<span style="color: red">**<span style="color: red">双流匹配</span>**</span>（dual flow-matching）机制。

![](assets/BagelVLA%20-%20交错式视觉语言动作生成/figures/01-teaser.png)

> **图 1：框架总览（Figure 1）。** BagelVLA 把语言规划、视觉预测、动作生成统一进单一框架，并构建「通用多模态数据 + 大规模机器人数据」的混合数据集；带子任务与关键帧标注的机器人数据把基础模型的通用推理与视觉生成能力迁移到具身场景。左侧展示了「三步交错」的核心思想：先用文字拆出下一步要操作的子任务，再用像素想象该子任务的完成状态，最后生成动作。
> 来源：论文 Figure 1，第 2 页，<https://arxiv.org/abs/2602.09849>

### 3.2 理解专家与生成专家

两个专家采用 Qwen2.5-LLM-7B 的架构。遵循 Bagel 的配置，模型用**两个不同的视觉编码器**分别负责视觉-语言理解和目标图像预测：

- **理解专家（LLM Expert）**：输入观测 $v_t$ 由 SigLIP2 编码后与文本指令 $L$（及 $l_t$）拼接；自回归生成子任务 $l_t$ 时 attend 到 ViT 特征。用自回归交叉熵损失 $\mathcal{L}_l = -\log p_\theta(l_t \mid v_t, L)$ 优化；
- **生成专家（Generation Expert）**：图像用 FLUX 的 VAE 编码；对关键帧去噪时 attend 到所有输入视角的 VAE 与 ViT 特征及文本信息，用 Flow Matching 迭代去噪生成关键帧，$\mathcal{L}_v = -\log p_\theta(v_{t+k} \mid v_t, L, l_t)$。

### 3.3 动作专家（Action Expert）

动作专家是挂在 MoT 框架上的独立 transformer，负责处理 proprioception 与动作模态。它与 Qwen2.5-LLM 同构，但把 MLP 的 intermediate size 缩到原来的 **1/5，得到 2B 参数**——这个紧凑体积使其能通过 KV-cache 与异步动作生成获得更高执行频率。动作用 Flow Matching 学习动作 chunk：$\mathcal{L}_a = -\log p_\theta(a_t \mid v_t, L, l_t, v_{t+k})$。去噪过程中，动作序列可以 attend 到输入视角的 VAE/ViT 特征、全局指令 $L$、生成的子任务 $l_t$，以及动作专家自己的 proprioception 输入。

**关键细节**：动作专家 attend 的是「正在生成的图像」的中间隐状态——这涉及生成专家与动作专家两个 Flow Matching 模块之间的**非对称信息交互**，是本文的核心机制，详见 3.4。

![](assets/BagelVLA%20-%20交错式视觉语言动作生成/figures/02-method-pipeline.png)

> **图 2：BagelVLA 框架示意（Figure 2）。** MoT 架构含三个独立 transformer（语言/视觉/动作专家）。为应对长程任务与语义泛化，把语言条件动作学习表述为长序列的交错规划问题：三种模态被组织成统一序列，使模型能基于交错上下文在三种模态上同时生成预测；并设计专门机制协调多个流匹配专家之间的交互、提升推理效率。
> 来源：论文 Figure 2，第 5 页，<https://arxiv.org/abs/2602.09849>

### 3.4 双流匹配的三种条件机制（核心贡献）

关键帧预测与动作生成共用一条交错序列，如何在训练与推理间保持一致是关键。论文提出三种交互机制（Figure 3），并在 CALVIN 上消融后选定默认方案：

- **Scheme 1：Complete Denoise（完全去噪）**——生成专家先把关键帧完整去噪（$N_1$ 步），生成图再作为动作生成的条件。训练时为让动作专家看到「完全去噪」的图，在去噪序列后 append 真值关键帧。本质是「世界模型 WM + 逆动力学 IDM」的组合，理论合理但推理延迟高（$N_1+N_2$ 步）且会累积视觉误差；
- **Scheme 2：Joint Denoise（联合去噪）**——关键帧与动作同步去噪，动作生成 attend 到正在去噪的噪声图，$N$ 步内同时生成两者，显著降低延迟；
- **Scheme 3：Single-step Denoise（单步去噪）**——动作生成只 attend 关键帧**第一步去噪**的 KV-cache，即以「初始噪声作关键帧输入」直接生成动作，进一步压缩计算。

三者的动作流匹配损失分别写作（$v$ 为对应模态的速度预测，$\tau$ 为去噪时刻，$\tau=0$ 表初始噪声、$\tau=1$ 表真值目标）：

$$\mathcal{L}_v = \mathbb{E}\left\| v_{v,\theta}(L,v_t,l_t,\tau,v^\tau_{t+k}) - (v^1_{t+k}-v^0_{t+k}) \right\|_2^2,\quad v^\tau_{t+k}=(1-\tau)v^0_{t+k}+\tau v^1_{t+k}$$

$$\mathcal{L}_{a1} = \mathbb{E}\left\| v_{a,\theta}(L,v_t,l_t,v^{\tau=1}_{t+k},\tau,a^\tau_t) - (a^1_t-a^0_t) \right\|_2^2$$

$$\mathcal{L}_{a2} = \mathbb{E}\left\| v_{a,\theta}(L,v_t,l_t,v^{\tau}_{t+k},\tau,a^\tau_t) - (a^1_t-a^0_t) \right\|_2^2$$

$$\mathcal{L}_{a3} = \mathbb{E}\left\| v_{a,\theta}(L,v_t,l_t,v^{\tau=0}_{t+k},\tau,a^\tau_t) - (a^1_t-a^0_t) \right\|_2^2$$

在 Scheme 3 基础上，论文进一步提出两种噪声初始化变体：

$$\text{Naive Single-step}:\ v^{\tau=0}_{t+k}\sim\mathcal{N}(0,I) \qquad \text{RFG}:\ v^{\tau=0}_{t+k}\sim\mathcal{N}(v_t, I)$$

**Residual Flow Guidance（RFG）** 的关键在于把当前帧 $v_t$ 注入初始图像噪声：这让世界模型只需建模「机器人操控带来的变化」而不是「重建静态背景」，从而显著减少所需去噪步数（见 Figure 5）。论文在 4.3 节消融后选定 **Single-step Denoise（RFG）** 为默认设置。

![](assets/BagelVLA%20-%20交错式视觉语言动作生成/figures/03-dual-denoise.png)

> **图 3：三种双去噪机制的示意（Figure 3）。** (a) Complete Denoise：图像预测与动作生成分开，共 $N_1+N_2$ 步；(b) Joint Denoise：同步去噪，共 $N$ 步；(c) Single-Step Denoise：动作生成直接以图像预测的第一步去噪上下文为条件。具体拼接序列与注意力掩码见附录 B。
> 来源：论文 Figure 3，第 6 页，<https://arxiv.org/abs/2602.09849>

### 3.5 数据引擎与两阶段训练

为构建具身场景下「子任务规划 + 关键帧预测」的大规模预训练数据，论文针对四大类数据设计针对性处理流水线：

- **机器人数据**：自采专家演示（手工标注、分割视频得到 $l_t$，保证高质量规划与关键帧预测）+ 公开多本体数据（缺乏细粒度标注的，用 Seed-1.5-VL-thinking 合成 $l_t$ 并识别时间边界 start/end 帧，再过滤高质量样本）。这部分仅用于预训练；
- **通用数据**：egocentric 人类视频 + 大规模图像-文本 VQA。人类视频同样用 Seed-1.5-VL-thinking 生成语言标注，但因人类场景复杂不标注子任务、只预测每段操作的最后一帧。这部分主要用于保留基础模型的原有理解与生成能力。

**两阶段训练**：

- **Stage 1（预训练）**：只微调理解与生成专家，学文本规划 + 关键帧预测，并配通用 QA 数据做语言协同训练以保留通用语言能力。数据构成：通用 VQA 2.98M 对、人类手部数据（视觉动态）310k episodes、开源机器人数据（语言规划+视觉动态）146k /（仅视觉动态）297k、自采真机（语言规划+视觉动态）75k；
- **Stage 2（微调）**：引入动作标签的下游机器人数据，全模型同时微调三个规划任务。四个场景：Calvin ABC（视觉+动作）、RoboTwin（语言+视觉+动作，50 任务 × 50 episodes = 2.5k）、ALOHA 基本任务（视觉+动作，3k）、ALOHA 长程任务（语言+视觉+动作，1.5k）。

### 3.6 推理流程与复杂度

推理时模型交错生成文本规划、关键帧与动作；每个去噪步骤只激活一个专家（文本与关键帧用 7B，动作用 2B）。单步去噪进一步压缩计算。具体地，把当前帧 + 指令上下文 + 纯噪声图拼接计算理解与生成专家的 KV 对，再条件到动作生成上。BagelVLA 在单张 RTX 5090 上 **1.2 秒 / chunk**（chunk size 48 时真机动作频率 40Hz）。再叠加 Asynchronous Execution（异步执行，训练时随机用前一帧替换当前帧）后，执行频率可到 **72Hz**。

---

## 4. 数据集与实验设置

### 4.1 仿真与真机环境

- **CALVIN ABC→D**：ABC split 训练、D 环境零样本评测，指标为平均任务完成长度 Avg. Len（1000 个 length-5 任务）；
- **RoboTwin 2.0**：50 任务、每任务 50 条 clean 演示训练，Clean 与 Randomized 两种设定、未见指令评测，指标为成功率（每任务测 100 次）；
- **真机 AgileX 双机械臂（Aloha-AgileX）**：两类任务——**基本任务**（9 项技能，含 pick&place、浇水、叠积木、插花、叠碗、倒薯条、扫地、按按钮、开关抽屉，3,000 条轨迹）与**长程规划任务**（彩色积木按序叠放、算数符号块摆放，1,500 条轨迹），全部手工标注子任务与关键帧。

### 4.2 Baseline 与评价指标

仿真对比 π0、RDT、UP-VLA、VPP（前两者为纯动作/流模型，后两者为含未来预测能力的 VLA）；真机对比 π0 与 VPP。指标：CALVIN 用 Avg. Len，RoboTwin 用 Clean/Randomized 成功率，真机用成功率（每任务 20 次、随机初始位姿 + 干扰物）与规划准确率（20 个中间状态的动作趋势正确率）。

### 4.3 实现细节

- 学习率 $1\times10^{-5}$，FSDP 框架 + packed dataset；
- 预训练：64×A800、batch≈1600、20k 步；
- CALVIN：8×A800、batch 192、30k 步、chunk 10、无 proprioception、2 视角输入预测第 3 视角；
- RoboTwin：8×A800、batch 128、60k 步、chunk 16、每 3 步采样 1 动作（有效 horizon 48）、3 视角输入预测主视角；
- 真机：32×A800、batch 512、50k 步、chunk 24、3 视角（主 + 左右腕）输入预测主视角。

---

## 5. 实验结果

### 5.1 主要定量结果

**仿真（Table 1）**：

| 模型 | CALVIN ABC→D (Avg.Len) | RoboTwin Clean | RoboTwin Randomized |
|---|---|---|---|
| π0 | 3.648 | 46.42 | 16.34 |
| RDT | — | 34.50 | 13.72 |
| UP-VLA | 4.078 | 52.92 | 15.16 |
| VPP | 4.329 | — | — |
| w/o Textual-planning | — | 54.00 | 19.20 |
| w/o Keyframe-forecasting | 3.345 | 56.72 | 15.92 |
| **BagelVLA** | **4.405** | **75.26** | **20.87** |

CALVIN ABC→D 上 BagelVLA 达 Avg. Len 4.41（Table 1 记 4.405），超越 VPP（4.329）与 UP-VLA（4.078）；RoboTwin 上 Clean 75.26 远超 π0 的 46.42 与 UP-VLA 的 52.92，Randomized 20.87 同样 SOTA。值得注意的是：去掉 keyframe 预测后 CALVIN 掉到 3.345、去掉 textual 后 RoboTwin Clean 掉到 54.00，两者都证明「语言 + 视觉」双模态交错缺一不可。

**真机基本任务（Table 2，9 技能平均）**：BagelVLA **75.5%** vs π0 65.0% vs VPP 59.5%；尤其在 OOD 的 pick&place（未见物体）上 BagelVLA 85% 大幅领先 VPP 45%、π0 55%，优势来自微调中保留的语义特征（继承自理解与生成专家的预训练）。

**真机长程规划（Table 3）**：

| 任务 | π0 | VPP | w/o Keyframe | w/o Textual | BagelVLA |
|---|---|---|---|---|---|
| 按序叠积木 成功率 | 40.0 | 25.0 | 53.3 | 43.3 | **73.3** |
| 算数符号摆放 成功率 | 31.7 | 23.3 | 50.0 | 33.3 | **63.3** |

规划准确率上 BagelVLA 接近 90%，且观察到「成功率 < 规划准确率」的 gap，论文归因于模型与数据集在精细运动控制精度上的局限。

### 5.2 定性结果

![](assets/BagelVLA%20-%20交错式视觉语言动作生成/figures/04-realworld-results.png)

> **图 4：真机任务的交错规划可视化（Figure 4）。** 给定全局指令与当前观测，BagelVLA 先识别即时子任务、预测该子任务的目标图，再生成动作。图中展示「按序叠积木」「算数符号块摆放」以及 Agibot 数据集中的一个任务。可以直观看到「文字子任务 → 预测关键帧 → 动作」的三段交错，验证了模型在长程任务上的多模态推理能力。
> 来源：论文 Figure 4，第 9 页，<https://arxiv.org/abs/2602.09849>

### 5.3 消融实验

**双流匹配机制（Table 4，CALVIN 单视角）**：

| 方案 | 延迟↓ | ABC→D↑ |
|---|---|---|
| Complete Denoise | 6.04s | 2.480 |
| Joint Denoise | 2.90s | 2.038 |
| Single-step Denoise（Eq.2） | 1.23s | 3.345 |
| **RFG（Eq.3）** | **1.23s** | **3.600** |

单步去噪不仅任务成功率更高、推理也更快。论文解释：complete/joint denoise 在测试时遇到换色场景会遭遇 flow-matching 阶段的 OOD 中间态，导致动作执行显著退化；而 single-step 直接以第一步去噪上下文为条件，绕过了这一域漂移。

**RFG vs Naive（4.3.2）**：RFG 在 Calvin 与真机基本任务上均显著优于 naive single-step，且保持同样低的推理延迟。原因：naive 的动作生成依赖「纯高斯噪声做单步去噪」得到的中间特征；RFG 把初始帧注入噪声初始化，为动作生成提供更强先验。Figure 5 可视化显示 RFG 用极少的去噪步数（如 10 步）即可生成高质量未来帧，因为它让世界模型专注于「动态区域」而非重建静态背景。

**预训练有效性（4.3.3）**：对比 w/o pretrain，预训练基线在 pick&place（OOD）上成功率显著更高，说明仅靠「语言规划 + 视觉预测」预训练就足以提升语义泛化；在扫地、倒薯条、叠积木三个 medium-horizon 任务上联合预训练也带来更高精度。

**语言与视觉模态的有效性（4.3.4）**：textual-planning 在 RoboTwin 上带来 +21% 成功率，在两类真机长程任务上同样大幅增益；keyframe-forecasting 在仿真与真机上均提升动作规划精度。二者共同确认了交错规划框架的价值。

![](assets/BagelVLA%20-%20交错式视觉语言动作生成/figures/05-rfg-ablation.png)

> **图 5 / 6：RFG 与 Naive 单步去噪的对比（Figure 5、6）。** 上半（Figure 5）展示 naive（Eq.2）与 RFG（Eq.3）在不同去噪步数下的关键帧预测结果——RFG 能保留背景、用极少的步数生成高质量图；下半（Figure 6）为真机基本任务上的消融对比，RFG 在多项任务上显著优于 naive，且验证了预训练（w/o pretrain 掉点）与各模态的贡献。
> 来源：论文 Figure 5、6，第 11 页，<https://arxiv.org/abs/2602.09849>

### 5.4 泛化、效率与失败案例

- **泛化**：真机测试加入未见物体、干扰物、背景/物体颜色变化；BagelVLA 在未见物体 pick&place 上 85% 的成功率说明其语义特征在 VLA 微调中被很好地保留；
- **效率**：单卡 RTX 5090 上 1.2s/chunk（40Hz），异步执行推到 72Hz；RFG 相比 complete denoise（6.04s）把单步动作推理延迟压低了约 5 倍；
- **失败/局限**：论文未单列失败 case 图集，但明确承认「任务成功率与规划准确率之间的 gap」源于模型与数据集在精细运动控制精度上的不足——规划接近正确（近 90%）但动作映射仍有缺口，这是后续需要针对性补强的方向。

---

## 6. 与相关工作的关系

按论文 Related Works 与石麻笔记 EP06 专访口径梳理：

- **vs VPP（同组前作）**：VPP 只用「预测性视觉表征」作视觉编码器、无显式语言规划，且没有专门 VLM 主干；BagelVLA 把 VPP 的视觉预测扩展为「subtask + keyframe」两段式，并在理解专家（VLM）加持下获得复杂推理能力。CALVIN 上 4.41 vs 4.329、真机基本任务 75.5% vs 59.5%，说明「语言 + 视觉」双模态交错比单视觉预测更强；
- **vs UP-VLA**：同为「理解 + 预测」统一模型，但 UP-VLA 缺语言规划的主干对齐；BagelVLA 在 RoboTwin Randomized（20.87 vs 15.16）上明显领先；
- **vs Cosmos Policy / GR-2**：纯视频模型当策略、无 VLM 主干，指令跟随差；BagelVLA 用 Bagel 的统一理解-生成底座补齐了这一短板；
- **vs π0 / RDT**：纯动作流模型 / 扩散基础模型，无未来预测辅助；BagelVLA 通过视觉预测辅助任务获得一致增益；
- **vs Villa-X / F1 / UniUgp**：同为「统一模型 + action expert」路线，但 BagelVLA 显式建模语言↔视觉交错推理，而非仅外挂 action head，因此保留了逐步骤多模态 CoT；
- **vs Bagel（上游底座）**：Bagel 是字节的统一理解-生成 MoT 模型（Qwen2.5-7B 底座 + SigLIP2 + FLUX VAE），BagelVLA 是其具身化的直接延伸——这也把「统一多模态预训练 → 具身策略」的迁移路径走通了。

---

## 7. 局限与批判性评价

1. **精细运动控制的缺口**：论文自承「成功率 < 规划准确率」，规划接近正确但动作映射有 gap，说明低层控制精度仍是瓶颈——而这是长程精细操作（如插花、叠小积木）的命门；
2. **参数规模与推理成本仍高**：7B 理解 + 7B 生成 + 2B 动作 ≈ 16B 参数，单卡 RTX 5090 才能到 40Hz、异步才 72Hz；对比 Fast-WAM（Wan2.2-5B + 1B，190ms）与 VPP（1.5B SVD，<160ms），BagelVLA 的部署门槛明显更高；
3. **RFG 的「单步」是折中而非免费午餐**：虽然单步去噪绕开了 OOD 中间态，但 Figure 5 也显示 naive 在关键帧质量上不如多步；RFG 靠「当前帧先验」补偿，本质是用「结构化先验」换「预测丰富度」，对未来不确定性的多模态建模能力有限；
4. **数据依赖 Seed-1.5-VL-thinking 合成标注**：公开数据集缺乏细粒度标签，靠 Seed-1.5-VL-thinking 自动合成子任务与时间边界，标注质量上限受限于该 VLM，可能存在系统性噪声；
5. **长程任务的评测范围有限**：真机只覆盖 2 类长程任务、每任务 20 次评测，规模与多样性不如 CALVIN/RoboTwin 的统计性结论，且未报告跨本体迁移；
6. **缺少与 Motus / MMaDA-VLA 等同期统一模型的直接对比**：论文基线集中在 π0/RDT/UP-VLA/VPP，未纳入 2026 年新范式的代表（离散扩散、统一 WAM），横向定位略有滞后；
7. **代码未开源**（截至笔记日期项目页仅视频）：复现门槛高，实验全部依赖公司内部数据（字节 Seed）与算力。

---

## 8. 复现与实践建议

- **算力门槛**：预训练 64×A800、20k 步；下游微调 CALVIN/RoboTwin 用 8×A800、真机用 32×A800。推理单卡 RTX 5090 即可（1.2s/chunk），但 7B+7B+2B 的 MoT 显存不小；
- **底座依赖**：理解/生成专家初始化自 Bagel（Qwen2.5-7B + SigLIP2 + FLUX VAE），需先取得该统一模型的权重；动作专家 MLP 缩到 1/5（2B）是关键的工程取舍，建议复现时保持；
- **数据准备**：需混合「通用 VQA（LLaVA-Pretrain + FineVision）」「人类手部数据（EgoDex 310k）」「开源机器人数据（AgiBotWorld 120k + GR 80k + Galaxea 99k + Bridge 55k + RoboTwin 27.5k）」「自采真机（ALOHA 4.5k）」；公开机器人数据需用 Seed-1.5-VL-thinking 合成子任务与时间边界；
- **实现要点**：
  - 双流匹配的注意力掩码设计要「防模态间信息泄漏 + 对齐推理时变步数的时序采样」；
  - 生成专家 FM timestep 用 LogitNormal(0,1)、动作专家用 Beta(1.5,1)，两套分布需分别配置；
  - RFG 的噪声初始化 `v ~ N(v_t, I)` 是「单步即可」的关键，naive `N(0,I)` 会在真机任务上明显掉点；
- **常见踩坑**：complete/joint denoise 在换色 OOD 场景会遭遇 flow-matching 中间态域漂移，若下游任务有颜色/背景变化，务必选 single-step 方案。

---

## 9. 个人启发与后续问题

1. **「统一理解-生成底座 → 具身策略」的迁移路径被打通**：BagelVLA 证明字节的 Bagel（Qwen2.5-7B + SigLIP2 + FLUX VAE）这类统一多模态模型可以直接长出 action 专家、变成 SOTA VLA。相比从零训 VLA 或「VLM + 视频模型」双主干拼接，MoT 共享注意力更有优势——这与 Motus（生数 + 清华）、F1 是同一方向的不同实现；
2. **RFG 是「预测性视觉表征」的升级**：VPP 用「视频模型当 encoder、一步前向」，BagelVLA 用「当前帧当噪声先验、单步去噪」，两者共同指向一个判断——**视频生成模型的价值在「预测性表征」而非「完整生成」**。Fast-WAM 甚至直接跳过测试时未来想象。这条线值得继续追问：预测性表征能否进一步压缩到 latent/token 而不再需要像素级解码？
3. **「语言规划 + 视觉预测」的双重 grounding 是长程任务的关键**：去掉任一模态都掉点（+21% 的 textual 增益尤其显著），说明长程操控确实需要「文字拆任务 + 像素想象结果」两步，纯动作模型在长程上天然吃亏；
4. **规划准确率 vs 执行成功率的 gap 是普遍痛点**：BagelVLA 近 90% 的规划准确率却只有 73% 的执行成功率，暴露了「会想不会做」的系统性短板——这与 VLA 领域的普遍观察一致，精细低层控制（末端位姿、力度）仍是数据与模型的双重瓶颈；
5. **未解问题**：（i）RFG 的单步去噪在「需要精确空间感知」的任务上是否够用？（ii）16B 的 MoT 能否蒸馏到实时边端？（iii）合成标注（Seed-1.5-VL-thinking）的质量上限如何量化？（iv）与 Motus/MMaDA-VLA 的直接对比何时补齐？

---

## 10. 材料来源

### Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2602.09849> | 全文（22 页 + 附录） | 2026-08-27 | 精读主来源 |
| `figures/01-teaser.png` | Figure 1：框架总览 | arXiv:2602.09849 | 第 2 页 | 2026-08-27 | 概念对比 |
| `figures/02-method-pipeline.png` | Figure 2：BagelVLA 框架示意 | arXiv:2602.09849 | 第 5 页 | 2026-08-27 | 核心方法 |
| `figures/03-dual-denoise.png` | Figure 3：三种双去噪机制 | arXiv:2602.09849 | 第 6 页 | 2026-08-27 | 核心机制 |
| `figures/04-realworld-results.png` | Figure 4：真机交错规划可视化 | arXiv:2602.09849 | 第 9 页 | 2026-08-27 | 真机定性 |
| `figures/05-rfg-ablation.png` | Figure 5/6：RFG vs Naive 与真机消融 | arXiv:2602.09849 | 第 11 页 | 2026-08-27 | 关键消融 |
| 项目页 <https://cladernyjorn.github.io/BagelVLA.github.io> | 视频/补充材料 | — | — | 2026-08-27 | 实验视频 |
