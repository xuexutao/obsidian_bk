---
type: paper-note
status: draft
domain: VLA
paper: "RT-1: Robotics Transformer for Real-World Control at Scale"
year: 2022
arxiv: 2212.06817
doi: null
source: https://arxiv.org/abs/2212.06817
project: https://robotics-transformer1.github.io/
code: https://github.com/google-research/robotics_transformer
tags:
  - VLA
  - Robotics-Transformer
  - TokenLearner
  - FiLM
  - Imitation-Learning
created: 2026-08-27
updated: 2026-08-27
---

# RT-1：大规模真实机器人控制的 Robotics Transformer

## 基本信息

- **作者**：Anthony Brohan*、Noah Brown*、Justice Carbajal*、Yevgen Chebotar* 等（按字母序，* 共同一作，40+ 位作者）
- **机构**：Robotics at Google、Everyday Robots、Google Research Brain Team
- **发表**：arXiv 2212.06817（2022-12 初版，v2 更新于 2023-08；后被 CoRL 2023 接收）
- **项目页**：<https://robotics-transformer1.github.io/>
- **代码**：<https://github.com/google-research/robotics_transformer>
- **原文 PDF**：<https://arxiv.org/pdf/2212.06817>

## 一句话结论

RT-1 把机器人操控建模成「图像序列 + 语言指令 → 离散动作」的序列预测：ImageNet 预训练的 EfficientNet-B3 提取 6 张历史图像的视觉特征（每张 81 个 token），用 FiLM 将 USE 语言嵌入以仿射变换方式注入视觉编码器，TokenLearner 把每张图的 81 个 token 压缩到 8 个，再接入 decoder-only Transformer 输出离散化的动作 token（7 维手臂动作 + 3 维底座动作 + 1 维模式切换，每维 256 bin）；在 13 台机器人、17 个月采集的 130k 条真实演示、744 个任务指令上，RT-1 以 35M 参数在 3 Hz 下取得 seen 任务 97%、未见指令 76% 的成功率与强鲁棒性，并通过 SayCan 完成最多 50 步的长程组合任务，正式确立了「视觉 token + 语言条件 + Transformer 动作解码」的 VLA 经典范式。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

传统端到端机器人学习（模仿学习或强化学习）的工作流是「为单一或少数任务采集任务专属数据 → 训练任务专属策略 → 部署」，任务之间几乎不共享知识。而在 CV、NLP 领域，近几年的大模型革命早已抛弃了这种「竖井式（siloed）」路线，转向「在开放、任务无关的大规模数据集上预训练一个高容量通用模型，再零样本或小样本迁移到下游」。RT-1 的作者于是追问一个对机器人格外关键的问题：**能否训练一个单一的、高容量的多任务 backbone 策略，让它吸收大量多样化的真实机器人数据，并像语言/视觉大模型那样展现出对全新任务、环境、物体的零样本泛化？**

这个问题在机器人领域尤其重要，因为真实机器人数据要么依赖昂贵的遥操作人类演示，要么依赖工程复杂的自主采集；如果「预训练 + 泛化」的范式能成立，单任务数据采集的负担就能被大幅摊薄。

### 1.2 两大挑战：数据集与模型

论文把难点拆成两块：

- **数据集**：机器人数据集通常是「机器人专属」且靠手工采集的，难以同时做到规模（scale）与广度（breadth）。而且任务之间必须「连接得足够紧密」，模型才能发现结构相似任务之间的规律，进而组合出新的任务。RT-1 用 13 台机器人、17 个月时间采集了 130k 条真实演示，覆盖 744 个任务指令来正面回答这一点。
- **模型**：多任务学习需要高容量模型，Transformer 天然擅长，尤其是用语言指令做条件时；但机器人控制器又必须实时运行，这对 Transformer 的推理延迟构成硬约束。RT-1 的核心贡献就是设计一个「既能吸收海量数据、又能以 3 Hz 实时闭环」的紧凑架构。

### 1.3 本文核心贡献

1. 提出 **Robotics Transformer 1（RT-1）** 架构：把高维输入（相机图像、语言指令）和输出（马达指令）都编码成紧凑 token，交给 Transformer 做序列建模；
2. 构建并公开了一个大规模真实机器人操控数据集：**130k episodes、13 台机器人、17 个月、744 个任务指令**；
3. 系统验证了 RT-1 的多项能力：seen 任务 97%、未见指令 76%，对干扰物/背景的鲁棒性分别领先最强基线 36% / 18%，并在 SayCan 框架下完成最多 50 步的长程任务；
4. 证明 RT-1 能「吸收」异构数据（仿真数据、Kuka 机器人数据），在不损失原任务性能的同时显著提升泛化。

---

## 2. 任务定义与输入输出

### 2.1 问题形式化

RT-1 把控制策略定义为一个语言条件、视觉观测驱动的序列决策问题。在时刻 $t=0$，策略 $\pi$ 收到语言指令 $i$ 与初始图像观测 $x_0$，采样动作 $a_0 \sim \pi(\cdot \mid i, x_0)$ 执行；此后每步策略根据完整历史采样 $a_t \sim \pi(\cdot \mid i, \{x_j\}_{j=0}^{t})$，直到触发终止条件。一次完整交互 $i, \{(x_j, a_j)\}_{j=0}^{T}$ 称为一个 episode，结束时获得二元奖励 $r \in \{0, 1\}$，表示机器人是否成功完成了指令 $i$。目标是最大化指令、初始状态、转移动力学分布上的平均成功率。

### 2.2 输入、输出与假设

- **输入**：一段图像历史（RT-1 取 6 张，每张 $300 \times 300$）+ 一条自然语言指令；
- **输出**：每一步一个离散化的动作 token；
- **动作空间**（11 维 + 1 维模式）：
  - **手臂 7 维**：$(x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper opening})$；
  - **底座 3 维**：$(x, y, \text{yaw})$；
  - **模式 1 维**：离散切换「控制手臂 / 控制底座 / 终止 episode」三种模式；
- **核心假设**：模仿学习（行为克隆）——数据集里所有演示都是成功的（最终奖励为 1），用极大似然学习动作条件分布即可逼近演示者。

### 2.3 关键符号与目标函数

动作每一维被离散化到 256 个 bin（均匀分布在各自取值范围内），策略输出的是动作 token 的类别分布。行为克隆的负对数似然目标为：

$$\mathcal{L}_{\text{BC}} = -\sum_{t} \log \pi_\theta\!\left(a_t \mid i,\ \{x_j\}_{j=0}^{t}\right)$$

其中 $\pi_\theta$ 由 Transformer 参数化，$a_t$ 是离散化后的动作 token。训练时采用标准的 **分类交叉熵损失 + 因果掩码（causal masking）**——与 Gato、Lee et al. (2022) 等先前 Transformer 控制器一致。

---

## 3. 核心方法

### 3.1 总体框架

RT-1 的推理管线自顶向下（对应 Figure 3）是：

$$
\underbrace{\text{6 张图像 + 指令}}_{\text{输入}} \xrightarrow{\text{FiLM-EfficientNet}} \underbrace{6 \times 81}_{\text{视觉 token}} \xrightarrow{\text{TokenLearner}} \underbrace{6 \times 8 = 48}_{\text{紧凑 token}} \xrightarrow{\text{Decoder-only Transformer}} \underbrace{\text{离散动作 token}}_{\text{输出}}
$$

整个模型约 **35M 参数**（FiLM-EfficientNet-B3 约 16M + Transformer 约 19M），在 3 Hz 控制频率下闭环运行，单次推理预算 < 100 ms。它没有像 Gato 那样把图像 patchify 成视觉 token，而是直接用 EfficientNet 的卷积特征图展平。

![](assets/RT-1%20-%20大规模真实机器人控制的Robotics%20Transformer/figures/01-overview.png)

> **图 1：RT-1 架构 / 数据集 / 评测总览。** （a）RT-1 接收图像序列与自然语言指令，输出离散化的底座与手臂动作，尽管只有 35M 参数，仍能以 3 Hz 实时运行——这得益于其「高效又高容量」的结构：FiLM 条件的 EfficientNet + TokenLearner + Transformer。（b）RT-1 的大规模真实训练（130k 演示）与评测（3000 次真实试验）展现了显著的泛化、鲁棒性与从多样数据中学习的能力。
> 来源：论文 Figure 1，第 2 页，<https://arxiv.org/abs/2212.06817>

### 3.2 关键模块一：图像与指令的 tokenization（FiLM-EfficientNet）

RT-1 用 **ImageNet 分类任务预训练的 EfficientNet-B3** 作为视觉编码器：把 6 张 $300 \times 300$ 的图像历史送入网络，取最后一个卷积层输出的空间特征图，shape 为 $9 \times 9 \times 512$，展平后得到 **81 个视觉 token**（每张图）。与 Gato 的关键区别在于：**RT-1 不做图像 patchify**，而是直接用卷积特征图，保留了空间的全局感知。

语言指令的融合发生在编码器的**早期**，而不是像 Gato 那样把语言 token 和视觉 token 并列喂给 Transformer。具体做法：

1. 指令先经 **Universal Sentence Encoder（USE）** 编码为预训练语言嵌入；
2. 该嵌入作为 **identity 初始化的 FiLM 层**（Perez et al., 2018）的输入，对 EfficientNet 的中间特征做仿射变换。

FiLM 的作用形式是逐通道的仿射调制：

$$\text{FiLM}(x) = \gamma(l) \odot x + \beta(l)$$

其中 $\gamma(l), \beta(l)$ 由两层全连接（记为 `fc` 和 `hC`）从语言嵌入 $l$ 生成。**关键工程细节**：如果把 FiLM 直接插进预训练网络内部，会破坏中间激活分布、抵消预训练权重的好处。因此 RT-1 把生成 $\gamma, \beta$ 的密集层**初始化为零**，使 FiLM 在训练初期退化为恒等映射，从而保留预训练权重的功能。作者还发现，即使从零初始化 EfficientNet，identity 初始化的 FiLM 也比随机初始化更好，但仍不如「ImageNet 预训练 + identity FiLM」的组合。整套 FiLM-EfficientNet-B3 含 26 层 MBConv + FiLM，共约 16M 参数，输出 81 个「视觉-语言」融合 token。

### 3.3 关键模块二：TokenLearner

为了给下游 Transformer 减负，RT-1 在视觉 token 之后接入 **TokenLearner**（Ryoo et al., 2021）。TokenLearner 是一个逐元素注意力模块，学习把大量 token 映射到远更少的 token，从而「软选择」信息量大的 token 组合，只把重要的 token 传给后续 Transformer。在 RT-1 中，它把每张图的 81 个视觉 token 压缩到 **8 个**。

TokenLearner 可理解为学习一组注意力权重矩阵，把 $N$ 个输入 token 聚合为 $M$ 个输出 token（$M \ll N$）：

$$Z = \rho(X)\; X, \qquad X \in \mathbb{R}^{N \times d},\ Z \in \mathbb{R}^{M \times d}$$

其中 $\rho(X) \in \mathbb{R}^{M \times N}$ 是学到的逐 token 软注意力。这一步把后续 Transformer 需要 attend 的 token 数量从 $6 \times 81 = 486$ 降到 $6 \times 8 = 48$，是 RT-1 能实时运行的关键之一（论文报告 TokenLearner 使推理提速 2.4 倍）。

### 3.4 关键模块三：Decoder-only Transformer 与动作解码

8 个 token/图 × 6 张历史图 = **48 个 token**，加上位置编码后送入 Transformer backbone。这是一个 **decoder-only 序列模型**，含 **8 层 self-attention**、约 19M 参数，输出动作 token。

### 3.5 关键模块四：动作离散化（Action Tokenization）

RT-1 把动作的每个维度都**离散化到 256 个 bin**（在每个维度取值范围内均匀划分），包括：

- 手臂 7 维：$x, y, z, \text{roll}, \text{pitch}, \text{yaw}$，夹爪开合；
- 底座 3 维：$x, y, \text{yaw}$；
- 模式 1 维：三选一——控制手臂 / 控制底座 / 终止 episode。

因此每个时间步的动作被表示为一个 11 维的 one-hot（各 256 类）集合，Transformer 输出的是每维的类别分布。这种「离散化 + 分类」的动作表示，是后续 VLA（RT-2、OpenVLA、π0 等）沿用的标准做法。

![](assets/RT-1%20-%20大规模真实机器人控制的Robotics%20Transformer/figures/02-architecture.png)

> **图 2：RT-1 架构细节图（Figure 3）。** 指令先被转成 USE 嵌入，经 FiLM 层条件化预训练的 EfficientNet；得到的视觉-语言 token 被 TokenLearner 压缩，再喂给 decoder-only Transformer，输出 tokenized 动作。这张图完整展示了「早期语言融合（FiLM）+ 视觉 token 压缩（TokenLearner）+ 序列解码（Transformer）」三层结构。
> 来源：论文 Figure 3，第 6 页，<https://arxiv.org/abs/2212.06817>

### 3.6 训练目标与推理速度

- **损失**：标准分类交叉熵 + 因果掩码（见第 2.3 节）；
- **控制频率**：RT-1 以 3 Hz 做闭环控制，直到输出「终止」动作或触达预设步数上限；作者测过人类执行同类指令需 2–4 秒，因此 3 Hz（< 100 ms 推理预算）是合理下限；
- **提速技巧**：① TokenLearner 把视觉 token 从 81 压到 8，推理提速 2.4 倍；② 历史图像窗口有重叠，重叠部分的视觉 token 只算一次、在后续推理中复用，再提速 1.7 倍。

---

## 4. 数据集与实验设置

### 4.1 数据采集规模

RT-1 的主数据集是 **约 130k 条机器人演示**，由 **13 台 Everyday Robots 移动机械臂**在 **17 个月**内采集，覆盖 **700+ 个任务指令**（论文 Table 1 精确列出 744 个）。机器人是 7 自由度手臂 + 二指夹爪 + 移动底座的移动机械臂。数据在「机器人教室（robot classroom）」——一系列仿照真实厨房搭建的台面环境——中采集；评测则在两个真实办公室厨房（Kitchen1、Kitchen2）进行，二者在光照、背景、完整厨房几何上都有偏移。

每个 episode 由人类遥操作演示，并配一条文字指令描述（通常含一个动词 + 一个或多个名词，如 “pick iced tea can”）。论文按「技能（skill）= 动词」和「物体（object）= 名词」对指令分组。

### 4.2 技能与指令构成（Table 1）

| 技能 Skill | 指令数 | 描述 |
|---|---|---|
| Pick Object | 130 | 把物体从表面拿起 |
| Move Object Near Object | 337 | 把第一个物体移到第二个物体附近 |
| Pick Object from Receptacle and Place on Counter | 162 | 从容器里拿起并放到台面 |
| Place Object into Receptacle | 84 | 把物体放进容器 |
| Place Object Upright | 8 | 把细长物体竖直放好 |
| Knock Object Over | 8 | 把细长物体碰倒 |
| Open / Close Drawer | 3 + 3 | 打开 / 关闭抽屉 |
| Sec 6.3 & 6.4 长程指令 | 9 | 真实厨房的长程任务（如开坚果罐、抽纸巾、拿舀勺） |
| **合计** | **744** | |

为提升「pick」技能的物体多样性，作者专门扩充了一批物体（Figure 2f），以检验对未见物体的泛化。

### 4.3 Baseline 与评价指标

对比两个当时最强的架构基线：**Gato**（Reed et al., 2022）和 **BC-Z**（Jang et al., 2021，SayCan 的底层策略），外加一个与 RT-1 参数规模相当的 **BC-Z XL**。注意：所有模型都**在 RT-1 采集的数据上训练**（原始 Gato/BC-Z 在各自原数据集上无法泛化到本评测），因此这个对比只检验**架构**差异，且对基线相当有利。

评价维度分四类（共 3000+ 次真实试验）：

- **Seen tasks**：训练集内指令（200+ 条），但物体摆放、时间、机器人位置都随机变化；
- **Unseen tasks**：21 条全新指令（技能与物体都在训练集出现过，但以新颖方式组合）；
- **Distractor robustness**：30 个含干扰物任务（0–9 个干扰物 + 遮挡）；
- **Background robustness**：22 个背景变化任务（新厨房、花格桌布等）。

### 4.4 实现细节

- 视觉编码器：ImageNet 预训练 EfficientNet-B3，输入 $6 \times 300 \times 300$，输出 $9 \times 9 \times 512$ 特征图 → 81 token/图；
- TokenLearner：81 → 8 token/图；Transformer：8 层 decoder-only，19M 参数；总参数量 35M；
- 语言编码：Universal Sentence Encoder；
- 动作：11 维（臂 7 + 底座 3 + 模式 1），每维 256 bin；
- 控制频率 3 Hz，闭环，推理预算 < 100 ms。

---

## 5. 实验结果

### 5.1 主要定量结果（Table 2）

| 模型 | Seen Tasks | Unseen Tasks | Distractors | Backgrounds |
|---|---|---|---|---|
| Gato | 65 | 52 | 43 | 35 |
| BC-Z | 72 | 19 | 47 | 41 |
| BC-Z XL | 56 | 43 | 23 | 35 |
| **RT-1** | **97** | **76** | **83** | **59** |

- **Seen 任务 97%**：比 BC-Z 高 25 个百分点、比 Gato 高 32 个百分点；
- **Unseen 任务 76%**：比次优基线高 24 个百分点——语言条件使策略能理解「已见概念的新组合」；
- **Distractors 83%、Backgrounds 59%**：分别比次优基线高 36%、18%。

需要强调：RT-1 的优势不只来自 Transformer，也来自「语言早期融合（FiLM）+ 视觉 token 压缩 + 离散动作」整套设计；论文在附录 D.4 对这些组件做了逐一消融。

### 5.2 真实场景泛化（Table 3，L1/L2/L3）

作者在真实厨房构造了三档递进难度：**L1**（新台面布局 + 光照）、**L2**（再加未见干扰物）、**L3**（再加全新任务设置/全新物体/水槽附近等未见位置）。RT-1 在 All / L1 / L2 / L3 上的成功率分别为 **70 / 88 / 75 / 50**，全面领先 Gato（30/63/25/0）和 BC-Z 系。其中 Gato 在 L3 直接掉到 0，而 RT-1 在最具挑战的 L3 仍保持 50%，说明其泛化是跨多轴同时成立的。

### 5.3 吸收异构数据（Table 4、Table 5）

- **吸收仿真数据**：加入「仅在仿真中出现过的物体」的仿真演示后，RT-1 在真实物体 seen 任务上仅降 2%（92→90），而在「只在仿真中见过的物体」上从 23% 跃升至 **87%（+64%）**；未见技能 + 仿真物体从 7% 提到 33%（+26%）。这显示 RT-1 能高效做域迁移，把仿真经验反哺真实世界。
- **吸收不同机器人数据**：混入 Kuka IIWA 的 QT-Opt bin-picking 数据（209k episodes，由 RL 采集、动作分布与人类演示完全不同）后，原「教室评测」只降 2%（92→90），而新的 bin-picking 评测从 22% 提升到 **39%（+17%，近 2 倍）**。反过来，仅用 Kuka 数据训练在 EDR 机器人上 bin-picking 为 0%，证明跨本体迁移很困难，但**混合数据**能让 RT-1 借他人经验学到新技能。

### 5.4 SayCan 长程组合（Table 6）

把 RT-1 作为 SayCan 框架的底层技能策略，在真实厨房执行「把桌上所有东西扔掉」这类高层指令（由 SayCan 自动拆成约 10 步的低层技能序列）。规划成功率均为 87%；**执行成功率 RT-1 在 Kitchen1 为 67%，且在更难的 Kitchen2 仍保持 67%**（不降），而 SayCan+Gato 在 Kitchen2 为 0%、SayCan+BC-Z 仅 13%。更惊人的是，RT-1 支撑了 **最多 50 步**的超长程任务（如开关未见过的抽屉、多步整理），这是低层技能成功率高企 + 对底座位置鲁棒共同作用的结果。

### 5.5 数据规模 vs 数据多样性（Table 7）

作者把数据量与多样性解耦：固定任务多样性、逐步把每任务样本上限降到 200/100/50（对应 51% / 37% / 22.5% 数据），发现性能与泛化逐步下降；而「保留 97% 数据但砍掉 25% 任务（多样性变窄）」时，泛化下降幅度相当于砍掉 49% 的数据量。结论：**数据多样性比数据量更重要**——这是对真实机器人「采什么数据」最实用的指引。

### 5.6 定性结果

![](assets/RT-1%20-%20大规模真实机器人控制的Robotics%20Transformer/figures/03-trajectories.png)

> **图 3：不同指令下的评估轨迹示例（Figure 5）。** 展示了 RT-1 在多种技能、环境、物体上的轨迹，包括「从底部抽屉拿水瓶放台面」「把海绵移到青椒薯片旁」「把红牛罐放进中间抽屉」「从纸巾盒抽纸巾」「把可乐罐竖直放」「开顶层抽屉」「从碗里拿苹果」。这些轨迹共同说明 RT-1 的单条策略能覆盖抓取、移动、放置、开关抽屉、竖直摆放等多类行为，且在真实厨房中稳定执行。
> 来源：论文 Figure 5，第 11 页，<https://arxiv.org/abs/2212.06817>

---

## 6. 与相关工作的关系

### 6.1 与同期/前作的关系

- **vs Gato（Reed et al., 2022）**：同属 Transformer 通才模型，但 Gato 把图像 patchify 成 token、语言与视觉分离、无预训练文本编码、且缺 TokenLearner 与「去自回归动作」等实时设计；本评测中 Gato 在 seen 上仅 65%。
- **vs BC-Z（Jang et al., 2021）**：BC-Z 是 ResNet 前馈模型，不用历史帧、用连续动作；RT-1 用历史图像 + 离散动作 + Transformer，在 unseen 上 76% 对 19%，差距悬殊。
- **vs 语言条件多任务先驱（CLIPort、Hiveformer、MCIL 等）**：这些是仿真为主的语言条件模仿学习；RT-1 把同一思想推到 130k 真实演示的规模。
- **vs SayCan（Ahn et al., 2022）**：SayCan 是「大语言模型规划 + 底层价值函数」的框架，RT-1 恰好成为其更强的底层技能策略（Table 6），两者是互补而非竞争关系。

### 6.2 VLA 谱系定位：RT-1 → RT-2 → OpenVLA → π0

RT-1 是「Vision-Language-Action（VLA）大模型」这条线的**奠基之作**，它确立的配方——「视觉 token + 语言条件 + Transformer 动作解码 + 动作离散化」——被后继工作反复沿用并逐步放大：

- **RT-1（2022）**：ImageNet 预训练 EfficientNet + FiLM + TokenLearner + 35M 参数的小型 Transformer，动作离散化到 256 bin。是「小模型 + 大真实数据」的 1.0 版本；
- **RT-2（2023）**：把视觉编码器 + 语言条件换成**预训练 VLM**（PaLI-X / PaLM-E），直接用 VLM 的 token 输出动作，实现「视觉语言动作一体化」，把 VLA 从「专门设计的机器人架构」推进到「互联网规模 VLM 微调」；这是 RT-1 的 VLM 化后继；
- **OpenVLA（2024）**：开源 7B 参数的 VLA（Llama-2 + SigLIP/DINOv2），在 970k 条真实机器人轨迹（含 RT-1 数据集）上训练，把 RT-1 的「数据吸收」思想扩展到跨 21 种本体的开源社区尺度，成为学界默认的 VLA 底座；
- **π0（2024）**：Physical Intelligence 的 3B 级 VLA，引入 flow matching 生成连续动作、并做跨机器人联合训练，进一步把「动作离散化」升级为连续动作生成。

一句话概括谱系：**RT-1 定配方（视觉 token + 语言 + Transformer + 动作 token），RT-2 定主干（VLM），OpenVLA 定开源底座，π0 定动作生成方式（flow matching）。** RT-1 的数据集还直接成为 VPP、OpenVLA 等后续工作的训练来源之一。

---

## 7. 局限与批判性评价

1. **模仿学习的上限**：RT-1 是行为克隆，性能受制于演示者，无法超越演示水平，也难以从失败中自我改进；
2. **泛化限于「已见概念的新组合」**：对完全没见过的运动模式（全新技能）无能为力，unseen 泛化本质是语言条件带来的组合泛化，而非真正的开放世界能力；
3. **动作离散化是双刃剑**：256 bin 带来简单、可并行、稳定的分类训练，但也引入量化误差，且 11 维动作间的联合建模被拆成逐维独立分类（后续 π0 用连续 flow matching 解决）；
4. **数据依然「单本体、单场景」为主**：主数据集来自 Everyday Robots 一种形态 + 厨房类场景，虽演示了吸收 Kuka/仿真数据的可能性，但跨本体仍靠混合数据「软迁移」，未做到真正的形态无关；
5. **语言编码器偏弱**：USE 是 2018 年的句向量模型，语义表达力远不如后来的 CLIP/PaLM/VLM，RT-2 把它替换成 VLM 是必然方向；
6. **模型规模受实时约束**：35M 参数是为 3 Hz 实时妥协的结果，容量有限；论文也坦诚「机器人领域里模型规模常不是瓶颈，延迟才是」；
7. **评测维度虽全但规模有限**：unseen 仅 21 条指令、distractor 30 个、background 22 个，统计口径偏窄，长程 SayCan 也依赖上游 LLM 规划的正确性。

---

## 8. 复现与实践建议

- **代码**：官方实现已开源（`google-research/robotics_transformer`），含 RT-1 模型与训练脚本；
- **数据**：RT-1 数据集是后续 OpenVLA 等公开训练集的一部分，可经 Google DeepMind 的 `Open X-Embodiment` 渠道获取；
- **模型要点**：
  - 视觉编码器务必用 ImageNet 预训练权重；FiLM 的 `fc`/`hC` 要零初始化，否则会破坏预训练特征；
  - TokenLearner 是实时推理的关键（81→8 token，提速 2.4 倍），压缩比例是精度/延迟的重要权衡；
  - 动作离散化用 256 bin 且均匀划分，loss 用分类交叉熵 + 因果掩码；
  - 历史图像窗口重叠部分 token 可缓存复用，再省 1.7 倍；
- **实践建议**：论文最反直觉的结论是「数据多样性 > 数据量」——在预算有限时，与其堆同质数据，不如把有限的采集预算花在**更多样化的任务/物体/场景**上；若想跨本体迁移，混合多机器人数据比单机器人数据更有效；
- **常见踩坑**：直接用原版 1.2B Gato 在真机上推理要 1.9 秒，远超实时预算，必须压缩到与 RT-1 同量级（35M 上下）才能上真机闭环。

---

## 9. 个人启发与后续问题

1. **「配方」比「规模」先行**：RT-1 只有 35M 参数，却靠「早期语言融合 + token 压缩 + 离散动作」在真实机器人上立住了 VLA 范式。它证明：在真实机器人这个延迟敏感、数据昂贵的领域，**架构效率与数据多样性的价值不亚于模型规模**。
2. **数据是真正的护城河**：130k 条真实演示、13 台机器人、17 个月——RT-1 最难以复刻的不是模型，而是这套规模化采集管线与随之而来的数据集（它养活了 RT-2、OpenVLA、VPP 等一大批后续工作）。这提示我们：做机器人大模型，数据工程才是「隐形主角」。
3. **离散动作的得与失**：256 bin 的分类范式简单稳定、天然适合 Transformer 解码，但量化误差与逐维独立假设是硬伤；从 RT-2 沿用、到 π0 用 flow matching 弃用，这条「离散 → 连续」的演化线值得深挖。
4. **语言融合的时机**：RT-1 用 FiLM 在编码器内部做「早期融合」，显著优于 Gato 的「晚期并列」——这暗示多模态条件注入的**位置**对机器人策略至关重要，后续 VLA 普遍用 cross-attention 在骨干内部注入语言，延续了同一直觉。
5. **未解问题**：（i）USE 换成 VLM 后，早期融合还有多大增益？（ii）数据多样性的「最优配方」是什么——多少任务、多少物体、多少场景才能逼近泛化上限？（iii）跨本体是否可能做到真正的形态无关，而非靠混合数据软迁移？（iv）模仿学习的上限能否通过「演示 + 自举」被打破？

---

## 10. 材料来源

### Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2212.06817> | 全文（15 页 + 附录） | 2026-08-27 | 精读主来源 |
| `figures/01-overview.png` | Figure 1：RT-1 架构/数据集/评测总览 | arXiv:2212.06817 | 第 2 页 | 2026-08-27 | 总体框架 |
| `figures/02-architecture.png` | Figure 3：RT-1 架构细节（FiLM + TokenLearner + Transformer） | arXiv:2212.06817 | 第 6 页 | 2026-08-27 | 核心方法 |
| `figures/03-trajectories.png` | Figure 5：不同指令下的评估轨迹示例 | arXiv:2212.06817 | 第 11 页 | 2026-08-27 | 定性结果 |
