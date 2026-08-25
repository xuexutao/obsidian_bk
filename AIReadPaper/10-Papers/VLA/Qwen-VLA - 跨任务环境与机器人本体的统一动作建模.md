---
type: paper-note
status: done
domain: VLA
paper: Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
year: 2026
arxiv: 2605.30280
doi: null
source: https://arxiv.org/abs/2605.30280
project: https://qwen.ai/blog?id=qwenvla
code: https://github.com/QwenLM/Qwen-VLA
tags:
  - VLA
created: 2026-05-31
updated: 2026-08-21
---

# Qwen-VLA：跨任务、环境与机器人本体的统一动作建模

## 基本信息

| 项目 | 内容 |
|------|------|
| **论文标题** | Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments |
| **作者团队** | Qwen Team（核心贡献者含 Qiuyue Wang、Mingsheng Li、Jian Guan、Jinhui Ye 等） |
| **arXiv 编号** | 2605.30280v2 [cs.RO] |
| **发布日期** | 2026-06-01（v2） |
| **项目页** | https://qwen.ai/blog?id=qwenvla |
| **代码仓库** | https://github.com/QwenLM/Qwen-VLA |
| **类别** | VLA（Vision-Language-Action）/ 具身基础模型 |

## 一句话结论

Qwen-VLA 以 Qwen3.5-4B 为视觉语言骨干，外挂 DiT 流匹配动作专家，把"机械臂操作、视觉语言导航、人类第一视角动作、轨迹预测"四类异构具身任务统一改写为同一动作-轨迹预测问题，并通过 embodiment-aware prompt 条件、$H\times K$ 统一动作张量与四阶段渐进训练（T2A → CPT → SFT → RL）实现跨任务、跨环境、跨机器人本体的联合学习；最终在 LIBERO 97.9%、Simpler-WidowX 73.7%、RoboTwin-Easy/Hard 86.1%/87.2%、R2R OSR 69.0%、RxR SR 59.6%、真实 ALOHA 平均 OOD 76.9% 与 DOMINO 零样本 26.6% SR 上同时取得领先或可比结果，是 Qwen 系列从"感知-理解-推理"向"可执行动作"扩展的标志性工作。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

具身智能的目标是让智能体能"看物理世界、理解自然语言指令、推演时空上下文，并执行动作完成任务"。在视觉语言模型（VLM）一侧，Qwen 系列等通用 VLM 已经把开放世界视觉理解与语言 grounding 推到了非常成熟的水平；而以 $\pi_0$、$\pi_{0.5}$ 为代表的扩散/流匹配策略也在建模连续高维机器人动作上展现了很强的潜力。然而现有的具身系统仍然存在严重的"任务/本体碎片化"：操作模型多面向桌面或灵巧手控制，导航模型面向室内 waypoint 或动作预测，训练数据、控制接口、动作维度、评估协议都不一样。这种碎片化阻碍了跨任务、跨环境、跨本体的迁移，也让具身学习难以像通用 VLM 那样靠大规模预训练扩展。

Qwen-VLA 要回答的核心问题因此可以表述为：**这些表面异构的具身决策问题，能否被统一到同一个视觉-语言-动作模型中？** 更具体地，它要同时回答四件事：

1. 怎么把机械臂操作、视觉语言导航、人类第一视角动作、轨迹预测放进同一个建模框架？
2. 怎么在不强行统一物理语义的前提下，兼容不同机器人本体、控制接口、动作维度？
3. 怎么让 VLM 的视觉理解、空间 grounding 与语言推理真正接到连续动作生成上？
4. 怎么让联合预训练得到的表征在多 benchmark、OOD 场景、真实机器人部署上体现收益？

### 1.2 现有方法的瓶颈

论文明确指出现有方法的三类瓶颈：

- **任务专用化（task-specialization）**：多数 embodied system 只在窄任务族、单一本体或固定评测设定下训练，泛化能力受限于训练分布；
- **动作接口异质性**：机械臂末端位姿与关节角并存、绝对与增量控制并存、单臂与双臂并存、是否含夹爪/灵巧手/底盘并存，使得"统一模型"在工程上看起来很贵；
- **认知与运动脱节**：VLM 预训练强、动作解码器随机初始化，直接联合训练既低效也不稳定，decoder 要同时学动作分布形状、语言条件、flow matching 动力学和视觉 grounding 四个问题。

### 1.3 本文核心贡献

论文把贡献凝练为四点：

1. **统一 VLA 模型**：以 Qwen3.5-4B 为视觉语言骨干 + DiT 流匹配策略头，把操作、导航、第一视角动作建模统一到共享动作-轨迹空间，单模型支持多平台多任务族；
2. **大规模联合预训练混合数据**：覆盖多机器人操作轨迹、人类第一视角、合成仿真、导航、轨迹中心监督和精选视觉-语言数据，并提出 embodiment-aware prompt 条件统一多平台；
3. **渐进式四阶段训练配方**：T2A 动作预训练 + CPT 持续预训练 + SFT 监督微调 + RL 强化学习，弥合离散 VL token 与连续动作轨迹之间的差距；
4. **全面评估**：在操作、导航、OOD 鲁棒性、跨本体泛化基准上同时验证，证明显式预训练 + 渐进学习在场景/物体/光照/本体变化下都改善多任务性能。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

Qwen-VLA 把所有具身任务统一写成一个条件预测问题。在时间步 $t$，模型接收：

- 视觉上下文 $o_t$：单帧或多帧图像、视频片段、历史窗口；
- 语言指令 $x$：自然语言任务描述；
- 具身描述 $e$：描述当前机器人平台、臂型、控制频率、动作长度等元信息的文本提示（见 2.3 节模板）；
- 任务标识 $z$（可选）：当任务族需要时显式给出。

模型被训练为在预测视界 $H$ 内输出目标序列 $y_{t:t+H-1}$，形式化为：

$$p_{\theta}(y_{t:t+H-1} \mid o_t, x, e, z)$$

输出 $y_{t:t+H-1}$ 在不同任务族里具有不同物理语义，但都在同一个动作-轨迹张量空间里：

- **机械臂操作**：末端位姿增量、关节角、夹爪开合、灵巧手关节角；
- **视觉语言导航**：相对地面位移 $(\Delta x, \Delta y, \Delta \theta)$ 的 waypoint 序列；
- **轨迹预测/自动驾驶**：连续坐标空间中的未来轨迹；
- **人类第一视角动作**：相对手腕 SE(3) 位姿 + 关节角 eigengrasp 系数。

这一抽象层次的统一，是论文后续所有工程设计（统一张量、embodiment prompt、四阶段训练）的基石。

### 2.2 关键符号和目标函数

为方便后续讨论，先把核心符号固定下来：

| 符号 | 含义 |
|---|---|
| $H$ | 固定预测视界（时间步数） |
| $K$ | 固定通道维度（所有控制模式共享的最大通道数） |
| $c$ | 某一具体 embodiment 实际使用的有效通道数（$c \le K$） |
| $\mathbf{Y}_0 \in \mathbb{R}^{H \times K}$ | 干净目标动作张量 |
| $\mathbf{Y}_1 \sim \mathcal{N}(0, I)$ | 噪声张量 |
| $\mathbf{Y}_\tau = (1-\tau)\mathbf{Y}_0 + \tau \mathbf{Y}_1$ | 时刻 $\tau \in [0, 1]$ 的线性插值 |
| $\mathbf{M} \in \{0, 1\}^{H \times K}$ | 逐通道有效位掩码 |
| $v_\theta(\mathbf{Y}_\tau, \tau \mid o_{1:t}, x, e, z)$ | DiT 动作专家预测的速度场 |

整体训练目标为 action loss 与 VL loss 的加权和：

$$\mathcal{L} = \lambda_{\text{act}} \mathcal{L}_{\text{act}} + \lambda_{\text{vl}} \mathcal{L}_{\text{vl}}$$

其中 $\mathcal{L}_{\text{act}}$ 是带逐通道两级平均的 flow-matching 损失（详见 §3.4），$\mathcal{L}_{\text{vl}}$ 是标准的视觉语言 next-token 预测损失：

$$\mathcal{L}_{\text{vl}} = -\sum_i \log p_{\theta}(w_i \mid w_{<i}, o_{1:t})$$

---

## 3. 核心方法

### 3.1 总体框架

Qwen-VLA 的总体框架如图 1 所示：左侧是 VLA（操作）、VLN（导航）、VL（视觉语言理解）三类异构数据源，模型用 Qwen3.5 多模态骨干统一编码观察图像与文本（含 embodiment prompt），再把 VLM 隐藏状态与"噪声动作块"拼成一条序列，送给 DiT 动作专家做去噪，输出清洁动作序列；最终在右侧同时支持 manipulation 与 navigation 任务。整体结构可以理解为"通用 VLM 认知 + 专用 DiT 动作小脑"的双流协同。

![](assets/Qwen-VLA%20-%20跨任务环境与机器人本体的统一动作建模/qwenvla_overview.png)

> 图 1：Qwen-VLA 总体架构。该图说明 Qwen3.5 VLM 把视觉 token、文本（含 embodiment prompt）和噪声动作 token 拼成单一流，DiT 动作专家通过 AdaLN 时间步条件做去噪；同一模型同时覆盖 VLA、VLN、VL 三类任务。来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2605.30280

四阶段训练流程如图 2 所示：(I) T2A 冻结 VLM 只训 DiT，刻意不给图像，让 decoder 学"语言到动作的解压缩"；(II) CPT 同时解冻两个模块，把动作先验接到视觉观测上；(III) SFT 分出多任务和真机两条并行支路；(IV) RL 用 PPO + 稀疏二元成功奖励在仿真里把任务成功率接进来，得到 Qwen-VLA-Instruct。

![](assets/Qwen-VLA%20-%20跨任务环境与机器人本体的统一动作建模/qwenvla_training_recipe.png)

> 图 2：Qwen-VLA 的四阶段训练配方。T2A 阶段不带图像只解压缩动作；CPT 阶段引入视觉 grounding；SFT 阶段分出多任务与真机两条路；RL 阶段在仿真里用环境奖励优化闭环任务成功率。来源：论文 Figure 2，第 5 页，https://arxiv.org/abs/2605.30280

### 3.2 关键模块一：Qwen3.5 视觉语言骨干

Qwen-VLA 选用 Qwen3.5-4B（论文中写作 Qwen3.5-4B）作为视觉语言骨干。Qwen3.5 是早期视觉-语言融合的原生多模态模型：ViT 通过空间合并产生视觉 token，再与文本 token 交错送入同一 transformer；其混合注意力机制在多数层使用门控线性注意力（gated linear attention），并按固定间隔插入分组查询 softmax 注意力，兼顾长多模态序列的高效编码与全局推理精度。

论文把 Qwen3.5-4B 的选型作为关键的设计前提，原因有三：

- 强细粒度视觉感知、稳健指代 grounding、多语言指令跟随与结构化推理能力，对具身任务中"空间关系/被指代物体/多步指令"三类高频需求是直接相关的能力；
- 它本身就是强预训练后的视觉语言模型，避免"从头学看图"的开销；
- Qwen 系列的开放生态让 Qwen-VLA 的工程实现门槛相对低。

### 3.3 关键模块二：DiT 流匹配动作专家

Qwen-VLA 在 VLM 之上挂一个**单流 DiT 风格 flow matching 策略**作为 action expert。其核心设计是：

- 把 VLM 隐藏状态与噪声动作块拼成一条序列，做**联合自注意力**，并配 **AdaLN 时间步条件**和**与骨干对齐的多段 RoPE**；
- 用 **flow-matching 目标**训练，推理时用少量 Euler 积分步生成动作序列，从而获得低延迟实时控制；
- 整个动作专家约含 1.15B 参数：16 个 DiT 块每块 70.8M，合计 1.13B；其余分配在 action projection MLP（4.9M，把原始动作维度映射到 DiT 隐空间）、VLM 隐藏状态线性变换（3.9M）、时间步嵌入（2.8M）和输出 AdaLN 调制（4.7M）。

DiT 风格解码器相比离散 token 自回归头有两个本质优势：

1. 它天然适合**连续、高维、时序相关**的动作分布，避免把动作强制离散化后丢失精度；
2. 联合自注意力让动作 token 与 VLM 隐藏状态在每一层都做交互，**视觉 grounding 与动作生成真正耦合**而不是拼接。

#### 3.3.1 Embodiment-aware Prompt 条件

为了让同一模型兼容多种机器人本体，论文为每个训练样本前置一段 embodiment-specific 文本提示，模板为：

> The robot is {robot_tag} with {single arm / dual arms}[, waist][, and mobile base]. The control frequency is {FPS} Hz. Please predict the next {chunk_size} control actions to execute the following task: {ori_instruction}.

这段 prompt 同时描述了：机器人 tag（WidowX、ALOHA、AgiBot A2-D…）、臂型（单/双，含腰、含移动底盘可选）、控制频率 FPS、预测 chunk 大小、原始任务指令。导航样本使用类似结构描述导航约定与 waypoint 视界。**prompt 是模型唯一知道"当前我在控制哪种身体"的接口**，所以部署时只需要替换 prompt 段，VLM 骨干与 DiT 解码器完全不用动。

#### 3.3.2 统一动作张量与通道掩码

每个训练样本贡献目标张量 $\mathbf{Y} \in \mathbb{R}^{H \times K}$，其中：

- $H$：固定预测视界；
- $K$：所有控制模式共享的最大通道维度；
- 控制模式实际占用 $c \le K$ 个通道，**有效值放在前 $c$ 维，其余 $K-c$ 维零填充**；
- 同时维护一个**逐通道二值掩码** $\mathbf{M} \in \{0,1\}^{H\times K}$，记录哪些通道、时间步是有效信号：$M_{h,k} = 1$ 当且仅当 $k < c$ 且 $h$ 处于任务的 chunk 长度 $H_{\text{task}} \le H$ 之内。

这一设计带来三个好处：

- **不强制统一物理语义**：每个数据集保留自己原生控制约定（绝对/相对、位姿/关节、含/不含夹爪/灵巧手），由 embodiment prompt 通知模型；
- **不需要 embodiment-specific 输出头**：同一套 DiT 参数处理所有控制模式；
- **padding 不污染梯度**：掩码使 loss 不会从零填充位置反向传播。

#### 3.3.3 RoboInF 合成仿真数据

为了让合成仿真数据既有任务级抽象又具备可扩展性，论文自建了 RoboInF 仿真流水线（基于 IsaacLab + cuRobo 运动规划），如图 3 所示。它生成两类互补数据：

- **视觉-语言-动作数据**：20 个桌面场景 × 10 种物体初始位姿 = 200 个基础配置，450 个操作任务（短视界+长视界），每任务 300 条成功轨迹，含环境和执行增强；视觉随机化覆盖约 3K 背景 + 1K 桌面纹理。最终得到约 359,848 条完整成功轨迹（含子任务片段）。
- **语言-动作数据**：6 种任务模板（pick-and-place、线性推动、线性拉动、旋转重定位、朝向视点旋转、两物体位置交换）× 6 种单臂机器人配置（Franka Panda、UR10e、UR5e、Kinova Gen3、TM12、xArm7），每机器人-任务对约 200k 条轨迹，总计约 7.2M 条轨迹、超过 14,000 小时，50 Hz 记录关节位置、速度、末端位姿、夹爪状态。这部分**主要用作 Stage I T2A 预训练语料**。

![](assets/Qwen-VLA%20-%20跨任务环境与机器人本体的统一动作建模/qwenvla_sim_demo.png)

> 图 3：RoboInF 合成的仿真数据示例。上两行为短视界任务（"Place the two green staplers side by side" 与 "Turn the cake server so it points toward the left side of the table"），下行为长视界任务"Group the drinks together and leave the cleaning sponge by itself"及其子任务分解（Pick 7Up / Place 7Up / Pick Red Bull / Place Red Bull / Pick Sponge / Place Sponge）。长视界任务在合成时被自动切成多个子片段，为模型提供分阶段监督。来源：论文 Figure 3，第 9 页，https://arxiv.org/abs/2605.30280

### 3.4 训练目标与损失函数

Qwen-VLA 的训练目标由两个互补部分组成：

**Flow-matching 动作损失（带两级平均）。** 对每个有连续控制目标的样本，定义每个有效通道的均方误差：

$$\ell_k = \frac{\sum_{h=1}^{H} M_{h,k}\,\big\|\big(v_{\theta}(\mathbf{Y}_{\tau},\tau \mid o_{1:t},x,e,z) - (\mathbf{Y}_1 - \mathbf{Y}_0)\big)_{h,k}\big\|_2^2}{\sum_{h=1}^{H} M_{h,k}}$$

再对 $c$ 个有效通道均匀平均：

$$\mathcal{L}_{\text{act}} = \mathbb{E}_{\tau,\mathbf{Y}_0,\mathbf{Y}_1}\!\left[\frac{1}{c}\sum_{k=0}^{c-1}\ell_k\right]$$

**两级平均的直觉**：先对每个有效通道单独平均，再对所有有效通道均匀平均，确保每个控制维度对梯度的贡献相等；padding 位置被掩码完全排除。这样 7-DoF 末端位姿和 29-DoF 全身控制能在同一个 loss 下公平训练。

**视觉语言损失。** 在辅助 VL 数据、细粒度具身动作标注、自动驾驶 VQA、通用 VL 语料上保留标准 next-token 预测目标 $\mathcal{L}_{\text{vl}}$，防止 VLM 骨干在大量具身数据上发生灾难性遗忘。

**联合目标**为 $\mathcal{L} = \lambda_{\text{act}} \mathcal{L}_{\text{act}} + \lambda_{\text{vl}} \mathcal{L}_{\text{vl}}$。在 SFT 阶段，论文使用 $\lambda_{\text{act}}=1.0$（动作）与 $\lambda_{\text{vl}}=0.1$（VL next-token）。

**Flow-matching 时间步分布的两段式选择。** 论文在 §5.2.1 消融中发现一个反直觉但有解释力的现象：T2A 阶段没有视觉条件，decoder 主要靠语言到动作的解压缩，因此 Sigmoid-Normal（密度峰在中间噪声水平）比标准 Beta（密度峰靠近 clean 端）更能提供有效梯度；CPT/SFT 阶段有了 VLM 隐藏状态做条件，Beta 分布反而更省样本。论文最终固定"T2A 用 Sigmoid-Normal、CPT/SFT 用 Beta"为默认配置。

### 3.5 推理流程与复杂度

推理时，Qwen-VLA 用少量 Euler 积分步（论文未披露具体步数，标注"论文未披露"，但文本多次强调"few Euler steps"，与 $\pi_0$ 一类工作类似可推测为 5-10 步）从 $\tau=1$ 迭代到 $\tau=0$ 生成动作序列。RL 阶段额外做了两件事：

- **温度从 $\tau=1.0$ 降到 $\tau=0.6$**：在训练-推理一致性上保留适度探索，在部署时更确定；
- **Flow-matching 下的对数概率估计**：把确定性概率流 ODE 转成 SDE，在每步 Euler 去噪中注入受控噪声，使每步成为显式高斯分布，可解析计算对数概率；每 rollout 随机选一个去噪步估计对数概率。

**复杂度量级（论文未给出明确数字，标注"论文未披露"）**：骨干 Qwen3.5-4B（约 4B 参数）+ DiT 1.15B，总计约 5B+ 级别参数；每次前向编码一次当前帧与一段 prompt，再做若干步 Euler 解码。论文提到 RL 阶段使用 128 个并行环境实例、每迭代 8 个 rollout epoch × 128 环境步、每迭代 8,192 个 $H=16$ 的转移块，属于较标准的解耦客户端-服务器架构。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

Qwen-VLA 的预训练混合数据是整篇工作最具工程含量的部分之一，按论文表 1 披露的配比共八类：

| 数据来源 | 占比 (%) |
|---|---|
| 机器人操作轨迹 | 74.2 |
| 导航轨迹 | 7.5 |
| 人类第一视角轨迹 | 6.0 |
| 合成仿真轨迹（自有） | 3.7 |
| 通用视觉-语言数据 | 3.4 |
| 空间定位（2D） | 2.5 |
| 自动驾驶 VQA | 2.4 |
| 细粒度具身动作标注 | 0.2 |

**机器人操作轨迹**包括 RobotSet、Galaxea、AgiBot World、RoboCOIN、RoboMIND V1/V2、RDT-1B、DROID、BridgeData V2、RH20T、RT-1、BC-Z 等公开数据（合计超过 10,000 小时），加上 1,000+ 小时自采真机轨迹和 8M+ 合成仿真轨迹，覆盖的具身类型与控制约定见论文表 2：WidowX、Google Robot、Franka Panda（单/双）、ARX5、Fourier GR-1、Mobile ALOHA、AgiBot A2-D（含 DH 灵巧手）、Galaxea R1、AIRBOT MMK2（含 DH）、TienKung（含 DH）、Real Human（MANO $\Delta$EEF）等。

**第一视角人类数据**来自四个来源：(1) Ego4D + EPIC-KITCHENS（经 VITRA 处理），(2) EgoDex（829 小时、194 个桌面任务，Apple Vision Pro 采集），(3) EgoVerse（1,300+ 小时、1,965 任务、240 场景），(4) Xperience（同步第一视角 + 深度 + 手/体运动 + 层级语言）。动作表示为相对手腕 SE(3) 平移+轴角（每只手 6 维）+ 10 维 eigengrasp（45 维手部关节角 PCA 降维），合计每时间步 32 维。

**导航数据**包含指令跟随（4.3%）、物体搜索（2.3%）、目标跟踪（1.0%）三子类，移动机器人假设 3 自由度（平面平移 + 航向旋转），导航视频采样 2 FPS。

**视觉-语言数据**包含细粒度具身动作标注（约 48,000 条视频-标注对，13 维动作原语+执行者+物体+接触区域+源/目标+轨迹+夹爪+身体运动）、自动驾驶 VQA（LingoQA、DriveAction、MMAU、Impromptu-VLA、nuScenes-QA、nuScenes-MQA、MapLM、WaymoQA、CODA-LM、Talk2Car、DrivingVQA、DriveLM、W3DA、GRAID、Bench2Drive-VL、DriveGPT4、OmniDrive、Senna、NAVSIM-RecogDrive、NAVSIM-Traj 等）、2D 边界框空间定位数据、通用 VL 数据（OCR、指代、空间关系、视频中心与 3D 感知）。

**关键数据处理细节**：

- **动作归一化**：用分位数归一化，对每个数据集每个动作维度计算第 1/99 百分位后线性映射到 $[-1,1]$；
- **语言质量过滤**：语言标注与观测运动不一致的轨迹被丢弃；
- **相机视角表示**：每张图像用视角特定边界 token 包裹，如 `<|tag_start|> 〈image〉<|tag_end|>`，让 VLM 知道这是 ego / cam_left_wrist / cam_right_wrist 中的哪一个；
- **伪动作恢复**：无显式 action 标签的数据集通过 proprioceptive 状态序列的有限差分得到伪动作；
- **多阶段清洗**：剔除坏帧、近零方差动作序列、异常长度 episode。

### 4.2 Baseline 与评价指标

论文在五个层面做评估：

1. **仿真操作 benchmark**（论文表 4）：LIBERO、RoboCasa-GR1、Simpler-WidowX、RoboTwin-Easy、RoboTwin-Hard 上的任务成功率；
2. **真实 ALOHA 操作**（论文表 5-6）：六类 in-domain 任务（Pick and Place、Table Cleaning、Bowl Stacking、Bowl Pick & Place、Towel Folding、Fine-grained Manipulation）与五类 OOD 任务（Color、Instance、Position、Background、Instruction）的平均成功率；
3. **导航**（论文表 7）：VLN-CE 的 R2R 与 RxR Val-Unseen 划分上的 NE↓、OS↑、SR↑、SPL↑、（RxR nDTW↑）；
4. **静态操作 OOD**（论文表 8）：自建 SimplerEnv-OOD，6 个 OOD 任务（MoveAway / MoveRight / PlaceNear / PlaceRight / PutFront / StackYellow）平均成功率；
5. **动态操作 OOD**（论文表 9）：DOMINO（35 个 suite）的 SR% 与 MS（Manipulation Score）。

**主要 Baseline**包括 $\pi_0$、$\pi_{0.5}$、GR00T N1.6、OpenVLA、OpenVLA-OFT、$\pi_0$-FAST、RDT-1B、InternVLA-M1、VLA-Adapter、StarVLA-OFT、ABot-M0、Being-H0.5、PUMA、LingBot-VA、NaVid、Uni-NaVid、NaVILA、StreamVLN 等 Specialist，以及 Qwen-VLA-Base（CPT 后）与 Qwen-VLA-Instruct（SFT+RL 后）两个 Generalist 变体。

### 4.3 实现细节

- **骨干**：Qwen3.5-4B；
- **动作专家**：16 个 DiT 块，每块 70.8M，合计 1.13B，加 projection/timestep/output AdaLN 后约 1.15B；
- **预测视界**：操作 SFT $H=16$、导航 SFT 每块 8 个 waypoint；
- **SFT 损失权重**：动作 1.0 / VL next-token 0.1；
- **RL 算法**：PPO + GAE，$\gamma=0.99$、$\lambda=0.95$、$\epsilon=0.2$，每 rollout 4 个优化 epoch；
- **价值头**：轻量级 head 挂 VLM 隐藏状态上做 mean-pool + 线性投影到标量，对 VLM 隐藏状态施加 stop-gradient，价值头学习率 $10^{-4}$（约为 actor 的 $5\times 10^{-6}$ 的 20 倍）；
- **奖励**：稀疏二元 $R \in \{0,1\}$，完全由仿真器 ground-truth 提供，无学习奖励模型；
- **RL 基础设施**：解耦客户端-服务器，128 个并行环境实例，每次迭代 8 个 rollout epoch × 128 环境步，每迭代 8,192 个 $H=16$ 的转移块；
- **推理温度**：从训练时 $\tau=1.0$ 降到部署时 $\tau=0.6$。

---

## 5. 实验结果

### 5.1 主要定量结果

#### 5.1.1 仿真操作（论文表 4）

| 方法 | 类型 | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard |
|---|---|---|---|---|---|---|
| $\pi_0$ | Specialist | 94.4 | – | – | 65.9 | 58.4 |
| StarVLA-OFT | Specialist | 96.6 | 48.8 | 64.6 | 50.4 | – |
| GR00T N1.6 | Specialist | 97.2 | 49.9 | 63.2 | 47.6 | – |
| $\pi_{0.5}$ | Specialist | 97.6 | 37.0 | 46.9 | 82.7 | 76.8 |
| ABot-M0 | Specialist | 98.6 | 58.3 | – | 86.0 | 85.0 |
| Being-H0.5 | Specialist | 97.6 | 53.3 | – | – | – |
| **Qwen-VLA-Base** | **Generalist** | **90.8** | **40.4** | **64.3** | **64.3** | **66.4** |
| **Qwen-VLA-Instruct** | **Generalist** | **97.9** | **56.7** | **73.7** | **86.1** | **87.2** |

**结果说明了什么**：Qwen-VLA-Instruct 在五个 benchmark 的四个上超过或追平所有 specialist，唯一的明显弱项是 LIBERO 略低于 ABot-M0（97.9 vs 98.6）。从 Qwen-VLA-Base 到 Qwen-VLA-Instruct 的提升非常显著（LIBERO +7.1、RoboCasa-GR1 +16.3、Simpler-WidowX +9.4、RoboTwin-Easy +21.8、RoboTwin-Hard +20.8），说明 SFT+RL 后训练对 generalist 模型的"专业化"是有效且必需的；同时也证明，**单一统一模型完全有能力在多个 manipulation benchmark 上同时达到 specialist 水平**。

#### 5.1.2 真实 ALOHA 操作（论文表 5-6）

| 模型 | Pick&Place | Table Cleaning | Bowl Stacking | Bowl Pick&Place | Towel Folding | Fine-grained | Avg. |
|---|---|---|---|---|---|---|---|
| GR00T N1.6 | 30.8 | 38.5 | 53.8 | 19.2 | 19.2 | 10.3 | 28.6 |
| $\pi_{0.5}$ | 73.1 | 84.6 | 88.5 | 69.2 | 80.8 | 33.3 | 71.6 |
| Qwen-VLA-aloha w/o pretrain | 30.8 | 53.8 | 61.5 | 64.1 | 50.0 | 30.8 | 48.5 |
| **Qwen-VLA-aloha w/ pretrain** | **96.2** | **92.3** | **98.7** | **87.2** | **65.4** | **61.5** | **83.6** |

OOD 平均成功率（论文表 6）：Qwen-VLA-aloha w/ pretrain 76.9%，$\pi_{0.5}$ 41.5%，GR00T N1.6 25.4%，Qwen-VLA-aloha w/o pretrain 36.2%。尤其在 Background（80.8）和 Instruction（84.6）上提升最显著。

**结果说明了什么**：在 ALOHA 真实硬件上，预训练带来的收益**远不只是架构红利**。同一个 Qwen-VLA 架构，从头训练只能拿到 48.5% / 36.2% 的域内/OOD 成功率，而从 Qwen-VLA-Base 微调则能到 83.6% / 76.9%，提升 35-40 个百分点。预训练让模型具备了"在未见过的颜色、物体、位置、背景、指令下还能做事"的能力，这部分增量主要来自大规模多源联合预训练，而不是新引入的 DiT 头或 embodiment prompt。

#### 5.1.3 导航（论文表 7）

| 方法 | R2R NE↓ | R2R OS↑ | R2R SR↑ | R2R SPL↑ | RxR NE↓ | RxR SR↑ | RxR SPL↑ | RxR nDTW↑ |
|---|---|---|---|---|---|---|---|---|
| NaVid | 5.7 | 49.2 | 41.9 | 36.5 | 5.7 | 45.7 | 38.2 | – |
| Uni-NaVid | 5.6 | 53.3 | 47.0 | 42.7 | 6.2 | 48.7 | 40.9 | – |
| NaVILA | 5.2 | 62.5 | 54.0 | 49.0 | 6.8 | 49.3 | 44.0 | 58.8 |
| StreamVLN | 5.0 | 64.2 | 56.9 | 51.9 | 6.2 | 52.9 | 46.0 | 61.9 |
| Qwen-VLA-Base | 5.2 | 61.7 | 53.8 | 49.4 | 6.4 | 55.1 | 45.8 | 56.2 |
| **Qwen-VLA-Instruct** | **5.1** | **69.0** | **57.5** | 51.2 | **5.8** | **59.6** | **47.8** | 57.1 |

**结果说明了什么**：Qwen-VLA-Instruct 在 R2R 拿到 OSR 69.0、SR 57.5，在 RxR 拿到 SR 59.6、SPL 47.8，都在多数指标上领先或追平 specialist 导航模型。值得注意的是，Qwen-VLA 是**一个同时在做操作任务的模型**，仍然在 VLN 上击败了 NaVILA、StreamVLN 这类专用导航模型，这说明把导航以 waypoint 形式塞进统一动作张量是可行且有效的。

### 5.2 定性结果

图 4 展示了真实 ALOHA 评估任务的整体图谱，中间是 6 类 in-domain 任务的 rollouts，四周是 5 类 OOD 任务（Color、Instance、Position、Background、Instruction）的示例。可以看到 OOD 任务在颜色、物体实例、位置、背景、光照（夜间 RGB 灯光）、指令措辞等多个维度上系统地变化，让 OOD 评估具备较强诊断力。

![](assets/Qwen-VLA%20-%20跨任务环境与机器人本体的统一动作建模/qwenvla_task_overview.png)

> 图 4：ALOHA 双臂平台真实评估任务总览。中间 6 条 in-domain 任务（Pick and Place、Table Cleaning、Bowl Stacking、Bowl Pick & Place、Towel Folding、Fine-grained Manipulation）展示了 3 个 RGB 相机视角下的连续操作；两侧为 5 类 OOD 评估（Color、Instance、Position、Background、Instruction），分别测试模型对未见颜色、未知物体实例、未见空间位置、未见背景/光照、未见指令措辞的泛化能力。来源：论文 Figure 4，第 16 页，https://arxiv.org/abs/2605.30280

图 5 是 ALOHA 上 4 类 OOD 定性 rollouts 的 4×4 网格。可以看到：左上四宫格是颜色 grounding（红/绿/蓝/黄球按指令抓取）；右上两行为"干净桌面"组合任务（依次抓取蓝伞、鸭子、酸奶放入 bin）；左下为完全未见物体（太阳镜、毛绒玩具、鸭子）的"approach"指令；右下是拔笔帽并放回桌面的精细两阶段操作，且背景换成黄色未见场景。整体上模型在物体识别、指令理解、组合任务分解、精细两阶段动作上都展现了较强鲁棒性。

![](assets/Qwen-VLA%20-%20跨任务环境与机器人本体的统一动作建模/qwenvla_realgrid.jpg)

> 图 5：Qwen-VLA-Base 在 ALOHA 双臂平台上的定性 OOD 泛化。左上：颜色 grounding（红/绿/蓝/黄球按指令抓取）；右上：组合任务"clean up the table"（蓝伞、玩具鸭、瓶装酸奶依次入筐）；左下：完全未见物体（太阳镜、毛绒玩具、玩具鸭）的"approach"指令；右下：拔笔帽并放回桌面的精细两阶段操作，背景换成未见过的黄色。模型在多种 OOD 条件下都能完成抓取、组合和精细两阶段动作。来源：论文 Figure 5，第 20 页，https://arxiv.org/abs/2605.30280

### 5.3 消融实验

#### 5.3.1 T2A 预训练消融（论文图 6）

论文在 Simpler-WidowX 上做了一组 T2A 消融，所有变体在 CPT+SFT 之后评测：

- **数据配比**：纯真实 T2A 数据 51.0%、纯合成 64.1%、$\sim$20% 合成 + 80% 真实最优 71.1%，比无 T2A baseline（60.9%）提升 +10.2 pp；
- **预测模式**：full-sequence 全面优于 chunk（10% 合成时 +4.9 pp，纯真实时 +2.9 pp）；
- **视觉输入**：T2A 阶段加入图像会拖累 2.9 pp（10% 合成时 chunk 模式 57.6% vs 60.4%），验证"图像应被完全抑制在 T2A 阶段"的设计选择；
- **flow-matching 时间步分布**：T2A 用 Sigmoid-Normal + SFT 用 Beta 最优 71.1%；T2A 用 Beta 跌到 65.4%；SFT 用 Sigmoid-Normal 跌到 62.8%；都 Beta 最差 59.4%；
- **T2A 训练时长**：2k 步峰值 71.1%，4k 步 67.5%，10k 步 67.2%，40k 步 60.4%，存在过拟合拐点。

**这些消融说明了什么**：T2A 不是简单 warm-up，而是承担"把语言到动作的解压缩映射学扎实"的关键阶段。数据配比、预测模式、视觉开关、时间步分布、训练步数每一个超参都有清晰的 sweet spot；论文把"$\sim$20% 合成 + 80% 真实 / full-sequence / 无图像 / Sigmoid-Normal → Beta / 2k 步"作为默认 recipe，每一步都有数字支持。

#### 5.3.2 多本体联合训练与投影设计（论文表 10）

| 设计 | Bridge（单平台） | Robocasa（单平台） | Multi-MLP | Concat. | Zero-Pad |
|---|---|---|---|---|---|
| Bridge | 62.8 | – | 63.3 | 63.0 | 63.0 |
| Robocasa | – | 53.4 | 52.1 | 52.8 | 53.2 |

**这说明了什么**：三种 projection 设计（Multi-MLP / Concat. / Zero-Pad）的最终性能差距 < 1.2 pp，说明"一旦建立共享潜空间，projection 选型对任务成功影响有限"。但 Zero-Padding 参数最少（$2h \cdot d_{\max}$ vs $2h \sum_i d_i$），是工程上最轻的选择，因此被选为默认方案。同时所有联合训练变体都追平了单平台 baseline，证明多本体联合训练**几乎没有性能代价**。

#### 5.3.3 视觉语言协同训练消融（论文图 7）

- **VLA-Only vs VL+VLA**：在 Libero / Simpler-WidowX 上两者几乎相等；在需要细粒度物体识别与组合指令理解的 RoboCasa-GR1 上 VL+VLA 提升 +4.9 pp（51.1% → 56.0%）；在 RoboTwin-2.0 上提升 +4.6 pp（81.8% → 86.4%）。**说明在操作训练中混入 VL 数据几乎没有副作用，并对细粒度任务有正向收益。**
- **预训练 DiT 的可迁移性**：把预训练 DiT 接到 fresh Qwen3.5-4B VLM 上，仍然比从零随机初始化 DiT 收敛更快、最终峰值更高，**证明 T2A 阶段学到的动作先验是 backbone-agnostic 的**。

#### 5.3.4 后训练各阶段累积效果（论文表 11）

| Stage | Simpler | RoboCasa | RoboTwin-E | RoboTwin-H | LIBERO | SimplerOOD | DOMINO SR | DOMINO MS |
|---|---|---|---|---|---|---|---|---|
| CPT | 64.3 | 40.4 | 64.3 | 66.4 | 90.8 | 25.3 | 21.1 | 37.4 |
| + SFT | 70.8 | 56.0 | 86.3 | 87.1 | 97.8 | 31.6 | 25.7 | 39.1 |
| + RL | 73.7 | 56.7 | 86.1 | 87.2 | 97.9 | 32.0 | 26.6 | 39.5 |

**这说明了什么**：CPT → SFT → RL 三阶段每一步都提供互补增益。SFT 在所有 benchmark 上带来 6-16 pp 的大幅提升（RoboCasa +15.6、RoboTwin-E +22.0），是"任务特化"的主力。RL 在 RL 训练分布所在的 SimplerEnv 上额外 +2.9 pp（70.8 → 73.7），且在训练分布之外的 RoboCasa、RoboTwin-H、LIBERO、SimplerOOD、DOMINO 上**没有灾难性遗忘**，甚至有小幅正向迁移，验证了在单一仿真环境里优化任务成功率可以外溢到其他设置。

#### 5.3.5 状态条件消融（论文表 12）

| Conditioning | RoboTwin-Easy | RoboTwin-Hard |
|---|---|---|
| No State | 88.7 | 87.4 |
| State in VLM Prompt | 89.3 | 88.7 |
| State in DiT | 89.4 | 88.3 |

**这说明了什么**：把本体感受态（关节角）注入到 VLM prompt 或 DiT，带来的提升最多只有 +0.7 pp / +1.3 pp，且两种注入方式差距 ≤ 0.4 pp。这与很多机器人学习方法"显式给 proprioception 很重要"的直觉相反。论文把这一结果归因于两点：(1) 多视角视觉已经包含足够多的本体配置信息；(2) flow-matching 预测的是相对动作位移，对当前状态的显式参考需求较弱。**因此 Qwen-VLA 默认不引入 proprioception，简化跨平台接口**。

### 5.4 泛化、效率与失败案例

#### 5.4.1 静态 OOD（SimplerEnv-OOD，论文表 8）

| 方法 | MoveAway | MoveRight | PlaceNear | PlaceRight | PutFront | StackYellow | Avg. |
|---|---|---|---|---|---|---|---|
| $\pi_{0.5}$ | 26.1 | 0.0 | 0.0 | 32.1 | 13.0 | 4.2 | 12.6 |
| Qwen-VLA-Base | 31.3 | 31.6 | 16.7 | 47.1 | 6.3 | 18.8 | 25.3 |
| **Qwen-VLA-Instruct** | **43.8** | **33.3** | **39.6** | **47.9** | 4.2 | **22.9** | **32.0** |

$\pi_{0.5}$ 在 MoveRight、PlaceNear 上完全失败（0%），Qwen-VLA-Instruct 则分别拿到 33.3% / 39.6%。在 StackYellow（训练只见过 green-on-yellow，测试反过来 yellow-on-green）这种颜色-对象反绑任务上，Qwen-VLA-Instruct 也以 22.9% vs 4.2% 完胜。**这说明在训练分布外，Qwen-VLA 的预训练先验比 $\pi_{0.5}$ 这类 specialist 更稳健**。

#### 5.4.2 动态 OOD（DOMINO，论文表 9）

Qwen-VLA-Instruct 零样本 SR 26.6% / MS 39.5，超过所有经过 dynamic fine-tuning 的 specialist（包括 PUMA 17.2/35.0、StarVLA-OFT 10.9/30.5、OpenVLA-OFT 9.1/24.1、$\pi_{0.5}$ 9.6/26.2），同时也超过零样本基线 LingBot-VA 24.1/36.1。**这尤其值得注意**：Qwen-VLA 训练数据里完全没有动态操作，却在动态基准上超过所有在动态数据上专门微调过的模型。论文把这一现象归因于两点：(1) flow-matching 给出"决定性"的动作块生成，犹豫更少、动作更准；(2) 大规模联合预训练（操作+导航+轨迹+VL）形成了可迁移的"空间到运动"先验。

#### 5.4.3 真实 OOD 失败案例

论文在 §5.1.6 中明确报告了几类失败：
- 对玩具鸭这种**不规则形状**的抓取，模型能正确定位和接近，但形状不规则导致稳定抓取失败；
- 对太阳镜这种**薄而扁平**的几何体，模型能正确定位和接近，但物理上无法稳定抓握（不是视觉识别或指令理解失败，而是物理形状泛化的边界）；
- Towel Folding 仍是真实任务中相对最弱的（65.4%，远低于 Pick&Place 的 96.2%），说明**可形变物体操纵**仍是当前 generalist VLA 的明显短板。

**这些失败案例的意义**（个人判断）：它们揭示了 Qwen-VLA 的能力边界在"视觉 grounding 与指令理解"上其实已经相当强，真正的瓶颈在于**接触物理、形状几何、可形变动力学**这些更深层的问题，这意味着 VLA 下一步的突破口可能不再是更大的 VLM 或更多数据，而是更丰富的物理反馈信号（力、触觉、proprioception）+ 更强的接触动力学建模。

---

## 6. 与相关工作的关系

Qwen-VLA 处在三条主线的交汇点：

- **VLM/VLA 路线**：与 $\pi_0$、$\pi_{0.5}$、GR00T N1.6、OpenVLA、OpenVLA-OFT、StarVLA-OFT、ABot-M0、Being-H0.5、InternVLA-M1、VLA-Adapter、$\pi_0$-FAST、LingBot-VA、PUMA、ReconVLA、PrimitiveVLA 同属"通用 VLM + 连续动作头"路线，但 Qwen-VLA 是其中把"跨任务族 + 跨本体 + 跨环境"统一到最完整的；VITRA 同样以 egocentric 人类数据为 VLA 预训练，但 Qwen-VLA 把它和其他 embodied 数据源、合成数据、导航数据都混在了一起。
- **导航路线**：与 NaVid、Uni-NaVid、NaVILA、StreamVLN 同属 VLN-CE 路线，但 Qwen-VLA 把导航以 waypoint 形式塞进统一动作张量，没有专门的导航 head；这是它在 VLN-CE 仍然能追平或超过 specialist 的关键。
- **世界模型路线**：与 DreamZero、Fast-WAM、DreamDojo 同属"从基础模型角度构建 embodied agent"的工作，但 Qwen-VLA 明确不做视频预测；它更强调"语言 grounded 的 VLA 作为可执行接口"，而世界模型更强调"未来可视化以辅助规划"。论文在 §6 结论中也明确把 Qwen-VLA 与视觉预测中心的世界模型做了对比，认为两者方向互补。

（个人判断）从 lineage 角度看，Qwen-VLA 与 $\pi_0$ 系列最接近（都用 DiT-style flow matching 做连续动作），但其工程重心在"统一多任务 + 大规模预训练 + 训练 recipe 的可解释性"，而 $\pi_0$ 系列更强调"单一机器人平台 + 单一任务族的高效执行"。两者一起代表了"generalist VLA"这条线在 2026 年的两种典型走法。

---

## 7. 局限与批判性评价

论文自己在 §7 Limitations and Future Work 中承认了三类局限：

1. **具身数据规模与多样性远不如通用 VL 预训练数据**，长尾物体、环境、本体、接触密集交互的鲁棒性仍有上限；
2. **联合训练 VL + navigation + action 存在优化折中**：动作导向训练能提升 policy 表现，但会轻度回退一部分纯 VL 与导航评测，需要更好的目标平衡、数据课程和模块化特化；
3. **现有评估仍以短视界 benchmark 为主**，长程、易失败的真实部署仍是开放问题。

**作者提出的未来方向**：扩展真实交互数据（自主采集 + 仿真 + sim-to-real）、大规模人类视频（第一/第三人称）、长程规划 + episodic memory + 世界模型、引入力/触觉/proprioception 等物理反馈、sim2real + 真实世界大规模 RL。

**（个人判断）补充几个论文未充分讨论但值得关注的局限**：

- **"统一接口 ≠ 统一语义"**：当前方案更像统一了训练与输入接口，而不是真的学到了完全跨本体的共享动作语义；论文的跨本体实验主要还是在同平台内微调后做 OOD，缺乏"在 A 上预训练、直接在 B 上零样本"的严格检验；
- **数据配比偏经验工程**：表 1 给出的 74.2/7.5/6.0/3.7/3.4/2.5/2.4/0.2 比例以及 §5.2.1 中"$\sim$20% 合成 + 80% 真实"的最优点，都是经验调出来的，没有严格的消融解释每个比例的贡献；
- **RL 仅在单一仿真环境**进行，奖励为稀疏二元，成功外溢到其他 benchmark 是亮点，但也意味着 RL 阶段的"决策能力"主要是从模仿 + 单环境 RL 学来的，未必能直接 scale 到长程复杂任务；
- **proprioception 的处理**：论文选择不用显式本体感受态（§5.2.4），这虽然简化了跨平台接口，但同时放弃了一条可被未来工作利用的物理信号；力/触觉信号目前在 Qwen-VLA 里完全没有进入；
- **T2A 阶段"无图像"的设计**虽然带来 +10.2 pp 收益，但代价是 T2A 阶段只能"想象动作"——这种纯文本到动作的预训练是否在更大规模时仍能保持收益，还是会很快饱和，仍是开放问题；
- **论文未充分披露的关键工程细节**：DiT 推理 Euler 步数、actor 与 critic 的精确学习率、训练总 token 量 / 训练步数、训练硬件（论文未披露），这些都会显著影响复现成本评估。

---

## 8. 复现与实践建议

**复现成本与门槛**（论文未披露关键训练资源数据，结合常识推测）：

- **骨干 Qwen3.5-4B + DiT 1.15B ≈ 5B+ 参数**的全量微调，至少需要数十张 H100/A100 级别 GPU、数周到数月的训练时间；
- 预训练数据混合了 10,000+ 小时公开机器人数据 + 1,000+ 小时自采真机 + 8M+ 合成仿真 + 大规模 VL 数据，**数据获取与组织成本比模型本身更重**；
- 真实 ALOHA 部署需要双 6-DoF 臂 + 平行夹爪 + 3 个 RGB 相机 + 遥操作系统；
- RL 阶段需要 128 个并行仿真环境实例 + 解耦客户端-服务器架构；
- 代码仓库已开放在 https://github.com/QwenLM/Qwen-VLA，是降低复现门槛的关键资源。

**（个人判断）实践建议**：

- 如果目标是**多平台机器人研究**，优先复现 embodiment-aware prompt + 统一 $H\times K$ 张量 + 通道掩码这一组设计，因为它们是"统一多本体"的工程核心，独立于具体 backbone；
- 如果想做**新任务的 VLA 微调**，建议在 Qwen-VLA-Base 上做 CPT+ SFT，并先在自己任务上跑一组 T2A 数据配比消融（论文 §5.2.1 的曲线非常清晰可参考）；
- 如果想做**导航 + 操作联合**，直接复用 waypoint-as-action 的设计是性价比最高的入口；
- 如果想做**纯研究意义的小规模复现**，可以从"Qwen3.5-4B + 一个开源机器人数据子集 + 文本-only T2A 阶段 + 无 RL"开始，逐步验证 embodiment prompt、零填充、flow-matching loss 三个核心组件。

---

## 9. 个人启发与后续问题

**（个人判断）核心启发**：

1. **"统一"的真正位置在损失和张量，而不是在网络架构**。Qwen-VLA 没有发明新 backbone，也没有发明新 attention，只是用 embodiment prompt + 零填充 + 通道掩码把"异构动作空间"对齐到同一个 $H\times K$ 张量 + 同一个 loss 上。这种"工程统一"在多本体训练里的可复用性比任何"统一架构"都高；
2. **T2A 这种"无图像的语言到动作解压缩"是值得抄的预训练范式**。它本质上是说"动作生成里有很大一部分是纯语言结构决定的，与视觉无关"，把这部分先学扎实再去做视觉 grounding，比"端到端一锅炖"更稳也更省；
3. **DiT + flow-matching 已经基本成为 VLA 动作端的默认范式**。无论是 Qwen-VLA、$\pi_0$ 系、GR00T 还是 PUMA，都选择了 DiT-style flow matching；离散 token 自回归 head 在 VLA 里的位置正在被快速替换；
4. **"统一动作张量 + 通道掩码"是一个被低估的小技巧**。它让 7-DoF 末端位姿和 29-DoF 全身控制能在同一个 loss 下公平训练，又不强迫语义统一——这在多平台训练里是性价比极高的设计选择；
5. **预训练对真实 OOD 的收益，远比"在 benchmark 上刷分"重要**。Qwen-VLA 在 ALOHA 上的 w/o pretrain vs w/ pretrain 对比（48.5 → 83.6 域内 / 36.2 → 76.9 OOD）可能是整篇工作最有说服力的数据点。

**后续值得跟踪的问题**：

1. **真正的 cross-embodiment transfer**：在 A 上预训练、零样本到 B（完全不微调）能否 work？这是当前 Qwen-VLA 还没严格回答的问题；
2. **长程 + 失败恢复**：Qwen-VLA 的所有评测都是短视界任务；长程任务 + 失败恢复 + episodic memory 仍是开放方向；
3. **物理反馈信号的接入**：力、触觉、proprioception 在 Qwen-VLA 里要么缺失要么被消融为"不用"，未来是否能与 Qwen-VLA 协同增益；
4. **RL 阶段的可扩展性**：当前 RL 只在单一仿真环境、稀疏二元奖励；如何 scale 到多环境 + 课程奖励 + 真实世界 RL；
5. **T2A 的规模上限**：2k 步最佳、40k 步过拟合，是否在更大数据 / 更强骨干下仍成立？如果 T2A 的 sweet spot 很短，说明"语言到动作"的结构非常低维、几乎是一个可压缩的有限任务；
6. **与 ReconVLA、PrimitiveVLA、DreamZero、Fast-WAM 等同期工作的关系**：ReconVLA 用目标区域重建替代 grounding、PrimitiveVLA 用动作原语、DreamZero/ Fast-WAM 用世界模型——这些是 Qwen-VLA 范式的"互补变体"还是"替代品"，值得做一次统一视角对比。

---

## 10. 材料来源

### 本地附件

- `sources/paper.pdf`：Qwen-VLA 论文 PDF（arXiv 2605.30280v2，34 页）
- `sources/paper.txt`：pdftotext 提取的论文全文文本（用于精读检索）

### 论文图片（保留原有相对路径引用）

- `qwenvla_overview.png` = 论文 Figure 1：Qwen-VLA 总体架构概览
- `qwenvla_training_recipe.png` = 论文 Figure 2：四阶段训练配方（T2A → CPT → SFT → RL）
- `qwenvla_sim_demo.png` = 论文 Figure 3：RoboInF 合成仿真数据示例（含子任务分解）
- `qwenvla_task_overview.png` = 论文 Figure 4：ALOHA 双臂平台真实评估任务总览
- `qwenvla_realgrid.jpg` = 论文 Figure 5：ALOHA 4×4 OOD 定性泛化网格

### 外部来源

| 资源 | URL | 类型 | 用途 |
|---|---|---|---|
| arXiv 论文页 | https://arxiv.org/abs/2605.30280 | 论文主页 | 精读主来源 |
| arXiv HTML 版 | https://arxiv.org/html/2605.30280v2 | 论文 HTML | 精读主来源、图表下载 |
| ar5iv 镜像 | https://ar5iv.labs.arxiv.org/html/2605.30280 | 论文 HTML 镜像 | 备选精读来源 |
| Qwen 项目页 | https://qwen.ai/blog?id=qwenvla | 官方项目页 | 方法说明、补充材料 |
| 代码仓库 | https://github.com/QwenLM/Qwen-VLA | 官方代码 | 复现与实践 |
| 微信公众号原文 | https://mp.weixin.qq.com/s/ZiY7YwQOYAoz56dcKt3W0w | 二手解读 | 背景补充，不替代论文 |
