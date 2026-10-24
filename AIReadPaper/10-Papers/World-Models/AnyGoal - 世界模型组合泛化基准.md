---
type: paper-note
status: draft
domain: World-Models
paper: "AnyGoal: On World Model Generalization Beyond Expert Rollouts"
year: 2026
arxiv: 2610.26844
doi: 10.48550/arXiv.2610.26844
source: https://arxiv.org/abs/2610.26844
project: https://hazel-heejeong-nam.github.io/anygoal/
code: https://github.com/galilai-group/AnyGoal
tags:
  - World-Models
  - compositional-generalization
  - benchmark
  - goal-shift
  - planning
created: 2026-10-24
updated: 2026-10-24
---

# AnyGoal：世界模型在专家轨迹之外的组合泛化

## 基本信息

- **论文**：AnyGoal: On World Model Generalization Beyond Expert Rollouts
- **作者**：Heejeong Nam, Divyat Mahajan, Quentin Le Lidec, Randall Balestriero, David Klindt
- **机构**：Brown University, Cold Spring Harbor Laboratory, Mila (Université de Montréal), New York University
- **发表**：arXiv 2610.26844v1, 2026-10-19
- **项目页**：[https://hazel-heejeong-nam.github.io/anygoal/](https://hazel-heejeong-nam.github.io/anygoal/)
- **代码**：[https://github.com/galilai-group/AnyGoal](https://github.com/galilai-group/AnyGoal)

## 一句话结论

AnyGoal 通过「组合泛化」视角系统评估世界模型在目标偏移（goal-shift）下的泛化能力：即使单个状态因子和动作都在训练支持内，它们的新组合也会导致当前世界模型广泛失败，尤其在接触丰富任务中；失败源于动力学预测和表示学习的双重瓶颈，而理论分析指出局部机制覆盖（而非全局覆盖）才是组合泛化的关键条件。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

世界模型的核心目标是学习环境动力学，使智能体能够借助模型规划和模拟行动后果。然而，一个可靠的世界模型不应仅记忆训练分布中的行为偏差——它需要学习**可重用的动力学**（reusable dynamics），在超出专家轨迹观察到的状态-动作组合时仍能正确预测。

本文的核心问题是：**当前世界模型能否将熟悉的状态因子和动作重新组合成有效的动力学预测？** 换言之，当测试时遇到训练分布中未出现的「状态-动作组合」，但这些组合的每个单独因子都在训练支持内时，模型是否能正确泛化？

### 1.2 现有方法的瓶颈

现有世界模型评估存在几个关键缺陷：

1. **分布内评估的局限性**：在训练分布内的高预测精度并不能证明模型学到了可重用的动力学。模型可能仅学习了行为策略诱导的状态-动作相关性（即「行为偏差」），而非动作的真实因果效应。
2. **数据覆盖的现实困难**：离线数据集通常由目标导向策略收集，引入强烈的状态-动作相关性；即使混合多策略数据，也难以确认覆盖是否充分；在离线设置中针对性地收集缺口区域的轨迹也很困难。
3. **OOD 评估的泛化范围不当**：现有 OOD 评估往往引入训练支持之外的状态或因子，但这可能超出了可重用动力学模型应支持的泛化范围——要求模型处理全新物理因子是不合理的。

![](assets/AnyGoal%20-%20世界模型组合泛化基准/figures/01-teaser-goal-shift-failure.png)

> 图 1：目标偏移下的泛化失败示例。在 Push-T 任务中，专家轨迹将「远离方块的状态」与「朝向方块移动的动作」强耦合；模型可能学习了这种轨迹规律性而非动作的真实效应。当在 Avoid-T（需要避开方块）上评估时，模型失败。  
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2610.26844

### 1.3 本文核心贡献

1. **形式化**组合泛化标准，将其与目标偏移和训练数据中的行为偏差联系起来；
2. **证明**偏差通过针对熟悉但覆盖不足的状态-动作组合的定向动作干预经验性地实现，引发退化的模型响应；
3. **提出 AnyGoal**：保持已观察到的状态、动作、任务相关因子和物理规则的边际支持，同时改变由训练行为分布诱导的联合组合；
4. **发现**跨建模范式（包括生成式世界模型和 JEPA 风格架构）的广泛失败；
5. **刻画**退化跨多个维度：失败可能源于动力学预测和表示学习，严重程度取决于任务复杂性、评估指标和模型家族；
6. **概述理论方向**，朝向能够超越行为偏差轨迹的组合泛化世界模型。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

**符号体系**：设 $z_t = (z_{1,t}, \ldots, z_{K,t})$ 为环境的真实底层状态，每个组件 $z_{k,t}$ 对应一个语义变化因子（如物体位置或末端执行器姿态）。观测建模为 $s_t = f(z_t)$，其中 $f$ 将底层状态映射为 RGB 观测。动作配对轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots, s_H)$ 可等价地通过物理因子和动作序列 $(z_0, a_0, z_1, a_1, \ldots, z_H)$ 表示。

**核心假设**：训练和测试分布共享相同的环境动力学 $T$，但由不同的目标 $g_{tr}$ 和 $g_{te}$ 诱导的行为策略 $\pi_{tr}$ 和 $\pi_{te}$ 生成数据。关键条件是测试支持引入了训练支持中不存在的状态-动作对（$\mathcal{C}^{te} \not\subset \mathcal{C}^{tr}$），但每个单独因子仍在其边际训练支持内。

### 2.2 关键符号和目标函数

**训练联合状态-动作支持**：

$$\mathcal{C}^{tr} := \bigcup_{t=0}^{H-1} \text{supp}_{p_{tr}}(Z_{1,t}, \ldots, Z_{K,t}, A_t)$$

**组合空间**（边际训练支持的笛卡尔积）：

$$\mathcal{C}^{\times} := \left(\prod_{k=1}^{K} \mathcal{Z}_k^{tr}\right) \times \mathcal{A}^{tr}$$

**组合缺口**（combinatorial gap）：组合空间 $\mathcal{C}^{\times}$ 中不在训练联合支持 $\mathcal{C}^{tr}$ 中的区域。AnyGoal 要求 $\mathcal{C}^{te} \subseteq \mathcal{C}^{\times}$，且选择具有较大 under-covered 区域质量的 goals。

**动作条件预测退化**：在离线设置中，数据由行为策略 $\pi$ 收集，观察到的转移反映了 $\pi$ 诱导的状态-动作配对：

$$\hat{T}^{\pi}(s_{t+1}|s_t) = \int T(s_{t+1}|s_t, a)\pi(a|s_t)da$$

在此类数据上训练的预测器可能退化为**动作无知模型**：$\hat{T}(s_{t+1}|s_t, a_t) \approx \hat{T}^{\pi}(s_{t+1}|s_t)$，即对 supplied action $a_t$ 的依赖实际上丢失了。

---

## 3. 核心方法

### 3.1 总体框架

AnyGoal 基于四个广泛使用的世界模型任务构建，产生**八个 goal-shifted 变体**：

| 基础任务 | 变体 | 类型 |
|---------|------|------|
| TwoRoom | TR-Route | 自由空间 |
| Push-T | Avoid-T | 自由空间 |
| Push-T | Push-T-Hard | 接触丰富 |
| Insert-Peg | Avoid-Peg | 自由空间 |
| Insert-Peg | Insert-Peg-Hard | 接触丰富 |
| OGBench Cube | Cube-Hop | 自由空间 |
| OGBench Cube | Cube-Stack | 接触丰富 |
| OGBench Cube | Cube-Flip | 接触丰富 |

![](assets/AnyGoal%20-%20世界模型组合泛化基准/figures/02-overview-anygoal.png)

> 图 2：AnyGoal 总体概览。每个任务保持对应基础环境的环境动力学，同时通过目标偏移改变行为策略诱导的状态-动作联合分布。  
> 来源：论文 Figure 2，第 2 页，https://arxiv.org/abs/2610.26844

每个变体发布 **400 episodes**，并提供数据收集工具。核心设计原则是：单个状态因子和动作都在训练支持内，但它们的新组合在专家轨迹中代表性不足。

### 3.2 因子级支持验证

为确保测试样本的每个因子都在训练支持内，AnyGoal 对每个基础-目标任务对应用三种因子级支持检查：

1. **范围检查（Range check）**：所有查询样本必须在坐标级参考范围内；
2. **Mahalanobis 检查**：使用参考均值和协方差，在参考 0.99 分位数处阈值距离，要求至少 95% 的查询样本低于该值；
3. **KDE 检查**：在 0.01 分位数处阈值参考密度，要求至少 95% 的查询样本高于该值。

最终支持判定是所有三个检查的合取。所有 AnyGoal 变体设计为通过这些因子级检查。

### 3.3 组合缺口覆盖的度量

为量化每个目标分布与基础任务联合支持的重叠程度，作者对每个基础-目标对连接动作、智能体状态和任务对象或目标状态，拟合全维高斯 KDE 到参考向量，设置密度阈值为 log 密度的经验 0.01 分位数。**较低的联合分数表示与基础任务联合分布重叠较少，组合缺口覆盖更大**。

![](assets/AnyGoal%20-%20世界模型组合泛化基准/figures/04-combinatorial-gap.png)

> 图 4：AnyGoal 针对组合缺口。(a) 因子级支持检查示例，确认每个单独因子在训练支持内；(b) 联合分布检查确认组合缺口存在。  
> 来源：论文 Figure 4，第 5 页，https://arxiv.org/abs/2610.26844

### 3.4 评估协议

**相对评估**：模型仅在基础数据集上训练，在基础和 goal-shifted 轨迹上评估以测量相对变化。

**预测误差（Prediction Error）**：
- 在每个模型的原生预测空间中测量 MSE（学习的潜在空间、oracle 状态空间或像素空间）
- 评估自回归预测，horizon 为 5 和 25 环境步，frame skip 为 5
- 每个 seed 采样 100 episodes，在 episode 内所有有效起始点上平均

**MPC 规划成功率**：
- 使用 Stable-WorldModel 中相同的基于 CEM 的 MPC 实现
- 规划 horizon 为 25 环境步
- 完整 25 步动作序列在执行前重规划

![](assets/AnyGoal%20-%20世界模型组合泛化基准/figures/05-evaluation-illustration.png)

> 图 5：同一 base-trained 模型在 Push-T（训练目标）和 Avoid-T（goal-shifted 目标）上的标准化评估流程。  
> 来源：论文 Figure 5，第 6 页，https://arxiv.org/abs/2610.26844

### 3.5 测试的模型家族

论文评估了两类模型：

**端到端世界模型**：
| 模型 | 表示 | 动作条件方式 |
|------|------|-------------|
| LeWorldModel (LeWM) | JEPA | 学习潜在 + AdaLN |
| Sensorimotor WM (SMWM) | JEPA | 学习潜在 + AdaLN |
| DINO-WM | JEPA | 冻结 DINO 特征 + 特征连接 |
| DriftWorld | 生成式 | 像素 + 条件生成器 |
| IRASim | 生成式 | 冻结 SDXL VAE + DiT 条件 |

**Oracle 状态预测器**（直接使用真实物理状态）：
- AdaLN Transformer
- Action-Concatenated Transformer
- Toy-DriftWorld

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

AnyGoal 基于四个成熟的控制环境构建：
- **TwoRoom**：二维导航任务，智能体在两个房间间移动
- **Push-T**：智能体将 T 形方块推到目标位置
- **DMC Manipulator Insert-Peg**：机械臂将销钉插入孔洞
- **OGBench Cube**：机械臂操控立方体

每个基础任务有对应的 goal-shifted 变体，通过改变目标位置或任务约束来诱导新的状态-动作组合。

### 4.2 Baseline 与评价指标

主要指标为**相对变化**：将 goal-shifted 任务上的性能与基础任务上的性能对比。这避免了不同模型间绝对性能的直接比较，聚焦于「同一模型在组合偏移下的退化程度」。

### 4.3 实现细节

- 规划使用 CEM-based MPC，horizon 25 步
- 预测评估使用自回归 rollout，horizon 5/25 步
- 所有模型使用各自原生预测空间计算 MSE
- 动作敏感性度量：对每个训练状态 $s$，用来自固定干预集 $\mathcal{A}_{cf}$ 的动作替换原动作 $a$，测量模型表示变化：

$$S_\theta(s) = \frac{1}{|\mathcal{A}_{cf}|} \sum_{a' \in \mathcal{A}_{cf}} \|\phi_\theta(s,a) - \phi_\theta(s,a')\|_2$$

---

## 5. 实验结果

### 5.1 主要发现：目标偏移下的广泛失败

跨所有模型家族和任务，核心发现是：**当前世界模型在组合泛化上广泛失败**。即使单个因子和动作都在训练支持内，它们的新组合会导致显著的性能下降。

### 5.2 Oracle 状态预测器揭示交互依赖的动力学缺口

移除视觉抽象后，结果在不同 AnyGoal 变体间不均匀：

- **自由空间变体**（TR-Route, Avoid-T, Avoid-Peg, Cube-Hop）：MPC 性能保持或相对于基础任务有所改善
- **接触丰富变体**：始终退化，尽管使用相同的特权状态输入

![](assets/AnyGoal%20-%20世界模型组合泛化基准/figures/03-oracle-state-dynamics-failure.png)

> 图 6：Oracle 状态预测器揭示动力学级别的失败。即使提供任务相关的特权状态输入，接触丰富变体仍然退化，表明困难不仅在于状态访问，而在于目标偏移是否需要精确的物体交互。  
> 来源：论文 Figure 6，第 7 页，https://arxiv.org/abs/2610.26844

这一不对称性表明：困难不仅在于访问相关状态因子，而在于目标偏移是否需要精确的物体交互。加法或无交互设置特别有利，因为因子特定机制可以从观察到的组合中学习并在新重组下重用。接触丰富变体偏离了这个有利机制：动作效应可能不再分解为可重用的因子特定机制。

### 5.3 端到端世界模型中的表示效应

**因子重组可诱导观测级偏移**：端到端模型观察渲染输入 $s = f(z)$，重组边际熟悉的因子可能创建超出专家轨迹联合视觉支持的观测，添加超越动力学预测的编码器推理挑战。

**潜在几何塑造目标偏移下的退化**：TwoRoom 提供了一个有用的案例——因为智能体位置是唯一变化的因子，TR-Route 位置在边际上仍在 TwoRoom 训练支持内，相应观测保持在视觉支持内。即使在无明显视觉偏移的情况下，端到端模型退化而 oracle 状态预测器保持稳定。这表明**表示质量仍然重要**：学习的潜在空间可能无法精确保存智能体位置，引入不必要的维度或干扰变化，或以不利于预测和规划的几何方式组织状态。

### 5.4 通过动作敏感性诊断行为策略捷径

跨所有环境（TwoRoom、Push-T、Insert-Peg、Cube），观察到在与目标强烈约束训练行为的区域中对干预动作的敏感性降低。这些观察与理论分析一致：在行为偏差强的区域，模型退化为近似动作无知的预测器。

### 5.5 指标对齐取决于任务复杂性

rollout 误差和规划性能有时在自由空间变体中发散，因为规划可以容忍不完美的预测。接触丰富设置中更一致的一致性可能反映规划问题的结构：rollout 误差衡量模型是否准确预测完整轨迹，而 MPC 只需要找到某个到达目标的动作序列。自由空间任务通常允许多条这样的轨迹，而接触丰富任务允许更少可行序列。

---

## 6. 与相关工作的关系

**组合泛化诊断**：早期序列到序列基准如 SCAN 和 CFQ 通过受控组合分割暴露失败模式。随后在视觉和多模态设置中采用（CLEVR、Winoground、CVR）。ConceptMix 表明文本到图像模型随组合复杂度退化。AnyGoal 将这一视角带入世界模型领域。

**世界模型泛化**：最近的工作通过对象为中心的组合动力学、可组合因果组件、多任务世界模型等研究泛化。AnyGoal 与这些方法互补：它评估世界模型是否在目标偏移下学习可重用动力学，与离线 RL 中的支持不匹配问题密切相关。

**与 Hansen and Wang (2026) 的区别**：后者研究世界模型中的幻觉检测和缓解，而 AnyGoal 在统计受控空间中评估泛化，具有明确的状态-动作支持检查。

---

## 7. 局限与批判性评价

### 7.1 方法局限

1. **任务规模和复杂度有限**：四个基础任务均为经典控制环境，与真实世界的开放域场景差距较大。在更复杂的场景中，因子分解本身可能就不明确。
2. **接触丰富的建模挑战**：论文识别出接触丰富任务是主要失败来源，但未提出解决这一问题的具体架构或训练方案——仅停留在理论讨论层面。
3. **因子级支持检查的充分性**：三种统计检查（Range/Mahalanobis/KDE）虽合理，但高维空间中分布支持的验证本身是困难问题，可能存在检查通过但实际仍有分布偏移的情况。

### 7.2 实验局限

- 测试的模型虽然覆盖了 JEPA 和生成式两大范式，但缺少最新的规模化世界模型（如 Cosmos 系列、Genie 系列等）
- 规划评估仅使用 CEM-based MPC，未探索其他规划方法（如 MPPI、梯度优化）是否表现不同
- 每个变体仅 400 episodes，对某些结论的统计功效可能不足

### 7.3 个人判断

这是一篇**方法论上非常扎实**的 benchmark 论文。其核心洞察——「组合泛化」作为世界模型评估的视角——是有价值的，因为它隔离了一个具体的失败模式（行为偏差 vs 可重用动力学），而非笼统的 OOD 泛化。理论分析（附录 A）对「零训练误差不意味着动作条件动力学」的论证清晰有力。

不足之处在于：论文更像是诊断而非治疗——它清楚地展示了问题，但未提供实质性的解决方案。理论部分（附录 J）关于局部机制覆盖的条件是有启发性的，但距离实际指导模型设计还有较大距离。

---

## 8. 复现与实践建议

- **代码可用性**：开源 Python 库，提供 CLI 和 Python API
- **复现成本**：中等——需要训练多个世界模型（LeWM、SMWM、DINO-WM、DriftWorld、IRASim）和 oracle 预测器，但模型规模不大
- **数据需求**：每个变体 400 episodes，数据收集工具已提供
- **推理速度**：评估阶段使用 CEM-MPC，horizon 25 步，计算量适中
- **使用建议**：CLI 接口简洁（`anygoal eval planning --model lewm --dataset pusht --seed 0`），适合快速验证新模型

---

## 9. 个人启发与后续问题

1. **与 JEPA 路线的关系**：论文发现 JEPA 风格模型（LeWM、SMWM、DINO-WM）和生成式模型（DriftWorld、IRASim）在组合泛化上都有失败，说明这不是特定架构的问题，而是当前训练范式（行为策略数据 + 最大似然）的固有限制。
2. **接触丰富的本质困难**：这提示我们，世界模型在操控任务中的应用可能需要超越纯数据驱动的方式——注入物理约束、接触建模或因果结构可能是必要的。
3. **局部机制覆盖的实用化**：理论部分指出，如果每个 next-state 因子的转移仅依赖于状态-动作坐标的小子集（$\|\nabla_x T_{k,r_t}(x)\|_0 \ll \dim(x)$），则组合泛化是可能的。这为模块化/稀疏化世界模型设计提供了理论动机。
4. **与 3D JEPA / Asset-JEPA 的关联**：对于三维资产领域，如果世界模型需要理解 mesh 拓扑、材质等因子的组合效应（如不同拓扑 + 不同材质的物理交互），AnyGoal 揭示的组合泛化问题同样适用。

---

## 10. 材料来源

| 材料 | 来源 |
|------|------|
| 论文 PDF | [arXiv 2610.26844](https://arxiv.org/abs/2610.26844) |
| 项目页 | [https://hazel-heejeong-nam.github.io/anygoal/](https://hazel-heejeong-nam.github.io/anygoal/) |
| 代码 | [https://github.com/galilai-group/AnyGoal](https://github.com/galilai-group/AnyGoal) |
| 图片 | 从官方 PDF 裁剪，详见 `sources.md` |
