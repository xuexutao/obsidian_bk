---
type: survey
status: done
domain: VLA
title: "StarVLA 系列：从统一基础设施、受控强基线到表示中心继续预训练"
year: 2026
arxiv: null
doi: null
source: null
project: https://starvla.github.io
code: https://github.com/starVLA/starVLA
tags:
  - VLA
  - survey
  - infrastructure
  - strong-baseline
  - representation-learning
created: 2026-10-21
updated: 2026-10-21
---

# StarVLA 系列：从统一基础设施、受控强基线到表示中心继续预训练

## 基本信息

| 时间 | 工作 | arXiv | 研究对象 |
|---|---|---|---|
| 2026-04 | [[StarVLA - 乐高式VLA开源开发框架]] | 2604.05014 | 开源基础设施、统一接口、动作头框架、训练/评测/部署 |
| 2026-04 | [[StarVLA-alpha - 降低VLA系统复杂度]] | 2604.11757 | 受控简洁强基线、VLA 设计轴实验、generalist evaluation |
| 2026-08 | [[VLAct - 表示中心的VLA继续预训练]] | 2608.27550 | VLA-oriented VLM backbone、表示中心 continued pre-training |

> [!important] 先澄清命名
> 这三篇工作**不是同一个模型的三个版本，也不是 v1→alpha→VLAct 的 checkpoint 迭代**。它们共享 StarVLA 社区、代码生态、Qwen3-VL 与若干动作头组件，作者也高度重叠，但研究对象分别是：**基础设施、受控 baseline、预训练表示**。把三者写成“同一模型不断升级”会同时误读它们的输入输出、训练目标和评测归因。

**一句话综述**

这条路线的真正演化不是“模型越来越大”，而是研究问题逐层上移：StarVLA 先把骨干、动作头、数据和 benchmark 从各自孤岛中拆出来，使比较成为可能；StarVLA-$\alpha$ 再借此固定强 VLM 与简单 MLP，发现复杂连续头、异域动作预训练和重数据工程的收益常被高估；VLAct 最后接住 $\alpha$ 暴露的矛盾——朴素预训练可域内提升却跨本体负迁移——把目标从拟合一个 policy 改为学习可被新 head、新任务和新本体复用的 backbone 表示。

---

## 1. 领域要解决的根本问题

### 1.1 表面问题：从图像与语言生成机器人动作

三篇工作共享最外层任务：给定一个或多个视觉观测、自然语言指令和可选状态，预测未来 action chunk。若只看这个外层 I/O，它们都属于 VLA；但它们分别处理这个任务的不同瓶颈。

- StarVLA 问：不同 VLA 为什么无法公平比较、复用和部署？
- StarVLA-$\alpha$ 问：在公平比较后，哪些复杂设计真的带来稳定收益？
- VLAct 问：即便数据和训练都统一，继续预训练为何仍可能得到只适配某个 head/本体的表示？

### 1.2 深层问题一：变量纠缠

一个 VLA 系统的数字同时受 VLM、视觉塔、动作头、数据量、action coordinate、normalization、chunk、优化器、adapter 和 evaluator 影响。跨论文比较把这些变量一起变化，结果无法归因。StarVLA 首先提供统一坐标系；$\alpha$ 才能在同一坐标系里做 controlled experiment。

### 1.3 深层问题二：强下游结果不等于通用表示

OFT pre-training 后再用 OFT 微调，在 RoboTwin 可从 61.7 提到 75.8；若只看这一数字，预训练很成功。但把同一 backbone 换成 GR00T head，结果从 scratch 的 51.2 降到 28.9。VLAct 由此指出：matched-head policy performance 会高估 backbone reusability。表示可能携带动作信息，却以另一个 decoder 读不出的几何组织。

### 1.4 三条工作路线的结构关系

| 层次 | 核心问题 | 主要手段 | 成功标准 |
|---|---|---|---|
| 基础设施层 | 代码、接口、数据与 benchmark 碎片化 | 统一 raw I/O、骨干—head 契约、训练器、server-client | 同一环境下可替换组件并复现强 baseline |
| 实验方法层 | 哪些 VLA 复杂度真的有效 | 固定 Qwen3-VL + MLP，逐轴消融 head/pretraining/engineering | 收益能归因，简洁 baseline 足够强 |
| 表示学习层 | 预训练 backbone 是否跨 head/本体可复用 | shallow protection、caption、多连续 head、部分统一动作空间 | 新 head、新任务、新本体下仍提高下游性能 |

---

## 2. 第一次问题转移：从方法碎片化到统一实验平台

### 2.1 StarVLA 解决了什么

StarVLA 将政策写成：

$$\pi(\mathbf{a}_{t:t+k},\mathbf{y}_{\mathrm{aux}}\mid\mathbf{x}_{\le t},\ell)$$

并把目标写成：

$$\mathcal{L}=\mathcal{L}_{\mathrm{action}}+\mathcal{L}_{\mathrm{aux}}$$

直接 VLA 令辅助项为零；VLM-VLA 可预测语言规划/grounding；world-model VLA 可预测未来视觉。真正统一的是系统契约，而非宣称这些概率模型完全相同。

外部接口固定为原始 images/language/optional state 输入与 normalized action chunk 输出；内部以 VL backbone hidden states 连接可插拔 action head。FAST、OFT、$\pi$、GR00T 四种头因此可共享训练、benchmark 和部署栈，Qwen-VL 与 Cosmos 等骨干也能在相同外部协议中替换。

![](../../10-Papers/VLA/assets/StarVLA%20-%20乐高式VLA开源开发框架/figures/03-training-dataflow.png)

> StarVLA 的代表图：统一数据流与模型 forwarding。它的贡献点是把异构数据、骨干、动作头、训练器和 policy server 组织成稳定边界，而不是提出唯一的 StarVLA policy。来源：StarVLA Figure 3。

### 2.2 结果证明了什么

在单 benchmark 最小工程设置下，StarVLA-OFT 的 LIBERO 平均 96.6，StarVLA-GR00T 的 SimplerEnv WidowX 65.3，StarVLA-OFT 的 RoboCasa-GR1 48.8，三类连续头在 RoboTwin clean/random 约 88%。generalist 单模型在 RoboCasa 达 57.3，高于自身最佳 specialist 53.8。

这些结果证明平台不是“只能跑通”的 demo，而能产出强 baseline。8→256 GPU 弱扩展时吞吐 87→2200 samples/s，32 GPU 后约 79%–80% scaling efficiency，说明训练栈具备多节点可用性。

### 2.3 新瓶颈如何暴露

一旦四类动作头被放进同一平台，新的问题出现：复杂头真的必要吗？更多机器人预训练为何有时不增反降？benchmark adapter 被统一后，剩余收益到底来自 backbone、head 还是数据？StarVLA 本身提供能力展示，但没有做完整全因子受控实验，这直接引出 StarVLA-$\alpha$。

---

## 3. 第二次问题转移：从“能比较”到“哪些复杂度值得保留”

### 3.1 StarVLA-$\alpha$ 的受控设计

$\alpha$ 固定 Qwen3-VL-4B、原始 RGB + 指令、训练 split action normalization 和轻量 MLP 连续动作头，不默认使用 proprioception、历史帧、额外机器人预训练或复杂动作坐标。它并非宣称这些组件先验无用，而是建立一个足够强的零复杂度起点，让每个新增设计都必须证明边际价值。

![](../../10-Papers/VLA/assets/StarVLA-alpha%20-%20降低VLA系统复杂度/figures/02-method.png)

> StarVLA-$\alpha$ 的代表图：把传统 VLA 的视觉塔/LLM/head 拼装和 benchmark-specific 模型，压缩为 Qwen + MLP + 统一 generalist pipeline。来源：StarVLA-$\alpha$ Figure 2。

### 3.2 设计轴一：动作头

同一 backbone 和训练下：

| Head | LIBERO | WidowX | Google VM | RoboTwin random* | RoboCasa |
|---|---:|---:|---:|---:|---:|
| MLP/OFT | **98.8** | 64.6 | 76.0 | 88.3 | **53.8** |
| FAST | 97.8 | 35.6 | 60.1 | 83.2 | 45.0 |
| GR00T | 98.7 | 65.3 | 75.3 | 88.5 | 52.8 |
| $\pi$ | 98.1 | **65.9** | **76.6** | **88.8** | 48.9 |

离散 FAST 多数任务明显落后；三种连续头没有稳定赢家。$\alpha$ 因而给出一个务实结论：强 VLM 下，MLP 是需要被新方法认真击败的默认 baseline。但它没有证明 flow matching 在高频、动态、多解或灵巧控制中无价值。

### 3.3 设计轴二：机器人预训练

在 RoboTwin 低数据 setting，无额外预训练 50.3，InternData-A1 63.6，RoboTwin-Rand 79.7；域对齐预训练显著有效。但同样 checkpoint 转到 RoboCasa 时，无预训练 53.8，InternData-A1 35.4，RoboTwin-Rand 33.3；OXE 更降至 27.8。结论不是“预训练无用”，而是朴素 action-specific pretraining 可把表示拉向预训练本体，导致跨本体负迁移。

### 3.4 设计轴三：数据工程

proprioception 在 RoboTwin Clean $50\times50$ 把 50.3 提到 60.8，但数据充足时 88.0 与默认 88.2 几乎相同；delta action 在 RoboCasa $24\times10$ 把 9.8 提到 15.8，$24\times1000$ 时仅 54.8 对 53.8。数据工程在低数据时是有效 inductive bias，数据充分时边际收益下降。

### 3.5 设计轴四：generalist optimization

2B→4B 在 WidowX 由 52.1 到 70.2，4B→8B 仅到 71.5；RoboCasa 50.7→57.3→58.2。相比之下，total batch 64→1024 将 RoboCasa 40.0→59.2、WidowX 35.4→70.2。该配方中 batch diversity 比继续放大 4B 模型更关键，说明多任务梯度覆盖是 generalist 学习的一阶因素。

### 3.6 新瓶颈如何暴露

$\alpha$ 找到一个看似矛盾的事实：不做 action pretraining 的强 VLM 已经很好；做域内 pretraining 可显著提高低数据目标，却伤害未见本体。问题因此从“要不要预训练”转变成“如何预训练才不破坏通用表示”。这正是 VLAct 的起点。

---

## 4. 第三次问题转移：从动作拟合到可复用表示

### 4.1 VLAct 不再优化固定下游模型

VLAct 的预训练动作头是脚手架。继续预训练时，Qwen3-VL backbone 同时接受 OFT、PI、GR00T 连续 head 监督和 caption loss；微调时三头全部丢弃，为每个下游任务重新初始化目标 head。因而最终交付物是 VLA-oriented VLM backbone，而不是一个绑定控制器的通用 policy。

![](../../10-Papers/VLA/assets/VLAct%20-%20表示中心的VLA继续预训练/figures/02-method.png)

> VLAct 的代表图：保护浅层与 caption 保留 VLM prior；多连续 head 约束共享表示；部分统一动作空间和 wrap loss 对齐跨本体语义；下游重新初始化 head。来源：VLAct Figure 3。

### 4.2 机制一：防止 VLM prior 遗忘

继续预训练冻结 vision encoder 与下半 LLM，只训练上半 LLM 和动作头；同时混入 caption，以 $0.5\mathcal{L}_{\mathrm{VLM-CE}}$ 锚定视觉语言表征。全更新/冻 vision/冻 vision+下半 LLM 在 LIBERO-Plus 为 78.9/81.3/82.6，在 RoboTwin 为 77.1/79.3/80.5。caption 单源使 LIBERO-Plus 从 robot-only 75.0 到 82.6。

### 4.3 机制二：防止 decoder lock-in

OFT-only 预训练后换未见 PI head，RoboTwin 55.1，低于 scratch PI 60.5；OFT+GR00T 后换 PI 为 63.1；OFT+PI+GR00T 为 77.0。matched-head 中，多头也比单头高：OFT 80.5 vs 78.8，PI 77.0 vs 75.4，GR00T 76.0 vs 71.7。head diversity 因而既改善兼容性，也起 regularizer 作用。

### 4.4 机制三：跨本体只共享正确语义

20 维布局将 AgileX 左/右臂 absolute joints、Franka delta EEF 分槽，只共享 gripper 等物理一致维；inactive dimensions mask。separate head/unified head/语义统一表示在 RoboTwin 为 78.5/79.5/80.5，在 LIBERO-Plus 为 81.1/81.4/82.6。

绝对 joint 先 wrap 到 $[-\pi,\pi]$，残差也做 modulo：

$$\delta_{\mathrm{wrap}}=((\hat a-a)+\pi)\bmod2\pi-\pi$$

raw joint / joint wrap / wrap + loss 在 RoboTwin 为 75.5/78.6/80.5。这一结果提醒：$\alpha$ 所说“重数据工程收益有限”不能扩展到物理拓扑错误；正确动作几何仍不可替代。

### 4.5 结果为何比普通 SOTA 更有解释力

在下游 data、head、optimizer、budget 全固定时，只换 backbone：LIBERO-Plus 75.0→82.6，VLA-Arena 33.4→54.8，RoboTwin Base 61.7→80.5。未见 GR-1 本体上，20% 下游数据 49.5 超过 GR00T-N1.6 全数据 47.6；真机双臂 72.0 vs baseline 44.0。该 protocol 把收益更可信地归因于表示，而非复用 head 或更多下游数据。

---

## 5. 技术时间线：问题如何连续演化

| 时间 | 工作 | 解决了什么 | 暴露的新问题 |
|---|---|---|---|
| 2026-04 | StarVLA（2604.05014） | 统一 backbone/head、raw I/O、训练器、benchmark 和部署接口 | 能比较后，发现还缺严格受控的设计轴实验 |
| 2026-04 | StarVLA-$\alpha$（2604.11757） | 建立 Qwen3-VL + MLP 强 baseline，系统消融 head/pretraining/engineering | 朴素 action pretraining 域内提升、跨本体负迁移；matched-head 可能掩盖表示专化 |
| 2026-08 | VLAct（2608.27550） | 以 prior preservation、多 head、部分统一动作空间学习可复用 backbone | 规模仅 4B、Memory 弱、预训练细节披露与更广本体扩展仍不足 |

两篇 2026-04 工作不是“先发布模型再发布 alpha 轻量版”。StarVLA 是社区框架报告，$\alpha$ 是基于该框架进行方法学实验的论文；二者几乎同期出现，功能关系是“平台支持实验”，不是版本号关系。

---

## 6. 作者与生态重叠

StarVLA 以 Community 名义署名；$\alpha$ 与 VLAct 的作者高度重叠，包括 Jinhui Ye、Senqiao Yang、Yuxin Chen、Zixuan Wang、Pengguang Chen、Shu Liu、Jiaya Jia 等。VLAct 也明确构建于 StarVLA codebase，并直接复用 OFT/PI/GR00T head。

作者重叠说明三篇形成同一研究路线，但不等于同一模型。更准确的描述是：**同一生态提供共享实验底座，不同论文在不同抽象层提出问题。**

---

## 7. 横向对比

| 维度 | StarVLA | StarVLA-$\alpha$ | VLAct |
|---|---|---|---|
| 交付物 | 开源 codebase/framework | 默认 Qwen3-VL+MLP baseline 与设计轴结论 | 继续预训练后的 VLA-oriented backbone |
| 默认输入 | raw images + language + optional state/sensors | raw RGB + language | robot images + language + actions；辅助 caption 数据 |
| 输出 | normalized action chunk + optional auxiliary output | 下游原生连续 action chunk；generalist padding 到 32 维 | 预训练每步 20 维 masked canonical action；下游输出由新 head 决定 |
| 核心机制 | 双层接口、可插拔骨干/head、server-client | 最小处理、简单 MLP、受控比较 | shallow protection、caption、多连续 head、部分统一空间、wrap loss |
| FAST/OFT/$\pi$/GR00T | 作为可插拔实例 | 作为动作头设计轴 | OFT/PI/GR00T 作为预训练脚手架，FAST 用于 pilot |
| 训练目标 | action + optional auxiliary | 连续动作回归为默认 | 三 head action loss + 0.5 VLM CE + wrap loss |
| 下游 head | 框架配置决定 | 默认 MLP，消融换 head | 预训练 head 全丢弃，下游重新初始化 |
| 主要证据 | 多 benchmark 可复现、generalist、扩展效率 | head/pretraining/engineering/batch 消融 | unseen-head、本体迁移、prior protection 与动作空间消融 |
| 主要局限 | adapter/版本/算力、RL 未完成 | Qwen3-VL 与桌面任务范围、统计不足 | 仅 4B、Memory 弱、训练细节不集中、schema 可扩展性 |

---

## 8. 关键数字如何连起来看

### 8.1 基础设施阶段：先证明统一不等于性能妥协

StarVLA 在 LIBERO 提供 95%–96% 级多 head、多骨干结果，在 RoboTwin 连续 head 约 88%，RoboCasa OFT 48.8，generalist RoboCasa 57.3。其目标是证明统一框架可成为强研究底座。

### 8.2 受控基线阶段：复杂度未必是收益来源

$\alpha$ specialist 在 LIBERO 98.8、RoboCasa 53.8、RoboTwin random 88.3；FAST 在 WidowX 仅 35.6，而连续头 64.6–65.9。大规模 OXE 预训练使 RoboCasa 从 53.8 降至 27.8；本域 RoboTwin-Rand 可把低数据 RoboTwin 50.3 提到 79.7，却把 RoboCasa 降到 33.3。这里首次把“预训练收益”拆成域内与跨域两面。

### 8.3 表示阶段：在同 protocol 下只换 backbone

VLAct 将 LIBERO-Plus 75.0→82.6、VLA-Arena 33.4→54.8、RoboTwin Base 61.7→80.5；20% RoboCasa 49.5 超过全数据 baseline；真机双臂 44.0→72.0。相较跨论文 SOTA，这些“只换 backbone”的差值更能支撑表示中心主张。

### 8.4 仍不能忽略绝对难度

VLAct 在 DOMINO SR 18.5、RoboDojo SR 7.6、RoboDojo Memory 0.56、真机 fold towel 50%。这说明三篇工作改善了研究与训练方式，但没有解决长时记忆、动态 manipulation 和柔性物体控制。

---

## 9. 评测体系的变化

### 9.1 从单 benchmark specialist 到 all-in-one generalist

StarVLA 支持多个 benchmark，但早期结果仍以各自 specialist 为主；$\alpha$ 把“单模型联合训练、无 benchmark-specific 微调”提升为 generalist evaluation，要求一个模型同时处理 LIBERO、SimplerEnv、RoboTwin 和 RoboCasa。这个变化把评测目标从“某个环境调到最好”转成“跨任务与本体能否共存”。

### 9.2 从 matched-head performance 到 backbone transfer

VLAct 进一步改变评价单位：预训练用什么 head 并不重要，重要的是 backbone 换新 head 后是否仍强。unseen-head transfer、same-head adaptation 和 head re-initialization 共同防止把 decoder 适配误认为通用知识。

### 9.3 从内部 benchmark 到外部 leaderboard 与真机

RoboDojo 是外部定义的 42 任务、35 policy 榜单；VLAct 的排名说明方法不只对作者选择的 benchmark 有效。但 leaderboard 不统一训练数据与算力，因此它是 external validity，不是严格受控因果证据。真机固定初始状态和共享 demonstrations 则提供更受控的现实评测。

---

## 10. 当前正在攻坚的问题

### 10.1 长时记忆与阶段保持

证据来自 VLAct RoboDojo Memory 0.56% success，以及真机 baseline 常跳过 scooping 阶段。VLAct 改善了长时任务，但 memory 仍是最弱轴。未来需显式 temporal state、episodic memory、视频目标或层级规划。

### 10.2 更大模型下的最优配方

VLAct 作者明确限制只研究 4B；$\alpha$ 显示 4B→8B 下游收益很小，但这不等于 continued pre-training 配方在大模型上不变。冻结比例、caption 权重、head 数量和学习率都可能随 scale 改变。

### 10.3 动作空间 schema 的可扩展性

32 维 simple padding 在 $\alpha$ 的四 benchmark 上有效；20 维语义布局在 VLAct 的 Franka/AgileX 上更严谨。扩到移动底盘、灵巧手、力矩控制、触觉和高频轨迹后，手工维度分配是否可持续，是统一接口与统一表示共同面对的问题。

### 10.4 动态、柔性物体与闭环恢复

DOMINO 18.5% SR、fold towel 50% 显示表示提升未替代控制与动力学建模。世界模型、在线修正、触觉反馈与短 horizon receding control 可能是互补方向；StarVLA 的统一框架可用于将这些模块与 VLAct backbone 受控组合。

### 10.5 可复现性与资源可及性

StarVLA 报告 8–256 GPU scaling，$\alpha$ generalist 64×A100，VLAct 16 GPU。虽然相对工业专有系统更开放，普通实验室仍难重跑全部矩阵。VLAct PDF 又未集中披露完整 pretraining steps/LR/mixture ratio。未来需要 experiment manifest、checkpoint provenance、小算力配方和多 seed 结果。

---

## 11. 已确立事实与我的判断

### 11.1 已由论文实验支持的事实

- StarVLA 的统一接口能承载至少四类动作头、VLM/世界模型骨干、多 benchmark 与 server-client 部署。
- 在 $\alpha$ 的控制条件下，连续动作头普遍优于 FAST，而 MLP/GR00T/$\pi$ 没有稳定绝对赢家。
- action-specific pretraining 的收益依赖域对齐，并可能损害跨本体迁移。
- VLAct 的 shallow protection、caption、多 head、统一动作表示和 wrap loss 均有对应消融支持。
- 在相同下游协议中，只换 VLAct backbone 可在多个 benchmark 与真机上提高性能。

### 11.2 我的判断

- 这条路线最重要的贡献是**逐步提高实验可归因性**，而不是三个榜单模型。
- $\alpha$ 与 VLAct 不是互相否定：$\alpha$ 说明“朴素额外复杂度不值得默认采用”，VLAct 说明“有机制目标、受控消融的预训练复杂度可以值得”。
- StarVLA 的长期价值取决于接口是否持续稳定、配置与 checkpoint 是否可追溯，而不是当前集成 benchmark 数量。
- VLAct 的下一步应把表示分析从下游 success 推进到直接 probes，并解决 memory/dynamics，而非继续只扩动作头数量。

---

## 12. 实践选型建议

### 12.1 想开发或比较新 VLA 组件

从 StarVLA 入手，先遵守统一 raw I/O 和 backbone-head contract。固定一个强 baseline，再只替换目标组件。不要同时改数据、骨干和 head。

### 12.2 想建立可靠下游 baseline

从 StarVLA-$\alpha$ 入手：Qwen3-VL-4B + MLP/OFT、最小处理、官方 evaluator。只有在这个 baseline 上有稳定增益，才值得引入复杂 flow head、状态输入或动作工程。

### 12.3 想做跨任务/本体预训练

从 VLAct 的 protocol 入手：保护 VLM prior，测试 matched/unseen head，重新初始化下游 head，统一动作空间时只对齐物理同义维度，并尊重角度等动作流形。不要只报告预训练 head 复用后的结果。

### 12.4 最小复现顺序

1. StarVLA + LIBERO + OFT 跑通 raw I/O、policy server 和 evaluator；
2. 复现 $\alpha$ 的 MLP vs FAST/flow 小矩阵；
3. 在 RoboTwin Base 复现 OFT-only→PI 与 OFT+GR00T→PI；
4. 做 raw joint→wrap→wrap loss；
5. 最后才扩到多 benchmark、真机和 16+ GPU continued pre-training。

---

## 13. 文献索引与材料来源

### 13.1 三篇核心工作

1. StarVLA Community. *StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing*. arXiv:2604.05014, 2026-04. 详见 [[StarVLA - 乐高式VLA开源开发框架]]。
2. Ye et al. *StarVLA-$\alpha$: Reducing Complexity in Vision-Language-Action Systems*. arXiv:2604.11757, 2026-04. 详见 [[StarVLA-alpha - 降低VLA系统复杂度]]。
3. Yang et al. *Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models*. arXiv:2608.27550, 2026-08. 详见 [[VLAct - 表示中心的VLA继续预训练]]。

### 13.2 关键技术前置

- FAST：连续 action chunk 的离散 tokenization 与自回归预测；
- OpenVLA-OFT：action token hidden state 上的并行连续回归；
- $\pi_0$：flow-matching 动作 expert；
- GR00T：VLM 语义系统与 DiT motor system 的双系统设计；
- Qwen3-VL：三篇实验的核心强 VLM backbone；
- LIBERO/LIBERO-Plus、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1、DOMINO、VLA-Arena、RoboDojo：从标准桌面成功率到鲁棒性、动态、未见本体与记忆的评测轴。

### 13.3 本地材料

- `10-Papers/VLA/assets/StarVLA - 乐高式VLA开源开发框架/sources.md`
- `10-Papers/VLA/assets/StarVLA-alpha - 降低VLA系统复杂度/sources.md`
- `10-Papers/VLA/assets/VLAct - 表示中心的VLA继续预训练/sources.md`

所有核心数字、公式、时间与限制均依据本地官方 PDF 和官方 arXiv LaTeX 源码核对；本综述未修改 `Paper-Index.md`。
