---
type: paper-note
status: complete
domain: VLA
paper: "StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing"
year: 2026
arxiv: "2604.05014"
doi: null
source: https://arxiv.org/abs/2604.05014
project: https://starvla.github.io
code: https://github.com/starVLA/starVLA
tags:
  - VLA
  - infrastructure
  - open-source
  - action-head
  - benchmark
created: 2026-10-21
updated: 2026-10-21
---

# StarVLA：乐高式 VLA 开源开发框架

## 基本信息

| 项目 | 内容 |
|---|---|
| **正式标题** | *StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing* |
| **作者/组织** | StarVLA Community 与 HKUST Von Neumann Institute |
| **arXiv** | 2604.05014，2026 年 4 月技术报告 |
| **研究对象** | VLA 研发基础设施、统一接口、可插拔骨干与动作头、训练/评测/部署流水线 |
| **代码** | <https://github.com/starVLA/starVLA> |
| **核心定位** | 不是提出一个唯一的“StarVLA 模型”，而是提供可以实例化多种 VLA 的开放框架与强基线集合 |

## 一句话结论

StarVLA 的主要贡献不是某个新的动作生成公式，而是把原本绑定在各自代码库中的 VLM/世界模型骨干、FAST/OFT/$\pi$/GR00T 动作头、异构数据加载、分布式训练、跨 benchmark 评测和真机部署抽象成可替换的标准组件；它用多组可复现结果证明，这种“基础设施先统一”的路线既能降低实验变量纠缠，也能在 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1 等任务上提供有竞争力的锚点，但 benchmark adapter、训练成本、版本演化和尚未完成的 RL 支持仍决定了它不是一键消除 VLA 工程复杂度的万能层。

---

## 1. 研究背景与问题定义

### 1.1 真正的问题是系统碎片化，而非缺少第五种动作头

VLA 研究看似都在解决“图像 + 指令 $\rightarrow$ 动作”，实际工程接口却高度不一致。不同工作选择不同视觉语言骨干、图像预处理、动作参数化、action chunk 长度、归一化统计、推理循环和 benchmark 协议。一个方法的增益往往同时混入骨干更强、数据更多、动作头更复杂、评测脚本不同等变量，导致论文间的数字难以直接解释。StarVLA 将这种状态称为 VLA 的“巴别塔”：想复现一个动作头，通常不得不连带搬运整套数据与部署代码；想比较两个动作头，又很难保证其余条件真的一致。

论文把碎片化拆成三层。第一是**架构层**：动作可被离散成自回归 token，也可由 MLP 并行回归，还可由 diffusion/flow matching 迭代生成，甚至可采用慢 VLM + 快动作专家的双系统。第二是**系统层**：模型、数据处理、训练器和推理服务被紧耦合，组件不能交叉复用。第三是**评测层**：各工作只覆盖若干 benchmark，且动作后处理、相机输入和 rollout 细节不一致。StarVLA 的回答是先找不随算法变化的“稳定边界”，再把变化限制在少量模块内部。

### 1.2 论文提出的两个不变量

第一个不变量是部署边界：机器人最终总是提供原始传感器观测和语言目标，策略最终总要返回可执行动作。第二个不变量是模型内部的骨干—动作头边界：视觉语言骨干提供共享表示，动作头把表示变成 action chunk。基于这两个不变量，StarVLA 试图让骨干、动作头、训练器和 benchmark 适配器可以独立替换。

这不是纯软件重构。对研究方法而言，只有当输入、数据、训练循环和评测保持不变时，动作头的受控比较才有意义；对工程而言，只有当训练和推理使用同一类原始样本字典时，才更不容易出现“训练预处理在 dataloader 内、部署时忘记复刻”的静默分布偏移。

### 1.3 核心贡献

1. 提出统一的策略中心视角，把直接 VLA、VLM-based VLA 和 world-model-based VLA 都写成动作预测加可选辅助预测。
2. 以 `BaseFramework.forward()` 与 `BaseFramework.predict_action()` 建立统一训练/推理接口，并以标准隐藏状态契约连接骨干和动作头。
3. 实现四种代表性动作解码范式：StarVLA-FAST、StarVLA-OFT、StarVLA-$\pi$、StarVLA-GR00T；骨干还可在 Qwen-VL 等 VLM 与 Cosmos-Predict2 等世界模型间切换。
4. 将机器人行为克隆、多模态 co-training、跨本体数据混合、分布式训练和 server-client 部署纳入同一配置体系。
5. 集成 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1、BEHAVIOR-1K 等评测，并给出 specialist、generalist 和效率测量。

![](assets/StarVLA%20-%20乐高式VLA开源开发框架/figures/01-unified-formulation.png)

> 图 1（论文 Figure 1，p.3）：统一 VLA 形式。真正被固定的是“多模态观测与语言输入到未来 action chunk”的政策接口；视觉预测、语言规划等属于可选辅助输出。这个抽象允许直接策略、VLM-VLA 与 WM-VLA 共用外部契约。来源：官方 arXiv 源码与 PDF，<https://arxiv.org/abs/2604.05014>。

---

## 2. 任务定义与输入输出

### 2.1 策略中心的统一形式

论文将 VLA 写成：

$$\pi(\mathbf{a}_{t:t+k},\mathbf{y}_{\mathrm{aux}}\mid\mathbf{x}_{\le t},\ell)$$

其中 $\mathbf{x}_{\le t}$ 是截至时间 $t$ 的多模态观测历史，可包含 RGB、深度、触觉、本体状态等；$\ell$ 是自然语言任务指令；$\mathbf{a}_{t:t+k}$ 是未来 $k$ 步动作块；$\mathbf{y}_{\mathrm{aux}}$ 可包含未来视觉、语言规划或其他模态预测。训练目标概括为：

$$\mathcal{L}=\mathcal{L}_{\mathrm{action}}+\mathcal{L}_{\mathrm{aux}}$$

直接动作策略令 $\mathcal{L}_{\mathrm{aux}}=0$；VLM-based 路线可加入语言规划、grounding 或推理 token；世界模型路线则可加入未来图像/视频预测。这里的价值不是声称这些方法在统计建模上完全相同，而是识别出它们可以共享的系统接口。

### 2.2 训练入口与推理入口

统一训练入口可概括为：

```text
forward({raw images, instruction, action chunk, ...}) -> {action_loss, auxiliary_loss, ...}
```

统一推理入口为：

```text
predict_action({raw images, instruction, optional state, ...}) -> {normalized_actions, ...}
```

论文强调输入保留为接近环境层的原始样本，而不是让 framework 只接受某个 dataloader 预先加工出的私有 tensor。这样做使训练输入更接近部署观测，也把 resize、tokenization、hidden-state selection 等骨干特有逻辑收进 adapter。输出使用 normalized action chunk，最终由 benchmark 或机器人侧 adapter 根据 checkpoint 中的 `dataset_statistics.json` 反归一化，再完成 sticky gripper、delta/absolute 转换和 action ensembling。

### 2.3 输入输出并没有被过度强行统一

StarVLA 的统一是一种“契约统一”，不是把所有机器人动作硬解释成同一物理量。LIBERO 可输出 8 步连续 EEF 控制；SimplerEnv、RoboCasa 和 RoboTwin 可拥有不同维度与 chunk 长度；generalist 实验才使用简单 padding 将低自由度动作扩展到共享 32 维。换言之，框架允许任务保留原生控制语义，只要求 adapter 把它们装进标准字典并在边界处转换。

---

## 3. 核心方法：从代码组织到动作头框架

### 3.1 双层模块化：外部 I/O 与内部表示契约

StarVLA 同时标准化两条边界：

- **外部边界**：环境原始观测 $\rightarrow$ policy server $\rightarrow$ normalized action chunk；
- **内部边界**：VL foundation model $\rightarrow$ hidden states $\rightarrow$ action head。

骨干和动作头均通过 YAML 声明式配置装配。接入新骨干时，需要实现原始多模态输入到共享隐藏状态的 adapter；接入新动作头时，只需消费规定的表示并返回统一损失和动作。训练器、benchmark harness 与部署服务因此不需要了解动作头内部是 token prediction 还是 flow matching。

### 3.2 四类动作头实例化

![](assets/StarVLA%20-%20乐高式VLA开源开发框架/figures/02-action-head-frameworks.png)

> 图 2（论文 Figure 2，p.6）：同一骨干上的四类动作解码。图的重点不是宣称四者等价，而是让算法差异只停留在动作头及其所需的表示连接方式中。来源：官方 arXiv 源码与 PDF。

#### 3.2.1 StarVLA-FAST：离散自回归动作 token

FAST tokenizer 把连续 action chunk 压缩并离散化为 token，VLM 继续使用 next-token prediction 产生动作序列。其优势是最大程度复用语言模型原生接口，动作可和文本共享自回归框架；代价是离散化误差、串行生成延迟，以及连续精细控制可能丢失幅度和时序细节。

#### 3.2.2 StarVLA-OFT：轻量并行连续回归

OFT 在预定义 action token 的 hidden state 上接一个 MLP，以 L1 损失并行回归整段连续动作。它是四者中结构最简单、推理路径最短的方案，也最适合把 VLM 表示质量与动作生成复杂度分开研究。论文在多个 benchmark 上发现它并不弱：LIBERO 平均 96.6%、RoboCasa-GR1 48.8%，且常与更复杂连续头相当或更好。

#### 3.2.3 StarVLA-$\pi$：cross-DiT flow matching

$\pi$ 风格头使用多层 VL hidden states 通过 cross-attention 条件化 DiT 动作专家，从高斯噪声出发迭代去噪得到连续 action chunk。与 OFT 的单点 L1 回归相比，它更适合表达多模态连续动作分布，但推理需要多个积分/去噪步骤，部署延迟和实现复杂度更高。

#### 3.2.4 StarVLA-GR00T：慢推理与快控制双系统

GR00T 变体把 VL 骨干视为 System 2，把 DiT flow-matching 模块视为 System 1。其差异不仅是损失函数，还包括推理时计算模式：语义表示和电机动作生成更明确解耦。StarVLA 能让这种双系统与 OFT 共享 benchmark client，说明其抽象覆盖了不同 inference pattern，而不只是相似 MLP 的配置差异。

### 3.3 数据流、训练器与部署

![](assets/StarVLA%20-%20乐高式VLA开源开发框架/figures/03-training-dataflow.png)

> 图 3（论文 Figure 3，p.7）：框架数据流。异构机器人数据和图文/视频数据经过可插拔 dataloader 进入统一 `DataDict`，模型通过统一 `forward` 与 `predict_action` 连接训练和 policy server；benchmark 与真机控制保留在独立客户端环境。来源：官方 arXiv 源码与 PDF。

数据层提供 `LeRobotMixtureDataLoader`、Qwen-VL JSONL loader、video-text loader 与可扩展自定义 loader，最后形成含 images、language、action、可选 proprioception 的灵活字典。模型构建由 `build_framework(cfg)` 完成，训练使用显式 PyTorch loop、Hugging Face Accelerate 与 DeepSpeed，支持 bf16、梯度累积、gradient clipping、cosine schedule 及最小学习率。

server-client 架构解决了一个常见但容易被论文忽略的问题：不同 simulator 的依赖环境互相冲突。模型 checkpoint 在 GPU 环境中由 WebSocket server 托管；LIBERO、SimplerEnv、RoboTwin 等 evaluator 留在各自官方 conda 环境中，通过 `msgpack` 发送观测字典。这样既避免把 simulator 依赖塞进模型环境，也让真机控制器复用相同 client 协议。

---

## 4. 训练范式

### 4.1 单 benchmark 行为克隆

最基础路径是 robot-only supervised learning。模型从公开 VL 预训练权重初始化，直接在 benchmark 标准演示上优化动作损失。论文有意不使用额外 VLA 预训练、图像增强或 DAgger，目的是把结果作为可复现的“锚点”，方便后来者衡量新增数据、预训练或动作头带来的边际收益。

LIBERO 示例使用 8 张 A100、每卡 batch 16、最多 100K step；报告的强结果多在 30K step 已取得。SimplerEnv 和 RoboCasa 使用 16 张 A100；RoboTwin 使用 48 张 A100、每卡 batch 4、150K step。由此可见，“框架降低门槛”不等于“实验便宜”：完整复现实验矩阵仍是数十卡级。

### 4.2 多模态 co-training

robot-only SFT 会在数千至数万步内破坏 VLM 的 grounding、指令理解和场景表征。StarVLA 使用 VLA 与 VLM 双 dataloader，每个优化 step 进行两次 forward/backward：机器人样本产生 action loss，web/VQA/grounding 样本产生 language-modeling loss，并通过配置项控制 VLM loss scale。

框架报告的 ST4VLA 案例中，vanilla VLA 的 RefCOCO-g 表现约 20K step 就接近随机；普通 co-training 有恢复但振荡；空间引导并带空间预训练的版本同时达到 RefCOCO-g IoU@0.5 71.2、RoboRefIt 74.3、Google Robot VM/VA 84.6/75.9、WidowX 73.2。应注意这些数字来自基于 StarVLA 的 ST4VLA 研究，不是仅靠基础框架自动获得的结果。

### 4.3 跨本体与跨 benchmark 混训

`LeRobotMixtureDataset` 根据 `(dataset name, sampling weight, robot type)` 采样多机器人轨迹并携带 embodiment tag。论文最直接的 generalist 实验把 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1 合并训练，学习率 $10^{-4}$、总 batch 256，不做 benchmark-specific fine-tuning，只把动作 padding 到 32 维。

该 generalist 在 LIBERO 平均 97.8、SimplerEnv WidowX/Google VA/Google VM 为 70.2/73.8/79.3、RoboTwin clean-random 为 88.7/87.8、RoboCasa-GR1 为 57.3。尤其 RoboCasa 从最佳框架 specialist 53.8 提升到 57.3，说明异构数据可能提供正迁移；但 RoboTwin 表中 generalist 的一个 clean 条件未报告，不能把这张表解释成对所有维度全面胜出。

### 4.4 强化学习仍是计划而非已交付能力

论文明确说 RL fine-tuning 正与 RLinf 集成，当前公开代码重点仍是监督学习和 co-training。因而 README、论文愿景或路线图中出现 RL，不应被写成已经完成验证的功能。对于希望复现 SFT 后在线 RL 的用户，仍需检查代码版本和实际可运行入口。

---

## 5. Benchmark 与主要数字

### 5.1 LIBERO：数据效率强，但已接近饱和

LIBERO 输入使用主视角与腕部 RGB，不使用 proprioception、历史堆叠或图像增强；输出为 chunk=8 的连续 EEF 控制。每套 10 个任务、每任务 50 episode，共 500 次试验。Qwen3-VL-4B 骨干下，FAST/OFT/$\pi$/GR00T 平均分别为 95.4/96.6/95.7/96.5；Cosmos-Predict2-2B 下 OFT/$\pi$/GR00T 为 95.8/95.5/95.2。OpenVLA-OFT 以 175K step、223 epoch 得到 97.1，StarVLA-OFT 用 30K step、约 9.54 epoch 得到 96.6。这个结果更能说明训练管线效率，而不是说明 96.6 绝对超过 97.1。

### 5.2 SimplerEnv：同一框架可覆盖 WidowX 与 Google Robot

WidowX Visual Matching 上，Qwen3-VL-4B 的 FAST/OFT/$\pi$/GR00T 平均为 31.6/64.6/60.9/65.3；Cosmos 版本 OFT/$\pi$/GR00T 为 61.2/58.7/61.6。离散 FAST 在这里明显落后，表明 benchmark 难度和控制精度会放大动作头差异。Google Robot 上仅重点评测 StarVLA-OFT：Visual Matching 76.0、Variant Aggregation 70.2；前者略高于 SpatialVLA 75.1 和 CogACT 74.8，后者与 SpatialVLA 70.7 接近。

### 5.3 RoboTwin 2.0：多任务双臂控制

训练覆盖 50 个任务，clean/randomized 共 10,000 次评测。StarVLA-FAST 为 72.5 clean / 83.2 random；OFT 为 88.2/88.3；GR00T 为 88.0/88.5；$\pi$ 为 88.1/88.8。连续头高度接近，FAST 在 clean 上低很多却在 random 上异常更高，提示不同 checkpoint、训练混合或离散化可能与数据条件相互作用；论文没有进一步做统计显著性或多 seed 分析。

### 5.4 RoboCasa-GR1：动作头差异更明显

24 个 humanoid-style tabletop 任务上，$\pi_{0.5}$ 37.0、GR00T-N1.6 47.6；StarVLA-FAST 39.0、$\pi$ 43.9、GR00T 47.8、OFT 48.8。与接近饱和的 LIBERO 相比，RoboCasa 更能区分动作头：简单 MLP 不但没有被复杂生成头压倒，反而是单 benchmark 最佳，这为后续 StarVLA-$\alpha$ 的“简洁强基线”研究提供了直接动机。

### 5.5 结果应如何解读

这些表不是一次严格统一预算的全因子消融。不同 benchmark 使用不同 GPU 数、训练 step、checkpoint 选择与数据集规模；某些 baseline 是官方报告，某些是重实现。StarVLA 最可靠的结论是“统一框架可以产出强而可复现的基线，并支持受控比较”，而不是“某个动作头普遍优于所有其他动作头”。真正将动作头、预训练和数据工程作为设计轴严格控制的是后续 StarVLA-$\alpha$。

---

## 6. 效率与扩展性

### 6.1 单节点吞吐

在 8×A100 上以 StarVLA-GR00T + Qwen3-VL-4B、RoboCasa-GR1 测量：每卡 batch 从 2 增至 24 时，step latency 从 0.703 秒增到 2.404 秒，samples/s 从 22.7 增到 79.9，GPU utilization 从 74% 增至 96%。这说明“每步更慢”和“总体吞吐更高”可以同时发生，训练规划应按固定 step 还是固定样本量来选择 batch。

### 6.2 多节点弱扩展

![](assets/StarVLA%20-%20乐高式VLA开源开发框架/figures/04-multinode-scaling.png)

> 图 4（论文 Figure 6，p.18）：每卡 batch 固定为 8，从 8 扩到 256 GPU。step latency 由 0.735 秒升至约 0.93 秒后趋稳，样本吞吐由 87 增至 2200 samples/s，32 GPU 以后并行效率稳定在约 79%–80%。来源：官方 arXiv 源码与 PDF。

8/16/32/64/128/256 GPU 的吞吐分别为 87.0/150.7/284.7/553.8/1111.5/2200.0 samples/s，对应扩展效率 100/86.7/81.9/79.6/79.9/79.1%。论文给出的实践判断是：若目标是处理固定数据量，扩卡有效；若目标是跑固定优化步数，跨节点通信使单步并不会更快。值得注意的是，效率数据来自公开 issue 中的 profiling，未覆盖四种动作头、所有骨干和不同 simulator 的端到端评测延迟。

---

## 7. 消融、证据边界与可证伪点

### 7.1 框架论文缺少完整全因子消融

StarVLA 提供了骨干和动作头交叉组合、多 benchmark specialist/generalist、co-training 案例和扩展性测量，但没有在每个 benchmark 上用完全相同数据、step、算力和 seed 做 $4\times N$ 全因子矩阵。因此它证明“统一接口可工作”和“若干实例很强”，尚未完全证明抽象不会对任何新范式造成性能损失。

### 7.2 可以从现有数字读出的三个趋势

1. **强骨干可迁移**：LIBERO 上 Qwen3-VL 与 Cosmos 都能达到 95% 以上，说明接口没有被某个 VLM 写死。
2. **动作头重要性依任务而变**：LIBERO 连续头差距小；SimplerEnv 与 RoboCasa 中 FAST 明显弱；RoboCasa 上简单 OFT 最强。
3. **多数据混训可正迁移，但非全维度无损**：generalist 提升 RoboCasa，却并非所有 benchmark 都优于 specialist。

### 7.3 论文内部的版本性信号

摘要和引言重点列出五个 benchmark，而能力比较表把 `#Bench` 写为七。作为持续维护技术报告，这种差异更可能反映代码库迭代，而非实验造假；但它提醒使用者：论文描述的是某个时间截面，复现前必须锁定 commit、配置和 checkpoint，不能把当前主分支直接等同于 PDF 中的实验版本。

---

## 8. 局限与批判性评价

### 8.1 作者明确或文本直接显示的局限

- RL 微调仍在集成，当前能力集中于 SFT 与 co-training。
- benchmark 差异仍需薄 adapter 处理，包括 resize、反归一化、sticky gripper、action conversion；统一框架隔离了复杂度，但没有消灭复杂度。
- 真机安全逻辑、设备 SDK 与控制循环位于 client 外部，框架不替代机器人安全工程。
- 多 benchmark 结果主要是 capability demonstration，不是完整消融；部分比较使用他人报告数字，训练预算并不完全等价。
- 大型实验成本很高：RoboTwin 48×A100，generalist/大规模设置需要多节点；“开源可复现”仍受硬件门槛约束。

### 8.2 我的判断：最大的价值是建立实验坐标系

StarVLA 最有长期价值的部分不是当前 benchmark 排名，而是把“骨干—动作头—数据—评测”四个轴显式分离。它让后续工作可以回答“收益来自哪里”，而不是继续发布不可拆解的大系统。StarVLA-$\alpha$ 正是这种基础设施上的受控实验：它借助同一框架固定骨干、数据和训练，发现连续动作头之间差距有限、额外机器人预训练可能负迁移、数据工程收益依赖数据规模。

### 8.3 不能从本文推出的结论

不能推出“所有 VLA 都应使用 OFT”，因为高频、多模态动作分布和真机延迟可能需要更丰富生成头；不能推出“世界模型与 VLM 在算法上相同”，论文只证明它们能共享策略接口；不能推出“padding 足以解决所有本体差异”，这里只在给定 benchmark 集上验证；也不能把框架结果与后续 VLAct 的表示学习结果混为同一模型版本。

---

## 9. 复现成本与实践建议

### 9.1 最小可行复现

若目标是验证框架而不是重跑全部报告，建议先选择 LIBERO + Qwen3-VL-4B + OFT：输入只含主视角/腕部 RGB 与语言，不用状态和增强，动作 chunk=8，训练 30K step 已有强锚点。验证顺序应是：数据格式与统计文件、单 batch `forward`、单 episode `predict_action`、policy server/client、官方 evaluator，最后才扩大训练。

### 9.2 完整研究复现的成本

- **环境成本**：模型、LIBERO、SimplerEnv、RoboTwin、RoboCasa 各自依赖栈，需要 server-client 隔离；
- **数据成本**：需下载 LeRobot 格式 benchmark 数据，核对 action normalization、camera key 与 task language；
- **算力成本**：从 8×A100 到 48×A100；若重现多动作头、多骨干和 generalist 矩阵，成本远高于单次训练；
- **评测成本**：RoboTwin 单条件 50×100 episode，clean + random 共 10,000 rollout；SimplerEnv 还需整套评测重复五次取均值；
- **版本成本**：必须记录仓库 commit、YAML、checkpoint selection 与 evaluator 版本。

### 9.3 推荐的受控研究用法

将 StarVLA 当作实验平台时，每次只改变一个轴：固定数据与 checkpoint selection，比较动作头；固定动作头，比较骨干；固定模型，比较 co-training 数据；固定所有训练条件，比较 benchmark adapter。若一次同时替换骨干、动作头和数据，就重新回到了论文所批评的变量纠缠。

---

## 10. 与 StarVLA-$\alpha$、VLAct 的关系

三篇工作**不是同一模型的三个版本**。StarVLA 是基础设施与统一接口层；StarVLA-$\alpha$ 是在该基础设施上构建的简洁受控基线，用于研究哪些设计轴真的重要；VLAct 则进一步改变问题，把目标从“某个下游 policy 训练得多好”转向“继续预训练后得到的 VLM backbone 是否可被不同动作头、任务和本体复用”。

可以把三者理解为三个层次：

1. **StarVLA：让实验可组合、可比较。** 解决代码与评测碎片化。
2. **StarVLA-$\alpha$：让结论可归因。** 固定强 VLM 与简单 MLP，在受控条件下拆解动作头、预训练和数据工程。
3. **VLAct：让继续预训练学到可迁移表示。** 用浅层保护、caption 混训、多连续头共监督和部分统一动作空间，避免 VLM prior 遗忘与 decoder lock-in。

共同点是共享 StarVLA 社区、代码基础和部分作者，也共享 Qwen3-VL、OFT/$\pi$/GR00T 等实验组件；差异在于研究对象：框架、基线、backbone 预训练配方分别处于不同抽象层。

---

## 11. 个人启发与后续问题

### 11.1 基础设施也是科学方法的一部分

当系统差异大到无法控制变量时，统一代码库并非“纯工程”。它决定了动作头比较是否可信、benchmark 数字是否可复现，也决定后续表示学习能否把改进归因到 backbone 而非隐藏工程技巧。

### 11.2 下一步最值得补的实验

- 同一数据、同一 step、同一 seed 下的骨干 × 动作头全因子矩阵；
- policy server 端到端延迟、显存和控制频率，而不只报告训练吞吐；
- adapter 单元测试与动作语义 schema，验证不同 benchmark 的坐标系和 gripper 约定；
- RL 集成完成后的离线 SFT—在线 RL 接口稳定性；
- 对 commit、模型权重、数据版本与 evaluator 版本做可机器读取的 experiment manifest。

### 11.3 最终评价

StarVLA 是一篇“平台论文”：它的学术贡献要以是否帮助社区减少重复造轮子、提高受控实验密度来衡量，而不能只看单个榜单第一。已有结果证明平台能覆盖多种动作头、VLM/世界模型骨干及多 benchmark，并给出强 baseline；其可信边界则是仍需大量 adapter、算力和版本管理。后续 StarVLA-$\alpha$ 与 VLAct 能在同一生态中分别研究简洁基线和表示中心预训练，本身就是基础设施价值的最好证据。

---

## 12. 材料来源

- 官方论文：<https://arxiv.org/abs/2604.05014>
- 官方代码：<https://github.com/starVLA/starVLA>
- 官方项目页：<https://starvla.github.io>
- 本地来源与图片追溯：`assets/StarVLA - 乐高式VLA开源开发框架/sources.md`
- 本笔记中的论文数字均依据本地官方 PDF 与官方 LaTeX 源码核对；未修改 `Paper-Index.md`。
