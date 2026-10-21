---
type: paper-note
status: complete
domain: VLA
paper: "StarVLA-alpha: Reducing Complexity in Vision-Language-Action Systems"
year: 2026
arxiv: "2604.11757"
doi: null
source: https://arxiv.org/abs/2604.11757
project: https://starvla.github.io
code: https://github.com/starVLA/starVLA
tags:
  - VLA
  - strong-baseline
  - controlled-experiment
  - generalist-policy
  - ablation
created: 2026-10-21
updated: 2026-10-21
---

# StarVLA-$\alpha$：降低 VLA 系统复杂度

## 基本信息

| 项目 | 内容 |
|---|---|
| **正式标题** | *StarVLA-$\alpha$: Reducing Complexity in Vision-Language-Action Systems* |
| **作者** | Jinhui Ye、Ning Gao、Senqiao Yang、Jinliang Zheng、Zixuan Wang、Yuxin Chen、Pengguang Chen、Yilun Chen、Shu Liu、Jiaya Jia |
| **arXiv** | 2604.11757，2026 年 4 月技术报告 |
| **基础设施** | 构建于 StarVLA 统一框架之上 |
| **默认模型** | Qwen3-VL-4B + 指定 action token hidden state + 轻量 MLP 连续动作头 |
| **研究目标** | 通过控制骨干、数据、训练和评测条件，重新检验动作头、机器人预训练、数据工程与跨本体动作设计是否真的必要 |

## 一句话结论

StarVLA-$\alpha$ 不是追求更复杂结构，而是把 Qwen3-VL、原始 RGB + 指令、训练集统计归一化和轻量 MLP 动作回归固定为一个“受控、简洁、足够强”的基线：它在 LIBERO 得到 98.8%、SimplerEnv Google VM 76.0%、RoboTwin random 88.3%、RoboCasa-GR1 53.8%，并通过同框架消融发现连续动作预测普遍优于离散 FAST、三类连续头差距很小、异域机器人预训练可能负迁移、状态/历史/动作工程主要在低数据时有条件收益；但这些结论仍受 Qwen3-VL 家族、给定 benchmark、算力规模和有限 seed/统计报告约束，不能被外推成“复杂动作模型永远无用”。

---

## 1. 研究背景与问题定义

### 1.1 为什么“模型越来越复杂”不等于“知道什么有效”

VLA 论文通常同时改变多个变量：视觉塔、LLM、动作头、预训练数据、本体接口、动作坐标系、状态输入、历史帧、action chunk、batch size 和 benchmark 调参。即使新系统的成功率更高，也很难判断收益来自新算法还是更强 VLM、更多数据或隐含工程。StarVLA-$\alpha$ 将研究问题改写为：**在一个强 VLM 已经提供视觉、语言和空间先验时，哪些 VLA 专用复杂度仍然有稳定边际收益？**

因此，本文的贡献不是发明新 head，而是建立 controlled substrate。它固定 Qwen3-VL、统一最小数据处理、官方评测协议和 StarVLA 训练基础设施，逐轴替换动作头、action-specific pre-training 和数据工程。只有在其余条件相同时，才能比较 FAST、MLP、GR00T、$\pi$，或 OXE、InternData-A1、RoboTwin-Rand 预训练的真实作用。

### 1.2 “简洁强基线”包含什么

所谓 clean 有两个含义：

- **最小数据处理**：输入原始 RGB 与 benchmark 提供的语言指令，不默认加入 proprioception、历史堆叠、复杂 prompt 或 benchmark-specific 格式；动作只按训练 split 做零均值、单位方差归一化。
- **简单架构**：使用原生多模态 Qwen3-VL，不再拼接独立 SigLIP/DINO/CLIP 视觉塔；在指定 action token hidden state 上接轻量 MLP，一次并行回归 action chunk。

论文没有否认 adapter 的存在，而是把 benchmark 异质性限制在薄层：标准化 observation key、action interface 与 evaluator entry point。它追求的是实验主体不再为每个 benchmark 改模型，而不是宣称物理机器人接口完全同构。

![](assets/StarVLA-alpha%20-%20降低VLA系统复杂度/figures/01-teaser.png)

> 图 1（论文 Figure 1，p.2）：论文把碎片化归因于异构数据、VLA 结构和重度接口工程，并用“简单架构、最小数据处理、跨 benchmark 统一评测”构建受控基线。右侧三项结论是要检验的假设，不应先验理解为普适定律。来源：官方 arXiv 源码与 PDF，<https://arxiv.org/abs/2604.11757>。

### 1.3 核心贡献

1. 在四个模拟 benchmark、五类机器人本体和真机任务上建立 Qwen3-VL + MLP 强基线。
2. 在共享骨干、数据和训练条件下比较离散 token、直接回归与两类 flow-matching 动作头。
3. 检验 OXE、InternData-A1 和域内 RoboTwin-Rand 等 action pre-training 对不同目标本体的收益/负迁移。
4. 检验 proprioception、历史帧、delta action、relative action 的数据规模依赖。
5. 提出 all-in-one generalist evaluation：一个模型联合训练后直接跨 LIBERO、SimplerEnv、RoboTwin、RoboCasa 评测，不做 benchmark-specific 微调。

---

## 2. 任务定义与输入输出

### 2.1 默认 specialist 策略

时间步 $t$ 的输入是一个或多个 RGB 视角 $o_t$ 和语言指令 $\ell$。默认设置不使用机器人状态或历史帧。Qwen3-VL 生成多模态 token hidden states，模型插入专用 action query/token；MLP 从对应 hidden state 预测未来 $K$ 步连续动作：

$$\hat{A}_t=g_\phi(h_t^{\mathrm{act}}),\quad \hat{A}_t\in\mathbb{R}^{K\times d_a}$$

目标动作按训练 split 统计逐维标准化：

$$\tilde{A}_t=(A_t-\mu_{\mathrm{train}})/\sigma_{\mathrm{train}}$$

训练采用直接连续回归，论文将其视为 OFT-style MLP。虽然正文未在方法段重写完整损失，框架与相关实现使用 L1 型并行动作回归；推理时再用训练统计反归一化为 benchmark 原生控制量。

### 2.2 generalist 的跨本体输出

联合训练包含 LIBERO、SimplerEnv、RoboTwin 2.0 和 RoboCasa-GR1 的五个数据集。不同机器人动作维度直接 padding 到 32 维；低维 embodiment 的非活动维为零。论文将这种 simple padding 与 RDT action、multi-action head 比较，用以回答“是否必须为每个本体设计专用 action projector”。

这个方案的统一主要是数值维度统一，不提供显式物理语义对齐。它依赖强 VLM 和数据分布从视觉/语言上下文识别当前本体，因此简单，但也可能在训练混合进一步扩大时遇到坐标冲突。VLAct 后来采用“仅共享物理语义一致维度、其余维度 mask”的部分统一空间，正是对这里简单 padding 的更细化回应。

### 2.3 四种动作头的受控比较

- **FAST**：把 action chunk 离散化并自回归生成 token；
- **MLP/OFT（默认 $\alpha$）**：一次前向并行回归连续动作；
- **GR00T-style**：VLM System 2 + DiT flow-matching System 1；
- **$\pi$-style**：由 flow-matching expert 迭代生成连续动作。

重要的是四者共享 Qwen3-VL、输入数据、训练和 evaluator。因而表格差异比跨论文比较更接近动作头本身的因果效应。

---

## 3. 核心机制：刻意不做复杂化

### 3.1 原生多模态骨干代替视觉塔拼装

Qwen3-VL 自带视觉编码与语言建模，避免单独选择 CLIP/SigLIP/DINO、projector 和 LLM 的组合。论文的假设是：当 VLM 已具备强视觉-语言先验时，VLA 的主要瓶颈未必在额外视觉专家，而可能在训练和数据是否让先验得到保留和正确读取。

### 3.2 MLP 动作头作为线性可读性探针

MLP 头的科学价值在于“简单到难以隐藏能力”。如果它已经能从 action token hidden state 回归高质量动作，则说明 backbone representation 暴露了足够控制信息；如果复杂 flow head 只带来少量增益，就不能把系统性能主要归功于生成式 decoder。相反，如果某任务具有明显多模态动作分布、长时动态或高频控制，复杂头仍可能显示优势。

![](assets/StarVLA-alpha%20-%20降低VLA系统复杂度/figures/02-method.png)

> 图 2（论文 Figure 2，p.4）：上半部展示传统 VLA 中动作类型选择、视觉塔/语言模型/动作头组合和 benchmark-specific 模型；下半部是 StarVLA-$\alpha$ 的 absolute joint + Qwen + MLP + 单 generalist 模型。来源：官方 arXiv 源码与 PDF。

### 3.3 薄 adapter，而不是“没有工程”

输入仍需适配相机数量、分辨率和键名；输出仍需匹配 absolute joint、EEF delta、gripper 约定和控制频率。论文的做法是将这些差异限制在 adapter 与数据字段中，不让其改变主干结构或为每个 benchmark 另做模型。这种边界划分让实验更可控，但并不意味着把模型从模拟器迁到真机可以零成本完成。

### 3.4 训练初始化才是关键变量

补充实验显示，generalist 模型随机初始化时 LIBERO 77.5、WidowX 24.8、RoboCasa 28.8；Florence-2 为 93.2/53.4/39.2；Qwen2.5-VL 为 95.5/65.6/53.6；Qwen3-VL 为 97.8/70.2/57.3。Qwen3.5 在 LIBERO 与 SimplerEnv 更强，但 RoboCasa 为 56.1，略低于 Qwen3-VL 57.3。结论不是“越新越好”绝对成立，而是预训练多模态 prior 对 generalist 训练是一阶变量，统一训练本身无法替代强初始化。

---

## 4. 训练设置与复现条件

### 4.1 默认优化

默认从 Qwen3-VL-4B 初始化 VLM，动作头随机初始化；不做额外 action-specific pre-training。VLM 学习率 $10^{-5}$，动作头学习率 $10^{-4}$，cosine schedule，最多 100K step，每 GPU batch 16。specialist 在单一 benchmark/本体数据上训练；generalist 合并多 benchmark 数据。

算力随数据规模调整：LIBERO 8×A100；SimplerEnv、RoboCasa-GR1、RoboTwin-Clean 各 16×A100；RoboTwin-Clean + randomized 为 48×A100；RoboChallenge Table-30 为 32×A100；所有 benchmark 联训为 64×A100。因而它的“简单”主要指结构和数据管线，不代表低成本。

### 4.2 benchmark 数据规模

- **LIBERO**：130 个任务，常用每任务 50 条 demonstration，约 6.5K trajectory；
- **RoboTwin 2.0**：50 个任务，标准 clean 每任务 50 条；扩展设置再加每任务 500 条 randomized，共约 27.5K；
- **RoboCasa-GR1**：24 个任务、每任务约 1000 条，共约 24K；
- **RoboChallenge**：真机 Table-30，论文重点报告 ARX5 11 个任务及多平台扩展结果。

### 4.3 generalist 训练

正文给出学习率 $10^{-4}$、总 batch 256，并说在五个数据集上训练；附录进一步显示 all-benchmark 需要 64×A100。动作 padding 到 32 维。模型、action head、optimizer 和 unified padding 保持不变，用以研究 action parameterization、模型规模、初始化与 batch size。

---

## 5. 主要 benchmark 结果

### 5.1 specialist 主结果

| Benchmark | StarVLA-$\alpha$ | 关键对比 |
|---|---:|---|
| LIBERO 平均 | **98.8** | OpenVLA-OFT 97.1，$\pi_{0.5}$ 96.9，GR00T-N1.6 97.0 |
| SimplerEnv WidowX | 64.6 | GR00T-N1.6 62.0，$\pi_{0.5}$ 46.9 |
| Google Robot VA / VM | 70.2 / **76.0** | $\pi_{0.5}$ 68.4 / 72.7 |
| RoboTwin clean / clean* / random* | 50.3 / 88.2 / **88.3** | $\pi_{0.5}$ 60.2 / 82.7 / 76.8 |
| RoboCasa-GR1 | **53.8** | GR00T-N1.6 47.6，$\pi_{0.5}$ 37.0 |

RoboTwin 的 50.3 与 88.2 对应不同训练规模：前者是仅 clean、每任务较少数据，后者包含更大 random 数据混合，不能在同一列内直接当作同条件重复。论文主张的是统一基线在多种设置都强，而不是所有单元格都绝对 SOTA。

### 5.2 generalist 主结果

单一 generalist 在 LIBERO 97.8、WidowX 65.2、Google VA/VM 69.8/74.3、RoboTwin clean*/random* 88.7/87.8、RoboCasa 57.3。与自身 specialist 相比，RoboCasa 从 53.8 增至 57.3，WidowX从 64.6 到 65.2；LIBERO 和 Google Robot 略降。该结果支持联合数据可产生跨任务正迁移，但也说明 generalist 不是每个 benchmark 上都无代价。

### 5.3 真机 RoboChallenge

ARX5 的 11 个任务上，StarVLA-$\alpha$ 平均成功率 33.6、progress score 54.5；$\pi_{0.5}$ 为 12.7/27.6，$\pi_0$ 为 3.6/14.7。摘要所说“超过 $\pi_{0.5}$ 20%”对应成功率约 20.9 个百分点，而不是相对提升 20%。任务级表现差异很大：例如 put cup on coaster 达 100% SR，但 fold dishcloth 为 0%，说明平均值不能掩盖长时/柔性物体失败。

### 5.4 LIBERO-Plus 与真机 OOD

仅在标准 LIBERO 训练、直接测 perturbed LIBERO-Plus：specialist total 77.8，generalist 79.7；OpenVLA-OFT 69.6、$\pi_0$ 53.6、$\pi_0$-FAST 61.6。generalist 在 lighting/background/noise 上为 97.8/98.1/80.2，显示强 VLM 与混合训练的鲁棒性。

Franka 真机 OOD 三任务中，pick colored egg ID/OOD 77.1/75.0，egg-carton placement 91.3/68.8，waste sorting 87.5/85.0，总平均由 85.3 降至 76.3。坐标组合任务下降最大，说明组合空间 grounding 仍是明显弱点。

---

## 6. 关键消融一：动作头设计

![](assets/StarVLA-alpha%20-%20降低VLA系统复杂度/figures/03-action-head-ablation.png)

> 图 3（论文 Figure 3，p.6）：在同一 VLM 和训练管线下比较 FAST、MLP、GR00T 与 $\pi$。这种同框架比较比跨论文数字更接近对“动作头”本身的受控实验。来源：官方 arXiv 源码与 PDF。

### 6.1 全表结果

| 动作头 | LIBERO Avg | WidowX | Google VA | Google VM | RoboTwin clean* | RoboTwin random* | RoboCasa |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP/OFT | **98.8** | 64.6 | 70.2 | 76.0 | **88.2** | 88.3 | **53.8** |
| FAST | 97.8 | 35.6 | 58.8 | 60.1 | 72.5 | 83.2 | 45.0 |
| GR00T | 98.7 | 65.3 | 70.7 | 75.3 | 88.0 | 88.5 | 52.8 |
| $\pi$ | 98.1 | **65.9** | **72.8** | **76.6** | 88.1 | **88.8** | 48.9 |

### 6.2 能支持的结论

离散 FAST 在 WidowX、Google、RoboTwin clean 和 RoboCasa 明显落后，说明连续控制精度不能总靠 tokenization 无损保留。三种连续头则大多差距很小：$\pi$ 在 SimplerEnv 略好，MLP 在 LIBERO/RoboCasa 更好，GR00T 居中。因而在此训练规模与任务族上，强 VLM + 连续输出比“哪一种连续生成器”更关键。

### 6.3 不能支持的结论

论文没有对所有任务报告多 seed 方差，也未比较实时推理延迟、控制频率和多模态 action distribution。复杂 flow head 可能在高频动作、演示多解性、灵巧手或长 horizon 中有不同价值。因此合理结论是“MLP 是强而便宜的默认基线”，不是“flow matching 没用”。

---

## 7. 关键消融二：action-specific pre-training

### 7.1 设计

固定 $\alpha$ 架构与超参，比较无额外预训练、OXE 232.6K、InternData-A1 630K 和 RoboTwin-Rand 25K。RoboTwin 与 InternData-A1/RoboTwin-Rand 本体或域更接近；RoboCasa 则用于观察跨本体转移。

### 7.2 结果

无预训练在 RoboTwin Clean $50\times50$ / +Random×100 / +Random×500 为 50.3/78.5/88.2，RoboCasa $24\times10$ / ×100 / ×1000 为 9.8/39.4/53.8。OXE 变为 30.2/40.6/83.6 与 1.2/17.7/27.8，全面下降。InternData-A1 在 RoboTwin 低数据提升到 63.6/80.4/88.6，却使 RoboCasa 降到 2.8/27.6/35.4。RoboTwin-Rand 把 RoboTwin 提高到 79.7/84.1/88.8，却使 RoboCasa 降到 2.2/27.3/33.3。

最重要的反直觉结论是：更多机器人轨迹并不自动产生更通用表示。域对齐预训练可大幅改善域内低数据适配，但也可能让 backbone/动作接口过拟合特定本体，损害未见本体。VLAct 随后正面解决这个问题：预训练目标不再只是 action fitting，而要保持 VLM prior、避免单一 decoder lock-in，并显式共享可对齐动作语义。

---

## 8. 关键消融三：数据工程、动作空间与优化轴

### 8.1 proprioception、历史帧、delta 与 relative action

LIBERO 上默认 98.8，加入 proprioception/history/delta/relative 后为 98.5/97.8/98.1/98.7，均无提升。RoboTwin 最小数据 Clean $50\times50$ 时，proprioception 从 50.3 提到 60.8；数据充分的 +Random×500 时为 88.0，对默认 88.2 几乎无增益。RoboCasa $24\times10$ 时 delta action 从 9.8 提到 15.8；×1000 时 relative action 55.5 相对默认 53.8 仅小幅提升。

因此工程技巧具有明显数据规模依赖：低数据时状态和合适坐标系能缩小搜索空间；数据充分时强 VLM + 行为数据可自行吸收许多模式，工程收益变小。历史帧在这些表中没有稳定收益，不代表所有动态任务都不需要 temporal context，因为评测主要仍是低速桌面操作。

### 8.2 simple padding 对比专用跨本体设计

RDT action / multi-action head / simple padding 在 LIBERO 为 97.2/97.2/97.8，在 WidowX 为 63.9/60.6/65.2，在 Google VM 为 71.4/67.8/74.3，在 RoboCasa 为 52.3/53.5/57.3。simple padding 在给定混合上最好，说明强 VLM 可从上下文识别 embodiment。但其 32 维槽位没有物理语义约束；当本体数量和控制范式继续扩大时，VLAct 的“部分统一 + mask”更稳健。

### 8.3 模型规模与 batch size

![](assets/StarVLA-alpha%20-%20降低VLA系统复杂度/figures/04-scaling-trends.png)

> 图 4（论文 Figure 5，p.9）：左图显示 2B 到 4B 有显著收益，4B 到 8B 收益趋缓；右图显示 total batch 从 64 增至 1024 时 generalist 性能持续上升。来源：官方 arXiv 源码与 PDF。

2B/4B/8B 的 WidowX 为 52.1/70.2/71.5，RoboCasa 50.7/57.3/58.2；4B 已捕获多数增益。batch size 64/128/256/512/1024 时 RoboCasa 为 40.0/44.6/48.3/57.3/59.2，WidowX 为 35.4/55.4/62.3/65.8/70.2。batch 的作用比 4B→8B 更稳定，论文解释为每步看到更多任务/本体多样性，减少 generalist 优化陷入局部模式。

不过这里“同总训练规模”下 batch 变化仍可能改变有效优化步数、学习率适配和梯度噪声，论文没有完整报告 scaling rule。结论应表述为该配方中 batch diversity 很重要，而非任何 VLA 都应盲目使用 1024。

---

## 9. 局限与批判性评价

### 9.1 论文没有独立 limitations 章节

作者结论强调复杂动作头、重数据工程和 task-specific pretraining 并非严格必要，但没有单列 limitations/future work。因此下面区分“论文事实”和“基于设置的边界”，避免把批评伪装成作者原话。

### 9.2 实验边界

- 主体依赖 Qwen3-VL 家族；换成弱 VLM、video-native backbone 或无原生视觉语言对齐的模型，MLP 是否仍足够未知。
- benchmark 以桌面 manipulation 为主，缺少高速 locomotion、灵巧手、移动操作、力觉与长时开放世界任务。
- 复杂动作头比较侧重成功率，未系统报告推理延迟、控制频率、显存、能耗和动作多样性。
- 多数表缺少多 seed 误差条；小于 1–2 点的差异不一定有统计意义。
- generalist 使用 64×A100、大 batch；简单结构并不等于容易由普通实验室完整复现。
- 真机任务 rollout 数有限，ARX5 多项任务只有低成功率；柔性物体和长时任务仍明显失败。

### 9.3 对“预训练有害”的正确解释

OXE/InternData/RoboTwin 的负迁移不能推出机器人预训练本身无价值。它说明**朴素 action fitting + 不对齐动作空间**可能把强 VLM 拉向特定本体。VLAct 的 4B backbone 继续预训练在保持下游 protocol 不变时取得 7.6–21.4 点收益，表明关键不是“要不要预训练”，而是“预训练塑造了什么表示”。

### 9.4 对“数据工程无用”的正确解释

低数据 RoboTwin 和 RoboCasa 上，proprioception、delta/relative action 确有明显收益。论文真正支持的是：当数据和 VLM 足够强时，一些工程的边际收益下降；而不是部署时可以忽略控制坐标、状态估计与传感器时序。

---

## 10. 复现成本与实践建议

### 10.1 最小复现路径

先复现一个 specialist：Qwen3-VL-4B + MLP/OFT，原始 RGB + 指令，训练集统计标准化，不用状态和历史。LIBERO 使用 8×A100；若资源不足，可缩小 batch/step，但应明确不能直接对照论文数字。必须锁定 StarVLA commit、benchmark evaluator 和 checkpoint selection。

### 10.2 受控消融的必要条件

比较动作头时必须固定数据样本、增强、总 step、优化器、action chunk 与 evaluator；比较预训练时必须重新初始化下游 head，避免把预训练 head 适配优势误当 backbone 表示；比较数据工程时应同时报告低、中、高数据规模，因为本文证明结论会随数据量反转。

### 10.3 完整复现预算

完整矩阵涉及 8–64×A100、多个 100K-step 训练、多 simulator 依赖、数千至上万 rollout 和真机任务。复现成本可分为：模型训练 GPU、benchmark 数据存储、模拟器评测时长、真机/相机/GELLO 或 hosted benchmark、以及版本管理。论文结构简单，但科学复现并不便宜。

---

## 11. 与 StarVLA、VLAct 的关系

三者**不是一个模型的 v1/v2/v3**：

- **StarVLA** 是框架/基础设施，定义统一 I/O、骨干—动作头契约、训练器、benchmark 与部署接口。
- **StarVLA-$\alpha$** 是使用该框架构建的受控强基线与实验方法学；它的核心是固定 Qwen3-VL + MLP，逐轴研究动作头、预训练、数据工程和 generalist evaluation。
- **VLAct** 是表示中心的 continued pre-training recipe/预训练 backbone；它接受 $\alpha$ 暴露的问题——朴素 action pretraining 可能域内增益、跨本体负迁移——并用 shallow-layer protection、caption mixing、多连续头共监督和部分统一动作空间来优化可迁移表示。

作者重叠明显：Jinhui Ye、Senqiao Yang、Yuxin Chen、Zixuan Wang、Pengguang Chen、Shu Liu、Jiaya Jia 等贯穿后两篇，且都在 StarVLA 生态中实现。这说明它们是一条研究路线的分层推进，但研究对象分别是**基础设施、受控基线、预训练表示**，不能在综述中写成同一 checkpoint 的连续升级。

---

## 12. 个人启发与后续问题

### 12.1 强 baseline 是新方法的门槛

若新动作头只比 MLP 高 1 点，却同时用了更多预训练和特殊数据工程，其创新归因并不可靠。StarVLA-$\alpha$ 提供的意义是把“强 VLM + 简单连续头”设为默认对照，让复杂度必须用受控边际收益来证明。

### 12.2 generalist 优化可能比模型扩大更重要

4B→8B 收益不足 1 点，而 batch diversity 带来更大提升，提示多本体训练的关键可能是 sampling、梯度冲突与表示共享，而非单纯参数量。后续应研究分层 sampler、task-balanced gradient、在线难例重采样以及 batch diversity 与总 token budget 的解耦。

### 12.3 最值得补充的实验

1. 多 seed 与置信区间，尤其是连续头间 0–2 点差异；
2. 高频、动态和多解任务，检验 MLP point estimate 的边界；
3. 小算力 LoRA/部分冻结版本，评估“简洁”是否能转化为可及性；
4. 将 simple padding、部分统一动作空间、embodiment-specific head 在更多机器人上做规模曲线；
5. 对负迁移做 representation probing，而不只看最终 success；
6. 分别报告模型推理延迟、动作平滑性、错误恢复和安全停机。

### 12.4 最终评价

这篇论文最强的地方不是某一项榜单，而是以负结果和反直觉结果纠正 VLA 研究习惯：复杂连续头没有稳定压倒 MLP，海量异域动作预训练可能伤害新本体，工程技巧的收益随数据规模减弱，强初始化与 batch diversity 反而更关键。其最弱之处是缺少系统统计显著性、研究范围偏桌面 manipulation、算力仍很大。作为“后续新方法必须击败的默认基线”，它非常有价值；作为对所有机器人控制范式的最终答案，则证据不足。

---

## 13. 材料来源

- 官方论文：<https://arxiv.org/abs/2604.11757>
- 官方项目页：<https://starvla.github.io>
- 官方代码：<https://github.com/starVLA/starVLA>
- 本地来源、页码与原图追溯：`assets/StarVLA-alpha - 降低VLA系统复杂度/sources.md`
- 本笔记数字均依据本地官方 PDF 及官方 LaTeX 源码核对；论文未披露或未做的实验已明确标注。
