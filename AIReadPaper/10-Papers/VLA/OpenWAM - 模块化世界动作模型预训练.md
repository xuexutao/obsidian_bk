---
type: paper-note
status: complete
domain: VLA
paper: "OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining"
year: 2026
arxiv: "2609.07398"
doi: "10.48550/arXiv.2609.07398"
source: "https://arxiv.org/abs/2609.07398"
project: "https://openwam-official.github.io/"
code: "https://github.com/OpenWAM-Official/OpenWAM"
model: "https://huggingface.co/OpenWAM"
tags:
  - VLA
  - WAM
  - World-Action-Model
  - Video-Generation
  - Robot-Learning
  - Embodied-Pretraining
  - Flow-Matching
created: 2026-10-15
updated: 2026-10-15
---

# OpenWAM：模块化世界动作模型预训练的系统性探索

## 基本信息

- **论文标题**：OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining
- **作者**：Yuran Wang、Siqiao Huang、Mingleyang Li、Chenhao Zhang、Jiaqi Liang 等；来自新加坡国立大学、清华大学、北京大学、中国香港大学、浙江大学、中国香港中文大学、上海交通大学
- **arXiv**：[2609.07398](https://arxiv.org/abs/2609.07398)，v1，2026-09-07
- **项目页**：https://openwam-official.github.io/
- **代码**：https://github.com/OpenWAM-Official/OpenWAM
- **模型与数据**：https://huggingface.co/OpenWAM
- **研究对象**：World–Action Model（WAM），即从视频生成模型继承世界动态先验，再通过具身数据把这种先验转化为可执行动作的机器人策略模型
- **核心规模**：OpenWAM-α 使用 518.5M 帧、6,369 小时数据预训练；世界流采用 Wan2.2-TI2V-5B，动作流采用独立的 1B ActionDiT；预训练使用 128 张 NVIDIA H200，约 7 天

## 一句话结论

OpenWAM 最重要的价值不是“又做了一个更大的机器人模型”，而是把 WAM 中原本纠缠在一起的**视频骨干、视觉表征、动作容量、跨模态信息流、去噪路径和数据配方**拆成可控变量，再用连续的受控实验得到一条可复现的设计链：**强视频生成骨干 + 紧凑且信息丰富的视觉 latent + 独立动作容量 + 训练期 world-to-action 信息流 + 推理期同步联合去噪 + ego/robot 一阶段联合预训练**。最终模型在多数仿真和真机任务中处于第一梯队，但 LIBERO-Plus 的明显失利也揭示了长时域像素预测 WAM 对相机与噪声扰动的结构性脆弱。

> **我的总体评价：★★★★★。** 这篇论文的高价值主要来自“研究方法论”：它不回避失败结果，也不把所有榜单差异都包装成模型胜利，而是尝试解释何种知识被继承、世界与动作怎样耦合、预训练收益来自哪里。需要保持克制的是，若干小于 1 个百分点的结论没有多随机种子或置信区间支持，跨论文 VLA/WAM 对比也仍受数据规模与训练配方混杂。

---

## 1. 研究背景与问题定义

### 1.1 从行为克隆、VLA 到 WAM

机器人策略学习长期受制于一个不对称事实：**世界变化的视频很多，带可执行动作标签的机器人轨迹很少**。ACT、Diffusion Policy 等方法直接从机器人示范同时学习视觉规律和控制规律，动作监督扎实，但世界覆盖面受限；RT-2、OpenVLA、π0 一类 VLA 则把 VLM 的语义和语言知识迁移到动作生成，解决了开放词汇理解与任务条件化，却不显式建模“世界下一步会怎样变化”。

WAM 的出发点不同：视频生成模型已经在大规模视频上学习了时序变化、物体运动和视觉动态，因此可以把视频生成先验视为一种可迁移的“世界知识”。机器人学习再负责补上动作与世界变化之间的因果接口。于是，WAM 的核心假设不是简单地“把视频模型当视觉编码器”，而是：

1. 世界预测可以为动作生成提供状态、动态和可能未来；
2. 动作学习反过来可以让世界模型聚焦于与控制有关的变化；
3. 两者应在同一个生成过程里形成协同，而不是视频预测完再外挂一个动作头。

但此前的 DreamZero、Cosmos Policy、Fast-WAM、LingBot-VA、Motus 等系统通常把视频骨干、视觉 latent、动作模块、attention 结构、训练数据和推理日程绑定在一起。模型 A 比模型 B 好时，很难知道到底是架构、数据、表征还是推理路径造成的。OpenWAM 因而把问题从“哪个模型最好”改写为“**哪些设计变量真正决定 WAM 的知识继承、协同和扩展**”。

### 1.2 三个核心科学问题

论文把全文组织成三个递进问题：

1. **What to inherit：WAM 应从上游继承什么世界知识？** 需要区分视频生成骨干中的动态先验和视觉编码器诱导的 latent 表征先验。
2. **How to interact：世界学习与动作学习怎样产生协同？** 需要分别判断动作专属容量、训练期信息流和推理期去噪路径。
3. **How to scale：这种协同怎样跨域扩展？** 需要解释 robot data 与 egocentric human video 的作用分工，以及预训练是否改变最优交互方式。

这三个问题不是并列的超参数搜索，而是一条累积证据链：前一阶段选择的默认配置成为下一阶段的控制条件，最终组合成 OpenWAM-α。

### 1.3 核心贡献

1. **OpenWAM-Infra**：把 WAM 形式化为可组合的视觉编码器 $E$、流骨干集合 $S$、可见性掩码 $M$ 和组合规则 $C$，并统一训练、部署、推理和评测接口。
2. **OpenWAM-Study**：在统一框架内连续比较 5 个视频骨干、4 类视觉编码器、6 类架构、4 种 attention mask、3 类去噪日程和多种预训练配方。
3. **OpenWAM-α**：将受控研究得到的设计原则放大到 6,369 小时多域数据，形成可公开复现的预训练 WAM。
4. **跨范式经验结论**：WAM 在 ID 拟合上往往更强，VLA 在 OOD 泛化上往往更稳；二者都可被足够丰富的数据补偿，因此“VLA 或 WAM 谁绝对占优”并没有被实验支持。
5. **开放研究栈**：发布基础设施、评测协议、模型权重、数据配方和 46 个 checkpoint，使论文不仅是一次结果报告，也是一套研究平台。

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/01-overview.png)

> **图 1：OpenWAM 的三层结构。** 左侧 OpenWAM-Infra 负责把变量拆开，中间 OpenWAM-Study 负责受控实验，右侧 OpenWAM-α 负责把实验结论放大到多域预训练。论文的逻辑不是先造模型再补消融，而是先建立实验系统、再形成配方、最后扩展模型。  
> 来源：论文 Figure 1，第 1 页，[arXiv](https://arxiv.org/abs/2609.07398)。

---

## 2. 任务定义、输入输出与统一目标

### 2.1 样本与模型输入输出

一个训练样本写为 $(\ell,o_{1:T},a_{1:H},q,m)$：

- $\ell$：自然语言任务指令；
- $o_{1:T}$：包含当前帧与未来帧的视频窗口；
- $a_{1:H}$：长度为 $H$ 的动作块；
- $q$：可选的本体状态 proprioception；
- $m$：逐维 validity mask，用于标记某个 embodiment 实际占用哪些动作维度。

视觉编码器把视频窗口编码为 latent $z$，语言和本体状态组成上下文 $c$。世界流预测未来视觉 latent 的去噪速度，动作流预测动作块的去噪速度。推理时真正执行的是动作，未来视频不需要解码为像素；世界流的作用是以生成式中间监督和共享注意力为动作流提供动态信息。

### 2.2 联合 flow matching 目标

论文采用 $t=0$ 为纯噪声、$t=1$ 为干净数据的约定。视频和动作分别拥有独立时间步 $t_v,t_a$：

$$z_{t_v}=t_vz+(1-t_v)\epsilon_v,\qquad a_{t_a}=t_aa+(1-t_a)\epsilon_a$$

模型在一次联合前向中预测两个速度场：

$$(\hat v_z,\hat v_a)=v_\theta(z_{t_v},a_{t_a},t_v,t_a,c)$$

总损失为：

$$\mathcal L=\lambda_v\mathbb E_{t_v,\epsilon_v}\left[w(t_v)\|\hat v_z-(z-\epsilon_v)\|_2^2\right]+\lambda_a\mathbb E_{t_a,\epsilon_a}\left[w(t_a)\|m\odot(\hat v_a-(a-\epsilon_a))\|_2^2\right]$$

这条公式有三层关键含义：

1. **世界和动作不是先后训练，而是在一次前向里共同优化。** $\lambda_v$ 与 $\lambda_a$ 控制两类监督的权重，OpenWAM-α 最终取 $1:1$。
2. **两个时间步独立采样。** 训练覆盖整个二维噪声平面 $(t_v,t_a)$，所以同步、视频领先、动作领先等推理轨迹都不会落到完全未见的噪声组合。
3. **动作 mask 解决跨 embodiment 异构。** 未被当前机器人使用的 80-D 槽位不产生梯度，从而允许不同动作宽度共用一个输出头。

### 2.3 可见性掩码的形式化

视频与动作间的信息流写成分块 attention mask：

$$M=\begin{pmatrix}M_{V\leftarrow V}&M_{V\leftarrow A}\\M_{A\leftarrow V}&M_{A\leftarrow A}\end{pmatrix}$$

其中 $M_{X\leftarrow Y}$ 表示模态 $X$ 是否能读取模态 $Y$。论文固定：

- $M_{V\leftarrow V}$ 为 first-frame causal attention：带噪未来帧可看当前干净帧和彼此，干净当前帧只看自身；
- $M_{A\leftarrow A}$ 为动作块内双向 attention；
- 两个跨模态块决定 Isolated、Video Sees Action、Action Sees Video、Mutual 四种模式。

这里必须注意方向：**Action Sees Video 才是世界到动作的信息流**。实验也表明真正必要的是 $M_{A\leftarrow V}$，而不是让视频读取动作。

### 2.4 统一 80-D 动作空间

为了让不同机器人共用动作头，OpenWAM 定义 $u\in\mathbb R^{80}$：

- 左臂 34-D；
- 右臂 34-D；
- 每个 34-D 块包含 EEF 位置 3-D、6D rotation 6-D、gripper 1-D、dexterous hand 24-D；
- 另有 12-D 保留给移动底盘等 embodiment-specific 通道。

每个数据集提供本地动作维到统一槽位的索引映射 $\pi$：

$$u=\operatorname{Scatter}_\pi(\operatorname{Norm}(a)),\qquad a=\operatorname{Norm}^{-1}(\operatorname{Gather}_\pi(u))$$

它的优点是共享结构落在固定坐标上，便于跨 embodiment 迁移；局限是“固定槽位语义”仍是人工设计，不是从数据中自动发现的动作结构，而且 80-D 中大量维度在单个样本上被 mask，参数利用率取决于混合数据是否足够均衡。

---

## 3. OpenWAM-Infra：把 WAM 变成可控实验系统

### 3.1 组合形式 $C(E,S,M)$

OpenWAM 把具体架构写成 $C(E,S,M)$：

- $E$：视觉编码器，决定世界流要预测什么 latent；
- $S$：流骨干集合，包括世界流 $W$、动作流 $A$，可选理解流 $U$；
- $M$：模态之间的可见性；
- $C$：组合规则，决定这些流在何处、用何种 attention 相互作用。

组合规则自身不带参数，学习容量都属于各自骨干。每个骨干统一分成 `prepare → per-layer block step → finalize`；若需要跨流 attention，则把 block 拆成 pre-attention 与 post-attention，在中间替换联合注意力计算。这样可以在不改 backbone 内部实现的前提下比较架构。

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/02-infrastructure.png)

> **图 2：OpenWAM-Infra 的模型组合空间。** 上部是视觉编码器、流骨干和可见性 mask，底部是 Single / Dual / Tri-System 六类架构。重要的是这些选择彼此正交配置，而不是每种模型各写一套 trainer 和 server。  
> 来源：论文 Figure 2，第 5 页，[arXiv](https://arxiv.org/abs/2609.07398)。

### 3.2 视觉编码器

框架支持两类冻结编码器：

- **重建式**：Wan2.2-VAE、FLUX.2-VAE，目标是保留可重建像素信息；
- **表征式**：DINOv3、V-JEPA 2.1，目标是学习语义与时序结构；可增加 S-VAE 把 768/1024-D 高维特征压到 48-D。

视觉编码器冻结意味着比较主要落在 latent 空间属性上，但 S-VAE 仍是额外训练模块，因而“encoder 先验”和“适配器容量”没有被完全分开。

### 3.3 六类架构及其发展关系

| 系统 | 变体 | 世界/动作参数关系 | 交互机制 | 对应的算法思想 |
|---|---|---|---|---|
| Single-System | Vanilla | 视频与动作共享同一 Transformer | action token 与 video token 直接拼接 | 最小改造视频 DiT，类似把动作当新 token 类型 |
| Single-System | MoE | self-attention 共享，动作走独立 FFN expert | hard routing | 用局部专属容量缓解完全共享冲突 |
| Dual-System | Joint Self-Attention | Video DiT 与 ActionDiT 参数分离 | bridge layer 合并 token，做联合 self-attention | 世界和动作有独立容量，同时进行 token-level 双向交互 |
| Dual-System | Joint Cross-Attention | 参数分离 | Action query 读取 video features | 明确的 video-to-action 通道；可选择是否让动作梯度更新视频流 |
| Dual-System | IDM | 参数分离 | 先生成视频，再用 inverse dynamics 预测动作 | 典型 planning-then-control / two-stage 假设 |
| Tri-System | Joint Self-Attention | 增加独立 VLM understanding stream | 三流联合注意力；理解流只读 | 把语义理解、世界生成、动作生成拆成三专家，接近 Motus 路线 |

从算法演进看，Single-System 代表“尽量复用视频生成器”，Dual-System 代表“给控制单独容量但保留世界耦合”，Tri-System 则代表统一多模态模型中常见的专家化趋势。OpenWAM-α 最终选 Dual-System，不是因为 Tri-System 分数更低，而是 Tri-System 只高 0.24 pp，却引入 VLM 理解流这一额外变量；为了专注研究 world–action 本身，作者选择更可解释的折中。

### 3.4 统一训练、checkpoint 与部署

训练系统支持 DeepSpeed ZeRO-1/2、bf16、gradient checkpointing、CPU activation/optimizer offload、fine-tuning 和精确 resume。每个 checkpoint 自带：

1. 模型权重；
2. 完整解析后的模块配置与 tokenizer 等依赖；
3. 动作归一化统计。

部署时缺失 normalization record 会直接报错，避免“训练与评测静默使用不同动作缩放”的常见复现问题。这个设计看似工程细节，但对机器人 benchmark 很关键：动作单位错误可以让任何模型直接失效。

### 3.5 同步、视频领先与动作领先去噪

推理日程是二维噪声平面中的路径 $\tau=\{(t_v^i,t_a^i)\}_{i=0}^N$。同步去噪沿对角线前进；异步日程由两类函数控制：

$$f_\alpha(s)=\frac{\alpha s}{1+(\alpha-1)s},\qquad h_o(s)=\max\left\{\frac{s-o}{1-o},0\right\}$$

- variance shift $f_\alpha$ 让领先模态更早接近干净状态；
- linear offset $h_o$ 让滞后模态在一段时间内保持纯噪声；
- $(\alpha,o)=(1,0)$ 退化为同步日程。

IDM 直觉认为“先想象未来，再反推出动作”应有帮助；但 OpenWAM 的实验更支持另一种解释：视频预测主要作为**表征学习与联合监督**增强动作，而不是在测试时必须完成一次显式 planning rollout。

### 3.6 推理模式与速度

框架支持同步和异步 action buffer。OpenWAM-α 的正式评测统一使用同步执行：buffer 为空时暂停，重新预测动作块后再执行，因此结果没有被异步流水线掩盖。

部署侧包含 prompt embedding cache、跳过 VAE 解码、`torch.compile` + CUDA Graph、DiT velocity cache。Wan2.2-TI2V-5B 下，不同架构在 RTX 5090 上可获得约 2.01–2.63× 加速；OpenWAM-α 的 10-step 联合去噪约 170 ms/动作块。这个速度是动作块级实时，不等价于每控制步 170 ms，因为一次动作块包含多个动作；论文没有系统报告动态环境下同步暂停对闭环稳定性的影响。

### 3.7 统一评测协议

八个仿真 benchmark 和真机都作为 WebSocket thin client：客户端只传图像、语言与状态，服务端完成图像拼接、缺失相机黑帧填充、proprio/action 映射、归一化、推理和反归一化。这样 benchmark 不需要 import 训练代码，减少不同模型因评测包装器差异导致的混杂。

---

## 4. OpenWAM-Study：逐项解析受控实验

### 4.1 共同实验环境与控制逻辑

设计研究主要在 RoboTwin2.0 上进行，指标为成功率：

- **RoboTwin2.0-Full**：2,500 条 clean + 25,000 条 heavy randomization 训练，分别在 clean/randomized 测试，更接近 ID 多任务拟合；
- **RoboTwin2.0-Clean2Random**：只在 clean 训练，在 clean 测试为 ID，在 randomized 测试为 OOD。

这套选择的优点是能把“拟合能力”和“场景迁移”拆开；风险是设计原则主要在一个 benchmark 上发现，可能存在对 RoboTwin 的 recipe overfitting。后续多 benchmark 结果能够验证外部有效性，却不是所有消融的严格重复。

### 4.2 Q1-A：视频生成骨干——强先验确实有用，但不是纯规模律

| 视频骨干 | 参数规模 | RoboTwin2.0-Full 平均成功率 |
|---|---:|---:|
| Wan2.1-VACE | 1.3B | 90.14 |
| Cosmos-Predict2.5 | 2B | 91.64 |
| Wan2.2-TI2V | 5B | 92.39 |
| Wan2.1-I2V | 14B | 93.79 |

性能随骨干规模单调上升，说明视频生成先验不是无关初始化。14B 比 5B 高 1.40 pp，但参数约为 3 倍；作者因此选 5B 作为性能—效率折中。

**需要精确理解这组实验的边界：** 四个 backbone 不只参数量不同，架构、数据与预训练目标也不同，所以它证明的是“backbone 选择是重要变量”，不是“参数越大必然线性提升”。论文在正文中主动承认这一点，论证较克制。缺失项包括 FLOPs 配平、相同模型族内部 scaling 和多随机种子。

### 4.3 Q1-B：视觉 latent——关键是紧凑且信息丰富

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/03-visual-representation.png)

> **图 7：视觉表征对比。** 语义 encoder 直接输出高维 latent 时明显较差，但经 S-VAE 压缩到 48-D 后，DINOv3 达到 90.18，几乎追平原生 Wan2.2-VAE 的 90.30。  
> 来源：论文 Figure 7，第 12 页，[arXiv](https://arxiv.org/abs/2609.07398)。

| 编码器 | 类型 | 适配 | 平均成功率 | 相对同类无 S-VAE |
|---|---|---|---:|---:|
| DINOv3 | representation | 无 | 76.42 | — |
| V-JEPA 2.1 | representation | 无 | 80.91 | — |
| FLUX.2-VAE | reconstructive | 原生 latent | 84.26 | — |
| V-JEPA 2.1 | representation | S-VAE→48-D | 88.46 | +7.55 pp |
| DINOv3 | representation | S-VAE→48-D | 90.18 | +13.76 pp |
| Wan2.2-VAE | reconstructive | 原生 48-D + 原生时序压缩 | 90.30 | — |

这组实验的结论不是“重建特征胜过语义特征”，而是：DiT 是否容易建模该 latent，取决于**维度、token 数、时序压缩和信息密度**。DINOv3 的 768-D、V-JEPA 的 1024-D 特征直接送入面向 VAE latent 设计的 DiT 会发生架构错配；S-VAE 把它们压到 48-D 后，大幅恢复性能。

作者把 Wan2.2-VAE 最后 0.12 pp 的优势解释为两点：它与 Wan DiT 原生匹配，且 4× 时间压缩是学出来的，而 DINO/V-JEPA 只是每四帧平均。这个解释合理，但 0.12 pp 没有误差条，不应被理解成稳定优胜。

**批判性判断：** S-VAE 同时增加了可训练适配器和新的压缩目标，因此这不是纯粹的“representation quality”比较。更严格的实验应在相同 adapter 参数量、相同 token 数、相同时序压缩下比较不同 encoder。不过，DINO + S-VAE 从 76.42 到 90.18 的巨大跃升已经足以说明：WAM 的瓶颈很可能不在语义特征本身，而在 latent 与生成骨干之间的表示接口。

### 4.4 Q2-A：动作专属容量——独立 ActionDiT 比共享 token 更重要

| System | Variant | Clean | Randomized | Average |
|---|---|---:|---:|---:|
| Single | Vanilla | 85.20 | 85.80 | 85.50 |
| Single | MoE | 86.22 | 83.04 | 84.63 |
| Dual | Joint Self-Attention | 92.34 | 92.38 | 92.36 |
| Dual | Joint Cross-Attention | 87.86 | 88.64 | 88.25 |
| Dual | Detached Cross-Attention | 92.06 | 91.64 | 91.85 |
| Dual | IDM | 87.76 | 88.14 | 87.95 |
| Tri | Joint Self-Attention | 92.84 | 92.36 | **92.60** |

Single Vanilla 到 Dual Joint Self-Attention 提升 6.86 pp，说明动作 token 仅复用视频 DiT 的共享容量不够。Single MoE 反而从 85.50 降到 84.63，说明“给动作增加一个 FFN expert”也不足以替代完整独立动作流；真正有效的组合是**参数分离 + 逐层联合注意力**。

Detached Cross-Attention 达到 91.85，而 end-to-end Cross-Attention 只有 88.25，是很值得注意的结果：如果 action learning 的梯度通过单向 cross-attention 强行更新 video stream，可能破坏已有世界先验；detach 后保留 world-to-action 特征读取，又减少反向干扰，因此大幅恢复。Joint Self-Attention 则通过对称的联合 attention 让两流共同优化，达到 92.36。

Tri-System 的 92.60 为最高，但仅比 Dual Self-Attention 高 0.24 pp。论文选择 Dual 作为后续默认，是为了避免 VLM understanding stream 混入“世界—动作协同”分析。这个决定体现了实验设计优先于追榜单。

### 4.5 Q2-B：训练期信息流——动作必须看见世界

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/04-information-flow-and-denoising.png)

> **图 8、表 2 与图 9：训练期可见性和推理期去噪路径。** 上半部分显示只要动作流能读取世界流，成功率就从约 87.5% 跃升到约 92.3%；下半部分显示各种视频领先或动作领先日程都没有超过同步去噪。  
> 来源：论文 Figure 8、Table 2、Figure 9，第 14 页，[arXiv](https://arxiv.org/abs/2609.07398)。

| Mask | Clean | Randomized | Average |
|---|---:|---:|---:|
| Isolated | 88.08 | 86.74 | 87.41 |
| Video Sees Action | 87.92 | 87.34 | 87.63 |
| Action Sees Video | **92.98** | **91.80** | **92.39** |
| Mutual | 92.50 | **91.80** | 92.15 |

分组非常清楚：动作不能读取视频时平均约 87.5；动作能读取视频时平均约 92.3，差约 4.8–5.0 pp。反向的 action-to-video 通道在 from-scratch 场景中几乎没有贡献：Mutual 甚至比 Action Sees Video 低 0.24 pp。

因此，WAM 的“世界—动作联合预测”不能只看输出是否同时包含视频和动作，还必须检查**动作预测路径是否真的接收到世界特征**。Isolated 也共同预测两种目标，却没有协同收益；这直接反驳了“只要加视频损失就自然形成 world-action synergy”的简单观点。

### 4.6 Q2-C：推理期同步去噪胜过显式领先

论文固定 Mutual mask，比较：

- variance shift：$\alpha\in\{4,8,16,32\}$，分别让视频或动作领先；
- linear offset：$o\in\{0.2,0.4,0.6,0.8\}$，分别延迟视频或动作；
- synchronized：两流沿对角线同步。

同步去噪取得 **93.0**。variance-shift 的所有结果为 88.5–91.9，最佳仍低 1.1 pp；linear-offset 为 91.1–92.3，最佳低 0.7 pp。没有任何异步日程超过同步。

这个结果支持：世界流不是先生成一个完整未来计划，再由 IDM 读取；更好的方式是让动作与未来世界在每个噪声层级同步细化。世界 latent 更像持续提供的预测性表征，而不是测试时必须“想象完”的显式计划。

**结论边界：** 所有模型在训练时独立采样 $t_v,t_a$，但并没有专门为某条异步轨迹训练。同步更好可能来自优化稳定和两流噪声质量匹配，不能外推为“所有 planning-then-action 推理都无效”。

### 4.7 Q3-A：预训练主要扩展 OOD，而不是把 ID 再拟合一点

在相同 600 小时预算下比较：

- From Scratch；
- Robot Only：600h RoboCOIN；
- Ego→Robot：350h EgoDex 后 250h RoboCOIN；
- Ego+Robot：350h + 250h 一阶段联合训练。

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/05-pretraining-ood.png)

> **图 10：具身预训练主要提升 OOD。** ID 从 87.00 最多只到 88.50，而 OOD 从 14.50 提升到 26.62；混入 egocentric video 后 OOD 收益最大。  
> 来源：论文 Figure 10，第 15 页，[arXiv](https://arxiv.org/abs/2609.07398)。

| 配方 | Clean2Clean ID | 相对 scratch | Clean2Random OOD | 相对 scratch |
|---|---:|---:|---:|---:|
| From Scratch | 87.00 | — | 14.50 | — |
| Robot Only | **88.50** | +1.50 pp | 23.80 | +9.30 pp |
| Ego→Robot | 87.10 | +0.10 pp | 26.50 | +12.00 pp |
| Ego+Robot | 87.68 | +0.68 pp | **26.62** | **+12.12 pp** |

最强证据是收益量级差异：最佳 ID 只提升 1.50 pp，最佳 OOD 提升 12.12 pp，相对 scratch 为 +83.6%。Robot-only 在 ID 最强，说明带动作标签的机器人轨迹更直接地强化动作 grounding；混合 ego 数据在 OOD 更强，说明无动作的人类视频主要扩大场景、物体与交互覆盖。

Ego→Robot 与 Ego+Robot 的 OOD 只差 0.12 pp，ID 差 0.58 pp，不能声称联合训练在统计上显著更好。论文采用 co-training 的主要理由应理解为：表现至少不差，而且省去阶段切换，工程上更直接。

**潜在混杂：** 600 小时并不等于 600 小时的信息量相同；ego 和 robot 的帧率、动作密度、任务粒度都不同。混合策略同时改变了“是否含 ego”与“robot 数据从 600h 减到 250h”，因此它更像数据 recipe 比较，不是对 ego 因果贡献的完全隔离。

### 4.8 Q3-B：预训练后 Mutual 才开始占优

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/06-pretraining-visibility.png)

> **图 11：预训练改变可见性偏好。** from-scratch 时 Mutual 比 Action Sees Video 低 0.24 pp；预训练后在 Full、C2R-ID、C2R-OOD 上分别高 0.16、0.70、0.72 pp。  
> 来源：论文 Figure 11，第 16 页，[arXiv](https://arxiv.org/abs/2609.07398)。

作者的解释是：小数据下，action-to-world 通道学不到稳定对应，可能引入噪声；大规模具身预训练后，动作可以反过来帮助世界流生成机械臂运动，因此 Mutual 开始有益。这种机制解释很有启发，但数值都小于 1 pp，且没有误差条，证据强度明显弱于前面的 5 pp 信息流实验。

另一个细节是：Figure 11 的 C2R-OOD 单向基线 25.70 加 0.72 得 26.42，而 Figure 10 的 co-train OOD 为 26.62，可能来自不同运行配置、舍入或重复实验。论文未进一步解释，因此不应把 0.2 pp 级差异过度解读。

### 4.9 累积得到的设计配方

| 阶段 | 证据支持的判断 | OpenWAM-α 默认选择 |
|---|---|---|
| Inherit | 强视频骨干 + 紧凑、信息丰富的 latent 最利于迁移 | Wan2.2-TI2V-5B + Wan2.2-VAE |
| Interact | 独立动作容量、world-to-action 信息流、同步去噪最有效 | Dual Joint Self-Attention + Sync |
| Consolidate | 预训练主要改善 OOD；ego 扩覆盖，robot 保 grounding | One-stage ego+robot + Mutual |

---

## 5. OpenWAM-α：从受控原则到预训练基础模型

### 5.1 完整架构与数据流

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/07-openwam-alpha-architecture.png)

> **图 12：OpenWAM-α 总体架构。** 左上与右上是独立 Video DiT / ActionDiT，中间每层做共享注意力；左下是 518.5M 帧数据混合，中下显示训练覆盖二维噪声平面、推理走同步对角线，右下是统一 80-D 动作空间。  
> 来源：论文 Figure 12，第 17 页，[arXiv](https://arxiv.org/abs/2609.07398)。

一次前向的数据流可以逐步写成：

1. 多相机视图被拼成一张 canvas，形成当前观测 $o_1$；训练时还包含未来帧 $o_{2:T}$。
2. 冻结 Wan2.2-VAE 因果编码视频：第一帧单独编码，后续每 4 帧一组，得到 $T'=1+(T-1)/4$ 个 latent frame；当前帧 latent 始终是干净锚点。
3. 视频 latent 用 $(1,2,2)$ patch embedding 变成宽度 3072 的 world token，并使用 $(frame,height,width)$ 的 3D RoPE。
4. 每个带噪动作步线性映射为一个宽度 1024 的 action token，使用 action chunk index 的 1D RoPE。
5. 冻结 umT5 把语言编码为 4096-D context；统一 80-D proprioception 经线性投影后作为额外 context token。
6. Video DiT 与 1B ActionDiT 各有 30 个 paired layer；每一层都是 bridge layer。两流通过各自 QKV projection 映射到 24 heads × 128 dimensions 的共享 attention 空间。
7. Mutual mask 允许未来 world token 与 action token 双向读取；当前干净帧行既不读取带噪未来，也不读取动作，防止条件被噪声污染。
8. 两流分别通过 AdaLN 注入 $t_v$ 与 $t_a$，最后各自线性头输出速度场。
9. 推理用 10 步同步 Euler 更新；完成后只把动作通过 $\pi^{-1}$ 映射回机器人原生单位并执行，视频不解码。

### 5.2 训练参数与算力

| 项目 | 配置 |
|---|---|
| GPU | 16 节点 × 8 NVIDIA H200 = 128 GPU |
| 时长 | 约 7 天 |
| Optimizer | AdamW，$\beta_1=0.9,\beta_2=0.95$ |
| Global batch | 3,072 clips（每 GPU 24） |
| 学习率 | $1\times10^{-4}$，cosine，5% warmup，minimum ratio 0.01 |
| Weight decay | 0.01 |
| 训练步数 | 155,862，1 epoch |
| 精度与并行 | bf16，DeepSpeed ZeRO-2 |
| 输入 | 33 帧，video stride 4，window stride 1，384×320，多视角 |
| 增强 | ColorJitter(0.2,0.2,0.2,0.0) |
| Flow shift | video/action 均为 5.0 |
| 损失权重 | $\lambda_v=\lambda_a=1$ |

OpenWAM-α 不是低成本可从零复现的模型。论文的开放性主要体现在代码、配置、checkpoint 和评测可用，而不是普通实验室可以轻易复刻 128 H200×7 天的预训练。更实际的复现路线是下载 foundation checkpoint，在 8×80GB GPU 上做 benchmark fine-tuning；官方 README 也把 8×80GB 作为 5B 及以下架构训练的推荐配置。

### 5.3 预训练数据混合

| 来源 | 类型 | Embodiment | 原始规模 | 实际使用 | 每 epoch 占比 |
|---|---|---:|---:|---:|---:|
| 自建 Egocentric | 人类第一视角视频 | human | 744.9M 帧 / 6,897h | 155.7M / 1,442h | 30.1% |
| AgiBotWorld-Beta | 真机 | 1 | 124.5M / 2,306h | 96.9M / 1,794h | 18.6% |
| RoboCOIN | 真机 | 15 | 104.5M / 956h | 74.1M / 686h | 14.3% |
| DROID | 真机 | 1 | 46.3M / 1,285h | 36.3M / 1,007h | 7.0% |
| InternData-A1 | 仿真 | 4 | 313.7M / 2,904h | 155.5M / 1,440h | 30.0% |
| **总计** | human + robot | 21 robot + human | 1,333.9M / 14,348h | **518.5M / 6,369h** | 100% |

30% ego 数据没有 action/proprio supervision，只训练世界流；40% 真机数据提供最可信的物理与动作 grounding；30% synthetic 数据扩大单臂、双臂和环境变化。数据按实际帧数比例采样，每个保留样本每 epoch 恰好访问一次。

自建 ego 数据包含 71.6K 条 0.25–6 分钟长视频和 3,006 个日常操控任务。它带来广泛视觉和交互覆盖，但不是带动作的人类遥操作数据，因此无法直接告诉模型“机器人应如何做”。

### 5.4 数据清洗为什么重要

视觉层清洗去除解码失败、冻结/重复、纯黑白、曝光异常、模糊、跳帧以及手离开视野的 ego 片段。机器人信号清洗更严格：

1. 若 EEF state 与 action 幅度比 ≥3× 且逐轴相关 <0.5，整条 episode 删除；
2. 视频—信号错位超过 2% 帧，删除；
3. 用 state 做首尾 idle detection，运动阈值结合 p99.5 单帧 delta、平均平移 <2 cm/s、旋转 <5°/s；中间停顿不剪，避免拼接不相邻时刻；
4. state 在动而所有相机静止，视为 jitter 并裁剪；机械臂真动但相机冻结，删除 episode；
5. 若损失超过 70% 帧，或视频冻结 ≥90%，整条删除。

这部分体现了论文对大规模机器人数据的现实理解：WAM 同时学习像素未来与动作，一旦视频、state、action 时间错位，两个流会收到互相矛盾的监督，比纯 action policy 更敏感。

---

## 6. 对比对象与实验结果深度分析

### 6.1 Baseline 不是一类模型：先按机制分类

| 模型 | 范式 | 主要预测目标 / 表征 | 是否大规模具身预训练 | 与 OpenWAM 比较时最有价值的维度 |
|---|---|---|---|---|
| π0 / π0.5 | VLA | VLM 特征 + 连续 flow action | 是 | 代表 action-only 连续流策略；跨仿真与真机最常见参照 |
| OpenVLA / X-VLA / StarVLA | VLA | 语义视觉 + 动作 | OpenVLA/X-VLA 有，StarVLA 对照中无 embodied pretrain | 检验 VLA 开源路线、跨 embodiment 与模块化设计 |
| Qwen-RobotManip | VLA | 大规模 action-centric VLA | 是 | 多榜单强上界，但数据/训练规模与 OpenWAM 不配平 |
| Fast-WAM | WAM | 长时域未来像素 latent + 动作 | 无 embodied pretrain | 与 StarVLA 构成论文最接近的 from-scratch VLA/WAM 对照 |
| ImageWAM | WAM | 单个未来帧编辑 + 动作 | 无 | 用于判断长 rollout 误差是否是 OOD 脆弱来源 |
| ABot-M0.5 | WAM | 长时域像素 latent + 动作 | 是，单臂 8,000+h | 与 OpenWAM 的预测目标接近，适合分析“同类架构的数据覆盖” |
| Being-H0.7 | WAM | V-JEPA 2.1 temporal semantic latent + 动作 | 是，单臂 5,700+h | 用于分析 semantic latent 对相机/噪声扰动的鲁棒性 |
| LingBot-VA | WAM | causal world modeling + action | 是 | 真实单臂和仿真同类模型参照 |
| Motus | WAM / unified model | world、action、understanding 多专家 | 是 | Tri-System 路线代表 |

跨论文 leaderboard 不能当作严格因果实验，因为参数量、数据小时、视角、动作表示、SFT 步数、增强和推理预算均不同。**真正可信度最高的是 OpenWAM 内部消融；外部 baseline 更适合回答“OpenWAM 处于什么水平”，不适合直接回答“哪个范式本质更好”。**

### 6.2 八个仿真基准覆盖什么

- **单臂**：LIBERO、LIBERO-Plus、VLABench；
- **双臂**：RoboTwin2.0、RoboDojo；
- **移动单臂**：RoboCasa365；
- **移动双臂**：EBench；
- **灵巧手人形**：RoboCasa-GR1。

RoboTwin 同时报告 Full 和 Clean2Random，因此是八个 benchmark、九个 leaderboard。不同 benchmark 的分数定义不完全相同，不能把 99.3 与 11.92 直接横向比较。

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/08-simulation-benchmarks.png)

> **图 13：九个仿真 leaderboard 的代表性对比。** OpenWAM-α 在 LIBERO、EBench、RoboCasa-GR1、RoboCasa365、RoboDojo 和 VLABench 的图示子集中领先，但在 LIBERO-Plus 与 RoboTwin-C2R 仍落后最强 VLA。  
> 来源：论文 Figure 13，第 21 页，[arXiv](https://arxiv.org/abs/2609.07398)。

### 6.3 总体结果、排名与差距

| Benchmark | OpenWAM-α | 强基线 / 最佳展示值 | 直接解读 |
|---|---:|---:|---|
| LIBERO | 99.3 | ABot-M0.5 99.4；Qwen-RM/Being-H0.7 99.2 | 近饱和，差异缺乏辨识度 |
| RoboTwin2.0-Full | 93.60 | ABot-M0.5 94.10；Qwen-RM 93.85 | 第一梯队，距最佳 0.50 pp |
| LIBERO-Plus | 69.2 | Qwen-RM 89.0；ABot-M0.5 83.4 | 明显失败，尤其 camera/noise |
| RoboTwin2.0-C2R | 69.0 | Qwen-RM 77.1 | 平均次优但 split 方向相反：ID 最强、OOD 弱 |
| EBench | **49.4 SR / 64.7 Score** | Qwen-RM 45.6 / 60.0 | 明确 SOTA：+3.8 / +4.7 pp |
| RoboCasa-GR1 | 60.5 | PhysBrain 64.5；最佳其他 WAM LDA-1B 55.4 | 最强 WAM，距全体最佳 4.0 pp |
| RoboCasa365 | 38.2 | Xiaomi-Robotics-1 57.4；ABot-M0.5 40.4 | 并非“微小差距”，距全体最佳 19.2 pp |
| RoboDojo | 11.92 SR / 17.18 Score | DM0.5 19.34 / 24.90；最佳其他 WAM 3.83 / 7.69 | WAM 内领先很大，但整体绝对成功率低 |
| VLABench | 58.9 | Xiaomi-Robotics-1 59.1 | 距最佳 0.2 pp；ID SR 83.4 很强 |

### 6.4 逐 benchmark 解读

#### 6.4.1 LIBERO：榜单已接近饱和

OpenWAM 平均 99.3，四个 suite 分别为 Spatial 99.6、Object 99.6、Goal 99.8、Long 98.2。相对最强 ABot-M0.5 的 99.4 只低 0.1 pp，相对 Qwen-RM / Being-H0.7 的 99.2 只高 0.1 pp。每 0.1 pp 都可能对应极少数 episode，不能据此判断稳定优胜。更有意义的是 Long 仍达到 98.2，说明 33 帧输入和 action chunk 设计没有在长程 suite 上明显崩溃。

#### 6.4.2 RoboTwin2.0-Full：预训练 WAM 与强 VLA 基本持平

OpenWAM Clean 93.74、Randomized 93.46、平均 93.60；ABot-M0.5 平均 94.10，Qwen-RM 93.85，Galaxea G0.5 93.25。由于训练集本身包含大量 randomization，这里的 Randomized 仍接近训练分布，主要检验多任务拟合而非真正 OOD。OpenWAM 两个 split 非常接近，说明在充分随机化训练后模型没有明显偏向 clean。

#### 6.4.3 RoboTwin2.0-Clean2Random：最能说明“WAM 拟合强、泛化弱”

OpenWAM 在 Clean 上达到 **89.4**，高于 Qwen-RM 的 84.7，是全表最高；但 Randomized 只有 **48.7**，比 Qwen-RM 69.4 低 20.7 pp。平均 69.0 低 Qwen 8.1 pp，但比次强 WAM 4D-WAM 的 61.7 高 7.3 pp。

这说明平均分会掩盖机制：OpenWAM 并不是整体能力不足，而是对训练视觉分布拟合非常强、对未见随机化迁移不足。它与 LIBERO-Plus 的 camera/noise 失败共同指向 pixel-level future prediction 的分布敏感性。

#### 6.4.4 EBench：最清楚的正向 SOTA

EBench 是移动双臂任务。OpenWAM 总体 49.4 SR / 64.7 Score，超过 Qwen-RM 的 45.6 / 60.0；相对提升分别为 3.8 pp（约 8.3%）和 4.7 pp（约 7.8%）。Long Horizon 分项达到 44.3 / 72.6，也高于 Qwen 的 29.9 / 55.0。由于差距明显、任务复杂、指标同时提升，这是比 LIBERO 0.1 pp 更有说服力的结果。

#### 6.4.5 RoboDojo：WAM 内部领先，但离真正通用还远

OpenWAM 平均 11.92 SR / 17.18 Score，高于最佳其他 WAM X-WAM 的 3.83 / 7.69，却低于最佳 VLA DM0.5 的 19.34 / 24.90。Gen-Std SR 25.56 尚可，Open SR 只有 1.08。这里“相对提升 2–3 倍”容易显得很亮眼，但绝对成功率仍低，说明开放任务和强泛化仍未解决。

#### 6.4.6 RoboCasa365：论文措辞需要更克制

OpenWAM 平均 38.2，Atomic 69.7、Compositional-Seen 32.1、Compositional-Unseen 8.9。它接近 ABot-M0.5 的 40.4，但远低于 Xiaomi-Robotics-1 的 57.4。论文把它归入“距最佳模型仅很小差距”的 top tier 并不准确；19.2 pp 是显著差距。更合理的说法是“WAM 组内有竞争力，但在移动单臂 compositional generalization 上仍落后超大规模 VLA”。

#### 6.4.7 RoboCasa-GR1 与 VLABench：接近强模型，但不是绝对第一

RoboCasa-GR1 上 OpenWAM 60.5，是最强 WAM，低 PhysBrain 64.5；VLABench 总体 SR 58.9，低 Xiaomi 59.1 仅 0.2。值得注意的是 VLABench In-dist SR 83.4，高于 Xiaomi 75.6，仍呈现 ID 强、综合泛化后接近但不完全领先的模式。

### 6.5 LIBERO-Plus：全篇最值得细读的失败分析

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/09-libero-plus-and-data.png)

> **表 5 与图 15：LIBERO-Plus 分项及单臂预训练数据。** OpenWAM 在 Light 97.0、Language 88.0 上并不差，但 Camera 33.8、Noise 39.8 明显崩溃；其单臂数据只有约 1,100+h，且来源比 ABot-M0.5 与 Being-H0.7 单一。  
> 来源：论文 Table 5、Figure 15，第 22 页，[arXiv](https://arxiv.org/abs/2609.07398)。

| 模型 | 类型/关键设计 | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-RobotManip | VLA，大规模预训练 | 87.2 | 75.5 | 85.6 | 96.6 | 97.7 | 97.7 | 87.3 | **89.0** |
| ACoT-VLA | VLA，action CoT | 72.6 | 82.6 | 87.5 | 97.7 | 96.5 | 87.8 | 88.1 | 86.6 |
| π0.5 | VLA，flow action | 78.4 | 73.6 | 80.8 | 96.2 | 94.1 | 89.0 | 84.5 | 84.4 |
| ABot-M0.5 | WAM，像素 latent，8,000+h 单臂 | 70.5 | **87.4** | 88.6 | 94.0 | 89.7 | 75.5 | 85.2 | 83.4 |
| ImageWAM | WAM，单帧编辑 | 80.8 | 50.3 | **91.4** | **98.1** | 85.5 | 93.8 | 80.5 | 83.1 |
| Being-H0.7 | WAM，V-JEPA latent，5,700+h 单臂 | 82.0 | 59.0 | 82.8 | 97.8 | 90.0 | 93.5 | **88.5** | 82.1 |
| Fast-WAM | WAM，长时像素 latent，无预训练 | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| **OpenWAM-α** | WAM，长时像素 latent，约 1,100+h 单臂 | **33.8** | 76.1 | 88.0 | 97.0 | 87.1 | **39.8** | 77.5 | **69.2** |

作者从两个方向解释失败：

**数据方向。** OpenWAM 的单臂来源主要是固定平台/固定相机的 DROID 和 InternData-A1 单臂子集，约 1,100+h；ABot-M0.5 超过 8,000h，Being-H0.7 超过 5,700h，且涵盖更多 embodiment、场景和视角。因此 OpenWAM 虽然总数据达 6,369h，但与 LIBERO-Plus 最相关的单臂数据并不充分。总小时数不能替代 embodiment-specific coverage。

**架构方向。** Fast-WAM、ABot-M0.5、OpenWAM 都预测跨时间的 pixel-level video latent，三者在 Camera 与 Noise 上都有共同弱点。ImageWAM 只预测当前图像的一次未来编辑，不做长 rollout，因此不累积逐帧误差；Being-H0.7 预测 V-JEPA semantic latent，对像素噪声和视角变化更稳。这与 Section 4.1.2 的“紧凑且鲁棒的 representation latent”呼应。

我的判断是，这个解释有较强相关证据，但还不是严格因果证明，因为数据规模和视觉表征同时变化。最关键的缺失实验有两个：

1. 在完全相同数据下比较 Wan2.2-VAE、V-JEPA+S-VAE 和单帧 ImageWAM；
2. 对 OpenWAM 固定架构，只扩展单臂数据的小时数、相机多样性和噪声增强，画出 scaling curve。

尽管如此，论文愿意把 69.2 的失败拆开分析，而不是只展示有利平均分，是本篇实验部分写得好的重要原因。

### 6.6 VLA 与 WAM：不是谁获胜，而是归纳偏置不同

![](assets/OpenWAM%20-%20模块化世界动作模型预训练/figures/10-vla-vs-wam.png)

> **图 16：VLA/WAM 的 ID 与 OOD 对比。** 无 embodied pretraining 的 Fast-WAM 在 LIBERO、RoboTwin Clean/Full 上高于 StarVLA，但在 LIBERO-Plus 和 C2R-Random 上反转；预训练模型也呈现相似趋势。  
> 来源：论文 Figure 16，第 24 页，[arXiv](https://arxiv.org/abs/2609.07398)。

最接近公平的对照是都没有 embodied pretraining 的 Fast-WAM 与 StarVLA：

| Split | Fast-WAM | StarVLA | 更优范式 |
|---|---:|---:|---|
| LIBERO Avg（ID） | 97.6 | 96.6 | WAM +1.0 |
| RoboTwin C2R Clean（ID） | 77.8 | 46.5 | WAM +31.3 |
| RoboTwin Full Avg（近 ID） | 91.85 | 88.25 | WAM +3.60 |
| LIBERO-Plus Avg（OOD） | 51.5 | 74.1 | VLA +22.6 |
| RoboTwin C2R Randomized（OOD） | 1.9 | 3.2 | VLA +1.3，二者都崩溃 |

论文的机制解释是：WAM 多了 future video latent 监督，优化时信息更密，因此更容易拟合训练分布；但长时未来预测在 OOD 下会累计视觉误差，action-only VLA 不承担这个负担，所以泛化更稳。

这个方向性结论与多组结果一致，但“视频 latent 导致差异”的因果归因仍不完整。Fast-WAM 与 StarVLA 不是只相差一个 loss 的同骨干模型；预训练模型之间的数据规模差异更大。例如 Xiaomi-Robotics-1 使用超过 100k 小时真机轨迹，不能与 6,369h OpenWAM 直接归因到 VLA/WAM 范式。

更稳妥的结论是：

- **未来 latent 监督是一种强 ID 拟合偏置；**
- **长时像素预测是一种 OOD 风险；**
- **丰富且接近目标分布的数据可以补偿两边缺陷；**
- 真正值得发展的可能是把 VLA 的稳健语义表征与 WAM 的预测性中间监督结合，而不是在两者之间二选一。

### 6.7 真机结果

#### 6.7.1 单臂 Franka

每个任务用 100 条 task-specific 示范微调，再做 20 次真机试验：

| 模型 | 平均成功率 |
|---|---:|
| π0.5 | 55.0%（66/120） |
| LingBot-VA | 77.5%（93/120） |
| **OpenWAM-α** | **82.5%（99/120）** |

OpenWAM 在辣椒入抽屉、Jenga 入抽屉均为 100%，总体比 LingBot 高 5 pp、比 π0.5 高 27.5 pp。固定场景、每任务 20 次使置信区间仍较宽，但至少说明仿真优势能够迁移到物理控制。

#### 6.7.2 双臂 RoboDojo-Real

跨 ARX X5、Piper、Piper X 三种 embodiment、18 个任务：

- OpenWAM-α：**37.6 Score / 24.4 SR**；
- π0.5：22.9 / 12.8；
- 相对差值：+14.7 Score、+11.6 SR，分别约 +64% 与 +91%。

但任务间差异很大，OpenWAM 在部分任务仍为 0。总体均值明显领先不等于已经实现稳定通用双臂策略。

#### 6.7.3 未见过的灵巧手 embodiment

Wuji hand（21 DoF）+ Tianji arm（6-DoF EEF pose）及其动作空间从未出现在 OpenWAM 预训练混合中。经过任务微调后，OpenWAM 在 Stack Toy Tower、Collect Shuttlecocks、Put Away Clothes、Twist off Bottle Cap 的所有 ID/OOD 单元都超过 π0.5。例如：

- Put Away Clothes OOD：91.7 Score / 85% SR，π0.5 为 75.0 / 45%；
- Bottle Cap OOD：90.0 / 80%，π0.5 为 65.0 / 30%；
- Stack Toy Tower OOD：56.7 / 30%，π0.5 为 40.0 / 10%。

这是很有价值的**可适配性**证据，但不是 zero-shot：模型经过下游 SFT，因此证明的是预训练初始化能迁移到新 embodiment，而不是无需示范直接控制新机器人。

---

## 7. 实验证据强度分级

### 7.1 证据较强的结论

1. **动作流读取世界流是必要的。** 是否有 $A\leftarrow V$ 带来约 5 pp 的清晰分组差异。
2. **视觉 latent 的紧凑性极其重要。** DINO + S-VAE 提升 13.76 pp、V-JEPA + S-VAE 提升 7.55 pp，量级远大于噪声。
3. **具身预训练主要提升 OOD。** 最佳 OOD +12.12 pp，而 ID 只有 +0.68 pp；趋势清楚且具有机制解释。
4. **同步联合去噪优于本论文测试的异步路径。** 两大 schedule family、双方向、多参数均未超过 sync。
5. **OpenWAM 是强 WAM baseline。** 在多个 benchmark 中为最佳 WAM，并在 EBench 与真机上取得明显优势。

### 7.2 有启发但证据偏弱的结论

1. **预训练后 Mutual 必然优于单向。** 增益只有 0.16–0.72 pp，无 seed 方差。
2. **Tri-System 比 Dual-System 更好。** 只高 0.24 pp，且参数量、理解特征和训练复杂度同时变化。
3. **Wan2.2-VAE 比 DINO+S-VAE 更优。** 只高 0.12 pp，无法说明稳定排序。
4. **WAM 天生 ID 强、VLA 天生 OOD 强。** 多组榜单方向一致，但严格 matched model 对照不足。

### 7.3 缺失的统计与公平性信息

- 无多随机种子、标准差、bootstrap 置信区间或显著性检验；
- 真机常为每任务 10 或 20 次，二项比例方差较大；
- 跨论文 baseline 缺少统一参数量、FLOPs、数据小时、数据重合、SFT demonstrations、action horizon、输入相机和推理预算表；
- 设计原则主要在 RoboTwin 上形成，缺少第二个开发 benchmark 上的完整复验；
- 缺少 ego-only、pretraining data scaling、matched VLA/WAM、matched pixel/semantic latent 等关键实验；
- OpenWAM 各 benchmark 的 fine-tuning 步数从 6k 到 100k 不等，外部 baseline 是否享有同等调参预算未知。

---

## 8. 与算法架构发展脉络的关系

### 8.1 从 VLA 的“语义继承”到 WAM 的“动态继承”

RT-2、OpenVLA、π0 的核心是从 VLM 继承语义与语言对齐；WAM 则从视频生成模型继承动态先验。两者不是互斥：VLA 擅长“看懂当前是什么、指令要什么”，WAM 擅长“未来可能怎样变化”。OpenWAM 的 Tri-System 已经把 understanding stream 加回来，说明最终统一模型很可能同时需要语义、动态与动作三类专属容量。

### 8.2 从共享 backbone 到 modality-specific expert

早期 WAM 倾向于最小改造视频 DiT，把动作 token 拼到视频 token 中。OpenWAM 的消融说明，单纯共享导致动作容量不足；硬路由 FFN expert 也不够。Dual-System 反映了统一多模态模型的一般发展规律：**共享 attention 负责信息交换，独立 FFN/DiT 负责模态专属计算。** 这与近年来 multimodal MoT/MoE 的趋势一致。

### 8.3 从显式“先想象再行动”到同步生成

DreamZero/IDM 路线容易被理解为先生成未来视频，再从视频反演动作。OpenWAM 的同步去噪实验表明，动作并不需要等待未来完全生成；更有效的是在每个噪声尺度上交换中间表征。这与联合 diffusion/flow 的思想一致：协同发生在生成轨迹内部，而不是只发生在最终输出之间。

### 8.4 从像素世界模型到 representation world model

Wan2.2-VAE 在受控实验中表现最好，但 LIBERO-Plus 暴露了像素 latent 对相机与噪声的脆弱。DINO/V-JEPA + S-VAE 的结果指向下一阶段：用紧凑 representation latent 预测与控制相关的语义动态，而不是承担所有像素细节。理想表示应同时满足：

- 对视角、噪声、背景变化鲁棒；
- 保留物体、接触和动作相关的细粒度信息；
- 原生时间压缩；
- token 和 channel 维度足够小，便于大规模 DiT 建模；
- 必要时仍能提供局部重建约束，避免语义特征丢失几何细节。

---

## 9. 局限与批判性评价

### 9.1 作者明确承认的局限

1. 主要研究 embodied pretraining，post-training 与 adaptation recipe 尚未系统研究；
2. 架构主要覆盖 cross-modality attention 和 hard-routed MoE，尚未探索 tokenization-level early fusion、soft-routed MoE；
3. 未纳入 UMI/DexUMI 一类同时具有场景多样性与可执行动作的数据；
4. Wan2.2-VAE 是当前折中而非理想 encoder，对视角、噪声和场景变化不够稳健。

### 9.2 我的进一步判断

**第一，论文解决了“可控研究”问题，但没有完全解决“公平比较”问题。** 框架内部消融很干净，跨论文榜单仍受不可控数据与训练差异影响。论文自己最后得出“无范式绝对胜出”，是合理且诚实的。

**第二，OpenWAM-α 的最大瓶颈是相关数据覆盖，而不是总数据量。** 6,369h 看似很大，但单臂只有约 1,100+h，且视角单一；LIBERO-Plus 直接揭示 embodiment/environment coverage 比总小时数更重要。

**第三，pixel video prediction 同时是优势与负担。** 它提供密集监督，带来强 ID 拟合；也迫使模型解释与动作无关的相机、噪声和背景细节，在 OOD 下累积误差。后续应考虑语义 latent、短时关键帧、残差预测或多尺度目标。

**第四，Mutual visibility 的故事目前比数据更强。** “预训练让双向对应涌现”是很漂亮的解释，但增益不足 1 pp。需要多 seed、attention intervention、梯度分析或 representation probing 才能证明机制。

**第五，真机结果证明的是强初始化与可适配性，不是零样本通用控制。** 每个任务仍需 task-specific SFT；对于真正 generalist robot policy，更关键的测试应是 unseen task + unseen embodiment + 少量或零示范。

---

## 10. 复现与实践建议

### 10.1 最现实的复现层级

1. **低成本机制复现**：使用 OpenWAM-Study checkpoint，在 RoboTwin 或 LIBERO 上复验 mask、sync denoising、encoder latent；重点报告多 seed。
2. **中成本 benchmark 微调**：下载 OpenWAM-Alpha foundation checkpoint，按官方 Hydra 配置微调；Wan2.2-5B 或更小架构推荐 8×80GB GPU。
3. **高成本从零预训练**：需要约 128 H200×7 天、6,369h 清洗数据、统一动作映射和多源数据加载，不适合一般实验室直接重做。

### 10.2 复现时必须核对

- checkpoint 内自带 action normalization，不能替换；
- 33 帧、stride 4 对应 32-step action horizon；
- $t_v,t_a$ 独立采样，推理 10-step sync；
- Wan VAE 与 umT5 冻结，Video DiT、ActionDiT、proprio encoder 更新；
- LIBERO-Plus 直接使用 LIBERO checkpoint，不额外微调；
- 仿真和真机客户端必须复现训练时 crop/resize/camera composition；
- 对异构 embodiment，检查 80-D slot map 与 validity mask；
- 推理结果应同时报告是否启用 DiT cache、compile、同步/异步 buffer，因为速度与行为可能变化。

### 10.3 我建议优先补的实验

1. **Matched VLA vs WAM**：同视觉 backbone、同参数、同数据、同训练步数，只切换 future latent loss；
2. **Matched latent**：同数据与架构比较 Wan-VAE、V-JEPA+S-VAE、DINO+S-VAE、single-frame latent；
3. **单臂数据 scaling**：1k/3k/6k/10k 小时，并分别控制视角多样性与纯小时数；
4. **Ego-only 与 UMI-style**：判断无动作 ego、可执行 human interface、robot data 的边际贡献；
5. **多 seed 与置信区间**：特别验证 0.16–0.72 pp 的 Mutual 结论；
6. **动态闭环评测**：比较同步暂停、异步 prefetch、不同 action horizon 对真实环境扰动的影响；
7. **表示鲁棒性 probing**：直接测 latent 对 camera/noise/background 的不变性，并关联策略 OOD 成功率。

---

## 11. 个人启发与后续问题

### 11.1 对统一具身模型的启发

OpenWAM 说明，“统一”不等于所有模态共用同一组参数。更可行的统一方式是：

- 语义理解、世界动态、动作控制分别保留专属容量；
- 在多个层级通过共享 attention 交换信息；
- 用统一噪声坐标和动作空间把训练协议统一；
- 让不同数据类型只监督其拥有的模态，例如 ego 只监督 world stream。

这比把所有数据强行转成同一种标签更自然，也为“理解—生成—动作”统一模型提供了明确工程接口。

### 11.2 对世界模型研究的启发

未来 world model 不应只问“能否生成漂亮视频”，而应问 latent 是否：

1. 对控制有足够信息；
2. 对无关视觉扰动稳健；
3. 能以较低 token 成本预测；
4. 能通过信息流真正改善动作，而不是只增加辅助 loss。

OpenWAM 的实验表明，高质量视频生成器的知识能迁移，但最终瓶颈正在从“有没有世界先验”转向“用什么 latent 表达、如何把它送到动作流”。

### 11.3 值得继续追踪的问题

- 能否用 V-JEPA / DINO 类语义 latent 加局部重建头，兼顾鲁棒性和接触细节？
- 能否让模型自动学习跨 embodiment 的 action basis，而不是固定 80-D 槽位？
- Mutual visibility 的增益是否会随数据规模继续扩大，还是在 <1 pp 饱和？
- WAM 的 future latent loss 是否可以只关注 task-relevant region、object motion 或 contact event，减少背景像素负担？
- VLA 与 WAM 是否可以做双层目标：VLA 主干负责稳健 action，WAM expert 只在高不确定性或长程规划时介入？
- 对动态环境，异步执行的收益是否会超过同步去噪的离线成功率优势？

---

## 12. 最终评价

OpenWAM 是一篇“模型、系统、实验方法论”三者同时成立的工作。它最值得学习的不是某个 99.3 的榜单数字，而是以下研究范式：

1. 先把设计空间拆成可交换模块；
2. 明确每个实验改变什么、保持什么；
3. 让结论按“继承→交互→扩展”逐级累积；
4. 把最终模型视为实验结论的规模化验证；
5. 对失败 benchmark 做数据与架构双轴诊断；
6. 最后承认 VLA/WAM 各有优势，而不是强行宣布范式胜利。

如果只保留三条结论，应当是：

- **WAM 的有效性不来自“同时预测视频和动作”这件事本身，而来自动作流真正读取世界流、并拥有足够的专属容量。**
- **具身预训练最大的价值是拓展 OOD 世界覆盖；robot data 提供可执行 grounding，ego data 提供场景和交互多样性。**
- **下一代 WAM 的核心突破点不是更长的像素 rollout，而是更紧凑、更鲁棒、与控制更相关的时序 latent，以及与 VLA 语义能力的融合。**

---

## 13. 材料来源

- 论文 PDF：[https://arxiv.org/pdf/2609.07398](https://arxiv.org/pdf/2609.07398)
- arXiv 页面：[https://arxiv.org/abs/2609.07398](https://arxiv.org/abs/2609.07398)
- 项目页：[https://openwam-official.github.io/](https://openwam-official.github.io/)
- 代码仓库：[https://github.com/OpenWAM-Official/OpenWAM](https://github.com/OpenWAM-Official/OpenWAM)
- 模型与数据：[https://huggingface.co/OpenWAM](https://huggingface.co/OpenWAM)
- 本地来源清单：[[assets/OpenWAM - 模块化世界动作模型预训练/sources]]
- 本地官方 PDF：`assets/OpenWAM - 模块化世界动作模型预训练/sources/paper.pdf`
