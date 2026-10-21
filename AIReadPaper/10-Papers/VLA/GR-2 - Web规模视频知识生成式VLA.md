---
type: paper-note
status: complete
domain: VLA
paper: "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation"
year: 2024
arxiv: 2410.06158
doi: null
source: https://arxiv.org/abs/2410.06158
project: https://gr2-manipulation.github.io
code: null
tags:
  - VLA
  - Generative-VLA
  - Video-Generation
  - Web-Scale-Pretraining
  - Action-Chunking
  - VQGAN
  - Robot-Manipulation
  - ByteDance-Research
  - GR-Series
created: 2026-10-21
updated: 2026-10-21
---

# GR-2：Web 规模视频知识驱动的生成式 VLA

## 基本信息

- **作者**：Chi-Lam Cheang、Guangzeng Chen、Ya Jing、Tao Kong、Hang Li、Yifeng Li、Yuxiao Liu、Hongtao Wu、Jiafeng Xu、Yichu Yang、Hanbo Zhang、Minzhao Zhu（按字母序）
- **机构**：ByteDance Research
- **版本**：arXiv 2410.06158v1，2024-10-08
- **任务**：通用语言条件真机操控、多任务学习、OOD 泛化、工业 bin picking、CALVIN 长程操作
- **预训练规模**：3800 万 text-video clips，约 500 亿 token；数据来自 HowTo100M、Ego4D、EPIC-KITCHENS、Kinetics-700、Something-Something V2、RT-1、Bridge
- **默认模型**：230M 总参数，其中 95M 可训练；另研究 30M/95M/312M/719M 四档可训练规模
- **主要基线**：GR-1；CALVIN 上另比较 RT-1、MT-ACT、HULC、RoboFlamingo
- **官方材料**：<https://arxiv.org/abs/2410.06158>；<https://gr2-manipulation.github.io>

## 一句话结论

GR-2 把 GR-1 的“80 万视频、像素回归、单步动作”推进为“3800 万 Web 视频、VQGAN 离散 token 自回归生成、cVAE 动作 chunk”：它在 105 个真机任务的简单设置达到 97.7% 成功率，在 122 个对象的端到端 bin picking 中把 GR-1 平均成功率从 33.3% 提到 79.0%，CALVIN 五任务平均长度从 4.21 提到 4.64；其核心证据不是单一 SOTA 数字，而是视频生成验证损失和真机成功率都随模型规模改善，说明“Web 视频生成知识 → 机器人动作”的路线开始出现可扩展性。

---

## 1. 研究背景与问题演化

### 1.1 为什么需要 GR-2

[[GR-1 - 大规模视频生成预训练机器人策略]] 已证明非机器人视频的生成式预训练能改善 CALVIN 和小规模真机泛化，但仍存在明显上限：Ego4D 仅 80 万 clip；模型约 195M；未来图像由 patch MSE 一次重建；动作是局部单步回归；真机任务和物体覆盖有限。若要走向“通用机器人 agent”，关键问题不再只是“视频预训练有没有用”，而是能否把视频数据、模型容量、任务数量和真实部署一起扩展。

GR-2 直接把问题改写为 Web 规模生成式 VLA。它先在 3800 万视频上学习“文本 + 当前/历史画面 → 后续画面”，再在机器人轨迹上联合生成未来多视角图像与动作轨迹。与只把 VLM 知识迁移到离散动作 token 的 RT-2 不同，GR-2 认为 Web 视频不仅含物体语义，还含“世界如何变化”的动态知识；与纯视频规划 + 逆动力学不同，它把视觉未来和动作放在同一个策略中学习。

### 1.2 GR-1 暴露的四个瓶颈

1. **数据尺度不足**：0.8M clip 很难验证 scaling；GR-2 扩到 38M，约 47.5 倍；
2. **图像建模接口不理想**：GR-1 用连续 patch MSE，预训练到微调可能因视觉 token 组织改变而损失知识；GR-2 用冻结 VQGAN 离散化图像，统一视频 token；
3. **单步动作不平滑**：真机需要连续、实时轨迹；GR-2 改为 cVAE action chunk；
4. **系统控制缺失**：高层策略输出笛卡尔轨迹后仍需平滑、避碰和关节控制；GR-2 配套 trajectory optimization + whole-body control（WBC），以 200 Hz 执行。

### 1.3 论文贡献

- 构建 38M text-video / 50B token 的预训练混合数据，并做手部过滤与 re-caption；
- 提出统一 GPT 式视频—语言—动作模型，在机器人微调时同时预测多视角未来视频和动作 chunk；
- 用 4 个模型规模验证视频生成 loss 与真机成功率的同步 scaling；
- 在 105 个桌面任务、122 个对象 bin picking 与 CALVIN 上验证多任务、数据效率和 OOD 泛化；
- 给出可部署的 Kinova Gen3 + Robotiq 2F-85 + 双相机 + WBC 系统。

![](assets/GR-2%20-%20Web规模视频知识生成式VLA/figures/01-overview.png)

> **图 1：GR-2 两阶段训练总览（论文 Figure 1，p.2）。** 蓝色部分使用 Web 与机器人视频做 Video-Language Pre-training，橙色部分在机器人轨迹上做 Video-Language-Action Fine-tuning。图中未来视频与动作都由同一 GR-2 产生，体现其“生成式 VLA”定位。来源：<https://arxiv.org/abs/2410.06158>。

---

## 2. 任务定义与输入输出

### 2.1 语言条件策略

给定语言指令 $l$、从 $t-h$ 到 $t$ 的环境观测 $o_{t-h:t}$ 与机器人状态 $s_{t-h:t}$，策略输出从 $t$ 开始、长度为 $k$ 的动作轨迹：

$$a_{t:t+k}=\pi(l,o_{t-h:t},s_{t-h:t})$$

微调阶段同时输出下一未来图像与动作 chunk：

$$\pi(l,o_{t-h:t},s_{t-h:t})\rightarrow(o_{t+1},a_{t:t+k})$$

严格说，机器人数据通常有多个相机，因此 $o$ 是多视角视频序列，输出也包括每个视角的未来图像。状态含末端位置、旋转和二值夹爪状态。

### 2.2 输入与输出表示

- **语言**：冻结文本编码器产生 token；论文引用 CLIP；
- **图像**：冻结 VQGAN 把每帧量化为离散视觉 token，生成后由 VQGAN decoder 还原；VQGAN 在大规模 Internet 图像和域内机器人数据上训练；
- **状态**：位置、旋转、夹爪经可训练线性层编码；
- **未来视觉**：GPT 自回归生成离散图像 token；
- **动作**：cVAE 生成笛卡尔 action trajectory，而不是单步动作。

离散视觉 token 的意义在于预训练和微调都使用同一“下一个 token”接口，减少 GR-1 从像素辅助头切换到动作头时的结构缝隙。动作则不强行离散化，而用 cVAE 表达多模态连续轨迹。

### 2.3 关键假设

第一，Web 视频中的时间动态能迁移到机器人策略；第二，生成的视觉轨迹可充当动作生成的隐式 planner；第三，预训练规模与模型容量增加会同时降低视频生成损失并提高策略成功率；第四，动作 chunk 与控制器共同解决平滑和实时性，模型指标不能脱离整个 robot stack 理解。

---

## 3. 核心方法

### 3.1 GPT 式自回归视频主干

GR-2 的主干接收文本与离散视觉 token，预训练目标是自回归预测未来视觉 token。论文没有给出完整 token 排布公式和网络层数，但明确说明未来图像由 GPT-style transformer 输出，再经冻结 VQGAN decoder 解码。相较 GR-1 的连续 patch MSE，GR-2 把视觉预测变成离散序列建模，更接近成熟语言模型的 scaling 配方。

从概率角度，预训练近似最小化：

$$\mathcal{L}_{\text{video}}=-\sum_{i=1}^{N}\log p_\theta(z^{\text{future}}_i\mid l,z^{\text{history}},z^{\text{future}}_{<i})$$

其中 $z$ 是 VQGAN code。论文未显式写出此公式，上式是对其“GPT 自回归离散未来图像 token”机制的形式化重述，不应误作论文原式。

### 3.2 Web 规模视频数据管线

预训练混合包含：HowTo100M 约 3600 万视频、Ego4D 约 120 万、Kinetics-700 约 12.1 万、EPIC-KITCHENS 约 4.6 万、Something-Something V2 约 4.6 万，以及 RT-1 约 8.2 万、Bridge 约 2.5 万机器人视频。总计 3800 万 clip、约 500 亿 token。

为了让 Web 视频更贴近机器人操控，数据管线做两项关键处理：

1. **hand filtering**：筛选含手—物交互的片段，提高动作相关性；
2. **re-captioning**：重写/补充视频文本，使语言更准确描述动作与对象。

这比 GR-1 直接使用 Ego4D narration 更主动地控制数据分布。加入 RT-1、Bridge 还为预训练提供少量域内机器人视觉。

![](assets/GR-2%20-%20Web规模视频知识生成式VLA/figures/02-pretraining-data.png)

> **图 2：预训练数据（论文 Figure 2，p.3）。** 左侧列出各数据源，右侧展示手—物交互的视觉多样性，底部为高频动词的对数分布。HowTo100M 决定总体规模，Ego4D/EPIC-KITCHENS/SSV2 提供第一视角与细粒度动作，RT-1/Bridge 提供机器人域锚点。来源：<https://arxiv.org/abs/2410.06158>。

### 3.3 多视角 VLA 微调与 cVAE 动作 chunk

机器人微调时，模型输入多相机视频、语言和状态，并共同输出各视角未来帧与动作轨迹。论文强调 action trajectory 而非 single-step action 对平滑和实时性能“至关重要”。cVAE 可以写成：

$$z_a\sim q_\phi(z_a\mid a_{t:t+k},c),\qquad \hat{a}_{t:t+k}=g_\psi(z_a,c)$$

其中条件 $c=(l,o_{t-h:t},s_{t-h:t})$。训练通常包含动作重建和 KL 正则，但论文没有披露具体 ELBO、潜变量维度、chunk 长度和损失权重，因此不能补造。

### 3.4 数据增强

多任务真机微调还使用生成式视频增强：先用扩散模型把新对象插入指定区域；背景区域由 SAM 分割；再用条件视频生成模型根据原视频与修补帧生成保持机器人运动一致的增强视频。该方法针对未见背景和环境，而不是简单颜色抖动。

这套增强显著提升 OOD：GR-2 w/ DA 在 Unseen Environments 达 87.0%，三种泛化设置平均 74.7%。但增强模型本身引入额外训练和质量控制成本，论文没有报告伪影过滤率或合成/真实比例。

### 3.5 WBC 与真机执行

硬件是 7-DoF Kinova Gen3 + Robotiq 2F-85，静态头部相机与末端相机。策略输出笛卡尔轨迹；先经 trajectory optimization 平滑，再由 WBC 转成关节级控制，以 200 Hz 执行，并纳入碰撞约束和 manipulability。这里的 200 Hz 是低层控制频率，不代表 GR-2 神经网络以 200 Hz 重新推理。

### 3.6 模型规模与知识转移

默认 GR-2 为 230M 总参数、95M 可训练。scaling 实验的可训练参数分别是 30M（S）、95M（B）、312M（L）、719M（XL）。Figure 11 显示在 Ego4D、RT-1 和自采机器人验证集上，模型越大，视频生成验证损失越低；真机成功率也从小模型约 38% 上升到 XL 约 60%。论文只给图而未提供精确表值，故这里保留约数。

论文还说明 scaling 实验把域内机器人视频纳入预训练，并在机器人轨迹微调时冻结预训练参数。这一点支持“知识转移尽量无损”的主张，但正文未充分拆分哪些层冻结、动作 cVAE 如何连接、默认实验是否完全采用相同冻结策略，是复现缺口。

---

## 4. 数据与实验设置

### 4.1 105 个真机多任务

团队通过 teleoperation 收集 105 个桌面任务，覆盖 picking、placing、uncapping、capping、opening、closing、pressing、pouring 八类技能。完整数据约 4 万条、平均 400 条/任务；低数据设置约 1/8，即 50 条/任务。

评测分五种：Simple、Distractor、Unseen Backgrounds、Unseen Environments、Unseen Manipulation。最后一项最难，因为对象类别、实例和指令可能都未在机器人训练数据中出现。

### 4.2 工业 bin picking

训练数据约 9.4 万条、55 个对象。测试共 122 个对象，其中 55 个 seen、67 个 unseen；四个设置为 Seen、Unseen、Cluttered Seen、Cluttered Unseen。普通场景源框中 5–9 个对象，cluttered 场景增至 12–17 个。指令只有“move any object from the right basket to the left basket”，因此主要考察抓取、遮挡和跨对象泛化，不考复杂语言。

### 4.3 CALVIN

使用 ABCD-D split，超过 2 万条 34 任务专家演示，评测 1000 条、每条 5 任务指令链。基线包括 RT-1、MT-ACT、HULC、RoboFlamingo、GR-1。论文没有测试 ABC→D、10% data 和 unseen language 等 GR-1 的全部设置，因此两代对比主要在 ABCD→D。

### 4.4 评测统计边界

论文大量结果以柱状图呈现，没有完整数值表、置信区间、每任务 rollout 次数或统计显著性。清晰可核对的数字来自正文；其余图上高度不应精确到小数。训练超参数、硬件、总 FLOPs、推理延迟、action chunk 长度也未披露。

---

## 5. 实验结果

### 5.1 多任务与泛化

GR-2 在 105 个任务的 Simple 设置达到 **97.7%**。正文报告：Unseen Backgrounds 71.4%，Unseen Environments 71.7%，均约为 GR-1 的两倍；加入数据增强后 Unseen Environments 达 **87.0%**，三种泛化设置平均 **74.7%**。只用 50 条/任务时，Simple 仍有 **73.9%**，并在三种 OOD 设置全部超过 GR-1。

Unseen Manipulation 成功率为 **55.8%**，明显低于 seen 和背景变化，失败包括：抓取新形状对象失败，以及看到未见对象指令时选错对象。这说明视频预训练对环境和外观变化更成熟，对“新语义 + 新几何 + 新技能”的组合仍不足。

![](assets/GR-2%20-%20Web规模视频知识生成式VLA/figures/03-multitask-results.png)

> **图 3：多任务成功率（论文 Figure 6，p.7）。** GR-2(400) 在所有设置超过 GR-1；GR-2(50) 体现低数据适配；DA 对未见背景和环境提升最大，但对 Unseen Manipulation 没有同等幅度，说明语义/动作新颖性不是视觉合成增强即可解决。来源：<https://arxiv.org/abs/2410.06158>。

### 5.2 bin picking

GR-2 在四种 bin-picking 设置的平均成功率达到 **79.0%**，GR-1 为 **33.3%**。更关键的是，GR-2 在 unseen 和 cluttered 设置与 seen 设置相近，而 GR-1 一遇未见对象或高拥挤就显著退化。定性示例包含透明、可变形和反光对象，说明离散视频预训练与大规模对象覆盖改善了视觉鲁棒性。

但应谨慎解释：该任务语言固定，不等同于开放词汇指令；WBC、轨迹优化和更大数据也共同变化，无法把全部提升归因于 38M 视频或模型架构。

### 5.3 CALVIN

| 方法 | 连续 1 任务 | 连续 5 任务 | Avg. Len. |
|---|---:|---:|---:|
| GR-1 | 94.9% | 73.1% | 4.21 |
| GR-2 | **98.6%** | **85.9%** | **4.64** |

GR-2 把一任务错误率从 5.1% 降到 1.4%，相对减少约 72.5%；五任务成功率提升 12.8 个百分点。链式指标放大单步可靠性，因此 4.64 已接近五任务上限，但 CALVIN 与真机环境差距仍大。

### 5.4 视频生成与动作的关系

论文展示 multi-task、bin picking、CALVIN 的 Pred/GT 长序列。作者观察预测视频与真实 rollout 高度对齐，并据此称视觉轨迹在为动作提供 planner：动作似乎在“复现”预测视频中的过程。这个证据是定性的，尚未做反事实干预，例如替换/扰乱未来 token 后测动作变化，也没有比较“不解码未来视频、只保留 latent”的策略。

因此比较稳妥的结论是：联合模型学到的视觉未来与动作相互一致；“动作因果依赖可视化未来”仍是解释，而非被严格验证的机制。

### 5.5 scaling 结果

![](assets/GR-2%20-%20Web规模视频知识生成式VLA/figures/04-scaling.png)

> **图 4：规模实验（论文 Figure 11，p.10）。** S/B/L/XL 在 Ego4D、RT-1、自采机器人视频上的验证损失随模型变大而单调降低，真机成功率也上升。该图是 GR-2 相比 GR-1 最新增的证据：不仅“预训练有效”，而且模型容量增加在视频域和控制域呈同向收益。来源：<https://arxiv.org/abs/2410.06158>。

需要强调，四点曲线还不足以建立精确 scaling law：数据量固定，模型训练预算和是否充分收敛未披露，真机柱状图无误差棒，且成功率只在一组子设置上评估。它更适合被称为“正向 scaling evidence”，不是可外推的幂律。

---

## 6. GR-1→GR-2→GR-3→GR-RL 的位置

### 6.1 相对 GR-1：同一命题的规模化

GR-2 与 GR-1 作者高度重叠，论文也明确称其为 GR-1 的 evolution。两者共享“语言条件视频生成预训练 + 机器人轨迹联合预测视频与动作”的核心。变化包括：

| 维度 | GR-1 | GR-2 |
|---|---|---|
| Web 视频 | 0.8M Ego4D clips | 38M 多源 clips / 50B token |
| 图像表示 | MAE-ViT + 像素 patch MSE | 冻结 VQGAN 离散 token，自回归生成 |
| 动作 | 单步 Smooth-L1/BCE | cVAE action chunk |
| 真机范围 | 10 任务、少量对象 | 105 任务、122 对象 bin picking |
| 系统 | Kinova Gen2 | Kinova Gen3 + WBC 200 Hz |
| scaling | 未系统验证 | 30M→719M 可训练参数 |

所以 GR-2 不是范式跳跃，而是把 GR-1 的假设做成规模化系统。

### 6.2 相对 GR-3：主轴从视频生成转为 VLM 共训练

[[GR-3 - 通用机器人基础模型]] 仍继承 Web 知识迁移，但不再报告 38M 视频生成预训练，而是以 Qwen2.5-VL-3B 为语义主干，将机器人 flow matching 与视觉语言 next-token prediction 等权共训练，再加入人类手轨迹 few-shot。GR-2 强在“预测世界如何演化”，GR-3 强在“理解抽象指令、对象与 grounding”。这反映瓶颈从视觉动态和规模转向语言服从、双臂移动、少样本与长程灵巧。

### 6.3 相对 GR-RL：模仿学习的上限

GR-2 的全部策略提升来自预训练和模仿学习，无法超越示范质量，也不处理训练动作与部署后处理的分布错位。[[GR-RL - 长时程灵巧真机强化学习]] 从 GR-3 出发，用 critic 过滤噪声示范并在线强化，专门解决这类闭环可靠性问题。GR-RL 因此是 GR 系列的“expert specialization”分支，而非 GR-2 式生成规模的延续。

### 6.4 BagelVLA 不是 GR 编号主线

[[BagelVLA - 交错式视觉语言动作生成]] 同样由 ByteDance Seed 参与，并显式使用语言规划、关键帧预测和动作生成；它与 GR-2 在生成式 VLA 上相近，也可能使用 GR 数据。但它是清华—Seed 合作下的 Bagel 具身化路线，模型命名、底座和论文作者组织独立，不能把它当作 GR-4，亦不能取代 GR-3/GR-RL 在编号主线中的位置。

---

## 7. 局限、失败案例与批判性评价

### 7.1 论文明确的 future work

作者明确把 future work 聚焦于 action prediction 的泛化与鲁棒性，尤其是 unseen manipulation：新对象、新形状和新技能仍会导致抓取或对象选择失败。这与实验中 55.8% 的最弱 OOD 设置一致。

### 7.2 信息披露不足

1. **方法公式少**：没有完整自回归损失、cVAE ELBO、损失权重、token 布局或冻结层清单；
2. **训练成本缺失**：未报告 GPU、训练时长、batch、学习率、分辨率和上下文长度；
3. **统计不足**：主结果多为图，无 per-task 表格、试验次数、误差棒和置信区间；
4. **系统变量耦合**：GR-1→GR-2 同时改变数据、模型、视觉 tokenizer、动作头、增强和控制器，主结果不能分配贡献；
5. **scaling 证据有限**：只改变模型规模，未联合扫描数据规模和计算最优前沿；
6. **视频 planner 的因果性未验证**：未来视频与动作一致不等于动作依赖可视化未来。

### 7.3 失败模式

- Unseen Manipulation：新形状抓取失败，或选择错误对象；
- 高拥挤 bin picking：尽管平均较强，遮挡、透明和可变形对象仍可能导致抓取不稳定；
- 自回归视觉预测：长序列可能累积模糊/身份漂移，但论文只做定性展示；
- 合成增强：生成伪影可能成为策略 shortcut，报告未给质量过滤分析。

### 7.4 个人判断

GR-2 最有价值的不是“97.7%”，因为 Simple 环境可能接近训练分布；更关键的是三点：38M 视频规模、50 条/任务仍保有 73.9%、以及 S→XL 的视频 loss 与真机成功率同向改善。三者共同支持“生成式预训练有可扩展性”。

但论文无法证明 VQGAN 自回归生成优于同规模 VLM 共训练或 latent predictive objective，因为没有参数/数据匹配的对照。GR-3 后来转向 Qwen2.5-VL 共训练，也暗示像素级视频生成不是通用 VLA 的唯一最优接口。

---

## 8. 复现成本与实践建议

### 8.1 成本构成

- **预训练数据**：38M clip / 50B token 的下载、清洗、hand filtering、re-caption 和去重是最高门槛；HowTo100M 的版权与可用性还需逐项遵循原数据许可；
- **视觉 tokenizer**：VQGAN 不只是通用模型，还在 Internet + robot 域训练；若 tokenizer 域偏差大，未来帧 token 会损失手爪细节；
- **机器人数据**：105 任务约 4 万轨迹，bin picking 9.4 万轨迹；
- **系统**：Kinova Gen3、Robotiq 夹爪、双相机、trajectory optimization 和 WBC；
- **训练/推理**：论文未披露硬件、时长和策略延迟，无法给出可信 GPU 小时。

### 8.2 最小可验证复现

1. 用公开 Ego4D/SSV2/Bridge/RT-1 子集建立小规模 VQGAN + GPT 视频预测；
2. 在 CALVIN 复现 GR-1→GR-2 的视觉 tokenizer 与 action chunk 变化；
3. 做四个关键对照：无视频预训练、静态图像预训练、latent future prediction、离散视频生成；
4. 固定模型与数据，只替换 single-step/cVAE/diffusion action head；
5. 用 predicted video token 随机置换或停止梯度，检验动作是否真正依赖视觉未来；
6. 报告推理延迟、chunk 长度、WBC 频率和策略频率，避免把 200 Hz 低层控制误报为 VLA 频率。

### 8.3 工程风险

大规模自回归图像 token 上下文长，训练吞吐和显存压力高；多相机让 token 数进一步增长。VQGAN 的小物体/细长结构重建误差可能伤害精细操控。cVAE 的 posterior collapse 与多模态 latent 选择也可能导致动作平均化。论文未给代码，复现者需要自行决定大量关键超参数。

---

## 9. 启发与后续问题

### 9.1 已确立事实

- GR-2 在相同系列设置下显著超过 GR-1；
- 视频生成验证 loss 随模型增大而下降；
- 真机成功率随模型增大而上升；
- 数据增强主要改善背景和环境泛化；
- 新 manipulation 仍是最难 OOD 维度。

### 9.2 仍属推测

- 未来视频“规划”导致动作提升；
- Web 视频动态而非对象覆盖/语言多样性是主要收益来源；
- 继续增加模型和视频规模会维持同样控制收益；
- VQGAN 离散视觉 token 是无损知识转移的关键。

### 9.3 值得继续追问

- 生成完整像素是否必要，还是中间预测特征已足够？
- 模型规模、视频数据量、机器人数据量三者的 compute-optimal 配比是什么？
- action cVAE 与视频自回归主干之间如何共享注意力，是否存在信息瓶颈？
- 多视角未来预测是否真的改善 3D 几何，还是只增加辅助监督？
- 若把 GR-2 的 38M 视频知识与 GR-3 的 VL co-training 合并，是否能同时获得动态与抽象语义？
- 生成视频的错误能否被不确定性估计捕获，避免误导动作？

---

## 10. 材料来源

| 本地材料 | 位置 | 用途 |
|---|---|---|
| 官方论文 PDF | `assets/GR-2 - Web规模视频知识生成式VLA/sources/paper.pdf` | 全文、数字与图注核对 |
| 来源记录 | `assets/GR-2 - Web规模视频知识生成式VLA/sources.md` | PDF、原图与项目页溯源 |
| Figure 1 | `figures/01-overview.png` | 两阶段训练 |
| Figure 2 | `figures/02-pretraining-data.png` | 数据规模与组成 |
| Figure 6 | `figures/03-multitask-results.png` | 多任务与 OOD 结果 |
| Figure 11 | `figures/04-scaling.png` | 模型规模实验 |

> 本笔记基于本地官方 PDF（21 页）逐页阅读，数字优先采用正文明确值；只有图而无数值表的结果不伪造精确小数。图片来自 arXiv 官方源码并已逐张检查。关联综述：[[ByteDance GR 系列 - Survey]]。
