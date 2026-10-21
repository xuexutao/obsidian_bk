---
type: survey
status: complete
domain: VLA
topic: ByteDance GR Series
year: 2026
source: null
project: null
code: null
papers:
  - "[[GR-1 - 大规模视频生成预训练机器人策略]]"
  - "[[GR-2 - Web规模视频知识生成式VLA]]"
  - "[[GR-3 - 通用机器人基础模型]]"
  - "[[GR-RL - 长时程灵巧真机强化学习]]"
related:
  - "[[BagelVLA - 交错式视觉语言动作生成]]"
tags:
  - VLA
  - Robot-Foundation-Model
  - Video-Pretraining
  - Vision-Language-CoTraining
  - Real-World-RL
  - ByteDance-Research
  - ByteDance-Seed
  - GR-Series
created: 2026-10-21
updated: 2026-10-21
---

# ByteDance GR 系列：从视频生成预训练到真机强化学习

## 基本信息

- **主线范围**：GR-1（2023.12）→ GR-2（2024.10）→ GR-3（2025.07）→ GR-RL（2025.12）
- **团队演化**：ByteDance Research → ByteDance Seed；Hongtao Wu、Tao Kong、Hang Li、Yichu Yang、Jiafeng Xu、Yuxiao Liu、Chilam Cheang 等作者在多篇中重叠
- **共同目标**：把 Web 规模视觉/语言/视频知识迁移到真实机器人，并逐步解决数据稀缺、开放语义、双臂长程、精细控制与闭环可靠性
- **相关但非主线**：[[BagelVLA - 交错式视觉语言动作生成]] 是清华大学—ByteDance Seed 合作线，使用 Bagel 统一理解—生成底座；它不是 GR 编号续作，也不应被称作 GR-4

## 一句话结论

ByteDance GR 系列并不是一条始终坚持“视频生成当策略”的单技术路线，而是一条**瓶颈驱动的机器人基础模型演化线**：GR-1 用 0.8M Ego4D 视频证明生成式未来预测能改善策略，GR-2 把它扩到 38M clips/50B token 并验证 scaling，GR-3 把主轴切换为 Qwen2.5-VL 共训练、flow action expert 和人类轨迹 few-shot，以应对抽象语义与双臂移动任务，GR-RL 再用分布式进度 critic、过滤 BC、镜像增强和在线 latent steering 突破模仿上限；不变的是“让机器人从昂贵真机示范以外的数据与经验中学习”，变化的是每一阶段最稀缺的知识。

---

## 1. 领域根本问题：机器人缺的不是单一种数据

### 1.1 四种知识缺口

通用机器人至少需要四类知识：

1. **视觉动态**：物体、手、工具在时间上如何变化；
2. **语义与常识**：对象类别、抽象属性、空间关系和指令含义；
3. **本体执行**：具体机器人如何用连续动作完成双臂、移动和接触；
4. **闭环纠错**：策略自己造成偏差后如何恢复并超过示范质量。

单靠机器人行为克隆很难同时覆盖。机器人轨迹昂贵，分布窄；Web 视频有动态但无动作；视觉语言数据有语义却不保证可执行；人类示范有动作意图但本体不同；在线真机经验最贴近部署，却最昂贵且有安全风险。

GR 系列的演化可以看作不断把新的数据源接入同一个控制问题：

| 阶段 | 最稀缺知识 | 引入的数据/信号 | 主要学习目标 |
|---|---|---|---|
| GR-1 | 动态先验 | Ego4D 语言视频 | 未来帧 MSE + 动作 BC |
| GR-2 | 规模化动态与对象覆盖 | 38M Web/机器人视频 | 离散视频 token 自回归 + cVAE action chunk |
| GR-3 | 开放语义、本体与低成本适配 | VL 数据 + 机器人轨迹 + 人类手轨迹 | VL next-token + action flow matching |
| GR-RL | 示范质量与闭环恢复 | 成功/失败轨迹 + 真机在线经验 | 分布式 TD、过滤 BC、latent-space RL |

### 1.2 真正主线：Web 知识迁移，而非固定架构

如果只看 GR-1/2，会把系列归类为“video generative policy”；如果只看 GR-3/RL，会把它归类为“VLM + flow VLA + RL”。跨四篇看，稳定主题其实是：**怎样把机器人轨迹之外的可扩展知识，转成真实机器人可执行且可靠的动作**。

因此 GR-3 放弃 GR-2 的完整未来视频生成，不是路线中断，而是瓶颈变化。GR-2 已证明规模化视频动态有效；GR-3 面对的核心变成抽象指令、无效任务、双臂移动和 few-shot；GR-RL 面对的则是毫米级误差、示范次优和训练—部署错位。

---

## 2. 时间线与作者连续性

### 2.1 精确时间线

| 时间 | 工作 | 团队 | 解决的核心问题 | 新暴露的瓶颈 |
|---|---|---|---|---|
| 2023-12 | [[GR-1 - 大规模视频生成预训练机器人策略]] | ByteDance Research | 非机器人语言视频能否通过未来帧生成帮助控制 | 数据/模型/任务规模小，动作单步、接触精度不足 |
| 2024-10 | [[GR-2 - Web规模视频知识生成式VLA]] | ByteDance Research | 视频、模型、任务能否规模化，并形成真机生成式 VLA | 新语义/新技能仍弱，公式与训练细节披露不足 |
| 2025-07 | [[GR-3 - 通用机器人基础模型]] | ByteDance Seed | 如何获得抽象语义、双臂移动、few-shot 与长程灵巧 | 纯模仿遇 OOD 无法恢复、受示范质量上限 |
| 2025-12 | [[GR-RL - 长时程灵巧真机强化学习]] | ByteDance Seed | 如何把通用 base 专家化为可靠精细策略 | 单任务、在线 behavior drift、通用能力是否保留未知 |
| 2026-02 | [[BagelVLA - 交错式视觉语言动作生成]] | 清华大学 + ByteDance Seed | 统一理解—生成底座中交错语言规划、关键帧与动作 | **合作线，不属于 GR 编号主线** |

### 2.2 作者重叠为何重要

GR-1 与 GR-2 的 Hongtao Wu、Ya Jing、Chilam Cheang、Guangzeng Chen、Jiafeng Xu、Hang Li、Tao Kong 等高度重叠，GR-2 明确自称 GR-1 evolution。GR-3 的架构、训练和管理继续包含 Hongtao Wu、Jiafeng Xu、Yichu Yang、Tao Kong、Hang Li；GR-RL 则由 Xiao Ma、Jiafeng Xu、Yunfei Li 核心推进，并明确以 GR-3 为 starting point。

这种作者、模型和硬件连续性支持把四篇视为一条问题演化线，而非名称碰巧相似的平行论文。GR-3 局限中写下“未来引入 RL”，半年后的 GR-RL 正面完成该目标，是最强的直接承接证据。

---

## 3. 第一次转移：GR-1 证明视频生成预训练有效

### 3.1 问题

2023 年的关键疑问不是“大模型能不能控制机器人”，而是没有动作标签的人类视频是否真的能帮助机器人。GR-1 使用 80 万 Ego4D clip、约 800 万帧，训练语言条件未来帧预测；随后在机器人轨迹上保留图像预测，并加入 `[ACT]` 查询预测连续动作。

### 3.2 机制

预训练：

$$\pi(l,o_{t-h:t})\rightarrow o_{t+\Delta t}$$

微调：

$$\pi(l,o_{t-h:t},s_{t-h:t})\rightarrow(o_{t+\Delta t},a_t)$$

冻结 CLIP 和 MAE-ViT，用 GPT 主干、Perceiver、`[OBS]`/`[ACT]` token 连接各模态。总损失是 arm Smooth-L1、gripper BCE 与 pixel MSE 之和。

![](../../10-Papers/VLA/assets/GR-1%20-%20大规模视频生成预训练机器人策略/figures/01-overview.png)

> **GR-1 代表图。** 预训练与微调共享主干，证明视频未来预测可以作为策略初始化与辅助目标。来源：GR-1 Figure 1。

### 3.3 结果与新瓶颈

CALVIN ABCD→D Avg. Len. 4.21，最佳基线 3.06；ABC→D 一任务成功率 85.4%，最佳基线 53.3%；10% 数据 Avg. Len. 2.00，最佳基线 1.11。消融从“无视频、无预训练”3.33 到“有视频、无预训练”3.82，再到完整 4.21，拆分了辅助目标和外部初始化的贡献。

新瓶颈是：0.8M 规模仍小；pixel MSE 预测细节模糊；动作单步回归；真机新类别只有 30%；没有模型/data scaling 证据。

---

## 4. 第二次转移：GR-2 把视频知识做成 Web 规模生成式 VLA

### 4.1 为什么 GR-2 出现

GR-1 已回答“有用”，GR-2 要回答“能否扩展”。它把预训练数据增至 38M clips/50B token，换成冻结 VQGAN 离散视觉 token 和 GPT 自回归视频生成；动作从单步回归改为 cVAE chunk，并配 trajectory optimization + WBC。

### 4.2 数据与架构变化

HowTo100M 提供 36M 主体规模，Ego4D、EPIC-KITCHENS、Kinetics-700、SSV2 增加动作类型，RT-1/Bridge 提供机器人域。hand filtering 提高手—物相关性，re-caption 提高文本质量。

![](../../10-Papers/VLA/assets/GR-2%20-%20Web规模视频知识生成式VLA/figures/02-pretraining-data.png)

> **GR-2 代表图。** 38M 视频横跨教学、第一视角、动作识别和机器人数据；底部动词分布显示其覆盖的操作语义。来源：GR-2 Figure 2。

### 4.3 结果

- 105 个真机任务 Simple：97.7%；
- Unseen Backgrounds/Environments：71.4%/71.7%；
- 数据增强后 Unseen Environments：87.0%，三类泛化平均 74.7%；
- 每任务 50 条轨迹时 Simple：73.9%；
- 122 对象 bin picking：GR-2 平均 79.0%，GR-1 33.3%；
- CALVIN：一任务 98.6%、五任务 85.9%、Avg. Len. 4.64；GR-1 为 94.9%、73.1%、4.21。

S/B/L/XL 四档模型在 Ego4D、RT-1、自采视频验证集上的 loss 随规模下降，真机成功率也随规模上升。它给出“可扩展性证据”，但只有四点、无误差棒和训练预算，尚不是严格 scaling law。

### 4.4 新瓶颈

Unseen Manipulation 仍只有 55.8%，失败包括新形状抓取和选错新对象。背景增强有效，却不能解决抽象语义和新技能。GR-2 的方法与训练细节披露也较少：缺少完整 loss、token 布局、超参数、推理延迟和代码。

---

## 5. 第三次转移：GR-3 从视频生成转向 VLM 共训练

### 5.1 为什么不是 GR-2 的简单放大

新任务需要理解“最大”“左边”“海洋动物”“有触手”等概念，视频动态不等于开放语义。双臂移动机器人还需要连续高维 action chunk 和可靠系统。GR-3 于是采用 Qwen2.5-VL-3B-Instruct，Action DiT 通过 flow matching 生成动作，机器人轨迹与 VL 数据共训练，人类手轨迹用于 few-shot。

### 5.2 核心机制

$$a_t^\tau=(1-\tau)\epsilon+\tau a_t$$

$$\mathcal{L}_{\text{action}}=\mathbb{E}\left[\|v_\theta(l,o_t,s_t,a_t^\tau)-(a_t-\epsilon)\|_2^2\right]$$

机器人 batch 更新 VLM + Action DiT，VL batch 只更新 VLM next-token prediction，两类 batch 等权动态混合。额外 RMSNorm 稳定 Action DiT；Task Status 预测 Ongoing/Terminated/Invalid，并通过随机无效指令防止模型忽略语言。

![](../../10-Papers/VLA/assets/GR-3%20-%20通用机器人基础模型/figures/02-model-architecture.jpg)

> **GR-3 代表图。** 同一个 VLM 一侧为 Action DiT 提供 KV-cache，另一侧保留 VL next-token 学习；这是系列从“生成未来像素”切换到“保持并利用 VLM 语义”的结构标志。来源：GR-3 Figure 3。

### 5.3 三类数据各解决什么

| 数据 | 直接监督 | 解决问题 | 证据 |
|---|---|---|---|
| 机器人轨迹 | 动作 flow | 本体控制与技能 | 三类真机任务基础能力 |
| VL 数据 | next-token | 新对象、抽象指令、grounding | Unseen Instructions/Objects 的 co-training 消融 |
| 人类手轨迹 | egocentric + hand trajectory | 低成本新对象适配 | 10-shot 未见对象 57.8%→86.7% |

### 5.4 结果与新瓶颈

Unseen Instructions 中 π0 40%，GR-3 77.1%；Unseen Objects 中 π0 40%，GR-3 57.8%；每对象 10 条人类轨迹后 86.7%。Table bussing IF 成功率 π0 53.8%、GR-3 97.5%。布料悬挂平均进度 Basic 86.7%、Position 83.9%、Unseen 75.8%。

但纯 imitation learning 在 rollout 进入 OOD 后不会恢复，且受人类示范犹豫和错误限制。报告明确把 RL 作为 future work，直接引出 GR-RL。

---

## 6. 第四次转移：GR-RL 用真实闭环经验突破模仿上限

### 6.1 问题从“会不会”转成“稳不稳”

GR-3 在系鞋带上直接 BC 只有 45.7%。精度、柔性体、双手交接和长时程误差相互放大；部署时 trajectory optimization/receding horizon 又让实际执行动作偏离训练标签。继续加示范不一定有效，因为示范本身次优。

### 6.2 三阶段方法

1. **Offline progress critic**：TD3+BC + 稀疏终局奖励 + retry 前缀失败轨迹，学习有界分布式 Q；
2. **Filtered BC**：用 Q 分布均值 $\rho_t$ 识别进度下降，删除次优 transition；
3. **Morphological symmetry**：同步镜像图像、左右腕、状态、动作和语言；
4. **Online steering**：冻结大部分 base，学习 51.5M noise predictor，在 flow 初始噪声空间优化 Q，同时用 norm penalty 限制偏离。

![](../../10-Papers/VLA/assets/GR-RL%20-%20长时程灵巧真机强化学习/figures/03-training-results.png)

> **GR-RL 代表图。** 45.7%→61.6%→72.7%→83.3% 依次对应过滤、增强、在线 RL；在线曲线先受分布偏移下降，再恢复。来源：GR-RL Figure 5。

### 6.3 结果为何有解释力

阶段成功率显示，GR-3 最大损失发生在穿鞋眼：90.6% 拿起后只剩 58.7% 穿入；GR-RL 分别达到 97.9% 和 89.6%。过滤与在线 RL 主要减少穿孔失败，镜像增强改善交接与各阶段覆盖。增益位置与各模块目标一致，比只看最终成功率更强。

### 6.4 新瓶颈

实验只有系鞋带一个任务族；在线 RL 仍会 behavior drift；轻量 noise predictor 容量与 latent credit assignment 可能限制稳定性；专家化后是否保留通用能力未知。下一步是把专家能力蒸馏回 base，形成 generalist—specialist—generalist 回流。

---

## 7. 四篇横向技术对比

| 维度 | GR-1 | GR-2 | GR-3 | GR-RL |
|---|---|---|---|---|
| 年月 | 2023.12 | 2024.10 | 2025.07 | 2025.12 |
| 角色 | 可行性证明 | 规模化生成式 VLA | 通用 VLM-flow 基础模型 | GR-3 专家化 RL 框架 |
| 输入 | 语言、历史图像、状态 | 语言、多视角视频、状态 | 语言、多视角图像、状态 | 同 GR-3 + critic action 条件 |
| 输出 | 单步动作 + 未来图像 | cVAE action chunk + 未来视频 | flow action chunk | flow action chunk + value distribution |
| Web 数据 | 0.8M Ego4D | 38M clips / 50B token | 大规模 VL 数据，量未披露 | 继承 GR-3 |
| 人类数据 | Ego4D 视频 | Web 人类视频 | VR 手轨迹 few-shot | 遥操作示范 + retry 标注 |
| 机器人数据 | CALVIN + 单臂真机 | 105 任务 + 94k bin-picking | 双臂移动三任务族 | 系鞋带专家数据 + 在线 rollout |
| 主干 | 195M GPT，46M trainable | 230M default，95M trainable | 4B Qwen2.5-VL + Action DiT | 5B policy+critic，51.5M online actor |
| 训练范式 | video pretrain → BC | video pretrain → VLA finetune | robot/VL/human co-training | offline RL filter → BC → online RL |
| 关键公式 | pixel MSE + Smooth-L1/BCE | 离散视频 NTP + cVAE（细节未披露） | rectified flow action loss | distributional TD + noise-space actor |
| 主要本体 | Kinova Gen2 | Kinova Gen3 | ByteMini v1 | ByteMini-v2 |
| 主要优点 | 场景泛化、低数据 | 规模、多任务、bin picking | 抽象指令、few-shot、双臂长程 | 精度、恢复、超过示范 |
| 最大局限 | 小规模、单步动作 | 新 manipulation、披露不足 | OOD 恢复、计算/数据不透明 | 单任务、drift、通用性未知 |

---

## 8. 评测体系如何变化

### 8.1 从 benchmark 到系统任务

GR-1 以 CALVIN 的连续 1–5 任务和 Avg. Len. 为中心；GR-2 加入 105 个真机任务、122 对象 bin picking 与模型 scaling；GR-3 几乎完全转向真实双臂移动任务，指标拆为 IF rate、success、task progress；GR-RL 进一步使用阶段成功率和在线 episode reward。

这说明评测成熟度在提高：

- **链长指标**揭示误差累积；
- **OOD 维度拆分**区分背景、环境、对象、指令、技能；
- **task progress**让长程任务允许部分成功；
- **阶段漏斗**定位失败发生在哪一步；
- **在线曲线**揭示 offline-to-online shift。

### 8.2 仍缺的评测

四篇普遍缺少置信区间、多随机种子、统一硬件吞吐、能耗、跨任务遗忘和安全指标。GR-2/3 的很多结果只有图，没有完整数表。系列内部每代同时改变数据、模型、动作头和硬件，使严格 attribution 困难。

未来需要固定平台的纵向 evaluation suite：同一批任务、对象、相机、控制器下比较 GR-1/2/3/RL；同时报告模型策略频率与低层控制频率，避免把 WBC 200 Hz 混同 VLA 推理 200 Hz。

---

## 9. 数据路线的演变

### 9.1 GR-1：跨域视频作为动态先验

Ego4D 提供 3500+ 小时原始视频，截成 80 万 3 秒 clip。优势是第一视角手—物交互与语言 narration；不足是数据源单一、语言噪声和机器人域差。

### 9.2 GR-2：规模、筛选、重字幕、少量机器人域锚点

38M 混合用 HowTo100M 扩规模，用 hand filtering 保操作相关，用 re-caption 修文本，再用 RT-1/Bridge 对齐机器人域。它开始把数据工程视为核心模型能力。

### 9.3 GR-3：按知识类型拆成三路

机器人轨迹、VL 数据、人类轨迹不再混成一个“大视频池”，而是有不同 loss 和缺失模态处理。人类轨迹收集速度约 450/h，机器人 teleoperation 约 250/h；few-shot 实验用 450 条/30 分钟规模完成 45 个新对象适配。

### 9.4 GR-RL：从静态数据集转向经验质量与部署分布

GR-RL 不追求更大离线数据量，而是问哪些 transition 值得学、策略自己的 rollout 如何回流。retry 前缀把一条最终成功示范拆出多个失败轨迹；在线 buffer 只保留最近两个 checkpoint 的 on-policy 数据，其余转 off-policy。这是从“数据量”转向“数据价值与时效性”。

---

## 10. BagelVLA：合作线而非 GR 编号主线

### 10.1 为什么容易混淆

BagelVLA 同样有 ByteDance Seed 作者/资源，讨论 GR-2、VPP 等生成式策略，也使用 MoT、flow matching、Web 多模态预训练，并在长程任务中同时生成语言子任务、关键帧和动作。这些技术邻近性容易让人把它接在 GR-3 后。

### 10.2 正确关系

BagelVLA 的直接底座是 Bagel（Qwen2.5-7B、SigLIP2、FLUX VAE 的统一理解—生成模型），论文团队是清华大学 IIIS + ByteDance Seed；它提出 interleaved planning 与 Residual Flow Guidance。GR-3/GR-RL 的直接底座则是 Qwen2.5-VL-3B + Action DiT，硬件是 ByteMini，论文明确互相承接。

因此：

- **可说**：BagelVLA 是 Seed 在统一理解—生成具身方向的相关合作线；与 GR-2 的视频预测、GR-3 的 MoT/VLM action expert 有技术交叉；
- **不可说**：BagelVLA 是 GR-4、GR-3 正式续作、GR-RL 的 base，或 GR 编号主线成员。

### 10.3 对主线的补充价值

BagelVLA 重新把 GR-3 弱化的“显式未来预测”带回来，但不生成完整长视频，而用语言子任务 + 关键帧 + 单步 RFG 特征辅助动作。它提示一种可能的合流：GR-3 的 VLM 语义、GR-2 的视觉未来、GR-RL 的闭环价值可以在统一模型中分别负责“理解、预见、纠错”。

---

## 11. 当前正在攻坚的问题

### 11.1 新对象不等于新技能

GR-2 在新背景/环境较强，Unseen Manipulation 只有 55.8%；GR-3 在抽象指令 77.1%，未见对象基础成功率 57.8%。识别概念、选择对象和完成新接触技能是三个层次。Web 语义能解决前两者的一部分，真正新技能仍需 action data、结构化轨迹表示或交互学习。

### 11.2 长程的核心是恢复，而非只会规划

GR-1/2 的 CALVIN 链式指标说明单步错误会放大；GR-3 明确承认 OOD 后不能恢复；GR-RL 的重抓、重穿、重定位才把恢复放到中心。未来评测应主动注入掉落、遮挡、位置偏移，而不只随机初始状态。

### 11.3 精度需要多模态反馈

布料滑移和鞋带穿孔都受限于纯视觉。GR-RL 在 RGB 上已显著提升，但毫米级接触的最终上限可能需要触觉、力矩、事件相机或更高分辨率局部视觉。ByteMini 的硬件迭代也表明本体可靠性与学习算法不可分。

### 11.4 从专家回到通用

GR-RL 专家化后是否遗忘 GR-3 的开放语义未知。论文明确提出蒸馏回 base。理想闭环是：generalist 提供初始化与语义，specialist 用 RL 学高质量行为，成功 rollout 回流通用数据池，再训练下一代 foundation model。

### 11.5 数据和计算透明度

GR-1 披露较完整超参数；GR-2/3/RL 的模型规模增大后，训练预算、数据采样、冻结策略、延迟等反而更少。缺少代码和完整配置使外部难以判断性能来自架构、数据、硬件还是系统调参。这是该系列复现性的最大非算法障碍。

---

## 12. 复现成本分级

| 工作 | 最小可复现部分 | 完整复现主要成本 | 公开缺口 |
|---|---|---|---|
| GR-1 | CALVIN + Ego4D 子集 + 195M GPT | 80 万 clip、真机 4631 轨迹 | 官方代码、GPU/延迟 |
| GR-2 | 小规模 VQGAN video pretrain + CALVIN | 38M/50B、105 任务、94k bin-picking、WBC | 训练公式/超参/代码 |
| GR-3 | Qwen2.5-VL + Action DiT 小模型、invalid status | 4B、多源 VL、286h+ 任务时长、ByteMini | 数据总量、训练预算、延迟、代码 |
| GR-RL | 离线 distributional progress + filter | 5B critic、真机在线交互、安全复位、ByteMini-v2 | 示范量、阈值、atoms、在线超参、代码 |

最推荐的研究复现顺序不是直接造 ByteMini，而是：CALVIN 复现 GR-1 辅助预测 → 小规模离散视频/latent 预测比较 → 公共双臂数据上搭 GR-3 Action DiT + invalid status → 可复位精密任务上做 GR-RL progress filtering 与 latent steering。

---

## 13. 综合判断与展望

### 13.1 已由论文确立的事实

1. GR-1 的视频辅助预测和大规模初始化分别带来增益；
2. GR-2 在更大视频、模型和任务规模下显著超过 GR-1，且 loss/成功率随模型增大改善；
3. GR-3 的 VL co-training 主要改善 unseen instruction/object，而非 seen motor skill；
4. 少量人类手轨迹能高效适配未见对象；
5. GR-RL 的过滤、镜像、在线 steering 都有独立增益，分布式 progress 比回归更稳；
6. 从 GR-3 到 GR-RL 的直接承接由论文方法、作者、架构、硬件和 future work 同时支持。

### 13.2 我的判断

**判断一：系列已从“生成模型策略”升级为“多源知识与经验编排”。** 视频生成只是第一、二阶段最有效的知识接口，不是不可变教条。

**判断二：GR-3 是路线中心点。** 它把 Web 语义、机器人动作、人类 few-shot、双臂移动系统汇合，同时暴露 imitation 的最后瓶颈；GR-RL 的全部价值都建立在这个强 base 上。

**判断三：未来最可能的 GR 主线不是单纯更大 VLM，而是 generalist—specialist 闭环。** 通用模型负责知识和初始化，RL 专家负责高精度部署，专家经验再蒸馏回通用模型。

**判断四：显式未来预测仍可能回归，但形式会更轻。** BagelVLA 的关键帧/RFG、VPP 的 predictive representation 都表明完整像素视频不一定要在部署时生成；未来更可能是 latent/关键帧/点轨迹与 value 的联合预测。

### 13.3 值得持续追踪的实验

- GR-2 视频预训练 + GR-3 VL co-training 的同模型对照；
- 统一 ByteMini benchmark 上 GR-3/GR-RL/BagelVLA 的直接比较；
- GR-RL 专家蒸馏回通用 base 后的跨任务遗忘；
- 触觉/力觉加入布料和鞋带后的阶段成功率变化；
- model/data/online interaction 三维 scaling；
- 自动 retry/失败标注与跨任务 progress critic；
- 相同算力下完整视频生成、latent prediction、纯 VLM 共训练的收益对比。

---

## 14. 文献索引

### 14.1 GR 编号主线

1. [[GR-1 - 大规模视频生成预训练机器人策略]] — *Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation*，arXiv:2312.13139，2023-12。
2. [[GR-2 - Web规模视频知识生成式VLA]] — *GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation*，arXiv:2410.06158，2024-10。
3. [[GR-3 - 通用机器人基础模型]] — *GR-3 Technical Report*，arXiv:2507.15493，2025-07。
4. [[GR-RL - 长时程灵巧真机强化学习]] — *GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation*，arXiv:2512.01801，2025-12。

### 14.2 相关合作线

- [[BagelVLA - 交错式视觉语言动作生成]] — Bagel 统一理解—生成底座的具身合作线；不是 GR 编号主线。

### 14.3 关键外部参照

- RT-1 / RT-2：大规模真实控制与 Web VLM 知识迁移；
- CALVIN：GR-1/2 的长程 benchmark；
- π0 / π0.5 / π*0.6：flow VLA、开放世界与经验学习的重要对照；
- Qwen2.5-VL：GR-3/GR-RL 的 VLM 基座；
- TD3+BC、distributional RL、Q-chunking：GR-RL 离线进度与 value 建模基础；
- Latent-space diffusion policy steering：GR-RL 在线 noise predictor 的直接技术前置。

> 本综述的数字、公式、局限与 future work 均回溯到四篇本地官方 PDF；四篇独立精读均包含至少 4 张官方图、`sources.md` 与复现成本分析。BagelVLA 仅作为相关合作线链接，不纳入 GR 编号时间线。
