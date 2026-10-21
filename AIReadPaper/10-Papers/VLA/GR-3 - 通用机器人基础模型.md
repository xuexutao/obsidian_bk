---
type: paper-note
status: complete
domain: VLA
paper: "GR-3 Technical Report"
year: 2025
arxiv: 2507.15493
doi: null
source: https://arxiv.org/abs/2507.15493
project: https://seed.bytedance.com/GR3
code: null
tags:
  - VLA
  - Robot-Foundation-Model
  - Vision-Language-CoTraining
  - Flow-Matching
  - Human-Trajectory
  - Bimanual-Manipulation
  - Mobile-Manipulation
  - ByteDance-Seed
  - GR-Series
created: 2026-10-21
updated: 2026-10-21
---

# GR-3：通用机器人基础模型

## 基本信息

- **作者/团队**：ByteDance Seed；报告按贡献列出完整作者。模型架构：Hongtao Wu、Xin Xiao；训练：Sijin Chen、Xiao Ma、Hongtao Wu、Xin Xiao、Jiafeng Xu、Yichu Yang；Team Lead：Tao Kong、Hang Li
- **版本**：arXiv 2507.15493v2，2025-07-22；报告日期 2025-07-23
- **任务**：自然语言条件的双臂移动操控，覆盖开放语义 pick-and-place、长时程 table bussing、灵巧布料悬挂
- **模型**：Qwen2.5-VL-3B-Instruct + Action Diffusion Transformer，Mixture-of-Transformers 风格，总参数 4B
- **训练数据**：机器人轨迹、Web 视觉语言数据、人类第一视角手轨迹三类；完整总量未披露
- **机器人**：ByteMini，22-DoF 双臂移动平台；策略 rollout 控制 19 DoF
- **主要基线**：π0；另有 w/o Co-Training、w/o Norm、w/o Task Status 等内部消融
- **官方材料**：<https://arxiv.org/abs/2507.15493>；<https://seed.bytedance.com/GR3>

## 一句话结论

GR-3 标志着 ByteDance GR 主线从“生成未来视频”转向“以预训练 VLM 为语义骨干的通用机器人基础模型”：它用 Qwen2.5-VL-3B 处理多视角与语言、用 4B MoT/Action DiT 以 flow matching 生成双臂移动 action chunk，并把机器人 imitation loss 与 Web 视觉语言 next-token loss 共训练，再用少量人类手轨迹适配新对象；结果是在未见抽象指令上把 π0 成功率从 40% 提到 77.1%，用每对象 10 条人类轨迹把未见对象成功率从 57.8% 提到 86.7%，同时在 table bussing 和布料悬挂中显著提高长程与灵巧能力，但仍受 OOD 恢复和模仿学习上限约束。

---

## 1. 研究背景与问题转移

### 1.1 从视频动态到通用语义与本体

[[GR-1 - 大规模视频生成预训练机器人策略]] 与 [[GR-2 - Web规模视频知识生成式VLA]] 的核心命题是：人类视频中蕴含物理动态，未来帧生成可作为动作预测先验。GR-2 已把规模推到 38M clips，并证明模型容量与真机成功率同向提升。到 2025 年，瓶颈开始转移：机器人需要理解“最大的物体”“左边的可乐”“有触手的动物”“海洋动物”这类机器人轨迹从未出现的抽象语义，还要控制双臂、移动底盘、长时程接触和可变形物体。

GR-3 因此不再围绕完整未来视频生成展开，而是直接采用 Qwen2.5-VL-3B-Instruct，把 Web 视觉语言数据与机器人动作训练共置于一个多模态模型。Web 数据提供对象、空间、常识和 grounding；机器人轨迹提供可执行动作；人类轨迹提供低成本跨本体适配。这三类数据对应三个目标：零样本泛化、强模仿控制、few-shot 新对象适配。

### 1.2 现有 VLA 的三类不足

1. **指令服从不足**：模型容易利用场景共现而忽略语言，尤其在 novel destination、invalid instruction 和抽象概念上；
2. **机器人数据成本高**：新对象/环境若每次都需真机 teleoperation，扩展速度受限；
3. **长程灵巧不稳定**：误差随步骤累积，布料等可变形物体还需要双臂协调和恢复；
4. **系统与本体耦合**：通用策略必须落到可靠硬件、全身遥操作、轨迹优化与控制，而不只是离线 benchmark。

### 1.3 核心贡献

- 构建 4B 参数、Qwen2.5-VL-3B + Action DiT 的多视角 VLA；
- 引入额外 RMSNorm、后半层 KV-cache、causal action attention 等架构选择，改善训练稳定和语言服从；
- 将机器人 flow-matching loss 与 Web VL next-token loss 等权共训练；
- 用 task status（Ongoing/Terminated/Invalid）和随机无效指令强迫动作分支读取语言；
- 用 PICO 4 Ultra 收集人类第一视角与手轨迹，以每对象 1/5/10 条样本做跨本体 few-shot；
- 同步开发 ByteMini 双臂移动平台与全身控制系统，并在三个高难真机任务上对比 π0。

![](assets/GR-3%20-%20通用机器人基础模型/figures/01-teaser.jpeg)

> **图 1：GR-3 总览（论文 Figure 1，p.2）。** 顶部三类数据分别是视觉语言、机器人轨迹和人类轨迹；底部三类能力分别是灵巧布料、开放对象 pick-and-place、长程 table bussing。相较 GR-2，Web 数据不再只以视频生成形式出现，而被明确拆为语义知识、机器人控制和低成本适配三路。来源：<https://arxiv.org/abs/2507.15493>。

---

## 2. 任务定义与输入输出

### 2.1 端到端 action chunk 策略

GR-3 是策略 $\pi_\theta$。给定自然语言指令 $l$、多相机观测 $o_t$ 和机器人状态 $s_t$，输出长度为 $k$ 的动作 chunk：

$$a_t=a_{t:t+k}=\pi_\theta(l,o_t,s_t)$$

观测包括头部相机、左右腕相机；机器人状态包含双臂、夹爪和移动底盘相关 proprioception。策略 rollout 控制 19 DoF，排除升降机构的 3 DoF 和头部。报告没有披露 $k$、动作维数、观测分辨率或策略执行频率。

### 2.2 Flow Matching 动作生成

真实 action chunk $a_t$ 与高斯噪声 $\epsilon\sim\mathcal{N}(0,I)$ 线性插值得到：

$$a_t^\tau=(1-\tau)\epsilon+\tau a_t,\qquad \tau\sim U(0,1)$$

目标速度为 $u(a_t^\tau\mid a_t)=a_t-\epsilon$，训练损失：

$$\mathcal{L}_{\text{action}}(\theta)=\mathbb{E}_{\{a_t,o_t,s_t,l\}\sim\mathcal{D}}\left[\|v_\theta(l,o_t,s_t,a_t^\tau)-u(a_t^\tau\mid a_t)\|_2^2\right]$$

推理从 $a_t^{\tau=0}\sim\mathcal{N}(0,I)$ 出发，用 Euler 积分：

$$a_t^{\tau+\Delta\tau}=a_t^\tau+v_\theta(l,o_t,s_t,a_t^\tau)\Delta\tau$$

实验取 $\Delta\tau=0.2$，即从 0 到 1 需要 5 个 Euler 更新。训练时一次 VLM 前向对应多个采样 flow timestep，以复用昂贵的视觉语言 KV-cache、提高吞吐。

### 2.3 视觉语言共训练

机器人 batch 同时更新 VLM 与 Action DiT，优化 flow matching；VL batch 只更新 VLM，做 next-token prediction。两类 mini-batch 动态混合且等权，总目标可概括为：

$$\mathcal{L}=\mathcal{L}_{\text{action}}+\mathcal{L}_{\text{VL-NTP}}$$

报告没有给每类 batch 的采样比例之外的更细权重。VL 数据覆盖 image captioning、VQA、image grounding、interleaved grounded image captioning，并经过过滤和重标注。

### 2.4 人类轨迹 few-shot

PICO 4 Ultra Enterprise 记录第一视角视频与双手轨迹。人类数据没有腕相机、关节和夹爪状态，训练时对缺失腕部视角填空图，只监督手轨迹。模型先完成机器人 + VL 共训练，再加入人类轨迹做三类数据联合继续训练。核心假设是 action representation 足够统一，使人手轨迹能迁移到 ByteMini 双臂。

---

## 3. 模型架构与关键机制

### 3.1 Qwen2.5-VL + Action DiT

GR-3 使用 Qwen2.5-VL-3B-Instruct 处理语言和多视角图像；Action Diffusion Transformer 处理机器人状态和 noisy action chunk。动作 DiT 接收 VLM KV-cache 作为条件，层数为 VLM 的一半，并只读取 VLM 后半层的 KV-cache，以降低推理成本。动作 chunk 被表示成 $k$ 个 token，与状态 token 拼接；flow timestep 经 AdaLN 注入；action token 使用因果 attention mask 以建模 chunk 内时间依赖。总参数为 4B。

这种结构属于 Mixture-of-Transformers 的“模态专家共享条件”形式：VLM 不直接自回归吐出离散动作，而让连续生成专家读取语义/视觉缓存。相比 GR-2 的 GPT + cVAE，它把语义推理和连续动作生成分得更清晰。

![](assets/GR-3%20-%20通用机器人基础模型/figures/02-model-architecture.jpg)

> **图 2：GR-3 模型（论文 Figure 3，p.4）。** 左侧机器人轨迹路径由多视角和指令进入 VLM，KV-cache 条件化 Action DiT；右侧 VL 数据只训练 VLM 做 next-token prediction。两种训练共享 VLM，使 Web 语义能够进入动作分支。来源：<https://arxiv.org/abs/2507.15493>。

### 3.2 额外 RMSNorm：稳定性与语言服从

早期训练频繁不稳定。受 QK norm 启发，作者在 DiT block 的 attention 和 FFN 内部线性层后增加 RMSNorm。报告称这不仅稳定训练，还显著提高下游 instruction following。Table bussing 消融中，去掉 Norm 在 Flat 与 IF 都退化，尤其无法泛化到 novel destinations。

值得注意的是，归一化看似底层优化细节，却会影响语言条件是否被动作专家利用。可能原因是流模型的 noisy action 与 VLM cache 尺度不匹配，额外 norm 改善跨模态条件信号；论文没有进一步分析激活统计，因此机制解释仍待验证。

### 3.3 Task Status：阻止忽略语言

策略新增一个动作维度预测任务状态：Ongoing=0、Terminated=1、Invalid=-1。训练中随机用与当前观测不相容的指令替换原指令；无效样本只监督 status，不监督其他动作维度。若场景里没有刀，“把刀放进篮子”就是 Invalid。

这一机制直接针对 VLA shortcut：多视角场景和固定数据分布可能让策略根据物体共现猜任务，不读取语言。Invalid 负样本迫使 Action DiT 判断“指令是否可执行”，也为部署提供拒绝机制。去掉 Task Status 后 table bussing 的 IF 性能下降，验证其有效性。

### 3.4 数据调度与质量控制

真机收集不是自由采样，而由 scheduler 指定动作、对象组合与背景。teleoperator 按新配置布置环境，收集后过滤无效和低质量轨迹。该设计把数据多样性当成工程优化目标，减少常见组合过多、长尾组合缺失。

![](assets/GR-3%20-%20通用机器人基础模型/figures/03-training-data.jpg)

> **图 3：GR-3 三类数据（论文 Figure 4，p.6）。** 顶部展示对象/背景/动作调度后的机器人遥操作，中部是 VR 手部追踪的人类轨迹，底部是 caption、VQA、grounding 与 grounded caption 等 VL 数据。来源：<https://arxiv.org/abs/2507.15493>。

### 3.5 ByteMini 与控制栈

ByteMini 是 22-DoF 双臂移动机器人：两条 7-DoF arm，球形 wrist，omni mobile base，lift，头部与双腕 RGB-D。球形 wrist 紧凑，肘内收可达 2.53 rad，适合胸前双臂协同。QDD 执行器提升透明性与稳定性；双电池续航超过 10 小时；无线急停保证数据收集安全。

遥操作侧把人类全身/双手动作重定向到可行机器人运动，同时考虑 manipulability、奇异点和关节约束。部署侧对 action chunk 做 pure pursuit 与实时参数化 trajectory optimization，最小化 jerk 并平滑 chunk 间衔接。论文结果因此属于“模型 + 数据 + 本体 + 控制”一体系统，不应把所有成功率归功于 VLA 参数。

---

## 4. 训练数据与成本

### 4.1 机器人轨迹

已明确披露的任务级数据：

- Generalizable pick-and-place：35k 轨迹、101 个训练对象、69 小时；
- Table bussing：约 101 小时机器人轨迹；
- Cloth manipulation：116 小时机器人轨迹。

报告没有披露三者轨迹总数、各阶段训练步数、batch、学习率、GPU、训练时长或 VL 数据规模。不能把 35k 直接当作 GR-3 全部机器人数据。

### 4.2 VL 数据

来源引用 Seed1.5-VL、CLEVR、LLaVA-NeXT-Interleave、GLaMM、BLIP-3 等，任务覆盖 caption、VQA、grounding 和 interleaved grounding。论文强调过滤和重标注，却未公布样本数、合成标注比例、机器人/VL batch 比之外的采样细节。

### 4.3 人类轨迹效率

人类手轨迹收集速度约 450 条/小时，机器人 teleoperation 约 250 条/小时。未见对象 few-shot 实验对 45 个对象最多收集每对象 10 条，共 450 条，总时长约 30 分钟；然后继续三类数据共训练 20k steps。这个成本优势成立，但 450 条/小时与 30 分钟/450 条看似口径不同：前者是一般吞吐，后者为该实验有效轨迹总视频时长或实际采集过程描述，复现时需向作者确认计时定义。

---

## 5. 实验设置与定量结果

### 5.1 Generalizable Pick-and-Place

测试四种设置：Basic、Unseen Environments、Unseen Instructions、Unseen Objects。Basic 和新环境使用 54 个见过对象，每 6 个一组；新环境包括收银台、会议室、办公桌和休息室。Unseen Instructions 要理解大小、空间关系与常识，例如“最大的物体”“雪碧旁边的可乐”“海洋动物”。Unseen Objects 使用 45 个新对象，每 5 个一组，其中超过 70% 类别也未在机器人轨迹中出现。

指标有两个：IF rate 表示是否正确接近指令对象；Success rate 表示是否最终放入容器。主要正文数字：

- Unseen Instructions：π0 成功率 **40%**，GR-3 **77.1%**；
- Unseen Objects：π0 **40%**，GR-3 **57.8%**；
- 对 45 个未见对象加入每对象 10 条人类轨迹后：成功率从 **57.8%** 提至 **86.7%**；
- co-training 对 Basic 和新环境影响不大，却在新指令、新对象上显著提升；只用机器人轨迹的 GR-3 甚至弱于 π0，作者认为 π0 的大规模跨本体预训练是原因。

![](assets/GR-3%20-%20通用机器人基础模型/figures/04-pick-place-results.jpg)

> **图 4：开放语义与 few-shot 结果（论文 Figure 7，p.9）。** 左侧显示 π0、无 VL 共训练、完整 GR-3；共训练的优势集中在 Unseen Instructions/Objects，而非 seen 设置。右侧显示 1/5/10-shot 人类轨迹持续改善未见对象，同时没有明显破坏 seen 对象。来源：<https://arxiv.org/abs/2507.15493>。

### 5.2 Long-Horizon Table Bussing

任务要求清理一张混乱餐桌：食物装入外带盒，餐具放入 bussing box，垃圾放入垃圾桶，还要移动底盘跨工作区。Flat 设置只给“clean up the dining table”，用平均子任务进度；IF 设置逐个给子任务指令，包含 Basic、Multiple Objects、Multiple Destinations、二者组合、Novel Destinations、Invalid Tasks 六类。

GR-3 在 IF 平均成功率达到 **97.5%**，π0 为 **53.8%**。π0 常混淆叉子和勺子，并在 Novel Destinations 中依赖训练共现，把物体放回熟悉容器；GR-3 能执行多对象/多目的地，并在 Invalid Tasks 中保持 10 秒不操作。

消融表明：去掉额外 RMSNorm 会在两种设置下降，尤其不服从 novel destination；去掉 Task Status 也降低 IF。论文 Figure 8 还给 Flat 的平均进度，但正文没有列精确数值，故不从柱状图补造。

### 5.3 Dexterous Cloth Manipulation

任务分四个里程碑：拿起衣架、把右肩套上、把左肩套上、挂到晾衣架。训练 116 小时机器人轨迹；测试 Basic（6 件 seen 衣服）、Position（旋转/揉皱）和 Unseen Instances（4 件新衣服，其中 2 件短袖，而训练全是长袖）。指标为平均任务进度。

| 设置 | GR-3 平均进度 |
|---|---:|
| Basic | **86.7%** |
| Position | **83.9%** |
| Unseen Instances | **75.8%** |

GR-3 三种设置都高于 π0。主要失败阶段是右肩完成后把左肩套上：左侧领口常折在衣架后，机器人需要一只手持衣架、另一只手拉出领口；另一个失败是衣架从夹爪滑落，导致最后悬挂失败。

![](assets/GR-3%20-%20通用机器人基础模型/figures/05-cloth-results.jpg)

> **图 5：布料操控阶段结果（论文 Figure 10，p.13）。** 左侧 Sankey/阶段条显示两模型都在“左肩上衣架”损失最大；右侧 GR-3 在 Basic、Position、Unseen Instances 都高于 π0。该图比单一最终成功率更能定位灵巧瓶颈。来源：<https://arxiv.org/abs/2507.15493>。

---

## 6. 消融与机制证据

### 6.1 VL Co-Training

Basic/Unseen Environment 中，GR-3 与 w/o Co-Training 接近；Unseen Instructions/Objects 中，完整模型大幅更好。这个模式说明 VL 数据的作用不是普遍提高 motor skill，而是提供机器人轨迹缺失的概念、对象和 grounding。它也是 GR-3 与 GR-2 的关键差异：GR-2 强调视频动态，GR-3 对语义 OOD 给出更直接证据。

### 6.2 Few-Shot Human Trajectory

0→1→5→10-shot 对未见对象的 IF 和成功率整体上升，10-shot 成功率 86.7%，seen 对象无明显遗忘。由于 human trajectory 只有 egocentric + hand tracks，结果支持跨本体迁移。不过没有“同数量机器人轨迹”“纯视频无手轨迹”“只微调 action head”等对照，无法判断收益来自数据效率、视角一致性还是继续训练本身。

### 6.3 RMSNorm

w/o Norm 在 table bussing Flat 和 IF 都变差，尤其不能遵守 novel destination。作者把它归因于训练稳定与条件利用。缺少训练 loss 曲线、梯度范数和多随机种子，因此“显著稳定”仍主要是工程观察，但下游消融支持其必要性。

### 6.4 Task Status

w/o Task Status 的 IF 降低。Invalid instruction 负样本通过 status 头为语言服从提供显式监督。这比仅靠自然轨迹中的正指令更容易区分“看场景猜动作”和“真正读取语言”。它也为安全停机提供接口，但论文没有报告 Invalid 的误拒/漏拒混淆矩阵。

---

## 7. GR 系列关系与 BagelVLA 边界

### 7.1 GR-1 → GR-2 → GR-3

GR-1 提出生成式视频预训练；GR-2 把视频、模型、任务和系统规模扩大；GR-3 则更换核心预训练接口：

| 维度 | GR-1 | GR-2 | GR-3 |
|---|---|---|---|
| Web 知识 | 0.8M Ego4D 视频 | 38M 视频 / 50B token | Qwen2.5-VL + curated VL data |
| 生成目标 | 像素未来帧 MSE | VQGAN token 未来视频 | VL next-token + 动作 flow |
| 动作 | 单步回归 | cVAE chunk | Action DiT flow matching |
| 本体 | 单臂桌面 | 单臂 + WBC | 双臂移动 ByteMini |
| 新数据 | 机器人轨迹 | 机器人轨迹 + 合成增强 | 机器人 + VL + 人类手轨迹 |
| 主要瓶颈 | 场景泛化 | 规模与多任务 | 抽象语义、few-shot、长程灵巧 |

GR-3 不是“GR-2 再放大视频模型”，而是承认 VLM 语义知识和连续 flow action expert 更适合开放指令与复杂本体。系列的恒定主线是 Web 知识迁移，具体预训练任务从视频生成转为多模态共训练。

### 7.2 GR-3 → GR-RL

本报告局限章节明确提出未来引入 RL，解决 rollout 进入 OOD 后无法恢复、模仿学习无法超越示范的问题。[[GR-RL - 长时程灵巧真机强化学习]] 正是这一 future work 的直接落地：以 GR-3 为 base，训练分布式 critic 估计进度、过滤示范，再做镜像增强与在线 latent-noise RL，把系鞋带成功率从 45.7% 提到 83.3%。GR-RL 与 GR-3 的作者、Qwen2.5-VL + Action DiT、ByteMini 平台都高度连续。

### 7.3 BagelVLA 的正确定位

[[BagelVLA - 交错式视觉语言动作生成]] 是 ByteDance Seed 与清华合作的统一理解—生成 VLA，使用 Bagel MoT、语言子任务、关键帧预测和动作流。它与 GR-3 共享 Seed、多模态共训练和 MoT/action expert 思想，也引用 GR-2 等生成式 VLA；但它没有 GR 编号，也不是 GR-3→GR-RL 的直接同一模型谱系。应称为“相关合作线”，不能称 GR-4 或 GR 主线成员。

---

## 8. 局限、失败案例与批判性评价

### 8.1 论文明确局限

作者明确承认：

1. 未见指令涉及新概念/对象时仍会出错；
2. 未见形状对象的抓取仍困难；
3. 与所有纯 imitation learning 类似，rollout 进入 OOD 状态后会卡住，缺乏失败恢复；
4. future work 是扩大模型与数据，并引入 RL，在复杂灵巧任务上突破模仿上限。

### 8.2 实验局限

- 主要 baseline 只有 π0，缺少 π0.5、OpenVLA、RDT、GR00T N1 等同代横向；
- 三个任务分别训练还是共享一个 checkpoint 的细节不够透明；table bussing 明确 Flat/IF 各训模型，不能把所有能力理解成单一万能策略；
- 完整训练数据、VL 数据量、超参数、计算资源、推理延迟和代码均未披露；
- 真机 trial 数、置信区间、随机种子缺失；
- ByteMini 硬件与控制器是系统优势，模型和平台贡献未分离；
- 人类 few-shot 没有与同量机器人示范比较。

### 8.3 失败机制

布料任务的左肩阶段需要在遮挡和形变下做双臂精细协同；夹爪滑移说明视觉策略缺少触觉/力反馈。pick-and-place 对新形状失败说明 VLM 识别“是什么”不等于掌握“怎么抓”。table bussing 中 π0 的共现 shortcut 被 task status 缓解，但 GR-3 仍可能在更开放环境中遇到新的无效/矛盾指令。

### 8.4 个人判断

GR-3 最强的证据是**增益按问题类型出现**：VL co-training 在 seen 设置几乎不变，却在 unseen concept/object 大幅提升；这比所有任务统一涨点更能证明机制。10-shot 人类轨迹从 57.8% 到 86.7% 也显示低成本跨本体数据值得投入。

但“generalist foundation model”仍需谨慎：实验只有三大任务族，而且部分设置单独训练；未证明单一 checkpoint 可任意组合新任务。更准确的评价是：GR-3 是一个强通用初始化和多源训练配方，能够高效适配多类高难真机任务。

---

## 9. 复现成本与实践建议

### 9.1 计算成本

Qwen2.5-VL-3B + Action DiT 总计 4B，多视角输入、5 步 Euler flow inference，训练还混合 Web VL batch。报告未给 GPU 数和时长，但显然高于 GR-1/GR-2。只使用后半 VLM KV-cache和较浅 Action DiT 是主要节省手段。

### 9.2 数据与硬件成本

- 至少 69h pick-and-place、101h table bussing、116h cloth 的真机时长；
- 自定义 ByteMini、双臂球形 wrist、RGB-D、VR 全身遥操作和轨迹优化；
- 视觉语言数据需要过滤、重标注与多任务采样；
- 人类轨迹收集便宜，但跨本体坐标、空腕视角 padding、手—夹爪动作映射仍需工程实现。

### 9.3 最小复现建议

1. 在公开双臂数据上搭建 Qwen2.5-VL-3B + 小型 Action DiT，先验证 5 步 flow inference；
2. 构造 valid/invalid 指令对，复现 task status 对语言服从的提升；
3. 用小规模 caption/VQA/grounding 数据与机器人轨迹等权交替，比较只冻结/微调 VLM；
4. 记录有/无额外 RMSNorm 的 loss、梯度和成功率，多随机种子验证；
5. few-shot 实验必须加入等量 robot demonstration 与 human video-only 对照；
6. 在仿真或可复位真机任务上先评估 OOD recovery，再尝试 RL。

### 9.4 不可忽略的系统细节

动作 chunk 的轨迹优化会改变训练动作与部署动作，这一 mismatch 后来成为 GR-RL 的核心问题。若复现 GR-3，只对离线 action loss 做到一致还不够，必须记录实际执行轨迹并评估 post-processing 造成的偏移。否则离线指标好而真机失败，很可能不是 VLM 或 flow loss 本身。

---

## 10. 启发与后续问题

### 10.1 对路线的启发

GR-3 显示通用机器人模型至少需要三种不同知识：Web VL 语义、机器人动力学/执行、人类操作先验。它们不能简单拼接，必须通过共享 VLM cache、动作生成专家和统一采样策略连接。相比 GR-2 的完整视频生成，GR-3 更强调“保持并利用 VLM 能力”，说明开放语义是进入家庭/办公场景的前置条件。

### 10.2 开放问题

- 视频动态预训练与 VL 共训练是否互补，还是会竞争模型容量？
- Invalid status 能否扩展为可解释拒绝理由和安全动作，而不只是停住？
- 人类手轨迹如何对齐不同夹爪/灵巧手，本体差异多大时会失效？
- Action DiT 只读 VLM 后半层是否丢失细粒度几何；多层可学习路由是否更优？
- 长程任务需要显式高层规划还是 flat chunk policy 足够？
- 如何把 RL 提升重新蒸馏到通用 base，避免专家化后遗忘？
- 视觉之外是否需要触觉/力觉，特别是布料与滑移？

---

## 11. 材料来源

| 本地材料 | 位置 | 用途 |
|---|---|---|
| 官方技术报告 PDF | `assets/GR-3 - 通用机器人基础模型/sources/paper.pdf` | 全文、公式、实验与局限核对 |
| 来源记录 | `assets/GR-3 - 通用机器人基础模型/sources.md` | PDF、原图和项目页溯源 |
| Figure 1 | `figures/01-teaser.jpeg` | 三类数据与能力总览 |
| Figure 3 | `figures/02-model-architecture.jpg` | 模型架构 |
| Figure 4 | `figures/03-training-data.jpg` | 数据管线 |
| Figure 7 | `figures/04-pick-place-results.jpg` | OOD 与 few-shot |
| Figure 10 | `figures/05-cloth-results.jpg` | 灵巧任务阶段结果 |

> 本笔记基于本地官方 PDF（20 页）逐页阅读。报告未披露的信息均明确标注，未用二手材料补造训练规模。五张图来自 arXiv 官方源码并已逐张检查。关联综述：[[ByteDance GR 系列 - Survey]]。
