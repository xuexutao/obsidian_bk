---
type: paper-note
status: complete
domain: VLA
paper: "Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models"
year: 2026
arxiv: "2608.27550"
doi: null
source: https://arxiv.org/abs/2608.27550
project: https://starvla.github.io/VLAct
code: null
tags:
  - VLA
  - continued-pretraining
  - representation-learning
  - cross-embodiment
  - action-head
created: 2026-10-21
updated: 2026-10-21
---

# VLAct：表示中心的 VLA 继续预训练

## 基本信息

| 项目 | 内容 |
|---|---|
| **正式标题** | *Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models* |
| **核心贡献者** | Senqiao Yang、Chengyao Wang、Yuxin Chen、Zixuan Wang、Longxiang Tang、Haokun Gui、Jinhui Ye、Changsheng Lu、Xiaoyang Wu、Mingkang Zhu 等 |
| **arXiv** | 2608.27550，2026 年 8 月技术报告 |
| **项目页** | <https://starvla.github.io/VLAct> |
| **基础模型** | Qwen3-VL-4B |
| **研究对象** | 不是固定下游 policy，而是经过多本体机器人数据继续预训练、可供不同动作头重新微调的 VLA-oriented VLM backbone |
| **资源声明** | 全开源数据、16 GPU 继续预训练；PDF 未集中披露完整预训练步数/学习率/总 batch 配比 |

## 一句话结论

VLAct 把 VLA 预训练从“用更多轨迹拟合一个动作头”改写成“用固定轨迹预算塑造可迁移 backbone 表示”：它冻结 vision encoder 与下半层 LLM、混入 caption 以保留 VLM prior，用 OFT/PI/GR00T 三种连续动作头共监督避免 decoder lock-in，并以 20 维部分统一动作布局与角度 wrap-aware loss 共享可对齐的跨本体语义；下游丢弃所有预训练动作头、重新初始化目标 head，在完全相同微调协议下仍把 LIBERO-Plus 提到 82.6%、VLA-Arena 54.8%、RoboTwin Clean 92.5%，并用 20% RoboCasa-GR1 数据达到 49.5%，超过 GR00T-N1.6 全数据 47.6%，说明表示配方是独立于数据规模的重要轴，但 4B 单一骨干、部分未披露训练细节和 RoboDojo Memory 0.56% 等失败仍限制其泛化叙事。

---

## 1. 研究背景与问题转移

### 1.1 从“数据不够多”转向“每条轨迹塑造了什么表示”

互联网图文可大规模抓取，机器人轨迹却必须通过真机执行、仿真专家或人类遥操作获得。物理交互空间又由场景、物体、任务、本体、控制接口和接触动力学组合而成，即便几十万轨迹也只是稀疏覆盖。因此 VLA 不能只照搬语言模型的“继续堆数据”逻辑：在固定数据预算下，预训练如何把有限轨迹转化成可复用的视觉—动作知识，成为独立瓶颈。

VLAct 的核心观点是：下游 policy 成功率不仅由轨迹数量决定，也由 action supervision 如何重塑 backbone feature geometry 决定。一个预训练 policy 在同一动作头上表现高，不代表其 backbone 可供另一个动作头读取；一个大机器人数据集带来域内提升，也可能破坏 VLM 的通用视觉语言 prior 或把表示锁死在特定本体。

### 1.2 朴素 VLA 继续预训练的三个失败模式

1. **VLM prior 漂移**：机器人数据视觉分布窄、相机固定、场景重复，全参数 action-only 更新会覆盖 web-scale 视觉语言能力。
2. **单动作头过专化**：backbone 与一个 decoder 联合优化后，动作信息可能沿着该 head 最容易读取的方向组织，其他 head 难以解码，即 decoder lock-in。
3. **本体语义割裂**：每个机器人独立 head 会把物理上相同的 gripper open/close 放进互不相交的空间，减少跨本体共享；但简单把不同关节硬 padding 到相同坐标又会制造错误对齐。

![](assets/VLAct%20-%20表示中心的VLA继续预训练/figures/01-teaser.png)

> 图 1（论文 Figure 1，p.3）：上方的 naive pre-training 只用单动作头拟合预训练场景，难以迁移到新任务、环境、本体和 head；下方 VLAct 把 shallow-layer protection、caption mixing、多动作头和统一动作表示作为“塑造 backbone”的训练脚手架。右侧结果同时覆盖模拟与真机。来源：官方 arXiv 源码与 PDF，<https://arxiv.org/abs/2608.27550>。

### 1.3 “继续预训练”的精确定义

VLAct 不从头训练 foundation model。它从已预训练 Qwen3-VL-4B 出发，先在广泛、多本体机器人轨迹和辅助 VLM 数据上进行 continued pre-training，再针对下游任务微调。预训练阶段的动作头、统一输出布局和 caption stream 都是脚手架；下游只继承 backbone 权重，动作头重新随机初始化。这个 protocol 是论文归因成立的关键：若复用预训练动作头，就无法区分 backbone representation 与 head 已适配目标动作分布的收益。

---

## 2. Pilot study：动作监督不是中性的

### 2.1 离散 FAST：能迁移，但有信息损失

在 LIBERO-Plus 上，FAST 预训练的 backbone 换成连续 GR00T head 后，比从零 GR00T 略有提升：75.9 对 75.9 附近并显示 +0.8 的 transfer gain；但继续使用 FAST head 仅 45.2，相比从零 FAST 61.4 还低 16.2。论文解释为离散 token 可注入粗粒度动作结构，却损失连续幅值与时间细节；“能转给连续 head”与“自己下游表现好”并不矛盾。

### 2.2 单连续 OFT：同头提升、跨头崩塌

RoboTwin 上，从零 OFT 为 61.7，OFT 预训练再 OFT 微调达到 75.8，提升 14.1；但 OFT 预训练换 PI 只有 55.1，比从零 PI 60.5 低 5.4；换 GR00T 仅 28.9，比从零 GR00T 51.2 低 22.3。说明 action knowledge 不是简单“存在/不存在”，其几何组织方式会对特定 decoder 友好、对其他 decoder 不可读。

![](assets/VLAct%20-%20表示中心的VLA继续预训练/figures/03-head-transfer.png)

> 图 2（论文 Figure 2，p.4）：FAST 的问题是离散化损失，OFT 单头预训练的问题是 decoder lock-in。该 pilot study 直接动机是让多个连续 head 同时读取同一 latent，迫使 backbone 暴露更通用的动作信息。来源：官方 arXiv 源码与 PDF。

### 2.3 Pilot 的方法学意义

如果只报告 OFT→OFT 的 75.8，会得出“预训练很有效”；加入 OFT→GR00T 的 28.9 后，结论变成“预训练形成了强但专用的表示”。因此评价 VLA backbone 不能只看 matched-head fine-tuning，必须测试 unseen-head transfer、same-head adaptation 和未见本体。这也是 VLAct 与一般 end-to-end policy 论文的关键区别。

---

## 3. 输入、输出与两阶段训练任务

### 3.1 继续预训练阶段

输入包括多视角 RGB、自然语言任务指令和机器人轨迹；caption 样本则输入 web/image-caption 对。机器人样本经 Qwen3-VL backbone 得到共享 latent $z$，同时送入 OFT、PI、GR00T 三个连续动作头，预测同一 action chunk。动作先映射到每步 20 维部分统一空间；不属于当前本体的维度 mask 掉。caption 样本使用 VLM next-token cross entropy。

预训练动作总损失为：

$$\mathcal{L}_{\mathrm{action}}=\mathcal{L}_{\mathrm{OFT}}+\mathcal{L}_{\mathrm{PI}}+\mathcal{L}_{\mathrm{GR00T}}$$

加入辅助 VLM 数据时：

$$\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{action}}+0.5\mathcal{L}_{\mathrm{VLM-CE}}$$

对绝对旋转关节，还为每个 head 的最终动作预测加入 wrap-aware loss。

### 3.2 下游微调阶段

丢弃 OFT/PI/GR00T 预训练 head、caption stream 与预训练特定输出层，只保留 VLAct backbone；根据下游任务新建 OFT、PI 或 GR00T head 并随机初始化，完整解冻模型，在与 baseline 完全相同的数据、optimizer、step 和动作格式下微调。因此实验中的“唯一变化”是 backbone 初始化，论文把 7.6–21.4 点增益归因于表示质量。

### 3.3 输出空间不是单一机器人控制格式

Franka 数据使用 6D delta EEF + 1 gripper；AgileX 使用双臂 12 个 absolute joint + 2 gripper。VLAct 不把两者强行转成相同控制器，而是在 20 维 canonical layout 中按角色分槽：

| 维度 | 语义 |
|---|---|
| 1–6 | AgileX 左臂 6-DoF absolute joint |
| 7–12 | AgileX 右臂 6-DoF absolute joint |
| 13–18 | Franka 单臂 6D delta EEF |
| 19 | 共享 gripper：Franka 单夹爪 / AgileX 左夹爪 |
| 20 | AgileX 右夹爪 |

只有 gripper 等物理意义可对齐的维度共享；不兼容的 arm dimensions 分槽并 mask。它比 StarVLA-$\alpha$ 的全 32 维 zero padding 更强调语义正确性，又比每本体独立 head 提供更多共享监督。

---

## 4. 核心机制一：保留 VLM prior

### 4.1 Shallow-layer protection

继续预训练时冻结整个 vision encoder 和 LLM 下半层，只更新 LLM 上半层与动作头。论文的 attention visualization 显示浅层更广泛关注视觉与空间信息，深层更集中于语义/任务区域。冻结浅层的目的不是减少算力本身，而是保护最容易被窄机器人分布破坏的感知与早期视觉语言对齐；下游微调时再全部解冻。

消融结果：全 backbone 更新在 LIBERO-Plus/RoboTwin Base 为 78.9/77.1；只冻结 vision encoder 为 81.3/79.3；再冻结下半 LLM 为 82.6/80.5。相对全更新分别提升 3.7 和 3.4 点。

### 4.2 Caption-mixed pre-training

caption 对物体、属性、空间关系和场景提供稠密语义梯度，分布也更接近原始 VLM 训练。论文还比较 BBox-QA、Point-QA、Spatial-QA 和纯语言 instruction；所有辅助源都优于 robot-only，caption 单源最好，将 LIBERO-Plus 从 75.0 提到 82.6。多源 mixture 为 82.5，略低于 caption-only，作者解释为固定 step budget 下 caption 采样频率降低，而非混合本身有害。

辅助数据包括 LLaVA-ReCap-CC3M、LLaVA OneVision/文中分析提及 ShareGPT4V、RefCOCO、COCO-ReM、PixMo-Points、RoboPoint、SenseNova-SI-800K 和 Nemotron instruction 数据。正文与附录对个别 caption 源命名存在版本痕迹，复现应以公开配置与具体 commit 为准。

### 4.3 表示保护不是“冻结越多越好”

论文只比较全更新、冻 vision、冻 vision + 下半 LLM，没有扫过 1/4、3/4、layer-wise LR 或 adapter。4B 模型上最优的冻结比例不一定适用于更大 backbone。作者在 limitations 中也明确说只研究 4B，规模变化可能改变最优配方。

---

## 5. 核心机制二：多连续动作头共监督

### 5.1 OFT、PI、GR00T 的互补约束

OFT 是一次并行 L1 回归。若第 $k$ 个 action query hidden state 为 $h^{\mathrm{act}}_{t,k}$：

$$\hat{a}_{t+k}=g_\phi(h^{\mathrm{act}}_{t,k})$$

$$\mathcal{L}_{\mathrm{OFT}}=\frac{1}{Kd_a}\sum_{k=0}^{K-1}\left\|g_\phi(h^{\mathrm{act}}_{t,k})-a_{t+k}\right\|_1$$

PI 采用 conditional flow matching。采样 $\epsilon\sim\mathcal{N}(0,I)$ 与 $\tau\in[0,1]$，构造：

$$A_t^\tau=(1-\tau)\epsilon+\tau A_t$$

目标速度为 $A_t-\epsilon$：

$$\mathcal{L}_{\mathrm{PI}}=\mathbb{E}\left[\left\|v_\phi(A_t^\tau,\tau;H_t)-(A_t-\epsilon)\right\|_2^2\right]$$

GR00T 也是 flow matching，但在更明确的 dual-system DiT motor module 中加入 robot state $s_t$ 与 embodiment $e$：

$$\mathcal{L}_{\mathrm{GR00T}}=\mathbb{E}\left[\left\|v_\phi(A_t^\tau,\tau;H_t,s_t,e)-(A_t-\epsilon)\right\|_2^2\right]$$

三者同时作用于共享 backbone latent。由于 decoder bias 不同，backbone 很难只编码一个 head 的捷径，被迫保留多个参数化都能读取的动作信息。

### 5.2 Unseen-head 与 same-head 证据

以 PI 为未见下游 head：无预训练 60.5；OFT-only 55.1；OFT+GR00T（预训练未见 PI）63.1；OFT+PI+GR00T 77.0。关键对比是 55.1→63.1，说明增加第二个 head 能缓解 lock-in；77.0 因 PI 已在预训练见过，包含直接 compatibility，不应与 unseen-head 结论混为一谈。

same-head 微调中，OFT 单头/多头为 78.8/80.5，PI 为 75.4/77.0，GR00T 为 71.7/76.0。多头没有以牺牲 matched-head 为代价，反而提高 1.6–4.3 点，支持其作为 representation regularizer。

### 5.3 计算成本

三个 head 共用一次 backbone forward，额外成本主要是 head-specific MLP/DiT，而不是把 Qwen3-VL 跑三遍。但 PI/GR00T 的动作模块仍会增加显存与训练 FLOPs。论文给出“16 GPU 可完成”这一总体级别，却未在 PDF 中逐项报告单头与三头的 wall-clock、显存和能耗，因此复现成本只能做定性评估。

---

## 6. 核心机制三：部分统一动作空间与 wrap-aware loss

### 6.1 为什么不能只用独立 head 或朴素 padding

独立 head 灵活，但相同 gripper 语义不能共享；朴素统一把第 $i$ 维对齐，却可能让 Franka EEF translation 与 AgileX joint angle 占同一槽位，产生错误监督。VLAct 只共享物理语义一致维，其他维度保持本体角色分槽并 mask，无需额外 router 或 embodiment adapter。

统一表示消融：separate heads 在 RoboTwin/LIBERO-Plus 为 78.5/81.1；unified head 79.5/81.4；完整 unified action representation 80.5/82.6。收益不算巨大但两项一致，说明正确的输出语义比单纯共享参数更有效。

### 6.2 角度 wrap

绝对关节角先映射到 canonical range：

$$a_{\mathrm{wrap}}=(a+\pi)\bmod 2\pi-\pi$$

预测与目标残差也在圆周上计算：

$$\delta_{\mathrm{wrap}}=((\hat{a}-a)+\pi)\bmod 2\pi-\pi$$

$$\mathcal{L}_{\mathrm{wrap}}=|\delta_{\mathrm{wrap}}|$$

每个 head 的总目标为：

$$\mathcal{L}^{(h)}_{\mathrm{total}}=\mathcal{L}^{(h)}_{\mathrm{action}}+\mathcal{L}^{(h)}_{\mathrm{wrap}}$$

wrap 只用于 absolute periodic joint，不用于 gripper 或 delta EEF。RoboTwin Base 中直接 raw joint 回归为 75.5，数据侧统一角空间 78.6，再加 wrap loss 达 80.5。这个 +5.0 点说明动作参数化的几何正确性仍然重要；它也修正了对 StarVLA-$\alpha$“数据工程不重要”的粗糙解读：无意义的技巧可被强 VLM 吸收，但物理拓扑错误不会自动消失。

### 6.3 数据清洗与动作对齐

VLAct 删除空任务名、`unknown_task`、`n/a`、`none`、`no action` 等无效语言；delta EEF 按 FPS 转成每秒动作，translation velocity 超过 0.5 或 rotation velocity 超过 1.0 的 step 标无效，chunk 内无效比例 $r_\mathcal{C}>0.5$ 才丢整个 chunk，其余只 mask 异常 step。有效 translation/rotation 再分别除以 0.5/1.0。absolute joint 超出 $[-2\pi,2\pi]$ 被过滤，其余 wrap 到 $[-\pi,\pi]$；gripper 按数据集 min-max 到 $[0,1]$。

这套处理并非“无数据工程”，而是把工程集中到物理一致性与噪声治理，而非为每个 benchmark 添加模型捷径。

![](assets/VLAct%20-%20表示中心的VLA继续预训练/figures/02-method.png)

> 图 3（论文 Figure 3，p.6）：预训练阶段四个关键编号依次是保护浅层、caption 混训、多动作头共监督、统一动作空间与 wrap-aware loss；微调时重新初始化目标 head。图清楚区分了“训练脚手架”和“实际继承的 backbone”。来源：官方 arXiv 源码与 PDF。

---

## 7. 数据、训练与实现细节

### 7.1 机器人数据

Franka 单臂使用 DROID v1.0.0/v1.0.1 与 MolmoAct，动作 6D delta EEF + 1 gripper；AgileX 双臂使用 InternData-A1 与 RoboCoin，动作 12 absolute joints + 2 grippers。下游未见本体包括 RoboCasa-GR1 的 GR-1 humanoid 与 RoboDojo 的 ARX X5。

论文还加 20K 条 10Kh-RealOmin-OpenData UMI 风格异构轨迹，转为 delta EEF、其余设置不变，使 LIBERO-Plus 从 82.6 到 83.7，说明配方可吸收新的收集范式。但这只是 +1.1 点单实验，不能独立证明无限扩数据都线性有效。

### 7.2 继续预训练资源

论文声明 Qwen3-VL-4B、全开源数据和 16 GPU，强调相较工业系统资源适中；然而 PDF 没有集中给出精确 pre-training steps、learning rate、总 batch、各机器人数据采样比例与 GPU 型号。项目称模型与 pipeline 已开源，但论文未列独立 repo URL，因此 frontmatter 的 `code` 保留 `null`，以项目页为官方入口。

### 7.3 真机微调

Franka Research 3 使用外部 RealSense D435 和腕部 D405，图像 resize 到 $224\times224$；GELLO 遥操作收集。每个单臂任务 50 条 demonstration，每个双臂任务 100 条；VLAct 与 Qwen3VL-OFT baseline 使用完全相同数据。单臂和双臂各训练一个模型，50K step、8×H800；每任务固定 10 个共享初始状态进行评测。

---

## 8. 主要 benchmark 结果

### 8.1 LIBERO-Plus

VLAct 82.6，总分高于同骨干同 OFT 下游 head 的 Qwen3VL-OFT 75.0（+7.6），也高于 ABot-M0 80.5。分维度 Camera/Robot/Language/Light/Background/Noise/Layout 为 73.9/68.4/81.5/96.7/96.7/86.0/83.3。相对 baseline 最大收益集中在 camera、robot、noise、layout，符合“表示鲁棒性”而非只记住动作的解释。

### 8.2 VLA-Arena

Safety/Distractor/Extrapolation/Long-Horizon/Avg 为 63.2/64.0/36.9/50.0/54.8。Qwen3-VL-OFT 平均 33.4，$\pi_{0.5}$ 44.3。相对 $\pi_{0.5}$，Long-Horizon 提高 21.0 点、Safety 提高 11.7 点。论文把 21.4 写成相对同骨干 baseline 的“点数/百分比”提升，阅读时应理解为 54.8−33.4=21.4 个百分点。

### 8.3 RoboTwin 2.0

Base 仅每任务 50 clean trajectory：Qwen3VL-OFT 61.7，VLAct-OFT 80.5、PI 77.0、GR00T 76.0。Data Scaling 再加每任务 500 randomized：VLAct-OFT clean/random 为 92.5/90.8，PI 93.0/88.8，GR00T 89.6/87.4。HoloBrain-0 Random 为 92.3、Fast-WAM 91.8，因此论文谨慎说“最强之一”而避免绝对 SOTA。

### 8.4 DOMINO

35 个动态 manipulation 任务上，VLAct-OFT Success Rate 18.50、Manipulation Score 34.20；Qwen3VL-OFT 10.86/30.49，$\pi_{0.5}$ 9.63/26.17。SR 提高 7.64 点，说明 continued pre-training 不只适用于静态桌面场景，但绝对成功率 18.5% 仍低，动态控制远未解决。

### 8.5 未见本体 RoboCasa-GR1

VLAct continued pre-training 只见 Franka 和 AgileX，未见 GR-1。下游 10%/20%/50%/100% 数据得到 41.42/49.5/51.0/54.0。20% 数据 49.5 已超过 GR00T-N1.6 全数据 47.6，并略高于 Qwen3VL-OFT 全数据 48.8；$\pi_{0.5}$ 全数据 37.0。该实验是论文最强的 cross-embodiment 证据，因为预训练本体与目标 humanoid 形态、动作空间都不同。

### 8.6 RoboDojo 外部 leaderboard

42 个 ARX X5 任务、35 个 policy 的 2026-08-24 snapshot 中，VLAct 平均 partial score 10.66、success 7.60%，按 score 第 8、按 success 第 6；超过所有明确标注 WAM 的条目。它相对 StarVLA-$\alpha$ 的 6.40/3.24 提高 4.26/4.36 点。最明显短板是 Memory 0.66 score / 0.56% success；Open 也仅 2.37/2.25。排行榜不统一训练数据与算力，适合说明外部竞争力，不适合做严格因果比较。

---

## 9. 真机结果与失败模式

![](assets/VLAct%20-%20表示中心的VLA继续预训练/figures/04-realworld-transfer.png)

> 图 4（论文 Figure 5，p.12）：(a) 单臂短程，(b) 单臂长时，(c) 双臂协同，(d) 未见 GR-1 本体数据效率。灰色为无 continued pre-training 的 Qwen3VL-OFT baseline。来源：官方 arXiv 源码与 PDF。

### 9.1 单臂短时与 OOD

ID 四任务平均 92.5%，baseline 77.5%。cube stacking 90 vs 60，pen-in-cup 80 vs 60。失败分析称 baseline 在未覆盖起始位置会 freeze，抓取失败后产生 cascading error；VLAct 对后方 cup 定位与恢复更稳定。novel object from pot / in cup 均 90.0，baseline 分别 73.3/65.0。

### 9.2 长时任务

Table cleaning 86.6 vs 73.3，scooping beans 80.0 vs 33.3。baseline 常跳过“scoop”直接空勺倾倒，说明不是纯定位问题，而是任务阶段逻辑丢失。extended-sequence 与 full-substitution OOD 中 VLAct 82.5/83.3，baseline 47.5/46.6。

### 9.3 双臂协同

虽然 continued pre-training 只用单臂 Franka 与 AgileX 数据混合，真机双臂任务平均 72.0，baseline 44.0。unplugging 80 vs 60，breakfast preparation 90 vs 70，fold pants 70 vs 40，fold towel 50 vs 20。baseline 的典型失败包括布料抓点不准导致碰桌、双臂速度不同步导致毛巾滑脱。即便 VLAct 改善明显，fold towel 50% 仍表明 deformable-object coordination 是开放难题。

---

## 10. 消融总结与因果链

### 10.1 表示保护

全更新 78.9/77.1；冻 vision 81.3/79.3；冻 vision + 下半 LLM 82.6/80.5。caption-only 把 LIBERO-Plus 75.0 提到 82.6，mixed auxiliary 82.5。证据链是：窄动作数据会漂移 → 保护浅层与补充原生 VLM 监督 → 下游鲁棒性提升。

### 10.2 动作头多样性

OFT-only→PI 为 55.1，OFT+GR00T→PI 为 63.1；完整三头→PI 为 77.0。same-head 也提升 1.6–4.3。证据链是：单头会 lock-in → 多 decoder 强迫共享 latent 可读 → unseen/same head 均改善。

### 10.3 动作空间几何

separate heads 78.5/81.1，unified head 79.5/81.4，语义统一 80.5/82.6；raw joint 75.5，joint wrapping 78.6，再加 wrap loss 80.5。证据链是：共享参数不足 → 需共享正确物理语义并尊重角度周期拓扑。

### 10.4 数据扩展仍有价值

额外 20K UMI 轨迹带来 +1.1。VLAct 并非反对 scaling，而是主张 scaling 与表示配方互补。其标题“Beyond Data Scaling”应理解为“除了数据规模之外”，不是“无需更多数据”。

---

## 11. 局限与批判性评价

### 11.1 作者明确限制

作者在 Future Work 中明确：资源约束使研究集中于 4B VLM，没有探索显著更大模型；更大模型的视觉、空间和语言 prior 更强，最优 freezing、辅助数据与 head diversity 配方可能变化。未来应构建更适合物理任务、跨任务/环境/本体泛化的 VLA-oriented VLM。

### 11.2 复现缺口

PDF 未集中披露 continued pre-training 的完整 step、learning rate、总 batch、GPU 型号、机器人数据精确比例与 caption sampling ratio。项目声明开源，但论文没有独立 repo URL。核心结果可理解，精确复现仍需项目配置、checkpoint metadata 与 commit。

### 11.3 方法边界

- 只研究 Qwen3-VL-4B，不能确定 Llama/PaliGemma/video backbone 或更大模型是否同样受益。
- 三个连续 head 都来自同一动作生成家族范围；“head-agnostic”未覆盖 token policy、autoregressive continuous、hierarchical planner 等更广 head。
- 部分统一空间目前只覆盖 Franka 与 AgileX 角色布局，扩到移动底盘、灵巧手、力矩控制时 20 维 schema 需重设计。
- caption-only 最优可能受固定 step budget 影响，尚未把“数据类型”和“每类样本量”完全解耦。
- RoboDojo Memory 几乎失败，说明当前表示中心配方没有解决长期状态记忆。
- DOMINO SR 18.5%、RoboDojo SR 7.6%、fold towel 50%，绝对能力仍离通用机器人很远。

### 11.4 我的判断：最重要的是改变预训练评价单位

VLAct 把评价对象从“预训练 policy 的 matched-head 成功率”改为“backbone 被新 head、新本体、新任务重新读取的能力”。这是比某个单项 SOTA 更耐久的贡献。其风险在于表示质量仍主要通过下游 success 间接衡量，缺少 CKA、effective rank、linear probe 或因果干预等更直接 representation analysis；附录注释甚至列出这些可选分析，但最终未报告。

---

## 12. 复现成本与实践建议

### 12.1 资源层级

- **完整 continued pre-training**：Qwen3-VL-4B、三连续 head、机器人 + caption 双流、16 GPU；具体 wall-clock 未披露。
- **下游模拟微调**：RoboTwin 可按 Base 50×50 先验证，Data Scaling 需 27.5K 级轨迹与大规模 rollout。
- **真机**：Franka Research 3、D435 + D405、GELLO；单臂每任务 50 demo、双臂 100 demo、8×H800 训练 50K step。
- **评测**：LIBERO-Plus、VLA-Arena、RoboTwin、DOMINO、RoboCasa、RoboDojo 各有独立依赖与协议。

### 12.2 最小机制验证

资源有限时可做三个最小实验：

1. Qwen3-VL-OFT 与 shallow protection + caption 的 LIBERO-Plus 对比；
2. OFT-only 与 OFT+GR00T 预训练后换未见 PI head，在 RoboTwin Base 比较 55.1 vs 63.1 趋势；
3. raw joint、wrapped joint、wrap loss 三阶消融，验证 75.5→78.6→80.5。

这三项分别覆盖 prior preservation、decoder lock-in 与物理动作几何，能验证论文主张而不重跑所有 leaderboard。

### 12.3 必须记录的复现元数据

仓库 commit、Qwen3-VL checkpoint、冻结层索引、三 head 参数量、robot/caption sampling ratio、action chunk、20 维 mask、FPS、异常阈值、gripper normalization、预训练 step、optimizer/LR、下游 head 随机种子、checkpoint selection 和 evaluator 版本。缺少其中任一项都可能使“backbone only changes”的归因失效。

---

## 13. 与 StarVLA、StarVLA-$\alpha$ 的关系

三篇工作**不是同一个模型的三个版本**，而是共享团队与代码生态的三个研究层次：

1. **StarVLA**：基础设施层。建立统一原始输入、`forward`/`predict_action`、骨干—动作头、训练器、benchmark server-client 和部署接口。
2. **StarVLA-$\alpha$**：受控 baseline 层。用 Qwen3-VL + MLP 证明强 VLM 下很多复杂度收益有限，并暴露“域对齐预训练提升域内、损害未见本体”的矛盾。
3. **VLAct**：representation-centric continued pre-training 层。它不追求固定下游 policy，而是针对 $\alpha$ 暴露的负迁移和动作头依赖，设计可复用 backbone。

VLAct 使用 StarVLA codebase，也复用 OFT/PI/GR00T 等动作头作为训练脚手架；但下游 checkpoint 不是 StarVLA-$\alpha$ 简单改名。$\alpha$ 的默认模型是未经 action-specific pretraining 的 Qwen3-VL + MLP；VLAct 则继承经过多本体、caption、多 head 和统一动作空间继续预训练的 backbone，再新建下游 head。

这条路线的“问题演化”是：先让系统可比较，再用简洁 baseline 找出真正问题，最后把研究重点移到表示如何跨 decoder、本体和任务复用。

---

## 14. 个人启发与后续问题

### 14.1 预训练 head 应被视为表示塑形器

动作头不仅消费 backbone feature，也通过梯度决定 feature 的几何。未来任何 VLA 预训练都应报告 matched-head 与 mismatched-head transfer，否则“强预训练”可能只是 decoder compatibility。

### 14.2 冻结与辅助数据是信息保持策略

冻结浅层和 caption mixing 可以理解为两个互补约束：前者减少参数漂移自由度，后者给可训练深层提供原始分布锚点。更系统的方法可使用 feature distillation、Fisher/EWC、低秩更新、layer-wise LR 或 representation alignment，而非固定冻结一半。

### 14.3 动作空间需要尊重流形

角度在圆上而不在实线上，179° 与 −179° 物理接近。wrap-aware loss 的 5 点收益提醒：再强的 foundation model 也不能替代正确的目标几何。对 SE(3)、quaternion、接触 mode 和双臂相对位姿，也应使用相应流形距离或等价类表示。

### 14.4 最值得继续验证的问题

- head diversity 的收益来自梯度正交、ensemble regularization 还是增加计算量？
- 不共享 PI 直接曝光时，OFT+GR00T→PI 仅 +2.6，能否通过显式 latent alignment 放大 unseen-head transfer？
- caption 最优是否只是样本密度和训练分布接近造成？在固定 caption token 数后 grounding 是否仍有补充收益？
- Memory 失败能否通过 recurrent state、episodic memory 或 video temporal objective 改善？
- 20 维部分统一布局扩展到十几种本体时，手工 schema 是否会成为新瓶颈？
- 4B→更大模型后，冻结比例、辅助 loss 权重 0.5 和三 head 组合是否仍最优？

### 14.5 最终评价

VLAct 最有说服力的不是单个 92.5%，而是受控 protocol：同下游数据、head、初始化方式和训练预算，只换 backbone，仍在多个 benchmark、真机和未见本体上稳定提升；同时 pilot 与消融给出“为什么”的机制证据。其不足是预训练细节披露不够集中、4B 范围窄、表示分析仍主要依赖下游指标。总体上，它把 VLA 研究从“继续加数据/做更大 end-to-end policy”推进到“研究 backbone 如何保存、组织和共享动作知识”，是 StarVLA 系列中方法论最明确的一步。

---

## 15. 材料来源

- 官方论文：<https://arxiv.org/abs/2608.27550>
- 官方项目页：<https://starvla.github.io/VLAct>
- 本地来源、页码与原图追溯：`assets/VLAct - 表示中心的VLA继续预训练/sources.md`
- 本笔记数字、公式、消融与限制均依据本地官方 PDF 及官方 LaTeX 源码核对；论文未披露的训练细节未补造。
