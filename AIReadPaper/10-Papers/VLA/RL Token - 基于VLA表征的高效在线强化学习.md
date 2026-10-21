---
type: paper-note
status: complete
domain: VLA
paper: "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"
year: 2026
arxiv: 2604.23073
doi: 10.48550/arXiv.2604.23073
source: https://arxiv.org/abs/2604.23073
project: https://pi.website/research/rlt
code: null
tags:
  - VLA
  - Physical-Intelligence
  - Online-RL
  - Representation-Learning
  - Actor-Critic
  - Action-Chunking
  - Human-Intervention
created: 2026-10-20
updated: 2026-10-20
---

# RL Token：基于 VLA 表征的高效在线强化学习

## 基本信息

- **作者**：Charles Xu、Jost Tobias Springenberg、Michael Equi、Ali Amin、Adnan Esmail、Sergey Levine、Liyiming Ke
- **机构**：Physical Intelligence
- **发表**：arXiv 2604.23073v2（2026）
- **项目页**：<https://pi.website/research/rlt>
- **原文 PDF**：<https://arxiv.org/pdf/2604.23073>
- **基础模型**：π0.6（SigLIP 400M + Gemma 4B VLM，约 860M action expert）
- **核心简称**：RLT（Reinforcement Learning from RL Token）

## 一句话结论

RLT 不用 RL 更新整个数十亿参数 VLA，而是先用 encoder-decoder 把 π0.6 的内部 token 压成一个 **2048 维 RL token**，冻结 VLA 后训练小型 chunk-level TD3 actor-critic，并用 VLA 参考动作作输入与行为正则；它用 15 分钟到 5 小时真实机器人数据就能在螺丝、扎带、网线和充电器精密操作上显著提速增稳，关键阶段最高约 3× 加速，螺丝成功率从 20% 提到 65%，甚至在网线插入上快于专家遥操作。

---

## 1. 研究背景与问题定义

### 1.1 “最后一毫米”问题

通用 VLA 能开箱完成抓取、搬运和长程任务，但在螺丝刀尖对准螺丝、扎带尾穿过窄锁孔、RJ45 插头对准凹陷端口等最后几毫米常出现试探、回退和重复尝试。示范很难稳定覆盖这些高精度接触状态，而 RL 可以在现场练习中直接优化成功与速度。

问题在于真实机器人预算极紧：一次 episode 要几十秒到两分钟，失败会磨损硬件；直接对 4B+ VLA 做在线 RL 既计算昂贵又样本低效。反过来，SERL 等高效真机 RL 使用小 ResNet 策略，能几小时学会，却丢掉 VLA 的语义、视觉和行为先验。

### 1.2 核心问题

**怎样保留大 VLA 的感知与动作知识，同时让在线 RL 只优化一个足够小、足够稳定、能在数分钟至数小时内学会的接口？** RLT 的答案是：把 VLA 当冻结的表征器与行为 prior，压缩出 RL token；让小 actor 只在 VLA 建议附近编辑 action chunk，而非从像素和高维动作空间从零探索。

### 1.3 核心贡献

1. 用 encoder-decoder reconstruction 学出单个 RL token，压缩 VLA 内部视觉表征并保留任务相关信息。
2. 在 chunk-level 状态—动作空间训练轻量 off-policy actor-critic，缩短 50 Hz 稀疏奖励任务的 credit-assignment horizon。
3. Actor 同时接收 VLA 参考 chunk，并用 L2 BC regularizer 锚定，兼顾稳定探索和超越示范。
4. 在 4 个高精度真机任务、critical phase 与 full task 两类设置中，与 HIL-SERL、PLD、DSRL、DAgger 及关键组件消融比较。

---

## 2. 任务定义与输入输出

### 2.1 VLA 接口

π0.6 输入最多 4 路相机图像、语言指令 $\ell$ 和本体状态 $s_t^p$，生成长度 $H=50$ 的动作 chunk：

$$\widetilde a_{t:t+H-1}\sim\pi_{\mathrm{vla}}(\cdot\mid s_t,\ell),
\qquad \widetilde a\in\mathbb R^{H\times d}.$$

50 Hz 下 $H=50$ 对应 1 秒，基础 VLA 通常开环执行前 20 步再重规划。RLT 的 actor 使用更短 $C=10$，以提高反应性；双臂每步 14 维，因此 actor 输出 140 维 chunk。

### 2.2 Chunk-level MDP

状态输入为

$$x_t=(z_t^{\mathrm{rl}},s_t^p),$$

其中 $z_t^{\mathrm{rl}}\in\mathbb R^{2048}$ 是 RL token。策略输出 $a_{t:t+C-1}$。论文只有稀疏二元终点奖励：成功 $r_T=1$，失败为 0。Chunk-level critic 估计

$$Q^\pi(x_t,a_{t:t+C-1})=
\sum_{t'=t}^{t+C-1}\gamma^{t'-t}r_{t'}+
\gamma^C\mathbb E_{a'\sim\pi}[Q^\pi(x_{t+C},a')].$$

将一次 Bellman 决策跨越 $C$ 个控制步，相比 $C=1$ 将有效 horizon 缩短约 10 倍，这对数百至数千步后才出现奖励的任务尤为重要。

---

## 3. 核心方法

### 3.1 总体架构与分工

![](assets/RL%20Token%20-%20基于VLA表征的高效在线强化学习/figures/01-overview.png)

> **图 1（原论文 Figure 1，第 1 页）：RLT 总流程。** 任务数据先适配 VLA 并训练 RL token；冻结后，token 送入 critic，token 与 VLA action chunk 送入 actor；小型在线 RL 网络在螺丝、扎带、网线和电源插入上做局部精调。来源：<https://arxiv.org/abs/2604.23073>

RLT 把职责拆开：冻结 VLA 负责视觉语言理解、强初始动作和多模态行为 prior；小 actor/critic 负责目标任务最难阶段的快速在线优化。这样在线更新不需要反传穿过 4B VLM 和 860M action expert。

### 3.2 用重构瓶颈学习 RL token

令 VLA 最后一层 token 为

$$z=f(s,\ell;\theta_{\mathrm{vla}})=\{z_1,\ldots,z_M\}.$$

在序列末附加可学习 embedding $e_{\mathrm{rl}}$，送入轻量 encoder transformer $g_\phi$，取特殊位置输出：

$$z^{\mathrm{rl}}=g_\phi([z_{1:M},e_{\mathrm{rl}}])_{M+1}.$$

再用 decoder $d_\phi$ 从该 bottleneck 自回归重构原 VLA embeddings。记 $\bar z_i=\operatorname{sg}(z_i)$ 为 stop-gradient，目标为

$$\mathcal L_{\mathrm{ro}}(\phi)=
\mathbb E_D\left[
\sum_{i=1}^{M}
\left\|h_\phi(d_\phi([z^{\mathrm{rl}},\bar z_{1:i-1}]))_i-
\bar z_i\right\|_2^2
\right].$$

重构迫使单个 2048 维 token 保存足以解释 VLA 全部输入 embedding 的信息，而不是任取某层或做平均池化。训练用少量任务示范，可同时以权重 $\alpha$ 监督微调 VLA；之后 encoder 和 VLA 一并冻结。

论文任务语言固定，因此实验中重构阶段去掉语言 embedding；这意味着当前结果更接近“任务专用 RL state”，尚未证明一个 token 可以在开放词汇任务间共享。

![](assets/RL%20Token%20-%20基于VLA表征的高效在线强化学习/figures/02-architecture.png)

> **图 2（原论文 Figure 2，第 4 页）：RL token 提取细节。** π0.6 的 image embeddings 经 encoder 压成 $1\times2048$ token，decoder 以它重构完整 embedding 序列；action expert 仍独立生成连续参考 chunk。来源：<https://arxiv.org/abs/2604.23073>

### 3.3 Chunk critic

Replay buffer $\mathcal B$ 混合 VLA warmup、在线 actor rollout 与人类 intervention。Critic 使用 TD3 风格 target network：

$$\mathcal L_Q(\psi)=
\mathbb E_{(x,a_{1:C},x')\sim\mathcal B}
\left[\left(\widehat Q-Q_\psi(x,a_{1:C})\right)^2\right],$$

$$\widehat Q=
\sum_{t'=1}^{C}\gamma^{t'-1}r_{t'}+
\gamma^C\mathbb E_{a'\sim\pi_\theta}
[Q_{\psi'}(x',a')].$$

Off-policy 属性允许所有历史策略、基础 VLA 和人类纠正被重复利用。系统 rollout 与学习异步进行，每次 actor 更新配两次 critic 更新，并采用较高 update-to-data ratio 提高样本效率。

### 3.4 参考动作条件化 actor

Actor 不从零生成，而是输入状态 $x$ 与 VLA 采样的前 $C$ 步参考 chunk $\widetilde a_{1:C}$：

$$\pi_\theta(a_{1:C}\mid x,\widetilde a_{1:C})=
\mathcal N\left(\mu_\theta(x,\widetilde a_{1:C}),\sigma^2I\right).$$

直接传入参考动作有两个作用：一是告诉 actor 一个强初始解；二是保留 VLA 多模态采样选择的 mode，否则单峰 Gaussian actor 很难从 token 单独恢复“当前决定采用哪种动作策略”。

Actor 目标为

$$\mathcal L_\pi(\theta)=
\mathbb E_{x\sim\mathcal B,\,a\sim\pi_\theta}
\left[-Q_\psi(x,a_{1:C})+
\beta\|a_{1:C}-\widetilde a_{1:C}\|_2^2\right].$$

第一项推动高回报，第二项把探索锚定在 VLA 动作附近。它与 KL-regularized RL 精神相同，但对固定方差 Gaussian 直接使用 L2。$\beta=0$ 时 actor 仅凭稀疏 Q 梯度在 140 维动作空间搜索，消融中性能下降最大。

### 3.5 Reference action dropout

参考动作与 BC 正则都鼓励复制 VLA，训练早期 critic 尚不可靠时 actor 可能永远学不会独立生成。因此训练时对 **50%** 样本把参考 chunk 置零，强迫 actor 建立独立通路；推理始终提供参考。Critic 有效后，actor 才在 Q 值支持下偏离参考。

### 3.6 Warmup、干预和 replay

在线训练先用基础 VLA 跑 $N_{\mathrm{warm}}$ 步填充 buffer。之后每个 chunk 边界由 VLA 产生参考和 RL token，actor 生成 $C$ 步动作；安全或接触困难时人可 teleoperate 覆盖 actor。若发生 intervention，人类动作同时替换 replay 中的执行动作和参考动作，使其成为可学习的局部安全先验。

为充分利用 50 Hz 数据，论文按 stride 2 从一个执行片段存多个重叠 transition：$(x_0,a_{0:C})$、$(x_2,a_{2:C+2})$、……，每秒约产生 25 个 RL 样本。Off-policy critic 允许这些相关、重叠 chunk 被重放。

### 3.7 完整执行策略

螺丝和扎带先只训练 critical phase，再扩到 full task：基础 VLA 完成抓取/搬运等前段，到关键阶段切换 RLT。这个手工 phase switch 显著减少探索空间，也是样本效率的重要来源，不应把全部提升只归功于 RL token。

---

## 4. 数据、在线收集与实验设置

### 4.1 四个真实任务

1. **Screw installation**：电动螺丝刀驱动 M3 螺丝，要求亚毫米刀头—螺丝头对齐，且 10 cm 杠杆放大末端姿态误差。
2. **Zip tie fastening**：双臂操纵可变形扎带，把尾部穿入狭窄锁孔，主要依赖腕部相机。
3. **Ethernet insertion**：对准凹陷端口并果断插入，小角度误差会卡住外壳。
4. **Charger insertion**：在插脚和插座并非总可见时完成厘米级对齐和接触插入。

完整任务持续 **30–120 秒**，50 Hz 下约 **1500–6000 控制步**；critical phase 为 5–20 秒、250–1000 步。Controlled critical-phase 每种方法/任务评测 **50 episodes**；full-task 从 home position 开始，前段状态分布由基础策略自然产生，难度更高。

### 4.2 数据量与训练预算

每任务收集 **1–10 小时 teleoperation demonstrations**，用于任务 VLA 微调和 RL token 训练；在线 RL 运行 **400–1000 episodes**。扣除 reset 和系统开销，每个实验实际机器人数据约 **15 分钟–5 小时**。螺丝与扎带完整训练约 5 小时数据。

奖励由人类给 episode 二元成功标签；指标为 success rate 与每 10 分钟成功次数（throughput）。后者能区分“都成功但一个反复试探、另一个一次插入”的策略质量。

### 4.3 Baseline

- **Base π0.6**：无在线 RL。
- **HIL-SERL**：ResNet 表征 + 小 actor-critic + intervention，单步动作。
- **PLD**：单步 residual policy，叠加到冻结 VLA 动作。
- **DSRL**：在 flow VLA 的 latent noise 空间做 RL，强约束于 VLA 可生成 mode。
- **DAgger**：用同一批 intervention 数据继续微调 VLA。

为公平起见，各 RL 方法使用相同数据量。附录说明 HIL-SERL 用 20 条示范初始化；PLD 先用 50 条 base-policy rollout 做 Cal-QL critic 预训练。HIL-SERL 在本设置中受 50 Hz 和无手工动作 bounding box 影响，未能学会。

---

## 5. 实验结果

### 5.1 对基础 VLA 的提升

RLT 在所有 critical phase 上提高速度和/或成功率。对基础策略已经可靠的 Ethernet、Charger，成功率接近饱和，主要收益是**约 3× 提速**；对困难的 Screw、Zip Tie，成功率增幅更大。论文摘要给出 Screw 的精确例子：**20% → 65%**。

Full-task 中，前段错误会累积，绝对成功率低于 isolated phase；RLT 仍使 Screw 提高 **40 个百分点量级**、Zip Tie 提高 **60 个百分点量级**。论文正文写作“improves ... by 40% / 60%”，对应 Figure 5 的柱状差约为 5%→45% 和 25%→85%，应理解为图示近似，而非未经表格确认的小数。

![](assets/RL%20Token%20-%20基于VLA表征的高效在线强化学习/figures/03-results.png)

> **图 3（原论文 Figure 5 的成功率部分，第 8 页）：Full-task 与 critical-phase 成功率。** 左侧虚线框是从 home state 开始的完整螺丝/扎带任务；右侧依次为四项 critical phase。RLT 在困难任务增益最大，在 Ethernet/Charger 已较强时保持或抬高成功率。来源：<https://arxiv.org/abs/2604.23073>

### 5.2 与替代 RL/IL 方法比较

Ethernet 上，HIL-SERL 与 PLD 这类单步方法难以从跨数百步的稀疏奖励传播价值。DAgger 与 DSRL 能达到接近 RLT 的成功率，但速度提升小：DAgger受限于人类示范速度；DSRL 在 latent mode 内探索很稳，却难以大幅偏离 VLA。RLT 在保持高成功率的同时，相比 base policy 把平均完成步数约减半，吞吐显著最高。

### 5.3 消融

- **w/o RL token**：用 ImageNet ResNet-10 特征替代，吞吐下降 **50%**，表明 VLA 压缩表征包含标准视觉 encoder 缺失的操控结构。
- **w/o Chunk（$C=1$）**：credit-assignment horizon 大幅拉长，且无法以 50 Hz 查询 VLA/RL token，不能可靠匹配 base policy。
- **w/o BC Regularizer（$\beta=0$）**：单项降幅最大，说明高维真实探索必须由 VLA 锚定。
- **w/o Pass-Through**：不把 $\widetilde a$ 输入 actor，最终在简单 Ethernet 上可追上，但学习更慢、早期失败更多并出现退化行为。

完整 RLT 只消耗 **5 分钟 critical-phase 数据**（计入 reset 等总实验约 40 分钟）便超过替代策略；消融学习曲线同时显示 pass-through 和 RL token 主要提升早期样本效率，而 BC 正则决定稳定上限。

### 5.4 超越遥操作的涌现策略

Ethernet critical phase 的 episode 长度中位数：**teleop 146 步、base VLA 228 步、RLT 66 步**。一半 RLT episode 比全部专家示范都快。基础 VLA 常靠近端口后反复进退试探；RLT 学会流畅插入，第一次失败时施压并轻微 wiggle 利用结构顺应性。这种策略未显式出现在示范中，说明 RL 不只是复制更快的人类动作。

### 5.5 结果是否支撑主张

论文对“少量在线数据 + 精密阶段提速”支持较强：四种不同接触任务、多个 baseline、学习曲线和逐组件消融形成闭环。仍需注意三点：成功标签和 intervention 由人提供；critical phase 起点由人工重置、phase switch 也手工指定；数值多在图中而非表中。RLT 证明的是“强 VLA 周围的局部在线精调”，尚不是端到端自主长程 RL。

---

## 6. 与相关工作及系列关系

### 6.1 与 π星0.6/RECAP 的关系

RECAP 更新整套 $\pi^*_{0.6}$，用分布式 value + advantage indicator 做 iterated offline RL，适合叠衣、咖啡、纸盒等长程技能，数据与训练规模较大。RLT 冻结 π0.6，压缩表征后在线训练小 actor-critic，适合现场关键阶段的分钟/小时级调优。前者更新“基础策略分布”，后者编辑“当前 VLA 建议”。

可组合路径是：先用 RECAP 得到任务级强策略，再用 RLT 在具体工位、具体插口公差和硬件磨损条件下快速适配。

### 6.2 与 residual/latent RL 的差异

Policy Decorator/PLD 输出单步 residual，需要手调缩放且 horizon 长；DSRL 改 flow 噪声，探索天然局限在 VLA 可生成 mode。RLT 直接生成短 action chunk，以显式 L2 正则而非固定 residual scale 控制偏离幅度；参考动作作为输入保留 mode，同时允许 actor 学到 VLA 分布之外的局部接触策略。

### 6.3 与 RTC 的关系

RLT 的 actor 使用 $C=10$ 短 chunk 并需频繁更新，基础 VLA 参考生成仍可能有延迟。RTC/Training-Time RTC 可为参考 VLA 提供异步连续执行；但 RLT 本身 actor 很小且高频，系统整合时需要明确：VLA 参考多久更新、actor 在两个参考 chunk 之间是否保持连续，以及 replay 中记录的是哪种执行协议。

---

## 7. 局限与批判性评价

1. **仍需人工 supervision**：人提供成功/失败、intervention，并手工切换 critical phase 与 base policy；作者明确将自动 reward/progress model 列为未来方向。
2. **任务语言固定**：RL token 训练阶段实验中去掉语言 embedding，尚未验证开放词汇、多任务共享 token。
3. **局部阶段假设很强**：方法把最难阶段隔离，先在随机化较小的状态训练；对无法清晰切段的长程任务不一定适用。
4. **冻结 VLA 限制表征可塑性**：若 VLA 根本没看见关键微小几何或触觉信息，2048 维压缩不能创造缺失特征。
5. **奖励极稀疏且人为**：二元标签简单但昂贵；错误标签、介入策略和操作者差异未系统分析。
6. **探索仍有硬件风险**：BC 正则降低风险但无形式安全保证；接触任务中的施压/wiggle 可能在不同公差或材料上损坏硬件。
7. **部署栈依赖大 VLA**：虽然 RL 网络小，推理仍需冻结 π0.6 提供 token 和参考动作，不是纯边缘轻量策略。
8. **复现开放性有限**：论文未披露官方代码、权重、全部超参数和机器人平台细节。

---

## 8. 复现与实践建议

### 8.1 最小复现顺序

1. 对一个已有 chunked policy 采任务示范，提取最后层 token。
2. 训练 encoder-decoder bottleneck，验证 token 对原 embeddings 的重构和下游成功预测。
3. 冻结 backbone，先在离线 buffer 上训练 chunk critic，再做低风险 online rollout。
4. Actor 同时输入 token、本体状态和 reference chunk，启用 BC 正则与 50% reference dropout。
5. 从 critical phase 开始，成功后再扩展 full task；始终保留 intervention 和硬件安全边界。

### 8.2 关键对照

至少比较：VLA feature pooling、ImageNet encoder、$C=1/5/10/20$、无参考输入、无 BC、不同 $\beta$、不同 dropout 比例、无 overlapping replay subsampling。报告的不只最终成功率，还应包括达到 base-policy 成功率所需分钟数、训练期间累计失败、介入次数与硬件保护触发。

### 8.3 成本评估

相对 full-model VLA RL，RLT 的学习计算成本低，但系统成本仍高：每任务 1–10 小时示范、400–1000 episodes、15 分钟–5 小时有效真机数据，多相机双臂平台和人工监督。外部团队若已有 π0.6 类 VLA 与真机 RL 栈，复现难度中高；从零复刻论文四任务则属于高成本机器人系统工程。

---

## 9. 个人启发与后续问题

RLT 提供了一个很通用的 foundation-policy 适配接口：**不要让 RL 接管大模型，而是让大模型暴露一个可学习的状态 token 和一个可编辑的动作 proposal**。下一步值得研究：多任务共享 RL token；加入触觉/力觉避免视觉瓶颈；用自动 reward model 替代人工标签；让 phase selector 由 value/progress model自动决定；以及把 actor 对 reference 的偏离约束从固定 $\beta$ 改为状态不确定性和安全风险自适应。

---

## 10. 材料来源

本笔记基于官方 PDF 全文、项目页及论文 Figure 1/2/5 撰写；本地材料逐项溯源见 [sources.md](assets/RL%20Token%20-%20基于VLA表征的高效在线强化学习/sources.md)。论文未披露独立官方代码仓库，故 `code: null`。
