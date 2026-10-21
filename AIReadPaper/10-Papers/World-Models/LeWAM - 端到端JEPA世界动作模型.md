---
type: paper-note
status: complete
domain: World-Models
paper: "LeWAM: End-to-End World and Action Modeling with JEPAs"
year: 2026
arxiv: "2610.12818v1"
doi: "10.48550/arXiv.2610.12818"
source: "https://arxiv.org/abs/2610.12818v1"
project: "https://le-wam.github.io/"
code: "https://github.com/MinghaoFu/lewam"
tags:
  - World-Models
  - JEPA
  - Robot-Learning
  - Flow-Matching
  - Visual-Planning
  - Behavior-Cloning
created: 2026-10-16
updated: 2026-10-16
---

# LeWAM：端到端 JEPA 世界—动作模型

## 基本信息

- **论文**：LeWAM: End-to-End World and Action Modeling with JEPAs
- **作者**：Minghao Fu、Tavis Siebert、Eryk Halicki、Randall Balestriero
- **版本**：arXiv:2610.12818v1，2026-10-08 提交，33 页（正文与附录合并）
- **机构**：UC San Diego、ETH Zurich、University of British Columbia、Brown University
- **任务**：从无奖励的离线视觉—动作轨迹中，同时学习视觉表征、动作条件潜在动力学与随机策略，并以同一模型支持直接策略执行和视觉目标规划
- **模型规模**：仿真视觉规划配置约 16.9M 可训练参数
- **材料**：[[assets/LeWAM - 端到端JEPA世界动作模型/sources|来源清单]]；[官方 PDF](assets/LeWAM%20-%20端到端JEPA世界动作模型/sources/paper.pdf)

## 一句话结论

LeWAM 的关键不是再给 JEPA 外接一个控制器，而是让 **next-latent prediction 与 action flow matching 在 shared encoder 和同一个 Mixture-of-Transformers 中联合训练**：前者学习“动作会把世界带到哪里”，后者学习“在当前状态与目标下哪些动作像数据中的可行动作”。论文实测显示，这个组合在接触丰富规划和长时域规划上显著超过以 CEM 驱动的 LeWorldModel，并把四任务平均规划吞吐从 1.62 提升到 52.50 plans/s；但真实机器人平均成功率并未超过 Diffusion Policy，因此更准确的结论是“统一世界模型与策略且保持竞争力”，而不是“全面替代专用策略”。

---

## 1. 研究背景与问题定义

### 1.1 JEPA 世界模型缺的不是预测器，而是与动作选择闭环的训练信号

Joint-Embedding Predictive Architecture（JEPA）不重建未来像素，而是在表征空间预测未来状态。给定视觉观测 $o_t$，编码器生成潜变量 $z_t$；动作条件动力学再根据当前潜变量与动作预测未来潜变量。与像素生成相比，这条路线希望把容量留给可预测、与任务相关的变化，而不是纹理、光照等像素细节。

DINO-WM、V-JEPA 2-AC 与 LeWorldModel（LeWM）已经证明潜在动力学可以用于视觉规划，但它们通常把“学世界”和“决定怎么行动”拆成阶段：先得到冻结或单独训练的表征与动力学，再在测试时用 CEM、MPPI、iCEM 等求解器搜索动作，或事后训练一个 inverse dynamics/policy head。这样做有两个问题。

第一，**表征学习未必保留控制所需的信息**。JEPA 只要求未来潜变量可预测，可能偏爱时间上稳定但与动作无关的视觉因素；如果动作监督不回传到 encoder，编码器没有充分理由区分“外观相似、但应采取不同动作”的状态。

第二，**从宽高斯分布盲搜动作在接触事件附近尤其困难**。平滑动力学里，相近动作往往产生相近结果，有限样本搜索还能逐步逼近；抓取、悬挂、推挤、双臂交接等接触任务存在模式切换，相近动作可能落入“接触成功”和“完全错过”两个分支。若候选集没有采到有效接触，目标潜变量距离几乎不给求解器有用排序信号。

> **作者解释**：LeWAM 的 action model 不只是部署时的 policy，也是一个数据驱动的 proposal distribution；它把搜索起点从宽高斯先验移到训练数据支持的动作流形附近，再让世界模型做后续评价和修正。

> **个人判断**：这篇论文最重要的概念变化，是把“world model + downstream controller”改写成“共享表征的 consequence model + behavior model”。它仍然不是完整的通用 agent，因为没有奖励、价值函数、语言高层规划或跨任务统一模型；但它确实把低层世界预测和动作生成放进了一个可互相塑形的训练闭环。

### 1.2 任务输入、输出与两种评测协议

训练数据是无奖励离线轨迹 $\bigl(o_{1:T},a_{1:T}\bigr)$。在 goal-conditioned visual planning 中，模型还接收目标图像的潜变量 $z_g$ 与归一化剩余时域；在 goal-agnostic task completion 中不提供目标图像，模型必须从观测历史隐式判断任务进度。

论文严格区分两种部署协议：

1. **Visual planning**：给当前随机起点和目标图像，在固定动作预算内到达目标；模型可在线搜索。
2. **Task completion / behavior cloning**：不给目标图像，策略直接执行完整任务，不进行测试时规划。

这个区分很重要。视觉规划的成功不能直接等价为无提示任务完成能力，而 behavior cloning 的结果也不能证明模型拥有显式目标推理能力。

### 1.3 核心贡献

1. 提出约 16.9M 参数的端到端 JEPA，在一次训练中联合学习视觉 encoder、action-conditioned latent dynamics 与 stochastic action policy。
2. 用 Mixture-of-Transformers（MoT）把 clean actions 驱动的未来潜变量预测和 noisy actions 驱动的 flow matching 放进同一 predictor，同时以掩码阻断不应出现的信息泄漏。
3. 提出 policy-initialized gradient planning：先从学习到的动作分布提出多条计划，再通过冻结的潜在动力学对动作做梯度细化和 best-of-N 选择。
4. 在十个异质仿真环境、长时域视觉目标、仿真行为克隆和三项真实机器人任务上评估，并用线性探针、rollout、动作多模态实验和组件消融解释联合训练的作用。

---

## 2. 端到端 JEPA 的总体结构

### 2.1 shared encoder：同一个视觉坐标系服务于历史、目标与未来

![](assets/LeWAM%20-%20端到端JEPA世界动作模型/figures/01-training-overview.png)

> **图 1（论文 Figure 1 上半部，PDF p.1）**：历史图像、未来目标图像与 goal image 都由 shared encoder 映射到潜空间；MoT 同时接收状态 token、clean action 与 noisy action。蓝色分支做未来潜变量 MSE，橙色分支做 action velocity MSE；SIGReg 同时约束编码后的潜变量分布。来源：官方 arXiv TeX 矢量图。

仿真模型的 encoder 是从零训练的 ResNet-18-style backbone，使用 16 组 GroupNorm。输入由 $224\times224$ 随机裁到 $202\times202$；一个 $1\times1$ 卷积产生 32 张空间图，spatial softmax 将其压成 32 对归一化坐标，再由两层 MLP 投影到 192 维潜变量。历史观测、未来监督帧和目标图像使用完全相同的 encoder，因此 dynamics loss、action loss 与 SIGReg 都会共同塑造这个坐标系。

定义视觉潜变量为：

$$z_t=\operatorname{Enc}(o_t)$$

模型保留两个历史潜变量。goal-reaching 使用历史间隔 $\Delta=K$，goal-free task completion 使用 $\Delta=1$。动力学 stride 固定为 $K=5$ 个原始环境步，而策略预测连续的 $C=2K=10$ 步 action chunk。也就是说，状态分支每次向前想象 5 步，动作分支一次生成 10 步，部署时通常执行前 5 步再重采样。

### 2.2 LeWAM 在相关路线中的定位

![](assets/LeWAM%20-%20端到端JEPA世界动作模型/figures/02-positioning.png)

> **图 2（论文 Figure 2，PDF p.2）**：LeWAM 相对 LeWM/V-JEPA 2、SIGReg/DINOv2、生成式 video WAM 与 CEM/MPPI/iCEM 的定位。它强调统一 world model 与 control、从零端到端训练、不做像素重建，并用 policy-guided proposals 加 gradient refinement。来源：官方 arXiv TeX 矢量图。

图 2 的定位可以压缩成一句话：**LeWAM 继承 LeWorldModel 的 reconstruction-free、from-scratch、SIGReg latent world model，但把 action prediction 从后处理模块提升为与 dynamics 同步训练的主目标。**

这不意味着 state prediction 与 action prediction 完全共享参数。共享的是 encoder、训练样本与 MoT 的 masked attention；状态和动作两种 modality 在每层拥有各自的 Q/K/V、输出投影、归一化和 FFN。作者用这种“共享信息流、分离模态参数”的结构降低两个损失彼此拉扯的风险。

---

## 3. Mixture-of-Transformers 如何联合预测世界与动作

### 3.1 两条预测接口

MoT predictor 含 state branch $f_z$ 与 action branch $f_a$。状态分支从历史潜变量与前 $K$ 个 clean actions 预测 $z_{t+K}$；动作分支从 flow time $s$ 上的 noisy action chunk、历史潜变量与可选 goal context 预测 action velocity：

$$\hat z_{t+K}=f_z\!\left(q^z_{t+K};z_{\le t},A_{t,K}\right),\qquad \hat v_{t,C}=f_a\!\left(A_{t,C}^{(s)};s,z_{\le t},c_t\right)$$

仿真配置的 MoT 宽度为 192，4 个 block、4 个 attention head、dropout 0.1。无 goal context 时，token 序列由两个历史 latent、一个 learned future-state query、5 个 clean-action tokens 与 10 个 noisy-action tokens 组成。

### 3.2 掩码不是实现细节，而是联合目标成立的前提

state query 可以读取历史 latent 和 clean actions，因此它能学习 forward dynamics。noisy action tokens 可以读取历史 latent、goal token（若有）和自身 causal prefix，因此它能学习条件动作分布。历史 latent 只能互相注意，不能通过多层 attention 间接读取 clean action；动作 flow tokens 也不能偷看用于 dynamics supervision 的 clean actions。

> **作者设计意图**：既允许 state/action 两个分支交换共同的视觉状态信息，又防止 action branch 在训练时直接复制真值 action，或通过 history token 的中转形成泄漏。

> **个人判断**：MoT 的贡献不应被简化为“用了 Transformer”。真正关键的是信息因果结构：clean actions 只为 consequence prediction 提供条件，noisy actions 只沿 flow path 学习去噪速度，goal 只进入 action branch。这个约束让两个任务共享 encoder，却没有破坏部署时可获得信息的边界。

### 3.3 next latent prediction：动作会把世界带到哪里

状态分支以 MSE 对齐真实未来帧的 encoder 表征：

$$\mathcal L_{\mathrm{dyn}}=\mathbb E_{\mathcal D}\!\left[\operatorname{MSE}\!\left(\hat z_{t+K},z_{t+K}\right)\right]$$

这里的 target encoder 不是 EMA teacher，也没有 state noise 或 latent flow integration；论文报告的配置使用 online encoder target。由于 encoder 同时被 SIGReg 与 action supervision 约束，单纯把所有图像映射到常量不能成为联合目标的好解。

这仍然是确定性的 one-step latent regression；长时域 rollout 通过每 5 步递归调用预测器完成。因此 LeWAM 并没有解决所有 autoregressive error accumulation，只是让潜空间更控制相关，并让 action proposals 更可能处于可预测、可执行区域。

### 3.4 action flow matching：学习条件动作分布，而不是条件均值

对数据中的 action chunk $A_{t,C}$，采样 $s\sim\mathcal U[0,1]$ 与高斯噪声 $\epsilon\sim\mathcal N(0,I)$，在噪声和真实动作之间构造直线路径：

$$A_{t,C}^{(s)}=(1-s)\epsilon+sA_{t,C},\qquad v_{t,C}=\frac{\mathrm dA_{t,C}^{(s)}}{\mathrm ds}=A_{t,C}-\epsilon$$

动作分支回归这个 velocity field：

$$\mathcal L_{\mathrm{act}}=\mathbb E_{\mathcal D,s,\epsilon}\!\left[\operatorname{MSE}\!\left(\hat v_{t,C},A_{t,C}-\epsilon\right)\right]$$

推理时从高斯噪声出发，对学习到的速度场做 ODE 积分，即可采样一条 action chunk。仿真采用 8 个 Euler steps，真实机器人配置采用 10 个 Euler steps。

与 deterministic MSE action head 相比，flow matching 的目的不是预测唯一的平均动作。若同一起点到目标存在上下两条路线，MSE 容易输出两种模式之间的折中；flow model 则可以从不同噪声样本生成不同合法模式。附录的 two-door navigation 实验中，flow head 覆盖两条门路径，而 MSE head 主要集中到下方路径并出现撞墙。

### 3.5 SIGReg 与 isotropic Gaussian constraint：防坍塌，也让潜距离可用于规划

SIGReg 收集 minibatch 中历史和未来图像的编码结果 $Z$，沿 $M$ 个随机单位方向 $u^{(m)}$ 投影，并用一维 Epps–Pulley statistic $T$ 检验每个投影是否匹配标准高斯：

$$\mathcal L_{\mathrm{sig}}=\frac{1}{M}\sum_{m=1}^{M}T\!\left(Zu^{(m)}\right)$$

完整目标为：

$$\mathcal L_{\mathrm{LeWAM}}=\mathcal L_{\mathrm{dyn}}+\mathcal L_{\mathrm{act}}+\lambda_{\mathrm{sig}}\mathcal L_{\mathrm{sig}},\qquad \lambda_{\mathrm{sig}}=0.04$$

仿真使用 1,024 个随机投影和 17 个 knots。这个 isotropic Gaussian constraint 不要求每个样本都服从独立高斯噪声，而是约束 encoder 产生的总体潜变量边缘分布：各方向有足够方差、尺度较均衡，避免所有样本塌缩到常量或低维子空间。

论文还给出 action supervision 的 conditional-risk 解释：若视觉历史确实会改变条件 action velocity，那么丢弃视觉历史会增加最优 flow-regression risk。因此 action loss 本身也构成 anti-collapse signal；但实验显示它不能替代 SIGReg，因为用于目标比较和梯度规划的潜空间仍需要更均匀、信息更丰富的几何结构。

> **论文实测**：在 Push-T 上，去掉 SIGReg 的 LeWM 所有已报告 probe 都是 $R^2=0$；带 action head 的 LeWAM 即使无 SIGReg，仍保留 object $R^2=0.88$、proprioception $R^2=0.85$、action $R^2=0.38$。不过完整 LeWAM 的 planning success 仍比 no-SIGReg 高 25.3 个百分点。

> **作者解释**：action supervision 锚定了与动作相关的区分，SIGReg 则保留更全面的状态信息，使 latent distance-to-goal 对物体状态变化更敏感。

> **个人判断**：证据支持“二者互补”，但还不足以证明 isotropic Gaussian 是唯一或最优正则。论文没有与 VICReg、EMA target、whitening 或其他现代 anti-collapse 机制做同规模对照。

---

## 4. 同一模型的两种控制模式

### 4.1 Direct policy execution：只采样，不搜索

固定历史和可选 goal context，从标准高斯 action chunk 出发，沿 action branch 的速度场积分：

$$\frac{\mathrm dA_{t,C}^{(s)}}{\mathrm ds}=f_a\!\left(A_{t,C}^{(s)};s,z_{\le t},c_t\right),\qquad A_{t,C}^{(0)}\sim\mathcal N(0,I),\qquad \hat A_{t,C}=A_{t,C}^{(1)}$$

模型执行前 $K=5$ 个动作，再编码新观测并重采样。这就是 receding-horizon direct policy，部署路径不调用 dynamics rollout，也不做 test-time optimization。无 goal 的 behavior cloning 和真实机器人实验主要使用此模式。

Direct execution 的优势是低延迟、流程简单；缺点是完全依赖训练动作分布覆盖，一旦遇到需要偏离示范的状态，没有显式目标代价帮助纠错。

### 4.2 Planning：action proposal、latent rollout、gradient refinement、best-of-N

![](assets/LeWAM%20-%20端到端JEPA世界动作模型/figures/03-planning-mechanism.png)

> **图 3（论文 Figure 4，PDF p.5）**：每个 MoT stage 从 learned flow policy 提出 action chunks，并用其前 $K$ 步把潜状态向前推进；形成 $N$ 条长度 $L$ 的候选后，通过冻结 dynamics 对整条动作序列做梯度下降，最后选择目标 latent MSE 最低的历史 iterate。来源：官方 arXiv TeX 矢量图。

规划不是用 action head 一次性给答案，而是以下四步：

1. **Proposal**：在当前 history、goal latent 和剩余 horizon 下，从 Gaussian noise 经 action flow 采样 $N$ 个 action chunks。
2. **Autoregressive imagination**：每条候选只取 chunk 的前 $K$ 步送入 state branch，预测对应的 $z_{t+K}$；把预测状态追加进固定长度 history，再继续采样下一 chunk，直到计划长度达到 $L$。
3. **Refinement**：把整条动作序列当作可优化变量，冻结 LeWAM 参数，通过潜在动力学反向传播，最小化最终预测状态和目标状态的距离。
4. **Selection and MPC execution**：每条候选保留搜索过程中目标代价最低的 iterate，跨候选做 best-of-N；执行最终计划的前 $K$ 步，看到新观测后重新规划。

目标代价是最终潜变量 MSE：

$$\mathcal C(A)=\operatorname{MSE}\!\left(\hat z_{t+L}(A),z_g\right)$$

动作更新为：

$$a_{t+k}\leftarrow a_{t+k}-\eta\frac{\partial\mathcal C}{\partial a_{t+k}},\qquad k=0,\ldots,L-1$$

实际优化还加入 proposal proximity penalty：

$$\mathcal J_{\mathrm{prox}}\!\left(A;A^{\mathrm{sample}}\right)=\mathcal C(A)+\lambda_{\mathrm{prox}}\left\|A-A^{\mathrm{sample}}\right\|_F^2$$

论文设置 Adam learning rate 为 0.05、gradient norm clip 为 10、normalized action clip 为 $[-3,3]$、$\lambda_{\mathrm{prox}}=0.01$。penalty 只影响更新；候选保留与最终选择仍使用不含 penalty 的 $\mathcal C$。速度对比使用 5 次 gradient steps；Push-T sensitivity 实验显示超过 5 次后成功率基本饱和而延迟继续上升，候选数超过 8 也没有一致收益。

> **作者解释**：proposal distribution 负责先找到“像样的行为分支”，dynamics gradient 负责在该分支附近针对当前目标微调；proximity penalty 防止优化把动作推到模型没见过、动力学不可靠的区域。

> **个人判断**：这是 amortized planning 与 online optimization 的折中。它比纯 policy 多一次 model-based correction，比 CEM 少大量盲 rollout；但目标仍只是 terminal latent MSE，没有显式路径约束、碰撞代价或不确定性惩罚，因此“模型误差可被优化器利用”的经典 model exploitation 风险仍存在。

---

## 5. 实验环境、数据与基线

### 5.1 十个仿真环境与三项真实机器人任务

十个仿真环境覆盖：Two-Room、Reacher、Push-T、Cube、PointMaze-Large、Scene、Puzzle、ToolHang、Transport、DrawerCleanup。前四项沿用 LeWM benchmark；后六项增加大型迷宫、对象交互、按钮逻辑、精细工具悬挂、双臂运输与抽屉操作。数据混合 noisy heuristic、uniform random、non-expert interactions、scripted expert、human demonstrations 与 generated demonstrations，共 71,988 条轨迹、13,101,054 帧、约 235.51 小时。

多数短时域视觉规划使用从同轨迹晚 25 步的图像作目标，动作预算为 50；Two-Room 按 LeWM 协议使用 100 步 goal offset 和 150 动作预算。每项通常报告 3 个 seeds、每 seed 50 episodes。Cube 的成功条件同时检查 cube 与 end-effector 是否在目标位置 $0.04$m 内，避免“物体恰好没动但已在阈值内”的伪成功。

真实机器人平台使用 reBot Arm 102 leader 与 reBot B601-DM follower，通过 LeRobot 做 joint-space teleoperation。数据包括 Pick and Place T（200 episodes，29 分钟）、Push-T（100 episodes，34 分钟）与 Tape Stack（100 episodes，13 分钟）。场景相机和腕部相机以 30Hz 采集，动作是 7 维 joint targets；每种方法、每个任务评估 30 次闭环试验。

### 5.2 视觉规划基线

| 方法 | 表征与动力学 | 动作产生/规划 | 与 LeWAM 的关键差别 |
|---|---|---|---|
| **LeWM (CEM)** | 从零训练 encoder；SIGReg；action-conditioned latent dynamics | CEM | 没有 learned policy，搜索从宽先验开始 |
| **LeWM (IDM) / GC-IDM** | 先训练并冻结 LeWM | 事后训练 goal-conditioned inverse dynamics | 两阶段，action supervision 不塑造 encoder |
| **DINO-WM** | 冻结 DINOv2 表征，再学 dynamics | CEM | 依赖预训练视觉特征，不端到端 |
| **GCBC** | 冻结 DINOv2 | 直接 goal-conditioned behavior cloning | 无 forward dynamics，不做 model-based refinement |
| **LeWAM** | shared encoder + SIGReg + joint latent/action predictor | direct flow policy 或 proposal + gradient planning | 同一训练 pass 同时优化 dynamics 与 policy |

行为克隆仿真基线是 convolutional Diffusion Policy；真实机器人基线是 Diffusion Policy 与视觉—语言—动作模型 $\pi_{0.5}$。这些比较回答的是“统一 JEPA 是否至少能达到专用策略的量级”，并非所有模型在预训练规模、参数量和计算量上完全匹配。

### 5.3 训练与实现细节

Goal-conditioned visual planning 训练 50 epochs、batch size 128、base learning rate $1.5\times10^{-4}$、5 个 warmup epochs。Goal-free task completion 训练 120 epochs、batch size 64、learning rate $10^{-4}$、10 个 warmup epochs。二者使用 AdamW、weight decay $10^{-4}$、cosine schedule、full precision、gradient clip 1、seed 42。数据按有效决策点做 90/10 train/validation split，checkpoint 按最低 validation action flow-matching loss 选择，不用任务成功率挑模型。

> **个人判断**：按 validation action loss 选 checkpoint 避免了用测试成功率调参，是合理的；但单一训练 seed 42 与大量 author-supplied ablation/timing records 限制了对训练稳定性的判断。评测有多 seed，不等于训练随机性的完整覆盖。

---

## 6. 主要结果：哪里明显有效，哪里只是“竞争力相当”

### 6.1 十环境视觉规划与 contact-rich 结果

![](assets/LeWAM%20-%20端到端JEPA世界动作模型/figures/04-main-results.png)

> **图 4（论文 Figure 6，PDF p.7）**：十个环境的短时域视觉目标成功率。LeWAM 在每个任务上均超过 LeWM (CEM)；接触丰富的 ToolHang、Transport、DrawerCleanup 差距尤其大。星形标注使用 CEM 的方法，三角形标注直接策略。来源：官方 arXiv TeX 矢量图。

**论文实测**：十任务平均成功率为 93.5%，LeWM (CEM) 为 50.9%，差 42.7 个百分点。论文首页对“contact-rich manipulation”子集给出的聚合数字是 89.7% 对 28.6%。具体例子包括：

- ToolHang：LeWAM 94.7%，LeWM (CEM) 35.4%；
- Transport：LeWAM 83.3%，LeWM (CEM) 2.8%；
- DrawerCleanup：LeWAM 80.7%，LeWM (CEM) 12.4%；
- Puzzle：LeWAM 80.7%，GCBC 44.0%。

这些结果与论文的机制主张一致：高维、接触丰富任务最不适合从宽动作先验做大规模盲搜，而 learned flow proposals 恰好在这些任务上价值最大。

不过 Figure 6 不能把改进全部归因于某一个组件。LeWAM 同时改变了 encoder 训练、action head、planner 和搜索预算。组件归因必须看后面的 matched ablation。

### 6.2 长时域：平均提升显著，但 Puzzle 仍明显失败

同一 checkpoint 在 Push-T、Cube、Scene、Puzzle 上把目标沿同一 held-out trajectory 推到 25/50/75/100 步。100-step goal offset 的平均成功率为 46.2%，LeWM 为 10.9%。逐项 100-step 结果是：

| 任务 | LeWAM | LeWM (CEM) |
|---|---:|---:|
| Push-T | 43.3 ± 4.6 | 2.7 ± 1.2 |
| Cube | 82.0 ± 2.0 | 4.0 ± 3.5 |
| Scene | 56.0 ± 4.0 | 36.0 ± 6.0 |
| Puzzle | 3.3 ± 1.2 | 0.7 ± 1.2 |
| **四任务均值** | **46.2** | **10.9** |

**论文实测**明确支持“比 LeWM 更能扩展到长 horizon”，但不支持“长时域问题已解决”。Puzzle 从 25 步 74.0% 降到 100 步 3.3%，说明需要离散逻辑顺序和多阶段状态切换的任务仍会因 autoregressive dynamics error、proposal coverage 或 terminal-only objective 而崩溃。

### 6.3 32.3× 规划速度的证据链

论文的 32.3× 不是单个最好案例，也不是理论 FLOPs 比，而是同一 H100 上四任务 **per-task plans/s 的算术平均之比**。LeWM 使用 $300\times30$ CEM，LeWAM 使用 5 次 gradient refinement：

| 任务 | LeWM ms/plan | LeWM plans/s | LeWAM ms/plan | LeWAM plans/s | 加速 |
|---|---:|---:|---:|---:|---:|
| Scene | 705 | 1.42 | 18.8 | 53.19 | 37.5× |
| Puzzle | 731 | 1.37 | 16.3 | 61.35 | 44.8× |
| Cube | 771 | 1.30 | 21.1 | 47.39 | 36.5× |
| Push-T | 414 | 2.42 | 20.8 | 48.08 | 19.9× |
| **任务均值** | — | **1.62** | — | **52.50** | **32.3×** |

计算依据为：

$$\frac{52.50\ \mathrm{plans/s}}{1.62\ \mathrm{plans/s}}\approx32.4$$

表中报告 32.3×，差异来自未四舍五入的原始 per-task throughput。这个证据说明在作者选定的 matched hardware 和 planner budgets 下，batched gradient search 的吞吐远高于 $300\times30$ CEM。

> **个人判断**：应把结论限定为“相对论文采用的 LeWM-CEM 配置快 32.3×”。它不能外推为相对所有 CEM/iCEM/MPPI 实现都快 32.3×；若更换 population、iterations、编译策略或精度，倍率会变化。论文也未报告训练总成本、端到端控制回路的网络/相机延迟或能耗。

### 6.4 仿真 behavior cloning：平均略胜，任务间互有胜负

Goal-free task completion 中，LeWAM 不接收 goal context，也不调用 planner：

| 方法 | DrawerCleanup | Transport | ToolHang | 平均 |
|---|---:|---:|---:|---:|
| Diffusion Policy | **54.7 ± 1.2** | 45.3 ± 9.0 | **84.7 ± 3.1** | 61.6 |
| LeWAM | 52.7 ± 5.0 | **61.3 ± 9.9** | 80.0 ± 3.5 | **64.7** |

LeWAM 的平均优势主要来自 Transport 的 +16.0 个百分点；在 DrawerCleanup 和 ToolHang 上并没有胜过 Diffusion Policy。因此论文使用“competitive”而不是“state of the art”是准确的。

### 6.5 真实机器人：统一模型可用，但没有超过专用 policy baseline

真实机器人成功次数如下：

| 方法 | Pick and Place T | Push-T | Tape Stack | 平均成功率 | GFLOPs/完整 action chunk |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy | 19/30 | 16/30 | 0/30 | **38.9%** | 270.0 |
| $\pi_{0.5}$ | 14/30 | 3/30 | **9/30** | 28.9% | 3603.6 |
| LeWAM（Frozen V-JEPA 2.1 Encoder） | 15/30 | 12/30 | 5/30 | 35.6% | 270.2 |
| LeWAM（end-to-end） | 16/30 | 8/30 | 6/30 | 33.3% | 270.2 |

**论文实测**：end-to-end LeWAM 平均 33.3%，与 frozen V-JEPA 2.1 版本的 35.6% 接近；相同示范数据下，联合 latent/action prediction 确实能驱动真实机器人闭环控制。其一个 action chunk 为 270.2 GFLOPs，$\pi_{0.5}$ 为 3603.6 GFLOPs，对应约 13.3× operation-count efficiency。

**不能过度解读的地方**：Diffusion Policy 平均 38.9% 仍高于两种 LeWAM；Push-T 上 end-to-end LeWAM 8/30，只有 Diffusion Policy 的一半；每格仅 30 次试验且未报告置信区间。真实机器人证据证明“能工作且计算高效”，没有证明“世界模型目标提升了 imitation learning”。作者也明确把后一个问题留给更广泛评测。

---

## 7. 消融与解释性分析

### 7.1 总览

![](assets/LeWAM%20-%20端到端JEPA世界动作模型/figures/05-ablations.png)

> **图 5（论文 Figure 8，PDF p.8）**：从左到右分别检验 SIGReg、视觉 encoder、stochastic flow vs. MSE action head，以及与 Diffusion Policy 的 goal-free task completion 对比。来源：官方 arXiv TeX 矢量图。

### 7.2 SIGReg：action supervision 能防完全坍塌，但规划仍需要潜空间几何

移除 SIGReg 后，相对完整模型的 planning success 在 Push-T、Scene、Puzzle 分别下降 25.3、4.0、5.3 个百分点。Push-T 上完整 LeWAM direct/planning 为 82.7/97.3，no-SIGReg 为 69.3/72.0；规划相对 direct 的增益从 +14.7 缩到 +2.7。

这组结果要区分两个量：Figure 8(a) 的 25.3 pp 是“完整模型 planning 与 no-SIGReg planning 的差”，而表格中的 +14.7 pp 是“同一完整模型 planning 相对 direct 的增益”。二者回答不同问题，不能混用。

### 7.3 为什么 shared encoder 要端到端训练

四任务 equal-weight 平均：

- end-to-end encoder：direct 86.7%，planning 91.5%，gain +4.9 pp；
- frozen DINOv2 CLS：direct 79.0%，planning 82.7%，gain +3.6 pp；
- frozen DINOv2 patch mean：direct 75.8%，planning 77.7%，gain +1.9 pp。

**论文实测**支持“在这组数据与结构中，联合 action/dynamics supervision 学到的 encoder 更适合控制”。但它不是对 foundation encoder 的普遍否定：真实机器人上 frozen V-JEPA 2.1 版本还略高于 from-scratch LeWAM，说明预训练在数据较少、视觉变化更大的物理场景仍有价值。

### 7.4 stochastic flow action head 为什么比 MSE 更适合 proposal

在相同 latent-dynamics objective 下，Push-T 的 planning gain：flow +14.7 pp，MSE +5.3 pp；Scene：+5.3 对 +0.7；Puzzle：+4.7 对 -2.0。无 SIGReg 时，flow head 的 object probe $R^2=0.88$，MSE head 为 0.66，而 proprioception 接近（0.85 对 0.86）。

**作者解释**：MSE 学条件均值，可能抹掉多模态动作之间的区别；flow matching 学完整条件分布，因此 proposal 更丰富，也迫使 encoder 保留能够区分动作模式的视觉信息。

**个人判断**：two-door 实验清楚展示了多模态现象，但仍是构造性案例。要把“多模态是 contact-rich 提升的主因”变成强因果结论，还需要匹配采样数、模型容量和计算预算的 mixture density、diffusion action head 或离散 action model 对照。

### 7.5 dynamics 是否真的保留了世界预测能力

作者在评测时移除 LeWAM action head，只用其 action-conditioned latent dynamics 配合 CEM。Push-T 上 LeWAM (CEM) 84.0 ± 1.3，LeWM (CEM) 92.4 ± 1.2；Cube 上 LeWAM 86.0 ± 3.2，LeWM 63.8 ± 2.2。配合 post-hoc pixel decoder，LeWAM 在 Cube、Scene、Puzzle 的机器人手臂、夹爪和物体细节更清晰。

这说明联合训练没有让 dynamics 退化成“只服务 policy 的隐变量”，并可能改善 3D 接触相关特征；但 Push-T 上 dynamics-only 反而低于 LeWM，故不能概括为联合训练普遍提升 forward model。

### 7.6 线性探针与可解释性证据

在 Cube、Scene、Puzzle 等任务上，相比 LeWM，action modeling 使平均 proprioception、action、IDM $R^2$ 分别提高约 +0.21、+0.20、+0.19。Push-T 完整 LeWAM 的 object/proprio/action $R^2$ 为 0.90/0.86/0.40，使用 $(z_t,z_{t+K})$ 做 IDM 时 action $R^2$ 达 0.75。

这些 probe 表明潜变量线性可解码更多控制信息，但“可线性解码”不等于模型的规划决策确实因该变量而正确。更强的解释证据来自三个方向共同一致：probe、recorded-action latent rollout、以及 planner/flow-head matched ablation。

---

## 8. 与相关工作的关系

### 8.1 DINO-WM

DINO-WM 在 frozen DINOv2 features 上学习 action-conditioned world dynamics，再用 CEM 规划。它证明了预训练视觉特征可支持 zero-shot-style visual planning，但 action search 不参与 encoder 学习。LeWAM 的变化是从零训练 shared encoder，并让 dynamics loss、flow policy loss、SIGReg 同时回传。因此两者的核心差异不是“有没有 JEPA”，而是 **frozen representation + downstream search** 对 **end-to-end control-shaped representation + learned proposal**。

### 8.2 LeWorldModel（LeWM）

LeWM 是 LeWAM 最直接的母体和主要基线：二者都从像素端到端学习、都不重建像素、都用 SIGReg 维持潜空间、都预测 action-conditioned next latent。LeWAM 增加 action flow matching 与 MoT 联合结构，并把测试时 CEM 替换为 policy proposals + gradient refinement。可以把 LeWAM 看成从“只学 consequence”到“jointly learn behavior and consequence”的扩展。

### 8.3 Flow-JEPA

Flow-JEPA 用 conditional flow matching 生成 **未来 latent trajectory**，目标是替代 LeWM 的 deterministic pointwise/autoregressive dynamics，提高视觉扰动下的鲁棒性。LeWAM 则保留 MSE next-latent dynamics，把 flow matching 用于 **action distribution**。二者使用 flow 的对象完全不同：

- Flow-JEPA：$\text{noise}\rightarrow\text{future latent trajectory}$；
- LeWAM：$\text{noise}\rightarrow\text{action chunk}$。

> **来源边界**：LeWAM v1 正文没有引用 Flow-JEPA；这里基于 Flow-JEPA 官方 arXiv:2608.29029v3 做外部对照。

> **个人判断**：两条路线高度互补。一个自然延伸是用 trajectory-level latent flow 替换 LeWAM 的 autoregressive next-latent branch，同时保留 action flow branch；代价是双 flow 的训练、积分和联合注意力稳定性会更复杂。

### 8.4 V-JEPA 2-AC 与 V-JEPA 2.1 encoder variant

V-JEPA 2-AC 在冻结的预训练 V-JEPA 2 表征上学习 action-conditioned dynamics，支持真实机器人规划，但没有与 dynamics 联合训练的 policy。LeWAM 主模型从零训练并联合建模 actions。论文真实机器人实验里的 “LeWAM with frozen V-JEPA 2.1 encoder” 只是把 LeWAM 的视觉 backbone 换成 frozen V-JEPA 2.1 表征，action/dynamics joint predictor 仍属于 LeWAM；它不应与 V-JEPA 2-AC 本身混为一谈。

### 8.5 GC-IDM

GC-IDM 在冻结 JEPA/world-model embedding 上 post-train goal-conditioned inverse dynamics，能够直接从当前与目标表征预测动作。论文中的 LeWM (IDM) 就是该两阶段配置。LeWAM 与它的差别是 action supervision 从第一天就参与 encoder 学习，并以 flow 表示多模态 action distribution；同时 LeWAM 保留 dynamics，可在 direct policy 之外继续做 action refinement。

### 8.6 PRISM

PRISM 在冻结 world-model features 上训练 Gaussian prior head，以预测均值和方差引导 MPPI imagination sampling。它与 LeWAM 都承认“搜索应由 learned prior 引导”，但 PRISM 是 frozen model 上的后训练 Gaussian prior + MPPI，LeWAM 是 end-to-end flow policy + differentiable dynamics refinement。PRISM 更模块化，LeWAM 的表征—策略耦合更强；前者容易替换底层 world model，后者理论上能让 action supervision 直接改善表征。

---

## 9. 局限、证据边界与批判性评价

### 9.1 作者明确承认的局限

1. **数据覆盖有限**：contact-rich 数据中的 play/exploration 不足，困难接触转换覆盖不全。
2. **泛化尚未验证**：陌生物体、相机视角、接触状态与视觉变化仍需系统测试。
3. **单任务、单 embodiment**：每个任务和机器人分别训练或适配，尚未实现跨任务、跨机器人共享。
4. **缺少层级规划**：LeWAM 是低层连续控制器，长任务仍需要高层 subgoal decomposition。
5. **训练与测试时扩展性未充分探索**：当前视觉规划模型约 17M 参数；更大模型、更多数据、更多候选与 refinement steps 仍是未来工作。
6. **物理部署需独立安全机制**：论文没有把 learned world model 当作安全保证。

### 9.2 个人判断：还需要警惕的四个问题

**第一，terminal latent MSE 不是通用任务目标。** 当前 goal image 与数据轨迹提供了强监督，且成功谓词与视觉状态高度对应。对路径依赖、不可逆约束、遮挡或视觉等价但语义不同的状态，仅比较最终 latent 可能不足。

**第二，planner 可能利用模型误差。** 梯度直接穿过 learned dynamics；没有 ensemble uncertainty、out-of-distribution penalty 或 consistency check。proximity penalty 只是启发式限制，不能保证 refined actions 仍处于可靠区域。

**第三，真实机器人证据规模较小。** 三任务、每格 30 trials、单一机器人、单一熟练操作者、总数据 76 分钟左右，且未报告训练 wall-clock、端到端 latency 与置信区间。论文足以证明 feasibility，不足以证明 robust deployment。

**第四，若干结论的实验来源不完全对称。** 主文 TeX 注释显示部分 timing、probe、ablation 数值为 author-supplied records，并非所有不确定性都可用；训练 seed 也主要固定为 42。代码已公开有助于复核，但读者应把精确百分点视为当前 v1 的报告结果，而非已被广泛复现实验确认的常数。

### 9.3 适用边界

LeWAM 最适合：有离线视觉—动作轨迹、动作连续、目标可由图像表达、需要在 direct execution 与 online refinement 间切换的低层控制任务。它不直接解决：语言任务分解、reward-based exploration、跨 embodiment action tokenization、部分可观测长记忆、严格安全约束或开放世界大规模预训练。

---

## 10. 复现与实践建议

### 10.1 最小复现路径

1. 先复现 goal-conditioned Push-T：两帧 history、$K=5$、$C=10$、192 维 latent、4-block MoT。
2. 同时记录 dynamics MSE、flow MSE、SIGReg、latent effective rank 与 direct success，避免只看一个训练 loss。
3. 先运行 direct policy，再接入 dynamics rollout；最后加入 5-step Adam refinement 和 $\lambda_{\mathrm{prox}}=0.01$。
4. 必做三组 matched ablation：无 SIGReg、MSE action head、frozen encoder。
5. 分开测 direct、proposal-only best-of-N、proposal + refinement 与 dynamics-only CEM，才能判断收益来自 policy prior 还是 world-model gradient。

### 10.2 容易踩坑的实现点

- clean actions 不能通过 attention 间接泄漏给 flow branch；必须复现 token-level mask。
- goal token 只应对 noisy action tokens 可见，state dynamics 不应偷看目标后预测“想要的未来”。
- 训练时只 subsample frames，不 subsample action chunk；state stride 是 5，但 action 仍保持原始步率。
- planner 更新的是 normalized actions，模型参数冻结；每步更新后要 clip。
- best iterate 按不含 proximity penalty 的 goal cost 保存，否则选择准则会偏向“没改动作”。
- Cube 评测必须同时检查 object 与 end-effector，避免高达 42.7% 的 zero-action 伪成功地板。

### 10.3 资源与可复现性评估

论文给出模型结构、优化器、学习率、batch size、epoch、掩码、规划伪代码、数据统计和基线配置，且项目页与代码仓库已公开，方法复现信息总体充分。缺口主要是完整训练 wall-clock、训练硬件、所有 ablation 的训练随机种子、部分真实机器人延迟与逐 trial 记录。官方 PDF 已包含 supplementary appendix，没有独立 supplementary 文件。

---

## 11. 个人启发与后续问题

### 11.1 从“预测一切”转向“预测可行动的区别”

LeWAM 展示了一个有价值的原则：世界模型不必保留全部视觉细节，但必须保留会改变可行动作集合和动作后果的区别。SIGReg 提供全局分布几何，action flow 提供控制条件风险，next-latent prediction 提供后果一致性；三者分别回答“别塌缩”“该怎么动”“动了会怎样”。

### 11.2 proposal 与 model-based correction 可以成为通用接口

与其在“纯 policy”与“纯 planner”之间二选一，不如让 policy 产生低熵、数据内候选，让 world model 只负责局部修正。这个接口可迁移到 diffusion policy、VLA action expert、trajectory transformer，甚至离散技能规划：关键是让 proposal 有多样性，让 consequence model 有可微或可搜索的评价信号。

### 11.3 值得继续验证的问题

1. 把 Flow-JEPA 的 latent trajectory flow 与 LeWAM 的 action flow 合并，能否同时降低 dynamics error accumulation 和动作搜索难度？
2. 若使用 ensemble uncertainty 或 conservative latent cost，是否能减少 gradient exploitation？
3. 多任务联合训练时，SIGReg 的单一 isotropic Gaussian 是否会压平 task-specific geometry？
4. action flow 对 encoder 的塑形是否会牺牲与当前行为数据无关、但迁移时有用的状态信息？
5. 在真实机器人上，世界模型目标究竟改善了 policy，还是只增加了一个不伤害性能的辅助任务？需要 matched policy-only ablation 才能回答。

---

## 12. 关键术语表

- **JEPA**：在 embedding 空间预测目标表示，不要求重建像素。
- **SIGReg**：通过随机一维投影与分布检验，把 latent marginal 约束到近似 isotropic Gaussian 的防坍塌正则。
- **Mixture-of-Transformers（MoT）**：不同 modality 使用独立投影/FFN，但在受控掩码下共享 attention 信息流的结构。
- **Action flow matching**：学习从 Gaussian noise 到条件 action distribution 的连续 velocity field。
- **Action proposal**：从 learned action distribution 采样的候选动作序列，是在线规划的初始化。
- **Gradient refinement**：通过冻结的 latent dynamics 对 action sequence 求梯度，降低目标潜变量距离。
- **Direct execution**：只运行 action flow policy，不调用在线动作搜索。
- **Planning**：proposal、latent rollout、gradient refinement 与 best-of-N selection 的组合。

---

## 13. 图表索引

1. **图 1：LeWAM 总体训练架构**——shared encoder、SIGReg、MoT、next-latent prediction 与 action flow matching。
2. **图 2：方法定位**——相对 LeWM/V-JEPA 2、DINOv2/SIGReg、video WAM 与 population search 的差异。
3. **图 3：规划机制**——learned action proposals、潜在 rollout、gradient refinement、best-of-N。
4. **图 4：十环境主要结果**——短时域 visual planning 与 contact-rich 任务表现。
5. **图 5：消融与行为克隆**——SIGReg、encoder、flow vs. MSE 和 Diffusion Policy 对比。

---

## 14. 材料来源

- 论文主页：https://arxiv.org/abs/2610.12818v1
- 官方 PDF：[本地文件](assets/LeWAM%20-%20端到端JEPA世界动作模型/sources/paper.pdf)
- 项目页：https://le-wam.github.io/
- 官方代码：https://github.com/MinghaoFu/lewam
- 完整本地溯源表：[sources.md](assets/LeWAM%20-%20端到端JEPA世界动作模型/sources.md)
- Flow-JEPA 外部对照：https://arxiv.org/abs/2608.29029v3
