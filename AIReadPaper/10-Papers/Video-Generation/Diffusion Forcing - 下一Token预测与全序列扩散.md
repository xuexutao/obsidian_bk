---
type: paper-note
status: done
domain: Video-Generation
paper: "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion"
year: 2024
arxiv: 2407.01392
doi: null
source: https://arxiv.org/abs/2407.01392
project: https://boyuan.space/diffusion-forcing
code: https://github.com/buoyancy99/diffusion-forcing
tags:
  - Video-Generation
  - Diffusion-Forcing
  - Sequence-Modeling
  - Planning
created: 2026-09-04
updated: 2026-09-04
---

# Diffusion Forcing：下一 Token 预测与全序列扩散

## 基本信息

- **作者**：Boyuan Chen、Diego Martí Monsó、Yilun Du、Max Simchowitz、Russ Tedrake、Vincent Sitzmann
- **发表**：NeurIPS 2024
- **论文**：https://arxiv.org/abs/2407.01392
- **项目页**：https://boyuan.space/diffusion-forcing
- **官方代码**：https://github.com/buoyancy99/diffusion-forcing
- **阅读口径**：正文与数字以 arXiv v4 官方 PDF 为主，项目页与官方代码仅用于核对演示、数据和复现状态。

## 一句话结论

Diffusion Forcing 把扩散噪声解释为连续程度的“mask”，训练一个因果序列模型同时处理**每个 token 各不相同的噪声级别**，从而在同一模型中兼得下一 token 模型的可变长度、自回归与反馈控制，以及全序列扩散的联合生成和长视野 guidance；它在低分辨率视频、离线规划和机器人模仿学习上验证了这一统一视角，但原论文仍是 RNN 级概念验证，而不是可直接替代现代视频 DiT 的大规模生成系统。

---

## 1. 研究背景与问题定义

### 1.1 两类序列生成器各自缺什么

下一 token 模型按 $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$ 分解序列，天然支持只生成一步、生成任意长度、在线接收反馈和树搜索。然而在连续空间中，teacher forcing 训练时始终读取真实历史，推理时却读取自己的预测，微小误差会沿时间累积；在 Minecraft、DMLab 这类像素序列上，长于训练窗口后尤其容易发散。

全序列扩散则直接建模固定长度序列的联合分布，把所有 token 置于相同扩散时刻共同去噪。它擅长连续信号，也可以利用 classifier guidance 等机制让整条轨迹满足目标，但通常依赖非因果网络和固定窗口：想获得第一个 token 也要处理整个序列，难以流式输出、按需改变 horizon 或执行闭环控制。

![](assets/Diffusion%20Forcing%20-%20下一Token预测与全序列扩散/figures/01-capabilities.png)

> 图 1：论文用 guidance、tree search、compositionality、causal uncertainty 和 flexible horizon 五项能力概括两类范式的互补性。Diffusion Forcing 的目标不是简单折中，而是用统一训练目标覆盖两者的可用采样方式。  
> 来源：论文 Figure 1，第 2 页，https://arxiv.org/abs/2407.01392

### 1.2 核心问题

论文要解决的问题是：能否训练一个因果序列生成器，使它既能像 AR 模型一样按需生成任意前缀，又能像全序列扩散一样保留未来不确定性并对长轨迹做 guidance？其关键不是换一个网络，而是把“时间轴”和“扩散噪声轴”从绑定关系中解开。

### 1.3 核心贡献

1. 提出 **Diffusion Forcing（DF）**：对序列中每个 token 独立采样噪声级别，训练模型恢复任意部分被不同程度破坏的序列。
2. 给出因果实例 **Causal Diffusion Forcing（CDF）**，并证明其目标在适当条件下优化所有噪声模式、因而也包括所有子序列似然的变分下界。
3. 设计二维噪声调度矩阵，使同一模型可在推理时切换成下一 token、自回归滚动、全序列扩散或两者混合，而不重新训练。
4. 基于“近未来低噪、远未来高噪”的因果不确定性，提出 Monte Carlo Guidance（MCG），用于对未来分布的期望回报而非单条采样轨迹做 guidance。
5. 在视频预测、D4RL Maze2D 规划、真实机器人模仿学习和时间序列预测上展示统一模型的跨任务能力。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

给定连续序列 $x_{1:T}$，下标 $t$ 表示真实时间。对每个 token 独立选择扩散级别 $k_t\in\{0,\ldots,K\}$，得到

$$
x_t^{k_t}=\sqrt{\bar\alpha_{k_t}}x_t+\sqrt{1-\bar\alpha_{k_t}}\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
$$

$k_t=0$ 表示干净 token，$k_t=K$ 近似纯高斯噪声，中间值相当于“部分 mask”。因果模型以过去隐藏状态 $z_{t-1}$、当前带噪 token $x_t^{k_t}$ 和噪声级别 $k_t$ 为输入，预测噪声或干净 token，并更新隐藏状态 $z_t$。输出可以是下一 token、未来若干 token、固定窗口，或持续滚动的无限长序列。

原论文使用卷积 RNN 实现 CDF。作者明确指出 masked Transformer 也可实现相同条件依赖，但大规模 latent DiT 并未在本文实验中验证，因此不能把后续 Diffusion Forcing 2 的结果倒推为本论文结论。

### 2.2 与 Teacher Forcing、Full-sequence Diffusion 的统一关系

![](assets/Diffusion%20Forcing%20-%20下一Token预测与全序列扩散/figures/02-method-overview.png)

> 图 2：DF 在训练时为每个时间位置独立加噪；推理时可给各 token 安排不同的去噪进度。Teacher Forcing 对应“历史完全干净、只预测下一个 token”，全序列扩散对应“所有 token 噪声级别相同”，二者都可视为 DF 调度空间中的特殊切片。  
> 来源：论文 Figure 2，第 3 页，https://arxiv.org/abs/2407.01392

这里最重要的建模判断是：**噪声不仅是生成过程，也是信息可见度**。传统 mask 只有可见/不可见两档，而扩散噪声提供从完全可见到完全未知的连续刻度。因此，DF 学到的是“在任意历史可靠度与任意未来不确定度下恢复序列”的条件分布族。

---

## 3. 核心方法

### 3.1 独立逐 token 噪声训练

对训练轨迹 $x_{1:T}$，算法逐位置独立采样 $k_t$，前向加噪后依次更新因果隐藏状态：

$$
z_t=f_\theta(z_{t-1},x_t^{k_t},k_t),
\qquad
\hat\epsilon_t=\epsilon_\theta(z_{t-1},x_t^{k_t},k_t).
$$

训练损失是标准扩散噪声回归在序列上的求和：

$$
\mathcal L_{\mathrm{DF}}(\theta)=
\mathbb E_{x_{1:T},k_{1:T},\epsilon_{1:T}}
\left[
\sum_{t=1}^{T}
\left\|\epsilon_t-\epsilon_\theta(z_{t-1},x_t^{k_t},k_t)\right\|_2^2
\right].
$$

差异只在于 $k_{1:T}$ 不是一个共享标量，而是从 $[K]^T$ 独立均匀采样的向量。均匀采样迫使模型覆盖各种“哪些历史可信、哪些未来未知”的组合，避免只适配某条固定噪声斜线。其代价是组合空间巨大，有限容量模型不可能等精度覆盖所有模式；论文的理论正确性依赖充分表达能力，工程上仍需选噪声分布和损失权重。

### 3.2 ELBO 与“所有子序列”解释

论文 Theorem 3.1 / Appendix A 证明，上述目标可解释为对随机噪声序列下观测似然的 ELBO 重加权。对任意固定噪声模式分布 $\mathcal D(k_{1:T})$，相应下界中的项都包含在均匀训练目标覆盖的非负去噪项中；若隐藏状态充分表达历史、变分近似精确且各项均能达到最优，则优化 DF 也是最大化所有噪声模式似然的有效替代。

特别地，当 $k_t\in\{0,K\}$ 时，某些 token 完全可见、另一些完全被 mask，这就覆盖了从联合分布抽取任意子序列、条件补全任意未来的情形。应注意，这不是有限模型上“所有条件分布都同样准确”的保证，而是目标函数覆盖面的理论说明。

### 3.3 二维采样调度矩阵

推理用矩阵 $\mathbf K\in[K]^{M\times T}$ 描述：列对应真实时间 $t$，行对应采样迭代 $m$，元素 $K_{m,t}$ 指定第 $t$ 个 token 当前噪声级别。模型逐行、从左到右更新，最后一行满足 $K_{0,t}=0$。

这使多种行为仅由采样调度决定：

- **下一 token**：历史列保持 $0$，只把下一个位置从 $K$ 去噪到 $0$；
- **全序列扩散**：所有列共享相同噪声轨迹；
- **滚动生成**：前端 token 已干净，中段部分去噪，远端仍接近白噪声；
- **补全与条件化**：把可靠观测设为低噪，把缺失或不可信观测设为高噪；
- **可变 horizon**：增减列数即可改变预测长度，不改变参数。

### 3.4 稳定的长时自回归 rollout

普通 AR 模型把上一帧的点预测当作绝对真值继续递推。DF 则可在更新隐藏状态时把上一帧保持在一个很小但非零的噪声级别 $0<k\ll K$，显式表达“过去预测并非完全确定”。模型训练时见过不同可靠度的历史，因此能把小误差吸收到条件分布中，而不是把它当成真实世界结构继续放大。

![](assets/Diffusion%20Forcing%20-%20下一Token预测与全序列扩散/figures/03-video-rollout.png)

> 图 3：在 DMLab 与 Minecraft 上，从未见过的条件帧出发滚动到第 1000 帧。Teacher Forcing 很快退化为噪声或纹理，因果全序列扩散发生场景跳变；DF 仍维持可辨认且连贯的第一视角环境。  
> 来源：论文 Figure 3，第 7 页，https://arxiv.org/abs/2407.01392

### 3.5 Guidance、因果不确定性与 MCG

对未来轨迹加入目标 $c(x_{1:H})$ 后，可像普通扩散一样在部分去噪状态上加入 $\nabla_x\log c$。CDF 的因果结构让未来奖励的梯度反向影响当前 token，同时又允许远期 token 保持更高噪声，表达随 horizon 增长的不确定性。

Monte Carlo Guidance 不只沿一条未来样本计算梯度，而是从当前部分去噪状态采样多个未来，平均其 guidance：

$$
g(x_t)\approx\frac{1}{S}\sum_{s=1}^{S}
\nabla_{x_t}R\big(x_t,x_{t+1:T}^{(s)}\big).
$$

这近似“选择使未来期望回报最大”的动作，而不是追逐一次偶然的高回报 rollout。论文把这种做法类比为 MPPI 等 shooting 方法；它依赖 DF 可以在固定当前状态的同时反复重采样高噪未来，这是普通一次性全序列扩散较难表达的操作。

### 3.6 推理复杂度与工程行为

RNN 实现逐时间步更新隐藏状态，时间复杂度随 rollout 长度线性增长，内存主要由隐藏状态而非完整历史决定。代价是每个 token 内还要执行扩散去噪，真实速度受扩散步数影响；论文没有报告面向高清视频的 FPS，因为实验分辨率仅 DMLab 64×64、Minecraft 128×128，重点是长期稳定性而非生产级吞吐。

---

## 4. 数据集与实验设置

### 4.1 视频预测

- **DMLab**：第一视角导航视频，原始分辨率 64×64；论文模型训练窗口 36 帧。
- **Minecraft**：第一视角游戏视频，原始分辨率 128×128；训练窗口 72 帧。
- **共同结构**：Diffusion Forcing、Teacher Forcing 与 causal full-sequence diffusion 使用同一卷积 RNN，以隔离训练范式差异。
- **评测重点**：超出训练长度后的定性稳定性，而不是 FVD/VBench 等现代文生视频指标。主文展示 1000 帧，项目页和附录展示可超过 2000 帧且不使用 sliding window。

### 4.2 规划与控制

离线规划采用 D4RL Maze2D / Multi2D 的 U-Maze、Medium、Large 六个环境。token 取为 $x_t=[a_t,r_t,o_{t+1}]$，模型既生成动作，也预测执行动作后的奖励和观测。基线包括 MPPI、CQL、IQL、Diffuser、直接执行 Diffuser 生成动作的变体，以及去掉 MCG 的 DF。

真实机器人任务使用 Franka 机械臂交换苹果和橙子的位置。当前画面可能无法判断应移动哪种水果，因此任务要求记住初始状态；另加入遮挡和视觉干扰测试鲁棒性。

### 4.3 复现设置的可获得性

官方仓库保留论文使用的 `paper` 分支，并提供 DMLab、Minecraft 的 mini 数据与预训练 checkpoint 快速验证；完整数据下载、D4RL/MuJoCo 规划环境和机器人数据复现成本明显更高。项目主分支后来换成 temporal attention 重实现，性能与结构不应和论文 RNN 结果混为一谈。

---

## 5. 实验结果

### 5.1 视频长期稳定性

论文的核心视频证据是行为差异而非单一分数。三种方法在训练 horizon 内都能生成局部合理帧，但只有 DF 在 496、500、996、1000 帧仍保持几何和场景连续；Teacher Forcing 累积误差后崩成纹理，causal full-sequence diffusion 则出现帧间跳变。这个结果支持“对历史不确定性建模”能减轻连续空间 AR 漂移，但由于缺少现代高分辨率指标和人评，它不能证明 DF 在感知质量上优于大型视频扩散模型。

### 5.2 Maze2D 定量结果与 baseline

![](assets/Diffusion%20Forcing%20-%20下一Token预测与全序列扩散/figures/04-planning-results.png)

> 表 1 / 实验图：上半部分展示 DF 与 Diffuser 的去噪规划轨迹，下半部分给出六个 Maze2D / Multi2D 环境的平均回报。DF 使用因果不确定性与 MCG 后在全部环境取得最高分。  
> 来源：论文 Table 1，第 8 页，https://arxiv.org/abs/2407.01392

| 设置 | MPPI | CQL | IQL | Diffuser* | Diffuser 直接动作 | DF 无 MCG | **DF** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maze2D 单任务平均 | 16.2 | 7.7 | 47.0 | 119.5 | 8.7 | 129.67 | **141.7** |
| Multi2D 多任务平均 | 21.5 | — | 16.9 | 129.4 | 20.6 | 127.7 | **146.2** |

最重要的不是仅比 Diffuser* 高约 18–17 分，而是 **Diffuser 直接执行自己生成的动作时平均分骤降到 8.7 / 20.6**；其公开实现依赖手工 PD controller 把生成状态翻译成动作。DF 将 token 设为动作—奖励—下一观测并保持因果一致，生成动作可直接执行。去掉 MCG 后仍强于多数基线，但单任务平均从 141.7 降到 129.67、多任务从 146.2 降到 127.7，说明对未来分布取期望确有独立贡献。

### 5.3 机器人实验与鲁棒性

在水果换位任务上，DF 成功率为 **80%**，而无记忆的 Diffusion Policy 无法完成需要区分初始配置的任务。加入视觉干扰或完全遮挡后，DF 把异常观测标为 $k>0$、更多依赖先验动态，成功率仅降至 **76%**；next-frame diffusion baseline 为 **48%**。这组实验直接体现了“噪声级别是观测可信度”的 Bayes filtering 解释。

### 5.4 消融、失败案例与证据边界

- **MCG 消融**：Maze2D 平均回报明显下降，是主文最清晰的组件消融。
- **训练范式对照**：视频实验固定网络结构后，Teacher Forcing 长 rollout 发散，causal full-sequence diffusion 在窗口内也有帧间不连续。
- **失败案例**：项目页披露 Minecraft 中角色会在两格高的泥土或石块前卡住，反映训练数据行为分布本身的局限；视频仍可能语义停滞，只是像素不爆炸。
- **缺失证据**：论文未给高分辨率 T2V 的 FVD/VBench、人评、端到端 FPS，也未系统报告噪声采样分布、稳定化噪声大小和模型规模的消融。

---

## 6. 与相关工作的关系

Teacher Forcing 只训练“干净历史 → 下一 token”；full-sequence diffusion 训练“全序列同噪声 → 全序列”；Rolling Diffusion 预设沿时间变化的噪声形状；DF 则训练任意逐 token 噪声组合，再在推理时选择调度。后续 History-Guided Video Diffusion / Diffusion Forcing 2 把该思想扩展到 Wan2.1、latent diffusion 和 DiT，但它们解决了原论文尚未验证的规模化问题，不能视作同一实验。

与后来的 CausVid、Self-Forcing、Causal Forcing 相比，本论文讨论的是**如何训练原生序列扩散模型**；后三者主要讨论**如何把已有双向视频扩散模型蒸馏成 few-step 因果生成器**。它们共享逐块因果和训练—推理分布问题，但技术目标不同。

---

## 7. 局限与批判性评价

作者明确承认当前实现基于 RNN，尚未研究互联网规模数据和 Transformer 扩展。除此之外，我认为还有四个边界：

1. 理论结论依赖充分表达的隐藏状态与近似可达最优，现实模型在指数级噪声组合上的容量分配仍是开放问题。
2. 视频结论建立在 64×64 / 128×128 游戏环境，稳定 rollout 不等于高保真开放域视频生成；其成功更接近世界模型中的长期状态保持。
3. “无限 rollout”主要指不数值爆炸，不能保证长期语义新颖性、任务进展或物理正确性；Minecraft 卡住就是反例。
4. MCG 的多未来采样会提高规划时计算成本，论文没有给出回报—延迟—样本数的完整 Pareto 曲线。

因此，这篇工作的长期价值在于训练与采样坐标系，而不是某个视觉 benchmark 的领先成绩。

---

## 8. 复现与实践建议

- **最低成本验证**：使用官方 mini DMLab/Minecraft 数据和预训练 checkpoint，先复现长 rollout；不必从头下载完整数据。
- **严格复现论文**：切换到官方仓库 `paper` 分支。主分支是后续 temporal-attention 重实现，结果不能直接对照论文 RNN。
- **环境成本**：Python 3.10、PyTorch、Weights & Biases；规划实验还需旧版 D4RL 与 MuJoCo 2.1，依赖安装比视频预测更脆弱。
- **训练资源**：论文未披露完整 GPU 型号、卡数和总 GPU 小时。数据与 checkpoint 可用，但严格算力成本无法从论文复原。
- **实现检查点**：必须确认每个 token 的 $k_t$ 独立采样并传入网络；若整段共享同一 $k$，就退化成因果全序列扩散。长 rollout 时还要复现“小非零历史噪声”的稳定化设置。

---

## 9. 个人判断与后续问题

我认为 DF 最有启发性的命题是“**不确定性也是上下文的一部分**”。普通 AR 模型只记录历史值，DF 同时记录历史值和对它的置信度；这使生成、过滤和规划可由同一状态更新器表达。它也解释了为什么后续视频工作反复回到训练—推理条件分布的一致性：一旦模型把自己生成的历史误当成无误差真值，长序列必然漂移。

后续值得继续追问：独立均匀采样 $k_t$ 是否浪费容量，能否根据时序相关长度自适应选择噪声块；Transformer/SSM 是否能以缓存方式保持 DF 的任意噪声调度；以及如何把“像素稳定”扩展为可度量的长期因果、物理和语义一致性。

---

## 10. 材料来源

本地 PDF、四张完整关键图及逐项来源记录见：

- [sources.md](assets/Diffusion%20Forcing%20-%20下一Token预测与全序列扩散/sources.md)
- [paper.pdf](assets/Diffusion%20Forcing%20-%20下一Token预测与全序列扩散/sources/paper.pdf)
