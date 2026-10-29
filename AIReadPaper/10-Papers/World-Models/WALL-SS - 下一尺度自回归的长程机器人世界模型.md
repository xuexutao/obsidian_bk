---
title: "WALL-SS：下一尺度自回归的长程机器人世界模型"
aliases:
  - "WALL-SS"
  - "Scaling Long-horizon World Models via Next-Scale Autoregression"
authors: "Maeve Zhang, Rain Sun, Xiang Wang, Cyril Zhang, Shalfun Li, et al."
institution: "X Square Robot Team（自变量机器人）"
date: 2026-08-26
arxiv: "2608.26239"
project: "https://x2robot.com/en/pages/ss"
code: "截至 2026-10-29 未开源；官方仓库仅含项目页、论文和展示图片，训练/推理代码仍为 TODO"
tags:
  - paper
  - world-model
  - robotics
  - autoregressive-video
  - long-horizon-memory
  - policy-evaluation
rating: "★★★★☆（4.5/5）"
---

# WALL-SS：下一尺度自回归的长程机器人世界模型

> [!abstract] 一句话总结
> WALL-SS 把机器人轨迹建模为**观测与动作交错的因果序列**，用 next-scale autoregression 在每个未来时刻从粗到细生成视觉状态，再以“近处细、远处粗”的 time–scale KV memory 支撑分钟级流式 rollout，并通过动作一致性与长期一致性奖励对视觉生成策略做 on-policy 对齐。它最有价值的结果不是画面指标，而是：在 600 对匹配的仿真—真机闭环实验中，世界模型能以 0.89 的 pairwise accuracy 排序五个策略 checkpoint，说明生成式世界模型开始具备**低成本策略评估器**的雏形。

![](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/figures/01-overview.jpg)

*图 1：WALL-SS 总览。异构机器人/UMI 数据进入 next-scale 自回归模型，生成状态通过有界时间–尺度记忆递归传播；外部策略再消费生成观察形成闭环。来源：论文 Figure 1 / PDF p.2，官方 GitHub 原图。*

---

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | **WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression** |
| 团队 | X Square Robot Team（自变量机器人） |
| 核心作者 | Maeve Zhang、Rain Sun、Xiang Wang、Cyril Zhang、Shalfun Li；项目负责人 Shalfun Li，通讯作者 Hao Wang |
| 时间 | arXiv v1：2026-08-26；PDF 标注 2026-08-28 |
| 论文 | [arXiv](https://arxiv.org/abs/2608.26239) · [PDF](https://arxiv.org/pdf/2608.26239) |
| 项目页 | [WALL-SS](https://x2robot.com/en/pages/ss) |
| 官方仓库 | [X-Square-Robot/wall-ss](https://github.com/X-Square-Robot/wall-ss) |
| 代码状态 | **未开源**。截至 2026-10-29，仓库只有 README、论文、三张展示图和 MIT License；训练/推理代码仍在 TODO，未见权重、配置和数据脚本 |
| 基座 | InfinityStar next-scale 视频生成骨干 |
| 核心能力 | 动作条件未来生成、60 秒流式 rollout、闭环策略评估、联合动作专家真机控制 |
| 本文评级 | **★★★★☆（4.5/5）**：问题定义和 sim-real 评估非常强，但复现信息严重不足，结论依赖私有数据与内部评测 |

---

## 1. 研究背景：为什么“会生成视频”还不等于“世界模型”

### 1.1 Clip-level 视频预测的四个结构性缺口

机器人世界模型不仅要预测“下一段看起来合理的视频”，还要回答：**给定同一个现场，如果机器人执行不同动作，世界分别会怎样变化？** 传统视频扩散或 action-conditioned video diffusion 常以固定长度 clip 为基本单位，因此存在四个缺口：

1. **动作与结果耦合不够强**：场景和任务先验往往足以猜到一个“最可能成功”的未来，模型可能忽略具体控制轨迹。
2. **预测时长固定**：一次输出固定帧数，不自然支持持续交互和任意长度 rollout。
3. **递归生成会遗忘或爆内存**：只看最近 clip 会丢失早期物体状态；保存所有历史 token，又会让 KV cache 随时间线性增长。
4. **扩散模型不方便直接做奖励优化**：WALL-SS 希望得到显式 token 概率，把视觉生成本身当作可用 PPO/GRPO 更新的随机策略。

论文的核心主张是：问题不只是“加一个 action condition”，而是要找到一个同时统一**视觉、动作、记忆和奖励优化**的生成表述。

### 1.2 论文真正关心的是策略评估

WALL-SS 最终目标不是取代真机，而是在真机部署前回答两个工程问题：

- 某个策略 checkpoint 在这个任务上大概有多强？
- 多个 checkpoint 的优劣排序在生成世界和真实世界里是否一致？

这比单纯比较 FID/FVD 更严格。只要世界模型在接触、失败或恢复阶段过于乐观，就会错误地把弱策略排在前面，直接损害 checkpoint selection。

---

## 2. 核心贡献

1. **把具身轨迹改写成观测—动作交错的因果序列。** 每个未来观察都由对应动作块、任务文本和已提交历史共同决定，自然支持可变长度和流式扩展。
2. **提出 action-conditioned next-scale prediction。** 动作不是一次性的全局 prompt，而是按尺度对齐，在每个粗到细视觉决策中持续可见；多视角只交换同步且因果可用的视觉状态。
3. **提出 scale-compressed time–scale memory。** 最近历史保留细粒度 KV，远期历史只保留粗尺度 KV，初始观察作为永久身份锚点；固定淘汰计划让内存不随 rollout 时长增长。
4. **用 scale-wise dream forcing 缩小 exposure gap。** 训练时主动让模型从腐化历史和 detached 自生成后缀继续预测，避免推理时被自身小误差逐步拖垮。
5. **把自回归视觉动态当作策略做 on-policy 对齐。** 通过 action-following 与 long-horizon consistency 奖励、组相对优势、KL 约束和真实轨迹 replay 修正长程漂移。
6. **给出面向策略评估的 sim-real 验证。** 在 600 对闭环 rollout 中比较同一个外部策略在生成世界与真机中的结果，而不是只做 open-loop 视频重放。

---

## 3. 方法：Generate、Memory、Align

![](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/figures/02-framework.jpg)

*图 2：整体框架。左侧将多视角观察、动作块和任务文本融合进 causal Transformer；中部使用 next-scale AR 与 time–scale KV memory；右侧在训练期以新鲜 rollout 做 on-policy visual alignment。来源：论文 Figure 3 / PDF p.7，官方 GitHub 原图。*

### 3.1 统一因果序列：物理时间 × 视觉尺度

WALL-SS 沿两个维度展开自回归：

- **物理时间维**：第 $c$ 个 clip 的动作 $A_c$ 导致未来观察 $V_c$，完成后将状态提交给下一 clip。
- **尺度维**：同一个未来观察不是一次生成，而是按 coarse → mid → fine 逐尺度预测残差视觉代码。

因此，模型的因果顺序可以理解为：先在当前物理区间内生成粗结构，再补中尺度和细节；完整多视角 bundle 提交后，才进入下一个物理时间段。相比“每段视频独立扩散”，这里的状态、动作和生成概率属于一条持续演化的序列。

### 3.2 动作条件 next-scale prediction

#### 3.2.1 动作如何与视频对齐

对每个相机视角 $v$，确定性渲染器 $\mathcal{R}^{(v)}$ 使用当前 rollout 状态、相机标定和指定控制，把末端执行器轨迹投影到对应视频时间轴。它**不读取未来真实遥测或目标视频**。动作编码器再按与 RGB 相同的 clip 区间切片，得到多尺度动作条件：

$$\mathbf{X}^{A}_{c,j},\quad j\in\mathcal{S}_A$$

对视觉尺度 $\ell$，$\kappa(\ell)$ 选择不超过该视觉尺度的最细动作尺度。这样 coarse visual token 读取 coarse action，fine visual token 才读取更精细的控制，而不是将所有动作粗暴池化成一个向量。

#### 3.2.2 严格的因果掩码

位于 $(c,\ell,r,v)$ 的视觉查询只能读取：

- 任务文本 $T_g$；
- 当前 clip、当前视角、匹配尺度的动作 $(c,\kappa(\ell),v)$；
- 已合法提交的视觉状态与记忆。

它不能直接读取其他相机的动作、相邻 clip 的动作或未来控制。多视角通过 residual cross-view adapter 交换同步视觉状态，但不会破坏直接动作条件的 block-diagonal mask。

#### 3.2.3 自回归目标

论文对 clip、尺度和 residual repeat 做均衡采样，最小化：

$$\mathcal{L}_{\mathrm{AR}}=-\mathbb{E}_{\mathcal{D}}\mathbb{E}_{(c,\ell,r)\sim\mathrm{Bal}}\left[\frac{1}{N_{c,\ell,r}}\log p_\theta\left(\mathbf{Z}_{c,\ell,r}\mid\mathbf{Z}_{c,\prec(\ell,r)},\mathcal{M}_c,\mathbf{X}^{A}_{c,\kappa(\ell)},T_g\right)\right]$$

这里的关键不是普通 next-token loss，而是**动作被放进每一个构造未来视觉状态的因果决策里**。缺少可靠动作标定的样本仍可训练视觉似然，但动作条件置为空。

![](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/figures/04-counterfactual-actions.png)

*图 3：固定同一个初始观察，只改变未来控制轨迹，三行生成出几何上不同的未来；蓝线是命令轨迹，橙线是生成视频中恢复出的实际轨迹。来源：论文 Figure 7 / PDF p.16。*

### 3.3 Scale-compressed time–scale memory

#### 3.3.1 “近处细、远处粗”

只保留最新 clip 会丢掉持久状态；保存全历史 KV 又不可扩展。WALL-SS 反向利用 next-scale 层级：

- 初始观察 $O_0$ 形成永久 anchor $H_{\mathrm{anc}}$，稳定对象身份与全局布局；
- 最近 clip 保留 fine-scale causal state；
- 中期历史保留 mid-scale state；
- 更远历史只保留 coarse state；
- 每条视觉记忆始终与产生它的动作历史配对。

在预预测边界 $s$ 记录：

$$\mathbf{B}_{u,s}=\left(\mathbf{H}^{\mathrm{pre}}_{u,s},\mathsf{A}_u\right),\quad \mathsf{A}_u=\left\{\mathbf{X}^{A}_{u,j}\right\}_{j\in\mathcal{S}_A}$$

第 $c$ 个 clip 使用的记忆为：

$$\mathcal{M}_c=\left\{\mathbf{H}_{\mathrm{anc}}\right\}\cup\left\{\mathbf{B}_{c-\delta,s(\delta)}\mid\delta\in\mathcal{D}_c\right\},\quad\delta_1<\delta_2\Rightarrow s(\delta_1)\ge s(\delta_2)$$

年龄越小，使用的尺度越细。这里不是在线把 fine KV 压缩成 coarse KV，而是生成过程中提前保留多个可寻址边界，随着年龄变化切换读取哪一级。

#### 3.3.2 有界内存

记录在最后一次计划读取后淘汰，因此：

$$\sup_c\operatorname{mem}\left(\mathcal{R}^{\mathrm{mat}}_c\right)\le B_{\mathrm{KV}}$$

理论上，持久历史内存和单 clip 工作内存都不随已完成 clip 数增长。遗憾的是，论文没有给出 $W_M$、$B_{\mathrm{KV}}$、各尺度 token 数或真实显存曲线，无法判断该设计在工程上究竟节省多少。

### 3.4 Scale-wise dream forcing：训练时模拟“吃自己输出”

teacher forcing 永远看到干净历史，而流式推理持续消费模型自己的生成结果。WALL-SS 用两种方式制造不完美历史：

1. 按年龄腐化记忆 $\widetilde{\mathcal{M}}_c\sim\mathcal{C}_{\mathrm{age}}(\mathcal{M}_c)$；
2. 用滞后模型生成 detached 后缀 $\widehat{\mathbf{Z}}_{c:c+d-1}$，再要求当前模型从这段自生成上下文预测干净未来。

$$\mathcal{L}_{\mathrm{self}}=-\log p_\theta\left(\mathbf{Z}^{\mathrm{gt}}_{c+d}\mid\operatorname{sg}\left[\mathcal{U}^{(d)}(\mathcal{M}_c,\widehat{\mathbf{Z}}_{c:c+d-1},\mathbf{X}^{A}_{c:c+d-1})\right],\mathbf{X}^{A}_{c+d},T_g\right)$$

总的长程目标为：

$$\mathcal{L}_{\mathrm{long}}=\mathcal{L}_{\mathrm{AR}}+\lambda_{\mathrm{cor}}\mathcal{L}_{\mathrm{cor}}+\lambda_{\mathrm{self}}\mathcal{L}_{\mathrm{self}}$$

它本质上是**监督式 exposure-gap 训练**，并不是 RL。消融结果也证明，保留相同推理记忆但去掉 dream forcing 后，轨迹误差、边界误差和状态一致性会随时长明显恶化。

### 3.5 World Action Policy：世界模型和动作专家共享已提交状态

论文联合训练一个 flow-matching action expert。它只读取已提交的视觉记忆、当前本体状态和任务文本：

$$H_c^A=F_\omega^A\left(\operatorname{Read}_V(\mathcal{M}_c),E_s(s_c),T_g\right)$$

noisy action slots 对该上下文 cross-attention，预测下一个动作块 $A_c$。关键边界是：

- 动作专家不能看到当前视觉目标、当前动作条件或未来记录；
- 可执行动作 $A_c$ 与视觉条件 $X_c^A$ 是两种对象；
- 预测动作和外部动作都经同一个 grounding map $\Gamma$ 转成视觉条件；
- 视觉模型不读取动作专家的私有 hidden state。

联合目标为：

$$\mathcal{L}_{\mathrm{cotrain}}=\mathcal{L}_{\mathrm{long}}+\lambda_A\mathcal{L}_{\mathrm{act}}$$

这使预测世界和选择动作共享同一“已提交因果状态”，而不是两个互不相干的模型串联。

### 3.6 On-policy alignment：优化视觉动态，不优化机器人策略

#### 3.6.1 奖励定义

冻结 behavior copy 在相同初始状态、任务和指定动作下采样 $K\ge2$ 条新鲜 rollout。两个冻结评分器分别衡量：

- $R_{\mathrm{act}}$：生成转移是否服从指定动作；
- $R_{\mathrm{long}}$：状态、clip 边界和跨视角是否长期一致。

$$r_{i,c}=\lambda_{\mathrm{act}}R_{\mathrm{act}}(\widehat{\mathbf{V}}^i_{c-1:c},A_c)+\lambda_{\mathrm{long}}R_{\mathrm{long}}(O_0,\widehat{\mathbf{V}}^i_{b_c-1:c},\mathbf{A}_{b_c:c})$$

**任务进度和最终成功不进入奖励**，避免模型直接对闭环评测目标投机。

#### 3.6.2 组相对更新与先验约束

同一条件下的 $K$ 条 rollout 做组内标准化，不训练独立 critic；视觉 token 使用 PPO/GRPO 风格裁剪概率比。最终再加两个稳定项：

$$\mathcal{L}_{\mathrm{align}}=\mathcal{L}_{\mathrm{PG}}+\beta\,\mathbb{E}\left[D_{\mathrm{KL}}(\pi_\theta^V\Vert\pi_{\mathrm{ref}}^V)\right]+\lambda_{\mathrm{AR}}\mathcal{L}_{\mathrm{AR}}^{\mathrm{real}}$$

- KL 约束模型别偏离预训练视觉分布；
- 真实轨迹 AR replay 防止 reward hacking 和画质退化；
- 只更新视觉生成器及条件 adapter；tokenizer、评分器、参考生成器和动作模块冻结。

从受监督 checkpoint 到对齐后，Action Following 从 0.264 升到 0.290，Trajectory Accuracy 从 0.512 升到 0.539，Cross-clip Boundary Error 从 0.118 降到 0.104；真实机器人 Task Progress 也从 64.6 升到 69.1。这说明奖励主要修复 residual dynamics error，而不是重新学习世界。

---

## 4. 数据：为什么失败与恢复样本是关键

### 4.1 数据组成

| 来源 | 规模/作用 |
|---|---|
| AgiBotWorld-Beta | 原始 1,003,672 条轨迹，100+ 机器人、100+ 真实场景；处理后得到 987,508 个带 caption clips，来自 165,560 个唯一视频 |
| ManipArena | 扩展推理型与接触密集的真机操控任务 |
| X2-Robot 私有数据 | 双臂、多视角、机器人状态、末端轨迹和夹爪状态；缩小目标平台的 embodiment/camera gap，规模未披露 |
| 非机器人 UMI | 手持夹爪第一视角交互；标定完整时转为统一末端表示，否则只作为普通视频 |
| 干预与失败恢复 | human takeover、missed/unstable contact、misalignment、re-grasp、drop、collision、retry、rollback 等 |

所有样本统一到 **10 FPS**。只有同时通过时间戳、内外参和运动学检查的样本才进入 action-conditioned pool；其他视觉有效样本降级为 video-only pool，而不是附上错误动作监督。

### 4.2 失败数据解决“磁吸抓取”捷径

若训练集几乎都是成功示教，“闭合夹爪”后通常紧跟“物体被提起”。生成模型很容易学成：无论是否真正接触，只要看到抓取意图，就让物体自动贴到夹爪上。

论文保留两类干预：

- **Immediate intervention**：策略出现错误趋势时立即人工接管，从当前失败状态继续；
- **Rollback-and-replay**：先保留错误分支，再回到近似错误前状态，执行纠正控制，形成上下文近似但动作和结果不同的局部配对。

这类数据迫使模型区分“动作意图”和“物理结果”。不过物理 rollback 无法精确复原物体姿态、接触和柔顺性，因此它只是**局部匹配干预样本**，不是严格反事实。

### 4.3 动作如何变成视觉条件

不同本体的动作被转换为统一的左右腕 6-DoF 位姿和归一化夹爪开度，再用相机标定投影到主视角：

$$\widetilde{\mathbf{u}}=\mathbf{K}_h\mathbf{T}_{h\leftarrow e}\widetilde{\mathbf{x}}_e,\qquad \mathbf{u}=(\widetilde u/\widetilde w,\widetilde v/\widetilde w)$$

投影视频绘制末端中心、局部方向轴、左右臂和夹爪状态，并以标记尺度编码相机相对深度。这样动作流和 RGB 共享时间戳、空间变换与 tensor layout；代价是强依赖可靠标定。

---

## 5. 训练与推理流程

### 5.1 三阶段训练

1. **Action-grounded AR learning**：从 InfinityStar 初始化；冻结视觉 tokenizer，联合训练自回归生成器、动作/视角 adapter 和动作专家。
2. **Long-horizon adaptation**：在连续 multi-clip 窗口上启用 time–scale memory，混合干净历史、年龄腐化历史和 detached 自生成后缀。
3. **Visual-policy alignment**：训练动作与长期一致性 scorer，然后对新鲜 rollout 做组相对策略优化，并用 reference KL 与真实 AR replay 保持视觉先验。

论文没有披露参数量、层数、宽度、token 尺度、学习率、batch size、训练步数、硬件或推理吞吐。它给出了算法，但没有给出可复现 recipe。

### 5.2 流式推理状态机

每个 clip 的顺序固定为：

1. 组装 anchor 与按年龄选择的 time–scale memory；
2. 编码当前动作块；
3. 按 coarse → fine 生成所有同步视角；
4. 完整 bundle 生成完后一次性 commit；
5. 记忆记录老化、切换尺度、淘汰过期项；
6. 解码观察并进入下一 clip。

同一个 committed state 可以 fork 不同动作或随机种子，得到反事实未来。闭环评估中，外部策略读取最新生成观察并输出下一动作；初始化后不再注入任何真实未来帧或真实动作。

---

## 6. 实验结果

### 6.1 具身视频生成：强在动作与交互，不是单帧画质

评测包含 200 个 ID + 100 个 OOD 任务，覆盖新 object–verb 配对、改写指令、未见场景和组合任务；80% 有同步控制轨迹，20% 为 text/image-to-video。

| 模型 | Interaction ↑ | Instruction Following ↑ | Trajectory Accuracy ↑ | Action Following ↑ |
|---|---:|---:|---:|---:|
| InfinityStar | 0.484 | 0.406 | 0.251 | – |
| Wan2.2-14B | 0.476 | 0.394 | 0.159 | – |
| CogVideoX-5B | 0.400 | 0.380 | 0.177 | – |
| Cosmos3-Nano | 0.516 | 0.410 | 0.202 | 0.044 |
| **WALL-SS** | **0.546** | **0.471** | **0.539** | **0.290** |

WALL-SS 的 Image Quality 为 0.697，并非第一（CogVideoX-5B 为 0.718）；Motion Smoothness 0.715，也低于 Wan2.2-14B 的 0.793。它真正领先的是交互质量、指令跟随、轨迹准确率和动作跟随。因此论文的证据支持“更像机器人世界模型”，但不支持“所有视觉指标都最好”。

### 6.2 60 秒长程 rollout 与记忆消融

![](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/figures/05-long-horizon-ablation.png)

*图 4：Figure 9 的关键消融。recent-clip only 在约 30 秒后轨迹/边界误差快速增大、状态一致性跌破 0.5；去掉 dream forcing 也持续退化；完整模型接近 full-history KV，但使用固定预算。来源：PDF p.18。*

四个变体的含义：

- **Ours**：time–scale memory + dream forcing；
- **w/o dream forcing**：推理记忆相同，但训练时没见过自生成历史；
- **recent-clip only**：只保留最新 clip；
- **full-history KV**：保存所有历史状态，作为质量上界，但内存/注意力开销持续增长。

完整模型在 60 秒内保持更低的轨迹误差与边界误差、更高的状态一致性，接近 full-history KV；这支持两个独立结论：长期状态需要远期记忆，而记忆还必须在自生成上下文上训练。Figure 10 又显示它在 successive 3-second windows 上的 PSNR/FID/FVD 全程优于对比视频模型；Figure 11 的 3D probe 与跨视角像素对应误差处于更优的左下区域。

但论文只给曲线，没报告 time–scale memory 的 token 数、显存或延迟，因此“固定预算”只在复杂度形式上成立，工程收益无法定量验证。

### 6.3 最重要实验：sim-real 策略评估一致性

![](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/figures/03-closed-loop-calibration.jpg)

*图 5：左为 30 个 task–checkpoint cell 的真机/生成成功率校准；右为六个任务内五个 checkpoint 的排序。来源：论文 Figure 12 / PDF p.20，官方 GitHub 原图。*

实验使用：

- 6 个 held-out 任务；
- 5 个 WALL-WM 策略 checkpoint；
- 每个任务/checkpoint 20 个匹配初始状态；
- 共 $6\times5\times20=600$ 对生成—真机闭环 rollout；
- 盲评员标注成功与有序子任务，无法恢复的生成伪影按失败计。

| 评价层级 | 指标 | WALL-SS |
|---|---|---:|
| Task-level calibration | MAE ↓ | 0.062 |
|  | Bias → 0 | +0.028 |
|  | Slope → 1 | 0.84 |
|  | Pearson $r$ ↑ | 0.93 |
| Within-task ranking | Pairwise Accuracy ↑ | 0.89 |
|  | Mean regret ↓ | 0.025 |
|  | Mean Spearman $\bar\rho$ ↑ | 0.88 |
| Episode agreement | Balanced Accuracy ↑ | 0.88 |
|  | Success Recall ↑ | 0.90 |
|  | Failure Recall ↑ | 0.86 |
|  | False Positive Rate ↓ | 0.136（表中四舍五入 0.14） |

模型在 59 个无平局真机 checkpoint pair 中保留了 89% 的正确次序，并在 6 个任务中选对 4 个最佳 checkpoint。平均 regret 只有 0.025。这比“生成画面看起来像”更接近世界模型的下游用途。

![](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/figures/06-sim-real-outcomes.png)

*图 6：600 对 episode 中，527 对生成与真实结果一致；残余乐观偏差集中在 contact/insertion。来源：论文 Figure 13 + Table 3 / PDF p.21。*

同时必须重视风险方向：332 个真机失败中，有 45 个在模拟里被判成成功，FPR 为 0.136。偏差在 grasp 阶段约 −0.01、transport/alignment +0.04、contact +0.08，插入阶段条件转移偏差最高 +0.12。也就是说，WALL-SS 仍会对最关键的接触/插入阶段**系统性乐观**，这正是策略筛选最危险的错误。

### 6.4 真机动作专家

![](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/figures/07-real-robot-results.png)

*图 7：联合训练的动作专家在七类桌面双臂任务上平均 Task Progress 69.1，高于 $\pi_{0.5}$、DreamZero 和 LingBot-VA。来源：论文 Figure 14 / PDF p.22。*

平均 Task Progress：

| 方法 | 平均分 |
|---|---:|
| **WALL-SS** | **69.1** |
| $\pi_{0.5}$ | 49.6 |
| DreamZero | 44.1 |
| LingBot-VA | 34.0 |

WALL-SS 在 Pick Items into Basket 达到 98.5，在 Sort Headphone / Classify Items as Shape / Press Button in Order 上分别为 70 / 80 / 63。作者将优势归因于动作专家与视觉生成器共享 committed causal state，使过去完成的子目标和动作后果可被保留。

这里应谨慎：论文报告的是 dense Task Progress，不是每个任务完整的 strict success rate；私有平台数据、训练规模和试验次数也未披露，无法判断统计稳定性。

---

## 7. 局限与批判性评价

### 7.1 最大优点：评测目标终于对准“世界模型有没有用”

这篇工作最强的不是 next-scale 本身，而是把评测从 open-loop 视频相似度推进到**同一策略、同一初始状态、生成世界与物理世界的闭环结果一致性**。校准、任务内排序、episode confusion matrix 和阶段偏差共同组成了相对完整的策略评估证据链。

### 7.2 最大缺陷：可复现性几乎为零

截至 2026-10-29：

- 训练/推理代码和权重未发布；
- 私有 X2-Robot 数据规模未披露；
- 模型参数量、层数、宽度、尺度表和 memory budget 未披露；
- 优化器、学习率、batch size、训练步数和硬件未披露；
- 推理帧率、单 clip 延迟、显存和 full-history 对比成本未披露。

因此读者可以理解算法，却无法重建系统或验证“bounded memory”的真实性价比。

### 7.3 内部 benchmark 和私有 scorer 可能形成闭环偏差

Embodied Video Generation benchmark 来自团队的数据混合；action/long-horizon scorer 的结构与训练规模也未公开。虽然 RL 奖励排除了任务进度与成功率，降低了直接泄漏，但动作表示、训练分布和评测协议仍可能共享同一套标定与数据先验。

### 7.4 60 秒 rollout 不等于真正的长期物理模拟

Figure 9 的曲线很有说服力，但评测仍是固定动作重放和短时窗口度量。模型没有显式接触求解、力、摩擦、质量或 3D 状态，长期一致性仍是视觉 token 层面的统计维持。对布料、液体、遮挡后接触状态和高精度插入，像素级世界模型仍可能“看起来连续但物理上错”。

### 7.5 Sim-real 校准已有价值，但还不能安全替代真机

0.89 的 pairwise accuracy 很强，但仍意味着约一成无平局 checkpoint pair 会排错；最佳 checkpoint 只选对 4/6 个任务。尤其 13.6% 的 real-failure false positive 会静默提升弱策略。更合理的应用是**先用世界模型大规模筛选，再用少量真机确认**，而不是完全替代真机评估。

### 7.6 真机结果的报告粒度不足

Figure 14 使用 Task Progress，缺少每项任务的试验次数、置信区间和完整 strict success rate 表。WALL-SS 在 Put Ring onto Rod 上并非最佳，但正文没有给出精确分数。真机结论仍需要更多独立平台、未见本体与接触任务验证。

---

## 8. 开源与复现状态

官方仓库 commit `04951f54db010ba289409a210fb8cb31ab7ea6f7` 只包含：

- `README.md`；
- `wall-ss-paper.pdf`；
- `assets/teaser.jpg`、`framework.jpg`、`closed-loop-calibration.jpg`；
- 社区二维码与 MIT License。

README 明确写着“Code will be released under the MIT License”，TODO 中“Release the training and inference code”仍未完成。因此本笔记**不写伪代码级张量流**，也不把 InfinityStar 的公开配置当作 WALL-SS 配置。若后续代码发布，应补做：

1. 视觉 token 的每级尺寸、残差重复次数和生成顺序；
2. action video 经 VAE 后的具体 tensor shape 与 scale alignment；
3. time–scale memory 的边界集合、aging schedule、KV 数量与显存；
4. dream forcing 的 rollout 深度 $d$ 与滞后模型更新方式；
5. scorer、PPO/GRPO 的 batch、clip range、KL 系数与实际训练成本。

---

## 9. 与相关方向的关系及启发

### 9.1 与视频扩散世界模型的差别

WALL-SS 不再把“每个 clip 反复去噪”当作唯一生成原语，而是显式保留 token 级自回归概率。这样可变长度、KV reuse 和 RL 对齐更加自然，但也引入 next-scale 离散生成的级联误差，因此必须配合 dream forcing 和长期记忆。

### 9.2 与 [[Masked Visual Actions - 像素掩码动作统一世界建模]] 的互补

两者都把动作提升为空间对齐的视觉条件，而不是抽象低维向量：Masked Visual Actions 用可见像素掩码轨迹，WALL-SS 用投影后的双臂末端轨迹和尺度对齐动作 token。前者强调跨本体与 forward/inverse 统一，后者强调长程因果状态与策略评估。共同启发是：**动作若要真正约束视频，最好先与图像坐标、时间轴和局部接触区域对齐。**

### 9.3 对后续研究最值得借鉴的三点

1. **世界模型评测应看 policy ranking，而不只是生成指标。** checkpoint selection regret 是比 FVD 更有工程意义的指标。
2. **长期记忆可以沿生成层级做非均匀保真。** 远期历史未必需要所有 fine token，但动作历史不能与视觉状态脱钩。
3. **失败数据不是噪声，而是物理因果监督。** 成功示教只告诉模型“常见结果”，干预和失败分支才告诉模型“动作变化会怎样改变结果”。

### 9.4 一个值得追问的问题

WALL-SS 用粗尺度 KV 保留远期状态，但“粗视觉尺度”是否真的等价于“长期任务状态”？对象是否已经抓取、抽屉是否关闭、某个子目标是否完成，可能是低空间频率但高语义精度的信息。未来更合理的方向可能是把视觉 time–scale memory 与显式 object/contact/event state 结合，而不只依赖生成骨干内部 KV。

---

## 10. 最终评价

**评级：★★★★☆（4.5/5）**

WALL-SS 是一篇系统观非常强的机器人世界模型论文。它把 next-scale autoregression、动作尺度对齐、有界时标记忆、dream forcing、视觉动态 RL 和动作专家组织在一条因果链上；更重要的是，它用 600 对 sim-real 闭环 rollout 证明世界模型不仅能“生成”，还能在一定程度上用于**策略校准与 checkpoint 排序**。

我认为它最值得记住的不是某个单独模块，而是一个研究范式转变：

> **机器人世界模型的终极指标，不是未来视频有多漂亮，而是它对动作后果的判断，能否在真实世界里支撑正确决策。**

当前版本的硬伤同样明确：代码和权重未开源，关键超参、模型规模、memory budget、算力和私有数据规模全部缺失；接触/插入阶段仍有系统性乐观偏差。它已经展示了可用方向，但还没有成为可独立复现、可安全替代真机的通用模拟器。

---

## 11. 关键术语

| 术语 | 含义 |
|---|---|
| Next-scale autoregression | 在同一未来观察内部，从粗尺度到细尺度逐级预测残差视觉代码 |
| Scale-aligned action | 给不同视觉尺度提供匹配粒度的动作条件，而不是一个全局动作向量 |
| Time–scale memory | 最近历史保留细尺度 KV，远期历史保留粗尺度 KV 的固定预算记忆 |
| Identity anchor | 由初始观察产生、永不老化的身份与全局布局锚点 |
| Dream forcing | 用腐化历史与 detached 自生成后缀训练模型继续预测干净未来 |
| On-policy visual alignment | 把视觉生成器而非机器人控制器当作策略，用自身新鲜 rollout 的奖励更新 |
| Policy checkpoint ranking | 比较世界模型能否复现不同机器人策略 checkpoint 的真实优劣顺序 |
| Selection regret | 模拟器选出的 checkpoint 相对真机最优 checkpoint 损失了多少真实成功率 |

---

## 12. 图表索引与材料

| 图 | 内容 | 来源 |
|---|---|---|
| 图 1 | 系统总览与闭环 rollout | Figure 1，PDF p.2 |
| 图 2 | next-scale、动作条件、time–scale memory、RL 完整框架 | Figure 3，PDF p.7 |
| 图 3 | 同初始状态下的反事实动作未来 | Figure 7，PDF p.16 |
| 图 4 | 长程记忆与 dream forcing 消融 | Figure 9，PDF p.18 |
| 图 5 | sim-real 成功率校准与 checkpoint 排序 | Figure 12，PDF p.20 |
| 图 6 | episode confusion matrix、阶段偏差与汇总表 | Figure 13 + Table 3，PDF p.21 |
| 图 7 | 真实机器人 Task Progress | Figure 14，PDF p.22 |

- 完整来源、官方材料、仓库 commit 与图片裁剪说明： [sources.md](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/sources.md)
- 本地官方 PDF： [paper.pdf](assets/WALL-SS%20-%20下一尺度自回归的长程机器人世界模型/sources/paper.pdf)
