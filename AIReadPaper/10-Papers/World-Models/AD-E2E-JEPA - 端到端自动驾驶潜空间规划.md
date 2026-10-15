---
type: paper-note
status: complete
domain: World-Models
paper: "AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving"
year: 2026
arxiv: "2609.34085"
doi: null
source: "https://arxiv.org/abs/2609.34085"
project: null
code: "https://github.com/HaoranZhuExplorer/AD-E2E-JEPA"
tags:
  - World-Models
  - JEPA
  - Autonomous-Driving
  - Latent-Planning
  - SIGReg
  - NAVSIMv2
created: 2026-10-15
updated: 2026-10-15
---

# AD-E2E-JEPA：端到端自动驾驶潜空间规划

## 基本信息

- **作者：** Haoran Zhu、Wancong Zhang、Yann LeCun、Anna Choromanska
- **版本：** arXiv v1，2026-09-28；本笔记以官方 PDF、附录和官方 TeX 源包为主
- **任务：** 在不训练驾驶 policy 的前提下，用动作条件 JEPA 世界模型从候选轨迹库中选择最接近目标未来观测的轨迹
- **数据与评测：** NAVSIM 训练数据、NAVSIMv2 `navtest-stage-1` 指标；前视单相机，2 秒历史、4 秒未来
- **重要性评估：** ★★★★☆（4/5）。价值在于把 dense patch world model 的规划代价压到可评测规模，并系统暴露了“潜空间几何准确”与“驾驶安全”并非同一目标

## 一句话结论

AD-E2E-JEPA 在冻结 DINOv3 ViT-L 后增加一个由 SIGReg 约束的可学习 patch projector，把每帧 token 从 $16\times32\times1024$ 压到 $4\times8\times256$，再以相对位姿轨迹为动作滚动预测未来潜表示；它用 256 条候选轨迹时在单张 A100 上约 0.8 秒完成 8 帧 rollout，比 dense-patch 的 DINO-WM/JEPA-WM 快约两个数量级，同时在 NAVSIMv2 保持可比规划质量。不过其 goal 是数据中真实未来图像，属于 oracle 诊断设定，且规划代价没有显式安全约束，不能直接等同于可部署的闭环自动驾驶系统。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

端到端自动驾驶通常把传感器输入直接回归为未来轨迹。这样做部署快，但训练本质上是模仿人类行为：模型学到“看到这个画面通常怎么开”，而不是显式学习“执行某条候选轨迹后世界会变成什么”。动作条件 world model 则提供另一条路线：先根据当前观测和候选动作预测未来状态，再比较不同未来与目标的距离。它把行为选择变成“想象—评分—选择”，理论上能在不针对每个 goal 训练 policy 的情况下重用同一个动力学模型。

论文没有直接训练新 policy，而是刻意隔离世界模型质量。对每个测试场景，它把 $F$ 帧后的**真实前视图像**当作 goal，让 LeWM、DINO-WM、JEPA-WM 和 AD-E2E-JEPA 对同一组轨迹候选做 rollout，再选终点 latent 最接近 goal latent 的轨迹。这样可以回答：表征与动力学是否足以识别真实未来对应的行为？但它没有回答实际车辆如何从导航指令生成目标图像，这一边界必须贯穿全文理解。

![](assets/AD-E2E-JEPA%20-%20端到端自动驾驶潜空间规划/figures/01-overview.png)

> **图 1：AD-E2E-JEPA 的任务与结果总览。** 中间展示由历史图像出发、通过轨迹词表 rollout 并匹配真实未来 goal 的流程；周围同时汇总 EPDMS、规划时间、终点误差、命中率与下游迁移收益。来源：论文 Figure 1，第 1 页，官方 arXiv TeX 矢量源图。

### 1.2 现有 JEPA world model 的速度—质量矛盾

LeWM 使用单个全局 CLS latent，token 少、规划快，但全局压缩容易丢失道路边界、局部车道关系和远处几何，在本文 NAVSIM 适配中准确性不足。DINO-WM 冻结预训练视觉 encoder，并预测密集 patch token；JEPA-WM 又系统引入 DINOv3、AdaLN、RoPE 与 rollout training，局部空间信息更强，但每条候选轨迹、每个未来时刻都要处理密集 token。CEM 式反复搜索在驾驶场景尤其昂贵，即使本文改为固定候选库，256 条轨迹仍分别需要 91.8 秒和 101.0 秒。

本文的核心问题不是再换一个更大的 predictor，而是：**能否学习一个对规划友好的压缩坐标系，使 patch 数和通道数大幅下降，但终点 latent 距离仍保留足够的几何与行为区分能力？** 这也是 SIGReg projector 的真正角色。

### 1.3 核心贡献

1. **将 goal-conditioned zero-shot latent planning 系统化引入 E2EAD。** 除 EPDMS 外，同时报告时间、终点位姿误差和 Top-k hit rate，把“开得像不像”“几何上准不准”“模型是否可靠地给 GT 排高分”拆开。
2. **提出可训练的压缩 patch projector。** 两层 stride-$2\times2$ Conv2D 同时将 patch 数缩小 16 倍、embedding 维度缩小 4 倍，使 dense spatial representation 不必退化为单 CLS。
3. **用 stop-gradient 与 SIGReg 防止 projector 坍塌。** 冻结 DINOv3 只能保证原始 target 不坍塌；一旦共享的 projector 可训练，它仍可能把不同 patch 映成常数，因此需要新的 anti-collapse 约束。
4. **以轨迹词表替代连续 CEM 搜索。** 从 8192 条聚类驾驶轨迹中按角度均匀抽样，批量 rollout 后直接 argmin，得到清晰的质量—时间可调旋钮。
5. **证明 projector 具有迁移价值。** 世界模型自监督预训练的 projector 接到简单模仿学习解码器后，将 EPDMS 从 80.2 提高到 85.4。

---

## 2. 任务定义与输入输出

### 2.1 历史、动作、预测与最终输出

每个时刻的前视图像记为 $I_j\in\mathbb{R}^{3\times H\times W}$，ego pose 为 $P_j=[x_j,y_j,\theta_j]^\top$，所有 pose 都转换到当前时刻 $t$ 的车体坐标系，使 $P_t=[0,0,0]^\top$。NAVSIM 以 2 Hz 采样；模型读取 $K+1=4$ 帧，即 2 秒历史，并规划 $F=8$ 帧，即 4 秒未来。

训练时的“动作”不是油门和方向盘，而是连续位姿之间的相对变化：

$$
a_t=\operatorname{Relative}(P_t,P_{t+1})=[\Delta x_{t\to t+1},\Delta y_{t\to t+1},\Delta\theta_{t\to t+1}]^\top.
$$

这一区分很重要。模型学习的是**轨迹级运动条件下的视觉 latent 变化**，而不是车辆底层控制器、轮胎动力学或 actuator delay。测试时输入为历史图像、过去 pose/action、一个真实未来 goal 图像和候选轨迹库；世界模型输出每条候选的未来 patch latents；最终系统输出候选库中的一条未来 ego-pose 轨迹。

### 2.2 基线 JEPA-WM 的表示预测

冻结的 DINOv3 ViT-L 独立编码历史与下一帧：

$$
s_{t-K:t+1}=\operatorname{Enc}(I_{t-K:t+1}).
$$

动作经线性层 $E_a$ 后注入带 AdaLN 与 RoPE 的 predictor：

$$
\hat s_{t-K+1:t+1}=\operatorname{Pred}(s_{t-K:t},E_a(a_{t-K:t})),
$$

并用 MSE 拟合冻结 encoder 的下一时刻 dense patch target：

$$
\mathcal L_{\mathrm{pred}}=\operatorname{MSE}(\hat s_{t-K+1:t+1},s_{t-K+1:t+1}).
$$

DINO-WM/JEPA-WM 不需要另加 encoder anti-collapse loss，因为 target encoder 固定；但这并不意味着任何接在其后的可训练压缩层都自动安全。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/AD-E2E-JEPA%20-%20端到端自动驾驶潜空间规划/figures/02-method-pipeline.png)

> **图 2：完整 architecture。** 左侧是冻结 DINOv3、共享 projector、动作条件 patch predictor、MSE 与 SIGReg；左下明确给出 $16\times32\times1024\rightarrow4\times8\times256$；右上用 8192 轨迹词表做 rollout 与 goal matching；右下将 pretrained projector 迁移到模仿学习。来源：论文 Figure 2，第 4 页，官方 arXiv TeX 矢量源图。

数据流可概括为：历史/未来图像先经冻结 DINOv3 得到 dense patches；同一个 projector 处理 context 与 target；相对位姿动作经线性层注入 predictor；predictor 预测下一时刻 projected patches；训练同时最小化潜预测 MSE 和 projected distribution 的 SIGReg；规划时固定历史与 goal，批量展开每条候选轨迹，选终点潜距离最小者。

### 3.2 patch 与维度压缩为什么能快 100 倍

DINOv3 ViT-L 每帧产生 $16\times32=512$ 个 patch，每个 1024 维。projector 由两层 stride-$2\times2$ Conv2D 组成，空间分辨率连续减半两次，最终为 $4\times8=32$ 个 patch；通道降为 256。因此：

- token 数 $512\rightarrow32$，减少 **16 倍**；
- token 维度 $1024\rightarrow256$，减少 **4 倍**；
- 单帧标量表示量减少 $16\times4=64$ 倍；
- predictor 的注意力与 MLP 都受 token 数和通道影响，实测 256 候选、8 帧 rollout 从约百秒降到 0.8 秒，达到约 100 倍级速度提升。

这里不是简单平均池化。两层卷积可学习哪些局部视觉结构应在下采样时保留；共享权重保证 context 和 target 落在同一坐标系。代价是压缩层本身可以作弊：把所有输入映射为相近向量，预测 MSE 会迅速变小，所以必须显式约束表示分布。

### 3.3 stop-gradient 与 patch-wise SIGReg

投影后表示为

$$
z_{t-K:t+1}=\operatorname{Proj}(s_{t-K:t+1}),
$$

predictor 在压缩空间工作：

$$
\hat z_{t-K+1:t+1}=\operatorname{Pred}'(z_{t-K:t},E_a(a_{t-K:t})).
$$

目标分支停止梯度：

$$
\mathcal L^{\mathrm{proj}}_{\mathrm{pred}}=
\operatorname{MSE}\!\left(\hat z_{t-K+1:t+1},\operatorname{sg}(z_{t-K+1:t+1})\right).
$$

但 stop-gradient 主要降低两分支共同追逐平凡解的风险，并不从分布上排除低秩或聚类表示。SIGReg 进一步要求 projected embeddings 接近各向同性高斯。论文不是像 LeWM 那样对每个时刻的全局 CLS 做正则，而是对**每个时间步、每个 patch 位置，跨 batch 独立收集 $D$ 维样本**；沿 $M=1024$ 个随机单位方向 $u^{(m)}$ 投影到一维，再用 Epps–Pulley 正态性检验 $\mathcal T$，最后对时间、空间和方向平均：

$$
\mathcal L_{\mathrm{SIGReg}}=
\frac{1}{NH'W'M}\sum_{l=1}^{NH'W'}\sum_{m=1}^{M}
\mathcal T\!\left(\{\langle z_{l,b},u^{(m)}\rangle\}_{b=1}^{B}\right).
$$

由 Cramér–Wold 定理，若所有一维投影分布都匹配，则联合分布匹配；有限随机方向给出可计算近似。附录使用默认 1024 个方向，Epps–Pulley 积分以区间 $[0,3]$ 上 17 个 knots 近似。总损失为

$$
\mathcal L=\mathcal L^{\mathrm{proj}}_{\mathrm{pred}}+\lambda\mathcal L_{\mathrm{SIGReg}},
$$

默认 $\lambda=0.09$。直觉上，MSE 要求“未来可预测”，SIGReg 要求“不能靠丢掉全部差异来变得可预测”；两者共同逼 projector 保存对动力学有用、又具有足够信息量的 patch 几何。

### 3.4 rollout training

单步版本只监督第一个未来帧。rollout 版本覆盖完整 $F=8$ 帧：一部分是 teacher forcing，在真实 context 上预测下一帧；另一部分把前一步预测 append 回滑动窗口，计算 $k=2,\dots,F$ 的终点一致性损失。实现使用 truncated BPTT，在每次后续 predictor 调用前停止通过旧 autoregressive context 的梯度，只让当前调用可微。总体写成

$$
\mathcal L_{\mathrm{multi}}=
\mathcal L_{\mathrm{TF}}+\frac{\mathcal L_2+\cdots+\mathcal L_F}{F}
+\lambda\mathcal L^{t-K:t+F}_{\mathrm{SIGReg}}.
$$

它直接针对规划时的自回归分布偏移，实验中显著改善 FDE 和 hit rate，但训练时间也从 20 小时增加到 1 天 22 小时（navtrain，单张 A100）。

### 3.5 轨迹候选库与规划公式

本文不用 CEM 在连续动作空间迭代采样，而采用已有的 8192 条聚类驾驶轨迹词表 $\mathcal V$。为了得到 256 条轻量候选，先按轨迹的角坐标排序，再等间隔抽样，使直行、左右转向的覆盖较均匀。每条 pose trajectory 转换成相对位姿 action sequence，驱动同一个 predictor 自回归 $F$ 步。

目标是 $F$ 帧后的真实图像 $I_{t+F}$ 经 encoder 与 projector 得到的 $z_{t+F}$。第 $i$ 条候选的代价为

$$
C^i=\left\|z_{t+F}-\hat z^i_{t+F}\right\|_2^2,
$$

最终选择

$$
i^*=\arg\min_{i\in\{1,\dots,|\mathcal V_{\mathrm{sampled}}|\}}C^i,
\qquad P^*_{t:t+F}=P^{i^*}_{t:t+F}.
$$

这是一种检索式规划：模型不生成任意轨迹，只负责在先验词表中排序。因此稳定、可并行，但上限受 vocabulary 覆盖度限制；若安全可行轨迹不在库中，再好的 latent distance 也无法选出来。

---

## 4. 数据集与实验设置

### 4.1 NAVSIM / NAVSIMv2 设置

公平对比使用 `navtrain`：10 小时驾驶视频，2 Hz 采样。扩展实验使用 `trainval` 训练部分的 70 小时。所有模型输入 4 帧历史、预测 8 帧未来；DINO-WM、JEPA-WM 与 AD-E2E-JEPA 使用冻结 DINOv3 ViT-L，LeWM 则用从头训练的 ViT-L。所有配置训练 30 epochs，AdamW，一轮 warmup 后 cosine annealing。

对比昂贵的 dense-patch 模型只在 NAVSIM test 中固定抽取的 100 个场景运行；高效模型还在完整 12,146 个测试场景上评测。100 场景集合缺少时间相邻样本，因此 EPDMS/EPDMS† 不计 extended comfort（EC）；主表小样本差异应结合 full-set 结果判断。

### 4.2 训练与硬件成本

| 配置 | GPU / batch | $\lambda$ | 训练时间 |
|---|---:|---:|---:|
| LeWM，navtrain | 4×A100 / 8 | 0.09 | 1 天 |
| DINO-WM，navtrain | 4×A100 / 64 | — | 11 小时 |
| JEPA-WM，navtrain | 4×A100 / 64 | — | 13 小时 |
| AD-E2E-JEPA，navtrain | 1×A100 / 128 | 0.09 | 20 小时 |
| AD-E2E-JEPA + rollout，navtrain | 1×A100 / 128 | 0.09 | 1 天 22 小时 |
| AD-E2E-JEPA，trainval | 4×A100 / 512 | 0.025 | 2 天 5 小时 |
| AD-E2E-JEPA + rollout，trainval | 4×A100 / 256 | 0.09 | 4 天 2 小时 |

论文只给 A100 级配置，没有消费 GPU 显存/吞吐测试。DINOv3 ViT-L、NAVSIM 数据预处理和 NAVSIMv2 pseudo-simulation 评测栈也是复现依赖，不应把“projector 轻量”误读成全流程低成本。

### 4.3 指标与安全口径

EPDMS 将安全项作为乘法门控，再乘驾驶质量加权平均：

$$
\operatorname{EPDMS}_i=
\operatorname{NC}_i\operatorname{DAC}_i\operatorname{DDC}_i\operatorname{TLC}_i
\frac{5\operatorname{EP}_i+5\operatorname{TTC}_i+2\operatorname{LK}_i+2\operatorname{HC}_i+2b_i\operatorname{EC}_i}{14+2b_i}.
$$

NC、DAC、DDC、TLC 分别表示无责任碰撞、可行驶区域、行驶方向和红绿灯合规；EP、TTC、LK、HC、EC 衡量进度、碰撞时间、车道保持与舒适性。$b_i$ 表示该样本是否能计算 EC。论文另报 $\operatorname{EPDMS}^{\dagger}$，即去掉四个安全乘法项，只保留加权质量部分。后者用于观察 latent goal matching 的“纯驾驶质量”，但不是安全分数；两者差距大，恰好说明方法没有直接优化安全。

其余指标包括终点位置误差 FDE、纵向/横向误差 $\Delta x/\Delta y$、航向误差 $\Delta\theta$，以及把 GT trajectory 额外放入候选集合后，其 latent cost 是否进入 Top-1/Top-5 的 hit rate。

---

## 5. 实验结果

### 5.1 100 场景公平比较

| 方法（navtrain） | EPDMS↑ | EPDMS†↑ | 时间/场景↓ | FDE↓ | $\Delta\theta$↓ | Hit@1/5↑ |
|---|---:|---:|---:|---:|---:|---:|
| LeWM | 48.3 | 73.9 | **0.7 s** | 12.4 m | 13.6° | 6% / 18% |
| DINO-WM | 68.3 | 91.4 | 91.8 s | **3.9 m** | 5.9° | 40% / 73% |
| JEPA-WM | 74.2 | 90.9 | 101.0 s | 4.0 m | 5.2° | **45% / 75%** |
| AD-E2E-JEPA | **76.6** | **92.4** | 0.8 s | 4.2 m | 4.7° | 27% / 59% |
| AD-E2E-JEPA + rollout | 70.4 | 92.2 | 0.8 s | 3.5 m | **4.1°** | 34% / 67% |

AD-E2E-JEPA 的主张因此成立在一个很具体的意义上：它保留 dense patches 的局部规划质量，却接近 CLS 模型的速度。普通版本在 100 场景 EPDMS 最高，但 hit rate 不及 JEPA-WM；rollout 版本几何更准，EPDMS 反而下降，显示 latent 终点距离、GT 排名和安全加权驾驶分数并不完全同向。作者也承认 100 场景方差可能放大差异。

![](assets/AD-E2E-JEPA%20-%20端到端自动驾驶潜空间规划/figures/03-planning-comparison.png)

> **图 3：四种 JEPA world model 的定性轨迹。** LeWM 选出的轨迹明显平移；DINO-WM、JEPA-WM 与 AD-E2E-JEPA 更接近绿色 GT。该图支持“CLS 过度压缩、compressed patches 仍保留几何”的解释，但单个案例不能替代全量指标。来源：论文 Figure 3，第 8 页，官方 arXiv TeX 矢量源图。

### 5.2 完整 12,146 场景与候选规模消融

trainval + rollout、256 候选在 full set 达到 67.3 EPDMS、84.1 EPDMS†、4.0 m FDE、3.5°航向误差和 53.8%/82.7% hit rate。将候选数增加到 8192 后，EPDMS 提高到 72.9，EPDMS† 到 86.5，FDE 降到 2.8 m，航向误差降到 2.0°；代价是时间从 0.8 秒增加到 18.2 秒。

| 候选数 | 时间/场景 | EPDMS | EPDMS† | FDE | Hit@1/5 |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.8 s | 67.3 | 84.1 | 4.0 m | 53.8% / 82.7% |
| 512 | 1.4 s | 69.2 | 85.0 | 3.6 m | 45.2% / 73.9% |
| 1024 | 2.5 s | 70.5 | 85.5 | 3.2 m | 36.3% / 64.3% |
| 2048 | 4.7 s | 71.5 | 86.0 | 3.0 m | 27.4% / 53.2% |
| 4096 | 9.3 s | 72.1 | 86.3 | 2.9 m | 20.7% / 43.3% |
| 8192 | 18.2 s | **72.9** | **86.5** | **2.8 m** | 15.3% / 33.8% |

hit rate 随候选增多反而下降并不与几何改善矛盾：GT trajectory 要在更大的集合中进入前 1/5，排名标准更严格；与此同时，词表更密，使 argmin 更可能找到接近 GT 的非 GT 轨迹。这个指标不能跨候选规模直接当作同一难度比较。

### 5.3 rollout、数据规模与迁移消融

在 full set，navtrain 普通版到 rollout 版的 FDE 从 6.3 m 降到 4.5 m，Hit@1/5 从 32.9%/65.3% 提升到 42.6%/71.0%；trainval 上 rollout 更明显，把 EPDMS 从 63.2 提到 67.3、FDE 从 6.2 m 降到 4.0 m、Hit@1/5 从 31.0%/64.8% 提到 53.8%/82.7%。这支持“训练时暴露于自身预测”是多步规划可靠性的关键。

下游迁移则固定简单 ViT 驾驶架构，只替换 projector 初始化：随机 projector 得到 80.2 EPDMS，world-model pretrained projector 得到 85.4。说明压缩层不是只为计算量服务，它确实吸收了未来可预测结构。但论文没有单独报告“相同 stride projector 去掉 SIGReg”“只降通道不降 patch”“固定池化替代可学习卷积”等隔离消融，因此 5.2 点增益和规划保持率究竟各有多少来自 SIGReg、卷积结构或训练数据，尚不能严格分解。

![](assets/AD-E2E-JEPA%20-%20端到端自动驾驶潜空间规划/figures/04-imitation-transfer.png)

> **图 4：projector 的下游迁移。** 保留 DINOv3 与自监督 projector，丢弃 world-model predictor；历史图像 token 与驾驶命令作为 K/V，未来轨迹 query 经 cross-attention 和 MLP 回归 GT 轨迹。来源：论文 Figure 4，第 16 页，官方 arXiv TeX 矢量源图。

### 5.4 “100×速度”应怎样理解

0.8 秒口径是：**单张 NVIDIA A100 80 GB、每场景 256 条候选全部在 GPU 并行、每条 rollout 8 个未来帧的平均 planning time**。比较对象是在相同候选式规划下处理 dense patches 的 DINO-WM 91.8 秒和 JEPA-WM 101.0 秒，因此“约 100×”成立。它不是摄像头到执行器的整车闭环延迟，也没有计入目标图像从何而来、轨迹库构建、车辆控制执行和真实传感器管线。候选扩到 8192 时已经是 18.2 秒，因而本文解决的是大规模 benchmark 与离线/低频规划可行性，还不是实时驾驶控制频率。

### 5.5 安全细项与结果应怎样解读

完整测试集能看出 EPDMS 总分背后的结构。trainval + rollout、256 候选时，NC/DAC/DDC/TLC 分别为 95.2/82.0/94.4/99.6，说明红绿灯与碰撞项较高，主要短板是可行驶区域合规；驾驶质量部分 EP/TTC/LK/HC/EC 为 85.2/93.3/89.3/92.3/35.5，其中 extended comfort 明显最低。候选增至 8192 后，四个安全项提高到 96.9/86.0/96.7/99.7，LK、HC、EC 也提高到 91.8/95.1/43.8，于是 EPDMS 从 67.3 升到 72.9。密集 vocabulary 确实更容易找到兼顾路线几何与评测规则的轨迹，而不仅是把终点位置拉近。

但这些数字不能反向证明模型“学会了安全规则”。首先，代价函数里仍只有终点 latent L2；安全改善可能来自候选库中的轨迹本来就由人类驾驶数据聚类而成，增加候选相当于提高安全先验的覆盖密度。其次，乘法指标具有门控效应：任一场景的 NC、DAC、DDC 或 TLC 较低，都会把质量项整体压低；因此 EPDMS 与 EPDMS† 的差值不能简单解释成“事故率”。再次，pseudo-simulation 依赖日志与代理模型，并不包含所有真实交互反馈。最合理的结论是：压缩表示没有破坏 NAVSIMv2 可测的安全相关排序能力，但系统仍需要显式约束和闭环实车验证。

一个有意义的后续消融，是固定同一候选集合与同一 world model，分别用纯 latent distance、latent distance 加安全过滤、以及可学习 safety scorer 排序。这样才能区分表示是否包含风险信息、词表是否自带安全先验、以及显式安全目标能贡献多少。论文当前给出了完善的事后分解指标，却没有完成这一因果拆分。

---

## 6. 与 LeWM、DINO-WM、JEPA-WM 的关系

| 方法 | 视觉坐标系 | 规划表示 | anti-collapse | 优势 | 在本文中的主要短板 |
|---|---|---|---|---|---|
| LeWM | 从头训练 ViT-L | 全局 CLS | SIGReg | token 极少，0.7 s | 局部道路几何丢失，FDE 12.4 m |
| DINO-WM | 冻结 DINOv3 ViT-L（本文适配） | dense patches | 冻结 target | 几何准确、目标固定稳定 | 91.8 s，规划成本高 |
| JEPA-WM | 冻结 DINOv3 + AdaLN/RoPE | dense patches，可 rollout | 冻结 target | hit rate 与规划准确性强 | 101.0 s，full-set 不可承受 |
| AD-E2E-JEPA | 冻结 DINOv3 + 可训练 projector | 32 个、256 维 compressed patches | stop-gradient + patch-wise SIGReg | 接近 dense-patch 质量、接近 CLS 速度 | goal 与安全设定仍理想化 |

AD-E2E-JEPA 可以理解为 DINO-WM/JEPA-WM 与 LeWM 之间的中点：保留冻结 foundation encoder 的稳定语义与 patch 空间结构，但把“规划真正需要多少 token”交给自监督 projector 学习；又借用 LeWM/LeJEPA 的 SIGReg 思想，避免可训练压缩头坍塌。相对 JEPA-WM，它主要创新不是动力学 predictor，而是**规划表示的带正则压缩**和**轨迹词表检索**。

需要注意，本文的 DINO-WM 已换成 DINOv3 ViT-L 以统一比较，不完全等同于原始 DINO-WM 论文的 DINOv2 配置；JEPA-WM 则是论文 *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?* 的最佳设计适配到驾驶。因此这张表比较的是本文控制下的 E2EAD 版本，不是跨论文原始 benchmark 的绝对排名。

从技术谱系看，四者解决的是不同层级的问题。DINO-WM 证明冻结的空间化视觉特征足以支撑动作条件规划；JEPA-WM 追问 backbone、动作注入、位置编码和多步训练怎样影响物理规划；LeWM 追问能否从像素端到端稳定学习紧凑 latent；AD-E2E-JEPA 则接受冻结 foundation encoder，把注意力集中到“怎样把 dense patches 压成便于大规模候选搜索的规划状态”。因此它并没有取代前三者，而是给出了面向驾驶吞吐的系统折中。若未来计算预算允许，仍可以把本文 projector 接到更强的 rollout predictor、层级 planner 或带不确定性的 ensemble 上；若部署预算更紧，则可继续探索自适应 token pruning，而不是固定 $4\times8$ 网格。

这种定位也解释了为什么本文最该被复现的不是单个 EPDMS 数字，而是完整 Pareto 前沿：同一 encoder 和候选库下，逐步改变 patch 数、维度、rollout 深度与候选规模，同时记录几何误差、安全子项、命中率、显存和时延。只有这样，才能判断 32 个 token 是否是普适甜点，还是 NAVSIM 分辨率和四秒 horizon 下的特定最优点。

---

## 7. 局限与批判性评价

### 7.1 ground-truth future observation 是 oracle goal

最大限制是 $I_{t+F}$ 来自数据集中真实未来。部署车辆在当前时刻不可能看到四秒后的真实相机帧；导航系统通常只给 route、地图 waypoint、语言目的地或稀疏目标状态。因此实验绕过了最困难的接口之一：如何把高层意图转成与 encoder 同域、视觉上可实现、又不泄露未来的 goal。该设定很适合诊断 world model 是否能从候选中认出 GT，但“zero-shot”准确说是**对给定 oracle image goal 不训练 policy 的候选排序**，不是对未知城市、未知动力学或真实导航目标的零样本驾驶。

### 7.2 安全并未进入规划目标

规划只最小化终点 latent L2，没有碰撞、道路边界、红灯、舒适性、不确定性或轨迹可执行性项。EPDMS 里的 NC/DAC/DDC/TLC 是事后评测，不是优化目标。full-set 256 候选的 67.3 EPDMS 与 84.1 EPDMS† 相差 16.8 分，正说明几何接近 goal 不自动保证安全。8192 候选虽提高到 72.9/86.5，也不能证明 planner 在冲突场景具备风险意识。更现实的版本需要 safety cost、可行性过滤、碰撞预测与不确定性惩罚共同参与排序。

### 7.3 候选词表把大量先验移到模型外

8192 条轨迹来自聚类驾驶数据，天然携带人类驾驶分布和道路几何先验。world model 没有从连续控制中创造新动作，只在词表覆盖范围内检索。因此性能来自三者组合：trajectory vocabulary 的覆盖、latent model 的排序质量、NAVSIM pseudo-simulation 的指标。若道路曲率、紧急避障或车辆动力学超出词表，增加候选数也无济于事。

### 7.4 评测与建模边界

实验是单前视相机、4 秒 horizon、数据驱动 pseudo-simulation，而不是多传感器真实车闭环。确定性 MSE predictor 无法显式表达遮挡、他车意图和多模态未来，也没有 ensemble 或 epistemic uncertainty 防止 planner 利用模型误差。历史 pose 和候选动作均为精确相对位姿，真实定位噪声、控制跟踪误差和时间延迟没有进入模型。论文也缺少跨城市、天气、相机标定和长时 route 的独立泛化实验。

### 7.5 证据仍缺一组关键消融

论文展示了压缩后仍然有效，也展示了 pretrained projector 可迁移，但没有完整的 projector 设计矩阵：无 SIGReg、仅 stop-gradient、平均池化、不同 stride、不同 token/通道预算、CLS+patch 混合等。因而“SIGReg 是保持规划性能的必要原因”目前是机制合理、结果一致，却不是被严格隔离验证的因果结论。后续工作应画出 token 数—维度—速度—FDE—EPDMS 的 Pareto 曲线。

---

## 8. 复现成本与实践建议

截至 2026-10-15，官方 GitHub 仓库已经建立，但 README 明确表示源代码仍在整理、预计 2026-10-19 前公开；页面当前只有 README 与图片，尚无训练脚本、依赖锁定、配置和 checkpoint。因此论文级数字暂时不能完整复现，尤其是 DINOv3 feature extraction、NAVSIM 数据切片、轨迹 vocabulary、rollout batching 与 NAVSIMv2 评测版本都可能影响结果。

待代码公开后，建议分四级验收：第一，固定 encoder/projector，核对张量确实为 $16\times32\times1024\rightarrow4\times8\times256$；第二，只跑 one-step prediction，监控 patch 方差、协方差谱和 SIGReg，确认无 rank/cluster collapse；第三，在 100 场景上复现 256 候选的 0.8 秒、FDE 与 hit rate，并明确计时是否含编码；第四，再训练 rollout 版本并扩到 full set。不要一开始追 8192 候选，因为它把单场景时间提高到 18.2 秒，也更难定位误差来自模型还是 vocabulary。

最低可行复现仍需 NAVSIM/NAVSIMv2 数据与评测栈、DINOv3 ViT-L 权重和至少一张大显存 GPU。论文最轻训练为单张 A100 20 小时；完整 trainval + rollout 为 4×A100、4 天 2 小时。若目标只是验证 projector 思想，可先冻结预计算 DINOv3 features，在小子集上训练 projector/predictor；但这只能验证表示与速度，不能宣称复现完整 EPDMS。

---

## 9. 个人启发与后续问题

第一，world model 的效率瓶颈可能首先是**规划表示**，而非 predictor 参数量。保留空间 patch 不等于必须保留全部 patch；学习一个受信息约束的 task-agnostic bottleneck，比直接换 CLS 更有希望得到速度—几何折中。

第二，评价 latent planner 应至少分成四轴：安全加权任务分数、几何误差、GT 排名可靠性和墙钟时间。本文最有价值的不只是某个 EPDMS，而是显示这些轴会互相分离：rollout 能降 FDE、升 hit rate，却不保证小样本 EPDMS 同步上升。

第三，下一步应把 oracle image goal 替换为可部署目标接口，例如由地图 route/waypoint 生成 goal embedding，或学习跨模态 goal encoder；同时把 collision、drivable area 与 uncertainty 加入 cost。只有当模型不再依赖真实未来图像，并在闭环中处理观测更新和控制误差时，AD-E2E-JEPA 才从“世界模型诊断器”走向真正的自动驾驶 planner。

第四，SIGReg 的位置值得推广。它不是只能约束最终全局 latent，也可以约束可训练压缩头的每个时空位置。这一设计可迁移到机器人操作、视频规划和 VLA：冻结大 encoder，训练小型 structured projector 与 dynamics，再用分布正则守住信息量，可能是比全模型微调更经济的路线。

---

## 10. 材料来源

本地官方 PDF、arXiv TeX 源包、四张矢量原图渲染与逐项溯源见：

- [sources.md](assets/AD-E2E-JEPA%20-%20端到端自动驾驶潜空间规划/sources.md)
- [官方 PDF](assets/AD-E2E-JEPA%20-%20端到端自动驾驶潜空间规划/sources/paper.pdf)
- [官方 arXiv TeX 源包](assets/AD-E2E-JEPA%20-%20端到端自动驾驶潜空间规划/sources/arxiv-source.tar.gz)
- [arXiv](https://arxiv.org/abs/2609.34085)
- [官方代码仓库](https://github.com/HaoranZhuExplorer/AD-E2E-JEPA)
