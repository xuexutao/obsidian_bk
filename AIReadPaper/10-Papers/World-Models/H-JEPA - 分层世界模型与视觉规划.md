---
type: paper-note
status: complete
domain: World-Models
paper: "H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning"
year: 2026
arxiv: "2610.06805v2"
doi: "10.48550/arXiv.2610.06805"
source: "https://arxiv.org/abs/2610.06805v2"
project: "https://h-jepa.com/"
code: "https://github.com/kevinghst/H-JEPA"
code_commit: "97054ecaff83e13839195ec4b92d74c08a0127c5"
tags:
  - World-Models
  - JEPA
  - Hierarchical-Planning
  - Visual-Planning
  - Robot-Learning
created: 2026-11-05
updated: 2026-11-05
---

# H-JEPA：端到端分层世界模型与视觉规划

## 基本信息

- **论文**：H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning
- **作者**：Wancong Zhang、Basile Terver、Michael Rabbat、Yann LeCun、Randall Balestriero
- **版本**：arXiv:2610.06805v2，2026-11-01 修订
- **任务**：从无奖励的离线视觉—动作轨迹中端到端学习多时间尺度、每层独立表征的 action-conditioned JEPA，并用层级潜变量做目标图像条件规划
- **评测**：FourRoom Distractors、Visual AntMaze、Push-T、OGBench Cube，以及 DROID 真实机器人离线视频
- **代码状态**：官方代码、数据与权重均公开；本文代码分析固定在 commit `97054ecaff83e13839195ec4b92d74c08a0127c5`
- **材料**：[官方 PDF](assets/H-JEPA%20-%20分层世界模型与视觉规划/sources/paper.pdf)；[完整来源清单](assets/H-JEPA%20-%20分层世界模型与视觉规划/sources.md)

## 一句话结论

H-JEPA 把 LeCun 2022 年 H-JEPA 研究纲领中“越高层越抽象、预测越远、规划自顶向下”的设想变成了可端到端训练和可消融验证的视觉世界模型：三层模型在 Visual AntMaze 上把 LeWM 的 $18.0\pm3.5\%$ 提高到 $73.3\pm3.5\%$，证据表明收益同时来自时间分解与高层目标表征；但选择性抽象依赖数据中的时间频率分离，更深层级会遭遇有效 clip 锐减，DROID 结果也只是离线开放环路径 fidelity，而不是真机闭环成功率。

---

## 1. 研究定位：从 2022 年架构纲领到端到端实证

### 1.1 单层 JEPA 世界模型为何不够

**论文事实。** LeWM 一类 task-agnostic JEPA 世界模型不重建未来像素，而是把图像编码为潜在状态，再根据动作预测下一潜在状态。它已经证明：从像素端到端训练 encoder 与 predictor、在 latent space 中用目标距离规划，是可以工作的。然而单层模型让同一个潜在空间承担两种冲突职责：低层控制需要保留腿部姿态、接触和局部几何等快速细节；长程规划却更需要“我在迷宫哪里、离目标还有多远”这类慢变量。细节保留得越全，长程 rollout 越长、误差累积越严重，目标距离也越容易被与任务无关的姿态差异支配。

**作者解释。** 一个合理的层级应让低层在短时间尺度上预测细节，高层跨更长 stride 预测慢变量；高层若无法长期预测某些快速细节，联合优化会促使 encoder 把这些细节抽象掉。这样，层级不仅缩短每个 planner 的时间跨度，还改变“什么叫接近目标”的几何。

**个人判断。** 这篇论文应准确定位为 [[A Path Towards Autonomous Machine Intelligence - JEPA世界模型纲领]] 中 H-JEPA 与层级规划设想的端到端实证落地，而不是把概念归为某位作者的单独新主张。2022 年文章由 Yann LeCun 单独署名，本文则是五位作者共同完成的算法与实验论文；它验证的是其中一条具体路线，并没有实现那篇纲领里的 Configurator、内在代价、工作记忆或多模态不确定性系统。

### 1.2 三个对照模型必须分清

| 模型 | 时间层级 | 表征层级 | 训练与规划的核心性质 |
|---|---:|---:|---|
| **LeWM** | 否 | 单一 latent space | 单层 action-conditioned JEPA；一次预测跨 5 个环境步；平坦 latent planning |
| **HWM** | 是 | 否 | 多时间尺度、分层子目标规划，但上层 state encoder 是 identity；各层共享 level-1 latent space |
| **H-JEPA** | 是 | 是 | 每层有独立 state encoder、action encoder、predictor 和 latent space；联合端到端训练，自顶向下规划 |

HWM 是这篇论文最重要的机制对照。HWM 相对 LeWM 的改进主要测量**时间分解**：把长任务拆成较短子目标。H-JEPA 相对 HWM 的额外改进主要测量**表征分层**：高层用不同 latent geometry 评价远期进展。AntMaze 上三层 H-JEPA 为 $73.3\%$，三层 HWM 为 $38.7\%$，说明共享空间的时间层级并不足以解释全部提升；但 FourRoom 和 Push-T 上两种层级的差异较小，又说明“独立高层表征”并非在所有数据上都自动有利。

### 1.3 核心贡献与证据边界

论文的贡献可以压缩为四点：第一，给出多层 action-conditioned JEPA 的联合训练 recipe；第二，展示层级深度与时间尺度分离共同诱发选择性抽象；第三，给出 coarse-to-fine latent planning，并把时间分解和抽象 cost 的贡献分别消融；第四，用 inverse dynamics 修复 DROID 上的 slow-feature collapse，把模型扩展到多场景真实机器人视频。

这仍是**目标图像条件、离线数据驱动、梯度式动作搜索**的世界模型，不是通用机器人策略。模拟环境有闭环执行成功率；DROID 没有模拟器，只有固定录像上的开放环离线规划评分。后者不能被写成真实机器人任务成功率。

---

## 2. 任务定义、层级表示与训练样本

### 2.1 输入、输出与每层模块

训练数据是无奖励轨迹 $\{(o_t,a_t)\}$。$o_t$ 是 RGB 观测，AntMaze 还提供去掉全局 $xy$ 后的 27 维 proprioception；$a_t$ 是从当前观测到下一采样观测之间的动作块。Level 1 每一步跨 5 个环境步，因此原始动作会先按这 5 步拼接。

每一层 $\ell$ 都是独立 JEPA：

- state encoder $E^{(\ell)}$：把本层观测或低层 latent window 编码为 $z_t^{(\ell)}$；
- action encoder $A^{(\ell)}$：把动作块或低层 action embeddings 聚合为 $a_t^{(\ell)}$；
- causal predictor $F^{(\ell)}$：根据此前 latent states 与 actions 预测下一 latent；
- SIGReg：约束本层 latent marginal 接近各向同性高斯，阻止常量、低秩或聚类式坍塌；
- DROID 版本另有 inverse-dynamics head $I^{(\ell)}$。

Level 1 直接编码像素；更高层只读取下一低层的 latent stream，不返回像素预测。训练输出是一组可预测、非坍塌的多尺度 latent dynamics；部署输出是 level-1 primitive actions。

### 2.2 stride 与 window 是两件不同的事

对 $\ell>1$，stride $s_\ell$ 决定相邻高层状态隔多少低层步，window $w_\ell$ 决定一个高层状态汇总多少低层状态：

$$z_t^{(\ell)}=E^{(\ell)}\left(z_{ts_\ell:ts_\ell+w_\ell}^{(\ell-1)}\right)$$

$$a_t^{(\ell)}=A^{(\ell)}\left(a_{ts_\ell+w_\ell-1:(t+1)s_\ell+w_\ell-1}^{(\ell-1)}\right)$$

state window 长度是 $w_\ell$，而 action chunk 长度始终是 $s_\ell$，因为 macro-action 必须覆盖从当前高层状态锚点到下一高层状态的完整转移。论文的模拟主实验统一使用 $s_\ell=2,w_\ell=1$；state encoder 因而是逐点 MLP，但 action encoder 仍聚合两个低层动作。Level 1、2、3 的一次预测分别跨 5、10、20 个环境步。

### 2.3 clip 构造决定“更深是否有足够训练数据”

每层 predictor 都需要 3 个 context states 和 1 个 successor，即顶层每个 sample 要有 4 个 states。若顶层需要 $n_L=4$ 帧，则低层长度递推为：

$$n_{\ell-1}=(n_\ell-1)s_\ell+w_\ell$$

因此 LeWM、两层、三层、四层分别需要 4、7、13、25 个 level-1 frames，对应最短 16、31、61、121 个环境步。代码先从数据集中取一条足以构造顶层的长 level-1 stream，逐层按 stride/window chunk，再把每一层随机裁回 4 个 states，使每层都提供 3 个 teacher-forced transitions。

| 环境 | LeWM clips/epoch | 两层 | 三层 | 四层 |
|---|---:|---:|---:|---:|
| FourRoom | 952,708 | 915,298（96%） | 840,478（88%） | 690,838（73%） |
| AntMaze | 9,575,000 | 9,200,000（96%） | 8,450,000（88%） | 6,950,000（73%） |
| Push-T | 1,981,721 | 1,701,446（86%） | 1,143,623（58%） | 285,931（14%） |
| Cube | 1,783,600 | 1,636,600（92%） | 1,342,600（75%） | 754,600（42%） |

**论文事实。** 训练按 epoch 而不是 optimizer steps 或 FLOPs 对齐，故深模型实际得到更少 gradient steps。Push-T 平均 episode 只有 125 步，三层和四层仅保留 LeWM 的 58% 与 14% clips，这是深层失败的主要解释。

**个人判断。** 这同时构成公平性边界：不同深度共享原始数据集和 epoch budget，却不共享有效 sample 数、更新步数和训练 FLOPs。论文的设置能回答“按同一 epoch recipe 扩深会怎样”，不能完全回答“在训练计算和有效 transition 严格匹配时，层级本身是否仍失败”。

---

## 3. 分层预测学习：为什么会形成抽象

### 3.1 teacher-forced latent prediction

每层用 $c_\ell=3$ 个真实 latent states 作为 context。训练时 predictor 读取真实历史 latent，而不是把自己先前预测递归喂回去：

$$\mathcal L_{\mathrm{pred}}^{(\ell)}=\frac{1}{c_\ell}\sum_{\tau=1}^{c_\ell}\left\|F^{(\ell)}\left(z_{<\tau}^{(\ell)},a_{<\tau}^{(\ell)}\right)-z_{\tau+1}^{(\ell)}\right\|_2^2$$

这使优化更稳定，但规划时必须 autoregressive rollout，训练—推理之间仍有 exposure gap。因果 attention 保证位置 $\tau$ 不能读取未来 latent；动作通过 adaptive LayerNorm 进入 predictor，使状态变化条件于动作，而不是只做视频外观外推。

### 3.2 每层 SIGReg 与端到端梯度流

单独最小化 latent prediction MSE 存在平凡解：encoder 把所有输入变成常量，predictor 就能零误差预测。每层目标为：

$$\mathcal L^{(\ell)}=\mathcal L_{\mathrm{pred}}^{(\ell)}+\lambda_\ell\operatorname{SIGReg}\left(Z^{(\ell)}\right)$$

SIGReg 在 1,024 个随机方向上投影 latent，并用 Epps–Pulley statistic 检验一维投影与标准高斯的差异。模拟实验中 AntMaze、Push-T、Cube 各层 $\lambda_\ell=0.09$，FourRoom 的 H-JEPA 为 0.18。所有层的损失相加后一次反向传播，高层预测与 SIGReg 梯度会穿过高层 encoder，继续进入更低层 encoder；没有 stop-gradient target、EMA teacher 或预训练权重。

**作者解释。** prediction loss 要求留下“能跨本层时间跨度预测”的信息，SIGReg 则要求 latent 不能简单压缩成常量。二者拉扯后，高层会保留慢、可预测的因素，丢弃快、远期难预测的细节。

**个人判断。** 这不是“网络越深自然越语义化”，而是时间尺度、预测压力与 anti-collapse 正则共同造成的选择性信息瓶颈。若不同实体变化频率相近，或者快速变量仍高度可预测，层级未必产生预期抽象。

### 3.3 选择性抽象与 frequency gap

![](assets/H-JEPA%20-%20分层世界模型与视觉规划/figures/depth-probes-wide.png)

> **论文 Figure 4 的 probe 面板（PDF p.5；项目页 Fig.2）**：Visual AntMaze 中 body state 的 probe NMSE 随层级从接近 0 增至约 0.8，而全局 position 始终接近 0；FourRoom 中 distractor 越来越难恢复，受控 ego 位置仍清晰。NMSE 越高表示信息越不可恢复。该项目页原图只包含 probe，不含论文 Figure 4 的 decoder 面板。来源：https://h-jepa.com/assets/figures/depth_probes_wide.png

论文进一步计算每个实体轨迹的 spectral centroid，并以最快与最慢实体频率之比定义 frequency gap。AntMaze 单条代表性轨迹及全数据 gap 都约为 $8.3\times$；FourRoom、AntMaze、Humanoid 的 gap 为 $3.1$–$12.7\times$，高层明显丢失快实体；Push-T、Cube、DROID 只有 $1.2$–$1.6\times$，level 1 到 level 2 的实体可恢复性变化很小。Figure 5 显示 frequency gap 越大，fast entity 的 probe NMSE 增量通常越大，而 slow entity 保持在 level-1 水平；低于约 $2\times$ 的区域很少出现选择性抽象。

**证据边界。** probe 说明某信息能否从冻结 latent 中被 MLP 解码，不等于 planner 确实使用了该信息；频率相关性也不是因果证明。作者自己指出，快速周期运动仍可能可预测。因此论文后续必须再用 projected goal cost 和 HWM 对照，验证抽象是否真正改变规划。

---

## 4. 自顶向下的 coarse-to-fine 规划

### 4.1 总体流程

![](assets/H-JEPA%20-%20分层世界模型与视觉规划/figures/overview.png)

> **论文 Figure 1（PDF p.1；项目页 Fig.1）**：三层 AntMaze 中，level 3/2/1 每步分别覆盖 20/10/5 个环境步。顶层向最终 goal 规划，其第一个预测 latent 下传为 level 2 子目标，再递归到 level 1 输出原始动作。右侧 Pareto 曲线显示三层模型在更低规划计算量下达到更高成功率。来源：https://h-jepa.com/assets/figures/overview.png

当前观测与目标图像先自底向上编码到每层，得到 $z_0^{(\ell)}$ 和 $g^{(\ell)}$。顶层优化 macro-action sequence：

$$a_{0:H_L-1}^{(L),*}=\arg\min_{a_{0:H_L-1}^{(L)}}\left\|\hat z_{H_L}^{(L)}-g^{(L)}\right\|_2^2$$

顶层 rollout 的预测状态不是直接执行的动作，而是下层要追踪的 latent subgoals。下层先在自己的空间 rollout，再用上层 state encoder 把预测投影到上层空间：

$$\tilde z_i^{(\ell+1)}=E^{(\ell+1)}\left(\hat z_{is_{\ell+1}:is_{\ell+1}+w_{\ell+1}}^{(\ell)}\right)$$

随后优化低层 action sequence，使投影后的状态匹配上层预测：

$$a_{0:H_\ell-1}^{(\ell),*}=\arg\min\left[\left\|\tilde z_{K_{\ell+1}}^{(\ell+1)}-\hat z_{K_{\ell+1}}^{(\ell+1),*}\right\|_2^2+\beta\sum_{i=1}^{K_{\ell+1}-1}\left\|\tilde z_i^{(\ell+1)}-\hat z_i^{(\ell+1),*}\right\|_2^2\right]$$

这里 $K_{\ell+1}$ 是追踪的上层子目标数，低层 horizon 为 $H_\ell=K_{\ell+1}s_{\ell+1}+w_{\ell+1}-1$。每层都用 gradient descent 直接优化连续动作；最终只返回 level-1 actions。

### 4.2 open-loop 与 closed-loop

**Closed-loop 层级规划**取 $K_{\ell+1}=1$，低层只追踪上层第一个子目标。模拟评测每次执行两个 level-1 action blocks，即 10 个环境步，然后读取新观测并从头重规划；这属于 receding-horizon 闭环，而不是一次规划到底。最严格的一步闭环应每个 primitive command 都重规划，论文当前 commit width 仍是 10 步。

**Open-loop 层级规划**取 $K_{\ell+1}=H_{\ell+1}$。$\beta=0$ 时只匹配最后一个高层预测，$\beta>0$ 时还匹配中间 subgoals。DROID 采用开放环：level 2 规划 12 个 macro-steps，level 1 一次规划整个 36-step window，不读取新观测，不重规划，$\beta=0.5$。

### 4.3 规划为何可能更省算力

层级有两条相互独立的杠杆：

1. **时间杠杆**：高层一步跨更长时间，低层只解较短 subgoal problem，减少连续 rollout 深度与动作搜索维度；
2. **表征杠杆**：高层 latent 删除与远期目标无关的姿态细节，使距离函数更像“任务进度”而不是“像素/姿态相似度”。

这两条杠杆分别由 HWM 和 projected-cost ablation 检验。若只看完整 H-JEPA 对 LeWM，无法知道提升来自哪一条。

---

## 5. 抽象 cost geometry 与时间分解消融

### 5.1 高层空间改变“离目标多远”

![](assets/H-JEPA%20-%20分层世界模型与视觉规划/figures/goal-costs.png)

> **论文 Figure 7（PDF p.7；项目页 Fig.3）**：固定星形 anchor 后，level 1 距离在大部分走廊上几乎饱和为红色，远处状态梯度很弱；level 2/3 沿可通行走廊形成更连续的蓝—绿—黄—红过渡，anchor 附近低成本盆地也更宽。来源：https://h-jepa.com/assets/figures/goal-costs.png

在迷宫里，理想 cost 应近似绕墙后的最短路径距离，而不是欧氏图像相似度。Figure 7 表明 level-1 latent 只在 anchor 附近区分距离；更高层 cost 沿走廊平滑变化，为梯度 planner 提供远距离方向信号。

为隔离表征因素，作者保持同一个 level-1 dynamics 和同一 planner 不变，只把预测与目标先投影到 level 2/3/4，再计算距离。AntMaze 结果如下：

| 模型深度 | Native L1 | L2 cost | L3 cost | L4 cost | 完整层级规划 |
|---|---:|---:|---:|---:|---:|
| 2 层 | $23.3\pm3.3$ | $31.3\pm2.7$ | — | — | $39.3\pm3.7$ |
| 3 层 | $16.7\pm1.8$ | $20.7\pm4.8$ | $22.0\pm2.3$ | — | $73.3\pm3.5$ |
| 4 层 | $10.0\pm0.0$ | $20.7\pm1.8$ | $21.3\pm4.7$ | $26.7\pm2.9$ | 论文 Table 1 为 $63.3\pm7.7$；官方仓库深度表转录为 $67.3\pm2.4$ |

这里最后一格出现 v2 PDF Table 1 与官方仓库 `docs/RESULTS.md` 的转录差异；核心判断不受影响：所有深度至少有一个高层投影 cost 胜过 native cost，但三层完整规划 $73.3\%$ 远高于纯投影的 $22.0\%$，说明抽象 cost 只解释一部分，时间分解仍是主贡献。

### 5.2 projected cost 并非普遍有效

FourRoom 的高层投影 cost 明显改善 flat planning；Push-T 与 Cube 没有稳定收益，深层投影甚至会降低成功率。这与前述 frequency gap/probe 一致：没有形成选择性抽象时，“更高层”不保证是更好的目标空间。

**个人判断。** 这是论文最可信的一组负结果。它阻止我们把“高层 latent”神化为天然语义空间：只有训练数据、时间尺度和任务变量共同支持时，高层距离才可能更接近规划进度。

### 5.3 cost monotonicity：短子目标比终点目标更一致

作者在 held-out expert trajectories 上计算 cost 随时间的负 Spearman correlation，越接近 1 表示越单调下降。H-JEPA 的 full-horizon goal cost 在四环境为 $0.48$–$0.75$，level-1 对 level-2 subgoal 的短程 tracking cost 为 $0.89$–$1.00$：AntMaze $0.68\rightarrow0.94$，Push-T $0.77\rightarrow0.89$，Cube $1.00\rightarrow0.99$，FourRoom $0.85\rightarrow1.00$。Cube 已有近乎完美的 full-goal monotonicity，因此层级没有进一步提高；其余环境的子目标 cost 更稳定。

HWM 也出现 subgoal monotonicity 改善，支持“时间分解本身有效”。但这些数值沿 expert trajectory 测量，不是优化后计划成功率；它们解释 cost geometry，不单独证明 planner 一定成功。

---

## 6. 模拟实验：成功率、计算量与“更深不一定更好”

### 6.1 环境、数据与成功率口径

- **FourRoom Distractors**：受控红点跨房间导航，独立蓝色 distractor 随机移动/传送；训练 level 1 使用 1.0M transitions。
- **Visual AntMaze**：8-DoF quadruped、$64\times64$ 图像、8D torque 与 27D proprioception；explore 与 stitch 各 12,500 条 400-step episodes，训练约 10.05M transitions。
- **Push-T**：2D agent 推 T 形块，18,685 个训练 episodes、2.34M frames，平均 125 steps。
- **OGBench Cube**：机械臂抓放单方块，10,000 个 200-step episodes，9,800/200 划分。

最终深度比较在每环境固定 50 个新 start-goal tasks 上评估，结果为三个配对 model/planner seeds（42、43、44）的 mean ± SE。规划器设置先在另一套任务上选定，并限制在平均 100 TFLOPs/episode 内，再迁移到最终任务集。

### 6.2 18.0→73.3 的准确含义

Visual AntMaze 的 $18.0\pm3.5\%\rightarrow73.3\pm3.5\%$ 指：

- 同一类目标图像条件导航任务；
- LeWM 单层模型配 flat gradient planning 对三层 H-JEPA 配三级 top-down planning；
- 每种方法在 50 个固定任务上评估；
- 成功由环境 goal test 判断，AntMaze 是最终到达目标 $xy$ 的 1.0 距离内；
- 数值是三个训练 seed 与配对 planner seed 的 mean ± SE；
- 不是同一个模型只切换 planner，也不是单一 seed 的最好结果。

因此这组数字证明完整系统提升很大，但不能把 55.3 个百分点全部归因于 learned abstraction。HWM、projected cost、monotonicity 三组实验才把机制拆开。

### 6.3 深度主结果

| 环境 | LeWM | H-JEPA 2 | H-JEPA 3 | H-JEPA 4 | HWM 2 | HWM 3 | HWM 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AntMaze | $18.0\pm3.5$ | $39.3\pm3.7$ | **$73.3\pm3.5$** | $67.3\pm2.4$* | $34.0\pm1.2$ | $38.7\pm3.5$ | $14.0\pm2.0$ |
| FourRoom | $40.7\pm6.8$ | $81.3\pm6.7$ | $96.0\pm1.1$ | $88.0\pm2.3$ | $75.3\pm3.3$ | $99.3\pm0.7$ | $92.7\pm4.4$ |
| Cube | $32.0\pm6.1$ | $47.3\pm2.7$ | $60.0\pm5.0$ | $62.7\pm2.4$ | $46.7\pm2.9$ | $53.3\pm5.2$ | $34.0\pm3.1$ |
| Push-T | $40.0\pm2.3$ | $45.3\pm1.3$ | $17.3\pm3.5$ | $0.7\pm0.7$ | $42.0\pm6.4$ | $12.0\pm2.0$ | $0.0\pm0.0$ |

\* AntMaze H-JEPA 4 的 $67.3\pm2.4$ 来自官方仓库深度结果表；v2 PDF Table 1 的另一评测引用写 $63.3\pm7.7$。两者都低于三层，不影响“最优深度不是越深越好”。

### 6.4 成功率—计算量 Pareto

![](assets/H-JEPA%20-%20分层世界模型与视觉规划/figures/planning-results-pareto.png)

> **论文 Figure 6 上半部（PDF p.6；项目页 Fig.4）**：FourRoom、AntMaze、Cube 中，增加到三层通常把 Pareto 前沿推向左上；Push-T 的两层模型仍可竞争，三层紫线却明显崩溃。横轴是每 episode planner TFLOPs，包含 forward 与 backward；不是模型训练 FLOPs或真实 wall-clock。来源：https://h-jepa.com/assets/figures/planning-results-pareto.png

论文的“更少 compute”是 planner 计算账：候选动作通过模型 forward rollout，再通过 backward 更新，作者把两者都计入每 episode TFLOPs。计算扫描只改变每层 candidate trajectory 数，其他规划器设置固定。AntMaze 三层发布配置每层最多做 90 次 AdamW 更新，level 1/2/3 的 candidates 分别是 100/100/400，horizon 是 4/2/5；每执行两个 5-step level-1 action blocks 后重规划。

未披露的是端到端控制延迟、显存、能源和 wall-clock。因此“Pareto 更优”应限定为作者 FLOP 计数下的梯度规划，不应直接写成某个毫秒加速倍数。

### 6.5 Push-T 失败告诉了什么

Push-T 从两层 $45.3\%$ 降到三层 $17.3\%$、四层 $0.7\%$，而 HWM 也同步下降，说明失败不是高层 learned encoder 独有。最直接原因是 episode 太短：三层只有 58%、四层只有 14% 的 clips；epoch 对齐又让深模型经历更少更新。与此同时 Push-T 的频率 gap 小、没有明显选择性抽象，projected cost 也无清晰收益。

**个人判断。** 更深层级只有在三项条件同时满足时才可能有利：数据段足够长、慢/快因素确实分离、任务 horizon 足以抵消额外建模误差。论文支持“按环境选深度”，不支持“层数越多越接近智能”。

---

## 7. DROID：slow-feature collapse、inverse dynamics 与 Fréchet fidelity

### 7.1 为什么 SIGReg 仍可能保留错的信息

DROID 每个 episode 的场景、光照、相机标定和物体都变化。纯预测 JEPA 很容易找到一个满足 SIGReg、又好预测的解：编码 episode 内稳定的 scene identity，却丢掉动作相关、快速变化的机器人。latent 仍有高方差，因此不是常量坍塌；但约 75% 方差来自 episode 之间，而 Cube 只有约 28%。作者把这种“分布看起来不塌、控制主体却消失”的解称为 slow-feature collapse。

这说明 SIGReg 只约束 marginal distribution 不够：它能要求 latent 有多样性，却不保证多样性来自可控 agent。必须对相邻时间状态的 joint structure 加约束。

### 7.2 inverse dynamics 迫使表征保留可控变化

每层 inverse-dynamics head 从相邻 latent 回归连接二者的 action：

$$\mathcal L_{\mathrm{idm}}^{(\ell)}=\frac{1}{T-1}\sum_{t=1}^{T-1}\frac{1}{d_a}\left\|I^{(\ell)}\left(z_t^{(\ell)},z_{t+1}^{(\ell)}\right)-\operatorname{sg}\left[a_t^{(\ell)}\right]\right\|_2^2$$

高层目标是聚合后的 macro-action，不是 primitive action；target stop-gradient，只让 IDM 梯度经两个 state representations 回到 encoder。DROID level 1/2 的 SIGReg 权重分别是 0.08/0.16，IDM 权重是 100/50；level 2 还对 macro-action embeddings 加 0.005 的 SIGReg。

![](assets/H-JEPA%20-%20分层世界模型与视觉规划/figures/droid-ablation.png)

> **论文 Figure 9(b)（PDF p.9；项目页 Fig.6）**：无 IDM 的单层模型 fidelity 约为 0，加入 IDM 后升到约 34%；共享空间两层 HWM 约 35%；learned level-2 的 H-JEPA 约 40%。官方仓库精确转录为 LeWM+IDM $33.57\pm1.52$、HWM $35.65\pm2.22$、H-JEPA $38.42\pm0.94$。来源：https://h-jepa.com/assets/figures/droid-ablation.png

H-JEPA 超过 HWM，说明 DROID 上不仅是 stride 3 的时间分解有用，learned level-2 representation 也提供额外收益。但 DROID 的 frequency gap 小、probe 没显示模拟导航中那种强选择性抽象，因此这里的“表征层级有用”不应等同于“机器人被抽掉、目标物被保留”这一具体机制。

### 7.3 Fréchet fidelity 的定义与不能跨越的边界

DROID 没有可用模拟器。作者把规划动作积分成三维末端执行器累计路径 $p$，与记录的专家路径 $\tilde p$ 计算离散 Fréchet distance：

$$\operatorname{fidelity}=\frac{d_F(\mathbf 0,\tilde p)-d_F(p,\tilde p)}{d_F(\mathbf 0,\tilde p)}$$

- 0%：与机械臂完全不动相当；
- 100%：路径与专家完全相同；
- 负值：比不动更差。

评测在固定 16 个 held-out clips、5 fps 下进行，level 1 开放环规划 36 步，整条计划一次产生。它衡量路径形状 fidelity，比只比较终点更适合非贪心 pick-and-place；但它不测视觉反馈后的纠错、接触成功、抓取稳定性、碰撞、安全性或真实任务完成。因此 **DROID 只是离线开放环路径 fidelity，不是真机闭环成功率**。论文结论也明确把物理机器人闭环作为下一步。

---

## 8. 训练设置、资源与复现公平性

### 8.1 模拟环境训练 recipe

Level 1 是 ViT-Tiny + causal Transformer；高层是两层 latent MLP + macro-action encoder + causal Transformer。模拟环境 batch size 为 128，AdamW weight decay $10^{-3}$，1% linear warm-up 后 cosine decay，gradient clip 1.0，bf16；level-1 peak LR $5\times10^{-5}$，高层 $10^{-5}$。AntMaze 训练 5 epochs，FourRoom/Push-T/Cube 分别为 7/9/8 epochs。三种训练 seeds 为 42、43、44。

论文披露在 H200 GPU 上训练，但没有给出模拟实验 wall-clock、GPU 数、峰值显存或能耗，不能补造“训练几小时”。

### 8.2 DROID 资源账

论文 v2 的 DROID 语料口径为 74,530 episodes、220,084 camera views、三相机、$256\times256$；官方仓库 README 的发布训练说明写 74,896 episodes，属于当前发布 recipe 与论文表述差异。仓库标明完整训练数据约 91 GB；两层 DROID headline run 使用 2 GPUs、每卡 batch 128（总 batch 256）、100 epochs、每 epoch 292 iterations、bf16，seeds 为 1/1000/10000。论文说明硬件为 H200，但没有披露总训练时长。

权重下载量也不可忽略：模拟环境 4 个环境 × 7 个模型 × 3 seeds 共 84 个 checkpoint，仓库称约 9.7 GB；DROID 3 个模型 × 3 seeds 共 9 个 checkpoint，约 2.7 GB。若只验证方法，不应先复制全部权重和 91 GB DROID 数据。

### 8.3 推荐复现顺序

1. **先验证 AntMaze 三层张量与单 batch loss**：检查三层 latent 方差、SIGReg、teacher-forced MSE 和高层梯度是否进入 level 1。
2. **再复现固定 50-task evaluation**：直接下载官方 checkpoint 与 `assets/eval_trajs/ant/expert_grid_d3_n50.pt`，核对 success-rate 口径。
3. **做机制消融**：同一 H-JEPA checkpoint 上跑 native L1、L2/L3 projected cost；再与同深度 HWM 比较。
4. **检查深度公平性**：除论文 epoch-matched 设置外，额外做 optimizer-step-matched 和 consumed-transition-matched 对照，尤其是 Push-T。
5. **最后才做 DROID**：先用 252 MB 的 16 个评测 episodes 与官方 checkpoints 验证 Fréchet fidelity；只有需要重训时再下载约 91 GB 语料。

---

## 9. 代码级实现

### 9.1 仓库入口与论文模块映射

固定提交：`97054ecaff83e13839195ec4b92d74c08a0127c5`。

| 论文模块 | 代码入口 | 作用 |
|---|---|---|
| 训练总入口 | `h_jepa/main_hjepa.py` | Hydra config、dataset、Lightning、逐层 optimizer parameter groups |
| 模型构建与 loss | `h_jepa/hjepa_utils.py` | 构建每层 JEPA；teacher forcing；SIGReg/IDM；各层 loss 求和 |
| 层级容器与 clip lifting | `h_jepa/models/hjepa.py` | level-1 stream 逐层 chunk/encode、随机四帧 crop、稀疏像素编码 |
| 单层 JEPA | `h_jepa/models/jepa.py` | state/action window 对齐、encode、predict、autoregressive rollout、latent MSE cost |
| state/action encoders | `h_jepa/models/encoders/` | ViT、ResidualLatentMLP、SequenceEncoder/SequenceMLPEncoder |
| causal predictor | `h_jepa/models/predictors/predictors.py` | causal self-attention、AdaLN action conditioning |
| SIGReg / IDM | `h_jepa/loss.py` | 1,024 random projections 的 ECF statistic；相邻 latent 逆动力学 |
| 层级规划 | `h_jepa/hierarchical_solver.py` | top-down solve、跨层投影、子目标传递、goal anchoring、horizon schedule |
| 动作优化器 | `stable_worldmodel/solver/gd.py` | 对 candidate action tensors 做 AdamW 反向优化 |

### 9.2 Visual AntMaze 三层 H-JEPA 的一条具体训练张量流

以下形状来自 `base/ant.yaml`、`ant_hjepa_l3.yaml` 和实际 chunk 代码，不是按论文示意图猜测。令 batch $B=128$。

| 阶段 | 实际张量形状 | 说明 |
|---|---|---|
| 原始 level-1 pixels | `(128, 13, 3, 64, 64)` | 三层、stride 2、window 1、顶层 4 states 反推需要 13 帧 |
| level-1 proprio | `(128, 13, 27)` | qpos 去掉全局 xy 后与 qvel 合并 |
| level-1 raw action blocks | `(128, 13, 40)` | 8D torque × frameskip 5 拼接 |
| pixel encoder 输出 | `(128, 13, 256)` | ViT-Tiny CLS 经 MLP projector |
| proprio encoder 输出 | `(128, 13, 128)` | 两层 MLP |
| level-1 latent $z^{(1)}$ | `(128, 13, 384)` | 256+128 concat，fusion projector 为 identity |
| level-1 action embedding | `(128, 13, 384)` | 40D action block 经 `Embedder` |
| level-2 state 输入/输出 | `(128, 7, 384)` → `(128, 7, 384)` | starts 为 0,2,...,12；pointwise ResidualLatentMLP |
| level-2 action chunks | `(128, 7, 2, 384)` → macro `(128, 7, 4)` → `(128, 7, 384)` | 一层 Transformer `SequenceEncoder` 汇总两个低层 action embeddings，再 `Embedder` |
| level-3 state 输入/输出 | `(128, 4, 384)` → `(128, 4, 384)` | 从 7 个 level-2 states 取 0,2,4,6 |
| level-3 action chunks | `(128, 4, 2, 384)` → macro `(128, 4, 4)` → `(128, 4, 384)` | 末端不足处由 chunk 代码零填充并传 padding mask |
| 每层训练 crop | `(128, 4, 384)` | level 1/2 随机裁回 4 states；level 3 已是 4 |
| predictor context/actions | `(128, 3, 384)` / `(128, 3, 384)` | $c_\ell=3$，三层相同 |
| prediction/target | `(128, 3, 384)` | target 为同层 latent 的 positions 1:4 |
| SIGReg 输入 | `(4, 128, 384)` | 代码将 `(B,T,D)` 转为 `(T,B,D)`，每 timestep 跨 batch 测 normality |

这个流揭示两个容易误读的点。其一，高层 action latent 的“优化维度”是 4，而 predictor 条件张量在 action embed 后仍是 384 维；不能把 macro-action dim 与 condition embedding dim 混为一谈。其二，训练 loss 实际在每层随机四帧 crop 上计算，13 帧主要用于让顶层窗口可构造，并不意味着 level 1 每个 batch 都用 12 个预测 targets。

### 9.3 逐模块四段拆解

#### 9.3.1 Level-1 多模态状态 encoder

**输入输出。** 输入是 `(B,T,3,64,64)` 像素和 `(B,T,27)` proprioception；ViT-Tiny CLS 投影到 256 维，proprio MLP 输出 128 维，拼接得到 384 维 latent。

**为什么这样设计。** 图像提供全局迷宫位置，proprio 提供从单帧难以可靠读出的身体构型；移除 proprio 中的全局 $xy$，避免直接把任务答案作为低维状态输入。

**关键超参。** 图像/patch 为 $64/4$，latent 384，level-1 predictor 为 4 layers、8 heads、hidden width 192、FF width 1024、dropout 0.1；一次状态转移对应 5 个环境步。

**代码与论文差异/补充。** 代码支持 `sparse_level1_encode=True`：只对最终各层 crop 真正会读取的 level-1 frames 跑 ViT，再 scatter 回零填充长序列；这与 dense encode 数值等价，是发布实现的显存/计算优化，论文方法公式没有展开。

#### 9.3.2 高层 state encoder 与 action encoder

**输入输出。** 高层 state encoder 在 $w=1$ 时逐点接收 `(B,T,384)`，经 hidden width 384 的 ResidualLatentMLP 输出 `(B,T,384)`；action encoder 接收 `(B,T,2,384)`，用一层小 Transformer 汇总为 `(B,T,4)` macro-action。

**为什么这样设计。** state 与 action 的时间支持不同：state 只需表示采样锚点，macro-action 必须覆盖两个锚点之间的完整低层转移。独立 action encoder 让高层 planner 在紧凑 4D 空间搜索，而不直接搜索一串低层动作。

**关键超参。** AntMaze level 2/3 均为 stride 2、window 1、state latent 384、macro-action 4；action encoder queue 保留最近 4,000 个训练 macro-actions，用于拟合规划初始化 prior。

**代码与论文差异/补充。** 论文公式允许 $w>1$，发布代码也实现 full state windows、action padding mask 与 sequence state encoder；但论文所有主实验 $w=1$，因此“高层汇总状态时间窗”的一般性尚未被主结果验证。

#### 9.3.3 causal predictor、teacher forcing 与损失

**输入输出。** 每层 predictor 接收三步 `(B,3,384)` latent 与 action condition，输出 `(B,3,384)` 的逐位置 successor predictions。高层 predictor 为 5 layers、12 heads、hidden width 224、FF width 1536。

**为什么这样设计。** causal mask 允许同一次 forward 对三个前缀位置做 teacher-forced next-step prediction；AdaLN 把 action 作为每层 Transformer block 的条件，比简单拼接更直接地调制状态动力学。

**关键超参。** 模拟实验 `rollout_n=1`，prediction MSE 权重 1，SIGReg 0.09，1,024 projections、17 knots；各层损失直接求和，level 1 LR $5\times10^{-5}$，高层 LR $10^{-5}$。

**代码与论文差异/补充。** 代码还支持 `rollout_n>1` 的 autoregressive training anchor、detach target、action SIGReg 和 IDM，但 AntMaze headline config 没启用这些分支。公式写 $1/c_\ell$ 的 state-wise 均值，PyTorch 实现用 `F.mse_loss` 对 batch、time、feature 全元素取均值；只是归一化常数口径，不改变最优解。

#### 9.3.4 HierarchicalSolver 与跨层 cost

**输入输出。** solver 先把 current/goal 各自编码成每层 `(B,1,384)`；按 level 3→2→1 顺序优化候选 actions。上层返回包含初始状态与预测状态的 latent trajectory，下层取第一个或多个预测为 goals，最终输出 level-1 action blocks。

**为什么这样设计。** 顶层负责长程方向，低层只在自己可预测的短 horizon 内实现子目标；低层 predictions 经上层 encoder 后再比较，确保 cost 与 subgoal 位于同一 latent space。

**关键超参。** AntMaze 三层发布配置最多 90 个 AdamW steps；candidates 100/100/400，LR 0.06/0.1/0.1，horizon 4/2/5；level-1 action block 5，receding horizon 2 blocks，故每 10 个环境步重规划。level-1 cost 看最后两步，上层看最后一步，cross-level action cost 在 AntMaze 为 0。

**代码与论文差异/补充。** 实现有递减 horizon schedule：episode 后期顶层 horizon 降到 1 时会跳过该层；可选择仍在被跳过高层的 latent space 计算目标 cost。代码还会在整条上层链已锚定时，把最后一个预测 subgoal 替换为真实 goal embedding，减少 drift。这些是论文附录和实现细节，不应把主文图理解为每次都固定 horizon 的静态递归。

### 9.4 规划计算账与实操建议

GradientSolver 的主成本近似随“层数 × candidates × optimizer iterations × rollout horizon × predictor cost”增长，并且每步同时付 forward 与 backward。层级不是免费增加模块，而是用高层较短时间轴和低维 macro-action 搜索，换取低层较短 horizon；只有任务够长且上层 cost 更有指导性时，这笔交换才划算。

复现时应记录每层：实际 active horizon、未早停迭代数、candidate 数、forward/backward FLOPs、峰值显存与每次 replan wall-clock。论文只报告 TFLOP Pareto，不能用配置中的 90 iterations 直接推断真实每次都跑满，也不能从 FLOPs 推断端到端延迟。

---

## 10. 与 LeWM、LeWAM 和 2022 H-JEPA 纲领的关系

### 10.1 相对 LeWM

[[LeWorldModel - 像素端到端稳定JEPA世界模型]] 解决的是单层像素 JEPA 如何用 SIGReg 稳定端到端训练并用于 latent planning。H-JEPA 直接以 LeWM 为 level 1，把相同预测—正则范式递归到多层。新增贡献不是新的 anti-collapse 原理，而是独立 latent spaces、固定 stride 的多时间尺度训练与 top-down subgoals。

### 10.2 相对 LeWAM

[[LeWAM - 端到端JEPA世界动作模型]] 让 action flow matching 与 latent dynamics 联合训练，用 learned policy proposals 替代宽先验盲搜；H-JEPA 仍依赖每层 gradient solver，没有联合动作生成 policy。两者互补：LeWAM 解决“候选动作从哪里来”，H-JEPA 解决“长任务在哪个时间尺度、哪个目标空间中搜索”。自然延伸是高层 H-JEPA 产生抽象 subgoal，低层 LeWAM 用 policy proposals + model refinement 实现。

### 10.3 相对 2022 年 H-JEPA 设想

2022 年纲领把高层预测描述为忽略短期细节、服务长期子目标。本文给出了三个实证支点：probe 看到快信息逐层丢失；projected cost 看到高层几何更适合长程目标；HWM 对照看到时间分解与表征分层可以拆开。它仍没有学习可变时长事件边界，stride 是固定的；也没有学习任务分解器，高层 subgoal 只是 latent predictor 的状态预测。

---

## 11. 局限、失败模式与真实部署边界

### 11.1 作者明确承认的边界

1. 高层固定 stride/window，不能适配动作持续时间或事件边界；
2. DROID 尚未做物理机器人闭环，下一步才是验证开放环 fidelity 是否转化为真实成功；
3. 当前目标由图像指定，没有语言目标、奖励或任务级语义；
4. 层级从 action-labeled video 学习，尚未扩展到 action-free video；
5. 高层 rollout 仍是确定性预测，没有显式多模态未来或 uncertainty-aware planning。

### 11.2 进一步的个人判断

**模型误差利用。** 每层都对动作求梯度，没有 ensemble uncertainty 或 out-of-distribution penalty；macro-action prior 的 percentile clipping 只是经验性边界，不能保证计划在真实动力学支持集内。

**表征过度抽象。** 高层丢掉快速细节在导航中有利，但接触任务的成败可能正由瞬时姿态决定。Push-T 的负结果显示，如果 fast detail 与任务不可分，抽象可能没有收益。

**训练比较不完全公平。** 深模型有更多参数，却因长 clip 得到更少 updates；不同深度既非参数匹配，也非训练 FLOPs 匹配。当前结果最适合评价发布 recipe，而非层数的纯因果效应。

**目标距离仍很窄。** squared latent distance 没有路径安全、碰撞、能耗或不可逆状态约束。AntMaze cost 更平滑不等于可以直接部署到安全关键机器人。

### 11.3 真实部署前必须补的证据

至少需要：真实机器人闭环成功率与失败分类；相机/物体/背景分布外泛化；控制频率和 replan latency；接触阶段的安全约束；模型不确定性与 planner exploitation 检测；与等计算 learned policy、MPC 和 VLA baseline 的匹配比较。DROID 的 16 clips 足以验证离线机制，不足以证明 robust deployment。

---

## 12. 关键术语

- **H-JEPA**：每层拥有独立 latent space 与时间尺度的 action-conditioned JEPA hierarchy。
- **LeWM**：单层、端到端、SIGReg 稳定的 latent world model，是 H-JEPA level 1 和 flat baseline。
- **HWM**：有时间层级和分层规划、但高层 encoder 为 identity 的共享 latent-space 对照。
- **stride $s_\ell$**：高层相邻状态在低层时间轴上的采样间隔。
- **window $w_\ell$**：一个高层状态汇总的低层状态数；与 stride 独立。
- **macro-action**：高层 action encoder 聚合多个低层 action embeddings 后的紧凑动作表示。
- **teacher forcing**：训练 predictor 时使用真实历史 latent，而非此前预测 latent。
- **SIGReg**：用随机一维投影的正态性统计约束 latent marginal，防止表示坍塌。
- **selective abstraction**：高层保留慢、可预测、任务相关状态，同时丢失快、远期难预测细节。
- **projected goal cost**：保持低层 dynamics 不变，只把预测与目标投影到高层 latent 后计算距离。
- **slow-feature collapse**：latent 有方差但主要编码 episode/场景的慢特征，丢掉可控 agent。
- **inverse dynamics**：由相邻 latent 回归动作，迫使表征保留动作相关变化。
- **Fréchet fidelity**：以完整三维末端路径的离散 Fréchet distance 衡量计划对专家轨迹的离线保真度。

---

## 13. 图表索引

1. **Figure 1 / `overview.png`**：三层时间尺度、top-down subgoals 与 AntMaze 成功率—计算量前沿。
2. **Figure 4 probe / `depth-probes-wide.png`**：AntMaze body 与 FourRoom distractor 随深度变得不可恢复，受控位置保留。
3. **Figure 6 top / `planning-results-pareto.png`**：四环境 Pareto，对比 LeWM、两层和三层 H-JEPA。
4. **Figure 7 / `goal-costs.png`**：AntMaze 高层 latent distance 形成更平滑的走廊进度几何。
5. **Figure 9(b) / `droid-ablation.png`**：IDM 修复 slow-feature collapse，H-JEPA 再超过 HWM。
6. **Table 1**：AntMaze projected-cost ablation，分离表征 cost 与完整层级规划收益。
7. **Table 3**：深层 clip 长度与每 epoch 有效训练 clips，解释 Push-T 深层失败。
8. **Table 10**：full-goal 与 subgoal cost monotonicity，支持时间分解机制。

---

## 14. 材料来源

- arXiv v2：https://arxiv.org/abs/2610.06805v2
- arXiv HTML v2：https://arxiv.org/html/2610.06805v2
- 作者项目页：https://h-jepa.com/
- 官方代码：https://github.com/kevinghst/H-JEPA
- 固定 commit：`97054ecaff83e13839195ec4b92d74c08a0127c5`
- 官方数据：https://huggingface.co/datasets/jepa-world-models/h-jepa
- 官方权重：https://huggingface.co/jepa-world-models/h-jepa
- 本地 PDF：[paper.pdf](assets/H-JEPA%20-%20分层世界模型与视觉规划/sources/paper.pdf)
- 完整溯源、校验值与读取限制：[sources.md](assets/H-JEPA%20-%20分层世界模型与视觉规划/sources.md)
