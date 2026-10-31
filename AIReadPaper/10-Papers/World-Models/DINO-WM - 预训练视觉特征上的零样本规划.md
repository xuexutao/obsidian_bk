---
type: paper-note
status: complete
domain: World-Models
paper: "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"
year: 2025
arxiv: "2411.04983"
doi: "10.48550/arXiv.2411.04983"
venue: "ICML 2025 (PMLR 267:79115-79135)"
source: "https://arxiv.org/abs/2411.04983"
project: "http://dino-wm.github.io"
code: "https://github.com/gaoyuezhou/dino_wm"
tags:
  - World-Models
  - JEPA
  - DINOv2
  - Model-Predictive-Control
  - Offline-RL
created: 2026-10-15
updated: 2026-10-31
---

# DINO-WM：预训练视觉特征上的零样本规划

## 基本信息

- **作者：** Gaoyue Zhou、Hengkai Pan、Yann LeCun、Lerrel Pinto
- **正式发表：** 第 42 届 International Conference on Machine Learning（ICML 2025），PMLR 267:79115–79135
- **版本：** arXiv v1 于 2024-11-07 提交，v2 于 2025-02-01 更新；本笔记以 v2 官方 PDF 及其 TeX 源包为主
- **研究对象：** 从离线视觉—动作轨迹学习任务无关的潜空间动力学，并在测试时针对新的目标图像直接优化动作
- **开源情况：** 官方代码、数据及 PointMaze / PushT / Wall checkpoint 已公开；论文未披露完整训练墙钟时间与总 GPU 小时

## 官方实现锁定与源码入口（2026-10-31）

### 入口表

| 对象 | 固定 revision | 入口 | 本次核验结论 |
|---|---|---|---|
| 官方代码 | `gaoyuezhou/dino_wm@0a9492fa12044b852ae9e001cc74604b79c8bb0c`（`main`） | `train.py` | Hydra 实例化 encoder/predictor/decoder，训练入口；该 commit 是公开仓当前 HEAD |
| 模型主链 | 同上 | `models/dino.py`、`models/visual_world_model.py`、`models/vit.py` | DINOv2 patch→动作/本体拼接→frame-causal ViT→latent MSE→可选 decoder |
| 规划 | 同上 | `planning/cem.py`、`planning/mpc.py`、`planning/objectives.py` | 候选 `[N,T,K]` 批量 rollout、终点 latent MSE、执行前若干动作后重规划 |
| 训练 config | 同上 | `conf/train.yaml`、`conf/predictor/vit.yaml`、`conf/encoder/dino.yaml`、`conf/decoder/vqvae.yaml` | `H=3`、224 输入、DINOv2-S/14、6 层/16 头 predictor、action/proprio emb=10 |
| PushT 规划 config | 同上 | `conf/plan_pusht.yaml` | CEM `horizon=5,num_samples=300,topk=30,opt_steps=30`，MPC 每轮执行 5 个聚合动作 |
| 数据+权重镜像 | `gaoyuezhou/dino_wm` HF dataset `d5c69e8fe60c3ed6b01d2f6e88fbfd66be078e57` | `checkpoints/outputs.zip`、四个 dataset archive | checkpoint 只覆盖 PointMaze / PushT / Wall；Rope/Granular 无公开 checkpoint |
| 论文 | arXiv `2411.04983v2` / PMLR 267 | 本地 `paper.pdf` | 方法与实验主张；若论文文字和仓库当前 config 不同，本文分别标注 |

### frame → DINO patches → latent dynamics → decoder/loss → CEM 的全链路 shape

记批大小 $B$，历史长度 $H$，预测步数 $P=1$，RGB 高宽 $S=224$，DINO patch 边长 $p=14$，patch 数 $N=(196/14)^2=196$，DINO-S/14 通道 $E=384$，环境原始动作维 $K_0$，`frameskip=F`，聚合动作维 $K=F K_0$，action/proprio embedding 均为 10。

| 阶段 | 输入 | 代码变换与原因 | 输出 |
|---|---|---|---|
| dataset window | RGB `[B,H+P,3,224,224]`；action `[B,H+P,K]` | 取连续窗口；`F=5` 时把 5 个低层动作 concat，令一个模型步对应 5 个环境步 | 默认 `[B,4,3,224,224]` 与 `[B,4,K]` |
| DINO preprocess | `[B(H+P),3,224,224]` | 为让 DINO patch 网格与 16× decoder 网格都为 14×14，代码 resize 到 `196×196` | `[B(H+P),3,196,196]` |
| frozen DINOv2-S/14 | 上一行 | 取 `x_norm_patchtokens`，冻结感知坐标系避免端到端表征坍塌 | `[B,H+P,196,384]`；默认 `[B,4,196,384]` |
| action / proprio encoder | action `[B,H+P,K]`，proprio `[B,H+P,K_p]` | 各用 Conv1d/线性式投影到 10 维，再 tile 到 196 个 patch；动作条件在每个空间 token 都可见 | 各 `[B,H+P,196,10]` |
| predictor input | 三者 concat | `concat_dim=1`，通道 $D=384+10+10=404$；取前 $H$ 帧并展平 time×patch | `[B,H·196,404]`；默认 `[B,588,404]` |
| latent dynamics | 上一行 | 6 层 ViT、16 heads、head dim 64、MLP 2048；frame-level causal mask 阻止看未来 | `[B,H·196,404]→[B,H,196,404]` |
| supervised target | 编码序列右移一帧 | 只在 visual+proprio 上做 MSE，排除 action 通道；目标 detach | visual `[B,H,196,384]`，proprio `[B,H,196,10]`；标量 `z_loss` |
| decoder（辅助） | predicted/frozen DINO visual `[B,H,196,384]` | 默认 `VQVAE(channel=384,n_embed=2048,quantize=False)`，两次 4× 上采样；decoder loss 对 predictor `detach` | `[B,H,3,224,224]`；像素 MSE 只训练 decoder |
| open-loop rollout | 初始 latent `[B,H,196,404]` + actions `[B,T,K]` | 每步预测下一组 patch、覆盖其中的 action 条件、再拼回历史 | visual trajectory `[B,H+T+1,196,384]` |
| CEM | 规划代码按 $B=1$，均值/方差 `[T,K]` | 每轮采样 `N_c` 条 `[N_c,T,K]`，并行 rollout；按终点 visual MSE + $\alpha$·proprio MSE 排序 | elite `[K_e,T,K]` → 新 `μ,σ:[T,K]` |
| MPC | CEM 最优动作 `[B,T,K]` | 执行前 `k` 个动作，读取真实新帧，再规划；降低开环误差 | 最终执行序列 `[B,T_{exec},K]` |

**代入官方 PushT config**：`H=3,P=1,F=5`。PushT 底层动作通常为 2D，因此聚合 $K=10$；单个训练样本从 `[4,3,224,224]` 经 DINO 得 `[4,196,384]`，再拼 action/proprio 后 predictor 输入 `[588,404]`。规划时 `T=5,N_c=300,K_e=30`，每轮候选动作张量 `[300,5,10]`；展开到环境层等价于 25 个低层步。30 轮 CEM 后 MPC config 执行 5 个聚合动作再取新观测。论文正文曾用 100 samples/10 iterations 报告延迟，**不能与仓库 PushT 当前 300/30 config 混写**。

### 模块选择、超参与差异

| 模块 | 源码超参 | 为什么 | 与常见替代方案的差异 |
|---|---|---|---|
| encoder | `dinov2_vits14`，patch 14，384D，冻结 | 保留局部空间几何；固定 target 防 collapse | CLS 只有 `[B,1,384]`，规划显著更差；端到端 encoder 可适配但需 anti-collapse |
| dynamics | depth 6、heads 16、MLP 2048、dropout 0.1 | 19M 级模型足够拟合环境局部动力学 | 一帧内并行预测全部 patch，不做逐 patch 自回归 |
| action injection | emb 10，tile + channel concat | 让每个 patch 直接获得同一动作条件，结构简单 | 不是 action token cross-attention，也不是 inverse model |
| loss | latent/proprio MSE；decoder MSE 隔离梯度 | 优先控制相关表示，不让像素细节扭曲 predictor | decoder 只用于可视化/图像指标，不参与规划代价 |
| optimization | 100 epochs，batch 32；源码 `predictor_lr=5e-4`、decoder/action `3e-4/5e-4` | 小模型单环境训练 | 原笔记中的 `predictor_lr=5e-5` 与当前公开 config 不符，已按源码修正 |
| planner | CEM + receding-horizon MPC | 接触任务多峰、非凸，黑盒搜索比动作梯度稳 | 成本是大量并行 rollout 与较高延迟 |

### 复现成本

- **最低**：下载固定 revision 的 PointMaze checkpoint 与数据，只跑 `plan_point_maze.yaml`；需要兼容的 MuJoCo 2.1/Linux 环境。
- **标准**：单 GPU 训练 predictor，公开 Slurm config 请求 1×H100、batch 32、100 epochs；实际也可按显存减 batch。冻结 DINO features 可预缓存以降成本。
- **PushT**：训练本身不大，主要成本是 300×5×30 的 CEM rollout 与 50 次评估。
- **Rope/Granular**：还需 Docker、NVIDIA Docker、PyFleX 编译和昂贵模拟；官方无 checkpoint，复现成本最高。
- **不可由公开材料精确还原**：论文未报告训练 wall-clock/GPU-hours，不能编造总费用。

## 一句话结论

DINO-WM 的关键不是发明新的 planner，而是把**冻结的 DINOv2 空间 patch 特征**当作稳定、不可坍塌的视觉状态，把世界模型训练简化为动作条件的下一时刻特征回归；在同一环境动力学内，它无需奖励、价值函数或目标任务训练，就能把任意目标图像转成潜空间终点，并用 CEM-MPC 在六类环境中规划。它证明了“预训练表征质量”可以比“更复杂的生成式世界模型”更决定规划效果，但其所谓 zero-shot 主要是**新目标零样本**，不是对全新动力学或真实机器人域的零样本。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

传统策略学习把观测直接映射为动作。部署时只有一次前向推理，面对训练中没覆盖的目标或布局时，策略没有显式的“想象—比较—修正”过程。模型式方法则先学习环境转移，再在测试时搜索动作，但常见 world model 仍有三类依赖：在线与环境交互以持续修正模型、面向固定任务的奖励/价值监督、或像素重建与生成所带来的巨大建模负担。

DINO-WM 将问题限定为视觉控制 POMDP $(\mathcal{O},\mathcal{A},p)$。离线数据只包含轨迹中的 RGB 观测与执行动作；测试时给定当前观测 $o_0$ 和目标观测 $o_g$，系统要找出动作序列 $a_{0:T-1}$，使真实环境最终接近目标。这里没有目标类别、语言指令、人工奖励或预训练 inverse model，目标本身就是一张图。

作者希望同时满足三点：第一，模型能完全从预收集的离线轨迹训练；第二，部署时仍能通过优化行为而不是只执行固定 policy；第三，同一个模型可以面对训练时没有定义过的目标。核心问题因此变成：**是否能找到一种足够通用、足够空间化、又不必在每个环境从头学起的视觉状态表示，让简单动力学预测足以支撑规划？**

### 1.2 现有方法的瓶颈

像素生成模型把背景纹理、光照、高频细节与真正影响控制的物体位置、形状和接触关系一起预测，优化难且多步误差容易累积。DreamerV3、IRIS 等方法在其原始设定中还会联合奖励、终止或 actor-critic；拿掉这些任务信号后，表征未必仍适合目标到达。另一条路线是把整张图压成一个 CLS 向量，但全局向量可能丢失“物体在左还是右、门开在哪里、颗粒分布在哪个区域”等局部几何。

DINO-WM 的判断是：大规模自监督预训练已经替世界模型解决了相当一部分“什么是稳定视觉实体、哪里有边界与空间结构”的问题。与其在小型环境数据上重新学 encoder，不如冻结 DINOv2，直接学习其 patch token 的时序变化。

![](assets/DINO-WM%20-%20预训练视觉特征上的零样本规划/figures/01-teaser.png)

> **图 1：训练、测试时优化与性能动机总览。** 左侧只用观测—动作—下一观测训练 DINO 特征动力学；中间在测试时搜索动作，使预测终点 $z_T$ 接近目标特征 $z_g$；右侧以 Granular 的 Chamfer Distance 展示 DINO-WM 明显优于 IRIS、TD-MPC2 与 DreamerV3。来源：论文 Figure 1，第 3 页，arXiv v2 TeX 源图。

### 1.3 核心贡献

1. **冻结预训练 patch 表征的世界模型。** DINOv2 encoder 不随环境训练，transition 只预测未来 patch features，既避开像素重建，也保留局部空间信息。
2. **把任务求解统一成视觉目标到达。** 任意目标 RGB 图像经同一 encoder 得到 $z_g$，测试时用 CEM 与 MPC 搜索动作，无需另训 reward model、value、policy 或 inverse dynamics。
3. **跨六类环境的系统实验。** 任务覆盖迷宫、隔墙导航、机械臂姿态、接触丰富的 Push-T、绳索和颗粒操控，并包含未见布局/物体形状/粒子数量的配置泛化。
4. **证明表征选择比“是否生成像素”更关键。** DINO patch 显著优于 R3M、ImageNet ResNet 和 DINO CLS；独立 decoder 仅用于解释结果，反向传播重建损失反而降低规划成功率。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

训练数据为离线轨迹 $\{(o_t,a_t)\}$。$o_t$ 是 RGB 图像，所有环境对外统一为 $224\times224$；实现中 DINOv2 输入再调整到 $196\times196$，产生 $14\times14$ 个、每个 384 维的 patch token。若环境提供 proprioception，模型以与动作相同的方式把它拼接进每个 patch token，但核心方案可以只依赖图像与动作。

训练输出不是未来像素，而是下一帧固定 DINOv2 表征 $z_{t+1}$。测试输入是当前图 $o_0$、目标图 $o_g$ 与动作边界；输出是一段由规划器反复优化和执行的动作序列。可选 decoder 只把预测 latent 还原为图像供人观察和计算 LPIPS/SSIM，不参与默认 transition 训练，也不参与规划代价。

方法依赖四个重要假设：离线数据覆盖将要规划的关键状态—动作区域；DINOv2 patch 距离与控制任务中的“接近目标”有足够一致性；动作标签真实可用；短期 transition 可通过有限历史 $H$ 近似。若目标要求区分 DINOv2 不敏感的属性，或规划把模型推到离线覆盖之外，潜空间 MSE 很可能不再代表真实成功。

### 2.2 潜空间动力学

每帧由冻结 encoder 映射为 patch 矩阵：

$$
z_t=\operatorname{enc}(o_t)\in\mathbb{R}^{N\times E},
$$

其中 $N$ 是 patch 数、$E$ 是 token 维度。动作 $a_t\in\mathbb{R}^{K}$ 经 MLP $\phi$ 升维后拼接到每个 patch。transition $p_\theta$ 读取最近 $H$ 帧潜状态与动作，预测下一帧全部 patch：

$$
\hat z_{t+1}=p_\theta(z_{t-H:t},\phi(a_{t-H:t})).
$$

训练损失是简单平方误差：

$$
\mathcal{L}_{\mathrm{pred}}=
\left\|p_\theta\!\left(\operatorname{enc}(o_{t-H:t}),\phi(a_{t-H:t})\right)
-\operatorname{enc}(o_{t+1})\right\|_2^2.
$$

公式看似朴素，但目标空间的选择承担了绝大多数 inductive bias：DINOv2 把像素压缩成带语义和空间结构的 token；模型不需要分配容量去逐像素复原，只要学会在动作作用下哪些 token 如何变化。

---

## 3. 核心方法

### 3.1 总体框架与完整数据流

![](assets/DINO-WM%20-%20预训练视觉特征上的零样本规划/figures/02-method-pipeline.png)

> **图 2：DINO-WM 的完整 pipeline。** 历史观测经冻结 DINOv2 变成 patch latents，动作序列驱动 transition 自回归预测；目标图独立编码为 $z_g$，planner 只在潜空间比较 $\hat z_T$ 与 $z_g$。橙色圆圈强调动作是在测试时优化而非由预训练 policy 给出。来源：论文 Figure 2，第 5 页；由 arXiv v2 的矢量源图 `arch.pdf` 完整渲染。

训练阶段可概括为“固定感知坐标系 + 可学习动力学”。DINOv2 始终冻结，ViT transition 接收 patch token、动作 embedding 和可选本体状态。训练使用 teacher forcing：预测每一步时读取真实历史 latent，而不是先前自己生成的 latent。测试阶段则必须把预测重新喂给模型，展开多步 imagined rollout。由此产生明确的 train-test gap，但 MPC 的闭环重规划会在执行少量动作后重新读取真实观测，抑制长期开环漂移。

transition 基于 ViT，但去掉原始图像 tokenization 层，因为输入已经是 patch embeddings。模型深度 6、16 个 attention heads、MLP 维度 2048，约 19M 参数。每个时间点的一组 patch 被当作一个完整 frame 同步预测，而不是像 IRIS 那样在帧内继续逐 token 自回归。这样既减少串行预测，也让同一帧全部空间位置共同形成下一时刻的全局结构。

### 3.2 因果注意力为什么关键

如果一次输入多个连续观测但没有正确 mask，模型在训练时预测较早输出时可能偷看更晚的输入帧。它会学到一种测试时不可用的捷径，历史越长反而越差。DINO-WM 采用 frame-level causal mask，使时刻 $t$ 的预测只能关注过去状态和动作。

PushT 消融非常直接：历史长度 $h=1$ 时有无 mask 都是 0.76；$h=2$ 时无 mask 降到 0.36，而有 mask 达到 0.88；$h=3$ 时无 mask 只剩 0.08，有 mask 达到 0.92。这里提升不是“Transformer 更强”，而是消除了信息泄漏，并让更长历史真正提供速度、加速度和动量线索。

### 3.3 anti-collapse：为什么这里不需要 teacher 或 stop-gradient

联合训练两路 encoder 的 JEPA/BYOL 类目标若只最小化两个 embedding 的距离，存在所有输入都输出同一常数的平凡解，因此常见方案引入 EMA teacher、stop-gradient、方差/协方差正则或负样本。DINO-WM 的结构不同：**目标 encoder 是已经训练好的 DINOv2，并且完全冻结；输入侧使用的也是同一个冻结 encoder。** transition 无权把所有目标 $z_{t+1}$ 改成常数，因此“通过 encoder 共谋使 loss 归零”的表征坍塌路径被结构性切断。

这不等于 transition 不会欠拟合。若条件信息不足，多解未来的 MSE 最优值会趋向条件均值；模型也可能忽略动作，预测数据中最常见的变化。那是动力学平均化或 action collapse，不是 encoder representation collapse。论文通过动作条件规划与多环境结果间接证明模型利用了动作，但没有专门报告 action sensitivity、互信息或多模态预测校准。因此应准确表述为：DINO-WM **消除了端到端联合表征坍塌问题**，却没有消除确定性 MSE 预测面对多模态未来时的均值化问题。

### 3.4 可选 decoder 与“无重建”边界

可选 decoder $q_\psi$ 用转置卷积把 $z_t$ 映回 $o_t$：

$$
\mathcal{L}_{\mathrm{rec}}=\|q_\psi(z_t)-o_t\|_2^2.
$$

默认情况下，decoder 独立训练，梯度不进入 transition。作者所说“without reconstructing the visual world”指的是**世界模型学习和规划不需要像素重建目标**，不是论文从未训练 decoder；为了展示 rollout 并计算图像指标，他们仍训练了解码器。把 decoder loss 反传到 predictor 后，PushT 成功率从 0.92 降到 0.80，支持“像素细节会干扰控制相关潜预测”的解释，但这项消融只在一个环境报告，不能外推成所有 reconstruction objective 都有害。

### 3.5 测试时规划：CEM + MPC

当前图和目标图首先被编码：

$$
\hat z_0=\operatorname{enc}(o_0),\qquad z_g=\operatorname{enc}(o_g).
$$

对候选动作序列展开动力学后，终点代价为：

$$
\mathcal{C}(a_{0:T-1})=\|\hat z_T-z_g\|_2^2,
\qquad \hat z_{t+1}=p_\theta(\hat z_t,a_t).
$$

CEM 从高斯分布采样 $N$ 条动作序列，保留代价最低的 $K$ 条更新均值与协方差，迭代后执行前 $k$ 个动作，再基于新观测重规划。论文也尝试直接对动作做梯度下降，但 PointMaze / PushT 只有 0.22 / 0.28，明显低于纯 CEM 的 0.80 / 0.86；加入 receding-horizon MPC 后进一步到 0.98 / 0.90。可能原因是非凸长时 rollout 中梯度局部性强、模型误差可被优化器利用，而 CEM 对离散接触和多峰动作更稳健。

### 3.6 推理复杂度

transition 本身不大，但规划要批量展开大量候选序列。论文在 NVIDIA A6000 上报告 batch 32 的单步 inference 为 0.014 秒；CEM 使用每轮 100 样本、10 轮优化时，完整 planning 为 53 秒。对比高成本可变形体模拟器单步 3 秒，模型 rollout 很快；但对实时机器人闭环而言，53 秒仍远非“轻量实时”。DINO-WM 的计算优势准确说是**相对昂贵物理仿真更便宜**，不是端到端规划延迟已经解决。

---

## 4. 数据集与实验设置

### 4.1 六类环境与离线数据

| 环境 | 核心任务 | 训练数据 |
|---|---|---|
| PointMaze / Maze | 2-DoF 小球在迷宫中到达目标 | 2,000 条随机轨迹，每条 100 步 |
| Wall | 穿过门洞，从一侧房间导航到另一侧 | 固定布局 1,920 条、每条 50 步；随机布局 10,240 条 |
| Reacher / Reach | 两关节机械臂整体姿态匹配，而非只匹配末端 | 3,000 条、每条 100 步 |
| Push-T | 推动 T 形块并同时匹配 agent 与物块目标位置 | 18,500 个带不同噪声的已发布轨迹样本，长度 100–300 |
| Rope | XArm 操作软绳到目标形状 | 1,000 条随机轨迹；训练切片使用 5 步 |
| Granular | 约百颗粒聚拢成指定方块 | 1,000 条随机轨迹；训练切片使用 5 步 |

前三个配置泛化族为 WallRandom、PushObj 和 GranularRandom。WallRandom 测试未见门墙位置；PushObj 用四种形状训练、两种新形状测试；GranularRandom 测试粒子数少于训练的一半。值得注意的是，摘要强调“不需要 expert demonstrations”，但附录明确说 Push-T 数据来自已发布 expert trajectories 的重放并加噪。较稳妥的理解是：方法不需要**针对测试目标的示范或示范条件损失**，但并非每个训练数据集都完全与 expert behavior 无关。

### 4.2 Baseline 与指标

主要 world-model baseline 为 IRIS、DreamerV3、TD-MPC2，作者把它们都改为使用同一离线数据、去掉奖励和任务信息，再在 learned dynamics 上做 MPC。表征消融包括 R3M、ImageNet ResNet-18、DINO CLS。生成模型对比为 AVDC 及其 action-conditioned 变体。

Maze、Wall、Reach、PushT 各抽 50 组初始—目标状态，评价 Success Rate；Rope 与 Granular 因环境步进昂贵，各评 10 个实例，用 Chamfer Distance。预测图像经独立 decoder 还原后报告 LPIPS 与 SSIM。成功率样本量不大，尤其可变形体只有 10 个实例，论文也没有给置信区间，因此小差异不宜过度解读。

### 4.3 实现细节与复现入口

所有环境共享约 19M 的 predictor，AdamW 训练 100 epochs，batch size 32；当前官方仓 `conf/train.yaml@0a9492f` 中 predictor / decoder / action encoder 学习率分别为 $5\times10^{-4}$、$3\times10^{-4}$、$5\times10^{-4}$，action embedding 维度 10（论文若采用其他运行配置，应以论文实验记录为准）。PointMaze、Reacher、PushT 的历史 $H=3$、frameskip=5；Wall 的 $H=1$、frameskip=5；Rope 与 Granular 的 $H=1$、frameskip=1。官方仓库给出 Conda 环境、训练与规划命令；可变形体环境还需 Docker、NVIDIA Docker 和 PyFleX，工程门槛显著高于普通 MuJoCo 任务。

---

## 5. 实验结果

### 5.1 主要定量结果

| 模型 | Maze SR↑ | Wall SR↑ | Reach SR↑ | PushT SR↑ | Rope CD↓ | Granular CD↓ |
|---|---:|---:|---:|---:|---:|---:|
| IRIS | 0.74 | 0.04 | 0.18 | 0.32 | 1.11 | 0.37 |
| DreamerV3 | **1.00** | **1.00** | 0.64 | 0.30 | 2.49 | 1.05 |
| TD-MPC2 | 0.00 | 0.00 | 0.00 | 0.00 | 2.52 | 1.21 |
| DINO-WM | 0.98 | 0.96 | **0.92** | **0.90** | **0.41** | **0.26** |

结果说明简单导航并不是 DINO-WM 的绝对优势区：DreamerV3 在 Maze / Wall 达到 1.00。真正拉开差距的是精细姿态、接触和多物体场景。TD-MPC2 在无奖励版本中全部失败，也提示它的 latent 本来就是被 reward/value 塑形的；这不是对原始 TD-MPC2 在线控制能力的公平否定，而是说明其表征不适合被直接拿来做无任务潜空间目标匹配。

预训练表征消融更加关键。DINO patch 在 Wall / Reach / PushT 上分别为 0.96 / 0.92 / 0.90，DINO CLS 只有 0.58 / 0.60 / 0.44；R3M 为 0.34 / 0.40 / 0.42，ResNet 为 0.12 / 0.06 / 0.20。Maze 对空间精度要求低，各 encoder 都接近 0.94–0.98；任务一旦要求局部几何，patch token 的优势显现。这支持作者最核心的因果链：不是“DINO 名字本身”带来提升，而是**预训练且保持空间分辨率的 patch representation**。

### 5.2 定性结果与预测质量

![](assets/DINO-WM%20-%20预训练视觉特征上的零样本规划/figures/04-open-loop-rollouts.png)

> **图 4：Push-T 与 Granular 的开环未来预测。** 在给定初帧与动作序列后，DINO-WM 经独立 decoder 还原的轨迹更贴近 GT；DreamerV3 与 IRIS 在接触、物块姿态或颗粒聚集上逐步漂移。来源：论文 Figure 4，第 6 页，arXiv v2 TeX 源图。

DINO-WM 在 PushT / Wall / Rope / Granular 的 LPIPS 为 0.007 / 0.0016 / 0.009 / 0.035，均为最优；DINO CLS 对应 0.039 / 0.004 / 0.029 / 0.086，AVDC 为 0.046 / 0.030 / 0.060 / 0.106。SSIM 同样为 0.985 / 0.997 / 0.985 / 0.940。需要谨慎的是，这些图像来自后训 decoder；低 LPIPS 证明预测 latent 保留了可解码的视觉结构，但不等于模型在像素似然或生成多样性上优于专用视频生成模型。

![](assets/DINO-WM%20-%20预训练视觉特征上的零样本规划/figures/03-key-results.png)

> **图 8：三类规划结果定性比较。** DINO-WM 在 PointMaze 找到正确通道，在 Push-T 同时对齐蓝色 agent 与 T 块，在 Granular 更接近方块目标；DINO CLS 与 DreamerV3 暴露出空间表示或长时动力学不足。来源：论文 Figure 8，附录第 16 页，arXiv v2 TeX 源图。

### 5.3 配置泛化

| 模型 | WallRandom SR↑ | PushObj SR↑ | GranularRandom CD↓ |
|---|---:|---:|---:|
| IRIS | 0.06 | 0.14 | 0.86 |
| DreamerV3 | 0.76 | 0.18 | 1.53 |
| DINO CLS | 0.64 | 0.18 | 1.36 |
| DINO-WM | **0.82** | **0.34** | **0.63** |

WallRandom 表明模型能在未见门墙组合中利用局部空间结构；GranularRandom 在粒子数量大幅变化时仍最好，作者推测 patch 表示让每个局部区域的变化仍较接近训练分布。PushObj 的 0.34 虽然领先，却也说明仅见四种形状不足以推断新物体的接触参数。这一结果更像“有限组合泛化”，距离跨材质、跨相机、跨真实机器人还有明显距离。

### 5.4 消融实验

数据规模在 PushT 上呈单调收益：200 / 1,000 / 5,000 / 10,000 / 18,500 条数据对应成功率 0.08 / 0.48 / 0.72 / 0.88 / 0.92，LPIPS 从 0.056 降到 0.005。它说明冻结强 encoder 并没有消除动力学数据需求，只是让数据更有效地用于学习 transition。

关键消融还包括：因果 mask 在 $h=3$ 时将成功率从 0.08 提升到 0.92；去掉 decoder 反传把 PushT 从 0.80 提升到 0.92；规划器从 GD 的 0.22 / 0.28 提升到 CEM 的 0.80 / 0.86，再由 MPC 达到 0.98 / 0.90。换言之，论文结果由三部分共同构成：DINO patch 坐标系、无泄漏的时序训练、以及闭环黑盒规划。只复制网络而忽略 mask 和 MPC，很难复现主表。

### 5.5 失败模式

1. **数据覆盖外推。** CEM 会主动搜索低代价动作，最容易找到模型的错误盲区；MPC 能纠偏但不能挽救严重 OOD。
2. **目标图距离不等于任务成功。** DINO 特征可能弱化颜色、纹理、细小接触或关节角差异，终点 MSE 也没有显式碰撞、安全和能耗约束。
3. **确定性均值化。** 单一未来 MSE 无法表达多模态动力学、遮挡后物体状态或随机接触，长期预测可能变成平均轨迹。
4. **teacher forcing 暴露偏差。** 训练看真实历史，规划看自身预测；开环越长，误差越可能累积。
5. **新物理属性。** PushObj 只有 0.34，说明视觉形状先验不能替代质量、摩擦、柔性等动力学辨识。
6. **规划延迟。** 53 秒 CEM 规划对在线控制过慢，且论文主要是模拟器，没有报告真实机械臂闭环延迟与安全性。

---

## 6. 与 JEPA 主线及相关工作的关系

DINO-WM 与 JEPA 共享“不要为理解世界而重建所有像素，而是在抽象 embedding 中预测”的核心立场。它的 transition loss 也是 target embedding prediction，因此可视为一种**动作条件、时间轴上的 latent predictive architecture**。但它不是 I-JEPA / V-JEPA 意义上的端到端 JEPA：encoder 已由 DINOv2 预训练并冻结，论文没有通过当前环境的预测目标继续塑造视觉表征，也没有 context encoder 与 EMA target encoder 的联合训练。

这一区别决定了 anti-collapse 策略。I-JEPA、V-JEPA 需要 stop-gradient / teacher 路径稳定移动目标；LeJEPA 试图用 SIGReg 约束分布；DINO-WM 则把 encoder 完全移出优化问题，以“固定靶”换取稳定。优点是训练简单、绝不发生联合 encoder collapse；代价是表示上限和盲点被 DINOv2 锁死，动力学数据无法让 encoder 对摩擦、速度或控制关键细节变得更敏感。

与 Dreamer/IRIS 的差异是，它不学习任务 reward，也不把世界压成单个全局 latent 或离散 token 序列；与 TD-MPC2 的差异是，它追求 task-agnostic target matching，而不是让 reward/value 把 latent 塑成某个任务的最优控制坐标系。两类方法不存在简单优劣：固定任务中，reward-shaped latent 可能更高效；开放目标中，DINO-WM 的目标图接口更灵活。

### 6.1 在 JEPA 技术谱系中的准确定位

如果把 JEPA 主线拆成“目标表示如何产生、预测器预测什么、如何防止坍塌、下游怎样使用”四个问题，DINO-WM 的选择十分鲜明。目标表示来自外部完成预训练的 DINOv2；预测器输出整帧 patch token；防坍塌依靠彻底冻结表示网络；下游用潜距离直接驱动控制。I-JEPA 主要通过图像遮挡预测学习静态语义，V-JEPA 把目标扩展到视频时空块，LeJEPA 则试图用显式分布正则取代 teacher-student 启发式。DINO-WM 没有回答“如何从当前机器人数据端到端学出更好的表示”，而是回答“已有强表示时，最小化的动作动力学系统能走多远”。

这也给后续研究留下一个可检验的中间路线：保持 DINOv2 大部分层冻结，只开放高层 adapter 或低秩参数；一边维持固定目标避免整体坍塌，一边用动作条件预测让表示对速度、接触和可控性更敏感。评价时必须同时监控视觉语义保持、动作可辨识性、多步误差和规划成功率，否则 end-to-end 微调可能只是把自然图像能力换成训练环境捷径。还应加入跨相机、跨背景和跨材质测试，确认表示真正学习了可控因素，而不是依赖渲染器中的偶然相关性。另一个方向是让 SIGReg 一类 anti-collapse 正则约束可训练 adapter，并以冻结 DINO 分支作为语义锚点，从而连接 DINO-WM 的稳定性与 LeJEPA 的可塑性。

---

## 7. 局限、未来工作与批判性评价

### 7.1 作者明确承认的局限

作者列出三点：离线数据必须覆盖足够多状态—动作区域；训练依赖 ground-truth action，无法直接利用海量无动作互联网视频；当前直接在低层动作空间规划，未来应加入高层规划与低层控制的层级结构。论文还提出可结合 exploration 在线扩充数据，并为 gradient planning 加正则以提高稳定性。

### 7.2 个人批判

**第一，“zero-shot”容易被读得过宽。** 模型已经在目标环境的大量轨迹上训练，测试只是换目标或受控配置。它没有证明拿到一个全新真实场景即可零样本行动。更准确的术语是“offline-trained, goal-zero-shot planning”。

**第二，强预训练先验与公平比较纠缠。** DINO-WM 借用了 DINOv2 的互联网级视觉预训练，而 DreamerV3/IRIS 主要从当前离线数据学表征。实验确实证明“预训练 patch prior 有价值”，但不能单独归因于 world-model architecture 更优。理想对照应让 baseline 也获得同等级预训练局部特征，或比较不同规模 DINOv2。

**第三，终点 latent MSE 过于简化。** 它忽略路径约束、碰撞、安全、动作平滑和语义不等价问题。目标图中不受控背景变化也会进入代价。论文成功环境都能用单一终态图较清楚地定义任务；更开放的长时任务需要层级子目标、可达性判别和不确定性估计。

**第四，样本与统计报告偏薄。** 四个成功率任务各 50 次、可变形任务 10 次，未给标准差或 bootstrap 区间。配置泛化尤其是 PushObj 的绝对成功率仍低。主张“泛化”成立于这些受控变化，但证据不足以支持跨域泛化。

**第五，表征稳定不等于动力学可靠。** 冻结 encoder 解决了 collapse，却也让 transition 的 epistemic uncertainty 无处表达。规划器没有 ensemble、置信区间或 OOD 惩罚，可能把低 latent loss 的模型幻觉当成可执行计划。

我的总体判断是：DINO-WM 最重要的贡献是一个清晰、强有力的 engineering baseline——**先固定一个空间化的 foundation representation，再把研究资源集中在动力学与规划上。** 它不是通用世界模型的终局，因为感知、动力学与任务几何之间仍有断层；但它建立了一个很难忽视的基线：任何更复杂的端到端 JEPA/world model，都应证明自己不仅能训练，而且确实超过“冻结 DINOv2 + 简单 predictor + MPC”。

---

## 8. 复现成本与实践建议

最低成本复现可从 PointMaze 开始：官方已有数据和 checkpoint，环境是 MuJoCo，训练命令与 CEM 配置齐全。PushT 能检验接触与 causal mask，是更有价值的第二阶段。Rope/Granular 依赖 PyFleX、Docker、NVIDIA Docker 与特定编译链，数据和模拟耗时都高，不适合作为首次复现入口。

资源方面，predictor 约 19M，训练 batch 32，不算大；真正成本来自离线轨迹生成、DINOv2 特征计算和 CEM 大量 rollout。论文只披露 A6000 推理测试，没有训练 GPU 数、墙钟时间和能耗，因此无法可靠给出完整预算。实践时建议预先缓存冻结 DINO features；分别验证 one-step latent MSE、open-loop drift、action sensitivity，再上 planner；同时加入 ensemble variance 或 rollout disagreement，避免 CEM 利用模型漏洞。

复现验收不应只看 LPIPS。至少应同时报告：同分布 one-step / multi-step 误差、规划成功率、不同数据规模曲线、无 mask 对照、DINO CLS 对照、CEM 与 MPC 对照，以及随机种子方差。若迁移到真实机器人，还应加入相机扰动、动作延迟、domain shift、碰撞约束和规划耗时。

---

## 9. 个人启发与后续问题

1. **世界模型的“状态坐标系”可能比 predictor 花样更重要。** DINO-WM 的 19M ViT 并不新奇，真正的跃迁来自空间 patch prior。
2. **anti-collapse 可以通过系统边界设计解决。** 冻结 encoder 是最彻底但最保守的方案；LeJEPA/SIGReg 则试图保留端到端适应能力。二者代表稳定性与可塑性的两端。
3. **规划 benchmark 应把模型误差与表征误差拆开。** 可用 oracle state dynamics + learned encoder、oracle encoder + learned dynamics 做二维消融，定位瓶颈究竟在哪。
4. **下一步需要不确定性与层级规划。** 单终点 MSE + CEM 适合短时 goal reaching；长时任务需要可达子目标、动作抽象和 OOD 惩罚。
5. **无动作视频仍是关键缺口。** 若能联合 latent action discovery 或 inverse dynamics，再保留 DINO-WM 的固定空间先验，才可能扩展到互联网视频规模。

---

## 10. 材料来源

本地官方 PDF、四张 arXiv 源图及逐项溯源见：

- [sources.md](assets/DINO-WM%20-%20预训练视觉特征上的零样本规划/sources.md)
- [官方 PDF](assets/DINO-WM%20-%20预训练视觉特征上的零样本规划/sources/paper.pdf)
- [arXiv](https://arxiv.org/abs/2411.04983)
- [ICML 2025 / PMLR](https://proceedings.mlr.press/v267/zhou25t.html)
- [项目页](http://dino-wm.github.io)
- [官方代码](https://github.com/gaoyuezhou/dino_wm)
