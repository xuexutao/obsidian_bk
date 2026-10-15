---
type: paper-note
status: complete
domain: World-Models
paper: "Solaris: Building a Multiplayer Video World Model in Minecraft"
year: 2026
arxiv: "2602.22208"
doi: null
source: https://arxiv.org/abs/2602.22208
project: https://solaris-wm.github.io/
code: https://github.com/solaris-wm/solaris
tags:
  - World-Models
  - Multi-Agent
  - Minecraft
  - Self-Forcing
created: 2026-09-04
updated: 2026-09-04
---

# Solaris：多人 Minecraft 视频世界模型

## 基本信息

- **作者**：Georgy Savva、Oscar Michel、Daohan Lu、Suppakit Waiwitlikhit、Timothy Meehan、Dhairya Mishra、Srivats Poddar、Jack Lu、Saining Xie
- **机构**：New York University
- **版本**：arXiv v2
- **核心资源**：[论文](https://arxiv.org/abs/2602.22208) · [项目页](https://solaris-wm.github.io/) · [模型代码](https://github.com/solaris-wm/solaris) · [数据引擎](https://github.com/solaris-wm/solaris-engine) · [数据](https://huggingface.co/collections/nyu-visionx/solaris-data) · [模型](https://huggingface.co/collections/nyu-visionx/solaris-models)

## 一句话结论

Solaris 把单玩家 Matrix-Game 2.0 改造成联合生成两名玩家第一人称视频的多人 DiT，并用 12.64M 同步帧、共享跨玩家注意力、四阶段训练和 Checkpointed Self Forcing 改善跨视角一致与长序列稳定；它首次把“多人世界状态是否在不同视角同步”变成系统性任务，但建造成功率、长期记忆和两人以上扩展仍远未解决。

---

## 1. 研究背景与问题定义

### 1.1 从单视角预测到共享世界

单玩家视频世界模型只需让一个观察流在时间上自洽。多人模型还要满足**横向一致性**：玩家 1 放置方块，玩家 2 的画面也必须出现；玩家 1 离开视野后，玩家 2 对其持续观察所提供的信息应能帮助玩家 1 再次定位；两人看向同一区域时应看到同一地形。

![](assets/Solaris%20-%20多人Minecraft视频世界模型/figures/01-teaser.png)

> **Figure 1**：输入两名玩家各自的起始帧和细粒度动作序列，模型联合生成两路第一人称视频；左右第三人称起止状态只用于可视化，不输入模型。  
> 来源：论文 Figure 1，第 3 页，https://arxiv.org/pdf/2602.22208

这使“世界模型”一词更接近其本义：输出不应只是每个摄像机各自好看，而应对应同一个潜在世界。Minecraft 特别适合测试遮挡、3D 方位、对象持久性、建造/破坏和随机天气。

### 1.2 核心贡献

1. 开源 **SolarisEngine**：协调 Mineflayer 控制器、官方客户端相机、服务器插件与 Docker worker，采集同步多人画面和动作。
2. 采集 9,240 episodes、每位玩家 6.32M 帧、合计 **12.64M 帧**的双玩家数据。
3. 在 Matrix-Game 2.0 DiT 内加入 shared multiplayer self-attention 与 player ID embedding，保留每玩家独立动作和首帧条件。
4. 设计四阶段训练：单玩家双向 → 多人双向 → 多人因果 → Self Forcing。
5. 提出 **Checkpointed Self Forcing**，把滑动窗口反传显存从 $O(L_tL_s)$ 降为 $O(L_t)$，并允许 KV 表示反传。
6. 建立 Movement、Grounding、Memory、Building、Consistency 五类多人评测。

---

## 2. 任务定义与输入输出

### 2.1 联合状态

传统单玩家 latent frame 为 $x_{SP}\in\mathbb{R}^{H\times W\times C}$。Solaris 增加玩家维，联合状态为

$$
x_t=\{x_t^1,\ldots,x_t^P\},\qquad a_t=\{a_t^1,\ldots,a_t^P\}.
$$

批序列张量分别为

$$
x\in\mathbb{R}^{B\times P\times T\times H\times W\times C},\qquad
a\in\mathbb{R}^{B\times P\times T\times D}.
$$

模型因果分解为

$$
p_\theta(x)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t},a_{<t}).
$$

当前实验固定 $P=2$。每位玩家输入第一帧、历史视觉和自己的 Minecraft 动作，输出两路未来第一人称视频。第三人称状态、地图坐标和游戏引擎内部状态均不输入模型。

### 2.2 动作空间

在 Matrix-Game 2.0 的 WASD + camera 基础上扩展到 23 维语义动作：前后左右、jump、sprint、sneak、yaw/pitch、attack、use、mount/dismount、place block/entity、mine、hotbar 1–9。SolarisEngine 不记录原始鼠标事件，只记录相对 yaw/pitch；也不支持打开/关闭背包，因此无法采集背包 GUI 与 crafting。

### 2.3 Flow Matching 目标

前向插值为

$$
x_\sigma=(1-\sigma)x+\sigma\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
$$

训练向量场：

$$
\mathcal L_\theta=\mathbb E_{x,a,\sigma,\epsilon}\left[\left\|v_\theta(x_\sigma,\sigma,a)-(\epsilon-x)\right\|_2^2\right].
$$

双向模型对所有玩家和帧共享一个 $\sigma\sim U(0,1)$，联合去噪；因果模型采用 Diffusion Forcing，为每个 $(p,t)$ 独立采样 $\sigma_{p,t}$。后者让不同帧处于不同生成进度，从训练目标中自然产生自回归能力。

---

## 3. 核心方法

### 3.1 SolarisEngine：先解决多人数据

![](assets/Solaris%20-%20多人Minecraft视频世界模型/figures/02-solaris-engine.png)

> **Figure 2**：每个逻辑玩家由 Mineflayer Controller 与官方 Minecraft Java Camera 两个进程组成；服务器插件同步状态和动画，episode code 调用 movement/mining/combat/building 等技能并记录低层动作。  
> 来源：论文 Figure 2，第 5 页，https://arxiv.org/pdf/2602.22208

Mineflayer 有高级控制但无图形渲染，MineRL/MineDojo 有图形却难以零成本产生有意义的多人协作。SolarisEngine 将二者拆开：Controller bot 执行高层技能并记录动作；Camera bot 在无头官方客户端用 GPU 渲染；服务端插件实时镜像位置、朝向和动画；后处理按时间戳对齐到 20 FPS。

Controller、Camera 和 server 都封装为 Docker 容器，Docker Compose 形成 worker，Python 编排多个 worker。每个 episode 随机传送到新地点。若 bot 卡死、报错或任务失败，系统通知所有组件并整体中止该 episode，再从新状态继续。这个机制提高吞吐，但也可能让最终数据偏向“容易成功”的情形。

CPU headless rendering 在森林等复杂场景会出现 20 FPS 下连续 4 个重复帧；作者改用 GPU 渲染和 NVENC，才获得稳定视频。由此可见数据成本不只是 bot 逻辑，还包含游戏客户端 GPU 渲染。

### 3.2 多人 DiT

![](assets/Solaris%20-%20多人Minecraft视频世界模型/figures/03-multiplayer-dit.png)

> **Figure 5**：不同玩家的视觉 token 在序列维拼接，加入 player ID embedding 后进入共享 self-attention；动作模块、首帧 cross-attention 与 FFN 按玩家独立执行并共享权重。  
> 来源：论文 Figure 5，第 8 页，https://arxiv.org/pdf/2602.22208

核心修改非常克制。将玩家维折入序列，令 shared self-attention 成为跨玩家信息通道；每位玩家独立应用 3D RoPE，player ID embedding 区分视角来源。动作模块按玩家独立运行，形式上通过 `B P T D -> (B P) T D` 折叠 batch；首帧 cross-attention 也独立。

相比 Multiverse 的 frame/channel concat，这一设计不会在输入层把多视角压进固定通道，而是允许每个 DiT block 中所有玩家 token 交换信息。代价是 self-attention token 数随玩家数增长，计算量理论上随总 token 数近似二次增长。论文虽称架构可扩展到任意玩家数，实际只训练两人，没有给出 $P>2$ 的显存或质量曲线。

### 3.3 四阶段训练

1. **Stage 1：Bidirectional Single-Player**。从 Matrix-Game 2.0 `base_distilled_model` 初始化，在 2,000+ 小时 VPT 上扩展完整动作空间。33 帧上下文、120K steps；VAE 始终冻结。
2. **Stage 2：Bidirectional Multiplayer**。加入多人架构，在 Duet 数据上做全序列扩散 120K steps；最终 checkpoint 是 Self Forcing 的 teacher/critic。
3. **Stage 3：Causal Multiplayer**。从 Stage 2 的 60K 中间 checkpoint 分支，使用因果 mask、Diffusion Forcing 和 6 latent（24 real frames）滑动窗口训练 60K steps。
4. **Stage 4：Self Forcing**。因果 student 在自己的 rollout 上训练，由长上下文双向 teacher 提供分布监督；最终变成 few-step 自回归模型。

与 Matrix-Game 2.0 的 ODE regression + Pre-DMD 不同，Solaris 发现简单 causal finetuning 已足够初始化，few-step 能力可以在 Self Forcing 时同时学得。

### 3.4 Self Forcing 的训练—推理对齐

Flow Matching 从 noisy latent 得到 clean estimate：

$$
\hat{x}_0=x_\sigma-\sigma v_\theta(x_\sigma,\sigma).
$$

Self Forcing 对每次迭代随机采样截断噪声级 $\sigma_{stop}$，自回归生成

$$
\hat{x}^{1:N}_0\sim p_\theta=\prod_{i=1}^{N}p_\theta(\hat{x}^i_0\mid\hat{x}^{<i}_0),
$$

再对整段 clean estimates 应用 DMD loss。与 teacher forcing 的关键差别是上下文也由模型自己生成，因此模型直接学习如何在带误差的历史上继续。

### 3.5 Checkpointed Self Forcing

![](assets/Solaris%20-%20多人Minecraft视频世界模型/figures/04-checkpointed-self-forcing.png)

> **Figure 7**：先无梯度滚动并缓存 clean/noisy states，再用自定义 mask 单次并行重算最后去噪步；显存随层数增长显著慢于 naive Self Forcing，并能对 KV 表示反向传播。  
> 来源：论文 Figure 7，第 11 页，https://arxiv.org/pdf/2602.22208

若 student 窗口为 $L_s$、teacher 总长度为 $L_t$，naive 滚动会保留大量重叠窗口，内存为 $O(L_tL_s)$。Solaris 分两步：

1. **no-grad rollout**：顺序生成，缓存 clean estimates $\hat{x}^{1:N}_0$ 与相应 noisy transition states $x^{1:N}_\sigma$；
2. **parallel recomputation**：把 clean 和 noisy 序列拼接，使用 TeacherForcingMask 单次前向，精确模拟“noisy query 只看自身 noisy key 和窗口内过去 clean key；clean query 只看过去 clean key”。

这样内存降到 $O(L_t)$。节省出的显存允许去掉原始 Self Forcing 对 KV cache 的 stop-gradient，让梯度穿过 K/V 表示。实验显示 KV backprop 显著改善 FID，但会牺牲部分动作 VLM 分数，形成视觉质量与动作服从之间的权衡。

### 3.6 多人生成的一次完整前向与训练闭环

在单个时间步，两名玩家的像素先由冻结 3D VAE 编码。每一路 token 独立加入 3D RoPE 与 player ID；动作模块只消费本玩家动作，首帧 cross-attention 只消费本玩家初始图像。随后两路 token 在 shared self-attention 中相互可见，因此玩家 1 的“放置方块”造成的视觉变化，可以经玩家 1 token 传播到玩家 2 token。FFN 和其余条件模块按玩家独立执行，避免把两个视角粗暴混成一幅图。

双向 teacher 在完整多人片段上联合去噪，能利用未来和两路全局上下文，提供较强目标分布；因果 student 只能读取 6 个 latent 的历史窗口。Stage 3 先让 student 学会在这个信息约束下预测，Stage 4 再把它放到自己的 rollout 上，由 teacher/critic 对整段样本进行分布监督。Checkpointed Self Forcing 只重算需要反传的最后去噪步，既保留推理时的滑动依赖，又避免保存每个重叠窗口的全部计算图。

这套闭环中，跨玩家一致性依赖 shared attention，时间稳定性依赖 causal training + Self Forcing，动作服从依赖 per-player action module，视觉先验依赖 Matrix-Game 2.0/VPT 初始化。消融显示四者不能互相替代：没有单玩家预训练会出现人体复制和场景漂移；frame concat 无法稳定交换高层信息；没有 Self Forcing 长 rollout 退化；KV-BP 虽改善画质却会改变动作指标。

---

## 4. 数据集与实验设置

### 4.1 多人训练数据

总计 **9,240 episodes**、每位玩家 **6.32M 帧**、合计 **12.64M 帧**，20 FPS。多数 episode 为 128–512 帧，即 6.4–25.6 秒。Normal/Superflat 各半，覆盖 building、combat、movement、mining 四类、14 种 episode：合作建房/墙/塔/桥、PvP/PvE、追逐、轨道移动、探索、采矿、跟随放火把等。

随机传送会产生水下失败轨迹：6,000 个 Normal episodes 中有 340 个水下样本。作者利用 HUD 氧气泡图标的线性分类器过滤，报告 100% accuracy。相机动作分布仍明显偏向 Pathfinder 的 178°/s 最大转速，这是数据中的系统偏差。

### 4.2 Solaris Eval

评测只有 **7 个独立 held-out 真实 episodes**，但从五个维度发问：

- **Movement**：一人移动/旋转，另一人观察，判断屏幕位置变化；
- **Grounding**：一人转离再转回，利用另一人持续观察保持定位；
- **Memory**：两人都转离再转回，测试不可见期间的对象/环境记忆；
- **Building**：一人建造，另一人是否看到结构；
- **Consistency**：两人看同侧应见相同景物，看反侧应不同。

指标为 FID↓ 与 VLM-as-a-judge accuracy↑。VLM 对 ground-truth 的 episode accuracy 并非全 100%，Memory 为 92.71%，部分 Rotation/Grounding 为 96.88%，Consistency opposite 为 93.75%，说明评测本身存在判断噪声。每次 VLM eval 重复三次估标准差。

### 4.3 训练资源

Stage 1 使用 **v5p-128 TPU**；Stage 2–4 使用 **v5p-64 TPU**。学习率/步数：单玩家 $10^{-4}$/120K，多人双向 $10^{-4}$/120K，多人因果 $10^{-4}$/60K，Self Forcing generator $3\times10^{-6}$/240，critic $3\times10^{-7}$/1200。官方代码说明训练仅支持至少 95 GB 内存的 TPU v5p；80 GB GPU 会 OOM。GPU 推理每卡 batch 1，至少 48 GB。

---

## 5. 实验结果

### 5.1 定性长序列与失败案例

![](assets/Solaris%20-%20多人Minecraft视频世界模型/figures/05-qualitative-comparison.png)

> **Figure 9**：224 帧 PvP 对比。Frame concat 的第二视角严重退化、第一视角纹理变平；无单玩家预训练会复制玩家身体、生成错误弹窗并漂移到水下；完整 Solaris 保持更稳定的双视角地形和战斗。  
> 来源：论文 Figure 9，第 13 页，https://arxiv.org/pdf/2602.22208

完整模型还能同步背包计数、天气开始时刻、火把放置/破坏动画和复杂地形 PvP。但这些是选择性案例；作者明确承认，当玩家互相离开视野时共享上下文会丢失，两条轨迹逐渐分叉。

### 5.2 架构对比

![](assets/Solaris%20-%20多人Minecraft视频世界模型/figures/06-quantitative-and-ablation.png)

> **Table 2–3**：上表比较 frame concat、无单玩家预训练与完整 Solaris；下表消融因果初始化、Pre-DMD 和 KV backprop。  
> 来源：论文 Table 2–3，第 15 页，https://arxiv.org/pdf/2602.22208

| 方法 | Movement VLM/FID | Grounding VLM/FID | Memory VLM/FID | Building VLM/FID | Consistency VLM/FID |
|---|---:|---:|---:|---:|---:|
| Frame concat | **77.1** / 68.9 | 53.1 / 66.6 | **37.5** / 74.4 | 0.0 / 103.2 | 49.5 / 129.4 |
| Solaris w/o pretrain | 69.3 / 42.5 | 29.2 / 49.9 | 18.8 / 67.8 | 0.0 / 86.6 | 49.5 / 121.4 |
| **Solaris** | 68.2 / **38.5** | **62.5** / **38.0** | **37.5** / **55.1** | **20.8** / **83.6** | **71.4** / **99.4** |

Solaris 五类 FID 全部最低，并显著提高 Grounding、Building、Consistency。Movement VLM 却低于 frame concat，说明复杂多人结构并非所有动作跟随都更强；作者观察 concat 在 no-op 时会产生动作幻觉，VLM 指标未充分惩罚这种错误。

Building 的最佳 VLM 也只有 **20.8**，Memory 只有 **37.5**，绝对水平仍低。论文证明“多人建模比简单拼接更好”，没有证明“多人世界状态已可靠”。

### 5.3 Self-Forcing 消融

- **ODE Reg + Pre-DMD + KV-BP**：Movement 23.4、Grounding 3.1、Memory 0.0，整体很差。
- **Causal FT + Pre-DMD + KV-BP**：仍只有 Movement 21.4、Grounding 3.1，表明 Self Forcing 前的 few-step distillation 无益。
- **Causal FT，无 Pre-DMD、无 KV-BP**：动作 VLM 最强，Movement **78.6**、Grounding **72.9**、Memory **49.0**，但 FID 较差。
- **Causal FT，无 Pre-DMD、有 KV-BP**：即最终 Solaris。FID 从 60.3/55.2/63.8/87.4/105.1 改善为 **38.5/38.0/55.1/83.6/99.4**；但前三项 VLM 降至 68.2/62.5/37.5。

最重要的结论不是“KV-BP 总是更好”，而是它把模型推向更真实、更跨视角一致的图像分布，同时削弱部分动作判别。世界模型训练需要显式管理这一 Pareto trade-off。

### 5.4 Benchmark 与结论强度

五类任务比单纯 FVD 更接近多人世界建模，但只有 7 个 held-out episodes，且依赖 VLM 判分。没有超过两人、跨服务器版本、自由人类多人游戏或分钟级回环。实验足以支持架构和训练的相对优势，不足以支持“通用多人模拟器”。

### 5.5 五类任务揭示的能力层级

Movement 主要检查局部动作—像素映射，因而简单 frame concat 也能取得较高分；Grounding 要求一个视角借助另一个视角保持相对位置，开始检验跨玩家信息传递；Memory 同时让两人失去对方，必须依赖内部历史，因此所有方法都低；Building 要把一个玩家造成的持久环境变化同步到另一视角，是最接近“共享状态”的任务，也恰好是绝对分数最低的任务；Consistency 则检查共同可见区域是否来自同一世界。

这种由局部控制到共享持久状态的难度梯度，比把所有问题压成 FVD 更有解释力。Solaris 在 FID 上全面领先，却在 Movement VLM 落后，说明生成质量、动作服从和世界一致不是同一个轴。未来报告结果应给出 Pareto 前沿，而非把多个指标平均成单一总分。

从统计可靠性看，VLM 标准差在 Consistency 上可达 7.1，7 个 episode 也不足以覆盖地形、遮挡和任务组合。更稳妥的扩展应增加确定性游戏状态 oracle：直接比较玩家坐标、方块图、库存和视锥内对象，再用 VLM 评价感知质量。这样才能区分“视频看起来一致”和“底层世界确实一致”。

此外，Table 2 的所有方法在 Building 上都远低于其它任务，说明生成共享环境变化比同步相机运动更难；这正是未来多人世界模型应优先优化的瓶颈，而不是继续只提高单帧观感。

---

## 6. 与相关工作的关系

Solaris 直接继承 Matrix-Game 2.0 的单玩家可控视频 DiT、动作模块、滑动窗口和 distilled checkpoint。与 Multiverse 的 channel concat 相比，它在深层注意力中交换玩家 token；与原始 Self-Forcing 相比，它允许 teacher 上下文长于 student，并用并行 mask 重算节省显存；与并行工作的 RELIC 相比，它一次并行 forward 完成重算，而不是多次 rolling pass。

它也反过来修正了 Matrix-Game 2.0 的训练经验：无需 ODE regression 和预先 DMD，只需 causal Diffusion Forcing 初始化，再让 few-step 与稳定 rollout 在 Self Forcing 中共同学习。

---

## 7. 局限与批判性评价

### 7.1 作者明确指出的限制

1. 数据完全合成，视觉和动作分布与自然人类多人游戏有差距；
2. 没有 persistent memory，玩家离开彼此视野后共享上下文丢失；
3. 世界只由两张初始图像指定，没有底层持久状态或更广环境控制。

### 7.2 进一步判断

第一，“联合像素生成”仍不等于“共享状态建模”。如果两路视频偶然都看起来合理，却对应不同隐藏地图，当前指标未必能发现。第二，玩家数增加会放大 attention 成本与一致性约束，$P=2$ 的成功不能直接推出多人规模化。第三，自动终止失败 episode、Pathfinder 高速相机偏差和预定义任务模板会造成数据选择偏差。第四，Building 20.8 和 Memory 37.5 表明最关键的持久状态能力仍很弱。

---

## 8. 复现与实践建议

### 8.1 开源完整度

Solaris 的开放程度较高：JAX 模型代码、四阶段 runner、权重、训练/评测数据、VLM judge 和 SolarisEngine 均有官方入口。简单 GPU 推理命令使用 `src/inference.py`，默认读取 `pretrained/solaris.pt` 和评测数据。

### 8.2 硬件成本

GPU 推理要求每卡至少 **48 GB**；训练仅官方支持 **95 GB v5p TPU**，且 Stage 1/其余阶段分别用 v5p-128/v5p-64。代码没有 FSDP，模型和优化器状态在设备间复制，80 GB GPU 训练会 OOM。即使数据公开，完整复现仍是大型实验室级成本。

SolarisEngine 还要求 Minecraft Java、服务器插件、Mineflayer、Docker、GPU headless rendering 与 NVENC。要重建 12.64M 帧数据，必须承担多客户端并行渲染、任务失败重试和后处理存储。

### 8.3 推荐复现顺序

1. 先用公开权重在 48 GB GPU 上跑官方 eval；
2. 验证 VLM judge 对 ground truth 的准确率，避免把评测噪声当模型差异；
3. 只训练 Stage 4，复现 KV-BP 的 FID/VLM trade-off；
4. 再用小规模 Duet 子集比较 frame concat 与 shared attention；
5. 最后才尝试 SolarisEngine 数据采集与四阶段全训。

---

## 9. 个人启发与后续问题

Solaris 的价值在于把“多人一致”拆成可验证的具体能力，而不是笼统地看视频是否逼真。未来应建立更强的状态探针：坐标/对象/方块图一致性、跨视角重投影、遮挡后重现、多人因果干预和长时间回环。

架构上，shared attention 可以作为短期通信，但不应承担全部长期世界状态。更合理的下一步可能是：视频生成器之外维护持久 3D/对象 memory，玩家视角从共享 memory 渲染；Self Forcing 负责适应自身像素误差，显式 memory 负责跨时间和跨玩家守恒。另一个关键方向是把玩家数从 2 扩到可变 $P$，研究计算复杂度、身份交换和异步动作。

---

## 10. 材料来源

完整来源、图表追溯与资料缺口见：[sources.md](assets/Solaris%20-%20多人Minecraft视频世界模型/sources.md)。本地保存官方 PDF：`assets/Solaris - 多人Minecraft视频世界模型/sources/paper.pdf`。
