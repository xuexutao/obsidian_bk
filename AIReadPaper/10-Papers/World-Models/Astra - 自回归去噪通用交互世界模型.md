---
type: paper-note
status: complete
domain: World-Models
paper: "Astra: General Interactive World Model with Autoregressive Denoising"
year: 2026
venue: ICLR 2026
arxiv: "2512.08931"
source: https://arxiv.org/abs/2512.08931
project: https://eternalevan.github.io/Astra-project/
code: https://github.com/EternalEvan/Astra
tags:
  - World-Models
  - Video-Generation
  - Autoregressive-Denoising
  - Action-Control
  - Kling
created: 2026-10-16
updated: 2026-10-16
---

# Astra：自回归去噪通用交互世界模型

## 基本信息

- **正式题名**：Astra: General Interactive World Model with Autoregressive Denoising
- **作者**：Yixuan Zhu、Jiaqi Feng、Wenzhao Zheng、Yuan Gao、Xin Tao、Pengfei Wan、Jie Zhou、Jiwen Lu
- **机构**：清华大学、快手科技
- **发表**：ICLR 2026；arXiv v3
- **基础模型**：Wan2.1-1.3B
- **核心关键词**：自回归去噪、因果视频块、噪声增强历史、ACT-Adapter、Mixture of Action Experts（MoAE）、Action-Free Guidance（AFG）

## 一句话结论

Astra 的关键不是简单把动作接到视频扩散模型上，而是正面处理“长历史越有利于画面连续、却越让模型忽略新动作”的**视觉惯性**：它用 noise-augmented history 降低历史帧的支配力，用 ACT-Adapter 与 AFG 提升动作信号，用 MoAE 统一相机、机器人和键鼠等异构动作，从而把 Wan2.1 改造成可流式 rollout 的通用交互世界模型。

---

## 1. 研究背景与问题定义

### 1.1 从视频生成到交互世界模型

普通 I2V/T2V 模型生成的是一次性短片；交互世界模型则要连续接收动作，并在每个时刻生成与动作相符的未来。它同时受到三组约束：单块视频要有扩散模型的保真度，跨块要保持长期一致性，新动作还必须及时改变运动趋势。Astra 覆盖世界探索、自动驾驶、机器人操作、相机控制和多智能体交通等场景，动作既可能是连续位姿，也可能是离散键鼠命令。

![](assets/Astra%20-%20自回归去噪通用交互世界模型/figures/01-teaser.png)

> 图 1：Astra 在世界探索、机器人操作和自动驾驶中的动作条件视频预测。图中完整展示了不同动作接口，而非只在单一游戏域内训练控制器。来源：论文 Figure 1。

### 1.2 核心矛盾：一致性与响应性

增加历史上下文通常能提高主体、背景和运动连续性，但也让模型更容易从过去画面平滑外推，忽略当前动作。论文把这种现象称为 **visual inertia**。它不是简单的动作编码容量不足，而是条件竞争：高维视觉历史比低维动作信号更容易解释下一帧。

![](assets/Astra%20-%20自回归去噪通用交互世界模型/figures/04-visual-inertia.png)

> 附录 Figure C：上下文长度从约 1 增至 81 帧时，成像质量持续上升，而动作跟随从约 0.68 降至约 0.19。曲线是 noise-augmented history 的直接动机。来源：论文 Appendix Figure C。

### 1.3 贡献拆解

1. 以视频块为单位的 autoregressive denoising，使扩散模型能够流式生成长视频。
2. 以随机噪声“软遮罩”历史条件，打破干净长历史对动作的压制。
3. 用 ACT-Adapter 直接在每层 DiT 特征空间注入动作，并用 AFG 放大有/无动作预测差异。
4. 用 MoAE 稀疏路由异构动作，支持 camera pose、robot pose、keyboard/mouse 等统一训练。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

视频 latent 被切为 $N$ 个块 $\bm z^{1:N}$。第 $i$ 步输入包括历史视频块 $\bm z^{1:i-1}$、截至当前的动作 $\bm a^{1:i}$、可选文本 $\bm c$，并预测下一块：

$$p(\bm z^{1:N})=\prod_{i=1}^{N}p(\bm z^i\mid\bm z^{<i})$$

完整条件写作：

$$\mathcal C=\{\bm z^{1:i-1},\bm a^{1:i},\bm c\}$$

输出 $\bm z^i$ 被解码为视频，并追加到历史继续 rollout。任务默认动作与视频时间对齐；动作每 4 帧插值一次，以匹配 3D VAE 的时间压缩率。

### 2.2 Flow matching 目标

目标块的噪声插值为：

$$\bm z_t^i=(1-t)\bm z_0^i+t\bm\epsilon,\quad\bm\epsilon\sim\mathcal N(0,I)$$

模型拟合条件速度场：

$$\mathcal L(\theta)=\mathbb E_{i,t,\bm\epsilon}\left[\left\|\bm v_\theta(\bm z_t^i,t\mid\bm z^{<i})-\bm v^*(\bm z_t^i,t\mid\bm z^{<i})\right\|_2^2\right]$$

块间使用因果历史，块内仍保留扩散/flow 的多步去噪，因此它兼具自回归的可延展性和视频扩散的局部画质。

---

## 3. 核心方法

### 3.1 自回归去噪与 ACT-Adapter

![](assets/Astra%20-%20自回归去噪通用交互世界模型/figures/02-method-pipeline.png)

> 图 3：左侧是训练/推理的 chunk-wise autoregressive denoising，右侧是 action-aware flow transformer。火焰表示只训练 causal self-attention 与 ACT-Adapter，雪花表示冻结的线性层、cross-attention 和 FFN。来源：论文 Figure 3。

历史 latent 与当前噪声块沿时间维拼接。动作先经 encoder 对齐到视频特征空间，再在每个 Transformer block 中以逐元素相加注入；self-attention 后插入单线性层 ACT-Adapter，并采用单位初始化，避免训练初期破坏预训练主干。与“把动作当 cross-attention token”相比，这一设计把动作解释为视频 latent 的逐层特征偏移，更直接地改变时空运动。

主干为 30 个 flow transformer blocks。模型冻结大部分 Wan2.1 参数，只训练 self-attention、动作编码器、ACT-Adapter 和 MoAE，总可训练参数为 **366.8M**，明显小于 YUME 的约 14B 全量微调。

### 3.2 Noise-augmented history：噪声即软掩码

训练时，历史条件视频被独立随机噪声污染；目标块仍按正常 flow matching 加噪。历史噪声不删除 token，也不改变 denoiser 架构，而是降低历史细节的可复制性，迫使模型同时读取动作。推理时仍可使用干净历史帧。

这项设计的实质是重新平衡条件信噪比：若历史绝对可靠，最小损失策略往往是延续历史；当历史被适度扰动，动作成为解释未来变化所必需的信息。论文还保留第一帧，并压缩中间历史 token，以延长可用跨度。不过论文未披露历史噪声的精确分布、幅度和采样概率，这是复现中的最大缺口之一。

### 3.3 Mixture of Action Experts（MoAE）

![](assets/Astra%20-%20自回归去噪通用交互世界模型/figures/03-moae.png)

> 图 4：各动作模态先投影到共享空间并加入历史/当前二值 mask，router 计算 top-$k$ 分数，稀疏激活专家后形成统一动作 embedding。来源：论文 Figure 4。

模态专用投影器把不同输入映射到共享空间：

$$\widetilde{\bm a}^{i}=\mathcal R_m(\bm a_m^i),\quad m\in\{\mathrm{cam},\mathrm{rob},\mathrm{cmd}\}$$

router 输出门控分数 $\bm g^i$，专家聚合为：

$$\bm e^i=\sum_{k=1}^{K}\bm g_k^iE_k(\widetilde{\bm a}^{i})$$

其中 $E_k$ 为独立 MLP。动作表示还带有 history mask，用于区分历史动作与当前动作。正文称 top-$K$ 路由，附录又称每步只激活一个专家，论文未明确所有实验的实际 $K$、专家总数和负载均衡损失，因此 MoAE 的完整复现仍依赖代码。

### 3.4 Action-Free Guidance（AFG）

训练时随机丢弃动作，推理时组合有动作与无动作速度场：

$$\bm v_{\mathrm{guided}}=\bm v_\theta(\bm z_t,t,\emptyset)+s\left[\bm v_\theta(\bm z_t,t,\bm a)-\bm v_\theta(\bm z_t,t,\emptyset)\right]$$

论文使用 $s=3.0$。AFG 与 noisy history 的分工不同：前者在推理时放大动作差分，后者在训练时防止历史条件垄断预测。

### 3.5 推理流程与复杂度

每步从高斯噪声初始化 33 帧目标块，执行 50 步去噪，解码并追加到历史，再读取下一段动作。论文主评测生成 96 帧、20 FPS、$480\times832$ 视频；能力表给出的交互时域为 8–10 秒。多步去噪乘以多块 rollout，决定了其尚不能实时。

---

## 4. 数据集与实验设置

### 4.1 训练数据

| 数据集 | 动作 | 场景 | 使用规模 |
|---|---|---|---:|
| nuScenes | 7D camera | 自动驾驶 | 850 |
| Sekai | 12D camera | 步行/无人机 | 50K |
| SpatialVid | 7D camera + 键鼠 | 开放视频 | 200K |
| RT-1 / Open X-Embodiment | 7D robot pose | 机器人操作 | 9,978 |
| Multi-Cam Video | 12D camera | 人体多视角 | 表中 136K |

总计约 **397K 视频、360 小时**。Multi-Cam Video 主表写 136K，附录另称使用 10K，口径不一致，应在复现时以代码清单为准。

### 4.2 训练配置

8 张 80GB GPU，每卡 batch 1，AdamW，学习率 $10^{-5}$，训练 30 epochs，约 24 小时收敛。条件帧数从 1–128 随机采样，目标固定 33 帧。Astra-Bench 从 5 个数据集各取 20 个 held-out 样本，共 100 个。

### 4.3 指标与 baseline

baseline 为 Wan-2.1、MatrixGame、YUME，附录增加 NWM。Instruction Following 由 20 名用户判断动作方向与类型是否匹配；其余使用 VBench 的主体一致性、背景一致性、运动平滑、美学和成像质量。另用 MegaSaM 估计相机轨迹，报告 RotErr/TransErr。

---

## 5. 实验结果

### 5.1 主结果

| 方法 | 指令跟随 ↑ | 主体一致 ↑ | 背景一致 ↑ | 运动平滑 ↑ | 美学 ↑ | 成像质量 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Wan-2.1 | 0.061 | 0.854 | 0.903 | 0.958 | 0.489 | 0.691 |
| MatrixGame | 0.268 | 0.916 | 0.928 | 0.981 | 0.441 | **0.748** |
| YUME | 0.652 | 0.936 | 0.938 | 0.985 | 0.523 | 0.741 |
| **Astra** | **0.669** | **0.939** | **0.945** | **0.989** | **0.531** | 0.747 |

Astra 在六项中五项最佳；成像质量 0.747 略低于 MatrixGame 的 0.748。因此论文文字中的“all metrics”应谨慎理解。客观轨迹上，Astra 的 RotErr/TransErr 为 **1.23/4.86**，优于表中其他方法。

值得注意的是，Table 2 把 MatrixGame/YUME 的 instruction-following 写为 0.268/0.652，而 Appendix Table A 写为 0.652/0.268，存在对调。本文按主表记录，不替作者消解冲突。

### 5.2 消融：哪个设计真正起作用

| 变体 | 指令跟随 ↑ | 主体一致 ↑ | 背景一致 ↑ | 平滑 ↑ | 美学 ↑ | 成像 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| w/o AFG | 0.545 | 0.841 | 0.892 | 0.957 | 0.492 | 0.703 |
| w/o noise | 0.359 | 0.903 | 0.927 | 0.979 | 0.523 | 0.739 |
| cross-attn adapter | 0.642 | 0.926 | 0.903 | 0.948 | 0.512 | 0.694 |
| w/o MoAE | 0.651 | 0.930 | 0.941 | 0.975 | 0.520 | 0.727 |
| **完整模型** | **0.669** | **0.939** | **0.945** | **0.989** | **0.531** | **0.747** |

最关键的反事实是 **w/o noise：动作跟随从 0.669 暴跌到 0.359**。这说明 Astra 的主要瓶颈不是“动作有没有编码进去”，而是“历史视觉是否压过动作”。AFG 其次，去除后降至 0.545，并同步伤害画质。ACT-Adapter 相比 cross-attention 的优势不仅体现在动作跟随（0.669 vs 0.642），还体现在运动平滑和成像质量。MoAE 对单项指令分数提升较小，但承担统一异构动作与跨域训练的结构性作用；w/o MoAE 版本事实上只能训练 camera action。

### 5.3 泛化与失败边界

CityWalker 100 个未见场景中，Astra 六项指标均最佳，Instruction Following 为 0.641、Imaging Quality 为 0.695。论文还展示室内、Minecraft 和动画风格 OOD 定性结果，但没有 OOD 聚合指标。未展示动作冲突、MoAE 路由错误、物理违反或极长 rollout 崩溃的系统 failure cases。

---

## 6. 与 DecMem、GIM-World、AnchorWorld 的关系

Astra 解决的是**动作条件与视觉历史的竞争**，其记忆仍以历史视频 token 为主；DecMem 把问题推进到分钟级，直接设计 Sparse Global + Anchored Local 两类记忆以控制成本和注意力分散；GIM-World 则不保留大量显式历史，而把历史压缩成固定 token，并用几何教师约束记忆内容。AnchorWorld 的重点不同：它用混合第一/第三人称监督学习完整人体动作，并以带 pose 和 evolution prompt 的 anchor views 显式定制局部世界。四者分别回答“如何响应动作”“如何记很久”“记忆里如何保留几何”“如何定义可定制世界状态”。

---

## 7. 局限与批判性评价

### 7.1 作者明确局限

主要限制是推理效率：每个块需要多步去噪，长视频又要多次自回归，难以用于实时在线控制。作者提出 diffusion distillation 与 student–teacher compression 作为后续方向。

### 7.2 证据边界

1. 交互时域仍为 8–10 秒，“长期”与 DecMem 的分钟级并非同一量级。
2. 人工动作跟随指标基于每数据集 20 个样本，规模有限。
3. noisy history 的噪声策略、AFG dropout、MoAE 专家数和路由正则未披露。
4. 两张表对 MatrixGame/YUME 的动作分数互相对调。
5. OOD 主要为定性证据，且没有实时 FPS。

### 7.3 个人判断

Astra 最有价值的发现是把“动作不灵”重新解释为**条件间的信息不平衡**。noise-as-mask 是便宜但强力的干预；它也提示后续长程世界模型不能只追求更长上下文，否则可能在控制性上倒退。MoAE 更像扩展数据和接口的工程骨架，其收益不应只由单一指令分数衡量。

---

## 8. 复现与实践建议

1. 先用 Wan2.1-1.3B、单一 camera-action 数据复现 chunk-wise causal rollout，再加入多域数据，避免同时调试动作接口与长程稳定性。
2. 首先复现 w/o noise 与 noisy history 的动作跟随差异；这是验证论文核心主张的最低成本实验。
3. 单独记录历史长度—画质—动作响应三条曲线，不要只看短视频 VBench。
4. ACT-Adapter 应采用单位初始化，并严格核对冻结/训练参数列表；目标可训练参数约 366.8M。
5. 复现 MoAE 前需从代码确认专家数、top-$K$、router loss 和每模态投影器；仅凭论文无法完整还原。
6. 推理采用 50 steps、AFG $s=3.0$；若追求交互，优先测试少步 flow distillation，而不是先扩大模型。

**资源估计**：论文训练使用 8×80GB GPU、约 24 小时；官方仓库存在，但项目页仍将完整训练代码、数据处理和部分场景流程列为后续发布项，实际一键复现能力需要以仓库当前内容为准。

---

## 9. 个人启发与后续问题

- noisy history 能否改为动作自适应噪声：动作突变时增噪、匀速时减噪？
- MoAE 是否应按动作模态路由，还是按“动作对视觉的因果影响类型”路由？
- 当历史被压缩成 DecMem/GIM-World 式记忆后，视觉惯性是否仍存在，如何定义统一的响应性指标？
- 若引入 AnchorWorld 的锚点状态，动作跟随与显式世界约束冲突时应由谁优先？

---

## 10. 材料来源

- 官方论文：https://arxiv.org/abs/2512.08931
- 项目页：https://eternalevan.github.io/Astra-project/
- 代码：https://github.com/EternalEvan/Astra
- 本地 PDF 与逐图溯源：`assets/Astra - 自回归去噪通用交互世界模型/sources.md`
