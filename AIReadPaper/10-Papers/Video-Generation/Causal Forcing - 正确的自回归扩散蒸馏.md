---
type: paper-note
status: done
domain: Video-Generation
paper: "Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation"
year: 2026
arxiv: 2602.02214
doi: null
source: https://arxiv.org/abs/2602.02214
project: https://thu-ml.github.io/CausalForcing.github.io/
code: https://github.com/thu-ml/Causal-Forcing
tags:
  - Video-Generation
  - Causal-Distillation
  - ODE-Distillation
  - DMD
created: 2026-09-04
updated: 2026-09-04
---

# Causal Forcing：正确的自回归扩散蒸馏

## 基本信息

- **作者**：Hongzhou Zhu、Min Zhao、Guande He、Hang Su、Chongxuan Li、Jun Zhu
- **发表**：ICML 2026，PMLR 306
- **arXiv**：2602.02214v5，2026-06-01
- **项目页**：https://thu-ml.github.io/CausalForcing.github.io/
- **官方代码**：https://github.com/thu-ml/Causal-Forcing
- **基础模型**：Wan2.1-T2V-1.3B；论文同时训练 chunk-wise 与 frame-wise 因果模型，主对比采用每块 3 个 latent frames 的 chunk-wise 设置。

## 一句话结论

Causal Forcing 指出 CausVid / Self Forcing 的 ODE 初始化把**双向教师的整段视频流映射**回归给**只能看过去的 AR 学生**，违反 frame-level injectivity，MSE 最优解因此退化为条件均值并产生模糊；它改用 teacher-forced AR diffusion 作为 ODE 教师，再沿用 Self Forcing 的自展开 DMD，在相同 17 FPS、0.69 秒延迟和相同后训练预算下，把 Dynamic Degree、VisionReward、Instruction Following 分别从 Self Forcing 的 57/5.820/48 提升到 68/6.326/56。

---

## 1. 研究背景与问题定义

### 1.1 “步数鸿沟”之外还有“架构鸿沟”

few-step 视频蒸馏通常关注把 50 步采样压缩到 4 步，即 sampling-step gap。但从双向视频 DiT 到自回归生成器还发生了更深的变化：教师在去噪第 $i$ 帧时看到前后所有帧，学生只能依赖 $x_{<i}$。CausVid 和 Self Forcing 都先用双向教师的 PF-ODE 轨迹回归初始化 AR 学生，再做 DMD；它们经验上有效，却没有回答这组训练对是否是一个确定函数。

![](assets/Causal%20Forcing%20-%20正确的自回归扩散蒸馏/figures/01-existing-gap.png)

> 图 1：同一双向基础模型上，标准 DMD 的双向学生在动态程度和指令遵循上明显高于 Self Forcing 的 AR 学生，说明差距不只来自采样步数，而与注意力架构改变有关。  
> 来源：论文 Figure 1，第 2 页，https://arxiv.org/abs/2602.02214

### 1.2 论文要回答的问题

论文把问题精确化为：当目标学生是逐帧/逐块生成的 AR 模型时，什么样的教师 ODE 轨迹才能给出**可学习、单值且与推理条件一致**的初始化？作者进一步问，Diffusion Forcing 是否真的是训练多步 AR diffusion teacher 的最佳选择。

### 1.3 核心贡献

1. 把 AR 视频蒸馏中的结构损失定位到 ODE 初始化阶段，而非后续 DMD；控制实验显示，仅靠 DMD 无法修复坏初始化。
2. 提出 **frame-level injectivity**：对 AR 学生而言，同一个当前帧噪声状态在给定因果前缀后必须对应唯一干净帧。
3. 证明双向教师到 AR 学生的 ODE 回归一般不满足该条件，MSE 解会成为条件期望而非教师 flow map。
4. 提出三阶段 **Causal Forcing**：Teacher Forcing 训练 AR teacher → causal ODE distillation → 与 Self Forcing 相同的 self-rollout DMD。
5. 给出 causal consistency distillation 扩展，说明“AR teacher 监督 AR student”的原则不依赖特定 DMD 实现。

---

## 2. 任务定义与输入输出

### 2.1 自回归视频扩散

视频 $x_0^{1:N}$ 按时间分解为

$$
p_\theta(x_0^{1:N})=\prod_{i=1}^{N}p_\theta(x_0^i\mid x_0^{<i}),
$$

每个条件分布内部由 diffusion / flow matching 建模。训练输入包括文本提示、干净历史帧或历史 latent chunk、当前高斯噪声 $x_t^i$ 与时间 $t$；输出是当前帧/块的速度或干净样本预测。推理时按因果顺序生成，每块 4 步去噪并复用 KV cache。

前向过程采用

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
$$

论文的 flow matching 设定为 $\alpha_t=1-t,\sigma_t=t,T=1$，目标速度 $v_t=\epsilon-x_0$，采样解 PF-ODE：

$$
dx_t=v_\theta(x_t,t)dt,\qquad x_T\sim\mathcal N(0,I),\quad t:T\rightarrow0.
$$

### 2.2 输出与部署指标

主模型输出 832×480、81 帧视频；chunk-wise 版本每次生成 3 个 latent frames。H100 上吞吐量 17.0 FPS、首块延迟 0.69 秒，与论文复现的 Self Forcing 相同。官方代码另提供 frame-wise / chunk-wise checkpoint，并宣称经工程优化可在单张 RTX 4090 流式运行；后者属于代码与项目页能力，论文表格的速度口径仍是单 H100。

---

## 3. 核心方法

### 3.1 为什么双向 ODE 教师对 AR 学生不是合法回归目标

标准 ODE distillation 学习 flow map $G_\theta:(x_t,t)\mapsto x_0$。双向教师的 PF-ODE 对**完整视频**是可逆的：给定 $x_t^{1:N}$，存在唯一 $x_0^{1:N}=\phi^{\mathrm{Bi}}(x_t^{1:N},t)$。但 AR 学生在生成第 $i$ 帧时看不到 $x_t^{>i}$，它实际要学的是

$$
G_\theta(x_t^i,x_t^{<i},t)\rightarrow x_0^i.
$$

如果固定 $x_t^i$ 与可见前缀，却改变未来噪声帧 $x_t^{>i}$，双向教师可能给出不同 $x_0^i$。于是学生输入相同、监督目标不同，映射不是单值函数。

![](assets/Causal%20Forcing%20-%20正确的自回归扩散蒸馏/figures/02-injectivity-principle.png)

> 图 3：左侧是标准双向教师到双向学生的 video-level injectivity；中间是 AR teacher 到 AR student 的 frame-level injectivity；右侧是 Self Forcing 的双向教师到 AR student，同一当前噪声帧可因不可见未来不同而对应不同干净帧。  
> 来源：论文 Figure 3，第 4 页，https://arxiv.org/abs/2602.02214

论文将所需条件写成：若两个样本在第 $i$ 帧的噪声状态相同，则 AR flow map 的第 $i$ 帧输出也必须相同，

$$
x_t^i=y_t^i\Rightarrow
\phi^{\mathrm{AR}}(x_t^i,t)=\phi^{\mathrm{AR}}(y_t^i,t).
$$

双向 teacher 通常违反这一条件。此时以 MSE 训练的最优解为

$$
G_\theta^*(x_t^i,x_t^{<i},t)
=\mathbb E[x_0^i\mid x_t^i,x_t^{<i},t],
$$

它平均掉不可见未来导致的多种合法结果，视觉上表现为模糊、运动变弱与时间不一致。论文的 Proposition 3.3 进一步指出该条件均值分布一般不等于数据分布。

### 3.2 DMD 为什么救不回错误初始化

DMD 优化生成分布与真实分布的 reverse KL：

$$
\nabla_\theta\mathbb E_t D_{\mathrm{KL}}(p_{\theta,t}\|p_{\mathrm{data},t})
=-\mathbb E\left[
(s_{\mathrm{real}}(\tilde x_t,t)-s_{\mathrm{fake}}(\tilde x_t,t))
\frac{\partial\tilde x}{\partial\theta}
\right].
$$

作者先用标准 DMD 得到 few-step 双向模型，再只切换学生为因果注意力，从而消除采样步数差，仅保留结构差。结果仍显著落后于双向 DMD，说明后续 score distillation 的有限训练不能跨越一个已经坍缩的初始化盆地。换言之，Self Forcing 的 self-rollout 对 exposure bias 有效，却不能自动补回 ODE 回归阶段丢失的未来信息。

### 3.3 Stage 1：用 Teacher Forcing 训练 AR diffusion teacher

Causal Forcing 先从 Wan2.1-T2V-1.3B 微调一个多步 AR diffusion teacher。作者比较两种训练分布：

- **Teacher Forcing**：当前 noisy frame 条件化于干净真实前缀 $x_0^{<i}$，直接学习 $p_{\mathrm{data}}(x_0^i\mid x_0^{<i})$；
- **Diffusion Forcing**：当前 frame 条件化于独立加噪前缀 $x_t^{<i}$，学习 $p_{\mathrm{DF}}(x_0^i\mid x_t^{<i})$。

论文 Proposition 3.4 认为，实际 AR 推理条件是已去噪的历史，因此 DF 的重噪历史与推理条件存在 KL 大于零的分布差：

$$
\mathbb E_{y\sim p_{\mathrm{data}}(x_0^{<i})}
D_{\mathrm{KL}}
\left(p_{\mathrm{DF}}(x_0^i\mid y)\|p_{\mathrm{data}}(x_0^i\mid y)\right)>0.
$$

这是对原始 Diffusion Forcing 命题的一个重要限定：DF 擅长让模型适应不确定历史，但若目标是构造一个以**干净真实前缀**为条件的高质量多步 AR teacher，Teacher Forcing 更贴近所需条件分布。实验中 TF 的 VisionReward 为 3.343，DF 为 1.583；DF 的 Dynamic Degree 更高（60 vs 50）主要来自画面坍缩造成的病理运动，而非更好动态。

### 3.4 Stage 2：Causal ODE Distillation

作者从 AR teacher 采样 PF-ODE 轨迹。先从真实数据取干净历史 $x_{\mathrm{gt}}^{<i}$，再让 AR teacher 在该历史条件下从 $x_T^i\sim\mathcal N(0,I)$ 解 ODE，保存中间状态 $x_t^i$ 与终点 $x_0^i$。学生目标为

$$
\theta^*=\arg\min_\theta
\mathbb E_{x_{\mathrm{gt}}^{<i},t,i,x_t^i}
\left\|G_\theta(x_t^i,x_{\mathrm{gt}}^{<i},t)-x_0^i\right\|_2^2.
$$

因为教师和学生都以同一干净因果前缀为条件，当前帧 PF-ODE flow map 是单值的，学生不再需要对不可见未来求平均。这里的“causal”指**轨迹来源与条件信息结构一致**，不只是把 attention mask 改成三角形。

### 3.5 Stage 3：沿用 Self Forcing 的 Asymmetric DMD

获得正确的 causal ODE 初始化后，论文不再改造 Self Forcing 的后半段：学生以自己生成的 prefix 做自回归 self-rollout，真实 score 仍由强双向模型提供，fake score 在线拟合学生分布，再用 DMD 更新生成器。这一组合分别解决两个问题：

1. causal ODE 初始化处理 **architectural gap**；
2. self-rollout DMD 处理 **train–inference gap** 并把质量拉回双向教师分布。

因此论文标题中的 “Done Right” 并不是否定非对称 DMD，而是把**前向 KL / MSE 风格的初始化教师**与**反向 KL 风格的最终分布教师**分开选择：前者必须因果，后者仍可双向。

### 3.6 Causal Consistency Distillation 扩展

论文还用 AR teacher 构造一致性目标：

$$
\min_\theta\mathbb E\left[
w(t)d\left(
G_\theta(x_t^i,x_{\mathrm{gt}}^{<i},t),
G_{\theta^-}(\hat x_{t-\Delta t}^i,x_{\mathrm{gt}}^{<i},t-\Delta t)
\right)
\right].
$$

其中 $\hat x_{t-\Delta t}^i$ 由 AR teacher 沿 ODE 前进一步获得。初步实验中 causal CD 明显优于 asymmetric CD，但仍弱于 score distillation；作者将更强 CD 初始化留作未来工作。

---

## 4. 数据集与实验设置

### 4.1 训练流程

- **基础模型**：Wan2.1-T2V-1.3B，输出 81 帧、832×480 视频。
- **Stage 1**：用基础双向模型合成的 3K 视频数据 $D_{\mathrm{Bi}}$，Teacher Forcing 微调 AR diffusion 2K steps；同时保存双向 ODE 中间状态供 Self Forcing 基线使用。
- **Stage 2**：AR teacher 生成 3K causal ODE trajectories $D_{\mathrm{Causal}}$，学生回归 1K steps。
- **Stage 3**：VidProM prompt 上按 Self Forcing 协议训练 asymmetric DMD 750 steps。
- **结构**：chunk-wise，每块 3 个 latent frames；消融另报告 frame-wise。
- **公平性**：Causal Forcing 与 Self Forcing 都执行合计 3K steps 初始化，再用相同 DMD 预算。

### 4.2 评测与 baseline

主要使用 VBench 的 Total、Quality、Semantic，并在 100 个富运动复杂动作 prompt 上评估 Dynamic Degree、VisionReward 与 Instruction Following。另有 10 名参与者、10 个 prompt 的全方法排序，数值越低越好。速度在单 H100 上测量。

基线分三类：双向扩散 LTX-1.9B、Wan2.1-1.3B；原生 AR diffusion NOVA、Pyramid Flow、SkyReels-V2、MAGI-1；蒸馏 AR 模型 CausVid 与 Self Forcing。

---

## 5. 实验结果

### 5.1 主要定量结果

![](assets/Causal%20Forcing%20-%20正确的自回归扩散蒸馏/figures/04-main-results.png)

> 表 1：Causal Forcing 在保持 Self Forcing 相同吞吐量与延迟的同时，在 VBench、Dynamic Degree、VisionReward、Instruction Following 和用户排序上均取得最佳结果。  
> 来源：论文 Table 1，第 8 页，https://arxiv.org/abs/2602.02214

| 模型 | FPS ↑ | 延迟 s ↓ | VBench Total ↑ | Quality ↑ | Semantic ↑ | Dynamic ↑ | VisionReward ↑ | 指令遵循 ↑ | 用户排序 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan2.1-1.3B | 0.78 | 103 | 83.37 | 84.30 | 79.65 | 61 | 5.275 | 42 | 2.29 |
| CausVid | 17.0 | 0.69 | 81.33 | 83.98 | 70.72 | 62 | 5.741 | 12 | 4.27 |
| Self Forcing | 17.0 | 0.69 | 83.74 | 84.48 | 80.77 | 57 | 5.820 | 48 | 2.87 |
| **Causal Forcing** | **17.0** | **0.69** | **84.04** | **84.59** | **81.84** | **68** | **6.326** | **56** | **1.64** |

相对 Self Forcing，Dynamic Degree 提升 $19.3\%$、VisionReward 提升 $8.7\%$、指令遵循提升 $16.7\%$。VBench Total 只增加 0.30，说明优势主要集中在高运动 prompt 和偏好模型，而不是所有常规 VBench 维度都大幅跃升。与 Wan2.1 相比，吞吐量从 0.78 到 17 FPS，论文称提高 2079%，同时 Total 从 83.37 到 84.04。

### 5.2 定性结果与失败模式

![](assets/Causal%20Forcing%20-%20正确的自回归扩散蒸馏/figures/03-qualitative-comparison.png)

> 图 6：相同 prompt 下，CausVid 容易过饱和，Self Forcing 动态较弱；Causal Forcing 的主体运动幅度更大，同时保持人物、动物和背景细节。图像只能展示稀疏帧，连续运动判断仍应结合项目页视频。  
> 来源：论文 Figure 6，第 7 页，https://arxiv.org/abs/2602.02214

论文展示的典型失败链条是：双向 ODE 初始化后的 AR 模型产生模糊；Self Forcing DMD 能锐化，但动态与指令遵循仍受初始化限制；causal ODE 初始化使后续 DMD 从更合理的流映射起步。Causal Forcing 自身仍会在直接外推超过 5 秒时退化，因为训练 attention context 只有 5 秒；官方仓库也明确提醒，不应把原始 81 帧 checkpoint 当作长视频模型，需另接 Rolling Forcing、LongLive、Infinity-RoPE 或 Deep Forcing。

### 5.3 消融实验

| 方法 | Total ↑ | Quality ↑ | Semantic ↑ | Dynamic ↑ | VisionReward ↑ | 指令遵循 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| AR Diffusion：Diffusion Forcing | 81.76 | 82.52 | 78.71 | 60 | 1.583 | 30 |
| AR Diffusion：Teacher Forcing | 82.12 | 82.73 | 79.67 | 50 | 3.343 | 32 |
| Chunk-wise：Self Forcing ODE + DMD | 82.00 | 82.18 | 81.29 | 24 | 3.330 | 38 |
| **Chunk-wise：Causal ODE + DMD** | **84.04** | **84.59** | **81.84** | **68** | **6.326** | **56** |
| Frame-wise：Self Forcing ODE + DMD | 81.83 | 82.66 | 78.50 | 2 | 1.951 | -4 |
| **Frame-wise：Causal ODE + DMD** | **83.75** | **84.35** | **81.37** | **64** | **6.204** | **42** |

最有说服力的是 frame-wise 消融：随着 AR 链更长，错误初始化的缺陷被放大，Self Forcing 的 Dynamic Degree 仅 2、指令遵循为 -4；换成 causal ODE 后分别为 64 和 42。chunk-wise 下，causal ODE 相比 Self Forcing ODE 的 Dynamic Degree 提升 183.3%，VisionReward 提升 90.0%，说明收益并非只来自 Stage 1 的 TF 微调。

Consistency Distillation 消融同样支持教师结构匹配原则：Asymmetric CD 的 VisionReward 为 -7.983、指令遵循 -42；Causal CD 提升到 1.798 和 18，但当前 vanilla LCM 实现仍不如 DMD。

### 5.4 实验充分性判断

论文通过相同基础模型、相同初始化步数和相同 DMD 预算做了较干净的控制，且把理论缺陷对应到 frame-wise 极端退化，证据链完整。欠缺之处是 $D_{\mathrm{Bi}}$ 与 $D_{\mathrm{Causal}}$ 都由内部 3K 合成数据构成，外部研究者难以逐样本核验“数据质量完全相同”；用户研究仅 10 人 × 10 prompts，统计规模偏小；VisionReward 与 Dynamic Degree 仍可能偏好高饱和或大幅运动，不能替代真实视频连贯性评价。

---

## 6. 与相关工作的关系

- **CausVid** 首次建立双向教师→因果学生的 asymmetric DMD，并用双向 ODE 轨迹初始化；Causal Forcing 保留 DMD，修正初始化教师。
- **Self Forcing** 通过 self-rollout 使 DMD 训练条件与推理一致；Causal Forcing 指出 on-policy 后训练无法补救此前非单值的 ODE 回归。
- **Diffusion Forcing** 强调 noisy history 的覆盖和不确定性建模；本文则指出，在其特定 AR teacher 构造目标中，训练条件与最终 clean-prefix 推理不一致，TF 更合适。两者问题设置不同，不能概括为“DF 普遍错误”。
- **APT2** 用 teacher-forced consistency initialization 加 GAN；目标方向与 causal ODE 一致，但没有本文的 frame-level injectivity 分析，也不采用双向 score teacher 或 temporal KV cache。

---

## 7. 局限与批判性评价

1. **仍依赖三阶段训练**：先得到 AR teacher、再生成 3K ODE 轨迹、再 DMD；方法正确性提高了，但流水线没有变简单。
2. **理论假设与真实网络之间有间隙**：frame-level injectivity 对确定 PF-ODE 成立，但实际 VAE 压缩、有限精度、近似 ODE solver 和 stochastic sampler 都可能破坏理想条件。
3. **长视频未解决**：模型只训练 5 秒、注意力也只覆盖 5 秒；流式输出不等于无限稳定生成。
4. **数据可复现性有限**：Stage 1/2 使用内部合成集，虽然代码和 checkpoint 开源，完整从数据构造开始复现仍不完全开放。
5. **评价偏置**：主增益集中在额外 100 个高运动 prompt 与偏好模型评分，VBench Total 的绝对增量较小；需要更大规模人评和物理一致性指标。
6. **Teacher Forcing 结论有边界**：作者证明的是本论文定义的 noisy-prefix DF 与 clean-prefix 推理不匹配，不代表所有采用自生成历史、重新加噪或 history guidance 的 DF 变体都应被 TF 替代。

---

## 8. 复现与实践建议

- **代码与权重**：官方仓库公开 frame-wise、chunk-wise checkpoint，环境与 Self Forcing 基本一致；可先在单卡复现推理和 Table 1 的速度口径。
- **训练成本**：论文主流程为 2K AR teacher + 1K causal ODE + 750 DMD steps。未给总 GPU 小时；Self Forcing 同类后训练通常用 64×H100，论文只明确速度测试在单 H100，故不应据此编造完整训练卡时。
- **数据成本**：需 3K 双向合成视频及中间状态，再由 AR teacher 生成 3K causal ODE trajectories；这比海量真实视频预训练便宜，但轨迹存储和多步 teacher 推理仍昂贵。
- **关键复现检查**：ODE 轨迹的当前 chunk 必须由 AR teacher 在相同 clean prefix 下生成；若误用双向 teacher 或让目标看到未来，方法会退化回 Self Forcing 初始化。
- **长视频**：原 checkpoint 不应直接外推；需额外长视频适配。官方仓库已提供与 Rolling Forcing 等方法的集成入口，但属于后续扩展。

---

## 9. 个人判断与后续问题

我认为本文最有价值的不是“再高 0.30 VBench”，而是把**教师选择取决于损失方向**说清楚：用 MSE/forward-KL 学单值 transport map 时，教师必须和学生具有相同可见信息；用 reverse-KL/DMD 对齐最终分布时，双向强教师反而有价值。这个原则可推广到语言、音频、机器人轨迹等“非因果 teacher → 因果 student”的蒸馏。

值得继续研究的问题包括：能否在线生成 ODE target，避免 3K 轨迹存储；能否用更强 causal consistency / mean-flow 初始化取代两阶段 ODE 回归；以及能否把 frame-level injectivity 推广为带 stochastic latent 的多模态 transport，而不是用确定 MSE 映射压缩所有未来可能性。

---

## 10. 材料来源

本地 PDF、四张完整关键图及逐项来源记录见：

- [sources.md](assets/Causal%20Forcing%20-%20正确的自回归扩散蒸馏/sources.md)
- [paper.pdf](assets/Causal%20Forcing%20-%20正确的自回归扩散蒸馏/sources/paper.pdf)
