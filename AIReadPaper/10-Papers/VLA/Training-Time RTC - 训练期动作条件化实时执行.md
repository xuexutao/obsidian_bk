---
type: paper-note
status: complete
domain: VLA
paper: Training-Time Action Conditioning for Efficient Real-Time Chunking
year: 2025
arxiv: 2512.05964
doi: 10.48550/arXiv.2512.05964
source: https://arxiv.org/abs/2512.05964
project: null
code: null
tags:
  - VLA
  - Physical-Intelligence
  - Real-Time-Control
  - Action-Chunking
  - Flow-Matching
  - Prefix-Conditioning
created: 2026-10-20
updated: 2026-10-20
---

# Training-Time RTC：训练期动作条件化实时执行

## 基本信息

- **作者**：Kevin Black、Allen Z. Ren、Michael Equi、Sergey Levine
- **机构**：Physical Intelligence
- **发表**：arXiv 2512.05964v2（2025）
- **原文 PDF**：<https://arxiv.org/pdf/2512.05964>
- **前作**：[[RTC - 实时动作分块执行]]
- **真机基础模型**：π0.6；任务来自 π星0.6/RECAP 的纸盒组装与意式咖啡制作设置

## 一句话结论

Training-Time RTC 把原 RTC 在每个去噪步用反向传播完成的动作前缀 inpainting，改成训练时随机模拟推理延迟：将真实前缀作为无噪声条件、只对 postfix 做 flow-matching；它复用相同异步 runtime 接口，却把真机端延迟从 inference-time RTC 的 135 ms 降到 108 ms，在高延迟 Kinetix 上更强，并在咖啡与纸盒任务上取得相近或更好的成功率和速度。

---

## 1. 研究背景与问题定义

### 1.1 从 RTC 的成功到计算悖论

RTC 已证明：动作 chunk 必须异步生成，且新 chunk 必须条件化于旧 chunk 已承诺的动作，否则边界会抖动或切换策略。但 inference-time RTC 用 pseudoinverse guidance 做 inpainting，每个去噪步都需要 vector-Jacobian product。以 π0.5 为例，vanilla 推理 76 ms，RTC 97 ms；在 π0.6 真机实验中，RTC 的端到端延迟达到 135 ms。

这形成一个悖论：为了抵抗推理延迟，算法本身又增加推理延迟。本文追问：**能否让模型在训练时就学会“给定已执行前缀，补全剩余动作”，从而让部署端只做普通前向去噪？**

### 1.2 目标与边界

方法需要保持 RTC 的 runtime 契约：输入动作前缀 $A_{t:t+d}$ 与 delay $d$，输出 postfix $A_{t+d:H}$。这样控制线程、异步队列和 chunk 交换逻辑都不用修改。它不追求 inference-time RTC 的任意 soft mask 灵活性，而是专门优化真实部署中最关键的“硬前缀条件化”。

### 1.3 核心贡献

1. 用训练期 delay simulation 取代推理期 pseudoinverse guidance，不新增可学习参数，部署时无反向传播。
2. 给出对 DiT/action expert 的三项最小修改：逐 token flow timestep、干净前缀 + 噪声 postfix、postfix-only loss。
3. 在与 RTC 相同的 12 个动态 Kinetix 环境中证明：$d\ge2$ 后训练期方案优于推理期 inpainting，且差距随延迟增大。
4. 在 π0.6 真机咖啡和纸盒任务上，以较低延迟保持 inference-time RTC 的性能与速度，并明显快于同步执行。

---

## 2. 任务定义与输入输出

### 2.1 时间轴与条件变量

基础策略预测

$$A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}]\sim\pi(\cdot\mid o_t),$$

推理开始于 $t$，耗时 $d$ 个控制步，因此结果在 $t+d$ 才能使用。旧 chunk 与新 chunk 在 $[t,t-s+H)$ 重叠，其中 $A_{t:t+d}$ 是推理期间必然由旧 chunk 执行的 **action prefix**；$A_{t+d:H}$ 是模型需要生成的 **action postfix**。只要 $d\le H-s$，推理期间始终有旧动作可执行。

![](assets/Training-Time%20RTC%20-%20训练期动作条件化实时执行/figures/01-overview.png)

> **图 1（原论文 Figure 1，第 2 页）：前一 chunk 与当前 chunk 的重叠。** 红色前缀对应 $d$ 个已经承诺的动作；黄色位置在 inference-time RTC 中可由 soft mask 辅助约束，但 Training-Time RTC 只把红色硬前缀作为条件；绿色为必须新生成的区域。来源：<https://arxiv.org/abs/2512.05964>

### 2.2 基础 flow-matching 目标

对真实 chunk $A_t$、高斯噪声 $\epsilon\sim\mathcal N(0,I)$ 和 flow timestep $\tau$，论文写作

$$A_t^\tau=\tau A_t+(1-\tau)\epsilon,$$

$$\mathcal L(\theta)=\mathbb E\left\|v_\theta(A_t^\tau,o_t,\tau)-(\epsilon-A_t)\right\|^2.$$

$v_\theta$ 是 action expert 的速度场。普通训练中，同一 chunk 的所有时间 token 共用 $\tau$；普通推理从噪声积分到动作。Training-Time RTC 的关键是让 prefix token 与 postfix token 使用不同的噪声状态和 loss mask。

---

## 3. 核心方法

### 3.1 学习条件分布 $p(A_{t+d:H}\mid o_t,A_{t:t+d})$

训练样本本来就包含完整真实 chunk。作者随机采样一个 delay $d$，把前 $d$ 个真实动作直接交给网络，把后面动作正常加噪，训练网络只复原 postfix。模型因此显式学习

$$p_\theta(A_{t+d:H}\mid o_t,A_{t:t+d},d).$$

部署时，旧 chunk 提供前缀，新观测提供 $o_t$，模型按普通去噪过程生成 postfix；这正好实现 RTC runtime 所需的接口，无需 guidance。

### 3.2 三项最小修改

![](assets/Training-Time%20RTC%20-%20训练期动作条件化实时执行/figures/02-method.png)

> **图 2（原论文 Figure 2，第 2 页）：Training-Time RTC 的 DiT 条件方式。** 前缀使用真值、flow timestep 固定为 1，且不计算 loss；postfix 使用随机 $\tau$ 加噪并计算 flow-matching loss。每个 token 的 timestep 同时向网络编码“前缀到哪里结束”。来源：<https://arxiv.org/abs/2512.05964>

**第一，逐动作 token 的 flow timestep。** 标准 DiT 通过 adaLN-zero 注入一个全局 timestep。这里允许 scale、shift、gate 随 token 不同：前缀 token 的 timestep 设为 1，postfix token 为采样的 $\tau$。参数量不变，只是 timestep embedding 从标量广播改为逐 token 输入。

**第二，前缀永远干净。** prefix 直接使用 ground-truth action，不加噪；$\tau=1$ 表示它位于数据端点。postfix 按普通 flow matching 构造 $A^\tau$。网络由此把前缀当条件而非待生成变量。

**第三，只在 postfix 计算 loss。** 对前缀输出做 mask，不要求模型重建已经给出的动作。若 $m_i(d)=\mathbf1[i\ge d]$，可将目标写成

$$\mathcal L_{\mathrm{TT\mbox{-}RTC}}=
\mathbb E_{A,o,\tau,d,\epsilon}
\sum_{i=0}^{H-1}m_i(d)
\left\|v_{\theta,i}(\widetilde A,o,\widetilde\tau)-u_i\right\|_2^2,$$

其中 $\widetilde A_i=A_i,\widetilde\tau_i=1$（$i<d$），而 $i\ge d$ 时用正常加噪动作与 $\tau$；$u_i$ 是论文 flow 目标。这个 mask 防止固定前缀浪费训练监督，也避免网络尝试改写条件。

### 3.3 随机 delay 训练

真实端到端延迟随硬件、网络和观测处理变化，训练时无法知道精确 $d$，因此从预设分布采样。Kinetix 中从 $\{0,1,2,3,4\}$ 以指数递减权重采样，高延迟需要的采样频率更低；真机中对 $d\in[0,10]$ 均匀采样，对应 50 Hz 下最高 200 ms。

延迟分布不是无关紧要的实现细节：$d$ 决定每个位置获得多少生成监督。若总是采大 $d$，chunk 前部很少被预测；若训练分布过短，部署长延迟时模型面对未见过的长前缀。论文在 Kinetix 的 $d=0/1$ 上略逊于 inference-time RTC，作者就将其归因于部分早期动作得到的生成监督更少。

### 3.4 推理与在线执行

生成时输入实际旧 chunk 前缀和 delay，prefix token 固定为动作端点，postfix 从噪声开始正常积分。无需计算 $\partial\widehat A^1/\partial A^\tau$，也没有 guidance clipping 或 soft-mask 日程。生成后的完整 chunk 继续由 RTC 的后台推理线程原子替换，因此控制 runtime 不变。

与 inference-time RTC 的本质差异是：前者借助模型局部 Jacobian 在采样时强迫任意预训练分布满足约束；后者直接让条件分布成为训练分布的一部分。前者灵活但近似误差和计算都随前缀变长而增大，后者对训练覆盖到的 delay 更稳且更便宜。

### 3.5 复杂度

Training-Time RTC 的参数量不变，推理复杂度退回普通 VLA 的 $n$ 次前向去噪；额外成本只在训练：随机 mask、逐 token timestep，以及某些位置不计 loss。真机端到端平均延迟为 **108 ms（约 $d=5$）**，inference-time RTC 为 **135 ms（约 $d=7$）**，减少 27 ms、约 20%。两者都在远端 H100 上用 5 步去噪。

---

## 4. 数据集与实验设置

### 4.1 Kinetix

完全沿用 RTC 的 12 个动态任务与数据：$H=8$，4 层 MLP-Mixer，专家混合数据，训练 32 epochs。naive async 和 inference-time RTC 共用普通训练 checkpoint；Training-Time RTC 从 epoch 24 恢复，做 8 epochs prefix-conditioning 微调，使所有方法总训练量相同。固定 $s=\max(d,1)$，测试 $d=0\ldots4$，每个点 **2048 trials**，报告二元 solve rate 和 95% Wilson 区间。

### 4.2 真机

基础模型为 π0.6，任务是：

- **Box building**：从扁平纸板折出箱体，长程、易受材料弯曲与粘连影响；
- **Espresso making**：磨豆、压粉、萃取和倒饮品，包含刚性接触、液体与多阶段操作。

同步 baseline 与 inference-time RTC 使用同一普通 checkpoint；Training-Time RTC 使用 prefix-conditioned checkpoint。二者都从 π0.6 基座针对任务微调 **8000 gradient steps、batch size 512**。真机 50 Hz，训练 delay 均匀采样 0–10；评测以成功率和 episode duration 衡量。论文图中误差条分别是成功率的 68% Wilson 区间与时长的 ±1 SEM。

### 4.3 Baseline

- **Synchronous**：chunk 结束后等待下一次推理，边界有可见暂停；
- **Naive asynchronous**：完成即切换，不约束旧 chunk；
- **Inference-time RTC**：伪逆 guidance + soft mask，免重训但需要每步反向；
- **Training-time RTC**：本文方法，硬前缀条件化、普通前向去噪。

---

## 5. 实验结果

### 5.1 Kinetix：高 delay 时训练期条件化更稳

Figure 3 显示，$d=0/1$ 时 Training-Time RTC 略低于 inference-time RTC；从 **$d\ge2$ 开始反超**，且到 $d=4$ 差距明显扩大。论文解释是：长 prefix 使推理期 inpainting 必须通过模型局部线性化“更用力”地扭转样本，近似质量下降；训练期方案学到的是长前缀下的真实条件生成。每个点 2048 次试验，但论文没有提供逐点数值表，因此只能把“$d\ge2$ 反超”作为精确的离散结论，不能把曲线目测值写成精确百分比。

相对 naive async，两种 RTC 都保持明显优势，说明关键并非只做异步，而是显式维持跨 chunk 条件一致性。

### 5.2 真机成功率和速度

![](assets/Training-Time%20RTC%20-%20训练期动作条件化实时执行/figures/03-results.png)

> **图 3（原论文 Figure 5，第 3 页）：咖啡与纸盒任务的成功率和时长。** 由图坐标可读：咖啡任务同步/推理期 RTC/训练期 RTC 成功率约 0.82/0.89/0.93，时长约 143/115/112 秒；纸盒约 0.50/0.56/0.66，时长约 170/133/131 秒。论文未附数值表，因此这些是图读近似值；精确结论是两种 RTC 性能与速度相当，且都明显快于同步执行。来源：<https://arxiv.org/abs/2512.05964>

训练期方案没有为低延迟付出真机性能代价：它在两项任务上成功率不低于 inference-time RTC，时长几乎相同；由于自身推理更快，实际 runtime delay 还从 $d\approx7$ 降为 $d\approx5$。同步策略的成功率并未完全崩溃，但 chunk 之间可见停顿使总时长增加约 30–40 秒。

### 5.3 消融与机制证据

论文篇幅短，没有独立的逐组件真机消融表，但提供了三类等价证据：

1. **同训练预算比较**：Kinetix 只把最后 8/32 epochs 改成 prefix-conditioning，总训练量匹配，排除了额外训练步数带来的优势。
2. **delay sweep**：性能交叉点位于 $d=2$；随 prefix 变长，推理期 RTC 下降更快，直接支持“Jacobian 线性化在长约束下受限”的机制解释。
3. **推理延迟对比**：真机同为远端 H100、5 步去噪，108 ms 对 135 ms，直接量化了去掉 VJP 的收益。

尚缺少的消融包括：delay 采样分布（均匀/指数/按真实日志）、prefix dropout、混合 soft+hard 条件、不同 $H$ 和不同 VLA 架构。这些缺失使“几行代码即可普适替代”的外推仍需谨慎。

### 5.4 结果是否支撑主张

论文主张是“Training-Time RTC 是更便宜的 drop-in replacement”，实验对此基本充分：同 runtime 接口、相近真机任务指标、更低延迟，并在高仿真延迟下更强。但真机只有 2 个任务、单一 π0.6 系列，且 Training-Time 与 inference-time 使用不同 checkpoint，尚不能分离“条件化训练的正则化收益”和“执行算法收益”。更严格的验证应交叉评测多个预训练数据规模和未见 delay。

---

## 6. 与相关工作及系列关系

### 6.1 RTC 系列的两种约束位置

- **RTC（2506.07339）**：约束位于采样器；优势是任何预训练 flow/diffusion VLA 即插即用、可 soft-mask 任意重叠区；劣势是 VJP 开销、长前缀局部线性化误差。
- **Training-Time RTC（2512.05964）**：约束位于模型训练分布；优势是推理零额外反向开销、长前缀更稳；劣势是必须重训/微调、只能处理训练覆盖的硬前缀形式。

二者不是简单新旧替代，而是灵活性与效率的 Pareto 选择。已有 checkpoint 或未知约束形式适合 inference-time RTC；可控制训练 recipe、部署 delay 稳定的产品系统更适合 Training-Time RTC。

### 6.2 与 π星0.6 的关系

本文的两项真机任务和实验平台来自 π星0.6/RECAP。RECAP 让策略通过经验与纠正提高“会不会、做得好不好”，Training-Time RTC 让同一策略“边想边做、不中断”。前者作用于数据和策略参数，后者作用于动作条件与执行调度，理论上可以联合训练：RECAP 收集在线数据时也采用 prefix-conditioned runtime，可减少 train/deploy 执行协议偏移。

### 6.3 与层级 System 1/System 2 的关系

Gemini Robotics、GR00T 等用轻量 System 1 高频控制、重型 System 2 低频规划。Training-Time RTC 不拆模型，而是让同一个 action expert 学会从已承诺前缀续写，因此不限制低层模型容量；代价是每次仍需完整 VLA 前向。两类方法可组合：低频高层规划 + prefix-conditioned 高频 chunk 生成。

---

## 7. 局限与批判性评价

1. **硬前缀不等于完整 soft mask**：论文明确承认，训练期方法无法灵活利用 prefix 之外的重叠动作，可能在小 $d$、多模态分叉时失去 RTC 的连续性优势。
2. **需要预先设计 delay 分布**：硬件、网络或模型升级会改变 $d$；训练覆盖不当会使某些动作位置监督不足，或部署落入 OOD delay。
3. **仍受 horizon 上限约束**：若 $d>H-s$，旧 chunk 不够覆盖推理时间，训练期条件化也无法凭空提供动作。
4. **仅实证 flow/DiT 类 action expert**：逐 token adaLN timestep 对 DiT 很自然，但对 U-Net diffusion、离散动作 token 或非迭代生成器是否同样“几行代码”并未验证。
5. **真机规模较小**：两项复杂任务有代表性，但不足以覆盖突发外力、动态人机环境、腿式机器人和安全关键场景。
6. **无完整数值表与公开代码**：主结果主要以柱状图呈现，且论文未披露独立代码仓库，降低精确复算和实现一致性。

---

## 8. 复现与实践建议

### 8.1 最小实现

在现有 flow action expert 中，把 timestep tensor 从 `[batch]` 扩展为 `[batch,H]`；随机采样 $d$，拼接 `clean_prefix` 与 `noisy_postfix`；构造 `[0]*d+[1]*(H-d)` loss mask。推理 API 固定为 `(observation, prefix, delay) -> postfix`，再接入 RTC 的异步控制循环。

### 8.2 必做测试

- $d=0$ 应退化为普通 flow policy；$d=H$ 时 loss 为空，应在训练 sampler 中避免或显式处理。
- prefix 在采样全过程保持不变，postfix 起点应为噪声。
- 分别按实际 delay histogram、均匀分布、指数分布训练，比较每个动作位置的有效监督次数。
- 验证未见 delay 与突发延迟，特别是 $d$ 接近 $H-s$ 的边界。
- 同时记录成功率、时长、跨 chunk 位置/速度/加速度和真实端到端 latency。

### 8.3 复现成本

Kinetix 版本成本中等，只需小型 MLP-Mixer 与仿真数据；π0.6 真机复现成本高，需要模型权重/训练基础设施、远端 H100、50 Hz 双臂平台、咖啡机/纸盒任务数据和人工评测。论文只披露 8000 steps、batch 512、5 denoising steps 与 delay 范围，优化器、学习率等细节不足，严格复现需额外材料。

---

## 9. 个人启发与后续问题

这篇工作的核心设计模式可推广为：**把部署时不可撤销的历史决策显式编码为生成模型的训练条件**。除机器人动作外，流式语音、视频生成、交互式代理和实时规划也可把已输出前缀设为终点状态、只训练后缀。后续值得研究：让模型连续条件化于真实毫秒延迟而非离散 $d$；联合学习 soft mask；按在线延迟日志自适应重采样；以及把安全急停作为比动作连续性优先级更高的条件 token。

---

## 10. 材料来源

本笔记基于官方 PDF 全文与其中 Figure 1/2/5 撰写；本地 PDF 与三张图的逐项溯源见 [sources.md](assets/Training-Time%20RTC%20-%20训练期动作条件化实时执行/sources.md)。论文未提供项目页或独立代码仓库，故 frontmatter 保留 `null`。
