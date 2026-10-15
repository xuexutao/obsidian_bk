---
type: paper-note
status: done
domain: VLM
paper: "Kimi K2.5: Visual Agentic Intelligence"
year: 2026
arxiv: "2602.02276"
doi: "10.48550/arXiv.2602.02276"
source: "https://arxiv.org/abs/2602.02276"
project: null
code: null
tags:
  - VLM
  - MoE
  - Multimodal-Agent
  - Reinforcement-Learning
  - Multi-Agent
created: 2026-10-15
updated: 2026-10-15
---

# Kimi K2.5：视觉智能体与并行智能体强化学习

## 基本信息

- **作者：** Kimi Team
- **版本：** arXiv v2，2026-08-07 修订
- **模型定位：** 原生图文视频多模态 MoE、工具使用智能体、并行多智能体编排
- **基座：** Kimi K2，1.04T 总参数、32B 激活参数、384 个专家、每 token 激活 8 个专家
- **本地原文：** [paper.pdf](assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/sources/paper.pdf)
- **完整溯源：** [sources.md](assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/sources.md)

**一句话结论：**

Kimi K2.5 的真正主线不是“给 K2 接一个视觉塔”，而是把视觉提前纳入长周期联合预训练，再用 **Zero-Vision SFT → outcome-based 视觉 RL → 按能力组织的联合图文 RL** 激活视觉智能体，最后以 **PARL** 只训练 orchestrator，将串行 agent 扩展为可学习的并行 **Agent Swarm**；它在开放评测上证明了跨模态互促和并行加速，但训练配比、硬件成本、PARL 超参数与内部评测仍披露不足。

---

## 1. 研究背景与问题定义

### 1.1 从“多模态回答”转向“视觉智能体”

传统 VLM 主要优化“图像加问题到文本答案”，而智能体还必须决定何时调用工具、如何操作图像、如何管理长上下文，并在复杂任务中持续规划。这里有两层瓶颈。第一层是训练瓶颈：很多模型先完成语言预训练，后期再以很高比例补入视觉 token，视觉成为附加能力，跨模态共同演化时间太短。第二层是执行瓶颈：单 agent 把检索、浏览、下载、验证和写作串成一条长链，任务越宽、上下文越长，延迟和工具调用预算越容易成为上限。

K2.5 分别用“原生联合多模态训练”和“可学习并行编排”处理这两层问题。前者试图让文本与视觉相互迁移，后者试图把计算并行真正转化为任务质量和墙钟时间收益。

![](assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/figures/01-main-results.png)

> **图 1：Kimi K2.5 主要结果。** 图中把 agent、coding、image、video 放在同一模型下比较；K2.5 并非每项都优于闭源模型，但在 HLE+tools、BrowseComp、DeepSearchQA、MathVision、OmniDocBench、VideoMMMU 和 LongVideoBench 等任务上显示出强竞争力。来源：论文 Figure 1，arXiv v2。

### 1.2 论文要回答的核心问题

论文实际上回答四个相互关联的问题：视觉应该何时进入预训练；文本 SFT 能否在没有视觉 SFT 轨迹时启动视觉工具能力；视觉 RL 会不会损伤语言，还是可以反向增强文本；以及并行 agent 是否能够通过 RL 学会“是否并行、如何拆分、创建多少子智能体”，而不是依赖固定启发式。

### 1.3 核心贡献

1. **MoonViT-3D。** 用共享参数和共享 embedding space 统一图像与视频，在时间维进行 patch-and-pack 与轻量池化，实现约 4 倍时间压缩。
2. **早期视觉融合与联合优化。** 在固定视觉/文本 token 预算下，早期、低比例、长周期共训优于晚期、高比例注入。
3. **Zero-Vision SFT 与联合多模态 RL。** 仅以文本 SFT 启动视觉工具行为，再用视觉结果奖励纠正忽略图像的问题，并按能力而非模态组织 RL。
4. **Agent Swarm 与 PARL。** 让可训练 orchestrator 动态创建、分工和调度冻结子智能体，只优化高层协调策略。
5. **训练和推理效率机制。** DEP 将视觉 encoder 从主 pipeline 解耦；Toggle 降低冗余推理 token；Agent Swarm 把宽搜索的墙钟时间缩短约 3–4.5 倍。

---

## 2. 任务定义与输入输出

### 2.1 模型输入

K2.5 接受纯文本、单图/多图、原始分辨率且可变宽高比的图像、视频、长文本、长视频、OS 截图、GUI 状态和工具返回。评测默认上下文长度为 256K；长上下文中训将序列从 32K 逐步扩展到 262,144。论文在长视频评测中输入超过 2,000 帧，但没有将其声明为模型硬上限，也没有披露最大图像数、最大分辨率、帧率或视频时长。

### 2.2 模型输出

输出仍是自回归 token，但语义上覆盖普通回答、长链推理、代码、结构化工具调用、IPython 图像操作、GUI 动作、搜索与浏览动作、子智能体创建和任务委派，以及最终的软件或长文档产物。论文没有披露最大输出 token、thinking/instant 的精确切换算法，也没有公开 Agent Swarm 工具 schema。

### 2.3 统一目标

K2.5 的关键不是为图像、视频、工具分别增加任务头，而是把视觉表示通过 MLP projector 接入 K2 MoE decoder，然后以 next-token prediction、结果奖励和生成式奖励模型统一训练。视觉定位、OCR、计数、代码和 agent 轨迹最终都进入一个 token 生成接口。

---

## 3. 核心机制一：MoonViT-3D 与早期视觉融合

### 3.1 三组件架构

模型由 MoonViT-3D、MLP projector 和 Kimi K2 MoE 组成。语言基座的 1.04T/32B-A32B 稀疏结构提供容量，视觉塔负责原始分辨率图像与视频，projector 负责表示空间对接。论文没有披露 MoonViT-3D 的层数、hidden size、head 数、patch size、视觉 token 上限和 projector 维度，因而无法从报告独立复刻网络配置。

### 3.2 MoonViT-3D 如何统一图像与视频

MoonViT 继承 SigLIP-SO-400M 初始化和 NaViT 的 patch packing：图像切成 patch 后展平，不同尺寸样本可打包成一维序列。MoonViT-3D 把这一思路扩展到时间维：最多四个连续帧构成一个时空块，来自各帧的二维 patch 共同展平并参与同一套 attention。进入 projector 前，相同时间块中的 patch 被轻量时间池化，获得约 4 倍时间压缩。

这个设计的价值不只是少 token。图像与视频完全共享参数和 embedding space，静态图像学习到的 OCR、物体、布局与世界知识可以直接迁移到视频；额外时间注意力又保留高速运动和视觉特效线索。代价是论文没有给出池化算子、帧采样率和位置编码细节，也没有画出视觉塔内部结构。

### 3.3 早期视觉融合消融

![](assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/figures/04-early-fusion-ablation.png)

> **图 9：固定视觉/文本总预算下的视觉注入策略。** 红色的 10:90 早期融合拥有最长共训轨迹，并在视觉知识、视觉推理、OCR、文本知识和代码上整体更好；这表明“共训时长”比“末期瞬时视觉比例”更关键。来源：论文附录 Figure 9。

对应 Table 1 的末端结果如下：

| 策略 | 视觉注入时点 | 视觉:文本 | 视觉知识 | 视觉推理 | OCR | 文本知识 | 文本推理 | 代码 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Early | 0% | 10:90 | **25.8** | **43.8** | **65.7** | **45.5** | 58.5 | **24.8** |
| Mid | 50% | 20:80 | 25.0 | 40.7 | 64.1 | 43.9 | **58.6** | 24.0 |
| Late | 80% | 50:50 | 24.2 | 39.0 | 61.5 | 43.1 | 57.8 | 24.0 |

应谨慎区分消融配置和正式模型配方：10:90 是受控实验中的 Early 设置，论文只称正式训练采用较温和、恒定的视觉比例，并未披露最终精确比例。

---

## 4. 核心机制二：Zero-Vision SFT 与联合多模态 RL

### 4.1 Zero-Vision SFT 的含义

Zero-Vision SFT 不是零视觉预训练，也不是零视觉能力，而是指后训练的 SFT 冷启动阶段只用文本指令和文本 agent 轨迹，不加入人工视觉 SFT 轨迹。其成立依赖前一阶段已经形成的图文联合表示。模型从文本工具调用中学到规划与 API 使用，再把图像操作代理到 IPython，例如 crop、rotate、flip、二值化、像素计数、尺寸估计、定位和 OCR。

论文报告，初步的 text-vision SFT 在视觉智能体任务上反而“much worse”，作者将原因归于高质量视觉轨迹不足。不过报告没有给出数据规模、基准、分差或 SFT 超参数，因此这个结论应视为有方向性、但尚不充分量化的经验。

### 4.2 为什么还需要视觉 RL

仅文本 SFT 启动后仍会出现两个失败：模型偶尔忽略图像；即便任务必须看图，也可能把注意力放在语言先验上。于是论文使用 outcome-based visual RL，刻意选择答案必须依赖视觉输入的三类任务：grounding 与计数、图表与文档理解、vision-critical STEM。视觉轨迹还被用于 rejection-sampling fine-tuning，形成自改进数据闭环。

![](assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/figures/03-vision-rl-curves.png)

> **图 2：从 Zero-Vision SFT 起点继续扩展视觉 RL。** MMMU-Pro、MathVision、CharXiv(RQ) 与 OCRBench 随 RL FLOPs 整体上升，说明文本冷启动并非性能终点；真正稳定的视觉注意和任务能力来自持续的结果反馈。来源：论文 Figure 2。

### 4.3 视觉 RL 对文本的反向迁移

| 文本 benchmark | 视觉 RL 前 | 视觉 RL 后 | 提升 |
|---|---:|---:|---:|
| MMLU-Pro | 84.7 | 86.4 | +1.7 |
| GPQA-Diamond | 84.3 | 86.4 | +2.1 |
| LongBench v2 | 56.7 | 58.9 | +2.2 |

作者的解释是，视觉 RL 强化了结构化信息抽取、计数和 OCR 式校准，类似能力迁移到文本。更可靠的结论是：在这三个指标上没有观察到语言退化且有提升；但论文没有提供多随机种子或更广泛文本基准，不能把它泛化为“视觉 RL 必然提升所有文本能力”。

### 4.4 按能力而非模态组织 RL

K2.5 不建立“文本 RL 专家”和“视觉 RL 专家”，而按 knowledge、reasoning、coding、agentic 等能力域划分。每个域同时接收纯文本和多模态 query，GRM 也跨模态评价异构轨迹。这个组织方式的含义是：输入模态只是证据载体，真正被优化的是可迁移能力。

### 4.5 策略优化与奖励

每个问题由旧策略采样 $K$ 条回答，使用组内平均奖励作为基线，并对 token 级策略比率做 clipping，同时加入 log-ratio 的平方惩罚：

$$L_{\mathrm{RL}}(\theta)=\mathbb{E}_{x\sim\mathcal D}\left[\frac{1}{N}\sum_{j=1}^{K}\sum_{i=1}^{|y_j|}\mathrm{Clip}\!\left(\frac{\pi_\theta(y_j^i\mid x,y_j^{0:i})}{\pi_{\mathrm{old}}(y_j^i\mid x,y_j^{0:i})},\alpha,\beta\right)(r(x,y_j)-\bar r(x))-\tau\left(\log\frac{\pi_\theta(y_j^i\mid x,y_j^{0:i})}{\pi_{\mathrm{old}}(y_j^i\mid x,y_j^{0:i})}\right)^2\right]$$

其中 $N$ 是生成 token 总数，$\bar r(x)$ 是同一问题 $K$ 条回答的平均奖励。论文称 clipping 更像梯度 mask：比率超出区间的 token 梯度归零，并不按 advantage 正负采用 PPO 的非对称裁剪。$\alpha,\beta,\tau$ 与 $K$ 均未披露。

可验证任务用结果规则奖励；开放式任务由多套 GRM rubric 评价 helpfulness、相关性、细节、审美和指令遵循。视觉任务采用 IoU/高斯距离软匹配的 F1、mask IoU、归一化编辑距离和计数误差。复杂视觉 puzzle 使用 Kimi K2 verifier。

---

## 5. 核心机制三：Agent Swarm 与 PARL

### 5.1 架构：只训练 orchestrator

![](assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/figures/02-agent-swarm.png)

> **图 3：Agent Swarm 架构。** Orchestrator 根据任务动态创建角色不同的冻结子智能体，把大量独立子任务并行分派，只接收任务相关结果后完成聚合。来源：论文 Figure 3。

Agent Swarm 包含一个可训练 orchestrator 和一组从固定中间策略 checkpoint 实例化的冻结 subagents。orchestrator 学习是否并行、何时并行、创建多少 agent、每个 agent 的专长、任务分配和结果聚合。子智能体轨迹不进入可微优化，输出只作为环境 observation。

冻结子智能体是一个关键取舍。端到端联合训练面临模糊 credit assignment：最终答案正确不代表每个子任务正确，最终失败也不意味着所有分支都错；子策略持续变化还会让 orchestrator 面对非平稳环境。将低层执行冻结后，高层调度的奖励更稳定。训练先用小子智能体，再切换到大子智能体，以降低 orchestrator 学习成本。

### 5.2 PARL 奖励

$$r_{\mathrm{PARL}}(x,y)=\lambda_1r_{\mathrm{parallel}}+\lambda_2r_{\mathrm{finish}}+r_{\mathrm{perf}}(x,y)$$

$r_{\mathrm{parallel}}$ 奖励创建子智能体，避免 orchestrator 退化为串行单 agent；$r_{\mathrm{finish}}$ 奖励子任务完成率，抑制为了刷并发而大量生成无意义 agent；$r_{\mathrm{perf}}$ 才是最终任务结果。训练中 $\lambda_1$、$\lambda_2$ 逐步退火至零，使最终策略回到真正的任务质量。初值、退火曲线、最大子智能体数量和并发上限均未披露。

### 5.3 Critical Steps

并行组的墙钟成本由最慢分支决定，因此论文采用类似关键路径的度量：

$$\mathrm{CriticalSteps}=\sum_{t=1}^{T}\left(S_{\mathrm{main}}^{(t)}+\max_iS_{\mathrm{sub},i}^{(t)}\right)$$

这个指标不会因为无意义增加 agent 数就变好；只有平衡拆分并缩短最长分支，才能减少 critical steps。它更接近并行系统的延迟目标，但仍不是 FLOPs 或经济成本：并发资源越多，总计算量可能上升。

### 5.4 训练任务与主动上下文管理

PARL 的合成 prompt 不直接命令“请并行”，而是构造顺序执行难以在预算内完成的 wide search、deep search、长文档阅读、大规模下载和长文写作任务。模型必须自行发现并行策略。

Agent Swarm 还可视为 context sharding。Hide-Tool-Result、Summary 或 Discard-all 是上下文即将溢出后的截断；Agent Swarm 在执行前将任务分到独立局部上下文，仅将摘要性结果返回主 agent，避免完整工具轨迹污染全局上下文。这一机制提升的是系统级有效上下文，而不是单模型窗口本身。

---

## 6. 训练数据与阶段

### 6.1 预训练三阶段

| 阶段 | 序列长度 | 论文表中 token | 可训练组件 | 主要数据 |
|---|---:|---:|---|---|
| ViT Training | 4,096 | 1T | ViT；短桥接阶段只训 projector | alt text、合成 caption、grounding、OCR、图像/视频文本对 |
| Joint Pre-training | 4,096 | 15T | ViT + LLM | 文本、视觉、知识交错、视频、OS 截图、coding 等 |
| Joint Long-context Mid-training | 32,768→262,144 | 500B→200B | ViT + LLM | 高质量图文、长文本、长视频、reasoning、Long-CoT |

ViT 从 SigLIP 持续预训练，只使用条件 caption 的交叉熵 $L_{caption}$，不使用对比损失。第一步将 MoonViT-3D 与 Moonlight-16B-A3B 对齐，约 1T token；随后短暂只训练 projector，对接 1T 参数 LLM。联合预训练从接近末期的 K2 checkpoint 继续约 15T vision-text token，并增加 coding 权重、控制各数据源最大 epoch。正式数据源比例和视觉/文本比例没有公开。

表中的“500B→200B”没有解释是两个子阶段 token 数还是筛选后的预算变化，因此不应自行求和为一个确定总量。

### 6.2 SFT 与 RL 数据

SFT 候选回答来自 K2、K2 Thinking、内部专家模型、人工标注、prompt engineering 和多阶段验证。论文没有披露 SFT 总样本数、token 数、人工/合成比例、学习率和 epoch。

RL 数据包括可验证的推理、数学、代码和 agent 任务，开放式 chat/coding/search/artifact 任务，以及视觉定位、文档、OCR、计数和视觉 STEM。PARL 还使用宽搜索、深搜索、100+ 文档、批量下载和超长写作类工作负载。数据量和混合权重未披露。

### 6.3 DEP 训练基础设施

DEP 分三步：先把视觉 encoder 复制到所有 GPU，按图像数或 patch 数平衡执行 vision forward，只保留最终输出；再运行主 Transformer 的 forward/backward；最后重算视觉 encoder forward 并 backward。这样可以沿用纯文本 K2 的并行策略，并使多模态训练效率达到纯文本训练的约 90%。论文没有披露 GPU 型号、数量、总 FLOPs、训练时长和能耗。

---

## 7. 定量结果与 baseline

### 7.1 通用、视觉、视频与 GUI

| 任务 | K2.5 | 主要比较与解读 |
|---|---:|---|
| HLE-Full | 30.1 | 低于 Gemini 3 Pro 37.5；有工具后升至 **50.2**，高于 GPT-5.2 45.5 |
| AIME 2025 | 96.1 | 接近 GPT-5.2 100，高于 Claude 92.8 |
| SWE-Bench Verified | 76.8 | 低于 Claude 80.9、GPT-5.2 80.0 |
| MMMU-Pro | 78.5 | 低于 Gemini 81.0 |
| MathVista mini | **90.1** | 略高于 Gemini 89.8 |
| OCRBench | **92.3** | 高于 Gemini 90.3 |
| LongVideoBench | **79.8** | 高于对照中的 GPT-5.2 76.5 与 Gemini 77.7 |
| OSWorld-Verified | 63.3 | 低于 Claude 66.3；只用 GUI 动作 |

K2.5 的结果支持“统一模型覆盖多域”，但不是全面绝对领先。尤其 SimpleQA Verified 36.9 显著低于 Gemini 72.1，SWE/Terminal/PaperBench 等真实工程任务仍落后于部分闭源模型；这让论文的 SOTA 叙述必须按具体 benchmark 阅读。

### 7.2 Agent Swarm

| Benchmark | Agent Swarm | 单 Agent K2.5 | 绝对提升 |
|---|---:|---:|---:|
| BrowseComp | **78.4** | 60.6 | +17.8 |
| WideSearch | **79.0** | 72.7 | +6.3 |
| In-house Swarm Bench | **58.3** | 41.6 | +16.7 |

WideSearch 达到不同 Item-F1 目标时，Agent Swarm 的墙钟时间比单 agent 快约 3–4.5 倍。内部 Swarm Bench 含 WildSearch、Batch Download、100+ 文档 WideRead 与超过 100k words 的 Long-Form Writing，但数据和评分器未公开，不能独立复现。

### 7.3 Token 效率

与 K2 Thinking 相比，K2.5 在 AIME、HMMT、IMO-AnswerBench、LiveCodeBench 和 HLE-Text 上通常用更少输出 token且得分更高。Toggle 训练使平均输出减少约 25%–30%，但与 Gemini 3 Pro 相比，K2.5 多数任务仍消耗更多 token。Toggle 的 $\lambda$、切换间隔 $m$、预算百分位 $\rho$ 均未披露。

---

## 8. 消融与证据强度

### 8.1 有力证据

早期融合实验控制了视觉和文本总 token 预算，是论文中因果性较强的消融。Agent Swarm 与单 agent 的同模型比较也直接支持并行编排收益。视觉 RL 前后文本 benchmark 的变化说明至少没有明显的灾难性语言遗忘。

### 8.2 证据不足之处

Zero-Vision SFT 对比视觉 SFT 只有定性描述；MoonViT-3D 缺少与独立图像/视频编码器、无时间池化版本的正式消融；PARL 没有报告去掉 $r_{parallel}$、$r_{finish}$ 或不同退火策略的对照；Agent Swarm 延迟缺少绝对秒数、硬件、并发资源和总计算量。因此“更快”应理解为额外并行资源下的墙钟时间更短，而非总成本更低。

---

## 9. 版本关系

K2.5 继承 Kimi K2 的 1.04T MoE 语言基座和 MuonClip/QK-Clip 训练体系；相对 K2，它加入原生多模态预训练、MoonViT-3D、视觉/文本联合后训练和 Agent Swarm。Kimi-VL 提供 MoonViT 和多模态设计前史，但 K2.5 改为更深度地与 K2 联合训练，并扩展到视频和 agent。论文公开的是 post-trained K2.5 checkpoint；网络和数据的全部训练细节不足以从零重建。

---

## 10. 局限、失败模式与未来工作

论文没有独立 limitations 章，但正文暴露出清晰边界。Zero-Vision SFT 后会忽略图像；长链 agent 会遭遇上下文和工具预算瓶颈；多 agent 有 credit assignment、稀疏延迟奖励和非平稳性；PARL 可能 serial collapse 或 spurious parallelism；固定 token 预算可能 length overfitting；多模态训练存在视觉负载不均。

更关键的未覆盖问题包括：训练数据授权与去重策略、完整安全评测、多智能体错误传播和恶意工具调用、subagent 数量失控、内部 benchmark 污染、Agent Swarm 成本与并发公平性。未来工作应公开可复现的 PARL 环境和成本归一化评测，并把中间任务完成质量纳入 process reward，而不是只看最终产物。

---

## 11. 复现成本与实践建议

### 11.1 从零训练

从零复现几乎不可行：K2 基座 1.04T 总参数，K2.5 联合预训练约 15T token，且关键数据配比、SFT/RL 规模、GRM、硬件和并行配置均未披露。即便有 checkpoint，也无法复刻训练结论的全部因果链。

### 11.2 可行的分层复现

1. **早期融合小规模验证：** 用 3B–8B VLM 固定视觉/文本总 token，比较 10:90 early、20:80 mid、50:50 late。
2. **Zero-Vision SFT：** 先做图文预训练，再只用文本工具轨迹 SFT，最后用视觉可验证任务 RL；重点检测是否忽略图像。
3. **PARL 最小环境：** 固定 subagent checkpoint，只训练小 orchestrator；任务限制为可验证的并行检索，并同时报告成功率、critical steps、总 tool calls、总 token 和总 GPU/CPU 时间。
4. **成本公平性：** 单 agent 与 swarm 必须固定或分别报告总计算预算，否则墙钟加速不能等价于效率提升。

---

## 12. 个人判断

我认为 K2.5 最有价值的不是某个 benchmark，而是把三类“训练顺序”问题串起来：视觉要早点进入、SFT 不必先制造低质量视觉轨迹、并行 agent 只先学高层调度。尤其 **Zero-Vision SFT + 视觉 outcome RL** 暗示：如果基础模型已建立跨模态表示，后训练数据不一定要逐模态复制全部行为轨迹，能力可以从较成熟模态迁移，再由目标模态的可验证反馈校正。

Agent Swarm 的思想同样合理：冻结执行层、训练协调层，比端到端多 agent 更容易稳定。不过它目前更像“系统能力扩展”而非单模型能力：更高并发换取更低延迟和更宽搜索，优势依赖资源供给与任务可分解性。对于强耦合、必须共享细粒度中间状态的任务，context sharding 可能丢失关键证据，PARL 也未证明一定优于精心设计的串行规划。

总体评级应偏高但保留条件：训练路线有启发，跨模态和并行结果亮眼；可复现性、成本透明度和 Agent Swarm 的公开评测不足，是判断其一般性的主要障碍。

---

## 13. 材料来源

官方 PDF、arXiv source、图片文件与版本说明见 [sources.md](assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/sources.md)。本文所有定量结果均以 arXiv v2 为准；论文未披露处已明确标注，未以二手材料补造。