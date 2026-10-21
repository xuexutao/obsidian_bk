---
type: paper-note
status: draft
domain: VLA
paper: OpenVLA: An Open-Source Vision-Language-Action Model
year: 2024
arxiv: 2406.09246
source: https://arxiv.org/abs/2406.09246
project: https://openvla.github.io/
code: https://github.com/openvla/openvla
tags:
  - VLA
  - Open-Source
  - Vision-Language-Action
  - Prismatic-VLM
  - Action-Tokenization
created: 2026-08-27
updated: 2026-08-27
---

# OpenVLA：开源视觉语言动作模型

## 基本信息

- **作者**：Moo Jin Kim*、Karl Pertsch*、Siddharth Karamcheti*、Ted Xiao、Ashwin Balakrishna、Suraj Nair、Rafael Rafailov、Ethan Foster、Grace Lam、Pannag Sanketi、Quan Vuong†、Thomas Kollar、Benjamin Burchfiel、Russ Tedrake、Dorsa Sadigh、Sergey Levine、Percy Liang、Chelsea Finn（* 共同一作）
- **机构**：Stanford University、UC Berkeley、Toyota Research Institute、Google DeepMind、Physical Intelligence、MIT
- **发表**：CoRL 2024（arXiv 2406.09246，v3 2024-09-05）
- **项目页**：<https://openvla.github.io/>
- **代码/权重**：<https://github.com/openvla/openvla>（HuggingFace 可下载 checkpoint）
- **原文 PDF**：<https://arxiv.org/pdf/2406.09246>

## 一句话结论

OpenVLA 是首个**完全开源**的 7B 视觉-语言-动作模型（VLA）：它以 Prismatic VLM（Llama-2 7B + 融合 DINOv2/SigLIP 的视觉编码器 + MLP 投影）为基座，把 7 维连续动作逐维离散化成 256 bins、作为「离散动作 token」塞进语言模型的词表，在 Open X-Embodiment 的 970k 条真实机器人轨迹上训练；zero-shot 泛化平均成功率约 55%，反超闭源的 55B 参数 RT-2-X（约 44%），并在 Franka 上以 15–100 条演示的数据高效微调超越 Diffusion Policy，系统性地验证了「VLM 基座 + 离散动作 token」这一配方，成为后续大量 VLA 与世界模型工作的开源底座和对比基线。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

机器人操作策略最大的痛点在于**无法泛化到训练分布之外**：传统单任务策略能外推到新的物体位置、光照，但对场景干扰物、新物体、未见过的语言指令几乎束手无策。反观视觉与语言领域，CLIP、SigLIP、Llama 2 这类互联网规模预训练的基础模型恰恰具备这种泛化能力。OpenVLA 的核心洞察是一个朴素而有力的类比：机器人领域最大的数据集（Open X-Embodiment）也只有 100K–1M 条轨迹，远不足以复现互联网规模的预训练；与其从头训练机器人策略，不如**直接微调现成的 VLM 去输出动作**，把互联网预训练捕获的先验迁移到机器人控制上。

### 1.2 现有方法的瓶颈

论文明确指出了阻碍 VLA 大规模采用的两个关键原因：

- **现有 VLA 大多闭源、不可访问**：RT-2、RT-2-X、PaLI 系列等虽然效果强，但架构、训练流程、数据配比都对外不可见，社区无法在其上做研究；
- **缺乏高效微调的最佳实践**：已有工作只研究「开箱即用」的 zero-shot 评测，几乎没人回答「如何把 VLA 高效适配到新机器人、新环境、新任务，尤其是在消费级 GPU 上」，而这恰恰是落地采用的关键。

这两点共同导致：机器人领域缺少一个像开源 LLM 生态（LLaMA、Mistral 等）那样的「开源基座」，供后续研究在训练目标、数据配比、推理优化上持续迭代。

### 1.3 本文核心贡献

1. 提出并开源 **OpenVLA**：7B 参数、以 Prismatic VLM（融合 DINOv2 + SigLIP）为基座、在 970k 条 OXE 真实轨迹上训练的首个完全开源 VLA；
2. 系统研究了 **VLA 高效微调**：首次展示 LoRA 与模型量化能让 OpenVLA 在消费级 GPU 上微调与部署，而不损失成功率；
3. 在 BridgeData V2（WidowX）与 Google robot 两个本体上做 zero-shot 评测，**以约 1/7 参数（7B vs 55B）反超闭源 RT-2-X**；
4. 在 Franka 上验证数据高效微调（10–150 演示）优于 Diffusion Policy、Octo 等基线，并开源训练代码、微调 notebook 与全部权重。

![](assets/OpenVLA%20-%20开源视觉语言动作模型/figures/01-overview.png)

> **图 1：OpenVLA 总览。** 左上：以 Llama 2 7B 为基座，把视觉输入与语言指令映射为机器人动作；右上：在 970k 条 OXE 机器人轨迹上大规模训练；下方：支持多机器人开箱即用控制与高效微调，且数据、权重、代码全开源。这张图点明了 OpenVLA 的定位——不是单个任务的策略，而是一个可复用、可微调的开源具身基座。
> 来源：论文 Figure 1，第 1 页，<https://arxiv.org/abs/2406.09246>

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：单张当前观测图像 $\mathbf{I}$（第三视角相机）+ 一条自然语言任务指令（如 "Put eggplant in bowl"）；
- **输出**：7 维机器人控制动作 $\mathbf{a} = (\Delta x, \Delta y, \Delta z, \Delta \theta_x, \Delta \theta_y, \Delta \theta_z, \text{grip})$，即末端执行器的相对位姿变化（6 维）+ 夹爪开合（1 维）；
- **核心假设**：把「动作预测」重新表述为一个「视觉-语言」任务——输入图像 + 指令，映射到一串预测出来的机器人动作 token，从而完全复用 VLM 的 next-token 训练范式，无需为机器人控制定制任何新架构。

### 2.2 动作离散化：连续动作 → 离散 token

要让语言模型主干输出动作，OpenVLA 沿用 RT-2 的思路，把连续动作离散化为词表内的 token。具体做法：

- 对 7 维动作的**每一维**独立离散化，映射到 256 个 bin 之一，即每一维是一个整数 $t_i \in [0, 255]$，整个 7 维动作变成 7 个离散 token；
- **关键改进**：bin 的划分区间用训练数据中该维动作的 **1% 分位数到 99% 分位数**均匀等分，而非 RT-2 用的 min–max 边界。用分位数可以忽略数据里的离群动作（outliers），避免离群值把离散区间拉得过宽、稀释动作粒度；
- 由于 Llama 2 词表只预留了 100 个新 token 位置，装不下 256 个动作 token，OpenVLA 采取最简单粗暴的做法：直接**覆写 Llama 词表中使用频率最低的 256 个 token**（即词表末尾的 256 个）作为动作 token。

### 2.3 训练目标

动作被处理成 token 序列后，OpenVLA 用标准的 **next-token prediction 目标**训练，交叉熵损失**只在预测出的动作 token 上计算**（不监督文本 token 的预测）。这保证了梯度信号集中在对机器人控制真正重要的动作预测上。

---

## 3. 核心方法

### 3.1 总体框架

OpenVLA 由三部分组成（Figure 2）：
(1) **视觉编码器**，把输入图像映射为一组图像 patch embedding；
(2) **投影器**，把视觉特征映射到语言模型的输入空间；
(3) **LLM 主干**，一个 7B 参数的 Llama 2。
训练时把动作预测当作 VLM 的「视觉问答」，输入图像 + 指令 prompt（格式为 "What should the robot do to {task}? A:"），模型自回归输出动作 token，再经 action de-tokenizer 还原为 7 维连续动作。

![](assets/OpenVLA%20-%20开源视觉语言动作模型/figures/02-architecture.png)

> **图 2：OpenVLA 架构图。** 给定图像观测与语言指令，模型预测 7 维机器人控制动作。三个关键组件：(1) 视觉编码器，把 DINOv2 与 SigLIP 的特征在通道维拼接融合；(2) MLP 投影器，把视觉特征映射到语言嵌入空间；(3) LLM 主干（Llama 2 7B），自回归生成动作 token，再由 de-tokenizer 还原成 $(\Delta x, \Delta y, \Delta z, \Delta\theta, \text{grip})$ 等 7 维动作。
> 来源：论文 Figure 2，第 4 页，<https://arxiv.org/abs/2406.09246>

### 3.2 基座：Prismatic VLM 与视觉编码器融合

OpenVLA 的基座是 Karamcheti et al. 的 **Prismatic-7B VLM**，它遵循标准的 VLM 三件套：600M 参数视觉编码器 + 2 层 MLP 投影器 + 7B Llama 2 主干。Prismatic 最特别之处在于它的**两段式视觉编码器**：输入图像 patch 分别经过预训练的 **SigLIP** 与 **DINOv2**，然后把两个编码器输出的特征向量在**通道维直接拼接**。

这样设计背后的直觉是：SigLIP 捕获高层语义（物体是什么），DINOv2 捕获低层空间结构（物体在哪里、边界在哪）。相比只用 CLIP 或 SigLIP 的常见做法，**融合 DINOv2 特征能显著增强空间推理能力**，而这对机器人控制（需要精确定位、抓取）尤其重要。作者在消融中证实：Prismatic 基座比 LLaVA 基座在单物体与多物体语言定位任务上平均高约 10%，比 IDEFICS-1 高约 35%（见 5.4 节）。

### 3.3 关键设计决策消融

在正式训练前，作者先在 BridgeData V2 上做小规模实验，逐项确定设计：

- **VLM 基座**：对比 IDEFICS-1、LLaVA、Prismatic。LLaVA 与 IDEFICS-1 在单物体任务上相当，但多物体语言定位任务上 LLaVA 比 IDEFICS-1 高 35%；Prismatic 又比 LLaVA 高约 10%。归因于 DINOv2+SigLIP 融合带来的空间推理提升，最终选定 Prismatic；
- **图像分辨率**：224×224 与 384×384 性能无差异，但 384 训练慢 3 倍（patch token 数更多、上下文更长、计算量平方增长），故选 224×224。作者也指出这不同于 VLM 里「分辨率越高越好」的规律，VLA 里暂时没看到这个趋势；
- **是否微调视觉编码器**：VLM 训练里通常冻结视觉编码器能保特征、性能更好；但 VLA 训练里**微调视觉编码器至关重要**——作者假设预训练视觉 backbone 缺乏机器人控制所需的精细空间细节，需要继续适配；
- **训练轮数**：LLM/VLM 通常只训 1–2 个 epoch，但 VLA 需要**反复多轮**迭代数据集，真实机器人性能持续提升直到动作 token 准确率超过 95%。最终模型跑了 **27 个 epoch**；
- **学习率**：多数量级扫描后，最佳是固定的 $2\times10^{-5}$（与 VLM 预训练一致），且 warmup 无收益。

### 3.4 训练数据与数据治理

训练目标是最大化「本体、场景、任务」的多样性。OpenVLA 以 Open X-Embodiment（OXE）为基座——写作时 OXE 含 70+ 个机器人数据集、超 2M 条轨迹。作者做了两层治理：

1. **统一输入输出空间**：只保留「有至少一个第三视角相机 + 单臂末端执行器控制」的操作类数据集（与 RT-X、Octo 一致）；
2. **平衡数据配比**：沿用 Octo 的 mixture weights，对多样性低的数据集降权、对任务/场景多样性的数据集升权。

此外尝试加入了 OXE 后续新增的 DROID 数据集（保守权重 10%），但发现训练全程其动作 token 准确率偏低（需要更大权重或更大模型才能拟合），于是在最后 1/3 训练阶段移除。最终得到 **970k 条轨迹**的训练集（附录 A 有完整数据集与权重清单）。作为对比，RT-2-X 的训练集约 350k 条——更大的数据规模也是 OpenVLA 反超 RT-2-X 的重要原因之一。

### 3.5 训练与推理基础设施

最终模型在 **64 块 A100 GPU 上训练 14 天**，共 **21,500 A100-小时**，batch size 2048。推理侧：bfloat16 精度加载需 15GB 显存，单块 RTX 4090 上约 **6Hz**（不做编译、投机解码等加速）；量化后显存进一步压缩（见 5.3 节）。作者还实现了远程推理服务器，把动作预测流式推送到机器人，让控制端不必持有大算力设备。

---

## 4. 数据集与实验设置

### 4.1 训练数据

如上所述，训练集是从 OXE 治理出的 **970k 条真实机器人轨迹**，覆盖 WidowX、Franka、Google robot 等众多本体、任务与场景。输入统一为单张 224×224 第三视角图像 + 语言指令；动作为 7 维相对位姿控制。与 Octo 相比，OpenVLA 是端到端直接微调 VLM 输出动作 token，而非用「预训练组件 + 从零初始化的额外模块」去拼接，这是架构上的根本差异。

### 4.2 评测设置与基线

zero-shot「开箱即用」评测在两个本体上进行：

- **BridgeData V2 WidowX**（Figure 3）：17 个任务 × 10 次 = **170 次 rollout**，覆盖视觉泛化（未见背景/干扰物/外观）、运动泛化（未见位置/朝向）、物理泛化（未见尺寸/形状）、语义泛化（未见物体/指令/概念）与语言定位能力；
- **Google robot**（移动操作臂，Figure 4）：12 个任务 × 5 次 = **60 次 rollout**，分 in-distribution 与 OOD 任务。

所有评测采用 **A/B 对照**（相同任务、相同初始状态）保证公平。对比三个先前通用策略：**RT-1-X**（35M，从零训练的 transformer）、**Octo**（93M，开源策略里的 SOTA）、**RT-2-X**（55B，闭源 VLA）。

### 4.3 微调与效率实验设置

- **数据高效微调**：Franka-Tabletop（桌面固定 Panda 7-DoF 臂，5Hz 控制器）与 Franka-DROID（DROID 数据集的 Franka 平台，15Hz 控制器），每任务 10–150 条演示，全参数微调用 8 块 A100 训 5–15 小时；
- **对比基线**：从零训练的 Diffusion Policy、Diffusion Policy (matched)（匹配 OpenVLA 输入输出规格）、微调的 Octo、以及 **OpenVLA (scratch)**（直接微调底层 Prismatic VLM、跳过 OXE 预训练，用于评估大规模机器人预训练的收益）；
- **PEFT 与量化**：全参数微调 vs 仅末层 / 冻结视觉 / sandwich / LoRA（rank 32、64）；推理精度 bfloat16 vs int8 vs int4。

---

## 5. 实验结果

### 5.1 zero-shot 泛化：反超闭源 RT-2-X

**BridgeData V2 WidowX（Figure 3）**：OpenVLA 取得最高的综合性能，在**除语义泛化外的所有类别**都反超闭源 RT-2-X。语义泛化上 RT-2-X 略胜，这是预期内的——RT-2-X 用了更大规模的互联网预训练，且训练时同时混入互联网数据（co-fine-tune）以保留预训练知识，而 OpenVLA 只在机器人数据上微调。综合来看，OpenVLA 的 zero-shot 平均成功率约 **55%**，高于 RT-2-X 的约 **44%**（论文摘要口径：29 个任务上相对 RT-2-X 有 **16.5% 的绝对成功率提升**，且参数量只有其 **1/7**，7B vs 55B）。

**Google robot（Figure 4）**：OpenVLA 与 RT-2-X 表现相当，均显著优于 RT-1-X 与 Octo。RT-1-X 与 Octo 在干扰物存在时常常抓错物体、甚至无目的地挥臂——这正说明**没有互联网预训练的模型难以应对更大的泛化压力**，而 VLM 基座的价值在此凸显。定性上，RT-2-X 与 OpenVLA 都表现出明显更稳健的行为：在干扰物中靠近正确物体、对齐末端朝向、甚至能从「没抓稳」的失误中恢复。

作者把 OpenVLA 反超 RT-2-X 归因于三点：(1) 训练集更大（970k vs 350k）；(2) 数据清洗更细致（如过滤掉 Bridge 数据集中的全零动作）；(3) 融合了语义与空间特征的 DINOv2+SigLIP 视觉编码器。

![](assets/OpenVLA%20-%20开源视觉语言动作模型/figures/03-results.png)

> **图 3：BridgeData V2 WidowX 评测任务与结果。** 横轴是覆盖多个泛化维度（未见背景/干扰物/外观、未见位置/朝向、未见尺寸/形状、未见物体与指令、语言定位）的任务，纵轴为成功率。OpenVLA（深色柱）整体最高，除语义泛化外全面领先闭源 RT-2-X（浅色柱），RT-1-X 与 Octo 则明显落后。每个方法共 170 次 rollout。
> 来源：论文 Figure 3，第 7 页，<https://arxiv.org/abs/2406.09246>

### 5.2 数据高效微调：超越 Diffusion Policy

在 Franka 上做 10–150 演示的微调实验（Figure 5）：

- 在**窄的单指令任务**（如 "Put Carrot in Bowl"）上，从零训练的 Diffusion Policy 与 OpenVLA/Octo 相当甚至更优；
- 但在**含多物体、需语言定位的多样化多指令任务**上，预训练通用策略（Octo、OpenVLA）明显更好，因为 OXE 预训练赋予了语言接地能力——**OpenVLA (scratch)**（跳过 OXE 预训练）性能更低，直接证明了大规模机器人预训练的价值；
- 综合来看，**OpenVLA 平均成功率最高，且是唯一在所有测试任务上都达到 ≥50% 成功率的方法**，说明它可以作为模仿学习任务（尤其含多样语言指令时）的稳健默认选择；
- 对更精细、灵巧的任务，Diffusion Policy 仍能给出更平滑精确的轨迹——作者指出 OpenVLA 目前缺少 action chunking 与时间平滑，是未来改进方向。

### 5.3 参数高效微调与量化

**PEFT（Table 1）**：只微调末层（30.3%）或冻结视觉编码器（47.0%）都显著变差，说明「视觉特征继续适配目标场景」不可或缺；sandwich 微调（62.1%）因微调了视觉编码器而更好；**LoRA（rank=32）取得最佳性能-算力权衡：只训练 1.4% 参数（97.6M）就追平全参数微调（68.2% vs 69.7%）**，且 rank 对性能影响可忽略，推荐默认 rank=32。用 LoRA 可在**单块 A100 上 10–15 小时**完成新任务微调，相比全参数微调算力降低 8 倍。

**量化推理（Table 2）**：bfloat16（71.3% 成功率 / 16.8GB）、int8（58.1% / 10.2GB）、int4（71.9% / 7.0GB）。有意思的是 **4-bit 量化不仅显存减半以上，成功率还追平 bfloat16**；而 8-bit 反而掉点——原因是 8-bit 量化引入额外算子开销、推理变慢（A5000 上只有 1.2Hz），改变了系统动力学，与训练时的 5Hz 控制器失配；4-bit 因显存传输减少反而吞吐更高（3Hz），更贴合采集时的动力学。

### 5.4 关键消融小结

综合正文与附录的消融结论，OpenVLA 的核心配方可概括为五点：**(1) 视觉编码器融合**（DINOv2+SigLIP 优于单一 SigLIP，提升空间推理与语言定位）；**(2) 离散动作 token**（把动作离散化进词表，比连续回归更能复用 VLM 的 next-token 训练范式和互联网先验）；**(3) 基座 VLM 的选择**（Prismatic 优于 LLaVA、IDEFICS-1）；**(4) 大规模机器人预训练**（OpenVLA vs OpenVLA(scratch) 证明 OXE 970k 预训练是关键）；**(5) 微调视觉编码器 + 多轮训练**（VLA 不同于 VLM，需解冻视觉、跑 27 个 epoch）。

---

## 6. 与相关工作的关系

按时间谱系梳理，OpenVLA 处在「从闭源 VLA 到开源具身基座」的分水岭上：

- **vs RT-1 / RT-2**：RT-1（2022）首次把动作当 token 用 transformer 从零训练，但单机器人、闭源、无互联网先验；RT-2（2023）把 PaLI/PaLM-E 这样的 VLM 与机器人数据 + 互联网数据 co-fine-tune，验证了 VLA 范式，但闭源。OpenVLA 继承了 RT-2 的「离散动作 token」与 next-token 思路，但改用**完全开源**的 Llama 2 + DINOv2/SigLIP，并在动作 bin 划分上从 min–max 改进为分位数；
- **vs RT-2-X / Octo**：RT-2-X（2023）是 OXE 上训练的 55B 闭源 VLA，Octo（2023）是 93M 开源策略但用「预训练组件 + 从零模块拼接」而非端到端微调 VLM。OpenVLA 以 7B 端到端微调 VLM，用更大、更干净的 970k 数据反超 RT-2-X、远超 Octo；
- **OpenVLA 的历史定位**：它是首个把「7B VLA 的权重 + 训练代码 + 微调 notebook」全部开源的工作，成为后续大量 VLA / 世界模型工作的**底座与对比基线**。之后 Physical Intelligence 的 **π0**（2024）把离散 token 换成 flow matching 连续动作、加入动作专家并联合 VLM 训练，在灵巧操作上更进一步，但其对比与社区评测里 OpenVLA 仍是常设参照——可以说 OpenVLA 定义了「VLM 基座 + 动作头」这一配方在开源社区的起点，π0 则代表这一路线向更大数据、连续动作、联合训练的下一跳。

---

## 7. 局限与批判性评价

1. **只支持单帧观测**：OpenVLA 只能吃单张图像，没有本体感受（proprioception）与观测历史，而真实机器人环境高度异构（多相机、多模态）。扩展到多图像、本体感受、历史帧是明确的方向；用「交错图文数据」预训练的 VLM 或能天然支持这种灵活输入；
2. **推理吞吐不足**：6Hz 的推理频率无法支撑 ALOHA 这类 50Hz 的高频控制，也限制了向更灵巧的双手操作任务扩展；action chunking 或投机解码（speculative decoding）是潜在解药；
3. **可靠性仍有天花板**：虽然反超先前通用策略，但多数任务成功率 <90%，离可靠部署仍有距离；
4. **设计问题未穷尽**：受算力所限，基座 VLM 大小的影响、机器人动作数据与互联网 VL 数据 co-training 的收益、最适合 VLA 的视觉特征等关键问题都还是开放问题——这也是作者寄望于开源社区共同探索的动机；
5. **离散化信息损失**：7 维动作离散到 256 bins 本质上是粗粒度的有损编码，且动作 token 是「覆写词表末 256 个 token」的权宜之计，词表冲突可能带来副作用；后续 π0 改用 flow matching 连续动作正是对这一点的回应；
6. **数据仍是小规模**：970k 轨迹相对互联网规模仍小，且 RT-2-X 在语义泛化上更优，说明「只在机器人数据上微调、不做互联网数据 co-training」会牺牲一部分预训练知识保留。

---

## 8. 复现与实践建议

- **算力门槛**：预训练需 64×A100 14 天（21,500 A100-小时，batch 2048）；全参数微调 8×A100 5–15 小时；**LoRA 微调单块 A100 10–15 小时即可**；推理 bf16 需 15GB 显存（RTX 4090 约 6Hz），int4 量化后降到 7GB；
- **数据准备**：训练直接用 OXE（HuggingFace 可下），注意按 Octo mixture weights 治理、只保留单臂 + 第三视角相机的操作数据；微调自备 10–150 条目标任务演示即可；
- **代码可获得性**：GitHub `openvla/openvla` 已开源，含模块化 PyTorch 训练代码（AMP、FlashAttention、FSDP）、HuggingFace AutoModel 集成、LoRA 微调与量化推理、远程推理服务器；
- **实现要点**：
  - 动作离散化用 **1%–99% 分位数**划分 256 bins（不要用 min–max，会被离群值稀释）；
  - 词表不够时直接**覆写 Llama 词表末 256 个低频 token**；
  - 损失**只在动作 token 上**算交叉熵；
  - 一定要**解冻视觉编码器**微调，且**多轮迭代**（OpenVLA 跑了 27 个 epoch，动作 token 准确率要 >95%）；
  - 学习率用固定的 $2\times10^{-5}$，无需 warmup；
- **常见踩坑**：不要用 384×384 分辨率（慢 3 倍、无收益）；int8 量化可能因推理变慢导致掉点，优先用 int4；微调时 LoRA rank 对性能影响很小，直接用默认 rank=32。

---

## 9. 个人启发与后续问题

1. **「离散动作 token」是打通 VLM 与机器人控制的最小改动**：OpenVLA 几乎没有为机器人定制任何新架构，只是把动作离散化塞进词表，就完整继承了 VLM 的预训练先验与训练/推理基础设施。这种「复用现有基础模型 + 最小改动」的工程哲学，是它在当时能迅速被社区接受并成为基座的根本原因；
2. **开源是生态的分水岭，不只是权重**：OpenVLA 同时开源了权重、训练代码、微调 notebook 与数据治理方案，这比单纯放权重更能催生下游工作——它之后成为大量 VLA / 世界模型工作的底座与对比基线，是「开源具身基座」的开端；
3. **从离散到连续：π0 的接力**：OpenVLA 验证了「VLM 基座 + 动作头」配方，但其离散动作 token 的有损、低吞吐问题，恰好被后续 π0 的 flow matching 连续动作、动作专家与 VLM 联合训练所解决。把 OpenVLA 与 π0 对照着读，能清晰看到「数据规模、动作表征、联合训练」三个维度上这一路线的演进脉络；
4. **待追问的问题**：(i) 离散动作 token 的信息损失到底多大？换成连续动作表征（flow matching / diffusion）在同等算力下能涨多少？(ii) 为什么 VLA 里「分辨率越高越好」的规律失效，而 VLM 里成立？(iii) 机器人与互联网 VL 数据 co-training 能否在不牺牲泛化的前提下提升语义泛化？(iv) 如何把本体感受与观测历史无痛地纳入 VLM 范式？

---

## 10. 材料来源

### Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2406.09246> | 全文（正文 + 附录） | 2026-08-27 | 精读主来源 |
| `figures/01-overview.png` | Figure 1：OpenVLA 总览（7B VLA + 真实机器人泛化） | arXiv:2406.09246 | 第 1 页 | 2026-08-27 | 概念总览 |
| `figures/02-architecture.png` | Figure 2：OpenVLA 架构图（DINOv2+SigLIP 融合 → LLM → 离散动作 token） | arXiv:2406.09246 | 第 4 页 | 2026-08-27 | 核心方法 |
| `figures/03-results.png` | Figure 3：BridgeData V2 WidowX 评测任务与结果 | arXiv:2406.09246 | 第 7 页 | 2026-08-27 | 关键定量结果 |
| 项目页 <https://openvla.github.io/> | 视频/补充材料 | — | — | 2026-08-27 | 实验视频与 rollout |
| 代码 <https://github.com/openvla/openvla> | 官方实现 | — | — | 2026-08-27 | 复现参考 |
