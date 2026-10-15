---
type: paper-note
status: complete
domain: VLM
paper: "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"
year: 2023
arxiv: "2312.14238"
doi: null
source: "https://arxiv.org/abs/2312.14238"
project: null
code: "https://github.com/OpenGVLab/InternVL"
tags:
  - VLM
  - vision-foundation-model
  - contrastive-learning
  - multimodal-alignment
  - InternVL
created: 2026-10-15
updated: 2026-10-15
---

# InternVL：规模化视觉基础模型与通用对齐

## 基本信息

- **论文定位**：InternVL 系列的起点工作。核心不是今天常见的“ViT + MLP + LLM”聊天模型，而是先训练一个 **6B 视觉基础模型 InternViT-6B**，再用 **8B、由多语言 LLaMA 初始化的 QLLaMA** 做重型语言中间件，覆盖视觉感知、对比检索、生成和多模态对话。
- **主要组件**：InternViT-6B、QLLaMA、可选的 Vicuna/InternLM 解码器。
- **训练规模**：阶段 1 使用清洗后的 4.98B 图文对并累计见过 28.7B 样本；阶段 2 从中筛到 1.03B 高质量图文对；阶段 3 使用约 4M 指令样本。
- **重要性判断**：★★★★☆。它建立了 InternVL 的视觉底座和“先大规模视觉—语言对齐、再接聊天 LLM”的路线；但原始 QLLaMA 方案训练与部署极重，后来系列转向更直接的 ViT–MLP–LLM 架构。

**一句话结论：**

InternVL 的关键贡献不是单纯把 ViT 做到 6B，而是证明：**大视觉编码器、与 LLM 表示空间一致的重型语言中间件、按数据质量逐级切换目标函数**可以共同得到一个兼顾感知、检索、生成与对话的通用视觉语言底座；其真正遗产是 InternViT 和渐进式对齐思想，而非 QLLaMA 这一具体连接形态。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

论文观察到 2023 年的 LLM 已扩展到数十亿乃至更大规模，但多模态系统常用的视觉编码器仍约在 1B 以下，并通过随机初始化的线性层或 Q-Former 接入 LLM。作者将瓶颈归纳为三点：视觉与语言组件参数规模失衡、视觉特征与 LLM 表示空间不一致、轻量连接层不足以表达复杂跨模态依赖。

因此，本文要解决的并非单一 VQA，而是一个基础模型问题：能否训练一个视觉侧规模与 LLM 相称、特征已向 LLM 对齐、又能被不同任务复用的视觉语言底座？它既应作为视觉 backbone 输出稠密特征，也应支持图文对比学习、文本生成和开放式对话。

![](assets/InternVL%20-%20规模化视觉基础模型与通用对齐/figures/01-motivation.png)

> 图 1：从监督视觉预训练、CLIP 式双塔对比预训练，到 InternVL 的“6B 视觉编码器 + 大语言中间件 + 生成式 LLM”方案。重点是同时扩大视觉侧并直接面向 LLM 表示对齐。来源：论文 Figure 1，第 1 页。

### 1.2 现有方法的瓶颈

轻量 glue layer 的优点是便宜，但它把跨模态对齐压力压缩在很少参数中；而视觉编码器若主要由 ImageNet/JFT 或 BERT 对齐目标训练，其特征分布也未必天然适合自回归 LLM。另一方面，互联网图文数据规模大但噪声高：对比学习可以容忍弱配对，生成训练却会直接学习坏 caption。InternVL 因而把“模型规模”“表示一致性”“数据质量—目标函数匹配”作为同一个系统问题处理。

### 1.3 核心贡献

1. 将 vanilla ViT 扩展为约 6B 参数的 InternViT-6B，并验证其在线性探测、分割和跨域分类上的视觉表征质量。
2. 设计由多语言 LLaMA-7B 初始化的 QLLaMA：保留语言模型主体，新增 96 个可学习 query 与约 1B 随机初始化的 cross-attention 参数。
3. 提出三阶段渐进对齐：海量噪声图文对做对比训练，高质量图文对做生成式对齐，最后用约 4M 指令数据接入聊天 LLM。
4. 用同一底座覆盖 32 个视觉—语言 benchmark，展示“视觉感知—检索—描述—问答—对话”的广度。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与四种使用方式

给定图像 $I\in\mathbb{R}^{H\times W\times3}$，InternViT 产生特征图：

$$F=\operatorname{InternViT}(I),\quad F\in\mathbb{R}^{H/14\times W/14\times D}$$

这一路输出可直接供分类、语义分割等视觉任务使用。对比任务中，视觉特征经过 attention pooling 得到全局向量 $I_f$，文本由 QLLaMA 的 `[EOS]` 特征得到 $T_f$，二者余弦相似度用于零样本分类或检索。生成任务中，96 个 query 经 cross-attention 重组视觉特征，作为 QLLaMA 的前缀条件，自回归生成 caption。对话任务则进一步把视觉表示经 MLP 接到 Vicuna/InternLM。

论文区分两个对比推理版本：**InternVL-C** 主要使用 InternViT；**InternVL-G** 进一步使用 QLLaMA 编码视觉特征。后者在阶段 2 后通常有更强图文对齐。

### 2.2 关键训练目标

阶段 1 采用 CLIP 式对称交叉熵。设 batch 内归一化后的图像、文本嵌入为 $I_i,T_i$，温度为 $\tau$，则图到文方向可写为：

$$\mathcal{L}_{i\rightarrow t}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(I_i^\top T_i/\tau)}{\sum_{j=1}^{N}\exp(I_i^\top T_j/\tau)}$$

文本到图方向同理，最终：

$$\mathcal{L}_{\mathrm{ITC}}=\frac{1}{2}(\mathcal{L}_{i\rightarrow t}+\mathcal{L}_{t\rightarrow i})$$

阶段 2 遵循 BLIP-2，把三类信号相加：

$$\mathcal{L}_{\mathrm{stage2}}=\mathcal{L}_{\mathrm{ITC}}+\mathcal{L}_{\mathrm{ITM}}+\mathcal{L}_{\mathrm{ITG}}$$

其中 ITM 判断图文是否匹配，ITG 是图像条件下的 next-token loss：

$$\mathcal{L}_{\mathrm{ITG}}=-\sum_{t=1}^{L}\log p(y_t\mid y_{<t},I)$$

这种安排的直觉是：ITC 建立全局几何结构，ITM 学更细的配对边界，ITG 迫使 query 保留足以生成文本的语义细节。

---

## 3. 核心方法

### 3.1 InternViT-6B：把视觉侧真正扩到 LLM 量级

InternViT-6B 使用 patch size 14 的标准 ViT。作者搜索深度 $\{32,48,64,80\}$、head dimension $\{64,128\}$ 和 MLP ratio $\{4,8\}$，最终选用 width 3200、depth 48、MLP hidden size 12800、25 heads 的变体，参数量约 5.9B。搜索结果显示，在约 6B 总参数固定时，这些形状对 100M LAION-en 子集上的零样本 ImageNet 差异很小；选择更多受吞吐与稳定性驱动，而不是“更深必然更好”。

这一设计与 ViT-22B 的方向不同：InternVL 并不只追求纯视觉规模，而是希望视觉特征从一开始就与语言空间联合训练。阶段 1 中 InternViT 随机初始化，说明其能力主要来自 4.98B 清洗图文数据和巨量有效 batch，而不是继承已有 CLIP。

### 3.2 QLLaMA：重型而非轻型的跨模态中间件

QLLaMA 由多语言 LLaMA-7B 初始化，并插入 96 个 learnable queries 与 cross-attention 层；新增部分约 1B 参数，QLLaMA 总体约 8B。它有三重角色：作为文本编码器参加图文对比；通过 query cross-attention 从 InternViT 特征中抽取语言相关视觉信息；作为生成模型直接完成 caption，或把视觉前缀交给外部聊天 LLM。

作者声称它比约 188M 的 Q-Former 大 42 倍。优势是表示空间天然贴近 LLM，且不依赖随机小投影器承担全部对齐；代价是训练和推理都非常重，视觉侧 6B 加中间件 8B 后，即使最终聊天 LLM 冻结，总系统也不是轻量 VLM。

### 3.3 三阶段渐进式对齐

![](assets/InternVL%20-%20规模化视觉基础模型与通用对齐/figures/02-training-strategy.png)

> 图 3：InternVL 的三阶段训练。阶段 1 全量训练 InternViT 与 LLaMA 文本编码器；阶段 2 冻结二者，只训练 query/cross-attention；阶段 3 接入 Vicuna，训练 MLP 与可选的 QLLaMA 参数。来源：论文 Figure 3，第 4 页。

#### 3.3.1 阶段 1：对比预训练

原始数据合计 6.03B 图文对，清洗后保留 4.98B，来自 LAION-en、LAION-multi、LAION-COCO、COYO、CC12M、CC3M、SBU 和中文 Wukong。InternViT 与 LLaMA 文本编码器均可训练；总 batch 164K，175K iterations，累计 28.7B seen samples，使用 640 张 A100 80G。先以 $196^2$ 分辨率训练，最后 0.5B 样本切到 $224^2$，以降低超大规模训练成本。

#### 3.3.2 阶段 2：生成式预训练

数据进一步筛到 1.03B，尤其大幅削减原始 LAION-en 和 LAION-multi，只保留更适合生成的 caption。InternViT 和从阶段 1 继承的 QLLaMA 主体被冻结，仅训练约 1B 的 query/cross-attention；总 batch 20K、80K iterations、累计 1.6B seen samples，使用 160 张 A100 80G。这里体现了论文最重要的数据原则：**不是所有可用于对比学习的数据都适合生成学习**。

#### 3.3.3 阶段 3：监督指令微调

约 4M 样本由 588K caption、1.1M VQA、294K OCR、323K grounding、284K grounded caption 和 1.4M conversation 组成。视觉表示经 MLP 接入 Vicuna-7B/13B 或 InternLM。由于 QLLaMA 已在 LLM 表示空间中，作者可以冻结聊天 LLM，仅训练 MLP 或 MLP+QLLaMA，从而尽量保留语言能力。

### 3.4 推理复杂度与“瑞士军刀”属性

论文把可组合性视作优势：只取 InternViT 做视觉 backbone；用 InternViT/QLLaMA 做检索；用 QLLaMA 生成 caption；或再接 LLM 做对话。但从系统工程角度看，这种通用性通过多个大组件实现，计算并未统一。尤其 dense vision 仍需 $H/14\times W/14$ token，QLLaMA 又有 8B 参数，因此“一个模型覆盖所有任务”不等于“一条低成本推理路径覆盖所有任务”。

---

## 4. 数据集与实验设置

### 4.1 数据与阶段配比

| 阶段 | 数据规模 | 训练目标 | 主要可训练参数 | 资源 |
|---|---:|---|---|---|
| Stage 1 | 4.98B 清洗图文对；28.7B seen samples | 对称 ITC | InternViT-6B + LLaMA-7B，共 13B | 640×A100 80G |
| Stage 2 | 1.03B 高质量图文对；1.6B seen samples | ITC + ITM + ITG | 1B query/cross-attention；其余冻结 | 160×A100 80G |
| Stage 3 | 约 4M 指令样本 | 自回归 SFT | MLP，或 MLP+QLLaMA | 论文正文未统一披露 |

阶段 1 的峰值学习率分别为图像编码器 $10^{-3}$、文本编码器 $10^{-4}$；阶段 2 cross-attention 为 $5\times10^{-5}$。两阶段均为 AdamW、cosine decay、bf16。

### 4.2 Baseline 与指标

视觉表征对比包括 ViT-22B、EVA-02、DINOv2、OpenCLIP；VLM 对比包括 BLIP-2、InstructBLIP、Qwen-VL、LLaVA-1.5、IDEFICS。指标横跨 ImageNet top-1、ADE20K mIoU、Kinetics 分类、Flickr/COCO Recall、CIDEr、VQA accuracy、MME 与 POPE。广度是优点，但不同方法的预训练数据、输入分辨率与可训练参数并不完全等价，因此“32 个 benchmark SOTA”应按各表的设定逐项理解。

---

## 5. 实验结果

### 5.1 视觉感知与对比任务

InternViT-6B 在 ADE20K 线性探测达到 47.2 mIoU，比论文报告的 ViT-22B 高 12.6；这说明图文对比预训练没有牺牲像素级迁移。零样本视频分类中，单帧设置在 Kinetics-400/600/700 分别为 76.1%、75.5%、67.5%，相对 EVA-02-CLIP-E+ 分别提升 6.3、6.2、4.1 个点；8 帧输入还能至少再提升 3.3 点。

InternVL-C 在多语言分类和图文检索上表现强，阶段 2 后的 InternVL-G 进一步提高检索，支持 QLLaMA 的 query 重组确实增加了语言相关信息，而非只增加参数。

### 5.2 生成与对话结果

![](assets/InternVL%20-%20规模化视觉基础模型与通用对齐/figures/03-generic-task-results.png)

> 图 2：论文汇总的分类、检索、caption 和对话结果。它最有价值的信息是任务覆盖宽度，而不是把所有柱状值视为严格同分布比较。来源：论文 Figure 2，第 2 页。

在统一的高配 InternVL-Chat（InternViT-6B + QLLaMA + Vicuna-13B，336 分辨率）中，论文报告 COCO CIDEr 146.2、Flickr30K 92.2、NoCaps 126.2、VQAv2 81.2、GQA 66.6、VizWiz 58.5、TextVQA 61.5、MME 1586.4、POPE 87.6。对比 LLaVA-1.5-13B 的 VQAv2 80.0、GQA 63.3、VizWiz 53.6、TextVQA 61.3、MME 1531.3、POPE 85.9，优势主要体现在 GQA、VizWiz、MME 与 POPE；TextVQA 基本持平。

应注意表中带星号的任务训练标注曾被模型见过，且 InternVL 使用约 1B 阶段 2 预训练样本与 4M SFT，远多于 LLaVA-1.5 的 558K+665K。因此结果证明的是“整套规模化方案有效”，不能单独归因给 QLLaMA。

### 5.3 消融实验

最关键的组件消融在只用 LLaVA-Mix-665K、仅训练 MLP 的简化设定下进行：EVA-E+MLP+Vicuna-7B 的 MME/NoCaps/OKVQA/VizWiz/GQA 为 970.5/75.1/40.1/25.5/41.3；换成 InternViT-6B 后变为 1022.3/80.8/42.9/28.3/45.8；再将 MLP 换为 QLLaMA 后升到 1227.5/94.5/51.0/38.4/57.4。它支持两个结论：InternViT 本身更强；QLLaMA 对多项生成任务的增益大于单纯更换视觉 backbone。

InternViT 形状消融中，六个约 6B 变体的零样本 ImageNet 仅在 65.8–66.2 之间，说明在固定参数与短程训练下，width/depth/MLP ratio 不是主导因素。最终变体 3 不是绝对准确率赢家，而是在稳定性与吞吐之间折中。

### 5.4 失败边界

论文没有系统展示失败案例。可以从实验设计推断出三类边界：输入分辨率主要为 224/336，细粒度 OCR 的上限受限；对话表现依赖 4M SFT 和外部 LLM；超大规模公开图文数据虽可访问，复现清洗、去重和训练吞吐并不现实。这里前两项可由配置直接观察，第三项是复现判断，而非作者的显式实验结论。

---

## 6. 与后续 InternVL 路线的关系

### 6.1 原始 InternVL 不是后来常见的 ViT–MLP–LLM

必须把两代架构分开：

| 路线 | 视觉到语言的连接 | 训练重点 | 系统特征 |
|---|---|---|---|
| 原始 InternVL（本文） | InternViT-6B → QLLaMA（96 queries + cross-attention）→ 可选 MLP → Vicuna/InternLM | 先训练大视觉—语言基础模型，再构建聊天模型 | 重型中间件，可独立做对比/生成 |
| 后续 InternVL Chat / InternVL2–3.x | InternViT → pixel-shuffle/MLP projector → LLM | 动态高分辨率、原生多模态预训练、SFT/RL | 路径更直接，便于端到端扩展与部署 |

也就是说，本文的“aligning with LLM”主要通过 LLaMA 初始化的 QLLaMA 完成；后续系列则把视觉 token 更直接地投影进 LLM，并把能力提升重心转向分辨率组织、数据配方、原生多模态预训练与强化学习。若把本文直接概括成“ViT-MLP-LLM”，会抹掉其最独特也最昂贵的设计。

### 6.2 哪些思想被保留

被保留的是 InternViT 视觉底座、视觉与语言联合扩展、数据质量分层和多阶段训练；被弱化或替换的是 8B QLLaMA 作为通用中间件的方案。后续转型说明：QLLaMA 在早期证明“重对齐层有效”，但当 LLM、数据和端到端训练成熟后，简单 projector 具有更好的扩展与部署性。

---

## 7. 局限、未来工作与批判性评价

### 7.1 论文明确与未明确的局限

论文没有独立 limitations 章节，也没有量化训练总能耗、数据清洗误差、社会偏差与安全性。作者只在结论中表达希望推动 VLLM 社区，未给出具体 future-work 清单。因此以下评价需明确视为个人判断：

1. **因果归因弱**：模型规模、数据规模、QLLaMA、三阶段目标同时变化，组件消融只在较小 SFT 设定下进行，无法完整分离每项贡献。
2. **训练成本极高**：阶段 1 使用 640×A100 80G、阶段 2 使用 160×A100 80G；仅有模型权重不足以复现数据管线。
3. **重型中间件增加延迟**：6B ViT 与 8B QLLaMA 的组合不适合边缘部署，也解释了后续架构为何走向直接 projector。
4. **评测可比性有限**：不同 baseline 的数据、分辨率、预训练和可训练参数不同，汇总图更适合展示覆盖面而非严格排名。
5. **公开数据仍有治理问题**：LAION/COYO 等 web 数据的版权、噪声、多语言分布与偏差并未在正文中深入讨论。

### 7.2 合理的后续方向

从本文暴露的问题出发，最自然的未来方向是：减少语言中间件成本；提升原生高分辨率与 OCR；让视觉和语言从预训练开始联合而不是分段桥接；用偏好优化或可验证强化学习提高推理；公开更可复现的数据配方和清洗规则。这些方向后来确实成为 InternVL2/3/3.5 的重点，但属于系列演化的后见观察，不是本文原文承诺。

---

## 8. 复现成本与实践建议

### 8.1 完整复现

完整复现几乎只适合拥有大规模 GPU 集群的机构。阶段 1 的 640 张 A100 与 164K batch、阶段 2 的 160 张 A100，加上 4.98B/1.03B 图文数据清洗，成本远高于普通学术团队。真正难点不只是算力，还包括多源数据下载、去重、caption 质量筛选、超大 batch 的通信和训练稳定性。

### 8.2 可行的缩小版验证

更现实的复现顺序是：直接加载公开 InternViT 权重；在小规模图文集上比较 `InternViT+MLP` 与 `InternViT+query/cross-attention`；固定 LLM，仅训练 projector；在检索、caption 和 VQA 各选一个任务验证。若研究重点是今天的 InternVL 系列，则不建议从头重建 QLLaMA，而应把本文作为视觉底座与训练哲学的来源，转向后续 ViT–MLP–LLM 实现。

### 8.3 预计资源

权重级推理仍至少需要容纳 6B 视觉编码器、8B QLLaMA 和可选 7B/13B LLM；即便 bf16，也不含 KV cache 和激活就已是数十 GB。量化可降低显存，但论文未提供量化实验，不能假设精度保持。

---

## 9. 个人判断与后续问题

### 9.1 最值得保留的洞察

这篇论文最值得记住的是“**目标函数要与数据质量匹配**”：噪声 web 图文对先做容错较强的对比学习，只有筛过的 caption 才进入生成训练。它比“把所有数据统一做 next-token prediction”更保守，却在 2023 年的大规模训练条件下更稳健。

第二个洞察是视觉侧不能永远作为廉价附件。即便后续 InternVL 不再保留 QLLaMA，持续增强 InternViT、动态高分辨率和视觉 token 组织，仍延续了本文的基本判断。

### 9.2 我的结论

InternVL 是一篇“搭底座”的工作：它用巨大工程投入证明大视觉模型与 LLM 对齐能覆盖广泛任务，也留下了足够明确的负面信号——重型 QLLaMA 并不是最终形态。若从今天回看，**InternViT 与渐进数据配方是长期资产，QLLaMA 是完成路线探索的过渡结构**。

---

## 10. 材料来源

- 官方论文：<https://arxiv.org/abs/2312.14238>
- 本地 PDF：`assets/InternVL - 规模化视觉基础模型与通用对齐/sources/paper.pdf`
- 来源与 Figure/Page 记录：`assets/InternVL - 规模化视觉基础模型与通用对齐/sources.md`
- 官方代码：<https://github.com/OpenGVLab/InternVL>
