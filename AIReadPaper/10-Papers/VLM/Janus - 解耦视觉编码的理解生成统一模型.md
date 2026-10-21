---
type: paper-note
status: done
domain: VLM
paper: "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation"
year: 2024
arxiv: "2410.13848"
doi: "10.48550/arXiv.2410.13848"
source: "https://arxiv.org/abs/2410.13848"
project: "https://github.com/deepseek-ai/Janus"
code: "https://github.com/deepseek-ai/Janus"
tags:
  - VLM
  - unified-multimodal
  - autoregressive-generation
  - decoupled-visual-encoding
  - SigLIP
  - VQ-tokenizer
created: 2026-10-16
updated: 2026-10-16
---

# Janus：解耦视觉编码的理解生成统一模型

## 0. 基本信息

- **作者：** Chengyue Wu、Xiaokang Chen、Zhiyu Wu、Yiyang Ma、Xingchao Liu、Zizheng Pan、Wen Liu、Zhenda Xie、Xingkai Yu、Chong Ruan、Ping Luo
- **主模型：** DeepSeek-LLM 1.3B，共享上下文长度 4096
- **输入分辨率：** 理解与生成均为 $384\times384$
- **本地原文：** [Janus PDF](assets/Janus%20-%20解耦视觉编码的理解生成统一模型/sources/paper.pdf)
- **后续工作：** [Janus-Pro PDF](assets/Janus%20-%20解耦视觉编码的理解生成统一模型/sources/janus-pro.pdf)
- **完整溯源：** [sources.md](assets/Janus%20-%20解耦视觉编码的理解生成统一模型/sources.md)

## 1. 一句话结论

Janus 的关键不是把理解和生成强行压进同一个视觉 tokenizer，而是**只共享负责推理与序列建模的 LLM，视觉入口按任务解耦**：理解走 SigLIP 连续语义特征，生成走 VQ 离散重建 token，再用两个 adaptor 接入同一自回归 Transformer。消融表明，共用单一 VQ 表示会严重损害理解，共用“语义增强 tokenizer”又会在联合训练中产生明显任务冲突；双表示以少量视觉侧冗余换来更好的统一性能。

![](assets/Janus%20-%20解耦视觉编码的理解生成统一模型/figures/01-overview.png)

> **图 1：Janus 的理解与生成结果概览。** 1.3B Janus 同时覆盖理解和文生图；论文强调的是统一模型的性能—规模优势，而不是在每个单项上绝对领先所有专用模型。来源：论文 Figure 1。

---

## 2. 问题定义：理解与生成需要不同的视觉信息

### 2.1 统一模型的真正冲突发生在视觉表示层

“图像理解”和“图像生成”都处理图片，但它们对表示的要求并不相同：

1. **理解表示要语义不变。** 物体类别、属性、关系和图文对齐最重要；局部像素扰动、纹理和精确颜色可以被压缩。SigLIP/CLIP 类视觉编码器因此倾向于学习高维、抽象、判别性的连续特征。
2. **生成表示要可逆重建。** 输出必须恢复颜色、纹理、边缘和空间细节；VQ tokenizer 因此把图像压缩成低维离散码，并保留足够局部信息供 decoder 重建。
3. **单表示存在 Pareto 冲突。** 只优化重建的 codebook 不一定形成线性可读的语义结构；强行注入语义监督又可能牺牲像素重建或让一个 tokenizer 同时承担互相拉扯的目标。

Janus 的判断是：统一不等于所有模块都必须共享。应把“跨模态推理与序列建模”统一到 LLM，把“如何把原始图像变成合适的 token/feature”留给任务专用入口。

### 2.2 核心贡献

1. **双视觉编码路径。** 理解使用 SigLIP，生成使用 VQ tokenizer，直接绕开语义抽象与像素重建的表示冲突。
2. **单一共享自回归主干。** 文本、理解图像特征和生成图像 token 都映射到同一 LLM embedding 空间，由一个 Transformer 处理。
3. **简单统一目标。** 不引入扩散目标或特殊 attention mask；理解监督文本 token，生成监督图像 token，均使用 next-token cross-entropy。
4. **针对冲突的控制实验。** 比较普通 VQ、语义增强 tokenizer、SigLIP+VQ 双入口及单任务训练，证明“解耦视觉编码”而非仅增大模型带来主要理解收益。
5. **可扩展接口。** 后续 Janus-Pro 保持架构不变，仅通过训练策略、数据和模型规模扩展显著提升性能。

---

## 3. 总体架构：双入口、双 adaptor、共享 LLM、双 head

![](assets/Janus%20-%20解耦视觉编码的理解生成统一模型/figures/02-architecture.png)

> **图 2：Janus 总体架构。** 左侧理解路径将 SigLIP 连续视觉特征作为前缀；右侧生成路径把 VQ codebook embedding 作为自回归视觉序列；二者共享同一个 Transformer。来源：论文 Figure 2。

### 3.1 三种输入如何进入同一个隐空间

设 LLM 隐藏维度为 $d$，共享 Transformer 为 $T_\theta$。

#### 3.1.1 纯文本

文本 tokenizer 将输入变成离散 ID：

$$
\mathbf{t}=(t_1,\ldots,t_n),\qquad \mathbf{e}^{\text{text}}_i=E_{\text{text}}(t_i)\in\mathbb{R}^{d}.
$$

这些 embedding 直接进入 LLM。

#### 3.1.2 多模态理解：SigLIP 连续语义入口

对输入图像 $I$，SigLIP-Large-Patch16-384 输出二维特征网格：

$$
\mathbf{H}^{u}=f_{\text{SigLIP}}(I)\in\mathbb{R}^{h\times w\times d_v}.
$$

$384/16=24$，因此其空间网格约为 $24\times24$；展平后约 576 个视觉位置。理解 adaptor 是两层 MLP：

$$
\mathbf{Z}^{u}=A_u\!\left(\operatorname{Flatten}(\mathbf{H}^{u})\right)\in\mathbb{R}^{hw\times d}.
$$

$A_u$ 的任务不是重建图像，而是把 SigLIP 的语义空间对齐到 LLM token 空间。视觉特征与用户文本 embedding 拼成前缀，LLM 用原有 **text head** 输出答案。

#### 3.1.3 视觉生成：VQ 离散重建入口

生成 encoder 将图像压缩为 codebook ID 网格：

$$
\mathbf{q}=Q(I)=(q_1,\ldots,q_m),\qquad q_i\in\{1,\ldots,K\},
$$

其中 codebook 大小 $K=16{,}384$，空间下采样率为 16。对 $384\times384$ 图像，离散网格为 $24\times24$，即 $m=576$ 个图像 token。codebook embedding 再经过生成 adaptor：

$$
\mathbf{Z}^{g}=A_g\!\left(E_{\text{VQ}}(\mathbf{q})\right)\in\mathbb{R}^{m\times d}.
$$

$A_g$ 同样是两层 MLP。训练时，文本条件位于图像 token 之前；LLM 用独立、随机初始化的 **image head** 预测下一个 VQ ID。生成完毕后，VQ decoder 把 ID 网格还原为 RGB 图像。

### 3.2 两个 adaptor 为什么不能合并

两个 adaptor 的输入统计和目标完全不同：

- $A_u$ 接收 SigLIP 的连续、高维、语义判别特征；
- $A_g$ 接收 VQ codebook embedding，重点是可预测的局部视觉状态；
- 前者服务“看懂后输出文本”，后者服务“根据文本逐 token 输出图像”。

如果共享 adaptor，就会把解耦后的两个表示重新压回同一个接口函数，增加不必要的梯度冲突。Janus 用两个很小的 MLP 保持入口独立，而把参数量最大的 LLM 共享。

### 3.3 两个输出 head 的职责

| Head | 词表 | 任务 | 训练位置 |
|---|---|---|---|
| Text head | LLM 原生文本词表 | 纯文本与图像理解回答 | 仅对目标文本 token 计 loss |
| Image head | 16,384 个 VQ code | 文生图图像 token | 仅对目标图像序列计 loss |

这避免把文本 token 与图像 code 强行放进完全相同的输出归一化空间。统一发生在 Transformer hidden state，而非输出类别集合。

### 3.4 统一序列与 attention mask

Janus 保持 decoder-only 的标准因果分解：

$$
p_\theta(\mathbf{x})=\prod_{i=1}^{L}p_\theta(x_i\mid x_{<i}).
$$

论文明确强调**不需要专门设计 attention mask**，主干使用标准 causal mask：

$$
M_{ij}=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

- 理解时，图像特征在回答文本之前，因此每个回答 token 可以看到完整视觉前缀和问题。
- 生成时，每个图像 token 可以看到文本条件与此前所有图像 token，但不能看到未来图像 token。
- 代价是二维图像被固定展平为一维 raster order，生成误差会沿序列累积；论文也把“因果+双向混合注意力”列为未来改进，而非当前模型已有能力。

---

## 4. 训练目标与三阶段训练

### 4.1 统一交叉熵，按任务选择监督区间

总目标是：

$$
\mathcal{L}(\theta)=-\sum_{i=1}^{L}\log p_\theta(x_i\mid x_{<i}).
$$

实际 loss mask 随任务变化：

$$
\mathcal{L}_{u}=-\sum_{i\in\mathcal{T}_{\text{answer}}}\log p_\theta(t_i\mid \mathbf{Z}^{u},t_{<i}),
$$

$$
\mathcal{L}_{g}=-\sum_{i\in\mathcal{T}_{\text{image}}}\log p_\theta(q_i\mid \mathbf{t},q_{<i}).
$$

理解任务只监督文本回答；生成任务只监督图像 token。原论文没有给不同任务额外 loss 权重，而是用 batch 内数据比例控制梯度占比。

![](assets/Janus%20-%20解耦视觉编码的理解生成统一模型/figures/03-training-stages.png)

> **图 3：三阶段冻结/解冻。** Stage I 只建立视觉—语言接口；Stage II 解冻 LLM 做统一预训练；Stage III 再解冻理解 encoder，而生成 tokenizer 始终保持冻结。来源：论文 Figure 3。

### 4.2 Stage I：训练 adaptor 与 image head

- 冻结 SigLIP、VQ encoder 和 LLM。
- 只训练 $A_u$、$A_g$ 与 image head。
- 数据：1.25M ShareGPT4V 图文 caption；约 1.2M ImageNet-1k 类别名→图像样本。
- 训练 10K steps，batch 256，学习率 $10^{-3}$，数据比 **理解:文本:生成 = 1:0:1**。

这一步相当于接口校准：不改变既有视觉和语言表征，只学习“如何把两类视觉信号翻译到 LLM 空间”以及“如何从 LLM hidden state 读出 VQ code”。

### 4.3 Stage II：统一预训练

- 解冻 LLM；两个视觉 encoder 仍冻结。
- 数据同时包含纯文本、图文交错、caption、表格/图表理解和文生图。
- 训练 180K steps，batch 512，学习率 $10^{-4}$，数据比 **2:3:5**。
- 前 120K steps 的生成数据使用 ImageNet 类别提示，后 60K steps 转向开放域图文数据。

数据来源包括 DeepSeek-LLM 文本语料、WikiHow、WIT、多种 caption 数据、DeepSeek-VL 表格/图表数据、多种公开图文数据及约 2M 内部生成数据。对部分生成数据按美学和尺寸过滤，仅保留约 20%。

### 4.4 Stage III：统一监督微调

- 除生成 encoder 外，其余模块参与更新；SigLIP 理解 encoder 在此阶段解冻。
- 只监督 assistant answer，system/user prompt 被 mask。
- 训练 24K steps，batch 256，学习率 $2\times10^{-5}$，weight decay 0.1，数据比 **7:3:10**。
- 理解使用多种 instruction 数据；生成使用 Stage II 的子集与 4M 内部数据。

三阶段都使用 AdamW，$\beta_1=0.9,\beta_2=0.95$，gradient clip 1.0。全流程在 $16\times8=128$ 张 A100 40GB 上约 7 天。

### 4.5 数据比例意味着什么

Stage II 的 2:3:5 与 Stage III 的 7:3:10 都让生成占较大比重。由于每张图有 576 个预测 token，生成样本本身也会贡献密集梯度；论文没有做 token 数归一化或任务 loss 重标定。由此可见，Janus 的“统一目标”很简单，但训练平衡仍主要依赖人工数据配比。

---

## 5. 推理：文本采样与图像 CFG

理解任务按普通 LLM 方式逐 token 采样。图像生成使用 classifier-free guidance（CFG）：

$$
\ell_{\text{cfg}}=\ell_u+s(\ell_c-\ell_u),
$$

其中 $\ell_c$ 是有文本条件 logits，$\ell_u$ 是无条件 logits，默认 $s=5$。训练时以 10% 概率把文本条件替换为 pad token，使同一模型学会无条件分支。CFG 每个位置需要条件/无条件两次前向或等价批处理，因此提升条件遵循的同时增加推理成本。

最终生成 576 个 VQ token，再由固定 VQ decoder 重建 $384\times384$ 图像。论文未完整披露 temperature、top-k/top-p 和重复惩罚等采样参数，严格复现需以官方代码配置为准。

---

## 6. 实验结果

### 6.1 多模态理解

| 模型 | LLM 参数 | POPE | MME-P | MMBench | SEED | VQAv2 | GQA | MMMU | MM-Vet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Show-o | 1.3B | 73.8 | 948.4 | — | — | 59.3 | 48.7 | 25.1 | — |
| LLaVA-v1.5 | 7B | 85.9 | 1510.7 | 64.3 | 58.6 | 78.5 | 62.0 | 35.4 | 31.1 |
| Emu3-Chat | 8B | 85.2 | — | 58.5 | 68.2 | 75.1 | 60.3 | 31.6 | — |
| **Janus** | **1.3B** | **87.0** | **1338.0** | **69.4** | **63.7** | **77.3** | **59.1** | **30.5** | **34.3** |

Janus 相对 Show-o 在 MME-P 上由 948.4 升至 1338，在 GQA 上由 48.7 升至 59.1；对 7B LLaVA-v1.5，则在 POPE、MMBench、SEED 和 MM-Vet 更高，但 MME-P、VQAv2、GQA、MMMU 并非全面领先。合理结论是：**同规模统一模型中非常强，且参数效率突出**，而不是所有理解基准绝对 SOTA。

### 6.2 文生图

- GenEval overall：**0.61**，高于 Show-o 0.53、SDXL 0.55、DALL-E 2 0.52。
- COCO-30K FID：**8.53**。
- MJHQ-30K FID：**10.10**。
- 生成分辨率只有 $384\times384$；FID 与更高分辨率模型比较时需注意设置差异。

GenEval 分项为：Single Object 0.97、Two Objects 0.68、Counting 0.30、Colors 0.84、Position 0.46、Color Attribution 0.42。单物体和颜色较强，计数与属性绑定仍明显困难。

### 6.3 核心消融：单视觉表示为何失败

| Exp | 视觉表示 | 训练任务 | POPE↑ | MMB↑ | SEED↑ | MMMU↑ | COCO-FID↓ |
|---|---|---|---:|---:|---:|---:|---:|
| A | 普通 VQ tokenizer | 理解+生成 | 60.1 | 35.0 | 34.9 | 24.7 | 8.72 |
| B | 语义增强 tokenizer | 理解+生成 | 82.4 | 52.7 | 54.9 | 26.6 | **7.11** |
| C | 语义增强 tokenizer | 仅理解 | 83.9 | 62.1 | 60.8 | 27.5 | — |
| D | **SigLIP + VQ** | 理解+生成 | **87.0** | 69.4 | 63.7 | **30.5** | 8.53 |
| E | SigLIP | 仅理解 | 85.9 | **70.6** | **64.8** | 28.8 | — |
| F | 普通 VQ tokenizer | 仅生成 | — | — | — | — | 8.92 |

可得到四个重要结论：

1. **重建型 VQ 不等于理解表征。** A 的生成 FID 8.72 尚可，但理解全面崩塌。
2. **给 tokenizer 加语义监督只能部分补救。** B 相比 A 大幅改善理解，并取得最佳 FID 7.11，但理解仍显著落后 D。
3. **单表示中的任务冲突可以直接测出。** B 与 C 使用同一语义 tokenizer；加入生成联合训练后 MMBench 从 62.1 降到 52.7、SEED 从 60.8 降到 54.9。
4. **双入口的联合训练基本不伤主任务。** D 与理解专用 E 接近；D 的 POPE、MMMU 甚至略高。D 的 FID 8.53 也优于仅生成 F 的 8.92，说明共享 LLM 和混合数据可能产生正迁移。

### 6.4 “语义—重建冲突”应如何理解

附录中的 semantic tokenizer 在 VQ 后增加 12 层 ViT semantic decoder，用 SigLIP teacher 做语义重建，同时保留 RGB reconstruction。其目标可概括为：

$$
\mathcal{L}_{\text{tokenizer}}=
\mathcal{L}_{\text{RGB}}+0.25\,\mathcal{L}_{\text{semantic}},
$$

其中语义项最大化预测语义特征与 SigLIP 特征的 cosine similarity。它比普通 VQ 更懂语义，却仍必须让同一码承担两种用途。B→C 的差距说明冲突不只存在于 tokenizer 预训练目标，也存在于后续共享 LLM 的联合训练梯度中。Janus 的解决方法不是寻找“完美折中码”，而是让两种码都存在。

---

## 7. Janus-Pro：架构不变，训练、数据与规模升级

Janus-Pro（arXiv:2501.17811）明确说明架构与 Janus 相同，因此不是第三条视觉路径，也没有新的融合模块。增量集中在三方面。

### 7.1 优化三阶段职责

1. **Stage I 延长到 20K steps。** 在 LLM 冻结时更充分地用 ImageNet 学像素依赖和类别条件生成。
2. **Stage II 移除 ImageNet。** 原 Janus 在 Stage II 的文生图步骤中约 66.67% 用于 ImageNet 类别提示；Pro 把这部分移到 Stage I，Stage II 只用常规密集描述图文数据，实际在 270K steps early stop。
3. **Stage III 比例从 $7:3:10$ 改为 $5:1:4$。** 归一化后理解占比从 35% 提到 50%，生成从 50% 降到 40%，以提高理解同时维持生成。

### 7.2 数据扩展

- 理解侧新增约 **90M** 样本，包括 YFCC caption、表格、图表、Docmatix 文档等；SFT 还加入 MEME、中文对话和对话体验数据。
- 生成侧新增约 **72M** 合成美学样本；统一预训练中真实:合成达到 **1:1**。
- 作者观察到合成数据带来更快收敛、更稳定的短提示词生成和更高美学质量，但未给单变量数值消融。

### 7.3 模型扩展与结果

Janus-Pro 提供名义 1B 与 7B 两档；论文配置表中前者的 LLM 实为约 1.5B。7B 模型 hidden size 4096、30 层、32 heads；上下文仍是 4096，图像仍是 $384\times384$，VQ codebook 仍为 16,384。

| 模型 | MMBench | MMMU | MM-Vet | GenEval | DPG-Bench |
|---|---:|---:|---:|---:|---:|
| Janus | 69.4 | 30.5 | 34.3 | 0.61 | 79.68 |
| Janus-Pro-1B/1.5B | 75.5 | 36.3 | 39.8 | 0.73 | 82.63 |
| Janus-Pro-7B | **79.2** | **41.0** | **50.0** | **0.80** | **84.19** |

必须谨慎归因：Pro 同时改变训练日程、数据质量/规模和参数规模，没有把每项增量做严格控制变量消融；因此能确认“整体配方有效”，不能量化 90M 理解数据、72M 合成数据或 7B 扩容各自贡献。

---

## 8. 与 Emu3 的统一范式对比

参见：[[Emu3 - 图像视频文本统一自回归]]。

| 维度 | Janus | Emu3 |
|---|---|---|
| 视觉表示 | **双表示**：SigLIP 连续语义特征 + VQ 离散重建 token | **单表示**：同一 MoVQGAN 离散 token 用于理解、图像生成、视频生成 |
| 主干 | 共享 1.3B decoder-only LLM | 从零训练 8B decoder-only Transformer |
| 视觉接口 | 两个两层 MLP adaptor | 无独立视觉 adaptor；扩展统一 embedding/词表 |
| 输出 head | 文本 head 与图像 head 分离 | 统一词表上的 next-token head |
| attention mask | 标准 causal；无特殊任务 mask | 标准 causal；图像/视频按栅格和时间次序串行化 |
| 理解目标 | 视觉是连续条件，只监督文本答案 | 图像 token 与文本 token 都可成为预测目标；理解后训练忽略视觉 loss |
| 生成目标 | 预测 VQ token；CFG | 预测 MoVQGAN token；视觉 loss 权重 0.5，QFT/DPO，CFG |
| 冲突策略 | 结构性隔离语义与重建梯度 | 接受单 tokenizer 折中，以更大数据/模型统一建模 |
| 视频 | 原版不支持 | 原生视频 token、文生视频与未来预测 |
| 上下文 | 4096 | 131,072 |

**核心分歧：** Janus 认为理解与生成的最优视觉表征不同，统一应发生在 LLM；Emu3 认为只要视觉 tokenizer 足够强、模型与数据足够大，所有模态都可以变成同一种离散 next-token 问题。Janus 的消融对“单表示冲突”给出直接证据；Emu3 则以跨图像/视频/文本的结果证明单表示路线可行，但没有提供与双表示控制变量相同的冲突消融。

---

## 9. 局限与批判性思考

1. **分辨率上限低。** 理解和生成固定 $384\times384$；OCR、小目标和人脸细节受限，Janus-Pro 也未解决。
2. **生成仍受 VQ reconstruction ceiling 限制。** LLM 即使预测正确 code，也无法恢复 tokenizer 已丢失的高频细节。
3. **自回归图像推理慢且误差累积。** 576 个图像位置必须串行生成，CFG 还增加条件/无条件计算。
4. **“统一”仍有视觉冗余。** 两个 encoder、两个 adaptor、两个 head 不是最小模块数；它优化的是性能冲突，不是端到端参数最少。
5. **数据透明度有限。** Stage II/III 含 2M/4M 内部数据；Janus-Pro 又加入 72M 合成数据，生成器、去重和版权治理细节未完整披露。
6. **消融规模有限。** 核心表很好地验证双入口，但只在 1.3B、384 分辨率与特定 tokenizer 下进行；不能证明任何规模下双表示都必然优于单表示。
7. **Janus-Pro 归因混杂。** 策略、数据和规模同时改变，缺少逐项消融。

---

## 10. 复现路线与检查点

### 10.1 最小可行复现

1. 准备 SigLIP-Large-Patch16-384、论文同类 VQ tokenizer 与 1B 级 decoder-only LLM。
2. 实现两套视觉路径与两个两层 MLP，确认最终 hidden size 与 LLM embedding 对齐。
3. 文本 head 复用 LLM 输出层；新增 $K=16{,}384$ 类 image head。
4. 实现两种序列模板与 loss mask：理解只训 answer，生成只训 VQ token。
5. 严格复现三阶段冻结表；不要在 Stage I 意外更新视觉 encoder 或 LLM。
6. 生成训练以 10% 条件 dropout 建立 CFG 无条件分支；推理先用 $s=5$。
7. 在 POPE/MMBench/SEED/MMMU 与 COCO-FID/GenEval 上同时评估，避免只看一侧。

### 10.2 关键 sanity checks

- VQ 重建的 FID/LPIPS 必须先单独合格，否则 LLM 无法突破 tokenizer 上限。
- 检查 SigLIP 与 VQ 路径是否真的使用独立参数，避免误共享 embedding/adaptor。
- 记录每批**有效监督 token 数**而不只记录样本数；576 个视觉 token 会改变真实梯度权重。
- 复现消融 A/B/C/D/E/F，尤其关注 B→C 的联合训练退化，这是验证冲突的关键。
- 生成评估固定分辨率、CFG、采样策略与样本数；FID 对这些设置敏感。

### 10.3 资源口径

原版报告为 128×A100 40GB、约 7 天。完整复现还依赖未公开内部数据，严格数值复现难；更现实的是复现架构与冲突消融，再用公开 caption/VQA/文生图数据验证趋势。

---

## 11. 关键术语

- **SigLIP：** 通过 sigmoid 对比目标学习图文语义对齐的视觉 encoder，适合理解而非像素重建。
- **VQ tokenizer：** 把图像压缩成离散 codebook ID，并由 decoder 重建 RGB。
- **Adaptor：** 把外部视觉特征映射到 LLM hidden size 的小型网络。
- **CFG：** 同时使用条件与无条件 logits 放大文本条件影响的生成技巧。
- **Representation conflict：** 同一视觉码同时追求语义抽象和像素可逆时产生的优化冲突。

## 12. 图表索引

1. `01-overview.png`：理解与生成总体结果，论文 Figure 1。
2. `02-architecture.png`：双视觉入口与共享 Transformer，论文 Figure 2。
3. `03-training-stages.png`：三阶段冻结/解冻流程，论文 Figure 3。