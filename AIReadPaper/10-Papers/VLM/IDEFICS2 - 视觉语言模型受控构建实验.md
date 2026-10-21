---
type: paper-note
status: complete
domain: VLM
paper: "What matters when building vision-language models?"
year: 2024
arxiv: "2405.02246"
doi: null
source: "https://arxiv.org/abs/2405.02246"
project: "https://huggingface.co/collections/HuggingFaceM4/idefics2-661d1971b7c50831dd3ce0fe"
code: null
tags:
  - VLM
  - fully-autoregressive
  - perceiver-resampler
  - document-understanding
  - OCR
  - IDEFICS2
created: 2026-10-16
updated: 2026-10-16
---

# IDEFICS2：视觉语言模型受控构建实验

## 0. 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | What matters when building vision-language models? |
| 作者 | Hugo Laurençon、Léo Tronchon、Matthieu Cord、Victor Sanh |
| 模型 | Idefics2，约 8B；SigLIP-SO400M + Mistral-7B-v0.1 |
| 核心问题 | 在尽量控制 backbone、数据量与训练步数时，架构、适配方法、视觉 token 数、分辨率与 OCR 数据分别贡献什么？ |
| 本地论文 | [paper.pdf](assets/IDEFICS2%20-%20视觉语言模型受控构建实验/sources/paper.pdf) |
| 来源清单 | [sources.md](assets/IDEFICS2%20-%20视觉语言模型受控构建实验/sources.md) |

**一句话结论：** IDEFICS2 的价值不只是一个 8B 模型，而是用受控实验说明：冻结 backbone 时 cross-attention 占优，但允许 LoRA 适配后 fully autoregressive 反超；64-query Perceiver、保宽高比输入、切图与专门 OCR/PDF 数据共同构成了高效率文档视觉能力。

---

## 1. 研究背景与问题定义

### 1.1 为什么要问“What matters”

2023–2024 年的 VLM 往往同时更换语言模型、视觉编码器、connector、训练数据和分辨率，最终榜单上涨却很难归因。典型分歧有两类：Flamingo/IDEFICS1 把 cross-attention 插入 LLM 多层；LLaVA 类方法把视觉 token 直接拼进文本序列。前者的视觉通道更专门，后者结构和部署更简单，但此前缺少在相同训练环境下的正面对照。

论文因此固定或显式记录关键变量，以 VQAv2、TextVQA、OKVQA、COCO 的 4-shot 平均分作为快速代理指标，逐项比较 backbone、融合架构、适配方法、视觉 token 压缩、分辨率组织与 OCR 数据，再把结论汇总成 IDEFICS2。

### 1.2 核心贡献

1. **架构归因**：证明“cross-attention 还是 fully autoregressive”不能脱离可训练参数讨论；冻结两端时前者更强，LoRA 适配两端后后者更强。
2. **高效视觉接口**：用 Perceiver Resampler 把每图 729 个特征压为 64 个 token，平均分反而提高 8.5，而不是用更多 token 换分数。
3. **分辨率与 OCR 配方**：保留原始宽高比、分阶段提高最大分辨率，并加入 PDF/OCR/合成文字数据，使文档任务增益不再只依赖粗暴放大图像。
4. **开放训练资产**：发布 base、instruct、chat 模型与 The Cauldron 指令数据集合，降低结果复核门槛。

---

## 2. 任务、输入输出与张量形状

### 2.1 统一接口

输入是任意交错的图像与文本，输出只生成文本。设一张 RGB 图像为 $I\in\mathbb{R}^{H\times W\times3}$，SigLIP 输出可变长 patch 序列 $X_v\in\mathbb{R}^{N_v\times d_v}$；其中 $N_v$ 随宽高比与分辨率变化。两层 modality projection 把通道映射到中间空间，Perceiver 用 64 个 latent query 汇聚为 $Z_v\in\mathbb{R}^{64\times d_l}$，再与文本 embedding $E_t\in\mathbb{R}^{N_t\times d_l}$ 拼成 LLM 输入。

$$X=[Z_v;E_t]\in\mathbb{R}^{(64+N_t)\times d_l}$$

多图时每张图各产生 64 个视觉 token；instruction 阶段可把原图与四个 crop 作为五幅图输入，因此标准 64 token/图变为 320 token/原始样本。输出 $y_{1:L}$ 由 Mistral decoder 自回归生成。

![](assets/IDEFICS2%20-%20视觉语言模型受控构建实验/figures/01-architecture.png)

> 图 1：Figure 2。视觉特征经 projection 与 learned pooling 得到每图 64 个 token，再与文本 token 交错拼接进入同一个 causal LLM；这就是论文所谓 fully autoregressive，而不是“模型同时生成图像”。

### 2.2 Q/K/V、mask 与 position

**视觉编码器内部。** SigLIP 是 ViT，自注意力中每个 patch 都产生 $Q_v=X_vW_Q$、$K_v=X_vW_K$、$V_v=X_vW_V$，在二维 patch 序列上做双向视觉注意力。输入不再强制缩成统一方形，预训练位置 embedding 通过插值适配新的网格；论文指出固定分辨率下可视为绝对位置，而变分辨率后其作用更接近网格内相对位置提示。

**Perceiver Resampler。** 64 个可学习 latent 是 query，投影后的视觉特征提供 key/value；经过 3 层 cross-attention 后，输出长度恒为 64。可写成：

$$Z^{(k+1)}=Z^{(k)}+\operatorname{Attn}(Q=Z^{(k)}W_Q,K=X_vW_K,V=X_vW_V)$$

论文没有重新定义一套 Perceiver 公式，但明确消融了 connector、层数和 projection。1/3/12 层平均分为 69.3/68.6/69.0，说明深度不是主要增益来源；三层是保守容量选择。额外两层 projection 置于 Resampler 前，将平均分从 69.6 提到 71.4。

**LLM 内部。** visual token 与 text token 使用同一 hidden size，经过同一组 Mistral self-attention Q/K/V。mask 是标准 causal mask：位置 $i$ 只能看不晚于自己的序列位置；图像 token 若放在问题文本之前，后续问题与答案可读取它。训练目标不要求把视觉 token 当离散标签预测，而是通过文本 next-token loss 把梯度传回视觉侧。

### 2.3 训练目标与生成

预训练目标是条件语言建模：

$$\mathcal{L}_{\mathrm{NLL}}=-\sum_{t=1}^{L}\log p_\theta(y_t\mid y_{<t},X_v,x_{\mathrm{text}})$$

instruction tuning 时只对 assistant 答案 token 计损失，user prompt、图像 token 和上下文只作为条件。推理时使用标准自回归解码；基础评测采用 open-ended generation 或 in-context examples，chat 版本再通过少量对话数据改变输出风格。

---

## 3. Fully autoregressive 与 cross-attention 的受控对照

### 3.1 两种架构到底差在哪里

| 维度 | Cross-attention | Fully autoregressive |
|---|---|---|
| 视觉注入 | 在 LLM 多层插入 text→image cross-attention | 视觉 token 直接加入 LLM token 序列 |
| Q/K/V | 文本 hidden 作 Q，视觉特征作 K/V；语言 self-attention 另走一路 | 视觉与文本共同产生 LLM self-attention 的 Q/K/V |
| LLM 序列长度 | 每图只需占一个占位位置，但各插层仍读取视觉 K/V | 标准设置每图显式占 64 个位置 |
| 新增参数 | 多层 cross-attention，约 2B 可训练参数 | projection + Perceiver，新增参数更少 |
| 推理代价 | 论文测得约多 10% FLOPs | 结构直接，可合并 LoRA，无新增推理分支 |
| 归纳偏置 | 强制语言在指定层读取视觉 | 让 LLM 自注意力统一学习跨模态关系 |

这里最容易误读的是“fully autoregressive 更好”。冻结所有预训练 backbone 时，fully autoregressive 为 60.3，cross-attention 为 66.7；cross-attention 的专门视觉通道和更多可训练参数更占优。只有在给 LLM/视觉侧增加可适配自由度后结论才反转。

### 3.2 冻结与 LoRA 改变了结论

论文尝试全参数解冻 fully autoregressive，但出现 loss divergence，降低学习率和渐进解冻都未稳定。改用 LoRA 后，fully autoregressive 从 60.3 跃升至 69.5（+12.9），cross-attention 只从 66.7 到 67.3（+0.6）。LoRA 一方面让预训练 backbone 能适应新视觉序列，另一方面限制更新秩，降低大规模联合训练的破坏性；训练后又能合并权重，不增加在线路径。

![](assets/IDEFICS2%20-%20视觉语言模型受控构建实验/figures/03-architecture-ablation.png)

> 图 2：Figure 4。无论按 optimizer step、处理图像数还是文本 token 数对齐，LoRA 条件下 fully autoregressive 后期均领先；作者特别按三种横轴报告，是因为两种架构的序列长度与每步样本构成并不相同。

### 3.3 结论边界

这组实验不能证明 cross-attention 在所有规模上都过时。对照中 cross-attention 新增约 1.3B 更多可训练参数，视觉信息也不占 64 个显式 LLM 位置；而 fully autoregressive 的优势与 LoRA、Perceiver 和具体数据预算绑定。稳妥结论是：**当 backbone 允许稳定适配时，共享 LLM 自注意力足以学习跨模态融合，并以更简单的推理图获得更好分数。**

---

## 4. Perceiver、切图、OCR 与训练配方

### 4.1 Perceiver 为什么既压缩又提分

不做 Perceiver 的 fully autoregressive 冻结基线仅 51.8；加入 Perceiver 后为 60.3，且视觉 token 从 729 降到 64。附录中 Linear Projection、Mapping Network、Perceiver 分别为 44.5、51.8、60.3。其作用不只是压缩：learnable queries 会按任务相关性聚合全局与局部证据，过滤大量冗余 patch，使 LLM 获得更紧凑的视觉条件。

64 与 128 query 的平均分分别为 71.7 和 71.2，未见“token 越多越好”。这说明在该训练预算下，瓶颈更可能是视觉—语言对齐质量而不是原始 patch 容量；但作者也承认更大数据/更长训练下结论可能改变。

### 4.2 保宽高比与位置适配

SigLIP 原生在固定方形低分辨率训练。IDEFICS2 不扭曲图像，而是把原始宽高比对应的 patch 网格送入视觉编码器，并插值位置 embedding，再用 LoRA 适配。方形 768 输入平均分 73.1，保宽高比、378–768 动态输入为 72.1，轻微差异换来显存与吞吐的按图分配能力。对长文档和横向表格，这比统一拉伸到方形更合理。

### 4.3 切图与 thumbnail 式全局视图

instruction 阶段把一张图分成四个等分 crop，并把原图一并作为第五张图；每幅图各 64 token，总计 320。训练只对 50% 样本切图即可保留主要收益，说明模型同时学会低成本单图模式和高成本细节模式。切图特别提升 TextVQA、DocVQA；但把 crop 分辨率继续拉高的收益很小：TextVQA 73.0→73.6，DocVQA 72.9→72.7，主要增益来自局部视野而非无限加像素。

### 4.4 OCR 数据与分辨率存在协同

第一阶段最大分辨率 384，batch 2,048，混合 70% OBELICS 与 30% image-text pairs；第二阶段最大分辨率升至 980，加入 20% PDF 文档，配比变为 45% OBELICS、35% image-text pairs、20% PDF。PDF 部分包括约 19M OCR-IDL 行业文档、18M PDFA 页面和 Rendered Text 合成文字。

DocVQA 消融非常关键：无 OCR、384 分辨率为 22.6；无 OCR、768 为 42.9；有 OCR、768 为 49.9。高分辨率贡献约 20.3，OCR 数据再贡献 7.0。附录说明零/少样本 checkpoint 差异不显著，经过 500-step DocVQA 微调后差异才明显，因此它更像**可迁移的文字表征能力**，不是所有 prompt 下立即显现的零样本能力。

### 4.5 数据与训练规模

- OBELICS：350M 图像、115B 文本 token 的交错网页文档，用于多图、长上下文与保持语言能力。
- image-text pairs：PMD 与经合成 caption 改写、NSFW 过滤的 LAION-COCO；synthetic caption 快速消融由 49.8 提到 52.9。
- 预训练：约 2 epochs、1.5B 图像、225B 文本 token，学习率 $10^{-4}$；所有阶段都训练全模型，但部分 backbone 参数通过 LoRA 更新。
- instruction：The Cauldron 汇集 50 个视觉语言数据集，并混入文本指令；DoRA 更新，NEFTune 加 embedding 噪声，随机放大分辨率并打乱多轮顺序。
- chat：在 LLaVA-Conv 与 ShareGPT4V 上再训练数百步，主要改变“短 benchmark 答案”与“人类偏好长回答”的风格冲突。

---

## 5. 实验结果与消融

### 5.1 Base 模型

![](assets/IDEFICS2%20-%20视觉语言模型受控构建实验/figures/02-ocr-base-result.png)

> 图 3：Table 8 + Figure 3。Idefics2-base 每图只用 64 token，8-shot 下 VQAv2/TextVQA/OKVQA/COCO 为 70.3/57.9/54.6/116.0；下方手写信转录例子说明其 OCR 能力不只来自 instruction 模板。

与 144 token 的 MM1 相比，IDEFICS2 在 VQAv2、TextVQA、OKVQA 更高，COCO CIDEr 基本持平；与 cross-attention 的 IDEFICS1、Flamingo 相比，TextVQA 优势尤其明显。结果支持“压缩视觉 token 并不必然损失 OCR”，但不能忽略其训练规模达到 1.5B 图像，远大于许多开放 baseline。

### 5.2 Instruction 模型

同一权重可在 64 token/图和切图后的 320 token/图两种推理预算下运行。64-token 模式在 MMMU val/test、MathVista、TextVQA、MMBench 为 43.5/37.9、51.6、70.4、76.8；320-token 模式为 43.0/37.7、51.4、73.0、76.7。额外 token 主要帮助 OCR，通用推理几乎不变，印证分辨率应按任务选择。

### 5.3 关键消融汇总

| 变量 | 对照 | 结论 |
|---|---|---|
| LLM backbone | LLaMA-1-7B 62.5 → Mistral-7B 67.6 | 固定参数档时，语言 backbone 质量影响最大 |
| vision backbone | CLIP-H 57.4、EVA-5B 60.2、SigLIP-400M 60.7 | 大不等于训练充分；SigLIP 更高效 |
| 架构+适配 | FA frozen 60.3；CA frozen 66.7；FA LoRA 69.5；CA LoRA 67.3 | 架构排序取决于可训练自由度 |
| connector | Linear 44.5、Mapping 51.8、Perceiver 60.3 | learned query pooling 是核心组件 |
| OCR/分辨率 | 22.6 → 42.9 → 49.9 | 像素可读性与 OCR 数据互补 |
| 推理切图 | 64→320 token | OCR 增益明显，通用任务接近饱和 |

---

## 6. 局限、失败边界与批判性判断

1. **实验并非完全等成本。** Cross-attention 与 fully autoregressive 的新增参数、显式序列长度、每步图像数和文本 token 数不同；Figure 4 缓解但没有消除全部计算不等价。
2. **数据规模与架构贡献耦合。** IDEFICS2 的强结果建立在 1.5B 图像、225B 文本 token、专门 PDF/OCR 与大规模 instruction mixture 上，不能只归因给 Perceiver 或 FA。
3. **OCR 消融依赖下游微调。** 预训练 checkpoint 的零/少样本差异不显著，DocVQA 500-step 适配后才拉开；论文没有系统说明这种能力在更多文档域上的迁移边界。
4. **位置与长上下文分析不足。** 保宽高比依赖插值位置 embedding，多图又线性增加视觉 token；论文未给出极长多图下的 position extrapolation 与 KV-cache 曲线。
5. **输出仍是文本。** 检测框、mask 或网页代码都必须序列化，精确几何任务的 token 顺序和误差传播没有像 Florence-2 那样专门建模。
6. **chat 与 benchmark 冲突。** chatty 微调提高人类偏好，却可能损害短答案格式；论文选择另设模型版本，而不是解决同一策略内的可控长度与格式遵循。

我的判断是：论文最可靠的贡献是“把架构结论写成条件命题”。它没有宣称某个 connector 永久最优，而是展示冻结策略、优化稳定性、token 预算和数据类型如何共同改变排序。这比单次 SOTA 更有复用价值。

---

## 7. 复现路线

### 7.1 最小可验证实验

1. 固定同一个 7B LLM、SigLIP、训练数据和总更新步。
2. 实现 cross-attention 与 `projection → 64-query Perceiver → token concat` 两条路径。
3. 分别比较 frozen 与 LoRA，并同时报告 step、seen images、seen text tokens、FLOPs 和峰值显存。
4. 固定 FA 后比较无 pooling、Linear、Mapping、Perceiver，并用 64/128 queries 做 token-budget 曲线。
5. OCR 实验采用 `384 无 OCR / 768 无 OCR / 768 有 OCR` 三组，再统一做短程 DocVQA 适配。

### 7.2 工程检查点

- 视觉网格尺寸必须随输入宽高比变化，位置 embedding 插值与 padding mask 要同步。
- 每图 64 latent 的顺序要稳定；多图插入位置必须和 `<image>` 占位符一一对应。
- LLM 使用 causal mask，answer-only SFT loss 不能误算到 user、padding 或视觉 token。
- 切图推理应保留原图全局视图，并记录 64/320 token 两档的延迟与 KV cache。
- LoRA merge 后重新核对推理结果，防止 adapter target module 配置遗漏。

全量复现需要 1.5B 图像级数据和海量计算，普通团队更现实的目标是复现**相对排序**而不是绝对分数。

---

## 8. 与相关路线的关系

- **相对 Flamingo/IDEFICS1**：保留 Perceiver 的固定视觉瓶颈，但去掉深层 gated cross-attention，视觉 token 直接进入 LLM causal self-attention。
- **相对 LLaVA**：同为 fully autoregressive token concat，但不把全部 patch 直接塞给 LLM，而是用 Perceiver 压到 64，并更系统地处理多图、宽高比和 OCR。
- **相对 InternVL1.5**：两者都重视动态分辨率和 OCR；IDEFICS2 用 Perceiver 固定每图 token，InternVL1.5 用 tile + pixel shuffle 让 token 数随图像细节预算增长。
- **相对 Florence-2**：IDEFICS2 面向开放文本对话与文档推理；Florence-2 用 encoder-decoder 和位置 token 统一检测、grounding、分割等结构化视觉任务。

---

## 9. 图表索引

1. `figures/01-architecture.png`：fully autoregressive 架构与 64-token Perceiver 接口。
2. `figures/02-ocr-base-result.png`：base 模型主结果与文字转录例子。
3. `figures/03-architecture-ablation.png`：FA/CA 在 LoRA 条件下按三种计算轴的收敛对比。

## 10. 材料来源

- [官方论文 PDF](assets/IDEFICS2%20-%20视觉语言模型受控构建实验/sources/paper.pdf)
- [来源与原图清单](assets/IDEFICS2%20-%20视觉语言模型受控构建实验/sources.md)
