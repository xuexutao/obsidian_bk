---
type: paper-note
status: done
domain: VLM
series: BLIP
series_order: 4
series_hub: "[[BLIP 系列：从数据自举到统一理解生成 - Survey]]"
paper: "BLIP-3: A Family of Open Large Multimodal Models"
year: 2025
arxiv: "2408.08872"
doi: "10.1109/ICCVW69036.2025.00644"
source: https://arxiv.org/abs/2408.08872
project: https://www.salesforceairesearch.com/opensource/xGen-MM/index.html
code: https://github.com/salesforce/LAVIS/tree/xgen-mm
tags:
  - VLM
  - BLIP-3
  - multimodal-pretraining
  - interleaved-data
  - any-resolution
  - open-data
created: 2026-10-16
updated: 2026-10-16
---

# BLIP-3：开放大规模多模态模型家族

> 系列导航：[[BLIP - 生成式字幕过滤的统一视觉语言预训练]] → [[BLIP-2 - 冻结底座的高效视觉语言桥接]] → [[InstructBLIP - 指令感知视觉特征抽取]] → **BLIP-3** → [[BLIP3-o - 全开放统一多模态理解与生成]] · [[BLIP 系列：从数据自举到统一理解生成 - Survey|系列综述]]

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | BLIP-3: A Family of Open Large Multimodal Models |
| 早期标题 | xGen-MM (BLIP-3): A Family of Open Large Multimodal Models |
| 作者 | Le Xue、Manli Shu、Anas Awadalla、Jun Wang、An Yan 等 |
| 机构 | Salesforce AI Research、Intel Labs、University of Washington |
| 发表 | ICCV 2025 Workshops / Findings，pp. 6183–6194 |
| arXiv | 2408.08872；v1：2024-08-16，v4：2025-09-16 |
| 模型家族 | 4B、14B；base、single-image instruction、multi-image/interleaved instruction |
| 一句话结论 | BLIP-3 不是把 BLIP-2 的 Q-Former 做大，而是把整个范式改造成“冻结 SigLIP 视觉塔 + 逐 patch Perceiver 采样 + 可训练 Phi-3 LLM + 单一 next-token loss”，再用 100B→30B→6B→4B 多模态 token 的四阶段数据工程换取高分辨率、OCR 与交错多图能力。 |

> **版本口径。** arXiv v4 和 ICCV Workshops 正式版标题已改为 BLIP-3；项目页、模型仓库与早期引用仍大量使用 xGen-MM。本文以 v4 为主，同时单独核验项目页当前实际发布状态。

---

## 1. 研究背景：BLIP 系列的瓶颈为什么从“对齐”转向“扩展”

### 1.1 BLIP-2 解决了昂贵对齐，却没有解决规模化数据与多图输入

BLIP-2 的核心问题是如何把两个冻结且彼此陌生的底座接起来。它用 32 个可学习 query 构成 Q-Former，先通过 ITC、ITM、ITG 学“哪些视觉信息对语言有用”，再把 query 输出映射成冻结 LLM 的 soft prompt。这个方案参数高效、模块化，也把视觉—语言桥接从大规模端到端训练降为小型 connector 训练。

但到 2024–2025 年，竞争重点已经变化。BLIP-2 的训练数据主要是单图—短描述，结构上也把一张图压成固定 32 token；它没有原生的自由交错图文序列、any-resolution 文档输入与多图上下文学习。更重要的是，Q-Former 的多 mask、多目标、两阶段表示学习虽然精巧，却不利于把训练扩展到上百亿 token 的统一流水线。

### 1.2 BLIP-3 的核心选择：减少专用机制，把预算转移到数据与统一训练

BLIP-3 把问题重新定义为：如果视觉塔和语言模型已经足够强，能否用更简单、可横向扩展的接口，直接在自然图文序列上做语言建模？它做了三处结构性改动：

1. 用 Perceiver Resampler 替代 Q-Former，不再维护 BERT 式文本分支与 ITC/ITM/ITG 多目标；
2. 视觉 token 与文本 token 按原始顺序串入 decoder-only LLM，所有阶段统一为文本 next-token prediction；
3. 从单图配对扩展到大规模 interleaved HTML/PDF/arXiv、OCR、grounding、高分辨率文档和多图指令数据。

![](assets/BLIP-3%20-%20开放大规模多模态模型家族/figures/01-framework-and-stages.png)

> **Figure 1，PDF p.2。** 上半直接对比 BLIP-2 与 BLIP-3：复杂 Q-Former 和 ITC/ITM/ITG 被 scalable vision token sampler 与统一 next-token loss 取代；下半给出四阶段 token 预算：Stage-1 100B、Stage-2 30B、单图微调 6B、多图微调 4B。来源：arXiv v4 TeX 原图。

### 1.3 贡献不只是模型，而是一组“可开放研究对象”

论文把贡献定义为模型、训练配方、代码与数据的组合，而不只是 checkpoint：

- 4B/14B 的 base 与 instruction 模型；
- BLIP3-OCR-200M、BLIP3-GROUNDING-50M、BLIP3-OCR-HD-30M 三个新数据集；
- 100B token 基础预训练与 30B token 高分辨率继续预训练；
- 单图与交错多图两级 SFT；
- 对视觉塔、token sampler、分辨率策略与 instruction-aware sampler 的消融。

论文的“开放”主张很强，但必须区分**论文承诺**与**当前可核验发布**：4B Phi-3-mini 系列、代码、OCR-200M 和 Grounding-50M 有官方入口；官方 collection 中未见 14B 与 OCR-HD-30M 条目。开放性因此是 BLIP-3 的核心价值，也是需要审计的实验变量。

---

## 2. 模型架构：SigLIP、Perceiver 与 Phi-3 如何分工

### 2.1 输入、输出与统一序列

模型输入不是固定的“单图 + 问题”，而是自由形式的交错序列：

$$[I_1,T_1,I_2,T_2,\ldots]$$

每张图先产生视觉 token，每段文字由 Phi-3 tokenizer 产生文本 token，二者按原始顺序拼接进入同一个 LLM self-attention 序列。输出仍是普通文本；训练损失只计算文本位置：

$$\mathcal{L}_{\mathrm{LM}}=-\sum_{t\in\mathcal{T}}\log p_\theta(x_t\mid x_{<t},V)$$

其中 $\mathcal{T}$ 是需要监督的文本 token 位置，$V$ 是前序图像经视觉塔和 sampler 得到的 token。视觉位置不直接承担重建或对比损失，这一点与 BLIP-2 第一阶段有根本差异。

![](assets/BLIP-3%20-%20开放大规模多模态模型家族/figures/02-architecture.png)

> **Figure 2，PDF p.2。** 图像经冻结 Vision Transformer 与可训练 Token Sampler 转为视觉 token，文本单独 tokenization；视觉与文本 token 按自然顺序进入可训练的预训练 LLM。火焰表示 sampler 与 LLM 更新，雪花表示 ViT 冻结。来源：arXiv v4 TeX 原图。

### 2.2 Phi-3：不是“另一个组件”，而是被视觉训练继续改写的语言主体

论文只写使用 **Phi-3 series**，没有在 v4 正文中明确说明 4B/14B 各对应哪个精确 checkpoint。结合官方 4B 仓库名，可确认实际发布的 4B 主线基于 `phi3-mini`；Phi-3 技术报告中 mini 为 3.8B、medium 为 14B，但不能仅据参数量断言论文 14B 模型就是某个具体 `Phi-3-medium` checkpoint。

Phi-3 在 BLIP-3 中承担三项职责：

1. 提供文本 tokenizer、语言先验和 decoder-only 自回归接口；
2. 用 self-attention 同时整合视觉 token、文本上下文和多图示例；
3. 在多模态预训练中**继续更新参数**，把视觉 token 内化为语言模型可直接操作的上下文状态。

第三点是与 BLIP-2/InstructBLIP 的重大断裂。BLIP-2 冻结 LLM，把跨模态适配压力放在 Q-Former；BLIP-3 只冻结视觉塔，Perceiver 与 LLM 都训练。这样更容易通过单一损失扩展，却失去“语言底座完全不变”的参数效率与可诊断性，也可能造成文本能力漂移。论文加入纯文本 OpenOrca、Orca-Math 数据，正是为了缓解视觉微调对语言能力的侵蚀，但没有给出完整 text-only 保持性消融。

### 2.3 CLIP 与 SigLIP：BLIP-3 消费的是对齐后的视觉表示

CLIP 与 SigLIP 都用图文配对学习开放词汇视觉表示，但优化形式不同。设批内图像和文本嵌入相似度为 $s_{ij}$，CLIP 使用双向 batch softmax：

$$\mathcal{L}_{\mathrm{CLIP}}=-\frac{1}{2N}\sum_i\left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}+\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]$$

它把同批其他样本作为负例，归一化依赖全批相似度矩阵。SigLIP 改为独立图文对的 sigmoid 二分类；若 $y_{ij}=1$ 表示匹配、$-1$ 表示不匹配，可写为：

$$\mathcal{L}_{\mathrm{SigLIP}}=-\frac{1}{N^2}\sum_{i,j}\log\sigma\left(y_{ij}(s_{ij}-b)\right)$$

SigLIP 不需要全局 softmax 归一化，分布式训练更简单，小 batch 也更稳定。对 BLIP-3 而言，这些损失只发生在上游视觉塔预训练；正式多模态训练冻结视觉塔，不再重新优化 CLIP/SigLIP 对比目标。

BLIP-3 的视觉 backbone 消融比较的是 DFN 与 SigLIP，而不是原始 CLIP。两者都用 10B token 预训练时，SigLIP 的 4/8-shot TextVQA 为 49.1/50.5，DFN 为 41.1/41.9；TextCaps 为 84.7/88.6 对 78.2/79.9；OK-VQA 基本持平。论文据此选择 SigLIP，最强证据是 OCR 表征，而不是“所有任务全面优于 CLIP”。

### 2.4 Perceiver Resampler：从固定 32 query 改成逐 patch 128 query

ViT 对每个图像 patch 产生 729 个视觉 token；消融中 Perceiver 使用 128 个 query，将每 patch 压缩约：

$$r_c=\frac{729}{128}\approx5.7$$

query 通过 cross-attention 从视觉 token 中读取信息，再把 128 个输出 token 交给 LLM。与 Q-Former 相似之处是两者都用 learnable queries 形成信息瓶颈；不同之处是 Perceiver 没有 BERT 文本分支、ITC/ITM/ITG 的 attention mask 体系，也不先做独立图文表示学习。它是纯视觉重采样器，语义对齐主要由后端语言建模完成。

这种简化降低了 connector 的任务专用性，也使 token 数从 BLIP-2 的“每图固定 32”变成“每个分辨率 patch 固定 128、整图总量随 patch 数增加”。它更适合保留文档和小字细节，但 LLM 上下文成本随图像数和分辨率线性增长。

### 2.5 Any-resolution：先分块，再分别采样，最后拼接

Stage-2 和全部 SFT 采用动态高分辨率：

1. 尽量保留原始宽高比，把图像切成多个 $384\times384$ patch；
2. 每个 patch 独立经过冻结 SigLIP；
3. 每个 patch 的 729 token 独立经 Perceiver 压到 128；
4. 额外加入一张缩小的全图，提供全局信息；
5. 所有采样结果拼接后送入 Phi-3。

正式 Stage-2 支持最多 12 个局部 patch，最长边 1536；消融表又把 SFT 上限写为 13 image patches，最合理的口径是“12 局部 + 1 全图”，但论文没有明确解释二者差异。若 13 组都各输出 128 token，则理论上单图最多约 1664 视觉 token；这是根据图示和消融配置的推算，不是论文直接报告值。

逐 patch 采样与“先把所有 patch embedding 拼起来，再统一采成固定 128/256 token”不同。后者在高分辨率图像上压缩率随 patch 数急剧升高，小字容易消失；BLIP-3 让视觉 token 预算随图像信息量增长，用更多上下文成本换取 OCR 保真度。

---

## 3. 四阶段训练：统一 loss 不等于单阶段训练

### 3.1 阶段总览

| 阶段 | token 预算 | 图像策略 | 主要数据 | 直接目标 |
|---|---:|---|---|---|
| Stage-1 基础分辨率预训练 | 100B multimodal tokens | $384\times384$ | interleaved + caption + OCR + grounding | 建立基础图文对齐、OCR、grounding 与 in-context 能力 |
| Stage-2 高分辨率预训练 | 30B multimodal tokens | 最多 12 局部 patch，最长边 1536 | 加入 OCR-HD-30M、IDL，调整采样比例 | 补高分辨率 OCR/文档能力，缩小预训练与 SFT 输入差异 |
| Single-image SFT | 6B multimodal tokens | 与 Stage-2 相同 | 约 3M 公共多模态/纯文本指令样本，1 epoch | 指令跟随与单图 benchmark 能力 |
| Interleaved multi-image SFT | 4B multimodal tokens | 同样 any-resolution | MANTIS、MMDU 等多图数据 + 部分单图回放 | 恢复并增强多图、交错问答与 multimodal ICL |

四阶段使用同一类文本自回归损失，但数据分布、输入分辨率和任务格式逐步变化。“统一目标”解决的是工程与扩展复杂度，不代表一个 checkpoint 用同一数据混合一次训完。

### 3.2 Stage-1：先用低分辨率把规模做上去

Stage-1 将 SigLIP 默认的 $384\times384$ 作为统一输入，以约 100B 多模态 token 建立图文对齐。低分辨率阶段便宜，适合吸收 MINT-1T、OBELICS 等超大规模交错数据，也避免一开始就为每张图支付多 patch 成本。

Figure 5 的数据规模消融从 2B、60B 扩到 100B token：2B→60B 在 COCO-Caps 与 TextCaps 上显著提升，60B→100B 仍有收益但已明显趋缓。该结果支持“规模有效”，同时也暴露边际收益递减；论文没有把额外 40B token 的收益与算力成本一起报告。

### 3.3 Stage-2：用 30B token 专门买回分辨率

Stage-2 从 Stage-1 checkpoint 继续训练，引入高分辨率 any-resolution，并用 OCR-HD-30M 替换 OCR-200M、加入 IDL 文档数据。消融显示：没有 Stage-2 时 OCR 平均 64.6；最多 4 patch 时 65.3；最多 12 patch 时 67.2。GeneralVQA 分别为 63.5、63.2、63.3，基本不变。

这说明 Stage-2 的作用非常集中：它不是普遍增强“视觉智能”，而是让模型适应高分辨率 token 布局并学习读取文字。把 OCR 增益解释成 LLM 推理能力提升是不准确的。

### 3.4 单图 SFT：3M 样本、1 epoch，但会破坏多图能力

单图 SFT 汇集约 3M 公共样本，覆盖多模态对话、caption、通用/知识/OCR VQA、文档图表、科学数学，并混入 OpenOrca、Orca-Math 等纯文本指令数据。训练 1 epoch，Figure 1 折算为约 6B multimodal tokens。

关键反直觉结果是：base model 已在交错数据上预训练，但单图 SFT 后多图 benchmark 明显下降。原因不是架构不能处理多图，而是 SFT 分布把模型重新塑造成“一张图对应一个指令”的局部最优。预训练能力不会自动抵抗后续任务格式遗忘。

### 3.5 多图 SFT：不是额外功能包，而是恢复被单图 SFT 擦掉的能力

MI 阶段从 SI checkpoint 出发，使用 MANTIS、MMDU 等公共多图/交错指令数据，并回放一部分单图样本，Figure 1 给出约 4B multimodal tokens。它同时改善跨图比较、图像对应关系、交错问答与 multimodal in-context learning。

![](assets/BLIP-3%20-%20开放大规模多模态模型家族/figures/05-multi-image-examples.png)

> **Figure 6，PDF p.8。** 三个样例依次展示跨图选择、跨图对象对应，以及把论文 Figure 3 当作上下文回答比例问题。第三例尤其说明“多图/交错能力”不仅是同时看多张自然图，还包括把图表嵌入上下文后继续语言推理。来源：arXiv v4 TeX 原图。

### 3.6 DPO 的版本陷阱

项目页和官方 Hugging Face 确实发布了 `xgen-mm-phi3-mini-instruct-dpo-r-v1.5`，报告 VLGuard 违规率 5.2、HallusionBench 56.6、POPE 86.8 等结果；项目页也称它用于减少有害行为与幻觉。

但 **arXiv v4 正文没有 DPO 阶段**。TeX 源码中 DPO、VLFeedback、LoRA 和 safety fine-tuning 段落全部被注释，正式主表也不包含该模型。因此：

- 可以说“xGen-MM 项目另有已发布 DPO 变体”；
- 不能说“BLIP-3 v4 论文的正式四阶段训练包含 DPO”；
- 也不能把被注释的 62.6K preference、1 epoch DPO 等细节当成已正式报告的论文方法。

---

## 4. 数据配方与治理：开放性的实质和缺口

### 4.1 Stage-1 数据比例

![](assets/BLIP-3%20-%20开放大规模多模态模型家族/figures/03-stage1-data-recipe.png)

> **Figure 3，PDF p.5。** Stage-1 由自然交错文档、dense caption、OCR、grounding 与传统图文对共同组成。MINT-1T、OBELICS 合计 50%，说明 base model 的上下文结构主要来自真实文档，而不是把独立 image-caption pair 人工拼接。来源：arXiv v4 TeX 原图。

| 数据源 | Stage-1 比例 | 角色 |
|---|---:|---|
| MINT-1T | 32.5% | HTML/PDF/arXiv 交错文档；其内部按 7:5:1 混合 |
| OBELICS | 17.5% | HTML 交错文档 |
| BLIP3-KALE | 25% | 知识增强、高质量 dense caption |
| Datacomp-1B | 10% | 未整理的通用图文对 |
| BLIP3-OCR-200M | 5% | OCR 文本监督 |
| BLIP3-GROUNDING-50M | 5% | 对象与位置监督 |
| CC12M / CC3M / SBU / Visual Genome | 各 1.25% | 传统 caption 与区域语义数据 |

这组配方把 50% 预算给交错文档、35% 给 KALE+DataComp、10% 给专项 OCR/grounding、5% 给传统数据。与 BLIP-2 的单图 caption 中心配方相比，数据结构本身已经在教模型“图片如何嵌在长文本中”。

### 4.2 三个新数据集如何造监督

![](assets/BLIP-3%20-%20开放大规模多模态模型家族/figures/04-dataset-examples.png)

> **Figure 4，PDF p.6。** 左侧是高分辨率发票 OCR；右上展示 OCR-200M 的六级文本/位置表达；右下展示 Grounding-50M 的 coarse 与 fine-grained 定位 caption。图中也暴露监督并非纯人工：OCR、tagger、detector 与模板共同生成标注。来源：arXiv v4 TeX 原图。

**BLIP3-OCR-200M** 从 DataComp-1B 选 2 亿图像，用现成 OCR engine 自动标注。数据提供六级粒度，既可只有文本，也可包含归一化 bbox。BLIP-3 实际预训练只使用不带 bbox 的文字信息，并在 dataloader 中删除 “the text” 等容易预测的 filler phrase，防止简单模板 token 稀释真正 OCR token 的损失。

**BLIP3-GROUNDING-50M** 从 DataComp-1B 整理 5000 万图像，以开放世界 tagging 与 object detector 识别对象，再把位置注入 caption。位置可表达为精确 bbox、自然语言坐标或粗粒度方位。模型学习的 grounding 因此高度依赖上游检测器质量和 caption 中对象短语的匹配准确率。

**BLIP3-OCR-HD-30M** 只保留宽高都大于 512 的图像，并重新以高分辨率 PaddleOCR 标注。它不是把 OCR-200M 直接放大，而是重新生成与高清输入匹配的监督，避免图像细节增加、标签信息量却不变。

### 4.3 Stage-2 配方有一个未解释的 10% 缺口

正文说 Stage-2 以 Stage-1 为基础：OCR-200M 换成 OCR-HD-30M（10%），新增 IDL（10%），KALE 从 25% 降到 10%，DataComp 从 10% 降到 2.5%，Grounding 从 5% 降到 2.5%，其余不变。按 Figure 3 的 Stage-1 数字直接相加，明确列出的 Stage-2 比例只有 90%。

论文没有解释剩余 10% 是重新归一化、另一数据源还是文字疏漏。复现时不能擅自把各项归一化为 100%，因为这会改变交错数据和专项 OCR 的相对权重。这个缺口与缺少完整训练配置一起，削弱了“完整 recipe 可复现”的强主张。

### 4.4 数据治理：论文披露了处理，但没有披露完整治理链

论文明确披露的治理动作包括：MINT-1T 子集 7:5:1 混合；OCR-HD 尺寸过滤；OCR/grounding 自动标注；filler phrase 清理；Stage-2 权重调整；MI 阶段回放单图数据。模型卡另外说明因已知 CSAM 风险排除了 LAION，并提醒互联网数据、LLM 与商业 API 可能带来偏见。

但论文没有系统披露以下关键环节：

- URL、图像与 caption 的精确去重和 near-duplicate 策略；
- benchmark contamination 检查；
- PII、隐私、人脸与敏感 OCR 文本过滤；
- NSFW、暴力和其他有害内容分类器及阈值；
- copyright、许可证兼容性、opt-out 与删除机制；
- 机器标注噪声率、人工抽检比例与不同语言覆盖；
- 上游 detector/OCR/API 的版本、偏差与失败模式。

因此 BLIP-3 的数据开放度高于只放权重的模型，但“数据可下载”不等于“数据来源、权利与清洗决策完全可审计”。

### 4.5 开放性审计：论文声称、当前可见与真正可复现

| 层级 | 论文声称 | 当前可核验 | 仍缺失/不确定 |
|---|---|---|---|
| 权重 | 4B、14B；base 与 instruction | 4B Phi-3-mini base、single-image、interleave、DPO | 官方 collection 未见 14B |
| 数据 | 三个新数据集及所有使用/预处理数据 | KALE、OCR-200M、Grounding-50M 等有入口 | 未见 OCR-HD-30M；上游许可证异质 |
| 代码 | training code | LAVIS `xgen-mm` 分支、模型推理 notebook | 完整 130B-token 预训练超参数、集群配置未在论文中披露 |
| 许可证 | 开放模型/代码 | 模型卡称权重与代码 Apache-2.0 | 不能覆盖所有上游训练数据许可证 |
| 可复现性 | 开放 recipe | 主要阶段、数据比例、架构和部分消融明确 | optimizer、LR、batch、steps、算力、14B 映射与 Stage-2 10% 缺口 |

BLIP-3 因而应被评价为“比同期 open-weight 模型更开放的数据/代码框架”，而不是“从数据到训练完全可一键复现”。

---

## 5. 定量结果：竞争力来自哪里，又没有覆盖什么

### 5.1 单图主结果

| 模型 | 规模 | TextVQA | OCRBench | MMVet | MMMU | MathVista | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phi-3-vision | 4B | 73.3 | 63.7 | 43.2 | 46.1 | 45.1 | 58.4 |
| Phi-3.5-vision | 4B | 72.0 | 59.9 | 43.2 | 43.0 | 43.9 | 58.8 |
| **BLIP-3-4B** | 4B | **76.5** | **66.0** | 41.2 | 42.3 | 44.5 | **59.2** |
| Pixtral | 12B | 75.7 | 72.2 | **58.5** | **51.1** | **56.3** | **64.9** |
| Cambrian | 13B | 72.8 | 61.0 | 48.9 | 41.6 | 47.7 | 59.8 |
| **BLIP-3-14B** | 14B | **79.0** | **71.1** | 44.5 | 47.0 | 45.4 | 62.0 |
| GPT-4V | 闭源 | 78.0 | 67.8 | 49.0 | 53.8 | 48.2 | 62.1 |

BLIP-3-4B 的平均分高于 Phi-3-vision/Phi-3.5-vision，优势主要来自 TextVQA、OCRBench、RealWorldQA 等感知与文字任务；它在 MMVet、MMMU 上并未超过 Phi-3-vision。BLIP-3-14B 接近 GPT-4V 的表内平均，但低于 Pixtral-12B 的 64.9，尤其在 MMVet、MMMU、MathVista 上差距明显。

这组结果与训练配方一致：两阶段 OCR 数据和 any-resolution 显著提高读字能力，但大规模感知数据不会自动变成高级科学、数学和复合推理能力。论文的正确结论是“同规模开放模型中有竞争力”，不是全面 SOTA。

### 5.2 多图主结果：MI SFT 的收益远大于多数结构微调

| 模型 | BLINK | QBench | MuirBench | Mantis-Eval |
|---|---:|---:|---:|---:|
| BLIP-3-4B-SI | 42.9 | 51.9 | 35.0 | 49.3 |
| **BLIP-3-4B-MI** | **47.1** | **69.6** | **38.0** | **56.2** |
| BLIP-3-14B-SI | 47.1 | 55.4 | 49.1 | 55.3 |
| **BLIP-3-14B-MI** | **53.9** | **73.4** | **56.2** | **61.3** |
| GPT-4V | 51.1 | 73.4 | 未报告 | 62.7 |

4B 的 MI−SI 增益分别为 +4.2、+17.7、+3.0、+6.9；14B 为 +6.8、+18.0、+7.1、+6.0。QBench 的约 18 点提升说明数据格式对能力保持的影响远大于“base 预训练中见过交错数据”这一事实。多图能力需要在最后的 instruction 分布中持续出现。

### 5.3 幻觉与安全结果不能被过度解读

主表包含 POPE 和 HallusionBench：BLIP-3-4B 为 87.9/36.5，14B 为 86.0/38.7。它们没有随着参数规模同步改善；14B 的 POPE 反而略低。项目级 DPO 变体报告更好的 VLGuard/POPE/HallusionBench，但不在 v4 正式训练链中，且模型卡没有披露足以独立重建 DPO 的完整数据与超参数。

单一 hallucination benchmark 也不覆盖拒答校准、隐私泄漏、偏见、有害 OCR 内容或跨图错误归因。BLIP-3 的开放性有助于后续安全研究，但论文自身不是完整安全评测。

---

## 6. 消融：真正有效的设计与反直觉结果

### 6.1 视觉 backbone：SigLIP 的优势集中在 OCR

| Backbone（10B token 预训练） | TextVQA 4/8-shot | OK-VQA 4/8-shot | COCO-Caps 4/8-shot | TextCaps 4/8-shot |
|---|---:|---:|---:|---:|
| DFN | 41.1 / 41.9 | **48.4 / 49.5** | 107.2 / 109.4 | 78.2 / 79.9 |
| SigLIP | **49.1 / 50.5** | 48.4 / 48.9 | **108.7 / 110.2** | **84.7 / 88.6** |

SigLIP 明显提升 TextVQA/TextCaps，对 OK-VQA 几乎没有帮助。这表明它改善的是文字与视觉表征质量，而非 LLM 的世界知识。

### 6.2 Perceiver 并非每项都优于简单 MLP

| Vision token sampler | GeneralVQA | OCR | Sci. & Math | Hallucination |
|---|---:|---:|---:|---:|
| **Perceiver Resampler** | **63.1** | 70.4 | **43.4** | 62.2 |
| MLP + $2\times2$ pooling | 61.7 | **72.2** | 41.8 | **62.7** |

Perceiver 的压缩率更高（5.7 对 4），GeneralVQA 与 Sci/Math 更好，但 OCR 和 hallucination 指标略低。作者选择 Perceiver 的关键理由不是所有指标最优，而是多图场景下更高压缩率能显著节省 LLM 上下文。这个工程权衡比“Perceiver 更强”的笼统结论更准确。

### 6.3 逐 patch 采样是高分辨率收益的关键

![](assets/BLIP-3%20-%20开放大规模多模态模型家族/figures/06a-anyres-ablation.png)

> **Figure 7(a)，PDF p.8。** base resolution 最弱；把所有 patch 拼后统一采 128/256 token 虽有提升，但逐 patch 独立采样在 DocVQA、ChartQA、OCRBench 上都最好。收益来自避免总 token 固定造成的过度压缩。来源：arXiv v4 TeX 原图。

固定分辨率把整图缩入视觉塔默认尺寸，文字直接在像素层消失；`anyres-fixed-sampling` 保留像素，却把多个 patch 统一压成固定 token，瓶颈转移到 sampler；BLIP-3 的 patch-wise 方案同时保留像素和每 patch token 预算。它证明“支持任意分辨率”不能只看预处理是否切图，还要看 connector 是否让 token 预算随图像复杂度增长。

### 6.4 Instruction-aware sampler 没有复现 InstructBLIP 的优势

![](assets/BLIP-3%20-%20开放大规模多模态模型家族/figures/06b-instruction-aware-ablation.png)

> **Figure 7(b)，PDF p.8。** 把 instruction token 附加到 Perceiver query 后，Perception、OCR、Sci/Math 与 Hallucination 平均分都与原版接近，部分略升、部分略降。作者因此保留更简单的非 instruction-aware sampler。来源：arXiv v4 TeX 原图。

这不是对 InstructBLIP 结论的否定。InstructBLIP 的 instruction token 与 Q-Former query 在多层共享 self-attention 中交互，再由 query 跨注意力读图；BLIP-3 只是把文本 token 附到 Perceiver query，机制和初始化都不同。另一方面，BLIP-3 每 patch 已保留 128 token、总视觉带宽远大于 InstructBLIP 固定 32 token，任务条件化压缩的重要性自然下降。

### 6.5 消融仍不完整

论文没有系统消融：

- 32/64/128/256 query 的完整质量—成本曲线；
- 最大 patch 数与 LLM 上下文长度的联合影响；
- 冻结/解冻 Phi-3 对文本能力、视觉能力与灾难性遗忘的影响；
- Stage-1 各数据源逐项移除；
- OCR/grounding 自动标注噪声的影响；
- 4B 与 14B 是否共享完全相同的最佳配方；
- 多图 SFT 回放多少单图数据才足以避免遗忘。

因此论文证明了所选方案“可行且有竞争力”，但没有证明它是全局最优配方。

---

## 7. 与 BLIP-2、InstructBLIP 的继承与断裂

### 7.1 三代方法对照

| 维度 | BLIP-2 | InstructBLIP | BLIP-3 |
|---|---|---|---|
| 视觉塔 | 冻结 CLIP/EVA-CLIP | 冻结 EVA-CLIP | 冻结 SigLIP |
| Connector | 32-query Q-Former | instruction-aware 32-query Q-Former | 每 patch 128-query Perceiver Resampler |
| LLM | 冻结 OPT/FlanT5 | 冻结 FlanT5/Vicuna | **可训练 Phi-3 series** |
| 预训练目标 | ITC + ITM + ITG，再做 LM | 继承 BLIP-2，再做 instruction tuning | 全阶段统一文本 next-token loss |
| 图像分辨率 | 主要 $224^2$ | $224^2$ | 基础 $384^2$；Stage-2/SFT any-resolution |
| 多图/交错 | 非原生 | 视频仅帧级拼接，非原生训练 | 原生 interleaved 预训练 + MI SFT |
| 视觉 token 预算 | 每图固定 32 | 每图固定 32，受指令条件化 | 每 patch 固定 128，整图总量动态 |
| 数据中心 | 单图 caption | 26 个任务、held-in/out instruction | 100B+30B 大规模混合数据与专项 OCR/grounding |

### 7.2 继承：learnable query 仍然是视觉—语言接口

BLIP-3 并没有抛弃 BLIP-2 最重要的归纳偏置：不要把全部 ViT patch token 原样灌给 LLM，而是用少量可学习 query 读取视觉网格。Perceiver 与 Q-Former 都把视觉接口变成可训练的信息瓶颈，也都允许冻结昂贵的视觉塔。

它还继承了 BLIP 系列的模块化思想：视觉 backbone、connector 和语言模型可分别替换；训练数据负责塑造能力，connector 负责控制视觉带宽，LLM 负责生成。

### 7.3 断裂一：从“冻结两端、训练桥”到“冻结视觉、训练语言主体”

BLIP-2 的安全边界是 LLM 不变，Q-Former 负责适配；BLIP-3 让 Phi-3 全量参与多模态学习。收益是对齐更直接、统一 loss 更易扩展，视觉 token 可在所有语言层中通过 self-attention 被重新组合。代价是训练参数与显存大幅增加，文本能力可能漂移，模型不再是轻量 connector 训练范式。

### 7.4 断裂二：从固定语义瓶颈到随分辨率扩张的视觉带宽

BLIP-2/InstructBLIP 每图固定 32 token，计算稳定，但 OCR、文档和密集对象容易在压缩时丢失。BLIP-3 每 patch 128 token，把信息量与上下文成本绑定：图越大、图越多，送入 Phi-3 的视觉 token 越多。它牺牲固定成本，换取高分辨率和多图保真度。

### 7.5 断裂三：从显式对齐目标到隐式 next-token 对齐

BLIP-2 用 ITC 学全局对齐、ITM 学细粒度匹配、ITG 学图像条件生成；BLIP-3 把三者压成一个语言建模目标。优点是数据格式统一，任何交错文本都能训练；缺点是模型可能通过语言先验预测容易 token，视觉 grounding 不再由独立目标强制保证。OCR filler phrase 清理正说明：当唯一目标是 next-token loss 时，模板中大量容易文本会稀释真正视觉证据的梯度。

### 7.6 InstructBLIP 的继承与否定是“有条件的”

BLIP-3 继承了 InstructBLIP 的问题意识：视觉采样是否应由用户指令控制；也直接实现了 instruction-aware Perceiver 消融。但结果没有显著收益，于是正式模型回到 instruction-agnostic sampler。

这形成一条清晰演化链：BLIP-2 用固定 32 query 学通用视觉摘要；InstructBLIP 用 instruction 改写这 32 个 query，减少任务无关信息损失；BLIP-3 则通过更高的动态视觉带宽缓解瓶颈，使指令条件化不再是必需。它不是简单“退回 BLIP-2”，而是用 token 预算和数据规模替代了 connector 内的任务条件化。

---

## 8. 局限、风险与批判性评价

### 8.1 论文没有正式 Limitations 章节

v4 从结论直接进入参考文献，没有独立 limitations、broader impact 或安全章节。能从实验直接确认的限制包括：

1. 单图 SFT 会显著破坏多图能力，训练阶段之间存在灾难性遗忘；
2. Perceiver 的 OCR 分数低于 MLP，压缩效率与文字保真度不能同时最优；
3. 60B→100B token 收益趋缓，继续扩数据的成本效益不明；
4. 固定总 token 的多 patch 采样会丢失文字，说明 connector 仍是信息瓶颈；
5. instruction-aware sampler 未显著改善常见领域，机制有效性依赖架构与带宽；
6. 14B 在高级推理上仍明显落后 Pixtral-12B，数据规模没有消除 reasoning 短板。

### 8.2 论文未披露的复现关键项

正文没有给出 optimizer、各模块学习率、batch size、warmup、训练 steps、context length、精度、并行策略、硬件数量、训练时间与能耗。也没有明确 4B/14B 的具体 Phi-3 checkpoint、Perceiver 层数/隐藏维度/heads、14B 是否使用相同视觉配置。

因此可以复现其**架构思想和下游微调**，却无法仅凭论文严格复现 100B+30B token 的 base model。开放代码不自动补足未发布的数据快照、采样器状态和集群配置。

### 8.3 数据与模型风险

互联网网页、图片站、PDF、arXiv 与机器生成标注会带来：

- 网页偏见、刻板印象与错误知识；
- OCR 对姓名、地址、发票和界面内容的隐私复制风险；
- detector/tagger 错误被模板化为看似精确的 grounding 标签；
- 数据重复和 benchmark 泄漏；
- 对版权与许可证来源不清晰的图像进行再分发；
- 多图问答中把一个图像的属性错误绑定到另一图像；
- 可训练 LLM 在视觉微调后产生文本能力漂移与新的幻觉模式。

模型卡排除 LAION 是积极治理信号，但论文没有给出完整过滤器、阈值、误杀漏检和审计报告，不能据此推断风险已解决。

### 8.4 个人判断

**已确立事实：** BLIP-3 证明“冻结强视觉塔 + 动态 Perceiver 视觉带宽 + 可训练小型 LLM + 大规模交错/OCR 数据”可以构建有竞争力的 4B/14B LMM；高分辨率 Stage-2 对 OCR 有明确因果证据，多图 SFT 对恢复交错能力有大幅收益。

**我的判断：** 这篇工作的最大贡献是把 BLIP 系列从 connector-centric 研究转成 data-and-systems-centric 研究。BLIP-2 的问题是如何桥接两个冻结模型；BLIP-3 的问题是如何让统一序列训练在 130B 预训练 token、高分辨率和多图输入上可扩展。Q-Former 被替换不是因为 learnable query 失效，而是因为其专用多目标训练不再匹配新的规模化工程目标。

但“开放模型家族”这一标题也应接受与 benchmark 同等严格的审计：当前可见发布与 v4 声称的 14B、OCR-HD-30M、全部预处理数据并不完全对应；关键超参数和 Stage-2 配方仍有缺口。BLIP-3 比多数 open-weight 工作更开放，却还没有达到完整可重复训练实验的标准。

---

## 9. 复现与实践建议

### 9.1 最小可信复现

若目标是验证论文核心结论，而非重训 130B token 模型，建议固定官方 4B base checkpoint，完成以下对照：

1. 单图 SFT 与单图后再 MI SFT，分别报告 BLINK/QBench/MuirBench/Mantis-Eval，验证多图遗忘与恢复；
2. base resolution、anyres-fixed-sampling、anyres-patch-sampling 三组 OCR 对照；
3. Perceiver-128 与 MLP+$2\times2$ pooling，联合报告质量、视觉 token 数、显存与吞吐；
4. instruction-aware 与普通 Perceiver，在同一 checkpoint、同一数据和同一随机种子下比较；
5. 附加 text-only benchmark，检查可训练 Phi-3 在多模态微调后的语言能力漂移。

### 9.2 工程注意事项

- 多图输入的主要成本落在 Phi-3 self-attention，而不是冻结 SigLIP；必须限制每图 patch 数和总视觉 token。
- 逐 patch 采样不能误实现为“全部 patch 拼接后固定采样”，两者正是 Figure 7(a) 的关键差异。
- 训练与评测应固定图像网格、全图 thumbnail、patch 顺序和视觉 token 分隔方式；这些细节会影响 OCR 与跨图引用。
- 数据混合必须保存每步数据源、采样概率和随机种子；Stage-2 明示比例只合计到 90%，复现报告需公开如何处理剩余 10%。
- 不应把 DPO 模型卡结果混入 v4 SI/MI 主表；它们是不同训练链与不同证据等级。

### 9.3 复现成本判断

官方 Apache-2.0 权重和 LAVIS 分支使推理、LoRA/局部微调和 MI 能力验证相对可行；完整预训练需要 130B multimodal tokens、海量图像存储、OCR/grounding 预处理与多阶段分布式训练。论文未披露硬件与 GPU-hours，无法可靠估算论文级成本。对个人或小团队，合理目标应是基于 4B checkpoint 做结构/数据消融，而不是从零重建全部模型家族。

---

## 10. 后续问题

1. 若视觉 token 预算已经随分辨率增长，能否用内容自适应 query 数，在空白区域少采、文字密集区域多采？
2. instruction-aware sampler 在 OCR、referring、计数等“必须选择局部证据”的单项任务上是否仍无收益，还是被领域平均分掩盖？
3. 能否在 MI SFT 中用少量 replay 或正则化保留多图能力，而不额外花 4B token？
4. Perceiver 的高压缩率带来多图效率，但 MLP OCR 更高；是否可使用 OCR-aware 路由或双分支 connector？
5. 训练 LLM 是统一 loss 成功的关键，还是冻结 Phi-3 + 更强 connector 也能接近结果？论文没有直接消融。
6. 14B 与 4B 的收益主要来自语言容量还是视觉—语言训练规模？缺少同数据、同 token 的 scaling analysis。
7. 当训练数据真正完整发布后，Stage-2 缺失的 10% 与 OCR-HD-30M 的许可证、去重、隐私治理应如何核验？

---

## 11. 材料来源

官方 PDF、7 张 arXiv TeX 完整原图、Figure/Page 映射、项目页、模型卡、数据集与上游论文链接见：[sources.md](assets/BLIP-3%20-%20开放大规模多模态模型家族/sources.md)。
