---
type: paper-note
status: complete
domain: VLM
paper: "How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites"
year: 2024
arxiv: "2404.16821"
doi: null
source: "https://arxiv.org/abs/2404.16821"
project: "https://internvl.opengvlab.com"
code: "https://github.com/OpenGVLab/InternVL"
tags:
  - VLM
  - dynamic-high-resolution
  - pixel-shuffle
  - OCR
  - bilingual
  - InternVL
created: 2026-10-16
updated: 2026-10-16
---

# InternVL 1.5：动态高分辨率多模态套件

## 0. 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites |
| 作者 | Zhe Chen、Weiyun Wang、Hao Tian 等 |
| 总参数 | 约 26B：InternViT-6B + InternLM2-Chat-20B + MLP projector |
| 核心改动 | 连续训练视觉 encoder、动态高分辨率、英文/中文高质量数据 |
| 分辨率 | 训练 1–12 个 $448\times448$ tile；测试可扩至 40 tile（约 4K） |
| 本地论文 | [paper.pdf](assets/InternVL%201.5%20-%20动态高分辨率多模态套件/sources/paper.pdf) |
| 来源清单 | [sources.md](assets/InternVL%201.5%20-%20动态高分辨率多模态套件/sources.md) |

**一句话结论：** InternVL 1.5 以一个很直接的 ViT–MLP–LLM 架构，把能力提升集中到 6B 视觉底座、按宽高比切成 448px tiles、thumbnail 全局视图、pixel shuffle token 压缩和高比例双语 OCR 数据上；它确立了后续 InternVL2/2.5/3 一直沿用的动态高分辨率主干。

---

## 1. 研究背景与核心贡献

### 1.1 开源与闭源模型的三类差距

论文把 2024 年初的差距归纳为：开源 MLLM 通常用约 300M 视觉 encoder 配 7B/13B LLM，视觉规模偏小；固定 336/448 方形输入会破坏宽高比并丢失文档细节；训练数据以英文为主，中文 OCR 和跨语言场景弱。InternVL 1.5 没有发明复杂 connector，而是分别补强视觉容量、输入分辨率和数据覆盖。

### 1.2 三项核心改进

1. **Strong Vision Encoder**：将 InternViT-6B 从 224px 持续训练到 448px，再从固定 448 推进到动态 1–12 tiles，并证明其特征可从 Yi-34B 迁移到 InternLM2-20B。
2. **Dynamic High Resolution**：按原图宽高比匹配预定义网格，resize 后切成 $448\times448$ tiles，并追加 thumbnail；测试可零样本扩到 40 tiles。
3. **High-Quality Bilingual Dataset**：预训练中 OCR-large 占 32.0%、OCR-small 占 8.9%，再通过 LLM 翻译管线把部分英文 caption、grounding、conversation 数据转为中文。

---

## 2. 输入输出、shape 与总体架构

### 2.1 数据流

输入是一条包含图像与用户文本的消息，输出为文本回答。图像先经 dynamic high-resolution 预处理得到 $n$ 个局部 tile；当 $n>1$ 时再加入一个全图 thumbnail。每个 tile 独立进入 InternViT-6B，视觉 token 经 pixel shuffle 降采样，再由随机初始化 MLP projector 映射到 InternLM2 hidden dimension，与文本 embedding 拼接后由 decoder-only LLM 生成。

![](assets/InternVL%201.5%20-%20动态高分辨率多模态套件/figures/02-architecture.png)

> 图 1：Figure 3。主干是直接的 ViT–pixel shuffle–MLP–LLM；视觉与文本最终在 InternLM2 causal self-attention 中融合，没有 QLLaMA、Q-Former 或多层 cross-attention。

### 2.2 单 tile 的 token shape

InternViT patch size 为 14，因此 $448\times448$ tile 形成 $32\times32=1024$ 个 patch token。论文称 pixel shuffle 将 token 数减为四分之一；从张量视角，它把相邻 $2\times2$ 空间 token 重排进通道：

$$F\in\mathbb{R}^{32\times32\times C}\rightarrow\widetilde F\in\mathbb{R}^{16\times16\times4C}$$

随后展平为 256 个 token，再经 MLP 投到 LLM hidden size $D_l$：

$$Z=\operatorname{MLP}(\operatorname{reshape}(\widetilde F))\in\mathbb{R}^{256\times D_l}$$

论文使用“pixel shuffle”命名；按常见图像算子语义，这个空间降采样、通道扩张操作更接近 pixel unshuffle。复现应以发布代码的 reshape/permute 顺序为准。

### 2.3 多 tile 的总长度

若原图匹配到 $n$ 个局部 tile，视觉 token 约为 $256n$；多 tile 时加入 thumbnail 后约为 $256(n+1)$。因此训练最大 12 tiles 时为 $13\times256=3328$，测试最大 40 tiles 时为 $41\times256=10496$。单 tile 情况可不重复加入 thumbnail，所以最小值为 256。

视觉序列与用户文本 embedding 组成 LLM 输入，最终 shape 为：

$$X\in\mathbb{R}^{(N_{vis}+N_{text})\times D_l}$$

---

## 3. Dynamic High Resolution：tile、thumbnail 与预算选择

### 3.1 宽高比匹配

训练阶段枚举 1–12 个 tile 能组成的 35 种网格宽高比，如 $1{:}1,1{:}2,2{:}1,3{:}1,\ldots,2{:}6$。对原图宽高比 $r=W/H$，选择绝对差最小的候选 $a/b$；若多个网格同样匹配，优先选择目标面积不超过原图面积两倍的方案，避免把低分辨率图像过度放大。

$$a^*,b^*=\arg\min_{(a,b)\in\mathcal{R}}\left|\frac{W}{H}-\frac{a}{b}\right|$$

图像随后 resize 到 $(448a)\times(448b)$ 并切为 $ab$ 块。这个过程不保留原像素尺寸，但尽量保留宽高比与局部可读性。

### 3.2 Thumbnail 的作用

局部 tile 放大细节，却会破坏跨 tile 的对象关系和整页布局。额外 $448\times448$ thumbnail 提供低频全局视图，使 LLM 同时看到“局部文字/图表细节”和“整图结构”。它不是特征金字塔中的共享分支，而是又一幅独立编码的图像，因此额外增加 256 个视觉 token。

![](assets/InternVL%201.5%20-%20动态高分辨率多模态套件/figures/03-dynamic-resolution.png)

> 图 2：Figure 4。$800\times1300$ 图像匹配到 $2{:}3$ 网格，resize 为 $896\times1344$，切成 6 个 tile，并追加一个 thumbnail。

### 3.3 计算—精度权衡

tile 数同时增加 ViT 前向次数、LLM prefill 长度和 KV cache。40 tiles 不是免费“支持 4K”：单图就有 10,496 个视觉 token，尚未计问题和输出。模型允许调用者按任务选预算：普通场景描述用低分辨率；DocVQA、InfoVQA、TextVQA、OCRBench 用更多 tile。

---

## 4. Q/K/V、mask、position、目标与生成

### 4.1 Vision Transformer

每个 tile 独立做 ViT 双向 self-attention：

$$Q_v=HW_Q^v,\quad K_v=HW_K^v,\quad V_v=HW_V^v$$

同一 tile 内 1024 patch 互相可见，没有 causal mask；padding/resize 已在切图阶段规则化。tile 之间不在 ViT 内直接 attention，跨 tile 信息只能在进入 LLM 后融合，thumbnail 因而很重要。

### 4.2 LLM 融合

视觉 token 经 MLP 后与文本 token 共用 InternLM2 self-attention：

$$Q_l=XW_Q^l,\quad K_l=XW_K^l,\quad V_l=XW_V^l$$

使用标准 causal mask；问题与答案可读取此前视觉 token，未来答案不可见。论文没有插入额外 cross-attention，视觉信息和语言信息都依赖同一 LLM attention 混合。

### 4.3 Position

- tile 内二维位置由 InternViT 的视觉位置表示承担。
- tile 顺序、thumbnail 与用户消息进入 LLM 后使用一维 token position；论文没有提出专门二维 RoPE 或跨 tile 坐标编码。
- 因为每块都重用相同视觉位置网格，绝对全图位置主要由 tile 排列顺序和训练格式间接学习。对复杂版面，这是一个隐含瓶颈。

### 4.4 训练目标与生成

论文沿用 LLaVA-1.5 response formatting，以文本 next-token loss 训练：

$$\mathcal{L}_{\mathrm{text}}=-\sum_{t\in\mathcal{A}}\log p_\theta(y_t\mid y_{<t},Z,x)$$

$\mathcal{A}$ 表示监督回答 token；论文未引入独立 contrastive、region regression、OCR CTC 或重建损失。视觉能力通过文本生成误差回传到 MLP 与可训练 ViT。推理时标准自回归生成；grounding 等输出也由文本化坐标表达。

---

## 5. Strong Vision Encoder：连续学习而非一次初始化

### 5.1 从 InternVL 到 1.2

原始 InternViT-6B 有 48 层。团队发现倒数第四层特征最适合多模态任务，因此丢弃最后三层，得到 45 层视觉 encoder；再把输入从 224 提至 448，与 Nous-Hermes-2-Yi-34B 连接，联合训练视觉 encoder 和 MLP，加入 caption 与 OCR 数据，得到 InternViT-6B-448px-V1.2。

这一步说明预训练视觉模型最后层未必最适合作为 MLLM 输入：越靠后的特征越受原始对比目标约束，倒数层可能保留更多可迁移视觉细节。

### 5.2 从 1.2 到 1.5

1.5 在 V1.2 上继续训练：固定 448 变成 1–12 tile 动态分辨率，扩大数据规模、质量和视觉域，强化 OCR 与文档。语言侧从 Yi-34B 换为 InternLM2-20B，InternViT 仍保持强表现；作者据此主张视觉特征没有与单一 LLM 过度绑定。

### 5.3 大 LLM 需要大 VFM

论文用 LLaVA-NeXT（约 300M VFM + 34B LLM）与 InternVL1.2（6B VFM + 34B LLM）做非严格对照。排除五个 OCR 数据集、ConvBench 和 RealWorldQA 后，InternVL1.2 在剩余 11 项中的 9 项更高。该结果支持视觉容量重要，但作者明确承认数据和分辨率不等价，因此它是趋势证据，不是干净 scaling law。

---

## 6. 双语与 OCR 数据

### 6.1 预训练混合

| 类别 | 占比 | 代表数据 |
|---|---:|---|
| Captioning | 53.9% | LAION-EN/ZH、COYO、GRIT、COCO、TextCaps |
| Detection/Grounding | 5.2% | Objects365、GRIT、All-Seeing |
| OCR-large | 32.0% | Wukong-OCR、LAIONCOCO-OCR、Common Crawl PDF |
| OCR-small | 8.9% | MMC-Inst、LSVT、ST-VQA、RCTW、ReCTs、ArT、SynthDoG、ChartQA、DocVQA 等 |

OCR 相关数据总占比超过 40%，是 TextVQA、ChartQA、DocVQA 领先的重要先验，不能把结果全部归因于分辨率或视觉模型规模。

### 6.2 Fine-tuning 数据

任务覆盖 caption、general QA、science、chart、math、knowledge、OCR、document、grounding、conversation 与 text-only。文本数据用于维持 LLM 原语言能力；视觉任务通过统一对话格式监督。训练 context length 为 4096，在视觉 token 最多 3328 的训练设置下，留给问题与答案的余量已经有限。

### 6.3 翻译管线

使用开源 LLM 或 GPT-3.5 把英文样本转为中文，并通过 prompt 要求保留专有名词、技术术语、引语语气和自然表达。COYO、GRIT、ShareGPT4V、部分 instruction 数据被扩为中文。优点是低成本扩语言；风险是翻译文本不等于真实中文图文分布，OCR 框中文字也不能仅靠翻译答案获得。

---

## 7. 两阶段训练

### 7.1 Stage 1：视觉预训练/对齐

训练 InternViT-6B 与 MLP projector，优化视觉特征抽取和进入 LLM 的接口；语言模型保持冻结。动态 1–12 tiles、双语 caption、检测和大规模 OCR 数据在此阶段塑造视觉表征。

### 7.2 Stage 2：全模型微调

解冻并微调整个约 26B 模型，覆盖多模态与 text-only instruction。两个阶段都使用 context length 4096 和 LLaVA-1.5 response template。论文没有披露足够完整的 optimizer、batch、训练步数、总 GPU-hour 与各数据采样权重，因此权重可用不等于训练可完全重现。

---

## 8. 实验结果与消融

### 8.1 代表结果

![](assets/InternVL%201.5%20-%20动态高分辨率多模态套件/figures/01-commercial-comparison.png)

> 图 3：Figure 1。InternVL1.5 在 MathVista 53.5、TextVQA 80.6、ChartQA 83.8、DocVQA 90.9，高于图中若干闭源模型；MMMU 45.2 与 AI2D 80.7 则明显不是全面领先。

完整表中，InternVL1.5 在 DocVQA/ChartQA/InfoVQA/TextVQA/OCRBench 为 90.9/83.8/72.5/80.6/724；MME 2187.8、MMBench EN/CN 82.2/82.0、CCBench 69.8、MMVet 62.8、SEED 76.0、HallusionBench 49.3、MathVista 53.5。论文称 18 项中 8 项 SOTA，优势集中在 OCR、文档、图表和中英文本。

### 8.2 动态分辨率消融

![](assets/InternVL%201.5%20-%20动态高分辨率多模态套件/figures/04-resolution-ablation.png)

> 图 4：Figure 6。OCR 类任务通常随 tile 增加快速上升后饱和；AI2D、MMMU、MMBench、HallusionBench 在高 tile 数可能下降。虚线标出每项最佳预算。

代表最佳点：DocVQA 约 18 tiles、ChartQA 12、InfoVQA 24、TextVQA 24、OCRBench 24；MMMU 与 MMBench 约 6，AI2D 约 6，RealWorldQA 到 40 仍较稳。训练只见 1–12 tiles，测试可零样本扩到 40，说明模型具备一定长度外推；但不是所有任务都从更多像素获益。

高分辨率导致退化的可能原因包括：视觉 token 稀释问题文本、LLM 上下文压力、thumbnail/局部重复、resize 噪声，以及 benchmark 本身主要依赖全局语义。合理部署应做任务感知的 token budget，而不是固定 40 tiles。

### 8.3 多轮与通用推理边界

在 ConvBench 多轮对话上，InternVL1.5 仍显著落后 GPT-4V；MMMU 也从 InternVL1.2 的 51.6 降到 45.2，作者归因于 LLM 从 34B 换为 20B。它说明 6B VFM 和高分辨率能提高感知，但不能替代更强语言推理和多轮策略。

---

## 9. 从 1.5 到 InternVL2 / 2.5 / 3 的谱系

### 9.1 被 1.5 定型的主干

InternVL1.5 完成了相对原始 InternVL 的关键转向：从 `InternViT-6B → 8B QLLaMA → LLM` 的重型中间件，变成 `InternViT → pixel shuffle/unshuffle → 2-layer MLP → LLM`。**动态切图、thumbnail、每 tile 256 token、直接拼入 LLM**成为后续家族的稳定骨架。

### 9.2 InternVL2：扩模型族与输入类型

InternVL2 延续 1.5 的动态高分辨率和 ViT–MLP–LLM，扩展从小型到数十 B 的多档 LLM/ViT 组合，并把单图能力系统扩到多图、视频和更广任务。变化重点主要在模型族、数据与训练配方，而不是替换 1.5 的视觉接口。

### 9.3 InternVL2.5：加强 SFT 与推理后训练

2.5 继续保留同一视觉管线，以更大规模、更精细的数据混合和约 16.3M SFT 样本提高 OCR、数学、文档、视频与综合推理，并成为 InternVL3 的直接 baseline。该阶段的演进表明：1.5 解决“看清楚和接进去”，2/2.5 更集中解决“任务覆盖和答得更好”。

### 9.4 InternVL3：原生多模态预训练

InternVL3 仍沿用动态 tile、pixel unshuffle 与 MLP，但把视觉从后装 adapter 提升为预训练期的一等数据源：约 200B token 中纯文本与多模态共同训练，ViT、MLP、LLM 全参数更新；再加入 V2PE、MPO 和 VisualPRM Best-of-N。它不是推翻 1.5，而是在 1.5 的输入/架构主干上改造训练时机、位置预算和后训练。

| 版本 | 稳定架构 | 主要新增 |
|---|---|---|
| InternVL 1.5 | 6B ViT → pixel shuffle → MLP → 20B LLM | dynamic tiles、thumbnail、双语 OCR |
| InternVL2 | 同主干，多尺寸模型 | 多图/视频、任务与数据扩展 |
| InternVL2.5 | 同主干 | 更强 SFT/推理配方、系列能力整合 |
| InternVL3 | 同主干 | native multimodal pre-training、V2PE、MPO、VisualPRM |

可对照 [[InternVL - 规模化视觉基础模型与通用对齐]] 与 [[InternVL3 - 原生多模态预训练与测试时扩展]]。谱系中真正长期保留的不是原始 QLLaMA，而是大视觉底座、动态高分辨率和直接 token 接口。

---

## 10. 局限与批判性判断

1. **切图没有显式全图坐标。** 每个 tile 重用相同 ViT 网格位置，LLM 主要靠 tile 顺序学习全局布局；复杂表格、跨栏文档和目标跨 tile 时可能混淆。
2. **context 预算紧张。** 训练最大 3328 视觉 token、总上下文 4096；多轮长答案可用空间有限。测试 40 tiles 的 10,496 token 又超出训练视觉长度，稳定性只由 benchmark 曲线间接证明。
3. **分辨率不是单调收益。** 多个通用/推理 benchmark 在高 tile 数下降；“4K 支持”应解释为可输入，不是所有任务的推荐配置。
4. **OCR 结果由三项因素耦合。** 6B ViT、动态分辨率与 40%+ OCR 数据同时变化，论文没有完整正交消融，无法精确归因。
5. **大 VFM 对照不完全公平。** LLaVA-NeXT 与 InternVL1.2 的训练数据和分辨率不同，作者已明确承认；不能据此直接推导视觉参数 scaling law。
6. **训练可复现性不足。** 模型和代码开放，但总样本数、采样权重、优化器超参、GPU-hour、翻译与过滤 pipeline 不完整。
7. **闭源模型比较有版本效应。** GPT-4V、Claude、Gemini 的 API 版本和 prompt 不完全一致，图中领先只能按对应时间与测试设置理解。
8. **多轮与幻觉仍弱。** ConvBench、HallusionBench 表明高分辨率感知并未自动解决长对话一致性和视觉忠实性。

我的判断是：InternVL1.5 最重要的系统贡献是把“高分辨率”拆成三个可工程化步骤：几何网格选择、局部+全局视图、token 压缩。它证明文档/OCR 需要更高视觉预算，也同时证明预算必须按任务分配。后续版本沿用这一骨架，说明它比 1.5 当时的具体 20B LLM 更具长期价值。

---

## 11. 复现路线

### 11.1 最小模型

1. 使用 patch14 的 448px ViT；输入一块得到 $32\times32$ token。
2. 按官方 reshape/permute 做 $2\times2$ pixel unshuffle，得到 $16\times16$、通道 $4C$。
3. 两层 MLP 投影到 LLM hidden size；把视觉 token 放在用户问题之前。
4. 先冻结 LLM 训练 ViT+MLP，再全参数 SFT；文本 loss 只监督 assistant response。

### 11.2 动态切图伪流程

```text
枚举 tile 数不超过 Nmax 的整数网格 (a,b)
按 |W/H-a/b| 选择网格；同分时避免目标面积超过原图两倍
resize 到 (448a, 448b)
按固定顺序切成 ab 个 tile
若 ab>1，追加 448×448 thumbnail
逐 tile 编码并按同一顺序拼接 256-token 表示
```

### 11.3 必做验证

- 单 tile、6 tiles、12 tiles、40 tiles 的 token 数应为 256、1792、3328、10496（后 3 项含 thumbnail）。
- thumbnail 不得替代任何局部 tile；tile 顺序在训练与推理一致。
- 视觉 token、文本 token、labels 与 position ids 长度完全一致；padding 不计 loss。
- 分别报告 OCR 与非 OCR 的 tile sweep，不使用一个全局最佳分辨率掩盖任务差异。
- 对跨 tile 表格增加位置扰动测试：打乱 tile 顺序、去 thumbnail、只保留 thumbnail，以判断布局依赖。

全量复现的主要障碍不是网络代码，而是 6B ViT 连续训练、双语数据清洗与 26B 全参数微调成本。

---

## 12. 图表索引

1. `figures/01-commercial-comparison.png`：与闭源模型的代表 benchmark 对比。
2. `figures/02-architecture.png`：Dynamic HR → InternViT → pixel shuffle → MLP → InternLM2。
3. `figures/03-dynamic-resolution.png`：宽高比匹配、局部 tile 与 thumbnail。
4. `figures/04-resolution-ablation.png`：不同任务随 tile 数变化的曲线。

## 13. 材料来源

- [官方论文 PDF](assets/InternVL%201.5%20-%20动态高分辨率多模态套件/sources/paper.pdf)
- [来源与原图清单](assets/InternVL%201.5%20-%20动态高分辨率多模态套件/sources.md)
