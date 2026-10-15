---
type: paper-note
status: done
domain: VLM
paper: "Flamingo: a Visual Language Model for Few-Shot Learning"
year: 2022
arxiv: 2204.14198
doi: null
source: https://arxiv.org/abs/2204.14198
project: null
code: null
tags:
  - VLM
  - few-shot-learning
  - in-context-learning
  - multimodal-generation
created: 2026-10-15
updated: 2026-10-15
---

# Flamingo：少样本视觉语言上下文学习

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | Flamingo: a Visual Language Model for Few-Shot Learning |
| 作者 | Jean-Baptiste Alayrac 等，DeepMind |
| 发表 | NeurIPS 2022 |
| arXiv | 2204.14198 |
| 官方论文 | https://arxiv.org/abs/2204.14198 |
| 官方代码 | 论文未发布官方训练代码或模型权重 |
| 一句话结论 | Flamingo 首次系统证明：冻结强视觉编码器和强语言模型，只训练视觉重采样器与穿插式门控跨注意力层，就能把图像、视频和文本组织成多模态上下文，并以少量示例完成开放式视觉语言任务。 |

---

## 1. 研究背景

### 1.1 CLIP 之后缺少的能力

CLIP/ALIGN 让视觉模型获得开放词汇零样本分类能力，但双塔输出主要是图文相似度：它适合在有限候选中打分，却不能自然生成字幕、自由回答问题或进行视觉对话。另一边，大语言模型已经展示了 in-context learning——不更新权重，只在提示中给出几个输入输出示例，就能识别任务格式并生成答案。Flamingo 的问题是：能否让同一机制跨越图像、视频和文本，使视觉语言任务也可被统一为“多模态上下文后的下一词预测”？

难点不只是把视觉特征接到语言模型。视觉输入长度取决于图像分辨率和视频帧数；少样本提示又会含多幅交错图像；直接微调超大语言模型容易灾难性遗忘；端到端训练 70B 级模型的成本不可接受。Flamingo 因此采取“冻结两端、训练桥梁”的系统设计。

### 1.2 核心贡献

论文提出三件彼此配合的组件：Perceiver Resampler 把任意大小的图像/视频特征压成固定 64 个视觉 token；gated cross-attention dense 层穿插进冻结 Chinchilla LM，用零初始化门控稳定注入视觉信息；per-image attention mask 让每段文本直接看最近一幅图，同时通过语言自注意力间接保留更早视觉上下文。模型在网页交错图文、图文对和视频文本对的混合数据上训练，因而推理时可以把若干 `(视觉输入, 文本答案)` 示例与新 query 串成一个 prompt。

![](assets/Flamingo%20-%20少样本视觉语言上下文学习/figures/01-results-overview.png)

> **Figure 2，PDF 第 3 页。** 左图把 Flamingo-80B 的 32-shot 结果除以各任务当时 fine-tuned SOTA：16 项中多项已达到或超过 100%；右图显示参数规模与 shot 数同时增大时聚合性能上升，且 80B 比小模型更能利用更多示例。它证明少样本收益来自模型、桥接结构与训练数据共同形成的上下文学习能力，而非单一任务微调。来源：https://arxiv.org/abs/2204.14198

---

## 2. 任务定义与方法

### 2.1 输入、输出与概率模型

输入是任意交错的图像/视频与文本序列，例如若干支持样例 `<image> Question ... Answer ...`，再接一幅 query 图像与问题。输出是自由文本；闭集任务也可以比较候选答案的条件似然。模型按视觉输入在文本中的相对位置做因果条件生成：

$$p(y\mid x)=\prod_{\ell=1}^{L}p(y_\ell\mid y_{<\ell},x_{\leq\ell})$$

其中 $x_{\leq\ell}$ 表示当前文本 token 之前出现的视觉输入。该接口覆盖字幕、VQA、视觉对话、视频问答和分类，但不直接覆盖检测框、分割图、光流等结构化或连续密集输出。

### 2.2 总体架构

视觉端是对比预训练并冻结的 NFNet-F6；视频按帧独立编码并加入时间位置嵌入。语言端是冻结的 Chinchilla decoder-only LM。两端之间只有新训练的 Perceiver Resampler 与 gated xattn-dense 层参与 Flamingo 训练。3B、9B、80B 三档模型分别建立在 1.4B、7.1B、70B 冻结 LM 上，总参数量约 3.2B、9.3B、80B；最大模型新增约 10B 跨模态参数与 194M Resampler 参数。

![](assets/Flamingo%20-%20少样本视觉语言上下文学习/figures/02-architecture.png)

> **Figure 3，PDF 第 4 页。** 图像分别经过冻结视觉编码器与可训练 Resampler；文本进入冻结 LM，在若干 LM block 前插入可训练的门控跨注意力与前馈层。蓝色雪花部分保留预训练知识，紫色部分学习视觉条件。该图解释了 Flamingo 的核心工程判断：不是把视觉 token 当普通词一次性投影，而是在语言网络深度方向反复提供受控视觉通道。来源：https://arxiv.org/abs/2204.14198

### 2.3 Perceiver Resampler

视觉编码器输出的空间或时空网格长度可变。Resampler 维护 $R=64$ 个可学习 latent query，每层让 latent 作为 query，对视觉特征与 latent 拼接形成的 key/value 做注意力，再经 FFN 更新。它把高分辨率图像、多帧视频统一压缩为固定 token 数，避免语言模型跨注意力成本随像素/帧数失控。

$$Z^{(k+1)}=Z^{(k)}+\operatorname{Attn}\left(Q=Z^{(k)},K=V=[X,Z^{(k)}]\right)$$

![](assets/Flamingo%20-%20少样本视觉语言上下文学习/figures/03-perceiver-resampler.png)

> **Figure 5，PDF 第 23 页。** 左侧展示视频帧特征加时间嵌入后展平，固定 latent queries 迭代读取可变长视觉网格；右侧给出官方伪代码。把 latent 同时放入 K/V 允许其在读取视觉特征时保留和整合自身状态。来源：https://arxiv.org/abs/2204.14198

### 2.4 门控跨注意力与多图掩码

每个新增层先让语言 hidden state 查询视觉 token，再过 dense FFN。新增残差分支乘以 $\tanh(\alpha)$，且 $\alpha=0$ 初始化：

$$h'=h+\tanh(\alpha)\,\operatorname{XAttn}(h,z)$$

因此训练第 0 步模型严格退化为原语言模型，避免随机视觉层突然破坏语言分布。多图提示中，每个文本 token 只直接跨注意最近的前置图像；更早图像的信息已写入此前文本 token，再由 LM 自注意力传播。训练最多每序列 5 张图，测试却能从多达 32 个视觉示例获益，这种掩码设计避免模型依赖固定图像索引。

---

## 3. 训练数据与训练流程

### 3.1 数据混合

训练使用四个官方描述的数据源：M3W 从约 4,300 万网页中按 DOM 位置提取交错图文，每段随机采样 256 token、最多 5 张图；ALIGN 含 18 亿 alt-text 图文对；LTIP 含 3.12 亿更长、更高质量的图文对；VTP 含 2,700 万平均约 22 秒的视频—文本对。配对数据统一加 `<image>` 与 `<EOC>` token，使格式接近网页序列。

对数据集 $\mathcal{D}_m$ 使用加权自回归负对数似然：

$$\mathcal{L}=\sum_{m=1}^{M}\lambda_m\,\mathbb{E}_{(x,y)\sim\mathcal{D}_m}\left[-\sum_{\ell=1}^{L}\log p(y_\ell\mid y_{<\ell},x_{\leq\ell})\right]$$

M3W、ALIGN、LTIP、VTP 的权重分别为 1.0、0.2、0.2、0.03。每次更新分别计算各数据源梯度后加权累积，而不是 round-robin 轮流更新；消融证明混合权重与累积策略本身是重要设计。

### 3.2 优化、视频处理与资源

图像按比例缩放并 padding 到 $320\times320$，使用水平翻转与颜色增强；视频训练采 8 帧、1 FPS，推理可用 30 帧、3 FPS，并插值 Resampler 的时间位置嵌入。AdamW 初始 5,000 step 将学习率线性升至 $10^{-4}$，之后保持不变，通常训练 500k step；Resampler 不做 weight decay，其余新参数 weight decay 为 0.1。

最大 80B 模型在 1,536 个 TPUv4 chip 上训练 15 天，16 路模型并行，并用 ZeRO stage 1 分片优化器状态。视觉编码器本身还先在 ALIGN+LTIP 上以 CLIP 式对比损失训练 120 万更新步、使用 512 个 TPUv4。论文未公开 M3W、LTIP、VTP、Chinchilla 权重或完整训练代码，因此完整复现不可行；即使有数据，算力门槛也远高于普通研究组。

---

## 4. 实验与消融

### 4.1 评测设计与主要结果

作者评测 16 个图像/视频语言 benchmark。COCO、OKVQA、VQAv2、MSVDQA、VATEX 是开发集；另外 11 项不参与结构或超参数选择，用于减小 few-shot 评测共同适配。任务覆盖字幕、VQA、视觉对话、视频理解、OCR、仇恨模因和多选推理。开放任务用 beam search，闭集任务比较候选似然。

下表摘录官方主表中的代表结果；不同任务指标不同，数值不可横向平均：

| 模型/设置 | OKVQA top-1 | VQAv2 top-1 | COCO CIDEr | MSVDQA top-1 | VATEX CIDEr | TextVQA top-1 |
|---|---:|---:|---:|---:|---:|---:|
| 既有 zero/few-shot SOTA | 43.3 | 38.2 | 32.2 | 35.2 | 论文未列 | 论文未列 |
| Flamingo-3B，32-shot | 45.9 | 57.1 | 99.0 | 42.6 | 59.2 | 30.6 |
| Flamingo-80B，4-shot | 57.4 | 63.1 | 103.2 | 41.7 | 56.0 | 36.5 |
| Flamingo-80B，32-shot | **57.8** | **67.6** | **113.8** | **52.3** | **65.1** | **37.9** |
| 当时 fine-tuned SOTA | 54.4 | 80.2 | 143.3 | 47.9 | 76.3 | 54.7 |

主文称 80B 模型在 16 项中的 6 项超过 fine-tuned SOTA，而 Table 1 图注写成 7 项；官方版本内部存在这一处表述不一致，本文不擅自消解。稳定结论是：32 个示例已在多个任务超过使用成千上万标注的专门模型，但在 VQAv2、COCO、VATEX、TextVQA 等任务仍有明显差距。若允许任务微调，Flamingo-80B 在 VQAv2 达 82.0，超过其 32-shot 的 67.6 与此前 81.3；论文还报告在五项任务刷新 SOTA。

### 4.2 关键消融

基线 Flamingo-3B 的五项开发任务归一化 Overall 为 70.7。去掉 M3W 降到 53.4，去掉图文对降到 60.9，说明网页交错序列是上下文学习核心，但高质量单图配对仍不可替代。round-robin 数据更新降到 62.9；去掉零初始化 tanh gate 降到 66.5 且出现训练不稳定；vanilla cross-attention 为 66.9、grafting 为 63.1，均低于 gated xattn-dense。

Perceiver Resampler 的 Overall 70.7，高于 MLP 的 66.6 与普通 Transformer 的 66.7，且后两者 step 更慢。每层都插跨注意力最好，但改为每 4 层插一次可加速 66%、仅损失 1.9%；这是 9B/80B 模型稀疏插层的依据。视觉端换成 CLIP ViT-L/14 后 Overall 为 64.9，换小 NFNet-F0 为 62.7，表明强视觉底座仍决定上限。最关键的是冻结策略：随机初始化 LM 仅 57.8，微调预训练 LM 为 62.7，均低于冻结 LM 的 70.7，构成灾难性遗忘的直接证据。

扩展消融还显示，让文本只直接关注最近一幅图比关注所有历史图高 7.2%；把 MassiveText LM 换成 C4 预训练 LM 低 7.9%；解冻并训练已预训练视觉编码器低 2.6%，从头训练视觉端低 9.3%。Flamingo 的优势不是单一模块，而是“强冻结底座 + 稳定桥接 + 交错网页数据 + 正确混合策略”的组合。

---

## 5. 局限与失败案例

### 5.1 幻觉、无依据猜测与上下文限制

![](assets/Flamingo%20-%20少样本视觉语言上下文学习/figures/04-failure-cases.png)

> **Appendix Figure（failure cases），PDF 第 42 页。** 左例受语言先验驱动，把手机屏幕说成朋友短信；中例把窗外误答为停车场；右例在图片无法确定“给谁发短信”时仍猜测“司机”。三个例子分别体现视觉 grounding 不足、问题诱导和缺乏拒答/不确定性表达。来源：https://arxiv.org/abs/2204.14198

冻结 LM 会完整继承其弱点：文本先验既帮助少样本任务定位，也会压过视觉证据导致幻觉。LM 训练长度为 2,048 token，而多样例视觉对话可轻易达到 4,096–8,192 token；VisDial 因此最多报告 16-shot。模型对示例顺序和格式敏感，超过 32-shot 后收益快速平台化；不缓存 KV 时，in-context inference 随 shot 数呈二次成本增长，缓存后仍线性增长。

### 5.2 任务边界与 future work

生成式目标在分类上落后直接优化图文检索的对比模型，标签先验、示例次序和 prompt 格式会影响闭集分数。纯文本接口也不适合框、掩码、稠密时空预测和连续输出。作者提出的后续方向包括：融合对比与生成目标、改进 few-shot calibration 与示例选择、结合 in-context 与梯度式适配、研究视觉语言 scaling law、扩展结构化输出与音频模态、降低语言建模的数据和计算低效。

数据与治理方面，网页数据可能包含有害内容和偏差；模型输出开放、能力难穷举，论文强调部署前需要场景化评估。由于官方没有发布最大模型权重、核心数据和代码，这些风险分析与结果也难由外部研究者完整审计。

---

## 6. 与后续工作的关系

Flamingo 把 CLIP 的“视觉—语言对齐”推进为“多模态序列上的生成式上下文学习”，并确立了冻结底座、训练桥接层的路线。BLIP-2 延续冻结两端，但用只有 32 个 query 的 Q-Former 先做视觉语言表示学习，再接冻结 OPT/FlanT5，参数与训练成本明显更低；其重点是高效对齐，而非 Flamingo 的任意交错多图上下文。

LLaVA 选择更简单的 CLIP patch token 线性投影 + Vicuna，通过 GPT-4 生成的视觉指令数据获得对话能力，牺牲 Flamingo 的深层跨注意力和视频/多图原生设计，换取可复现性与开放生态。CogVLM 则进一步质疑“浅桥接足够”：它在每层为视觉 token 增加独立 QKV/FFN 专家，同时冻结语言参数，既保持语言能力又增加深层视觉容量。后续 OpenFlamingo、IDEFICS 等开放工作直接复刻了 Flamingo 的架构思想，但原论文最强数据和 80B 结果仍不可复现。

---

## 7. 个人判断

Flamingo 的历史价值在于把 few-shot 从“分类器上少量标注”改写成“提示中的多模态演示”。其真正创新不是简单拼接视觉 token，而是围绕冻结大模型设计了一整套稳定信息通道：固定 token 瓶颈控制视觉成本，零初始化 gate 保护语言能力，最近图掩码支持训练短、推理长的图像序列，网页交错数据让模型见过类似上下文结构。

局限同样清楚。80B 成绩依赖 1,536 TPUv4、未公开数据和 Chinchilla，复现成本与可审计性很差；视觉问答中的幻觉说明“能按上下文格式生成”不等于“视觉证据充分 grounding”。如果今天复现其机制，合理目标应是用开放 LM/视觉编码器验证 gate、Resampler、attention mask 与 interleaved data 的相对贡献，而不是追逐论文绝对分数。综合判断：Flamingo 建立了生成式 VLM 的多图上下文范式，但也把数据、算力、长上下文和视觉忠实性四个瓶颈完整暴露给了后续工作。