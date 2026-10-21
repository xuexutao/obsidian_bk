---
type: paper-note
status: done
domain: VLM
series: BLIP
series_order: 3
series_hub: "[[BLIP 系列：从数据自举到统一理解生成 - Survey]]"
paper: "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning"
year: 2023
arxiv: 2305.06500
doi: null
source: https://arxiv.org/abs/2305.06500
project: https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
code: https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
tags:
  - VLM
  - instruction-tuning
  - Q-Former
  - BLIP-2
created: 2026-10-16
updated: 2026-10-16
---

# InstructBLIP：指令感知视觉特征抽取

> 系列导航：[[BLIP - 生成式字幕过滤的统一视觉语言预训练]] → [[BLIP-2 - 冻结底座的高效视觉语言桥接]] → **InstructBLIP** → [[BLIP-3 - 开放大规模多模态模型家族]] → [[BLIP3-o - 全开放统一多模态理解与生成]] · [[BLIP 系列：从数据自举到统一理解生成 - Survey|系列综述]]

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning |
| 作者 | Wenliang Dai、Junnan Li、Dongxu Li、Anthony M. H. Tiong 等 |
| 机构 | Salesforce Research、HKUST、NTU Singapore |
| arXiv | 2305.06500v2，2023-06-15 |
| 官方实现 | https://github.com/salesforce/LAVIS/tree/main/projects/instructblip |
| 一句话结论 | InstructBLIP 不只把 instruction 交给 LLM，还把同一 instruction 输入 Q-Former，让 32 个 query 按任务选择视觉证据；再以 13 个 held-in 数据集做平衡指令微调，从而显著提升 13 个 held-out 数据集的零样本泛化。 |

---

## 1. 研究问题与核心贡献

### 1.1 BLIP-2 的静态视觉提示为何不够

BLIP-2 先把每幅图压成固定的 32 个视觉 prompt，再把文字指令交给冻结 LLM。这样同一图像在“数一数”“读招牌”“解释原因”“简短描述”四种请求下得到完全相同的视觉表示，任务差异只能由 LLM 在压缩后的信息里补救。如果 Q-Former 已经丢掉小文字或空间关系，后端 LLM 无法恢复。

InstructBLIP 的关键改动是把 instruction 同时送入 Q-Former 和 LLM。Q-Former 的 query 在抽取图像时先通过 self-attention 读取 instruction，从而让相同图像产生任务相关视觉 token。这是“instruction-aware visual feature extraction”，不是简单把更多文本拼到最终 prompt。

### 1.2 三项核心贡献

1. 在 BLIP-2 的 Q-Former 中加入 instruction token，使视觉 connector 本身条件化于任务。
2. 汇集 26 个公开数据集、11 类任务，严格拆成 13 held-in 与 13 held-out，专门衡量未见数据集和未见任务的零样本泛化。
3. 用平方根数据采样平衡规模差异，并证明自然语言 instruction 比普通 multitask 格式更能迁移到 held-out 任务。

---

## 2. 架构、QKV、mask、position 与 token shape

### 2.1 三段式结构

视觉端为冻结 EVA-CLIP ViT-g/14；connector 是从 BLIP-2 初始化并继续训练的 Q-Former；语言端比较冻结 FlanT5-XL/XXL 与 Vicuna-7B/13B。指令微调时只更新 Q-Former 及其到 LLM 的投影，image encoder 与 LLM 均冻结。

![](assets/InstructBLIP%20-%20指令感知视觉特征抽取/figures/01-architecture.png)

> **Figure 3，PDF 第 5 页。** instruction token 一路进入 Q-Former，影响 query 对图像的读取；另一路与投影后的视觉 token 一起进入冻结 LLM。右侧放大图显示 query 与 instruction 共享 self-attention，只有 query 分支通过 cross-attention 读取 image embeddings。来源：https://arxiv.org/abs/2305.06500

### 2.2 QKV 信息流

Q-Former 继承 BLIP-2 的 32 个、768 维 learnable query。对 batch $B$，冻结 ViT 输出可写为 $X_v\in\mathbb{R}^{B\times N_v\times d_v}$，query 为 $Q_0\in\mathbb{R}^{B\times32\times768}$，instruction token 为 $T\in\mathbb{R}^{B\times L_i\times768}$。Q-Former 输出只取 query 对应位置：

$$Z=\operatorname{QFormer}(Q_0,T;X_v)_{1:32}\in\mathbb{R}^{B\times32\times768}$$

在 self-attention 中，query token 与 instruction token 共同形成 $Q,K,V$，所以 query 可读取任务语义；在视觉 cross-attention 中，query hidden state 是 $Q$，冻结图像 token 是 $K,V$。instruction token 不直接作为视觉 cross-attention query，作用是通过共享 self-attention 改写 32 个视觉 query 的状态。

线性 connector 把 $Z$ 投影为 $P_v\in\mathbb{R}^{B\times32\times d_{LLM}}$。因此无论图像 patch 数多少，LLM 固定只接收 32 个视觉 soft-prompt token；这控制了 LLM 上下文成本，也形成高分辨率/OCR/密集定位的信息瓶颈。

### 2.3 attention mask 与位置编码

Q-Former 对 query+instruction 使用可双向交互的 self-attention，而视觉特征通过独立 cross-attention 被读取；与 BLIP-2 第一阶段的 ITC/ITG/ITM 多 mask 不同，InstructBLIP 指令微调阶段的重点是让 instruction 可见于 query。LLM 侧取决于底座：FlanT5 encoder 双向编码视觉 prompt+instruction，decoder 对答案使用 causal mask；Vicuna 把视觉 prompt、instruction 与答案前缀串成 decoder-only 序列，用标准左到右 causal mask。

视觉位置由 ViT-g/14 的 patch position 表示提供；instruction token 继承 Q-Former/BERT 的文本位置表示；投影后的视觉 token 在 LLM 中占据前缀位置，后接 instruction。论文没有提出新的多模态 position encoding，也没有对此做消融。

### 2.4 与 BLIP-2 两阶段初始化的关系

InstructBLIP 不是从零训练 connector。Q-Former 先按 BLIP-2 完成两阶段预训练：第一阶段在冻结图像 encoder 上用 ITC/ITM/ITG 学语言相关视觉表示；第二阶段接冻结 LLM，把 query 输出适配为 soft visual prompt。InstructBLIP 再做第三阶段指令微调，引入 instruction-aware 查询。Vicuna 不是原始 BLIP-2 配置，因此作者先按 BLIP-2 流程预训练 Vicuna 版本，再进行指令微调。

---

## 3. 训练数据、目标与训练策略

### 3.1 26 数据集与 held-in/held-out 设计

数据覆盖 caption、读图字幕、视觉推理、普通/知识型/OCR VQA、问题生成、视频 QA、Visual Dialog、图像分类和 LLaVA-Instruct-150K。每类任务人工设计 10–15 个自然语言模板；对于天然短答案任务，一些模板显式加入 `short`、`briefly`，避免模型把所有回答都学成固定短格式。LLaVA-Instruct-150K 已是指令格式，不再套模板。

![](assets/InstructBLIP%20-%20指令感知视觉特征抽取/figures/02-held-in-held-out-tasks.png)

> **Figure 2，PDF 第 4 页。** 黄色为 13 个 held-in 数据集，白色为 13 个 held-out 数据集。held-out 既包含“见过任务但没见过数据域”，也包含视觉推理、视频 QA、对话 QA、分类等完全未见任务。来源：https://arxiv.org/abs/2305.06500

模型仅混合 held-in 训练集，评估时检查对应 held-in 测试集和全部 held-out 数据集。涉及场景文字的数据会把 OCR token 作为 instruction 的补充输入，因此 OCR 结果不能完全归因于视觉 encoder 自己读字。

### 3.2 语言建模目标与数据平衡

训练目标是给定图像和 instruction 自回归生成 response：

$$\mathcal{L}_{\mathrm{IT}}=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},P_v(I,q),q)$$

不同数据集大小差异很大，直接按样本数会淹没小任务，按数据集均匀采样又会过拟合小集。作者采用平方根平滑：

$$p_d=\frac{\sqrt{S_d}}{\sum_{i=1}^{D}\sqrt{S_i}}$$

其中 $S_d$ 是数据集规模；此外手工降低 A-OKVQA 多选权重、提高开放生成 OKVQA 权重。消融显示去掉平衡后不同任务在不同 step 达峰，统一 checkpoint 难兼顾。

### 3.3 优化与成本

四种模型共用 ViT-g/14，LLM 分别为 FlanT5-XL 3B、FlanT5-XXL 11B、Vicuna-7B、Vicuna-13B。最多训练 60K step，每 3K step 验证并为所有数据集选择同一 checkpoint。3B/7B/11–13B 版本 batch 分别为 192/128/64；AdamW 的 $\beta_1=0.9$、$\beta_2=0.999$、weight decay 0.05；前 1K step 从 $10^{-8}$ warmup 到 $10^{-5}$，之后 cosine decay 到 0。训练使用 16×A100 40GB，1.5 天内完成。

### 3.4 生成方式

开放字幕和 VQA 直接生成文本。多选/分类任务不让模型自由输出，而是对候选答案计算条件 log-likelihood，选最高者；二分类把正负类扩展成 `yes/true` 与 `no/false` 等 verbalizer。视频 QA 均匀采 4 帧，每帧独立经过 ViT+Q-Former，四组视觉 token 拼接后送入 LLM；模型训练未见时间序列，因此这只是帧级晚融合。

---

## 4. 定量结果、baseline 与消融

### 4.1 held-out 零样本结果

| 模型 | NoCaps CIDEr | GQA | TextVQA | ScienceQA-IMG | MSRVTT-QA | iVQA |
|---|---:|---:|---:|---:|---:|---:|
| BLIP-2 FlanT5-XL | 104.5 | 44.0 | 43.1 | 54.9 | 16.2 | 40.4 |
| InstructBLIP FlanT5-XL | 119.9 | 48.4 | 46.6 | 70.4 | 25.0 | 53.1 |
| BLIP-2 Vicuna-7B | 107.5 | 38.6 | 40.1 | 53.8 | 9.2 | 27.5 |
| InstructBLIP Vicuna-7B | **123.1** | 49.2 | **50.1** | 60.5 | 22.1 | 52.2 |

FlanT5-XL 版相对其 BLIP-2 起点的 held-out 平均相对提升为 15.0%；4B 级最小 InstructBLIP 在与 Flamingo-80B 共有的六项零样本评测上平均相对高 24.8%。不同 LLM 的优势不同：FlanT5 更擅长多选/分类，Vicuna 更擅长开放生成，说明冻结 LLM 的原始指令分布直接决定下游偏好。

### 4.2 关键结构消融

| FlanT5-XL 设置 | Held-in Avg. | GQA | ScienceQA | VizWiz | iVQA |
|---|---:|---:|---:|---:|---:|
| 完整 InstructBLIP | **94.1** | **48.4** | **70.4** | **32.7** | **53.1** |
| 去掉 instruction-aware visual features | 89.8 | 45.9 | 63.4 | 25.1 | 47.5 |
| 去掉 data balancing | 92.6 | 46.8 | 66.0 | 31.8 | 51.1 |

对 Vicuna-7B，去掉 instruction-aware 特征使 iVQA 从 52.2 跌到 36.8，降 15.4 点。空间/时间推理下降最大，支持“instruction 在视觉抽取阶段决定关注区域”的解释，而不是泛泛的模板增益。

### 4.3 Instruction tuning 不等于普通 multitask

![](assets/InstructBLIP%20-%20指令感知视觉特征抽取/figures/03-instruction-vs-multitask.png)

> **Figure 4，PDF 第 8 页。** 普通 multitask 的 held-in 平均可接近 InstructBLIP（最高 93.7 vs 93.8），但 held-out 只有 46.8，接近 BLIP-2 的 46.1；自然语言 instruction 使 held-out 升至 52.9。收益主要体现在未见任务泛化，而不是训练集拟合。来源：https://arxiv.org/abs/2305.06500

下游微调时，保持 $224^2$ 分辨率并冻结视觉 encoder，只训练 188M Q-Former，即可把 ScienceQA-IMG 提到 90.7、OCR-VQA 提到 73.3。这说明指令微调 checkpoint 也是比 BLIP-2 更好的任务初始化；但与 PaLM-E 562B 的 OKVQA 66.1 相比，InstructBLIP Vicuna-7B 为 62.1，开放知识仍受冻结 LLM 上限约束。

---

## 5. 与 CLIP、Flamingo、BLIP-2、LLaVA 的关系

- **CLIP**：ViT-g/14 来自强对比视觉预训练，但 InstructBLIP 不使用独立图文对比检索作为最终接口。CLIP 提供 patch 表征，Q-Former 再按 instruction 选择信息。
- **Flamingo**：两者都冻结视觉与语言底座。Flamingo 在 LM 深层反复插门控 cross-attention，并靠 interleaved data 获得多图 few-shot；InstructBLIP 只在输入侧产生 32 个 soft prompt，训练参数少、公开可复现，但深层视觉访问与原生多图上下文较弱。
- **BLIP-2**：这是直接前身。InstructBLIP 保留两阶段 Q-Former 初始化、32-query 固定瓶颈和冻结 LLM，只增加 instruction 对 Q-Former 的条件化，再做多任务指令微调。它回答的是“该抽取什么视觉特征”，不是重新设计底座。
- **LLaVA**：InstructBLIP 使用了 LLaVA-Instruct-150K 的 23K 详细描述、77K 推理、58K 对话样本，但主体数据还包括大量公开 benchmark。LLaVA 用线性 projector 并微调 Vicuna，结构更简单、对话生态更强；InstructBLIP 冻结 LLM、训练 188M Q-Former，更参数高效，也用 held-out 协议系统验证任务泛化。

---

## 6. 局限、future work 与批判性评价

### 6.1 明确局限

论文明确指出模型继承冻结 LLM 的幻觉与偏见，不能未经场景化安全/公平评估直接部署。Q-Former 虽更 grounding，但没有可靠拒答、置信度或事实校准机制。输入固定为 $224^2$ 且只有 32 个视觉 token，密集 OCR、小对象、精确位置与多页文档容易受瓶颈限制。

“13 held-out”并不等于与互联网预训练完全无关：冻结 ViT/LLM 和 BLIP-2 预训练已接触海量数据，作者主要控制的是 instruction-tuning 数据集交叉污染。视频结果仅拼接 4 帧，没有时序位置建模或视频训练，不能据此宣称获得完整视频理解。

### 6.2 Future work

结论只笼统提出推动 general-purpose multimodal AI，没有列出具体技术路线。根据论文明确暴露的问题，后续可检验但不能冒充作者承诺的方向包括：提高视觉分辨率与动态 token 数；加入多图/视频时序训练；对 Q-Former query 数、cross-attention 深度和位置编码做系统消融；引入事实校准、拒答和安全评测；减少对外部 OCR token 的依赖。

### 6.3 个人判断

InstructBLIP 最有价值的结论是“指令应进入 connector，而不只进入 LLM”。同一个视觉 encoder 可以保持冻结，但视觉压缩必须由任务语义控制，否则固定 32-token 瓶颈会提前丢失证据。它也以严格 held-in/held-out 设计证明，自然语言 instruction 的泛化收益不能被普通 multitask fitting 替代。不过，模型仍是浅层软提示路线，LLM 在后续层无法反复回看高分辨率视觉网格；这正是 CogVLM、深层 cross-attention 和更长视觉 token 路线继续发展的原因。

---

## 7. 复现成本与实践建议

官方 LAVIS 提供配置、代码和模型，复现实验远比 CoCa、PaLI、Flamingo 现实。单次论文级指令微调仍需 16×A100 40GB、约 1.5 天，并需准备 13 个 held-in 数据集、OCR 输入、模板与平方根采样；仅下载 checkpoint 做推理则可在更少 GPU 上运行，具体显存取决于 3B/7B/11B/13B LLM 和精度。

最小可信复现应固定 BLIP-2 checkpoint，做三组对照：完整 instruction-aware Q-Former；instruction 只给 LLM；去掉平方根采样。held-out 指标必须与 held-in 分开报告。若只比较聊天样例，很容易把冻结 LLM 的语言风格误当成视觉 grounding 改善。

---

## 8. 结论

InstructBLIP 将 BLIP-2 的静态 Q-Former 变成受 instruction 条件控制的视觉检索器：query 先与指令 self-attend，再以 cross-attention 从冻结 ViT 中读取与任务相关的区域，最后压成 32 个 soft prompt 交给冻结 LLM。实验显示，这一结构和数据平衡都显著改善 held-out 泛化，自然语言 instruction 又明显优于普通 multitask 标签。其优势是参数高效、模块化、官方开源；边界是固定视觉瓶颈、低分辨率、冻结 LLM 继承问题与缺乏真正时序建模。

---

## 9. 材料来源

完整来源、Figure/Page 映射与获取日期见 [sources.md](assets/InstructBLIP%20-%20指令感知视觉特征抽取/sources.md)。
