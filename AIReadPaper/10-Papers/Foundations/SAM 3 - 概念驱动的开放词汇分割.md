---
type: paper-note
status: complete
domain: Foundations
paper: "SAM 3: Segment Anything with Concepts"
year: 2025
arxiv: "2511.16719"
doi: null
source: https://arxiv.org/abs/2511.16719
project: https://ai.meta.com/sam3
code: https://github.com/facebookresearch/sam3
tags:
  - promptable-concept-segmentation
  - open-vocabulary-segmentation
  - video-object-tracking
  - vision-language
  - data-engine
  - segment-anything
created: 2026-10-15
updated: 2026-10-15
---

# SAM 3：概念驱动的开放词汇分割

## 基本信息

| 项目 | 内容 |
|---|---|
| 正式名称 | **SAM 3: Segment Anything with Concepts** |
| 作者 / 机构 | Nicolas Carion、Laura Gustafson、Yuan-Ting Hu、Shoubhik Debnath、Ronghang Hu 等；Meta Superintelligence Labs |
| 论文 | [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) |
| 官方代码 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) |
| 归档领域 | 基础模块 / 开放词汇分割与跟踪 |
| 核心关键词 | Promptable Concept Segmentation、开放词汇检测、实例分割、视频跟踪、presence head、SA-Co、数据引擎 |
| 本地原文 | `assets/SAM 3 - 概念驱动的开放词汇分割/sources/paper.pdf` |
| 重要性评估 | **★★★★★（5/5）**：它把 SAM 从“点/框指定一个实例”推进到“用概念找全所有实例”，并把开放词汇检测、像素级分割、视频身份跟踪和交互修正统一为可提示系统。 |

> [!important]
> 本文研究的是 **SAM 3: Segment Anything with Concepts**，不是 SAM 3D；前者解决图像/视频中的概念检测、分割和跟踪，后者属于三维任务，二者不可混用。

**一句话结论：**

SAM 3 定义 **Promptable Concept Segmentation（PCS）**：用户用短名词短语、正/负图像 exemplar 或二者组合描述一个概念，模型返回图像或视频中**所有匹配实例的 mask 与唯一身份**；它以共享 PE backbone 的 DETR 检测器和 SAM 2 式记忆跟踪器为核心，通过 presence head 解耦“概念是否存在”与“实例在哪里”，并依靠含 4M 独特短语、52M 高质量 masks 及大规模合成标注的数据引擎，在 SA-Co 图像和视频 benchmark 上把既有系统性能提高到约两倍。

---

## 1. 研究背景与问题定义

### 1.1 从 PVS 到 PCS

SAM 1/2 的核心任务是 **Promptable Visual Segmentation（PVS）**：给一个点、框或 mask，分割该视觉提示指向的单个对象；SAM 2 再把该对象沿视频传播。它们擅长回答“分割**这个**对象”，却没有系统解决“找出画面里**所有属于某个概念**的对象”。

SAM 3 提出的 PCS 则回答：

- 图中所有“striped cat”在哪里？
- 视频中所有“hard hat”是谁，跨帧如何保持 ID？
- 文本概念不够精确时，能否框一个正例或误检负例，让模型概括并修正整类结果？

![](assets/SAM%203%20-%20概念驱动的开放词汇分割/figures/01-promptable-concept-segmentation.png)

> **图 1：PVS 与 PCS 的任务差异。** 左侧点提示只修改一个实例；右侧短名词短语或正/负 exemplar 定义一个概念，模型需要在图像或视频中找全所有匹配实例，并允许继续交互修正。来源：论文 Figure 1，p.1。

### 1.2 为什么“检测 + 分割 + 跟踪”不能简单拼接

传统开放词汇管线常用 detector 产生 box，再把 box 交给 SAM 分割；视频中再接 tracking-by-detection。这种级联有三类问题：

1. detector 的漏检决定了最终 mask 上限，分割器无法恢复未提出的实例；
2. 每个 query 同时承担“概念是否存在”和“对象在哪里”，hard negative 容易制造大量假阳性；
3. detector 希望对身份无关、找全同类对象，tracker 却要区分相似实例并长期维持身份，两者联合端到端训练会发生目标冲突。

SAM 3 的答案是：共享视觉 backbone，但显式解耦 **recognition、localization、tracking**。

### 1.3 核心贡献

1. **定义 PCS 任务**：统一文本短语、正/负图像 exemplar、点/框/mask refinement；输出所有概念实例的 mask，并在视频中保持唯一 ID。
2. **统一但解耦的架构**：PE 图文编码器 + DETR detector + presence head + SAM 2 memory tracker；检测与跟踪共享 backbone，但保留不同目标。
3. **大规模数据引擎**：用 ontology、SAM 3 proposal、MLLM noun-phrase generation、AI/human mask verification 与 exhaustivity verification 构成闭环，重点挖掘 hard negatives 和当前模型失败例。
4. **SA-Co 数据与 benchmark**：训练集覆盖 4M 独特高质量 noun phrases 和 52M masks；评测集含 207K 独特概念、121K 图像/视频与 3M+ media-phrase pairs。
5. **图像和视频统一验证**：在开放词汇图像分割、视频 PCS、VOS、交互分割、计数与 MLLM agent 场景中取得显著提升。

---

## 2. 任务定义与输入输出

### 2.1 输入

给定图像或不超过约 30 秒的短视频 $V=\{I_t\}_{t=1}^{T}$，提示 $P$ 可包含：

- **短名词短语（noun phrase, NP）**：名词及少量修饰词，如“yellow school bus”“striped cat”；文本提示对所有帧全局有效。
- **正/负 image exemplars**：某一帧中的 bounding box 加二元标签；正例说明“像这个”，负例说明“不要这种误检”。
- **PVS visual prompts**：点、框或 mask，用于对某个实例继续精修。

所有概念级提示必须保持类别定义一致，例如不能用“fish”文本再用“tail”正例修改，因为这改变了概念本身。

### 2.2 输出

对图像，输出所有匹配实例：

$$
\mathcal{O} = \{(m_i,b_i,s_i)\}_{i=1}^{N},
$$

其中 $m_i$ 是像素 mask，$b_i$ 是 box，$s_i$ 是置信度；另可输出概念对应的 semantic mask。对视频，输出 masklets：

$$
\mathcal{M}=\{(\text{id}_i,\{m_{i,t}\}_{t=1}^{T})\}_{i=1}^{N},
$$

即每个实例跨帧的 masks 与稳定身份 ID。

### 2.3 任务边界

SAM 3 的文本范围刻意限制为可视觉落地的简单名词短语，不直接解决长 referring expression、关系推理或复杂组合问题。论文把复杂语言交给 MLLM 分解为多个 noun phrases，再调用 SAM 3 作为视觉工具。

---

## 3. 核心方法

### 3.1 总体架构

SAM 3 约 **850M 参数**：PE vision encoder 约 450M、text encoder 约 300M，detector + tracker 约 100M。模型由三条主线组成：

1. **概念编码**：文本 encoder 编码 noun phrase；exemplar encoder 用位置、正负标签和 ROI-pooled image feature 编码图像示例。
2. **当前帧检测与分割**：融合后的 prompt tokens 条件化图像特征；DETR object queries 定位所有匹配对象并输出 box、instance mask、semantic mask。
3. **跨帧跟踪**：SAM 2 式 tracker 从 memory bank 传播已有 masklets；detector 持续发现新实例，两者经 IoU matching 合并，并由 detector 定期纠正漂移。

![](assets/SAM%203%20-%20概念驱动的开放词汇分割/figures/02-detector-tracker-architecture.png)

> **图 2：SAM 3 的检测—跟踪双路径。** 黄色 detector 负责按文本/示例在当前帧发现概念实例；蓝色 tracker 依据前一帧和 memory bank 传播已有身份；两者共享 PE 图像编码器，最后合并既有与新检测 masks。来源：论文 Figure 4，p.4。

### 3.2 PE 图文 backbone 与 prompt 融合

image/text encoders 先在 **5.4B image-text pairs** 上做 PE 式对比预训练。视觉侧输入通常为 1008×1008，32 层中仅 4 层使用 global attention，其余使用 336×336 对应窗口注意力；文本 encoder 为 causal transformer，最大上下文 32。

exemplar 由三部分组成：

- box position embedding；
- positive/negative label embedding；
- ROI-pooled visual feature。

小型 transformer 将多个 exemplar 编成 prompt tokens。文本 token 与 geometry/exemplar tokens 共同进入 6-block fusion encoder，通过 cross-attention 条件化 frame embeddings。随后 6-block DETR decoder 使用默认 $Q=200$ 个 learned object queries 产生候选对象。

### 3.3 Presence head：把“有没有”与“在哪里”拆开

开放词汇检测的关键难点是：每个 proposal query 若同时理解全图语义并做局部定位，两个目标会互相干扰。SAM 3 将概率分解为：

$$
p(q_i\text{ matches NP})
= p(q_i\text{ matches NP}\mid \text{NP appears})
\cdot p(\text{NP appears}).
$$

- **presence token** 汇聚全图上下文，单独预测 $p(\text{NP appears})$；
- 每个 object query 只在概念存在的条件下学习局部匹配与排序；
- 推理时二者相乘得到最终 object score。

训练中，image-level negative 只监督 presence binary cross-entropy，不把所有 object queries 都压成负类。这样 hard negatives 主要改善“是否存在”的校准，而不会破坏定位 query 的相对排序。presence score 还复用于 semantic segmentation head；在已知概念必然存在的计数/PVS 场景可直接设为 1。

### 3.4 分割头与歧义处理

instance/semantic segmentation 共享 MaskFormer 风格 head；SimpleFPN 将单尺度 ViT 特征转为多尺度表示。semantic mask 直接来自 conditioned frame features，instance mask 还结合 decoder object query。

开放词汇概念常有多义性，例如“large circular shape”可能合理指向不同区域。若只训练单一输出，模型会把多个解释都以中等置信度输出，产生重叠 mask。论文采用 $K=2$ 个 ambiguity experts，并使用 winner-takes-all：

$$
k^*=\arg\min_k \mathcal{L}_k,\qquad
\mathcal{L}_{\text{WTA}}=\mathcal{L}_{k^*}.
$$

只有最低 loss 的 expert 接收梯度，使不同 expert 学会不同解释；另训练 classifier 选择推理时的 expert。该设计使 overlapping instances 下降约 15%。

### 3.5 视频：detector 发现，tracker 保持身份

每帧执行：

$$
\hat M_t=\operatorname{propagate}(M_{t-1}),
$$

$$
O_t=\operatorname{detect}(I_t,P),
$$

$$
M_t=\operatorname{match\_and\_update}(\hat M_t,O_t).
$$

tracker 继承 SAM 2 的 prompt encoder、mask decoder、memory encoder 与 memory bank。检测结果与传播 mask 通过 IoU matching 关联；未匹配的新 detection 会生成新 masklet。为减少拥挤场景中的重复和漂移，系统增加：

- **15 帧 confirmation delay**：观察半秒后再确认新轨迹；
- **Masklet Detection Score**：统计时间窗内与 detector 一致的次数，移除或抑制不稳定 masklet；
- **duplicate removal**：两个 masklet 持续匹配同一 detection 时删除较晚启动者；
- **periodic re-prompting**：默认每 16 帧用高置信 detection mask 重置 tracker memory；
- **detection-guided re-prompting**：检测到 tracker box 明显漂移时立即用 detection 纠正。

这种设计没有强迫 detector 与 tracker 共用相同 identity objective：detector 对身份无关、负责“找全”，tracker 对身份敏感、负责“不串 ID”。

### 3.6 交互：概念级与实例级两种模式

- PCS exemplar 交互会泛化到整类：框一个漏检正例，可找出更多相似实例；框一个误检负例，可压制整类混淆。
- PVS point/box 交互只修改一个实例及其视频 masklet。

论文实验显示，概念级交互前几次更高效：3 次 clicks 后比 text-only 高 **21.6 cgF1**，并比逐实例的理想 PVS 修正高 2.0；约 4 次后 plateau，此时切换 PVS 修边界更合适。

---

## 4. 数据引擎与训练

### 4.1 数据引擎闭环

SAM 3 的提升很大程度来自“模型—AI verifier—人类”闭环，而非只靠架构。流程为：

1. 从多域 media pool 挖掘图像/视频；
2. ontology 与 MLLM 生成 noun phrases，包含长尾、细粒度概念和针对当前模型的 hard negatives；
3. SAM 3 产生候选 masks；
4. **Mask Verification（MV）** 判断单个 mask 是否正确；
5. **Exhaustivity Verification（EV）** 判断是否找全该概念的全部实例；
6. 通过者直接入库，失败样本交给人类添加、删除或修改 mask；
7. 用新数据更新 SAM 3 与 verifier，再主动挖掘下一轮失败样本。

![](assets/SAM%203%20-%20概念驱动的开放词汇分割/figures/03-data-engine.png)

> **图 3：SAM 3 数据引擎。** ontology 引导 data mining 和 noun phrase proposal，SAM 3 生成 masks；AI/human verifier 分别检查 mask 质量和穷尽性，只有困难失败例进入人工 correction。虚线回路表示新标注继续训练 proposal/verifier 模型。来源：论文 Figure 5，p.5。

### 4.2 四个数据阶段

1. **Phase 1：全人工验证**。captioner + parser 提 NP，SAM 2 + 开放词汇 detector 提 mask；收集 4.3M image-NP pairs 形成初始 SA-Co/HQ。
2. **Phase 2：人类 + AI 验证**。用 Phase 1 标签微调 Llama 3.2，作为 MV/EV verifier；更新 SAM 3 六次，新增 122M image-NP pairs，并开始生成对当前模型有攻击性的 hard negatives。
3. **Phase 3：规模与领域扩展**。扩至 15 个数据域；使用基于 Wikidata 的 **22.4M-node ontology**（17 个顶级、72 个子类别）覆盖长尾和细粒度概念；再更新 SAM 3 七次、verifier 三次，新增 19.5M pairs。
4. **Phase 4：视频**。用 scene/motion filter、对象密度与跟踪失败挖掘困难片段，生成并校正 masklets；得到 52.5K videos、467K masklets。

最终高质量图像数据 SA-Co/HQ 含 **5.2M images、4M unique NPs、52M masks**；合成 SA-Co/SYN 含约 **38M phrases、1.4B masks**。AI verifier 使图像数据引擎总体吞吐约翻倍；在新 Food&Drink 域，负短语标注约 5× 加速，正短语约 36% 加速。

### 4.3 SA-Co benchmark

评测集包含 **207K unique phrases、121K images/videos、3M+ media-phrase pairs**，且包含大量 hard negative labels。主要 split：

- **Gold**：7 domains，每个 image-NP pair 由 3 名专家标注，用于估计人类上限和歧义；
- **Silver**：10 domains，单人标注；
- **Bronze / Bio**：来自既有数据集或 box→SAM 2 mask；
- **VEval**：视频 PCS，覆盖 3 domains。

指标把 PCS 拆成两个方面：

- $\mathrm{pmF1}$：只在正 media-phrase pairs 上衡量 localization；
- $\mathrm{IL\_MCC}\in[-1,1]$：图像级判断概念是否存在，衡量 recognition/calibration；
- 主指标：

$$
\mathrm{cgF1}=100\times\mathrm{pmF1}\times\mathrm{IL\_MCC}.
$$

所有预测使用 0.5 confidence threshold，避免 AP 对校准不敏感的问题。Gold 对同一短语保存三种专家解释，采用 oracle comparison 允许合理歧义。

### 4.4 四阶段模型训练

1. **Stage 1：PE 预训练**。5.4B image-text pairs；视觉/文本 encoder 分别约 450M/300M。
2. **Stage 2：detector pre-training**。约 95K iterations，batch 896，联合训练 detector 与图文 encoder；同时用图像/视频帧训练 PCS 和 PVS。20% 样本丢弃文本转 visual query，20% 增加 box exemplar。
3. **Stage 3：高质量微调与 presence post-training**。只保留通过 exhaustivity verification 的高质量数据，加入正/负 exemplar 模拟交互和 presence loss；约 5K iterations，batch 512，学习率缩小为 Stage 2 的 0.025。
4. **Stage 4：tracker training**。冻结 backbone，约 190K iterations，batch 512；随后再用 16/32-frame context 训练 60K iterations。

将 tracker 放在最后且冻结 backbone，可以强化时空 identity 而不破坏前面已学到的开放词汇识别和图像空间 grounding。

---

## 5. 实验结果

### 5.1 图像 PCS：开放词汇检测与分割

主要 baseline 包括 OWLv2、Grounding DINO、LLMDet、APE、DINO-X、Gemini 2.5；对只输出 box 的模型，论文用其 box prompt SAM 1 得到 segmentation。

关键结果：

- LVIS zero-shot instance mask AP：SAM 3 **48.5**，DINO-X 报告值 38.5；
- SA-Co/Gold instance cgF1：SAM 3 **54.1**，最强对比 OWLv2* 为 24.6，人类估计 72.8；
- SA-Co/Gold box cgF1：SAM 3 **55.7**，OWLv2* 为 24.5；
- 开放词汇 semantic segmentation：ADE-847 / PascalConcept-59 / Cityscapes 为 **13.8 / 60.8 / 65.2 mIoU**，均高于表中 APE specialist baseline。

![](assets/SAM%203%20-%20概念驱动的开放词汇分割/figures/04-open-vocabulary-results.png)

> **图 4：开放词汇定性结果。** 从“white flower”“young plant”到“Adirondack chair”“toilet roll holder”，SAM 3 能对细粒度、长尾 noun phrases 生成更完整、更贴边界的 masks；OWLv2 级联方案更容易漏实例、混淆概念或产生粗糙区域。来源：论文 Figure 2，p.2。

这些数字支持“约两倍”主张，但要注意：SA-Co 是新提出 benchmark，训练数据与评测概念体系均由同一项目构建；论文通过 Gold 多人标注、跨域 split 和外部 LVIS/COCO 结果减轻了这一偏差，却不能完全消除 benchmark ownership 带来的优势。

### 5.2 exemplar、few-shot 与计数

单个 exemplar 的正样本 AP+：

| 数据集 | T-Rex2 image exemplar | SAM 3 image exemplar | 提升 |
|---|---:|---:|---:|
| COCO | 58.5 | **76.8** | +18.3 |
| LVIS | 65.8 | **76.0** | +10.2（论文正文记 +10.3） |
| ODinW13 | 61.8 | **82.2** | +20.4（论文正文记 +20.5） |

text + image exemplar 在 COCO/LVIS 可进一步达到 78.1/78.4。计数方面，SAM 3 在 CountBench 达 MAE 0.12、accuracy 93.8，优势来自“先枚举并分割每个对象再计数”，而不是让 MLLM 直接回归数字。

### 5.3 视频 PCS 与身份跟踪

在 SA-Co/VEval：

| 模型 | SA-V cgF1 / pHOTA | YT-Temporal-1B cgF1 / pHOTA | SmartGlasses cgF1 / pHOTA |
|---|---:|---:|---:|
| LLMDet + SAM 3 Tracker | 2.3 / 30.1 | 8.0 / 37.9 | 0.3 / 18.6 |
| SAM 3 Detector + tracking-by-detection | 25.7 / 55.7 | 47.6 / 68.2 | 29.7 / 60.0 |
| **SAM 3** | **30.3 / 58.0** | **50.8 / 69.9** | **36.4 / 63.6** |
| Human | 53.1 / 70.5 | 71.2 / 78.4 | 58.5 / 72.3 |

替换 detector 会大幅下降，说明 tracker 无法弥补概念识别缺陷；只用 tracking-by-detection 也弱于 memory tracker，说明视频一致性不等于逐帧检测后简单关联。SAM 3 在 pHOTA 上达到人类约 80% 以上，但 cgF1 仍有较大差距，主要瓶颈仍是开放词汇概念是否存在及是否找全。

在传统 VOS 上，SAM 3 也继承并提升 PVS：MOSEv2 J&F 从 SAM 2.1 L 的 47.9 提到 **60.3**；DAVIS17 92.2，LVOSv2 88.5，均为表中最佳或接近最佳。

### 5.4 与 MLLM 组合

SAM 3 Agent 让 MLLM 把复杂查询拆成 noun phrases，调用 SAM 3 返回 masks，再根据视觉结果迭代。Qwen2.5-VL 72B 版本在 ReasonSeg val/test 达 74.6/70.8 gIoU；Gemini 2.5 Pro 版本达 77.0/74.0。该结果说明 SAM 3 适合作为 MLLM 的像素级工具，但复杂 query 的最终正确性仍取决于 MLLM 的概念分解和验证。

### 5.5 消融实验

#### 5.5.1 Presence head

短训练消融中，presence head 将 cgF1 从 50.7 提到 **52.2**，IL_MCC 从 0.77 提到 **0.82**，但 pmF1 从 65.4 略降至 63.4。它主要改善“不要在不存在时乱检”，不直接改善正样本局部 mask。最佳监督策略是：presence 与 object score 分开监督，负概念不向 object query 反传，推理时再相乘。

#### 5.5.2 Hard negatives

每图 hard negatives 从 0 增至 30：cgF1 从 28.3 增至 **43.0**，IL_MCC 从 0.44 增至 **0.68**，pmF1 基本保持约 62—63。它再次证明开放词汇 PCS 的首要短板是校准和概念拒识，而不只是 mask quality。

#### 5.5.3 数据组合与 scaling

- 仅 SA-Co/EXT：23.7 cgF1；
- + SA-Co/SYN：32.8（+8.8）；
- EXT + HQ：45.5；
- EXT + SYN + HQ：47.4。

HQ 的收益最大，说明 exhaustive masks 和人工 correction 仍不可替代；SYN 在覆盖广度与域扩展上继续提供增益。HQ 与 SYN scaling curve 均未完全饱和。

#### 5.5.4 AI verifiers

完整 SAM 3 在内部消融设置为 54.0 cgF1；以 EV verifier presence 替换模型 presence 后为 61.2，再用 MV verifier 过滤坏 masks 后为 **62.3**。AI verifier 系统补上了约一半人类差距，也暴露主模型尚未完全吸收 data engine teacher 的知识。

#### 5.5.5 Backbone

PE-L+ + 对齐 text encoder 在 SA-Co/Gold/COCO-O 为 43.2/42.5；同视觉 encoder 换 DistilRoBERTa 为 38.1/39.6；DINOv2-L + DistilRoBERTa 为 35.3/31.9；Hiera-L 为 32.8/22.0。PCS 同时需要强空间定位和已对齐的广覆盖图文语义，单独换一个强视觉 backbone 并不足够。

---

## 6. 与相关工作的关系

### 6.1 相对 SAM 1/2

SAM 3 保留 PVS 与交互 refinement，但把提示从“几何指定实例”扩展到“语义定义概念”。其增量不是简单增加 text prompt，而是新建 detector、presence head、exemplar encoder 与 SA-Co 数据体系，同时沿用 SAM 2 tracker 处理 identity。

### 6.2 相对开放词汇检测/分割

OWLv2、Grounding DINO、LLMDet 等主要输出 boxes；将 box 再交给 SAM 会形成不可联合校准的两阶段系统。SAM 3 的 detector 直接产生 box、instance mask 和 semantic mask，并支持正/负 exemplar 交互。与 GLEE 等图像/视频 grounding 模型相比，SAM 3 重点支持 exemplar 与交互式纠错。

### 6.3 相对 tracking-by-detection

传统方法逐帧检测后按运动/外观关联；SAM 3 让 memory tracker 连续传播 masklet，同时用 detector 发现新对象和校正漂移。其优势来自“检测负责召回、记忆负责身份”，而非把所有目标压进同一个 query 机制。

---

## 7. 与 VLM 的关系

### 7.1 SAM 3 是视觉—语言 grounding 基础模块

SAM 3 本身已有对齐的 image/text encoder，能把 noun phrase 映射到 box、mask 和跨帧 ID，是从 VLM 高层语义到像素空间执行的重要桥梁。相比只返回文本答案的 VLM，它提供：

- exhaustive instance enumeration；
- 像素级 grounding；
- 可计算的对象数量、面积、轨迹和遮挡；
- 正/负 exemplar 形式的交互式概念学习。

### 7.2 它不是完整 MLLM

模型文本上下文仅 32，任务限定为简单 noun phrases；它不负责关系推理、长指代表达、因果解释和对话规划。论文的 SAM 3 Agent 体现了合理分工：**MLLM 解析复杂意图并规划 noun phrases，SAM 3 负责可靠地把概念落到 pixels/instances。**

### 7.3 对 VLM 系统设计的启示

1. grounding 不应只作为 VLM decoder 的附属 token head，而可由专门视觉工具承担；
2. hard negatives 和 presence calibration 对减少“视觉幻觉”很关键；
3. 将 recognition 与 localization 解耦，类似于把“是否看见”与“在哪里”拆成两个可校准模块；
4. video VLM 若只做逐帧 grounding 会丢 ID，需要外部 memory tracker 或对象级共享记忆。

---

## 8. 局限与批判性评价

### 8.1 作者明确指出的局限

1. **域外细粒度概念**：飞机型号、医学术语、热成像等 niche domain 的零样本泛化不足；少量微调或 data engine 自动扩域可改善，但需要额外训练。
2. **语言范围有限**：只支持一到两个属性的简单 noun phrase，不直接支持长 referring expressions 和复杂多属性组合。
3. **视频成本随对象数线性增长**：每个对象都有独立 SAM 2 masklet。30 FPS 时，约 10 objects 需 2×H200，28 objects 需 4×H200，64 objects 需 8×H200。
4. **缺少对象间共享上下文**：每个 masklet memory 相对独立，拥挤、多对象互相遮挡时难以利用全局关系消歧。
5. **交互模式割裂**：concept-level PCS 与 instance-level PVS 之间使用硬 mode switch，不能完全无缝混合。

### 8.2 进一步批判

- **数据优势难以外部复现**：5.4B 图文对、1.4B synthetic masks、22.4M-node ontology 与内部 media pool 决定了相当部分能力；开源 checkpoint 可用，但第三方难以重建训练因果。
- **benchmark 与训练生态同源**：SA-Co 的任务定义、ontology、数据引擎和评测由同一团队设计，模型可能更适应其 phrase 风格；外部 benchmark 的强结果很重要，但 PCS 的跨机构复核仍需时间。
- **“两倍性能”依赖新指标和低 baseline 区间**：SA-Co 上 baseline cgF1 较低，倍数提升很显著；同时应看人类上限——Gold 上 54.1 对 72.8，仍有约 18.7 点绝对差距。
- **presence head 的收益主要是拒识**：正样本 localization 并未同步提升；实际系统若 noun phrase 必然存在，presence 可能无益，论文在 counting 中也将其关闭。
- **系统工程复杂度高**：detector、tracker、verifier、ambiguity experts、memory bank 与多阶段训练共同贡献性能，很难用单一小规模实验完整复现。

---

## 9. 复现成本、实践建议与 Future Work

### 9.1 完整复现成本

完整训练接近不可由普通实验室复现：Stage 1 需要 5.4B image-text pairs，Stage 2/3 需要上亿 image-NP pairs 和十亿级 masks，Stage 4 还要大规模视频 masklets；模型约 850M 参数、输入 1008²，训练含 95K×batch896、5K×batch512、190K×batch512 以及 60K 长上下文 steps。论文未披露完整 GPU 数与总 GPU-hours，因此不能给出可靠总成本。

### 9.2 现实可行的使用/复现实验

- **直接推理**：优先使用官方 checkpoint；论文报告 H200 上单图、100+ objects 约 30 ms，但其他 GPU、视频和对象数需自行 benchmark。
- **验证 presence 设计**：可在较小 DETR + open-vocabulary dataset 上实现全局 presence token，并对比“所有 query 监督负类”与“仅 presence 监督负概念”。
- **验证视频组合**：固定 detector 与 SAM 2 tracker，逐步加入 confirmation delay、MDS suppression 和 periodic re-prompting，使用 HOTA/ID switch 分析每项收益。
- **领域适配**：先用少量人工数据校准 EV/MV verifier，再生成 in-domain SYN data；不要直接把未验证 pseudo masks 大量灌入模型。
- **VLM 集成**：让 MLLM 输出受约束 noun phrases 和正/负条件，再由 SAM 3 返回结构化 masks；保留失败重试与实例级 PVS 修正通道。

### 9.3 Future Work

论文明确或自然导出的后续方向包括：

1. 为多个对象设计**共享全局 memory**，同时降低线性随对象数增长的计算量并增强遮挡消歧；
2. 无缝融合 concept-level 与 instance-level prompts，替代当前硬 mode switch；
3. 用自动域扩展和 verifier teacher 持续覆盖医学、遥感、热成像、工业等细粒度概念；
4. 将更复杂文本查询直接分解为可验证的概念程序，而不是依赖一次性 MLLM noun-phrase 生成；
5. 让主模型蒸馏吸收 MV/EV verifier 的额外能力，缩小“单模型”与“模型 + verifier 系统”的差距；
6. 为视频设计多对象并行或 multiplexed tracker，以更低成本支持密集场景。

---

## 10. 材料来源

完整来源与图片追溯见 [sources.md](assets/SAM%203%20-%20概念驱动的开放词汇分割/sources.md)。
