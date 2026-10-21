---
type: paper-note
status: done
domain: VLM
series: BLIP
series_order: 2
series_hub: "[[BLIP 系列：从数据自举到统一理解生成 - Survey]]"
paper: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
year: 2023
arxiv: 2301.12597
doi: null
source: https://arxiv.org/abs/2301.12597
project: https://salesforce-lavis.github.io/projects/blip2.html
code: https://github.com/salesforce/LAVIS
tags:
  - VLM
  - Q-Former
  - frozen-models
  - multimodal-generation
created: 2026-10-15
updated: 2026-10-15
---

# BLIP-2：冻结底座的高效视觉语言桥接

> 系列导航：[[BLIP - 生成式字幕过滤的统一视觉语言预训练]] → **BLIP-2** → [[InstructBLIP - 指令感知视觉特征抽取]] → [[BLIP-3 - 开放大规模多模态模型家族]] → [[BLIP3-o - 全开放统一多模态理解与生成]] · [[BLIP 系列：从数据自举到统一理解生成 - Survey|系列综述]]

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models |
| 作者 | Junnan Li 等，Salesforce Research |
| 发表 | ICML 2023 |
| arXiv | 2301.12597 |
| 官方项目 | https://salesforce-lavis.github.io/projects/blip2.html |
| 官方代码 | https://github.com/salesforce/LAVIS |
| 一句话结论 | BLIP-2 用两阶段训练的 188M Q-Former，把冻结视觉编码器压缩出的语言相关视觉信息转换成冻结 LLM 可解释的软提示，以远少于 Flamingo 的可训练参数获得强零样本生成与 VQA。 |

---

## 1. 研究背景

### 1.1 冻结模型之间仍存在模态鸿沟

视觉编码器和 LLM 都已在各自领域形成强大基础模型，直接复用它们看似比端到端重训 VLM 更经济。但“冻结”也放大了对齐难题：LLM 从未见过图像，其表示空间不会主动迁就视觉特征；若只靠图像到文本生成损失训练一个投影层，桥接模块既要抽取语言相关视觉内容，又要一次性学会 LLM 的输入分布，容易训练不足或让生成模型产生灾难性遗忘。

Flamingo 的方案是在冻结 LM 各层插入门控跨注意力，能力强但最大版本仍需约 10.2B 可训练参数与私有交错数据。BLIP-2 的核心判断是先把“看懂图文关系”与“让 LLM 读懂视觉提示”拆开：第一阶段只连接冻结视觉编码器，显式训练表示对齐；第二阶段再连接冻结 LLM，学习生成接口。

### 1.2 核心贡献

Q-Former 以 32 个可学习 query 从冻结 ViT 中抽取固定长度视觉表示，形成强信息瓶颈。第一阶段用 ITC、ITG、ITM 三个目标和不同 attention mask 学到语言相关视觉表示；第二阶段把 query 输出线性投影成 LLM 词嵌入维度，作为 soft visual prompt。视觉端与 LLM 全程冻结，只有 Q-Former 和投影层训练，因此能替换 ViT-L/14、EVA ViT-g/14、OPT 与 FlanT5 等不同底座。

![](assets/BLIP-2%20-%20冻结底座的高效视觉语言桥接/figures/01-overview.png)

> **Figure 1，PDF 第 1 页。** 左半先用冻结图像编码器训练 Q-Former 的视觉语言表示，右半再把同一桥接器接到冻结 LLM 做生成学习。两阶段拆分是 BLIP-2 与“直接投影后生成”的根本差别：Q-Former 在见到 LLM 前已经被迫提取与文本最相关的视觉证据。来源：https://arxiv.org/abs/2301.12597

---

## 2. 任务定义与方法

### 2.1 输入、输出与 Q-Former 瓶颈

预训练输入是一幅图像 $I$ 与配对文本 $T$。冻结图像编码器产生 $M$ 个视觉 token；Q-Former 的 32 个 768 维 query 经每隔一个 block 插入的 cross-attention 读取视觉 token，输出 $Z\in\mathbb{R}^{32\times768}$。例如 ViT-L/14 输出约 $257\times1024$，Q-Former 将其压缩为固定 32 token，且长度不随分辨率改变。

Q-Former 含两个共享 self-attention 的子路径：image transformer 通过 cross-attention 读视觉特征；text transformer 既可编码也可解码文本。其参数由 BERT-base 初始化，cross-attention 随机初始化，总计 188M 参数。不同任务不更换主干，而是改变 query 与 text 的可见性掩码。

### 2.2 第一阶段：视觉语言表示学习

![](assets/BLIP-2%20-%20冻结底座的高效视觉语言桥接/figures/02-stage1-qformer.png)

> **Figure 2，PDF 第 3 页。** 左侧是 Q-Former 共享 self-attention、视觉 cross-attention 与 query/text 双分支；右侧三种 mask 分别服务 ITM、ITG、ITC。完整图说明三个目标不是三套网络，而是同一参数在不同信息流约束下学习对齐、生成与细粒度匹配。来源：https://arxiv.org/abs/2301.12597

**ITC（Image-Text Contrastive）** 使用 unimodal mask，query 与文本互不可见。文本以 `[CLS]` 表示，图像—文本相似度取所有 query 与文本向量相似度的最大值：

$$s(I,T)=\max_{q\in\{1,\dots,32\}}z_q^\top t$$

再以批内其他配对为负样本做双向对比学习。`max` 允许不同 query 专门响应不同视觉语义，而无需先平均掉局部差异。

**ITG（Image-grounded Text Generation）** 使用多模态因果 mask：query 可彼此交互但不可看文本；每个文本 token 可看全部 query 与过去文本。要生成配对描述，图像信息必须先穿过 32-query 瓶颈，因此 Q-Former 被迫保留足以支撑语言生成的视觉内容。

**ITM（Image-Text Matching）** 使用双向 mask，让 query 与文本充分交互，再对每个 query 做二分类并平均 logits；训练使用 hard negative mining。它补足 ITC 的全局粗粒度对齐，学习局部、细粒度配对关系。

### 2.3 第二阶段：冻结 LLM 的视觉软提示

![](assets/BLIP-2%20-%20冻结底座的高效视觉语言桥接/figures/03-stage2-llm.png)

> **Figure 3，PDF 第 4 页。** 对 decoder-only OPT，投影后的 query token 前置到文本 embedding 并做标准语言建模；对 encoder-decoder FlanT5，视觉 token 与文本前缀进入 encoder，decoder 生成后缀。两条路径说明 Q-Former 与具体 LLM 架构解耦。来源：https://arxiv.org/abs/2301.12597

第二阶段用全连接层把 $Z$ 投影到 LLM embedding 维度，得到视觉提示 $P_v$。对 OPT 类 decoder-only LLM：

$$\mathcal{L}_{LM}=-\sum_t\log p(y_t\mid P_v,y_{<t})$$

对 FlanT5 类 encoder-decoder LLM，文本随机切为 prefix/suffix，encoder 接收视觉提示与 prefix，decoder 预测 suffix。LLM 参数保持冻结，视觉提示只需落入其已有语言空间；第一阶段预训练显著减轻第二阶段对齐负担。论文曲线显示，去掉第一阶段时零样本 VQA 明显下降，OPT 甚至随着第二阶段训练继续而性能恶化，作者将其解释为灾难性遗忘/对齐失败。

---

## 3. 训练数据与训练流程

### 3.1 数据与合成字幕

BLIP-2 复用 BLIP 的 1.29 亿图像数据：COCO、Visual Genome、CC3M、CC12M、SBU，以及从 LAION-400M 取的 1.15 亿图像。作者用 BLIP-large 为每幅网络图生成 10 条候选字幕，再与原网页字幕一起由 CLIP ViT-L/14 图文相似度排序，保留前两条，每步随机采一条。这一 CapFilt 流程提高语言质量，但也把 BLIP 与 CLIP 的偏差继续传入训练数据。

### 3.2 底座、优化与算力

视觉端比较 CLIP ViT-L/14 与 EVA-CLIP ViT-g/14，移除 ViT 最后一层并取倒数第二层特征；语言端比较 OPT 2.7B/6.7B 与 FlanT5-XL/XXL。第一阶段训练 250k step，ViT-L/ViT-g batch 分别为 2,320/1,680；第二阶段 80k step，OPT/FlanT5 batch 分别为 1,920/1,520。图像为 $224\times224$，AdamW 峰值学习率 $10^{-4}$、warmup 2k step、weight decay 0.05。

最大 ViT-g + FlanT5-XXL 在单台 16×A100 40GB 上，第一阶段少于 6 天、第二阶段少于 3 天。与 Flamingo-80B 的 1,536 TPUv4×15 天相比成本显著下降，但仍需要 1.29 亿图像、BLIP 字幕生成与 CLIP 重排序。官方 LAVIS 提供实现和权重，使下游推理与微调可复现；从头重建训练数据流水线仍昂贵。

---

## 4. 实验与消融

### 4.1 零样本 VQA 与基线

官方零样本表显示，ViT-g + FlanT5-XXL 的总参数 12.1B，但视觉语言预训练中可训练参数仅 108M；主表统一以 Q-Former 的 188M 作为框架级可训练模块规模。不同表的参数口径有所差异：188M 是完整 Q-Former，108M 是特定第二阶段/配置的实际可训练部分，阅读时不能混为一谈。

| 模型 | 可训练参数 | 总参数 | VQAv2 test-dev | OK-VQA test | GQA test-dev |
|---|---:|---:|---:|---:|---:|
| Flamingo-3B | 1.4B | 3.2B | 49.2 | 41.2 | 论文未列 |
| Flamingo-9B | 1.8B | 9.3B | 51.8 | 44.7 | 论文未列 |
| Flamingo-80B | 10.2B | 80B | 56.3 | **50.6** | 论文未列 |
| BLIP-2 ViT-g + OPT-6.7B | 108M | 7.8B | 52.6 | 36.4 | 36.4 |
| BLIP-2 ViT-g + FlanT5-XL | 107M | 4.1B | 63.0 | 40.7 | 44.2 |
| BLIP-2 ViT-g + FlanT5-XXL | 108M | 12.1B | **65.0** | 45.9 | **44.7** |

最佳 BLIP-2 在 VQAv2 比 Flamingo-80B 高 8.7 个百分点，并使用约 54 倍更少的可训练参数；但 OK-VQA 仍落后 4.7 点，作者认为该任务更依赖开放世界知识，而 Flamingo 的 70B Chinchilla 比 11B FlanT5-XXL 知识容量更大。底座消融规律一致：ViT-g 优于 ViT-L，同系列更大 LLM 更好，指令微调 FlanT5 明显优于无监督 OPT。

### 4.2 字幕、检索与训练目标消融

BLIP-2 主表报告 NoCaps 零样本 CIDEr 121.6、SPICE 15.8，高于 BLIP 的 113.2/14.8 与 SimVLM 的 112.2；Flickr30K 零样本 text retrieval R@1 为 97.6、image retrieval R@1 为 89.7。COCO 检索消融中，ITC+ITM 的 image-to-text R@1/R@5 为 84.5/96.2，加入 ITG 后为 85.4/97.0；text-to-image 从 67.2/87.1 升至 68.3/87.7。这说明生成目标不只是服务字幕，还迫使 query 保留更多与语言相关的视觉细节。

第一阶段表示学习是论文最关键的结构消融。若跳过 ITC/ITM/ITG，只在冻结视觉端与冻结 LLM 之间做生成对齐，OPT 和 FlanT5 的零样本 VQA 都显著更差；这直接支持“两阶段优于单阶段投影”的主张。论文没有系统消融 query 数量、cross-attention 插入频率和 188M 参数规模，因此这些选择的最优性未被完整证明。

### 4.3 指令式零样本生成

![](assets/BLIP-2%20-%20冻结底座的高效视觉语言桥接/figures/04-zero-shot-examples.png)

> **Figure 4，PDF 第 5 页。** 完整样例覆盖知识问答、常识推理、对话续问、食谱、地标识别、电影知识、个性化文案和动物对话。它展示了冻结 LLM 的语言能力可被视觉软提示调用；同时部分回答依赖世界知识而非纯视觉证据，为后续失败案例埋下伏笔。来源：https://arxiv.org/abs/2301.12597

---

## 5. 局限与失败案例

### 5.1 缺少多图上下文学习

BLIP-2 每个预训练样本只有一幅图与一段文本，因此没有学到多组图文对在同一序列中的对应关系。作者尝试在 VQA prompt 中加入 in-context 示例，并未观察到性能提升；这与 Flamingo 使用 M3W 交错网页数据形成鲜明对比。未来工作明确提出构建类似的多图交错数据集。

![](assets/BLIP-2%20-%20冻结底座的高效视觉语言桥接/figures/05-failure-cases.png)

> **Appendix Figure，PDF 第 12 页。** 左例把他人名言错误归给爱因斯坦，属于 LLM 知识错误；中例看到短裤却忽略“加拿大十二月”的天气条件，属于错误推理路径；右例把 iPhone 14 识别成 iPhone 11 并编造规格，属于知识过时与细粒度识别失败。来源：https://arxiv.org/abs/2301.12597

### 5.2 冻结底座的继承性风险与 future work

Q-Former 能降低灾难性遗忘，却不能修正冻结模型中的错误知识、社会偏差、冒犯性语言或隐私泄漏倾向。输出可能由语言先验主导，在视觉证据不足时仍给出自信答案。作者建议用指令约束生成、过滤训练数据，并继续使用更强视觉和语言底座；论文的 future work 还包括多图交错训练，以获得 Flamingo 式 in-context learning。

从任务形式看，32-query 固定瓶颈适合全局语义生成，但高分辨率 OCR、密集定位和多对象精确关系可能在压缩时丢失信息。论文没有专门量化 query 瓶颈在这些任务上的信息损失，也没有给出系统拒答或校准实验。

---

## 6. 与后续工作的关系

BLIP-2 继承了 CLIP 的冻结视觉表征与 Flamingo 的冻结 LLM 思路，但把深层跨注意力改为“视觉侧先对齐、语言侧只接软提示”。相比 Flamingo，它训练参数少、官方实现开放、单图任务复现成本更低；代价是缺乏原生多图/视频交错上下文和语言层内部的反复视觉读取。

LLaVA 几乎同时走向另一端：直接用 CLIP ViT-L/14 patch token 加线性投影接 Vicuna，再以视觉指令数据对 projector+LLM 做训练。LLaVA 结构更简单、对话风格更强，但会更新 LLM，且没有 Q-Former 第一阶段的显式 ITC/ITM/ITG 对齐。CogVLM 则认为浅软提示会形成容量瓶颈，为每层视觉 token 增加独立 QKV/FFN 专家，以深层融合换取 grounding 与 OCR 能力。

BLIP-2 的 Q-Former 此后成为高效多模态连接器的标准原型，被 InstructBLIP 等工作继续使用；其两阶段思想也给出一个清晰工程准则：当两端都冻结时，先学“视觉中哪些内容对语言有用”，再学“如何让具体 LLM 消费这些内容”，通常比一个损失同时解决两件事稳定。

---

## 7. 个人判断

BLIP-2 最有价值的不是某个 benchmark SOTA，而是把 VLM 对齐问题压缩成一个可替换、可训练、可诊断的中间模块。32 个 query 既是效率来源也是归纳偏置：它迫使模型舍弃像素冗余，只向 LLM 传递语言相关证据。第一阶段的多目标、多 mask 训练使这个瓶颈不只是线性投影，而是具备检索、生成和细粒度匹配能力的视觉语言接口。

复现层面，它比 Flamingo 现实得多：官方 LAVIS 可直接验证推理与微调，单机多卡可重训桥接器；但从头复现论文仍需 1.29 亿图像、CapFilt 字幕生成和 16×A100 多日训练。其能力边界也不应被“涌现”样例掩盖：单图配对数据无法自动产生多图上下文学习，冻结 LLM 的知识错误不会因视觉输入而消失。综合判断：BLIP-2 确立了冻结基础模型时代的高效桥接范式，但固定 query 瓶颈与浅层软提示也直接引出了后续视觉专家和深层融合路线。