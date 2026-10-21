---
type: paper-note
status: complete
domain: VLM
paper: "Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding"
year: 2026
arxiv: "2601.10611"
doi: null
source: https://arxiv.org/abs/2601.10611
project: null
code: https://github.com/allenai/molmo2
tags:
  - VLM
  - Video-LMM
  - Grounding
  - Pointing
  - Tracking
  - Counting
  - Multi-Image
  - Open-Data
created: 2026-10-16
updated: 2026-10-16
---

# Molmo2：视频与多图时空指向

## 基本信息

| 项目 | 内容 |
|---|---|
| 正式标题 | Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding |
| 作者 | Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park 等，Allen Institute for AI / University of Washington |
| 论文版本 | arXiv:2601.10611v4，2026-04-02 |
| 模型家族 | Molmo2-4B、Molmo2-8B（Qwen3）；Molmo2-O-7B（OLMo 3） |
| 视觉骨干 | SigLIP 2 So400m/14，约 380M，名义输入 384px |
| 核心能力 | 单图/多图/视频 QA、密集描述、计数、时空 pointing、开放词汇多目标 tracking |
| 上下文 | 常规 SFT 16,384；long-context SFT 36,864；最多 384 帧 |
| 开放性 | 开放权重、训练代码和多模态数据；不从闭源 VLM 蒸馏 |
| 本地原文 | [paper.pdf](assets/Molmo2%20-%20视频与多图时空指向/sources/paper.pdf) |
| 来源清单 | [sources.md](assets/Molmo2%20-%20视频与多图时空指向/sources.md) |

## 一句话结论

Molmo2 的突破不只是“把 Molmo 扩成视频模型”，而是把自然语言回答与**可解析的时空点序列**统一到同一个自回归输出空间：同一模型既能回答视频问题，也能在特定时间点指向对象、以一致 ID 跟踪它、通过“先指再数”完成计数；其性能主要来自规模化人类 grounding 数据、紧凑点格式、双向视觉注意力、任务平衡 loss 与高效 message-tree packing 的共同作用。

---

## 1. 研究背景与问题定义

### 1.1 为什么需要视频 grounding

普通视频 VLM 的输出通常是自由文本。它可以说“杯子掉下桌子了”，却不能告诉用户事件发生在第几秒、杯子在哪；它可以回答“有 16 头野牛”，却无法给出 16 个实例的视觉证据；它可以描述“白衣舞者向右移动”，却不能跨帧维持同一对象 ID。机器人、视频检索、体育分析和审计场景要求回答可定位、可验证，因此需要把空间 grounding 扩展到时间轴。

Molmo2 将任务拆成三种相互关联的输出：

1. **Video pointing：** 对对象、动作或事件输出代表性时间 $t$ 与空间点 $(x,y)$。
2. **Video tracking：** 对每个时间点输出对象 ID 与点坐标，同一对象跨帧复用 ID。
3. **Counting：** 先为每个独立实例输出点与顺序 ID，再从最大 ID/实例集合得到数量，而不是直接猜一个整数。

### 1.2 开放生态的缺口

当时最强视频 VLM 多为闭源；部分开源权重模型依赖闭源 VLM 生成训练数据，或不披露数据和 recipe。Molmo2 明确避免闭源 VLM 蒸馏，转而用人类标注、开放视觉模型、闭源**纯文本** LLM 辅助改写/生成问题。这不等于整个链路完全无闭源组件，但避免了“必须先有一个更强闭源视觉模型才能复现视觉监督”的循环依赖。

### 1.3 核心贡献

1. **九套新数据。** 覆盖密集视频 caption、长问答、多图 QA、视频/多图 pointing 与视频 tracking，并公开构建流程。
2. **统一时空点语言。** 用紧凑 HTML-like 文本编码时间、对象 ID 和归一化坐标，让普通 LLM 直接生成 grounding。
3. **可扩展 VLM 架构。** SigLIP 2 + attentional pooling connector + Qwen3/OLMo 3，单图、多图和视频共享参数。
4. **训练系统。** 双向视觉 attention、token weighting、online packing、message tree mask 和 Ulysses context parallel，使密集长输出与多任务训练可行。
5. **强 grounding 结果。** 在视频 pointing、tracking、counting 上显著超过同尺寸开放模型，部分指标超过闭源 API。

![](assets/Molmo2%20-%20视频与多图时空指向/figures/01-data-and-capabilities.png)

> 图 1：左侧展示 caption、AskModelAnything、pointing、tracking 与 synthetic 数据流水线，右侧对应细粒度理解、交互、时空定位、对象跟踪和视觉技能。Molmo2 的方法重心明显是“数据—输出格式—任务混合”共同设计，而非新增大型视频 encoder。来源：论文 Figure 1，第 2 页。

---

## 2. 任务定义与输入输出

### 2.1 输入模态

模型接受三类视觉输入：单图；2–5 张或更多有语义关系的多图集合；按时间采样的视频帧。图像训练最多使用 8 个重叠高分辨率 crop，推理最多 24 个；视频默认 2 FPS，常规最大 128 帧，long-context 阶段/主评测可达 384 帧。若原视频更长则均匀采样，并强制保留首尾帧；tracking 训练为保持标注对齐，按 2 FPS 截断而不是任意均匀重采样。

视频帧 token 与文本时间戳交错，时间写到 0.1 秒；多图则插入 `Image 1`、`Image 2` 等索引。所有视觉输入放在 BOS 之后、用户文本之前；视频/图像、多 crop/低分辨率 crop 使用不同起止 special token。

### 2.2 输出空间

自由文本任务输出普通自然语言。Grounding 使用紧凑格式，例如：

`<points coords="1 1 555 169;2 3 649 154 4 709 162">waterfall</points>`

`<tracks coords="0.0 1 635 522;0.5 1 606 490 2 511 124;1.0 2 515 164">object</tracks>`

字段依次包含图像索引或视频时间戳、对象 ID、$x$、$y$。坐标归一化到 $[0,1000]$；对象 ID 从 1 递增；点按时间/图像索引再按坐标排序。同一 ID 跨时间重复即构成 track，最大 ID 同时给出实例数。相比 JSON，这种压缩串显著减少输出 token。

### 2.3 Pointing、tracking、counting 的统一解释

对一组预测点可写成：$ \mathcal{P}=\{(t_i,k_i,x_i,y_i)\}_{i=1}^{M} $

其中 $t_i$ 是时间或图像索引，$k_i$ 是对象 ID，$(x_i,y_i)$ 是位置。Pointing 只需在关键时刻输出若干点；tracking 要求对同一 $k_i$ 在多个 $t_i$ 上持续输出；counting 使用唯一 ID 集合大小：$ \hat{c}=|\{k_i:(t_i,k_i,x_i,y_i)\in\mathcal{P}\}| $

这使三项能力共享监督。论文消融证明“point then count”比直接预测数字更好，说明计数提升来自显式实例分解，而非仅靠语言模型学习数字分布。

### 2.4 Token shape

SigLIP 2 的有效 crop 由附录描述为缩放到 378px，patch size 14，形成约 $27\times27=729$ 个 patch。视频使用 3×3 window attentional pooling，因此每帧约为 $9\times9=81$ 个视觉 token；Table 20 的 128 帧约 10.6K visual token 与此一致。图像使用 2×2 window，边缘不足窗口时缩小 pooling window。

| 阶段 | Shape | 说明 |
|---|---|---|
| 视频帧 | $378\times378\times3$ | 单 crop，名义 ViT 配置为 384px |
| ViT patch | $27\times27\times1152$ | SigLIP 2 patch features |
| 视频 pooling window | $9\times1152$ | 每个 3×3 patch window |
| 单帧视觉序列 | 约 $81\times D$ | 9×9 个 pooled token，投影到 LLM 维度 |
| $F$ 帧视觉序列 | 约 $81F\times D$ | 另加 frame start/end、时间戳等 token |
| 长上下文输入 | 最长 36,864 token | $F\le384$，视觉与文本/字幕共同占用窗口 |

4B LLM hidden size 为 2560，7B/8B 为 4096；connector 参数量约 57M/80M/88M。视觉 encoder 输出维度 1152，connector MLP 的输出维度与对应 LLM 对齐。

### 2.5 训练目标与 token weighting

基础目标仍是 assistant 输出位置的加权 next-token cross entropy：$ \mathcal{L}=-\sum_{e}w_e\sum_{i\in\mathrm{answer}(e)}\log p_\theta(y_i\mid x_e,y_{<i}) $

视频 caption 可超过 4000 输出 token，若按 token 等权，会淹没只有一个 token 的多选题。Molmo2 对视频 caption 固定 $w=0.1$，pointing 固定 $w=0.2$，其他任务按回答长度 $n$ 使用：$ w(n)=4/\sqrt{n} $

该权重不是让长任务不重要，而是控制每条样本对总梯度的量级。消融中去掉 weighting，QA 平均从 64.8 降至 64.0，caption F1 却从 39.5 升到 40.0，清楚体现多任务间的 Pareto trade-off。

---

## 3. 核心方法

### 3.1 Encoder–connector–LLM 总体架构

![](assets/Molmo2%20-%20视频与多图时空指向/figures/02-model-architecture.png)

> 图 2：视频帧经视觉 encoder 和共享 connector 变成视觉 token，与问题 token 一同输入 LLM；模型直接自回归生成带时间、对象 ID 和坐标的 `<points>` 文本。来源：论文 Figure 2，第 6 页。

模型使用 SigLIP 2 So400m/14 作为 ViT，取倒数第 3 层和倒数第 9 层特征。图像/视频都先编码 patch，再由共享 attentional pooling connector 聚合局部窗口，随后共享 MLP 映射到语言维度。4B/8B 版本接 Qwen3，O-7B 版本接 OLMo 3；所有参数在预训练阶段均可更新。

它没有采用专用视频 Transformer。不同帧的视觉交互发生在 LLM 中，而不是 ViT 中；因此 architecture 简洁，视频能力主要由帧序、时间 token、双向视觉 mask 和数据监督提供。

### 3.2 Connector 的 Q/K/V

对某个 pooling window $H_W=[h_1,\ldots,h_m]$，图像 $m=4$，视频 $m=9$。connector 先以窗口均值作为 query，patch 特征作为 key/value：$ q_W=\frac{1}{m}\sum_{j=1}^{m}h_j,\quad K_W=V_W=H_W $

多头注意力聚合后再过共享 MLP：$ z_W=\operatorname{MLP}(\operatorname{MHA}(q_W,K_W,V_W)) $

与固定平均池化相比，均值 query 提供稳定的窗口摘要，MHA 又能对窗口内细节做内容自适应加权。图像用 2×2、视频用 3×3 是精度与上下文预算的折中；3×3→4×4 消融对高层 QA 影响较小，却显著伤害密集 caption，说明细节任务更敏感。

### 3.3 LLM Q/K/V 与双向视觉 attention

进入 LLM 后，视觉 token 和文本 token 使用语言骨干原生的 Q/K/V 投影；Qwen3 版本继承其 grouped-query attention，OLMo 版本继承自身 attention 配置。Molmo2 没有添加独立 cross-attention 层，而是 early-fusion 地把视觉 embedding 当作序列 token。

标准 causal LM 中，前面的视觉 token不能看到后面的视觉 token。Molmo2 放宽 mask：所有图像 token 即使来自不同帧/不同图片，也允许 forward-attend，等价于在视觉子序列内提供双向信息；文本回答仍保持因果生成。这样早期帧 token 能汇聚后续帧证据，对 tracking、全局 caption 和跨图比较更自然。

### 3.4 Message tree 与 packing mask

一段视频通常对应多个问答或 grounding 标注。若把每个问答复制一遍视觉 token，成本极高；若简单串起来，后一个答案会看到前一个答案，造成信息泄漏。Molmo2 将视觉输入作为 message tree 根节点，各个 QA 作为独立分支，再线性化成一条序列并构造定制 attention mask。

![](assets/Molmo2%20-%20视频与多图时空指向/figures/03-packing-attention-mask.png)

> 图 3：深粉视觉 token 允许向前看；每个 QA 分支能看共享视觉根和自己历史，但不能看同一视觉样本的其它 QA 分支，也不能看 packed 序列中的其它样本。来源：论文 Figure 3，第 6 页。

设例子 $e$ 的视觉根为 $I_e$、第 $b$ 个文本分支为 $T_{e,b}$。Mask 允许 $I_e\leftrightarrow I_e$ 的双向视觉 attention，允许 $T_{e,b}$ 读取 $I_e$ 及本分支先前 token，但禁止 $T_{e,b}$ 与 $T_{e,b'}$（$b\ne b'$）互看，也禁止不同 $e$ 之间互看。论文采用 PyTorch SDPA 而非 FlashAttention，原因正是后者不支持这类定制 mask。

Online packing 从 48 个预处理样本池中用动态规划选择组合，优化 $T+30I$，约束 $T\le16{,}384$、crop 数 $I\le128$；long-context 阶段改为 36,864 token、384 images。平均每个视觉样本有 4 个标注，每条 packed sequence 装入 3.8 个例子，论文报告相对朴素方案约 15× 训练效率提升。

### 3.5 Position、时间与多图索引

LLM 沿用 RoPE；4B/8B 的 theta 为 $10^6$，O-7B 为 $5\times10^5$，LayerNorm 与 RoPE 以全精度计算。更关键的是显式文本 time token：每帧视觉块旁插入以秒为单位、保留 1 位小数的时间戳；多图则插入 `Image i`。这把真实时间和一维 RoPE 序号分开：RoPE 表示 token 顺序，文本 timestamp 表示采样对应的物理时间。

消融去掉 time tokens，QA 平均从 64.8 降到 64.5，caption F1 从 39.5 降到 37.4。对长描述的影响更大，说明仅靠 frame order 不足以稳定表达持续时间和事件位置。

### 3.6 Pointing、tracking 与 counting 的训练技巧

VideoPoint 最多允许每条样本 60 个点，并构造同一视频上的多轮 pointing/counting 对话。Tracking 除完整轨迹外，还有两个辅助任务：预测对象首次/末次出现帧；给定 query 与初始点后继续 tracking。对象 ID 让模型同时学习检测与关联。

数据混合也针对长尾重加权：视频 tracking 强调稀有概念；视频 pointing 上采样中高计数样本。消融显示 AcademicVideoPoint 与 Molmo2-VideoPoint 互补；pointing 数据加入 tracking 后，J&F/F1/HOTA 从 64.9/70.0/68.4 提升到 65.7/71.1/69.4。

---

## 4. 数据工程与训练

### 4.1 九套新数据的构成

| 数据集 | 类型与规模 | 关键构建方式 |
|---|---|---|
| Molmo2-Cap | 人工；104K 整视频、431K clip caption | 人先口述 clip，Whisper 转写，文本 LLM 改写；再融合 Molmo 帧级 caption；平均 924 words/视频 |
| Molmo2-AskModelAnything | 人工；140K QA | 人写细粒度问题，文本 LLM 基于早期 Molmo2 caption 给初答，人机迭代修订 |
| Molmo2-CapQA | 合成；1M QA / 200K 视频 | 自有 captioner 分场景描述，文本 LLM 基于 caption/metadata 生成问题 |
| Molmo2-SubtitleQA | 合成；300K QA / 100K 视频 | 加入 Whisper transcript，要求联合视觉与语言推理 |
| Molmo2-VideoPoint | 人工；650K queries / 280K 视频 | 2 FPS 选帧并点击精确位置，覆盖对象、动作、指代表达、异常等 8 类 |
| Molmo2-VideoTrack | 人工；3.6K clips、15K query | 基于开放 mask/bbox track，让标注员写复杂、多对象自然语言指代；平均 2.28 对象/query |
| Molmo2-MultiImageQA | 人工；72K QA、45K 图集 | 按 caption 相似度组成 2–5 图集合，人写跨图问题并修订答案 |
| Molmo2-MultiImagePoint | 合成；470K+ | 对 PixMo-Points 软聚类、同义归一并保留词汇多样性 |
| Molmo2-SynMultiImageQA | 合成；188K | 适配 CoSyn，侧重图表、表格、文档等文字密集多图 |

此外，AcademicVideoPoint 把 6 个开放 tracking 数据集转成约 49K pointing/counting QA；AcademicVideoTrack 复用 7 个 Ref-VOS 和 11 个 bbox tracking 数据源，并以 SAM 2 生成 mask/point 监督。

数据设计最值得借鉴的是“用结构化标注转任务”：现成 mask/bbox 不直接要求模型输出昂贵分割，而是抽样为点；同一对象的 point track 又可派生首次/末次、计数和关联任务。这样一个自回归接口吸收了检测、分割、跟踪数据。

### 4.2 三阶段训练

| 阶段 | 数据/目标 | Steps / Batch | Sequence / Frames |
|---|---|---|---|
| 1. Image-only pre-training | 60% caption、30% image pointing、10% Tulu NLP；所有参数训练 | 32K / 128 | 2,560 |
| 2. Joint SFT | 图像、视频、多图、NLP、pointing、tracking 联合 | 30K / 128 | 16,384 / 最多 128 帧 |
| 3. Long-context SFT | 同一 SFT mixture，短阶段扩长 | 2K / 128 | 36,864 / 最多 384 帧；CP=8 |

Long-context 使用 Ulysses attention，因为 all-gather 便于支持 message-tree custom mask；ViT 帧处理与 connector pooling 也分布到 context parallel group。论文只做 2K steps，因为长上下文开销显著。

### 4.3 SFT mixture

类别采样率为：Image QA 22.7%、Video QA 18.2%、Captions/Long QA 13.6%、Video Pointing 13.6%、Video Tracking 13.6%、Image Pointing 9.1%、NLP 9.1%。类别内按数据集规模平方根采样，并人工下采样大规模合成集、上采样收敛慢的 grounding 与长尾样本。

这种 mixture 同时应对三种不平衡：数据集规模不平衡、输出长度不平衡、概念频率不平衡。若只解决第一种，长 caption 仍会从 loss token 数上主导；若只做 token weighting，稀有 tracking 概念仍可能看不到。

### 4.4 优化与算力

预训练对 ViT/connector/LLM 使用独立学习率：$6\times10^{-6}$、$2\times10^{-4}$、$2\times10^{-4}$；SFT 为 $5\times10^{-6}$、$5\times10^{-6}$、$10^{-5}$。AdamW betas=(0.9,0.95)、epsilon=$10^{-6}$、cosine decay 到峰值 10%、无 weight decay。实现使用 FSDP2、SDPA、`torch.compile`、bf16 AMP。

8B 模型的预训练/SFT/long-context 分别约 780/8100/3300 H100 GPU-hours；4B 约 490/7500/3200。SFT 计算很大一部分来自 ViT，因为视频中每个 LLM visual token 对应 9 个 ViT patch，增大 LLM 对总时长的影响被视觉开销部分抵消。

---

## 5. 实验结果

### 5.1 综合视频表现与 baseline

Molmo2-8B 在 Table 2 的 12 项视频 QA、caption、counting 综合 average 为 63.1，优于 Qwen3-VL-8B 的 59.5、Eagle2.5-8B 的 60.7、MiniCPM-V-4.5-8B 的 56.6，也明显超过完全开放的 PLM-8B 56.2。它在短视频平均、密集 caption 和计数上尤其强；长视频仍落后最强闭源 API，并受公开 10+ 分钟训练数据不足与 ultra-long 训练成本限制。

推理通常使用 384 帧、greedy decoding；caption/人评改用 top-p 0.95、temperature 0.7、frequency penalty 0.1。Long-context SFT 将 long-video QA 平均从 64.4 提升到 67.4，但 short QA 69.6→69.4、caption F1 42.3→39.9，说明扩长阶段提升长视频的同时存在 caption 干扰。

### 5.2 Counting 与 spatio-temporal pointing

![](assets/Molmo2%20-%20视频与多图时空指向/figures/04-counting-pointing-results.png)

> 表 3：Molmo2 在 BURST 视频计数和 Molmo2 自建计数/pointing 上显著领先。Molmo2-8B 的计数准确率 35.5，高于 Qwen3-VL-8B 的 29.6；pointing F1 为 38.4，高于 Gemini 3 Pro 的 20.0 与 Qwen3-VL-8B 的 1.5。来源：论文 Table 3，第 9 页。

计数还报告 close accuracy：当 $|\hat{c}-c|\le1+\lfloor0.05c\rfloor$ 时算接近正确。Molmo2 在 BURST-VideoCount exact accuracy 达约 61%，说明它对实例集合的分解能力比直接计数 baseline 更稳。对高计数 25–60 的样本，论文附录指出 Qwen3-VL 为 0%，Molmo2 仍超过 10%。

Pointing 评测以 2 FPS、最多 384 帧输入，在标注点附近 3 秒窗口内用 SAM 2 生成并人工过滤 mask；预测点落入目标 mask 才算匹配。这个指标比单纯比较坐标距离更能容忍对象大小差异，但自建验证集只有 181 例，结果仍需更大公开基准复核。

### 5.3 Tracking

Tracking 在 MeViS、Ref-YT-VOS、Ref-DAVIS、ReasonVOS 和 Molmo2-Track 上评测。预测 point 经 SAM 2 转为 segmentation mask 后计算 J&F；同时报告 1 FPS point F1 和适配点匹配的 HOTA，后者把检测正确性与 ID association 分开衡量。

论文摘要给出的代表结果是 Molmo2 tracking J&F 56.2，对比 Gemini 3 Pro 41.1。作者称 Molmo2 在全部 tracking benchmark 上领先 API、VideoMolmo 及专用 segmentation baseline，特别是在需要复杂指代、遮挡和推理的 ReasonVOS/Molmo2-Track 上优势明显。需要注意：Molmo2 输出的是点，J&F 依赖 SAM 2 后处理，故该数字同时反映 VLM 点定位和分割器质量。

### 5.4 架构、数据与计数消融

![](assets/Molmo2%20-%20视频与多图时空指向/figures/05-model-and-grounding-ablations.png)

> 表 8–9：去掉双向视觉 attention、token weighting、time token 都会降低 QA；把视频 pooling 从 3×3 改成 4×4 对 caption 伤害最大。Point-then-count 将 Molmo2-VideoCount 从 28.1 提到 34.5；组合人工与 academic pointing、上采样中高计数样本也带来互补增益。来源：论文 Tables 8–9，第 13 页。

关键消融可归纳为：

- **No bidir：** QA 64.8→64.4，caption F1 39.5→38.5，视觉 token 的跨帧双向读取有效。
- **No token weighting：** QA 64.8→64.0，caption 39.5→40.0，证明 weighting 在多任务平衡上有利，但牺牲少量 caption 专项最优。
- **No time tokens：** QA 64.8→64.5，caption 39.5→37.4，真实时间表达对密集叙述尤为关键。
- **Pool 3×3→4×4：** QA 64.8→64.3，caption 39.5→37.0，过强空间压缩先伤害细粒度描述。
- **Count→point-then-count：** BURST 61.3→61.5，Molmo2-VideoCount 28.1→34.5，实例定位是计数主因。
- **中高频点上采样：** Molmo2-VideoCount 32.1→34.5，VideoPoint F1 28.1→31.8，直接改善长尾计数/pointing。

### 5.5 定性结果与失败案例

![](assets/Molmo2%20-%20视频与多图时空指向/figures/06-pointing-qualitative.png)

> 图 35：模型可在多帧中为高速公路车辆分配多个点，也能回答需要 OCR/字幕的腌制时间、识别比赛获胜队伍，并定位生成视频中的视觉伪影。来源：论文 Figure 35，第 57 页。

![](assets/Molmo2%20-%20视频与多图时空指向/figures/07-failure-cases.png)

> 图 36：模型会在杂乱桌面和瀑布视频中产生 pointing 假阳性，也会在企鹅 tracking 中漏掉实例。错误集中体现了高密度对象、跨帧 re-identification 与长输出退化。来源：论文 Figure 36，第 58 页。

---

## 6. 与相关工作的关系

相对 Molmo1，Molmo2 延续“直接生成 point 文本”的接口，但扩展到多图、时间戳、对象 ID 和 track；相对 Qwen3-VL 等支持 box/point 的 VLM，Molmo2 的区别是大规模时空 grounding 数据和显式 point-then-count 训练；相对 SAM 2 等专用跟踪器，它能接受开放词汇复杂自然语言 query、一次指代多个对象并同时回答问题，但空间精度依赖点和后处理，不等价于原生 mask decoder。

从架构看，它与“ViT + connector + decoder-only LLM”主流范式一致，创新主要落在 connector pooling、attention mask、输出协议与数据工程。论文因此是一个很典型的结论：在基础架构已稳定时，能力边界往往由**数据表示与训练系统**决定。

---

## 7. 局限与批判性评价

1. **仍依赖闭数据视觉 encoder。** 即便 OLMo 3 语言骨干开放，SigLIP 2 的训练数据不开放，严格意义上并非全栈开放数据模型。
2. **使用闭源文本 LLM。** 问答生成、答案改写和过滤使用闭源 text-only LLM，虽未蒸馏闭源视觉能力，复现透明度仍受影响。
3. **视频 grounding 绝对值仍低。** 最强 pointing/计数指标不到 40%，显著低于图像 PointBench 常见的 70%–90%；论文明确承认视频 re-identification 与长序列更难。
4. **长视频 grounding 受限于约 3 分钟。** 标注固定 2 FPS，超过训练长度若降低采样率会使标注与帧错位；当前方案没有解决长视频的自适应证据帧保留。
5. **点序列会退化。** 高频对象或长视频中会重复同一坐标、在单帧画长点串；联合多任务的 interference 可能是原因之一。
6. **Point tracking 不稳定。** 同一对象上的预测点会在物体内部漂移，源于由 bbox/mask 抽点时没有强制跨帧位置一致。
7. **J&F 依赖 SAM 2。** VLM 输出点经过外部分割器才能与 mask baseline 比较，端到端能力和组件贡献应分开解释。
8. **Caption 超长生成会重复。** Greedy decoding 在数千 token 后可能出现重复文本，说明训练长度和解码策略仍不足。
9. **自建 grounding 验证集较小。** Molmo2-VideoPointVal 仅 181 个过滤样本，可能难覆盖开放词汇与复杂运动分布。
10. **Baseline 格式公平性有限。** API 模型无法稳定输出论文点格式，作者改用 box 再取中心；不同模型的输出接口和 prompt 适配不完全等价。

---

## 8. 复现与实践建议

### 8.1 最小复现路径

1. 先使用官方 Molmo2-4B 配置验证单图 pointing 格式，检查 tokenizer 对数字、分号、标签的 tokenization。
2. 接入 SigLIP 2 倒数第 3/第 9 层特征，单图用 2×2、视频用 3×3 attentional pooling；断言视频约 81 visual token/帧。
3. 实现视觉双向、文本因果、分支隔离、样本隔离四种 mask 关系，并用 Figure 3 的两样本两分支 toy case 做单元测试。
4. 先训练/微调 pointing，再做 point-then-count；若直接训练纯整数计数，很难复现 Table 9 增益。
5. Tracking 中检查同一对象 ID 跨帧一致性；评测同时报告 point F1、HOTA 和经 SAM 2 的 J&F，避免只报后处理后的 mask 分数。
6. 最后启用 36,864 context、384 帧与 Ulysses CP=8；先确认时间戳和 frame sampling 对齐，再扩长。

### 8.2 关键 shape、QKV、mask、position 检查

- **Encoder：** 378/14→27×27 patch；视频 3×3 pooling 不得跨帧窗口。
- **Connector QKV：** 每个窗口的 mean patch 作 query，窗口 patch 作 key/value；图像和视频共享参数但窗口大小不同。
- **LLM QKV：** 使用底座原生投影；不要额外插入 cross-attention，否则不再是论文结构。
- **Mask：** 视觉 token 双向；文本严格 causal；不同 message-tree 分支互斥；packed 样本 block-diagonal。
- **Position：** RoPE 负责序号，文本 timestamp 负责真实时间；可变 FPS 必须按 timestamp 抽帧而非 frame index。
- **Loss：** 只对目标输出计 loss，并应用任务/长度权重；跨 GPU 按全局平均 loss-token 数归一化，避免短回答被隐式放大。

### 8.3 数据复现重点

数据最难复现的不是下载，而是保持“证据—问题—输出格式”一致：VideoPoint 点击帧必须与 2 FPS 时间戳一致；由 mask/bbox 抽点时应控制点位稳定性；tracking query 需覆盖对象子集而非简单类别名；高计数样本必须上采样；caption 的人类口述、ASR、帧级补充和文本改写每一步都要保留溯源。

若资源有限，优先复现三条结论：point-then-count、双向视觉 attention、message-tree packing。它们分别代表任务表示、模型 mask 和训练系统，能够覆盖论文最核心的三类贡献。

### 8.4 资源评估

完整训练需要 32–128 张 H100 级 GPU，各阶段总计上万 GPU-hours；长上下文单独约 3.2K–3.3K H100 GPU-hours。官方开放模型与代码大幅降低推理/微调门槛，但从头复现数据和全训练仍是机构级项目。实用路线是基于公开 checkpoint 做领域 pointing/tracking SFT，而不是重做 32K+30K+2K steps。

---

## 9. Future Work 与个人启发

### 9.1 后续研究方向

1. **长视频 evidence routing。** 先低帧率检索候选时间段，再以 2 FPS/高分辨率重编码，避免 grounding 标注因均匀降采样而错位。
2. **一致点轨迹生成。** 从 mask/bbox 中选择物体归一化位置一致的点，或预测对象中心坐标系内的相对点，降低 track 漂移。
3. **结构化 decoder。** 保持语言模型开放词汇优势，同时用 constrained decoding/grammar 保证 `<points>` 可解析、ID 不重复、时间单调。
4. **Grounded chain-of-thought。** 中间推理步骤必须引用时间点和视觉点，让答案可审计，而不是只在最终输出附证据。
5. **开放视觉 encoder。** 用完全公开数据训练可竞争 SigLIP 2 的 ViT，补齐“open weights/data/code”中最后的闭数据组件。
6. **多任务冲突控制。** 对 caption、QA、pointing、tracking 使用梯度冲突分析或模块化 adapter，减少长点串和重复 caption 等 interference。

### 9.2 个人判断

Molmo2 最值得迁移的思想是：**把 perception task 重新编码成语言模型可生成、可共享的数据结构。**一个带 $t$、ID、$x$、$y$ 的点序列同时服务 pointing、tracking 和 counting，使开放词汇 VLM 吸收了传统检测/跟踪数据；随后真正困难的问题从“设计另一个视觉头”转成“如何构造稳定数据、平衡 loss、保证 mask 不泄漏”。这比单纯扩大模型更可能成为可复用的视频智能基础设施。

---

## 10. 材料来源

- 官方论文：[arXiv:2601.10611](https://arxiv.org/abs/2601.10611)
- 官方代码：[allenai/molmo2](https://github.com/allenai/molmo2)
- 本地 PDF：[assets/Molmo2 - 视频与多图时空指向/sources/paper.pdf](assets/Molmo2%20-%20视频与多图时空指向/sources/paper.pdf)
- 完整来源与图片溯源：[sources.md](assets/Molmo2%20-%20视频与多图时空指向/sources.md)
