---
type: paper-note
status: done
domain: VLM
paper: "Kosmos-2: Grounding Multimodal Large Language Models to the World"
year: 2023
arxiv: 2306.14824
doi: null
source: https://arxiv.org/abs/2306.14824
project: https://aka.ms/kosmos-2
code: https://aka.ms/kosmos-2
tags:
  - VLM
  - grounding
  - location-tokens
  - GrIT
  - causal-language-model
created: 2026-10-16
updated: 2026-10-16
---

# Kosmos-2：语言与区域统一 Grounding

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | Kosmos-2: Grounding Multimodal Large Language Models to the World |
| 作者 | Zhiliang Peng、Wenhui Wang、Li Dong、Yaru Hao、Shaohan Huang、Shuming Ma、Furu Wei |
| 机构 | Microsoft Research |
| arXiv | 2306.14824v3，2023-07-13 |
| 官方代码/权重入口 | https://aka.ms/kosmos-2 |
| 参数规模 | 约 1.6B；24 层 2048 宽 Magneto causal Transformer + CLIP ViT-L/14 视觉端 |
| 一句话结论 | Kosmos-2 把框坐标离散成 1,024 个 location token，并用 Markdown 式 span–box 链接与 91M 图像规模 GrIT 数据训练，使同一个 causal LM 能双向完成“文字→框”和“框→文字”，将 grounding 变成语言生成问题。 |

---

## 1. 研究背景与核心问题

### 1.1 文本回答不等于视觉 grounding

Flamingo、BLIP-2、Kosmos-1 等模型能够看图并生成文本，但输出中的名词短语通常没有可验证的空间指向。模型说“篝火旁有雪人”，用户无法知道它依据了哪个区域；反过来，用户指一个框，模型也缺少统一接口理解“这个区域”。传统检测器能输出框，却依赖固定类别、object query、proposal 与专用 head，不容易与开放文本生成和 in-context learning 统一。

Kosmos-2 的核心问题是：能否把二维坐标变成语言模型词表中的离散符号，使框和词一样被同一 causal LM 读取、预测和组合？如果可行，phrase grounding、referring expression comprehension、referring expression generation、grounded caption/VQA 都可写成 next-token prediction。

### 1.2 核心贡献

1. 引入 1,024 个 location token，把二维网格位置离散进词表；一个框由左上角与右下角两个 location token 表示。
2. 用 `<p>span</p><box><loc_i><loc_j></box>` 形成 Markdown/hyperlink 式文本—区域绑定。
3. 从 COYO-700M 与 LAION-2B 子集自动构建 GrIT，得到约 91M 图像、115M text span、137M bounding box。
4. 继承 Kosmos-1 的 causal MLLM 与多模态数据，在不加检测 head 的情况下生成坐标，并保留普通语言/视觉语言能力。

![](assets/Kosmos-2%20-%20语言与区域统一Grounding/figures/01-overview.png)

> **Figure 1，PDF 第 1 页。** 上半展示模型把 `[a campfire]` 生成为一对 location token；下半展示把 `[It](locs) sits next to ...` 作为输入理解用户所指区域。Grounding 与 referring 由同一词表和 next-token 模型双向实现。来源：https://arxiv.org/abs/2306.14824

---

## 2. Location tokens：从坐标到语言序列

### 2.1 32×32 空间量化

对宽 $W$、高 $H$ 的图像，将两个轴各均分为 $P=32$ 段，得到 $P^2=1024$ 个二维 bin，每个 bin 对应一个 `<loc_k>`。在论文 $224\times224$ 输入下，每个 bin 约为 $7\times7$ 像素。连续点 $(x,y)$ 量化为二维 bin 后再展平为词表索引：

$$k=\operatorname{clip}\!\left(\left\lfloor\frac{yP}{H}\right\rfloor,0,P-1\right)P+\operatorname{clip}\!\left(\left\lfloor\frac{xP}{W}\right\rfloor,0,P-1\right)$$

一个框 $b=(x_1,y_1,x_2,y_2)$ 不输出四个独立坐标 token，而是输出左上角 bin 与右下角 bin：

$$b\mapsto\texttt{<box><loc}_{k_1}\texttt{><loc}_{k_2}\texttt{></box>}$$

若一个短语对应多个框，在 `<box>...</box>` 内用 `<delim>` 分隔多组角点。由于 1024 个 location token 与普通 subword 共用 embedding 表和 softmax head，预测框就是预测两个离散词；无效序列如 `<box><loc_1></box>` 在评测时直接记错。

### 2.2 Markdown 式 span–box 链接

单框文本 span 写作：

`<p> a campfire </p><box><loc_4><loc_1007></box>`

`<p>...</p>` 标识语言短语，后面的 `<box>...</box>` 是其视觉地址，`<grounding>` 告诉模型需要输出 grounding。完整序列可混合 `<s>`、`<image>64 个视觉 embedding</image>`、自然语言和 location token。这个表示的关键不是表面语法，而是让同一个短语—位置关系可正向或反向训练：给短语生成框，或给框生成指代表达。

---

## 3. GrIT：Grounded Image-Text 数据构造

### 3.1 Step 1：noun chunk 到检测框

从 COYO-700M 与 LAION-2B 的图文对出发，spaCy 从 caption 抽取 noun chunks，并去掉 `time`、`love`、`freedom` 等难视觉验证的抽象名词。随后用预训练 grounding detector（论文举例 GLIP）以 noun chunk 查询图像，保留置信度大于 0.65 的框，并跨短语做 NMS；没有任何保留框的样本被丢弃。

### 3.2 Step 2：扩展为 referring expression

仅有 `dog` 这类 noun chunk 不足以学习复杂指代。作者沿 spaCy dependency tree 递归加入子节点，把 `a dog` 扩成 `a dog in a field of flowers`；含 conjunct 的 chunk 不扩展。若多个表达存在包含关系，只保留最长、不被其他表达包含的一个，并把原 noun chunk 的框赋给扩展表达。

![](assets/Kosmos-2%20-%20语言与区域统一Grounding/figures/02-grit-pipeline.png)

> **Figure 3，PDF 第 4 页。** 上路由依存关系扩展短语，下路由用 grounding detector 产生框，最后组合并删除被长表达包含的 substring。图也暴露了噪声链：parser、detector、NMS 和“把 noun 的框赋给长表达”任何一步都可能错。来源：https://arxiv.org/abs/2306.14824

最终 GrIT 有 90,614,680 张图、114,978,233 个 text span、137,349,210 个框，平均表达长度 4.7；规模远超 Flickr30K Entities、RefCOCO/+/g 和 Visual Genome。它的优势是规模，弱点是伪标签质量和依赖预训练 detector，模型可能蒸馏其漏检与类别偏差。

---

## 4. Encoder、decoder、connector、QKV、mask 与 position

### 4.1 继承 Kosmos-1 的主干

Kosmos-2 明确沿用 Kosmos-1 架构与目标。视觉 encoder 是预训练 CLIP ViT-L/14：24 层、hidden 1024、FFN 4096，输入 $224^2$，得到约 $16\times16=256$ patch 级视觉特征（外加实现中的特殊 token）。训练时除最后一层外冻结 CLIP。connector 是 Flamingo 风格 Resampler，将可变/较长视觉网格压成固定 64 个 image embedding，再匹配到语言模型 2048 hidden width。

语言 decoder 是 24 层 Magneto Transformer，hidden 2048、FFN 8192、32 attention heads，因此每头维度为 64；总可训练规模约 1.6B。Magneto 在 self-attention 和 FFN 子层增加额外 LayerNorm，并使用专门初始化提升深网训练稳定性。

### 4.2 token shape 与 QKV

主路径可写为：CLIP 输出 $X_v\in\mathbb{R}^{B\times N_v\times1024}$；Resampler 用 64 个 latent query 读取视觉特征，输出 $Z_v\in\mathbb{R}^{B\times64\times2048}$；文本与 location token 通过 lookup 得到 $E_t\in\mathbb{R}^{B\times L\times2048}$；最终 `[text, <image>, Z_v, </image>, text/location]` 被展平成一个长度不超过 2048 的序列。

Resampler 中 latent 是 $Q$，视觉 token 是主要 $K,V$；causal LM 的每层 self-attention 则对已出现的文本、视觉 embedding、location token 统一计算 QKV，不再设置独立视觉 cross-attention。也就是说，视觉 connector 只在输入前压缩一次，后续 24 层把图像 token 当作序列前缀反复自注意。

### 4.3 attention mask 与 position

语言主干使用严格左到右 causal mask：位置 $t$ 可看所有过去的文本、图像 embedding 和 location token，不能看未来输出。训练损失只落在离散 token（文本与 location token）上，不要求重建连续 image embedding。模型采用 xPos 相对位置编码，最大序列长度 2048；视觉 ViT 保留自身二维 patch 位置表示。location token 的“空间位置”不是由 xPos 表达，而是由其词 ID 的 32×32 网格语义表达，两类 position 不应混淆。

---

## 5. 训练目标、数据混合与生成方式

### 5.1 统一 next-token 目标

对于混合序列 $u_{1:T}$，只在离散 token 集合 $\mathcal{D}$ 上计损失：

$$\mathcal{L}_{\mathrm{NTP}}=-\sum_{t\in\mathcal{D}}\log p_\theta(u_t\mid u_{<t})$$

训练数据包括 Kosmos-1 的纯文本、普通 image-caption、interleaved image-text，再加入 GrIT grounded pairs。每 step 约 419K token：185K 文本、215K 普通+grounded image-caption、19K interleaved；训练 60K step，约 25B token。

### 5.2 训练配置与 instruction tuning

预训练使用 AdamW，$\beta=(0.9,0.98)$、weight decay 0.01、dropout 0.1；375 step warmup 到 $2\times10^{-4}$，之后线性衰减。使用 256×V100，约 1 天，所有参数更新；新增 1,024 个 location embedding 随机初始化，其余从 Kosmos-1 初始化。

之后做 10K step instruction tuning，混合 LLaVA-Instruct、Unnatural Instructions、FLANv2、原训练数据及 GrIT 派生 grounding 指令，学习“短语→框”和“框→表达”。该阶段学习率 $10^{-5}$，仍更新全模型。

### 5.3 生成方式

phrase grounding 与 referring comprehension 让模型自回归生成 `<box><loc_i><loc_j></box>`，再反量化为框；无效语法直接判错。referring expression generation 给框 token，使用 greedy decoding 生成描述。普通 Flickr30K caption 使用 beam size 5，VQAv2 使用 greedy decoding。统一格式还允许把少量 `(框,描述)` demonstration 放入上下文做 few-shot referring。

![](assets/Kosmos-2%20-%20语言与区域统一Grounding/figures/03-grounding-format.png)

> **Figure 4，PDF 第 7 页。** phrase grounding 输入整句前缀并标记当前 phrase，以减少多对象歧义；referring expression comprehension 只给被 `<p>` 标记的表达。两者都让模型生成同一 `<box><loc_i><loc_j></box>` 格式。来源：https://arxiv.org/abs/2306.14824

---

## 6. 定量结果、baseline 与消融缺口

### 6.1 Grounding 与 referring

| 任务 | Kosmos-2 | 代表 baseline | 解读 |
|---|---:|---:|---|
| Flickr30K Entities phrase grounding test R@1 | **78.7** zero-shot | GRILL 18.9 zero-shot；VisualBERT 71.3 finetuned；GLIP 87.1 finetuned | 超过早期 zero-shot 与部分 finetuned 模型，但落后专用强 detector |
| RefCOCOg referring comprehension test | **61.65** zero-shot | GRILL 47.5 zero-shot；FIBER 87.32 finetuned | 长表达迁移较强，距专用监督模型仍远 |
| RefCOCO val / RefCOCO+ val | 52.32 / 45.48 | FIBER 90.68 / 85.74 finetuned | 对游戏式短语、方位词和域差异敏感 |
| RefCOCOg referring generation CIDEr | 60.3 zero-shot | SLR 59.2 finetuned | 已超过 SLR 1.1；4-shot 为 62.3 |

phrase grounding 的 R@1/R@5/R@10 接近，是因为 Kosmos-2 不依赖 proposal 集合，通常直接生成少量位置；这既减少后处理，也限制了通过多 proposal 提高 recall 的空间。

### 6.2 普通视觉语言与语言能力

未做 instruction tuning 的公平比较中，Flickr30K zero-shot caption CIDEr 从 Kosmos-1 的 65.2 升至 66.7，但 VQAv2 从 46.7 降至 45.6。语言任务也并非全面保持：BoolQ 56.4→62.0、COPA 63.0→67.0，但 CB 44.6→30.4。新增 grounding 能力没有完全免费，数据混合会重新分配容量。

### 6.3 消融实验的真实情况

论文**没有提供标准结构消融表**：没有分别去掉 GrIT、location token、`<grounding>`、dependency expansion、NMS，亦没有比较 16×16/32×32/64×64 坐标量化。因此无法从原论文定量拆分“数据规模”“坐标词表”“初始化于 Kosmos-1”各自贡献。

最接近受控对照的是 Kosmos-1→Kosmos-2：架构基本不变、加入 GrIT 与 location token 后出现 grounding 能力，同时普通 VLM/语言指标有升有降。但这同时改变数据和词表，不能当作严格单变量消融。缺少真正消融本身是本论文实验上的重要局限。

---

## 7. 与 CLIP、Flamingo、BLIP-2、LLaVA 的关系

- **CLIP**：Kosmos-2 直接以 CLIP ViT-L/14 作为视觉 encoder，但不保留 CLIP 的双塔检索接口。CLIP 输出经 Resampler 变为语言序列前缀；grounding 精度受 CLIP 空间表征与 32×32 量化共同限制。
- **Flamingo**：Resampler 思路来自 Flamingo，但 Flamingo 在 LM 深层用 gated cross-attention 读取视觉 token；Kosmos-2 把 64 个 image embedding 直接嵌入 causal 序列，后续使用统一 self-attention。Flamingo 强在多图 interleaved few-shot，Kosmos-2 新增结构化区域 token。
- **BLIP-2**：两者都用固定数量视觉 token 连接冻结/强视觉端与语言模型。BLIP-2 的 Q-Former 有显式 ITC/ITM/ITG 两阶段对齐并通常冻结 LLM；Kosmos-2 的 Resampler 更接近通用压缩器，全模型以 next-token 目标训练，且把位置也纳入词表。
- **LLaVA**：Kosmos-2 instruction tuning 混入 LLaVA-Instruct，但架构不是 LLaVA。LLaVA 用 CLIP patch + 线性 projector + Vicuna，主要输出文本；Kosmos-2 用 Resampler + Magneto causal LM，并可生成合法框 token。后者 grounding 接口更统一，前者开放对话生态和语言规模更强。

---

## 8. 局限、future work 与批判性评价

### 8.1 明确局限与失败来源

32×32 网格把连续坐标量化到约 7 像素 bin，两个角点的误差会放大小物体 IoU 波动；输出又必须满足严格语法，任一 token 错误都会使整个框无效。RefCOCO/RefCOCO+ 明显低于 finetuned detector，论文归因于这些数据常用 `left bottom` 等短表达，而 GrIT 平均表达长 4.7，训练分布不匹配。

GrIT 是 detector 和 parser 生成的伪标注：置信度 0.65、NMS、抽象词过滤、依存扩展和 substring 删除都是启发式。把 noun chunk 的框赋给扩展表达可能引入语义范围错误；使用 GLIP 类 detector 又会继承其类别和域偏差。论文没有测量 GrIT 标注精度、噪声对模型的影响或 location 分辨率消融。

模型最大上下文 2048，视觉前缀固定 64 token；多图、长文档和高分辨率密集 grounding 受限。论文的伦理声明只给原则性限制，未提供系统公平性、幻觉、拒答或有害 grounding 审计。

### 8.2 Future work

论文正文明确提出的 future goal 是增强 MLLM 对更多类型人类表达的准确理解，尤其缓解 RefCOCO/RefCOCO+ 短表达分布差异。结论还把 grounding 指向 embodied AI、语言—感知—动作—世界模型的融合，但这属于愿景，不是已验证的行动能力。

合理但属于本文判断的后续方向包括：连续/多尺度坐标编码；语法约束解码；更高质量伪标注与人工抽检；加入 segmentation/polygon；对 location vocabulary、GrIT 规模和 detector 选择做严格消融；将 grounding 置信度与拒答纳入评测。

### 8.3 个人判断

Kosmos-2 的核心突破是表示层而非模型规模：只要框能成为 token，定位就能复用 LLM 已有的序列建模、prompt、few-shot 和双向生成能力。它把专用检测 head 的结构归纳偏置换成了数据归纳偏置——大量 GrIT 教模型 location token 的语义。优点是接口统一，缺点是精度、合法性和数据噪声都压在自回归序列上。它更像“grounded language model 的最小可行证明”，不是专用检测器的替代品。

---

## 9. 复现成本与实践建议

官方提供代码与 checkpoint，推理和小规模指令微调可复现。完整训练仍需 Kosmos-1 初始化、约 25B token、91M 级 GrIT、256×V100 约 1 天；而重建 GrIT 还需要抓取 COYO/LAION 子集、spaCy、grounding detector、NMS 和大规模存储。论文只公开 GrIT 子集，数据完全复现存在缺口。

最值得做的机制复现是固定 Kosmos-1/相近 causal VLM，构造小型 grounded 数据，对比：无 GrIT；noun chunk 框；dependency-expanded expression；不同 location 网格；自由解码 vs 语法约束解码。评测必须同时报告 IoU grounding、无效序列率与普通 VQA/语言退化，避免只看新增能力。

---

## 10. 结论

Kosmos-2 通过 1,024 个 location token 和 `<p>span</p><box>locs</box>` 链接，把区域理解与区域生成纳入同一个 causal vocabulary；GrIT 则以 91M 图像规模提供这种语言—位置映射。它在零样本 phrase grounding 与 referring generation 上证明统一生成接口可行，同时保留大部分 Kosmos-1 能力。最重要的边界是坐标量化、伪标注噪声、短表达域差异、缺少结构消融和与专用 finetuned detector 的明显精度差距。

---

## 11. 材料来源

完整来源、Figure/Page 映射、Kosmos-1 架构核对来源与获取日期见 [sources.md](assets/Kosmos-2%20-%20语言与区域统一Grounding/sources.md)。
