---
type: paper-note
status: complete
domain: VLM
paper: "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks"
year: 2023
arxiv: "2311.06242"
doi: null
source: "https://arxiv.org/abs/2311.06242"
project: null
code: null
tags:
  - VLM
  - vision-foundation-model
  - encoder-decoder
  - task-prompt
  - grounding
  - segmentation
  - FLD-5B
created: 2026-10-16
updated: 2026-10-16
---

# Florence-2：统一提示式视觉任务建模

## 0. 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks |
| 作者 | Bin Xiao、Haiping Wu、Weijian Xu、Xiyang Dai、Houdong Hu、Yumao Lu、Michael Zeng、Ce Liu、Lu Yuan |
| 模型规模 | Florence-2-B 232M；Florence-2-L 771M |
| 核心数据 | FLD-5B：126M 图像、5.4B 级综合标注 |
| 主结构 | DaViT image encoder + BART 式 multimodal Transformer encoder-decoder |
| 本地论文 | [paper.pdf](assets/Florence-2%20-%20统一提示式视觉任务建模/sources/paper.pdf) |
| 来源清单 | [sources.md](assets/Florence-2%20-%20统一提示式视觉任务建模/sources.md) |

**一句话结论：** Florence-2 把 caption、检测、grounding、OCR、region caption 与 segmentation 都重写成“图像 + task prompt → 文本/位置 token 序列”的翻译问题，并用 FLD-5B 的多层级标注让一个不足 1B 的 encoder-decoder 同时获得强零样本、通用微调与视觉 backbone 迁移能力。

---

## 1. 研究问题与核心贡献

### 1.1 二维任务空间：空间层级 × 语义粒度

传统视觉基础模型往往只统一 backbone，不统一任务接口：分类读全图语义，检测输出框，分割输出 mask，caption 输出语言，grounding 还要对齐短语与区域。Florence-2 把差异放在两个轴上：

- **Spatial hierarchy**：image-level、region-level、pixel-level。
- **Semantic granularity**：无语义几何、粗粒度类别、细粒度自然语言。

一个真正通用的视觉模型既要横跨全图到像素，也要横跨类别到长描述。问题因此不是“多做几个 benchmark”，而是能否用同一组参数、同一损失与同一输出词表覆盖这个二维空间。

![](assets/Florence-2%20-%20统一提示式视觉任务建模/figures/01-task-space.png)

> 图 1：Figure 1。分类、region proposal、检测、grounding、phrase segmentation 与 detailed caption 被放进同一个“空间层级—语义粒度”坐标系；FLD-5B 提供覆盖，Florence-2 提供统一映射。

### 1.2 核心贡献

1. **Task prompt → sequence**：以文本 prompt 激活任务，把所有输出统一成普通文本与位置 token 的序列，无需任务专用 head。
2. **标准 encoder-decoder 而非超大 LLM**：视觉 token 与 prompt 在 encoder 中双向融合，decoder 自回归产生任务结果；最大模型仅 771M。
3. **FLD-5B 数据引擎**：用专家模型初标、规则与模型过滤、Florence-2 多轮重标注，得到 126M 图像上的 500M text、1.3B region-text、3.6B text-phrase-region 标注。
4. **三种能力验证**：同一预训练模型既做 zero-shot generalist，也能在公开监督数据上形成单一通用模型，还能作为 COCO/ADE20K 下游 backbone。

---

## 2. 统一任务接口与区域序列化

### 2.1 输入输出

输入图像 $I\in\mathbb{R}^{H\times W\times3}$ 经 DaViT 得到展平视觉特征 $V\in\mathbb{R}^{N_v\times D_v}$；线性投影与 LayerNorm 将其变为 $V'\in\mathbb{R}^{N_v\times D}$。任务 prompt 经扩展 tokenizer 得到 $T_p\in\mathbb{R}^{N_t\times D}$，encoder 输入为：

$$X=[V';T_p]\in\mathbb{R}^{(N_v+N_t)\times D}$$

输出是目标序列 $y=(y_1,\ldots,y_L)$。caption 是自然语言，检测是类别与坐标 token，grounding 是短语与框，segmentation 是多边形顶点序列。任务差异主要由 prompt 与目标 serialization 决定，而不是网络 head。

![](assets/Florence-2%20-%20统一提示式视觉任务建模/figures/02-unified-architecture.png)

> 图 2：Figure 2。不同 prompt 与同一视觉序列进入 Transformer encoder；decoder 统一生成文本和 location token。任务头被“输出语法”取代。

### 2.2 Prompt 如何定义任务

代表性 prompt 包括：

- `What does the image describe?` → caption / detailed caption。
- `Locate the objects in the image.` → object detection。
- `Locate the phrases in the caption: ...` → phrase grounding。
- `What does the region (...) describe?` → region caption。
- `What is the polygon mask of region (...)?` → referring segmentation。

Prompt 本质上是离散 task identifier 加自然语言参数。它让同一图像在不同请求下输出不同粒度的结果，也使新任务可通过 prompt+target 数据加入统一训练，而不修改网络结构。

### 2.3 坐标、框与 mask 的序列化

论文向 tokenizer 加入 1,000 个 location bins。连续归一化坐标先量化到离散位置 token，再按任务约定排序：

$$q(x)=\operatorname{round}(999\cdot\operatorname{clip}(x,0,1))$$

上式是对论文“1,000 bins 量化坐标”机制的等价写法，用于解释离散化；论文正文没有指定取整边界的实现细节。

- **Box**：$(x_0,y_0,x_1,y_1)$，按左上角、右下角输出，用于 detection、dense region caption 等。
- **Quad box**：$(x_0,y_0,\ldots,x_3,y_3)$，从左上角开始顺时针，用于 OCR 检测与识别。
- **Polygon**：$(x_0,y_0,\ldots,x_n,y_n)$，顶点按顺时针排列，用于 referring segmentation。

这种设计把连续几何变为离散语言建模，但也引入三个边界：量化上限约为图像边长的千分之一；同一 mask 可有多种等价顶点起点/采样密度；长 polygon 会显著拉长解码并累积误差。

---

## 3. Encoder-decoder 的 Q/K/V、mask、position 与生成

### 3.1 Multimodal encoder

视觉和 prompt token 拼接后进入标准 Transformer encoder。每层中：

$$Q_e=XW_Q^e,\quad K_e=XW_K^e,\quad V_e=XW_V^e$$

encoder self-attention 是双向的；视觉 token 可以读取 prompt，prompt token 也可以读取视觉 token。padding token 被 attention mask 排除，但没有 decoder 式因果遮罩。这一点与 decoder-only VLM 不同：任务指令在生成前已与全部图像位置充分交互。

### 3.2 Autoregressive decoder

在 decoder 第 $t$ 步，masked self-attention 的 Q/K/V 来自已生成目标前缀 $y_{<t}$，使用 causal mask；随后 cross-attention 中 query 来自 decoder hidden state，key/value 来自完整 multimodal encoder memory：

$$Q_d=H_{<t}W_Q^d,\quad K_m=H_eW_K^m,\quad V_m=H_eW_V^m$$

因此生成一个位置 token 与生成一个普通词在概率模型上没有区别，二者都条件于全部图像+prompt memory 和此前输出。

### 3.3 Position

- 图像内部二维位置由 DaViT 的层级视觉表征负责，论文没有为 multimodal encoder 另提出新的二维 RoPE 或坐标注意力。
- 拼接后的 encoder 序列沿用标准序列位置编码；prompt 与视觉 token 通过各自序列位置区分。
- 显式几何位置不依赖 hidden position，而通过 1,000 个 location token 出现在 decoder 输出词表中。

这形成了两层“位置”：输入侧是视觉 encoder 的 patch 位置，输出侧是可被语言模型预测的离散坐标语法。

### 3.4 目标与生成

所有任务使用同一个 cross-entropy objective：

$$\mathcal{L}=-\sum_{i=1}^{|y|}\log P_\theta(y_i\mid y_{<i},I,p)$$

训练时 teacher forcing；推理时由 decoder 自回归生成，遇到结束 token 停止，再由任务 parser 将 location token 还原成框、四边形或 polygon。论文重点是统一表示，没有把某种 beam size 作为核心贡献；复现时应把 decoding 配置与坐标 parser 一并固定，否则检测/分割分数会受重复框、非法坐标和提前结束影响。

---

## 4. FLD-5B 数据引擎

### 4.1 数据规模与组成

FLD-5B 由 126M 图像构成，来源包含 ImageNet-22K、Objects365、Open Images、Conceptual Captions 与经筛选的 LAION。标注不是单一 caption，而是三族：

| 标注族 | 规模 | 主要内容 |
|---|---:|---|
| Text | 500M+ | brief、detailed、more detailed caption |
| Region-text | 1.3B | box + 类别/短语/region caption、OCR 区域 |
| Text-phrase-region | 3.6B | 全图文本中的 noun phrase 与对应 box/mask |

brief/detailed/more detailed 平均长度约 7.95/31.65/70.53 token；平均动作数约 0.58/4.21/8.76。数据规模不只是“更多 caption”，而是刻意提高动作、属性、实体和几何关联密度。

### 4.2 三阶段自动标注

![](assets/Florence-2%20-%20统一提示式视觉任务建模/figures/03-data-engine.png)

> 图 3：Figure 3。专家模型生成初标，文本/区域过滤器清洗，再用 Florence-2 在大数据上训练、用小规模增强数据微调并循环重标注。

1. **Initial annotation**：Azure OCR API、caption、grounding、object/proposal detection、segmentation 等 specialist 并行标注；已有人工框与合成标签合并。
2. **Filtering and enhancement**：文本侧用 SpaCy 解析 object/attribute/action，过滤对象过多的噪声文本并保留较高依存复杂度；区域侧做置信度阈值、NMS、blacklist 与 Florence-1 region score。
3. **Iterative refinement**：先训练统一模型，再用其重新预测训练图；把更好的预测与原始标签合并，进入下一轮。稀缺任务先借大规模预训练获得表示，再用少量数据微调成新的 specialist 扩标。

### 4.3 三类标注如何互补

- **Text** 提供全局语义与描述粒度。
- **Region-text** 让局部视觉区域具备类别和自然语言语义；OCR 区域由文字识别服务产生。
- **Text-phrase-region** 用 Grounding DINO 从描述中找 noun phrase 与框，再用 SAM 生成 mask，建立“长文本中的短语—图像区域—像素轮廓”链条。

它们共同支持 task-prompt 训练：同一图像可转化成 caption、detection、grounding、region caption、segmentation 等多条 seq2seq 样本。

### 4.4 数据风险

FLD-5B 的大部分监督是模型生成，错误可能在“训练统一模型→统一模型再标注”的闭环中自我强化。规则过滤降低明显噪声，但不能证明语义无偏；在线服务和 specialist 版本也未完全开放。5.4B 标注的“规模可复现”与“质量可审计”是两件不同的事。

---

## 5. 训练设置

### 5.1 初始化与规模

- image encoder：DaViT，由 UniCL 初始化。
- multimodal encoder-decoder：BART 初始化。
- Florence-2-B / L：232M / 771M 总参数。
- 图像分辨率：先以 $384\times384$ 训练；B 额外以 $768\times768$ 训练 0.5B samples，L 高分辨率训练 0.1B samples。
- 总训练：达到 3B effective samples；batch size B/L 为 2048/3072。
- 优化：AdamW + cosine decay；B/L 最大学习率 $10^{-4}/10^{-5}$；前 5,000 step 线性 warmup；混合精度与 DeepSpeed。

### 5.2 为什么 encoder-decoder 适合这里

相较把视觉 token 塞入大 decoder-only LLM，Florence-2 的 encoder 可双向融合图像与任务 prompt，decoder 只承担输出序列建模。检测/分割输出通常比输入视觉序列短，这种结构能复用 encoder memory；同时 BART 初始化提供成熟的 seq2seq 生成先验。代价是它不天然继承超大聊天 LLM 的开放世界知识和多轮对话能力。

---

## 6. 实验结果

### 6.1 Zero-shot generalist

Florence-2-L 在 COCO caption 达 135.6 CIDEr，参数不足 Flamingo-80B 的 1%；在 Flickr30K grounding Recall@1 相对 Kosmos-2 提升 5.7，并在 RefCOCO/+/g 分别约提升 4/8/8 个百分点；RefCOCO referring segmentation 达 35.8 mIoU，是当时通用基础模型少见的直接 mask 输出能力。

这些结果验证了“统一输出语法”可覆盖区域与像素任务。但与不同模型比较时，预训练数据、是否见过相似合成标注、解码与 prompt 均不同，不能只按参数量得出普适 scaling 结论。

### 6.2 单一 supervised generalist

加入公开监督数据后，仍保持一套参数覆盖多任务。Florence-2-L 在 COCO caption 为 140.0 CIDEr、TextVQA 为 81.5（无需外部 OCR token）；RefCOCO 系列与 RES 上可与或超过专门设计 regression head、Deformable DETR/DINO backbone 的 specialist。统一建模并未迫使每个任务都接受明显性能折损。

### 6.3 作为下游视觉 backbone

DaViT-B + Florence-2 预训练在 Mask R-CNN 上 COCO box AP 53.6、mask AP 46.4，在 DINO 上 AP 59.2；相对 ImageNet-1K 监督预训练，标准 12-epoch schedule 下分别提高 6.9 与 5.5，并获得论文所称约 4× 训练效率。ADE20K UperNet 为 54.9 mIoU，也显著高于同架构监督预训练。

该结果重要之处在于：生成式多任务预训练学到的不只是可解码文本，还能迁移回稠密视觉任务。

---

## 7. 消融实验

### 7.1 标注层级决定迁移能力

![](assets/Florence-2%20-%20统一提示式视觉任务建模/figures/04-multitask-transfer.png)

> 图 4：Figure 7。仅 image-level 预训练对 caption 最友好，却在 detection、grounding、segmentation 明显不足；加入 region 与 pixel 标注后，局部任务大幅提升，完整三层级在 referring segmentation 最强。

三组预训练分别使用 image-only、image+region、image+region+pixel 标注。结论不是“所有任务越多越好”：caption 上 image-only 更快且最终略高；但检测、grounding、RES 需要 region/pixel supervision。完整多任务模型牺牲少量 caption 峰值，换来任务覆盖与统一迁移。

### 7.2 模型和数据 scaling

- Florence-2-B→L：COCO caption 118.7→124.4、COCO detection 19.7→22.6、Flickr30K grounding 76.3→78.2、RES mIoU 18.6→21.5。
- 预训练数据从 0.12M→12M 时，多数 zero-shot 指标持续提升；RES 在 1.2M 附近略高于 12M，说明数据量并非每项单调。
- detailed caption 和更密的 phrase-region 监督显著增加动作、属性和对象覆盖，但论文没有逐项拆出每种自动标注器的独立贡献。

### 7.3 冻结阶段

在 COCO/ADE20K 下游任务中，冻结 DaViT 前两 stage 仅带来约 0.2–0.3 的下降；完全冻结则明显下降，但仍可超过 ImageNet 监督预训练。这说明浅层通用视觉特征较稳定，任务适配主要需要中高层更新；同时也支持低资源场景采用部分冻结。

---

## 8. 局限与批判性判断

1. **位置序列化不是无损几何。** 1,000 bins、polygon 顶点次序与自回归误差会限制高精度边界；论文没有系统比较 raster mask、RLE、polygon 与专用 mask decoder。
2. **任务统一依赖手工语法。** Prompt 与输出格式仍需人为设计，parser 也按任务编写；它统一的是网络和损失，不是完全消除任务工程。
3. **FLD-5B 难以完整复刻。** 云 OCR、多个 specialist、LLM/LMM 标注、过滤阈值和多轮 checkpoint 组成闭环，论文没有发布足以逐样本重建的数据 lineage。
4. **自动标注偏差可能被放大。** Florence-2 参与后续重标注，模型自身盲点可能转化为数据分布；“wisdom of crowds”只有在标注器误差不高度相关时才可靠。
5. **不等价于聊天型 MLLM。** 模型擅长受 prompt 控制的视觉任务，但开放知识、多轮对话、复杂语言推理不由本文证明；与 7B–80B LLM-VLM 的比较应限定在视觉 benchmark。
6. **高分辨率成本仍高。** $768^2$ tuning 和长 polygon 输出都会提高计算；论文强调参数紧凑，但没有给出完整 latency、显存和每任务解码长度曲线。
7. **安全与治理讨论不足。** 126M web images 与模型生成描述涉及版权、人物、文化偏差和错误 OCR，正文主要关注精度与规模。

我的判断是：Florence-2 的核心不是某个新 attention 层，而是把“视觉任务统一”具体化为可训练的数据协议：prompt 定义操作，location token 定义几何语法，FLD-5B 提供跨层级监督。架构的简洁反而让数据表示成为主要创新与主要风险源。

---

## 9. 复现路线

### 9.1 最小架构复现

1. 选择公开 DaViT/ViT 与 BART-small/base，加入 image projection + LayerNorm。
2. 扩展 tokenizer：至少实现 1,000 location tokens、任务 token 与结束 token。
3. 将 caption、box detection、phrase grounding、polygon segmentation 转为严格序列格式。
4. encoder 使用视觉+prompt 双向 attention mask；decoder 使用 causal mask，并对完整 encoder memory 做 cross-attention。
5. 统一 cross-entropy 训练，评测前实现非法坐标裁剪、重复框解析、polygon 闭合与反量化。

### 9.2 最小数据消融

构建同一批图像的三层监督：caption；caption+box/category；caption+box+mask+phrase grounding。控制图像数与 step，比较 caption CIDEr、detection AP、grounding Recall@1、RES mIoU，即可复现 Figure 7 的关键因果问题。若只追求小规模验证，不必复制 5B 标注。

### 9.3 关键自检

- 坐标统一按原图还是 resize 后图像归一化，训练与 parser 必须一致。
- Quad box 从左上顺时针，polygon 顶点顺序必须确定，否则同一目标出现多种标签。
- encoder padding mask 与视觉 token 数同步；decoder target padding 不计 loss。
- 检测输出需规定 object 排序，否则集合任务被强行序列化后标签熵过大。
- 报告 exact task prompt、beam/greedy、最大输出长度与 parser 失败率。

---

## 10. 与相关方法的关系

- **相对 Unified-IO/ Pix2Seq**：同样把视觉任务转为序列，但 Florence-2 以更大规模、多粒度自动标注构建通用预训练。
- **相对 SAM**：SAM 强于 promptable mask 几何，但缺少自然语言语义与 caption；Florence-2 把 mask 变成可生成位置序列，精度与效率未必优于专用 decoder。
- **相对 LLaVA/IDEFICS2**：后者以 decoder-only LLM 处理开放问答；Florence-2 用 encoder-decoder 直接服务结构化视觉任务，参数更小、输出语法更强，但语言知识较弱。
- **相对检测/分割 specialist**：不再依赖每任务 head，代价是集合与连续几何要通过离散自回归序列近似。

---

## 11. 图表索引

1. `figures/01-task-space.png`：空间层级与语义粒度二维任务空间。
2. `figures/02-unified-architecture.png`：图像+task prompt 到 text/location sequence。
3. `figures/03-data-engine.png`：FLD-5B 自动标注、过滤与迭代重标注。
4. `figures/04-multitask-transfer.png`：image/region/pixel 三层监督消融。

## 12. 材料来源

- [官方论文 PDF](assets/Florence-2%20-%20统一提示式视觉任务建模/sources/paper.pdf)
- [来源与原图清单](assets/Florence-2%20-%20统一提示式视觉任务建模/sources.md)
