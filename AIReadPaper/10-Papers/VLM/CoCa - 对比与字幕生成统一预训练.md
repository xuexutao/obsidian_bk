---
type: paper-note
status: done
domain: VLM
paper: "CoCa: Contrastive Captioners are Image-Text Foundation Models"
year: 2022
arxiv: 2205.01917
doi: null
source: https://arxiv.org/abs/2205.01917
project: null
code: null
tags:
  - VLM
  - contrastive-learning
  - image-captioning
  - encoder-decoder
created: 2026-10-16
updated: 2026-10-16
---

# CoCa：对比与字幕生成统一预训练

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | CoCa: Contrastive Captioners are Image-Text Foundation Models |
| 作者 | Jiahui Yu、Zirui Wang、Vijay Vasudevan、Legg Yeung、Mojtaba Seyedhosseini、Yonghui Wu |
| 机构 | Google Research |
| 版本 | arXiv:2205.01917v2，2022-06-14；论文标注为 Preprint |
| 核心规模 | 最大模型 2.1B：1B ViT 图像编码器 + 1.1B 文本解码器 |
| 一句话结论 | CoCa 用一个“下半段纯文本、上半段跨模态”的因果文本解码器，在一次前向传播中联合对比损失与字幕损失，使同一 checkpoint 同时具备 CLIP 式检索/零样本分类、生成式字幕与融合式视觉语言理解能力。 |

---

## 1. 研究背景与问题定义

### 1.1 三类视觉语言预训练长期割裂

在 CoCa 之前，自然语言监督的视觉模型大致分成三类。单编码器分类模型以固定类别词表监督图像编码器，视觉表征强，但不能开放词汇生成；CLIP/ALIGN 式双塔模型把整幅图与整句文本压成单向量并做对比学习，检索和零样本分类强，但缺少细粒度跨模态融合；SimVLM 等 encoder-decoder 模型通过自回归字幕学习细节与生成能力，却没有天然可直接做全库检索的独立图像/文本嵌入。

CoCa 的问题不是再堆一个任务头，而是让同一计算图同时产出两种表示：一组互不条件化的图像/文本全局向量用于对比学习，一组图像条件化的逐 token 隐状态用于生成与理解。它因此把“表示学习”和“条件生成”放回同一个模型，而非通过多个预训练阶段拼接能力。

### 1.2 核心贡献

1. 将因果文本 decoder 按深度拆成 unimodal 与 multimodal 两段，前半不看图、后半通过 cross-attention 看图。
2. 从同一 image encoder 输出上用任务专用 attention pooler 产生 1 个全局图像 token 与 256 个细粒度图像 token。
3. 一次前向同时计算双向图文对比损失和 teacher-forcing 字幕损失，端到端、单阶段从头训练。
4. 用同一 checkpoint 覆盖视觉分类、图文/视频检索、字幕、VQA、SNLI-VE 与 NLVR2。

![](assets/CoCa%20-%20对比与字幕生成统一预训练/figures/01-overview.png)

> **Figure 1，PDF 第 2 页。** 左侧是联合预训练，右侧分别复用 image encoder、unimodal text decoder 和 multimodal text decoder 完成视觉识别、跨模态对齐、字幕与多模态理解。来源：https://arxiv.org/abs/2205.01917

---

## 2. 输入输出、token 与整体数据流

### 2.1 输入与输出

预训练样本是一对图像 $I$ 与文本 $y_{1:T}$。图像经过 ViT 后得到 patch token 序列；文本经 64K SentencePiece 词表编码，并在末尾追加可学习 `[CLS]`。模型有两个直接输出：

- 对比支路：归一化图像向量 $x_i$ 与文本 `[CLS]` 向量 $y_i$，用于相似度、检索和零样本分类；
- 生成支路：每个文本位置上的条件分布 $p(y_t\mid y_{<t},I)$，用于字幕和开放词表生成。

下游视觉分类可丢弃 decoder，只使用 image encoder；图文检索只使用两个 unimodal 表示；字幕和 VQA 等融合任务使用 multimodal decoder。

### 2.2 token shape 与 position 位置编码

默认预训练分辨率为 $288\times288$，patch 为 $18\times18$，因此视觉网格是 $16\times16=256$ 个 patch token。最大 CoCa 的共同隐藏宽度为 1408、16 个 attention head，单头维度为 88；图像 encoder 有 40 层，文本 decoder 为 18 层 unimodal 加 18 层 multimodal。CoCa-Base/Large 的隐藏宽度分别为 768/1024。

可用 shape 表示主数据流：图像 token 为 $X\in\mathbb{R}^{B\times256\times d}$；对比 pooler 输出 $X_c\in\mathbb{R}^{B\times1\times d}$；生成 pooler 输出 $X_g\in\mathbb{R}^{B\times256\times d}$；文本隐状态为 $H\in\mathbb{R}^{B\times(T+1)\times d}$。论文没有把位置编码列为创新点，ViT 与 Transformer decoder 沿用各自标准位置表示；高分辨率续训将视觉 token 数随空间网格增加。

### 2.3 QKV、attention mask 与 connector

attention pooler 是 CoCa 的视觉 connector：以 $n_{query}$ 个可学习 query 作为 $Q$，ViT 输出同时作为 $K,V$，做一次 multi-head attention。对比目标用 $n_{query}=1$ 得到全局图像向量；字幕目标用 $n_{query}=256$ 保留区域级细节。论文最终默认采用 cascade：先得到生成 token，再在其上做对比池化。

文本 decoder 全部采用左到右 causal mask，任何 token 都不能看未来 token。前 $n_{uni}$ 层只有 masked self-attention，故 `[CLS]` 放在句末时能汇总整句、又不会受到图像条件污染；后 $n_{multi}$ 层除 masked self-attention 外增加 cross-attention，其中文本 hidden state 是 $Q$，生成 pooler 的视觉 token 是 $K,V$。最大模型取 $n_{uni}=n_{multi}=18$。

![](assets/CoCa%20-%20对比与字幕生成统一预训练/figures/02-architecture.png)

> **Figure 2，PDF 第 4 页。** 图中同时展示 1-query 对比池化、256-query 生成池化、句末 `[CLS]`、前半 unimodal decoder 与后半 cross-attention decoder。右侧伪代码强调两个损失共享一次图像与文本前向。来源：https://arxiv.org/abs/2205.01917

---

## 3. 训练目标、数据与生成方式

### 3.1 双向对比目标

批内 $N$ 对图文样本的归一化表示做双向 softmax，对比损失为：

$$\mathcal{L}_{\mathrm{Con}}=-\frac{1}{N}\left[\sum_i\log\frac{\exp(x_i^\top y_i/\sigma)}{\sum_j\exp(x_i^\top y_j/\sigma)}+\sum_i\log\frac{\exp(y_i^\top x_i/\sigma)}{\sum_j\exp(y_i^\top x_j/\sigma)}\right]$$

温度初值为 0.07 且可学习。它直接塑造可索引的全局空间，因此无需额外融合网络即可做 cosine retrieval 或以类别 prompt 做零样本分类。

### 3.2 自回归字幕目标与联合损失

multimodal decoder 在 teacher forcing 下预测下一 token：

$$\mathcal{L}_{\mathrm{Cap}}=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},I)$$

总目标为：

$$\mathcal{L}_{\mathrm{CoCa}}=\lambda_{\mathrm{Con}}\mathcal{L}_{\mathrm{Con}}+\lambda_{\mathrm{Cap}}\mathcal{L}_{\mathrm{Cap}}$$

最终使用 $\lambda_{\mathrm{Con}}=1$、$\lambda_{\mathrm{Cap}}=2$。对比目标只读 unimodal decoder 的句末 `[CLS]`，字幕目标读完整 multimodal 输出；两者共享 image encoder、视觉 pooler 的大部分计算和文本前半层。

### 3.3 训练数据与优化

训练数据由 JFT-3B 标注图像与 ALIGN 噪声 alt-text 构成。JFT 多标签被随机打乱后拼入 CLIP 风格 prompt，所有标签都按文本处理；两类数据各占 batch 一半。作者针对下游集合做近重复去除，共移除 360 万图像。

所有参数从头端到端训练 500K step，batch size 65,536，约等于 JFT 5 epoch 与 ALIGN 10 epoch。优化器为 Adafactor，$\beta_1=0.9$、$\beta_2=0.999$、weight decay 0.01；前 2% step warmup 到 $8\times10^{-4}$ 后线性衰减。主训练约使用 2,048 个 Cloud TPUv4、5 天，随后再以 $576\times576$ 分辨率训练 1 epoch。

### 3.4 推理与生成

字幕和开放生成按标准自回归解码；论文主结果在 COCO 微调时只用交叉熵，没有使用 CIDEr-specific 强化优化。检索阶段完全绕过 multimodal decoder，预编码图像与文本后按余弦相似度排序。视频零样本检索对均匀采样的 16 帧图像嵌入取均值；视频分类则让一个新的 attention query 汇聚所有空间×时间 token。

---

## 4. 定量结果与 baseline

### 4.1 代表性结果

| 任务 | CoCa | 关键 baseline | 结论 |
|---|---:|---:|---|
| ImageNet zero-shot top-1 | 86.3 | 论文对比 CLIP、ALIGN、Florence、LiT 等 | 同时依靠强视觉表征与图文对齐 |
| ImageNet fine-tuned top-1 | 91.0 | 当时专用视觉模型 | 单一多模态 checkpoint 仍能达到强视觉结果 |
| Flickr30K I2T / T2I R@1 | 92.5 / 80.4 | Florence 90.9 / 76.7 | 生成目标没有破坏双塔式检索 |
| MSCOCO I2T / T2I R@1 | 66.3 / 51.2 | Florence 64.7 / 47.2 | 两个方向均提升 |
| VQAv2 test-dev | 82.3 | SimVLM 80.0、METER 80.3 | multimodal decoder 可直接服务融合任务 |
| COCO CIDEr | 143.6 | SimVLM 143.3、BLIP 136.7 | 未使用 CIDEr 优化仍有竞争力 |
| NoCaps test CIDEr | 120.6 | SimVLM 110.3、LEMON 114.3 | 大规模开放词汇预训练改善长尾描述 |

这些数字来自不同迁移协议，不能把 zero-shot、冻结特征和全量微调混成一个统一排名。论文真正要证明的是能力覆盖面，而非每一项都以同一计算预算公平胜出。

![](assets/CoCa%20-%20对比与字幕生成统一预训练/figures/03-results-radar.png)

> **Figure 4，PDF 第 7 页。** 紫色 CoCa 在 ImageNet/Kinetics、图文检索、NoCaps、VQA、SNLI-VE、NLVR2 等轴上同时接近或超过专用模型，视觉化地支持“单 checkpoint 覆盖三类范式”的主张；雷达图各轴量纲不同，只适合看覆盖范围，不适合计算平均优势。来源：https://arxiv.org/abs/2205.01917

### 4.2 消融实验

| 设置 | ImageNet ZS | VQA | 关键解释 |
|---|---:|---:|---|
| 仅 $\mathcal{L}_{Con}$ | 70.7 | 59.2 | 有检索空间，但融合推理弱且 VQA 还需额外融合 |
| 仅 $\mathcal{L}_{Cap}$ | — | 68.9 | 生成/理解强，却没有直接 zero-shot 双塔接口 |
| CoCa 双目标 | **71.6** | **69.0** | 仅比 caption-only 训练成本 1.18× vs 1.17× |
| $n_{uni}=3/6/9$ | 70.2/71.6/71.4 | 69.0/69.0/68.8 | 对半拆分是稳妥折中，不是极尖锐最优点 |
| 生成 query 数 1/32/256 | 69.3/71.2/**默认 256** | 64.4/68.2/**69.0** | 固定为 1 个视觉 token 会明显损失细粒度信息 |

损失权重消融中，caption:contrastive 为 2:1 得到 71.6 ZS、69.0 VQA，优于 1:1 的 71.5/68.6 与 1:2 的 71.0/68.1。文本聚合用单个句末 `[CLS]` 优于 8 个 `[CLS]` 或把原文本 token 一起聚合，说明专用句向量能减少生成目标与对比目标之间的表示干扰。视觉 pooler 的 cascade 也略优于 parallel。

---

## 5. 与 CLIP、Flamingo、BLIP-2、LLaVA 的关系

- **相对 CLIP**：CoCa 保留独立图像/文本 embedding 和双向对比损失，因此保留检索与零样本分类；新增的 multimodal decoder 与字幕损失使其能生成和做融合推理。代价是训练与推理结构显著更重。
- **相对 Flamingo**：两者都用 cross-attention 注入视觉信息，但 CoCa 端到端从头预训练、主要面向单图，视觉信息进入 decoder 的上半层；Flamingo 冻结大 LM，在多个层间插门控 cross-attention，并靠交错网页数据学习多图 in-context。CoCa 不具备 Flamingo 原生的多图少样本接口。
- **相对 BLIP-2**：CoCa 联合更新 2.1B 全模型并用 256 个生成视觉 token；后来的 BLIP-2 冻结 ViT 与 LLM，用 32-query Q-Former 作为强瓶颈，复现门槛更低。BLIP-2 的两阶段目标更强调“先对齐，再接 LLM”，而 CoCa 强调单阶段共享计算。
- **相对 LLaVA**：CoCa 早于 LLaVA。LLaVA 用 CLIP patch token + 线性 projector + Vicuna，并以合成视觉指令数据追求对话；CoCa 没有指令微调，connector 更复杂，优势在统一对比/生成预训练与视觉表征，不应把其字幕能力等同于后来的聊天式指令遵循。

---

## 6. 局限、future work 与批判性评价

### 6.1 明确局限

训练依赖私有 JFT-3B、ALIGN、Lingvo/GSPMD 与 2,048 TPUv4，论文未发布完整训练数据、权重和官方训练代码，绝对结果难以独立复现。单图 alt-text/标签训练也没有建立多轮对话、交错多图上下文、区域框输出或可靠 OCR 接口。256 个视觉 token 在 $576^2$ 高分辨率续训时仍是池化瓶颈，细小文字和密集对象可能被压缩。

论文的 broader impacts 还明确提醒：互联网与大规模标注数据会继承有害关联、公平性和社会偏差；对已测 corruption 更稳健不代表真实环境安全。雷达图虽直观，但混合不同指标与迁移协议，不能替代逐任务公平比较。

### 6.2 Future work

论文没有给出独立、具体的 future-work 清单，只表示希望统一框架推动新的 image-text foundation model 方向。因此，多图交错训练、开放指令遵循、结构化 grounding、公开数据复现等属于根据方法边界提出的后续问题，而不是作者原文承诺。

### 6.3 个人判断

CoCa 的关键不是“多加一个 loss”，而是为两个 loss 设计互不冲突的读出位置：对比学习停在未看图的句末 `[CLS]`，生成学习继续进入看图的 decoder。这个分界把独立 embedding 与融合 hidden state 放进同一条深度路径。它证明生成目标能反哺 zero-shot alignment，但没有证明端到端重训是最经济选择；BLIP-2、InstructBLIP 与 LLaVA 的流行恰恰说明，公开生态更偏好冻结强底座并训练小连接器。

---

## 7. 复现成本与实践建议

### 7.1 完整复现

按论文规格完整复现几乎只适合大型工业实验室：约 1.8B 级数据、JFT 私有标签、65,536 batch、500K step、2,048 TPUv4×5 天，之后还有高分辨率阶段。数据去重和 JFT 标签文本化也是结果的一部分，不能只复刻网络结构。

### 7.2 可行的机制复现

研究级复现应缩小为 CoCa-Base：使用公开 ViT 与 causal decoder，在公开图文对上实现 1-query/多-query pooler，比较 contrastive-only、caption-only、联合目标以及不同 $n_{uni}:n_{multi}$。最值得验证的不是 91.0 ImageNet，而是三条论文结论：联合目标是否以接近 caption-only 的成本增加检索能力；生成 query 数是否决定 VQA；句末单 `[CLS]` 是否稳定优于多 token 聚合。

---

## 8. 结论

CoCa 把 CLIP 式全局对齐与 encoder-decoder 式细粒度生成统一在单阶段训练中：同一因果 decoder 的深度分界同时提供“未看图的文本向量”和“看图后的多模态表示”，attention pooler 再按任务控制视觉 token 粒度。实验支持联合目标、对半 decoder 与长视觉序列的有效性，但其私有数据/超大算力、单图接口和缺少开放权重限制了可复现性。历史上它是从“双塔对齐”走向“生成式 VLM”的关键桥梁；工程上，后续冻结底座与轻量 connector 路线更易扩散。

---

## 9. 材料来源

完整来源、Figure/Page 映射与获取日期见 [sources.md](assets/CoCa%20-%20对比与字幕生成统一预训练/sources.md)。
