---
type: paper-note
status: done
domain: VLM
paper: "PaLI: A Jointly-Scaled Multilingual Language-Image Model"
year: 2023
arxiv: 2209.06794
doi: null
source: https://arxiv.org/abs/2209.06794
project: null
code: null
tags:
  - VLM
  - multilingual
  - encoder-decoder
  - scaling
  - WebLI
created: 2026-10-16
updated: 2026-10-16
---

# PaLI：多语言视觉语言联合规模化

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | PaLI: A Jointly-Scaled Multilingual Language-Image Model |
| 作者 | Xi Chen、Xiao Wang、Soravit Changpinyo 等，Google Research |
| 发表 | ICLR 2023 |
| arXiv | 2209.06794v4 |
| 最大配置 | PaLI-17B：ViT-e 3.9B + mT5-XXL 13B，总计 16.9B |
| 一句话结论 | PaLI 将大 ViT 的 patch token 与多语言 mT5 encoder-decoder 接成统一“image+query→text”接口，并用 1.6B 多任务样本证明：语言、视觉和分辨率需要联合扩展，其中增加视觉容量的单位参数收益尤其高。 |

---

## 1. 研究问题与设计动机

### 1.1 从“语言模型很大、视觉塔很小”到联合规模化

PaLI 观察到早期大 VLM 的参数分布严重偏向语言端：Flamingo-80B 的语言模型约 70B，而视觉 encoder 只有数亿参数。若视觉表征先成为瓶颈，再大的 decoder 也无法恢复输入中未被编码的文字、细粒度对象和跨语言视觉语义。因此论文不只扩大总参数，而是分别控制语言端和视觉端规模，比较 PaLI-3B、15B、17B。

第二个问题是任务接口碎片化。分类、字幕、VQA、OCR、检测和纯文本理解原本需要不同 head。PaLI 把所有输出都变为文本 token：输入是图像与文本 query，输出是开放词表 answer；任务由 prompt 指定，不增加 task-specific head。

### 1.2 核心贡献

1. 复用 mT5 与 ViT 单模态 checkpoint，构建简单的 encoder-decoder VLM。
2. 新训练 3.9B 参数 ViT-e，并实证视觉规模从 ViT-G 1.8B 增至 ViT-e 时，多模态收益高于同等甚至更大的语言侧增量。
3. 构建 WebLI：10B 图像、12B alt-text、29B OCR 配对、109 种语言；筛选 top 10% 图文对用于训练。
4. 用 8 类预训练目标和统一生成损失覆盖字幕、OCR、VQA/VQG、object-aware QA、生成式检测与纯文本 denoising。

---

## 2. 架构、QKV、mask、position 与 token shape

### 2.1 encoder-decoder 总体结构

PaLI 由 ViT 图像 encoder、mT5 Transformer encoder 和 mT5 Transformer decoder 组成。图像被切成 patch 后由 ViT 产生一串视觉 token；视觉 token 与 prompt token 一起进入 Transformer encoder，decoder 再对 encoder memory 做 cross-attention，自回归输出答案。PaLI 没有 CLIP 式独立文本塔，也没有 BLIP-2 式 Q-Former 瓶颈，所有视觉 patch 都保留到语言 encoder。

![](assets/PaLI%20-%20多语言视觉语言联合规模化/figures/01-architecture.png)

> **Figure 1，PDF 第 4 页。** 图像进入 ViT，图像 token 与文字 query 共同进入 mT5 encoder，mT5 decoder 输出文本答案。其价值在于模块化和可扩展，而不是发明新的注意力算子。来源：https://arxiv.org/abs/2209.06794

### 2.2 QKV 与 attention mask

在视觉端，ViT patch token 通过标准 multi-head self-attention 形成视觉上下文；在 mT5 encoder 中，视觉与文本输入经过双向 encoder attention，任一输入 token 可读取完整可见输入；decoder 使用因果 self-attention，只看已生成前缀，同时以 decoder hidden state 为 $Q$、完整 encoder 输出为 $K,V$ 做 cross-attention。训练采用 teacher forcing，故所有目标 token 可并行计算，但 causal mask 仍阻断未来答案 token。

论文没有提出独立的 connector 或专门 modality mask，也没有把维度匹配层作为贡献或消融对象。视觉 patch feature 被送入 encoder-decoder 的共同 hidden space；精确投影实现未在论文中单列披露。与 Flamingo 的深层门控 cross-attention、BLIP-2 的 query bottleneck 相比，PaLI 的连接方式更接近“保留所有 patch 后交给 seq2seq encoder 统一融合”。

### 2.3 token shape 与位置编码

ViT-e 使用 patch size 14、width 1792、56 层、16 heads、MLP 15360。分辨率决定视觉序列长度：

- $224\times224$：$16\times16=256$ patch token；
- $490\times490$：$35\times35=1225$ patch token；
- $588\times588$：$42\times42=1764$ patch token。

可写成 $X_v\in\mathbb{R}^{B\times N_v\times d_v}$，其中 $N_v=(H/14)(W/14)$；与文本 query 的 $N_t$ 个 token 融合后，encoder memory 长度约为 $N_v+N_t$。PaLI 不做视觉 pooling，所以高分辨率会二次增加 encoder self-attention 成本。

视觉位置使用 ViT 的二维位置 embedding。由 $224$ 升至 $588$ 时，作者对二维 positional embedding 做 bilinear interpolation；patch size 不变。语言侧继承 mT5 的相对位置机制。论文没有报告新的跨模态位置编码或视觉/文本 segment embedding 消融。

---

## 3. 训练数据、统一目标与训练流程

### 3.1 统一生成目标

所有任务都写成 `(image, prompt) → target text`，优化标准序列到序列负对数似然：

$$\mathcal{L}_{\mathrm{seq2seq}}=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},I,q)$$

分类不再训练 softmax head，而是生成类名；检测输出离散坐标与标签序列；纯文本任务可用空图像/无图输入继续训练 mT5 能力。这种接口的优势是任务可共享参数，代价是 exact match 会把同义答案误判为错，闭集分类还需对候选字符串逐一打分。

### 3.2 WebLI 与多任务混合

WebLI 原始规模为 10B 图像、12B alt-text，覆盖 109 种语言；外部 OCR 服务再产生 29B image-OCR pairs。图文对先按跨模态相似度打分，只保留 top 10%，约 1B alt-text 样本。图像还对 68 个常见 benchmark 的 train/val/test 做近重复去除，去除比例约 0.36%。

![](assets/PaLI%20-%20多语言视觉语言联合规模化/figures/03-webli-dataset.png)

> **Figure 4，PDF 第 21 页。** 上半展示英语、法语、泰语、中文的 alt-text/OCR；下半展示语言长尾与相对 CLIP、ALIGN、LiT 的规模。WebLI 的“多语言”不是均匀分布，英语 alt-text 约占 40%，低资源语言仍处于长尾。来源：https://arxiv.org/abs/2209.06794

预训练混合总计约 1.566B 样本：text-only 100M、WebLI split-caption 1000M、OCR 100M、CC3M-35L caption 100M、VQA 100M、VQG 100M、object-aware QA 50M、generative detection 16M。主要目标分别是：

- mT5 span corruption：保持纯语言理解与生成；
- split-caption：给 alt-text 前半，生成后半；
- 多语言 caption：按语言代码生成描述；
- OCR：生成图中文字串；
- VQA/VQG：问题与答案双向任务；
- object-aware QA：列对象或判断对象存在；
- detection：生成量化到 0–999 的 `ymin xmin ymax xmax label` 序列。

### 3.3 两阶段训练与冻结策略

主预训练阶段以 $224^2$ 分辨率遍历 1.6B 混合样本一遍，只更新 mT5 语言 encoder-decoder，冻结 ViT。PaLI-17B 使用 batch 4096，在 1,024 TPUv4 上约 7 天。优化器为 Adafactor，1K step warmup 后 inverse-square-root decay；15B/17B 峰值学习率为 $5\times10^{-3}$。

随后 PaLI-17B 在 $588^2$ 上做 10K step 高分辨率阶段，batch 1024、约 10M 样本，只保留 OCR、CC3M-35L 和 VQA 三类并等权；此阶段解冻全部参数，在 512 TPUv4 上额外约 3 天。下游主要在 490 或 588 分辨率微调，继续用交叉熵。

### 3.4 生成方式与任务决策

字幕、VQA、OCR 均直接自回归生成。零样本 ImageNet 对 1000 个类名字符串分别计算条件得分，选择似然最高者，prompt 为类似 `Generate alt_text in EN at 2: Photo of ...`。这种生成式分类无需新 head，但成本随候选数线性增长。

---

## 4. 定量结果与 baseline

### 4.1 主要结果

| 任务 | PaLI-17B | 代表 baseline | 说明 |
|---|---:|---:|---|
| COCO Caption CIDEr | **149.1** | BEiT-3 147.6、CoCa 143.6、Flamingo-80B 138.1 | 均未用 CIDEr-specific 优化的口径下领先 |
| NoCaps test CIDEr | 124.4 | GIT2 124.8、CoCa 120.6 | 总体接近 GIT2；out-of-domain 为 126.7 |
| VQAv2 test-dev | **84.3** | Flamingo-80B 82.0、BEiT-3 84.2 | PaLI 使用开放词表生成，而部分 baseline 是闭集分类 |
| OKVQA val | **64.5** | Flamingo-80B 32-shot 57.8、KAT 54.4 | 大模型容量带来知识问答收益 |
| TextVQA val / test | 71.81 / 73.06 | GIT2 68.38 / 67.27 | 主结果使用外部 OCR 字符串 |
| XM3600 35 语言平均 CIDEr | **53.6** | 先前工作 28.9 | 多语言 WebLI 与 mT5 的直接收益 |
| ImageNet zero-shot | 72.11 | Flamingo-80B 1-shot 71.9 | 生成式对 1000 类打分，不能与 CLIP 单次 embedding 检索等价比较成本 |

TextVQA、TextCaps 等主结果明显依赖外部 OCR 服务：移除 OCR 输入后，TextVQA 从 73.06 降至 58.80，TextCaps 从 160.4 降至 135.4。这说明“端到端看懂场景文字”的能力低于主表容易给人的印象。

### 4.2 联合扩展结果

![](assets/PaLI%20-%20多语言视觉语言联合规模化/figures/02-scaling-results.png)

> **Figure 2，PDF 第 8 页。** 蓝→红把语言端从 mT5-L 1B 扩到 mT5-XXL 13B；红→橙把视觉端从 ViT-G 1.8B 扩到 ViT-e 3.9B；空心橙再加入 $588^2$ 高分辨率阶段。七任务平均中，增加约 12B 语言参数带来 +3.1，增加约 2B 视觉参数带来 +3.2，高分辨率再带来 +2.0。来源：https://arxiv.org/abs/2209.06794

ViT-e 单独在 ImageNet 上只略优于 ViT-G，但与同一 mT5-XXL 组合时，COCO 从 146.2 升到 149.0、VQAv2 从 82.9 升到 83.4（均不含高分辨率阶段）。这支持论文最重要的判断：视觉 encoder 的饱和点取决于下游任务，分类趋于饱和不等于视觉语言任务也饱和。

### 4.3 消融实验

目标混合消融在 3B 模型、200M data coverage 下进行：

| 变化 | COCO | TextVQA | VQAv2 | XM3600 EN / 6L | 主要结论 |
|---|---:|---:|---:|---:|---|
| 完整混合 | 141.4 | 41.6 | 76.0 | 93.8 / 42.5 | 基准 |
| 去掉 split-caption | 140.4 | 38.8 | 75.5 | 87.5 / 41.5 | WebLI split-caption 是最稳定的通用目标 |
| 去掉 OCR | 142.3 | 39.9 | 75.9 | 95.4 / 43.6 | OCR 提升文字任务，却略损普通 caption |
| 去掉 VQG | 141.4 | 41.3 | 75.8 | 95.1 / 42.0 | VQG zero-shot 从 96.7 大跌至 17.9，其他任务影响小 |
| 去掉 object-related | 140.9 | 40.2 | 75.4 | 90.9 / 41.8 | 对对象意识与多任务迁移都有帮助 |

附录还表明：从预训练单模态 checkpoint 初始化优于从头训练；主阶段冻结 ViT 有轻微收益；去掉非英语 WebLI 会伤害多/跨语言性能。这些消融支持“复用单模态能力 + 多目标混合 + 多语言数据”三者共同作用，而不只是模型变大。

---

## 5. PaLI-X 演化：独立后续论文，不属于 PaLI 原文

> [!important] 边界说明
> 本节讨论的是独立论文 **PaLI-X: On Scaling up a Multilingual Vision and Language Model**（arXiv:2305.18565，2023-05），不是 arXiv:2209.06794 的实验、附录或修订内容。

PaLI-X 延续 PaLI 的模块化 encoder-decoder 路线，但把总规模扩到约 55B：视觉端采用约 22B 的 ViT-22B，语言端以约 32B 的 UL2 为起点，参数分配约 40%/60%，比 PaLI-17B 的约 25% 视觉占比更均衡。任务混合也从 PaLI 的图像 caption/VQA/OCR/检测扩展到文档理解、图表/图示理解、视频问答与字幕，并系统研究 few-shot in-context learning。

因此演化主轴不是更换 connector，而是三件事：继续联合扩大视觉与语言组件；把预训练任务扩展到文档和视频；让 seq2seq 模型处理多示例上下文。PaLI-X 报告复杂计数和多语言检测等未显式训练能力，但这些结论必须归于 PaLI-X 论文，不能倒写成 PaLI-17B 已经证明。

---

## 6. 与 CLIP、Flamingo、BLIP-2、LLaVA 的关系

- **CLIP**：CLIP 是双塔对比模型，独立 embedding 使大规模检索便宜；PaLI 是 encoder-decoder 生成模型，没有天然独立文本/图像索引。PaLI 的优势是统一开放词表输出和多语言生成，代价是分类/检索要对候选序列逐一打分。
- **Flamingo**：Flamingo 冻结 LM、插入门控 cross-attention，并用 interleaved data 获得多图 in-context；PaLI 让 mT5 encoder 统一处理视觉 patch 与文本 query，主阶段冻结 ViT、更新整个语言 encoder-decoder，重点是 supervised/multitask transfer 和多语言，而非原生交错多图 few-shot。
- **BLIP-2**：BLIP-2 用 32-query Q-Former 把视觉压成固定短序列，两端冻结，训练成本远低于 PaLI；PaLI 无 query bottleneck，保留 256–1764 patch token，视觉细节上限更高，但高分辨率 attention 成本和总训练成本更大。
- **LLaVA**：LLaVA 后于 PaLI，采用 CLIP + 线性 projector + decoder-only Vicuna，通过 GPT-4 合成指令数据获得视觉对话。PaLI 是 encoder-decoder、多语言、多任务生成模型，依赖大规模内部数据和下游微调；它不是聊天模型，也没有 LLaVA 的开放指令生态。

---

## 7. 局限、future work 与批判性评价

### 7.1 论文明确局限

作者指出，训练 caption 多为短而简单的网页文本，复杂场景中对象多时描述不充分；英语单语微调会损失部分多语言能力；开放生成评测把正确同义词或释义算错，而闭集分类虽规避此问题却无法泛化到新答案。现有 benchmark 还可能有西方中心偏差。

WebLI 来自自动抓取和自动过滤，仍可能包含偏见、刻板印象和不准确信息。论文提供 data/model card 并做初步年龄/性别分析，但明确称这只是第一步，需要在新任务上继续测量和缓解。模型、WebLI 全量数据和完整 PaLI-17B 权重未开放，使结果难审计。

### 7.2 Future work

论文直接给出的方向包括继续扩大视觉 backbone，因为 ViT-e 在 VLM 上尚未饱和；深入研究识别对性别、年龄等属性的偏差以及 scaling 对偏差的影响；使用多语言数据混合进行下游微调以减轻能力遗忘；改进多语言与非西方中心评测。PaLI-X 是后一阶段的实际延伸，但属于独立工作。

### 7.3 个人判断

PaLI 的历史价值是把“视觉端也需要大规模化”变成受控实验结论。它同时揭示一个容易被主表掩盖的问题：强 OCR 分数部分来自外部 OCR token，强多语言能力也依赖巨大且私有的数据管线。架构本身非常朴素，成果更多来自容量、数据、任务混合和分辨率的协同；这既是 scaling 论文的力量，也是复现与因果归因的弱点。

---

## 8. 复现成本与实践建议

### 8.1 完整复现成本

ViT-e 先在 JFT-3B 上以 batch 16,384 训练 1M step；PaLI-17B 再用 1,024 TPUv4 训练约 7 天，高分辨率阶段追加 512 TPUv4 约 3 天。WebLI 原始抓取、OCR API、top-10% 相似度筛选与 68 benchmark 去重同样是巨大隐性成本。完整复现不可由普通实验室完成。

### 8.2 可行复现

现实方案是复用公开 ViT/mT5 权重，在较小规模上验证四个机制：视觉 token 不池化的 seq2seq 接口；冻结 ViT vs 联合微调；split-caption/OCR/VQA 目标混合；$224\rightarrow$更高分辨率的位置插值。若目标是产品落地，PaliGemma 等后续开放模型比重建 PaLI-17B 更可行，但不能把后续开放实现的结果当作原论文复现。

---

## 9. 结论

PaLI 以简单 encoder-decoder 证明了“联合规模化”而非单纯扩大语言模型：从 mT5-L 到 XXL、从 ViT-G 到 ViT-e、从 $224^2$ 到 $588^2$ 都带来增益，且视觉容量的单位参数回报更高。WebLI 与八目标混合让一个生成接口覆盖多语言 caption、VQA、OCR、检测和纯语言任务。其主要边界是私有数据与超高算力、外部 OCR 依赖、长视觉序列成本和非开放生态。PaLI-X 延续这条路线，但必须作为 2023 年独立论文理解。

---

## 10. 材料来源

完整来源、Figure/Page 映射、PaLI-X 独立论文链接与获取日期见 [sources.md](assets/PaLI%20-%20多语言视觉语言联合规模化/sources.md)。
