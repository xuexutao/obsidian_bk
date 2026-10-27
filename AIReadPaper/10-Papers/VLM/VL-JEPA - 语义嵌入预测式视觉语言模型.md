---
type: paper-note
status: complete
domain: VLM
paper: "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
year: 2025
arxiv: "2512.10942"
doi: "10.48550/arXiv.2512.10942"
source: "https://arxiv.org/abs/2512.10942"
project: null
code: "仅第三方复现；官方代码、推理代码和权重均未发布"
code_commit: null
third_party_code: "https://github.com/Soumya-Chakraborty/VL-JEPA"
third_party_code_commit: "7400361ecbe520504f285d4c7e906e1205b68469"
tags:
  - VLM
  - JEPA
  - representation-learning
  - video-understanding
  - contrastive-learning
created: 2026-10-27
updated: 2026-10-27
---

# VL-JEPA：语义嵌入预测式视觉语言模型

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | **VL-JEPA: Joint Embedding Predictive Architecture for Vision-language** |
| 作者 | Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung |
| 机构 | Meta FAIR、HKUST、Sorbonne Université、NYU |
| 版本 | arXiv:2512.10942v2，2026-02-02 |
| 领域 | Vision-Language Model、JEPA、视频理解、表征学习 |
| 核心模型 | 冻结 V-JEPA 2 ViT-L + 8 层 Llama-3.2-1B Predictor + EmbeddingGemma-300M Y-Encoder |
| 官方代码 | **截至 2026-10-27 未发布** |
| 官方权重 | HF 有作者账号占位仓库，但文件树无权重、config、tokenizer 或代码，`usedStorage=0` |
| 第三方实现 | 有；本笔记代码章节审阅 `Soumya-Chakraborty/VL-JEPA`，仅作实现参考 |
| 个人评级 | **★★★★☆**：范式清晰、受控实验有启发，但 Y-Decoder 与关键训练细节缺失，在线选择性解码证据弱于宣传 |

## 一句话结论

VL-JEPA 把视觉语言学习从“根据视觉与问题逐 token 写出答案”改写为“预测目标答案的连续语义嵌入”，从而把理解、检索、分类与按需文本输出统一在同一个表示空间里；它最有价值的贡献不是已经替代生成式 VLM，而是证明了：**对于不总需要自然语言输出的任务，可以把昂贵的自回归解码从默认计算路径中移除。**

> [!summary] 先看结论
> - **方法亮点**：用查询条件下的目标嵌入预测替代 next-token prediction，语义等价但措辞不同的答案可在连续空间中靠近。
> - **实验证据**：受控小规模对比中，15M samples 后 VL-JEPA 的平均 caption CIDEr 为 14.8、分类 Top-5 为 41.0%，VLM 基线分别为 7.1 和 27.2%。
> - **任务覆盖**：同一嵌入接口支持 caption、候选式 VQA、开放词汇分类、文本到视频检索和状态变化匹配。
> - **效率亮点**：EgoExo4D 上 0.35Hz 的选择性解码达到均匀 1Hz 解码的相近 CIDEr，文本解码次数约减少 2.85×。
> - **关键保留**：实验中的选择性解码是看完整段序列后做时序层次聚类，并非真正在线的因果触发；Y-Decoder 架构与训练方式也没有披露。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

典型生成式 VLM 接收视觉输入 $X_V$ 和文本查询 $X_Q$，再自回归生成目标文本 $Y$：

$$f_{\mathrm{VLM}}:(X_V,X_Q)\mapsto\hat Y$$

这条路线的优势是接口自然、能输出任意文本，也能复用成熟 LLM；但论文指出，模型训练时被迫同时学习两类信息：

1. **任务相关语义**：画面里发生了什么、问题的正确答案是什么；
2. **表层语言形式**：选择哪些词、采用什么语序、风格和同义改写。

例如，“the lamp is turned off”和“the room will go dark”可能都正确，但在 one-hot token 空间几乎正交。next-token cross-entropy 必须拟合多个分离的离散模式，而任务本身往往只关心二者共享的语义。

论文提出的问题因此不是“如何再造一个更小的 LLM”，而是：

> 能否先预测答案的**语义表示**，只有在确实需要人类可读文本时再解码？

### 1.2 两个现实瓶颈

#### 1.2.1 训练目标包含不必要的语言表面变化

传统 VLM 的训练目标可写成：

$$\mathcal L_{\mathrm{VLM}}=-\sum_t\log p(y_t\mid X_V,X_Q,y_{<t})$$

即使多个答案语义等价，只要 token 不同，模型仍需分别分配概率质量。论文认为这降低了样本效率，并让视觉语义学习与语言写作能力耦合。

#### 1.2.2 流式视频并不需要持续生成文本

智能眼镜、机器人状态监控和在线规划要持续处理视频，但“每来一帧就生成一句话”通常没有必要。生成式 VLM 必须先完成 token 解码，才能知道语义是否变化；而一个连续语义流可以先低成本监控变化，仅在事件切换时输出文本。

### 1.3 与现有范式的定位

| 范式 | 学习映射 | 原生强项 | 原生短板 |
|---|---|---|---|
| CLIP / JEA | $X_V\mapsto S_V,\ Y\mapsto S_Y$，对齐 $S_V$ 与 $S_Y$ | 分类、检索、zero-shot 表征 | 缺少查询条件下的答案预测与开放文本输出 |
| 生成式 VLM | $(X_V,X_Q)\mapsto\hat Y$ | caption、开放问答、推理、工具接口 | 默认必须生成 token，理解与语言表面形式耦合 |
| VL-JEPA | $(S_V,X_Q)\mapsto\hat S_Y$ | 语义预测、分类、检索、候选问答、按需解码 | 文本生成依赖额外 Decoder，长推理能力尚未验证 |

VL-JEPA 的核心位置可以概括为：**在 CLIP 的共享嵌入空间上加入“查询条件下的预测器”，再为需要自然语言输出的场景挂接按需 Decoder。**

### 1.4 核心贡献

1. 提出面向通用视觉语言任务的 JEPA 架构，把监督从离散 token 空间移动到连续目标文本嵌入空间。
2. 在视觉编码器、数据、batch、训练步数等尽量对齐的实验中，展示嵌入预测相较 token 预测的样本效率优势。
3. 用一个统一嵌入接口覆盖 caption、候选式 VQA、开放词汇分类和文本到视频检索。
4. 基于连续语义流实现选择性解码，在 EgoExo4D 上以更低解码频率维持相近 CIDEr。
5. 构建 BASE + SFT 两阶段 1.6B 模型，在 8 个视频分类、8 个视频检索和 4 个 VQA 基准上验证通用性。

---

## 2. 任务定义与统一数学视角

### 2.1 训练样本与四个映射

训练样本为三元组：

$$\langle X_V,X_Q,Y\rangle$$

其中 $X_V$ 是图像或视频帧，$X_Q$ 是 prompt / 问题，$Y$ 是目标 caption 或答案。VL-JEPA 包含四个模块：

$$S_V=f_X(X_V)$$

$$\hat S_Y=f_P(S_V,X_Q)$$

$$S_Y=f_Y(Y)$$

$$\hat Y=f_D(\hat S_Y)$$

训练主目标发生在嵌入空间：

$$\mathcal L_{\mathrm{VL\text{-}JEPA}}=D(\hat S_Y,S_Y)$$

与生成式 VLM 的差别不是简单“换一种 loss”，而是改变了预测对象：VLM 预测可见的 token 序列，VL-JEPA 预测由 Y-Encoder 学出的抽象语义坐标。

### 2.2 CLIP、VLM 与 VL-JEPA 的差别

CLIP 直接对齐当前图像和文本：

$$\mathcal L_{\mathrm{CLIP}}=D(S_V,S_Y)$$

VL-JEPA 则在查询条件下，从视觉表示预测目标语义：

$$\mathcal L_{\mathrm{VL\text{-}JEPA}}=D(f_P(S_V,X_Q),S_Y)$$

因此它比 CLIP 多了一个**条件映射**：同一段视频在不同问题下可以对应不同的目标嵌入；也比生成式 VLM 少了一项默认义务：不必每次把语义写成完整句子。

### 2.3 “语义压缩”到底压掉了什么

论文希望 $S_Y$ 保留任务相关语义，并对词序、同义改写、风格等变化不敏感。直觉上，离散 token 空间中的多峰目标分布被 Y-Encoder 压成较紧凑的连续簇，Predictor 不必同时拟合多个互不重叠的字符串模式。

但这里有一个重要边界：**压缩并不天然等于只保留正确语义。**目标空间究竟保留哪些区别，由 Y-Encoder、对比损失和训练数据共同决定。如果 Y-Encoder 对否定、关系或计数不敏感，Predictor 也会继承这个盲点。因此论文专门用 SugarCrepe++ 和 VISLA 测试文本硬负样本。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/VL-JEPA%20-%20语义嵌入预测式视觉语言模型/figures/01-architecture-overview.png)

> 图 1：VL-JEPA 的抽象结构。视觉输入经 X-Encoder 得到 $S_V$，Predictor 在查询 $X_Q$ 条件下预测 $\hat S_Y$；训练目标是让它接近 Y-Encoder 产生的 $S_Y$，Y-Decoder 只在需要文本时启用。  
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2512.10942

![](assets/VL-JEPA%20-%20语义嵌入预测式视觉语言模型/figures/02-method-and-applications.png)

> 图 2：左侧给出论文实际实例化：V-JEPA 2 视觉编码器、Llama 3 Predictor 与 EmbeddingGemma Y-Encoder；右侧说明同一个嵌入接口如何支持文本生成、候选式 VQA / 分类和文本到视频检索。  
> 来源：论文 Figure 2，第 3 页，https://arxiv.org/abs/2512.10942

### 3.2 X-Encoder：从视觉输入到视觉 token

论文主模型使用冻结的 **V-JEPA 2 ViT-L**：

- 参数量：304M；
- 输入分辨率：$256^2$；
- 图像输入通过复制同一图像来匹配视频输入形状；
- 输出是视觉 token 序列 $S_V$。

冻结视觉编码器的动机是把研究重点放在“如何从成熟视觉表示预测语言语义”，同时降低可训练参数和大规模训练成本。它也让 BASE 模型继承 V-JEPA 2 对运动与动作的偏置，这可以解释 VL-JEPA 在 SSv2、EK100、EgoExo4D 等运动中心数据集上表现突出，而在 Kinetics-400 等外观中心任务上相对弱一些。

论文没有披露主模型的 patch size、每帧 token 数、默认推理帧数、时空下采样比例和 X-Encoder 输出宽度，所以不能从论文可靠恢复具体 $[B,N_V,d_V]$。

### 3.3 Predictor：查询条件下预测目标语义

Predictor 是 VL-JEPA 的核心：

- 从 Llama-3.2-1B 的最后 8 个 Transformer 层初始化；
- 约 490M 可训练参数；
- tokenizer 与 token embedding 同样来自 Llama-3.2-1B；
- 查询最大 512 tokens，短查询补 `[PAD]`；
- 关闭 causal mask，使视觉 token 与查询 token 双向联合注意；
- 对 non-`[PAD]` token 做平均池化，再线性投影到目标空间。

在因果 LLM 中，后出现的 query token 可以看前面的视觉 token，但前面的视觉 token 看不到 query。VL-JEPA 需要回答“在这个问题下，当前视觉意味着什么”，因此让视觉与问题双向交互更自然。消融中恢复因果注意力后，VQA 从 42.5 降到 40.6，下降 1.9 点；分类和检索变化很小，说明双向条件融合主要服务问答，而不是基础视觉语言对齐。

论文对 pooling 范围的措辞是“non-[PAD] tokens”，没有明确说明是否包含所有视觉 token 与查询 token，还是只池化查询侧输出。这是官方代码缺失时无法确认的实现细节。

### 3.4 Y-Encoder：共同学习目标语义空间

Y-Encoder 使用 **EmbeddingGemma-300M** 初始化：

- 最大文本长度 512；
- 与 Predictor 各自通过线性 projection 映射到 1,536 维共享空间；
- Y-Encoder 不是固定 teacher，而是与 Predictor 联合更新；
- Y-Encoder 学习率为主学习率的 $0.05\times$。

因此训练输出至少可以抽象为：

$$\hat S_Y,S_Y\in\mathbb R^{B\times1536}$$

让 Y-Encoder 可训练的好处是目标空间可以适应视觉语言任务，而不只是继承通用句向量模型；风险是 Predictor 和 Y-Encoder 可能一起坍塌到常数表示，或者目标空间移动过快导致不稳定。论文用 InfoNCE 的 batch uniformity 约束与较小的 Y-Encoder 学习率同时控制这两个问题。

学习率消融支持“慢速移动目标”的必要性：

| Y-Encoder LR multiplier | 分类 | 检索 | VQA |
|---:|---:|---:|---:|
| 0.05 | 27.3 | 30.2 | 42.5 |
| 1.00 | 23.7 | 28.8 | 40.7 |
| 0.10 | 26.9 | 30.2 | 42.9 |
| 0.01 | 25.6 | 27.7 | 41.0 |
| 0.00（冻结） | 20.0 | 25.9 | 41.4 |

0.05–0.10 是较稳定区间；完全冻结明显损伤分类与检索，而全速更新也不理想。

### 3.5 双向 InfoNCE 与反坍塌

设 batch 中预测与目标嵌入的相似度为：

$$s_{ij}=\operatorname{sim}(\hat S_{Y,i},S_{Y,j})$$

双向 InfoNCE 可写成：

$$\mathcal L_{\hat S\rightarrow S}=-\frac1B\sum_i\log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)}$$

$$\mathcal L_{S\rightarrow\hat S}=-\frac1B\sum_i\log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ji}/\tau)}$$

$$\mathcal L=\frac12(\mathcal L_{\hat S\rightarrow S}+\mathcal L_{S\rightarrow\hat S})$$

这里的显式写法是标准双向 InfoNCE 展开；论文只说明采用双向 InfoNCE，没有披露温度 $\tau$、是否可学习、跨卡负样本的 all-gather 方式、重复 caption 去重和 false-negative 处理。

InfoNCE 同时包含两种作用：

- **alignment**：让同一样本的 $\hat S_{Y,i}$ 与 $S_{Y,i}$ 靠近；
- **uniformity**：让不同样本分散，避免所有表示收缩为同一个常数。

损失消融为：

| 损失 | 分类 | 检索 | VQA |
|---|---:|---:|---:|
| InfoNCE | 23.3 | 30.3 | 44.3 |
| Cosine | 16.5 | 20.2 | **46.6** |
| L1 | 14.8 | 15.5 | 41.9 |
| L2 | 13.5 | 11.7 | 43.7 |

InfoNCE 在分类和检索上优势明显，但 cosine 的 VQA 高 2.3 点。这说明统一嵌入空间与候选式问答之间存在目标权衡；论文选择 InfoNCE 的理由不仅是平均性能，也因为它允许联合训练 Y-Encoder 而不易坍塌。

### 3.6 Y-Decoder：论文最大复现缺口

论文只说 Y-Decoder 在需要文本时把 $\hat S_Y$ 转换为 $\hat Y$，且不参与“主训练阶段”。但以下关键信息都没有披露：

- Decoder 的具体基础模型和参数量；
- 单个 1,536 维向量如何注入语言模型；
- Decoder 是自回归、扩散式还是检索增强；
- Decoder 的训练数据、目标函数与步数；
- Decoder 是否计入表中的 1.6B 总参数；
- caption 与选择性解码实验使用的采样策略、最大长度和 beam 设置。

由于 caption 结果和 Figure 3 的文本解码延迟都依赖这个模块，Decoder 缺失使论文的“开放文本输出”链路无法独立复现，也很难判断 caption 增益来自 embedding prediction 还是额外 Decoder。

### 3.7 多任务推理接口

#### 3.7.1 Caption 与开放式 VQA

Predictor 先输出 $\hat S_Y$，再由 Y-Decoder 生成文字。理解与写作被拆成两段，但真正生成文字时仍可能需要自回归解码。

#### 3.7.2 开放词汇分类与候选式 VQA

把所有候选标签或答案通过 Y-Encoder 编码为 $S_{Y,k}$，选择与预测嵌入最接近的候选：

$$k^*=\arg\max_k\operatorname{sim}(\hat S_Y,S_{Y,k})$$

这一路径不需要生成 token，计算稳定且适合封闭候选集。

#### 3.7.3 文本到视频检索

每个候选视频通过一个 retrieval caption prompt 映射为预测嵌入；检索查询由 Y-Encoder 编码，再按相似度排序。与 CLIP 相比，视频侧表示是 prompt-conditioned 的，而不是固定的视觉全局向量。

---

## 4. 两阶段训练与计算成本

### 4.1 第一阶段：query-free 大规模预训练

#### 4.1.1 图像阶段

- 数据：DataComp + YFCC-100M；
- 每个输入 1 帧；
- batch size 24k；
- 100k iterations；
- 常量学习率 $5\times10^{-5}$；
- 论文称见过 2B samples；
- ImageNet zero-shot 61.6%，未使用 prompt ensembling。

这里有数字疑点：$24,000\times100,000=2.4B$，而不是 2B。论文没有解释是否存在动态 batch、跳过样本或只是近似写法。

#### 4.1.2 视频阶段

- 数据：Action100M，基于 HowTo100M 视频生成动作描述与 caption；
- 8 帧训练 60k iterations；
- 最后 32 帧训练 10k iterations；
- 视频阶段 batch size 未披露。

这一步把图文对齐扩展到运动语义，是 BASE 在 motion-centric benchmark 上强势的重要原因。

### 4.2 第二阶段：query-conditioned SFT

数据来自 PLM mixture：

- 25M VQA；
- 2.8M captioning；
- 1.8M classification；
- 加入下采样的预训练数据防止灾难性遗忘，但比例未披露。

训练配置：

- 83k steps；
- batch size 3,072；
- cosine learning-rate annealing；
- 24 节点；
- PDF 文本写作“约 2.5s days”，明显存在排版或笔误。

$83,000\times3,072\approx255M$ 次样本呈现，与表中 BASE 3.3B 到 SFT 3.6B 的增量量级一致。作者明确承认 SFT 使用大量人工标注和 in-domain 数据，因此 SFT 结果不应被解释成 zero-shot。

### 4.3 硬件成本

预训练耗时 4 周，使用 24 个节点、每节点 8 张 NVIDIA H200，即 192 张 H200。若按持续满载估算：

$$4\times7\times24\times192=129{,}024\ \text{H200 GPU-hours}$$

这是基于论文数字的估算，不是作者直接报告值。对于一个强调效率的工作，这仍是很高的训练门槛；效率主要体现在学习目标与在线解码路径，而不是“低算力可复现”。

### 4.4 参数账目无法闭合

论文披露：

- X-Encoder：304M；
- Predictor：490M；
- Y-Encoder：300M；
- 表中总模型：1.6B。

前三者之和为 1.094B，仍有约 0.5B 未解释。不能未经证据把差额全部归因于 Y-Decoder，因为论文没有说明 Decoder 是否计入、线性层与 embedding 又占多少。

---

## 5. 实验设置与主要结果

### 5.1 分类与检索

BASE 使用 3.3B samples，表中结果为：

| 模型 | 参数 | 训练样本 | 8 个视频分类平均 Top-1 | 8 个视频检索平均 R@1 |
|---|---:|---:|---:|---:|
| PE-Core-G | 2.3B | 86B | 44.7 | 58.1 |
| VL-JEPA BASE | 1.6B | 3.3B | **52.5** | **63.7** |
| VL-JEPA SFT | 1.6B | 3.6B | **75.4** | 63.8 |

BASE 相比最强 zero-shot 对照 PE-Core-G：

- 分类平均 +7.8；
- 检索平均 +5.6；
- 训练样本少一个数量级以上。

但 SFT 的 75.4 分类平均来自 in-domain 训练，不可与 zero-shot 模型作同性质归因；检索平均只从 63.7 到 63.8，部分数据集还下降，说明 SFT 主要提升分类与问答，而非普遍改善共享空间。

### 5.2 候选式 VQA

| 数据集 | VL-JEPA SFT 1.6B | 表内最佳值 | 结论 |
|---|---:|---:|---|
| GQA | 61.5 | 66.6 | 竞争力强，但非最佳 |
| TallyQA | 69.9 | 76.8 | 计数仍有差距 |
| POPE | 85.7 | 87.5 | 幻觉评测接近强基线 |
| POPEv2 | 86.3 | 91.3 | 明显落后最佳模型 |

这些评测刻意选择偏视觉感知、计数和物体幻觉的任务，且采用候选答案最近邻，而不是自由生成。它们不能证明 VL-JEPA 已覆盖知识问答、OCR、开放式多轮对话或长链推理。

### 5.3 WorldPrediction-WM 与动作预测

WorldPrediction-WM 给定初始与最终状态图，从四个候选动作视频中选出解释状态变化的动作：

- BASE：63.9%；
- SFT：65.7%；
- 表内此前最高：57.0%。

该任务非常适合 VL-JEPA：它本质上是“状态变化语义嵌入”与“动作语义嵌入”的最近邻匹配，而不是需要自然语言生成的开放任务。

动作预判结果：

| 模型 | 1s | 2s | 4s | 10s |
|---|---:|---:|---:|---:|
| V-JEPA2 ViT-g-384 | 39.7 | 28.6 | 19.3 | 8.2 |
| V-JEPA2 ViT-L-256 | 32.7 | 23.4 | 15.9 | 7.1 |
| VL-JEPA ViT-L-256 | **34.2** | **26.0** | **19.4** | **11.7** |

与同尺寸视觉编码器相比，VL-JEPA 在所有预测间隔更好，尤其 10 秒长时距；但在 1 秒处仍低于更大的 ViT-g。

COIN next-step forecasting 上，VL-JEPA 达到 56.2%，超过 VideoLLM-online 49.1、VideoLLM-MoD 49.7 与 ProVideLLM 53.6。

### 5.4 嵌入预测与 token 预测的受控对比

![](assets/VL-JEPA%20-%20语义嵌入预测式视觉语言模型/figures/03-embedding-vs-token.png)

> 图 3：在共享视觉编码器、训练数据、batch 和步数的对比中，VL-JEPA 更快提升 caption 与分类性能；右侧也显示 0.5B Predictor 与 1B LLM 的可训练参数差异，以及按需解码的延迟分解。  
> 来源：论文 Figure 3，第 7 页，https://arxiv.org/abs/2512.10942

实验共同使用：

- 冻结 Perception Encoder ViT-L/14；
- $336^2$ 分辨率，无 tiling；
- 每视频 16 帧；
- effective batch 128；
- 相同预训练数据与 LR scheduler。

区别是 VL-JEPA 用 0.5B Predictor 预测目标嵌入，VLM 用 1B LLM 做 next-token prediction。

| Samples seen | VL-JEPA caption CIDEr | VLM caption CIDEr | VL-JEPA 分类 Top-5 | VLM 分类 Top-5 |
|---:|---:|---:|---:|---:|
| 0.5M | 1.23 | 1.35 | 14.9 | 14.0 |
| 5M | 14.7 | 图中约 6–7 | 35.3 | 图中约 20–22 |
| 15M | **14.8** | **7.1** | **41.0** | **27.2** |

这组实验支持“嵌入目标更容易学习”的主张，但并非严格单变量实验：

1. Predictor 是 0.5B，VLM LLM 是 1B；
2. VL-JEPA 还依赖 Y-Encoder 和 Y-Decoder；
3. 两条输出路径不同；
4. 比较最多只到 15M samples，远小于最终 3.3B 训练规模。

因此它是有价值的系统级受控比较，但不能把全部增益唯一归因于 loss 空间。

### 5.5 Y-Encoder 的文本语义质量

| 模型 | SugarCrepe++ | VISLA |
|---|---:|---:|
| PE-Core | 58.6 | 38.3 |
| SigLIP2 | 56.5 | 40.4 |
| VL-JEPA BASE | **63.9** | **42.9** |
| VL-JEPA SFT | 58.4 | 39.5 |

BASE Y-Encoder 对属性、物体、关系和词汇扰动更稳健，说明联合训练确实塑造了文本语义空间；但 SFT 后两个基准都退化，表明问答/分类调优会损伤基础嵌入结构，统一模型仍存在灾难性遗忘和任务干扰。

---

## 6. 选择性解码

### 6.1 理想中的在线机制

方法章节给出的概念是：连续生成 $\hat S_Y(t)$，做局部平滑，监控滑动窗口方差；只有当语义变化超过阈值时才调用 Y-Decoder。这类似对语义流做 online change-point detection。

### 6.2 实验实际采用的算法

实验并未直接实现局部阈值触发，而是：

1. 先获得完整 embedding sequence；
2. 用带时间连通约束的 agglomerative clustering；
3. 以 Ward distance 把序列分成 $N$ 个连续片段；
4. 选每段中点作为 decoding point；
5. 从中点 embedding 或段内平均 embedding 解码。

因此实验算法是**离线、固定聚类数、看完整序列后分段**，不能直接等价于实时因果系统。

### 6.3 评估与结果

![](assets/VL-JEPA%20-%20语义嵌入预测式视觉语言模型/figures/04-selective-decoding.png)

> 图 4：左侧展示均匀采样与 embedding-guided 分段，右侧曲线显示在相同平均 CIDEr 下，选择性解码需要更少的文本解码操作；段内平均池化还能起到去噪作用。  
> 来源：论文 Figure 4，第 8 页，https://arxiv.org/abs/2512.10942

数据与协议：

- EgoExo4D validation；
- 218 个视频；
- 平均时长约 6 分钟；
- 每视频平均 143 个原子动作标注；
- 每个标注匹配时间上最近的已解码输出，再计算 CIDEr；
- 扫描 2.0Hz 到 0.01Hz，即平均 0.5 秒到 100 秒解码一次。

选择性解码在 0.35Hz（约 2.85 秒一次）时达到均匀 1Hz 解码的相近性能，因此文本解码操作约减少：

$$\frac{1}{0.35}\approx2.85\times$$

### 6.4 这 2.85× 应如何理解

它表示**达到相近 CIDEr 时所需的文本解码次数减少**，不是整体 FLOPs、端到端延迟或功耗直接下降 2.85×。持续运行的 X-Encoder、Predictor、embedding 平滑和聚类成本仍存在。

另外，最近时间匹配协议没有披露时间容差、一对一约束和检测延迟，一个解码结果可能服务多个附近标注；实验也没有报告误差条。因此应把结果理解为有希望的 Pareto 改善，而不是已经完成实时系统验证。

---

## 7. 消融实验：真正决定性能的因素

![](assets/VL-JEPA%20-%20语义嵌入预测式视觉语言模型/figures/05-ablation-table.png)

> 表 7：预训练、Y-Encoder 学习率、损失、Predictor 深度/初始化与 Y-Encoder 类型的完整消融。蓝色为默认配置。  
> 来源：论文 Table 7，第 9 页，https://arxiv.org/abs/2512.10942

### 7.1 Caption 预训练是分类与检索的基础

移除预训练后：

- 分类：49.0 → 27.3，下降 21.7；
- 检索：47.5 → 30.2，下降 17.3；
- VQA：46.1 → 42.5，只下降 3.6。

这说明大规模 caption 对齐主要建立通用跨模态空间；VQA 更依赖 SFT 数据与查询条件融合。

### 7.2 更多 Predictor 层主要改善 VQA

从 Layer 8–16 的 8 层设置扩展到 Layer 0–16：

- 分类 27.3 → 27.4；
- 检索 30.2 → 31.0；
- VQA 42.5 → 45.5。

视觉语言对齐对 Predictor 容量不太敏感，而查询条件下的复杂判别更依赖深层融合。

### 7.3 Llama 初始化并非普遍优势

去掉 Llama-3 初始化后：

- 分类略升 0.8；
- 检索略升 0.2；
- VQA 下降 1.9。

预训练语言权重主要帮助问答的语言条件建模；对分类和检索，随机 Transformer 也能通过大规模对比训练形成有效空间。

### 7.4 更大 Y-Encoder 存在任务权衡

更大的 Qwen3-Embedding 或视觉对齐 PE 文本编码器显著提升分类/检索，但 VQA 不一定提升：PE-Core-G 的分类 33.9、检索 32.0，却只有 41.8 VQA。EmbeddingGemma 在分类/检索较弱，但 VQA 42.5。

这说明“最好的句向量空间”不一定是“最好的问答目标空间”；分类/检索偏好全局可分性，VQA 需要更细的条件区分。

---

## 8. 与相关工作的关系

### 8.1 与 JEPA 系列

[[JEPA - 联合嵌入预测架构]]、[[V-JEPA - 视频潜空间预测表征学习]] 和 [[V-JEPA 2 - 自监督视频世界模型的理解预测与规划]] 主要预测图像/视频状态表示。VL-JEPA 将“预测目标表示而非重建原始数据”的原则扩展到视觉—语言目标：目标不是未来帧 latent，而是目标文本的语义 embedding。

它与世界模型式 JEPA 的相似点是避免重建不可预测细节；差别是 VL-JEPA 的 Predictor 还显式接收文本查询，目标空间由可训练文本编码器定义。

### 8.2 与 CLIP

[[CLIP - 自然语言监督的可迁移视觉模型]] 学习静态图文对齐，而 VL-JEPA 学习查询条件下的目标预测。可以把 VL-JEPA 看成“CLIP 共享空间 + 条件 Predictor + 按需 Decoder”，但它联合训练 Y-Encoder，使目标空间不再固定。

### 8.3 与生成式 VLM

生成式 VLM 的统一接口是 token；VL-JEPA 的统一接口是 embedding。前者天然适合开放生成、推理链和工具调用，后者天然适合分类、检索、候选决策和低频输出。二者不是简单替代关系，更合理的方向可能是：**用 embedding-space 模块持续理解和监控，用生成式模块在需要解释、规划或工具调用时介入。**

### 8.4 与连续潜空间推理

论文把 VL-JEPA 与 Large Concept Models、COCONUT 等 latent-space reasoning 联系起来，但当前模型只预测一个目标语义向量，并没有展示多步潜状态递推。它更接近“latent answer prediction”，而不是已经完成“latent chain-of-thought”。

---

## 9. 局限与批判性评价

### 9.1 论文明确承认的边界

作者在结论中说明，现阶段目标不是普遍替代 VLM；reasoning、tool use 和 agentic behavior 等生成式 VLM 擅长的能力尚未评估，参数与数据规模扩展也未充分探索。

### 9.2 Y-Decoder 缺失让开放生成结论不完整

这是最严重的复现问题。若没有 Decoder 架构、训练数据与推理配置，就无法判断：

- caption 质量由 Predictor 还是 Decoder 决定；
- “轻量”究竟是多少参数与 FLOPs；
- Decoder 是否用了额外语料或蒸馏；
- BASE/SFT 的 1.6B 是否包含 Decoder；
- 与 VLM 的文本生成比较是否公平。

### 9.3 “50% 更少可训练参数”口径模糊

摘要的 50% 主要对应受控实验中 0.5B Predictor 对 1B LLM；但主模型还联合训练 300M Y-Encoder 与 projection heads。若全部计入，训练参数差距不会简单等于 50%。

### 9.4 “严格受控”不是严格单变量实验

两边不仅目标函数不同，模型宽度、参数量、输出头和解码路径也不同。更严格的验证应固定同一 Transformer 参数规模，只替换 CE token 目标与 InfoNCE embedding 目标，并分别控制 Y-Encoder / Decoder 的额外参数与数据。

### 9.5 推理延迟结论强于证据

Figure 3 显示真正生成文本时，两者总延迟是同一量级；VL-JEPA 的优势来自可以跳过 Decoder，而不是每次生成文本都更快。论文结论中的“significantly lower inference latency”应限定为低解码频率的在线监控场景。

### 9.6 选择性解码实验不是在线因果评估

实验使用完整序列上的时序层次聚类和段内平均，会访问未来帧。真正部署需要在线 change-point detection、延迟约束、误触发/漏触发指标与有限缓存；这些尚未验证。

### 9.7 InfoNCE 可能把语义等价样本当负样本

论文的动机是把同义答案拉近，但普通 in-batch InfoNCE 会把 batch 中其他样本一律视作负样本。如果两个不同视频恰好对应语义等价 caption，它们反而被推开。大 batch 24k 会增加 hard negatives，也会增加 false negatives。论文未说明去重、多正样本对比或软标签策略。

### 9.8 单向量目标可能成为信息瓶颈

把最长 512 token 的目标文本压成单个 1,536 维向量，对短 caption 或候选答案可能足够；对长解释、复杂关系、多步推理和精细实体绑定，单向量可能难以保留生成所需全部信息。多向量目标、分层 latent 或可变长度 concept sequence 可能更合理。

### 9.9 数字与文本存在内部不一致

- 24k batch × 100k steps 等于 2.4B，正文写 2B；
- 表中 BASE 是 3.3B samples，正文一处称 BASE 只有 3.6B，后者实际更像 SFT；
- SFT 耗时文本“2.5s days”存在排版或笔误；
- 参数账无法从已披露组件闭合到 1.6B。

这些问题不推翻主结论，但降低了复现可信度。

---

## 10. 复现与实践建议

### 10.1 论文级复现的现实成本

截至 2026-10-27，官方代码、推理代码与权重均未发布；两个作者 HF 仓库为空占位。即使按论文重新实现，也需要：

- V-JEPA 2 ViT-L 权重；
- Llama-3.2-1B 的层级裁剪与非因果改造；
- EmbeddingGemma-300M；
- DataComp、YFCC-100M、Action100M / HowTo100M；
- 大 batch 跨卡对比学习；
- 未公开的 Y-Decoder；
- 约 192 张 H200、4 周级别的预训练资源。

因此当前只能做结构性复现，无法可靠复现论文最终指标。

### 10.2 最小研究路径

如果目标是验证“embedding prediction 是否优于 token prediction”，更合理的最小实验是：

1. 固定同一个视觉编码器和同一个 8 层 Transformer；
2. 保持参数量、数据、batch、优化器和输入完全一致；
3. 分别使用：
   - token CE 头；
   - 冻结文本 encoder + cosine；
   - 可训练文本 encoder + InfoNCE；
4. 同时评估 caption、候选分类、检索和文本硬负样本；
5. 记录全部可训练参数和文本 Decoder 的额外成本；
6. 用真正在线的 change-point detection 替代离线聚类。

### 10.3 对现有系统的工程启发

最值得迁移的不是完整 VL-JEPA 训练，而是**分层推理架构**：

- 第一层：视觉 encoder + 轻量 semantic predictor 持续运行；
- 第二层：基于语义变化做事件触发；
- 第三层：只有在需要人类沟通、复杂推理或工具调用时启用大语言模型。

这与 always-on 智能眼镜、机器人世界状态监控、视频检索和长视频摘要非常契合。

---

## 11. 代码级实现（第三方复现，非官方）

> [!warning] 可信度边界
> 本节审阅的是第三方仓库 `Soumya-Chakraborty/VL-JEPA`，commit `7400361ecbe520504f285d4c7e906e1205b68469`。它只能帮助理解一种工程化落地方式，**不能代表论文官方实现，也不能用于复核论文指标**。仓库未提供 LICENSE。

### 11.1 仓库概览与模块映射

| 论文模块 | 第三方代码入口 | 实际行为 | 产物 |
|---|---|---|---|
| X-Encoder | `vljepa/models/vljepa.py::VisionEncoder` | torchvision ViT / toy CNN / HF backbone；默认逐帧编码并把空间 patch 平均 | 每帧一个向量 |
| Query embedding | `VLJEPA.__init__` / `predict_embedding` | 加载 Llama AutoModel，但前向只使用 input embedding | query token embedding |
| Predictor | `nn.TransformerEncoder` | 随机初始化的通用 Transformer，无 causal mask | query-conditioned 表示 |
| Y-Encoder | `encode_target_text` | HF AutoModel + 手工 masked mean pooling | 目标文本向量 |
| Projection | `pred_proj` / `y_proj` | 映射到共享维度 | 默认 1,536 维 |
| Loss | `vljepa/models/losses.py` | batch-local 双向 InfoNCE | scalar loss |
| Y-Decoder | `NearestNeighborDecoder` | 固定 text bank 最近邻，不生成新文本 | bank 中已有字符串 |
| Selective decoding | `vljepa/inference/selective.py` | Ward + temporal connectivity 的离线固定 $N$ 聚类 | 片段代表 embedding |

训练入口：

- `scripts/train_pretrain.py`；
- `scripts/train_sft.py`；
- `scripts/run_inference.py` 支持 caption、discriminative VQA 和 selective 三种模式。

### 11.2 第三方默认配置的全链路张量总表

第三方 `pretrain.yaml` 默认：$B=32,T=8,H=W=256$，Predictor width 1024、8 层、16 heads、FFN 4096，query/target 最长 512，shared dim 1536。

| # | 模块 | 关键输入 | 关键输出 | 备注 |
|---:|---|---|---|---|
| 1 | DataLoader | 32 个图像/视频 | `[32,8,3,256,256]` | 图像复制 8 次，视频均匀采 8 帧 |
| 2 | 每帧视觉编码 | `[256,3,256,256]` | `[256,F]` | 时间维先并入 batch |
| 3 | 空间池化与投影 | 每帧 patch token | `[32,8,1024]` | 每帧只保留 1 个向量 |
| 4 | Query tokenizer | 32 个字符串 | `[32,L_q]`，$L_q\le512$ | 动态 padding，不是固定补到 512 |
| 5 | Query embedding | `[32,L_q]` | `[32,L_q,1024]` | Llama input embedding 后线性投影 |
| 6 | 拼接 | `[32,8,1024]` + `[32,L_q,1024]` | `[32,8+L_q,1024]` | 无 causal mask |
| 7 | Predictor | 上述序列 | 同形状 | 8 层通用 Transformer |
| 8 | Query pooling | query 段 `[32,L_q,1024]` | `[32,1024]` | 只池化 query token |
| 9 | Predictor projection | `[32,1024]` | `[32,1536]` | 预测目标嵌入 |
| 10 | Y-Encoder | `[32,L_y]` | `[32,L_y,d_y]` | HF AutoModel hidden states |
| 11 | Y pooling/projection | `[32,L_y,d_y]` | `[32,1536]` | masked mean + linear |
| 12 | InfoNCE | 两个 `[32,1536]` | logits `[32,32]` | 每样本仅 31 个本地负样本 |
| 13 | 最近邻“Decoder” | `[1,1536]` 与 bank `[M,1536]` | `[1,M]` 相似度 | 返回 bank 文本，非开放生成 |

注意：stock torchvision 预训练 `vit_l_16` 通常要求 224×224，而第三方默认配置 resize 到 256，并调用私有 `_process_input`，因此 paper-style 默认路径很可能在输入尺寸检查处失败。仓库真正可离线 smoke-test 的是 tiny 配置。

### 11.3 Tiny 路径的具体张量流

Tiny 配置为 $B=2,T=2,64\times64$ 图像、Predictor width 64、2 层、4 heads、shared dim 64：

| 张量 | 形状 | 构造方式 |
|---|---|---|
| 输入 batch | `[2,2,3,64,64]` | 两个样本、每个重复两帧 |
| 拉平帧 | `[4,3,64,64]` | batch 与时间合并 |
| toy CNN 输出 | `[4,64]` | 两次 stride-2 conv + global pool |
| 视觉序列 | `[2,2,64]` | 恢复时间维 |
| query ids | `[2,5]` | 4 个词 + EOS |
| raw query embedding | `[2,5,128]` | ToyTextEncoder |
| query projection | `[2,5,64]` | 线性投影 |
| Predictor 输入 | `[2,7,64]` | 2 个视觉向量 + 5 个 query token |
| Predictor 输出 | `[2,7,64]` | 2 层双向 Transformer |
| query mean pool | `[2,64]` | 仅取 query 段 |
| target ids | `[2,3]` | 2 个词 + EOS |
| target hidden | `[2,3,128]` | ToyTextEncoder |
| target projection | `[2,64]` | mean pool + linear |
| InfoNCE logits | `[2,2]` | 预测与目标两两相似度 |
| text bank | `[3,64]` | red/green/blue square |
| 推理相似度 | `[1,3]` | 只能输出三句之一 |

### 11.4 逐模块差异

#### 11.4.1 X-Encoder

**输入输出**：代码将 `[B,T,3,H,W]` 拉平为 `[B\cdot T,3,H,W]`，逐帧编码后恢复成 `[B,T,H_p]`。

**为什么这么设计**：实现简单，可让图像与视频共享接口，并降低时空 token 数。

**关键超参**：默认 `vit_l_16`、8 帧、256 分辨率、冻结 backbone。

**与论文差异**：不是 V-JEPA 2；没有视频时空注意力；空间 patch 在进入 Predictor 前全部平均；新建的 vision projection 位于 X-Encoder 内，却被连同 backbone 一起冻结，随机投影无法学习。

#### 11.4.2 Query 与 Predictor

**输入输出**：query token 经 embedding 和线性投影，与每帧视觉向量拼接，通过 8 层 Transformer，再只对 query 段做平均池化。

**为什么这么设计**：保留“视觉 + 查询双向融合 → 单个语义向量”的高层逻辑。

**关键超参**：width 1024、8 层、16 heads、FFN 4096、query max 512。

**与论文差异**：加载完整 Llama-3.2-1B，却只用其 input embedding；Predictor 是随机初始化的 `nn.TransformerEncoder`，没有从 Llama 后 8 层复制权重，也没有 RMSNorm、RoPE、SwiGLU、GQA 或显式位置编码。完整 Llama 其余层常驻内存但不参与前向。

#### 11.4.3 Y-Encoder

**输入输出**：HF AutoModel 输出 `[B,L_y,d_y]`，masked mean pool 后投影到共享维度。

**为什么这么设计**：以统一接口兼容 EmbeddingGemma 和 toy encoder。

**关键超参**：默认 EmbeddingGemma、max 512、Y-Encoder LR multiplier 0.05。

**与论文差异**：没有调用 EmbeddingGemma 的完整句向量 pooling / dense / prompt 接口，只把通用 hidden states 手工平均；`y_proj` 也没有进入 0.05 LR 参数组，而是使用普通学习率。

#### 11.4.4 InfoNCE

**输入输出**：两个 `[B,D]` 归一化向量矩阵相乘得到 `[B,B]` logits，双向 cross-entropy 取平均。

**关键超参**：温度固定 0.07。

**与论文差异**：只有当前进程 batch negatives，没有分布式 all-gather；默认 batch 32 与论文 24k 有巨大差距；温度 0.07 是第三方选择，论文未披露。

#### 11.4.5 Y-Decoder 与选择性解码

**输入输出**：所谓 Decoder 只是将预测向量与固定 text bank 比较，返回最近候选；选择性模块对完整 embedding stream 做固定 $N$ 的 Ward 聚类。

**为什么这么设计**：在没有官方 Y-Decoder 的情况下，提供可运行的 closed-set 演示。

**与论文差异**：不能生成 bank 外的新句子；无 Decoder 训练；caption/chat 本质是检索；selective 不是在线阈值触发。README 示例默认 8 帧却要求 12 segments，代码会直接退化为每帧都返回，实际上不减少解码点。

### 11.5 第三方计算账与复现建议

默认第三方预训练：

- 100k steps × batch 32 = 3.2M 样本呈现；
- 每样本 8 帧，共 25.6M 帧呈现；
- Predictor 序列最长 $8+512=520$；
- InfoNCE 矩阵只有 $32\times32$；
- 主要开销来自逐帧 ViT、Predictor 和 Y-Encoder。

最小可运行路径应使用仓库自带 tiny checkpoint，而不是 paper-style 默认：

```bash
PYTHONPATH=. python scripts/run_inference.py \
  --config vljepa/configs/inference_tiny.yaml \
  --checkpoint outputs/tiny_pretrain/step_0000002.pt \
  --manifest data/tiny_infer_manifest.jsonl \
  --text-bank data/tiny_text_bank.txt \
  --mode caption
```

这条路径只验证数据流、InfoNCE 学习与最近邻输出，不能验证论文性能。

第三方仓库还存在若干工程风险：

1. 无 DDP/FSDP 与跨卡负样本；
2. 无 validation、early stopping 和真正的 resume CLI；
3. checkpoint 加载普遍使用 `strict=False`，架构不匹配可能静默留下随机参数；
4. 推理入口部分路径没有 `torch.no_grad()`；
5. SFT 读取候选答案，但训练循环不使用候选，训练逻辑与预训练几乎相同；
6. 仓库没有 LICENSE，不建议直接二次分发代码。

---

## 12. 个人启发与后续问题

### 12.1 对 VLM 的核心启发

VL-JEPA 的真正挑战不是“生成是否应该消失”，而是“生成是否应该始终在线”。对大量视觉任务，系统可以先维护一个低延迟、连续更新的语义状态，只有在需要沟通、解释或规划时才让生成模型介入。这比让一个大 VLM 每秒反复写句子更符合实时系统的计算结构。

### 12.2 与世界模型研究的连接

VL-JEPA 把目标文本视为世界状态或动作概念的语义读出。如果进一步让 Predictor 在 latent space 中递归预测未来状态，并让语言只作为查询和读出接口，就可能形成“视觉世界模型 + 语言语义层”的组合，而非把所有世界知识都塞进 token generation。

### 12.3 值得继续验证的问题

1. 在完全相同的 Predictor 参数量下，只替换 CE 与 InfoNCE，优势是否仍成立？
2. 官方 Y-Decoder 如何训练，是否引入额外数据或蒸馏？
3. 使用多正样本或去重 InfoNCE，能否解决同义 caption 被当负样本的问题？
4. 把单个 1,536 维向量扩展为可变长度 concept tokens，能否覆盖长答案与多步推理？
5. 用真正因果的 change-point detection 后，2.85× 优势还能保留多少？
6. 如何同时保持 BASE 的文本硬负样本能力与 SFT 的 VQA 能力？
7. embedding-space 预测能否作为 VLA / 机器人策略的语义中间层，而不是只输出文本？

---

## 13. 最终评价

| 维度 | 评价 |
|---|---|
| 问题重要性 | 高：理解与 token 生成长期绑定确实造成训练和在线推理浪费 |
| 方法新颖性 | 高：把通用视觉语言任务系统化改写为 query-conditioned target embedding prediction |
| 受控实验说服力 | 中高：小规模趋势清晰，但不是严格单变量比较 |
| 大模型结果 | 中高：分类、检索和状态变化匹配强，但 VQA 覆盖有限且 SFT 含 in-domain 数据 |
| 选择性解码 | 概念强、验证中等：2.85× 有吸引力，但实验是离线聚类而非在线因果机制 |
| 可复现性 | 低：官方代码/权重为空，Y-Decoder 与多项关键细节缺失 |
| 长期影响判断 | 值得关注：更可能与生成式 VLM 形成分层系统，而不是短期全面替代 |

**最终判断**：VL-JEPA 是一篇“方向价值高于当前完成度”的工作。它成功把 JEPA 的抽象原则迁移到视觉语言任务，并用实验说明语义嵌入可以成为分类、检索、候选决策和低频文本输出的统一接口；但开放生成、在线因果触发和通用推理仍没有形成闭环。最合理的研究结论不是“token generation 已经过时”，而是：**在实时多模态系统里，token generation 应从默认主循环降级为按需服务。**

---

## 14. 材料来源

- 论文：https://arxiv.org/abs/2512.10942
- 官方 PDF：https://arxiv.org/pdf/2512.10942v2
- DOI：https://doi.org/10.48550/arXiv.2512.10942
- OpenReview：https://openreview.net/forum?id=tjimrqc2BU
- 作者主页：https://chendelong.world/
- 第三方代码：https://github.com/Soumya-Chakraborty/VL-JEPA
- 本地来源清单：`assets/VL-JEPA - 语义嵌入预测式视觉语言模型/sources.md`
