---
type: paper-note
status: complete
domain: World-Models
paper: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
year: 2023
arxiv: 2301.08243
source: https://arxiv.org/abs/2301.08243
project: https://facebookresearch.github.io/ijepa/
code: https://github.com/facebookresearch/ijepa
code_commit: 52c1ae95d05f743e000e8f10a1f3a79b10cff048
tags: [World-Models, JEPA, Self-Supervised, Energy-Model, Representation-Learning]
created: 2026-08-27
updated: 2026-10-31
---

# I-JEPA：联合嵌入预测架构

## 基本信息

- **作者**：Mahmoud Assran*、Quentin Duval、Ishan Misra、Piotr Bojanowski、Pascal Vincent、Michael Rabbat、Yann LeCun、Nicolas Ballas（* 第一作者）
- **机构**：Meta AI（FAIR），部分作者兼属 McGill University、Mila（Quebec AI Institute）、New York University
- **发表**：CVPR 2023（arXiv 2301.08243）
- **原文 PDF**：<https://arxiv.org/pdf/2301.08243>
- **官方代码**：<https://github.com/facebookresearch/ijepa>
- **概念源头**：LeCun 2022 纲领性论文《A Path Towards Autonomous Machine Intelligence》（OpenReview: <https://openreview.net/forum?id=BZ5a1r-kVsf>）首次提出 JEPA（Joint-Embedding Predictive Architecture）这一世界模型构想

## 一句话结论

I-JEPA 用「在表征空间预测、而非像素空间重建」的 JEPA 目标，在完全不用手工数据增强的前提下，以 ViT-H/14 在 ImageNet-1K 线性评估上达到 79.3%（预训练仅需 16 块 A100 约 72 小时，比 MAE 快 10 倍以上），在 1% 标签半监督上达到 73.3%，学到比像素重建法（MAE）更语义、比不变性法（SimCLR/DINO）保留更多局部结构的可迁移表征——它是 LeCun JEPA 世界模型纲领的图像实例，与生成式视频世界模型（VPP/SVD 路线）构成互补。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

自监督学习要回答一个根本问题：**一个系统如何在不依赖标签的情况下，从输入信号之间的关系中自动学到有用的表征**。在计算机视觉里，这个问题的答案长期以来被两条路线垄断——以 SimCLR、DINO 为代表的「不变性」（invariance-based）方法，以及以 MAE、BEiT 为代表的「生成式」（generative）方法。I-JEPA 的出发点是把这两条路线的缺陷都绕开：不变性方法需要一堆手工设计的数据增强（缩放、裁剪、颜色抖动）来构造「同一图像的多个视角」，这些增强带有强偏置，难以泛化到不同抽象层次的迁移任务，也无法迁移到音频等其他模态；生成式方法虽然不需要增强，但它强迫模型去重建被掩码的像素或 token 细节，把大量容量浪费在纹理、背景等与语义无关的细节上，最终学到的表征语义层次偏低，离线评估（如 linear probing）表现不如不变性方法。

### 1.2 现有方法的瓶颈

I-JEPA 把这两条路线的病根各打一板：

- **不变性方法（JEA，Joint-Embedding Architecture）**：核心挑战是「表征塌缩」（representation collapse）——能量面变得平坦，编码器无论输入什么都输出常数。过去几年靠对比损失推开负样本（SimCLR、MoCo）、非对比损失最小化冗余（Barlow Twins、VICReg）、聚类法最大化平均嵌入熵（SwAV、DINO）、或非对称架构（BYOL、SimSiam）来防塌缩。但即便防住了塌缩，它仍然离不开手工增强，且学到的表征被「全局不变性」偏置主导，丢掉了对目标计数、深度估计等低级/稠密任务至关重要的局部结构。
- **生成式方法（Generative Architecture）**：用掩码自编码器（MAE）重建缺失像素，虽然不需要增强、易扩展到多模态，但像素重建目标逼着模型去恢复那些「不可预测的细节」，学到的表征语义层次低，在 linear probing 和少标签迁移上明显弱于不变性方法，往往需要端到端微调才能发挥威力。

I-JEPA 想回答的，是「能不能在不引入手工增强、也不强迫重建像素的前提下，学到高语义表征」——也就是同时拿到两条路线各自的优点。

### 1.3 JEPA 概念源头：LeCun 的世界模型纲领

I-JEPA 不是凭空冒出来的，它是 Yann LeCun 在 2022 年纲领性文章《A Path Towards Autonomous Machine Intelligence》里提出的 JEPA 构想的第一份图像落地。LeCun 的核心判断是：**智能系统不该学「重建世界的像素」，而该学「预测世界的抽象表征」**——因为世界的真实状态里充满了大量不可预测、又与语义无关的细节，一个聪明的世界模型应当只去预测「可预测的那部分」抽象内容，把不可预测的细节主动丢弃。这篇纲领把这一思想命名为 Joint-Embedding Predictive Architecture，I-JEPA 就是它在图像模态、用掩码机制实现的最小可用版本。

### 1.4 本文核心贡献

1. 提出 I-JEPA：**非生成式**的图像自监督方法，从单个 context 块预测同一图像中多个 target 块的表征，损失只算在表征空间；
2. 提出**多块掩码策略**（multi-block masking）：用足够大（语义级）的目标块 + 足够信息量（空间分散）的上下文块，是引导表征走向语义的关键设计；
3. 用 EMA 更新的 target encoder 产生目标表征，用非对称架构 + 潜空间预测天然规避表征塌缩；
4. 实证上：I-JEPA 无增强即超越 MAE 的 linear / 1% 半监督 / 语义迁移表现，在目标计数与深度预测上反超不变性方法，且收敛比像素重建快约 5 倍。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：单张图像 $y$，切成 $N$ 个不重叠 patch；再按掩码策略分成一个 context 块 $x$ 和 $M$ 个 target 块；
- **输出**：context encoder 对 context 块的表征 $s_x$，以及 predictor 对每个 target 块预测出的 patch 级表征 $\hat{s}_y(i)$；
- **核心假设**：如果只让模型在抽象表征空间里「补齐」缺失区域，那么它被迫学到的就是那些真正可预测、跨位置共享的语义规律，而不是像素纹理。这个假设是整个 JEPA 世界模型哲学在图像上的缩影。

### 2.2 关键符号与目标函数

设 context encoder 为 $f_\theta$、target encoder 为 $f_{\bar\theta}$、predictor 为 $g_\phi$。目标：给定 context 块 $x$，预测 target 块的表征。损失就是预测表征与目标表征之间的平均 L2 距离：

$$\mathcal{L} = \frac{1}{M}\sum_{i=1}^{M}D\big(\hat{s}_y(i),\, s_y(i)\big) = \frac{1}{M}\sum_{i=1}^{M}\sum_{j \in B_i}\Big\lVert \hat{s}_{y_j} - s_{y_j}\Big\rVert_2^2$$

其中 $\hat{s}_y(i) = g_\phi(s_x, \{m_j\}_{j \in B_i})$ 是 predictor 以位置掩码 token 为条件做出的预测，$s_y(i)$ 是 target encoder 输出的目标表征。$\theta$、$\phi$ 走梯度优化，$\bar\theta$ 走 EMA 更新：

$$\bar\theta \leftarrow m\,\bar\theta + (1-m)\,\theta, \qquad m: 0.996 \to 1.0$$

---

## 3. 核心方法

### 3.1 三大自监督架构的对比

I-JEPA 的定位，最好放进一个统一的「能量模型」（Energy-Based Models）框架里看：自监督的目标可以表述为「给不兼容的输入赋高能量、给兼容的输入赋低能量」。据此，作者把自监督方法分成三类（Figure 2）：

![](assets/JEPA%20-%20联合嵌入预测架构/figures/01-architectures.png)

> **图 1（论文 Figure 2）：自监督三大架构对比。** (a) 联合嵌入架构（JEA）：让兼容输入 $x, y$ 输出相似嵌入、不兼容输入输出相异嵌入，典型代表是 SimCLR/DINO，靠「不变性」学习，风险是表征塌缩；(b) 生成式架构：用一个 decoder 从兼容信号 $x$ 直接重建信号 $y$，典型代表是 MAE，靠「像素重建」学习，风险是把容量浪费在细节上；(c) 联合嵌入预测架构（JEPA）：用一个 predictor 从 $x$ 预测 $y$ 的**嵌入**而非 $y$ 本身，损失只算在嵌入空间。I-JEPA 走的是第三条路——既有 JEA 的语义高度，又像生成式一样不需要手工增强。
> 来源：论文 Figure 2，第 2 页，<https://arxiv.org/abs/2301.08243>

### 3.2 总体框架与三件套

I-JEPA 的整体结构（Figure 3）可以拆成三个组件：context encoder、target encoder、predictor。三者都用 ViT 架构，但分工不同——context encoder 负责读可见的上下文、target encoder 负责产出「学习目标」、predictor 负责在潜空间里做预测。

![](assets/JEPA%20-%20联合嵌入预测架构/figures/02-method.png)

> **图 2（论文 Figure 3）：I-JEPA 方法示意。** 从单张图像里抽出一个大的 context 块（喂给 context encoder，得到 $s_x$），再抽若干个 target 块（喂给 target encoder，得到目标表征 $s_y$）；predictor 以 context 表征 $s_x$ 为输入、以彩色位置 token 为条件，预测每个 target 块的表征 $\hat{s}_y$；损失是 $\hat{s}_y$ 与 $s_y$ 之间的 L2。target encoder 的权重是 context encoder 权重的指数滑动平均（EMA）。注意：target 块是对 **target encoder 的输出**做掩码得到的，而不是对输入图像做掩码——这一点至关重要，它保证了目标表征处于较高的语义层次。
> 来源：论文 Figure 3，第 3 页，<https://arxiv.org/abs/2301.08243>

### 3.3 Target：目标块与目标编码器

Target 是 I-JEPA 里要预测的东西，对应图像块的**表征**。给定输入图像 $y$，先切成 $N$ 个 patch，整体过 target encoder 得到 patch 级表征 $s_y = \{s_{y_1}, \dots, s_{y_N}\}$。然后从 $s_y$ 里随机采样 $M$ 个（可重叠的）块，第 $i$ 个块的掩码记为 $B_i$，其表征集合记为 $s_y(i) = \{s_{y_j}\}_{j \in B_i}$。默认 $M=4$，块的宽高比在 $(0.75, 1.5)$、尺度在 $(0.15, 0.2)$ 之间随机采样。

这里有一个非常反直觉但关键的设计：**掩码作用在 target encoder 的输出上，而不是输入上**。如果掩码作用在输入上，每个 target 区域都要单独过一遍 target encoder 前向，而且学到的目标表征语义层次会变低。消融实验（Table 11）验证了这一点：ViT-H/16 上「输出掩码」67.3% vs「输入掩码」56.1%。target encoder 用 EMA 更新（momentum 从 0.996 线性升到 1.0），与 BYOL、DINO 的做法一致，这是 JEA 类方法用 ViT 训练时防塌缩的必备技巧，I-JEPA 同样发现它必不可少。

### 3.4 Context：上下文块与上下文编码器

Context 是 I-JEPA 的「观察窗口」。从图像里随机采样**一个**大的 context 块，尺度在 $(0.85, 1.0)$、宽高比为单位（即接近整图的方形大块）。由于 target 块与 context 块是独立采样的，两者可能有显著重叠；为了保证预测任务非平凡，作者从 context 块里**去掉与所有 target 块重叠的区域**。之后，掩码后的 context 块 $x$ 过 context encoder，得到 patch 级表征 $s_x = \{s_{x_j}\}_{j \in B_x}$。

这个「大而稀疏」的 context 设计有两层用意：一是大块能提供充分的上下文信息，二是去掉 target 重叠区域后 context 变得稀疏、计算高效。Figure 4（论文原文）展示了实际的 context/target 掩码样例：context 是几乎覆盖全图、但挖掉几个洞的稀疏大块，target 是 4 个相对语义的小块。

### 3.5 Predictor：潜空间预测

Predictor $g_\phi$ 是 I-JEPA 里唯一「做预测」的组件。对第 $i$ 个 target 块，它把 context encoder 的输出 $s_x$ 和该 target 块对应的掩码 token $\{m_j\}_{j \in B_i}$ 一起作为输入，输出 patch 级预测 $\hat{s}_y(i) = \{\hat{s}_{y_j}\}_{j \in B_i} = g_\phi(s_x, \{m_j\}_{j \in B_i})$。掩码 token 由共享的可学习向量加位置编码构成，用来告诉 predictor「我要预测图像里哪个位置」。因为要预测 $M$ 个 target 块，predictor 会被应用 $M$ 次，每次以不同的位置 token 为条件。

Predictor 被刻意设计成**窄（narrow）ViT**：embedding 维度固定为 384（远小于 backbone 的 1024），深度按 backbone 大小取 6（ViT-B）/ 12（ViT-L/H）/ 16（ViT-G），自注意力头数与 backbone 一致。这个宽度瓶颈是有意为之——消融显示 384 宽度的 predictor 反而比 1024 宽度的更好（70.7 vs 68.4），因为窄 predictor 迫使信息在预测时被压缩，避免它「抄近路」。

### 3.6 损失函数与训练细节

损失就是 2.2 节那个平均 L2。值得注意的是，I-JEPA **不做归一化、不做对比、不做蒸馏式的 softmax 目标**，就是一个朴素到极点的回归——这在 2023 年的自监督图景里相当另类。它之所以能 work，靠的是「表征空间预测」这个结构性选择本身：predictor 只需让预测表征逼近目标表征，而目标表征又被 EMA 的 target encoder 持续「提纯」。

预训练细节（Appendix A）：用 AdamW 优化 context encoder 和 predictor；默认 batch size 2048；学习率先从 $10^{-4}$ 线性升到 $10^{-3}$（前 15 个 epoch），再按余弦退火降到 $10^{-6}$；weight decay 从 0.04 线性升到 0.4。I-JEPA **不使用 [cls] token**，评测时对 target encoder 输出做 average pooling 得到全局图像表征。

### 3.7 多块掩码策略（multi-block masking）

这是 I-JEPA 除「潜空间预测」之外的第二大方法论贡献。作者系统对比了四种掩码策略（Table 6，ViT-B/16、300 epoch、1% ImageNet 线性评估）：

| 掩码类型 | Target 形态 | Context 形态 | Top-1 |
|---|---|---|---|
| multi-block | 4 个 block(0.15, 0.2) | block(0.85, 1.0) × 补集 | **54.2** |
| rasterized | 3 个象限 | 补集 | 15.5 |
| block | 1 个 block(0.6) | 补集 | 20.2 |
| random | 随机 patch(0.6) | 补集 | 17.6 |

多块策略相对其他策略是碾压级的（54.2 vs 最高 20.2）。进一步的尺度消融（Appendix C）揭示了两个规律：**预测大块**（语义级 target）比预测小块好——target 尺度从 $(0.075, 0.2)$ 的 19.2 一路升到 $(0.15, 0.2)$ 的 54.2，再放大到 $(0.2, 0.3)$ 又回落到 33.6；**上下文要够信息量**——context 尺度从 $(0.40, 1.0)$ 的 31.2 升到 $(0.85, 1.0)$ 的 54.2；**target 数量越多越好**——从 1 个块的 9.0 升到 4 个块的 54.2。一句话：要预测「几个足够大的语义块」，同时给它「一个足够分散的上下文」。

### 3.8 与 MAE / SimCLR 的结构差异

把 I-JEPA 放到坐标轴上看最清楚：

- **vs MAE（生成式）**：MAE 的 encoder 也只看可见 patch，decoder 重建被掩码的**像素**；I-JEPA 的 predictor 预测的是 target 块的**表征**。结构上很像（都是 ViT + 掩码），但损失空间完全不同——这决定了 I-JEPA 学到的表征语义层次更高。代价是每次迭代要算 target encoder 前向，约比 MAE 慢 7%，但因为收敛快约 5 倍，总计算量反而大幅下降。
- **vs SimCLR / DINO（不变性）**：它们需要一张图构造多个增强视角、每个视角都过 encoder，且用对比/蒸馏/聚类等显式或隐式正则防塌缩；I-JEPA 每张图**只处理一个视角**（单个 context 块），靠 EMA target encoder 的非对称架构防塌缩，完全抛弃手工增强。

---

## 4. 数据集与实验设置

### 4.1 数据集

- **预训练**：ImageNet-1K（约 128 万张图）；缩放实验中还用 ImageNet-22K（约 1400 万张）验证数据规模收益；
- **下游分类**：ImageNet-1K（linear probe 与 1% 半监督）、CIFAR100、Places205、iNaturalist18（线性迁移）；
- **低级/稠密任务**：Clevr/Count（目标计数）、Clevr/Dist（深度估计），用线性探针评测，考察表征是否保留局部结构。

### 4.2 预训练设置

ViT 家族作为 context/target encoder，predictor 为窄 ViT（宽度 384）；batch size 2048；AdamW + 余弦退火；target encoder EMA momentum 0.996→1.0。默认分辨率 224×224，另有一个 ViT-H/16 在 448×448 分辨率下预训练的变体（记为 ViT-H/16@448）。所有 I-JEPA 模型**不使用任何视角数据增强**，这是全文反复强调的核心卖点。

### 4.3 下游评测协议

- **Linear evaluation**：冻结 backbone，只训一个线性分类头；I-JEPA 无 [cls] token，改用 average-pooled patch 表征（或最后 4 层 average-pool 的拼接）喂给线性头；ImageNet 上用 LARS、batch 16384、50 epoch 训练线性探针；
- **1% 半监督**：只用 ImageNet 约 1% 标签（每类约 12–13 张图），用 MAE 的微调协议（AdamW、50 epoch、layer decay 0.75、label smoothing 0.1），但不加 mixup/cutmix/random erasing；
- **迁移**：按 VISSL 默认配方，对 Clevr 任务只用 center crop + 水平翻转（避免随机裁剪删掉要计数的物体）。

---

## 5. 实验结果

### 5.1 ImageNet-1K 线性评估（Table 1）

在无增强方法里，I-JEPA 全面领先：

| 方法 | 架构 | Epochs | Top-1 |
|---|---|---|---|
| MAE | ViT-H/14 | 1600 | 77.2 |
| CAE | ViT-L/16 | 1600 | 78.1 |
| data2vec | ViT-L/16 | 1600 | 77.3 |
| I-JEPA | ViT-H/14 | **300** | **79.3** |
| I-JEPA | ViT-H/16@448 | 300 | **81.1** |

用 300 个 epoch 就超过了 MAE 用 1600 个 epoch 的结果；放大到 448 分辨率后，81.1% 甚至**追平了依赖手工增强的不变性方法 iBOT（ViT-L/16，81.0%）**。这直接回答了 I-JEPA 的核心主张：不用增强也能达到不变性方法的语义高度。

### 5.2 1% 标签半监督（Table 2）

| 方法 | 架构 | Epochs | Top-1 |
|---|---|---|---|
| MAE | ViT-H/14 | 1600 | 71.5 |
| data2vec | ViT-L/16 | 1600 | 73.3 |
| DINO | ViT-B/8 | 300 | 70.0 |
| MSN | ViT-B/4 | 300 | 75.7 |
| I-JEPA | ViT-H/14 | **300** | **73.3** |
| I-JEPA | ViT-H/16@448 | 300 | **77.3** |

I-JEPA ViT-H/14 以更少 epoch 匹配了 data2vec ViT-L/16；448 分辨率下 77.3% 超过包括 MSN、DINO、iBOT 在内的所有依赖增强的方法。半监督是自监督表征「可用性」最直接的试金石，这个结果说明 I-JEPA 学到的是真正可迁移的语义，而不是靠增强刷出来的假象。

### 5.3 迁移学习：分类与低级任务（Table 3/4）

- **分类迁移**（Table 3）：I-JEPA ViT-H/14 在 CIFAR100 87.5 / Places205 58.4 / iNat18 47.6，全面碾压同属无增强的 MAE（77.3 / 55.0 / 32.9）和 data2vec，并在 CIFAR100、Places205 上反超了 DINO（84.9 / 57.9）；
- **低级任务**（Table 4）：Clevr/Count 86.7（vs DINO 86.6、iBOT 85.7 基本持平），Clevr/Dist 72.4（vs DINO 53.4、iBOT 62.8 **大幅领先**）。这说明 I-JEPA 不像不变性方法那样把局部几何信息全部洗掉——它既懂「这是什么」，也保留了「物体在哪、有几个、多远」。

### 5.4 可扩展性（Figure 1 / Figure 5）

I-JEPA 的算力效率是其隐藏的最大亮点。论文摘要给出：**ViT-H/14 用 16 块 A100、不到 72 小时完成 ImageNet 预训练**（即 <1200 GPU hours），比 iBOT ViT-S/16 快 2.5 倍以上，比 MAE ViT-H/14 快 10 倍以上。原因有二：一是收敛快约 5 倍（潜空间预测 vs 像素重建），二是单视角处理（不变性方法要多视角各过一遍 encoder）。Figure 5 把 1% 半监督精度对 GPU hours 画出来，I-JEPA 的曲线在左下角——用最少的算力拿到了不输最强方法的精度。数据规模（IN1k→IN22k）和模型规模（ViT-H→ViT-G）都能带来稳定提升，说明这条路线可扩展。

### 5.5 Predictor 表征可视化（Figure 6）

![](assets/JEPA%20-%20联合嵌入预测架构/figures/03-visualization.png)

> **图 3（论文 Figure 6）：predictor 表征可视化。** 第一列是原图，第二列是 context 块（送入预训练 I-JEPA ViT-H/14 编码器），后续列的绿框是「用 RCDM 框架训练一个生成式 decoder，把 predictor 以某位置 token 为条件输出的 average-pool 表征解码回像素空间」得到的多个随机采样。**跨采样一致的品质**（如鸟的背部、车顶的姿态）代表表征里真正包含的信息；**跨采样变化的品质**（纹理细节、背景）代表被表征丢弃的信息。可以看到：predictor 正确捕捉了位置不确定性，产出了**姿态正确的高层物体部件**，同时主动丢弃了精确低级细节与背景信息。
> 来源：论文 Figure 6，第 8 页，<https://arxiv.org/abs/2301.08243>

这张图是 I-JEPA 哲学最直观的「可视化证词」：模型学会预测的是「一只鸟的背部该长什么样」，而不是「这块区域该是什么像素颜色」——它把不可预测的细节丢掉了，只保留可预测的语义。对比附录里的 MSN 可视化（Figure 8），不变性方法连物体姿态、实例数量这类全局结构都丢了不少，而 I-JEPA 保留了局部结构。

### 5.6 消融实验汇总

- **预测空间（Table 7）**：损失算在像素空间 vs 表征空间，1% ImageNet 线性评估 40.7 vs 66.9——从像素换到表征空间，性能提升近 26 个百分点，是全文最核心的「一句话证据」；
- **掩码策略（Table 6）**：多块 54.2 碾压 rasterized 15.5 / block 20.2 / random 17.6；
- **target 掩码位置（Table 11）**：输出掩码 67.3 vs 输入掩码 56.1；
- **predictor 深度（Table 12）**：6 层 64.0 vs 12 层 66.9；
- **predictor 宽度（Table 14）**：384 宽 70.7 vs 1024 宽 68.4，宽度瓶颈有益；
- **weight decay（Table 13）**：0.04→0.4 利于 linear eval（77.8 vs 76.4），固定 0.05 利于微调（70.7 vs 69.4）；
- **全量微调（Table 15）**：I-JEPA ViT-H/16@448 全量微调 87.1，与 MAE ViT-H/14@448 的 87.8 只差不到 1%，但 epoch 少了 5.3 倍。

---

## 6. 与相关工作的关系

- **vs data2vec / CAE**：最接近的工作。data2vec 也预测缺失 patch 的在线表征、避免手工增强，可跨视觉/文本/语音；CAE 用 encoder-decoder 加「重建 + 对齐」双损失。I-JEPA 相对它们在计算效率和表征语义度上都有显著提升（data2vec ViT-L/16 1600ep 77.3 vs I-JEPA ViT-L/16 600ep 77.5）。
- **vs MAE / BEiT / SimMIM（生成式）**：同属掩码预测，但 MAE 重建像素、BEiT 预测离散 token、SimMIM 预测 HOG 特征；I-JEPA 的预测空间是训练中**在线学习**的表征空间，语义层次更高、无需端到端微调即可用。
- **vs DINO / MSN / iBOT（不变性）**：这些方法靠手工增强 + 防塌缩正则学语义；I-JEPA 无需增强、单视角处理，且在低级任务上反超它们，说明其表征「语义 + 局部」兼顾。
- **vs Contrastive Predictive Coding（CPC）**：CPC 用对比目标让 patch 表征可预测相邻 patch，本质仍是判别/对比；I-JEPA 是非对比的，纯回归目标，不做 patch 间判别。
- **后续延伸**：V-JEPA（视频版，把「预测缺失图像块的表征」推广到「预测缺失时间段的表征」）直接继承了 I-JEPA 的潜空间预测哲学，是 JEPA 路线从静态图像走向时序世界模型的关键一步。

---

## 7. 局限与批判性评价

1. **「无增强」的代价是更强的掩码工程**：I-JEPA 虽然丢掉了手工数据增强，却把复杂度转移到了掩码策略上——target 尺度、context 尺度、target 数量三个超参对性能影响巨大（54.2 → 9.0 的跨度），某种意义上只是把「增强先验」换成了「掩码先验」；
2. **纯回归目标的理论基础仍薄弱**：损失是朴素的 L2 回归，凭什么不会塌缩、凭什么比对比/蒸馏更优，论文主要靠 EMA target encoder 的工程经验而非理论保证；LeCun 纲领里设想的 EBM 能量函数训练（如对比散度）在 I-JEPA 里并未真正用到；
3. **评测仍以 ImageNet 为中心**：主要结论建立在 ImageNet-1K 的 linear / 1% 半监督上，缺乏对大规模开放世界数据、多模态、时序任务的直接验证；
4. **是「世界模型」的雏形而非完整世界模型**：I-JEPA 只做静态图像内的块预测，没有动作/控制变量、没有跨时间演化、没有不确定性建模，离 LeCun 纲领里「可规划、可推理、模块化 JEPA」的完整世界模型还差得很远；
5. **丢弃细节可能是一把双刃剑**：表征里被主动抹掉的「低级纹理与背景」，对分类、计数、深度估计是好事，但对需要精确定位的稠密预测（如实例分割、关键点）未必总是有利，论文对此着墨不多。

---

## 8. 复现与实践建议

- **算力门槛**：ViT-H/14 预训练 16×A100 约 72 小时（<1200 GPU hours），单卡即可跑 ViT-B/16 的小规模复现；linear probe 用 LARS + batch 16384；
- **代码**：官方实现 <https://github.com/facebookresearch/ijepa> 已开源，掩码采样器用 PyTorch batch-collator 几行代码即可实现；
- **实现要点**：
  - 掩码采样要**作用在 target encoder 输出上**，不是输入上（这是最容易踩的坑）；
  - predictor 保持窄宽度（384），别顺手加宽——加宽反而掉点；
  - target encoder 的 EMA momentum 从 0.996 线性升到 1.0，与 DINO/BYOL 一致，务必保留；
  - 去掉 context 与 target 的重叠区域，否则预测任务退化成「抄答案」；
  - 无 [cls] token，评测用 average pooling；
- **常见踩坑**：batch size 要够大（2048）且同卡上的 context/target 掩码尺寸要统一，否则无法高效批处理；Clevr 评测不要用随机裁剪，会删掉要计数的物体。

---

## 9. 个人启发与后续问题

1. **「预测表征而非像素」是世界模型路线的一次范式转向**：I-JEPA 用最朴素的方式证明了 LeCun 的判断——把重建目标从像素换成可学习的表征，就能同时收获「高语义」和「高效率」。这个转向的意义不只在图像，它为后续 V-JEPA（视频）、以及更广义的「非生成式世界模型」奠定了方法论基础。
2. **JEPA 与生成式视频世界模型（VPP/SVD 路线）是互补而非对立**：两者都认同「预测未来」是学世界模型的正确目标，但实现路径截然相反——JEPA 走**表征级预测**，主动丢弃不可预测的像素细节，只学可预测的语义，因此高效、防塌缩、不浪费容量在纹理上；VPP/SVD 路线走**生成式像素/潜空间重建**，用视频扩散模型显式预测未来帧，保留细节、可直接可视化、并天然携带互联网规模预训练带来的物理世界先验。前者回答「怎么高效地学到可迁移表征」，后者回答「怎么把物理直觉直接用于动作决策」。
3. **两者在机器人/世界模型里的互补关系**：I-JEPA 式的潜空间预测更适合当「感知 backbone」——语义强、局部结构保留、算力便宜，适合作为世界模型的状态编码器；VPP 式的视频扩散更适合当「未来模拟器」——能显式预测未来帧、支撑逆动力学反推动作。一条自然的融合路径是：用 JEPA 表征压缩观测、用生成式视频模型做未来想象、用逆动力学做决策——恰好对应 LeCun 纲领里「世界模型 + 代价函数 + actor」的模块化分工。V-JEPA 已证明潜空间预测可扩展到视频，未来一个值得追问的问题是：JEPA 的表征级预测能否直接替代 VPP 里的像素级视频预测，从而把世界模型的推理成本再降一个量级。
4. **「预测」的边界在哪里**：I-JEPA 的潜空间预测默认「所有 target 块都该被预测」，但真实世界里大量区域是不可预测的。LeCun 纲领强调的正是「只预测可预测的部分」，而 I-JEPA 只是用「大块 + 表征空间」粗粒度地逼近了这一理想。如何在潜空间里显式建模「可预测性」、给不确定性留位置，是一个尚未被充分回答的问题。
5. **未解问题**：（i）纯 L2 回归 + EMA 的防塌缩机制能否给出理论解释？（ii）JEPA 表征能否替代 MAE 式表征成为视频模型（如 SVD）的条件编码器？（iii）把动作/控制变量注入 context，能否把 I-JEPA 直接升级成一个可规划的世界模型？

---

## 10. 代码级实现：严格沿 I-JEPA 训练 pipeline 追踪 shape

### 10.1 代码范围、版本与默认配置

本节以官方仓库 `facebookresearch/ijepa` 的 commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`（2023-06-13）为准，并以官方 `configs/in1k_vith14_ep300.yaml` 作为唯一默认配置来源。仓库覆盖 ImageNet 预训练主链路，但不包含论文全部 linear probe、fine-tune、低样本评测与 RCDM 可视化 decoder，因此本节重点解释“从图像到 JEPA loss”的真实训练数据流。

先纠正一个容易被口头描述带错的数字：ViT-H/14 在 224×224 输入下不是 196 个 patch，而是：

$$N=(224/14)^2=16^2=256$$

196 对应 224/16 的 ViT-H/16 或 ViT-B/16 口径。官方配置的 ViT-H/14 encoder width 为 1280、depth 32、heads 16；predictor width 384、depth 12、heads 16。

默认每 GPU 的 local batch 为 128，官方复现实验用 16 张 A100 80GB，因此有效 global batch 为：

$$B_{global}=128\times16=2048$$

代码从不把全局激活真正拼成 `[2048,...]`；每卡只持有 `[128,...]`，反向时由 DDP 聚合梯度。下文先用通用 $B$，再给单样本和单卡的具体数字。

### 10.2 论文模块到代码入口

| 论文 pipeline | 官方代码入口 | 职责 |
|---|---|---|
| YAML 入口 | `main.py`、`main_distributed.py` | 读取实验配置并启动单机/Submitit 任务 |
| ImageNet 数据 | `src/datasets/imagenet1k.py` | `ImageFolder`、`DistributedSampler`、DataLoader |
| 图像变换 | `src/transforms.py` | `RandomResizedCrop(224)`、ToTensor、ImageNet normalize |
| multi-block masks | `src/masks/multiblock.py::MaskCollator` | 每 batch 采 target/context 的大小与位置，并定长 |
| context/target patch 抽取 | `src/masks/utils.py::apply_masks` | 按 patch index gather，并把 mask 数并入 batch 维 |
| context/target encoder | `src/models/vision_transformer.py::VisionTransformer` | ViT-H/14；同构、不同更新方式 |
| predictor | `VisionTransformerPredictor` | 1280→384、拼 target mask token、12 层预测、384→1280 |
| 主训练循环 | `src/train.py` | target forward、context/predictor forward、loss、优化、EMA |
| optimizer/schedule | `src/helper.py`、`src/utils/schedulers.py` | AdamW、LR/WD cosine、参数分组 |
| EMA teacher | `src/train.py` | optimizer step 后更新 target encoder |

### 10.3 全链路张量总表

记 context mask 数 $C=1$、target mask 数 $T=4$、每个 target token 数为 $N_t$、当前 local batch 经截断后的 context token 数为 $N_c$。

| # | 阶段 | 通用 shape | 单样本 $B=1$ | 单卡 $B=128$ |
|---:|---|---|---|---|
| 1 | crop 后图像 | `[B,3,224,224]` | `[1,3,224,224]` | `[128,3,224,224]` |
| 2 | patch Conv | `[B,1280,16,16]` | `[1,1280,16,16]` | `[128,1280,16,16]` |
| 3 | 全 patch token | `[B,256,1280]` | `[1,256,1280]` | `[128,256,1280]` |
| 4 | context index | `[B,N_c]` | `[1,N_c]` | `[128,N_c]` |
| 5 | 每个 target index | `[B,N_t]` | `[1,N_t]` | `[128,N_t]` |
| 6 | target encoder 全图输出 | `[B,256,1280]` | `[1,256,1280]` | `[128,256,1280]` |
| 7 | gather 4 个 target | `[TB,N_t,1280]` | `[4,N_t,1280]` | `[512,N_t,1280]` |
| 8 | context encoder 输出 | `[CB,N_c,1280]` | `[1,N_c,1280]` | `[128,N_c,1280]` |
| 9 | predictor context 投影 | `[CB,N_c,384]` | `[1,N_c,384]` | `[128,N_c,384]` |
| 10 | target mask+position tokens | `[TCB,N_t,384]` | `[4,N_t,384]` | `[512,N_t,384]` |
| 11 | predictor 拼接序列 | `[TCB,N_c+N_t,384]` | `[4,N_c+N_t,384]` | `[512,N_c+N_t,384]` |
| 12 | predictor 输出 $\hat h$ | `[TCB,N_t,1280]` | `[4,N_t,1280]` | `[512,N_t,1280]` |
| 13 | target $h$ | `[TCB,N_t,1280]` | `[4,N_t,1280]` | `[512,N_t,1280]` |
| 14 | Smooth L1 | scalar | scalar | scalar/rank |

> 注：上表标题中的 predictor 输出应读作预测表征 $\hat h$。实际 global batch 为 2048 时，概念上的 target/prediction 总量是 `[8192,N_t,1280]`，但物理上分布在 16 张 GPU 上；各 rank 的 $N_c$ 还可能不同，不能简单视为一个稠密全局 tensor。

### 10.4 DataLoader 与 multi-block mask：latent shape 在进入模型前就被决定

#### 10.4.1 输入输出与内部张量流

数据侧只使用 `RandomResizedCrop(224,scale=(0.3,1.0))`、ToTensor 与 ImageNet normalize；默认关闭 horizontal flip、color jitter 和 Gaussian blur。DataLoader 的 collator 返回：

```text
collated_batch         # images + ImageNet labels
collated_masks_enc     # list 长度 C=1，每项 [B,N_c]
collated_masks_pred    # list 长度 T=4，每项 [B,N_t]
```

训练循环只读取 `udata[0]` 的图像，ImageNet label 被完全丢弃。

Patch 网格是 $16\times16$。MaskCollator 每次调用只采一次 target block size，所以**同一个 local batch 的 128 张图、4 个 target 都共享相同的 $h_t,w_t$**，只是位置不同。对 `[0.15,0.2]` target scale 与 `[0.75,1.5]` aspect ratio 静态枚举，当前实现可能得到：

$$N_t\in\{35,36,42,48,54\}$$

对应约 `5×7`、`6×6`、`6×7/7×6`、`8×6`、`9×6` 等矩形。4 个 target 之间可以重叠。

Context 候选的 scale 为 `[0.85,1.0]`、aspect ratio 固定 1；经过代码的 round 与“必须小于网格边长”逻辑，默认候选总是 `15×15=225` patch。再删去与 4 个 target 重叠的区域，得到每张图不同长度的 context。为了 batch 化，collator 取 local batch 内最小长度并对所有样本做：

```python
cm[:min_keep_enc]
```

所以最终 $11\le N_c\le225$，但通常明显小于 225。

#### 10.4.2 为什么这样设计

Target 大小在 batch 内固定，保证 `[B,N_t]` 可以直接 stack；context 取 local batch 最短长度，则让 `[B,N_c]` 能进入标准 dense Transformer，而不用 padding mask。代价是部分图像会丢掉本可用的 context token，且截断按 raster index 顺序发生，天然偏向保留索引较小的区域。

#### 10.4.3 关键超参

| 项 | 值 |
|---|---:|
| input / patch | 224 / 14 |
| patch grid / total token | 16×16 / 256 |
| target 数 | 4 |
| target scale | `[0.15,0.2]` |
| target aspect ratio | `[0.75,1.5]` |
| context 数 | 1 |
| context candidate scale | `[0.85,1.0]` |
| `min_keep` | 10，即有效长度必须 >10 |
| target/context overlap | 配置上禁止，但代码可超时回退 |

#### 10.4.4 代码与论文的差异及实现边界

这里有三条源码级细节必须单独记住：

1. **scale 与 aspect ratio 共用同一个随机数**，二者正相关，而不是论文文字容易让人理解的独立采样。
2. 位置用 `torch.randint(0,height-h)`，上界少了应有的 `+1`。默认 16×16 网格下，所有 block 都无法覆盖最后一行/列；`15×15` context 的 top/left 更只能取 0，因此候选 context 永远固定在左上角。这是该 commit 的明确 off-by-one 局限。
3. `allow_overlap=false` 是软约束：若 20 次仍找不到足够大的 context，代码会逐步减少必须避开的 target 区域，最终可能允许重叠。

这些行为不会改变 JEPA 的概念目标，却会改变真实训练样本分布，是“论文 pipeline”与“代码 pipeline”之间最值得关注的差异之一。

### 10.5 Target encoder：完整 256-token 图像先编码，再 gather target latent

#### 10.5.1 输入输出与内部张量流

Target encoder 是 context encoder 初始化后的 `deepcopy`，参数不走梯度，而是在每次 optimizer step 后做 EMA。它始终看到完整图像：

```text
images                     [B,3,224,224]
patch Conv                  [B,1280,16,16]
flatten + fixed 2D sincos   [B,256,1280]
32-layer ViT-H/14           [B,256,1280]
模型内部 LayerNorm          [B,256,1280]
训练循环额外 F.layer_norm   [B,256,1280]
apply 4 target masks        [4B,N_t,1280]
```

`apply_masks` 对每个 `[B,N_t]` index 做 gather，再沿 batch 维 `cat`。其内存顺序是“target 0 的 B 个样本 → target 1 的 B 个样本 → …”。默认 $C=1$ 时不再扩张；一般 $C>1$ 时，`repeat_interleave_batch` 会把 target 再复制为 `[TCB,N_t,1280]`。

#### 10.5.2 为什么这样设计

先用完整图像编码、再在输出 latent 上取 target，保证 teacher target 包含该区域在全图上下文中的语义，而不是一个被裁碎、信息不足的局部编码。这也是论文消融中“output masking”明显优于“input masking”的代码对应。Target encoder 不反传，既提供稳定目标，又避免 context 与 target 两条路径共同追逐瞬时噪声。

#### 10.5.3 关键超参

- ViT-H/14：width 1280、depth 32、heads 16、head dim 80、MLP 5120。
- 无 CLS token；固定二维 sin-cos position embedding `[1,256,1280]`。
- EMA momentum 从 0.996 逐 iteration 线性升到 1.0。

#### 10.5.4 代码与论文的差异

论文公式写 target 表征 $s_y$，没有突出 loss 前的额外标准化；代码在模型尾部 LayerNorm 之后，又对 feature 维执行一次无可学习参数的 `F.layer_norm`，而 predictor 输出没有对应的第二次标准化。这个不对称是实际训练目标的一部分。

### 10.6 Context encoder：patch projection 全图算，32 层 Transformer 只看 $N_c$ token

#### 10.6.1 输入输出与内部张量流

Context encoder 与 target encoder 同为 ViT-H/14，但 `forward(x,masks_enc)` 的顺序是：

```text
全图 patch Conv         [B,3,224,224] → [B,256,1280]
加完整位置编码          [B,256,1280]
apply context mask      [B,N_c,1280]       # C=1
32 层 Transformer       [B,N_c,1280]
最终 LayerNorm          [B,N_c,1280]
```

因此“context encoder 只编码可见 patch”只对昂贵的 Transformer 部分成立；最前面的 patch Conv 仍对完整图像计算。

#### 10.6.2 为什么这样设计

在位置编码之后 gather，可以保留每个可见 patch 的绝对二维位置；随后让注意力只在 $N_c$ 个 token 上运行，计算量从 $256^2$ 降为 $N_c^2$。这正是 multi-block masking 同时承担学习目标设计和计算稀疏化的地方。

#### 10.6.3 关键超参与差异

Context encoder 的结构参数与 target 完全相同，唯一差异是参数更新：context 走 AdamW 梯度，target 走 EMA。论文常把 encoder 输入抽象为“context block $x$”，代码并没有先裁出一张小图，而是在完整 patch 网格上通过 index gather 得到稀疏 token 序列。

### 10.7 Predictor：$T\times C$ 次预测被向量化到 batch 维

#### 10.7.1 输入输出与内部张量流

设 context encoder 输出为 `[CB,N_c,1280]`。Predictor 首先做：

```text
Linear(1280→384)
[CB,N_c,1280] → [CB,N_c,384]
```

然后从固定 `[1,256,384]` position embedding 中 gather context 位置并相加。对 4 个 target，从同一位置表 gather `[TB,N_t,384]`，再为 $C$ 个 context 复制为 `[TCB,N_t,384]`。可学习 mask token `[1,1,384]` 广播相加，得到 target query token。

Context 同时按 target 数整体复制：

```text
context:       [CB,N_c,384]
repeat T 次 →  [TCB,N_c,384]
target token:  [TCB,N_t,384]
cat seq dim →  [TCB,N_c+N_t,384]
```

默认 $B=128,C=1,T=4$：

```text
context projected       [128,N_c,384]
target mask+pos tokens  [512,N_t,384]
repeated context        [512,N_c,384]
predictor input         [512,N_c+N_t,384]
12-layer predictor      [512,N_c+N_t,384]
drop context slice      [512,N_t,384]
Linear(384→1280)        [512,N_t,1280]
```

预测结果与 target latent `[512,N_t,1280]` 一一对应。论文说“对每个 target 调用 predictor 一次”，代码则把 4 次独立调用打包到 batch 维；attention 不跨 batch，因此数学上等价。

#### 10.7.2 为什么这样设计

384 维 bottleneck 把 predictor 变成“难以抄答案的窄模型”，迫使 1280 维视觉语义压缩后再预测。Target query 由“同一个可学习 mask 向量 + 该 patch 的固定位置编码”组成：前者表示未知内容，后者明确告诉 predictor 要填图像的哪个位置。把 4 个 target 向量化到 batch 维则避免 Python 循环，也保证每个 target 只与自己的 context 做 self-attention。

#### 10.7.3 关键超参

| 项 | 值 |
|---|---:|
| encoder→predictor | 1280→384 |
| predictor depth / heads | 12 / 16 |
| head dim / MLP dim | 24 / 1536 |
| position embedding | fixed 2D sin-cos `[1,256,384]` |
| mask token | learnable `[1,1,384]` |
| predictor→target space | 384→1280 |

#### 10.7.4 代码与论文的差异

代码的 predictor 输入不是“context 表征 + 一个 target block 的 token”单次张量，而是 `[TCB,N_c+N_t,384]` 的批量展开。阅读 shape 时若漏掉 `x.repeat(len(masks),1,1)`，就会错误地把输出写成 `[B,N_t,1280]`；真实默认 shape 是 `[4B,N_t,1280]`。

### 10.8 Loss、AdamW 与 EMA 更新顺序

#### 10.8.1 真实训练顺序

每个 iteration 严格执行：

```text
1. target_encoder(imgs)（no_grad）
2. 额外 LayerNorm + gather target latent h
3. context encoder 得到 z_context
4. predictor 得到 z_pred
5. Smooth L1(z_pred,h)
6. backward + AdamW step（只更新 context encoder/predictor）
7. optimizer.zero_grad()
8. target encoder EMA update
```

Target encoder 不在 optimizer 参数组中。Bias 和所有一维参数（包括 LayerNorm 参数）不做 weight decay；其余参数的 weight decay 从 0.04 cosine 增至 0.4。学习率前 40 epochs 从 $2\times10^{-4}$ warmup 到 $10^{-3}$，再 cosine 降到 $10^{-6}$；训练 300 epochs，BF16 开启。

#### 10.8.2 论文与代码最直接的目标函数差异

论文公式写 squared L2：

$$\mathcal L_{paper}=\frac1M\sum_i\sum_{j\in B_i}\|\hat s_{y_j}-s_{y_j}\|_2^2$$

当前官方代码却调用 `F.smooth_l1_loss(z,h)`，默认 `beta=1,reduction="mean"`。因此真实实现对小误差使用二次惩罚、对大误差转为线性惩罚，并在 `[TCB,N_t,1280]` 的所有元素上取 mean。这个差异会降低异常 target feature 对梯度的支配，不应在代码复现时擅自换回 MSE 而不做消融。

### 10.9 计算账：I-JEPA 为什么比像素重建收敛快但单步并非“免费”

| 阶段 | 序列长度 | width | depth | 单 iteration 执行次数 |
|---|---:|---:|---:|---:|
| Target ViT-H/14 | 256 | 1280 | 32 | 1，完整图像，no-grad |
| Context ViT-H/14 | $N_c$ | 1280 | 32 | 1，需反传 |
| Predictor | $N_c+N_t$ | 384 | 12 | 向量化为 $T=4$ 份，需反传 |

I-JEPA 每步仍要运行一次完整 target encoder，所以单步不一定比 MAE 便宜；节省来自两点：context 主干只在 $N_c$ token 上做注意力，predictor 只有 384 维，而且表征目标更容易优化、所需 epoch 更少。默认 16 GPU 下每步 target 数量为 $4\times2048=8192$ 个 block，每个 block 包含 35–54 个 1280 维 latent，loss 直接在这一大批 patch 表征上平均。

### 10.10 代码局限、复现建议与最小调试路径

1. **先核对 256 patch**：对 ViT-H/14@224 应打印 `[B,256,1280]`；出现 196 通常是误用了 patch size 16 的口径。
2. **记录 $N_t$ 与 $N_c$**：`N_t` 在一个 local batch 内固定，$N_c$ 是该 rank 的最短 context；不同 GPU 的 $N_c$ 可不同，这是合法的 DDP 分片行为。
3. **不要把配置 `batch_size:128` 当成 global batch**：只有 16 ranks 才得到论文的 2048；少卡复现要明确有效 batch 与学习率变化。
4. **保留 target output masking**：先全图过 target encoder，再 gather；若把 mask 提前到输入，语义与论文消融都改变。
5. **保留 predictor 384 维瓶颈**：不要为了“容量更大”随手改成 1280，论文消融表明更宽反而更差。
6. **按代码复现时默认用 Smooth L1**：论文写 L2，但官方 commit 的真实训练目标是 Smooth L1；两者应分别记录。
7. **警惕 mask off-by-one**：该 commit 的最后一行/列永远采不到，15×15 context 固定左上；若修复，训练分布会变化，应视为新的实现版本而不是 bitwise reproduction。
8. **位置编码换分辨率有风险**：模型没有 CLS token，但插值函数带有 `-1` 的 CLS 遗留逻辑；224 原分辨率走直接返回不受影响，变分辨率前要先修正并测试。
9. **最小可跑路径**：单卡先用 ViT-B/16 小配置验证 `images → masks → h/z shape → Smooth L1 → EMA`；再迁移到 ViT-H/14。真正复现 2048 global batch 需要多卡 DDP，而不是 gradient accumulation 的默认实现。
10. **仓库覆盖边界**：官方代码能复现预训练主链路，但没有论文完整下游评估脚本；预训练 loss 跑通不等于 linear probe、1% fine-tune 与可视化结果均已复现。

---

## 11. 材料来源

### 11.1 Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2301.08243> | 全文（含附录） | 2026-10-31 | 精读主来源 |
| `figures/01-architectures.png` | Figure 2：自监督三大架构对比（JEA / Generative / JEPA） | arXiv:2301.08243 | 第 2 页 | 2026-08-27 | 架构对比 |
| `figures/02-method.png` | Figure 3：I-JEPA 方法示意（context/target encoder + predictor） | arXiv:2301.08243 | 第 3 页 | 2026-08-27 | 核心方法 |
| `figures/03-visualization.png` | Figure 6：predictor 表征可视化（RCDM decoder 解码） | arXiv:2301.08243 | 第 8 页 | 2026-08-27 | 关键定性结果 |
| `sources/in1k_vith14_ep300.yaml` | 官方 ViT-H/14 预训练配置 | <https://github.com/facebookresearch/ijepa> | `configs/in1k_vith14_ep300.yaml` | 2026-10-31 | shape 与超参一手依据 |
| `sources/code-version.txt` | 官方仓库版本 | <https://github.com/facebookresearch/ijepa> | commit `52c1ae95...` | 2026-10-31 | 固定代码阅读版本 |
| `sources.md` | 完整来源与代码人工复核清单 | 本地 | — | 2026-10-31 | 溯源 |
