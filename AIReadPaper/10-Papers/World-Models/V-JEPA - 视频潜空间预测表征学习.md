---
type: paper-note
status: complete
domain: World-Models
paper: Revisiting Feature Prediction for Learning Visual Representations from Video
year: 2024
arxiv: 2404.08471
doi: null
source: https://arxiv.org/abs/2404.08471
project: https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/
code: https://github.com/facebookresearch/jepa
tags: [World-Models, JEPA, Self-Supervised, Video-Representation, Feature-Prediction, V-JEPA]
created: 2026-09-03
updated: 2026-09-03
---

# V-JEPA：视频潜空间预测表征学习

## 基本信息

- **作者**：Adrien Bardes、Quentin Garrido、Jean Ponce、Xinlei Chen、Michael Rabbat、Yann LeCun、Mahmoud Assran、Nicolas Ballas（* 共同一作；Bardes / Garrido 一作，Assran / Ballas 通讯）
- **机构**：FAIR at Meta；Inria、École normale supérieure（CNRS、PSL）、Univ. Gustave Eiffel（CNRS、LIGM）、Courant Institute / Center for Data Science（NYU）
- **发表**：arXiv:2404.08471（v1：2024-02-15；v3：2024-05-01；CVPR 2025？）
- **论文正式名**：*Revisiting Feature Prediction for Learning Visual Representations from Video*
- **官方代码**：<https://github.com/facebookresearch/jepa>（与 I-JEPA 同仓库）
- **概念源头**：LeCun 2022 纲领《A Path Towards Autonomous Machine Intelligence》提出的 JEPA；本工作的图像版落地是 [I-JEPA](arXiv:2301.08243)（CVPR 2023），V-JEPA 是它在视频模态的延伸

## 一句话结论

V-JEPA 是 **JEPA 在视频自监督学习上的首次独立落地**：完全抛弃像素重建、对比损失、负样本、pretrained image encoders、文本监督与手工数据增强，只用一个被回归到 L1 损失的"从 context 表征预测 masked spatio-temporal patch 的表征"目标训练 ViT-H/16，就在 Kinetics-400（81.9%）、Something-Something-v2（72.2%）和 ImageNet-1K（77.9%）上以 frozen backbone 同时刷新「运动」与「外观」两类任务的 SOTA，并明确证明 feature prediction 是视频自监督的一个**可独立成立**的目标。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

自监督视频表征学习要回答一个根本问题：**一个视觉系统能否在没有人工标注的前提下，从原始视频中同时学到「运动」（fine-grained temporal）和「外观」（scene-level appearance）两种性质截然不同的视觉知识**。V-JEPA 的研究问题被作者表述得很简洁（论文标题就是「Revisiting Feature Prediction」）：

> *How effective is feature prediction as a stand-alone objective for unsupervised learning of visual representations from video with modern tools?*

也就是说，把 feature prediction 从众多自监督目标（pixel重建、聚类、对比、时序不变性等）中**单独抽出来**作为一个视频自监督目标，它还能 work 吗？工作到什么程度？

### 1.2 现有方法的瓶颈

到 2023 年底，视频自监督学习被两条路线主导，每条都有自己的根本病**：

- **Pixel reconstruction（VideoMAE / Hiera / OmniMAE）**：把掩码后的视频 patch 用 decoder 重建为像素或低层 token。优点是不需要增强、易扩展；缺点是**把大量模型容量浪费在「不可预测又与任务无关」的纹理细节上**（草地每根草尖的精确位置），学到的表征语义层次偏低，离线评估（frozen probing）常常输给对比方法。
- **Contrastive / 不变性方法（CVRL、VideoMoCo、RSP）**：用多个时序相邻 clip 构造正样本，靠对比损失或时序一致性目标拉近正样本。优点是在「外观」任务（K400）上很强；缺点是**手工增强设计、负样本设计、时序平滑假设都对「细粒度运动」（如 SSv2 里的「推 vs 拉」）不友好**，运动任务表现明显偏弱。
- **Image-model transfer（DINOv2 迁移到视频）**：把图像自监督模型直接搬到视频任务，能在「外观」类任务拿到不错的分数，但在「运动」类任务（SSv2）只拿 50 出头，落后像素重建路线 15+ 个百分点（见 Figure 1 散点图）。

也就是说，**「外观」与「运动」两类任务在 2024 年初还没有任何一个视频自监督模型能同时拿下**——这是 V-JEPA 想要正面回答的痛点。

### 1.3 与 I-JEPA 的关系：JEPA 谱系的视频版

V-JEPA 不是凭空冒出来的。它是 [I-JEPA](arXiv:2301.08243)（CVPR 2023，笔记见《[[JEPA - 联合嵌入预测架构]]》）在视频模态的直接延伸：

- **方法论完全一致**：从 context 表征预测 masked patches 的表征（潜空间预测，而非像素）；用 EMA target encoder 防塌缩；非生成式架构。
- **关键变化是 patch 维度**：从图像 patch（2D）升级为 spatio-temporal patch（3D：时间 × 高 × 宽）；从单帧 square mask 升级为时空多块 mask。

论文 3.2 节明确说「The main idea behind a JEPA is to learn by predicting the representation of an input y from the representation of another input x」，并强调 I-JEPA 在图像上的成功（ImageNet linear 79.3% / 1% 半 73.3%）说明这条路线值得往高维、时序迁移。

### 1.4 本文核心贡献

1. **Feature prediction as a stand-alone objective**：证明在视频上可以完全抛弃像素重建、对比损失、负样本、pretrained image encoders、文本监督和人工标注，只用「在表征空间里做 masked prediction」一个目标就足以学出强视频表征。
2. **Multi-block spatio-temporal masking strategy**：把 I-JEPA 的 square mask 升级为时空多块掩码（短程 15% 帧 + 长程 70% 帧的混合），平均 mask ratio ~90%，保留「短程运动」与「长程时空」两类预测难度。
3. **L1 + EMA target encoder 的实用 recipe**：作者试验发现 L1 比 L2 更稳定（与 I-JEPA 用 L2 不一样），并把 target encoder 的 EMA momentum 设置好（m: 0.999 → 1.0）。
4. **Frozen evaluation 上的双 SOTA**：ViT-H/16 在 K400 81.9%、SSv2 72.2%、ImageNet-1K 77.9%，**用一个 frozen backbone 同时跨域**（运动 + 外观）刷 SOTA。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：一段 $T$ 帧 × $H \times W$ 像素的视频片段，预训练默认 $T=16$、$H=W=224$ 或 $384$；采样时 clip 帧率为 4（对应粗糙 3 秒 clip，平均 3 个相邻 clip 重叠）。
- **输出**：每个时空 patch 的 patch-level 表征 $\{s_y^{(i)}\}$。
- **patch 化**：把视频切成 $2 \times 16 \times 16$ 大小的 spatio-temporal token（$T \times H \times W$ grid），ViT-L/16 用 $16 \times 16$ patch，ViT-H/16 用 $16 \times 16$ patch；时空 grid 的总数记为 $L = (T/2) \times (H/16) \times (W/16)$。
- **核心假设**：与 I-JEPA 一脉相承——**如果只让模型在表征空间里「补齐」被 mask 的时空块，模型被迫学到的就是那些真正可预测、跨时空共享的运动规律，而不是像素纹理**。视频是冗余信息量更大的模态，因此潜空间预测的好处被进一步放大。

### 2.2 关键符号与目标函数

设 $x$-encoder 为 $E_\theta$、$y$-encoder 为 $E_{\bar\theta}$、predictor 为 $P_\phi$。训练目标：

$$\mathcal{L}_{\theta,\phi,\Delta_y} \;=\; \min \Big\lVert P_\phi\big(E_\theta(x),\,\Delta_y\big) \;-\; \mathrm{sg}\big(E_{\bar\theta}(y)\big) \Big\rVert_1$$

其中：

- $x$：从视频中抽出的 mask 后 token 序列（context 输入）；
- $y$：同一视频的完整 token 序列（用于构造 target）；
- $\Delta_y$：可学习的 mask token + 位置编码，指定「要预测哪个时空位置」；
- $\mathrm{sg}(\cdot)$：stop-gradient；
- $\bar\theta$：$\theta$ 的 EMA（$m: 0.999 \to 1.0$），由 x-encoder 权重指数滑动平均得到。

推理时只保留 $E_\theta$；下游任务在 $E_\theta$ 输出上做 linear probe 或 attentive probe。

---

## 3. 核心方法

### 3.1 总体框架

V-JEPA 的整体结构是「**两个 encoder + 一个 narrow predictor**」的三件套。它与 I-JEPA 在结构上几乎一样，差异在于 patch 维度（图像 2D vs 视频 3D）和掩码策略。

![](assets/V-JEPA%20-%20视频潜空间预测表征学习/figures/02-jepa-architecture.png)

> **图 1（论文 Figure 2）：Joint-Embedding Predictive Architecture 通用框架。** 红色变量 $z$ 提供给 predictor，编码要预测的「变换或扰动」信息（如位置、时间步）。predictor 接收 $z$ 与 $E_\theta(x)$，输出对目标表征 $\hat{s}_y$ 的预测；$E_{\bar\theta}(y)$ 给出真实 $s_y$；二者用 $D(\hat{s}_y, s_y)$ 比较。JEPA 的核心是「从 $x$ 的表征预测 $y$ 的表征」，而非像素。
> 来源：论文 Figure 2，第 3 页，<https://arxiv.org/abs/2404.08471>

### 3.2 Feature Prediction vs Pixel Reconstruction

论文 3.1 节用一个核心段落把为什么「预测表征比预测像素更适合视频」讲清楚：

> *「Approaches that predict in pixel space must dedicate significant model capacity and compute to capture all the low-level detail in the visual input. By contrast, approaches that predict in latent space have the flexibility to eliminate irrelevant or unpredictable pixel-level details from the target representation, leading to versatile representations that perform well across many downstream tasks through linear probing or low-shot adaptation.」*

也就是说，**像素预测把模型变成了一个视频压缩器**（必须记住每一个不可预测的细节），而**特征预测让模型把容量集中在「可预测的部分」**——对视频来说这恰好是「运动与场景」的核心信息。这是 V-JEPA 方法论的最高一层。

### 3.3 V-JEPA 训练流程（详细）

下面这张图把训练 pipeline 完整展开，每个组件的 tensor shape 都标注在下方，便于工程实现：

![](assets/V-JEPA%20-%20视频潜空间预测表征学习/figures/03-method-pipeline.png)

> **图 2（论文 Figure 3）：V-JEPA 训练 pipeline。** 流程（左→右）：① 把 $T \times H \times W$ 视频 flatten 成 $L$ 个 spatio-temporal token；② 用 Binary Mask 在 x-encoder 入口和 y-encoder 出口分别构造 masked sequence；③ 移除 masked tokens（x-encoder）保留 $[L \times d]$，移除 unmasked tokens（y-encoder）保留 $[L \times d]$；④ x-encoder 输出 $[N \times d]$，与可学习 mask tokens（含位置编码）拼接成 $[L \times d]$；⑤ predictor $P_\phi$ 输出每个 masked patch 的表征 $[M \times d]$；⑥ L1 回归到 $y$-encoder 输出（stop-grad），目标也是 $[M \times d]$。$y$-encoder 权重 $\bar\theta$ 是 $x$-encoder 权重的指数滑动平均（m）。
> 来源：论文 Figure 3，第 4 页，<https://arxiv.org/abs/2404.08471>

关键工程点：

- **Mask 施加在 y-encoder 输出上，而不是输入上**——这是 I-JEPA 已经验证过的设计（消融显示「输出掩码」67.3 vs「输入掩码」56.1 on IN-1K linear eval），V-JEPA 沿用。视频版同样要求先过一遍完整 y-encoder 再构造 target 表征，避免信息泄露。
- **EMA target encoder** 是 JEA 类方法用 ViT 训练时防塌缩的标配。momentum schedule 与 I-JEPA 一致。
- **L1 而非 L2**：作者明确指出（论文 Eq. 1 上方段落）L2 与 I-JEPA 等价目标完全相同，但实际实验中 L1 在视频上「more stable」。理论动机来自 Grill et al. 2020 的 BYOL 分析——L1 对异常值更鲁棒，对视频中偶尔出现的遮挡/光照闪烁不敏感。

### 3.4 x-encoder 与 y-encoder

两个 encoder 都是标准 ViT（无 [cls] token），但有以下细节：

- **架构选型**：ViT-L/16 和 ViT-H/16（patch 16×16），batch size 分别 3072 和 2400。
- **时空 patch 化**：把 $T \times H \times W$ 的视频重排成 $(T/2) \times (H/16) \times (W/16)$ 的 token grid，每个 token 是一个 $2 \times 16 \times 16$ 的 3D spatio-temporal cube（2 帧、16×16 像素）。
- **位置编码**：用绝对 1D sinusoidal（与 ViT 默认一致）。

### 3.5 Predictor

Predictor 是 V-JEPA 里唯一「做预测」的组件，刻意设计成**窄 ViT（narrow）**：

- **Embedding dim = 384**（远小于 ViT-H 的 1024）
- **Depth**：12 层（ViT-L）/ 16 层（ViT-H）
- **Self-attention 头数**：与 backbone 一致

Predictor 的输入是 `[x-encoder 输出] + [可学习 mask tokens]`，mask token 包含一个共享的可学习向量加上位置编码，用来告诉 predictor「要预测哪个时空位置」。由于视频有 $L$ 个 patch，可能预测位置多达 $L$，predictor 会被应用 $L$ 次。

**窄宽度是有意为之**：宽度瓶颈迫使信息在预测时被压缩，避免 predictor「抄近路」——这与 I-JEPA 的发现一致（Table 14，384 宽 70.7 vs 1024 宽 68.4）。

### 3.6 Spatio-Temporal Multi-Block Masking

这是 V-JEPA 与 I-JEPA 相比的第二大方法论变化（patch 维度变化是第一大）。V-JEPA 把 I-JEPA 的 square mask 升级为**时空多块掩码**：

- **短程掩码**：从每个时间步的 patch grid 里采 8 个 block，每个 block 覆盖 15% 帧面积；
- **长程掩码**：从每个时间步采 2 个 block，每个 block 覆盖 70% 帧面积；
- **采样参数**：长宽比 $(0.75, 1.5)$；
- **总 mask ratio ≈ 90%**（因为短程 + 长程几乎覆盖每个时间步的全部 patch）。

短程掩码让模型学到「细粒度运动」（小窗口内的局部光流），长程掩码让模型学到「跨时间步的事件关联」（如「推门→门开」的多帧因果）。

论文 4.1 节明确说：「Masking a large continuous block that covers the full temporal dimension limits information leakage due to spatial and temporal redundancy of videos, and results in a harder prediction task」。长程大块是为了保证任务难度足够——视频有大量时空冗余，必须用足够大的 mask 才能让模型真正学到跨帧推演。

### 3.7 训练目标与防塌缩机制

训练目标就是 2.2 节那个 L1 loss。**塌缩防护机制有两道**：

1. **EMA target encoder**：$y$-encoder 权重 $\bar\theta$ 不直接走梯度，而是 $\theta$ 的指数滑动平均。$m$ 从 0.999 线性升到 1.0，确保 target encoder 始终比 $x$-encoder 慢半拍，predictor 永远需要学习「预测未来」而不是「抄近路到 target」。
2. **Stop-gradient**：目标 $s_y$ 不反向传播通过 $y$-encoder，防止 encoder 把所有输入都映射到同一向量来最小化 L1。

理论上，如果允许反向传播通过 target encoder，模型很快会发现「输出常数 $c$」能让 L1 最小化（塌缩到 $c$），所以 stop-grad 是必须的。

论文给出了一个 BYOL 风格的理论动机（来自 Grill et al. 2020）：在最优 predictor 下，$x$-encoder 的梯度与「$|E_\theta(x) - \mathrm{median}(Y)| E_\theta(x)$」成正比，意味着 encoder 必须学到关于 $y$ 的尽可能多的「可预测成分」，否则 loss 不会下降。这是 EMA 防塌缩的统计学根据。

### 3.8 与 I-JEPA 的关键差异总结

| 维度 | I-JEPA | V-JEPA |
|---|---|---|
| 模态 | 静态图像 | 视频 |
| Patch 化 | 2D patch（$H \times W$） | 3D spatio-temporal patch（$T \times H \times W$） |
| Mask 策略 | 单帧 square mask | 时空多块 mask（短程 15% × 8 + 长程 70% × 2） |
| 总 mask ratio | ~75% | ~90% |
| 损失 | L2 | L1（更稳定） |
| Backbone | ViT-H/14 | ViT-L/16 / ViT-H/16 |
| 评估 | ImageNet-1K linear | K400 + SSv2 + ImageNet-1K 三类任务联合评估 |

两者在方法论上几乎完全一致，主要差异是「视频带来 patch 维度提升」和「时空冗余迫使更激进的 mask 策略」。

---

## 4. 数据集与实验设置

### 4.1 数据集：VideoMix2M

V-JEPA 的预训练数据被统称为 **VideoMix2M**，由以下公开数据集混合而成：

- **HowTo100M**（约 130 万段教学视频，136M clip）
- **Kinetics-400/600/700**（约 50–70 万段动作分类视频）
- **Something-Something-v2**（约 22 万段细粒度动作视频）

作者把 HowTo100M 的视频按 Kinetics 和 SSv2 类别去掉重叠验证集，最终聚合约 200 万 videos。训练时一个 sample 是 16 帧 clip、224×224（ViT-L/16）或 384×384（ViT-H/16），帧率约 4 fps，对应 3 秒视频。

### 4.2 训练超参

| 超参 | ViT-L/16 | ViT-H/16 |
|---|---|---|
| Patch | 16×16 | 16×16 |
| 分辨率 | 224×224 | 384×384 |
| 时空 token 数 | $(T/2) \times (H/16) \times (W/16)$ | 同 |
| Batch size | 3072 | 2400 |
| 迭代次数 | 90K | 90K |
| Optimizer | AdamW | AdamW |
| 学习率 schedule | Cosine | Cosine |
| EMA $m$ | 0.999 → 1.0 | 同 |
| Loss | L1 | L1 |

### 4.3 评估协议

- **Frozen evaluation**：冻结 $E_\theta$，在输出表征上训线性分类头（linear probe）或 4-layer attentive probe。视频任务采用 average pooling 帧间 token。
- **End-to-end fine-tuning**：解冻全部权重做 fine-tune。
- **评测任务**：
  - **Kinetics-400**：场景级动作分类（外观主导）
  - **Something-Something-v2**：细粒度动作分类（运动主导）
  - **ImageNet-1K**：图像分类（迁移能力，验证 frozen 表征的通用性）

---

## 5. 实验结果

### 5.1 主要定量结果（Frozen Evaluation）

下表汇总 V-JEPA 在三类任务上的 frozen 协议结果（论文 Table 6）：

| 方法 | Backbone | 帧 | 训练数据 | K400 | SSv2 | ImageNet-1K |
|---|---|---|---|---|---|---|
| I-JEPA（图像版） | ViT-H/16 | — | IN-1K | — | — | 77.5 |
| **V-JEPA** | **ViT-L/16** | **16 | **VideoMix2M | **75.6** | **65.4** | — |
| **V-JEPA** | **ViT-H/16** | **16 | **VideoMix2M | **81.9** | **72.2** | **77.9** |
| VideoMAEv2 | ViT-H/16 | 16 | VideoMix2M | — | — | ~70 |
| OmniMAE | ViT-H/16 | — | ImageNet-22K | 70+ | 60+ | — |
| Hiera | ViT-H/16 | — | K4 | 71 | 60+ | — |
| DINOv2（图像） | ViT-g/14 | — | LVD-142M | ~83 | ~55 | 83+ |

> 注：原 Table 6 中 Video Feature Pred 类方法（含 V-JEPA）在 SSv2 比 Image Models 类高 ~15-20 个百分点；在 K400 上接近或超越图像模型。这是 V-JEPA 的核心主张。

![](assets/V-JEPA%20-%20视频潜空间预测表征学习/figures/01-teaser.png)

> **图 3（论文 Figure 1）：V-JEPA 在 K400 与 SSv2 上的 frozen-evaluation 性能散点图。** 圆圈（蓝）= Video Feature Pred（V-JEPA 系列），三角 = Image Models（DINOv2 / I-JEPA / OpenCLIP），方块 = Video Pixel Pred（VideoMAE / Hiera / OmniMAE）。**V-JEPA 用 frozen backbone 同时站到 K400 与 SSv2 的右上角**：在运动任务（SSv2 70-72%）比 DINOv2 等图像模型高 15-20 个百分点；在外观任务（K400 81%）接近甚至超越 SOTA 任务专用模型。这是 V-JEPA 「feature prediction as stand-alone objective」最直观的可视化证据。
> 来源：论文 Figure 1，第 1 页，<https://arxiv.org/abs/2404.08471>

### 5.2 vs Pixel Reconstruction（VideoMAE / Hiera / OmniMAE）

论文 Table 6 + Table 7 直接给出了 Video Feature Pred vs Video Pixel Pred 的 head-to-head：

- **Frozen protocol**：V-JEPA 显著优于所有 pixel-pred 方法（ViT-H/16 81.9 vs VideoMAEv2 76+ on K400；72.2 vs 60+ on SSv2）。这与 I-JEPA 论文在 ImageNet 上的发现一致——特征预测学到的表征更通用、更可迁移。
- **End-to-end fine-tuning**：两者差距缩小（V-JEPA ViT-H/16 fine-tune K400 接近 84%，VideoMAEv2 fine-tune 约 83%），但 V-JEPA **训练 schedule 显著更短**（90K iter vs 240K+ iter for VideoMAE）。
- **Label-efficient regime**（论文 Table 7）：当下游标注数据减少时，V-JEPA 与 pixel-pred 的差距**进一步拉大**，证明 feature prediction 的表征更稳健、不依赖大量下游数据。

### 5.3 vs Image Models（DINOv2）

V-JEPA 与 DINOv2 的对比是论文反复强调的故事线：

- **K400（外观）**：DINOv2 ViT-g/14 ~83%，V-JEPA ViT-H/16 81.9%——图像模型略强（毕竟 DINOv2 训练了 142M 图），但差距很小；
- **SSv2（运动）**：DINOv2 ~55%，V-JEPA **72.2%**——V-JEPA 拉开 17 个百分点。
- **ImageNet-1K**：V-JEPA 77.9%，I-JEPA 77.5%，DINOv2 83%+ —— V-JEPA 与 I-JEPA 接近，输给用海量图训练过的 DINOv2。

**核心叙事**：图像自监督模型缺乏时序信号迁移能力；纯视频 pixel-pred 模型缺乏语义抽象；**只有 video feature pred 同时拿到两类任务的 high performance**。

### 5.4 关键消融

论文 Table 1 给出了**最核心的一句话证据**：

| 预测目标 | Backbone | K400 | SSv2 |
|---|---|---|---|
| **Pixel space** | ViT-L/16 | 73.5 | 60.2 |
| **Feature space (V-JEPA)** | ViT-L/16 | **75.6** | **65.4** |

**把预测目标从像素换成表征空间，K400 +2.1、SSv2 +5.2**——这就是 feature prediction as a stand-alone objective 的最强单一证据。

### 5.5 计算效率

论文没有给精确的 GPU hours 数据，但与 I-JEPA 类似：

- ViT-L/16: 90K iter × batch 3072 × 224×224 × 16 帧 ≈ 32 × A100 / 数天
- ViT-H/16: 90K iter × batch 2400 × 384×384 × 16 帧 ≈ 64 × A100 / 一周

比 pixel-pred 路线（VideoMAE ViT-H/16 需 240K iter）训练时长**缩短约 2.5 倍**。

---

## 6. 与相关工作的关系

V-JEPA 在自监督视频表征学习坐标系里的位置，可以拆成四条轴：

- **vs I-JEPA**：方法论同源（潜空间预测 + EMA），patch 维度从 2D 升级到 3D。V-JEPA 证明了 I-JEPA 的设计原则对视频同样有效。
- **vs VideoMAE / Hiera（生成式 / pixel-pred）**：V-JEPA 在 frozen 协议下显著领先，fine-tune 协议下两者接近但 V-JEPA 训练更短、label-efficient 优势明显。
- **vs DINOv2 / I-JEPA（图像自监督迁移）**：DINOv2 在外观任务（K400）强但在运动任务（SSv2）弱；V-JEPA 反过来。互补而非替代。
- **vs V-JEPA 2（V-JEPA 的 scale up + 动作条件版）**：V-JEPA 2 用 1B 参数 + 1M 小时视频重做 V-JEPA，并加 V-JEPA 2-AC 做零样本机器人规划。V-JEPA 是它的概念先导。

V-JEPA 与「生成式视频世界模型（Cosmos、Sora、Marble、VPP）」是**互补而非对立**：

- V-JEPA 主动丢弃不可预测的像素细节，只学可预测的语义，因此高效、防塌缩；
- 生成式路线显式预测未来帧，保留细节、可直接可视化、携带互联网规模物理先验。
- V-JEPA 适合当「感知 / 规划 backbone」，生成式适合当「未来想象器」。

---

## 7. 局限与批判性评价

1. **「stand-alone feature prediction」的代价是更激进的 mask engineering**：V-JEPA 把像素重建的负担转嫁给了时空多块掩码设计（短程 15% × 8 + 长程 70% × 2），某种意义上只是把「像素先验」换成了「掩码先验」。论文对 mask 比例 / 长程 vs 短程比例的消融只在 Appendix B 简略带过，没有系统性的 mask recipe 搜索。
2. **L1 vs L2 的理论依据不充分**：作者说 L1 在视频上「more stable」，但缺乏消融对比 L1 vs L2 在 K400/SSv2 上的具体数字（只在 Table 1 提了一次）；也没解释为什么视频上 L1 比 L2 好而图像上 I-JEPA 反而用 L2。
3. **评测仍以标准 benchmark 为中心**：结论建立在 K400 / SSv2 / ImageNet-1K 这三个 frozen-evaluation 数字上，缺少在 dense prediction（目标检测、视频分割、光流估计）、、任务（机器人动作预测）上的直接验证。
4. **是「视频自监督表征」而非「视频世界模型」**：V-JEPA 只做视频里的时空块预测，**没有动作 / 控制变量、没有跨时间演化、没有不确定性建模**，离 LeCun 纲领里「可规划、可推理、模块化 JEPA」还差很远。要做规划需要后续的 V-JEPA 2-AC（V-JEPA 2 论文）。
5. **clip 长度受限**：16 帧 × 4 fps = 约 3 秒视频，远不够建模 long-form event（分钟级）。Mask 设计因此被迫用「覆盖整个时间维的大块」来增加难度，但 short-form 的时空冗余仍然是大问题。
6. **丢弃细节是一把双刃剑**：被主动抹掉的「精确纹理与外观细节」，对动作分类是好事，但对需要精确定位的稠密预测（如 video object segmentation、tracking）未必有利，论文着墨不多。

---

## 8. 复现与实践建议

- **算力门槛**：
  - ViT-L/16：约 32 × A100 / 数天
  - ViT-H/16：约 64 × A100 / 一周
  - Frozen linear probe：单 GPU 即可
- **数据准备**：VideoMix2M 需自己下载 HowTo100M、K400/600/700、SSv2 并去重合并，单卡存储约 5-10 TB，下载是主要瓶颈
- **代码**：官方实现 <https://github.com/facebookresearch/jepa> 已开源；mask sampling 只需修改 I-JEPA 的 sampler 从 2D 升到 3D
- **实现要点**：
  - 时空 patch 化用 $(T/2) \times (H/16) \times (W/16)$ 的 grid，不能直接 flatten 成 1D
  - Predictor 必须保持窄宽度（384），加宽会掉点
  - Target encoder EMA momentum 从 0.999 升到 1.0，务必保留
  - Mask sampling 的短程 / 长程比例 + 数量是核心超参
  - L1 loss 而不是 L2
- **常见踩坑**：
  - 直接用 I-JEPA 的 mask sampler 会丢失「长程 mask」的设计，必须自己写 3D 多块 sampler
  - Mask 作用在 y-encoder 输出而非输入上（最容易踩的坑）
  - Frame 采样如果用均匀采样而非 4 fps 截帧，会破坏 clip 间的时序一致性

---

## 9. 个人启发与后续问题

1. **「feature prediction as a stand-alone objective」在视频上成立**：V-JEPA 用最朴素的方式（潜空间预测 + L1 + EMA）证明了视频自监督学习不需要像素重建、对比损失、负样本、pretrained image encoder、人工标注也能跨「运动」与「外观」两类任务同时刷 SOTA。这是 I-JEPA 在视频域的成功延伸，为后续 V-JEPA 2 的 scale up 铺好了方法论基础。
2. **「时空 patch 化 + 时空多块 mask」是视频版 JEPA 的两个核心适配**：把 I-JEPA 的 2D patch / square mask 升级为 3D spatio-temporal patch / multi-block mask 是 V-JEPA 能 work 的关键工程改动。这两个改动未来应该成为「视频版 MAE」类工作的标配。
3. **V-JEPA 与生成式视频世界模型互补而非对立**：V-JEPA 回答「如何高效学到可迁移的视频表征」，生成式视频模型回答「如何把物理直觉用于未来想象」。V-JEPA 适合当世界模型的**感知 backbone / 表征空间**，生成式视频模型适合当**未来模拟器**。V-JEPA 2-AC 的出现证明 V-JEPA 表征可以天然接动作条件 + MPC 做规划，验证了这条互补路径的工程价值。
4. **「丢弃不可预测的细节」的视频版解读**：视频相比图像有更严重的「不可预测细节」问题（光照闪烁、相机抖动、压缩伪影、遮挡），潜空间预测的「丢弃细节」优势在视频上被进一步放大。这条原则可以推广到其他高噪声模态（雷达、LIDAR、EEG 等），不一定只限于视频。
5. **未解问题**：
   - L1 vs L2 的理论依据是什么？为什么视频偏好 L1 而图像偏好 L2？
   - 时空 mask ratio 的最优值与「视频内容熵」是否有关？是否存在自适应 mask 策略？
   - V-JEPA 表征能否直接替代 VPP 里的 SVD 表征，作为更高效的视频条件编码器？
   - 把动作 / 控制变量注入 context，能否把 V-JEPA 直接升级成一个可规划的 video world model？（V-JEPA 2-AC 已迈出第一步）
   - V-JEPA 的 frozen 表征在 dense prediction（分割、光流、深度）上的迁移性如何？是否真的「丢了细节」？

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2404.08471> | 全文（含附录） | 2026-09-03 | 精读主来源 |
| `figures/01-teaser.png` | Figure 1：K400 vs SSv2 frozen evaluation 散点图 | arXiv:2404.08471 | 第 1 页 | 2026-09-03 | 概述 / 主要定量结果 |
| `figures/02-jepa-architecture.png` | Figure 2：JEPA 通用架构图 | arXiv:2404.08471 | 第 3 页 | 2026-09-03 | 核心方法（JEPA 原理） |
| `figures/03-method-pipeline.png` | Figure 3：V-JEPA 训练流程（含 mask token、predictor、L1、EMA target encoder） | arXiv:2404.08471 | 第 4 页 | 2026-09-03 | 核心方法（视频版 pipeline） |
| <https://github.com/facebookresearch/jepa> | 官方代码仓库（与 I-JEPA 同源） | — | — | 2026-09-03 | 复现参考 |
| <https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/> | 官方项目页 | — | — | 2026-09-03 | 论文/博客补充 |