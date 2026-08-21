---
type: paper-note
status: done
domain: Foundations
paper: ViT³: Unlocking Test-Time Training in Vision
year: 2025
arxiv: "2512.01643"
doi: null
source: https://arxiv.org/abs/2512.01643
project: null
code: https://github.com/LeapLabTHU/ViTTT
tags:
  - Foundations
created: 2026-05-28
updated: 2026-08-21
---

# ViT³：视觉测试时训练序列建模

## 基本信息

- **论文正式名称**：ViT³: Unlocking Test-Time Training in Vision
- **方法别名**：ViTTT / ViT³ / Vision Test-Time Training
- **作者**：Dongchen Han, Yining Li, Tianyu Li, Zixuan Cao, Ziming Wang, Jun Song, Yu Cheng, Bo Zheng, Gao Huang（清华；阿里）
- **arXiv ID**：2512.01643（v2）
- **代码仓库**：<https://github.com/LeapLabTHU/ViTTT>
- **所属领域**：基础模块（视觉序列建模 / 高效骨干网络）
- **归类原因**：它对"视觉序列建模"这一基础问题给出了一条独立于 Softmax 注意力、线性注意力与状态空间模型的新范式，并系统给出设计原则。

## 一句话结论

ViT³ 把视觉序列建模重新表述为"对键值对做测试时在线学习一个内层模型"的过程，配套给出 6 条视觉专属设计规则与一个由 GLU + 深度卷积构成的内层模型，从而在保持线性复杂度的同时，在分类、检测、分割、生成四类任务上一致地匹配或超越 Mamba / 线性注意力等强基线，并在高分辨率下取得 4.6× 速度提升与 90.3% 显存下降。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

视觉骨干网络的核心计算单元是序列建模。给定图像被切成的 $N$ 个 token（含 patch token 与/或特征 token），主流 Vision Transformer 在每一层都做一次 Softmax 注意力，其计算量随序列长度呈 $\mathcal{O}(N^2)$ 增长。当输入分辨率从 224² 提升到 1248² 时，token 数 $N$ 增加到数千甚至更多，self-attention 的显存与浮点开销迅速成为瓶颈。这意味着在检测、分割、长视频等天然长序列任务中，"用更大的 backbone 看得更清楚"和"在 1248² 上做端到端训练"之间存在直接冲突。

论文关心的核心问题是：**在保持全局序列建模能力的前提下，能否换掉 Softmax 注意力，把视觉序列建模做成一个可线性扩展、又具备足够表达力的计算单元？**这条问题在 NLP 侧曾被线性注意力和 Mamba 类工作部分回答，但在视觉侧一直缺乏系统的设计指南与可复现基线。

### 1.2 现有方法的瓶颈

围绕上述问题，近年工作大致走三条线，但都各有局限：

1. **Softmax Attention（ViT / Swin / CSWin 等）**：表达力强、训练稳定，但 $\mathcal{O}(N^2)$ 复杂度在高分辨率下不可承受。FlashAttention 等工程优化只能"压榨常数"，无法改变增长阶。
2. **线性注意力（Performer、SOFT、RetNet 等）**：用 $K^{\top}V$ 把上下文压缩为 $d \times d$ 线性层，复杂度降到 $\mathcal{O}(N)$。但压缩后的状态容量固定，朴素乘法压缩常常丢信息，论文明确指出"有限的线性状态"是其在视觉上表现不佳的主要原因。
3. **状态空间 / Mamba 类（Vim、VMamba 等）**：用选择性状态空间把序列建模写成固定大小状态的递归更新，表达力优于纯线性注意力，但仍是"用一个固定向量承载整个上下文"的路子，且对视觉全局与局部融合的适配需要额外设计。

(个人判断) 这三条线有一个共同点——都把"上下文"压缩成某种固定形态的载体（softmax 概率、$d \times d$ 矩阵、固定状态向量）。一旦载体表达能力受限，再怎么调优都难以同时获得"长上下文 + 强表达 + 易优化"三项。TTT 路线的关键差异在于，它允许载体本身是**一个可训练的小网络**，从而把"如何用权重编码上下文"变成一个可学习问题。

### 1.3 本文核心贡献

论文的贡献可以拆成两件：

- **系统经验研究**：在视觉任务上对 TTT 设计空间做大规模消融，从内层损失、训练批量、训练 epoch、inner learning rate、inner model 容量、inner model 深度、inner model 架构、残差与初始化等多个维度比较，提炼出 **6 条可直接复用的设计规则**。这是论文最有"方法论价值"的部分。
- **可落地基线 ViT³**：把上述规则蒸馏成一个由"简化 GLU + 深度卷积"作为内层模型的纯 TTT 架构，并扩展为非层次化（ViT³）、4 阶段层次化（H-ViT³）、替换 DiT 中注意力（DiT³）三套家族，在分类、检测、分割、生成四类任务上验证。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

把单层 TTT 块（或 Transformer 块）视作黑盒，其输入输出与普通 ViT block 完全兼容：

- **输入**：上一层的 token 序列 $x \in \mathbb{R}^{N \times C}$，$N$ 是 token 数，$C$ 是 embedding 维度。在分类任务中 $N = HW/P^2$，$P$ 是 patch size；在 1248² 分辨率下，$P=16$ 时 $N \approx 6084$。
- **输出**：同一长度、同维度的 token 序列 $x' \in \mathbb{R}^{N \times C}$，再接残差和 FFN 形成一个 block。
- **内层假设**：在 block 内部，输入先经 Norm + 三个独立线性投影得到 $Q, K, V \in \mathbb{R}^{N \times d}$，$d$ 是 head 维度。$K$ 和 $V$ 被当作"在线训练数据集"$\mathcal{D} = \{(K_i, V_i)\}_{i=1}^N$，用来更新内层模型 $\mathcal{F}_W: \mathbb{R}^d \to \mathbb{R}^d$ 的权重 $W$；更新后的 $W^*$ 再作用于 $Q$ 得到输出。

论文默认采用的"内层"设定为：**单轮（1 epoch）全批量梯度下降、inner learning rate $\eta = 1.0$、Dot Product 内层损失**，这些值在 6 条设计规则中通过消融确定。

### 2.2 关键符号和目标函数

- **外层（整个 ViT³ 训练）**：标准监督学习损失 $L_{\text{outer}}$（如分类交叉熵、Mask R-CNN 损失、扩散 MSE）。外层梯度会通过内层展开（unroll）反传到 $W_0$、$W_Q, W_K, W_V$ 等外部参数。
- **内层（每次前向中的在线学习）**：给定一个小批 $\mathcal{B} \subseteq \{1, \ldots, N\}$，先做内层前向 $\hat{V}_\mathcal{B} = \mathcal{F}_W(K_\mathcal{B})$，再用自监督损失 $\mathcal{L}(\hat{V}_\mathcal{B}, V_\mathcal{B})$ 计算梯度，沿负梯度方向更新 $W$。这是"在线学习 inner model"的核心操作。
- **复杂度目标**：若 $\mathcal{F}_W$ 本身是线性复杂度模块（论文最终用 depthwise conv + GLU），则 TTT 层继承 $\mathcal{O}(N)$ 的时间与显存复杂度。

---

## 3. 核心方法

### 3.1 总体框架：从 Softmax Attention 到 TTT

![](assets/ViT³%20-%20视觉测试时训练序列建模/figures/01-overview.png)

> 图 1：Softmax 注意力、线性注意力与 TTT 三类序列建模方式的统一对比。(a) Softmax 注意力可视为一个宽度为 $N$ 的两层 MLP，复杂度 $\mathcal{O}(N^2)$；(b) 线性注意力把 $K, V$ 压缩为 $d \times d$ 线性权重，复杂度 $\mathcal{O}(N)$，但状态容量受限；(c) TTT 把 $K, V$ 视为训练数据，通过自监督在线学习更新 inner model 的权重 $W^*$，再作用于 $Q$。  
> 来源：论文 Figure 1，第 2 页，<https://arxiv.org/abs/2512.01643>

这张图把"视觉 TTT 该长什么样"在概念层交代清楚。它把 Softmax 注意力、线性注意力、TTT 三条线并排画在同一张图上，统一视角是"用一个映射作用于 $Q$"：

- Softmax 注意力等价于一个宽度为 $N$ 的两层 MLP，参数由 $K, V$ 直接给出，激活是 Softmax；
- 线性注意力把这个 MLP 简化为一个 $d \times d$ 线性层（$K^\top V$）；
- TTT 进一步把映射从"线性层"推广为"任意可微模块 $\mathcal{F}_W$"，权重 $W$ 通过在 $(K, V)$ 上做几次自监督在线学习得到。

### 3.2 TTT 层的数学重新表述

**Softmax 注意力等价于一个宽度 $N$ 的 MLP。** 把 $Q = xW_Q$、$K = xW_K$、$V = xW_V$ 代入，有

$$
O = \sigma(QK^\top) V \triangleq \sigma(QW_1)W_2 = \mathrm{MLP}(Q),
$$

其中 $W_1 = K^\top$、$W_2 = V$、$\sigma$ 是逐行 Softmax。含义是：Softmax 注意力对 $Q$ 做一次宽度为 $N$ 的 MLP，隐藏维度是序列长度本身。

**线性注意力等价于一个 $d \times d$ 线性层。** 用线性核 $\phi(\cdot)$ 替代 Softmax，可得

$$
O = Q(K^\top V) \triangleq QW = \mathrm{FC}(Q),
$$

$K^\top V$ 是 $d \times d$ 矩阵，复杂度降到 $\mathcal{O}(N)$。但这个线性层容量是 $d \times d$，再大就涨不上去了。

**TTT 把它推广为任意 inner model。** 把映射推广为 $\mathcal{F}_W: \mathbb{R}^d \to \mathbb{R}^d$，权重 $W$ 通过自监督在线学习得到：

$$
\hat{V}_\mathcal{B} = \mathcal{F}_W(K_\mathcal{B}),\quad W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}(\hat{V}_\mathcal{B}, V_\mathcal{B})}{\partial W}.
$$

只要 $\mathcal{F}_W$ 本身是线性复杂度的（如 4 维扩展的两层 MLP、深度卷积、GLU 等），整个 TTT 层就继承 $\mathcal{O}(N)$ 时间与 $\mathcal{O}(N)$ 显存复杂度。这一点是后续所有结果的理论基础。

外层梯度则通过"展开内层更新"反传到外部参数。作者把内层前向–反向–再前向三步看成一次"虚拟训练"，让它和外部反向传播端到端打通，从而可以用标准优化器训练整个 ViT³。

### 3.3 视觉 TTT 的 6 条设计规则

这是论文方法论价值最高的部分。它把"视觉 TTT 到底该怎么设"这一开放问题压成 6 条具体规则，每条都对应一个消融表。

**规则 1：避免二阶混合导数消失的损失函数。** 反向传播到值投影矩阵 $W_V$ 的外循环梯度涉及内层损失对 $(\hat{V}, V)$ 的混合二阶导数。若该混合导数接近零，外层信号在反向穿过内层后消失。作者在 Table 1 中实测了 5 种损失：

| 内层损失 | Top-1 (%) |
|---|---|
| Dot Product | 78.9 |
| MSE (L2) | 79.2 |
| RMSE | 78.8 |
| MAE (L1) | 76.5 |
| Smooth L1 | 78.1 |

MAE 的导数是符号函数，混合二阶导几乎为零，因此表现最差。Smooth L1 在部分区间也存在同样问题，逊于 Dot Product / MSE。**说明视觉 TTT 应该避免 L1 系损失。**

**规则 2：单 epoch + 全批量是视觉上的甜点。** 与语言建模中"小 mini-batch 多 epoch"的经验相反，论文在 Table 2 中显示：

| Epoch | Batch Size | Top-1 (%) |
|---|---|---|
| 1 | N (full) | **78.9** |
| 1 | N/2 | 78.6 |
| 1 | N/3 | 78.3 |
| 1 | N/4 | 78.1 |
| 2 | N | 79.1 |
| 3 | N | 79.2 |
| 4 | N | 57.0\*（发散） |

(个人判断) 视觉数据没有因果结构，顺序 mini-batch 引入的位置偏置并不自然；多 epoch 虽然把 Top-1 推高 0.2–0.3 个点，但显著增加单步计算量，4 epoch 时训练已经发散。因此论文把"单 epoch 全批量"作为默认设定。

**规则 3：inner learning rate 取 1.0 附近。** Table 3 的扫描结果（$\eta \in \{0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0\}$）显示 1.0 与 2.0 同分（78.9），更高就开始发散。**说明视觉 TTT 偏好较大的内层学习率**，与 NLP 中常见的 $\eta \ll 1$ 显著不同。一种解释是视觉序列更短、batch 更大，单步梯度方差小，可以承受更大步长。

**规则 4：inner model 容量增大能稳定提升精度。** Table 4 中把两层 MLP 的扩展比从 1 提到 4，Top-1 从 78.9 提到 79.6。这条规则对应 TTT 相对线性注意力的根本优势——inner model 不必是 $d \times d$ 矩阵，可以是任意更复杂的非线性映射。代价是单次前向–反向–前向大约相当于 4 次前向的 FLOPs。

**规则 5：当前设定下深 inner model 难以优化。** 把两层 MLP 换成三层 MLP 反而让 Top-1 从 78.9 掉到 77.5（图 3 的消融把这一点展示得最直观）。论文尝试了残差连接、特殊初始化等手段（Table 6），都没有超过"约束设计"$\mathcal{F}_W(x) = \mathrm{SiLU}(\mathrm{FC}(x))$。这说明在 ViT³ 的训练框架下，inner model 还没把"更深带来的指数级更强表达力"兑现出来，是公开的开放问题。

**规则 6：卷积型 inner module 特别适合视觉。** Table 4 中 3×3 深度卷积在参数量最小（22.9M）、FLOPs 最低（4.25G）的情况下拿到最高 Top-1（80.1%）。原因有两点：(a) 视觉数据有强烈的局部性，卷积天然编码这种归纳偏置；(b) 把全局上下文 $K, V$ 压缩进一个卷积核，等于让"全局信息通过权重"和"局部信息通过感受野"两条路同时起作用，**全局与局部显式解耦地共存于 inner model 内部**。

### 3.4 ViT³ 架构与内层模型

![](assets/ViT³%20-%20视觉测试时训练序列建模/figures/02-method.png)

> 图 2：TTT 块与 Transformer 块的宏观对比。宏观上，TTT 块沿用 Norm + TTT Block + 残差 + FFN + Norm 的结构；块内部用三个线性层生成 $K, V, Q$，中间的 "TTT Calculation" 即"用 $K, V$ 在线更新 inner model 权重、再作用于 $Q$"那一步。  
> 来源：论文 Figure 2，第 2 页，<https://arxiv.org/abs/2512.01643>

这张图说明 ViT³ 之所以能"即插即用"——它在外形上完全沿用 Transformer block，只把内部的"注意力计算"换成"TTT Calculation"。因此 ViT³ 可以直接替换任何基于 Transformer 的视觉 backbone 的 attention slot，包括分类骨干、检测/分割 backbone、DiT 注意力。

最终 ViT³ 的 inner model 用了两个被规则 4、5、6 共同选出的模块：

- **简化 GLU**：$\mathcal{F}_1(x) = \mathrm{FC}(x) \odot \mathrm{SiLU}(\mathrm{FC}(x))$。门控结构在几乎不增加参数的情况下把朴素 $d \times d$ 状态的容量翻倍，且保持易优化。
- **深度卷积**：$\mathcal{F}_2(x) = \mathrm{DWConv}(x)$。3×3 深度卷积引入局部性，规则 6 证明它在视觉上效果最好。

**多头组合方式**：在每个 TTT 块内，作者把 $H$ 个 head 分成两部分——一个 head 用 $\mathcal{F}_2$ (DWConv)，其余 $H-1$ 个 head 用 $\mathcal{F}_1$ (GLU)。这种"局部 head + 全局 head"的混合，相当于把"局部与全局"显式地分散到不同 head 上，而不是让一个 inner model 同时承担两种归纳偏置。

模型家族上，论文给出三套：

- **ViT³**（非层次化）：和 DeiT 同款，patch 化后接若干 block，加分类头。Tiny / Small / Base 三档。
- **H-ViT³**（4 阶段层次化）：和 Swin / VMamba 同款，4 阶段下采样，每阶段内堆叠若干 TTT block，可作为检测 / 分割 backbone。
- **DiT³**（用于生成）：把 DiT 中的 Softmax 注意力替换为 TTT block，保留 AdaLN 等结构，用于 class-conditional 图像生成。

### 3.5 训练目标与计算复杂度

外层损失 $\mathcal{L}_{\text{outer}}$ 取决于下游任务（分类用交叉熵，检测用 Mask R-CNN 多任务损失，分割用像素级交叉熵，生成用扩散 MSE）。内层损失 $\mathcal{L}$ 选 Dot Product，即

$$
\mathcal{L} = -\frac{1}{B} \sum_{i \in \mathcal{B}} \langle \mathcal{F}_W(K_i), V_i \rangle.
$$

(个人判断) Dot Product 比 MSE 略低 0.3 个点（Table 1 中 78.9 vs 79.2），但作者最终选它，应是出于**稳定性与计算量**的折中——Dot Product 不需要显式做"差–平方–求和"，反向传播路径短，显存占用更友好。

单步 FLOPs 上，每个 TTT 块要算"内层前向 + 内层反向 + 内层再次前向 + Q 投影"四步；当 inner model 是线性复杂度模块时，总复杂度为 $\mathcal{O}(N)$。具体到 Tiny 模型，论文把"一次前向–反向–前向"折算为约 4 次前向等效 FLOPs（见 Table 4 的 FPS 数据），与 Softmax 注意力的 1 次前向相比代价上升，但因常数项小、且与序列长度解耦，在长序列下反而占优。

### 3.6 推理流程与可并行性

推理时，"在线学习"被展开到计算图中：

1. 取当前 block 输入 $x_{l-1}$，经 Norm 与三个线性层得到 $Q, K, V$。
2. 初始化 inner model 权重 $W_0$（外层学到的可学习参数）。
3. 在 $(K, V)$ 上做 1 步 full-batch 梯度下降，得到 $W^*$。
4. 用 $W^*$ 作用于 $Q$，得到 $O$。
5. 残差、FFN、下一层 block。

由于 full-batch + 1 epoch + 较大的 inner learning rate，整个内层更新可写成两个矩阵乘的"closed-form-like"操作：前向计算 $\hat{V} = \mathcal{F}_W(K)$、反向计算更新后的 $W$、再用更新后的 $W$ 做第二次前向。每一步都只是矩阵乘加，没有序列依赖，因此整层在 GPU 上**可完全并行**，不依赖 RNN 式的串行扫描。这是 TTT 与传统 RNN-based online learning 的关键工程差异。

---

## 4. 数据集与实验设置

### 4.1 数据集与评测任务

论文在四类任务上系统评估，覆盖度比"只报 ImageNet"的工作更宽：

- **图像分类**：ImageNet-1K（1.28M 训练，50K 验证，1000 类），Top-1 准确率。
- **目标检测 / 实例分割**：COCO 2017，Mask R-CNN 框架，$1280 \times 800$ 输入，报 AP^b 和 AP^m。
- **语义分割**：ADE20K，UperNet 头，$512 \times 2048$ 输入，报 mIoU。
- **图像生成**：ImageNet-1K 类条件生成，$256^2$ 分辨率，FID-50K。

### 4.2 Baseline 与评价指标

分类对比覆盖四大类基线：

- **ConvNet**：ConvNeXt-T/S；
- **Transformer**：DeiT-T/S/B、Swin-T/S/B（隐含在 hierarchical 对比）；
- **Mamba**：Vim-T/S、VMamba-T/S/B、LocalVMamba-S；
- **Linear attention**：SOFT++ (T/S/B)、MILA-T/S/B（部分带 MESA 训练策略 ‡）。

检测对比 InternImage、VMamba、MILA 等；分割对比 VMamba、SOFT++、TransNeXt 等；生成对比 DiT-S/B 在 patch 8/4/2 三种粒度下的结果。

### 4.3 实现细节

- **分类训练协议**：基本沿用 Swin Transformer 的 300 epoch 从头训练。优化器 AdamW，weight decay 0.05，总 batch size 4096，初始学习率 $4 \times 10^{-3}$，前 20 epoch 线性 warm-up + 余弦衰减。数据增强：RandAugment、Mixup、CutMix、random erasing。带 ‡ 的结果额外加 MESA 训练策略。
- **检测**：Mask R-CNN 框架下，$1\times$（12 epoch）和 $3\times$（36 epoch）两种 schedule。
- **分割**：UperNet，标准 ADE20K 训练流程。
- **生成**：DiT 原配置，patch 粒度 2/4/8 三档，仅替换 attention block。
- **TTT 内层超参**：inner learning rate 1.0、1 epoch、full batch（即 batch 大小等于序列长度 $N$）、内层损失 Dot Product。
- **模型规模**：ViT³-T/S/B 分别约 6M/24M/90M；H-ViT³-T/S/B 分别约 29M/54M/94M；DiT³-S/B 与对应 DiT 配置参数相近。
- **效率测试**：单卡 RTX 3090，比较 FPS 与 per-image GPU 显存。

---

## 5. 实验结果

### 5.1 主要定量结果：图像分类

**(a) 层次化对比（Table 5）**

| 模型 | 类型 | #Params | FLOPs | Top-1 (%) |
|---|---|---|---|---|
| ConvNeXt-T | ConvNet | 29M | 4.5G | 82.1 |
| VMamba-T | Mamba | 31M | 4.9G | 82.5 |
| SOFT-S++ | Linear | 27M | 4.5G | 82.6 |
| MILA-T‡ | Linear | 25M | 4.2G | 83.5 |
| **H-ViT³-T** | **TTT** | **29M** | **4.9G** | **83.5** |
| **H-ViT³-T‡** | **TTT** | **29M** | **4.9G** | **84.0** |
| **H-ViT³-S** | **TTT** | **54M** | **8.8G** | **84.4** |
| **H-ViT³-S‡** | **TTT** | **54M** | **8.8G** | **84.9** |
| **H-ViT³-B** | **TTT** | **94M** | **16.7G** | **84.9** |
| **H-ViT³-B‡** | **TTT** | **94M** | **16.7G** | **85.5** |

**说明什么**：在 Tiny / Small / Base 三档上，H-ViT³ 的 Top-1 都达到或超过同体量 Mamba / 线性注意力对手，与 MILA 几乎打平，加上 MESA 训练策略后能进一步提升 0.3–0.6 个点。说明 **TTT 在 ImageNet 这一类"短序列中等任务"上至少与最强的线性方法并列**，但距 SwinV2 / TransNeXt 等"高度优化的 Transformer"仍有 1–2 个点（论文未披露具体对比表，论文定性承认"仍有差距"）。

**(b) 非层次化对比（Table 7）**

| 模型 | 类型 | #Params | FLOPs | Top-1 (%) |
|---|---|---|---|---|
| DeiT-T | Transformer | 6M | 1.2G | 72.2 |
| Vim-T | Mamba | 7M | 1.5G | 76.1 |
| **ViT³-T** | **TTT** | **6M** | **1.2G** | **76.5** |
| DeiT-S | Transformer | 22M | 4.6G | 79.8 |
| Vim-S | Mamba | 26M | 5.1G | 80.3 |
| **ViT³-S** | **TTT** | **24M** | **4.8G** | **81.6** |
| DeiT-B | Transformer | 87M | 17.6G | 81.8 |
| **ViT³-B** | **TTT** | **90M** | **18.0G** | **82.6** |

**说明什么**：在非层次化架构上，ViT³ 相对 DeiT 有 +4.3 / +1.8 / +0.8 的提升（按 T/S/B 顺序），相对 Vim-T 也有 +0.4 的提升。这印证了"用 TTT 直接换掉 attention 即可获得明显收益"这一简单结论，也表明非层次化场景下，**TTT 的在线学习机制在保持线性复杂度的同时可以接近 Softmax 注意力的精度**。

### 5.2 目标检测与语义分割

**(a) COCO 检测（Table 8，$1\times$ schedule）**

| 模型 | FLOPs | AP^b | AP^m |
|---|---|---|---|
| InternImage-T | 270G | 47.2 | 42.5 |
| VMamba-T | 271G | 47.3 | 42.7 |
| **H-ViT³-T** | **271G** | **47.3** | **42.8** |
| InternImage-S | 340G | 47.8 | 43.3 |
| VMamba-S | 357G | 48.7 | 43.7 |
| **H-ViT³-S** | **349G** | **49.1** | **44.1** |
| VMamba-B | 485G | 49.2 | 43.9 |
| **H-ViT³-B** | **510G** | **50.0** | **44.6** |

在 $3\times$ schedule 下 H-ViT³-B 进一步达到 AP^b 51.0 / AP^m 45.3，超过 CSWin-B（50.8 / 44.9）和 VMamba-B（49.2 / 43.9）。

**说明什么**：当输入分辨率提升到 $1280 \times 800$、token 数 $N$ 远大于维度 $d$ 时，纯线性状态容量受限的弱点会被放大。H-ViT³ 借助 inner model 的非线性和在线学习机制，在长序列上**明显吃到了上下文压缩能力的红利**——这点和 ImageNet 上"打平"的结果形成对比。

**(b) ADE20K 语义分割（Table 9）**

| Backbone | 类型 | #Params | FLOPs | mIoU |
|---|---|---|---|---|
| VMamba-T | Mamba | 62M | 949G | 47.9 |
| SOFT-T++ | Linear | 60M | 948G | 46.5 |
| **H-ViT³-T** | **TTT** | **58M** | **946G** | **48.0** |
| LocalVMamba-S | Mamba | 81M | 1095G | 50.0 |
| SOFT-S++ | Linear | 81M | 1040G | 48.9 |
| **H-ViT³-S** | **TTT** | **84M** | **1026G** | **50.2** |
| TransNeXt-S | Transformer | 80M | 1089G | 52.2 |
| VMamba-B | Mamba | 122M | 1170G | 51.0 |
| SOFT-B++ | Linear | 121M | 1204G | 49.2 |
| **H-ViT³-B** | **TTT** | **124M** | **1195G** | **51.7** |
| TransNeXt-B | Transformer | 121M | 1268G | 53.0 |

**说明什么**：H-ViT³ 在三个尺寸上分别建立一个新的"线性复杂度 SOTA"，稳定超过 VMamba / SOFT++ / VVT。但 TransNeXt 仍以 1.3–1.5 mIoU 领先——这正是论文自己承认的"与高度优化 Transformer 仍有差距"。(个人判断) 在密集预测任务上，**目前 TTT 的最优 inner model（GLU + DWConv）仍属于"小容量表达"**，碰到需要更复杂上下文推理的场景（如 ADE20K 的长程依赖）就显得表达力不够，这和规则 5 的开放问题是同一根源。

### 5.3 图像生成

**(a) ImageNet $256^2$ 类条件生成（Table 10）**

| 模型 | #Params | FLOPs | FID↓ |
|---|---|---|---|
| DiT-S/8 | 33M | 0.36G | 153.60 |
| **DiT³-S/8** | 35M | 0.40G | 143.49 |
| DiT-S/4 | 33M | 1.41G | 100.41 |
| **DiT³-S/4** | 35M | 1.57G | 93.77 |
| DiT-S/2 | 33M | 6.06G | 68.40 |
| **DiT³-S/2** | 35M | 6.23G | 62.65 |
| DiT-B/8 | 131M | 1.42G | 122.74 |
| **DiT³-B/8** | 135M | 1.51G | 120.41 |
| DiT-B/4 | 130M | 5.56G | 68.38 |
| **DiT³-B/4** | 134M | 5.88G | 65.25 |
| DiT-B/2 | 130M | 23.01G | 43.47 |
| **DiT³-B/2** | 134M | 23.35G | 39.31 |

**说明什么**：把 DiT 中的 Softmax 注意力替换为 TTT block 后，所有 6 档配置 FID 都稳定下降（最大改善 10.11，发生在 DiT-S/8 上；最小 2.33，发生在 DiT-B/8 上），参数量与 FLOPs 的增加都很小（约 5%）。这说明 TTT **不是只对判别任务有效**——它同样能服务于扩散 Transformer 这种长上下文生成，验证了规则 6 的"卷积型 inner module 兼顾全局与局部"在生成场景下依然成立。(个人判断) 生成任务对全局上下文建模的要求比分类更高，DiT³ 一致地拿到 FID 提升，意味着 TTT 路线在"压缩长上下文但不丢信息"这一点上比 Softmax 注意力还有优势。

### 5.4 消融实验

**(a) 内层模型架构（Table 4 关键子集）**

| Inner model | #Params | FLOPs | FPS | Top-1 (%) |
|---|---|---|---|---|
| MLP, r1, l2 | 23.5M | 4.58G | 1315 | 78.9 |
| MLP, r4, l2 | 25.2M | 5.62G | 836 | 79.6 |
| SiLU(FC(x)) | 23.2M | 4.40G | 1456 | 79.4 |
| SwiGLU(x) | 23.8M | 4.75G | 1103 | 79.0 |
| FC(x)⊙SiLU(FC(x)) | 23.5M | 4.58G | 1194 | 79.7 |
| Conv(x) | 25.5M | 5.27G | 979 | 79.9 |
| **DWConv(x)** | **22.9M** | **4.25G** | **1366** | **80.1** |

**说明什么**：深度卷积在"参数量最小 + FLOPs 最低 + FPS 最高"的同时拿到最高 Top-1。说明**视觉的局部归纳偏置可以被 inner model 直接吸收**，根本不需要外层再叠一个 Local Attention 之类的额外结构。

**(b) 深层 inner model 优化困难（Figure 3）**

![](assets/ViT³%20-%20视觉测试时训练序列建模/figures/04-ablation.png)

> 图 3：inner model 层数对训练损失（左）与测试准确率（右）的影响。1 层（FC）训练损失最低、测试准确率最高；2 层（两层 MLP）次之；3 层（三层 MLP）训练损失反而最高、测试准确率最低。**说明在当前 TTT 训练框架下，更深的 inner model 既难优化（外循环），又难收敛到好的解（内循环）。**  
> 来源：论文 Figure 3，第 5 页，<https://arxiv.org/abs/2512.01643>

这张图把规则 5 的"深层 inner model 难以优化"从数字表格变成训练曲线。**3 层 MLP 的训练 loss 在全程高于 1 层 / 2 层**，最终测试准确率也最低——视觉上"模型越深越强"的常识在 ViT³ 当前的训练框架下并不成立。这正是论文公开承认的开放问题。

**(c) 残差与初始化（Table 6）**

| Inner model | Top-1 (%) |
|---|---|
| SiLU(xW₁)W₂ + x | 78.8 |
| SiLU(xW₁)(W₂ + I) | 79.1 |
| SiLU(xW₁)W₂, W₂ initialized as I | 79.0 |
| **SiLU(FC(x)) (constrained)** | **79.4** |

**说明什么**：在 2 层 MLP 上试了"残差、参数恒等映射初始化、输出层限制为单位映射"等标准技巧，没有一种超过"约束输出层为单位映射"的简单设计。这反向印证规则 5——深层 inner model 的优化困难**不是靠简单工程技巧就能解决**的，需要更系统的方案。

### 5.5 效率分析与失败案例

**(a) 高分辨率下的效率**

![](assets/ViT³%20-%20视觉测试时训练序列建模/figures/03-results.png)

> 图 4：DeiT-T 与 ViT³-T 在不同输入分辨率下的 (a) FPS（log scale）与 (b) 单图 GPU 显存占用。分辨率越高，DeiT-T 的 FPS 衰减越快、显存近线性增长；ViT³-T 始终保持在更高 FPS、显存增长极缓。在 $1248^2$ 分辨率（每图 6,084 tokens）下，ViT³-T 达到 **4.6× 速度提升**、**显存下降 90.3%**。  
> 来源：论文 Figure 4，第 8 页，<https://arxiv.org/abs/2512.01643>

这张图是 ViT³ 最常被引用的"工程卖点"图。**结论是清晰的**：在 $1248^2$ 上，DeiT-T 的 self-attention 已经非常吃力（显存 ~0.9 GB/图，FPS 跌到 1.3 左右），而 ViT³-T 由于线性复杂度保持高 FPS 与低显存。

**说明什么**：ViT³ 的"线性复杂度 + 在线学习"组合在**长序列视觉任务**下收益最大。这恰好解释为什么它在 COCO / ADE20K 这种高分辨率任务上吃到的红利比 ImageNet 多。

**(b) 失败案例 / 限制**

论文没有专门的"失败案例图"，但 6 条规则本身就指出了它的"不能用场景"：

- **inner model 不能太深**：3 层 MLP 在当前训练框架下反而比 1 层/2 层差。
- **内层损失不能用 L1 / Smooth L1**：会因二阶混合导数消失导致外层梯度无法反传。
- **mini-batch 多 epoch 不适用**：4 epoch 直接发散。
- **inner learning rate 太大不行**：5.0 / 10.0 训练不稳定。
- **in-context dynamic learning rate 在视觉上效果差**：$\eta_i = \eta \cdot \mathrm{Sigmoid}(x_i W_\eta)$ 这种 token-wise 自适应方案在实验中得分 78.7，低于固定 $\eta = 1.0$ 的 78.9。
- **对超长序列的鲁棒性**：论文未披露在 $N > 10000$（如 $2048^2$ patch=8、$4K$ 视频）的实验。

---

## 6. 与相关工作的关系

按"被本文如何引用"分组，相关工作大致分四块：

- **Softmax Attention 与 Vision Transformer**：经典 ViT、Swin、CSWin、TransNeXt 等。ViT³ 把它们视为 baseline，承认在分类和分割上与"高度优化的 Transformer"仍有差距。
- **线性注意力家族**：Performer、cosFormer、FLatten Transformer、SOFT++、MILA 等。ViT³ 继承了它们"$\mathcal{O}(N)$"的复杂度目标，但通过把"压缩对象"从 $d \times d$ 矩阵换成"可学习 inner model 权重"突破了表达力瓶颈。
- **状态空间 / Mamba 家族**：Vim、VMamba、LocalVMamba 等。ViT³ 与它们在分类和分割上正面 PK，H-ViT³ 在大部分尺寸上击败 VMamba。但范式上不同：Mamba 用选择性状态空间做固定向量的递归，TTT 用可学习 inner model 做"在线学习更新"——后者更接近元学习，前者更接近 RNN 的连续形式。
- **TTT 自身家族**：LaCT（语言建模）、One-minute video generation（一分钟视频）、TTT3R（3D 重建）等。ViT³ 自觉把这篇工作定位为"把 TTT 系统引入视觉领域"，补充 TTT 范式在视觉上的设计指南。

(个人判断) ViT³ 最有意思的对话对象其实是 Mamba——两者都强调"固定大小载体 + 长上下文"，但 Mamba 走的是"载体结构 + 选择性扫描"，ViT³ 走的是"载体可学习 + 一次性 closed-form update"。如果未来有工作能把 Mamba 的"选择性门控"和 TTT 的"在线学习权重"嫁接起来，视觉序列建模可能再上一个台阶。

---

## 7. 局限与批判性评价

论文作者明确承认的局限：

1. **深层 inner model 难以优化**。3 层 MLP 在当前框架下表现比 2 层差，标准残差 / 初始化都解决不了。这是规则 5 的开放问题，也是 ViT³ 相对 Transformer 的根本短板——**"我们能换掉 attention，但换不掉 attention 的深度"**。
2. **与高度优化的 Transformer 仍有差距**。在 ADE20K 上，H-ViT³-B 比 TransNeXt-B 仍低 1.3 mIoU。说明当前 TTT 范式能逼近但不能超越"经过多年工程打磨的 Softmax 注意力"在密集预测上的表现。
3. **inner model 增大的算力代价高**。一次前向–反向–前向约等于 4 次前向 FLOPs。要让 inner model 容量继续扩大，必须配套更轻量、更并行的 inner module 设计。
4. **视觉特定的 mini-batch 内层训练尚无答案**。论文用的是"全批量 1 epoch"——这是语言建模迁移过来的对比基线，并不是专门为视觉设计的。设计"因果+无序混合"的内层采样策略可能进一步提升表现。
5. **inner learning rate 动态化方案在视觉上失败**。$\eta_i = \eta \cdot \mathrm{Sigmoid}(x_i W_\eta)$ 这种输入依赖的方案拿到 78.7，低于固定 1.0 的 78.9，意味着 TTT 的 inner learning rate 在视觉上目前只能用"全局常数"。
6. **未在超长序列上验证**。论文最大分辨率是 $1248^2$（$N \approx 6084$），没有 $4K$ 视频、$2048^2$ 图像、3D token 序列的实验。要把 TTT 推进到视频 / 3D / world model 场景，**长序列稳定性仍是未知数**。
7. **缺乏与 FlashAttention 类工程实现的端到端对比**。ViT³ 的"线性 + 4×前向等效 FLOPs"是理论分析，论文未报告与 FlashAttention-2/3 的端到端 wall-clock 对比——而后者在工程侧已经把 attention 优化到非常接近线性实现的常数。

---

## 8. 复现与实践建议

- **代码可用性**：官方仓库 <https://github.com/LeapLabTHU/ViTTT> 已开源，包含 ViT³ / H-ViT³ / DiT³ 三套实现，分训练、评估、检测、分割、生成 5 个子目录。**论文未披露**专用预训练权重的 download link（README 未提供），但代码本身可以跑。
- **训练资源**：分类 300 epoch × 4096 batch，论文未披露 GPU 数量与单卡显存占用；按 Swin 同档规模估计，ViT³-B 的 300 epoch ImageNet 训练需要 16–32 张 A100 几天。检测和分割按对应基线（Mask R-CNN、UperNet）的资源放大。DiT³ 与 DiT 同档规模。
- **超参选择**：直接采用本文 6 条规则的默认值（Dot Product 损失 / 1 epoch 全批量 / $\eta = 1.0$ / GLU+DWConv 混合 / 2 层 inner MLP / 3×3 DWConv）即可达到论文 Table 5/7/9/10 的结果。
- **想自己改进时**：(1) 优先尝试**更轻的 inner module**（如可分离卷积、SRU 等），因为当前 inner model 的 4× 前向代价仍是主要瓶颈；(2) 在 inner model 深度上做谨慎实验，超过 2 层需要新的优化技巧；(3) 在超长序列上做实测，本文未在 $N > 6084$ 上验证；(4) inner learning rate 不要照搬 NLP 的小值，1.0 附近是合理起点。
- **不建议用 L1 / Smooth L1 作内层损失**——二阶混合导数消失会让外层梯度几乎归零，是最容易踩的坑。

---

## 9. 个人启发与后续问题

- **范式价值**：ViT³ 最值得借鉴的不是具体数字，而是**"把序列建模重写成在线学习 inner model"** 这一视角。它为"上下文压缩"问题打开了第三个维度——压缩对象不再必须是固定大小的向量或矩阵，而可以是一个**结构先验明确的可学习网络**。这条思路在视频、3D 点云、agent 轨迹等多模态长序列上有非常自然的迁移前景。
- **Mamba 与 TTT 的对话**：Mamba 用"选择性门控 + 状态空间"压缩上下文，ViT³ 用"在线学习 inner model"压缩上下文，两者的差异在于**状态是否可学习**。把两者结合是值得跟踪的方向。
- **inner model 的局部性红利**：规则 6 的"3×3 DWConv 在最小参数下拿到最高 Top-1"非常反直觉——直觉上"小模型不应该比 GLU 强"。可能的解释是：视觉的局部性是一种**强结构先验**，用极少参数就能编码，因此能"腾出容量给别处"。这条经验在视频帧间建模、点云邻域建模、agent 局部观察建模上都成立。
- **训练成本与推理成本要分开看**：ViT³ 的"4× 前向等效 FLOPs"指训练时；但推理时所有"在线学习"也被展开成计算图，等价于"两次前向 + 一次反向"。在高吞吐推理场景下，**这是否仍优于 FlashAttention-2/3 的硬件优化版本**，是工业落地前必须实测的问题。
- **可继续追问的问题**：
  1. 深层 inner model 的优化困难是否可以通过 inner learning rate schedule、layer norm、PreNorm / PostNorm 切换等标准技巧系统性解决？
  2. TTT 在视频、3D 视觉、agent 轨迹等"超长 + 局部性强"场景上是否能进一步放大优势？
  3. inner model 是否可以设计成"预训练一次、跨任务复用"的形态，把 TTT 推向"meta-sequence-modeling"方向？
  4. 把 TTT 与 FlashAttention-2/3 的硬件优化结合，wall-clock 上的真实优势有多大？

---

## 10. 材料来源

- **论文 PDF / HTML**：<https://arxiv.org/abs/2512.01643>，<https://arxiv.org/html/2512.01643>
- **代码仓库**：<https://github.com/LeapLabTHU/ViTTT>
- **方法图 / 数据**：Figure 1 (三类序列建模对比)、Figure 2 (TTT Block 宏观结构)、Figure 3 (inner model 深度消融)、Figure 4 (DeiT-T vs ViT³-T 效率对比) 均从 arXiv 官方 HTML 直接取图或下载 PNG，原始来源 <https://arxiv.org/html/2512.01643>。
- **本地图片**：
  - `assets/ViT³ - 视觉测试时训练序列建模/figures/01-overview.png`（Figure 1）
  - `assets/ViT³ - 视觉测试时训练序列建模/figures/02-method.png`（Figure 2）
  - `assets/ViT³ - 视觉测试时训练序列建模/figures/03-results.png`（Figure 4）
  - `assets/ViT³ - 视觉测试时训练序列建模/figures/04-ablation.png`（Figure 3，本次新增）
- **线索来源**：小红书分享《CVPR2026 最佳论文候选！清华阿里提出 ViT³》<http://xhslink.com/o/5YCCXMdtTwB>，作为本次精读笔记的初始信息入口；原帖 Web 侧需登录，仅作线索使用。
- **检索日期**：2026-08-21
