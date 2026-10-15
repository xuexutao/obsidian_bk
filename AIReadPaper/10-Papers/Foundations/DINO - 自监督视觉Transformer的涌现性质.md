---
type: paper-note
title: "DINO - 自监督视觉Transformer的涌现性质"
paper: "Emerging Properties in Self-Supervised Vision Transformers"
paper_title: "Emerging Properties in Self-Supervised Vision Transformers"
authors:
  - Mathilde Caron
  - Hugo Touvron
  - Ishan Misra
  - Hervé Jégou
  - Julien Mairal
  - Piotr Bojanowski
  - Armand Joulin
year: 2021
venue: ICCV 2021
arxiv: "2104.14294"
source: https://arxiv.org/abs/2104.14294
status: complete
domain: Foundations
tags:
  - self-supervised-learning
  - vision-transformer
  - knowledge-distillation
  - representation-learning
created: 2026-10-15
updated: 2026-10-15
---

# DINO：自监督视觉 Transformer 的涌现性质

## 基本信息

- **英文题目**：Emerging Properties in Self-Supervised Vision Transformers
- **作者**：Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin
- **机构**：Facebook AI Research、Inria、Sorbonne University
- **会议**：ICCV 2021，pp. 9650–9660
- **arXiv**：2104.14294v2，2021-05-24
- **论文类型**：视觉自监督学习 / 表征学习 / Vision Transformer
- **官方 PDF**：[arXiv PDF](https://arxiv.org/pdf/2104.14294)
- **ICCV 页面**：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)
- **官方实现**：[facebookresearch/dino](https://github.com/facebookresearch/dino)
- **本地原文**：[[assets/DINO - 自监督视觉Transformer的涌现性质/sources/paper.pdf]]
- **溯源清单**：[[assets/DINO - 自监督视觉Transformer的涌现性质/sources.md]]
- **精读日期**：2026-10-15

## 1. 一句话总结（TL;DR）

DINO 用“无标签自蒸馏”把同一图像不同视角上的 teacher 概率分布作为 student 的学习目标，并依靠 EMA teacher、multi-crop、teacher centering 与 sharpening 避免坍塌。
它不仅让 ViT 在 ImageNet 上得到很强的 k-NN/linear 表征，还使最后一层 `[CLS]` 自注意力在没有像素标签的情况下自动对齐对象区域与边界。

## 2. 研究背景与动机

### 2.1 论文在问什么

2021 年时，Vision Transformer 已证明可以把图像切成 patch 后作为 token 序列处理。
但早期 ViT 的成功高度依赖大规模有标签预训练。
与卷积网络相比，ViT 计算更贵、需要更多数据，其特征也尚未显出独有性质。
作者提出一个关键问题：视觉 Transformer 表现平平，是否部分源于其预训练仍依赖图像级监督？

### 2.2 为什么监督信号可能限制 ViT

一张图像包含对象、部件、材质、几何布局与背景关系。
ImageNet 式监督却把这些信息压缩成一个类别标签。
这个目标鼓励网络找到“足够分类”的证据，不要求保留完整场景结构。
相比之下，NLP 中 BERT/GPT 的成功与自监督预训练密切相关。
作者因此推测，自监督目标可能更适合释放 Transformer 的全局建模能力。

### 2.3 研究假设

第一，自监督 ViT 可能学到比监督 ViT 更丰富的空间语义。
第二，合适的自监督组件组合可能让 ViT 特征形成天然良好的邻域结构。
第三，ViT 没有卷积的强局部归纳偏置，训练目标可能比架构本身更决定 attention 学到什么。
第四，已有 SSL 方法中避免坍塌的复杂机制，未必都不可缺少。

### 2.4 最直观的涌现现象

![[assets/DINO - 自监督视觉Transformer的涌现性质/figures/figure-1-attention-emergence.png]]

> **图源**：Figure 1，PDF p.1，https://arxiv.org/pdf/2104.14294
> **解读**：DINO ViT-S/8 最后一层中，以 `[CLS]` 为 query 的自注意力自然集中到鸟、乐器、建筑、长颈鹿、船、自行车、狗和飞机等前景对象；训练从未使用类别或分割标签。

这张图是论文标题中“Emerging Properties”的第一层含义。
对象区域不是通过分割头、CAM 损失或像素监督得到的。
它直接存在于标准 ViT 的 self-attention 中。
但应谨慎表述：attention 的可视化显示了空间选择性，不等于严格的因果解释。

## 3. 核心贡献

1. **证明自监督训练能让 ViT 出现显式对象级空间结构。**
   最后一层 `[CLS]` attention 可直接产生相当清晰的对象区域，而监督 ViT 和卷积网络没有同样直接的接口。

2. **发现 DINO-ViT 特征异常适合非参数 k-NN。**
   ViT-S/8 在 ImageNet 上达到 78.3% k-NN top-1，接近其 79.7% linear probe，说明类别邻域已经在特征空间中自然组织起来。

3. **提出简洁的无标签自蒸馏框架 DINO。**
   student 跨视角预测 EMA teacher 的 sharpened distribution，不使用负样本、队列、显式聚类或 student predictor。

4. **系统识别关键训练组件。**
   momentum teacher、multi-crop、cross-entropy 与小 patch 是性能核心；centering 和 sharpening 的互补作用足以在 momentum teacher 下防止坍塌。

5. **验证表征的广泛可迁移性。**
   DINO 特征在复制检测、地标检索、视频对象分割、无监督对象发现和多数据集迁移上都表现强劲。

## 4. 方法论详解

### 4.1 整体框架：没有标签的 teacher–student 蒸馏

![[assets/DINO - 自监督视觉Transformer的涌现性质/figures/figure-2-dino-teacher-student.png]]

> **图源**：Figure 2，PDF p.2，https://arxiv.org/pdf/2104.14294
> **解读**：student 对视图 $x_1$ 产生分布 $p_1$，teacher 对另一视图 $x_2$ 产生经 centering 与 temperature softmax 的分布 $p_2$；teacher stop-gradient，参数由 student 的 EMA 更新。

DINO 的核心不是传统“大模型教小模型”。
teacher 与 student 采用相同网络架构。
两者参数不同，teacher 不是外部预训练模型。
student 通过反向传播更新。
teacher 不接收梯度，只由 student 历史参数的指数滑动平均更新。
因此 teacher 是训练过程中动态生成的时间集成目标。

网络写成：

$$g = h \circ f$$

其中 $f$ 是 ViT 或 ResNet backbone。
$h$ 是仅用于自监督预训练的 projection head。
下游任务丢弃 $h$，使用 backbone $f$ 的表征。

### 4.2 输出不是类别，而是潜在概率分布

给定图像视图 $x$，student 输出 $K$ 维 logits：

$$g_{\theta_s}(x) \in \mathbb{R}^{K}$$

student 的第 $i$ 个概率为：

$$P_s(x)^{(i)}=
\frac{\exp(g_{\theta_s}(x)^{(i)}/\tau_s)}
{\sum_{k=1}^{K}\exp(g_{\theta_s}(x)^{(k)}/\tau_s)}$$

teacher 有同形定义 $P_t(x)$，但使用自己的温度 $\tau_t$，并在 softmax 前减去 center。
这 $K$ 个维度不是 ImageNet 类别，也没有人工语义标签。
它们是网络自己形成的潜在输出坐标。

固定 teacher 时，基本蒸馏目标是：

$$\min_{\theta_s} H(P_t(x),P_s(x))$$

其中：

$$H(a,b)=-\sum_i a_i\log b_i$$

真正的 DINO 不比较同一输入，而是比较同一图像的不同视图。
这就得到 cross-view self-distillation。

### 4.3 Cross-view self-distillation 到底在约束什么

对同一原图生成视图集合 $V$。
其中包含两个 global views：$x_1^g,x_2^g$。
还包含若干分辨率更小、覆盖范围更局部的 local views。

所有视图都送入 student。
只有两个 global views 送入 teacher。
对每个 teacher global view，都要求 student 在其他视图上复现它的分布：

$$
\min_{\theta_s}
\sum_{x\in\{x_1^g,x_2^g\}}
\sum_{\substack{x'\in V\\x'\neq x}}
H(P_t(x),P_s(x'))
$$

注意求和条件 $x'\neq x$。
student 不是简单复制 teacher 对同一像素输入的输出。
它必须从另一个裁剪、尺度或颜色扰动中恢复一致语义。

“cross-view”的关键效果是消除视图特定噪声。
如果 global crop 看见整个对象，而 local crop 只看见部件，student 仍要匹配 global teacher 的语义分布。
这构成 local-to-global correspondence。

从信息角度看，student 被迫保留跨裁剪稳定的信息。
对象身份、部件关系与场景中的主要实体往往比背景纹理更稳定。
这为对象级 attention 涌现提供了直接训练压力。

### 4.4 Multi-crop：不仅是更强的数据增强

论文的基本设定使用两个 $224\times224$ global crops。
local crops 被缩放到 $96\times96$。
global crop 覆盖原图较大区域，local crop 覆盖较小区域。

multi-crop 的计算优势在于，小 crop 增加视角数却比再加一个 $224^2$ crop 便宜。
它的语义优势在于，模型必须把局部部件与全局语义对应起来。
所以 multi-crop 不是“多看几遍同一图”这么简单。
它改变了学习问题：目标从 global-to-global 一致性扩展到 local-to-global 一致性。

论文在计算实验中比较了：

- $2\times224^2$
- $2\times224^2+2\times96^2$
- $2\times224^2+6\times96^2$
- $2\times224^2+10\times96^2$

增加 local views 能显著提高单位时间内的精度。
但 local view 越多，显存需求也越高。
超过一定数量后收益递减。

multi-crop 不是对所有 SSL 框架都普遍有效。
DINO 在 300 epochs 下加入 6 个 local crops，linear 从 72.5% 提升到 75.9%。
MoCo-v2 也受益。
SwAV 受益。
作者的 BYOL-ViT 实现反而从 linear 71.4% 降至 64.8%。
这说明 multi-crop 与损失、teacher 稳定性和归一化机制存在耦合。

### 4.5 Teacher EMA：动态目标与时间集成

teacher 更新规则是：

$$\theta_t \leftarrow \lambda\theta_t+(1-\lambda)\theta_s$$

动量系数 $\lambda$ 按 cosine schedule 从 0.996 增长到 1。
训练初期 teacher 较快吸收 student 更新。
训练后期 teacher 变化越来越慢，提供更稳定的目标。

EMA 有三个作用。

第一，**去噪**。
单次 SGD 更新含有 minibatch 噪声，参数平均产生更平滑的模型。

第二，**打破即时自复制回路**。
如果 teacher 只是当前 student 的硬拷贝，两侧输出几乎同步，目标容易退化。
EMA teacher 滞后于 student，构成时间上的非对称性。

第三，**在线模型集成**。
作者把 teacher 解释为带指数衰减的 Polyak–Ruppert averaging。
实验中 teacher 在训练全过程持续优于 student。
student 因而追随一个质量更高、变化更慢的目标。

这里 EMA 与 MoCo 的用途不同。
MoCo 的 momentum encoder 主要维持队列中字典特征的一致性。
DINO 没有负样本队列。
其 EMA 更像 Mean Teacher：直接产生稳定的自蒸馏目标。

### 4.6 为什么不能直接复制 student 当 teacher

消融结果非常明确。
使用 student copy 作为 teacher，k-NN top-1 只有 0.1%，即发生坍塌。
使用 previous iteration 也只有 0.1%。
冻结上一 epoch 的 student 作为 teacher 可达到 66.6%。
EMA momentum teacher 达到 72.8%。

因此，关键并非“有两个网络”这一形式。
关键是目标网络必须有足够时间平滑和延迟。
上一 epoch teacher 已能工作，说明 teacher 不必严格是 EMA。
但 EMA 在稳定性和性能上最好。

### 4.7 Student temperature 与 teacher temperature

student temperature 设为：

$$\tau_s=0.1$$

teacher temperature 在前 30 epochs 线性 warm-up：

$$\tau_t: 0.04\rightarrow0.07$$

两者承担不同角色。
student 的分布相对平滑，便于接收多个 teacher 维度上的梯度。
teacher 的较低温度把分布变尖，形成更有区分度的目标。

当 $\tau\rightarrow0$ 时，softmax 趋近 argmax，目标接近 one-hot。
温度过低会使目标过硬、梯度集中。
温度过高则使 teacher 分布接近均匀，失去区分信号。

附录显示，固定 teacher 温度高于约 0.06 时容易坍塌。
但先从 0.04 开始再升到 0.07 可以避免这个问题。
这说明 warm-up 不是装饰性超参数，而是在表征尚未形成时控制目标熵。

### 4.8 Centering：压制“单一维度统治”

teacher softmax 前先减去一个运行中心 $c$。
中心更新为：

$$
c\leftarrow mc+(1-m)\frac{1}{B}\sum_{i=1}^{B}g_{\theta_t}(x_i)
$$

$B$ 是 batch size。
$m$ 是 center momentum。
它只依赖一阶 batch 统计量。
因此 DINO 不要求复杂的 batch-wise assignment，也不依赖同步 BatchNorm 才能工作。

如果某个输出维度在所有样本上都系统性偏大，减去全局均值会消除这类偏置。
这阻止所有样本都落到同一个输出维度。
但 centering 单独使用会把输出推向均匀分布。
所以它只解决一种坍塌，却会鼓励另一种坍塌。

### 4.9 Sharpening：压制“所有维度都一样”

sharpening 通过 teacher 的低温 softmax 实现。
它降低 teacher target entropy。
样本因此必须在若干维度上形成明确偏好。
这与 centering 的作用方向相反。

centering 抑制某个维度对所有样本的全局统治。
sharpening 抑制每个样本都输出同一个均匀分布。
两者结合，允许不同样本形成不同的尖锐分布。

作者用交叉熵分解解释这一点：

$$H(P_t,P_s)=h(P_t)+D_{KL}(P_t\Vert P_s)$$

若 $D_{KL}=0$ 且输出对输入不变，就发生坍塌。
缺少 centering 时，target entropy 趋近 0，表示单维度主导。
缺少 sharpening 时，entropy 趋近 $\log K$，表示均匀输出。
两者同时存在，teacher 目标既不单点全局垄断，也不完全均匀。

### 4.10 防坍塌机制的完整逻辑

DINO 不靠一个“神奇部件”避免坍塌。
其稳定性来自三种非对称性的组合。

- **参数时间非对称**：teacher 是 student 的 EMA。
- **视图非对称**：teacher 只看 global crops，student 看 global 与 local crops。
- **分布非对称**：teacher 经过 centering 和更低温度 sharpening。

stop-gradient 确保 teacher 不被当前 loss 直接拉向 student。
student 只能追随 teacher。
teacher 又缓慢追随 student 的历史平均。
这个闭环如果缺少时间平滑或输出调节，就可能退化为常数映射。

### 4.11 Projection head 与 BN-free 设计

projection head 是 3 层 MLP。
隐藏层宽度为 2048。
其后接 $\ell_2$ normalization bottleneck。
再接 weight-normalized fully connected layer。
默认 bottleneck 维度是 256。
默认输出维度 $K=65536$。

较大 $K$ 能提升性能，但不应把每个维度解释成稳定语义类别。
它更像高容量的软分配空间。

DINO 不使用 student predictor。
student 与 teacher 的网络结构完全相同。
这点与 BYOL 明显不同。

ViT backbone 默认使用 LayerNorm 而非 BatchNorm。
DINO 的 ViT projection head 也不使用 BN。
因此整个 DINO-ViT 系统是 BN-free 的。
这减少了跨 GPU 同步统计的工程开销，也降低了方法对 batch composition 的依赖。

### 4.12 ViT 结构为何适合观察“涌现”

输入图像被切成不重叠的 $N\times N$ patches。
论文主要使用 $N=16$ 或 $N=8$。
每个 patch 经线性投影形成 token。
另加一个可学习 `[CLS]` token。

`[CLS]` 不与人工类别标签绑定。
它只是在 Transformer 中聚合全序列信息。
最后一层 attention 可直接观察 `[CLS]` 如何向 patch tokens 分配权重。
这提供了一个原生的“全局 token → 空间位置”读出接口。

卷积网络也可能含有空间语义。
但卷积特征通常需要额外归因或分解方法才能提取对象区域。
ViT attention 则把全局选择关系显式暴露出来。
因此论文所谓“显式信息”更多指可直接读取，而不是只有 ViT 才可能编码对象。

### 4.13 为什么小 patch 很重要

$16\times16$ patch 在 $224\times224$ 输入上形成 $14\times14=196$ 个 patch tokens。
加上 `[CLS]` 后序列长度为 197。
$8\times8$ patch 形成 $28\times28=784$ 个 patch tokens。
加上 `[CLS]` 后序列长度为 785。

更小 patch 带来更细空间网格。
对象边界不必被粗 patch 强行量化。
这对 dense correspondence、视频分割和 attention mask 尤其重要。

代价来自 self-attention 对 token 数的高复杂度。
ViT-S/16 在论文测量中约 1007 images/s。
ViT-S/8 降到 180 images/s。
ViT-B/16 是 312 images/s。
ViT-B/8 只有 63 images/s。

所以小 patch 没增加 backbone 参数量，却显著增加计算和显存。
它是以 token 数换取空间分辨率，而不是免费的精度增益。

### 4.14 为什么 attention 会涌现出对象

论文没有给出严格的机制证明，但可以从目标与架构共同解释。

第一，跨视图一致性偏好语义稳定因素。
随机颜色、模糊、太阳化与裁剪会破坏局部纹理，但对象身份往往仍在。

第二，local-to-global 迫使局部证据对齐全局语义。
只有能代表对象或辨识性部件的区域，才持续有助于预测 global teacher。

第三，`[CLS]` 必须汇总全局视图。
在没有标签指定“看哪里”的情况下，最稳定策略是聚合跨增强仍一致的前景区域。

第四，小 patch 提供足够细的空间自由度。
attention 可以沿对象边界改变，而不必以粗网格近似。

第五，EMA teacher 提供低噪声目标。
稳定目标减少 student 追逐单批次纹理偶然性的倾向。

第六，sharpened distribution 促使样本作出明确选择。
这种选择压力可能让不同 attention heads 专门关注不同对象或部件。

但“DINO 独有”不是正确结论。
附录显示 MoCo-v2、BYOL、SwAV 等自监督 ViT 也能产生较高的 attention-mask Jaccard。
更稳妥的结论是：对象边界是**自监督 ViT 的共享涌现性质**，而 DINO 让这种性质与优异的 k-NN 表征同时出现。

### 4.15 训练配置

预训练数据是 ImageNet-1k，但训练时不使用标签。
ViT-S/16 的主要配置使用 batch size 1024，分布在 16 张 GPU 上。
优化器是 AdamW。
学习率前 10 epochs 线性 warm-up。
基础学习率按 batch size 线性缩放：

$$lr=0.0005\times\frac{batch\ size}{256}$$

warm-up 后使用 cosine decay。
weight decay 也从 0.04 cosine 增长到 0.4。
数据增强沿用 BYOL 的 color jitter、Gaussian blur、solarization。
位置编码通过 bicubic interpolation 适配不同 crop 尺度。

## 5. 实验设置、主要结果与消融

### 5.1 评测协议：为什么同时看 k-NN 和 linear

linear probe 冻结 backbone，只训练一个有标签线性分类器。
它衡量类别信息能否被线性读出。
但结果对学习率、训练轮数和数据增强较敏感。

k-NN 完全不训练参数分类头。
作者先冻结预训练模型，存储下游训练集特征。
测试样本通过余弦相似度找到近邻并加权投票。
论文通常使用 $k=20$。
投票权重为：

$$\alpha_i=\exp(T_i^Tx/0.07)$$

k-NN 不需要微调、数据增强或多轮下游训练。
因此，k-NN 与 linear 差距小，意味着特征空间本身已有清晰类别邻域。

ViT-S linear evaluation 拼接最后 4 层的 `[CLS]` token。
ViT-B 使用最后一层 `[CLS]`，并拼接平均池化后的 patch tokens。
所以 linear 数字不是简单地在“最后一个 `[CLS]`”上训练同一分类头。
比较结果时必须保留这个协议细节。

### 5.2 ImageNet k-NN 与 linear 主结果

![[assets/DINO - 自监督视觉Transformer的涌现性质/figures/table-2-imagenet-results.png]]

> **图源**：Table 2，PDF p.5，https://arxiv.org/pdf/2104.14294
> **解读**：DINO 在相同 ViT-S 架构上明显超过 BYOL、MoCo-v2 与 SwAV；ViT-S/8 达到 78.3% k-NN，ViT-B/8 达到 80.1% linear。

关键结果如下：

| 模型 | 参数量 | 吞吐（im/s） | Linear top-1 | k-NN top-1 |
|---|---:|---:|---:|---:|
| DINO ResNet-50 | 23M | 1237 | 75.3 | 67.5 |
| DINO ViT-S/16 | 21M | 1007 | 77.0 | 74.5 |
| DINO ViT-S/8 | 21M | 180 | 79.7 | **78.3** |
| DINO ViT-B/16 | 85M | 312 | 78.2 | 76.1 |
| DINO ViT-B/8 | 85M | 63 | **80.1** | 77.4 |

相同 ViT-S 架构下：

| 方法 | Linear | k-NN |
|---|---:|---:|
| BYOL | 71.4 | 66.6 |
| MoCo-v2 | 72.7 | 64.4 |
| SwAV | 73.5 | 66.3 |
| DINO | **77.0** | **74.5** |

DINO 对相同 ViT-S 的 linear 至少提升 3.5 个点。
k-NN 至少提升 7.9 个点。
更重要的是，DINO ViT-S/16 的 k-NN 74.5% 只比 linear 77.0% 低 2.5 点。
这类“k-NN 近似 linear”的性质没有在同表的 ResNet-50 或其他 ViT SSL 实现中同样明显出现。

小 patch 的作用甚至大于从 Small 扩到 Base。
ViT-S/8 的 linear 79.7% 高于 ViT-B/16 的 78.2%。
这说明高空间 token 分辨率对 DINO 表征非常关键。
但 ViT-S/8 吞吐只有 ViT-S/16 的约 18%。
因此实际部署不能只看 top-1。

### 5.3 如何理解 78.3% k-NN

这个数字不是 few-shot 微调，也不是训练一个非线性头。
它只使用冻结的 DINO 特征和 ImageNet 训练标签做邻域投票。
没有数据增强。
没有下游参数学习。

因此 78.3% 表明同类样本在特征空间中自然相邻。
DINO 学到的不是“线性头勉强可分”的表征。
它形成了具有良好局部几何的语义流形。
这正是图像检索、复制检测和快速域迁移所需要的性质。

### 5.4 复制检测结果

Copydays strong 子集包含模糊、插入、打印扫描等强变换。
评测还加入 10K 个 YFCC100M 干扰图像。
DINO 直接用特征余弦相似度完成复制检测。

ViT-B 描述子由 `[CLS]` 与 patch tokens 的 GeM pooling 拼接而成。
得到 1536 维描述子。
特征再用额外 20K 张 YFCC100M 图像学习 whitening。

结果如下：

| 方法 | 架构 | 输入分辨率 | strong mAP |
|---|---|---:|---:|
| Multigrain | ResNet-50 | $224^2$ | 75.1 |
| Multigrain | ResNet-50 | 最长边 800 | 82.5 |
| 监督预训练 | ViT-B/16 | $224^2$ | 76.4 |
| DINO | ViT-B/16 | $224^2$ | 81.7 |
| DINO | ViT-B/8 | $320^2$ | **85.5** |

DINO ViT-B/16 在相同 $224^2$ 输入下比监督 ViT-B/16 高 5.3 mAP。
ViT-B/8 在更高输入分辨率下达到 85.5 mAP。
这与 k-NN 结果一致：DINO 特征保留了强实例级相似性。

需要注意，DINO-B/8 的 85.5 使用 $320^2$ 输入。
它不能被解释为只由训练方法带来的纯粹增益。
whitening 数据和描述子拼接也属于评测管线的一部分。

### 5.5 地标图像检索结果

论文在 Revisited Oxford 和 Revisited Paris 的 Medium/Hard split 上报告 mAP。
ImageNet-DINO ViT-S/16 在 ROx M/H 上为 41.8/13.7。
监督 ViT-S/16 为 33.5/8.9。
在 RPar M/H 上两者分别为 63.1/34.4 与 63.0/37.2。

当 DINO 改用无标签 GLDv2 预训练后：

- ROx Medium：51.5
- ROx Hard：24.3
- RPar Medium：75.3
- RPar Hard：51.6

这说明 SSL 能直接利用领域数据而不要求地标标签。
但它也说明检索效果很依赖预训练数据分布。
“DINO 特征通用”与“领域内无标签预训练有效”应分开理解。

### 5.6 视频对象分割：冻结特征仍保留空间对应

DAVIS-2017 评测采用相邻帧 patch 特征的 nearest-neighbor correspondence。
模型不为视频分割训练专用 head。
不微调 DINO 参数。
输入为 480p。

| 方法 | 架构 | $(J\&F)_m$ | $J_m$ | $F_m$ |
|---|---|---:|---:|---:|
| 监督 ImageNet | ViT-S/8 | 66.0 | 63.9 | 68.1 |
| DINO | ViT-S/16 | 61.8 | 60.2 | 63.4 |
| DINO | ViT-B/16 | 62.3 | 60.7 | 63.9 |
| DINO | ViT-S/8 | 69.9 | 66.6 | 73.1 |
| DINO | ViT-B/8 | **71.4** | **67.9** | **74.9** |

ViT-B 从 /16 换到 /8，$(J\&F)_m$ 提高 9.1 点。
这再次说明 dense task 更依赖 patch 分辨率，而不只是模型容量。

DINO ViT-S/8 比监督 ViT-S/8 高 3.9 点。
表明自监督预训练保留的空间对应关系优于只为图像分类优化的监督特征。

但 DINO 仍低于使用专门视频数据和方法的 STM 81.8。
所以论文证明的是“通用冻结特征具有意外强的密集对应能力”，不是 DINO 已解决视频分割。

### 5.7 无监督对象发现与 attention mask

![[assets/DINO - 自监督视觉Transformer的涌现性质/figures/figure-4-segmentation-comparison.png]]

> **图源**：Figure 4，PDF p.8，https://arxiv.org/pdf/2104.14294
> **解读**：保留 attention 质量 60% 并阈值化后，DINO mask 更连续地覆盖目标；PASCAL VOC12 上 Jaccard 明显高于监督 ViT。

主文的 attention probing 不训练分割模型。
作者从最后一层选择 attention head。
把 `[CLS]` 到 patch 的 attention 按质量阈值化，保留 60% attention mass。
再与 PASCAL VOC12 ground-truth mask 计算 Jaccard。

| 架构 | 随机权重 | 监督训练 | DINO |
|---|---:|---:|---:|
| ViT-S/16 | 22.0 | 27.3 | **45.9** |
| ViT-S/8 | 21.8 | 23.7 | **44.7** |

DINO 比监督预训练高约 18–21 Jaccard 点。
定性图也显示监督 attention 容易落在背景碎片上。
DINO attention 更集中在完整对象。

但有三个限制必须同时记录。
第一，结果使用“最佳 head”，不是固定规则自动选择所有图像的 head。
第二，mask 只是阈值化平滑 attention，边界不等于专用分割模型输出。
第三，/8 的主表 Jaccard 反而略低于 /16，不能简单声称 patch 越小 attention-mask Jaccard 必然越高。

附录进一步给出 ViT-S/16 的自监督方法对比：

- DINO：45.9
- DINO 无 multi-crop：45.1
- MoCo-v2：46.3
- BYOL：47.8
- SwAV：46.8

因此，对象边界涌现并非 DINO 独占。
DINO 的独特点是同时获得强 attention objectness 与极好的 k-NN/linear 表征。

### 5.8 迁移学习结果

DINO 初始化在多数下游数据集上优于监督 ImageNet 初始化。
ViT-S/16 在 CIFAR-100 从 89.5 提升到 90.5。
iNaturalist 2018 从 70.7 提升到 72.0。
iNaturalist 2019 从 76.6 提升到 78.2。
Cars 从 92.1 提升到 93.0。
ImageNet finetune 从 79.9 提升到 81.5。

ViT-B/16 的 ImageNet finetune 从 81.8 提升到 82.8。
Flowers 从 98.4 提升到 98.8。
Cars 从 92.1 提升到 93.0。

并非每项都提升。
ViT-B/16 在 iNaturalist 2018 从 73.2 降到 72.6。
这提醒我们，自监督预训练通常更可迁移，但不是所有数据分布上严格占优。

### 5.9 低标注评测

在冻结 DINO ViT-S/16 特征上训练 logistic regression：

- ImageNet 1% labels：64.5% top-1
- ImageNet 10% labels：72.2% top-1

这没有 finetune backbone。
也没有下游数据增强。
在相似参数规模下，它可与一些会 finetune 的半监督方法竞争。
但不能把它与更大模型或使用知识蒸馏的结果作完全等价比较。

### 5.10 关键组件消融

![[assets/DINO - 自监督视觉Transformer的涌现性质/figures/table-7-component-ablation.png]]

> **图源**：Table 7，PDF p.8，https://arxiv.org/pdf/2104.14294
> **解读**：EMA momentum 是防坍塌底座，multi-crop 与 cross-entropy 分别贡献显著性能；predictor 和 Sinkhorn-Knopp 对 DINO 并非必要。

消融统一使用 ViT-S/16、300 epochs。
默认 DINO 为 momentum + multi-crop + cross-entropy，无 Sinkhorn-Knopp、无 predictor。

| 变体 | k-NN | Linear | 结论 |
|---|---:|---:|---|
| 默认 DINO | **72.8** | **76.1** | 最佳组合 |
| 去掉 momentum | 0.1 | 0.1 | 完全坍塌 |
| 加 Sinkhorn-Knopp | 72.2 | 76.0 | 几乎无额外收益 |
| 去掉 multi-crop | 67.9 | 72.5 | 明显下降 |
| CE 改 MSE | 52.6 | 62.4 | 匹配分布方式很关键 |
| 加 predictor | 71.8 | 75.6 | predictor 不必要 |
| BYOL | 66.6 | 71.4 | ViT 上明显较弱 |
| MoCo-v2 | 62.0 | 71.6 | k-NN 尤其弱 |
| SwAV | 64.7 | 71.8 | multi-crop 本身不够 |

momentum 被移除后，centering 不足以稳定训练。
加入 Sinkhorn-Knopp 可在无 momentum 的 SwAV 路线中工作，但性能较低。
这说明 EMA teacher 同时提高目标质量与训练稳定性。

multi-crop 移除后，k-NN 降 4.9 点，linear 降 3.6 点。
它不是边缘性 augmentation。
它直接塑造了特征的跨尺度邻域几何。

把 CE 换成 MSE 后仍不完全坍塌，但 k-NN 降 20.2 点。
DINO 的 sharpened probability matching 与 BYOL 的 normalized regression 并不等价。

加入 predictor 反而略降。
这支持“EMA + center/sharpen 已提供足够非对称性”的解释。

### 5.11 Teacher 构造消融

teacher 构造方式与 k-NN top-1：

| Teacher | k-NN top-1 |
|---|---:|
| 当前 student copy | 0.1 |
| previous iteration | 0.1 |
| previous epoch | 66.6 |
| EMA momentum | **72.8** |

previous epoch 能工作，证明“滞后的 teacher”本身已很重要。
EMA 更好，是因为它不是某一个旧 checkpoint，而是连续时间集成。
训练曲线中 teacher 始终优于 student。
这为“self-distillation”提供了实证基础，而不只是命名类比。

### 5.12 Centering 与 sharpening 消融

只使用 sharpening 时，模型趋向单维度输出。
只使用 centering 时，模型趋向均匀输出。
两者缺一，teacher 与 student 的 KL 最终趋近 0。
这表示输出不再依赖输入。

center momentum 在 0、0.9、0.99 时都可训练。
过慢的 0.999 更新会坍塌。
说明 center 必须跟上 teacher logits 分布的变化。

teacher temperature 实验显示：

- 0.02：66.7 k-NN
- 0.04：69.6 k-NN
- 0.06：68.7 k-NN
- 0.08：坍塌至 0.1
- 0.04→0.07 warm-up：69.7 k-NN

低温太极端也不是最优。
温度 warm-up 在稳定性和目标丰富度间折中。

### 5.13 Patch size 消融

ViT-S 从 16×16 降到 8×8，再到 5×5，k-NN 持续提高。
模型参数量基本不变。
但 5×5 patch 的吞吐只有 44 images/s。
8×8 仍有 180 images/s。
16×16 是 1007 images/s。

这组消融说明 DINO 对空间粒度的收益真实存在。
也说明论文的最高精度部分依赖很昂贵的 token 序列。
在现代大分辨率输入上，这个成本会进一步放大。

### 5.14 Multi-crop 的精度、时间和显存权衡

在两台 8-GPU 机器上训练 ViT-S/16：

| Crops | 100 ep top-1 / 时间 | 300 ep top-1 / 时间 | 峰值显存/卡 |
|---|---|---|---:|
| $2\times224^2$ | 67.8 / 15.3h | 72.5 / 45.9h | 9.3G |
| + $2\times96^2$ | 71.5 / 17.0h | 74.5 / 51.0h | 10.5G |
| + $6\times96^2$ | 73.8 / 20.3h | 75.9 / 60.9h | 12.9G |
| + $10\times96^2$ | 74.6 / 24.2h | 76.1 / 72.6h | 15.4G |

没有 local crops 时，300 epochs、45.9 小时得到 72.5。
加入 10 个 local crops 时，100 epochs、24.2 小时已达到 74.6。
也就是更高精度且约一半墙钟时间。

这不是说总 FLOPs 一定更少。
它表明 local-to-global 目标提供了更高的优化效率。
代价是峰值显存从 9.3G 增至 15.4G。

### 5.15 小 batch 可行性

100 epochs、无 multi-crop 时：

| Batch size | k-NN top-1 |
|---:|---:|
| 128 | 57.9 |
| 256 | 59.1 |
| 512 | 59.6 |
| 1024 | 59.9 |

batch 从 1024 降到 128 只损失 2.0 点。
论文甚至用 batch size 8 在 50 epochs 达到 35.2%。
这说明 DINO 不像依赖大规模负样本的对比方法那样必须用巨 batch。

但小 batch 结果较低，而且没有为不同 batch 重新充分调 momentum 等超参数。
所以“可训练”不等于“低资源即可复现最佳结果”。

### 5.16 与 BYOL 的区别

DINO 与 BYOL 都使用 online/student 和 EMA target/teacher。
两者都不需要显式负样本。
但核心匹配方式不同。

| 维度 | DINO | BYOL |
|---|---|---|
| 目标 | sharpened softmax distribution | $\ell_2$ normalized representation |
| 损失 | cross-entropy | MSE / cosine-style regression |
| Student predictor | 不需要 | 防坍塌关键组件 |
| 输出调节 | centering + teacher temperature | predictor、归一化与训练非对称 |
| 视图 | global teacher + global/local student | 经典设定主要是两个大视图 |
| 网络结构 | student/teacher 完全同构 | online 多 predictor |

DINO 可看作受 BYOL 启发的 Mean Teacher 自蒸馏。
但把 DINO 简化为“BYOL 换成 ViT”是不准确的。
Table 7 中 DINO 把 CE 换成 MSE 后，linear 从 76.1 降至 62.4。
BYOL 加 multi-crop 在作者实现中也出现后期性能下降。

### 5.17 与 SwAV 的区别

SwAV 通过在线聚类与 prototype assignment 学表征。
它用 Sinkhorn–Knopp 约束 batch 内分配近似均衡。
它也是 multi-crop 的重要来源。

DINO 保留 multi-crop，但不用在线聚类的 balanced assignment。
它用 EMA teacher 生成 soft targets。
只需运行中心 centering，而不是反复 Sinkhorn 归一化。

论文 Table 15 显示：

- momentum + centering：76.1
- momentum + batch-axis softmax：75.8
- momentum + Sinkhorn–Knopp：76.0
- 无 momentum + centering：0.1
- 无 momentum + batch-axis softmax：72.2
- 无 momentum + Sinkhorn–Knopp（SwAV）：71.8

结论不是 Sinkhorn 无效。
结论是有稳定 EMA teacher 后，简单 centering 已足够。
没有 momentum 时，centering 单独无法阻止坍塌，必须使用更强的 batch balancing。

### 5.18 与 MoCo 的区别

MoCo-v2 是对比学习。
它通过 InfoNCE 把正样本拉近、负样本推远。
queue 存储大量负样本。
momentum encoder 保证 queue 中不同时间生成的 key 具有一致编码。

DINO 没有 negatives。
没有 queue。
没有实例辨别的分母。
EMA teacher 的作用是产生稳定目标分布，而非维护负样本字典。

在 ViT-S/16、300 epochs 对比中：

- MoCo-v2：62.0 k-NN / 71.6 linear
- DINO：72.8 k-NN / 76.1 linear

MoCo-v2 的 linear 尚可，但 k-NN 明显落后。
这表明“是否线性可分”和“局部邻域是否自然成簇”是不同性质。
DINO 的优势尤其体现在后者。

### 5.19 实验是否支撑论文主张

对“DINO 能训练强 ViT 表征”的主张，证据充分。
ImageNet、检索、复制检测、视频分割、迁移与 low-shot 形成了多任务证据链。

对“momentum、multi-crop、CE 和小 patch 很重要”的主张，消融也较完整。
各组件的性能变化幅度足够大。

对“self-supervised ViT 会涌现对象边界”的主张，定性图、VOC Jaccard 和 DAVIS correspondence 相互印证。

但对“为什么会涌现”的因果机制，论文主要给直觉而非严格证明。
对象 attention 也出现在其他 SSL 框架中，说明现象不能只归因于 DINO 某个独立部件。

## 6. 局限性与批判性思考

### 6.1 对象发现评测存在选择偏差

主文选择最佳 attention head 生成 mask。
这比无需 oracle 的自动对象发现更宽松。
不同图像是否使用同一 head、如何选择 head，会影响实际可部署性。

attention mass 阈值也是人为设定。
60% 阈值在 VOC 上有效，不保证跨数据集稳定。

### 6.2 Attention 不是分割模型，也不是解释证明

attention 高响应区域与对象重合，只能说明表征含有空间 objectness。
它不证明模型的分类决策完全由这些区域因果驱动。
也不等于语义分割中的多类别像素预测。

平滑 attention mask 对单显著对象友好。
面对多个同类对象、stuff 类别、细结构或遮挡时，边界质量仍有限。

### 6.3 预训练数据仍然高度策展

DINO 不使用 ImageNet 标签，但使用 ImageNet 图像。
ImageNet 本身是对象中心、类别均衡、经过人工收集的数据集。
所以“无标签”不等于“无数据偏置”。

论文结尾提出在随机、未策展图像上训练更大 ViT。
这恰好说明本文尚未验证开放世界数据上的同等涌现。

### 6.4 计算成本并不低

论文代表性设置用 16 张 GPU 训练约 3 天。
最高结果使用 8×8 patches。
其 token 数是 16×16 的约 4 倍，attention 计算和显存大幅增长。

官方仓库给出的 vanilla ViT-S、8 GPU、100 epochs 约需 1.75 天。
结果约为 69.3% k-NN、74.0% linear。
增强设置 16 GPU、300 epochs 约需 2.6 天，约 73.3% k-NN、76.0% linear。
这还不是论文 800-epoch /8 模型的最高配置。

### 6.5 复现环境陈旧且敏感

官方代码基于 Python 3.6、PyTorch 1.7.1、CUDA 11.0、torchvision 0.8.2。
这些版本在现代 GPU 和系统上需要兼容处理。

仓库特别提示 DAVIS 评测在 PyTorch 1.8.1 上无法复现相同结果。
这暴露出最近邻传播、插值或底层算子版本敏感性。

完整复现还需要 ImageNet、DAVIS、Copydays、YFCC100M 干扰/whitening 数据，以及 GLDv2（若复现检索）。
数据准备成本不应被忽略。

### 6.6 超参数耦合强

teacher temperature、warm-up、center momentum、EMA momentum、crop scale 和 crop 数量共同决定稳定性。
某个超参数在 DINO 中有效，不保证直接迁移到 BYOL 或其他 backbone。

例如 multi-crop 对 DINO 有益，却在作者的 BYOL 配置中导致后期退化。
因此这些组件不能被当作独立可叠加插件。

### 6.7 公平比较仍有协议差异

不同方法可能使用不同训练 epochs、patch size、输入分辨率、projection head 和公开权重。
Table 2 的跨架构部分明确不是严格方法对比。

ViT-B/8 的高精度伴随显著低吞吐。
复制检测最高结果还使用更高测试分辨率和 whitening。
只报最高分会掩盖这些成本。

### 6.8 结论的适用边界

DINO 最适合学习全局语义与对象中心表征。
它不天然保证细粒度纹理、几何度量、长尾类别或非对象场景同样优秀。

跨视图不变性也可能删除任务需要的信息。
例如颜色识别、精确姿态和局部纹理任务可能与增强策略冲突。

## 7. 启发与延伸

### 7.1 方法层面的启发

好的 SSL 系统不必依赖显式负样本。
稳定 target network 与合适的目标分布调节可以替代大队列和大 batch。

防坍塌应理解为多机制平衡。
“让分布不均匀”和“让不同样本不都选同一维”是两个不同问题。
centering 与 sharpening 分别处理它们。

multi-crop 的价值不仅是 augmentation 强度。
它构造了 local-to-global 的学习关系。
这类关系后来可自然扩展到 patch-level teacher–student、masked image modeling 与多尺度密集任务。

### 7.2 ViT 表征的启发

ViT 的全局 token 和 patch token 同时存在，为全局分类与密集对应提供统一接口。
小 patch 能提高空间能力，但计算瓶颈会迅速转向 token 数。
后续模型需要更高效 attention、分层结构或分辨率自适应策略。

“attention 可直接读出对象”提示可以用无标签 attention 生成伪 mask。
它可用于弱监督分割、对象提议、视频跟踪或数据清洗。
但实际系统应增加自动 head 选择、multi-head 融合和置信度校准。

### 7.3 评测层面的启发

linear probe 不是表征质量的唯一尺度。
k-NN 能更直接观察局部几何。
检索、复制检测和 dense correspondence 能暴露分类精度看不到的能力。

未来评价通用视觉表征时，应同时报告：

- 参数与吞吐
- k-NN 与 linear
- 少样本迁移
- 实例级检索
- 密集对应
- 对分布偏移的鲁棒性

### 7.4 对后续工作的影响

DINO 建立了“EMA teacher + 自蒸馏分布 + ViT”这一强范式。
后续工作可以把图像级 `[CLS]` 蒸馏扩展到 patch-level 目标。
也可以结合 masked prediction，让模型同时学习跨视图不变性与局部内容恢复。

从研究史角度看，DINO 的重要性不只在 80.1% ImageNet linear。
更重要的是它把视觉 Transformer 从“需要大量标签的分类器”推进为“可从无标签数据形成对象结构的通用表征器”。

## 8. 关键术语表

- **DINO**：以无标签自蒸馏训练视觉表征的方法；teacher 由 student 的 EMA 构造。
- **Cross-view self-distillation**：teacher 与 student 接收同一图像的不同视图，student 预测 teacher 的输出分布。
- **Momentum teacher / EMA teacher**：用 student 历史参数的指数滑动平均更新的目标网络。
- **Multi-crop**：两个大分辨率 global crops 加多个低分辨率 local crops 的多视图训练。
- **Centering**：减去 teacher logits 的运行均值，防止某个输出维度统治所有样本。
- **Sharpening**：用低 teacher temperature 降低目标熵，防止输出变成均匀分布。
- **Collapse**：输出不再依赖输入；可表现为单维度主导或所有维度均匀。
- **Projection head**：预训练时把 backbone 表征映射到自监督目标空间的 MLP，评测时丢弃。
- **k-NN evaluation**：冻结特征后通过最近邻加权投票分类，不训练参数分类头。
- **Linear evaluation**：冻结 backbone，只训练线性分类器。
- **`[CLS]` token**：聚合 token 序列信息的可学习全局 token；DINO 中没有类别标签监督。
- **Patch token**：图像 patch 的嵌入，保留空间网格结构。
- **Sinkhorn–Knopp**：SwAV 用于产生近似均衡 prototype assignment 的迭代归一化。
- **InfoNCE**：MoCo 等对比学习方法使用的正负样本对比损失。
- **Object emergence**：模型未接收分割标签，却在 attention 中形成对象区域与边界。

## 9. 图表索引

1. **Figure 1 — Attention/object emergence**
   - 文件：[[assets/DINO - 自监督视觉Transformer的涌现性质/figures/figure-1-attention-emergence.png]]
   - 位置：PDF p.1
   - URL：https://arxiv.org/pdf/2104.14294
   - 含义：无监督 DINO-ViT 的 `[CLS]` attention 自动覆盖前景对象。

2. **Figure 2 — Teacher–student 自蒸馏方法**
   - 文件：[[assets/DINO - 自监督视觉Transformer的涌现性质/figures/figure-2-dino-teacher-student.png]]
   - 位置：PDF p.2
   - URL：https://arxiv.org/pdf/2104.14294
   - 含义：cross-view distillation、stop-gradient、centering、softmax temperature 与 teacher EMA 的整体关系。

3. **Table 2 — ImageNet k-NN/linear 主要结果**
   - 文件：[[assets/DINO - 自监督视觉Transformer的涌现性质/figures/table-2-imagenet-results.png]]
   - 位置：PDF p.5
   - URL：https://arxiv.org/pdf/2104.14294
   - 含义：DINO 在 ViT 上的主要性能与跨架构比较。

4. **Figure 4 — 监督 ViT 与 DINO 的对象 mask 对比**
   - 文件：[[assets/DINO - 自监督视觉Transformer的涌现性质/figures/figure-4-segmentation-comparison.png]]
   - 位置：PDF p.8
   - URL：https://arxiv.org/pdf/2104.14294
   - 含义：attention 阈值化后的对象区域与 PASCAL VOC12 Jaccard。

5. **Table 7 — 关键组件消融**
   - 文件：[[assets/DINO - 自监督视觉Transformer的涌现性质/figures/table-7-component-ablation.png]]
   - 位置：PDF p.8
   - URL：https://arxiv.org/pdf/2104.14294
   - 含义：momentum、multi-crop、CE、predictor 与 Sinkhorn–Knopp 的作用。

## 10. 最终评价

DINO 的技术形式很简洁，但简洁来自对不稳定因素的精确配平，而不是组件很少。
EMA teacher 解决目标质量与时间平滑。
centering 与 sharpening 解决两种相反的坍塌。
multi-crop 把局部证据绑定到全局语义。
ViT 的 patch/`[CLS]` 结构则把这种语义组织显式暴露为 attention。

论文最经得住时间检验的结论不是某个 ImageNet 数字。
而是：**当训练目标从标签预测改成跨视图自蒸馏后，Vision Transformer 会自然形成兼具全局语义邻域与对象级空间结构的表征。**
