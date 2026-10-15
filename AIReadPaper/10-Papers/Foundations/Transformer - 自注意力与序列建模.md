---
title: "Transformer - 自注意力与序列建模"
aliases:
  - "Attention Is All You Need"
  - "Transformer"
authors:
  - Ashish Vaswani
  - Noam Shazeer
  - Niki Parmar
  - Jakob Uszkoreit
  - Llion Jones
  - Aidan N. Gomez
  - Łukasz Kaiser
  - Illia Polosukhin
year: 2017
venue: NeurIPS 2017
arxiv: "1706.03762"
tags:
  - paper
  - Transformer
  - attention
  - self-attention
  - sequence-modeling
  - machine-translation
status: 精读完成
---

# Transformer：自注意力与序列建模

> **论文**：*Attention Is All You Need*  
> **作者**：Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
> **会议**：NeurIPS 2017（当时名为 NIPS 2017）　|　**arXiv**：[1706.03762](https://arxiv.org/abs/1706.03762)  
> **版本说明**：本笔记核对的是官方 arXiv PDF 当前 v7（PDF 标注 2023-08-02）；核心工作与会议发表年份为 2017。  
> **本地材料**：[[assets/Transformer - 自注意力与序列建模/sources.md|sources.md]]；官方 PDF 位于 `assets/Transformer - 自注意力与序列建模/sources/paper.pdf`。

## 1. 一句话结论与阅读地图

**一句话结论**：Transformer 用全局 self-attention、逐位置前馈网络、位置编码以及残差归一化，完全替代序列对齐的循环与卷积层；它在 WMT 2014 机器翻译上同时改善 BLEU、训练并行度和训练成本，并把任意两位置之间的信息路径缩短到常数级。

标题“Attention Is All You Need”需要精确理解：论文去掉的是 **encoder/decoder 中负责序列混合的 recurrence 与 convolution**，不是只剩一个 attention 算子。完整模型仍包含 embedding、位置编码、两层 FFN、残差连接、LayerNorm、线性输出层和 softmax；而且 decoder 在自回归生成时仍逐 token 解码。论文证明的是“纯 attention 可以成为序列转导架构的主干”，不是“神经网络中的其他组件都不需要”。

阅读可分为四层：

1. **架构层**：六层 encoder、六层 decoder，三种 attention 连接模式；
2. **算子层**：scaled dot-product attention 与 multi-head attention；
3. **序列层**：位置编码、因果 mask、残差与 LayerNorm 共同保证顺序性和可训练性；
4. **证据层**：复杂度比较、WMT 翻译结果、模型变体实验和 constituency parsing 泛化实验。

## 2. 研究背景、问题定义与核心贡献

### 2.1 2017 年的主流范式与瓶颈

当时的神经机器翻译通常采用 encoder–decoder：encoder 把输入序列变为连续表示，decoder 自回归地产生输出。主干多为 LSTM/GRU 等 RNN，较新的 ByteNet、ConvS2S 则用卷积提高训练并行性；attention 通常作为 RNN/CNN 旁边的对齐模块，而非唯一的序列混合机制。

RNN 的关键限制是隐藏状态递推：

$$
h_t=f(h_{t-1},x_t).
$$

同一条样本中的第 $t$ 步依赖第 $t-1$ 步，因此训练不能沿序列维完全并行；两个相距 $O(n)$ 的位置之间，前向信息和反向梯度都要穿过 $O(n)$ 条计算路径。卷积能并行，但普通局部卷积要堆叠多层才能连接远距离位置：连续卷积的最长路径可达 $O(n/k)$，空洞卷积也需 $O(\log_k n)$。

论文提出的核心问题是：**能否直接让任意两个序列位置在一层中交互，从而同时得到全局依赖、训练并行和较短的信号路径，而不再依赖循环或卷积？**

### 2.2 核心贡献

1. **提出纯 attention 的序列转导架构 Transformer**：encoder 和 decoder 的序列混合完全由 multi-head attention 完成，不使用序列对齐的 RNN/CNN。
2. **系统化 scaled dot-product 与 multi-head attention**：用 $1/\sqrt{d_k}$ 稳定大维度点积，再以多个低维投影头并行捕获不同位置和表示子空间的关系。
3. **用位置编码补回顺序信息**：将不同频率的正弦/余弦编码直接加到 token embedding，使无循环、无卷积的模型仍能感知位置。
4. **从复杂度、并行度和路径长度解释 self-attention 的优势**：当机器翻译常见的 $n<d$ 时，每层 self-attention 的 $O(n^2d)$ 可低于 recurrent layer 的 $O(nd^2)$，且最少串行操作数和最长路径均为 $O(1)$。
5. **给出当时有竞争力的实证结果**：WMT 2014 EN–DE 达 28.4 BLEU，表中 EN–FR 达 41.8 BLEU；并在 constituency parsing 上展示跨任务泛化。

## 3. 整体架构：信息如何流动

### 3.1 Encoder–decoder 全景

![](assets/Transformer%20-%20自注意力与序列建模/figures/fig01_transformer_architecture.png)

*图 1：原文 Figure 1。左侧 encoder 把整段输入编码为序列表示，右侧 decoder 在 masked self-attention、cross-attention 和 FFN 中逐层生成输出分布。*

原文的 base Transformer 使用 $N=6$ 层 encoder 和 $N=6$ 层 decoder，所有子层输出维度统一为

$$
d_{\text{model}}=512,
$$

以便残差相加。

**每个 encoder layer 有两个子层：**

1. multi-head self-attention；
2. position-wise feed-forward network（FFN）。

**每个 decoder layer 有三个子层：**

1. masked multi-head self-attention；
2. 对 encoder 输出做 multi-head cross-attention；
3. position-wise FFN。

在 decoder 的 cross-attention 中，query 来自上一 decoder 子层，key/value 来自 encoder 最顶层输出；因此每个目标位置都能按内容读取全部源序列位置。

### 3.2 三种 attention 的 $Q,K,V$ 来源

| 场景 | Query | Key / Value | 可见范围 | 作用 |
|---|---|---|---|---|
| Encoder self-attention | encoder 上一层 | encoder 上一层 | 全部输入位置 | 建模输入内部关系 |
| Decoder masked self-attention | decoder 上一层 | decoder 上一层 | 当前及此前目标位置 | 保持自回归因果性 |
| Encoder–decoder attention | decoder 中间表示 | encoder 最终表示 | 全部源位置 | 做源–目标对齐与条件读取 |

Self-attention 的“self”不是说 $Q=K=V$ 数值相同，而是说三者来自同一组输入表示；它们仍分别乘以不同的可学习投影矩阵。

### 3.3 Position-wise FFN

每层 attention 后还有一个对每个位置独立、参数共享的两层 MLP：

$$
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2.
$$

base 模型使用

$$
512\rightarrow 2048\rightarrow 512.
$$

Attention 在 **token 维**混合信息，FFN 在 **通道/特征维**做非线性变换；二者角色互补。FFN 可等价看作核大小为 1 的两层卷积，但不同 Transformer layer 的 FFN 参数不共享。

### 3.4 Embedding、权重共享与输出

论文共享源 embedding、目标 embedding 与输出前 softmax 线性层的权重矩阵，并在 embedding 端乘以 $\sqrt{d_{\text{model}}}$。直观上，embedding 乘此尺度后与幅值约为常数量级的位置编码相加时更平衡；输出端则用共享词向量作为分类器权重。

Decoder 顶部通过线性映射和 softmax 得到下一个 token 的概率。训练时目标序列整体右移作为 decoder 输入；推理时仍需把已生成 token 逐步反馈给 decoder。

## 4. Scaled Dot-Product Attention：公式、尺度与 mask

### 4.1 从 query–key 匹配到 value 聚合

![](assets/Transformer%20-%20自注意力与序列建模/figures/fig02_attention_modules.png)

*图 2：原文 Figure 2。左侧先计算 query–key 相似度、缩放、可选 mask 与 softmax，再加权 value；右侧把这一过程放入多个投影子空间并行执行。*

设一批 query、key、value 分别组成

$$
Q\in\mathbb R^{n_q\times d_k},\quad
K\in\mathbb R^{n_k\times d_k},\quad
V\in\mathbb R^{n_k\times d_v}.
$$

Scaled dot-product attention 为

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

逐步看：

1. $QK^\top\in\mathbb R^{n_q\times n_k}$ 生成所有 query–key 对的兼容性分数；
2. 沿每个 query 对应的 key 维做 softmax，把分数变成和为 1 的权重；
3. 权重矩阵乘 $V$，得到每个 query 对所有 value 的加权和。

它不是固定的卷积核：权重由当前输入内容动态产生。它也不把历史压入一个固定隐藏状态，而是允许每个位置直接读取所有可见位置。

### 4.2 为什么除以 $\sqrt{d_k}$

原文给出一个方差论证。若 $q_i,k_i$ 独立、均值为 0、方差为 1，则

$$
q\cdot k=\sum_{i=1}^{d_k}q_i k_i,
$$

其方差约为

$$
\operatorname{Var}(q\cdot k)=d_k.
$$

因此点积的典型幅度随 $\sqrt{d_k}$ 增长。若不缩放，大幅 logits 会把 softmax 推到接近 one-hot 的饱和区，非最大项的梯度变得很小。除以 $\sqrt{d_k}$ 后，logit 方差回到 $O(1)$，训练更稳定。

这一区别于 additive attention：两者理论复杂度相近，但 dot-product attention 可直接调用高效矩阵乘；论文观察到在较大 $d_k$ 下，未缩放点积不如 additive attention，而缩放用于修正这一问题。

### 4.3 Causal mask 的精确定义

Decoder 的 masked self-attention 不能读取未来目标 token。可在 softmax 前加入矩阵 $M$：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),
$$

其中

$$
M_{ij}=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

于是 softmax 后 $A_{ij}=0$（当 $j>i$），第 $i$ 个目标位置只能依赖当前位置及更早位置。再配合目标 embedding 右移一位，预测位置 $i$ 时只能使用真实的 $<i$ 前缀。

需要区分两个概念：

- **因果 mask**：原文明确讨论，用于禁止 decoder 看未来；
- **padding mask**：工程实现中常用于屏蔽补齐 token，但它不是论文该段的主要创新点，不能与 causal mask 混为一谈。

Mask 解决信息泄漏，不解决推理并行：训练时因所有目标 token 已知，可一次矩阵计算全部位置；自回归生成时下一个 token 未知，仍然存在输出序列方向的串行依赖。

## 5. Multi-Head Attention：多个子空间而非简单重复

### 5.1 定义与张量形状

论文定义

$$
\operatorname{MultiHead}(Q,K,V)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O,
$$

其中

$$
\operatorname{head}_i=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V).
$$

投影矩阵为

$$
W_i^Q\in\mathbb R^{d_{\text{model}}\times d_k},\quad
W_i^K\in\mathbb R^{d_{\text{model}}\times d_k},\quad
W_i^V\in\mathbb R^{d_{\text{model}}\times d_v},
$$

$$
W^O\in\mathbb R^{hd_v\times d_{\text{model}}}.
$$

base 模型取

$$
h=8,\qquad d_k=d_v=d_{\text{model}}/h=64.
$$

拼接后维度 $hd_v=512$，再经 $W^O$ 混合各头并回到 residual stream 的 512 维。

### 5.2 为什么总成本没有乘八

若用一个 full-width head，attention score 的主要成本约为 $O(n^2d_{\text{model}})$。使用 $h$ 个头、每头宽度 $d_k=d_v=d_{\text{model}}/h$ 时，所有头的 score 计算总成本仍约为

$$
h\cdot O\left(n^2\frac{d_{\text{model}}}{h}\right)
=O(n^2d_{\text{model}}).
$$

代价没有凭空消失：Q/K/V 与输出投影还包含 $O(nd_{\text{model}}^2)$ 的线性层成本，注意力矩阵本身需要与 $n^2$ 成正比的中间存储。原文 Table 1 聚焦层级主项和序列交互，不等于完整 kernel 的逐项成本模型。

### 5.3 多头的作用与证据边界

单头 attention 会把多个关系压进同一个加权平均；多头允许模型在不同投影子空间、不同位置模式上并行读取。例如某些头可能偏向局部邻接，另一些头连接长距离依赖。

但“每个 head 都对应清晰语言学功能”不是严格定理。原文附录给出若干可视化例子，属于支持性定性证据；它说明分工 **可能** 出现，不证明所有头都可解释、都必要或都稳定复现。

## 6. 位置编码：在无递归架构中注入顺序

### 6.1 为什么必须显式加入位置

若没有 positional encoding，self-attention 对 token 集合的排列是等变的：交换输入位置，输出也只做对应交换；模型无法仅凭 attention 区分“狗咬人”和“人咬狗”的次序。论文在 encoder 和 decoder 底部都做

$$
X_0=\operatorname{Embedding}(tokens)+PE.
$$

$PE$ 与 embedding 都是 $d_{\text{model}}$ 维，因此可直接相加。

### 6.2 正弦/余弦公式

对位置 $pos$ 和通道索引 $i$：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),
$$

$$
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).
$$

各二维 sin/cos 对的波长构成从 $2\pi$ 到 $10000\cdot2\pi$ 的几何级数：高频通道分辨局部位移，低频通道携带更长尺度的位置变化。

### 6.3 为什么固定偏移可由线性变换表示

对任意频率 $\omega$ 和固定偏移 $k$：

$$
\begin{bmatrix}
\sin(\omega(pos+k))\\
\cos(\omega(pos+k))
\end{bmatrix}
=
\begin{bmatrix}
\cos(\omega k)&\sin(\omega k)\\
-\sin(\omega k)&\cos(\omega k)
\end{bmatrix}
\begin{bmatrix}
\sin(\omega pos)\\
\cos(\omega pos)
\end{bmatrix}.
$$

因此固定相对位移对应每个频率块中的旋转线性变换。作者据此 **假设** 模型较容易学习按相对位置注意；这是一种设计动机，不是论文通过专门实验严格证明的机制。

原文还比较了 learned positional embedding：Table 3(E) 的 dev BLEU 为 25.7，几乎等于正弦版本 base 的 25.8。作者最终选择正弦编码，是因为它 **可能** 外推到训练未见的更长长度；原论文并没有给出系统长上下文外推实验来证实这一点。

## 7. 残差连接、LayerNorm 与训练稳定性

每个子层采用

$$
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).
$$

执行顺序是：Sublayer → dropout → 与 $x$ 残差相加 → LayerNorm。这是今天通常所称的 **Post-LN Transformer**。

- **残差连接**保留 identity path，让每层只需学习增量变换，并改善深层网络的梯度传播；
- **LayerNorm**在单个 token 的特征维归一化，不依赖 batch 统计，适合可变长度序列；
- **统一的 $d_{\text{model}}$**保证每个子层都可直接与 residual stream 相加；
- **dropout**施加在子层输出进入残差相加之前，也施加于 token embedding 与位置编码之和。

必须避免时代错置：后续大模型常见的 Pre-LN、RMSNorm、额外 residual scaling 等并不是这篇论文提出的配置。原始架构明确是 Post-LN。

## 8. 为什么 Self-Attention：复杂度、并行度与路径长度

![](assets/Transformer%20-%20自注意力与序列建模/figures/table01_complexity.png)

*表 1：原文 Table 1。Self-attention 用二次序列复杂度换取常数级串行操作数和任意位置间的一跳连接。*

设序列长度为 $n$，表示维度为 $d$，卷积核宽为 $k$，局部 attention 邻域为 $r$：

| 层类型 | 每层复杂度 | 最少串行操作数 | 最长路径 |
|---|---:|---:|---:|
| Self-attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted self-attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

### 8.1 三个评价维度不是一回事

1. **算术复杂度**：当 $n<d$ 时，$n^2d<nd^2$，全局 self-attention 比 dense recurrent layer 更有利；论文指出机器翻译中的 subword sentence 通常处于该区间。
2. **训练并行度**：self-attention 和 convolution 对所有位置同时计算；RNN 必须执行 $n$ 个递推步骤。
3. **依赖路径长度**：self-attention 中任意两位置一层直连；这让长距离信号无需穿过很多中间状态。

“常数路径”不意味着总计算为常数，也不意味着无限长序列高效。全局 attention 的 $n^2$ score 矩阵会在长序列上成为时间和显存瓶颈；原文已经把 restricted attention 作为未来方向，并明确指出其最长路径会增加到 $O(n/r)$。

### 8.2 论文比较的边界

Table 1 是渐近层级比较，不包含硬件利用率、内存带宽、投影层常数、batch 形状等工程因素。论文 2017 年的结论针对当时常见机器翻译长度尤其合理；不能直接据此断言在任意 $n,d$、任意硬件下 self-attention 都比 RNN/CNN 快。

## 9. 训练配置与可复现细节

### 9.1 数据与分词

- **WMT 2014 English–German**：约 450 万句对，共享 source–target BPE 词表约 37,000；
- **WMT 2014 English–French**：约 3,600 万句对，32,000 word-piece 词表；
- 按近似句长分桶；每 batch 约含 25,000 source tokens 和 25,000 target tokens。

### 9.2 Base 与 Big 配置

| 配置 | $N$ | $d_{\text{model}}$ | $d_{ff}$ | $h$ | $d_k,d_v$ | Dropout | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 6 | 512 | 2048 | 8 | 64, 64 | 0.1 | 65M |
| big | 6 | 1024 | 4096 | 16 | 64, 64（表中未另列） | 0.3 | 213M |

English–French big 模型例外地使用 $P_{drop}=0.1$，而非 0.3。

### 9.3 硬件、步数与优化器

训练使用一台 8×NVIDIA P100：

- base：约 0.4 秒/step，100,000 steps，约 12 小时；
- big：约 1.0 秒/step，300,000 steps，约 3.5 天。

Adam 超参数为

$$
\beta_1=0.9,\qquad \beta_2=0.98,\qquad \epsilon=10^{-9}.
$$

学习率日程为

$$
\operatorname{lrate}
=d_{\text{model}}^{-1/2}
\min\left(step^{-1/2},\;step\cdot warmup^{-3/2}\right),
$$

其中 $warmup=4000$。前 4,000 步线性升温，之后按 $step^{-1/2}$ 衰减；$d_{\text{model}}^{-1/2}$ 使不同模型宽度的步长尺度更可比。

### 9.4 正则化与推理

- residual dropout：base 为 0.1；
- embedding 与 positional encoding 之和也做 dropout；
- label smoothing：$\epsilon_{ls}=0.1$。它让模型不那么过度确信，因此 perplexity 可能变差，但 accuracy/BLEU 改善；
- base 结果平均最后 5 个 checkpoint，big 平均最后 20 个 checkpoint；
- beam size 4，length penalty $\alpha=0.6$；
- 最大输出长度设为输入长度 + 50，并允许提前结束。

这些解码与 checkpoint averaging 细节会影响 BLEU，复现时不能只复制网络结构。

## 10. 翻译实验：质量、成本与结果一致性

![](assets/Transformer%20-%20自注意力与序列建模/figures/table02_translation_results.png)

*表 2：原文 Table 2。Transformer big 在 EN–DE 达 28.4 BLEU、表中 EN–FR 达 41.8 BLEU；base EN–DE 的估算训练 FLOPs 明显低于列出的强基线。*

### 10.1 主要结果

- **WMT 2014 EN–DE newstest2014**：base 27.3 BLEU，big 28.4 BLEU；big 比表中此前最佳 ensemble 26.36 高约 2 BLEU。
- **WMT 2014 EN–FR newstest2014**：base 38.1 BLEU，Table 2 与摘要给 big 41.8 BLEU。
- **训练成本**：Table 2 估计 base EN–DE 为 $3.3\times10^{18}$ FLOPs，big 为 $2.3\times10^{19}$ FLOPs；作者据此强调质量与成本的共同优势。

训练 FLOPs 是以训练时间、GPU 数量和估计的单卡持续单精度吞吐相乘得到，不是逐算子精确计数。跨论文硬件利用率和数据管线不同，因此它更适合作为数量级比较，而不是严格同平台 benchmark。

### 10.2 一个应保留的原文数字矛盾

官方 PDF 的摘要和 Table 2 给 English–French big **41.8 BLEU**，但 Section 6.1 正文写的是 **41.0 BLEU**。本笔记不擅自替作者统一数字：引用主结果时按摘要/Table 2 记 41.8，同时明确保留正文冲突。精读的原则应是呈现来源差异，而不是选择更漂亮的数字后忽略另一处。

### 10.3 实验是否支撑论文主张

对“无需 recurrence/convolution 也能做好序列转导”这一主张，证据很强：模型在两个大规模翻译任务上达到当时 SOTA，同时训练周期短。对“普遍适用于所有序列任务和长序列”的外推则证据不足：主体实验仍集中于机器翻译，序列长度也远未达到今天的长上下文场景。

## 11. 模型变体与消融：哪些设计真正有证据

![](assets/Transformer%20-%20自注意力与序列建模/figures/table03_ablation.png)

*表 3：原文 Table 3。它不是把模块逐个移除的严格 ablation，而是在 EN–DE dev 上系统改变头数、key 维度、深宽度、正则化和位置表示。*

### 11.1 头数：不是越多越好

在近似固定计算量下：

| 头数 $h$ | 每头 $d_k=d_v$ | PPL | dev BLEU |
|---:|---:|---:|---:|
| 1 | 512 | 5.29 | 24.9 |
| 4 | 128 | 5.00 | 25.5 |
| 8（base） | 64 | 4.92 | 25.8 |
| 16 | 32 | 4.91 | 25.8 |
| 32 | 16 | 5.01 | 25.4 |

单头比最佳设置低 0.9 BLEU，支持多头有益；但 32 个过窄的头也下降，说明头数与每头容量存在折中。该表不支持“头越多表达力越强”的单调结论。

### 11.2 Key 维度：匹配函数需要容量

Table 3(B) 把 $d_k$ 降到 16 或 32，dev BLEU 分别为 25.1、25.4，低于 base 的 25.8。作者据此认为 query–key compatibility 并不简单，过小的 key 子空间会伤害质量，甚至提出更复杂兼容函数可能有益。

### 11.3 深度、宽度与 FFN 宽度

- $N=2/4/8$ 层：BLEU 为 23.7/25.3/25.5；太浅明显不足，继续加深收益变小且参数增加；
- $d_{\text{model}}=256/1024$：BLEU 为 24.5/26.0；更宽模型更好但参数从 28M 增至 168M；
- $d_{ff}=1024/4096$：BLEU 为 25.4/26.2；FFN 容量是重要性能来源，Transformer 并非只靠 attention；
- big 配置 dev BLEU 26.4、PPL 4.33、参数 213M。

### 11.4 Dropout 与 label smoothing

- residual dropout 从 0.1 改为 0.0，BLEU 从 25.8 降到 24.6，显示明显过拟合；改为 0.2 得 25.5；
- label smoothing 从 0.1 改为 0.0，PPL 反而改善到 4.67，但 BLEU 降到 25.3；改为 0.2 时 PPL 变差到 5.47、BLEU 为 25.7。

这清楚展示“更好 likelihood/perplexity 不必然带来更好序列级 BLEU”，也说明训练目标的校准方式会影响生成质量。

### 11.5 正弦与 learned position

Learned positional embedding 的 PPL 4.92、BLEU 25.7，几乎等于 base 正弦版本的 4.92、25.8。该实验支持“二者在训练/开发长度内效果接近”，不支持“正弦位置编码在所有设置中更优”；作者偏好正弦编码主要基于潜在长度外推。

## 12. 注意力可视化与解析实验

![](assets/Transformer%20-%20自注意力与序列建模/figures/fig03_attention_visualization.png)

*图 3：原文 Figure 3。第 5 层 encoder self-attention 的多个头从 making 连向远处的 more/difficult，展示一层中直接连接长距离短语依赖的能力。*

附录还展示代词指代和句法结构相关的 head。它们与论文的常数路径论点相呼应：远距离 token 可以直接交换信息，不必沿 RNN 状态逐步传递。

但 attention weight 不应自动等同于完整因果解释：输出还受 value、其他 head、输出投影、残差路径、FFN 和后续层共同影响。原文使用了谨慎措辞（如 “apparently involved” 和 “seems related”）；把这些图说成“Transformer 已证明学会句法”会超过原始证据。

### 12.1 Constituency parsing 泛化

论文另在 Penn Treebank WSJ 上训练 4 层、$d_{\text{model}}=1024$ 的 Transformer：

- 仅约 40K WSJ 训练句：WSJ Section 23 F1 = 91.3；
- 约 17M 句半监督语料：F1 = 92.7。

该任务输出有强结构约束，且输出通常比输入长。结果说明架构并非只适用于翻译，但作者只做了少量超参搜索，且 92.7 仍低于表中的 multi-task 93.0 和 generative 93.3；因此“泛化良好”合理，“全面优于任务专用解析器”则不成立。

## 13. 局限性与批判性思考

### 13.1 原论文明确或隐含承认的局限

1. **全局 attention 对长序列是二次复杂度**：作者提出未来研究 local/restricted attention，用于图像、音频、视频等大输入输出。
2. **自回归生成仍然串行**：结论把“让 generation 更少串行”列为未来目标；训练并行不等于解码并行。
3. **位置外推只是设计假设**：正弦编码可能支持超过训练长度的外推，但论文未系统验证。
4. **attention 会平均多个位置、降低有效分辨率**：作者认为 multi-head 可缓解，但并未给出完整理论保证。

### 13.2 从今天回看仍应注意的问题

1. **任务覆盖窄**：核心证据来自 EN–DE/EN–FR 翻译，外加一个 parsing 任务；不能从该论文单独推出语言建模、视觉、语音或多模态上的普适结论。
2. **长上下文未验证**：Table 1 已揭示 $O(n^2d)$，但实验没有覆盖超长序列；“任意两点一跳”与“可扩展到任意长度”是两回事。
3. **消融不完全**：Table 3 很有价值，但没有严格逐项移除 residual、LayerNorm、cross-attention、FFN 等；很多组件的必要性更多来自架构常识和整体结果，而非本论文独立实验。
4. **统计报告有限**：未报告多随机种子均值/方差，许多小幅 BLEU 差异不宜过度解释。
5. **跨论文成本比较近似**：FLOPs 来自硬件持续吞吐估计；数据处理、优化程度、预处理与评估协议都可能影响公平性。
6. **可视化是案例而非机制证明**：个别 head 的清晰模式说明模型能形成这类关系，但不证明这些权重是预测的唯一原因。
7. **Post-LN 深度扩展性**：原始 6 层配置训练成功，但后来的更深模型常改用 Pre-LN 等方案。这个结论来自后续实践，不应倒写成原论文发现。

### 13.3 适用边界

Transformer 的关键优势在于：序列长度适中、并行硬件充足、需要全局内容寻址。若 $n$ 极长、内存严格受限、必须流式固定状态推理，标准全局 attention 未必是最佳选择。反之，在需要任意位置直接交互且训练吞吐重要的场景，它提供了非常通用的基础算子。

## 14. 与后续常用变体的关系：哪些是原论文，哪些不是

以下关系用于建立技术谱系，**不表示 2017 年论文已提出或验证这些后世方案**。

### 14.1 Encoder-only、decoder-only 与 encoder–decoder

- 原论文是完整 **encoder–decoder Transformer**；
- 后续 BERT 类模型主要保留双向 encoder，用于表示学习；
- GPT 类模型主要保留 causal decoder，用于自回归语言建模；
- T5 等继续采用 encoder–decoder，但训练目标与位置偏置等已改变。

三者共享 attention/FFN/residual 的骨架，但任务条件和 mask 不同，不能把 decoder-only LLM 的全部经验归功于原论文实验。

### 14.2 归一化变体

- 原文：Post-LN，$\operatorname{LN}(x+F(x))$；
- 后续常见：Pre-LN，$x+F(\operatorname{LN}(x))$，以及 RMSNorm 等。

它们旨在改善更深网络训练或简化归一化，但不是原始 Transformer 公式。

### 14.3 位置表示变体

- 原文：绝对正弦/余弦位置编码，加到 embedding；
- 后续：learned absolute position、relative position bias、RoPE、ALiBi 等。

这些方案直接回应相对位置建模、长度外推或实现效率；其有效性需要各自论文证据，不能说原文已经提出 RoPE/relative bias。

### 14.4 长序列与高效 attention

原文已经指出 restricted attention 可把复杂度降为 $O(rnd)$，但后来才出现大量 sparse/local、linear/kernel、low-rank 等架构，以及通过精确重排计算降低 IO 开销的 FlashAttention。需要区分：

- **架构近似/稀疏化**改变可见连接或 attention 计算形式；
- **FlashAttention 类 kernel**通常保持标准 softmax attention 的数学结果，主要改善显存访问与中间存储。

### 14.5 Head 与 KV 共享

原文使用标准 multi-head attention，每头都有独立 Q/K/V 投影。后续 MQA/GQA 让多个 query heads 共享较少的 K/V heads，以降低自回归推理的 KV cache 与带宽；这是服务大模型解码的后续工程/架构折中，不属于原始设计。

## 15. 可复用的理解框架与工程启发

1. **把“序列建模”拆成 token mixing 与 channel mixing**：attention 负责跨位置读取，FFN 负责逐位置变换；很多后续架构仍沿用这一分工。
2. **计算图的可并行性与 FLOPs 同等重要**：RNN 渐近算术量未必夸张，但串行依赖限制硬件利用；Transformer 的成功同时是表示设计和并行计算设计。
3. **短路径是一种归纳偏置**：任意位置一跳可达利于长距离依赖，但也弱化了局部性，需要位置编码、多头或局部结构补充。
4. **缩放因子来自统计尺度**：$1/\sqrt{d_k}$ 不是经验魔法，而是控制点积方差与 softmax 饱和。
5. **模型质量来自组合而非单算子**：消融显示 FFN 宽度、dropout、label smoothing、头维度都重要；“只有 attention”是标题表达，不是实现事实。
6. **报告结果时保留来源冲突**：EN–FR 的 41.8/41.0 不一致提醒我们，经典论文也可能有版本或正文数字差异，复现应记录具体表格、版本和评测设置。

## 16. 关键术语表

| 术语 | 含义 |
|---|---|
| Self-attention | $Q,K,V$ 来自同一序列表示的 attention，用内容动态混合位置 |
| Cross-attention | Query 来自 decoder，Key/Value 来自 encoder 的条件读取 |
| Scaled dot-product | 对 $QK^\top$ 除以 $\sqrt{d_k}$ 后做 softmax 的 attention |
| Multi-head attention | 在多个低维投影子空间并行 attention，再拼接输出 |
| Causal mask | 将未来位置 logit 置为 $-\infty$，保证自回归因果性 |
| Position-wise FFN | 对每个 token 独立应用、跨位置共享参数的两层 MLP |
| Positional encoding | 注入 token 顺序的向量；原文采用固定 sin/cos |
| Post-LN | 先子层与残差相加，再做 LayerNorm 的原始配置 |
| Label smoothing | 用少量均匀概率平滑 one-hot 目标，改善校准与 BLEU |
| Maximum path length | 任意两位置间信息/梯度必须穿过的最长计算路径 |

## 17. 图表索引与核验结论

| 本地文件 | 原文位置 | 用途 | 人工核验 |
|---|---|---|---|
| `figures/fig01_transformer_architecture.png` | Figure 1，PDF p.3 | 完整 encoder–decoder 架构 | 完整、清晰、含原图注 |
| `figures/fig02_attention_modules.png` | Figure 2，PDF p.4 | scaled dot-product 与 multi-head 流程 | 完整、清晰、含原图注 |
| `figures/table01_complexity.png` | Table 1，PDF p.6 | 层复杂度、串行操作与路径长度 | 四行数据与表头完整可读 |
| `figures/table02_translation_results.png` | Table 2，PDF p.8 | WMT BLEU 与估算训练成本 | 表头、基线与 Transformer 行完整可读 |
| `figures/table03_ablation.png` | Table 3，PDF p.9 | 架构变体与正则化实验 | A–E、base、big 各组完整可读 |
| `figures/fig03_attention_visualization.png` | Figure 3，PDF p.13 | 长距离 attention 定性案例 | 彩色连线、token 与图注完整可读 |

所有图表均从官方 `paper.pdf` 以 300 DPI 按原页坐标裁剪；未使用 PDF 内嵌的碎片位图冒充完整原图。完整来源、文件校验值与裁剪坐标见 [[assets/Transformer - 自注意力与序列建模/sources.md|sources.md]]。
