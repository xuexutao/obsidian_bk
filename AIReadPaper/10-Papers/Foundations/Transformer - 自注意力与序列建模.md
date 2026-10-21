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
> **会议**：NeurIPS 2017（当时名为 NIPS 2017）  
> **arXiv**：[1706.03762](https://arxiv.org/abs/1706.03762)  
> **核对版本**：官方 arXiv PDF v7，文件标注日期为 2023-08-02；论文核心工作发表于 2017 年。  
> **本地材料**：[[assets/Transformer - 自注意力与序列建模/sources.md|sources.md]]；官方 PDF：`assets/Transformer - 自注意力与序列建模/sources/paper.pdf`。

## 1. 一句话结论与阅读地图

**一句话结论**：Transformer 把序列中跨位置的信息混合交给全局 self-attention，用 multi-head 投影、位置编码、逐位置 FFN、残差连接与 LayerNorm 组成完整的 encoder–decoder，在不使用序列对齐 RNN/CNN 的前提下，提高了训练并行度、缩短了长距离依赖路径，并在 WMT 2014 机器翻译上取得当时的最佳结果。

标题“Attention Is All You Need”不能按字面理解成“模型里只有 attention”。原始 Transformer 仍然包含 embedding、正弦位置编码、两层 FFN、残差连接、LayerNorm、线性输出层、softmax、dropout 与自回归解码。论文真正证明的是：**负责序列位置之间交互的主干可以完全由 attention 构成，不必再依赖 recurrence 或 convolution。**

建议按四个层次阅读：

1. **架构层**：六层 encoder、六层 decoder，以及三种 attention 连接方式；
2. **算子层**：scaled dot-product attention 与 multi-head attention；
3. **序列层**：位置编码、causal mask、Post-LN 残差块如何共同维持顺序性与因果性；
4. **证据层**：复杂度比较、训练配置、翻译结果、模型变体与 parsing 泛化实验。

## 2. 研究背景、问题定义与核心贡献

### 2.1 2017 年序列转导的主流范式

当时的神经机器翻译通常采用 encoder–decoder。Encoder 把输入序列编码成连续表示，decoder 再自回归地产生输出。主干多为 LSTM、GRU 等 RNN；ByteNet、ConvS2S 等工作则尝试用卷积改善并行性。Attention 已经广泛用于源端与目标端对齐，但通常只是 RNN/CNN 主干旁边的辅助模块。

RNN 的隐藏状态满足递推关系：

$$h_t=f(h_{t-1},x_t).$$

同一条样本中，第 $t$ 步依赖第 $t-1$ 步，因此序列维无法在训练时完全并行。相距较远的两个位置之间，信号与梯度还要穿过较长的状态链。卷积可以并行计算全部位置，但局部卷积要通过多层堆叠才能连接远距离位置；连续卷积的路径长度随距离线性增长，空洞卷积也通常需要对数深度。

### 2.2 论文要回答的问题

论文提出的核心问题是：**能否让任意两个序列位置在一层中直接交互，同时保留 encoder–decoder 的条件生成能力，并彻底移除序列对齐的循环和卷积？**

这要求模型同时解决四件事：

1. 用内容决定“当前 token 应读取哪些位置”；
2. 在没有递归与卷积时显式注入顺序；
3. 在 decoder 中阻止未来信息泄漏；
4. 保持足够的非线性表示能力与训练稳定性。

### 2.3 核心贡献

1. **提出纯 attention 的序列转导架构 Transformer**：encoder 和 decoder 的跨位置交互都由 multi-head attention 完成，不再使用序列对齐 RNN/CNN。
2. **系统化 scaled dot-product attention**：用矩阵乘高效计算所有 query–key 相似度，并以 $1/\sqrt{d_k}$ 控制点积尺度和 softmax 饱和。
3. **提出 multi-head attention 的标准形式**：把表示投影到多个低维子空间，使模型能并行读取不同位置和不同类型的关系。
4. **用位置编码与 causal mask 补回顺序和因果约束**：固定 sin/cos 编码提供位置，decoder 上三角 mask 保证自回归条件成立。
5. **从复杂度、串行操作数和路径长度解释 self-attention 的优势**：在机器翻译常见的 $n<d$ 区间，全局 self-attention 兼具可并行性和常数级最长路径。
6. **给出质量与成本证据**：WMT 2014 EN–DE 达 28.4 BLEU；摘要和 Table 2 报告 EN–FR 达 41.8 BLEU，并在 constituency parsing 上验证跨任务适用性。

## 3. 整体架构：信息如何从输入流向输出

### 3.1 Encoder–decoder 全景

![](assets/Transformer%20-%20自注意力与序列建模/figures/fig01_transformer_architecture.png)

*图 1：原文 Figure 1。左侧 encoder 以 self-attention 与 FFN 编码完整输入；右侧 decoder 依次执行 masked self-attention、encoder–decoder attention 与 FFN，再输出下一个 token 的概率。*

原文 base Transformer 由 $N=6$ 层 encoder 和 $N=6$ 层 decoder 构成。所有子层及 embedding 的输出维度统一为：

$$d_{\text{model}}=512.$$

统一维度使每个子层都能与 residual stream 直接相加。

每个 **encoder layer** 包含两个子层：

1. multi-head self-attention；
2. position-wise feed-forward network（FFN）。

每个 **decoder layer** 包含三个子层：

1. masked multi-head self-attention；
2. 对 encoder 顶层输出执行 multi-head cross-attention；
3. position-wise FFN。

Decoder 输入是右移一位的目标序列。训练时可以并行处理所有已知目标位置，但因 causal mask，每个位置只能读取自己及此前位置；推理时未来 token 尚未生成，因此仍然必须逐 token 自回归解码。

### 3.2 三类 attention 的 $Q$、$K$、$V$ 来源

| 场景 | Query 来源 | Key / Value 来源 | 可见范围 | 主要作用 |
|---|---|---|---|---|
| Encoder self-attention | encoder 上一层输出 | encoder 上一层输出 | 全部输入位置 | 建模输入内部依赖 |
| Decoder masked self-attention | decoder 上一层输出 | decoder 上一层输出 | 当前及此前目标位置 | 建模已生成前缀并维持因果性 |
| Encoder–decoder attention | decoder 中间表示 | encoder 最顶层输出 | 全部源位置 | 按目标状态读取源端条件 |

Self-attention 中的“self”表示 $Q$、$K$、$V$ 来自同一组输入表示，不表示三者数值完全相同。它们分别乘以独立的可学习投影矩阵，承担查询、索引和内容载体三种角色。

### 3.3 Position-wise FFN

每个 attention 子层之后都有一个逐位置应用的两层 MLP：

$$\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2.$$

Base 模型使用：

$$512\rightarrow 2048\rightarrow 512.$$

“Position-wise”意味着同一层内所有 token 共享一套 FFN 参数，但每个 token 独立计算；不同 Transformer layer 之间不共享 FFN 参数。Attention 负责在 **token 维**混合信息，FFN 负责在 **通道维**进行非线性变换。把 Transformer 简化为“attention 层的堆叠”会忽略 FFN 提供的大量参数与表示容量。

### 3.4 Embedding、权重共享与输出层

源 token、目标 token 首先映射到 $d_{\text{model}}$ 维 embedding，并与位置编码相加。论文共享源 embedding、目标 embedding 与输出前 softmax 线性层的权重矩阵；在 embedding 端将词向量乘以：

$$\sqrt{d_{\text{model}}}.$$

Decoder 顶部经线性层和 softmax 得到下一 token 的概率分布。权重共享既减少参数，也让输入词表示与输出分类器处于共同语义空间。

## 4. Scaled Dot-Product Attention：内容寻址的核心算子

### 4.1 从相似度到加权读取

![](assets/Transformer%20-%20自注意力与序列建模/figures/fig02_attention_modules.png)

*图 2：原文 Figure 2。左侧是点积、缩放、可选 mask、softmax 和 value 聚合；右侧是多个投影后的 attention head 并行计算、拼接并线性映射。*

设 query、key、value 矩阵为：

$$Q\in\mathbb R^{n_q\times d_k},\quad K\in\mathbb R^{n_k\times d_k},\quad V\in\mathbb R^{n_k\times d_v}.$$

Scaled dot-product attention 定义为：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

计算过程可拆为三步：

1. $QK^\top\in\mathbb R^{n_q\times n_k}$ 计算所有 query 与 key 的兼容性分数；
2. 沿 key 维做 softmax，把每个 query 的分数归一化为权重；
3. 权重矩阵乘 $V$，得到每个 query 对可见 value 的加权和。

该算子是一种**由输入内容动态产生权重的全局读取**。它不同于固定卷积核，也不同于把历史压入单个递归状态：每个 query 可以直接访问所有允许访问的位置。

### 4.2 为什么必须除以 $\sqrt{d_k}$

若 $q_i$ 与 $k_i$ 相互独立、均值为 0、方差为 1，则：

$$q\cdot k=\sum_{i=1}^{d_k}q_i k_i.$$

其方差近似为：

$$\operatorname{Var}(q\cdot k)=d_k.$$

因此点积的典型幅度会随 $\sqrt{d_k}$ 增长。若不缩放，维度较大时 logits 容易把 softmax 推入接近 one-hot 的饱和区，非最大项梯度会很小。除以 $\sqrt{d_k}$ 后，logit 的方差回到常数量级，使 softmax 更容易工作在有用的梯度区间。

论文比较的另一种主流方法是 additive attention。二者理论复杂度相近，但 dot-product attention 可以直接利用高度优化的矩阵乘。缩放因子不是装饰性超参数，而是连接表示维度、数值尺度和优化稳定性的关键设计。

### 4.3 Attention 输出的含义

Attention 输出不是选择单个 token，而是对所有可见 value 做凸组合。其优点是可微、可并行、能同时利用多个位置；其代价是多个位置会被平均，可能降低有效分辨率。论文将 multi-head attention 作为缓解这一问题的主要机制，让不同 head 在不同子空间分别聚合。

## 5. Multi-Head Attention：并行的表示子空间

### 5.1 数学定义

论文定义：

$$\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O.$$

其中：

$$\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V).$$

各投影矩阵形状为：

$$W_i^Q\in\mathbb R^{d_{\text{model}}\times d_k},\quad W_i^K\in\mathbb R^{d_{\text{model}}\times d_k},\quad W_i^V\in\mathbb R^{d_{\text{model}}\times d_v}.$$

输出投影为：

$$W^O\in\mathbb R^{hd_v\times d_{\text{model}}}.$$

Base 模型使用：

$$h=8,\qquad d_k=d_v=d_{\text{model}}/h=64.$$

八个 head 的输出拼接后仍为 $8\times64=512$ 维，再经 $W^O$ 混合各头信息并回到 residual stream。

### 5.2 多头为什么有用

单个 full-width attention 会把各种关系压进同一组权重并一次平均。多头机制允许模型：

1. 在不同线性投影子空间中定义不同的相似度；
2. 同时关注局部邻接、长距离依赖或不同语义线索；
3. 让不同 head 在不同位置形成不同的稀疏或分散模式；
4. 在拼接后由 $W^O$ 学习如何组合各头结果。

但“每个 head 都对应一个稳定、清晰的语言学功能”不是论文证明的定理。附录可视化仅说明这种分工可能出现，不能证明每个 head 都必要、可解释或具有因果作用。

### 5.3 为什么八个 head 没把主成本乘八

每个 head 的宽度是 $d_{\text{model}}/h$。若忽略投影常数，所有 head 的 attention score 总成本为：

$$h\cdot O\left(n^2\frac{d_{\text{model}}}{h}\right)=O(n^2d_{\text{model}}).$$

因此多头的 score 计算与一个 full-width head 同阶。完整层仍包含 Q/K/V 与输出投影，其成本约为 $O(nd_{\text{model}}^2)$；attention 权重矩阵的中间存储则与 $hn^2$ 成正比。论文 Table 1 是层级渐近比较，不是对现代 kernel、显存带宽或缓存行为的完整性能模型。

## 6. Mask：训练并行与生成因果性的分界

### 6.1 Causal mask

Decoder 的 self-attention 不能看到未来目标 token。可在 softmax 前加入 mask 矩阵 $M$：

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right).$$

其中：

$$M_{ij}=\begin{cases}0,&j\le i,\\-\infty,&j>i.\end{cases}$$

于是当 $j>i$ 时，softmax 后 $A_{ij}=0$。第 $i$ 个目标位置只能依赖当前位置及其左侧位置。再配合“目标序列右移一位”，模型在预测 $y_i$ 时只使用已知前缀 $y_{<i}$。

### 6.2 为什么训练能并行，推理仍串行

训练时完整目标序列已知，可以一次构造所有位置的 $Q$、$K$、$V$，再用 mask 阻止信息泄漏；因此目标侧各位置可以并行计算。推理时，$y_i$ 尚未生成就无法构造依赖它的后续输入，所以仍需逐步生成。

这一区分非常重要：**Transformer 消除的是层内序列递推，不是自回归概率分解本身。**

### 6.3 Causal mask 与 padding mask 的边界

- **Causal mask**：论文明确使用，用于屏蔽未来位置；
- **Padding mask**：工程实现中常用于屏蔽补齐 token，但不是论文的核心创新；
- **Cross-attention mask**：实际系统可据源端 padding 进一步限制可见位置，同样属于实现层面的必要处理。

不能把这些 mask 混为一谈：它们屏蔽的对象和解决的问题不同。

## 7. 位置编码：在无递归架构中注入顺序

### 7.1 为什么需要位置表示

没有位置编码时，self-attention 对输入排列是置换等变的：交换 token 顺序，输出只会对应交换，模型无法仅凭内容区分不同词序。原文在 encoder 和 decoder 底部都执行：

$$X_0=\operatorname{Embedding}(tokens)+PE.$$

位置编码与 embedding 维度相同，因此可逐元素相加，不增加序列长度或 residual stream 宽度。

### 7.2 正弦与余弦编码

对位置 $pos$ 和通道索引 $i$：

$$PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

$$PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

不同通道对应不同频率，波长从 $2\pi$ 到 $10000\cdot2\pi$ 形成几何级数。高频通道对局部位置差更敏感，低频通道能在更长尺度上缓慢变化。

### 7.3 固定偏移为何可由线性变换表示

对任意频率 $\omega$ 和固定偏移 $k$：

$$\begin{bmatrix}\sin(\omega(pos+k))\\\cos(\omega(pos+k))\end{bmatrix}=\begin{bmatrix}\cos(\omega k)&\sin(\omega k)\\-\sin(\omega k)&\cos(\omega k)\end{bmatrix}\begin{bmatrix}\sin(\omega pos)\\\cos(\omega pos)\end{bmatrix}.$$

固定相对位移对应每个 sin/cos 二维子空间中的旋转线性变换。作者据此假设模型较容易学习按相对位置进行 attention。需要注意：这是设计动机，不是原论文通过专门实验严格证明的机制。

### 7.4 与 learned positional embedding 的实验边界

Table 3(E) 中，learned positional embedding 的 dev PPL 为 4.92、BLEU 为 25.7；正弦 base 为 4.92、25.8，二者几乎相同。论文选择正弦编码，是因为它**可能**外推到训练时未见的更长长度，而不是因为实验已证明其普遍更优。原论文没有系统的长上下文外推实验。

## 8. Post-LN、残差连接与训练稳定性

### 8.1 原始子层顺序

原始 Transformer 的每个子层采用：

$$\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).$$

精确顺序是：Sublayer → dropout → 与输入 $x$ 残差相加 → LayerNorm。这就是后来所称的 **Post-LN Transformer**。

### 8.2 各组件的作用

- **残差连接**提供 identity path，使每层只需学习增量更新，并帮助梯度传播；
- **LayerNorm**沿单个 token 的特征维归一化，不依赖 batch 统计，适合可变长度序列；
- **统一的 $d_{\text{model}}$**保证 attention、FFN 和 embedding 都能进入同一 residual stream；
- **Residual dropout**施加在子层输出与输入相加之前；
- **Embedding dropout**施加在 token embedding 与位置编码之和上。

### 8.3 不要把后世配置倒写进原论文

Pre-LN 的形式通常为 $x+F(\operatorname{LN}(x))$，RMSNorm 则省略均值中心化；这些设计在后来的深层 Transformer 和大语言模型中很常见，但都不是这篇论文的原始配置。原论文的成功证据针对六层 Post-LN encoder/decoder，不能直接外推为任意深度的稳定训练结论。

## 9. 为什么选择 Self-Attention：复杂度、并行度与路径长度

![](assets/Transformer%20-%20自注意力与序列建模/figures/table01_complexity.png)

*表 1：原文 Table 1。全局 self-attention 以二次序列复杂度换取常数级串行操作数和任意位置之间的一跳连接。*

设序列长度为 $n$，表示维度为 $d$，卷积核宽度为 $k$，局部 attention 邻域大小为 $r$：

| 层类型 | 每层复杂度 | 最少串行操作数 | 最长路径 |
|---|---:|---:|---:|
| Self-attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted self-attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

### 9.1 三个指标不能混为一谈

1. **算术复杂度**：当 $n<d$ 时，$n^2d<nd^2$，全局 self-attention 比 dense recurrent layer 更有利；论文指出机器翻译中的 subword sentence 通常处于该区间。
2. **并行度**：self-attention 与 convolution 可同时计算所有位置；RNN 必须按时间步执行 $n$ 次递推。
3. **依赖路径长度**：self-attention 中任意两位置在一层内直接相连；RNN 和局部卷积需要穿过更多中间状态。

较短路径有利于前向信息与反向梯度跨越长距离，但“最长路径为常数”不表示总计算为常数，也不表示长序列一定高效。

### 9.2 二次复杂度是结构性代价

全局 attention 要显式或隐式处理 $n\times n$ 的兼容性关系，时间和中间存储随 $n^2$ 增长。论文已提出 restricted attention 作为未来方向：把每个位置的可见范围限制到 $r$，复杂度降为 $O(rnd)$，但最长路径增加到 $O(n/r)$。

这揭示了一个长期存在的折中：**全局连通性、计算成本和路径长度不能同时免费获得。**

### 9.3 比较结论的适用边界

Table 1 是渐近层级分析，没有计入投影层常数、硬件利用率、内存带宽、kernel 融合、batch 形状和缓存策略。它能解释 2017 年机器翻译场景下的结构优势，但不能据此断言标准 attention 在任意序列长度和任意硬件上都比 RNN/CNN 更快。

## 10. 训练配置与可复现细节

### 10.1 数据、词表与 batching

- **WMT 2014 English–German**：约 450 万句对，共享 source–target BPE 词表约 37,000；
- **WMT 2014 English–French**：约 3,600 万句对，32,000 word-piece 词表；
- 句对按近似长度分桶；
- 每个 batch 约含 25,000 source tokens 和 25,000 target tokens。

按 token 数而不是句子数控制 batch，有助于稳定不同句长下的显存与计算负载。

### 10.2 Base 与 Big 配置

| 配置 | $N$ | $d_{\text{model}}$ | $d_{ff}$ | $h$ | $d_k,d_v$ | Dropout | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 6 | 512 | 2048 | 8 | 64, 64 | 0.1 | 65M |
| big | 6 | 1024 | 4096 | 16 | 64, 64 | 0.3 | 213M |

English–French big 模型例外使用 $P_{drop}=0.1$，而不是 0.3。

### 10.3 硬件与训练时长

训练使用一台配备 8 张 NVIDIA P100 的机器：

- base：约 0.4 秒/step，共 100,000 steps，约 12 小时；
- big：约 1.0 秒/step，共 300,000 steps，约 3.5 天。

这些时间体现了当时硬件和实现环境下的训练吞吐，不应直接与现代硬件的 wall-clock 时间比较。

### 10.4 Adam 与学习率日程

Adam 超参数为：

$$\beta_1=0.9,\qquad\beta_2=0.98,\qquad\epsilon=10^{-9}.$$

学习率为：

$$\operatorname{lrate}=d_{\text{model}}^{-1/2}\min\left(step^{-1/2},\;step\cdot warmup^{-3/2}\right).$$

其中 $warmup=4000$。前 4,000 步学习率线性上升，之后按 $step^{-1/2}$ 衰减；$d_{\text{model}}^{-1/2}$ 则让不同模型宽度下的更新尺度更可比。这套 warmup + inverse-square-root schedule 后来成为 Transformer 训练的经典配置。

### 10.5 正则化与解码

- residual dropout：base 为 0.1；
- embedding 与 positional encoding 之和也做 dropout；
- label smoothing：$\epsilon_{ls}=0.1$；
- base 结果平均最后 5 个 checkpoint；
- big 结果平均最后 20 个 checkpoint；
- beam size 为 4；
- length penalty 为 $\alpha=0.6$；
- 最大输出长度设为输入长度 + 50，并允许提前终止。

Label smoothing 会让模型不那么确信，因而可能使 perplexity 变差，却改善 accuracy 与 BLEU。Checkpoint averaging、beam search 和 length penalty 都会实质影响最终 BLEU，复现不能只复制网络层数和维度。

## 11. 机器翻译结果：质量、成本与数字冲突

![](assets/Transformer%20-%20自注意力与序列建模/figures/table02_translation_results.png)

*表 2：原文 Table 2。Transformer big 在 EN–DE 达 28.4 BLEU，在表中 EN–FR 达 41.8 BLEU；base EN–DE 的估算训练成本显著低于所列强基线。*

### 11.1 WMT 2014 主要结果

- **EN–DE newstest2014**：base 27.3 BLEU，big 28.4 BLEU；big 比表中此前最佳 ensemble 26.36 高约 2 BLEU；
- **EN–FR newstest2014**：base 38.1 BLEU；摘要和 Table 2 给出 big 41.8 BLEU；
- **估算训练成本**：base EN–DE 为 $3.3\times10^{18}$ FLOPs，big 为 $2.3\times10^{19}$ FLOPs。

原文训练 FLOPs 由训练时间、GPU 数量和估计的单卡持续单精度吞吐相乘得到，不是逐算子精确计数。跨论文比较还会受数据管线、实现成熟度和硬件利用率影响，因此更适合做数量级参考，而不是严格同平台 benchmark。

### 11.2 EN–FR 的 41.8 与 41.0 冲突

官方 PDF 的摘要和 Table 2 报告 EN–FR big 为 **41.8 BLEU**，但 Section 6.1 正文写的是 **41.0 BLEU**。本笔记保留这一冲突：引用论文主表结果时采用 41.8，同时注明正文存在 41.0 的不一致，不擅自替作者统一数字。

### 11.3 结果是否支撑核心主张

对“序列转导可以不依赖 recurrence/convolution”这一主张，证据很强：模型在两个大规模翻译任务上达到当时 SOTA，并显著缩短训练周期。对“适用于所有序列任务、所有长度和所有生成方式”的外推则证据不足，因为主体实验仍集中于中等长度的文本翻译。

## 12. 模型变体与消融：哪些设计有直接证据

![](assets/Transformer%20-%20自注意力与序列建模/figures/table03_ablation.png)

*表 3：原文 Table 3。它在 EN–DE newstest2013 开发集上系统改变头数、key/value 维度、模型深宽度、正则化和位置表示；未列出的参数与 base 相同。*

### 12.1 头数：多头有益，但不是越多越好

在近似固定计算量下：

| 头数 $h$ | 每头 $d_k=d_v$ | PPL | dev BLEU |
|---:|---:|---:|---:|
| 1 | 512 | 5.29 | 24.9 |
| 4 | 128 | 5.00 | 25.5 |
| 8（base） | 64 | 4.92 | 25.8 |
| 16 | 32 | 4.91 | 25.8 |
| 32 | 16 | 5.01 | 25.4 |

单头比最佳设置低 0.9 BLEU，支持多头机制有效；32 个过窄 head 又出现下降，说明头数与每头容量存在折中。实验不支持“head 越多越好”的单调结论。

### 12.2 Key 维度：匹配函数需要容量

将 $d_k$ 降到 16 或 32 时，dev BLEU 分别为 25.1 和 25.4，低于 base 的 25.8。作者据此认为 query–key compatibility 并不简单，过小的 key 子空间会削弱匹配能力，甚至提示更复杂的 compatibility function 可能有益。

### 12.3 深度、模型宽度与 FFN 宽度

- $N=2/4/8$：BLEU 为 23.7/25.3/25.5；过浅明显不足，继续加深的收益变小；
- $d_{\text{model}}=256/1024$：BLEU 为 24.5/26.0；更宽模型更好，但参数量从 28M 增至 168M；
- $d_{ff}=1024/4096$：BLEU 为 25.4/26.2；FFN 容量是重要性能来源；
- big：PPL 4.33，dev BLEU 26.4，参数量 213M。

这些结果再次说明 Transformer 的性能来自 attention、FFN、宽度、深度和正则化的组合，而不是单个 attention 公式。

### 12.4 Dropout 与 label smoothing

- residual dropout 从 0.1 改为 0.0，BLEU 从 25.8 降到 24.6，显示明显过拟合；改为 0.2 时为 25.5；
- label smoothing 从 0.1 改为 0.0，PPL 改善到 4.67，但 BLEU 降到 25.3；改为 0.2 时 PPL 为 5.47、BLEU 为 25.7。

这里清楚展示了：**更低的 token-level perplexity 不保证更高的 sequence-level BLEU。** 训练目标的校准性质会影响生成质量。

### 12.5 正弦与 learned position

Learned positional embedding 的 PPL 4.92、BLEU 25.7，几乎等于正弦 base 的 4.92、25.8。直接证据只支持“二者在该训练与开发长度内表现接近”，不支持“正弦编码在所有任务和长度上更优”。

### 12.6 消融证据的缺口

Table 3 并不是严格逐组件删除实验。论文没有分别移除 residual、LayerNorm、FFN、cross-attention 或 causal mask 并报告结果，因此不能把这些组件的必要性都归为本表直接证明。它更准确地说是一组 architecture variations 与 regularization studies。

## 13. 注意力可视化与跨任务泛化

### 13.1 长距离依赖可视化

![](assets/Transformer%20-%20自注意力与序列建模/figures/fig03_attention_visualization.png)

*图 3：原文 Figure 3。第 5 层 encoder self-attention 的多个 head 从 “making” 直接连到远处的 “more/difficult”，展示一层中跨越长距离短语依赖的能力。*

该图与常数路径分析相呼应：远距离 token 可以在单层直接交换信息，不必沿 RNN 状态逐步传递。附录还展示了与指代和句法结构相关的 head。

但 attention weight 不能自动等同于完整因果解释。输出还受到 value 内容、其他 head、输出投影、残差路径、FFN 与后续层影响。原文只把这些图当作定性案例；它们说明模型能形成结构化模式，不证明某条注意力边就是预测的唯一原因。

### 13.2 Constituency parsing 实验

论文在 Penn Treebank WSJ 上训练四层、$d_{\text{model}}=1024$ 的 Transformer：

- 仅约 40K WSJ 训练句：WSJ Section 23 F1 = 91.3；
- 使用约 17M 句半监督语料：F1 = 92.7。

该任务的输出具有强结构约束，且通常比输入更长。结果表明 Transformer 不只适用于翻译。但作者只进行少量超参数搜索，92.7 也低于表中的 multi-task 93.0 与 generative 93.3，因此合理结论是“具有跨任务泛化能力”，而不是“全面优于所有任务专用解析器”。

## 14. 局限性与批判性思考

### 14.1 原论文明确或隐含承认的局限

1. **长序列二次复杂度**：全局 attention 的 $O(n^2d)$ 会随长度迅速增长，作者把 local/restricted attention 列为未来方向；
2. **自回归生成仍然串行**：论文结论明确把“让 generation 更少串行”作为后续目标；
3. **位置外推未经系统验证**：正弦编码可能外推到更长序列，但论文只给出设计动机；
4. **加权平均会损失分辨率**：multi-head 可缓解，但没有理论保证能完全消除；
5. **任务范围有限**：主体证据来自机器翻译，另加一个 constituency parsing 任务。

### 14.2 实验与报告层面的限制

1. **缺少多随机种子统计**：没有报告均值与方差，小幅 BLEU 差异不宜过度解读；
2. **消融不完整**：没有对每个核心组件做逐项删除；
3. **跨论文成本比较近似**：FLOPs 是按持续吞吐估计，公平性受实现和硬件利用率影响；
4. **可视化是案例而非机制证明**：清晰 attention pattern 不能直接证明因果重要性；
5. **结果文本存在数字冲突**：EN–FR 的 41.8 与 41.0 提醒读者核对具体表格和版本。

### 14.3 架构层面的限制

1. **缺少内建局部性**：全局内容寻址通用，但没有卷积那样强的局部归纳偏置；
2. **标准 attention 的 KV 与 score 存储昂贵**：尤其在长序列和自回归服务中，显存与带宽可能先于 FLOPs 成为瓶颈；
3. **原始 Post-LN 不代表深层可扩展性已解决**：六层模型训练成功，不能推出数十或数百层也同样稳定；
4. **训练并行不等于端到端低延迟**：解码仍受 token-by-token 依赖限制。

### 14.4 适用边界

标准 Transformer 最适合：序列长度适中、并行硬件充足、任务需要全局内容寻址的场景。若序列极长、显存严格受限、要求流式固定状态推理，标准全局 attention 未必是最佳选择。其核心贡献是提供一个通用、可并行、短路径的序列混合骨架，而不是对所有计算约束都占优。

## 15. 后世变体边界：哪些属于原论文，哪些不属于

以下内容用于建立技术谱系，不表示 2017 年论文已经提出或验证这些后续方案。

### 15.1 Encoder-only、decoder-only 与 encoder–decoder

- **原论文**：完整 encoder–decoder Transformer；
- **BERT 类**：主要保留双向 encoder，用于掩码建模和表示学习；
- **GPT 类**：主要保留 causal decoder，用于自回归语言建模；
- **T5 类**：继续采用 encoder–decoder，但训练目标、位置偏置等已经改变。

三者共享 attention、FFN 和 residual 骨架，但条件结构、mask 与目标函数不同。Decoder-only LLM 的能力和工程经验不能全部倒归为原论文的直接实验结论。

### 15.2 归一化与残差变体

- **原文**：Post-LN，$\operatorname{LN}(x+F(x))$；
- **后续常见**：Pre-LN、RMSNorm、residual scaling、parallel residual 等。

这些方案主要面向更深网络的稳定训练或更简化的归一化，不属于原始 Transformer 公式。

### 15.3 位置表示变体

- **原文**：绝对 sin/cos 编码，加到 embedding；
- **后续**：learned absolute position、relative position bias、RoPE、ALiBi 等。

后续方法更直接地建模相对位置、长度外推或长上下文。原论文只实验了固定正弦与 learned absolute embedding，不能说它已经提出 relative bias 或 RoPE。

### 15.4 高效与长序列 attention

原文已经讨论 restricted attention，但后续才出现大量 sparse/local、sliding-window、low-rank、kernel/linear attention 等方法。

还需区分两类优化：

- **架构近似或稀疏化**：改变可见连接或 attention 的数学形式；
- **FlashAttention 类 kernel**：通常保持标准 softmax attention 的数学结果，主要通过分块与 IO-aware 实现减少显存访问和中间存储。

二者都可提高效率，但技术目标和结果边界不同。

### 15.5 MQA、GQA 与 KV cache

原文采用标准 multi-head attention，每个 head 都有独立 Q/K/V 投影。后续 MQA/GQA 让多个 query heads 共享更少的 K/V heads，以降低自回归推理时的 KV cache 和带宽压力。这是面向大模型服务的后续折中，不属于原始设计。

### 15.6 FFN 与激活函数变体

原文 FFN 是两层线性变换加 ReLU。后续常见 GELU、SwiGLU、GeGLU、Mixture-of-Experts 等，都改变了通道混合和参数使用方式。它们沿用“attention 做 token mixing、FFN 做 channel mixing”的骨架，但不是原论文已验证的组件。

## 16. 可复用的理解框架与工程启发

1. **把序列建模拆成 token mixing 与 channel mixing**：attention 跨位置读取，FFN 逐位置变换；
2. **可并行性与 FLOPs 同等重要**：RNN 的问题不仅是算术量，更是序列依赖导致的硬件利用限制；
3. **短路径是一种归纳偏置**：任意位置一跳可达有利于长距离依赖，但会牺牲局部性和长序列成本；
4. **缩放因子来自统计尺度**：$1/\sqrt{d_k}$ 用于控制点积方差和 softmax 饱和；
5. **Mask 定义了信息边界**：同一个 attention 算子通过不同 mask 可实现双向编码、因果生成或局部连接；
6. **模型质量来自组件组合**：FFN 宽度、head 维度、dropout、label smoothing 与解码策略都显著影响结果；
7. **实验结论要和证据范围对齐**：翻译 SOTA 不等于任意任务、任意长度和任意硬件上的全面优势；
8. **经典论文也要做版本核对**：表格、摘要与正文可能不一致，应记录来源而不是静默修正。

## 17. 关键术语表

| 术语 | 含义 |
|---|---|
| Self-attention | $Q$、$K$、$V$ 来自同一序列表示，按内容动态混合位置 |
| Cross-attention | Query 来自 decoder，Key/Value 来自 encoder 的条件读取 |
| Scaled dot-product attention | 对 $QK^\top$ 除以 $\sqrt{d_k}$ 后做 softmax 并聚合 $V$ |
| Multi-head attention | 在多个低维投影子空间并行 attention，再拼接和输出投影 |
| Causal mask | 把未来位置的 logit 置为 $-\infty$，维持自回归因果性 |
| Position-wise FFN | 对每个 token 独立应用、跨位置共享参数的两层 MLP |
| Positional encoding | 注入 token 顺序的向量；原文采用固定 sin/cos |
| Post-LN | 先做子层和残差相加，再执行 LayerNorm 的原始配置 |
| Label smoothing | 平滑 one-hot 目标，降低过度确信并改善 BLEU |
| Maximum path length | 任意两个位置间信息或梯度需要经过的最长计算路径 |
| Residual stream | 各子层通过残差连接持续更新的统一宽度表示 |
| KV cache | 自回归推理中缓存历史 key/value 的后续工程机制，原论文未专门讨论 |

## 18. 图表索引与核验结论

| 本地文件 | 原文位置 | 本文用途 | 核验结果 |
|---|---|---|---|
| `figures/fig01_transformer_architecture.png` | Figure 1，PDF p.3 | 完整 encoder–decoder 架构 | 完整、清晰、含图注 |
| `figures/fig02_attention_modules.png` | Figure 2，PDF p.4 | scaled dot-product 与 multi-head 流程 | 完整、清晰、含图注 |
| `figures/table01_complexity.png` | Table 1，PDF p.6 | 复杂度、串行操作数与路径长度 | 表头及四类层完整可读 |
| `figures/table02_translation_results.png` | Table 2，PDF p.8 | WMT BLEU 与训练成本 | 基线与 Transformer 行完整可读 |
| `figures/table03_ablation.png` | Table 3，PDF p.9 | 头数、维度、深宽度、正则化和位置变体 | A–E、base、big 完整可读 |
| `figures/fig03_attention_visualization.png` | Figure 3，PDF p.13 | 长距离 attention 定性案例 | token、连线、色块与图注完整 |

六张图表均为 `assets/Transformer - 自注意力与序列建模/figures/` 下的现有文件，引用路径相对当前笔记有效。图表来自官方 `paper.pdf` 的 300 DPI 裁剪；来源、SHA-256、页码和裁剪坐标见 [[assets/Transformer - 自注意力与序列建模/sources.md|sources.md]]。
