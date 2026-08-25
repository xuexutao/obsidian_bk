# 多模态大模型架构与训练全流程

## 基本信息

- **领域**：VLM / 基础模块
- **类型**：领域综述（Survey）
- **主题**：Transformer 架构原理、大模型训练全流程（CPT / Mid-train / SFT / RL / OPD）、主流 VLM 架构（LLaVA / Qwen-VL / InternVL）、分布式训练框架（Megatron-LM / DeepSpeed）、强化学习对齐框架（veRL / OpenRLHF）
- **写作时间**：2026-08-25
- **定位**：作为"旭涛的技术视野"知识沉淀的体系化综述，把多模态大模型从底层架构到训练、对齐、分布式落地的完整技术栈串成一条可复用的认知主线。本文不求面面俱到，而求把每个环节的**原理、关键公式、实现差异、工程权衡**讲透。

---

## 1. 研究背景：从 Transformer 到多模态大模型

要讲清楚多模态大模型，得先回到它唯一的计算底座——Transformer。今天几乎所有 VLM（视觉语言模型）、视频生成模型、乃至 3D 生成模型，其主干都是同一个东西：一个用自注意力机制堆叠起来的深度网络。理解 Transformer 的注意力计算，是理解后面"为什么 VLM 要把图像编码成 token"、"为什么分布式训练要把矩阵乘法切开来"这些问题的起点。

### 1.1 Transformer 架构原理

Transformer 由 Vaswani 等人在 2017 年的《Attention Is All You Need》中提出，最初用于机器翻译，但它真正的价值在于彻底抛弃了 RNN 的串行递归，把序列建模变成了纯并行的矩阵运算。一个 Transformer 由 Encoder 与 Decoder 两个对称的堆叠模块组成（现代仅解码器的 LLM 只保留 Decoder 一侧），每个模块内部是两条固定的子层：多头自注意力（Multi-Head Attention）与位置前馈网络（Feed-Forward Network），两条子层外都包着一层残差连接和 LayerNorm。

#### 1.1.1 缩放点积注意力：注意力到底在算什么

先看三个矩阵从哪来。给定输入序列的隐藏表示 $X\in\mathbb{R}^{n\times d}$（$n$ 是序列长度，$d$ 是隐藏维度），用三个可学习的权重矩阵 $W_Q, W_K, W_V$ 做线性投影：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

注意力可以直观理解为一次"软检索"：用 Query 去匹配 Key 得到相关性权重，再用这个权重对 Value 做加权求和。形式上就是缩放点积注意力：

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

其中除以 $\sqrt{d_k}$（$d_k$ 是 Key 的维度）这一步不是随手加的，它有严格的方差论证：假设 $q,k$ 的每个分量独立同分布、均值 0 方差 1，那么点积 $q\cdot k=\sum_i q_i k_i$ 的方差就是 $d_k$。这意味着点积值会随维度线性发散到 $\pm\sqrt{d_k}$ 量级，softmax 的输入一旦落入饱和区，梯度就趋近于零，训练会变得极慢甚至不收敛。除以 $\sqrt{d_k}$ 把方差拉回 1，让 softmax 落在梯度敏感区。这也是它被称为"缩放"点积注意力的原因。（这里"每个分量独立同分布、均值 0 方差 1"的假设从何而来——哪一部分由 LayerNorm 保证、哪一部分其实由 $W_Q,W_K$ 的初始化保证——见 1.1.5 节。）

另一个容易忽略的细节是**掩码**。自回归生成时，位置 $i$ 只能看到 $i$ 及之前的位置，所以要在 softmax 之前把上三角（$j>i$ 的位置）的对数置为 $-\infty$，softmax 之后这些位置的权重自然为 0。这就是 Decoder 侧自注意力的"因果掩码（Causal Mask）"。

最后是这个算子的复杂度：$QK^{T}$ 是 $n\times n$ 的矩阵，计算量 $O(n^2 d)$、显存 $O(n^2)$。这个**平方复杂度**是 Transformer 一切后续优化的根源——FlashAttention 用分块 + 在线 softmax 省显存，稀疏/线性注意力用近似换掉 $n^2$，但都只是缓解而非根治。多模态里视觉 token 动辄上千，这个问题会被放大，后面 1.2.4 还会回到它。

#### 1.1.2 多头注意力：为什么要"多头"

单头注意力只有一组 $W_Q,W_K,W_V$，表达能力有限，而且 softmax 本身有"平均化"倾向——当多个位置都相关时，注意力权重会被摊薄。多头注意力的做法是把 $d$ 维切成 $h$ 个低维子空间，每组独立做注意力，最后拼接起来再投影：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^{O},\quad \text{head}_i=\text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
$$

每个头的维度 $d_k=d/h$。多头的意义在于让不同头"各司其职"：有的头关注位置临近性，有的头关注句法依存，有的头关注指代关系，有的头关注语义相关性。这相当于给了模型 $h$ 个并行的"关系探测器"，比单头更灵活。用 PyTorch 写出来非常短：

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None, scale=None):
    # q,k,v: (batch, heads, seq, d_k)
    d_k = q.size(-1)
    scale = scale if scale is not None else d_k ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (b, h, n, n)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)
```

这段代码就是 LLM 里每一层、每一个头的核心计算，后面 VLM 里的视觉 token 也是以同样方式参与这段计算——只是序列里混入了图像 patch 的位置。

图 1 给出了完整的架构示意图。

![Transformer 架构](assets/多模态大模型架构与训练全流程%20-%20Survey/figures/01-transformer.png)

*图 1：Transformer 架构（Vaswani et al., 2017）。左侧 Encoder 与右侧 Decoder 都由多头注意力 + 前馈网络构成，外部包残差连接与 LayerNorm；底部给出了缩放点积注意力与多头注意力的计算公式。图中青绿色箭头表示 Encoder 输出作为 Key/Value 喂入 Decoder 的交叉注意力（Cross Attention），这是 Encoder-Decoder 架构信息传递的关键路径。来源：自绘，公式引自原论文。*

#### 1.1.3 位置编码：从绝对位置到 RoPE

自注意力本身是置换不变的：如果把输入序列整体打乱，注意力输出只是跟着重排，数值不变。换句话说，它"看不见顺序"。要让模型知道 token 的先后，必须显式注入位置信息。这条线经历了三代演进：

1. 绝对位置（Sinusoidal）：原论文用正弦/余弦函数按位置生成编码，$PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$，$PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d})$。它免费、无需训练，但对"训练长度之外"的外推能力有限。
2. 可学习位置：直接学一个位置嵌入表，简单直观，但表长固定，无法外推到更长序列，且占显存。
3. RoPE（Rotary Position Embedding，旋转位置编码）：苏剑林提出，是现代 LLM 与 VLM 的主流方案。它不做加法、做旋转——把 $q,k$ 相邻两维视为复数的实部虚部，按位置 $m$ 乘上一个旋转角 $m\theta_i$（$\theta_i=10000^{-2i/d}$）：

$$
\begin{bmatrix} q_{m,2i} \\ q_{m,2i+1} \end{bmatrix}
=
\begin{bmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{bmatrix}
\begin{bmatrix} q_{m,2i} \\ q_{m,2i+1} \end{bmatrix}
$$

RoPE 有两个关键性质：一是旋转后 $q_m^\top k_n$ 只依赖**相对位置** $m-n$，天然编码了相对关系；二是它支持外推（配合 YaRN 等技巧可以做长上下文扩展）。更重要的是，RoPE 的"位置"可以推广到多维——这正是 Qwen2-VL 的 M-RoPE（时间 + 高度 + 宽度三维位置）的思想来源，第 3.2 节会再展开。

#### 1.1.4 前馈网络与激活函数

如果注意力负责"交流"（让不同位置的 token 互相交换信息），那么 FFN 负责"思考"（对每个位置的表示单独做非线性变换，存储知识）。原始形式是两层线性加一个激活：

$$
\text{FFN}(x)=W_2\,\sigma(W_1 x)
$$

中间维度通常是 $4d$（"扩 4 倍再缩回"），这也是为什么 FFN 占了 Transformer 约 2/3 的参数量——参数主要堆在这里。激活函数经历了 ReLU → GELU → **SwiGLU** 的演进：SwiGLU 引入门控机制，用一条门控支路与一条变换支路逐元素相乘：

$$
\text{SwiGLU}(x)=\text{Swish}(xW_1)\odot(xW_3)\,W_2
$$

SwiGLU 被 PaLM、LLaMA 等模型验证能带来稳定的质量提升，代价是中间维度要调到约 $\frac{8}{3}d$ 才能保持总参数量不变。

#### 1.1.5 归一化与残差：稳定深层训练的基石

残差连接 $x+\text{Sublayer}(x)$ 让梯度有一条"高速公路"直达底层，缓解了深层网络的梯度消失。归一化则有两层演进：

- **LayerNorm → RMSNorm**：LayerNorm 是减均值、除方差、再仿射三步；RMSNorm 去掉"减均值"，只按均方根缩放。RMSNorm 计算更便宜、数值更稳定，效果相当，被 LLaMA 系全面采用。
- **Post-Norm → Pre-Norm**：原论文把归一化放在残差**之后**（Post-Norm），现代模型几乎都改为放在**之前**（Pre-Norm，即 $x+\text{Sublayer}(\text{LN}(x))$）。Pre-Norm 让梯度更平稳、训练更不容易崩，尤其是配合大学习率时。这一点呼应了深度学习里"谱塌缩 / 权重矩阵低秩退化"那类训练动力学议题——归一化的位置与形式，直接影响训练中权重矩阵的谱结构是否健康。

**LayerNorm 究竟沿哪个维度统计。** 对单个 token 的隐状态 $x\in\mathbb{R}^{d}$，LN 沿**特征维**求均值与方差，再归一化、仿射：

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2,\qquad
\mathrm{LN}(x)=\underbrace{\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}}_{\hat x}\odot\gamma+\beta
$$

关键在"沿哪个维度"：LN 是**逐样本、跨特征**，与 batch 完全无关；BatchNorm 反过来，是**逐特征、跨样本**。这正是 Transformer 选 LN 的原因——变长序列与自回归单步推理下都成立，不需要 batch 统计量，训练与推理行为一致（BN 还得维护 running mean/var 这套双轨制）。图 7 用一个 $(N, C, H\!\times\!W)$ 的张量把四种归一化的统计范围画在了一起。

![归一化方式对比](assets/多模态大模型架构与训练全流程%20-%20Survey/figures/07-normalization-variants.svg)

*图 7：四种归一化的统计范围对比（Wu & He, 2018, Group Normalization）。蓝色格子表示"共享同一组均值/方差"的元素集合：BatchNorm 固定通道、跨 batch 与空间统计；LayerNorm 固定样本、跨全部通道与空间统计；InstanceNorm 固定样本与通道；GroupNorm 固定样本、按通道分组。映射到 Transformer：$N$ 对应 batch × 序列位置，$C$ 对应隐藏维 $d$，$H\!,\!W$ 维退化——于是 LN 就是"每个 token 独立地在自己的 $d$ 维特征上归一化"。来源：重绘自原论文图 2。*

**LN 给的是范数，不是独立性。** §1.1.1 里"假设 $q,k$ 各分量独立同分布、均值 0 方差 1"这句常被含混地归因于 LayerNorm，但严格说 LN 并不让分量独立，它甚至引入了轻微的负相关。注意 $\hat x$ 满足的是两条**逐样本硬约束**（不是期望意义上成立）：$\sum_i\hat x_i=0$ 与 $\frac{1}{d}\sum_i\hat x_i^2=1$，即 $\|\hat x\|=\sqrt d$。几何上就是把 $x$ 投影到超平面 $\mathbf 1^\top x=0$ 再拉到半径 $\sqrt d$ 的球面上。而第一条约束恰恰**制造**了依赖：两边平方求期望得 $0=d+d(d-1)\rho$，故任意两分量的相关系数恰为 $\rho=-\frac{1}{d-1}$。

真正的分工要看 $q=W_Q x$ 这一步（$x$ 是已 LN 过的隐状态，$W_Q\in\mathbb{R}^{d_k\times d}$ 各元素 iid、均值 0、方差 $1/d$）：

- 方差为 1 ← LayerNorm 的功劳。 $\mathrm{Var}[q_i]=\sigma_W^2\|x\|^2=\frac{1}{d}\cdot d=1$。LN 把 $\|x\|^2$ 钉在 $d$ 上，与初始化尺度 $1/d$ 正好对消。没有 LN，残差流逐层累加会让 $\|x\|$ 随深度漂移，方差就不再是 1。
- 分量间不相关 ← $W_Q$ 行间独立的功劳，与 LN 无关。 对 $i\ne j$，$\mathrm{Cov}[q_i,q_j]=\sum_{a,b}\mathbb{E}[W_{ia}W_{jb}]x_ax_b=0$，因为 $W$ 的不同行独立且均值为 0——这个结果与 $x$ 是什么完全无关。随机投影把 LN 在 $x$ 里造出的那点 $-1/(d-1)$ 负相关彻底洗掉了；若 $W$ 取高斯，$q$ 的各分量严格独立。

而且 §1.1.1 实际需要的条件比"独立"弱得多。方差那步只需两两不相关：由 $q\perp k$（不同权重矩阵）且各自分量不相关、均值 0，有 $\mathrm{Cov}[q_ik_i,q_jk_j]=\mathbb{E}[q_iq_j]\mathbb{E}[k_ik_j]=0$，交叉项全灭，于是 $\mathrm{Var}[q\cdot k]=d_k$。高斯性也不必靠中心极限定理——**先条件在 $q$ 上**更干净：给定 $q$，$q\cdot k$ 是 iid 高斯的加权和，故 $q\cdot k\mid q\sim\mathcal N(0,\|q\|^2)$ **精确**成立；而 $\|q\|^2$ 是 $d_k$ 个独立平方项之和，相对波动仅 $\sqrt{2/d_k}$（$d_k=64$ 时约 18%），高度集中，于是 $z\sim\mathcal N(0,d_k)$ 站得住。

**这套论证只在初始化时刻成立**，有三处真实的破口，都指向后文的训练稳定性议题：可学习增益 $\gamma$ 训练后不再是 $\mathbf 1$，$\|x\|$ 随之偏离 $\sqrt d$；$W_Q,W_K$ 的谱范数在训练中增长，$\mathrm{Var}[q_i]$ 早已不是 1——这两条都是 **attention logit explosion** 的来源，也正是 QK-Norm（对 $q,k$ 各做一次 RMSNorm）要硬性约束的东西；此外只有 **Pre-LN** 才把 LN 放在注意力块**输入**上，$q,k$ 确实由刚归一化过的向量算出，Post-LN 里喂给注意力的是未归一化的残差流，$\|x\|$ 随深度增长，论证会明显变松，这也是 Post-LN 需要 warmup 才训得稳的原因之一。顺带一提，**RMSNorm** 去掉中心化后反而更契合上面的推导：它同样把 $\|x\|$ 钉在 $\sqrt d$，却不引入 $\sum_i\hat x_i=0$ 那条约束，连 $-1/(d-1)$ 的负相关都没有。

#### 1.1.6 现代 LLM 的工程化改造清单

把 GPT-2 到 LLaMA 的演化串起来，可以提炼出一份"现代 LLM 底座标准配方"：Post-Norm → Pre-Norm、LayerNorm → RMSNorm、绝对位置 → RoPE、ReLU → SwiGLU、多头注意力 → **GQA/MQA**。前四项上面已讲，最后一项值得单独说：

KV Cache 与 GQA。自回归生成时，每个新 token 都要与所有历史 token 做注意力，如果每次都重算会重复大量计算，所以把历史的 $K,V$ 缓存起来（KV Cache）。KV Cache 的大小正比于"层数 × 头数 × 序列长度"，长序列或视觉 token 多时，它成了显存大头。MQA（Multi-Query Attention） 让所有 query 头共享同一组 $K,V$，把 KV Cache 缩到 1/头数；GQA（Grouped-Query Attention） 折中，把 query 头分组，每组共享一组 $K,V$（如 32 个 query 头共享 8 组 KV）。这是理解"为什么 VLM 的视觉 token 开销大"的前置知识。

#### 1.1.7 训练目标：next-token 预测

对 LLM 而言，训练目标就是自回归的 next-token 预测：给定前文 $x_{<t}$，最大化下一个 token $x_t$ 的条件概率，损失是交叉熵：

$$
\mathcal{L}=-\sum_t \log P(x_t \mid x_{<t})
$$

这一句"预测下一个词"看似简单，却是后续所有能力——语法、世界知识、推理、代码、工具使用——的涌现来源。理解了这一点，就能理解后面 VLM 为什么只是"在词表之外，再塞进一堆视觉 token"。生成时用 temperature 缩放 logits 控制随机性，再配合 top-k / top-p 采样截断长尾。

### 1.2 从纯文本到多模态：VLM 的通用范式

一个 VLM 本质上就是把"视觉"这一新的模态接入 LLM 的 next-token 预测框架。做法高度统一，几乎所有 VLM 都遵循三段式结构：视觉编码器（Vision Encoder）→ 连接器（Projector）→ 大语言模型（LLM）。这个范式的精妙之处在于：它没有改动 LLM 本身，而是把视觉"翻译"成 LLM 已经会处理的 token 语言。下面逐个组件拆开。

#### 1.2.1 视觉编码器：ViT 与它的四条强化路线

视觉编码器通常是一个 ViT（Vision Transformer）：把图像切成 $16\times16$ 的 patch，每个 patch 展平后线性投影成一个 token，再走标准的 Transformer 编码器。一张 $448\times448$ 的图像会变成 $28\times28=784$ 个 patch token。

ViT 本身用图像分类训练，但 VLM 里几乎都换成经过某种大规模预训练强化的版本。这里值得注意的是：**分类它们的关键轴不是"用了什么自监督信号"，而是"输出特征是否已经落在语言空间附近"**——后者直接决定连接器要做多重的活、模态对齐阶段要训多久。按这条轴，主流选项分成四条路线：

**（1）对比对齐系：特征原生贴着语言空间。**

- **CLIP ViT**：OpenAI 用 4 亿图文对做对比学习，让图像编码与对应文本编码在共享空间里"拉近"、与无关文本"推远"。CLIP 的视觉特征天然与语言空间对齐，成了 VLM 视觉编码器的默认起点（LLaVA 就用它）。
- **SigLIP / SigLIP 2**：SigLIP 把 CLIP 的 softmax 对比损失换成逐样本的 sigmoid 二分类损失，训练更稳、可扩展到更大 batch。而 **SigLIP 2** 已不只是"换个对比损失"——它把 caption 生成损失、自蒸馏、掩码预测、在线数据筛选缝进同一个配方，本身就是"对比 + 自监督"的混合体，提供 ViT-B/L/So400m/g 四档（86M / 303M / 400M / 1B）并支持多分辨率与原生宽高比。它是 Qwen3-VL 的选择。

**（2）纯自监督系：空间定位强，但语言对齐要另外补。** 代表是 **DINOv3**：自监督自蒸馏，不用任何文本。它的关键贡献是 **Gram anchoring**，解决了长训练调度下 dense feature map 逐渐退化这个已知但久未解决的问题；与文本的对齐只是 post-hoc 附加的能力。这类特征在分割、深度、对应点匹配等 dense/几何任务上明显强于 CLIP 系，代价是接入 LLM 时连接器（或额外的对齐阶段）要承担更多工作。

**（3）蒸馏效率系：拿部署成本换。** 代表是 **EUPE（Efficient Universal Perception Encoder）**：面向边缘设备的算力约束，用一个小编码器顶替多个领域专家。做法上不是直接从多教师缩小，而是先把多个领域专家蒸馏进一个大的 proxy teacher、再从这个单一教师缩下来。它与 DINOv3 同属蒸馏路线，区别在于 DINOv3 是自蒸馏（自己教自己）、EUPE 是异构多教师蒸馏，且卖点是 size/latency 而非表征上限。

**（4）时空系：唯一带时序建模的一路。** 代表是 **V-JEPA 2.1**：联合嵌入预测（Joint-Embedding Predictive Architecture），在**潜空间**做掩码预测而非重建像素。2.1 相对 V-JEPA 2 的改动是让可见 token 也参与训练信号、跨中间层做层级化深监督、并用多模态 tokenizer 统一图像与视频训练，从而得到"空间有结构、语义连贯、时间一致"的 dense 特征。它是视频 VLM 与 VLA 的自然选择。这条路线与 vault 里 [[视觉世界模型 - Survey]]、[[LeJEPA - 世界模型线性可识别性理论]] 讨论的表征学习主线是同一脉。

四者放在一起对比：

| 编码器 | 训练信号 | 语言对齐 | 对 VLM 的含义 |
|---|---|---|---|
| CLIP ViT | 图文对比（softmax） | 原生 | 一层 MLP 就能接，LLaVA 路线 |
| SigLIP 2 | sigmoid 对比 + caption + 自蒸馏 + 掩码 | 原生 | 同上，且 dense 任务更强 |
| DINOv3 | 自监督自蒸馏（Gram anchoring） | 无（可 post-hoc 加） | 空间定位/dense 更好，需连接器或额外阶段补对齐 |
| V-JEPA 2.1 | 潜空间掩码预测 + 层级深监督 | 无 | 唯一带时序，视频 VLM / VLA 的选择 |
| EUPE | 多教师 → proxy teacher 蒸馏 | 无 | 端侧/小模型场景，用 size 换部署 |

这张表引出一条对理解 VLM 能力边界很关键的结论：**CLIP 系"语言对齐好但空间定位弱"，DINO 系恰好反过来。** 这正是第 6 节局限里"空间推理、精细定位偏弱"那条的技术根源之一。也因此，近年不少 VLM 干脆**并联多个编码器**——把 CLIP/SigLIP 特征与 DINO 特征在通道维拼接后一起喂给连接器，拿对比学习的语义对齐加自监督的几何定位，各取一半。

视觉编码器负责"看懂"，它输出的 patch token 序列，就是后面要交给 LLM 的"视觉语言"。

#### 1.2.2 连接器：视觉与语言的"翻译层"

连接器把视觉特征投影到 LLM 的词嵌入空间，是 VLM 最灵活、也是各家差异最大的部件。常见方案：

- 线性层 / 两层 MLP：最简单。LLaVA-1.0 用单层线性，LLaVA-1.5 换成两层 MLP。便宜、易训，证明"下限可以很低"。
- Q-Former：BLIP-2 提出，用一组可学习的 query 通过交叉注意力从视觉特征里"抽取"固定数量的 token，比朴素投影多了信息筛选。
- Perceiver Resampler：Flamingo 提出，与 Q-Former 类似，用固定数量的 latent query 把可变长度的视觉特征压缩成固定长度。
- pixel-unshuffle / 卷积下采样：把空间维度的信息"折叠"进通道维度，再线性投影，能在不丢太多细节的前提下大幅减少 token 数（Qwen-VL 系列常用）。

#### 1.2.3 视觉 token 爆炸与压缩

这里要直面一个工程痛点：视觉 token 太多了。文本一个 token 是一个词，而一张图轻轻松松就是几百上千个 patch token，视频更是几十帧的叠加。token 越多，注意力计算量平方增长、KV Cache 线性膨胀，推理成本飙升。所以 VLM 的竞争点之一就是"怎么用更少的 token 保留更多视觉信息"：池化、卷积下采样、Q-Former 式抽取、乃至直接把视觉信息压缩进固定数量的 soft token（如 Qwen 的视觉压缩）都是这一思路的体现。这也是为什么连接器的设计（1.2.2）如此重要——它不只是"对齐"，更是"压缩"。

#### 1.2.4 两阶段训练范式

VLM 训练一般分两步。第一步模态对齐：冻结 LLM（有时连同视觉编码器），只训练连接器，用大规模图文对让视觉特征"对齐"到 LLM 的词嵌入空间，让 LLM 能"读懂"视觉 token。第二步指令微调：放开更多参数，用图文对话数据让模型学会"看图回答问题"。这个两阶段范式是 LLaVA 确立的，后续几乎成为行业默认。图 3 展示了这一数据流与三大模型族的对比。

---

## 2. 大模型训练全流程：PT → CPT → Mid-train → SFT → RL → OPD

现代大模型的训练早已不是"预训练 + 微调"两步这么简单，而是演化为一条分工明确的多阶段流水线。每一阶段解决一类特定的能力问题，阶段之间有严格的先后依赖。图 2 把六个阶段串成了完整链条。

![大模型训练全流程](assets/多模态大模型架构与训练全流程%20-%20Survey/figures/02-training-pipeline.png)

*图 2：大模型训练全流程（PT → CPT → Mid-train → SFT → RL → OPD）。每个阶段上方是训练数据与目标，下方是该阶段产出的能力。整体遵循"知识底座 → 领域专精 → 能力打磨 → 指令对齐 → 偏好对齐 → 蒸馏提纯"的能力层次递进。来源：自绘。*

### 2.1 Pre-training（预训练）

预训练是整个体系的底座：在海量无标注文本（数万亿 token）上，用自回归 next-token 预测 + 交叉熵损失，让模型学到通用语法、世界知识、常识与初步的推理模式。这一阶段最贵、最慢，决定了模型的"智力上限"。

值得展开的是数据工程——预训练的质量很大程度上由数据决定。原始语料要经历：去重（文档级与近重复段落级，去掉互联网上大量复制的文本）、质量过滤（用分类器或启发式规则剔除低质、乱码、广告内容）、毒性过滤（去有害内容）、配比（网页、书籍、代码、学术论文、百科按经验比例混合）。训练侧则有一套标准做法：warmup 把学习率从 0 线性升到峰值，再用 cosine 衰减到接近 0；用 BF16/FP16 混合精度 + 梯度缩放；大批次配合梯度累积。BPE/SentencePiece 分词器把文本切成 subword，词表大小（几万到十几万）直接影响"每个 token 的信息密度"。

### 2.2 CPT（Continued Pre-Training，继续预训练）

CPT 是在预训练基座之上，用领域/专业语料继续做 next-token 预测，把模型"补课"成某个领域的专家。典型场景：通用模型要变成法律助手、医疗助手、代码专家，或者要更新知识截止日期（把 2024 年之后的知识灌进去）。

<span style="color: red">CPT 与预训练用的是同一种损失（交叉墒损失）</span>，真正的难点在灾难性遗忘（Catastrophic Forgetting）——训练新领域知识时把旧能力冲掉。应对手段通常是三条：
- 一是**小学习率**，用远低于预训练的学习率做增量更新；
- 二是**数据配比**，在领域语料里混入一定比例的通用语料"锚定"旧能力；
- 三是**经验回放**，把预训练阶段有代表性的样本拿出来一起训。CPT 的目标是"专精而不失底"。

### 2.3 Mid-train（中期训练）

Mid-train 是 CPT 之后、SFT 之前的能力打磨阶段，用**高质量长文、代码、数学、推理数据**去扩展上下文长度、打磨专项技能。它的核心目标不是注入新知识，而是把已有能力"磨锋利"。这一阶段近年才被正式命名和重视，被视为"从会到精"的关键跳板。

两个值得展开的技术点。一是**长上下文扩展**：预训练通常在 4k/8k 上下文上做，要在推理时支持 128k 甚至更长，需要做位置编码外推——RoPE 的基频调整、YaRN（结合缩放与频率插值）等都是这里的标准工具，Mid-train 阶段用长文档数据配合这些技术做微调。二是**合成数据与课程学习**：用更强的模型生成难题与分步解答，让目标模型"刷题"，并按难度从易到难编排（课程学习）。代码、数学、多步推理是这一阶段的三大主战场。

### 2.4 SFT（Supervised Fine-Tuning，监督微调）

SFT 把模型从"会续写"变成"会对话、会听指令"。数据是精心构造的指令-回答对，损失仍然是交叉熵，但**只对回答部分的 token 计算损失**——这是 SFT 与预训练在实现上最关键的差异。用代码表达最清楚：

```python
def sft_loss(logits, input_ids, labels):
    # labels 里，指令部分（prompt）的位置被置为 -100，
    # 只有回答（response）部分的 token 才参与 loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,       # 忽略 -100（指令部分）
    )
    return loss
```

这条 `ignore_index=-100` 是 SFT 的"灵魂"：它保证模型只在"回答该说什么"上被优化，而不是在"如何复述指令"上。SFT 数据要套用统一的 **chat template**（如 `<|im_start|>user...<|im_end|><|im_start|>assistant...`），让模型学会对话格式、遵循指令、拒绝不当请求。但 SFT 有个天花板：它的回答质量受限于训练数据分布，模型只是"模仿"数据里的回答，无法自行判断"什么样的回答更好"——这正是下一阶段 RL 要解决的。

### 2.5 RL（Reinforcement Learning，强化学习对齐）

RL 阶段解决"对齐"问题：让模型不仅会回答，而且回答符合人类偏好——有用、无害、诚实。最经典的路线是 RLHF（Reinforcement Learning from Human Feedback），由三个组件构成：SFT 模型作为初始策略 $\pi$、一个用人类偏好排序训练出来的**奖励模型** $r$、以及用 PPO 做策略优化的循环。

**PPO 的目标**是最大化奖励，同时用 KL 散度约束新策略不要偏离参考模型太远（防止"奖励黑客"——模型钻奖励模型的空子输出高分但无意义的内容）：

$$
\max_{\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}\big[r(x,y)\big]-\beta\,\text{KL}\big(\pi_\theta(y|x)\,\|\,\pi_{\text{ref}}(y|x)\big)
$$

PPO 的具体优化用裁剪（clipped）的 surrogate 目标，防止单步更新幅度过大：

$$
\mathcal{L}^{\text{CLIP}}(\theta)=\mathbb{E}_t\big[\min\big(r_t(\theta)A_t,\ \text{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)A_t\big)\big],\quad r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

其中 $A_t$ 是优势函数（advantage，衡量某个动作比平均好多少），通常由 GAE 从价值模型估计。PPO 需要同时维护策略模型、参考模型、奖励模型、价值模型四个网络，训练复杂且不稳定。于是出现了一批免 RM、免 critic 的轻量替代：

- **DPO（Direct Preference Optimization）**：把 RLHF 的"先训 RM 再 PPO"两步合为一步。它从 RLHF 的最优解形式反推出隐式奖励 $r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，直接在 chosen/rejected 偏好对上做优化，不需要显式奖励模型。损失为：

$$
\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\big[\log\sigma\big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\big]
$$

  其中 $y_w$ 是优选回答、$y_l$ 是劣选回答。DPO 实现简单、稳定，但理论上不如 PPO 贴近"在线采样"。

- **GRPO（Group Relative Policy Optimization）**：DeepSeek-R1 采用的方法，去掉 critic（价值模型）。它对同一个 prompt 采样一组 $G$ 个回答，用**组内相对得分**做 baseline，优势直接用标准化后的奖励：

$$
A_i=\frac{r_i-\text{mean}(r_{1:G})}{\text{std}(r_{1:G})}
$$

  因为没有价值模型，GRPO 大幅降低了显存开销，是当前开源 RL 最常用的算法之一。

图 5 展示了这三条对齐路线的关系。

![RLHF 对齐流程](assets/多模态大模型架构与训练全流程%20-%20Survey/figures/05-rlhf-pipeline.png)

*图 5：对齐流程。左侧是传统 RLHF 三件套（SFT + RM + PPO 循环），PPO 循环内含"采样 rollout → RM 打分 + KL 惩罚 → 策略梯度更新 π"的闭环，参考模型 π_ref 提供 KL 正则防止偏离太远；右下角是免 RM 的 DPO 与免 critic 的 GRPO 两种轻量替代。来源：自绘。*

### 2.6 OPD（On-Policy Distillation，在策略蒸馏）

OPD 是训练流水线最新的一环，也是这条链里最容易被误解的术语。它的正式名称是 **On-Policy Distillation（在策略蒸馏）**，核心思想是：用一个更强的**教师模型**来训练一个更小的**学生模型**，但关键在"在策略"三个字——学生模型在**自己的采样轨迹（rollout）**上做生成，教师模型对这条轨迹提供 **token 级的稠密 log-prob 监督**，学生最小化与教师分布之间的散度。

为什么强调"在策略"？这里有一个重要的理论动机。传统知识蒸馏是**离线**的：教师提前生成一批静态数据，学生在这批数据上训练。问题在于，学生训练时看到的是教师分布、推断时却面对自己的分布，二者不一致会产生 **exposure bias（曝光偏差）**——误差会在自回归生成中逐 token 累积放大，理论分析表明偏差量级约 $O(\varepsilon T^2)$（$\varepsilon$ 是单步误差，$T$ 是序列长度）。而 OPD 让学生在**自己的** rollout 上对齐教师，误差被降到约 $O(\varepsilon T)$，几乎线性而非平方增长。通俗地说：离线蒸馏是"背教师的答案"，OPD 是"让学生自己答题、教师逐题批改"，后者才能真正逼近乃至超越教师。

从优化目标看，OPD 本质上是最小化学生分布与教师分布之间的 **reverse KL**（即 $\text{KL}(\pi_\theta\|\pi_{\text{teacher}})$，用学生分布做采样、在教师 log-prob 下取期望），因为 reverse KL 有 mode-seeking（找主峰而非平均）的特性，更适合"复现教师最擅长的回答方式"。OPD 已被 Qwen 3.6/3.7、DeepSeek-V4、GLM-5.1 等一线模型采用，作为 RL 之后的"蒸馏提纯"环节。在 veRL 框架里，OPD 有两种具体实现变体：**GKD OPD**（Generalized Knowledge Distillation，广义知识蒸馏）与 **PG OPD**（Policy Gradient OPD，策略梯度式）。它是理解"小模型如何逼近大模型"这一当下产业最关心问题（降本 + 保持质量）的钥匙。

---

## 3. 主流 VLM 架构：LLaVA / Qwen-VL / InternVL

三大 VLM 家族都遵循第 1.2 节的通用范式，但各自的演进路线、连接器设计、位置编码方案差异显著。图 3 把它们放在一起对比。

![VLM 通用架构与三模型对比](assets/多模态大模型架构与训练全流程%20-%20Survey/figures/03-vlm-architecture.png)

*图 3：上排是 VLM 通用数据流（图像分块 → 视觉编码器 ViT → 连接器 Projector → LLM），下排是 LLaVA / Qwen-VL / InternVL 三大模型族在连接器设计、位置编码、视觉编码器与 LLM 四个维度的对比。来源：自绘。*

### 3.1 LLaVA 系列：最简洁的范式开创者

LLaVA（Large Language and Vision Assistant）是"线性连接器 + 冻结 LLM"这一最简范式的开创者，其最大贡献在于**证明了只用极简的投影层，就能让 LLM 获得不错的视觉能力**。它的演进脉络清晰，每一代都在解决一个具体痛点：

- **LLaVA（1.0）**：用单层线性层把 CLIP ViT 的视觉特征投影到 LLM 词嵌入空间，训练分两阶段——先冻结 ViT 与 LLM 只训练线性层做对齐，再指令微调。它确立了两阶段范式。
- **LLaVA-1.5**：把单层线性换成**两层 MLP**，引入更强的数据配比与分辨率，性能大幅跃升，成为后续无数工作的基座。这一步说明连接器的表达能力确实要紧。
- **LLaVA-NeXT（1.6）**：引入 **AnyRes 动态分辨率**。此前模型把图像统一缩放到固定分辨率，小目标、密集文本会被糊掉；AnyRes 把高分辨率图像切成多个子图分别编码，再与整图特征拼接，显著提升 OCR 与细粒度理解。这是"分辨率"成为 VLM 竞争维度的开端。
- **LLaVA-OneVision**：把单图、多图、视频统一进同一个接口，朝"任意视觉输入"的通用助手演进。

LLaVA 的价值在于"做减法"：它告诉我们，VLM 的下限可以很低，一条线性投影就能打通视觉与语言，这让多模态能力得以快速扩散到各种 LLM 基座上。但它的局限也在于此——视觉编码器与 LLM 相对独立，深层次的视觉-语言融合不是它的强项。

### 3.2 Qwen-VL 系列：从任务专用到原生多模态

Qwen-VL 是阿里通义的多模态旗舰，演进节奏快、架构创新密集，每一代都在解决一个明确痛点：

- **Qwen-VL（初代）**：用 MLP 连接器 + 交叉注意力引入视觉信息，但仍是相对独立的视觉分支。
- **Qwen2-VL**：引入 **M-RoPE（Multimodal Rotary Position Embedding）**。这是 1.1.3 里 RoPE 的多维推广——把"一维位置"扩展为"时间（视频帧）+ 高度 + 宽度"三个维度，分别用三套旋转角编码，让视频理解与图像理解统一到同一个位置编码框架下。这是 Qwen-VL 系列最具标志性的创新。
- **Qwen2.5-VL**：把视觉编码器换成**窗口注意力 ViT**，把高分辨率图像拆成窗口做注意力，兼顾细节与计算效率，同时引入更强的 OCR 与文档理解。
- **Qwen3-VL**：全面升级为 **SigLIP-2 视觉编码器 + Interleaved-MRoPE（交织 M-RoPE）+ DeepStack**，并推出 **MoE 235B-A22B** 版本——"235B 总参数、每次激活 22B"，用稀疏激活（MoE）把模型规模推上百亿到千亿量级的同时控制推理成本。这是当下"大而快"的典型范式：参数规模大以保证容量上限，激活参数小以保证推理速度。

Qwen-VL 路线的核心思路是"把多模态做原生"：不是把视觉当外挂，而是从位置编码（M-RoPE）、视觉编码器（窗口注意力 ViT、SigLIP-2）、到主干（MoE）都围绕多模态原生设计。这条路线对视频、高分辨率、长文档等"重视觉"任务尤其友好。

### 3.3 InternVL 系列：原生多模态预训练与可变位置编码

InternVL（上海 AI Lab / OpenGVLab）的差异化在于**训练理念**：它强调"原生多模态预训练"——不把视觉对齐当作 LLM 训练之后的事后补丁，而是在预训练阶段就联合文本与多模态数据做单阶段训练。

- **InternVL（早期）**：ViT-MLP-LLM 结构，并率先提出用大视觉编码器（InternViT，6B）配合 LLM，主张"视觉编码器也要足够大"。
- **InternVL3**：三个关键点——① **原生多模态预训练**：ViT-MLP-LLM 单阶段联合训练，而非"先训 LLM 再对齐"，让视觉与语言在预训练阶段就深度融合；② **V2PE（Variable Visual Position Embedding，可变视觉位置编码）**，支持长多模态上下文；③ 后训练用 **SFT + MPO（Mixed Preference Optimization，混合偏好优化）**。InternVL3-78B 在 MMMU 上取得 72.2 分，刷新开源 MLLM 标杆。

InternVL 的启示在于：视觉与语言的融合越早越好，把多模态能力"焊进"预训练，比事后对齐更稳固。这与 Qwen 的"原生"思路殊途同归，但 InternVL 更强调"预训练阶段就融合"，而 Qwen 更强调"位置编码与主干的原生设计"。

### 3.4 三者的本质对比

抛开细节，三条路线的分野可以概括为一句话：**LLaVA 极简、Qwen-VL 原生、InternVL 预训练融合**。落到具体维度：

| 维度 | LLaVA | Qwen-VL | InternVL |
|---|---|---|---|
| 连接器 | 线性 → 两层 MLP | MLP / 卷积下采样 | ViT-MLP |
| 位置编码 | 常规 RoPE | M-RoPE（三维） | V2PE（可变视觉位置编码） |
| 视觉编码器 | CLIP ViT | 窗口注意力 ViT → SigLIP-2 | InternViT（大视觉编码器） |
| 训练理念 | 事后对齐（极简） | 原生多模态（大而快） | 原生预训练融合（深而稳） |
| 规模代表 | 中小模型 | MoE 235B-A22B | 78B 稠密 |

选择哪条路线，取决于你是要快速起一个基线、还是要工业级原生多模态、还是要理论上的融合上限。

---

## 4. 分布式训练框架：Megatron-LM 与 DeepSpeed

当模型参数量达到百亿、千亿级时，单卡显存（甚至单机多卡）根本装不下，训练必须跨成百上千张 GPU 并行。这里有两套主流方案：**Megatron-LM 的 3D 并行**与 **DeepSpeed 的 ZeRO**。图 4 给出了两者的原理示意。

![分布式并行](assets/多模态大模型架构与训练全流程%20-%20Survey/figures/04-distributed-parallelism.png)

*图 4：大模型分布式训练。上半部分是 3D 并行三剑客——数据并行 DP（每卡持完整模型、梯度 AllReduce 同步）、张量并行 TP（层内矩阵乘法切分、每层仅 2 次 AllReduce）、流水线并行 PP（按层分组、1F1B 微批次调度）；下半部分是 DeepSpeed ZeRO 的三种状态分片等级（ZeRO-1 分片优化器状态、ZeRO-2 再分片梯度、ZeRO-3 再分片模型参数按需 AllGather）。来源：自绘。*

### 4.1 Megatron-LM：3D 并行

NVIDIA 的 Megatron-LM 把并行拆成三个正交的维度，三者可以叠加。

**数据并行（Data Parallelism, DP）**：每张卡持有一份完整的模型副本，各自处理不同的数据切片，反向传播后梯度做 AllReduce 同步。这是最基础的并行，但每张卡都要装下完整模型，无法解决"模型太大装不下"的问题——它只扩展吞吐，不扩展容量。

**张量并行（Tensor Parallelism, TP）**：把单个 Transformer 层内的矩阵乘法在多个 GPU 上切分。以 MLP 为例，把 $W_1$ 按列切到两张卡上（各算一半的中间维度），各自过激活函数，再让 $W_2$ 按行切分（各自乘自己那半），最后 AllReduce 相加。注意力层的 $Q,K,V$ 则按头切分。TP 的通信发生在**层内**，每层前向、反向各 1 次 AllReduce（共 2 次），通信量小、粒度细，适合单机内高速互联（NVLink）。代价是每层都有通信同步点。

**流水线并行（Pipeline Parallelism, PP）**：把模型按层切成若干段，每张卡负责一段，像工厂流水线一样处理微批次（micro-batch）。朴素做法会让 GPU 大量空转（前面的卡干完等后面），所以用 **1F1B（one-forward-one-backward）** 调度——每个 worker 交替做前向和反向，填满流水线。PP 的通信只在"段与段"之间发生，通信量极小，适合跨机（机间带宽低）。代价是**流水线气泡（bubble）**：首尾阶段仍有空闲，气泡率约 $\frac{p-1}{p-1+m}$（$p$ 是段数，$m$ 是微批次数），微批次越多气泡越小。

实践中三者叠加使用：**机内用 TP（高速互联）+ 机间用 PP（减少跨机通信）+ 全局用 DP（扩展吞吐）**。这正是"百亿/千亿参数大规模训练经验"的核心工程直觉：并行策略的选择本质是在**计算、显存、通信**三者之间做权衡——TP 通信密集但省显存，PP 通信少但有气泡，DP 吞吐高但受显存限制。

### 4.2 DeepSpeed：ZeRO 显存优化

微软的 DeepSpeed 走的是另一条路——它不改变数据并行"每卡一份模型"的前提，而是把**冗余的中间状态分片**，把省下来的显存换成更大的模型。ZeRO（Zero Redundancy Optimizer）的洞察是：数据并行里，每张卡存的优化器状态、梯度、参数是**完全冗余**的，把这些冗余分片，显存就能被几乎全部用于模型本身。

以一个 7.5B 参数模型 + Adam 优化器 + fp16 为例，训练时显存的大头是：参数 15GB（fp16）+ 梯度 15GB（fp16）+ Adam 状态 60GB（fp32 的一阶动量、二阶方差，以及 fp32 的参数副本，各 15GB，再加一份）。ZeRO 分三级逐一拆解：

- **ZeRO-1**：分片**优化器状态**（Adam 的 m、v、fp32 参数副本），显存节省约 4 倍；
- **ZeRO-2**：在 ZeRO-1 基础上再分片**梯度**，约 8 倍；
- **ZeRO-3**：再分片**模型参数本身**，每张卡只存自己的参数分片，前向时按需 AllGather 收集，用后即弃。接近线性的显存扩展——理论上能让任意大的模型在任意多卡上跑起来。

ZeRO-3 与 Megatron 的 TP/PP 也常结合使用（ZeRO-3 管显存、TP/PP 管通信），各取所长。

### 4.3 配套技术：混合精度、检查点与混合引擎

把两套框架放一起看，百亿/千亿规模训练还有几个"省显存 / 提吞吐"的标配：

- **混合精度**：用 BF16/FP16 存参数、算前向反向，用 FP32 存优化器状态与参数主副本，配合动态/静态**梯度缩放**防止小梯度下溢。
- **梯度检查点（Gradient Checkpointing）**：不存每层的中间激活，反向时重算，用计算换显存。
- **3D-HybridEngine**：在 RL 场景里，模型一会儿要前向采样（推理）、一会儿要反向更新（训练），如果每次切换都重新切分权重开销极大，3D-HybridEngine 让训练与推理之间**零冗余重分片**，这套东西在 veRL 里被用到极致（见 5.1）。

这些不是零散技巧，而是一套围绕"如何把一块放不下的模型，高效地训练出来"的系统工程。

### 4.4 大规模训练的经验要点

归结起来是几条：**显存是第一约束**（用 ZeRO-3 或 TP+PP 消解）；**通信是第二约束**（梯度同步、参数聚合都要尽量留在机内高速互联上）；**并行策略是动态权衡**（TP 通信密集适合机内，PP 有气泡适合大模型，DP 吞吐高但受显存限制）；**故障恢复**是隐藏成本（千卡训练必须要有 checkpoint 断点续训与故障自动重调度）。这些决定了百亿/千亿模型的训练仍是少数团队的护城河。

---

## 5. 强化学习对齐框架：veRL 与 OpenRLHF

第 2.5 节讲的 RL 对齐算法，要在百亿/千亿规模真正跑起来，离不开专门的训练框架。RL 训练的特殊性在于它是一个**"训练 + 推理"的混合流水线**：既要前向采样 rollout（推理，需要高吞吐的推理引擎），又要反向传播更新策略（训练，需要高效的分布式训练后端），两个环节的硬件需求完全不同，因此框架的核心挑战是**如何编排训练 worker 与推理 worker**。主流方案有两套：字节 Seed 的 veRL 与开源的 OpenRLHF。图 6 对比了两者架构。

![RL 训练框架对比](assets/多模态大模型架构与训练全流程%20-%20Survey/figures/06-rl-frameworks.png)

*图 6：veRL 与 OpenRLHF 架构对比。veRL 采用 HybridFlow 混合控制器，把控制流与计算流解耦，训练 worker 用 Megatron-LM/FSDP、推理 worker 用 vLLM/SGLang，3D-HybridEngine 实现训练/推理零冗余重分片；OpenRLHF 用 Ray 编排多角色 actor，rollout 用 vLLM（约占 80% 时间），训练用 DeepSpeed ZeRO-3 + AutoTP。底部小结：选型分水岭在训练并行后端——超大模型（需 Megatron 张量/流水线并行）选 veRL，偏好 DeepSpeed-ZeRO + 清晰代码快速上手选 OpenRLHF，单机学习用 TRL。来源：自绘。*

### 5.1 veRL：HybridFlow 混合控制器

veRL 是字节跳动 Seed 团队开源的 RL 训练框架，它的核心贡献是 **HybridFlow 混合控制器**：用**一个单一的控制器**同时管理"算法逻辑"与"数据流"，把**控制流与计算流解耦**。这意味着算法工程师可以专注写算法（采样、打分、更新这些逻辑），框架自动把训练负载调度到 Megatron-LM/FSDP 的 worker、把推理负载调度到 vLLM/SGLang 的 rollout worker。

这里的关键是 **3D-HybridEngine**：RLHF 里模型一会儿要前向采样（推理）、一会儿要反向更新（训练），如果每次切换角色都重新切分权重，开销极大。3D-HybridEngine 让模型在两种角色之间切换时**零冗余重分片**，把 4.3 节提到的那套工程优化做到极致，也让 veRL 能稳定跑到千亿级模型的 RL 训练。

veRL 支持的算法非常全：PPO、GRPO、GSPO、DAPO、RLOO，以及第 2.6 节提到的 GKD OPD / PG OPD 两种在策略蒸馏变体。它的定位是"工业级、可扩展到千亿模型"的 RL 基座，是训练超大模型的 RL 环节首选。

### 5.2 OpenRLHF：Ray 编排的清晰实现

OpenRLHF 是另一个流行的开源 RLHF 框架，用 **Ray** 做分布式编排，把 RLHF 的各个角色（策略、参考、奖励、价值模型）拆成 Ray 的多个 actor 分别调度。它的技术栈是 **Ray + vLLM + DeepSpeed**：rollout 用 vLLM（实践中 rollout 往往占据约 80% 的训练时间，是优化重点），训练用 **DeepSpeed ZeRO-3 + AutoTP（自动张量并行）**。

OpenRLHF 的最大优点是**代码清晰、可读性强、易于上手**，适合快速改造和教学研究，支持 PPO、GRPO、REINFORCE++、DPO 等算法。在 70B 以上的规模，Ray 的 actor 模型能很好地解耦各 RLHF 角色。它的代价是 Ray 编排本身的调度开销，以及超大模型（需要 Megatron 张量/流水线并行）时不如 veRL 那样原生支持。

### 5.3 选型与补充：TRL

如果把需求再降一档——单机、中小模型、快速实验——那么 Hugging Face 的 **TRL（Transformer Reinforcement Learning）** 是更轻量的选择。它直接构建在 transformers 生态上，几行代码就能跑 SFT/DPO/GRPO，适合学习与快速验证，但不适合大规模生产。

三者的选型逻辑可以归结为一句话：**训练并行后端是分水岭**。需要 Megatron 张量/流水线并行来训超大模型 → veRL；偏好 DeepSpeed-ZeRO 生态、看重代码清晰和上手速度 → OpenRLHF；单机学习或小模型 → TRL。

---

## 6. 局限与评价

这份综述串起了一条完整的技术栈，但每一环都有尚未解决的问题，值得诚实标注：

- **Transformer 的二次复杂度**：自注意力计算量随序列长度平方增长（$O(n^2)$），长上下文、高分辨率多模态都会撞上这堵墙。FlashAttention、稀疏/线性注意力只是缓解而非根治，多模态场景下视觉 token 的数量压力（1.2.3）让这个问题更尖锐。
- **多阶段训练的成本与脆弱性**：PT → CPT → Mid-train → SFT → RL → OPD 六阶段链条每一步都在累积数据与算力成本，且灾难性遗忘、奖励黑客（reward hacking）、蒸馏过程中的教师偏差等问题贯穿始终。OPD 虽缓解了 exposure bias，但教师模型本身的能力上限仍是学生的天花板——蒸馏只能逼近、很难超越教师。
- **VLM 的"视觉理解深度"存疑**：把图像编码成 token 塞进 LLM，能解决看图说话、OCR、图表理解，但空间推理、精细定位、3D 几何等深层次视觉能力是否真的被 LLM "理解"了，还是只是学会了统计关联，仍是开放问题。一个具体的技术根源见 1.2.1 节：主流 VLM 的视觉编码器多来自对比学习（CLIP/SigLIP），其特征长于语言对齐而短于空间定位，这也是"并联 DINO 系编码器"成为近年常见补救手段的原因。这也正是 3D 多模态 LLM（如 VLM³ 那类工作）试图突破的方向。
- **分布式训练的高门槛**：3D 并行 + ZeRO + RL 混合流水线的工程复杂度极高，调参、故障恢复、通信优化都需要大量经验，这决定了百亿/千亿模型的训练仍是少数团队的护城河。

**总体评价：★★★★☆（4/5）。** 作为一份体系化综述，它把"从 Transformer 到多模态大模型落地"的完整技术栈梳理成了一条可复用的认知主线，覆盖了架构原理、训练全流程、三大 VLM 家族、分布式框架与 RL 对齐框架五个相互咬合的层面，并对每个层面给出了公式、实现差异与工程权衡。扣掉的一星在于：这是一份基于公开资料与既有知识的综述，而非对某篇单一论文的深度精读，个别前沿细节（如 OPD 的具体实现、最新模型的精确架构参数）仍应回到原始论文与官方文档二次核实。

---

*本综述为技术视野的体系化沉淀，可与 `AIReadPaper/20-Surveys/VLM/三维多模态大语言模型架构 - Survey.md`（聚焦 3D LLM 架构）相互参照，二者分别覆盖"通用 VLM 与训练"与"3D 多模态 LLM"两条主线。*
