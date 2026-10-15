---
type: paper-note
status: complete
domain: Foundations
paper: "iBOT: Image BERT Pre-Training with Online Tokenizer"
year: 2022
arxiv: "2111.07832"
doi: null
source: https://arxiv.org/abs/2111.07832
project: https://openreview.net/forum?id=ydopy-e6Dg
code: https://github.com/bytedance/ibot
tags:
  - self-supervised-learning
  - masked-image-modeling
  - online-tokenizer
  - knowledge-distillation
  - vision-transformer
  - dense-features
created: 2026-10-15
updated: 2026-10-15
---

# iBOT：在线分词器驱动的图像 BERT 预训练

## 0. 基本信息

| 项目 | 内容 |
|---|---|
| 正式名称 | **iBOT: Image BERT Pre-Training with Online Tokenizer** |
| 作者 | Jinghao Zhou、Chen Wei、Huiyu Wang、Wei Shen、Cihang Xie、Alan Yuille、Tao Kong |
| 机构 | ByteDance、Johns Hopkins University、Shanghai Jiao Tong University、UC Santa Cruz |
| 会议 | ICLR 2022，Poster |
| arXiv | [2111.07832](https://arxiv.org/abs/2111.07832) |
| OpenReview | [ydopy-e6Dg](https://openreview.net/forum?id=ydopy-e6Dg) |
| 代码 | [bytedance/ibot](https://github.com/bytedance/ibot) |
| 本地原文 | `assets/iBOT - 在线分词器驱动的图像BERT预训练/sources/paper.pdf` |
| 核心关键词 | MIM、online tokenizer、self-distillation、EMA teacher、patch token、CLS token |

> [!summary] 一句话结论
> iBOT 把 masked image modeling 改写成 teacher-student 自蒸馏：student 从带遮挡的图像预测 patch 级概率分布，EMA teacher 以未遮挡图像实时生成软目标，同时用跨视图 CLS 自蒸馏注入全局语义；这一单阶段“在线分词”机制让 ViT 同时获得强分类与强局部表征，并成为 DINOv2、DINOv3 配方中的 patch-level objective。

## 1. 研究背景与问题定义

### 1.1 从 BERT 的 MLM 到视觉 MIM

BERT 的 masked language modeling（MLM）有两个关键步骤：

1. tokenizer 先把文本切成语义相对稳定的离散 token；
2. 模型根据上下文预测被遮挡 token 的类别。

把这一范式直接搬到图像并不容易。

图像 patch 是连续信号，同一个局部区域可能同时表达颜色、纹理、形状、部件和对象语义。

因此视觉 MIM 的难点不只是“遮住什么”，而是：**被遮住 patch 应该预测什么目标。**

早期路线主要有三类目标：

- 原始像素或低阶统计量；
- 离线聚类得到的离散 id；
- 预训练 dVAE 产生的视觉 token。

像素回归容易把容量花在高频细节上。

离线 tokenizer 则引入额外训练阶段、固定领域偏置和不可共同适配的问题。

BEiT 使用预训练 DALL-E dVAE tokenizer，把 MIM 转成离散分类，这是重要进展。

但 iBOT 作者观察到，BEiT 的离线 token 更偏向颜色和纹理，patch 表征中的高层局部语义不足。

### 1.2 论文要回答的核心问题

论文的核心问题可以写成：

> 能否让产生 MIM 目标的视觉 tokenizer 与目标网络一起学习，并且让 tokenizer 在训练过程中逐渐获得语义，而不是预先固定？

这要求同时解决三个问题：

1. **目标生成**：未标注图像上，谁来给被遮挡 patch 产生监督？
2. **语义注入**：只做局部恢复时，如何避免目标停留在颜色和纹理？
3. **稳定训练**：teacher 与 student 同源，如何避免两者一起坍塌到常数输出？

### 1.3 核心贡献

1. **把 MIM 解释为知识蒸馏。**
   BEiT 的离线 dVAE 可视为固定 teacher；iBOT 则把 teacher 替换为模型自身的 EMA 历史。

2. **提出 online tokenizer。**
   teacher 对未遮挡图像的 patch token 输出软分布，在线充当 visual tokenizer，不再需要额外 tokenizer 预训练阶段。

3. **联合全局与局部自蒸馏。**
   CLS token 做跨视图蒸馏以获得全局语义，patch token 做同视图 masked prediction 以学习局部结构。

4. **展示 patch 级语义涌现。**
   模型自动形成车灯、狗耳、鸟翼等局部语义簇，而不仅是颜色或纹理簇。

5. **验证通用迁移。**
   iBOT 在分类、低样本学习、无监督聚类、检测、实例分割和语义分割上均取得强结果。

## 2. 任务定义：MIM 的输入与输出

### 2.1 Patch 序列与遮挡

给定图像 patch token 序列：

$$
x=\{x_i\}_{i=1}^{N},
$$

其中 $N$ 是 patch 数。

对于 224×224 输入和 16×16 patch，$N=14\times14=196$。

采样二值 mask：

$$
m\in\{0,1\}^{N},
$$

$m_i=1$ 表示第 $i$ 个 patch 需要预测。

student 的输入把这些位置替换为可学习的 mask embedding：

$$
\hat{x}_i=(1-m_i)x_i+m_i e_{[MASK]}.
$$

因此，**student 输入**是带 `[MASK]` token 的 patch 序列。

teacher 不接收 mask，看到同一视图的完整 patch 序列。

### 2.2 预测对象不是像素，也不是固定 token id

student 对每个 masked patch 输出 $K$ 维 logits，经 softmax 得到概率分布。

teacher 对未遮挡位置输出同维度概率分布，作为软标签。

论文默认 projection head 输出维度为：

$$
K=8192.
$$

这里的 8192 个维度不是人工定义类别。

它们更像在线演化的 prototype / latent token vocabulary。

因此 iBOT 的“tokenizer”不是把 patch 永久映射到一个固定 id，而是输出一个随训练演化的软 token 分布。

### 2.3 最终训练产物

预训练结束后，真正保留的是 ViT 或 Swin backbone。

它可以输出：

- CLS token：图像级全局表示；
- patch tokens：空间排列的局部表示。

projection head、mask token 和 teacher 主要服务于预训练。

下游任务通常使用 teacher backbone，因为官方仓库报告 teacher 一般优于 student。

## 3. 方法总览

![](assets/iBOT%20-%20在线分词器驱动的图像BERT预训练/figures/01-online-tokenizer-framework.png)

> **图 1｜在线 tokenizer 与双重自蒸馏。** Figure: 原论文 Figure 3；Page: 4；URL: https://arxiv.org/pdf/2111.07832 。两种增强视图 $u,v$ 进入 student 与 EMA teacher；student 看带 mask 的视图，teacher 看完整视图。CLS 使用跨视图损失，patch 使用同视图 masked prediction，teacher backbone 与 patch head 合起来就是 online tokenizer。

### 3.1 一张图像产生两类视图

对无标签图像 $x\sim\mathcal I$ 采样两次增强：

$$
u=t_1(x),\qquad v=t_2(x).
$$

随后在 global crop 上构造 masked views：

$$
\hat u=\operatorname{Mask}(u),\qquad \hat v=\operatorname{Mask}(v).
$$

student 网络 $g_s=h_s\circ f_s$ 接收 $\hat u,\hat v$。

teacher 网络 $g_t=h_t\circ f_t$ 接收未遮挡的 $u,v$。

### 3.2 两个目标分别回答“是什么”和“在哪里”

iBOT 同时优化：

$$
\mathcal L=\mathcal L_{[CLS]}+\mathcal L_{MIM}.
$$

两项默认直接相加，不额外缩放。

- $\mathcal L_{[CLS]}$：跨视图 CLS 自蒸馏，学习图像级语义与增强不变性；
- $\mathcal L_{MIM}$：同视图 patch 自蒸馏，预测被遮挡位置的 teacher token 分布。

“跨视图”和“同视图”是理解 iBOT 的关键。

CLS 要求不同 crop 共享对象级语义，所以做 cross-view matching。

patch 有明确空间位置，必须让 student 在位置 $i$ 预测 teacher 同一视图、同一位置 $i$ 的目标，因此做 in-view matching。

### 3.3 端到端信息流

以视图 $u$ 为例：

1. 完整 $u$ 输入 teacher；
2. 遮挡后的 $\hat u$ 输入 student；
3. teacher 在每个 patch 位置产生软 token distribution；
4. student 只在 $m_i=1$ 的位置承担 MIM 交叉熵；
5. 另一条分支用 student 的 $\hat u$ CLS 预测 teacher 的 $v$ CLS；
6. student 反向传播更新；
7. teacher 由 student 参数做 EMA 更新，不接收梯度。

这个设计同时具备：

- BERT 式遮挡预测；
- DINO 式跨视图自蒸馏；
- BYOL/MoCo 式 momentum teacher；
- 在线软聚类式视觉 vocabulary。

## 4. Online tokenizer：为什么它能工作

### 4.1 从固定 tokenizer 到在线 teacher

BEiT 的 MIM 目标可写为：

$$
-\sum_{i=1}^{N}m_i\,P_{\phi}(x_i)^\top\log P_{\theta}(\hat x_i),
$$

$P_{\phi}$ 是固定 dVAE tokenizer，$P_{\theta}$ 是目标模型。

从形式上看，这已经是 knowledge distillation。

iBOT 的关键替换是：

$$
\phi\quad\longrightarrow\quad\theta',
$$

即把固定 tokenizer 参数换成 teacher 的 EMA 参数。

teacher 的 backbone $f_t$ 与 patch head $h_t^{patch}$ 共同组成：

$$
\operatorname{Tokenizer}_{online}=h_t^{patch}\circ f_t.
$$

### 4.2 “在线”并不等于 teacher 与 student 同步

teacher 不是 student 的即时副本。

其更新为：

$$
\theta_t\leftarrow\lambda\theta_t+(1-\lambda)\theta_s.
$$

其中 $\lambda$ 是较大的 momentum coefficient。

因此 teacher 是 student 历史轨迹的低通滤波。

它比当前 student 更平滑，为 masked prediction 提供相对稳定的目标。

teacher 输出会 `detach`，所以梯度只经过 student。

这构造了必要的不对称性：

- student 快速学习；
- teacher 缓慢追踪；
- 目标随数据和模型共同演化，但不会逐步反传追逐自己。

### 4.3 为什么使用软 token distribution

文本 token 通常有明确离散含义。

图像 patch 则高度歧义：一块狗耳 patch 同时可能属于“棕色”“毛发”“耳朵”“狗”。

强制 one-hot 会过早丢弃这种多义性。

iBOT 使用 teacher softmax 后的完整分布：

$$
p_t(i)=\operatorname{softmax}((z_t(i)-C')/\tau_t').
$$

student 分布为：

$$
p_s(i)=\operatorname{softmax}(z_s(i)/\tau_s').
$$

软标签保留 prototype 之间的相对相似性。

Table 18 说明 soft distribution 与较低 teacher temperature 很关键。

在无 centering 的设置下：

- 普通 softmax：49.8 k-NN / 63.5 linear；
- hardmax：64.8 / 71.9；
- sharpened softmax：69.4 / 73.9。

这说明“软”本身不是全部，**软分布还必须足够尖锐，才能成为有信息量的目标。**

### 4.4 为什么 CLS 与 patch head 共享

论文默认：

$$
h_s^{[CLS]}=h_s^{patch},\qquad h_t^{[CLS]}=h_t^{patch}.
$$

CLS 的跨视图目标更容易形成对象级语义。

共享 head 相当于把这套 prototype vocabulary 复用于 patch token。

于是 patch 的类别空间不只是局部颜色码本，而被全局语义目标持续塑形。

Table 9 中，完整共享 head 的 linear 为 74.2，取消共享为 73.8。

提升不大，但方向稳定。

更重要的是，它给出了一个机制解释：全局语义通过参数共享进入局部 tokenizer。

## 5. 两类蒸馏损失

### 5.1 Patch token 的 MIM 损失

teacher 对完整视图 $u$ 输出：

$$
u_t^{patch}=P_{\theta_t}^{patch}(u).
$$

student 对 masked view 输出：

$$
\hat u_s^{patch}=P_{\theta_s}^{patch}(\hat u).
$$

损失只在被遮挡位置计算：

$$
\mathcal L_{MIM}^{u}
=-\sum_{i=1}^{N}m_i\,
P_{\theta_t}^{patch}(u_i)^\top
\log P_{\theta_s}^{patch}(\hat u_i).
$$

视图 $v$ 同理，最终对两项求平均。

这里有四个要点：

1. teacher 看见真实 patch，student 看见 `[MASK]`；
2. 监督位置严格对齐；
3. 目标是概率分布，不是 RGB；
4. 未遮挡位置不进入主要 MIM loss。

student 只能利用未遮挡上下文、位置编码和图像内结构来恢复目标。

因此 patch token 被迫同时包含局部语义与上下文关系。

### 5.2 CLS token 的跨视图损失

teacher 的 $v$ 视图监督 student 的 $\hat u$ 视图：

$$
\mathcal L_{[CLS]}^{u\leftarrow v}
=-P_{\theta_t}^{[CLS]}(v)^\top
\log P_{\theta_s}^{[CLS]}(\hat u).
$$

再交换 $u,v$ 对称化：

$$
\mathcal L_{[CLS]}
=\frac{1}{2}\mathcal L_{[CLS]}^{u\leftarrow v}
+\frac{1}{2}\mathcal L_{[CLS]}^{v\leftarrow u}.
$$

与 DINO 相比，student 的 CLS 来自 masked crop。

这让全局表示也必须对局部缺失保持稳健。

### 5.3 两项损失为何缺一不可

只有 CLS loss，本质接近 DINO，擅长全局判别，但没有直接要求 patch 恢复。

只有 MIM loss，online targets 可能稳定在低层统计，甚至缺乏可用于线性分类的高层语义。

两者联合后：

- CLS 提供全局语义组织；
- MIM 把这种组织细化到空间位置；
- 共享 head 让两种目标使用相同的 prototype 空间。

这不是简单的两个 task 相加。

它是一个“全局目标训练 tokenizer，tokenizer 再监督局部恢复”的闭环。

## 6. 遮挡策略与 multi-crop 稳定性

### 6.1 论文默认遮挡调度

主论文采用 block-wise masking。

在 multi-crop 配方中，每个样本的 prediction ratio：

- 以 0.5 概率取 $r=0$；
- 以 0.5 概率从 $[0.1,0.5]$ 均匀采样。

$r=0$ 时不执行 MIM，该样本退化为 DINO 式全局自蒸馏。

$r>0$ 时两个 global crops 都执行 MIM。

论文也把默认写成：

$$
r=0.3\pm0.2.
$$

### 6.2 为什么要允许零遮挡

直接在所有 global crops 上持续遮挡，而 local crops 不遮挡，会产生分布不匹配：

- student 的 global 分支总看 masked images；
- student 的 local 分支看 natural images；
- teacher 始终看 natural images。

附录 Figure 7–8 显示，这会带来 NMI 曲线突降和训练不稳定。

随机加入 $r=0$ 的 global crops，相当于在训练中不断提供自然图像路径。

它让 DINO 分支与 MIM 分支共享输入分布的交集，从而稳定 multi-crop。

### 6.3 mask ratio 的消融

在 0.05–0.4 范围内，性能对固定 ratio 不太敏感。

给 ratio 增加随机方差通常更好，可视为更强的数据增强。

官方仓库后续又给出一项更新：

- 使用 random masking；
- prediction ratio 提高到 $[0.65,0.75]$；
- ViT-B/16 达到 84.1% ImageNet fine-tuning；
- COCO box AP 达到 51.5。

这说明原论文的 block-wise、较低比例设置不是唯一有效配置。

但仓库更新与论文默认模型不应混成同一行结果。

### 6.4 与 MAE 的高遮挡率差异

MAE 常用 75% random masking，因为 encoder 只处理可见 patch，遮挡越高越能节省计算并迫使全局建模。

iBOT 的 student 仍保留完整序列长度，用 `[MASK]` token 占位；teacher 还要处理完整视图。

因此高遮挡率不会像 MAE 那样直接带来同等 encoder 加速。

两者的 mask ratio 不能脱离计算图直接比较。

## 7. 防坍塌机制

### 7.1 坍塌风险来自哪里

如果 teacher 与 student 都输出同一个常数分布，交叉熵也可能变得缺乏训练信号。

在线 tokenizer 没有外部标签，因此必须自己维持非退化 vocabulary。

iBOT 继承并扩展了 DINO 的防坍塌组合。

### 7.2 Stop-gradient 与 EMA teacher

teacher 分支不反向传播。

EMA 让 teacher 只缓慢追踪 student。

这避免了两个网络在同一步中相互追逐，并为每次更新提供时间上平滑的目标。

EMA 本身并不保证完全不坍塌，但提供了关键的不对称性。

### 7.3 Centering

teacher logits 先减去移动平均 center：

$$
C\leftarrow mC+(1-m)\operatorname{mean}(z_t^{[CLS]}),
$$

patch 另有独立 center：

$$
C'\leftarrow m'C'+(1-m')\operatorname{mean}(z_t^{patch}).
$$

centering 抑制某些 prototype 长期占据全部概率质量。

它针对“所有样本都落到少数维度”的坍塌。

### 7.4 Sharpening

teacher 使用较低温度：

$$
p_t=\operatorname{softmax}((z_t-C)/\tau_t),\qquad \tau_t<\tau_s.
$$

sharpening 防止 teacher 接近均匀分布。

centering 与 sharpening 的作用方向相反但互补：

- centering 防止单一 prototype 垄断；
- sharpening 防止所有 prototype 均匀无差别。

### 7.5 CLS 语义自蒸馏是 patch tokenizer 的稳定器

Table 9 是最关键的证据。

去掉 $\mathcal L_{[CLS]}$ 后：

- k-NN：69.1 → 9.5；
- linear：74.2 → 29.8；
- fine-tuning：81.5 → 79.4。

这说明只做 MIM 并没有完全数值发散，但学到的冻结特征几乎失去全局语义。

fine-tuning 仍有 79.4，表明 backbone 尚保留可适配的低层结构。

因此这里的“坍塌”更准确地说是**语义表征退化**，而非所有权重变成常数。

### 7.6 Soft targets 与共享 head

soft targets 保留视觉语义的模糊性。

共享 head 把 CLS prototype 空间传给 patch。

这两者不是传统意义上的防坍塌正则，却能避免 online tokenizer 退化成纯纹理码本。

## 8. 实验设置

### 8.1 预训练数据与骨干

默认预训练数据是 ImageNet-1K train split。

另外扩展到 ImageNet-22K。

骨干包括：

- ViT-S/16；
- ViT-B/16；
- ViT-L/16；
- Swin-T，window size 7 或 14。

### 8.2 训练日程

ImageNet-1K：

- ViT-S/16：800 epochs；
- ViT-B/16：400 epochs；
- ViT-L/16：250/300 epochs，表格与最终模型设置存在版本差异；
- Swin-T：300 epochs。

ImageNet-22K：

- ViT-B/16：80 epochs；
- ViT-L/16：50 epochs。

优化器为 AdamW。

默认 batch size 为 1024。

学习率按 batch size 线性缩放：

$$
\mathrm{lr}=5\times10^{-4}\times\frac{\mathrm{batch\ size}}{256}.
$$

前 10 epochs warmup。

### 8.3 Multi-crop

强结果使用：

- 2 个 224×224 global crops；
- 10 个 96×96 local crops。

local/global scale 分界默认约为 0.32。

MIM 主要作用于 global crops。

### 8.4 评价协议

论文覆盖：

- k-NN；
- frozen linear probing；
- end-to-end fine-tuning；
- 1% / 10% 标签的低样本学习；
- 无监督图像聚类；
- COCO object detection / instance segmentation；
- ADE20K semantic segmentation；
- 小数据集迁移；
- 鲁棒性、patch pattern、attention map 与 correspondence 分析。

## 9. 分类与迁移结果

### 9.1 ImageNet-1K 冻结特征

| 模型 | 预训练数据 | k-NN | Linear |
|---|---|---:|---:|
| iBOT ViT-S/16 | IN-1K | 75.2 | 77.9 |
| iBOT ViT-B/16 | IN-1K | 77.1 | 79.5 |
| iBOT ViT-L/16 | IN-1K | 78.0 | 81.0 |
| iBOT ViT-L/16 | IN-22K | 72.9 | **82.3** |

与 DINO 相比：

- ViT-S/16 linear：77.9 vs 77.0，+0.9；
- ViT-B/16 linear：79.5 vs 78.2，+1.3。

模型变大后增益更明显，支持 MIM 对容量 scaling 有帮助。

IN-22K 预训练的 k-NN 反而较低，说明 frozen feature 评价对预训练/目标分布匹配很敏感。

### 9.2 ImageNet-1K 微调

| 模型 | Fine-tuning Top-1 |
|---|---:|
| iBOT ViT-S/16 | 82.3 |
| iBOT ViT-B/16 | 84.0 |
| iBOT ViT-L/16 | 84.8 |
| iBOT ViT-L/16，IN-22K，512px | **87.8** |

值得注意的反例：

- IN-1K 上 ViT-L/16，BEiT 为 85.2，iBOT 为 84.8；
- IN-22K 上 ViT-L/16，iBOT 86.6，BEiT 86.0；
- 论文据此判断，大模型 iBOT 更依赖数据规模。

所以不能概括成 iBOT 在每个 fine-tuning setting 都胜过 BEiT。

### 9.3 跨数据集迁移

ViT-B/16 上，iBOT 在 CIFAR10/CIFAR100/iNat18/iNat19/Flowers/Cars 的 Top-1 为：

$$
99.2/92.2/74.6/79.6/98.9/94.3.
$$

相对 DINO，iNaturalist18 为 74.6 vs 72.6，提升 2.0。

这类长尾细粒度数据更能体现局部语义的价值。

## 10. 低样本、无监督聚类与“对象发现”边界

![](assets/iBOT%20-%20在线分词器驱动的图像BERT预训练/figures/03-main-results-lowshot-dense.png)

> **图 2｜低样本、无监督聚类与稠密下游主结果。** Figure: 原论文 Tables 4–6；Page: 6；URL: https://arxiv.org/pdf/2111.07832 。iBOT 在 1%/10% 标签、无监督聚类以及 COCO/ADE20K 上均超过对照，说明收益同时存在于全局和局部尺度。

### 10.1 1% / 10% 标签

端到端低样本微调：

| 方法 | 1% labels | 10% labels |
|---|---:|---:|
| DINO ViT-S/16 | 60.3 | 74.3 |
| **iBOT ViT-S/16** | **61.9** | **75.1** |

iBOT 分别提高 1.6 和 0.8 个点。

附录进一步比较协议：

- frozen backbone + logistic regression：65.9 / 73.4；
- frozen backbone + linear：62.5 / 72.2；
- 从 projection head 第一层端到端微调：61.9 / 75.1。

1% 标签时，冻结特征加 logistic regression 更好。

10% 标签时，端到端微调更好。

这说明“低样本结果”高度依赖 evaluation protocol，不能只看单一数字。

### 10.2 无监督图像聚类

ImageNet-1K 无监督聚类：

| 方法 | ACC | ARI | NMI | FMI |
|---|---:|---:|---:|---:|
| DINO ViT-S/16 | 41.4 | 29.8 | 76.8 | 32.8 |
| **iBOT ViT-S/16** | **43.4** | **32.8** | **78.6** | **35.6** |

论文正文把 78.6 NMI 相对先前方法的提升写作 1.8。

该实验说明 patch MIM 不只改善局部任务，也能反过来增强 CLS 的全局聚类结构。

### 10.3 无监督对象发现：论文没有标准数值

这里必须区分两个概念：

- Table 5 是 **unsupervised image clustering**；
- 无监督对象发现通常报告 VOC/COCO 上的 CorLoc、mIoU 或 object mask quality。

iBOT 论文没有报告 VOC07/VOC12/COCO CorLoc 等标准 object discovery benchmark。

因此不能把 43.4 ACC 或 78.6 NMI 写成“对象发现结果”。

论文提供的是定性证据：

- Figure 6 的 CLS self-attention 能分离前景与背景；
- 不同 attention heads 聚焦车轮、鸟喙等可判别部件；
- Figure 18 相比 DINO 展示更清楚的对象/部件分离；
- Figure 19 展示同类图像间 patch correspondence。

结论应限定为：**iBOT 显示出无监督对象与部件定位的涌现能力，但原论文未用标准 object discovery 指标做定量验证。**

## 11. 检测、实例分割与语义分割

### 11.1 COCO 检测与实例分割

ViT-S/16 + Cascade Mask R-CNN：

- box AP：49.4；
- mask AP：42.6。

监督 ViT-S/16 为 46.2 / 40.1。

ViT-B/16：

| 方法 | Box AP | Mask AP |
|---|---:|---:|
| Supervised | 49.8 | 43.2 |
| BEiT | 50.1 | 43.5 |
| DINO | 50.1 | 43.4 |
| **iBOT** | **51.2** | **44.2** |

相对 DINO，iBOT 提升 1.1 box AP 和 0.8 mask AP。

官方仓库后续 random masking 模型把 box AP 提高到 51.5。

### 11.2 ADE20K 语义分割

ViT-B/16 的 frozen linear head：

- supervised：35.4 mIoU；
- BEiT：27.4；
- DINO：34.5；
- iBOT：38.3。

这个设置最直接测量 patch feature 的线性可分性。

BEiT 的 27.4 显著偏低，支持“离线 dVAE token 偏低层”的判断。

使用 UPerNet 全量微调：

- supervised：46.6；
- BEiT：45.8；
- DINO：46.8；
- iBOT：**50.0**。

iBOT 相对 DINO 提升 3.2 mIoU。

### 11.3 实验解读

分类提升约 1 个点，ADE20K UPerNet 却提升 3.2 mIoU。

这说明 iBOT 的主要价值不是单纯让 CLS 更强，而是让 patch representation 真正可用于 dense prediction。

同时，BEiT 在 fine-tuning 分类上并不弱，却在线性分割上明显弱。

这揭示了一个重要评价原则：

> end-to-end fine-tuning 可以用任务监督重塑特征，不能单独证明预训练 patch token 已经具有局部语义。

## 12. 局部语义可视化与鲁棒性

![](assets/iBOT%20-%20在线分词器驱动的图像BERT预训练/figures/02-local-semantic-patterns.png)

> **图 3｜patch token 的局部语义与纹理 pattern。** Figure: 原论文 Figure 4（上方同时包含 Table 7）；Page: 7；URL: https://arxiv.org/pdf/2111.07832 。左侧聚类对应车灯和狗耳等语义部件，右侧对应条纹、卷曲表面等纹理；它说明 online tokenizer 同时保留高层部件和低层外观，而非只学一种粒度。

### 12.1 Pattern layout 如何构造

作者使用 800-epoch ViT-S/16。

在 ImageNet validation set 中，对 projection head 的每个输出维度搜索置信度最高的 patch。

每个中心 patch 为 16×16，并显示周围 5×5 context。

如果同一维度的 top patches 反复落在狗耳、车灯等部位，就说明该 latent token 形成了可解释局部语义。

### 12.2 高层与低层模式共存

Figure 4 展示：

- 车灯；
- 狗耳；
- 条纹表面；
- 卷曲表面。

Appendix Figure 13 还展示头发、肩臂、肘部、狗鼻、鸟翼、蜻蜓翼、车窗和建筑窗户。

Appendix Figure 14 展示颜色、曲率、直线、动物皮肤纹理。

好的视觉 tokenizer 不应只追求“全部语义化”。

纹理和颜色对匹配、分割边界、细粒度识别仍然有用。

iBOT 的优势是多粒度共存。

### 12.3 Part-wise linear probing

作者按 CLS attention score 排序 patch tokens，平均 top-k patch 后做线性分类。

标准 CLS linear 下，iBOT 相对 DINO只高 0.9。

但直接使用 patch：

- top-56 patch 时，iBOT 高 5.9；
- 只用 top-1 patch 时，iBOT 高 17.9。

随着可用 patch 变少，差距变大。

这说明 iBOT 把更多类别信息压到单个关键局部中。

### 12.4 鲁棒性

ViT-S/16 线性评估中：

- ImageNet-A：iBOT 13.8，DINO 12.3；
- ImageNet-C mCE：iBOT 48.1，DINO 51.7，越低越好；
- salient/non-salient patch dropping 下，iBOT 的准确率下降更小；
- 多数 background-change 设置下，iBOT 更稳健。

MIM 训练不断要求模型从不完整输入恢复语义目标。

因此它对遮挡、局部缺失和背景扰动更不敏感，是符合机制预期的结果。

## 13. 关键消融

![](assets/iBOT%20-%20在线分词器驱动的图像BERT预训练/figures/04-tokenizer-ablation.png)

> **图 4｜语义 tokenizer 的关键消融。** Figure: 原论文 Table 9；Page: 9；URL: https://arxiv.org/pdf/2111.07832 。完整 iBOT 的 69.1/74.2/81.5 显著优于去掉 CLS loss、离线 DALL-E tokenizer 或简单 BEiT+DINO 多任务，证明关键不是“多加一个 loss”，而是在线 teacher 与全局语义共同塑造 patch 目标。

### 13.1 去掉 CLS loss

完整 iBOT：

$$
69.1\;k\text{-NN},\quad74.2\;linear,\quad81.5\;fine\text{-}tune.
$$

去掉 $\mathcal L_{[CLS]}$：

$$
9.5,\quad29.8,\quad79.4.
$$

这是论文最强的因果证据。

MIM 不能自动保证 tokenizer 语义化。

### 13.2 Standalone DINO 作为固定 tokenizer

预训练 DINO 作为 tokenizer：

$$
44.3\;k\text{-NN},\quad60.0\;linear,\quad81.7\;fine\text{-}tune.
$$

它比 BEiT tokenizer 更语义化，但冻结后仍明显弱于 online joint learning 的 69.1 / 74.2。

这说明在线共同适配很重要。

### 13.3 BEiT + DINO 多任务不等于 iBOT

BEiT + DINO：

$$
48.0\;k\text{-NN},\quad62.7\;linear,\quad81.2\;fine\text{-}tune.
$$

完整 iBOT linear 高 11.5，fine-tuning 高 0.3。

如果只是把离线 MIM loss 与 DINO loss 相加，patch targets 仍来自固定 dVAE。

而 iBOT 让 DINO 获得的语义直接进入 tokenizer 本身。

### 13.4 共享 head

不共享 head：69.0 / 73.8 / 81.5。

完整共享：69.1 / 74.2 / 81.5。

增益主要反映在 frozen linear，而非 fine-tuning。

这符合“共享 head 改善预训练表征组织，但下游微调可补偿”的解释。

### 13.5 MIM 与 dense self-distillation

附录比较：

| 方法 | k-NN | Linear |
|---|---:|---:|
| DINO | 67.9 | 72.5 |
| DINO + position match | 67.1 | 72.5 |
| DINO + feature match | 68.5 | 73.4 |
| **iBOT** | **69.1** | **74.2** |

只在不同 views 中做 patch matching 需要处理位置和对应关系。

MIM 的同视图、同位置监督绕开了跨视图 patch correspondence 的歧义。

### 13.6 Loss ratio 与输出维度

$\mathcal L_{[CLS]}:\mathcal L_{MIM}$ 直接 1:1 相加时 linear 为 74.2。

改变 MIM 权重到 0.5 或 2 没有更好。

输出维度：

- 4096：74.5 linear；
- 8192：74.2；
- 16384：74.0。

8192 并非消融中的绝对最优，但在 k-NN、效果和 patch 级计算成本之间折中。

## 14. 与 BEiT、DINO、MAE 的系统比较

### 14.1 总表

| 维度 | BEiT | DINO | MAE | iBOT |
|---|---|---|---|---|
| 主要目标 | 预测离散视觉 token | 跨视图 CLS 自蒸馏 | 重建像素 | CLS + masked patch 自蒸馏 |
| tokenizer | 离线 DALL-E dVAE | 无 | 无 | EMA teacher 在线生成 |
| patch target | one-hot token id | 无显式 masked target | RGB 像素 | sharpened soft distribution |
| teacher | 固定 dVAE | EMA teacher | 无 | EMA teacher |
| mask | block-wise | 无 | 约 75% random | block-wise/随机 ratio |
| encoder 输入 | 含 mask token | 完整 crops | 仅 visible patches | 含 mask token |
| decoder | 轻量 prediction head | projection head | 专用轻量 decoder | shared projection head |
| 强项 | 微调适配 | 全局语义、attention | 高效大规模像素 MIM | 全局 + 局部语义 |

### 14.2 相对 BEiT

iBOT 的主要改进：

- 离线 tokenizer → online tokenizer；
- 固定 hard labels → 动态 soft labels；
- 两阶段 → 单阶段联合训练；
- 低层 dVAE token → CLS 语义塑形的 patch token。

代价是 iBOT 训练更重。

BEiT 不需要 teacher backbone 双前向，也不使用 12-crop 配方。

此外，BEiT 在部分大模型 fine-tuning setting 仍可更好，说明离散低层目标并非无效。

### 14.3 相对 DINO

DINO 只显式监督 CLS 的跨视图一致性。

patch semantics 是 attention 和 ViT 结构中“涌现”的副产品。

iBOT 把 patch token 变成显式训练对象。

所以 iBOT 可以理解为：

$$
\text{iBOT}=\text{DINO global self-distillation}+\text{masked patch self-distillation}.
$$

但它不是普通 DINO 加 dense matching。

MIM 只在同一 crop 的对齐位置预测，避免跨 crop patch correspondence 噪声。

### 14.4 相对 MAE

MAE 与 iBOT 几乎代表两种相反哲学：

- MAE 相信简单像素重建加高遮挡率足以逼出语义；
- iBOT 相信语义 teacher target 比像素 target 更直接。

MAE 的优点：

- 无 teacher；
- 无 prototype centering/sharpening；
- encoder 只看 25% visible patches，计算高效；
- 训练结构简单。

iBOT 的优点：

- 直接优化语义分布；
- 同时得到强 CLS 与 patch tokens；
- frozen linear 与 dense transfer 更自然；
- 在线目标适配当前数据域。

二者的根本差异不是“都做 mask”，而是目标空间：

$$
\text{MAE: pixel space}\qquad\text{iBOT: learned semantic distribution space}.
$$

## 15. iBOT 如何进入 DINOv2 与 DINOv3 配方

### 15.1 DINOv2：从完整方法变成 patch-level loss 组件

DINOv2 明确把训练写成 DINO 与 iBOT 的组合。

其 patch-level objective 保留 iBOT 的核心：

1. student 的部分 patch 被 mask；
2. teacher 看完整图像；
3. 对每个 masked 位置，匹配 teacher 与 student 的概率分布；
4. teacher 由 student EMA 更新。

但 DINOv2 改了几个关键细节：

- DINO CLS head 与 iBOT patch head **不再共享**；
- teacher normalization 使用 Sinkhorn-Knopp centering；
- 加入 KoLeo regularizer，让全局特征在 batch 中展开；
- 加入高分辨率短训练阶段；
- 工程实现约比原 iBOT 快 2×、显存约为 1/3。

这与 iBOT 小规模消融“共享 head 略好”并不矛盾。

DINOv2 在更大数据与模型规模下观察到共享 head 会让 patch 端欠拟合、image 端过拟合，因此改为 untied heads。

### 15.2 DINOv3：保留 iBOT，但承认它不足以长期维持局部性

DINOv3 主训练仍包含：

$$
\mathcal L_{Pre}
=\mathcal L_{DINO}
+\mathcal L_{iBOT}
+0.1\mathcal L_{DKoLeo}.
$$

这说明 iBOT 已从一篇独立 MIM 方法，变成 DINO 系列的标准局部监督组件。

但 DINOv3 发现：

- 随大模型和超长训练推进，CLS/global 指标继续变好；
- patch-level dense performance 却先升后降；
- 单独的 iBOT loss 不能稳定维护 patch 两两关系。

因此 refinement 阶段加入 Gram anchoring：

$$
\mathcal L_{Ref}
=w_D\mathcal L_{DINO}
+\mathcal L_{iBOT}
+w_{DK}\mathcal L_{DKoLeo}
+w_{Gram}\mathcal L_{Gram}.
$$

$\mathcal L_{Gram}$ 对齐 student 与早期 Gram teacher 的 patch-patch similarity matrix。

DINOv3 中 iBOT 负责“masked patch 应预测什么”，Gram loss 负责“整幅图内 patch 关系不能在长训练中退化”。

因此技术脉络是：

1. DINO：全局自蒸馏；
2. iBOT：加入显式 patch MIM；
3. DINOv2：把 DINO+iBOT 工程化、规模化；
4. DINOv3：在保留 iBOT 的同时，再约束 patch 关系结构。

## 16. 训练成本与复现难度

### 16.1 论文报告的硬成本

Table 19 在两台 8×V100、batch size 1024 下比较 800 epochs：

| 方法 | Crops | 时间 | 单卡显存 | Linear | Fine-tune |
|---|---|---:|---:|---:|---:|
| BEiT | 1×224² | 90.1h | 5.6G | 24.2 | 81.4 |
| DINO | 2×224² | 111.6h | 9.3G | 73.7 | 81.6 |
| iBOT | 2×224² | 126.4h | 13.1G | 76.2 | 82.0 |
| DINO multi-crop | 2×224²+10×96² | 180.0h | 15.4G | 77.0 | 82.0 |
| iBOT multi-crop | 2×224²+10×96² | **193.4h** | **19.5G** | **77.9** | **82.3** |

与 multi-crop DINO 相比，论文总结 iBOT：

- 显存增加约 25%；
- 实际训练时间增加约 7.4%。

原因是每个 patch token 都要经过 projection head 并产生 8192 维分布。

193.4 小时 × 16 V100 约为 3094 V100 GPU-hours，仅代表一次完整 800-epoch run。

### 16.2 复现的工程成本

官方实现依赖多机分布式训练。

ViT-S/16 官方示例使用 2 节点 × 8 GPU。

关键命令包含：

- 800 epochs；
- batch size per GPU 64；
- output dimension 8192；
- 10 local crops；
- shared head；
- teacher temperature 0.07；
- 30-epoch teacher temperature warmup。

要复现 COCO/ADE20K，还需独立检测与分割栈。

旧版 mmcv/mmseg/mmdet 与现代 CUDA/PyTorch 的兼容性可能比预训练本身更麻烦。

### 16.3 可行的低成本复现路径

不必从 16×V100 开始。

可按以下顺序验证核心因果链：

1. 使用 ViT-Tiny 或 ViT-Small 和 ImageNet 子集；
2. 关闭 10 个 local crops，先跑无 multi-crop 版本；
3. 同时记录完整 iBOT、去 CLS loss、固定 DINO tokenizer 三组；
4. 比较 frozen k-NN / linear，而不是只看 fine-tuning；
5. 可视化 patch prototype top-k；
6. 再加入 random-$r$ 与 multi-crop，观察 NMI 稳定性；
7. 最后才做 COCO/ADE20K。

最小复现的目标应是验证：

- online tokenizer 是否比 fixed tokenizer 更语义化；
- CLS loss 是否防止 patch vocabulary 退化；
- patch linear probing 是否优于 DINO。

## 17. 局限与批判性评价

### 17.1 方法局限

1. **计算和显存昂贵。**
   双网络、multi-crop、每 patch 8192 维 head 共同放大成本。

2. **online targets 仍依赖复杂防坍塌配方。**
   EMA、center、temperature warmup、multi-crop 和随机零遮挡缺一项都可能影响稳定性。

3. **prototype 语义不是可命名词表。**
   8192 个维度可以出现可解释 pattern，但没有一一对应的人类概念。

4. **遮挡位置监督依赖严格空间对齐。**
   它适合图像内同视图恢复，却没有直接解决跨图像、跨视角的几何 correspondence。

5. **大模型需要更多数据。**
   ViT-L 在 IN-1K 上 fine-tuning 低于 BEiT，扩展到 IN-22K 后才反超。

6. **标准 object discovery 验证缺失。**
   attention map 和 pattern layout 很有说服力，但不能替代 CorLoc 或 mask-level 定量评估。

7. **原论文的局部性结论并非无限 scaling 下成立。**
   DINOv3 后来显示，DINO+iBOT 在超长训练中仍会出现 patch consistency 退化。

### 17.2 实验局限

论文结果受 multi-crop 与 effective epochs 影响。

DINO/iBOT 每 epoch 处理 2 个 global + 10 个 local crops，而 BEiT 每 epoch 的图像视图数不同。

简单比较 epoch 数不公平。

论文因此定义 effective training epochs，但这个折算仍只按像素面积近似，不能完整反映 teacher forward、head 成本和通信成本。

部分 fine-tuning 结果对 layer-wise learning-rate decay、DeepSpeed 和训练日程敏感。

Table 11 表明同一预训练方法换 recipe 可变化接近 1 个点。

### 17.3 对“online tokenizer”命名的评价

它与 NLP tokenizer 有本质差异。

NLP tokenizer 是训练前固定、可重复的符号切分器。

iBOT tokenizer 是训练中的 teacher distribution generator。

它不负责把原始图像压缩成独立可存储 token 序列。

所以更精确的理解是：

> iBOT 是在线语义目标生成器，而不是传统意义上的图像离散编码器。

这个命名有启发性，但也容易让读者误以为它产生稳定离散码本。

## 18. 论文影响与实践启发

### 18.1 最重要的影响

iBOT 把两条当时分离的路线连接起来：

- DINO 式判别自蒸馏；
- BEiT 式 masked image modeling。

它证明 MIM 不必依赖预训练 dVAE，也不必只回归像素。

teacher 自身就可以成为视觉语义目标来源。

### 18.2 对视觉基础模型配方的启发

后续 DINOv2/DINOv3 表明，一个强纯视觉基础模型通常需要同时照顾：

- global semantic invariance；
- local patch prediction；
- feature-space diversity；
- 长训练下的 patch relation stability。

iBOT 解决了第二项，并通过 CLS loss部分解决第一项。

它不是最终完整配方，但成为不可缺少的中间层。

### 18.3 对工程实践的启发

如果下游任务是分类，DINO、MAE、iBOT 都可能合适。

如果重点是 segmentation、detection、matching 或区域级 VLM，应该专门检查 patch feature：

- frozen linear segmentation；
- patch retrieval；
- local correspondence；
- 遮挡鲁棒性；
- 长训练中的 patch similarity map。

不能只依据 ImageNet fine-tuning 选择 backbone。

## 19. 关键术语表

- **MIM**：Masked Image Modeling，遮挡部分图像 patch 后预测目标。
- **Online tokenizer**：由 EMA teacher 动态产生 patch 级软目标的网络。
- **Patch token**：ViT 中对应固定空间 patch 的局部表示。
- **CLS token**：聚合全图信息、用于图像级任务的特殊 token。
- **Self-distillation**：teacher 与 student 架构同源，teacher 由 student 历史参数构造。
- **EMA teacher**：对 student 参数做指数移动平均得到的教师。
- **Centering**：减去 teacher 输出的移动平均中心，抑制 prototype 垄断。
- **Sharpening**：用较低温度使 teacher 分布更尖锐，避免均匀输出。
- **Soft token**：完整概率分布，而不是 argmax 后的 one-hot id。
- **Multi-crop**：同一图像生成多个 global/local crops 共同训练。
- **Part-wise probing**：直接用部分 patch tokens 做分类，测量局部表征语义。
- **Pattern layout**：按 projection-head 维度检索 top-confidence patches，观察 latent prototype 的可解释模式。

## 20. 阅读结论

iBOT 的真正创新不是“再做一次 masked prediction”。

它把视觉 tokenizer 从外部固定模块变成模型内部的慢速 teacher，并用 CLS 跨视图自蒸馏持续给这个 tokenizer 注入语义。

patch loss 让局部 token 可用，CLS loss 让局部目标有语义，EMA/centering/sharpening让联合系统不退化。

实验最有说服力的部分也不是 87.8% 单一数字，而是三条互相支撑的证据：

1. 去掉 CLS loss 后 frozen representation 几乎失效；
2. patch pattern 出现车灯、狗耳等局部语义；
3. ADE20K frozen linear 与 UPerNet 显著超过 BEiT/DINO。

从历史位置看，iBOT 是 DINO 系列从“全局自蒸馏”走向“全局 + 局部视觉基础模型”的关键桥梁。

DINOv2 把它标准化为 patch-level objective。

DINOv3 则说明它仍需 Gram anchoring 才能在 frontier-scale 长训练中维持局部关系。

## 21. 图表索引

1. **图 1：Online tokenizer 框架**  
   Figure: Figure 3；Page: 4；URL: https://arxiv.org/pdf/2111.07832  
   本地：`assets/iBOT - 在线分词器驱动的图像BERT预训练/figures/01-online-tokenizer-framework.png`

2. **图 2：低样本、无监督聚类与 dense 主结果**  
   Figure: Tables 4–6；Page: 6；URL: https://arxiv.org/pdf/2111.07832  
   本地：`assets/iBOT - 在线分词器驱动的图像BERT预训练/figures/03-main-results-lowshot-dense.png`

3. **图 3：局部语义 pattern layout**  
   Figure: Figure 4（含 Table 7）；Page: 7；URL: https://arxiv.org/pdf/2111.07832  
   本地：`assets/iBOT - 在线分词器驱动的图像BERT预训练/figures/02-local-semantic-patterns.png`

4. **图 4：Tokenizer 设计消融**  
   Figure: Table 9；Page: 9；URL: https://arxiv.org/pdf/2111.07832  
   本地：`assets/iBOT - 在线分词器驱动的图像BERT预训练/figures/04-tokenizer-ablation.png`

## 22. 材料来源

完整来源、版本差异与图片裁剪追溯见 [sources.md](assets/iBOT%20-%20在线分词器驱动的图像BERT预训练/sources.md)。
