---
type: paper-note
title: DiT - 可扩展扩散Transformer
aliases:
  - Scalable Diffusion Models with Transformers
  - Diffusion Transformer
  - DiT
tags:
  - Foundations
  - generative-modeling
  - diffusion-model
  - transformer
  - latent-diffusion
  - image-generation
domain: Foundations
paper: "Scalable Diffusion Models with Transformers"
arxiv: "2212.09748"
doi: "10.48550/arXiv.2212.09748"
source: "https://arxiv.org/abs/2212.09748"
project: "https://www.wpeebles.com/DiT"
code: "https://github.com/facebookresearch/DiT"
code_commit: "ed81ce2229091fd4ecc9a223645f95cf379d582b"
year: 2022
venue: CVPR 2023
status: complete
created: 2026-10-16
updated: 2026-10-30
---

# DiT：可扩展扩散 Transformer

> **论文**：William Peebles, Saining Xie. *Scalable Diffusion Models with Transformers*  
> **版本**：arXiv:2212.09748v2，2023-03-02（v1 提交于 2022-12-19；CVPR 2023）  
> **来源**：[arXiv 摘要页](https://arxiv.org/abs/2212.09748) · [官方 PDF](https://arxiv.org/pdf/2212.09748) · [[assets/DiT - 可扩展扩散Transformer/sources/paper.pdf|本地 PDF]]  
> **本地材料**：[[assets/DiT - 可扩展扩散Transformer/sources.md|来源与逐图核验记录]]  
> **精读日期**：2026-10-16

## 1. 一句话总结（TL;DR）

DiT（Diffusion Transformer）把 latent diffusion 中负责去噪的卷积 U-Net 换成近似标准 ViT：先把带噪 VAE latent 切成 patch token，再用 Transformer 预测噪声与反向过程协方差，并以 **adaLN-Zero** 注入时间步和类别条件。论文最重要的结论不是简单的“参数越多越好”，而是：在其 12 个模型组成的设计空间内，无论通过加深/加宽网络还是减小 patch、增加 token，只要 Transformer 的单次前向计算量提高，FID 就稳定改善；最终 DiT-XL/2 在类条件 ImageNet $256\times256$ 上达到 FID 2.27。

## 2. 研究背景、动机与问题设定

### 2.1 扩散模型为什么长期依赖 U-Net

DDPM 的去噪网络需要在每个时间步同时处理局部纹理、物体结构和全局语义。卷积 U-Net 通过多尺度下采样—上采样、局部卷积和跨层 skip connection，天然具备局部性、平移等变和多尺度归纳偏置，因此从 DDPM、Improved DDPM、ADM 到 LDM，一直是图像扩散模型的默认骨干。

与此同时，NLP、视觉识别和自回归图像生成已经逐步统一到 Transformer。扩散模型仍由高度专用的 U-Net 主导，意味着研究者很难直接复用 Transformer 在规模扩展、优化稳定性和跨模态表示方面的经验。DiT 的出发点因此非常直接：**扩散模型真正需要的是 U-Net，还是只需要一个足够强、能接收时间条件的函数逼近器？**

### 2.2 论文要回答的三个核心问题

1. **必要性问题**：高质量图像扩散是否必须依赖卷积 U-Net？
2. **结构问题**：尽量接近标准 ViT 的 Transformer，能否直接承担 latent 去噪骨干？
3. **扩展问题**：质量如何随深度、宽度和 token 数量变化；参数量、训练计算和单次前向 Gflops 中，哪一个最能解释 FID？

论文刻意不改 VAE、前向扩散过程和主要训练 recipe，而把变量集中在 backbone、patch size 与条件注入方式上。因此它首先是一项结构与扩展性研究，而不是新的扩散概率模型或新采样器。

### 2.3 实验逻辑

作者构造三条相互独立的设计轴：

- **模型规模**：S、B、L、XL；
- **latent patch size**：$p\in\{8,4,2\}$；
- **条件注入**：in-context、cross-attention、adaLN、adaLN-Zero。

前两条轴形成 12 个主模型，用来观察规模、token 数和 Gflops 的关系；第三条轴单独做结构消融。这样可以区分“更多参数”和“同参数但更多 token 计算”带来的收益。

## 3. 核心贡献

1. **提出 Diffusion Transformer（DiT）骨干。** 它在 latent diffusion 管线中用固定分辨率的 Transformer token 序列替代多尺度卷积 U-Net，证明 U-Net 不是扩散去噪的必要条件。
2. **建立系统化 DiT 设计空间。** 作者同时比较模型深宽、patch size 和四种条件化方式，不只展示一个大模型的最终分数。
3. **提出 adaLN-Zero。** 时间步与类别条件生成 LayerNorm 的 scale、shift 和 residual gate；调制层零初始化使每个 block 在训练开始时近似恒等映射，兼顾稳定性、条件带宽和低计算开销。
4. **用 Gflops 统一解释两类扩展路径。** 加深/加宽会增加参数和计算，减小 patch 则几乎不改变参数量却显著增加 token 与计算；两种路径都降低 FID，说明参数量不是充分指标。
5. **给出强 ImageNet 生成基线。** 长训练与 CFG 下，DiT-XL/2 在 ImageNet $256^2$ 和 $512^2$ 上分别达到 FID 2.27 和 3.04，并展示出优于多种像素空间 U-Net 基线的 backbone 计算效率。

## 4. 方法论详解

### 4.1 总体架构与 latent diffusion I/O

![](assets/DiT%20-%20可扩展扩散Transformer/figures/01-architecture-conditioning.png)

> **图 1 解读**：左侧是完整 latent DiT：带噪 latent 经 patchify 后进入 $N$ 个 DiT block，时间步 $t$ 与类别 $y$ 作为条件，末端同时输出噪声和对角协方差。右侧展示 adaLN-Zero、cross-attention 与 in-context 三种 block；普通 adaLN 与 adaLN-Zero 共享调制思路，但没有后者的零初始化 residual gate。来源：官方 TeX `figures/block.pdf`，原论文 Figure 3。

DiT 改造的是 latent diffusion 的 **denoiser**，不是整条 LDM 管线。以 ImageNet $256\times256$、VAE 下采样因子 8、patch size $p=2$ 为例：

1. **VAE 编码**：冻结的卷积 encoder 把 RGB 图像压缩为 4 通道 latent。

$$x\in\mathbb R^{256\times256\times3},\qquad z_0=E(x)\in\mathbb R^{32\times32\times4}$$

2. **前向加噪**：随机采样时间步 $t$ 与高斯噪声 $\epsilon$，直接构造 $z_t$。
3. **Patchify**：把 $32\times32\times4$ 的带噪 latent 切成 $2\times2$ 非重叠 patch，得到 $16\times16=256$ 个 image token。
4. **位置编码**：为 image token 加固定二维 sine-cosine positional embedding。
5. **条件编码**：时间步 $t$ 经频率编码和 MLP；类别 $y$ 经 embedding lookup；二者组成全局条件。
6. **Transformer 去噪**：token 经过 $N$ 个 DiT block，默认最佳版本为 adaLN-Zero。
7. **线性解码**：每个 token 映射为 $p\times p\times2C$ 个数，其中 $C=4$；两组 $C$ 通道分别服务于噪声/均值参数化和 learned variance。
8. **Unpatchify**：把 token 重排回空间布局，得到噪声预测 $\epsilon_\theta$ 与对角协方差参数 $\Sigma_\theta$。
9. **反向采样与 VAE 解码**：从高斯 latent 开始迭代得到 $z_0$，再输出 $\hat x=D(z_0)$。

完整形状流可以概括为：

$$256\times256\times3\xrightarrow{E}32\times32\times4\xrightarrow{q(z_t\mid z_0)}32\times32\times4\xrightarrow{\mathrm{patchify},\,p=2}256\times d\xrightarrow{\mathrm{DiT}}256\times(2\cdot2\cdot8)\xrightarrow{\mathrm{unpatchify}}32\times32\times8$$

最后的 8 通道不是新的 latent 维数，而是 $4$ 通道噪声预测与 $4$ 通道方差参数的拼接。

### 4.2 为什么选择 latent space

若直接在 $256\times256$ RGB pixel space 上做细粒度 patchify，序列会很长。VAE 把空间边长缩小 8 倍，token 数在相同 patch size 下缩小 64 倍，作者因此可以把计算预算集中在 Transformer backbone 的系统性比较上。

论文使用 Stable Diffusion 提供的预训练 VAE：

$$256\times256\times3\xrightarrow{E}32\times32\times4$$

$$512\times512\times3\xrightarrow{E}64\times64\times4$$

所以 DiT 是混合系统：**卷积 VAE 负责感知压缩和图像重建，Transformer 负责带噪 latent 的分布建模**。论文声称架构也能用于 pixel space，但正文并未验证；“Transformer 替换 U-Net”不能误读为“整个系统不再含卷积”。

### 4.3 Patchify：把二维 latent 变成 token

设 latent 空间大小为 $I\times I\times C$，patch 边长为 $p$，则 token 数为：

$$T=\left(\frac{I}{p}\right)^2$$

每个 $p\times p\times C$ patch 被展平并线性投影到隐藏维 $d$。论文考察 $p\in\{8,4,2\}$：

| 图像分辨率 | VAE latent | patch size $p$ | token 数 $T$ |
|---|---:|---:|---:|
| $256^2$ | $32\times32\times4$ | 8 | 16 |
| $256^2$ | $32\times32\times4$ | 4 | 64 |
| $256^2$ | $32\times32\times4$ | 2 | 256 |
| $512^2$ | $64\times64\times4$ | 2 | 1024 |

将 $p$ 减半会把 token 数放大 4 倍；它只轻微改变首尾线性层的权重，却至少把 Transformer 的 token 相关计算放大约 4 倍。因此 patch size 是一个几乎不增加参数、但显著增加前向计算和空间建模粒度的旋钮。

全局 self-attention 含 $O(T^2d)$ 项，但整网还包含 attention projection 和 $O(Td^2)$ 的 MLP，所以论文报告的总 Gflops 并不严格按 $T^2$ 增长。尽管如此，从 /8 到 /2 的计算增幅依然很大。

### 4.4 时间步与类别条件编码

时间步与类别承担不同角色：$t$ 告诉网络当前噪声水平，$y$ 指定要生成的 ImageNet 类别。

- **时间步 $t$**：先做 256 维 frequency embedding，再经过两层 SiLU MLP，输出维度对齐 Transformer hidden size。
- **类别 $y$**：通过可学习 embedding table 得到 class vector；为 CFG 训练额外保留 null class embedding。
- **全局组合**：adaLN 系列把两者相加。

$$c=e_t(t)+e_y(y)$$

$c$ 对同一样本的全部空间 token 相同。每个 DiT block 再用自己的调制 MLP，把 $c$ 转换为该层所需的 scale、shift 和 residual gate。

### 4.5 四种条件化方式

#### 4.5.1 In-context conditioning

把时间与类别 embedding 视为两个普通 token，直接追加到 image token 序列：

$$[h_1,\ldots,h_T,e_t,e_y]$$

所有 token 一起经过标准 ViT block，最终输出前丢弃两个条件 token。它的优点是不改 Transformer 结构、额外 Gflops 极小；缺点是条件只能通过 self-attention 间接扩散到图像 token，条件路径长，实验效果最差。

#### 4.5.2 Cross-attention conditioning

保留 image token 主序列，把 $[e_t,e_y]$ 组成长度为 2 的条件序列，并在每个 self-attention 后增加 multi-head cross-attention：image token 作 query，条件 token 作 key/value。

这种设计最接近文本条件 LDM，具备更一般的条件序列接口；代价是每层多一套 attention，论文估计约增加 15% Gflops，是四种方案中前向计算最贵者。对只有两个全局向量的 ImageNet 类别条件，它的表达能力没有转化为最佳 FID。

#### 4.5.3 Adaptive LayerNorm（adaLN）

adaLN 不新增条件 token，而让 $c$ 生成逐通道 scale 与 shift：

$$\operatorname{adaLN}(h;c)=\left(1+\gamma(c)\right)\odot\operatorname{LN}(h)+\beta(c)$$

attention 和 MLP 两个子层前各使用一组 $(\gamma,\beta)$。它几乎不增加与 token 数相关的计算，因此是普通三种条件化方案中最省 Gflops 的；限制是同一样本的所有空间 token 共用同一组调制，无法直接表达 token-specific 条件。

#### 4.5.4 adaLN-Zero

adaLN-Zero 在两组 scale/shift 之外，再预测两个逐通道 residual gate：

$$(\beta_1,\gamma_1,\alpha_1,\beta_2,\gamma_2,\alpha_2)=\operatorname{MLP}(c)$$

一个 block 的前向可写为：

$$\tilde h=h+\alpha_1(c)\odot\operatorname{MSA}\!\left(\left(1+\gamma_1(c)\right)\odot\operatorname{LN}(h)+\beta_1(c)\right)$$

$$h'=\tilde h+\alpha_2(c)\odot\operatorname{MLP}\!\left(\left(1+\gamma_2(c)\right)\odot\operatorname{LN}(\tilde h)+\beta_2(c)\right)$$

调制 MLP 的输出层被零初始化；官方实现由此让六组输出在初始化时都为零，特别是 residual gate 满足：

$$\alpha_1=\alpha_2=0\quad\Longrightarrow\quad h'=h$$

因此每个 block 在训练开始时近似恒等映射，优化过程再逐步“打开”attention 与 MLP 分支。这个思路承接了 ResNet 的 zero-init residual，以及扩散 U-Net 对 residual block 末端卷积做零初始化的经验。

adaLN-Zero 的价值来自三件事的组合，而不是 adaLN 这个名字本身：

1. **短条件路径**：$t$ 与 $y$ 直接控制每层的归一化和残差强度；
2. **稳定初始化**：深层堆叠开始时不大幅扰动 token；
3. **低计算成本**：不新增随 $T$ 增长的 attention，Gflops 与普通 adaLN 接近。

### 4.6 Transformer 主体与最终输出头

Patch token 经过 $N$ 个固定分辨率 DiT block。与 U-Net 不同，主干中没有显式下采样、上采样或 U 形 skip connection；全局 self-attention 从第一层起就允许任意两个 latent patch 交互。

最后一个 block 之后，模型执行 final LayerNorm；使用 adaLN 系列时，该归一化也受条件调制。每个 token 再线性解码为：

$$o_i\in\mathbb R^{p\times p\times2C}$$

所有 token 经 unpatchify 恢复空间布局：

$$[o_1,\ldots,o_T]\longrightarrow[\hat\epsilon,\hat\Sigma]\in\mathbb R^{I\times I\times2C}$$

最终线性层也零初始化，所以模型初始输出为零。block 的恒等初始化和输出头的零初始化共同构成 DiT 的稳定训练 recipe。

### 4.7 前向扩散、反向过程与训练目标

论文沿用高斯前向扩散：

$$q(z_t\mid z_0)=\mathcal N\!\left(z_t;\sqrt{\bar\alpha_t}z_0,(1-\bar\alpha_t)\mathbf I\right)$$

利用重参数化，可以一步采样任意时间的带噪 latent：

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad\epsilon\sim\mathcal N(0,\mathbf I)$$

模型学习高斯反向转移：

$$p_\theta(z_{t-1}\mid z_t,y)=\mathcal N\!\left(\mu_\theta(z_t,t,y),\Sigma_\theta(z_t,t,y)\right)$$

与 ADM / Improved DDPM 一致，噪声分支使用简单 MSE：

$$\mathcal L_{\mathrm{simple}}=\mathbb E_{z_0,t,\epsilon,y}\!\left[\left\|\epsilon-\epsilon_\theta(z_t,t,y)\right\|_2^2\right]$$

learned variance 则由 variational lower bound 中的 KL 项训练。也就是说，DiT 的核心贡献不在损失函数；它有意继承成熟的扩散目标，以便把质量差异归因于骨干结构。

### 4.8 Classifier-Free Guidance（CFG）

训练时以一定概率把类别 $y$ 替换为 null 条件 $\varnothing$，让同一个网络同时学习条件和无条件预测。采样时组合两者：

$$\hat\epsilon_\theta(z_t,y)=\epsilon_\theta(z_t,\varnothing)+s\left[\epsilon_\theta(z_t,y)-\epsilon_\theta(z_t,\varnothing)\right],\qquad s>1$$

- $s=1$ 等价于普通条件预测；
- 增大 $s$ 往往提高类别一致性、Precision 和感知质量；
- 过强 guidance 会压缩样本覆盖，降低 Recall。

必须区分两组实验：主 scaling sweep 在 400K steps 时不使用 CFG，最终 SOTA 表才使用长训练与 CFG。因此 FID 19.47 和 2.27 不是同一训练预算下的可直接比较数字。

附录还有一个特殊实现细节：最终实验默认只对四个 latent channel 中的前三个施加 guidance。作者报告，对四个通道全部引导并降低 scale 可得到近似结果，例如三通道 $s=1.5$ 为 FID 2.27，四通道 $s=1.375$ 为 FID 2.20；论文没有给出理论解释。

## 5. 模型规模、参数量与 Gflops

### 5.1 四档 Transformer 配置

| 模型 | Block 数 $N$ | Hidden size $d$ | Heads | 参数量（约） | $p=4$ 前向 Gflops |
|---|---:|---:|---:|---:|---:|
| DiT-S | 12 | 384 | 6 | 33M | 1.4 |
| DiT-B | 12 | 768 | 12 | 130M | 5.6 |
| DiT-L | 24 | 1024 | 16 | 458M | 19.7 |
| DiT-XL | 28 | 1152 | 16 | 675M | 29.1 |

S/B/L 大体沿用 ViT 配置，XL 是作者新增的最大规格。模型名同时编码 backbone 与 latent patch size：`DiT-XL/2` 表示 XL backbone、$p=2$；`DiT-B/8` 表示 Base backbone、$p=8$。

### 5.2 为什么参数量不足以解释质量

参数量描述模型存储了多少权重，却不描述每次前向对这些权重应用了多少 token：

- 增加 depth/width：参数量与 Gflops 同时增加；
- 减小 patch size：参数量基本不变，但 token 数与 Gflops 大幅增加。

所以同一 backbone 的 /8、/4、/2 参数量近似相同，FID 却差距巨大；不同规格若 Gflops 接近，质量也常接近，例如 DiT-S/2 与 DiT-B/4。论文据此把 **backbone 单次 forward Gflops** 作为统一横轴。

这个指标仍有明确边界：它只统计 DDPM/DiT backbone，不包含 84M 参数 VAE、CFG 的条件/无条件双分支、设备通信、内存访问和 kernel 利用率，因此不能直接等价为 wall-clock latency 或完整系统成本。

## 6. 实验设置

### 6.1 数据、训练和评估协议

- **数据集**：ImageNet，类条件生成；
- **分辨率**：$256\times256$ 与 $512\times512$；
- **优化器**：AdamW，恒定学习率 $10^{-4}$，weight decay 0，batch size 256；
- **数据增强**：仅 horizontal flip；
- **EMA**：decay 0.9999，报告结果使用 EMA 权重；
- **扩散过程**：$t_{\max}=1000$，线性 variance schedule 从 $10^{-4}$ 到 $2\times10^{-2}$，其余参数化沿用 ADM；
- **采样**：与既有工作对比时使用 250 个 DDPM sampling steps；
- **指标**：FID-50K 为主，另报 sFID、Inception Score、Precision、Recall；
- **评估实现**：使用 ADM 的 TensorFlow evaluation suite，减少不同 FID 实现造成的偏差；
- **训练硬件**：JAX + TPU-v3 pod；DiT-XL/2 在 TPU v3-256、global batch 256 时约 5.7 iterations/s。

所有规模共用同一套超参数，没有逐模型调 learning rate、warmup、Adam $\beta$ 或 weight decay。作者也未观察到常见的 loss spike，这支持“架构与初始化本身足够稳定”的主张。

## 7. 定量结果与消融实验

### 7.1 四种条件化：adaLN-Zero 最优

![](assets/DiT%20-%20可扩展扩散Transformer/figures/02-conditioning-ablation.png)

> **图 2 解读**：四种 DiT-XL/2 条件化方式在训练全程的 FID-50K；adaLN-Zero 从早期到 400K steps 始终最低。来源：官方 TeX `figures/conditioning.pdf`，原论文 Figure 5。

根据 400K steps 曲线终点与模型复杂度统计：

| Block | Gflops | 参数量 | FID-50K ↓ |
|---|---:|---:|---:|
| In-context | 119.37 | 449M | 35.24 |
| Cross-attention | 137.62 | 598M | 26.14 |
| adaLN | 118.56 | 600M | 25.21 |
| **adaLN-Zero** | **118.64** | **675M** | **19.47** |

结论有四层：

1. in-context 最接近原生 ViT，却明显落后，说明条件如何进入网络非常关键；
2. cross-attention 更贵，但在两个全局条件向量的场景里没有超过 adaLN；
3. adaLN-Zero 显著优于普通 adaLN，表明收益不只是 adaptive normalization；
4. 四者不是严格等参数比较，但 adaLN-Zero 与 adaLN 的 Gflops 几乎相同，排除了“只是每次前向算得更多”的解释。

此后所有主实验固定使用 adaLN-Zero。

### 7.2 12 个模型：增大 backbone 和增加 token 都有效

![](assets/DiT%20-%20可扩展扩散Transformer/figures/03-gflops-fid.png)

> **图 3 解读**：左图气泡面积表示前向 Gflops；同一颜色内，patch 越小、气泡越大、FID 越低。右图显示带 CFG 的 DiT-XL/2 在当时扩散模型中兼具较低 FID 与较低 backbone 前向计算。来源：官方 TeX `figures/teaser_new.pdf`，原论文 Figure 2。

![](assets/DiT%20-%20可扩展扩散Transformer/figures/04-scaling-curves.png)

> **图 4 解读**：上排固定 patch size 比较 S/B/L/XL，下排固定 backbone 比较 /8、/4、/2；两条扩展轴在训练各阶段都有效，不只是改变最终收敛点。来源：官方 TeX `figures/scaling_new.pdf`，原论文 Figure 6。

400K steps、无 CFG 的 FID-50K：

| Backbone | /8 | /4 | /2 |
|---|---:|---:|---:|
| S | 153.60 | 100.41 | 68.40 |
| B | 122.74 | 68.38 | 43.47 |
| L | 118.87 | 45.64 | 23.33 |
| XL | 106.41 | 43.01 | **19.47** |

对应 /2 模型从 S 到 XL 的单次前向计算量为：

$$6.06\rightarrow23.01\rightarrow80.71\rightarrow118.64\ \text{Gflops}$$

表格支持三个判断：

- **固定 patch size**：加深/加宽 backbone 总能降低 FID；
- **固定参数规格**：从 /8 到 /2 增加 token 数，也总能降低 FID；
- **统一解释变量**：Gflops 比参数量更好地对齐两种扩展路径。

不过，这只是固定数据、训练 recipe 和同一家族 12 个模型中的强相关性。论文没有拟合可跨任务、跨架构外推的幂律，因此称其为“严格 scaling law”会过度解读。

### 7.3 定性样例：改善的不只是纹理

![](assets/DiT%20-%20可扩展扩散Transformer/figures/05-visual-scaling.png)

> **图 5 解读**：12 个模型共享相同初始噪声、采样噪声与类别；向右增加 Transformer 规格、向下减小 patch size，动物形体、类别辨识度、背景结构与局部细节逐步改善。来源：官方 TeX `figures/visual_scaling_new.pdf`，原论文 Figure 7。

共享 seed 的设计控制了随机样本差异，使 scale 的影响更直观。它说明小模型常不只是纹理模糊，而是会生成错误物种、畸形结构或不一致背景；更大 backbone 和更多 token 改善的是语义与结构建模。但这仍是选定类别和 seed 的展示，不能替代分布级指标。

### 7.4 训练计算：大模型可能更高效

作者用下式粗略估计总训练计算：

$$\text{Training Compute}\approx\text{Forward Gflops}\times\text{Batch Size}\times\text{Training Steps}\times3$$

因子 3 近似“前向 1 倍 + 反向 2 倍”。实验显示，小模型即使训练更久，在同等累计理论计算下仍可能落后于大模型；例如累计约 $10^{10}$ Gflops 后，XL/2 超过 XL/4。

这说明扩大模型和 token 数可能改善单位训练计算换来的质量，而不只是暴力消耗更多算力。但结论基于理论 FLOPs，不包含真实设备的内存、通信和吞吐差异。

### 7.5 ImageNet 256：FID 2.27 的组成

作者把 DiT-XL/2 从 scaling sweep 的 400K steps 继续训练到 7M steps：

| DiT-XL/2，$256\times256$ | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|
| 无 CFG | 9.62 | 6.85 | 121.50 | 0.67 | **0.67** |
| CFG 1.25 | 3.22 | 5.28 | 201.77 | 0.76 | 0.62 |
| **CFG 1.50** | **2.27** | **4.60** | **278.24** | **0.83** | 0.57 |

正确解读是：

- 2.27 来自 **DiT-XL/2 + 7M steps + CFG 1.5 + ft-EMA VAE decoder**，不是 400K-step 架构消融的结果；
- CFG 将 FID 从 9.62 降到 2.27，同时 Recall 从 0.67 降到 0.57，显示 fidelity—diversity trade-off；
- 当时表中的先前最佳扩散结果 LDM-4-G 为 3.60，DiT 降到 2.27，也略优于 StyleGAN-XL 的 2.30。

### 7.6 ImageNet 512：序列扩到 1024 token

$512\times512$ 图像经 VAE 得到 $64\times64\times4$ latent，DiT-XL/2 的 token 数为：

$$T=(64/2)^2=1024$$

单次 backbone 前向为 524.6 Gflops。训练 3M steps 后：

| DiT-XL/2，$512\times512$ | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|
| 无 CFG | 12.03 | 7.12 | 105.25 | 0.75 | **0.64** |
| CFG 1.25 | 4.64 | 5.77 | 174.77 | 0.81 | 0.57 |
| **CFG 1.50** | **3.04** | **5.02** | **240.82** | **0.84** | 0.54 |

论文统计像素空间 ADM 为 1983 Gflops、ADM-U 为 2813 Gflops，而 DiT-XL/2 为 524.6 Gflops。这个优势一部分来自 Transformer backbone，一部分来自 DiT 工作在压缩 latent 而 ADM 工作在 pixel space；不能把全部差距都归功于“Transformer 比 U-Net 高效”。

### 7.7 增加采样步数不能补足小 backbone

作者让 12 个 400K-step 模型分别使用 16、32、64、128、256 和 1000 个采样步。一个代表性比较是：

- DiT-L/2，1000 steps：每张图约 80.7 Tflops，FID-10K 25.9；
- DiT-XL/2，128 steps：每张图约 15.2 Tflops，FID-10K 23.7。

XL/2 用约五分之一采样计算得到更好 FID，说明在论文的 DDPM sampler 与模型家族内，test-time 多迭代无法完全补偿 model capacity 不足。这不代表 sampler 不重要，只说明“先扩大 backbone”在该实验范围内更有效。

### 7.8 VAE decoder 消融

同一个 DiT latent 可接不同的预训练 decoder，因为三者 encoder 相同，不需要重训 denoiser：

| Decoder | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|
| LDM original | 2.46 | 5.18 | 271.56 | 0.82 | 0.57 |
| ft-MSE | 2.30 | 4.73 | 276.09 | 0.83 | 0.57 |
| ft-EMA | **2.27** | **4.60** | **278.24** | **0.83** | 0.57 |

差距不大，说明 DiT 的主体优势并非由 decoder 偷换而来；但最终 2.27 仍包含从原始 LDM decoder 到 ft-EMA 带来的 0.19 FID 改善。

## 8. 结论边界与局限性

### 8.1 实验充分支持的结论

1. 在给定 LDM/ADM recipe 下，纯 Transformer 可以替换 U-Net denoiser。
2. adaLN-Zero 在论文测试的四种条件化方式中效果最好，且前向开销接近普通 adaLN。
3. 对 12 个 DiT 变体，提高 forward Gflops 与降低 FID 强相关。
4. 加深/加宽 backbone 与增加 token 数是两条都有效的扩展轴。
5. 小 backbone 的劣势无法仅靠更多 DDPM sampling steps 消除。

### 8.2 不应过度外推的结论

1. 论文没有证明所有生成任务上 Transformer 都优于 U-Net；核心实验集中在类条件 ImageNet。
2. 论文没有证明 Gflops 是跨硬件、跨架构的唯一复杂度指标。
3. 论文没有建立可预测的通用 scaling law，只展示有限范围内的稳定单调趋势。
4. 原始 DiT 没有验证长文本、多模态条件、视频、3D 或 pixel-space 训练。
5. DiT 没有消除 VAE，也没有解决多步扩散采样的延迟。

### 8.3 VAE bottleneck

生成质量上限受冻结 VAE 的重建能力约束。细小文字、高频纹理和 VAE 难以编码的结构，可能在 denoiser 看到之前已经丢失，或在最终 decode 时失真。因此原始系统更准确的名称是“Transformer latent denoiser + convolutional VAE”。

### 8.4 算力成本与系统指标

最强 $256^2$ 模型训练 7M steps，$512^2$ 模型训练 3M steps，总预算依然巨大。理论 Gflops 还忽略 attention 内存、带宽、通信、activation checkpointing、混合精度、CFG 双前向与 VAE 成本。“backbone Gflops 更低”不等价于同比例 wall-clock 加速。

### 8.5 条件化消融不是严格等参数

四种 block 的参数量从 449M 到 675M 不等。adaLN-Zero 的大幅优势很有说服力，但不能完全拆分条件形式、参数量、residual gate 与 zero initialization 各自的贡献。更严格的实验应加入参数匹配、gate-but-nonzero、zero-init-without-gate 等变体。

### 8.6 条件类型过窄

ImageNet 条件只是一个离散类别，全局共享的 adaLN 调制非常适合它；自然语言则是长序列，包含实体、属性、关系和位置约束。原论文没有证明把文本压成单个全局向量后，adaLN 仍能维持细粒度文本对齐。

### 8.7 FID 与 CFG 的边界

FID 只衡量 Inception feature 分布距离，不能完整反映构图、文字、语义正确性或训练数据记忆。CFG 大幅优化 FID 的同时降低 Recall，所以 2.27 是一个偏向 fidelity 的工作点，不代表所有质量维度同步改善。

### 8.8 采样仍然缓慢

DiT 改进每次网络评估的表示能力与扩展性，却没有改变 DDPM 多次迭代的基本机制。论文最终对比使用 250 sampling steps；快速 ODE/SDE solver、蒸馏、consistency model 和 flow matching 属于与 backbone 正交的研究方向。

## 9. 与 U-Net 及后续 SiT、MMDiT 的关系

### 9.1 DiT 与扩散 U-Net：替换了什么，继承了什么

| 维度 | 扩散 U-Net | 原始 DiT |
|---|---|---|
| 主干 | 卷积 ResBlock + 部分分辨率 attention | 全局 self-attention + MLP |
| 空间组织 | 多尺度下采样/上采样，U 形 skip | 固定 latent 分辨率的 patch token 序列 |
| 归纳偏置 | 局部性、平移等变、多尺度 | 视觉先验较弱，更依赖数据与规模 |
| 条件注入 | adaptive norm、cross-attention | adaLN-Zero、cross-attention 或 condition token |
| 主要扩展轴 | channel、block、分辨率层级 | depth、width、token 数 |
| 论文输入空间 | pixel 或 latent 均常见 | 实验为 VAE latent |
| 输出目标 | 噪声、均值、方差等 | 噪声与 learned variance |

DiT 并未否定 U-Net 的全部经验。它继承了 timestep/class embedding、adaptive normalization、residual 分支零初始化、ADM 的噪声目标、variance parameterization 与训练配置。真正改变的是去噪特征提取器的宏观拓扑：从多尺度卷积 U 形网络，变为固定分辨率的 Transformer 堆叠。

这带来清晰的工程取舍：U-Net 用强视觉先验提高局部与多尺度效率；DiT 用更统一、更容易扩展和跨模态连接的 token 接口换取规模收益。

### 9.2 【后世视角】SiT：保留 DiT 骨干，改写生成动力学

> 本节讨论后续工作，不属于原论文贡献或实验。

SiT（*Scalable Interpolant Transformers*, arXiv:2401.08740）直接建立在 DiT backbone 上，尽量保持网络参数量和 Gflops 可比，主要改变概率路径、预测目标与采样动力学：

- 从离散 DDPM 扩展到 stochastic interpolant 的统一连续时间框架；
- 比较不同 interpolant、velocity/score 等预测参数化；
- 同时研究确定性 ODE 与随机 SDE sampler；
- 在尽量不改变 backbone 的前提下，分离“网络架构”和“生成动力学”的贡献。

二者关系可概括为：

$$\text{DiT}=\text{可扩展的 Transformer 扩散骨干}$$

$$\text{SiT}=\text{DiT 类骨干}+\text{更一般的 interpolant / flow-diffusion 动力学}$$

SiT 反过来证明了 DiT 的基础价值：骨干标准化后，研究者可以固定网络，单独研究 diffusion、flow 与 sampler。

### 9.3 【后世视角】MMDiT：从单流全局调制到多模态联合注意力

> 本节讨论后续工作，不属于原论文贡献或实验。

*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*（arXiv:2403.03206）提出 MMDiT（Multimodal Diffusion Transformer），面向高分辨率 text-to-image。它继承 latent patch token、Transformer 扩展与扩散/流式生成范式，但针对长文本条件重构了条件通路：

1. **模态专属参数**：文本与图像 token 使用各自的线性映射和 Transformer 子层权重，而非把文本压成单个 class vector；
2. **Joint attention**：文本和图像的 $Q/K/V$ 在联合注意力中交换信息，图像读取细粒度文本，文本表示也能被视觉上下文更新；
3. **更高条件带宽**：从 DiT 的“单流 image token + 全局 adaLN 条件”升级为图像/文本双流参数化；
4. **Rectified flow 目标**：训练动力学不再是原始 DiT 的 DDPM noise-prediction recipe；
5. **大规模 T2I 工程**：后续加入 QK-normalization、分辨率相关 timestep shifting 和 mixed aspect ratio 等原始 DiT 未覆盖的设计。

因此 MMDiT 不是简单地把 class embedding 换成 text embedding，而是解决原始 DiT 在复杂条件序列上的信息瓶颈：DiT 提供可扩展视觉生成骨干，MMDiT 则把它扩展为高带宽多模态生成架构。

## 10. 复现检查表与常见误读

### 10.1 最小复现检查表

1. 使用下采样因子 8 的冻结 VAE，并确认 latent scaling 与所用 VAE 一致。
2. 对带噪 $z_t$ 做非重叠 patchify，加入固定二维 sine-cosine 位置编码。
3. timestep 使用 256 维 frequency embedding 和两层 SiLU MLP。
4. class embedding 与 timestep embedding 相加形成全局条件。
5. 每个 block 从条件向量生成六组 adaLN-Zero 参数。
6. 调制 MLP 输出层和最终 projection 层均零初始化。
7. 输出维度为 $2C$，同时学习 noise 与 diagonal covariance。
8. 基础训练配置为 AdamW、lr $10^{-4}$、weight decay 0、batch size 256、EMA 0.9999。
9. 比较 scaling 时必须同时注明 backbone、patch size、训练步数、Gflops、CFG 与 VAE decoder。
10. 报告 FID 时注明采样步数、评估实现、是否使用 CFG 及 guidance scale。

### 10.2 最容易读错的五点

1. `DiT-XL/2` 的 `/2` 是 **latent patch size**，不是 VAE 下采样倍率、层数或采样步数。
2. FID 2.27 是 7M-step + CFG 1.5 + ft-EMA decoder 的最终结果，不是 400K-step scaling sweep 的结果。
3. DiT 输出不只有 $\epsilon$，还输出 learned covariance 参数，所以末端是 $2C$ 通道。
4. “Transformer 替换 U-Net”只发生在 diffusion backbone，VAE 仍是卷积网络。
5. 论文展示的是有限设计空间中的 Gflops—FID 强相关，不是已经拟合完成、可任意外推的通用幂律。

## 11. 代码级实现

### 11.1 官方仓库、版本与开源边界

- 仓库：https://github.com/facebookresearch/DiT
- 固定 commit：`ed81ce2229091fd4ecc9a223645f95cf379d582b`
- 框架：PyTorch + timm + diffusers，训练使用 NCCL DDP。
- 许可证：CC-BY-NC。
- 开源范围：模型定义、最小训练脚本、单卡与分布式采样、预训练权重、ADM 风格扩散实现。

需要严格区分论文与代码：论文模型原本在 JAX/TPU 上训练，仓库中的预训练权重由 JAX 直接移植到 PyTorch。官方 README 报告 PyTorch 版在数十万步范围能复现相近趋势，并提供完整的最小 DDP 训练脚本；但它没有自动 FID、断点续训、AMP/bfloat16 等完整训练基础设施。因此应称为**官方 PyTorch 对应实现**，不能写成论文 JAX 训练系统逐行开源。

| 论文模块 | 官方代码入口 | 阶段 | 主要产物 |
|---|---|---|---|
| VAE latent | `train.py::AutoencoderKL` | 训练输入 | `[B,4,32,32]` scaled latent |
| Patchify | `models.py::DiT.x_embedder` | 前向 | `[B,256,1152]` tokens |
| 时间步编码 | `models.py::TimestepEmbedder` | 前向 | `[B,1152]` |
| 类别编码与 CFG dropout | `models.py::LabelEmbedder` | 训练/推理 | `[B,1152]` |
| adaLN-Zero block | `models.py::DiTBlock` | 28 层主干 | `[B,256,1152]` |
| 输出头 | `models.py::FinalLayer` | 前向 | `[B,256,32]` |
| Unpatchify | `models.py::DiT.unpatchify` | 前向 | `[B,8,32,32]` |
| 加噪与训练损失 | `diffusion/gaussian_diffusion.py` | 训练 | MSE + VB loss |
| CFG | `models.py::DiT.forward_with_cfg` | 采样 | 指导后的 epsilon/variance |
| DDPM respacing | `diffusion/__init__.py` | 采样 | 1000 基础步中的 250 个离散步 |

### 11.2 DiT-XL/2 全链路张量总表

默认实例：ImageNet $256\times256$、DiT-XL/2、VAE 下采样 8 倍、patch size 2、hidden 1152、28 blocks、16 heads。

| # | 模块 | 具体输入 | 具体输出 | 说明 |
|---:|---|---|---|---|
| 1 | RGB 输入 | — | `[B,3,256,256]` | `train.py`归一化到约`[-1,1]` |
| 2 | VAE encoder | `[B,3,256,256]` | `[B,4,32,32]` | 从 posterior 采样 |
| 3 | Latent scaling | `[B,4,32,32]` | `[B,4,32,32]` | 乘`0.18215` |
| 4 | 前向加噪 | `z_0,ε,t` | `[B,4,32,32]` | 每样本随机一个 timestep |
| 5 | PatchEmbed | `[B,4,32,32]` | `[B,256,1152]` | $32/2=16$，共$16^2=256$ tokens |
| 6 | 位置编码 | `[B,256,1152]` | `[B,256,1152]` | 固定二维 sin-cos，参数形状`[1,256,1152]` |
| 7 | Timestep frequency | `[B]` | `[B,256]` | 256 维频率特征 |
| 8 | Timestep MLP | `[B,256]` | `[B,1152]` | `256→1152→1152`，SiLU |
| 9 | Class embedding | `[B]` | `[B,1152]` | 1000 类 + 1 个 null 类 |
| 10 | 条件和 | `t_emb+y_emb` | `[B,1152]` | 对全部空间 token 共享 |
| 11 | 每层调制 | `[B,1152]` | `[B,6912]` | $6\times1152$，拆为 shift/scale/gate 两组 |
| 12 | Attention | `[B,256,1152]` | `[B,256,1152]` | 16 heads，每头 72 维 |
| 13 | MLP | `[B,256,1152]` | `[B,256,4608]→[B,256,1152]` | ratio 4 |
| 14 | 28 blocks | `[B,256,1152]` | `[B,256,1152]` | token 数与宽度固定 |
| 15 | FinalLayer | `[B,256,1152]` | `[B,256,32]` | 每 token 输出$2^2\times8=32$ |
| 16 | Unpatchify | `[B,256,32]` | `[B,8,32,32]` | 恢复 latent 网格 |
| 17 | 输出拆分 | `[B,8,32,32]` | 两个`[B,4,32,32]` | epsilon + learned-range variance |

这里最容易误读的是最后 8 通道。它不是把 VAE latent 扩成 8 维，而是 4 通道噪声预测与 4 通道方差参数拼接。

### 11.3 VAE 与 latent scaling

#### 11.3.1 输入输出与内部张量流

训练脚本把`[B,3,256,256]`图像送入`stabilityai/sd-vae-ft-{ema|mse}`，从 posterior 采样`[B,4,32,32]`，再乘`0.18215`作为扩散数据$z_0$。采样结束后执行逆操作：latent 除以`0.18215`，再交给 decoder 恢复 RGB。

#### 11.3.2 为什么这么设计

VAE 把边长缩小 8 倍，空间位置从$256^2=65,536$降到$32^2=1,024$；再用$2\times2$ latent patch，只剩 256 tokens。DiT 的计算预算因此可以集中在大 Transformer，而不是消耗在像素级长序列。

#### 11.3.3 关键超参

- latent channel：4；
- 下采样率：8；
- scaling：`0.18215`，代码硬编码；
- 训练 CLI 默认 VAE：`ema`；
- `sample.py`默认 VAE：`mse`。

#### 11.3.4 代码与论文差异

论文最终 FID 2.27 使用 ft-EMA decoder，而默认`sample.py`使用 MSE VAE；训练时二者 encoder 相同，因此官方注释称 VAE 选择不影响 denoiser 训练。**默认脚本可视化结果不自动等于论文 2.27 的评估口径。**

### 11.4 Patchify、固定位置编码与主干

#### 11.4.1 输入输出与内部张量流

`PatchEmbed`把`[B,4,32,32]`变成`[B,256,1152]`。每个原始 patch 只有$2\times2\times4=16$个数，经共享投影升到 1152 维。固定二维 sin-cos 位置编码直接相加；DiT 没有 ViT 的 class token，因为输出必须对齐整张 latent 网格。

#### 11.4.2 为什么这么设计

Patch size 2 在空间细节与 token 成本之间折中。若 patch 从 4 降到 2，token 数从 64 增到 256，而主干权重几乎不变；这正是论文用来区分“参数规模”和“前向计算规模”的关键实验轴。

#### 11.4.3 关键超参

- hidden size 1152；
- 28 blocks；
- 16 heads，head dimension 72；
- MLP ratio 4，内部宽度 4608；
- LayerNorm 无 affine 参数，`eps=1e-6`；
- attention 使用`qkv_bias=True`；
- MLP 使用 tanh approximation GELU，dropout 0。

#### 11.4.4 代码与论文差异

论文写“展平 latent patch 后线性投影”，代码借用 timm`PatchEmbed`，本质是 kernel/stride 等于 patch size 的卷积，数学上等价。代码只实现最终选定的固定二维 sin-cos 与 adaLN-Zero 模型，没有保留 in-context、cross-attention 和普通 adaLN 的消融分支。

### 11.5 Timestep、类别条件与 adaLN-Zero

#### 11.5.1 输入输出与内部张量流

标量$t$先变成`[B,256]`频率编码，再经 SiLU MLP 得到`[B,1152]`；类别$y$查表得到同形状向量。二者相加成全局条件$c$。每个 block 用自己的线性层把$c$映射成`[B,6912]`，切成六个`[B,1152]`向量：attention 前的 shift/scale/gate 与 MLP 前的 shift/scale/gate。

#### 11.5.2 为什么这么设计

时间步告诉网络当前噪声水平，类别决定目标语义。把二者变成每层调制与残差 gate，比追加两个普通 token 的路径更短，又不引入 cross-attention 的 token 相关开销。零初始化 gate 让每个 block 初始近似恒等映射，深层训练更稳定。

#### 11.5.3 关键超参

- timestep frequency dim：256；
- `max_period=10000`；
- class 数：1000；
- label dropout：0.1；
- null class index：1000；
- 调制线性层与 gate 全零初始化。

#### 11.5.4 代码与论文差异

结构与论文最终 adaLN-Zero 一致。论文比较了四种条件注入，当前代码只保留最佳分支；`eps=1e-6`、tanh-GELU 等实现细节是代码具体化，不应从论文框图擅自补出。

### 11.6 FinalLayer、方差头与训练目标

#### 11.6.1 输入输出与内部张量流

FinalLayer 先对`[B,256,1152]`做条件化 LayerNorm，再线性映射为`[B,256,32]`，unpatchify 为`[B,8,32,32]`。前 4 通道是 epsilon，后 4 通道控制 learned-range log variance。

训练时每个样本随机采一个$t$，一次构造$z_t$，只执行一次 DiT forward。Epsilon 分支用 MSE；方差分支进入 variational bound 项，而且代码对 epsilon 输出`detach`后计算 VB，使 VB 主要训练方差头：

$$\mathcal{L}=\mathcal{L}_{\mathrm{MSE}}+\mathcal{L}_{\mathrm{VB}}$$

#### 11.6.2 为什么这么设计

沿用 ADM/Improved-DDPM 的成熟目标，可以把论文实验差异集中归因于 denoiser backbone，而不是新损失。Learned-range variance 在 posterior clipped log variance 与$\log\beta_t$之间插值，比完全自由预测方差更稳定。

#### 11.6.3 关键超参

- 基础扩散步数：1000；
- linear beta schedule：$10^{-4}\rightarrow2\times10^{-2}$；
- mean type：epsilon；
- variance type：learned range；
- loss type：MSE + VB；
- FinalLayer 调制与输出线性层全零初始化。

#### 11.6.4 代码与论文差异

论文概括为“预测噪声和对角协方差”，代码给出更具体的 learned-range 插值实现。当前`create_diffusion`选择普通`MSE`而非`RESCALED_MSE`；这是理解实际 loss 权重时必须看代码才能确认的细节。

### 11.7 CFG 的真实代码路径

`sample.py`先生成$N$个噪声 latent，再复制一份并拼接 null class，形成条件/无条件的$2N$ batch。模型一次 batched forward 后，当前活动代码执行：

- `eps = model_out[:, :3]`；
- `rest = model_out[:, 3:]`；
- 只对前三个 epsilon latent 通道做 CFG；
- 第 4 个 epsilon 通道与 4 个方差通道都留在`rest`中，不做 guidance。

因此“只指导前三通道”绝不是对 RGB 三通道做 CFG，而是对**四维 VAE latent 的 epsilon 预测前三维**做 guidance。完整四通道 guidance 的写法仍以注释保留。

默认`sample.py`配置：250 个 respaced DDPM steps、CFG 4.0、MSE VAE。论文 FID 2.27 的配置则是 CFG 1.5 和 ft-EMA decoder。正式复现指标时应使用`sample_ddp.py`生成 50K 样本，再用 ADM TensorFlow evaluation suite 评估，不能用默认 8 张 sample grid 代替 FID。

### 11.8 训练与推理计算账

论文给 DiT-XL/2@256 的单样本前向计算量为 118.64 Gflops。训练时每个样本只随机一个 timestep，不会跑 1000 次网络。用论文粗估：

$$\text{训练计算}\approx118.64\ \text{Gflops}\times B\times\text{steps}\times3$$

默认 global batch 256 时，约为 91.12 Tflops/step；400K steps 约 36.45 EFLOPs，7M steps 约 637.81 EFLOPs。该估算不含 VAE、优化器、EMA、DDP 通信和硬件利用率。

推理时 250 steps：

- 无 CFG：$118.64\times250\approx29.66$ Tflops/图；
- CFG 条件/无条件双分支：算术量近似 59.32 Tflops/图；
- CFG 虽合并成一次$2N$ batch forward，但计算量并没有消失。

### 11.9 最小复现建议

最小预训练权重采样：

```bash
python sample.py \
  --image-size 256 \
  --num-sampling-steps 250 \
  --cfg-scale 4.0 \
  --seed 0
```

更接近论文 FID 2.27 的视觉采样口径：

```bash
python sample.py \
  --image-size 256 \
  --num-sampling-steps 250 \
  --cfg-scale 1.5 \
  --vae ema \
  --seed 0
```

最小 DDP 训练：

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --global-batch-size 256 \
  --data-path /path/to/imagenet/train
```

复现时应明确：

1. CUDA + NCCL 是训练脚本硬要求；
2. ImageNet 需要`ImageFolder`目录；
3. 默认训练 1400 epochs，真实 step 数取决于数据规模；
4. checkpoint 含 model、EMA、optimizer 与 args，评估优先使用 EMA；
5. 官方 README 只声称 PyTorch 在数十万步范围复现 JAX 趋势，不等于重跑全部 7M-step 旗舰实验；
6. TF32、VAE decoder、CFG 通道数与 scale 都会让最终数字偏离论文。

### 11.10 代码与论文最关键的差异汇总

| 论文口径 | 官方 PyTorch 代码事实 |
|---|---|
| 原始 JAX/TPU 训练 | 仓库为官方 PyTorch 对应实现，权重从 JAX 移植 |
| DiT 预测噪声与方差 | 具体为 4 epsilon + 4 learned-range variance |
| CFG 作为条件/无条件组合 | 活动代码只指导 epsilon 前 3 个 latent 通道 |
| 最终 FID 使用 CFG 1.5 | 默认`sample.py`为 CFG 4.0 |
| 最终结果使用 ft-EMA decoder | 默认采样脚本使用 MSE VAE |
| 250 sampling steps | 代码在 1000-step 基础 schedule 上做 250-step respacing |
| 训练过程 | 每样本随机一个$t$、每 step 只跑一次 DiT，而非 1000 次 forward |
| 论文完整训练系统 | 代码是最小 DDP 脚本，缺少 AMP、断点续训、自动 FID 等工程功能 |

---

## 12. 关键术语表

- **DiT**：Diffusion Transformer，以 ViT 风格 Transformer 作为扩散去噪骨干。
- **Latent diffusion**：在 autoencoder latent space 而非 RGB pixel space 中执行扩散与去噪。
- **Patchify**：把二维 latent 划分为非重叠 patch，并映射为 token 序列。
- **adaLN**：由条件向量生成 LayerNorm 的逐通道 scale 与 shift。
- **adaLN-Zero**：在 adaLN 上增加 residual gate，并用零初始化使 block 初始近似恒等映射。
- **CFG**：Classifier-Free Guidance，组合条件与无条件预测，形成质量—多样性权衡。
- **FID-50K**：用 50,000 个生成样本计算的 Fréchet Inception Distance，越低越好。
- **Gflops**：一次理论前向的十亿浮点操作数；本文用它近似 DiT backbone complexity。
- **SiT**：在 DiT 类骨干上研究 stochastic interpolant、flow/diffusion 目标与采样器。
- **MMDiT**：使用模态专属参数与联合注意力连接文本、图像 token 的多模态扩散 Transformer。

## 13. 图表索引

| 图 | 本地文件 | 内容 |
|---|---|---|
| 图 1 | `figures/01-architecture-conditioning.png` | latent DiT 总体 I/O，以及 adaLN-Zero、cross-attention、in-context block |
| 图 2 | `figures/02-conditioning-ablation.png` | 四种条件注入方式的 FID 训练曲线 |
| 图 3 | `figures/03-gflops-fid.png` | Gflops/模型规模与 FID，以及和 U-Net diffusion 基线的计算比较 |
| 图 4 | `figures/04-scaling-curves.png` | 固定 patch 或固定 backbone 的 12 模型 scaling 曲线 |
| 图 5 | `figures/05-visual-scaling.png` | 相同噪声下扩大模型与减小 patch 的定性质量变化 |

五张图均由官方 arXiv v2 TeX 源包中的矢量 PDF 等比例栅格化，未重绘、未修改实验数据；原始矢量文件、尺寸、SHA-256 和逐图核验记录见 [[assets/DiT - 可扩展扩散Transformer/sources.md|sources.md]]。

## 14. 来源

- Peebles, W.; Xie, S. *Scalable Diffusion Models with Transformers*. arXiv:2212.09748 / CVPR 2023.
- 官方摘要页：https://arxiv.org/abs/2212.09748
- 官方 PDF：https://arxiv.org/pdf/2212.09748
- 官方 TeX 源包：https://arxiv.org/e-print/2212.09748
- 官方代码：https://github.com/facebookresearch/DiT
- 固定代码提交：https://github.com/facebookresearch/DiT/commit/ed81ce2229091fd4ecc9a223645f95cf379d582b
- 本地代码核验记录：[[assets/DiT - 可扩展扩散Transformer/sources.md]]
- 【后世关系】Ma et al. *SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers*. arXiv:2401.08740.
- 【后世关系】Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. arXiv:2403.03206.
