---
type: paper-note
status: complete
domain: Foundations
paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
year: 2021
arxiv: "2010.11929"
doi: "10.48550/arXiv.2010.11929"
source: "https://arxiv.org/abs/2010.11929"
project: "https://openreview.net/forum?id=YicbFdNTTy"
code: "https://github.com/google-research/vision_transformer"
code_commit: "64801f1b3b367b3611cc27a3d45cc22870a36fb3"
tags:
  - Foundations
  - Vision-Transformer
  - Representation-Learning
  - Image-Classification
created: 2026-10-30
updated: 2026-10-30
---

# ViT：图像分块与纯 Transformer 视觉建模

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* |
| 作者 | Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby |
| 会议 | ICLR 2021 |
| 版本 | arXiv:2010.11929v2，2021-06-03 |
| 论文 | [arXiv](https://arxiv.org/abs/2010.11929) · [[assets/ViT - 图像分块与纯Transformer视觉建模/sources/paper.pdf\|本地 PDF]] |
| 官方代码 | [google-research/vision_transformer](https://github.com/google-research/vision_transformer)，Apache-2.0，固定 commit `64801f1b3b367b3611cc27a3d45cc22870a36fb3` |
| 开源结论 | **官方代码已开源**；该仓库主要提供 JAX/Flax 模型、公开 checkpoint 与微调代码，原始大规模预训练主要在 `big_vision`，JFT-300M 数据不公开 |
| 本地材料 | [[assets/ViT - 图像分块与纯Transformer视觉建模/sources.md\|来源与代码核验记录]] |
| 个人评级 | ★★★★★ |

## 1. 一句话结论

ViT 的关键不是“在 CNN 后面加注意力”，而是把图像直接切成 patch token，几乎原样交给标准 Transformer Encoder；它证明**卷积不是视觉识别的必要骨干**，但同时揭示了一个重要交换：弱化局部性和平移等变等视觉归纳偏置后，模型会更依赖大规模预训练数据，而一旦数据与算力足够，纯 Transformer 的规模扩展和迁移效率反而更强。

---

## 2. 研究背景与问题定义

### 2.1 论文要推翻的默认前提

2020 年前，视觉注意力大致有两类用法：一类把 attention 嵌进卷积网络；另一类替换 CNN 的某些组件，但仍保留局部卷积、特征金字塔或分层下采样。论文提出一个更激进的问题：**图像识别是否真的需要卷积骨干，还是可以像 NLP 一样，把输入变成 token 序列后直接使用标准 Transformer？**

困难在于图像和文本并不同构。文本天然是一维离散序列，而图像是高分辨率二维连续信号；如果把每个像素当 token，$224\times224$ 图像会产生 50,176 个 token，全局 self-attention 的二次复杂度不可接受。因此 ViT 首先不是设计新 attention，而是设计一个合适的视觉 tokenization：用较大的非重叠 patch 把序列长度压到可管理范围。

### 2.2 CNN 归纳偏置与 ViT 的赌注

CNN 把三类视觉先验写进每一层：局部感受野、二维邻域和平移等变。ViT 除了输入端切 patch、微调时二维插值位置编码，主干几乎不显式编码二维结构。它押注的是：

1. 大规模数据可以让模型自己学出局部和全局视觉规律；
2. 标准 Transformer 的全局信息交换、成熟实现和扩展经验，长期收益会超过手工视觉先验；
3. 预训练—迁移范式能够把大数据学到的通用表征转移到中小规模任务。

这个判断并不是“ViT 在任何数据规模都优于 CNN”。论文真正证明的是：**归纳偏置与数据规模存在交换关系**。数据少时 CNN 更有优势；数据足够大时，较弱先验带来的表达自由度和可扩展性开始占优。

### 2.3 核心贡献

1. **提出纯 Transformer 图像分类骨干。** 图像经 patchify 和线性投影后，直接输入标准 pre-LN Transformer，不依赖卷积主干。
2. **建立视觉 patch token 范式。** Patch size 同时控制空间粒度、序列长度和计算量，后来成为 MAE、DINO、CLIP、DiT 和大量 VLM 的共同接口。
3. **系统研究数据与规模效应。** ImageNet、ImageNet-21k、JFT-300M 的对照显示，大模型只有在足够大的预训练数据上才释放优势。
4. **证明迁移效率。** 在相近迁移性能下，ViT 在论文实验中大约使用 ResNet 的 $1/2$ 到 $1/4$ 预训练计算。
5. **展示 Transformer 的视觉表征自组织。** 学到的位置编码恢复二维邻近关系，不同 attention head 在浅层同时呈现局部与长程模式。

---

## 3. 任务定义与输入输出

### 3.1 图像如何变成 token

输入图像记为$x\in\mathbb{R}^{H\times W\times C}$，patch 边长为$P$。图像被切成

$$N=\frac{HW}{P^2}$$

个非重叠 patch，每个 patch 展平为$x_p^i\in\mathbb{R}^{P^2C}$，再用共享线性映射$E\in\mathbb{R}^{(P^2C)\times D}$投到隐藏维$D$。前面拼接一个可学习的 class token，并加可学习绝对位置编码：

$$z_0=[x_{\mathrm{class}};x_p^1E;x_p^2E;\ldots;x_p^NE]+E_{\mathrm{pos}}$$

其中$E_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times D}$。

以最典型的$224\times224$、ViT-B/16 为例：

- $P=16$，patch 网格为$14\times14$；
- patch token 数$N=196$；
- 加 class token 后序列长度为 197；
- 每个原始 patch 是$16\times16\times3=768$维，恰好映射到$D=768$。

### 3.2 分类输出

Transformer 最后一层的 class token 经 LayerNorm 作为图像表示：

$$y=\operatorname{LN}(z_L^0)$$

预训练时论文使用带一层隐藏层的 MLP head；迁移到下游任务时，移除预训练 head，换成一个$D\times K$的零初始化线性分类器，其中$K$为下游类别数。

### 3.3 更高分辨率微调

预训练通常使用$224^2$，微调常升到$384^2$或更高。Patch size 保持不变，因此 ViT-B/16 在$384^2$时从$14\times14=196$个 patch 增长为$24\times24=576$个 patch，含 class token 共 577 个。主干权重可以直接复用，但位置编码长度不匹配，官方做法是把去掉 class token 后的位置向量按二维网格插值，再拼回 class token 位置。

---

## 4. 核心方法

### 4.1 总体架构

![](assets/ViT%20-%20图像分块与纯Transformer视觉建模/figures/01-architecture.png)

> **图 1：ViT 总体架构。** 左侧把图像切成 patch、线性投影、加入位置与 class token；右侧是标准 pre-LN Transformer Encoder，依次执行 LayerNorm、Multi-Head Attention、残差，再执行 LayerNorm、MLP、残差。来源：论文 Figure 1，PDF p.3，[官方 PDF](https://arxiv.org/pdf/2010.11929)。

整条 pipeline 是：

$$\text{RGB图像}\rightarrow\text{非重叠patch}\rightarrow\text{线性投影}\rightarrow\text{位置编码+class token}\rightarrow L\text{层Transformer}\rightarrow\text{分类head}$$

它没有 CNN 常见的逐级降采样和多尺度特征金字塔。所有 token 从第一层到最后一层都保持固定宽度$D$和固定序列长度，空间信息完全由 token 内容、位置编码和 self-attention 学习。

### 4.2 Pre-LN Transformer block

第$\ell$层先做多头自注意力：

$$z'_\ell=\operatorname{MSA}(\operatorname{LN}(z_{\ell-1}))+z_{\ell-1}$$

再做两层 GELU MLP：

$$z_\ell=\operatorname{MLP}(\operatorname{LN}(z'_\ell))+z'_\ell$$

LayerNorm 放在子层之前，残差在子层之后。这是 pre-LN 结构，有利于深层 Transformer 的梯度传播。ViT-B 的 MLP 从 768 扩张到 3072，再投回 768；因此在$224^2$下，主要计算其实来自 MLP 和 QKV/输出线性投影，attention 的$N^2$项尚未占主导。分辨率提高或 patch 变小时，序列长度增加，attention 二次项才快速放大。

### 4.3 Multi-Head Self-Attention

对输入序列$z\in\mathbb{R}^{N\times D}$：

$$[q,k,v]=zU_{qkv}$$

$$A=\operatorname{softmax}\left(\frac{qk^\top}{\sqrt{D_h}}\right)$$

$$\operatorname{SA}(z)=Av$$

多个 head 独立计算，再拼接并投影：

$$\operatorname{MSA}(z)=[\operatorname{SA}_1(z);\ldots;\operatorname{SA}_k(z)]U_{\mathrm{msa}}$$

ViT-B/16 使用 12 个 head，每头 64 维。每个 patch 从第一层就可以读取任意其他 patch，这与 CNN 依靠堆叠局部卷积逐步扩大感受野不同。代价是缺少内置局部先验，也使高分辨率的 token 成本更高。

### 4.4 Patch size 是视觉版的 token 粒度旋钮

模型名`ViT-L/16`表示 Large 规格、$16\times16$ patch。Patch 边长减半会让 token 数变为四倍：

| 输入 | Patch | Patch token 数 | 含 class token |
|---|---:|---:|---:|
| $224^2$ | 32 | 49 | 50 |
| $224^2$ | 16 | 196 | 197 |
| $224^2$ | 14 | 256 | 257 |
| $384^2$ | 16 | 576 | 577 |
| $512^2$ | 16 | 1024 | 1025 |

更小 patch 保存更细空间信息，通常提升精度，但同时增加 attention、MLP 激活和显存。论文附录发现，减小 patch size 在几乎不增加主干参数的情况下稳定提升表现；这与后来的 DiT scaling 逻辑高度一致：**参数量不是唯一容量指标，token 数和单次前向计算同样重要。**

### 4.5 位置编码为什么只用一维可学习向量

论文比较无位置编码、一维绝对、二维分解和相对位置编码。ViT-B/16 的 ImageNet 5-shot 结果显示：完全没有位置编码明显较差，但一维、二维和相对位置编码之间差异很小。作者推测 patch 网格只有$14\times14$，学习二维关系并不困难，因此简单的一维绝对位置编码已经够用。

这不是说二维结构不重要。相反，位置编码可视化显示模型自行恢复了行、列和邻近关系；结论应是：在粗 patch 网格上，不必把复杂二维先验硬编码进每一层。

### 4.6 学到的视觉表征

![](assets/ViT%20-%20图像分块与纯Transformer视觉建模/figures/04-learned-representations.png)

> **图 7：ViT 学到的内部表征。** 左：patch projection 的主成分像局部纹理与颜色基；中：位置编码相似度自动呈现二维邻近与行列结构；右：浅层同时存在局部 head 和跨大范围 head，随网络加深平均注意距离增大。来源：论文 Figure 7，PDF p.9，[官方 PDF](https://arxiv.org/pdf/2010.11929)。

这张图直接反驳“没有卷积就不会学局部结构”的简单判断。ViT 不是没有局部性，而是**局部性由数据学习，而非架构强制**。部分浅层 head 的平均注意距离很小，功能类似卷积局部核；另一些浅层 head 已跨越大部分图像，提供 CNN 需要很多层才能获得的全局交互。混合 CNN+ViT 模型的浅层局部 head 更少，也侧面说明卷积 stem 已提前完成局部特征提取。

---

## 5. 数据集、训练与实验设置

### 5.1 预训练数据

| 数据集 | 规模 | 类别 | 可公开性 |
|---|---:|---:|---|
| ImageNet-1K | 约 1.3M | 1K | 公开 |
| ImageNet-21k | 约 14M | 21K | 公开 |
| JFT-300M | 约 303M | 18K | Google 内部 |

下游覆盖 ImageNet/ReaL、CIFAR-10/100、Oxford-IIIT Pets、Flowers-102 和 VTAB-19。VTAB 每个任务只提供 1,000 个训练样本，用于衡量低数据迁移。

### 5.2 模型规格

| 模型 | 层数 | Hidden $D$ | MLP | Heads | 参数量 |
|---|---:|---:|---:|---:|---:|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

Base 和 Large 直接借鉴 BERT 规格，Huge 是视觉实验新增的更大版本。论文不仅扩深和扩宽，也通过/32、/16、/14改变 token 数。

### 5.3 训练 recipe

预训练统一使用 Adam，$\beta_1=0.9$、$\beta_2=0.999$、batch size 4096，主设置 weight decay 0.1，线性 warmup 10K steps。所有预训练分辨率为$224^2$。具体数据集还会调节学习率、训练轮数、dropout、label smoothing 与 weight decay。

微调使用 SGD momentum 0.9、batch 512、cosine 学习率衰减、无 weight decay、global gradient clipping 1，通常在$384^2$进行。Table 2 的 ImageNet 最强结果更进一步：ViT-L/16 用$512^2$，ViT-H/14 用$518^2$，并使用 0.9999 Polyak averaging。因此，88.55% 不是普通$224^2$单阶段训练的结果。

---

## 6. 实验结果

### 6.1 大规模预训练后的旗舰结果

| 模型与预训练 | ImageNet | ImageNet-ReaL | CIFAR-100 | VTAB-19 | TPUv3-core-days |
|---|---:|---:|---:|---:|---:|
| ViT-H/14，JFT-300M | **88.55** | **90.72** | **94.55** | **77.63** | 2.5K |
| ViT-L/16，JFT-300M | 87.76 | 90.54 | 93.90 | 76.28 | 0.68K |
| ViT-L/16，ImageNet-21k | 85.30 | 88.62 | 93.25 | 72.72 | 0.23K |
| BiT-L R152x4，JFT-300M | 87.54 | 90.54 | 93.51 | 76.29 | 9.9K |

ViT-L/16 在同样 JFT 预训练下全面超过 BiT-L，却只用约$0.68/9.9\approx6.9\%$的 TPUv3-core-days。这个差距不能全部归因于架构，因为优化器、训练日程和工程实现也不同；论文因此又做了受控 compute scaling 对比。

### 6.2 数据规模决定大模型是否真正受益

![](assets/ViT%20-%20图像分块与纯Transformer视觉建模/figures/02-data-scaling.png)

> **图 3：预训练数据规模效应。** ImageNet 规模下，ViT-L 甚至弱于 ViT-B 与 BiT；到 ImageNet-21k 后差距缩小；JFT-300M 上更大的 ViT-L/H 才明显释放优势。来源：论文 Figure 3，PDF p.7，[官方 PDF](https://arxiv.org/pdf/2010.11929)。

这是论文最重要、也最容易被后来“ViT 取代 CNN”口号掩盖的结果。ViT 的弱视觉先验意味着样本效率更差；在 9M 级数据上，ViT-B/32 甚至明显落后于计算量相近的 ResNet50。但 ResNet 随数据增大更早饱和，ViT 在 90M 以上反超。换句话说，ViT 的优势不是免费得到的，而是用数据和规模换来的。

### 6.3 性能—计算量扩展

![](assets/ViT%20-%20图像分块与纯Transformer视觉建模/figures/03-compute-scaling.png)

> **图 5：迁移性能与预训练计算。** 在论文控制的 JFT 预训练实验中，ViT 的 Pareto 曲线整体优于 ResNet；小模型阶段混合架构略占优，但规模扩大后差距消失。来源：论文 Figure 5，PDF p.7，[官方 PDF](https://arxiv.org/pdf/2010.11929)。

论文报告达到同等迁移性能时，ViT 大约只需 ResNet 的$1/2$到$1/4$预训练计算。附录进一步显示：增加深度的收益最大且可持续到很深；减小 patch size 也稳定有效；单纯增宽收益最小。这里已经出现后来 Transformer scaling 的核心经验：**按比例扩展通常可靠，depth 与 token granularity 往往比只堆 width 更划算。**

### 6.4 位置编码、class token 与自监督消融

- **位置编码**：没有位置编码明显变差；一维、二维与相对位置编码接近。
- **Class token vs GAP**：最初 GAP 表现差主要是学习率没调对；正确调参后 class token 与 GAP 接近，因此 class token 不是性能来源的充分解释。
- **初步自监督**：masked patch prediction 的 ViT-B/16 达 ImageNet 79.9%，比从头监督训练高约 2 点，但仍比大规模监督预训练低约 4 点。后来 MAE 证明“masked patch”路线远未被原论文充分开发。

### 6.5 实验是否支撑主张

论文充分支撑三点：纯 Transformer 可以做好视觉分类；弱归纳偏置需要更大数据；ViT 在大规模预训练下具有良好迁移与 compute scaling。但它没有证明 ViT 在所有视觉任务、所有数据预算、所有硬件上都优于 CNN，也没有从头训练出一个公开数据可完全复现的旗舰结果。

---

## 7. 与相关工作的关系

### 7.1 与 CNN 的关系

ViT 不是“把卷积计算得更快”，而是改变表示接口：CNN 以二维 feature map 和局部核为中心，ViT 以 token sequence 和全局 attention 为中心。它牺牲局部先验换取统一结构、全局交互和跨模态兼容性。

### 7.2 与后续自监督视觉 Transformer

原始 ViT 主要依赖监督大数据；MAE、DINO、iBOT、BEiT 等工作解决了“如何不用 JFT 标签学到强 ViT 表征”。ViT 提供骨干，后续工作的主要创新转移到训练目标、teacher-student 机制、mask 策略、数据与正则化。

### 7.3 与 CLIP/VLM

ViT 的 patch token 让图像天然进入 Transformer 生态。CLIP 可以把 ViT 作为视觉编码器与文本编码器做对比学习；后续 VLM 又把视觉 token 投影进 LLM。ViT 的历史意义不只是分类精度，而是把视觉从 feature map 接口迁移到 token 接口。

### 7.4 与 DiT

[[DiT - 可扩展扩散Transformer]]几乎复用了 ViT 的核心骨架：patchify、固定宽度 Transformer、全局 self-attention、深度/宽度/patch size scaling。区别是 DiT 的输入是带噪 VAE latent，没有 class token，输出要恢复空间 latent，并通过 adaLN-Zero 注入 timestep 与类别条件。可以把 DiT 理解为“**ViT 骨干从判别式图像 token 编码器，改造成条件扩散去噪器**”。

---

## 8. 局限与批判性评价

### 8.1 旗舰结果依赖不可公开数据

JFT-300M 是内部数据，最强 ViT-H/14 无法由公众严格复现。ImageNet-21k 版本虽然公开可做，但与 88.55% 旗舰仍有显著差距。论文证明了“在这套大数据上成立”，却没有提供完全公开的等价规模基准。

### 8.2 小数据样本效率差

弱归纳偏置带来扩展自由，也带来更强数据饥渴。论文自己的受控实验清楚表明，小数据下 ViT 会过拟合并落后于 CNN。这一问题后来由 DeiT 的蒸馏和强数据增强、MAE/DINO 的自监督、ConvNeXt/Hybrid 等路线缓解，但不是原始 ViT 已经解决的问题。

### 8.3 Patchify 会损失细粒度边界

非重叠大 patch 在第一层就把$P\times P$像素压到一个 token。分类任务可以容忍这种压缩，检测、分割、深度等 dense prediction 更依赖边缘与局部细节，通常需要更小 patch、多尺度层级、卷积 stem 或专门 decoder。原论文只系统验证分类与迁移，不能直接外推到所有视觉任务。

### 8.4 全局 attention 的高分辨率成本

序列长度$N=HW/P^2$，全局 attention 具有$O(N^2D)$项。$224^2$、/16 时只有 197 tokens，二次项不大；升到高分辨率或视频后，token 数会迅速成为瓶颈。Swin、PVT、Hiera、稀疏 attention 与 token pruning 等后续工作本质上都在补 ViT 的空间效率。

### 8.5 对比并非完全等训练配方

论文承认预训练效率还会受到优化器、正则化、训练时长与实现影响。Table 2 还叠加高分辨率微调和 Polyak averaging，因此“ViT 架构天然省 4 倍算力”不应脱离具体实验口径。更稳妥的表述是：在论文控制的 JFT scaling study 中，ViT 显示出更好的性能—计算 Pareto 前沿。

### 8.6 注意力图不等于因果解释

Figure 6/7 说明模型关注语义区域并学习局部/全局模式，但 attention weight 不能直接当作模型决策因果解释。它更适合证明“模型确实使用全局交互能力”，而不是证明某个区域必然导致分类结果。

---

## 9. 复现与实践建议

### 9.1 三种复现目标要分开

1. **结构复现**：随机输入跑通 ViT-B/16，核对 token 与参数量；普通单卡即可。
2. **公开 checkpoint 微调**：加载官方 ImageNet-21k 权重，在 CIFAR 或 ImageNet 下游微调；这是官方仓库最直接支持的路径。
3. **论文级预训练复现**：重跑 JFT-300M、ViT-H/14 和高分辨率迁移；数据和 TPU 成本使其基本不可行。

不要把第 2 类写成第 3 类。官方 `vision_transformer` 仓库 README 也明确指出：原始模型预训练主要发生在 `big_vision`，当前仓库重点是公开模型与微调。

### 9.2 工程检查点

- 输入尺寸必须和 patch size 对齐；官方 patch Conv 使用`VALID`，不整除时边缘会被舍弃。
- 改分辨率时要二维插值位置编码，不能简单截断或一维拉伸。
- 加载下游任务时移除整个预训练 head，并使用零初始化线性 head。
- 报告结果时同时注明：预训练数据、模型规格、patch size、微调分辨率、是否 Polyak averaging。
- 小数据训练不要照搬 JFT recipe；需要更强增强、正则化、蒸馏或自监督预训练。

---

## 10. 个人启发与后续问题

### 10.1 ViT 真正统一的是“接口”

ViT 最深远的贡献不是 88.55% 本身，而是把图像变成一等公民 token。此后表征学习、VLM、生成模型、世界模型和 VLA 都能围绕统一的 Transformer 工具链工作。对三维问题同样如此：point、Gaussian、mesh face、UV chart 或局部几何块一旦 token 化，就可以共享 attention、masking、JEPA 和 diffusion 的建模语言。

### 10.2 Tokenizer 决定 Transformer 看见什么

ViT 常被概括成“Transformer 处理 patch”，但 patchify 不是无害预处理。$P$决定信息压缩、局部边界、序列长度和计算账。对于 3D asset，类似问题会变成：token 应按规则网格、局部曲面、拓扑邻域、Gaussian ellipsoid 还是语义 part 切分？骨干统一后，**tokenization 与邻域定义反而成为最关键的归纳偏置来源**。

### 10.3 Scaling 不等于只堆参数

ViT 附录和 DiT 主论文都指向同一结论：增加 token 数可以在参数几乎不变时提升质量，计算量往往比参数量更能解释表现。研究 3DGS grouping 或 Asset-JEPA 时，也不应只报 backbone 参数量，还要报告 primitive/token 数、邻域规模、attention FLOPs 与分辨率。

---

## 11. 代码级实现

### 11.1 仓库概览与开源边界

- 官方仓库：https://github.com/google-research/vision_transformer
- 固定 commit：`64801f1b3b367b3611cc27a3d45cc22870a36fb3`
- 框架：JAX + Flax
- 许可证：Apache-2.0
- 公开内容：ViT/Hybrid/Mixer 模型、checkpoint 加载、推理、下游微调、公开 checkpoint。
- 关键边界：原始多主机预训练与部分配置在`google-research/big_vision`；JFT-300M 不公开。

| 论文模块 | 代码文件/类 | 运行阶段 | 产物 |
|---|---|---|---|
| Patch projection | `vit_jax/models_vit.py::VisionTransformer.__call__` | 前向 | `[B,H/P,W/P,D]` patch feature |
| Class token | `VisionTransformer.__call__` | 前向 | `[B,N+1,D]` 序列 |
| 位置编码 | `AddPositionEmbs`、`Encoder` | 前向 | 加位置后的 token |
| MSA + MLP | `Encoder1DBlock`、`MlpBlock` | 每层 | 同形状 token |
| Encoder 堆叠 | `Encoder` | 前向 | `[B,N+1,D]` |
| 分类 head | `VisionTransformer.__call__` | 预训练/微调 | `[B,K]` logits |
| 位置插值与 head 替换 | `vit_jax/checkpoint.py` | checkpoint 加载 | 新分辨率/新类别参数 |
| 微调 | `vit_jax/train.py` | 下游训练 | SGD 更新后的 checkpoint |

### 11.2 ViT-B/16 全链路张量总表

默认实例：$224\times224$ RGB、ViT-B/16、ImageNet-1K，$D=768$、12 层、12 heads、MLP 3072。

| # | 模块 | 输入 | 输出 | 具体说明 |
|---:|---|---|---|---|
| 1 | RGB 输入 | — | `[B,224,224,3]` | JAX 使用 NHWC |
| 2 | Patch Conv | `[B,224,224,3]` | `[B,14,14,768]` | kernel=stride=16，等价于共享 patch 线性层 |
| 3 | Flatten | `[B,14,14,768]` | `[B,196,768]` | 行优先展开空间网格 |
| 4 | Class token | `[B,196,768]` | `[B,197,768]` | 参数原形状`[1,1,768]`，按 batch tile |
| 5 | Position embedding | `[B,197,768]` | `[B,197,768]` | 参数`[1,197,768]` |
| 6 | Q/K/V | `[B,197,768]` | 各`[B,197,12,64]` | 12 heads，每头 64 |
| 7 | Attention map | Q/K | `[B,12,197,197]` | 每层全局 token-token 关系 |
| 8 | Attention output | `[B,197,12,64]` | `[B,197,768]` | 拼接后输出投影 |
| 9 | MLP | `[B,197,768]` | `[B,197,3072]→[B,197,768]` | GELU，两层全连接 |
| 10 | 12 blocks | `[B,197,768]` | `[B,197,768]` | 固定序列长度与宽度 |
| 11 | Final LN | `[B,197,768]` | `[B,197,768]` | Encoder 末端归一化 |
| 12 | Token pooling | `[B,197,768]` | `[B,768]` | 默认取`x[:,0]` |
| 13 | Head | `[B,768]` | `[B,1000]` | 当前配置为零初始化线性层 |

### 11.3 Patch embedding 与 class token

#### 11.3.1 输入输出与内部张量流

`VisionTransformer.__call__`用`nn.Conv`完成 patch projection。对于非重叠 patch，卷积 kernel 与 stride 都是 16，因此每个输出位置只覆盖自己的$16\times16\times3$像素块，数学上等价于论文的“flatten + 共享线性矩阵”。随后 reshape 为 196 个 token，前置一个可学习 class token。

#### 11.3.2 为什么这样设计

Conv 实现可以直接调用高效图像 kernel，而不需要显式`im2col`和 reshape；class token 让整个 encoder 保持标准序列建模接口，分类信息通过 attention 汇聚到一个位置。

#### 11.3.3 关键超参

- ViT-B/16：patch 16、hidden 768；
- class token：`[1,1,768]`，代码中零初始化；
- `padding='VALID'`，因此输入最好整除 patch size。

#### 11.3.4 代码与论文差异

论文写显式线性投影，代码用 Conv；二者在非重叠设置下等价。代码还支持`gap`、`unpooled`和`token_unpooled`分类器，而主论文默认 class token。论文附录证明 GAP 调好学习率后可接近 class token。

### 11.4 位置编码与分辨率迁移

#### 11.4.1 输入输出与内部张量流

`AddPositionEmbs`维护`[1,N+1,D]`参数，与 token 逐元素相加。高分辨率微调时，`checkpoint.py`单独保留 class 位置，把 patch 位置编码 reshape 为方形网格，用二维线性插值变到新网格，再拼回 class 位置。

#### 11.4.2 为什么这样设计

绝对位置编码简单，且论文表明在$14\times14$级 patch 网格上，复杂二维或相对位置方案未带来明显收益。二维插值则让主干可以适配更高分辨率而不重学全部空间关系。

#### 11.4.3 关键超参

- 初始化标准差 0.02；
- B/16@224：`[1,197,768]`；
- B/16@384：`[1,577,768]`；
- 插值代码要求旧、新 patch 网格均可恢复为正方形。

#### 11.4.4 代码与论文差异

论文抽象地说 ViT 可处理任意序列长度；当前官方 checkpoint 插值实现实际偏向方形图像网格。位置插值不在模型 forward，而发生在加载 checkpoint 时。

### 11.5 Transformer block

#### 11.5.1 输入输出与内部张量流

每层输入输出均为`[B,197,768]`。先 LayerNorm，再由 Flax`MultiHeadDotProductAttention`生成 12 个 64 维 head，attention 权重为`[B,12,197,197]`；输出投影后加残差。第二支先 LayerNorm，再执行`768→3072→768`的 GELU MLP并加残差。

#### 11.5.2 为什么这样设计

固定宽度与标准 pre-LN block 最大化复用 NLP Transformer 的实现与扩展经验。全局 attention 让浅层即可连接远距离 patch，宽 MLP 则承担主要通道混合容量。

#### 11.5.3 关键超参

- depth 12；
- heads 12；
- head dim 64；
- MLP ratio 4；
- 当前`get_b16_config()`的 attention dropout 与普通 dropout 均为 0。

#### 11.5.4 代码与论文差异

结构与论文一致。论文不同数据集预训练时会使用非零 dropout，例如 ImageNet-21k 的部分配置为 0.1；当前仓库模型结构配置默认 0，具体训练正则需要由训练配置覆盖。Attention 公式在论文中显式展开，代码委托给 Flax primitive。

### 11.6 分类头与微调

#### 11.6.1 输入输出与内部张量流

默认取最终 class token`[B,768]`。代码保留可选`Dense+tanh`的`pre_logits`，但当前 B/16 配置`representation_size=None`，因此直接进入零初始化线性分类 head，输出`[B,K]`。

#### 11.6.2 为什么这样设计

迁移时丢弃整个旧 head，避免预训练标签空间的特定结构干扰新任务；零初始化使初始 logits 为零，刚开始的梯度主要塑造新 head，再逐渐调整 backbone。

#### 11.6.3 关键超参

论文微调使用 SGD momentum 0.9、batch 512、cosine decay、无 weight decay、global clip 1，通常 crop 384。

#### 11.6.4 代码与论文差异

论文预训练 head 是一层隐藏 MLP，微调 head 是单线性层；当前公开 B/16 配置默认没有隐藏`pre_logits`。官方仓库`train.py`主要负责加载预训练`.npz`后做下游 SGD 微调，不是 JFT 原始 Adam 预训练脚本。

### 11.7 参数量与计算账

ViT-B/16@224、1000 类 head 的代码级参数估算：

| 模块 | 参数量 |
|---|---:|
| Patch embedding | 590,592 |
| Class token | 768 |
| 位置编码 | 151,296 |
| 单个 attention | 2,362,368 |
| 单个 MLP | 4,722,432 |
| 单 block 两个 LN | 3,072 |
| 12 blocks | 85,054,464 |
| Final LN | 1,536 |
| 1000 类 head | 769,000 |
| **总计** | **86,567,656** |

这与论文约 86M 一致。按一次乘加记一个 MAC、忽略 LayerNorm/softmax/GELU：

| 部分 | 单图 MACs |
|---|---:|
| Patch embedding | 0.116G |
| 12 层 QKV 与输出投影 | 5.577G |
| 12 层$QK^\top$与$AV$ | 0.715G |
| 12 层 MLP | 11.155G |
| 总计 | **约 17.56 GMAC** |

若乘、加分别记一个 FLOP，则约 35.1 GFLOPs。$224^2$下 MLP 与通道投影占主导；升到$384^2$后序列从 197 增到 577，attention 二次项与激活显存显著上升。

### 11.8 最小复现路径

最现实的官方路径是加载 ImageNet-21k 的 ViT-B/16 checkpoint，在 CIFAR-10 等下游数据微调：

```bash
python -m vit_jax.main \
  --workdir=/tmp/vit-run \
  --config=vit_jax/configs/vit.py:b16,cifar10 \
  --config.pretrained_dir=gs://vit_models/imagenet21k
```

复现时要注意：

1. `b16,cifar10` preset 通常使用 384 crop，因此 token 长度是 577，而不是本文结构讲解的 197；
2. 若只验证张量形状，应显式固定 224；
3. 若复现 README 微调准确率，应保留官方 384 preset；
4. JFT 旗舰结果无法仅靠此仓库与公开数据严格复现；
5. 当前 commit 的代码依赖较旧 JAX/Flax 生态，直接安装时可能需要按环境文件固定版本。

### 11.9 代码层最值得记住的差异

| 论文叙述 | 官方代码事实 |
|---|---|
| 展平 patch 后线性投影 | 用 kernel=stride=patch 的 Conv 实现，数学等价 |
| 预训练使用 MLP head | 当前 B/16 默认`representation_size=None`，微调直接线性 head |
| 预训练使用 Adam | 当前`vit_jax/train.py`主路径是加载 checkpoint 后用 SGD 微调 |
| 预训练分辨率 224 | 下游 preset 常用 384，需要二维位置编码插值 |
| ViT 可处理更长序列 | checkpoint 插值代码偏向方形 patch 网格 |
| 论文包含 JFT 旗舰训练 | 此仓库不是完整 JFT 多主机预训练复现；完整训练基础设施主要在`big_vision` |

---

## 12. 材料来源

- Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
- 官方论文：https://arxiv.org/abs/2010.11929
- OpenReview：https://openreview.net/forum?id=YicbFdNTTy
- 官方代码：https://github.com/google-research/vision_transformer
- 固定代码提交：https://github.com/google-research/vision_transformer/commit/64801f1b3b367b3611cc27a3d45cc22870a36fb3
- 本地逐图、代码文件和 SHA-256 记录：[[assets/ViT - 图像分块与纯Transformer视觉建模/sources.md]]
