---
type: paper-note
status: complete
domain: Foundations
paper: "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"
year: 2024
arxiv: "2404.02905"
doi: "10.48550/arXiv.2404.02905"
source: "https://arxiv.org/abs/2404.02905"
project: "https://var.vision/"
code: "https://github.com/FoundationVision/VAR"
tags:
  - Image-Generation
  - Autoregressive-Modeling
  - VQVAE
  - Scaling-Law
created: 2026-10-13
updated: 2026-10-13
---

# VAR：下一尺度自回归图像生成

## 基本信息

- **论文**：Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction
- **作者**：Keyu Tian、Yi Jiang、Zehuan Yuan、Bingyue Peng、Liwei Wang
- **机构**：北京大学、字节跳动
- **版本**：arXiv v2，2024-06-10
- **代码**：https://github.com/FoundationVision/VAR
- **项目页**：https://var.vision/

## 一句话结论

VAR 的关键不是换一个更大的 Transformer，而是把图像自回归的基本单位从“按光栅顺序预测下一个 token”改成“由粗到细预测下一张尺度图”：跨尺度保持自回归，尺度内 token 并行生成。这个重新定义显著缩短了推理链，并让 GPT 式视觉自回归模型在 ImageNet 256×256 上以常规采样达到 FID 1.92，加入 rejection sampling 后达到 FID 1.73、IS 350.2，首次在论文设定下超过强 diffusion Transformer。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

语言模型天然是一维序列，next-token prediction 同时符合文本顺序和因果分解。图像却是二维信号。此前的 VQGAN、iGPT 等方法先用 VQ tokenizer 把图像变成二维离散 token map，再强行按从左到右、从上到下的 raster scan 展平成一维序列。这使图像可以套用 GPT，但引入了并不自然的生成顺序。

VAR 追问的是：**图像自回归是否必须等价于逐像素或逐 token 扫描？** 作者的回答是否定的。自回归只要求联合分布可以按某组变量进行条件分解，并不要求每一步必须是一个标量 token。因此，可以把“一张尺度上的 token map”作为一个自回归单元。

### 1.2 传统 next-token 图像 AR 的瓶颈

论文把问题概括为四点。第一，VQ 编码器产生的图像特征存在双向空间相关，而 raster scan 却假设当前 token 只依赖一维前缀，表示方式和因果假设并不一致。第二，固定的左上到右下顺序妨碍补全、外扩等任务的零样本泛化。第三，二维相邻位置在线性序列里可能相隔很远，破坏空间局部性。第四，也是工程上最直接的问题：最终 latent 为 $n\times n$ 时，逐 token AR 要执行 $O(n^2)$ 次模型前向，标准自注意力下总计算复杂度达到 $O(n^6)$。

![](assets/VAR%20-%20下一尺度自回归图像生成/figures/01-ar-vs-var.png)

> 图 2：语言 AR、图像 raster-scan AR 与 VAR 的对比。VAR 不再逐 token 扫描，而是依次生成 $r_1,r_2,\ldots,r_K$ 多尺度 token map；尺度内的 token 同时预测。来源：论文 Figure 2，第 2 页，https://arxiv.org/abs/2404.02905

### 1.3 本文核心贡献

1. 提出 next-scale prediction，把图像自回归因子分解从单 token 改为整张尺度图。
2. 设计多尺度残差 VQ tokenizer，把图像编码成由粗到细的一组离散 token map。
3. 在 ImageNet 类别条件生成上展示出比传统 AR 更好的质量、速度和扩展性，并在论文对比口径下超过 DiT。
4. 对参数量和最优训练计算量拟合出稳定幂律关系，同时展示无需微调的补全、外扩和类别条件编辑。

---

## 2. 任务定义与概率分解

### 2.1 输入、输出与基本表示

输入图像 $im$ 先通过编码器得到连续特征：

$$f=\mathcal{E}(im),\quad f\in\mathbb{R}^{h\times w\times C}$$

传统 VQVAE 将每个位置量化为 codebook $Z\in\mathbb{R}^{V\times C}$ 中最近的向量，得到单张离散 token map $q\in[V]^{h\times w}$。VAR 则把一张图编码为多尺度离散表示：

$$R=(r_1,r_2,\ldots,r_K),\quad r_k\in[V]^{h_k\times w_k}$$

其中尺度从 $1\times1$ 开始逐级增大，最终 $h_K\times w_K=h\times w$。各尺度共用同一个大小为 $V=4096$ 的 codebook，因此它们使用同一套离散视觉词表。

### 2.2 Next-scale prediction

传统视觉 AR 使用：

$$p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t})$$

VAR 改为：

$$p(r_1,r_2,\ldots,r_K)=\prod_{k=1}^{K}p(r_k\mid r_1,r_2,\ldots,r_{k-1})$$

这条公式是整篇论文最核心的变化。自回归顺序现在只发生在尺度之间；给定所有更粗尺度后，第 $k$ 个尺度内的 $h_k w_k$ 个 token 可以并行预测。也就是说，它并没有取消自回归，而是把 AR 粒度从“位置”提升为“分辨率层级”。

---

## 3. 核心方法

### 3.1 总体框架

VAR 分成两个相互独立的训练阶段。第一阶段训练多尺度 VQ autoencoder，负责把每张图像转成 $K$ 张 token map；第二阶段固定 tokenizer，用 decoder-only Transformer 学习 next-scale prediction。

![](assets/VAR%20-%20下一尺度自回归图像生成/figures/02-method-pipeline.png)

> 图 4：VAR 的两阶段训练流程。左侧 tokenizer 用残差式多尺度量化生成监督 token map；右侧 Transformer 输入 $([s],r_1,\ldots,r_{K-1})$ 并预测 $(r_1,\ldots,r_K)$，训练时用 block-wise causal mask 限制跨尺度依赖。来源：论文 Figure 4，第 5 页，https://arxiv.org/abs/2404.02905

### 3.2 多尺度残差量化

多尺度 token 不是对原图独立缩放后分别量化，而是逐级量化残差。令当前残差特征初始为 $f=\mathcal{E}(im)$。第 $k$ 个尺度执行：

1. 把当前残差插值到 $h_k\times w_k$；
2. 用共享 codebook 量化得到 $r_k$；
3. 查表得到连续 code 特征 $z_k$；
4. 将 $z_k$ 上采样回最终 latent 尺寸；
5. 经尺度专属卷积 $\phi_k$ 后，从残差中减去。

可写成：

$$r_k=\mathcal{Q}(\operatorname{interpolate}(f,h_k,w_k))$$

$$f\leftarrow f-\phi_k(\operatorname{interpolate}(\operatorname{lookup}(Z,r_k),h_K,w_K))$$

因此，$r_1$ 主要承担全局轮廓与类别，后续尺度编码此前尚未解释的高频残差。解码时反过来把各尺度贡献逐级相加：

$$\hat f\leftarrow\hat f+\phi_k(\operatorname{interpolate}(\operatorname{lookup}(Z,r_k),h_K,w_K))$$

最终通过解码器得到 $\hat{im}=\mathcal{D}(\hat f)$。论文沿用 VQGAN 主体，只新增 $K$ 个轻量尺度卷积，额外参数约 0.03M；tokenizer 在 OpenImages 上训练，空间下采样率为 16×。

### 3.3 Tokenizer 的训练目标

多尺度 VQ autoencoder 使用重建、特征量化、感知和对抗损失的组合：

$$\mathcal L=\lVert im-\hat{im}\rVert_2+\lVert f-\hat f\rVert_2+\lambda_P\mathcal L_P(\hat{im})+\lambda_G\mathcal L_G(\hat{im})$$

这里 tokenizer 决定了离散表示的上限。如果 tokenizer 无法保留纹理和细节，后面的 Transformer 再强也不能恢复。因此，论文虽然把创新重点放在 AR 范式，但最终质量仍依赖 VQ 重建质量。

### 3.4 VAR Transformer 与注意力掩码

Transformer 输入为 $([s],r_1,r_2,\ldots,r_{K-1})$，输出目标为 $(r_1,r_2,\ldots,r_K)$。类别条件 embedding 既充当起始 token $[s]$，也进入 AdaLN。训练损失是标准 token-level cross entropy。

关键是 block-wise causal mask：属于尺度 $r_k$ 的 token 可以看到 $r_{\le k}$，但看不到未来更细尺度。尺度内不是传统左到右因果关系，因此同尺度 token 可建立双向联系并并行输出。推理时按尺度逐级生成，并使用 KV cache。

模型采用较朴素的 GPT-2 风格 decoder-only Transformer，使用 AdaLN 和 query/key normalization，但未采用 RoPE、SwiGLU、RMSNorm。作者有意保持骨干简单，以说明收益主要来自 next-scale 范式而不是堆叠架构技巧。

### 3.5 计算复杂度与推理步数

对最终 $n\times n$ token map，传统逐 token AR 需要 $O(n^2)$ 次自回归迭代，总计算量为 $O(n^6)$。若尺度按几何级数增长，VAR 只需要 $O(\log n)$ 个尺度步骤，总计算量降为 $O(n^4)$。在 ImageNet 256×256 实验中，VAR 使用 10 个尺度，因此表格中的 #Step 为 10，而 VQGAN 为 256、DiT 为 250。

需要注意，这里的 $O(n^4)$ 仍不是“免费”的：较细尺度上一次前向要同时处理很多 token，自注意力的内存和计算仍随 token 数快速增长。VAR 主要消除了长串行链，显著改善 latency，却没有从根本上解决高分辨率全局注意力的二次复杂度。

### 3.6 训练与采样配置

论文使用 AdamW，$\beta_1=0.9$、$\beta_2=0.95$、weight decay 0.05；batch size 为 768–1024，模型训练 200–350 epochs。基础学习率按 batch size 256 设置为 $10^{-4}$。采样侧使用 top-$k$（$k=600$）和 classifier-free guidance（CFG=2.0）；最大 2B 模型还报告 rejection sampling 版本。

---

## 4. 实验设置

### 4.1 数据集与模型规模

主要评测是 ImageNet 256×256 类别条件生成。作者训练了深度 $d=6$ 到 $30$ 的 12 个 VAR，参数量从 18.5M 扩展到 2.0B。模型宽度与深度按 $w=64d$ 缩放，注意力头数设为 $h=d$。最大模型训练 token 总量约 305B。

Tokenizer 在 OpenImages 上训练，而 Transformer 在 ImageNet 上训练。这个设定与 VQGAN baseline 尽量保持 tokenizer 架构和数据来源一致，使“AR→VAR”的消融更可信。

### 4.2 指标与基线

主要指标包括 FID、Inception Score、precision、recall、推理步数和相对 wall-clock time。对比对象覆盖传统 AR（VQGAN、ViT-VQGAN、RQ-Transformer）、masked generation、GAN，以及 ADM、DiT、L-DiT 等 diffusion 模型。

### 4.3 结果口径

论文同时报告常规采样和 rejection sampling。引用结果时必须区分：常规 VAR-d30 的 FID 是 1.92；VAR-d30-re 的最佳 FID 是 1.73。后者不能直接当成普通采样结果。

---

## 5. 实验结果

### 5.1 主要定量结果

ImageNet 256×256 上，模型规模增加时表现稳定改善：

| 模型 | 参数量 | FID↓ | IS↑ | Precision↑ | Recall↑ | 步数 |
|---|---:|---:|---:|---:|---:|---:|
| VAR-d16 | 310M | 3.30 | 274.4 | 0.84 | 0.51 | 10 |
| VAR-d20 | 600M | 2.57 | 302.6 | 0.83 | 0.56 | 10 |
| VAR-d24 | 1.0B | 2.09 | 312.9 | 0.82 | 0.59 | 10 |
| VAR-d30 | 2.0B | 1.92 | 323.1 | 0.82 | 0.59 | 10 |
| VAR-d30-re | 2.0B | **1.73** | **350.2** | 0.82 | 0.60 | 10 |

与传统 VQGAN baseline 相比，摘要强调 FID 从 18.65 降到 1.73，IS 从 80.4 升到 350.2，推理约快 20×。与 diffusion 对比时，DiT-XL/2（675M）为 FID 2.27、250 步；L-DiT-3B 为 2.10；L-DiT-7B 为 2.28。常规 VAR-d30 的 1.92 已优于这些结果，rejection sampling 后进一步到 1.73。

![](assets/VAR%20-%20下一尺度自回归图像生成/figures/03-scaling-comparison.png)

> 图 3：不同生成范式的 FID、推理时间与参数规模。VAR 随规模扩大持续向左下角移动，即质量提高同时仍维持低延迟；论文以 ImageNet 验证集 FID 1.78 作为参考线。来源：论文 Figure 3，第 3 页，https://arxiv.org/abs/2404.02905

### 5.2 消融实验：收益主要来自范式改变

表 3 的消融链很有说服力：

| 设置 | 参数量 | 相对推理成本 | FID↓ |
|---|---:|---:|---:|
| AR baseline | 227M | 1.000 | 18.65 |
| AR → VAR | 207M | 0.013 | 5.22 |
| + AdaLN | 310M | 0.016 | 4.95 |
| + top-$k=600$ | 310M | 0.016 | 4.64 |
| + CFG=2.0 | 310M | 0.022 | 3.60 |
| + q/k normalization | 310M | 0.022 | 3.30 |
| + 扩展至 2.0B | 2.0B | 0.052 | 1.73 |

只把 raster AR 换成 VAR，在参数量略低的情况下 FID 就从 18.65 降到 5.22，同时推理成本降到 0.013×。这说明主增益并不是后续 AdaLN、CFG 或扩大模型，而是自回归顺序本身。后续技巧继续优化，但属于第二层贡献。

值得提醒的是，表 3 最后一行写 1.73，却没有像表 1 那样显式标注 `-re`；官方 HTML 没有解释这一口径差异。严谨引用时应把表 1 的常规 1.92 和 rejection 1.73 分开。

### 5.3 Scaling law

论文对参数量 $N$ 与测试 loss/token error 做幂律拟合。例如最后尺度和全尺度平均 loss 分别为：

$$L_{\mathrm{last}}=(2.0N)^{-0.23}$$

$$L_{\mathrm{avg}}=(2.5N)^{-0.20}$$

在对数坐标中，Pearson 相关系数接近 -0.998。对达到某个 loss 所需的最优训练计算量 $C_{\min}$，论文同样得到近似幂律，相关系数接近 -0.99，跨度约 6 个数量级。

这里需要准确理解：作者证明的是 **cross-entropy 与 token error** 相对参数量/计算量的 scaling law，而不是直接证明 FID 严格遵循幂律。FID 随规模改善是实验趋势，不能与拟合对象混为一谈。

### 5.4 零样本补全、外扩与编辑

VAR-d30 在没有下游微调的情况下可以做三类任务：mask 内 in-painting、画布外 out-painting，以及限定 bounding box 的类别条件编辑。做法是固定 mask 外真实 token，只让模型生成指定区域；编辑时再加入目标类别条件。

![](assets/VAR%20-%20下一尺度自回归图像生成/figures/04-zero-shot-tasks.png)

> 图 8：VAR 的零样本补全、外扩与类别条件编辑。结果说明多尺度条件生成比固定 raster 顺序更容易接受任意空间约束，但论文只给出定性样例，没有系统的下游定量评测。来源：论文 Figure 8，第 9 页，https://arxiv.org/abs/2404.02905

这些结果说明 next-scale 表示具有一定任务泛化能力，但证据强度有限：没有专门 benchmark，没有与编辑模型比较，也没有成功率、保真度或感知指标。因此，“像 LLM 一样的零样本泛化”更适合作为早期信号，而不是已经充分验证的结论。

---

## 6. 与相关工作的关系

VAR 与 VQ-VAE/VQGAN 路线一脉相承：都先把图像压缩成离散 token，再让 Transformer 建模。区别不在离散化本身，而在离散 token 的组织和概率分解。VQVAE-2、RQ-Transformer 也包含层级或残差结构，但仍主要按 token 顺序展开；VAR 将“尺度”正式提升为自回归轴。

它与 diffusion 的关系则更像路线竞争。Diffusion 通过多步去噪逐渐形成图像，VAR 通过多尺度离散预测逐渐细化图像。二者都是 coarse-to-fine，但随机过程和训练目标不同：前者通常回归噪声/速度场，后者做离散交叉熵。VAR 的突出优势是串行步数少、训练目标简单；代价是受 tokenizer 上限约束，并且每个尺度的离散分布建模仍较重。

从后续视觉生成发展看，VAR 的价值不只是某个 ImageNet FID，而是重新打开了“视觉 AR 应该沿什么轴自回归”的设计空间。后来的 next-scale、next-set、next-frame-group、混合 AR-diffusion 路线，都可从这个视角理解。

---

## 7. 局限与批判性评价

### 7.1 Tokenizer 是隐藏瓶颈

VAR 的生成上限受 VQ tokenizer 的重建质量、codebook 利用率和量化误差约束。论文刻意使用接近 VQGAN 的 tokenizer 以公平验证范式，但这也意味着“AR 模型性能”与“离散表示性能”难以完全解耦。若未来 tokenizer 改变，scaling 曲线是否保持仍需重新验证。

### 7.2 任务范围窄于现代文生图

主实验是 ImageNet 类别条件生成，不是开放词表文本到图像。论文没有验证长文本对齐、组合概念、文字渲染、高分辨率写实生成等现代 T2I 关键能力。因此，“超过 diffusion Transformer”只在论文的 ImageNet 256×256 设定下成立，不能外推为全面取代 diffusion。

### 7.3 零样本泛化主要是定性证据

补全、外扩、编辑展示有启发性，但样例数量有限，没有定量 benchmark，也没有失败案例统计。它证明“机制上可做”，尚未证明“稳定、可控、优于专用方法”。

### 7.4 高分辨率仍有算力压力

VAR 把迭代次数从 $O(n^2)$ 降到 $O(\log n)$，显著降低推理延迟；但最高尺度仍包含大量 token，同尺度全局注意力的成本不会消失。论文只重点验证到 256/512 分辨率，对 1024 乃至更高分辨率的显存、吞吐与局部一致性没有系统展开。

### 7.5 评测口径要谨慎

最佳 FID 1.73 使用 rejection sampling，而常规 VAR-d30 是 1.92；不同模型的 tokenizer、训练 epoch、数据和采样设置并非完全统一。VAR 的趋势和效率优势很强，但单一最佳数字不宜脱离这些条件传播。

---

## 8. 复现与实践建议

官方代码和模型已公开。复现可以拆成三个层级：

1. **推理验证**：直接使用预训练 tokenizer 与 VAR 模型，检查 10 个尺度的生成过程和采样速度。
2. **小模型训练**：先训练 d16 或更小模型，重点核对多尺度 token 序列、block-wise mask、位置/尺度 embedding 和 teacher forcing 是否正确。
3. **完整复现**：单独训练 OpenImages tokenizer，再进行 ImageNet 200–350 epoch 的 Transformer 训练。最大 2B 版本成本较高，不适合作为首次复现目标。

最容易出错的地方不是普通 Transformer 模块，而是：残差多尺度量化与反量化必须严格对称；不同尺度展开后的索引和位置 embedding 必须对应；mask 只能阻止未来尺度，不能错误地在同尺度内施加逐 token 因果约束；采样评测必须明确是否使用 CFG、top-$k$ 和 rejection sampling。

---

## 9. 个人启发与后续问题

VAR 最重要的启发是：**自回归模型的核心不是固定的 token 顺序，而是选择一组有意义的条件变量，并把联合分布分解成可学习、可并行、可扩展的阶段。** 对图像来说尺度是自然轴；对视频可以是时空尺度或帧组；对 3D 可以是占据层级、八叉树层级、几何到纹理层级；对机器人世界模型可以是意图、粗轨迹和细动作层级。

后续值得关注的问题包括：多尺度离散表示是否能扩展到开放文本条件；高分辨率下是否需要局部/稀疏注意力；连续 token 或 flow matching 能否保留 next-scale 的结构优势；scaling law 在更大数据和更多任务上是否仍成立；以及零样本编辑是否能从定性展示发展成可靠接口。

**个人判断：★★★★★。** 这篇论文的长期价值高于它的单项 FID：它改变了视觉 AR 的建模坐标系，使“GPT 式图像生成”从低效 raster scan 变成可扩展的 coarse-to-fine 路线。局限是验证仍集中在 ImageNet 类别条件生成，文本条件和高分辨率通用生成尚未在本文中完成。

---

## 10. 材料来源

- [arXiv 论文页](https://arxiv.org/abs/2404.02905)
- [官方 PDF](https://arxiv.org/pdf/2404.02905)
- [项目页](https://var.vision/)
- [官方代码](https://github.com/FoundationVision/VAR)
- [本地来源清单](assets/VAR%20-%20下一尺度自回归图像生成/sources.md)
