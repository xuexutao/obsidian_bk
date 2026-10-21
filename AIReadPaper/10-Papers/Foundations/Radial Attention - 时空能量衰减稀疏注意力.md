---
type: paper-note
status: complete
domain: Foundations
paper: "Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation"
year: 2025
venue: NeurIPS 2025
arxiv: "2506.19852"
doi: "10.48550/arXiv.2506.19852"
source: https://arxiv.org/abs/2506.19852
project: null
code: https://github.com/mit-han-lab/radial-attention
license: Apache-2.0
tags:
  - Foundations
  - Sparse-Attention
  - Video-Diffusion
  - Long-Video
  - LoRA
  - Systems
created: 2026-10-20
updated: 2026-10-20
---

# Radial Attention：时空能量衰减稀疏注意力

## 基本信息

- **正式题名**：Radial Attention: $O(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation
- **作者**：Xingyang Li、Muyang Li、Tianle Cai、Haocheng Xi、Shuo Yang、Yujun Lin、Lvmin Zhang、Songlin Yang、Jinbo Hu、Kelly Peng、Maneesh Agrawala、Ion Stoica、Kurt Keutzer、Song Han
- **机构**：MIT、NVIDIA、Princeton、UC Berkeley、Stanford、First Intelligence
- **会议**：NeurIPS 2025
- **版本**：arXiv v2，2025-12-06；NeurIPS 2025 camera-ready
- **任务**：在尽量保持预训练视频扩散模型质量的前提下，降低长视频 3D self-attention 的训练与推理成本，并支持 2×–4× 长度扩展
- **验证模型**：HunyuanVideo 13B、Wan2.1-14B、Mochi 1 10B
- **核心关键词**：Spatiotemporal Energy Decay、static block sparsity、radial mask、attention sink、length extrapolation、LoRA
- **资源**：[[assets/Radial Attention - 时空能量衰减稀疏注意力/sources|来源记录]]；[官方 PDF](assets/Radial%20Attention%20-%20时空能量衰减稀疏注意力/sources/paper.pdf)

## 一句话结论

**Radial Attention 观察到视频 DiT 的 softmax 后注意力会随时间与空间距离近似指数衰减，于是把时间距离划为对数尺度的环带：距离每翻倍，空间注意力窗口或远程对角线采样密度减半，使每个尺度付出近似恒定计算量，在固定分辨率、增长帧数时把注意力复杂度从 $O(n^2)$ 降为 $O(n\log n)$；其静态 128×128 block mask 无需逐 head 动态 profiling，默认长度可免训练加速约 1.8–1.9×，HunyuanVideo 509 帧时则以 88.3% 稀疏率将训练从 93.6 小时降到 21.4 小时、推理从 2895 秒降到 781 秒，同时保持与 dense full fine-tuning 相当的质量。**

---

## 1. 研究背景与问题定义

### 1.1 长视频真正的计算瓶颈

现代视频扩散模型通常把时空 latent patch 展平为长度为 $n$ 的 token 序列，并在 DiT 中执行 3D dense self-attention：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V$$

其中 $Q,K,V\in\mathbb R^{n\times d}$。时间和显存复杂度随 token 数平方增长：

$$T_{\mathrm{dense}},M_{\mathrm{dense}}=O(n^2)$$

视频多出一个时间维度。论文以 HunyuanVideo 为例，5 秒、720p 视频约需要 110K latent tokens；当长度扩展到 4× 时，注意力矩阵面积理论上增长约 16×。FlashAttention 能减少中间显存读写，却不会消除 $QK^\top$ 的二次计算量，因此“内核优化”不足以解决分钟级或超长视频。

### 1.2 既有稀疏路线的矛盾

| 路线 | 代表方法 | 优点 | 论文指出的问题 |
|---|---|---|---|
| 动态稀疏 | SVG | 按 head 识别 spatial/temporal pattern，质量较好 | 在线 profiling；遇到未见过的长序列可能误分类；不利于训练 |
| 固定局部窗口 | STA | 规则、硬件友好 | 感受野有限，丢失远程时序依赖 |
| 通用 $O(n\log n)$ | PowerAttention、LogSparse 等 | 次二次复杂度 | 未利用视频特有的时空结构，硬件效率或生成质量有限 |
| 线性注意力 | SANA 等 | $O(n)$ | 替换 softmax 架构，局部细节和预训练兼容性变差，少量微调难恢复 |
| 训练免费长度外推 | RIFLEx | 不需要微调 | 约 2× 尚可，4× 时模糊、慢动作和质量退化明显 |

Radial Attention 选择中间路线：**保留 softmax、保留全局感受野，但用与视频统计结构匹配的静态 block-sparse mask 删除低能量关系。**

### 1.3 核心贡献

1. **提出 Spatiotemporal Energy Decay**：实证发现 post-softmax attention score 随时空距离近似指数衰减。
2. **把能量衰减翻译为计算密度衰减**：时间距离每翻倍，允许的空间交互密度减半，形成 radial/logarithmic sparsity。
3. **给出复杂度和误差分析**：固定空间分辨率时复杂度为 $O(n\log n)$；在指数衰减假设下，注意力分布误差有指数上界。
4. **静态、块稀疏、训练推理统一**：无需逐 head 动态选择，使用 128×128 block，可直接接入稀疏 GPU kernel。
5. **结合 LoRA 扩展长度**：只调 Q/K/V/O 投影即可支持 2×–4× 长视频，降低训练和推理成本。

![](assets/Radial%20Attention%20-%20时空能量衰减稀疏注意力/figures/01-teaser.png)

> **图 1（Figure 1，PDF p.1）**：上半部分显示默认 117 帧 HunyuanVideo 在相近质量下从 1649 秒降到 876 秒；下半部分是 509 帧、4× 长度，Radial+LoRA 的 Vision Reward 为 0.134，训练成本 171 GPU-hours、推理 781 秒，分别约为 dense full fine-tuning 的 1/4.4 和 1/3.7。来源：<https://arxiv.org/abs/2506.19852>。

---

## 2. 任务与符号

### 2.1 视频 token 组织

设视频 latent 有：

- $f$ 个时间帧；
- 每帧 $s$ 个空间 tokens；
- 总 token 数 $n=fs$；
- query token 位于帧 $i_0$、帧内展平空间位置 $k_0$。

其对全部 keys 的 post-softmax attention 分布记为：

$$\boldsymbol p=\operatorname{softmax}(Q_{i_0s+k_0}K^\top),\qquad\boldsymbol p\in[0,1]^n$$

论文设计一个 mask $M\in\{-\infty,0\}^{n\times n}$：

$$\operatorname{SparseAttention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top+M}{\sqrt d}\right)V$$

$M_{a,b}=0$ 表示保留 token pair，$M_{a,b}=-\infty$ 表示跳过对应计算。

### 2.2 目标

设计一个静态 mask，使它同时满足：

1. 保留视频注意力中的高能量时空关系；
2. 对所有 heads 使用统一规则，不依赖输入内容 profiling；
3. 允许训练和推理真正跳过 block；
4. 仍保留跨整段视频的稀疏远程连接；
5. 固定 $s$、增大 $f$ 时，关系数量只按 $O(n\log n)$ 增长。

---

## 3. 关键观察：Spatiotemporal Energy Decay

### 3.1 两类典型注意力头

作者在 HunyuanVideo 的 post-softmax attention map 中观察到两种模式：

- **Spatial Attention**：主要看当前/邻近帧的较宽空间区域。其时间衰减快、空间衰减慢，即 $\alpha$ 大、$\beta$ 小。
- **Temporal Attention**：跨较长时间关注相同或相近空间位置。其时间衰减慢、空间衰减快，即 $\alpha$ 小、$\beta$ 大。

这里的“spatial/temporal”来自 SVG 的命名。Radial Attention 不再对 head 做二分类，而是用一种统一 mask 同时覆盖中央局部区域和远距离同位置关联。

![](assets/Radial%20Attention%20-%20时空能量衰减稀疏注意力/figures/02-energy-decay.png)

> **图 2（Figure 4，PDF p.5）**：左侧两个注意力图分别呈现 spatial 与 temporal 模式；右侧统计表明，固定空间位置时 attention score 随时间距离下降，固定帧时则随空间距离下降。曲线是多个随机 attention maps、heads 和 diffusion steps 的平均值。来源：<https://arxiv.org/abs/2506.19852>。

### 3.2 指数衰减假设

论文假设存在 $\alpha,\beta>0$ 与 $C_{\mathrm{rel}}>0$，使第 $j$ 帧、空间位置 $l$ 的 attention score 满足：

$$p_{js+l}\le C_{\mathrm{rel}}e^{-\alpha|j-i_0|-\beta|l-k_0|}p_{i_0s+k_0}$$

- $\alpha$ 控制时间距离的衰减速度；
- $\beta$ 控制空间距离的衰减速度；
- $C_{\mathrm{rel}}$ 允许不同 query/head 的相对幅值变化。

作者用：

$$y=e^{-ax+b}$$

拟合平均时间/空间衰减曲线，得到 $R^2>0.985$；Figure 8 中两个实例分别达到约 0.999 和 0.985。

### 3.3 这个观察应如何理解

这不是说每个 attention head、每个 diffusion step、每个视频都严格服从同一指数律。论文测量的是**平均统计趋势**。少数强远程依赖可能被平均掩盖，例如：

- 主体离开很久后重新出现；
- 周期运动；
- 两个远隔区域发生语义互动；
- 镜头切换后重新回到旧场景。

因此指数衰减更像稀疏预算的工程先验，而不是视频注意力的普适物理定律。

---

## 4. 核心方法：从能量衰减到计算密度衰减

### 4.1 时间距离分级

对帧对 $(i,j)$，时间距离为：

$$\Delta_{ij}=|i-j|$$

定义对数 level：

$$r(i,j)=\left\lfloor\log_2\max(|i-j|,1)\right\rfloor$$

于是：

| 时间距离 $\Delta$ | Level $r$ | 目标计算密度 |
|---:|---:|---:|
| 0–1 | 0 | 1 |
| 2–3 | 1 | $1/2$ |
| 4–7 | 2 | $1/4$ |
| 8–15 | 3 | $1/8$ |
| $2^r$–$2^{r+1}-1$ | $r$ | $2^{-r}$ |

时间范围每翻倍，计算密度减半。整个帧级 attention map 因而被划为：

$$2\left\lceil\log_2\max(f,2)\right\rceil-1$$

个围绕主对角线的 bands。

### 4.2 近距离：逐级缩窄空间对角带

在 level $r$，frame-to-frame block 中保留的空间窗口宽度为：

$$W_r=\left\lfloor\frac{s}{2^r}\right\rfloor$$

对 query 位置 $k$ 和 key 位置 $l$，若：

$$|k-l|+1\le\frac{s}{2^r}$$

则保留连接。随着时间距离增大，空间带宽逐级减半。

这里必须注意：论文形式化中的 $k,l$ 是**帧内展平 token index**，$|k-l|$ 不是严格的二维欧氏距离。实际视觉邻域是否连续，取决于空间 token 的展平顺序和 block layout；这比“真正的 2D radial radius”更容易做规则块计算，但也带有栅格序列化偏置。

### 4.3 远距离：降低对角线出现频率

当：

$$\frac{s}{2^r}<1$$

空间窗口无法继续缩小。方法改为只保留同位置连接 $k=l$，并以：

$$q_r=\left\lceil\frac{2^r}{s}\right\rceil$$

为帧间采样步长，仅在：

$$|i-j|\bmod q_r=0$$

时保留跨帧对角线。这样既保留稀疏全局感受野，又继续维持指数下降的计算密度。

### 4.4 完整 mask

论文先定义四维 mask：

$$\tilde M\in\{-\infty,0\}^{f\times f\times s\times s}$$

其形式为：

$$\tilde M_{i,j,k,l}=\begin{cases}0,&2^{r(i,j)}\le s\ \text{且}\ |k-l|+1\le\dfrac{s}{2^{r(i,j)}}\\0,&|i-j|\bmod\left\lceil\dfrac{2^{r(i,j)}}{s}\right\rceil=0\ \text{且}\ k=l\\-\infty,&\text{其他情况}\end{cases}$$

再按：

$$M_{is+k,js+l}=\tilde M_{i,j,k,l}$$

展平为标准 $n\times n$ attention mask。

![](assets/Radial%20Attention%20-%20时空能量衰减稀疏注意力/figures/03-radial-mask.png)

> **图 3（Figure 5，PDF p.6）**：左图是按时间 distance level 递减的计算密度；中图是理论 attention mask，中央带最密，外围逐渐缩窄并稀疏采样；右图是 HunyuanVideo 的实际大规模 mask。图中还额外加入了指向第一帧的 attention sink。来源：<https://arxiv.org/abs/2506.19852>。

### 4.5 Attention sink

纯 radial pattern 之外，论文把**第一帧**作为 attention sink，保留与第一帧的重要全局连接。第一帧往往定义主体、场景和构图，是视频扩散中的稳定锚点。该设计借鉴 attention sink 思想，但正文没有给出独立公式，因此 Equation (4) 只描述基础 radial mask，实际实现还包含 sink。

---

## 5. 为什么是 $O(n\log n)$

### 5.1 每个 level 的预算近似恒定

对于空间窗口仍不小于 1 的 level $r$：

- 该 level 覆盖的时间范围约为 $2^r$；
- 每个 frame-pair block 的空间计算量约按 $2^{-r}$ 缩小；
- 两者相乘后，每个 level 的总成本近似为常数 $O(fs^2)$。

而 level 数只有 $O(\log f)$，因此总量约为：

$$O(fs^2\log f)$$

### 5.2 论文给出的零元素上界

有效连接数量满足：

$$\#\{\tilde M=0\}\le4s^2f\log_2f$$

由 $n=fs$：

$$4s^2f\log_2f=4sn(\log_2n-\log_2s)$$

当每帧空间 token 数 $s$ 固定、只增长帧数 $f$ 时：

$$\boxed{T_{\mathrm{radial}}=O(n\log n)}$$

### 5.3 不要误读这个复杂度

该结论并不是对分辨率和时长同时增长的统一 $O(n\log n)$。表达式仍含 $s^2$：

$$T=O(s^2f\log f)$$

因此：

- **沿时间扩长**：次二次；
- **沿空间提高分辨率**：仍近似二次。

Radial Attention 主要解决“更长”，没有根治“更高分辨率”。

### 5.4 理论 FLOPs 与端到端速度

HunyuanVideo 509 帧中，论文报告 attention computation 约减少 9×，但端到端推理只加速 3.71×。原因是 FFN、归一化、VAE、通信和其他模块没有同比减少。稀疏注意力的系统价值必须看 wall-clock，而不能只看 mask sparsity 或理论 FLOPs。

---

## 6. 近似误差分析

### 6.1 Masked distribution

应用 radial mask 后：

$$\tilde{\boldsymbol p}=\operatorname{softmax}(Q_{i_0s+k_0}K^\top+M_{i_0s+k_0})$$

在指数衰减假设下，论文给出 $L_1$ 误差上界：

$$\|\tilde{\boldsymbol p}-\boldsymbol p\|_1\le C_{\mathrm{rel}}\left[\frac{8e^{-\beta(\frac s2+1)}}{(1-e^{-\alpha})(1-e^{-\beta})}+4\frac{1+e^{-\beta}}{1-e^{-\beta}}\frac{e^{-\alpha(s+1)}}{1-e^{-\alpha}}\right]$$

渐近写为：

$$\|\tilde{\boldsymbol p}-\boldsymbol p\|_1=O\left(C_{\mathrm{rel}}e^{-\min(\beta/2,\alpha)s}\right)$$

### 6.2 两类被删关系

附录把误差分为：

1. **保留帧中的空间尾部**：窗口外的远空间 tokens；
2. **远距离被下采样跳过的帧**：空间窗口已经小于 1 后，通过 stride 丢弃的 frame pairs。

两部分都可由几何级数求和得到指数上界。

### 6.3 理论边界

这个保证依赖很强的前提：

- attention score 的确满足相对指数上界；
- 时间轴可扩到无限、空间索引为有限一维网格；
- 误差分析针对单个 softmax row，而不是多层 DiT 最终生成分布；
- 没有分析 block rounding、有限精度与稀疏 kernel 引入的额外误差。

因此它证明的是“若能量真的这样衰减，则 mask 删除的概率质量有限”，不是对最终视频质量的端到端理论保证。

---

## 7. 硬件实现

### 7.1 128×128 block sparsity

理论 mask 以 token pair 为单位，但实际执行使用 **128×128 attention blocks**：

- 选中的 block 内执行 dense attention；
- 未选 block 完全跳过；
- 静态规则可预先构造，不需要运行时 top-k 或 head profiling。

这牺牲了一部分细粒度稀疏率，却能让 GPU 真正获得加速。若硬件/后端不能高效执行 block-sparse kernel，理论 $O(n\log n)$ 不一定转化为实际收益。

### 7.2 Kernel

- 推理：FlashInfer；
- 训练：Block-Sparse-Attention + FlashAttention-2 backend；
- 当前 Radial 结果使用 FA2 级后端；STA 基线使用 FA3，因此速度比较并非完全等内核；
- 作者把升级 FA3 视为与算法正交的未来工作。

### 7.3 Dense warm-up 与 dense early layers

Radial Attention 不是从第一层、第一步开始全程稀疏：

- 默认长度推理：前 12 个 diffusion steps 使用 dense attention；
- 4× 长度推理：前 2 个 steps 使用 dense attention；
- 默认长度主实验：第一个 DiT block 保持 dense；
- 长视频训练：前两个 DiT blocks 保持 dense。

这表明早期去噪和浅层全局信息聚合对稀疏化更敏感。论文标题中的 $O(n\log n)$ 描述主要针对被替换的 attention 部分；混合 dense/radial 系统的实际成本还包含这些 dense warm-up。

---

## 8. 用 LoRA 扩展视频长度

### 8.1 为什么换 mask 还不够

默认长度下可以直接把 dense attention 换成 Radial Attention；但生成 2×–4× 长视频时，预训练模型仍面临长度分布偏移：运动减速、重复、模糊和主体/场景漂移。Radial mask 降低了计算，却不会自动让短视频模型学会长时间动力学。

因此论文在 attention 的 Q/K/V/O projections 上加入 LoRA：

$$W'=W+BA$$

其中 $A,B$ 为低秩矩阵。只调整 attention 投影，保持主体生成能力和大部分预训练权重不变。

### 8.2 训练设置

| 项目 | 配置 |
|---|---|
| 数据 | OpenVid-1M |
| 每种长度样本 | 按 aesthetic/motion score 选 2,000 个视频 |
| GPU | 8×H100 |
| LoRA rank | 128 |
| HunyuanVideo | global batch 1 + sequence parallelism；2× 2400 steps，4× 1200 steps |
| Mochi 1 | global batch 1 + sequence parallelism；2×/4× 均 1500 steps |
| Wan2.1-14B | global batch 8；2× 2500 steps |
| Dense early blocks | 训练时保留前两个 DiT blocks |

模型 block 数分别为：HunyuanVideo 60、Mochi 1 48、Wan2.1-14B 40。因此保留 2 个 dense blocks 的比例很小，但为全局信息提供了稳定入口。

### 8.3 为什么 Radial+LoRA 可能比 Dense+LoRA 更好

Figure 8(a) 显示，dense attention 下 LoRA 往往落后 full fine-tuning；Radial Attention 下，LoRA 却能匹配或超过 full fine-tuning。作者认为 radial mask 先把容量集中到最重要的时空关系，使低秩更新不需要同时修正大量弱相关连接。

更谨慎的解释是：**结构性稀疏本身也是正则化**。它既减少计算，也缩小适配空间，可能降低长序列微调中的过拟合或优化噪声。但论文没有通过固定参数量、固定 FLOPs、多随机种子实验单独验证这一机制。

![](assets/Radial%20Attention%20-%20时空能量衰减稀疏注意力/figures/05-lora-regression.png)

> **图 5（Figure 8，PDF p.10）**：左图比较 dense/radial 与 full FT/LoRA FT；Radial+LoRA 在较长视频上表现最好。右图用指数函数拟合时间和空间 attention decay，$R^2>0.985$。来源：<https://arxiv.org/abs/2506.19852>。

### 8.4 与风格 LoRA 合并

长度扩展 LoRA 可以与已有 style LoRA 直接 merge。附录显示其在默认长度和 4× 长度下基本保留风格，但作者也承认：长度 LoRA 只用较小数据集训练，可能引入轻微 style bias。更全面的数据适配可能缓解这一问题。

---

## 9. 实验设置

### 9.1 模型与长度

| 模型 | 参数量 | 默认能力 | 长度实验 |
|---|---:|---|---|
| HunyuanVideo | 13B | 约 5 秒、720p、125 帧 | 125 / 253 / 509 帧 |
| Mochi 1 | 10B | 约 5 秒、480p、约 162/163 帧 | 163 / 331 / 667 帧 |
| Wan2.1-14B | 14B | 约 5 秒、720p、81 帧 | 81 / 161 帧 |

默认长度免训练加速表使用略不同口径：HunyuanVideo 117 帧、Wan2.1 69 帧，分辨率均为 768×1280。

### 9.2 指标

- PSNR、SSIM、LPIPS：与原 dense 模型输出的相似度；
- Vision Reward：自动化视频偏好模型分数；
- VBench Subject Consistency、Aesthetic Quality、Imaging Quality；
- PFLOPs、wall-clock latency、training time、sparsity、speedup。

PSNR/SSIM/LPIPS 高只能说明稀疏版本接近原模型，不保证原模型本身生成正确。Vision Reward 也只是自动指标，论文没有正式人工用户研究。

### 9.3 Baselines

- SVG：动态 spatial/temporal head profiling；
- STA：滑动 3D tile；
- Spatial / Temporal：分别使用 SVG 的单一 mask；
- PowerAttention：按 2 的幂次距离连接的 $O(n\log n)$ 注意力；
- Harmonic Series：对角宽度与距主对角线距离成反比的另一种 $O(n\log n)$ mask；
- LongLoRA：shifted local attention；
- SANA：线性 attention 替换；
- RIFLEx：RoPE 频率外推；
- Full：目标长度上的 dense full fine-tuning；
- Original：预训练模型直接生成目标长度。

---

## 10. 默认长度：免训练加速

### 10.1 HunyuanVideo，117 帧，768×1280，单 H100

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Vision Reward ↑ | PFLOPs ↓ | 延迟 | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | — | — | — | **0.141** | 612 | 1649 s | 1.00× |
| STA (FA3) | 26.7 | 0.866 | 0.167 | 0.132 | 331 | **719 s** | **2.29×** |
| PowerAttention | 22.1 | 0.764 | 0.256 | 0.140 | 339 | 1002 s | 1.65× |
| SVG | 27.2 | **0.895** | **0.114** | **0.144** | 340 | 867 s | 1.90× |
| **Radial** | **27.3** | 0.886 | **0.114** | 0.139 | 339 | 876 s | **1.88×** |

Radial 的速度略低于 SVG，质量总体相当；相较 STA，Radial 稍慢但质量明显更好。由于 STA 使用 FA3、Radial 使用 FA2，算法和 kernel 版本混在一起，不能把全部速度差归因于 mask。

### 10.2 Wan2.1-14B，69 帧，768×1280，单 H100

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Vision Reward ↑ | PFLOPs ↓ | 延迟 | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | — | — | — | **0.136** | 560 | 1630 s | 1.00× |
| STA (FA3) | 22.9 | 0.830 | 0.171 | **0.132** | 322 | **812 s** | **2.01×** |
| PowerAttention | 22.4 | 0.790 | 0.176 | 0.126 | 324 | 978 s | 1.67× |
| SVG | 23.2 | 0.825 | 0.202 | 0.114 | 324 | 949 s | 1.71× |
| **Radial** | **23.9** | **0.842** | **0.163** | 0.128 | 323 | 917 s | **1.77×** |

Radial 在三项相似度指标上优于其他稀疏方法，但 Vision Reward 从原模型 0.136 降到 0.128。因此“质量保持”应理解为总体接近，而非完全无损。

---

## 11. 长视频结果

### 11.1 HunyuanVideo 4×：509 帧、720p

| 方法 | 稀疏率 | 训练时间 | 训练加速 | 推理时间 | 推理加速 | Vision Reward | S.C. | A.Q. | I.Q. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0% | — | — | 2895 s | 1.00× | 0.054 | 0.988 | 0.545 | 0.451 |
| RIFLEx | 0% | — | — | 2895 s | 1.00× | 0.037 | **0.989** | 0.539 | 0.456 |
| LongLoRA | 88.4% | 20.9 h | 4.48× | 803 s | 3.61× | 0.130 | 0.936 | 0.618 | **0.689** |
| PowerAttention | 88.2% | 21.8 h | 4.29× | 766 s | 3.78× | 0.128 | 0.950 | 0.590 | 0.648 |
| Full dense FT | 0% | 93.6 h | 1.00× | 2895 s | 1.00× | 0.133 | 0.977 | 0.590 | 0.635 |
| **Radial+LoRA** | **88.3%** | **21.4 h** | **4.37×** | **781 s** | **3.71×** | **0.134** | 0.973 | **0.623** | 0.672 |

这是论文最强的系统结果：Radial 以约四分之一训练时间和约四分之一推理时间，达到略高于 dense full FT 的 Vision Reward。注意力计算约减少 9×，端到端只加速 3.71×。

### 11.2 Mochi 1 4×：667 帧

| 方法 | 稀疏率 | 训练时间 | 训练加速 | 推理时间 | 推理加速 | Vision Reward | S.C. | A.Q. | I.Q. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0% | — | — | 992 s | 1.00× | -0.091 | 0.916 | 0.383 | 0.322 |
| Full dense FT | 0% | 49.2 h | 1.00× | 992 s | 1.00× | 0.099 | 0.934 | 0.613 | 0.613 |
| PowerAttention | 86.5% | 17.3 h | 2.84× | 381 s | 2.60× | 0.107 | 0.956 | **0.633** | **0.650** |
| **Radial+LoRA** | **85.5%** | **17.4 h** | **2.83×** | **386 s** | **2.57×** | **0.113** | **0.958** | 0.618 | 0.638 |

Radial 的 Vision Reward 和主体一致性最好，PA 的美学/成像质量更高。它不是所有维度都支配 baseline。

### 11.3 Wan2.1-14B 2×：161 帧

| 方法 | 稀疏率 | 训练时间 | 训练加速 | 推理时间 | 推理加速 | Vision Reward | S.C. | A.Q. | I.Q. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0% | — | — | 5735 s | 1.00× | 0.109 | 0.946 | 0.598 | 0.614 |
| Full dense FT | 0% | 28.0 h | 1.00× | 5735 s | 1.00× | **0.150** | 0.966 | 0.590 | **0.689** |
| **Radial+LoRA** | **73.6%** | **14.5 h** | **1.93×** | **2847 s** | **2.01×** | 0.145 | **0.981** | **0.607** | 0.677 |

Wan 是一个重要反例：Radial 明显更省，但 Vision Reward 与 I.Q. 仍略低于 dense full FT。作者没有报告 4× Wan，理由是计算与显存成本过高。

![](assets/Radial%20Attention%20-%20时空能量衰减稀疏注意力/figures/04-long-video-results.png)

> **图 4（Figures 6–7，PDF p.9）**：上半部分是 Wan 默认长度，Radial 在 917 秒内保持与 1630 秒 dense 相近的葡萄和人物视频；下半部分是 HunyuanVideo 509 帧，未经适配的 Original 已严重退化，Radial+LoRA 达到 0.134 Vision Reward。来源：<https://arxiv.org/abs/2506.19852>。

---

## 12. 消融与分析

### 12.1 Attention output error

在 Wan2.1-14B 上，相对 dense attention output 的平均 MSE：

| 方法 | MSE ↓ |
|---|---:|
| **Radial** | **$3.9\times10^{-3}$** |
| SVG | $4.4\times10^{-3}$ |
| STA | $1.5\times10^{-2}$ |

Radial 比 SVG 低约 11%，比 STA 低约 74%，说明统一静态 mask 虽不按 head 自适应，却更好保留了 dense output。

### 12.2 默认长度 dense warm-up steps

Wan2.1-14B 69 帧，在匹配总体计算预算的条件下：

| Warm-up steps | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|
| 0 | 12.8 | 0.486 | 0.522 |
| 4 | 18.5 | 0.693 | 0.267 |
| 8 | 21.7 | 0.778 | 0.183 |
| 11 | 23.2 | 0.813 | 0.151 |
| **12** | **23.6** | **0.823** | **0.146** |
| 13 | 23.5 | 0.819 | 0.150 |

前 12 步 dense 最优。完全取消 warm-up 会灾难性退化，说明早期 diffusion steps 的全局交互不可被同一静态 mask 安全替代。

### 12.3 4× 长度 warm-up steps

HunyuanVideo 509 帧：

| Warm-up steps | Vision Reward ↑ |
|---:|---:|
| 0 | 0.154 |
| 1 | 0.160 |
| **2** | **0.163** |
| 3 | 0.157 |

长视频训练后只需 2 个 dense warm-up steps。需要注意，这张表的绝对 Vision Reward 与主表口径不同，适合做同表相对比较，不宜直接与主表 0.134 混合。

### 12.4 Dense early layers

HunyuanVideo 117 帧，在相同计算预算下：

| Dense layers | Vision Reward ↑ |
|---:|---:|
| 0 | 0.139 |
| 1 | 0.156 |
| **2** | **0.163** |
| 3 | 0.157 |

两个 dense early layers 最优，进一步支持“浅层先建立全局结构，深层再稀疏细化”的混合架构。

### 12.5 与 Harmonic Series mask 比较

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Vision Reward ↑ |
|---|---:|---:|---:|---:|
| Harmonic Series | 27.0 | 0.881 | 0.119 | 0.136 |
| **Radial** | **27.3** | **0.886** | **0.114** | **0.139** |

二者均为 $O(n\log n)$，但 Radial 的指数层级 pattern 更匹配实测 energy decay，四项指标均略优。

---

## 13. 与 FlashAttention、SVG、线性注意力的关系

### 13.1 与 FlashAttention

FlashAttention 优化 dense attention 的 IO，保持精确结果；Radial Attention 改变计算图，直接删除关系。二者互补：Radial 决定算哪些 blocks，FlashAttention/FlashInfer 决定选中 blocks 如何高效计算。

### 13.2 与 SVG

SVG 为每个 head 在线判断 spatial 或 temporal mask，理论上更自适应；Radial 用统一 mask，把中央稠密带和远距离对角线放在一起。它牺牲逐 head 最优，换来：

- 无 profiling；
- 可用于训练；
- 避免长视频分布下 head 分类错误；
- 规则 block sparsity 更容易调度。

### 13.3 与线性 attention

线性 attention 可以达到 $O(n)$，但通常改变 softmax attention 的归纳偏置。论文中 SANA 替换虽然最快，却在 Hunyuan 253 帧得到 -0.205 Vision Reward，说明仅靠少量微调无法恢复预训练视频模型的行为。Radial 的优势是**保留 softmax 和预训练权重接口**，而不是复杂度阶数最低。

### 13.4 与通用 $O(n\log n)$ 稀疏模式

PowerAttention 和 Harmonic Series 说明：复杂度相同不等于质量相同。视频稀疏 attention 的关键不是“达到 log 复杂度”本身，而是连接图是否匹配时空相关性。Radial 的贡献主要是把经验衰减规律变成规则连接图。

---

## 14. 局限与批判性评价

### 14.1 作者明确局限

1. **指数衰减是假设性简化**：自然视频存在比距离更复杂的时空依赖。
2. **空间分辨率仍二次**：复杂度是 $O(s^2f\log f)$，不是对所有 token 轴统一次二次。
3. **未来仍需更强预训练策略**：作者提到结合 SANA、MoBA 等方法支撑更长、更高分辨率视频。

### 14.2 静态统一 mask 的适用边界

平均趋势不等于最坏情况。统一 mask 可能对以下关系分配不足：

- 跨远距离重现的主体身份；
- 大幅相机移动后同一物体落在完全不同空间 index；
- 多主体跨画面交互；
- 剪辑式镜头跳转；
- 非局部物理因果事件。

尤其是远距离部分主要保留 $k=l$ 的同位置连接。若相机运动导致内容在屏幕中大幅位移，远程相关 token 不再位于同一索引，radial prior 可能错过真正 correspondence。

### 14.3 论文没有系统 failure benchmark

定性图以较平滑、单镜头、主体连续的视频为主。论文没有专门按：

- 相机运动幅度；
- 遮挡与重现间隔；
- 运动速度；
- 镜头切换；
- 多主体交互；
- 分辨率；
- 视频语义复杂度

统计失败率，也没有正式用户研究。Vision Reward 不能完全替代人工偏好和任务特定一致性评价。

### 14.4 系统比较并非完全同条件

- STA 使用 FA3，Radial 使用 FA2；
- 默认长度与长视频测试的帧数、GPU 数和分辨率口径不同；
- 长视频 Hunyuan/Mochi 使用 8 H100，Wan 使用单 H100 推理；
- 不同方法的稀疏率相近但 block layout 与 kernel 易用性不同。

因此同模型同表的 speedup 可比，跨模型绝对秒数和跨 kernel 的纯算法速度不宜直接比较。

### 14.5 “4× 更长”不等于长程世界一致性

论文主要评价视觉质量、主体一致性和美学，没有闭环重访、场景记忆、对象持久状态或物理一致性指标。Radial Attention 让一次性生成更长、更便宜，但它不是 [[WorldCrafter - 隐式三维记忆视频世界模型]]、[[DecMem - 解耦记忆的分钟级一致世界生成]] 那类长期世界记忆机制。

### 14.6 个人判断

这篇论文最强的地方不是提出一个“物理定律”，而是完成了从**统计观察→规则 mask→复杂度证明→块稀疏 kernel→多模型验证→长度适配**的完整闭环。最需要警惕的是命名中的“radial”：形式化空间距离只是展平 index 距离，实际方法更接近“log-time bands + shrinking spatial diagonals”，并不是旋转对称的真正二维径向注意力。

---

## 15. 复现与实践建议

### 15.1 官方环境

官方仓库推荐：

- Linux + NVIDIA GPU；
- Python 3.12；
- PyTorch 2.5.1；
- CUDA 12.4；
- FlashAttention；
- FlashInfer；
- 可选 Sparse SageAttention v1/v2；
- 最新 Diffusers 用于 Wan2.2 / LightX2V 支持。

代码许可证为 Apache-2.0，但底层 HunyuanVideo、Wan、Mochi、OpenVid-1M 等需遵守各自许可证。

### 15.2 复现优先级

1. **默认长度、免训练替换**：先在固定 prompt/seed 下比较 dense 与 radial 的 PSNR/LPIPS、latency 和 PFLOPs；
2. **检查 kernel 转化率**：同时记录 mask sparsity、attention kernel time、端到端 latency；
3. **验证 warm-up**：复现 0/4/8/12 dense steps，确认质量崩溃点；
4. **长视频 LoRA**：先做 2×，再扩到 4×；不要把长度外推失败误判为 mask 失败；
5. **压力测试远程依赖**：设计主体消失重现、强相机位移、周期事件和多主体互动；
6. **比较二维位置定义**：尝试用真正的 2D Manhattan/Euclidean 邻域替代展平 index，评估质量与 block 规则性的权衡。

### 15.3 复现成本

| 目标 | 难度 | 主要成本 |
|---|---|---|
| 默认长度 Radial 推理 | 高 | 10B–14B 视频模型、H100 级显存、稀疏 kernel 编译 |
| 默认长度质量对齐 | 高 | 固定 seed、dense reference、长时间推理 |
| 2×/4× LoRA | 很高 | 8×H100、OpenVid 数据筛选、长视频训练 |
| 从头复现完整论文 | 很高 | 三套大模型、多种 baseline、不同 GPU/内核配置 |

论文的 LoRA 微调虽称“minimal”，绝对成本仍是 8×H100、8–21 小时。它相对 dense full FT 便宜，不等于低门槛。

---

## 16. 对后续研究的启发

### 16.1 Query-aware 或 content-aware Radial Attention

静态 radial mask 可作为保底骨架，再用少量动态 budget 增补强远程连接。例如每个 query 保留：

- radial static blocks；
- 少量 content top-k blocks；
- 第一帧/关键帧 sinks；
- 对象轨迹或相机几何引导的 correspondence blocks。

这样既保持规则 block 的硬件效率，又补偿静态距离先验对远程语义事件的遗漏。

### 16.2 与世界模型记忆结合

Radial Attention 解决 chunk 内/一次性长序列的二次成本，WorldCrafter、DecMem 等解决跨 chunk 的长期记忆。两者可以组合：

- chunk 内使用 Radial Attention；
- chunk 间通过显式/隐式 memory tokens；
- memory readout 作为额外全局 blocks 或 sinks；
- 用相机 pose 决定哪些远程 radial blocks 应被补回。

### 16.3 对 3DGS / 资产序列建模的启发

对 3DGS、mesh 或部件序列，可将“距离”从视频的时间/展平空间改成：

- Gaussian 图上的 geodesic 或 kNN 层级；
- mesh 拓扑 hop；
- 对象部件树距离；
- 帧间相机投影对应；
- 动作时间距离。

核心不是照搬 mask，而是沿结构距离建立**范围翻倍、密度减半**的多尺度稀疏图，使每个尺度成本近似恒定。

### 16.4 对 DiT 设计的启发

消融表明前两层和早期 diffusion steps 更依赖 dense attention。未来可显式设计：

1. 少量 global structure layers；
2. 多数 radial/local refinement layers；
3. 噪声高时 dense、噪声低时 sparse 的 timestep-adaptive schedule；
4. 用 head/layer 统计学习各自 $\alpha,\beta$，但量化到有限几种静态 block templates。

这可能比“所有层同一个 mask”获得更好质量—速度 Pareto。

---

## 17. 最终评价

**评级：★★★★★**

Radial Attention 是一篇完成度很高的基础模块论文。它没有发明复杂动态路由，而是从视频 DiT 已有 attention 分布中提炼一个稳定统计规律，用极简单的对数分带和空间对角收缩把它变成可证明、可训练、可部署的静态稀疏模式。默认长度、2× 和 4× 长度在 HunyuanVideo、Wan2.1、Mochi 三个 10B–14B 模型上都得到一致收益；附录还把 warm-up steps、dense layers、另一种 $O(n\log n)$ mask 和 LoRA rank/steps 交代清楚。

五星评价主要基于它的**方法—理论—系统—实验闭环**，而不是认为它已经解决长视频生成。其边界同样明确：空间维仍二次、mask 对内容无感、远距离主要依赖同位置连接、质量并非所有指标无损，而且 4× 长度仍需 8×H100 LoRA 微调。更准确地说，它为“如何把视频扩散中的冗余注意力变成真实 wall-clock 收益”提供了强基线，也为后续 content-aware、geometry-aware、memory-aware 稀疏注意力提供了结构骨架。

---

## 18. 材料来源

- [arXiv](https://arxiv.org/abs/2506.19852)
- [官方代码](https://github.com/mit-han-lab/radial-attention)
- [Apache-2.0 License](https://github.com/mit-han-lab/radial-attention/blob/main/LICENCE.txt)
- [[assets/Radial Attention - 时空能量衰减稀疏注意力/sources|本地来源记录]]
