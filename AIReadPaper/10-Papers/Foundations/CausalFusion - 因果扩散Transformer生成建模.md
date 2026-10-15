---
type: paper-note
status: done
domain: Foundations
paper: Causal Diffusion Transformers for Generative Modeling
year: 2024
arxiv: 2412.12095
doi: 10.48550/arXiv.2412.12095
source: https://arxiv.org/abs/2412.12095
project: https://github.com/causalfusion/causalfusion
code: https://github.com/causalfusion/causalfusion
tags:
  - Foundations
  - Diffusion
  - Autoregressive
  - Causal-Transformer
  - Multimodal-Generation
  - ImageNet
created: 2026-09-04
updated: 2026-09-04
---

# CausalFusion：因果扩散 Transformer 生成建模

## 基本信息

- **论文**：[Causal Diffusion Transformers for Generative Modeling](https://arxiv.org/abs/2412.12095)
- **作者**：Chaorui Deng、Deyao Zhu、Kunchang Li、Shi Guang、Haoqi Fan
- **机构**：ByteDance Research
- **版本**：arXiv:2412.12095v2，2024-12-17，21 页
- **官方代码**：https://github.com/causalfusion/causalfusion
- **任务**：ImageNet 类条件生成、文生图、图像描述、视觉表征迁移、零样本局部图像编辑
- **重要性评估**：★★★★☆（4/5）

## 一句话结论

CausalFusion 把自回归的“沿 token 序列分解”与扩散的“沿噪声级别分解”放进同一个二维生成平面：每个自回归阶段只对一组 token 做扩散去噪，并因果条件于此前已经生成的干净 token。这个设计把 DiT 改造成兼容 GPT/LLaMA 风格的 decoder-only Transformer，在 ImageNet 256×256 上以 1B 参数取得 FID 1.57，同时自然获得随机顺序补全、零样本局部编辑和图文联合建模能力；但其“任意长度推理”和“AR/扩散平滑切换”更多是架构潜力，论文并未用长序列推理 benchmark 或完整速度曲线充分证明。

---

## 1. 研究背景与问题定义

### 1.1 两种生成范式各自擅长什么

自回归模型与扩散模型都在学习高维数据的联合分布，但二者选择了不同的因子分解方向。

自回归模型把数据视为有序 token 序列：

$$q(x_{1:L})=q(x_1)\prod_{l=2}^{L}q(x_l\mid x_{1:l-1})$$

它沿**序列轴**逐 token 展开，优势是序列长度可变、上下文可以持续增长，天然适合语言建模和 in-context reasoning；代价是早期 token 可见上下文少、误差容易累积，而且图像若逐 token 生成，推理串行度很高。

扩散模型则沿**噪声轴**展开。前向过程逐步加噪：

$$q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$$

联合分布写成：

$$q(x_{0:T})=q(x_0)\prod_{t=1}^{T}q(x_t\mid x_{t-1})$$

反向过程学习：

$$p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t),\Sigma_\theta(x_t))$$

扩散模型在每个噪声级别并行处理全部图像 token，适合连续视觉信号；它可以通过增加采样步数换取质量，但通常不能像语言模型那样自然地扩展到任意长度的 token 序列。

### 1.2 现有 AR+Diffusion 组合的局限

此前的 DART、MAR、BiGR、Transfusion 等工作已经尝试结合 AR 与扩散，但多半把二者当作两个相邻模块：例如先自回归地产生高层表示，再交给扩散头解码；或者文本走 AR、图像走扩散，只在同一网络里并置两类损失。CausalFusion 的观点是：**AR 和 diffusion 的核心都不是某个特定网络模块，而是概率分布的因子分解方式**。因此更自然的统一方式，是让同一批数据同时沿序列轴和噪声轴因子化。

### 1.3 核心贡献

1. 提出 **Causal Diffusion / CausalFusion**，用“token 分组 × 噪声级别”的双重因子分解，把 AR 与 diffusion 统一在同一个 decoder-only Transformer 中。
2. 系统分析训练难度：AR step 越靠后、每步预测 token 越少，任务越容易，训练损失虽低却可能学成局部插值；据此提出指数衰减的 AR-step 采样、随机 token 顺序和 AR 轴损失重加权。
3. 在 ImageNet 256×256/512×512 上取得有竞争力的生成结果，并展示零样本局部编辑、图文双向生成与视觉表征迁移。
4. 将视觉条件、时间条件和文本都作为 in-context token 处理，使架构比 AdaLN 风格 DiT 更接近 GPT/LLaMA 的 decoder-only 形式。

![](assets/CausalFusion%20-%20因果扩散Transformer生成建模/figures/01-dual-factorization.png)

> 图 1：CausalFusion 的“双重因子分解”视角。纵轴是扩散噪声级别，横轴是 AR step；一次生成既沿 token 序列前进，也在当前 token 组内完成扩散去噪。右下角显示 In-context DiT 在参数更少时已明显优于原始 DiT，CausalFusion 又在不改变架构和参数量的前提下进一步降低 FID。  
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2412.12095

---

## 2. 任务定义与输入输出

### 2.1 图像 token 化

在 256×256 消融实验中，图像先经 Stable Diffusion VAE 压缩为 32×32 latent，再以 2×2 patchify 得到 $L=256$ 个连续 latent token。系统级模型改用 MAR VAE：16× 下采样、16 个 latent channel。模型的目标不是直接对离散视觉词表做交叉熵，而是对当前 token 组执行连续空间的扩散噪声预测。

令 $S$ 为总 AR step 数，$\kappa_s$ 为第 $s$ 个 AR step 负责的 token 索引集合，$|\kappa_s|$ 是该步要生成的 token 数。$x_{t,\kappa_s}$ 表示这些 token 在扩散时刻 $t$ 的带噪状态，$x_{0,\kappa_{1:s-1}}$ 表示此前 AR step 已经生成完毕的干净 token。

### 2.2 输入与输出

- **输入**：条件 token（类别或文本）、此前 AR step 的干净图像 token、当前 token 组的带噪表示、扩散时间步 embedding。
- **输出**：当前 token 组的噪声预测，随后通过 DDPM 反向过程得到更干净的 token。
- **生成顺序**：训练时默认随机打乱图像 token，再随机划分为若干组；推理时可指定 AR step 数与 token 顺序。
- **两个端点**：$S=1$ 时一次处理全部 token，退化为常规扩散式生成；$S=L$ 时每步仅处理一个 token，接近纯 AR 生成。

### 2.3 双重因子分解

论文把标准扩散的式（2）扩展为：

$$q(x_{0:T,\kappa_s}\mid x_{0,\kappa_{1:s-1}})=q(x_{0,\kappa_s})\prod_{t=1}^{T}q(x_{t,\kappa_s}\mid x_{t-1,\kappa_s},x_{0,\kappa_{1:s-1}})$$

该式的关键不是数学形式有多复杂，而是条件关系发生了变化：当前 token 组不仅依赖自身上一噪声级别，还依赖所有已生成的干净 token。于是模型需要在所有 $s,t$ 上逼近：

$$p_\theta(x_{t-1,\kappa_s}\mid x_{t,\kappa_s},x_{0,\kappa_{1:s-1}})$$

这就是“next-token(s) diffusion”：不是一次扩散整张图，也不是逐 token 做离散分类，而是逐组推进，每组内部再做连续扩散。

---

## 3. 核心方法

### 3.1 从 DiT 改造成 decoder-only Transformer

原始 DiT 用 AdaLN-zero 把类别与扩散时间注入每层 block，这种条件注入方式与 GPT/LLaMA 的“所有输入都是 token”不一致。CausalFusion 先构造一个 In-context DiT 基线：

1. 类别条件改成普通上下文 token；消融阶段默认 4 个 class token，系统级比较使用 64 个 class token。
2. 时间步不再依赖 AdaLN，而是把 timestep embedding 加到图像 token embedding 上。
3. 自注意力加入 head-wise QK normalization，配合 40 epoch learning-rate warmup 与 batch size 2048，稳定大批量训练。
4. 移除 AdaLN-zero 后，In-context DiT-L/2 约 308M 参数；原始 DiT-L/2 为 458M 参数。

Table 1 的逐项改造很能说明问题：原始 in-context 版本去掉 AdaLN-zero 后 FID-10k 恶化到 26.71；加入新训练 recipe 后为 21.94，再加入 timestep embedding、QK-norm、warmup，最终达到 13.78，接近论文引用的 DiT-XL/2 12.92，同时参数更少。

### 3.2 广义因果注意力

![](assets/CausalFusion%20-%20因果扩散Transformer生成建模/figures/03-dit-vs-causalfusion.png)

> 图 3：DiT 与 CausalFusion 的概念对比。DiT 对全部带噪图像 token 做全注意力，并通过 AdaLN 注入条件；CausalFusion 每次只去噪随机 token 子集，该子集可以看到此前的干净 token 与条件 token，但过去的干净 token不能反向读取当前噪声 token。  
> 来源：论文 Figure 3，第 4 页，https://arxiv.org/abs/2412.12095

广义 causal mask 维持三条约束：

1. 当前 $x_{t,\kappa_s}$ 可以看条件 token；
2. 当前 $x_{t,\kappa_s}$ 可以看此前所有干净 token $x_{0,\kappa_{1:s-1}}$；
3. 前序干净 token 不能看到当前带噪 token，当前不同 AR group 之间也不能越过因果边界泄漏未来信息。

这使同一 decoder-only Transformer 可以同时处理文本 token、类别 token、干净图像 token 与带噪图像 token。与 masked autoencoder 相似，模型被迫在“部分可见”的上下文中恢复被遮蔽区域；区别在于它恢复的是一组连续 token 的完整扩散轨迹，而不只是一次性重建。

### 3.3 为什么随机 AR step 不能直接均匀采样

乍看之下，只要训练时从 $S\in[1,L]$ 均匀采样，就能覆盖从纯扩散到纯 AR 的全部模式。但 token 组大小近似为 $L/S$：当 $S$ 均匀采样时，大量样本落在较大的 $S$，导致超过 95% 的 AR step 只需预测不超过 16 个 token。模型因此大多数时间都能看到大量干净上下文，任务退化为局部插值。

![](assets/CausalFusion%20-%20因果扩散Transformer生成建模/figures/04-task-difficulty.png)

> 图 4：任务难度失衡的证据。左图显示 AR step 越多，训练 loss 越低；中图显示均匀采样会极端偏向小 token 组；右图显示越靠后的 AR step 验证 loss 越低。低 loss 在这里不代表生成更好，而是任务变得过于简单。  
> 来源：论文 Figure 4，第 5 页，https://arxiv.org/abs/2412.12095

固定 AR step 的结果进一步说明了这一点：

| 训练 AR step | $S_{eval}=1$ | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|
| 1 | **13.78** | 356.69 | 404.67 | 390.18 |
| 2 | 16.69 | **14.77** | 47.49 | 136.04 |
| 4 | 24.14 | 15.37 | **18.13** | 33.14 |
| 8 | 54.08 | 24.49 | 22.66 | **20.01** |
| 256（纯 AR） | 313.28 | 321.62 | 261.26 | 192.25 |
| 均匀随机 | 21.31 | 22.17 | 23.54 | 25.05 |

固定 $S$ 训练的模型在训练/推理步数不匹配时几乎崩溃；纯 AR 训练在给出的 240 epoch 设置中也很差。均匀随机虽然跨步数更稳定，却不如 $S=1$ 的 In-context DiT。这表明真正困难的不是“允许任意 AR step”，而是**如何让不同难度的生成状态都提供有效梯度**。

### 3.4 指数衰减采样与随机 token 顺序

论文用 $\gamma\le1$ 控制 AR step 的指数衰减采样：

- $\gamma=1$：均匀采样，FID-10k 为 21.31；
- $\gamma=0$：退化为单 AR step 的 In-context DiT；
- $\gamma=0.9$：默认值，$S_{eval}=1/2/4/8$ 时分别为 12.89/15.57/18.83/22.72；
- $\gamma=0.8$：单步 FID 略优到 12.78，但多步设置稍差，因此未选为默认值。

指数衰减增加小 $S$ 的采样概率，也就是增加“大 token 组、少可见上下文”的困难任务。与此同时，随机 token 顺序 FID 12.89，优于 raster 14.46、8×8 block raster 14.76、4×4 block raster 14.62 与 dilated order 15.54。作者的解释是：固定栅格顺序会让当前区域过度依赖空间邻居，随机顺序则迫使模型学习非局部关系。

### 3.5 AR 轴损失重加权

标准 DDPM 噪声预测目标为：

$$\min_\theta\mathbb{E}_{x_0,\epsilon,t}\left[w(t)\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right]$$

CausalFusion 把权重扩展为 $w(s,t)$。论文没有把完整损失单独编号，但实现思想很直接：第一个 AR step 的权重设为 $\lambda\ge1$，随后沿 AR 轴线性衰减到最后一步的 1；扩散时间 $t$ 仍使用常数权重。这样模型会更重视上下文最少、最容易出错的早期生成阶段。

| AR 权重（首步→末步） | $S_{eval}=1$ | 2 | 4 |
|---|---:|---:|---:|
| 1→1 | 12.89 | 15.57 | 18.83 |
| 1.5→1 | 12.61 | 15.49 | 18.32 |
| 2→1 | **12.13** | **15.15** | 18.09 |
| 2.5→1 | 12.32 | 15.22 | 17.99 |
| 3→1 | 12.50 | 15.28 | **17.92** |

这组结果支持“早期 AR step 更有价值”的判断，但最优 $\lambda$ 会随推理 AR step 改变，并不存在一个对所有模式都最优的统一权重。

### 3.6 扩散时间采样与条件 token

不同 AR step 共用同一扩散时间 $t$ 时 FID-10k 为 12.13；每步随机不同 $t$ 为 12.27；每步采 4 或 8 个 $t$ 为 12.19/12.23，差异很小。因为干净 token 可以在多个噪声时刻间共享，额外训练计算量约 10%。

类别条件方面，4 个独立 token 的 FID 为 12.13；64 个独立 token 改善到 11.84，却增加 62.5M 参数。将 4 个 token 各重复 16 次，参数只增加 3.9M，却达到 11.75。这个结果提示：条件 token 获得的**计算预算**可能比独立参数量更重要。

### 3.7 推理模式与计算复杂度

理论上，CausalFusion 可在不同 $S_{eval}$ 下工作：

- $S_{eval}=1$：一次处理全部图像 token，最接近传统扩散；
- 中间值：逐组 next-token(s) diffusion；
- $S_{eval}=L$：逐 token 的纯 AR 极端。

每个 AR step 只扩散局部 token，论文声称总计算复杂度与 DiT 相近或略低。但主结果全部采用 $S_{eval}=1$ 与 DDPM 250 步；论文没有给单图延迟、吞吐、FLOPs、显存，亦没有给 $S_{eval}$ 从 1 增大到 256 的速度—质量曲线。因此不能把它解读为一个已经证实更快的采样器。

---

## 4. 数据集与实验设置

### 4.1 消融实验

- 数据：ImageNet-1K，256×256 类条件生成；
- 模型：24 层、hidden size 1024、16 heads；
- 图像表示：SD VAE，8× 下采样，4 latent channels，2×2 patch；
- 优化器：AdamW，$\beta_1=0.9,\beta_2=0.95$，weight decay 0；
- 学习率：$10^{-4}$，constant schedule，40 epoch warmup；
- batch size：2048；训练 240 epochs；
- 采样：DDPM 250 steps；评价：ADM 实现的 FID-10k。

### 4.2 系统级比较

| 模型 | Hidden | 层数 | Heads | 参数量 |
|---|---:|---:|---:|---:|
| CausalFusion-L | 1024 | 24 | 16 | 368M |
| CausalFusion-XL | 1280 | 32 | 20 | 676M |
| CausalFusion-H | 1408 | 40 | 22 | 1B |

系统级模型改用 MAR VAE、16× 下采样、16 latent channels、learnable positional embedding，训练 800 epochs、batch size 2048，评估 FID-50k。主结果默认单 AR step、DDPM 250 steps。

### 4.3 多模态训练

多模态版本继续对图像使用扩散损失，对文本使用 GPT 式 next-token prediction：

- 90% 样本排列为“文本→图像”，只计算图像损失；
- 10% 样本排列为“图像→文本”，同时计算 caption 文本损失与用于 CFG 的图像损失；
- 文本损失权重为 0.01；
- tokenizer 为 LLaMA-3 tokenizer，最大文本长度 35；
- 数据为 ImageNet 重描述集，每张图由 Qwen2VL-7B 生成 10 条 caption；
- 训练 240 epochs，在 MSCOCO 上零样本评测 FID-30k 与 CIDEr。

---

## 5. 实验结果

### 5.1 ImageNet 主结果

| 模型 | 参数量 | 256 无 CFG FID | 256 有 CFG FID | 512 有 CFG FID |
|---|---:|---:|---:|---:|
| DiT-XL | 675M | 9.62 | 2.27 | 3.04 |
| SiT-XL | 675M | 8.30 | 2.06 | 2.62 |
| MAR-L | 479M | 2.60 | 1.78 | **1.73** |
| CausalFusion-L | 368M | 5.12 | 1.94 | — |
| CausalFusion-XL | 676M | **3.61** | **1.77** | 1.98 |

CausalFusion-XL 相对 DiT-XL 的提升很明显：参数量相近，无 CFG 时 FID 从 9.62 降到 3.61，有 CFG 时从 2.27 降到 1.77；512×512 时从 3.04 降到 1.98。CausalFusion-L 用约一半于 DiT-XL 的参数，在有 CFG 时也达到 1.94。

大模型比较中，CausalFusion-H 用标准 DDPM 250 步达到 FID 1.64，加入 CFG interval 后为 1.57；与之相比，DiMR-G/2R 为 1.63，MAR-H 为 1.55。因采样器、tokenizer 与 sampling trick 不同，这里更适合看作“进入第一梯队”，而不是严格证明绝对 SOTA。

### 5.2 零样本局部图像编辑

![](assets/CausalFusion%20-%20因果扩散Transformer生成建模/figures/02-zero-shot-editing.png)

> 图 2：上半部分是 512×512 ImageNet 类条件样本；下半部分展示中心、上半部或下半部遮挡后的重新生成。模型仅接受 ImageNet 类条件生成训练，却能依据未遮挡区域与新类别标签完成局部重绘。  
> 来源：论文 Figure 2，第 2 页，https://arxiv.org/abs/2412.12095

编辑能力来自训练时的随机 token 分组：模型本来就在学习“给定一部分干净 token，重建另一部分 token”，因此可把未遮挡区域当作上下文，把遮挡区域当作待扩散的 token 组。这种能力不需要额外编辑微调。

但实验只给定性样例，没有编辑 FID、LPIPS、边界一致性、用户偏好或失败率。图中任务主要是 ImageNet 类别替换，尚不足以证明模型能处理复杂自然语言指令、精细属性绑定或大范围结构变化。

### 5.3 多模态联合建模

![](assets/CausalFusion%20-%20因果扩散Transformer生成建模/figures/05-multimodal-generation.png)

> 图 5：同一个 CausalFusion-XL 同时执行文生图与图像描述。图像 token 用扩散损失，文本 token 用 next-token loss，二者通过统一的因果 token 序列组织。  
> 来源：论文 Figure 5，第 7 页，https://arxiv.org/abs/2412.12095

在对齐架构、VAE、tokenizer 与训练数据后，论文自行实现 Transfusion 作为对照：

| 模型 | 训练数据 | MSCOCO FID-30k ↓ | Zero-shot CIDEr ↑ |
|---|---|---:|---:|
| Transfusion-L | IN1KCap 1M | 8.1 | 34.5 |
| CausalFusion-L | IN1KCap 1M | **7.1** | **47.9** |

这是一个有价值的 proof-of-concept：同一 decoder-only 主干可以让离散文本走 AR、连续图像走 diffusion。不过数据规模只有约 1M ImageNet 重描述图像，且 Transfusion 是作者复现而非官方实现，因此不能直接外推到大规模开放域多模态模型。

### 5.4 视觉表征迁移

将 ImageNet 生成预训练模型用于分类与 COCO captioning 微调：

| 模型 | FID-10k ↓ | ImageNet Acc ↑ | COCO CIDEr ↑ |
|---|---:|---:|---:|
| DiT | 18.2 | 83.5 | 94.4 |
| CausalFusion | 11.8 | 84.2 | 98.0 |
| CausalFusion† | **9.3** | **84.7** | **103.2** |

† 使用 MAR VAE 并直接预测 latent，而不是预测噪声。

作者认为随机分组 token diffusion 隐式执行了 masked representation prediction，因此不仅提升生成，也学到更强视觉表征。这个解释与结果一致，但尚缺少更广泛的分割、检测、深度等迁移任务验证。

### 5.5 实验是否充分支撑主张

实验充分支撑了三个较窄的结论：

1. 顺序因子分解能显著提升 DiT 式 ImageNet 生成；
2. AR step 的训练难度分布必须被主动校正；
3. 随机分组机制带来局部补全与表征迁移收益。

但它对更宏大的主张支撑不足：没有长序列 in-context reasoning benchmark，没有开放域大规模图文预训练，没有纯 AR 高质量生成结果，也没有证明多 AR step 推理在实际系统中更高效。

---

## 6. 与相关工作的关系

### 6.1 相对 DiT

CausalFusion 保留 DiT 的连续 latent 扩散，但去掉 AdaLN 式专用条件注入，将所有模态视为上下文 token，并引入按 token 组划分的因果注意力。其提升不仅来自双重因子分解，也来自 timestep embedding、QK-norm、warmup、大 batch 与条件 token 扩展等训练 recipe，因此不能把所有增益都归因于一个单独模块。

### 6.2 相对 MAR

MAR 也是连续 token 的 AR+Diffusion：AR 主干逐位置预测，位置上的连续值由小 diffusion head 建模。CausalFusion 不把扩散降为 token 级输出头，而是让主 Transformer 本身在随机 token 组上执行扩散，并把 AR step 数作为可调维度。MAR-H 的最佳 ImageNet FID 1.55 略好于 CausalFusion-H 1.57，但 MAR 使用 AR 256 + DDPM 100 与温度采样，推理机制不同。

### 6.3 相对 Transfusion

Transfusion 把文本 AR 与图像扩散拼接在统一序列中，但图像块内部仍主要按标准 diffusion 处理。CausalFusion 进一步把图像 token 自身也沿序列轴因子化，使它能自然地把一部分干净图像 token 作为上下文，对另一部分执行扩散。

### 6.4 对后续统一多模态模型的启发

这篇论文的重要性不只在 FID，而在于提出一个可复用的设计原则：**离散/连续不是必须分开建模的边界；真正需要统一的是条件依赖与因子分解。** 对视频、3D latent、动作轨迹等连续序列，可以考虑让 token 组内部使用 diffusion/flow matching，组间使用 causal prediction，从而兼顾连续输出质量与长序列上下文。

---

## 7. 局限与批判性评价

### 7.1 “平滑切换”仍有条件

固定 $S$ 训练在不同 $S_{eval}$ 下会严重崩溃；即便采用随机 AR step 与衰减采样，$S_{eval}$ 增大时 FID 仍从 12.89 逐步恶化到 22.72。论文证明的是“同一模型可以在多个模式下工作”，而不是“各模式质量等价地平滑过渡”。

### 7.2 主结果实际仍是单步 AR 的扩散模式

最强 ImageNet 结果默认 $S_{eval}=1$，即一次处理全部图像 token。双重因子分解最独特的多 AR-step 推理并未成为主结果配置；因此其性能收益更像一种训练时的随机部分观测正则化，而非已经成熟的混合采样方案。

### 7.3 计算与可扩展性证据不足

论文称计算复杂度与 DiT 相近或略低，却没有实测延迟、吞吐、FLOPs、显存或成本。系统级模型训练 800 epochs、batch 2048，但未披露 GPU 型号、数量、训练天数、GPU-hours 与总能耗。对普通研究团队而言，完整复现实验成本仍较高。

### 7.4 多模态验证规模偏小

图文实验仅基于 ImageNet 重描述数据与短 caption，文本最长 35 token。它证明了机制可行，但不能说明模型已经具备复杂语言推理、长上下文、多轮生成或开放世界知识。

### 7.5 缺少系统失败案例与风险分析

论文没有独立 Limitations 章节，也没有报告 caption hallucination、编辑失败率、复杂组合生成、文字渲染、计数、数据偏差、安全性与授权问题。零样本编辑只给精选定性结果，应谨慎解读。

### 7.6 个人评价

**我认为它是一篇“统一视角很强、证据边界也很清楚”的基础架构论文。** 最值得复用的不是 1.57 FID，而是“训练难度分布决定双轴因子化是否有效”这一发现：如果多数训练样本都能借助大量可见上下文轻易完成，模型会得到更低 loss，却学不到更强的生成能力。指数采样、随机顺序、早期 step 加权三者共同构成了这篇工作的真正技术核心。

---

## 8. 复现与实践建议

### 8.1 官方资源状态

官方仓库已公开：

- `train.py`、`sample.py`、模型与 diffusion 实现；
- CausalFusion-XL 预训练权重下载链接；
- 8 GPU 训练与采样命令；
- 指定依赖：PyTorch 2.5.1、torchvision 0.20.1、transformers 4.46.2；
- 外部依赖：MAR 的预训练 VAE 与 ADM 评估代码。

仓库 README 未明确许可证，发布内容主要面向 ImageNet 类条件生成；论文中的多模态联合训练、H 模型及全部系统级复现细节是否完整覆盖，需要进一步核对代码。

### 8.2 最小复现实验

若目的是验证论文核心结论，而不是追求 FID 1.57，建议按以下顺序：

1. 先复现 240 epoch 的 In-context DiT-L/2，确认 timestep embedding、QK-norm、warmup 后 FID-10k 接近 13.78；
2. 实现 generalized causal mask，固定 $S\in\{1,2,4,8\}$，复现训练/推理不匹配导致的 FID 崩溃；
3. 加入随机 AR step，比较均匀采样与 $\gamma=0.9$ 指数衰减；
4. 比较 raster 与 random token order；
5. 加入 $2\to1$ 的 AR 轴损失权重，验证单步 FID 从 12.89 改善到约 12.13；
6. 最后再切换到 MAR VAE、64 个条件 token 与 800 epoch 系统级训练。

### 8.3 资源与风险预估

论文的消融默认 batch 2048，官方 README 的示例则是在 8 GPU 上以 global batch 256 训练 CausalFusion-XL；这意味着直接照论文系统配置复现需要梯度累积或更多 GPU。主评测还需生成 50K 图像并使用 ADM 流程计算 FID。若只做概念验证，可先缩小模型、减少 epoch，并保留相同的 AR-step 分布与注意力掩码，优先观察相对趋势而非绝对 FID。

---

## 9. 个人启发与后续问题

### 9.1 可迁移到视频与动作序列

视频生成和 VLA 都同时具有“连续值输出”与“长序列因果结构”。可以把一段未来帧或一组动作 chunk 作为 $\kappa_s$，组内用 diffusion/flow matching，组间用 causal context。这样既避免逐标量 AR，又保留滚动生成和任意长度能力。

### 9.2 训练难度比形式统一更关键

CausalFusion 最有普适性的经验是：统一架构并不自动带来统一能力。若采样分布让任务过易，模型会依赖局部上下文；应同时设计 token 分组分布、顺序、mask ratio、噪声级别和 loss weighting。这个结论与 MAE 的 mask ratio、diffusion 的 SNR weighting、课程学习和 hard-example mining 本质相通。

### 9.3 值得继续追问的问题

1. 用 flow matching 替代 DDPM 后，双重因子分解是否仍有同样收益？
2. 能否学习自适应 token 分组与生成顺序，而不是随机或手工指定？
3. 多 AR step 是否能通过 KV cache 或并行调度真正降低墙钟时间？
4. 在视频、3D latent、机器人 action chunk 上，$S$ 与组大小如何共同决定长程一致性？
5. 若用大规模交错图文数据训练，随机图像 token 分组是否会带来更强的多模态 in-context reasoning？
6. 纯 AR 极端表现差，是训练配方问题、连续 token diffusion head 问题，还是视觉序列本身不适合逐 token 生成？

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2412.12095 | 全文 | 精读主来源 |
| `figures/01-dual-factorization.png` | 总览图 | 同上 | Figure 1, p.1 | 双重因子分解 |
| `figures/02-zero-shot-editing.png` | 生成与编辑结果 | 同上 | Figure 2, p.2 | 定性结果 |
| `figures/03-dit-vs-causalfusion.png` | 架构对比 | 同上 | Figure 3, p.4 | 核心方法 |
| `figures/04-task-difficulty.png` | 难度分析 | 同上 | Figure 4, p.5 | 采样与加权动机 |
| `figures/05-multimodal-generation.png` | 多模态样例 | 同上 | Figure 5, p.7 | 图文联合建模 |

完整溯源记录见：[sources.md](assets/CausalFusion%20-%20因果扩散Transformer生成建模/sources.md)。
