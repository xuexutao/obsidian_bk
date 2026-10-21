# Improved DDPM：可学习方差与扩散规模化

> [!info] 论文信息
> - **原题**：Improved Denoising Diffusion Probabilistic Models
> - **作者**：Alex Nichol, Prafulla Dhariwal（Equal Contribution）
> - **会议 / 年份**：ICML 2021
> - **arXiv**：[2102.09672](https://arxiv.org/abs/2102.09672)，v1，2021-02-18
> - **官方实现**：[openai/improved-diffusion](https://github.com/openai/improved-diffusion)
> - **本地原文**：[[assets/Improved DDPM - 可学习方差与扩散规模化/Improved_DDPM_arXiv_2102.09672.pdf]]
> - **来源与核验记录**：[[assets/Improved DDPM - 可学习方差与扩散规模化/sources]]
> - **关键词**：DDPM、learned variance、hybrid objective、cosine schedule、importance sampling、fast sampling、precision/recall、scaling law

## 1. 一句话总结

**Improved DDPM（常写作 IDDPM）并没有推翻 DDPM，而是围绕反向方差、训练目标、噪声日程和时间步采样做了几项克制但关键的改造：在基本不牺牲样本质量的前提下显著改善对数似然，并让训练 4000 步的模型用约 50–100 个反向步仍能生成高质量样本；同时，论文用 precision/recall 与 scaling 实验说明扩散模型比同级 GAN 覆盖更多模式，而且会随模型和算力平滑改善。**

## 2. 论文试图解决什么问题

### 2.1 DDPM 已经“会画”，但还不够像一个完整的概率模型

Ho et al.（2020）的 DDPM 已经在 CIFAR-10、LSUN 上展示出很强的样本质量，但留下四个问题：

1. **似然不够有竞争力**：FID 很好不等于模型覆盖了整个数据分布；DDPM 的 bits/dim 落后于自回归模型和 VAE。
2. **反向方差被固定**：原 DDPM 主要学习均值（等价地预测噪声），将反向过程方差固定为 $\beta_t$ 或后验方差 $\tilde\beta_t$，`L_simple` 本身不给方差任何学习信号。
3. **采样链太长**：通常要几百到上千次网络前向，部署成本高。
4. **规模化规律不清楚**：在 ImageNet 这类高多样性数据上，模型容量和训练算力是否能稳定换来更好结果，当时缺少系统证据。

论文的中心问题不是“能否再造一个新扩散框架”，而是：**能否保留 DDPM 稳定、简单的训练方式，同时把似然、采样速度、分布覆盖和规模化都补齐？**

### 2.2 三组容易混淆但必须分开的目标

- **NLL / bits-per-dim**：衡量概率模型给真实数据分配的概率，越低越好；对不可感知的像素级细节非常敏感。
- **FID / precision**：更偏向感知质量和保真度，越低的 FID、越高的 precision 通常意味着“像真的”。
- **recall**：更偏向模式覆盖，越高表示模型覆盖的数据流形范围越大。

本文一个贯穿始终的发现是：**这些目标并不自动一致。直接优化 VLB 能改善 NLL，却可能显著恶化 FID；`L_hybrid` 的价值正是在两者之间取得工程上更好的平衡。**

## 3. DDPM 基线：理解改进前先固定符号

### 3.1 前向加噪过程

对真实样本 $x_0\sim q(x_0)$，前向过程每一步加入方差为 $\beta_t$ 的高斯噪声：

$$q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),\quad q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t\mathbf I)$$

令 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$，则可跳过中间步骤直接采样任意噪声时刻：

$$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)\mathbf I),\quad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

这里 $\bar\alpha_t$ 是累计信号保留率，$1-\bar\alpha_t$ 是累计噪声方差。后文的 cosine schedule 本质上就是不再先规定 $\beta_t$，而是直接设计 $\bar\alpha_t$ 的形状。

### 3.2 真实后验与学习到的反向过程

给定 $x_0$ 时，单步后验仍是高斯分布：

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t(x_t,x_0),\tilde\beta_t\mathbf I),\quad \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

生成时不知道 $x_0$，因此用神经网络近似反向条件分布：

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$$

DDPM 通常不直接预测 $\mu_\theta$，而是预测加入的噪声 $\epsilon$：

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

### 3.3 `L_vlb` 与 `L_simple` 的张力

负变分下界可分解到各个时间步：

$$L_{\mathrm{vlb}}=L_0+L_1+\cdots+L_{T-1}+L_T$$

其中中间项是 $q(x_{t-1}\mid x_t,x_0)$ 与 $p_\theta(x_{t-1}\mid x_t)$ 的 KL 散度。原 DDPM 实践中更常用重加权后的噪声预测均方误差：

$$L_{\mathrm{simple}}=\mathbb E_{t,x_0,\epsilon}\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right]$$

`L_simple` 的优点是梯度稳定、样本质量好；缺点是它不依赖 $\Sigma_\theta$，因此无法学习反向方差，也不是严格按似然目标加权每个时间步。

## 4. 核心贡献一：在窄区间内学习反向方差

### 4.1 为什么固定方差对 FID 影响小，却会拖累 NLL

![[assets/Improved DDPM - 可学习方差与扩散规模化/fig01_variance_and_vlb_terms.png]]

*图 1：原文 Figure 1–2。除接近 $t=0$ 的区域外，$\tilde\beta_t$ 与 $\beta_t$ 很接近；但 VLB/NLL 的主要贡献恰好集中在最前面的少数时间步。*

这两张图共同解释了一个看似矛盾的现象：

- 当扩散步数足够多时，大部分时刻都有 $\tilde\beta_t\approx\beta_t$，所以固定成上界或下界，对宏观样本质量影响不大。
- 但最靠近数据端的少数反向步骤决定像素级微小细节，并对 NLL 贡献最大；这里的方差误差即使视觉上不明显，也会被 log-likelihood 放大。

因此，**学习方差的主要收益首先体现在概率校准，而不是“让图像立刻更好看”**。

### 4.2 learned variance interpolation

直接回归方差或 log-variance 很不稳定，因为合理范围很窄。作者让网络多输出一个与数据同维的向量 $v$，在两个有理论意义的端点之间做 log-space 插值：

$$\Sigma_\theta(x_t,t)=\exp\left(v\log\beta_t+(1-v)\log\tilde\beta_t\right)$$

关键细节：

1. **在对数域插值**：等价于对方差做几何插值，适合跨数量级的正数尺度。
2. **端点有明确含义**：$\beta_t$ 与 $\tilde\beta_t$ 分别是原 DDPM 使用过的上、下界式选择。
3. **论文没有把 $v$ 硬限制在 $[0,1]$**：理论上允许外插；实验中网络没有明显跑出区间，说明这个区间已足够表达。
4. **均值和方差共享网络主体**：不是另训一个方差模型，只是在输出端增加通道，因此增量计算很小。

### 4.3 为什么这个参数化会意外帮助少步采样

当采样只选训练时间步的一个子序列时，单步跨越的噪声区间变大。固定训练方差并不能自然适配新的大步长，而插值参数 $v$ 可以在重新计算的 $\beta_{S_t}$ 与 $\tilde\beta_{S_t}$ 之间复用。换言之，网络学到的不是某个绝对方差，而更像是**“在当前时间步的合理方差区间里应取什么相对位置”**。这正是 learned variance 能迁移到短链采样的机制基础。

## 5. 核心贡献二：hybrid objective 把均值与方差的职责拆开

### 5.1 目标函数

由于 `L_simple` 不训练方差，论文加入一个很小的 VLB 项：

$$L_{\mathrm{hybrid}}=L_{\mathrm{simple}}+\lambda L_{\mathrm{vlb}},\quad \lambda=0.001$$

作者还对 VLB 分支中的 $\mu_\theta$ 施加 **stop-gradient**：

- $L_{\mathrm{simple}}$ 主要负责学习均值/噪声预测，保住 DDPM 已验证的感知质量和训练稳定性。
- $L_{\mathrm{vlb}}$ 主要把学习信号送给 $\Sigma_\theta$，改善概率校准。
- 很小的 $\lambda$ 防止数值尺度较大的 VLB 压过 `L_simple`。

这不是简单的“两个 loss 相加”，而是一个明确的**职责分离**：用感知上更有效的重加权目标训练均值，用概率上正确的目标训练方差。

### 5.2 为什么直接优化 `L_vlb` 反而更难

![[assets/Improved DDPM - 可学习方差与扩散规模化/fig03_objectives_and_ablation.png]]

*图 2：原文 Figure 6–7 与 Table 1–2。均匀采样时间步时，纯 VLB 的梯度噪声比 hybrid 高数个数量级；importance-resampled VLB 才获得最好的 NLL。右侧表格同时暴露了 NLL 与 FID 的冲突。*

VLB 是不同时间步损失的总和，而这些项的尺度相差巨大。若仍然均匀采样 $t$，mini-batch 有时抽到极大项、有时抽到极小项，形成高方差梯度。于是出现反直觉结果：**优化严格的似然代理，训练曲线和最终似然反而不如 hybrid**。

## 6. 核心贡献三：cosine noise schedule 更合理地分配信息破坏速度

### 6.1 线性 $\beta_t$ 的浪费

![[assets/Improved DDPM - 可学习方差与扩散规模化/fig02_cosine_schedule.png]]

*图 3：原文 Figure 3–5。在线性日程下，低分辨率图像在最后约四分之一前向过程已接近纯噪声；cosine 日程更慢、更均匀地销毁信息。*

Ho et al. 的线性 $\beta_t$ 在高分辨率图像上可用，但对 $32\times32$、$64\times64$ 图像破坏过快。论文发现，线性日程模型跳过最多约 20% 的反向前缀时 FID 变化不大，说明对应训练步骤几乎没有提供有效学习信号。

### 6.2 cosine schedule 的定义

作者直接设计累计信号 $\bar\alpha_t$：

$$\bar\alpha_t=\frac{f(t)}{f(0)},\quad f(t)=\cos^2\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)$$

再反推出每步噪声：

$$\beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}$$

实现细节：

- 取 $s=0.008$，使初始标准差 $\sqrt{\beta_0}$ 略小于 8-bit 图像的像素 bin 宽度 $1/127.5$，避免起始噪声小到难以预测。
- 将 $\beta_t$ 截断到不超过 `0.999`，避免 $t\approx T$ 时出现数值奇异。
- 中段近似线性降低信号，两端变化较缓，避免突然改变 SNR。

### 6.3 不应过度神化 cosine

作者明确说选择 $\cos^2$ 带有任意性，许多相似形状可能同样有效。附录还显示：

- 对 CIFAR-10，cosine 优化更快，但也更早过拟合，需要把 dropout 从线性日程常用的 `0.1` 提高到 `0.3`。
- 对 ImageNet $256\times256$，论文反而使用 linear schedule，因为实验中它更好。

所以真正可迁移的原则是：**按数据分辨率与 SNR 轨迹设计信息销毁速度**，而不是“所有任务都必须 cosine”。

## 7. 核心贡献四：用 importance sampling 降低 VLB 梯度噪声

### 7.1 二阶矩重采样

为了直接优化 VLB，论文不再均匀抽取时间步，而令：

$$L_{\mathrm{vlb}}=\mathbb E_{t\sim p_t}\left[\frac{L_t}{p_t}\right],\quad p_t\propto\sqrt{\mathbb E[L_t^2]},\quad \sum_t p_t=1$$

除以 $p_t$ 保持估计无偏；按损失二阶矩的平方根采样，则把更多预算分给方差更大的时间步，降低 Monte Carlo 估计的总体方差。

### 7.2 在线估计细节

- 为每个时间步维护最近 **10 个** $L_t$ 值。
- 训练初期尚无历史时，先均匀采样。
- 直到每个 $t\in[0,T-1]$ 至少积累 10 个样本，再启用动态重采样。
- 该方法对高噪声的 `L_vlb` 明显有效，但对本来更稳定的 `L_hybrid` 没有帮助。

这项技巧的重要性不只在 diffusion：当总目标是许多异质子损失之和时，**抽哪个子任务本身就是优化算法的一部分**。

## 8. 少步采样：为什么训练 4000 步，推理可以只走 50–100 步

### 8.1 时间步 respacing

设训练时间步为 $1,\ldots,T$，采样时只选择子序列 $S=(S_1,\ldots,S_K)$。在保留训练累计噪声水平 $\bar\alpha_{S_t}$ 的前提下，重新计算短链方差：

$$\beta_{S_t}=1-\frac{\bar\alpha_{S_t}}{\bar\alpha_{S_{t-1}}},\quad \tilde\beta_{S_t}=\frac{1-\bar\alpha_{S_{t-1}}}{1-\bar\alpha_{S_t}}\beta_{S_t}$$

论文用 $K$ 个均匀分布在 $[1,T]$ 的实数并四舍五入到整数，得到采样子序列。learned variance 的 $v$ 再被解释为新短链上下界之间的插值位置，无需微调模型。

### 8.2 实验结果

![[assets/Improved DDPM - 可学习方差与扩散规模化/fig04_fewer_sampling_steps.png]]

*图 4：原文 Figure 8。固定方差的 `L_simple` 模型在缩短采样链后明显退化；learned-sigma 的 `L_hybrid` 模型在约 100 步已接近最佳 FID。*

所有模型训练时都用 4000 步，评测 25、50、100、200、400、1000、4000 个采样步：

- fully-trained `L_hybrid` 模型约 **100 步**即可得到接近最优 FID。
- 摘要所说的 **50 步**强调“仍能得到好样本”，不应误读为所有数据集、所有 checkpoint 都已完全饱和。
- DDIM 在少于 50 步时更好；达到 50 步及以上时，本文 learned-variance ancestral sampler 更好。
- 减少 FID 所需步数不等于能同样准确计算 NLL。附录 E 表明，均匀跳步会严重破坏 NLL；计算似然时还需额外保留 $1$ 到 $T/K$ 的早期密集时间步。

> [!important] 关键判断
> 本文的加速不是蒸馏，也不是重新训练一个短链模型；它是对同一模型做时间步重参数化。其成功高度依赖 learned variance，说明反向方差不仅影响似然，也是离散求解器跨大步长时的误差控制量。

## 9. 实验设置与架构细节

### 9.1 数据与指标

- **数据集**：CIFAR-10、unconditional ImageNet $64\times64$、class-conditional ImageNet $64\times64$，以及 LSUN bedroom 和 ImageNet $256\times256$ 附录实验。
- **指标**：NLL（bits/dim）、FID、Improved Precision/Recall。
- **FID 样本量**：通常 50K；但 unconditional ImageNet-64 的大规模消融只采 10K，因此数值存在向上偏差。class-conditional GAN 对比使用 50K。
- **优化**：Adam，典型 batch size 128、学习率 $10^{-4}$、EMA 0.9999。

### 9.2 U-Net 的小幅架构改造

相对 Ho et al. 的 DDPM：

1. attention 从单头改为 4-head，但总通道数不变；
2. 除 $16\times16$ 外，也在 $8\times8$ 分辨率使用 attention；
3. 时间条件从 `GroupNorm(h + v)` 改为 scale-shift 形式 `GroupNorm(h) * (w + 1) + b`，即接近 FiLM/AdaGN；
4. ImageNet-64 主消融模型 $C=128$，约 120M 参数，每次前向约 39 GFLOPs；
5. 作者的初步实验发现，无下采样的 ResNet 风格架构 NLL 更好但 FID 更差，最终仍选择 U-Net。

这些结构改动不是论文标题中的主角，却提醒我们：表中“算法改进”并非在完全不变的 Ho et al. 网络上得到，复现比较时必须对齐代码配置。

## 10. 消融实验：每一项改进到底贡献了什么

### 10.1 ImageNet $64\times64$

| 迭代 | $T$ | 日程 | 目标 | NLL ↓ | FID ↓ |
|---:|---:|---|---|---:|---:|
| 200K | 1K | linear | $L_{simple}$ | 3.99 | 32.5 |
| 200K | 4K | linear | $L_{simple}$ | 3.77 | 31.3 |
| 200K | 4K | linear | $L_{hybrid}$ | 3.66 | 32.2 |
| 200K | 4K | cosine | $L_{simple}$ | 3.68 | **27.0** |
| 200K | 4K | cosine | $L_{hybrid}$ | 3.62 | 28.0 |
| 200K | 4K | cosine | $L_{vlb}$ + resampling | **3.57** | 56.7 |
| 1.5M | 4K | cosine | $L_{hybrid}$ | 3.57 | **19.2** |
| 1.5M | 4K | cosine | $L_{vlb}$ + resampling | **3.53** | 40.1 |

逐步解读：

- 把 $T$ 从 1K 提到 4K：NLL `3.99 → 3.77`，FID 也略好，说明更细的离散化有利于 likelihood。
- 加 learned variance + hybrid：在线性日程下 NLL `3.77 → 3.66`，但 FID 基本不变甚至略差，符合“方差主要改善概率校准”的判断。
- linear 改 cosine：`L_simple` 的 FID `31.3 → 27.0`，说明日程直接影响有效训练难度。
- 纯 VLB：得到最好 NLL，却把 FID 拉到 `56.7/40.1`，清楚证明“似然最优”不是“感知最优”。
- 延长训练：hybrid FID `28.0 → 19.2`，NLL `3.62 → 3.57`，支持可扩展性主张，但也需要结合后文过拟合现象理解。

### 10.2 CIFAR-10

| 迭代 | $T$ | 日程 | 目标 | NLL ↓ | FID ↓ |
|---:|---:|---|---|---:|---:|
| 500K | 1K | linear | $L_{simple}$ | 3.73 | 3.29 |
| 500K | 4K | linear | $L_{simple}$ | 3.37 | **2.90** |
| 500K | 4K | linear | $L_{hybrid}$ | 3.26 | 3.07 |
| 500K | 4K | cosine | $L_{simple}$ | 3.26 | 3.05 |
| 500K | 4K | cosine | $L_{hybrid}$ | 3.17 | 3.19 |
| 500K | 4K | cosine | $L_{vlb}$ + resampling | **2.94** | 11.47 |

CIFAR-10 上同样是纯 VLB 的 NLL 最好、FID 最差。cosine 并没有在最终 500K checkpoint 上刷新最好 FID，原因之一是它更早过拟合；这再次说明消融必须同时看训练轨迹，而不能只看固定迭代数的终点。

### 10.3 hybrid 与 VLB 在时间步上的互补

![[assets/Improved DDPM - 可学习方差与扩散规模化/fig07_objective_tradeoff.png]]

*图 5：原文 Figure 13–14。`L_vlb` 模型在扩散链开头和结尾的 VLB 项更好，`L_hybrid` 在中段更好；纯 VLB 更关注难以感知的边界细节。*

附录将两种模型拼成 ensemble：中段 $t\in[100,T-100)$ 用 $\theta_{hybrid}$，两端用 $\theta_{vlb}$，得到 ImageNet-64 **FID 19.9、NLL 3.52**。它比 hybrid 的 NLL 更好，又几乎保住 FID。这说明目标冲突不是“两个模型整体谁更强”，而是**不同时间段承担不同感知语义**。

## 11. 与 GAN 的 precision/recall 对比

![[assets/Improved DDPM - 可学习方差与扩散规模化/fig05_precision_recall_and_samples.png]]

*图 6：原文 Table 4 与 Figure 9。大模型 Improved Diffusion 在 FID 上优于 BigGAN-deep，并在相近 precision 下显著提高 recall。*

实验使用 class-conditional ImageNet $64\times64$、Inception-V3 特征、$K=5$、50K 生成样本。BigGAN-deep 不做 truncation，以免人为压低 recall。

| 模型 | 参数/训练 | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---:|---:|---:|
| BigGAN-deep | 约 100M，125K iter | 4.06 | **0.86** | 0.59 |
| Improved Diffusion (small) | 100M，1.7M iter | 6.92 | 0.77 | **0.72** |
| Improved Diffusion (large) | 270M，250K iter | **2.92** | 0.82 | 0.71 |

解读：

- BigGAN 的 precision 最高，说明单张样本更集中在高密度、典型模式附近。
- 两个 diffusion 模型 recall 都约 `0.71–0.72`，明显高于 BigGAN 的 `0.59`，支持“扩散模型更少 mode dropping”。
- small diffusion 的 FID 不如 BigGAN，但 recall 已更高；large diffusion 同时获得更低 FID 和更高 recall。
- 这不是严格同算力比较：迭代数、网络结构和训练机制不同。论文证明的是分布覆盖优势，而非给出完全受控的 compute-normalized GAN 排名。

## 12. Scaling 实验：模型和算力是否可预测地换来性能

![[assets/Improved DDPM - 可学习方差与扩散规模化/fig06_scaling.png]]

*图 7：原文 Figure 10。ImageNet-64 上，FID 在 log-log 图中近似沿幂律下降；NLL 也改善，但拟合不如 FID 干净。*

### 12.1 设置

作者用 `L_hybrid` 训练四种 U-Net 宽度：

| 首层通道 $C$ | 参数量 |
|---:|---:|
| 64 | 30M |
| 96 | 68M |
| 128 | 120M |
| 192 | 270M |

学习率按 $1/\sqrt{\text{channel multiplier}}$ 缩放，使 128-channel 模型使用 $10^{-4}$。横轴是理论训练 FLOPs，并假设硬件满利用率。

### 12.2 结果与边界

- FID 的经验拟合约为 `4.00 + (2.500e-25 * C)^(-0.22)`，在图示区间内呈近似幂律。
- NLL 拟合约为 `3.40 + (3.000e-15 * C)^(-0.17)`，偏离更明显。
- NLL scaling 较差可能来自较高 irreducible loss、过拟合，或因为模型用的是兼顾 FID 的 `L_hybrid`，而非纯 `L_vlb`。

这里的 `C` 表示训练 compute（FLOPs），不是通道数。两条拟合线是**特定数据、架构、算力区间的经验描述**，还不足以宣布普适 diffusion scaling law。论文真正有力的结论是：模型容量与训练时长增加时，性能没有出现明显不稳定或突然饱和，扩散训练具备良好的工程可扩展性。

## 13. 高分辨率与级联实验

在 class-conditional ImageNet $256\times256$ 上，论文比较直接生成与 `64→256` 上采样级联：

| 模型 | FID ↓ |
|---|---:|
| VQ-VAE-2（two-stage） | 38.1 |
| Improved Diffusion（single-stage） | 31.5 |
| Improved Diffusion（two-stage） | **12.3** |
| BigGAN | 7.7 |
| BigGAN-deep | 7.0 |

结论应分两层理解：

1. 在 likelihood-based 模型中，Improved Diffusion 明显缩小了与 GAN 的差距。
2. 但本文在 $256\times256$ 上仍未超过 BigGAN；真正把 diffusion 推到更高保真度的是后续 ADM 工作中的更强架构与 classifier guidance。

值得注意的是，这组高分辨率实验使用 linear schedule，再次说明 cosine 不是跨分辨率无条件占优。

## 14. 过拟合、早停与“训练越久越好”的限定

![[assets/Improved DDPM - 可学习方差与扩散规模化/fig08_overfitting.png]]

*图 8：原文 Figure 16。cosine 日程更快达到最佳 FID，但之后 FID 回升；NLL 仍可能继续下降，显示似然与感知质量再次分叉。*

- CIFAR-10 上所有模型都会过拟合；cosine 因噪声更少或优化更容易，过拟合更早。
- class-conditional ImageNet-64 即使数据更大，也观察到后期 FID 变差。
- EMA 0.9999/0.99995 最好；增加少量 dropout 可改善最佳 FID但不能消除最终过拟合。
- 作者的实践建议是：**按 FID early stopping；如果还有算力，优先增大模型，而不是无限延长同一模型训练。**

这也修正了“scaling 曲线意味着训练越久必然越好”的表面理解：规模化成立的前提是容量、正则化、早停与训练预算共同匹配。

## 15. 局限性与批判性思考

### 15.1 指标间冲突没有被真正统一

`L_hybrid` 是有效折中，但 $\lambda=0.001$ 属于经验权重；论文没有给出从多目标优化角度自动选择权重的方法。NLL 与 FID 的错位被清楚展示，却没有理论解释“哪些时间步对应语义、哪些对应不可感知细节”。附录 ensemble 暗示按时间步自适应组合目标可能更合理。

### 15.2 加速仍不是实时生成

从 4000 步降到 50–100 步是数量级改进，但每步仍需完整 U-Net 前向。相比单次前向 GAN，当时的 diffusion 仍慢；论文也没有讨论端到端吞吐、显存、批量大小和实际硬件延迟。

### 15.3 少步 FID 与少步 likelihood 不是同一件事

均匀 respacing 能保住 FID，却会显著破坏 NLL；附录需要在数据端额外保留密集步。说明短链采样在感知上可接受，不代表它仍精确实现原概率模型的 likelihood 计算。

### 15.4 cosine schedule 的适用性有限

它在低分辨率数据上有效，但高分辨率 ImageNet-256 采用 linear；在 CIFAR-10 上还更易过拟合。论文没有把 schedule 与分辨率、频谱或 SNR 的关系系统化。

### 15.5 learned variance 的外插缺少约束分析

$v$ 未限制在 $[0,1]$，作者只报告实践中没有明显越界。若换数据、混合精度或更激进的短链，方差外插是否导致不稳定，文中没有压力测试。

### 15.6 scaling 证据仍属初步

只有四种宽度、一个主要数据集和有限算力区间；横轴是理论 FLOPs并假设满利用率。拟合常数还借助 in-distribution FID 与当时 SOTA NLL 设定，不宜外推到不同架构、分辨率或现代 solver。

### 15.7 GAN 对比不是严格等算力

模型参数、训练迭代、目标函数和计算图不同；precision/recall 的结论支持“更好的 mode coverage”，但不能单独证明 diffusion 在相同资源下全面优于 GAN。

### 15.8 FID 样本数不完全一致

unconditional ImageNet-64 消融多用 10K 样本，class-conditional 对比用 50K；作者指出 50K 会让估计 FID 约低 2 点。因此跨表直接比较 FID 会产生偏差。

## 16. 与 DDPM、ADM 的关系

### 16.1 技术谱系

| 维度 | DDPM（Ho et al., 2020） | Improved DDPM / IDDPM（本文） | ADM（Dhariwal & Nichol, 2021） |
|---|---|---|---|
| 核心定位 | 建立高质量扩散图像生成基线 | 改善 likelihood、采样速度、覆盖与 scaling | 通过架构升级与 classifier guidance 追求更高保真度 |
| 均值参数化 | 预测 $\epsilon$ | 延续预测 $\epsilon$ | 延续扩散/score 参数化体系 |
| 反向方差 | 固定 $\beta_t$ 或 $\tilde\beta_t$ | log-space learned interpolation | 继承 improved/guided diffusion 代码谱系中的 learned-sigma 能力 |
| 训练目标 | 主要 $L_{simple}$ | $L_{hybrid}$ 或 resampled $L_{vlb}$ | 在改进 backbone 上训练，并将 classifier gradient 用于条件采样 |
| 噪声日程 | 线性 | 低分辨率常用 cosine；高分辨率仍可 linear | 按任务配置，不应简化为单一 schedule |
| 采样 | 长链 ancestral | respacing 后约 50–100 步仍可用 | guidance + diffusion sampler；论文报告少至 25 次前向仍可匹配 BigGAN-deep |
| fidelity/diversity 控制 | 无显式 guidance knob | 主要展示高 recall | classifier guidance scale 显式交换 diversity 与 fidelity |

### 16.2 三者不是平行模型名，而是递进路线

1. **DDPM** 给出稳定的噪声预测训练范式，证明 diffusion 可以生成高质量图像。
2. **Improved DDPM** 解决“作为概率模型是否够好、能否少步采样、能否规模化”的问题，是从概念验证走向可扩展系统的桥梁。
3. **ADM** 通常指后续 *Diffusion Models Beat GANs on Image Synthesis* 的架构/模型谱系。它不是本文某个 loss 的别名，而是在本文工程基础上继续增强 U-Net，并用 classifier guidance 将条件分类器梯度加入采样，主动把分布推向高密度类别区域。

因此可用一句话记忆：**DDPM 建立范式，Improved DDPM 校准概率与计算，ADM 强化架构并引入 guidance。**

## 17. 复现与工程检查清单

### 17.1 论文主配置

- ImageNet-64：U-Net，`num_channels=128`，`num_res_blocks=3`，约 120M 参数。
- `diffusion_steps=4000`。
- learned sigma：开启。
- 低分辨率主实验：cosine schedule。
- hybrid：噪声 MSE + `0.001 * VLB`，VLB 对均值分支 stop-gradient。
- 纯 VLB：`use_kl=True`，时间步 sampler 使用 loss second moment。
- Adam，batch 128，lr $10^{-4}$，EMA 0.9999。
- CIFAR-10 cosine：dropout 0.3；linear 的最佳 dropout 为 0.1。
- 少步采样：使用 timestep respacing；论文与仓库都区分本文 stride 和 DDIM stride。

### 17.2 最容易复现错的点

1. 把 `learn_sigma=True` 打开，却仍只训练 `L_simple`，导致方差头没有正确 VLB 信号。
2. 在 hybrid 的 VLB 项里忘记 stop-gradient 均值分支。
3. 用 cosine 公式时漏掉 $s=0.008$ 或 `beta <= 0.999` 截断。
4. importance sampling 没有除以 $p_t$，从而把无偏估计变成改变目标本身。
5. 一开始就启用二阶矩采样，而不是先为每个时间步收集 10 个历史值。
6. 将训练 4000 步与采样 100 步混为“训练只需 100 步”。
7. 用 10K 与 50K 样本算出的 FID 横向比较却不标注样本数。
8. 在高分辨率任务上机械套用 cosine，而不重新验证 SNR 轨迹和过拟合。

## 18. 我的核心理解与后续启发

### 18.1 learned variance 是“离散化适配器”

表面上它服务 likelihood，深层作用却是让反向转移能适应不同步长。后续很多快速 sampler 转向 ODE/SDE solver，但本文已经说明：**采样器跨多大步，与模型对局部不确定性的表示能力密切相关。**

### 18.2 时间步不是同质训练样本

Figure 2 与 importance sampling 表明，不同 $t$ 的损失尺度、语义和优化难度完全不同。后续可自然延伸到：

- 按 SNR / loss / gradient norm 做时间步加权；
- 为不同时间区间采用不同参数化或专家网络；
- 把目标权重和时间步分布联合学习，而不是手工设定。

### 18.3 likelihood 与感知质量的冲突具有“时间结构”

`L_vlb` 在两端更强，`L_hybrid` 在中段更强；两者 ensemble 同时改善 NLL 与 FID。这比泛泛而谈“两个指标不一致”更具体：**冲突可以被定位到扩散轨迹的不同区间**。这为后来的 min-SNR weighting、分段 solver、不同 prediction target 等研究提供了直觉基础。

### 18.4 cosine schedule 的真正遗产是 SNR 视角

cosine 本身不是终点；它把 schedule 设计从“线性设置每步 $\beta$”提升到“设计累计信号和 SNR 如何随时间变化”。现代 diffusion/flow 模型中，噪声参数化、loss weighting 与采样时间分布仍沿用这一视角。

## 19. 关键术语

- **Learned variance**：由网络学习反向高斯分布方差，而非固定为前向噪声或真实后验方差。
- **Log-space interpolation**：在 $\log\beta_t$ 与 $\log\tilde\beta_t$ 间线性插值，再指数映射回正方差。
- **Hybrid objective**：以 `L_simple` 训练均值，以小权重 VLB 训练方差的复合目标。
- **VLB**：variational lower bound；本文以负 VLB 作为可优化的 NLL 上界。
- **Noise schedule**：规定累计信号/噪声如何随 $t$ 变化的日程。
- **Importance sampling over timesteps**：按损失二阶矩动态重采样时间步，以降低 VLB 梯度方差。
- **Timestep respacing**：推理时从训练时间网格抽取更稀疏的子序列，并重算短链转移参数。
- **Precision / Recall for generative models**：分别近似衡量生成样本保真度与真实分布覆盖率。
- **ADM**：后续 *Diffusion Models Beat GANs on Image Synthesis* 中的扩散架构/模型谱系，核心增量包括架构消融与 classifier guidance。

## 20. 图表索引

1. `fig01_variance_and_vlb_terms.png`：方差上下界与 VLB 时间步贡献，支撑 learned variance 动机。
2. `fig02_cosine_schedule.png`：linear/cosine 的加噪速度与累计信号差异。
3. `fig03_objectives_and_ablation.png`：目标曲线、梯度噪声、ImageNet/CIFAR 消融。
4. `fig04_fewer_sampling_steps.png`：采样步数与 FID，展示 learned variance 的短链鲁棒性。
5. `fig05_precision_recall_and_samples.png`：BigGAN 对比、precision/recall 与 class-conditional 样本。
6. `fig06_scaling.png`：不同模型规模下 FID/NLL 随 compute 的变化。
7. `fig07_objective_tradeoff.png`：hybrid 与 VLB 在扩散时间上的互补。
8. `fig08_overfitting.png`：cosine schedule 的优化速度与过拟合。

## 21. 最终评价

这篇论文的重要性不在于提出一个耀眼的新生成范式，而在于把 DDPM 从“样本质量惊艳但慢、似然一般”的研究原型，推进成一个**似然可竞争、短链可部署、分布覆盖更完整、能稳定吃模型与算力**的生成系统。其最持久的技术遗产是三点：

1. 用有界、可解释的参数化学习反向方差；
2. 把时间步分布视为优化设计变量；
3. 用累计信号/SNR 而不是孤立的 $\beta_t$ 思考噪声日程。

与此同时，论文也诚实暴露了 diffusion 的核心张力：**更好的 likelihood、FID、速度和覆盖不是同一个目标。** Improved DDPM 的贡献，是给出了一套简单、可复现的折中，使后续 ADM、guidance 和更先进采样器有了可靠起点。