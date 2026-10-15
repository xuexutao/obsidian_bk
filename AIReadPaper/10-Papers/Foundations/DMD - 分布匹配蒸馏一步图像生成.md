---
type: paper-note
status: done
domain: Foundations
paper: One-step Diffusion with Distribution Matching Distillation
year: 2024
arxiv: 2311.18828
doi: null
source: https://arxiv.org/abs/2311.18828
project: https://tianweiy.github.io/dmd/
code: https://github.com/tianweiy/DMD
tags:
  - Foundations
  - Distribution-Matching
  - Distillation
  - DMD
  - Text-to-Image
created: 2026-08-29
updated: 2026-10-15
---

# DMD：分布匹配蒸馏一步图像生成

## 基本信息

- **完整标题**：One-step Diffusion with Distribution Matching Distillation
- **作者**：Tianwei Yin¹、Michaël Gharbi²、Richard Zhang²、Eli Shechtman²、Frédo Durand¹、William T. Freeman¹、Taesung Park²（¹MIT、²Adobe Research）
- **发表**：CVPR 2024
- **arXiv**：2311.18828v4 [cs.CV]，2024 年 10 月 4 日最新修订
- **项目页**：https://tianweiy.github.io/dmd/
- **代码仓库**：https://github.com/tianweiy/DMD
- **历史地位**：这条因果视频蒸馏路线的**源头之一**。第一作者 Tianwei Yin 后来把 DMD 的"分布匹配蒸馏"思想迁移到视频，做出 CausVid（双向教师→因果自回归学生），因此 DMD 是理解整条路线不可绕过的基础模块。

## 一句话结论

DMD 把"多步迭代采样的扩散模型"蒸馏成"单次前向的一步生成器"，不追求复现教师的具体采样轨迹，而是**在分布层面**让学生的输出与教师不可区分——具体做法是最小化一个近似 KL 散度，其梯度恰好等于「真实分布分数函数」与「生成分布分数函数」之差；再辅以一个 LPIPS 回归损失保证大规模结构对齐，在 ImageNet 64×64 上达到 FID 2.62、COCO-30k 上 11.49，且用 FP16 达到 20 FPS。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

扩散模型（DDPM、Score-SDE、LDM、DiT 等）已成为图像生成的主流范式，采样质量高、训练稳定、不易模式坍缩，但采样是一个迭代去噪过程——把一份高斯噪声经过数十到数百次神经网络前向，逐步精炼成一张图。这导致单张 512×512 图像的生成常需秒级延迟，无法支持实时交互式创作。

DMD 要回答的问题很直接：能否把已经训练好的扩散模型，压缩成一个"一次前向"的一步生成器，同时尽量不损失图像质量？

### 1.2 现有加速方法的瓶颈

已有的加速路线大体分两类：

1. 快速采样器（few-step sampler）：如 DPM-Solver、UniPC、PNDM 等，把千步采样压缩到 20–50 步。但步数再往下压，质量会灾难性下降——一步到位基本不可行。
2. 轨迹蒸馏（trajectory distillation）：如渐进蒸馏（Progressive Distillation）、InstaFlow、一致性模型（Consistency Model）、TRACT 等，训练学生去拟合教师"噪声→图像"的逐点对应映射。

第二类方法的痛点是：拟合"噪声→图像"这一高维复杂映射本身很难，而且为了算一次学生损失，往往需要跑完教师整条去噪轨迹（progressive distillation 用"减半步数"来缓解）。更关键的是，作者指出这类方法把学生过度绑定在教师的具体采样路径上，学生的质量上限被教师路径限制住。

### 1.3 本文核心思路

DMD 换了一个立场：不要学生复现教师的具体路径，只要学生生成的图像和教师"看起来是同一个分布"。这与 GAN、GMMN 这类"分布匹配"生成模型的动机一致。但 GAN 在大规模文生图上训练难稳定，DMD 的做法是"站在巨人的肩膀上"——从已经在大规模数据上训练好的扩散模型出发，用两个扩散模型分别去估计"真实分布"和"生成分布"的分数，用分数之差作为梯度方向来更新生成器。

![](assets/DMD%20-%20分布匹配蒸馏一步图像生成/figures/01-which-is-which.png)

> 图 1：DMD 的 teaser。"哪个是哪个？"——图中一部分图像由 Stable Diffusion 以 2590ms/张生成，另一部分由 DMD 一步生成器以 90ms/张生成，肉眼难以区分。这直接点出核心主张：一步生成器可以在质量上与昂贵的多步扩散模型比肩，速度却快两个数量级。
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2311.18828

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- 输入：随机高斯噪声 $z \sim \mathcal{N}(0, I)$，以及（可选）文本提示 / 类别标签等条件。
- 输出：一张真实感图像 $x = G_\theta(z)$。
- 假设：给定一个已训练好的扩散模型（教师，记为 $\mu_{\text{base}}$），其架构与权重是固定的；DMD 只在其基础上微调出一个"去掉时间条件、单步出图"的生成器 $G_\theta$。
- 关键立场：$G_\theta$ 的输出称为 fake（假图），训练数据的图像称为 real（真图），沿用 GAN 的命名习惯——尽管 $G_\theta$ 的最终目标是让 fake 与 real 不可区分。

### 2.2 关键符号与目标函数

- 真实数据分布 $p_{\text{real}}$，生成器分布 $p_{\text{fake}}$。
- 扩散前向过程：$q_t(x_t | x) \sim \mathcal{N}(\alpha_t x, \sigma_t^2 I)$，$\alpha_t, \sigma_t$ 由噪声调度决定。
- 分数函数：$s_{\text{real}}(x_t, t) = \nabla_{x_t}\log p_{\text{real},t}(x_t)$，$s_{\text{fake}}(x_t, t) = \nabla_{x_t}\log p_{\text{fake},t}(x_t)$。

核心优化目标是在扩散各时间步上最小化 fake 与 real 分布的近似 KL 散度（详见 3.2 节式 (1)(2)），并加上一个回归损失（式 (9)）。

## 3. 核心方法

### 3.1 总体框架

![](assets/DMD%20-%20分布匹配蒸馏一步图像生成/figures/02-method-overview.png)

> 图 2：方法总览。一步生成器 $G_\theta$ 把噪声 $z$ 映射成图像；为了匹配多步采样输出，预先计算一批"噪声-图像"配对数据集并间歇加载，对生成器施加 LPIPS 回归损失；同时把 fake 图像注入随机噪声后分别送入"真实分数扩散模型"和"动态训练的 fake 分数扩散模型"，两者的去噪输出之差就是"更真实、更少假"的分布匹配梯度，反传回生成器。
> 来源：论文 Figure 2，第 3 页，https://arxiv.org/abs/2311.18828

整体上 DMD 维护三个网络：

1. **一步生成器 $G_\theta$**：架构与教师去噪器相同但**去掉时间条件**，初始化为 $\theta \leftarrow \mu_{\text{base}}$。
2. **真实分数模型 $\mu_{\text{base}}$（冻结）**：就是预训练的教师，建模 $p_{\text{real}}$ 的分数。
3. **fake 分数模型 $\mu^{\text{fake}}_{\phi}$（动态）**：从 $\mu_{\text{base}}$ 初始化，训练中持续更新，追踪生成器输出分布 $p_{\text{fake}}$ 的分数。

### 3.2 分布匹配损失（核心创新）

理想目标是让 $p_{\text{fake}}$ 与 $p_{\text{real}}$ 不可区分，受 ProlificDreamer（VSD）启发，最小化 KL 散度：

$$D_{KL}(p_{\text{fake}} \parallel p_{\text{real}}) = \mathbb{E}_{z\sim\mathcal{N}(0,I),\ x=G_\theta(z)}\big[-\big(\log p_{\text{real}}(x) - \log p_{\text{fake}}(x)\big)\big] \tag{1}$$

直接算这个概率密度不可行，但训练只需要对 $\theta$ 的梯度：

$$\nabla_\theta D_{KL} = \mathbb{E}_{z,\ x=G_\theta(z)}\Big[-\big(s_{\text{real}}(x) - s_{\text{fake}}(x)\big)\frac{dG}{d\theta}\Big] \tag{2}$$

直觉上，$s_{\text{real}}$ 把 $x$ 推向真实分布的模态，$-s_{\text{fake}}$ 把样本推开、防止坍缩（见 Figure 3(a,b)）。

但式 (2) 有两个障碍：① 对低概率样本分数发散（尤其 fake 样本处 $p_{\text{real}}$ 趋零）；② 扩散模型只给出**扩散后分布**的分数。Score-SDE 的理论给出解答——**给数据注入不同方差的高斯噪声**，得到一族支撑在全空间、彼此重叠的"模糊分布"，使梯度处处有定义（Figure 4）；训练好的扩散模型恰好逼近这个扩散分布的分数。

因此用两个扩散去噪器分别建模扩散后 real/fake 分布的分数：

$$s_{\text{real}}(x_t, t) = -\frac{x_t - \alpha_t \mu_{\text{base}}(x_t, t)}{\sigma_t^2} \tag{4}$$

$$s_{\text{fake}}(x_t, t) = -\frac{x_t - \alpha_t \mu^{\text{fake}}_{\phi}(x_t, t)}{\sigma_t^2} \tag{5}$$

其中 fake 分数模型 $\mu^{\text{fake}}_{\phi}$ 用标准去噪目标在生成样本上**在线**更新：

$$L_{\phi}^{\text{denoise}} = \big\|\mu^{\text{fake}}_{\phi}(x_t, t) - x_0\big\|_2^2 \tag{6}$$

最终的近似分布匹配梯度为：

$$\nabla_\theta D_{KL} \simeq \mathbb{E}_{z,t,x,x_t}\Big[w_t \alpha_t \big(s_{\text{fake}}(x_t, t) - s_{\text{real}}(x_t, t)\big)\frac{dG}{d\theta}\Big] \tag{7}$$

权重 $w_t$ 用于**归一化不同噪声级别下的梯度幅度**：

$$w_t = \frac{\sigma_t^2}{\alpha_t}\frac{C\,S}{\big\|\mu_{\text{base}}(x_t, t) - x\big\|_1} \tag{8}$$

其中 $S$ 是空间位置数、$C$ 是通道数。实验表明该加权优于 DreamFusion / ProlificDreamer 的 $\sigma_t/\alpha_t$、$\sigma_t^3/\alpha_t$ 方案（Table 2 右，FID 提升约 0.9）。

### 3.3 回归损失与最终目标

分布匹配损失在 $t$ 较大（噪声很多）时定义良好，但在噪声很小时 $s_{\text{real}}(x_t,t)$ 不可靠（$p_{\text{real}}(x_t,t)\to 0$）；且分数对概率密度缩放不变，优化易受**模式坍缩 / 模式丢失**影响。为此引入回归损失，确保所有模态都被保留：

$$L_{\text{reg}} = \mathbb{E}_{(z,y)\sim\mathcal{D}}\ \ell\big(G_\theta(z), y\big) \tag{9}$$

配对数据集 $\mathcal{D}=\{z,y\}$ 由教师用确定性 ODE 采样器（CIFAR-10 用 Heun 18 步、ImageNet 用 Heun 256 步、LAION 用 PNDM 50 步）离线生成；距离 $\ell$ 用 LPIPS（也可用 L2，见附录）。最终目标：

$$\mathcal{L} = D_{KL} + \lambda_{\text{reg}} L_{\text{reg}},\quad \lambda_{\text{reg}}=0.25$$

### 3.4 分类器无关引导（CFG）的适配

对文生图（CFG 引导）的教师，DMD 的适配很干净：回归配对数据由**引导后**的模型采样得到；分布匹配梯度中的真实分数替换为引导模型的均值预测，fake 分数公式不变；生成器以固定引导尺度训练。

## 4. 数据集与实验设置

### 4.1 数据集

- **CIFAR-10 32×32**（cond/uncond）：教师为 EDM（`edm-cifar10-32x32-cond-vp` / `-uncond-vp`）。
- **ImageNet 64×64**（class-cond）：教师为 EDM（`edm-imagenet-64x64-cond-adm`）。
- **MS-COCO 2014**（zero-shot 文生图）：教师为 Stable Diffusion v1.5，训练数据 LAION-Aesthetic-6.25+（约 300 万张）。

### 4.2 评价指标

- **FID**：CIFAR/ImageNet 各生成 50k 张，用 EDM 评测代码；COCO 用 clean-fid（512→256 Lanczos 下采样，与 40,504 张真实图对比）。
- **CLIP Score**：OpenCLIP-G 骨干。

### 4.3 实现细节

- 优化器 AdamW，CIFAR lr=5e-5 / ImageNet lr=2e-6 / LAION lr=1e-5，warmup 500 步，梯度裁剪 L2=10，关闭 dropout。
- 分布匹配损失与 fake 去噪损失权重均为 1；回归损失 $\lambda_{\text{reg}}$：CIFAR-cond 0.25、CIFAR-uncond 0.5、其余 0.25。
- 训练规模：ImageNet 用 7 GPU、batch 336、35 万迭代；LAION-6.25+ 用 72 A100、batch 2304、2 万迭代（约 36 小时）。

## 5. 实验结果

### 5.1 主要定量结果

**ImageNet 64×64（Table 1）**：DMD 一步 FID 2.62，比 Consistency Model（6.20）提升 2.4×，比 Diff-Instruct（5.57）、TRACT（7.43）等大幅领先，仅比教师 EDM（512 步，2.32）差 0.3——而推理成本降 512 倍。

**零样本 COCO（Table 3）**：DMD 一步 FID 11.49，超过 InstaFlow（13.10）、UFOGen（12.78）、LCM-LoRA（23.62），逼近教师 SD v1.5（8.78），速度快约 30×，FP16 下 20 FPS。

**CIFAR-10（Table 6）**：cond 模型 FID 2.66、uncond 3.77。

### 5.2 消融实验（Table 2 / Figure 5）

- **去掉分布匹配损失**：CIFAR FID 3.82 / ImageNet 9.21，图像缺乏真实感与结构完整性。
- **去掉回归损失**：CIFAR 5.58 / ImageNet 5.61，出现模式坍缩（灰色汽车反复出现），训练不稳。
- **采样加权策略**：DMD 的式 (8) 加权（2.66）优于 $\sigma_t/\alpha_t$（3.60）与 $\sigma_t^3/\alpha_t$（3.71）。

### 5.3 定性结果

![](assets/DMD%20-%20分布匹配蒸馏一步图像生成/figures/03-sd-qualitative.png)

> 图 3：从 Stable Diffusion 出发的文生图定性对比。同一提示词下，DMD 一步生成（90ms）在质量上明显优于 DPM++（4 步 260ms）、InstaFlow（1 步）、LCM（1/2 步），与 50 步 SD（2590ms）接近，而速度更快。
> 来源：论文 Figure 6，第 7 页，https://arxiv.org/abs/2311.18828

### 5.4 失败案例与边界

- 与 100/1000 步的更细离散化相比，一步模型仍有轻微质量差距。
- 训练需要同时微调 fake 分数模型与生成器，显存开销大（论文建议用 LoRA 缓解）。

## 6. 与相关工作的关系

- **相对 Progressive Distillation / 一致性模型**：这些方法拟合"噪声→图像"的逐点轨迹；DMD 只做分布匹配，不要求一一对应。
- **相对 VSD（ProlificDreamer）**：VSD 把扩散模型当分布匹配损失用于 3D 粒子优化；DMD 将其精化并扩展到"训练一个深度生成网络"，且补上回归损失做稳定化。
- **相对 GAN / ADD / LADD**：DMD 的公式并非建立在 GAN 上（不引入显式判别器），而是纯分数差。
- **相对 Diff-Instruct**：两者同为分数差框架，DMD 额外引入回归损失，取得更优的文本到图像结果。

## 7. 局限与批判性评价

1. **回归损失是双刃剑**：它保证了训练稳定和模态覆盖，但也让学生质量被教师采样路径上界限制，且在大规模文生图上构造百万级噪声-图像对代价高昂（这正是 DMD2 要解决的问题）。
2. **三网络显存开销**：冻结教师 + fake 分数模型 + 生成器，比传统扩散训练更吃显存。
3. **引导尺度固定**：训练时固定 CFG 尺度，推理时用户无法灵活调节。
4. **评价上的启示**：DMD 用 FID 逼近教师、甚至后续工作（DMD2）反超教师，说明"蒸馏学生一定不如教师"的直觉并不成立——迭代采样本身会累积误差。

## 8. 复现与实践建议

- 官方代码：https://github.com/tianweiy/DMD（另有社区实现 OpenDMD 被 DMD2 引用）。
- 关键超参：$\lambda_{\text{reg}}=0.25$（cond）/ 0.5（uncond）；$T_{\min}=0.02T,\ T_{\max}=0.98T$（沿用 DreamFusion）；梯度裁剪 L2=10。
- 显存紧张时优先对 $\mu^{\text{fake}}_{\phi}$ 与 $G_\theta$ 用 LoRA；LAION 解码回归损失用 Tiny VAE 降显存。
- 复现成本：ImageNet 约 2 天（7×A100）；LAION-6.25+ 约 36 小时（72×A100）。

## 9. 个人启发与后续问题

DMD 最重要的思想贡献是**把"蒸馏"从"模仿轨迹"重新定义为"匹配分布"**，并用"两个分数函数之差"给出了一个干净、可扩展的梯度方向。这条线后续的每一步演化都建立在这个范式上：

- **DMD2** 去掉回归损失（TTUR 稳定 fake critic）、引入 GAN、支持多步，反超教师。
- **SiD** 用 Fisher 散度（MESM）替代 KL，引入分数梯度，实现指数级 FID 收敛。
- **CausVid** 把这个分布匹配梯度搬进"双向教师→因果自回归学生"的非对称蒸馏，成为因果视频蒸馏路线的起点。

### 9.1 从图像一步蒸馏到视频世界模型：迁移边界

原始 DMD 优化的是静态条件 $c$ 下的单图分布 $p_\theta(x\mid c)$：一次采样没有历史状态，fake score 只需追踪当前生成器的图像边缘分布。到了视频世界模型，真正可迁移的是**对加噪学生样本使用 real/fake score difference 近似 reverse-KL 梯度**这一原则；不能原样搬用的是样本单位、条件分布、因果分解与历史构造。此时目标更接近

$$p_\theta(x_{1:N}\mid c)=\prod_{i=1}^{N}p_\theta(x_i\mid H_i,c),$$

而 $H_i$ 往往还是由学生 rollout、KV cache、空间检索或记忆重构动态产生的随机变量。三种后续用法改变的层次并不相同：

| 工作 | DMD 实际作用 | 条件分布 / 历史结构的变化 | 不能与原始 DMD 混同之处 |
|---|---|---|---|
| **Lyra 2.0：35→4 步 DMD** | 把完整模型的 35-step FlowUniPC + CFG 压成 4-step 学生，并把 CFG 合并进条件前向；蒸馏时保留 self-augmentation | 空间检索 slots、FramePack 时间上下文、相机/文本条件与自劣化历史都已由 Lyra 2.0 定义；DMD 主要压缩**每个 80 帧 chunk 内部的去噪步数**，没有改写 chunk 间自回归分解或历史检索策略 | 这是有持久记忆条件的 **4 步视频蒸馏**，不是“单噪声→单图”的一步 DMD；其长程一致性主要来自空间路由、FramePack 与 self-augmentation，而非 DMD 本身 |
| **Self-Forcing：video-level DMD** | 对学生按推理方式 self-rollout 得到的**完整视频联合分布**做 DMD，而非逐帧独立匹配 | 历史从 teacher forcing / Diffusion Forcing 的真值或加噪真值，改为学生自己的 $\hat{x}_{<i}$ 与 rolling KV cache；因此训练样本真正来自推理时的 $p_\theta$ | DMD 在这里只是 video-level divergence；决定性改动是 self-rollout 使条件分布 inference-aligned。若仍喂 GT 历史，即使损失叫 DMD，也没有解决 exposure bias |
| **WorldPlay：Context Forcing** | 用 DMD reverse-KL 把 100 NFE 的记忆感知教师压到 4 NFE 学生 | 每个 chunk 的条件 $C_i$ 由最近 3 个时间 chunks、按 FOV/距离检索的 1 个远程空间 chunk 及首 chunk attention sink 重构，并用 Temporal Reframing 改写远帧时间索引；教师/real-fake score 网络必须看到与学生**同构且对齐**的记忆上下文，同时 mask 当前待预测序列 | 它蒸馏的是 $p(x_i\mid C_i)$，不是无历史的 $p(x)$；Context Forcing 的核心是把“记忆怎样进入条件”纳入师生分布定义，错误地用无记忆教师或错位上下文会直接破坏蒸馏 |

因此，三者不能被概括成“把原始 DMD 直接用于视频”。更准确的分层是：**Lyra 2.0 改的是既定长视频系统中的 NFE；Self-Forcing 改的是训练时历史的来源，使匹配对象变成 student-induced rollout distribution；WorldPlay 改的是师生共享的记忆条件结构，使分布匹配对检索历史显式条件化。** 从图像 DMD 迁移到世界模型前，必须先回答“一个样本是帧、chunk 还是整段 rollout”“历史由谁生成”“教师与学生是否看到同一记忆”三个问题，再谈 score difference 如何计算。

值得追问的问题：分布匹配的"分数差"是否天然适合处理**非对称的师生结构**（学生是因果的、教师是双向的）？这一问句正是 CausVid 与 Self-Forcing 系列要回答的。

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | arXiv 2311.18828v4 | 全文 | 2026-08-29 | 精读主来源 |
| `figures/01-which-is-which.png` | Teaser | arXiv HTML 渲染 | Figure 1, p.1 | 2026-08-29 | 第 1 章"研究背景" |
| `figures/02-method-overview.png` | 方法图 | arXiv HTML 渲染 | Figure 2, p.3 | 2026-08-29 | 第 3.1 节"总体框架" |
| `figures/03-sd-qualitative.png` | 定性对比图 | arXiv HTML 渲染 | Figure 6, p.7 | 2026-08-29 | 第 5.3 节"定性结果" |
