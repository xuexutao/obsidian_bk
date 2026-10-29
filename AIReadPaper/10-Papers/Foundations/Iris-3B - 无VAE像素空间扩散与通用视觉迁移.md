---
type: paper-note
status: complete
domain: Foundations
paper: "Iris-3B: Going Beyond the Latent with Pixel-Space Diffusion Training, Conversion and Fine-Tuning"
year: 2026
arxiv: "2610.09450"
doi: null
source: https://arxiv.org/abs/2610.09450
project: https://speridlabs.com/research/iris
code: https://github.com/speridlabs/iris-3b
code_commit: 13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e
model: https://huggingface.co/speridlabs/iris-3b
license: Apache-2.0
tags:
  - Foundations
  - Pixel-Space-Diffusion
  - Flow-Matching
  - DiT
  - Generative-Prior
  - Dense-Prediction
created: 2026-10-29
updated: 2026-10-29
---

# Iris-3B：无 VAE 像素空间扩散与通用视觉迁移

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *Iris-3B: Going Beyond the Latent with Pixel-Space Diffusion Training, Conversion and Fine-Tuning* |
| 作者 | Hanqiu Li Cai、Chema Garabito（SperidLabs） |
| 时间 | arXiv v1：2026-10-07 |
| 模型 | 2,987,511,888 个可训练参数；另使用冻结的 Qwen3-VL-4B-Instruct 文本编码器 |
| 核心任务 | 像素空间文生图；潜空间模型到像素空间的转换；单目深度估计；图像恢复与 4× 超分辨率 |
| 官方代码 | 已完整开源基础模型预训练、采样和 Turbo 蒸馏代码；下游任务仅开源推理与权重，未包含深度/恢复训练器；FLUX.2 Klein 转换训练代码未开源 |
| 代码版本 | `13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e` |
| 权重 | 基础文生图约 12 GB；depth、upscaler 各约 12 GB；Turbo 约 6 GB |
| 许可证 | Apache-2.0 |

## 一句话结论

Iris-3B 证明了**不经 VAE、直接在 RGB 像素空间训练的 3B flow-matching Transformer 可以扩展到 1024²，并达到强潜空间模型附近的文生图质量**；但论文最重要、也最诚实的结论是否定性的：**去掉 VAE 并没有让生成先验在深度估计和 4× 图像恢复上稳定优于潜空间模型**，反而暴露出 16×16 非重叠 patch 的网格伪影与算力成本。

---

## 1. 研究背景与问题定义

### 1.1 潜空间扩散到底牺牲了什么

现代高分辨率生成模型大多先用 VAE 将图像压缩到低分辨率 latent，再在 latent 中运行扩散或 flow matching。它的优势非常直接：如果空间尺度缩小 8 倍，二维 token 数可缩小约 64 倍，训练和采样成本显著下降。代价也同样明确：VAE 是独立训练的有损 codec，纹理、文字、细边缘和精确几何一旦在编码阶段被抹掉，后面的生成器不可能恢复原本不存在的信息。

这使得“像素空间生成器”产生一个看似自然的假设：若模型直接操作 RGB，不再经过偏向自然图像重建的 VAE，那么它不仅应更擅长生成细节，也应更适合作为单目深度、图像恢复等 dense prediction 任务的 backbone。

![](assets/Iris-3B%20-%20无VAE像素空间扩散与通用视觉迁移/figures/01-teaser.png)

> 图 1：Iris-3B 在约一百万像素预算、不同宽高比下直接生成 RGB。文字、材质和摄影样例说明像素空间训练已经能扩展到实用的文生图质量，而不只是 ImageNet 级的小分辨率实验。来源：论文 Figure 1，p.1，https://arxiv.org/abs/2610.09450

### 1.2 论文真正要检验的核心假设

论文不是只发布一个“无 VAE 文生图模型”，而是同时比较两条得到像素 backbone 的路线：

1. **从零训练**：预训练 Iris-3B，按 256→512→1024 的分辨率课程直接学习 RGB flow；
2. **潜转像素**：从已有 FLUX.2 Klein base 4B 出发，移除 VAE，重新学习像素输入投影和像素输出头。

然后将像素模型和潜模型微调到：

- 单目相对深度估计；
- 单步图像恢复与 4× 超分辨率。

核心问题因此不是“像素空间能不能生成图”，而是：**去掉 VAE 后保留下来的细粒度信息，是否能转化为下游 dense task 的稳定收益？**

### 1.3 核心贡献

1. **把 PixelDiT/PiT 路线扩展到 3B 与 1024²。** Iris-3B 用约 4096 个 16×16 patch token 表达一张 1024² 图像，再以浅层 PiT 恢复每个像素，在可承受的 token 数下直接输出 RGB。
2. **系统消融像素空间预训练配方。** 论文比较 $x$-prediction 与 $v$-prediction，并对 REPA、iREPA、DINOv3 teacher、Self-Flow 做同预算消融，最终选择 $v$-prediction + DINOv2 REPA。
3. **给出潜空间模型转换到像素空间的工程配方。** 对 FLUX.2 Klein 使用 gain-matched ridge 初始化新输入投影，并降低 trunk 学习率，使模型在 1K steps 内从 patch 噪声恢复为可生成图像。
4. **完成生成先验到深度和恢复任务的对照实验。** 结果没有支持“pixel prior 必然更好”的直觉，并明确记录负结果、混杂因素和失败模式。
5. **开放接近完整的基础模型实现与权重。** 代码可追踪到 RGB patch、混合双/单流主干、PiT 像素头、REPA、Muon 训练和 4-step Turbo；但论文中的下游训练和 FLUX 转换仍不完整。

---

## 2. 任务定义与输入输出

### 2.1 像素空间 flow matching

给定干净图像 $x_0\in\mathbb{R}^{3\times H\times W}$、高斯噪声 $\epsilon\sim\mathcal{N}(0,I)$、文本条件 $c$ 和噪声水平 $\sigma$，训练路径为：

$$x_\sigma=(1-\sigma)x_0+\sigma\epsilon$$

模型预测恒定速度：

$$v=\epsilon-x_0$$

对应损失为：

$$\mathcal{L}_{\mathrm{flow}}=\mathbb{E}_{x_0,\epsilon,\sigma}\left[\left\|v_\theta(x_\sigma,\sigma,c)-(\epsilon-x_0)\right\|_2^2\right]$$

与 latent diffusion 最大的差别不在 flow 公式，而在 $x_0$：Iris-3B 的 $x_0$ 就是 RGB，不是 VAE latent。因此模型既要建模全局语义，也要承担颜色、纹理、边缘和局部统计的全部生成负担。

### 2.2 时间分布与分辨率 shift

基础噪声变量 $u$ 服从 logit-normal，随后用分辨率相关的 shift 调整：

$$\sigma=\frac{su}{1+(s-1)u},\qquad u\sim\operatorname{logit\text{-}normal}(0,1)$$

256、512、1024 阶段分别使用 $s=2,3,4$。分辨率越高，训练越需要覆盖更偏向高噪声的一侧，以适应更长、更复杂的像素序列。

### 2.3 REPA 辅助目标

256 阶段额外使用 DINOv2-B/14 patch feature 作为表征教师：

$$\mathcal{L}=\mathcal{L}_{\mathrm{flow}}+\lambda\mathcal{L}_{\mathrm{REPA}}$$

其中 $\mathcal{L}_{\mathrm{REPA}}$ 是投影后主干 token 与 DINOv2 patch feature 的负余弦相似度，256 阶段 $\lambda=0.5$，512 和 1024 阶段关闭。REPA 的作用不是替代生成目标，而是在早期让像素主干更快形成语义结构，减少纯 RGB 回归的优化难度。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/Iris-3B%20-%20无VAE像素空间扩散与通用视觉迁移/figures/02-method-architecture.png)

> 图 4：Iris-3B 的完整架构。文本经过冻结 Qwen3-VL-4B 与两层 adapter；图像被切为 16×16 patch；8 个双流块先分别处理文本/图像并联合注意，16 个单流块再统一建模；最后 4 层 PiT 用主干 semantic token 调制 patch 内像素并直接输出 RGB velocity。来源：论文 Figure 4，p.5，https://arxiv.org/abs/2610.09450

1024² 输入经过 16×16 patchify 后得到 $64\times64=4096$ 个 image token。主干宽度为 2560，由 8 个 dual-stream MMDiT block 和 16 个 single-stream block 组成。文本侧读取 Qwen3-VL-4B-Instruct 的 12 个隐藏层，用 layerwise attention pooling 汇聚为 300 个文本 token，再通过两层 refiner 对齐到主干宽度。

### 3.2 混合双流与单流主干

前 8 层双流块让文本和图像保留各自的归一化、调制与 FFN 参数，同时在 joint attention 中交换信息；后 16 层将两类 token 拼接为统一序列，集中完成跨模态融合。对 1024²、文本长度 300 的配置：

- 图像 token：4096；
- 文本 token：300；
- 联合序列：4396；
- hidden width：2560；
- query heads：20；
- KV heads：5；
- head dimension：128。

Grouped-Query Attention 将 KV 宽度从 20 heads 降至 5 heads，降低长序列注意力的 KV 成本。主干还采用 QK RMSNorm、sandwich RMSNorm 和 sigmoid output gate，以稳定 3B 模型的训练。

### 3.3 Shared-bias adaLN

常规 DiT 往往每层各自生成 scale、shift 和 gate。Iris-3B 使用 shared-bias adaLN：跨层共享一个核心调制 bias，再保留每层必要的低秩或局部调制。论文报告在同等 FLOPs 下可减少约 28% 参数，使更多容量用于主干和像素头，而不是重复的条件映射。

### 3.4 PiT 像素头如何把 patch token 还原为像素

PiT 并非简单用一个线性层把 2560 维 token 展开成 $16\times16\times3$。它再次读取原始带噪 RGB，将每个像素映射到 16 维，形成每个 patch 的 $256\times16$ pixel token。主干输出的 semantic token 对这些像素做 pixel-wise adaLN；随后将 patch 内 $256\times16=4096$ 维压缩为一个 1280 维 token，在 4096 个 patch 之间进行 10-head attention，再展开回 256 个像素。该过程重复 4 次，最后线性投影到 RGB velocity。

这个设计的计算动机很明确：

- 24 层主干只处理 4096 个 patch token，而不是 1,048,576 个像素 token；
- 只有 4 层窄 PiT 处理 patch 内像素；
- 全局交互在压缩后的每 patch 单 token 上完成。

但它也直接带来论文观察到的失败模式：**相邻 patch 边界两侧的像素从未在完整像素粒度上共同参与注意力**，跨 patch 信息必须经过 1280 维压缩 token，容易形成 16×16 网格痕迹。

### 3.5 预训练课程与优化器

| 阶段 | 论文 steps | 全局 batch | 样本数 | 主要设置 |
|---|---:|---:|---:|---|
| 256 | 0–367K | 1024 | 375.8M | shift 2；REPA 0.5；LR $10^{-4}$ |
| 512 multi-AR | 367K–530K | 1024 | 166.9M | 21 个宽高比桶；shift 3 |
| 1024 multi-AR | 530K–625K | 512 | 48.6M | shift 4；LR $5\times10^{-5}$ |
| 1024 SFT | 625K–665K | 512 | 20.5M | LR $4\times10^{-5}$ |

训练数据混合真实照片、近期生成器产生的合成图像和文字渲染图。512 与 1024 阶段使用 21 个宽高比桶，覆盖约 1:4 到 4:1，并要求短边达到目标分辨率。优化器为 Muon + AdamW 混合：隐藏矩阵使用 Muon（momentum 0.95、Nesterov、RMS-matched LR），embedding、norm、bias 使用 AdamW（$\beta=(0.9,0.95)$）；weight decay 0、gradient clip 0.5、2K warmup、EMA 0.9999、caption dropout 10%。

### 3.6 潜空间 FLUX.2 Klein 如何转换为像素模型

作者移除 FLUX.2 Klein 的 VAE，加入 16×16 RGB patch 输入投影和 DiP/PiT 输出头。随机初始化输入投影时，模型训练 1K steps 后仍只生成与 prompt 无关的 patch-grid 噪声。原因是 RGB patch 与原 latent token embedding 并不存在高质量的逐 patch 线性等价映射：最佳线性拟合只解释约 14% 的方差。

论文给出两项修复：

1. **Gain-matched ridge initialization**：用岭回归拟合 RGB patch 到源 latent embedding 的映射，但正则强度不追求 held-out $R^2$ 最大，而是匹配源 latent 路径的噪声增益，避免新输入破坏预训练 trunk 的激活尺度；
2. **降低 trunk LR**：trunk 学习率设为输出头的 0.12 倍，让新接口快速学习，同时避免预训练主干被大步更新破坏。

二者结合后，转换模型能在 1K steps 内从 patch 噪声转为符合 prompt 的图像。正式下游实验采用训练 30K steps 的 PiT 版本。但论文未报告该转换模型的 GenEval、DPG 或 FID，因此无法判断它的生成先验到底保留了多少，这也是后续“转换像素模型下游较差”的关键混杂因素。

### 3.7 推理流程与成本

基础版默认 1024²、100-step 二阶 FlowDPM-Solver++、CFG 3。CFG 每步需要 unconditional/conditional 两次前向；按仓库 FLOPs 脚本口径，单次 IrisDiT 前向约 23.7 TFLOPs，因此粗略为：

$$23.7\times100\times2\approx4.74\text{ PFLOPs/image}$$

这还不含 Qwen 文本编码器和未计入脚本的归一化、激活、RoPE 等。像素空间没有 VAE 解码开销，但主干直接处理 4096 个图像 token，整体采样仍非常昂贵。

---

## 4. 数据集与实验设置

### 4.1 文生图评测

Iris-3B 在 1024² 下用 CFG 3、100 steps 采样；每个 prompt 生成 4 个样本，不做 prompt rewriting，也不做 best-of-N。评测覆盖：

- GenEval：对象、颜色、位置、计数等组合遵循；
- DPG-Bench：复杂提示词对齐；
- LongText-Bench：长文本与文字渲染；
- OneIG-Bench：综合图像生成能力。

### 4.2 深度估计

训练数据为 Hypersim 480×640 与 Virtual KITTI 2 352×1216，按 9:1 混合，仅用合成深度监督。三种 backbone 都做单次确定性前向、空 prompt、零输入噪声，并预测相对 log depth。损失为像素 L1 加 5 倍梯度 L1：

$$\mathcal{L}_{\mathrm{depth}}=\frac{1}{|M|}\sum_{p\in M}\left(|\hat d_p-d_p|+5\|\nabla\hat d_p-\nabla d_p\|_1\right)$$

使用 AdamW、LR $3\times10^{-5}$、global batch 32、10K steps。零样本评测 NYUv2、KITTI、ETH3D、ScanNet 与 DIODE，报告 AbsRel 和 $\delta_1$，并用 SEE 衡量软边缘误差。

### 4.3 4× 图像恢复

采用 HYPIR 单步恢复方案：把退化输入双三次上采样到目标尺寸，空 prompt 单次前向，损失为：

$$\mathcal{L}_{\mathrm{SR}}=\|\hat y-y\|_2^2+5\mathcal{L}_{\mathrm{LPIPS}}(\hat y,y)+0.5\mathcal{L}_{\mathrm{adv}}(\hat y)$$

训练使用 Real-ESRGAN 二阶退化、1024² crop。感知损失采用 VGG 特征，对抗 critic 使用冻结的 OpenCLIP ConvNeXt-XXL trunk。Iris-3B 训练 10K steps，并在推理端额外使用 wavelet color correction 与 overlap tiling。

---

## 5. 实验结果

### 5.1 预训练配方消融

在 1.3B、256²、global batch 256 的 PixelDiT 上，$v$-prediction 训练稳定，与 $x$-prediction 在 GenEval 基本持平，但 DPG 与 FID 更好，因而被用于规模化训练。需要注意，两组使用了不同的 timestep distribution，实验没有完全隔离 prediction target 本身。

表征对齐中：

- DINOv2 REPA 的 GenEval 在各 milestone 均领先，DPG/CLIP 也不吃亏；
- iREPA 在 150K/200K 的 FID 一度较好，但到 250K 优势几乎消失，GenEval 与 CLIP 反而偏低；
- DINOv3 teacher 的 FID 更差，最终 DPG 小幅领先被作者判断在评测噪声内；
- Self-Flow 四项指标均明显落后。

因此规模化配方选择 $v$-prediction + DINOv2 REPA，但 REPA 只用于 256 阶段。

### 5.2 文生图：像素空间已经“能用”，但没有全面胜出

| 模型 | 参数量 | GenEval↑ | DPG↑ | LongText↑ | OneIG↑ |
|---|---:|---:|---:|---:|---:|
| FLUX.1-dev | 12B | 0.66 | 83.8 | 0.607 | 0.434 |
| i1* | 3B | 0.84 | 86.7 | 0.922 | — |
| Z-Image* | 6B | 0.84 | 88.1 | 0.935 | 0.546 |
| Qwen-Image | 20B | 0.87 | 88.3 | 0.943 | 0.539 |
| **Iris-3B** | **3B** | **0.798** | **86.52** | **0.857** | **0.540** |

`*` 表示使用 prompt rewriting，不能与 Iris-3B 完全同条件比较。

Iris-3B 明显优于 FLUX.1-dev，并在 OneIG 上与 Qwen-Image 接近，说明 3B 像素模型确实能达到强文生图水平。但它在 GenEval、DPG 和 LongText 上仍落后于更强模型；子项中对象与颜色较好，位置和计数较弱。最准确的结论是“pixel-space 可扩展并具竞争力”，而不是“无 VAE 已经系统超越 latent diffusion”。

### 5.3 深度估计：从零预训练像素模型持平，转换像素模型退步

![](assets/Iris-3B%20-%20无VAE像素空间扩散与通用视觉迁移/figures/03-depth-results.png)

> 图 7：三种 backbone 在 Hypersim 的深度定性比较。Iris-3B 与 latent Klein 的整体结构接近，但单看可视化很难发现 16×16 patch 网格，因此论文另行设计边界二阶差分指标。来源：论文 Figure 7，p.8，https://arxiv.org/abs/2610.09450

| 模型 | 平均 AbsRel↓ | 平均 $\delta_1$↑ | Hypersim SEE↓ |
|---|---:|---:|---:|
| Klein latent | 0.072 | 0.947 | 0.528 |
| Klein pixel | 0.081 | 0.934 | 0.526 |
| **Iris-3B** | **0.071** | **0.946** | **0.496** |

Iris-3B 与 latent Klein 在平均指标上几乎相同：它在 KITTI、DIODE 更好，在 NYUv2、ETH3D、ScanNet 更差；边缘 SEE 最好，但并未形成整体优势。由 FLUX.2 Klein 转换而来的 pixel Klein 则多数基准退步，说明短时像素转换没有自动获得更适合 dense prediction 的 prior。

更关键的是 patch-grid ratio：无网格应接近 1，Klein latent 在不同集合为 0.97–0.99，Klein pixel 为 1.31–1.44，Iris-3B 达到 1.61–2.25。也就是说，Iris-3B 虽然边缘误差较低，却更明显地留下 16×16 patch 周期性结构。

### 5.4 4× 恢复：细节观感有竞争力，保真度没有优势

![](assets/Iris-3B%20-%20无VAE像素空间扩散与通用视觉迁移/figures/04-restoration-results.png)

> 图 8：DIV2K 4× 恢复定性结果。Iris-3B 能生成锐利的动物毛发与局部纹理，但视觉锐利不等于忠实重建；论文的 fidelity 指标仍由 latent Klein 领先。来源：论文 Figure 8，p.10，https://arxiv.org/abs/2610.09450

| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ | NIQE↓ | MUSIQ↑ | DeQA↑ |
|---|---:|---:|---:|---:|---:|---:|
| **Klein latent** | **20.99** | **0.557** | 0.275 | **3.26** | 67.11 | **4.06** |
| Klein pixel | 20.87 | 0.504 | **0.274** | 4.11 | 67.03 | 4.04 |
| Iris-3B | 20.68 | 0.481 | 0.292 | 3.94 | **68.01** | 3.87 |

Iris-3B 的 MUSIQ 最高，说明感知锐度和主观观感有竞争力；但 PSNR、SSIM、LPIPS 和 DeQA 都没有领先。匹配更严格的 Klein latent/pixel 对照中，pixel 版也没有超过 latent 版，尤其 SSIM 与 NIQE 明显较差。论文没有做多次运行或显著性检验，因此作者谨慎表述为“没有优势”，而不是断言像素模型必然更差。

### 5.5 最新项目更新：Iris-3B Turbo

Turbo 是论文发布后项目仓库增加的工程扩展，不属于 arXiv v1 主体实验。它用 Trajectory Distribution Matching 对全部 3B 参数做 data-free 蒸馏，只需文本 prompt，不需真实训练图像。发布版固定 4 个噪声点：

$$\sigma=[1.0,0.938,0.833,0.625]$$

CFG 3.5 被蒸馏进 teacher target，因此推理每步仅一次前向；官方报告 H100、batch 4 下生成 1024² 约 0.7 秒。相对基础版：GenEval 0.798→0.820、DPG 86.52→86.68，但 LongText 0.857→0.804、OneIG 0.540→0.516，且不同 seed 的多样性下降。它证明 Iris-3B 可有效蒸馏为少步模型，但代价是长文字、综合质量和样本多样性。

---

## 6. 与相关工作的关系

Iris-3B 可以放在三条技术线上理解：

1. **相对 Latent Diffusion / FLUX**：它没有改变 flow matching 的概率路径，而是取消 VAE，将表示空间从 latent 换回 RGB；因此“VAE vs 无 VAE”与“DDPM vs Flow Matching”是两条正交选择。
2. **相对 PixelDiT**：Iris-3B 沿用 PiT 头的关键思想，但把模型规模、文本条件和分辨率扩到 3B/1024²，并系统研究 REPA 与下游迁移。
3. **相对 DINOv2 等视觉基础模型**：它尝试让生成 backbone 兼任 dense vision learner，但实验证据只支持“可用”，尚不足以证明生成先验优于专门的判别式表征。

论文也对“generative prior 天生保存细节”的直觉作了重要修正：表示空间不经过 VAE，并不意味着 backbone 一定学到更适合几何或恢复的特征；训练目标、patch 接口、下游读出方式、预训练数据和适配预算都可能比“是否有 VAE”更关键。

---

## 7. 局限与批判性评价

### 7.1 对照并没有完全隔离“像素空间”变量

Iris-3B 没有同数据、同算力、同架构的 latent twin；它与 FLUX.2 Klein 在预训练数据、参数、训练时长和工程配方上都不同。FLUX.2 Klein 的 pixel 版虽然与 latent 版共享 trunk 起点，但只做了 30K steps 转换，且论文没有报告转换后的生成质量。因此，负结果可以说明“这两条现实可行的 pixel 路线没有显示优势”，却不能严格证明“pixel prior 在理想条件下不优于 latent prior”。

### 7.2 “无 VAE”消除了 codec，却没有消除 bottleneck

PiT 为了控制计算，把 16×16 patch 内 256 个像素压缩为一个 1280 维 token再跨 patch 交互。这本身就是新的局部瓶颈。换言之，Iris-3B 去掉了学习式 VAE codec，却引入了固定 patch 划分和 patch-level communication bottleneck。网格指标证明该瓶颈不是抽象担忧，而是可测的输出结构。

### 7.3 实验预算偏短，缺少方差与显著性

深度与恢复均是单次、10K 级短训练，没有多 seed 方差或显著性检验。许多差异只有千分位或百分位，不能过度解释。作者在结论中主动承认这一点，这是论文可信度的加分项，但也限制了最终判断强度。

### 7.4 数据和下游训练没有完整开源

基础预训练代码较完整，但训练数据、SFT 数据比例和过滤流程未公开；Turbo prompt 集也未公开。官方仓库没有论文中的深度训练器、Real-ESRGAN 退化与对抗恢复训练器，也没有 FLUX.2 Klein ridge 转换代码。因此当前能够复现的是架构、基础训练入口与推理，而不是论文所有关键对照。

### 7.5 计算代价仍是核心障碍

基础版 100-step CFG 推理粗略达到 4.74 PFLOPs/图；官方 release 的三种任务模型同时驻 GPU 约需 45 GB，offload 后单任务峰值约 20 GB。Turbo 缓解了推理步数，却不改变 3B backbone 的单次前向成本；像素空间若要真正替代 latent 路线，需要同时解决 token 数、输出头和少步采样。

### 7.6 个人判断

这篇论文最有价值的部分不是“无 VAE 的 3B 模型”，而是**对一个非常诱人的直觉做了高成本证伪**：VAE 的确有损，但下游 dense task 的瓶颈并不一定由 VAE 主导。Iris-3B 在生成侧成功、迁移侧不占优，提示未来研究应把问题从“删不删 VAE”改写为“哪一种中间表示、空间接口和训练目标能同时保留生成质量与可读出的几何信息”。综合方法完整性、负结果价值和开源程度，评级为 **★★★★☆**。

---

## 8. 复现与实践建议

### 8.1 最小推理路径

```bash
git clone https://github.com/speridlabs/iris-3b.git
cd iris-3b
pip install -e .

hf download speridlabs/iris-3b \
  --local-dir iris-3b \
  --exclude "depth/*" "upscaler/*" "turbo/*"

python scripts/sample.py \
  --checkpoint iris-3b \
  --prompt "a red fox sleeping in fresh snow, golden hour"
```

环境要求 Python 3.11+、PyTorch 2.7.1+、NVIDIA CUDA。基础版默认 1024²、100 steps、CFG 3。Qwen3-VL-4B-Instruct 会在首次运行时自动下载。

### 8.2 下游推理

```bash
python scripts/depth.py photo.jpg --out depth_out
python scripts/upscale.py photo.jpg --out upscaled
```

深度输出是相对 log depth，不是米制深度；比较真值时需在 log space 拟合 scale 和 shift。超分默认 4×，大图按 1024² tile、stride 512 重叠推理，并做 wavelet color correction。

### 8.3 不建议直接尝试完整预训练

官方配置对应 64-GPU FSDP2 hybrid 作业，且数据未公开。更现实的研究路径是：

1. 先运行 release 权重验证基础生成、深度和超分；
2. 在 256²、小模型上复现 $v$-prediction + REPA；
3. 若研究 patch artifact，优先替换 PiT 输出头为重叠卷积或 PixelShuffle，而不是复刻 665K steps；
4. 做 pixel/latent 公平比较时，必须共享数据、算力、trunk 和下游训练预算，并额外报告生成先验质量。

---

## 9. 个人启发与后续问题

### 9.1 对“VAE 到底压缩什么”的直接回答

VAE 不只压缩像素数量，还选择性压缩局部纹理、高频边缘、精细文字与严格颜色。Iris-3B 说明：去掉这种 codec loss 可以让生成器自己承担全部像素建模，但**信息没有被 VAE 丢弃**与**这些信息能被下游线性或轻量读出**是两回事。

### 9.2 表示学习与生成能力不能简单等同

生成质量高说明模型能在条件分布上合成逼真样本，不等于中间特征天然具有最适合深度/几何的坐标系。Iris-3B 甚至使用 DINOv2 REPA 来帮助早期预训练，这反过来表明判别式视觉表征仍是生成训练的重要教师。

### 9.3 值得继续追踪的问题

1. 若 pixel 与 latent 模型使用完全相同的数据、计算和架构，结论是否仍然成立？
2. 重叠卷积或 PixelShuffle 输出头能否消除 16×16 网格，同时保持 PiT 的全局建模能力？
3. 直接预测法线、边缘、光流等更局部的任务，是否比相对深度更能体现无 VAE 优势？
4. 对统一理解-生成模型而言，RGB patch 是否能成为真正共享的视觉接口，还是最终仍需要多尺度 latent？
5. Turbo 的 data-free TDM 是否可以用于深度/恢复的少步或单步适配，而不牺牲多样性和文字能力？

---

## 10. 代码级实现

### 10.1 仓库概览与模块映射

| 论文模块 | 代码入口 | 作用 |
|---|---|---|
| Iris-3B 主体 | `src/iris3b/models/dit.py::IrisDiT` | patchify、文本适配、8+16 主干、PiT 与 RGB fold |
| 双流块 | `models/blocks/mmdit.py::MMDiTBlock` | 文本/图像独立参数与联合注意力 |
| 单流块 | `models/blocks/single_stream.py::SingleStreamBlock` | 统一 4396-token 序列建模 |
| PiT 头 | `models/blocks/pit.py::PiTBlock` | patch 内像素调制、压缩、跨 patch 注意力与展开 |
| 文本编码 | `text/qwen3_vl.py::Qwen3VLTextEncoder` | 提取 Qwen 12 层、每层 300×2560 表征 |
| 文本聚合 | `nn/embeddings.py::LayerwiseTextEmbedder` | layer pooling 与两层 token refiner |
| Flow | `flow/transport.py::RectifiedFlow` | 构造 $x_\sigma$ 与 velocity target |
| REPA | `repa.py::REPALoss` | 第 10 个 block token 对齐 DINOv2-B/14 |
| 基础采样 | `sampling.py::generate`、`flow/solver.py::FlowDPMSolver` | 100-step DPM-Solver++ 与 CFG |
| 基础训练 | `scripts/train.py`、`train/trainer.py` | 256→512→1024→SFT |
| 深度/恢复 | `downstream/depth.py`、`downstream/restoration.py` | 发布权重推理；不含论文训练器 |
| Turbo | `scripts/distill.py`、`distill/trainer.py` | TDM 全参数蒸馏与 4-step 推理 |

### 10.2 1024² 全链路张量总表

以下以 $B=1$、文本长度 300、不计 CFG 批次翻倍为例：

| # | 模块 | 输入形状 | 输出形状 | 关键计算 |
|---:|---|---|---|---|
| 1 | noisy RGB | `[1,3,1024,1024]` | `[1,3,1024,1024]` | $x_\sigma=(1-\sigma)x_0+\sigma\epsilon$ |
| 2 | `F.unfold` | `[1,3,1024,1024]` | `[1,4096,768]` | 4096 patches；每 patch $16\times16\times3=768$ |
| 3 | patch embed | `[1,4096,768]` | `[1,4096,2560]` | Linear 768→2560 |
| 4 | Qwen 12 层 | token ids | `[1,300,12,2560]` | 层索引 2,5,…,35 |
| 5 | text adapter | `[1,300,12,2560]` | `[1,300,2560]` | layer pooling + 2-block refiner |
| 6 | 8 dual blocks | image `[1,4096,2560]`；text `[1,300,2560]` | 同形状 | joint attention 序列长 4396 |
| 7 | 16 single blocks | `[1,4396,2560]` | `[1,4396,2560]` | 20 Q heads / 5 KV heads / head dim 128 |
| 8 | pixel embed | RGB `[1,3,1024,1024]` | `[4096,256,16]` | 每 patch 256 pixels，每像素 16 维 |
| 9 | PiT compress | `[4096,256,16]` | `[1,4096,1280]` | 每 patch $256\times16=4096$→1280 |
| 10 | PiT attention | `[1,4096,1280]` | `[1,4096,1280]` | 10 heads，4 个 PiT blocks |
| 11 | PiT expand | `[1,4096,1280]` | `[4096,256,16]` | 1280→4096→256×16 |
| 12 | RGB projection/fold | `[4096,256,16]` | `[1,3,1024,1024]` | Linear 16→3 后 `F.fold` |

基础版默认 CFG 3，solver 会将 conditional/unconditional 拼成 $B=2$，因此联合序列实际为 `[2,4396,2560]`，PiT pixel batch 变为 `[8192,256,16]`。

### 10.3 逐模块拆解

#### 10.3.1 文本编码与 12 层聚合

**输入输出。** Qwen3-VL-4B-Instruct 使用固定 chat prefix，将 prompt 填入 300-token 窗口；提取第 `[2,5,8,11,14,17,20,23,26,29,32,35]` 层，堆成 `[B,300,12,2560]`。LayerwiseTextEmbedder 先沿“层”维做 2 层 Transformer 与 learned pooling，得到 `[B,300,2560]`，再做 2 个 token refiner block。

**为什么这样设计。** 最后层往往更偏语言生成，而不同层包含词法、句法和语义的不同粒度；可学习层聚合允许图像模型决定每个 token 应读取哪一层。冻结 Qwen 可降低训练成本，并避免生成训练破坏语言能力。

**关键超参。** 文本长度 300；12 个隐藏层；layer-axis 32 heads、MLP ratio 1.3；token refiner 2 blocks、20 heads、SwiGLU ratio 4；同时使用 learned absolute position 和 text RoPE。

**代码与论文差异。** 论文只写“12 层 layerwise attention pooling + 2-block refiner”，代码才给出具体层索引、chat template、head 数、双位置编码和 padding 行为。主干 joint attention 不屏蔽 pad token，mask 只用于文本 adapter，这是潜在的实现细节风险。

#### 10.3.2 8+16 混合主干

**输入输出。** 双流阶段维持 image `[B,4096,2560]` 与 text `[B,300,2560]`；Q 为 20×128，K/V 为 5×128，联合注意力长度 4396。单流阶段直接拼接为 `[B,4396,2560]`，16 层后再取 image 部分。

**为什么这样设计。** 双流阶段允许两个模态各自归一化和非线性变换，适合早期对齐；单流阶段让跨模态关系被统一建模，减少后段重复参数。GQA 缩小 KV 投影与缓存，适配 4396-token 序列。

**关键超参。** width 2560；8 dual + 16 single；20 Q / 5 KV heads；QK RMSNorm；sandwich RMSNorm；sigmoid content gate；SwiGLU；shared-bias adaLN。

**代码与论文差异。** 高层结构一致。代码额外确认 single-stream 尾部复用 image modulation core、SwiGLU 实际中间宽度为 6826，并通过单元测试锁定精确参数量 2,987,511,888。

#### 10.3.3 PiT 像素头

**输入输出。** noisy RGB 逐像素 Linear 3→16，重排为 `[B\times4096,256,16]`。每个 PiT block 用 semantic token 生成 pixel-wise adaLN 参数；将每 patch 的 4096 维像素特征压缩到 1280，重排为 `[B,4096,1280]` 做 10-head attention，再展开回 256×16。4 层后 Linear 16→3 并 fold 回图像。

**为什么这样设计。** 若对百万像素直接做 24 层全局 attention，成本不可接受；PiT 把全局语义计算放在 patch token，把局部像素恢复放在浅层窄网络，形成“语义低分辨率、像素高分辨率”的两级结构。

**关键超参。** depth 4；pixel width 16；attention width 1280；10 heads；GELU MLP ratio 4；post modulation；固定二维 sin/cos pixel position。

**代码与论文差异。** 论文披露 4 blocks 与 1280 compact token，代码进一步给出 pixel width=16、10 heads、post modulation、GELU 和绝对位置编码。论文建议未来改用重叠卷积或 PixelShuffle，以缓解当前非重叠 patch 的网格伪影。

#### 10.3.4 Flow 与 REPA

**输入输出。** RectifiedFlow 由 RGB $x_0$ 和 noise 构造 $x_\sigma$，目标为 `[B,3,H,W]` 的 $\epsilon-x_0$。REPA 捕获第 10 个 trunk block 的 token，经 2560→2048→2048→768 projector，对齐 DINOv2-B/14 的 16×16 patch grid。

**为什么这样设计。** $v$-prediction 对整个路径使用同一目标方向，实验中比 $x$-prediction 稳定。256² 输入与 DINOv2 的 224² teacher 都产生 16×16 token grid，REPA 无需复杂插值即可一一对齐。

**关键超参。** 1000 train timesteps；logit-normal $(0,1)$；shift 2/3/4；REPA layer 10；teacher dim 768；256 阶段权重 0.5。

**代码与论文差异。** 公式和阶段设置一致；capture block、projector 宽度以及 teacher resize/normalization 主要由代码补充。

### 10.4 代码与论文的关键差异

1. **Stage 1 steps 不一致。** 论文 Table 1 为 367K，当前仓库 config/README 为 365K；相差 2K×1024=2.048M 个训练样本。
2. **下游训练并未随仓库完整发布。** 深度仅有推理、1×1 depth head 和 checkpoint exporter；恢复仅有单步推理、tiling 与 wavelet color fix。论文中的深度损失与数据加载、Real-ESRGAN 退化、LPIPS、OpenCLIP critic、对抗训练均未找到。
3. **FLUX.2 Klein conversion 未开源。** 仓库不包含 ridge 初始化、latent-to-pixel 转换和对应训练脚本，因此论文最关键的 matched latent/pixel 对照无法直接复现。
4. **预训练数据与 SFT 配比未公开。** 代码能运行在自有 WebDataset/WIDS 数据上，但不能重建官方 checkpoint 的数据分布。
5. **Turbo 是后续工程扩展。** 当前仓库完整包含 TDM Turbo，而 arXiv v1 主体没有该训练配方；阅读时不应把 Turbo 指标混入论文原始结论。

### 10.5 计算账与复现建议

| 阶段 | 序列/分辨率 | 宽度与层数 | 前向次数 | 资源判断 |
|---|---|---|---:|---|
| 基础版单步 | 4096 image + 300 text | 2560；8+16+4 blocks | 1 | 约 23.7 TFLOPs（仓库计数口径） |
| 基础版采样 | 1024² | 同上 | 100×2 CFG | 约 4.74 PFLOPs/图 |
| Turbo | 约 1 MP | 同上 | 4，无 CFG | 约 94.8 TFLOPs/图；H100 batch 4 约 0.7 s |
| 基础预训练 | 256→512→1024 | 2.99B + 冻结 Qwen | 665K updates | 官方 config 为 64-GPU FSDP2 hybrid |
| 三任务模型驻留 | generation + depth + upscaler | 每个约 12 GB FP32 | — | 同时驻 GPU 约 45 GB；offload 峰值约 20 GB |

最小可跑路径应优先使用官方 release；若研究训练，先在 256² 和较小模型上验证数据、flow 与 REPA。完整论文复现目前受训练数据、下游 trainer 和 FLUX conversion 代码缺失限制，不能把“基础训练入口可运行”误写为“论文全部实验可复现”。

---

## 11. 材料来源

- 论文：https://arxiv.org/abs/2610.09450
- 官方项目页：https://speridlabs.com/research/iris
- 官方代码：https://github.com/speridlabs/iris-3b
- 官方模型：https://huggingface.co/speridlabs/iris-3b
- 本地 PDF、配置、关键图与代码阅读记录： [sources](assets/Iris-3B%20-%20无VAE像素空间扩散与通用视觉迁移/sources.md)

