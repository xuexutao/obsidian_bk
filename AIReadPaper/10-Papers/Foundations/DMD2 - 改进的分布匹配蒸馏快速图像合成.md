---
type: paper-note
status: done
domain: Foundations
paper: Improved Distribution Matching Distillation for Fast Image Synthesis
year: 2024
arxiv: 2405.14867
doi: null
source: https://arxiv.org/abs/2405.14867
project: https://tianweiy.github.io/dmd2/
code: https://github.com/tianweiy/DMD2
tags:
  - Foundations
  - Distribution-Matching
  - Distillation
  - DMD
  - GAN
  - Text-to-Image
created: 2026-08-29
updated: 2026-08-29
---

# DMD2：改进的分布匹配蒸馏快速图像合成

## 基本信息

- **完整标题**：Improved Distribution Matching Distillation for Fast Image Synthesis
- **作者**：Tianwei Yin¹、Michaël Gharbi²、Taesung Park²、Richard Zhang²、Eli Shechtman²、Frédo Durand¹、William T. Freeman¹（¹MIT、²Adobe Research）
- **发表**：arXiv 预印本（Preprint, Under review）
- **arXiv**：2405.14867v2 [cs.CV]，2024 年 5 月 24 日
- **项目页**：https://tianweiy.github.io/dmd2/
- **代码仓库**：https://github.com/tianweiy/DMD2
- **历史地位**：DMD 的直系改进版，同一位第一作者。它解决了 DMD 两个根本痛点——回归损失的昂贵数据构造与学生质量被教师上界限制——并通过"两时间尺度更新规则 + GAN + 多步 backward simulation"反超教师，为后续 Self-Forcing 系列在视频上使用 GAN/SiD 混合目标提供了直接的方法论来源。

## 一句话结论

DMD2 通过三项技术升级把 DMD 推向新高度：① **彻底去掉回归损失**，用"两时间尺度更新规则（TTUR）"解决 fake critic 估计不准导致的训练不稳；② **引入 GAN 损失**在真实数据上训练判别器，让学生摆脱教师真实分数估计误差的束缚、得以反超教师；③ **支持多步采样**，并用"backward simulation"在训练时模拟推理期输入、消除训练/推理不匹配。最终 ImageNet 64×64 FID 1.28、COCO 2014 零样本 8.35，均反超教师，且能蒸馏 SDXL 生成 1024×1024 百万像素图。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

DMD 已在蒸馏 SD 1.5 上达到 SOTA，但其两大限制在本文被明确点出：

1. **回归损失太贵**：为构造回归损失，需用教师跑完整采样步数生成数百万张噪声-图像配对。对 SDXL，单张配对约 5 秒，覆盖 LAION 6.0 的 1200 万提示词需约 **700 A100 天**——光是数据集构造就超过 DMD2 总训练算力的 4 倍。
2. **回归损失与分布匹配理念相悖**：回归损失强迫学生贴合教师的采样路径，导致学生质量**被教师路径上界限制**，永远无法反超教师。

因此 DMD2 的核心问题是：**能不能去掉回归损失，同时保持训练稳定，甚至让一步/少步学生反超教师？**

### 1.2 本文核心贡献

1. 提出**无需回归损失**的稳定分布匹配蒸馏，消除昂贵数据构造。
2. 定位不稳定的根因是"fake 扩散 critic 训练不足"，用**两时间尺度更新规则（TTUR）**解决。
3. 把**GAN 目标**整合进 DMD（在真实数据上训练判别器），在分布层面增强监督、缓解教师分数近似误差。
4. 支持**多步生成器**，并用 backward simulation 消除训练/推理输入不匹配。

![](assets/DMD2%20-%20改进的分布匹配蒸馏快速图像合成/figures/01-sdxl-samples.png)

> 图 1：DMD2 蒸馏 SDXL 得到的 4 步生成器产出的 1024×1024 样张。证明分布匹配蒸馏可以扩展到百万像素级高分辨率，且视觉质量突出。
> 来源：论文 Figure 1，第 2 页，https://arxiv.org/abs/2405.14867

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：高斯噪声 $z\sim\mathcal{N}(0,I)$；多步生成器还会接收中间时间步 $t_i$ 的噪声图像。
- **输出**：真实感图像（一步或 $N$ 步，$N=4$ 为主）。
- **假设**：给定预训练扩散教师 $\mu_{\text{real}}$（EDM / SD v1.5 / SDXL），DMD2 在其上微调出少步生成器 $G_\theta$，并额外用一批**真实图像**训练 GAN 判别器。

### 2.2 关键符号

沿用 DMD 的符号：扩散前向 $q_t(x_t|x)\sim\mathcal{N}(\alpha_t x,\sigma_t^2 I)$，分数 $s_{\text{real}}(x_t,t)=-\frac{x_t-\alpha_t\mu_{\text{real}}(x_t,t)}{\sigma_t^2}$（式 (1)），DMD 的分布匹配梯度（式 (2)）与回归损失（式 (3)）为对照基线。

## 3. 核心方法

### 3.1 总体框架

![](assets/DMD2%20-%20改进的分布匹配蒸馏快速图像合成/figures/02-method.png)

> 图 3：DMD2 把昂贵的扩散教师（灰，右）蒸馏成一步或多步生成器（红，左）。训练交替两步：① 用隐式分布匹配目标的梯度（红箭头）加 GAN 损失（绿）优化生成器；② 训练 fake 分数函数（蓝）建模生成器输出分布，同时训练 GAN 判别器（绿）区分 fake 与真实图像。生成器可以是一步或多步（图中为含中间步输入的多步模型）。
> 来源：论文 Figure 3，第 6 页，https://arxiv.org/abs/2405.14867

### 3.2 改进一：去掉回归损失

DMD 的回归损失（式 (3)）保证模态覆盖与稳定，但让大规模蒸馏笨重、且与分布匹配理念冲突。DMD2 的第一步就是**移除它**，成为"纯分布匹配"。

### 3.3 改进二：两时间尺度更新规则（TTUR）

直接去掉回归损失会导致训练不稳定、质量显著下降（Table 3：ImageNet FID 从 2.62 掉到 3.48），作者观察到生成样本的**平均亮度等统计量剧烈波动、无法收敛**。根因是：fake 扩散模型 $\mu^{\text{fake}}_{\phi}$ 在生成器的**非平稳输出分布**上动态优化，无法准确追踪 fake 分数，产生近似误差与有偏的生成器梯度。

解决思路借鉴 Heusel et al. 的 TTUR：**以不同频率更新 $\mu^{\text{fake}}_{\phi}$ 与生成器 $G$**，确保 fake 分数准确追踪生成器分布。实验发现**每更新 1 次生成器、更新 5 次 fake 分数**，无需回归损失即可达到与原始 DMD 相当的质量（Table 3），且收敛更快。频率分析（附录 Fig.9）：1 次不稳、10 次过慢，5 次是稳定性与速度的最佳平衡。

### 3.4 改进三：GAN 损失 + 真实数据

即便稳定了，蒸馏生成器与教师之间仍有性能差距。作者归因于**真实分数函数 $\mu_{\text{real}}$ 的近似误差**——DMD 从不接触真实数据，无法从这些误差中恢复。为此引入 GAN 目标：判别器 $D$ 区分真实图像与生成器图像，在真实数据上训练，不继承教师的局限，使学生得以反超教师。

GAN 采用极简设计：在 fake 扩散去噪器（UNet）**瓶颈层上加一个分类分支**，用非饱和 GAN 目标训练分类分支与上游编码器特征：

$$L_{\text{GAN}} = \mathbb{E}_{x\sim p_{\text{real}},t}\big[\log D(F(x,t))\big] + \mathbb{E}_{z,t}\big[-\log\big(D(F(G_\theta(z),t))\big)\big] \tag{4}$$

其中 $F$ 是前向扩散（噪声注入）。该 GAN 目标不需要配对数据、与教师采样轨迹无关，因此与分布匹配哲学更一致。

### 3.5 改进四：多步生成器

一步蒸馏在 ImageNet/COCO 上已能匹配教师，但 SDXL 这类大模型因容量受限、噪声到高清图的直接映射优化困难，难以一步到位。DMD2 扩展为多步采样：固定时间步调度 $\{t_1,\dots,t_N\}$（训练=推理一致），推理时交替"去噪"与"噪声注入"（借鉴一致性模型）：

$$\hat{x}_{t_i} = G_\theta(x_{t_i}, t_i),\qquad x_{t_{i+1}} = \alpha_{t_{i+1}}\hat{x}_{t_i} + \sigma_{t_{i+1}}\epsilon$$

4 步模型用调度 $\{999, 749, 499, 249\}$（教师 1000 步）。

### 3.6 改进五：backward simulation（消除训练/推理不匹配）

此前多步生成器训练时用"噪声化的真实图像"作为输入，但推理时（除第一步外）输入来自**前一步生成器的采样输出**，造成训练/推理不匹配。DMD2 用**当前学生生成器跑几步产生的噪声合成图像**替换训练时的噪声真实图像（backward simulation），模拟推理期输入。由于学生只有几步，这可行；且分布匹配损失不依赖学生输入，进一步避免了并发工作 Imagine Flash 中"教师回归损失因从未见过合成图而累积误差"的问题。

## 4. 数据集与实验设置

### 4.1 数据集

- **ImageNet 64×64**（class-cond）：教师 EDM，一步生成器。
- **COCO 2014**（zero-shot 文生图）：教师 SD v1.5 与 SDXL，用 LAION-Aesthetic 约 300 万提示词蒸馏；另取 50 万张 LAION-Aesthetic 5.5+ 真实图训练 GAN 判别器（过滤 <1024×1024 与不安全内容）。

### 4.2 评价指标

FID（clean-fid）、CLIP Score（OpenCLIP-G）、SDXL 额外报告 **Patch FID**（299 中心裁剪）；另做人工偏好研究（PartiPrompts 的 128 提示词，Prolific 平台，5 人投票）。

### 4.3 实现细节

- ImageNet：lr=2e-6、batch 280、7×A100、20 万迭代（约 2 天），fake 更新频率 5，GAN 权重 3e-3；加长训练版先无 GAN 预训练 40 万迭代再启用 GAN 续训 15 万迭代（约 5 天）。
- SD v1.5：两阶段（先无 GAN lr=1e-5、64×A100、4 万迭代，再启用 GAN lr=5e-7、5 千迭代），约 26 小时。
- SDXL：4 步生成器 lr=5e-7、batch 128、64×A100、2 万迭代（约 60 小时）。

## 5. 实验结果

### 5.1 主要定量结果

**ImageNet 64×64（Table 1）**：DMD2 一步 FID **1.51**，加长训练 **1.28**，超过 DMD（2.62）、CTM（1.92）、StyleGAN-XL（1.52），并**反超教师 EDM-ODE（2.32）**。

**COCO 零样本（Table 5）**：DMD2 一步 FID **8.35**，比 DMD（11.49）提升 3.14，反超 SD v1.5 50 步 ODE 教师（8.59）。

**SDXL（Table 2）**：4 步 FID **19.32**、Patch FID 20.86、CLIP 0.332，与 SDXL 教师（100 步，19.36/21.38/0.332）持平，成本降 25×。

### 5.2 消融实验

- **ImageNet（Table 3）**：去回归损失→FID 3.48（不稳）；+TTUR→2.61（恢复并更快）；+GAN→1.51；对比"纯 GAN"2.56、"GAN+TTUR"2.52，说明**分布匹配与 GAN 联合**才是关键。
- **SDXL（Table 4）**：去 GAN→FID 26.90（过饱和、过平滑）；去分布匹配（纯 GAN）→13.77 但文本对齐差、缺 CFG；去 backward simulation→20.66。

### 5.3 定性结果

![](assets/DMD2%20-%20改进的分布匹配蒸馏快速图像合成/figures/03-visual-comparison.png)

> 图 6：DMD2 与 SDXL 教师及 LCM/Turbo/Lightning 的可视化对比，蒸馏模型均用 4 步、教师 50 步、同噪声同提示词。DMD2 在真实感与文本对齐上优于各基线。
> 来源：论文 Figure 6，第 10 页，https://arxiv.org/abs/2405.14867

## 6. 与相关工作的关系

- **相对 DMD**：去掉回归损失、加 TTUR、加 GAN、支持多步，反超教师。
- **相对 ADD / LADD / SDXL-Lightning / SDXL-Turbo**：这些是 GAN 路线或"对抗+轨迹蒸馏"混合；DMD2 是"分布匹配+GAN"的另一种融合，保留了分布匹配在 CFG 上的天然优势。
- **相对 Imagine Flash（并发）**：同样做 backward distillation 缩小训练/测试差距，但其教师回归损失仍会因合成输入累积误差；DMD2 的分布匹配损失不依赖学生输入，规避了这一问题。
- **相对 SiD**：SiD 用 Fisher 散度与分数梯度实现指数收敛，是分布匹配家族的另一种 loss 设计，与 DMD2 的 GAN/TTUR 路线互补。

## 7. 局限与批判性评价

1. **多样性略降**：相较教师，蒸馏生成器多样性（LPIPS diversity）稍有退化（附录 Table 6）。
2. **仍需 4 步**：匹配最大 SDXL 教师质量仍要 4 步，未做到一步。
3. **引导尺度固定**：训练固定 CFG，用户推理无法灵活调节。
4. **训练算力门槛高**：SDXL 蒸馏仍需 64×A100 数十小时，普通研究者难以复现。

## 8. 复现与实践建议

- 官方代码与预训练模型已开源：https://github.com/tianweiy/DMD2
- 关键超参：fake 更新频率 5（新模型/数据集建议调到"保证像素亮度等统计量稳定"的最小值）；GAN 权重 3e-3（ImageNet）/ 1e-3（SD v1.5）。
- 4 步调度 $\{999,749,499,249\}$；SDXL 一步模型有 block noise 伪影，可用 timestep shift（条件时间步设 399）缓解。

## 9. 个人启发与后续问题

DMD2 的"backward simulation"思想——**训练时用学生自己的采样输出作为后续步输入，模拟推理路径**——正是后来 Self-Forcing 系列在视频上"self-rollout + 可微 KV cache"的先声：二者都在解决"训练与推理暴露不一致"这一根本问题，只是 DMD2 在图像多步采样上、Self-Forcing 在因果视频自回归上。

同时，DMD2 引入 GAN 在真实数据上训练、让蒸馏学生反超教师的做法，被 Self-Forcing 直接继承为"video-level SiD/DMD/GAN 混合目标"，是这条路线从"逼近教师"走向"超越教师"的关键转折点。

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | arXiv 2405.14867v2 | 全文 | 2026-08-29 | 精读主来源 |
| `figures/01-sdxl-samples.png` | 高分辨率样张 | arXiv HTML 渲染 | Figure 1, p.2 | 2026-08-29 | 第 1 章"研究背景" |
| `figures/02-method.png` | 方法图 | arXiv HTML 渲染 | Figure 3, p.6 | 2026-08-29 | 第 3.1 节"总体框架" |
| `figures/03-visual-comparison.png` | 定性对比图 | arXiv HTML 渲染 | Figure 6, p.10 | 2026-08-29 | 第 5.3 节"定性结果" |
