---
type: paper-note
status: done
domain: Foundations
paper: Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation
year: 2024
arxiv: 2404.04057
doi: null
source: https://arxiv.org/abs/2404.04057
project: null
code: https://github.com/mingyuanzhou/SiD
tags:
  - Foundations
  - Score-Identity
  - Distillation
  - Fisher-Divergence
  - Data-Free
  - One-Step
created: 2026-08-29
updated: 2026-08-29
---

# SiD：分数恒等蒸馏一步生成

## 基本信息

- **完整标题**：Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation
- **作者**：Mingyuan Zhou¹,²、Huangjie Zheng¹、Zhendong Wang¹、Mingzhang Yin³、Hai Huang²（¹The University of Texas at Austin、²Google、³University of Florida）
- **发表**：ICML 2024（Proceedings of the 41st International Conference on Machine Learning, PMLR 235）
- **arXiv**：2404.04057v3 [cs.LG]，2024 年 5 月 24 日
- **代码仓库**：https://github.com/mingyuanzhou/SiD
- **历史地位**：作者 Mingyuan Zhou 正是后续 Self-Forcing（NeurIPS 2025 spotlight）的作者之一。SiD 提出的**数据无关（data-free）分数恒等蒸馏**与 Fisher 散度（MESM）损失，是因果视频蒸馏路线"分布匹配"源流的另一支——它证明即便**完全不接触真实数据**，学生也能逼近甚至反超教师，为 Self-Forcing 在视频上的 video-level SiD 目标提供了理论根基。

## 一句话结论

SiD 是一种**数据无关**的一步蒸馏方法：把扩散前向过程看作半隐式分布，借助三个"分数恒等式"构造出一个**模型基显式分数匹配（MESM，即 Fisher 散度）**损失，用它来对齐"扩散真实分布"与"扩散生成分布"在各噪声级别上的分数；训练只用生成器自己合成的图像、完全不需要真实数据或反向采样，却实现 FID 的**指数级快速下降**，在 CIFAR-10 / ImageNet 64×64 / FFHQ / AFHQ 四个基准上一举追平乃至反超 EDM 教师。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

扩散模型的采样是迭代的：EDM 在 CIFAR-10 上要 35 NFE、ImageNet 64×64 上要 511 NFE 才能取得好效果。已有加速方法（快速采样器、轨迹蒸馏、一致性模型、GAN 混合）要么仍需不少步数，要么依赖训练数据或额外 GAN 损失。SiD 的目标更激进：**用一个单步生成器蒸馏预训练扩散模型中蕴含的知识，且整个过程不碰真实数据**。

### 1.2 核心洞察

作者从一个新视角——**把前向扩散过程看作半隐式分布（semi-implicit distribution）**——重新审视蒸馏。前向边缘分布

$$p_{\text{data}}(x_t) = \int q(x_t | x_0)\, p_{\text{data}}(x_0)\, dx_0 \tag{1}$$

的转移条件 $q(x_t|x_0)=\mathcal{N}(\alpha_t x_0,\sigma_t^2 I)$ 解析可知，但 $p_{\text{data}}(x_0)$ 未知、仅由经验样本表示——这正是"半隐式分布"的定义。生成器分布 $p_\theta(x_g)$（$x_g=G_\theta(z)$）经扩散也形成半隐式分布：

$$p_\theta(x_t) = \int q(x_t | x_g)\, p_\theta(x_g)\, dx_g \tag{3}$$

在此基础上，作者提出三个分数恒等式，融合成一个可优化的蒸馏损失。

![](assets/SiD%20-%20分数恒等蒸馏一步生成/figures/01-distillation-progression.png)

> 图 1：SiD 的蒸馏进程。同一组随机噪声，用预训练 ImageNet 64×64 扩散模型蒸馏的 SiD 生成器，在训练使用 0 → 5000 万张自合成图像的过程中，画面从纯噪声快速演化为清晰照片，对应 FID 从 153.52 一路降到 3.07，直观展示了"指数级快速收敛"。
> 来源：论文 Figure 1，第 2 页，https://arxiv.org/abs/2404.04057

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：高斯噪声 $z\sim\mathcal{N}(0,I)$，生成器以 $x_g = G_\theta(\sigma_{\text{init}} z)$ 出图，$\sigma_{\text{init}}=2.5$ 固定。
- **输出**：一张图像 $x = G_\theta(\sigma_{\text{init}} z)$。
- **假设**：给定预训练分数网络 $f_\phi$（教师，如 EDM）。SiD 维护三个网络：冻结的教师 $f_\phi$、生成器 $G_\theta$、生成器分数网络 $f_\psi$（均与 $f_\phi$ 同尺寸，从 $\phi$ 初始化）。

### 2.2 关键符号

- 教师分数近似：$\nabla_{x_t}\ln p_{\text{data}}(x_t) \approx S_\phi(x_t) := \sigma_t^{-2}\big(f_\phi(x_t,t) - x_t\big)$，其中 $f_\phi(x_t,t)\approx\mathbb{E}[x_0|x_t]$。
- 分数差（score difference）：$\delta_{\phi,\theta}(x_t) := S_\phi(x_t) - \nabla_{x_t}\ln p_\theta(x_t)$（式 (9)）。

## 3. 核心方法

### 3.1 三个分数恒等式

**恒等式 1（真实数据的 Tweedie 公式）**：对半隐式分布 $p_{\text{data}}(x_t)$，由贝叶斯规则与 Tweedie 公式有

$$\mathbb{E}[x_0 | x_t] = x_t + \sigma_t^2\, \nabla_{x_t}\ln p_{\text{data}}(x_t) \tag{4}$$

**恒等式 2（生成数据的 Tweedie 公式）**：对 $p_\theta(x_t)$ 同理

$$\mathbb{E}[x_g | x_t] = x_t + \sigma_t^2\, \nabla_{x_t}\ln p_\theta(x_t) \tag{5}$$

**恒等式 3（分数投影恒等式）**：由于 $\nabla_{x_t}\ln p_\theta(x_t)$ 不可解析计算，引入投影向量技巧

$$\mathbb{E}_{x_t\sim p_\theta(x_t)}\big[u^T(x_t)\nabla_{x_t}\ln p_\theta(x_t)\big] = \mathbb{E}_{(x_t,x_g)\sim q(x_t|x_g)p_\theta(x_g)}\big[u^T(x_t)\nabla_{x_t}\ln q(x_t|x_g)\big]$$

正是这个恒等式把显式分数匹配（ESM）与去噪分数匹配（DSM）联系起来（Vincent, 2011）。SiD 的创新在于把这三个恒等式**融合**成一个损失。

### 3.2 MESM 损失（Fisher 散度）

理论上，对某个反向扩散时间步 $t$，蒸馏损失定义为分数差的期望 L2 范数：

$$L_\theta = \mathbb{E}_{x_t\sim p_\theta(x_t)}\big[\|\delta_{\phi,\theta}(x_t)\|_2^2\big],\quad \delta_{\phi,\theta}(x_t) := S_\phi(x_t) - \nabla_{x_t}\ln p_\theta(x_t) \tag{8,9}$$

这就是**模型基显式分数匹配（MESM）**，即 Fisher 散度。与 ESM 的区别在于期望是相对**生成分布 $p_\theta(x_t)$** 而非真实分布取的。

一个常被默认的假设是"学生蒸馏的性能被教师的反向采样限制"，但 SiD 用结果反驳了这一直觉——单步学生可反超迭代教师，说明**反向采样本身会累积误差**，即便用很细的步与高级求解器。

### 3.3 两种损失近似及其失效

- **近似 L⁽¹⁾**：把 $\nabla\ln p_\theta$ 用可训练网络 $f_\psi$ 近似（忽略其对 $\theta$ 的依赖），得 $L^{(1)}_\theta = \mathbb{E}_{x_t\sim p_\theta}\|\delta_{\phi,\psi}(x_t)\|^2$。但 Proposition 4 的玩具例证明它在训练初期（$p_\theta$ 与 $p_{\text{data}}$ 不匹配时）**不产生有意义的梯度**。
- **近似 L⁽²⁾（投影分数匹配）**：由恒等式 3，MESM 可等价写成

$$L_\theta = \mathbb{E}_{q(x_t|x_g,t)p_\theta(x_g)}\big[\sigma_t^{-2}\,\delta_{\phi,\psi^*(\theta)}(x_t)^T\big(f_\phi(x_t,t) - x_g\big)\big] \tag{19}$$

近似后 $L^{(2)}_\theta$ **既不受 $\|\Delta_{\psi,\psi^*(\theta)}\|^2$ 支配、也不受分数差 $\delta_{\phi,\psi^*(\theta)}$ 支配**，其中 $f_\phi(x_t,t)-x_g$ 反映"预训练模型对损坏假图的去噪能力"，相对稳定（Proposition 6 证明它能提供有用梯度）。

### 3.4 SiD 融合损失

经验上 $L^{(1)}$ 完全失效、$L^{(2)}$ 好但 FID 差（图偏暗、色彩失配），作者假设差值项 $L^\Delta_\theta = L^{(2)}_\theta - L^{(1)}_\theta$ 才是指向正确方向的梯度，于是考虑 $L^{(2)}_\theta - \alpha L^{(1)}_\theta = (1-\alpha)L^{(1)}_\theta + L^\Delta_\theta$，最终加权损失为：

$$\tilde{L}_\theta(x_t,t,\phi,\psi) = (1-\alpha)\frac{\omega(t)}{\sigma_t^4}\|f_\phi - f_\psi\|_2^2 + \frac{\omega(t)}{\sigma_t^4}\big(f_\phi(x_t,t) - f_\psi(x_t,t)\big)^T\big(f_\psi(x_t,t) - x_g\big) \tag{23}$$

关键区别：SiD 要**反传通过两个分数网络的梯度**（分数梯度）：

$$\nabla_\theta f_\phi(x_t,t) = \frac{\partial f_\phi(x_t,t)}{\partial x_t}\nabla_\theta G_\theta(\sigma_{\text{init}}z),\qquad \nabla_\theta f_\psi(x_t,t) = \frac{\partial f_\psi(x_t,t)}{\partial x_t}\nabla_\theta G_\theta(\sigma_{\text{init}}z) \tag{24}$$

这与 Diff-Instruct / DMD 不同——它们不使用分数梯度 $\partial f/\partial x_t$。

### 3.5 交替更新与噪声调度

SiD 交替更新 $\psi$（按式 (10) 的去噪目标，给定 $\theta$）与 $\theta$（按式 (23)，给定 $\psi$）。权重 $\omega(t)=C\sigma_t^4/\|x_g - f_\phi(x_t,t)\|_{1,\text{sg}}$（式 (25)），噪声级别按 EDM 的 $\sigma_t$ 调度（式 (26)，$\sigma_{\min}=0.002,\ \sigma_{\max}=80,\ \rho=7.0$）。

## 4. 数据集与实验设置

### 4.1 数据集

沿用 EDM 的四个基准：**CIFAR-10 32×32**（cond/uncond）、**ImageNet 64×64**（cond）、**FFHQ 64×64**、**AFHQ-v2 64×64**，教师均为 VP-EDM checkpoint。

### 4.2 评价指标

FID（50k 样本）与 Inception Score（IS）；ImageNet 额外报告 Precision/Recall。

### 4.3 实现细节

- 生成器 $G_\theta$ 与分数网络 $f_\psi$ 均从教师 $f_\phi$ 复制架构与参数初始化。
- 优化器 Adam；$f_\psi$ 用 $\beta_1=0$（否则不收敛），$G_\theta$ 除 FFHQ（$\beta_1=0.9$）外用 $\beta_1=0$。
- 训练规模：多数基准处理 5 亿张合成图，CIFAR-cond（$\alpha=1.2$）8 亿、ImageNet-64（$\alpha=1.2$）10 亿。
- ImageNet batch 由 1024 增到 8192（FP16 在 1024 时发散，疑似精度问题）。

## 5. 实验结果

### 5.1 主要定量结果

- **CIFAR-10 uncond（Table 1）**：SiD $\alpha=1.2$ FID **1.923** / IS 9.980，反超教师 VP-EDM（35 步，1.97），超过 CTM（1.98）、DMD（3.77）、Diff-Instruct（4.53）。
- **CIFAR-10 cond（Table 2）**：FID **1.710**（$\alpha=1.2$），反超教师（1.79）。
- **ImageNet 64×64（Table 3）**：FID **1.524**（$\alpha=1.2$），反超教师 VP-EDM 511 步（1.36 的 ODE 结果是 1.36？见表 3 教师 VP-EDM 511 步 FID 1.36、Prec 0.71），仅 RIN（1.23，1000 步）更优。
- **FFHQ / AFHQ（Table 4/5）**：FID 1.550 / 1.711（$\alpha=1.2$），均反超教师（2.39 / 1.96）。

### 5.2 收敛速度：指数级 FID 下降

![](assets/SiD%20-%20分数恒等蒸馏一步生成/figures/02-fid-evolution-cifar.png)

> 图 2：CIFAR-10（uncond）上 FID 随处理图像数的演化。横轴为迭代的千张图像数、纵轴为 FID（对数尺度），紫色/绿色/红色横线分别是 EDM、DMD、Diff-Instruct。SiD（$\alpha=1.0/1.2$）呈现近似的"对数-对数线性"，说明 FID 随蒸馏**指数下降**，且更快越过 DMD 与 Diff-Instruct 的水平线。
> 来源：论文 Figure 2，第 7 页，https://arxiv.org/abs/2404.04057

作者强调：SiD 只需远少于 5 亿张合成图即可反超 Diff-Instruct 与 DMD——例如 ImageNet 上不到 4 千万张即超 DMD、不到 7 百万张即超 Diff-Instruct；CIFAR-uncond 上不到 2 千万张即超二者（16×A100 上不到 10 小时）。

### 5.3 定性结果

![](assets/SiD%20-%20分数恒等蒸馏一步生成/figures/03-imagenet-samples.png)

> 图 13：SiD 单步生成器在 ImageNet 64×64 上的随机样本（对应 FID 1.524）。展示数据无关蒸馏也能产出高真实感、多类别的图像。
> 来源：论文 Figure 13（附录），https://arxiv.org/abs/2404.04057

### 5.4 消融实验

- **$\alpha$ 取值（Fig.7/8）**：$\alpha\in[0.75,1.2]$ 效果最佳，$\alpha$ 为负时 FID 很大；最终取 $\alpha\in\{1.0,1.2\}$。
- **ImageNet batch 敏感性**：batch 1024 时 FID 指数下降至接近 DMD 的 2.62 后突然发散（疑似 FP16），换 8192 + lr 降到 4e-6 后稳定。

## 6. 与相关工作的关系

- **相对 Diffusion-GAN / Diff-Instruct / DMD**：同属"对齐扩散真实与假分布"的框架，但 SiD 改用**Fisher 散度（模型基分数匹配）**替代 JSD / KL，并显式使用**分数梯度**。
- **相对一致性模型 / 渐进蒸馏**：这些依赖真实数据或回归配对；SiD 全程数据无关、无需反向采样。
- **相对 VSD / Score Distillation（文生 3D）**：SiD 把分数恒等式思想系统化成一个单步图像生成器的训练损失。

## 7. 局限与批判性评价

1. **三网络显存开销**：同时维护 $f_\phi$、$f_\psi$、$f_\theta$（同尺寸），比传统扩散训练更吃显存；作者建议用 LoRA 缩小 $f_\psi$、$f_\theta$。
2. **计算量增加约 1/3**：因需反传分数梯度（式 (24)），每迭代比 Diff-Instruct 慢约 28–36%。
3. **FP16 数值敏感**：ImageNet batch 1024 时发散，需加大 batch 与降低 lr，收敛更慢但最终 FID 更低。
4. **ImageNet 上未全面反超**：SiD 1.524 仍略逊于 RIN（1.23，1000 步），是唯一未反超教师的基准。

## 8. 复现与实践建议

- 官方代码：https://github.com/mingyuanzhou/SiD（基于 EDM 代码库）。
- 关键超参：$\alpha\in\{1.0,1.2\}$、$\sigma_{\text{init}}=2.5$、$t_{\max}=800$、$f_\psi$ 用 $\beta_1=0$。
- 显存优化方向：对 $f_\psi$、$f_\theta$ 用 LoRA（作者列为 future work）。

## 9. 个人启发与后续问题

SiD 的"数据无关"是这条路线里极具冲击力的一个性质：它证明**蒸馏可以完全不看真实数据**，仅靠"真实分数（教师）+ 生成分数（自训练 critic）"的对齐就逼近教师。这个性质在因果视频蒸馏里意义重大——视频教师（如 Wan2.1、LTX-2）往往不开源训练数据，SiD 式的 data-free 目标让"蒸馏一个闭源/受限教师"成为可能。

更关键的是作者脉络：Mingyuan Zhou 是 SiD 与 Self-Forcing 的共同作者，因此 SiD 的 MESM/Fisher 散度损失就是 Self-Forcing 在视频上采用"video-level SiD 目标"的直接来源——两条源流（Tianwei Yin 的 DMD/DMD2/CausVid 与 Mingyuan Zhou 的 SiD/Self-Forcing）正是在"因果视频蒸馏"这个交汇点上合流。

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | arXiv 2404.04057v3 | 全文 | 2026-08-29 | 精读主来源 |
| `figures/01-distillation-progression.png` | 蒸馏进程图 | arXiv HTML 渲染 | Figure 1, p.2 | 2026-08-29 | 第 1 章"研究背景" |
| `figures/02-fid-evolution-cifar.png` | FID 演化图 | arXiv HTML 渲染 | Figure 2, p.7 | 2026-08-29 | 第 5.2 节"收敛速度" |
| `figures/03-imagenet-samples.png` | 定性样本图 | arXiv HTML 渲染 | Figure 13（附录） | 2026-08-29 | 第 5.3 节"定性结果" |
