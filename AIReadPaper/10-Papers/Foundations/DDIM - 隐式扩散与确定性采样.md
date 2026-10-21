---
title: DDIM - 隐式扩散与确定性采样
aliases:
  - Denoising Diffusion Implicit Models
  - DDIM
  - 去噪扩散隐式模型
tags:
  - paper
  - generative-modeling
  - diffusion-model
  - implicit-model
  - deterministic-sampling
  - neural-ode
arxiv: "2010.02502"
year: 2020
venue: ICLR 2021
status: 精读完成
---

# 0. DDIM：隐式扩散与确定性采样

> **论文**：Jiaming Song, Chenlin Meng, Stefano Ermon. *Denoising Diffusion Implicit Models*  
> **版本**：arXiv:2010.02502v4，2022-10-05；ICLR 2021 conference paper  
> **来源**：[arXiv 摘要页](https://arxiv.org/abs/2010.02502) · [官方 PDF](https://arxiv.org/pdf/2010.02502) · [[assets/DDIM - 隐式扩散与确定性采样/DDIM_arXiv_2010.02502.pdf|本地官方 PDF]]  
> **本地材料**：[[assets/DDIM - 隐式扩散与确定性采样/sources.md|来源与逐图核验记录]]  
> **精读日期**：2026-10-16

## 1. 一句话总结（TL;DR）

DDIM 的关键发现是：DDPM 的逐时刻噪声预测训练只依赖条件边际 $q(x_t\mid x_0)$，并不唯一决定整条联合前向过程；因此可以把 DDPM 的 Markov 前向扩散替换为一族具有相同边际的 non-Markovian 前向过程，复用同一个噪声预测网络，却得到不同的反向生成过程。把反向方差设为零时，采样从“每步重新加随机噪声”变成由初始 $x_T$ 唯一决定的确定性隐式生成；再只访问训练时间网格的子序列，就能用 20–100 步达到接近 1000 步模型的质量。

更深一层看，DDIM 不是单纯“把 DDPM 的噪声关掉”：它把**训练时学习的逐噪声级去噪场**与**推理时选择的概率路径/数值轨迹**解耦。确定性 DDIM 可解释为一个 ODE 的 Euler 离散化，这解释了其轨迹一致性、潜变量插值和近似可逆重建，也直接启发了后来的扩散 ODE 求解器与 inversion/editing 工作。

## 2. 研究背景与问题设定

### 2.1 DDPM 的质量优势与序列采样瓶颈

DDPM 通过固定前向扩散把数据逐步变成高斯噪声，再训练神经网络近似反向条件。它不依赖对抗训练，生成质量在 2020 年已能接近 GAN；代价是一次样本通常要顺序调用去噪网络 $T=1000$ 次。论文给出的直观量级是：单张 GPU 上生成 50k 张 $32\times32$ 图像约需 20 小时，而 GAN 可在一分钟内完成；$256\times256$ 的 50k 张采样甚至可接近 1000 小时。

这种慢并非网络不能跨大步，而是标准推导把生成过程绑定为某个细粒度 Markov 扩散的反向近似：前向有多少步，朴素反向就走多少步。DDIM 追问的是：**训练目标真的唯一绑定了这条 Markov 联合路径吗？**

### 2.2 DDPM 的基本记号

论文采用一个容易与后续文献混淆的记号：其 $\alpha_t$ 对应 Ho et al. (2020) 常写的累计乘积 $\bar\alpha_t$，而不是单步 $\alpha_t=1-\beta_t$。本文以下沿用 DDIM 原论文记号。

标准 DDPM 的前向 Markov 链为：

$$q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),\qquad q(x_t\mid x_{t-1})=\mathcal N\!\left(\sqrt{\frac{\alpha_t}{\alpha_{t-1}}}x_{t-1},\left(1-\frac{\alpha_t}{\alpha_{t-1}}\right)I\right).$$

它有可直接采样的条件边际：

$$q(x_t\mid x_0)=\mathcal N\!\left(\sqrt{\alpha_t}x_0,(1-\alpha_t)I\right).$$

因此训练时无需真的从 $x_0$ 模拟到第 $t$ 步，只需一次重参数化：

$$x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

噪声预测目标写成：

$$L_\gamma(\epsilon_\theta)=\sum_{t=1}^{T}\gamma_t\,\mathbb E_{x_0,\epsilon}\left[\left\|\epsilon_\theta^{(t)}\!\left(\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon\right)-\epsilon\right\|_2^2\right].$$

Ho et al. 实际使用的简化目标令 $\gamma_t=1$。DDIM 的全部出发点就在于：这个 loss 每一项只看 $(x_0,x_t)$，也就是 $q(x_t\mid x_0)$；它没有直接观察不同时间变量之间如何联合耦合。

### 2.3 论文要解决的三个问题

1. **联合分布是否唯一？** 保持每个 $q(x_t\mid x_0)$ 不变，能否构造不同于 Markov diffusion 的联合前向过程？
2. **训练能否复用？** 新联合过程对应的变分目标，能否仍由已有 DDPM 的噪声预测网络优化？
3. **采样能否缩短并确定化？** 能否选择一个反向过程，让采样不再需要每步注入随机噪声，并允许跳过大量时间点？

## 3. 核心贡献

1. **提出同边际、非 Markov 的前向过程族。** 论文显式构造 $q_\sigma(x_{1:T}\mid x_0)$：所有时刻的 $q_\sigma(x_t\mid x_0)$ 与 DDPM 完全相同，但跨时刻联合依赖由 $\sigma$ 控制。
2. **统一新过程与 DDPM 训练目标。** 证明对 $\sigma>0$，新变分目标 $J_\sigma$ 等于某个加权噪声预测目标 $L_\gamma$ 加常数；在论文说明的理想条件下，常用的 $L_1$ 可作为所有这些过程的共享 surrogate。
3. **导出一族可调随机性的生成过程。** 单步更新由“预测的 $x_0$、指向 $x_t$ 的方向、随机噪声”三部分组成；$\eta$ 或 $\sigma_t$ 连续控制 DDPM 式随机采样与确定性 DDIM 之间的过渡。
4. **通过时间子序列实现训练后跳步。** 训练仍可用 $T=1000$ 个噪声级，推理只访问长度为 $S\ll T$ 的子序列 $\tau$，无需重训。
5. **建立 ODE 视角并展示新能力。** 确定性 DDIM 是特定 ODE 的 Euler 离散化；同一初始噪声跨步数保持高层语义一致，并支持 latent 插值与近似反向编码/重建。

## 4. 同边际的 non-Markovian 前向过程

### 4.1 图模型：变的是联合耦合，不是单时刻边际

![](assets/DDIM%20-%20隐式扩散与确定性采样/fig1_graphical_models.png)

> **图 1 解读**：左侧 DDPM 前向过程是 $x_0\to x_1\to\cdots\to x_T$ 的 Markov 链；右侧 DDIM 构造的 inference joint 在给定 $x_0$ 时允许 $x_{t-1}$ 同时依赖 $x_t$ 与 $x_0$。两者的每个 $q(x_t\mid x_0)$ 相同，但完整 joint 不同。来源：原论文 Figure 1，第 2 页。

论文定义由非负向量 $\sigma=(\sigma_1,\ldots,\sigma_T)$ 索引的一族 inference distributions：

$$q_\sigma(x_{1:T}\mid x_0)=q_\sigma(x_T\mid x_0)\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,x_0).$$

终点条件仍为：

$$q_\sigma(x_T\mid x_0)=\mathcal N\!\left(\sqrt{\alpha_T}x_0,(1-\alpha_T)I\right).$$

对 $t>1$，反向写法的条件分布被定义为：

$$q_\sigma(x_{t-1}\mid x_t,x_0)=\mathcal N\!\left(\sqrt{\alpha_{t-1}}x_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\,\frac{x_t-\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}},\,\sigma_t^2I\right).$$

这一定义乍看方向“反了”：它先从 $x_T$ 往 $x_1$ 分解 inference joint，而后可用 Bayes rule 写出对应的前向条件 $q_\sigma(x_t\mid x_{t-1},x_0)$。关键不是采样方向，而是这个 joint 满足论文 Lemma 1：

$$q_\sigma(x_t\mid x_0)=\mathcal N\!\left(\sqrt{\alpha_t}x_0,(1-\alpha_t)I\right),\qquad \forall t.$$

证明直觉很简单：条件均值中的第二项携带从 $x_t$ 归一化出的噪声方向，其方差是 $1-\alpha_{t-1}-\sigma_t^2$；再叠加条件方差 $\sigma_t^2$，总方差恰好回到 $1-\alpha_{t-1}$。由 $t=T$ 向下归纳，每个单时刻边际都与 DDPM 一致。

### 4.2 为什么它是 non-Markovian

从 Bayes rule 得到的前向条件为：

$$q_\sigma(x_t\mid x_{t-1},x_0)=\frac{q_\sigma(x_{t-1}\mid x_t,x_0)q_\sigma(x_t\mid x_0)}{q_\sigma(x_{t-1}\mid x_0)}.$$

它一般显式依赖 $x_0$，所以给定 $x_{t-1}$ 后，$x_t$ 仍未与 $x_0$ 条件独立；这违反一阶 Markov 性。要严格区分：

- **相同的是条件边际** $q_\sigma(x_t\mid x_0)$，因此训练时任意单个噪声级的样本分布相同；
- **不同的是联合分布** $q_\sigma(x_{1:T}\mid x_0)$，因此跨时间变量如何相关、对应的反向生成轨迹如何走，可以不同；
- “non-Markovian forward process” 是论文对 inference joint 的称呼，并不表示训练时需要模拟一条带记忆的长轨迹。

### 4.3 $\sigma_t$ 的含义与两个端点

$\sigma_t$ 是在已知 $(x_t,x_0)$ 时，$x_{t-1}$ 仍保留的条件随机性：

- 取 DDPM 对应的方差时，恢复原来的 Markov diffusion 与 DDPM 反向过程；
- 当 $\sigma_t\to0$ 时，给定 $(x_t,x_0)$ 后，$x_{t-1}$ 变成确定值，跨时间噪声被完全耦合；
- $\sigma_t=0$ 的极限不在 Theorem 1 的 $\sigma>0$ 严格条件内，论文通过“任意小正方差的极限”解释它，并直接定义对应生成器。

所以 DDIM 的“implicit”不是说模型没有概率，也不是把 $x_T$ 固定成常数；它表示从随机 latent $x_T\sim\mathcal N(0,I)$ 经一个确定性映射生成 $x_0$，通常不显式给出输出密度。

## 5. 为什么可以沿用 DDPM 的训练目标

### 5.1 从噪声预测得到 $x_0$ 预测

给定带噪样本 $x_t$，网络预测其噪声 $\epsilon_\theta^{(t)}(x_t)$，于是可反解出 clean sample 的估计：

$$f_\theta^{(t)}(x_t)=\frac{x_t-\sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}.$$

生成过程把真实但未知的 $x_0$ 替换为 $f_\theta^{(t)}(x_t)$，再套用上一节的 $q_\sigma(x_{t-1}\mid x_t,x_0)$：

$$p_\theta^{(t)}(x_{t-1}\mid x_t)=q_\sigma\!\left(x_{t-1}\mid x_t,f_\theta^{(t)}(x_t)\right),\qquad t>1.$$

### 5.2 统一变分目标

论文对新 joint 定义负 ELBO 形式的目标：

$$J_\sigma(\epsilon_\theta)=\mathbb E_{q_\sigma(x_{0:T})}\left[\log q_\sigma(x_{1:T}\mid x_0)-\log p_\theta(x_{0:T})\right].$$

Theorem 1 证明：对所有严格正的 $\sigma$，存在正权重 $\gamma$ 和与模型参数无关的常数 $C$，使得：

$$J_\sigma(\epsilon_\theta)=L_\gamma(\epsilon_\theta)+C.$$

附录给出的权重为：

$$\gamma_t=\frac{1}{2d\sigma_t^2\alpha_t},$$

其中 $d$ 是 $x_0$ 的维数。其核心推导是：同方差高斯之间的 KL 退化为均值的平方误差，而 $x_0-f_\theta^{(t)}(x_t)$ 又与 $\epsilon-\epsilon_\theta^{(t)}(x_t)$ 成比例。于是改变 joint 与反向方差，最终只改变各时间步 denoising loss 的权重。

### 5.3 “与 DDPM 相同训练目标”的精确边界

“DDIM 与 DDPM 完全相同训练”常被说得过强。论文的逻辑分三层：

1. **严格等式**：每个 $J_\sigma$ 等于某个带权 $L_\gamma$ 加常数；这不是说所有 $\sigma$ 对应同一个数值目标。
2. **相同最优解的理想论证**：若每个时间步的 $\epsilon_\theta^{(t)}$ 参数互不共享，则正权重只缩放彼此独立的子问题，不改变各自全局最优解，因此 $L_1$ 与所有 $L_\gamma$ 共享最优解。
3. **实际神经网络**：真实 U-Net 跨 $t$ 共享绝大多数参数，有限容量、有限数据和非凸优化下，不同权重未必严格给出同一参数最优点。论文仍把 $L_1$ 当作实用 surrogate，并用预训练 DDPM checkpoint 验证无需重训即可工作。

因此最稳妥的结论是：**DDIM 复用 DDPM 的噪声条件训练信号和 checkpoint；理论上同边际保证监督样本一致，变分目标差异表现为时间加权，严格“同最优解”还带有参数不共享/理想优化条件。**

## 6. 生成过程：从 DDPM 到确定性 DDIM

### 6.1 通用单步更新的三部分

从 $x_t$ 生成 $x_{t-1}$ 的更新为：

$$x_{t-1}=\sqrt{\alpha_{t-1}}\underbrace{\left(\frac{x_t-\sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}\right)}_{\text{predicted }x_0}+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta^{(t)}(x_t)}_{\text{direction pointing to }x_t}+\underbrace{\sigma_t\epsilon_t}_{\text{random noise}},\quad \epsilon_t\sim\mathcal N(0,I).$$

三项分别承担不同作用：

1. **预测数据项**：把网络预测的 clean sample 缩放到时刻 $t-1$；
2. **保留噪声方向项**：沿网络认为与当前 $x_t$ 一致的噪声方向移动；
3. **随机噪声项**：决定每一步是否产生新的随机分叉。

所有这些采样器调用的是同一个 $\epsilon_\theta$。改变 $\sigma_t$ 不改变模型参数，只改变如何积分这个去噪场。

### 6.2 DDPM 端点

当：

$$\sigma_t=\sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}\sqrt{1-\frac{\alpha_t}{\alpha_{t-1}}},$$

相应前向过程退化为 Markov diffusion，生成过程对应 DDPM。此时每一步都会注入新的独立高斯噪声；即使起始 $x_T$ 相同，不同随机种子仍可能得到不同 $x_0$。

### 6.3 DDIM 端点：$\sigma_t=0$

当所有 $\sigma_t=0$ 时：

$$x_{t-1}=\sqrt{\alpha_{t-1}}\hat x_0(x_t,t)+\sqrt{1-\alpha_{t-1}}\,\epsilon_\theta^{(t)}(x_t).$$

这里不再有每步随机噪声。一旦初始 $x_T$、时间轨迹和模型确定，输出 $x_0$ 就唯一确定。仍需注意两点：

- 整个生成模型仍是随机的，因为 $x_T\sim\mathcal N(0,I)$；“确定性”只指条件于初始 latent 后的轨迹；
- DDIM 不是把 DDPM posterior variance 粗暴设零而保持其余推导不变，而是由一个不同的同边际 non-Markovian inference joint 导出的合法生成过程。

### 6.4 用 $\eta$ 连续控制随机性

为便于实验，论文把子序列上的方差写成：

$$\sigma_{\tau_i}(\eta)=\eta\sqrt{\frac{1-\alpha_{\tau_{i-1}}}{1-\alpha_{\tau_i}}}\sqrt{1-\frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}.$$

其中：

- $\eta=0$：确定性 DDIM；
- $\eta=1$：对应 DDPM 形式的方差；
- $0<\eta<1$：在确定性与随机性之间插值；
- 论文还比较了 Ho et al. CIFAR-10 实现采用的更大噪声标准差 $\hat\sigma$，它在短轨迹上明显更差。

工程库常把 `eta` 暴露为 sampler 参数。它控制的是**采样路径的随机注入量**，不是 classifier-free guidance 强度，也不是训练噪声 schedule。

## 7. 跳步：在训练网格上选择短轨迹

### 7.1 子序列 $\tau$

![](assets/DDIM%20-%20隐式扩散与确定性采样/fig2_accelerated_generation.png)

> **图 2 解读**：训练可定义在完整噪声网格上，但生成只访问子序列 $\tau=[1,3]$，直接从 $x_3$ 跳到 $x_1$，跳过灰色的 $x_2$。保留的节点仍使用相同 $q(x_{\tau_i}\mid x_0)$ 边际与同一个去噪器。来源：原论文 Figure 2，第 5 页。

令 $\tau=(\tau_1,\ldots,\tau_S)$ 是 $\{1,\ldots,T\}$ 的递增子序列，且 $\tau_S=T$。生成时按反序只访问：

$$x_T=x_{\tau_S}\to x_{\tau_{S-1}}\to\cdots\to x_{\tau_1}\to x_0.$$

把通用更新中的 $(t,t-1)$ 换成 $(\tau_i,\tau_{i-1})$ 即可跨步：

$$x_{\tau_{i-1}}=\sqrt{\alpha_{\tau_{i-1}}}\hat x_0(x_{\tau_i},\tau_i)+\sqrt{1-\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2}\,\epsilon_\theta^{(\tau_i)}(x_{\tau_i})+\sigma_{\tau_i}\epsilon.$$

关键原因仍是：训练网络学的是每个噪声级上的 denoising function，不是只能执行相邻一步的固定 transition。只要在被访问的 $\tau_i$ 上能评估网络，就可选择更粗的离散化。

### 7.2 计算—质量权衡

采样步数从 $T$ 减到 $S$ 后，网络调用量近似按 $S/T$ 缩放。代价是离散化误差变大，而且网络在大跨度更新中的预测误差被更强地放大。论文使用两类简单 schedule：

- **Linear**：$\tau_i=\lfloor ci\rfloor$；CelebA 与 LSUN 使用；
- **Quadratic**：$\tau_i=\lfloor ci^2\rfloor$；CIFAR-10 使用；
- 常数 $c$ 被选到使末端接近 $T$，论文按数据集选用验证效果稍好的方案。

这不是自适应 ODE solver，也不是训练一个少步蒸馏模型；它只是对已有 1000 步模型做训练后时间网格抽稀。

## 8. ODE 关系：确定性轨迹为何可逆

### 8.1 从 DDIM 更新到 Euler 积分

对确定性 DDIM，把更新改写为：

$$\frac{x_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}}=\frac{x_t}{\sqrt{\alpha_t}}+\left(\sqrt{\frac{1-\alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}}-\sqrt{\frac{1-\alpha_t}{\alpha_t}}\right)\epsilon_\theta^{(t)}(x_t).$$

引入重参数化：

$$\bar x(t)=\frac{x(t)}{\sqrt{\alpha(t)}},\qquad \sigma(t)=\sqrt{\frac{1-\alpha(t)}{\alpha(t)}},\qquad x(t)=\frac{\bar x(t)}{\sqrt{1+\sigma(t)^2}}.$$

则连续极限对应：

$$d\bar x(t)=\epsilon_\theta^{(t)}\!\left(\frac{\bar x(t)}{\sqrt{1+\sigma(t)^2}}\right)d\sigma(t).$$

DDIM 的离散更新可看作相对于噪声尺度 $\sigma$ 的 Euler 步。由此得到三个直觉：

1. 固定 $x_T$ 后，确定性 ODE 轨迹只有一条；
2. 改变步数是对同一连续动力系统使用不同粗细的离散网格，因此高层语义可相近；
3. 理论上可以把 ODE 从数据端反向积分到噪声端，获得与样本对应的 latent encoding。

### 8.2 与 probability flow ODE 的关系

论文 Proposition 1 指出：当 $\epsilon_\theta$ 最优时，上述 ODE 等价于 Song et al. (2020) 中 VE-SDE 的一个 probability flow ODE。利用 denoising score matching 关系：

$$\nabla_{\bar x}\log p_t(\bar x)=-\frac{1}{\sigma(t)}\epsilon_\theta^{(t)}\!\left(\frac{\bar x}{\sqrt{1+\sigma(t)^2}}\right),$$

可把 probability flow ODE 化到相同连续形式。

但论文明确提醒：**连续 ODE 等价不等于有限步采样器完全相同。** DDIM 更新相当于对 $d\sigma$ 做 Euler 步，而同期 probability-flow 实现按 $dt$ 离散；当相邻时间点足够近时两者趋同，少步时会产生实质差异。这个观察后来成为专门设计高阶、低误差 diffusion ODE solver 的起点之一。

## 9. 潜空间一致性、插值与重建

### 9.1 同一 $x_T$ 的轨迹一致性

![](assets/DDIM%20-%20隐式扩散与确定性采样/fig5_trajectory_consistency.png)

> **图 5 解读**：每一列固定同一个随机 $x_T$，只改变 DDIM 采样步数。10、20、50、100、1000 步的细节质量不同，但物体类别、姿态、布局等高层特征通常保持一致，符合“同一连续轨迹的不同离散精度”解释。来源：原论文 Figure 5，第 8 页。

在 DDPM 中，即便 $x_T$ 固定，后续每步噪声也会改变样本；在 $\eta=0$ 的 DDIM 中，$x_T$ 基本承担完整随机 code 的角色。实验显示 20 步输出已常与 1000 步输出保持相似高层语义，更多步主要改善细节和质量。

这里的“一致”不是数学上的像素完全相同：有限步离散误差、网络误差和不同时间子序列都会改变终点。论文只通过可视化论证 high-level feature consistency，没有给出感知距离或语义一致率等定量指标。

### 9.2 latent 球面插值

![](assets/DDIM%20-%20隐式扩散与确定性采样/fig6_latent_interpolation.png)

> **图 6 解读**：对两个高斯 latent 做 spherical linear interpolation（slerp），再用 50 步 DDIM 解码，人物身份/姿态、卧室布局和教堂结构出现连续语义过渡。来源：原论文 Figure 6，第 8 页。

给定两个初始 latent $x_T^{(0)}$ 与 $x_T^{(1)}$，论文用 slerp：

$$x_T^{(a)}=\frac{\sin((1-a)\theta)}{\sin\theta}x_T^{(0)}+\frac{\sin(a\theta)}{\sin\theta}x_T^{(1)},\qquad \theta=\arccos\frac{{x_T^{(0)}}^\top x_T^{(1)}}{\|x_T^{(0)}\|\|x_T^{(1)}\|}.$$

slerp 比普通线性插值更贴合高维高斯样本集中在近似球壳上的几何。因为 DDIM 从 latent 到图像是确定映射，连续 latent 路径对应连续输出路径。论文对 DDPM 的对比边界是：若只插值起始 $x_T$，后续随机噪声会掩盖对应关系；它并未证明任何随机扩散模型都无法通过耦合整条噪声序列实现插值。

### 9.3 原论文的 encoding/reconstruction

论文把 DDIM ODE 从 $x_0$ 反向积分到 $x_T$，再从该 $x_T$ 正向生成回 $x_0$。CIFAR-10 测试集上的逐维 MSE（图像缩放到 $[0,1]$）如下：

![](assets/DDIM%20-%20隐式扩散与确定性采样/table2_reconstruction_error.png)

> **表 2 解读**：编码和解码均使用 $S$ 步；$S$ 从 10 增至 500 时误差从 0.014 降至 0.0001，说明主要误差来自有限步数值离散与模型近似，而非每步随机性。来源：原论文 Table 2，第 9 页。

这表明确定性轨迹具有 Neural ODE / invertible flow 式的近似可逆性。不过论文脚注明确说 $x_T$ 与 $x_0$ 维度相同，因此它没有主张压缩；也没有报告 exact likelihood 或严格双射保证。

### 9.4 必须区分：原论文重建 vs. 后世“DDIM inversion”

> [!important] 概念边界
> 原论文只证明并实验了：把**无条件 CIFAR-10 DDIM 的连续 ODE 近似反向积分**，可把观察 $x_0$ 编码到某个 $x_T$，再解码重建。它没有提出后来文本到图像编辑文献中的完整 inversion pipeline。

后世常说的 “DDIM inversion” 通常是把确定性采样公式离散地反向使用，从真实图像或其 VAE latent 估计一串 $x_t$，再在文本条件下重新生成/编辑。两者相关，但不可混为一谈：

- 原论文没有 Stable Diffusion、VAE latent diffusion 或 classifier-free guidance；
- 原论文没有 prompt 条件切换、attention control、Prompt-to-Prompt 或 Null-text Inversion；
- 原论文没有优化 unconditional text embedding，也没有声称对强 guidance 下的离散采样严格可逆；
- 后世 inversion 的误差还来自 VAE 编解码、条件分支、CFG 放大、离散轨迹不对称与 prompt mismatch；这些不属于本文实验结论；
- 把论文的 ODE 可逆性当作后世所有 “DDIM inversion” 方法都能精确重建的证明，是过度外推。

准确的历史表述应是：**本文提供了确定性 diffusion trajectory、ODE 解释和反向编码实验；后续工作在此基础上发展出面向条件扩散编辑的 DDIM inversion 及其误差修正技术。**

## 10. 实验设置

### 10.1 数据集、模型与指标

论文覆盖四个图像数据集：

| 数据集 | 分辨率 | 任务/备注 |
|---|---:|---|
| CIFAR-10 | $32\times32$ | 无条件生成 |
| CelebA | $64\times64$ | 人脸生成；作者自行训练模型 |
| LSUN Bedroom | $256\times256$ | 使用 DDPM 官方 checkpoint |
| LSUN Church | $256\times256$ | 使用 DDPM 官方 checkpoint |

CIFAR-10、Bedroom、Church 直接使用 Ho et al. 的预训练 DDPM；CelebA 使用相同 $L_1$ denoising objective 训练。网络是基于 Wide ResNet 的 U-Net。所有模型训练网格均为 $T=1000$，实验只改变：

- 采样子序列长度 $S=\dim(\tau)$；
- 随机性参数 $\eta$ / $\sigma$；
- 时间子序列采用 linear 或 quadratic schedule。

主要定量指标是 FID；另报告生成 50k 图像的 wall-clock time 和重建 MSE。

### 10.2 核心对照设计

实验不是比较不同训练模型，而是尽可能固定同一 denoiser，只换采样过程。这一点非常重要：论文要证明的是 **post-training sampler choice** 的价值，而不是更大网络或新训练 recipe 的收益。

比较项包括：

- DDIM：$\eta=0$；
- 部分随机：$\eta=0.2,0.5$；
- DDPM 对应方差：$\eta=1$；
- 更大方差实现：$\hat\sigma$；
- 不同 $S\in\{10,20,50,100,1000\}$。

## 11. 实验结果与消融

### 11.1 FID：短轨迹下确定性更占优

![](assets/DDIM%20-%20隐式扩散与确定性采样/table1_fig3_fid_samples.png)

> **表 1 / 图 3 解读**：同一 denoiser、同一步数下，$\eta$ 越小通常越适合短轨迹；尤其 $\hat\sigma$ 在 10–20 步时几乎退化为噪声。随着步数增多，各方法差距缩小，1000 步 $\hat\sigma$ 在两个数据集上略优于 DDIM。来源：原论文 Table 1 与 Figure 3，第 7 页。

关键数字：

| 数据集 | 步数 $S$ | DDIM $\eta=0$ | DDPM $\eta=1$ | $\hat\sigma$ |
|---|---:|---:|---:|---:|
| CIFAR-10 | 10 | **13.36** | 41.07 | 367.43 |
| CIFAR-10 | 20 | **6.84** | 18.36 | 133.37 |
| CIFAR-10 | 50 | **4.67** | 8.01 | 32.72 |
| CIFAR-10 | 100 | **4.16** | 5.78 | 9.99 |
| CIFAR-10 | 1000 | 4.04 | 4.73 | **3.17** |
| CelebA | 20 | **13.73** | 26.03 | 183.83 |
| CelebA | 100 | **6.53** | 13.93 | 45.20 |
| CelebA | 1000 | 3.51 | 5.98 | **3.26** |

结论不是“零随机性永远最好”，而是：**在网络调用预算很小、每一步跨度很大时，额外随机扰动显著破坏粗粒度轨迹；在 1000 步细网格上，DDPM 的 $\hat\sigma$ 可略优。**

LSUN 附录结果也支持这个趋势：

- Bedroom 20 步：DDIM 8.89 vs. DDPM 22.77；100 步：6.62 vs. 6.81；1000 步 DDPM 基准 6.36。
- Church 20 步：DDIM 12.47 vs. DDPM 23.37；100 步：10.58 vs. DDPM 8.27；1000 步 DDPM 基准 7.89。

Church 100 步时 DDPM 反而优于 DDIM，进一步说明最优随机性依数据集与步数而变，并非单调定律。

### 11.2 时间随步数近似线性

![](assets/DDIM%20-%20隐式扩散与确定性采样/fig4_sampling_efficiency.png)

> **图 4 解读**：在 Nvidia 2080 Ti 上生成 50k 张图像，耗时随网络调用步数近似线性；因此从 1000 步减到 20–100 步会直接带来数量级墙钟加速。来源：原论文 Figure 4，第 8 页。

论文报告 DDIM 在 20–100 步即可得到接近 1000 步模型的质量，对应约 **10×–50×** wall-clock speedup；更宽泛的实验描述提到相对原始 1000 步过程可达 10×–100× 加速。摘要中的主要结论采用较保守的 10×–50×。

“步数线性”不意味着所有现代实现都严格线性：batch size、显存带宽、kernel launch、编译优化和不同 timestep embedding 开销都会影响常数项；论文图只验证其硬件与实现上的近似关系。

### 11.3 $\eta$ 消融的真正含义

固定 $S$ 时，Table 1 从 $\eta=0,0.2,0.5,1$ 构成连续消融。CIFAR-10 在 $S=20$ 时 FID 依次为 6.84、7.11、8.35、18.36；CelebA 为 13.73、14.11、16.06、26.03。短轨迹中，增加随机性总体单调恶化。

这支持论文的机制解释：大步 Euler 式更新已产生较大离散误差，再叠加独立噪声会使样本偏离网络预测的 coherent trajectory。它不证明随机采样没有价值；随机性可能改善多样性、corrector 行为或其他质量指标，而本文主要以 FID 衡量。

### 11.4 时间子序列消融

论文比较 linear 与 quadratic timestep selection，并按数据集选择略优者：CIFAR-10 用 quadratic，其余用 linear。这个结果说明“同样 50 步”并非唯一采样器，节点在噪声尺度上的分配也会影响误差。

但正文没有给出完整 schedule 对照表或误差分析，因而无法从论文中得出普适的最优时间网格。后续专门的 diffusion solver 正是在改进这个粗糙的 Euler + 手工网格方案。

### 11.5 实验是否充分

实验对核心主张是有说服力的：同 checkpoint、系统扫描步数与随机性、覆盖多分辨率数据，并同时测 FID、时间、一致性、插值与重建。尤其 Table 1 清楚隔离了“少步”和“少随机”的作用。

不足也明显：

- 主要质量结论集中在较旧、较小的数据集和 FID；
- 插值与一致性主要是视觉案例，没有定量语义指标；
- 没有系统报告 precision/recall、mode coverage 或 likelihood；
- 采样 schedule 的消融较浅；
- 没有与更高阶 ODE solver 比较，因为这类方法当时尚未成熟。

## 12. 关键机制的直觉总览

可以把 DDIM 压缩成以下逻辑链：

1. 训练只随机抽一个 $t$，构造 $x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon$；
2. 因而训练看到的是一组 conditional marginals，而不是完整的 path coupling；
3. 多个 joint 可以共享同一组 marginals；
4. 每个 joint 诱导不同的 reverse generative process；
5. 选择零条件方差的 joint，可让反向过程在给定 $x_T$ 后确定；
6. 确定性过程近似某个 ODE，可在粗时间网格上直接积分；
7. 粗网格减少网络调用，代价是离散化误差；
8. 同一 ODE 的不同网格解释了跨步数的语义一致；反向积分则给出近似 encoding。

这一视角把 diffusion model 从“固定 1000 步 Markov chain”提升为“训练一个随噪声尺度变化的向量场，再选择数值路径来使用它”。

## 13. 局限性与批判性思考

### 13.1 理论等价依赖理想条件

Theorem 1 给出 $J_\sigma=L_\gamma+C$，但从 $L_\gamma$ 跳到实践中统一使用 $L_1$，依赖各 $t$ 参数不共享时的同最优解论证。真实共享 U-Net 下，不同时间权重可能改变容量分配和优化结果。论文的经验成功足以证明“可复用”，但不能把 surrogate 写成无条件严格等价。

### 13.2 $\sigma=0$ 是退化极限

严格 theorem 假设 $\sigma>0$；DDIM 的完全确定性端点通过极限和直接构造获得。此时 transition 退化为 Dirac 型映射，不再是普通有密度的高斯链。把所有 ELBO 结论机械延伸到 $\sigma=0$ 需要谨慎。

### 13.3 少步质量受一阶离散化限制

原始 DDIM 使用一阶 Euler 式更新。步数很少时，离散误差和 denoiser 误差累积明显；论文自己在讨论中提出 Adams–Bashforth 等多步方法可能继续改善。这一局限后来被 PNDM、DPM-Solver、EDM sampler 等工作系统解决。

### 13.4 确定性不等于更好的分布覆盖

FID 改善说明特征均值/协方差更接近参考分布，但不能单独判断 mode coverage。固定 $x_T$ 后只有一个输出，有利于一致性和编辑，却失去同一 latent 下的随机分支。论文没有用 precision/recall 或 density/coverage 指标回答随机性对覆盖面的影响。

### 13.5 ODE 可逆性只是近似

若向量场精确、求解器误差为零，ODE flow 可逆；实际 $\epsilon_\theta$ 有建模误差，正反离散公式也未必严格互逆。Table 2 显示重建误差随步数下降，恰好说明有限步“inversion”是数值近似，不应宣称 exact inverse。

### 13.6 对离散与组合数据只给出形式推广

附录给出 categorical non-Markovian forward process，并把目标上界连接到多类分类 loss；但没有实验。正文“可推广到非高斯/组合结构”的讨论属于理论可能性，而非已验证结论。

### 13.7 论文不解决训练成本

DDIM 显著降低 inference latency，但训练仍使用与 DDPM 相同的噪声级覆盖、模型和数据规模。它不是 distillation，也不减少训练网络调用。对只生成极少样本的场景，训练成本不会因 DDIM 自动下降。

## 14. 历史影响与后续启发

### 14.1 训练—采样解耦成为扩散模型标准范式

DDIM 让研究者意识到，同一个 denoiser 可以搭配多个 sampler。今天常见的 scheduler/sampler 切换，本质上沿用了这种“固定模型，改变数值轨迹”的思想。

### 14.2 从离散 Markov chain 转向连续动力系统

DDIM 把确定性 diffusion sampling 写成 ODE 离散化，并讨论与 probability flow ODE 的等价。这为高阶 solver、误差控制、噪声参数化和更优 timestep spacing 提供了直接入口。

### 14.3 一致 latent 支持可控生成与编辑

确定性 $x_T\leftrightarrow x_0$ 对应关系使 latent interpolation、reconstruction 和 trajectory manipulation 更自然。后来的图像编辑方法借用这一结构进行 inversion、prompt-conditioned regeneration 和 attention intervention，但这些是后续扩展，不应倒写成 DDIM 原论文已完成的贡献。

### 14.4 随机性成为可调设计轴

$\eta$ 揭示采样随机性不是“有或无”的二元选择。少步时 deterministic trajectory 往往更稳定；更多步或其他任务中，适量随机性可能有利。现代 SDE/ODE sampler、ancestral sampler 与 stochastic churn 都可视为继续探索这一设计轴。

## 15. 常见误解澄清

1. **误解：DDIM 训练了一个新模型。** 实际上核心卖点正是复用 DDPM 的 $\epsilon_\theta$ 训练目标与 checkpoint。
2. **误解：DDIM 的前向过程就是普通 DDPM diffusion。** 它保留单时刻边际，但 joint 一般 non-Markovian；$\sigma=0$ 时甚至不再是通常意义的随机 diffusion。
3. **误解：DDIM 完全没有随机性。** $\eta=0$ 时每步无新噪声，但初始 $x_T$ 仍随机；$\eta>0$ 的 generalized sampler 也可称 DDIM 家族的随机成员。
4. **误解：$\eta=1$ 总比 $\eta=0$ 差。** 短步数中通常如此，1000 步 $\hat\sigma$ 在 Table 1 反而略优。
5. **误解：跳步只是把 DDPM 的相邻公式随便跨着用。** 论文从保留子序列边际的 inference process 重新论证了相应生成过程。
6. **误解：ODE 等价意味着任意有限步实现相同。** 连续极限等价；不同离散变量、网格和 solver 在少步时差异很大。
7. **误解：原论文已经提出 Stable Diffusion 的 DDIM inversion。** 原文只有无条件模型上的 ODE encoding/reconstruction；现代条件编辑 pipeline 是后续工作。

## 16. 关键术语表

| 术语 | 含义 |
|---|---|
| Conditional marginal | 给定 $x_0$ 的单时刻分布 $q(x_t\mid x_0)$；DDIM 与 DDPM 保持一致 |
| Joint / inference process | 所有 $x_{1:T}$ 在给定 $x_0$ 下的联合分布；DDIM 与 DDPM 的主要差别所在 |
| Non-Markovian | $x_t$ 的条件关系不能只由相邻状态概括；本文前向条件一般还依赖 $x_0$ |
| Implicit generative model | 通过随机 latent 和生成过程定义样本，不要求显式给出可计算的输出密度 |
| Sampling trajectory $\tau$ | 推理时真正访问的时间步子序列，其长度 $S$ 决定网络调用数 |
| $\eta$ | 缩放采样方差的超参数；$0$ 为确定性 DDIM，$1$ 对应 DDPM 型方差 |
| Probability flow ODE | 与某 SDE 共享时间边际、但轨迹确定的 ODE；最优模型下与 DDIM 连续形式相关 |
| DDIM inversion | 后世对确定性采样轨迹的反向估计；需与原论文的 ODE 重建实验严格区分 |
| Trajectory consistency | 固定 $x_T$、改变步数时，高层语义大致保持而细节随离散精度变化 |

## 17. 复现与阅读检查清单

- [x] 记号确认：论文 $\alpha_t$ 等价于 DDPM 常用的 $\bar\alpha_t$。
- [x] 前向过程确认：相同的是 $q(x_t\mid x_0)$，不是完整 joint。
- [x] 训练等价确认：$J_\sigma=L_\gamma+C$；$L_1$ 的共享最优解论证有参数条件。
- [x] 确定性确认：$\eta=0$ 去掉逐步噪声，但 $x_T$ 仍从高斯采样。
- [x] 跳步确认：$\tau$ 是训练时间网格的子序列，不需要重训。
- [x] ODE 确认：连续形式等价不保证有限步离散更新相同。
- [x] 实验确认：少步数下 DDIM 优势最大，1000 步并非始终最佳。
- [x] inversion 边界确认：原文是 ODE encoding/reconstruction，不含后世文本条件编辑技术。

## 18. 图表索引

| 本地文件 | 原文位置 | 用途 |
|---|---|---|
| `fig1_graphical_models.png` | Figure 1，第 2 页 | DDPM Markov joint 与 DDIM non-Markovian joint 对比 |
| `fig2_accelerated_generation.png` | Figure 2，第 5 页 | 时间子序列与跳步生成图模型 |
| `table1_fig3_fid_samples.png` | Table 1 + Figure 3，第 7 页 | 步数、$\eta$、FID 与样本质量消融 |
| `fig4_sampling_efficiency.png` | Figure 4，第 8 页 | 采样耗时与步数近似线性 |
| `fig5_trajectory_consistency.png` | Figure 5，第 8 页 | 固定 $x_T$ 的跨轨迹一致性 |
| `fig6_latent_interpolation.png` | Figure 6，第 8 页 | latent slerp 的语义插值 |
| `table2_reconstruction_error.png` | Table 2，第 9 页 | 反向编码与重建误差 |

## 19. 最终评价

DDIM 最重要的贡献不是某一条采样公式，而是一个建模自由度：**denoising objective 约束的是各噪声级的条件边际监督，却没有唯一钉死跨时间 joint。** 利用这份自由度，论文把扩散采样从必须模拟 1000 步随机 Markov 链，改造成可选择随机性、可抽稀时间网格、可解释为 ODE 的生成过程。

从今天回看，原始 DDIM 的一阶 solver 已不再是少步采样的最终答案，但它完成了关键的概念转向：训练一个 score/denoising field，与如何数值穿越该 field 是两个可以分别研究的问题。后续快速 sampler、ODE solver、inversion 与编辑技术，几乎都沿着这条分界继续发展。
