---
title: "VAE - 变分推断与重参数化生成模型"
aliases:
  - "Auto-Encoding Variational Bayes"
  - "Variational Auto-Encoder"
  - "VAE"
authors:
  - Diederik P. Kingma
  - Max Welling
year: 2013
venue: ICLR 2014
arxiv: "1312.6114"
tags:
  - paper
  - generative-modeling
  - variational-inference
  - variational-autoencoder
  - reparameterization-trick
  - amortized-inference
status: 精读完成
---

# VAE：变分推断与重参数化生成模型

> **论文**：Diederik P. Kingma, Max Welling. *Auto-Encoding Variational Bayes*  
> **发表信息**：arXiv:1312.6114，初稿 2013；ICLR 2014  
> **核对版本**：arXiv v11，2022-12-10  
> **来源**：[arXiv 摘要页](https://arxiv.org/abs/1312.6114) · [官方 PDF](https://arxiv.org/pdf/1312.6114) · [[assets/VAE - 变分推断与重参数化生成模型/sources/paper.pdf|本地官方 PDF]]  
> **本地核验记录**：[[assets/VAE - 变分推断与重参数化生成模型/sources.md|sources.md]]  
> **精读日期**：2026-10-16

## 1. 一句话总结（TL;DR）

这篇论文解决的是：当连续隐变量模型的真实后验和边缘似然都不可解时，怎样仍然用小批量随机梯度高效学习。答案是把变分后验采样写成“与参数无关的噪声 + 可微确定性变换”，由此得到低方差、可反向传播的 **SGVB** 估计器；再用共享的 recognition model 摊销每个样本的后验推断，形成 **AEVB**，其神经网络实例就是 VAE。

最值得记住的不是“自编码器加 KL loss”，而是三个彼此咬合的思想：**ELBO 给出概率目标，重参数化给出路径梯度，摊销推断把逐样本优化变成一次前向传播。**

## 2. 研究问题与背景

### 2.1 隐变量生成模型的基本设定

论文考虑带连续隐变量的有向生成模型。对每个观测样本，先从先验生成潜变量，再由潜变量生成观测：

$$p_\theta(x,z)=p_\theta(z)p_\theta(x\mid z).$$

理想情况下，我们希望同时完成三件事：

1. 学习生成参数 $\theta$，即近似最大化数据边缘似然；
2. 给定 $x$ 时推断其潜变量后验 $p_\theta(z\mid x)$；
3. 利用学到的模型进行生成、表示、去噪、补全或可视化。

困难在于边缘似然需要积分：

$$p_\theta(x)=\int p_\theta(x,z)\,dz,$$

而真实后验又依赖这个通常不可解的积分：

$$p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}.$$

当 $p_\theta(x\mid z)$ 由非线性神经网络参数化时，常见的共轭结构消失，既不能精确求后验，也不能直接计算对数边缘似然。逐样本运行 MCMC 或迭代式变分优化又难以扩展到大数据。

### 2.2 为什么传统变分推断仍不够

变分推断用可处理的 $q_\phi(z\mid x)$ 近似真实后验。传统 mean-field Variational Bayes 往往依赖共轭指数族，使坐标更新中的期望可解析；但论文关心的是一般非线性模型，此时期望本身也可能不可解。

即使改用 Monte Carlo，朴素 score-function 估计器也会遇到高方差：

$$\nabla_\phi\mathbb E_{q_\phi(z)}[f(z)]=\mathbb E_{q_\phi(z)}\left[f(z)\nabla_\phi\log q_\phi(z)\right].$$

这个恒等式适用范围广，却让高噪声的函数值 $f(z)$ 直接乘在 score 上。论文明确指出，这类估计器的方差对其目标而言过高，难以实用。

### 2.3 论文处理的图模型

![[assets/VAE - 变分推断与重参数化生成模型/figures/figure-1-graphical-model.png]]

*图 1：原文 Figure 1。实线表示生成模型 $p_\theta(z)p_\theta(x\mid z)$，虚线表示用来逼近不可解真实后验的 $q_\phi(z\mid x)$；$\theta$ 与 $\phi$ 被联合学习，方框表示 $N$ 个 i.i.d. 样本。*

图中有两套方向相反但共同训练的计算：

- **生成模型 / probabilistic decoder**：$z\rightarrow x$，由 $p_\theta(x\mid z)$ 定义；
- **推断模型 / recognition model / probabilistic encoder**：$x\rightarrow z$，由 $q_\phi(z\mid x)$ 定义。

推断模型不是生成模型本身的一部分，而是为近似后验推断和训练引入的辅助分布。生成时只需要先验与 decoder；encoder 的作用是训练期间提供高效近似后验，并在测试时快速编码观测。

## 3. 核心贡献

1. **提出 Stochastic Gradient Variational Bayes（SGVB）估计器。** 通过重参数化，把对参数化分布的随机采样改写为对固定噪声源的可微变换，从而可用标准反向传播优化变分下界。
2. **提出 Auto-Encoding Variational Bayes（AEVB）算法。** 对每个数据点都带局部连续隐变量的 i.i.d. 数据，用共享 recognition model 摊销后验推断，避免为每个样本单独运行昂贵的迭代推断。
3. **给出 Variational Auto-Encoder（VAE）实例。** 使用高斯概率编码器输出均值和方差，用 Bernoulli 或 Gaussian 概率解码器定义观测似然，并联合训练 $\theta$ 与 $\phi$。
4. **给出更低方差的解析 KL 版本。** 当 $q_\phi(z\mid x)$ 与先验之间的 KL 可解析时，只对重构期望做 Monte Carlo 采样。
5. **验证在线、小批量学习的可行性。** 在 MNIST 与 Frey Face 上，AEVB 比 wake-sleep 更快达到更高的变分下界，并在低维潜空间中与 MCEM 比较边缘似然收敛。

## 4. ELBO：从不可解似然到可优化下界

### 4.1 KL 分解推导

对单个样本 $x$，引入近似后验 $q_\phi(z\mid x)$。从 KL 散度定义出发：

$$D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p_\theta(z\mid x)\right)=\mathbb E_{q_\phi(z\mid x)}\!\left[\log q_\phi(z\mid x)-\log p_\theta(z\mid x)\right].$$

利用 Bayes 公式 $\log p_\theta(z\mid x)=\log p_\theta(x,z)-\log p_\theta(x)$，可整理为：

$$\log p_\theta(x)=\mathcal L(\theta,\phi;x)+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p_\theta(z\mid x)\right).$$

其中：

$$\mathcal L(\theta,\phi;x)=\mathbb E_{q_\phi(z\mid x)}\!\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right].$$

由于 KL 散度非负：

$$\mathcal L(\theta,\phi;x)\leq\log p_\theta(x).$$

因此 $\mathcal L$ 是 Evidence Lower Bound，即 ELBO。这个分解同时说明：

- 对 $\theta$ 最大化 ELBO，会提高数据的近似对数似然；
- 对 $\phi$ 最大化 ELBO，会缩小近似后验与真实后验的 KL；
- 下界与真实 log evidence 的差恰好是真实后验方向上的 KL。

### 4.2 Jensen 不等式视角

同一下界也可以直接由 Jensen 不等式得到：

$$\log p_\theta(x)=\log\mathbb E_{q_\phi(z\mid x)}\!\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]\geq\mathbb E_{q_\phi(z\mid x)}\!\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right].$$

两个视角各有作用：Jensen 推导解释“为什么这是下界”，KL 分解解释“下界离真实似然差多少”。如果变分族表达能力不足，即使优化完全收敛，也可能存在不可消除的 approximation gap。

### 4.3 重构项与 KL 项

把联合分布展开为 $p_\theta(x,z)=p_\theta(x\mid z)p_\theta(z)$，得到论文式（3）：

$$\mathcal L(\theta,\phi;x)=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p_\theta(z)\right).$$

两项的概率含义是：

1. **期望条件对数似然**：从近似后验得到的 $z$ 应让 decoder 给真实观测 $x$ 较高概率；
2. **后验—先验 KL**：近似后验不能任意漂离先验，否则从先验采样时会落入 decoder 未见过的潜空间区域。

工程实现常最小化负 ELBO：

$$\mathcal J_{\mathrm{VAE}}=-\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p_\theta(z)\right).$$

这里的“重构误差”不是固定的 MSE，而是由观测似然决定：Bernoulli decoder 对应 binary cross-entropy 形式；固定方差 Gaussian decoder 才与加权 MSE 等价。把任意像素损失都称作 VAE 的重构项，会丢失其概率模型含义。

## 5. 重参数化技巧：把随机性移到参数之外

### 5.1 一般形式

原问题是 $z\sim q_\phi(z\mid x)$，样本本身随 $\phi$ 改变，难以直接沿采样操作反传。论文将其改写为：

$$\epsilon\sim p(\epsilon),\qquad z=g_\phi(\epsilon,x),$$

其中辅助噪声分布 $p(\epsilon)$ 不依赖 $\phi$，而 $g_\phi$ 对 $\phi$ 可微。于是：

$$\mathbb E_{q_\phi(z\mid x)}[f(z)]=\mathbb E_{p(\epsilon)}[f(g_\phi(\epsilon,x))].$$

对应的 Monte Carlo 估计为：

$$\mathbb E_{q_\phi(z\mid x)}[f(z)]\approx\frac{1}{L}\sum_{l=1}^{L}f\!\left(g_\phi(\epsilon^{(l)},x)\right),\qquad\epsilon^{(l)}\sim p(\epsilon).$$

给定一次噪声样本后，$x\rightarrow$ 后验参数 $\rightarrow z\rightarrow$ decoder $\rightarrow$ loss 成为普通可微计算图。梯度沿样本路径传播：

$$\nabla_\phi\mathbb E_{p(\epsilon)}[f(g_\phi(\epsilon,x))]=\mathbb E_{p(\epsilon)}\!\left[\nabla_z f(z)\frac{\partial g_\phi(\epsilon,x)}{\partial\phi}\right].$$

重参数化没有消除随机性，而是把随机性与待优化参数分离。与 score-function 梯度相比，pathwise gradient 利用了 $f$ 对 $z$ 的局部导数信息，通常方差更低。

### 5.2 对角高斯情形

若近似后验为：

$$q_\phi(z\mid x)=\mathcal N\!\left(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right),$$

则可以采样：

$$\epsilon\sim\mathcal N(0,I),\qquad z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.$$

实际实现一般让 encoder 输出 `logvar`，再计算标准差：

$$\sigma=\exp\!\left(\frac{1}{2}\log\sigma^2\right).$$

这样既保证标准差为正，也避免直接预测很小的正数。最常见的实现错误之一是写成 `std = exp(logvar)`，从而把方差指数化了两次。

### 5.3 哪些分布可以重参数化

论文列出三类构造策略：

1. **逆 CDF**：令 $\epsilon\sim U(0,I)$，通过目标分布的 inverse CDF 得到样本；
2. **location-scale family**：使用标准分布噪声，并令样本等于 location 加 scale 乘噪声；
3. **组合变换**：把目标变量表示为基础随机变量的可微组合。

论文列举的可用例子包括 Gaussian、Laplace、Student's t、Logistic、Uniform、Gumbel 等。原始方法的核心适用边界是**连续且可微重参数化的随机变量**；离散采样不能直接套用这条路径，需要 score-function、连续松弛或其他后续估计器。

## 6. SGVB 估计器

### 6.1 通用估计器 $\widetilde{\mathcal L}^{A}$

把重参数化样本代入 ELBO，得到论文式（6）：

$$\widetilde{\mathcal L}^{A}(\theta,\phi;x)=\frac{1}{L}\sum_{l=1}^{L}\left[\log p_\theta(x,z^{(l)})-\log q_\phi(z^{(l)}\mid x)\right].$$

其中：

$$z^{(l)}=g_\phi(\epsilon^{(l)},x),\qquad\epsilon^{(l)}\sim p(\epsilon).$$

这是通用的 SGVB 估计器。它直接估计联合密度与变分密度之差，因此适用面广，但两个随机项都进入估计，方差可能较大。

### 6.2 解析 KL 的低方差估计器 $\widetilde{\mathcal L}^{B}$

当 $D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z))$ 可解析时，使用论文式（7）：

$$\widetilde{\mathcal L}^{B}(\theta,\phi;x)=-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p_\theta(z)\right)+\frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x\mid z^{(l)}).$$

此时只有期望条件对数似然需要采样，KL 项精确计算，通常方差更低。现代基础 VAE 实现通常采用这个版本。

### 6.3 从单样本到小批量

对完整数据集 $X=\{x^{(i)}\}_{i=1}^{N}$，随机抽取大小为 $M$ 的 minibatch $X^M$，完整数据 ELBO 的无偏估计为：

$$\widetilde{\mathcal L}^{M}(\theta,\phi;X^M)=\frac{N}{M}\sum_{i=1}^{M}\widetilde{\mathcal L}(\theta,\phi;x^{(i)}).$$

论文实验使用 $M=100$，每个样本只抽 $L=1$ 个潜变量样本。直觉是：minibatch 本身已经对许多数据点做了平均，在 batch 足够大时，不必为每个 $x$ 再抽很多个 $z$ 才能得到可用梯度。

## 7. AEVB 算法：摊销推断与联合学习

### 7.1 Algorithm 1

![[assets/VAE - 变分推断与重参数化生成模型/figures/algorithm-1-aevb.png]]

*图 2：原文 Algorithm 1。每一步抽取 minibatch 与辅助噪声，用可微的 minibatch ELBO 计算 $\nabla_{\theta,\phi}$，再同时更新生成参数与推断参数。论文实验采用 $M=100$、$L=1$。*

AEVB 的一次训练迭代可以展开为：

1. 从数据集随机抽取 minibatch；
2. encoder 对每个 $x$ 输出近似后验参数；
3. 从固定噪声分布采样 $\epsilon$；
4. 通过 $z=g_\phi(\epsilon,x)$ 得到潜变量样本；
5. decoder 计算 $p_\theta(x\mid z)$；
6. 计算重构项与 KL 项，形成 minibatch ELBO；
7. 一次反向传播同时更新 $\theta$ 和 $\phi$。

AEVB 不是另一条独立于 SGVB 的估计公式，而是把 SGVB 用于“每个数据点有局部隐变量、且用共享推断模型近似后验”的具体学习算法。

### 7.2 摊销推断为何关键

传统变分推断为每个样本维护独立局部参数 $\lambda_i$，新样本到来后还要重新优化。AEVB 改为：

$$\lambda_i=f_\phi(x^{(i)}).$$

共享 encoder 直接预测每个样本的后验参数。其收益包括：

- 训练时不必为每个样本嵌套运行局部迭代优化；
- 测试时一次前向传播即可得到近似后验；
- 参数共享可利用数据集级统计规律；
- 推断网络与生成模型可以在同一个 ELBO 上联合学习。

代价是 **amortization gap**：给定变分族后，某个样本可能存在更优的局部变分参数，但共享网络未必能准确输出它。于是 inference gap 可理解为两部分：变分族受限造成的 approximation gap，以及共享推断映射造成的 amortization gap。后一套术语和系统分析主要由后续研究明确化，不是原文的中心表述。

### 7.3 “Auto-Encoding”到底指什么

ELBO 中自然出现以下数据流：

$$x\xrightarrow{q_\phi(z\mid x)}z\xrightarrow{p_\theta(x\mid z)}x.$$

但 VAE 与普通 deterministic autoencoder 有本质区别：

| 维度 | 普通自编码器 | VAE / AEVB |
|---|---|---|
| encoder 输出 | 一个确定性向量 | 潜变量分布的参数 |
| 潜码 | 通常确定 | 从 $q_\phi(z\mid x)$ 采样 |
| decoder | 通常输出重构值 | 定义条件分布 $p_\theta(x\mid z)$ |
| 训练目标 | 经验重构损失 | 有概率意义的 ELBO |
| 潜空间约束 | 由经验正则决定 | 近似后验通过 KL 与先验匹配 |
| 无条件生成 | 不天然支持 | 从 $p(z)$ 采样后解码 |

因此，VAE 首先是一个隐变量概率模型与变分推断算法，其次才呈现出 autoencoder 式架构。

## 8. VAE 实例：高斯编码器与概率解码器

### 8.1 模型选择

论文实例采用标准正态先验：

$$p(z)=\mathcal N(0,I).$$

encoder 使用对角高斯近似后验：

$$q_\phi(z\mid x)=\mathcal N\!\left(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right).$$

decoder 根据数据类型选择观测分布：

- 二值数据使用 multivariate Bernoulli；
- 连续数据使用对角协方差 Gaussian；
- 分布参数由单隐层 MLP 输出。

这种设计不是说真实后验本身是对角高斯，而是选择一个便于采样、算密度和计算 KL 的变分族去近似不可解后验。

### 8.2 高斯 KL 的闭式表达

对第 $j$ 个潜变量维度：

$$D_{\mathrm{KL}}\!\left(\mathcal N(\mu_j,\sigma_j^2)\Vert\mathcal N(0,1)\right)=\frac{1}{2}\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).$$

各维相加得到：

$$D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p(z)\right)=\frac{1}{2}\sum_{j=1}^{J}\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).$$

因此论文式（10）的单样本 ELBO 估计为：

$$\widetilde{\mathcal L}(\theta,\phi;x)=\frac{1}{2}\sum_{j=1}^{J}\left(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\right)+\frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x\mid z^{(l)}).$$

其中：

$$z^{(l)}=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon^{(l)},\qquad\epsilon^{(l)}\sim\mathcal N(0,I).$$

第一项是**负 KL**。实现时必须分清是在最大化 ELBO，还是在最小化 negative ELBO；符号多翻或少翻一次都会让 encoder 被错误地推离先验。

### 8.3 训练、编码与生成

**训练路径：** encoder 输出 $\mu$ 与 $\log\sigma^2$，采样噪声并重参数化得到 $z$，decoder 定义 $p_\theta(x\mid z)$，然后最大化 ELBO。

$$x\rightarrow(\mu,\log\sigma^2)\rightarrow z\rightarrow p_\theta(x\mid z).$$

**后验编码：** 给定新样本 $x$，可以用 $\mu_\phi(x)$ 作为确定性表征，也可以从 $q_\phi(z\mid x)$ 采样以表达不确定性。

**无条件生成：**

$$z\sim\mathcal N(0,I),\qquad x\sim p_\theta(x\mid z).$$

生成时不需要 encoder。若从 $q_\phi(z\mid x)$ 采样再解码，那是重构或条件于输入的生成，不是从模型先验进行无条件生成。

## 9. 与同时代方法的差异

### 9.1 Wake-Sleep

两者都有生成网络和 recognition model，但优化目标不同：

- Wake-Sleep 交替优化 wake objective 与 sleep objective，它们合在一起并不等价于同一个边缘似然下界；
- AEVB 让 encoder 与 decoder 共同优化一个 ELBO，可通过一次反向传播得到联合梯度；
- Wake-Sleep 的优势是也能处理离散隐变量，而原始重参数化路径主要面向连续隐变量。

### 9.2 MCEM 与逐样本 MCMC

MCEM 不使用 encoder，而是针对每个样本从真实后验近似采样。论文实现使用 HMC：每次执行 10 个 leapfrog steps，并自动调节步长使接受率约为 90%，随后进行 5 次参数更新。其推断成本随样本数累积，不适合大规模在线学习。

AEVB 把重复后验推断的成本摊销到 encoder 参数中，新样本只需一次前向传播。代价是 encoder 近似误差与变分族限制。

### 9.3 普通 Autoencoder

普通 autoencoder 常用 denoising、sparsity 或 contractive penalty 等经验正则；VAE 的 KL 项来自边缘似然下界，不是任意添加的正则超参数。另一方面，优化 ELBO 并不保证潜表示一定最适合分类等下游任务，也不保证像素级样本质量优于所有判别式或生成式替代方案。

## 10. 实验设置与结果

### 10.1 数据集、模型与训练细节

论文在两个图像数据集上训练：

- **MNIST**：二值观测，使用 Bernoulli decoder；
- **Frey Face**：连续观测，使用 Gaussian decoder，均值经 sigmoid 限制在 $(0,1)$。

主要训练设置如下：

| 项目 | 设置 |
|---|---|
| 网络 | encoder 与 decoder 均为单隐层 MLP |
| MNIST 隐层 | 各 500 个 hidden units |
| Frey Face 隐层 | 各 200 个 hidden units，以降低小数据集过拟合 |
| minibatch | $M=100$ |
| 每样本潜变量样本数 | $L=1$ |
| 参数初始化 | $\mathcal N(0,0.01)$ |
| 优化器 | Adagrad |
| 候选全局步长 | $\{0.01,0.02,0.1\}$，按训练初期表现选择 |
| 参数先验 | 小权重衰减，对应 $p(\theta)=\mathcal N(0,I)$ |
| 实际学习解释 | 使用 ELBO 梯度近似 MAP 学习 |

对比方法包括 Wake-Sleep；在低维 MNIST 实验中还包括使用 HMC 的 MCEM。评价指标包括平均 variational lower bound，以及仅在低维潜空间中才较可靠的 estimated marginal log-likelihood。

### 10.2 ELBO 收敛：AEVB 对比 Wake-Sleep

![[assets/VAE - 变分推断与重参数化生成模型/figures/figure-2-lower-bound-comparison.png]]

*图 3：原文 Figure 2。红色 AEVB 在 MNIST 与 Frey Face 的多种潜变量维度下均比绿色 Wake-Sleep 更快达到更高的变分下界；实线与虚线分别对应 train 与 test。*

图中支持的结论是：

- AEVB 在全部展示设置中收敛更快，并达到更好的下界；
- 增加潜变量维度没有明显导致过拟合，作者将其归因于变分下界的正则作用；
- 下界估计器方差小于 1，因此图中省略误差条；
- 当时使用有效算力约 40 GFLOPS 的 Intel Xeon CPU，每评估一百万个训练样本约需 20–40 分钟。

读图时要注意纵轴是每个数据点的平均变分下界。“更高”意味着下界更接近真实 log evidence；MNIST 上数值为负时，越接近 0 越好。Frey Face 使用连续密度，log-likelihood 可为正，不能跨数据集直接比较绝对数值。

### 10.3 边缘似然：AEVB、Wake-Sleep 与 MCEM

![[assets/VAE - 变分推断与重参数化生成模型/figures/figure-3-marginal-likelihood.png]]

*图 4：原文 Figure 3。左图训练集规模为 1000，右图为 50000；AEVB 快速提升，MCEM 因逐样本 HMC 成本无法高效扩展到完整 MNIST。*

该实验把 encoder 与 decoder 改为各 100 个 hidden units，并把潜变量维度设为 3。原因不是三维最优，而是论文附录的边缘似然估计器在潜空间高于约 5 维后会变得不可靠。

主要观察为：

- 在 $N_{\mathrm{train}}=1000$ 的小训练集上，AEVB 的训练边缘似然接近 MCEM，但测试曲线显示过拟合；
- 在 $N_{\mathrm{train}}=50000$ 的完整训练集上，AEVB 明显优于 Wake-Sleep；
- MCEM 不是在线算法，完整数据上的推进速度很慢，体现了逐样本 MCMC 的扩展瓶颈。

### 10.4 潜空间可视化与生成

论文附录展示了二维 Frey Face、MNIST 潜空间的连续生成流形，以及不同潜变量维度下的 MNIST 随机样本。二维可视化通过在单位方格上取点，再用高斯逆 CDF 映射到潜空间坐标，最后绘制对应的 $p_\theta(x\mid z)$。

这些图说明先验高概率区域内的连续移动可以带来平滑的输出变化，但只能作为定性证据。它们不能单独证明边缘似然更好、样本更真实，或表示更适合下游任务。

### 10.5 实验证据是否充分

以 2013 年的研究目标看，实验有效支持了“方法可训练、收敛快、适合 minibatch、可扩展到大数据”的主张。但证据边界很清楚：

- 只有 MNIST 与 Frey Face 两个小型图像数据集；
- 模型只是单隐层 MLP；
- 没有现代意义上的系统 ablation；
- 主要比较对象有限；
- 边缘似然估计只能在低维潜空间可靠使用；
- 没有系统分离后验质量、样本质量、表示质量和 likelihood 之间的关系。

因此，这是一篇证明新训练范式有效的**奠基方法论文**，而不是证明 VAE 在所有生成任务上都优于替代方案的全面 benchmark。

## 11. 局限性与批判性思考

### 11.1 方法本身的边界

1. **连续且可重参数化的隐变量。** 原始 pathwise estimator 不能直接穿过离散采样。
2. **受限的变分族。** 对角高斯无法表达多峰、偏斜或强相关真实后验，可能产生较大的 approximation gap。
3. **ELBO 可能较松。** 最大化下界不等于直接最大化精确 log-likelihood；更灵活的后验或更紧的多样本界可能显著改变结果。
4. **摊销推断引入额外误差。** 一次前向传播换来效率，但共享 encoder 不一定输出每个样本的最优变分参数。
5. **观测似然假设影响生成质量。** 简单 Bernoulli 或对角 Gaussian 可能偏向均值式、模糊的重构；这不是重参数化单独造成的，而是 decoder 分布与目标共同作用的结果。
6. **实验规模有限。** 浅层网络与小图像数据不足以揭示强 decoder、层级潜变量和高维复杂数据下的优化问题。
7. **边缘似然评估受维度限制。** 原文附录估计器要求采样空间低于约 5 维且样本足够多，因此 Figure 3 的结论不能直接外推到高维潜空间。

### 11.2 不应把 KL 只理解为“让潜空间平滑”

“reconstruction loss + KL regularizer”是方便的工程简称，但容易遮蔽概率语义：

- 重构项是条件对数似然，不是固定距离；
- KL 比较的是每个样本的近似后验与先验，不只是视觉上的潜空间平滑度；
- 原始 ELBO 中两项的相对权重由概率模型固定为 1；
- 任意改变 KL 权重会得到不同目标，例如后来的 $\beta$-VAE，而不再是原始模型的标准 ELBO。

### 11.3 论文没有解决的评估问题

较高 ELBO 可能来自更好的生成模型、更好的推断模型，或两者共同改善。若只报告 ELBO，很难知道 posterior approximation 是否准确，也难判断 latent code 是否有用。后续实践应同时报告 likelihood estimate、重构质量、prior samples、KL 使用量、active units 和下游表示指标。

## 12. 后世视角：Posterior Collapse

> [!important] 历史边界
> **Posterior collapse（后验坍塌）不是本文提出、命名或系统分析的结论。** 原论文研究的是浅层 MLP 图像模型，并主要把 KL 解释为有益正则。强 decoder 可能完全忽略 $z$ 的问题，是 VAE 后来被用于文本、自回归解码器和更强生成模型时被广泛认识和系统研究的现象。

### 12.1 什么是 posterior collapse

当 decoder 足够强时，它可能在几乎不使用潜变量的情况下拟合数据：

$$p_\theta(x\mid z)\approx p_\theta(x).$$

此时 encoder 可以令：

$$q_\phi(z\mid x)\approx p(z),$$

于是：

$$D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p(z)\right)\approx0.$$

潜变量几乎不携带关于 $x$ 的信息，decoder 对 $z$ 不敏感。训练 loss 或 ELBO 仍可能看起来正常，但“学习有意义潜表示”的目标已经失败。

### 12.2 为什么原始 ELBO 允许坍塌

令 $q(x)$ 表示经验数据分布，aggregate posterior 为 $q_\phi(z)=\mathbb E_{q(x)}[q_\phi(z\mid x)]$，平均 KL 可分解为：

$$\mathbb E_{q(x)}D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p(z)\right)=I_q(x;z)+D_{\mathrm{KL}}\!\left(q_\phi(z)\Vert p(z)\right).$$

这说明 KL 不仅推动 aggregate posterior 匹配先验，也会惩罚互信息 $I_q(x;z)$。若强 decoder 能独立解释 $x$，优化器可能偏好低互信息、低 KL 的解。ELBO 的目标是 likelihood bound，而不是强制潜变量必须被使用。

### 12.3 诊断与缓解

常见诊断信号包括：

- 总 KL 很快降到接近 0；
- 大多数潜变量维度的逐维 KL 接近 0；
- active units 很少；
- 固定 $x$ 改变 $z$ 时 decoder 输出变化很小；
- prior samples 与重构都主要由 decoder 的自回归能力决定。

后续常见缓解思路包括：

- **KL annealing**：训练早期降低 KL 权重，让 decoder 先学会使用 $z$；
- **free bits / KL floor**：给各潜变量维度保留一定免费信息容量；
- **调整 decoder 能力或训练日程**：避免 decoder 过早学成不依赖 $z$ 的模型；
- **更灵活的后验与层级潜变量**：减小 inference gap；
- **显式控制 rate–distortion 或互信息**：把潜变量信息量变成可观测、可调节的设计量。

这些方法都属于后续发展，不是原始 AEVB 算法的一部分，也没有一种能在所有模型与数据上保证消除坍塌。

## 13. 复现与实现建议

### 13.1 最小正确目标

设 encoder 输出 `mu` 与 `logvar`，重参数化为：

$$\epsilon\sim\mathcal N(0,I),\qquad z=\mu+\exp(0.5\,\mathrm{logvar})\odot\epsilon.$$

标准正态先验下，每个样本的 KL 为：

$$\mathrm{KL}(x)=-\frac{1}{2}\sum_j\left(1+\mathrm{logvar}_j-\mu_j^2-\exp(\mathrm{logvar}_j)\right).$$

若训练代码最小化 negative ELBO：

$$\mathrm{loss}=\mathrm{reconstruction\ NLL}+\mathrm{KL}.$$

`reconstruction NLL` 必须与 decoder likelihood 对应：二值数据用 Bernoulli NLL；实值数据需要明确 Gaussian 的均值与方差如何建模，不能无说明地把任意 MSE 当作同一概率模型。

### 13.2 reduction 与尺度必须统一

重构项通常跨观测维度求和，KL 跨潜变量维度求和，然后再对 batch 求平均。若重构先对像素取 mean、KL 却对潜变量取 sum，两项相对比例会随图像分辨率和潜变量维度改变，等价于无意中修改了目标。

推荐明确记录：

- 每项是按样本 sum 还是 mean；
- batch 维度如何归约；
- 报告的 loss 是总和、每样本均值还是每维均值；
- 训练曲线中的 ELBO 是否与评估代码采用同一标度。

### 13.3 对照原论文的复现实验

若目标是复现论文而不是训练现代高性能 VAE，建议保持：

1. MNIST 使用 Bernoulli decoder，Frey Face 使用 Gaussian decoder；
2. 单隐层 MLP，MNIST 隐层 500，Frey Face 隐层 200；
3. minibatch size 为 100，每个样本 $L=1$；
4. 参数从 $\mathcal N(0,0.01)$ 初始化；
5. 使用 Adagrad，并记录从 $\{0.01,0.02,0.1\}$ 选择全局步长的规则；
6. 同时报告 train/test ELBO 随“已评估训练样本数”的变化，而不只按 epoch；
7. 与 Wake-Sleep 比较时确保 encoder、decoder 容量和初始化规则一致；
8. 只有在潜变量很低维且采样充分时，才复现附录中的 MCMC marginal-likelihood estimator。

### 13.4 建议增加的现代复现检查

在忠实复现之外，建议增加以下检查以避免“loss 能下降但模型失效”：

- 报告 reconstruction NLL、KL 和 ELBO 三条曲线；
- 报告逐维 KL 与 active units；
- 同时展示重构样本与从 $p(z)$ 生成的 prior samples；
- 测试固定 $x$ 时多次后验采样的输出变化；
- 检查 posterior collapse，而不是只看总 loss；
- 固定随机种子并报告多次运行的均值与方差；
- 清楚区分训练用单样本 ELBO 与评估用 importance-weighted likelihood estimate。

### 13.5 常见实现错误

1. 把 `std` 写成 `exp(logvar)`，正确形式是 `exp(0.5 * logvar)`；
2. KL 符号写反，或混淆最大化 ELBO 与最小化 negative ELBO；
3. 训练时直接用 $z=\mu$，使随机目标退化；
4. reconstruction 与 KL 使用不一致的 reduction；
5. 忘记 decoder 输出的是分布参数，而不是天然等同于最终样本；
6. 生成时从 $q(z\mid x)$ 而不是 $p(z)$ 采样，却称作无条件生成；
7. 只看重构图，不检查 prior samples、KL 与 active dimensions；
8. 把较高 ELBO 自动解释为更好的感知样本质量。

## 14. 后续影响与方法演化

### 14.1 三个最深远的思想

1. **Pathwise gradient estimator** 成为连续随机节点反向传播的标准工具；
2. **Amortized variational inference** 让复杂生成模型可以用 encoder 快速推断；
3. **概率 encoder–decoder + ELBO** 成为深度隐变量模型的统一模板。

### 14.2 典型后续方向

- **更紧的下界**：IWAE 使用多样本 importance weighting 收紧下界；
- **更灵活的后验**：normalizing flows、auxiliary variables、hierarchical posterior；
- **可控表示与解耦**：$\beta$-VAE、FactorVAE 等改变 rate–distortion 权衡或增加解耦约束；
- **层级 VAE**：使用多层潜变量提升表达能力，同时面对更困难的优化与坍塌问题；
- **条件生成**：Conditional VAE 将标签、文本或其他模态作为条件；
- **离散潜变量**：使用 score-function、Gumbel-Softmax、VQ 等绕过直接重参数化限制。

必须区分“原论文直接提出了什么”与“VAE 家族后来包含什么”。本文的直接贡献是 SGVB、AEVB 及基础高斯 VAE 实例，不包括上述所有后续变体。

## 15. 关键术语表

| 术语 | 含义 |
|---|---|
| Evidence / marginal likelihood | $p_\theta(x)=\int p_\theta(x,z)dz$，观测在模型下的边缘概率 |
| ELBO | 对 $\log p_\theta(x)$ 的变分下界，也是 VAE 的标准训练目标 |
| Recognition model | $q_\phi(z\mid x)$，从观测预测近似后验的推断模型 |
| Probabilistic encoder | recognition model 的自编码器表述，输出分布而非单点 |
| Probabilistic decoder | $p_\theta(x\mid z)$，给定潜变量定义观测分布 |
| SGVB | 用重参数化构造可微 Monte Carlo 变分下界估计器 |
| AEVB | 用 SGVB 联合训练摊销推断模型与生成模型的算法 |
| Reparameterization trick | 将参数化随机采样写成固定噪声经可微变换 |
| Pathwise gradient | 沿 $\epsilon\rightarrow z\rightarrow\mathrm{loss}$ 路径计算梯度 |
| Amortized inference | 用共享网络代替每个样本独立的变分优化 |
| Approximation gap | 变分族受限导致的最优 ELBO 与真实 log evidence 差距 |
| Amortization gap | 共享 encoder 未输出每个样本最优变分参数造成的额外差距 |
| Posterior collapse | 后续研究强调的现象：$q(z\mid x)\approx p(z)$，decoder 忽略潜变量 |

## 16. 图表索引

| 本笔记图号 | 原文对象 | 内容 | 本地文件 |
|---|---|---|---|
| 图 1 | Figure 1 | 生成模型与变分后验的图结构 | `figures/figure-1-graphical-model.png` |
| 图 2 | Algorithm 1 | minibatch AEVB/SGVB 训练流程 | `figures/algorithm-1-aevb.png` |
| 图 3 | Figure 2 | AEVB 与 Wake-Sleep 的 ELBO 收敛比较 | `figures/figure-2-lower-bound-comparison.png` |
| 图 4 | Figure 3 | AEVB、Wake-Sleep、MCEM 的边缘似然比较 | `figures/figure-3-marginal-likelihood.png` |

四张图均直接从本地保存的 arXiv 官方 PDF 以 300 DPI 裁取，保留完整标题、图例、坐标轴或图注，并已逐张复核。页码、裁取区域与文件校验信息见 [[assets/VAE - 变分推断与重参数化生成模型/sources.md|sources.md]]。

## 17. 精读结论

VAE 的历史意义不在某个特定 MLP 架构，而在于它为“神经网络参数化的连续隐变量模型”建立了一个极简、可扩展的训练接口：从固定噪声采样，运行可微的 encoder–decoder 计算图，最大化具有概率意义的变分下界。

它的局限也来自同一框架：变分族会限制后验近似，摊销会引入额外误差，ELBO 可能较松，KL 还会与潜变量承载的信息量竞争。理解这些收益与代价，才能把 VAE 看成变分推断和概率建模方法，而不只是一个带随机瓶颈的自编码器。
