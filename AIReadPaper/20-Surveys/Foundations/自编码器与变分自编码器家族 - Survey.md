---
type: survey
status: done
domain: Foundations
title: 自编码器与变分自编码器（VAE）家族综述
year: 2024
source: https://zhuanlan.zhihu.com/p/682709613
tags:
  - Foundations
  - VAE
  - 生成模型
  - 自编码器
created: 2026-08-25
updated: 2026-08-25
---

# 自编码器与变分自编码器（VAE）家族综述

## 基本信息

**报告主题**：从「确定性的压缩-重构」到「概率化的生成」——自编码器（AE）与变分自编码器（VAE）家族的原理脉络

**覆盖范围**：4 种基础自编码器 + 12 种 VAE 及其变体（VAE、CVAE、β-VAE、VQ-VAE、VQ-VAE-2、TD-VAE、AAE、Dynamic VAE、Seq2Seq VAE、Hierarchical VAE、Invariant VAE、Neural Process VAE）

**材料来源**：知乎专栏文章《一文搞懂所有 VAE 模型（4个AE+12个VAE原理汇总）》，作者 APlayBoy，发表于 2024 年

**撰写日期**：2026-08-25

**报告定位**：以「确定性表示 → 概率表示 → 离散表示 → 序列化表示」这条演化主线，梳理 AE/VAE 家族「为什么每个后续变体会出现」，而非按模型逐个罗列

---

## 1. 研究背景与动机

### 1.1 VAE 为什么在今天重新变得重要

随着 Stable Diffusion 在图像生成、Sora 在视频生成上取得的质量与帧率突破，一个看似老旧的组件重新回到了舞台中央——变分自编码器（VAE）。它的价值不在于直接生成内容，而在于**把高维的图片或视频压缩到一个低维、紧凑、可微的潜空间（latent space）**，让后续的扩散模型、自回归模型在这个压缩空间上计算。

这带来两重收益：一是**效率**，在低维潜空间上做采样、去噪、迭代，计算量远小于直接在像素空间操作；二是**质量**，一个训练良好的潜空间本身就是对数据分布的平滑正则，能抑制像素级噪声、保留语义结构。因此，理解 AE 与 VAE 家族，是理解当前主流生成式模型底座（latent diffusion、VQ 系列、序列世界模型）的前置知识。

### 1.2 一条贯穿全文的演化主线

本文把 16 个模型组织成一条清晰的技术脉络，而不是平铺罗列：

- **确定性表示**（Autoencoder → Denoising → Sparse → Contractive）：输入被映射为**一个确定向量** $z$，目标是「压缩后能重构回来」。
- **概率表示**（VAE → CVAE → β-VAE）：输入被映射为**一个分布** $q_\phi(z|x)$，目标是「不仅重构，还能从先验采样生成新样本」。
- **离散表示**（VQ-VAE → VQ-VAE-2）：潜空间从连续向量退化为**离散 codebook 索引**，为自回归生成铺路。
- **序列化表示**（TD-VAE → Dynamic VAE → Seq2Seq VAE → Neural Process VAE）：潜空间被赋予**时间维度与动态结构**，用来建模序列与不确定性。

理解这条主线，就能理解每个变体「补的是什么短板」。

---

## 2. 基础自编码器：确定性的压缩-重构

### 2.1 Autoencoder：一切的原点

Autoencoder 是一种自监督神经网络，核心目标是通过「压缩—重构」学习数据的有效表示。它由一个编码器和一个解码器组成，整体学习的是一个近似的恒等映射：

$$x \approx f(x) = D_{\theta}(E_{\phi}(x))$$

其中编码器把高维输入压到低维隐编码 $z = E_{\phi}(x)$，解码器再从隐编码恢复 $x' = D_{\theta}(z)$。训练的损失函数通常取重构误差，例如均方误差：

$$L_{(\phi,\theta)} = \frac{1}{n}\sum_{i=1}^{n}(x_i - x'_i)^2$$

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/02-autoencoder-arch.jpg)

> 图 1：Autoencoder 的编码器—解码器结构。编码器把高维输入 $x$ 压缩到低维隐编码 $z$，解码器再把它重构回 $x'$。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 3.1 节配图，https://zhuanlan.zhihu.com/p/682709613

文章特别点出一个直觉：Encoder 的功能类似于 PCA 或矩阵分解的降维，但 Autoencoder 用神经网络能捕捉**非线性**关系，且解码器被专门优化用来「从 $z$ 恢复数据」。一个好的中间表示既要能捕捉潜在变量，又要对解压友好。典型应用包括数据降维、特征提取、去噪和生成。

### 2.2 Denoising Autoencoder：靠「破坏-修复」学鲁棒表示

Denoising Autoencoder（DAE）针对 Autoencoder 的一个致命弱点——学恒等函数时容易过拟合。当网络参数量超过数据复杂度，模型可能只是把输入「抄」过去，学不到有意义的表示。

DAE 的思路很朴素：**先破坏输入，再强迫网络恢复原始未破坏的版本**。具体做法是对输入施加随机扰动 $\tilde{x} \sim Corruption(x)$（掩蔽、高斯噪声、椒盐噪声等），训练目标是 $x \approx D_{\theta}(E_{\phi}(\tilde{x}))$，损失仍用 MSE。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/03-denoising-ae.jpg)

> 图 2：Denoising Autoencoder 的训练流程。输入先被随机破坏成 $\tilde{x}$，网络被训练去恢复原始 $x$。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 3.2 节配图，https://zhuanlan.zhihu.com/p/682709613

这个设计的灵感来自人类视觉：即使图像被部分遮挡，人也能识别场景。DAE 正是靠「修复损坏部分」迫使网络捕获输入空间中的统计关系。文章还纠正了一个常见误解——DAE 那种「随机置零」的做法看起来像 dropout，但 DAE 最早于 2008 年提出，比 dropout 论文早了整整 4 年。

### 2.3 Sparse Autoencoder：只让少数神经元说话

Sparse Autoencoder 在隐藏层上施加「稀疏」约束，强制大部分神经元在大多数时间处于非激活状态，以防止过拟合、增强鲁棒性。稀疏性通过「平均激活度」来度量：设稀疏参数 $\rho$（一般很小，如 0.05）代表神经元激活的预期比例，第 $j$ 个神经元的实际平均激活度为 $\hat{\rho}_j = \frac{1}{m}\sum_{i=1}^{m} a_j(x^{(i)})$，希望 $\hat{\rho}_j \approx \rho$。

约束通过 KL 散度惩罚项实现，比较两个伯努利分布（均值分别为 $\rho$ 和 $\hat{\rho}_j$）的差异：

$$KL(\rho \,\|\, \hat{\rho}_j) = \rho\log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}$$

总损失为 $L_{SAE}(x) = L(x) + \beta\sum_{j} KL(\rho \|\hat{\rho}_j)$，其中 $L(x)$ 是重构误差。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/04-sparse-kl.jpg)

> 图 3：稀疏约束的 KL 散度示意——比较均值为 0.25 与均值在 0–1 之间的伯努利分布。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 3.3 节配图，https://zhuanlan.zhihu.com/p/682709613

文章还介绍了 K-Sparse Autoencoder（Makhzani & Frey, 2013）：不再用 KL 惩罚，而是**只保留瓶颈层线性激活中最大的 $k$ 个值，其余置零**，反向传播也只经过这 $k$ 个被激活单元。这种「硬稀疏」比 KL 惩罚更直接。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/05-ksparse-filters.jpg)

> 图 4：K-Sparse Autoencoder 在 MNIST 上用 1000 个隐藏单元学到的不同稀疏度 $k$ 的滤波器。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 3.3 节配图，https://zhuanlan.zhihu.com/p/682709613

### 2.4 Contractive Autoencoder：让表示对输入扰动「不敏感」

Contractive Autoencoder（CAE）的目标同样是学更稳健的表示，但手段不同：它惩罚**编码器对输入的敏感度**，即编码函数关于输入的 Jacobian 矩阵的 Frobenius 范数：

$$||J_h(x)||_F^2 = \sum_i \left(\frac{\partial h_i(x)}{\partial x_i}\right)^2$$

这个惩罚项本质是「学习到的编码关于输入维度的偏导数的平方和」。偏导越小，表示对输入的小扰动越不敏感，从而在小扰动下保持稳定。它与 DAE、Sparse AE 殊途同归——都指向「鲁棒、不过拟合、可泛化」的表示，只是从不同角度（破坏输入、稀疏激活、抑制敏感度）去逼近。

---

## 3. 标准 VAE：从确定向量到概率分布

### 3.1 核心思想：把输入映射成分布

VAE 与前面所有 AE 的本质区别在于：它**不把输入映射到一个固定向量，而是映射到一个概率分布**。这使它不仅能重构，还能生成与训练数据相似的新样本。

假设真实参数为 $\theta^*$，生成一个样本分两步：先从先验 $p_{\theta^*}(z)$ 采样 $z^{(i)}$，再从条件分布 $p_{\theta^*}(x|z=z^{(i)})$ 生成 $x^{(i)}$。优化目标是对数似然：

$$\theta^* = \arg\max_\theta \sum_{i=1}^{n}\log p_\theta(x^{(i)})$$

但 $p_\theta(x^{(i)}) = \int p_\theta(x^{(i)}|z)\,p_\theta(z)\,dz$ 要对所有 $z$ 积分，难以直接计算。于是引入一个近似函数 $q_\phi(z|x)$ 作为「概率编码器」，把输入映射到可能编码的分布。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/06-vae-graphical-model.jpg)

> 图 5：VAE 的贝叶斯网络结构。实线是生成过程 $p_\theta$，虚线是用 $q_\phi(z|x)$ 近似难处理后验 $p_\theta(z|x)$。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.1 节配图，https://zhuanlan.zhihu.com/p/682709613

这个结构里，$p_\theta(x|z)$ 是概率解码器，$q_\phi(z|x)$ 是概率编码器——与 AE 的 decoder/encoder 一一对应，只是把「确定映射」换成了「分布」。

### 3.2 ELBO 推导与「为什么用后向 KL」

我们希望近似后验 $q_\phi(z|x)$ 尽量接近真实后验 $p_\theta(z|x)$。一个关键细节是：**为什么用 $D_{KL}(q_\phi\|p_\theta)$ 而不是反过来的前向 $D_{KL}(p_\theta\|q_\phi)$？**

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/07-kl-forward-backward.jpg)

> 图 6：前向 KL 与后向 KL 的差异。前向 $D_{KL}(P\|Q)$ 会高估 $P$ 的信息（平均行为），后向 $D_{KL}(Q\|P)$ 会低估（mode-seeking，聚焦高概率区域）。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.1 节配图，https://zhuanlan.zhihu.com/p/682709613

文章给出的直觉是：前向 KL 会「高估」$P(z)$ 的信息、倾向于覆盖整个分布；后向 KL 会「低估」、倾向于抓住最可能的那部分（mode-seeking）。VAE 选后向 KL，是因为我们关心的是「采样出的 $z$ 能生成可信样本」，而不是「覆盖所有可能的 $z$」。

从 $D_{KL}(q_\phi(z|x)\|p_\theta(z|x))$ 展开，经过一系列代数整理（核心是利用 $p(z|x)=p(z,x)/p(x)$、$\int q\,dz=1$、$p(z,x)=p(x|z)p(z)$），可以得到：

$$\log p_\theta(x) - D_{KL}(q_\phi(z|x)\,\|\,p_\theta(z|x)) = \mathbb{E}_{z\sim q_\phi(z|x)}\log p_\theta(x|z) - D_{KL}(q_\phi(z|x)\,\|\,p_\theta(z))$$

等式左边正是我们想最大化的「对数似然减去近似误差」。因为 KL 非负，右边就是 $\log p_\theta(x)$ 的一个下界（变分下界，ELBO）。取负号得到 VAE 的损失：

$$L_{VAE}(\theta,\phi) = -\mathbb{E}_{z\sim q_\phi(z|x)}\log p_\theta(x|z) + D_{KL}(q_\phi(z|x)\,\|\,p_\theta(z))$$

第一项是重构损失，第二项是正则项——把近似后验拉向先验 $p_\theta(z)$（通常取标准正态）。最小化 $L_{VAE}$ 等价于最大化生成真实数据概率的下界。

### 3.3 重参数化技巧：让采样可微

损失里的期望项依赖 $z \sim q_\phi(z|x)$，而采样是随机过程，梯度无法直接反传。重参数化技巧把随机变量改写为 $z = \mathcal{T}_\phi(x, \epsilon)$，其中 $\epsilon$ 是独立的辅助随机变量（噪声源），与参数 $\phi$ 无关。

对最常见的对角协方差高斯 $q_\phi(z|x^{(i)}) = \mathcal{N}(z; \mu^{(i)}, \sigma^{2(i)}I)$：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/08-reparameterization.jpg)

> 图 7：重参数化技巧示意。把「从 $q_\phi(z|x)$ 采样」拆成「确定性计算 $\mu,\sigma$」+「从与参数无关的 $\epsilon$ 采样」，使梯度能流过 $\mu,\sigma$。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.1 节配图，https://zhuanlan.zhihu.com/p/682709613

这样随机性被隔离到 $\epsilon$ 里，$\mu$ 和 $\sigma$ 成为可学习的确定性输出，梯度可以正常反传。文章强调，重参数化不只适用于高斯，也适用于其它分布——只要能把采样写成「参数 + 独立噪声」的形式。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/09-vae-gaussian.jpg)

> 图 8：多元高斯分布下的 VAE 完整模型。编码器输出 $\mu,\sigma$，经重参数化采样 $z$，解码器从 $z$ 重构 $x'$。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.1 节配图，https://zhuanlan.zhihu.com/p/682709613

---

## 4. VAE 的关键扩展

### 4.1 Conditional VAE：把条件注入生成过程

CVAE 在编码器和解码器中都引入条件变量 $c$（类标签、部分数据等），让生成可控。编码器输出 $q_\phi(z|x,c)$ 的均值方差 $\mu(x,c), \sigma^2(x,c)$，解码器结合 $z$ 与 $c$ 重构 $x'$。损失为：

$$L(x,x',z,c) = -\mathbb{E}_{z\sim q(z|x,c)}[\log p(x|z,c)] + KL(q(z|x,c)\,\|\,p(z|c))$$

应用场景包括有条件生成、风格迁移和数据增强。

### 4.2 β-VAE：用 β 换取解耦

β-VAE（Higgins et al., 2017）的目标是学**解耦（disentangled）的潜在因子**。它在标准 VAE 的 KL 项前加一个超参数 $\beta$：

$$L_{BETA}(\phi,\theta) = -\mathbb{E}_{z\sim q_\phi(z|x)}\log p_\theta(x|z) + \beta\,D_{KL}(q_\phi(z|x)\,\|\,p_\theta(z))$$

当 $\beta=1$ 时退化为标准 VAE；$\beta>1$ 时对潜空间施加更强约束，鼓励每个潜在维度捕获独立的生成因子（如人脸的微笑、肤色、发色、是否戴眼镜等）。代价是：$\beta$ 过大时重构质量与解耦程度之间存在权衡——约束越强，重构越可能受损。

### 4.3 VQ-VAE：把潜空间离散化

VQ-VAE（van den Oord et al., 2017）的核心创新是**把连续潜空间离散化**，用一个可学习的 codebook 把编码器输出映射到最近的嵌入向量。这使它特别适合 NLP、语音等天然离散的数据，也为后续的自回归生成铺路。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/10-vqvae-arch.jpg)

> 图 9：VQ-VAE 架构。编码器输出 $z_e$，经最近邻量化映射到 codebook 中的 $e_k$（即 $z_q$），解码器从 $z_q$ 重构。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.4 节配图，https://zhuanlan.zhihu.com/p/682709613

设潜嵌入空间为 $\mathbf{e}\in\mathbb{R}^{K\times D}$（$K$ 个 codebook 向量，维度 $D$）。量化过程是最近邻查找：

$$z_q(x) = \mathrm{Quantize}(E(x)) = e_k,\quad k = \arg\min_i \|E(x) - e_i\|_2$$

由于 $\arg\min$ 在离散空间不可微，VQ-VAE 用 **straight-through estimator**：把解码器输入 $z_q$ 的梯度 $\nabla_z L$ 直接复制给编码器输出 $z_e$。损失由三部分组成：

$$L = \|x - D(e_k)\|_2^2 + \|sg[E(x)] - e_k\|_2^2 + \beta\,\|E(x) - sg[e_k]\|_2^2$$

其中 $sg[\cdot]$ 是 stop-gradient 操作符：第一项是重构损失；第二项是 VQ 损失，让 codebook 向量向编码器输出靠拢（不反传编码器）；第三项是承诺损失，让编码器输出向 codebook 向量靠拢（不更新 codebook）。codebook 向量通过指数移动平均（EMA）更新。

### 4.4 VQ-VAE-2：层次化 + 自回归先验

VQ-VAE-2（Razavi et al., 2019）是 VQ-VAE 的升级版，引入**两级层次化结构**：底层 codebook 捕捉局部纹理，顶层 codebook 捕捉全局结构（如物体形状）。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/11-vqvae2-hierarchy.jpg)

> 图 10：VQ-VAE-2 的分层 VQ-VAE 体系结构与多级图像生成。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.5 节配图，https://zhuanlan.zhihu.com/p/682709613

训练分两阶段：第一阶段训练层次化 VQ-VAE，底层较大 codebook 的训练依赖于顶层较小 codebook，使底层不必从零学所有信息；第二阶段在离散 codebook 上学习**先验分布**，用带自注意力机制的自回归模型拟合它，从而可以采样生成新图像。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/12-vqvae2-pipeline.jpg)

> 图 11：VQ-VAE-2 的两阶段算法流程。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.5 节配图，https://zhuanlan.zhihu.com/p/682709613

这个「离散潜空间 + 自回归先验」的组合，正是后来 DALL·E、VQ-GAN 等离散 token 生成路线的重要基础。

### 4.5 TD-VAE：给潜空间加上时间

TD-VAE（Gregor et al., 2019）把 VAE 推广到序列数据，结合状态空间模型与时间差分学习的理念。它维护一个「信念状态」$b_t = belief(x_1,\dots,x_t)$，对过去状态做压缩编码，用来预测未来。

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/13-state-space-model.jpg)

> 图 12：作为马尔可夫链模型的状态空间模型——一系列未观测隐状态 $z$ 决定观测 $x$。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.6 节配图，https://zhuanlan.zhihu.com/p/682709613

其关键能力是**跳跃预测（jumpy prediction）**：不逐帧预测，而是直接从 $t_1$ 时刻的信念状态跳着预测 $t_2 > t_1$ 时刻的状态。目标函数是最大化两个时间戳 $t_1<t_2$ 上的序列 ELBO：

$$J_{t_1,t_2} = \mathbb{E}\left[\log p_D(x_{t_2}|z_{t_2}) + \log p_B(z_{t_1}|b_{t_1}) + \log p_T(z_{t_2}|z_{t_1}) - \log p_B(z_{t_2}|b_{t_2}) - \log p_S(z_{t_1}|z_{t_2},b_{t_1},b_{t_2})\right]$$

![](assets/自编码器与变分自编码器家族%20-%20Survey/figures/14-tdvae-arch.jpg)

> 图 13：TD-VAE 架构。  
> 来源：知乎专栏《一文搞懂所有 VAE 模型》第 4.6 节配图，https://zhuanlan.zhihu.com/p/682709613

TD-VAE 为时间序列预测、序列决策（强化学习）、视频生成和异常检测提供了一种时间连贯的生成框架。

### 4.6 其它六个变体（要点速览）

文章末尾以「概述 + 原理 + 应用」的紧凑形式补充了六个变体：

- **AAE（Adversarial Autoencoder）**：用 GAN 的对抗机制替代 KL 项，强迫潜空间服从某个预定义分布。适合需要精确控制生成分布的任务。
- **Dynamic VAE**：在潜空间中建模时间依赖性，捕捉序列的动态特征，面向视频、音频等时序数据。
- **Seq2Seq VAE**：编码器把输入序列编码成潜在表示，解码器再解码回输出序列，用于机器翻译、文本生成。
- **Hierarchical VAE**：潜空间分层，不同层捕捉不同粒度的特征（如图像的细节与结构）。
- **Invariant VAE**：学对特定变换（旋转、缩放等）不变的潜表示，用于医学影像等需要稳定性的场景。
- **Neural Process VAE**：融合神经过程与 VAE，擅长处理不确定性与复杂数据分布，适合序列预测、函数逼近。

---

## 5. 总结与个人评价

### 5.1 一图看懂家族脉络

| 类别 | 模型 | 核心机制 | 解决的关键问题 |
|---|---|---|---|
| 确定性 AE | Autoencoder | 压缩-重构恒等映射 | 非线性降维与特征提取 |
| | Denoising AE | 破坏输入再修复 | 过拟合、鲁棒性 |
| | Sparse AE | 稀疏激活约束（KL / 硬 k） | 过拟合、可解释性 |
| | Contractive AE | Jacobian 惩罚 | 对输入扰动敏感 |
| 概率 VAE | VAE | 分布映射 + ELBO + 重参数化 | 生成新样本 |
| | CVAE | 条件变量注入 | 可控生成 |
| | β-VAE | 加权 KL | 解耦表示 |
| 离散 VAE | VQ-VAE | codebook 量化 + straight-through | 离散潜空间 |
| | VQ-VAE-2 | 层次化 + 自回归先验 | 高质量图像生成 |
| 序列 VAE | TD-VAE | 信念状态 + 跳跃预测 | 时序建模 |
| | 其余 6 种 | AAE/Dynamic/Seq2Seq/Hierarchical/Invariant/NP | 分布控制、时序、结构化、不变性、不确定性 |

### 5.2 个人判断

这篇知乎文章的价值在于**把散落在各篇论文里的 AE/VAE 变体串成一条可读的线索**，尤其 ELBO 推导和「前向/后向 KL 的选择」这两处讲得清楚，适合作为 VAE 家族的第一张地图。但它有几个需要注意的边界：

1. **它是二手科普，不是论文原文**。公式推导只保留了骨架，省略了具体假设（如 ELBO 推导中 $q_\phi$ 的因子分解、TD-VAE 里 $p_B/p_T/p_S$ 三个条件分布的严格定义）。若要真正实现或复现，应回到原论文（Kingma & Welling 2014；van den Oord 2017；Razavi 2019；Gregor 2019）。
2. **12 个 VAE 里「其它六个」只有定性描述**，没有公式与实验，属于「点到为止」。真正要理解 AAE、Neural Process 等，需要另找专门材料。
3. **缺少量化对比与失败案例**。文章没有给出各模型的重构质量、生成质量、解耦度等指标，也几乎不提「后验坍缩（posterior collapse）」这类 VAE 训练中的经典难题。
4. 文中的符号术语图（图 0）因手机端 LaTeX 显示问题做成了截图，公式部分有少量排版瑕疵，阅读时以正文公式为准。

对后续工作的启发：这条「确定性 → 概率 → 离散 → 序列」的主线，恰好对应了当前生成模型底座的技术栈——VAE 是 latent diffusion 的压缩器，VQ 是离散 token 生成的前置，TD-VAE 式的信念状态是序列世界模型的雏形。把 VAE 家族吃透，是理解这些更大系统的可靠起点。

---

## 6. 材料来源

- 知乎专栏原文：《一文搞懂所有 VAE 模型（4个AE+12个VAE原理汇总）》，作者 APlayBoy，https://zhuanlan.zhihu.com/p/682709613
- 文中关联阅读：变分推断、VAE 精读、VQ-VAE 精读、VQ-VAE-2 精读（见原文「相关阅读」小节）
- 本笔记所有配图均取自原文专栏，原始 URL 记录于 `sources.md`

> 本综述基于二手科普文章整理，用于建立 VAE 家族的脉络地图；如需严谨推导与实现细节，请以各原论文为准。
