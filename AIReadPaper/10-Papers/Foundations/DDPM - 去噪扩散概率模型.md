---
title: DDPM - 去噪扩散概率模型
aliases:
  - Denoising Diffusion Probabilistic Models
  - DDPM
authors:
  - Jonathan Ho
  - Ajay Jain
  - Pieter Abbeel
year: 2020
venue: NeurIPS 2020
arxiv: "2006.11239"
tags:
  - paper
  - diffusion-model
  - generative-model
  - score-matching
status: 精读完成
---

# 0. 论文信息

> **论文**：Denoising Diffusion Probabilistic Models  
> **作者**：Jonathan Ho, Ajay Jain, Pieter Abbeel  
> **会议**：NeurIPS 2020，Vol. 33，pp. 6840–6851  
> **arXiv**：[2006.11239](https://arxiv.org/abs/2006.11239)  
> **官方代码**：[hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)  
> **本地材料**：[[assets/DDPM - 去噪扩散概率模型/DDPM - Denoising Diffusion Probabilistic Models.pdf|官方 PDF]] · [[assets/DDPM - 去噪扩散概率模型/sources.md|来源与图表核验]]  
> **精读日期**：2026-10-16

# 1. 一句话总结

DDPM 把生成建模改写成一件极其朴素的事：先用一个固定的高斯马尔可夫链把真实图像缓慢毁成标准高斯噪声，再训练一个共享参数的 U-Net 在每个噪声等级预测“这一步里混入了什么噪声”，最后从噪声出发反复去噪。它真正奠基性的地方不只是“能从噪声生成图像”，而是把**变分推断、噪声预测、denoising score matching 与 Langevin-like 采样**压进了同一个可训练框架，并用一个去掉 ELBO 时间权重的简单均方误差换得了当时极强的样本质量。

# 2. 问题意识：为什么 2020 年还需要另一种生成模型

## 2.1 三条既有路线的结构性取舍

2020 年前后的图像生成大致存在三种典型交换。GAN 的采样只需一次前向传播，视觉质量高，却没有显式似然，训练又是一个容易失衡的双人博弈；自回归模型给出清楚的似然分解，但逐像素生成慢，且固定的坐标顺序把归纳偏置写死；VAE 与 normalizing flow 拥有概率语义，却常在感知质量、架构约束或计算量上付出代价。

扩散概率模型并非本文首创。Sohl-Dickstein 等人在 2015 年已经提出：设计一个逐步加噪的前向过程，再学习其时间反演，就可以把复杂数据分布连接到简单先验。然而这条路线长期更像一个理论上优雅、生成质量却不够有说服力的旁支。DDPM 要回答的问题很具体：**若把扩散模型的参数化、损失权重和网络架构重新整理，它能否生成真正高质量的自然图像？**

## 2.2 论文最重要的判断

论文的关键判断是，困难不在于“逆扩散是否存在”，而在于用什么量来监督逆过程。直接让网络回归一步后验均值当然可行，但这个目标把易学的线性部分和真正需要模型推断的部分混在一起。作者把逆过程均值改写为“已知的线性校正 + 网络预测的噪声”，于是每一个时间步都变成同一种去噪任务；网络不必直接发明图像，只需估计当前样本里哪一部分像高斯噪声。

![[assets/DDPM - 去噪扩散概率模型/figure-2-directed-graphical-model.png]]

**图 1（原文 Figure 2）**：虚线是固定的前向加噪链 $q(x_t\mid x_{t-1})$，实线是学习到的反向去噪链 $p_\theta(x_{t-1}\mid x_t)$。图中最值得注意的不是链很长，而是正反两个局部转移在足够小的噪声步长下都可用高斯条件分布表达。

# 3. 核心贡献：论文究竟新增了什么

1. **把扩散模型推进到高质量图像生成。** 论文在无条件 CIFAR-10 上得到 IS 9.46、FID 3.17，并在 $256\times256$ LSUN 上达到接近 ProgressiveGAN 的视觉质量，证明扩散不再只是可计算似然的概念模型。
2. **提出并验证 $\epsilon$-prediction 参数化。** 网络预测前向过程中加入的标准高斯噪声，而不是直接预测逆过程均值；这一参数化把逆扩散与多噪声尺度的 denoising score matching 显式连接起来。
3. **用 $L_{\mathrm{simple}}$ 重加权变分目标。** 作者删去严格 ELBO 中随时间变化的系数，对随机时间步上的噪声预测误差做普通 MSE。它牺牲了一些无损似然，却显著改善感知样本质量。
4. **给出可直接复现的训练与采样算法。** 任意 $x_t$ 都能从 $x_0$ 一步采样，因此训练不必真的跑完整个前向链；采样则从 $x_T\sim\mathcal N(0,I)$ 开始执行 1000 次共享网络的反向更新。
5. **揭示扩散链的渐进编码结构。** 大尺度语义先出现、细节后出现；论文用 rate–distortion、随机解码和潜变量插值说明，中间状态 $x_t$ 是按噪声尺度组织的信息瓶颈，而不只是“坏掉的图片”。

# 4. 方法总览：把复杂分布连接到标准高斯

设真实数据为 $x_0\sim q(x_0)$，并引入与图像同维的潜变量 $x_1,\dots,x_T$。整个模型包含两条方向相反的链：前向链 $q$ 是人为指定、不可学习的“编码器”；反向链 $p_\theta$ 是生成模型。

## 4.1 前向链：每一步只加入一点高斯噪声

作者选取方差日程 $\beta_1,\dots,\beta_T$，定义：

$$q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1})$$

$$q(x_t\mid x_{t-1})=\mathcal N\!\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right)$$

记 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$。连续展开高斯转移后，可以越过中间节点，直接从干净图像采样任意噪声等级：

$$q(x_t\mid x_0)=\mathcal N\!\left(x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\right)$$

等价的重参数化形式是：

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$

这个闭式公式是 DDPM 训练高效的根源。一次训练只需抽取一个 $t$ 和一个 $\epsilon$，不需要把 $x_0$ 真的加噪 $t$ 次。$\sqrt{\bar\alpha_t}$ 是剩余信号幅度，$\sqrt{1-\bar\alpha_t}$ 是累计噪声幅度；两者平方之和为 1，使各时间步的输入尺度基本稳定。

在原论文所有实验中，$T=1000$，$\beta_t$ 从 $10^{-4}$ 线性增长到 $0.02$。终点被安排得足够接近标准高斯：论文报告 $D_{\mathrm{KL}}(q(x_T\mid x_0)\|\mathcal N(0,I))\approx10^{-5}$ bits/dim。于是采样时直接从标准高斯出发带来的 train–sample distribution shift 很小。

## 4.2 反向链：学习把噪声逐步还原为数据

生成模型定义为：

$$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t),\qquad p(x_T)=\mathcal N(0,I)$$

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N\!\left(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\right)$$

当单步 $\beta_t$ 很小时，真实的反向条件也接近高斯，因此局部高斯近似并不苛刻。原论文没有让网络自由学习全部协方差，而是令 $\Sigma_\theta(x_t,t)=\sigma_t^2I$，并比较 $\sigma_t^2=\beta_t$ 与 $\sigma_t^2=\tilde\beta_t$；两者实验表现相近。作者最终把主要建模能力集中在均值上。

## 4.3 已知 $x_0$ 时，一步后验也是闭式高斯

变分训练需要把模型反向转移与前向过程的真实后验比较。条件在 $x_0$ 上时，这个后验可以精确计算：

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N\!\left(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I\right)$$

$$\tilde\mu_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

$$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

这意味着训练时比较的是两个高斯分布，KL 可以闭式计算，而不是用高方差 Monte Carlo 估计。要区分两个事实：$q(x_{t-1}\mid x_t,x_0)$ 可解，是因为训练时知道原图；真正生成时没有 $x_0$，所以必须用神经网络从 $x_t$ 估计其所需信息。

# 5. ELBO：长链如何拆成可训练的局部任务

## 5.1 从负对数似然到变分上界

模型的边缘似然需要积分掉 $x_{1:T}$，直接求解困难。以前向链作为近似后验，可得负对数似然的变分上界：

$$\mathbb E_{q(x_0)}[-\log p_\theta(x_0)]\leq\mathbb E_q\!\left[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]\equiv L$$

通过把马尔可夫因子重新排列，论文将它拆成三类语义清楚的项：

$$L=\mathbb E_q\!\left[\underbrace{D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))}_{L_T}+\sum_{t=2}^{T}\underbrace{D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t))}_{L_{t-1}}+\underbrace{[-\log p_\theta(x_0\mid x_1)]}_{L_0}\right]$$

更不容易混淆的写法是把末项单独命名：

$$L_T=D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T)),\quad L_{t-1}=D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)),\quad L_0=-\log p_\theta(x_0\mid x_1)$$

三部分各司其职。$L_T$ 要求扩散终点匹配标准高斯；因为 $\beta_t$ 固定，它对 $\theta$ 是常数。中间的 $L_{1:T-1}$ 逐步拟合逆转移，是主要训练信号。$L_0$ 则把连续高斯输出转成离散像素概率，是严格 likelihood/codelength 解释的落点。

## 5.2 为什么局部 KL 最终是均值回归

固定反向方差 $\sigma_t^2I$ 后，每一个中间 KL 关于模型参数的部分都化为：

$$L_{t-1}=\mathbb E_q\!\left[\frac{1}{2\sigma_t^2}\left\|\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\right\|^2\right]+C$$

$C$ 与 $\theta$ 无关。最直接的方法是让网络预测 $\tilde\mu_t$，但作者继续代入 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，发现真实后验均值可以写成：

$$\tilde\mu_t(x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$$

因此，反向均值可以参数化为：

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

从学习问题看，$x_t$ 这一线性项网络已经拿在手里，真正未知的是造成当前污染的 $\epsilon$。这就是噪声预测不是一个经验技巧，而是由后验均值代数变形导出的原因。

## 5.3 严格 ELBO 中的噪声预测权重

把上述参数化代回局部 KL，得到：

$$L_{t-1}-C=\mathbb E_{x_0,\epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\left\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\right\|^2\right]$$

因此“预测噪声的 MSE”本来就藏在 ELBO 里，只是带有依赖 $t$ 的权重。权重体现 likelihood 的偏好：对像素级无损编码重要的低噪声细节，会得到与感知生成不一定一致的强调。

# 6. $L_{\mathrm{simple}}$：故意不忠于似然，反而生成得更好

作者最终采用均匀抽取时间步、去掉所有时间权重的目标：

$$L_{\mathrm{simple}}(\theta)=\mathbb E_{t,x_0,\epsilon}\!\left[\left\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\right\|^2\right],\qquad t\sim\mathrm{Uniform}\{1,\dots,T\}$$

论文称它为 simplified objective，但它并非与 ELBO 毫无关系。准确地说，它是**对 ELBO 各时间项重新加权后的目标**：删去严格系数，令各噪声等级在采样频率上等权。就本文的线性 $\beta$ 日程而言，这会相对降低小 $t$、低噪声、细微像素修复任务的影响，使有限模型容量更多投入到高噪声下的结构恢复。

这揭示了论文中一个容易被摘要遮住的张力：

- 直接优化 $L$，CIFAR-10 的 NLL 更好，达到不超过 3.70 bits/dim；
- 优化 $L_{\mathrm{simple}}$，NLL 略差到不超过 3.75 bits/dim，却把 FID 从 13.51 大幅降到 3.17。

所以 DDPM 的高样本质量不是“最大似然自然带来的副产品”，而是一次明确的率失真取舍：少花能力编码人眼几乎不可见的残差，多花能力恢复决定感知质量的结构。

# 7. 与 denoising score matching 的关系

## 7.1 条件 score 与噪声只有一个比例因子

对前向扰动核 $q(x_t\mid x_0)$ 求关于 $x_t$ 的对数梯度：

$$\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

因此噪声预测器可以直接换算为 score 估计器：

$$s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$$

这里的 score 是带噪分布在当前位置指向高密度区域的梯度场。预测 $\epsilon$ 看似在回归一次随机采样的噪声，MSE 的最优解实际上是条件期望 $\mathbb E[\epsilon\mid x_t]$；用 denoising score matching 的恒等式，它等价于带噪边缘分布 $q_t(x_t)$ 的 score：

$$\nabla_{x_t}\log q_t(x_t)=-\frac{\mathbb E[\epsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}}$$

这也解释了为什么网络不能把每个随机 $\epsilon$ “背出来”，却仍能学到有用方向：它学习的是在数据先验约束下，给定当前噪声图像时平均应减去的噪声，也就是把样本推回该噪声尺度数据流形的向量场。

## 7.2 采样为什么像 annealed Langevin dynamics

DDPM 的一步采样为：

$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)+\sigma_t z$$

把 $\epsilon_\theta=-\sqrt{1-\bar\alpha_t}\,s_\theta$ 代入，可见均值更新含有沿 score 方向移动的项，同时再注入方差为 $\sigma_t^2$ 的噪声。这在形式上类似逐渐降低噪声尺度的 Langevin dynamics。不同之处也很重要：NCSN 一类方法先学习多尺度 score，再在训练后手工选择退火 Langevin 的步长与噪声；DDPM 则从同一前向 $\beta_t$ 日程推导训练权重和采样系数，用变分推断直接训练这条有限步采样链。

所以最准确的表述不是“DDPM 就是 Langevin dynamics”，而是：**特定的 $\epsilon$ 参数化使 DDPM 的离散反向链呈现 Langevin-like 更新，而其加权噪声 MSE 等价于多噪声尺度的 denoising score matching 目标。**

## 7.3 原论文没有做到的连续时间统一

本文工作在固定的离散马尔可夫链上，$t\in\{1,\dots,1000\}$。把它写成 variance-preserving SDE、推导 reverse-time SDE 与 probability-flow ODE，是之后 Score SDE 框架完成的统一；把同一训练目标配上非马尔可夫、可确定性采样路径，是 DDIM 的贡献。阅读时不应把这些后来的连续时间结论倒写成 DDPM 原文已经提出的内容。

# 8. 训练与采样算法：最小实现需要什么

## 8.1 训练

一次迭代只做四件事：从数据集取 $x_0$；均匀抽取 $t$；采样标准高斯 $\epsilon$；构造 $x_t$ 后最小化噪声 MSE。

$$x_0\sim q(x_0),\qquad t\sim\mathrm{Uniform}\{1,\dots,T\},\qquad \epsilon\sim\mathcal N(0,I)$$

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

$$\theta\leftarrow\theta-\eta\nabla_\theta\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2$$

时间 $t$ 通过 Transformer 式正弦位置编码注入每个残差块。一个网络共享全部时间步参数，而不是训练 1000 个独立去噪器；时间条件告诉它当前应处理的是近净图细修，还是近纯噪声的全局结构恢复。

## 8.2 采样

首先从标准高斯采样 $x_T$，随后逆序运行所有时间步：

$$x_T\sim\mathcal N(0,I)$$

$$z\sim\mathcal N(0,I)\ \text{if}\ t>1,\qquad z=0\ \text{if}\ t=1$$

$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)+\sigma_t z$$

最后一步不再加噪，返回 $x_0$。这段算法的工程缺点同样直接：原版每张图需要 1000 次串行网络评估，无法像 GAN 那样一次前向完成，也难以把时间维度并行化。

## 8.3 $L_0$ 与离散像素似然

图像先从整数 $\{0,\dots,255\}$ 线性映射到 $[-1,1]$。若只关心采样，最后输出 $\mu_\theta(x_1,1)$ 即可；若要报告离散数据的 bits/dim，论文把末端高斯在每个像素 bin 上积分，形成独立的 discretized Gaussian decoder。这个细节保证 ELBO 真能解释为离散图像的无损码长，而不是对任意 dequantization 约定的连续密度。

# 9. 网络、数据与实验协议

## 9.1 结构

骨干是基于 Wide ResNet 的 U-Net，风格接近去掉掩码的 PixelCNN++：

- $32\times32$ 模型使用四个特征分辨率，从 $32\times32$ 到 $4\times4$；$256\times256$ 模型使用六个分辨率。
- 每个分辨率有两个卷积残差块，全网使用 GroupNorm。
- 在 $16\times16$ 特征图分辨率加入 self-attention。
- 正弦时间嵌入注入每个残差块。
- CIFAR-10 模型 35.7M 参数；LSUN/CelebA-HQ 模型 114M；大型 LSUN Bedroom 模型约 256M。

这套设计在今天看来像扩散 U-Net 的标准模板，但当时的重要性在于：它把 PixelCNN++ 的多尺度卷积归纳偏置、Transformer 时间编码与局部注意力组合成一个稳定的多噪声尺度去噪器。

## 9.2 数据、日程与优化

实验覆盖无条件 CIFAR-10、CelebA-HQ $256^2$ 以及 LSUN Bedroom/Church/Cat $256^2$。所有模型都用 $T=1000$ 和线性 $\beta$ 日程。CIFAR-10 的 batch size 为 128、learning rate 为 $2\times10^{-4}$、dropout 为 0.1；大图 batch size 为 64、learning rate 为 $2\times10^{-5}$、dropout 为 0。优化器是 Adam，EMA decay 为 0.9999。CIFAR-10 与多数数据使用随机水平翻转，LSUN Bedroom 除外。

作者并没有对每个数据集重新大规模搜索超参数：主要在 CIFAR-10 上选择配置，再迁移到其他数据集。最终实验只训练一次，并在训练过程中持续评估；主表报告训练轨迹上的最小 FID。这一点有利于避免“多次训练挑最好 seed”，但按当代更严格标准看，单次训练与沿时间挑最佳 checkpoint 仍不足以完整呈现方差。

# 10. 主结果：高感知质量与普通似然并存

## 10.1 CIFAR-10

无条件 CIFAR-10 上，$L_{\mathrm{simple}}$ 模型得到 **IS $9.46\pm0.11$、FID 3.17**。论文按当时惯例相对训练集计算 FID；相对测试集计算时 FID 为 5.24。这个 3.17 优于表中绝大多数无条件模型，甚至优于若干条件模型，但不要忽略同表 StyleGAN2+ADA v1 的条件结果 FID 2.67，也不要把不同条件设定和评估协议混成一个绝对排行榜。

likelihood 方面，$L_{\mathrm{simple}}$ 的测试 NLL 不超过 3.75 bits/dim，而 Sparse Transformer 为 2.80 bits/dim。DDPM 的突破因此是**样本质量**，不是 likelihood SOTA。严格 ELBO 模型的 NLL 更好，但生成样本更差，再次说明像素级码长与感知质量并不等价。

## 10.2 LSUN

原文附录报告 $256\times256$ LSUN FID：Bedroom 小模型 6.36、大模型 4.90；Church 7.89；Cat 19.75。相较 ProgressiveGAN，Bedroom 与 Church 有竞争力；相较 StyleGAN/StyleGAN2，DDPM 并未全面领先，尤其 Cat 的差距明显。论文自己的准确主张是“与 ProgressiveGAN 类似”，而不是“在所有高分辨率数据集超越 GAN”。

## 10.3 最近邻与过拟合

CIFAR-10 训练集与测试集的 lossless codelength 差距至多 0.03 bits/dim，最近邻图也没有显示简单复制。它支持“模型不是靠记住训练样本取得 FID”的判断，但最近邻检查不是严格的隐私或记忆审计，不能据此推出模型绝无训练数据记忆。

# 11. 消融：真正有效的是哪一个组合

![[assets/DDPM - 去噪扩散概率模型/table-2-ablation.png]]

**图 2（原文 Table 2）**：消融同时改变逆过程目标的参数化、训练目标和方差处理。最强结果不是单独来自“预测噪声”，而来自 **$\epsilon$-prediction + 固定各向同性方差 + 无时间权重的 $L_{\mathrm{simple}}$**。

逐项看：

1. 预测后验均值 $\tilde\mu$，用严格 $L$ 和固定方差，可得 IS 8.06、FID 13.22；若直接对均值做无权 MSE，训练不稳定，未给出有效结果。
2. 预测 $\epsilon$，用严格 $L$ 和固定方差，可得 IS 7.67、FID 13.51，与均值参数化大致同级。由此不能说“换成噪声预测本身立刻把 FID 降到 3.17”。
3. 预测 $\epsilon$ 并采用 $L_{\mathrm{simple}}$，才得到 IS 9.46、FID 3.17。决定性增益来自参数化与重加权之间的配合。
4. 在原论文设置中学习对角方差导致更差或不稳定的训练；固定方差更可靠。但这不是扩散模型的永久定律，Improved DDPM 后来用方差重参数化和混合目标成功学习方差，并显著减少采样步数。

消融的不足也很明确：表格只有 CIFAR-10，且没有报告多 seed 的训练方差、不同 $T$、更丰富噪声日程或网络容量交互。它足以支持本文配方，却不足以穷尽设计空间。

# 12. 渐进生成：中间状态到底保存了什么

## 12.1 从粗到细不是视觉偶然，而是噪声尺度的结果

![[assets/DDPM - 去噪扩散概率模型/figure-6-progressive-generation.jpg]]

**图 3（原文 Figure 6）**：从左到右，模型对最终 $x_0$ 的预测先确定大尺度颜色与轮廓，再补物体类别相关结构，最后收紧纹理和边缘。低频、全局结构在高噪声阶段仍有统计可辨识性，细节则必须等到信噪比升高后才能决定。

论文把 $L_1+\cdots+L_T$ 解释为 rate，把 $L_0$ 解释为 distortion。最佳样本模型的 rate 为 1.78 bits/dim，distortion 为 1.97 bits/dim，对应 $[0,255]$ 尺度 RMSE 0.95。超过一半无损码长都用于描述视觉上几乎不可察觉的失真。这正是 $L_{\mathrm{simple}}$ 改善 FID、恶化 NLL 的信息论解释。

但作者也明确承认，文中的 progressive compression 只是概念证明：它依赖 minimal random coding 一类在高维数据上并不实用的过程。DDPM 给出了理解 ELBO 的压缩视角，并没有交付一个实用图像编解码器。

## 12.2 随机解码暴露了层级语义

![[assets/DDPM - 去噪扩散概率模型/figure-7-stochastic-decoding.jpg]]

**图 4（原文 Figure 7）**：对同一个中间潜变量 $x_t$ 分叉运行多条反向链。共享 $x_{1000}$ 时结果差异很大；分叉发生得越晚，样本越共享姿态、发色、眼镜和表情等高层属性，只在细节上变化。

这里不能简单说“某个时间步显式编码了发色”。$x_t$ 仍是与图像同维、被高斯噪声污染的连续张量，没有像 VAE 那样预先分配语义坐标。更准确的说法是：给定 $x_t$ 后，反向条件分布的不确定性随时间缩小；不同尺度的可恢复信息呈现从全局到局部的层级。

## 12.3 插值展示流形投影能力

![[assets/DDPM - 去噪扩散概率模型/figure-8-interpolation.jpg]]

**图 5（原文 Figure 8）**：两个真实图像先扩散到 $t=500$，在带噪潜空间线性插值，再由反向链去噪。相比像素空间直接叠影，反向过程把插值点拉回自然图像流形，姿态、肤色、发型、表情和背景平滑变化。

这里的“latent”不同于低维语义编码：它仍与像素等维，且编码是随机的。$t$ 控制的是保留多少源图信息。$t$ 小时插值偏细粒度，$t$ 大时模型必须补全更多结构；到 $t=1000$，源信息几乎消失，插值更接近新样本。

# 13. 局限与批判性阅读

## 13.1 采样极慢，而且天然串行

原版 DDPM 每个样本需要 1000 次 U-Net 评估。论文在 TPU v3-8 上，生成一批 256 张 CIFAR-10 图像需 17 秒；生成一批 128 张 $256^2$ 图像需 300 秒。批内可以并行，时间步之间却有严格依赖。后续 DDIM、Improved DDPM、数值 ODE/SDE 求解器、EDM 与蒸馏路线之所以重要，很大程度上都在处理这一瓶颈。

## 13.2 likelihood 不是强项

$L_{\mathrm{simple}}$ 主动偏离严格变分界，换来视觉质量。这是成功的工程选择，但也意味着“DDPM 是显式 likelihood-based model”不能自动推出它在密度估计上有优势。原文 3.75 bits/dim 明显落后于 Sparse Transformer 的 2.80；而高 NLL 中相当一部分来自难以感知的像素细节。

## 13.3 方差与噪声日程仍是手工设计

原文固定反向方差，并选用线性 $\beta$ 日程。这个选择简单稳定，却把不确定性结构压成各向同性标量，也没有证明线性 schedule 最优。后续 learned variance、cosine schedule、连续噪声参数化和 EDM preconditioning 都说明这些不是理论必然，而是第一代成功配方。

## 13.4 实验范围窄于今天对“通用生成”的理解

实验是无条件小图与单域 $256^2$ 图像；没有文本条件、复杂多域数据、分辨率扩展、组合泛化或人工偏好评估。CIFAR-10 FID 证明核心机制有效，却不能单独支撑开放域高分辨率生成。后来真正把扩散推向大规模应用的，是条件引导、潜空间压缩、大数据和更大网络的共同作用。

## 13.5 指标与协议存在时代局限

主 FID 按训练集计算，checkpoint 取训练轨迹中的最小 FID，最终训练仅一次。论文透明说明了这些做法，但按今天的复现要求，最好报告固定评估预算、多随机种子置信区间、precision/recall、采样 NFE 与墙钟时间，并严格统一参考集和特征提取实现。

## 13.6 生成模型的一般风险没有因训练稳定而消失

扩散避免了 GAN 的对抗训练不稳定，却仍会继承数据集偏差、产生误导性合成内容，并可能记忆训练样本。论文的最近邻实验只能部分缓解“纯复制”疑问，不能替代数据治理、隐私测试、内容来源标记与部署约束。

# 14. 复现成本：从“能跑”到“复现论文数字”

## 14.1 原论文给出的硬成本

所有实验使用 TPU v3-8，作者称其大致相当于 8 张 V100。CIFAR-10 模型在 batch size 128 下约 21 step/s，训练 800k steps，约 10.6 小时。$256^2$ 模型在 batch size 64 下约 2.2 step/s：CelebA-HQ 0.5M steps，约 63 小时；LSUN Church 1.2M steps，约 152 小时；LSUN Cat 1.8M steps，约 227 小时；LSUN Bedroom 2.4M steps，约 303 小时。256M 参数的大 Bedroom 模型训练 1.15M steps，成本仍更高。

这些只是由原文吞吐线性换算出的单次训练墙钟量级，不含数据预处理、周期性生成 50k 样本、FID 评估、失败实验和超参数搜索。若目标是复现整张论文表而非训练一个模型，真实预算会显著增加。

## 14.2 软件与数据摩擦

官方实现基于 TensorFlow 1.15、Python 3.5、TPU 与 Google Cloud Storage 路径约定。今天照原仓库逐字复现会遇到旧依赖、已变化的数据接口与指标实现。使用现代 PyTorch 复现更容易运行，却必须逐项对齐：数据缩放到 $[-1,1]$、linear $\beta$、$T=1000$、固定方差、U-Net 通道数、GroupNorm、$16^2$ attention、时间嵌入、dropout、EMA 0.9999、训练步数以及 FID 参考集。

## 14.3 三档可行复现目标

- **机制复现**：在 MNIST 或 CIFAR-10 子集上验证 $q(x_t\mid x_0)$、噪声 MSE 和反向采样，单卡即可；目标是看见从噪声到结构的过程，不追论文 FID。
- **CIFAR-10 结果复现**：完整 35.7M U-Net、800k steps、EMA、50k 样本 FID；需要高性能 GPU 和较长训练，且采样评估本身不可忽略。
- **整论文复现**：再训练多个 114M/256M 的 $256^2$ 模型并完成消融、最近邻、插值、rate–distortion。按原始方案属于多设备、多周级项目，不适合作为“顺手跑一下”的基线。

最小正确性检查应先于大训练：验证由 $x_0$ 闭式采样得到的 $x_t$ 与逐步前向采样统计一致；检查 $\bar\alpha_t$ 单调下降且终点近零；确认最后一步 $z=0$；可视化不同 $t$ 的噪声强度；在小批样本上确认预测噪声的尺度和反向均值公式没有符号错误。

# 15. 后世影响与边界：DDPM 留下了什么，后来又改变了什么

## 15.1 真正被保留下来的核心

后续模型频繁更换网络、采样器、噪声坐标甚至状态空间，但 DDPM 的三个核心至今仍在：从已知噪声核构造不同信噪比的训练对；用时间/噪声条件网络学习去噪场；从简单先验沿逆向路径恢复数据。$\epsilon$-prediction 仍是常见目标，尽管 $x_0$-prediction、score prediction 和 $v$-prediction 在不同参数化下也很常见。

## 15.2 DDIM：训练目标不等于唯一采样路径

DDIM 证明，可以构造与 DDPM 共享边缘 $q(x_t\mid x_0)$ 和训练目标的非马尔可夫过程，并使用确定性或低随机性的更短轨迹采样。这划清了第一条边界：DDPM 的 1000 步随机反向链是一个可行生成过程，不是噪声预测模型唯一合法的解码器。

## 15.3 Score SDE：离散链只是连续扩散的一种离散化

Score SDE 把噪声等级连续化，用 reverse-time SDE 统一 DDPM 与 score-based model，并给出 probability-flow ODE。它让采样器设计与数值求解器连接起来，也能进行 likelihood 计算。DDPM 原文已经指出 score/Langevin 联系，但没有给出这套连续时间统一。

## 15.4 Improved DDPM：原文中失败的 learned variance 并非死路

Improved DDPM 通过插值式方差参数化、$L_{\mathrm{simple}}$ 与 VLB 的混合损失以及时间步重要性采样，获得更好的 likelihood，并在较少反向步下保持样本质量。这说明原论文消融的结论应限定为“在其参数化和训练目标下，学习对角方差不稳定”，而不是“扩散模型永远应固定方差”。

## 15.5 Guidance 与 Latent Diffusion：现代文生图的关键能力不是原论文已有

Classifier guidance 与 classifier-free guidance 把条件信号转化为对 score 的引导，允许在保真度和多样性间调节；Latent Diffusion 把扩散从像素空间迁移到预训练自编码器的压缩潜空间，并用 cross-attention 注入文本等条件。这些技术继承 DDPM 的去噪训练骨架，却分别解决可控性和高分辨率成本问题。把 Stable Diffusion 等系统直接等同于原始 DDPM，会漏掉最关键的条件机制、表示空间和规模化工程。

## 15.6 EDM 与现代求解器：理论包装中的设计自由度被显式拆开

EDM 将噪声参数化、网络预条件、训练权重、时间离散和采样算法拆成相对独立的设计轴，并把采样降到几十次网络评估。它继承的是 DDPM 揭示的多噪声尺度去噪场，而不是原版线性 $\beta$ 与 1000 步 ancestral sampler 的全部细节。

## 15.7 更宽的历史定位

DDPM 不是扩散思想的起点，也不是现代扩散系统的终点。Sohl-Dickstein 等人给出扩散概率模型；NCSN 展示多噪声尺度 score matching；DDPM 的历史作用，是找到一个极简而有效的交汇点：**固定高斯前向链、U-Net、噪声预测和重加权目标共同把理论可行性变成了高质量生成。** 后来的工作大多围绕四个问题展开：更快采样、更好条件控制、更低训练/推理成本、更清晰的连续时间与参数化设计空间。

# 16. 阅读结论：应当怎样记住 DDPM

如果只把 DDPM 记成“不断加噪，再不断去噪”，会错过论文最深的部分。它的关键不是噪声动画，而是一个精确的代数重排：前向高斯链使任意噪声等级和一步后验可解；ELBO 把全局生成问题分解为局部高斯匹配；后验均值又可重写成噪声预测；噪声预测进一步等价于多尺度 score 学习。最终，一条 1000 步的生成链被训练成许多共享参数的局部去噪任务。

同时也要记住，论文最好的图像来自对严格 likelihood 目标的主动偏离。$L_{\mathrm{simple}}$ 把模型容量从几乎不可见的低噪声残差移向更困难、更决定感知的结构恢复。DDPM 因而不仅是一篇“新模型”论文，也是一篇关于生成建模评价目标的论文：**更短的无损码长并不必然意味着更好的样本，而一个看似更粗糙的重加权目标，可能更符合人类视觉关心的信息层级。**

# 17. 关键术语速查

- **Forward process / diffusion process**：固定的逐步高斯加噪马尔可夫链 $q$。
- **Reverse process**：从标准高斯出发、由网络参数化的逐步去噪链 $p_\theta$。
- **Noise schedule**：每一步噪声方差 $\beta_t$ 的序列；原文使用从 $10^{-4}$ 到 $0.02$ 的线性日程。
- **$\bar\alpha_t$**：累计信号保留率，$\bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s)$。
- **ELBO / VLB**：对负对数似然的变分上界，可拆成先验匹配、中间转移 KL 和末端重建项。
- **$\epsilon$-prediction**：从 $x_t$ 和 $t$ 预测前向过程所加标准高斯噪声。
- **Score**：$\nabla_x\log p(x)$，指向当前分布高密度区域的梯度场。
- **Denoising score matching**：用被噪声扰动的样本训练 score；在 DDPM 参数化下与加权噪声 MSE 对应。
- **Ancestral sampling**：按学习到的随机条件分布从 $x_T$ 逐步采样到 $x_0$。
- **EMA**：参数指数滑动平均；原文 decay 为 0.9999，用于提高评估稳定性。
- **bits/dim**：离散图像平均每维码长，越低表示 likelihood 越好，但不必然对应更好的视觉质量。

# 18. 图表索引

1. [[assets/DDPM - 去噪扩散概率模型/figure-2-directed-graphical-model.png|原文 Figure 2：前向/反向有向图模型]]——TeX 原始矢量图转 PNG，解释两条马尔可夫链。
2. [[assets/DDPM - 去噪扩散概率模型/table-2-ablation.png|原文 Table 2：参数化与训练目标消融]]——由官方 arXiv 源码编译版 PDF 精确提取的完整表格，说明最优结果来自 $\epsilon$-prediction 与 $L_{\mathrm{simple}}$ 的组合。
3. [[assets/DDPM - 去噪扩散概率模型/figure-6-progressive-generation.jpg|原文 Figure 6：CIFAR-10 渐进生成]]——TeX 原图，显示结构先于细节出现。
4. [[assets/DDPM - 去噪扩散概率模型/figure-7-stochastic-decoding.jpg|原文 Figure 7：共享中间潜变量的随机解码]]——TeX 原图，展示 $x_t$ 保留的信息层级。
5. [[assets/DDPM - 去噪扩散概率模型/figure-8-interpolation.jpg|原文 Figure 8：CelebA-HQ 潜空间插值]]——TeX 原图，展示反向链的流形投影效果。

图表的原始出处、提取方式与逐张目视核验结果见 [[assets/DDPM - 去噪扩散概率模型/sources.md#3. 原文关键图表与逐张核验记录|sources.md 的核验记录]]。
