---
type: paper-note
title: "BYOL - 无负样本的自举表征学习"
paper: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
paper_title: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
authors:
  - Jean-Bastien Grill
  - Florian Strub
  - Florent Altché
  - Corentin Tallec
  - Pierre H. Richemond
  - Elena Buchatskaya
  - Carl Doersch
  - Bernardo Avila Pires
  - Zhaohan Daniel Guo
  - Mohammad Gheshlaghi Azar
  - Bilal Piot
  - Koray Kavukcuoglu
  - Rémi Munos
  - Michal Valko
year: 2020
venue: NeurIPS 2020
arxiv: "2006.07733"
source: https://arxiv.org/abs/2006.07733
doi: "10.48550/arXiv.2006.07733"
domain: Foundations
status: complete
tags:
  - paper
  - self-supervised-learning
  - representation-learning
  - siamese-network
  - knowledge-distillation
  - Foundations
created: 2026-10-15
updated: 2026-10-15
---

# BYOL：无负样本的自举表征学习

## 1. 论文信息

> **原题**：Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning  
> **作者**：Jean-Bastien Grill 等 14 位作者  
> **机构**：DeepMind；Imperial College London  
> **会议**：NeurIPS 2020 Oral  
> **arXiv**：[2006.07733](https://arxiv.org/abs/2006.07733)  
> **官方 PDF**：[arXiv PDF](https://arxiv.org/pdf/2006.07733)  
> **本地 PDF**：[[assets/BYOL - 无负样本的自举表征学习/sources/paper.pdf]]  
> **来源清单**：[[assets/BYOL - 无负样本的自举表征学习/sources.md]]  
> **精读日期**：2026-10-15

#### 1.1 一句话总结

BYOL 用一个可学习的 **online 网络**去预测同一图像另一增强视图在慢速 **target 网络**中的表示，依靠非对称 predictor、stop-gradient 与 EMA target 构造稳定的自举目标，在**不使用显式负样本**的情况下学到高质量视觉表征。

#### 1.2 十秒钟结论

- 核心不是“两个相同网络做一致性”，而是一个带 predictor 的在线分支追逐一个停止梯度、缓慢变化的目标分支。
- 论文最重要的经验事实是：无负样本目标虽然存在常数解，但标准 BYOL 在大规模实验中没有坍塌。
- ResNet-50 线性评估达到 ImageNet top-1 **74.3%**，当时超过同规模 SimCLR、MoCo v2 与 InfoMin。
- BYOL 对 batch size 和图像增强的退化更鲁棒，但仍依赖合适增强、BatchNorm、长周期预训练与昂贵算力。

## 2. 问题背景与研究动机

#### 2.1 对比学习在 2020 年的主流范式

当时的强自监督视觉方法通常构造两类配对：

1. 同一图像的两个增强视图形成正样本对；
2. 不同图像的视图形成负样本对；
3. 训练目标拉近正对、推远负对；
4. InfoNCE 一类目标把表征学习转化为实例判别。

这种范式有效，却带来三个结构性成本：

- 需要大 batch、memory bank 或 queue 才能提供足够多的负样本；
- 负样本中可能混入语义相同的“假负样本”；
- 性能容易依赖增强策略，因为模型可能利用颜色直方图等捷径完成实例判别。

BYOL 提出的尖锐问题是：

> 如果只让同一图像的两个视图彼此预测，不显式推开其他图像，能否仍然避免坍塌并学到可迁移表征？

#### 2.2 为什么“直接做一致性”会坍塌

若对所有图像都输出同一个常量向量 $c$，则任意两个增强视图的表示完全相同。

因此简单目标

$$
\min_\theta \|f_\theta(t(x))-f_\theta(t'(x))\|_2^2
$$

有一个显然的零损失解：

$$
f_\theta(\cdot)=c.
$$

这个解不携带图像信息，却能完美满足跨视图一致性。

对比学习用负样本显式排斥这种解；BYOL 则尝试通过**训练动力学的不对称性**避免它。

#### 2.3 BYOL 的起点：预测一个固定随机网络也能改进表示

论文先做了一个反直觉实验：

- target 是固定随机初始化网络；
- online 网络预测 target 对另一视图的输出；
- 固定随机网络自身线性评估仅 **1.4%**；
- 训练出来的 online 表征达到 **18.8%**。

这说明“预测一个已有表示”并非只能复制目标质量。

online 网络必须跨增强恢复 target 中可预测的部分，因而可能得到比原 target 更有用的表示。

BYOL 将这个过程连续化：不定期替换 checkpoint，而是让 target 成为 online 参数的慢速指数滑动平均。

## 3. 核心贡献

1. **提出无显式负样本的自监督框架 BYOL。**  
   它直接回归同图像另一视图的 target projection，不需要实例级负样本、memory bank 或负样本挖掘。

2. **用 online/target 非对称结构实现表征自举。**  
   online 分支多出 predictor；target 分支停止梯度，并由 online 参数的 EMA 更新。

3. **在 ImageNet 线性、半监督和迁移评测中取得当时领先结果。**  
   标准 ResNet-50 达到 74.3% top-1，更大模型达到 79.6%。

4. **通过系统消融研究坍塌、EMA、predictor、batch size 与增强的作用。**  
   论文不仅报告最终分数，还展示哪些不对称设计对无负样本训练至关重要。

5. **把“负样本不是避免坍塌的唯一机制”变成可复现的经验事实。**  
   这为 SimSiam、DINO 以及后续非对比式自监督方法打开了设计空间。

## 4. 方法总览

#### 4.1 整体框架

![BYOL online/target 架构](assets/BYOL%20-%20无负样本的自举表征学习/figures/fig02_online_target_architecture.png)

> **图 1｜原论文 Figure 2，PDF 第 4 页。** online 分支依次经过 encoder、projector、predictor；target 分支只有 encoder 与 projector，输出经 stop-gradient 后作为回归目标。  
> **来源 URL**：https://arxiv.org/pdf/2006.07733

对每张图像 $x$，采样两个随机增强：

$$
v=t(x),\qquad v'=t'(x).
$$

随后执行两条路径：

- online：$v\rightarrow f_\theta\rightarrow g_\theta\rightarrow q_\theta$；
- target：$v'\rightarrow f_\xi\rightarrow g_\xi\rightarrow \operatorname{sg}$。

online 的 prediction 要接近 target 的 projection。

再交换两个视图的角色，构成对称损失。

一个训练步结束后：

1. 用梯度更新 online 参数 $\theta$；
2. 不对 target 参数 $\xi$ 反向传播；
3. 用 EMA 把新的 online 参数缓慢写入 target。

训练完成后丢弃 projector、predictor 和整个 target 分支，只保留 online encoder $f_\theta$。

#### 4.2 模块与张量流

给定 online 视图 $v$：

$$
y_\theta=f_\theta(v),
$$

$$
z_\theta=g_\theta(y_\theta),
$$

$$
p_\theta=q_\theta(z_\theta).
$$

给定 target 视图 $v'$：

$$
y'_\xi=f_\xi(v'),
$$

$$
z'_\xi=g_\xi(y'_\xi).
$$

其中：

- $y$ 是 encoder representation；
- $z$ 是 projector 输出；
- $p$ 是 predictor 输出；
- 下游任务使用 $y_\theta$，而不是 $z_\theta$ 或 $p_\theta$。

## 5. 关键组件逐项拆解

#### 5.1 Online encoder：真正要保留的表征网络

online encoder $f_\theta$ 是 ResNet 主干。

标准配置使用 ResNet-50 v1，最终全局平均池化输出为 2048 维。

它承担两个角色：

- 训练时生成可供 projector 变换的语义表示；
- 训练后作为下游任务的特征提取器。

只保留 encoder 的设计把“训练目标空间”与“下游表示空间”分开。

这延续了 SimCLR 的经验：projection head 可以吸收对自监督目标有利、但未必适合线性评估的变换。

#### 5.2 Projector：把表示映射到自监督损失空间

online projector $g_\theta$ 与 target projector $g_\xi$ 结构相同：

1. 线性层，输出 4096 维；
2. BatchNorm；
3. ReLU；
4. 线性层，输出 256 维；
5. 最后输出层不使用 BatchNorm。

projector 的意义不是单纯降维。

它创建一个专门承受跨视图回归约束的空间，使 encoder representation 不必直接拟合全部增强不变性。

论文消融表明，默认的两层 projector 配两层 predictor 表现最好；过浅时性能明显下降。

例如 projector 深度为 1、predictor 深度为 1 时 top-1 为 61.9%，默认 2+2 时为 72.5%。

投影维度从 64 到 512 时结果大多在 72.2%—72.6%，说明在合理范围内不特别敏感；16 维则降至 69.9%。

#### 5.3 Predictor：制造关键结构非对称

predictor $q_\theta$ 只存在于 online 分支，结构与 projector 相同。

target 分支没有 predictor。

这意味着优化目标不是让两个同构分支直接互相复制，而是让 online 在一个额外可学习映射后预测 target projection。

predictor 的作用至少有三层：

- 在结构上打破 online/target 对称；
- 让在线表示通过可调整映射追踪缓慢目标；
- 在论文的条件方差直觉中，近似最优预测器 $q^*$。

Table 5 的直接证据是：在无负样本设置 $\beta=0$ 中，只有同时具有 predictor 和 target network 的标准 BYOL 获得 72.5%；移除关键部件的变体接近随机水平。

后续 SimSiam 进一步表明，即使没有 EMA momentum encoder，predictor 与 stop-gradient 的组合也能学到非坍塌表示。

因此 predictor 不应被理解为“仅提高几分的 MLP head”，而是非对比式孪生训练动力学的重要部件。

#### 5.4 Target network：不由梯度训练的慢教师

target 网络参数记为 $\xi$，包含 encoder $f_\xi$ 与 projector $g_\xi$。

它不使用优化器更新，而在每个训练步后执行：

$$
\xi\leftarrow \tau\xi+(1-\tau)\theta.
$$

当 $\tau$ 接近 1 时，target 变化慢；当 $\tau=0$ 时，相当于立即复制 online。

EMA 有两种效果需要区分：

1. **stop-gradient 效果**：target 不接收当前损失的反向梯度；
2. **stable-and-stale 效果**：target 是 online 历史参数的平滑集合，目标变化更慢。

论文 Appendix F.4 的补充消融认为，主要增益更多来自稳定且略滞后的目标，而不仅是停止梯度本身。

#### 5.5 Stop-gradient：阻断 target 分支的即时共适应

记 $\operatorname{sg}(z'_\xi)$ 为数值上等于 $z'_\xi$、但梯度为零的张量。

损失反传时：

- online encoder、projector、predictor接收梯度；
- target encoder 与 projector 不接收梯度；
- target 只通过 EMA 间接跟随 online。

如果两侧都沿同一相似度损失同步更新，它们可以共同走向常数表示。

stop-gradient 让 target 在当前优化步中成为固定回归目标，避免两端即时“串通”降低损失。

但必须谨慎：stop-gradient 本身不是论文给出的充分数学保证。

Table 5 中“stop gradient of online，$\tau=0$”只有 0.3%，说明标准 BYOL 中仅阻断梯度而没有合适的慢目标与 predictor 动力学并不够。

#### 5.6 EMA 动量调度

论文标准配置不是固定 $\tau$，而是从 $\tau_{base}=0.996$ 逐渐增加到 1：

$$
\tau_k=1-(1-\tau_{base})\frac{\cos(\pi k/K)+1}{2},
$$

其中 $k$ 是当前步，$K$ 是总步数。

直觉上：

- 训练早期 online 变化快，target 需要适度跟随；
- 训练后期把 $\tau$ 推向 1，使 target 更平滑，减少目标抖动。

Table 5a 展示了两端极值都不好：

- $\tau=1$：target 永远固定，只有 18.8%；
- $\tau=0$：target 每步硬复制，只有 0.3%；
- $\tau_{base}=0.99$：达到 72.5%；
- 0.9 到 0.999 的中间范围都明显优于两个极端。

## 6. 损失函数与完整训练动态

#### 6.1 单方向归一化回归损失

先做 $\ell_2$ 归一化：

$$
\bar p_\theta=\frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|_2},
\qquad
\bar z'_\xi=\frac{z'_\xi}{\|z'_\xi\|_2}.
$$

单方向损失为：

$$
\mathcal L_{\theta,\xi}(v,v')
=\|\bar p_\theta-\operatorname{sg}(\bar z'_\xi)\|_2^2.
$$

因为两个向量都归一化，这等价于：

$$
\mathcal L_{\theta,\xi}(v,v')
=2-2\frac{\langle q_\theta(z_\theta),z'_\xi\rangle}
{\|q_\theta(z_\theta)\|_2\|z'_\xi\|_2}.
$$

所以 BYOL 实际是在最大化 online prediction 与 target projection 的 cosine similarity。

它没有任何跨样本求和项，也没有温度控制下的负样本 softmax 分母。

#### 6.2 对称损失

一个方向只让 $v$ 预测 $v'$。

论文交换视图角色，再计算：

$$
\widetilde{\mathcal L}_{\theta,\xi}
=\mathcal L_{\theta,\xi}(v',v).
$$

总损失是：

$$
\mathcal L^{\mathrm{BYOL}}_{\theta,\xi}
=\mathcal L_{\theta,\xi}(v,v')
+\mathcal L_{\theta,\xi}(v',v).
$$

对称化让每个视图都经过 online 分支一次，也都作为 target 一次。

这减少了由两套非完全对称增强分布 $T,T'$ 带来的方向偏差。

#### 6.3 参数更新顺序

完整一步可写成：

$$
\theta\leftarrow\operatorname{optimizer}
\left(\theta,\nabla_\theta\mathcal L^{\mathrm{BYOL}}_{\theta,\xi},\eta\right),
$$

$$
\xi\leftarrow\tau\xi+(1-\tau)\theta.
$$

关键细节是：

- 梯度只对 $\theta$ 计算；
- 更新 online 后再更新 target；
- target 是训练状态，不是独立优化变量；
- 因而 BYOL 不是对 $\theta,\xi$ 联合最小化同一个静态目标。

#### 6.4 伪代码级理解

```text
for each minibatch x:
    v  = augment_1(x)
    v' = augment_2(x)

    p1 = q_online(g_online(f_online(v)))
    p2 = q_online(g_online(f_online(v')))

    z1_target = stopgrad(g_target(f_target(v)))
    z2_target = stopgrad(g_target(f_target(v')))

    loss = normalized_mse(p1, z2_target)
         + normalized_mse(p2, z1_target)

    theta = optimizer_step(theta, grad_theta(loss))
    xi = tau * xi + (1 - tau) * theta
```

这段过程没有类别标签，没有负样本队列，也没有聚类伪标签。

## 7. 为什么无显式负样本仍不坍塌

#### 7.1 必须先承认：目标函数确实允许坍塌解

若所有输入都映射到同一个归一化向量，online prediction 与 target projection 可以完全一致。

因此“BYOL 的 loss 没有坍塌最优解”是错误表述。

正确问题是：

> 为什么标准初始化、优化器、predictor、stop-gradient 和 EMA 组成的训练动力学没有走向这个可行解？

#### 7.2 论文直接提供的经验证据

![BYOL target、EMA、predictor 与坍塌消融](assets/BYOL%20-%20无负样本的自举表征学习/figures/table05_target_predictor_collapse_ablation.png)

> **图 2｜原论文 Table 5，PDF 第 9 页。** 左表说明 target 更新太快或完全不更新都不好；右表说明无负样本时 predictor 与 target network 的组合是获得有效表示的关键。  
> **来源 URL**：https://arxiv.org/pdf/2006.07733

论文证据可以分成四组：

1. **标准 BYOL 不坍塌。**  
   predictor + EMA target + 无负样本得到 72.5% top-1。

2. **立即复制 online 到 target 会失败。**  
   $\tau=0$ 的 stop-gradient online 变体只有 0.3%。

3. **固定随机 target 不坍塌但质量低。**  
   $\tau=1$ 得到 18.8%，说明稳定目标能阻止常数解，却无法持续自举改进。

4. **predictor 与 target 的配合很关键。**  
   去掉 predictor 的无监督 Mean Teacher 式变体坍塌；论文正文明确把 predictor 视为关键组件。

此外，论文还发现：

- 去掉 EMA 后，把 predictor 学习率提高到主干的 10 倍，仍可达到约 66.6%；
- 用 batch 上闭式最优线性 predictor，也能在无慢 target 时避免完全坍塌；
- predictor 学习率不高于 projector 时更容易坍塌。

这些结果支持“保持 predictor 接近最优”是稳定训练的重要条件。

#### 7.3 论文给出的条件方差直觉

论文假设 predictor 对当前 online/target 表示近似最优：

$$
q^*(z_\theta)
=\mathbb E[z'_\xi\mid z_\theta].
$$

对应最小回归误差是 target 在给定 online 表示后的条件方差：

$$
\mathbb E\left[\|q^*(z_\theta)-z'_\xi\|_2^2\right]
=\mathbb E\left[\sum_i\operatorname{Var}(z'_{\xi,i}\mid z_\theta)\right].
$$

如果 online representation 丢弃信息，通常不会降低预测 target 所需的条件方差。

因此，在 target 保持丰富变化且 predictor 足够好的前提下，online 有动力保留能预测 target 的信息，而不是变成常数。

EMA 的角色是让 target 变化足够慢，使 predictor 有机会维持近似最优；同时 target 又会逐渐吸收 online 新捕获的变化。

这个直觉解释了“稳定但可移动”的平衡：

- target 太快：predictor 永远追不上；
- target 不动：无法把 online 的改进纳入新目标；
- target 适度慢：形成逐步提升的 bootstrap loop。

#### 7.4 论文没有证明什么

论文没有给出一般神经网络训练下“必然不坍塌”的定理。

条件方差推导依赖若干理想化假设：

- predictor 接近最优；
- 为简化分析忽略了归一化与对称化；
- 讨论的是局部训练动力学直觉；
- 实际 SGD、BatchNorm、有限 batch 和网络参数化更复杂。

所以论文结论应表述为：

> BYOL 在所测试的大规模配置中经验上不坍塌，消融支持 predictor 与慢 target 的关键作用，并给出条件方差视角的解释假说。

#### 7.5 后续工作的解释：与论文证据分开看

**后续证据一：SimSiam。**

SimSiam 表明 momentum encoder 不是避免坍塌的绝对必要条件；无负样本、无大 batch、无 EMA 时，只要保留 predictor 与 stop-gradient，也能学到有效表示。

这强化了“结构与优化不对称比负样本更根本”的看法，也说明 EMA 更像稳定器和性能增强器，而不是唯一防坍塌机制。

**后续解释二：隐式方差/协方差维持。**

后续许多分析把 predictor、归一化、BatchNorm、优化器和 stop-gradient 视为共同塑造特征谱与方差的机制。

这些工作通常认为，训练动力学可对低方差方向施加不同的隐式作用，从而不必用显式负样本逐对推开样本。

但不同理论依赖线性网络、白化、无限 batch 或特定归一化假设，尚不能简化为单一、普适的“BYOL 不坍塌定理”。

**本笔记的判断。**

最稳妥的解释是组合性的：

- stop-gradient 阻止两支在同一步共同追向平凡解；
- predictor 提供适应目标的非对称自由度；
- EMA 降低目标漂移，改善 predictor 的可跟踪性；
- 增强与归一化决定必须保留哪些跨视图信息；
- 优化器、权重衰减和 BatchNorm 进一步影响实际动力学。

## 8. 实现细节与复现配方

#### 8.1 数据增强

BYOL 基本沿用 SimCLR 增强：

- 随机裁剪并缩放到 $224\times224$；
- 随机水平翻转；
- brightness、contrast、saturation、hue 抖动；
- 随机灰度化；
- Gaussian blur；
- solarization。

两套增强分布并不完全相同：

- 第一视图 blur 概率 1.0，不做 solarization；
- 第二视图 blur 概率 0.1，solarization 概率 0.2。

这种轻度不对称扩大了两个视图的差异，同时避免目标完全同质。

#### 8.2 网络结构

标准 ResNet-50 配置：

- encoder 输出：2048 维；
- projector：2048 → 4096 → 256；
- predictor：256 → 4096 → 256；
- 隐层：Linear + BatchNorm + ReLU；
- 最后线性层后不加 BatchNorm。

训练期间维护 online 与 target 两套 encoder/projector，因此训练态参数与激活成本高于最终部署态。

推理时只保留 online encoder，所以部署成本与普通 ResNet-50 相近。

#### 8.3 优化配置

论文主实验配置：

- 预训练：1000 epochs；
- optimizer：LARS；
- base learning rate：0.2；
- 按 batch size 线性缩放：$0.2\times\text{BatchSize}/256$；
- warmup：10 epochs；
- learning-rate schedule：cosine decay；
- weight decay：$1.5\times10^{-6}$；
- batch size：4096；
- target 初始动量：$\tau_{base}=0.996$，随后余弦增加到 1；
- 硬件：512 个 Cloud TPU v3 cores；
- ResNet-50 预训练约 8 小时。

官方仓库说明该配置应得到约 74.0%—74.5% top-1，并提供 74.4% 的 ResNet-50 checkpoint 与 79.6% 的 ResNet-200 2× checkpoint。

#### 8.4 小 batch 复现

论文 Appendix G 给出 batch size 512 的替代设置：

- 64 个 TPU cores；
- 1000 epochs；
- base learning rate 0.4，再按 batch 规则缩放；
- $\tau_{base}=0.9995$；
- 训练约 4 天；
- top-1 73.7%。

这说明 BYOL 对 batch size 的鲁棒性不等于“单卡廉价”。

较小 batch 会显著拉长训练时间，BatchNorm 还可能成为额外不稳定来源。

#### 8.5 复现时最容易忽略的细节

1. target 参数初始化应与 online 参数一致，而不是独立随机初始化后长期偏离。
2. loss 的 target 端必须 stop-gradient。
3. predictor 只在 online 分支出现。
4. 两个视图都要分别走一次 online→target 对称方向。
5. EMA 应在 online optimizer step 之后更新。
6. 最终线性评估冻结 encoder 参数与 BatchNorm 统计。
7. weight decay 不应随意设为 0；论文观察到 BYOL 和 SimCLR 都可能发散。
8. 论文报告的 74.3% 使用标准线性评估；改变评估增强和 logits regularization 可升至 74.8%，不能混为同一协议。
9. 官方代码提示 ResNet block 最后 BatchNorm 的 $\gamma$ 初始化会影响大模型复现。
10. ImageNet 数据准备、分布式输入吞吐和 checkpoint 评估本身都有明显工程成本。

## 9. 实验设计与主要结果

#### 9.1 评测框架

预训练数据是 ImageNet ILSVRC-2012 的训练集，不使用标签。

论文从四个层面评估：

1. ImageNet frozen encoder 线性评估；
2. 1%/10% 标签的 ImageNet 半监督微调；
3. 12 个分类数据集的线性迁移与全量微调；
4. 语义分割、目标检测和单目深度估计。

这个评测组合不仅问“ImageNet 线性分数高不高”，也问表示是否可迁移到类别、像素、框和几何任务。

#### 9.2 ImageNet 线性评估

![BYOL ImageNet 主要性能](assets/BYOL%20-%20无负样本的自举表征学习/figures/fig01_imagenet_performance.png)

> **图 3｜原论文 Figure 1，PDF 第 1 页。** 在多种模型规模下，BYOL 形成明显高于同期非监督基线的参数—精度曲线，并逼近监督预训练。  
> **来源 URL**：https://arxiv.org/pdf/2006.07733

标准 ResNet-50 的结果：

| 方法 | Top-1 | Top-5 |
|---|---:|---:|
| SimCLR | 69.3 | 89.0 |
| MoCo v2 | 71.1 | — |
| InfoMin Aug. | 73.0 | 91.1 |
| **BYOL** | **74.3** | **91.6** |

BYOL 相比此前最佳 InfoMin top-1 提高 1.3 个百分点。

与论文采用的 SimCLR 监督基线 76.5% 相比，差距缩小到 2.2 个百分点。

更大模型的结果：

| 架构 | 参数量 | BYOL Top-1 | BYOL Top-5 |
|---|---:|---:|---:|
| ResNet-50 2× | 94M | 77.4 | 93.6 |
| ResNet-50 4× | 375M | 78.6 | 94.2 |
| ResNet-200 2× | 250M | **79.6** | **94.8** |

需要注意，大模型结果同时改变了深度、宽度和参数量，不能把全部增益归因于 BYOL 算法本身。

#### 9.3 半监督 ImageNet

使用固定的 1% 与 10% 标签 split 微调整个网络。

ResNet-50 结果：

| 方法 | 1% Top-1 | 10% Top-1 | 1% Top-5 | 10% Top-5 |
|---|---:|---:|---:|---:|
| Supervised from scratch | 25.4 | 56.4 | 48.4 | 80.4 |
| SimCLR | 48.3 | 65.6 | 75.5 | 87.8 |
| **BYOL** | **53.2** | **68.8** | **78.4** | **89.0** |

BYOL 对低标签比例尤其有效：

- 1% 标签 top-1 比 SimCLR 高 4.9；
- 10% 标签 top-1 高 3.2；
- 表明预训练表示在标签极少时提供了更强归纳偏置。

使用 ResNet-200 2× 时，1%/10% 标签 top-1 达到 71.2%/77.7%。

用 100% ImageNet 标签微调 ResNet-50 时，BYOL 初始化得到 77.7%，高于普通监督基线 76.5%，但低于 MaxUp 78.9%。

#### 9.4 分类迁移

论文在 Food101、CIFAR-10/100、Birdsnap、SUN397、Cars、Aircraft、VOC2007、DTD、Pets、Caltech-101、Flowers 上评估。

主要结论：

- 线性评估中，BYOL 在 12 个任务上全部优于论文的 SimCLR 复现；
- 相比 ImageNet 监督预训练，BYOL 在 12 个任务中的 7 个更好；
- 全量微调时，差距缩小，但 BYOL 仍总体与强基线相当或更好。

若看具体线性迁移：

- Food101：BYOL 75.3，SimCLR 复现 72.8；
- CIFAR-100：78.4 对 74.4；
- Birdsnap：57.2 对 42.4；
- Cars：67.8 对 49.3；
- Aircraft：60.6 对 49.8；
- Flowers：96.1 对 92.6。

Birdsnap、Cars、Aircraft 等细粒度任务提升较大，说明 BYOL 表示并非只适合 ImageNet 粗粒度类别。

但迁移结论仍受下游超参数搜索预算影响；论文对每个数据集进行了验证集选择，复现成本不低。

#### 9.5 检测、分割与深度估计

PASCAL 任务：

| 预训练 | VOC07 Detection AP50 | VOC12 Segmentation mIoU |
|---|---:|---:|
| Supervised-IN | 74.4 | 74.4 |
| MoCo | 74.9 | 72.5 |
| SimCLR 复现 | 75.2 | 75.2 |
| **BYOL** | **77.5** | **76.3** |

BYOL 比监督预训练高 3.1 AP50、1.9 mIoU，也优于 SimCLR。

NYU v2 深度估计：

| 方法 | pct.<1.25 ↑ | rms ↓ | rel ↓ |
|---|---:|---:|---:|
| Supervised-IN | 81.1 | 0.573 | **0.127** |
| SimCLR 复现 | 83.3 | 0.557 | 0.134 |
| **BYOL** | **84.6** | **0.541** | 0.129 |

BYOL 在多数深度指标上最好，但 relative error 略逊于监督预训练。

这比“所有任务全面领先”更准确：结果强，但并非每一个指标都占优。

#### 9.6 Places365 预训练的外部有效性

论文还在 Places365 上用同一超参数预训练 1000 epochs。

BYOL-Places 在线性评估中：

- Places365：53.2；
- 同数据预训练的 SimCLR：53.0；
- ImageNet：58.5 对 SimCLR 的 56.5。

BYOL-Places 在大多数任务上仍不如 BYOL-ImageNet，只有 Places365 与 SUN397 等场景任务更占优。

这说明算法可迁移到另一个大规模数据集，但预训练数据域仍显著决定表征的下游适用性。

## 10. 消融实验：读懂 BYOL 的关键

#### 10.1 Batch size 鲁棒性

![BYOL batch size 与增强鲁棒性](assets/BYOL%20-%20无负样本的自举表征学习/figures/fig03_batch_augmentation_robustness.png)

> **图 4｜原论文 Figure 3，PDF 第 8 页。** 左图显示 BYOL 在 256—4096 batch size 范围内基本稳定；右图显示逐步删除增强时，BYOL 的性能下降小于 SimCLR。  
> **来源 URL**：https://arxiv.org/pdf/2006.07733

300-epoch 消融的完整数字：

| Batch size | BYOL Top-1 | SimCLR Top-1 |
|---:|---:|---:|
| 4096 | 72.5 | 67.9 |
| 2048 | 72.4 | 67.8 |
| 1024 | 72.2 | 67.4 |
| 512 | 72.2 | 66.5 |
| 256 | 71.8 | 64.3 |
| 128 | 69.6 | 63.6 |
| 64 | 59.7 | 59.2 |

解释：

- SimCLR batch 变小会减少同一步中的负样本数量；
- BYOL 没有这种直接依赖，因此 256 以上较稳定；
- BYOL 在 128 和 64 仍下降，论文把主要原因归于 BatchNorm 的小 batch 行为；
- 因此“不用负样本”减少了 batch 依赖，但没有消除所有 batch 依赖。

#### 10.2 增强鲁棒性

300-epoch 结果：

| 增强配置 | BYOL Top-1 | SimCLR Top-1 |
|---|---:|---:|
| Baseline | 72.5 | 67.9 |
| 去除 color jitter + grayscale | 63.4 | 45.7 |
| 去除 color jitter | 71.8 | 63.7 |
| 去除 grayscale | 70.3 | 61.9 |
| 仅 random crop | 59.4 | 40.3 |
| crop + blur | 61.1 | 41.7 |
| crop + color | 70.7 | 64.2 |

论文解释是：

- SimCLR 做实例判别时可能利用跨 crop 稳定、跨图像不同的颜色直方图捷径；
- 去掉颜色扰动后，判别任务可被低层统计轻易解决；
- BYOL 要预测 target 中已有的多种信息，不只需要找到区分实例的最短捷径；
- 因而去除颜色增强后下降更小。

但 BYOL 只用 crop 仍从 72.5 降到 59.4，下降 13.1 点。

所以论文证明的是“更鲁棒”，不是“无需增强”。

#### 10.3 Target momentum 消融

Table 5a 的核心信息：

| Target 模式 | $\tau_{base}$ | Top-1 |
|---|---:|---:|
| 固定随机网络 | 1 | 18.8 ± 0.7 |
| online EMA | 0.999 | 69.8 |
| online EMA | 0.99 | **72.5** |
| online EMA | 0.9 | 68.4 |
| stop-gradient online | 0 | 0.3 |

这不是简单的“动量越大越好”。

最优区域取决于训练长度、学习率和模型变化速度。

主实验 1000 epochs 使用 0.996，而 300-epoch 消融的基线使用 0.99；小 batch 512 的 1000-epoch 配方又使用 0.9995。

因此 $\tau$ 应被视为“目标时间尺度”超参数，而非固定魔数。

#### 10.4 Predictor/target/negative term 联合消融

论文把 SimCLR 与 BYOL 写到带系数 $\beta$ 的统一目标中：

- $\beta=1$：保留负样本 dispersion term；
- $\beta=0$：移除负样本，成为 BYOL 式正对回归。

结果显示：

- 标准 BYOL：predictor + target + $\beta=0$，72.5%；
- 加回负样本但不重新调温度：70.9%；
- SimCLR：无 predictor、无 target、$\beta=1$，69.4%；
- 无 target 且 $\beta=0$ 的变体接近 0；
- 只有 target、没有合适 predictor 的无负样本变体也接近 0。

重要含义不是“负样本有害”。

Appendix F.4 重新调温度后，加负样本的版本可以匹配 BYOL，只是没有显著超过。

因此更准确的结论是：BYOL 证明了高质量表征**不需要显式负样本**，而不是证明负样本在所有配置下都无用。

#### 10.5 归一化消融

300-epoch 结果：

| Loss 输出归一化 | Top-1 | Top-5 |
|---|---:|---:|
| $\ell_2$ normalization | **72.5** | **90.8** |
| LayerNorm | 72.5 ± 0.4 | 90.1 |
| 无归一化 | 67.4 | 87.1 |
| BatchNorm | 65.3 | 85.3 |

无归一化时 projection 范数会快速增至约 $3\times10^6$，但仍得到 67.4%。

这说明归一化显著改善尺度控制和性能，却不能单独解释全部非坍塌行为。

#### 10.6 学习率、权重衰减与 head 深度

学习率在 0.2—0.5 范围结果较稳：

- 0.2：71.7；
- 0.3：72.5；
- 0.4：72.3；
- 0.5：71.5。

权重衰减在 $10^{-7}$ 到 $10^{-6}$ 附近较稳，过大开始下降；设为 0 可能导致不稳定或发散。

默认 2 层 projector + 2 层 predictor 最好，但邻近结构差异不大。

这表明 BYOL 并非只在唯一精确超参数点工作，但也远不是完全免调参。

#### 10.7 消融实验是否足以支撑主张

**支撑充分的部分：**

- 不用显式负样本仍可达到强性能；
- predictor、慢 target、stop-gradient 组合与稳定训练高度相关；
- batch size 与增强退化明显优于 SimCLR 复现；
- 表征可以迁移到多种任务。

**仍不充分的部分：**

- 没有一般性理论证明排除坍塌；
- 很多消融仍在大 batch、TPU、ResNet、ImageNet 体系内；
- BatchNorm、优化器与 predictor 的交互没有完全解耦；
- 与同期方法的训练预算、架构和评估协议并非所有条目严格相同。

## 11. 与 SimCLR、MoCo、DINO 的关系

#### 11.1 BYOL 与 SimCLR

共同点：

- 都使用同一图像的两个强增强视图；
- 都使用 encoder + MLP projector；
- 都用归一化表示和 cosine 相似性；
- 原始大规模配置都使用 LARS、长周期训练和大 batch。

差异：

| 维度 | SimCLR | BYOL |
|---|---|---|
| 目标 | InfoNCE 实例判别 | 跨视图回归 |
| 负样本 | batch 内大量负样本 | 无显式负样本 |
| 分支 | 基本对称 | online/target 非对称 |
| predictor | 无 | online 独有 |
| target EMA | 无 | 有 |
| 小 batch 敏感性 | 较高 | 较低但仍受 BN 影响 |

从统一目标看，BYOL 可被理解为移除 InfoNCE dispersion term，并加入 predictor 与 momentum target 的变体。

#### 11.2 BYOL 与 MoCo

共同点：

- 都维护 momentum encoder；
- 都用 EMA 获得缓慢变化的表示；
- 都只对 query/online 分支做梯度更新。

关键差异：

- MoCo 的 momentum encoder 主要为 queue 中的负样本提供一致 key；
- BYOL 的 target network 直接生成正对回归目标；
- MoCo 的核心仍是对比学习，BYOL 则没有显式负样本与队列。

BYOL 的消融还显示，给 SimCLR 加 target network 即使负样本数量不变，也可提升约 1.6 点。

这说明 momentum encoder 的价值不只在“维护更多负样本”，还包括目标稳定化。

#### 11.3 BYOL 与 DINO

DINO 在 BYOL 之后延续 teacher-student + EMA 的自蒸馏结构，但学习对象不同。

| 维度 | BYOL | DINO |
|---|---|---|
| 主干 | 原论文以 ResNet 为主 | 强调 Vision Transformer |
| 目标空间 | 归一化连续向量回归 | teacher/student 概率分布交叉熵 |
| 防坍塌关键 | predictor、stop-grad、EMA 等动力学 | EMA teacher、centering、sharpening |
| 多视图 | 两个主要视图，对称回归 | multi-crop，局部/全局视图 |
| 显著现象 | 强线性与迁移性能 | ViT 特征涌现语义分割与优秀 k-NN |

关系上，DINO 可以被看作把 BYOL 式“慢教师自蒸馏”扩展到概率输出、multi-crop 和 ViT 的代表性工作。

两者都说明：teacher 不需要外部标签，也不需要比 student 更强；teacher 可以是 student 历史的平滑版本。

#### 11.4 BYOL 在技术谱系中的位置

BYOL 连接了几条此前分散的思想：

- 强增强下的跨视图一致性；
- 深度强化学习中的 target network；
- Mean Teacher 的参数平均教师；
- SimCLR 的 projection head 与大规模优化配方；
- 自蒸馏中的 teacher-student 预测。

它的历史意义不只是一个 ImageNet 分数，而是把非对比式联合嵌入方法变成主流研究方向。

## 12. 局限、风险与复现成本

#### 12.1 仍然依赖人工设计的数据增强

论文自己明确承认：BYOL 依赖适合视觉的增强集合。

换到音频、文本、视频或医学图像时，“哪些变换保持语义”必须重新定义。

不恰当增强会导致：

- 视图过于相似，任务太容易；
- 视图语义被破坏，目标互相冲突；
- 模型学习错误不变性。

#### 12.2 不坍塌机制缺少完备理论

论文的条件方差解释有启发性，但不构成一般保证。

实际稳定性来自多个因素耦合，导致：

- 修改归一化、optimizer、head 或架构后可能出现新失效模式；
- 小规模成功不能保证大规模成功，反之亦然；
- 很难仅凭静态 loss 预测训练是否坍塌。

#### 12.3 训练成本高

论文主结果使用：

- ImageNet 全量无标签图像；
- 1000 epochs；
- batch size 4096；
- 512 个 TPU v3 cores；
- online/target 双网络；
- 大量增强与同步 BatchNorm。

虽然报告约 8 小时，但这是高度并行硬件上的墙钟时间，不代表低总算力成本。

官方代码还指出公共 Cloud TPU 的数据输入可能成为瓶颈。

#### 12.4 实验比较存在时代与协议边界

74.3% 是 2020 年特定训练预算和线性评估协议下的结果。

不能直接与后来使用 ViT、更大数据、更长训练、多裁剪或不同评估增强的方法横比。

论文内部一些条目也来自公开结果而非统一复现，模型参数量和训练 epoch 并不完全一致。

#### 12.5 BatchNorm 仍引入跨样本依赖

BYOL 不使用显式负样本，但标准实现广泛使用 BatchNorm。

因此每个样本的归一化统计仍会依赖同一 batch 的其他样本。

这不等同于“偷偷使用负样本”，但说明标准 BYOL 也不是完全逐样本独立的目标。

小 batch 下降进一步说明 BatchNorm 是工程边界之一。

#### 12.6 数据偏差与双重用途

论文 broader impact 指出：

- 视觉数据集中的偏差可能被预训练表示复制；
- 通用视觉表示可服务有益应用，也可能服务负面用途；
- 无标签训练并不会自动消除数据采集偏差。

## 13. 我的判断与实践建议

#### 13.1 最值得记住的思想

BYOL 的核心思想是：

> 可以用模型自身缓慢变化的历史表示作为学习目标，只要训练结构与时间尺度足够不对称。

这比“去掉负样本”更深刻。

它把表示学习从“当前 batch 内的样本判别”转向“跨时间的自举预测”。

#### 13.2 如果今天复现，先做什么

建议按三档执行：

1. **机制验证档**：CIFAR 或 Imagenette、ResNet-18、较短训练，先监控 loss、特征标准差和线性 probe。
2. **可信复现档**：ImageNet-100 或完整 ImageNet、ResNet-50，严格复刻增强、EMA、head 和线性评估。
3. **论文对齐档**：1000 epochs、大规模分布式训练、多学习率线性评估，记录总算力与评测选择。

监控项不应只有训练 loss：

- 每维特征标准差；
- 特征 covariance 的非对角项；
- 特征奇异值谱/有效秩；
- online 与 target cosine similarity；
- predictor 与 projector 的梯度范数；
- frozen linear probe。

坍塌时 loss 可能很好看，所以表示统计比 loss 更重要。

#### 13.3 适用场景

BYOL 式方法适合：

- 有大量无标签图像、但标签昂贵的场景；
- 需要迁移到多类下游视觉任务的预训练；
- 不方便维护超大负样本队列的系统；
- 希望研究 teacher-student、自蒸馏或跨视图预测机制的基线。

不一定适合：

- 算力很紧、只能短训练的项目；
- 无法定义语义保持增强的模态；
- batch 极小且不能替换/同步归一化的训练环境；
- 需要严格理论稳定性保证的系统。

#### 13.4 最终评价

BYOL 最有价值的贡献不是宣布“负样本过时”，而是用强实验说明：

- 坍塌并不只能靠显式样本排斥避免；
- 目标网络的时间平滑本身是一种强学习信号设计；
- predictor 与 stop-gradient 可以改变优化动力学；
- 自监督表示质量可以在没有标签、负样本与聚类步骤时持续自举。

它也留下一个至今重要的问题：

> 对非对比式联合嵌入方法，究竟哪些显式或隐式机制在维持特征方差、传播语义信息并决定稳定性？

## 14. 关键术语表

- **Bootstrap**：此处是“用已有输出继续改进自身”的惯用义，不是统计学 bootstrap sampling。
- **Online network**：接受梯度更新、包含 predictor 的主训练分支。
- **Target network**：不接受梯度，通过 online 参数 EMA 更新的目标分支。
- **Encoder**：产生最终下游表示的主干网络。
- **Projector**：把 encoder 表示映射到自监督损失空间的 MLP。
- **Predictor**：仅存在于 online 分支、用于预测 target projection 的 MLP。
- **Stop-gradient**：前向保留数值，反向梯度置零的操作。
- **EMA / momentum encoder**：参数的指数滑动平均，产生平滑且略滞后的网络。
- **Collapse**：不同输入得到相同或低秩表示，使自监督目标低但表征无信息。
- **Linear evaluation**：冻结 encoder，仅训练线性分类器评估表示可分性。
- **Self-distillation**：teacher 不依赖人工标签，student 学习 teacher 输出的训练范式。

## 15. 图表索引

1. **图 1：online/target 架构**——原论文 Figure 2，PDF 第 4 页，展示 encoder、projector、predictor、stop-gradient 与损失流向。
2. **图 2：target/predictor/坍塌消融**——原论文 Table 5，PDF 第 9 页，展示 EMA 时间尺度与关键组件组合。
3. **图 3：ImageNet 主要性能**——原论文 Figure 1，PDF 第 1 页，展示不同参数规模下 BYOL 与监督/自监督基线。
4. **图 4：batch size 与增强鲁棒性**——原论文 Figure 3，PDF 第 8 页，比较 BYOL 和 SimCLR 的性能下降。

## 16. 复习问题

1. 为什么 BYOL 的静态目标允许坍塌，但训练动态仍可能不坍塌？
2. predictor、stop-gradient 与 EMA target 分别制造了哪种不对称？
3. 为什么 $\tau=0$ 与 $\tau=1$ 都不是好选择？
4. 对称损失为什么要交换两个视图的 online/target 角色？
5. BYOL 不使用显式负样本，为什么仍然对 batch size 有一定依赖？
6. “BYOL 对增强更鲁棒”与“BYOL 不需要增强”有什么区别？
7. MoCo 与 BYOL 都有 momentum encoder，它们的用途有何本质差异？
8. SimSiam 的结果如何修正我们对 EMA 防坍塌作用的理解？
9. DINO 继承了 BYOL 的哪些结构，又增加了哪些防坍塌机制？
10. 复现 BYOL 时，为什么仅监控 loss 不足以判断表示质量？
