---
title: Flow Matching - 条件概率路径与连续生成
aliases:
  - Flow Matching for Generative Modeling
  - Flow Matching
  - Conditional Flow Matching
tags:
  - paper
  - generative-modeling
  - flow-matching
  - continuous-normalizing-flow
  - optimal-transport
arxiv: "2210.02747"
year: 2022
status: 精读完成
---

# 0. Flow Matching：条件概率路径与连续生成

> **论文**：Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. *Flow Matching for Generative Modeling*  
> **版本**：arXiv:2210.02747v2，2023-02-08（初稿发表于 2022 年）  
> **来源**：[arXiv 摘要页](https://arxiv.org/abs/2210.02747) · [官方 PDF](https://arxiv.org/pdf/2210.02747) · [[assets/Flow Matching - 条件概率路径与连续生成/sources/paper.pdf|本地 PDF]]  
> **本地材料**：[[assets/Flow Matching - 条件概率路径与连续生成/sources.md|来源与图片核验记录]]  
> **精读日期**：2026-10-16

## 1. 一句话总结（TL;DR）

Flow Matching（FM）把连续归一化流（Continuous Normalizing Flow, CNF）的训练改写为：先指定一条从噪声到数据的概率路径，再监督网络回归生成该路径的速度场。真正不可计算的边际速度可以由可采样、可解析的条件速度替代，因为 Conditional Flow Matching（CFM）与理想 FM 只相差一个与模型参数无关的常数，因而具有相同的期望梯度。

论文的关键意义不只是提出一个 loss，而是把**概率路径、训练目标和 ODE 采样器**拆成可独立设计的模块：diffusion path 只是一般高斯路径的特例；条件 OT displacement path 则给出直线、常速的条件轨迹，并在论文的受控实验中取得更好的 NLL、FID 与 NFE 权衡。

## 2. 研究背景与核心问题

### 2.1 从 Normalizing Flow 到 Continuous Normalizing Flow

离散 Normalizing Flow 用一串可逆变换把简单先验变成复杂数据分布。CNF 把这种变换连续化：神经网络不再输出一层可逆映射，而是在每个时间与空间位置输出瞬时速度，样本通过常微分方程连续运动。

CNF 的优点是表达灵活、可计算 likelihood，并且理论上能承载广泛的概率路径；但经典最大似然训练需要在训练环中求解 ODE、反向传播并计算散度，序列化积分成本使其难以扩展到高分辨率图像。

### 2.2 Diffusion 的优势与路径约束

Diffusion model 通过预设前向加噪过程，获得可直接采样的条件分布，再用 denoising score matching 学习 score。训练只需随机抽取一个时间点，不必模拟完整轨迹，因此稳定且易扩展。

代价是概率路径通常受 VE、VP 等扩散过程约束。标准 diffusion path 在有限时间并不严格到达纯高斯端点，数据端还可能出现尖锐或近奇异的时间行为；采样时往往需要较多网络调用，或依赖专门设计的快速求解器。

### 2.3 论文要解决的问题

论文提出的问题可以压缩为一句话：

> 能否保留 diffusion 的 simulation-free 训练优势，同时摆脱对扩散过程的依赖，直接指定更一般、更容易学习和积分的概率路径？

论文的答案分四步：

1. 为每个数据样本设计一条容易采样的条件概率路径；
2. 写出生成该条件路径的解析速度；
3. 用 CFM 在随机时刻回归条件速度；
4. 通过条件混合定理保证，网络在总体上学到的是正确边际概率路径的速度。

概念上的关键转变是：**不必先定义随机 SDE 再推导路径，而可以先选择概率路径，再构造生成该路径的确定性向量场。**

## 3. 核心贡献

1. **提出 Flow Matching 目标。** 给定目标概率路径及其生成速度，直接以平方误差训练 CNF 向量场，训练时不求解 ODE。
2. **提出可计算的 Conditional Flow Matching。** 证明 CFM 与不可直接计算的边际 FM 只差参数无关常数，所以两者的期望梯度完全相同。
3. **给出一般各向同性高斯条件路径的闭式速度。** 任意满足边界条件的可微均值日程与标准差日程，都能转化为位置—速度监督对。
4. **统一 diffusion path 与非 diffusion path。** VE、VP 的 probability-flow ODE 可在同一高斯路径框架中恢复，而无需把 diffusion SDE 当作训练起点。
5. **引入条件 OT displacement path。** 线性均值与线性标准差产生直线、常速的条件轨迹；在论文实验中，它比所比较的 diffusion 路径更易训练、生成质量更好、ODE 求解所需 NFE 更少。

## 4. CNF、流与连续性方程

### 4.1 向量场与流 ODE

设数据空间为 $\mathbb R^d$，时间相关向量场为 $v_t:\mathbb R^d\to\mathbb R^d$。它定义流映射 $\phi_t$：

$$\frac{d}{dt}\phi_t(x)=v_t\!\left(\phi_t(x)\right),\qquad \phi_0(x)=x.$$

若初始样本 $x_0\sim p_0$，则 $x_t=\phi_t(x_0)$ 的密度是初始密度的 push-forward：

$$p_t=[\phi_t]_*p_0.$$

对可逆流，普通变量替换公式写成：

$$[\phi_t]_*p_0(x)=p_0\!\left(\phi_t^{-1}(x)\right)\left|\det\frac{\partial\phi_t^{-1}}{\partial x}(x)\right|.$$

用神经网络 $v_t(x;\theta)$ 参数化速度场，便得到 CNF。生成时从 $p_0$ 抽噪声并把 ODE 从 $t=0$ 积分到 $t=1$；密度评估时则从数据端反向积分。

### 4.2 连续性方程：概率质量守恒

向量场 $v_t$ 生成概率路径 $p_t$ 的判据是连续性方程：

$$\partial_t p_t(x)+\nabla_x\cdot\left(p_t(x)v_t(x)\right)=0.$$

它表达局部概率质量守恒：某一区域内密度的变化，等于穿过区域边界的净概率流。本文中，连续性方程承担三项核心工作：

- 验证条件速度确实生成指定条件路径；
- 证明条件路径混合后，相应的后验加权速度生成正确边际路径；
- 导出沿 ODE 轨迹的 log-density 变化。

给定同一条 $p_t$，生成它的速度并不唯一。可以加入满足 $\nabla\cdot(p_t w_t)=0$ 的分量，例如对旋转对称分布加入旋转流，而不改变密度。论文选用高斯仿射传输诱导的规范速度，以避免对生成无贡献的绕行。

### 4.3 瞬时变量替换与 likelihood

沿轨迹 $x_t=\phi_t(x_0)$，log-density 的变化率为：

$$\frac{d}{dt}\log p_t(x_t)=-\nabla_x\cdot v_t(x_t).$$

积分得到：

$$\log p_1(x_1)=\log p_0(x_0)-\int_0^1\nabla_x\cdot v_t(x_t)\,dt.$$

因此 FM 训练不需要 ODE，但 likelihood 评估仍需要反向积分状态，并累计散度。高维中显式 Jacobian trace 很贵，可用 Hutchinson estimator：若随机向量 $z$ 满足 $\mathbb E[zz^\top]=I$，则

$$\nabla\cdot v_t(x)=\operatorname{tr}\!\left(\frac{\partial v_t}{\partial x}\right)=\mathbb E_z\!\left[z^\top\frac{\partial v_t}{\partial x}z\right].$$

所以“simulation-free”严格指**训练目标不在内环模拟 ODE**，并不表示生成、likelihood 或诊断阶段完全不做数值积分。

## 5. 从 Flow Matching 到 Conditional Flow Matching

### 5.1 理想 Flow Matching 目标

设 $p_t$ 是从简单先验 $p_0=p$ 到目标附近分布 $p_1\approx q$ 的概率路径，$u_t$ 是生成 $p_t$ 的目标速度。理想 FM 目标为：

$$\mathcal L_{\mathrm{FM}}(\theta)=\mathbb E_{t\sim U[0,1],\,x\sim p_t}\left\|v_t(x;\theta)-u_t(x)\right\|^2.$$

若损失降到零，学习到的 CNF 就生成目标路径 $p_t$。困难在于真实数据只提供 $x_1\sim q$ 的样本，通常既没有边际密度 $p_t$ 的闭式表达，也没有边际速度 $u_t$ 的闭式表达。

### 5.2 用条件路径构造边际路径

对每个数据样本 $x_1\sim q$，定义条件路径 $p_t(x\mid x_1)$，满足：

- $p_0(x\mid x_1)=p(x)$，所有条件路径从同一噪声先验出发；
- $p_1(x\mid x_1)$ 集中在 $x_1$ 附近，例如 $\mathcal N(x_1,\sigma_{\min}^2I)$。

对数据分布混合，得到边际概率路径：

$$p_t(x)=\int p_t(x\mid x_1)q(x_1)\,dx_1.$$

终点是数据分布与小高斯核的卷积：

$$p_1(x)=\int\mathcal N(x\mid x_1,\sigma_{\min}^2I)q(x_1)\,dx_1\approx q(x).$$

若 $u_t(x\mid x_1)$ 生成条件路径，则论文定义边际速度：

$$u_t(x)=\int u_t(x\mid x_1)\frac{p_t(x\mid x_1)q(x_1)}{p_t(x)}\,dx_1.$$

Bayes 视角下，它就是后验条件均值：

$$u_t(x)=\mathbb E\left[u_t(x\mid x_1)\mid x_t=x\right].$$

这条式子解释了条件流如何汇总为边际流：同一位置 $x$ 可能由多个数据条件产生，每个条件提出不同速度；边际速度按照“该条件在当前位置产生概率质量的后验概率”进行加权平均。

### 5.3 定理 1：条件速度混合生成正确边际路径

每条条件路径满足连续性方程：

$$\partial_t p_t(x\mid x_1)+\nabla\cdot\left(p_t(x\mid x_1)u_t(x\mid x_1)\right)=0.$$

对 $q(x_1)$ 积分，并在正则条件下交换积分、微分与散度，可得：

$$\partial_t p_t(x)=-\nabla\cdot\int p_t(x\mid x_1)u_t(x\mid x_1)q(x_1)\,dx_1.$$

利用边际速度定义：

$$\int p_t(x\mid x_1)u_t(x\mid x_1)q(x_1)\,dx_1=p_t(x)u_t(x).$$

于是：

$$\partial_t p_t(x)+\nabla\cdot\left(p_t(x)u_t(x)\right)=0.$$

因此，虽然边际速度通常不能显式计算，它在理论上确实生成混合后的边际概率路径。

### 5.4 可训练的 Conditional Flow Matching

论文改为优化：

$$\mathcal L_{\mathrm{CFM}}(\theta)=\mathbb E_{t,\,x_1\sim q,\,x\sim p_t(\cdot\mid x_1)}\left\|v_t(x;\theta)-u_t(x\mid x_1)\right\|^2.$$

一次随机训练只需要：

1. 抽取数据样本 $x_1$；
2. 抽取时间 $t\sim U[0,1]$；
3. 从条件路径采样 $x_t$；
4. 计算解析条件速度 $u_t(x_t\mid x_1)$；
5. 对网络做普通平方回归。

这里不需要求解 ODE、不需要评估未知数据密度 $q(x_1)$，也不需要显式计算边际 $p_t$ 或 $u_t$。

### 5.5 定理 2：FM 与 CFM 的等价性

将 CFM 平方误差展开：

$$\mathcal L_{\mathrm{CFM}}=\mathbb E\|v_t(x;\theta)\|^2-2\mathbb E\langle v_t(x;\theta),u_t(x\mid x_1)\rangle+\mathbb E\|u_t(x\mid x_1)\|^2.$$

先抽 $x_1\sim q$、再抽 $x\sim p_t(\cdot\mid x_1)$，其边际分布正是 $x\sim p_t$，所以第一项与 FM 相同。对交叉项使用塔式法则：

$$\mathbb E\langle v_t(x;\theta),u_t(x\mid x_1)\rangle=\mathbb E\left\langle v_t(x;\theta),\mathbb E[u_t(x\mid x_1)\mid x]\right\rangle.$$

再代入 $u_t(x)=\mathbb E[u_t(x\mid x_1)\mid x]$，得到与 FM 相同的模型相关交叉项。因此：

$$\mathcal L_{\mathrm{CFM}}(\theta)=\mathcal L_{\mathrm{FM}}(\theta)+C,$$

其中：

$$C=\mathbb E\|u_t(x\mid x_1)\|^2-\mathbb E\|u_t(x)\|^2$$

与 $\theta$ 无关，所以：

$$\nabla_\theta\mathcal L_{\mathrm{CFM}}(\theta)=\nabla_\theta\mathcal L_{\mathrm{FM}}(\theta).$$

从平方回归角度，最优预测器就是条件均值：

$$v_t^*(x)=\mathbb E[u_t(x\mid x_1)\mid x_t=x]=u_t(x).$$

必须严格区分以下三种说法：

- 正确：CFM 与 FM 的**期望梯度相同**；
- 正确：两个期望目标只相差参数无关常数；
- 错误：单个 minibatch 上的 CFM loss 数值等于不可见的 FM loss。

定理还依赖 $p_t(x)>0$、相关二阶矩存在，并允许交换积分、微分和期望等正则条件。

## 6. 一般 Gaussian probability paths

### 6.1 条件高斯路径与边界条件

论文考虑各向同性高斯条件路径：

$$p_t(x\mid x_1)=\mathcal N\!\left(x\mid\mu_t(x_1),\sigma_t(x_1)^2I\right).$$

边界条件设为：

$$\mu_0(x_1)=0,\qquad \sigma_0(x_1)=1.$$

$$\mu_1(x_1)=x_1,\qquad \sigma_1(x_1)=\sigma_{\min}>0.$$

因此所有条件路径从标准高斯出发，在终点到达以 $x_1$ 为均值、方差很小的高斯。$\sigma_{\min}$ 不是纯实现细节：它让终点密度保持非退化，也决定数据分布被平滑的尺度。

### 6.2 仿射条件流

令 $x_0\sim\mathcal N(0,I)$，定义条件仿射流：

$$\psi_t(x_0)=\sigma_t(x_1)x_0+\mu_t(x_1).$$

显然 $\psi_t(x_0)\sim p_t(\cdot\mid x_1)$。对时间求导：

$$\frac{d}{dt}\psi_t(x_0)=\dot\sigma_t(x_1)x_0+\dot\mu_t(x_1).$$

又因为：

$$x_0=\frac{x-\mu_t(x_1)}{\sigma_t(x_1)},$$

所以条件速度为论文定理 3：

$$u_t(x\mid x_1)=\frac{\dot\sigma_t(x_1)}{\sigma_t(x_1)}\left(x-\mu_t(x_1)\right)+\dot\mu_t(x_1).$$

这就是任意可微高斯路径的通用公式。第一项控制以当前均值为中心的收缩或膨胀，第二项控制均值整体平移。

### 6.3 重参数化后的最简训练形式

训练时无需显式把 $x_0$ 消去。直接构造：

$$x_t=\sigma_t(x_1)x_0+\mu_t(x_1).$$

$$\dot x_t=\dot\sigma_t(x_1)x_0+\dot\mu_t(x_1).$$

然后优化：

$$\mathbb E_{t,x_1,x_0}\left\|v_t(x_t;\theta)-\dot x_t\right\|^2.$$

工程上，FM 的核心因此可以概括为：**按选定路径随机生成“时间—位置—速度”监督三元组。**

### 6.4 路径设计的三个自由度

一般高斯路径把生成建模拆成三个可分析自由度：

1. **均值日程 $\mu_t$**：决定信号以何种速度进入状态；
2. **标准差日程 $\sigma_t$**：决定噪声以何种速度消失；
3. **端点平滑 $\sigma_{\min}$**：在数据逼真度、终点正则性和数值稳定性之间折中。

论文只系统实验了 diffusion path 与线性 OT path，并没有证明线性日程对任意数据分布、网络架构或算力预算都最优。

## 7. Diffusion probability paths 是特例

### 7.1 反向 VE 路径

把传统 data-to-noise 的 Variance Exploding（VE）过程按 noise-to-data 方向书写：

$$p_t(x\mid x_1)=\mathcal N\!\left(x\mid x_1,\sigma_{1-t}^2I\right).$$

此时 $\mu_t=x_1$、条件标准差为 $\sigma_{1-t}$，代入通用公式：

$$u_t(x\mid x_1)=-\frac{\dot\sigma_{1-t}}{\sigma_{1-t}}(x-x_1).$$

速度指向 $x_1$，但时间尺度因子可在端点附近变得很大。

### 7.2 反向 VP 路径

定义：

$$\alpha_t=\exp\!\left(-\frac12T(t)\right),\qquad T(t)=\int_0^t\beta(s)\,ds.$$

反向 Variance Preserving（VP）条件路径为：

$$p_t(x\mid x_1)=\mathcal N\!\left(x\mid\alpha_{1-t}x_1,\left(1-\alpha_{1-t}^2\right)I\right).$$

对应：

$$\mu_t(x_1)=\alpha_{1-t}x_1,\qquad \sigma_t(x_1)=\sqrt{1-\alpha_{1-t}^2}.$$

通用高斯公式给出论文式（19）：

$$u_t(x\mid x_1)=\frac{\alpha'_{1-t}}{1-\alpha_{1-t}^2}\left(\alpha_{1-t}x-x_1\right).$$

等价写成 $T$ 的形式：

$$u_t(x\mid x_1)=-\frac{T'(1-t)}{2}\frac{e^{-T(1-t)}x-e^{-T(1-t)/2}x_1}{1-e^{-T(1-t)}}.$$

论文附录从 Fokker–Planck 方程出发验证：该条件速度与相应 diffusion SDE 的 deterministic probability-flow ODE 一致。FM 并未否定 diffusion；它提供的是**直接学习同一概率路径之速度场的另一种训练方式**。

### 7.3 Path 与训练目标必须分开

相同 VP probability path 可以配不同监督目标：

| 概率路径 | 学习目标 | 论文中的称呼 |
|---|---|---|
| VP diffusion path | 条件 score 或 noise | Score Matching、DDPM、ScoreFlow |
| VP diffusion path | 条件 velocity | FM w/ Diffusion |
| 条件 OT path | 条件 velocity | FM w/ OT |

因此 FM w/ Diffusion 与 score-based diffusion 的路径可以相同，差别在预测对象：前者直接预测 probability-flow ODE 的速度，后者先预测 score/noise，再换算为 ODE drift。

### 7.4 有限时间端点与截断

标准 diffusion 过程在有限时间通常只近似到达高斯先验，而不是精确到达。论文的 VP 实现使用线性 $\beta$ 日程：

$$\beta(s)=\beta_{\min}+s(\beta_{\max}-\beta_{\min}),\qquad \beta_{\min}=0.1,\quad \beta_{\max}=20.$$

时间在 $[0,1-\varepsilon]$ 上采样，$\varepsilon=10^{-5}$。这一截断避免数据端附近的奇异表达式，但也说明 diffusion path 的有限端点处理比下一节的线性条件路径更麻烦。

## 8. 条件 OT displacement path

### 8.1 线性均值与线性标准差

论文选择：

$$\mu_t(x_1)=tx_1,\qquad \sigma_t=1-(1-\sigma_{\min})t.$$

于是条件状态为：

$$x_t=\left[1-(1-\sigma_{\min})t\right]x_0+tx_1.$$

沿给定端点对 $(x_0,x_1)$，轨迹导数是常数：

$$\dot x_t=x_1-(1-\sigma_{\min})x_0.$$

消去 $x_0$ 后，条件速度也可写成：

$$u_t(x\mid x_1)=\frac{x_1-(1-\sigma_{\min})x}{1-(1-\sigma_{\min})t}.$$

CFM 目标化为：

$$\mathcal L_{\mathrm{CFM}}=\mathbb E_{t,x_1,x_0}\left\|v_t(x_t;\theta)-\left[x_1-(1-\sigma_{\min})x_0\right]\right\|^2.$$

对固定端点对，监督速度不随时间变化；这比强时变的 diffusion target 更规整。

![[assets/Flow Matching - 条件概率路径与连续生成/fig2_vector_fields.png]]

*图 1（原论文 Figure 2 的官方 TeX 面板重排）：上排展示 diffusion conditional score，下排展示 OT conditional vector field。OT 目标的方向随时间更一致；颜色表示幅值大小。注意上排是 score、下排是 velocity，图的作用是比较监督几何，而不是声称两者量纲相同。*

### 8.2 为什么称为条件 OT

在线性路径的两个条件端点：

$$p_0(x\mid x_1)=\mathcal N(0,I),\qquad p_1(x\mid x_1)=\mathcal N(x_1,\sigma_{\min}^2I)$$

之间，上述仿射映射正是二次代价下的高斯 Wasserstein-2 最优传输映射。相应 displacement interpolation 为：

$$p_t=[(1-t)\operatorname{id}+t\psi]_*p_0.$$

单个条件粒子沿欧氏直线匀速运动，不会像论文展示的 diffusion 条件轨迹那样弯曲、越过终点再回折。

![[assets/Flow Matching - 条件概率路径与连续生成/fig3_ot_vs_diffusion_trajectories.png]]

*图 2（原论文 Figure 3 官方 TeX 原图拼接）：左侧 diffusion 条件轨迹可能弯曲并出现 overshoot/backtracking；右侧条件 OT 轨迹为直线。*

### 8.3 conditional OT 不等于 marginal OT

这是全文最容易被误读的限定：

- 已证明的是每个固定 $x_1$ 下，两个条件高斯之间的传输为 OT；
- 边际路径是对所有 $x_1$ 的条件路径进行混合；
- 网络学习的是多个条件速度在给定 $x_t$ 下的后验平均；
- 这不推出边际速度是从 $p_0$ 到真实数据分布 $q$ 的全局 Monge OT 解；
- 也不推出学习到的 CNF 最小化全局 Wasserstein action。

若 $x_0$ 与 $x_1$ 独立采样，端点 coupling 本身通常不是两个边际分布间的全局最优耦合。因此更准确的说法是：**Flow Matching 使用了条件高斯端点之间的 OT displacement interpolation。**

### 8.4 条件直线不保证学习后的边际轨迹严格直线

即使每条条件轨迹都是直线，不同条件轨迹仍会在状态空间交叉。网络在交叉位置回归条件速度的平均，所得边际向量场可能弯曲。论文对条件路径的几何保证，不能直接提升为“每条最终生成轨迹都严格是直线”。

## 9. 训练、生成与 ODE 数值求解

### 9.1 通用训练算法

对每个 minibatch：

1. 抽取数据 $x_1\sim q$；
2. 抽取先验噪声 $x_0\sim\mathcal N(0,I)$；
3. 抽取 $t\sim U[0,1]$；
4. 由路径日程构造 $x_t$；
5. 由日程导数构造目标速度；
6. 最小化网络速度与目标速度的均方误差。

通用位置与速度分别为：

$$x_t=\sigma_t(x_1)x_0+\mu_t(x_1).$$

$$u_t=\dot\sigma_t(x_1)x_0+\dot\mu_t(x_1).$$

损失为：

$$\operatorname{loss}=\operatorname{mean}\left\|v_t(x_t;\theta)-u_t\right\|^2.$$

训练与普通监督学习同构，不沿时间展开，也不对 ODE 求解器反向传播。论文图像实验使用 Dhariwal 与 Nichol 的 U-Net，并做适配连续时间向量场输出所需的最小修改。

### 9.2 ODE 生成

生成先抽：

$$x_0\sim\mathcal N(0,I).$$

再求解初值问题：

$$\frac{dx_t}{dt}=v_t(x_t;\theta),\qquad t:0\rightarrow1.$$

终点 $x_1$ 是生成样本。论文默认评估用 `dopri5` 自适应 Dormand–Prince 求解器，绝对与相对容差均为 $10^{-5}$；低成本实验还比较 Euler、midpoint 与 RK4 固定步长求解器。

### 9.3 NFE 与“采样步数”

NFE（Number of Function Evaluations）是求解器调用神经网络的次数。固定 20 个时间步不一定等于 20 NFE：Euler 每步通常一次调用，midpoint 通常两次，经典 RK4 通常四次。比较采样成本时必须同时报告求解器与 NFE，不能只写模糊的“steps”。

### 9.4 训练快与采样快的机制不同

- **训练快**来自 CFM 构造：随机时刻监督回归，不在训练内环求解 ODE；
- **采样快**来自经验上的向量场平滑性和轨迹几何：路径越直、速度越规则，低阶求解器越可能以较少 NFE 达到给定误差；
- 前者由算法结构直接保证，后者受路径、端点 coupling、网络逼近误差与求解器共同影响，并非无条件理论保证。

## 10. 实验设置

### 10.1 数据集与任务

论文实验覆盖：

- CIFAR-10 无条件生成；
- ImageNet 32×32、64×64、128×128 无条件生成；
- ImageNet 64×64 到 256×256 条件超分辨率；
- 二维 checkerboard 玩具分布，用于可视化轨迹和低 NFE 行为。

### 10.2 模型、训练与评估

- 图像模型：Dhariwal 与 Nichol U-Net；
- 二维实验：5 层 MLP，每层 512 个神经元；
- 优化器：Adam，$\beta_1=0.9$、$\beta_2=0.999$、weight decay 为 0；
- 精度：CIFAR-10、ImageNet-32 使用 32-bit；ImageNet-64/128/256 使用 16-bit mixed precision；
- 主要指标：NLL/BPD、FID、自适应 ODE 求解平均 NFE；
- 默认评估：`dopri5`，`atol=rtol=1e-5`；
- likelihood：uniform dequantization，并用多个去量化样本做 importance-weighted estimate；
- 公平性：FM-OT、FM-Diffusion、SM-Diffusion 使用相同架构、超参数和 epoch；部分 diffusion baseline 被允许训练更多迭代以改善收敛。

### 10.3 主要对照轴

实验实际上同时考察两个问题：

1. 固定 diffusion path，更换训练目标：Score Matching 对 FM-Diffusion；
2. 固定 FM 训练框架，更换概率路径：FM-Diffusion 对 FM-OT。

这种两轴设计使“目标函数收益”与“路径几何收益”能够部分分离，但并没有穷尽所有 score 参数化、loss weighting、noise schedule 或现代快速采样器。

## 11. 主要结果与消融

### 11.1 同架构定量结果

原论文 Table 1 的主要结果如下，NLL 单位为 BPD，三项指标均越低越好：

| 数据集 | 方法 | NLL/BPD ↓ | FID ↓ | NFE ↓ |
|---|---|---:|---:|---:|
| CIFAR-10 | DDPM | 3.12 | 7.48 | 274 |
| CIFAR-10 | Score Matching | 3.16 | 19.94 | 242 |
| CIFAR-10 | ScoreFlow | 3.09 | 20.78 | 428 |
| CIFAR-10 | FM-Diffusion | 3.10 | 8.06 | 183 |
| CIFAR-10 | **FM-OT** | **2.99** | **6.35** | **142** |
| ImageNet-32 | DDPM | 3.54 | 6.99 | 262 |
| ImageNet-32 | Score Matching | 3.56 | 5.68 | 178 |
| ImageNet-32 | ScoreFlow | 3.55 | 14.14 | 195 |
| ImageNet-32 | FM-Diffusion | 3.54 | 6.37 | 193 |
| ImageNet-32 | **FM-OT** | **3.53** | **5.02** | **122** |
| ImageNet-64 | DDPM | 3.32 | 17.36 | 264 |
| ImageNet-64 | Score Matching | 3.40 | 19.74 | 441 |
| ImageNet-64 | ScoreFlow | 3.36 | 24.95 | 601 |
| ImageNet-64 | FM-Diffusion | 3.33 | 16.88 | 187 |
| ImageNet-64 | **FM-OT** | **3.31** | **14.45** | **138** |

在这组受控配置中，FM-OT 在三个数据集上同时取得最低 NLL、FID 和 NFE。ImageNet-128 的 FM-OT 报告 NLL 2.90、FID 20.9；论文将它与当时若干无条件 GAN 比较，但应注意 IC-GAN 使用额外自监督条件，不属于同等设置。

### 11.2 固定 diffusion path：FM 对 score matching

固定 VP diffusion path，比较不同训练目标：

- CIFAR-10：FM-Diffusion 的 FID 为 8.06，Score Matching 为 19.94；NFE 为 183 对 242；
- ImageNet-64：FID 为 16.88 对 19.74；NFE 为 187 对 441；
- ImageNet-32：FM-Diffusion 的 FID 6.37 略差于 Score Matching 的 5.68，NFE 193 也略高于 178；
- NLL 上也不是每个数据集都由 FM-Diffusion 最优。

因此论文关于 FM 训练更稳定、鲁棒的总体结论有实验支持，但不能改写成“FM 在任何 diffusion 设置、任何指标上都严格优于 score matching”。

### 11.3 固定 FM：OT path 对 diffusion path

FM-OT 相比 FM-Diffusion：

- CIFAR-10：FID 8.06→6.35，NFE 183→142；
- ImageNet-32：FID 6.37→5.02，NFE 193→122；
- ImageNet-64：FID 16.88→14.45，NFE 187→138；
- 三个数据集的 NLL 也均略有改善。

这是论文最有力的消融：在训练框架相同的前提下，更换概率路径仍带来一致收益，说明路径几何本身是重要设计变量。

### 11.4 低成本 ODE 采样

论文在 ImageNet-32 上用 Euler、midpoint、RK4 比较低 NFE 采样，并把低成本解与 1000 NFE 解的逐像素 MSE 进行对照。FM-OT 达到相同数值误差阈值时，大约只需要 diffusion 模型 60% 的 NFE；在相同低 NFE 预算下，它也给出更好的 FID—成本折中。

这个结果支持“OT 路径更容易数值积分”，但仍是特定模型、路径和求解器下的经验结论，不能推出任意 FM-OT 模型都天然具备固定倍率的加速。

### 11.5 采样轨迹可视化

![[assets/Flow Matching - 条件概率路径与连续生成/fig6_imagenet64_sample_paths.png]]

*图 3（原论文 Figure 6）：在相同初始噪声下，Score Matching + Diffusion、FM + Diffusion、FM + OT 的 ImageNet-64 轨迹。FM-OT 较早显现图像结构，而 diffusion 路径的大部分可见去噪集中在后段。*

该图展示的是不同训练/路径组合下的时间参数化行为。它不能单独证明最终样本一定更好，但与 FM-OT 的低 NFE 误差和 FID 曲线相互印证。

### 11.6 条件超分辨率

ImageNet 64×64→256×256 的结果为：

| 方法 | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| Reference | 1.9 | 240.8 | — | — |
| Regression | 15.2 | 121.1 | 27.9 | 0.801 |
| SR3 | 5.2 | 180.1 | 26.4 | 0.762 |
| FM-OT | **3.4** | **200.8** | 24.7 | 0.747 |

FM-OT 在 FID、IS 上优于 SR3，但 PSNR、SSIM 更低。这反映感知质量与逐像素保真度的典型冲突：结果支持 FM 可扩展到条件生成，却不能概括为所有质量维度全面领先。

### 11.7 实验充分性评价

实验对论文的核心主张提供了相当直接的支持：同架构比较、固定路径换目标、固定目标换路径、多个分辨率、likelihood、FID、NFE 与低成本求解器分析形成了较完整证据链。

不足在于：

- 架构主要是单一 U-Net 家族；
- 路径消融集中于 VP diffusion 与线性条件 OT；
- 没有系统分解端点 $\sigma_{\min}$、时间采样、loss weighting 与 coupling 的影响；
- 没有与后来出现的高阶 diffusion ODE/SDE sampler 或 Transformer backbone 比较；
- 能耗只做定性讨论，没有直接测量。

## 12. 局限性与批判性思考

### 12.1 终点不是原始数据分布

当 $\sigma_{\min}>0$ 时：

$$p_1(x)=\int\mathcal N(x\mid x_1,\sigma_{\min}^2I)q(x_1)\,dx_1$$

是数据分布的高斯平滑，而非严格的 $q$。减小 $\sigma_{\min}$ 可以减小平滑偏差，却可能使终点附近的速度更尖锐、训练与积分更困难。它同时是统计近似参数和数值正则参数。

### 12.2 CFM 监督存在条件方差

网络要学习的是条件速度在 $x_t$ 给定时的均值，但 SGD 每次看到的是单个条件速度。当不同条件轨迹在同一区域交叉且方向差异很大时，条件方差会增加，训练噪声也会增大。因此 path geometry 与端点 coupling 不仅影响采样曲率，也影响监督信号方差。

### 12.3 条件 OT 不是全局 OT

论文只保证单个条件高斯对之间的 OT。独立抽样得到的 $(x_0,x_1)$ coupling 通常不是先验与数据边际之间的最优 coupling；条件速度后验平均后的边际流也未被证明是全局 Benamou–Brenier geodesic。把 FM-OT 简称为“求出了噪声到数据的最优传输”是不准确的。

### 12.4 训练免模拟，生成仍有成本

FM 消除了训练内环的 ODE simulation，但生成仍需积分学习到的向量场。论文展示了 NFE 降低，却没有把 FM 变成天然 one-step generator。低步数下的表现依赖求解器、网络容量、误差容限和向量场平滑性。

### 12.5 向量场非唯一性

相同概率路径可能由多个速度场生成。论文的高斯仿射速度是自然且简单的规范选择，但“路径简单”不自动等价于“所有生成它的边际向量场都易学习”。在高维多模态数据中，后验平均仍可能产生复杂动力学。

### 12.6 路径族仍然有限

各向同性高斯、标量 $\sigma_t$ 与欧氏空间直线插值主要为解析和工程便利服务，并非普适最优。对流形、离散数据、守恒系统或受约束状态，欧氏插值可能离开有效域。原论文也把非各向同性高斯与更一般 kernel 留作未来方向。

### 12.7 指标与社会影响边界

FID、IS、BPD 衡量的是不同统计侧面，不直接保证语义正确、多样性充分或安全性。作者指出图像生成可能被滥用，也提到减少训练迭代和吞吐可能节能；但论文没有直接测量能耗，因此不能把效率指标自动换算为明确环境收益。

## 13. Rectified Flow 与 SiT：严格边界

### 13.1 Rectified Flow 是同期独立路线

Rectified Flow（Liu、Gong、Liu，arXiv:2209.03003）与本文是同期工作。Flow Matching 原文在 Related Work 中将其列为同时得到类似 simulation-free conditional objective 的工作；两者不是彼此的后续模块，也不应建立虚假的单向继承关系。

当本文线性条件路径取 $\sigma_{\min}=0$ 时：

$$x_t=(1-t)x_0+tx_1.$$

对应目标速度：

$$u_t=x_1-x_0.$$

训练目标变为：

$$\mathbb E\left\|v_\theta((1-t)x_0+tx_1,t)-(x_1-x_0)\right\|^2.$$

它与 Rectified Flow 在给定端点 coupling 下的第一轮直线插值回归具有相同核心形式。共同点是随机时间速度回归、训练时不模拟 ODE、直线条件路径有利于粗步长采样。

两条路线的理论叙事和算法边界不同：

| Flow Matching 原论文 | Rectified Flow 同期工作 |
|---|---|
| 从 CNF、概率路径和条件混合出发 | 从两个分布的 coupling 与 transport 出发 |
| 核心定理是边际路径构造及 FM/CFM 等价 | 强调整流 coupling、降低凸 transport cost |
| 一般高斯路径包含 VE、VP 与条件 OT | 直线化和 rectification 是主线 |
| 原文没有提出多轮 reflow | 可递归 reflow，使后续 coupling 与轨迹进一步变直 |

因此，现代实现中的线性速度回归可被称为 linear/OT Conditional Flow Matching，也可能被称为 Rectified Flow objective；严谨写法应交代端点 coupling、$\sigma_{\min}$、是否 reflow，以及采用哪套理论框架。

### 13.2 SiT 是 2024 年后续工作

SiT（*Scalable Interpolant Transformers*, arXiv:2401.08740）发表于本文之后，不属于 Lipman et al. 的原始贡献或实验。它把 stochastic interpolant / flow-diffusion 设计空间与可扩展 Transformer backbone 结合，系统研究 interpolant、预测参数化与 deterministic ODE / stochastic SDE sampler 的组合。

与本文的联系是：两者都把连接噪声与数据的 interpolant/probability path 视为一等设计对象，并允许直接学习 velocity。严格区别是：

1. **本文不是 Transformer 论文。** 原实验主干是 U-Net；SiT 的 Transformer 扩展和规模化结果不能回填给本文。
2. **本文主线是确定性 CNF/FM。** SiT 在更广的 stochastic interpolant 框架下同时讨论 ODE 与 SDE 采样。
3. **本文证明的是特定 FM/CFM 构造。** SiT 的参数化、训练技巧与采样结论需要按 SiT 自身理论和实验评价。
4. **SiT 的 FID 与消融不是本文证据。** 它们说明插值/流建模思想后来能与 DiT 类架构结合，而不是 2022 年论文已经完成了这些验证。

### 13.3 技术脉络中应保留的边界

可以把相关路线概括为：

1. Diffusion / score model：由随机加噪过程确定路径，学习 score/noise；
2. Flow Matching、Rectified Flow、Stochastic Interpolants：同期从不同理论起点发展 simulation-free velocity learning；
3. SiT 等后续工作：把 interpolant、prediction target、ODE/SDE sampler 与 Transformer 规模化组合。

这是一条概念演化脉络，不是“后一项都由 Flow Matching 原论文提出”的贡献链。

## 14. 实现配方与常见陷阱

### 14.1 线性 CFM / OT-CFM 最小配方

抽样：

$$x_0\sim\mathcal N(0,I),\qquad x_1\sim q,\qquad t\sim U[0,1].$$

构造位置：

$$x_t=\left[1-(1-\sigma_{\min})t\right]x_0+tx_1.$$

构造目标：

$$\operatorname{target}=x_1-(1-\sigma_{\min})x_0.$$

计算损失：

$$\operatorname{loss}=\operatorname{mean}\left\|v_\theta(x_t,t)-\operatorname{target}\right\|^2.$$

生成时从高斯噪声出发，积分 $dx/dt=v_\theta(x,t)$。若加入类别、文本或低分辨率图像条件，把条件输入向量场即可；CFM 的条件混合逻辑本身不变。

### 14.2 常见实现陷阱

- **时间方向混淆**：本文约定 $t=0$ 为 noise、$t=1$ 为 data；许多 diffusion 文献采用相反方向。
- **错误复用线性 target**：$x_1-x_0$ 只对应 $\sigma_{\min}=0$ 的直线插值；一般高斯路径应使用 $\dot\sigma_t x_0+\dot\mu_t$。
- **忽略端点奇异性**：diffusion VF 在数据端可能含趋于零的分母，需要时间截断或稳定参数化。
- **把条件直线当成边际直线**：应实际检查 learned ODE trajectory，而不是只看条件插值公式。
- **只报告 steps 不报告 NFE**：不同求解器每步网络调用次数不同。
- **把 simulation-free 误写成 sampling-free**：训练免 ODE，不代表生成免 ODE。
- **把 conditional OT 误写成 global OT**：若未优化端点 coupling，就没有全局最优传输保证。

## 15. 关键概念对照表

| 术语 | 含义 | 在本文中的作用 |
|---|---|---|
| Probability path $p_t$ | 连接先验与数据的时间相关密度族 | 生成过程的设计对象 |
| Vector field $u_t$ | 使概率质量沿路径演化的瞬时速度 | FM 的监督目标 |
| Flow $\phi_t$ | 向量场 ODE 的解映射 | 将 $p_0$ push-forward 到 $p_t$ |
| CNF | 用神经网络参数化连续速度场的 normalizing flow | 最终生成模型 |
| Continuity equation | 概率质量守恒 PDE | 验证速度与路径匹配 |
| FM | 回归边际速度场 | 理想目标 |
| CFM | 回归可采样的条件速度场 | 实际训练目标 |
| Probability-flow ODE | 与 diffusion SDE 共享边际密度的确定性 ODE | 连接 FM 与 diffusion |
| OT displacement interpolation | 沿 OT map 线性插值的概率路径 | 构造直线条件轨迹 |
| NFE | ODE 求解中网络函数调用次数 | 衡量采样计算量 |
| Reflow | Rectified Flow 的递归再整流过程 | 不属于本文原算法 |
| Stochastic interpolant | 同时组织确定性流与随机扩散的插值框架 | SiT 等后续路线的基础之一 |

## 16. 读后结论

### 16.1 最重要的理论结论

CFM 用可观测的条件速度样本学习不可直接计算的边际速度条件均值。连续性方程保证这个边际速度生成正确混合路径，平方回归与塔式法则保证 CFM 和理想 FM 的期望梯度相同。

### 16.2 最重要的工程结论

训练数据生成只需要插值与解析求导，不需要在训练内环求解 ODE。路径越平滑、条件监督越规则，通常越容易优化，也越可能得到低 NFE 的生成 ODE；但后半句仍需用实际模型和求解器验证。

### 16.3 最需要记住的限定语

> FM-OT 的“OT”首先是条件高斯端点之间的 OT。它提供有利的条件几何，但不自动等价于噪声边际到真实数据边际的全局最优传输。

## 17. 图表索引

1. [[assets/Flow Matching - 条件概率路径与连续生成/fig2_vector_fields.png|Diffusion score 与 OT conditional vector field]]：对应原论文 Figure 2，比较两类监督目标随时间的几何复杂度。
2. [[assets/Flow Matching - 条件概率路径与连续生成/fig3_ot_vs_diffusion_trajectories.png|Diffusion 与 OT 条件轨迹]]：对应原论文 Figure 3，展示弯曲、过冲/回折与直线匀速的差异。
3. [[assets/Flow Matching - 条件概率路径与连续生成/fig6_imagenet64_sample_paths.png|ImageNet-64 采样轨迹]]：对应原论文 Figure 6，展示三种训练/路径组合的生成进程。
4. [[assets/Flow Matching - 条件概率路径与连续生成/sources.md|附件来源与逐张核验记录]]：记录官方链接、版本、图片来源、尺寸、SHA-256 与边界说明。
