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

# Flow Matching：条件概率路径与连续生成

> **论文**：Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. *Flow Matching for Generative Modeling*  
> **版本**：arXiv:2210.02747v2，2023-02-08（初稿发表于 2022 年）  
> **来源**：[arXiv 摘要页](https://arxiv.org/abs/2210.02747) · [官方 PDF](https://arxiv.org/pdf/2210.02747) · [[assets/Flow Matching - 条件概率路径与连续生成/sources/paper.pdf|本地 PDF]]  
> **本地材料**：[[assets/Flow Matching - 条件概率路径与连续生成/sources.md|来源与图片核验记录]]  
> **精读日期**：2026-10-15

## 1. 一句话总结（TL;DR）

Flow Matching（FM）把连续归一化流（Continuous Normalizing Flow, CNF）的训练改写为**对预先指定概率路径之速度场的监督回归**：虽然边际速度场不可计算，但可以采样“以单个数据点为条件”的高斯路径并回归其条件速度，且 Conditional Flow Matching（CFM）与理想 FM 具有相同的期望梯度。这个视角既包含 diffusion probability path，也允许使用更直、更易积分的条件 OT displacement path，从而在论文的 ImageNet 实验中得到更快训练、更低 ODE 求解成本和更好的 NLL/FID 权衡。

## 2. 研究背景与核心问题

### 2.1 从离散层 Normalizing Flow 到连续时间 CNF

Normalizing Flow 用一系列可逆映射把简单先验分布变成数据分布。CNF 则把离散层极限化为时间连续的 ODE：网络不再输出一次变换，而是在每个时间与空间位置输出瞬时速度。

这带来两个优点：

1. 理论上可表达非常一般的连续概率路径；
2. 通过瞬时变量替换公式仍可评估 likelihood。

但经典 CNF 的最大似然训练必须反复求解 ODE，还要计算向量场散度，训练成本高、序列依赖强，因此很难扩展到 ImageNet 级图像。

### 2.2 Diffusion 为什么易训练，却限制了路径设计

Diffusion model 通过固定前向加噪过程获得可采样的条件分布，再用 denoising score matching 学 score。其关键工程优势是：训练时随机抽一个时间点即可，不必完整模拟轨迹。

代价是概率路径通常来自某类预设扩散 SDE。路径选择被 VP、VE 等过程束缚，且接近数据端时可能快速去噪、产生较强数值刚性；采样常需很多网络评估或专门的快速求解器。

### 2.3 论文真正要回答的问题

> 能否像 diffusion 一样进行 simulation-free 训练，却不把路径限制在 diffusion 过程上，而是直接规定任意易处理的概率路径并学习它的确定性流？

论文的答案是：

- 先规定一族容易采样的**条件概率路径**；
- 写出每条条件路径的解析速度；
- 用 CFM 回归这些条件速度；
- 利用条件混合定理保证学到的网络对应正确的**边际概率路径**。

这里最重要的概念转移是：**从“先定义随机扩散过程，再推出路径”转向“直接定义路径，再构造生成该路径的向量场”。**

## 3. 核心贡献

1. **提出 Flow Matching 目标。** 给定目标概率路径及其速度场，直接用平方误差训练 CNF 的向量场，不在训练环中求解 ODE。
2. **提出可计算的 Conditional Flow Matching。** 证明条件目标与不可计算的边际 FM 目标只差一个与模型参数无关的常数，因此两者期望梯度一致。
3. **给出一般高斯条件路径的解析速度。** 任意可微的均值日程 $\mu_t$ 与标准差日程 $\sigma_t$ 都能导出闭式条件向量场。
4. **统一 diffusion path 与非 diffusion path。** VE/VP diffusion probability path 是该高斯族的特例；训练无需显式从 SDE 出发。
5. **引入条件 OT displacement interpolation。** 线性变化的均值与方差产生直线、常速的条件轨迹，并在论文实验中改善训练、生成质量与 ODE 采样效率。

## 4. CNF、流与连续性方程

### 4.1 流 ODE

令数据空间为 $\mathbb{R}^d$，时间相关向量场为 $v_t:\mathbb{R}^d\to\mathbb{R}^d$。它定义流映射 $\phi_t$：

$$
\frac{d}{dt}\phi_t(x)=v_t\!\left(\phi_t(x)\right),
\qquad \phi_0(x)=x.
$$

若 $x_0\sim p_0$，则 $x_t=\phi_t(x_0)$ 的密度为

$$
p_t=[\phi_t]_*p_0,
$$

其中 $[\phi_t]_*$ 表示 push-forward。用神经网络 $v_t(x;\theta)$ 参数化速度场，就得到 CNF。

### 4.2 连续性方程：概率质量守恒

向量场 $v_t$ 生成概率路径 $p_t$ 的充要条件是满足连续性方程：

$$
\partial_t p_t(x)+\nabla_x\cdot\bigl(p_t(x)v_t(x)\bigr)=0.
$$

直觉上：某一区域概率质量的变化，等于穿过区域边界的概率流量。该 PDE 是全文三个理论结果的连接点：

- 验证某个条件速度是否生成条件路径；
- 证明条件路径混合后仍由相应的边际速度生成；
- 导出沿 ODE 轨迹的 log-density 变化。

需要注意：给定同一 $p_t$，生成它的速度场不唯一。可以加入不改变密度的散度自由分量，例如旋转流。论文选择由高斯仿射传输诱导的“最简单”速度，避免无意义的绕行。

### 4.3 likelihood 仍然可计算

沿着轨迹 $x_t=\phi_t(x_0)$，瞬时变量替换公式为

$$
\frac{d}{dt}\log p_t(x_t)=-\nabla\cdot v_t(x_t).
$$

积分得到

$$
\log p_1(x_1)=\log p_0(x_0)-\int_0^1 \nabla\cdot v_t(x_t)\,dt.
$$

实际评估数据点 $x_1$ 时，要把状态从 $t=1$ 反向积分到 $t=0$，同时累计散度。高维图像中完整 Jacobian trace 很贵，论文使用 Hutchinson trace estimator：若 $\mathbb E[zz^\top]=I$，则

$$
\nabla\cdot v_t(x)=\operatorname{tr}\!\left(\frac{\partial v_t}{\partial x}\right)
=\mathbb E_z\left[z^\top\frac{\partial v_t}{\partial x}z\right].
$$

因此 FM 的训练不需要 ODE，但 likelihood 评估仍需要 ODE 与散度估计；“simulation-free”只描述**训练目标**，不表示整个模型永远不做数值积分。

## 5. 从 Flow Matching 到 Conditional Flow Matching

### 5.1 理想的 FM 目标

设 $p_t$ 是从简单先验 $p_0=p$ 到目标附近分布 $p_1\approx q$ 的概率路径，$u_t$ 是生成 $p_t$ 的目标速度场。Flow Matching 定义为

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t\sim U[0,1],\,x\sim p_t}
\left\|v_t(x;\theta)-u_t(x)\right\|^2.
$$

若网络精确回归 $u_t$，从 $p_0$ 积分学习到的 ODE 就会沿 $p_t$ 到达 $p_1$。

问题在于：真实数据只有样本，没有显式密度 $q$；边际 $p_t$ 与边际速度 $u_t$ 通常也没有闭式表达。因此直接计算 FM loss 不可行。

### 5.2 用“以数据点为条件”的路径构造边际路径

对每个数据样本 $x_1\sim q$，定义简单条件路径

$$
p_t(x\mid x_1),
$$

使得：

- $t=0$ 时都等于同一个噪声先验 $p(x)$；
- $t=1$ 时集中在 $x_1$ 附近，如 $\mathcal N(x_1,\sigma_{\min}^2I)$。

对数据分布混合后得到边际路径：

$$
p_t(x)=\int p_t(x\mid x_1)q(x_1)\,dx_1.
$$

终点是围绕所有训练样本的小高斯混合，因此 $p_1\approx q$。若每条条件路径由 $u_t(x\mid x_1)$ 生成，则定义

$$
u_t(x)
=\int u_t(x\mid x_1)
\frac{p_t(x\mid x_1)q(x_1)}{p_t(x)}\,dx_1.
$$

这也可写成后验条件期望：

$$
u_t(x)=\mathbb E\left[u_t(x\mid x_1)\mid x_t=x\right].
$$

这条式子非常关键：在同一个位置 $x$，许多“指向不同训练样本”的条件速度会竞争；边际速度是按“该样本在当前位置产生质量的后验概率”加权后的平均。

### 5.3 定理 1：条件流如何合成为正确的边际流

把条件连续性方程对 $q(x_1)$ 积分：

$$
\begin{aligned}
\partial_t p_t(x)
&=\int \partial_t p_t(x\mid x_1)q(x_1)\,dx_1\\
&=-\int \nabla\cdot\left[p_t(x\mid x_1)u_t(x\mid x_1)\right]q(x_1)\,dx_1\\
&=-\nabla\cdot\left[p_t(x)u_t(x)\right].
\end{aligned}
$$

所以边际 $u_t$ 的确生成混合后的 $p_t$。这一步把“容易写出的单样本条件流”提升为“正确的数据分布级流”。

### 5.4 CFM：真正可训练的目标

论文改为优化

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=\mathbb E_{t,\,x_1\sim q,\,x\sim p_t(\cdot\mid x_1)}
\left\|v_t(x;\theta)-u_t(x\mid x_1)\right\|^2.
$$

一次训练样本只需：

1. 抽数据 $x_1$；
2. 抽时间 $t$；
3. 从条件路径抽 $x_t$；
4. 计算解析条件速度；
5. 做普通监督回归。

不需要沿时间展开、不需要求解 ODE、不需要评估 $q(x_1)$，也不需要显式计算难解的边际速度。

### 5.5 定理 2：FM 与 CFM 为什么等价

将平方误差展开。两种目标的模型相关项分别是

$$
\mathbb E\|v_t(x;\theta)\|^2
-2\mathbb E\langle v_t(x;\theta),u_t(\cdot)\rangle.
$$

由于边际采样等价于先抽 $x_1$ 再抽 $x\mid x_1$，二者第一项相同；又因为

$$
u_t(x)=\mathbb E[u_t(x\mid x_1)\mid x],
$$

交叉项也相同。因此

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathcal L_{\mathrm{FM}}(\theta)+C,
$$

其中

$$
C=
\mathbb E\|u_t(x\mid x_1)\|^2-
\mathbb E\|u_t(x)\|^2
$$

与 $\theta$ 无关，从而

$$
\nabla_\theta\mathcal L_{\mathrm{CFM}}
=
\nabla_\theta\mathcal L_{\mathrm{FM}}.
$$

更直观地说，平方回归的最优预测器就是条件均值：

$$
v_t^*(x)=\mathbb E[u_t(x\mid x_1)\mid x_t=x]=u_t(x).
$$

这里是**期望目标/期望梯度**等价，不是说单个 minibatch 的 CFM loss 数值等于 FM loss。定理还依赖 $p_t(x)>0$、积分与微分可交换、相关函数可积/有界等正则条件。

## 6. 一般 Gaussian probability paths

### 6.1 条件路径与边界条件

论文考虑各向同性高斯条件路径：

$$
p_t(x\mid x_1)=
\mathcal N\!\left(x\mid \mu_t(x_1),\sigma_t(x_1)^2I\right),
$$

并设置

$$
\mu_0(x_1)=0,\quad \sigma_0(x_1)=1,
\qquad
\mu_1(x_1)=x_1,\quad \sigma_1(x_1)=\sigma_{\min}.
$$

于是所有条件路径从同一标准高斯出发，终止于数据点附近的小高斯。

### 6.2 仿射条件流及解析速度

取 $x_0\sim\mathcal N(0,I)$，定义

$$
\psi_t(x_0)=\sigma_t(x_1)x_0+\mu_t(x_1).
$$

显然 $\psi_t(x_0)\sim p_t(\cdot\mid x_1)$。对时间求导，并用

$$
x_0=\frac{x-\mu_t(x_1)}{\sigma_t(x_1)}
$$

消去潜变量，可得条件向量场（定理 3）：

$$
u_t(x\mid x_1)
=
\frac{\dot\sigma_t(x_1)}{\sigma_t(x_1)}
\left(x-\mu_t(x_1)\right)
+
\dot\mu_t(x_1).
$$

训练时甚至不必先写出上式。直接重参数化：

$$
x_t=\psi_t(x_0),
\qquad
\dot x_t=\dot\sigma_t(x_1)x_0+
\dot\mu_t(x_1),
$$

然后优化

$$
\mathbb E_{t,x_1,x_0}
\left\|v_t(x_t;\theta)-\dot x_t\right\|^2.
$$

这说明 FM 的工程核心可以压缩成一句话：**随机生成带时间标签的“位置—速度”监督对。**

## 7. Diffusion probability paths 是高斯路径的特例

### 7.1 VE 路径

把传统 data-to-noise 的 Variance Exploding（VE）过程反向写成 noise-to-data：

$$
p_t(x\mid x_1)=
\mathcal N\!\left(x\mid x_1,\sigma_{1-t}^2I\right).
$$

代入一般公式可得

$$
u_t(x\mid x_1)
=-\frac{\dot\sigma_{1-t}}{\sigma_{1-t}}(x-x_1).
$$

其方向始终朝向 $x_1$，但缩放系数随时间变化，并在端点附近可能变得尖锐。

### 7.2 VP 路径

令

$$
\alpha_t=\exp\!\left(-\frac12T(t)\right),
\qquad
T(t)=\int_0^t\beta(s)\,ds.
$$

反向的 Variance Preserving（VP）条件路径是

$$
p_t(x\mid x_1)=
\mathcal N\!\left(
 x\mid \alpha_{1-t}x_1,
 \left(1-\alpha_{1-t}^2\right)I
\right).
$$

因此

$$
\mu_t=\alpha_{1-t}x_1,
\qquad
\sigma_t=\sqrt{1-\alpha_{1-t}^2},
$$

再代入高斯速度公式即可得到论文式（19）的 conditional VF。

论文在附录中从 Fokker–Planck 方程出发，证明这个速度与 diffusion 的 deterministic probability-flow ODE 一致。也就是说，**FM 并未否定 diffusion 的 probability-flow ODE，而是给出了另一种直接学习同一 ODE 向量场的方法。**

### 7.3 FM-Diffusion 与 score matching 的区别

二者可以使用相同的 VP probability path，但监督对象不同：

- score matching 学 $\nabla_x\log p_t(x)$ 或条件 score，再转换为 ODE drift；
- FM-Diffusion 直接回归该概率路径的速度 $u_t(x)$。

因此，“path”和“training objective”是两个可分离的设计轴：

| 概率路径 | 训练目标 | 论文简称 |
|---|---|---|
| VP diffusion path | score / noise matching | SM-Diffusion、DDPM、ScoreFlow |
| VP diffusion path | conditional velocity regression | FM-Diffusion |
| 条件 OT path | conditional velocity regression | FM-OT |

Diffusion path 在有限时间通常不能精确到达理想高斯端点，且数据端可能出现 $\sigma_t\to0$ 的奇异性。论文实验中对 VP 路径使用 $t\in[0,1-\varepsilon]$，$\varepsilon=10^{-5}$；这也是它比下一节的有限端点 OT 路径更麻烦之处。

## 8. OT displacement interpolation：直线条件路径

### 8.1 线性均值与线性标准差

论文选择

$$
\mu_t(x_1)=t x_1,
\qquad
\sigma_t=1-(1-\sigma_{\min})t.
$$

于是

$$
x_t
=\left[1-(1-\sigma_{\min})t\right]x_0+t x_1,
$$

沿每个已配对的 $(x_0,x_1)$，轨迹导数是常数：

$$
\dot x_t=x_1-(1-\sigma_{\min})x_0.
$$

将 $x_0$ 消去，条件速度也可写成

$$
u_t(x\mid x_1)
=
\frac{x_1-(1-\sigma_{\min})x}
{1-(1-\sigma_{\min})t}.
$$

因此 CFM 训练目标化为

$$
\mathcal L_{\mathrm{CFM}}
=
\mathbb E_{t,x_1,x_0}
\left\|
 v_t(x_t;\theta)
-
\left[x_1-(1-\sigma_{\min})x_0\right]
\right\|^2.
$$

监督目标沿单条条件轨迹不随时间改变，回归问题比强烈时变的 diffusion target 更规整。

![[assets/Flow Matching - 条件概率路径与连续生成/fig2_vector_fields.png]]

*图 1（原论文 Figure 2 的官方 TeX 面板重排）：上排 diffusion conditional score 的方向与尺度随时间明显变化；下排 OT conditional vector field 的方向保持得更一致。蓝色表示较大幅值、红色表示较小幅值。*

### 8.2 为什么称为 OT path

在两个高斯

$$
\mathcal N(0,I)
\quad\text{与}\quad
\mathcal N(x_1,\sigma_{\min}^2I)
$$

之间，上述仿射映射是 Wasserstein-2 最优传输映射；随 $t$ 线性插值就是 McCann displacement interpolation：

$$
p_t=[(1-t)\operatorname{id}+t\psi]_*p_0.
$$

它让单个粒子沿欧氏直线以常速移动，避免 diffusion 条件轨迹的弯曲、过冲和回折。

![[assets/Flow Matching - 条件概率路径与连续生成/fig3_ot_vs_diffusion_trajectories.png]]

*图 2（原论文 Figure 3 官方 TeX 原图拼接）：Diffusion 条件轨迹可能弯曲并越过终点后回折；OT 条件轨迹为直线。*

### 8.3 最容易被误读的一点：conditional OT 不等于 marginal OT

论文明确提醒：这里证明最优的是**每个条件高斯对之间的传输**。对 $x_1\sim q$ 混合后，边际速度是许多条件速度的后验平均，通常不能推出它是从 $p_0$ 到 $q$ 的全局 Monge OT map，也不能直接声称学到的 CNF 实现全局最小 Wasserstein action。

所以更准确的名称是“使用条件 OT displacement interpolation 的 Flow Matching”，而不是“求解了噪声分布到数据分布的全局最优传输”。

## 9. 训练与 ODE 采样

### 9.1 最小训练算法

对每个 minibatch：

1. 抽取数据 $x_1\sim q$；
2. 抽取先验噪声 $x_0\sim\mathcal N(0,I)$；
3. 抽取 $t\sim U[0,1]$；
4. 根据选定日程构造
   $$x_t=\sigma_t(x_1)x_0+\mu_t(x_1);$$
5. 构造监督速度
   $$u_t=\dot\sigma_t(x_1)x_0+\dot\mu_t(x_1);$$
6. 最小化
   $$\|v_t(x_t;\theta)-u_t\|^2.$$

这与普通监督学习同构，内存与计算不再随 ODE 步数增长。论文使用 Dhariwal & Nichol 的 U-Net，只做适应连续时间向量场输出所需的最小修改。

### 9.2 生成阶段

先抽

$$
x_0\sim\mathcal N(0,I),
$$

再解初值问题

$$
\frac{dx_t}{dt}=v_t(x_t;\theta),
\qquad t:0\to1,
$$

输出 $x_1$。论文默认评估使用 `dopri5` 自适应 Runge–Kutta 求解器，绝对与相对容差均为 $10^{-5}$；低成本实验还比较 Euler、midpoint 和 RK4 固定步长方法。

### 9.3 “训练快”与“采样快”来自不同机制

- **训练快**：CFM 不通过 ODE 反传，只回归随机时间点的解析速度。
- **采样快**：若学到的向量场平滑、轨迹较直，低阶求解器用更少 NFE（Number of Function Evaluations）也能保持误差。

前者由目标构造直接保证；后者不是纯理论保证，而是路径几何、网络拟合误差与数值求解器共同作用的经验结果。

## 10. 实验设置、结果与消融

### 10.1 设置

- **数据集**：CIFAR-10；ImageNet 32×32、64×64、128×128；另做 ImageNet 64→256 超分辨率。
- **架构**：图像模型统一使用 Dhariwal & Nichol U-Net；二维 checkerboard 使用 5 层、每层 512 单元的 MLP。
- **优化**：Adam，$\beta_1=0.9$、$\beta_2=0.999$、weight decay 0；ImageNet-64/128/256 使用 16-bit mixed precision。
- **主要对照**：DDPM、Score Matching、ScoreFlow、FM-Diffusion、FM-OT。
- **指标**：NLL/BPD（越低越好）、FID（越低越好）、自适应 ODE 求解的平均 NFE（越低越好）。
- **公平性**：FM-OT、FM-Diffusion、SM-Diffusion 使用相同架构、超参数和 epoch；论文还让部分 baseline 训练更久以获得更好收敛。

### 10.2 主要定量结果

原论文 Table 1 的同架构比较如下：

| 数据集 | 方法 | NLL/BPD ↓ | FID ↓ | NFE ↓ |
|---|---|---:|---:|---:|
| CIFAR-10 | DDPM | 3.12 | 7.48 | 274 |
|  | Score Matching | 3.16 | 19.94 | 242 |
|  | ScoreFlow | 3.09 | 20.78 | 428 |
|  | FM-Diffusion | 3.10 | 8.06 | 183 |
|  | **FM-OT** | **2.99** | **6.35** | **142** |
| ImageNet-32 | DDPM | 3.54 | 6.99 | 262 |
|  | Score Matching | 3.56 | 5.68 | 178 |
|  | ScoreFlow | 3.55 | 14.14 | 195 |
|  | FM-Diffusion | 3.54 | 6.37 | 193 |
|  | **FM-OT** | **3.53** | **5.02** | **122** |
| ImageNet-64 | DDPM | 3.32 | 17.36 | 264 |
|  | Score Matching | 3.40 | 19.74 | 441 |
|  | ScoreFlow | 3.36 | 24.95 | 601 |
|  | FM-Diffusion | 3.33 | 16.88 | 187 |
|  | **FM-OT** | **3.31** | **14.45** | **138** |

可见在这组受控实验中，FM-OT 在三个数据集上同时取得最低 NLL、FID 与 NFE。ImageNet-128 的 FM-OT 报告 NLL 2.90、FID 20.9；论文将其与当时若干无条件 GAN 比较，并注明使用额外自监督条件的 IC-GAN 不在该表中。

### 10.3 目标函数消融：FM 是否优于 score matching

固定 VP diffusion path，比较 SM-Diffusion 与 FM-Diffusion：

- CIFAR-10：FM 的 FID 从 19.94 降到 8.06，NFE 从 242 降到 183；
- ImageNet-64：FID 从 19.74 降到 16.88，NFE 从 441 降到 187；
- ImageNet-32：FM-Diffusion 的 FID 6.37 反而略差于 Score Matching 的 5.68，NFE 也略高（193 对 178）。

因此原文“更稳定、更鲁棒”的整体判断有实验支持，但不能简化成“FM-Diffusion 在所有指标上总胜过 score matching”。最强且最一致的提升来自**同时换成 FM 目标和 OT path**。

### 10.4 路径消融：OT path 是否优于 diffusion path

固定 FM 目标，FM-OT 相比 FM-Diffusion：

- CIFAR-10：FID 8.06→6.35，NFE 183→142；
- ImageNet-32：FID 6.37→5.02，NFE 193→122；
- ImageNet-64：FID 16.88→14.45，NFE 187→138。

这组对照最直接支撑论文的路径设计主张：更简单的条件轨迹不仅让监督信号更规整，也让最终边际 ODE 更容易数值求解。

### 10.5 训练速度与采样轨迹

论文报告 ImageNet-64 上 FM-OT 的 FID 下降更快。ImageNet-128 中，Dhariwal & Nichol baseline 使用 4.36M iterations、batch 256；FM 使用 500K iterations、batch 1.5K，虽模型约大 25%，总 image throughput 仍少约 33%。这不是完全等算力比较，但说明 FM 在训练步利用率上有优势。

![[assets/Flow Matching - 条件概率路径与连续生成/fig6_imagenet64_sample_paths.png]]

*图 3（原论文 Figure 6 官方 TeX 原图重排）：三种模型使用同一初始噪声。FM-OT 较早形成全局图像结构，而 diffusion path 的可辨结构主要在末段出现。注意三种方法终点并非必须是同一图像，因为学习到的边际耦合不同。*

在固定步长、低 NFE 的 ImageNet-32 实验中，FM-OT 达到同一数值误差阈值时约只需 diffusion model 的 60% NFE；在 Euler、midpoint、RK4 下都给出更好的 FID—计算量权衡。这说明“直线路径”并非只是示意图上的美观，而会反映为 ODE 离散误差与采样成本差异。

### 10.6 条件生成：64→256 超分辨率

原论文 Table 2：

| 方法 | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| Regression | 15.2 | 121.1 | 27.9 | 0.801 |
| SR3 | 5.2 | 180.1 | 26.4 | 0.762 |
| FM-OT | **3.4** | **200.8** | 24.7 | 0.747 |

FM-OT 的感知生成指标 FID/IS 更好，但逐像素重建指标 PSNR/SSIM 更低。它更倾向产生视觉上合理的高频细节，而不是最大化与单一 ground truth 的像素一致性。

### 10.7 实验是否充分

实验很好地回答了论文自己的三个问题：FM 能否扩展、路径是否影响训练、路径是否影响 ODE 采样。但仍存在边界：

- 主要是 U-Net 图像实验，缺少语言、音频或非欧氏数据；
- 论文时代的 baseline 与后来的 DiT/大规模 latent model 不同；
- 没有把 wall-clock、能耗、硬件利用率统一折算；
- “OT 更简单”主要由条件几何与经验指标支持，缺少边际 ODE stiffness 的完整理论界。

## 11. 局限性与批判性思考

### 11.1 终点是平滑数据分布，不是严格经验分布

当 $\sigma_{\min}>0$ 时，终点

$$
p_1(x)=\int\mathcal N(x\mid x_1,\sigma_{\min}^2I)q(x_1)\,dx_1
$$

是数据分布的高斯平滑。$\sigma_{\min}$ 越小，越接近数据，但端点速度可能更尖锐；它同时扮演统计平滑与数值正则化角色。

### 11.2 条件直线不保证学到的边际轨迹完全直

条件样本对 $(x_0,x_1)$ 走直线，但网络在位置 $x_t$ 上回归的是多条条件速度的平均。若不同条件轨迹大量交叉，平均后会弯曲，甚至产生复杂局部动力学。论文 Figure 3 展示的是条件轨迹，不应当被解读为“训练后的每条生成轨迹严格是一条直线”。

### 11.3 “OT”命名可能让人高估全局保证

前面已经强调，conditional Gaussian OT 不推出 marginal OT。独立采样 $x_0\sim p_0$ 与 $x_1\sim q$ 所形成的耦合本身通常不是全局最优配对。论文给出的是一种几何上有利、易回归的条件路径设计，而非高维数据分布间 OT 问题的完整求解器。

### 11.4 训练免模拟，生成仍有数值成本

FM 消除了训练时的 ODE simulation，但采样仍需求解学习到的 ODE。即使 NFE 明显下降，也不是天然 one-step generator。采样速度依赖模型容量、向量场平滑性、容差和求解器；低 NFE 下的感知误差也可能与高精度解不同。

### 11.5 向量场回归存在条件方差

CFM 单样本监督 $u_t(x\mid x_1)$ 可能相对边际均值 $u_t(x)$ 有较大方差。当许多条件路径在同一区域交叉而速度差异大时，SGD 噪声会增大。这解释了为什么**端点 coupling 与 path geometry**不仅影响采样，也影响训练难度；后续方法常从 coupling、时间采样和 loss weighting 继续优化。

### 11.6 论文没有穷尽路径设计空间

高斯、各向同性、标量 $\sigma_t$ 是方便而非必然。原文结论也指出未来可研究非各向同性高斯或更一般 kernel。对流形、离散结构、受约束物理状态，欧氏直线插值可能离开有效数据域。

### 11.7 评估与社会影响

FID、IS、BPD 分别衡量不同侧面，不代表语义可靠性或样本多样性完全得到保证。作者还提醒图像生成可被滥用，并指出大模型训练能耗问题；更少更新与更低 image throughput 可能降低能耗，但论文没有给出直接能耗测量。

## 12. 与 Rectified Flow、SiT 的关系：原论文与后续工作严格分开

### 12.1 Rectified Flow：同期、数学目标高度相关，但研究叙事不同

**原论文事实**：Lipman et al. 在 Related Work 中把 Liu et al.（Rectified Flow，arXiv:2209.03003）列为同期工作，称其得到类似的 simulation-free conditional objective。两篇工作不是前后继承关系，不能说 Rectified Flow 是 Flow Matching 论文后来提出的模块，反之亦然。

当 $\sigma_{\min}=0$ 时，本文 OT 条件插值退化为

$$
x_t=(1-t)x_0+t x_1,
\qquad
u_t=x_1-x_0,
$$

这与 Rectified Flow 第一轮、给定端点 coupling 时的直线插值回归在核心形式上相同。共同点包括：

- 用随机时间点上的平方回归学习 ODE velocity；
- 训练时无需模拟 ODE；
- 直线路径有利于粗步长采样。

但强调重点不同：

| Flow Matching（本文） | Rectified Flow（同期工作） |
|---|---|
| 从 CNF 与概率路径出发，证明条件混合生成正确边际路径 | 从两个分布间 transport/coupling 出发，强调 rectification |
| 建立 FM/CFM 等价，并覆盖一般 Gaussian path、VE/VP diffusion path | 重点研究把任意 coupling 整流为确定性 coupling、凸 transport cost 不增 |
| OT path 是一般路径族中的一个实例 | 直线化与重复 reflow/rectification 是核心算法叙事 |
| 原文没有提出后来的多轮 reflow 训练流程 | 可递归 rectification，使轨迹进一步变直 |

因此，在现代代码里看到

$$
\mathbb E\|v_\theta((1-t)x_0+tx_1,t)-(x_1-x_0)\|^2
$$

把它称作“linear/OT Conditional Flow Matching”或“Rectified Flow objective”都可能有文献依据；准确归因应说明两条同期路线及各自理论框架。

### 12.2 SiT：2024 年的后续扩展，不属于原论文贡献

SiT（*Scalable Interpolant Transformers*, arXiv:2401.08740）发表于本文之后。它把 stochastic interpolant / flow-diffusion 设计空间与 DiT backbone 结合，系统研究：

- 离散时间还是连续时间训练；
- 预测 velocity、score 或其他参数化；
- 选择何种 interpolant；
- 使用 deterministic ODE 还是 stochastic SDE sampler。

与本文的概念联系是：都把“连接噪声与数据的路径/插值”视为一等设计对象，而不是把生成模型等同于某个固定 diffusion schedule；velocity prediction 也可通过条件回归训练。

严格区别如下：

1. **本文不是 Transformer 论文。** 原实验使用 U-Net；SiT 的“Scalable Transformer”架构与大规模条件 ImageNet 结果是后续工作。
2. **本文主线是确定性 CNF/FM。** SiT 进一步把 deterministic flow 与可调 diffusion coefficient 的 stochastic sampler 放在统一 interpolant 框架中。
3. **SiT 的 FID 与消融不能回填为本文证据。** 它们说明 FM/插值思想后来适配了 DiT 规模化路线，不是 2022 原论文已经完成的实验。

### 12.3 一条更大的技术脉络

可以把演化概括为：

1. **Diffusion / score model**：固定随机加噪过程，学习 score/noise；
2. **Flow Matching / Rectified Flow / Stochastic Interpolants（同期）**：直接学习连接两个分布的连续速度，路径与训练目标开始解耦；
3. **SiT 等后续工作**：把 interpolant、prediction target、ODE/SDE sampler 与可扩展 Transformer 架构模块化组合。

本文的历史作用不只是提出一个 loss，而是促成了“**生成建模 = 选择概率路径 + 学习运输速度 + 选择数值采样器**”这一更模块化的分析框架。

## 13. 实现时最值得保留的最小配方

### 13.1 线性 CFM / OT-CFM

给定 batch 数据 $x_1$：

$$
x_0\sim\mathcal N(0,I),\quad t\sim U[0,1],
$$

$$
x_t=\left[1-(1-\sigma_{\min})t\right]x_0+t x_1,
$$

$$
target=x_1-(1-\sigma_{\min})x_0,
$$

$$
loss=\operatorname{mean}\|v_\theta(x_t,t)-target\|^2.
$$

生成时从 $x_0$ 积分 $dx/dt=v_\theta(x,t)$。若采用 class conditioning，把类别/文本条件输入 $v_\theta$ 即可，CFM 证明结构不变。

### 13.2 常见实现陷阱

- 混淆时间方向：本文约定 $t=0$ 是 noise、$t=1$ 是 data；许多 diffusion 文献相反。
- 把 $x_1-x_0$ 当作所有 Gaussian path 的 target：它只对应 $\sigma_{\min}=0$ 的线性插值。
- 训练时错误地 detach $x_t$ 对模型输入之外的必要条件，或让 target 参与不需要的梯度图。
- 在 $t=1$ 直接计算含 $1/\sigma_t$ 的 diffusion VF，引发数值奇异；需要截断端点。
- 用条件轨迹直线性替代对实际 learned marginal trajectories 的检查。
- 报告固定步数时只写“steps”而不写求解器；Euler 20 步与 RK4 20 步的 NFE 不同。

## 14. 关键概念对照表

| 术语 | 含义 | 在本文中的作用 |
|---|---|---|
| Probability path $p_t$ | 随时间连接先验与数据的密度族 | 生成过程的设计对象 |
| Vector field $u_t$ | 使概率质量沿 $p_t$ 演化的瞬时速度 | FM 的监督目标 |
| Flow $\phi_t$ | 向量场 ODE 的解映射 | 把 $p_0$ push-forward 到 $p_t$ |
| CNF | 用神经网络参数化连续速度场的 normalizing flow | 最终生成模型 |
| FM | 回归不可直接计算的边际速度场 | 理想目标 |
| CFM | 回归可采样、可解析的条件速度场 | 实际训练目标 |
| Continuity equation | $\partial_t p+\nabla\cdot(pu)=0$ | 证明速度与路径匹配 |
| Probability-flow ODE | 与某 diffusion SDE 共享边际密度的确定性 ODE | 连接 FM 与 diffusion |
| OT displacement interpolation | 沿 OT map 线性插值的路径 | 构造直线条件轨迹 |
| NFE | 求解 ODE 时调用网络的次数 | 采样计算量指标 |

## 15. 读后结论

### 15.1 最重要的理论结论

CFM 的本质是用带噪声的条件监督样本，学习不可见的边际速度条件均值。其逻辑与 denoising score matching 相似，但对象从 score 推广到了任意可构造概率路径的 velocity。

### 15.2 最重要的工程结论

训练数据生成只需要插值与解析求导，完全不需要在训练环内解 ODE。路径越平滑、条件 target 越规整，通常越容易训练，并更可能得到低 NFE 的采样 ODE。

### 15.3 最需要记住的限定语

> FM-OT 的“OT”是条件高斯路径的 OT；它提供有利的局部/条件几何，但不自动等价于噪声分布到真实数据分布的全局最优传输。

## 16. 图表索引

1. [[assets/Flow Matching - 条件概率路径与连续生成/fig2_vector_fields.png|Diffusion score 与 OT conditional vector field]]：对应原论文 Figure 2，解释回归目标随时间的复杂度差异。
2. [[assets/Flow Matching - 条件概率路径与连续生成/fig3_ot_vs_diffusion_trajectories.png|Diffusion 与 OT 条件轨迹]]：对应原论文 Figure 3，展示弯曲/过冲与直线常速的差异。
3. [[assets/Flow Matching - 条件概率路径与连续生成/fig6_imagenet64_sample_paths.png|ImageNet-64 采样轨迹]]：对应原论文 Figure 6，展示三种训练/路径组合的去噪进程。
4. [[assets/Flow Matching - 条件概率路径与连续生成/sources.md|附件来源与逐张核验记录]]：含官方链接、TeX 内部路径、尺寸、SHA-256 与核验说明。
