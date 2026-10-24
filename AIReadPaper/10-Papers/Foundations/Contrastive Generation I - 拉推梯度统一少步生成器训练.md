---
type: paper-note
status: done
domain: Foundations
paper: "Contrastive Generation I: Training a Few-Step Generator by Pulling and Pushing Its Samples"
year: 2026
arxiv: null
doi: null
source: https://edapinenut.github.io/contrast_gen/
project: https://edapinenut.github.io/contrast_gen/
code: https://github.com/EDAPINENUT/contrast_gen
tags:
  - Foundations
  - Few-Step-Generation
  - MeanFlow
  - DMD
  - Drifting-Models
  - Pull-Push-Gradient
created: 2026-10-24
updated: 2026-10-24
---

# Contrastive Generation I：拉-推梯度统一少步生成器训练

## 基本信息

- **完整标题**：Contrastive Generation I: Training a Few-Step Generator by Pulling and Pushing Its Samples
- **作者**：Haitao Lin（HY & Talent Lab）
- **发表**：研究博客（非传统 arXiv 论文），2026 年 10 月
- **项目主页**：https://edapinenut.github.io/contrast_gen/
- **代码仓库**：https://github.com/EDAPINENUT/contrast_gen
- **定位**：这是一篇高质量技术长文，以统一视角解读少步生成器（few-step generator）训练的三种主流方法——MeanFlow、DMD 和 drifting models——揭示它们本质上使用同一个"拉-推梯度"（pull-push gradient）更新规则。

## 一句话结论

MeanFlow、DMD 与 drifting models 训练少步生成器的梯度都可分解为**拉力**（将生成点移向携带数据信息的正样本）与**推力**（将点推离描述生成器自身的负样本）的合成；三者的差异仅在于负样本的获取方式（内在 JVP / 外在 fake 网络 / 非参数核估计）和构建状态（同一噪声层级 / 重噪声样本），而正样本始终是教师速度场或数据批次的后验均值估计。

---

## 1. 研究背景与问题定义

### 1.1 少步生成的核心矛盾

Flow Matching（[[Flow Matching - 条件概率路径与连续生成]]）和扩散模型在图像生成上质量优异，但依赖数十到数百步的 ODE 积分。教师模型从噪声到可靠样本通常需要 16–64 步 Euler 或 RK4 求解——这个迭代成本是实时生成的主要障碍。少步生成器要回答的问题是：**能否训练一个网络，在 1–4 步内直接从噪声跳到数据分布？**

### 1.2 现有方法的碎片化

当前训练少步生成器的方法各自独立发展：

1. **MeanFlow**：预测"平均速度"，用一次前向从 $x_t$ 跳到 $x_r$（$r < t$），通过 Jacobian-Vector Product（JVP）获得学生自身的瞬时速度作为负样本。
2. **Distribution Matching Distillation（DMD）**（[[DMD - 分布匹配蒸馏一步图像生成]]）：训练一个额外的 fake 网络估计生成分布的分数，用反向 KL 散度驱动学生。
3. **Drifting Models**：完全不用教师，用数据批次和生成批次的核加权均值作为正负样本训练单步生成器。

这三种方法看起来动机不同、公式不同、实现不同，但作者的核心洞察是：**它们的更新规则本质相同**。

### 1.3 本文的切入角度

作者从 flow matching 的线性插值路径出发，定义"流图"（flow map）和"平均速度"的概念，然后将三种方法的梯度统一写成"pull-push"形式，系统比较它们在设计空间中的位置，并在 Two-moons 玩具数据集上以可控实验验证理论分析。

---

## 2. 理论基础：速度场与流图

### 2.1 线性插值路径

Flow matching 使用噪声 $\epsilon$ 和数据 $x_0$ 之间的线性插值：

$$x_t = (1-t)x_0 + t\epsilon$$

其中 $t \in [0,1]$，$t=0$ 为数据端，$t=1$ 为噪声端。

### 2.2 平均速度与流图

**平均速度** $u(x_t, t, r)$ 定义为从层级 $t$ 到层级 $r$（$r < t$）的"割线"速度：

$$u(x_t, t, r) = \frac{x_t - x_r}{t - r}$$

它满足一个关键性质：给定 $x_t$ 和平均速度，可以**一步精确跳到** $x_r$：

$$x_r = x_t - (t-r) \cdot u(x_t, t, r)$$

这就是"流图"（flow map）——它描述了从 $x_t$ 直接跳到更低噪声层级 $x_r$ 的映射。

### 2.3 教师与学生

教师模型通过 ODE 积分需要多步才能从 $x_t$ 到达可靠的 $x_r$。学生（少步生成器）的目标是直接学习这个跳跃，用一次前向替代教师的数十步积分。核心问题是：**如何给学生提供有效的训练信号？**

---

## 3. 拉-推梯度的统一形式

### 3.1 通用 pull-push 分解

作者证明，三种方法的梯度都可以写成如下统一形式：

$$\nabla_\theta L_{pp} = -2W \left(\frac{\partial \hat{x}^r_\theta}{\partial \theta}\right)^\top (x_{pos} - x_{neg})$$

对应的损失函数：

$$L_{pp} = W \|\hat{x}^r_\theta - \text{sg}(\hat{x}^r_\theta + x_{pos} - x_{neg})\|^2$$

其中：
- $\hat{x}^r_\theta$ 是学生从 $x_t$ 一步跳到层级 $r$ 的预测
- $x_{pos}$ 是**正样本**——携带关于数据分布的信息
- $x_{neg}$ 是**负样本**——描述生成器自身当前分布
- $W > 0$ 是每样本权重
- $\text{sg}$ 表示 stop-gradient

这个分解的物理直觉是：**拉力**把学生移向正样本（数据流形），**推力**把学生推离负样本（生成器自身的错误模式）。

### 3.2 三种方法的具体实例化

| 方面 | Simulation（模拟） | MeanFlow | DMD |
|------|-------------------|----------|-----|
| **移动点** | $\hat{x}^r_\theta$ | $\hat{x}^r_\theta$ | $\hat{x} = G_\theta(\epsilon)$ |
| **正样本** | 模拟点 $\hat{x}^r_{sim}$ | $X[x_t, v_{teacher}, r]$ | $X[x_{t'}, v_{teacher}, 0]$ |
| **负样本** | 移动点自身 $\text{sg}(\hat{x}^r_\theta)$ | $X[x_t, \text{sg}(u_\theta) + J, r]$ | $X[x_{t'}, v_\phi, 0]$ |
| **构建位置** | 同一 $x_t$ | 同一 $x_t$ | 重噪声样本 $x_{t'}$ |
| **额外成本** | 完整 ODE 求解 | 一次 JVP | 第二个网络（fake） |
| **负样本类型** | — | 内在的（intrinsic） | 外在的（extrinsic） |

#### 3.2.1 MeanFlow：内在负样本

MeanFlow 的负样本来自学生网络自身：$v_{ind} = u_\theta + J$（其中 $J$ 是 Jacobian 项）。不需要额外参数，只需学生网络及其导数。收敛时匹配的是**轨迹级别（trajectory-level）**——每个 $(x_t, t, r)$ 有各自的不动点。

关键消融发现：
- 冻结负样本中的学生项 → **训练发散**（几百步内 SW2 超过 50）
- 保留学生项但去掉 $J$ → 稳定但**学不到东西**

这说明 MeanFlow 的学生项（负样本中跟随梯度更新的部分）是训练稳定性的关键——它提供了自适应的"排斥力"，防止学生坍缩到平凡解。

#### 3.2.2 DMD：外在负样本

DMD 使用独立的 fake 网络 $v_\phi$ 学习学生分布的速度场，通过反向 KL 散度训练。收敛时匹配的是**分布级别（distribution-level）**——不动点是 $p_{\theta,0} = p_{data}$。DMD 本质上是 on-policy 的：它需要在学生当前生成的样本上评估。

#### 3.2.3 Simulation：理想但不可扩展

Simulation 用完整 ODE 求解的结果作为正样本，用移动点自身作为负样本。它是理论上最直接的"轨迹蒸馏"，但计算成本最高（每步都需要完整 ODE 积分），实际中不可扩展。

---

## 4. 设计空间：三个独立选择

作者将三种方法的差异归纳为三个正交的设计维度：

### 4.1 状态来源

- **Off-policy**：使用噪声数据（如随机采样的 $x_t = (1-t)x_0 + t\epsilon$，$x_0$ 来自数据集）
- **On-policy**：使用学生自身生成样本的加噪版本

实验表明 on-policy 对 DMD 至关重要（KL 0.45 vs off-policy 1.07），而 MeanFlow 在玩具实验中 off-policy 略优。

### 4.2 目标层级 $r$

- **固定 $r=0$**：只训练从任意 $t$ 直接跳到数据端
- **随机 $r \in [0,t]$**：训练从 $t$ 跳到任意更低层级

随机 $r$ 对 MeanFlow 的 coupling error 有显著改善（0.037 vs 0.060），因为它迫使学生在所有层级上都保持准确的流图。

### 4.3 学生瞬时速度获取方式

- **内在（MeanFlow）**：1 次 JVP，无额外网络
- **外在（DMD）**：额外训练一个 fake 网络

---

## 5. 扩展到 Drifting Models

### 5.1 无教师的单步生成

Drifting models 更进一步：完全不用教师模型。正样本是数据批次的核加权均值，负样本是生成样本批次的核加权均值。漂移场：

$$V(\hat{x}) = x_{pos} - x_{neg}$$

本质上是另一种构建后验均值的方式——用**非参数核密度估计**替代 DMD 的学习型 fake 网络。

### 5.2 三种变体

1. **Native drift**：原始 drifting model，使用 Laplacian 核
2. **FM-ideal drift**：将核替换为高斯核（与 Flow Matching 理想速度场更一致）
3. **Semi-drift**：正样本用教师，负样本用高斯核均值（无辅助网络）

### 5.3 样本量需求

| N（样本数） | Native drift | FM-ideal drift | Semi-drift |
|------------|-------------|---------------|------------|
| 2 | 发散 | 发散 | 发散 |
| 4 | 1.95 / 0.170 | 1.90 / 0.170 | 崩溃 |
| 16 | 0.83 / 0.083 | 0.88 / 0.130 | 1.86 / 0.259 |
| 256 | 0.39 / 0.053 | 0.29 / 0.047 | 0.39 / 0.085 |
| 1024 | 0.28 / 0.052 | 0.25 / 0.041 | 0.51 / 0.059 |

（格式：KL / SW2；参考 DMD trained fake 的 KL 为 0.45）

关键观察：drifting models 需要大量样本才能工作（至少 256），且 FM-ideal drift 在高样本量下略优于 native drift。Semi-drift 在中等样本量时表现不稳定。

---

## 6. 实验结果

### 6.1 玩具实验设置

- **数据**：Two-moons 数据集（噪声标准差 0.05）
- **教师**：4×256 MLP，flow matching 训练
- **学生**：相同架构的平均速度网络，从教师初始化
- **训练**：3000 步，batch 256，3 个随机种子，CPU 运行
- **全量复现**：约 40 分钟（12 个并行训练进程）

### 6.2 主要结果

| 设置 | SW2 (1步) | Coupling (1步) | KL(student‖data, 1步) | KL(student‖data, 2步) |
|------|-----------|----------------|----------------------|----------------------|
| DMD, on-policy, r=0 | **0.072** | 1.26 | **0.45** | 0.50 |
| MeanFlow, on-policy, r=0 | 0.099 | 0.060 | 1.12 | 1.68 |
| DMD, off-policy, r=0 | 0.246 | 2.52 | 1.07 | 0.51 |
| MeanFlow, off-policy, r=0 | 0.088 | 0.051 | 0.84 | 1.37 |
| DMD, on-policy, r random | 0.093 | 0.66 | 0.51 | **0.48** |
| MeanFlow, on-policy, r random | 0.080 | **0.037** | 0.95 | 0.76 |

### 6.3 关键发现

1. **分布 vs 轨迹匹配的本质差异**：在 SW2（Wasserstein-2 距离）上两种方法表现相近，但 KL 上 on-policy DMD 明显最优（0.45 vs MeanFlow 的 1.12）。然而 coupling error（衡量与教师 ODE 轨迹的偏差）反转了排名——MeanFlow 几乎完美追踪教师 ODE（0.037），DMD 则从"错误"噪声产生好样本（1.26）。这说明 MeanFlow 是精确的轨迹蒸馏，DMD 是灵活的分布匹配。

2. **On-policy 对 DMD 的必要性**：DMD on-policy 在一步时远优于 off-policy（KL 0.45 vs 1.07），因为 fake 网络需要在学生当前分布上准确估计分数。

3. **随机 $r$ 对 MeanFlow 的增益**：随机层级训练使 MeanFlow 的 coupling error 从 0.060 降到 0.037，因为它强制学生在所有中间层级都保持准确。

---

## 7. 局限与评价

### 7.1 高维不可扩展性

非参数核估计（drifting models 的核心）在图像空间中需要指数级增长的样本量。文中指出 drifting models 在 ImageNet 上依赖预训练编码器的特征空间（4096 维潜空间，仅 64 个样本），当只有 1 个正样本时 FID 从 8.46 升至 20.43。这意味着 drifting models 在实际大规模生成中不如 DMD 或 MeanFlow。

### 7.2 实验规模有限

所有实验在 Two-moons 玩具数据集上完成，虽然可控性强、便于理论验证，但距离 ImageNet 256×256 或更高分辨率的实际生成场景有很大差距。在大规模设置下，JVP 的内存开销、fake 网络的训练稳定性、on-policy 采样的效率等问题可能呈现不同的面貌。

### 7.3 理论贡献与实用价值

本文的核心价值是**理论统一**——它揭示了 MeanFlow、DMD 和 drifting models 不是三种独立的方法，而是同一个"pull-push gradient"框架在不同设计选择下的实例化。这为未来方法设计提供了清晰的设计空间导航图。但作为研究博客，它没有提出新的方法或改进，也未在大规模任务上验证新的 SOTA。

### 7.4 与既有笔记的关联

- [[DMD - 分布匹配蒸馏一步图像生成]]：本文从 pull-push 视角重新解读 DMD 的"外在负样本"机制
- [[Flow Matching - 条件概率路径与连续生成]]：本文的理论起点，平均速度和流图的概念直接建立在 Flow Matching 之上
- [[Consistency Models - 一步与少步一致性生成]]：另一种少步生成范式，本文的框架未涵盖但可作为对照

### 7.5 个人判断

★★★★☆（4/5）。这是一篇思路清晰、技术深度足够的统一性综述。它的价值不在于提出新方法，而在于**把碎片化的文献整合为一个设计空间**，让研究者能够系统地思考"正样本从哪来、负样本从哪来、在什么状态构建"这三个问题。对理解少步生成这一快速演进领域的底层逻辑非常有帮助。局限是实验规模小、无 arXiv 正式发表、部分结论可能在高维下不成立。
