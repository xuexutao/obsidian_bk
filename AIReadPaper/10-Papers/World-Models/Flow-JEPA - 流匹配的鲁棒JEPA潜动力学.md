---
type: paper-note
status: complete
domain: World-Models
paper: "Flow-JEPA: Robust Latent Dynamics for JEPA World Models via Flow Matching"
year: 2026
arxiv: "2608.29029"
doi: "10.48550/arXiv.2608.29029"
source: "https://arxiv.org/abs/2608.29029v3"
project: null
code: "https://github.com/HuoYanchen/Flow-JEPA"
tags:
  - World-Models
  - JEPA
  - Flow-Matching
  - Model-Predictive-Control
  - Robustness
created: 2026-10-15
updated: 2026-10-15
---

# Flow-JEPA：流匹配的鲁棒 JEPA 潜动力学

## 基本信息

- **论文**：Flow-JEPA: Robust Latent Dynamics for JEPA World Models via Flow Matching
- **作者**：Yanchen Huo、Ziying Song、Yadan Luo
- **机构**：南洋理工大学、昆士兰大学
- **版本**：arXiv v3，2026-09-29
- **代码**：https://github.com/HuoYanchen/Flow-JEPA
- **直接基线**：LeWorldModel（LeWM）

## 一句话结论

Flow-JEPA 保留 LeWM 的无重建 JEPA 表征与 SIGReg，却把“一步潜状态回归、递归 rollout”改成“以当前状态和动作序列为条件，从高斯源联合输运整段未来 latent trajectory”；四个短时域控制环境的平均成功率在 clean 条件下 86%→92%、局部视觉噪声下 67%→86%，但论文尚未拆分“联合轨迹预测”和“随机 flow path”各自贡献，且每次规划需要 8 次向量场求值，单规划周期约慢 3.8×。

---

## 1. 研究背景与问题定义

### 1.1 JEPA 世界模型的承诺

JEPA 不重建像素，而是预测未来观测的抽象 embedding，希望把容量集中在任务相关状态，避免背景纹理、光照等无关细节。LeWorldModel 用 next-embedding prediction+SIGReg，从像素端到端训练紧凑世界模型，并在 latent space 用 MPC 规划。

但“无像素重建”不自动等于鲁棒。LeWM 仍使用确定性的一步模型递归预测：

$$\hat z_{t+k}=g_\phi(\mathcal H_{t+k-1},a_{t+k-1})$$

训练只监督干净一步转移：

$$\mathcal L_{\mathrm{pred}}=\|g_\phi(\mathcal H_t,a_t)-z_{t+1}\|_2^2$$

早期误差会被重新喂给模型并累积；局部视觉噪声将$z_t$推离训练流形时，也没有机制把它拉回有效轨迹。

![](assets/Flow-JEPA%20-%20流匹配的鲁棒JEPA潜动力学/figures/01-success-rates.png)

> 图 1：四环境 clean/noisy 成功率。F-JEPA 在所有环境都表现出更小的噪声退化；注意纵轴从 0.5 开始。来源：论文 Figure 1，p.2，https://arxiv.org/abs/2608.29029v3

### 1.2 核心假设

条件流匹配从随机源轨迹到干净未来轨迹学习向量场，训练点覆盖插值路径的邻域，因此模型可能天然学会“把受扰 latent 拉回有效未来”；整段未来联合生成还消除了递归调用的一步误差链。

---

## 2. 任务定义与输入输出

视觉编码器$f_\xi$把当前观测$o_t$编码为$z_t\in\mathbb R^d$，动作编码器$h_\psi$把动作编码为$e_t^a$。给定当前观测和未来$h$步动作序列，模型输出未来潜轨迹：

$$Z=[z_{t+1},z_{t+2},\ldots,z_{t+h}]\in\mathbb R^{h\times d}$$

$$E^a=[e_t^a,e_{t+1}^a,\ldots,e_{t+h-1}^a]$$

主实验$h=5$。它不解码未来像素，只把终点 latent 与目标 latent 的距离作为 MPC/CEM 规划代价。

---

## 3. 核心方法

### 3.1 条件流匹配基础

采样源$x_0\sim\mathcal N(0,I)$、目标$x_1\sim p_{\mathrm{data}}$与$\tau\sim U[0,1]$，线性插值：

$$x_\tau=(1-\tau)x_0+\tau x_1$$

目标速度：

$$u_\tau=x_1-x_0$$

标准 CFM 损失：

$$\mathcal L_{\mathrm{CFM}}=\mathbb E\|v_\theta(x_\tau,\tau)-(x_1-x_0)\|_2^2$$

训练无需积分 ODE，只在随机中间状态回归速度场。

### 3.2 轨迹级 Flow-JEPA

对未来每个位置独立采样高斯源$z^0_{t+i}$，组成$Z_0$。插值整段轨迹：

$$Z_\tau=(1-\tau)Z_0+\tau Z$$

向量场以插值轨迹、流时间、当前 latent 和动作序列为条件：

$$v_\theta(Z_\tau,\tau,z_t,E^a)$$

训练目标：

$$\mathcal L_{\mathrm{FM}}=\mathbb E\left\|v_\theta(Z_\tau,\tau,z_t,E^a)-(Z-Z_0)\right\|_2^2$$

总损失为：

$$\mathcal L=\mathcal L_{\mathrm{FM}}+\lambda_{\mathrm{SIG}}\mathcal L_{\mathrm{SIG}}$$

Push-T 的$\lambda_{\mathrm{SIG}}=0.09$，其他环境 0.1。

![](assets/Flow-JEPA%20-%20流匹配的鲁棒JEPA潜动力学/figures/02-method.png)

> 图 3：训练时编码当前/未来观测，Gaussian source trajectory 通过向量场对齐到未来 latent，SIGReg 防坍塌；预测器用 self-attention 联合更新未来 token，以当前状态和动作作 cross-attention context，flow time 经 AdaLN 注入。来源：论文 Figure 3，p.5。

### 3.3 SIGReg 的角色

对潜变量矩阵$Z\in\mathbb R^{N\times d}$，在随机单位方向$u^{(m)}$上投影：

$$h^{(m)}=Zu^{(m)}$$

对每个一维投影计算 Epps–Pulley 正态性统计量$\mathcal T$：

$$\mathcal L_{\mathrm{SIG}}(Z)=\frac1M\sum_{m=1}^{M}\mathcal T(h^{(m)})$$

SIGReg 约束 encoder 聚合分布接近各向同性高斯，防止表示坍塌；Flow source 则定义动力学输运起点。二者都涉及高斯，但作用不同。SIGReg 不保证单个受扰观测与原观测具有等价动力学，这正是 flow predictor 要补的部分。

### 3.4 网络结构

- 输入图像：224×224；
- 视觉编码器：ViT-Tiny，潜维度 192；
- 动作 MLP：输出 192 维；
- 向量场 Transformer：6 blocks、16 heads、head dim 64、FFN 2048；
- 流时间用 sinusoidal embedding+AdaLN；
- 当前状态与动作序列作 cross-attention context。

Two-Room/OGBench-Cube 使用$\mathcal N(0,I)$源与双向未来 token attention；Reacher/Push-T 使用$\mathcal N(z_t,0.5^2I)$与因果 attention。论文没有对 source 和 attention 模式做数值消融。

### 3.5 推理与 ODE

推理从$Z_0$开始积分：

$$\frac{dZ_\tau}{d\tau}=v_\theta(Z_\tau,\tau,z_t,E^a)$$

用显式 Euler：

$$Z_{\tau+\Delta\tau}=Z_\tau+\Delta\tau v_\theta(Z_\tau,\tau,z_t,E^a),\quad\Delta\tau=1/N_{\mathrm{flow}}$$

主设置$N_{\mathrm{flow}}=8$。增加到 16/32 步并不提升控制成功率，说明数值积分精度不是主要瓶颈。

### 3.6 MPC/CEM 规划

每次规划 5 个 action blocks，每 block 含 5 个底层动作，覆盖 25 个环境步；CEM 每轮采 300 条候选，取 30 条 elite 更新分布，只用预测轨迹终点计算代价。选出的 5-block 序列执行完再重规划。

Push-T 主结果中 F-JEPA 用 10 次 CEM refinement，LeWM 报告值用 30 次；论文另做同预算对照，证明提升不是因为搜索预算更大。

---

## 4. 数据与扰动设置

### 4.1 四个环境

- Two-Room：目标导航；
- Reacher：平面机械臂到达；
- Push-T：推动 T 形物体；
- OGBench-Cube：目标条件立方体操控。

训练 20 epochs，batch 128，单张 RTX 6000 Ada。每项成功率基于 50 rollouts。

### 4.2 局部高斯视觉扰动

只在任务无关背景施加空间衰减高斯噪声，保持物理状态与动力学不变：

$$\tilde o_{p,k}=\operatorname{clip}\left(o_{p,k}+\exp\left(-\frac{\|p-c\|_2^2}{2r^2}\right)\epsilon_{p,k},0,255\right)$$

$$\epsilon_{p,k}\sim\mathcal N(0,s^2)$$

主设置$s=100$；Two-Room/Reacher/OGBench-Cube 的$r=35$，Push-T 用$r=10$，因为$r=35$会让 LeWM 接近随机水平。

---

## 5. 实验结果

### 5.1 Clean 成功率

| 方法 | Two-Room | Reacher | Push-T | Cube | Mean |
|---|---:|---:|---:|---:|---:|
| PLDM | 97 | 78 | 78 | 65 | 80 |
| DINO-WM | **100** | 79 | 74 | **86** | 85 |
| LeWM | 87 | 86 | **96** | 74 | 86 |
| **F-JEPA** | **100** | **90** | **96** | 82 | **92** |

F-JEPA 平均提升 6 点，但 OGBench-Cube 仍低于 DINO-WM 4 点。PLDM/DINO-WM/LeWM 的 clean 数字来自 LeWM 论文，不是本文全部重跑。

### 5.2 Noisy 成功率

| 方法 | Two-Room | Reacher | Push-T | Cube | Mean |
|---|---:|---:|---:|---:|---:|
| LeWM | 76 | 74 | 58 | 58 | 67 |
| **F-JEPA** | **96** | **90** | **90** | **66** | **86** |

平均提升 19 点；clean→noisy 的下降从 LeWM 的 19 点缩到 F-JEPA 的 6 点。噪声对比只有 LeWM，没有 PLDM/DINO-WM。

![](assets/Flow-JEPA%20-%20流匹配的鲁棒JEPA潜动力学/figures/03-tworoom-rollouts.png)

> 图 4：Two-Room 中 clean 条件两者都到达目标；背景加入任务无关噪声后，LeWM rollout 偏离，F-JEPA 仍成功。来源：论文 Figure 4，p.8。

### 5.3 噪声强度与半径压力测试

Reacher 在$r=10\ldots40$所有半径上 F-JEPA 都领先 2–16 点；固定$r=35$、$s=50\ldots200$时领先 6–20 点。附录跨环境扫描中大多保持优势，唯一反转是 OGBench-Cube、$s=150$时 F-JEPA 60%、LeWM 62%。因此结论是“整体显著更鲁棒”，不是“任何扰动都严格领先”。

### 5.4 Flow steps/NFE

Push-T noisy：4/8/16/32 步成功率分别 82/90/82/84，时间 35.0/47.7/78.0/130.2 秒（50 rollouts）。8 步是最佳性能—效率折中；更细积分反而不稳定，说明 world model 表示和规划误差大于 ODE 数值误差。

### 5.5 CEM 预算

相同 10 iter 下，Push-T clean 为 F-JEPA 96 vs LeWM 94，noisy 为 90 vs 60。CEM 从 10 增到 30 并未让 F-JEPA 更好，noisy 反而从 90 降到 84，可能因为更强搜索会利用模型误差，但论文没有深入分析。

### 5.6 计算代价

Two-Room 相同 10 CEM iter：LeWM 单次预测 2.1ms、单规划 105ms、50 rollouts 21.7s；F-JEPA 每 flow step 5.0ms、单规划 400ms、50 rollouts 34.4s。单规划慢约 3.8×，端到端只慢 1.6×，作者解释为更有效规划需要更少重规划周期。

---

## 6. 与相关工作的关系

Flow-JEPA 是 [[LeWorldModel - 像素端到端稳定JEPA世界模型]]的直接动力学替换：表征仍是 JEPA+SIGReg，变化发生在“未来如何展开”。与像素视频 flow model 不同，它在抽象 latent trajectory 上输运，不要求生成可视未来；与 Dreamer/RSSM 的随机 latent 不同，它直接学习整段条件向量场。

它也与 Goku 等生成模型的 flow matching 形成有趣对照：相同数学工具，在 Goku 中用来生成视觉 latent，在 Flow-JEPA 中用来生成可规划状态轨迹。flow matching 因而不只是画图算法，而是一种连续条件分布建模接口。

---

## 7. 局限与批判性评价

### 7.1 两个改变没有拆开

F-JEPA 同时从“一步”改为“整轨迹”，又从“确定性回归”改为“随机 flow”。没有联合轨迹 MSE、autoregressive flow 等中间基线，无法判断鲁棒性主要来自哪一项。作者在 limitations 中明确承认这一点。

### 7.2 初始化不完全统一

Push-T 从自回归 checkpoint 初始化并冻结视觉 encoder/projector，Reacher 也热启动；另外两个环境从头训练。这削弱“只改变动力学形式”的严格性。

### 7.3 任务短、扰动简单

预测跨度仅$h=5$，环境仅四个；视觉偏移主要是背景高斯 patch。现实中的光照、背景替换、遮挡、相机抖动、任务相关物体噪声和动力学变化尚未验证。

### 7.4 推理成本

8 次向量场调用让单规划周期变慢 3.8×。长时域是否会因联合轨迹更新而摊薄成本，只是作者推测，未实证。

### 7.5 未来工作

论文明确提出：受控拆分联合轨迹与 flow 的贡献；更长时域任务；更广泛分布偏移；更高效的 flow latent predictor、低 NFE 或蒸馏。

---

## 8. 复现与实践建议

官方代码已给出 four-task 配置，并复用 LeWM 数据。建议先复现 Push-T：

1. 准备 LeWM HDF5 数据和 checkpoint；
2. 核对 action block=5、horizon=5、latent dim=192；
3. 实现轨迹级$Z_\tau$和动作前缀因果约束；
4. 先测 clean，再启用像素高斯 patch；
5. 扫描 NFE 4/8/16，确认 8 步附近最优。

不要只比较单次 forward latency，应同时报告规划周期、任务完成重规划次数与 end-to-end rollout 时间。

---

## 9. 个人判断

**★★★★☆。** 论文规模不大，但问题切得准：JEPA 的抽象表示仍可能因动力学形式而脆弱。19 点 noisy 提升很有说服力，且压力测试、CEM 预算和 NFE 成本都报告得较完整。扣分点是核心机制未拆分、初始化不统一、任务和噪声都偏玩具。它更像一篇强“方向验证”论文，为 latent world model 的非自回归轨迹建模打开了明确路线。

---

## 10. 材料来源

- [arXiv v3](https://arxiv.org/abs/2608.29029v3)
- [官方代码](https://github.com/HuoYanchen/Flow-JEPA)
- [本地来源清单](assets/Flow-JEPA%20-%20流匹配的鲁棒JEPA潜动力学/sources.md)
