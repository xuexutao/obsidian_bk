---
type: paper-note
status: complete
domain: Foundations
paper: "What, Where, and How: Probing Spatiotemporal Representations in Video Foundation Models"
year: 2026
arxiv: "2609.01551"
doi: "10.48550/arXiv.2609.01551"
source: "https://arxiv.org/abs/2609.01551"
project: "https://vid-rep-pca.netlify.app/"
code: null
code_commit: null
tags:
  - Video-Foundation-Models
  - Mechanistic-Interpretability
  - Linear-Probing
  - Representation-Geometry
  - Steering
created: 2026-11-09
updated: 2026-11-09
---

# What-Where-How：视频基础模型的时空表征探针与几何导航

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *What, Where, and How: Probing Spatiotemporal Representations in Video Foundation Models* |
| 作者 | Sharon S. Musa, Fereshteh Forghani, Harrish Thasarathan, Sonia Joseph, Matthew Kowal, Konstantinos G. Derpanis |
| 单位 | York University、Vector Institute、FAIR at Meta Superintelligence Labs、Mila & McGill University、Goodfire AI |
| 发表 | Mechanistic Interpretability Workshop at ICML 2026 |
| 版本 | arXiv v1，2026-09-01 |
| 原文 | [arXiv](https://arxiv.org/abs/2609.01551) · [本地 PDF](assets/What-Where-How%20-%20视频基础模型时空表征探针/sources/paper.pdf) |
| 交互页面 | [V-JEPA 2 · PCA Explorer](https://vid-rep-pca.netlify.app/) |
| 代码 | **截至 2026-11-09 未发现官方开源仓库**；论文、arXiv HTML 与交互页面均未给出代码链接 |
| 补充材料 | 未发现独立 Supplementary Material；数据统计、逐任务表格与额外曲线已放在论文附录 |
| 个人评级 | ★★★☆☆（3/5）：问题重要、相机运动结果扎实，但“物理未编码”和“流形 steering”的证据都应收窄表述 |

## 1. 一句话结论

这篇论文用逐层线性 SVM 探针发现：V-JEPA 2 与 VideoMAE-v2 的中层已经能高质量读出相机运动，深层能中等程度读出异常事件，但用全局均值池化后的线性表示几乎无法区分 IntPhys 2 的物理可行/违规视频。更进一步，作者观察到单个视频的时间 patch 特征在 PCA 中形成弯曲轨迹，并证明由**已观测中间特征拟合**的 spline 路径比只连端点的直线路径更贴近原轨迹；这说明表示空间存在局部曲率，但还不足以证明发现了可泛化的低维“物理流形”或真正可控的视频生成方向。

---

## 2. 研究背景与问题定义

### 2.1 从“下游效果好”转向“内部到底学了什么”

视频基础模型通常以掩码预测进行大规模自监督预训练：V-JEPA 2 预测被遮挡区域的潜在表示，VideoMAE-v2 则重建被遮挡像素。两者都能在动作识别、视频理解等任务上迁移，但下游分数并不回答三个更基础的问题：模型内部表示到底编码了什么时序概念，这些概念在网络哪一层变得可访问，以及时间信息在高维空间里是线性方向、弯曲轨迹，还是更复杂的结构。

相比图像，视频还多出时间连续性、运动、对象持久性和事件级异常。只检查最终层会把“低层运动、中层几何、深层语义”的分工压缩成一个数字；只训练下游大模型又会混入任务头本身的容量。因此作者选择冻结编码器，在每个 Transformer block 后训练轻量探针，得到表示可读出性随深度变化的曲线。

### 2.2 论文的四个问题

论文将标题中的 What、Where、How 扩展成四问：

1. **What is encoded?** 相机运动、直觉物理、异常事件三类属性能否从表示中读出？
2. **Where is it encoded?** 哪些层最容易线性读出这些属性？
3. **How is it structured?** 单个视频的逐时间特征是否形成平滑几何轨迹？
4. **Can we navigate it?** 沿弯曲轨迹的 spline/tangent steering 是否比端点直线插值更贴近经验表示？

三类探针覆盖了不同抽象层级：相机运动主要依赖几何与低—中层运动线索；直觉物理要求理解对象的持续存在、不可穿透、连续运动与形状稳定；异常检测则需要把对象、动作和场景上下文聚合为事件级判断。

### 2.3 核心贡献

第一，作者在统一协议下比较 V-JEPA 2 Large、V-JEPA 2 Giant 与 VideoMAE-v2 Giant 的逐层表示，给出三类时间属性的“可线性读出性剖面”。第二，论文没有停在分类准确率，而是保留单个视频的时间分辨率，将时间 patch 特征视作表示空间轨迹。第三，作者把这个观察变成 spline/tangent steering，并用最近邻帧与到经验轨迹的距离比较直线和曲线路径。

> **边界提醒：**线性探针测量的是“给定表示经过指定池化后，标签是否线性可访问”，不是模型是否在自身计算中使用了该信息，更不等于具备语义层面的“理解”。

---

## 3. 任务定义与输入输出

### 3.1 三个模型与逐层观测点

| 模型 | 预训练范式 | 主干 | Transformer block | 探针点数 |
|---|---|---|---:|---:|
| V-JEPA 2 Large | 预测掩码区域的 latent target | ViT-L | 24 | 25 |
| V-JEPA 2 Giant | 预测掩码区域的 latent target | ViT-G | 40 | 41 |
| VideoMAE-v2 Giant | 重建掩码像素 | ViT-G | 40 | 41 |

每个 block 都在 MLP 后取表示，再额外取最终 LayerNorm 输出。因此 ViT-L 有 $24+1=25$ 个观测点，ViT-G 有 $40+1=41$ 个观测点。附录明确把 V-JEPA 2 L 的 block 输出记为 `encoder.layer.0.mlp.fc2` 至 `encoder.layer.23.mlp.fc2`，最终输出是 `encoder.layernorm`。

这里存在一个需要复现者警惕的编号问题：附录映射给出的 V-JEPA 2 L 合法索引是 0–24，但 Table 2 的 `tilt down` 最优层写成 25；ViT-G 表中还出现 layer 41，同时表里又出现 layer 0。论文没有说明 25/41 是否改用 1-based 编号指代 LayerNorm，因此本文保留原表，不擅自把它们改成 24/40。

### 3.2 三类属性与标签

#### 3.2.1 Camera Motion：15 个二分类任务

- 平移：move up/down/left/right/in/out；
- 旋转：pan left/right、tilt up/down、roll clockwise/counterclockwise；
- 变焦：zoom in/out；
- 静止：static。

这些标签不一定要求三维重建，但至少需要从连续帧中读出视角变化。值得注意的是，没有单帧、乱序帧或反向播放对照，因此高 AUC 不能排除运动模糊、场景构图或数据集偏差。

#### 3.2.2 Intuitive Physics：4 个条件与 1 个聚合任务

- **solidity**：对象不应相互穿透；
- **permanence**：对象应持续存在；
- **continuity**：对象应在时空中连续移动；
- **immutability**：对象应维持形状与结构；
- **all**：聚合全部条件。

任务是区分 possible 与 impossible 事件。它比相机运动更接近对象关系和因果约束，但本文探针先把所有时空 token 均值池化成单一向量，可能主动抹去“对象 A 在时刻 $t_1$ 与时刻 $t_2$ 的关系”这类结构。

#### 3.2.3 Anomaly Detection：事件级二分类

任务是在 UBnormal 中区分 normal 与 abnormal。异常可能来自动作、对象交互或场景动态偏离常态，需要更高层语义聚合。

### 3.3 输出与评价指标

对每个模型、任务与层，输出一条 ROC-AUC；再通过 validation folds 选择最优层和 SVM 正则参数 $C$，最终在测试集报告 ROC-AUC。ROC-AUC 的随机机会水平是 0.5（或 50%）。

---

## 4. 核心方法

### 4.1 视频级特征：先冻结，再做全时空均值池化

给定第 $l$ 层的视频 token，作者先对空间与时间 patch 全部取均值，得到一个视频级向量 $x_i^{(l)}$，随后做 $L_2$ 归一化。层 $l$ 的探针数据可写成：

$$\{(x_i^{(l)},y_i)\}_{i=1}^{N}$$

其中 $y_i$ 是对应的二分类标签。这个设计的优点是探针极简、不同模型接口统一；缺点是会丢失对象、时间位置和成对关系，尤其可能不利于 IntPhys 2。

### 4.2 线性 SVM 探针

每层单独训练一个线性 SVM：

$$g(x)=w^\top x+b$$

其中 $w$ 与 $b$ 是探针参数。作者选择线性模型，是为了避免强任务头“重新学会”目标任务，从而把分数解释为标签在冻结表示中的线性可访问性。

但“线性不可读”只有条件性含义：它可能表示模型没编码目标信息，也可能表示信息分布式、非线性、位于 token 关系中，或被全局均值池化破坏。论文自己也承认非线性 probe 可能在 IntPhys 2 上揭示额外结构。

### 4.3 层与正则参数的联合搜索

SVM 正则参数搜索范围为：

$$C\in\{10^{-3},10^{-2},\ldots,10^3\}$$

同时搜索所有候选层。ViT-L 对应约 $25\times7=175$ 个 layer–$C$ 组合，ViT-G 对应约 $41\times7=287$ 个组合。每层先得到最优 $C_l^*$，再按 validation ROC-AUC 选最优层；最终 test ROC-AUC 使用 validation folds 选出的层与超参。

实验重复 10 次，并报告均值与标准差。CameraBench 与 UBnormal 在每次运行前对多数类做欠采样；IntPhys 2 不做该平衡。论文没有说明 validation fold 数、十次运行究竟是否重切数据、欠采样是否只作用于训练集，以及随机种子，因此完整复现仍缺关键配置。

### 4.4 从视频级分类转向逐时间轨迹

为了观察几何，作者不再对时间维池化，而是只平均空间 patch。第 $l$ 层、时间位置 $t$ 的特征定义为：

$$f_t=\frac{1}{S}\sum_{s=1}^{S}H_l^{(t,s)},\qquad t\in\{0,\ldots,T-1\}$$

其中 $H_l^{(t,s)}$ 是第 $l$ 层、时间位置 $t$、空间 patch $s$ 的表示，$S$ 是空间 patch 数，最终得到序列 $\{f_t\}_{t=0}^{T-1}$。

以附录披露的 V-JEPA 2 L 为例，输入分辨率 $256\times256$、patch size 16、tubelet size 2，则：

$$S=(256/16)^2=256$$

每个时间 token 覆盖两帧，因此：

$$T=N_{\mathrm{frames}}/2$$

16 帧对应 $T=8$，64 帧对应 $T=32$；缓存激活的形状是 $(T\times S,D)$，V-JEPA 2 L 的 $D=1024$。例如 32 帧视频对应 $16\times256=4096$ 个 token，空间平均后得到 $[16,1024]$ 的时间序列。

### 4.5 单视频、单层 PCA：用于观察，不是内在维数估计

作者对**单个 clip、单个层**的 $\{f_t\}$ 独立做 PCA，并投影到三维。点的颜色用对应时间的相机平移参数编码，从而观察颜色是否沿轨迹平滑变化。

![](assets/What-Where-How%20-%20视频基础模型时空表征探针/figures/01-temporal-pca-trajectories.png)

> 图 1：V-JEPA 2 L 在 RealEstate10K 上的逐层时间 patch PCA。早层的点云结构较松散，中层与后层逐渐出现弯曲轨迹，颜色沿相机 $y$ 方向平移平滑变化。每个层的 PCA 都独立拟合，因此只能比较“是否有局部结构”，不能把各列坐标轴当成同一全局坐标系。来源：论文 Figure 1，PDF p.4，[arXiv](https://arxiv.org/abs/2609.01551)。

这幅图支持“相邻时间表示形成有序曲线”的定性观察，但论文没有报告前三主成分 explained variance、participation ratio、重构误差或 local intrinsic dimension。故“低维流形”仍是待检验假设，而不是已被 PCA 图证明的事实。

### 4.6 Cubic spline 与局部切向 steering

给定同一 clip 的时间特征 $\{f_t\in\mathbb{R}^d\}_{t=0}^{T-1}$ 和两个锚点 $f_a,f_b$（$a<b$），作者使用锚点之间的**实际中间特征**拟合三次样条：

$$s:[0,1]\rightarrow\mathbb{R}^d,\qquad s(t_j)=f_j,\quad j=a,\ldots,b$$

其中 $t_j\in[0,1]$ 是归一化时间坐标。接着计算局部切向量 $s'(t)$，并用前向 Euler 更新：

$$f_{i+1}^{\mathrm{spline}}=f_i^{\mathrm{spline}}+\Delta t\,s'(t_i)$$

$$\Delta t=\frac{1}{N-1},\qquad t_i=\frac{i}{N-1},\qquad f_0^{\mathrm{spline}}=f_a$$

这里 $N$ 是插值步数。这条路径是 clip-specific 的：它不从多视频学习全局 steering field，而是沿某一个视频已经观测到的时间轨迹重走。

### 4.7 Linear baseline

直线基线只使用两端点：

$$f_{i+1}^{\mathrm{lin}}=f_i^{\mathrm{lin}}+\Delta,\qquad \Delta=\frac{f_b-f_a}{N-1},\qquad f_0^{\mathrm{lin}}=f_a$$

如果真实时间轨迹有曲率，端点 chord 会切过曲线内部并离开经验支持区域；spline 则由于看到了中间点，天然更容易贴着原轨迹。

### 4.8 Steering 的定性与定量评价

定性评价对每个 steering 表示使用余弦相似度，在**原视频自身的时间特征集合**中检索最近邻帧；定量评价计算它到同一经验轨迹最近特征的欧氏距离。距离越小，说明路径越贴近已观测表示。

这个评价可以回答“直线 chord 是否偏离经验轨迹”，但存在明显的非独立性：spline 用中间经验点拟合，评价又奖励接近这些经验点。因此它不能单独证明 spline 学到了可泛化流形，也没有展示 latent 被解码成新的生成视频。Figure 5 中的图像是最近邻检索帧，不是模型生成的 steered video。

---

## 5. 数据集与实验设置

### 5.1 数据统计

| 任务 | 数据集 | Train | Test | 划分与标签 | 每轮欠采样 |
|---|---|---:|---:|---|---|
| 相机运动 | CameraBench | 1402 | 1071 | 预定义 train/test；正文称全集含 3000+ 视频 | 是 |
| 直觉物理 | IntPhys 2 | 1012（公开训练集总数） | 官方标签不可用 | 从公开训练集做分层 70/30；论文未列划分后精确整数 | 否 |
| 异常检测 | UBnormal | 332 | 211 | 预定义 train/test；normal/abnormal | 是 |
| 几何与 steering | RealEstate10K | 未披露 | Figure 6 使用 200 clips | 密集逐帧相机轨迹；只分析 V-JEPA 2 L | — |

CameraBench 正文说有 3000 多个视频，但 Table 1 的 Train+Test 只有 2473，论文没有解释筛选掉的样本、不同二分类任务的有效样本量或缺失标签。IntPhys 2 的官方测试标签不可用，使本文的 70/30 自建测试只能视作内部 held-out 评估，不能等同于官方 benchmark 测试。

### 5.2 误差条到底表示什么

- Figure 2：任务族内部的**跨子任务**标准差；异常检测只有一个任务，所以没有误差条。
- Figure 3：逐层曲线的阴影同样是跨子任务标准差。
- Figure 6：200 个 clip 上的标准差。

因此 Figure 2/3 的误差条不是十次运行的置信区间，也不能直接用于判断模型差异是否显著。论文没有给 paired bootstrap、run-level CI 或层选择频率。

### 5.3 复现所缺配置

论文未披露完整 checkpoint 标识、SVM solver、validation fold 数、全部视频采样率与 clip 截取方式、随机种子、欠采样的作用范围、spline 边界条件、锚点选择策略和插值步数。没有官方代码使这些缺口无法通过实现反查。

---

## 6. 实验结果

### 6.1 三类能力呈现清楚的层级差异

![](assets/What-Where-How%20-%20视频基础模型时空表征探针/figures/02-layerwise-probing.png)

> 图 3：三类任务的逐层交叉验证 ROC-AUC。相机运动在早层已经高于机会水平，中层达到高位；异常检测随深度总体上升，在更深层达到峰值；直觉物理在所有模型、所有层都接近或低于 0.5。来源：论文 Figure 3，PDF p.6，[arXiv](https://arxiv.org/abs/2609.01551)。

这张图最可靠的结论不是“某层产生某种能力”，而是三类标签对线性读出的依赖不同：相机运动更接近中层几何信号，异常检测更依赖高层语义聚合，而 IntPhys 2 的 possible/impossible 标签没有在当前读取协议下形成稳定线性边界。

### 6.2 CameraBench：中层相机运动信号最强

三模型在 15 个相机运动任务上的 mean test AUC 分别为：

| 模型 | Mean test AUC | 主要峰值深度 |
|---|---:|---|
| V-JEPA 2 Large | 91.0% | 约 60%–70%，个别任务更深 |
| V-JEPA 2 Giant | 90.2% | 约 60%–70% |
| VideoMAE-v2 Giant | **93.8%** | 约 60%–70% |

V-JEPA 2 从 Large 扩到 Giant 没有提升，反而从 91.0% 降至 90.2%；这说明“更大”并不自动提高相机运动标签的线性泛化。VideoMAE-v2 在 CameraBench 最强，但不能直接归因于像素重建目标，因为三个模型没有控制主干、预训练数据与训练预算。

附录的逐任务最佳层与 CV AUC 如下，格式为“层 / AUC%”：

| 任务 | V-JEPA 2 L | V-JEPA 2 G | VideoMAE-v2 G |
|---|---:|---:|---:|
| move down | 12 / 88.4 | 41 / 90.6 | 28 / 91.4 |
| move up | 21 / 90.1 | 31 / 92.1 | 23 / 91.5 |
| move in | 15 / 96.6 | 30 / 97.1 | 22 / 97.2 |
| move out | 15 / 88.3 | 28 / 89.6 | 25 / 90.7 |
| move left | 15 / 91.6 | 30 / 93.0 | 22 / 94.2 |
| move right | 15 / 93.9 | 31 / 93.1 | 22 / 94.5 |
| pan left | 15 / 95.9 | 28 / 96.1 | 22 / 98.1 |
| pan right | 15 / 96.2 | 28 / 95.9 | 24 / 97.8 |
| tilt up | 10 / 97.4 | 30 / 97.7 | 28 / 97.9 |
| tilt down | 25 / 96.1 | 30 / 96.9 | 18 / 97.0 |
| zoom in | 16 / 84.7 | 21 / 85.8 | 29 / 88.9 |
| zoom out | 17 / 97.2 | 30 / 96.9 | 24 / 97.7 |
| roll cw | 19 / 94.7 | 29 / 95.8 | 24 / 98.0 |
| roll ccw | 19 / 92.5 | 36 / 94.6 | 27 / 97.7 |
| static | 12 / 97.0 | 20 / 97.1 | 14 / 96.8 |

`zoom in` 是三模型共同的相对弱项；`move in`、pan、tilt、zoom out 等多项超过 95%。不过缺少单帧与乱序对照，使我们无法判断分类器到底依赖真实时序、光流样式、运动模糊，还是 CameraBench 的静态背景偏差。

### 6.3 IntPhys 2：当前探针没有发现稳健物理违规信号

| 子任务 | V-JEPA 2 L：层 / CV AUC% | V-JEPA 2 G：层 / CV AUC% | VideoMAE-v2 G：层 / CV AUC% |
|---|---:|---:|---:|
| all | 15 / 49.5 | 33 / 50.7 | 4 / 47.5 |
| solidity | 3 / 43.2 | 0 / 50.0 | 3 / 42.8 |
| permanence | 20 / 45.3 | 0 / 50.0 | 10 / 43.1 |
| continuity | 17 / 50.9 | 35 / 51.2 | 25 / 44.5 |
| immutability | 15 / 53.2 | 35 / **53.7** | 17 / 44.3 |

最高 CV AUC 只有 53.7%，而且各层曲线没有稳定抬升。Figure 7 进一步显示，CameraBench 的 `pan right` 在 layer 15 出现明显类别分离，而 IntPhys 2 的 `solidity` 在 layer 3 与 layer 20 都高度重叠。

![](assets/What-Where-How%20-%20视频基础模型时空表征探针/figures/06-class-pca-separation.png)

> 图 7：上排 CameraBench 在中层形成较明显的类别结构；下排 IntPhys 2 的 possible/impossible 样本在三维 PCA 中仍重叠。PCA 只能展示前三主成分中的可见结构，不能排除高维非线性信号。来源：论文 Figure 7，PDF p.15，[arXiv](https://arxiv.org/abs/2609.01551)。

论文将结果解释为模型没有稳健编码“更深层的物理推理”。更严谨的结论应缩为：**在合成 IntPhys 2 域上，经过全时空均值池化的冻结表示没有形成可泛化的线性违规信号。** 训练表中还出现多个值得 sanity check 的异常，例如 V-JEPA 2 L `all` 的 train AUC 只有 25.8%，VideoMAE-v2 `continuity` 为 28.9%，而 V-JEPA 2 G `immutability` 又有 84.2% train / 53.7% val；论文没有解释标签方向、score 符号或采样流程。

### 6.4 UBnormal：更深层出现中等异常信号

| 模型 | 最优层 | 归一化深度 | 最优 $C$ | CV AUC | Test AUC |
|---|---:|---:|---:|---:|---:|
| V-JEPA 2 L | 19 / 24 | 79% | 1000 | 83.4 | 67.7 |
| V-JEPA 2 G | 34 / 40 | 85% | 1000 | 86.5 | **68.4** |
| VideoMAE-v2 G | 25 / 40 | 63% | 1000 | 80.4 | 62.1 |

异常检测峰值比相机运动更深，符合“需要聚合对象、动作和场景语义”的直觉。V-JEPA 2 G 的 test AUC 68.4% 高于 VideoMAE-v2 G 的 62.1%，与 CameraBench 排名相反。作者猜测 latent prediction 目标更适合语义异常，但这只是解释性假设。

三模型的最优 $C$ 都位于搜索上界 1000，且 CV 到 test 下跌 18–20 个百分点，提示超参网格可能截断、validation winner's curse 或 train/test 分布差异。论文没有 nested CV，因此不应把“第 34 层优于第 33 层”当成稳定定位。

### 6.5 时间轨迹不只是一条单调曲线

Figure 1 的多个 clip 在中层呈现平滑弯曲轨迹；但当相机主要沿单方向平移、空间变化有限、帧数增加时，作者还观察到正弦或振荡式轨迹。

![](assets/What-Where-How%20-%20视频基础模型时空表征探针/figures/03-oscillatory-geometry.png)

> 图 4：同一向上平移视频在 layer 5 与 layer 15、16/32/64 帧下的时间特征。上半部分是 PCA 轨迹，下半部分是时间 patch 两两余弦相似度。帧数增加后轨迹出现振荡，相似度矩阵呈局部平滑与近块状/带状结构。来源：论文 Figure 4，PDF p.7，[arXiv](https://arxiv.org/abs/2609.01551)。

作者明确说没有确定解释，只提出一种假设：若相似度主要由某个连续变量上的距离决定，近 Toeplitz 的相似度结构可能产生 Fourier-like 正弦模态。这一现象反而提醒我们，视频表示的几何可能比“单一一维流形”更复杂；PCA 曲线不能被直接等价成可解释物理坐标。

### 6.6 Spline 比直线更贴近原轨迹，但证据带有构造性优势

![](assets/What-Where-How%20-%20视频基础模型时空表征探针/figures/04-spline-steering.png)

> 图 5：两段 RealEstate10K 视频的表示空间 steering。灰色是原始时间轨迹，红色是端点直线，绿色是由中间真实特征拟合的 spline/tangent 路径。右侧展示原始、Linear 与 Spline 路径对应的最近邻帧；Spline 的时间推进更连贯，Linear 更容易跳帧或错配内容。来源：论文 Figure 5，PDF p.8，[arXiv](https://arxiv.org/abs/2609.01551)。

![](assets/What-Where-How%20-%20视频基础模型时空表征探针/figures/05-nearest-neighbor-distance.png)

> 图 6：200 个 clip 上，Spline 在几乎所有层到经验轨迹的 mean nearest-neighbour distance 都低于 Linear；差距在中间层尤其明显。来源：论文 Figure 6，PDF p.9，[arXiv](https://arxiv.org/abs/2609.01551)。

结果说明端点 chord 的确会切离弯曲的经验轨迹。但 spline 看到了锚点间的真实特征，随后又用“离这些真实特征多近”作为评价，存在一定循环性。更公平的验证应只用端点或稀疏控制点拟合，把未参与拟合的中间特征作为 held-out target，并比较 piecewise-linear、cubic spline、kNN graph geodesic 等基线。

### 6.7 论文没有传统意义上的充分消融

作者提供了跨模型、跨层、跨帧数对比，但缺少决定主张强度的消融：

- 没有单帧、重复帧、随机乱序、时间反转；
- 没有 mean pooling、CLS、token-level、关系型 probe 对比；
- 没有受控容量的非线性 probe 与标签置换；
- 没有 class-weighted SVM 对多数类欠采样；
- 没有固定层/固定 $C$ 对联合选择；
- 没有 held-out spline 中间点；
- 没有真正解码视频或测量 steering 后下游行为。

因此，这篇工作的价值更像一张“值得继续验证的现象地图”，而不是对视频基础模型内部机制的闭环证明。

---

## 7. 与相关工作的关系

### 7.1 相比传统线性 probing

论文继承了用简单分类器检验冻结表示的经典范式，但把目标从图像语义、3D 几何扩展到相机运动、物理与异常三种时序属性，并系统绘制层深曲线。它与 [[ViT - 图像分块与纯Transformer视觉建模]]、DINO 系列的中间层分析一脉相承，但强调视频中的 temporal accessibility。

### 7.2 相比视频模型可解释性

过去的视频可解释工作常研究注意力、场景偏差、静态/动态信息比例或概念发现。本文的增量在于把“能否读出”“在哪一层”“几何长什么样”“能否沿几何移动”串成一条链路。

### 7.3 相比激活 steering

语言模型中的 steering 常寻找全局线性方向；近期工作则强调信念状态可能位于弯曲流形，直线干预会 off-manifold。本文把这一直觉移到视频表征，但其 spline 是单 clip 的经验轨迹重建，还不是可跨视频迁移的语义方向。

---

## 8. 局限与批判性评价

### 8.1 作者明确承认的局限

1. IntPhys 2 与 UBnormal 是合成视频，视觉统计可能偏离模型预训练的自然视频。
2. 线性 probe 只能检测线性可解码信息；非线性 probe 可能改变 IntPhys 2 的结论。
3. 几何与 steering 只覆盖 V-JEPA 2 L + RealEstate10K，尚不能证明是视频模型的一般性质。
4. 振荡轨迹的机制仍是开放问题。

### 8.2 线性可读出不等于模型理解或使用

CameraBench 的高 AUC 证明标签可以从特定层表示线性恢复，但不证明模型形成了显式相机运动变量，更不证明下游预测依赖它。没有因果干预、表示擦除或行为变化实验时，“编码”应理解为“可读出”。

### 8.3 “物理未编码”的结论过强

IntPhys 2 的 negative result 很重要，但其含义受三重限制：合成域偏移、样本量仅 1012、以及全时空均值池化。物理违规经常依赖对象级对应与时间顺序，均值池化恰好消除这些结构。更准确的说法是“未发现线性全局摘要中的物理违规信号”。

### 8.4 层定位存在 winner's curse

同时搜索 25/41 层与 7 个 $C$，再报告最佳验证点，会放大偶然峰值。异常检测的 CV–test 落差和 $C=1000$ 触顶都提示选择不稳定。论文适合讨论“中层 vs 深层”的粗粒度趋势，不适合强调单一精确层号。

### 8.5 PCA 不能直接证明低维流形

连续视频的相邻帧天然相似，即便是像素、随机网络或普通图像特征也可能形成平滑曲线。缺少 explained variance、内在维数估计、随机/图像模型基线和跨 clip 共享坐标系，使“流形”主要停留在可视化层面。

### 8.6 Steering 评价不独立

Spline 用实际中间特征拟合，再按到这些特征的最近距离评估，天然优于只看端点的 linear path。结果证明“经验轨迹有曲率”比证明“方法可以 steering”更有力。最近邻帧也来自同一个原 clip，不是新生成结果。

### 8.7 复现性偏弱

截至 2026-11-09 没有官方代码；关键配置未披露；表格还有层号越界、训练 AUC 异常、附录 Figure 8/9/10 顺序描述与实际图注不一致等小问题。作为 workshop 探索论文可以接受，但复现实验需要大量猜测。

---

## 9. 复现与实践建议

### 9.1 最小复现路径

1. 固定一个公开 checkpoint，先复现 V-JEPA 2 L + CameraBench。
2. 缓存每个 block MLP 输出与最终 LayerNorm；不要每次探针都重复视频前向。
3. 明确记录输入分辨率、帧数、采样率、tubelet size、checkpoint hash、层编号映射。
4. 对空间与时间 token 做全局均值，$L_2$ 归一化后训练线性 SVM。
5. 使用 nested CV：内层选 layer 与 $C$，外层估计泛化，并锁定 test。
6. 报告十次 run 的 paired CI、最佳层选择频率和标签置换基线。
7. 再增加 token-level / temporal-relation probe 检查 IntPhys 2，而不是直接换成容量很大的 MLP。

### 9.2 激活缓存规模

V-JEPA 2 L 在 $256^2$ 输入下，每个时间位置有 256 个空间 patch，32 帧对应 16 个时间 patch，总 token 数是 4096，维度 1024。单层单样本若用 FP16，原始激活约为：

$$4096\times1024\times2\ \mathrm{bytes}\approx8\ \mathrm{MiB}$$

若缓存 25 个探针点，就是约 200 MiB/clip；对上千视频直接保存全 token 会迅速达到数百 GiB。工程上应在前向时立刻完成所需池化，只保存视频级向量；几何实验再对少量 RealEstate10K clip 单独保留时间序列 $[T,1024]$。

### 9.3 更严谨的后续实验

- **时间依赖对照：**单帧、乱序、反向、重复帧与光流基线，检验 CameraBench 分数是否真来自时序。
- **物理信息分层探针：**mean-pooled linear → token-level linear → 小型关系型 probe，并控制参数量、做样本量曲线。
- **内在维数检验：**报告 PCA explained variance、participation ratio、local intrinsic dimension 与邻域保持率。
- **公平 steering：**只用稀疏控制点拟合，未见中间帧作测试；加入 piecewise-linear 与 kNN geodesic。
- **行为层验证：**测量 steering 后相机参数回归、时间秩相关、下游动作识别或真正视频解码，而不只看同 clip 最近邻。

### 9.4 复现成本判断

SVM 本身很轻；主要成本是三个大型视频模型的全层前向和激活缓存。若只保存均值池化向量，单卡离线提特征可行；若完整复现每层 token 几何、三模型、三数据集、十次运行与多个帧数，成本为中高。由于没有官方代码，当前最大的风险不是纯算力，而是数据切分、层索引与采样细节不一致。

---

## 10. 个人启发与后续问题

### 10.1 中层可能是视频几何的最佳接口

相机运动在中层达到峰值、异常在深层更强，说明视频模型的不同层可能对应不同控制接口：中层更适合相机、光流、位姿等几何控制，深层更适合事件与语义异常。对世界模型或视频生成而言，直接在最后层寻找可控方向未必最佳。

### 10.2 Negative result 的价值在于暴露接口不匹配

IntPhys 2 失败不应只被解读为模型“没有物理”。它也可能说明当前读出接口——全局均值 + 线性分类——不适合对象关系。后续应把“表示里有没有信息”拆成：局部 token 是否有对象、跨时是否可对应、关系是否可组合、最终是否能做违规判断。

### 10.3 从直线 steering 转向局部几何是正确方向

即使本文的 spline 评价偏构造性，它仍揭示一个重要工程事实：端点差分向量可能穿过数据稀疏区。未来更值得研究的是共享局部切空间、图测地线、跨 clip 对齐以及带行为监督的几何控制，而不是把任意两个激活之差都当作全局语义方向。

### 10.4 仍待回答的问题

- 振荡轨迹是 tubelet/位置编码造成，还是相机平移的真实对称性？
- 相同相机运动跨场景是否共享可对齐的几何方向？
- 不同模型的中层轨迹是否可通过 Procrustes/CCA 对齐？
- 当模型换成生成式视频 DiT、世界模型或 VLA 视觉编码器时，层级趋势是否仍成立？
- 物理违规信号是不存在，还是存在于对象 token 的非线性关系中？

---

## 11. 材料来源

- 论文：Musa et al., *What, Where, and How: Probing Spatiotemporal Representations in Video Foundation Models*, arXiv:2609.01551v1, 2026。
- 官方 PDF：[本地副本](assets/What-Where-How%20-%20视频基础模型时空表征探针/sources/paper.pdf) · [arXiv](https://arxiv.org/pdf/2609.01551)。
- 官方交互可视化：[V-JEPA 2 · PCA Explorer](https://vid-rep-pca.netlify.app/)。
- 附件与图表溯源：[[assets/What-Where-How - 视频基础模型时空表征探针/sources|sources.md]]。
- 开源核查：截至 2026-11-09 未发现作者官方仓库，因此没有“代码级实现”章节；所有方法、公式与超参均以论文正文和附录为准。
