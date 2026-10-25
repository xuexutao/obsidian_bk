---
type: paper-note
status: complete
domain: Foundations
paper: "Contrastive Flow Matching"
year: 2025
arxiv: "2506.05350"
doi: "10.48550/arXiv.2506.05350"
source: "https://arxiv.org/abs/2506.05350"
project: null
code: "https://github.com/gstoica27/DeltaFM"
model: "https://huggingface.co/gstoica3/DeltaFM"
venue: "ICCV 2025"
tags:
  - Foundations
  - Flow-Matching
  - Diffusion
  - Contrastive-Learning
  - Classifier-Free-Guidance
  - Image-Generation
created: 2026-10-25
updated: 2026-10-25
---

# Contrastive Flow Matching：用对比目标增强条件流的分离

## 基本信息

- **论文标题**：Contrastive Flow Matching（文中简称 ΔFM）
- **作者**：George Stoica†、Vivek Ramanujan◇、Xiang Fan◇、Ali Farhadi、Ranjay Krishna、Judy Hoffman（†通讯，◇共同贡献）
- **机构**：Georgia Tech、University of Washington
- **发表**：ICCV 2025；arXiv [2506.05350](https://arxiv.org/abs/2506.05350)，v1，2025-06-05（截至 2026-10-25 仍只有 v1）
- **代码**：https://github.com/gstoica27/DeltaFM （基于 REPA 代码库，作者说明“几乎所有文件与 REPA 相同”）
- **权重**：https://huggingface.co/gstoica3/DeltaFM （REPA SiT-XL/2，256 与 512 分辨率）
- **实验规模**：ImageNet-1k 256/512 上的 SiT-B/2、SiT-XL/2（含 / 不含 REPA），CC3M 256 上的 REPA-MMDiT；统一训练 400K iteration、batch 256、λ=0.05
- **核心结果**：不用 CFG 时 FID 最多降低 8.9（SiT-B/2 256：42.28→33.39）；REPA SiT-XL/2 256 上 11.14→7.29；配合 CFG 时 2.09→1.97；作者称训练快 9 倍、采样步数少 5 倍

## 一句话结论

ΔFM 在标准条件 flow matching 损失上加了一项“推离随机负样本速度目标”的项：$\mathcal{L}=\|v_\theta-v^+\|^2-\lambda\|v_\theta-v^-\|^2$，其中负样本取同 batch 里另一个样本的 $(x,\epsilon)$ 所构成的速度目标。它零额外前向、几行代码即可接入，在 ImageNet 和 CC3M 上一致降低 FID、大幅降低 sFID、提高 precision。但按论文附录自己给出的闭式解，**ΔFM 的总体最优解只是把 FM 速度场放大 $1/(1-\lambda)$ 倍、再减去一个与条件和 $x_t$ 都无关的常数向量**。换句话说，它在数学上等价于“以数据均值轨迹为无条件分支、引导强度约 1.05 的 CFG”，被直接烘焙进了训练目标里。它的收益更像低温度的引导效应（precision↑、recall 不变），而不是标题所说的让条件之间的流“彼此对比、更加多样”。

> **我的总体评价：★★★★☆。** 作为即插即用的训练技巧，它的工程价值很高：实现成本几乎为零，增益在多个尺度、分辨率和框架上都一致。扣分主要在解释层面：论文的“流唯一性 / 条件分离 / 多样性”叙事与它自己的闭式解和 recall 数据并不吻合，toy 实验中“减少歧义”本质上是偏离了真实条件分布；batch size 表存在标签不一致；也缺少与其他“免 CFG 训练”方法的对比。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

Flow matching 通过回归噪声分布与数据分布之间的理想概率流速度场来训练连续归一化流，轨迹接近直线，是当前图像、视频和 3D 生成的主流训练范式之一（基础见 [[Flow Matching - 条件概率路径与连续生成]]）。作者关注的是**条件生成**：在类别条件或文本条件下，vanilla flow matching 生成的样本往往像该条件下所有可能图像的“平均”，缺乏一个清晰模态应有的特异性。

目前缓解这一问题的主要手段有两类：

- **表征对齐（REPA）**：把 DINOv2 等预训练视觉编码器的特征蒸馏进扩散模型中间层，改善收敛与画质，代价是需要外部编码器和额外的辅助损失；
- **Classifier-Free Guidance（CFG）**：训练时随机丢弃条件，推理时用条件与无条件两次前向的外推把样本推向条件目标。效果显著，但采样成本翻倍，训练上也需要同时学一个隐式的无条件生成器。

ΔFM 想做的是：**只改训练目标、不增加推理成本，就获得部分 CFG 式的条件增强效果，并且能与 REPA、CFG 叠加。**

### 1.2 现有方法的瓶颈：作者的“流唯一性”论证

作者的论证从 flow uniqueness 出发：一个定义良好的 ODE 的轨迹在 $t\in[0,T)$ 内不会相交，因此无条件 flow matching 的每个 $x_t$ 都对应唯一的流动方向，模型可以在 $x_t$ 中逐步细化与目标样本相关的判别性特征。到了条件 flow matching，作者认为：

1. 模型只保证**同一条件内**的流是唯一的，不同条件之间的流可以重叠，这会阻碍 $x_t$ 存储类别特有的特征，降低生成质量；
2. 条件 FM 目标在训练时不知道其他条件的分布范围，当条件分布彼此重叠时，流可能生成落在多个条件“混合区域”的歧义样本。

![](assets/Contrastive%20Flow%20Matching%20-%20对比目标增强条件流分离/figures/02-flow-overlap-motivation.png)

> 图 2：作者的动机示意。左侧是标准 flow matching：流是直的，但相近类别（如风景 / 桥梁与鹦鹉）的轨迹在中途重叠，终点样本类别特征不鲜明；右侧加入 ΔFM 后，两类轨迹更早分开，终点图像更能代表各自类别（桥梁的拱结构、鹦鹉的形态更清晰）。
> 来源：论文 Figure 2，第 2 页，https://arxiv.org/abs/2506.05350

需要先指出（个人判断）：这一论证在理论上并不严密。不同条件 $y$ 对应的是**不同的 ODE**，它们的轨迹在同一空间中相交并不违反任何唯一性；真正的条件分布若本身有重叠，完美的条件模型就**应该**在重叠区域生成样本。所以“流重叠导致平均化”更像一个直觉性的动机，而不是被证明的缺陷。后面 3.4 节会看到，ΔFM 的实际作用机制也并不是“让不同条件的流彼此分离”。

### 1.3 本文核心贡献

1. 提出 Contrastive Flow Matching（ΔFM）：在条件 FM 目标上加一个对比正则项，最大化预测速度与“任意其他样本对的流”之间的差异；
2. 给出闭式最优解，并据此推导出 ΔFM 与 CFG 共存时的修正采样公式；
3. 在 ImageNet-1k（SiT-B/2、SiT-XL/2，256/512，含 / 不含 REPA）与 CC3M（REPA-MMDiT）上验证：FID 一致下降，训练迭代最多省 9 倍，去噪步数最多省 5 倍。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

任务与标准条件 flow matching 完全一致：训练数据为 $(x,y)\sim p(x,y)$，$x$ 是 SD-VAE 编码后的图像 latent（ImageNet 256 上为 $32\times32\times4$），$y$ 是类别标签或文本；模型 $v_\theta(x_t,t,y)$ 预测速度；推理时从高斯噪声出发积分 ODE / SDE 得到样本。ΔFM **不改变网络结构、不改变推理流程、不需要额外数据或额外前向**，只修改每个 batch 的损失计算。

### 2.2 关键符号和目标函数

**随机插值。** 作者沿用 SiT 的 stochastic interpolant 写法：

$$\hat{x}_t=\alpha_t\hat{x}+\sigma_t\epsilon$$

其中 $\alpha_t$ 递减、$\sigma_t$ 递增。论文正文写作 $\alpha_T=\sigma_0=1$、$\alpha_0=\sigma_T=0$，这与“$\alpha_t$ 递减”自相矛盾，属于笔误；官方代码中线性路径为 $\alpha_t=1-t$、$\sigma_t=t$，即 $t=0$ 是数据、$t=1$ 是噪声，$\dot{\alpha}_t=-1$、$\dot{\sigma}_t=1$。

**Flow matching。** 理想速度场是给定 $x_t$ 时目标速度的条件期望：

$$v(x_t,t)=\dot{\alpha}_t\mathbb{E}[\hat{x}|x_t=x]+\dot{\sigma}_t\mathbb{E}[\epsilon|x_t=x]$$

用网络回归逐样本目标 $\dot{\alpha}_t\hat{x}+\dot{\sigma}_t\epsilon$ 即可在期望意义下逼近它。条件版本只是在网络输入中加上 $y$，并令 $\hat{x}\sim p(x|y)$：

$$\mathcal{L}^{(FM)}_{cond}(\theta)=\mathbb{E}\left[\|v_\theta(x_t,t,y)-(\dot{\alpha}_t\hat{x}+\dot{\sigma}_t\epsilon)\|^2\right]$$

---

## 3. 核心方法

### 3.1 总体框架：一正一负的速度回归

ΔFM 为每个训练样本 $(\hat{x},y,\epsilon)$ 额外取一个“负样本对” $(\tilde{x},\tilde{\epsilon})$，其中 $\tilde{x}\sim p(x|\tilde{y})$、$\tilde{\epsilon}\sim\mathcal{N}(0,I)$，要求 $\tilde{x}\neq\hat{x}$、$\tilde{\epsilon}\neq\epsilon$，而 $\tilde{y}$ **可以等于也可以不等于** $y$。作者强调不假设存在某个 $t$ 使 $x_t=\alpha_t\tilde{x}+\sigma_t\tilde{\epsilon}$，即负样本对代表一条与当前 $x_t$ 完全独立的流。

对比正则希望当前预测速度 $v_\theta(x_t,t,y)$ 尽量远离这条独立流的速度目标：

$$\max\ \mathbb{E}\left[\|v_\theta(x_t,t,y)-(\dot{\alpha}_t\tilde{x}+\dot{\sigma}_t\tilde{\epsilon})\|^2\right]$$

由于 $\tilde{x}$ 来自边缘分布 $p(x)$ 而非 $p(x|y)$，作者称这一项迫使模型产生“无条件意义下唯一”的流。合在一起：

$$\mathcal{L}^{(\Delta FM)}(\theta)=\mathbb{E}\left[\|v_\theta(x_t,t,y)-(\dot{\alpha}_t\hat{x}+\dot{\sigma}_t\epsilon)\|^2-\lambda\|v_\theta(x_t,t,y)-(\dot{\alpha}_t\tilde{x}+\dot{\sigma}_t\tilde{\epsilon})\|^2\right]$$

$\lambda\in[0,1)$ 控制对比强度，$\lambda=0$ 时退化为 FM，所有实验固定 $\lambda=0.05$。$\lambda<1$ 这一约束很重要：展开后 $v_\theta$ 的二次项系数为 $1-\lambda>0$，目标仍是关于 $v_\theta$ 的凸二次函数，有有限的最小值；若 $\lambda\geq1$，模型只要把速度无限放大就能让损失趋于负无穷。

### 3.2 实现：batch 内取负样本，零额外前向

论文 Algorithm 1 的批处理逻辑是：对 batch 中每个样本 $i$，采样 $t\sim\mathcal{U}(0,1)$，构造 $x_t$；从同一 batch 中随机取另一个样本 $(\tilde{x},\tilde{y},\tilde{\epsilon})$；计算 $\hat{v}=v_\theta(x_t,t,y_i)$、正目标 $v=\dot{\alpha}_tx_i+\dot{\sigma}_t\epsilon_i$、负目标 $\tilde{v}=\dot{\alpha}_t\tilde{x}+\dot{\sigma}_t\tilde{\epsilon}$；累加 $\|\hat{v}-v\|^2-\lambda\|\hat{v}-\tilde{v}\|^2$。

核对官方 `triplet_loss.py`：

- 默认路径 `compute_triplet_loss_efficiently` 把 batch 内所有样本的**速度目标** `model_target` 当作负样本池，为每个样本随机挑一个**非自身**的下标作为负目标，**不检查类别是否相同**；
- 代码中还有一个 `class_conditioned_sampling` 函数，可以只从不同类别中取负样本，但默认训练路径不调用它；
- `dont_contrast_on_unconditional` 选项可以对 CFG 训练中被丢弃条件的 null class 样本关闭对比项，默认关闭；
- 超参在代码里被命名为 `temperature`，其实就是 $\lambda$（`--contrastive-weight=0.05`）。

等价的极简写法（个人整理，便于移植到其他 flow matching 代码库）：

```python
import torch


def delta_fm_loss(v_pred, v_target, lam=0.05):
    """Contrastive Flow Matching loss with in-batch negatives."""
    batch_size = v_pred.shape[0]
    offset = torch.randint(1, batch_size, (batch_size,), device=v_pred.device)
    negative_index = (torch.arange(batch_size, device=v_pred.device) + offset) % batch_size

    positive_error = (v_pred - v_target).pow(2).flatten(1).mean(dim=1)
    negative_error = (v_pred - v_target[negative_index]).pow(2).flatten(1).mean(dim=1)
    return (positive_error - lam * negative_error).mean()
```

因为负目标直接复用 batch 中已有样本的 $(x,\epsilon)$，所以没有任何额外前向，计算开销只是一次额外的 MSE。

### 3.3 Toy 实验：二维两类重叠分布

![](assets/Contrastive%20Flow%20Matching%20-%20对比目标增强条件流分离/figures/03-toy-2d-flows.png)

> 图 3：二维 toy 实验。三层 MLP 把紫色二维高斯噪声搬运到蓝、橙两个约有 50% 重叠的类别分布。上：FM 学到的两类流在终点高度重叠，大量样本落在两类之间的模糊区域；下：ΔFM 的流明显分开，样本更集中在各自类别的“纯”区域。
> 来源：论文 Figure 3，第 5 页，https://arxiv.org/abs/2506.05350

作者把这张图作为“ΔFM 内在地分离不同类别的流”的直观证据。但需要注意（个人判断）：两类分布本来就有约 50% 重叠，**真实的条件分布 $p(x|y)$ 本身就包含重叠区域**。上图 FM 在重叠区域生成样本恰恰是在正确地拟合条件分布；下图 ΔFM “减少歧义”意味着它**系统性地偏离了真实条件分布**，把每一类的样本推向远离另一类的一侧。这与 CFG 在 toy 例子中的行为如出一辙：牺牲分布保真度，换取样本的“类别典型性”。作者在结论里也承认：“偏离完美的分布建模，可能反而带来更好的图像生成”。

### 3.4 闭式解：ΔFM 实际上做了什么

这是理解 ΔFM 最关键的一节。论文附录 B.1 推导了 ΔFM 目标的最优解。把逐点目标展开（记 $v=v_\theta(x_t,t,y)$，$v^+=\dot{\alpha}_t\hat{x}+\dot{\sigma}_t\epsilon$，$v^-=\dot{\alpha}_t\tilde{x}+\dot{\sigma}_t\tilde{\epsilon}$）：

$$\mathbb{E}\left[(1-\lambda)v^\top v-2v^\top(v^+-\lambda v^-)\right]+\text{const}$$

对 $v$ 求导置零，得到给定 $(x_t,t,y)$ 时的最优速度：

$$v^*_{\Delta FM}=\frac{\mathbb{E}[v^+|x_t,y]-\lambda\mathbb{E}[v^-]}{1-\lambda}=\frac{v_{FM}(x_t,t,y)-\lambda\hat{T}}{1-\lambda}$$

其中 $v_{FM}$ 是普通条件 FM 的最优解，$\hat{T}=\mathbb{E}[\dot{\alpha}_t\tilde{x}+\dot{\sigma}_t\tilde{\epsilon}]$ 被作者称为“训练集所有样本轨迹的均值”。（论文 Eq.12 写成了 $\min_\theta\mathcal{L}^{(\Delta FM)}=(\min_\theta\mathcal{L}^{(FM)}-\lambda\hat{T})/(1-\lambda)$，把“最优速度”与“最小损失值”混为一谈，正确的含义应是上式中的最优速度。）

**进一步展开 $\hat{T}$（以下为个人推导）。** 由于负样本对与当前 $x_t$、$y$ 独立，$\mathbb{E}[v^-]$ 不依赖 $x_t$ 和 $y$。在官方使用的线性路径下 $\dot{\alpha}_t=-1$、$\dot{\sigma}_t=1$，且 $\mathbb{E}[\tilde{\epsilon}]=0$，于是：

$$\hat{T}=-\mu,\quad\mu=\mathbb{E}_{p(x)}[x]$$

即 $\hat{T}$ 就是负的数据 latent 均值，是一个**与时间、条件、当前状态都无关的常数向量**（这也解释了为什么官方仓库只需要存一个 `imnet256_interference_vector.pt`）。代回得到：

$$v^*_{\Delta FM}=v_{FM}+\frac{\lambda}{1-\lambda}\left(v_{FM}-\hat{T}\right)$$

把它和 CFG 对比：

$$v_{CFG}=v_c+(w-1)(v_c-v_u)$$

两者形式完全相同：**ΔFM 的总体最优解等价于一次 CFG，其中“无条件分支” $v_u$ 被替换成常数 $\hat{T}$（数据均值轨迹），引导强度 $w=1/(1-\lambda)$**。当 $\lambda=0.05$ 时，$w\approx1.053$。

这一视角可以解释论文中的大部分现象：

1. **为什么不用 CFG 也能显著降 FID、提高 precision 和 IS**：它本身就是一个弱引导；
2. **为什么 recall 不升反略降**：引导类方法提高保真度、牺牲覆盖度，这与“更多样”的叙事相反（详见 5.1 节）；
3. **为什么与 CFG 叠加时收益缩小**：两者作用方向高度重叠，Table 4 中 FID 只从 2.09 降到 1.97；
4. **为什么 toy 实验中“歧义区域”消失**：沿着“远离均值”的方向外推，会把样本推离两类的交界处；
5. **为什么需要修正 CFG 公式**：作者在 5.4 节把训练好的 ΔFM 模型反解回 FM 速度再做 CFG，等于先抵消掉烘焙进去的那一份引导。

同时要承认这一分析的边界：它是**无限容量、总体期望意义下**的最优解。实际训练中，负样本是逐样本随机抽取的，会给梯度引入额外方差；有限容量的网络也未必精确收敛到这个仿射变换。λ 的消融（5.3 节）显示，λ=0.15（对应 $w\approx1.18$）会让 FID 从 11.14 恶化到 19.21、IS 从 115.83 降到 82.20——如此温和的引导强度在 CFG 中通常是有益的，所以大 λ 的崩溃更可能来自训练动力学（排斥项的高方差梯度、模型容量被用于拟合偏移），而不是总体最优解本身。换句话说，“等价于弱 CFG”能解释方向，但不能解释全部数值。

另外，这个解释揭示了一个实践上的推论：**ΔFM 的效果依赖数据 latent 均值 $\mu$**。若 latent 经过逐通道零均值归一化，则 $\hat{T}\approx0$，ΔFM 退化为把速度整体放大 $1/(1-\lambda)$ 倍；SD-VAE latent 只乘了 0.18215 的缩放、没有去均值，所以 $\mu$ 不为零。

### 3.5 与 CFG 的耦合修正

作者在 5.4 节指出，ΔFM 把条件流推离“任意其他流”，与 $x_t$ 无关；CFG 则基于 $x_t$ 把生成推离无条件流估计。两者不总是对齐，直接叠加可能冲突。利用闭式解 $v_{FM}=(1-\lambda)v_{\Delta FM}+\lambda\hat{T}$，代入 CFG 公式得到：

$$\widehat{CFG}=(1-\lambda)\left[w\,v(x_t|y)+(1-w)\,v(x_t|\varnothing)\right]+\lambda\hat{T}$$

其中 $v(\cdot)$ 是 ΔFM 训练出的模型。该修正只在引导区间 $[\sigma_{low},\sigma_{high}]$ 内使用，区间外直接用未修改的 ΔFM 模型。官方生成脚本对应的参数是 `--interference-weight 0.05`、`--velocity-weight 0.95`。（正文中该式末项被误写为 $\lambda\tau$，附录 B.2 中为 $\lambda\hat{T}$。）作者还给出另一种未验证的耦合方式 $\widetilde{CFG}=(w+\lambda)v(x_t|y)-(1-w)v(x_t|\varnothing)-\lambda\hat{T}$，留作未来工作。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

- **ImageNet-1k**：256×256 与 512×512，按 ADM 的预处理流程，用 Stable Diffusion VAE 编码为 latent（256 分辨率为 $32\times32\times4$）；
- **CC3M**：256×256，按 U-ViT（Bao et al.）的处理流程，用于文生图实验。

### 4.2 Baseline 与评价指标

Baseline 就是同一模型、同一训练配置下的标准 flow matching：SiT-B/2、SiT-XL/2、REPA SiT-B/2、REPA SiT-XL/2、REPA-MMDiT。作者强调“严格遵循 REPA 的设置，没有为 ΔFM 调整任何训练条件”。**论文没有与其他改进条件生成的训练方法做横向对比**（例如免 CFG 的训练式引导方法、autoguidance 等）。

指标：类别条件实验用 50K 样本计算 FID、IS、sFID、Precision、Recall；文生图用整个验证集计算 FID。采样器统一为 SDE Euler-Maruyama（扩散系数 $w_t=\sigma_t$），默认 NFE=50。

### 4.3 实现细节

| 项目 | 设置 |
|---|---|
| 训练长度 | 400K iteration（Figure 5 训练曲线延伸到 2M） |
| Batch size | 256（Table 6 额外测试 512、1024） |
| λ | 0.05（所有实验固定） |
| REPA | DINOv2 ViT-B 特征对齐到 SiT-B/2 第 4 层、SiT-XL/2 第 8 层，其余超参沿用 REPA |
| 采样 | SDE Euler-Maruyama，NFE=50，默认不使用 CFG |
| CFG 实验 | 网格搜索 $w\in\{1.25,1.75,1.8,1.85,2.25\}$，$\sigma_{low}=0$，$\sigma_{high}\in\{0.50,0.65,0.75,1.0\}$ |
| 硬件与训练时长 | 论文未披露 |

---

## 5. 实验结果

### 5.1 主要定量结果

![](assets/Contrastive%20Flow%20Matching%20-%20对比目标增强条件流分离/figures/06-main-tables.png)

> 图（Table 1–3）：左为 SiT 在 ImageNet 256（a）与 512（b）上的结果，右上为 REPA SiT 在 256 / 512 上的结果，右下为 CC3M 文生图结果。全部为 400K iteration、NFE=50、无 CFG。
> 来源：论文 Table 1、Table 2、Table 3，第 6 页，https://arxiv.org/abs/2506.05350

把各设置的变化整理成一张表（FID / sFID / Precision / Recall，均为 FM → ΔFM）：

| 设置 | FID | sFID | Prec. | Rec. |
|---|---|---|---|---|
| SiT-B/2, 256 | 42.28 → 33.39 | 11.35 → 5.67 | 0.50 → 0.53 | 0.62 → 0.63 |
| SiT-XL/2, 256 | 20.01 → 16.32 | 8.45 → 5.08 | 0.63 → 0.66 | 0.63 → 0.63 |
| SiT-B/2, 512 | 50.26 → 41.59 | 14.88 → 6.13 | 0.57 → 0.62 | 0.61 → 0.63 |
| SiT-XL/2, 512 | 22.98 → 19.67 | 10.71 → 4.98 | 0.73 → 0.76 | 0.60 → 0.60 |
| REPA SiT-B/2, 256 | 27.33 → 20.52 | 11.70 → 5.47 | 0.57 → 0.61 | 0.64 → 0.63 |
| REPA SiT-XL/2, 256 | 11.14 → 7.29 | 8.25 → 4.93 | 0.67 → 0.71 | 0.65 → 0.64 |
| REPA SiT-B/2, 512 | 31.90 → 24.48 | 13.78 → 5.89 | 0.67 → 0.71 | 0.62 → 0.61 |
| REPA SiT-XL/2, 512 | 11.32 → 7.64 | 10.21 → 4.72 | 0.76 → 0.79 | 0.63 → 0.62 |
| REPA-MMDiT, CC3M 256 | 24 → 19 | — | — | — |

这张表说明了几件事：

1. **增益非常一致**：8 个类别条件设置全部改善，FID 降幅 3.3–8.9，且小模型（B/2）收益更大，作者称这体现了 ΔFM 在小模型上的优势。
2. **sFID 的变化最惊人**：无论基线 sFID 是 8.25 还是 14.88，ΔFM 之后都落在 4.7–6.1 的窄区间里，几乎减半。sFID 基于 Inception 中间层的空间特征，对全局结构与空间布局敏感，这与论文定性图中“全局结构更连贯”的观察吻合。但如此整齐的收敛区间也值得独立复现确认（个人判断）。
3. **Precision 稳定提升，Recall 基本不变甚至略降**：在 REPA 的四个设置中 recall 全部下降 0.01。Precision 衡量样本落在真实流形内的比例，Recall 衡量真实分布被覆盖的程度。所以 ΔFM 提升的是**保真度而非多样性**。论文引言反复说 ΔFM “鼓励更多样（diverse）的条件生成”，这一说法没有数据支撑，表现反而与 3.4 节“弱 CFG”的解释完全一致。
4. **文生图只给了一个 FID 数字**：24→19，没有 CLIPScore 或任何图文对齐指标。对于一个声称增强“条件分离”的方法，在文本条件下不测对齐度是明显的缺口。

### 5.2 定性结果

![](assets/Contrastive%20Flow%20Matching%20-%20对比目标增强条件流分离/figures/01-teaser.png)

> 图 1：ImageNet 512 上相同类别、相同初始噪声下的成对比较（每对左为 FM、右为 ΔFM）。ΔFM 的样本全局结构更连贯，例如北极狐、哈士奇的面部结构更完整，鹦鹉站姿更合理。
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2506.05350

![](assets/Contrastive%20Flow%20Matching%20-%20对比目标增强条件流分离/figures/04-denoising-trajectories.png)

> 图 4：REPA SiT-XL/2（ImageNet 256，400K 步）在 30 步 SDE 采样中每隔 5 步的预测终图，每组上行为 FM、下行为 ΔFM，同一初始噪声。ΔFM 在第 15–20 步就已出现清晰的类别标志性结构，例如桥面的斜拱、动物眼睛、火车车窗；FM 同期仍较模糊，部分样本最终也没有形成完整结构（如石拱门、狮子）。
> 来源：论文 Figure 4，第 7 页，https://arxiv.org/abs/2506.05350

作者据此认为 ΔFM 让类别分化发生得更早。从图中还能看到（个人观察）：ΔFM 的中间预测和终图整体对比度、饱和度略高，颜色更“实”。这也符合“沿远离数据均值方向外推”的解释：均值 latent 对应偏灰、偏平均的图像，远离它自然会增强对比度。

### 5.3 消融实验

![](assets/Contrastive%20Flow%20Matching%20-%20对比目标增强条件流分离/figures/07-cfg-lambda-batch-tables.png)

> 图（Table 4–6）：左为配合 CFG 的结果；右上为 λ 消融（REPA SiT-XL/2，ImageNet 256）；右下为 batch size 消融。
> 来源：论文 Table 4、Table 5、Table 6，第 8 页，https://arxiv.org/abs/2506.05350

**λ 的影响（Table 5）。** λ = 0 / 0.001 / 0.01 / 0.05 / 0.1 / 0.15 对应的 FID 依次为 11.14 / 10.93 / 9.93 / 7.29 / 9.86 / 19.21，IS 依次为 115.83 / 115.70 / 119.41 / 129.89 / 116.27 / 82.20。最优点在 0.05，两侧下降都很快：λ 过小基本等同 FM，λ 过大则产生无法刻画类别结构的退化分布。作者称 λ=0.05 在所有模型和数据设置下都稳定，但 λ 消融只在 REPA SiT-XL/2 256 这一个设置上做过。“有效窗口窄”意味着换到新数据（如视频、3D latent）时很可能需要重新搜索 λ。

**Batch size 的影响（Table 6）。** 作者认为对比类损失依赖 batch 内负样本的代表性，并得出“ΔFM 随 batch size 扩展”“甚至优于两倍 batch 的 FM”。细看数据（FID，FM → ΔFM，括号内为差值）：

| 模型 | B=256 | B=512 | B=1024 |
|---|---|---|---|
| “REPA SiT-B/2” | 42.28 → 33.39（−8.89） | 24.45 → 17.06（−7.39） | 22.00 → 15.23（−6.77） |
| REPA SiT-XL/2 | 11.14 → 7.29（−3.85） | 10.15 → 6.36（−3.79） | — |

ΔFM 带来的**绝对增益并没有随 batch 增大而增大**，B/2 上反而从 8.89 缩小到 6.77；“随 batch 扩展”主要来自基线本身随 batch 变好。这其实与 3.4 节的分析一致：总体最优解只依赖负目标的**期望**（数据均值），batch 越大只是让负样本梯度方差越小，而不是让“对比”越充分。

此外这张表有一个**标签不一致**：标为 “REPA SiT-B/2, B=256” 的一行（42.28 / 38.04 / 11.35 与 33.39 / 43.44 / 5.67）与 Table 1 中**不带 REPA** 的 SiT-B/2 数字完全相同，而 Table 2 的 REPA SiT-B/2 是 27.33 / 20.52。因此 B/2 那一组的 batch 对比可能混入了不同训练框架，结论需要打折扣。

**配合 CFG（Table 4）。** REPA SiT-XL/2 256、NFE=50、各自网格搜索最优 CFG 参数：FM 为 $w=1.75$、$\sigma_{high}=0.75$，FID 2.09 / sFID 5.55 / IS 280.33；ΔFM（使用修正 CFG）为 $w=1.85$、$\sigma_{high}=0.65$，FID 1.97 / sFID 4.49 / IS 281.95。FID 相对降低 5.7%，绝对值只有 0.12。无 CFG 时 FID 差距 3.85，有 CFG 时只剩 0.12，说明两者的作用大部分重叠；论文没有报告不加修正、直接对 ΔFM 做标准 CFG 的结果，所以修正公式本身的贡献无法单独评估。

### 5.4 训练与推理效率

![](assets/Contrastive%20Flow%20Matching%20-%20对比目标增强条件流分离/figures/05-training-inference-efficiency.png)

> 图 5：ImageNet 256 上的 FID-50K。左：随训练迭代变化（对数轴，50K–2M），ΔFM 在约 200K 步时达到 FM 在约 2M 步时的 FID，作者据此称训练快 9 倍；右：随去噪步数变化，ΔFM 用 50 步即优于 FM 用 250 步，作者称推理快 5 倍。
> 来源：论文 Figure 5，第 9 页，https://arxiv.org/abs/2506.05350

这两个“倍数”需要放在评测协议下理解（个人判断）：

- **“9 倍训练加速”是在 NFE=50 下测得的**。左图中 FM 曲线从约 400K 到 2M 几乎持平在 FID≈10.5，而 REPA 原论文报告 REPA SiT-XL/2 在更长训练和 250 步采样下能继续下降。FM 在 50 步下的平台期很可能主要来自采样离散化误差，而不是模型没有训练好。ΔFM 的速度放大恰好部分补偿了少步采样的“欠冲”，所以 9 倍是“同一个 50 步评测协议下的倍数”，不能直接理解为“模型学得快 9 倍”。
- **“5 倍推理加速”** 是 ΔFM@50 步（FID≈7.3）对 FM@250 步（FID≈7.8）。右图中 100 步以后两条曲线逐渐靠近，ΔFM 的优势主要集中在少步区间。这同样符合“弱引导 / 速度放大更适合少步采样”的解释。

**失败案例与局限**：论文没有展示失败样本，也没有分析 λ 过大时退化样本的形态。

---

## 6. 与相关工作的关系

- **与 CFG 及其变体**：ΔFM 自我定位为“CFG 的替代方案”。按 3.4 节的推导，它更准确的定位是**训练期烘焙进模型的一次弱引导**：以数据均值轨迹替代无条件模型，引导强度由 λ 固定。它和 autoguidance（用“坏版本的自己”做无条件分支）、引导区间（limited interval guidance）等工作处在同一谱系，只是把引导从推理期移到了训练期，因此推理零成本，但也失去了推理时可调的灵活性。据我所知，同期也有其他直接训练“免 CFG 条件模型”的方法（如 Model-guidance 类工作），论文没有与它们对比。
- **与 REPA**：两者正交。REPA 改善中间表征，ΔFM 改变输出速度场的偏置，实验也显示二者可以叠加。
- **与对比学习**：名字和动机来自 FaceNet triplet loss、SimCLR / MoCo 一类 batch 内负样本方法（见 [[SimCLR - 视觉对比学习的极简框架]]、[[MoCo - 动量编码器与动态对比字典]]），但机制差别很大：经典对比学习在归一化的嵌入空间里用 softmax / margin 做“相对比较”，而 ΔFM 是对**原始速度目标**做无归一化的平方距离排斥，且总体最优只依赖负目标的均值，所以它并不是在学一个判别性的表征空间。
- **与“拉—推”统一视角**：[[Contrastive Generation I - 拉推梯度统一少步生成器训练]] 把 MeanFlow、DMD 和 drifting models 的梯度统一分解为“拉向正样本 + 推离负样本”。ΔFM 正是一个最朴素的拉—推实例：正样本是自身的速度目标，负样本是 batch 里随机另一条轨迹的速度目标；只是它的负样本来自数据而不是生成器自身，所以推力退化成了指向数据均值反方向的常数。
- **与基础 DiT / Flow Matching 路线**：方法完全不改结构，可以直接用于 [[DiT - 可扩展扩散Transformer]]、SiT、MMDiT 这类基于 flow matching 训练的模型；背景知识参见 [[Flow Matching - 条件概率路径与连续生成]] 与 [[VAE、Attention、DiT 与 Flow Matching 基础模型路线 - Survey]]。

---

## 7. 局限与批判性评价

**方法价值。** ΔFM 最大的价值是“性价比”：一项几行代码、零推理成本、零额外前向的损失修改，在多个模型尺度、分辨率和训练框架上一致带来数个点的 FID 改善和大幅的 sFID 改善，并且能与 REPA、CFG 叠加。对于需要少步采样、又不想付出两倍 CFG 成本的场景，它是一个值得默认尝试的开关。

**解释与证据的落差。**

1. **“对比 / 条件分离 / 流唯一性”的叙事与闭式解不符**：总体最优解中的排斥项与条件 $y$、当前状态 $x_t$ 都无关，只是一个常数偏移加整体缩放。说 ΔFM “让不同条件的流彼此远离”并不准确，它让所有流都远离同一个均值轨迹。
2. **“更多样”的说法没有证据**：Recall 不升反略降，precision 上升，toy 实验也显示它偏离了真实条件分布。ΔFM 的收益是保真度 / 典型性，而不是多样性。
3. **9 倍 / 5 倍加速依赖 NFE=50 的评测协议**：在更多采样步数下，两者差距明显缩小。
4. **Batch size 表的标签不一致**，“随 batch 扩展”的结论也与数据不完全吻合。
5. **公式笔误较多**：插值边界条件、Eq.12 的“损失值 / 最优解”混用、正文 CFG 修正式中的 $\tau$，以及 5.4 节称“Eq.4 的闭式解”实为 Eq.6。

**实验覆盖的不足。**

- 没有与任何其他免 CFG / 训练期引导方法比较；
- 文生图只有 CC3M 一个 FID，没有对齐指标，也没有更大规模数据；
- 没有多随机种子与置信区间；
- λ 消融只在一个设置上进行，而有效窗口很窄；
- 没有报告不带修正的 “ΔFM + 标准 CFG”，无法评估修正公式的必要性；
- 最强结果（CFG 下 FID 1.97）只训练到 400K，与 REPA 的长训练 SOTA 不在同一可比区间。

**适用边界（个人判断）。**

- 由于 $\hat{T}=-\mu$，效果依赖数据 latent 是否零均值。很多视频 / 3D latent 模型会做逐通道归一化，此时 ΔFM 近似退化为单纯的速度缩放，收益可能不同；
- 对于需要精确拟合分布的任务（例如似然评估、科学数据生成、需要完整覆盖模态的场景），ΔFM 带来的分布偏移是缺点而非优点；
- 若推理已使用较强 CFG，叠加 ΔFM 的边际收益很小。

---

## 8. 复现与实践建议

**代码可用性：好。** 官方仓库直接基于 REPA 代码库，训练只需加 `--loss-type "contrastive" --contrastive-weight=0.05`，带 CFG 评测需要传入预先计算的 interference vector（`interference_vectors/imnet256_interference_vector.pt`）和对应权重；HuggingFace 提供了 REPA SiT-XL/2 256 / 512 的 ΔFM 检查点。

**复现成本**：与 REPA 相同。ΔFM 本身没有额外开销，但复现 REPA SiT-XL/2 400K iteration 仍需要多卡 A100 / H100 级别的训练资源（论文未披露具体硬件和时长）；SiT-B/2 的实验成本低得多，适合快速验证。

**最值得做的验证实验（个人建议）**：

1. **“推理期等价”测试**：取一个普通 FM 训练的模型，推理时直接使用 $v'=(v_{FM}+\lambda\mu)/(1-\lambda)$，其中 $\mu$ 为训练集 latent 均值，看能复现多少 ΔFM 的收益。如果大部分收益都能复现，就说明训练期的对比项主要起“弱引导”作用，在已有模型上可以零训练成本获得；如果复现不了，则说明训练动力学（负样本噪声的正则化效应）才是关键。这是检验论文机制最便宜、也最有信息量的实验。
2. **负样本采样方式**：比较默认的“任意非自身样本”与代码中已有但未启用的 `class_conditioned_sampling`（只取不同类别）。后者的负目标期望变为“其他类的均值”，与类别相关，才更接近论文所说的“条件间对比”。
3. **在多步采样（如 250 步）和 ODE 采样器下**重新测量 FM 与 ΔFM 的差距，确认加速倍数对评测协议的依赖程度。
4. **迁移到新领域时重新搜索 λ**：建议在 {0.01, 0.03, 0.05, 0.08} 范围内扫描，同时监控 precision / recall 的权衡，而不只看 FID。

---

## 9. 个人启发与后续问题

1. **“训练目标里烘焙引导”是一个可推广的思路。** ΔFM 本质上把一次固定强度的引导写进了损失函数。顺着这个方向，可以设计“推离某个参考预测器”的训练目标：例如推离一个小模型或早期 checkpoint 的速度（训练期 autoguidance），推离类均值而非全局均值，或者让 λ 随时间步 $t$ 变化，从而只在高噪声段施加引导，对应 CFG 中的引导区间。
2. **对 3D 生成的直接价值。** TRELLIS、TripoSG 等 3D 生成模型都用 rectified flow 训练，并大量依赖 CFG。ΔFM 可以零成本接入这些训练流程，但要先检查 3D latent 是否做了零均值归一化：若已零均值，ΔFM 退化为速度缩放，此时直接在推理期做速度缩放或许就够了。
3. **与“VAE 到底在压缩什么”的关联。** ΔFM 的作用依赖 latent 的一阶统计量 $\mu$。这提示 latent 空间的统计结构（均值、各通道尺度）不只是归一化细节，还会通过这类损失直接影响生成偏置。在比较不同 VAE / tokenizer 时，latent 的均值与尺度应作为显式变量记录。
4. **值得追问的开放问题**：既然总体最优只依赖负目标的均值，那么“对比”带来的额外随机性（负样本方差）究竟是正则化的帮助，还是只是噪声？λ=0.15 时的崩溃说明训练动力学与总体最优解之间存在实质差异，这部分论文完全没有分析。

---

## 10. 材料来源

- 官方论文 PDF：`assets/Contrastive Flow Matching - 对比目标增强条件流分离/sources/paper.pdf`（arXiv 2506.05350v1，13 页，附录 A–B 在同一 PDF 中，无单独补充材料）
- 来源清单：`assets/Contrastive Flow Matching - 对比目标增强条件流分离/sources.md`
- 官方代码：https://github.com/gstoica27/DeltaFM （核对了 `triplet_loss.py` 中的负样本采样与损失计算）
- 官方权重：https://huggingface.co/gstoica3/DeltaFM
