---
title: "VAE、Attention、DiT 与 Flow Matching 基础模型路线"
type: survey
status: complete
domain: Foundations
created: 2026-10-16
updated: 2026-10-16
---

# VAE、Attention、DiT 与 Flow Matching：从表示空间到生成动力学

## 1. 核心结论：四者不是同层替代

VAE、Transformer、DiT 与 Flow Matching 经常同时出现在现代图像生成系统中，但它们回答的不是同一个问题。把四者理解成一条“旧模型被新模型依次替代”的路线，会同时误读它们的论文贡献与系统关系。更准确的框架是：

$$\text{表示空间}\rightarrow\text{信息交互}\rightarrow\text{生成骨干}\rightarrow\text{概率路径与训练动力学}$$

四篇源论文分别落在这四层：

1. [[VAE - 变分推断与重参数化生成模型]]讨论如何学习连续潜变量、近似后验与可微随机表示，回答“数据在哪个状态空间中表达，以及如何从观测推断 latent”；
2. [[Transformer - 自注意力与序列建模]]讨论 token 之间如何按内容动态交换信息，回答“状态内部的不同位置怎样交互”；
3. [[DiT - 可扩展扩散Transformer]]把 latent patch token、时间条件与 Transformer 组装成可扩展的扩散去噪骨干，回答“由什么网络执行每个时刻的生成预测”；
4. [[Flow Matching - 条件概率路径与连续生成]]把生成训练表述为对概率路径速度场的回归，回答“概率质量沿什么路径运动、网络应预测什么，以及采样时积分什么方程”。

因此，VAE 不与 Transformer 竞争，DiT 也不与 Flow Matching 竞争。一个现代生成器完全可以同时使用：

$$\text{VAE latent}+\text{Attention/Transformer}+\text{DiT backbone}+\text{Flow Matching dynamics}$$

最重要的判断原则是：看到一个新方法时，先问它改变了哪一层，再讨论它是否替代已有组件。

---

## 2. 四层框架与模块接口

### 2.1 表示空间：把高维观测变成可建模状态

表示层决定生成模型直接操作的变量。对图像而言，可以直接在像素空间建模，也可以先经编码器得到压缩 latent。VAE 的基础概率模型为：

$$p_\theta(x,z)=p(z)p_\theta(x\mid z)$$

其价值不只是“压缩图片”，而是给出一套带先验、近似后验、概率解码器和可训练下界的连续隐变量框架。后来的 latent diffusion 通常使用更强的卷积 autoencoder，并加入感知或对抗式重建目标；它与 2013 年 VAE 有思想继承，但不是原始 VAE 架构的直接照搬。

表示层向下一层交付的是 $z$，同时也设定了信息瓶颈：编码器已丢失的信息，后续再强的 DiT 或 Flow Matching 也无法完全恢复。

### 2.2 信息交互：token 之间如何读取彼此

Transformer 的核心算子是 scaled dot-product attention：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

它让每个 query 依据内容从可见 key/value 中读取信息。Multi-head attention 在多个投影子空间中并行执行这种读取；FFN 则负责逐 token 的通道变换。Attention 是信息交互算子，Transformer 才是由 attention、FFN、位置表示、残差连接和归一化共同构成的架构。

信息交互层不规定输入必须是文本，也不规定训练目标必须是自回归、扩散或流。只要状态可被组织成 token，attention 就可以成为其混合机制。

### 2.3 生成骨干：把当前状态与条件映射为预测

DiT 将带噪 latent 划分为 patch token，以 Transformer 作为时间条件网络。其典型函数接口可以抽象为：

$$f_\theta(z_t,t,c)\rightarrow y_t$$

其中 $z_t$ 是当前生成状态，$t$ 是时间或噪声水平，$c$ 是类别、文本或其他条件，$y_t$ 可以被动力学层解释为噪声、速度、score、clean sample 或其他参数化。

DiT 的关键贡献是骨干结构和扩展性，而不是发明新的概率路径。原始 DiT 固定了成熟的 DDPM/ADM 训练 recipe，以便研究 Transformer backbone、patch size、计算量与生成质量之间的关系。

### 2.4 概率路径与训练动力学：状态怎样从噪声走向数据

动力学层定义三个对象：

1. 中间时刻状态服从什么概率路径 $p_t$；
2. 网络应预测噪声、score、速度还是其他量；
3. 生成时使用离散反向转移、ODE 还是 SDE，以及采用什么求解器。

Flow Matching 直接学习连续速度场：

$$\frac{dx_t}{dt}=v_\theta(x_t,t)$$

它把“路径”和“骨干”解耦：同一个 U-Net 或 DiT 可以在不同 probability path 上训练；同一条 path 也可以配不同预测参数化。采样效率不仅由网络强弱决定，也受轨迹曲率、时间参数化与数值求解器影响。

### 2.5 四层的输入输出契约

| 层 | 输入 | 输出 | 核心问题 | 代表工作 |
|---|---|---|---|---|
| 表示空间 | 原始数据 $x$ | latent $z$ 与解码器 | 在哪里建模，保留什么信息 | VAE、VQ-VAE、VQGAN、LDM autoencoder |
| 信息交互 | token 序列 | 混合后的 token | 哪些位置交换什么信息 | Transformer、ViT、FlashAttention |
| 生成骨干 | $z_t,t,c$ | 时间相关预测 $y_t$ | 用什么网络逼近生成映射 | U-Net、DiT、U-ViT、MMDiT |
| 路径与动力学 | 端点分布、路径、目标 | loss 与 sampler | 状态如何运动、如何监督和积分 | DDPM、Flow Matching、Rectified Flow、SiT |

这张表也揭示了跨层组合的自由度：更换 VAE 会改变状态空间；更换 attention kernel 主要改变交互实现；更换 DiT/U-Net 会改变函数逼近器；更换 DDPM/FM/RF 会改变路径、目标或采样动力学。

---

## 3. 历史边界：原论文、桥接工作与后世组合

### 3.1 时间线不能直接当成替代链

| 时间 | 工作 | 原始贡献所在层 | 没有直接解决的问题 |
|---|---|---|---|
| 2013/2014 | VAE / AEVB | 连续潜变量、重参数化、摊销推断 | 高保真图像 tokenizer、扩散生成、Transformer |
| 2017 | Transformer | 全局 self-attention 序列交互与并行架构 | 视觉 patch 化、latent diffusion、连续生成路径 |
| 2022-10 | Flow Matching | CNF 的 simulation-free 速度回归与可设计概率路径 | Transformer 生成骨干、大规模 latent DiT |
| 2022-12 / 2023 | DiT | latent diffusion 中的可扩展 Transformer denoiser | 新概率路径、Flow Matching、快速 ODE 求解 |

Flow Matching 与 DiT 是同期但从不同问题出发的工作。原始 Flow Matching 图像实验主要使用 U-Net；原始 DiT 使用 DDPM/ADM 式噪声预测，而不是 Flow Matching。今天常见的“DiT + flow/rectified flow”是模块成熟后形成的组合，不能倒写成任一原论文已经完成的系统。

### 3.2 VAE 到 DiT 之间缺少 LDM 这座桥

VAE 原论文使用浅层 MLP，在 MNIST 与 Frey Face 上验证 SGVB/AEVB。DiT 使用的是 Stable Diffusion/LDM 路线提供的预训练卷积 VAE，将 $256\times256\times3$ 图像压缩为 $32\times32\times4$ latent。两者之间还经历了更强 autoencoder、感知压缩和 latent diffusion 的发展。

所以“DiT 建立在 VAE 上”在系统层面成立，但准确表述应是：DiT 建立在 **LDM 风格预训练 autoencoder 的 latent 接口**上，而不是直接复用 2013 年 VAE 的网络与训练设置。

### 3.3 Transformer 到 DiT 之间缺少 ViT 这座桥

原始 Transformer 面向机器翻译，采用 encoder–decoder、词 token、正弦位置编码和 Post-LN。ViT 将二维图像切成 patch token，建立了视觉 Transformer 的基本输入接口。DiT 再把对象从 clean image patch 改成带噪 latent patch，并加入 timestep 与类别条件。

因此 Attention → Transformer → ViT → DiT 是一条“交互算子逐步进入视觉生成”的路线，而不是原始 Transformer 直接提出了图像扩散骨干。

### 3.4 Flow Matching 与 Rectified Flow 是同期相邻路线

Flow Matching 从 CNF、概率路径与条件混合出发；Rectified Flow 从 coupling、直线插值与整流出发。在线性插值、$\sigma_{\min}=0$ 的特例下，两者都可出现：

$$x_t=(1-t)x_0+tx_1,\qquad u_t=x_1-x_0$$

但两者的理论叙事和算法扩展不同。Flow Matching 原论文不包含多轮 reflow；Rectified Flow 也不能被简单写成 Flow Matching 的后续版本。严谨比较必须说明 coupling、端点平滑、路径定义和是否进行 reflow。

### 3.5 SiT 与 MMDiT 属于后续系统化组合

SiT 保留 DiT 类可扩展骨干，把 interpolant、预测参数化、ODE/SDE sampler 作为主要实验变量，清晰展示“架构层”和“动力学层”可以分离研究。MMDiT 则把文本与图像 token 的条件交互升级为多模态联合注意力，并与 rectified-flow 式训练组合，用于高分辨率文生图。

这两项工作说明四层框架具有解释力，但它们的结果不能反向作为 VAE、Transformer、DiT 或 Flow Matching 原论文的实验证据。

---

## 4. 第一层：VAE 建立可学习的连续表示空间

### 4.1 ELBO 把不可解似然变成可优化目标

对近似后验 $q_\phi(z\mid x)$，VAE 最大化：

$$\mathcal L_{\mathrm{VAE}}=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p(z)\right)$$

第一项是由观测模型定义的条件对数似然，不天然等于任意 MSE；第二项约束近似后验与先验。ELBO 同时训练生成模型与推断模型，但它只是 log evidence 的下界，受变分族和摊销误差限制。

### 4.2 重参数化让随机节点可反向传播

对角高斯近似后验可写为：

$$z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad\epsilon\sim\mathcal N(0,I)$$

随机性被移到与参数无关的噪声 $\epsilon$ 中，梯度可以沿 $x\rightarrow(\mu,\sigma)\rightarrow z\rightarrow\text{decoder}$ 传播。其意义是 pathwise gradient，而不只是某个“采样技巧”。

### 4.3 摊销推断建立了 encoder 接口

传统变分推断可能为每个样本单独优化局部后验参数；AEVB 用共享 encoder 直接输出后验参数：

$$\lambda_i=f_\phi(x^{(i)})$$

这使新样本的推断只需一次前向传播，也为后来“先编码到 latent，再训练另一个生成模型”提供了可复用接口。

### 4.4 表示层的收益与代价

VAE/LDM 风格 latent 带来三项主要收益：降低空间分辨率、过滤部分感知冗余、让生成骨干把算力用于语义结构。但压缩也会形成不可逆上限：细小文字、高频纹理和难以重建的几何可能在 denoiser 训练前已经丢失。

评估 latent 生成器时，必须单独报告 VAE reconstruction upper bound。否则无法判断最终误差来自表示瓶颈，还是来自 DiT 或生成动力学。

---

## 5. 第二层：Attention 定义全局信息交互

### 5.1 Attention 是内容寻址，不是完整模型

$QK^\top$ 计算 query 与 key 的兼容性，softmax 把分数变成读取权重，最后聚合 value。除以 $\sqrt{d_k}$ 是为了控制高维点积方差和 softmax 饱和。Multi-head 允许不同投影子空间并行建立关系，但并不保证每个 head 都对应稳定、可解释的语义功能。

完整 Transformer 还包含 FFN、位置表示、残差路径、归一化、mask 与输出层。把它缩写成“只有 attention”会忽略大量容量与训练稳定性来源。

### 5.2 全局交互的优势与结构性成本

标准 self-attention 让任意两个位置在一层内直接交互，训练时可以并行计算所有 token；代价是 attention score 和中间存储随 token 数近似二次增长：

$$\text{attention cost}=O(T^2d)$$

在 DiT 中，patch size 减半会让二维 token 数增至四倍，从而显著增加计算。原始 DiT 的实验说明，增加 token 计算可提升质量；但在高分辨率生成中，这也会迅速暴露 attention 的显存和 IO 瓶颈。

### 5.3 FlashAttention 改变实现，不必改变数学

FlashAttention 类方法通常保持标准 softmax attention 的数学结果，通过 tiling、kernel 融合和减少 HBM 读写提升速度、降低显存占用。它属于“信息交互层的系统实现优化”，不能与改变可见连接或近似 attention 的架构方法混为一谈。

这一区分很重要：渐近复杂度未必变化，但真实吞吐和可训练序列长度可能大幅改善。

---

## 6. 第三层：DiT 把 Transformer 变成生成骨干

### 6.1 DiT 的端到端形状流

以 ImageNet $256^2$、VAE 下采样 8 倍、latent patch size $p=2$ 为例：

$$256\times256\times3\xrightarrow{E}32\times32\times4\xrightarrow{\mathrm{patchify}}256\ \text{tokens}\xrightarrow{\mathrm{DiT}}\text{noise/variance prediction}\xrightarrow{D}\hat x$$

DiT 改变的是 diffusion denoiser 的宏观拓扑：从多尺度卷积 U-Net 改为固定 latent 分辨率的 Transformer block 堆叠。VAE 仍然存在，前向扩散和主要 loss 也沿用既有扩散 recipe。

### 6.2 时间与条件必须进入每一层

DiT 的网络不仅要看 $z_t$，还必须知道当前噪声水平 $t$ 和条件 $c$。论文比较 in-context token、cross-attention、adaLN 与 adaLN-Zero。最佳的 adaLN-Zero 由条件生成归一化 scale、shift 与 residual gate，并对调制输出做零初始化，使 block 初始近似恒等映射。

这说明条件化不是外围接口，而是生成骨干的核心结构变量。类别是单个全局条件，适合 adaLN；长文本是条件 token 序列，通常需要更高带宽的 cross/joint attention。

### 6.3 DiT 原始训练目标仍是扩散目标

原始 DiT 采用高斯前向扩散：

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad\epsilon\sim\mathcal N(0,I)$$

主要噪声预测目标为：

$$\mathcal L_{\mathrm{simple}}=\mathbb E\left[\left\|\epsilon-\epsilon_\theta(z_t,t,c)\right\|_2^2\right]$$

所以 DiT 首先是 backbone 论文，而不是新 diffusion path 论文。其输出也不只有噪声：原始实现还预测 learned variance，末端输出为 $2C$ 通道。

### 6.4 DiT 的尺度结论及边界

在论文的 12 个模型设计空间内，增加深度/宽度或减小 patch、增加 token，都会提升 forward Gflops 并改善 FID。这个结果支持“更多 backbone 计算可换来更强生成能力”，但不构成跨架构、跨任务、跨硬件的通用 scaling law。

最终 FID 还受训练步数、CFG、VAE decoder 和评估协议影响。比较不同 DiT 时，至少要同时标明 backbone、patch size、训练步数、采样步数或 NFE、guidance scale 与 decoder。

---

## 7. 第四层：Flow Matching 重写概率路径与训练监督

### 7.1 从 CNF 到 simulation-free 训练

连续归一化流用向量场定义状态演化：

$$\frac{d}{dt}\phi_t(x)=v_t(\phi_t(x)),\qquad p_t=[\phi_t]_*p_0$$

概率密度满足连续性方程：

$$\partial_t p_t(x)+\nabla_x\cdot\left(p_t(x)v_t(x)\right)=0$$

经典 CNF 最大似然训练常把 ODE 求解放进训练内环。Flow Matching 改为先指定概率路径及其目标速度，再在随机时间点做监督回归，训练时不需要展开完整 ODE 轨迹。

### 7.2 CFM 为什么可以替代不可计算的边际监督

理想 FM 需要边际速度 $u_t(x)$，但它通常不可直接计算。CFM 为每个数据样本构造可采样条件路径，并回归解析条件速度：

$$\mathcal L_{\mathrm{CFM}}=\mathbb E_{t,x_1,x_t}\left\|v_\theta(x_t,t)-u_t(x_t\mid x_1)\right\|^2$$

边际最优预测器是条件速度的后验平均：

$$v_t^*(x)=\mathbb E[u_t(x\mid x_1)\mid x_t=x]=u_t(x)$$

CFM 与理想 FM 的期望目标只差一个与参数无关的常数，因此期望梯度相同。这个结论不表示单个 minibatch 上两个 loss 数值相等。

### 7.3 线性条件路径的最小形式

当使用直线插值且不保留端点平滑时：

$$x_0\sim\mathcal N(0,I),\qquad x_1\sim q,\qquad t\sim U[0,1]$$

$$x_t=(1-t)x_0+tx_1$$

$$u_t=x_1-x_0$$

训练变成普通速度回归：

$$\mathcal L=\mathbb E\left\|v_\theta(x_t,t,c)-(x_1-x_0)\right\|^2$$

生成时从 $x_0$ 出发积分 $dx/dt=v_\theta(x,t,c)$。训练免 ODE simulation，不等于生成免 ODE；采样成本仍需用 NFE 衡量。

### 7.4 条件直线不等于边际直线，也不等于全局 OT

固定端点对的条件轨迹可以是直线，但不同条件轨迹在状态空间交叉时，网络学习的是条件速度的平均，最终边际轨迹仍可能弯曲。若 $x_0$ 与 $x_1$ 独立采样，端点 coupling 通常也不是两个边际分布之间的全局最优 coupling。

因此，“FM 学到噪声到数据的全局最优直线传输”是不准确的。正确说法是：特定条件路径具有直线或条件 OT 几何，它可能改善训练信号和低 NFE 积分，但需要实验验证最终边际向量场。

---

## 8. 端到端组合：四层如何装配成一个生成系统

### 8.1 常见的分阶段训练

现代 latent 生成系统通常不是把四层从零联合训练，而是分阶段优化：

1. **训练或选取 autoencoder**：学习 $E:x\rightarrow z$ 与 $D:z\rightarrow x$，确认重建质量和 latent scaling；
2. **冻结表示层**：把训练集编码为 latent，避免生成训练期间表示空间持续漂移；
3. **选择 token 接口**：对 $z_t$ patchify，加入二维位置表示；
4. **选择骨干**：用 DiT 接收 $(z_t,t,c)$，输出与 latent 同形状的预测；
5. **选择动力学**：使用 DDPM 噪声目标，或使用 Flow Matching/Rectified Flow 速度目标；
6. **选择采样器**：按离散扩散反向迭代，或用 ODE/SDE solver 积分；
7. **解码**：将生成 latent 经 $D$ 还原到像素空间。

这套装配关系可以写成：

$$x\xrightarrow{E}z_1,\quad z_t=\mathcal I_t(z_0,z_1),\quad y_\theta=f_\theta(z_t,t,c),\quad \hat z_1=\operatorname{Solve}(f_\theta),\quad \hat x=D(\hat z_1)$$

其中插值器 $\mathcal I_t$、预测语义 $y_\theta$ 和 `Solve` 由动力学层决定；$f_\theta$ 的拓扑由骨干层决定。

### 8.2 DDPM-DiT 组合

在原始 DiT 中，$z_1$ 是 clean latent，$z_t$ 由噪声日程构造，网络主要预测 $\epsilon$，采样器执行多步反向扩散。此时：

- VAE 决定 latent 表示；
- attention 决定 token 交互；
- DiT 决定 denoiser 网络；
- DDPM/ADM recipe 决定路径、目标与反向采样。

### 8.3 FM-DiT 组合

在 Flow Matching 版本中，可以保持同一 VAE、patchify 和 DiT，只修改训练样本构造与输出语义：

- 从先验 latent $z_0$ 与数据 latent $z_1$ 构造 $z_t$；
- DiT 输出 velocity $v_\theta(z_t,t,c)$；
- 用条件速度做 MSE；
- 采样时积分 ODE，而不是执行原始 DDPM 反向转移。

这正是“DiT 与 Flow Matching 不竞争”的直接证据：DiT 提供函数逼近器，FM 定义它应逼近的速度场。

### 8.4 条件通路是第五个横切维度

文本、类别、布局或图像条件会横跨交互层、骨干层和动力学层。类别可通过 adaLN 注入；文本通常需要 cross-attention 或 joint attention；CFG 又会在采样时组合条件与无条件预测。

条件化不能被简单归入某一层，但仍应分清：条件怎样编码和交互属于架构问题，guidance 怎样改变预测属于采样动力学问题。

---

## 9. 核心概念对照表

| 概念 | 所在层 | 精确定义 | 常见混淆 |
|---|---|---|---|
| Latent variable | 表示 | 生成观测的未观测随机变量 | 不等于任意压缩向量 |
| ELBO | 表示/推断 | 对 log evidence 的可优化变分下界 | 不只是“重构损失加正则” |
| Reparameterization | 表示/优化 | 将参数化采样改写为固定噪声的可微变换 | 不等于消除随机性 |
| Amortized inference | 表示/推断 | 用共享 encoder 预测样本后验参数 | 会引入 amortization gap |
| Self-attention | 信息交互 | 同一 token 集内部的内容寻址 | 不是完整 Transformer |
| Multi-head attention | 信息交互 | 在多个投影子空间并行 attention | 不保证每个 head 可解释 |
| Patchify | 接口 | 将二维状态分块并映射为 token | patch 越小，token 和成本越高 |
| DiT | 生成骨干 | 以 ViT 风格 Transformer 作为时间条件生成网络 | 不是一种独立概率路径 |
| adaLN-Zero | 骨干/条件 | 条件调制归一化与残差门，并零初始化 | 不等于 cross-attention |
| Probability path | 动力学 | 连接先验与数据端的时间相关分布族 $p_t$ | 不等于单条粒子轨迹 |
| Vector field | 动力学 | 定义状态瞬时速度的函数 $v_t(x)$ | 同一路径的速度场未必唯一 |
| Flow Matching | 动力学/目标 | 回归生成目标概率路径的边际速度 | 不是某种特定 backbone |
| Conditional Flow Matching | 动力学/目标 | 用可采样条件速度监督边际速度学习 | minibatch loss 不等于 FM loss |
| Rectified Flow | 动力学/coupling | 以直线插值与整流为核心的流模型路线 | 不等于 Flow Matching 的简单别名 |
| ODE solver | 采样 | 对学习到的向量场进行数值积分 | steps 不总等于 NFE |
| CFG | 条件/采样 | 组合条件与无条件预测增强条件一致性 | 会带来质量—覆盖率权衡 |

---

## 10. 十个高频误区

### 10.1 “VAE、Transformer、DiT、Flow Matching 是四代生成模型”

错误。它们分属表示、交互、骨干与动力学层，可以同时存在于同一系统中。

### 10.2 “VAE 就是 MSE 加 KL”

不完整。重构项来自 decoder likelihood；只有特定固定方差 Gaussian 观测模型才与加权 MSE 对应。VAE 的核心还包括 ELBO、重参数化和摊销推断。

### 10.3 “DiT 去掉了 VAE 和卷积”

错误。原始 DiT 实验使用预训练卷积 VAE，只把 latent diffusion 的 U-Net denoiser 换成 Transformer。

### 10.4 “Attention 就是 Transformer”

错误。Attention 负责 token mixing；完整 Transformer 还依赖 FFN、位置表示、残差、归一化、mask 和条件结构。

### 10.5 “Transformer 训练并行，所以生成也天然并行”

错误。原始自回归 Transformer 的推理仍逐 token；扩散或流式 Transformer 的网络内部可并行，但采样还要跨时间多次调用网络。

### 10.6 “DiT 是一种 diffusion loss”

错误。DiT 是骨干架构。原始论文使用 DDPM/ADM 式目标，后续可以改用 velocity、score、Flow Matching 或 stochastic interpolant 目标。

### 10.7 “Flow Matching 不需要采样步骤”

错误。FM 训练不在内环求解 ODE，但生成仍需积分。是否能用较少 NFE，取决于路径、coupling、网络误差和 solver。

### 10.8 “条件直线意味着最终生成轨迹都是直线”

错误。条件速度在轨迹交叉处被平均，学习到的边际向量场可能弯曲。

### 10.9 “FM-OT 求得了噪声与数据之间的全局最优传输”

错误。原论文严格保证的是条件高斯端点之间的 OT displacement path；独立端点 coupling 一般不是全局最优 coupling。

### 10.10 “采样 20 steps 就等于 20 次网络调用”

错误。Euler 通常每步 1 次，midpoint 约 2 次，RK4 约 4 次。跨论文比较应报告 NFE，而不是只报 steps。

---

## 11. 如何定位性能收益来自哪一层

### 11.1 表示层诊断

先检查 autoencoder reconstruction，而不是直接看生成 FID：

- 重构是否已损失文字、纹理或几何；
- latent scaling 是否与预训练 VAE 一致；
- decoder 版本是否变化；
- pixel-space 与 latent-space 比较是否控制了网络计算量。

如果重构上限已经很差，继续扩大 DiT 只会更准确地拟合受损 latent 分布。

### 11.2 交互与骨干层诊断

固定 VAE、路径、目标和采样器，只改变：

- U-Net / DiT / U-ViT；
- depth、width、head 数与 patch size；
- adaLN、cross-attention 或 joint attention；
- 标准 attention 与 FlashAttention 实现。

同时记录参数量、理论 FLOPs、真实吞吐、峰值显存与单次前向时间。参数量相同不代表 token 计算相同，FLOPs 相同也不代表 wall-clock 相同。

### 11.3 动力学层诊断

固定 VAE 与 backbone，比较：

- DDPM noise prediction；
- velocity prediction；
- FM-Diffusion；
- 线性 CFM / Rectified Flow；
- 不同时间采样、loss weighting 与 solver。

评价应包含固定 NFE 下的质量，以及达到同一质量所需 NFE。不同目标的训练 loss 数值没有直接可比性。

### 11.4 系统层诊断

最终系统指标至少应拆成：

| 指标 | 主要反映 |
|---|---|
| VAE reconstruction FID/LPIPS/PSNR | 表示瓶颈与 decoder 上限 |
| 训练 loss 分项 | 优化是否稳定、目标是否实现正确 |
| FID/KID/Precision/Recall | 分布质量与 fidelity–diversity 权衡 |
| NFE | 采样网络调用成本 |
| wall-clock 与峰值显存 | 真实系统效率 |
| 条件一致性指标 | 条件通路与 guidance 效果 |

单个 FID 无法归因，也无法说明采样效率或表示质量。

---

## 12. 推荐学习顺序

### 12.1 第一阶段：先学概率表示

先读 VAE，掌握联合分布、边缘似然、后验、KL、ELBO、重参数化与摊销推断。目标不是立即追求高保真生成，而是能清楚回答：encoder 输出的是点还是分布，decoder 定义的是什么 likelihood，生成时为什么从先验采样。

### 12.2 第二阶段：再学信息交互

读 Transformer 时，把注意力公式、multi-head、FFN、位置编码、mask、残差和归一化分开理解。重点区分训练并行、计算复杂度与自回归推理串行。

随后补 ViT，建立二维 patch 到 token 的桥梁；没有这一步，直接读 DiT 容易把视觉 token 化误认为 DiT 首创。

### 12.3 第三阶段：补齐 latent diffusion

在 DiT 前先读 LDM，理解为什么要在 autoencoder latent 中扩散、VAE 压缩倍率如何改变 token 数、latent scaling 如何进入训练，以及 cross-attention 如何承载文本条件。

### 12.4 第四阶段：读 DiT 的骨干与扩展实验

此时再读 DiT，重点观察三个轴：模型深宽、patch size、条件注入。把论文结论限定为“固定 LDM/ADM recipe 下的 backbone 研究”，不要把 FID 改善全部归因于 Transformer 名称。

### 12.5 第五阶段：读 Flow Matching 与 Rectified Flow

先从 CNF、连续性方程、概率路径、向量场和 NFE 入手，再读 CFM 等价性与线性路径。随后对照 Rectified Flow，分清 probability-path 叙事与 coupling/rectification 叙事。

### 12.6 第六阶段：用 SiT 与 MMDiT 收束

最后读 SiT 和 MMDiT：前者帮助分离骨干、interpolant、预测目标与 sampler；后者展示复杂文本条件如何推动单流全局调制升级为多模态联合交互。此时四层框架才真正闭环。

简化后的学习顺序为：

$$\text{VAE}\rightarrow\text{Transformer}\rightarrow\text{ViT}\rightarrow\text{LDM}\rightarrow\text{DiT}\rightarrow\text{Flow Matching/Rectified Flow}\rightarrow\text{SiT/MMDiT}$$

---

## 13. 最小复现配方

### 13.1 目标与原则

复现目标不是一次追求 SOTA，而是用同一数据、同一 VAE 和同一骨干，验证四层接口确实可独立替换。推荐先在 CIFAR-10、CelebA-64 或其他可承受的小图像数据上完成闭环，再扩展分辨率。

所有实验应固定随机种子，并报告多个随机种子的重复运行结果；每次只改变一层。不要同时更换 VAE、backbone、path 与 sampler，否则无法归因。

### 13.2 步骤一：验证表示层

训练一个最小连续 VAE，encoder 输出 `mu` 与 `logvar`：

$$z=\mu+\exp(0.5\,\mathrm{logvar})\odot\epsilon$$

最小化 reconstruction NLL 与 KL，并确保两项 reduction 一致。报告 reconstruction、prior sample、KL、active dimensions；检查 `std = exp(0.5 * logvar)`、KL 符号和无条件生成是否从 $p(z)$ 采样。

若主要目标是复现 DiT/FM 组合，也可以使用固定预训练 autoencoder，但必须锁定 encoder、decoder、latent scaling 与压缩倍率。

### 13.3 步骤二：建立最小 latent DiT

1. 将图像编码为 latent $z_1$；
2. 对带噪或插值后的 $z_t$ 做非重叠 patchify；
3. 加二维位置编码；
4. 用小型 DiT，例如 6–12 个 block；
5. 用 timestep embedding 调制每个 block；
6. 无条件实验可先省略文本，只保留时间条件；
7. 输出维度与 latent 对齐；若忠实复现原始 DiT learned variance，则输出 $2C$。

先用一个 batch 做 shape、过拟合和反向传播检查，再启动完整训练。

### 13.4 步骤三：训练 DDPM-DiT 基线

构造：

$$z_t=\sqrt{\bar\alpha_t}z_1+\sqrt{1-\bar\alpha_t}\epsilon$$

训练：

$$\mathcal L_{\epsilon}=\mathbb E\|\epsilon-\epsilon_\theta(z_t,t)\|^2$$

记录噪声日程、时间采样、EMA、采样步数和是否使用 CFG。先确保多步采样能稳定解码，再把该结果作为动力学替换的基线。

### 13.5 步骤四：保持骨干不变，替换为线性 CFM

保持同一 VAE、patchify、DiT 宽深与优化预算，只修改训练对：

$$z_0\sim\mathcal N(0,I),\qquad z_t=(1-t)z_0+tz_1,\qquad u=z_1-z_0$$

训练 velocity：

$$\mathcal L_v=\mathbb E\|v_\theta(z_t,t)-u\|^2$$

生成时用 Euler 先完成正确性验证，再比较 midpoint 或 RK4。报告 solver steps 与真实 NFE；不要把 RK4 的 20 steps 写成 20 NFE。

### 13.6 步骤五：做最小受控比较

至少比较以下项目：

| 比较 | 固定项 | 改变量 | 目的 |
|---|---|---|---|
| Pixel vs latent | 近似训练预算 | 表示空间 | 测表示压缩收益与上限 |
| U-Net vs DiT | VAE、目标、sampler | 骨干 | 测架构贡献 |
| 大 patch vs 小 patch | backbone 宽深 | token 数 | 测计算分配与空间粒度 |
| DDPM vs CFM | VAE、DiT、数据 | 路径与目标 | 测动力学贡献 |
| Euler vs midpoint/RK4 | 同一 FM 模型 | solver | 测数值积分效率 |
| 独立 coupling vs minibatch OT | 同一线性目标 | coupling | 测监督方差与轨迹几何 |

最终至少报告：VAE reconstruction upper bound、生成 FID/KID、Precision/Recall、固定 NFE 质量、达到目标质量所需 NFE、峰值显存与真实采样时间。

### 13.7 二维 toy 实验不可省略

在二维高斯混合或 checkerboard 上同时画出：条件插值线、学习到的边际向量场和 ODE 生成轨迹。这个实验能直观看出三个常被混淆的对象并不相同：

1. 固定端点对的条件路径；
2. 条件速度的边际平均；
3. 数值求解器实际得到的轨迹。

它也是检查时间方向、速度符号和 solver 实现是否正确的最低成本方法。

---

## 14. 下一批九篇精读队列

### 14.1 P0：补齐主干桥梁

1. **ViT**：补 Transformer → DiT 之间的视觉 patch 化。重点读 patch embedding、位置编码、class token，以及模型规模与数据规模的关系。
2. **LDM**：补 VAE → DiT 之间的 latent diffusion。重点读 autoencoder 目标、压缩倍率、latent scaling、cross-attention 与像素/latent 成本差异。
3. **Rectified Flow**：补 Flow Matching 的同期相邻路线。重点读 coupling、直线插值、rectification、reflow，以及它与线性 CFM 的同异。
4. **SiT**：验证“骨干与动力学解耦”。重点读 interpolant、velocity/score 参数化、ODE/SDE sampler 与 DiT 对照。

这四篇优先级最高，因为它们分别修补 Transformer→DiT、VAE→DiT、FM↔RF、DiT→interpolant 四个关键接口。

### 14.2 P1：扩展表示与骨干分支

5. **VQ-VAE**：研究连续 latent 之外的离散 codebook、向量量化与 straight-through estimator，理解“表示模型 + 先验模型”的两阶段拆分。
6. **VQGAN**：研究 perceptual/adversarial reconstruction 如何提升离散视觉 token 的感知质量，以及 tokenizer 目标如何影响后续生成上限。
7. **U-ViT**：研究 Transformer 如何重新引入 U-Net 式长跳连和多尺度归纳偏置，避免把 U-Net 与 Transformer 理解成互斥二选一。

这组三篇提供必要反例：表示层不只有连续 VAE latent，生成骨干也不只有原始固定分辨率 DiT。

### 14.3 P2：系统效率与多模态扩展

8. **FlashAttention**：区分 attention 数学、渐近复杂度与 IO-aware kernel 实现，理解为何不改变结果也能显著提升速度和显存效率。
9. **MMDiT**：作为综合收束，重点读文本/图像模态专属参数、joint attention、高带宽条件交互，以及它与 rectified-flow 训练的组合。

建议完整顺序为：

$$\text{ViT}\rightarrow\text{LDM}\rightarrow\text{Rectified Flow}\rightarrow\text{SiT}\rightarrow\text{VQ-VAE}\rightarrow\text{VQGAN}\rightarrow\text{U-ViT}\rightarrow\text{FlashAttention}\rightarrow\text{MMDiT}$$

若按主题并行，可分为三组：`ViT/LDM` 负责桥接，`Rectified Flow/SiT` 负责动力学，`VQ-VAE/VQGAN/U-ViT/FlashAttention/MMDiT` 负责分支与规模化。

---

## 15. 一套可复用的论文阅读问题

面对任何新生成论文，可以依次回答：

1. **状态空间是什么？** 像素、连续 latent、离散 token，还是其他结构？
2. **表示如何获得？** 联合训练、预训练并冻结，还是无 encoder 的直接建模？
3. **token 如何交互？** 卷积、self-attention、cross-attention、joint attention，还是混合结构？
4. **生成骨干是什么？** U-Net、DiT、U-ViT、MMDiT，输入输出形状是什么？
5. **条件怎样进入？** adaLN、条件 token、cross-attention、joint attention，还是 guidance？
6. **概率路径是什么？** VP/VE diffusion、线性插值、OT path、stochastic interpolant，还是自定义 schedule？
7. **网络预测什么？** noise、score、velocity、clean sample、mean/variance，还是多目标？
8. **训练是否需要模拟？** 随机单时刻回归，还是 ODE/SDE 内环？
9. **生成时解什么？** 离散反向链、ODE、SDE，solver 与 NFE 是多少？
10. **性能收益来自哪一层？** 是否有固定其他层的受控消融？

只要这十个问题被分别回答，就能避免把一个系统品牌名当成不可拆分的方法。

---

## 16. 总结

VAE、Attention、DiT 与 Flow Matching 最终汇合于现代生成系统，但它们的历史贡献位于不同抽象层。VAE 建立连续潜变量、重参数化和摊销推断接口；Transformer 提供全局内容寻址的信息交互；DiT 把视觉 latent token、时间条件与 Transformer 组装成可扩展生成骨干；Flow Matching 则把概率路径、速度监督与 ODE 采样开放为独立设计空间。

更准确的技术路线不是“VAE 被 Transformer 替代，扩散又被 Flow 替代”，而是接口逐步标准化：

$$\text{先选择生成状态}\rightarrow\text{再定义状态内部交互}\rightarrow\text{再选择时间条件网络}\rightarrow\text{最后规定路径、监督与求解器}$$

这个框架也解释了后续工作的角色：ViT 与 LDM 补齐两座桥，Rectified Flow 与 SiT 扩展动力学设计，VQ-VAE/VQGAN 与 U-ViT 提供表示和骨干分支，FlashAttention 改善交互层的系统效率，MMDiT 则把多模态条件交互与 flow-style 训练推向大规模系统。

最终应保留的不是一串模型名，而是一套分层判断：**数据在哪里建模，信息怎样交互，由什么网络预测，概率质量沿什么路径运动，以及采样器如何把预测转成样本。**

---

## 17. 四篇源笔记

1. [[VAE - 变分推断与重参数化生成模型]]
2. [[Transformer - 自注意力与序列建模]]
3. [[DiT - 可扩展扩散Transformer]]
4. [[Flow Matching - 条件概率路径与连续生成]]
