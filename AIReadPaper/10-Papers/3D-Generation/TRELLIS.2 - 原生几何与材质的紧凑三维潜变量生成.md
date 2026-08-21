---
type: paper-note
status: done
domain: 3D-Generation
paper: Native and Compact Structured Latents for 3D Generation
year: 2025
arxiv: 2512.14692
source: https://arxiv.org/abs/2512.14692
project: https://microsoft.github.io/TRELLIS.2/
code: null
tags: [3D-Generation]
---

# TRELLIS.2：原生几何与材质的紧凑三维潜变量生成

## 基本信息

- **论文全称**：Native and Compact Structured Latents for 3D Generation
- **项目名**：TRELLIS.2
- **arXiv**：[2512.14692](https://arxiv.org/abs/2512.14692)（v1，2025-12-16，cs.CV）
- **项目页**：[https://microsoft.github.io/TRELLIS.2/](https://microsoft.github.io/TRELLIS.2/)
- **作者机构**：Jianfeng Xiang（清华大学 / Microsoft Research 实习）、Xiaoxue Chen、Sicheng Xu、Ruicheng Wang、Zelong Lv（USTC）、Yu Deng、Hongyuan Zhu、Yue Dong 等，主要来自 Microsoft Research / Microsoft AI，并含清华、USTC 合作者
- **代码与数据**：论文声明模型、代码、数据集将全部公开；截至精读时项目页已给出工程信号，但 frontmatter 中 `code` 保持 `null`（未单独核实仓库地址）
- **领域**：3D-Generation（3D 资产生成 / 原生潜表示）

## 一句话结论

这篇论文提出 **O-Voxel**（Omni-Voxel）这一"无场"稀疏体素表示，把几何与 PBR 材质统一编码进同一结构化结构，再用 **SC-VAE**（16× 空间压缩）压成紧凑潜空间，最终训练 **4B 参数 flow-matching 模型**做端到端 3D 资产生成——在重建与生成质量上大幅领先现有方法，同时把 1024³ 全纹理资产的推理压缩到约 17s。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

3D 生成建模近年进展迅速，但论文认为领域仍然缺少一个能同时满足两点的**基础表示**：既能忠实刻画"任意"3D 资产的**全谱信息**（full-spectrum information），又能被神经网络高效地处理成潜变量。作者的判断是，表示层的瓶颈比生成模型本身的瓶颈更根本——生成质量的上限，很大程度上由"资产如何被数字化、被压缩"决定。

### 1.2 现有方法的瓶颈

论文把现有表示的缺陷归纳为两条主线：

- **几何表示的拓扑局限**。近期大规模 3D 生成模型（TRELLIS、Direct3D、SparseFlex 等）普遍依赖**等值面场**（iso-surface field），如 SDF、Flexicubes。这类场式表示在本质上有三处难以绕开的短板：无法稳健处理**开放曲面**（open surface）、**非流形几何**（non-manifold）、以及**封闭的内部结构**（enclosed interior structures）。例如一个内部有舱室的飞行器、一片有正反两面的叶子，场式表示很难同时表达。
- **几何与外观/材质被割裂建模**。多数工作只做**形状生成**，忽略与形状强相关的材质信息。TRELLIS（SLAT）虽然联合建模几何与外观，但它依赖**多视图 2D 图像特征输入**和**纯渲染监督**，对复杂结构与材质的捕获存在天然缺陷。

此外，潜表示本身还有一个效率问题：Perceiver 风格的**非结构化潜变量**压缩强但重建保真度受限；基于稀疏先验的**结构化潜变量**（如 SLAT）几何精度高但 token 数量大、压缩效率低。作者指出，此前的优化工作大多在"网络计算"上做文章，而非在"潜空间紧凑性"上突破。

### 1.3 本文核心贡献

论文的贡献可以归结为三个递进的层次：

1. **O-Voxel 表示**：一种"field-free"的稀疏体素结构，统一编码几何与 PBR 材质，能处理任意拓扑（含开放、非流形、封闭内部结构），并支持**瞬时双向转换**（mesh↔O-Voxel，无需优化、无需渲染）。
2. **SC-VAE**：针对 O-Voxel 设计的稀疏压缩 VAE，实现 **16× 空间下采样**，把 1024³ 全纹理资产压到约 **9.6K latent token**，且重建几乎无感知损失。
3. **4B 参数 flow-matching 生成模型**：在紧凑潜空间上做三阶段生成（结构→几何→材质），推理高效——H100 上 512³ 约 3s、1024³ 约 17s、1536³ 约 60s，生成质量远超现有模型。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

论文的核心目标是**从条件信号生成高分辨率、带完整 PBR 材质、任意拓扑的 3D 资产**。具体到论文验证的任务：

- **输入**：一张参考图像（image-to-3D，主要评测设置）；扩展能力上，第三阶段可独立接收"给定 mesh + 参考图"做 shape-conditioned texture generation。
- **输出**：一个可直接渲染的 3D 资产，含几何与六通道 PBR 材质（base color、metallic、roughness、opacity），而非仅几何壳体或仅贴图颜色。
- **关键假设**：① 输入资产可用"网格 + 纹理"形式表达（mesh → O-Voxel 转换是这条流水线的入口）；② 生成在**学习到的潜空间**中完成，而非直接在原始几何上建模；③ 几何与材质共享同一套原生 3D 潜域，天然空间对齐。

论文把表示本身设计成"原生 3D 数据 ↔ 神经网络"之间的**nexus（枢纽）**：一端面向网格/纹理资产，另一端面向稀疏卷积与 DiT。

### 2.2 关键符号和目标函数

O-Voxel 把一个资产表示为落在规则 $N\times N\times N$ 网格上的一组**稀疏体素特征元组**：

$$
\boldsymbol{f}=\left\{\left(\boldsymbol{f}^{\text{shape}}_{i},\,\boldsymbol{f}^{\text{mat}}_{i},\,\boldsymbol{p}_{i}\right)\right\}_{i=1}^{L}
$$

其中 $\boldsymbol{f}^{\text{shape}}_{i}$ 编码局部几何，$\boldsymbol{f}^{\text{mat}}_{i}$ 编码材质属性，$\boldsymbol{p}_{i}\in\{0,1,\dots,N-1\}^{3}$ 是第 $i$ 个 active 体素的坐标；不与资产相交的空体素被置为 inactive。整个表示是**稀疏**的——只有与表面相交的体素才携带信息，这既是对几何稀疏性的利用，也是后续 16× 压缩与高效推理的前提。

**优化总目标**分两层：表示转换层（O-Voxel↔mesh）追求**无损/近无损的双向映射**；学习层（SC-VAE 与生成模型）追求**在强压缩下保持重建/生成保真度**。具体损失函数在 §3.5 展开。

---

## 3. 核心方法

### 3.1 总体框架

论文的整体流水线由三段组成，对应一张 overview 图：

1. **O-Voxel 表示**：把网格/纹理资产无损转换为统一编码几何与材质的稀疏体素；
2. **SC-VAE 压缩**：把 O-Voxel 编码成紧凑的 structured latent；
3. **三阶段 flow-matching 生成**：在潜空间内依次生成"稀疏结构 → 几何 → 材质"，最后解码回可渲染资产。

![](assets/TRELLIS.2%20-%20原生几何与材质的紧凑三维潜变量生成/overview_v5.png)

> 图 1（论文 Figure 2）：方法总览。自左向右依次是 O-Voxel 表示（§3.1）、SC-VAE 潜空间学习（§3.2）与大规模 flow 生成模型（§3.3），完整串联了"原生 3D 资产 → 统一表示 → 紧凑潜变量 → 生成资产"的闭环。这张图回答了"怎么做"，是全篇的结构骨架。  
> 来源：论文 Figure 2，https://arxiv.org/abs/2512.14692

### 3.2 O-Voxel：统一几何与材质的原生表示

O-Voxel 是本文最核心的创新，其"omnipotent"体现在两点：一是**几何与外观的统一建模**，二是**对复杂度的鲁棒处理**。

#### 3.2.1 几何：Flexible Dual Grid

几何部分采用 **Flexible Dual Grid**（灵活对偶网格）表述，灵感来自 Dual Contouring（DC），但有一个关键区别：**DC 处理的是离散标量场（如 SDF），而 O-Voxel 完全不使用任何场表示**。作者直接利用网格表面本身来确定边的相交标志（而非 DC 的符号变化检测），并赋值 Hermite 数据（交点与法线）。

具体地，在"对偶"网格中：每个原始体素（primal cell）定义一个对偶顶点，每条原始边定义一个四边形面，连接相邻原始体素的对偶顶点。给定 Hermite 数据 $\{\boldsymbol{q}_i,\boldsymbol{n}_i\}$，对偶顶点 $\boldsymbol{v}$ 通过下述**二次误差函数（QEF）**闭式求解：

$$
\min_{\boldsymbol{v}\in\text{voxel}}\ e(\boldsymbol{v})
=\sum_{i} d_{\Pi,i}^{2}
+\lambda_{\text{bound}}\sum_{j} d_{L,j}^{2}
+\lambda_{\text{reg}}\, d_{\hat{\boldsymbol{q}}}^{2}.
$$

这个公式的三个分量各有直觉：

- 第一项 $d_{\Pi,i}^{2}=(\boldsymbol{n}_i\cdot(\boldsymbol{v}-\boldsymbol{q}_i))^{2}$ 是原始 DC 的 QEF，衡量 $\boldsymbol{v}$ 到由 $\{\boldsymbol{q}_i,\boldsymbol{n}_i\}$ 决定的平面的距离平方，保证顶点贴合表面；
- 第二项 $d_{L,j}^{2}$ 惩罚 $\boldsymbol{v}$ 与"穿过该原始体素的网格边界边"之间的距离，引导对偶顶点对齐边界边，**改善开放曲面的表达**（这是本文对 DC 的关键增强）；
- 第三项 $d_{\hat{\boldsymbol{q}}}^{2}=\|\boldsymbol{v}-\bar{\boldsymbol{q}}\|^{2}$ 是正则项，鼓励顶点靠近交点均值 $\bar{\boldsymbol{q}}$，使顶点分布更平滑、并**稳定 QEF 优化对抗奇异性**。

每个 active 体素的几何特征 $\boldsymbol{f}^{\text{shape}}_{i}$ 由三部分组成：对偶顶点 $\boldsymbol{v}_i\in\mathbb{R}_{[0,1]}^{3}$（表示局部表面形状）、**边相交标志** $\boldsymbol{\delta}_i\in\{0,1\}^{3}$（决定相邻对偶顶点间的四边形连接，取 $X/Y/Z$ 三轴上的 3 条预定义体素边）、以及**分裂权重** $\gamma_i\in\mathbb{R}_{>0}$（控制四边形如何自适应细分为三角形）。

Flexible Dual Grid 带来三点优势，直接对应论文动机：① **瞬时双向转换**——绕开了 SDF 评估、flood-fill、迭代优化等耗时步骤，mesh→O-Voxel 在单 CPU 上仅需数秒，O-Voxel→表面/材质重建仅需数十毫秒；② **任意拓扑建模**——不受 watertight/manifold 约束，能处理自相交曲面与封闭内部结构；③ **高精度与锐利特征保持**——对偶顶点按算法设计与局部特征对齐，可保持 sharp edges 与法线不连续。

#### 3.2.2 材质：六通道体素属性

材质特征 $\boldsymbol{f}^{\text{mat}}_{i}$ 对每个 active 体素含六通道，遵循标准 PBR 约定：

$$
\boldsymbol{f}^{\text{mat}}_{i}=(\boldsymbol{c}_{i},\,m_{i},\,r_{i},\,\alpha_{i}),
$$

其中 $\boldsymbol{c}_i\in\mathbb{R}_{[0,1]}^{3}$ 是 **base color**，$m_i$ 是 **metallic（金属度）**，$r_i$ 是 **roughness（粗糙度）**，$\alpha_i$ 是 **opacity（不透明度）**。注意 $\alpha$ 通道让 O-Voxel 能表达**半透明表面**——这是此前方法不具备的能力。材质与几何对齐：O-Voxel↔纹理的转换简单快速（投影采样 / 三线性插值重建），重建出的网格**无需后处理即可渲染**。

![](assets/TRELLIS.2%20-%20原生几何与材质的紧凑三维潜变量生成/representation.png)

> 图 2（论文 Figure 3）：O-Voxel 表示示意，以及 3D 资产与 O-Voxel 之间的瞬时双向转换。上半部分展示 Flexible Dual Grid 如何用对偶顶点/对偶面/分裂权重表达复杂拓扑；下半部分展示六通道 PBR 材质如何与几何体素对齐。这张图解释了 O-Voxel"为什么能表达任意拓扑与完整材质"。  
> 来源：论文 Figure 3，https://arxiv.org/abs/2512.14692

### 3.3 SC-VAE：稀疏压缩变分自编码器

SC-VAE 的目标是在 O-Voxel 上学习**紧凑且高保真**的潜空间。与 TRELLIS 等先前工作采用 transformer 结构不同，SC-VAE 是一个**全稀疏卷积网络**，在高分辨率下计算高效，且跨尺度泛化好。它沿用 U 形 VAE 设计：编码器通过多个残差块分层下采样稀疏体素特征，解码器镜像重建。

![](assets/TRELLIS.2%20-%20原生几何与材质的紧凑三维潜变量生成/03-scvae-architecture.png)

> 图 3（论文 Figure 4）：SC-VAE 的网络结构。这是一张完整的方法架构图：稀疏残差自编码层（residual autoencoding）、early-pruning 上采样器、优化的残差块共同构成了 16× 空间压缩的关键。  
> 来源：论文 Figure 4，https://arxiv.org/abs/2512.14692

SC-VAE 的三个关键设计：

1. **稀疏残差自编码层（Sparse Residual Autoencoding Layer）**：把 DC-AE 的 residual autoencoding 思想迁移到稀疏体素上，在上下采样块中引入**非参数残差捷径**，通过在稀疏网格的空间/通道维度之间搬移信息，缓解高压缩率下的优化困难。对 2× 下采样，把每个体素的 8 个子体素聚合到通道维：

$$
F_{\text{coarse}}^{\text{raw}}=\operatorname{stack}\big(F_{\text{child}_1},\dots,F_{\text{child}_8}\big)\in\mathbb{R}^{8C},\quad
F_{\text{coarse}}=\operatorname{avg\_groups}\big(F_{\text{coarse}}^{\text{raw}}\big)\in\mathbb{R}^{C'},
$$

缺失的体素因稀疏性贡献零向量。上采样则对称地用 channel-to-space 捷径把粗特征分发回邻域（$\operatorname{unstack}$ / $\operatorname{dup\_groups}$）。这条捷径被消融实验证明是**高压缩率下保持保真度的关键**（见 §5.4）。

2. **Early-pruning 上采样器**：每次上采样前预测二值掩码 $\hat{\boldsymbol{\rho}}\in\{0,1\}^{8}$，标记父节点的 active 子体素，跳过 inactive 节点，大幅降低运行时与显存成本。

3. **优化的残差块（Optimized Residual Block）**：稀疏卷积在高稀疏度数据上有效计算效率低，因此把标准"双卷积"残差块改成 ConvNeXt 风格的"单卷积 + 宽逐点 MLP"（MLP 类比 Transformer FFN，扩展通道提升非线性与表达能力）。该改动**不损失效率但提升重建质量**。

**解耦的潜空间**：为支持"先生成形状、再生成材质"的顺序生成，论文训练了**两个 SC-VAE**——一个建模形状，另一个在上采样阶段以形状 VAE 的细分结构为条件建模材质。

### 3.4 生成建模：三阶段 flow-matching

生成阶段沿用 TRELLIS 的总体设计，采用 **DiT 架构 + flow matching** 范式，并按新潜空间做了扩展。完整生成过程由三个模型、三个阶段串联：

1. **sparse structure generation**：预测稀疏体素网格的 occupancy 布局（粗先验，建立全局稀疏结构）；
2. **geometry generation**：在 active 体素内生成几何 latent（资产几何骨架）；
3. **material generation**：合成与几何结构对齐的材质 latent。

前两阶段大体沿用 TRELLIS 策略，构成资产的几何主干；**material generation 是全新阶段**，直接在原生 3D 空间中建模 PBR 材质——一个 sparse DiT 同时以输入图像和已生成的几何 latent 为条件预测材质 latent。这一设计把几何与材质生成统一到**同一个原生 3D 潜域**，保证任意拓扑下两者的空间对齐。

**架构与训练细节**：所有 DiT 采用 AdaLN-single 调制与 RoPE 位置编码（利于可扩展性与跨分辨率泛化）；图像条件特征用 **DINOv3-L** 提取。得益于 SC-VAE 的高压缩率，sparse DiT 得以**丢弃 TRELLIS 中的卷积打包与跳跃连接设计**，退化为 vanilla-style DiT，降低复杂度并提升效率。训练采用渐进策略：先用 512×512 条件图像学粗占用先验，再逐步提升空间与视觉分辨率——几何与材质生成器从 512³ 输出（32³ latent）扩展到 1024³（64³ latent），条件图像分辨率同步升到 1024。

### 3.5 训练目标与损失函数

**SC-VAE 两阶段训练**。第一阶段用低分辨率数据以直接重建损失 + KL 损失快速稳定学习：

$$
\begin{aligned}
\mathcal{L}_{\text{s1}}&=\lambda_{\text{v}}\|\hat{\boldsymbol{v}}-\boldsymbol{v}\|_{2}^{2}
+\lambda_{\delta}\,\mathrm{BCE}(\hat{\boldsymbol{\delta}},\boldsymbol{\delta})
+\lambda_{\boldsymbol{\rho}}\,\mathrm{BCE}(\hat{\boldsymbol{\rho}},\boldsymbol{\rho})\\
&+\lambda_{\text{mat}}\|\hat{\boldsymbol{f}}^{\text{mat}}-\boldsymbol{f}^{\text{mat}}\|_{1}
+\lambda_{\text{KL}}\,\mathcal{L}_{\text{KL}}.
\end{aligned}
$$

各分量的含义：对偶顶点位置 $\boldsymbol{v}$ 用 **MSE**，对偶面标志 $\boldsymbol{\delta}$ 用 **BCE**，材质属性 $\boldsymbol{f}^{\text{mat}}$ 用 **L1**，剪枝掩码 $\boldsymbol{\rho}$ 用 **BCE**，再加标准 KL 项。第二阶段在高分辨率下加入**渲染感知监督**：随机布设相机、用浅近平面"切过表面"，渲染 mask / depth / normal 图并施加 L1 损失，normals 上再叠加 SSIM 与 LPIPS 项；材质属性也经渲染后用同样的感知损失监督。总损失为 $\mathcal{L}_{\text{s2}}=\mathcal{L}_{\text{s1}}+\mathcal{L}_{\text{render}}$。相机切过表面的设计刻意鼓励模型同时捕捉外部与内部结构。

**生成模型的 flow-matching 目标**。生成模型采用 **rectified flow** 表述，前向过程是线性插值 $\boldsymbol{x}(t)=(1-t)\boldsymbol{x}_{0}+t\boldsymbol{\epsilon}$（$t\in[0,1]$，从数据样本 $\boldsymbol{x}_0$ 直线过渡到噪声 $\boldsymbol{\epsilon}$），反向过程由时变向量场 $\boldsymbol{v}(\boldsymbol{x},t)$ 引导。神经网络 $\boldsymbol{v}_{\theta}$ 通过最小化 **Conditional Flow Matching（CFM）** 目标训练：

$$
\mathcal{L}_{\text{CFM}}(\theta)=\mathbb{E}_{t,\boldsymbol{x}_{0},\boldsymbol{\epsilon}}\left\|\boldsymbol{v}_{\theta}(\boldsymbol{x}(t),t)-(\boldsymbol{\epsilon}-\boldsymbol{x}_{0})\right\|^{2}_{2}.
$$

直觉上，网络学习的是"从噪声直回数据"的速度方向 $\boldsymbol{\epsilon}-\boldsymbol{x}_{0}$。时间步采样沿用 TRELLIS 的 $\mathrm{logitNorm}(1,1)$ 分布以提升生成质量。训练用 AdamW（lr $1\times10^{-4}$，weight decay 0.01），classifier-free guidance 的 drop rate 0.1。

### 3.6 推理流程与复杂度

推理的关键效率优势来自 16× 压缩带来的 token 锐减。论文报告的 H100 端到端生成时延：512³ 全纹理资产约 **3s**、1024³ 约 **17s**、1536³ 约 **60s**。作为对照，仅 O-Voxel 的解码器（latent→网格+材质）在 512³ 时约 0.077s、1024³ 时约 0.301s（A100）。此外论文还提供了一个**级联式测试时缩放**机制（见 §5.5），可在训练分辨率之上或之内灵活换取更高细节或更干净布局。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

- **SC-VAE 训练数据**：采用 TRELLIS-500K 配置，过滤掉不含 PBR 材质的资产，得到来自 Objaverse-XL、ABO、HSSD 的精选集合。
- **生成模型训练数据**：约 **800K** 资产，额外用 **TexVerse** 增强 PBR 的多样性与真实感。图像 prompt 用 Blender 每资产渲染 16 视角，随机化 FoV 与光照。
- **重建评测集**：Toys4K 基准 + 一个精选测试集（含 90 个近两年 Sketchfab 复杂 PBR 资产），两者训练时均未见过。
- **生成评测集**：100 张 AI 生成图像 prompt（训练-测试不相交），用于质量对比与用户研究。

### 4.2 Baseline 与评价指标

- **重建 baseline**：Dora（Shape2Vecset）、TRELLIS、Direct3D-S2、SparseFlex。
- **生成 baseline**：TRELLIS、Hi3DGen、Direct3D-S2、Step1X-3D、Hunyuan3D 2.1。
- **纹理生成 baseline**：Hunyuan3D-Paint（多视图 PBR 生成融合）、TEXGen（UV 法）。
- **评价指标**：① Mesh Distance（MD，双向点到网格距离 + F1 分数，衡量含内部结构的网格重建保真度）；② Chamfer Distance（CD，在可见表面采样点云上计算 + F1，只看外部形状）；③ 渲染法线图的 PSNR / LPIPS（表面质量）；④ CLIP / CLIP-N（视觉对齐）、ULIP-2 / Uni3D（几何相似度）；⑤ 用户研究偏好率 Pref% / Pref-N%。PBR 渲染用 nvdiffrec 的 split-sum renderer，runtime 在 A100 上报。

### 4.3 实现细节

- SC-VAE 用 **Triton 优化的 Submanifold 卷积**训练，16 H100，batch size 128。
- 每个 DiT 约 **1.3B 参数**（width 1536、blocks 30、heads 12、MLP width 8192），32 H100，batch size 256；三个 DiT 合计约 **4B** 参数。
- 论文还自研了 **FlexGEMM** 稀疏卷积后端（Triton 实现，跨 NVIDIA/AMD），采用 Masked Implicit GEMM、Gray code 重排与 Split-K，基准测试比主流稀疏卷积库最高快 **2×**。

---

## 5. 实验结果

### 5.1 主要定量结果：重建

表 1 对比了形状重建的效率与保真度。核心结论是：**Ours 在全部指标上大幅领先所有 baseline，同时 token 数更少、解码更快**。代表数字如下（MD/CD 报告为 $\times 10^{6}$，runtime 在 A100）：

| 方法 | #Token | 压缩率 | Dec.(s)↓ | Toys4K MD↓ | Toys4K F1↑ | Toys4K PSNR↑ | Sketchfab MD↓ |
|---|---|---|---|---|---|---|---|
| Dora | 2.0K–4.1K | – | 37.7 | 366.1 | 0.019 | 27.26 | 987.2 |
| Trellis | 9.6K | 4× | 0.108 | 85.07 | 0.074 | 30.29 | 49.20 |
| Direct3D-S2 1024 | 17K | 8× | 13.0 | 73.17 | 0.001 | 27.38 | 70.13 |
| SparseFlex 1024 | 225K | 4× | 3.21 | 0.3132 | – | 37.34 | 0.7593 |
| **Ours 512** | **2.2K** | **16×** | **0.077** | 0.0323 | 0.888 | 39.54 | 0.1572 |
| **Ours 1024** | **9.6K** | **16×** | **0.301** | **0.0042** | **0.971** | **43.11** | **0.0170** |

> 表 1（论文 Table 1）：形状重建效率与保真度对比。**这段数据说明了什么**：① Ours 在 1024³ 只用 9.6K token（与 Trellis 相同量级但压缩率高 4 倍），MD 却比 SparseFlex（225K token）还低两个数量级（0.0042 vs 0.3132），说明**紧凑潜空间并未牺牲保真度，反而更准**；② Dora 这种非结构化潜变量虽然 token 少，但 MD 高达 366.1，重建保真度差；③ Ours 的解码时间（0.3s）比 Direct3D-S2 1024（13s）快 40 倍以上。这三点共同支撑了论文"native + compact"的核心主张。（个人判断：SparseFlex 用 225K token 换几何精度，而 Ours 用更少 token 拿到更高精度，说明**表示设计**而非 token 数量才是精度瓶颈。）

**材质重建**：由于没有合适的"仅给定形状编码材质"的 baseline，材质指标只报 Ours 自身——PBR 属性图 **38.89 dB / 0.033**（PSNR/LPIPS），着色图 **38.69 dB / 0.026**，说明材质复现保真、且与几何对齐一致。

### 5.2 主要定量结果：生成

表 2 对比 image-to-3D 生成质量，Ours 在**所有对齐指标上取得最高分**，用户研究更是压倒性领先：

| 方法 | CLIP↑ | CLIP-N↑ | ULIP-2↑ | Uni3D↑ | Pref%↑ | Pref-N%↑ |
|---|---|---|---|---|---|---|
| Trellis | 0.876 | 0.748 | 0.470 | 0.414 | 6.40% | 2.82% |
| Hunyuan3D 2.1 | 0.869 | 0.753 | 0.474 | 0.427 | 13.3% | 7.51% |
| Step1X-3D | 0.875 | 0.738 | 0.464 | 0.411 | 11.8% | 0.469% |
| **Ours** | **0.894** | **0.758** | **0.477** | **0.436** | **66.5%** | **69.0%** |

> 表 2（论文 Table 2）：image-to-3D 生成对比（-N 表示用 normal map 度量）。**这段数据说明了什么**：① CLIP/ULIP-2/Uni3D 这类自动指标上 Ours 只小幅领先，但这些指标饱和度高、区分度有限；② 真正拉开差距的是**用户研究**——约 40 名参与者在 100 个 prompt 上无筛选投票，Ours 偏好率 **66.5%**（次优 Hunyuan3D 2.1 仅 13.3%），接近 5 倍差距，说明**感知层面的几何细节、材质真实感与 prompt 对齐明显更优**。（个人判断：自动指标与人类偏好的大幅背离，提示 3D 生成评测仍严重依赖用户研究，现有 CLIP 系指标对 PBR 材质与几何细节的区分力不足。）

### 5.3 定性结果

定性上，Ours 生成的资产展现了此前方法难做到的能力：**薄结构、开放曲面、半透明区域**——如精细齿轮、封闭驾驶舱、开放叶片与花朵、玻璃与金属等反射/半透明材质，在新型光照下阴影物理一致。对比图（Figure 6）显示，TRELLIS / Direct3D-S2 / Hi3DGen 等 baseline 要么不含 PBR 材质、要么几何细节被过度平滑；Hunyuan3D 2.1 虽有 PBR，但纹理对齐与真实感弱于 Ours。

![](assets/TRELLIS.2%20-%20原生几何与材质的紧凑三维潜变量生成/04-qualitative-comparison.png)

> 图 4（论文 Figure 6）：视觉对比（主图为 normal 图，小图为最终渲染、base color、metallic、roughness）。这张图直观展示了 Ours 在几何细节（法线图更锐利）与材质（PBR 分解图更合理）上的双重优势。  
> 来源：论文 Figure 6，https://arxiv.org/abs/2512.14692

**Shape-Conditioned Texture Generation**（§4.3）：生成流水线的第三阶段可独立作为"给定 mesh + 参考图 → PBR 纹理"模型使用。多视图方法（Hunyuan3D-Paint）常出现形状-图像不一致与跨视图 ghosting/模糊，UV 法（TEXGen）受 UV chart 歧义与 seam 伪影困扰；Ours 在**原生 3D 中做外观推理**，纹理更锐利、形状-材质对齐更一致，且能合成**内部表面的纹理**（这对有遮挡或非流形几何的资产至关重要）。

### 5.4 消融实验

消融在 256³ 的精选 Sketchfab 资产上进行，验证 SC-VAE 架构设计（表 3）：

| 设置 | #Token | Dec.(ms)↓ | MD↓ | F1↑ | PSNR↑ | LPIPS↓ |
|---|---|---|---|---|---|---|
| SC-VAE f16c32（完整） | 503 | 28.6 | 1.032 | 0.312 | 27.26 | 0.072 |
| w/o Residual AE | 503 | 28.7 | 1.747 | 0.268 | 26.73 | 0.081 |
| w/o Opt. ResBlock | 503 | 29.6 | 1.198 | 0.285 | 26.67 | 0.083 |
| SC-VAE f32c128（完整） | 118 | 33.9 | 1.405 | 0.273 | 26.65 | 0.081 |
| w/o Residual AE（f32c128） | 118 | 34.0 | 7.394 | 0.192 | 25.01 | 0.102 |

> 表 3（论文 Table 3）：SC-VAE 架构设计消融。**说明了什么**：① 移除稀疏残差自编码层后，16× 压缩时 MD 恶化 69%、PSNR 掉 0.5dB，32× 压缩时 MD 恶化 526%、PSNR 掉 1.6dB——**压缩率越高，残差捷径越关键**，这是它在强空间瓶颈下维持保真度的直接证据；② 移除优化残差块后 MD 恶化 16%、PSNR 掉 0.6dB，而 runtime 不变——证明"单卷积 + 宽 MLP"的混合设计在不牺牲效率的前提下更好地保留细节。

### 5.5 泛化、效率与失败案例

**测试时计算与分辨率缩放**（§4.5）是本文一个实用的工程贡献。得益于 token 数量少，可以级联复用第二阶段生成器：例如把生成的 1024³ O-Voxel max-pool 成 96³ sparse structure，再重跑 geometry generation 得到 1536³ 形状——**在训练分辨率之上**合成更高分辨率。同分辨率内也能"先粗后精"：把 512³ O-Voxel 降采样为 64³ sparse structure，再生成 1024³，修正局部错误、得到更干净布局。

![](assets/TRELLIS.2%20-%20原生几何与材质的紧凑三维潜变量生成/05-test-time-scaling.png)

> 图 5（论文 Figure 8）：测试时分辨率与计算量缩放（左）与级联推理（右）。级联推理能产出更细细节与更稳结构，形成"计算量 ↔ 质量"的可控权衡。  
> 来源：论文 Figure 8，https://arxiv.org/abs/2512.14692

**失败案例与局限**（来自附录 F，详见 §7）：小于体素尺寸的细节会产生 aliasing（两平行面同入一体素时 QEF 取中间点、材质被平均）；重建/生成结果偶有小孔洞；表示不含高层结构/语义信息。

---

## 6. 与相关工作的关系

论文在相关工作部分梳理了三条脉络，本文的定位是"第三条脉络里的一次表示层重构"：

- **3D 表示（occupancy/SDF/NeRF/mesh/点云/Gaussian/SLAT）**：早期隐式场与 NeRF 要么几何质量低、要么采样成本高；非结构化表示（mesh/点云/3DGS）显式但缺结构规整性，难做潜压缩。近期 TRELLIS、SparseFlex 等用"场式等值面 + 稀疏体素"达到高分辨率，但受场基元限制，处理不了开放/非流形表面，也不处理材质。**O-Voxel 是这条线上第一个"去场化"的稀疏体素**。
- **潜 3D 表示**：非结构化潜变量（Perceiver 风格，如 Dora、Shape2Vecset 系）压缩强但重建保真受限；结构化潜变量（SLAT 及其扩展）几何准但 token 多。本文通过原生输入 + 16× 压缩，同时拿到"紧凑"与"保真"。
- **大规模 3D 资产生成系统**：主流范式是"形状生成 + 多视图纹理合成"两段式（借力 2D 扩散主干），但需要复杂的多视图渲染、烘焙与纹理对齐，易引入外观不一致。TRELLIS 虽做了材质，但仍依赖多视图烘焙。**本文是端到端原生生成**，直接产出高保真全纹理资产，无视图相关后处理。

一句话概括：TRELLIS.2 是 TRELLIS 的直接后继，但把 TRELLIS"从多视图 2D 信息建潜空间 + 场式几何"的两处依赖，替换为"原生 3D O-Voxel + 无场对偶网格"。

---

## 7. 局限与批判性评价

**论文自述的局限**（附录 F，明确结论）：

1. **分辨率受限的 aliasing**：与其他体素方法一样，O-Voxel 表达能力受空间分辨率约束。小于体素尺寸的几何细节会产生 aliasing——例如两个很近的平行面落入同一体素时，QEF 会取"最小化到两面误差"的中间点而非准确落在某一面上，该体素的体素材质也会是两面属性的平均，导致外观模糊。
2. **偶发小孔洞**：重建与生成结果有时出现小孔洞（大多可用标准 hole filling 后处理修复），根因是稀疏解码器难以保证从预测的高分辨率稀疏结构得到完美封闭流形表面。
3. **缺高层结构/语义信息**：O-Voxel 目前只编码几何与材质，未显式编码部件级分割或图拓扑结构，未来可扩展以解锁更多下游应用。

**批判性评价**（个人判断，区别于论文结论）：

- **这是一次"表示驱动"的范式胜利**。论文最有力的证据不是某个生成 trick，而是同一套方法在重建（表 1）与生成（表 2）两端同时刷新 SOTA——通常重建强不等于生成强，这里两者兼得，说明 O-Voxel + 16× 压缩确实是一个更优的"资产数字化底座"。
- **评测的隐忧**：生成部分的自动指标（CLIP/ULIP/Uni3D）区分度很低，结论高度依赖 40 人用户研究。样本量偏小、且未披露参与者背景（是否美术从业者），用户研究的设计细节放在附录（D.2），正文只引结果。对"工业可用"这一主张而言，还缺真实下游任务（如游戏引擎导入、重新打光、编辑）的量化验证。
- **"紧凑"是有代价的**：16× 压缩让 1024³ 资产只剩 9.6K token，意味着单个 latent token 要承载约 $64^{3}$ 体素的信息——一旦几何或材质有小于 token 尺度的变化，理论上就无法表达。论文在附录坦诚了这一点（aliasing），但正文对"分辨率上限"的讨论偏少。
- **未涉及的部分**：拓扑清洁度、UV 展开、与主流 DCC/引擎格式的兼容性、以及骨骼绑定角色的表现，论文均未展开，工程落地仍需外部验证。

---

## 8. 复现与实践建议

**复现成本评估**（基于论文披露）：

- **训练成本高**：SC-VAE 需 16×H100；生成模型约 4B 参数、32×H100，三阶段渐进训练。个人/小团队几乎无法从零复现完整生成模型。
- **推理门槛相对友好**：H100 上 512³ 约 3s、1024³ 约 17s；论文承诺开源模型与权重，若兑现，单卡 H100/A100 即可跑 image-to-3D 推理。
- **代码与后端**：自研 FlexGEMM 是 Triton 实现、跨 NVIDIA/AMD，工程上对复现者比 CUDA 专用库更友好；但这也意味着要自己搭 Triton 环境。

**实践建议**：

1. **先复用权重，不必从零训练**：关注项目页/官方仓库的预训练权重与推理脚本，先验证 O-Voxel 与 SC-VAE 的编解码质量（这本身就是最可迁移的组件——O-Voxel↔mesh 转换与 SC-VAE 可作为独立"3D 资产压缩器"用于其他任务）。
2. **优先评估 SC-VAE 作为编码器**：16× 压缩 + 9.6K token + 近无损重建，对任何需要"紧凑 3D 表示"的下游（检索、条件生成、world model 状态表示）都有直接价值，且比整条生成流水线轻量得多。
3. **注意评测陷阱**：复现对比时不要只看 CLIP 系指标（区分度低），应补用户研究或下游任务指标；重建部分 MD/CD 的归一化方式需对照附录 D.1 的评测协议。
4. **后处理预期**：论文自己承认偶发小孔洞，接入生产前应预留 hole filling / mesh cleanup 环节。

---

## 9. 个人启发与后续问题

**启发**：

1. **"表示即上限"的再印证**。这篇工作的增量主要不在生成模型（DiT + flow matching 都是成熟件），而在 O-Voxel 这一表示层。它提示：当生成模型卷到 4B 量级后，继续堆参数收益递减，而**换一个更贴合的表示**能带来数量级的精度与效率双重提升。
2. **"原生 vs 多视图烘焙"的分野**。TRELLIS.2 与 Hunyuan3D 系代表了 3D 资产生成的两条路线——前者在原生 3D 潜空间端到端生成，后者借 2D 扩散主干 + 多视图烘焙。本文用"几何+材质统一对齐 + 内部结构可生成"论证了原生路线的优势，但也应看到 2D 路线的数据/模型生态更成熟（个人判断）。
3. **紧凑潜空间解锁测试时缩放**。16× 压缩不仅省钱，还让"级联重采样"这种 test-time scaling 成为可能——这为后续"生成质量随算力平滑增长"提供了新范式。

**后续问题**：

1. 若把 O-Voxel 的 $\alpha$（opacity）与真实半透明渲染（折射、SSS）结合，能否真正支撑透明材质的工业级生成？
2. SC-VAE 的 16× 压缩是否是"最优压缩率"？更高压缩（如 32×）下 aliasing 与保真度的 trade-off 是否可量化？（消融已给 32× 残差捷径的关键性，但未给完整 32× 生成端对比）
3. 论文提出的"部件级分割 + 图拓扑"扩展，与 CubePart、PhysX-Omni 等"部件可控/仿真就绪"路线如何合流？这是把 TRELLIS.2 推向"可用资产"的关键一步。
4. 原生 3D 潜空间能否反向用于理解任务（如 VLM 的 3D 输入编码），与 VLM³、ShapeLLM-Omni 等"原生 3D MLLM"形成互补？

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `overview_v5.png` | 方法总览图 | https://arxiv.org/html/2512.14692 | Figure 2 | 已有（保留） | §3.1 总体框架 |
| `representation.png` | O-Voxel 表示图 | https://arxiv.org/html/2512.14692 | Figure 3 | 已有（保留） | §3.2 O-Voxel |
| `03-scvae-architecture.png` | SC-VAE 架构图 | https://arxiv.org/html/2512.14692 | Figure 4 | 2026-08-21 | §3.3 SC-VAE |
| `04-qualitative-comparison.png` | 定性对比图 | https://arxiv.org/html/2512.14692 | Figure 6 | 2026-08-21 | §5.2 定性结果 |
| `05-test-time-scaling.png` | 测试时缩放图 | https://arxiv.org/html/2512.14692 | Figure 8 | 2026-08-21 | §5.4 效率/缩放 |

- 论文原文（HTML 全文）：https://arxiv.org/html/2512.14692 （本次精读主来源，含附录 A–F）
- arXiv 摘要页：https://arxiv.org/abs/2512.14692
- 项目页：https://microsoft.github.io/TRELLIS.2/
- 补充材料（Supplementary）：论文 HTML 版本内附录 A（实现细节）、B（FlexGEMM）、C（数据准备）、D（评测协议与用户研究）、E（更多结果）、F（局限）已随 HTML 一并阅读，未单独下载 PDF/Supplementary。

> 说明：本文档全部事实性内容均基于论文原文（arXiv HTML 版 v1）；凡个人推断已用"（个人判断）"标注。论文未披露的信息（如代码仓库确切地址、用户研究参与者背景细节）在正文中以"未披露"处理，不补造事实。
