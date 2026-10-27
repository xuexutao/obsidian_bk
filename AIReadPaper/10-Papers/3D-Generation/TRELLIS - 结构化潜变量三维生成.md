本版基于官方原版论文 **Structured 3D Latents for Scalable and Versatile 3D Generation** 覆盖重写，不再沿用早期公众号/README 层面的简化整理。重点补足 SLAT 表示、两阶段生成、Rectified Flow Transformer、编辑能力，以及论文中的实验细节与对比结论。

## 一句话结论

TRELLIS 针对 mesh、NeRF、3DGS 各自绑定专用模型、难以同时兼顾几何、外观、多格式输出和局部编辑的问题，提出 **SLAT**：在稀疏 active voxels 上绑定高容量局部 latent vectors，并用两阶段 Rectified Flow Transformer 先生成空间结构、再只在活跃区域生成细节，最后由多个 decoder 将同一潜空间解码为 mesh、3D Gaussian 或 radiance field。约 50 万高质量资产和最高 20 亿参数的规模化训练使其在文生/图生 3D、重建质量和用户偏好上整体领先既有方法，并自然支持资产变体与局部重采样；其真正优势是把 SLAT 建成统一 3D foundation representation，而不是某个单一格式的生成器，但 PBR 解耦、透明/细薄结构和不同输出格式的生产可用性仍有限。

## 1. 背景

TRELLIS 试图解决的核心问题，是 3D 生成长期缺乏一个**既能统一几何与外观、又能灵活解码到多种 3D 表示形式**的中间潜空间。

在 2D 生成领域，latent representation 已经成为非常自然的基础设施；但在 3D 领域，表示形式高度割裂：

- **Mesh** 适合显式几何与编辑，但难以直接承载细致外观；
- **Radiance Field / NeRF** 擅长视图合成，但显式表面提取和资产导出不够直接；
- **3D Gaussian Splatting** 在渲染质量和效率上很强，但并不是天然的编辑友好资产格式。

这意味着，很多既有方法本质上只能在某一种 3D 表示内优化，难以同时兼顾：

1. 高质量几何；
2. 细致外观；
3. 多格式输出；
4. 局部可编辑性；
5. 可扩展的大模型训练。

TRELLIS 的动机因此非常明确：**先学习一种结构化统一 3D latent，再按需要解码成不同表示。** 这使它不仅是一个 text/image-to-3D 模型，更是一个朝“3D foundation representation”方向推进的工作。

下图是论文 Figure 4：TRELLIS 生成的高质量 3D 资产示例（Gaussian 与 Mesh 两种表示，来自文生/图生提示）。

![](assets/TRELLIS%20-%20结构化潜变量三维生成/results_v2.png)

### 1.1 正式论文信息

- **论文标题：** [Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506)
- **项目名：** TRELLIS
- **发表：** CVPR 2025
- **作者/机构：** 微软研究院、清华大学、中国科大等
- **项目页：** [TRELLIS Project Page](https://trellis3d.github.io/)

### 1.2 为什么值得看

1. 它提出的 **SLAT（Structured LATent）** 是论文真正的核心，不只是一个“模型名字”。
2. 它把生成目标从“固定一种 3D 表示”变成“先学统一潜空间，再决定输出格式”。
3. 它不仅支持 text-to-3D / image-to-3D，还支持 asset variants 和局部编辑，说明作者把它当成可持续操作的资产工作流来设计。

---

## 2. 文章主线 / 论文线索

### 2.1 论文试图回答什么问题

论文的核心问题可以表述为：

> 能否构造一种统一的 3D 潜表示，使它既保留高分辨率 3D 结构，又能承载足够细的外观信息，并且能被大规模生成模型稳定学习，最终灵活解码到 mesh、3DGS、radiance field 等不同格式？

作者的回答是 **SLAT**。

### 2.2 SLAT 的基本思想

SLAT 不是一个纯稠密体素特征网格，也不是一个纯隐式全局 latent。它由两部分组成：

1. **结构层（sparse structural scaffold）** 用稀疏激活体素描述 3D 空间里哪些位置属于物体表面附近，从而提供高分辨率但高效的 3D 结构框架。
2. **外观层（dense local latent features）** 在每个激活体素上附加局部 latent feature，这些特征来自多视图视觉信号聚合，承载细节纹理、局部几何与外观信息。

因此，SLAT 本质上是一个**结构化稀疏 3D latent field**：

- 既有显式空间位置；
- 又有附着其上的局部高维潜向量；
- 既保留结构，又不丢细节。

### 2.3 与传统路线相比的新意

相较只在某种目标表示上建模的方法，TRELLIS 的根本新意在于：

- 先学习一个中间层，而不是一开始就押注最终输出格式；
- 用统一 latent 连接 text/image 条件、结构生成、外观细化和下游解码；
- 把 3D 生成从“针对单格式设计模型”推进到“针对统一 3D latent 设计模型”。

**一句话理解：** TRELLIS 把 3D 资产生成拆成“先生成一个结构化 3D 潜空间，再把它解码成不同资产格式”。

---

## 3. Pipeline / Architecture + I/O 数据流

### 3.1 总体流程总览

下图是论文 Figure 2：方法总览，展示了 SLAT 编码（稀疏 3D 网格上的局部 latent）与两阶段生成（稀疏结构 + 局部 latent）以及多 decoder 解码的完整流程。

![](assets/TRELLIS%20-%20结构化潜变量三维生成/pipeline_v3.png)

TRELLIS 采用明确的**两阶段生成流程**：

1. 先生成稀疏结构；
2. 再在激活区域上生成局部 latent feature；
3. 最后把统一 latent 解码为不同 3D 表示。

|阶段|输入|输出|
|---|---|---|
|条件编码|文本 prompt 或单张图像|条件特征（文本特征 / 图像特征）|
|结构生成|条件特征|稀疏 3D 结构，占据哪些 active voxels|
|局部 latent 生成|条件特征 + active voxels|每个激活体素对应的局部 latent vectors|
|统一表示|结构 + 局部 latent|SLAT|
|解码输出|SLAT|Radiance Field / 3D Gaussian / Mesh / GLB|

### 3.2 SLAT 的具体构成

论文把 SLAT 记为结构化的 latent 集合，每个 active voxel 附带一个局部潜向量。其关键设计思想是：

- **空间位置是稀疏的**：只保留物体表面附近需要建模的区域；
- **局部特征是稠密的**：在这些位置上保留高容量特征，用于承载细节；
- **位置与特征绑定**：因此既不像全局 latent 那样过度压缩，也不像整块 dense voxel grid 那样昂贵。

作者默认使用较高分辨率的 3D 网格，但真正被显式建模的是其中激活的小部分，从而兼顾了：

- 高空间分辨率；
- 稀疏结构计算效率；
- 局部高频细节表达能力。

### 3.3 两阶段生成：为什么先结构、后细节

#### 第一阶段：生成结构

第一阶段的目标不是直接生成最终外观，而是先回答：

- 物体大概占据哪些 3D 区域？
- 哪些 voxels 是“非空 / 活跃”的？
- 粗几何骨架是什么？

为了让模型更稳定地学习稀疏结构，论文先把结构表示压缩到更低分辨率的连续表征，再进行生成，之后再恢复到目标分辨率。这样做的好处是：

- 更容易捕捉全局形状；
- 降低结构生成的复杂度；
- 为第二阶段提供明确的 active region。

#### 第二阶段：生成局部 latent vectors

有了结构以后，第二阶段只在 active voxels 上继续工作。此时模型不需要再关心整个 3D 空间，只需为每个激活位置生成局部 feature。这样做非常关键，因为它相当于把“细节建模”限制在真正重要的位置上，避免把大量容量浪费在空空间。

这些局部 latent 随后可被不同 decoder 消费，因此它既服务于几何，也服务于外观与渲染。

### 3.4 Rectified Flow Transformer 如何工作

TRELLIS 并没有使用最朴素的扩散 Transformer，而是采用 **Rectified Flow Transformer（RFT）** 作为骨干。Rectified Flow 的优势在于：

- 更适合大规模生成模型扩展；
- 在采样效率和稳定性上具备优势；
- 能自然兼容条件生成。

但 SLAT 与普通 2D latent 不同，它具有明显的**稀疏 3D 结构特性**。因此作者做了几项关键适配：

1. **适配稀疏结构的注意力/卷积设计** 使模型既能利用局部空间邻域，又不丢失长程结构依赖。
2. **窗口式/分层式局部交互** 增强局部 coherent 结构建模能力。
3. **归一化与训练稳定性处理** 使得 10 亿到 20 亿参数级别模型训练更稳定。

总的来说，RFT 在 TRELLIS 里扮演的角色不是“通用 backbone”，而是**为 structured sparse 3D latent 定制的大模型主干**。

### 3.5 不同 decoder：同一个 latent，多个输出

TRELLIS 的统一性真正体现在 decoder 端。

#### 3D Gaussian Decoder

- 将局部 latent 解码为一组高斯 primitives；
- 每个高斯包含位置、尺度、旋转、颜色、不透明度等参数；
- 适合高质量快速渲染与展示。

#### Radiance Field Decoder

- 将局部 latent 解码为局部辐射场表示；
- 适合视图合成与体渲染；
- 保留更连续的视角依赖外观表达。

#### Mesh Decoder

- 将 SLAT 解码为显式几何场，再进一步提取 mesh；
- 论文中借助可微表面提取路线，使网格输出能保持几何质量；
- 这是它能导出资产格式、靠近生产流程的关键一步。

因此，TRELLIS 的核心不是“同时支持多格式”这句功能描述本身，而是它真的建立了**同一个潜空间到多种 3D 表示的映射机制**。

### 3.6 编辑与变体能力的 I/O 逻辑

论文不只做从零生成，也强调资产工作流能力。

#### 变体生成（asset variation）

- **输入：** 已有生成结构 + 新的文本条件
- **机制：** 固定部分结构或共享第一阶段结果，仅改变第二阶段局部 latent 条件
- **输出：** 保持大结构一致、细节和外观发生变化的新资产

#### 局部编辑（region-specific editing）

- **输入：** 现有资产 + 指定编辑区域 + 条件信息
- **机制：** 仅重采样目标区域的结构或 latent
- **输出：** 局部修改后的 3D 资产

这说明作者把 TRELLIS 视为一个可以被“持续操作”的 3D latent system，而不是一次性出图模型。

---

## 4. 实验与关键信息

### 4.1 数据集与训练配置

论文使用大规模高质量 3D 资产集合，来源覆盖多个公开 3D 数据源，并对数据进行严格筛选、对齐和 caption 构建。关键点包括：

- 数据规模约 **50 万** 高质量 3D 资产；
- 使用高质量文字描述增强 text-to-3D 对齐；
- 构建多视图渲染与视觉特征，为 SLAT 提供跨视角细节监督；
- 模型家族覆盖从数亿到 **20 亿参数** 的规模。

这意味着 TRELLIS 不是“小规模 clever trick”，而是明确按**foundation model 路线**搭建的系统。

### 4.2 表示层面的实验结论

论文首先证明 **SLAT 本身就是强表示**。

与既有 3D latent / 3D representation baseline 相比，SLAT 在以下方面表现更好：

- 几何还原更准确；
- 外观更细致；
- 重建保真度更高；
- 解码到不同表示形式时质量更稳定。

这说明 TRELLIS 的成功不只是因为模型大，更因为它提出了一个更合理的 3D 中间表示。

### 4.3 生成结果与基线比较

下图是论文 Figure 5：TRELLIS 与先前方法的生成结果视觉对比（基于 AI 生成提示）。

![](assets/TRELLIS%20-%20结构化潜变量三维生成/comparisons_v3.png)

在 text-to-3D 与 image-to-3D 任务上，TRELLIS 对比多个强基线时总体体现出以下优势：

1. **几何更稳** 相比依赖 2D multi-view consistency 或单格式 3D 表示的方法，TRELLIS 输出的结构更完整，少出现几何破碎。
2. **外观更好** 局部 latent 加上视觉基础模型特征，使它在颜色、表面细节、边缘清晰度上更强。
3. **多格式一致性更强** 同一个 SLAT 可被稳定解码为 Gaussian、RF、Mesh，不需要为每种表示各训一套模型。
4. **用户偏好显著更高** 无论 text-to-3D 还是 image-to-3D，用户研究都显示 TRELLIS 胜率明显领先。

### 4.4 为什么 image-to-3D 特别强

论文和项目页都反复表明，TRELLIS 在 image-to-3D 上尤其突出。这背后的原因是：

- 图像条件能提供更强的外观和结构先验；
- SLAT 可以充分利用图像里的细节信号；
- 相比纯 text-to-3D，image-to-3D 的生成空间更收敛，因此统一 latent 的优势更容易兑现。

从工程实践看，这也解释了为什么官方更推荐把 TRELLIS 用作“图生 3D”或“文生图后再转 3D”的核心引擎。

### 4.5 论文体现出的限制

尽管结果很强，论文也揭示了几类边界：

1. **PBR / 光照解耦不充分** 输出结果仍可能携带输入图像中的 baked-in shading，材质与光照没有完全分离。
2. **真实世界复杂物体仍有难度** 尽管 image-to-3D 已很强，但对复杂透明、极细结构或强反射场景仍有挑战。
3. **统一 latent 不等于完全统一工作流** Mesh、Gaussian、Radiance Field 虽都能解码，但不同输出在下游编辑、渲染和资产制作上仍有各自边界。

**关键判断：** TRELLIS 最重要的贡献不是单次指标领先，而是用 SLAT 证明了“统一 3D latent”这条路是可行的，而且确实能带来多格式生成与编辑能力。

---

## 5. 个人评注 / 下一步

### 5.1 对当前技术视野的价值

TRELLIS 应归入 **3D生成**，但从方法论上看，它更像“3D foundation representation”的代表工作。相比许多只在某个 benchmark 上抬分的方法，它更值得长期跟踪，因为它试图重建的是**整个 3D 生成的中间层基础设施**。

### 5.2 与既有关注方向的关系

它和以下主线高度相关：

- 统一 latent 表示驱动 3D 生成；
- 多输出格式资产生成；
- 3D 资产编辑与可持续操作；
- 3D foundation model 的扩展路径。

### 5.3 建议下一步跟进

1. 继续对比 TRELLIS 与更紧凑 structured latent 路线，观察统一 latent 是否进一步压缩而不损失质量。
2. 重点跟踪其 mesh decoder 的真实可用性，评估是否已达到更高等级资产导出门槛。
3. 结合 SAM 3D、Hunyuan3D-PolyGen 等工作一起看，会更清楚近期 3D 主线正在分化为：**统一 latent、真实世界重建、生产级 mesh 资产化** 三条支线。
4. 如果后续做方向综述，TRELLIS 应作为“统一 3D latent 表示”主线的核心 anchor paper。

---

**建议写入主文档的一句话摘要：** TRELLIS 通过提出统一的结构化 3D 潜表示 SLAT，并采用两阶段 Rectified Flow Transformer 在该表示上生成和编辑 3D 资产，实现了从同一 latent 稳定解码到 mesh、3D Gaussian 和 radiance field 的能力，是统一 3D foundation representation 路线上的代表性工作。

---

## 6. 代码级细节：从资产编码到两阶段 Flow 生成

这一章专门把论文中的概念图落实到官方代码、文件格式和具体张量形状。理解 TRELLIS 时最容易混淆的地方，是把“对已有 3D 资产编码以制作训练数据”和“给定一张图片真正执行生成推理”看成同一个过程。两者共享同一套潜表示，但数据流并不相同：前者从真实资产构造两个 Flow Model 的监督目标，后者则从噪声依次生成这两个目标。

### 6.1 先区分两个上下文：训练数据制作与生成推理

对于一个已有 3D 资产，离线数据处理会构造两种 latent：

1. **Sparse Structure latent**：把 $64^3$ 二值表面占用网格压缩为 $8\times16^3$ 的稠密连续 latent，作为第一阶段 Structure Flow 的真实数据分布。
2. **Structured Latent（SLat）**：在 $64^3$ 坐标系的每个 active voxel 上绑定一个 8 维局部 latent，形成 $\{(\boldsymbol p_i,\boldsymbol z_i)\}_{i=1}^{N}$，作为第二阶段 SLat Flow 的真实数据分布。

因此，同一份 $64^3$ 表面体素结构有两个去处：一条支路只编码“哪里有表面”，另一条支路把多视角 DINOv2 特征附着到这些表面体素上，再编码局部几何与外观。两条支路不是串联关系，而是共享同一组 active voxels 的两个并行训练目标。

真正推理时并不存在一个可供渲染 150 个视角的真实 3D 资产。模型先用第一个 Flow 生成 Sparse Structure latent，经 SS VAE decoder 恢复 active voxel coordinates；随后第二个 Flow 固定这些坐标并生成每个位置的 8 维局部 latent，最终构成完整 SLat。

### 6.2 资产归一化与 $64^3$ 表面体素化

Blender 预处理首先把资产包围盒中心移动到原点，再按最长边做统一缩放，使最长边长度为 1。更准确地说，资产整体落入 $[-0.5,0.5]^3$，但只有最长轴接近填满该区间，另外两轴保持原始长宽比，并不是三个坐标轴分别独立拉伸到 $[-0.5,0.5]$。

`dataset_toolkits/voxelize.py` 使用 Open3D 在固定边界 $[-0.5,0.5]^3$ 内建立边长为 $1/64$ 的体素网格，只保留与三角形表面相交的 active voxels。若某个体素的整数网格索引为

$$\boldsymbol g_i=(g_i^x,g_i^y,g_i^z),\qquad g_i^\cdot\in\{0,\ldots,63\},$$

则写入 `voxels/<sha256>.ply` 的并不是任意表面采样点，而是该体素的格心坐标：

$$\boldsymbol p_i=\frac{\boldsymbol g_i+0.5}{64}-0.5.$$

因此可以无损地从格心坐标恢复整数索引：

$$\boldsymbol g_i=\operatorname{int}\left((\boldsymbol p_i+0.5)\times64\right).$$

这里的体素表示是**表面占用**而不是实体内部全部填充。PLY 文件只保存 $N$ 个 active voxel centers；需要送入 Sparse Structure VAE 时，`encode_ss_latent.py` 才把它们散射为稠密二值张量：

$$\boldsymbol X_{\mathrm{occ}}\in\{0,1\}^{1\times1\times64\times64\times64}.$$

### 6.3 第一条支路：Sparse Structure VAE 编码结构

`dataset_toolkits/encode_ss_latent.py` 默认加载：

```text
microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16
```

其配置位于 `configs/vae/ss_vae_conv3d_16l8_fp16.json`。编码器输入通道为 1，latent channel 为 8，主干通道依次为 `[32, 128, 512]`。每两个尺度之间使用一次 stride-2 的 $2\times2\times2$ 3D convolution，因此完整张量流是：

|阶段|张量形状|含义|
|---|---:|---|
|输入 occupancy|$1\times1\times64\times64\times64$|二值表面占用|
|输入卷积|$1\times32\times64\times64\times64$|初始局部特征|
|第一次下采样|$1\times128\times32\times32\times32$|空间尺寸减半|
|第二次下采样|$1\times512\times16\times16\times16$|累计下采样 4 倍|
|VAE 参数头|$1\times16\times16\times16\times16$|8 维 mean + 8 维 log-variance|
|确定性 latent|$1\times8\times16\times16\times16$|取 posterior mean|

代码调用 `encoder(ss, sample_posterior=False)`，所以数据预处理时不采样，而是直接取后验均值。保存到 NPZ 时又取掉 batch 维，因此文件只有一个 key：

```text
mean: float array, shape = [8, 16, 16, 16]
```

这正是第一个 Flow Model 学习的数据分布。图像条件版本的官方配置 `ss_flow_img_dit_L_16l8_fp16.json` 明确设置：

```text
resolution = 16
in_channels = 8
out_channels = 8
```

所以第一阶段 Flow 并不直接在 $64^3$ 二值网格上生成，而是在更平滑、更紧凑的 $8\times16^3$ 连续 VAE latent 上做 flow matching。推理完成后，SS VAE decoder 通过两次 2 倍上采样把它恢复成 $1\times64^3$ occupancy logits，再阈值化并提取 active coordinates，作为第二阶段的空间支架。

### 6.4 第二条支路的前半段：150 视角渲染与 DINOv2 特征

`dataset_toolkits/render.py` 默认渲染 150 个视角。相机位置通过球面 Hammersley sequence 采样，半径固定为 2，水平视场角固定为 $40^\circ$。Blender/Cycles 输出的是带透明背景的 $512\times512$ RGBA PNG，并在 `transforms.json` 中保存每个视角的水平 FOV 和 camera-to-world 矩阵。

`dataset_toolkits/extract_feature.py` 读取这些 RGBA 图像后执行以下处理：

1. 将图像从 $512\times512$ resize 到 $518\times518$；
2. 归一化到 $[0,1]$；
3. 计算 `RGB × Alpha`，即把透明背景合成为黑色；
4. 使用 ImageNet mean/std 标准化；
5. 输入 `dinov2_vitl14_reg`。

这里必须区分“渲染分辨率”和“DINOv2 输入分辨率”：**资产实际渲染为 $512^2$，送入 DINOv2 前才 resize 为 $518^2$。** 选择 518 是因为 ViT-L/14 的 patch size 为 14：

$$518/14=37.$$

DINOv2 ViT-L/14-reg 的 `x_prenorm` 对每个视角输出：

$$1+4+37^2=1374$$

个 token，其中包括 1 个 CLS token、4 个 register tokens 和 1369 个 patch tokens，每个 token 为 1024 维。代码执行：

```python
features["x_prenorm"][:, dinov2_model.num_register_tokens + 1:]
```

删除 CLS 与 register tokens，然后转置并 reshape 为：

$$\boldsymbol F_v\in\mathbb R^{1024\times37\times37}.$$

这里取的是最后一个 Transformer block 经过最终 LayerNorm 之前的 `x_prenorm`，而不是分类头输出，也不是 CLS token。

### 6.5 从 2D patch map 回投到 $64^3$ active voxels

对于 PLY 中每个 active voxel center $\boldsymbol p_i$，代码利用 `transforms.json` 中的相机参数构造外参和归一化内参，再把该 3D 点投影到每个视角：

$$\boldsymbol u_{iv}=\pi_v(\boldsymbol p_i).$$

接下来不是把坐标舍入到最近的 DINO patch，而是使用 `torch.nn.functional.grid_sample` 在 $37\times37$ patch feature map 上做双线性采样：

$$\tilde{\boldsymbol f}_{iv}=\operatorname{BilinearSample}(\boldsymbol F_v,\boldsymbol u_{iv})\in\mathbb R^{1024}.$$

最后对所有视角直接取算术平均：

$$\boldsymbol f_i=\frac{1}{V}\sum_{v=1}^{V}\tilde{\boldsymbol f}_{iv},\qquad V=150.$$

于是每个 active voxel 都获得一个 1024 维多视角视觉描述，输出文件 `features/dinov2_vitl14_reg/<sha256>.npz` 包含：

|Key|Shape|Dtype|含义|
|---|---:|---|---|
|`indices`|$N\times3$|`uint8`|$64^3$ 网格中的整数坐标，范围 0–63|
|`patchtokens`|$N\times1024$|`float16`|150 视角 DINOv2 特征的平均|

一个值得特别记录的实现细节是：官方 `extract_feature.py` **没有显式执行深度测试、遮挡判断或可见视角 mask**。投影落到图像外部时，`grid_sample` 默认贡献零值；落到图像内部但被其他表面遮挡时，代码也不会将其排除。最终仍然对全部 150 个视角做普通平均。这是官方实现相对于论文中“多视角特征聚合”这一高层描述更具体、也更粗糙的一面。

### 6.6 SLat VAE Encoder：不是简单的 $1024\rightarrow8$ 线性压缩

`dataset_toolkits/encode_latent.py` 读取上述 NPZ，并构造稀疏张量：

$$\mathcal F=\left\{(\boldsymbol p_i,\boldsymbol f_i)\right\}_{i=1}^{N},\qquad\boldsymbol p_i\in\{0,\ldots,63\}^{3},\quad\boldsymbol f_i\in\mathbb R^{1024}.$$

代码中的 sparse coordinate 实际存为 `[batch_id, x, y, z]`，单资产处理时 `batch_id` 全为 0。默认 SLat encoder 配置为：

```text
resolution = 64
in_channels = 1024
model_channels = 768
latent_channels = 8
num_blocks = 12
num_heads = 12
attn_mode = swin
window_size = 8
```

真实网络流程为：

1. `SparseLinear(1024, 768)` 把每个 voxel 的 DINOv2 特征映射到模型宽度；
2. 根据整数 3D 坐标加入 absolute positional embedding；
3. 经过 12 个 Sparse Transformer blocks；
4. attention 使用 3D Swin window，窗口大小为 8，相邻 block 交替采用 0 和 4 的 shifted window；
5. 对每个 active voxel 做 LayerNorm；
6. `SparseLinear(768, 16)` 输出 8 维 mean 与 8 维 log-variance；
7. `sample_posterior=False` 时直接取 8 维 mean 作为局部 latent。

因此，把这一过程简称为“DINO 特征 $1024\rightarrow8$”在维度上没有错，但容易误解为一个独立线性投影。实际上，每个 voxel 的 8 维 latent 已经通过 12 层稀疏自注意力融合了周围 active voxels 的空间上下文。

同样需要纠正“稀疏卷积和稀疏注意力共同完成编码”这一表述：**SLat VAE encoder 主体是 Sparse Transformer/Swin attention，并不包含 SS VAE 那种逐级下采样的 3D 卷积编码路径。** 它的输入和输出 active coordinates 完全一致：

$$\{(\boldsymbol p_i,\boldsymbol f_i\in\mathbb R^{1024})\}_{i=1}^{N}\longrightarrow\{(\boldsymbol p_i,\boldsymbol z_i\in\mathbb R^{8})\}_{i=1}^{N}.$$

编码后的 `latents/<model>/<sha256>.npz` 保存：

|Key|Shape|Dtype|含义|
|---|---:|---|---|
|`coords`|$N\times3$|`uint8`|保持不变的 active voxel coordinates|
|`feats`|$N\times8$|`float32`|每个 active voxel 的局部 latent mean|

### 6.7 “SLat 始终保持 $64^3$”的准确含义

SLat 的**坐标支撑域**始终是 $64^3$：

$$\boldsymbol p_i\in\{0,\ldots,63\}^{3}.$$

但它不是一个稠密的 $64\times64\times64\times8$ 张量。真正保存和计算的是 $N$ 个 active entries，通常满足 $N\ll64^3$。因此完整 SLat 应写为：

$$\mathcal Z=\left\{(\boldsymbol p_i,\boldsymbol z_i)\right\}_{i=1}^{N},\qquad\boldsymbol z_i\in\mathbb R^8,$$

而不是只写一个脱离坐标的 $N\times8$ 矩阵。坐标提供 coarse structure，8 维局部向量承载细几何和外观；两者共同构成表示。

第二阶段官方 Flow 配置 `slat_flow_img_dit_L_64l8p2_fp16.json` 设置 `resolution=64`、`in_channels=out_channels=8`。其中 `patch_size=2` 会让 Flow 主干在内部通过 Sparse ResBlock 做一次 2 倍 downsample，Transformer 处理后再 upsample 回去，以降低计算量；这只是生成网络内部的临时 packing/unpacking，并没有改变最终 SLat 的 $64^3$ 坐标定义，模型输出仍与输入 active support 对齐。

### 6.8 两个 Flow Model 到底分别学习什么

第一阶段的训练样本是一个固定形状的 dense tensor：

$$\boldsymbol z_S\in\mathbb R^{8\times16\times16\times16}.$$

Structure Flow 从高斯噪声出发，在图像或文本条件下预测 rectified-flow velocity，最终生成 $\boldsymbol z_S$。SS decoder 再执行：

$$\boldsymbol z_S\xrightarrow{\mathcal D_S}\hat{\boldsymbol X}_{\mathrm{occ}}\in\mathbb R^{1\times64^3}\xrightarrow{\text{threshold}}\{\boldsymbol p_i\}_{i=1}^{N}.$$

第二阶段的训练样本则是一个坐标已知、长度随物体变化的 sparse tensor：

$$\mathcal Z=\{(\boldsymbol p_i,\boldsymbol z_i)\}_{i=1}^{N}.$$

SLat Flow 不再生成 occupancy，也不负责决定 $N$ 个点位于哪里；它接收第一阶段给出的 active coordinates，只在这些位置上把 8 维噪声流动到目标局部 latent：

$$\{(\boldsymbol p_i,\boldsymbol\epsilon_i)\}_{i=1}^{N}\xrightarrow{\text{SLat Flow}}\{(\boldsymbol p_i,\boldsymbol z_i)\}_{i=1}^{N}.$$

最后，同一份 SLat 可分别送入 Gaussian、Radiance Field 或 Mesh decoder。整个推理链可以压缩为：

$$\text{图像/文本条件}\rightarrow8\times16^3\text{ structure latent}\rightarrow64^3\text{ active coordinates}\rightarrow N\times8\text{ local latents}\rightarrow\text{GS / RF / Mesh}.$$

### 6.9 全链路张量与文件总表

|步骤|表示或文件|Shape / Resolution|是否改变空间结构|
|---|---|---:|---|
|资产归一化|mesh|落入 $[-0.5,0.5]^3$|统一缩放与居中|
|表面体素化|`voxels/*.ply`|$N\times3$ 格心坐标|建立 $64^3$ active support|
|构造 occupancy|内存 tensor|$1\times1\times64^3$|稀疏点散射成稠密二值网格|
|SS VAE encoder|`ss_latents/*.npz: mean`|$8\times16^3$|空间下采样 4 倍|
|多视角渲染|`renders/*.png`|150 张 $512^2$ RGBA|不改变 3D support|
|DINOv2|`x_prenorm` patch map|每视角 $1024\times37\times37$|2D 特征抽取|
|投影聚合|`features/*.npz`|`indices`: $N\times3$；`patchtokens`: $N\times1024$|回到同一组 $64^3$ active voxels|
|SLat VAE encoder|`latents/*.npz`|`coords`: $N\times3$；`feats`: $N\times8$|坐标和 active 数量不变|
|Flow 1|dense latent|$8\times16^3$|生成结构 latent|
|SS VAE decoder|occupancy|$1\times64^3$|恢复并确定 active coordinates|
|Flow 2|sparse latent|$N\times8$ + $N\times3$ coords|固定坐标，生成局部向量|
|格式 decoder|GS / RF / Mesh|随目标表示变化|把统一 SLat 映射到具体资产格式|

### 6.10 最终准确表述

可以把 TRELLIS 的这一段流程准确概括为：

> TRELLIS 先把归一化资产的表面离散为 $64^3$ active voxels。一条支路将稠密二值 occupancy 经 3D Conv VAE 压缩成 $8\times16^3$，作为第一阶段 Structure Flow 的目标分布；另一条支路渲染 150 个 $512^2$ RGBA 视角，resize 到 $518^2$ 后用 DINOv2 ViT-L/14-reg 提取 $1024\times37\times37$ patch features，再把每个 active voxel center 投影到所有视角并做双线性采样与平均，得到 $N\times1024$ 特征。SLat VAE 通过 12 层 Sparse Swin Transformer 在不改变 active coordinates 的前提下将其编码为每体素 8 维，形成 $\{(\boldsymbol p_i,\boldsymbol z_i)\}_{i=1}^{N}$，作为第二阶段 SLat Flow 的目标。推理时先由 Flow 1 生成结构并恢复 $64^3$ active coordinates，再由 Flow 2 在这些坐标上生成 $N\times8$ 局部 latent，最后交给不同 decoder 输出 Gaussian、Radiance Field 或 Mesh。

### 6.11 对应官方代码入口

- 资产渲染：<https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/render.py>
- Blender 渲染脚本：<https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/blender_script/render.py>
- 表面体素化：<https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/voxelize.py>
- DINOv2 多视角特征聚合：<https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/extract_feature.py>
- Sparse Structure latent 编码：<https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/encode_ss_latent.py>
- SLat 编码：<https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/encode_latent.py>
- Sparse Structure VAE：<https://github.com/microsoft/TRELLIS/blob/main/trellis/models/sparse_structure_vae.py>
- SLat VAE Encoder：<https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_vae/encoder.py>
- SLat Flow：<https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_flow.py>
