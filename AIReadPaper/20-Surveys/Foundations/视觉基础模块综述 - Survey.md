---
type: survey
status: done
domain: Foundations
title: 视觉基础模块综述 —— 从 2D 特征到 3D 几何估计的基础模型
year: 2026
source: 多源（arXiv 与官方项目页）
tags:
  - Foundations
  - 深度估计
  - 特征提取
  - MVS
  - 手部姿态
created: 2026-08-26
updated: 2026-08-26
---

# 视觉基础模块综述 —— 从 2D 特征到 3D 几何估计的基础模型

## 基本信息

**报告主题**：系统梳理构成上层三维视觉系统（3D 生成、三维重建、世界模型、VLA）的底层「基础模块」——特征提取、单目深度、多视图深度 / MVS、手部 pose，以及相机位姿、光流、分割、法向等周边模块。

**覆盖范围**：18 篇代表性论文，按四条主线 + 周边速览组织：

- **特征提取**：CLIP、DINO、DINOv2、SAM
- **单目深度估计**：MiDaS、DPT、ZoeDepth、Metric3D、Depth Anything / V2
- **多视图深度与 MVS**：MVSNet、CasMVSNet、MVSFormer、DUSt3R、MASt3R（衔接 VGGT）
- **手部 pose 估计**：MANO、MediaPipe Hands、FrankMocap、HaMeR、WHAM
- **周边**：相机位姿（本质矩阵 / PnP）、光流（RAFT）、分割（Mask2Former）、法向（Omnidata / Marigold）

**材料来源**：各论文 arXiv 官方页与项目页，关键配图取自 arXiv HTML 版 / 官方 GitHub。

**撰写日期**：2026-08-26

**报告定位**：以「2D 特征 → 2.5D 深度 → 3D 几何 → 人体/手部姿态」这条感知升维主线组织内容，讲清每个模块「解决什么问题、核心机制是什么、谁是最关键的里程碑、如何被上层复用」，而非按论文逐篇罗列。

---

## 1. 研究背景与动机

### 1.1 什么是「基础模块」

在 3D 视觉大模型的技术栈里，有一批反复出现、相对独立、可被复用的能力单元：把图像编码成语义向量的**特征提取器**，从单张图恢复深度的**单目深度模型**，从多视角恢复几何的 **MVS / 前馈重建模型**，以及估计人体或手部姿态的**姿态估计模型**。它们本身不是最终产品，却是 3D 生成、三维重建、世界模型和 VLA 等上层系统共同的「感知底座」。

理解这批基础模块的价值，在于一个朴素的观察：**上层系统的能力瓶颈，往往不在上层创新，而在底层感知模块的质量**。例如，一个 image-to-3D 模型无论扩散架构多精妙，若喂给它的深度先验不准、特征对齐不好，输出的几何就会漂移；一个 VLA 若底层特征缺乏 3D 一致性，抓取就会在新的视角下失效（这正是 AtlasVLA、CVPR 2026 冠军奖那篇 VLA 空间表征工作的结论）。因此，把基础模块单独梳理成一条脉络，是进入 3D 视觉大模型的可靠起点。

### 1.2 一条贯穿全文的线索：感知升维

本文用一条连续的技术叙事把四类模块串起来：

1. **2D 特征**（第 2 章）：CLIP / DINO 系 / SAM 解决「图像里有什么、语义上是什么、哪些像素属于同一物体」。
2. **2.5D 深度**（第 3 章）：单目深度解决「每个像素离相机多远」，是单张图像到 3D 的第一步升维。
3. **3D 几何**（第 4 章）：多视图深度 / MVS / 前馈重建解决「多个视角如何联合还原完整几何与相机位姿」。
4. **人体/手部姿态**（第 6 章）：在人体这一高度结构化的对象上，把「几何」进一步参数化成语义关节与可变形网格。

沿着这条升维链路，能看到一个反复出现的范式迁移：**从任务专用的小模型，走向自监督 / 大规模数据驱动的「基础模型」**——特征提取从 ImageNet 监督走向 CLIP/DINO 自监督，深度估计从 MiDaS 走向 Depth Anything，MVS 从 MVSNet 走向 DUSt3R/VGGT 前馈大模型，手部重建从 FrankMocap 走向 HaMeR/WHAM。这是理解整条主线的最关键视角。

---

## 2. 特征提取模块：视觉表征的底座

### 2.1 从 ImageNet 预训练到自监督表征

在深度学习早期，视觉特征的标准来源是在 ImageNet 上做监督分类预训练，然后取分类网络的中间层作为通用特征。这条路线的问题是：**监督信号（类别标签）能提供的语义粒度有限，且标注成本高、难以扩展到海量数据**。

2019–2021 年，自监督学习把这条路线推向了新的阶段。对比学习（MoCo、SimCLR）把「同一图像的两个增强视图拉近、不同图像拉远」作为代理任务；掩码自编码（MAE）把「重建被掩掉的 patch」作为任务；而 DINO 走的是自蒸馏路线。这些方法的共同目标，是在**无标签**的前提下学出语义有意义的表征。这一阶段的终点，是把「特征提取」本身变成了一个可以大规模预训练、跨任务迁移的**基础能力**。

### 2.2 CLIP：对比语言-图像预训练

CLIP（Radford et al., 2021）的关键洞察是：**用自然语言作为图像的监督信号**，从而摆脱人工类别标签的束缚。它收集了 4 亿对「图像–文本」，用一个图像编码器和一个文本编码器分别把图像和文本映射到共享的嵌入空间，然后优化一个对比目标——让匹配的图文对在嵌入空间里靠近，不匹配的远离。

![](assets/视觉基础模块综述%20-%20Survey/figures/01-clip-contrastive.png)

> 图 1：CLIP 的对比预训练框架。图像编码器与文本编码器把 $N$ 对图文映射到共享空间，用对比损失让对角线上的正对靠近、其余负对远离。  
> 来源：OpenAI CLIP 论文 Figure 1，https://arxiv.org/abs/2103.00020

训练目标是一个对称的 InfoNCE 损失：给定一个 batch 里的 $N$ 个图像嵌入 $\{v_i\}$ 与 $N$ 个文本嵌入 $\{t_i\}$，图像到文本的相似度矩阵经过 softmax 后与对角标签做交叉熵：

$$\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(v_i^\top t_i / \tau)}{\sum_j \exp(v_i^\top t_j / \tau)} + \log\frac{\exp(t_i^\top v_i / \tau)}{\sum_j \exp(t_j^\top v_i / \tau)}\right]$$

其中 $\tau$ 是可学习的温度参数。这个目标的妙处在于**没有显式的分类头**：训练完成后，模型天然支持零样本分类——把每个候选类别名（如 "a photo of a dog"）编码成文本向量，与图像向量做相似度，取最高者即可。

CLIP 的历史地位在于它**定义了「视觉–语言对齐」这一基础能力**，成为后续几乎所有 VLM（LLaVA、Qwen-VL、InternVL）视觉编码器的默认起点，其变体（OpenCLIP、SigLIP、EVA-CLIP）进一步放大了数据规模与训练效率。对 3D 社区而言，CLIP 嵌入还是 text-to-3D（DreamFusion 一脉）里「文本条件」的语义来源。

### 2.3 DINO / DINOv2：自监督 Transformer

DINO（Caron et al., 2021）走了一条与对比学习不同的自监督路线：**自蒸馏**。它维护一个 student 网络和一个 teacher 网络（teacher 是 student 的指数移动平均），对同一图像的两个增强视图分别预测，让 student 的输出去匹配 teacher 的输出（用 centering + sharpening 防止表示坍缩）。这个目标的副产品令人惊讶：**ViT 的 attention map 自发涌现出清晰的物体分割**，无需任何像素级标签。

DINOv2（Oquab et al., 2023）则把这条路线推向「通用视觉特征基础模型」：通过大规模精选数据 + 蒸馏 + 判别式自监督，产出在分类、分割、深度估计、检索等任务上**无需微调即可取得强结果**的通用特征。

![](assets/视觉基础模块综述%20-%20Survey/figures/02-dinov2-fig1.jpg)

> 图 2：DINOv2 论文的概览图。自监督预训练产出的特征在分类、分割、深度估计、检索等多任务上无需微调即表现优异，展示了「通用视觉特征」的定位。  
> 来源：DINOv2 论文 Figure 1，https://arxiv.org/abs/2304.07193

DINOv2 对 3D 社区的影响极为深远：它的特征被广泛用作**稠密对应、深度估计（Depth Anything 的 teacher）、图像匹配（DUSt3R/Mast3R 的骨干）、以及 3D 重建的特征场**。如果说 CLIP 提供的是「语义级对齐」，DINOv2 提供的则是「像素级、几何友好的通用特征」——后者对 3D 任务往往更重要，因为它保留了局部几何与纹理细节。

### 2.4 SAM：可提示分割基础模型

SAM（Kirillov et al., 2023）解决了分割任务长期以来的痛点：**每换一个领域就要重新标注、重新训练一个分割模型**。它的核心思想是「可提示分割」（promptable segmentation）——用一个统一的模型，接受点、框、掩码或文本作为提示，输出对应物体的分割掩码。

![](assets/视觉基础模块综述%20-%20Survey/figures/03-sam-model.png)

> 图 3：SAM 的架构。图像编码器（MAE 预训练的 ViT）一次性编码图像，prompt 编码器编码点/框/掩码提示，轻量 mask 解码器融合两者输出分割掩码，支持交互式分割。  
> 来源：SAM 论文 Figure 2，https://arxiv.org/abs/2304.02643

SAM 由三部分组成：一个基于 MAE 预训练 ViT 的**图像编码器**、一个轻量的**提示编码器**、以及一个**掩码解码器**。训练数据来自数据引擎产出的 SA-1B 数据集——超过 10 亿个掩码。推理时，图像编码器只需计算一次，之后用户可交互式地给出提示，实时获得分割结果。由于分割本质上是模糊的（同一个提示可能对应多个合理物体），SAM 会输出多个掩码并附带置信度。

SAM 的价值不止于分割本身：它的「可提示」范式催生了一大批后续工作——SAM 3D 把它推广到 3D 资产分割，医学影像、视频分割、点云分割等领域纷纷效仿。它和 CLIP 一样，是「把一个任务做成了可复用基础能力」的典范。

### 2.5 特征模块如何被上层复用

小结这一章：三类特征模块分别提供不同粒度的基础能力——CLIP 提供**语义对齐**，DINOv2 提供**几何友好的通用特征**，SAM 提供**可提示的物体级分割**。它们是几乎所有上层 3D 系统的输入或初始化来源，这一点在第 8 章会做横向归纳。

---

## 3. 单目深度估计：单张图像的 2.5D 升维

### 3.1 问题定义与尺度歧义

单目深度估计的目标是：给定单张 RGB 图像 $I \in \mathbb{R}^{H\times W\times 3}$，预测每个像素的深度 $D \in \mathbb{R}^{H\times W}$。这是一个**病态问题（ill-posed）**：同一张图像可以由无数种「场景 + 相机距离」组合产生——物体近且相机离得近，与物体大且相机离得远，在单目观测下是不可区分的。这被称为**尺度歧义（scale ambiguity）**。

尺度歧义直接决定了两条路线：**相对深度估计**只要求预测的深度在「谁近谁远」上正确（尺度与偏移可以任意），而**度量深度估计**要求预测出真实的物理深度（以米为单位）。前者好做、泛化强，后者困难、但可用于下游的真实几何任务。这条「相对 vs 度量」的分野，是理解单目深度演化的关键线索。

### 3.2 相对深度：MiDaS 与 DPT

MiDaS（Ranftl et al., 2020）确立了相对深度估计的基本范式：**用大规模多源数据集混合训练 + 尺度不变的损失函数**。由于不同深度数据集的深度标度（scale）和零点（shift）各不相同，直接混合训练会被尺度不一致干扰。MiDaS 的解法是「尺度-平移不变」损失：先对预测深度 $\hat d$ 和真实深度 $d$ 做最小二乘对齐，再算误差。

具体地，令 $s, t$ 为对齐参数，求解

$$\min_{s,t}\sum_i (s\,\hat d_i + t - d_i)^2$$

得到最优对齐后再计算误差，等价于在训练中只惩罚「深度排序/相对关系」的错误，而不惩罚尺度和偏移的错误。这个损失让 MiDaS 能把大量异构数据集（KITTI、NYU、ScanNet 等）混在一起训练，学出跨数据集泛化的相对深度。

DPT（Ranftl et al., 2021）则解决了「如何把 ViT 用到密集预测」的问题：ViT 输出的是粗粒度的 patch token，不直接适配逐像素的深度图。DPT 通过 reassemble（把 token 重新组装成多尺度特征图）+ fusion（渐进上采样融合）把 ViT 变成密集预测网络，显著提升了深度图的边缘与结构质量。这一「ViT 做 dense prediction」的配方后来被广泛复用。

### 3.3 度量深度：ZoeDepth 与 Metric3D

度量深度估计的难点在于：真实深度受相机内参（焦距）影响，不同相机的同一场景深度标度不同。两条代表性工作分别给出了解法。

**ZoeDepth**（Bhat et al., 2023）采用「相对深度预训练 + 度量深度微调」两阶段：先在 MiDaS 上做相对深度预训练，然后在特定数据集的度量深度上微调一个轻量的 metric bins 模块——把深度范围离散成若干个 bin，预测每个像素落在哪个 bin 及其中心偏移。因为只在很小的模块上做度量微调，它既保留了相对深度的泛化能力，又能在目标域输出真实深度。其核心是「**用相对深度换泛化，用少量度量微调换精度**」。

**Metric3D**（Yin et al., 2023）则从另一个角度切入：它指出度量深度训练失败的一个根因是**训练数据相机内参不一致**。它提出把训练与推理都映射到一个 canonical camera space（焦距归一化），让模型在统一的相机坐标系下学习度量深度，从而跨相机泛化。这比 ZoeDepth 更进一步——**不是「微调适配」，而是「从数据归一化上根治尺度问题」**。

### 3.4 基础模型化：Depth Anything 系列

Depth Anything（Yang et al., 2024）把单目深度推向了「基础模型」的规模。它的核心洞察是：**无标注图像几乎无限，可以用它们做伪标签自训练**。具体做法是：先用标注数据训练一个 teacher 模型，让 teacher 给 6200 万张无标注图像打上伪深度标签，再用「标注 + 伪标注」混合数据训练 student。

![](assets/视觉基础模块综述%20-%20Survey/figures/04-depth-anything-pipeline.png)

> 图 4：Depth Anything 的训练流程。先用标注数据训练 teacher，teacher 为大规模无标注图像生成伪深度标签，再联合标注与伪标注数据训练 student，从而把数据规模放大到 6200 万张。  
> 来源：Depth Anything 论文 Figure 2，https://arxiv.org/abs/2401.10891

这个「伪标签自训练」范式与特征提取的演进形成呼应：两者都在把「数据规模」作为第一性原理。Depth Anything 的输出在零样本跨域深度估计上取得了当时最佳结果。

Depth Anything V2（Yang et al., 2024）针对 V1 的两个弱点——**大模型幻觉产生的伪标签噪声**与**细粒度/透明物体等困难区域的退化**——做了改进：用合成数据补充训练（合成数据的深度是精确的），并在更小、更高效的模型上训练。结果是更鲁棒、更快、细粒度更好的深度估计。随后 Depth Anything 3（DA3，2025）进一步把深度估计推广到「任意视角的视觉空间恢复」，进入前馈 3D 重建的范畴（该工作已在「三维重建」领域登记，见 [[DA3 - 任意视角恢复视觉空间]]）。

### 3.5 数据集与评价指标

单目深度的常用数据集与指标：

- **室内**：NYU-Depth v2（RGB-D），常用指标 RMSE、$\delta_1$（相对误差小于 1.25 的比例）、AbsRel。
- **室外驾驶**：KITTI、DDAD，用 AbsRel、SILog、RMSE。
- **跨域零样本**：Sintel、ETH3D、DIODE 等，衡量泛化能力。

需要特别指出的是：**相对深度与度量深度的指标不可直接比较**——前者用尺度不变误差（如 SILog），后者用绝对误差（如 RMSE）。读单目深度论文时，先分清它评的是相对还是度量，否则数字会完全误导。

---

## 4. 多视图深度估计与 MVS：多视角的 3D 几何

### 4.1 从两视图立体到多视图立体

单目深度的尺度歧义，本质上是因为**缺少第二个视角**。多视图立体（Multi-View Stereo, MVS）用「多个已知相对位姿的相机看到的同一场景」来消除歧义：给定 $N$ 张图像及其相机内外参，估计每个像素的深度或直接输出稠密点云。核心几何工具是**三角测量（triangulation）**——两条视线在 3D 空间的交点决定了深度。

MVS 的输入比单目多了一个关键约束：**相机位姿已知**（通常由 SfM 预先求得）。这使它的问题性质从「病态」变为「可解」，但代价是需要多视图和位姿。因此 MVS 与相机位姿估计天然耦合，这也是为什么后面 DUSt3R 能「把位姿和几何一起端到端预测」会成为范式级突破。

### 4.2 经典几何路线：COLMAP 与 PatchMatch

在学习式方法出现之前，MVS 的标准解是几何 + 优化的组合。**COLMAP** 是这一路线的集大成者：先做增量式 SfM 恢复相机位姿，再做稠密 MVS，其深度图估计 + 融合的流程至今仍是 3D 重建的「地基」——大量学习式方法仍用 COLMAP 做数据预处理或位姿初始化。**PatchMatch** 系列（Gipuma 等）则用「随机初始化 + 邻域传播 + 视图选择」的迭代优化在 GPU 上高效地估计深度，强调几何一致性与视差平滑。

这些经典方法的共同特征：**可解释、不依赖训练数据，但在弱纹理、遮挡、重复结构处容易失败**，因为它们依赖局部光度一致性假设。

### 4.3 学习式 MVS：MVSNet → CasMVSNet → MVSFormer

MVSNet（Yao et al., 2018）开创了「学习式 MVS」范式，其核心是**可微单应变换 + cost volume**。给定参考视图与源视图，把源视图通过单应变换 warp 到参考视图的一系列深度假设平面上：

$$H_i(d) = K_i R_i\left(I - \frac{(t_1 - t_i)n_1^\top}{d}\right)K_1^{-1}$$

其中 $K$ 是内参，$R,t$ 是相对位姿，$n_1$ 是参考相机的主轴方向，$d$ 是深度假设。把 $D$ 个深度假设上 warp 后的特征按**方差**聚合成一个 3D cost volume，再用 3D CNN 回归深度图。方差聚合（而非平均）让「多视图特征在正确深度处一致、错误深度处发散」这一性质被显式编码。

CasMVSNet（Gu et al., 2020）引入**由粗到细（coarse-to-fine）**的策略：先在低分辨率估计深度，再在深度范围内递进式细化，大幅降低 cost volume 的内存与计算开销——这是把 MVS 推向实用分辨率的关键一步。MVSFormer（Cao et al., 2022）则把 **ViT 特征**引入 MVS，用更强的图像特征（相比 MVSNet 的浅层 CNN 特征）提升匹配质量，验证了「**特征质量决定 MVS 上限**」这一判断——与第 2 章特征提取的结论形成了呼应。

### 4.4 前馈重建大模型：DUSt3R → MASt3R → VGGT

DUSt3R（Wang et al., 2023）带来了一次范式级的简化。传统 MVS 需要「先标定相机、再估位姿、再算深度」的漫长管线，而 DUSt3R 直接把**任意两张图像**输入一个 Transformer，回归出它们在**同一坐标系下的点图（pointmap）**——即每个像素对应的 3D 点，同时隐含了深度与相对位姿。

![](assets/视觉基础模块综述%20-%20Survey/figures/05-dust3r-overview.png)

> 图 5：DUSt3R 的概览。任意两张图输入网络，直接回归统一坐标系下的点图（pointmap），无需相机标定与显式位姿优化，就能恢复 3D 几何与相对位姿。  
> 来源：DUSt3R 论文 Figure 1，https://arxiv.org/abs/2312.14132

形式化地，给定两张图 $I_1, I_2$，网络预测两个点图 $X^{1,1}$ 与 $X^{2,1}$（都在 $I_1$ 的相机坐标系下），那么 $I_2$ 的深度和两相机间的相对位姿都可以从点图中解析出来。多张图则通过「全局对齐 + 置信度加权」融合成一个全局点云。这种「**端到端回归几何**」的方式省去了显式的匹配、三角测量与 Bundle Adjustment，是「前馈式 3D 重建」的奠基工作。

MASt3R（Leroy et al., 2024）在 DUSt3R 的点图之上加了一个**特征匹配头**，使网络在输出 3D 几何的同时也输出稠密 2D 对应与相对位姿，实现了「**无匹配器（matching-free）的匹配 + 重建统一**」。它进一步证明了「点图」这一表征可以作为 2D 匹配与 3D 重建的共同锚点。

这条线的顶点是 VGGT（2025）与 VGGT-Ω（2026）：用单网络同时预测深度、点云、相机参数与轨迹，且随数据与模型规模可预测提升（该工作已在「三维重建」领域登记，见 [[VGGT - 前馈式通用三维重建]]、[[VGGT-Ω - 可扩展前馈式静动态三维重建]]）。

### 4.5 单目与多视图的关系

值得点明一个趋势：**单目深度与多视图重建正在合流**。DUSt3R/VGGT 本质上是「把单目深度的前馈回归能力 + 多视图的一致性约束」统一进一个网络。而 DA3 则反过来，让单目深度模型具备「任意视角」的几何恢复能力。第 3 章和第 4 章看似分开，实则正收敛到同一个「前馈几何基础模型」的目标上——这是理解当前 3D 重建前沿最关键的一个判断。

---

## 5. 相机位姿估计（周边）

### 5.1 问题定义

相机位姿估计解决「相机在哪里、朝哪看」，是 MVS、SfM、SLAM 的前置。它分几个层次：**相对位姿**（两视图之间的旋转平移）、**绝对位姿**（相机在世界坐标系中的位姿）、以及 **PnP**（给定 3D 点与 2D 投影，求解位姿）。

### 5.2 两视图相对位姿：本质矩阵

两视图相对位姿的经典工具是**本质矩阵（Essential Matrix）** $E$。给定一对匹配点 $x_1, x_2$（归一化坐标），满足对极约束

$$x_2^\top E\, x_1 = 0$$

$E$ 可由 8 点法（或更少的 5 点法）线性求解，再分解出旋转 $R$ 与平移 $t$（相差一个尺度）。这是 MVS 里「先求位姿、再求深度」的几何基础。

### 5.3 前馈位姿回归

传统方法靠特征匹配 + RANSAC + Bundle Adjustment 求位姿，脆弱且慢。学习式方法（如 RelPose 等）用网络直接回归两视图的相对旋转，DUSt3R/Mast3R 则把位姿「隐含」在点图回归里一并输出。趋势是：**位姿估计从「几何求解」走向「前馈回归」，并与几何重建融合进同一个网络**——这也解释了为什么位姿在本文中被当作「周边」，因为它正被第 4 章的前馈重建大模型吸收。

---

## 6. 手部 pose 估计：从关键点到 3D 网格

### 6.1 手部的特殊性

手部 pose 估计与一般人体姿态相比有几大难点：**关节数量多且自由度极高**（21 个关键点、大量耦合的关节运动）、**自遮挡严重**（握拳、抓物时大部分手指被遮挡）、**对称性**（左右手、正反手易混淆）、以及**在图像中的尺度小**。因此手部重建对特征的表征能力和先验建模都要求更高。

### 6.2 2D 关键点估计：MediaPipe Hands

在 3D 网格重建之前，2D 关键点估计是更成熟的基础能力。OpenPose、MediaPipe Hands（Zhang et al., 2020）等能把手的 21 个关键点（甚至粗略的 3D 坐标）实时检测出来。MediaPipe Hands 以「轻量、实时、可端上部署」著称，它先用手部检测器定位手掌，再回归 21 个关键点的 2.5D 坐标。这类 2D/2.5D 关键点虽不能给出完整手部形状，但为手势识别、交互提供了足够的信息。

### 6.3 3D 参数化：MANO

MANO（Romero et al., 2017）是 3D 手部重建的**参数化标准**，地位相当于 SMPL 之于人体。它把一只手的形状和姿态参数化为：

- **姿态参数** $\theta \in \mathbb{R}^{48}$：16 个关节的旋转（每个用轴角表示，3 维）；
- **形状参数** $\beta \in \mathbb{R}^{10}$：手部形状在 PCA 基上的系数。

给定 $\theta, \beta$，MANO 通过**线性混合蒙皮（LBS）**生成带 778 个顶点的手部网格。MANO 的关键价值在于**可微**：它是一个可微的函数 $M(\theta, \beta)$，因此可以直接接入神经网络端到端训练，用 2D 关键点重投影误差、3D 关键点误差等作为监督。几乎所有现代 3D 手部重建方法都以「回归 MANO 参数」作为输出头。

### 6.4 3D 手部重建基础模型：FrankMocap / HaMeR / WHAM

**FrankMocap**（Rong et al., 2021）是单目实时手部 + 身体姿态捕捉的经典工作：一个轻量网络同时估计身体（SMPL）和手部（MANO），并用注意力机制整合手与身体的信息，解决「手在全身图像中尺度小」的问题。

**HaMeR**（Pavlakos et al., 2023）把手部重建推向了「基础模型」：它以 ViT 为骨干、MANO 为输出，在包含大量动作的手部视频数据集（如 Assembly101、EpicKitchens 等 in-the-wild 数据）上大规模训练，显著提升了真实场景（握持、遮挡、运动模糊）下的重建鲁棒性。

![](assets/视觉基础模块综述%20-%20Survey/figures/06-hamer-pipeline.png)

> 图 6：HaMeR 的架构与训练流程。ViT 骨干编码输入图像，回归 MANO 的姿态与形状参数，并联合 2D/3D 关键点重投影损失与大规模 in-the-wild 数据训练。  
> 来源：HaMeR 论文 Figure 2，https://arxiv.org/abs/2312.05251

**WHAM**（Goel et al., 2023）则把视角拉回到「世界坐标系」：它从单目视频重建**世界坐标下的人体 + 手部运动**，同时估计相机轨迹与脚部接触，输出全局一致、可放到真实 3D 场景中的运动序列。这与第 4 章「世界坐标下的几何重建」形成呼应——**手部重建也在从「相机系」走向「世界系」**。

---

## 7. 其它基础模块速览（光流 / 分割 / 法向）

### 7.1 光流：RAFT

光流估计「相邻两帧之间每个像素的运动矢量」，是视频理解、运动分割、世界模型的重要基础。RAFT（Teed & Deng, 2020）是这一方向的事实标准：它构建一个 4D 相关体（所有像素对之间的特征相似度），再用 GRU 迭代更新光流场，兼顾了精度与泛化。其「相关体 + 迭代更新」的设计被后续大量稠密对应方法借鉴（例如与 DUSt3R 的稠密对应头在设计思想上相通）。

### 7.2 通用分割：Mask2Former

在 SAM 之前，通用分割的代表是 Mask2Former（Cheng et al., 2022）：它用「masked attention + 统一查询」把实例分割、语义分割、全景分割统一进一个架构。SAM 与 Mask2Former 的对比很有意思：SAM 强调「可提示的零样本分割」，Mask2Former 强调「统一架构下的全景分割」，两者共同构成了现代分割基础能力的两翼。

### 7.3 法向估计：Omnidata 与 Marigold

表面法向是比深度更「局部」的几何属性，对材质、光照、重建都重要。Omnidata（2021）用统一的多任务网络同时估计深度与法向；Marigold（2023）则创新性地用**扩散模型先验**做法向/深度估计——把预训练图像扩散模型改造成「以图像为条件、迭代去噪出深度或法向」，展示了「生成式先验」作为几何估计新基座的潜力。这条「用生成模型的先验做几何」的路线，与本文第 3、4 章「基础模型化」的趋势一脉相承。

---

## 8. 横向总结：基础模块如何组合成上层系统

### 8.1 一表看懂脉络

| 模块 | 问题 | 代表性方法 | 关键机制 | 主要下游 |
|---|---|---|---|---|
| 特征提取 | 图像里有什么 | CLIP / DINOv2 / SAM | 对比学习 / 自蒸馏 / 可提示 | VLM、3D 生成、所有任务的编码器 |
| 单目深度 | 每个像素多远 | MiDaS → Depth Anything | 尺度不变损失 / 伪标签自训练 | 3D 生成先验、VLA、重建 |
| 多视图深度 / MVS | 多视角联合几何 | MVSNet → DUSt3R/VGGT | cost volume / 点图回归 | 三维重建、SLAM、世界模型 |
| 手部 pose | 手的姿态与形状 | MANO / HaMeR / WHAM | 参数化网格 / ViT 回归 | 人机交互、VR/AR、动捕 |
| 相机位姿 | 相机在哪 | 8 点法 / DUSt3R | 本质矩阵 / 前馈回归 | MVS、SfM、SLAM |
| 光流 | 像素怎么动 | RAFT | 4D 相关体 + GRU | 视频理解、世界模型 |
| 分割 | 哪些像素是一体 | SAM / Mask2Former | 可提示 / 统一查询 | 3D 分割、医学、编辑 |
| 法向 | 表面朝向 | Omnidata / Marigold | 多任务 / 扩散先验 | 材质、光照、重建 |

### 8.2 从基础模块到上层系统的依赖

把这张表放进更大的图景里，能看到清晰的依赖关系：

- **3D 生成**（TRELLIS、Pixal3D、LATO 等）依赖**特征提取**（图像条件编码）与**深度/法向先验**（几何约束）；
- **三维重建**（VGGT、DUSt3R）本质上是**特征 + 深度 + 位姿 + 点图**的端到端融合；
- **世界模型**（Cosmos、PlaNet）依赖**光流 / 深度**作为几何监督，**特征**作为状态编码；
- **VLA**（机器人）依赖**特征**（视觉编码）、**深度**（3D 感知）、**分割**（物体 grounding）、以及**手部/人体姿态**（灵巧操作）。

因此，本文反复强调的那个判断可以总结为一句话：**上层系统的能力上限，取决于底层基础模块的质量**。这也解释了为什么这些「看起来不性感」的基础模块，恰恰是 3D 视觉大模型时代最值得长期跟踪的底座。

---

## 9. 局限与批判性评价

1. **「基础模型化」不等于「处处可用」**。Depth Anything、DINOv2 这类模型在零样本上很强，但真到了特定下游（如室内 RGB-D 的度量精度、特定材质），仍需微调。泛化与精度的张力始终存在。
2. **几何一致性仍是被低估的缺口**。单目深度的输出在时间上（视频）和空间上（多视角）未必一致，这正是 MVS、DUSt3R、WHAM 世界坐标重建等方法试图弥补的；但「单帧基础模型 + 全局一致性」的缝合仍不成熟。
3. **评测的碎片化**。相对 vs 度量深度、相机系 vs 世界系手部重建，指标不可直接比较，读论文时极易被数字误导（本文 3.5 节已提醒）。
4. **手部重建的通用性仍落后于人体**。相比人体（已有 WHAM、SMPLify 等成熟世界系重建），手部的 in-the-wild 精度、手物交互建模、以及「世界系 + 全局一致」的手部重建仍是开放问题。
5. **本文覆盖范围的边界**：这是一篇「脉络综述」，聚焦各方向的代表性方法与范式迁移，未穷尽每个方向的全部工作（如深度估计的持续学习方法、MVS 的 Patchmatch 深度优化细节、手部的 2D 关键点最新 SOTA 等）。若要深入某个方向，应以各原论文与专门综述为准。

---

## 10. 材料来源

- CLIP：《Learning Transferable Visual Models From Natural Language Supervision》，https://arxiv.org/abs/2103.00020（配图取自 OpenAI 官方 GitHub）
- DINO：《Emerging Properties in Self-Supervised Vision Transformers》，https://arxiv.org/abs/2104.14294
- DINOv2：《DINOv2: Learning Robust Visual Features without Supervision》，https://arxiv.org/abs/2304.07193
- SAM：《Segment Anything》，https://arxiv.org/abs/2304.02643
- MiDaS：《Towards Robust Monocular Depth Estimation》，https://arxiv.org/abs/1907.01341
- DPT：《Vision Transformers for Dense Prediction》，https://arxiv.org/abs/2103.13413
- ZoeDepth：《Zero-shot Transfer by Combining Relative and Metric Depth》，https://arxiv.org/abs/2302.12288
- Metric3D：《Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image》，https://arxiv.org/abs/2307.10984
- Depth Anything：《Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data》，https://arxiv.org/abs/2401.10891
- Depth Anything V2，https://arxiv.org/abs/2407.25331
- MVSNet：《MVSNet: Depth Inference for Unstructured Multi-view Stereo》，https://arxiv.org/abs/1804.02505
- CasMVSNet：《Cascade Cost Volume for High-Resolution Multi-View Stereo》，https://arxiv.org/abs/1912.06378
- MVSFormer：《MVSFormer: Multi-View Stereo by Learning Robust Image Features and Temperature-based Depth》，https://arxiv.org/abs/2205.07514
- DUSt3R：《DUSt3R: Geometric 3D Vision Made Easy》，https://arxiv.org/abs/2312.14132
- MASt3R：《Grounding Image Matching in 3D with MASt3R》，https://arxiv.org/abs/2404.00576
- MANO：《Embodied Hands: Modeling and Capturing Hands and Bodies Together》，https://mano.is.tue.mpg.de/
- FrankMocap：《Monocular, Real-Time 3D Hand and Body Motion Capture》，https://arxiv.org/abs/2012.07098
- HaMeR：《Reconstructing Hands in Action》，https://arxiv.org/abs/2312.05251
- WHAM：《Reconstructing World-grounded Humans with Motion and Contact》，https://arxiv.org/abs/2312.07595
- RAFT：《RAFT: Recurrent All-Pairs Field Transforms for Optical Flow》，https://arxiv.org/abs/2003.12039
- Mask2Former：《Masked-attention Mask Transformer for Universal Image Segmentation》，https://arxiv.org/abs/2112.01527

> 本综述为「基础模块」方向的脉络梳理，覆盖各方向代表性工作；如需逐篇深入，请以各原论文为准。配图来源与本地文件对应关系记录于 `assets/视觉基础模块综述 - Survey/sources.md`。
