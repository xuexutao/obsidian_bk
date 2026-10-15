---
updated: 2026-10-15
---

**一句话结论：** VGGT（Visual Geometry Grounded Transformer）把"多视图三维重建"从"神经网络 + 几何后优化（BA）"的混合范式，推进到"一个前馈大 Transformer 一次前向直接输出相机位姿、点图、深度图、点轨迹"的纯网络范式。它没有花哨的 3D 归纳偏置，只靠海量 3D 标注数据 + 标准大 Transformer，就在相机估计、多视图深度、稠密点云、点跟踪四类任务上刷新 SOTA，且推理亚秒级——是 CVPR 2025 Best Paper，也是后续 DA3、π³、VGGT-Ω 一整条"前馈重建"路线的共同起点。

|         |                                                                                                                                                                                                                                                                  |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 论文名称    | **VGGT: Visual Geometry Grounded Transformer**                                                                                                                                                                                                                   |
| 归档领域    | **三维重建**                                                                                                                                                                                                                                                         |
| 重要性评估   | **★★★★★（5/5）**。理由：CVPR 2025 Best Paper；用"去除几何后优化"的姿态重新定义 3D 重建，是前馈重建路线的奠基性工作，也是后续 VGGT-Ω、DA3、π³ 等工作的直接对标对象。                                                                                                                                            |
| 会议 / 年份 | CVPR 2025（Best Paper）                                                                                                                                                                                                                                             |
| 作者 / 机构 | Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny；**Visual Geometry Group（牛津 VGG）+ Meta AI**                                                                                                                    |
| 官方出处    | [arXiv 2503.11651](https://arxiv.org/abs/2503.11651)；[官方项目页 / 代码](https://github.com/facebookresearch/vggt)                                                                                                                                                   |
| 数据来源    | 官方 PDF 全文精读（`assets/VGGT - 前馈式通用三维重建/sources/paper.pdf`）                                                                                                                                                                                               |

---

## 1. 研究背景

### 1.1 要解决的问题

传统 3D 重建长期由"视觉几何"主导：用特征匹配、对极几何、Bundle Adjustment（BA）等迭代优化手段逐步精化相机位姿和三维结构。机器学习在其中通常只扮演"补丁"角色，例如特征匹配（SuperPoint / LoFTR）或单目深度预测（Depth Anything）。即便像 VGGSfM 这样的 SOTA SfM 系统已经把可微 BA 端到端地接进了学习框架，几何优化仍然占据主导地位，带来复杂度和算力成本。

这篇论文的核心追问是：既然网络越来越强，能否干脆丢掉几何后处理，让一个神经网络直接解出所有 3D 属性？

### 1.2 前驱工作与缺口

- DUSt3R / MASt3R 已经证明"直接回归点图"这条路可行，但它们一次只能处理两帧，重建多帧时需要靠 pairwise 融合 + 后处理拼接。
- 这些方法仍逃不掉"用几何优化把两两结果粘合起来"的后处理步骤。

VGGT 的定位是再往前一步：单次前馈、同时处理一帧到数百帧、直接输出全局一致的完整 3D 属性，几乎完全不需要几何后处理。这也是论文名字里 "Grounded Transformer" 的含义——用 Transformer 把几何直接"落地"到预测结果里，而不是靠外部几何求解器。

## 2. 核心方法

### 2.1 总体设计：最小 3D 归纳偏置 + 海量数据

VGGT 的哲学是"最小 3D 归纳偏置"：不设计专门的几何网络结构，而是用一个标准的大 Transformer，靠大规模 3D 标注数据把几何能力"喂"出来。

![图1](assets/VGGT%20-%20前馈式通用三维重建/figures/fig1_teaser.png)

*图：Figure 1（论文首页 teaser）— VGGT 是一个最小 3D 归纳偏置的大前馈 Transformer，一次性接收多达数百张图像，在不到 1 秒内同时输出相机参数、点图、深度图和点轨迹，且经常在不做任何后处理的情况下就优于基于优化的替代方法。*

### 2.2 输入表示：DINO patchify + 特殊 token

输入是任意数量的图像集合。处理流程：

1. **patchify**：用 DINO 编码器把每张图切成 patch token（patch size 14×14）。
2. **追加特殊 token**：在序列中额外拼入
   - 1 个 **camera token**（承载相机估计的信息）；
   - 4 个 **register token**（用于稳定注意力、吸收全局信息，与 DINOv2 的 register 思想一致）。
3. 所有 token 一起送入主干。

### 2.3 主干：Alternating-Attention

主干是 24 层 Transformer，但注意力不是全局的，而是**交替注意力（Alternating-Attention）**：

- **frame-wise 注意力**：在单帧内部做注意力（理解单图局部结构）；
- **global 注意力**：跨所有帧、所有 token 做注意力（建立多视图对应关系）。

两层交替堆叠，兼顾"单帧局部感知"与"多视图全局关联"，同时把显存/算力控制在可训练范围。

### 2.4 多任务输出头

最终做的事情是在聚合后的Token上开出多个输出分支。比如输出Camera参数，包括Rotation、Translation等位姿信息；输出Depth；输出Tracks；以及其他各种几何相关的表征。

主干之上并行挂三个输出头，共享同一个 backbone 表征：

| 输出头                           | 结构                          | 输出                                                       |
| ----------------------------- | --------------------------- | -------------------------------------------------------- |
| **camera head**               | 4 层 self-attention + linear | 对每帧回归相机参数 $g_i = [q, t, f]$：四元数旋转 $q$ + 平移 $t$ + 视场角 $f$ |
| **dense head（DPT）**           | 类 DPT 的密集预测头                | 逐像素的深度图 / 点图 / tracking 特征，以及深度与点图各自的 aleatoric uncertainty（$\Sigma^D$、$\Sigma^P$）。展开见 [§2.4.1](#241-dense-head-展开dpt-是什么为什么需要它) |
| **tracking head（CoTracker2）** | 基于 CoTracker2               | 跨帧的 3D 点跟踪特征                                             |

> **口径提示**：论文原文只承认**两个 head**——图 2 的说明是 "A camera head makes the final prediction for camera extrinsics and intrinsics, and **a DPT head for any dense output**"。也就是说**并不存在独立的 "depth head"**，深度只是 DPT head 的一路输出分支；tracking 则是一个吃 DPT 特征的**独立模块** $\mathcal{T}$，而非挂在主干上的第三个头。上表按「功能」分三行便于对照，但引用架构时应以「camera head + DPT head」为准。

**depth 分支到底输出什么。** 两样东西，都是逐像素的：

| 输出 | 形状 | 含义 |
| --- | --- | --- |
| 深度图 $D_i$ | $\mathbb{R}^{H\times W}$ | 每个像素位置 $y\in\mathcal{I}(I_i)$ 一个深度值 $D_i(y)\in\mathbb{R}^+$，**从第 $i$ 台相机自身观察到的** |
| 不确定性图 $\Sigma^D_i$ | $\mathbb{R}_+^{H\times W}$ | aleatoric uncertainty，写进损失，训练后正比于模型置信度。见 [§2.4.2](#242-aleatoric-uncertainty让模型说出这个像素我不确定) |

**关键区别：depth 与 point map 的参考系不同。** 这一点极易混淆，但决定了两者不是同一个量的两种写法：

- **深度图 $D_i$ 是每相机自身视角下的**——$D_i(y)$ 是从相机 $i$ 看过去的距离；
- **点图 $P_i$ 是视点不变（viewpoint invariant）的**——沿用 DUSt3R 的约定，$P_i(y)\in\mathbb{R}^3$ 定义在**第一台相机 $g_1$ 的坐标系**里，$g_1$ 被取作世界参考系。

因此第一帧的外参被固定为单位变换：$q_1=[0,0,0,1]$、$t_1=[0,0,0]$。为让 Transformer 能识别出「哪一帧是参考帧」，第一帧的 camera / register token 用的是**另一组可学习参数** $\bar t_g,\bar t_R$（其余帧共享 $t_g,t_R$）。**于是网络对除第一帧以外的所有帧是置换等变的**——这正是后续 π³ 要挑战的那个「固定参考帧」假设（见 §4.2）。

**过完备预测（over-complete predictions）。** 论文明确指出这些输出**彼此并不独立**：例如相机参数 $g$ 本可以由视点不变的点图 $P$ 通过解 PnP 问题得到。既然冗余，为什么还要都预测？§3.2 的 ETH3D 对比给出了答案——**冗余反而更准**。

另外注意 register token 的输出 $\hat t^R_i$ 在预测阶段被**丢弃**，只有图像 token $\hat t^I_i$ 与 camera token $\hat t^g_i$ 参与预测。

![图2](assets/VGGT%20-%20前馈式通用三维重建/figures/fig2_architecture.png)

*图：Figure 2 — VGGT 架构总览（Architecture Overview）。输入图像经 DINO patchify 后追加 camera token 与 register token，进入 24 层 Alternating-Attention 主干；主干之上并行挂 camera head、dense head 与 tracking head，分别预测相机、点图/深度与点轨迹。*

关键点：这些任务不是独立训练的几个模型，而是共享一个 backbone、联合训练的多任务系统。论文强调，联合训练让相机、几何、跟踪三者的表征互相增强。

### 2.4.1 dense head 展开：DPT 是什么、为什么需要它

**先看矛盾在哪。** 主干吐出来的是 token：518px 的图按 14×14 patch 切开，一帧只有约 37×37 ≈ 1400 个 patch token；而 dense head 要输出的是 518×518 的逐像素深度图 / 点图。中间差了 14 倍的空间分辨率，且 token 是一维序列、显式的二维排布已经被抹掉。**DPT 就是补这个落差的解码器。**

DPT = **Dense Prediction Transformer**，出自 Ranftl, Bochkovskiy, Koltun 的《Vision Transformers for Dense Prediction》（ICCV 2021，arXiv:2103.13413）——也就是 MiDaS 团队（Intel Labs）把自家单目深度主干从 ResNeXt 换成 ViT 的那一篇。它的核心主张是：

> ViT 全程维持**恒定且相对高**的空间分辨率（token 数从头到尾不变），且**每一层都有全局感受野**——这两个性质正好是密集预测想要的；缺的只是一个把 token 还原成图的解码器。

这句话是对着卷积网络说的。FCN 系要靠逐级 downsample 才能把感受野撑大，代价是分辨率一路掉，细节在 bottleneck 处不可逆地丢失；ViT 没有这个 tradeoff。

#### Reassemble：三步把 token 摆回二维

DPT 的核心算子是 **Reassemble**，把任意一层的 token 序列还原成一张特征图：

$$\text{Reassemble}^{\hat D}_s(t) = \left(\text{Resample}_s \circ \text{Concatenate} \circ \text{Read}\right)(t)$$

其中 $s$ 是输出图相对输入图像的**尺寸比**（得到 $1/s$ 分辨率），$\hat D$ 是输出通道数。三步各自的职责：

| 步骤 | 作用 | 形状变化 |
| --- | --- | --- |
| **Read** | 处理掉那个不对应任何 patch 的 readout token | $\mathbb{R}^{(N_p+1)\times D} \to \mathbb{R}^{N_p\times D}$ |
| **Concatenate** | 按 patch 在原图中的位置，把 token 摆回二维网格 | $\mathbb{R}^{N_p\times D} \to \mathbb{R}^{\frac{H}{p}\times\frac{W}{p}\times D}$ |
| **Resample** | 重采样到目标分辨率与通道数 | $\to \mathbb{R}^{\frac{H}{s}\times\frac{W}{s}\times\hat D}$ |

**Read 的三个变体**（readout token 即 ViT 的 class token，它对密集预测没有明确用途，但可能承载有用的全局信息）：<span style="color: red">就是对ViT的class token全局token做操作的</span>

$$\text{Read}_{\text{ignore}}(t) = \{t_1,\dots,t_{N_p}\}$$
$$\text{Read}_{\text{add}}(t) = \{t_1 + t_0,\dots,t_{N_p} + t_0\}$$
$$\text{Read}_{\text{proj}}(t) = \{\text{mlp}(\text{cat}(t_1,t_0)),\dots,\text{mlp}(\text{cat}(t_{N_p},t_0))\}$$

消融（论文 Table 7，三数据集平均相对误差，越低越好）：`project` 0.0819 < `ignore` 0.0822 < `add` 0.0831。三者差距很小，但有个反直觉的细节值得记：**直接相加比干脆扔掉更差**——粗暴地把全局向量广播加到每个位置会污染局部特征，不如先 concat 再用 MLP+GELU 学一个投影。DPT 默认取 `project`。

**Resample 的实现**：先用 1×1 卷积把通道投到 $\hat D$，再根据方向选算子——$s \ge p$ 时用 strided 3×3 卷积下采样，$s < p$ 时用 strided 3×3 转置卷积上采样。

#### 四层抽取 + RefineNet Fusion

关键设计不是「取最后一层」，而是**从主干四个不同深度各抽一份，且深层配低分辨率、浅层配高分辨率**：

| 主干 | 抽取位置 | 说明 |
| --- | --- | --- |
| ViT-Base | 第 $\{3,6,9,12\}$ 层 | $D=768$ |
| ViT-Large | 第 $\{5,12,18,24\}$ 层 | $D=1024$ |
| ViT-Hybrid | ResNet50 的前两个 stage（R0, R1）+ 第 $\{9,12\}$ 层 | 低层细节直接来自卷积 embedding |

四个尺度对应 $\text{Reassemble}_4 / _8 / _{16} / _{32}$，统一输出 $\hat D = 256$ 通道。

消融（Table 6）证实「浅 + 深混搭」是必要的：ViT-Base 用 $\{3,6,9,12\}$ 平均误差 0.0822，换成集中在深层的 $\{6,8,10,12\}$ 是 0.0831、$\{9,10,11,12\}$ 是 0.0828。**只从深层抽会丢掉低层细节**——这与卷积 U-Net 需要 skip connection 是同一个道理，只是这里「该在哪里接 skip」不再由 downsample 位置天然决定（transformer 全程同分辨率），得靠实验选。

四张不同尺度的特征图之后进 **RefineNet 式的 Fusion block**（两个 residual conv unit + 逐级 2× 上采样），自低分辨率向高分辨率逐级融合——结构上就是 FPN / U-Net 的 skip-and-upsample，只不过 skip 的来源不是卷积 stage 而是 transformer layer。最终输出是输入图的 **1/2 分辨率**，再挂一个任务专用 head（深度 head：3 层卷积逐级减半通道，最后 bilinear 上采到全分辨率）。

![DPT 架构](assets/VGGT%20-%20前馈式通用三维重建/figures/dpt_fig1_architecture.png)

*图：DPT 论文 Figure 1（arXiv:2103.13413，Page 3）。**左**：整体架构——图像 embed 成 token（橙）后过多个 transformer stage，从四个不同深度 reassemble 成四种分辨率的特征图（绿），Fusion 模块（紫）逐级融合上采样，最后接 Head。**中**：$\text{Reassemble}_s$ 的三步展开（Read → Concatenate → Project/Resample）。**右**：Fusion block 内部——两个 residual conv unit + $\text{Resample}_{0.5}$（此处指 2× 上采样）+ project。*

#### 附带的好处：对输入尺寸鲁棒

DPT 只要求边长能被 patch size $p$ 整除、并对齐解码器 stride（32px），换分辨率时线性插值 position embedding 即可。论文 Figure 4 显示，把推理分辨率从训练的 384 一路推到 640，DPT-Hybrid 的性能衰减明显小于 ResNet-50 / ResNeXt-101——因为卷积网靠近输入的层是局部的、感受野被训练分辨率绑死，而 ViT 每层都全局。

**这一点正好是 VGGT 敢在训练时把 aspect ratio 在 0.33–1.0 之间随机化、并支持任意分辨率输入的前提。**

#### VGGT 里的 DPT 具体长什么样

回到 VGGT。论文 §3.3 的原文口径是：主干输出的图像 token $\hat t^I_i$ **先经一个 DPT 层转成稠密特征图** $F_i \in \mathbb{R}^{C''\times H\times W}$，然后：

- $F_i$ 各过一个 **3×3 卷积**，分别得到深度图 $D_i$ 与点图 $P_i$；
- 同一个 DPT head 还**额外吐一路稠密特征** $T_i \in \mathbb{R}^{C\times H\times W}$，作为 tracking head（CoTracker2）的输入；
- 同时预测两张不确定性图 $\Sigma^D_i,\ \Sigma^P_i \in \mathbb{R}_+^{H\times W}$。

所以上表里写的「dense head」，实质是**一个 DPT 主体 + 若干轻量输出分支**：深度、点图、tracking 特征、两张不确定性图，全部共享同一套 reassemble + fusion 解码器。这是 VGGT「共享表征」哲学在解码端的延续——连解码器都不给每个任务单独配一套。

顺带一个容易被忽略的结构含义：tracking head 的输入是 DPT 出来的稠密特征、而不是主干 token，这意味着**点跟踪的梯度会回传进 DPT 解码器**，反过来约束深度 / 点图分支的特征质量。§3.4 消融里「多任务联合训练有益」的一部分收益就来自这条通路。

> DPT 从 2021 年一路被继承到今天几乎所有前馈几何模型（DUSt3R、MASt3R、Depth Anything、VGGT、DA3……），谱系与演化见文末 [第 5 节](#5-附dpt-的谱系与发展路径)。

### 2.4.2 aleatoric uncertainty：让模型说出「这个像素我不确定」

dense head 除了深度和点图，还并行输出一张同尺寸的不确定性图。这里的 **aleatoric（偶然性）** 是不确定性二分法里的一支（Kendall & Gal, 2017）：

| 类型 | 来源 | 能否靠更多数据消除 |
| --- | --- | --- |
| **aleatoric**（偶然 / 数据） | 观测本身的固有噪声：无纹理白墙、镜面与透明物体、过曝、遮挡边界 | **不能**——信息本来就不在图里 |
| **epistemic**（认知 / 模型） | 模型没见够这类数据 | 能——加数据、加参数即可降低 |

VGGT 预测的是前者。关键在于它**不是训练完额外加的后处理，而是直接写进损失函数**（论文 §3.4）：

$$L_{\text{depth}} = \sum_{i=1}^{N} \left\lVert \Sigma^D_i \odot (\hat D_i - D_i) \right\rVert + \left\lVert \Sigma^D_i \odot (\nabla \hat D_i - \nabla D_i) \right\rVert - \alpha \log \Sigma^D_i$$

三项各司其职：

1. **误差项**被 $\Sigma$ 逐像素加权（$\odot$ 是沿通道广播的逐元素乘）；
2. **梯度项**：同样加权的深度梯度差，管边缘锐度——这是 VGGT 相对 DUSt3R 的一处增补，沿用单目深度界的常规做法；
3. **正则项** $-\alpha\log\Sigma$。

**这里有一个容易读反的方向。** 论文管 $\Sigma$ 叫 "uncertainty map"，但它在损失里是**乘**在误差上、并被 $-\alpha\log\Sigma$ 往高处拉——所以 $\Sigma$ 越大代表越**自信**（数学上它扮演的是精度 precision $\approx 1/\sigma$，而非方差）。论文自己的措辞也是：训练后 $\Sigma$ "proportional to the model's confidence in the predictions"。这套形式完全继承自 [[DUSt3R - 前馈式几何重建奠基]] 的置信度加权回归损失（那里直接叫 $C$），只是换了个名字。

**三项凑出一个自洽的博弈**：对某个像素声明「我不确定」（压低 $\Sigma$）可以减轻该处的误差惩罚，但 $-\alpha\log\Sigma$ 会因此罚它。于是模型只在**真的**没法确定的地方才降低置信度，不能靠全域摆烂来刷低 loss。

两处实际收益：

- **训练时**：无纹理区、反光面这些本质模糊的像素不会用巨大的残差污染梯度，模型不必硬拟合噪声。这一点对 VGGT 尤其重要——它的训练数据里有相当一部分 3D 标注来自 SfM 重建（见 §2.5 数据表），GT 本身在这些区域就是错的。
- **推理时**：$\Sigma$ 是可直接用的置信图。下游做点云融合、BA 精化、mesh 化时可以据此过滤或加权，而不是把所有像素当同等可信。§3.1 中「前馈 + BA」那一栏的提升，很大程度依赖这张图来筛可信的对应点。

点图损失 $L_{\text{pmap}}$ 结构完全相同，只是把 $\Sigma^D$ 换成 $\Sigma^P$。

### 2.5 训练目标与配置

**多任务损失**（各损失项加权求和）：

$$
L = L_{\text{camera}} + L_{\text{depth}} + L_{\text{pmap}} + 0.05 \cdot L_{\text{track}}
$$

- $L_{\text{camera}}$：相机损失，用 **Huber 损失**直接比较预测相机参数 $\hat g_i$ 与真值 $g_i$，即 $L_{\text{camera}} = \sum_{i=1}^{N}\lVert \hat g_i - g_i\rVert_\epsilon$。（注意：**它不带 uncertainty 加权**——uncertainty 只用在 depth 与 pmap 两项上。）
- $L_{\text{depth}}$：深度损失，uncertainty 加权 + 梯度项，展开见 [§2.4.2](#242-aleatoric-uncertainty让模型说出这个像素我不确定)。
- $L_{\text{pmap}}$：点图损失，结构同上，换用 $\Sigma^P$。
- $L_{\text{track}}$：点跟踪损失 $\sum_j\sum_i \lVert y_{j,i} - \hat y_{j,i}\rVert$，权重 $\lambda = 0.05$；另按 CoTracker2 加一项可见性损失（二元交叉熵）。

论文特别说明：camera / depth / pmap 三项的数值范围本就相近，**不需要互相加权**，只有 tracking 需要降权到 0.05。

**GT 坐标归一化**（一个容易漏掉但很关键的设计）：先把所有量转到第一台相机 $g_1$ 的坐标系，再用「点图中所有 3D 点到原点的平均欧氏距离」作为尺度，归一化相机平移、点图与深度。与 DUSt3R 不同的是，**VGGT 不对网络输出做这个归一化**，而是强迫 transformer 从数据里学会输出这个特定的归一化变体。

**训练配置**：

| 配置项  | 细节                              |
| ---- | ------------------------------- |
| 参数规模 | 约 **1.2B**（1215M）               |
| 训练数据 | 大规模 3D 标注数据混合（含 SfM 重建数据、合成数据等） |
| 帧数   | 每次 2–24 帧                       |
| 分辨率  | 518px                           |
| 优化器  | AdamW                           |
| 训练规模 | 160K iterations，64×A100，约 9 天   |

## 3. 实验与结果

### 3.1 相机位姿估计

在多视图相机估计基准上，VGGT 的前馈输出（不做 BA）就已达到甚至超过此前需要几何优化的方法；再叠一层 BA 还能进一步提升。

| 数据集 | AUC@30（前馈） | AUC@30（+BA） |
| --- | --- | --- |
| RealEstate10K | 85.3 | 93.5 |
| CO3Dv2 | 88.2 | 91.8 |

### 3.2 深度与稠密点云重建

**DTU 稠密 MVS（Table 2，越低越好）**——注意 VGGT 属于「不知道 GT 相机」的下半张表：

| 方法 | 是否已知 GT 相机 | Acc.↓ | Comp.↓ | Overall↓ |
| --- | --- | --- | --- | --- |
| MASt3R | ✓ | 0.403 | 0.344 | 0.374 |
| GeoMVSNet | ✗ | 0.331 | 0.259 | 0.295 |
| DUSt3R | ✗ | 2.677 | 0.805 | 1.741 |
| **VGGT** | ✗ | 0.389 | 0.374 | **0.382** |

**ETH3D 点图估计（Table 3，每场景随机 10 帧，Umeyama 对齐后按官方 mask 过滤）**——这张表是本节最有信息量的：

| 方法 | Acc.↓ | Comp.↓ | Overall↓ | 时间 |
| --- | --- | --- | --- | --- |
| DUSt3R（含全局对齐） | 1.167 | 0.842 | 1.005 | ~7 s |
| MASt3R（含全局对齐） | 0.968 | 0.684 | 0.826 | ~9 s |
| VGGT（Point，直接用点图头） | 0.901 | 0.518 | 0.709 | **~0.2 s** |
| **VGGT（Depth + Cam，深度反投影）** | **0.873** | **0.482** | **0.677** | **~0.2 s** |

两个结论：

**其一，前馈就打赢了带全局对齐的方法，且快 35–45 倍。** DUSt3R / MASt3R 每个场景要做约 10 秒的全局对齐优化，VGGT 一次前向 0.2 秒。

**其二——也是更有意思的一点——「深度头 + 相机头反投影」比「直接用点图头」更准**（0.677 vs 0.709）。论文自己给出的解释是：

> we found that the predictions from our depth and camera heads (i.e., **unprojecting** the predicted depth maps to 3D using the predicted camera parameters) yield higher accuracy. We attribute this to the benefits of **decomposing a complex task (point map estimation) into simpler subproblems (depth map and camera prediction)**, even though camera, depth maps, and point maps are jointly supervised during training.

换句话说：**把一个复杂任务拆成两个更简单的子问题（预测深度 + 预测相机），再用几何关系组合起来，比让网络直接端到端回归复合量更准**——即使三者在训练中是联合监督的。这正是 §2.4 里「过完备预测」的价值所在：冗余不是浪费，它给了推理时选择更优组合路径的自由度。这也是实际用 VGGT 时的一条工程建议——**要点云优先走 depth + camera 反投影，而不是直接取 point map**。

**ScanNet-1500 两视图匹配（Table 4）**：AUC@5 / @10 / @20 = **33.9 / 55.2 / 73.4**，超过专门做两视图匹配的 SOTA 方法 Roma（31.8 / 53.4 / 70.9）——而 VGGT 的 tracking head 并非为两视图场景特化设计。

在点云重建（point map）这一前馈方法的"主战场"上，VGGT 明显超越 DUSt3R / MASt3R 等两帧方法，且不需要 pairwise 融合。

![图3](assets/VGGT%20-%20前馈式通用三维重建/figures/fig3_qualitative.png)

*图：Figure 3 — 定性对比。VGGT 在多视图深度与点云重建上与同类方法（如 DUSt3R / MASt3R）对比，展现出更干净、更一致的稠密几何，尤其是大视角变化和动态区域下的稳定性。*

### 3.3 下游任务迁移

VGGT 的预训练表征可以直接赋能下游任务，无需重新训练：

- 前馈新视角合成（NVS）：在 GSO 数据集上，用 VGGT 特征做前馈 NVS，PSNR 达 30.41。
- 非刚体点跟踪：把 VGGT 特征接进 **CoTracker**，动态场景下的点跟踪可见性指标（$\delta_{\text{vis}}$）显著提升。

这说明 VGGT 学到的表征**可迁移、有通用性**，不只是"会重建"。

### 3.3.1 在 FantasyWorld、Lyra 与 DA3 下游中的实际角色边界

后续论文频繁使用“VGGT-style”或把 VGGT 放进几何实验，容易把**架构影响、评测工具与真正的 backbone 依赖**混为一谈。三类关系应分开：

| 后续工作 | VGGT 的实际角色 | 是否直接以 VGGT 为 backbone / 核心依赖 | 与 DA3 下游角色的区别 |
|---|---|---|---|
| **FantasyWorld** | 几何侧采用 **VGGT-style geometry branch**：复用“多层几何特征 + camera head / 3D DPT heads，联合预测 depth、point map、uncertainty 与 pose”的设计语言；但主干仍是冻结的 **Wan2.1 video diffusion**，几何分支从 WanDiT 中间特征分叉并与想象分支双向 cross-attention | **否。**“VGGT-style”表示架构借鉴，不等于加载 VGGT 主干或权重。论文的 3DGS 表格又单独使用外部 **VGGT 预测初始化 3DGS 优化**；这是评测管线中的独立初始化，用于比较有/无 3D branch 的生成视频，不是 FantasyWorld 在线前向的一部分 | FantasyWorld 对 VGGT 主要是“设计范式 + 外部评测初始化”；DA3 式下游则把几何模型本体放进任务管线并在其上接 head / 微调 |
| **Lyra 系列** | VGGT 主要是前馈几何路线中的相关工作、重建质量参照或可替换 benchmark，而不是生成系统的内部骨干 | **否。** Lyra 1.0 使用视频扩散自蒸馏与自家 3DGS 解码器；Lyra 2.0 的生成核心是 Wan2.1 + FramePack + 空间记忆，几何预处理/导出分别依赖 ViPE 与 **Depth Anything 3（DA3）**，并未把 VGGT 接入核心生成循环 | 在 Lyra 2.0 中，DA3 被实际采用为前馈 3DGS 几何基座：修改 Gaussian DPT head、在自生成视频上微调后参与最终资产导出；这比“拿 VGGT 做相关基准”是更强的工程依赖 |
| **DA3 自身的下游验证** | VGGT 是被比较和被替代的前代几何 backbone | **不是 VGGT 依赖。** DA3 用 vanilla backbone + depth-ray 目标建立自己的几何基座 | DA3 的 FF-NVS / FF-3DGS 直接在 **DA3 backbone** 上挂 GS-DPT head 并微调，且 Lyra 2.0 也直接消费其几何能力；这是“几何 backbone → 下游任务”的参数与特征复用，而非仅借鉴 VGGT-style 结构 |

因此，VGGT 对生成式世界模型的影响应写成：它确立了**前馈、多视图共享表征、camera + dense geometry 多头预测**这一设计范式，并提供强几何基准；不能据此推断 FantasyWorld 或 Lyra“以 VGGT 为 backbone”。判断是否构成核心依赖，要看 VGGT 的参数/特征是否进入主前向与训练图：FantasyWorld 中没有，Lyra 系列中也没有；反而 DA3 在 Lyra 2.0 的 3DGS 导出和自身 FF-NVS 下游中满足这一标准。

### 3.4 消融

**主干注意力设计（Table 5，ETH3D，越低越好）**——这是全文最干净的一个消融：

| 主干变体 | Acc.↓ | Comp.↓ | Overall↓ |
| --- | --- | --- | --- |
| Cross-Attention | 1.287 | 0.835 | 1.061 |
| Global Self-Attention Only | 1.032 | 0.621 | 0.827 |
| **Alternating-Attention** | **0.901** | **0.518** | **0.709** |

交替注意力显著优于「纯 global self-attention」（0.709 vs 0.827）与「cross-attention」（1.061）。注意纯 global 变体的算力其实并不更省——**它输的不是效率而是精度**，说明「先在单帧内部整理局部结构、再跨帧建立对应」这个交替归纳偏置本身是有价值的，不只是显存妥协。

- 多任务联合训练有益：相机、深度、点图、跟踪四个损失联合训练，比单独训练某几个任务效果更好——验证了"共享 backbone + 联合监督"的设计。（一条容易忽略的通路见 §2.4.1 末尾：tracking 的梯度经 DPT 解码器回传，反过来约束深度/点图分支的特征质量。）

### 3.5 效率

- 200 帧输入：推理约 8.75s，显存约 40.63GB（单卡）。
- 亚秒级处理单张/少量帧，这为"前馈重建"走向实时应用提供了可能。

## 4. 局限与评价

### 4.1 论文自述的局限

- **不支持鱼眼 / 全景相机**：假设针孔相机模型。
- **极端旋转退化**：在大角度旋转下，相机估计会退化。
- **严重非刚体形变失效**：对剧烈非刚体运动（如大幅形变的物体）处理能力不足。

### 4.2 个人评价

VGGT 的最大贡献**不是某个指标，而是一次范式切换**：它用"一个前馈大模型"替代"网络 + BA"的混合管线，证明了只要数据够、模型够大，3D 几何也能被 Transformer 直接"grounded"出来。这直接打开了后续一整条路线：

- **VGGT-Ω**（同库已精读）沿着 scaling 方向继续放大模型与数据；
- <span style="color: red">DA3 用更小的模型（0.30B）反超 VGGT，说明"最小建模"还有空间；</span>
- **π³** 指出 VGGT 的"固定参考帧"假设是一个隐患，用置换等变架构去除它。

所以读 VGGT，本质上是读这条路线的问题起点和方法基线。它确立了几个关键判断：**前馈可替代优化、共享 backbone 联合多任务有效、最小 3D 归纳偏置可行**——这些后来被 DA3、π³ 在各自方向上进一步验证或挑战。

### 4.3 复现成本与建议

- **复现门槛高**：约 1.2B 参数、64×A100 训练 9 天，个人复现成本极高；官方已开源代码与模型，推荐直接用预训练权重做推理与下游验证。
- **建议关注点**：camera head 的四元数 + 视场角参数化细节、Alternating-Attention 的具体层交替方式、多任务损失的 uncertainty 加权实现，是理解"为什么能前馈"的关键。

**最终评价：** VGGT 是前馈式三维重建的奠基性工作，CVPR 2025 Best Paper 实至名归。它最重要的价值是把"几何重建"从优化问题改写为回归问题，并证明这条路可 scale。后续所有"前馈重建"论文，本质上都在回答 VGGT 留下的问题：能不能更小、更鲁棒、更少归纳偏置、更能处理动态与任意参考系。

---

## 5. 附：DPT 的谱系与发展路径

DPT 是本文 §2.4.1 展开的那个解码器。把它单独拉一条时间线出来，是因为**它可能是整个 3D 视觉里被复用得最彻底、却最少被单独讨论的组件**——从 2021 到今天，几乎每一个前馈几何模型的输出端都是它。

### 5.1 上游：DPT 从哪来

DPT 不是凭空出现的，它是两条线的交汇：

| 上游                                | 贡献给 DPT 的东西                                              |
| --------------------------------- | -------------------------------------------------------- |
| **FCN**（Long et al., 2015）        | 「密集预测 = encoder-decoder + 逐级上采样」的基本骨架                    |
| **U-Net / FPN**（2015 / 2017）      | skip connection：浅层细节必须绕过 bottleneck 直连解码器                |
| **RefineNet**（Lin et al., 2017）   | DPT 的 Fusion block 直接沿用它的 residual conv unit 设计          |
| **MiDaS**（Ranftl et al., 2020）    | 同一批作者的前作。提供了任务、数据配方（多源混合 + 尺度平移不变损失）与被替换的基线（ResNeXt-101） |
| **ViT**（Dosovitskiy et al., 2021） | 恒定分辨率 + 全层全局感受野的主干，是 DPT 立论的前提                           |

所以 DPT 的定位可以一句话概括：把 U-Net 的解码思路移植到一个不做 downsample 的主干上。难点也正在这——ViT 全程同分辨率，「该在哪儿接 skip」失去了天然答案，于是变成一个需要消融决定的超参（论文 Table 6）。

### 5.2 主线：DPT 之后的演化

```
FCN / U-Net / RefineNet ──┐
                          ├──> DPT (2021) ──> 成为 ViT 密集预测的默认解码器
MiDaS (2020) ─────────────┤       │
ViT (2021) ───────────────┘       │
                                  ├──> 单目深度线：MiDaS v3 / Omnidata → Depth Anything V1/V2 → DA3
                                  ├──> 多视图几何线：DUSt3R → MASt3R → VGGT → π³ / VGGT-Ω
                                  └──> 其他密集任务：法向、分割、3DGS 参数、光流……
```

**分线来看：**

**(1) 单目深度线** —— DPT 的原生战场，主干越换越强，解码器基本不动。

| 年份   | 工作                    | 相对 DPT 的变化                                                                      |
| ---- | --------------------- | ------------------------------------------------------------------------------- |
| 2021 | **DPT / MiDaS v3**    | 原始形态。ViT 主干 + Reassemble + RefineNet Fusion                                     |
| 2021 | **Omnidata**          | 同一套 DPT 架构，换成大规模多任务合成数据训练                                                       |
| 2024 | **Depth Anything**    | 主干换 DINOv2，解码器仍是 DPT；靠 6200 万无标注图伪标签自训练把单目深度推到基础模型规模                            |
| 2024 | **Depth Anything V2** | 同上，改用合成数据 + 更小模型缓解伪标签噪声                                                         |
| 2025 | **DA3**               | **Dual-DPT head**：共享 reassembly + 两套独立 fusion（depth 分支 / ray 分支），一个解码器骨架输出两种几何量 |

这条线的规律很清楚：**主干从 ViT → DINOv2，数据从 140 万 → 6200 万，但解码器七年没换。** DPT 在这个位置上大概是「够好且没有明显瓶颈」。

**(2) 多视图几何线** —— DPT 被搬去输出点图，逐步长出更多分支。

| 年份   | 工作                             | DPT 在其中的角色                                                                                |
| ---- | ------------------------------ | ----------------------------------------------------------------------------------------- |
| 2024 | **[[DUSt3R - 前馈式几何重建奠基]]**     | 第一次把 DPT 用于**点图**而非深度：每像素输出 4 维（3D 坐标 + confidence）。**置信度加权损失也在此定型**，VGGT 的 $\Sigma$ 直接继承 |
| 2024 | **[[MASt3R - 匹配与重建统一]]**       | DPT head 上再加一个 local feature 分支，让同一个解码器同时供给「重建」与「匹配」                                      |
| 2025 | **VGGT**（本文）                   | DPT 成为**唯一**的 dense 解码器：深度 + 点图 + tracking 特征 + 两张 $\Sigma$，四类输出共享一套 reassemble/fusion    |
| 2025 | **[[π³ - 置换等变视觉几何学习]]**        | 沿用 DPT，但去掉 VGGT 的固定参考帧假设                                                                  |
| 2025 | **[[VGGT-Ω - 可扩展前馈式静动态三维重建]]** | 沿用 DPT，向 scaling 与动态场景扩展                                                                  |

值得注意的是 **DUSt3R 这一环的分量**：它不只是「用了 DPT」，而是确立了「DPT head 每像素多吐一个 confidence、并把它写进损失」这个范式。VGGT §2.4.2 那套 $\Sigma$ 机制，血缘在此而不在 DPT 原论文——**DPT 原文没有不确定性输出**，那是 DUSt3R 加的。

**(3) 泛化到任意密集量** —— 这是 DPT 生命力最强的证据。只要输出是「逐像素的某个量」，就可以换个 head 接上去：法向图（Omnidata / Marigold 系）、分割 logits（DPT 原论文 ADE20K 49.02 mIoU）、3DGS 逐像素参数（DA3 的 GS-DPT head，输出不透明度 / 四元数 / 尺度 / 颜色）、点跟踪特征（VGGT）。

### 5.3 为什么它能活这么久

三个原因，按重要性排：

1. **它解的是结构性矛盾，不是特定任务。** 「token 分辨率低、输出要逐像素」这个落差，只要主干是 ViT 就一直存在，与你预测深度还是法向无关。
2. **主干可替换。** DPT 对主干只有一个要求——能按层抽 token。所以 ViT → DeiT → DINOv2 → 自定义交替注意力主干（VGGT），解码器都不用改。这让它能免费搭上主干进步的便车。
3. **输出可扩展。** 加一个新的预测量只需在 fusion 输出后挂一个 3×3 卷积（VGGT 的做法），边际成本极低。这直接支撑了「多任务共享 backbone」这条路线。

### 5.4 局限与松动的迹象

DPT 也不是没有代价，且已经出现被绕开的信号：

- **输出天生只有 1/2 分辨率**，末端靠 bilinear 上采补齐——细边界（发丝、栏杆）存在结构性上限。
- **解码器是纯卷积的**，全局一致性完全依赖主干。跨视图一致性这类需求，DPT 自己给不了（VGGT 靠交替注意力在主干里解决）。
- **四层抽取位置是超参**，换主干就得重新消融，缺乏原则性依据。
- **有工作开始不用它了**：DA3 论文的一个核心论点是「最小化建模」——它虽然保留 Dual-DPT，但整体方向是削减专用结构。更激进的一支（如部分 3DGS 前馈工作）直接用 token 级预测 + 上采样替代整个卷积解码器。**如果哪天 ViT 的 patch size 降到 4 或更小、或 pixel-level transformer 变得可负担，DPT 存在的前提（分辨率落差）就消失了。**

**判断**：DPT 属于那种「解决了一个过渡期矛盾、因而与该矛盾共存亡」的组件。它会一直用到主干本身能直接输出高分辨率为止。作为读者，理解它的价值不在于记住 Reassemble 的三步，而在于看清**「低分辨率表征 → 高分辨率输出」这个落差在每一代模型里是怎么被填的**——这个视角能直接迁移到视频生成的 latent upsampler、3DGS 的 densification 等一系列看似无关的设计上。

### 5.5 DPT 相关来源

| 材料 | 链接 / 路径 |
| --- | --- |
| DPT 论文（ICCV 2021） | [arXiv:2103.13413](https://arxiv.org/abs/2103.13413) |
| 本地 PDF（本节精读依据） | `assets/VGGT - 前馈式通用三维重建/sources/dpt_2103.13413.pdf` |
| 官方代码（MiDaS 仓库内） | https://github.com/isl-org/DPT |
| MiDaS（前作，数据与损失配方） | [arXiv:1907.01341](https://arxiv.org/abs/1907.01341) |
| RefineNet（Fusion block 来源） | [arXiv:1611.06612](https://arxiv.org/abs/1611.06612) |
| Kendall & Gal（aleatoric / epistemic 二分） | [arXiv:1703.04977](https://arxiv.org/abs/1703.04977) |

vault 内相关：[[视觉基础模块综述 - Survey]] §3.2 有 MiDaS / DPT 在单目深度谱系中的定位；[[前馈式三维重建与生成式补全 - Survey]] 覆盖多视图这条线的整体版图。
