---
type: paper-note
status: complete
domain: Foundations
paper: "DINOv3"
year: 2025
arxiv: "2508.10104"
doi: null
source: https://arxiv.org/abs/2508.10104
project: null
code: null
tags:
  - self-supervised-learning
  - vision-foundation-model
  - dense-features
  - gram-anchoring
  - visual-encoder
  - VLM-backbone
created: 2026-10-15
updated: 2026-10-15
---

# DINOv3：大规模自监督视觉基础模型

## 基本信息

| 项目 | 内容 |
|---|---|
| 正式名称 | **DINOv3** |
| 作者 / 机构 | Oriane Siméoni、Huy V. Vo、Maximilian Seitzer 等；Meta AI Research、WRI、Inria |
| 论文 | [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) |
| 归档领域 | 基础模块 / 视觉基础模型 |
| 核心关键词 | 大规模自监督学习、稠密视觉特征、Gram anchoring、7B ViT、蒸馏、高分辨率适配、文本对齐 |
| 本地原文 | `assets/DINOv3 - 大规模自监督视觉基础模型/sources/paper.pdf` |
| 重要性评估 | **★★★★★（5/5）**：它不仅扩大 DINO 系列规模，更定位并修复了“大模型长训练使稠密特征变差”的关键失效模式，使自监督视觉编码器同时具备强全局语义与强局部几何。 |

**一句话结论：**

DINOv3 把纯图像自监督预训练扩展到 **17B 原始图像池、1.689B 主训练集和 6.7B 参数 ViT**，并用 **Gram anchoring** 约束 patch 两两相似关系，修复长训练中稠密特征从“清晰局部化”退化为“全局混杂噪声”的问题；由此得到一个冻结后即可服务分割、深度、跟踪、匹配、检测和三维几何的通用视觉骨干。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

DINOv2 已证明，不依赖人工标签或图文元数据的自监督学习（SSL）可以训练出兼顾全局分类和局部视觉任务的通用特征。但把这一路线继续扩展到更大数据、更大模型、更长训练时，出现三个系统性问题：

1. **无标签数据不是越多越好**：原始网络图像分布极不均衡，直接扩大数据量会放大重复、长尾缺失和领域偏差。
2. **固定训练日程不适合 frontier-scale SSL**：DINOv2 的多组 cosine schedule 要预先知道训练终点，大规模实验却很难事先判断何时收敛。
3. **全局能力与稠密能力发生分叉**：ImageNet 线性分类持续变好，但语义分割、深度、patch 匹配在约 200K steps 后开始变差；模型越大、训练越久，问题越明显。

论文真正要回答的不是“能否再训练一个更大的 DINO”，而是：**能否让自监督视觉模型像大语言模型一样持续从 scaling 获益，同时不牺牲空间局部性？**

### 1.2 为什么稠密特征比分类精度更关键

分类只消费 CLS token 或全局池化向量；分割、深度、视频跟踪、关键点匹配、机器人操作、3D 重建和 VLM 的区域级 grounding，则依赖每个 patch 的表示。一个模型即使 ImageNet 准确率很高，如果不同位置的 patch 逐渐变得彼此相似，仍会失去“这个像素属于哪里、和哪一处对应”的能力。

![](assets/DINOv3%20-%20大规模自监督视觉基础模型/figures/01-ssl-and-dense-features.png)

> **图 1：SSL 的演进与 DINOv3 的稠密特征定位。** 左上显示 SSL 在 ImageNet 线性探测上已追上监督/弱监督方法；左下显示 DINOv3 在深度、跟踪、分割上显著超过最佳弱监督骨干；右侧自然图像和遥感图像的 PCA 特征图说明无标签预训练能形成清晰、跨领域的空间结构。来源：论文 Figure 1，p.2。

### 1.3 核心贡献

1. **可扩展的数据与优化配方**：从约 17B 网络图像中构建 LVD-1689M，并混合检索数据和公共视觉数据；取消主训练阶段的 cosine 日程，采用可继续延长的常数超参数训练。
2. **6.7B 参数自监督视觉教师**：构建 40 层、4096 维、patch size 16、带 axial RoPE 与 register tokens 的 ViT-7B，将 DINO+iBOT 路线扩展到 7B 量级。
3. **Gram anchoring**：用早期、局部性更好的教师约束 student patch 的 Gram matrix，从关系结构而不是特征坐标本身恢复局部一致性。
4. **高分辨率适配与多学生蒸馏**：把 7B 教师压缩到 ViT-S/B/L/H+ 和 ConvNeXt 系列，并让模型在远高于训练分辨率的输入上保持稳定。
5. **冻结骨干的广泛验证**：在分割、深度、3D 对应、视频跟踪、检测、VGGT 等任务上，仅训练轻量 probe/decoder 或直接做非参数推理即可达到强结果。

---

## 2. 任务定义与输入输出

### 2.1 预训练输入与输出

- **输入**：无标签 RGB 图像。每张图产生 2 个 global crops（256×256）和 8 个 local crops（112×112）。
- **学生网络输出**：CLS token 与 patch tokens；部分 global crop 的 patch token 会被遮挡。
- **EMA 教师输出**：只处理 global crops，提供图像级与可见 patch 级自蒸馏目标。
- **最终输出**：一个通用视觉编码器。对任意分辨率图像，它输出全局表示与空间排列的稠密 patch 特征；默认下游使用冻结 backbone。

这与分类式预训练不同：DINOv3 的目标不是直接输出类别，而是学习可被不同任务头重用的视觉表示。

### 2.2 初始自监督目标

主训练沿用并扩展 DINOv2 的三类信号：

$$
\mathcal{L}_{\text{Pre}}
= \mathcal{L}_{\text{DINO}}
+ \mathcal{L}_{\text{iBOT}}
+ 0.1\,\mathcal{L}_{\text{DKoleo}}.
$$

其中：

- $\mathcal{L}_{\text{DINO}}$ 对齐不同视图的 CLS token，学习图像级不变语义；
- $\mathcal{L}_{\text{iBOT}}$ 用教师可见 patch 监督学生的 masked patch，承担局部表征学习；
- $\mathcal{L}_{\text{DKoleo}}$ 鼓励 batch 内 CLS 特征在表示空间均匀展开，避免坍缩。

DINO 与 iBOT 均使用独立 projection head，并以 Sinkhorn-Knopp 替代传统 centering。专门的 layer normalization 用在 global/local crop 的 backbone 输出上，论文报告它约带来 ADE20K **+1 mIoU**、NYUv2 **-0.02 RMSE**。

---

## 3. 核心方法

### 3.1 数据 scaling：多目标数据治理而非单一过滤器

初始池来自公开 Instagram 帖子，约 **17B** 张图像。论文构建三部分数据：

1. **聚类均衡数据 LVD-1689M**：以 DINOv2 embedding 做五级 hierarchical k-means（200M → 8M → 800K → 100K → 25K clusters），再平衡采样，得到 1.689B 图像，目标是覆盖网络世界的视觉多样性。
2. **检索数据**：以精选 seed datasets 为查询，从大池中检索相似图像，增强与常见下游概念的相关性。
3. **公共视觉数据**：ImageNet-1K、ImageNet-22K、Mapillary 等，用少量高质量或任务相关分布校准模型。

训练时，90% 情况使用除 ImageNet-1K 外的混合异质 batch，10% 使用纯 ImageNet-1K 同质 batch。Table 1 表明 raw、聚类、检索各自在不同任务上有优势，而组合数据在 IN1K k-NN/Linear、ObjectNet、iNaturalist、Paris retrieval 上取得最均衡结果。其方法论含义是：**“多样性覆盖”与“下游有用性”不是同一个数据目标，需要组合优化。**

### 3.2 模型 scaling 与可延长训练

DINOv3 教师从 DINOv2 的 1.1B ViT-g 扩为 **6.7B ViT-7B**：仍为 40 blocks，但 embedding 从 1536 扩为 4096，attention heads 从 24 增至 32，单头维度从 64 增至 128，patch size 由 14 改为 16，位置编码改为 RoPE，保留 4 个 register tokens。

为让模型适配分辨率和长宽比，论文引入 **RoPE-box jittering**：patch 坐标原本规范化到 $[-1,1]$，训练时随机缩放为 $[-s,s]$，其中 $s\in[0.5,2]$。这迫使模型弱化对固定绝对尺度的依赖。

优化上取消主训练阶段的周期调度，学习率、weight decay、teacher EMA momentum 均保持常数，仅在起始阶段 warmup。主训练配置为：

- 256 GPUs、global batch 4096；
- 每 batch 约 3.7M tokens；
- AdamW，bfloat16 与 FP8 matrix multiplication；
- 1M steps，常数学习率 $4\times10^{-4}$，100K warmup；
- weight decay 0.04，layer-wise LR decay 0.98，stochastic depth 0.4，EMA 0.999。

这种设计把“训练多久”从预设超参数变成可由下游性能决定的开放过程，但也暴露了稠密特征退化。

### 3.3 失效诊断：局部一致性为何随长训练消失

论文观察到 ViT-g 和 ViT-7B 的 ImageNet 线性分类持续改善，而 VOC segmentation 在约 200K steps 后下降。patch cosine similarity map 显示：早期参考 patch 只与同一物体/区域响应；600K—1M steps 后，大量无关位置被赋予高相似度，空间定位逐渐噪声化。

![](assets/DINOv3%20-%20大规模自监督视觉基础模型/figures/02-dense-feature-degradation.png)

> **图 2：长训练中的稠密特征退化。** 200K 时相似度集中于参考物体局部；到 600K—1M 时，高响应扩散到无关 patch，花朵轮廓和羊的局部结构明显破碎。来源：论文 Figure 6，p.11。

这不是 DINOv2 论文所讨论的 high-norm outlier：register token 已让 patch norm 保持稳定。真正变化是 **CLS 与 patch 越来越相似，全局判别目标逐步压过局部结构目标**。换言之，iBOT 并不足以长期稳定地约束 patch 之间的几何关系。

### 3.4 Gram anchoring：约束“关系结构”，不冻结特征坐标

给定一幅图像的 $P$ 个 patch，student 与 Gram teacher 的 L2-normalized patch feature 分别组成：

$$
X_S, X_G \in \mathbb{R}^{P\times d}.
$$

Gram matrix $XX^\top\in\mathbb{R}^{P\times P}$ 记录图像内任意两个 patch 的点积，也就是完整的两两相似关系。DINOv3 定义：

$$
\mathcal{L}_{\text{Gram}}
= \left\|X_SX_S^\top-X_GX_G^\top\right\|_F^2.
$$

这里的 Gram teacher 取自较早训练阶段，因为它的局部一致性更好。核心优点是：该损失**不要求 student 逐 patch 复制 teacher 的绝对向量**；只要 patch 间关系保持，整个特征空间仍可旋转、迁移并继续学习新的全局语义。这比直接 feature regression 更少束缚模型容量。

在 1M steps 后才启动修复阶段：

$$
\mathcal{L}_{\text{Ref}}
= w_D\mathcal{L}_{\text{DINO}}
+ \mathcal{L}_{\text{iBOT}}
+ w_{DK}\mathcal{L}_{\text{DKoleo}}
+ w_{\text{Gram}}\mathcal{L}_{\text{Gram}},
$$

其中 $w_{\text{Gram}}=2$。Gram teacher 每 10K steps 更新一次，最多三次，使锚点随修复后的 EMA teacher 前移。尽管在非常晚的 1M steps 才加入，该目标仍能“修复”已经严重退化的局部特征；同时 iBOT loss 会更快下降，而 DINO global loss 基本不受影响，支持“局部关系与全局语义可相对独立优化”的判断。

### 3.5 高分辨率 Gram teacher

普通 student 仍处理 256 分辨率，但 Gram teacher 接收 2× 分辨率图像（512），再将其 feature map 以 bicubic 方式下采样回 student 尺寸。高分辨率先捕获更细局部结构，下采样再平滑异常 patch；最终用这组特征计算 $X_GX_G^\top$。论文记该目标为 $\mathcal{L}_{\text{HRef}}$。

Figure 9 的消融显示：

| 设置 | IN1K Linear ↑ | ADE20K mIoU ↑ | NYUv2 RMSE ↓ |
|---|---:|---:|---:|
| Baseline | **88.2** | 50.3 | 0.307 |
| Gram，200K teacher，1× | 88.0 | 53.6 | 0.285 |
| Gram，200K teacher，2× | 88.0 | **55.7** | **0.281** |
| Gram，100K teacher，2× | 87.9 | **55.7** | 0.284 |
| Gram，1M teacher，2× | 88.1 | 54.9 | 0.290 |

早期 100K/200K teacher 差异不大；1M teacher 因自身已经退化而变差。该表也清楚展示了 trade-off：Gram anchoring 的主要收益在稠密任务，ImageNet 线性分类几乎不变且可能轻微下降 0.2。

![](assets/DINOv3%20-%20大规模自监督视觉基础模型/figures/03-gram-anchoring-results.png)

> **图 3：Gram anchoring 的定量恢复。** 在 1M steps 后加入 $\mathcal{L}_{\text{Ref}}$，VOC 与 ADE20K mIoU 在前 10K 修复 steps 内迅速反弹；高分辨率 Gram 目标 $\mathcal{L}_{\text{HRef}}$ 进一步提升稠密性能，而 ObjectNet 全局准确率保持稳定。来源：论文 Figure 8，p.13。

![](assets/DINOv3%20-%20大规模自监督视觉基础模型/figures/04-gram-anchoring-recovery.png)

> **图 4：Gram anchoring 的定性恢复。** 中间行是未做高分辨率 Gram 修复的噪声相似度图，底行重新形成围绕花朵、鸟、食物、海豹和羊群的连贯局部响应。它说明修复的不是“更锐的像素”，而是 patch 关系图重新获得对象级结构。来源：论文 Figure 10，p.14。

### 3.6 后训练：分辨率适配、蒸馏与文本对齐

#### 3.6.1 高分辨率适配

主模型仅在 256/112 crops 上训练，随后增加 10K steps 混合分辨率适配：global crop 从 {512, 768} 采样，local crop 从 {112, 168, 224, 336} 采样，Gram teacher 可到 768/1152。Gram anchoring 在此阶段仍不可缺少，否则高分辨率稠密性能会再次下降。适配后模型在训练未见过的 4K 以上输入仍能形成稳定 feature map。

#### 3.6.2 多学生蒸馏

固定 ViT-7B 作为 teacher，同时蒸馏 ViT-S（21M）、S+（29M）、B（86M）、L（0.3B）、H+（0.8B）以及 ConvNeXt T/S/B/L。由于固定教师已经具备稳定 patch 结构，蒸馏阶段不再需要 Gram anchoring。论文还共享一次昂贵的 teacher inference 给多个 student group，只为每个新增 student 支付自己的训练计算，从而摊薄 7B teacher 成本。

#### 3.6.3 文本对齐

DINOv3 本体是纯视觉编码器，不原生理解文字。论文遵循 LiT/dino.txt：冻结视觉 backbone，从头训练 text encoder，并在视觉侧增加两层 transformer；将 mean-pooled patch embedding 与 CLS token 拼接后对齐文本。这样既保持全局图文检索，又把局部 patch 带入文本对齐。

---

## 4. 数据集与实验设置

### 4.1 评估维度

论文将“通用视觉表示”拆成三层验证：

1. **轻量 probing / 非参数评估**：冻结最后层 patch，只训练线性层或直接相似度匹配；覆盖 ADE20K、Cityscapes、VOC、NYUv2、KITTI、NAVI、SPair、DAVIS、YouTube-VOS、MOSE。
2. **全局表征**：ImageNet、ObjectNet、细粒度分类、实例检索等。
3. **完整下游系统**：在冻结 backbone 上训练 Plain-DETR、Mask2Former/DPT 等 decoder，或替换 VGGT 的 DINOv2 backbone。

### 4.2 主要 baseline

- **自监督**：DINOv2、Web-DINO、Franca；
- **弱监督 / 图文监督**：SigLIP 2、Perception Encoder（PEcore）；
- **聚合蒸馏**：AM-RADIOv2.5、PEspatial（含 SAM/SAM 2 的空间监督）；
- **任务专用系统**：Depth Anything V2、VGGT、EVA-02/InternImage 检测与分割系统。

对比的关键不是单纯参数量，而是：DINOv3 是否能在**没有 mask 标签、没有 caption 元数据、backbone 冻结**的前提下超过使用更强监督的视觉编码器。

---

## 5. 实验结果

### 5.1 线性稠密探测

只在冻结最后层 patch 上训练线性变换：

| 模型 | ADE20K mIoU ↑ | Cityscapes mIoU ↑ | VOC mIoU ↑ | NYUv2 RMSE ↓ | KITTI RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-g | 49.5 | 75.6 | 83.1 | 0.372 | 2.624 |
| AM-RADIOv2.5 | 53.0 | 78.4 | 85.4 | 0.340 | 2.918 |
| PEspatial | 49.3 | 73.2 | 82.7 | 0.362 | 3.082 |
| Web-DINO 7B | 42.7 | 68.3 | 76.1 | 0.466 | 3.158 |
| **DINOv3 ViT-7B** | **55.9** | **81.1** | **86.6** | **0.309** | **2.346** |

DINOv3 在 ADE20K 上比 DINOv2 高 6.4 mIoU，也超过从 SAM 空间监督蒸馏而来的 AM-RADIO 2.9 mIoU；说明它的空间质量不是靠显式 mask annotation 获得。Web-DINO 同为 7B 却显著落后，进一步说明“扩大参数”不等于“获得稠密特征”，Gram anchoring 和训练设计才是差异来源。

### 5.2 对应、对象发现与视频一致性

- **3D correspondence**：NAVI recall 64.4，较 DINOv2 的 60.1 高 4.3；SPair 58.7，较 DINOv2 高 2.6。
- **无监督对象发现**：TokenCut + DINOv3 在 VOC07/12、COCO 上为 66.1/69.5/55.1 CorLoc；DINOv2 为 55.6/60.4/45.4。
- **视频 mask propagation**：DAVIS 大分辨率 J&F 83.3，比 DINOv2 高 6.7；该模型未用视频训练，只凭逐帧 patch 相似性即可跟踪。

这些结果共同验证：DINOv3 的 patch 既有语义聚类，也保留跨视角、跨帧的局部对应关系。

### 5.3 完整下游系统

- **检测**：冻结 7B backbone，仅训练约 100M Plain-DETR，在 COCO 达 65.6 mAP；TTA 后 66.1，COCO-O 66.4，超过需要微调整个 backbone 的强系统。
- **语义分割**：冻结 backbone + ViT-Adapter/Mask2Former，在 ADE20K 达 62.6 mIoU，TTA 后 63.0，与当时最强结果持平。
- **相对单目深度**：替换 Depth Anything V2 的 DINOv2 且冻结 DINOv3，在 NYUv2/KITTI/ETH3D/ScanNet 上达到 4.3/7.3/5.4/4.4 ARel（论文表中按百分数展示），多数指标为最佳；DIODE 仍存在明显域难度。
- **VGGT**：只把 VGGT 的 DINOv2-L 换成 DINOv3-L，Re10K/CO3Dv2 pose AUC@30 从 85.3/88.2 提升到 86.3/89.6；DTU overall 从 0.382 降至 0.368；ScanNet-1500 AUC@5/10 从 33.9/55.2 升至 35.2/56.1。

这里最重要的结论是**可替换性**：无需为每个系统重新发明视觉前端，仅换 backbone 就能提升检测、深度和三维几何。

### 5.4 模型家族与文本对齐

蒸馏后的 ViT-L（约 0.3B）在 ADE20K/NYUv2/DAVIS/NAVI/SPair 上取得 54.9/0.352/79.9/62.3/61.2，明显优于同规模 DINOv2-L 的 48.8/0.394/73.4/59.9/57.0。ViT-H+ 在多项任务上可接近 8 倍更大的 7B teacher，说明 frontier model 的价值能被下放到可部署规模。

文本对齐后的 DINOv3 dino.txt ViT-L 在 ADE20K/Cityscapes 零样本分割达 **24.7/36.9 mIoU**，高于 PE 的 17.6/21.4 和 SigLIP 2 的 10.8/16.3；但 ImageNet 零样本分类与图文检索仍略落后于最佳 PE/SigLIP 2。这表明 DINOv3 更像“空间视觉底座”，而不是天然的全局图文模型。

### 5.5 消融结论

1. **数据组合优于单一 curation**：LVD-1689M + retrieval + public data 才能同时兼顾自然类别、长尾物种、OOD 与检索。
2. **Gram teacher 必须来自局部性尚好的早期 checkpoint**：100K/200K 均可，1M teacher 效果下降。
3. **2× 高分辨率 teacher 优于 1×**：ADE20K 从 53.6 提升到 55.7，NYUv2 从 0.285 降到 0.281。
4. **Gram anchoring 不显著伤害全局能力**：ImageNet 线性准确率几乎不变，ObjectNet 还随继续训练轻微上升。
5. **高分辨率后训练依赖 Gram anchoring**：若移除，稠密性能会在分辨率适配阶段再次退化。

---

## 6. 与相关工作的关系

### 6.1 相对 DINOv2

DINOv2 解决了稳定训练和通用特征问题；DINOv3 则解决“继续 scale 之后会坏掉什么”。其变化不只是更大数据/模型，而是把 **patch 关系结构**作为显式训练对象。register tokens 处理 high-norm outlier，Gram anchoring 处理长训练中的局部一致性衰减，两者针对不同失效模式。

### 6.2 相对 CLIP、SigLIP 与 PE

CLIP 系模型通过图文监督获得开放词汇和全局语义，但 patch 局部关系不是核心目标。DINOv3 完全不使用 caption 即获得更强稠密特征；需要语言能力时再做 post-hoc LiT 对齐。这种模块化路线牺牲了一部分原生图文全局能力，却减少了视觉预训练对高质量文本元数据的依赖。

### 6.3 相对 SAM 蒸馏型骨干

AM-RADIO、PEspatial 通过蒸馏 SAM/SAM 2 获得空间先验；DINOv3 不借助 mask annotation 仍在线性分割和深度上超过它们，说明局部一致性可以从无标签图像内部关系中学习，而不一定依赖分割监督。

---

## 7. 与 VLM 的关系

### 7.1 DINOv3 在 VLM 中扮演什么角色

DINOv3 不是完整 VLM，而是高质量 **vision tower / visual tokenizer**：

- CLS token 提供图像级语义，可供图文对齐或多模态投影层消费；
- clean dense patch tokens 为 grounding、区域描述、OCR 之外的对象结构、细粒度空间问答提供更可靠输入；
- 高分辨率稳定性允许 VLM 在不更换 backbone 的情况下处理遥感、医学、文档或超高分辨率图像；
- 冻结 backbone 可复用到多个任务头，降低多任务 VLM 的视觉侧重复计算。

### 7.2 它不能替代什么

纯 DINOv3 没有语言编码器、token-level cross-modal alignment 和生成式推理能力；它不知道“文本里的词”与视觉区域如何对应。dino.txt 证明 post-hoc 文本对齐可行，但全局分类/检索仍略弱于原生大规模图文预训练。因此更合理的组合是：**DINOv3 提供空间稳定的视觉 token，LLM/MLLM 负责语言、组合推理和交互。**

### 7.3 对 VLM 研究的直接启发

VLM scaling 不应只关注 image-text loss 和语言端容量。若视觉 backbone 在长训练中失去 patch 局部性，模型可能在 caption benchmark 上继续提升，却在 grounding、指代、计数和 3D reasoning 上退化。Gram anchoring 提供一种低耦合正则：保留早期视觉关系结构，同时允许全局语义继续扩展。

---

## 8. 局限与批判性评价

### 8.1 局限

1. **数据不可完全复现**：核心 LVD-1689M 来自约 17B Instagram 图像池，外部研究者无法重建相同数据分布。
2. **训练成本极高**：单次 7B pretraining 为 61,440 H100 GPU-hours、约 47 MWh；整个项目估计 9M GPU-hours、约 2600 tCO2eq。
3. **稠密特征仍受 patch 粒度限制**：patch size 16 在像素级边界上仍需 ViT-Adapter、Mask2Former、DPT 等较重 decoder。
4. **Gram anchoring 依赖 checkpoint 选择**：如果早期 teacher 本身局部性不好，或领域发生巨大变化，锚定旧关系可能限制新知识；论文主要验证自然图像与遥感，尚未系统覆盖医学、文档、工业等域。
5. **结果存在规模与公平性混杂**：7B teacher 与许多较小 baseline 对比时，算法、数据和参数量同时变化；虽然 Web-DINO 7B 对比有助于隔离“只扩大模型”的影响，但不能完全拆分各因素。
6. **文本能力是后接模块**：dino.txt 的 dense alignment 强，但全局 zero-shot/检索并非最强，不能把 DINOv3 的视觉优势直接等同于完整 VLM 优势。

### 8.2 批判性判断

Gram anchoring 的价值不在“又加一个 loss”，而在于它准确选择了约束对象：不是逐 token 模仿，也不是冻结早期层，而是保留 patch 间二阶关系。其风险同样来自这一点——Gram matrix 是 $P\times P$，高分辨率下计算和显存增长快；论文通过只在 global crops、后期短阶段使用，并共享较小输出尺度来控制成本，但它未必适合所有在线持续学习场景。

---

## 9. 复现成本、实践建议与 Future Work

### 9.1 完整复现成本

完整复现基本只适合大机构：需要 256 GPU 并行、1M steps、超大私有数据池、FP8/bfloat16 和 FSDP 工程。论文估计单次 ViT-7B 训练 **61,440 H100 GPU-hours**。这还不含数据聚类、检索、消融、蒸馏和下游评估。

### 9.2 可行的低成本复现

更现实的验证路径是：

1. 选 ViT-B/L 规模和公开无标签数据；
2. 保存 100K/200K 等早期 EMA checkpoint；
3. 延长普通 DINO+iBOT 训练，持续监控分类与 VOC/ADE linear probe 的分叉；
4. 只在最后 10K—30K steps 加 $\mathcal{L}_{\text{Gram}}$；
5. 比较 1× teacher 与 2× teacher + downsampling；
6. 同时画 patch cosine maps，避免只看下游汇总指标。

最小实验不需要复制 7B scaling，也能验证“局部性退化—Gram 修复”这一核心因果链。

### 9.3 Future Work

论文和实验自然导向以下方向：

- 设计能直接利用 4K clean features 的轻量 decoder，减少 ViT-Adapter/Mask2Former 的参数负担；
- 把 Gram anchoring 扩展为持续学习中的动态关系记忆，而不是依赖固定早期 checkpoint；
- 研究视频内跨帧 Gram、跨视角 Gram，使局部一致性从单图二阶关系扩展到时空关系；
- 在医学、文档、机器人和具身数据上验证“纯 SSL + 后对齐”是否优于原生图文预训练；
- 将视觉关系锚定与 VLM 的 token pruning、resampler、region token 设计结合，检查是否能改善 grounding 与计数。

---

## 10. 材料来源

完整来源与图片追溯见 [sources.md](assets/DINOv3%20-%20大规模自监督视觉基础模型/sources.md)。
