---
type: paper-note
status: complete
domain: World-Models
paper: "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning"
year: 2026
arxiv: "2603.14482"
doi: null
source: https://arxiv.org/abs/2603.14482
project: null
code: null
tags:
  - World-Models
  - JEPA
  - V-JEPA
  - Video-Self-Supervised-Learning
  - Dense-Prediction
  - Deep-Self-Supervision
created: 2026-10-15
updated: 2026-10-15
---

# V-JEPA 2.1：视频稠密特征自监督学习

## 基本信息

- **正式标题**：*V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*
- **作者**：Lorenzo Mur-Labadia、Matthew Muckley、Amir Bar、Mido Assran、Koustuv Sinha、Mike Rabbat、Yann LeCun、Nicolas Ballas、Adrien Bardes
- **版本**：arXiv:2603.14482v3；v1 提交于 2026-03-15，v3 修订于 **2026-06-11**
- **发表状态**：截至本笔记获取日，arXiv 官方页未标注会议或期刊接收信息，因此应称为 **arXiv 预印本**，不能写成已正式同行评审发表
- **研究对象**：在保留 V-JEPA 2 全局视频理解与预测能力的同时，使最终层 patch token 具备可直接用于深度、分割、跟踪与空间交互预测的稠密结构

## 一句话结论

V-JEPA 2.1 把 V-JEPA 2 原本“只监督被遮挡 token”的目标改造成 **mask token 与可见 context token 同时受监督的 dense loss**，再以 **四层 deep self-supervision** 修复稠密目标对全局语义的伤害；配合原生图像/视频 tokenizer、VisionMix163M、2B ViT-G 与高分辨率 cooldown，模型把 ADE20K 线性分割从 22.2 提至 47.9 mIoU、NYUv2 深度 RMSE 从 0.682 降至 0.307，同时维持 SSv2 77.7、EPIC-KITCHENS Action R@5 40.8，并改善机器人抓取与潜空间导航。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

V-JEPA 系列的基本信念是：视频世界模型不必重建每个像素，只需在潜空间预测可预测、与任务相关的表示。V-JEPA 与 V-JEPA 2 已证明这一路线非常适合全局动作识别、未来动作预判、视频问答和潜空间规划，但它们暴露了一个结构性问题：**全局能力强，不代表每个 patch token 都保留了可读的局部空间信息。**

作者用 PCA 和冻结编码器线性探针检查 V-JEPA 2 的 patch feature，发现特征图噪声大、物体边界破碎：ADE20K 仅 22.2 mIoU，NYUv2 为 0.682 RMSE。换言之，模型知道“视频里发生了什么”，却不一定能在最终层准确回答“物体在哪、边界在哪、离相机多远”。这限制了它作为机器人感知 backbone、导航状态表征和通用视频 encoder 的价值。

![](assets/V-JEPA%202.1%20-%20视频稠密特征自监督学习/figures/01-dense-feature-teaser.png)

> **图 1（论文 Figure 1）**：上行为原始帧，中行为 V-JEPA 2 patch feature 的前三主成分，底行为 V-JEPA 2.1。V-JEPA 2.1 在道路、猫、人体和跳水者上形成更连续的语义区域，而且跨视频帧保持颜色一致，说明空间结构与时间一致性同时增强。来源：arXiv v3 官方源码原图，PDF p.1，<https://arxiv.org/abs/2603.14482>。

### 1.2 为什么原始 masked prediction 会丢局部结构

V-JEPA 2 的 predictor 同时接收可见 context token 和代表遮挡位置的 mask token，但损失只在 mask 位置计算。被遮挡 token 必须回归其位置对应的 target feature；可见 token 却没有“保持本位置语义”的直接约束。作者提出的解释是：这些 context token 可能逐渐变成类似 register token 的全局信息容器——它们对预测被遮挡区域有用，却不再忠实对应自身局部位置。

这不是“特征预测天然缺乏稠密能力”，而是**监督落点不完整**：同一 predictor 已经输出了可见位置，只是训练从未要求这些输出与完整视频中同位置的 target token 对齐。因此最直接的修复不是增加复杂 decoder，而是把损失扩展到 context token。

### 1.3 核心贡献

1. **Dense predictive loss**：在原有 masked predictive loss 之外，对可见 context token 施加距离加权 L1，使所有位置都获得训练信号。
2. **Deep self-supervision**：对三个中间层与最终层共四个层级预测，既向最终表示注入局部信息，又恢复单独 context loss 损伤的全局分类能力。
3. **Multi-modal tokenizer**：图像走原生 2D patch embedding，视频走 3D tubelet embedding，并加入 modality embedding，替代“把图像复制成静态视频”的低效做法。
4. **系统性扩展**：VisionMix163M、ViT-g/ViT-G、长训练和高分辨率 cooldown，使 dense 与 global 能力共同扩展。
5. **多任务验证**：从分割、深度、视频目标分割扩展到 Ego4D 短期交互预判、EPIC-KITCHENS 动作预判、真实机器人抓取和导航规划，证明稠密特征不是只对像素级 benchmark 有效。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与训练分支

输入可以是单张图像或视频片段。图像经 $16\times16$ 的 2D convolution 分块；视频经 $16\times16\times2$ 的 3D convolution 分块。两种模态共享 Transformer encoder，但使用不同 patchifier，并添加可学习的 image/video modality embedding 与 3D RoPE。

训练有两个分支：

- **在线分支** $E_\theta$：输入经过 masking，只处理可见集合 $C$；
- **目标分支** $E_{\bar\theta}$：输入完整图像/视频，参数由在线 encoder 的 EMA 更新，输出通过 stop-gradient；
- **预测器** $P_\phi$：接收在线多层 context feature 与 mask token，输出 mask 集合 $M$ 和 context 集合 $C$ 对应的多层预测。

推理和下游评测只保留 encoder；predictor、mask token 与目标 encoder 都是预训练组件。

### 2.2 原始 masked predictive loss

V-JEPA 2 的目标为：

$$
\mathcal{L}_{\mathrm{predict}}
=
\frac{1}{|M|}\sum_{i\in M}
\left\|
P_\phi\!\left(E_\theta(x),\Delta_y\right)_i
-
\operatorname{sg}\!\left(E_{\bar\theta}(y)_i\right)
\right\|_1.
$$

其中 $x$ 是被遮挡输入，$y$ 是完整输入，$M$ 是 mask 位置，$\Delta_y$ 是携带目标位置的可学习 mask token，$\operatorname{sg}$ 表示目标分支不反向传播。EMA target 与 stop-gradient 仍承担防塌缩作用。该式只要求 mask 位置正确，不要求 predictor 对可见位置保持局部对应。

### 2.3 新增 context loss 与 dense loss

V-JEPA 2.1 新增：

$$
\mathcal{L}_{\mathrm{ctx}}
=
\frac{1}{|C|}\sum_{i\in C}\lambda_i
\left\|
P_\phi\!\left(E_\theta(x),\Delta_y\right)_i
-
\operatorname{sg}\!\left(E_{\bar\theta}(y)_i\right)
\right\|_1,
$$

从而得到：

$$
\mathcal{L}_{\mathrm{dense}}
=
\mathcal{L}_{\mathrm{predict}}+\mathcal{L}_{\mathrm{ctx}}.
$$

直觉上，mask loss 教模型“从上下文推断缺失区域”，context loss 教模型“不要为了全局预测而忘记当前可见位置是什么”。这使训练信号从稀疏遮挡集合扩展到完整时空 token 网格。

若所有 context token 使用固定权重，模型可能走向“复制可见特征”的简单解，并明显损害 SSv2/IN1K。最终方案令靠近遮挡边界的位置权重更大：

$$
\lambda_i=\frac{\lambda}{\sqrt{d_{\min}(i,M)}}.
$$

$d_{\min}(i,M)$ 是 context token $i$ 到最近 mask token 的时空 block 距离。这样最强监督集中在“已知—未知交界处”，鼓励局部连续性，又避免远离 mask 的大量易复制 token 主导梯度。最终消融采用基础权重 $\lambda=0.5$，并在 epoch 50–100 对 context 权重 warm-up。

---

## 3. 核心方法

### 3.1 完整 pipeline

![](assets/V-JEPA%202.1%20-%20视频稠密特征自监督学习/figures/02-method-pipeline.jpg)

> **图 4（论文 Figure 4）**：图像/视频分别经 2D/3D patchifier；在线 encoder 丢弃 mask token 后提取四层特征，经归一化、拼接和 MLP 融合，再与 mask token 拼接送入 multi-level predictor；完整输入进入 EMA target encoder。蓝色路径是可见位置 context loss，红色路径是遮挡位置 prediction loss。来源：arXiv v3 官方源码原图，PDF p.5。

完整数据流可拆成七步：

1. **模态专用 patchify**：图像用 2D convolution，视频用 3D convolution；共享 encoder 不再要求图像伪装成 16 帧静态视频。
2. **位置与模态编码**：加入 3D RoPE 和可学习 modality embedding，使共享参数知道 token 来自 image 还是 video。
3. **mask corruption**：随机丢弃 patch；在线 encoder 只计算可见 token，延续 V-JEPA 的潜空间 mask denoising。
4. **多层读取**：从三个中间 block 与最终层取得表示，对每层归一化后沿通道拼接。
5. **轻量融合**：MLP 融合四层表示并降维，避免 predictor 输入维度随层数线性膨胀。
6. **多层预测**：融合后的 context token 与 mask token 沿序列维拼接，predictor 为四个目标层级分别输出预测。
7. **双位置、四层级损失**：每个层级都同时计算 mask prediction 与 weighted context loss；在线 encoder 和 predictor 走梯度，target encoder 由 EMA 更新。

### 3.2 Deep self-supervision 为什么必要

只加 context loss 时，ADE20K 从 22.2 跃升到 33.8，但 IN1K 从 82.2 跌到 72.6、SSv2 从 72.8 跌到 62.5。这揭示了一个冲突：**把最终层强迫成逐位置可读，可能削弱它对跨区域、跨时间语义的抽象。**

Deep self-supervision 的解法不是把 context loss 调小到几乎无效，而是同时对中间层和最终层建立预测目标。按论文文字可写成以下等价记法（原文未单独编号，也未说明层间是求和还是平均）：

$$
\mathcal{L}_{\mathrm{deep}}
=
\sum_{\ell\in\mathcal{S}}
\left(
\mathcal{L}^{(\ell)}_{\mathrm{predict}}
+
\mathcal{L}^{(\ell)}_{\mathrm{ctx}}
\right),\qquad |\mathcal{S}|=4.
$$

它的作用有两层。第一，中间层直接承担局部对应，减少所有局部细节都压到最终层的压力；第二，多层融合让 predictor 同时看到低层边界、中层部件和高层语义，最终层不必在“局部坐标”与“全局不变性”之间二选一。结果是 SSv2 恢复至 72.1、IN1K 恢复至 80.8，同时 ADE20K 继续升至 38.6。

### 3.3 Multi-modal tokenizer

V-JEPA 2 用同一个 3D convolution 处理两种模态：一张图像先沿时间复制，再作为静态视频输入。这既把图像 token 计算放大，也向模型注入“零运动视频”的偏置。V-JEPA 2.1 改为：

- 视频 patch：$16\times16\times2$；
- 图像 patch：$16\times16$；
- encoder/predictor 都加入 modality embedding；
- 图像 worker 和视频 worker 使用独立 batch，迭代末聚合梯度后统一更新共享模型。

该设计把“共享语义 backbone”与“模态原生 tokenization”分离：共享的是 Transformer 表征能力，不共享不合理的输入离散化方式。消融中 ADE20K 40.8→41.4，NYUv2 约 0.418→0.415，而 SSv2/IN1K 基本不变，说明它主要改善效率与稠密空间编码，不靠牺牲视频能力换收益。

### 3.4 数据、模型与分辨率扩展

训练数据升级为 **VisionMix163M**。视频部分包含 SSv2 168K、Kinetics 733K、HowTo100M 1.1M、YT-1B 19M，并加入 LVD-142M 图像。相较 V-JEPA 2，视频采样更偏向 YT-1B（0.188→0.720）和 motion-centric SSv2（0.056→0.170），降低 Kinetics（0.188→0.010）与 HowTo100M（0.318→0.100）；ImageNet 1M 被 LVD-142M 替代。图像/视频整体比例由 batch 资源控制：全局 batch 为 128 个 16 帧视频 clip 与 2,304 张图像。

主训练 135K iterations，视频与图像均以 $256^2$ 空间分辨率训练，视频为 16 帧；随后 12K cooldown iterations 降学习率，同时把视频升到 64 帧、$384^2$，图像升到 $512^2$。模型从 ViT-L 300M 扩至 ViT-g 1B 与 ViT-G 2B，另将 ViT-G 蒸馏到 ViT-L 300M 和 ViT-B 80M。蒸馏时冻结 ViT-G 取代 EMA teacher，学生仍维护只用于最终评估的 EMA；蒸馏只匹配教师末层，不再使用 deep self-supervision。

### 3.5 推理复杂度与用途

V-JEPA 2.1 不是像素生成器。预训练结束后通常只保留 encoder，因此单次特征提取复杂度与同尺寸 ViT 接近；新增 predictor、EMA target 与多层损失只增加训练成本。它的价值不是让普通分类更便宜，而是让**同一个最终层**同时服务全局与稠密任务，减少下游依赖 feature pyramid、复杂 decoder 或跨层拼接的需要。

### 3.6 为什么不是简单地给 patch 加重建头

从设计哲学看，V-JEPA 2.1 与像素重建仍有本质差异。像素 decoder 要恢复颜色、纹理、压缩噪声和不可预测背景细节，损失会奖励对低层外观的精确复原；context loss 回归的是 EMA encoder 产生的目标表示，目标空间本身可以丢弃无关像素，只保留教师认为稳定、可预测的语义与几何。因此它追求的“稠密”不是每个 patch 都保留原始像素，而是每个位置都有语义一致、跨帧稳定、可供简单探针读取的 latent code。

距离加权也体现了这种区别。若目标只是自编码，所有可见位置都可等权重复制；这里却故意强调靠近 mask 的 token，因为这些位置最需要把局部上下文与缺失区域连接起来。它相当于在时空遮挡边界附近施加一种软连续性先验：边界两侧表示既不能失去位置对应，也要支持预测跨越边界的内容。远离遮挡区的 context token 权重降低，可避免训练被大量“看见什么就复制什么”的容易样本支配。

Deep self-supervision 则处理表示层级的分工。浅层天然保留边缘、纹理和精确坐标，深层天然追求类别与动作不变性。如果只在最后层做逐位置回归，网络会被迫把互相竞争的信息塞进同一表示；四层目标允许局部信号沿深度逐渐抽象，同时在每个选定层接受校正。论文附录还比较只读末层与拼接四层的下游评测，核心目的正是验证：改进应沉淀到最终层，而不是依赖评测时临时把中间层重新拼回来。

因此，这套方法的最小概念单元不是某一个新 loss，而是三件事的组合：**监督覆盖所有位置、监督覆盖多个深度、目标仍位于潜空间。** 少掉第一项，patch 会变成全局寄存器；少掉第二项，全局语义明显退化；把第三项换成像素，又回到高成本重建路线。

---

## 4. 数据集与实验设置

### 4.1 评测协议

- **稠密单图任务**：冻结 encoder，仅取最终层 token，训练一个线性稠密投影；NYUv2/KITTI 用 RMSE，ADE20K/Cityscapes/VOC12 用 mIoU。
- **视频目标分割**：DAVIS-17 与 YouTube-VOS 上不训练参数，使用 patch cosine similarity 做标签传播；最佳配置为 15 个上下文帧、圆形邻域 12、top-5、temperature 0.2。
- **全局分类**：冻结 encoder，训练 4-layer attentive probe；SSv2、Diving-48、K400 与 IN1K 报 top-1。
- **Ego4D STA v2**：243 小时标注视频，输入 8 帧低分辨率视频加最后一帧 1080p 图像，预测交互物体框、noun、verb 和 time-to-contact；核心指标 mAP All。
- **EPIC-KITCHENS-100**：冻结 encoder 与 predictor，利用 predictor 生成未来 latent，三个 attentive query 分别预测 verb/noun/action，报 mean-class Recall@5。
- **机器人操作**：DROID raw 数据训练 24 层、约 300M 的 action-conditioned predictor；Franka Panda 上以视觉目标和 CEM 做 MPC，每项技能仅 10 个任务。
- **导航**：在 Recon、Tartan Drive、Scand、Sacson 上预测约 2 秒开环轨迹；latent diffusion 预测 clean representation 并用 DDIM，CEM 从 480 条 3-DoF 轨迹中选择。

### 4.2 Baseline

稠密视觉对比包括 DINOv2、DINOv3、Web-DINO、PEcore/PEspatial、AM-RADIO、InternVideo2、SigLIP2；视频分类包括 V-JEPA、V-JEPA 2、VideoMAEv2、VideoPrism、InternVideo2s；预判任务包括 STAformer、GANO、StillFast、InAViT、Video-LLaMA、PlausiVL；导航对比 NWM + SD-VAE；VideoQA 对比 InternVL、Qwen2/2.5-VL、PerceptionLM 与同数据重训的 V-JEPA 2。

需要谨慎的是，表中一部分为论文引用的既有结果，一部分是作者在统一 probe 下重测；参数量、输入分辨率和预训练数据也不总相同。因此“表中最好”不自动等于严格同计算预算的胜出。

---

## 5. 实验结果

### 5.1 组件累计消融

![](assets/V-JEPA%202.1%20-%20视频稠密特征自监督学习/figures/03-component-ablation.png)

> **图 5（论文 Figure 5）**：蓝柱为 ADE20K mIoU，橙柱为 SSv2 准确率。Context loss 首先大幅提高分割但破坏分类；deep self-supervision 恢复分类并继续提升分割；数据、tokenizer、模型和高分辨率训练随后稳定累积收益。来源：arXiv v3 官方矢量原图，300 DPI 转换，PDF p.6。

| 累计配置 | IN1K ↑ | SSv2 ↑ | NYUv2 RMSE ↓ | ADE20K mIoU ↑ |
|---|---:|---:|---:|---:|
| V-JEPA 2 | 82.2 | 72.8 | 0.682 | 22.2 |
| + Context Loss | 72.6 | 62.5 | 0.474 | 33.8 |
| + Deep Self-Supervision | 80.8 | 72.1 | 0.463 | 38.6 |
| + VisionMix163M | 81.6 | 72.6 | 0.418 | 40.8 |
| + Multi-Modal Tokenizer | 81.6 | 72.6 | 0.415 | 41.4 |
| + Model Scaling（ViT-G） | 84.8 | 76.1 | 0.365 | 47.1 |
| + High-Res Cooldown | **85.5** | **77.7** | **0.307** | **47.9** |

这张表最重要的结论不是“scale 越大越好”，而是 context loss 与 deep self-supervision 必须成对理解。前者提供局部监督，后者解决由此引入的局部—全局冲突；后续 scale 只是在正确目标上放大收益。

Context 权重消融也支持这一点：固定 $\lambda=0.05$ 时 ADE20K 26.4、SSv2 71.0；固定 0.5 时分割 27.5、分类跌至 53.8；加入 warm-up 后固定 0.5 达 32.2/61.5；最终距离加权 0.5 达 33.8/62.5。权重越强并非越好，空间上选择性地监督 mask 边界更合理。

### 5.2 稠密视觉

最终 ViT-G 在 NYUv2 达 **0.307 RMSE**，略优于表中 DINOv3 7B 的 0.309；KITTI 为 2.461，接近 DINOv3 7B 的 2.346。分割为 ADE20K 47.9、Cityscapes 73.5、VOC12 85.0，较 V-JEPA 2 的 24.4/45.9/63.9 有巨大提升，但仍低于 DINOv3 7B 的 55.9/81.1/86.6。由此应表述为：V-JEPA 2.1 **修复了视频 JEPA 的 dense weakness，并达到强竞争水平**，而不是全面超过最强图像稠密模型。

无参数视频目标分割中，DAVIS 为 69.0、YouTube-VOS 为 72.7，仅略低于 DINOv3 的 71.1/74.1。由于这里只做标签传播，它比线性分割更直接证明了 patch embedding 的跨帧一致性。

### 5.3 全局识别与动作预判

ViT-G 在 SSv2 为 **77.7**，Diving-48 89.2、K400 87.7、IN1K 85.5。它保住了 V-JEPA 2 的运动理解优势，但并非每项都提高：V-JEPA 2 ViT-g 的 Diving-48 为 90.2，同为 1B 时 V-JEPA 2.1 ViT-g 在四项分类上还略低于 V-JEPA 2 ViT-g。最终最好结果部分来自 2B scaling 与 cooldown，不能全部归因于 dense loss。

Ego4D STA 是 dense feature 与未来预测结合最充分的实验。V-JEPA 2.1 ViT-G 的 AP All 为 8.20、mAP All 为 **7.71**；V-JEPA 2 ViT-g 的 mAP All 为 6.02，STAformer 为 5.67。它不仅要分类 verb/noun，还要定位未来将交互的物体并估计接触时间，因此比普通动作分类更能检验“空间落地”。

EPIC-KITCHENS-100 上，ViT-G 达 Verb/Noun/Action R@5 = **64.3/59.9/40.8**。不过 1B V-JEPA 2.1 的 Action 38.4 低于 1B V-JEPA 2 的 39.7；扩到 2B 后才超过到 40.8。这个结果提醒我们：更稠密的 feature 不自动让所有高层 anticipation 受益。

### 5.4 机器人操作与导航

![](assets/V-JEPA%202.1%20-%20视频稠密特征自监督学习/figures/04-robot-grasping.png)

> **图 8（论文 Figure 8）**：左侧比较 V-JEPA 2 与 V-JEPA 2.1 按视觉目标执行抓取，右侧展示 V-JEPA 2.1 在动作平移维度上的能量地形。更好的深度编码使目标附近形成更可用于 MPC 搜索的低能量区域。来源：arXiv v3 官方矢量原图，300 DPI 转换，PDF p.13。

机器人结果为 Reach 100%、Grasp 80%、Pick-and-Place 80%。与 V-JEPA 2 的 100%/60%/80% 相比，最大提升在抓取：使用 8-step horizon 后为 **+20 个百分点**。但同为 horizon 1 时其实只从 60% 到 70%；且每项只有 10 个任务，一次成败就是 10 个百分点，论文未报置信区间。因此“20-point improvement”应理解为特定长视界配置的观察结果，而不是高统计置信度的普遍提升。失败主要来自夹爪过早闭合或搬运途中松开，说明 encoder 改善不能替代动作模型与控制器。

导航中，V-JEPA 2.1 使用 8 个 DDIM 去噪步，NWM/SD-VAE 至少需 128 步；规划耗时 **10.6 秒 vs 103.2 秒，约 10× 加速**。ViT-g 平均 ATE 2.975，ViT-G 平均 RTE 0.688，误差未因加速恶化。需要注意该实验在导航数据上 fine-tune，且只评估开环轨迹，不是零样本真实闭环导航。

### 5.5 VideoQA

与相同过滤后数据重训的 V-JEPA 2 相比，V-JEPA 2.1 平均分仅 57.8→57.9；PerceptionTest 80.1→83.1、MVP 40.9→43.2、MVBench 73.3→75.4，但 TemporalBench 32.3→28.5、TOMATO 41.4→38.0。论文原 V-JEPA 2 的 59.5 来自不同且更大的对齐数据，不能直接作为公平基线。结论应是：V-JEPA 2.1 对丰富空间视觉语义有帮助，但对纯时间事件问答没有稳定优势。

### 5.6 如何理解整组实验的证据强度

把所有实验放在一起看，可以把论文主张分成三层。第一层是最扎实的“表示结构修复”：同一 ViT-L 消融中，context loss 让 ADE20K 大幅上升，deep self-supervision 又把 SSv2 拉回原水平；这个因果链有直接组件对照，不依赖 2B 模型。第二层是“修复能够随数据、模型和分辨率扩展”：从 38.6 mIoU 继续升到 47.9，并且 NYUv2 降到 0.307；这一层证据充分，但各因素是累计加入，不能从最终模型反推出每个因素在不同规模上完全独立。第三层是“稠密表示改善世界模型与机器人”：Ego4D、抓取和导航给出积极结果，却同时改变了 predictor、规划 horizon 或下游训练设置，因果归因最弱，应视为应用可行性证据而非严格机制证明。

在指标选择上，论文有意覆盖三种“局部可读性”。线性深度/分割检验单帧空间结构；无参数视频目标分割检验跨帧 token 对应；STA 与机器人规划检验局部结构能否进入未来预测和动作选择。这一组合比只报 ADE20K 更有说服力，因为它说明 dense feature 不只是静态边界更锐利，还能跨时间维持对象身份，并对预测未来交互位置有帮助。

但这些协议仍有一个共同盲点：它们大多在冻结 backbone 后增加线性头、attentive probe、动作 predictor 或扩散规划器。最终性能是“表示质量 × 下游头容量 × 训练协议”的乘积。要更干净地证明 V-JEPA 2.1 学到了可规划几何，未来应增加统一的小头容量对比、跨数据集零样本 correspondence、相机位姿变化下的三维一致性，以及对 encoder 表征误差如何传递到 MPC 代价面的量化分析。

### 5.7 关键结果速查与正确读法

| 能力 | V-JEPA 2.1 结果 | 最合理的结论 | 不应过度声称 |
|---|---:|---|---|
| NYUv2 深度 | 0.307 RMSE | 最终层具有很强单目几何可读性 | 所有稠密任务全面超过 DINOv3 |
| ADE20K 分割 | 47.9 mIoU | 相比 V-JEPA 2 修复幅度巨大 | 已成为最强通用分割 backbone |
| DAVIS / YTVOS | 69.0 / 72.7 | 无训练标签传播显示跨帧一致性 | 已解决任意长视频跟踪 |
| SSv2 | 77.7 | dense 改造后仍保留强运动识别 | 1B 同规模一定优于 V-JEPA 2 |
| Ego4D STA | 7.71 mAP All | 局部定位与未来交互预测可结合 | 单一指标足以证明完整世界模型 |
| 机器人 Grasp | 80% | 长 horizon 配置下有积极改善 | 20 点差异具有充分统计显著性 |
| 导航 | 10.6 秒、约 10× | latent 规划可显著减少去噪步数 | 已验证真实闭环零样本导航 |

这套“正确读法”很重要：V-JEPA 2.1 真正完成的是从全局视频表征到时空稠密表征的明显跨越，并展示其向规划迁移的可能性；它还没有证明一个单一 frozen encoder 在所有感知、推理和控制场景中都统一占优。

---

## 6. 与相关工作的关系

### 6.1 与 V-JEPA 2

V-JEPA 2 的核心是大规模视频潜空间预测、全局理解和 action-conditioned planning；V-JEPA 2.1 不是推翻它，而是修复其 patch token 的空间可读性。二者都保留在线 encoder、EMA target、stop-gradient、mask token predictor 和 L1 latent regression。2.1 的本质变化是：

- 监督位置：仅 mask → mask + context；
- 监督深度：末层为主 → 四层 deep supervision；
- 图像输入：复制成静态视频 → 原生 2D tokenizer；
- 数据/模型：VideoMix22M、1B → VisionMix163M、最高 2B；
- 下游重点：全局理解/规划 → dense perception + 交互预测 + 规划。

因此“2.1”非常贴切：它是同一架构范式上的表示质量升级，不是新的 collapse-prevention 原理。

### 6.2 与 DINO 系列

DINOv2/3 的 patch feature 长期以分割、深度、对应关系见长，但主要从图像学习，不直接学习视频动力学。V-JEPA 2.1 试图占据中间地带：dense feature 接近图像 SSL，同时保留 SSv2、future anticipation 和 latent planning。实验表明这一目标部分实现：深度和 VOS 很强，SSv2 保持领先；但 ADE20K/Cityscapes 仍低于 DINOv3，说明图像语料组成和场景多样性仍是短板。

### 6.3 与 LeJEPA / LeVJEPA

LeJEPA 用 invariance + SIGReg 从目标函数层面排除 collapse，不需要 EMA teacher、stop-gradient 或 predictor；LeVJEPA 把这一原则扩展到视频，并通过 95% random token dropping 获得极高效率。V-JEPA 2.1 仍属于“非对称 teacher–student + masked latent prediction”路线，训练更重，但显式优化所有 patch 位置和多个层级，目标更直接对准 dense prediction。值得注意的是，LeVJEPA 的 patch token 即便没有 patch-level loss 也呈现稠密结构，这对“V-JEPA 必须显式 context loss 才能 dense”构成有价值的反例：问题可能不只在监督位置，也在 teacher/predictor 非对称和目标几何。

---

## 7. 局限与批判性评价

1. **最佳结果与规模耦合**：许多最终优势来自 2B ViT-G、LVD-142M 与高分辨率 cooldown。1B V-JEPA 2.1 在若干全局任务反而略低于 V-JEPA 2，因此不能把所有收益归结为 dense/deep loss。
2. **稠密与全局仍有真实冲突**：context loss 单独使 IN1K/SSv2 暴跌约 10 点。Deep self-supervision 能修复大部分，但说明目标并非天然兼容，后续模型换数据或规模时仍需重新检查。
3. **语义分割未全面 SOTA**：ADE20K 与 Cityscapes 明显低于 DINOv3。论文推测 VisionMix163M 对拥挤复杂场景覆盖不足，这意味着目标函数改进不能替代合适的数据分布。
4. **机器人样本太少**：每技能 10 个任务，10/20 个百分点只对应 1/2 次成功差异，且没有多 seed、置信区间或不同 embodiment。
5. **导航评测边界有限**：经过导航数据 fine-tune，只做开环约 2 秒轨迹；尚不能证明真实闭环系统在长时导航中的鲁棒性。
6. **VideoQA 结论不统一**：空间感知 benchmark 提升，TemporalBench/TOMATO 下降；对齐数据量也变化，难以把差异干净归因于视觉 encoder。
7. **预训练成本披露不足**：论文给出 135K+12K iterations、batch 与输入分辨率，但未披露 GPU 数量、墙钟时间、总 FLOPs 和能耗，外部团队难以准确预算。
8. **潜空间仍难解释**：PCA、线性 probe 与标签传播只是间接观察；模型预测错误时，没有像素 decoder 可直接展示“它想象的未来”。

---

## 8. 复现与实践建议

### 8.1 最小可行复现

完整 2B + 163M 数据训练不现实。更合理的最小实验是从公开 V-JEPA 2 ViT-L checkpoint 出发，在中等图像/视频混合数据上加入：

1. predictor 对 context token 的输出；
2. $\lambda=0.5$ 的距离加权 context loss；
3. epoch 50–100 的 context warm-up；
4. 三个中间层 + 最终层的四层 supervision；
5. ADE20K linear mIoU、NYUv2 RMSE 与 SSv2 attentive probe 三项联合监控。

只看分割上升会漏掉全局 collapse；必须同时复现“context 提 dense、deep supervision 恢复 global”的因果链。

### 8.2 资源与数据成本

公开信息可确认：主训练 135K iterations，cooldown 12K；视频 batch 128×16 帧、图像 batch 2304；最终视频达 64 帧 $384^2$、图像达 $512^2$；最大模型 2B。论文未披露 GPU 数与墙钟时间，因此不应编造具体 GPU-day。工程上最重的部分是：EMA target 的完整输入前向、多层 target 读取、2B encoder、混合图像视频数据管线，以及 cooldown 的高时空分辨率。

### 8.3 易错点

- $\mathcal{L}_{ctx}$ 的 target 必须来自完整输入同位置的 EMA feature，并 stop-gradient；
- 距离应在时空 token grid 上计算，而非 flatten 后的一维索引差；
- 不要省略 warm-up，否则固定较大 $\lambda$ 会显著伤害全局任务；
- 多层输出需先归一化再拼接，并用 MLP 降维；
- 图像与视频需独立 patchifier，modality embedding 同时进入 encoder/predictor；
- 下游 dense 评测应只读最终层，才能验证 deep self-supervision 是否真的把局部信息推到末层。

---

## 9. 个人启发与后续问题

V-JEPA 2.1 最重要的启发是：**世界模型的状态表示不能只以全局分类或未来预测来验收。** 对机器人而言，同一表示必须同时知道“将发生什么”和“发生在何处”。如果 patch token 退化为全局寄存器，那么 latent dynamics 再准确，也可能给控制器一个空间几何扭曲的状态。

第二个启发是 dense/global trade-off 可以通过“监督深度分工”缓解：中间层保留局部结构，最终层保留语义抽象，多层目标把两者连起来。这比单纯增大 decoder 更接近 backbone 本身的改造。

后续值得追问：

1. context loss 是否能被更原则化的分布正则替代，例如 LeJEPA/SIGReg，使 patch 结构自然涌现而不引入强位置回归？
2. 距离权重目前只依赖 mask 几何，能否按运动边界、光流不确定性或语义一致性自适应？
3. 机器人提升究竟来自深度几何、跨帧对应还是 2B scaling？需要受控 probe 与更多任务拆分。
4. action-conditioned predictor 是否也应做 deep self-supervision，使可规划 latent 在多个抽象层级保持一致？
5. 如果把 V-JEPA 2.1 的 dense encoder 与 LeVJEPA 的单 encoder + SIGReg 效率结合，是否能得到既便宜又具强空间落地的下一代视频 JEPA？

---

## 10. 材料来源

本地官方 PDF、arXiv 源码包、4 张已验证完整关键图及逐项溯源见：

- [sources.md](assets/V-JEPA%202.1%20-%20视频稠密特征自监督学习/sources.md)
- [官方 PDF](assets/V-JEPA%202.1%20-%20视频稠密特征自监督学习/sources/paper.pdf)

核心外部来源：

- arXiv 摘要与版本历史：<https://arxiv.org/abs/2603.14482>
- arXiv v3 官方 PDF：<https://arxiv.org/pdf/2603.14482v3>

> 取材说明：正文数字与公式以 arXiv v3 官方 PDF/TeX 为准。论文中个别正文与表格存在四舍五入差异，例如 Context Loss 后 ADE20K 33.9/33.8、NYUv2 0.473/0.474；本笔记在主表中采用表格数字，并在分析中保留差异说明。
