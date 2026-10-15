---
type: paper-note
status: complete
domain: Foundations
paper: "DINOv2: Learning Robust Visual Features without Supervision"
year: 2024
arxiv: "2304.07193"
doi: null
source: https://arxiv.org/abs/2304.07193
project: https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning
code: https://github.com/facebookresearch/dinov2
tags:
  - self-supervised-learning
  - vision-foundation-model
  - dense-features
  - knowledge-distillation
  - data-curation
  - visual-geometry
created: 2026-10-15
updated: 2026-10-15
---

# DINOv2：无监督鲁棒视觉特征

## 基本信息

| 项目 | 内容 |
|---|---|
| 正式标题 | **DINOv2: Learning Robust Visual Features without Supervision** |
| 作者 | Maxime Oquab、Timothée Darcet、Théo Moutakanni、Huy V. Vo 等 |
| 机构 | Meta AI Research、Inria |
| 发表 | Transactions on Machine Learning Research，2024-01 |
| arXiv | [2304.07193](https://arxiv.org/abs/2304.07193) |
| 官方项目页 | [Meta AI：DINOv2](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning) |
| 官方代码 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| 本地原文 | `assets/DINOv2 - 无监督鲁棒视觉特征/sources/paper.pdf` |
| 精读日期 | 2026-10-15 |
| 重要性评估 | **★★★★★（5/5）**：它把纯图像自监督从 ImageNet 范围内的算法竞赛推进为可直接复用的通用视觉骨干，尤其奠定了后续深度、对应、3D 与视觉世界模型使用稠密预训练特征的工程范式。 |

**一句话结论：**

DINOv2 的关键不是发明一个全新的自监督损失，而是把 **LVD-142M 自动数据治理、DINO 图像级蒸馏、iBOT patch 级蒸馏、KoLeo 特征均匀化、Sinkhorn-Knopp centering、高效大模型训练、高分辨率适配与教师蒸馏**组合成可扩展系统，使冻结的 ViT 特征同时服务分类、检索、分割、单目深度与跨图像对应，并在多数任务上追平或超过依赖文本监督的 OpenCLIP。

---

## 1. 研究背景与问题定义

### 1.1 论文真正要解决的问题

DINOv2 讨论的不是“如何再提高一点 ImageNet 分类准确率”，而是一个视觉基础模型问题：

> 能否只从图像本身学习一个通用编码器，使它在**不微调 backbone**的条件下，跨图像分布、跨任务、跨粒度地提供可直接使用的表示？

这里有三层要求：

1. **图像级通用性**：分类、细粒度识别、视频理解需要全局语义。
2. **实例级通用性**：地标、艺术品和历史街景检索要求同一实例在视角、时代和风格变化下仍接近。
3. **像素 / patch 级通用性**：分割、深度、密集匹配要求特征保留局部结构，而不是只知道“图里有什么”。

此前几条路线各有缺口：

- DINO 已展示 ViT 自监督特征会自发形成对象与部件结构，但主要在 ImageNet-1k 上训练，规模和跨域覆盖有限。
- iBOT 将 masked image modeling 引入在线 tokenizer 与 patch 蒸馏，局部特征更强，但大规模训练仍不够稳定、昂贵。
- MAE 擅长从遮挡重建中学表示，端到端微调很强，但原始 frozen feature 的线性可分性较弱。
- CLIP / OpenCLIP 通过图文对齐获得开放词汇语义，却受 caption 信息瓶颈约束；文本通常说出主体，却不会完整描述空间布局、细部和几何。
- 直接把 SSL 扩到未清洗网页图像，会因重复、分布偏斜和低质量样本而显著掉点。

因此 DINOv2 的核心假设是：

**现有判别式 SSL 已有足够好的归纳偏置；只要数据被系统治理，训练在模型与数据规模上稳定扩展，纯视觉自监督就能产生通用 frozen features。**

### 1.2 为什么纯视觉特征仍然重要

图文预训练把视觉映射到语言可表达的概念空间，这对开放词汇任务极其有效，但语言并不是图像的无损描述。

例如“一个房间里有一把椅子”通常不会编码：

- 椅子相对墙面的精确位置；
- 每条腿的边界与遮挡；
- 房间平面的深度梯度；
- 相似部件在另一张图中的对应位置。

DINOv2 不提供语言接口，却把这些难以被 caption 覆盖的视觉信息保留在 patch tokens 中。

这就是它后来适合作为深度、3D、机器人和 world model 视觉骨干的根本原因。

### 1.3 通用特征的直观证据

![](assets/DINOv2%20-%20无监督鲁棒视觉特征/figures/01-general-features.png)

> **图 1｜论文 Figure 1｜PDF 第 2 页｜URL：https://arxiv.org/pdf/2304.07193。** 作者对同一列图像的 patch features 联合做 PCA，并把前三主成分映射为 RGB。鸟翼、象鼻、马身与车辆部件在姿态、画风乃至类别变化下保持对应；这不是监督分割结果，而是冻结特征的线性结构。图像已由本地官方 PDF 完整裁取并核验。

这张图表达了 DINOv2 最有影响力的性质：

- 第一主成分常能近似分离前景与背景；
- 后续主成分会沿对象部件变化；
- patch descriptor 同时包含语义和位置相关信息；
- 相似“功能部件”可以跨类别匹配，例如鸟翼与飞机机翼。

但需要谨慎：

PCA 可视化是定性证据，不等于模型显式恢复了 3D、部件标签或因果结构。

---

## 2. 核心贡献

1. **提出 LVD-142M 自动数据治理管线。**
   从十亿级网页图像池出发，不依赖文本或人工标签，通过去重、与精选种子集的视觉检索和按来源限额平衡，得到 142,109,386 张预训练图像。

2. **把 DINO 与 iBOT 组合为可扩展的图像级 + patch 级判别式自蒸馏。**
   CLS token 学跨视图语义不变性，masked patch token 学局部内容；独立 heads、Sinkhorn-Knopp 与 KoLeo 提升大规模训练稳定性和检索几何。

3. **训练 1.1B 参数 ViT-g/14，并系统优化训练效率。**
   FlashAttention 类内核、sequence packing、跳过式 stochastic depth 和 FSDP 使代码相对 iBOT 约快 2 倍、显存约为其 1/3。

4. **用短时高分辨率适配改善稠密任务。**
   主训练后只增加 10k iterations 的 518×518 阶段，就接近全程高分辨率训练的效果，而避免约 3 倍训练开销。

5. **将 ViT-g 的能力蒸馏到 ViT-S/B/L。**
   小模型不是各自从零训练，而是复制冻结大教师的图像级与 patch 级分布；蒸馏后的 ViT-L 在 12 个基准上全部超过同架构从零训练版本。

6. **以 frozen-feature 为中心进行广泛评估。**
   论文覆盖分类、OOD 鲁棒性、细粒度识别、视频理解、实例检索、语义分割、单目深度与 dense matching，证明同一个 backbone 能被简单头部复用。

---

## 3. LVD-142M：数据构建方法

### 3.1 数据治理是论文的第一主角

DINOv2 的提升不能只归因于 loss。

论文最重要的工程判断是：

**无标签并不等于无需治理；SSL 不使用标签，但仍高度依赖数据分布。**

网页图像池存在四类问题：

- 热门概念数量压倒长尾概念；
- 同图、缩放图、压缩图和轻微编辑图大量重复；
- benchmark 的 train/test 近重复可能造成泄漏；
- 非安全内容、人脸与低质量网页素材需要过滤。

### 3.2 总体管线

![](assets/DINOv2%20-%20无监督鲁棒视觉特征/figures/03-data-curation-pipeline.png)

> **图 2｜论文 Figure 3｜PDF 第 4 页｜URL：https://arxiv.org/pdf/2304.07193。** 未整理网页图和精选种子图先进入统一 embedding 空间，网页池经过 deduplication，再以精选图为查询进行视觉 retrieval，最终与种子数据合并。该管线完全依赖图像相似度，不依赖 caption、hashtag 或类别标签。图像已由本地官方 PDF 完整裁取并核验。

流程可以拆成六步：

1. 从公开网页抓取库的 HTML `<img>` 标签收集图像 URL。
2. 丢弃不安全或受限制域名的 URL。
3. 下载后进行 PCA-hash 初筛去重、NSFW filtering 和可识别人脸模糊。
4. 用 copy-detection 特征进行十亿规模近重复图去重。
5. 用自监督 ViT-H/16 把图像映射到向量空间。
6. 以约 25 个精选第三方数据集为种子，从网页池检索视觉近邻并做配额平衡。

主文把初始处理后的来源概括为 **1.2B unique images**。

附录给出更细阶段：约 **1.3B** 原始候选经 self-dedup 后为 **1.1B**，再对评测数据做 relative dedup 后为 **744M**。

这不是严格冲突，而是主文与附录对“原始 / 初筛后 / 自去重后”阶段采用了不同粒度和四舍五入口径。

### 3.3 Self-deduplication：先消除网页冗余

作者使用 Pizzi et al. 的 copy-detection embedding：

- 为每张图检索 $k=64$ 个近邻；
- 只保留 cosine similarity 大于 0.6 的边；
- 在 k-NN 图上求 connected components；
- 每个重复连通分量只保留一个代表图。

这一步把约 1.3B 候选压到约 1.1B。

形式上，图像 $s,r$ 的相似度为：

$$
m(s,r)=\frac{f(s)^\top f(r)}{\|f(s)\|_2\|f(r)\|_2}.
$$

这里的 $f$ 是 copy-detection 或 retrieval encoder，而不是文本编码器。

### 3.4 Relative deduplication：控制 benchmark 泄漏

仅做数据池内部去重还不够。

作者进一步拿所有评测 benchmark 的 train / validation / test 作为参照，删除与它们过近的网页图像：

- 相似度阈值更严格，使用 $>0.45$；
- 一旦某个重复连通分量与参照图关联，整组都丢弃；
- 结果从约 1.1B 降到 744M。

这一步很重要，因为 LVD-142M 的种子本身包含多个下游数据集的训练划分。

作者试图防止 test / validation 的直接近重复进入预训练，但无法保证语义层面的“同地点、同对象、同事件”完全隔离。

### 3.5 Retrieval：用精选分布检索网页近邻

检索 embedding 来自一个在 ImageNet-22k 上自监督训练的 ViT-H/16。

对不同大小的种子集，作者使用两种策略。

#### 3.5.1 Sample-based retrieval

适用于百万级以上的大数据集：

- 对每个 query image 找固定数量的近邻；
- Google Landmarks v2 和 ImageNet-22k 通常使用 $k=4$；
- ImageNet-1k 使用 $k=32$，因此约 1.28M 训练图检索出约 41.0M 图像；
- 主文强调 $N=4$ 通常是质量与 collision 的折中，过大的 $N$ 会使同一网页图成为多个 query 的近邻。

#### 3.5.2 Cluster-based retrieval

适用于较小的细粒度、分割、深度和检索数据集：

- 先把 744M 网页池分成 100,000 个 k-means clusters；
- 找出包含至少 3 个种子图的 clusters；
- 从相关 cluster 中采样最多 10,000 张图；
- 对每个小型种子数据集，最终最多保留 1M 张 retrieved images。

Cluster-based retrieval 不是给每个 query 固定取邻居，而是先发现“这个小数据集覆盖哪些视觉概念簇”，再从簇中扩充。

### 3.6 Balancing：LVD-142M 如何避免被大来源支配

论文没有训练一个显式的“均衡优化器”。

它主要通过**来源级配额、不同 retrieval 倍率和小数据集上限**实现平衡：

- ImageNet-22k 原样加入 14,197,086 张；
- ImageNet-22k retrieval 加入 56,788,344 张；
- ImageNet-1k train retrieval 加入 40,997,344 张；
- Google Landmarks v2 原样加入 1,580,470 张，并检索 6,321,880 张；
- Mapillary SLS 原样加入 1,434,262 张；
- 多数细粒度、分割、深度和小型检索种子各自封顶约 1M；
- AmsterTime 新旧划分分别约 0.96M 和 0.83M。

最终精确规模为：

$$
|\mathrm{LVD\text{-}142M}|=142{,}109{,}386.
$$

这种 balancing 的本质是：

- 大数据源保证覆盖面和基本视觉统计；
- 小数据源通过 cluster retrieval 被“放大”；
- 每个小源的 1M cap 防止某个任务域过度占比；
- retrieval 让训练分布朝有用视觉概念倾斜，而不是对 744M 网页池均匀随机采样。

### 3.7 数据处理的计算代价

去重与检索基于 Faiss：

- 20 个节点；
- 每节点 8 张 V100-32GB；
- 使用 GPU inverted-file indices 与 product quantization；
- 全管线少于两天生成 LVD-142M。

这说明数据治理本身已是一个大规模分布式系统，而不是下载脚本。

### 3.8 数据消融告诉了什么

相同 ViT-g/14、相同迭代数下：

| 预训练数据 | INet-1k | ImageNet-A | ADE20K | Oxford-M | iNat2018 | iNat2021 | Places205 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ImageNet-22k | 85.9 | 73.5 | 46.6 | 62.5 | 81.1 | 85.6 | 67.0 |
| ImageNet-22k 去除 IN1k synsets | 85.3 | 70.3 | 46.2 | 58.7 | 80.1 | 85.1 | 66.5 |
| 随机未整理网页图 142M | 83.3 | 59.4 | 48.5 | 54.3 | 68.0 | 76.4 | 67.2 |
| LVD-142M | 85.8 | **73.9** | 47.7 | **64.6** | **82.3** | **86.4** | **67.6** |

结论不是“curated data 在每个任务都第一”。

未整理网页图在 ADE20K 上反而达到 48.5，高于 LVD 的 47.7；但它在分类、OOD 与检索上严重退化。

LVD-142M 的价值是取得**跨任务更均衡的 Pareto 点**，并在未用于 curation 的 iNaturalist / Places 上仍提高，说明收益不只是把目标 benchmark 样本检索进训练集。

---

## 4. 训练目标：DINO + iBOT + KoLeo

### 4.1 Student–teacher 总体结构

DINOv2 保留 DINO / iBOT 的 online teacher 结构：

- student 接收多个 global crops、local crops，以及被 mask 的 patches；
- teacher 只接收未 mask 的 global views；
- student 通过反向传播更新；
- teacher 不接收梯度，而是 student 参数的指数滑动平均（EMA）；
- teacher momentum 从 0.994 按 cosine schedule 增长到 1。

如果 student 参数为 $\theta_s$，teacher 参数为 $\theta_t$，则：

$$
\theta_t \leftarrow m\theta_t+(1-m)\theta_s.
$$

EMA teacher 相当于时间集成后的目标网络，避免 student 与目标同时剧烈移动。

### 4.2 DINO 图像级目标

对同一图像的不同 crops，student 与 teacher 的 CLS token 经过各自 DINO head，得到 prototype logits。

student 分布为：

$$
p_s=\operatorname{softmax}(z_s/\tau_s).
$$

teacher 分布经温度与 centering / balancing 后记为 $p_t$。

图像级损失是交叉熵：

$$
\mathcal{L}_{\mathrm{DINO}}=-\sum_k p_t^{(k)}\log p_s^{(k)}.
$$

直觉上，它让不同视角共享图像级语义：

- 局部 crop 要预测全局 crop 的语义分配；
- 颜色、尺度、视角扰动不应改变对象身份；
- prototype 不是人工类别，而是训练中形成的表示分区。

DINO 目标主要作用在 CLS token，因此天然偏向全局语义。

### 4.3 iBOT patch 级 masked objective

student 的部分输入 patches 被替换为 mask tokens，teacher 看完整图像。

只在被 mask 的位置 $i$ 上比较 student 与 teacher 的 patch prototype distribution：

$$
\mathcal{L}_{\mathrm{iBOT}}=-\sum_{i\in\mathcal{M}}\sum_k p_{t,i}^{(k)}\log p_{s,i}^{(k)}.
$$

它要求 student 根据上下文恢复 teacher 对局部 patch 的表示，而不是重建 RGB。

与 MAE 的差别非常关键：

- MAE 预测像素，目标包含纹理和低层细节；
- iBOT 预测 teacher representation，目标已被语义化；
- 因而 iBOT 更直接优化 frozen patch feature 的语义可分性。

消融中移除 MIM / iBOT 项：

- ADE20K 线性分割从 47.1 降到 44.2 mIoU；
- ImageNet 从 85.8 降到 85.3；
- Oxford-M 从 63.9 轻微变为 64.3。

这说明 iBOT 的主要价值在 dense prediction，而不是实例检索。

### 4.4 DINO 与 iBOT 使用独立 heads

原始 iBOT 报告共享 DINO / iBOT projection head 更好。

DINOv2 在大规模实验中观察到相反结论，因此解耦两个 heads：

- CLS head 可以专注跨视图全局语义；
- patch head 可以专注 masked local prediction；
- 避免两个粒度在同一 prototype 参数上竞争。

在 Table 1 的最后一步，untying heads：

- k-NN 从 81.7 提升到 82.0；
- linear 从 84.7 轻微降到 84.5。

因此这不是所有指标都增益，而是作者偏向 k-NN 与整体训练行为的选择。

### 4.5 Sinkhorn-Knopp centering

DINO 原始 teacher 采用 running mean centering，再做 sharpened softmax。

DINOv2 按 Ruan et al. 的建议，用 SwAV 风格的 Sinkhorn-Knopp（SK）batch normalization 替代 teacher 的普通 softmax-centering，并运行 3 次迭代。

可以把 SK 理解为在一个 batch 内对 teacher prototype assignments 做近似双随机归一化：

- 每个样本的分配需要归一；
- prototype 的 batch 使用量也被约束得更平衡；
- 防止所有样本集中到少数 prototypes；
- 相比单纯减去移动均值，它显式利用当前 batch 的分配统计。

student 侧仍使用普通 softmax。

要注意 Table 1 中加入 SK 后 ImageNet k-NN / linear 数字没有变化，仍为 81.7 / 84.7。

它的价值更多是：

- 提供稳定的 teacher assignment；
- 与大 batch、更多 prototypes 和后续 head 解耦协同；
- 不能从该单项消融声称它独立贡献了显著准确率。

### 4.6 KoLeo regularizer

KoLeo 来自 Kozachenko–Leonenko differential entropy estimator。

对 batch 内 L2-normalized CLS features $x_1,\dots,x_n$，定义最近邻距离：

$$
d_{n,i}=\min_{j\neq i}\|x_i-x_j\|_2.
$$

正则为：

$$
\mathcal{L}_{\mathrm{KoLeo}}=-\frac{1}{n}\sum_{i=1}^{n}\log d_{n,i}.
$$

最小化它会惩罚彼此过近的样本，推动特征在单位超球面上更均匀展开。

论文只对第一个 global crop 的 CLS tokens 使用 KoLeo，权重为 0.1，并且每张 GPU 内独立计算，不做跨 GPU 通信。

示意性的总目标为：

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{DINO}}
+\mathcal{L}_{\mathrm{iBOT}}
+0.1\mathcal{L}_{\mathrm{KoLeo}}.
$$

KoLeo 消融最能说明它的作用：

| 设置 | INet-1k | ImageNet-A | ADE20K | Oxford-M |
|---|---:|---:|---:|---:|
| 无 KoLeo | 85.3 | 70.6 | 47.2 | 55.6 |
| 有 KoLeo | 85.8 | 72.8 | 47.1 | **63.9** |

它对 ADE20K 几乎无帮助，却让 Oxford-M 检索提高 8.3 mAP。

原因是检索直接依赖 embedding 空间的邻域几何；KoLeo 防止 representation clumping，使余弦最近邻更可用。

### 4.7 为什么三种目标互补

可以把 DINOv2 的 representation 分工概括为：

- **DINO / CLS**：回答“整幅图在语义上是什么”。
- **iBOT / patch**：回答“被遮挡位置在上下文中应表示什么”。
- **KoLeo / batch geometry**：回答“不同图像的全局表示如何铺满空间，避免挤成团”。
- **SK / prototype balancing**：回答“teacher 的离散软分配如何避免 prototype collapse”。

它不是把所有能力压在一个损失上，而是同时约束跨视图一致性、局部可预测性和全局表示几何。

---

## 5. 模型、训练系统与高分辨率适配

### 5.1 ViT 模型族

DINOv2 发布四个 patch size 14 的 ViT：

| 模型 | 参数量 | Embed dim | Heads | Blocks | 训练方式 |
|---|---:|---:|---:|---:|---|
| ViT-S/14 | 21M | 384 | 6 | 12 | 从 ViT-g 蒸馏 |
| ViT-B/14 | 86M | 768 | 12 | 18 | 从 ViT-g 蒸馏 |
| ViT-L/14 | 300M | 1024 | 16 | 24 | 发布版为蒸馏模型 |
| ViT-g/14 | 1.1B | 1536 | 24 | 40 | 从零自监督训练 |

从零训练的 ViT-L/g 使用 SwiGLU FFN；蒸馏 S/B/L 使用普通 MLP FFN。

ViT-g 与原始 giant ViT 配置不同：

- embedding dim 改为 1536；
- 24 heads；
- 每头 64 维；
- 维度选择适配 GPU 矩阵乘内核。

这说明 DINOv2 的架构设计包含明确的 hardware-aware co-design。

### 5.2 主训练超参数

从零训练的 ViT-L/g 使用：

- 625k iterations；
- AdamW；
- batch size 3072；
- base learning rate $3.5\times10^{-4}$；
- 100k iterations warmup；
- weight decay 从 0.04 cosine 增至 0.2；
- stochastic depth 0.4；
- LayerScale 初值 $10^{-5}$；
- teacher momentum 0.994 → 1；
- backbone 使用 float16，DINO heads 的梯度归约使用 float32。

蒸馏 S/B/L 的 batch size 是 2048、学习率 $10^{-3}$、drop-rate 0。

### 5.3 多裁剪与 mask

主训练延续 DINO multi-crop：

- global crops 为较大视野；
- local crops 为较小视野；
- teacher 处理完整 global crops；
- student 同时处理 global / local crops；
- iBOT 在 student patches 上施加随机 mask。

论文实现讨论中以 224 global crop 和 98 local crop 解释 sequence packing。

这使模型同时获得：

- 全局到全局的一致性；
- 局部到全局的语义对齐；
- masked patch 到可见 patch 上下文的预测能力。

### 5.4 高效 attention 与 sequence packing

大 crop 和小 crop 的 token 序列长度不同，常规实现需要多次 forward。

DINOv2 把多条序列拼接成一条长序列，并在 attention 中使用 block-diagonal mask：

- 不同 crop 之间不能互相注意；
- 数学上等价于分别 forward；
- 工程上减少 kernel launch 与 padding 浪费；
- 依赖 xFormers 的 variable-length memory-efficient attention。

### 5.5 Efficient stochastic depth

普通 stochastic depth 先计算 residual branch，再把结果乘 0。

DINOv2 直接跳过被 drop 样本的 residual computation：

- 随机打乱 batch；
- 只对前 $(1-d)B$ 个样本计算 block；
- $d=0.4$ 时可节省接近 drop rate 比例的计算与显存。

这是“训练配方”能够扩展到 1.1B 参数的重要组成，而不是模型精度技巧。

### 5.6 Fully-Sharded Data Parallel

1B 参数 AdamW 训练至少要保存：

- student 参数；
- teacher 参数；
- optimizer 一阶矩；
- optimizer 二阶矩。

仅这些 float32 副本就约 16GB / 1B params。

FSDP 将它们跨 GPU 分片，并以 float16 做 backbone 权重广播和梯度归约，使：

- 单卡显存不再限制模型总规模；
- 相比 float32 DDP，通信量约减半；
- heads 仍用 float32 reduce 以避免数值不稳定。

论文报告同硬件下相对 iBOT：

- 训练速度约 2×；
- 显存约 1/3。

### 5.7 High-resolution adaptation

低分辨率足以学习图像级语义，却会让小物体和边界在 patchification 中消失。

全程高分辨率训练又非常昂贵。

DINOv2 的方案是：

1. 先完成常规分辨率主训练；
2. 从预训练权重继续训练 10k iterations；
3. 把输入提高到 518×518；
4. 把原 schedules 压缩到 10k steps；
5. 保持大多数超参数，仅降低 base learning rate。

![](assets/DINOv2%20-%20无监督鲁棒视觉特征/figures/05-high-resolution-ablation.png)

> **图 3｜论文 Figure 6｜PDF 第 10 页｜URL：https://arxiv.org/pdf/2304.07193。** ViT-L/16 若全程在 416 训练，跨分辨率表现最佳但成本约为 224 的 3 倍；先在 224 训练、末尾短暂切到 416 的 `224→416`，在 ImageNet 与 ADE20K 上接近全程 416，且明显优于只在 224 训练后直接高分辨率推理。图像已由本地官方 PDF 完整裁取并核验。

高分辨率适配解决的是 train–test resolution mismatch：

- 只把低分辨率模型拿到 768 推理，性能会明显下降；
- 短时适配让 positional / attention / patch representation 重新校准；
- 稠密任务收益尤其明显；
- 它不是改变 patch size，而是增加输入 patch 数量。

---

## 6. 蒸馏与 register tokens 的历史边界

### 6.1 DINOv2 的蒸馏不是普通 logit 蒸馏

作者固定 ViT-g/14 作为 teacher，再复用自监督训练框架训练较小 student。

相对主训练有四项变化：

- teacher 是已经训练好的更大模型，完全冻结；
- 保留一个 student EMA，并把 EMA student 作为最终模型；
- 移除 masking；
- 移除 stochastic depth；
- iBOT loss 改为作用于两个 global crops 的 patch tokens。

这意味着 student 同时复制：

- 大教师的 CLS semantic distribution；
- 大教师的 patch-level distribution；
- 大教师经过 LVD-142M 与大模型容量形成的跨域结构。

### 6.2 蒸馏为什么有效

同样是 ViT-L/14：

| 模型 | INet-1k | 8 类任务分割均值 | Depth ↓ | 分类均值 | 细粒度 | 检索 | A/R/Sketch | 视频 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT-L scratch | 84.5 | 72.2 | 1.10 | 90.2 | 75.8 | 71.3 | 69.5 | 67.3 |
| ViT-L distilled | **86.3** | **73.3** | **1.08** | **91.2** | **77.6** | **76.3** | **74.5** | **67.5** |
| ViT-g teacher | 86.5 | 73.4 | 1.00 | 92.1 | 78.3 | 75.2 | 77.0 | 69.3 |

蒸馏 ViT-L 在全部 12 个单项 benchmark 上超过从零训练 ViT-L，部分检索指标甚至超过 teacher。

这不违反知识蒸馏直觉：

- student 架构可能形成更适合某个 evaluation head 的表示；
- teacher soft targets 比原始自监督目标更平滑；
- student EMA 又提供额外时间集成。

![](assets/DINOv2%20-%20无监督鲁棒视觉特征/figures/04-distillation-ablation.png)

> **图 4｜论文 Figure 5｜PDF 第 10 页｜URL：https://arxiv.org/pdf/2304.07193。** 橙色蒸馏 ViT-L 在雷达图的 12 个指标上均包围灰色 scratch ViT-L；右侧汇总显示检索均值 71.3→76.3、OOD 分类 69.5→74.5。蒸馏是发布版小模型强度的主要来源，不应把 S/B/L 当作各自独立从零训练结果。图像已由本地官方 PDF 完整裁取并核验。

### 6.3 原始 DINOv2 论文没有 register tokens

这是最容易被后来的仓库状态混淆的历史点。

**arXiv:2304.07193 / TMLR 版本的原始 DINOv2 模型不使用 register tokens。**

原论文架构是：

- patch tokens；
- 一个 CLS token；
- learned positional embeddings；
- 没有额外的 register tokens。

### 6.4 带 registers 的 DINOv2 是后续版本

后续论文 **Vision Transformers Need Registers**（arXiv:2309.16588）发现大型 ViT 的少量 patch tokens 会出现异常高 norm，并承担类似内部计算缓存的角色。

作者显式加入 register tokens 承接这类全局计算。

官方 DINOv2 仓库在 **2023-10-26** 才加入带 4 个 registers 的模型：

- 原始模型：`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`dinov2_vitg14`；
- 后续 register 模型：对应名称增加 `_reg`；
- 权重文件常带 `reg4`；
- 每个模型增加 4 个 register tokens。

因此引用口径必须区分：

- “DINOv2 原论文证明了什么”应基于无 register 版本；
- “今天仓库里推荐 / 可下载的 DINOv2 变体”可以包含后续 register 版本；
- registers 改善特征伪影的结论属于后续研究，不应倒写成 2023 年 4 月原方法组件。

---

## 7. 实验：frozen features 到底有多强

### 7.1 评测哲学

论文刻意把 backbone 冻结，以检验“信息是否已在线性或简单头部可读的形式存在”。

主要协议包括：

- k-NN：无参数最近邻分类；
- linear probe：冻结 backbone，只训练线性分类器；
- logistic regression：用于 12 个迁移分类基准；
- retrieval：直接按 cosine similarity 排序；
- segmentation linear：每个 patch token 接线性类别头；
- segmentation +ms：拼接最后 4 层、提高分辨率并做 multi-scale 测试；
- depth lin.1 / lin.4：一层或四层 token 接深度分类头；
- depth DPT：冻结 backbone，只训练 DPT decoder。

这种评测比端到端微调更能区分“预训练表示强”与“下游训练配方强”。

代价是：

- linear probing 有较大的超参数 grid search；
- 其数值不是完全“零训练”；
- 对不同方法使用同一 probe 仍可能与各方法最佳特征读取方式不匹配。

### 7.2 八类任务上的 scaling

![](assets/DINOv2%20-%20无监督鲁棒视觉特征/figures/02-task-scaling.png)

> **图 5｜论文 Figure 2｜PDF 第 3 页｜URL：https://arxiv.org/pdf/2304.07193。** 蓝线显示 DINOv2 随模型 FLOPs 增长，在 ImageNet、分割、深度、通用分类、细粒度分类、实例检索、ImageNet-A/R/Sketch 与视频理解上整体上升；它显著高于此前 SSL，并在多数维度逼近或越过最佳 WSL 虚线。图像已由本地官方 PDF 完整裁取并核验。

图中最重要的不是某一个 SOTA 点，而是**同一个 scaling axis 同时改善不同任务**。

这支持“通用 backbone”主张，但也有两个边界：

- 不同任务使用不同 probe / metric，纵轴不能直接横向比较；
- 曲线来自同一数据与方法族，不能证明任意 SSL 扩大都同样受益。

### 7.3 ImageNet frozen classification

| 方法 | 架构 | 数据 | k-NN | Linear | ReaL | V2 |
|---|---|---|---:|---:|---:|---:|
| MAE | ViT-H/14 | IN1K | 49.4 | 76.6 | 83.3 | 64.8 |
| DINO | ViT-S/8 | IN1K | 78.6 | 79.2 | 85.5 | 68.2 |
| iBOT | ViT-L/16 | IN22K | 72.9 | 82.3 | 87.5 | 72.4 |
| OpenCLIP | ViT-G/14 | LAION-2B | 83.2 | 86.2 | 89.4 | 77.2 |
| DINOv2-L | ViT-L/14 | LVD-142M | 83.5 | 86.3 | 89.5 | 78.0 |
| DINOv2-g | ViT-g/14 | LVD-142M | **83.5** | **86.5** | **89.6** | **78.4** |

关键比较：

- DINOv2-g 相对 iBOT linear 提高 **4.2 points**；
- DINOv2-g 略高于 OpenCLIP-G 的 86.2；
- ImageNet-V2 上 DINOv2-g 为 78.4，高于 OpenCLIP-G 的 77.2；
- MAE 的 76.6 说明其原始 frozen linear feature 明显弱，但不能据此否定其 finetuning 能力。

作者还做了 supervised finetuning sanity check：

- 224 分辨率：86.5 linear → 88.5 finetuned；
- 448 分辨率：86.7 linear → 88.9 finetuned。

因此 DINOv2 并非“不能微调”，而是其 frozen feature 已接近微调结果。

### 7.4 OOD 鲁棒性

| 方法 | ImageNet-A ↑ | ImageNet-R ↑ | ImageNet-C ↓ | Sketch ↑ |
|---|---:|---:|---:|---:|
| OpenCLIP-G | 63.8 | **87.8** | 45.3 | **66.4** |
| MAE-H | 10.2 | 34.4 | 61.4 | 21.9 |
| DINO-B/8 | 23.9 | 37.0 | 56.6 | 25.5 |
| iBOT-L/16 | 41.5 | 51.0 | 43.9 | 38.5 |
| DINOv2-g | **75.9** | 78.8 | **28.2** | 62.5 |

DINOv2-g 对 iBOT 的提升很大：

- ImageNet-A：+34.4；
- ImageNet-R：+27.8；
- Sketch：+24.0；
- ImageNet-C error 从 43.9 降到 28.2。

但 OpenCLIP 在 ImageNet-R 和 Sketch 仍更好。

可能原因是语言监督对艺术风格和语义类别变化有强抽象能力，而 DINOv2 在自然对抗样本与 corruption 上更稳。

结论应是两类预训练互补，而不是 DINOv2 全面取代 CLIP。

### 7.5 细粒度分类与视频理解

DINOv2-g 的 frozen linear 结果：

- iNaturalist 2018：81.6；
- iNaturalist 2021：85.7；
- Places205：67.5；
- Kinetics-400：78.4；
- UCF-101：91.2；
- Something-Something v2：38.3。

对比 OpenCLIP-G：

- iNat2018：73.0，DINOv2 +8.6；
- iNat2021：76.0，DINOv2 +9.7；
- Places205：69.8，DINOv2 -2.3；
- K400：78.3，基本持平；
- UCF101：90.7，DINOv2 +0.5；
- SSv2：35.8，DINOv2 +2.5。

iNaturalist 的优势说明纯视觉特征保留了细微形态差异。

SSv2 更依赖物体交互与动作顺序；论文只取 8 帧并做特征拼接，仍能略胜 OpenCLIP，但它并不是时序预训练模型。

在 12 个 SimCLR transfer benchmarks 上：

- DINOv2-g 平均 92.1；
- OpenCLIP-G 平均 91.9；
- iBOT-L 平均 86.6；
- DINO-B/8 平均 85.1；
- MAE-H 平均 78.0。

OpenCLIP 在 SUN、Cars 等数据上仍有优势，说明文本监督对可命名类别和场景很有效。

### 7.6 实例检索

检索不训练参数，直接比较冻结 embedding 的 cosine similarity。

DINOv2-g 结果：

- Oxford Medium / Hard：73.6 / 52.3 mAP；
- Paris Medium / Hard：92.1 / 82.6 mAP；
- Met：GAP 36.8、GAP- 73.6、ACC 76.5；
- AmsterTime：46.7 mAP。

OpenCLIP-G：

- Oxford M/H：50.7 / 19.7；
- Paris M/H：79.2 / 60.2；
- AmsterTime：24.6。

DINOv2 在 Oxford-Hard 上相对 OpenCLIP 提高 32.6 mAP，相对 DINO 提高 38.6 mAP。

值得注意：

- ViT-L 在 Oxford / Paris 上有时略高于 ViT-g；
- 模型规模与所有单项指标并非严格单调；
- KoLeo 和检索导向的数据构成对这类结果非常关键。

### 7.7 语义分割

| 方法 | ADE20K lin / +ms | Cityscapes lin / +ms | VOC lin / +ms |
|---|---:|---:|---:|
| OpenCLIP-G | 39.3 / 46.0 | 60.3 / 70.3 | 71.4 / 79.2 |
| MAE-H | 33.3 / 30.7 | 58.4 / 61.0 | 67.6 / 63.3 |
| DINO-B/8 | 31.8 / 35.2 | 56.9 / 66.2 | 66.4 / 75.6 |
| iBOT-L/16 | 44.6 / 47.5 | 64.8 / 74.5 | 82.3 / 84.3 |
| DINOv2-g | **49.0 / 53.0** | **71.3 / 81.0** | **83.0 / 86.2** |

Linear setup 极其简单：

- 最后一层 patch token 接线性分类器；
- 输出低分辨率 logits；
- 双线性上采样到输入大小。

`+ms` setup 则：

- 拼接最后四层 patch tokens；
- 输入分辨率提高到 640；
- 使用 multi-scale test-time augmentation。

DINOv2-g +ms 的 ADE20K 53.0 mIoU 接近 MAE 全微调 + UperNet 的 53.6。

将冻结 DINOv2-g 接入 ViT-Adapter + Mask2Former 可达 60.2 mIoU；backbone 占系统 66% 参数且保持冻结。

这说明强 frozen backbone 可以降低下游训练范围，但“冻结”不等于系统轻量：adapter、decoder、高分辨率和多尺度推理仍可能很昂贵。

### 7.8 单目深度

| 方法 | NYUd lin.1 / lin.4 / DPT ↓ | KITTI lin.1 / lin.4 / DPT ↓ | NYUd→SUN lin.1 / lin.4 / DPT ↓ |
|---|---:|---:|---:|
| OpenCLIP-G | 0.541 / 0.510 / 0.414 | 3.57 / 3.21 / 2.56 | 0.537 / 0.476 / 0.408 |
| MAE-H | 0.517 / 0.483 / 0.415 | 3.66 / 3.26 / 2.59 | 0.545 / 0.523 / 0.506 |
| DINO-B/8 | 0.555 / 0.539 / 0.492 | 3.81 / 3.56 / 2.74 | 0.553 / 0.541 / 0.520 |
| iBOT-L/16 | 0.417 / 0.387 / 0.358 | 3.31 / 3.07 / 2.55 | 0.447 / 0.435 / 0.426 |
| DINOv2-g | **0.344 / 0.298 / 0.279** | **2.62 / 2.35 / 2.11** | **0.402 / 0.362 / 0.338** |

深度结果是 DINOv2 与 CLIP 差异最鲜明的证据。

caption 通常不描述连续深度关系，而 patch masking 与跨视图一致性迫使模型保留空间线索。

不过应准确表述：

- DINOv2 预训练没有深度标签；
- 下游仍用 NYUd / KITTI 深度监督训练 linear 或 DPT head；
- “zero-shot transfer”指在 NYUd 训练深度头后直接迁移到 SUN RGB-D，不是完全不训练深度头。

### 7.9 定性结果

![](assets/DINOv2%20-%20无监督鲁棒视觉特征/figures/06-dense-results.png)

> **图 6｜论文 Figure 7｜PDF 第 17 页｜URL：https://arxiv.org/pdf/2304.07193。** 在 ADE20K、NYUd、SUN RGB-D 和 KITTI 上，OpenCLIP-G 与 DINOv2-g 均冻结，只训练线性头。DINOv2 的分割区域更连贯，深度图更平滑且能恢复椅子、桌面、车辆等局部结构；这直观支持“caption 对齐容易丢失局部几何”的论点。图像已由本地官方 PDF 完整裁取并核验。

定性图不应被过度解读：

- 作者挑选了有限案例；
- 深度颜色平滑不直接等于 metric depth 更准；
- 仍需以 Table 10/11 的定量结果为主。

### 7.10 训练配方消融

ViT-L / ImageNet-22k 的逐步消融：

| 累积设置 | k-NN | Linear |
|---|---:|---:|
| iBOT 论文值 | 72.9 | 82.3 |
| 作者复现 | 74.5 | 83.2 |
| + LayerScale, Stochastic Depth | 75.4 | 82.0 |
| + 128k prototypes | 76.6 | 81.9 |
| + KoLeo | 78.9 | 82.5 |
| + SwiGLU FFN | 78.7 | 83.1 |
| + patch size 14 | 78.9 | 83.5 |
| + teacher momentum 0.994 | 79.4 | 83.6 |
| + warmup tweaks | 80.5 | 83.8 |
| + batch size 3k | 81.7 | 84.7 |
| + Sinkhorn-Knopp | 81.7 | 84.7 |
| + untying heads | **82.0** | **84.5** |

这张表给出三个重要认识：

1. 没有单一“魔法组件”；提升来自连续系统工程。
2. k-NN 与 linear 不总同步，优化表示邻域结构可能牺牲线性探测。
3. LayerScale / stochastic depth 虽短期掉 linear，却避免 NaN，为更大 batch 和模型提供稳定性。

### 7.11 实验是否支撑论文主张

总体上，实验覆盖面足够广，并且 frozen protocol 与论文“通用特征”主张高度一致。

最有说服力的证据是：

- 图像级与像素级任务同步提升；
- LVD 对未参与 curation 的 iNaturalist / Places 仍有收益；
- 简单 linear / k-NN / cosine retrieval 已可获得强结果；
- ViT-S/B/L/g 形成一致的 scaling family；
- 数据、loss、蒸馏和分辨率均有消融。

仍需保留的质疑是：

- 若干评测训练集参与了 retrieval curation；
- LVD-142M 不公开，最关键变量无法独立复现；
- 大量配方改动以累积方式加入，交互效应没有完整 factorial analysis；
- “无需微调”不等于“无需下游监督或训练”。

---

## 8. 与 DINO、iBOT、MAE、CLIP 的技术对比

| 维度 | DINO | iBOT | MAE | CLIP / OpenCLIP | DINOv2 |
|---|---|---|---|---|---|
| 监督信号 | 图像增强一致性 | 图像 + masked patch teacher | masked pixel reconstruction | 图文配对 | 图像 + masked patch teacher |
| 主要 token | CLS | CLS + patch | patch | 全局图像 / 文本 | CLS + patch |
| teacher | EMA | EMA online tokenizer | 无 EMA teacher | 无 EMA teacher | EMA / 蒸馏时冻结大 teacher |
| 数据 | 主要 IN1K | IN1K / IN22K | 主要 IN1K | 4亿至数十亿图文对 | LVD-142M curated images |
| frozen linear | 强 | 更强 | 相对弱 | 强 | 很强 |
| 稠密几何 | 有涌现 | 强 | 微调后强 | 通常较弱 | 很强 |
| 开放词汇 | 无 | 无 | 无 | 有 | 无 |
| 关键优势 | 对象与部件涌现 | patch 级语义 | 可扩展重建预训练 | 语言接口 | 数据治理 + 多粒度特征 + scaling |

### 8.1 相对 DINO

DINOv2 保留 DINO 的 EMA teacher 与跨 crop CLS 蒸馏，但新增或强化：

- iBOT masked patch objective；
- LVD-142M 数据治理；
- KoLeo；
- Sinkhorn-Knopp；
- 独立 DINO / iBOT heads；
- 1.1B ViT-g；
- 高分辨率适配；
- 大教师到小模型蒸馏；
- FSDP、sequence packing 等可扩展实现。

因此 DINOv2 更像“DINO 系列的系统化 scaling recipe”，不是仅改一条公式。

### 8.2 相对 iBOT

DINOv2 的训练骨架最接近 iBOT。

主要区别是：

- 从 ImageNet 范围扩大到 LVD-142M；
- 共享 heads 改为独立 heads；
- teacher centering 改为 SK；
- 加 KoLeo；
- 更大 prototype 数、batch 与稳定化组件；
- 强调 frozen all-purpose features，而非单一 benchmark。

从 Table 1 看，DINOv2 是在 iBOT 上逐层修正和扩展，而不是否定 iBOT。

### 8.3 相对 MAE

MAE 的 pixel reconstruction 让模型学习“如何补图”，其优势通常在 full finetuning。

DINOv2 的 target 是 teacher feature distribution：

- 不要求恢复每个像素；
- 过滤掉部分难以迁移的低层噪声；
- 直接形成可供 linear head 读取的语义空间；
- 代价是训练结构更复杂，需要 teacher、prototypes、centering 和防坍缩机制。

所以 MAE 与 DINOv2 的优劣取决于使用方式：

- 若允许大规模端到端微调，MAE 不应只用 frozen 表格判断；
- 若要一个开箱即用的 frozen backbone，DINOv2 明显更合适。

### 8.4 相对 CLIP

CLIP 的强项：

- 文本就是分类器接口；
- zero-shot open-vocabulary；
- 风格变化与语义迁移强；
- 可直接做图文检索。

DINOv2 的强项：

- 不需要 paired text；
- 局部 patch structure 更完整；
- 深度、分割、对应和实例检索更强；
- specialized domains 不必先有高质量 caption。

DINOv2 的缺点同样明确：

- 不能直接输入类别文本；
- 不具备语言 grounding；
- 做开放词汇任务仍需额外对齐模型。

正确理解不是“DINOv2 打败 CLIP”，而是：

**CLIP 更像语言可寻址的语义接口；DINOv2 更像几何与局部结构丰富的视觉状态编码器。**

---

## 9. 对后续 3D、Depth Anything 与 DINO-WM 的影响

### 9.1 对 3D 视觉的影响

DINOv2 本身不是 3D reconstruction model，也没有多视图相机模型。

它提供的是 3D 系统急需的前端条件：

- patch features 对视角、风格和姿态变化较稳；
- 局部 descriptor 含语义，可跨弱纹理区域匹配；
- 单目特征线性可读出深度；
- 高分辨率适配改善小物体与边界；
- ViT 的规则 patch grid 容易接入 DPT、cross-view transformer 和 point-map heads。

后续 3D 方法可以把任务从“先学视觉，再学几何”改为：

1. 直接复用 DINOv2 tokenization / encoder；
2. 在其上学习跨视图 attention；
3. 输出 camera、depth、point map 或 tracking。

一个明确例子是 **VGGT**：其消融比较普通 $14\times14$ convolution patchify 与 pretrained DINOv2，报告 DINOv2 带来更好性能、更稳定的早期训练，并降低对学习率 / momentum 的敏感性，最终选为默认 image tokenizer。

因此 DINOv2 对 3D 的影响不是提供现成几何答案，而是把“通用局部视觉对应”变成可下载的基础设施。

### 9.2 对 Depth Anything 的影响

Depth Anything 直接以 DINOv2 为 encoder 初始化，并沿用 patch size 14 的输入约束。

其关键继承有两点：

- DINOv2 已把大量语义与相对几何编码到 patch features；
- dense prediction 不必从随机初始化或只含语言语义的 backbone 开始。

Depth Anything 再通过大规模无标签图像、教师伪深度和 feature alignment，把“通用视觉几何先验”专门化为单目深度编码器。

其论文 Table 11 显示，原始 DINOv2 已是很强初始化，而深度专门化后进一步提升：

- NYUv2 AbsRel：0.066 → 0.056，$\delta_1$：0.973 → 0.984；
- KITTI AbsRel：0.058 → 0.046，$\delta_1$：0.971 → 0.982；
- ADE20K mIoU：58.8 → 59.4。

Depth Anything V2 的最大 teacher 进一步明确采用 DINOv2-Giant。

这条技术链可以概括为：

**DINOv2 学通用 patch geometry → Depth Anything 用海量伪标签把它压向深度等变性与边界精度。**

### 9.3 对 DINO-WM 的影响

DINO-WM 的核心选择是“不预测未来 RGB，而预测未来 DINOv2 patch features”。

它把 DINOv2 当作视觉状态空间：

- 当前图像被编码为空间 patch tokens；
- dynamics model 输入 tokens 与动作；
- 输出未来 patch features；
- 目标图像也编码到同一空间；
- 测试时优化动作序列，使预测未来特征接近目标特征。

这种设计利用 DINOv2 的三个属性：

1. 对颜色、纹理等任务无关变化更不敏感，降低像素预测负担；
2. patch grid 保留物体位置，适合迷宫、推物和粒子操作；
3. 语义相似状态在 feature space 更接近，使 goal matching 可微且更有任务意义。

DINO-WM 因而能在离线轨迹上训练，并在测试时做 zero-shot planning，不需要 expert demonstrations、显式 reward model 或预训练 inverse model。

它也暴露 DINOv2 表示的边界：

- 若 encoder 对控制相关的小位移不敏感，world model 也会忽略；
- 若两个视觉状态语义相似但可达性不同，feature distance 不等于控制 cost；
- 冻结表示把 perception bias 固定进 dynamics model。

DINOv2 对 world model 的真正影响，是推动“在预训练表征空间预测未来”成为像素生成之外的可行路线。

---

## 10. 局限性与批判性思考

### 10.1 LVD-142M 不公开，核心结果不可完全复现

代码和权重是开放的，但训练数据不是。

论文没有发布：

- 142M 图像 URL / ID 列表；
- 744M 候选池清单；
- 完整下载成功率与失效 URL；
- 每个安全过滤器的全部实现与版本；
- 人脸模糊、NSFW 与域名限制的完整规则；
- 所有 retrieval collision 的处理细节。

因此外部研究者可以复现“DINOv2 算法”，却不能严格复现“DINOv2 最终模型”。

而数据消融已证明，LVD 是性能的主要变量之一。

### 10.2 Curation 与 benchmark 的边界并不纯净

LVD 使用了很多 downstream datasets 的训练划分作为 retrieval seeds：

- ImageNet；
- ADE20K；
- Cityscapes；
- VOC；
- NYUv2；
- KITTI；
- Oxford / Paris；
- 多个细粒度数据集。

作者删除了 validation / test 的近重复，并保留 iNaturalist、Places 等未用于 curation 的外域评测。

这降低了直接泄漏风险，但评测分布已参与数据设计。

所以“跨域泛化”应区分：

- seed-covered domain；
- seed 未覆盖但视觉相近 domain；
- 真正远离自然网页图像的 domain。

### 10.3 数据平衡仍是人工任务先验

管线“不使用标签”只指网页候选过滤不依赖文本标签。

种子数据集的选择高度人为，并隐含了目标任务清单：

- 想要深度能力，就加入深度数据集作为 retrieval seed；
- 想要地标检索，就扩充 GLDv2；
- 想要细粒度能力，就为小数据集各分配 1M cap。

因此 LVD-142M 是自动扩充但任务导向的 curated dataset，不是从世界图像中无偏发现的自然分布。

### 10.4 训练成本高

复现一次 DINOv2-g 主训练，论文估计：

- 22,016 A100-40GB GPU-hours；
- 9.7 MWh；
- 约 3.7 tCO2eq（按固定 PUE 1.1 与美国平均碳强度估算）。

整个研究项目估计：

- 约 200k GPU-days；
- 0.5k–1k tCO2eq；
- 还不包含 GPU 制造与报废。

数据 curation 另需 160 张 V100 运行近两天。

开源权重降低了使用门槛，但没有降低独立验证 frontier training 的门槛。

### 10.5 “无需微调”容易被误读

论文的多数下游任务仍需要：

- 训练 linear classifier；
- 训练 logistic regression；
- 训练 DPT decoder；
- 训练 adapter + Mask2Former；
- 搜索 learning rate、读取层和 token pooling 方案。

正确说法是：

**backbone 可以冻结，任务头仍需监督训练。**

只有 k-NN、cosine retrieval、PCA 与 patch matching 更接近无训练直接使用。

### 10.6 没有语言接口

DINOv2 无法像 CLIP 那样：

- 用文本直接定义新类别；
- 做零样本图文检索；
- 通过 prompt 改变任务。

若系统需要开放词汇语义，仍要：

- 加文本对齐阶段；
- 与 CLIP / LLM 视觉接口融合；
- 或训练任务特定分类器。

后续 `dino.txt` 等工作属于扩展，不是原论文能力。

### 10.7 表示并不等于几何真值

DINOv2 的深度、PCA parts 和 dense matching 很强，但它没有：

- 相机内外参；
- metric scale；
- 多视图一致性约束；
- 遮挡与可见性模型；
- 物理动力学。

它提供的是适合几何学习的 representation，而不是直接替代 SfM、MVS、SLAM 或 world dynamics。

### 10.8 公平性仍有明显差距

Dollar Street 上 DINOv2-g：

- 欧洲 89.7；
- 非洲 74.0，差 15.7 points；
- 高收入 90.5；
- 低收入 67.4，差 23.1 points。

论文正文用相对下降口径报告更大的 25.7% 地区差与 31.7% 收入差。

这说明“更大、更精选、更通用”并未自动消除西方与高收入分布偏差。

### 10.9 register 历史混淆影响复现实验

今天从官方仓库下载模型时，用户可能选择 `_reg` 版本。

如果拿 register checkpoint 复现原论文表格，却把差异归因于原 DINOv2 配方，就会产生年代错配。

实验记录必须同时注明：

- backbone size；
- 是否 distilled；
- 是否带 4 registers；
- checkpoint 发布批次；
- 输入分辨率和读取层。

---

## 11. 复现与工程使用建议

### 11.1 若目标是特征抽取

优先明确任务粒度：

- 分类：CLS，或 CLS + mean pooled patch tokens；
- 检索：归一化全局 embedding，注意 KoLeo 带来的邻域结构；
- 分割 / 深度：使用 patch tokens，并考虑最后四层特征；
- 对应：保持高分辨率与规则 patch grid，不要只取 CLS。

### 11.2 若目标是复现论文数字

必须对齐：

- 原始无-register checkpoint；
- ViT-S/B/L/g 与是否 distilled；
- 224、448、518 或 640 输入分辨率；
- linear probe 的 1 层 / 4 层选择；
- 是否拼接 mean pooled patch token；
- 多尺度测试；
- 数据集版本与 split。

论文 linear evaluation 实际对以下参数做 grid search：

- 13 个 learning rates；
- 读取最后 1 层或 4 层；
- 是否拼接平均池化 patch tokens。

因此单一默认 linear head 不一定复现表格最佳值。

### 11.3 若资源有限

不应尝试从零复现 ViT-g。

更现实的层次是：

1. 加载官方 ViT-S/B/L；
2. 在目标域冻结抽特征；
3. 先做 k-NN / linear probe；
4. dense task 再加轻量 DPT / segmentation head；
5. 只有明确观察到 domain gap 时再考虑 LoRA 或小范围 fine-tuning。

### 11.4 若构建自己的 SSL 数据集

DINOv2 最值得复用的不是 142M 这个数字，而是数据原则：

- 先去重，再扩充；
- 相对 benchmark 去重；
- 用目标域种子做视觉 retrieval；
- 给小域设上限与配额；
- 保留未参与 curation 的真实外域评测；
- 单独报告随机网页池与 curated pool 的对照。

---

## 12. 关键术语表

- **Frozen feature**：保持预训练 backbone 权重不变，只读取其输出或训练轻量任务头。
- **EMA teacher**：参数由 student 指数滑动平均得到、无梯度更新的目标网络。
- **Prototype**：DINO / iBOT head 输出维度对应的可学习表示原型，不等于人工类别。
- **Centering**：对 teacher 输出做归一 / 去偏，避免所有样本集中到少数维度。
- **Sinkhorn-Knopp**：交替行列归一化，使 batch assignment 近似平衡的算法。
- **KoLeo**：基于最近邻距离的微分熵正则，推动向量均匀铺展。
- **LVD-142M**：DINOv2 的 142,109,386 图 curated pretraining dataset，未公开发布。
- **Sample-based retrieval**：逐 query image 取固定数量近邻。
- **Cluster-based retrieval**：先找种子覆盖的视觉簇，再从簇中限额采样。
- **Relative deduplication**：相对 benchmark splits 删除近重复，降低评测泄漏。
- **Sequence packing**：把不同长度 crop 序列拼接，用 block-diagonal attention 隔离。
- **High-resolution adaptation**：主训练后短时提高输入分辨率，以低成本校准 dense features。
- **Register token**：后续工作加入的额外全局 tokens；不属于原始 DINOv2 论文模型。

---

## 13. 图表索引

1. **图 1：通用 patch 特征与部件对应。**
   论文 Figure 1，PDF p.2，展示 PCA 后的跨姿态、跨风格部件一致性。

2. **图 2：LVD-142M 数据治理管线。**
   论文 Figure 3，PDF p.4，展示 embedding、deduplication、retrieval 与 curated merge。

3. **图 3：高分辨率适配消融。**
   论文 Figure 6，PDF p.10，比较 224、416 与 224→416。

4. **图 4：蒸馏消融。**
   论文 Figure 5，PDF p.10，比较 ViT-L scratch、ViT-L distilled 与 ViT-g teacher。

5. **图 5：八类任务 scaling。**
   论文 Figure 2，PDF p.3，比较 DINOv2、此前 SSL 与 WSL。

6. **图 6：分割和深度定性结果。**
   论文 Figure 7，PDF p.17，比较冻结 OpenCLIP-G 与 DINOv2-g 加线性头。

---

## 14. 最终评价

DINOv2 最深刻的结论不是“自监督比图文监督更好”，而是：

**视觉基础模型的通用性取决于数据分布、目标粒度、表示几何和训练系统的共同设计。**

DINO 提供了对象中心的自蒸馏；

iBOT 提供了 patch-level masked prediction；

KoLeo 修正了 embedding 邻域；

Sinkhorn-Knopp 平衡了 teacher assignments；

LVD-142M 把数据从“更多”变成“更有覆盖且较平衡”；

ViT-g scaling 与蒸馏把强能力转移到可部署模型；

高分辨率适配则把图像级表征真正推进到稠密视觉。

它的局限同样构成重要遗产：最关键的数据不可复现、训练成本极高、任务种子参与 curation、没有语言接口，而且原始模型与后续 register 版本容易混淆。

尽管如此，DINOv2 仍完成了一个范式转换：

在它之后，深度、3D、机器人与 world model 研究不再总要从像素或 ImageNet 分类特征起步，而可以把一个冻结的、语义与几何兼具的 patch feature field 当作通用视觉状态空间。
