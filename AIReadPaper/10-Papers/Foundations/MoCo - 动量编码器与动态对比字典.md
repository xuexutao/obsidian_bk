---
type: paper-note
title: MoCo - 动量编码器与动态对比字典
aliases:
  - Momentum Contrast
  - MoCo v1
paper: Momentum Contrast for Unsupervised Visual Representation Learning
paper_title: Momentum Contrast for Unsupervised Visual Representation Learning
authors:
  - Kaiming He
  - Haoqi Fan
  - Yuxin Wu
  - Saining Xie
  - Ross Girshick
year: 2020
venue: CVPR 2020
arxiv: "1911.05722"
source: https://arxiv.org/abs/1911.05722
doi: 10.1109/CVPR42600.2020.00975
domain: Foundations
status: complete
tags:
  - paper-reading
  - self-supervised-learning
  - contrastive-learning
  - representation-learning
  - computer-vision
created: 2026-10-15
updated: 2026-10-15
---

# MoCo：动量编码器与动态对比字典

> [!info] 论文信息
> **论文**：*Momentum Contrast for Unsupervised Visual Representation Learning*  
> **作者**：Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick  
> **会议**：CVPR 2020，pp. 9729–9738  
> **arXiv**：[1911.05722](https://arxiv.org/abs/1911.05722)  
> **官方 PDF**：[arXiv PDF](https://arxiv.org/pdf/1911.05722) · [CVPR Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html)  
> **官方代码**：[facebookresearch/moco](https://github.com/facebookresearch/moco)  
> **本地材料**：[[assets/MoCo - 动量编码器与动态对比字典/sources/paper.pdf|paper.pdf]] · [[assets/MoCo - 动量编码器与动态对比字典/sources.md|sources.md]]  
> **核对日期**：2026-10-15

## 1. 一句话总结

MoCo 把对比学习重述为“动态字典查找”，用**大容量 FIFO queue**扩大负样本集合，用**动量更新的 key encoder**减缓字典表征漂移，从而在不依赖超大 mini-batch 的前提下得到大而相对一致的负样本字典；其 ResNet-50 在 ImageNet 线性评估达到 60.6%，并在多项检测与分割迁移任务上达到或超过 ImageNet 监督预训练。

> [!abstract] 核心判断
> MoCo 最重要的贡献不是发明新的正样本构造，也不是更换 InfoNCE，而是把“负样本数量”和“编码器一致性”拆成两个可以分别控制的系统变量：queue 管容量，momentum encoder 管时间一致性。

## 2. 研究背景与问题定义

### 2.1 视觉自监督为何需要“字典”视角

自然语言天然具有离散 token，可以在词表上做预测。

图像则是连续、高维信号，不存在预先给定的语义词典。

早期视觉自监督通常人为设计代理任务，例如拼图、旋转预测、着色或上下文预测。

这些任务能提供监督信号，但也可能把模型能力绑在特定网络结构或局部偏置上。

对比学习换了一个问题：不要求恢复像素或预测人工类别，而要求一个样本的两种相关视图在特征空间靠近，不相关样本远离。

MoCo 进一步把这一过程解释为 dictionary look-up：

- query 是待查询的编码表示；
- key 是字典中的候选表示；
- 正 key 与 query 来自同一原始图像的不同增强；
- 其他 key 被当作负样本；
- 训练目标是在 $K+1$ 个候选中识别唯一正 key。

这个重述把不同的对比学习实现统一到了同一个工程问题：**字典如何构建、更新和保持可比性？**

### 2.2 两个互相牵制的目标

论文提出好的对比字典应同时满足：

1. **large**：负样本足够多，覆盖高维视觉空间中的多样模式；
2. **consistent**：不同 key 最好由相同或相近版本的编码器生成，否则相似度比较混入编码器版本差异。

直接端到端方法的一批 key 都来自当前 mini-batch，编码器一致，但字典大小受显存和 batch size 限制。

memory bank 可以存下全数据集特征，字典很大，但每个样本最后一次被编码的时间不同，陈旧程度横跨整轮训练，编码器版本不一致。

MoCo 的问题因此可以写成：

> 能否让字典规模独立于 mini-batch，同时把字典中 key 的“时间跨度”限制在最近若干步，并让生成 key 的编码器缓慢变化？

## 3. 核心贡献

1. **动态字典框架**：把视觉对比学习系统化为 query 对 key 字典的查找问题，明确提出字典规模与一致性是两个核心轴。
2. **FIFO queue**：复用最近 mini-batch 的 key，使负样本数 $K$ 与当前 batch size 解耦，默认可达 65,536。
3. **momentum encoder**：不用反向传播更新 key encoder，而用 query encoder 的指数移动平均更新它，降低队列中新旧 key 的表征漂移。
4. **Shuffle BN**：打乱 key 分支跨 GPU 的样本分组，阻断 BatchNorm 统计泄漏，避免模型通过 sub-batch “签名”作弊。
5. **迁移实证**：不仅报告 ImageNet 线性评估，还系统比较 VOC、COCO、LVIS、Cityscapes 等检测、实例分割、关键点、DensePose 和语义分割任务。

## 4. 方法详解

### 4.1 总体机制：两个编码器、一条队列、一个对比目标

![[assets/MoCo - 动量编码器与动态对比字典/figures/fig1_queue_mechanism.png]]

> **图 1｜Figure 1｜PDF p.1｜[官方 PDF](https://arxiv.org/pdf/1911.05722)**  
> query 由普通 encoder 编码；key 由缓慢变化的 momentum encoder 编码；当前正 key 与队列中大量历史 key 一起参与相似度 softmax。queue 解耦字典大小与 batch size，动量更新限制字典漂移。

对每张原始图像 $x$，独立采样两种随机增强：

$$
x^q = t_q(x), \qquad x^k = t_k(x).
$$

query 分支输出：

$$
q = f_q(x^q;\theta_q).
$$

key 分支输出：

$$
k_+ = f_k(x^k;\theta_k).
$$

两者都做 L2 归一化，因此点积等价于余弦相似度。

当前样本的 $k_+$ 是正 key。

queue 中此前 mini-batch 产生的 $K$ 个 key 是负 key。

一次前向得到 $1+K$ 个 logits：

$$
\ell = [q^\top k_+, q^\top k_0, q^\top k_1, \ldots, q^\top k_{K-1}] / \tau.
$$

官方实现把正样本固定放在第 0 列，因此交叉熵标签是全 0 向量。

反向传播只更新 $f_q$。

$f_k$ 在 `torch.no_grad()` 中用 EMA 更新。

本轮 $k_+$ 在损失计算后入队，等量最旧 key 出队。

### 4.2 InfoNCE：一个 $K+1$ 分类问题

论文使用的损失为：

$$
\mathcal{L}_q = -\log
\frac{\exp(q\cdot k_+/\tau)}
{\sum_{i=0}^{K}\exp(q\cdot k_i/\tau)}.
$$

其中：

- $q$：当前 query 的单位归一化表示；
- $k_+$：同一原图另一增强得到的正 key；
- $k_i$：一个正 key 与 $K$ 个负 key；
- $K$：负样本数量；
- $\tau$：温度，MoCo v1 默认 $0.07$。

从分类角度，它是在问：“$q$ 在当前字典中对应哪一个 key？”

从几何角度，分子提升正对的相似度，分母同时对所有候选做竞争。

当 $\tau$ 较小时，softmax 更尖锐，模型更关注最难区分的 key。

增大 $K$ 通常让负样本覆盖更丰富，但也会增加 false negatives：语义相同、实例不同的图像仍被推开。

论文没有显式估计互信息，也不需要把 InfoNCE 当成互信息下界才能理解 MoCo；这里更直接的解释是大规模归一化 softmax 查表。

### 4.3 query 与 key：角色不对称但输入语义对称

query/key 输入都来自同一图像的随机增强，因此正对在数据语义上对称。

计算图却不对称：

- query encoder 接收梯度；
- key encoder 不接收梯度；
- query 不进入 queue；
- key 在本轮结束后进入 queue；
- queue 只作为 detached 负样本矩阵。

这种不对称是系统设计，不表示 query 图像比 key 图像更重要。

下一轮重新采样增强时，同一图像完全可能以不同视图出现。

官方实现先把 `encoder_q` 参数复制给 `encoder_k`，再把 key 参数设为 `requires_grad=False`。

### 4.4 为什么用 FIFO queue

若只使用当前 mini-batch，字典大小最多约等于全局 batch size。

MoCo 把最近多个 batch 的 key 保存为矩阵：

$$
Q \in \mathbb{R}^{C\times K},
$$

其中 $C=128$，默认 $K=65{,}536$。

当前 batch 的 keys 在所有 GPU 上 `all_gather` 后共同入队。

queue pointer 按全局 batch size 循环移动。

原始实现为简化实现要求 `K % batch_size == 0`。

FIFO 有三层含义：

1. 让 $K$ 独立于 batch size；
2. 只保留最近若干步产生的 key；
3. 优先删除最陈旧、最可能与当前编码器不一致的 key。

以默认 $K=65{,}536$、全局 batch 256 为例，queue 大约覆盖最近 256 个 iteration 的 key。

因此 MoCo 不是“无限历史记忆”，而是一个有明确时间窗口的滑动字典。

### 4.5 momentum encoder 更新公式

key encoder 参数按下式更新：

$$
\theta_k \leftarrow m\theta_k + (1-m)\theta_q,
$$

其中 $m\in[0,1)$，默认 $m=0.999$。

只有 $\theta_q$ 由 SGD 更新。

$	heta_k$ 是 $	heta_q$ 历史轨迹的指数移动平均。

将递推展开可见：

$$
\theta_k^{(t)}=(1-m)\sum_{j=0}^{t-1}m^j\theta_q^{(t-j)} + m^t\theta_k^{(0)}.
$$

当 $m=0.999$ 时，最新 query 参数只贡献 $0.001$。

这让 key encoder 的变化时间尺度远慢于 query encoder。

EMA 不能让 queue 中所有 key 完全同分布，因为每个 key 仍由不同训练步的 $f_k$ 生成。

它做的是把相邻 key encoder 的参数差异压小，使字典具有**近似时间一致性**。

### 4.6 dictionary consistency 到底是什么

一致性不是指所有 key 数值固定。

也不是指 key encoder 永远不变。

它指的是：同一 query 与不同负 key 的相似度，应尽量在同一种表征坐标系中可比。

如果编码器每一步剧烈改变，那么旧 key 的方向变化既包含图像内容，也包含编码器版本偏移。

此时 InfoNCE 看到的是“样本差异 + 时间漂移”的混合量。

queue 控制 key 的最大年龄。

momentum encoder 控制单位时间内编码器的变化幅度。

两者共同约束总漂移。

可以把粗略漂移风险理解为：

$$
\text{drift risk} \propto \text{queue age} \times \text{encoder update speed}.
$$

这不是论文给出的严格公式，但准确概括了设计权衡：$K$ 越大，字典越丰富，也越需要更慢的 key encoder。

### 4.7 三种对比机制的结构差异

![[assets/MoCo - 动量编码器与动态对比字典/figures/fig2_contrastive_mechanisms.png]]

> **图 2｜Figure 2｜PDF p.3｜[官方 PDF](https://arxiv.org/pdf/1911.05722)**  
> end-to-end 对 query/key 两支都反传，字典受当前 batch 限制；memory bank 存储逐样本旧特征；MoCo 只对 query 分支反传，用 momentum encoder 产生新 key，并由 queue 维护近期历史。

| 机制 | 字典来源 | key 更新 | 字典规模 | 一致性 | 主要代价 |
|---|---|---|---:|---|---|
| end-to-end | 当前 batch | 反向传播 | 受 batch 限制 | 高 | 大 batch 显存与优化困难 |
| memory bank | 全数据集缓存 | 样本被访问时更新其特征 | 很大 | 较低 | 全数据索引、陈旧跨度可达一轮 |
| MoCo | 最近多个 batch 的 queue | key encoder 的 EMA | 很大 | 较高 | 双编码器、队列与 DDP 通信 |

memory bank 中的“momentum”如果存在，是对单个样本表示做更新。

MoCo 的 momentum 是对**整个 encoder 参数**做更新。

两者不应混淆。

### 4.8 Shuffle BN：阻断统计泄漏

标准 ResNet 含 BatchNorm。

在多 GPU 训练中，每张 GPU 通常独立计算局部 BN 统计量。

若 query 和对应 positive key 总被分到相同 GPU/sub-batch，它们共享相似的局部均值与方差。

模型可能利用这种统计“签名”定位正样本，而不是学习语义不变性。

MoCo 的做法是：

1. 在 key 分支编码前聚合全局 batch；
2. 生成一个全局随机排列；
3. 把打乱后的 key 图像重新分发到各 GPU；
4. 各 GPU 用本地 BN 编码；
5. 编码后按逆排列恢复样本顺序；
6. query 分支保持原顺序。

这样，正 query 与正 key 的 BN 统计来自不同随机子集。

注意 Shuffle BN 不是把 BN 替换成 LayerNorm，也不是同步 BN。

它保留 BN 的优化收益，但切断可被代理任务利用的配对泄漏。

### 4.9 一次训练迭代的精确顺序

1. 对每张图像独立生成 `im_q` 与 `im_k`。
2. 用 `encoder_q` 得到 $q$ 并 L2 归一化。
3. 用当前 $\theta_q$ 对 $\theta_k$ 做 momentum update。
4. 在所有 GPU 间打乱 `im_k`。
5. 用 `encoder_k` 得到 $k$ 并 L2 归一化。
6. 将 $k$ 恢复到与 $q$ 对齐的顺序。
7. 计算正 logits：逐行 $q\cdot k$。
8. 计算负 logits：$qQ$。
9. 拼接为 $N\times(1+K)$ logits，并除以 $T$。
10. 用全 0 标签计算交叉熵。
11. 将当前全局 batch 的 $k$ 入队并覆盖最旧位置。
12. 对 loss 反向传播，只更新 `encoder_q`。

论文伪代码把 SGD、EMA 与入队画在概念顺序中；官方实现把 EMA 放在 key 前向之前，并在 forward 末尾更新 queue。

关键不变量是：queue 与 key encoder 不接收 loss 梯度，当前正 key 不作为本样本的历史负样本参与本轮计算。

### 4.10 默认训练配置

| 项目 | MoCo v1 默认值 |
|---|---:|
| backbone | ResNet-50 |
| 特征维度 | 128 |
| queue size $K$ | 65,536 |
| key momentum $m$ | 0.999 |
| temperature $\tau$ | 0.07 |
| global batch size | 256 |
| optimizer | SGD |
| SGD momentum | 0.9 |
| weight decay | $10^{-4}$ |
| 初始学习率 | 0.03 |
| epochs | 200 |
| 学习率衰减 | epoch 120、160 各乘 0.1 |
| 硬件 | 8 GPU |
| ResNet-50 训练时间 | 约 53 小时 |

MoCo v1 的增强包括 RandomResizedCrop、随机灰度、ColorJitter、水平翻转与 ImageNet normalization。

官方仓库报告 8×V100 上 5 次完整预训练加线性评估为 $60.8\pm0.2$，单个发布 checkpoint 为 60.6。

官方实现只支持 DistributedDataParallel；单 GPU 路径会抛出 `NotImplementedError`。

## 5. 实验设计与结果

### 5.1 预训练数据与评测协议

主要预训练数据为 ImageNet-1M，即约 128 万张 ImageNet 训练图像，但不使用标签。

论文还使用 Instagram-1B，实际约 9.4 亿张公开图片，来自约 1500 个与 ImageNet 类别相关的 hashtag。

IG-1B 更长尾、更不均衡，也更接近非精选现实数据。

ImageNet 线性评估冻结 backbone，只在 global average pooling 特征上训练一个线性分类器 100 epochs。

线性分类器最优初始学习率为 30，weight decay 为 0。

指标是 ImageNet validation 的 single-crop top-1 accuracy。

迁移实验则端到端 fine-tune 所有层，并尽量让无监督与监督预训练使用相同架构、schedule 和超参数。

### 5.2 字典大小与机制消融

![[assets/MoCo - 动量编码器与动态对比字典/figures/fig3_dictionary_ablation.png]]

> **图 3｜Figure 3｜PDF p.5｜[官方 PDF](https://arxiv.org/pdf/1911.05722)**  
> 三种机制都从更大的 $K$ 获益；end-to-end 受 batch 上限停在 1024，memory bank 可继续扩大但始终落后，MoCo 在 $K=65{,}536$ 达到 60.6%。

关键数值：

| $K$ | end-to-end | memory bank | MoCo |
|---:|---:|---:|---:|
| 256 | 54.9 | 50.0 | 54.7 |
| 512 | 56.3 | 52.0 | 56.4 |
| 1,024 | 57.3 | 54.1 | 57.5 |
| 4,096 | — | 56.5 | 59.0 |
| 16,384 | — | 57.8 | 60.4 |
| 65,536 | — | 58.0 | 60.6 |

结论一：大字典不是 MoCo 独享的收益，三种机制都随 $K$ 增大而改善。

结论二：MoCo 的优势不仅来自“大”，还来自比 memory bank 更一致的 key 编码器。

在相同 $K=65{,}536$ 和 InfoNCE 下，MoCo 60.6%，memory bank 58.0%，差 2.6 个点。

结论三：end-to-end 在小 $K$ 与 MoCo 接近，说明 queue/EMA 的价值主要在把机制扩展到更大字典。

论文在 8×32GB Volta 上能承受的最大 end-to-end batch 是 1024。

batch 1024 还需要线性学习率缩放；不缩放会下降约 2 个点。

这正是 MoCo 不能被简化为“换一种写法的大 batch”的证据。

### 5.3 momentum 系数消融

该消融固定 $K=4096$：

| momentum $m$ | 0 | 0.9 | 0.99 | 0.999 | 0.9999 |
|---:|---:|---:|---:|---:|---:|
| top-1 | fail | 55.2 | 57.8 | **59.0** | 58.9 |

$m=0$ 相当于每步直接把 key encoder 复制为 query encoder。

此时 queue 仍保存旧 key，但生成坐标系每步快速跳动，loss 振荡且无法收敛。

$m=0.9$ 能训练，却比 $m=0.999$ 低 3.8 个点。

$0.99$ 到 $0.9999$ 都较稳健，说明核心不是某个极窄的神奇值，而是“足够慢”。

$m=0.9999$ 没有继续提高，提示过慢会让 key encoder 对 query 学到的新表征响应不足。

### 5.4 ImageNet 线性评估

![[assets/MoCo - 动量编码器与动态对比字典/figures/fig4_imagenet_linear.png]]

> **图 4｜Table 1 内嵌结果图｜PDF p.6｜[官方 PDF](https://arxiv.org/pdf/1911.05722)**  
> 橙色为 MoCo；随着 backbone 加宽，MoCo 从 R50 的 60.6% 提升到 R50w4× 的 68.6%，在相近参数量下优于当时多数方法。

MoCo 的主结果：

| backbone | 参数量 | top-1 |
|---|---:|---:|
| R50 | 24M | 60.6 |
| RX50 | 46M | 63.9 |
| R50w2× | 94M | 65.4 |
| R50w4× | 375M | 68.6 |

相近规模基线包括：

| 方法 | backbone | 参数量 | top-1 |
|---|---|---:|---:|
| InstDisc | R50 | 24M | 54.0 |
| LocalAgg | R50 | 24M | 58.8 |
| BigBiGAN | R50 | 24M | 56.6 |
| CPC v2 | R170 wider | 303M | 65.9 |
| AMDIM-large | 自定义 | 626M | 68.1 |
| MoCo | R50 | 24M | **60.6** |
| MoCo | R50w4× | 375M | **68.6** |

MoCo R50 的优势不是绝对最高精度，而是用标准 ResNet-50 和 24M 参数获得当时同规模最强或极具竞争力的结果。

标准 backbone 也降低了迁移时结构不匹配的干扰。

论文后来在 camera-ready 中提到 MoCo v2：更强增强、MLP projection head 与 cosine schedule 可将 R50 提到 71.1%。

这说明 queue/EMA 是底层机制，增强与投影头是正交改进。

### 5.5 Shuffle BN 消融

![[assets/MoCo - 动量编码器与动态对比字典/figures/fig5_shuffle_bn_ablation.png]]

> **图 5｜Figure A.1｜PDF p.10｜[官方 PDF](https://arxiv.org/pdf/1911.05722)**  
> 虚线是代理任务查表训练准确率，实线是 ImageNet kNN 验证准确率；无 Shuffle BN 时训练准确率迅速超过 99.9%，但验证曲线随后下降，形成典型“任务作弊”。

没有 Shuffle BN 时：

- 代理任务训练准确率几乎饱和；
- kNN 验证准确率先升后降；
- 训练越久，表征质量反而越差。

有 Shuffle BN 时：

- 代理任务仍可拟合，但不再利用稳定 sub-batch 签名；
- kNN 验证准确率持续提升并明显更高。

这个消融很重要，因为低 contrastive loss 并不保证学到可迁移语义。

代理任务存在信息捷径时，训练指标甚至会与下游质量负相关。

### 5.6 PASCAL VOC 检测迁移

在 VOC `trainval07+12` 上 fine-tune Faster R-CNN，`test2007` 评测，24k iterations，5 次平均。

R50-dilated-C5：

| 预训练 | AP50 | AP | AP75 |
|---|---:|---:|---:|
| 监督 IN-1M | 81.4 | 54.0 | 59.1 |
| MoCo IN-1M | 81.1 | **54.6** | **59.9** |
| MoCo IG-1B | **81.6** | **55.5** | **61.2** |

R50-C4：

| 预训练 | AP50 | AP | AP75 |
|---|---:|---:|---:|
| 监督 IN-1M | 81.3 | 53.5 | 58.8 |
| MoCo IN-1M | **81.5** | **55.9** | **62.6** |
| MoCo IG-1B | **82.2** | **57.2** | **63.7** |

最显著的提升出现在更严格的 AP75，而不是 AP50。

这暗示无监督表征并非只改善粗类别可分性，还可能改善定位敏感的细粒度特征。

同一预训练在 C4 上的优势大于 dilated-C5，说明迁移结论依赖下游架构。

三种对比机制在 VOC C4 上直接比较：

| 机制 | AP50 | AP | AP75 |
|---|---:|---:|---:|
| end-to-end | 80.4 | 54.6 | 60.3 |
| memory bank | 80.6 | 54.9 | 60.6 |
| MoCo | **81.5** | **55.9** | **62.6** |

MoCo 在所有指标领先，支持 queue + momentum encoder 的收益可以迁移到检测，而不只是线性分类。

### 5.7 VOC 小数据设置

只用 `trainval2007` 约 5k 张图 fine-tune C4 backbone 时：

| 预训练 | AP50 | AP | AP75 |
|---|---:|---:|---:|
| 监督 IN-1M | 74.4 | 42.4 | 42.7 |
| MoCo IN-1M | 74.9 | 46.6 | 50.1 |
| MoCo IG-1B | **75.6** | **47.6** | **51.7** |

MoCo IG-1B 相对监督预训练提升 +1.2 AP50、+5.2 AP、+9.0 AP75。

数据更少时预训练先验更重要，差距比 `07+12` 设置更大。

### 5.8 COCO 检测与实例分割

Mask R-CNN R50-FPN，2× schedule：

| 预训练 | bbox AP | AP50 | AP75 | mask AP | AP50 | AP75 |
|---|---:|---:|---:|---:|---:|---:|
| 监督 IN-1M | 40.6 | 61.3 | 44.4 | 36.8 | 58.1 | 39.5 |
| MoCo IN-1M | 40.8 | 61.6 | 44.7 | 36.9 | 58.4 | 39.7 |
| MoCo IG-1B | **41.1** | **61.8** | **45.1** | **37.4** | **59.1** | **40.2** |

Mask R-CNN R50-C4，2× schedule：

| 预训练 | bbox AP | AP50 | AP75 | mask AP | AP50 | AP75 |
|---|---:|---:|---:|---:|---:|---:|
| 监督 IN-1M | 40.0 | 59.9 | 43.1 | 34.7 | 56.5 | 36.9 |
| MoCo IN-1M | 40.7 | 60.5 | 44.1 | 35.4 | 57.3 | 37.6 |
| MoCo IG-1B | **41.1** | **60.7** | **44.8** | **35.6** | **57.4** | **38.1** |

1× schedule 下 FPN 的 MoCo IN-1M 略低于监督基线，说明论文结论不是“所有短 schedule 都赢”。

2× schedule 下，MoCo 在 FPN 和 C4 的全部指标超过监督对应项。

附录的 6× FPN 结果进一步显示：

- random init bbox AP 41.4；
- 监督 IN-1M 41.9；
- MoCo IN-1M 42.3；
- MoCo IG-1B 42.8。

长 schedule 让随机初始化追近监督预训练，却没有消除 MoCo 优势。

### 5.9 更多迁移任务

COCO keypoint detection：

| 预训练 | AP | AP50 | AP75 |
|---|---:|---:|---:|
| 监督 IN-1M | 65.8 | 86.9 | 71.9 |
| MoCo IN-1M | 66.8 | 87.4 | 72.5 |
| MoCo IG-1B | **66.9** | **87.8** | **73.0** |

COCO DensePose：

| 预训练 | AP | AP50 | AP75 |
|---|---:|---:|---:|
| 监督 IN-1M | 48.3 | 85.6 | 50.6 |
| MoCo IN-1M | 50.1 | 86.8 | 53.9 |
| MoCo IG-1B | **50.6** | **87.0** | **54.3** |

IG-1B 在 DensePose AP75 上比监督预训练高 3.7 个点。

LVIS v0.5 instance segmentation：

| 预训练 | mask AP | AP50 | AP75 |
|---|---:|---:|---:|
| 监督 IN-1M | 24.4 | 37.8 | 25.8 |
| MoCo IN-1M | 24.1 | 37.4 | 25.5 |
| MoCo IG-1B | **24.9** | **38.2** | **26.4** |

IN-1M MoCo 略低于最佳监督设置，IG-1B 才全面超过。

Cityscapes instance segmentation：

- 监督 IN-1M：mask AP 32.9，AP50 59.6；
- MoCo IN-1M：32.3，59.3；
- MoCo IG-1B：32.9，60.3。

Cityscapes semantic segmentation：

- 监督 IN-1M：74.6 mIoU；
- MoCo IN-1M：75.3；
- MoCo IG-1B：75.5。

VOC semantic segmentation 是明确负例：

- 监督 IN-1M：74.4 mIoU；
- MoCo IN-1M：72.5；
- MoCo IG-1B：73.6。

论文汇总称在 7 个检测或分割任务上超过监督预训练：VOC/COCO 检测、COCO/LVIS 实例分割、COCO keypoint、COCO DensePose、Cityscapes semantic segmentation。

这一汇总应按“任务上存在受控设置中的优势”理解，不能外推为每个指标都领先。

### 5.10 baseline 与证据链总结

论文的 baseline 设计有三层：

1. **机制 baseline**：end-to-end 与 memory bank，固定代理任务和 InfoNCE；
2. **表征 baseline**：当时的 InstDisc、LocalAgg、CPC、CMC、AMDIM 等；
3. **迁移 baseline**：random init 与 ImageNet supervised pre-training。

最有说服力的是第一层，因为它尽量只改变字典机制。

第二层受 backbone、参数量、增强和代理任务差异影响，需要看 accuracy—parameters 权衡。

第三层说明 MoCo 的价值不止在线性 probe，但迁移结果依赖任务、骨干与 schedule。

### 5.11 实验证据是否充分

对论文的核心主张“large and consistent dictionary”而言，证据链总体充分。

$K$ 消融验证了 large，$m$ 消融验证了 slowly evolving encoder，三机制对照把两者组合后的收益单独识别出来。

Shuffle BN 曲线又排除了一个重要的代理任务捷径。

迁移部分覆盖分类、检测、实例分割、关键点、DensePose 与语义分割，比只报告线性 probe 更有说服力。

但仍有三项缺口。

第一，没有直接测量 queue 中 key 的特征漂移或跨时间相似度，只用最终准确率间接支持 consistency。

第二，$K$ 与 $m$ 主要分别消融，缺少二维网格，无法观察“大 queue 是否系统地要求更高 momentum”。

第三，IG-1B 对普通研究者不可复现，且数据增益与数据来源先验纠缠。

因此实验足以证明 MoCo 机制有效，但不足以给出 $K$、$m$、学习率和训练动态之间的完整理论规律。

## 6. 为什么 MoCo 不是单纯“大 batch SimCLR”

### 6.1 负样本来源不同

SimCLR 的负样本来自当前全局 batch。

batch 越大，负样本越多，同时要求更多显存、跨设备通信和大 batch 优化技巧。

MoCo 的负样本来自跨多个历史 batch 的 queue。

因此在相同 batch size 下，MoCo 可以有远大于 batch 的 $K$。

### 6.2 key encoder 的时间语义不同

SimCLR 两个视图通常通过共享、当前、可反传的 encoder 计算，所有本轮 key 完全同步。

MoCo 的 key 由 EMA encoder 计算，并可在 queue 中停留多步。

MoCo 必须主动管理时间一致性；SimCLR 不需要处理历史 key 漂移。

### 6.3 优化耦合不同

SimCLR 中 batch size 同时决定：

- 每步样本数；
- 负样本数；
- BN 统计；
- 梯度估计；
- 学习率缩放策略。

MoCo 用 queue 把负样本数从这些因素中解耦。

因此 MoCo 可以在 batch 256 下使用 65,536 个负样本，而不是把 batch 扩成 65,536。

### 6.4 共同点

两者都依赖两种随机增强形成正对。

两者都用归一化表示、温度缩放和 InfoNCE 类 softmax。

两者都表明数据增强与投影头等细节对最终精度极其重要。

MoCo v2 直接吸收了 SimCLR 的强增强和 MLP head，进一步证明两条路线是可组合而非互斥的。

## 7. 与 BYOL、DINO 的关系

### 7.1 与 BYOL

BYOL 也有 online encoder 与 EMA target encoder。

因此从优化结构看，BYOL 延续了 MoCo 的“快速在线网络 + 慢速目标网络”。

关键差异是 BYOL 不使用显式负样本 queue，也不做 $K+1$ 类字典查找。

BYOL 增加 predictor，并通过 online prediction 匹配 stop-gradient target 表示避免坍塌。

MoCo 的防坍塌主要来自正负样本竞争；BYOL 说明 EMA teacher 可以脱离显式 negatives 使用。

### 7.2 与 DINO

DINO 采用 student/teacher 架构，teacher 也是 student 参数的 EMA。

DINO 不维护负样本 queue，而对不同视图的类别概率分布做自蒸馏。

它用 teacher centering 与 temperature sharpening 管理坍塌，并在 ViT 上展现突出的语义结构。

从谱系看，MoCo 把 EMA encoder 证明为稳定目标/字典的有效机制；BYOL 与 DINO 把这一机制推广到无显式 negatives 的自蒸馏。

### 7.3 概念演化

MoCo：**稳定历史 negatives**。

BYOL：**稳定无负样本 target**。

DINO：**稳定 teacher 分布与多视图自蒸馏**。

共同核心是 target 分支不能与 online/student 分支同速追逐，否则目标分布过于非平稳。

## 8. 局限与批判性思考

### 8.1 false negatives

实例判别把不同图像全部视为负样本。

两张语义相同的狗图也会互相排斥。

大 queue 增大了覆盖范围，也提高了遇到语义 false negative 的概率。

MoCo v1 没有类别感知、聚类或近邻过滤机制。

### 8.2 一致性只是近似

EMA 缓解但不消除 stale keys。

更大的 queue 延长 key 的驻留时间。

$m$、学习率、batch size、$K$ 和训练阶段共同决定漂移，论文主要做了单因素消融。

### 8.3 依赖工程细节

Shuffle BN、全局 gather、队列指针、正样本顺序恢复都必须正确。

一个看似能收敛的实现仍可能通过 BN 泄漏获得虚假低 loss。

官方 v1 代码只支持 DDP，提高了入门和调试成本。

### 8.4 算力与数据成本

标准配置是 8 GPU、200 epochs、约 53 小时。

IG-1B 使用 64 GPU、batch 1024、1.25M iterations，约 6 天。

因此“无标签”不等于低成本：标签成本被数据、算力、存储和分布式系统成本替代。

### 8.5 IG-1B 的可复现性

IG-1B 不是普通研究者可直接获得的标准公开训练集。

hashtag 选择还带有弱语义先验，并非完全无策展数据。

论文也承认从 IN-1M 到 IG-1B 的增益持续但相对有限，说明简单实例判别没有充分吃到十亿图规模。

### 8.6 迁移结论有边界

VOC semantic segmentation 仍落后监督预训练。

LVIS 的 IN-1M MoCo 也未超过最佳监督设置。

结果依赖 backbone、BN 策略和 fine-tuning schedule。

“关闭差距”应理解为多个受控任务上的经验结论，不是普遍定理。

### 8.7 时代局限

MoCo v1 使用 ResNet 和显式负样本，尚未探索 ViT、无负样本自蒸馏或 masked image modeling。

其 60.6% 线性精度很快被 MoCo v2、SimCLR 系列、BYOL、SwAV、DINO 等超过。

但后续精度提升并不削弱其基础贡献：MoCo 把 EMA target encoder 和动态字典明确工程化、可消融化。

## 9. 复现成本与检查清单

### 9.1 最小硬件与数据

完整论文设置需要 ImageNet-1K 训练集和多 GPU DDP。

官方复现参考为 8×V100、batch 256、200 epochs、约 53 小时。

4 GPU 可用 batch 128、学习率 0.015，README 称可得相近结果。

官方 v1 实现不直接支持单 GPU。

### 9.2 必须对齐的超参数

- ResNet-50；
- feature dim 128；
- $K=65{,}536$；
- $m=0.999$；
- $T=0.07$；
- batch 256；
- SGD lr 0.03；
- optimizer momentum 0.9；
- weight decay $10^{-4}$；
- 200 epochs；
- epoch 120/160 衰减；
- MoCo v1 augmentation；
- Shuffle BN；
- queue 的跨 GPU gather。

### 9.3 常见复现陷阱

1. 把 optimizer momentum 0.9 与 key encoder momentum 0.999 混淆。
2. 入队前未跨 GPU gather，导致每个 rank 的 queue 不同。
3. 忘记对 q、k 和初始 queue 做 L2 normalization。
4. 对 encoder_k 开启梯度，破坏 EMA target 设计。
5. 先入队再算当前 loss，使当前正 key 同时混入 negatives。
6. 打乱 key 后忘记 unshuffle，正对错位。
7. 使用普通 BN 配对分组，出现代理任务作弊。
8. 把 MoCo v2 的 MLP/强增强/cosine schedule 当成 v1 默认配置。
9. 线性评估时未冻结 backbone 或错误更新 BN statistics。
10. 只看训练 InfoNCE accuracy，不监控 kNN 或线性 probe。

### 9.4 合理的复现验收

- 预训练 loss 稳定下降，但查表准确率不应靠 BN 泄漏瞬间饱和；
- queue pointer 在各 rank 保持一致；
- encoder_k 不出现在 optimizer 可训练参数中；
- 参数更新后 $\theta_k$ 与 $\theta_q$ 接近但不完全相等；
- 线性 probe 接近论文 60.6 或官方 5 次均值 $60.8\pm0.2$；
- 若资源不足，优先验证 K/m/Shuffle BN 的趋势，而不是声称完整复现。

## 10. 进一步启发

### 10.1 “目标稳定性”比“目标是否有标签”更基础

自监督目标本身也在随模型变化。

如果正负目标更新过快，网络面对的是移动靶。

MoCo 的 EMA encoder 提供了低通滤波后的目标网络。

这一思想后来超越了对比学习，进入 teacher–student、自蒸馏和半监督学习。

### 10.2 把系统变量解耦

queue 的工程价值是把 dictionary size 从 mini-batch size 解耦。

好的算法设计经常不是加入更复杂的损失，而是解除两个原本被实现绑定的变量。

这使每个变量可以独立扩展、消融和调参。

### 10.3 代理任务必须防作弊

Shuffle BN 展示了一个通用原则：模型会利用任何比语义更便宜的统计捷径。

设计自监督任务时，应检查视图之间是否共享文件格式、裁剪边界、batch 统计、位置编码或数据管线标识。

低 pretext loss 不是目标，迁移表征质量才是目标。

### 10.4 线性 probe 与迁移应并列报告

MoCo 的意义很大一部分来自其下游检测与分割实验。

线性 probe 测量冻结特征的线性可分性，但不能完整代表定位、密集预测和 end-to-end adaptability。

现代基础模型评估仍应保留多种下游协议。

## 11. 关键术语

- **query**：当前待匹配的编码表示，由可反传的 query encoder 产生。
- **key**：字典候选表示；正 key 与 query 来自同一图像，负 key 来自其他图像。
- **dynamic dictionary**：内容与编码器都会随训练演化的特征字典。
- **queue**：保存最近 key 的 FIFO 缓冲区，使负样本数独立于 batch size。
- **momentum encoder**：通过 query encoder 参数 EMA 更新的 key encoder。
- **InfoNCE**：把正 key 与大量负 key 放进归一化 softmax 的对比损失。
- **temperature**：控制 softmax 尖锐程度的缩放参数。
- **dictionary consistency**：不同 key 尽量处在相近编码器版本定义的表征坐标系中。
- **Shuffle BN**：打乱 key 分支跨 GPU 分组，避免正对共享 BN 子批统计。
- **linear protocol**：冻结 backbone，只训练线性分类头的表征评估协议。
- **false negative**：语义相近但因实例不同而被对比损失当作负样本的样本。

## 12. 图表索引

1. **Figure 1，PDF p.1**：MoCo query—momentum key—queue 总体机制。  
   本地：`assets/MoCo - 动量编码器与动态对比字典/figures/fig1_queue_mechanism.png`  
   官方：[arXiv PDF](https://arxiv.org/pdf/1911.05722)
2. **Figure 2，PDF p.3**：end-to-end、memory bank、MoCo 三种机制。  
   本地：`assets/MoCo - 动量编码器与动态对比字典/figures/fig2_contrastive_mechanisms.png`  
   官方：[arXiv PDF](https://arxiv.org/pdf/1911.05722)
3. **Figure 3，PDF p.5**：dictionary size 与三种机制的线性准确率。  
   本地：`assets/MoCo - 动量编码器与动态对比字典/figures/fig3_dictionary_ablation.png`  
   官方：[arXiv PDF](https://arxiv.org/pdf/1911.05722)
4. **Table 1 内嵌结果图，PDF p.6**：ImageNet 线性评估准确率—参数量。  
   本地：`assets/MoCo - 动量编码器与动态对比字典/figures/fig4_imagenet_linear.png`  
   官方：[arXiv PDF](https://arxiv.org/pdf/1911.05722)
5. **Figure A.1，PDF p.10**：Shuffle BN 消融。  
   本地：`assets/MoCo - 动量编码器与动态对比字典/figures/fig5_shuffle_bn_ablation.png`  
   官方：[arXiv PDF](https://arxiv.org/pdf/1911.05722)

## 13. 最终结论

MoCo 的核心不是“更多 negatives”这一句口号，而是一套解决历史 negatives 可用性的机制：queue 提供规模，momentum encoder 提供时间一致性，Shuffle BN 阻断捷径。

其最强证据来自机制控制实验：相同代理任务和 InfoNCE 下，MoCo 在大 $K$ 时显著优于 memory bank，并且较小 momentum 会明显退化乃至训练失败。

其影响则超出显式对比学习：EMA target encoder 成为 BYOL、DINO 等后续 teacher–student 自监督方法的重要结构母题。

从今天看，MoCo v1 的绝对精度与训练效率已不是最优；但“解耦字典规模、稳定非平稳目标、警惕代理任务作弊”仍是理解现代自监督学习的基础原则。
