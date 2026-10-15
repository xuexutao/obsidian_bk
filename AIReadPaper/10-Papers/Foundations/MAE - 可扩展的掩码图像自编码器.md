---
type: paper-note
status: complete
domain: Foundations
paper: Masked Autoencoders Are Scalable Vision Learners
year: 2022
arxiv: 2111.06377
doi: 10.1109/CVPR52688.2022.01553
source: https://arxiv.org/abs/2111.06377
project: null
code: https://github.com/facebookresearch/mae
tags:
  - Foundations
  - self-supervised-learning
  - masked-image-modeling
  - vision-transformer
  - representation-learning
created: 2026-10-15
updated: 2026-10-15
---

# MAE：可扩展的掩码图像自编码器

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | Masked Autoencoders Are Scalable Vision Learners |
| 作者 | Kaiming He、Xinlei Chen、Saining Xie、Yanghao Li、Piotr Dollár、Ross Girshick |
| 机构 | Facebook AI Research（FAIR） |
| 发表 | CVPR 2022，pp. 16000–16009 |
| arXiv | 2111.06377（v1：2021-11-11；v3：2021-12-19） |
| 官方论文 | https://arxiv.org/abs/2111.06377 |
| CVPR 页面 | https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html |
| 官方代码 | https://github.com/facebookresearch/mae |
| 本地 PDF | [paper.pdf](../../assets/MAE%20-%20可扩展的掩码图像自编码器/sources/paper.pdf) |
| 来源记录 | [sources.md](../../assets/MAE%20-%20可扩展的掩码图像自编码器/sources.md) |

## 0. 一句话结论

MAE 通过“**随机遮掉约 75% 图像 patch，只让重型 ViT encoder 看剩余 25%，再用轻量 decoder 重建被遮像素**”，同时把视觉掩码建模做成了足够困难的语义任务和足够便宜的大模型预训练；它证明了简单的像素重建无需离散 tokenizer，也能随 ViT-B/L/H 稳定扩展，并在完整微调与多类迁移任务上明显优于从头监督训练。

> 最值得记住的不是“自编码器复兴”，而是一个系统设计原则：**把昂贵计算放在稀疏、真实的可见 token 上，把只服务预训练目标的密集计算推迟到便宜且可丢弃的模块里。**

---

## 1. 研究背景与核心问题

### 1.1 为什么把 BERT 式掩码建模搬到视觉并不直接

NLP 的 BERT 已经证明：删去一部分输入，再预测缺失内容，可以在没有人工标签的情况下学到可迁移表征。

视觉中也早有 denoising autoencoder、Context Encoder 和像素补全，但它们长期没有形成与 BERT 相当的可扩展预训练范式。

论文把原因归纳为三点。

第一，过去视觉主干以 CNN 为主，规则网格卷积不容易像 Transformer 那样自然插入 mask token 和位置编码；ViT 的出现消除了这个架构障碍。

第二，图像和语言的信息密度不同。

语言 token 是人类压缩后的高语义符号，遮掉少量词就可能要求模型理解句法和语义。

图像 patch 有很强的局部空间冗余，若只遮 15% 或 20%，模型容易从邻域纹理、边缘延拓中恢复像素，不必理解物体和场景。

第三，解码目标的语义层级不同。

BERT 预测词，目标天然是语义单位；视觉自编码器预测像素，目标更靠近低层信号。

因此视觉 decoder 的容量与位置会决定 encoder 是否被迫保留抽象语义，还是把大量能力浪费在低层重建上。

### 1.2 MAE 要回答的三个问题

1. 能否用最简单的像素回归，而不是预训练离散 tokenizer，得到强视觉表征？
2. 能否把掩码比例提高到足以打破局部冗余，同时不让训练成本爆炸？
3. 能否让自监督预训练随 ViT-B、ViT-L、ViT-H 扩展，并改善完整微调和下游迁移？

### 1.3 核心贡献

1. 提出非对称 encoder-decoder：encoder 只处理可见 patch，mask token 直到 decoder 才加入。
2. 发现约 75% 的高随机掩码率同时优化表征质量与计算效率。
3. 证明按 patch 归一化的原始像素是足够强、且比 dVAE token 更简单的重建目标。
4. 在 ImageNet-1K 上把 vanilla ViT-H/14 扩展到 86.9% top-1，448 分辨率达到 87.8%。
5. 在 COCO、ADE20K、iNaturalist、Places 与 ImageNet 鲁棒性数据上验证迁移和 scaling。

---

## 2. 问题定义与训练接口

### 2.1 输入 patch 序列

给定图像 $x\in\mathbb{R}^{H\times W\times 3}$，按 $P\times P$ 的不重叠窗口切成 $N=HW/P^2$ 个 patch。

在标准 224×224、patch size 16 的 ViT-B/L 中，$N=14\times14=196$。

把第 $i$ 个 patch 展平并线性投影为 token：

$$e_i = W_p\operatorname{vec}(x_i)+p_i,$$

其中 $W_p$ 是 patch embedding，$p_i$ 是固定的 sine-cosine 位置编码。

### 2.2 随机掩码集合

从 $N$ 个位置中无放回、均匀采样可见集合 $V$，其补集 $M$ 为被遮集合。

掩码率定义为：

$$r=\frac{|M|}{N}.$$

默认 $r=0.75$，因此 encoder 只接收 $|V|=0.25N$ 个真实 patch token。

对于 196 个 patch，这意味着只编码 49 个可见 patch；论文 Figure 2 的可视化使用 80% 掩码，只剩 39/196 个 patch。

### 2.3 预训练目标

encoder 输出可见 token 的潜表示：

$$z_V=f_{\theta}(e_V).$$

把 $z_V$ 投影到 decoder 维度，与共享、可学习的 mask token 拼接，再用逆置换恢复原空间顺序并加 decoder 位置编码：

$$\hat{x}=g_{\phi}(\operatorname{unshuffle}(z_V,m_M)+p^{dec}).$$

损失只在被遮 patch 上计算：

$$\mathcal{L}_{MAE}=\frac{1}{|M|}\sum_{i\in M}\left\|\hat{y}_i-y_i\right\|_2^2.$$

若使用归一化像素目标，对每个 patch 单独计算均值 $\mu_i$ 与标准差 $\sigma_i$：

$$y_i=\frac{\operatorname{vec}(x_i)-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}.$$

这不是把整幅图做统一标准化，而是对每个 patch 做局部对比度归一化。

论文实验显示，归一化像素将 ViT-L 消融设置下的完整微调从 84.9% 提升到 85.4%，线性探测从 73.5% 提升到 73.9%。

### 2.4 预训练到下游任务的接口

预训练结束后，整个 decoder 被丢弃。

下游输入不再掩码，encoder 接收完整 patch 序列。

分类任务可以读取 class token；论文也验证了去掉 class token、改用平均池化时表现相近。

线性探测冻结全部 encoder，只训练 BatchNorm（无 affine）加线性分类器。

完整微调更新全部 encoder 和分类头，并使用 layer-wise learning-rate decay。

检测与分割则把 ViT 中间层适配为 FPN 多尺度特征，再端到端更新 backbone 与任务头。

这个接口很干净：**decoder、mask token 与重建头都是预训练脚手架，不进入部署模型。**

---

## 3. 方法详解

### 3.1 非对称 encoder-decoder 全景

![](assets/MAE%20-%20可扩展的掩码图像自编码器/figures/fig1-asymmetric-architecture.png)

> **Figure 1，PDF 第 1 页。** 输入中大部分 patch 被直接移除；重型 encoder 只处理可见 patch。编码后才补入 mask token，恢复完整空间顺序，并交给较小 decoder 预测被遮像素。预训练后 decoder 被丢弃。官方链接：<https://arxiv.org/pdf/2111.06377#page=1>

图中真正关键的是 mask token 出现的位置。

若像 BERT/BEiT 那样把 mask token 一开始就放入 encoder，encoder 仍要处理长度为 $N$ 的完整序列。

MAE 把 mask token 推迟到 decoder，使 encoder 序列长度从 $N$ 降为 $(1-r)N$。

这既减少 FLOPs，也减少激活内存，并避免 encoder 在预训练时长期接触下游不会出现的人工 token。

### 3.2 Random masking：均匀、无放回、每次重采样

MAE 的 masking 没有显式对象先验，也不生成规则大块。

实现上先为全部 patch 生成 token，然后为每张图采样随机噪声并排序，相当于随机 shuffle。

保留 shuffle 后前 $(1-r)N$ 个 token 给 encoder，其余 token 被删除。

同时保存 `ids_restore`，decoder 阶段用它把可见表示与 mask token unshuffle 回原位置。

这种做法有三个优点。

1. 均匀采样避免中心区域被系统性遮挡或保留的偏差。
2. 无放回保证可见 token 数量固定，batch 内张量形状规则。
3. 每次迭代重采样 mask，本身就形成强数据扰动，因此预训练不依赖复杂颜色增强。

它也不需要稀疏卷积、稀疏 attention 或特殊 kernel；删除 token 后直接运行普通 dense Transformer 即可。

### 3.3 为什么是 75% 高掩码率

75% 不是为了模仿 BERT，而是针对图像冗余重新设定任务难度。

低掩码率留下大量相邻 patch，模型可以靠局部纹理插值完成任务。

高掩码率打散可用证据，迫使模型整合远距离轮廓、物体部件和场景布局。

但掩码率也不能无限提高。

当可见证据少到不足以约束语义时，重建会变成多解的条件生成，预训练信号噪声上升。

Figure 5 显示线性探测在 75% 左右达到峰值，90% 时明显下降；完整微调在 40%–80% 区间更平坦，但 90% 也退化。

因此 75% 同时位于两个甜点区：任务足够难，encoder 计算又足够稀疏。

### 3.4 只编码可见 token

MAE encoder 是标准 ViT，不需要专用架构。

每个可见 patch 经过线性投影、位置编码和一系列 Transformer block。

被遮 patch 完全不进入 encoder，也没有 `[MASK]` 占位符。

这一选择具有表征和效率两方面收益。

表征方面，encoder 在预训练和部署时都只看到真实图像 patch，减小了输入分布错位。

效率方面，75% 掩码使 token 数变为 1/4。

self-attention 的 token-token 项近似按序列长度平方缩放，所以该部分理论上可降至约 $1/16$；MLP 与投影部分约降至 $1/4$。

考虑轻量 decoder、patch embedding 与其他开销后，论文实测总训练 FLOPs 相对“encoder 含 mask token”降低 3.3 倍。

### 3.5 轻量 decoder 的角色

decoder 不是下游表征网络，而是一个只在预训练中存在的重建器。

默认 decoder 深度为 8 个 Transformer blocks、宽度 512。

对应的 ViT-L encoder 是 24 blocks、宽度 1024。

论文指出默认 decoder 每 token FLOPs 约为 ViT-L encoder 的 9%。

因此即使 decoder 要处理完整 196-token 序列，它仍只占总成本的一小部分。

decoder 至少需要一个 attention block，因为可见 token 的信息必须传播到 mask token；纯逐 token MLP 无法完成跨位置重建。

更深 decoder 对线性探测很重要，因为它能把像素重建的专门化压力吸收在 decoder 中，让 encoder 顶层保留更抽象的表示。

但完整微调能重新塑造 encoder 顶层，所以 decoder 深度对 fine-tuning 几乎不敏感。

这解释了为什么 1-block decoder 仍能得到 84.8% fine-tuning，同时将 ViT-L 训练加速到 3.7 倍。

### 3.6 像素归一化重建损失

MAE 最终主结果使用每 patch 归一化像素作为 target。

局部归一化去掉 patch 的整体亮度与对比度，把误差权重更多放到内部结构和高频变化上。

这不等于“像素目标天然语义化”；它只是避免模型通过预测低频统计获得过多收益。

PCA target 只保留 96 个主成分，反而降低表现，说明被丢弃的高频成分对表示学习有用。

dVAE token 把连续像素变成离散类别并用交叉熵预测，微调结果与归一化像素基本相当，却引入额外 tokenizer 预训练与推理开销。

因此 MAE 的结论不是任何像素 MSE 都足够，而是：**在高掩码率、非对称架构和合适 decoder 下，局部归一化像素已经是强 target。**

### 3.7 为什么只在 masked patches 上算 loss

若同时监督可见 patch，decoder 很容易学习复制已观察输入。

这会让大量 loss 被简单位置主导，稀释真正的条件推断任务。

论文脚注报告，在全部 patch 上计算损失会让准确率下降约 0.5%。

只监督 masked patches 也使目标与 BERT 式 masked modeling 一致：模型只为缺失信息负责。

需要注意，论文可视化没有把原可见 patch 覆盖回输出，所以可见位置的生成质量可能较差；这不是重建能力失败，而是这些位置从未进入损失。

### 3.8 重建结果如何理解

![](assets/MAE%20-%20可扩展的掩码图像自编码器/figures/fig2-imagenet-reconstruction-80-mask.png)

> **Figure 2，PDF 第 2 页。** 每组三幅依次是 masked input、MAE reconstruction、ground truth；这里掩码率为 80%，只留下 39/196 个 patch。时钟、动物、车辆、瀑布和建筑的整体结构被合理恢复，但细节不必与真值一致。官方链接：<https://arxiv.org/pdf/2111.06377#page=2>

重建图证明模型利用了全局上下文，但不能单独证明表征优良。

在高度不确定的区域，模糊输出可能只是 MSE 下的条件均值。

更有意义的证据是：同一 encoder 在 ImageNet 完整微调、COCO 检测分割、ADE20K 语义分割和细粒度分类上都获得增益。

Figure 4 还显示，模型用 75% 掩码训练，却能在 85% 和 95% 掩码输入上生成语义合理但与原图不同的内容。

这说明模型学到的是数据分布与场景先验，而不是逐 patch 的确定性恢复规则。

### 3.9 计算收益的定量结果

Table 2 在 128 个 TPU-v3 cores、800 epochs、decoder width 512、mask ratio 75% 下比较 wall-clock。

| Encoder | Decoder depth | Fine-tune | 时间 | 相对加速 |
|---|---:|---:|---:|---:|
| ViT-L，encoder 含 mask token | 8 | 84.2 | 42.4 h | 1× |
| ViT-L，仅可见 token | 8 | 84.9 | 15.4 h | 2.8× |
| ViT-L，仅可见 token | 1 | 84.8 | 11.6 h | 3.7× |
| ViT-H，encoder 含 mask token | 8 | — | 119.6 h（估算） | 1× |
| ViT-H，仅可见 token | 8 | 85.8 | 34.5 h | 3.5× |
| ViT-H，仅可见 token | 1 | 85.9 | 29.3 h | 4.1× |

这组结果说明收益不是只来自“少算一点 patch”。

更大的 encoder 越昂贵，把密集 token 推迟到小 decoder 的收益越明显。

论文还比较 ViT-L 的完整预训练：MAE 1600 epochs 在同硬件约 31 小时，MoCo v3 300 epochs 约 36 小时。

单看 epoch 数会误判成本，因为 MAE encoder 每 epoch 只看 25% patch，而两视图对比学习至少看 200%。

---

## 4. 实验设计与复现设置

### 4.1 ImageNet 评测协议

自监督预训练只使用 ImageNet-1K 训练图像，不使用标签。

消融以 ViT-L/16 为主，默认预训练 800 epochs。

主结果改为归一化像素 target，并预训练 1600 epochs。

分类报告单个 224×224 crop 的 top-1 validation accuracy；ViT-H 另报告 448 分辨率微调。

评估包含线性探测、部分微调和端到端完整微调。

### 4.2 默认预训练配置

| 配置 | 值 |
|---|---|
| optimizer | AdamW |
| base learning rate | 1.5e-4 |
| weight decay | 0.05 |
| betas | 0.9, 0.95 |
| batch size | 4096 |
| schedule | cosine decay |
| warmup | 40 epochs |
| augmentation | RandomResizedCrop |
| mask ratio | 75% |
| decoder | depth 8，width 512 |

学习率按 `base_lr × batch_size / 256` 线性缩放。

预训练不使用 color jitter、drop path 或 gradient clipping。

### 4.3 完整微调配置

完整微调使用 AdamW、base LR 1e-3、weight decay 0.05、batch size 1024。

使用 layer-wise LR decay 0.75、5 epochs warmup 和 cosine schedule。

ViT-B 微调 100 epochs，ViT-L/H 微调 50 epochs。

监督增强包含 RandAug、label smoothing、mixup、cutmix 和 drop path。

因此“MAE 预训练不依赖强增强”不代表下游微调也完全不用强增强。

### 4.4 线性探测配置

encoder 完全冻结，优化器为 LARS，base LR 0.1，weight decay 0，batch size 16384，训练 90 epochs。

分类器前加入 `affine=False` 的 BatchNorm 校准特征尺度。

该 BatchNorm 可吸收到最终线性变换，不引入非线性决策边界。

论文特别指出，mixup、cutmix、drop path、color jitter 等正则对 linear probe 有害，因此全部关闭。

### 4.5 迁移协议

COCO 使用 ViT Mask R-CNN，按 ViT 深度四等分抽取中间特征，并通过卷积形成 stride 4/8/16/32 的 FPN。

ADE20K 使用 UperNet，端到端微调 100 epochs，batch size 16。

iNaturalist 与 Places 沿用 ImageNet 微调配方，并为每个数据集调学习率与 epoch。

迁移结果因任务头、搜索空间与实现细节较多，更适合看同表内相对比较，不宜与不同代码库的数字直接横比。

---

## 5. 主要实验结果与消融

### 5.1 75% mask ratio 消融

Figure 5 的 ViT-L/16 结果显示，fine-tuning 对 40%–80% 掩码较稳健，而 linear probe 对掩码率非常敏感。

线性探测从 10% 掩码时的 54.6% 持续上升，到 75% 时达到 73.5%，90% 时回落到 66.1%。

完整微调从 10% 时的 83.2% 上升到约 60% 时的 85.0%，75% 仍为 84.9%。

这表明高掩码率首先改善的是冻结表征的抽象程度，而完整微调能补偿较弱预训练表征。

75% 并非唯一可用点，但它兼顾了接近峰值的准确率与只编码 25% token 的效率。

### 5.2 Decoder depth 与 width

| Decoder depth | Fine-tune | Linear probe |
|---:|---:|---:|
| 1 | 84.8 | 65.5 |
| 2 | 84.9 | 70.0 |
| 4 | 84.9 | 71.9 |
| 8 | 84.9 | 73.5 |
| 12 | 84.4 | 73.3 |

完整微调几乎不需要深 decoder；线性探测却从 1 block 到 8 blocks 提升 8.0 points。

这支持“decoder 吸收像素重建专门化”的解释。

12 blocks 没有继续提高，说明容量并非越大越好。

| Decoder width | Fine-tune | Linear probe |
|---:|---:|---:|
| 128 | 84.9 | 69.1 |
| 256 | 84.8 | 71.3 |
| 512 | 84.9 | 73.5 |
| 768 | 84.4 | 73.1 |
| 1024 | 84.3 | 73.1 |

512 维是精度与成本的良好平衡。

与 1024 维 ViT-L encoder 同宽并无收益，进一步证明 decoder 不需要复制 encoder 的规模。

### 5.3 Mask token 放置位置

当 encoder 处理 mask token 时，fine-tuning 为 84.2%，linear probe 只有 59.6%，FLOPs 为 3.3×。

去掉 encoder 中的 mask token 后，fine-tuning 升至 84.9%，linear probe 升至 73.5%，FLOPs 记为 1×。

这是全文最强的结构消融之一：同一个设计同时改善准确率、计算与预训练—部署分布一致性。

14 points 的 linear-probe 差距说明冻结表示特别容易受到人工 mask token 分布污染。

### 5.4 Reconstruction target 消融

| Target | Fine-tune | Linear probe |
|---|---:|---:|
| raw pixel | 84.9 | 73.5 |
| normalized pixel | **85.4** | **73.9** |
| PCA（96 维） | 84.6 | 72.3 |
| dVAE token | 85.3 | 71.6 |

归一化像素在两种协议下都最好。

dVAE token 在 fine-tuning 上接近，但 linear probe 更差，而且需要额外 tokenizer。

PCA 降低表现，说明仅保留大方差低频方向并不理想。

Table 7 在主训练设置和迁移任务上再次比较 target：normalized pixel 与 dVAE token 的差异处于 0.0 到 -0.2 point，统计上可忽略。

因此没有证据支持为 MAE 引入昂贵离散 tokenizer。

### 5.5 Mask sampling 策略

| 策略 | Ratio | Fine-tune | Linear probe |
|---|---:|---:|---:|
| random | 75% | **84.9** | **73.5** |
| block | 50% | 83.9 | 72.3 |
| block | 75% | 82.8 | 63.9 |
| grid | 75% | 84.0 | 66.0 |

block masking 在 75% 时太难，训练 loss 更高，重建更模糊。

grid masking 规则保留四分之一 patch，任务过于容易，重建更清晰但表征更差。

这揭示了一个反直觉点：**更漂亮的重建不等于更好的表示。**

random masking 在任务难度、证据覆盖与效率之间取得最好平衡。

### 5.6 数据增强消融

无增强时 fine-tuning/linear probe 为 84.0/65.7。

固定尺寸 crop 为 84.7/73.1，随机尺寸 crop 为 84.9/73.5。

再加 color jitter 反而降到 84.3/71.9。

MAE 的随机 mask 已经为同一图像产生大量不同观测，因此不需要对比学习那样依赖强视图增强来避免平凡解。

不过完全无增强仍损失较多 linear-probe 准确率，不能简化为“augmentation 没用”。

### 5.7 训练时长

ViT-L 从 100、200、400、800 到 1600 epochs，fine-tuning 依次为 82.3、83.3、84.3、84.9、85.1。

linear probe 依次为 57.3、64.4、69.7、73.5、75.1。

到 1600 epochs，线性探测仍未明显饱和。

但 MAE 每 epoch 的 encoder 只消费 25% patch，不能把 1600 epochs 与完整图或多 crop 方法的 1600 epochs 等价。

### 5.8 ImageNet 完整微调主结果

| 方法 | 预训练数据 | ViT-B | ViT-L | ViT-H | ViT-H 448 |
|---|---|---:|---:|---:|---:|
| scratch（论文实现） | — | 82.3 | 82.6 | 83.1 | — |
| DINO | IN1K | 82.8 | — | — | — |
| MoCo v3 | IN1K | 83.2 | 84.1 | — | — |
| BEiT | IN1K + DALLE tokenizer | 83.2 | 85.2 | — | — |
| MAE | IN1K | **83.6** | **85.9** | **86.9** | **87.8** |

ViT-B 上各方法差距较小，ViT-L 后 MAE 优势扩大。

这支持论文的核心论断：MAE 的价值主要体现在可扩展性，而不是只在小模型上做局部改进。

ViT-H 448 的 87.8% 使用更高微调分辨率，不能与 224 结果完全等同，但当时在只用 IN1K 数据的方法中超过此前 87.1%。

### 5.9 模型 scaling

![](assets/MAE%20-%20可扩展的掩码图像自编码器/figures/fig8-scaling-main-result.png)

> **Figure 8，PDF 第 7 页。** 蓝线显示 MAE 从 ViT-B/16 到 ViT-L/16、ViT-H/14 持续增益；同规模的 IN1K 监督训练很快饱和。MAE 只用 IN1K 无标签数据，却呈现接近 JFT-300M 监督预训练的随规模增长趋势。官方链接：<https://arxiv.org/pdf/2111.06377#page=7>

图中 supervised-IN1K 曲线从 ViT-B 到 ViT-H 只缓慢增加，而 MAE 曲线持续上升。

原因不是 MAE 消除了过拟合，而是自监督重建为每张图提供密集、不断变化的训练信号。

大模型拥有更高容量时，能够从同一百万张图像的不同 mask 中吸收更多结构信息。

需要谨慎的是，Figure 8 是三个离散模型规模点，不足以建立严格幂律 scaling law。

### 5.10 线性探测与部分微调

附录 Table 12 的 1600-epoch 模型在线性探测上得到：ViT-B 68.0%、ViT-L 75.8%、ViT-H 76.6%。

ViT-L 的 MoCo v3 为 77.6%，说明 MAE 冻结特征的线性可分性仍略弱于对比学习。

但 Figure 9 中，只微调 ViT-L 最后 1 个 block，MAE 就从 73.5% 跳到 81.0%。

只微调最后 block 的 MLP 子层也能达到 79.1%。

微调 4 或 6 个 block 已接近完整微调；只要更新至少一个 block，MAE 都优于 MoCo v3，微调 4 blocks 时领先 2.6 points。

这说明 MAE 学到的不是坏特征，而是**强但非线性可读出**的特征。

### 5.11 为什么 MAE 的 linear probe 不如 fine-tune

第一，像素重建要求 encoder 保留局部颜色、纹理和空间细节，最终语义可能分布在多个层和 token 中，而非集中成线性可分的 class token。

第二，预训练目标没有直接拉近同类图像、推远异类图像；DINO/MoCo 一类判别式目标更直接塑造全局类别几何。

第三，decoder 虽然吸收了部分重建专门化，但 encoder 顶层仍服务于“给 decoder 提供可重建信息”，不完全等价于分类表征。

第四，完整微调可以重新组织 attention、MLP 与 class token，把分布式视觉因素压缩到目标标签边界；linear probe 没有这种适配能力。

第五，部分微调实验给出直接证据：只开放一个 block 就增加 7.5 points，瓶颈主要在非线性读出接口，而不是底层信息缺失。

因此不应仅用 linear probe 排名判断预训练质量，尤其是要迁移到检测、分割等需要空间信息的任务时。

### 5.12 COCO 检测与实例分割

| 预训练 | ViT-B box AP | ViT-L box AP | ViT-B mask AP | ViT-L mask AP |
|---|---:|---:|---:|---:|
| supervised IN1K | 47.9 | 49.3 | 42.9 | 43.9 |
| MoCo v3 | 47.9 | 49.3 | 42.7 | 44.0 |
| BEiT | 49.8 | 53.3 | 44.4 | 47.1 |
| MAE | **50.3** | **53.3** | **44.9** | **47.2** |

MAE 对 supervised ViT-B 的 box AP 提升 2.4 points，对 ViT-L 提升 4.0 points。

模型越大，预训练优势越明显，与 ImageNet scaling 结论一致。

MAE 与 BEiT 在 ViT-L box AP 持平，mask AP 略高，同时不需要外部 tokenizer。

### 5.13 ADE20K 语义分割

| 预训练 | ViT-B mIoU | ViT-L mIoU |
|---|---:|---:|
| supervised IN1K | 47.4 | 49.9 |
| MoCo v3 | 47.3 | 49.1 |
| BEiT | 47.1 | 53.3 |
| MAE | **48.1** | **53.6** |

ViT-L 上 MAE 比 supervised 高 3.7 points，也略高于 BEiT。

分割需要密集空间信息，这类结果也解释了为何线性分类分数不能完整代表 MAE 表征质量。

### 5.14 iNaturalist 与 Places 分类迁移

| 数据集 | ViT-B | ViT-L | ViT-H | ViT-H 448 | 当时 previous best |
|---|---:|---:|---:|---:|---:|
| iNat 2017 | 70.5 | 75.7 | 79.3 | **83.4** | 75.4 |
| iNat 2018 | 75.4 | 80.1 | 83.0 | **86.8** | 81.2 |
| iNat 2019 | 80.5 | 83.4 | 85.7 | **88.3** | 84.1 |
| Places205 | 63.9 | 65.8 | 65.9 | **66.8** | 66.0 |
| Places365 | 57.9 | 59.4 | 59.8 | **60.3** | 58.0 |

细粒度 iNaturalist 上的 scaling 最明显，ViT-H 448 对当时最佳结果有大幅提升。

Places 的提升较温和，ViT-L 到 ViT-H 接近饱和，说明模型 scaling 的收益依赖下游任务。

论文强调此前 Places 最佳系统使用十亿级图像预训练，而 MAE 只用 IN1K；但这是系统级比较，不是完全控制变量实验。

### 5.15 ImageNet 分布偏移与鲁棒性

同一个在原始 ImageNet 微调的模型直接评估，不做专门适配。

ViT-H 的 IN-C mean corruption error 为 33.8，低于 supervised ViT-H 的 41.3。

ViT-H 的 IN-A accuracy 为 68.2，而 supervised counterpart 为 33.1，提升约 35 points。

ViT-H 448 在 IN-A、IN-R、IN-Sketch 分别达到 76.7、66.5、50.9。

但提高到 448 分辨率时 IN-C 从 33.8 变为 36.8，说明更高输入分辨率并非对所有扰动都有利。

---

## 6. 与 BEiT、iBOT、DINO 的差异

### 6.1 总览

| 方法 | 核心目标 | Target 来源 | Mask token / 输入 | 主要归纳偏置 | 下游特点 |
|---|---|---|---|---|---|
| MAE | 重建 masked patch 像素 | 原图按 patch 归一化像素 | encoder 只看可见 token；mask token 仅进 decoder | 生成式、保留局部与空间细节 | 完整微调和密集迁移强，线性可分性相对弱 |
| BEiT | 预测 masked patch 的离散视觉 token | 外部 dVAE tokenizer | encoder 处理含 mask token 的完整序列 | BERT 式离散分类 | target 更语义化，但依赖额外 tokenizer 与数据 |
| iBOT | online tokenizer + patch/class 自蒸馏 | EMA teacher 在线产生 token/分布 | masked student 对齐 teacher | 生成式 masking 与判别式 self-distillation 结合 | 兼顾 patch-level 与 class-level 语义 |
| DINO | 多 crop teacher-student 自蒸馏 | EMA teacher soft targets | 原版不以重建 masked pixels 为核心 | 视图不变性与全局语义聚类 | class token 线性可分性和注意力语义涌现强 |

### 6.2 MAE vs. BEiT

BEiT 先用 DALL-E dVAE 把 patch 映射成离散 token，再让 ViT 预测被遮 token 类别。

MAE 直接回归像素，不需要 tokenizer 的额外预训练阶段。

论文指出 BEiT 的 tokenizer 曾用 2.5 亿张 DALLE 数据，dVAE encoder FLOPs 约为 ViT-L 的 40%。

MAE 的 encoder 删除 masked token，因此每 epoch 比 BEiT 式完整序列编码显著更快。

在 Table 3 中，MAE ViT-L 为 85.9%，BEiT 为 85.2%；但 BEiT 的预训练目标和实现细节不同，不能把 0.7 point 全部归因于像素 target。

更直接的 MAE 内部消融显示：normalized pixels 与 dVAE tokens 在迁移上几乎相同，说明 tokenization 不是强性能的必要条件。

### 6.3 MAE vs. iBOT

二者都使用 masked image modeling，但 iBOT 的 target 由 EMA teacher 在线生成，而不是固定像素或离线 dVAE token。

iBOT 同时对 patch token 和 class token 做 self-distillation，显式鼓励全局语义一致性。

MAE 不需要 teacher 分支、centering、sharpening 或 teacher momentum，算法和内存路径更简单。

代价是 MAE 的 class token 没被直接训练成判别式全局表示，所以 linear probe 往往不如带 class-level 自蒸馏的方法。

MAE 原论文没有提供与 iBOT 的同协议直接表格比较，因此这里比较的是训练机制，不把不同论文数字混为一谈。

### 6.4 MAE vs. DINO

DINO 通过不同 crop 的 student/teacher 输出一致性学习视图不变表征，原版没有像素重建 decoder。

它依赖多 crop、强数据增强、EMA teacher 和防坍塌机制。

MAE 依靠随机高比例 masking 自然生成训练视图，不需要负样本，也不依赖复杂颜色增强。

DINO 的目标直接塑造全局语义聚类，因此 class token 通常更适合 linear probe；MAE 则保留更多局部与空间信息，对端到端微调和密集预测更友好。

Table 3 只给出 DINO ViT-B 完整微调 82.8%，MAE ViT-B 为 83.6%，没有大模型同表数据，不能据此断言 MAE 在所有设置全面优于 DINO。

### 6.5 本质差别：target、计算放置与评测偏好

BEiT/iBOT/DINO 更关心“模型应对哪个语义分布进行分类或蒸馏”。

MAE 更关心“如何用最少 encoder 计算，从部分观测中保留足以恢复输入的隐变量”。

MAE 的创新不只在 target，而在**计算图重排**：删除 encoder 中的 masked token，把密集重建留给便宜 decoder。

因此即便未来更换 target，visible-only encoder 仍是可独立迁移的系统思想。

---

## 7. 局限性与批判性思考

### 7.1 像素 MSE 与感知质量并不一致

MSE 倾向输出条件均值，在多解区域产生模糊结果。

论文的目标是表征学习而不是高保真生成，所以这不妨碍下游性能，却限制了 decoder 直接作为生成模型的价值。

Figure 2 的语义合理重建应被视作定性证据，不能当作生成质量评测。

### 7.2 75% 不是跨域通用常数

最佳 mask ratio 与 patch size、图像分辨率、数据纹理、模型容量和任务有关。

遥感、医学、文档或视频的冗余结构不同，直接复制 75% 可能过难或过易。

更合理的迁移方式是把“打破局部捷径，同时保留足够全局证据”作为原则重新消融。

### 7.3 长 epoch 仍意味着高总成本

visible-only encoder 大幅降低每步成本，但主结果预训练 1600 epochs，batch size 4096，仍需要大规模硬件。

“31 小时”基于 128 个 TPU-v3 cores，不代表普通实验室可以低成本复现 ViT-H。

效率结论应区分每样本 FLOPs、wall-clock、硬件并行规模和总能源成本。

### 7.4 主结果依赖完整微调

MAE 的最佳叙事建立在 end-to-end fine-tuning 上。

若应用场景要求冻结 backbone、只训练廉价线性头，DINO/MoCo/iBOT 式判别表征可能更合适。

这不是评测漏洞，但明确限定了“强表征”的含义：MAE 更像可塑性强的初始化，而不是立即线性可读的特征库。

### 7.5 对下游超参数的敏感性

论文对 COCO 等任务为各方法搜索 learning rate、weight decay、drop path 与 epochs，这提高了公平性，也增加了复现成本。

官方开源仓库主要覆盖 ImageNet 预训练、微调与线性探测，并不完整包含论文所有检测、分割配置。

跨代码库复现迁移表时需要特别谨慎。

### 7.6 外部比较并非全部控制变量

BEiT 使用额外 tokenizer 数据，DINO/MoCo 的增强、训练长度与目标不同。

Table 3 是系统比较，不是逐因素因果实验。

MAE 最可信的因果证据来自自身 Table 1–2 的架构、target 和 mask 消融。

### 7.7 数据偏差与幻觉式补全

模型依据 ImageNet 统计补全缺失内容，会继承训练数据偏差。

在高掩码下，它可能生成语义合理但实际上不存在的对象或细节。

若把这类重建用于医学、取证或科学测量，不能把“plausible”误当作“true”。

---

## 8. 复现资源与工程要点

### 8.1 官方仓库范围

官方仓库：<https://github.com/facebookresearch/mae>

仓库是 PyTorch/GPU 重实现，论文原始结果来自 TensorFlow+TPU。

主要入口包括：

- `main_pretrain.py`：自监督预训练；
- `main_finetune.py`：ImageNet 端到端微调；
- `main_linprobe.py`：冻结 encoder 的线性探测；
- `models_mae.py`：patchify、random masking、encoder、decoder 与 masked MSE；
- `models_vit.py`：下游 ViT；
- `demo/mae_visualize.ipynb`：重建可视化；
- `PRETRAIN.md` 与 `FINETUNE.md`：命令和配置。

仓库公开 ViT-B、ViT-L、ViT-H 预训练 checkpoint，并给出短 MD5 前缀 `8cad7c`、`b8b06e`、`9bdbb0`。

### 8.2 环境注意

官方 README 基于较老的 `timm==0.3.2`，并提示 PyTorch 1.8.1+ 需要兼容修补。

在现代环境中直接复现时，优先固定仓库 commit 和依赖，而不是盲目升级 timm API。

论文与仓库精度可能因 TensorFlow/TPU 到 PyTorch/GPU、数据管线、随机种子和增强实现略有差异。

### 8.3 复现优先级

若算力有限，建议先复现 ViT-B 而不是 ViT-L/H。

第一阶段只验证 random masking、mask ratio 75%、normalized pixel loss 和 1/8-block decoder 的趋势。

第二阶段分别跑 linear probe 与 full fine-tune，避免只用一个协议判断结果。

第三阶段记录实际 images/s、峰值显存和总 GPU-hours，验证 visible-only encoder 的系统收益。

若要复现 target 消融，必须保证训练长度、decoder、微调配方和随机种子一致。

### 8.4 最小实现检查清单

1. `patchify` 与 `unpatchify` 的空间顺序可逆。
2. random masking 对每个样本独立生成 shuffle，并保存正确 `ids_restore`。
3. encoder 输入中没有 masked token。
4. decoder 前补足 mask token，再按 `ids_restore` 恢复顺序。
5. encoder 与 decoder 都加入位置编码。
6. decoder prediction 维度等于 `patch_size² × channels`。
7. normalized pixel target 按 patch 计算 mean/variance。
8. loss 只平均 masked positions，不能被可见 patch 稀释。
9. fine-tuning 时删除 decoder，并输入未损坏的完整图像。
10. linear probe 的 encoder 参数真正冻结，BatchNorm 设置与论文一致。

---

## 9. 关键理解与延伸

### 9.1 MAE 的真正遗产

MAE 把 masked image modeling 从“需要语义 tokenizer 的 BERT 模仿”改写为“高掩码率下的稀疏编码问题”。

它证明 target 可以朴素，关键是任务难度、计算放置和下游接口共同设计。

后续视频 MAE、多模态 MAE、3D 点云 masking 和时空预测都继承了“只编码可见内容”的思路。

### 9.2 生成式目标与判别式目标不是简单二选一

MAE 的密集像素目标保留空间细节，DINO/iBOT 的自蒸馏更强调语义不变性。

二者差异解释了 linear probe 与 dense transfer 的不同优势。

后续方法可以联合重建与 self-distillation，让 class token 和 patch token 承担不同层级目标。

### 9.3 对大模型训练系统的普遍启发

当输入中存在大量可删除、可延迟或只服务辅助目标的 token 时，应尽量让昂贵 backbone 只处理信息密集子集。

辅助模块可以在训练时恢复完整结构，并在部署时丢弃。

这与稀疏 MoE、token pruning、latent bottleneck 和 delayed fusion 有共同系统哲学。

### 9.4 评价预训练表征要多协议

linear probe 测量冻结表示的线性可读性。

partial fine-tune 测量少量非线性适配后的可塑性。

full fine-tune 测量初始化对目标优化的帮助。

检测、分割等 dense tasks 测量空间信息保留。

MAE 的结果说明这四者可能不一致，单一排行榜会偏向特定目标。

---

## 10. 术语表

- **MAE**：Masked Autoencoder，通过部分观测重建缺失输入的自监督模型。
- **Masked image modeling**：遮挡图像 token 并预测缺失内容的预训练范式。
- **Mask ratio**：被删除 patch 数占全部 patch 的比例。
- **Visible token**：未被遮挡、实际进入 encoder 的 patch token。
- **Mask token**：表示缺失位置的共享可学习向量；MAE 中只进入 decoder。
- **Asymmetric encoder-decoder**：重型稀疏 encoder 与轻型密集 decoder 的非对称组合。
- **Normalized pixel target**：对每个 patch 独立标准化后的像素回归目标。
- **Linear probing**：冻结 backbone，只训练线性分类器的评测。
- **Partial fine-tuning**：只更新末端若干 block 的中间评测协议。
- **End-to-end fine-tuning**：更新整个 encoder 与任务头。
- **dVAE tokenizer**：把图像 patch 映射为离散视觉 token 的预训练编码器。
- **FPN**：Feature Pyramid Network，为检测分割提供多尺度特征。

---

## 11. 图表索引

1. **Figure 1 / PDF 第 1 页**：MAE 非对称 encoder-decoder，展示 visible-only encoder 与轻量 decoder。<https://arxiv.org/pdf/2111.06377#page=1>
2. **Figure 2 / PDF 第 2 页**：ImageNet 80% 掩码的输入、重建与真值三联图。<https://arxiv.org/pdf/2111.06377#page=2>
3. **Figure 8 / PDF 第 7 页**：MAE 相对 IN1K/JFT 监督训练的模型 scaling。<https://arxiv.org/pdf/2111.06377#page=7>

---

## 12. 最终评价

MAE 的强处不是复杂目标，而是把视觉掩码建模的三个矛盾同时解决：高掩码率让任务摆脱局部插值捷径，visible-only encoder 让高掩码率转化为真实计算节省，轻量 decoder 又把低层重建压力与下游 encoder 解耦。

其证据链也较完整：Table 1–2 证明架构与 target 的因果贡献，Table 3 与 Figure 8 证明 ImageNet scaling，Figure 9 解释 linear probe 与 fine-tune 的分歧，Tables 4–7 证明空间与跨域迁移。

需要保留的判断是：MAE 更适合作为**可塑性强、可扩展、保留空间信息的预训练初始化**，而不是天然最线性可分的冻结特征。

这一区分既解释了它的局限，也解释了它为何成为现代视觉 foundation model 预训练的重要基线。
