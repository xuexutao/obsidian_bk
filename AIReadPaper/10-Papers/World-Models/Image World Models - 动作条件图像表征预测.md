---
type: paper-note
status: complete
domain: World-Models
paper: "Learning and Leveraging World Models in Visual Representation Learning"
year: 2024
arxiv: "2403.00504"
doi: "10.48550/arXiv.2403.00504"
source: https://arxiv.org/abs/2403.00504
project: null
code: null
tags:
  - World-Models
  - JEPA
  - IWM
  - Self-Supervised-Learning
  - Representation-Learning
created: 2026-10-15
updated: 2026-10-15
---

# Image World Models：动作条件图像表征预测

## 基本信息

- **正式标题**：Learning and Leveraging World Models in Visual Representation Learning
- **作者**：Quentin Garrido、Mahmoud Assran、Nicolas Ballas、Adrien Bardes、Laurent Najman、Yann LeCun
- **机构**：FAIR at Meta、Université Gustave Eiffel/CNRS/LIGM、INRIA、New York University
- **版本**：arXiv:2403.00504v1，2024-03-01；23 页、16 图。当前可核验来源把它列为 arXiv 预印本，本笔记不编造会议录用信息。
- **核心对象**：Image World Model（IWM）。它把数据增强参数视作“动作”，让 predictor 在潜空间中预测该动作对图像表征的影响；预训练后不丢弃 predictor，而是把它微调成下游任务头。
- **术语边界**：本文的 action 是已知的 crop/mask、brightness、contrast、saturation、hue、grayscale、blur、solarization 等图像变换参数，不是机器人控制、环境动力学或自然视频中的未知动作。IWM 是“受控图像变换世界模型”，不是能模拟开放世界未来的生成式世界模型。

## 一句话结论

IWM 在 I-JEPA 式 masked latent prediction 上加入显式变换条件，并发现一个可复用 world model 需要三件事：**动作条件、足够难的变换、足够大的 predictor**；当 predictor 真正学会潜空间等变变换后，冻结 encoder、只微调 predictor 就能在 ImageNet 和 ADE20K 上达到或超过 encoder 微调，同时揭示了“高抽象不变表示”与“信息丰富等变表示”之间可由 predictor 能力控制的连续谱。

---

## 1. 研究背景与核心问题

### 1.1 为什么重新审视 SSL 的 predictor

在强化学习中，world model 预测动作后的状态，并常被继续用于规划和控制；在视觉自监督学习中，encoder-predictor 也很常见，但研究通常只保留 encoder，把 predictor/decoder 当作预训练脚手架后丢弃。作者认为这可能浪费了模型已经学到的“如何变换表征”的知识。

论文重新解释几类方法：masked autoencoder 的 decoder 是像素空间世界模型；I-JEPA 的 predictor 是潜空间世界模型；BYOL/SimSiam 虽有 predictor，但若不向它提供变换参数，最优策略倾向于让 encoder 对增强不变，从而无需真正模拟变换。于是核心问题有三个：

1. 如何让潜空间 predictor 确实学会应用 action，而不是退化成普通 augmentation-invariant SSL？
2. 学到的 predictor 能否像 RL world model 一样复用于下游任务，而非预训练后丢弃？
3. predictor 的能力会如何改变 encoder 表征中保留的信息与抽象程度？

![](assets/Image%20World%20Models%20-%20动作条件图像表征预测/figures/02-world-model-taxonomy.png)

> 图 2：论文对生成式 world model、无显式 world model 的 joint embedding，以及 JEPA latent world model 的统一视角。IWM 位于最右侧：在潜空间预测，并显式以变换参数 $a$ 为条件。  
> 来源：论文 Figure 2，第 2 页；arXiv 官方源码 `images/all_methods.pdf`。

### 1.2 贡献概括

1. **把光度增强扩展为 JEPA action**：不再只预测被 mask 的位置，还预测全局颜色与破坏性增强在 target latent 中造成的效果。
2. **总结强 IWM 的三要素**：conditioning 决定模型是否知道要做什么，transformation difficulty 决定 predictor 是否必须学习，capacity 决定它是否有能力完成。
3. **复用 world model**：提出 predictor finetuning；冻结 encoder，只把预训练 predictor 改造成分类/分割头，并扩展到多任务调优。
4. **解释表示抽象度**：弱 predictor 迫使 encoder 丢弃难以预测的变化，形成不变且易线性读取的表示；强 predictor 允许 encoder 保留更多变换信息，形成等变、信息丰富且更适合复杂适配的表示。

---

## 2. 任务定义与输入输出

### 2.1 输入图像、source 与 target

给定单张图像 $I$，先做共同的随机裁剪和水平翻转得到 $I'$。目标视图 $y$ 在 $I'$ 上施加 color jitter，但不施加 grayscale、blur、solarization 等会真正删除信息的变换，以确保 teacher target 尽量完整。source 视图 $x$ 从 $I'$ 出发，额外施加第二次 color jitter、grayscale、blur、solarization，再把四个矩形区域的 patch 丢弃形成 context。

这种非对称设计非常重要：source 比 target 更“损坏”，任务要求模型根据残余 context、待预测位置和变换参数，恢复 target 的 latent。它不是像素颜色重建，而是 teacher encoder 表征回归。

### 2.2 action 的定义

$a_{x\rightarrow y}$ 编码从 source 变回 target 所需的已知变换，包括 source/target 的 color jitter 差异，以及 destructive augmentation 是否发生。mask token 还携带几何位置，因此 action 同时包含“预测哪里”和“如何逆转光度变换”。严格说，crop/flip 的共同部分并不要求 predictor 推断自然运动；环境转移完全由人工增强程序产生并已知。

### 2.3 模型输入输出

student encoder $f_\theta$ 接收 source，输出可见 patch 表征 $z_x=f_\theta(x)$；EMA teacher $f_\theta^{\mathrm{EMA}}$ 接收 target，输出 $z_y$。predictor/world model $p_\phi$ 接收 source latent、action 及带目标位置的 mask tokens，输出被 mask target patch 的预测 $\hat z_y$。训练后可保留两部分：encoder 提供视觉表示；predictor 可继续接受 latent token，并经微调完成分类、分割或多任务预测。

---

## 3. 核心方法

### 3.1 完整 pipeline

![](assets/Image%20World%20Models%20-%20动作条件图像表征预测/figures/01-method-pipeline.png)

> 图 S1：IWM 的完整训练流。masked context 经 student encoder 得到可见 latent；action 参数与待预测位置共同条件化 predictor；EMA teacher 编码完整 target，二者仅在 target patch latent 上计算 $L_2$。红色箭头表示 target 分支停止梯度。  
> 来源：论文 Figure S1，第 13 页；arXiv 官方源码 `images/figure_1_color.pdf`。

训练步骤如下：

1. 对 $I$ 采样共同 crop/flip，分别构造信息较完整的 target $y$ 和进一步损坏的 source $x$。
2. 从 source 中删除四块矩形 patch，只把可见 token 输入 ViT-B/16 student encoder。
3. 完整 target 输入 student 参数的 EMA 副本，产生稳定的 target latent；teacher 不接收反向梯度。
4. 将每个 mask token 加上位置编码，再与 action 参数拼接，通过三层 MLP 混合，形成“位置 + 变换”条件 token。
5. predictor 同时读取 source latent 与条件 token，预测被 mask 位置在 target 中的 latent。
6. 用 target latent 回归更新 student 与 predictor，再以 EMA 更新 teacher。

与像素生成不同，latent target 允许 encoder 忽略不可预测或任务无关的细节；与普通 I-JEPA 不同，predictor 不只知道目标位置，还知道光度 action。

### 3.2 训练目标

核心损失为：

$$
\mathcal L(x,y)=\sum_{i\in M_x^C}
\left\|p_\phi\left(f_\theta(x),a_{x\rightarrow y},m_a\right)_i-
f_\theta^{\mathrm{EMA}}(y)_i\right\|_2^2.
$$

$M_x$ 表示 source 中保留/遮挡结构，论文记号 $M_x^C$ 对应要预测的 target 位置；$m_a$ 是融合位置和 action 的 mask token。EMA teacher 是避免 representation collapse 的关键。损失本身没有负样本、像素 decoder 或显式方差—协方差项，学习信号来自 masked prediction、teacher 动量更新和动作条件。

### 3.3 两种 action conditioning

**Sequence conditioning** 把每种变换表示为额外 token，并让不同 token 经过独立线性层以打破 Transformer 对排列的等变性。**Feature conditioning** 则把 action 标量拼到 mask token 的 feature 维，再通过 1×1/三层 MLP 混合回 predictor 维度。无条件时 MRR 为 0.00；sequence 为 0.82；feature 为 0.79。作者虽承认 sequence 的 MRR 略高，但因 feature 的下游表现更好而采用后者。

这组实验说明：若 predictor 不知道 action，同一个 source latent 面对多个可能 target，最小化均方误差的容易路径是让 encoder 去除 action 相关信息，回到传统不变 SSL；显式条件消除了这一多解性。

### 3.4 预测难度与容量

论文把 predictor 记为 $\mathrm{IWM}_{X,Y}^{Z}$，其中 $X$ 是深度、$Y$ 是 embedding dimension、$Z\in\{\mathrm{Inv},\mathrm{Equi}\}$ 表示观察到的不变/等变行为。encoder 统一为 ViT-B/16；主要比较 12 层、384 维和 18 层、384 维 predictor。

关键 MRR 结果如下：

| 设置 | I-JEPA 12×384 | IWM 12×384 | IWM 18×384 |
|---|---:|---:|---:|
| 普通 color jitter | 0.00 | 0.11 | 0.25 |
| + destructive aug. | 0.00 | 0.09 | 0.79 |
| + strong jitter | 0.00 | 0.81 | 0.85 |

MRR 越接近 1，表示 predictor 生成的 latent 越能从 256 个候选 target 中检索到正确变换。结果有一点反直觉：**变换更强反而更容易得到强 world model**。解释是弱任务让 encoder/predictor 可以走捷径，近似恒等或不变映射已经够用；强任务迫使 action 真正进入预测。容量同样关键：在含颜色的五种场景中，12 层 predictor 仅 1/5 次呈现颜色等变，18 层为 4/5 次，说明等变行为还带有训练不稳定和随机性。

### 3.5 不变与等变的表示谱

设变换为 $\mathcal T(a)$。不变表示追求：

$$
f_\theta(x)=f_\theta(\mathcal T(a,x)),
$$

因而简单线性头就容易读取语义；但实现这种不变的常见方式是直接丢掉颜色、纹理等变换相关信息。等变模型则追求存在 $p_\phi$ 使：

$$
p_\phi(f_\theta(x),a)\approx f_\theta(\mathcal T(a,x)),
$$

encoder 可以保留更多信息，再由 predictor 解释变化。论文的核心观点不是“等变永远更好”，而是下游 protocol 决定哪种表示占优：线性 probing 偏爱高度抽象的不变表示；复杂 predictor finetuning 更能利用等变表示中的丰富信息。

### 3.6 predictor finetuning

预训练 predictor 输出的是另一组视觉 latent，不能直接完成分类。作者冻结 encoder，把 teacher 输出与 null transformation/待预测 token 送入 predictor，在输出上接 attentive head，并微调 predictor 与任务头。默认让 predictor 预测一张未变换完整图的 latent；用 teacher 比 student 高 0.3 点，null latents 再高 0.1 点；只预测一个 token 可节省 token 数，但 ImageNet 从 83.3 降到 82.8。

与普通“大头微调”相比，关键对照是同架构随机初始化 predictor。若预训练 predictor 只比随机头略好，说明 world model 没学到可复用结构；等变 IWM 的预训练 predictor 比随机头高 1.8 点，提供了最直接的复用证据。

### 3.7 多任务 predictor tuning

每个任务增加一个 learned task token 和任务专用 head/loss，同一 predictor 共享参数，batch 均匀分给 ImageNet、iNaturalist18、SUN397、Places205。单任务平均 73.4，多任务平均 73.5；分别为 ImageNet 80.8→79.6、iNat18 72.4→72.0、SUN397 75.6→78.2、Places205 64.8→64.1。平均无损不等于每项无损：SUN397 的正迁移抵消了其他三项下降，但确实说明共享 predictor 可摊薄多任务适配成本。

---

## 4. 数据集、基线与实现细节

### 4.1 预训练与增强

所有主模型在 ImageNet-1k 上用 ViT-B/16 训练 300 epoch。共同 crop scale 为 0.3–1，水平翻转概率 0.5。target color jitter 概率 0.8，brightness/contrast/hue/saturation 最大强度分别为 0.4/0.4/0.1/0.2。source 在同类 jitter 外，再以 0.2 概率施加 blur、solarization、grayscale。mask 是四个矩形并集，每块面积占图像 0.15–0.2，长宽比 0.75–1.5。

优化器 AdamW，学习率 $10^{-3}$，$\beta_1=0.9,\beta_2=0.999$；40 epoch 线性 warmup 后 cosine annealing；weight decay 从 0.04 余弦增加到 0.4。论文披露了 epoch、优化器和评估 batch 等细节，但主文/附录没有清楚报告主预训练所用 GPU 型号、GPU 数量与墙钟时间，因此不能补造总算力。

### 4.2 评测数据与 baseline

表征学习主要与 DINO、MoCo v3、iBOT、MAE、I-JEPA 比较。ImageNet 评估包括 linear、attentive probing、encoder finetuning、frozen encoder + random/pretrained predictor、encoder/predictor end-to-end finetuning。ADE20K 使用 UperNet、160k iterations，报告 mIoU。域外 attentive probing 使用 iNaturalist18、SUN397、Places205。world model 质量用 256 候选增强视图上的 Mean Reciprocal Rank（MRR）。

需要注意 baseline 的预训练 epoch 不全相同：DINO/iBOT/MAE 的强结果常来自 1600 epoch，而 IWM 主模型是 300 epoch；“peak performance”表适合看可达到的结果，但不是严格同算力公平比较。更可信的是 IWM 与 I-JEPA 同为 ViT-B/16、300 epoch 的对照。

---

## 5. 实验结果

### 5.1 ImageNet 微调

| 方法（300 epoch，ViT-B/16） | encoder 微调 | 冻结 encoder + 随机 predictor | 冻结 encoder + 预训练 predictor | 端到端 |
|---|---:|---:|---:|---:|
| I-JEPA | 83.0 | 79.1 | 80.0（+0.9） | 82.0 |
| IWM 12×384 Inv | 83.3 | 80.5 | 81.3（+0.8） | 82.7 |
| IWM 12×384 Equi | 82.7 | 81.3 | 82.7（+1.4） | 83.3 |
| IWM 18×384 Equi | 82.9 | 81.5 | **83.3（+1.8）** | **84.4** |

不变 IWM 的 encoder 微调略强，而等变 IWM 的 predictor 复用和端到端上限更高。18 层等变 predictor 在冻结 encoder 时达到 83.3，与不变 encoder 微调持平；端到端 84.4 是表中 300 epoch 配方最好值。这里的交换关系非常清晰：高 MRR 并不直接带来最好线性/encoder-only 性能，却与 predictor 适配能力强相关。

![](assets/Image%20World%20Models%20-%20动作条件图像表征预测/figures/03-finetuning-efficiency.png)

> 图 3：横轴把可训练参数量换算成 encoder block 数。相近参数预算下，IWM predictor finetuning 的星标结果高于只微调 encoder block 的曲线，说明收益不只是“训练了更多参数”。但推理时仍需保留 predictor，不能把适配成本降低等同于零推理开销。  
> 来源：论文 Figure 3，第 6 页；arXiv 官方源码 `images/efficiency.pdf`。

### 5.2 ADE20K 分割

I-JEPA 的 encoder/predictor/end-to-end mIoU 为 44.2/45.4/45.1；不变 IWM 为 45.6/45.7/46.5；18 层等变 IWM 为 44.2/**46.8**/**47.0**。这比分类更能说明 predictor 不只是类别级 head：最后四层 predictor feature 可接 UperNet，且对稠密任务也有效。不过所有方案都使用细调过的 UperNet 训练，不能视作零样本分割能力。

### 5.3 线性、attentive 与域外评测

ImageNet linear/attentive：MoCo v3 为 76.3/76.4，I-JEPA 为 70.0/75.0，不变 IWM 为 **74.5/77.0**，等变 IWM 为 67.5/75.1。线性 probe 明显偏爱不变模型；attentive head 缩小差距。域外 attentive probing 中，等变 IWM 在 iNat18/SUN397/Places205 得 54.2/71.7/60.5，高于不变 IWM 的 51.6/71.0/59.4，支持“保留更多细节有助分布迁移”。

![](assets/Image%20World%20Models%20-%20动作条件图像表征预测/figures/04-equivariance-abstraction.png)

> 图 4：MRR 与不同评估协议的相关性。MRR 对 linear accuracy 为强负相关（$\rho=-0.85$），对 predictor finetuning 为强正相关（$\rho=0.93$），对 attentive probing 相关较弱。它把“表示质量”拆成了抽象可读性与可适配信息量，而不是用单一 probe 下结论。  
> 来源：论文 Figure 4，第 9 页；arXiv 官方源码 `images/modularity.pdf`。

### 5.4 关键消融

**conditioning。** 无 action 条件 MRR=0；sequence=0.82；feature=0.79。条件化是必要条件，但最高 MRR 不等于最好下游性能。

**增强强度。** 12 层 IWM 在默认增强下 MRR 0.09、linear 74.5、predictor FT 81.3；增强更强时 MRR 0.81、linear 降到 69.3、predictor FT 升到 82.7。18 层默认 MRR 0.79、linear 67.5、predictor FT 83.3；更强并非无限有益，极强 jitter 可使 linear 跌到 34.3，predictor FT 也降到 81.7。难度需要与容量匹配。

**非对称增强。** invariant predictor 改为对称增强时 attentive 下降 2 点、linear 下降 1.5 点；equivariant predictor 基本不受影响。强 world model 减少了对增强工程的敏感性，但这种结论只在论文测试的 photometric family 内成立。

**预测任务。** teacher、null latent 和 full-image prediction 组合最好（83.3）；单 token 约损失 0.5 点，显示计算与精度可权衡。

**规模。** ViT-L/16 下，不变 IWM encoder 微调 84.3；36 层、512 维等变 predictor 微调 85.0、端到端 85.4。作者经验上发现 predictor/encoder 参数比约 0.3 合适；扩大 width 会引入 EMA 与训练不稳定问题。

### 5.5 实验是否支撑主张

论文通过 MRR、随机 predictor 对照、多个 adaptation protocol、分类/分割/域外任务，较扎实地支撑“强 predictor 可复用”与“容量控制表示性质”。弱点是 world model action 全为人工增强，验证任务仍是标准识别；没有规划、控制、自然动力学或多步 rollout。因此实验支撑的是**视觉表征学习中的 latent transform model**，而不是广义物理世界模型。

### 5.6 从相关性到因果解释还差什么

论文把 MRR 与 linear、attentive、predictor finetuning 的相关曲线解释为“等变性控制抽象度”，这很有启发，但尚不是严格因果证明。改变 predictor 深度时，参数量、优化难度、梯度流和 teacher-student 动力学一起变化；改变增强强度时，source 信息量和任务难度也同时变化。因此高 MRR 可能既是等变能力的指标，也在部分反映更大模型或不同训练状态。论文用 random-init predictor、相同 ViT-B/16 和多种增强消融削弱了这些混淆，却没有做到固定参数量下只操纵等变性，也没有报告完整随机种子置信区间。更强的验证应包括：对同一预训练模型人为屏蔽 action、打乱 action 与样本配对、用低秩瓶颈匹配不同 predictor 的有效容量，以及在相同 FLOPs 下比较等变与不变解。还应直接测量 latent 中颜色、纹理、位置和语义信息的可解码性，而不只通过下游 accuracy 间接推断。因而我的结论是：论文非常有力地展示了一个稳定现象和可用 recipe，但“predictor 能力决定表示抽象层级”仍更接近受充分实验支持的机制假说，而非已被隔离变量完全证明的定律。

---

## 6. 与 I-JEPA、V-JEPA、MC-JEPA 的关系

### 6.1 对 I-JEPA 的直接扩展

I-JEPA 用 context patch 预测同一图像中 masked target patch 的 EMA latent，predictor 获得位置信息，但不显式处理 source 与 target 的全局 photometric 差异。IWM 保留 ViT student/EMA teacher/mask token/latent $L_2$ 骨架，把 action 参数混入 mask token，并刻意制造更强的 source-target 变换。可以把 I-JEPA 看成 IWM 的一个弱 action/主要做 spatial inpainting 的特例。

两者目标也不同：I-JEPA 主要追求 encoder 表征，predictor 可丢弃；IWM 把 predictor 是否能应用变换作为 MRR 评测对象，并主张下游继续使用。实验中 I-JEPA predictor 仅比随机头高 0.9 点，而 18 层等变 IWM 高 1.8 点，正是这一区别的证据。

### 6.2 与 V-JEPA 的关系

V-JEPA 把 latent masked prediction 扩展到视频时空 tube，让上下文预测被遮挡的视频区域；它不显式提供真实动作，时序变化来自自然视频。IWM 仍是静态图像，却提供一个完全可观测、可参数化的 action，适合研究 conditioning、equivariance 和 predictor reuse。前者更接近学习自然动态，后者更像可控的机制实验。

两篇工作在时间上接近，不能把 IWM 写成由 V-JEPA 派生。概念上可组合：未来模型可在视频中同时接收相机/机器人 action、光度或几何条件，预测未来 latent，并把 predictor 用于规划或多任务适配。IWM 给出了“显式 action condition 为什么重要”的静态证据，V-JEPA 给出“无动作视频 latent prediction 可以学到强表征”的动态证据。

### 6.3 与 MC-JEPA 的关系

MC-JEPA 用 PWC-Net 显式预测相邻帧光流，并把 flow reconstruction/cycle/smoothness 与 VICReg 相加，是混合多任务目标；IWM 不输出 flow 或像素，使用单一 EMA latent regression。MC-JEPA 的“运动”来自真实视频但被预先定义为二维对应；IWM 的“动作”是人工增强但由 predictor 在 latent 中执行。前者擅长稠密 correspondence，后者更纯粹地研究 world model 的条件化、容量和复用。

---

## 7. 局限、失败模式与未来工作

### 7.1 方法边界

1. **action 过于人工化**：brightness、hue、blur 等并不是环境动力学；其参数已知、低维、单步，离自然交互 action 很远。
2. **不可逆变换**：grayscale、blur 会丢失信息。predictor 只能根据数据先验猜测，无法真正逆转；论文的检索可视化明确出现 grayscale inversion 不准确。
3. **MRR 的封闭候选集**：从 256 个增强 target 中检索适合衡量是否识别变换，却不等价于潜空间预测误差足够支持开放式 rollout。
4. **容量—抽象权衡**：强 predictor 可能替 encoder 承担过多任务，导致 linear probe 变差；“信息丰富”必须配复杂 head 才能兑现。
5. **适配并非免费**：predictor finetuning少改 encoder 参数并可缓存 backbone，但推理时要保留深 predictor，参数量、延迟和显存会上升。
6. **等变形成不稳定**：12 层模型对相同类别设置只在部分运行中学成等变，18 层虽更稳定仍依赖增强强度、EMA 与容量比例。
7. **缺少官方代码/权重**：截至本笔记日期未找到可核验的官方 IWM 仓库，精确复现 action 编码和训练 recipe 有门槛。

### 7.2 失败模式

- **灰度反演**：source 已失去颜色，action 只能告诉模型“发生过灰度化”，不能告诉原色；nearest-neighbor 可能恢复语义相近却颜色错误的图。
- **任务太容易**：弱 jitter 或无 conditioning 时 MRR 接近 0，模型退化为不变 SSL；loss 仍可能下降，却没有学到可用 world model。
- **任务太难/容量不足**：增强强于 predictor 能力时，模型通过 encoder 丢弃变化相关信息；增加难度不会单调提升性能。
- **过强增强**：18 层模型在极强 jitter 下 MRR 仍高，但 linear accuracy 可降至 34.3，说明“会模拟增强”与“语义可读”可严重背离。
- **自然变换外推**：模型只在增强参数分布内训练；对组合范围外的颜色、真实照明、遮挡、形变或 3D 视角变化没有保证。
- **多任务负迁移**：平均值持平掩盖 ImageNet -1.2、Places205 -0.7 等下降；任务 token 不是自动消除冲突的机制。

### 7.3 一个重要的负结果

附录尝试对 8、16、32、128 个 action 预测取平均，构造“保留信息的 invariant representation”：

$$
z_x^{\mathrm{Inv}}=\frac{1}{N}\sum_{i=1}^N
p_\phi\bigl(f_\theta(x),A_i,m_{A_i}\bigr).
$$

ImageNet linear accuracy 在 $N=1,8,16,32,128$ 时分别约 64.5、64.3、64.6、64.6、64.4，没有收益，128 次预测还把计算放大约 64 倍。它说明对增强边缘化得到“不变”并不等同于对下游有用；对比学习有效，可能正因为它通过**删除**非语义信息形成不变，而非把所有变换都保留再平均。

### 7.4 未来方向

最直接的延伸是把 action 从 photometric 参数换成真实控制、相机运动、深度/姿态或视频时间间隔，并检验多步 latent rollout 是否稳定；把单一 $L_2$ 点预测扩展为能表达多模态不确定性的 energy-based 或分布预测；研究 predictor capacity 与 encoder information bottleneck 的可计算标度规律；把 predictor 复用到检测、深度、光流、VLA 控制和规划，而不只分类/分割；用任务路由、低秩适配或 mixture-of-experts 减少深 predictor 的推理负担。

---

## 8. 复现成本与实践建议

### 8.1 成本判断

主实验需要 ImageNet-1k、ViT-B/16、12/18 层 384 维 predictor、300 epoch 预训练，再加大量 90/100 epoch probing 和 finetuning、160k iteration ADE20K，以及增强/容量多组消融。虽然论文没有披露主预训练 GPU 数量和时长，但实验矩阵很大，完整复现成本显著高于单次 I-JEPA 训练。ViT-L 还需 36×512 predictor；作者明确提到 width scaling、EMA schedule 和 predictor/encoder 参数比会引发稳定性问题。

### 8.2 最小复现路线

1. 以公开 I-JEPA 实现为骨架，保留 ViT-B/16 student、EMA teacher 和 mask sampler。
2. 先只加入可逆 color jitter 参数，验证无 conditioning 时 MRR≈0、加入 feature conditioning 后 MRR 上升。
3. 实现 256 候选的 MRR，而不是只看 pretraining loss；否则无法判断模型是 world model 还是 invariant shortcut。
4. 固定 encoder/predictor 架构，做 random-init predictor 与 pretrained predictor 的相同微调对照，这是复用主张的必要证据。
5. 再加入 grayscale/blur/solarization，分别扫描 12/18 层 predictor、增强强度和多个随机种子，记录等变成功率。
6. 最后再做 ADE20K 与多任务 tuning；若资源有限，优先复核 ImageNet linear 与 predictor FT 的反向趋势。

监控项应包括 MRR、linear/attentive accuracy、action 参数对预测的敏感度、teacher/student 特征范数、EMA momentum、不同增强的分项 MRR，以及 predictor 相对 encoder 的参数量与 FLOPs。

### 8.3 个人判断

这篇论文最有洞察力之处，是把“predictor 只是训练时辅助头”的默认习惯变成了可检验问题，并用随机同构 predictor 对照证明：只有真的学会等变变换的 world model 才值得保留。它也提醒我们，linear probe 不是表示质量的唯一尺度；一个线性不可读、却能经少量结构化适配释放更多信息的表示，可能更适合作为通用底座。

但“Image World Model”这个名称容易让人高估其物理建模范围。当前 action 与 transition 都由数据增强器定义，模型没有发现环境动力学，也没有处理多步误差、随机未来或规划。我的判断是：IWM 更像一篇关于**可控 latent transformation 与 representation abstraction 的机制论文**，其理论启发大于直接世界模拟能力。若与 V-JEPA、动作条件视频模型结合，这套 conditioning/capacity/reuse 分析才可能走向真正的 action-conditioned world model。

---

## 9. 材料来源

- [arXiv 论文页](https://arxiv.org/abs/2403.00504)
- [官方 PDF](https://arxiv.org/pdf/2403.00504)
- [arXiv HTML](https://arxiv.org/html/2403.00504v1)
- 本地来源清单：`assets/Image World Models - 动作条件图像表征预测/sources.md`
- 本地官方 PDF：`assets/Image World Models - 动作条件图像表征预测/sources/paper.pdf`
- 本地 arXiv 源码：`assets/Image World Models - 动作条件图像表征预测/sources/arxiv-source/`

> 事实边界：公式、图、数字和配置均以官方 PDF/TeX 为准；未找到的信息明确保留为空。本文把“world model”限定为已知图像变换条件下的潜空间预测器，不把它扩写为未经实验验证的物理模拟或规划系统。