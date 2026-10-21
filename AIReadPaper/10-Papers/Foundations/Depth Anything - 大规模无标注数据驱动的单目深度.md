---
type: paper-note
status: complete
domain: Foundations
series: Depth Anything
series_order: 1
series_hub: "[[从 Depth Anything 到 DA3：前馈深度与三维几何 - Survey]]"
paper: "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"
year: 2024
arxiv: "2401.10891"
source: https://arxiv.org/abs/2401.10891
project: https://depth-anything.github.io/
code: https://github.com/LiheYoung/Depth-Anything
tags:
  - monocular-depth
  - foundation-model
  - semi-supervised-learning
  - DINOv2
created: 2026-10-15
updated: 2026-10-15
---

# Depth Anything：大规模无标注数据驱动的单目深度

> 系列导航：**Depth Anything 1** → [[Depth Anything V2 - 精确合成监督与真实伪标签桥接]] → [[DA3 - 任意视角恢复视觉空间]] · [[从 Depth Anything 到 DA3：前馈深度与三维几何 - Survey|系列综述]]

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data** |
| 作者 | Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao |
| 会议 | CVPR 2024 |
| 核心任务 | 单张 RGB 图像 → 仿射不变逆深度；微调后可做度量深度 |
| 模型 | DINOv2 ViT-S/B/L + DPT decoder，24.8M / 97.5M / 335.3M 参数 |
| 数据 | 约 1.5M 有标签图像 + 62M 无标签图像 |
| 重要性 | ★★★★★：把单目深度从“多数据集训练”推进到“视觉基础模型 + 超大规模无标签数据”阶段 |

> **一句话结论：** Depth Anything 的真正贡献不是新网络，而是一套可扩展的数据引擎：先用 1.5M 多源深度数据训练 teacher，再给 62M 开放域图像打伪标签；student 只有在面对强颜色扰动、CutMix 和 DINOv2 语义保持约束时，才会从伪标签中学到 teacher 之外的稳健知识。这证明单目深度的零样本泛化瓶颈主要在**数据覆盖与学习目标**，而不是更复杂的深度头。

---

## 1. 研究背景与问题定义

### 1.1 从数据集专用深度到开放域深度

单目深度本质上欠约束：一张图无法唯一决定真实尺度。早期模型通常在 NYUv2、KITTI 等固定域内学习米制深度，换相机、换场景就明显退化。MiDaS 用多数据集混合训练和尺度—平移不变损失，把目标改成跨域相对深度，但训练规模仍受人工或几何重建标签限制。

Depth Anything 追问：如果架构已经有 DINOv2 这样的强视觉先验，是否能通过海量无标签图像，把数据覆盖扩展到“任何图像、任何场景”？它选择不发明专用模块，而是把重点放在三件事上：大规模伪标签、强扰动学习、语义先验保持。

![](assets/Depth%20Anything%20-%20大规模无标注数据驱动的单目深度/figures/01-teaser.png)

> 图 1：开放域图像上的深度结果，覆盖室内、聚餐、低光、建筑、雾天与航拍。重点不在某个单项 benchmark，而在训练分布之外仍能给出合理层次。来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2401.10891

### 1.2 任务与输出

输入是单张 RGB 图像。基础模型输出仿射不变 inverse depth：不同图像之间没有统一米制尺度，预测允许全局 scale 和 shift。训练前把真实深度 $t$ 转为视差 $d=1/t$，并按图归一化。度量深度不是基础模型原生能力，而是把 Depth Anything encoder 接到 ZoeDepth 风格的 metric head 上，在 NYUv2 或 KITTI 上微调获得。

这一点很重要：V1 的“Anything”首先指**开放域相对几何泛化**，不是无需标定的通用米制测量。

---

## 2. 数据引擎

### 2.1 有标签数据：约 1.5M

论文混合 6 个来源：BlendedMVS、DIML、HRWSI、IRS、MegaDepth、TartanAir。标签分别来自双目、SfM 或合成渲染，覆盖室内、室外、驾驶、航拍等域。论文刻意不使用 NYUv2 与 KITTI 的训练图像，以保留严格 zero-shot 评测。

### 2.2 无标签数据：62M

8 个来源合计约 62.3M：BDD100K、Google Landmarks、ImageNet-21K、LSUN、Objects365、Open Images V7、Places365、SA-1B。只使用原始图像，不使用分类、检测或分割标注。

这里的关键并非“重复更多轮次”，而是跨数据集场景多样性。后续 V2 的消融进一步验证：只在 11M SA-1B 上多跑若干轮，无法替代 62M 多源真实图像。

---

## 3. 核心方法

### 3.1 架构：DINOv2 + DPT

- encoder：DINOv2 ViT-S/B/L；
- decoder：DPT，从多层 token 还原逐像素深度；
- 输入训练分辨率：短边缩放至 518，再随机裁剪为 $518\times518$；
- 推理时保持宽高比，把尺寸调整为 14 的倍数，最后插值回原图。

架构刻意保持普通，便于把性能归因于数据和训练目标。

### 3.2 Teacher–Student 流程

1. 在有标签数据上训练 teacher；
2. teacher 为 62M 无标签图像生成伪深度；
3. student 从头初始化，在真实标签与伪标签上联合训练；
4. 每个 batch 有标签与无标签样本比例为 1:2，student 遍历无标签集合一次。

![](assets/Depth%20Anything%20-%20大规模无标注数据驱动的单目深度/figures/02-training-pipeline.png)

> 图 2：V1 数据引擎。无标签图像的 student 输入受到强扰动，而伪标签由 clean image 上的 teacher 产生；DINOv2 特征约束用于保存语义先验。来源：论文 Figure 2，第 4 页。

### 3.3 为什么朴素伪标签不够

如果 teacher 与 student 共享类似架构和预训练，两者很容易复制同样的正确与错误。论文消融显示，只加入伪标签后六个测试集并没有稳定提升，NYU、ETH3D、DIODE 甚至略退化。因此“62M”本身不是答案，必须让 student 面对比 teacher 更难的任务。

### 3.4 强扰动：迫使 student 学到额外知识

只对无标签图像的 student 分支使用：

- Color Jitter；
- Gaussian Blur；
- 50% 概率 CutMix。

CutMix 样本写作：

$$u_{ab}=u_a\odot M+u_b\odot(1-M)$$

两块区域分别对齐各自 teacher 伪标签，再按区域面积加权。student 必须从颜色和空间结构被破坏的输入中恢复稳定深度，因此学到更强不变性，而不是机械复制 teacher。

### 3.5 语义保持：连续特征优于离散 mask

作者曾用 RAM、GroundingDINO、HQ-SAM 生成约 4K 类语义 mask，但没有收益。离散分割把丰富语义压成类别边界，且无法表达物体内部的深度变化。最终方法直接对齐 student encoder 与冻结 DINOv2 的逐像素特征：

$$\mathcal L_{feat}=1-\frac{1}{HW}\sum_i\cos(f_i,f'_i)$$

设置容忍阈值 $\alpha=0.85$：相似度超过阈值的像素不再优化。这样只要求保留足够语义，不强迫深度 encoder 完全复制 DINOv2。

---

## 4. 损失函数

### 4.1 仿射不变深度损失

预测与真值先做平移和尺度归一化：

$$\hat d_i=\frac{d_i-t(d)}{s(d)},\quad t(d)=\operatorname{median}(d),\quad s(d)=\frac{1}{HW}\sum_i|d_i-t(d)|$$

再计算逐像素 L1：

$$\mathcal L_l=\frac{1}{HW}\sum_i|\hat d_i^*-\hat d_i|$$

该目标允许不同数据集的深度尺度与偏移不一致，是多源相对深度联合训练的基础。

### 4.2 总目标

$$\mathcal L=\operatorname{avg}(\mathcal L_l,\mathcal L_u,\mathcal L_{feat})$$

三项分别对应真实标签、无标签伪深度、DINOv2 语义保持。

---

## 5. 实验结果

### 5.1 Zero-shot 相对深度

ViT-L 相比 MiDaS v3.1 ViT-L：

| 数据集 | MiDaS AbsRel | Depth Anything AbsRel | MiDaS $\delta_1$ | Depth Anything $\delta_1$ |
|---|---:|---:|---:|---:|
| KITTI | 0.127 | **0.076** | 0.850 | **0.947** |
| NYUv2 | 0.048 | **0.043** | 0.980 | **0.981** |
| Sintel | 0.587 | **0.458** | 0.699 | **0.760** |
| DDAD | 0.251 | **0.230** | 0.766 | **0.789** |
| ETH3D | 0.139 | **0.127** | 0.867 | **0.882** |
| DIODE | 0.075 | **0.066** | 0.942 | **0.952** |

MiDaS 使用过 KITTI/NYUv2 训练图像，而 Depth Anything 没有；因此前两项对 V1 更严格。

![](assets/Depth%20Anything%20-%20大规模无标注数据驱动的单目深度/figures/03-zero-shot-results.png)

> 图 3：六个未见数据集上的定性结果。模型在道路、人物、室内、动画与低纹理区域保持相对稳定的层次。来源：论文 Figure 3，第 7 页。

### 5.2 度量深度迁移

在 NYUv2 上，微调后 $\delta_1=0.984$、AbsRel=0.056、RMSE=0.206；在 KITTI 上 $\delta_1=0.982$、AbsRel=0.046、RMSE=1.896。更重要的是，Depth Anything encoder 替换 MiDaS encoder 后，在 NYUv2、KITTI 和语义分割迁移上都更好，说明它不是“只会输出深度”的任务头，而是更强的密集视觉表征。

### 5.3 语义分割迁移

- Cityscapes：84.8 mIoU（单尺度），86.2（多尺度）；
- ADE20K：59.4 mIoU。

大规模深度训练没有破坏 DINOv2 语义，反而保留甚至增强下游迁移能力。

---

## 6. 消融实验

### 6.1 从“伪标签无效”到完整模型

六数据集 AbsRel：

| 配置 | KITTI | NYU | Sintel | DDAD | ETH3D | DIODE |
|---|---:|---:|---:|---:|---:|---:|
| 仅有标签 | 0.085 | 0.053 | 0.492 | 0.245 | 0.134 | 0.070 |
| +朴素伪标签 | 0.085 | 0.054 | 0.481 | 0.242 | 0.138 | 0.073 |
| +强扰动 | 0.081 | 0.048 | 0.469 | 0.235 | 0.134 | 0.068 |
| +语义保持 | **0.076** | **0.043** | **0.458** | **0.230** | **0.127** | **0.066** |

最重要的反直觉结论是：**无标签规模不是自动收益；只有优化目标迫使 student 解决 teacher 没有直接给出的难题时，数据规模才真正转化成泛化。**

### 6.2 特征对齐的位置与强度

- 容忍阈值：$\alpha=0.85$ 的平均 AbsRel 0.175，优于 1.0 的 0.188；
- 只在无标签数据施加特征对齐，平均 AbsRel 0.175，优于有标签数据上的 0.179。

解释是：真实标签已经较可靠，额外语义约束可能干扰深度；伪标签噪声大，更需要语义先验抗噪。

---

## 7. 与后续工作的关系

- **V2** 将 V1 的问题进一步定位到“真实深度标签本身粗糙且有噪声”，用合成精确监督训练 giant teacher，再用真实伪标签桥接域差；
- **UniDepth / Metric3D v2 / Depth Pro / MoGe** 解决 V1 未原生解决的度量尺度、相机内参、边界锐度或直接 3D pointmap；
- **DA3** 把单目 depth foundation model 扩展为任意视图的空间恢复模型，以 depth+ray 统一深度、相机和多视图几何。

---

## 8. 局限与批判性评价

### 8.1 论文明确局限

- 最大仅 ViT-L；作者计划扩到 ViT-G；
- 训练分辨率 $518\times518$，对发丝、栏杆等高频边界仍不足；
- 基础模型只输出相对/仿射不变深度，米制深度依赖下游微调。

### 8.2 个人判断

V1 最值得复用的不是 62M 这个数字，而是**“伪标签 + 难化输入 + 保留基础模型先验”**的三件套。它也暴露出深度 benchmark 的结构性问题：传统稠密 GT 来自 LiDAR、双目或 SfM，本身在透明面、动态物体、细边界上有错误。V2 正是沿这个缺口出现。

---

## 9. 复现成本与实践建议

- 直接训练完整 V1 成本极高：需要为 62M 图像离线生成伪标签并训练大模型；
- 实践上优先使用官方预训练权重；
- 若做小规模复现，先验证“朴素伪标签无效 → 强扰动有效”这一核心消融，而不是追求 62M 规模；
- 需要 metric depth 时，必须明确室内/室外 domain 与相机条件，不能把相对深度可视化误当成米制测量。

---

## 10. 材料来源

见 [sources.md](assets/Depth%20Anything%20-%20大规模无标注数据驱动的单目深度/sources.md)。
