---
type: paper-note
status: complete
domain: VLM
paper: "LLaVA-OneVision: Easy Visual Task Transfer"
year: 2024
arxiv: "2408.03326"
doi: null
source: https://arxiv.org/abs/2408.03326
project: https://llava-vl.github.io/blog/2024-08-05-llava-onevision
code: null
tags:
  - VLM
  - multi-image
  - video-understanding
  - task-transfer
created: 2026-10-15
updated: 2026-10-15
---

# LLaVA-OneVision：单图多图视频统一迁移

## 基本信息

- **作者**：Bo Li、Yuanhan Zhang、Dong Guo 等
- **模型规模**：0.5B、7B、72B
- **基础组件**：Qwen-2 LLM、SigLIP 视觉编码器、2 层 MLP projector
- **研究目标**：用一个模型同时覆盖单图、多图和视频，并通过统一视觉序列与课程训练实现跨场景任务迁移。

**一句话结论：**

LLaVA-OneVision 的关键不是发明新骨干，而是把单图 crop、多图图像和视频帧统一成长度相近的视觉 token 序列，再以“强单图基础→混合场景后训练”实现迁移；实验显示多图和视频显著提升，但部分单图指标会轻微回退。

---

## 1. 问题定义与核心假设

### 1.1 三类视觉场景为何长期分裂

单图模型追求高分辨率细节，多图模型受上下文长度约束，视频模型则要在帧数和每帧 token 之间取舍。若三者分别设计编码器和训练集，能力难以共享。论文假设：只要三类输入都序列化为相似总长度的视觉 token，并在强单图模型上继续混合训练，图像中学到的 OCR、图表、空间推理和指代能力可以迁移到多图与视频。

### 1.2 输入与输出

输入 $\mathbf{X}_v$ 是单张图像、多张图像或视频帧序列，$\mathbf{X}_q$ 是文本指令，模型自回归生成答案 $\mathbf{X}_a$：

$$p(\mathbf{X}_a\mid\mathbf{X}_v,\mathbf{X}_q)=\prod_{i=1}^{L}p(\mathbf{x}_i\mid\mathbf{X}_v,\mathbf{X}_{q,<i},\mathbf{X}_{a,<i})$$

视觉编码器 $g_{\psi}$ 和 projector $p_{\theta}$ 给出：

$$\mathbf{Z}_v=g_{\psi}(\mathbf{X}_v),\quad\mathbf{H}_v=p_{\theta}(\mathbf{Z}_v)$$

架构保持 LLaVA 的简单性：视觉 token 与文本 token 拼接后进入 Qwen-2。三种场景的基本视觉单位分别是单图 crop、多图中的单张图和视频中的单帧。

![](assets/LLaVA-OneVision%20-%20单图多图视频统一迁移/figures/01-model-overview.png)

> **Figure 1，PDF p.3。** 单图、多图和视频共享 SigLIP—MLP—Qwen-2 结构；差异主要在视觉输入如何切分和压缩，而非另设场景专用网络。来源：arXiv 源码原图。

---

## 2. Higher AnyRes 与统一 token 策略

### 2.1 AnyRes 公式

若图像使用 $a\times b$ 个高分辨率 crop，并加一张 base image，每个输入原产生 $T$ 个 token，则总数：

$$L=(a\times b+1)T$$

当超过阈值 $\tau$ 时，对每个 crop 双线性插值到：

$$T_{\mathrm{new}}=\begin{cases}\tau/(a\times b+1),&L>\tau\\T,&L\leq\tau\end{cases}$$

作者的实验经验是：在固定计算量下，提高像素分辨率通常比单纯增加 token 更有效，因此优先保留图像细节，再通过 pooling/interpolation 控制序列长度。

### 2.2 三类场景的预算

- **单图**：AnyResMax-9，base image 加最多 9 个 crop，每个 729 token，最大 7,290。
- **多图**：每图只用 base resolution，示例最多 12 图，共 8,748 token。
- **视频**：每帧压到 196 token，示例 32 帧，共 6,272 token。

论文不是强制三者完全等长，而是把最大视觉 token 控制在同一数量级。单图使用长视觉序列，使其在表示形态上更接近视频，为后续迁移提供接口一致性。

![](assets/LLaVA-OneVision%20-%20单图多图视频统一迁移/figures/02-token-strategy.png)

> **Figure 3，PDF p.4。** 单图用高分辨率 crop，多图减少每图成本，视频降低每帧 token 以换取更多帧；三者总预算相近。来源：arXiv 源码原图。

---

## 3. 数据与课程训练

### 3.1 高质量知识数据

Stage-1.5 使用约 4M 样本：LLaVA-NeXT-34B 对 COCO118K、BLIP558K、CC3M 重写得到 3.5M 详细描述；UReader、SynDOG 等形成约 1.1M 文档/OCR 数据；另有 92K 中文描述和 143K 语言样本。论文称该部分约 99.8% 为合成数据。

这意味着“开放模型”不等于“不依赖闭源教师”：部分中文与自由回答使用 GPT-4V/o、Gemini 等标注。论文的贡献在训练组织和模型开放，而不是建立完全开放的数据链。

### 3.2 单图指令数据

单图阶段共 3.2M：General 36.1%、Doc/Chart/Screen 20.6%、Math/Reasoning 20.1%、Language 14.3%、General OCR 8.9%。数据源包含 LLaVA、ShareGPT4V/o、Vision FLAN、Cambrian，以及 VQAv2、GQA、DocVQA、ChartQA、MathQA 等学术数据。

![](assets/LLaVA-OneVision%20-%20单图多图视频统一迁移/figures/04-single-image-data-mixture.png)

> **Figure 4，PDF p.7。** 3.2M 单图混合在通用、文档、数学、语言和 OCR 之间做显式平衡，内圈进一步展示各来源。来源：arXiv 源码原图。

### 3.3 OneVision 混合数据

正文称总量 1.6M，同时列出 560K 多图、350K 视频、800K 单图，简单相加为 1.71M。**本笔记保留原文口径，不自行修正。** 图 5 给出的占比为单图 31.2%、多图 43.0%、视频 25.9%。单图部分是上一阶段数据的平衡重采样；视频含 ActivityNet、Ego4D、NextQA、ShareGPT4Video；多图/多视角含 NLVR、ScanQA、ALFRED、nuScenes、Spot-the-Diff 等。

### 3.4 三阶段训练

![](assets/LLaVA-OneVision%20-%20单图多图视频统一迁移/figures/03-training-process.png)

> **Table 1 顶部流程图，PDF p.8。** Stage-1 做图文对齐，Stage-1.5 注入高质量知识，Stage-2 先单图指令微调、再执行 OneVision 混合训练。来源：arXiv 源码原图。

| 阶段 | 数据 | 可训练模块 | Vision LR | Projector/LLM LR |
|---|---:|---|---:|---:|
| Stage-1 | 558K | Projector | $10^{-3}$ | $10^{-3}$ |
| Stage-1.5 | 4M | 全模型 | $2\times10^{-6}$ | $10^{-5}$ |
| Stage-2 单图 | 3.2M | 全模型 | $2\times10^{-6}$ | $10^{-5}$ |
| Stage-2 OneVision | 1.6M | 全模型 | $2\times10^{-6}$ | $10^{-5}$ |

各阶段训练 1 epoch。0.5B 全局 batch 512，7B/72B 为 256。

---

## 4. 定量结果与 Baseline

### 4.1 单图

72B 在 AI2D/ChartQA/DocVQA/MathVista/MMMU/RealWorldQA 上为 85.6/83.7/91.3/67.5/56.8/71.9。对比 GPT-4V 为 78.2/78.5/88.4/49.9/56.8/61.4；GPT-4o 为 94.2/85.7/92.8/63.8/69.1（部分指标未报告）。

**论文实测**：OneVision 在图表、文档和部分数学任务上超过 GPT-4V，但复杂视觉聊天仍落后 GPT-4o，例如 72B 的 Vibe-Eval 50.7，而 GPT-4o 为 63.1。

### 4.2 多图

72B 在 LLaVA-Interleave out-domain、MuirBench、Mantis、BLINK、Text-rich VQA 上为 79.9/54.8/77.6/55.4/83.7；GPT-4V 为 60.3/62.3/62.7/51.1/54.5。它在多个组合任务强，但 MuirBench 仍低于 GPT-4V。

### 4.3 视频

72B 在 ActivityNet-QA、EgoSchema、MLVU、MVBench、VideoMME、LongVideoBench 上为 62.3/62.0/68.0/59.4/66.2/63.2。GPT-4V 可比项为 57.0/—/49.2/43.5/59.9/60.7；GPT-4o 的 VideoMME/LongVideoBench 为 71.9/66.7。

结论应表述为“对开放模型和 GPT-4V 有竞争力”，而不是全面超过闭源前沿。

---

## 5. 迁移实验与消融

### 5.1 Single-Image checkpoint → OneVision checkpoint

7B 在 OneVision 后：MI-VQA 60.3→90.2、Spot-Diff 7.9→39.2、3D-Chat 24.5→62.8、ALFRED 32.0→61.0、SciVerse 30.0→79.1；视频方面 EgoSchema 52.9→60.1、NextQA 61.6→79.4、VideoMME 55.0/59.1→58.2/61.5。

这些大幅变化是论文最直接的“任务迁移”证据：单图 checkpoint 已有部分视频能力，但混合后训练把关系推理、时序任务与多视角任务显著拉升。

### 5.2 单图并非无损

7B 从 SI 到 OV 后，ChartQA 78.8→80.0、MathVista 56.1→63.2，但 AI2D 81.6→81.4、MMBench 81.7→80.8、MMVet 58.8→57.5。**论文表格实测**说明统一训练存在轻微负迁移，不能只报告新增能力。

### 5.3 视觉表示消融边界

论文正文明确声称分辨率扩展比 token 扩展更有效，但详细 ablation 引用早期 LLaVA-NeXT 工作，本文未给完整数字。引言还承认在有限算力下进行了一次 “yolo run”，没有充分 de-risk 各组件。因此，跨场景迁移主要由 checkpoint 前后对比支持，而不是严格逐因素因果消融。

---

## 6. 涌现能力

论文展示九类迁移示例：图表与图示联合理解、GUI agent、Set-of-Mark、图像到视频编辑指令、视频差异分析、自动驾驶多摄像头、组合子视频、视频视觉提示、以图像为查询目标的视频理解。

其中 Set-of-Mark 没有被显式加入训练数据，视频视觉提示也未使用对应视频训练样本。**论文声称**这些是跨任务组合产生的涌现；**个人判断**：定性案例能证明可行性，但不足以量化出现频率和稳健性，仍需专门 benchmark。

---

## 7. 与同类路线的关系

相比 Molmo，OneVision 更依赖 VLM 生成的合成数据，但覆盖单图—多图—视频更完整；相比 PaliGemma，它不是下游迁移 base model，而是直接面向通用指令跟随；相比 Pixtral，它通过 AnyRes 切块与插值适配分辨率，而非从头训练原生可变长宽比 ViT；相比 Cambrian-1，它保持单视觉编码器与简单 projector，把主要复杂度放在数据和 token 预算。

这条路线的核心哲学是：先把强单图模型做成“能力源”，再把场景统一视为不同长度和粒度的视觉序列问题。

---

## 8. 局限、Future Work 与复现成本

### 8.1 明确局限

- 设计受固定计算预算约束，各组件未被充分逐项验证。
- 高质量知识数据约 99.8% 为合成数据，并使用闭源模型生成部分监督。
- 多个 benchmark 训练集进入混合数据，带 `†` 结果应视为 in-domain。
- 复杂视觉聊天仍落后 GPT-4V/o。
- OneVision 后训练会让部分单图指标下降。
- 文中 1.6M 总量与分项之和存在口径不一致。

### 8.2 Future Work

作者明确提出更强 LLM、更大训练数据、更好的 preference learning、继续扩展数据与模型、增加每图/每帧 token，以及在该模型上构建具体应用。

### 8.3 复现成本

训练 72B 且支持约 7K–9K 视觉 token，显存、序列并行和数据吞吐成本很高。7B 更适合复现；关键难点不是 projector，而是复刻 4M 知识、3.2M 单图与 1.6M 混合数据的过滤、模板和采样。论文附录提供统计，但闭源教师生成数据仍限制完全独立复现。

### 8.4 个人判断

**判断**：OneVision 证明统一场景不一定需要新架构，但也表明“统一”不是免费午餐。强单图先验和相近 token 预算是迁移前提，混合训练则带来场景增益与单图回退。后续原生分辨率和动态 token 路线，实质上是在解决它仍需手工指定 crop/帧预算的问题。

---

## 9. 复现建议

1. 从 7B SI checkpoint 出发，先只复现 OneVision 后训练，避免重复 Stage-1/1.5。
2. 严格区分 in-domain 与 out-of-domain 多图结果。
3. 同时跟踪单图保留率，避免只优化视频/多图平均分。
4. 对视频帧数与每帧 token 做等 FLOPs 消融，验证迁移是否来自表示统一而非计算增加。

---

## 10. 材料来源

官方 PDF、四张完整源码图和来源记录见：[assets/LLaVA-OneVision - 单图多图视频统一迁移/sources.md](assets/LLaVA-OneVision%20-%20单图多图视频统一迁移/sources.md)。