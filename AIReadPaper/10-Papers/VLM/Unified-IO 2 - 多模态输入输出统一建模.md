---
type: paper-note
status: complete
domain: VLM
paper: "Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action"
year: 2023
arxiv: "2312.17172"
doi: null
source: "https://arxiv.org/abs/2312.17172"
project: null
code: null
tags:
  - VLM
  - unified-multimodal
  - multimodal-generation
  - encoder-decoder
  - autoregressive
  - VQGAN
created: 2026-10-16
updated: 2026-10-16
---

# Unified-IO 2：多模态输入输出统一建模

## 0. 基本信息与阅读边界

- **作者**：Jiasen Lu、Christopher Clark、Sangho Lee、Zichen Zhang、Savya Khosla、Ryan Marten、Derek Hoiem、Aniruddha Kembhavi
- **模型规模**：1.1B / 3.2B / 6.8B；均为 24 层 encoder + 24 层 decoder
- **核心定位**：从零训练的单一 encoder-decoder Transformer，把文本、图像、音频、稀疏结构和动作统一成序列输入与离散 token 输出。
- **本地原文**：[paper.pdf](assets/Unified-IO%202%20-%20多模态输入输出统一建模/sources/paper.pdf)
- **来源清单**：[sources.md](assets/Unified-IO%202%20-%20多模态输入输出统一建模/sources.md)

> [!important] 与普通 image-to-text VLM 的区别
> 本文要补的是现有 VLM 综述常遗漏的**视觉输出架构**：模型不仅把图像编码后回答文本，还会直接自回归生成 VQ-GAN 图像 token，从而输出自然图像、分割、深度、表面法线、未来帧；同一 decoder 还生成音频 token、坐标 token 与动作。若只按 VQA/captioning 理解 Unified-IO 2，会漏掉论文最核心的“输出空间统一”。

**一句话结论：** Unified-IO 2 证明了 encoder-decoder + 离散化输出可以用一个自回归模型覆盖理解、图像/音频生成与动作预测；其关键不是单一 tokenizer，而是输入侧模态专用 encoder、输出侧统一离散词表、Multimodal Mixture of Denoisers，以及为异构模态训练稳定性设计的 QK-Norm、2D RoPE、动态 masking 与 packing。

![](assets/Unified-IO%202%20-%20多模态输入输出统一建模/figures/01-capability-overview.png)

> **Figure 1，PDF p.1。** 红框是视觉输出，蓝框是文本/稀疏结构/动作输出，绿框是音频输出；这张图直接说明它不是“只能看图说话”的 VLM。来源：官方 PDF 完整图，已 Read 验证。

---

## 1. 问题定义：统一的是输入、主干、输出与任务接口

### 1.1 现有路线的缺口

典型 VLM 用视觉 encoder + adapter 把图像变成 LLM 的条件，输出仍只有文本。它能做 VQA、captioning、OCR，却不能在同一个原生输出空间里生成图像、dense map、声音或动作。另一类 any-to-any 系统通常拼接多个专家模型或扩散模型，模块多、训练目标割裂。

Unified-IO 2 选择更激进的路线：把所有可序列化输出离散化，让一个 encoder-decoder Transformer 学习条件分布。模型从零开始联合预训练，而不是先有一个只懂文本的 LLM 再外挂视觉模块。

### 1.2 输入—输出任务矩阵

| 输入 | 内部表示 | 可生成输出 | 代表任务 |
|---|---|---|---|
| 文本 | LLaMA BPE embedding | 文本、坐标、动作、图像、音频 | NLP、T2I、T2A、机器人指令 |
| 当前图像 | ViT patch feature + Linear | 文本、box/keypoint、mask/depth/normal、图像 | VQA、检测、分割、编辑、视图合成 |
| 图像历史 | ViT + Perceiver，每帧压到 32 token | 文本、动作、下一帧 | 视频理解、跟踪、具身任务 |
| 当前音频 | AST patch feature + Linear | 文本、音频 | 分类、caption、语音/声音生成 |
| 音频历史 | AST + Perceiver，每段压到 16 token | 文本、后续音频 | 长音频条件生成 |
| 稀疏结构/状态 | 1,000 个量化 special token | 文本或结构化 token | box、keypoint、pose、3D cuboid、robot state |

---

## 2. 整体架构：模态专用入口，单一 encoder-decoder 主干，离散输出

![](assets/Unified-IO%202%20-%20多模态输入输出统一建模/figures/02-architecture.png)

> **Figure 2，PDF p.4。** 不同模态先由专用 encoder 变成 embedding，再动态打包进共享 encoder；共享 decoder 自回归输出文本/动作/稀疏结构、图像或音频 token，最后由对应 de-tokenizer 还原。来源：官方 PDF 完整图，已 Read 验证。

### 2.1 为什么是 encoder-decoder，而不是 decoder-only

- **Encoder** 对条件输入做双向融合，适合图像、音频、历史帧与文本指令同时作为证据。
- **Decoder** 只负责按顺序产生目标 token，通过 cross-attention 读取 encoder 状态。
- 输入输出职责分离后，同一图像既可作为 ViT 连续 feature 输入，也可作为 VQ-GAN 离散 token 目标；“理解表示”和“生成表示”不必相同。

这与 Chameleon/Show-o 的 decoder-only early fusion 不同：Unified-IO 2 没有把输入图像也强制塞进生成词表，而是保留高效的视觉/音频 encoder。

### 2.2 视觉输入 tokenizer：ViT，不是 VQ token

当前图像缩放到 $384\times384$，ViT-B/16 形成最多 $24\times24=576$ 个 patch token。模型拼接 ViT 第 2 层与倒数第 2 层特征，再经线性投影进入共享维度，兼顾低层局部细节和高层语义。

历史图像最多 4 帧。每帧先经同一 ViT，再由两层 Perceiver resampler 压成 32 个 latent token；这样当前帧保留细节，历史帧只保留上下文。视频不是原生 3D encoder，而是“当前帧 + 压缩历史帧”的序列化近似。

### 2.3 视觉输出 tokenizer：dense VQ-GAN

输出图像固定为 $256\times256$。VQ-GAN 压缩率为 8，因此离散网格为

$$
\frac{256}{8}\times\frac{256}{8}=32\times32=1024\ \text{image tokens}.
$$

图像 codebook 在正文报告为 16,512，附录实现表列 image vocabulary 为 16,512；相比初代 Unified-IO 的 $16\times16$ token，序列变成 4 倍，但图像细节和 prompt faithfulness 更好。

同一视觉输出通道还承担 dense prediction：

- depth → 归一化灰度图；
- surface normal → $(x,y,z)$ 映射到 RGB；
- segmentation → 给定类别与 box，生成黑白二值 mask；
- instance segmentation → 先检测，再对每个框调用 localized segmentation。

这是一种“把视觉任务变成图像生成”的输出头复用，而不是额外增加 segmentation/depth head。

### 2.4 文本、坐标与动作词表

- 文本使用 LLaMA BPE，基础规模 32,000。
- 额外加入 1,000 个量化位置 token：点用 $(x,y)$ 两枚，框用 4 枚，3D cuboid 用 12 枚。
- 机器人离散动作可直接写成文本命令；连续状态/位姿先量化为 special token。
- 附录 Table 9 给出：主词表 33,280、图像词表 16,512、音频词表 8,320，`Logits via embedding=True`，说明 logits 由 embedding 矩阵产生。三种输出 token 域由任务/模态前缀约束，随后交给 BPE、VQ-GAN 或 ViT-VQGAN decoder。

### 2.5 音频路径

4.08 秒、16 kHz 波形转为 $128\times256$ log-mel spectrogram。AST/16 产生二维 patch 表示；生成侧用自训练 ViT-VQGAN 将 spectrogram 编成 512 个离散 token，codebook 规模 8,196（实现词表 8,320 含额外 token），最后由 HiFi-GAN vocoder 还原波形。

---

## 3. Attention 与位置编码

### 3.1 Encoder/decoder mask

- encoder 内部：同一示例的条件 token 双向 self-attention；动态 packing 后用 block mask 禁止不同样本互看。
- decoder：标准 causal self-attention，并 cross-attend encoder 输出。
- 图像/音频长目标：额外使用 row、column、convolution-shaped masked sparse attention，降低 $1024$ 图像 token 的代价，同时保留二维局部结构。

### 3.2 2D RoPE

文本使用标准 1D RoPE。二维模态 token 的每个 attention head 将 query/key 通道一分为二，对坐标 $(i,j)$ 分别旋转：

$$
q'_{i,j}=R_i q^{(h)}\oplus R_j q^{(w)},\qquad
k'_{i,j}=R_i k^{(h)}\oplus R_j k^{(w)}.
$$

图像/声谱图使用 $(h,w)$；历史模态使用 $(t,l)$，其中 $t$ 是帧/音频段索引，$l$ 是 Perceiver latent 索引。模态身份另由可学习 segment embedding 表示。

### 3.3 稳定性改造

多模态混合会让 attention logits 极端化并导致梯度爆炸。论文组合使用：

1. query/key 进入点积前 LayerNorm（QK-Norm）；
2. Perceiver 内使用 scaled cosine attention；
3. attention logits 用 float32；
4. 预训练冻结 ViT/AST，instruction tuning 末期再解冻；
5. global gradient norm clip = 1.0。

这些改动不是附属工程细节，而是 6.8B 模型在图像、文本、视频、音频混合下能否训练的必要条件。

---

## 4. 训练目标：自回归输出 + Multimodal Mixture of Denoisers

### 4.1 文本目标

沿用 UL2 的三种 denoiser：

- `[R]`：regular span corruption；
- `[S]`：causal language modeling；
- `[X]`：extreme span corruption。

decoder 的基本损失仍是离散 token 交叉熵：

$$
\mathcal L_{\mathrm{AR}}=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},E(x)).
$$

### 4.2 图像/音频目标

图像与音频使用两类对应目标：

- `[R]` masked denoising：随机遮住 $x\%$ 输入 patch feature，重建对应离散目标；
- `[S]` conditional generation：目标模态完全不进入条件，只根据其他模态自回归生成。

它们**不是 DDPM/连续扩散**，也不是 MaskGIT 式并行迭代解码；目标始终是离散 VQ token 的 next-token cross-entropy。

### 4.3 Dynamic Masking：避免 decoder 泄漏

![](assets/Unified-IO%202%20-%20多模态输入输出统一建模/figures/03-dynamic-masking.png)

> **Figure 5，PDF p.5。** (a) 普通 AR 会从 decoder 前缀看到本应被 encoder mask 的 token；(b) 全部 mask 虽无泄漏，却干扰生成；(c) dynamic masking 仅在预测当前位置时暴露对应 decoder 输入，兼顾因果生成与去噪。来源：官方 PDF 完整图，已 Read 验证。

若 encoder 将位置 1 mask，但 decoder 前缀仍包含 token 1，那么预测后续位置时可绕过去噪任务。作者让 decoder 中被 mask 的 token 保持遮挡，只在预测该 token 的对齐位置临时提供“current”输入，降低泄漏又不破坏 causal generation。论文报告直接混合 50% MAE 与 50% causal modeling 会显著损害图像生成，dynamic masking 是二者共存的关键。

---

## 5. 训练数据、课程与系统实现

### 5.1 预训练混合

| 类别 | 采样占比 | 代表来源/目标 |
|---|---:|---|
| NLP | 33% | RedPajama、C4、Wikipedia、Stack/Markdown；文本 denoising 与 LM |
| Image & Text | 40% | LAION-400M/Aesthetic、CC3M/12M、RedCaps、OBELICS；caption、T2I、交错文档 |
| Video & Audio | 25% | YT-Temporal-1B、ACAV100M、AudioSet、WebVid、HD-VILA、Ego4D |
| 3D & Embodiment | 1% | CroCo、Objaverse、ProcTHOR、Habitat；视图/状态/动作预测 |
| Synthetic augmentation | 1% | SAM mask、合成 patch detection/counting |

无监督多模态样本先随机选择目标模态，再随机保留/遮挡其他模态，并采样 `[R]/[S]/[X]` 目标。视频样本可同时产生帧、音频、transcript，使“任意输入到任意输出”成为数据构造过程，而非手工为每个组合写独立模型。

### 5.2 Instruction tuning

- 约 220 个任务、120+ 外部数据集；
- 自然语言约 25%，图像生成 17.6%，音频生成 7.5%，其余覆盖图像理解、稀疏/稠密标注、视频、音频与 embodied AI；
- 总训练 3M steps：预训练 1.5M + instruction tuning 1.5M；
- 附录称预训练约 1T token；正文另指出语言相关总暴露约 250B token，因此语言知识不及专门 LLM。

### 5.3 Dynamic packing

输入最长 1,152、目标最长 2,048。模型先让 ViT/AST/VQ 模块处理未打包数据，再在共享 Transformer 前把不同样本 token 打包，并用 block mask 隔离。启发式池将长样本和短样本配对，encoder/decoder 常用 packing length 为 864/1,280，吞吐接近 4 倍提升。

---

## 6. 采样与解码

### 6.1 文本/结构输出

共享 decoder 自回归生成 BPE、位置或动作 token；结构 token 再反量化为坐标/位姿。检测到的每个框可触发第二次 segmentation 查询，形成检测→mask 的级联。

### 6.2 图像输出

1. 根据 `[Image][S]` 等任务前缀进入图像 token 域；
2. 自回归生成固定 1,024 个 VQ-GAN token；
3. top-$p$ nucleus sampling，$T=1.0,p=0.95$；
4. 训练时 10% 把 prompt 换成 “An image of a random picture.”，推理时做 classifier-free guidance，scale $\alpha=10$；
5. VQ-GAN decoder 还原 $256\times256$ RGB 图像。

CFG 提高 prompt fidelity 与视觉质量，但附录显示也可能降低多样性；它还需要有条件/无条件两路 logits。与扩散不同，这里不能并行更新整块图像，采样步数与图像 token 数同阶。

### 6.3 音频与 dense map

音频同样用 $T=1,p=0.95$ 自回归采样 512 token，但不使用 CFG；超过 4.08 秒时分段生成，并把前段放入 audio history。dense map 用图像 token 路径，通常 $p=0.95$；segmentation 提高到 $p=0.97$。

---

## 7. 结果与消融

### 7.1 统一能力是否以理解性能为代价

- GRIT 总分 67.0，比初代 Unified-IO 64.3 高 2.7；分类、定位、分割、keypoint 提升明显。
- UIO-2XXL：VQAv2 79.4、ScienceQA 88.7、TallyQA 75.9、RefCOCO 90.7、POPE 87.7、MMBench 71.5。
- 结果支持“广泛输出能力没有完全牺牲常规 VLM 性能”，但 OK-VQA 较弱，暴露出纯文本预训练不足。

### 7.2 视觉生成

| 模型 | COCO FID↓ | TIFA↑ |
|---|---:|---:|
| CoDi | 11.26 | 71.6 |
| Emu | 11.66 | 65.5 |
| UIO-2L | 16.68 | 74.3 |
| UIO-2XL | 14.11 | 80.0 |
| UIO-2XXL | 13.39 | 81.3 |

UIO-2XXL 的 TIFA 说明语义遵循强，但 FID 仍弱于专业生成器；这是 fine-grained VQ-GAN 带来的“离散 AR fidelity 较强、感知质量仍受限”的典型权衡。

### 7.3 规模与关键消融

- 1.1B→3.2B→6.8B 时 GRIT ablation 总分 60.8→65.2→66.9，多数任务随规模增长。
- 未加稳定化改造时，加入 video 后 gradient norm 持续上升，XXL 在约 350k steps 后 loss 爆炸；QK-Norm、scaled cosine、fp32 logits 与冻结 encoder 后稳定。
- 普通 masked AR 存在信息泄漏；直接 MAE + causal 混合损害生成；dynamic masking 是关键。
- VQ-GAN 比 ViT-VQGAN 与 MoVQ 的生成效果更好，但论文没有给出充分的逐项量化 tokenizer 消融。

---

## 8. 局限、失败模式与复现

### 8.1 明确局限

1. 图像输出只有 $256^2$，且 1,024 步级别的 AR 解码慢于并行 diffusion/MaskGIT。
2. 图像 fidelity 强但 FID/画质仍不及 Stable Diffusion；VQ codebook 是不可逆瓶颈。
3. 视觉/音频输入只用 base ViT/AST；长视频被压成至多 5 帧，音频单段仅 4.08 秒。
4. depth、multi-object 3D detection、拥挤目标检测较弱；3D/embodied 仅占预训练约 1%。
5. 超参数探索受算力限制；文本暴露量低于专门 LLM，知识型 QA 受限。
6. “统一”仍依赖 ViT、AST、VQ-GAN、ViT-VQGAN、HiFi-GAN、Perceiver 等模态组件，并非原始像素/波形到输出的单体网络。

### 8.2 最小复现路线

1. 先用 1.1B 配置验证 text/image 两模态，固定 VQ-GAN 与 ViT。
2. 实现三种任务前缀和 dynamic masking，先检查 masked token 是否从 decoder prefix 泄漏。
3. 加 QK-Norm、fp32 logits、gradient clip，再逐步引入 video/audio；监控 output norm、attention saturation 与各模态 next-token accuracy。
4. 同时报告理解指标、TIFA/FID、每图 1,024-token 延迟，不能只报 VQA。
5. 复现 dense 输出时按 detection→per-box mask 的真实推理链评估，不能把单次 oracle box segmentation 当完整实例分割。

---

## 9. 与另外三条统一路线的关系

| 模型 | 主干 | 图像表示 | 图像目标 | 图像块 attention | 生成特点 |
|---|---|---|---|---|---|
| Unified-IO 2 | encoder-decoder | 输入 ViT；输出离散 VQ-GAN | AR cross-entropy + masked denoising | decoder causal + 2D sparse mask | 1,024 token 顺序生成，多输出模态 |
| Chameleon | decoder-only | 离散 VQ tokenizer | 纯 AR NTP | 全序列 causal | 最纯粹 early fusion，固定图像块顺序生成 |
| Transfusion | decoder-only | 连续 VAE latent patch | DDPM noise prediction | 图像块内双向 | 连续扩散，无图像词表 |
| Show-o | decoder-only | 离散 MAGVIT-v2 token | MaskGIT/MTP | 图像块 full / Omni-Attention | 25–50 轮并行去 mask |

Unified-IO 2 的优势是输出类型最广、任务序列化最完整；代价是组件多、分辨率低、视觉生成仍受长 AR 序列约束。

---

## 10. 材料路径与自检

| 检查项 | 路径/结论 | 状态 |
|---|---|---|
| 笔记 | `10-Papers/VLM/Unified-IO 2 - 多模态输入输出统一建模.md` | 已完成 |
| 官方 PDF | `assets/Unified-IO 2 - 多模态输入输出统一建模/sources/paper.pdf` | 已保存 |
| 来源记录 | `assets/Unified-IO 2 - 多模态输入输出统一建模/sources.md` | 已完成 |
| 完整原图 | `figures/01-*.png`、`02-*.png`、`03-*.png` | 3 张，已逐张 Read 验证 |
| 编号标题 | 0–10 连续 | 通过 |
| 公式 | AR、2D RoPE、图像 token 数 | 通过 |
| 视觉输出架构 | VQ-GAN、词表、mask、采样与 dense output 均覆盖 | 通过 |
| Paper-Index | 未写入、未修改 | 通过 |
