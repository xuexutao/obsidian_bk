---
type: paper-note
status: done
domain: VLM
paper: "mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models"
year: 2024
arxiv: "2408.04840"
doi: "10.48550/arXiv.2408.04840"
source: "https://arxiv.org/abs/2408.04840"
project: "https://github.com/X-PLUG/mPLUG-Owl"
code: "https://github.com/X-PLUG/mPLUG-Owl"
tags:
  - VLM
  - Long-Image-Sequence
  - Hyper-Attention
  - Cross-Attention
  - MI-RoPE
  - Video
created: 2026-10-16
updated: 2026-10-16
---

# mPLUG-Owl3：长图像序列稀疏超注意力

## 基本信息

- **作者：** Jiabo Ye、Haiyang Xu、Haowei Liu 等
- **骨干：** SigLIP-400M vision encoder + linear projector + Qwen2 decoder-only LLM
- **本地原文：** [paper.pdf](assets/mPLUG-Owl3%20-%20长图像序列稀疏超注意力/sources/paper.pdf)
- **完整溯源：** [sources.md](assets/mPLUG-Owl3%20-%20长图像序列稀疏超注意力/sources.md)

**一句话结论：**

mPLUG-Owl3 不再把成百上千张图的 patch token 直接拼进 LLM 上下文，而是维护**独立视觉序列**，文本序列只保留 `<|image|>` 占位符；仅在少数 LLM 层用 Hyper Attention 并行执行 text self-attention 与 text→vision cross-attention，再用 MI-RoPE、因果跨模态 mask 和文本条件门控融合。它以 4 个稀疏 HATB 换取更低首 token 延迟、更强多图/长视频抗干扰能力。

---

## 1. 问题定义与贡献

### 1.1 为什么直接拼接不适合长视觉序列

单图模型通常把视觉 token 与文本 token 拼成一个长序列。若有 $N$ 张图、每图 $P$ 个 patch，则 self-attention 长度约为 $L_T+NP$，视觉 token 不仅占满上下文，还使 attention 计算与 KV cache 急剧增长。更关键的是，大量无关图像可扰动文本隐藏状态，使模型在“视觉大海捞针”场景中失焦。

![](assets/mPLUG-Owl3%20-%20长图像序列稀疏超注意力/figures/01-capabilities.png)

> **图 1：效率、短中长多图与两小时视频示例。** 左上显示首 token 延迟随图像数增长仍较缓；其余子图展示检索增强、多轮多图和长视频。示例只能说明能力形态，定量结论应以表格与 Distractor Resistance 为准。来源：论文 Figure 1；官方 TeX 原图，已逐图复核。

### 1.2 核心贡献

1. **视觉/文本序列解耦：** 视觉 patch 留在外部 memory，LLM 上下文仅保留文本与 image placeholders。
2. **稀疏 Hyper Attention Transformer Block（HATB）：** 只改少数层，并行 self-attention 与 cross-attention。
3. **MI-RoPE + 因果跨模态 mask：** 让每张图携带其在交错文本中的位置，并禁止文本访问其右侧尚未出现的图像。
4. **Adaptive Gating：** 每个文本位置根据自身语义决定需要多少视觉补充。
5. **Distractor Resistance：** 用 1–400 张图测视觉干扰，而不只测 needle 是否可检索。

---

## 2. 总体计算图与 shape

![](assets/mPLUG-Owl3%20-%20长图像序列稀疏超注意力/figures/02-architecture.png)

> **图 2：独立视觉序列与稀疏 HATB。** 单图、多图、视频都编码为外部 patch 序列；文本侧用占位符保留原始交错顺序，少量橙色 HATB 读取视觉 memory。来源：论文 Figure 2；官方 TeX 原图，已逐图复核。

给定交错输入

$$
S=[T_1,I_1,T_2,I_2,T_3],
$$

视觉编码器与线性层产生

$$
H_{img}=[I_1^t;I_2^t]\in\mathbb{R}^{L_V\times D},
$$

文本序列则变成

$$
S_{text}=[T_1,T_{img},T_2,T_{img},T_3],\qquad
H_{text}\in\mathbb{R}^{L_T\times D}.
$$

这里 $T_{img}$ 是普通占位 token `<|image|>`。**视觉 patch 不计入语言上下文长度 $L_T$**，但仍会在 HATB 的 cross-attention 中产生 $O(L_TL_V)$ 计算与视觉 K/V memory；所以 Owl3 不是“视觉免费”，而是把全序列 $O((L_T+L_V)^2)$ 改成文本 self-attention 与稀疏 cross-attention。

默认 vision encoder 为 SigLIP-400M，分辨率 384×384；高分辨率单图可裁成最多 6 块并保留全局图。视频默认均匀采样 8 帧，每帧对应一个 `<|image|>`。

---

## 3. Hyper Attention Transformer Block

![](assets/mPLUG-Owl3%20-%20长图像序列稀疏超注意力/figures/03-hyper-attention.png)

> **图 3：Flamingo 串联 cross-attention 与 Owl3 并行 Hyper Attention。** 右侧 mask 清楚显示：每段文本只能读取位于其左侧的图像。来源：论文 Figure 3；官方 TeX 原图，已逐图复核。

### 3.1 稀疏插入

对 $N$ 层 Qwen2，只把 $K$ 个层扩展成 HATB。正文给出的最终层索引为 `[0, 9, 17, 25]`；消融表按 `[1, 9, 17, 25]` 记载，可能是 0/1-based 口径不一致，复现时应以代码为准。四层方案在单图与多图之间最好；两层已有较强表现，八层反而降低零样本多图泛化。

与 Flamingo 在每层串联额外 cross-attention+FFN 不同，HATB 复用原 block 的 Query、LayerNorm 与 FFN，将视觉 cross-attention 与文本 self-attention **并行**，避免显著加深网络。

### 3.2 Shared Input LayerNorm

视觉特征与文本隐藏状态虽然同为 $D$ 维，但分布不同。Owl3 没有像 Owl2 那样使用两套 LN，而是让视觉特征也经过 LLM block 原有 LayerNorm。作者发现这比新建视觉专用 LN 更易收敛，因为输出均值/方差更接近预训练 LLM 的工作分布。

这也是 Owl2→Owl3 的重要变化：Owl2 用**分离 LN**保护模态统计；Owl3 的视觉分支已在独立 cross-attention 中隔离，因此更重视让视觉 K/V 适配已有文本 Query 的分布，选择**共享 LN**。

### 3.3 文本 Query 复用，视觉 K/V 专用

- self-attention 保持原 Q/K/V 与 causal 计算。
- cross-attention 的 Query 直接复用文本 self-attention Query。
- 新增视觉投影

$$
W^{K\&V}_{img}\in\mathbb{R}^{2D\times D},
$$

并用 LLM 原 KV 权重初始化。

概念上可写为

$$
Q_t=LN(H^i_{text})W^Q_{text},
$$

$$
[K_v;V_v]=LN(H_{img})W^{K\&V}_{img},
$$

$$
\bar H^i=\operatorname{Softmax}\left(
\frac{Q_tK_v^\top}{\sqrt{d_h}}+M_{xattn}
\right)V_v.
$$

因此语言 Query 决定“缺什么视觉证据”，视觉专用 KV 保留高密度视觉细节；这是从 [[mPLUG-Owl2 - 模态协作自适应模块]] 继承的思想，但 Owl3 已把视觉 token 移到 LLM 序列之外。

### 3.4 Adaptive Gating

文本 self-attention 输出记为 $\hat H^i$，cross-attention 输出记为 $\bar H^i$。门控为

$$
g=\sigma(W_{gate}^{\top}\hat H^i),
$$

$$
H^i_{fused}=\bar H^i\odot g+\hat H^i\odot(1-g).
$$

$g$ 是由文本语义决定的位置自适应权重，而不是 Flamingo 的全局可学习标量。随后 $H^i_{fused}$ 进入原 FFN。直觉上，纯语言 token 可让 $g$ 接近 0，需要图像证据的 token 才打开视觉通路。

---

## 4. MI-RoPE、mask 与独立视觉位置

### 4.1 MI-RoPE

对第 $n$ 张图 $I_n$，先记录其 placeholder $T_{img,n}$ 在 $S_{text}$ 中的位置 $p_n$。该图的所有 patch 在 cross-attention 中共享 $p_n$ 的 rotary embedding：

$$
\operatorname{MI\mbox{-}RoPE}(I_{n,j})=\operatorname{RoPE}(I_{n,j};p_n).
$$

它编码的是**图像在交错对话中的顺序与文本位置**，不是图内二维 patch 坐标；图内空间由 SigLIP 特征保留。所有 patch 共享同一 interleaved position，也避免一张图因 patch 数量不同而挤压后续文本 position。

### 4.2 Cross-attention causal mask

对文本位置 $t$ 与图像 $n$：

$$
M_{xattn}(t,n)=
\begin{cases}
0,&p_n\le t,\\
-\infty,&p_n>t.
\end{cases}
$$

即文本只能读取左侧已出现的图像。它保留了图文交错的因果语义，也防止训练时后续图像泄漏给先前回答。文本 self-attention 仍使用 Qwen2 标准 causal mask。

### 4.3 代价边界

独立视觉序列节省 LLM self-attention 与文本 KV cache，但 HATB 仍需存储每层视觉 K/V。若有 400 张高分辨率图，$L_V$ 仍很大。论文主要报告首 token 延迟曲线，没有系统给出视觉 K/V cache、图像数×patch 数的显存公式或吞吐分解。

---

## 5. 训练、冻结与数据

### 5.1 三阶段设置

| 阶段 | 数据 | 可训练模块 | steps / batch | 上下文 |
|---|---|---|---|---|
| 1. Pretraining | 41M 图文对 | linear projector、visual KV、adaptive gate | 20K / 2048 | 768 |
| 2. Multi-image training | 交错图文、text-rich、视频 | linear projector、完整 LLM | 3K / 1024 | 4096 |
| 3. SFT | 单图、多图、视频指令 | linear projector、完整 LLM | 11K / 1024 | 4096 |

学习率分别为 $10^{-3}\to10^{-5}$、$2\times10^{-5}\to10^{-7}$、$2\times10^{-5}\to10^{-7}$；三阶段 warmup ratio 0.03，ZeRO-1，后两阶段 TP=4、gradient checkpointing。

### 5.2 冻结逻辑

- Stage 1：vision encoder 与原 LLM 冻结，只学习新接入模块。
- Stage 2/3：论文列出 linear projector 与 full language model 可训练，没有列 vision encoder，因此按披露应视为冻结。
- 这解释了论文在 AI2D/BLINK 等细粒度视觉任务上的短板：作者也明确归因于训练资源受限、未微调视觉 encoder。

### 5.3 多图与视频数据

- Interleaved：MMDU、LLaVA-Interleave；从 LLaVA-Recap 采 3–6 图构造交错/选择性 caption。
- Text-rich：UReader 的阅读与要点生成数据。
- Video：ShareGPTVideo 900K captions + 240K QA、VATEX 中英文 caption；训练固定 8 帧。
- 高分辨率裁图只用于单图任务，多图/视频不裁，避免视觉序列进一步爆炸。

---

## 6. 结果与消融

### 6.1 主要结果

- 单图 VQA：VQAv2 82.1、OK-VQA 60.1、GQA 65.0、VizWizQA 63.5、TextVQA 69.0。
- 长视频：VideoMME w/o subtitles 53.5，LongVideoBench-val 52.1，均为表中最佳。
- 多图：NLVR2 90.8、Mantis-Eval 63.1、MathVerse-mv 65.0、SciVerse-mv 86.2；BLINK 50.3 略低于 LLaVA-Interleave 52.6。

这些结果支持“长序列结构有效”，但并非所有视觉感知任务都领先；冻结视觉 encoder 是明显限制。

### 6.2 Attention 结构对照

| 结构 | GQA | TextVQA | MVBench | VideoMME | NLVR2 | Mantis-Eval |
|---|---:|---:|---:|---:|---:|---:|
| Concatenate | **59.0** | **51.6** | 22.4 | 25.1 | 55.7 | 38.7 |
| Pre-Cross | 53.8 | 45.2 | **43.0** | 38.9 | 55.3 | 44.7 |
| Post-Cross | 48.9 | 40.9 | 38.3 | 37.0 | 54.0 | 47.0 |
| Hyper Attention | 57.6 | 50.0 | 42.8 | **39.4** | **59.5** | **51.6** |

拼接在单图最好，却在视频/多图崩溃；Hyper Attention 保留接近拼接的单图能力，同时显著改善长视觉泛化。

### 6.3 HATB 数量

- 2 层 `[9,27]`：TextVQA 51.3，但多图较弱。
- 8 层：VideoMME 39.5 略高，却使 NLVR2/Mantis 降到 52.4/47.5。
- 4 层：GQA 57.6、NLVR2 59.5、Mantis 51.6，综合最好。

更多 cross-attention 并非更好；稀疏层像受控的信息注入，过密可能破坏预训练文本表示或过拟合有限多图数据。

### 6.4 Gate、共享 LN 与 MI-RoPE

| Gate | Shared LN | MI-RoPE | GQA | TextVQA | VideoMME | NLVR2 | Mantis |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 否 | 否 | 否 | 53.3 | 44.6 | 38.1 | 52.7 | 41.9 |
| 是 | 否 | 否 | 55.7 | 49.3 | 40.1 | 53.4 | 47.9 |
| 是 | 是 | 否 | **58.1** | 49.7 | 38.4 | 54.9 | 46.1 |
| 是 | 是 | 是 | 57.6 | **50.0** | 39.4 | **59.5** | **51.6** |

MI-RoPE 对多图最关键：NLVR2 +4.6、Mantis +5.5；对视频增益有限，说明视频自身内容时序可部分提供隐式顺序，而无时间连续性的多图更依赖显式位置。

### 6.5 Distractor Resistance

![](assets/mPLUG-Owl3%20-%20长图像序列稀疏超注意力/figures/04-distractor-resistance.png)

> **图 4：1–400 张图的抗干扰曲线。** mPLUG-Owl3 从单图约 74% 缓降到 50 图 43.09%、400 图 28.58%；其他模型在 20–100 图快速失效或无法运行。来源：论文 Figure 4；官方 TeX 原图，已逐图复核。

测试从 MMBench 为每题加入 $N-1$ 张随机干扰图，$N\in\{1,5,10,20,50,100,200,400\}$，并使用 CircularEval：不同选项顺序与干扰组合必须全答对才计分。它比普通 multimodal needle 更严格，但干扰图来自同一 benchmark，仍不等价于真实长视频的连续语义与细粒度定位。

---

## 7. Owl2→Owl3 的路线换轨

| 维度 | Owl2 | Owl3 |
|---|---|---|
| 主要问题 | 模态协作与干扰 | 长图像序列与上下文成本 |
| 视觉 token | 拼入 LLM 单序列 | 独立视觉 memory |
| attention | 全层单序列 self-attention | 少数层 text→vision cross-attention |
| 参数共享 | 共享 Q/FFN，分离 LN/K/V | 共享原 Q/LN/FFN，新增视觉 KV |
| position | abstractor 2D/正弦；多图弱 | placeholder 对齐的 MI-RoPE |
| mask | decoder causal，细节未展开 | 文本 causal + 跨模态“只看左侧图” |
| 视觉压缩 | 64 learnable queries | 保留 patch 序列，按需读取 |

因此 Owl3 不是“Owl2 加个长上下文”。它放弃 Owl2 的 64-token bottleneck 与全层 MAM，改用外部视觉 memory + 稀疏读取：**从模态参数解耦换轨为序列与计算解耦。**

---

## 8. 局限与批判性思考

1. 视觉 encoder 冻结，细粒度视觉基准仍受限；TextVQA/BLINK 的表现已体现此边界。
2. 训练多图通常仅 3–8 张，却评到 400 张；泛化令人关注，但 400 图准确率仅 28.58%，不能写成“解决超长视觉理解”。
3. 只报告首 token 延迟趋势，缺少总 decode 时延、视觉 K/V 显存、吞吐与固定硬件完整表。
4. MI-RoPE 让同图所有 patch 共享交错位置，图内二维位置完全依赖视觉 encoder；跨图精确空间/时间关系可能不足。
5. 视频默认 8 帧，长视频成绩可能依赖粗粒度语义而非事件级时间定位。
6. 最终 HATB 层索引在正文 `[0,9,17,25]` 与消融 `[1,9,17,25]` 存在口径差异。
7. Adaptive Gate 没有报告分布、稀疏度或是否饱和，难以确认“按需看图”的机制是否如直觉工作。

---

## 9. 复现清单

1. 分别维护 `H_text: [B,L_T,D]` 与 `H_img: [B,L_V,D]`；文本只插入 `<|image|>`。
2. 在 4 个均匀层插入 HATB；同时复现实验的 2/4/8 层配置。
3. 复用原 Query 与原 LayerNorm；新增视觉 KV，并从原 LLM KV 初始化。
4. 为每图保存 placeholder position；所有 patch 应用同一 MI-RoPE interleaved id。
5. 构造 cross mask：文本位置只能访问左侧图像；文本 self-attention 保持 causal。
6. 记录 gate 的均值、分位数及按任务/层/位置分布，确认没有退化为常数。
7. Stage 1 仅训练 projector、视觉 KV、gate；Stage 2/3 训练 projector+LLM，保持 vision encoder 冻结。
8. 除公开基准外，复现 1–400 图 Distractor Resistance，并报告硬件、视觉 token 数、prefill、峰值显存和总 latency。

---

## 10. 个人判断

mPLUG-Owl3 的价值在于把“视觉 token 是否压缩”改写成“视觉 token 是否必须常驻语言序列”。独立视觉 memory、稀疏 cross-attention、placeholder 对齐 position 和文本条件 gate 形成了完整闭环。其消融清楚表明：gate 改善单图/视频，MI-RoPE 主要拯救多图顺序，4 层注入比 8 层更稳。局限是视觉编码器冻结、视频帧数低、效率报告不完整；但作为 Owl2 之后的架构节点，它明确代表了一次从**模态适配**到**长视觉 memory 检索**的换轨。

---

## 11. 图表索引与材料来源

1. 图 1：效率与不同尺度多图/视频能力。
2. 图 2：独立视觉序列与稀疏 HATB 总体架构。
3. 图 3：Hyper Attention、Adaptive Gate 与 cross mask。
4. 图 4：Distractor Resistance 曲线。

官方 PDF、arXiv source、代码与图片逐项来源见 [sources.md](assets/mPLUG-Owl3%20-%20长图像序列稀疏超注意力/sources.md)。