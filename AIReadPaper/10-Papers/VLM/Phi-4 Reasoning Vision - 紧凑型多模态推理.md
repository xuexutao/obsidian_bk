---
type: paper-note
status: done
domain: VLM
paper: "Phi-4-reasoning-vision-15B Technical Report"
year: 2026
arxiv: "2603.03975"
doi: "10.48550/arXiv.2603.03975"
source: "https://arxiv.org/abs/2603.03975"
project: "https://github.com/microsoft/Phi-4-reasoning-vision-15B"
code: "https://github.com/microsoft/Phi-4-reasoning-vision-15B"
tags:
  - VLM
  - Multimodal-Reasoning
  - Dynamic-Resolution
  - Mixed-Reasoning
  - Computer-Use
  - Data-Curation
created: 2026-10-16
updated: 2026-10-16
---

# Phi-4 Reasoning Vision：紧凑型多模态推理

## 基本信息

- **作者：** Jyoti Aneja、Michael Harrison、Neel Joshi、Tyler LaBonte、John Langford、Eduardo Salinas
- **机构：** Microsoft Research
- **规模：** 15B open-weight multimodal reasoning model
- **本地原文：** [paper.pdf](assets/Phi-4%20Reasoning%20Vision%20-%20紧凑型多模态推理/sources/paper.pdf)
- **完整溯源：** [sources.md](assets/Phi-4%20Reasoning%20Vision%20-%20紧凑型多模态推理/sources.md)

**一句话结论：**

Phi-4-reasoning-vision-15B 采用 SigLIP-2 NaFlex 动态分辨率视觉编码器 + MLP + Phi-4-Reasoning 的中期融合，以 200B 多模态训练 token 和严格数据清洗构建紧凑模型；训练中用 `<think>` / `<nothink>` 混合约 20% reasoning 与 80% direct-response 数据，让同一模型在数学/科学任务上长思考、在 OCR/grounding/caption 上直接回答。其最强论点不是绝对分数第一，而是在 H100 单请求时延和输出 token 上形成较好的 accuracy–cost Pareto 前沿。

---

## 1. 研究问题与核心贡献

### 1.1 紧凑推理的目标

大模型视觉推理常同时扩大参数、训练 token、视觉分辨率与 inference-time chain-of-thought，造成交互延迟和成本。Phi-4 Reasoning Vision（下称 Phi-4-RV）试图回答：能否复用强语言推理底座，用较少多模态数据和更短输出，仍覆盖通用 VQA、数学科学、OCR、GUI grounding 与 computer use？

![](assets/Phi-4%20Reasoning%20Vision%20-%20紧凑型多模态推理/figures/timing_and_tokens.png)

> **图 1：准确率—计算时间/输出 token Pareto 图。** Phi-4-RV 的平均准确率不是最高，但位于左侧低时延、低 token 区域；比较基于 ChartQA、MathVista、MMMU、ScreenSpot 各抽 100 题。来源：论文 Figure 2；官方 arXiv source 原图，已逐图复核。

### 1.2 核心贡献

1. 选择 **mid-fusion** 而非 early fusion，以复用 SigLIP-2 与 Phi-4-Reasoning 的预训练能力。
2. 系统比较 Dynamic-S²、multi-crop、multi-crop+S² 与 SigLIP-2 NaFlex 动态分辨率。
3. 三阶段课程：MLP 对齐→全模型单图 instruction tuning→长上下文/多图/RAI。
4. 混合 reasoning/non-reasoning SFT，通过显式 mode token 控制推理长度。
5. 把数据过滤、答案修正、合成扩增作为主要性能杠杆，并给出数据比例消融。
6. 在统一评测与单请求 H100 时延下比较默认、强制 think、强制 nothink 及同类开源模型。

---

## 2. 架构：动态分辨率中期融合

![](assets/Phi-4%20Reasoning%20Vision%20-%20紧凑型多模态推理/figures/arch.png)

> **图 2：SigLIP-2—MLP—Phi-4-Reasoning 中期融合。** 视觉特征投影成 soft tokens，与文本 token 交错后进入语言模型。来源：论文 Figure 3；官方 arXiv source 原图，已逐图复核。

### 2.1 为什么选择 mid-fusion

- **Early fusion：** 从底层用同一 transformer 联合处理所有 patch 与文本，跨模态交互最自由，但训练、显存和数据成本高。
- **Mid fusion：** 先由预训练视觉 encoder 把图像变为视觉 token，再投影到预训练 LLM embedding 空间。它牺牲部分底层联合建模，换取组件复用与较低训练成本。

Phi-4-RV 的目标是紧凑和高效，因此采用 mid-fusion。

### 2.2 Shape 与信息流

对动态 patch 数 $N_p$ 的图像：

$$
X_{img}\in\mathbb{R}^{B\times H\times W\times3}
\xrightarrow{\text{SigLIP-2 NaFlex}}
H_v\in\mathbb{R}^{B\times N_p\times d_v}.
$$

跨模态 MLP 投影为 LLM hidden size $D$：

$$
Z_v=\operatorname{MLP}(H_v)\in\mathbb{R}^{B\times N_p\times D}.
$$

视觉 soft tokens 与文本 embedding 交错：

$$
H_0=[E(T_1);Z_{v,1};E(T_2);Z_{v,2};\ldots]
\in\mathbb{R}^{B\times L\times D},
$$

再由 Phi-4-Reasoning decoder 自回归生成。报告没有给出 MLP 层数/激活、SigLIP-2 精确 variant hidden size、最终最小 patch 数与完整 image token delimiter 模板；复现这些 shape 必须查官方 config。

### 2.3 动态分辨率策略消融

作者用 5B 小模型、10M 以 GUI/computer-use 为主的图文对比较：

| 方法 | Max visual tokens | MathVista | ScreenSpot | ScreenSpot-Pro | V*Bench |
|---|---:|---:|---:|---:|---:|
| Dynamic-S² | 3096 | 42.9 | 78.4 | 9.4 | 52.9 |
| Multi-crop | 3096 | 43.4 | 67.8 | 5.4 | 51.8 |
| Multi-crop + S² | 2048 | 43.4 | 79.1 | 10.6 | **57.1** |
| Dynamic resolution | 2048 | **45.2** | **81.5** | 9.2 | 51.3 |
| Dynamic resolution | 3600 | 44.9 | 79.7 | **17.5** | 56.0 |

3600 tokens 近似原生 720p，显著提升 ScreenSpot-Pro；2048 tokens 在 MathVista/ScreenSpot 更好，说明更高分辨率并非所有任务单调受益。最终报告采用动态分辨率方向，但未把最终 15B 的所有 min/max patch 配置完整列出。

### 2.4 Attention、mask 与 position

- 视觉 soft tokens 与文本进入同一 Phi-4-Reasoning decoder，生成 token 使用标准 causal attention。
- 动态图像的二维位置主要由 SigLIP-2 NaFlex 视觉 encoder 处理；报告未给出投影后视觉 token 在 LLM 中的 position-id 细节。
- TeX source 中有一整段 **bidirectional attention 实验被注释掉**；它不是最终报告正文中的已采用机制。不能把“intra-image bidirectional attention”写成发布模型已确认配置。
- 因而公开证据支持的最保守结论是：动态视觉特征经 MLP 交错进入标准自回归 LLM；更精细的 prompt mask/position 以官方实现为准。

---

## 3. 三阶段训练与冻结策略

### 3.1 训练配方

| 阶段 | MLP | Vision encoder | LLM | 样本 / token | max seq |
|---|---:|---:|---:|---:|---:|
| 1. MLP pretraining | 训练 | 冻结 | 冻结 | 2.0M / 1.4B | 2048 |
| 2. Instruction tuning | 训练 | 训练 | 训练 | 62.8M / 188.5B | 8192 |
| 3. Long context & RAI | 训练 | 训练 | 训练 | 3.2M / 12B | 16384 |

三阶段均 1 epoch、AdamW、bf16、DeepSpeed ZeRO-1。学习率分别为 $10^{-3}$、$2\times10^{-5}$、$7\times10^{-7}$；global batch 1024/1920/1920。

### 3.2 阶段逻辑

1. **Stage 1：** 只训练随机初始化 MLP，使 SigLIP-2 与 Phi-4-Reasoning embedding 初步对齐。作者扩大该阶段数据未见收益。
2. **Stage 2：** 全模型解冻，是主要训练阶段；覆盖单图 VQA、数学科学、grounding、caption、OCR、computer-use，并混合 think/nothink。
3. **Stage 3：** 全模型继续训练长文档、多图、连续图像与 RAI，max sequence 扩到 16K。

总多模态训练约 200B token；语言底座 Phi-4 本身来自 400B unique tokens，Phi-4-Reasoning 再训练 16B。论文将这一规模与 Qwen3-VL/Kimi-VL/Gemma3 的 1T+ 多模态 token 对比，但不同论文 token 口径与数据质量并不完全可比。

---

## 4. 数据：质量优先于无差别扩量

![](assets/Phi-4%20Reasoning%20Vision%20-%20紧凑型多模态推理/figures/phi4mm_data.png)

> **图 3：Stage 2 数据组成、mixed reasoning 样例与 GPT-4o caption 重写。** VQA 20.37%、Math & Science 18.66%、Grounding 18.24%、Web 15.19%、Text 11.04% 为主要部分。来源：论文 Figure 4；官方 arXiv source 原图，已逐图复核。

### 4.1 清洗分级

作者人工抽看每个公开数据集 5–10 分钟，将其分为：优秀、好问题/错答案、低质问题、低质图像、高质量但格式错误。对应处理为：

- 优秀数据保留并做轻量格式修复；
- 错答案/差 caption 用 GPT-4o 或 o4-mini 重生成，并用验证/多数投票；
- 低质问题若无法可靠修复则删除，但可把高质量图像重新生成 caption/simple VQA；
- 图像本身有根本缺陷的数据集删除；
- 统一修复 image tag、answer format、reasoning/final answer 边界等逻辑错误。

### 4.2 合成扩增

- 数学/科学/逻辑图像同时生成 detailed visual description，帮助语言推理底座先看懂图。
- 用同一 QA 设计不同答案格式，兼做 instruction following。
- 约 5 张图构造乱序 caption 与 image-caption matching，训练正确图像选择。
- 连续 screenshot 生成 “what changed?” 数据，服务 CUA/robotics。
- Grounding 坐标统一归一化到 $[0,1]$。

### 4.3 数据比例消融

5B 小模型固定 1M 通用图文，改变 math 与 CUA：

| General | Math | CUA | MMMU-CoT | MathVista | ScreenSpot-V2 |
|---:|---:|---:|---:|---:|---:|
| 1M | 150K | 450K | 44.0 | 37.4 | 48.2 |
| 1M | 150K | 850K | 44.1 | 37.3 | 60.0 |
| 1M | 450K | 450K | 45.3 | 36.0 | 48.3 |
| 1M | 450K | 850K | 43.4 | **38.9** | **63.1** |

在该小规模未饱和区间，增加 CUA 没有明显伤害数学，3× math 也可能改善 CUA；但总数据量高度相关，尚不能证明跨域“正迁移”会在大规模持续。

---

## 5. 双思考模式：何时 think，何时 nothink

### 5.1 为什么纯 reasoning 模型不适合所有视觉任务

OCR、caption、grounding、简单 VQA 通常不需要长 CoT；强迫推理会增加延迟，甚至诱发对清晰视觉证据的过度解释。数学/科学问题则受益于多步推导。单一固定模式会在准确率与成本之间失衡。

### 5.2 混合 SFT

- reasoning 样本：含 `<think>...</think>`，约占 20%，主要为数学/科学。
- non-reasoning 样本：以 `<nothink>` 开始，约占 80%，主要为 caption、grounding、OCR、简单 VQA。
- 默认模式由训练分布隐式学习；用户可在 prompt 中显式加入 `<think>` 或 `<nothink>` 覆盖。

对输入 $x$ 与模式 $m$，仍是标准自回归目标：

$$
\mathcal L=-\sum_t\log p_\theta(y_t\mid y_{<t},x,m).
$$

不同之处不是 decoder 结构，而是输出序列协议与数据 mixture。模型基于 Phi-4-Reasoning 起步，因此 20% 数据主要把已有语言推理能力**ground 到视觉证据**，而不是从零教 reasoning。

### 5.3 默认模式与强制模式

论文在 AI2D、ChartQA、HallusionBench、MathVerse、MathVision、MathVista、MMMU、MMStar、OCRBench、ScreenSpot 上比较：

- MathVerse：默认 44.9，force think 53.1，force nothink 43.8。
- MMMU：默认 54.3，force think 55.0，force nothink 52.0。
- ScreenSpot：默认 88.2，force nothink 88.3，force think 88.1。
- OCRBench：默认 76.0，force nothink 75.6，force think 73.7。

平均看默认混合路由优于强制单一模式，但边界并不可靠；作者明确承认会出现该思考时直接答、不该思考时冗长推理。

---

## 6. 结果、效率与消融

### 6.1 与 non-thinking 开源模型比较

默认 Phi-4-RV 在 AI2D 84.8、ChartQA 83.3、MathVista 75.2、ScreenSpot 88.2，优于旧 Phi-4-multimodal；但在 HallusionBench、MMMU、OCRBench 等低于 Qwen3-VL 8B/32B。结论是**同规模与低成本竞争力强，而非全面 SOTA**。

### 6.2 与 thinking 模型比较

force-think 在 MathVerse 53.1、MMMU 55.0，低于 Qwen3-VL Thinking 与 Kimi-VL Thinking；但其最大输出 4096、实际 completion tokens 和时延明显更低。论文定位是 accuracy–cost Pareto，而不是用更长 test-time compute 追最高分。

### 6.3 计时方法

- 四个 benchmark 各随机 100 题。
- NVIDIA H100，single thread、无并发、batch=1。
- 使用 Eureka ML Insights；温度 0、greedy decoding、max output 4096。
- 作者没有直接引用 leaderboard，而是统一自行重跑，并计划发布 logs。

这提高了可比性，但 4 个 benchmark 的平均会掩盖任务差异；H100 单请求 latency 也不等价于消费级部署或高吞吐服务。

### 6.4 高分辨率的成本

3600 visual tokens 对 ScreenSpot-Pro 有显著增益，但 LLM 序列 attention 与 KV cache 成本随上下文增长。论文指出当前视觉 tokenizer 不依赖文本问题，背景与关键区域同等编码；未来应研究 question-conditioned tokenization，在保留关键区域的同时压缩无关背景。

---

## 7. 能力示例与安全边界

![](assets/Phi-4%20Reasoning%20Vision%20-%20紧凑型多模态推理/figures/everyday.png)

> **图 4：旅行 caption、票据计算与洗涤符号解释。** 展示通用任务覆盖；个案不能替代 OCR/算术正确率统计。来源：论文 Figure 1；官方 arXiv source 原图，已逐图复核。

Stage 3 加入 Hateful Memes、VLGuard、Think-in-Safety、WildGuard 等 RAI 数据。自动安全评测报告 Text→Text defect rate 1.4%、Image→Text 4.5%。视觉输入安全缺陷率更高，说明图像可带来额外绕过与误解风险；论文没有给出各风险类别样本数与置信区间。

---

## 8. 局限与批判性思考

1. **动态分辨率精度换成本。** 3600 visual tokens 对 GUI 有益，却提高 prefill、KV cache 与 attention 成本；报告缺少逐分辨率显存/时延表。
2. **路由不稳定。** think/nothink 边界由数据隐式学习，20/80 比例未做系统 sweep；默认优于强制模式只在当前数据与 benchmark 上成立。
3. **CoT 监督风险。** reasoning trace 质量、视觉 groundedness 与是否含不可见 shortcut 未单独评估；最终答案正确不代表推理忠实。
4. **数据不可完全复现。** 虽以公开数据为主，但还含 Microsoft 内部数据、定向采购与 GPT-4o/o4-mini 重写；完整训练集与过滤器未全部发布。
5. **消融多用 5B proxy。** 分辨率与数据比例结论来自小模型/较小数据，不能保证线性外推到最终 15B。
6. **架构细节缺口。** MLP 结构、最终 NaFlex min/max patches、视觉 token delimiters、LLM position ids 与完整 mask 未披露。
7. **并非全面最优。** Qwen3-VL/Kimi-VL 在多项 reasoning 与 OCR 上更高；Phi-4-RV 的优势主要是 token 与时延效率。
8. **“modest hardware”证据有限。** 论文正式计时使用 H100，未给出手机/笔记本的真实吞吐、量化与内存结果。

---

## 9. 复现清单

1. 从官方 config 锁定 SigLIP-2 NaFlex variant、patch size、min/max patches、MLP 层数和 LLM hidden size。
2. 复现 2048/3600 dynamic resolution、Dynamic-S²、multi-crop、multi-crop+S²；同时报告平均实际 token，不只报上限。
3. Stage 1 严格只训练 MLP；Stage 2/3 全量解冻，并保存每阶段样本、token、max sequence 与 optimizer state。
4. 为 `<think>` 与 `<nothink>` 建独立数据统计：任务分布、平均长度、答案正确率、视觉证据引用率。
5. 做 reasoning ratio sweep（0/10/20/40/100%），在固定总 token 下测准确率、输出长度、时延与错误路由率。
6. 数据清洗应保留原始样本、错误类型、修正模型、验证投票与删除原因，避免不可审计的“高质量”标签。
7. 固定 100×4 timing 子集、H100、batch=1、greedy、4096，并额外报告 TTFT、decode tok/s、P50/P95、peak memory。
8. 明确 attention mask 与 position ids；不要把 TeX 中被注释的 bidirectional attention 段落误当发布模型事实。
9. 在公开数据-only 与含内部/采购数据两套配方上分别训练，量化不可公开数据的真实贡献。

---

## 10. 个人判断

Phi-4-RV 的关键经验是：紧凑推理不是“把大模型缩小”，而是把有限预算优先分给高分辨率感知、数据修正和需要 CoT 的任务。最有说服力的部分是训练 token 明细、三阶段冻结表、统一 H100 计时与强制模式对照；最薄弱的是架构实现细节、5B proxy 到 15B 的外推，以及不可完全公开的数据管线。双模式本质上是数据驱动的输出协议，而非专家路由器；它简单、有效，也因此容易误路由。总体上，这是一篇偏工程配方、证据较透明的紧凑多模态推理报告，价值主要在 accuracy–latency–token 三者的联合优化思路。

---

## 11. 图表索引与材料来源

1. 图 1：准确率—计算时间/输出 token Pareto。
2. 图 2：SigLIP-2 + MLP + Phi-4-Reasoning 中期融合。
3. 图 3：数据组成、mixed reasoning 与 caption 重写。
4. 图 4：通用视觉语言能力示例。

官方 PDF、arXiv source、代码与图片逐项来源见 [sources.md](assets/Phi-4%20Reasoning%20Vision%20-%20紧凑型多模态推理/sources.md)。