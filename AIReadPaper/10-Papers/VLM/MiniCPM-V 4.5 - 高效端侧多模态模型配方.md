---
type: paper-note
status: done
domain: VLM
paper: "MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe"
year: 2025
arxiv: "2509.18154"
doi: "10.48550/arXiv.2509.18154"
source: "https://arxiv.org/abs/2509.18154"
project: "https://github.com/OpenBMB/MiniCPM-V"
code: "https://github.com/OpenBMB/MiniCPM-V"
tags:
  - VLM
  - Efficient-MLLM
  - Video
  - OCR
  - Hybrid-RL
  - Edge-AI
created: 2026-10-15
updated: 2026-10-15
---

# MiniCPM-V 4.5：高效端侧多模态模型配方

## 基本信息

- **作者：** Tianyu Yu、Zefan Wang、Chongyi Wang 等
- **版本：** arXiv v1，2025-09-16
- **规模：** 摘要称 8B；效率表按 8.7B 统计，论文未解释口径差异
- **本地原文：** [paper.pdf](assets/MiniCPM-V%204.5%20-%20高效端侧多模态模型配方/sources/paper.pdf)
- **完整溯源：** [sources.md](assets/MiniCPM-V%204.5%20-%20高效端侧多模态模型配方/sources.md)

**一句话结论：**

MiniCPM-V 4.5 不是单靠缩小参数做“端侧模型”，而是同时压缩视觉 token、降低数据工程成本和控制推理长度：**3D-Resampler** 对图像做空间压缩、对视频做时空联合压缩；动态视觉破坏把文档知识与 OCR 统一为恢复任务；短/长思考混合 RL 以更少训练 token 获得接近或优于纯长思考的能力。它在 8×A100 上展现了显著推理效率，但没有真实手机/NPU/量化/功耗 benchmark，因此“端侧”仍主要是架构潜力而非设备端实证。

---

## 1. 研究背景与问题定义

### 1.1 MLLM 的三类效率瓶颈

高分辨率图像和视频会产生大量视觉 token，直接增加 LLM prefill 成本；科学论文、教材和报告知识密度高，但 PDF 解析容易产生阅读顺序和图文错配，需要繁重数据工程；长思考 RL 能提高复杂推理，却会增加训练与推理 token。

MiniCPM-V 4.5 分别给出三种配方：统一 3D-Resampler、文档/OCR 统一学习、短长思考混合 RL。三者的共同目标是减少进入 LLM 的无效信息，而不是单纯缩小 decoder。

![](assets/MiniCPM-V%204.5%20-%20高效端侧多模态模型配方/figures/01-architecture.png)

> **图 1：MiniCPM-V 4.5 总体架构。** 高分辨率图像先切片，视频先按相邻帧打包；共享视觉 encoder 与 3D-Resampler 把二者压缩成紧凑 token，再由同一 LLM 以短或长思考模式回答。来源：论文 Figure 1。

### 1.2 核心贡献

1. **统一 3D-Resampler：** 图像和视频共享架构/权重；视频 package 内进行时空联合 cross-attention，利用相邻帧冗余。
2. **文档知识与 OCR 统一目标：** 对文本区域施加不同强度破坏，低破坏训练 OCR，中破坏训练视觉+上下文联合推断，高破坏训练文档知识。
3. **Hybrid RL：** 在同一模型中随机交替短思考与长思考 rollout，兼顾直接性和复杂推理。
4. **多阶段训练配方：** 通过冻结/解冻不同模块，让质量不完美的 OCR/caption 数据先提升感知，不直接污染 LLM。
5. **效率实证：** OpenCompass 77.0；Video-MME 73.5，8×A100 上 0.26h、28GB，显著低于 Qwen2.5-VL-7B。

---

## 2. 输入、输出与能力边界

### 2.1 输入

模型支持单图、多图、高分辨率与极端宽高比图像、视频、长视频和纯文本。高分辨率图像采用 LLaVA-UHD 风格切片；视频沿时间轴分成由相邻帧组成的 package。每个视频最多采样 1080 帧，最大 10 FPS。

论文没有披露视觉 encoder 名称、默认切片上限、默认 package size、最大上下文长度，以及原始 PDF 文件是否可直接输入；文档训练对象是页面图像。

### 2.2 输出

输出是自然语言文本，有两种 prompt 控制的模式：短思考快速直接，长思考显式逐步推理。长思考 rollout 最大 8192 token。控制 prompt 的具体模板和切换失败率未披露。

### 2.3 “端侧”的准确理解

架构和 token 压缩确实有利于设备部署，但论文只在 8×A100 上报告效率，没有手机 SoC、CPU/NPU、Apple/Android/Jetson、量化、模型文件大小、首 token 延迟、tokens/s 或功耗。因而本笔记把“端侧”理解为 **面向低资源部署的设计目标**，不把它写成已被真实端侧 benchmark 证明的事实。

---

## 3. 核心机制一：统一 3D-Resampler

### 3.1 从 2D 到 3D

MiniCPM-V 早期 2D-Resampler 使用可学习 query 对视觉特征做 cross-attention，把变长视觉网格压成固定长度 token。4.5 将 query 同时加入二维空间位置和时间位置，允许一个 query 集合对多个相邻帧联合重采样。

论文没有给出正式 cross-attention 公式、层数、head 数、query 默认数量和位置编码形式。可以确认的是：压缩发生在视觉 encoder 之后、LLM 之前；图像与视频共享同一 Resampler 权重。

### 3.2 图像处理

模型根据输入分辨率估算理想切片数，在候选切片方案中选择“每片分辨率最接近视觉 encoder 预训练设置”的布局。每片经视觉 encoder 后，由带 2D 位置的 query 做 cross-attention，得到固定长度序列。论文举例：常见模型对 448×448 使用约 256 token，MiniCPM-V 可用 64 token；Figure 1 声称图像最高 16× 压缩，但未披露对应输入、基线和计算口径。

### 3.3 视频处理

视频沿时间维分为多个 package，每个 package 包含相邻帧。各帧先由视觉 encoder 编码，带 2D 空间+时间位置的 query 对整个 package 联合 cross-attention，输出固定长度序列；所有 package 的序列再拼接。训练时随机增强 package size 和 frame rate，推理时可按设备和任务调节。

论文称时间维额外压缩约 6×，6 个 448×448 帧可压缩为 64 token；6 秒、2 FPS、448×448 的 12 帧视频约为 128 token，而 Qwen2.5-VL/InternVL3 分别约 1536/3072 token。所谓视频 96× 总压缩率的精确基线未完整披露。

### 3.4 知识迁移

图像与视频共享 Resampler 让图像 OCR 和布局知识迁移到视频；作者称未专门收集视频 OCR 数据仍观察到合理能力。这个现象有启发，但没有专门 benchmark 和对照，不能当作严格消融结论。

---

## 4. 核心机制二：文档知识与 OCR 统一学习

### 4.1 统一目标

传统 PDF 管线先解析文本层和图表，再构造图文交错序列；复杂版式中容易出现结构噪声。MiniCPM-V 4.5 直接从文档图像学习，将目标区域原文作为 ground truth，对该区域施加不同程度视觉破坏，并要求恢复原文。论文未写显式损失公式，训练仍可理解为目标文本的自回归预测。

![](assets/MiniCPM-V%204.5%20-%20高效端侧多模态模型配方/figures/02-document-ocr-unified.png)

> **图 2：动态视觉破坏统一三种任务。** 低破坏保留字符证据，训练鲁棒 OCR；中破坏要求融合残余视觉、页面上下文和内部知识；高破坏完全遮住文字，迫使模型根据版式、图表、表格和其他文本推断。来源：论文 Figure 2。

### 4.2 三个破坏区间

1. **Low Corruption / Augmented OCR：** 文本仍可辨认，主要学习识别与轻噪声鲁棒性。
2. **Moderate Corruption / Integrated Inference：** 字符模糊，模型必须融合局部视觉线索、文档上下文和知识。
3. **High Corruption / Contextual Inference：** 文本完全 mask，只能依据其他文字、版式、图表、表格、图片与内部知识恢复。

这种设计把“增强过强会诱导 OCR 幻觉”的问题转化为任务切换：看得清就识别，看不清就允许上下文推断，而不是强迫模型把不可见像素当作可识别字符。

### 4.3 未披露细节

文本区域如何定位、ground truth 如何获得、是否完全不使用 PDF 文本层、三种破坏比例、噪声类型、mask 形状、目标区域数和文档语言比例都未披露。因此“消除外部 parser”更准确地说是避免把 parser 产出的图文交错结构作为主要训练输入，不代表数据准备阶段完全无需文本解析。

---

## 5. 核心机制三：短/长思考混合 RL

### 5.1 SFT 冷启动

SFT 分两步。General SFT 用多模态 instruction-response 激活通用能力，并混入 10% 高质量纯文本，防止语言退化。第二步同时加入 Long-CoT 与 3D-Resampler：Long-CoT 教反思、回溯等模式；少量高质量长视频与高帧率数据把 2D-Resampler 升级为 3D。

### 5.2 Hybrid rollout

RL 采用 GRPO，每批 128 prompts，每个 prompt 采样 8 responses，temperature=1.0，最大响应 8192 token，50% prompt 使用长思考，其余短思考；去掉 KL penalty 和 entropy loss。两种模式共享参数，作者假设长思考提供分析深度，短思考提供直接性和效率。

### 5.3 奖励塑形

最终奖励由准确性、格式、重复惩罚和标准化偏好奖励组成：

$$R=R_{\mathrm{acc}}+R_{\mathrm{format}}+R_{\mathrm{rep}}+\frac{1}{2}\widetilde R_{\mathrm{rm}}$$

同 prompt 的偏好分数做组内标准化：

$$\widetilde R_{\mathrm{rm}}=\frac{R_{\mathrm{rm}}-\bar R_{\mathrm{rm}}}{\sigma(R_{\mathrm{rm}})}$$

简单短答案使用规则验证，论文称 reward accuracy 约 98%；含单位或较长自然语言的复杂答案使用 RLPR 概率型奖励。标准 RM 难以评价分布外长推理链，所以只打分最终 answer，完全绕过显式 thinking。$R_{acc}$、$R_{format}$、$R_{rep}$ 的具体数值与阈值未披露。

### 5.4 RLAIF-V

为减少图像与视频幻觉，模型在相同生成条件下采样多个回答，将长回答拆成原子 claim，独立验证后构造“事实错误更少”的偏好对，再用 DPO 训练。RLAIF-V 配置为 global batch 256、学习率 $10^{-6}$、DPO $\beta=0.1$、400 steps。claim verifier 的具体模型未披露。

---

## 6. 预训练与数据配方

### 6.1 三阶段预训练

1. **Stage 1：** 只训练 2D-Resampler，其余冻结；使用 image-caption 建立低成本跨模态对齐。
2. **Stage 2：** 解冻视觉 encoder，LLM 仍冻结；使用 OCR-rich 与 caption 数据，避免语言质量较差的数据直接改变 decoder。
3. **Stage 3：** 全参数端到端训练；加入纯文本、图文交错、视频和前两阶段精选数据，使用 Warmup-Stable-Decay，并在 decay 阶段增加高质量 instruction 与知识密集数据。

这个冻结策略体现了一个重要工程原则：不完美数据先训练感知和桥接层，只有高质量数据才更新 LLM。

### 6.2 预训练数据

- Caption：LAION-2B、COYO 与中文网络图文对；CLIP 过滤、CapsFusion 风格 recaption、概念标签与语言/长尾平衡。
- Interleaved：Common Crawl、OmniCorpus、MINT-1T；破图、图文比例、相关性和知识密度过滤。
- OCR：自然场景合成文字、不同颜色/字体、HTML 页面渲染。
- Document：科学论文、学术报告、教材等高知识密度文档。
- Video：WebVid、Vript、OpenVid 与额外详细 caption。

总规模、各类比例、许可状态、语言分布和过滤保留率未披露。

### 6.3 SFT 与 RL 数据

SFT 的 STEM 题来自高中及以上教育网站，要求高视觉依赖并做答案一致性检查；长尾知识来自 Wikipedia；Long-CoT 来自 OpenThoughts 与内部管线，使用早期模型筛难题，再做正确性、claim-level RLAIF-V 和重复过滤。

RL 覆盖数学、文档/表格/图表、General Reasoning 和纯文本 Instruction Following，所有子集经过清洗、去重和 human-in-the-loop。具体总量和混合比例未披露。

### 6.4 已披露超参数

| 阶段 | 学习率 |
|---|---:|
| 预训练 stable | $5\times10^{-5}$ |
| 预训练 decay 终点 | $1\times10^{-5}$ |
| General SFT | $1\times10^{-5}\rightarrow1\times10^{-6}$ |
| Long-CoT + 3D warm-up 峰值 | $5\times10^{-6}$ |
| Long-CoT + 3D 终点 | $1\times10^{-6}$ |
| RL | 固定 $1\times10^{-6}$ |

预训练/SFT 总步数、总 token、GPU 数、训练时间、优化器、weight decay、batch 与混合精度未披露。

---

## 7. 主要结果与 baseline

### 7.1 综合能力

| 模型 | OpenCompass |
|---|---:|
| MiniCPM-V 4.5 | **77.0** |
| GLM-4.1V | 76.6 |
| Qwen2.5-VL 72B | 76.1 |
| GPT-4o-latest | 75.4 |
| InternVL3 8B | 73.6 |
| Qwen2.5-VL 7B | 70.5 |

OpenCompass 平均中，MiniCPM 只在 MMStar、MMVet、HallusionBench、MathVista、MMMU 五项使用长思考，其余模式的逐项细节未全部披露；GPT-4o 列也混合 latest 与 2024-11-20 版本，比较不是完全单一版本、同推理预算。

### 7.2 OCR、文档与 STEM

模型在 OCRBench 89.0、OmniDocBench EN 0.175/ZH 0.253 上表现突出；但 TextVQA 82.2 低于 Qwen2.5-VL-7B 的 84.9，DocVQA 94.7 低于 Qwen2.5-VL-72B 的 96.4，ChartQA 87.4 低于后者 89.5。

STEM 中 MMMU 67.7、MathVista 79.9、MathVerse 58.8。MathVerse 明显高于 Qwen2.5-VL-72B 47.3 与 GPT-4o 40.6，但低于 GLM-4.1V 68.4。结果支持“小模型综合竞争力”，不支持每项最优。

### 7.3 视频与 A100 推理效率

| 模型 | Video-MME | 时间 | 显存 |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 71.6 | 3.00h | 60GB |
| GLM-4.1V-Thinking | **73.6** | 2.63h | 32GB |
| MiniCPM-V 4.5 | 73.5 | **0.26h** | **28GB** |

相对 Qwen2.5-VL-7B，时间为 8.7%，显存为 46.7%。这是论文最有力的效率证据，但测试是在 8×A100，且“表中 73.5、正文 73.6”存在 0.1 分不一致。本笔记以表格 73.5 为主。

---

## 8. 消融实验

### 8.1 短、长与混合 RL

| 方法 | OpenCompass | 训练 token |
|---|---:|---:|
| 仅短思考 | 76.0 | **1.6B** |
| 仅长思考 | 77.0 | 4.4B |
| 混合模式 | **77.1** | 3.1B |

混合模式以纯长思考 70.5% 的训练 token 获得略高分数。提升只有 0.1 分，是否稳定跨随机种子未披露；但成本差异明显，支持两种模式共享能力的假设。

### 8.2 概率奖励

![](assets/MiniCPM-V%204.5%20-%20高效端侧多模态模型配方/figures/03-probability-reward-ablation.png)

> **图 4(a)：规则验证 VR 与 VR+概率奖励 PR。** 训练后段 VR+PR 的 OpenCompass 曲线持续高于仅 VR，说明复杂自然语言答案需要比 exact/rule 更密集的学习信号。来源：论文 Figure 4(a)。

![](assets/MiniCPM-V%204.5%20-%20高效端侧多模态模型配方/figures/04-response-length-ablation.png)

> **图 4(b)：响应长度曲线。** 两种奖励下长度均波动，VR+PR 没有简单通过无限拉长输出换分；论文还报告 entropy 子图保持稳定。来源：论文 Figure 4(b)。

### 8.3 文档/OCR 统一学习

使用 1M 高质量样本，其中 20% 是知识密集文档，并保持相同 SFT：

| 方法 | MMMU | AI2D | OCRBench |
|---|---:|---:|---:|
| External Parser | 49.0 | 74.9 | 576 |
| Unified Learning | **51.4** | **76.5** | **617** |

三项同时提升，支持统一目标既改善知识理解又改善 OCR。这里 OCRBench 使用原始积分，主表使用 89.0 的百分制样式，论文未解释两种尺度换算。

### 8.4 3D-Resampler

从 General SFT checkpoint 继续 300 steps，只改变 Resampler：

| 方法 | w/ sub | w/o sub | tokens/frame |
|---|---:|---:|---:|
| 2D-Resampler | 65.5 | 71.5 | 64.0 |
| 3D-Resampler | **67.3** | **72.5** | **21.3** |

每帧 token 约降至三分之一且分数提升。不过表中“有字幕低于无字幕”与主结果相反，可能是 HTML/表头口径问题；本笔记保留原表，不擅自调换。

---

## 9. 局限与未来工作

### 9.1 论文明确或直接可见的边界

视频最多 1080 帧、10 FPS；长思考最多 8192 token；标准 RM 不能可靠评价完整长推理链，只打分最终答案；规则奖励只适合简单答案；强视觉压缩可能遗漏短暂事件、细小文字和快速镜头切换，但论文没有系统失真曲线。

### 9.2 披露与评测缺口

没有真实端侧 benchmark，没有量化、功耗和移动设备时延；没有总训练成本、数据总量与许可分析；没有 benchmark contamination 报告；没有安全、偏见、越狱和低资源语言系统评估；没有短/长模式控制失败率；没有置信区间，仅部分 thinking 结果为三次均值。

### 9.3 未来值得验证的方向

3D-Resampler 应报告 package size、query 数和帧率的速度—精度曲线；统一文档目标应比较不同破坏分布与完全不使用 parser 的真实数据成本；端侧应覆盖 INT4/INT8、手机/PC NPU、首 token、decode 速度、峰值内存与能耗；Hybrid RL 应在固定推理预算下比较用户偏好和任务正确率。

---

## 10. 复现成本与实践建议

从零复现成本无法估算，因为 GPU、训练时长、总 token 与 batch 未披露。最现实的路径是基于官方 checkpoint 和代码做模块验证：

1. 先统计不同图像分辨率、视频帧数与 package size 的视觉 token 数、prefill 时间和显存。
2. 对 3D-Resampler 做 query 数、package size、frame rate、时间位置编码和 2D baseline 消融。
3. 对文档训练保留原始页面、文本区域与破坏参数，分别报告 OCR、知识恢复和幻觉率，避免只看综合分。
4. Hybrid RL 必须记录短/长模式比例、每种模式平均输出长度、准确率、训练 token 和推理成本。
5. 若目标是真端侧，必须加入量化、KV cache 和设备端 benchmark；A100 时间不能代替端侧结论。

---

## 11. 个人判断

MiniCPM-V 4.5 的优势在于三个机制围绕同一个资源目标协同：Resampler 减少输入 token，混合 RL 减少不必要输出 token，动态破坏减少文档解析和数据修复成本。这比单点“8B 小模型”更有工程意义。

3D-Resampler 的证据相对最强：同 checkpoint、300 steps、只变 Resampler，token/frame 降低且 Video-MME 提升。文档/OCR 统一学习也有清晰对照。Hybrid RL 的分数提升较小，但训练 token 节省明显。相对薄弱的是端侧论证：所有公开效率数据仍在 A100 上，模型 8.7B 的权重与 KV cache 对真实移动端并不轻。

我的总体判断是：这是一套值得复用的高效 MLLM 配方，尤其适合视频、文档和 OCR；“端侧”标签应谨慎使用，直到官方给出量化和真实设备数据。研究上最值得跟进的是时空压缩的精度损失边界，以及动态视觉破坏是否会把知识猜测误当成 OCR 正确性。

---

## 12. 材料来源

官方 PDF、arXiv source、代码仓库与四张原文关键图的完整溯源见 [sources.md](assets/MiniCPM-V%204.5%20-%20高效端侧多模态模型配方/sources.md)。本文保留论文中的分数口径不一致，并对未披露训练细节明确标注，未自行补造。