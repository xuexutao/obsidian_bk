---
type: paper-note
status: complete
domain: VLM
paper: "Qwen2.5-VL Technical Report"
year: 2025
arxiv: "2502.13923"
doi: null
source: "https://arxiv.org/abs/2502.13923"
project: null
code: "https://github.com/QwenLM/Qwen2.5-VL"
tags:
  - VLM
  - Qwen-VL
  - document-parsing
  - video-understanding
  - GUI-agent
  - native-resolution
created: 2026-10-15
updated: 2026-10-15
---

# Qwen2.5-VL：文档视频与视觉智能体

## 基本信息

- **作者**：Qwen Team，Alibaba Group；核心贡献者包括 Shuai Bai、Keqin Chen、Xuejing Liu、Jialin Wang、Wenbin Ge 等
- **论文版本**：arXiv:2502.13923v1；报告封面日期 2025-03-05，arXiv 首版日期 2025-02-19
- **模型家族**：Qwen2.5-VL-3B、7B、72B；共享 32 层、hidden size 1280 的视觉编码器，语言侧初始化自对应 Qwen2.5 LLM
- **本地原文**：[paper.pdf](assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/sources/paper.pdf)
- **来源清单**：[sources.md](assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/sources.md)

**一句话结论：**

Qwen2.5-VL 不是简单扩大 Qwen2-VL，而是围绕“高分辨率感知能否高效且可执行”重做视觉侧和数据侧：从零训练更接近 LLM 结构的原生动态分辨率 ViT，以大多数层的 window attention 控制视觉计算；把动态分辨率从空间扩展到动态 FPS，并将 MRoPE 的时间 ID 对齐绝对时间；再以 HTML 统一文档结构、真实像素坐标统一 grounding、跨端动作空间统一 GUI Agent。其文档、视频定位与 GUI 结果很强，但报告没有组件级消融，4.1T token 数据和关键训练资源也未完整公开。

![](assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/figures/01-benchmark-overview.jpg)

> 论文概览图：左侧比较多模态 benchmark，右侧比较纯文本任务。Qwen2.5-VL-72B 的优势集中在 MathVista、DocVQA、InfoVQA 等视觉/文档任务，同时纯文本结果接近 Qwen2.5-72B，说明多模态训练没有明显摧毁语言底座。来源：论文首页概览图，第 1 页；官方 arXiv 源码 `figures/head.jpg`。

---

## 1. 问题定义与前代增量

### 1.1 从 Qwen2-VL 到 Qwen2.5-VL：瓶颈如何转移

Qwen2-VL 已用 Naive Dynamic Resolution、ViT 2D-RoPE、LLM M-RoPE 和深度为 2 的 3D convolution 打通任意宽高图像与长视频。但新的瓶颈随之出现：视觉 token 数随像素增长，而全局 self-attention 的复杂度随 patch 数近似二次增长；视频若以固定 fps 抽帧，慢镜头会重复、快动作会漏帧，M-RoPE 的时间 ID 又只表示“第几帧”，不能区分不同采样率下的真实秒数；OCR 能力也尚未形成完整的版面、图表、公式与结构化输出系统。

Qwen2.5-VL 的目标因此从“能接收不同分辨率”推进到四个更实用的问题：**高分辨率视觉如何降成本、不同 fps 视频如何共享真实时间、复杂文档如何统一解析、视觉定位如何支撑跨设备闭环操作。**

### 1.2 经原文核验的前代增量

| 维度 | Qwen2-VL | Qwen2.5-VL | 作用 |
|---|---|---|---|
| 视觉编码器 | DFN ViT 初始化，动态分辨率 + 2D-RoPE | **从零训练原生动态分辨率 ViT**，RMSNorm + SwiGLU | 视觉结构与 LLM 风格对齐，为新训练配方服务 |
| 注意力 | 视觉全局注意力 | 28/32 层 window attention，4 层 full attention | 让视觉计算近似随 patch 数线性增长 |
| 视频采样 | 主要按 2 fps | **动态 FPS sampling** | 在慢/快视频间更合理分配帧 |
| 时间位置 | M-RoPE temporal ID 随帧序号增长 | temporal ID 对齐**绝对时间** | 同一真实时刻在不同 fps 下获得一致时间语义 |
| 空间坐标 | 归一化到 $[0,1000)$ | 使用输入图像实际尺寸下的绝对坐标 | 保留像素尺度，改善点/框定位与 GUI grounding |
| 文档 | OCR、文档 QA | HTML 统一段落、表格、图表、公式、乐谱、化学式及 bbox | 从“读字”扩为结构化 omni-parsing |
| Agent | 多类视觉 Agent 初步评测 | 手机、网页、桌面的共享 function-call 动作空间与推理轨迹 | 强化真实 GUI grounding 与多步执行 |
| 数据规模 | 报告口径约 1.2T pretraining token | **4.1T trained token**，另有约 2M SFT 条目 | 扩展长文档、长视频、Agent 与推理数据 |

需注意 Qwen2-VL 报告正文写两段预训练累计 1.4T token，而 Qwen2.5-VL 引言用 1.2T 作为前代口径。笔记保留各论文原始表述，不擅自统一这两个统计口径。

---

## 2. 输入输出与统一表示

### 2.1 输入

模型接收纯文本、任意宽高单图、多图、视频、文档页面与 GUI screenshot。图像高宽先调整为 28 的倍数，随后以 stride 14 切 patch；视频将连续两帧作为时间深度为 2 的 3D patch。不同尺寸与不同 FPS 最终映射成不同长度的视觉 token，并与文本共同进入 Qwen2.5 decoder。

### 2.2 输出

输出统一为自回归文本，可包含普通 VQA、数学推理、Markdown/HTML 文档转写、表格与公式、实际像素坐标下的 point/bounding box、视频秒级时间段，以及 GUI Agent 的 reasoning/function call/action parameters。统一目标可抽象为：

$$\mathcal{L}_{\mathrm{NTP}}=-\sum_t\log p_\theta(y_t\mid y_{<t},V_{1:m})$$

其中 $V_{1:m}$ 是一个或多个图像/视频的变长视觉序列。SFT 后还进行 DPO，以偏好对约束回答行为；ViT 在 SFT 与 DPO 中均冻结。

### 2.3 真实像素坐标的含义

Qwen2-VL 把坐标归一化到固定区间；Qwen2.5-VL 则直接使用输入图像尺寸下的实际坐标表示 box 和 point。若输入图宽高为 $(W,H)$，模型输出的点 $(x,y)$ 应满足 $0\le x<W,0\le y<H$。它让模型学习绝对尺度与分辨率之间的关系，也方便 GUI 直接映射到屏幕坐标；但部署端必须确认预处理后的图像尺寸与执行设备坐标系一致，否则 resize、缩放因子或 Retina 像素比都会造成点击偏移。

---

## 3. 核心架构

### 3.1 总体框架

![](assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/figures/02-architecture.jpeg)

> 图 1：Qwen2.5-VL 架构。原生分辨率图像和动态 FPS 视频经 Conv3D patchify、window/full attention 混合 ViT 与 $2\times2$ merger 后形成变长 token；视频 temporal ID 与绝对秒数对齐，再送入 Qwen2.5 LM Decoder。来源：论文 Figure 1，第 3 页；官方 arXiv 源码 `figures/qwen2.5vl_arc.jpeg`。

三种尺寸共享视觉配置：hidden size 1280、32 层、16 heads、intermediate size 3456、patch size 14、window size 112。四个全局注意力层索引为 `{7,15,23,31}`；其余 28 层使用窗口注意力。视觉—语言 merger 输入维度 1280，输出分别映射到 3B/7B/72B LLM 的 2048/3584/8192 维 embedding。

### 3.2 原生动态分辨率 ViT

视觉编码器从零训练，使用 DataComp 与内部数据完成初始化，而非沿用 Qwen2-VL 的 DFN ViT。每张图保留原宽高比，只将高宽调整到 28 的倍数；patch size 14 后，每个 $2\times2$ 相邻 patch 的特征拼接，再经两层 MLP 投影，因此 LLM 视觉网格步长相当于 28 像素。

从零训练的意义不是“随机初始化天然更好”，而是团队能够统一采用 RMSNorm、SwiGLU、2D-RoPE、window attention 与 3D patchification，不受旧视觉 checkpoint 结构限制。代价是必须付出 1.5T visual pre-training token 才能建立强视觉表征。

### 3.3 Window Attention：控制原生分辨率成本

全局视觉注意力对 $N$ 个 patch 的主要复杂度近似为 $O(N^2d)$；固定 $8\times8=64$ patch 的窗口后，大多数层变为 $O(N\cdot64\cdot d)$。只有四层执行 full attention，用于跨窗口信息交换。小于 112×112 像素的区域不 padding，以避免引入多余视觉 token。

这个设计比“所有层都局部”更稳健：局部层负责纹理、文字与邻域结构，全局层周期性聚合页面或整图上下文。但技术报告没有给出 window/full 层比例、窗口尺寸或速度—精度的消融，也没有公开统一分辨率下相对 Qwen2-VL 的端到端 latency，因此“显著降低开销”是有复杂度依据但缺少完整系统对账的结论。

### 3.4 动态 FPS 与绝对时间 MRoPE

Qwen2-VL 的视频 temporal ID 绑定抽取帧序号。假设同一 8 秒视频分别以 0.5、1、2 fps 采样，序号间隔都为 1，但每步实际经过 2、1、0.5 秒，模型无法仅凭帧号知道事件速度。Qwen2.5-VL 在训练时动态采样 FPS，并让 MRoPE temporal ID 的间隔与绝对时间对应。

抽象地，若视觉 token 来自时间 $\tau_i$，则 temporal position 可写为 $t_i=\operatorname{round}(s\tau_i)$，其中 $s$ 是固定的“每秒位置单位”；不同采样率只改变相邻 $t_i$ 的间隔，而不是重新把每帧连续编号。论文没有披露 $s$ 的具体数值与取整细节，因此不能补造。该设计无需额外时间定位 head，也不需要把 timestamp 作为文本反复插入视觉序列。

动态 FPS 与绝对时间必须配套：前者让训练看见不同时间间隔，后者把这些间隔显式编码。只做动态抽帧却仍使用连续帧号，会把快慢视频混为同一时间尺度。

---

## 4. 数据构建：能力增量主要来自哪里

### 4.1 预训练数据谱系

4.1T trained token 覆盖 caption、交错图文、OCR、视觉知识、多模态学术题、grounding、文档解析、视频描述、视频定位和 Agent 交互。网页交错数据先做常规清洗，再由内部评估模型从文本质量、图文相关性、信息互补性、信息密度平衡四方面打分。

grounding 数据同时含公开与内部数据，用 Grounding DINO、SAM、copy-paste augmentation 等合成 XML/JSON/自定义格式，覆盖超过 10,000 个目标类别；pointing 数据还包括 PixMo、检测/分割数据与细节指点合成管线。负类别查询和同类多实例图用于训练不存在目标与极端计数场景。

### 4.2 文档 omni-parsing：HTML 作为统一监督语言

传统文档系统把 layout analysis、OCR、table recognition、chart understanding 分给不同模型。Qwen2.5-VL 将段落、表格、图表、公式、图片 caption/OCR、乐谱、化学式统一成带 `data-bbox` 的 HTML：表格保留 `<table>`，公式包含图像框与公式内容，乐谱输出 ABC notation，化学式输出 SMILES。阅读顺序和各模块坐标也进入 ground truth。

这种设计把“看懂文档”转为结构化序列生成：既输出文字，也恢复版面与非文本元素。OCR 训练还加入法、德、意、西、葡、阿、俄、日、韩、越等语言；图表合成约 1M 样本，表格由约 6M 真实样本经离线识别与质量过滤得到。

### 4.3 视频数据

训练动态采样 FPS，使帧率分布更均衡；对超过半小时的视频，团队通过多帧 caption 合成长视频描述。视频 grounding 同时使用秒数格式和 hour-minute-second-frame 格式，训练模型理解与输出多种时间表达。报告没有披露视频总小时数、帧数分布和动态 FPS 的采样范围。

### 4.4 GUI Agent 数据

数据覆盖 mobile、web、desktop screenshot。合成引擎生成界面 caption 与 UI element grounding，分别训练“看懂界面”和“把功能落到坐标”。决策侧把三类平台操作统一成 function-call action space，并将开源轨迹和 Agent 框架在虚拟环境合成的多步轨迹重排为统一格式。

每步 reasoning 不是直接从动作模板复制：标注时先在截图突出 ground-truth 操作，再给全局任务、操作前后截图，让人工或模型解释该动作意图；模型过滤低质量 reasoning。作者认为这样可降低对动作标签的机械过拟合。

---

## 5. 训练流程与后训练

### 5.1 三段预训练

![](assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/figures/03-training-stages.png)

> 表 2：预训练依次为 1.5T visual pre-training、2T multimodal pre-training、0.6T long-context pre-training，总计 4.1T token；前两阶段长度 8192，第三阶段扩至 32768。来源：论文 Table 2，第 8 页；由官方 PDF 完整裁剪。

| 阶段 | Token | 序列长度 | 参数状态 | 主要数据 |
|---|---:|---:|---|---|
| Visual Pre-Training | 1.5T | 8192 | 只训练 ViT | caption、视觉知识、OCR |
| Multimodal Pre-Training | 2T | 8192 | ViT + LLM | 纯文本、交错图文、VQA、视频、grounding、Agent |
| Long-Context Pre-Training | 0.6T | 32768 | ViT + LLM | 长视频、长 Agent、长文档 |

训练使用按送入 LLM 的序列长度动态 packing，使不同 GPU 的计算负载更均衡。window attention 已降低视觉侧成本，系统因此优先平衡更昂贵的 LLM 序列。论文没有披露 optimizer、学习率、batch size、硬件数量、训练时长与总 FLOPs，这是从零复现的主要缺口。

### 5.2 SFT、拒绝采样与 DPO

SFT 约 2M 条，纯文本与多模态条目各占 50%；但后者包含视觉和时间信息，实际 token 与算力占比更高。数据以中英文为主，补充多语言，覆盖 VQA、caption、数学、代码、安全、Doc/OCR、grounding、video、Agent。

过滤分两步：Qwen2-VL-Instag 将 QA 分到 8 个主域与 30 个细类；随后各域使用规则和 reward model 过滤重复、截断、格式错误、无关、有害或视觉依据不足的样本。数学、代码和复杂 VQA 采用 rejection sampling：中间版模型生成 CoT，只有最终答案匹配 ground truth 且推理不过长、不重复、不乱切语言的样本被保留；还额外检查中间步骤是否真正使用视觉信息。

SFT 后执行 DPO；两阶段都冻结 ViT。DPO 只使用 image-text 与纯文本 preference data，每个样本处理一次。报告明确指出视觉与语言在中间推理步骤的最优对齐仍是 ongoing challenge，这也是少数直接披露的 future-work 方向。

---

## 6. 定量结果与 Baseline

### 6.1 综合与纯文本能力

主要 baseline 包括 GPT-4o-0513、Claude 3.5 Sonnet-0620、InternVL2.5-78B、Qwen2-VL-72B，以及各 benchmark 的 previous open-source SoTA。Qwen2.5-VL-72B 在 MMMU 70.2、MMMU-Pro 51.1、MathVista 74.8、MATH-Vision 38.1、MathVerse 57.6、MMStar 70.8、MMVet 76.2，均显著高于 Qwen2-VL-72B 已报告的对应结果或达到强基线水平。

它并非每项第一：MegaBench 51.3 低于 GPT-4o 54.2 与 Claude 52.1；MME sum 2448 低于 InternVL2.5-78B 2494；RealWorldQA 75.7 低于 InternVL2.5 的 78.7，也低于前代报告中的 77.8。不同论文版本、评测实现与提示词可能造成波动，不能只用代际标签解释所有差异。

纯文本上，Qwen2.5-VL-72B 与 Qwen2.5-72B 接近：MMLU-Pro 71.2 vs 71.1，MATH 83.0 vs 83.1，GPQA 都为 49.0；HumanEval 87.8、MultiPL-E 79.5、IFEval 86.3 甚至略高。它支持“多模态联合训练基本保留语言能力”，但不构成视觉数据必然提升文本模型的因果证据。

### 6.2 文档与 OCR

![](assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/figures/04-ocr-document-results.png)

> 表 5：Qwen2.5-VL-72B 在 CC-OCR、DocVQA、InfoVQA、OCRBench、OCRBench_v2 等文档/OCR 指标上领先多数 baseline；OmniDocBench 使用 edit distance，数值越低越好。来源：论文 Table 5，第 13 页；由官方 PDF 完整裁剪。

代表结果包括：CC-OCR 79.8，高于 Gemini 1.5 Pro 73.0、GPT-4o 66.9、InternVL2.5-78B 64.7；DocVQA 96.4，略高于 Qwen2-VL 的 96.5 报告口径附近并高于本表 GPT-4o 91.1；InfoVQA 87.3，高于 InternVL2.5 的 84.1；OCRBench 885，高于 InternVL2.5 的 854；OCRBench_v2 英/中 61.5/63.7，高于 Gemini 51.9/43.1。

反例同样重要：AI2D 88.7 略低于 InternVL2.5 89.1；ChartQA 89.5 低于 Claude 90.8；CharXiv reasoning-question 49.7 低于 Claude 60.2。Qwen2.5-VL 的优势更稳定地出现在 OCR、文档解析与信息抽取，而非所有图表推理任务。

### 6.3 Grounding 与计数

Qwen2.5-VL-72B 在 ODinW-13 达到 43.1 mAP，显示通用 VLM 与专用 detector 的差距缩小；CountBench 以“detect then count”提示达到 93.6，高于 Molmo-72B 91.2、Claude 89.7 和 GPT-4o 87.9。RefCOCO 系列则并非全面领先：例如 test-A 94.6，低于 InternVL2.5-78B 的 95.6；这说明新增 point/absolute-coordinate 能力很强，但专门 grounding 数据和评测实现仍决定结果。

### 6.4 视频理解与时间定位

| Benchmark | Gemini 1.5-Pro | GPT-4o | Qwen2.5-VL-72B | Qwen2.5-VL-7B |
|---|---:|---:|---:|---:|
| Video-MME 无字幕 | **75.0** | 71.9 | 73.3 | 65.1 |
| Video-MMMU | 53.9 | **61.2** | 60.2 | 47.4 |
| MVBench | 60.5 | 64.6 | **70.4** | 69.6 |
| LVBench | 33.1 | 30.8 | **47.3** | 45.3 |
| MLVU M-Avg | — | 64.6 | **74.6** | 70.2 |
| TempCompass Avg | 67.1 | 73.8 | **74.8** | 71.7 |
| Charades-STA mIoU | — | 35.7 | **50.9** | 43.6 |

绝对时间对齐与视频 grounding 数据最直接对应 Charades-STA 的 50.9 mIoU，以及 TempCompass、LVBench 的提升；但技术报告没有“帧序号 MRoPE vs 绝对时间 MRoPE”的受控消融，不能仅凭跨模型表格归因。所有视频评测仍最多 768 帧、24576 视频 token；所谓“小时级”理解依赖稀疏采样，不是逐帧无损输入。

### 6.5 GUI Agent

![](assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/figures/05-gui-agent-results.png)

> 表 9：Qwen2.5-VL-72B 的 GUI grounding 与离线控制相对 Qwen2-VL 大幅提升，ScreenSpot Pro 从 1.6 提至 43.6，Android Control LowEM 从 59.2 提至 93.7；OSWorld 8.83 仍低于 Claude 14.90 和 Aguvis-72B 10.26。来源：论文 Table 9，第 15 页；由官方 PDF 完整裁剪。

ScreenSpot 87.1 接近 Aguvis-72B 89.2；ScreenSpot Pro 43.6 显著高于 Aguvis 23.6；AndroidWorld success rate 35%，高于 GPT-4o 34.5%（带 SoM）和 Qwen2-VL 6%（带 SoM）；MobileMiniWob++ 68%。但 OSWorld 8.83 仍不高，说明局部 UI grounding 进步没有自动解决长程规划、应用状态、错误恢复和桌面工具使用。

不同 baseline 的输入条件并不完全等价：部分模型使用 Set-of-Mark，Qwen2.5-VL 不使用辅助标记。表格适合说明实际系统能力，但不是严格的同架构组件消融。

---

## 7. 消融证据与证据缺口

### 7.1 报告没有正式组件消融

技术报告没有独立 ablation section，也没有给出以下受控对比：window attention vs 全局 attention、从零 ViT vs DFN 初始化、固定 FPS vs 动态 FPS、帧序号 MRoPE vs 绝对时间 MRoPE、归一化坐标 vs 实际像素坐标、HTML 文档监督 vs 普通 OCR 文本、SFT-only vs SFT+DPO。

因此，架构机制的合理性可由复杂度和任务对应关系解释，最终性能则只能视为“架构 + 4.1T 数据 + Qwen2.5 LLM + 后训练”的整体结果。不能把 Charades-STA 或 ScreenSpot Pro 的全部提升归给单一模块。

### 7.2 可作为弱证据的代际对照

Qwen2.5-VL-72B 相比表中 Qwen2-VL-72B，在 MMMU 64.5→70.2、MathVista 70.5→74.8、MMStar 68.3→70.8；GUI 的 ScreenSpot Pro 1.6→43.6、Android Control HighEM 59.1→67.36、LowEM 59.2→93.7。代际对照显示升级有效，但由于底座从 Qwen2 换成 Qwen2.5、数据从约 1.2/1.4T 扩到 4.1T，不能建立组件级因果关系。

---

## 8. 局限、Future Work 与批判性评价

### 8.1 论文明确或直接暴露的局限

1. **视觉 CoT 对齐仍不充分。** 作者明确指出中间推理步骤可能忽略或误读视觉证据，“optimal modality alignment remains an ongoing challenge”。
2. **长视频仍是稀疏观察。** 评测封顶 768 帧和 24576 视频 token；数小时视频必然依赖抽样，短暂关键事件可能漏掉。
3. **GUI 长程成功率仍低。** OSWorld 8.83 表明真实电脑操作远未可靠，grounding 与完整 Agent 能力不可混同。
4. **数据与训练不可完全审计。** 4.1T 中含内部、购买、在线 query 与模型合成数据；比例、许可证、去重、硬件和优化超参未完整披露。
5. **绝对坐标带来接口耦合。** 输入 resize、设备缩放和 screenshot 坐标系必须严格一致，跨设备迁移需额外校准。

### 8.2 论文没有独立 Future Work 章节

报告结论主要总结能力，没有列出系统 future-work 清单。可直接追溯的后续方向只有后训练章节关于视觉—语言中间推理对齐仍需进步。以下属于**基于论文结果的个人判断**：下一步需要内容自适应帧选择而非只调 FPS、显式 Agent memory 与错误恢复、公开文档/GUI 闭环评测，以及固定 FLOPs 下验证 window attention 和绝对时间位置的因果贡献。

### 8.3 方法价值判断

Qwen2.5-VL 的重要性在于把“视觉感知”变成可消费的结构和动作：文档输出 HTML，不只回答问题；视频输出时间段，不只概述内容；GUI 输出实际坐标和 function call，不只描述界面。它比前代更接近通用视觉接口。但数据工程贡献与架构贡献高度耦合，缺少消融是技术报告最主要的证据短板。

---

## 9. 复现成本与实践建议

### 9.1 复现成本

- **权重与代码**：3B/7B/72B 及官方代码公开，推理和公开 benchmark 可复现。
- **训练**：从零 ViT 1.5T token、总计 4.1T token，且内部/合成/购买数据占比未知；从头复现对一般团队不可行。
- **后训练**：约 2M SFT 条目可知，但分类器、reward model、rejection sampling 管线与 preference data 未公开完整配方。
- **视频与 Agent**：还需稳定的视频解码、动态 FPS、GUI 环境、坐标映射和闭环评测基础设施，工程成本高于普通 VQA。

### 9.2 最小可行验证

1. 选 3B/7B，建立包含扫描件、表格、公式、图表和多语言的文档集，同时评估内容正确率与 reading order/HTML 结构。
2. 对同一视频构造不同 fps 采样版本，测试相同事件的秒级定位是否一致；报告 mIoU，而非只看问答准确率。
3. GUI 分开报告 element grounding、单步 action EM、完整 task success rate 与错误恢复率。
4. 对超高分辨率页面记录视觉 token 数、ViT latency、LLM prefill latency 和峰值显存，验证 window attention 只优化了哪一段。
5. 使用实际坐标时固定 screenshot 像素尺寸，并保存从模型坐标到设备事件坐标的变换矩阵。

---

## 10. 总结

Qwen2.5-VL 把 Qwen2-VL 的动态视觉接口推向高效与可执行：从零 ViT + window attention 承担高分辨率，动态 FPS + 绝对时间 MRoPE 承担真实时间，HTML 文档监督承担结构，实际像素坐标与统一动作空间承担 GUI。4.1T token 与精细合成/过滤让文档、时间定位和 Agent 指标出现大幅进步；同时，缺少组件消融意味着应把它理解为完整系统配方，而不是几个模块的独立胜利。其最清晰的后续问题，是让视觉证据在长推理和长交互中持续、可验证地影响每一步决策。

---

## 11. 材料来源

- 官方论文页：https://arxiv.org/abs/2502.13923
- 官方 PDF：https://arxiv.org/pdf/2502.13923
- 官方代码：https://github.com/QwenLM/Qwen2.5-VL
- 本地完整溯源：[sources.md](assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/sources.md)
