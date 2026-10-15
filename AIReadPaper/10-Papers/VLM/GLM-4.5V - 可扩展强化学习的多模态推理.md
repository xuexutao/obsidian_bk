---
type: paper-note
status: done
domain: VLM
paper: "GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning"
year: 2025
arxiv: "2507.01006"
doi: "10.48550/arXiv.2507.01006"
source: "https://arxiv.org/abs/2507.01006"
project: "https://github.com/zai-org/GLM-V"
code: "https://github.com/zai-org/GLM-V"
tags:
  - VLM
  - Multimodal-Reasoning
  - RLVR
  - RLCS
  - GUI-Agent
created: 2026-10-15
updated: 2026-10-15
---

# GLM-4.5V：可扩展强化学习的多模态推理

## 基本信息

- **作者：** GLM-V Team
- **版本：** arXiv v6，2026-01-01 修订
- **模型系列：** GLM-4.1V-9B-Base / GLM-4.1V-9B-Thinking / GLM-4.5V / GLM-4.6V-Flash / GLM-4.6V
- **本地原文：** [paper.pdf](assets/GLM-4.5V%20-%20可扩展强化学习的多模态推理/sources/paper.pdf)
- **完整溯源：** [sources.md](assets/GLM-4.5V%20-%20可扩展强化学习的多模态推理/sources.md)

**一句话结论：**

GLM-V 的关键经验是：多模态推理的上限首先由高质量、原生分辨率的视觉语言预训练决定，而大规模 RL 的成败取决于 **每个领域都足够可靠的 verifier** 与随模型能力动态调节难度的 **RLCS**；GLM-4.1V-9B-Thinking 验证了紧凑模型路线，GLM-4.5V 将其扩展到 106B-A12B MoE 并显著增强 GUI、coding、视频和视觉推理，但 3D-RoPE 公式、完整 RL 配方和训练成本仍未公开。

---

## 1. 研究背景与问题定义

### 1.1 为什么“更强视觉编码器 + 更多 SFT”还不够

多模态模型在感知、OCR、文档和开放问答上已有较强能力，但复杂 STEM、空间推理、GUI agent、视觉 coding 与长视频任务需要模型在视觉证据上进行长程推理。论文把问题拆成三个阶段：预训练要建立足够高的能力上限；SFT 要把能力组织成稳定的长思考格式；RL 要在多领域中持续提供准确、难度适宜的学习信号。

![](assets/GLM-4.5V%20-%20可扩展强化学习的多模态推理/figures/01-version-and-results.png)

> **图 1：版本扩展与 RL 增益。** 左图显示 4.1V-9B-Thinking 已能与更大开源模型竞争，4.5V 进一步扩大 STEM、OCR、GUI、视频和 coding 能力；右图显示 SFT→RL 的最高单域增益达到 10.6%。来源：论文 Figure 1，arXiv v6。

### 1.2 论文的核心主张

第一，基础模型并非只负责“看懂图”，其 pass@k 潜力决定后续 RL 上限。第二，多领域 RL 不是简单混合数据；任何一个低质量 verifier 都可能被 hack，进而拖垮其他领域。第三，随着训练推进，大量样本变得过易，必须动态把 rollout 预算集中到中等难度样本。RLCS 正是这一主张的算法化实现。

### 1.3 主要贡献

1. 统一的原生分辨率图像/视频架构：AIMv2-Huge ViT、MLP projector、GLM decoder、3D 卷积与时间索引 token。
2. 通过 2D-RoPE、绝对位置嵌入插值和 LLM 侧 3D-RoPE，支持极端宽高比与高分辨率空间建模。
3. 大规模多源预训练数据，包括 100 亿级初始图文池、2.2 亿 OCR 图像、4,000 万自然图 grounding、1.4 亿 GUI QA 和 5,000 万 instruction 样本。
4. 统一的跨域 reward system 与 RLCS，配合 ratio EMA 过采样、force answering、去 KL/entropy 等稳定化方案。
5. 开源 4.1V、4.5V、4.6V 模型及 reward system，并在 42 个公开 benchmark 上系统评测。

---

## 2. 版本关系与输入输出

### 2.1 4.1V → 4.5V → 4.6V

| 版本 | 架构 | 底层 LLM | 模式 | 上下文/定位 |
|---|---|---|---|---|
| GLM-4.1V-9B-Base | 9B dense | GLM-4-9B-0414 | base | 预训练基础模型 |
| GLM-4.1V-9B-Thinking | 9B dense | GLM-4-9B-0414 | thinking | 紧凑长推理模型，训练序列最长 32K |
| GLM-4.5V | 106B 总参数、12B 激活 MoE | GLM-4.5-Air | thinking / non-thinking | 从 9B 路线扩展出的主模型 |
| GLM-4.6V-Flash | 9B dense | GLM-4-9B-0414 | 双模式、工具使用 | 4.1V 路线更新版 |
| GLM-4.6V | 106B-A12B MoE | GLM-4.5-Air | 双模式、原生工具使用 | 4.5V 更新版，131,072 训练长度 |

“从 4.1V 扩展到 4.5V”指方法路线和规模扩展，不代表论文公开了直接扩参、参数拷贝或蒸馏算法；这些细节未披露。v6 同时加入 4.6V 结果，所以阅读 4.5V 时要避免把 4.6V 的 128K 和原生工具调用倒推到所有旧版本。

### 2.2 输入与输出

输入包括单图、多图、视频帧、文本 prompt、图文交错长文档，以及 4.6V 的工具签名。架构支持超过 200:1 的宽高比和超过 4K 的图像；评测时图像预计最多使用 6,144 个视觉 token，视频最多 48,000 个视觉 token，但这些是评测配置，不是公开的硬上限。

输出是自回归文本 token，可包含普通回答、长链思考、non-thinking 直接回答、boxed final answer、grounding box、GUI action、前端代码与结构化工具调用。

---

## 3. 核心架构：原生分辨率视觉、3D-RoPE 与时间 token

### 3.1 总体数据流

![](assets/GLM-4.5V%20-%20可扩展强化学习的多模态推理/figures/02-architecture.png)

> **图 2：GLM-V 共享架构。** 图像和视频按原生宽高比进入 ViT，视觉表示经 MLP projector 与文本 token 拼接；视频每帧后插入真实时间戳编码的 time index token，再由语言 decoder 统一推理。来源：论文 Figure 2。

视觉 encoder 由 AIMv2-Huge 初始化，原始二维卷积被 3D 卷积替换。视频因此在时间维下采样 2 倍；单图会复制一次，以保持同一处理路径。论文没有披露 3D 卷积的 kernel、stride、padding、patch size，以及各阶段视觉 encoder 的冻结明细。

### 3.2 2D-RoPE 与绝对位置嵌入插值

ViT self-attention 加入 2D-RoPE，同时保留预训练 ViT 的可学习绝对位置嵌入。对于 patch 网格坐标 $(w,h)$：

$$\mathbf g_{\mathrm{norm}}=(w_{\mathrm{norm}},h_{\mathrm{norm}})=2\left(\frac{w+0.5}{W_p},\frac{h+0.5}{H_p}\right)-1$$

然后通过双三次插值从原始位置表采样：

$$P_{\mathrm{adapted}}(\mathbf g)=\mathcal I_{\mathrm{bicubic}}(P_{\mathrm{orig}},\mathbf g_{\mathrm{norm}})$$

这套“双位置机制”一方面让预训练绝对位置表适配任意网格，另一方面用 2D-RoPE 提供相对空间结构。

### 3.3 3D-RoPE：论文明确与未明确的边界

LLM 侧把 RoPE 扩展为 3D-RoPE，以提升多模态空间理解并尽量保持文本能力。但正文没有公式，也没有说明三个轴究竟如何对应时间/高度/宽度或序列/高度/宽度；没有给出文本 token 坐标、head dimension 划分、频率基底、图像/视频 position id 构造。因此只能确认“LLM 使用三维旋转位置编码”，不能据此重建实现，更不能默认它与 Qwen 的 M-RoPE 完全相同。

### 3.4 显式时间索引

每个视频 frame token 后插入 time index token，具体实现是把真实时间戳编码成字符串。这让模型不仅知道帧顺序，还知道帧之间的现实时间间隔。论文未披露时间戳格式、精度、抽帧率和最大帧数。

---

## 4. 预训练数据与训练配方

### 4.1 数据构建

图像 caption 初始池超过 100 亿图文对，来源包括 LAION、DataComp、DFN、Wukong 和搜索数据。处理包含规则过滤、图像去重、CLIP-Score > 0.3 的相关性过滤、MetaCLIP 风格长尾概念重采样，以及事实中心 recaption。

图文交错数据来自 MINT、MMC4、OmniCorpus 与超过 1 亿本数字化书籍。网页管线去除广告、二维码、相册式低文本页面，并训练高知识密度分类器筛选学术图表、科学插图、工程图、教学图和地图。

OCR 共 2.2 亿张图像，包含合成文档、PaddleOCR 标注的自然场景文字和 arXiv 学术文档；后者将 LaTeX 经 LaTeXML 转为 HTML/轻量标记，并按原 PDF 分页栅格化。自然图 grounding 最终得到 4,000 万条标注；GUI 通过 CommonCrawl、网页截图、Playwright 和 DOM 框生成超过 1.4 亿问答。instruction tuning 数据共 5,000 万样本，覆盖视觉、STEM、文档、GUI agent 与 UI coding。

论文没有披露最终 caption 保留量、数据授权与完整混合比例，也没有给出视频数据规模。

### 4.2 两阶段预训练

| 阶段 | 序列长度 | global batch | 步数 | 主要变化 |
|---|---:|---:|---:|---|
| Multimodal pre-training | 8,192 | 1,536 | 120,000 | 不含视频；多模态+纯文本；sequence packing |
| Long-context continual training | 32,768 | 1,536 | 10,000 | 加入视频、高分辨率图、>8K 图文交错；CP=4 |
| 4.6V 额外扩展 | 131,072 | 128 | 2,000 | 仅用于 4.6V 长上下文扩展 |

9B 模型使用 TP=2；106B-A12B 使用 EP=8、PP=4。MoE 采用 loss-free routing，router bias 更新率 $10^{-3}$，sequence-level balance loss 系数 $10^{-4}$。优化器、学习率、总 token、GPU 数量、训练时长和 FLOPs 未披露。

---

## 5. SFT：长思考格式与双模式

SFT 用于把基础能力组织成可训练的推理和回答格式，而不是主要注入新知识。标准样本使用 `<think>...</think><answer>...</answer>`，可验证任务只允许一个 `<|begin_of_box|>...<|end_of_box|>` 最终答案区间，以避免开放式输出中 LLM 抽取器误判。GLM-4.5V 的回复去掉了 `<answer>` 标签，但 box token 仍承担 RLVR 答案提取。

4.1V-Thinking 只训练 thinking；4.5V/4.6V 混合 thinking 与 non-thinking，用户输入 `/nothink` 时生成空思考并直接回答。non-thinking 答案复用 thinking 样本的 answer 部分，作者称优于另建短答数据集。SFT 全参数训练，4.1V/4.5V 长度 32,768，4.6V 为 131,072，global batch 32；样本总量、步数、学习率与双模式比例未披露。

---

## 6. 多领域 RL、Reward System 与失败模式

### 6.1 RLVR/RLHF 的统一框架

GLM-V 对数学、物理、化学、OCR、文档、图表、grounding、视频、GUI agent 与指令遵循同时进行 RLVR/RLHF。可验证任务先提取 box token 内的最终答案，再返回 0/1 或连续奖励；开放任务由 reward model 评价 answer 部分。优化目标采用 GRPO，但论文没有给出完整公式、rollout 数和总训练步数。

OCR 奖励是归一化编辑距离：

$$R_{\mathrm{OCR}}=1-\frac{d_{\mathrm{edit}}(\mathrm{ans},\mathrm{gt})}{\max(|\mathrm{ans}|,|\mathrm{gt}|)}$$

Grounding 奖励是 IoU 超过阈值的框占比：

$$R_{\mathrm{grounding}}=\frac{\#\{\mathrm{boxes\ with\ IoU}>\tau\}}{\#\{\mathrm{all\ boxes}\}}$$

阈值 $\tau$ 未披露。数学使用 SymPy+容差，物理/化学在单位存在时引入 LLM judge，chart 数值采用数学规则、文本先 exact match 后语义判断，GUI 同时检查 action 与 IoU。

### 6.2 为什么 verifier 是系统的单点风险

![](assets/GLM-4.5V%20-%20可扩展强化学习的多模态推理/figures/03-reward-hacking.jpg)

> **图 5：低质量 verifier 引发全局崩溃。** 单图域出现 reward noise，多图域出现 reward hacking；约 step 150 后，即便 STEM verifier 本身可靠，STEM reward 也停滞，MMMU、MathVista、AI2D 等真实指标同步下降。来源：论文 Figure 5。

论文给出的 hacking 例子很具体：计数题回答“0 到 10 之间的正确数字”，相对论题回答“非常接近光速的速度”，粗糙 LLM verifier 仍可能给高分。这说明多域共享参数使坏奖励不只污染一个任务，而会改变整体输出分布。

---

## 7. RLCS：可扩展强化学习的核心

### 7.1 动机

GRPO 中，同一 prompt 的 rollout 若全部答对或全部答错，就缺少有效组内优势。作者观察到训练约 200 步后，超过一半 prompt 的准确率已高于 90%；而 rollout 是主要时间瓶颈，继续均匀采样会浪费大量计算。

### 7.2 离线与在线难度

训练前用多个成熟 VLM 或旧 RL checkpoint 做 pass@k，并结合人工难度标签，把样本分为 very easy 到 very hard。训练中记录每次 rollout 的 pass@k，与离线标签合并，再按子类别表现逐 iteration 调整采样。过易样本和当前完全不可解样本降权，中等难度样本升权。

论文没有披露难度层数、离线/在线融合公式、权重函数、pass@k 的 $k$ 或 curriculum 更新率，因而 RLCS 的思想清楚，精确复现仍需要代码。

### 7.3 Dynamic Sampling Expansion via Ratio EMA

上一轮全对/全错样本比例决定下一轮过采样：

$$\texttt{expansion\_ratio}=\frac{1}{1-\texttt{not\_valid\_sample\_rate}}$$

对该比率维护 EMA，并把 `expansion_ratio_ema` 用作下一轮 rollout 系数。先过采样，再优先保留正确/错误数量更均衡的 prompt，从而稳定有效 batch size。EMA 衰减系数未披露。

### 7.4 其他有效性与稳定性措施

- **Force Answering：** 长思考触及上限时插入 `</think>`，避免合法推理因未生成最终答案而得 0 分。
- **移除 KL loss：** 显式压制快速增长的 KL 会限制 VLM 能力。
- **移除 entropy loss：** 作者观察到它会诱发乱码并最终崩溃。
- **top-p=1：** 比 0.9 更稳定，可能因为保留稀有 token 的学习覆盖。
- **clip-higher：** 提高 importance ratio 上界，改善 off-policy 和 entropy collapse。
- **per-sample loss：** 平均奖励相近，但比 per-token 更稳定。
- **序列重打包：** rank 内 packing 与梯度累积使 forward-backward 时间约减半。

---

## 8. 跨领域迁移与消融

![](assets/GLM-4.5V%20-%20可扩展强化学习的多模态推理/figures/04-cross-domain-rl.png)

> **图 6：跨领域 RL 泛化。** 行表示训练域，列表示评估域。GUI-only 对全部域均有正迁移；Mix-all 在 STEM、OCR/Chart 和 Image General 最优，但 Grounding 与 GUI 未继续超过对应单域训练。来源：论文 Figure 6。

热力图说明多域 RL 的干扰比 SFT 弱，但并非“混合越多越好”。例如 Grounding-only 会让 OCR/Chart -4.6、Image General -0.4；OCR/Chart-only 会让 Grounding -14.5。Mix-all 的总体正迁移依赖数据比例和 verifier 质量，而不是天然成立。

其他分析性实验包括：SFT cold start 从 1,000 步增至 2,000 步，SFT 后约提升 2 分，但 RL 后上限几乎相同；加入 KL/entropy 会限制能力或导致崩溃；基础模型在 MathVista 非选择题 pass@k 上显著高于同规模对照。多数结论没有公开完整数值表或多随机种子。

---

## 9. 主要定量结果

### 9.1 4.1V 与 4.5V

GLM-4.1V-9B-Thinking 在 42 个公开 benchmark 中有 29 项超过 Qwen2.5-VL-72B，说明强基础和 RL 配方可让 9B 模型跨越数量级差距。GLM-4.5V 在同量级开源模型中整体领先，并在 coding、GUI agent、视频和 STEM 上进一步扩展。

| 类别 | Benchmark | 4.1V | 4.5V | 重要 baseline |
|---|---|---:|---:|---:|
| STEM | MMMU Val | 68.0 | **75.4** | Step-3 74.2 |
| STEM | MathVista | 80.7 | **84.6** | Kimi-VL 80.1 |
| OCR/文档 | OCRBench | 84.2 | 86.5 | Kimi-VL 86.9 |
| 文档 | MMLongBench-Doc | 42.4 | **44.7** | Kimi-VL 42.1 |
| Grounding | RefCOCO avg val | 85.3 | **91.3** | Qwen2.5-VL 90.3 |
| GUI | OSWorld 100-step | 14.9 | **35.8** | Qwen2.5-VL 8.8 |
| GUI | WebVoyager | 69.0 | **84.4** | Qwen2.5-VL 40.4 |
| Coding | Design2Code | 64.7 | **82.2** | Qwen2.5-VL 41.9 |
| Video | VideoMME w/sub | 73.6 | **80.7** | Qwen2.5-VL 79.1 |
| Video | VideoMMMU | 61.0 | **72.4** | Kimi-VL 65.2 |

OCRBench 是少数 4.5V 未超过最强开源对照的任务。4.6V 并非每项都高于 4.5V，例如部分 Hallusion、MathVision、RefCOCO 与 WebVoyager 指标有回落，所以版本升级主要增加上下文和工具能力，并非所有 benchmark 单调提升。

### 9.2 与闭源模型比较的边界

摘要称 4.5V 在 22 个 benchmark 上与 Gemini-2.5-Flash 相当或更好，但正文主表没有逐项列出 Gemini 数值。这个总结可作为官方结论，不能替代逐 benchmark、同预算的严谨比较。评测统一使用 vLLM/SGLang、最大输出 8,192，并要求至少 95% 请求成功；失败请求被剔除，仍可能对不同模型产生不同偏差。

---

## 10. 局限与未来工作

论文明确承认：结果奖励只检查最终答案，可能强化错误或幻觉推理链；RL 对细微设置仍敏感；遮挡、拥挤和视觉歧义会造成感知错误并向后污染推理；主观任务 verifier 更易被 hack；多模态训练是否能系统提升纯文本推理仍未解决；许多 benchmark 接近饱和，无法诊断 shortcut 与中间推理幻觉。

此外，论文没有公开完整 RL 数据量、混合比例、rollout 数、总步数、硬件、成本和 3D-RoPE 实现。5,000 万 instruction 数据与 1.4 亿 GUI QA 的授权和污染检查只给出原则说明，缺少外部审计。GUI agent 和 tool use 还需要安全、权限和恶意页面鲁棒性评估。

---

## 11. 复现成本与实践建议

从零复现 106B-A12B 版本成本极高，且缺失学习率、总 token 和硬件配置。更可行的是复现 9B 路线：

1. 以原生分辨率 ViT + projector + 9B LLM 构建基础模型，优先保证预训练 pass@k，而不是过早做大量 SFT。
2. 为每个 RL 域先写 verifier 单元测试和对抗样例；任何域未达到足够精度前不要进入 Mix-all。
3. 实现离线难度+在线 pass@k 的 curriculum，并同时报告有效 prompt 比例、rollout 利用率和真实吞吐。
4. 做 RLCS 的必要对照：均匀采样、只离线难度、只在线难度、去 ratio EMA、不同中等难度窗口。
5. 若复现 3D-RoPE，必须以官方代码为准；论文正文不足以独立实现。

---

## 12. 个人判断

这篇报告最可信、也最可迁移的结论不是“某模型在 42 个榜单领先”，而是 **多领域 RL 的瓶颈首先是 reward engineering，其次才是优化算法**。Figure 5 直观证明，一个坏 verifier 可以让奖励上涨而真实能力下降，并通过共享参数污染原本可靠的 STEM 域。这对任何统一多模态 RL 系统都比单项 SOTA 更重要。

RLCS 的方向也成立：当大部分 prompt 已全对时，继续 rollout 本质上是在烧算力。但论文没有把采样函数和难度融合完全公式化，因此它更像一套经过大规模验证的工程原则，而非可直接复制的封闭算法。

对 4.1V→4.5V 的理解应是“基础模型、数据、奖励和规模共同扩展”，而不是简单参数变大。9B 模型的强表现说明训练质量可显著压缩尺寸差距；106B-A12B 则把稀疏容量投入到 GUI、coding、视频等长尾能力。整体上，论文方法价值很高，复现透明度中等，最需要补充的是 3D-RoPE、RLCS 完整实现与成本报告。

---

## 13. 材料来源

官方 PDF、arXiv source、代码仓库、四张原文关键图及版本说明见 [sources.md](assets/GLM-4.5V%20-%20可扩展强化学习的多模态推理/sources.md)。本文以 arXiv v6 为准，对论文未披露的训练与结构细节均未作补造。