---
type: paper-note
status: complete
domain: VLM
paper: "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond"
year: 2023
arxiv: "2308.12966"
doi: null
source: "https://arxiv.org/abs/2308.12966"
project: null
code: "https://github.com/QwenLM/Qwen-VL"
tags:
  - VLM
  - Qwen-VL
  - visual-grounding
  - OCR
created: 2026-10-15
updated: 2026-10-15
---

# Qwen-VL：定位与文字读取的通用视觉语言模型

## 基本信息

- **作者：** Jinze Bai、Shuai Bai、Shusheng Yang 等，Alibaba Group
- **论文版本：** arXiv:2308.12966v3，2023-10-13
- **研究对象：** 以 Qwen-7B 为语言底座、统一覆盖图像描述、VQA、OCR/文档理解、指代表达定位与多图对话的通用视觉语言模型
- **模型组成：** 1.9B ViT + 0.08B VL Adapter + 7.7B LLM，总计约 9.6B 参数
- **本地原文：** [paper.pdf](assets/Qwen-VL%20-%20定位与文字读取的通用视觉语言模型/sources/paper.pdf)
- **来源清单：** [sources.md](assets/Qwen-VL%20-%20定位与文字读取的通用视觉语言模型/sources.md)

**一句话结论：**

Qwen-VL 的关键不是简单把视觉编码器接到 Qwen-7B，而是用**带二维位置信息的 256 个可学习 query**将高分辨率视觉序列压缩成固定长度，并把 bounding box 直接文本化，再配合三阶段训练，把通用对话、文字读取与视觉定位第一次较完整地收进同一个开源通用模型；代价是固定 448×448 输入和固定 256-token 视觉瓶颈限制了极高分辨率、长视频与细粒度信息保真。

![](assets/Qwen-VL%20-%20定位与文字读取的通用视觉语言模型/figures/01-capabilities.jpg)

> 图 2：Qwen-VL-Chat 的多图、多轮、中英对话、OCR 与目标定位示例。它直观展示了论文所说的“通用”并非只指 VQA，而是统一生成文本、框坐标和跨图比较。来源：论文 Figure 2，第 2 页。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

2023 年的开源 LVLM 多采用“视觉编码器 → 跨模态连接器 → LLM”范式，但与闭源模型相比通常训练规模不足，而且视觉能力偏粗粒度：能描述整图，却不一定能读出小字、理解文档，也不一定能把语言指称落到具体区域。Qwen-VL 要回答的问题是：**能否在一个约 10B 参数、可公开使用的生成式模型中，同时保留语言能力，并统一实现图像理解、文字读取和区域定位？**

### 1.2 输入、输出与任务边界

模型输入可以是纯文本、单图或多图与文本交错序列。图像经过 ViT 与视觉语言适配器后，以 `<img>` 和 `</img>` 包围的视觉 token 序列插入 LLM 上下文。输出仍是自回归文本，但可以包含普通回答、描述、OCR 转写以及由 `<box>`、`</box>` 包围的坐标字符串。

任意边界框首先缩放到 $[0,1000)$，再写成 $((x_1,y_1),(x_2,y_2))$；`<ref>` 与 `</ref>` 标记坐标所对应的词语。这样 grounding 没有新增检测头或坐标词表，而被转化为标准 next-token prediction。

统一训练目标可写为 $\mathcal{L}_{\mathrm{NTP}}=-\sum_t\log p_\theta(y_t\mid y_{<t},I)$，其中 $I$ 是一个或多个图像，$y_t$ 是需监督的文本或坐标 token。论文三个阶段都使用文本 token 的交叉熵作为主目标。

### 1.3 核心贡献

1. **位置感知的视觉—语言适配器。** 单层 cross-attention 用固定 **256 个 learnable queries**压缩 ViT 特征，并把二维绝对位置编码加入 query-key 配对，避免压缩时完全丢掉空间结构。
2. **统一的文本化输入输出接口。** 图像、文本、区域指称和边界框都进入同一自回归序列，使 caption、VQA、OCR、grounding 和多图对话可共享模型。
3. **三阶段训练课程。** 从 14 亿清洗图文对的低分辨率对齐，过渡到高分辨率多任务预训练，再以 35 万条多模态/文本指令数据得到 Qwen-VL-Chat。
4. **细粒度数据建设。** 第二阶段显式加入 OCR、PDF/HTML 渲染、grounding、referring grounding、grounded caption 与纯文本数据，而不是仅依赖通用 caption。
5. **同尺度开源模型中的强结果。** 在 Flickr30K、VQAv2、TextVQA、OCR-VQA 与 RefCOCO 系列上显著超过当时多种通用基线，并展示 few-shot 与中文多模态对话能力。

---

## 2. 核心方法

### 2.1 总体架构与数据流

Qwen-VL 由三块组成：OpenCLIP ViT-bigG 初始化的视觉编码器、随机初始化的单层 cross-attention adapter、Qwen-7B 初始化的语言模型。ViT 以 stride 14 切分图像；**<span style="color: red">第一阶段 224×224 产生 256 个 patch token，第二、三阶段 448×448 产生 1024 个 patch token</span>**。无论输入多少 patch，adapter 都输出固定 256 个视觉 token。

![](assets/Qwen-VL%20-%20定位与文字读取的通用视觉语言模型/figures/02-training-pipeline.png)

> 图 3：三阶段训练流程。Stage 1 冻结 QwenLM、训练 ViT 与 adapter；Stage 2 解冻全模型并提高分辨率；Stage 3 冻结 ViT、训练 adapter 与 LLM。来源：论文 Figure 3，第 4 页。

完整数据流如下：

| 阶段 | 输入 | 中间表示 | 输出 |
|---|---|---|---|
| 视觉编码 | 224² 或 448² 图像 | 256 或 1024 个 ViT patch 特征 | 带二维位置的视觉序列 |
| Query 压缩 | 视觉序列 + 256 个可学习 query | 单层 cross-attention | 固定 256 个视觉 token |
| 多模态拼接 | `<img>` + 视觉 token + `</img>` + 文本 | LLM 上下文序列 | 自回归隐状态 |
| 任务生成 | caption/VQA/OCR/grounding 指令 | next-token decoding | 文本、框坐标或多轮回复 |

### 2.2 固定 256 query：为何是首代的关键设计

设 ViT 输出为 $V\in\mathbb{R}^{N\times d_v}$，可学习查询为 $Q\in\mathbb{R}^{256\times d_q}$，适配器近似执行 $Z=\mathrm{CrossAttn}(Q,V+P_{2D})$。<span class="md-comment-hl" data-comment="这里比较想了解 这个 P_2D 的shape是什么，他是如何得到的" title="💬 这里比较想了解 这个 P_2D 的shape是什么，他是如何得到的">这里 $P_{2D}$ 把 patch 的二维绝对位置加入交互，$Z$ 的长度恒为 256。</span>

> 💬 **批注**（针对「这里 $P_{2D}$ 把 patch 的二维绝对位置加入交互，$Z$ 的长度恒为 …」）：这里比较想了解 这个 P_2D 的shape是什么，他是如何得到的

这个设计同时解决两个问题：一是把 448×448 下的 1024 个 patch 压到 LLM 可承受的长度；二是提供类似 BLIP-2 Q-Former 的内容选择机制。但它也是信息瓶颈：无论原图多复杂，LLM 最终都只看到 256 个视觉 token，而且输入还被统一 resize 到固定分辨率。

论文在附录比较 64、144、256、400 个 query。早期训练时 query 越少初始 loss 越低，但收敛后过少会丢信息、过多会增加优化与计算难度；考虑第二阶段 1024 个 ViT token，最终选 256。

![](assets/Qwen-VL%20-%20定位与文字读取的通用视觉语言模型/figures/04-query-ablation.png)

> 图 7：learnable query 数量消融。L256 在收敛区间整体最好或接近最好，支持固定 256 query 的选择，但实验只比较训练 loss，没有给出各下游任务的完整精度—成本曲线。来源：论文 Figure 7，第 22 页。

### 2.3 文本化 grounding 与 OCR

Qwen-VL 不使用额外检测 decoder。区域描述和定位都转换为带标记的文本：描述一个区域时，`<ref>对象</ref>` 与 `<box>(x1,y1),(x2,y2)</box>` 在序列中对齐；输入框、输出框和 grounded caption 共享同一表示。这一接口让模型能够在多轮对话中继续引用区域，也允许中文指称直接输出坐标。

OCR 能力主要来自数据与分辨率，而非独立 OCR 模块。第二阶段加入 2480 万条 OCR 样本，包含 SynthDoG 中英文合成图、Common Crawl PDF 与 HTML 页面。PDF 用 PyMuPDF 提取文本和框并渲染页面，HTML 用 Puppeteer 做类似处理；合成 OCR 还保留文本的四边形坐标。因而模型既学习“读什么”，也学习“文字在哪里”。

### 2.4 三阶段训练

#### 2.4.1 Stage 1：图文对齐预训练

原始约 50 亿图文对清洗后保留 14 亿，其中英文 77.3%、中文 22.7%。来源含 LAION、LAION-COCO、DataComp、Coyo、CC12M/3M、SBU、COCO Caption 与内部中文数据。图像为 224×224，只训练 ViT 与 adapter，Qwen-7B 冻结；50k steps、global batch 30,720、峰值学习率 $2\times10^{-4}$，累计约 15 亿样本、500B image-text tokens。

#### 2.4.2 Stage 2：高分辨率多任务预训练

分辨率升到 448×448，解冻全模型，LLM 序列长度 2048。七类训练数据为：19.7M caption、3.6M VQA、3.5M grounding、8.7M referring grounding、8.7M grounded caption、24.8M OCR、7.8M pure-text autoregression。训练 19k steps，global batch 4096，峰值学习率 $5\times10^{-5}$。

此阶段的本质增量不是“多一些数据”，而是把输出粒度从整图语义扩展到区域与文字，并用纯文本数据抑制语言能力退化。

#### 2.4.3 Stage 3：监督指令微调

使用约 35 万条指令数据，来源包括 caption/对话自指令、人工标注、模型生成与策略拼接，覆盖定位、多图理解、多语言和纯文本对话。视觉编码器冻结，只优化 LLM 与 adapter；训练 8k steps，global batch 128，峰值学习率 $10^{-5}$。该阶段产物是 Qwen-VL-Chat。

### 2.5 推理复杂度与首代边界

视觉自注意力对 patch 数近似为 $O(N^2)$。论文附录显示，在 448² 下 global attention 约 10 s/iter、window attention 约 9 s/iter；在 896² 下则从约 60 s/iter 降到 25 s/iter。不过正式 Qwen-VL 仍采用固定分辨率流程，window attention 只是附录实验，并未形成下一代那种原生动态分辨率架构。

---

## 3. 数据集与实验设置

### 3.1 Baseline 与指标

论文比较 Flamingo、BLIP-2、InstructBLIP、Kosmos-1/2、Shikra、mPLUG-DocOwl、Pix2Struct 等通用模型，并把 PALI-X、G-DINO-L、UNINEXT-H、ONE-PEACE 等任务专用模型作为上界参考。指标覆盖 CIDEr、VQA Score、Exact Match、ANLS、Relaxed EM、grounding accuracy，以及 TouchStone 的 GPT-4 评分。

需要注意：表中模型规模、训练数据和提示并不完全匹配；“generalist SOTA”较有可比性，和 specialist SOTA 的对比只能说明差距，不能视为严格公平竞赛。

### 3.2 主要定量结果

| 能力 | Qwen-VL / Qwen-VL-Chat | 关键 baseline | 解读 |
|---|---:|---:|---|
| Flickr30K 0-shot CIDEr | 85.8 / 81.0 | Kosmos-2 80.5；Flamingo-80B 67.2 | 9.6B 级模型在 caption 上超过更大通用模型 |
| VQAv2 | 79.5 / 78.2 | Shikra-13B 77.36 | 通用 VQA 强，但低于任务专用 PALI-X 86.1 |
| OKVQA | 58.6 / 56.6 | Shikra-13B 47.16 | 视觉与外部知识结合明显提升 |
| TextVQA | 63.8 / 61.5 | DocOwl 52.6；InstructBLIP 50.7 | 高分辨率与 OCR 数据直接体现 |
| OCR-VQA | 75.7 / 70.5 | Pix2Struct-L 71.3 | 通用模型超过文档专用通用基线 |
| RefCOCO test-B | 85.34 / 84.51 | Shikra-13B 81.81 | 文本化 box 接口确实学到精细定位 |
| SEED-Bench All | 56.3 / 58.2 | InstructBLIP 53.4 | 指令微调改善真实对话综合能力 |
| MME Perception | Qwen-VL-Chat 1487.58 | InstructBLIP 1212.82 | 细粒度感知领先当时聊天式 LVLM |

![](assets/Qwen-VL%20-%20定位与文字读取的通用视觉语言模型/figures/03-benchmark-radar.png)

> 图 1：Qwen-VL 与当时通用 LVLM 的能力雷达图。优势集中在 grounding、OCR/TextVQA、VQA 与 caption；雷达图将不同量纲归一后更适合看覆盖面，不应替代原始表格。来源：论文 Figure 1，第 1 页。

### 3.3 Few-shot、视频迁移与定性结果

模型可把若干图文示例作为上下文，在 OKVQA、VizWiz、TextVQA 与 Flickr30K 上获得 few-shot 增益，且未使用 RICES 等检索式示例选择。SEED-Bench 视频任务只采样 4 帧，Qwen-VL 仍得到 39.1，说明多图接口可迁移到极短稀疏视频，但这不能等价于真正的视频时序建模。

### 3.4 消融是否充分

论文给出 query 数量和 window/global attention 两组消融。query 实验解释了为什么选 256，但只报告 loss；attention 实验给出 loss 与训练速度，却没有系统报告下游精度。因此消融能支持“优化与成本折中”，不足以严格分离二维位置编码、OCR 数据、grounding 数据和分辨率提升各自贡献。

---

## 4. 与前后代的关系

### 4.1 相对同期方法

与 LLaVA 的线性投影相比，Qwen-VL 用 cross-attention query 主动压缩视觉信息；与 BLIP-2 Q-Former 相似，但强调二维位置感知与 grounding；与 Kosmos-2、Shikra 相比，它把定位、OCR、文档、caption、VQA 和多图对话放进更大规模的中英数据与 Qwen 语言底座。

### 4.2 对后续 Qwen-VL 系列的起点意义

这一代确立了三个系列传统：视觉编码器—merger—LLM 三段式结构、视觉/文本交错输入、坐标文本化输出。但后续迭代几乎都在解除首代限制：Qwen2-VL 用动态分辨率和 token merger 取代固定 256 query；Qwen2.5-VL 重训原生 ViT 并强化文档、视频和 GUI；Qwen3-VL 再引入 DeepStack、长上下文、显式时间戳与 Thinking/RL。

---

## 5. 局限与批判性评价

### 5.1 作者明确给出的 future work

作者提出三条方向：扩展到语音和视频；扩大模型、数据与输入分辨率；加入高保真图像与流畅语音生成。它们也侧面承认本模型仍是以静态图像理解为主的单向生成器。

### 5.2 方法边界

1. **固定 448² 输入。** 长文档、小字、超宽截图会被 resize，细节损失无法由后端语言模型恢复。
2. **固定 256-token 瓶颈。** 对复杂页面和密集场景，所有视觉证据争夺同一组 query；模型无法按输入复杂度自适应增加 token。
3. **缺少真正视频建模。** 四帧评测只能说明多图迁移，不能证明时序、速度与长视频记忆。
4. **私有数据与算力不可复现。** 14 亿清洗图文对中的内部数据、OCR 合成细节和 3D 并行资源未完整公开。
5. **基准污染与评估公平性未充分讨论。** 大规模网页与 PDF 数据可能与公开测试集存在重叠；TouchStone 又依赖 GPT-4 judge。
6. **坐标生成脆弱。** 生成式 bbox 容易出现格式错误、越界或对象—坐标错配，论文没有系统报告校准、置信度与失败案例。
7. **消融覆盖不足。** 核心 claim 同时依赖分辨率、位置编码和数据配方，现有实验无法精确归因。

### 5.3 个人判断

Qwen-VL 最具历史价值的部分不是某个单项分数，而是证明了“文字读取 + 定位 + 对话”可以共享一个纯生成接口。但固定 query 也意味着它更像一个把视觉压成语言可消费摘要的系统，而不是保留原生视觉结构的长期记忆体。这一矛盾直接推动了 Qwen2-VL 的动态分辨率路线。

---

## 6. 复现成本与实践建议

### 6.1 完整复现成本

完整复现需要约 1.5B 图文样本、第二阶段数千万细粒度样本、9.6B 参数模型与大规模分布式训练。论文未披露 GPU 型号、卡数和总 FLOPs，只给出 steps、batch 与并行配置，因此无法从原文精确换算成本。对普通实验室而言，从零复现不可现实；可行方案是基于公开权重做 LoRA/SFT 或缩小 ViT/LLM 与数据规模。

### 6.2 最小可验证复现

1. 使用公开 Qwen-VL 权重，在同一批图像上测试 224/448 重采样对小字与定位的影响。
2. 固定 ViT 与 LLM，只训练不同 query 数的 adapter，复现实验中的 64/144/256/400 loss 曲线，并补充 TextVQA、RefCOCO 精度和显存。
3. 对坐标输出做严格 parser，记录格式合法率、IoU、越界率和同义指称鲁棒性，不只看成功案例。
4. 若做领域适配，优先构造带区域坐标的 OCR/grounding 数据，而不是只添加整图 caption。

### 6.3 推理资源

9.6B 总参数以 BF16 仅权重约需 19 GB，实际还要加 KV cache、视觉编码器激活和框架开销；量化后可降低显存。视觉 token 固定 256，使单图上下文成本可预测，但多图会线性增加视觉 token 数。

---

## 7. 个人启发与后续问题

### 7.1 可迁移启发

- 若任务需要定位，先设计**可生成、可解析、可统一训练**的坐标协议，再决定是否需要专用 head。
- 视觉压缩器不能只关注语义，还要显式保留二维位置；否则 OCR 与 grounding 会首先退化。
- 课程学习中“低分辨率大规模对齐 → 高分辨率细粒度任务 → 指令对齐”的顺序兼顾成本与能力。
- 纯文本混训不是附属项，而是避免多模态训练损害语言底座的必要约束。

### 7.2 后续问题

1. 固定 256 query 的真实 Pareto 前沿是什么：精度、延迟、显存和输入复杂度如何共同变化？
2. 2D 绝对位置编码在跨分辨率、长宽比变化时能否稳定外推？
3. 生成式坐标是否能同时给出可靠置信度，并支持检测式 NMS？
4. OCR 与文档理解增益中，分辨率、合成数据、PDF/HTML 真值各占多少？

---

## 8. 图表索引

| 本笔记图片 | 原论文位置 | 用途 |
|---|---|---|
| `01-capabilities.jpg` | Figure 2, p.2 | 展示多图、OCR、定位与多轮交互 |
| `02-training-pipeline.png` | Figure 3, p.4 | 解释三阶段训练与冻结策略 |
| `03-benchmark-radar.png` | Figure 1, p.1 | 总览相对通用 baseline 的能力覆盖 |
| `04-query-ablation.png` | Figure 7, p.22 | 核验固定 256 query 的消融依据 |

---

## 9. 材料来源

本文全部事实与数字以本地官方 PDF 和 arXiv LaTeX source 为主，不以发布博客代替论文。逐项溯源见 [sources.md](assets/Qwen-VL%20-%20定位与文字读取的通用视觉语言模型/sources.md)。
