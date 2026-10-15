---
type: paper-note
status: complete
domain: VLM
paper: "Kimi-VL Technical Report"
year: 2025
arxiv: "2504.07491"
doi: null
source: "https://arxiv.org/abs/2504.07491"
project: null
code: "https://github.com/MoonshotAI/Kimi-VL"
tags:
  - VLM
  - mixture-of-experts
  - native-resolution
  - long-context
  - multimodal-reasoning
  - agent
created: 2026-10-15
updated: 2026-10-15
---

# Kimi-VL：高效多模态 MoE 与长思考

## 基本信息

- **基础模型**：MoonViT 400M + 两层 MLP projector + Moonlight MoE language decoder；语言侧总参数 16B、每 token 激活 2.8B，连同视觉侧约 3.2B activated。
- **上下文**：从 8K 逐级扩到 128K，联合训练长文本、长文档、长交错图文与长视频。
- **模型版本**：Kimi-VL-A3B-Instruct；经 long-CoT SFT 与 RL 的 Kimi-VL-A3B-Thinking；进一步整合感知、视频、长文档与 OS agent 能力的 Thinking-2506。
- **重要性判断**：★★★★★。它把原生分辨率、稀疏 MoE、128K 多模态上下文、长思考和电脑 Agent 放进一个激活仅约 3B 的统一模型中，是“低激活参数但高能力覆盖”的代表。

**一句话结论：**

Kimi-VL 的核心不是某个单点 benchmark，而是效率组合：**MoonViT 不切固定方块、按原分辨率打包视觉 patch；MoE decoder 以 16B 总容量换 2.8B 激活计算；联合预训练和 128K 激活保留语言/多模态长上下文；long-CoT SFT + online policy mirror descent 再把有限激活参数转化为更长的测试时推理。**

---

## 1. 研究背景与问题定义

### 1.1 目标矛盾

2025 年高性能开源 VLM 多采用 dense decoder、固定尺寸/切图式视觉编码器，长上下文、Agent 和 long-CoT 往往分散在不同模型。Kimi-VL 要解决的矛盾是：在激活参数和推理成本接近 3B 模型的情况下，同时保持 16B 级参数容量、超高分辨率感知、128K 多模态上下文、GUI 操作和长思考。

它的思路不是极端压缩某一环，而是把效率分配到三个层面：视觉侧避免无谓 resize/crop；语言侧只激活少量专家；难题才通过更长 thinking token 使用额外测试时计算。

### 1.2 输入与输出

输入可为文本、任意尺寸单图、多图、视频帧、长文档页面或 GUI screenshot。MoonViT 将每张图分 patch 后展平，并把同 batch 的变长图像序列打包；MLP 将视觉特征映射为语言 embedding，与文本 token 一起输入 MoE decoder。输出统一为文本：问答、OCR/Markdown/LaTeX、数学 CoT、视频时间段描述、GUI grounding 坐标，以及 Agent 的 thought/action/API 序列。

Thinking 版本推理仍是标准单路自回归生成，不需要显式树搜索或并行规划器；额外计算主要来自更长输出序列。

---

## 2. 模型架构

![](assets/Kimi-VL%20-%20高效多模态MoE与长思考/figures/01-architecture.png)

> 图 3：Kimi-VL 的 MoonViT–MLP–MoE 架构。不同尺寸图像、长视频、OCR 长条图和 GUI screenshot 都按原始宽高比进入 MoonViT；视觉 token 投影后与文本 token 共同驱动稀疏专家 decoder。来源：论文 Figure 3，第 3 页。

### 2.1 MoonViT：真正的原生分辨率视觉编码器

MoonViT 初始化自 SigLIP-SO-400M，但取消“所有图像 resize 到固定方形”或“先切若干固定 tile 再拼接”的强约束。它借鉴 NaViT：图像划分 patch、展平为 1D，多个不同分辨率样本通过 variable-length packing 拼在同一 batch，注意力由 FlashAttention 的变长序列机制执行。

空间位置同时使用两类编码：插值后的 SigLIP 固定绝对位置 embedding 保留初始化能力；沿高度和宽度的 2D RoPE 提升高分辨率外推和细粒度位置表达。两者配合，使模型在同一 batch 内处理小公式、超宽 OCR、长视频帧和大 GUI screenshot，而不需要 InternVL 式预定义宽高网格。

“原生分辨率”不等于零预处理：仍要 patchify、flatten、pack，也有最大像素预算。原版最大像素较低，Thinking-2506 继续训练 MoonViT 后支持单图最多 3.2M pixels，是原上限 4 倍。

### 2.2 MLP projector 与视觉 token 压缩

MoonViT 特征先做 $2\times2$ pixel shuffle：空间宽高各减半，token 数约降为四分之一，通道相应扩张；再经过两层 MLP 投影到 LLM embedding 维度。该 projector 很简单，意味着跨模态能力主要由 MoonViT 预训练与后续全参数联合训练承担。

### 2.3 MoE language decoder

语言模型来自 Moonlight 中间 checkpoint：已看过 5.2T 纯文本 token、上下文 8K。其结构近似 DeepSeek-V3 风格，包含 shared 与 non-shared experts；论文未披露完整专家数、top-k 和负载均衡超参，因此不应自行补齐。可核实的是总参数 16B、每 token 激活 2.8B。

对 token $h_t$，MoE 的抽象形式可写为：

$$\operatorname{MoE}(h_t)=\sum_{e\in\operatorname{TopK}(g(h_t))}\alpha_{t,e}E_e(h_t)+E_{shared}(h_t)$$

其中 router $g$ 为每个 token 选择少量专家，$\alpha_{t,e}$ 是归一化权重。容量由所有专家参数提供，而单 token 只支付激活专家成本。注意“2.8B activated”只描述 decoder 活跃参数，不等于模型总显存仅需 2.8B：部署仍需存放 16B 总权重，并承担 expert routing/communication。

### 2.4 Muon 优化器与并行

增强版 Muon 加入 weight decay、按参数调整 update scale，并实现 ZeRO-1 分布式版本；用于 ViT、projector、LLM 全参数训练。系统采用 DP、EP、PP、CP 四维并行，配合 ZeRO-1 与 selective activation checkpointing。论文称训练吞吐比 7B dense VLM 高约 60%，但未给出统一硬件、batch 和序列组成的完整对账，这一数字应视作内部系统结果。

---

## 3. 预训练：视觉先对齐，再全模型联合扩展

![](assets/Kimi-VL%20-%20高效多模态MoE与长思考/figures/02-pretraining-stages.png)

> 图 4：在 Moonlight 5.2T 纯文本预训练之后，Kimi-VL 再消费 4.4T token：2.1T ViT、1.4T joint pretrain、0.6T cooldown、0.3T long-context。来源：论文 Figure 4，第 4 页。

### 3.1 阶段一：MoonViT 训练与对齐

MoonViT 首先用 alt text、synthetic caption、grounding bbox 和 OCR 目标训练 2T token。目标结合 SigLIP 对比损失与图像条件 caption loss：

$$\mathcal{L}=\mathcal{L}_{\mathrm{SigLIP}}+\lambda\mathcal{L}_{\mathrm{caption}},\quad\lambda=2$$

随后用 0.1T token 把 MoonViT 对齐到 MoE LLM，仅更新 MoonViT 与 MLP。该阶段降低视觉 embedding 刚接入 LLM 时的 perplexity，避免全模型联合训练一开始被随机 projector/跨模态分布拖累。

### 3.2 阶段二：Joint pre-training

再训练 1.4T token，混合与 Moonlight 同分布的纯文本和 caption、交错图文、OCR、知识、视频、Agent 等多模态数据。最初若干 step 只用文本，之后逐步提高多模态比例；ViT 与 LLM 均更新。这个 progressive schedule 的目标是避免视觉数据突然扰动语言能力。

### 3.3 阶段三：Joint cooldown

0.6T token 使用高质量文本与多模态数据。文本侧增加合成数学、知识和代码 QA，经 rejection sampling 验证；多模态侧重写学术视觉数据为 QA、回放高质量子集。作者刻意控制 QA 比例，避免模型过拟合固定问答格式。

### 3.4 阶段四：128K 长上下文激活

0.3T token 将上下文从 8K 通过两个 4× 子阶段扩到 128K，RoPE inverse frequency 从 50,000 调到 800,000。每阶段 25% 为长数据，75% 回放前阶段短数据，以减少短上下文退化；长数据同时包含长文本、交错图文、长视频与长文档。

NIAH 中，文本/视频在 65,536 token 前均为 100%；65,536–131,072 区间降到 87.0%/91.7%。这说明 128K“可用”但不是全区间无损，也印证论文自己承认的 attention 有效容量限制。

---

## 4. 数据构造

### 4.1 六类多模态预训练数据

1. **Caption**：开放与内部中英文数据，严格去重和图文相关性过滤；限制 synthetic caption 比例，降低幻觉。
2. **Interleaving**：教材、网页、教程及开放交错数据，强调图文顺序修复，用于多图、知识和长上下文。
3. **OCR**：多语言、密集排版、网页、手写、表格、公式、图形和多页文档；加入旋转、畸变、颜色和噪声增强。
4. **Knowledge**：教材、论文和互联网知识，经 layout parser 与 OCR 处理，补充抽象图表、几何图和 infographic。
5. **Video**：短视频细粒度时空对应与长视频密集 caption；同样限制 synthetic dense description 比例。
6. **Agent**：Desktop/Mobile/Web screenshot-action、图标、dense grounding 与连续轨迹；人工收集多步电脑任务，再合成 CoT。

### 4.2 SFT 与 reasoning 数据

普通 SFT 中 text token:image token 约 1:1。非推理任务由人工 seed→seed model 多候选→人工排序与精修构造；数学/科学/视觉编程使用规则或模型 verifier 做 rejection sampling。

Reasoning 数据从有 ground truth 的多步问题出发，用 Kimi k1.5 采样多条长轨迹，结合 off-the-shelf reward model 和规则过滤错误 CoT。这里存在 teacher bias：长思考风格可能继承 k1.5 的解题习惯，论文没有分析多样性与错误模式。

---

## 5. 后训练：Instruct、Thinking、RL 与 Agent

![](assets/Kimi-VL%20-%20高效多模态MoE与长思考/figures/03-posttraining-stages.png)

> 图 5：基础模型先在 32K、128K 做两轮 joint SFT，得到 Instruct；再用 long-CoT SFT 激活规划/评估/反思/探索，最后 online RL 得到 Thinking。来源：论文 Figure 5，第 6 页。

### 5.1 两阶段 Joint SFT

LLM、MLP、MoonViT 全部优化，只在 answer 与特殊 token 上计算监督，system/user prompt mask 掉。先 32K 一 epoch，学习率 $2\times10^{-5}\rightarrow2\times10^{-6}$；再 re-warmup 到 $10^{-5}$，以 128K 训练一 epoch并降到 $10^{-6}$。packing 多个样本提高利用率。

### 5.2 Long-CoT SFT

小而高质量的 warmup 数据显式包含 planning、evaluation、reflection、exploration。它的作用不是最终能力上限，而是让 policy 在进入 RL 前学会产生可奖励的结构化思考，避免在线训练从短答案分布盲目探索。

### 5.3 Online policy mirror descent

第 $i$ 轮把当前 policy $\pi_{\theta_i}$ 作为 reference，最大化二元正确性 reward，同时用 KL 限制更新：

$$\max_\theta\mathbb{E}_{(x,y^*)\sim\mathcal{D}}\left[\mathbb{E}_{(y,z)\sim\pi_\theta}[r(x,y,y^*)]-\tau\operatorname{KL}(\pi_\theta\|\pi_{\theta_i})\right]$$

$r\in\{0,1\}$ 根据 ground truth 判断答案正确。另加 length-based reward 惩罚过长响应，抑制 overthinking；curriculum sampling 和 prioritized sampling 使用难度与单题成功率，把训练集中到有学习信号的样本。

这套 RL 主要适合可验证任务。开放式视觉问答、Agent 安全性和主观写作没有可靠二元 reward，论文未证明同样受益。

### 5.4 Agent 能力从哪里来

Agent 不是仅靠长 CoT 自发出现。预训练阶段已有 GUI screenshot/action、图标、连续轨迹；SFT 有 grounding、规划和多轮交互；人工电脑任务轨迹还带合成 CoT。模型输出 thought、action 与 API call，环境执行后再把新 screenshot 放回上下文。

Kimi-VL-A3B-Instruct 在 ScreenSpot-V2/Pro 为 92.8/34.5，OSWorld 8.22，WindowsAgentArena 10.4。单步定位很强，但多步真实完成率仍低，说明 grounding 并不等于可靠 Agent。Thinking-2506 的 OSWorld-G 52.5 是 grounding split，不应与完整 OSWorld 交互成功率直接混用。

---

## 6. 量化结果与 baseline

### 6.1 Instruct 模型

与高效 baseline 比，Kimi-VL-A3B（2.8B decoder + 0.4B vision activated）在 24 项中论文称胜过 Qwen2.5-VL-7B 19 项。代表值如下：

| 任务 | Kimi-VL-A3B | Qwen2.5-VL-7B | GPT-4o | 解读 |
|---|---:|---:|---:|---|
| MMMU | 57.0 | 58.6 | 69.1 | 学术能力接近同级 dense，但未追上大闭源 |
| MathVista | 68.7 | 68.2 | 63.8 | 短推理视觉数学较强 |
| MathVision | 21.4 | 25.1 | 30.4 | 难题暴露参数/思考深度不足 |
| InfoVQA | 83.2 | 82.6 | 80.7 | 原生高分辨率与 OCR 数据有效 |
| OCRBench | 867 | 864 | 815 | 文档/文字感知优势明显 |
| OSWorld | 8.22 | 2.5 | 5.03 | 相对提升大，绝对成功率仍低 |
| MMLongBench-Doc | 35.1 | 29.6 | 42.8 | 128K 与多页 OCR 有效但仍落后 GPT-4o |
| LongVideoBench | 64.5 | 56.0 | 66.7 | 接近大闭源，长视频是强项 |

### 6.2 Thinking 与 Thinking-2506

初版 Thinking 相对 Instruct 在 MathVista +2.6、MMMU +4.7、MathVision +15.4。2506 进一步达到 MathVision 56.9、MathVista 80.1、MMMU 64.0、MMMU-Pro 46.3、VideoMMMU 65.2。相比初版 Thinking，输出也更短：MMMU 平均约 2.9K→2.4K token，MathVision 5.8K→4.4K。

Thinking-2506 还修复了“会思考但普通感知退化”：MMBench 84.4、MMStar 70.4、MMVet 报告正文写 78.4（表中为 78.1，存在 0.3 的文本/表格不一致）、ScreenSpot-Pro 52.8、MMLongBench-Doc 42.1。它不是单纯在初版后多训 RL，还继续训练 MoonViT到 3.2M pixels，并整合 Instruct 的感知、视频、文档与 Agent 数据。

---

## 7. 消融与测试时扩展

### 7.1 思考长度消融

![](assets/Kimi-VL%20-%20高效多模态MoE与长思考/figures/04-test-time-scaling.png)

> 图 13：增加最大 thinking token 通常提高准确率，但不同 benchmark 饱和点不同。MathVision 从 1K 的 18.7% 升到 16K 的 36.8%；MathVista 约 4K 后基本饱和。来源：论文 Figure 13，第 18 页。

具体地，MathVision 在 1K/2K/4K/8K/16K 为 18.7/22.6/29.0/34.0/36.8；MMMU 为 49.2/52.4/56.2/60.1/61.7；MathVista 为 66.7/69.0/70.9/70.6/71.3。它证明测试时计算有效，也证明固定给 16K 并不经济：MathVista 4K 后甚至有小幅波动。

### 7.2 长上下文消融/诊断

NIAH 不是组件消融，而是上下文能力诊断：到 64K 完全召回，64K–128K 降到文本 87.0、视频 91.7。它提示 advertised 128K context window 的边缘可靠性有限，应在真实文档问答中配合分块或检索，而不是默认全量塞入。

### 7.3 缺失的结构消融

论文没有给出 MoonViT 原生分辨率 vs 固定切图、dense vs MoE、2D RoPE vs 仅插值绝对位置、各数据类别移除、Muon vs AdamW 的严格消融。这是报告的主要证据缺口。当前只能从跨模型结果看到系统整体有效，不能量化每个组件的独立贡献。

---

## 8. 局限与未来工作

### 8.1 作者明确列出的局限

1. 约 3B activated 的规模对高度专业、领域特化和强依赖语言能力的极复杂问题仍不足。
2. 多步推理尚未达到上限，深层上下文理解仍需改进。
3. 虽支持 128K，但 attention 层容量只相当于约 3B 模型，极长序列或高信息密度场景仍不够。

### 8.2 额外可观察的限制

MoE 降低每 token 计算但不降低总权重存储，EP 还引入 all-to-all 通信；低并发或单卡部署不一定获得论文中的吞吐优势。原生分辨率会让视觉 token 数随像素增长，3.2M pixel screenshot 的 prefill 与 KV 成本仍高。RL 依赖有 ground truth 的二元 reward，开放域 Agent 可能出现 reward blind spot。数据包含大量内部 caption、OCR、视频、Agent 与 VM 轨迹，外部团队难以严格复现。

### 8.3 作者未来计划

论文明确提出扩大模型规模、扩展预训练数据、改进 post-training 算法，并优化/发布更大版本；同时继续改善 test-time scaling 和 thinking model。合理的研究延伸还包括动态 thinking budget、MoE expert specialization 可解释性、原生分辨率 token budget 控制，以及在真实 Agent 环境中联合优化视觉 grounding 与长 horizon 成功率。

---

## 9. 复现成本与实践建议

### 9.1 推理

虽然每 token 仅激活约 3.2B 参数，16B decoder 全权重仍需驻留；bf16 仅权重约 32GB，再加 400M ViT、KV cache、router buffer 和长视觉序列，单卡消费级部署并不轻松。量化可降低显存，但论文未提供量化精度。128K/高分辨率场景需要按实际 token 预算评估，而不是只按 active params。

### 9.2 训练

从头复现包含 Moonlight 5.2T 文本前置训练，以及 Kimi-VL 的 4.4T token 后续训练，远超普通实验室。更现实的路线是使用公开 checkpoint：

1. 验证 MoonViT 对不同宽高比输入的 token 数与 OCR 表现；
2. 在固定问题集上复现 thinking length 曲线，同时记录 token、延迟与准确率；
3. 以可验证数学/科学问题做小规模 long-CoT SFT + KL-regularized online RL；
4. Agent 评测必须区分 grounding 与真实环境 pass@1。

### 9.3 复现风险

内部数据与 Kimi k1.5 teacher 是最大不可复现项；其次是 4D parallel、Muon 分布式实现和长上下文数据 packing。即使代码与权重公开，也不能从模型发布反推完整训练配方的样本级细节。

---

## 10. 个人判断与后续问题

### 10.1 最有价值的设计

我认为 MoonViT 的价值高于“又一种视觉塔”：它把视觉 batch 组织向 LLM 的变长 token packing 靠拢，使任意分辨率训练不再依赖复杂切图规则。MoE 则把大容量与低激活计算结合；二者共同让长视频、文档和 GUI 成为同一种变长序列问题。

Long-CoT/RL 的关键结果不是 16K 一定更好，而是任务存在不同的计算饱和点。MathVista 4K 后饱和，MathVision 到 16K仍增长，说明下一步应学习“何时停止思考”，而不是给所有问题固定最大 budget。2506 输出缩短约 20% 的结果比单纯提分更接近可部署方向。

### 10.2 尚未回答的问题

1. MoonViT 相对同 token budget 的动态切图方案，优势来自原生几何还是更多有效像素？
2. 视觉、文本和 Agent token 在 MoE 中是否形成专家分工，router 是否有模态偏置？
3. 128K 边缘召回下降是位置外推、attention 容量还是训练长样本质量造成？
4. length penalty 如何避免过早截断必要推理？
5. 高 grounding 分数如何转化为真实 OSWorld 长 horizon 成功率与安全性？

### 10.3 最终结论

Kimi-VL 是一套围绕“激活效率”组织的多模态系统：16B 总容量、2.8B decoder 激活；任意分辨率视觉输入但通过 pixel shuffle控 token；128K 上下文但保留短数据回放；难题允许长思考，简单任务则避免无意义计算。它的强项是统一覆盖与工程整合，弱项是关键架构消融不足、内部数据难复现，以及 advertised 低激活参数容易掩盖总显存和长上下文成本。

---

## 11. 材料来源

- 官方论文：<https://arxiv.org/abs/2504.07491>
- 本地 PDF：`assets/Kimi-VL - 高效多模态MoE与长思考/sources/paper.pdf`
- 来源与 Figure/Page 记录：`assets/Kimi-VL - 高效多模态MoE与长思考/sources.md`
- 官方代码：<https://github.com/MoonshotAI/Kimi-VL>
