---
type: paper-note
status: done
domain: VLM
paper: "CogVLM: Visual Expert for Pretrained Language Models"
year: 2024
arxiv: 2311.03079
doi: null
source: https://arxiv.org/abs/2311.03079
project: null
code: https://github.com/THUDM/CogVLM
tags:
  - VLM
  - visual-expert
  - deep-fusion
  - visual-grounding
created: 2026-10-15
updated: 2026-10-15
---

# CogVLM：视觉专家驱动的深层融合

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | CogVLM: Visual Expert for Pretrained Language Models |
| 作者 | Weihan Wang 等，清华大学与智谱 AI |
| 发表 | ICML 2024 |
| arXiv | 2311.03079 |
| 官方论文 | https://arxiv.org/abs/2311.03079 |
| 官方代码与权重 | https://github.com/THUDM/CogVLM |
| 一句话结论 | CogVLM 在冻结语言路径的同时，为每层视觉 token 配置独立 QKV 与 FFN“视觉专家”，以约 6.5B 可训练参数换取深层视觉语言融合，兼顾原始 NLP 能力、通用 VQA 与精细 grounding。 |

---

## 1. 研究背景

### 1.1 浅层对齐与全量微调之间的两难

BLIP-2、MiniGPT-4 一类浅层对齐方法只在输入端用 Q-Former 或线性层把图像特征映射到语言 embedding，LLM 本身冻结。它们收敛快、参数少，但视觉 token 在经过多层纯文本权重变换后会逐渐偏离各深层的预期分布；任务格式、描述长度和视觉细节都只能挤进输入前缀。另一条路线直接微调 LLM，视觉能力更强，却会让在大规模纯文本上形成的语言能力遭遇灾难性遗忘。

CogVLM 先给出实验证据：直接用 LAION 多模态预训练语言部分时，训练 loss 持续下降，MMLU 却从约 47 快速跌到 2,500 step 的 24.9 附近。也就是说，多模态训练目标变好不代表通用语言能力被保留。论文的问题由此变成：能否冻结原语言模型，使纯文本输入行为不变，同时在每一层为视觉 token 提供足够的可训练容量？

![](assets/CogVLM%20-%20视觉专家驱动的深层融合/figures/02-language-degradation.png)

> **Figure 3，PDF 第 3 页。** 橙色训练损失稳定下降，蓝色 MMLU 却急剧降至约 25，直接展示“拟合图文数据”与“保留语言知识”并不等价。该图是 visual expert 设计的动机证据，而不是最终 CogVLM 的训练曲线。来源：https://arxiv.org/abs/2311.03079

### 1.2 核心贡献

CogVLM-17B 从 Vicuna-1.5-7B 与 EVA2-CLIP-E 构建。文本 token 继续走冻结的原 QKV/FFN；图像 token 在每个 LLM layer 走新建、可训练、由语言权重初始化的视觉 QKV/FFN。两类 token 仍在同一个 attention 矩阵中交互，因此融合发生在网络深度方向，而不是只在输入端。作者还引入统一视觉 position id、大规模 caption+grounding 两阶段预训练和专门 grounding SFT，并公开权重与数据。

![](assets/CogVLM%20-%20视觉专家驱动的深层融合/figures/01-demo.png)

> **Figure 2，PDF 第 2 页。** 完整示例覆盖 OCR-free reasoning、详细描述、图表推理、meme、世界知识、视觉推理、带框字幕、指代表达理解和 grounded VQA。它展示的不是单一聊天风格，而是同一模型在开放文本与坐标输出之间切换的能力；同时这些均为成功样例，不能替代失败率分析。来源：https://arxiv.org/abs/2311.03079

---

## 2. 任务定义与方法

### 2.1 输入、输出与组件

输入由图像和文本 query 组成。EVA2-CLIP-E（移除专门聚合 `[CLS]` 的最后一层）产生 patch token，二层 SwiGLU MLP adapter 映射到 Vicuna hidden size；文本经原 word embedding。视觉 token 放在文本前，所有视觉 token 在 LLM RoPE 中共享同一个 position id，文本从 1 开始递增。输出统一为自回归文本，也可把归一化坐标离散成 `[[x0,y0,x1,y1]]` 完成 grounding。

共享视觉 position id 的动机是：几百到上千视觉 token 若沿用连续 RoPE，会让后续 query 与图像上部 token 距离过远并产生衰减，同时不必要地重复 ViT 内已有的二维位置信息。该设计让 LLM 将整幅图视为同一位置段，空间关系主要由 ViT 表征承载。

### 2.2 每层视觉专家

设 attention 输入 $X=[X_I;X_T]$，分别为图像与文本 hidden state。CogVLM 对两种 token 使用不同投影，再合并参与同一 causal attention：

$$Q=\operatorname{concat}(X_IW_I^Q,X_TW_T^Q),\quad K=\operatorname{concat}(X_IW_I^K,X_TW_T^K),\quad V=\operatorname{concat}(X_IW_I^V,X_TW_T^V)$$

$$\operatorname{Attention}(X)=\operatorname{softmax}\left(\frac{\operatorname{Tril}(QK^\top)}{\sqrt{D}}\right)V$$

FFN 同样按 token 类型路由：

$$\operatorname{FFN}(X)=\operatorname{concat}(\operatorname{FFN}_I(X_I),\operatorname{FFN}_T(X_T))$$

$W_T$ 与 $\operatorname{FFN}_T$ 是冻结 Vicuna 参数；$W_I$ 与 $\operatorname{FFN}_I$ 是视觉专家，由对应语言权重初始化并训练。注意力仍允许图文 token 互相作用，所以这不是互不通信的双塔；“专家”只负责按模态选择变换参数。

![](assets/CogVLM%20-%20视觉专家驱动的深层融合/figures/03-architecture.png)

> **Figure 4，PDF 第 3 页。** 左侧是 ViT→MLP→视觉 token 与文本 token 拼接；右侧紫色视觉 QKV/FFN 与黑色冻结语言 QKV/FFN并行。只有紫色参数训练。它在保持纯文本路径完全不变的同时，让视觉 token 在每一层都能重新对齐当前语义空间。来源：https://arxiv.org/abs/2311.03079

### 2.3 为什么是深层融合

论文把 shallow adapter 类比为 P-Tuning：只学习输入前缀，后续全部依赖冻结模型；视觉专家更接近 LoRA 的逐层适配，只是为视觉 token 提供全尺寸专用参数。它增加约一倍参数存储，但由于每个 token 只走对应模态的一套 QKV/FFN，作者称 FLOPs 不随双份权重同比翻倍。代价是训练参数从 BLIP-2 级别的百 M 上升到约 6.5–6.6B，参数效率显著降低。

---

## 3. 训练数据与训练流程

### 3.1 两阶段预训练

第一阶段使用 LAION-2B 与 COYO-700M 清洗后约 15 亿图文对，训练 caption next-token prediction 120,000 iteration，batch size 8,192，分辨率 $224\times224$。第二阶段混合 caption 与 REC：给定对象描述生成归一化框坐标，训练 60,000 iteration，batch size 1,024；最后 30,000 iteration 将分辨率从 $224\times224$ 提升到 $490\times490$。

作者另构建 4,000 万 grounding 图像：从 LAION-115M caption 抽 noun phrase，用 GLIPv2 预测框，并筛到超过 75% 图像含至少两个框。回答坐标量化为 000–999。两阶段只对 answer/text token 计算 next-token loss；可训练参数约 6.5B，论文消融表按 6.6B 记载，属于统计口径差异。

预训练超参数为 AdamW，$\beta=(0.9,0.95)$，weight decay 0.05，dropout 0.1；第一/二阶段学习率分别 $10^{-4}$/$10^{-5}$，warmup 12,000/1,200 step，cosine decay。

### 3.2 Instruction alignment

CogVLM-Chat 混合 VQAv2、OKVQA、TextVQA、OCRVQA、ScienceQA、LLaVA-Instruct、LRV-Instruction、LLaVAR 等公开数据。作者指出原始 LLaVA-Instruct 的 language-only GPT-4 流水线含错误，并进行了人工检查与修正。短答案任务使用 `Question: Short answer:`，长回答使用 `Question: Answer:`，减少回答风格冲突。

SFT 训练 6,000 iteration，batch size 1,024，学习率 $10^{-5}$，分辨率 $490^2$；视觉编码器此阶段解冻，学习率为其余参数的十分之一。CogVLM-Grounding 另外混合 grounded captioning、REG、REC 与 grounded VQA，来源包括 Flickr30K Entities、RefCOCO 系列、Visual7W、Visual Genome 与 Grounded CoT-VQA。

### 3.3 复现成本

官方代码、权重与 SFT 数据公开，可复现推理与下游微调；但从头训练需要 15 亿图文对、4,000 万伪框图像、180k 大 batch 更新与 6.5B 训练参数。论文正文没有披露完整预训练的硬件数量、墙钟时间和能耗，不能凭仓库配置反推为论文事实。与 LLaVA/BLIP-2 相比，CogVLM 用更高训练成本换取深层容量；研究组更现实的路径是从官方 checkpoint 做 SFT 或模块消融。

---

## 4. 实验与消融

### 4.1 通用 VQA/LVLM 结果

论文与 MiniGPT-4、IDEFICS-Instruct、OpenFlamingo、InstructBLIP、Qwen-VL、LLaVA-1.5、mPLUG-Owl2、Emu2 等 generalist 模型比较。星号表示相应任务出现在 SFT 中，因而不能被解读为零样本结果。

| 模型 | LLM | VQAv2 | OKVQA | TextVQA | MM-Vet | MMBench | LLaVA-Bench | POPE | MMMU | MathVista |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| InstructBLIP | Vicuna-7B | 未列 | 未列 | 50.1 | 26.2 | 33.9 | 59.8 | 53.8 | 未列 | 25.3 |
| LLaVA-1.5 | Vicuna-13B | 80.0* | 未列 | 61.3* | 35.4 | 67.7 | 64.6 | 85.9 | 33.6 | 26.1 |
| Emu2-Chat | LLaMA-33B | **84.9*** | **64.8*** | 66.6* | 48.5 | 63.6 | 56.4 | 未列 | 34.1 | 未列 |
| CogVLM-Chat | Vicuna-7B | 82.3* | **64.8*** | **70.4*** | **51.1** | **77.6** | **77.8** | **87.9** | **41.1** | **34.5** |

CogVLM 并非每项都第一：VQAv2 低于 Emu2-Chat 84.9，但在 OCR/TextVQA、综合 LVLM benchmark 与 grounding 上优势明显。grounding 方面，CogVLM-Grounding 在 RefCOCO val/test-A/test-B 为 92.76/94.75/88.99，RefCOCO+ 为 88.68/92.91/83.39，RefCOCOg val/test 为 89.75/90.79，Visual7W 为 91.05；九个 split 中五个超过列出的 specialist SOTA。

![](assets/CogVLM%20-%20视觉专家驱动的深层融合/figures/04-benchmark-overview.png)

> **Figure 1，PDF 第 1 页。** 雷达图汇总 VQA、OCR、幻觉、数学、多学科理解与三组 RefCOCO grounding。紫色 CogVLM 在多数轴外扩，但各轴量纲和是否经过任务 SFT 不一致，因此它适合展示覆盖面，不适合作为跨任务平均分。来源：https://arxiv.org/abs/2311.03079

### 4.2 视觉专家消融

官方消融统一训练 6,000 iteration、batch size 8,192。完整 CogVLM 在 COCO/NoCaps CIDEr 与 OKVQA/TextVQA/VQAv2 上为 142.8/120.1/59.3/45.3/80.0。

| 结构/训练设置 | 可训练参数 | COCO | NoCaps | OKVQA | TextVQA | VQAv2 |
|---|---:|---:|---:|---:|---:|---:|
| 仅 MLP adapter | 140M | 131.2 | 111.5 | 55.1 | 40.7 | 73.8 |
| LLM + MLP adapter | 6.9B | 140.3 | 118.5 | 56.8 | 44.7 | 78.9 |
| 每 4 层 full VE | 1.7B | 138.7 | 117.4 | 58.9 | 44.1 | 77.6 |
| 每层仅 VE-FFN | 4.4B | 140.0 | 118.7 | 58.2 | 45.1 | 78.6 |
| 完整 CogVLM | 6.6B | **142.8** | **120.1** | **59.3** | 45.3 | **80.0** |

这个表支持三点。第一，140M 浅 adapter 在所有任务明显落后。第二，在可训练参数相近时，冻结语言路径并训练视觉专家优于直接训练 LLM+adapter，尤其 OKVQA 等依赖知识的任务，说明保留语言权重有价值。第三，FFN-only 与稀疏每四层专家已取得大部分收益，提示 full expert 不是唯一性价比点。

其他消融并非全部支持复杂化：视觉专家随机初始化比从 LLM 初始化差；visual token 用 full attention mask 略差于 causal mask；增加 image SSL loss 没有改善综合结果；4.4B EVA2-E 换 300M EVA2-L 在多数任务只小幅下降，但 TextVQA 下降约 2.5；EMA 对若干任务有益。这些反直觉结果表明大视觉编码器与额外视觉损失的边际收益有限，深层 token-specific 参数路由才是主要增益来源。

---

## 5. 局限与失败案例

### 5.1 论文披露边界

论文没有独立 limitations 或 failure-cases 章节，也没有展示有代表性的错误输出分布；Figure 2 全为成功案例。因此对错误类型、拒答、置信度、对抗提示和长对话退化的证据不足。结论声称 17 个 benchmark SOTA，但部分任务已参与 SFT，表格中的星号需要逐项阅读，不能统称为零样本泛化。

### 5.2 结构、数据与复现限制

视觉专家把可训练参数提升到约 6.5B，已远离“轻量桥接”。虽然每 token 只走一套专家使 FLOPs 不按参数翻倍，但显存、优化器状态、checkpoint 与通信成本仍高。15 亿网络图文对和 4,000 万 GLIPv2 伪框会继承抓取噪声、伪标签误差与数据偏差；论文只说明清洗原则，未公开可一键重建的完整预训练语料快照。

所有视觉 token 共用一个 RoPE position id 避免距离衰减，却把二维空间责任完全交给 ViT 表征；当视觉 token 与文本经过很多层交互后，这种做法是否限制更复杂空间推理，论文未做针对性消融。模型仍使用 causal visual mask，虽实验优于 full mask，但机制解释只是作者假设。

### 5.3 Future work

结论明确列出更好的 SFT alignment、RLHF 与 anti-hallucination。结合论文缺失项，未来还需要独立纯文本回归测试以严格证明语言能力完全保持、按任务拆分的未见数据评测、训练数据可审计性，以及视觉专家在更小参数预算下的稀疏化。后两项是基于实验的个人判断，不是作者原文承诺。

---

## 6. 与后续工作的关系

CLIP 只在最终 embedding 点积处融合；Flamingo 在冻结 LM 层间插入跨注意力；BLIP-2 用 Q-Former 把视觉压成软提示；LLaVA 以线性 projector 接入并更新 LLM；CogVLM 则把问题重新表述为 token-specific computation：视觉与文本共享序列和 attention 图，但在每层使用不同 QKV/FFN。

这种设计同时吸收 Flamingo 的“冻结语言能力”和 LLaVA 的“深层视觉适配”，代价是更多参数。它也预示后续 mixture-of-experts、多模态专用子层、视觉专家 LoRA 等路线：不必让所有 token 使用同一组语言权重，也不必为每个模态建立完全分离的网络。视觉 grounding 数据与坐标文本化则延续了 Pix2Seq、Kosmos-2、Shikra、Qwen-VL 等统一序列输出路线。

相较后续 LLaVA-OneVision 等通过数据与动态分辨率扩展多图/视频，CogVLM 更强调单图深层融合；论文未验证视频和原生多图上下文。这说明它解决的是“单幅图像 token 如何贯穿语言网络”，而不是所有多模态序列问题。

---

## 7. 个人判断

CogVLM 最有说服力的部分是把两种失败路线放进同一框架：浅 adapter 保住语言却视觉不足，直接调 LLM 视觉增强却语言遗忘；视觉专家通过按 token 分流参数，允许图文在同一 attention 中通信，同时保护文本路径。完整专家相对 LLM+adapter 的提升并不巨大，但跨 caption、VQA、OCR 与 grounding 一致，说明深层模态专用容量确有价值。

它的代价也很明确：6.5B 可训练参数、15 亿图文对和高分辨率第二阶段让“冻结 LLM”不等于低成本。官方 checkpoint 使使用与 SFT 可行，从头复现则接近大型机构工程。论文缺乏失败案例和严格 NLP 回归表，因而“完全不牺牲 NLP”更多由结构保证推断，而不是被广泛文本 benchmark 充分证实。综合判断：CogVLM 是从浅连接器转向深层视觉专家的重要节点，适合高要求 OCR/grounding/视觉知识任务，但其参数效率、数据可审计性和多图视频扩展仍留给后续工作。