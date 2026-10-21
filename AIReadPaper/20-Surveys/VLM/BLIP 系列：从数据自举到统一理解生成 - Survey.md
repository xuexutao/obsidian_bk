---
type: survey
status: complete
domain: VLM
series: BLIP
period: "2022-2025"
created: 2026-10-20
updated: 2026-10-20
tags:
  - VLM
  - BLIP
  - Q-Former
  - instruction-tuning
  - unified-multimodal
---

# BLIP 系列：从数据自举到统一理解生成

## 基本信息

本页是 BLIP 系列的统一入口，按正式论文与技术连续性组织：

1. [[BLIP - 生成式字幕过滤的统一视觉语言预训练]]
2. [[BLIP-2 - 冻结底座的高效视觉语言桥接]]
3. [[InstructBLIP - 指令感知视觉特征抽取]]
4. [[BLIP-3 - 开放大规模多模态模型家族]]
5. [[BLIP3-o - 全开放统一多模态理解与生成]]

其中 BLIP3-o 是 BLIP-3/xGen-MM 的统一理解—生成分支，不应简单称为 BLIP-4。

---

## 1. 系列主线不是同一架构变大，而是瓶颈连续迁移

| 时间 | 工作 | 当时的主要瓶颈 | 核心解法 | 新暴露的问题 |
|---|---|---|---|---|
| 2022 | BLIP | Web 图文对噪声大，理解和生成各自建模 | MED + CapFilt，生成 caption 再过滤 | 大模型连接成本高，视觉—语言桥不够参数高效 |
| 2023 | BLIP-2 | 视觉 encoder 与 LLM 都很强，但端到端训练昂贵 | 冻结两端，用 32 query Q-Former 做信息瓶颈 | 固定 query 未感知具体任务，视觉证据选择不够针对性 |
| 2023 | InstructBLIP | 同一图像在不同指令下需要不同视觉证据 | instruction 同时输入 Q-Former 与 LLM | 仍以理解/文本生成为主，开放数据与大规模 recipe 不够完整 |
| 2024/2025 | BLIP-3 / xGen-MM | 开源 VLM 缺少从数据到权重的完整规模化配方 | 多阶段训练、开放数据、任意分辨率与交错多图 | 只理解仍不足以统一多模态输出 |
| 2025 | BLIP3-o | 图像理解和连续图像生成难在同一系统兼容 | 冻结 Qwen2.5-VL，CLIP feature + Flow Matching DiT | 两级扩散成本高，生成尚未反哺理解，8B 数据不完全公开 |

这一演化说明，“BLIP 系列”不是一个骨干的版本号升级。它每一代都换了第一矛盾：**数据质量 → 底座连接 → 指令条件视觉抽取 → 开放规模化 → 理解生成统一**。

---

## 2. 五篇论文的输入输出与接口演化

### 2.1 BLIP：先把训练数据变得可信

BLIP 的关键不是新视觉 backbone，而是 CapFilt：captioner 为 Web 图像生成更丰富描述，filter 删除图文不一致样本，再用清洗后的数据训练 MED。理解和生成共享大部分参数，但仍是 BERT 风格的 encoder/decoder 变体。

### 2.2 BLIP-2：用 Q-Former 跨越冻结底座

BLIP-2 把视觉 encoder 与 LLM 都冻结，只训练 Q-Former。32 个 query 通过 cross-attention 从图像 token 中提取语言相关证据，再作为冻结 LLM 的软提示。它把“多模态预训练”从重新训练巨型模型，改成训练一个约 188M 参数的桥。

![](../../10-Papers/VLM/assets/BLIP-2%20-%20冻结底座的高效视觉语言桥接/figures/01-overview.png)

> BLIP-2 的两阶段桥接：先做视觉—语言表征学习，再把 Q-Former 输出接到冻结 LLM。来源：BLIP-2 Figure 1。

### 2.3 InstructBLIP：让视觉查询先看任务

InstructBLIP 的改动看似很小：instruction 不仅交给 LLM，还进入 Q-Former。其意义是让 query 不再抽取“一般有用”的图像摘要，而是根据“描述、计数、OCR、推理”等任务选择不同区域与关系。

### 2.4 BLIP-3：从算法论文转向开放模型工程

BLIP-3/xGen-MM 更像完整模型家族：引入任意分辨率视觉采样、interleaved/multi-image 数据、多阶段训练与开放数据。它保留“视觉 token 先压缩再接 LLM”的思想，但 Q-Former 不再是唯一核心，重点转向可复现的大规模训练。

### 2.5 BLIP3-o：自回归模型决定语义，扩散模型生成连续视觉

BLIP3-o 不把图像量化成离散 token，而是让冻结 Qwen2.5-VL 产生 Visual Feature，再由 DiT 在 CLIP feature 空间做 Flow Matching，最后通过扩散 decoder 还原图像。理解与生成共享的是 CLIP 语义空间和自回归主干，而不是完全相同的输出头。

![](../../10-Papers/VLM/assets/BLIP3-o%20-%20全开放统一多模态理解与生成/figures/01-architecture.png)

> BLIP3-o 的统一是“共享语义空间 + 专用连续生成头”。它比统一词表更模块化，也更符合文本与图像分布的差异。来源：BLIP3-o Figure 1。

---

## 3. 核心技术对比

| 维度 | BLIP | BLIP-2 | InstructBLIP | BLIP-3 | BLIP3-o |
|---|---|---|---|---|---|
| 视觉输入 | ViT token | 冻结 ViT token | 冻结 ViT token | CLIP/SigLIP + 视觉采样 | Qwen2.5-VL/CLIP feature |
| 主要桥接 | MED 内部共享 | Q-Former 32 query | Instruction-aware Q-Former | Perceiver/vision token sampler | 自回归 Visual Feature + DiT |
| LLM 状态 | BERT 初始化并训练 | 冻结 OPT/FlanT5 | 冻结 Vicuna/FlanT5 | Phi-3 等开放 LLM | 冻结 Qwen2.5-VL |
| 输出 | 理解文本、caption | 文本/VQA/生成文本 | 指令跟随文本 | 图文理解与交错输入 | 理解文本 + 生成图像 |
| 关键数据策略 | CapFilt | 两阶段图文预训练 | 26 数据集指令化 | 开放大规模多阶段数据 | 25M/55M 预训练 + BLIP3o-60k |
| 主要效率杠杆 | 参数共享 | 冻结两端 | 复用 BLIP-2 | 开放工程配方 | 顺序训练、冻结理解主干 |

---

## 4. 为什么后面的工作会出现

### 4.1 BLIP → BLIP-2：数据干净后，成本成了瓶颈

BLIP 已能通过 CapFilt 提高数据质量，但模型规模进入 LLM 时代后，继续端到端训练视觉与语言底座代价过高。BLIP-2 因而把问题改写为“怎样用最小接口连接两个冻结专家”。

### 4.2 BLIP-2 → InstructBLIP：一般图像摘要不等于任务充分状态

固定 query 在 caption 中可能关注主体，在 OCR 中却应关注文字。InstructBLIP 把指令提前注入视觉抽取过程，使视觉状态从“通用摘要”变成“任务条件摘要”。

### 4.3 InstructBLIP → BLIP-3：模型效果之外还需要开放训练体系

单篇算法无法回答数据比例、分辨率、多图交错和大模型训练的工程问题。BLIP-3 把贡献重心转向开放模型、数据和 recipe，形成可复用基础设施。

### 4.4 BLIP-3 → BLIP3-o：理解模型不能直接输出连续视觉世界

理解模型擅长语义 token，却不擅长连续高维图像分布。BLIP3-o 选择不把图像硬塞进语言词表，而是让 AR 模型负责语义条件，Flow Matching DiT 负责随机连续生成。

---

## 5. 目前仍未解决的问题

1. **固定视觉瓶颈是否会丢失几何与文字细节。** 从 Q-Former 到 CLIP feature，系列一直依赖压缩视觉接口；高效与精确之间的矛盾仍在。
2. **理解与生成是否真的互相促进。** BLIP3-o 通过冻结主干避免冲突，但这更像能力拼接，还没有证明生成监督能改善理解与推理。
3. **开放性口径不统一。** BLIP-3 和 BLIP3-o 都强调开放，但部分模型用专有数据；必须区分开放权重、开放代码和训练数据可重建。
4. **真实交互不足。** 系列主要覆盖静态图像、文本输出和文生图，尚未成为可持续视频世界模型或具身行动模型。
5. **评测存在 judge 偏差。** BLIP3-o 在 DPG 自动评分落后 Janus-Pro，但人工偏好反超，说明统一模型不能只依赖一个 VLM judge。

---

## 6. 推荐阅读顺序

- 想理解数据治理：先读 BLIP 的 CapFilt。
- 想理解冻结底座连接：重点读 BLIP-2 的 Q-Former 两阶段训练。
- 想做通用指令模型：再读 InstructBLIP 的 instruction-aware query。
- 想训练开放 VLM：读 BLIP-3 的数据和训练 recipe。
- 想研究统一理解生成：最后读 BLIP3-o 的 CLIP feature + Flow Matching。

这五篇连起来，展示了 VLM 研究从“把图文数据洗干净”走向“让模型既能读图，也能画图”的完整问题演化。
