---
type: paper-note
status: complete
domain: VLM
paper: "Pixtral 12B"
year: 2024
arxiv: "2410.07073"
doi: null
source: https://arxiv.org/abs/2410.07073
project: https://mistral.ai/news/pixtral-12b/
code: https://github.com/mistralai/mistral-inference/
tags:
  - VLM
  - native-resolution
  - multimodal-evaluation
  - RoPE-2D
created: 2026-10-15
updated: 2026-10-15
---

# Pixtral 12B：原生分辨率视觉语言模型

## 基本信息

- **团队**：Mistral AI Science team
- **模型**：Mistral Nemo 12B decoder + 约 400M 参数的 Pixtral-ViT + 两层 MLP connector
- **上下文**：128K token，可交错输入任意数量图像与文本，但总 token 受窗口限制
- **许可**：论文 CC BY 4.0；模型权重 Apache 2.0
- **开放边界**：权重、推理代码、评测代码和 MM-MT-Bench 开放；训练数据、完整训练代码与训练配方未披露。

**一句话结论：**

Pixtral 12B 用从头训练的可变尺寸 ViT、RoPE-2D 和行/图像分隔 token 直接编码自然长宽比图像，在文档与图表任务上优于固定分辨率 CLIPA；但论文几乎没有公开训练数据、超参数和算力，因此模型可用、评测可复现，训练不可完整复现。

---

## 1. 问题定义与“原生分辨率”

### 1.1 固定方形输入的代价

固定 224/336/384 编码器会拉伸图像，或要求 AnyRes 式切 tile。前者损失文字与细节，后者把同一图像拆成多个独立视图并增加工程规则。Pixtral 试图让视觉编码器直接接受可变高度、宽度和长宽比，让用户通过输入像素数显式控制视觉 token 成本。

### 1.2 原生分辨率的准确边界

论文声称支持图像自然分辨率与长宽比，但没有披露训练时的分辨率分布、最大训练分辨率或正式推理硬上限。视觉编码器消融中使用“最大 1024 像素”，这只是该消融设置，不能等同于模型全局上限。

若 patch size 为 16，对高 $H$、宽 $W$ 的图像，patch 网格规模可写为：

$$n_h=\left\lceil\frac{H}{16}\right\rceil,\quad n_w=\left\lceil\frac{W}{16}\right\rceil$$

视觉序列在每个 patch 行后加入 `[IMAGE BREAK]`，图像末尾加入 `[IMAGE END]`，因此单图 token 量近似为：

$$N_{\mathrm{vision}}=n_hn_w+n_h+1$$

这是根据论文序列规则的直接计数，不是论文单列的训练公式。

![](assets/Pixtral%2012B%20-%20原生分辨率视觉语言模型/figures/01-main-results.png)

> **Figure 1，PDF p.2。** Pixtral 在 MM-MT-Bench 和 LMSys-Vision ELO 的性能/参数规模前沿优于多数同规模开放模型；Qwen2-VL-72B 绝对分更高，但参数显著更多。来源：arXiv 源码完整双子图。

---

## 2. 架构与输入输出

### 2.1 三个组件

1. **Pixtral-ViT**：约 400M，24 层，维度 1024，16 heads，patch 16，从头训练。
2. **Connector**：两层全连接，隐藏层与 decoder 嵌入同宽，GeLU。
3. **Multimodal decoder**：Mistral Nemo 12B，40 层、维度 5120、32 query heads/8 KV heads、128K context。

输入可为文本、单图或多图交错序列；所有视觉 token 与文本 token 在 decoder 中统一使用 1D RoPE 和因果自注意力。输出仍是文本 next-token prediction。

### 2.2 Pixtral-ViT 的四个设计

- **Break token**：区分 patch 数相同但形状不同的图像。
- **RoPE-2D**：奇数维编码高度、偶数维编码宽度，使注意力内积依赖二维相对位移，避免绝对位置表在新尺寸上插值。
- **Gated FFN**：视觉 Transformer 使用带门控前馈层。
- **Sequence packing**：多图沿序列拼接，以 block-diagonal mask 阻止图像间视觉编码注意力泄漏，提高 batch 利用率。

![](assets/Pixtral%2012B%20-%20原生分辨率视觉语言模型/figures/02-vision-encoder.png)

> **Figure 2，PDF p.3。** 多张不同形状图像打包进同一视觉序列；二维 RoPE 保留行列位置，block-diagonal mask 隔离图像，行末与图末 token 显式保留结构。来源：arXiv 源码原图。

### 2.3 完整多图序列

视觉编码器分别处理图像，connector 投影后，视觉和文本片段按对话次序交错送入同一个 decoder。与 LLaVA-OneVision 的显式场景 token 预算不同，Pixtral 的成本更直接由每张图的像素尺寸与图像数量决定。

![](assets/Pixtral%2012B%20-%20原生分辨率视觉语言模型/figures/03-full-architecture.png)

> **Figure 3，PDF p.4。** 图像和文本可以多轮交错；每个视觉块由同一 Pixtral-ViT 编码，decoder 同时完成跨图关系理解和文本生成。来源：arXiv 源码原图。

---

## 3. 训练配方：已知与未知

### 3.1 论文明确披露

- decoder 初始化自 Mistral Nemo 12B；
- Pixtral-ViT 从头训练；
- 经过 instruction tuning；
- 预训练数据形态是“大规模交错图文文档”；
- 支持多轮、多图；
- 视觉编码采用 RoPE-2D、break/end token、packing；
- 视觉编码器消融使用 Mistral-Nemo 12B-Instruct 初始化 decoder，并做短周期多模态指令微调。

### 3.2 论文未披露

**训练配方不透明是本文必须保留的结论。** 论文没有给出：

- 数据集名称、来源、许可、总量、语言与类别比例；
- 自然图像、文档、图表、PDF、代码截图和合成数据的比例；
- 去重、污染检查、质量过滤与安全过滤；
- 指令数据来源、规模、多轮/多图采样比例；
- 优化器、学习率、batch、step/epoch、warm-up、weight decay、梯度裁剪；
- 分阶段训练、冻结策略、训练分辨率分布与最大分辨率；
- 硬件、时长、总算力、checkpoint 选择；
- 完整训练代码。

因此只能复现推理与评测，不能根据论文重建训练过程。任何关于“用了哪些具体数据集”或“训练成本多少”的描述都属于猜测，应明确拒绝补造。

---

## 4. MM-MT-Bench 与评测协议

### 4.1 Benchmark

MM-MT-Bench 有 92 个对话：21 charts、19 tables、24 PDF pages、20 diagrams、8 miscellaneous；69 个单轮，23 个多轮。GPT-4o judge 按正确性和完整性打 1–10 分，历史轮次使用参考答案 teacher-forcing。它与 2024-10 LMSys-Vision ELO 的 Pearson 相关系数为 0.91。

### 4.2 为什么论文花大量篇幅讨论 prompt

多模态模型常因输出格式不匹配 exact-match 而被误罚。例如答案 `6.0`、`The answer is 6` 与 `6` 语义相同，但严格 parser 可能判错。反过来，过宽松 substring parser 可能把 `6000` 对参考答案 `6` 判对。论文把评测协议本身视为主要研究对象之一。

---

## 5. 定量结果与 Baseline

### 5.1 多模态主结果

| 模型 | MathVista | MMMU | ChartQA | DocVQA | VQAv2 | MM-MT |
|---|---:|---:|---:|---:|---:|---:|
| **Pixtral 12B** | **58.3** | 52.0 | **81.8** | 90.7 | 78.6 | **6.05** |
| Qwen2-VL 7B | 53.7 | 48.1 | 41.2 | **94.5** | 75.9 | 5.45 |
| LLaVA-OneVision 7B | 36.1 | 45.1 | 67.2 | 90.5 | 78.4 | 4.12 |
| Molmo-D 7B | 12.3 | 24.3 | 27.0 | 72.2 | 57.1 | 3.72 |
| Llama-3.2 90B | 49.1 | **53.7** | 33.8 | 85.7 | 67.0 | 5.50 |
| Qwen2-VL 72B | 68.2 | 60.3 | 66.6 | 96.3 | 81.6 | 6.59 |
| GPT-4o | 64.6 | 68.6 | 85.1 | 88.9 | 77.8 | 7.72 |

Pixtral 明显领先同量级开放模型，并在部分指标超过更大模型；但 Qwen2-VL-72B、GPT-4o 等在多项绝对分上更高。论文“best performance/cost ratio”是规模效率主张，不是所有任务绝对 SOTA。

### 5.2 文本能力

Pixtral 的 MT-Bench/MMLU/MATH/HumanEval 为 7.68/69.2/48.1/72.0，对比 Llama-3.2-11B 为 7.51/68.5/48.3/62.8。**论文实测**支持“多模态训练未明显牺牲文本能力”，但缺少训练前 Nemo checkpoint 的同协议前后对比，因果归因仍有限。

---

## 6. 消融实验

### 6.1 Prompt 敏感性

Naive→Explicit 时：GPT-4o 的 ChartQA 58.0→85.1，Llama-3.2-11B 0.0→14.8；Pixtral 84.3→81.8，变化较小。10 个 prompt 下，Pixtral 在 MathVista/MMMU/ChartQA 的标准差为 1.0/1.5/0.4，论文据此称其 prompt 稳健。

### 6.2 Parser 敏感性

Pixtral 的 Baseline→最宽松 Level 3 为 MathVista 58.3→58.5、MMMU 52.0→52.0、ChartQA 81.8→82.0；Llama-3.2-11B 则为 24.3→47.9、23.0→45.3、14.8→78.5。Pixtral 的高分较少依赖后处理，而部分 baseline 的统一评测会严重低估真实回答。

### 6.3 视觉编码器

![](assets/Pixtral%2012B%20-%20原生分辨率视觉语言模型/figures/04-vision-encoder-ablation.png)

> **Figure 6，PDF p.8。** 在短周期 instruction tuning 中，Pixtral-ViT@1024 在 ChartQA、DocQA、AI2D 明显高于固定 224 或 25-crop 1120 的 CLIPA，在 VQAv2 大体持平。来源：arXiv 源码原图。

这一消融支持可变分辨率编码器对细粒度文档任务有价值，但论文明确称其为 small-scale、short-horizon；不能据此分离 RoPE-2D、break token、gating 和数据的各自贡献。

---

## 7. 与同类路线的关系

- **LLaVA-OneVision**：用 AnyRes crop 与手工 token 预算统一单图、多图和视频；Pixtral 把尺寸适配下沉到原生可变形状 ViT。
- **PaliGemma**：用 224/448/896 独立 checkpoint；Pixtral 用 RoPE-2D 避免分辨率专用位置表。
- **Molmo**：通过重叠 crop 保存边界上下文，数据与训练高度公开；Pixtral 架构更原生，但数据配方最不透明。
- **Cambrian-1**：多编码器+SVA 追求表征互补；Pixtral 希望一个 400M ViT 同时承担自然图、文档和图表。

Pixtral 代表从“外部切块规则”走向“编码器内部原生可变尺寸”的转折，但没有通过完整逐组件消融证明每项设计的独立收益。

---

## 8. 局限、Future Work 与个人判断

### 8.1 论文明确局限或未覆盖项

- 多模态评测对 prompt/parser 高度敏感，统一排名可能失真。
- MM-MT-Bench 仅 92 个对话，规模有限且使用 GPT-4o judge。
- 视觉编码器实验是短周期小规模消融。
- 论文没有系统讨论安全、偏见、幻觉、语言覆盖、失败案例或部署显存。
- 训练数据、超参数、算力和训练代码缺失，训练不可复现。

### 8.2 Future Work

论文没有独立 Future Work，也没有明确承诺下一代规模、数据扩展或安全研究。可以确认的后续公共价值只有已发布的 MM-MT-Bench、评测 prompt 与代码；不应把作者未写的方向补成“未来计划”。

### 8.3 个人判断

**判断**：Pixtral 的架构方向很强，尤其适合文档、截图和非方形图像；但论文证据更像技术报告而非可复现训练论文。它最可信的贡献是可变尺寸视觉编码器与评测稳健性，最薄弱处是训练透明度。若把“开放模型”理解为权重可下载，它成立；若理解为可从数据到权重独立重建，则不成立。

---

## 9. 复现成本与实践建议

- **可复现部分**：下载权重，用官方 inference/evals 复现 benchmark、prompt 与 parser 敏感性；验证多图和不同分辨率推理。
- **不可复现部分**：从头训练 Pixtral-ViT 和 12B decoder 的完整过程，因为关键配方未披露。
- **推理成本**：视觉 token 近似随像素面积线性增长；高分辨率多图很快消耗 128K context，并增加 decoder 二次注意力成本。
- **建议**：部署时按任务选择分辨率，文档/OCR 提高像素，自然图 VQA 优先较低分辨率；同时记录真实视觉 token 数而非只记录图像张数。

---

## 10. 材料来源

官方 PDF、四张完整源码图和来源记录见：[assets/Pixtral 12B - 原生分辨率视觉语言模型/sources.md](assets/Pixtral%2012B%20-%20原生分辨率视觉语言模型/sources.md)。