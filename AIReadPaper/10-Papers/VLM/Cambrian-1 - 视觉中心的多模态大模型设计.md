---
type: paper-note
status: complete
domain: VLM
paper: "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs"
year: 2024
arxiv: "2406.16860"
doi: null
source: https://arxiv.org/abs/2406.16860
project: https://cambrian-mllm.github.io
code: https://github.com/cambrian-mllm/cambrian
tags:
  - VLM
  - vision-encoder
  - connector
  - instruction-tuning
created: 2026-10-15
updated: 2026-10-15
---

# Cambrian-1：视觉中心的多模态大模型设计

## 基本信息

- **作者**：Shengbang Tong、Ellis Brown、Penghao Wu 等
- **核心资产**：Cambrian-1 8B/13B/34B、Cambrian-7M/10M、CV-Bench、Spatial Vision Aggregator（SVA）
- **研究定位**：不是单点追求榜单分数，而是系统拆解视觉编码器、连接器、训练配方和指令数据对 MLLM 的贡献。
- **开放性**：论文、模型、代码、数据、训练与评测工具均公开；这是论文“fully open”主张的重要组成部分。

**一句话结论：**

Cambrian-1 的关键价值是把 MLLM 从“语言模型主导、视觉模块经验选型”拉回到可验证的视觉系统设计：多视觉编码器提供互补表征，SVA 用空间约束的动态聚合压缩 token，而经类别平衡的 Cambrian-7M 证明数据配比比简单扩量更重要。

---

## 1. 研究背景与问题定义

### 1.1 论文试图修正什么偏差

当时大量 MLLM 工作把主要注意力放在更大的 LLM、更多指令数据或更复杂的对话模板上，视觉编码器通常直接沿用 CLIP/SigLIP，连接器则采用线性层或 MLP。作者认为，这种研究范式会把视觉表征本身的差异隐藏在强语言先验后面：模型可能凭问题形式和常识答对，而不一定真正依赖图像。

论文因此提出三个连续问题：现有 benchmark 到底测视觉还是测语言；不同视觉预训练目标在统一 MLLM 接口下有什么差异；当多个高分辨率视觉编码器并存时，如何以较少 token 保留空间细节。

### 1.2 现有评测的语言泄漏

作者通过“保留视觉输入”和“禁用视觉输入”比较 benchmark。SQA-I、MMMU、MathVista、AI2D 的分差小于 5%，TextVQA 与 GQA 在无图像时仍显著高于随机水平。**论文实测**表明，一部分所谓多模态成绩高度依赖基础 LLM；因此仅看总分会高估视觉模块。

作者进一步对 23 个 MLLM 的 benchmark 成绩做 PCA，将任务归为 General、Knowledge、OCR & Chart、Vision-Centric 四类，并构建 CV-Bench。CV-Bench 共 2,638 题，覆盖 2D 空间关系、计数，以及 3D 深度顺序和相对距离，所有题目经过人工检查。

### 1.3 核心贡献

1. 在统一训练协议下比较 23 个视觉骨干，并分析冻结/解冻、分辨率和数据规模的影响。
2. 构建 CV-Bench，降低语言先验对视觉评测的干扰。
3. 提出 SVA，以空间局部 cross-attention 聚合多编码器高分辨率特征。
4. 从 Cambrian-10M 策展出 Cambrian-7M，系统研究来源阈值与类别配比。
5. 发布 8B、13B、34B 三档 Cambrian-1 及完整训练工具链。

![](assets/Cambrian-1%20-%20视觉中心的多模态大模型设计/figures/01-vision-centric-overview.png)

> **Figure 1，PDF p.4。** 蓝色为语言监督表征，红/粉色为自监督表征，绿色为其他视觉模型。语言监督模型总体占优，但 DINOv2 在 Vision-Centric 类任务上明显更强，说明“最适合图文对齐的编码器”不等于“最强纯视觉表征”。来源：arXiv 源码原图。

---

## 2. 任务定义与输入输出

### 2.1 统一接口

输入是图像（可由多个视觉编码器并行处理）与文本指令；输出是 LLM 自回归生成的文本。训练仍采用视觉指令微调的 next-token prediction。论文没有提出新的语言损失，创新集中在视觉特征的产生、聚合与训练数据组织。

### 2.2 SVA 的表示

设可学习 latent token 为 $\mathbf{x}\in\mathbb{R}^{C}$，复制为 $L\times L$ 网格 $\mathbf{X}\in\mathbb{R}^{L^2\times C}$。第 $k$ 个视觉编码器的特征图大小为 $m_kL\times m_kL\times C$。查询位置 $(i,j)$ 只聚合该位置对应的局部区域：

$$\mathbf{F}_k[m_ki:m_k(i+1),m_kj:m_k(j+1)]\in\mathbb{R}^{m_k^2\times C}$$

局部 cross-attention 为：

$$\mathbf{q}^{*}_{i,j}=\operatorname{softmax}\left(\frac{\mathbf{q}_{i,j}[\mathbf{k}_{i,j,1},\ldots,\mathbf{k}_{i,j,N}]^{\top}}{\sqrt{C}}\right)[\mathbf{v}_{i,j,1},\ldots,\mathbf{v}_{i,j,N}]$$

其中 $\mathbf{q}_{i,j}=\mathbf{W}^{Q}\mathbf{x}_{i,j}$，而 $\mathbf{k}_{i,j,k}$、$\mathbf{v}_{i,j,k}$ 分别由第 $k$ 个编码器对应局部块经 $\mathbf{W}^{K}_{k}$、$\mathbf{W}^{V}_{k}$ 投影得到。直觉上，每个 latent 只负责固定空间邻域，避免全局 resampler 在强压缩时抹平二维结构。

---

## 3. 核心方法

### 3.1 多视觉编码器的互补性

最终模型使用四个编码器：OpenAI CLIP ViT-L/14@336、SigLIP ViT-SO400M/14@384、OpenCLIP ConvNeXt-XXL@1024 和 DINOv2 ViT-L/14@518。前三者偏语言监督与 OCR/语义对齐，DINOv2 提供更强的纯视觉和空间表征。

朴素方案先把所有特征插值到 576 token，再沿通道拼接。作者指出其两项缺陷：高分辨率特征在插值中损失细节；固定拼接无法让模型按输入动态选择编码器。

### 3.2 Spatial Vision Aggregator

SVA 用 $L^2$ 个二维排列的 latent query 聚合多个编码器。它有三层设计：

- **空间归纳偏置**：每个 query 只看各编码器中对应局部块；
- **多组查询**：超参数 $G$ 控制并行 query 组，结果拼接以增大容量；
- **多层视觉聚合**：在 LLM 不同深度重复插入 cross-attention，使压缩 token 能再次访问未压缩视觉特征。

标准实验采用 cross-attention 深度 $D=3$、查询组 $G=1$，并以 stride 3 在 LLM 层间插入聚合模块。该设计不是简单“多编码器堆料”：真正的约束是以 576 个视觉 token 接入 LLM，同时保留不同分辨率与空间局部性。

![](assets/Cambrian-1%20-%20视觉中心的多模态大模型设计/figures/02-sva-architecture.png)

> **Figure 8，PDF p.13。** 左侧展示每个 latent 与多个编码器对应空间块的局部注意力，右侧展示 SVA 在 LLM 多层中反复访问视觉特征。来源：arXiv 源码原图。

### 3.3 数据策展与“Answer Machine”

Cambrian-10M 实际约 9.784M 样本。作者对单一来源设置 150K–450K 上限，并调整 General、OCR、Language、Counting、Math、Science、Code 的比例，得到 Cambrian-7M。最终比例约为 33.3%、27.6%、23.8%、8.5%、3.2%、2.9%、0.8%。

短答案 VQA 容易把模型训练成只吐关键词的“answer machine”。作者用系统提示显式要求短答案或推理过程，让训练目标与交互方式解耦。**论文声称并实测**：该提示不损害 benchmark，却改善自由对话表现。

### 3.4 训练流程

视觉骨干比较统一使用 Vicuna-1.5-7B：先冻结 LLM 与视觉编码器、只训练 connector，再用 737K 指令数据训练 LLM 与 connector，并在相关实验中解冻视觉骨干。最终 Cambrian-1 则用 2.5M adapter data 预训练 connector，再用 Cambrian-7M 指令微调；LLM 分别为 LLaMA-3-Instruct-8B、Vicuna-1.5-13B、Hermes-2-Yi-34B。

---

## 4. 数据、训练与复现条件

### 4.1 关键数据规模

| 环节 | 数据 | 作用 |
|---|---:|---|
| 视觉骨干比较 adapter stage | 1.2M | 对齐视觉特征与 LLM |
| 视觉骨干比较 instruction stage | 737K | 公平比较 23 个编码器 |
| 扩展比较 | 5M | 检验 CLIP 与 DINOv2 在解冻和增量数据下的差距 |
| 最终 connector pretraining | 2.5M | 训练多编码器到 LLM 的连接 |
| 最终 instruction tuning | Cambrian-7M | 平衡后的公开指令混合 |

### 4.2 复现成本

**已公开条件**：权重、代码、数据、CV-Bench 与评测工具齐全，复现实验路径比同期多数 MLLM 清晰。**成本判断**：完整复现仍昂贵，因为要并行运行四个大型视觉编码器、训练 8B–34B LLM，并执行多组 23 编码器和数据配比消融。较现实的复现顺序是先复现单编码器 737K 协议，再验证 SVA 的 576→36 token 压缩消融，最后扩展到四编码器。

---

## 5. 定量结果与 Baseline

### 5.1 视觉编码器与多编码器组合

SigLIP SO400M 在综合类别排名第一；OpenCLIP ConvNeXt-XXL 在 OCR & Chart 排名第一；DINOv2@518 在自监督模型中居首，并在 Vision-Centric 类排名突出。四编码器组合平均分 54.74，高于 SigLIP+DINOv2 的 51.61；但部分三编码器/双编码器组合已经非常接近，说明增量收益并非线性。

### 5.2 SVA 与连接器 Baseline

| Connector | General | Knowledge | OCR & Chart | Vision-Centric |
|---|---:|---:|---:|---:|
| Concat | 67.2 | 48.9 | 50.1 | 52.6 |
| Resampler | 63.1 | 46.5 | 27.1 | 42.6 |
| SVA-no-multi-agg | 68.0 | 49.5 | 55.2 | 52.6 |
| **SVA** | **68.5** | **49.7** | **55.5** | **53.2** |

在把 CLIP 的 576 token 压到 36 token 时，SVA 在 General/Knowledge/OCR/Vision-Centric 上为 65.5/44.5/31.4/46.9，均高于 Interpolate+MLP、C-Abstractor 与 LDPv2。最强证据不是 SVA 相对 concat 的小幅总体提升，而是强压缩时 OCR 和视觉中心任务仍保持优势。

### 5.3 最终模型

Cambrian-1-8B/13B/34B 只使用 576 个视觉 token，而 Mini-Gemini-HD 与 LLaVA-NeXT 使用 2,880 个。类别平均结果中，Cambrian-1-34B 为 General 76.8、Knowledge 67.0、OCR & Chart 71.9、Vision-Centric 68.5；对应两条开放 baseline 分别为 76.2/62.4/68.1/63.8 与 76.0/62.5/67.7/64.0。

![](assets/Cambrian-1%20-%20视觉中心的多模态大模型设计/figures/03-main-results.png)

> **Figure 13，PDF p.19。** Cambrian-1 在三个规模上相对 Mini-Gemini-HD、LLaVA-NeXT 的主要增益集中在 Knowledge、OCR & Chart 与 Vision-Centric；General 的差距较小。来源：arXiv 源码原图。

---

## 6. 消融实验

### 6.1 两阶段训练与解冻视觉编码器

connector 预训练从 0 增至 0.5M、1.2M 时总体提升；解冻视觉编码器几乎改善所有类别，自监督模型在 Vision-Centric 上收益尤其明显，但增加显存与计算成本。该结果支持“视觉编码器不能永远冻结”的主张。

### 6.2 SVA 容量

OCR & Chart 随 $D=2/3/4$ 从 52.1/52.4/52.8 提升；$G=1/2/3$ 从 52.4/52.6/53.1 提升；启用 multi-layer aggregation 从 52.4 升至 53.3。增益稳定但不巨大，说明 SVA 的基础空间约束比继续堆容量更重要。

### 6.3 数据策展

单来源阈值 250K 与 350K 的平均分均为 54.3，高于 150K 的 53.7。LLaVA-665K、Cambrian-10M、Cambrian-7M 的平均分分别为 40.7、54.8、55.9。**论文实测**：经过平衡的 7M 反而优于 10M，数据分布比名义规模更关键。

---

## 7. 与同类路线的关系

LLaVA 系列强调简洁 projector 与高质量指令数据；Mini-Gemini 通过高分辨率视觉 token 增强细节；Q-Former/Perceiver Resampler 通过全局 latent 压缩视觉序列。Cambrian-1 的差异是把“多编码器互补”和“空间局部压缩”绑定起来：它接受视觉专家并存，而不是期待单一 backbone 同时统治 OCR、语义和空间任务。

与 Molmo、LLaVA-OneVision、PaliGemma、Pixtral 相比，Cambrian-1 最像一份系统设计 cookbook。Molmo 的主轴是开放数据质量，OneVision 是跨场景迁移，PaliGemma 是 base model 的可迁移性，Pixtral 是可变分辨率编码；Cambrian-1 则把视觉表示评测、connector 和数据配比放在同一实验框架中。

---

## 8. 局限、Future Work 与个人判断

### 8.1 论文明确暴露的局限

- 一些 benchmark 主要测 LLM，现有总分不足以证明视觉 grounding。
- SVA 在编码器数量或分辨率继续增加时，单 latent 仍可能聚合过多局部特征。
- 解冻大型视觉编码器计算昂贵，部分模型受当前 sharding 限制未能完成测试。
- 自监督模型的 OCR 仍弱，说明纯视觉预训练未覆盖文字语义。
- 数据策展实验只是初步研究，比例结论未必跨模型规模稳定。

### 8.2 原文 Future Work

作者明确倡议：训练更强的 vision-only model；继续缩小语言监督与自监督表征差距；避免语言成为视觉学习捷径；发展类似 Chatbot Arena 的真实交互式 MLLM 评测。

### 8.3 个人判断

**判断**：Cambrian-1 最可靠的结论不是“多编码器一定优于单编码器”，而是不同视觉目标确实形成了可测的能力分工，且空间感知的压缩器优于无结构 resampler。四编码器方案本身成本很高，未必是长期最优部署形态；其更持久的影响可能是推动后来模型在单一原生分辨率 ViT 中吸收 OCR、空间和语义能力。

---

## 9. 复现建议

1. 用公开 737K 协议固定 LLM 和训练设置，先复现 SigLIP、DINOv2、CLIP 三个骨干的类别差异。
2. 复现 576→36 token 的 connector 消融，验证 SVA 的收益来自空间归纳偏置而非参数量。
3. 对 Cambrian-10M 做相同来源阈值与类别配比抽样，检查 7M 优于 10M 是否稳定。
4. 完整四编码器训练需评估视觉前向成本、LLM KV cache 与吞吐，不应只依据 benchmark 决策。

---

## 10. 材料来源

本笔记以官方 arXiv v2 PDF 与 LaTeX 源码为主。PDF、三张完整原文图及逐项来源记录见：[assets/Cambrian-1 - 视觉中心的多模态大模型设计/sources.md](assets/Cambrian-1%20-%20视觉中心的多模态大模型设计/sources.md)。