---
type: paper-note
status: complete
domain: VLM
paper: "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models"
year: 2024
arxiv: "2409.17146"
doi: null
source: https://arxiv.org/abs/2409.17146
project: https://molmo.allenai.org/blog
code: null
tags:
  - VLM
  - open-data
  - pointing
  - multimodal-training
created: 2026-10-15
updated: 2026-10-15
---

# Molmo：开放数据驱动的视觉语言模型

## 基本信息

- **作者**：Matt Deitke、Christopher Clark、Sangho Lee 等，Allen Institute for AI / University of Washington 团队
- **模型**：MolmoE-1B、Molmo-7B-O、Molmo-7B-D、Molmo-72B
- **数据**：PixMo-Cap、AskModelAnything、Points、CapQA、Docs、Clocks、Count
- **核心定位**：用开放数据和开放训练流程回答“能否不蒸馏专有 VLM，也构建高性能开放 VLM”。

**一句话结论：**

Molmo 的真正贡献不是某个 72B 榜单分数，而是把数据采集方式变成模型能力设计：PixMo 用长语音描述、人工问答、二维指点和受控合成数据提供监督，Molmo 再用重叠裁剪、注意力池化与“先指点后计数”把这些监督转成细粒度视觉能力。

---

## 1. 研究背景与 Molmo—PixMo 关系

### 1.1 问题定义

同期最强开放权重 VLM 常依赖 GPT-4V 等专有模型生成描述或指令，开放模型因而更像封闭模型的蒸馏产物。论文希望建立一条不依赖外部专有 VLM 的训练路线，并把模型权重、视觉语言数据、训练代码和评测一并开放。

### 1.2 Molmo 是模型，PixMo 是能力来源

- **Molmo** 是 Multimodal Open Language Model 家族，架构仍是视觉编码器—连接器—decoder-only LLM。
- **PixMo** 是 Pixels for Molmo 数据家族。PixMo-Cap 负责多模态预训练，其余 PixMo 数据与开放学术数据共同用于指令微调。
- 数据流程可使用语言模型、OCR、检测器、渲染器与语音转文字，但不使用外部 VLM 直接看图生成训练答案。

因此，PixMo 不是 Molmo 的附属 benchmark，而是模型能力的主要监督源。长描述对应细粒度识别；Points 对应定位与计数；Docs/Clocks 对应文档、图表与时钟读取；AskModelAnything 对应真实用户交互。

![](assets/Molmo%20-%20开放数据驱动的视觉语言模型/figures/01-pixmo-molmo-overview.png)

> **Figure 1，PDF p.2。** 左侧是 PixMo 的人工与合成数据流水线，右侧是 Molmo 获得的描述、交互、指点、计数和视觉技能。该图直接体现“数据形式决定能力接口”。来源：arXiv 源码原图。

---

## 2. 输入输出与模型结构

### 2.1 输入输出

输入可以是一张图像及文本指令；图像经低分辨率全图和多个重叠高分辨率 crop 编码。输出是普通文本，也可在文本中生成 XML 风格的点坐标：

```text
<point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>
```

坐标归一化到 $[0,100]$。多点按从上到下、从左到右的稳定顺序生成。计数任务不是只输出整数，而是先逐个指点目标，再给出数量。

### 2.2 模型家族

| 模型 | LLM | 参数说明 |
|---|---|---|
| MolmoE-1B | OLMoE-1B-7B | 1.2B 激活、6.9B 总参数 |
| Molmo-7B-O | OLMo-7B | fully open LLM 路线 |
| Molmo-7B-D | Qwen2-7B | 论文中最强 7B 版本 |
| Molmo-72B | Qwen2-72B | 最强模型 |

默认视觉编码器是 OpenAI CLIP ViT-L/14@336；论文还比较 MetaCLIP、SigLIP 和 DINOv2。这里需精确区分：论文承诺一个 MetaCLIP+OLMo 的“100% fully open”版本，但主结果通常使用开放权重而非训练数据完全开放的 OpenAI CLIP/Qwen2 组件。

### 2.3 视觉连接器

Molmo 拼接 ViT 倒数第 3 层与倒数第 10 层特征，对每个 $2\times2$ patch 窗口做多头注意力池化，query 为窗口 patch 的均值，再经 MLP 投影到 LLM 嵌入空间。消融显示注意力池化优于直接堆叠 patch 特征。

![](assets/Molmo%20-%20开放数据驱动的视觉语言模型/figures/02-model-architecture.png)

> **Figure 2，PDF p.3。** 图像由全局缩略图和高分辨率 crop 构成，经过视觉编码器与连接器生成视觉 token，再与文本 token 一同输入 LLM；输出可直接包含坐标。来源：arXiv 源码原图。

### 2.4 重叠多裁剪

相邻 crop 默认重叠 4 个 patch，即 56 像素。重叠区域的额外特征只用于提供编码上下文，不重复送入连接器，因此最终 patch 仍恰好平铺原高分辨率图像。这样，位于裁剪边界的文字或对象不会被割裂。

![](assets/Molmo%20-%20开放数据驱动的视觉语言模型/figures/03-overlapping-crops.png)

> **Figure 3，PDF p.3。** 无重叠裁剪会把自行车品牌切断；重叠裁剪让保留的中心 patch 在编码时拥有完整邻域。来源：arXiv 源码原图。

---

## 3. PixMo 数据体系

### 3.1 七个数据集

| 数据集 | 规模 | 生成方式与用途 |
|---|---:|---|
| PixMo-Cap | 712K 图像，1.3M 转录/描述 | 标注者语音描述 60–90 秒，经 ASR 和语言模型清理；用于预训练 |
| AskModelAnything | 73K 图像，162K QA | 人写问题，语言模型基于 OCR 与密集描述起草，标注者修订 |
| PixMo-Points | 223K 图像，2.3M question-points | 人工点击目标，训练定位、指点与计数 |
| PixMo-CapQA | 165K 图像，214K QA | 语言模型仅依据人工描述生成 QA |
| PixMo-Docs | 255K 图像，2.3M QA | 先生成文档/图表代码，再渲染并生成 QA |
| PixMo-Clocks | 826K | 程序化表盘与时间渲染 |
| PixMo-Count | 36K 训练图像 | 非 VLM 检测器生成对象中心点与计数，另有人工验证/测试集 |

人工数据有 Cap、AskModelAnything、Points；合成数据有 CapQA、Docs、Clocks、Count。“合成”不等于专有 VLM 蒸馏，关键是图像答案链条是否依赖外部看图模型。

### 3.2 采样公式

微调数据集的基础采样权重为：

$$p_i\propto\sqrt{N_i}$$

其中 $N_i$ 是第 $i$ 个数据集规模；大型合成集会人工降权，学习更慢的指点任务则显著加权。最终采样中 PixMo 人工数据占 38.1%，PixMo 合成数据占 31.6%，开放学术数据占 30.3%；PixMo-Points 单独占 28.8%。

### 3.3 长度条件

长描述 prompt 以带噪字符长度控制：

$$h=\left\lfloor\frac{L+\epsilon}{15}\right\rfloor,\quad\epsilon\sim\mathcal{N}(0,25^2)$$

$L$ 是描述字符数。90% 样本加入长度提示，10% 不加，从而既可控制输出长度，又保留默认生成能力。

---

## 4. 训练流程与成本

### 4.1 预训练

仅使用 PixMo-Cap，训练 4 epoch；视觉编码器、连接器和 LLM 全部训练，没有单独 connector-only 阶段。峰值学习率分别为视觉编码器 $6\times10^{-6}$、连接器 $2\times10^{-4}$、LLM $2\times10^{-5}$（72B 为 $1\times10^{-5}$）；batch 128，22.3K steps，最大序列长度 2304。

预训练只对文本 token 使用 dropout，目的是降低语言自回归捷径、迫使模型依赖图像。每张图的多个标注打包成一条序列，并用 attention mask 隔离不同标注；相较逐条训练，图像编码次数减少约三分之二，训练时间减半以上。

### 4.2 指令微调

batch 256，所有模块 warmup 200 steps，cosine 衰减到峰值的 10%。MolmoE/7B-D/7B-O/72B 分别训练 30K/30K/32K/20K steps。

### 4.3 算力

| 模型 | 预训练 | 微调 | 总 GPU-hours（约） |
|---|---:|---:|---:|
| MolmoE-1B | 8×33.3h | 64×13.3h | 1.1K |
| Molmo-7B-D | 64×8.6h | 128×11.2h | 2.0K |
| Molmo-7B-O | 64×8.9h | 128×13.5h | 2.3K |
| Molmo-72B | 128×33.3h | 256×32.4h | 12.5K |

硬件均为 H100。**复现判断**：7B 版本仍属高成本但可由研究机构复现；72B 的 12.5K H100 GPU-hours 与数据工程成本使个人/小团队难以完整重训。

---

## 5. 定量结果与 Baseline

### 5.1 主结果

11 个学术指标涵盖 AI2D、ChartQA、VQAv2、DocVQA、InfoQA、TextVQA、RealWorldQA、MMMU、MathVista、CountBenchQA 与 PixMo-Count。Molmo-72B 平均 81.2，Molmo-7B-D 为 77.3；对比 GPT-4o 78.5、Gemini 1.5 Pro 78.3、Claude 3.5 Sonnet 76.7、Qwen2-VL-72B 79.4、LLaVA-OneVision-72B 76.6。

但这不表示 Molmo 全面超过闭源模型：它在 CountBenchQA/PixMo-Count 达到 91.2/85.2，在 MMMU/MathVista 只有 54.1/58.6，推理短板明显。

论文自建人工评测包含 15K prompt、约 870 名标注者和 325K+ 成对比较。Molmo-72B Elo 1077，仅次于 GPT-4o 1079；然而独立 Chatbot Arena 中 Molmo-72B 为 1115，低于多个专有模型。两套结果必须同时报告，不能只引用内部 Elo。

![](assets/Molmo%20-%20开放数据驱动的视觉语言模型/figures/04-human-eval-elo.png)

> **正文 Table 1 与 Elo 讨论，PDF pp.6–7。** 左侧是 11 项学术平均分，右侧是论文内部人工偏好 Elo。内部评测显示 Molmo-72B 接近 GPT-4o，但独立 Arena 排名更保守。来源：arXiv 源码结果图。

### 5.2 专项能力

- 时钟读取：Molmo-7B-D 为 68.2，显著高于 GPT-4o 2.7、Qwen2-VL-72B 9.1，但仍低于专用模型 78.9。
- 指点：Molmo-72B F1 为 75.2；Molmo-7B-O 为 75.1。
- 训练/测试 crop 数不匹配时，7B-D 指点 F1 从 74.3 降至 58.1，说明坐标能力对视觉 token 布局敏感。

---

## 6. 消融实验

### 6.1 架构消融

多裁剪从单低分辨率的 cap F1/11-avg 46.7/62.8 提升到无重叠 53.4/75.7，再到有重叠 54.1/76.9。$2\times2$ 注意力池化为 54.1/76.9，高于特征堆叠 53.7/76.1。长度条件也把 53.0/76.2 提升到 54.1/76.9。

视觉编码器差异较小：OpenAI CLIP、MetaCLIP、SigLIP 的 11-avg 为 76.9/77.2/77.1，DINOv2 为 75.6。论文据此没有证明某个编码器绝对优越，反而说明数据和训练细节可压缩 backbone 差距。

### 6.2 数据消融

PixMo-Cap 从 89K 增至 712K 时 cap F1 从 49.6 增至 54.1，11-avg 从 75.5 增至 76.9。完整 PixMo+学术数据为 76.9，仅学术数据为 72.5；去掉 Docs 降至 75.8，去掉 pointing 降至 76.2。AskModelAnything 与 CapQA 对学术均分影响小，但人工偏好显示其改善用户体验。

### 6.3 “先指点再计数”

| 输出策略 | CountBenchQA | PixMo-Count |
|---|---:|---:|
| 只输出数量 | 87.9 | 80.2 |
| **先指点再计数** | **89.4** | **86.3** |
| 先计数再指点 | 81.5 | 77.6 |
| 只指点后用 regex 计数 | 88.4 | 85.4 |

空间排序的正确点为 89.4/86.3；随机位置但数量正确降为 85.9/76.3；特殊位置 token 也低于普通文本坐标。**论文实测**支持一个强结论：显式逐对象 grounding 不只是可解释输出，它直接提高计数正确率。

---

## 7. 与同类路线的关系

LLaVA-OneVision 大量依赖由早期 LLaVA 与 GPT-4V/o 生成的高质量合成数据；Molmo 则用人工长描述、人工点击和语言模型后处理，强调不使用专有 VLM 看图生成监督。PaliGemma 同样避免大商业 VLM 输出，但其目标是可迁移 base model，预训练数据仍部分私有。Cambrian-1 公开数据且重视数据平衡，但重点是视觉编码器与 connector。Pixtral 的数据配方最不透明。

因此 Molmo/PixMo 的路线不是“拒绝合成”，而是把合成限制为可审计的文本、代码渲染和非 VLM 工具链，并在关键感知监督上保留人类参与。

---

## 8. 局限与 Future Work

### 8.1 论文明确限制

- MMMU、MathVista 等高级推理较弱，训练混合缺少相应数据。
- 多模态微调会损害部分 MMLU、MATH、HumanEval 等文本能力；加入 Tulu 3 文本数据只能部分恢复。
- 指点与计数对 crop 配置敏感，且不训练超过 40 个目标的样本。
- pointing-as-explanation 可靠性较低，需 style tag 控制触发。
- caption F1 依赖 GPT-4o 拆解与判断，且人工描述不覆盖图像全部事实。
- prompt、答案格式和训练集重叠会使不同论文报告值相差约 10%，横向比较并不完全同协议。

### 8.2 原文 Future Work

作者提出把二维指点用于机器人、导航和 Web agent；移除 40 个目标上限；引入真实时钟图像；用少量 36-crop 后训练恢复高分辨率性能；加入更多文本数据；发布 MetaCLIP+OLMo 的完全开放版本。

### 8.3 个人判断

**判断**：Molmo 最可迁移的思想是“监督接口设计”。密集 caption 提供事实覆盖，pointing 把离散答案绑定到空间证据，程序化数据精准补足稀缺能力。它证明开放数据可以逼近闭源 VLM，但也暴露了开放路线的真实代价：不是免除成本，而是把成本从 API 蒸馏转移到人类标注、数据治理与大规模训练。

---

## 9. 复现建议

1. 优先复现 MolmoE-1B 或 7B-D，而非 72B。
2. 先验证 PixMo-Cap 规模、重叠裁剪与 text-only dropout，再加入全部微调混合。
3. 计数实验必须保持训练/测试 crop 数和点排序一致。
4. 同时报告学术均分、内部人工偏好与独立 Arena，避免单一评测叙事。

---

## 10. 材料来源

官方 PDF、四张完整源码图与来源定位见：[assets/Molmo - 开放数据驱动的视觉语言模型/sources.md](assets/Molmo%20-%20开放数据驱动的视觉语言模型/sources.md)。