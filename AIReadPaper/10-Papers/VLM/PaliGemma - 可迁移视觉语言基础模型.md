---
type: paper-note
status: complete
domain: VLM
paper: "PaliGemma: A versatile 3B VLM for transfer"
year: 2024
arxiv: "2407.07726"
doi: null
source: https://arxiv.org/abs/2407.07726
project: null
code: null
tags:
  - VLM
  - transfer-learning
  - base-model
  - prefix-lm
created: 2026-10-15
updated: 2026-10-15
---

# PaliGemma：可迁移视觉语言基础模型

## 基本信息

- **作者**：Lucas Beyer、Andreas Steiner、André Susano Pinto 等，Google 团队
- **参数规模**：SigLIP-So400m 约 400M + Gemma-2B，总计低于 3B
- **支持分辨率**：224、448、896，对应独立 checkpoint
- **定位**：面向下游迁移的 base VLM，不是开箱即用的聊天模型。

**一句话结论：**

PaliGemma 证明一个不足 3B 的简洁 SigLIP—线性投影—Gemma 模型，经过 1B 多模态样本预训练和分辨率继续训练后，可以迁移到近 40 个任务；其核心不是 zero-shot 对话，而是 Prefix-LM 目标、全参数联合训练和可预测的少样本适配。

---

## 1. 问题定义：为什么强调“for transfer”

### 1.1 Base model 与 instruction model 的区别

许多 VLM 直接用聊天或 benchmark zero-shot 评价，模型接口友好，但训练目标与具体下游可能不匹配。PaliGemma 刻意提供高密度监督的基础 checkpoint，再对每个任务执行 Stage3 全参数微调。论文希望把“基础表征质量”和“下游指令对齐”分开。

因此，基础 PaliGemma 不应按聊天助手标准评价。它的主要问题是：同一个小模型能否通过有限微调覆盖 caption、VQA、OCR、检测、分割、遥感、多图和视频。

### 1.2 输入输出

接口统一为 image + text prefix → text suffix。结构化任务也被序列化成文本：检测使用 1,024 个 `<loc>` 坐标 token，分割使用 128 个 `<seg>` VQ-VAE mask token。预训练实际覆盖自然语言、检测和实例分割；更广泛任务依靠迁移获得。

![](assets/PaliGemma%20-%20可迁移视觉语言基础模型/figures/01-model-architecture.png)

> **Figure 1，PDF p.2。** SigLIP-So400m 输出视觉 token，经零初始化线性层投影到 Gemma-2B 嵌入空间，与任务 prefix 拼接后由 decoder 生成 suffix。来源：arXiv 源码原图。

---

## 2. 核心结构与训练目标

### 2.1 Token 序列

序列组织为：

```text
[image tokens..., BOS, prefix..., SEP, suffix..., EOS, PAD...]
```

224/448/896 分辨率分别产生 256/1,024/4,096 个图像 token。多图输入时，各图单独编码后直接拼接，不加入图像 ID 或分隔 token。16 帧 224px 视频共有 $16\times256=4096$ 个图像 token，与一张 896px 图像等长。

### 2.2 Prefix-LM mask

图像与 prefix token 之间使用完整双向 block attention；suffix 使用自回归 causal attention；next-token loss 只计算 suffix。若 token 序列为 $x_{1:T}$，suffix 起点为 $s$，训练目标可写为：

$$\mathcal{L}=-\sum_{t=s}^{T}\log p(x_t\mid x_{<t},\text{image},\text{prefix})$$

该式是对论文“仅监督 suffix”的标准化表达，而非论文单列公式。关键含义是：模型不会浪费损失去重建输入问题，图像和任务描述可以双向融合，再由 suffix 做生成监督。

![](assets/PaliGemma%20-%20可迁移视觉语言基础模型/figures/02-prefix-lm-mask.png)

> **Figure 2，PDF p.3。** 绿色图像 token 和紫色 prefix token 可双向注意；红色 suffix/target 只能看过去；损失只落在目标输出。来源：arXiv 源码原图。

### 2.3 简单 connector

连接器只是零初始化线性层。Table 消融中，全模型训练时 Linear 为 77.2、MLP 为 77.1；仅训练 connector 时为 70.7 与 69.7。**论文实测**不支持在这里增加 MLP 复杂度，真正重要的是联合训练各组件。

---

## 3. 四阶段训练

### 3.1 Stage0：复用单模态模型

直接加载 SigLIP ViT-So400m 和 raw pretrained Gemma-2B。SigLIP 在 Stage0 见过约 40B image-text pairs；Gemma 预训练规模在本文未披露。

### 3.2 Stage1：多模态预训练

- 分辨率 224，256 图像 token，文本长度 128；
- 1B examples，略低于 350B token；
- 所有组件都训练；
- 视觉编码器使用更慢 warm-up，防止未对齐 LLM 的初始梯度破坏视觉表征。

任务混合包括多语言 caption、OCR、VQA/VQG、开放词汇检测、实例分割和 grounded captioning。论文明确不使用更大型商业 VLM 的输出来生成预训练监督，并对下游评测图像做近重复去除。

### 3.3 Stage2：分辨率继续训练

224 checkpoint 先以 448px 继续 50M examples，再以 896px 继续 10M；文本长度增至 512，OCR、检测、分割等高分辨率任务被上调。两个 Stage2 合计约 90B token。

### 3.4 Stage3：下游迁移

对全部参数微调，可形成单任务专家、多任务 mix、多图、视频或 instruction model。论文系统搜索分辨率、epoch、学习率、label smoothing、dropout、weight decay 和是否冻结 ViT；同时给出统一简化配置：学习率 $10^{-5}$、batch 256、无 dropout/label smoothing/weight decay、不冻结任何参数、epoch 不超过 10。

### 3.5 训练成本

Stage1 使用 TPUv5e-256、略低于 3 天；每个 Stage2 同配置约 15 小时。下游迁移使用 TPUv3-32，20 分钟至 10 小时。论文报告 55% MFU 和每设备每秒 5,189 token。

---

## 4. 数据与监督边界

### 4.1 预训练数据

- Caption：WebLI、CC3M-35L，覆盖 100+ 语言；
- OCR：公共 OCR 系统产生按 raster order 排列的文字；
- VQA/VQG：CC3M-35L 与 OpenImages 上的多语言问题、英文答案；
- Detection/Segmentation：OWL-ViTv2、SAM 风格的开放世界伪标注；
- Grounded captioning：位置 token 条件下描述局部区域。

### 4.2 开放性限制

代码框架 big_vision 开源，但论文明确说明部分预训练数据仍私有。模型没有依赖大型商业 VLM 生成输出，这降低了教师蒸馏依赖；但无法仅凭公开数据完整重建 Stage1/2，因此“开放模型”不等于“全数据可复现”。

---

## 5. 定量结果与 Baseline

### 5.1 近 40 项迁移结果

Table 1 主要展示 PaliGemma 224/448/896 三种分辨率，没有为每项并列完整 baseline；具体指标定义在附录。代表性结果如下：

| 任务 | 224 | 448 | 896 |
|---|---:|---:|---:|
| VQAv2 | 83.2 | 85.6 | — |
| ChartQA human | 40.0 | 54.2 | — |
| ChartQA aug | 74.2 | 88.5 | — |
| TextVQA | 55.5 | 73.2 | 76.5 |
| DocVQA | 43.7 | 78.0 | 84.8 |
| InfoVQA | 28.5 | 40.5 | 47.8 |
| RefCOCO testA | 75.7 | 77.9 | 78.7 |

高分辨率增益集中在 OCR、文档和分割，验证 Stage2 的用途；OKVQA、NLVR2 等任务则不保证随分辨率单调上升。

### 5.2 明确外部 baseline：MMVP

PaliGemma-224 在 MMVP paired accuracy 为 47.3%，对比 GPT-4V 38.7%、Gemini 40.7%；论文称其他模型包括 LLaVA 低于随机水平。由于 Table 1 不提供统一外部 baseline，本结论不应外推为 PaliGemma 在所有任务 SOTA。

### 5.3 视频与多图

16 帧 224px 的 ActivityNet-QA、MSRVTT-QA、MSVD-QA 分别为 50.8、50.1、60.2；视频 caption 指标也有报告。模型没有视频专用模块，这些结果主要展示统一 token 接口的迁移能力，而非视频 SOTA。

---

## 6. 消融实验

### 6.1 预训练时长

完全跳过 Stage1 最差；100M examples 是消融中的有效折中，相比正式 1B 缩短 10 倍且没有显著损害任何单项任务。正式模型仍使用 1B，说明长期训练改善总体稳健性。

### 6.2 Mask 与损失

Prefix-LM 最佳。给 prefix 加 causal mask 会明显下降；进一步让 image token 也 causal 更差；对 prefix 同时计算 next-token loss 虽可训练，但平均迁移性能下降。task-prefix 对最终分数影响小，但能降低预训练困惑度。

### 6.3 冻结与重置

Stage1 全量训练优于冻结语言模型；重置任一预训练组件都会严重损害性能。冻结视觉编码器与全量训练在部分迁移任务最终分数接近，但全量训练明显改善空间理解相关的预训练验证困惑度。

### 6.4 分辨率与序列长度

将 224 图像先降采样再放大到 448 的控制实验显示，高分辨率任务约一半增益来自新增像素信息，另一半来自更长序列和更多计算。不能把 448 对 224 的全部收益都归因于“看得更清楚”。

### 6.5 分辨率专用 checkpoint

![](assets/PaliGemma%20-%20可迁移视觉语言基础模型/figures/03-resolution-ablation.png)

> **Figure 10，PDF p.11。** 448 checkpoint 降到 224 会变差；224 checkpoint 直接迁移到 448 虽提升但不如原生 448；windowing 居中。论文据此为每个分辨率提供独立 Stage2 checkpoint。来源：arXiv 源码原图。

Windowing 把 448 图像切成四个 224 window，再拼接 1,024 token，最多只带来约 5% 训练提速，且性能仍低于原生 448 checkpoint。

### 6.6 少样本迁移

![](assets/PaliGemma%20-%20可迁移视觉语言基础模型/figures/04-low-shot-transfer.png)

> **Figure 12，PDF p.13。** 粗黑线是任务中位数：4K 样本时多数任务距离全量训练不足 10%，256 样本时多数不足 20%；但单任务方差较大。来源：arXiv 源码原图。

64/256/1,024/4,096 样本实验每个设置运行 5 个随机种子。少样本结果受样本选择、随机种子和 batch 顺序影响；RefCOCO 某些设置 mIoU 可波动 10%–30%。

---

## 7. 与同类路线的关系

PaliGemma 继承 PaLI 的多任务序列化思想，但把规模压到 3B 以下并强调公开 checkpoint 和下游迁移。相比 LLaVA-OneVision，它不追求一个直接聊天的统一模型，而把每项能力交给 Stage3；相比 Molmo，它的数据不完全开放，但不依赖商业 VLM 看图合成；相比 Pixtral，它依赖固定分辨率 checkpoint，而非原生可变长宽比；相比 Cambrian-1，它只用单一 SigLIP 和线性 connector。

这条路线更接近“多模态 BERT/GPT base model”：用户接受一次微调成本，以换取小模型在具体任务上的高适配性。

---

## 8. 局限、Future Work 与个人判断

### 8.1 明确局限

- base checkpoint 不是开箱聊天模型，需要 Stage3。
- 部分预训练数据私有，无法完整数据复现。
- 224/448/896 需要独立 checkpoint，跨分辨率不灵活。
- 去掉视觉编码器的 decoder-only 方案样本效率明显更低。
- 少样本迁移方差大。
- 部分历史 WidgetCap 数值存在评测问题，跨论文比较需谨慎。

### 8.2 Future Work

论文没有单列 Future Work，但明确提出：研究 instruction tuning；将 base model 迁移到具体应用；更清晰地区分 base 与 fine-tuned VLM；继续探索无需视觉编码器、直接处理 RGB patch 的 decoder-only 多模态模型。

### 8.3 个人判断

**判断**：PaliGemma 最重要的证据是“可迁移性可被系统测量”，而不是某个 zero-shot 分数。它用极简结构把实验变量集中到训练目标、分辨率和迁移协议。代价是用户必须维护多个分辨率 checkpoint，并承担下游微调；对于任务稳定、数据可得的工业场景，这反而可能比通用聊天 VLM 更实用。

---

## 9. 复现建议

1. 直接使用公开 224/448 checkpoint 做 Stage3，先验证统一超参数配置。
2. OCR/文档任务优先 448 或 896；通用 VQA 先做等算力比较。
3. 少样本实验至少 5 个种子，并固定样本集合与 batch 顺序。
4. 若研究灵活分辨率，应把 PaliGemma 的独立 checkpoint 作为 Pixtral/NaViT 类方法的清晰 baseline。

---

## 10. 材料来源

官方 PDF、四张完整源码图和来源记录见：[assets/PaliGemma - 可迁移视觉语言基础模型/sources.md](assets/PaliGemma%20-%20可迁移视觉语言基础模型/sources.md)。