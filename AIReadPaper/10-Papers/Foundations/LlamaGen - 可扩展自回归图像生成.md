---
type: paper-note
status: complete
domain: Foundations
paper: "Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation"
year: 2024
arxiv: "2406.06525"
doi: "10.48550/arXiv.2406.06525"
source: "https://arxiv.org/abs/2406.06525"
project: "https://peizesun.github.io/llamagen/"
code: "https://github.com/FoundationVision/LlamaGen"
tags:
  - Image-Generation
  - Autoregressive-Modeling
  - Visual-Tokenizer
  - Llama
created: 2026-10-15
updated: 2026-10-15
---

# LlamaGen：可扩展自回归图像生成

## 基本信息

- **论文**：Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation
- **作者**：Peize Sun、Yi Jiang、Shoufa Chen、Shilong Zhang、Bingyue Peng、Ping Luo、Zehuan Yuan
- **机构**：香港大学、字节跳动
- **版本**：arXiv v1，2024-06-10
- **代码**：https://github.com/FoundationVision/LlamaGen

## 一句话结论

LlamaGen 的意义不是提出新生成目标，而是证明“高质量离散 tokenizer + 原生 Llama decoder-only Transformer + 标准 next-token prediction”这套极简配方可以随模型规模稳定提升：3.1B 模型在 ImageNet 256×256 上以 FID 2.18 略超 DiT-XL/2 的 2.27；但胜利发生在类别条件基准、参数量更大，开放文生图仅有定性结果，因此它证明的是视觉 raster AR 仍可扩展，而不是全面终结 diffusion。

---

## 1. 研究背景与问题定义

### 1.1 为什么重新做视觉 AR

早期 VQGAN、DALL-E、RQ-Transformer 已经证明图像可离散成 token 后用 GPT 生成，但长期落后于 diffusion：tokenizer 重建损失高、码本坍塌、长序列难建模、逐 token 推理慢。LlamaGen 追问：如果不引入视觉专用归纳偏置，只把 tokenizer、Llama 架构、规模和数据质量做好，最朴素的 next-token prediction 能否达到强生成性能？

![](assets/LlamaGen%20-%20可扩展自回归图像生成/figures/01-teaser.png)

> 图 1：LlamaGen 的类别条件与文本条件生成样例。论文有意保留普通自回归范式，以显示能力主要来自 tokenizer 和规模化。来源：论文 Figure 1，p.1。

### 1.2 两个真正瓶颈

第一是视觉 tokenizer：若 VQ 重建已经丢掉细节，Transformer 再大也不能恢复。第二是生成模型的规模与序列长度匹配：更高分辨率 tokenizer 给出更长序列，只有足够大的模型才能利用更低的重建误差。

### 1.3 贡献

1. 通过低维 code vector、$\ell_2$归一化与大码本把离散 tokenizer 的重建质量推近连续 VAE。
2. 将 Llama 的 RMSNorm、SwiGLU、RoPE 与 decoder-only causal attention 直接用于图像 token。
3. 从 111M 扩展到 3.1B，展示 ImageNet FID 持续改善。
4. 证明 vLLM 类 LLM serving 基础设施能直接加速图像 AR 推理。
5. 扩展到 50M+10M 两阶段文生图训练，展示长提示对齐和视觉质量改善。

---

## 2. 任务定义与表示

### 2.1 视觉 token 化

图像$x\in\mathbb R^{H\times W\times3}$经卷积 encoder 得到连续特征$f$，每个位置量化到码本$Z\in\mathbb R^{K\times C}$的最近向量，得到二维离散 token map$q\in[K]^{h\times w}$，再按 raster scan 展平成长度$T=hw$的序列。

量化采用 straight-through estimator：

$$z=\operatorname{sg}[z-f]+f$$

VQ 损失为：

$$\mathcal L_{\mathrm{VQ}}=\|\operatorname{sg}[f]-z\|_2^2+\beta\|f-\operatorname{sg}[z]\|_2^2$$

其中$\beta=0.25$。重建目标结合像素$\ell_2$、LPIPS 与 PatchGAN：

$$\mathcal L_{\mathrm{AE}}=\ell_2(x,\hat x)+\mathcal L_P(x,\hat x)+\lambda_G\mathcal L_G(\hat x)$$

$\lambda_G=0.5$，对抗损失在 20k iterations 后启用。

### 2.2 自回归生成

在类别或文本条件$c$下：

$$p(q\mid c)=\prod_{t=1}^{T}p(q_t\mid q_{<t},c)$$

训练是标准 next-token cross entropy；推理逐 token 解码。类条件用可学习 class embedding 作为 prefix，文本条件由 FLAN-T5 XL 编码后经 MLP 投影为 prefix embeddings。

### 2.3 Classifier-Free Guidance

训练时以 0.1 概率丢弃条件。推理 logits 为：

$$\ell_g=\ell_u+s(\ell_c-\ell_u)$$

更高 CFG 通常提高 precision 与 IS，但会损害 recall；最佳 FID 的$s$随模型变化。

---

## 3. 核心方法

### 3.1 高利用率视觉 tokenizer

默认 tokenizer 使用下采样率 16、code dimension$C=8$、codebook size$K=16384$，并对 codebook 向量做$\ell_2$归一化。最反直觉的发现是：把 code dimension 从 256 降到 8，rFID 从 9.21 降到 2.19，codebook usage 从 0.29% 升到 97.0%。

| Code dim | rFID↓ | PSNR↑ | SSIM↑ | Usage↑ |
|---:|---:|---:|---:|---:|
| 256 | 9.21 | 18.32 | 0.575 | 0.29% |
| 32 | 3.22 | 19.98 | 0.646 | 20.9% |
| **8** | **2.19** | **20.79** | **0.675** | **97.0%** |
| 4 | 9.88 | 19.39 | 0.593 | 82.0% |

高维 codebook 向量更容易出现大量闲置 code；压到 8 维后，最近邻分区更容易被训练数据覆盖。继续降到 4 维则容量不足。

输入从 256 提到 384、512 时，token 数从 256 增到 576、1024，rFID 从 2.19 降到 0.94、0.70。下采样率 8、256 输入时可达 rFID 0.59，表中优于 SDXL-VAE 的 0.68；但更长序列会把压力转移到 AR 模型。

### 3.2 Llama 骨干

模型保留 Llama 风格：pre-norm、RMSNorm、SwiGLU、decoder-only causal attention，并使用 2D RoPE；没有采用 DiT 的 AdaLN。模型族如下：

| 模型 | 参数量 | 层数 | 宽度 | Heads |
|---|---:|---:|---:|---:|
| B | 111M | 12 | 768 | 12 |
| L | 343M | 24 | 1024 | 16 |
| XL | 775M | 36 | 1280 | 20 |
| XXL | 1.4B | 48 | 1536 | 24 |
| 3B | 3.1B | 24 | 3200 | 32 |

3B 不是简单继续加深，而是显著加宽。作者希望尽量排除视觉专用结构，让结果回答“语言模型骨干本身是否可迁移到视觉生成”。

### 3.3 Token 数与模型容量的耦合

50 epoch 对比中，小模型使用 576 token 反而比 256 token 差；3B 才从更低 tokenizer rFID 中获益：

| 模型 | 256 token FID↓ | 576 token FID↓ |
|---|---:|---:|
| B | 8.69 | 12.89 |
| L | 4.21 | 5.01 |
| XL | 3.39 | 3.42 |
| XXL | 3.09 | 2.89 |
| 3B | 3.06 | **2.61** |

这说明 tokenizer 指标不能脱离生成模型容量理解：更精细的 token 序列降低表示误差，却增加分布建模难度。

### 3.4 文生图两阶段训练

Stage I 从约 600M LAION-COCO 中按 URL、审美、水印、CLIP 相似度与尺寸筛到 50M，使用 256×256 和 BLIP 短 caption，先建立文本—图像对齐。Stage II 用 10M 内部高审美图像、512×512，并由 LLaVA 生成长描述，提升细节和美学。

![](assets/LlamaGen%20-%20可扩展自回归图像生成/figures/03-two-stage-t2i.png)

> 图 4：Stage I 已能跟随提示，但细节与构图较弱；Stage II 的高质量数据与更高分辨率显著改善视觉质量。来源：论文 Figure 4，p.9。

文本编码器仍是外置 FLAN-T5 XL，视觉与语言没有统一 vocabulary，因此这更像“文本条件的视觉 AR”，不是统一多模态 LLM。

### 3.5 推理基础设施复用

在 384×384、CFG 双分支、每批实际 8 张图的设置中，vLLM 将 111M–1.4B 模型加速约 3.26×–4.14×。3B 因 head dimension=100 不被当时 PagedAttention 支持。这个实验的重要性高于具体秒数：视觉 AR 可以直接复用 KV cache、continuous batching、PagedAttention 等 LLM serving 技术栈。

---

## 4. 实验设置

Tokenizer 在 ImageNet 上训练 40 epochs，AdamW，学习率$10^{-4}$，batch 128。类条件生成在 ImageNet train 上训练，预先为每张原图提取 10 个 crop codes，训练时随机取一个。评测使用 50k validation，指标包括 FID、IS、sFID、precision、recall。

生成模型使用 AdamW（$\beta_1=0.9,\beta_2=0.95$）、weight decay 0.05、gradient clipping 1.0；小模型用 DDP，大模型用 FSDP，训练硬件为 80GB A100。

---

## 5. 实验结果

### 5.1 ImageNet 类条件生成

| 模型 | 参数 | FID↓ | IS↑ | Precision↑ | Recall↑ |
|---|---:|---:|---:|---:|---:|
| DiT-XL/2 | 675M | 2.27 | 278.2 | 0.83 | 0.57 |
| StyleGAN-XL | 166M | 2.30 | 265.1 | 0.78 | 0.53 |
| LlamaGen-XL | 775M | 2.62 | 244.08 | 0.80 | 0.57 |
| LlamaGen-XXL | 1.4B | 2.34 | 253.90 | 0.80 | **0.59** |
| **LlamaGen-3B** | **3.1B** | **2.18** | 263.33 | 0.81 | 0.58 |

标题中的“Beats Diffusion”主要指 3B 的 FID 2.18 略低于 DiT-XL/2 的 2.27。必须同时说明：LlamaGen 参数约为 DiT 的 4.6 倍；任务是 ImageNet 类条件而不是开放 T2I；差距只有 0.09 FID。

3B 最精确结果是表 10 的 FID 2.189、CFG 1.65、300 epochs。CFG 提到 2.0 后 IS 升到 311.59，但 FID 退化到 2.816，体现质量—多样性权衡。

![](assets/LlamaGen%20-%20可扩展自回归图像生成/figures/02-cfg-ablation.png)

> 图 3(a)：LlamaGen-L 在 CFG 约 2.0 时 FID 最优，继续增大 CFG 会提高 IS，却使 FID 回升。来源：论文 Figure 3(a)，p.7。

### 5.2 Scaling 结论

B→L→XL→XXL 的 FID 在训练全程稳定改善；1.4B→3.1B 仍有收益但变小。作者把饱和归因于 ImageNet 只有约 1M 图像。证据支持“模型扩大有效”，但论文没有像 VAR 那样拟合严格的幂律方程。

### 5.3 Tokenizer 对比

下采样率 8、256 输入时，LlamaGen tokenizer 的 ImageNet rFID/PSNR/SSIM 为 0.59/24.45/0.813；SDXL-VAE 为 0.68/26.04/0.834。也就是说 LlamaGen 的 rFID 更低，但像素保真指标并非全面更好。COCO 上 rFID 4.19，略差于 SDXL-VAE 的 4.07。

### 5.4 文生图证据边界

论文展示 COCOPrompts、PartiPrompts 和长文本定性样例，但没有报告 COCO FID、CLIP Score、GenEval、DPG-Bench 或人类偏好。作者明确承认文字渲染、计数和常识错误。因此文生图部分证明了“可行且有潜力”，尚未证明达到当时 SOTA。

### 5.5 推理加速

| 模型 | Baseline | vLLM | 约加速 |
|---|---:|---:|---:|
| B | 7.80s | 2.39s | 3.26× |
| L | 13.72s | 3.48s | 3.94× |
| XL | 19.76s | 4.84s | 4.08× |
| XXL | 26.38s | 6.36s | 4.15× |

这仍然是批量吞吐设置，不代表单图交互延迟；逐 token raster scan 的串行长度并未被消除。

---

## 6. 与 VAR、Infinity 的关系

LlamaGen 与 [[VAR - 下一尺度自回归图像生成]]几乎同时出现，但选择相反：LlamaGen 坚持 next-token raster scan，回答“普通 Llama 能否扩展”；VAR 改写为 next-scale，回答“视觉 AR 的因果轴是否应重定义”。后续 [[Infinity - 位级视觉自回归高分辨率图像生成]]沿 VAR 路线，用 bitwise tokenizer、IVC 和 BSC 解决高分辨率表示与误差传播。

因此 LlamaGen 更像强基线与“最小假设实验”，VAR/Infinity 则是范式与表示创新。三者共同证明 diffusion 并非视觉生成唯一可扩展路线。

---

## 7. 局限与批判性评价

### 7.1 Raster scan 串行瓶颈仍在

576/1024 token 需要同等数量的自回归步骤；vLLM 主要优化实现与批量吞吐，没有改变算法串行深度。分辨率继续上升时，延迟仍是核心约束。

### 7.2 “超过 diffusion”口径狭窄

只在 ImageNet 256×256 类条件 FID 上略超一个 DiT baseline，且模型更大。不能外推到长文本、1024 分辨率、文字渲染、组合关系和编辑能力。

### 7.3 T2I 数据不可完全复现

Stage II 的 10M 内部高审美数据没有公开，定量 benchmark 缺失，难以拆分数据质量、分辨率和架构的贡献。

### 7.4 视觉与语言仍是两套系统

FLAN-T5 XL 只提供 prefix embeddings，视觉 token 与文本 token 没有共享词表或统一目标。论文提出未来统一理解与生成、扩到 7B+、提高到 1024 分辨率，但未在本文验证。

---

## 8. 复现与实践建议

最现实的复现顺序是：先直接使用官方 tokenizer 和 B/L 模型验证生成；再单独复现$C=8,K=16384$ tokenizer；最后训练 111M/343M 生成模型。3B 完整训练需要 80GB A100 集群和长周期，不适合作为入门目标。

实现重点包括：2D RoPE 索引；raster 展平与解码一致；condition dropout/CFG 双路；预提取 10-crop codes；以及明确评测时 384 生成缩放到 256 的口径。

---

## 9. 个人判断

**★★★★☆。** LlamaGen 的学术价值在于用最朴素配方建立了一个很强的视觉 AR scaling baseline，也证明 LLM 推理基础设施可以迁移。它的长期地位更像“GPT 式视觉生成重新起跑的基准点”，而不是最终形态；后续真正解决高分辨率、串行延迟和 teacher-forcing 误差的工作，更多沿 VAR/Infinity 方向展开。

---

## 10. 材料来源

- [arXiv](https://arxiv.org/abs/2406.06525)
- [项目页](https://peizesun.github.io/llamagen/)
- [代码与模型](https://github.com/FoundationVision/LlamaGen)
- [本地来源清单](assets/LlamaGen%20-%20可扩展自回归图像生成/sources.md)
