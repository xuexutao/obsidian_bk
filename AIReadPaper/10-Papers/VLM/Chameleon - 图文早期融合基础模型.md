---
type: paper-note
status: complete
domain: VLM
paper: "Chameleon: Mixed-Modal Early-Fusion Foundation Models"
year: 2024
arxiv: "2405.09818"
doi: null
source: "https://arxiv.org/abs/2405.09818"
project: null
code: null
tags:
  - VLM
  - unified-multimodal
  - early-fusion
  - autoregressive
  - image-tokenizer
  - decoder-only
created: 2026-10-16
updated: 2026-10-16
---

# Chameleon：图文早期融合基础模型

## 0. 基本信息与阅读边界

- **作者**：Chameleon Team / Meta FAIR
- **模型规模**：7B、34B；上下文长度 4,096
- **核心定位**：图像、文本、代码都先离散成 token，再由单个 decoder-only Transformer 自回归理解和生成任意交错图文序列。
- **本地原文**：[paper.pdf](assets/Chameleon%20-%20图文早期融合基础模型/sources/paper.pdf)
- **来源清单**：[sources.md](assets/Chameleon%20-%20图文早期融合基础模型/sources.md)

> [!important] 与普通 image-to-text VLM 的区别
> Chameleon 的关键不是“图像 encoder 接 LLM”，而是**没有独立视觉 encoder 参与主干推理**：输入图像先被量化成与文本并列的离散 token，输出图像也由同一 LM head 顺序预测，再经 image de-tokenizer 还原。它补足的是视觉输出与交错文档生成架构，而不只是 caption/VQA。

**一句话结论：** Chameleon 把“统一”推到最彻底的离散 early fusion：单词表、单 decoder、单 AR 目标即可同时完成理解和图像生成；真正困难不在接口，而在图像量化损失、1,024-token 图像块的串行采样、跨模态 softmax 竞争及 10T-token 级训练稳定性。

![](assets/Chameleon%20-%20图文早期融合基础模型/figures/01-early-fusion-architecture.png)

> **Figure 1，PDF p.2。** 左侧输入图像经 tokenizer 直接变成蓝色 token，右侧同一自回归 LM 生成文本与图像 token；没有 late-fusion connector 或独立生成模型。来源：官方 PDF 完整图，已 Read 验证。

---

## 1. 问题定义：为何做全离散 early fusion

### 1.1 三种常见方案

1. **理解型 VLM**：vision encoder → projector → LLM，只输出文本。
2. **模块化统一系统**：LLM 决定何时调用外部扩散模型，视觉生成不在共享参数内。
3. **Chameleon**：图文都离散化并在序列起点融合，单 Transformer 直接建模联合分布。

Chameleon 的目标分布可以写成：

$$
p_\theta(z_{1:T})=\prod_{t=1}^{T}p_\theta(z_t\mid z_{<t}),\qquad
z_t\in\mathcal V_{\text{text}}\cup\mathcal V_{\text{image}}.
$$

因此 text→image、image→text、image→text→image 只是不同 token 排列，而非不同网络。

### 1.2 输入输出形式

| 场景 | 序列化输入 | 自回归输出 |
|---|---|---|
| 文本 LM | text | text |
| 图像理解 | `<start_image> image_tokens <end_image> question` | answer text |
| 文生图 | text `<start_image>` | 1,024 image tokens + `<end_image>` |
| 交错生成 | text / image token 任意交错 | text 与固定长度 image block 交错 |
| 视觉对话 | image + multi-turn text | text，必要时插入 image block |

---

## 2. Tokenizer、词表与 output head

### 2.1 图像 tokenizer

论文新训练的 tokenizer 基于 Gafni et al.：

$$
512\times512\ \text{RGB}\longrightarrow 32\times32=1024\ \text{indices},
$$

codebook 大小为 8,192。训练仅使用 licensed images，并将含人脸图片上采样 2 倍。其主要已知上界是图中文字重建差，因而 OCR 和精确文字生成会被 tokenizer 先验限制。

这类 VQ tokenizer 包含 image encoder、quantizer 与 image de-tokenizer；Transformer 只看 code index，不看连续 patch feature。视觉信息瓶颈位于量化：若 tokenizer 已丢掉小字、空间细节，扩大 LM 也无法恢复。

### 2.2 统一词表

SentencePiece BPE 总词表大小 65,536，**其中包含 8,192 个 image code token**，另有 image start/end 等控制 token。文本与图像 embedding 从训练开始就在同一空间竞争容量。

### 2.3 主干与输出投影

- decoder-only，整体沿用 LLaMA-2：RMSNorm、SwiGLU、RoPE；34B 使用 GQA。
- 单一序列 hidden state 通过统一 vocabulary projection 预测下一个 token；论文没有单列多模态 output head。
- 推理时为 image-only 等场景 mask 非目标模态 logits，避免从图像块跳到文本 token。
- 论文没有明确披露 embedding 与 LM head 是否 weight tying，因此复现时不能默认二者绑定。

---

## 3. Attention mask：全序列 causal，没有图像块双向 attention

Chameleon 对文本和图像 token 一视同仁，使用标准 causal mask：

$$
M_{ij}=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

后一个图像 token 能看前面的文本与图像 token，前一个图像 token 不能看后面的图像区域。其好处是实现最简单、KV cache 与文本 LM 完全一致；代价是二维图像被强制 raster-order，生成必须执行 1,024 次依赖步骤，且每个位置无法在同一层利用未来图像上下文。

这正是 Transfusion 的 intra-image bidirectional attention 与 Show-o Omni-Attention 要修补的点。

---

## 4. 训练目标与采样/解码

### 4.1 预训练损失

所有模态共享 next-token cross-entropy：

$$
\mathcal L_{\mathrm{NTP}}
=-\sum_{t=1}^{T}\log p_\theta(z_t\mid z_{<t}).
$$

不存在单独图像重建损失、MaskGIT loss 或 DDPM noise loss；图像 token 与文本 token 的差异只在 token ID 与数据排列。图文 pair 有 50% 采用 image→caption 顺序，另 50% 为 caption→image，使理解与生成同时获得直接监督。

SFT 时仅对 answer token 计算损失，prompt token masked：

$$
\mathcal L_{\mathrm{SFT}}
=-\sum_{t\in\mathcal A}\log p_\theta(z_t\mid z_{<t}).
$$

### 4.2 图像采样

1. 文本 AR 生成遇到 `<start_image>`；
2. 切换到图像 token admissible set，mask 文本 logits；
3. 顺序生成固定大小的 1,024-token block；
4. 输出 `<end_image>`；
5. image de-tokenizer 把 $32\times32$ code grid 还原成 $512^2$ 图像；
6. 若继续混合文档，恢复文本 token 采样。

论文披露了流式/非流式工程：非流式图像块可使用 fused kernel，流式模式则每步检查控制 token。具体 image sampling 的 temperature/top-$p$ 未在论文中公开，复现时应明确记录实现默认值，而不能从文本评测设置外推。

---

## 5. 为什么训练会不稳定

### 5.1 跨模态 softmax 竞争

图像与文本 entropy、频率和局部统计差异大，但共享 attention 与最终 softmax。作者观察到各模态通过增大 hidden/output norm 竞争，bf16 有效表示范围最终被耗尽；发散可能到训练 20%–30% 后才出现。

![](assets/Chameleon%20-%20图文早期融合基础模型/figures/02-stability-ablation.png)

> **Figure 5，PDF p.6。** output norm 的失控增长比 loss 发散更早出现；QK-Norm 是必要条件，7B 还受 dropout 影响。来源：官方 PDF 完整图，已 Read 验证。

### 5.2 稳定化配方

1. **QK-Norm**：对每层 attention 的 query/key 先归一化，约束 attention softmax 输入范数。
2. **Norm reordering**：34B 采用 post-branch norm：

$$
h=x+\operatorname{Norm}(\operatorname{Attn}(x)),\qquad
out=h+\operatorname{Norm}(\operatorname{FFN}(h)).
$$

而 LLaMA-2 是对分支输入做 pre-norm。新顺序约束 residual 分支输出，避免 SwiGLU 乘性放大。

3. **z-loss**：对最终 softmax partition function 加

$$
\mathcal L_z=10^{-5}(\log Z)^2,\qquad Z=\sum_i e^{x_i},
$$

抑制 logit drift。
4. AdamW：$\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-5}$；warm-up 4,000；weight decay 0.1；gradient clip 1.0。
5. 7B 使用 dropout 0.1；34B 的 norm-reordering 与 dropout 组合不佳，最终不用 dropout。

### 5.3 重要消融结论

- 去掉 QK-Norm，7B 约一个 epoch 的 20% 后发散。
- 去掉 image generation 的训练不发散，说明视觉输出而非单纯视觉输入是主要稳定性压力。
- 仅加 dropout 无法稳定 34B；norm reordering 必须加入。
- 这些改动会影响文本效率：Transfusion 的受控实验显示，Chameleon stability recipe 与 image-token LM 都造成额外 text perplexity 损失。

---

## 6. 数据、训练与对齐

### 6.1 预训练数据

| 数据 | 规模 | 作用 |
|---|---:|---|
| Text-only | 2.9T token | LLaMA-2 + CodeLLaMA 语料，保持语言/代码能力 |
| Text-image pairs | 1.4B pairs / 1.5T token | captioning 与 T2I，图片 $512^2$ center crop |
| Interleaved text-image | 400B token | 长文档内图文早期融合 |

训练分两阶段：前 80% 使用大规模无监督混合；后 20% 将前述数据权重减半，加入更高质量和过滤后的 instruction tuning 数据。训练集唯一规模约 4.4T token，跑 2.1 epochs 后实际看过约 **9.2T token**，所以论文概括为约 10T。

### 6.2 计算与模型

- 7B：1,024 张 A100 80GB，856,481 GPU-hours；
- 34B：3,072 张 A100 80GB，4,282,407 GPU-hours；
- 全局 batch 约 8M/12M token；学习率均 $10^{-4}$。

### 6.3 SFT

SFT 类别包含 Text 1.6M、Code 14.1K、Visual Chat 15.6K、Image Generation 64.3K、Interleaved Generation 16.9K、Safety 95.3K。训练使用 cosine LR $10^{-5}$、batch 128、context 4096、weight decay 0.1、dropout 0.05，并对 prompt loss mask。

模态配比很关键：若 image-output 样本过多，模型会学到无条件插入图片；过少则只输出文本。early fusion 让“何时切换模态”也成为 token prior 的一部分。

---

## 7. 结果：理解能力只是证据的一半

### 7.1 图像理解与文本能力

- Chameleon-34B 2-shot：COCO CIDEr 120.2、Flickr30k 74.7、VQAv2 66.0。
- SFT：COCO 140.8、Flickr30k 82.3；multi-task VQAv2 69.6。
- 文本上 34B 在多项常识/阅读任务接近 Mixtral 8x7B，并在部分任务超过 LLaMA-2 70B；但它使用更多 token 与 compute，不能把差异全部归因于多模态协同。

### 7.2 原生混合模态输出

![](assets/Chameleon%20-%20图文早期融合基础模型/figures/03-mixed-modal-human-eval.png)

> **Figure 9，PDF p.12。** 1,048 个要求混合输出的自然 prompt 上，Chameleon 完全完成任务的比例更高；相对评测中也常胜过文本模型 + DALL-E 3 拼装基线。来源：官方 PDF 完整图，已 Read 验证。

- absolute fulfillment：Chameleon 55.2%，GPT-4V+ 44.7%，Gemini+ 37.6%。
- pairwise：对 Gemini+ 为 41.5% win / 34.5% tie / 24.0% loss；对 GPT-4V+ 为 35.8/31.6/32.6。
- 论文所称 60.4%/51.6% “preference rate”按 win + 0.5 tie 计算。

这组实验比普通 VQA 更能证明原生视觉输出价值，但 baseline 是“闭源文本响应 + DALL-E 3 补图”的拼装系统，且所有 prompt 都预期图像输出，天然不利于只返回文本的模型。

### 7.3 视觉生成证据缺口

论文展示大量 $512^2$ 生成与交错文档样例，但没有像后来的 Transfusion/Show-o 那样系统报告 COCO FID、CLIP、GenEval，也缺少 tokenizer/codebook/图像 AR mask 的受控质量消融。因此可确认其“能原生生成并融入长文档”，但不能仅凭本文证明其单图生成质量达到同期专业扩散模型。

---

## 8. 局限与批判性判断

1. **图像量化瓶颈**：8,192 codebook、1,024 token；OCR 与小字重建先天弱。
2. **采样慢**：$512^2$ 图像需 1,024 次 causal token 生成；没有图像块双向/并行去噪。
3. **统一 softmax 竞争**：图像、文本共享词表和 head，导致 norm drift，稳定化配方昂贵且会影响文本效率。
4. **高成本**：34B 超过 4.28M A100 GPU-hours，完整预训练难以外部复现。
5. **数据不可完全复现**：1.4B pairs 含 licensed data，400B interleaved web token 的清洗与组成未公开。
6. **生成评测不足**：缺少标准 T2I quantitative benchmark 与 tokenizer 重建指标。
7. **人评边界**：任务 fulfillment 的 Krippendorff’s alpha 约 0.338，且 prompt 非真实线上日志；“胜过 GPT-4V+”不等价于一般理解能力更强。
8. **固定分辨率/块长**：原生 512 center crop，复杂宽高比和高分辨率没有专门机制。

**判断**：Chameleon 的历史价值是证明“共享词表 + 单 causal Transformer”在大规模上可行，并暴露其稳定性边界。它是统一建模的干净基线，但未解决连续视觉生成最适合何种目标；Transfusion 和 Show-o 正是在此处分别引入连续 DDPM 与离散 diffusion。

---

## 9. 复现建议

1. 先单独报告 tokenizer 的 reconstruction FID、LPIPS、OCR 与人脸质量，再训练 LM；不要把 tokenizer 错误归因于 Transformer。
2. 以小规模模型复现 text/image 两模态，持续记录每模态 output norm、attention logit 与最终 $\log Z$。
3. 从 QK-Norm + z-loss 起步；分别消融 norm reordering/dropout，不能直接照搬 7B 配置到 34B。
4. 同时记录每图 1,024-step 延迟、KV cache、词表 mask 开销与 T2I 指标。
5. 与 Transfusion 对比时统一 VAE 数据、encoder/decoder 与 compute，避免离散/连续 tokenizer 质量成为混杂变量。
6. 交错生成评测需分开报告：文本正确性、图像质量、图文对应、图像插入位置与整体任务完成率。

---

## 10. 四路线对照中的位置

| 维度 | Chameleon 选择 | 直接后果 |
|---|---|---|
| 主干 | decoder-only | 结构最像 LLM，统一 KV cache |
| 图像 tokenizer | VQ，8,192 codes | 可纳入单词表，但有量化损失 |
| output head | 单离散 vocabulary projection | 文本/图像共享 softmax，需模态 mask |
| attention | 全序列 causal | 图像串行、无块内双向信息 |
| 图像 loss | NTP/CE | 训练统一但不利用 diffusion 并行去噪 |
| 解码 | 1,024 image tokens AR | 易做交错生成，推理慢 |

---

## 11. 材料路径与自检

| 检查项 | 路径/结论 | 状态 |
|---|---|---|
| 笔记 | `10-Papers/VLM/Chameleon - 图文早期融合基础模型.md` | 已完成 |
| 官方 PDF | `assets/Chameleon - 图文早期融合基础模型/sources/paper.pdf` | 已保存 |
| 来源记录 | `assets/Chameleon - 图文早期融合基础模型/sources.md` | 已完成 |
| 完整原图 | `figures/01-*.png`、`02-*.png`、`03-*.png` | 3 张，已逐张 Read 验证 |
| 编号标题 | 0–11 连续 | 通过 |
| 公式 | 联合 AR、causal mask、NTP、SFT、z-loss | 通过 |
| 视觉输出架构 | tokenizer、词表/head、mask、loss、采样均覆盖 | 通过 |
| Paper-Index | 未写入、未修改 | 通过 |
