---
type: paper-note
status: complete
domain: VLM
paper: "Ovis: Structural Embedding Alignment for Multimodal Large Language Model"
year: 2024
arxiv: "2405.20797"
doi: null
source: "https://arxiv.org/abs/2405.20797"
project: null
code: "https://github.com/AIDC-AI/Ovis"
tags:
  - VLM
  - structural-embedding-alignment
  - visual-tokenizer
  - embedding-table
  - probabilistic-token
  - Ovis
created: 2026-10-16
updated: 2026-10-16
---

# Ovis：结构化视觉嵌入对齐

## 0. 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | Ovis: Structural Embedding Alignment for Multimodal Large Language Model |
| 作者 | Shiyin Lu、Yang Li、Qing-Guo Chen、Zhao Xu、Weihua Luo、Kaifu Zhang、Han-Jia Ye |
| 模型变体 | Ovis-Qwen1.5-7B/14B、Ovis-Llama3-8B |
| 视觉侧 | CLIP ViT-L/14@336px |
| 视觉词表 | $K=2^{17}=131{,}072$ |
| 核心设计 | probabilistic visual token + 独立 Visual Embedding Table |
| 本地论文 | [paper.pdf](assets/Ovis%20-%20结构化视觉嵌入对齐/sources/paper.pdf) |
| 来源清单 | [sources.md](assets/Ovis%20-%20结构化视觉嵌入对齐/sources.md) |

**一句话结论：** Ovis 不再用 MLP 把连续视觉特征直接投到 LLM hidden space，而是先让每个 patch 对 131K“视觉词”产生概率分布，再以该分布从 Visual Embedding Table 取期望 embedding；在相同 backbone、数据和参数量对照下，这个 soft vocabulary bottleneck 平均领先传统 connector 8.8%。

---

## 1. 问题定义与动机

### 1.1 作者所谓“结构不对齐”

文本输入先被 tokenizer 离散成 one-hot token，再从 textual embedding table 查出向量。典型 MLLM 的图像则经 ViT 得到连续 patch feature，再由 Linear/MLP/Q-Former/cross-attention connector 映射到 LLM 维度。两者最终 shape 相同，却由不同的生成机制产生：文本 embedding 是词表原型的索引，视觉 embedding 是任意连续投影。

Ovis 的问题是：若视觉也经过“词表索引→embedding”结构，是否能让 LLM 更容易接收？它不要求视觉 token 硬离散，而是把一个 patch 表示成多个 visual words 的概率混合，以保留多义性。

![](assets/Ovis%20-%20结构化视觉嵌入对齐/figures/01-connector-comparison.png)

> 图 1：Figure 1。传统 connector 只对齐维度；Ovis 增加 visual tokenizer 与 Visual Embedding Table，使两种模态都通过 embedding table 形成 LLM 输入。

### 1.2 核心贡献

1. 提出 **probabilistic visual token**：每个连续 patch feature 映射为视觉词表上的概率分布，而非单一离散 code。
2. 提出独立 **Visual Embedding Table**：概率 token 对词表 embedding 加权求和，生成与 textual lookup 结构相似的视觉 embedding。
3. 全程只用文本生成 loss，通过三阶段冻结/解冻把语言梯度反传到视觉词表，不引入图像重建 decoder 或 VQ loss。
4. 在相同 LLM、ViT、数据和参数量的 connector 对照中全项提升，平均相对优势 8.8%。

---

## 2. 输入输出与张量形状

### 2.1 从图像到连续 patch representation

设图像 $I\in\mathbb{R}^{C\times W\times H}$，patch 大小为 $w\times h$，patch 数为：

$$n=\left\lceil\frac{W}{w}\right\rceil\left\lceil\frac{H}{h}\right\rceil$$

ViT $g_\theta$ 将每个 patch 变为 $r_i\in\mathbb{R}^{d}$，组成 $R\in\mathbb{R}^{n\times d}$。实际实现用 CLIP ViT-L/14@336px，规则网格约为 $24\times24$；是否保留额外 CLS token 应以发布代码配置为准。

### 2.2 概率视觉 token

Visual Head 是线性层 $W\in\mathbb{R}^{K\times d}$ 加 softmax。对第 $i$ 个 patch：

$$v_i=\operatorname{softmax}(Wr_i),\quad v_i\in\Delta^K$$

$v_i$ 不是整数 token id，而是对 $K$ 个 visual words 的概率分布。每个分量可解释为 patch 与一个视觉原型的相关性。由于不做 argmax，同一 patch 可以同时表达颜色、纹理、部件和对象等多个模式。

### 2.3 Visual Embedding Table

令视觉 embedding table 为 $E_v=[e_1;\ldots;e_K]\in\mathbb{R}^{K\times d'}$，其中 $d'$ 与 LLM textual embedding dimension 相同。最终视觉 embedding 为：

$$V_i=\sum_{k=1}^{K}v_{i,k}e_k=v_i^\top E_v\in\mathbb{R}^{d'}$$

也可写成期望：

$$V_i=\mathbb{E}_{k\sim v_i}[e_k]$$

文本 token $t_j$ 是 one-hot，直接从 $E_t\in\mathbb{R}^{|\mathcal{V}_t|\times d'}$ 取一行；视觉 token $v_i$ 则是 soft index，从 $E_v$ 取多行加权和。所谓 structural alignment 指的正是这层生成过程的同构，而不是强迫视觉和文本共用同一词表。

![](assets/Ovis%20-%20结构化视觉嵌入对齐/figures/03-structural-alignment.png)

> 图 2：Figure 3。patch 经 ViT 与 Linear+Softmax 变成概率 token，再从独立视觉表取多个 embedding 的加权组合；LLM 同时接收这种视觉 embedding 与普通 textual embedding。

### 2.4 多模态序列与输出

文本序列中 `<image>` 位置被 $n$ 个视觉 embedding 替换：

$$[T_1,\ldots,T_{\lambda-1},V_1,\ldots,V_n,T_{\lambda+1},\ldots,T_m]$$

LLM 输出普通文本 token $o_{1:l}$，覆盖 VQA、OCR、数学、描述与多轮指令回答。Ovis 不生成离散 visual word，也不重建图像；“视觉 tokenization”只发生在输入 embedding 构造中。

---

## 3. Q/K/V、mask、position、目标与生成

### 3.1 Vision Transformer

patch embedding 加视觉位置编码后进入 CLIP ViT。每层 self-attention 中：

$$Q_v=RW_Q^v,\quad K_v=RW_K^v,\quad V_v=RW_V^v$$

视觉 attention 在整张规则网格上双向建模，没有因果 mask。论文使用固定 336px 输入，没有额外动态高分辨率或多尺度 tile；这也是其高分辨率上限之一。

### 3.2 LLM 自注意力

经过 Visual Embedding Table 后，视觉与文本已经同维度，直接进入 Qwen1.5/Llama3 decoder。每层统一产生：

$$Q_h=HW_Q^h,\quad K_h=HW_K^h,\quad V_h=HW_V^h$$

LLM 使用标准 causal mask。位于图像之后的 prompt/answer token 可以关注此前全部视觉 embedding；视觉 token 与此前文本能按顺序交互，但不能看未来答案。padding 位置应由 attention mask 排除。Ovis 没有独立 cross-attention，也没有 query token；跨模态融合完全发生在 LLM self-attention 内。

### 3.3 Position

- 视觉 patch 的二维位置由预训练 CLIP ViT 的位置表示承担。
- 插入 LLM 后，视觉 embedding 与文本 token 共用一维序列位置；论文没有提出新的 2D-RoPE、位置重排或视觉专属 position id。
- 因而 Ovis 对齐的是 **embedding formation**，不是空间位置编码。高分辨率、长宽比和多图位置问题仍未解决。

### 3.4 单一生成目标

三阶段始终使用 ground-truth 文本的 next-token cross-entropy：

$$\mathcal{L}_{\mathrm{gen}}=-\sum_{t=1}^{l}\log p_\phi(o_t^*\mid o_{<t}^*,I,x)$$

梯度从 LLM 输出经 causal self-attention 回传到 $E_v$、Visual Head $W$ 与 ViT。与 VQ-VAE/VQGAN 不同，没有 codebook commitment、图像 reconstruction 或 adversarial loss，因此视觉词的语义由“哪些原型最有利于文本生成”共同决定。

推理仍是标准文本自回归生成。概率视觉 token 在输入阶段一次计算，不随解码 step 重采样；$V_i$ 是确定性的 soft expectation。

---

## 4. 与 connector 的本质对照

### 4.1 代数视角

传统两层 MLP connector 可写为：

$$V_i=W_2\sigma(W_1r_i)$$

Ovis 可写为：

$$V_i=E_v^\top\operatorname{softmax}(Wr_i)$$

两者都是 $\mathbb{R}^{d}\rightarrow\mathbb{R}^{d'}$ 的非线性映射。差别在中间层：MLP 使用逐维激活，Ovis 使用归一化概率 simplex；所有中间维之和为 1，因此输出位于视觉原型 embedding 的凸包中。这个约束提供“词表选择”归纳偏置，也让中间状态可分析稀疏性。

### 4.2 它不是硬离散 tokenizer

论文反复使用 token/index 语言，但 $v_i$ 是 dense softmax、$V_i$ 是期望，没有 argmax、采样或 straight-through estimator。因此：

- 优点：端到端可微，不丢失 patch 的多义语义，不需要重建 loss。
- 代价：并未获得真正离散 token 的压缩、缓存或可组合符号优势；$K=131{,}072$ 的 logits 和 embedding 加权本身很重。
- 解释边界：若概率不稀疏，Visual Embedding Table 只是一个大 softmax bottleneck connector；Appendix E 的稀疏统计用于缓解这一质疑，但未证明 visual words 具有人类可解释语义。

### 4.3 公平 connector 消融

对照模型使用相同 Qwen1.5-7B-Chat、CLIP ViT-L/14@336、相同训练数据和训练流程；connector 是两层 GELU MLP，hidden size 设为视觉词表大小，使参数量与 Ovis 对齐。结果：

| 架构 | MMStar | MMB EN/CN | MMMU val/test | MathVista | MME | HallusionBench | RealWorldQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP connector | 41.1 | 71.0 / 65.2 | 34.8 / 33.8 | 36.3 | 1757 | 54.0 | 56.1 |
| Ovis | 44.3 | 75.1 / 70.2 | 39.7 / 37.7 | 41.4 | 1882 | 56.4 | 60.0 |

相对提升为 4.4%–14.1%，平均 8.8%。这是论文最有说服力的证据，因为 backbone、数据和参数量被显式控制；但 MLP hidden size 被放大到 131K 并非常见高效 connector，其优化特性未必代表所有 projector。

---

## 5. 三阶段训练与数据

### 5.1 Stage 1：建立视觉词表接口

冻结 LLM 和 ViT 大部分层，随机重置 ViT 最后一个 block；训练该 block、Visual Head $W$ 与 $E_v$。使用从 COYO 过滤出的 10M caption 样本，两个 CLIP similarity 分数均须大于 0.36。输入格式类似 `<image>'s caption:`，目标是 caption。

这个阶段避免随机视觉词表直接扰动 LLM，也让最后一个视觉 block 开始围绕语言生成而非原始 CLIP 对比目标组织特征。

### 5.2 Stage 2：视觉全量适配

LLM 仍冻结，解冻整个 ViT，并继续训练 $W$ 与 $E_v$。数据约 2M visual descriptions：LLaVA-Pretrain 558K、ShareGPT4V-Pretrain 82K、ALLaVA caption 两组 485K/203K，以及约 1.011M 自建 LAION/CC12M descriptions。样本被转为对话式描述，语义比短 alt-text 更丰富。

### 5.3 Stage 3：全模型 instruction tuning

解冻 LLM，训练全模型约 3M multimodal instructions。数据包含 ScienceQA、TextVQA、LLaVA-Finetune、ArXivQA、Q-Instruct、Geo、LRV、Chart、SynthDoG OCR、ALLaVA instruction 与自建 CC12M-QA 等。此阶段让视觉词表和 LLM 同时适配问答、推理和指令格式。

### 5.4 训练超参数与成本

| 配置 | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|
| batch size | 8192 | 1024 | 1024 |
| Qwen 版学习率 | $10^{-4}$ | $10^{-4}$ | $2\times10^{-5}$ |
| Llama3 版学习率 | $10^{-4}$ | $10^{-4}$ | $10^{-5}$ |
| warmup ratio | 0.1 | 0.1 | 0.05 |
| epoch | 1 | 1 | 1 |
| optimizer / precision | AdamW / bf16 | AdamW / bf16 | AdamW / bf16 |

128 张 H100 上，7B/8B 三阶段约 6/2/7 小时，14B 约 10/6/21 小时。论文给出的硬件时间相对透明，但自建描述和指令数据依赖 Gemini-Pro/GPT-4V API，其内容质量与成本仍难完全复刻。

---

## 6. 实验结果

### 6.1 同参数档结果

![](assets/Ovis%20-%20结构化视觉嵌入对齐/figures/02-main-results.png)

> 图 3：Figure 2。Ovis-Llama3-8B 与 Ovis-Qwen1.5-14B 在多个综合基准形成更大的雷达面积；图只适合看覆盖面，MME 与百分制指标尺度不同，不应按面积作严格统计。

Ovis-Llama3-8B 在 MMStar/MMBench-EN/MMBench-CN/MMMU-val/MMMU-test 为 49.5/77.4/72.8/44.7/39.0；MathVista 40.8、MME 2009、HallusionBench 61.1、RealWorldQA 57.9。Ovis-Qwen1.5-14B 为 48.5/78.4/76.6/46.7/40.7，MathVista 43.4、MME 1961、HallusionBench 57.6、RealWorldQA 62.7。

论文称 14B 总体超过 Qwen-VL-Plus，并在部分项目接近 Qwen-VL-Max。需要注意闭源模型分数来自统一工具或公开报告，不代表相同 prompt、分辨率和版本；“overall”不应扩展为每项领先。

### 6.2 视觉利用与幻觉

MMStar 强调问题必须依赖图像，Ovis-8B 的 49.5 高于同档多数开源模型；HallusionBench 和 MME 也较强。结合 connector 对照，可认为结构化 bottleneck 确实改善视觉条件的可用性。但论文没有给出 grounding attribution、attention map 或 counterfactual image test，不能直接证明提升来自更“忠实”的视觉 grounding。

### 6.3 概率 token 的稀疏性

![](assets/Ovis%20-%20结构化视觉嵌入对齐/figures/04-token-sparsity.png)

> 图 4：Figure 8。10K 张 ImageNet-1K 图像上，仅 0.22% 概率值高于 $10^{-4}$，20.16% 位于 $10^{-5}$–$10^{-4}$，79.62% 低于 $10^{-5}$；softmax 输出经验上高度稀疏。

稀疏性说明一个 patch 的有效原型数远少于 131K，但统计只按阈值计数，没有报告 entropy、top-k 累积质量、原型使用均衡度或语义可视化。它支持“soft multi-index 并非均匀平均”，尚不足以支持“视觉词表形成离散概念”。

---

## 7. 局限与批判性判断

1. **高分辨率缺失。** 固定 336px ViT，没有 tile/任意宽高比机制；论文也明确承认高分辨率任务受限。
2. **只训练单图。** 多图比较、跨页文档和视频没有进入训练，LLM 序列中多个视觉块的 position 与 mask 行为未经验证。
3. **“结构对齐”只发生在 embedding 层。** LLM 内部仍以连续 hidden state 运算，视觉词表与文本词表不共享，不能自动得到跨模态符号对齐。
4. **计算与内存代价大。** $n\times131{,}072$ logits 和同规模 embedding table 很昂贵；论文没有报告相对普通 projector 的吞吐、显存和参数访问开销。
5. **softmax 参数存在可辨识性问题。** $W$ 与 $E_v$ 可共同旋转/重排 visual words，多组词表都可能产生相近输出；visual word 的稳定语义并无约束。
6. **消融宽度不足。** 只有一个等参数两层 MLP 对照，没有比较更常见的小 MLP、Q-Former、Perceiver、共享 textual table、hard/top-k routing 或不同 $K$。
7. **数据并非完全开放可复刻。** 虽然以开放数据为主，仍含自建 API 标注；闭源模型生成的数据会传递偏差和幻觉。
8. **安全风险。** 作者明确指出 hallucination、bias 和潜在误导；建议内容审核与透明发布，但没有给出专门安全评测。

我的判断是：Ovis 最值得保留的是“在 connector 内加入概率 simplex 和原型表”的可检验假设，而不是把它宣传为真正离散视觉语言。它提供了强受控增益，但仍需 top-k、共享/独立词表、吞吐和可解释性实验，才能确定收益来自结构同构、参数化方式还是优化条件。

---

## 8. 复现路线

### 8.1 最小实现

```text
image → CLIP ViT → R[n,d]
R → Linear(d,K) → softmax → P[n,K]
P @ VisualEmbeddingTable[K,d_llm] → V[n,d_llm]
replace <image> with V → causal LLM → text
```

训练顺序严格复刻三阶段：最后 ViT block+$W$+$E_v$；全 ViT+$W$+$E_v$；全模型。任何从第一步直接全解冻的实现都改变了论文的稳定化条件。

### 8.2 必做对照

1. 同参数量 GELU MLP connector。
2. $K\in\{8K,32K,131K\}$，同时报告准确率、吞吐与峰值显存。
3. dense softmax、top-k softmax、hard argmax+STE，比较语义保真与计算。
4. 独立 $E_v$、共享 textual table、由 textual table 初始化后分离三种方案。
5. 报告 entropy、top-1/top-8 累积概率、原型使用频率与 dead entries。

### 8.3 工程自检

- softmax 应沿视觉词表维 $K$ 计算，而不是 patch 维。
- $E_v$ 输出维必须与 LLM embedding 对齐；visual position ids 与替换后的 token 数一致。
- LLM causal mask 和 label mask 要排除视觉、prompt 与 padding，只监督 assistant 输出。
- 大词表 matmul 应检查 bf16 数值稳定性；必要时 logits 用 fp32 softmax，但需报告实现差异。
- 训练和评测固定 ViT resize/crop，避免把分辨率预处理差异误归因于结构。

---

## 9. 与其他 connector 路线的关系

| 路线 | 压缩/映射机制 | 跨模态交互位置 | 特点 |
|---|---|---|---|
| Linear/MLP（LLaVA） | 连续投影 | LLM self-attention | 简单、高效，缺少显式瓶颈语义 |
| Q-Former（BLIP-2） | learned queries + cross-attention | LLM 前 | 固定 query 数，先学习对齐 |
| Perceiver（Flamingo/IDEFICS2） | latent queries 汇聚 patch | LLM 前或多层 xattn | 固定视觉 token 预算 |
| Cross-attention（Flamingo） | 文本 Q、视觉 K/V | 插入 LLM 多层 | 深层视觉注入，但路径较重 |
| **Ovis** | $\operatorname{softmax}(Wr)E_v$ | LLM 前 | 不减 token 数，用视觉词表约束 embedding 生成 |

Ovis 解决的是 embedding parameterization，不解决高分辨率 token 爆炸；Perceiver/pixel shuffle 解决的是序列长度，不保证“词表式”结构。两者理论上可以组合，但论文未验证。

---

## 10. 图表索引

1. `figures/01-connector-comparison.png`：传统 connector 与结构化视觉 embedding 对比。
2. `figures/02-main-results.png`：7B/14B 同档 benchmark 总览。
3. `figures/03-structural-alignment.png`：概率视觉 token 与 Visual Embedding Table 全流程。
4. `figures/04-token-sparsity.png`：视觉词表概率分布稀疏性。

## 11. 材料来源

- [官方论文 PDF](assets/Ovis%20-%20结构化视觉嵌入对齐/sources/paper.pdf)
- [来源与原图清单](assets/Ovis%20-%20结构化视觉嵌入对齐/sources.md)
