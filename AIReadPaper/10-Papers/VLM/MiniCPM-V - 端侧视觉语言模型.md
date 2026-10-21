---
type: paper-note
status: done
domain: VLM
paper: "MiniCPM-V: A GPT-4V Level MLLM on Your Phone"
year: 2024
arxiv: "2408.01800"
doi: "10.48550/arXiv.2408.01800"
source: "https://arxiv.org/abs/2408.01800"
project: "https://github.com/OpenBMB/MiniCPM-V"
code: "https://github.com/OpenBMB/MiniCPM-V"
tags:
  - VLM
  - Edge-AI
  - Adaptive-Slicing
  - Perceiver-Resampler
  - RLAIF-V
  - Quantization
created: 2026-10-16
updated: 2026-10-16
---

# MiniCPM-V：端侧视觉语言模型

## 基本信息

- **作者：** Yuan Yao、Tianyu Yu、Ao Zhang 等，MiniCPM-V Team / OpenBMB
- **论文对象：** MiniCPM-V 1.0、2.0 与 MiniCPM-Llama3-V 2.5 系列，重点是 2.5
- **本地原文：** [paper.pdf](assets/MiniCPM-V%20-%20端侧视觉语言模型/sources/paper.pdf)
- **完整溯源：** [sources.md](assets/MiniCPM-V%20-%20端侧视觉语言模型/sources.md)

**一句话结论：**

MiniCPM-V 的“端侧”不是只靠小 LLM，而是一条贯通输入、训练、对齐和部署的系统路线：自适应切片保留任意宽高比与高分辨率细节，单层 Perceiver Resampler 把每片 1024 patch 压到 64/96 token，三阶段冻结策略避免低质图文数据污染 LLM，RLAIF-V 用原子 claim 反馈降低幻觉，最后通过 INT4、顺序加载、编译/配置优化与 NPU 加速把 8.5B 模型落到手机。

---

## 1. 研究目标与系列演进

### 1.1 端侧约束

手机/PC 的主要瓶颈不是单一 FLOPs，而是权重内存、视觉 prefill、KV cache、散热与框架支持。论文的目标是同时优化：

1. **模型规模：** 2.8B（V1.0/2.0）与 8.5B（2.5），而非数十至数百 B。
2. **视觉 token：** 高分辨率仍控制在 96–960 token，而同代拼接路线常为 1728–2880。
3. **质量：** OCR、任意宽高比、低幻觉与多语言不能因压缩而明显丢失。
4. **真实部署：** 给出 Xiaomi 14 Pro、vivo X100 Pro、MacBook Pro M1 的端侧测量。

### 1.2 三代模型

| 模型 | Base LLM | 分辨率/宽高比 | 预训练 | 对齐 |
|---|---|---|---|---|
| MiniCPM-V 1.0 | MiniCPM 2B | 0.2M pixels，固定比例 | Stage 1–2 | 无 |
| MiniCPM-V 2.0 | MiniCPM 2B | 1.8M pixels，任意比例 | Stage 1–3 | RLHF-V |
| MiniCPM-Llama3-V 2.5 | Llama3-Instruct 8B | 1.8M pixels，任意比例 | Stage 1–3 | RLAIF-V |

论文标题中的“GPT-4V level on your phone”主要指 2.5：8.5B、OpenCompass 65.1、可 4-bit 量化到约 5GB，并在 Snapdragon 8 Gen 3 手机上运行。

---

## 2. 总体架构与 shape

![](assets/MiniCPM-V%20-%20端侧视觉语言模型/figures/01-architecture.png)

> **图 1：总体架构与自适应视觉编码。** 原图（0）与局部切片（A–F）分别编码，经共享压缩层后按二维布局插入换行/边界 token，再与文本共同送入 LLM。来源：论文 Figure 3；官方 TeX 原图，已逐图复核。

### 2.1 三个模块

1. **Vision encoder：** SigLIP SoViT-400M/14。
2. **Shared compression layer：** 单层 cross-attention 的 Perceiver Resampler，带适量 learnable queries 与二维位置。
3. **LLM：** V1/V2 为 MiniCPM 2.4B，2.5 为 Llama3-Instruct 8B；视觉 soft tokens 与文本 token 共同做条件文本生成。

若每片送入视觉 encoder 后有 1024 个 patch token，$S$ 个切片形成

$$
H_v\in\mathbb{R}^{S\times1024\times d_v}.
$$

Resampler 对每片用 $K$ 个 query：

$$
Q\in\mathbb{R}^{K\times d},\qquad
Z_s=\operatorname{CrossAttn}(Q,H_{v,s},H_{v,s})\in\mathbb{R}^{K\times d}.
$$

V1/V2 的 $K=64$，2.5 的 $K=96$。拼接后视觉序列约为 $SK$，再投到 LLM hidden size。论文给 2.5 的视觉 token 范围为 96–960，说明总切片数最多约 10。

### 2.2 Decoder 与生成

视觉 token 与文本 token 输入 decoder-only LLM，使用自回归条件生成：

$$
\mathcal L_{LM}=-\sum_t\log p(y_t\mid y_{<t},Z_{1:S},x_{text}).
$$

没有独立 OCR、grounding 或分类头；表格转 Markdown、VQA、caption、坐标与多语言对话均由文本 token 生成。Grounding 坐标归一化到 $[0,1]$。论文没有给出统一 decoding 超参数，因此 benchmark 复现必须同时保存 prompt、temperature、beam/sample 与 max tokens。

---

## 3. 自适应切片：保分辨率又控 token

### 3.1 理想切片数

输入图像大小为 $(W_I,H_I)$，ViT 预训练大小为 $(W_v,H_v)$：

$$
N=\left\lceil\frac{W_IH_I}{W_vH_v}\right\rceil.
$$

候选网格 $(m,n)$ 满足 $mn\in\{N-1,N,N+1\}$。用宽高比匹配分数

$$
S(m,n)=-\left|\log\frac{W_I/m}{H_I/n}-\log\frac{W_v}{H_v}\right|,
$$

选择

$$
(m^*,n^*)=\arg\max_{(m,n)\in\bar{\mathbb C}}S(m,n).
$$

引入 $N\pm1$ 是为了避免 $N$ 为素数时只剩极端的 $N\times1$ 或 $1\times N$。实践中 $N<10$，上限约 1.8M pixels，例如 1344×1344。

### 3.2 切片编码与位置

每片按比例缩放，使面积接近 ViT 预训练面积；ViT 的一维位置 embedding

$$
P_1\in\mathbb{R}^{Q\times d_v},\quad Q=q^2
$$

先还原为 $P_2\in\mathbb{R}^{q\times q\times d_v}$，再二维插值到该切片网格。除局部切片外，还加入原图作为 global slice，补充整体语境。

切片输出以 `<slice>`、`</slice>` 包裹，不同行之间插入 `\n`，显式保留二维 schema。这样 LLM 不只看到一串局部 token，还知道它们在原图中的相对排布。

### 3.3 Attention 与 mask

- Resampler：learnable query 对每个切片视觉特征做 cross-attention，完成固定长度压缩。
- LLM：视觉与文本 token 进入同一 decoder 序列，沿用 causal attention；视觉前缀可被后续回答访问。
- Data packing：多个样本拼成固定长度时，论文明确修改 position ids 与 attention masks，使样本间互不可见，避免跨样本泄漏。
- 论文没有完整公开多切片 mask 矩阵、query head 数与 projector 维度；这些需以官方代码为准。

---

## 4. 三阶段预训练与冻结策略

### 4.1 Stage 1：只暖启动压缩层

- 视觉 encoder 与 LLM 冻结。
- 随机初始化并只训练 compression layer。
- 224×224 输入，从 caption 数据随机抽 200M。

目标是先建立视觉空间与 LLM embedding 的低风险连接。

### 4.2 Stage 2：只扩展视觉分辨率

- 视觉分辨率 224→448。
- **只训练整个 vision encoder**，其他模块冻结。
- 再抽取 200M caption 数据。

这是较少见的隔离设计：先让视觉 backbone 适应高分辨率，不同时移动语言空间。

### 4.3 Stage 3：适配任意比例与 OCR

- 训练 vision encoder + compression layer。
- LLM 继续冻结，避免低质量网页 caption/OCR 直接破坏语言能力。
- 加入 OCR 与知识数据，启用自适应切片。

预训练总体数据包括约 520M 英中 caption 与 50M OCR+knowledge。数据 packing 使预训练加速约 2–3×。

### 4.4 SFT：全参数解冻

高质量 SFT 阶段解冻全部参数。Part 1 侧重短回答识别，Part 2 侧重长回答、复杂交互与指令遵循；两部分按顺序输入。2.5 还加入 2M Cauldron 数据与 90K/36 语言样本。

冻结策略的核心原则是：**低质量大规模数据只塑造视觉与桥接层，高质量指令数据才允许更新 LLM。**

---

## 5. RLAIF-V：把幻觉反馈拆成原子 claim

![](assets/MiniCPM-V%20-%20端侧视觉语言模型/figures/02-rlaif-v.png)

> **图 2：RLAIF-V 三阶段。** 同一 policy 高温采样多答复；LLM 将答复拆成原子 claim，开源 MLLM 逐条转 yes/no 验证；按错误 claim 数构造偏好对并做 DPO。来源：论文 Figure 4；官方 TeX 原图，已逐图复核。

### 5.1 Response generation

对每条指令从待对齐模型 $M$ 高温采样 10 个回答

$$
Y=\{y_1,\ldots,y_{10}\}.
$$

使用同一 policy 的好处是偏好样本处于模型自身分布，避免把不同模型的文风差异误当事实质量。

### 5.2 Divide-and-conquer feedback

Llama-3 8B 把回答 $y_i$ 拆为原子 claim 集合 $C_i$。每个 claim 改写为 yes/no 问题，再由开源 MLLM 结合图像验证。2.0 用 OmniLMM-12B，2.5 用 LLaVA-NeXT-Yi-34B。回答分数为

$$
s_i=-n_{rej},
$$

即被拒绝的错误 claim 越少越好。

### 5.3 DPO

从每组回答随机采样不同分数的 $(y_w,y_l)$，构造 3K 图像、6K preference pairs 做 DPO。论文没有在本文重写 DPO 公式，目标可写为

$$
\mathcal L_{DPO}=-\mathbb E\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]\right).
$$

注意：这是对回答事实性的偏好优化，不保证 reasoning trace 本身正确；验证器错误也会进入偏好数据。

---

## 6. 端侧部署

![](assets/MiniCPM-V%20-%20端侧视觉语言模型/figures/03-deployment-framework.png)

> **图 3：CPU/GPU/NPU 与部署框架。** CPU 最通用但理想延迟较高，GPU 框架覆盖广，Qualcomm QNN 提供 NPU 路径。来源：论文 Figure 5；官方 TeX 原图，已逐图复核。

### 6.1 权重量化

2.5 的 FP16 权重需约 16–17GB；GGML 的 Q4_K_M 4-bit 量化降到约 5GB。论文给出简单量化表达：

$$
w_i'=\operatorname{round}(w_i/s).
$$

但未系统报告量化前后 benchmark 精度、per-channel/group size、KV cache 量化与视觉 encoder 精度变化。

### 6.2 顺序加载与硬件加速

- **Memory optimization：** 不同时常驻 ViT 与 LLM；先载入 ViT 完成视觉编码，释放后再载入 LLM，图像处理 45.2s→31.5s。
- **Compilation optimization：** 在目标设备编译，encoding 50.5s→17.0s，decode 1.3→3.2 token/s。
- **Configuration optimization：** 自动搜索 CPU core 分配，decode 3.2→8.2 token/s。
- **NPU：** 视觉 encoder 换 QNN backend，Xiaomi 14 Pro 图像编码 3.7s→1.3s；LLM 仍走 llama.cpp。

![](assets/MiniCPM-V%20-%20端侧视觉语言模型/figures/04-deployment-ablation.png)

> **图 4：Xiaomi 14 Pro 端侧优化消融。** 左图把 image processing 与 LLM prefill 分开；优化后瓶颈从视觉编码转向 LLM prefill。右图显示配置优化后 8.2 token/s。来源：论文 Figure 6；官方 TeX 原图，已逐图复核。

---

## 7. 主要结果与消融

### 7.1 综合与 OCR

MiniCPM-Llama3-V 2.5（8.5B）报告：

- OpenCompass 65.1，高于论文同表 GPT-4V-1106 的 63.5、Gemini Pro 62.9。
- MathVista 54.3、LLaVA Bench 86.7、RealWorldQA 63.5。
- OCRBench 725、TextVQA 76.6、DocVQA 84.8。
- 视觉 token 范围 96–960，低于 LLaVA-NeXT-Llama3-8B 的 1728–2880。

跨闭源模型比较受版本、prompt 和评测时间影响，应理解为 2024 报告口径，不是长期不变的绝对排名。

### 7.2 RLAIF-V 消融

加入 RLAIF-V 后，OpenCompass 64.5→65.1，LLaVA Bench 85.4→86.7；Object HalBench 表中的“正确率式”response/mention 指标从 86.9/93.6→89.7/95.0。MME、MMBench-en、MMMU 有轻微下降，说明偏好对齐并非所有能力单调改善，但主要幻觉指标显著变好。

### 7.3 多语言消融

仅用 90K、低于 0.5% SFT 的多语言数据，French 46.4→72.7、German 22.8→76.5、Japanese 13.8→88.0。增益很大但各语言不均，论文推测受 base LLM 语言能力影响。

### 7.4 端侧证据强弱

优点是论文真的给了手机 SoC、量化内存、encoding latency 与 token/s；这比仅在 A100 上宣称“端侧友好”更扎实。缺点是图中 latency 包含的阶段口径不完全一致，缺少能耗、温度、持续运行降频、首 token 分位数与量化精度损失。

---

## 8. 与 MiniCPM-V 4.5 的关系

[[MiniCPM-V 4.5 - 高效端侧多模态模型配方]] 不是简单的 2.5 扩容，而是沿本论文的三个核心方向继续演化：

| 2024 MiniCPM-V / 2.5 | 2025 MiniCPM-V 4.5 | 演进含义 |
|---|---|---|
| 每个图像切片独立 2D Perceiver Resampler | 图像/视频共享 3D-Resampler，视频 package 内时空联合压缩 | 从空间压缩扩展到时空压缩 |
| OCR/文档依赖多来源数据与 parser 产物 | 动态视觉破坏统一 OCR、上下文恢复与文档知识 | 从数据集合并到统一学习目标 |
| RLAIF-V/DPO 主攻幻觉 | RLAIF-V 保留，并加入短/长思考 Hybrid GRPO | 从可信回答扩展到推理长度控制 |
| 真实手机 INT4/llama.cpp/QNN 证据 | 论文效率主表主要为 8×A100 | 4.5 架构更高效，但真实端侧证据反而应回看本论文 |

家族连续性在于“尽量少把无效视觉/语言 token 送进 LLM”；变化在于 4.5 把压缩对象从单张图的空间冗余推进到视频的时空冗余，并把输出端的思考 token 也纳入效率优化。

---

## 9. 局限与批判性思考

1. 自适应切片把每片独立压成固定 token，跨切片关系主要依赖 LLM 与 spatial schema；切片边界目标可能被割裂。
2. Resampler 只有一层且 query 数固定，压缩率高；细小文字、密集表格和极端长图仍可能丢信息。
3. 论文没有给出切片数/query 数的系统速度—精度曲线，也没有公开 compression head 数与完整 shape。
4. RLAIF-V 依赖 Llama-3 8B 拆 claim 和 34B MLLM 验证；低资源“端侧模型”的对齐过程本身并不端侧。
5. 端侧测试设备有限，缺少功耗、热稳定性、长会话 KV cache 与多轮图像内存。
6. “GPT-4V level”来自特定 11 项平均，部分单项仍落后；闭源模型版本快速变化，标题不应被理解为全面能力等价。
7. 数据规模大且包含网页抓取/模型重写，完整许可、过滤阈值与训练算力未充分披露。

---

## 10. 复现清单

1. 复现切片选择公式并输出每张图的 $(m,n)$、global slice、最终像素与 token 数。
2. 逐片验证 ViT 位置 embedding 的 2D 插值；记录切片边界目标召回。
3. 单层 Perceiver Resampler 分别测试 64/96 queries；测 OCR、首 token、峰值内存。
4. 严格按 Stage 1 压缩层、Stage 2 vision encoder、Stage 3 vision+压缩层、SFT 全量解冻执行，并保存冻结清单。
5. Data packing 必须重置 position ids、构造 block-diagonal mask，做跨样本信息泄漏测试。
6. RLAIF-V 保存 10 个采样回答、claim 切分、逐 claim yes/no 分数、偏好对与 verifier 版本。
7. 端侧报告 FP16/INT4 精度、模型大小、首 token P50/P95、prefill、decode、峰值内存、功耗与持续运行温度。
8. 与 4.5 对比时固定视觉输入与硬件，分别测 2D Resampler、3D-Resampler 及思考模式产生的输入/输出 token 成本。

---

## 11. 个人判断

这篇工作的核心价值是“端侧”有端到端证据链：视觉切片与 Resampler 控输入 token，冻结课程控制训练风险，RLAIF-V 控事实错误，量化与硬件路径控制部署成本。最扎实的工程结论是：视觉压缩后，手机瓶颈会转向 LLM prefill；只优化 ViT 不足以实现低延迟交互。其不足是核心架构细节和量化精度披露不够，且 2.5 的 8.5B 即使 4-bit 仍约 5GB，对多数移动设备并不轻。它应被视为 MiniCPM-V 4.5 的基线配方和端侧实证来源，而不是被后续版本完全替代。

---

## 12. 图表索引与材料来源

1. 图 1：总体架构、自适应切片与二维位置。
2. 图 2：RLAIF-V response—claim feedback—DPO 流程。
3. 图 3：CPU/GPU/NPU 部署框架对比。
4. 图 4：Xiaomi 14 Pro 端侧优化消融。

官方 PDF、arXiv source、代码与图片逐项来源见 [sources.md](assets/MiniCPM-V%20-%20端侧视觉语言模型/sources.md)。