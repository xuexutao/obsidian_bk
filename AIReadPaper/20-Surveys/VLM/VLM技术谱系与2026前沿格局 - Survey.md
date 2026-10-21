---
type: survey
status: complete
domain: VLM
title: VLM技术谱系与2026前沿格局
aliases:
  - 视觉语言模型架构演进百科
  - Vision-Language Model Architecture Survey 2026
scope: 2021-03至2026-10
source_policy: 官方论文、官方代码/项目页、官方model card/system card、本地独立精读
created: 2026-10-15
updated: 2026-10-16
tags:
  - VLM
  - MLLM
  - Architecture
  - Multimodal-Reasoning
  - Visual-Grounding
  - Agent
  - Survey
---

# VLM 技术谱系与 2026 前沿格局：从像素到行动的架构演进百科
## 基本信息
这篇综述以**从像素到输出的完整计算图**为主轴，而不是按排行榜逐模型点名。核心问题是：原始像素或视频帧如何变成 patch/tube token，连接器如何把它们压缩或投影为语言模型可消费的 soft token，图文信息在 Transformer 的哪一层、以什么 Q/K/V 拓扑交互，位置与 attention mask 如何约束可见关系，最后又如何通过语言头、专用视觉头或工具运行时变成文本、坐标、掩码、动作与环境反馈。架构主线以 Vault 中 **22 篇独立精读**为一手材料；VLM³、DINOv3、SAM 3 与 DA3 只作为能力或工具分支补充，不混入这 22 篇主线计数。
全文采用三类证据标签：

| 标签 | 含义 | 使用原则 |
|---|---|---|
| **论文声称** | 作者在论文、项目页或官方报告中的主张 | 可准确转述，但不自动视为已被独立复现 |
| **公开实测** | 论文主表、消融、官方 model/system card 或公开 benchmark 结果 | 同时说明输入预算、训练数据、工具、Thinking/Best-of-N 等条件 |
| **本文判断** | 基于多篇本地独立精读做出的综合判断 | 明确标记，不伪装成论文原结论 |

> [!important]
> [[DINOv3 - 大规模自监督视觉基础模型]]、[[SAM 3 - 概念驱动的开放词汇分割]] 与 [[DA3 - 任意视角恢复视觉空间]] 不是完整生成式 VLM。它们分别提供稠密视觉 token、概念到像素/跨帧 ID 的接口和度量几何恢复能力。本文讨论它们如何成为 VLM 的视觉底座、监督教师或工具，不把它们冒充可对话模型。

> [!warning]
> Gemini、GPT、Claude 等闭源系统没有公开可复现的方法论文。本文只列官方 model/system card 已披露的输入输出、上下文、产品能力、安全评测与少量架构标签；视觉 encoder、connector、fusion 层、位置编码和训练配方未披露处一律记为“未知”，禁止按相似产品反推内部结构。

---

## 1. 读图：从像素到输出的统一端到端计算图
### 1.1 五类 token 必须严格区分
原始像素/帧是输入张量，不是 token：图像 $I\in\mathbb{R}^{B\times C\times H\times W}$，视频 $V\in\mathbb{R}^{B\times T\times C\times H\times W}$。进入模型后才出现五类语义与计算地位不同的 token。

| 类型 | 记号 | 典型 shape | 是否离散词表项 | 由谁产生 | 是否通常被 LM loss 预测 |
|---|---|---|---|---|---|
| patch/tube token | $X_v$ | $B\times N_p\times d_v$ | 否 | CNN/ViT/时空 ViT 的 patch/tube embedding 与 blocks | 否 |
| connector soft token | $Z_v$ | $B\times N_c\times d_l$ | 否 | Linear/MLP、Q-Former、Resampler、merger、pooler | 否 |
| structure token | $S_v$ | $B\times N_s\times d_l$，或 $N_s$ 个 special-token ID | 可为离散 special token，也可为学习向量 | 预处理器、connector 或模型词表 | 依实现；常作为输入不计 loss |
| text token | $X_t$ | ID 为 $B\times N_t$，embedding 为 $B\times N_t\times d_l$ | 是 | tokenizer + embedding | prompt 常 mask，assistant target 计 loss |
| output/action token | $Y$ | 文本 ID 为 $B\times N_o$；连续 head 为 $B\times N_o\times d_a$ | 文本/工具 token 是；连续动作不是 | LM head、mask decoder、action head | 是，或由专用损失监督 |

**patch/tube token**仍处在视觉 encoder 通道维 $d_v$；**connector soft token**已投影到 LLM hidden size $d_l$，但没有词表 ID。**structure token**标出换行、换图、换视图和换时间片，例如 DeepSeek-VL2 的 `<tile_newline>`、Pixtral 的 `[IMAGE BREAK]`、图像起止符和显式文本时间戳。**output/action token**只描述可由词表生成的输出；mask、深度和连续控制若走专用 head，则是输出张量而非语言 token。
### 1.2 图像 patchify 的 shape 公式
对 batch 图像 $I\in\mathbb{R}^{B\times C\times H\times W}$，patch size 为 $p_h\times p_w$，padding 后网格高宽为：
$$H_p=\left\lceil\frac{H}{p_h}\right\rceil,\qquad W_p=\left\lceil\frac{W}{p_w}\right\rceil$$
未计 CLS/register/边界 token 时：
$$N_p=H_pW_p, \qquad X_v\in\mathbb{R}^{B\times N_p\times d_v}$$
这里的 $N_p$ 是 **ViT 内部 patch 长度**，不是 LLM 最终视觉长度。若后续用 $r_h\times r_w$ space-to-depth、pixel shuffle/unshuffle 或 merger，每组相邻 token 合成一个：
$$N_c\approx\left\lceil\frac{H_p}{r_h}\right\rceil\left\lceil\frac{W_p}{r_w}\right\rceil, \qquad Z_v\in\mathbb{R}^{B\times N_c\times d_l}$$
Qwen2-VL、Qwen2.5-VL、Qwen3-VL、InternVL3 和 Kimi-VL 常见 $2\times2$ 合并，理想情况下 $N_c\approx N_p/4$。它们名字可能分别写 merger、pixel shuffle、pixel unshuffle，但共同计算事实是：**长度缩为四分之一，局部通道信息先拼接，再由 MLP 映射到语言维度**。
### 1.3 视频 tube token 的 shape 公式
视频 $V\in\mathbb{R}^{B\times T\times C\times H\times W}$ 若使用时间 kernel/stride $p_t$ 的 3D patch embedding，则：
$$T_p=\left\lceil\frac{T}{p_t}\right\rceil, \qquad N_p^{video}=T_pH_pW_p$$
输出为：
$$X_v^{video}\in\mathbb{R}^{B\times T_pH_pW_p\times d_v}$$
Qwen2-VL/Qwen2.5-VL 和 GLM-V 以时间深度 2 的卷积把相邻两帧形成 tube；Kimi K2.5 的 MoonViT-3D 最多把四帧放入时空块并在 projector 前做轻量时间池化；MiniCPM-V 4.5 则先逐帧编码，再让固定 query 对一个相邻帧 package 做时空 cross-attention。三者都“压时间”，但压缩发生位置和 Q/K/V 拓扑不同，不能只因都写 3D 就视为同架构。
### 1.4 统一计算图
```text
RGB pixels / decoded frames
  │  resize / tile / native-grid / frame sampling / augmentation
  ▼
patchify or tube embedding
  │  [B, N_patch, d_v]
  ▼
vision encoder blocks
  │  self-attention: Q=K=V=visual hidden states
  │  position: absolute / 2D-RoPE / axial RoPE / temporal position
  ▼
visual feature grid(s)
  │
  ├─ direct Linear/MLP projection ───────────────────────────────┐
  ├─ 2×2 merger / pixel shuffle / local attention pooling ──────┤
  ├─ learned-query cross-attention / Q-Former / Resampler ──────┤
  ├─ multi-tower SVA / semantic router / temporal Resampler ────┤
  └─ multi-layer DeepStack residual injection ──────────────────┤
                                                               ▼
connector soft tokens [B, N_connector, d_l]
  + structure tokens / image boundaries / row breaks / timestamps
  + text embeddings [B, N_text, d_l]
                                                               ▼
fusion topology
  ├─ late similarity only
  ├─ encoder-decoder cross-attention
  ├─ LM-layer gated cross-attention
  └─ visual-prefix unified self-attention / token-specific experts
                                                               ▼
LLM prefill → KV cache
  │
  ├─ LM head → text / <loc> / tool / action tokens
  ├─ specialized head → mask / depth / continuous action
  └─ runtime → tool call → environment observation → next turn
                                                               ▼
causal decode / action loop / verifier / Best-of-N / multi-agent
```
### 1.5 一个最小 decoder-only VLM 的 shape 追踪
设 connector 输出 $Z_v\in\mathbb{R}^{B\times N_c\times d_l}$，文本 embedding 为 $E_t\in\mathbb{R}^{B\times N_t\times d_l}$，则输入序列可以写成：
$$H_0=[e_{bos};e_{vision\_start};Z_v;e_{vision\_end};E_t]\in\mathbb{R}^{B\times L\times d_l}$$
其中：
$$L=N_c+N_t+N_s$$
$N_s$ 是起止、行分隔、图像分隔、时间戳等 structure token 数。第 $l$ 个 decoder block 执行：
$$H_l=\operatorname{Block}_l(H_{l-1};M,P)$$
$M$ 是 attention mask，$P$ 是位置编码。LM head 只把要生成位置的 hidden state 映射到词表：
$$\operatorname{logits}_t=H_{L+t}^{(last)}W_{vocab}^{\top}$$
视觉 soft token 通常不对应词表 ID，也不要求“重建第几个 patch”；它们通过目标文本 loss 的反向传播被训练为有用条件。
### 1.6 计算成本分解
视觉 encoder 全局 self-attention 的主要项近似：
$$C_{vision-attn}=O(L_vN_p^2d_v)$$
若使用固定窗口大小 $w$：
$$C_{window-attn}=O(L_vN_pwd_v)$$
LLM prefill 对统一序列的主要 attention 项近似：
$$C_{prefill}=O(L_lL^2d_l)$$
自回归 decode 在 KV cache 下，每个新 token 不再重算全部历史 K/V，但仍需查询长度为 $L+t$ 的缓存：
$$C_{decode-step}=O(L_l(L+t)d_l)$$
因此“视觉 encoder 用窗口注意力”只降低视觉侧成本；若 connector 仍向 LLM 送入数千 token，LLM prefill 与 KV cache 成本依然存在。MoE 主要降低 FFN 每 token 激活计算，也不会自动减少 $N_c$、attention 或 KV cache。
### 1.7 代表图：双塔起点与生成式终点
![](../../10-Papers/Foundations/assets/CLIP%20-%20自然语言监督的可迁移视觉模型/figures/01-method-overview.png)
> **CLIP Figure 1。** 图像与文本分别编码，最终只做全局 embedding 点积。它把类别变成语言查询，但没有 decoder、teacher forcing 或自由文本生成。来源：[[CLIP - 自然语言监督的可迁移视觉模型]]，arXiv:2103.00020。

---

## 2. Vision encoder 演进：视觉 token 从哪里来
### 2.1 CNN 到 ViT：从特征图到显式序列
早期视觉语言系统常以 CNN feature map 为视觉记忆；Flamingo 仍使用冻结 NFNet-F6。CNN 的局部感受野、尺度层级和下采样天然适合密集视觉，但与 Transformer LLM 对接时需把二维 feature map 展平。ViT 把图像直接切成 patch 序列，使视觉 token 与文本 token 在数据结构上同构，也方便使用 cross-attention、统一 packing 与 RoPE。
CNN→ViT 不是“卷积被完全淘汰”。Qwen/GLM 等模型仍用 2D/3D convolution 做 patch embedding，ConvNeXt 在 Cambrian 的 OCR/Chart 任务中表现突出，视频 tube 也常由 Conv3D 形成。真正变化是：**主干的长程交互和输出接口被序列化**。
### 2.2 CLIP：图文对比监督带来开放语义
CLIP 的 image encoder 可是 ResNet 或 ViT，文本 encoder 是 causal Transformer。训练时 batch 内 $N$ 对图文形成 $N\times N$ 相似度矩阵，做双向对比交叉熵。它优先优化全局 embedding 可分性，而不是 patch 级局部关系。
**本文判断：**CLIP 的历史作用不只是给 VLM 一个视觉 backbone，而是把“类别”改写为可组合的文本提示，使后续 connector 可以复用已语言化的视觉空间。缺点是 OCR、小对象、计数、位置与几何并非其主要目标。
### 2.3 SigLIP/SigLIP-2：从 batch softmax 到 pairwise sigmoid
SigLIP 将 CLIP 的 batch 归一化对比目标改为逐图文对的 sigmoid loss，降低对超大 batch 全局 softmax 的依赖。PaliGemma、OneVision、DeepSeek-VL2、Kimi-VL 和 Qwen3-VL 都使用 SigLIP 系视觉初始化，说明它已成为生成式 VLM 的常用视觉语义底座。
SigLIP-2 进一步强化多分辨率与定位能力，但在具体 VLM 中仍需区分：Qwen3-VL 基于 SigLIP-2 checkpoint继续训练；Kimi 的 MoonViT 从 SigLIP-SO-400M 初始化后改造成 NaViT 式变长 encoder；PaliGemma 则保留固定 224/448/896 checkpoint。共享初始化不等于共享最终计算图。
### 2.4 EVA/EVA2：大规模 CLIP 式视觉塔
BLIP-2 使用 EVA-CLIP ViT-g/14，CogVLM 使用 EVA2-CLIP-E。它们提供更大视觉容量和较强图文语义，常取倒数第二层或移除偏向 CLS 聚合的最后一层，以保留局部 patch 信息。
**公开实测：**BLIP-2 的 EVA ViT-g 优于 CLIP ViT-L；CogVLM 的 4.4B EVA2-E 换为 300M EVA2-L 时，多数任务只小幅下降、TextVQA 下降更明显。结论不是“视觉塔越大必然越好”，而是 OCR/grounding 对局部 token 质量更敏感。
### 2.5 DINO/DINOv2/DINOv3：从语言监督互补到稠密特征保护
DINO 系列不依赖 caption，以自监督蒸馏学习对象结构和 patch 对应。Cambrian-1 的统一比较显示，DINOv2 在 vision-centric 与空间任务上突出，但 OCR/语言语义不如 SigLIP/CLIP；这解释了多塔方案为何把 DINO 与语言监督 encoder 并用。
DINOv3 进一步发现：超大模型长训练时，全局分类可继续提升，patch 局部一致性却在约 200K steps 后退化。Gram anchoring 约束 student 与早期 teacher 的 patch Gram matrix：
$$\mathcal{L}_{Gram}=\left\|X_SX_S^{\top}-X_GX_G^{\top}\right\|_F^2$$
它保护 patch 两两关系而非绝对特征坐标，允许全局语义继续学习。
![](../../10-Papers/Foundations/assets/DINOv3%20-%20大规模自监督视觉基础模型/figures/02-dense-feature-degradation.png)
> **DINOv3 Figure 6。** 训练越久，参考 patch 对无关位置的相似度扩散；这提醒 VLM 不能只看 caption/VQA 总分选择视觉塔。来源：[[DINOv3 - 大规模自监督视觉基础模型]]，arXiv:2508.10104。

![](../../10-Papers/Foundations/assets/DINOv3%20-%20大规模自监督视觉基础模型/figures/03-gram-anchoring-results.png)
> **DINOv3 Figure 8。** Gram anchoring 恢复 VOC/ADE20K 稠密性能，同时保持全局能力。对 grounding、区域描述、计数和几何 VLM，稠密 token 质量是独立轴。
### 2.6 InternViT：视觉侧扩大到 LLM 量级
原始 InternVL 的 InternViT-6B 是 width 3200、depth 48、patch size 14、约 5.9B 参数的 vanilla ViT。它不是后续聊天模型里的普通 300M 视觉塔，而是可独立用于分类、分割、检索和生成的视觉基础模型。
后续 InternVL2/3 将部署路线简化为 InternViT→pixel unshuffle→MLP→LLM：小模型常用 InternViT-300M，大模型用 6B 版本。被保留的是视觉 scaling 与动态高分辨率；被替代的是 8B QLLaMA 重型中间件。
### 2.7 MoonViT：NaViT 式原生分辨率与 variable-length packing
MoonViT 从 SigLIP-SO-400M 初始化，取消固定方形 resize 和预定义 tile 网格。不同尺寸图像 patchify 后展平，通过变长 FlashAttention packing 放入同一 batch；位置同时使用插值绝对 embedding 与 2D-RoPE。
它与 Pixtral 同属 native/dynamic-grid 路线，但并非同一实现：Pixtral从头训练 400M ViT、patch 16、显式 row break；MoonViT 继承 SigLIP 初始化并采用 NaViT packing，后接 $2\times2$ pixel shuffle。
### 2.8 AIMv2 与 GLM-V：原生分辨率视觉初始化
GLM-V 以 AIMv2-Huge 初始化视觉 encoder，把二维 patch convolution 替换为 3D convolution，图像复制一次走统一路径，视频时间下采样 2 倍。ViT 内同时使用插值绝对位置 embedding 与 2D-RoPE；LLM 侧另称 3D-RoPE，但论文未公开公式。
**证据边界：**可以确认 AIMv2-Huge、原生宽高比、2D-RoPE、3D convolution 和文本时间索引；不能补造 patch size、3D-RoPE 三轴分配、时间戳格式或完整冻结计划。
### 2.9 固定 resize → AnyRes tile → native/dynamic resolution
| 路线 | 预处理 | token 规律 | 代表模型 | 优点 | 代价 |
|---|---|---|---|---|---|
| 固定 resize | 统一到 224/336/448 方形 | 固定 $N_p$ | CLIP、BLIP-2、Qwen-VL | batch 稳定、成本可预测 | 破坏宽高比，小字和长页面丢失 |
| 多尺度专用 checkpoint | 224→448→896 继续训练 | 每档固定 | PaliGemma | 每档优化充分 | 需维护多套 checkpoint |
| AnyRes tile | 全局缩略图 + 局部块 | 随 tile 数增长 | LLaVA-NeXT/OneVision、InternVL、DeepSeek-VL2 | 复用固定 ViT，细节好 | 边界重复、布局规则、负载不均 |
| 原生动态网格 | 保持宽高比，变长 patch | 随像素面积增长 | Qwen2/2.5/3-VL、Pixtral、MoonViT、GLM-V | 少手工切图，结构自然 | ViT/LLM 成本随像素增长 |
| 语义自适应预算 | 几何切块后按内容选压缩档 | 随信息量变化 | InternVL3.5 ViR | 将 token 分给高信息区域 | router 误判、额外蒸馏成本 |
### 2.10 self-attention、window/global attention 与位置
视觉 encoder 内 self-attention 的 Q/K/V 都来自视觉 hidden state：
$$Q=XW^Q, \qquad K=XW^K, \qquad V=XW^V$$
全局 attention 允许任意 patch 直接通信；窗口 attention 只在局部网格内计算。Qwen2.5-VL 的 32 层视觉塔中 28 层为 112×112 对应窗口、4 层为全局层；SAM 3 的 PE vision encoder 也只在少数层用 global attention。局部层保文字和纹理，全局层周期性交换页面级信息。
2D-RoPE 位于 **视觉 encoder 的 Q/K 投影**；LLM 的 M-RoPE/3D-RoPE 位于 **语言 decoder 的 Q/K 投影**。相同“二维/三维”字样跨层级不可互换。
### 2.11 图像 encoder → 时空 encoder
视频视觉入口可分四类：
1. **逐帧独立编码**：Flamingo 的视频帧经同一 NFNet，再加时间 embedding；长期时序主要由 Resampler/LLM承担。
2. **早期 tube embedding**：Qwen2/2.5-VL、GLM-V 用时间深度 2 的 Conv3D形成 tube。
3. **共享图像视频时空块**：MoonViT-3D 把最多四帧 patch 共同 attention，再时间池化。
4. **encoder 后 package resampling**：MiniCPM-V 4.5 逐帧编码，query 对一组相邻帧联合读取。

**本文判断：**“支持视频”至少要说明抽帧、时间压缩、位置、跨帧 attention 和总 token 预算；把若干图片按顺序拼接只能证明多图接口，不足以证明速度、持续时间和身份建模。

---

## 3. Connector 与 token compressor 演进
### 3.1 连接器不是单一层，而是三种功能的组合
连接器可能同时承担：
- **维度对齐**：$d_v\rightarrow d_l$；
- **长度变换**：$N_p\rightarrow N_c$；
- **内容选择/融合**：决定哪些视觉信息进入 LLM。

Linear projector 只做第一项；$2\times2$ merger 做前两项；Q-Former、Perceiver、SVA、3D-Resampler 三项都做。比较连接器时必须写出输入输出 shape、Q/K/V 来源和长度变化。
### 3.2 Linear 与 MLP projector
输入 $X_v\in\mathbb{R}^{B\times N_p\times d_v}$，线性层：
$$Z_v=X_vW+b, \qquad W\in\mathbb{R}^{d_v\times d_l}$$
输出 $Z_v\in\mathbb{R}^{B\times N_p\times d_l}$，长度不变。两层 MLP 通常是：
$$Z_v=W_2\phi(W_1X_v+b_1)+b_2$$
LLaVA 原版使用 Linear；LLaVA-1.5/OneVision、Pixtral、Kimi、InternVL3、DeepSeek-VL2 常用两层 MLP。它们简单、吞吐友好、易联合训练，但不主动选择内容，所有 patch 都进入 LLM 或需配合其他压缩。
**公开实测：**PaliGemma 全参数训练时 Linear 77.2、MLP 77.1；只训 connector 时 Linear 70.7、MLP 69.7。这个结果否定“MLP 必然更强”，说明可塑性和训练目标常比 connector 深度重要。
### 3.3 Perceiver Resampler
Flamingo 的视觉输入 $X\in\mathbb{R}^{B\times N_p\times d}$，维护 $R=64$ 个 latent $Z\in\mathbb{R}^{B\times R\times d}$。每层：
$$Z^{k+1}=Z^k+\operatorname{Attn}(Q=Z^k,K=[X,Z^k],V=[X,Z^k])$$
Q 来自 learned latent；K/V 来自视觉 token 与 latent 拼接。输出长度固定为 64，与图像分辨率和视频帧数无关。
![](../../10-Papers/VLM/assets/Flamingo%20-%20少样本视觉语言上下文学习/figures/03-perceiver-resampler.png)
> **Flamingo Figure 5。** 固定 latent 迭代读取变长视觉网格；latent 同时进入 K/V，使其在取证时保留自身状态。

优点是 LLM cross-attention 成本固定、适合多图视频；缺点是高分辨率 OCR、密集对象和长视频都争夺同一 64-token 带宽。
### 3.4 Q-Former
BLIP-2 的冻结 ViT 输出 $X_v\in\mathbb{R}^{B\times M\times d_v}$，32 个 query 经 image transformer 的 cross-attention 读取视觉 K/V；query/text 共享 self-attention 参数。输出：
$$Z_q\in\mathbb{R}^{B\times32\times768}$$
cross-attention 中 Q 来自 query hidden，K/V 来自冻结图像 encoder；ITC/ITG/ITM 通过不同 mask 控制 query 与 text 是否互见。第二阶段再线性投影到 LLM 维度，成为 32 个 soft prompt。
![](../../10-Papers/VLM/assets/BLIP-2%20-%20冻结底座的高效视觉语言桥接/figures/02-stage1-qformer.png)
> **BLIP-2 Figure 2。** 三个目标共用 Q-Former 参数，但 attention mask 不同。Q-Former 不是“一个 cross-attention 层”的同义词，而是兼具视觉读取、文本编码/解码和多目标预训练的桥接 Transformer。

优点是冻结两端时可先学“什么视觉信息与语言相关”；缺点是固定 32 query 对复杂页面形成瓶颈，且第二阶段只把摘要作为 LLM 前缀，不在 LLM 深层重复读原视觉网格。
### 3.5 QLLaMA
原始 InternVL 的 QLLaMA 不是 BLIP-2 Q-Former 的简单改名。它由多语言 LLaMA-7B 初始化，总体约 8B，新增 96 queries 与约 1B cross-attention 参数；既做文本 encoder、对比对齐，又做图像条件生成，还可把视觉前缀交给外部聊天 LLM。
输入 InternViT 特征，Q 来自 96 queries，K/V 来自视觉特征；输出长度固定 96。优点是表示空间贴近 LLM、容量大；缺点是视觉 6B + 中间件 8B 极重。InternVL2/3 后来改用 pixel unshuffle + MLP，说明“同家族后代”可以彻底更换 connector。
### 3.6 $2\times2$ merger、pixel unshuffle 与 space-to-depth
设视觉网格 $X\in\mathbb{R}^{B\times H_p\times W_p\times d_v}$。对每个 $2\times2$ 邻域沿通道拼接：
$$X'\in\mathbb{R}^{B\times\lceil H_p/2\rceil\times\lceil W_p/2\rceil\times4d_v}$$
再经 MLP 映射为 $d_l$，长度约为原来的四分之一。Q/K/V 不适用：这是确定性重排+前馈投影，不是 attention。
Qwen2/2.5/3-VL、Kimi-VL、InternVL3、DeepSeek-VL2 都使用此类操作，但细节不同：
- Qwen 系称 vision-language merger，动态网格保持变长；
- InternVL 对每个 448 tile 的 1024 token 压到 256；
- DeepSeek-VL2 把每个 384 tile 的 $27\times27$ 处理成 $14\times14$，并加入行/视图结构 token；
- Kimi 在 MoonViT 后先 space-to-depth，再两层 MLP。

优点是便宜、局部结构规则、长度可预测；缺点是固定几何压缩不判断内容重要性，细小文字可能与邻域混合。
### 3.7 Local attention pooling
Molmo 拼接 ViT 倒数第 3 与倒数第 10 层特征，在每个 $2\times2$ 局部窗口做多头 attention pooling。Q 是窗口 patch 的均值，K/V 是该窗口四个 patch 的多层特征；每个窗口输出一个 token，长度缩为四分之一。
![](../../10-Papers/VLM/assets/Molmo%20-%20开放数据驱动的视觉语言模型/figures/02-model-architecture.png)
> **Molmo Figure 2。** 全图缩略图与重叠高分辨率 crop 共享视觉塔；局部 attention pooling 既压长度又允许内容加权。公开消融中它优于直接堆叠 patch 特征。

它比固定 concat 多了内容自适应权重，比全局 Resampler 更保留局部网格；缺点是每个窗口视野有限，全局关系仍交给 LLM。
### 3.8 SVA：多塔、空间局部 cross-attention
Cambrian SVA 为 $L\times L$ latent 网格建立 query。位置 $(i,j)$ 的 Q 来自该 latent；K/V 来自 CLIP、SigLIP、ConvNeXt、DINOv2 在对应空间块的特征。输出长度为 $L^2$，可从 576 强压到 36；多个 query group 可扩充通道容量。
$$q_{ij}^{*}=\operatorname{softmax}\left(\frac{q_{ij}[k_{ij,1},\ldots,k_{ij,N}]^{\top}}{\sqrt C}\right)[v_{ij,1},\ldots,v_{ij,N}]$$
![](../../10-Papers/VLM/assets/Cambrian-1%20-%20视觉中心的多模态大模型设计/figures/02-sva-architecture.png)
> **Cambrian Figure 8。** 每个 latent 只读多塔对应局部块，并在 LLM 多个深度重复聚合。它不是“把四塔特征 concat 后做 MLP”。

优点是表征互补、空间约束强、强压缩下 OCR 保持较好；缺点是四个视觉塔重复前向，显存和服务复杂度高。
### 3.9 DeepStack：长度不变的多层视觉注入
Qwen3-VL 从三个 ViT 层级取特征，各经专用 merger 投影，注入 LLM 前三个对应层：
$$h_l\leftarrow h_l+M_l(z_{v,l}), \qquad l\in\{1,2,3\}$$
Q/K/V 不在 connector 中产生；注入后下一层统一 self-attention 才计算 Q/K/V。主视觉前缀长度不变，不额外占上下文 token，但增加三个 merger 和逐层残差带宽。
![](../../10-Papers/VLM/assets/Qwen3-VL%20-%20长上下文视觉推理与智能体/figures/02-architecture.jpg)
> **Qwen3-VL Figure 1。** 最终视觉 token 仍作为前缀，DeepStack 另把多层视觉特征送入早期 LLM 层。它是“深层输入注入”，不是 Flamingo 那种每层以语言 Q 查询独立视觉 K/V。

**公开实测：**受控 15B-A2B、200B token 实验中平均 74.7→76.0；TextVQA 80.6→80.5 略降。它有直接证据，但不是所有任务单调提升。
### 3.10 ViR：语义路由式视觉压缩
InternVL3.5 每个高分辨率 tile 原始 1024 token。高信息 tile 走 $1/4$ pixel shuffle 输出 256，低信息 tile 走 $1/16$ 输出 64。router 的监督来自高/低压缩下 teacher-student loss 比：
$$r_i=\frac{\mathcal{L}(y_i\mid I_{1/16})}{\mathcal{L}(y_i\mid I_{1/4})}$$
若比值低，说明强压缩几乎不影响输出，路由到 64；否则保留 256。Q/K/V 不由 ViR直接定义，它选择不同几何压缩路径。
优点是视觉 token 平均减少约 50%；缺点是二档路由粗糙、标签是 KL proxy，不直接优化任务成功率或真实延迟。ViCO 还需要冻结 reference model 和约 30B token consistency training，压缩不是“免费模块”。
### 3.11 3D-Resampler
MiniCPM-V 4.5 的 learned queries 带 2D 空间和时间位置。图像时，Q 来自固定 query，K/V 来自单图或切片视觉特征；视频时，Q 对一个相邻帧 package 的全部视觉 token 做 cross-attention，输出固定长度。
若 package 有 $F$ 帧、每帧 $N$ token，输入 K/V 长度为 $FN$，输出长度为固定 $R$：
$$Z_{pkg}=\operatorname{Attn}(Q\in\mathbb{R}^{R\times d},K,V\in\mathbb{R}^{FN\times d})$$
![](../../10-Papers/VLM/assets/MiniCPM-V%204.5%20-%20高效端侧多模态模型配方/figures/01-architecture.png)
> **MiniCPM-V 4.5 Figure 1。** 图像和视频共享视觉 encoder/Resampler，视频在 package 内联合压缩。论文未披露默认 query 数、层数和位置公式，不应补造。

**公开实测：**同 checkpoint 继续 300 steps，2D→3D Resampler 后 token/frame 64.0→21.3，Video-MME 两个口径均提升。这是其最强组件消融。
### 3.12 连接器统一对比矩阵
| 连接器 | 输入 shape | Q 来源 | K/V 来源 | 输出 shape | 长度变化 | 典型优势 | 主要风险 |
|---|---|---|---|---|---|---|---|
| Linear | $B,N,d_v$ | 无 | 无 | $B,N,d_l$ | 不变 | 最简单、部署友好 | 无内容选择 |
| 2-layer MLP | $B,N,d_v$ | 无 | 无 | $B,N,d_l$ | 不变 | 非线性维度对齐 | 长度成本保留 |
| $2\times2$ merger | $B,H,W,d_v$ | 无 | 无 | $B,HW/4,d_l$ | $1/4$ | 便宜、局部结构清晰 | 固定几何压缩 |
| Local attention pooling | 局部 $4\times d$ | patch mean | 局部 patch | 每窗 1 token | $1/4$ | 内容加权、保局部 | 全局关系后置 |
| Perceiver Resampler | $B,N,d$ | learned latent | visual+latent | $B,R,d$ | 固定 $R$ | 任意输入固定成本 | 密集细节瓶颈 |
| Q-Former | $B,N,d_v$ | 32 learned query | frozen ViT | $B,32,d_q$ | 固定 32 | 多目标预对齐 | 多图/高分辨率受限 |
| QLLaMA | $B,N,d_v$ | 96 query | InternViT | $B,96,d$ | 固定 96 | 大容量语言化桥接 | 极重中间件 |
| SVA | 多塔局部网格 | 2D latent | 多塔对应块 | $B,L^2,d$ | 可强压缩 | 多表征、空间保持 | 多塔成本 |
| DeepStack | 多层 ViT token | 无独立 Q | 无独立 K/V | 对齐主序列 | 长度不变 | 多层语义直达 LLM | 增加融合耦合 |
| ViR | tile token | 路由分类 | 压缩损失特征 | 每 tile 64/256 | 自适应 | 平均省 token | router 误差 |
| 3D-Resampler | $B,FN,d$ | 时空 query | package patch | $B,R,d$ | 固定 $R$ | 利用帧冗余 | 短事件可能丢失 |

---

## 4. Fusion 与 attention 拓扑：图文究竟在哪里相遇
### 4.1 双塔 late fusion
CLIP 的视觉与文本 encoder 完全独立，只有最终归一化 embedding 点积：
$$s(I,T)=\frac{f_I(I)^{\top}f_T(T)}{\tau}$$
Q/K/V 不跨模态，融合只有一次。优点是图像和文本可独立缓存、检索高效；缺点是 token 级关系、生成和多步组合推理弱。
### 4.2 Encoder-decoder cross-attention
经典图像 caption 或 T5 式 VLM 中，encoder 产生视觉/图文 memory $M$，decoder token hidden $H_t$ 作为 Q：
$$Q=H_tW^Q, \qquad K=MW^K, \qquad V=MW^V$$
cross-attention 输出写回 decoder residual stream：
$$H_t' = H_t+\operatorname{Attn}(Q,K,V)$$
每个 decoder layer 都可有 cross-attention，因此“encoder-decoder”通常是逐层融合；但具体层数由实现决定。BLIP-2 接 FlanT5 时，视觉 soft prompt 与文本 prefix 先进入 encoder，decoder再生成 suffix，和 decoder-only OPT 路线不同。
### 4.3 Flamingo：LM-layer gated cross-attention
Flamingo 的 Resampler先把每个视觉输入压为 64 token。冻结 LM 的若干 block 前插 gated cross-attention：Q 来自当前语言 hidden，K/V 来自对应图像 latent；输出乘零初始化 $\tanh(\alpha)$ 后写回语言 residual：
$$h'=h+\tanh(\alpha)\operatorname{XAttn}(Q=h,K=z_v,V=z_v)$$
![](../../10-Papers/VLM/assets/Flamingo%20-%20少样本视觉语言上下文学习/figures/02-architecture.png)
> **Flamingo Figure 3。** 跨模态融合沿 LM 深度重复发生，不是输入端一次拼接。冻结 LLM 不等于浅融合：Flamingo正是“冻结参数 + 深层跨注意力”的反例。

视觉 K/V 可以在多层被读取；语言 token 的 self-attention 又把视觉影响向后传播。每层插最好，每四层插一次可加速约 66%且 Overall 仅降 1.9%，显示融合频率是明确设计轴。
### 4.4 BLIP-2：pre-LLM Q-Former
BLIP-2 的深层 cross-attention 全发生在 Q-Former 内部，进入 LLM 后只剩 32 个视觉软提示。对 decoder-only OPT，视觉 soft token 与文本用统一 causal self-attention；对 FlanT5，视觉与 prefix 进入 encoder。
这是一种“**LLM 前完成主动视觉读取**”的拓扑。融合在 Q-Former 多层发生，但 LLM 不再访问原始 ViT K/V。它比 Flamingo 参数少、易换 LLM，却牺牲 LM 层内反复取证。
![](../../10-Papers/VLM/assets/BLIP-2%20-%20冻结底座的高效视觉语言桥接/figures/03-stage2-llm.png)
> **BLIP-2 Figure 3。** 同一 Q-Former 输出可接 decoder-only 或 encoder-decoder LLM；“Q-Former”描述的是 LLM 之前的视觉语言接口，不决定后端一定是哪种 decoder。
### 4.5 visual-prefix unified self-attention
LLaVA、PaliGemma、Pixtral、Qwen2/2.5/3-VL、InternVL3、Kimi-VL、DeepSeek-VL2等把视觉 soft token与文本 embedding 拼成一条序列。每个 decoder block 的 Q/K/V 都由统一序列 hidden 投影：
$$Q=[Z_v;E_t]W^Q, \qquad K=[Z_v;E_t]W^K, \qquad V=[Z_v;E_t]W^V$$
在 causal mask 下，前置视觉 token不能看后续文本，文本 token可看全部视觉前缀与过去文本；在 Prefix-LM mask 下，视觉+prefix 内部可双向互看。
这类结构常被叫“浅 projector”，但跨模态交互其实会在每个 LLM self-attention layer 发生。浅的是**入口变换**，不是后续 attention 深度。
### 4.6 LLaVA：极简入口、全层统一融合
![](../../10-Papers/VLM/assets/LLaVA%20-%20视觉指令微调/figures/02-architecture.png)
> **LLaVA Figure 1。** CLIP patch 只经 Projection $W$ 接入 Vicuna；Stage 2 更新 projector+LLM，使统一 self-attention 学会消费视觉 token。没有 Q-Former不等于没有深层图文交互。

原版不压视觉长度；它把连接器复杂度转移给指令数据和 LLM微调。其历史意义是证明“简单计算图 + 合适监督格式”足以形成开放视觉助手。
### 4.7 PaliGemma：Prefix-LM 统一融合
PaliGemma 的 image token、BOS、任务 prefix 构成双向 prefix block；suffix 采用 causal attention。视觉与问题在生成前可双向融合，loss 只落 suffix。
![](../../10-Papers/VLM/assets/PaliGemma%20-%20可迁移视觉语言基础模型/figures/02-prefix-lm-mask.png)
> **PaliGemma Figure 2。** attention mask 和 loss mask 是两张不同的图：prefix 可见关系不等于 prefix 被预测。
### 4.8 CogVLM：token-type-specific QKV/FFN
CogVLM 视觉和文本 token处于同一个 causal attention 图，但投影参数按 token 类型分流：
$$Q=\operatorname{concat}(X_IW_I^Q,X_TW_T^Q), \quad K=\operatorname{concat}(X_IW_I^K,X_TW_T^K), \quad V=\operatorname{concat}(X_IW_I^V,X_TW_T^V)$$
文本 $W_T$/FFN 冻结，视觉 $W_I$/FFN 可训练；attention 输出仍使两类 token互相作用。
![](../../10-Papers/VLM/assets/CogVLM%20-%20视觉专家驱动的深层融合/figures/03-architecture.png)
> **CogVLM Figure 4。** 它不是双塔，也不是普通 MoE：路由条件是视觉/文本 token 类型，专家存在于每个 LLM layer 的 QKV 与 FFN。

冻结语言路径保护纯文本行为，但约 6.5B 可训练视觉专家使“冻结 LLM”并不低成本。
### 4.9 Cambrian：多塔局部 attention + LLM 多层再聚合
Cambrian 先用 SVA 将多视觉塔局部块压成 latent 网格，再以 stride 3 在 LLM深度重复聚合原视觉特征。Q 来自当前 latent/语言侧状态，K/V 来自未压缩多塔对应区域。它同时有 connector 内局部 fusion 和 LLM深度方向的 multi-layer aggregation。
相较 Flamingo，Cambrian的 K/V 是多塔空间局部特征；相较 DeepStack，它使用显式 cross-attention重读视觉 memory；相较普通 visual prefix，它不只消费一次压缩结果。
### 4.10 Qwen3-VL DeepStack：逐层残差注入，不是 cross-attention
DeepStack 把多层 ViT特征经 merger 后直接加到前三个 LLM层视觉位置。这里没有语言 Q 去检索独立视觉 K/V；写回发生在对应 hidden state 上，然后统一 self-attention继续传播。
因此：
- Flamingo：语言 Q，视觉 K/V，逐层 cross-attention；
- Cambrian：空间 latent/语言态 Q，多塔局部 K/V，多层聚合；
- DeepStack：视觉特征先投影，直接 residual add，再统一 self-attention。

三者都可叫“深层融合”，但计算图不同。
### 4.11 冻结 LLM 不等于浅融合
| 模型 | LLM 是否冻结 | 融合深度 | 结论 |
|---|---|---|---|
| BLIP-2 | 冻结 | Q-Former 深、LLM入口一次 | 冻结 + pre-LLM 深桥接 |
| Flamingo | 冻结 | 多个 LM layer cross-attention | 冻结 + LM 内深融合 |
| CogVLM | 语言参数冻结 | 每层 token-specific QKV/FFN | 冻结语言路径 + 深层视觉专家 |
| LLaVA | Stage 2 更新 LLM | 每层统一 self-attention | 简单入口 + 全模型适配 |
| PaliGemma | 全参数联合 | Prefix-LM 全层融合 | 简单 Linear 也可深融合 |

---

## 5. Attention mask 与位置编码：可见关系和坐标系
### 5.1 Attention mask、loss mask 与 padding mask
三者不可混淆：
- **attention mask**：决定 token $i$ 是否能读取 token $j$；
- **loss mask**：决定位置 $t$ 的 next-token error 是否进入目标；
- **padding/packing mask**：决定不同样本或不同图像 block 是否隔离。

模型可以让 prompt token 被 assistant 读取，但不在 prompt 上计算 loss；也可以让图像内部双向 attention，却从不预测视觉 token。
### 5.2 Causal mask
标准 decoder-only mask：
$$M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases}$$
若序列为 `[vision][user][assistant]`，视觉 token彼此只能按展平顺序看过去，用户/assistant可看全部前置视觉。多数 LLaVA-style模型在 LLM侧采用这种拓扑。
### 5.3 Prefix-LM mask
设 prefix 结束位置为 $s$。prefix 内双向，suffix 对 prefix全可见、对 suffix 仅看过去：
$$M_{ij}=0\ \text{if}\ (i<s\land j<s)\ \text{or}\ (i\ge s\land j\le i),\ \text{otherwise}\ -\infty$$
PaliGemma 的图像+任务 prefix使用此 mask。公开消融显示，prefix改 causal会下降，图像也 causal更差；对 prefix计算 NTP loss也降低迁移表现。
### 5.4 Encoder-decoder mask
encoder通常对视觉/文本输入做双向 self-attention；decoder内部 causal；decoder cross-attention可读取全部 encoder memory。BLIP-2+FlanT5 属于此类，不能用“视觉前缀是否 causal”描述其全部拓扑。
### 5.5 Block-diagonal packing
Pixtral 将多张变长图像打包成一个视觉序列，用 block-diagonal mask阻止不同图像在 **视觉 encoder 内**互相 attention：
```text
image A patches:  [A A A | 0 0]
image B patches:  [0 0 0 | B B]
```
进入统一 LLM 后，多图是否互见由对话序列和 decoder mask决定。视觉 encoder packing 隔离图像，不等于 LLM无法跨图比较。
![](../../10-Papers/VLM/assets/Pixtral%2012B%20-%20原生分辨率视觉语言模型/figures/02-vision-encoder.png)
> **Pixtral Figure 2。** 2D-RoPE保行列，block mask隔离打包图像，row/image separator显式保存结构。
### 5.6 BLIP-2 的 ITC/ITG/ITM mask
| 目标 | query↔text 可见关系 | 目的 |
|---|---|---|
| ITC | query与text互不可见，各自编码 | 学全局图文对齐，避免信息泄漏 |
| ITG | query互见且不看text；text看全部query与过去text | 强迫32 query承载可生成视觉证据 |
| ITM | query与text双向交互 | 学细粒度匹配与 hard negative |

同一 Q-Former通过 mask切换任务，不是三套网络。
### 5.7 Flamingo per-image mask
多图交错序列中，每个文本 token只直接 cross-attend 最近前置图像；更早图像的信息通过此前文本 hidden 和语言 self-attention间接传播。此设计避免依赖固定图像编号，并允许训练最多 5 图、测试扩到更多示例。
**公开实测：**直接只看最近图比跨注意所有历史图高 7.2%。这说明“可见更多”不一定更好；合理结构偏置可减少图文错配。
### 5.8 多图与视频在 decoder 中的可见关系
典型 causal 序列：
```text
<image_1> V1 </image_1> question_1 answer_1
<image_2> V2 </image_2> question_2 answer_2
```
第二段文本可通过 causal attention看 V1、answer1、V2；V2 token本身不能看未来 question2。若所有图像集中前置：
```text
V1 separator V2 separator ... user question assistant target
```
question与target可看所有图像，图像间能否互见取决于 prefix/causal mask。视频常是 frame/tube token按时间展平，后帧 token在 causal视觉前缀下可看前帧而不能看后帧；若 ViT内部使用双向时空 attention，则 tube层已完成非因果帧融合。
### 5.9 1D absolute position 与 1D RoPE
固定绝对位置 embedding直接相加，需插值才能适应新长度/网格；RoPE在 Q/K通道旋转，使 attention点积编码相对位移，长外推通常更自然。LLM把所有视觉 token按一维连续编号会产生两个问题：二维邻近被展平顺序扭曲；数千视觉 token把后续文本推到很大 position ID。
CogVLM让所有视觉 token共享同一个 LLM position id，以避免图像上部离文本太远；空间主要交给 ViT。InternVL3的 V2PE则让视觉 token以 $\delta<1$ 缓慢推进位置窗口。
### 5.10 2D-RoPE
在视觉 encoder内，将 Q/K通道分给高度和宽度：
$$R_{2D}(h,w)=R_h(h)\oplus R_w(w)$$
Qwen2/2.5/3-VL、Pixtral、MoonViT、GLM-V都使用 2D-RoPE或相近轴向旋转。它作用于 patch self-attention，不直接告诉 LLM“这是第几张图”。多模型同时保留插值绝对 embedding，以继承旧 checkpoint和补绝对坐标。
### 5.11 M-RoPE
Qwen2-VL 在 LLM侧把 RoPE通道拆为 temporal/height/width：
$$R_i=R_t(t_i)\oplus R_h(h_i)\oplus R_w(w_i)$$
文本三轴 position相同，退化为1D；图像 temporal固定、h/w按网格；视频 t随帧增加。它位于 decoder，不是 ViT的2D-RoPE。
![](../../10-Papers/VLM/assets/Qwen2-VL%20-%20任意分辨率世界感知/figures/02-mrope.png)
> **Qwen2-VL Figure 3。** 文本、图像、视频共享 decoder，但使用不同三轴坐标规则。
### 5.12 Absolute-time M-RoPE
Qwen2.5-VL 让 temporal ID与真实秒数对齐，而不是每抽一帧固定加1。若时间为 $\tau_i$：
$$t_i\approx\operatorname{round}(s\tau_i)$$
论文未披露 $s$。它必须与动态 FPS配合：同一真实时间在不同采样率下获得一致位置语义。
### 5.13 Interleaved-MRoPE
Qwen3-VL不再让 t/h/w各占连续频率块，而按 t,h,w,t,h,w…交错分配 RoPE频率。若频率为 $\omega_j$、轴选择为 $a(j)$：
$$\theta_{i,j}=p_{i,a(j)}\omega_j$$
这样每个轴都获得低频长程与高频局部信号，减少连续分块导致的 spectral bias。论文未给单独消融，不能把全部视频提升归因于它。
### 5.14 3D-RoPE
GLM-V公开写明 LLM侧使用3D-RoPE，但未给三轴定义、频率分配、文本坐标或公式。因此只能登记为“decoder层三维旋转位置编码，细节未知”，不能自动等同 Qwen M-RoPE。
### 5.15 V2PE
InternVL3对文本位置加1，对同一图像视觉 token加 $\delta<1$：
$$p_i=p_{i-1}+\begin{cases}1,&\text{text}\\\delta,&\text{visual}\end{cases}$$
训练从 $1,1/2,\ldots,1/256$ 采样。公开消融中 $\delta=1/4$总体最好，过小 $1/256$反降，说明压缩位置窗口和保持视觉可分辨性之间有折中。
### 5.16 显式文本时间戳
Qwen3-VL在每组视频 patch前插入 `<3.0 seconds>` 等文本；GLM-V在每帧后插 time index token。时间语义由 tokenizer+LLM读取，而非只存在于 RoPE position id。
文本时间戳优点是可解释、可直接生成/定位、避免巨大稀疏时间 ID；缺点是占上下文、格式依赖强。它与 temporal RoPE可以并存，不是二选一的理论关系。
### 5.17 位置编码作用层级总表
| 方法 | 作用模块 | 编码对象 | 是否改变 token 数 | 代表 |
|---|---|---|---|---|
| 绝对二维 embedding | Vision encoder输入 | patch绝对网格 | 否 | CLIP/SigLIP、MoonViT、GLM-V |
| 2D-RoPE | Vision self-attention Q/K | 高/宽相对位置 | 否 | Qwen2+、Pixtral、MoonViT |
| 1D RoPE | LLM self-attention Q/K | 序列顺序 | 否 | LLaVA/Pixtral等 |
| M-RoPE | LLM self-attention Q/K | 时间/高/宽 | 否 | Qwen2-VL |
| absolute-time M-RoPE | LLM self-attention Q/K | 真实秒+空间 | 否 | Qwen2.5-VL |
| Interleaved-MRoPE | LLM self-attention Q/K | 三轴均衡频率 | 否 | Qwen3-VL |
| 3D-RoPE | LLM self-attention Q/K | 官方称3D，细节未知 | 否 | GLM-V |
| V2PE | LLM position progression | 视觉慢速占窗口 | 否 | InternVL3 |
| 文本时间戳 | 输入词表序列 | 可读真实时间 | 是 | Qwen3-VL、GLM-V |

---

## 6. LLM decoder 与生成：从相似度到 token、head 与行动
### 6.1 CLIP 相似度不是生成
CLIP输出图像/文本 embedding和相似度，可做零样本分类、检索、过滤与 reward；它没有自回归 decoder、LM output head或 teacher forcing。把候选答案写成文本再选相似度最高者仍是判别式选择，不是自由生成。
### 6.2 Encoder-decoder 生成
视觉/图文 encoder先编码全部输入，文本 decoder按 causal顺序生成。每层 decoder可通过 cross-attention读取完整 encoder memory。优点是输入理解与输出生成分工明确；缺点是模型结构双栈、KV/cache和部署路径较复杂。
BLIP-2接 FlanT5时，投影后的 query与文本 prefix进 encoder，decoder预测 suffix；这与OPT版把query前置到单一decoder不同。
### 6.3 Decoder-only NTP
现代开源 VLM主流把视觉 soft token当输入 embedding，语言 decoder执行 next-token prediction：
$$\mathcal{L}_{NTP}=-\sum_{t\in\mathcal{T}_{target}}\log p_\theta(y_t\mid y_{<t},Z_v,X_{prompt})$$
同一LM head可生成自然语言、Markdown、HTML、LaTeX、坐标字符串、工具 JSON与动作 token。统一接口降低任务头数量，但结构合法性、数值精度和置信校准变成新问题。
### 6.4 Prefix-LM
Prefix-LM不是独立模型类型，而是 decoder-only模型的一种 mask/目标配置。图像+任务 prefix双向，suffix causal；loss只算 suffix。PaliGemma的消融显示，它比全 causal输入或重建prefix更适合迁移。
### 6.5 Teacher forcing
训练时第 $t$ 个目标位置读取真实历史 $y_{<t}^{*}$；推理时读取模型自己生成的 $\hat y_{<t}$。这形成 exposure bias：训练 loss低不保证长CoT、坐标列表或Agent轨迹稳定。InternVL3的MPO、Kimi/GLM的在线RL、Best-of-N/PRM都在不同层面缓解自由生成分布问题，但不是同一机制。
### 6.6 Attention mask 与 loss mask
以对话为例：
```text
system + user + visual soft tokens + assistant answer
```
所有前缀可被answer读取；loss通常只落 assistant answer和必要 special token。LLaVA、DeepSeek-VL2、Kimi-VL都明确 mask system/user；视觉 soft token没有词表target，不被直接预测。
“只对文本 token计算 loss”不等于视觉 encoder无梯度：目标文本 loss通过attention与connector反向传播到视觉塔，只是没有像素重建或视觉离散 token预测项。
### 6.7 LM output head
最后 hidden $h_t\in\mathbb{R}^{d_l}$ 经词表矩阵：
$$p(y_t\mid\cdot)=\operatorname{softmax}(W_{vocab}h_t)$$
若坐标被离散为 `<loc0000>` 或普通数字字符串，也由同一 head生成。优点是任务统一；缺点是检测置信度、NMS、mask topology、连续控制等结构并非语言head的强项。
### 6.8 文本化 bbox 与 point
Qwen-VL用 `<ref>/<box>` 和 $[0,1000)$ 坐标；Qwen2-VL沿用归一化坐标；Qwen2.5-VL改用实际输入像素坐标；Qwen3-VL又回到 $[0,1000]$。CogVLM、DeepSeek-VL2也使用离散归一化框；Molmo用 XML风格 $[0,100]$ point。
文本化优点是无需检测头，可与对话、CoT、工具调用共享；风险包括格式错误、越界、对象—框错配、缺置信度和跨版本坐标协议变化。
### 6.9 PaliGemma `<loc>/<seg>`
PaliGemma提供1024个 `<loc>` token表示量化坐标，128个 `<seg>` token表示VQ-VAE mask code。检测/分割被序列化成 suffix，由LM head生成。
这比普通数字字符串更受约束，但mask是离散码序列，不等同 SAM类逐像素可校准 mask decoder。适合统一迁移，不必然适合穷尽实例与视频ID。
### 6.10 Mask decoder 与专用视觉 head
SAM 3使用DETR object queries、presence head、MaskFormer风格实例/语义 mask head和SAM2 memory tracker。MLLM可把复杂意图拆成 noun phrase，SAM3负责像素落地。
![](../../10-Papers/Foundations/assets/SAM%203%20-%20概念驱动的开放词汇分割/figures/02-detector-tracker-architecture.png)
> **SAM 3 Figure 4。** detector负责召回，tracker负责身份，presence head负责“概念是否存在”。它展示专用head与通用VLM的合理分工。
### 6.11 Tool/action token
GUI/Agent把动作序列化为 function name与参数，例如 tap(x,y)、type(text)、scroll(direction)、tool_call JSON。LM head生成离散动作；环境执行后把新截图/工具返回追加到上下文。
这不是“模型一次前向直接控制环境”，而是：
```text
observation → visual prefill → thought/action tokens → executor
→ new observation → next prefill/decode
```
单步ScreenSpot grounding高不等于OSWorld长程成功率高，因为后者还需要状态记忆、规划、错误恢复、权限和安全。
### 6.12 连续动作 head
机器人/VLA可能在LLM hidden上增加连续动作 head，输出关节、位姿、速度或轨迹分布，并以MSE、行为克隆或扩散/flow loss训练。此时“action token”未必是词表项。本文核心模型多数仍以文本/工具动作接口为主，不能把GUI坐标生成等同连续控制。
### 6.13 KV cache、prefill 与 decode
prefill对全部视觉+文本前缀并行计算K/V并缓存；decode每步只为新token计算Q/K/V，将K/V追加缓存。高分辨率、多图和视频主要抬高：
- vision encoder前向；
- LLM prefill二次attention；
- 每层KV cache长度；
- 第一个输出token延迟TTFT。

输出很长的Thinking/Agent轨迹则主要抬高逐步decode和缓存增长。视觉压缩与输出长度控制分别优化输入和输出两端，MiniCPM-V 4.5将3D-Resampler与Hybrid RL放在同一效率框架中正是因此。
### 6.14 多图/高分辨率成本公式
若第 $i$ 张图connector后有 $N_i$ token，文本长度 $N_t$：
$$L=N_t+\sum_{i=1}^{m}N_i+N_s$$
单纯“支持更多图”会让prefill近似按 $L^2$增长。若每图都使用高分辨率tile，多图数量与单图细节直接竞争同一上下文窗口。OneVision因此单图每crop高token，多图每图只用base，视频每帧压到196。
### 6.15 CoT、RL、Best-of-N 与 Agent runtime 不能混同
| 概念 | 参数是否更新 | 额外计算发生在哪里 | 典型例子 |
|---|---|---|---|
| CoT/Thinking decode | 否 | 单条输出更长 | Kimi-VL Thinking、Qwen3-VL Thinking |
| RL | 是 | 训练时rollout+reward+更新 | GLM RLCS、Kimi policy mirror descent |
| Best-of-N/PRM | 否（policy推理时） | 并行/重复生成+critic评分 | InternVL3 VisualPRM-Bo8 |
| Tool runtime | 模型可冻结 | 外部搜索、OCR、代码、图像工具 | Qwen3-VL thinking with images |
| Multi-Agent | 子模型常冻结 | 多上下文、多工具、并行资源 | Kimi K2.5 Agent Swarm |

同一benchmark上，single-pass、16K CoT、Bo8+PRM、工具搜索与4-Agent swarm不是同成本设置，必须分列。

---

## 7. 数据如何被编码、融合与监督
### 7.1 Caption 数据
最简单样本：
```text
<vision_start> visual soft tokens <vision_end>
Describe the image.
<assistant> a long caption ... <eos>
```
输入是图像+短prompt，target是caption；loss只算assistant。caption建立对象、属性、场景和世界知识，但若描述不要求局部证据，模型可能只学全局语义。
CLIP/SigLIP式对比数据不需要target caption生成；同一图文对在对比目标中只要求embedding匹配。InternVL强调：噪声网页数据适合容错对比学习，生成目标需要更高质量caption。
### 7.2 Interleaved 图文
网页/教材/教程可序列化为：
```text
text_0 <image_1> text_1 <image_2> text_2 ...
```
必须保留DOM/页序、图注对应与边界 token。Flamingo的M3W每段随机采256 text token、最多5图；去掉交错网页数据使少样本能力大降。Qwen3/Kimi/GLM又把交错扩展到整书、数百页和长视频。
多图能力不是只由架构产生：BLIP-2只有单图配对，加入in-context示例也未观察到明显提升；Flamingo有交错训练，才形成视觉ICL。
### 7.3 OCR、文档与图表
目标可以是纯转写、结构化HTML/Markdown、表格、公式或QA。Qwen2.5-VL把段落、表格、图表、公式、乐谱、化学式统一为带`data-bbox`的HTML；Qwen3-VL支持QwenVL-HTML/Markdown；DeepSeek-VL2可生成HTML或Python绘图代码。
结构边界很重要：
- row/newline token保存二维阅读顺序；
- image/page separator防止跨页串行混淆；
- bbox把文本内容与页面区域绑定；
- 表格`<tr>/<td>`或Markdown pipe保存单元格关系。

MiniCPM-V 4.5用低/中/高视觉破坏统一OCR、上下文推断与文档知识恢复。高破坏样本允许“猜测被遮文字”，因此评测必须区分像素可见OCR与知识补全，避免把幻觉当识别。
### 7.4 Grounding、point 与 mask
Grounding样本常有四种方向：
```text
region → text        给框，描述区域
text → bbox          给指称，输出框
image → grounded caption  描述中插框
question → point/mask      输出点击或mask code/tool call
```
Molmo Points用人工点击；计数时先生成所有点再给数量，公开实测优于只输出数字。PaliGemma用`<loc>/<seg>`；Qwen/Cog/DeepSeek用文本坐标；SAM3则用专用mask head与presence calibration。
### 7.5 视频与 timestamp
视频数据至少包含：
- clip caption：整段描述；
- dense caption：分时间段描述；
- temporal grounding：文本→起止秒；
- frame QA：局部帧问答；
- identity/action：跨帧对象和动作；
- trajectory：观察—动作—新观察。

抽帧需记录原始时间。固定FPS适合均匀动作；动态FPS适应快慢视频；内容自适应采样可防止稀疏关键事件被漏掉，但多数报告未充分消融。显式`<3.0 seconds>`使token与真实时间对齐，不能用帧序号替代。
### 7.6 GUI/tool trajectory
典型序列：
```text
system: tool schema
user: task
image: screenshot_0
assistant: thought_0 + action_0(args)
tool: execution result
image: screenshot_1
assistant: thought_1 + action_1(args)
...
assistant: final answer
```
loss可覆盖thought/action/final，也可只监督action/final。坐标系必须与预处理后截图一致。轨迹若只包含成功路径，模型缺少恢复能力；若包含工具输出，packing和上下文压缩必须防止旧截图迅速耗尽窗口。
Kimi K2.5 Zero-Vision SFT只用文本工具轨迹冷启动，再用必须看图的outcome RL纠正忽略图像。它依赖前置图文预训练，不意味着“无视觉数据也能凭空学会看图”。
### 7.7 纯文本 replay
多模态数据语言质量常更差、答案更短。Qwen-VL Stage2加入7.8M纯文本；DeepSeek-VL2约30%纯文本；InternVL3预训练50B text+150B multimodal；Kimi长上下文阶段回放75%短数据。
纯文本 replay作用是保护知识、代码、长文和指令行为，不是装饰性数据。CogVLM用结构冻结保护语言路径；联合训练路线则用数据配比和较小视觉学习率保护。
### 7.8 模态边界与结构 token
常见边界：
```text
<vision_start> ... <vision_end>
<image_id_1> ... <image_id_2>
[IMAGE BREAK] / [IMAGE END]
<tile_newline> / <view_separator>
<3.0 seconds>
<ref> ... </ref> <box> ... </box>
```
structure token有三类作用：防止不同媒体混淆、保存二维/时间结构、提供输出parser锚点。它们不是“视觉内容token”，但会消耗上下文并影响attention路径。
### 7.9 多图/视频 sampling
多图训练要区分：
- 同一事件的多视角；
- 不同图片对比；
- 网页图文交错；
- in-context示例；
- 视频帧退化成多图。

若训练只见单图，推理拼多图可能发生图文绑定错误。若每帧独立且无时间位置，视频任务可能只靠静态语义。采样策略应报告图数、帧数、FPS、每图token和总预算。
### 7.10 Packed sequence 与 block mask
packing把多个短样本拼成一条长序列提高利用率，但必须用block mask防止样本A读样本B。Molmo还把同一图像的多个caption打包，隔离不同标注却复用一次图像编码，训练时间减半以上。
动态分辨率产生不同视觉长度，按总送入LLM的序列长度做packing/负载均衡比按图像张数更合理。Qwen2.5/3、Kimi、GLM都需要序列/上下文并行处理极长样本。
### 7.11 多阶段冻结策略
| 阶段 | 常见可训练模块 | 典型目标 | 风险 |
|---|---|---|---|
| 视觉预训练 | ViT/ConvNet | 对比、SSL、caption、OCR | 全局语义压过局部结构 |
| connector对齐 | connector；ViT可冻/低LR；LLM常冻 | caption NTP或ITC/ITM/ITG | 随机接口扰动LLM |
| 联合继续预训练 | ViT+connector+LLM | text+multimodal NTP | 语言回退、数据噪声 |
| SFT | 全量或冻结ViT | assistant/action target | 模板捷径、短答化 |
| Preference/RL | policy+reference/verifier | DPO/MPO/GRPO/GSPO | reward hacking、过思考 |
| Test-time/Agent | 参数冻结 | BoN、PRM、工具、swarm | 成本不公平、错误累积 |
### 7.12 目标函数与损失 token
统一NTP并不等于所有token都计loss。常见规则：
- visual soft token：不预测；
- system/user/prefix：通常loss mask；
- assistant answer：预测；
- tool schema：输入，不预测；
- action/特殊格式：预测；
- padding与其他packed样本：隔离并mask。

BLIP-2另有ITC/ITM/ITG；CLIP/SigLIP用对比；Kimi MoonViT阶段结合SigLIP+caption；DINOv3用自监督与Gram；RL阶段使用结果/偏好奖励。不能把所有VLM训练概括成“都是NTP”。
### 7.13 合成数据与教师依赖
| 合成类型 | 例子 | 可审计性 | 主要风险 |
|---|---|---|---|
| 文本重写/教师回答 | LLaVA、OneVision | 中等，依赖教师版本 | 风格闭环、视觉未见细节 |
| VLM看图伪标注 | recaption、长描述、CoT | 低到中 | 教师幻觉、同源偏差 |
| 程序渲染 | PixMo-Docs/Clocks、HTML/OCR | 高 | 合成域差异 |
| detector/SAM伪框mask | Qwen/PaliGemma/SAM3数据引擎 | 中等，可IoU验证 | 漏检、ontology偏差 |
| rejection sampling | 数学/代码/视觉题 | 取决于verifier | 只保留易验证分布 |
| self-evolving Agent轨迹 | Qwen3、Kimi、GLM | 较低 | 环境/策略同源、错误放大 |

Molmo不依赖专有VLM直接看图生成监督，但仍使用LLM、OCR、检测器、ASR和渲染器；“不蒸馏闭源VLM”不等于“无合成数据”。

---

## 8. 家族谱系：同源、分叉与名字陷阱
### 8.1 总谱系树
```text
开放语义视觉
CLIP / ALIGN / SigLIP / DINO
├─ late-fusion 检索路线
├─ frozen-bridge 路线
│  ├─ Flamingo: Perceiver + LM-layer gated XAttn
│  └─ BLIP-2: Q-Former → frozen LLM soft prompt
├─ LLaVA-style visual-prefix 路线
│  ├─ LLaVA → 1.5 → NeXT → OneVision
│  ├─ PaliGemma（Prefix-LM / transfer）
│  ├─ Molmo（PixMo / local pooling）
│  ├─ Pixtral（native resolution）
│  ├─ DeepSeek-VL → VL2（dynamic tile + MoE/MLA）
│  ├─ InternVL Chat/2 → 3 → 3.5
│  └─ Kimi-VL（MoonViT + MoE）
├─ deep-specialist 路线
│  ├─ CogVLM: token-type-specific QKV/FFN
│  ├─ Cambrian: multi-tower SVA
│  └─ Qwen3-VL: DeepStack
└─ native multimodal + RL + Agent
   ├─ Qwen-VL → Qwen2-VL → Qwen2.5-VL → Qwen3-VL → Qwen3.5（取消 VL 后缀）
   ├─ InternVL3/3.5
   ├─ GLM-4.1V → 4.5V → 4.6V
   └─ Kimi K2.5（K2 language line + MoonViT-3D + PARL）
```
### 8.2 LLaVA 系
| 代际 | 视觉入口 | 关键变化 | 未变主轴 |
|---|---|---|---|
| LLaVA | CLIP ViT-L/14 + Linear | GPT-4合成指令、两阶段训练 | visual prefix + decoder-only |
| LLaVA-1.5 | CLIP + 2-layer MLP | 更高分辨率、更多学术数据 | 简单connector |
| LLaVA-NeXT | AnyRes tile | 高分辨率与长宽比 | tile后统一前缀 |
| OneVision | SigLIP + MLP | 单图/多图/视频同量级预算、Qwen2 | 以数据和token策略统一场景 |

OneVision不是全新fusion架构；它仍是简单visual-prefix，主要创新在AnyResMax、每场景token预算和课程迁移。
### 8.3 Qwen-VL 到 Qwen3.5：从独立 VL 分支到默认原生多模态
| 代际 | Vision encoder | Connector | Position/time | 核心变化 |
|---|---|---|---|---|
| Qwen-VL | OpenCLIP ViT-bigG，固定448 | 256 query cross-attn | adapter二维绝对位置 | OCR/grounding/多图统一接口 |
| Qwen2-VL | DFN ViT，动态网格 | $2\times2$ merger | ViT 2D-RoPE + LLM M-RoPE | 固定query→变长token；图像视频统一 |
| Qwen2.5-VL | 从零原生ViT，28 window+4 global | $2\times2$ merger | absolute-time M-RoPE | 文档HTML、动态FPS、真实像素坐标、GUI |
| Qwen3-VL | SigLIP-2系继续训练 | merger + DeepStack | Interleaved-MRoPE + 文本时间戳 | 256K、Dense/MoE、Thinking/RL/Agent |
| Qwen3.5 | 复用 Qwen3-VL 系视觉塔；公开配置未启用 DeepStack | $2\times2$ merger，early fusion | Interleaved-MRoPE；原生262K | 取消独立 VL 后缀；Qwen3-Next式3:1 GDN/full-attn + 高稀疏MoE + MTP + 异步RL |

Qwen-VL的“adapter query”与后代“merger”不是同一connector；Qwen2.5绝对时间position与Qwen3文本时间戳也不是简单累加，而是代际替换。Qwen3.5 的关键又变了：视觉不再是独立产品线，而成为所有 checkpoint 的默认能力；详细见 [[Qwen3.5 - 原生多模态智能体与混合架构]]。
### 8.4 InternVL 1 → 3.5
| 阶段 | 架构 | 主要瓶颈 |
|---|---|---|
| 原始 InternVL | 6B InternViT→8B QLLaMA(96 query)→可选聊天LLM | 中间件极重 |
| InternVL Chat/2/2.5 | InternViT→pixel unshuffle→MLP→LLM，动态tile | 部署更直接，依赖数据/SFT |
| InternVL3 | 同主架构 + V2PE | native joint pretrain、MPO、VisualPRM |
| InternVL3.5 | 同主架构 + ViCO/ViR | Cascade RL、视觉token压缩、DvD服务 |

InternVL3不是“QLLaMA升级版”，而是已经换轨的ViT–MLP–LLM。家族同源在InternViT、数据哲学与组织团队，不代表connector连续继承。
![](../../10-Papers/VLM/assets/InternVL%20-%20规模化视觉基础模型与通用对齐/figures/02-training-strategy.png)
> **原始 InternVL Figure 3。** 先对比训练大视觉/文本编码器，再冻结主体训练query/cross-attention；与后代直接pixel-unshuffle+MLP不同。

![](../../10-Papers/VLM/assets/InternVL3%20-%20原生多模态预训练与测试时扩展/figures/02-architecture.png)
> **InternVL3架构。** 动态tile、InternViT、pixel unshuffle与MLP形成后代主线。
### 8.5 Kimi-VL → Kimi K2.5：不是简单版本号继承
Kimi-VL是Moonlight 16B MoE语言线上的A3B VLM：MoonViT 400M、两层MLP、128K、约3.2B激活。Kimi K2.5则以Kimi K2 1.04T/32B-A32B语言基座为核心，引入MoonViT-3D、早期视觉融合、Zero-Vision SFT、视觉RL与Agent Swarm。
共享的是MoonViT/原生分辨率、MoE工程与多模态训练思想；不共享的是语言基座规模、视频encoder形态、预训练时间线和Agent后训练。更准确关系是：**Kimi-VL提供多模态视觉前史，K2.5是K2语言/Agent路线与MoonViT-3D重新合流**。
![](../../10-Papers/VLM/assets/Kimi-VL%20-%20高效多模态MoE与长思考/figures/01-architecture.png)
> **Kimi-VL Figure 3。** MoonViT–MLP–16B MoE，约3B激活。

![](../../10-Papers/VLM/assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/figures/04-early-fusion-ablation.png)
> **Kimi K2.5 Figure 9。** 固定图文总预算下，早期低比例视觉共训优于晚期高比例补课；这证明训练时点是独立变量。
### 8.6 DeepSeek-VL → VL2
DeepSeek-VL使用固定分辨率SigLIP+SAM-B双视觉encoder和dense/旧语言基座；VL2改为单SigLIP动态切片、全局缩略图、结构token、DeepSeekMoE，并在中/大模型使用MLA。
VL2不是简单“多一个MoE”：它同时更换视觉组织、视觉塔数量、connector布局、语言attention和数据规模。connector里的pixel shuffle与Qwen/InternVL同名近似，但VL2额外用newline/view separator重建二维tile语法。
### 8.7 MiniCPM-V 路线
MiniCPM-V早期即强调learned-query 2D-Resampler，以较少视觉token服务小型LLM。4.5把同一思想扩展为3D-Resampler：图像按空间压缩，视频按相邻帧package做时空联合cross-attention；再配合文档视觉破坏和短/长Hybrid RL。
它与Flamingo都叫Resampler，但Flamingo输出每视觉输入64 latent并在LM层间cross-attend；MiniCPM的Resampler直接产生LLM视觉前缀，视频还在package内联合压缩。
### 8.8 独立路线
| 路线 | 核心差异 | 不应被归入 |
|---|---|---|
| Molmo/PixMo | 开放人工长描述、pointing、局部attention pooling | 不是LLaVA数据蒸馏支线 |
| Pixtral | 从头训练native-resolution ViT、row break、block packing | 不是AnyRes tile |
| PaliGemma | 小型base、Prefix-LM、全参数transfer、`<loc>/<seg>` | 不是开箱聊天模型 |
| Cambrian | 多视觉塔+空间局部SVA+视觉中心评测 | 不是简单多encoder concat |
| CogVLM | 每层按token类型分QKV/FFN | 不是普通MoE或Flamingo |
### 8.9 共享 LLM 不代表同架构
Molmo-7B-D与OneVision都可用Qwen2，但前者有重叠crop+局部attention pooling+PixMo数据，后者有SigLIP+MLP+场景token预算；Qwen2-VL本身又有动态ViT、merger和M-RoPE。共享Qwen2只说明语言初始化同源，不说明视觉计算图同源。
同理，InternVL3和Qwen2.5-VL都用Qwen2.5系LLM，却在视觉encoder、tile/native grid、位置和后训练上不同。
### 8.10 connector同名不代表计算图相同
- “Resampler”：Flamingo Perceiver的K/V含visual+latent；MiniCPM 3D-Resampler的K/V是时空package；
- “MLP projector”：有的长度不变，有的前面已pixel-unshuffle；
- “merger”：Qwen是动态网格2×2，Qwen3另有DeepStack多层merger；
- “cross-attention”：BLIP-2在pre-LLM Q-Former，Flamingo在LM layers，Cambrian在多塔局部聚合；
- “MoE”：DeepSeek/Kimi/Qwen3主要是语言FFN稀疏；CogVLM是按token类型固定分流，语义不同。

---

## 9. 核心模型统一架构卡片与大矩阵
### 9.1 统一字段说明
每张卡统一回答：视觉预处理、encoder、connector/视觉token、fusion、position/mask、decoder/目标、输出、冻结与主要成本。数字仅采用本地官方论文精读可核验口径；未披露即写未知。
### 9.2 起点与桥接模型
#### 9.2.1 CLIP
- **预处理/encoder**：固定分辨率；ResNet或ViT；图像/文本双塔。
- **connector/token**：无生成式connector；全局图像embedding。
- **fusion/position**：最终余弦相似度late fusion；塔内各自位置编码。
- **decoder/目标**：无生成decoder；对称图文对比损失。
- **输出**：相似度、检索、零样本类别概率。
- **成本/边界**：缓存高效；不自由生成，patch局部不是主目标。
#### 9.2.2 Flamingo
- **预处理/encoder**：图像320²；视频训练8帧1FPS；冻结NFNet-F6。
- **connector/token**：Perceiver Resampler，固定64 token/视觉输入。
- **fusion**：冻结Chinchilla LM多层插gated cross-attention；语言Q、视觉K/V。
- **mask/position**：per-image最近前置图mask；视频加时间position。
- **目标/冻结**：冻结视觉与LM，只训Resampler和约10B跨模态层（80B版）。
- **成本/边界**：多图ICL强；1536 TPUv4×15天，数据/权重不开放。
#### 9.2.3 BLIP-2
- **预处理/encoder**：224²；冻结CLIP ViT-L或EVA ViT-g，取倒数第二层。
- **connector/token**：188M Q-Former，32×768 query输出。
- **fusion**：Q-Former先读视觉；投影后作为OPT前缀或FlanT5 encoder输入。
- **mask/position**：ITC/ITG/ITM三种mask；LLM后端按架构各自处理。
- **目标/冻结**：ViT/LLM冻结；阶段1 ITC+ITM+ITG，阶段2 LM生成。
- **成本/边界**：参数高效、单图强；固定32 query与多图数据缺失。
#### 9.2.4 LLaVA
- **预处理/encoder**：CLIP ViT-L/14，224/336级固定输入；视觉塔冻结。
- **connector/token**：Linear，长度不变。
- **fusion**：visual prefix与Vicuna文本统一causal self-attention。
- **position/mask**：标准LLM序列位置；assistant-only loss。
- **目标/冻结**：Stage1只训projector；Stage2训projector+Vicuna。
- **成本/边界**：8×A100数小时级；视觉粒度与GPT-4合成闭环是边界。
### 9.3 2023–2024 架构分叉
#### 9.3.1 Qwen-VL
- **预处理/encoder**：224→448固定方形；OpenCLIP ViT-bigG，patch14。
- **connector/token**：带2D位置的单层cross-attention，1024 patch→固定256 query。
- **fusion**：visual prefix进入Qwen-7B统一self-attention。
- **输出**：文本、OCR、`<ref>/<box>`归一化坐标、多图对话。
- **训练**：14亿图文对齐→高分辨率多任务→35万SFT；冻结计划分阶段变化。
- **成本/边界**：视觉长度可预测；固定448和256瓶颈。

![](../../10-Papers/VLM/assets/Qwen-VL%20-%20定位与文字读取的通用视觉语言模型/figures/02-training-pipeline.png)
> **Qwen-VL Figure 3。** 三阶段训练同时改变分辨率、数据和冻结策略。
#### 9.3.2 CogVLM
- **预处理/encoder**：EVA2-CLIP-E，224→490；二层SwiGLU adapter。
- **connector/token**：MLP不压长度；所有视觉token共享一个LLM position id。
- **fusion**：每层视觉专用QKV/FFN，文本路径冻结，统一causal attention。
- **目标/输出**：answer-only NTP；自然语言与双中括号格式的 `(x0,y0,x1,y1)` grounding。
- **冻结**：约6.5B视觉专家可训练；SFT时视觉塔低LR解冻。
- **成本/边界**：语言保持结构清晰；参数/训练重，视频多图未验证。
#### 9.3.3 原始 InternVL
- **预处理/encoder**：196/224→336；6B InternViT。
- **connector/token**：8B QLLaMA，96 queries，约1B新增cross-attention。
- **fusion**：QLLaMA先对比/生成，再可经MLP接Vicuna/InternLM。
- **目标**：Stage1 ITC；Stage2 ITC+ITM+ITG；Stage3 SFT。
- **冻结**：Stage2冻结InternViT/QLLaMA主体，只训query/cross-attention。
- **成本/边界**：640×A100阶段1；通用但重，后代弃用QLLaMA。
#### 9.3.4 Cambrian-1
- **预处理/encoder**：CLIP336、SigLIP384、ConvNeXt1024、DINOv2 518四塔。
- **connector/token**：SVA二维latent网格；标准576，可强压36。
- **fusion**：局部多塔cross-attention，并在LLM多层重复聚合。
- **position/mask**：空间局部块绑定；decoder为visual-prefix生成。
- **目标/冻结**：connector预训练→Cambrian-7M SFT；可解冻视觉塔。
- **成本/边界**：视觉表征互补；四塔前向昂贵。
#### 9.3.5 PaliGemma
- **预处理/encoder**：SigLIP-So400m；224/448/896独立checkpoint。
- **connector/token**：零初始化Linear；256/1024/4096视觉token。
- **fusion/mask**：图像+prefix双向Prefix-LM，suffix causal。
- **目标/输出**：suffix NTP；自然语言、1024 `<loc>`、128 `<seg>`。
- **冻结**：Stage1/2/3倾向全参数训练；全训优于冻结LM。
- **成本/边界**：<3B、迁移强；需下游微调，多分辨率checkpoint维护。

![](../../10-Papers/VLM/assets/PaliGemma%20-%20可迁移视觉语言基础模型/figures/01-model-architecture.png)
> **PaliGemma Figure 1。** SigLIP→Linear→Gemma，极简结构把变量集中到Prefix-LM和全参数迁移。
#### 9.3.6 Molmo
- **预处理/encoder**：CLIP ViT-L/14@336；全图+重叠高分辨率crop。
- **connector/token**：倒数3/10层拼接；2×2 local attention pooling→MLP。
- **fusion**：visual prefix进入OLMo/Qwen2/OLMoE decoder。
- **position/mask**：crop布局和point稳定排序；同图多caption packing隔离。
- **目标/输出**：文本NTP；XML point、先指点后计数。
- **成本/边界**：PixMo开放数据；72B约12.5K H100 GPU-hours，语言能力有回退。
#### 9.3.7 LLaVA-OneVision
- **预处理/encoder**：SigLIP；单图AnyResMax-9，多图base，视频抽帧。
- **connector/token**：2层MLP；单图最高7290，多图约8748，视频32×196=6272示例预算。
- **fusion**：Qwen2 visual-prefix unified self-attention。
- **position/mask**：LLM常规序列位置；场景差异主要在token组织。
- **训练**：projector对齐→4M知识→3.2M单图→约1.6M混合。
- **成本/边界**：场景迁移强；约99.8%知识数据合成，统一后部分单图回退。

![](../../10-Papers/VLM/assets/LLaVA-OneVision%20-%20单图多图视频统一迁移/figures/02-token-strategy.png)
> **OneVision Figure 3。** 单图保细节、多图保数量、视频保帧数，总视觉预算控制在同量级。
#### 9.3.8 Pixtral 12B
- **预处理/encoder**：原生宽高比；从头训练400M Pixtral-ViT，patch16。
- **connector/token**：两层MLP；$N=n_hn_w+n_h+1$，含row break与image end。
- **fusion**：visual/text交错进入Mistral Nemo 12B，统一causal self-attention。
- **position/mask**：ViT 2D-RoPE；图像packing block mask；LLM 1D RoPE。
- **目标/冻结**：instruction tuning已披露，完整阶段/冻结未知。
- **成本/边界**：128K且权重开放；训练数据、超参、算力未知。

![](../../10-Papers/VLM/assets/Pixtral%2012B%20-%20原生分辨率视觉语言模型/figures/03-full-architecture.png)
> **Pixtral Figure 3。** 多图与文本按对话次序交错，视觉尺寸直接决定token成本。
### 9.4 2024–2026 动态视觉、MoE 与 Agent
#### 9.4.1 Qwen2-VL
- **预处理/encoder**：Naive Dynamic Resolution；675M DFN ViT改2D-RoPE；Conv3D tube深度2。
- **connector/token**：2×2 merger；近似每28×28像素一个LLM视觉token。
- **fusion**：visual prefix进入Qwen2 2B/7B/72B。
- **position**：ViT 2D-RoPE；LLM M-RoPE(t,h,w)。
- **目标/输出**：text-only NTP；文本、OCR、框、函数/Agent动作。
- **成本/边界**：约1.4T口径；高像素直接抬高LLM长度，视频最多768帧评测。
#### 9.4.2 Qwen2.5-VL
- **预处理/encoder**：原生动态ViT，32层，28 window+4 global；动态FPS。
- **connector/token**：2×2 merger；LLM网格步长相当28像素。
- **fusion**：Qwen2.5 visual-prefix unified self-attention。
- **position**：ViT 2D-RoPE；LLM absolute-time M-RoPE。
- **目标/输出**：4.1T预训练，约2M SFT+DPO；HTML、实际像素point/box、GUI action。
- **成本/边界**：window只降ViT成本；缺组件消融与完整训练成本。

![](../../10-Papers/VLM/assets/Qwen2.5-VL%20-%20文档视频与视觉智能体/figures/02-architecture.jpeg)
> **Qwen2.5-VL Figure 1。** Conv3D、window/global ViT、2×2 merger与绝对时间共同构成视觉入口。
#### 9.4.3 Qwen3-VL
- **预处理/encoder**：SigLIP-2系动态ViT；图像/视频/长文档。
- **connector/token**：2×2 merger + DeepStack三层注入。
- **fusion**：Qwen3 Dense/MoE unified self-attention；前3层多级视觉残差。
- **position**：Interleaved-MRoPE；视频显式文本timestamp。
- **目标/输出**：S0 67B→S1约1T→S2约1T→S3 100B；Instruct/Thinking/RL；文本/3D框/工具。
- **成本/边界**：原生256K；235B-A22B仍需存总权重，1M为YaRN外推。
#### 9.4.4 InternVL3
- **预处理/encoder**：按宽高比匹配网格，448 tile+thumbnail；InternViT 300M/6B。
- **connector/token**：pixel unshuffle，每tile 1024→256；两层MLP。
- **fusion**：Qwen2.5/InternLM3 visual-prefix unified self-attention。
- **position**：V2PE让视觉位置增量可小于1。
- **目标/输出**：约200B native joint pretrain、21.7M SFT、300K MPO；文本/框/GUI/CoT。
- **成本/边界**：全参数联合训练；动态tile与KV成本，代码能力不全保留。

![](../../10-Papers/VLM/assets/InternVL3%20-%20原生多模态预训练与测试时扩展/figures/03-native-pretraining-ablation.png)
> **InternVL3 Figure 3。** 控制架构/初始化/数据后，native joint pretraining相对旧projector warmup有直接增益。
#### 9.4.5 InternVL3.5
- **预处理/encoder**：延续动态tile与InternViT。
- **connector/token**：普通每tile 256；Flash由ViR选择64/256。
- **fusion**：visual-prefix；DvD可把ViT+MLP放独立vision server。
- **position/decoder**：延续InternVL3路线；Dense/MoE Qwen3/GPT-OSS语言侧。
- **目标**：250B预训练、130B SFT、MPO→GSPO、ViCO约30B。
- **成本/边界**：约50%视觉token减少、平均降0.4–0.5分；压缩训练昂贵。

![](../../10-Papers/VLM/assets/InternVL3.5%20-%20级联强化学习与视觉压缩/figures/03-dvd-deployment.png)
> **InternVL3.5 Figure 4。** DvD将视觉并行计算与LLM自回归服务解耦；系统吞吐优化不等于模型FLOPs减少。
#### 9.4.6 DeepSeek-VL2
- **预处理/encoder**：384 tile，$mn\le9$，全局缩略图+局部tile；SigLIP-SO400M。
- **connector/token**：27×27→14×14 pixel shuffle+MLP；newline/view separator。
- **fusion**：DeepSeekMoE visual-prefix；Small/Large使用MLA。
- **position/mask**：结构token保存二维布局；LLM细节沿DeepSeek基座。
- **目标/输出**：约2B对齐→约800B全参预训练→约20B SFT；文本/框/代码。
- **成本/边界**：1.0/2.8/4.5B激活；总权重、专家通信和动态token仍在。

![](../../10-Papers/VLM/assets/DeepSeek-VL2%20-%20MoE动态高分辨率视觉语言模型/figures/02-architecture.png)
> **DeepSeek-VL2 Figure 2。** 单SigLIP动态切片与MoE/MLA语言侧组合，结构token是其视觉序列特色。
#### 9.4.7 Kimi-VL
- **预处理/encoder**：MoonViT原生分辨率、NaViT packing；后版单图最多3.2M pixels。
- **connector/token**：2×2 pixel shuffle+两层MLP。
- **fusion**：Moonlight 16B总/2.8B激活MoE visual-prefix。
- **position**：ViT插值绝对embedding+2D-RoPE；LLM 128K RoPE扩展。
- **目标/输出**：2.1T视觉/对齐+1.4T joint+0.6T cooldown+0.3T long；Instruct/Thinking/RL。
- **成本/边界**：约3.2B总激活不等于3.2B权重；128K边缘召回下降。
#### 9.4.8 Kimi K2.5
- **预处理/encoder**：MoonViT-3D，共享图像视频参数；最多4帧时空块，约4×时间压缩。
- **connector/token**：MLP projector；具体层/维未知。
- **fusion**：K2 1.04T总/32B激活MoE；早期视觉长周期联合预训练。
- **position**：视觉时空位置细节未知；上下文256K。
- **目标/输出**：Zero-Vision SFT→视觉outcome RL→跨模态能力RL→PARL；工具/子Agent token。
- **成本/边界**：并行Swarm降墙钟但可能增总算力；预训练/硬件披露不足。

![](../../10-Papers/VLM/assets/Kimi%20K2.5%20-%20视觉智能体与并行智能体强化学习/figures/02-agent-swarm.png)
> **Kimi K2.5 Figure 3。** 只训练orchestrator，冻结subagents；这是runtime/系统拓扑，不是视觉encoder内部融合。
#### 9.4.9 GLM-4.5V
- **预处理/encoder**：原生宽高比，AIMv2-Huge；2D卷积换Conv3D，图像复制。
- **connector/token**：MLP；评测图像最多6144、视频48000视觉token口径。
- **fusion**：GLM dense/MoE visual-prefix unified self-attention。
- **position**：ViT插值绝对embedding+2D-RoPE；LLM 3D-RoPE细节未知；文本time index。
- **目标/输出**：预训练→长上下文→SFT→多域RL/RLCS；文本/框/GUI/coding/tool。
- **成本/边界**：106B-A12B；verifier工程强，完整训练成本与3D-RoPE未披露。

![](../../10-Papers/VLM/assets/GLM-4.5V%20-%20可扩展强化学习的多模态推理/figures/02-architecture.png)
> **GLM-V Figure 2。** 可确认视觉入口和文本时间token；3D-RoPE内部实现仍未知。
#### 9.4.10 MiniCPM-V 4.5
- **预处理/encoder**：图像LLaVA-UHD式切片；视频相邻帧package；视觉塔名称未披露。
- **connector/token**：共享3D-Resampler；常见448图可64token，视频6帧可64token口径。
- **fusion**：Resampler前缀进入约8/8.7B LLM。
- **position**：query带2D+时间位置，形式未披露。
- **目标/输出**：分阶段冻结、文档视觉破坏、General/Long-CoT SFT、Hybrid GRPO、RLAIF-V。
- **成本/边界**：8×A100视频效率强；无手机/NPU/量化/功耗实测。
### 9.5 统一矩阵：结构与训练
| 模型 | 视觉预处理 | Vision encoder | Connector/视觉长度 | Fusion | Position/mask | Decoder/目标 |
|---|---|---|---|---|---|---|
| CLIP | 固定resize | ResNet/ViT | 全局pool | late dot-product | 塔内各自 | 双向对比，无生成 |
| Flamingo | 320²/抽帧 | NFNet-F6 | Perceiver 64 | LM层gated XAttn | per-image mask | Chinchilla NTP |
| BLIP-2 | 224² | CLIP/EVA ViT | Q-Former 32 | pre-LLM soft prompt | ITC/ITG/ITM masks | OPT/T5 LM |
| LLaVA | 固定224/336 | CLIP ViT-L | Linear，不压长 | unified causal | 1D LLM pos | Vicuna assistant NTP |
| Qwen-VL | 固定448 | ViT-bigG | 256 query | unified causal | adapter 2D abs | Qwen NTP |
| CogVLM | 固定→490 | EVA2-CLIP-E | MLP，不压长 | token-type专家 | 视觉共享pos id | Vicuna NTP |
| InternVL | 固定/336 | InternViT-6B | QLLaMA 96 | pre-LLM重桥接 | query masks | ITC/ITM/ITG+LM |
| Cambrian | 多塔多尺度 | 4视觉塔 | SVA 576/36 | 多层局部XAttn | 空间latent | LLM NTP |
| PaliGemma | 224/448/896 | SigLIP | Linear，不压长 | Prefix-LM | 双向prefix | Gemma suffix NTP |
| Molmo | 全图+重叠crop | CLIP/SigLIP等 | local pool $1/4$ | unified causal | crop布局/packing | OLMo/Qwen2 NTP |
| OneVision | AnyRes/多图/帧 | SigLIP | MLP，场景预算 | unified causal | 1D LLM pos | Qwen2 NTP |
| Pixtral | native grid | Pixtral-ViT | MLP+row token | unified causal | ViT 2D-RoPE | Nemo NTP |
| Qwen2-VL | dynamic grid/tube | DFN ViT | merger $1/4$ | unified causal | 2D-RoPE+M-RoPE | Qwen2 NTP |
| Qwen2.5-VL | dynamic/FPS | native ViT | merger $1/4$ | unified causal | abs-time M-RoPE | Qwen2.5 NTP+DPO |
| Qwen3-VL | dynamic/timestamp | SigLIP-2系 | merger+DeepStack | unified+残差注入 | interleaved MRoPE | Dense/MoE NTP+RL |
| Qwen3.5 | dynamic image/video | Qwen3-VL系27层ViT | merger $1/4$；公开配置无DeepStack | early-fusion unified causal | interleaved MRoPE | 3:1 GDN/full-attn + Dense/MoE + MTP + async RL |
| InternVL3 | AnyRes tile | InternViT | unshuffle 1024→256 | unified causal | V2PE | Qwen2.5 NTP+MPO |
| InternVL3.5 | tile+ViR | InternViT | 64/256 route | unified causal | 继承V2PE路线 | NTP+MPO+GSPO |
| DeepSeek-VL2 | 动态tile+global | SigLIP | 196/tile+结构token | unified causal | newline/separator | MoE NTP |
| Kimi-VL | native pack | MoonViT | shuffle $1/4$+MLP | unified causal | abs+2D-RoPE | MoE NTP+RL |
| Kimi K2.5 | native 3D pack | MoonViT-3D | temporal pool+MLP | unified MoE | 细节未知 | NTP+视觉RL+PARL |
| GLM-4.5V | native/Conv3D | AIMv2-Huge | MLP | unified causal | 2D+3D-RoPE/time token | MoE NTP+RLCS |
| MiniCPM-V4.5 | tile/package | 未披露 | 3D-Resampler固定query | unified prefix | 2D+time query | NTP+Hybrid RL |
### 9.6 统一矩阵：输出、冻结、成本与开放性
| 模型 | 主要输出 | 冻结/可塑性 | 主要成本瓶颈 | 开放性/证据边界 |
|---|---|---|---|---|
| CLIP | 相似度/类别 | 两塔共同训练 | 大batch对比 | 代码权重开放，WIT不可重建 |
| Flamingo | 文本 | 两端冻、桥梁训 | 10B跨层+私有数据 | 无权重/训练代码 |
| BLIP-2 | 文本 | ViT/LLM冻 | 1.29亿图+Q-Former | LAVIS/权重开放 |
| LLaVA | 文本 | ViT冻，LLM可调 | 指令数据与视觉长度 | 高度开放，依赖GPT-4 |
| Qwen-VL | 文本/框 | 分阶段冻解 | 固定256瓶颈 | 权重代码开，数据不全 |
| CogVLM | 文本/框 | 文本路径冻、视觉专家训 | 6.5B训练参数 | 权重代码开，预训练重 |
| InternVL | 检索/文本 | 阶段化 | 6B+8B+LLM | 权重开，超大数据难复现 |
| Cambrian | 文本 | connector→解冻视觉 | 四塔前向 | 模型/数据/工具较完整开放 |
| PaliGemma | 文本/loc/seg | 全参迁移 | 4096视觉token | checkpoint开，部分数据私有 |
| Molmo | 文本/point | 全参 | crop+大LLM | 数据训练高度开放 |
| OneVision | 文本 | 全参低视觉LR | 7K–9K视觉token | 权重开，教师数据依赖 |
| Pixtral | 文本 | 未披露 | native像素→长前缀 | 权重/评测开，训练不透明 |
| Qwen2/2.5/3 | 文本/框/工具 | 阶段化/全参 | 动态视觉+长上下文 | 报告详细，数据成本不全 |
| InternVL3/3.5 | 文本/框/GUI | 全参native+RL | tile、BoN、ViCO | 权重代码开，成本不全 |
| DeepSeek-VL2 | 文本/框/代码 | Stage1 LLM冻→全参 | 总权重/EP/4096窗口 | 结构资源较清楚，消融缺 |
| Kimi-VL | 文本/CoT/GUI | 全参 | 16B总权重+128K | 权重代码开，内部数据多 |
| Kimi K2.5 | 文本/工具/subagent | 大规模联合+RL | 1.04T总权重+并发 | 报告可核验，配方成本不足 |
| GLM-4.5V | 文本/GUI/code | 全参+RL | 106B-A12B+verifier | 权重/reward开，3D-RoPE未知 |
| MiniCPM-V4.5 | 文本/短长CoT | 分阶段冻解 | 8.7B+视频package | 权重代码开，无真端侧实测 |

---

## 10. 参数冻结、语言保持、MoE 与复杂度
### 10.1 参数冻结是一种优化约束，不是能力等级
冻结可保护已有表示、减少优化器状态和避免灾难性遗忘，但也限制两端共同适配。Flamingo冻结两端仍有深层跨注意力；BLIP-2冻结两端依靠多目标Q-Former；PaliGemma全参数训练反而证明简单Linear足够。
选择冻结策略应按数据质量和任务决定：低质量OCR/caption可先只训视觉/connector，避免污染LLM；高质量交错、推理与纯文本混合再全参联合。
### 10.2 可塑性梯度
```text
最保守：只训练 Linear/MLP
  → 训练 Q-Former/Resampler
  → 插 LM-layer cross-attention
  → token-specific visual experts
  → 低学习率解冻 ViT/LLM
  → 全参数 native multimodal pretraining
```
可塑性越强，模型越能重组跨模态表示，但语言回退和训练耦合风险越大。不存在单调最优。
### 10.3 语言能力保持
CogVLM公开展示：直接用多模态数据训练语言部分时MMLU从约47快速跌到24.9附近。Molmo观察MMLU/MATH/HumanEval部分回退；InternVL3总体文本更强但78B HumanEval/MBPP低于同源Qwen2.5-72B Chat。
常见对策：
- 冻结语言路径；
- 纯文本replay；
- 视觉/connector使用不同学习率；
- gradual unfreeze；
- token/sample loss reweight；
- 每阶段做同源LLM前后回归。
### 10.4 MoE到底优化什么
MoE将FFN总容量与每token激活计算解耦：
$$h'=\sum_{e\in\operatorname{TopK}}\alpha_eE_e(h)+E_{shared}(h)$$
它主要优化：
- 同激活FLOPs下增加参数容量；
- 允许专家分工；
- 大batch服务时提高计算效率。
### 10.5 MoE不优化什么
MoE不会自动降低：
- 视觉encoder FLOPs；
- visual token数量；
- attention $O(L^2)$；
- KV cache长度；
- 总权重存储；
- expert parallel通信；
- 长CoT输出token。

Kimi-VL约2.8B decoder激活仍需存16B权重；Qwen3-VL 235B-A22B仍需存/分片235B；K2.5 1.04T/32B激活更依赖大规模EP。
### 10.6 MLA优化的是KV cache，不是视觉token
DeepSeek-VL2 Small/Large用rank512 MLA，将多头K/V压为latent cache。它可降低长序列decode缓存，但论文没有在VL2报告具体每层字节或吞吐消融。即使MLA有效，动态tile产生的视觉前缀仍要prefill，MoE专家仍需通信。
### 10.7 训练复杂度
完整成本至少拆成：
$$C_{total}=C_{vision}+C_{connector}+C_{LLM-prefill}+C_{LLM-decode}+C_{rollout}+C_{verifier}+C_{tool}$$
训练还要乘以反向、优化器状态、数据解码与并行通信。只报总参数或active params无法解释真实成本。
### 10.8 推理复杂度与部署
高分辨率服务应报告：像素、patch、connector token、文本token、TTFT、decode tok/s、峰值显存、batch、量化、硬件。InternVL3.5 DvD说明视觉前向与语言decode节奏不同，拆分服务池可提高吞吐；这属于系统调度增益，不应归为模型精度提升。
### 10.9 能力与架构不能简单因果
当新模型同时改变encoder、LLM、数据、分辨率、上下文、SFT、RL与工具时，跨代分数不能证明某个模块有效。证据强度从高到低：
1. 同backbone/数据/token/训练的单组件消融；
2. 同家族受控代际对照；
3. 不同模型公开benchmark；
4. 定性成功案例；
5. 官方声称无协议细节。

Qwen3 DeepStack、InternVL3 native pretrain、MiniCPM 3D-Resampler、K2.5早期融合属于较强受控证据；Qwen2.5绝对时间、GLM 3D-RoPE、Kimi-VL MoonViT vs fixed-grid缺少独立消融，应标记为系统整体证据。

---

## 11. 时间线：瓶颈如何迁移
### 11.1 国际主线
| 年月 | 工作 | 架构节点 | 暴露的新瓶颈 |
|---|---|---|---|
| 2021-03 | CLIP | 双塔图文对比、语言索引视觉 | 无自由生成、局部交互弱 |
| 2022-04 | Flamingo | Perceiver+LM层gated XAttn | 固定latent、私有数据/算力 |
| 2023-01 | BLIP-2 | 32-query Q-Former、冻结两端 | 多图/高分辨率瓶颈 |
| 2023-04 | LLaVA | Linear visual prefix+视觉指令 | 数据闭环、视觉粒度 |
| 2024-06 | Cambrian-1 | 多塔SVA、视觉中心评测 | 多塔成本 |
| 2024-07 | PaliGemma | Prefix-LM小型transfer base | 分辨率checkpoint/需微调 |
| 2024-08 | OneVision | 单图多图视频统一token预算 | 手工crop/帧预算 |
| 2024-09 | Molmo/PixMo | 开放数据、pointing、local pooling | 高级推理/语言保持 |
| 2024-10 | Pixtral | native-resolution ViT、2D-RoPE | 训练透明度、长前缀成本 |
| 2025-08 | DINOv3 | 稠密关系Gram anchoring | 如何接入VLM并保持语言对齐 |
| 2025-11 | SAM 3 | presence+mask+tracker工具 | 复杂语言仍需MLLM分解 |
| 2025-11 | Gemini 3 Pro | 官方披露原生多模态sparse MoE | 方法细节未知 |
| 2026-04至07 | GPT-5.5/5.6、Claude 4.x | 产品级工具/computer use | 无可复现视觉架构 |
### 11.2 国内主线
| 年月 | 工作 | 架构节点 | 暴露的新瓶颈 |
|---|---|---|---|
| 2023-08 | Qwen-VL | 256 query、OCR/文本框 | 固定448/固定瓶颈 |
| 2023-11 | CogVLM | 每层视觉专家 | 6.5B训练参数 |
| 2023-12 | InternVL | 6B InternViT+8B QLLaMA | 重型中间件 |
| 2024-09 | Qwen2-VL | 动态网格、2D-RoPE、M-RoPE | 高像素成本 |
| 2024-12 | DeepSeek-VL2 | 动态tile+结构token+MoE/MLA | 消融与短上下文 |
| 2025-02 | Qwen2.5-VL | window ViT、绝对时间、HTML/GUI | 组件因果不清 |
| 2025-04 | Kimi-VL | MoonViT+MoE+128K+Thinking | 总权重/长上下文成本 |
| 2025-04 | InternVL3 | native pretrain+V2PE+MPO/PRM | test-time预算与数据透明度 |
| 2025-07 | GLM-4.5V | AIMv2+3D-RoPE+RLCS | verifier单点风险 |
| 2025-08 | InternVL3.5 | Cascade RL+ViR+DvD | 路由proxy与额外蒸馏 |
| 2025-09 | MiniCPM-V4.5 | 3D-Resampler+Hybrid RL | 真端侧证据缺失 |
| 2025-11 | Qwen3-VL | DeepStack+Interleaved-MRoPE+256K | MoE/长上下文服务成本 |
| 2025-11 | DA3 | vanilla DINOv2+depth-ray | 专用几何与VLM分工 |
| 2026-02 | Qwen3.5 | 原生视觉默认化+3:1 GDN/full-attn+高稀疏MoE | 无完整技术报告；线性注意力精确回看与集群部署成本 |
| 2026-02 | Kimi K2.5 | 早期视觉融合+视觉RL+Swarm | 并发公平性/成本透明度 |
### 11.3 四次范式转移
1. **开放语义对齐**：标签空间封闭→语言可索引视觉。
2. **冻结底座桥接**：能检索→能让强LLM读取视觉并生成。
3. **指令/动态视觉**：能描述→能按用户意图处理OCR、grounding、多图视频。
4. **原生联合+RL+Agent**：能回答→能长推理、调用工具、在环境闭环行动。

**本文判断：**瓶颈从未消失，只从标签转到桥接、从桥接转到token预算、从token预算转到证据保持与系统可靠性。

---

## 12. 数据、RL 与能力分支的架构解释
### 12.1 通用图像问答
通用VQA同时受视觉证据和LLM知识影响。必须加入无图、错图、遮挡与反事实对照。Cambrian实测部分benchmark去图后仍显著高于随机，说明总分不能单独评价视觉塔。
### 12.2 OCR、文档与图表
这一分支主要受有效像素、patch局部性、二维结构token、视觉token预算和结构化target影响。高分辨率并非单调：Qwen2-VL公开消融中OCRBench在过高`min_pixels`下反而下降，说明训练分布和过度放大同样重要。
### 12.3 Grounding与计数
文本框统一接口，但置信度/拒识较弱；pointing可迫使逐对象对齐；mask工具适合像素穷尽。Molmo“先指点再计数”与SAM3“先枚举mask再计数”都说明显式对象化中间表示可提升数字答案可靠性。
### 12.4 视频与时间
视频要分别评测逐帧感知、事件定位、身份保持、长视频检索。更多帧与更多token只是输入预算；tube、时间RoPE、timestamp和3D-Resampler决定模型是否知道帧间关系。SAM3 tracker提示：统一VLM仍缺可扩展对象级记忆。
### 12.5 GUI与Agent
链条是截图理解→元素grounding→action schema→环境反馈→长程规划。Qwen2.5、Kimi、GLM、Qwen3都显示单步grounding远高于真实OS任务成功率。架构上需要视觉前缀；系统上还需要权限、确认、状态验证、回滚与审计。
### 12.6 空间与三维
VLM³证明标准Qwen3-VL通过焦距归一、文本坐标和几何数据可学习深度、对应、位姿；这说明统一token I/O可吸收3D任务，但不意味着专用几何模型失效。
![](../../10-Papers/3D-Reconstruction/assets/DA3%20-%20任意视角恢复视觉空间/figures/fig2_pipeline.png)
> **DA3 Figure 2。** vanilla DINOv2+Dual-DPT输出depth/ray/camera，以专用连续几何表示恢复空间。它适合作为VLM工具或教师，而不是被文本坐标完全替代。
### 12.7 端侧与高效
端侧必须同时报告参数、active params、量化、视觉token、KV cache、TTFT、tok/s、峰值内存、功耗和设备算子。MiniCPM-V 4.5在8×A100的视频效率不是手机/NPU证据；Kimi-VL约3B激活仍需存16B总权重；DeepSeek-VL2“单卡”口径缺量化和上下文条件。
### 12.8 RL层级
- 偏好优化：DPO/MPO，使用chosen/rejected；
- 在线可验证RL：数学、OCR、grounding、GUI结果奖励；
- 过程/测试时：VisualPRM、Best-of-N；
- 系统RL：PARL优化是否并行和如何拆任务。

GLM-V最重要的公开经验是坏verifier可让reward上涨、真实指标和其他领域同时下降。优化算法名字不是首要瓶颈，reward单元测试才是。

---

## 13. 闭源透明度、开放性与评测边界
### 13.1 Gemini、GPT、Claude：只写已知
| 系统 | 官方可确认 | 关键未知 |
|---|---|---|
| Gemini 3 Pro | 文本/图像/音频/视频；1M输入、64K文本输出；原生多模态sparse MoE；instruction/RL/偏好后训练类别 | 参数、专家/top-k、vision/audio encoder、fusion、分辨率/采样、训练token/算力 |
| GPT-5.5/5.6 | 图像/文件、computer use、工具、长上下文/多Agent产品能力与system card安全评测 | 参数、dense/MoE、vision encoder、connector、位置、预训练数据/目标 |
| Claude 4.x | 文本+图像、extended thinking、computer use、工具；system card安全/agentic评测 | 参数、vision encoder、fusion、图文目标、数据比例、训练算力 |

不能因它们支持图像就写“使用SigLIP+MLP”，不能因支持视频就写“使用3D-RoPE”，也不能从API token计费反推视觉token内部数量。
### 13.2 开放性四轴
1. 权重开放；
2. 代码/推理开放；
3. 数据/生成脚本开放；
4. 训练配方/成本/评测harness开放。

Molmo/Cambrian接近开放训练；Pixtral是权重与评测开放但训练不透明；Qwen/InternVL/Kimi/GLM多为权重代码开放、核心数据和总成本部分披露；闭源系统主要开放能力与安全报告。
### 13.3 Benchmark饱和与污染
网页、PDF、学术题和benchmark训练集进入大规模混合后，标准测试可能有样本、模板或教师风格重叠。建议：图像近重复、文本n-gram/embedding检索、时间切分、私有held-out、跨语言/版式外推、污染阈值公开。
### 13.4 GPT-as-judge与parser
LLaVA中GPT-4参与数据、参考和裁判，形成风格闭环。Pixtral展示prompt/parser可让部分baseline分数剧烈变化。exact-match优先规则验证；开放问答应公布rubric、judge版本、长度控制与人工抽检。
### 13.5 视觉依赖评测
更可靠证据链：
1. 感知：对象、文字、布局、区域、时间、深度；
2. 推理：必须依赖视觉的数学/科学/反事实；
3. 行动：GUI/tool闭环成功；
4. 系统：延迟、token、总FLOPs、恢复与安全。

“去图仍能答对”应成为基础诊断；Qwen3 long-CoT还专门删除不看图也能答对的训练题，说明视觉必要性也应进入数据过滤。
### 13.6 推理预算公平性
| 成本项 | 必报字段 |
|---|---|
| 输入 | 像素、图数/帧数、patch token、connector token、文本token |
| 输出 | answer/thinking token均值与上限 |
| 搜索 | 候选数N、PRM/critic次数、重试 |
| 工具 | tool calls、网页/文件访问、环境step |
| 并行 | Agent数、critical path、总token/总算力 |
| 服务 | TTFT、端到端延迟、吞吐、显存、能耗 |
### 13.7 Benchmark/消融边界
跨模型主表证明“整个系统在某协议有效”，不证明具体encoder/connector/position有效。只有同设置消融才能建立局部因果。没有消融时应写“架构机制合理、整体结果支持”，不能写“该模块带来X分”。

---

## 14. 架构选型树与未来演进
### 14.1 架构选型树
```text
任务是否只需要检索/分类？
├─ 是：CLIP/SigLIP 双塔，优先缓存与吞吐
└─ 否：需要生成/行动
   ├─ LLM必须冻结且训练预算小？
   │  ├─ 单图摘要：Q-Former
   │  └─ 多图交错/深融合：Flamingo式 gated XAttn
   ├─ 可全参数联合训练？
   │  ├─ 普通图像：Linear/MLP visual-prefix
   │  ├─ OCR/文档：native/dynamic ViT + 2×2 merger + row/page结构
   │  ├─ 超高分辨率：window/global ViT + 动态token预算
   │  └─ 视频：tube/3D encoder或时空Resampler + timestamp
   ├─ 必须像素级输出？
   │  ├─ 可粗量化：文本bbox/<loc>/<seg>
   │  └─ 高精度/穷尽/跟踪：SAM类mask head与memory工具
   ├─ 必须低延迟？
   │  ├─ 压输入：merger/ViR/3D-Resampler
   │  ├─ 压语言计算：小dense或MoE
   │  ├─ 压KV：MLA/GQA/短上下文
   │  └─ 压输出：短/长模式路由
   └─ 必须长程Agent？
      ├─ 单Agent：工具schema+环境反馈+恢复
      └─ 可分解宽任务：orchestrator+并行subagents，单列总成本
```
### 14.2 视觉token价值将被直接优化
当前压缩按几何规则、固定query或teacher KL。ViR已迈向语义路由，但仍是64/256二档。未来应直接优化“此patch对最终答案/动作/不确定性是否必要”，并将精度、TTFT、显存和能耗纳入预算约束。
### 14.3 稠密视觉质量会成为独立scaling轴
DINOv3证明全局语义提升可与局部关系退化并存。未来VLM视觉塔训练应同时监控caption/VQA、线性分割、对应、深度和patch相似图，并探索Gram/区域/跨帧关系约束。
### 14.4 Grounding从格式升级为校准证据
文本坐标缺presence、置信度和NMS。SAM3的presence/localization分解可迁移到VLM系统：先判断概念是否存在，再定位，再生成答案；高风险场景让专用工具验证mask/box。
### 14.5 视频需要对象级记忆
长上下文和更多帧不能自动保持身份。合理未来系统可能组合：VLM做语义/规划、时空encoder做片段表征、tracker维持对象ID、事件图压缩历史、文本timestamp服务检索。
### 14.6 专用视觉头与工具长期共存
语言head适合开放回答、计划与统一协议；mask/depth/geometry/action head适合连续、高精度、可校准输出。DINOv3、SAM3、DA3分别说明视觉token、像素落地和几何恢复仍有专用最优结构。
### 14.7 RL瓶颈是verifier与课程
未来重点不是再命名一个PPO/GRPO变体，而是：verifier校准、对抗hacking、跨域干扰、有效样本比例、difficulty curriculum、过程视觉证据和真实环境奖励。
### 14.8 Agent需要成本、安全与责任边界
多Agent可降墙钟但增并发资源。GUI/浏览器Agent必须同时报告提示注入、敏感动作确认、最小权限、沙箱、审计、回滚和失败恢复。系统能力不能被包装成裸模型能力。
### 14.9 原生多模态的下一步
“原生”目前多指从强base权重开始联合继续预训练，而非随机初始化。未来关键问题是：视觉何时进入、比例如何随阶段变化、纯文本如何回放、是否能形成真正共享的世界表征，而不是追求术语纯度。
### 14.10 能力与架构的因果研究
领域需要更多同backbone、同数据、同token、同FLOPs的最小消融；同时报告负结果和能力回退。一个组件在系统大版本有效，不代表它在小模型、不同数据或不同预算下仍有效。

---

## 15. 阅读路线
### 15.1 第一阶段：建立计算图坐标系
1. [[CLIP - 自然语言监督的可迁移视觉模型]]：双塔、对比目标、相似度非生成。
2. [[Flamingo - 少样本视觉语言上下文学习]]：Perceiver、语言Q/视觉K/V、per-image mask。
3. [[BLIP-2 - 冻结底座的高效视觉语言桥接]]：Q-Former与三种mask。
4. [[LLaVA - 视觉指令微调]]：visual prefix、assistant-only loss与数据范式。
### 15.2 第二阶段：视觉与connector
1. [[Qwen-VL - 定位与文字读取的通用视觉语言模型]]：固定query与坐标文本化。
2. [[CogVLM - 视觉专家驱动的深层融合]]：token-type-specific QKV/FFN。
3. [[Cambrian-1 - 视觉中心的多模态大模型设计]]：多塔SVA和视觉依赖评测。
4. [[DINOv3 - 大规模自监督视觉基础模型]]：稠密特征退化与关系锚定。
### 15.3 第三阶段：分辨率与场景统一
1. [[PaliGemma - 可迁移视觉语言基础模型]]：Prefix-LM与可迁移base。
2. [[LLaVA-OneVision - 单图多图视频统一迁移]]：AnyRes与场景预算。
3. [[Molmo - 开放数据驱动的视觉语言模型]]：local pooling、pointing与开放数据。
4. [[Pixtral 12B - 原生分辨率视觉语言模型]]：2D-RoPE、row break与block packing。
5. Qwen-VL→Qwen2-VL→Qwen2.5-VL→Qwen3-VL：观察query→merger→window/time→DeepStack的连续变化。
### 15.4 第四阶段：原生预训练、MoE与RL
1. [[InternVL - 规模化视觉基础模型与通用对齐]]：InternViT与QLLaMA历史起点。
2. [[InternVL3 - 原生多模态预训练与测试时扩展]]：native pretrain、V2PE、MPO、VisualPRM。
3. [[InternVL3.5 - 级联强化学习与视觉压缩]]：ViR、Cascade RL、DvD。
4. [[DeepSeek-VL2 - MoE动态高分辨率视觉语言模型]]：动态tile、结构token、MoE/MLA。
5. [[Kimi-VL - 高效多模态MoE与长思考]]：MoonViT、active params、128K与Thinking。
6. [[GLM-4.5V - 可扩展强化学习的多模态推理]]：verifier失败与RLCS。
7. [[MiniCPM-V 4.5 - 高效端侧多模态模型配方]]：3D-Resampler与Hybrid RL。
8. [[Kimi K2.5 - 视觉智能体与并行智能体强化学习]]：早期融合、Zero-Vision SFT与PARL。
### 15.5 第五阶段：把VLM放回视觉系统
1. [[SAM 3 - 概念驱动的开放词汇分割]]：presence、mask、tracker与MLLM工具分工。
2. [[DA3 - 任意视角恢复视觉空间]]：depth-ray几何接口。
3. [[VLM³ - 原生三维学习的视觉语言模型]]：标准VLM如何用文本I/O吸收深度/对应/位姿。
### 15.6 阅读每篇固定追问
- pixels/frames怎样预处理？
- patch/tube token shape是多少？
- connector的Q/K/V来自哪里，长度是否改变？
- fusion发生一次还是逐层，写回哪个residual stream？
- attention mask和loss mask分别是什么？
- ViT位置与LLM位置分别是什么？
- 哪些模块冻结，语言能力如何回归？
- 结构化输出是文本化、离散code还是专用head？
- 主要成本在ViT、prefill、decode、KV、MoE通信还是runtime？
- 最强结论来自受控消融、主表、定性案例还是作者声称？

---

## 16. 文献索引与最终结论
### 16.1 本地独立精读索引
| 年月 | 工作 | 本地笔记 | 关键架构词 |
|---|---|---|---|
| 2021-03 | CLIP | [[CLIP - 自然语言监督的可迁移视觉模型]] | 双塔、对比、late fusion |
| 2022-04 | Flamingo | [[Flamingo - 少样本视觉语言上下文学习]] | Perceiver、gated XAttn |
| 2023-01 | BLIP-2 | [[BLIP-2 - 冻结底座的高效视觉语言桥接]] | Q-Former、ITC/ITG/ITM |
| 2023-04 | LLaVA | [[LLaVA - 视觉指令微调]] | Linear visual prefix |
| 2023-08 | Qwen-VL | [[Qwen-VL - 定位与文字读取的通用视觉语言模型]] | 256 query、OCR、bbox |
| 2023-11 | CogVLM | [[CogVLM - 视觉专家驱动的深层融合]] | visual expert QKV/FFN |
| 2023-12 | InternVL | [[InternVL - 规模化视觉基础模型与通用对齐]] | InternViT、QLLaMA |
| 2024-06 | Cambrian-1 | [[Cambrian-1 - 视觉中心的多模态大模型设计]] | 多塔、SVA |
| 2024-07 | PaliGemma | [[PaliGemma - 可迁移视觉语言基础模型]] | Prefix-LM、loc/seg |
| 2024-08 | OneVision | [[LLaVA-OneVision - 单图多图视频统一迁移]] | AnyRes、token预算 |
| 2024-09 | Molmo | [[Molmo - 开放数据驱动的视觉语言模型]] | PixMo、local pooling、point |
| 2024-09 | Qwen2-VL | [[Qwen2-VL - 任意分辨率世界感知]] | dynamic grid、M-RoPE |
| 2024-10 | Pixtral | [[Pixtral 12B - 原生分辨率视觉语言模型]] | native ViT、row break |
| 2024-12 | DeepSeek-VL2 | [[DeepSeek-VL2 - MoE动态高分辨率视觉语言模型]] | tile、结构token、MLA |
| 2025-02 | Qwen2.5-VL | [[Qwen2.5-VL - 文档视频与视觉智能体]] | window ViT、absolute time |
| 2025-04 | InternVL3 | [[InternVL3 - 原生多模态预训练与测试时扩展]] | native pretrain、V2PE |
| 2025-04 | Kimi-VL | [[Kimi-VL - 高效多模态MoE与长思考]] | MoonViT、MoE、128K |
| 2025-07 | GLM-V | [[GLM-4.5V - 可扩展强化学习的多模态推理]] | AIMv2、3D-RoPE、RLCS |
| 2025-08 | InternVL3.5 | [[InternVL3.5 - 级联强化学习与视觉压缩]] | ViR、ViCO、DvD |
| 2025-09 | MiniCPM-V4.5 | [[MiniCPM-V 4.5 - 高效端侧多模态模型配方]] | 3D-Resampler、Hybrid RL |
| 2025-11 | Qwen3-VL | [[Qwen3-VL - 长上下文视觉推理与智能体]] | DeepStack、Interleaved-MRoPE |
| 2026-02 | Qwen3.5 | [[Qwen3.5 - 原生多模态智能体与混合架构]] | 默认原生视觉、3:1 GDN/full-attn、高稀疏MoE、MTP、异步RL |
| 2026-02 | Kimi K2.5 | [[Kimi K2.5 - 视觉智能体与并行智能体强化学习]] | early fusion、PARL、Swarm |

以上共 23 篇，构成本文架构主线的一手材料集。
### 16.2 视觉基座、工具与能力分支索引
| 工作 | 类型 | 正确关系 |
|---|---|---|
| DINOv3 | 自监督视觉基础模型 | 提供稳定稠密patch token，不负责语言生成 |
| SAM 3 | 开放词汇分割/跟踪 | 将noun phrase落到mask/ID，可作为VLM工具 |
| DA3 | 三维几何基础模型 | 输出depth/ray/camera/3D，不是对话模型 |
| VLM³ | 统一几何能力分支 | [[VLM³ - 原生三维学习的视觉语言模型]]以标准VLM文本接口学习depth/correspondence/pose，不纳入22篇架构主线计数 |
### 16.3 官方论文补充索引
- Qwen2-VL：arXiv:2409.12191
- Qwen2.5-VL：arXiv:2502.13923
- Qwen3-VL：arXiv:2511.21631
- Qwen3.5：官方技术博客与开放权重（截至 2026-10-20 无独立 arXiv 技术报告），<https://qwen.ai/blog?id=qwen3.5>
- DeepSeek-VL2：arXiv:2412.10302
- Kimi-VL：arXiv:2504.07491
- GLM-V：arXiv:2507.01006
- InternVL3.5：arXiv:2508.18265
- MiniCPM-V 4.5：arXiv:2509.18154
- Kimi K2.5：arXiv:2602.02276
### 16.4 最终结论
VLM架构史不是connector从简单到复杂的单线升级。CLIP让语言成为视觉索引；Flamingo让冻结LLM在深层反复读取视觉；BLIP-2让视觉在进入LLM前被query压成语言相关摘要；LLaVA证明简单visual prefix配合指令数据也能形成强交互；Qwen、Pixtral、MoonViT、InternVL把焦点推进到动态分辨率、结构token和位置编码；CogVLM、Cambrian、DeepStack则分别用token-specific专家、多塔局部聚合和多层残差注入增加深层视觉容量。
到2026年，模型差异已扩展为一整条计算链：视觉encoder是否保留稠密关系，connector如何分配token预算，fusion在哪些层发生，mask与position如何定义多图/时间可见性，decoder如何生成结构化输出，训练何时冻结/联合，RL如何避免坏verifier，runtime如何调用工具与并行Agent。只比较总参数或榜单分数会丢掉这些决定可复现性、成本和可靠性的结构差异。
**本文判断：**未来最重要的单位不是“一个更大的VLM”，而是一个可审计的视觉智能系统：它知道哪些patch值得保留，能把答案绑定到可校准视觉证据，在需要时调用mask/几何/搜索工具，按任务难度分配Thinking与Agent预算，并在同等像素、视觉token、输出token、工具和并发资源下报告真实收益。
