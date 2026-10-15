---
type: paper-note
status: complete
domain: Video-Generation
paper: "Goku: Flow Based Video Generative Foundation Models"
year: 2025
arxiv: "2502.04896"
doi: "10.48550/arXiv.2502.04896"
source: "https://arxiv.org/abs/2502.04896"
project: "https://saiyan-world.github.io/goku/"
code: null
tags:
  - Video-Generation
  - Image-Video-Joint-Training
  - Rectified-Flow
  - DiT
  - 3D-VAE
created: 2026-10-15
updated: 2026-10-15
---

# Goku：联合图像视频整流流基础模型

## 基本信息

- **论文**：Goku: Flow Based Video Generative Foundation Models
- **作者**：Shoufa Chen、Chongjian Ge、Yuqi Zhang 等
- **机构**：香港大学、字节跳动
- **会议**：CVPR 2025
- **模型规模**：2B、8B；1B 仅用于 RF/DDPM 先导实验
- **项目页**：https://saiyan-world.github.io/goku/

## 一句话结论

Goku 不是靠单个“神奇模块”取胜，而是把联合 3D VAE、全时空注意力 Transformer、Rectified Flow、160M 图像+36M 视频的数据治理、三阶段渐进训练和超长序列分布式系统整合成工业级 pipeline；它以 VBench 84.85 和 GenEval 0.76 展示均衡能力，但主要消融偏定性、内部数据和训练成本不透明，论文的可复现价值更多在系统设计原则而非完整配方。

---

## 1. 研究背景与问题定义

### 1.1 联合图像—视频生成的矛盾

图像数据数量大、质量高，适合学习对象、构图和纹理；视频数据稀缺、噪声大，却提供运动和时间一致性。分开训练两个模型会浪费共享的视觉知识，直接混合又面临 token 长度、分辨率、时长、数据质量和模态不平衡。

Goku 的目标不是只刷一个 T2V 指标，而是建立一套统一基础模型：同一 latent space、同一 Transformer、同一 flow 目标，既生成图像也生成视频，并能轻量适配 I2V。

![](assets/Goku%20-%20联合图像视频整流流基础模型/figures/01-t2v-samples.png)

> 图 1(b)：Goku-T2V 的视频帧样例，展示复杂主体、运动与时间一致性。来源：论文 Figure 1(b)，p.2，https://arxiv.org/abs/2502.04896

### 1.2 论文的四个支柱

1. **表示**：图像作为$T=1$的视频，统一进入 3D joint VAE。
2. **骨干**：不拆 spatial/temporal attention，直接 full attention。
3. **生成目标**：用 Rectified Flow 替代 DDPM，提高收敛效率。
4. **工程与数据**：Patch n' Pack、3D RoPE、QK norm、并行训练和细粒度数据治理。

---

## 2. 任务定义与输入输出

图像或视频统一记为$x\in\mathbb R^{T\times H\times W\times3}$；图像时$T=1$。3D VAE 编码为时空 latent，视频在$H,W,T$上分别压缩$8,8,4$倍，图像空间压缩 8 倍。文本由 Flan-T5 编码，通过 cross-attention 条件化 Transformer。

模型支持：

- Text-to-Image；
- Text-to-Video；
- 从 T2V 权重适配的 Image-to-Video。

---

## 3. 核心方法

### 3.1 3D Image-Video Joint VAE

统一 VAE 让图像与视频落在同一 latent 坐标系，图像提供高质量空间先验，视频提供时间动力学。论文没有披露 VAE 层数、损失配方和重建指标，这是复现中最大的缺口之一。

### 3.2 Full-Attention Transformer

传统视频模型常交替空间注意力与时间注意力，以降低成本；Goku 对所有图像/视频 token 做 plain full attention，避免空间—时间分解限制复杂运动建模。每个 block 包括 self-attention、文本 cross-attention、FFN 与 adaLN-Zero。

| 模型 | 层数 | 维度 | FFN 维度 | Heads |
|---|---:|---:|---:|---:|
| Goku-1B | 28 | 1152 | 4608 | 16 |
| Goku-2B | 28 | 1792 | 7168 | 28 |
| Goku-8B | 40 | 3072 | 12288 | 48 |

Full attention 的代价是序列可超过 220K token，需要 FlashAttention、sequence parallelism 和 activation checkpointing 才能训练。

### 3.3 Patch n' Pack 与 3D RoPE

Patch n' Pack 将不同宽高比、分辨率与视频长度的样本沿序列维打包，不再为每种 shape 建 bucket，提高 token 利用率。3D RoPE 分别编码时间、高度和宽度，支持分辨率/时长变化，并比正弦位置编码在阶段切换时收敛更快。

### 3.4 Q-K Normalization

大模型训练中 attention logits 可能爆炸，出现 loss spike 或 checkpoint 崩坏。Goku 在 attention 前对 query/key 做 RMSNorm：

$$Q\leftarrow\operatorname{RMSNorm}(Q),\qquad K\leftarrow\operatorname{RMSNorm}(K)$$

论文把它作为大规模稳定训练的必要工程组件，但没有提供独立数值消融。

### 3.5 Rectified Flow

从噪声$x_0\sim\mathcal N(0,I)$到真实 latent$x_1\sim p_{\mathrm{data}}$采用线性路径：

$$x_t=tx_1+(1-t)x_0$$

目标速度为：

$$v_t=\frac{dx_t}{dt}=x_1-x_0$$

Transformer 学习条件速度场，推理通过 ODE 积分从噪声输运到数据。RF 的核心优势不是“无步采样”，而是路径简单、训练目标直接、收敛更快。

1B ImageNet 先导实验中，RF 400K steps 的 FID 2.1572，优于 DDPM 1000K 的 2.2568；但 RF 的 precision、sFID、IS 并非全部领先，因此证据支持的是更快 FID 收敛，不是所有维度全面优越。

### 3.6 数据治理

![](assets/Goku%20-%20联合图像视频整流流基础模型/figures/02-data-pipeline.png)

> 图 2：数据管线从采集、切片、审美/OCR/背景/DINO 相似度/NSFW 过滤，到标签、运动分数、caption 和语义重平衡。来源：论文 Figure 2，p.8。

最终训练池约 160M 图文对（100M LAION+60M 内部）和 36M 视频文本对（11M 公开+25M 内部）。视频先满足时长≥4s、短边≥480、码率≥500kbps、帧率≥24/23.976，再用 PySceneDetect+DINOv2 做镜头切分，最长 clip 10s。

视频过滤随分辨率加严：

| 阶段 | 数据量 | 最低分辨率 | DINO 相似度 | 审美分 | OCR | Motion |
|---|---:|---:|---:|---:|---:|---:|
| 480p | 36M | 480×864 | ≥0.85 | ≥4.3 | ≤0.02 | 0.3–20 |
| 720p | 24M | 720×1280 | ≥0.90 | ≥4.5 | ≤0.01 | 0.5–15 |
| 1080p | 7M | 1080×1920 | ≥0.90 | ≥4.5 | ≤0.01 | 0.5–8 |

Caption 管线用 InternVL2.0 描述关键帧、Tarsier2 描述整段视频和相机运动，再由 Qwen2 合并，并把 RAFT motion score 写入文本。语义标签分 9 个大类、86 个子类，通过下采样、过采样和人工数据补长尾，人物数据被特意加权。

### 3.7 三阶段训练

1. **Text-Semantic Pairing**：先做 T2I，建立文本—对象—场景映射。
2. **Image-and-Video Joint Learning**：混合图像和视频，分辨率从 288×512→480×864→720×1280。
3. **Modality-Specific Finetuning**：T2I 优化审美和细节，T2V 优化时间平滑与运动一致性。

I2V 从 T2V 权重出发，把第一帧 token 沿时间广播，与带噪视频 token 在通道维拼接，仅新增 MLP 对齐；使用约 4.5M text-image-video triplets 微调 10K steps。

### 3.8 分布式训练基础设施

- **Ulysses sequence parallelism**：沿序列切分，用 all-to-all 分发 Q/K/V 和 attention heads。
- **FSDP HYBRID_SHARD**：分片参数、梯度和 optimizer state，同时保留跨 shard group 数据并行。
- **Fine-grained activation checkpointing**：选择性保存激活以平衡重算与通信。
- **MegaScale 容错与 ByteCheckpoint**：支持数千卡 checkpoint；8B 保存阻塞低于 4 秒。

这些工程并非附属信息：没有它们，220K-token full attention 模型在实践中无法训练。

---

## 4. 实验设置

T2I 使用 GenEval、T2I-CompBench、DPG-Bench；T2V 使用 UCF-101 FVD/IS 与 VBench 16 维指标。UCF-101 原始只有类别标签，论文用 Tarsier-34B 生成详细 caption，再据此生成 13,320 段视频，因此与仅使用类名的 baseline 并非完全同口径。

---

## 5. 实验结果

### 5.1 Text-to-Image

| 模型 | GenEval | DPG-Bench |
|---|---:|---:|
| SDXL | 0.55 | 74.65 |
| DALL-E 3（rewrite） | 0.67 | 83.50 |
| SD3 | 0.74 | — |
| Goku-2B | 0.70 | **83.65** |
| Goku-2B（rewrite） | **0.76** | — |

Goku 0.76 依赖 prompt rewriting；原始 prompt 下 0.70 低于 SD3 0.74。DPG-Bench 83.65 只比 DALL-E 3 高 0.15。T2I-CompBench 的 Color/Shape/Texture 仍落后 DALL-E 3。结论应是“综合竞争力强”，不是所有维度第一。

### 5.2 UCF-101

| 分辨率 | FVD↓ | IS↑ |
|---|---:|---:|
| 256×256 | 246.17 | 45.77 |
| 240×360 | 254.47 | **46.64** |
| 128×128 | **217.24** | 42.30 |

最低 FVD 出现在 128×128；最高 IS 在 240×360，但仍低于 PYOCO 47.76。不同分辨率结果不应混成“同一设置全面 SOTA”。

### 5.3 VBench

| 模型 | Quality | Semantic | Overall |
|---|---:|---:|---:|
| CausVid | **85.65** | 78.75 | 84.27 |
| Luma | 83.47 | **84.17** | 83.61 |
| HunyuanVideo | 85.09 | 75.82 | 83.24 |
| **Goku** | 85.60 | 81.87 | **84.85** |

Goku 的优势是均衡：Quality 只差 CausVid 0.05，Semantic 低于 Luma，Dynamic Degree 也不及 CausVid，但总分最高。美学 67.22、成像质量 71.29、空间关系 85.72 是突出项。

### 5.4 模型扩展与联合训练消融

![](assets/Goku%20-%20联合图像视频整流流基础模型/figures/03-model-scaling.png)

> 图 5(a)：8B 相比 2B 更少出现错误肢体、轮子和物体结构，说明容量扩展改善复杂结构。来源：论文 Figure 5(a)，p.16。

![](assets/Goku%20-%20联合图像视频整流流基础模型/figures/04-joint-training.png)

> 图 5(b)：相同 T2I checkpoint、相同视频微调步数下，加入图像—视频联合训练能显著改善视频帧细节与照片感。来源：论文 Figure 5(b)，p.16。

两项都是定性图，没有 FVD、VBench 或人类偏好数值。它们支持方向性判断，却不足以精确量化“扩模型”和“加图像数据”分别贡献多少。

---

## 6. 与视频生成主线的关系

Goku 处在视频基础模型架构收敛的关键节点：3D VAE+DiT+flow matching+3D RoPE+联合图像视频训练，后来成为大模型常见配方。它与 [[Wan2.1 - 开源大规模视频生成基础模型]]、HunyuanVideo 的区别更多在数据、full attention 和系统工程取舍，而非完全不同的生成范式。

与 LlamaGen/VAR/Infinity 的离散 AR 路线相比，Goku 回到连续 latent 和 flow matching；它强调图像与视频统一，而不是视觉 token 的离散词表扩展。

---

## 7. 局限与批判性评价

### 7.1 可复现性弱

25M 内部视频、60M 内部图像、内部 OCR/分类器和具体数据配比均不可得；3D VAE 细节也未披露。外部最多复现架构思想，无法复刻训练分布。

### 7.2 缺少训练和推理成本

论文没有给出总 GPU-hours、GPU 型号/数量、采样步数、单视频延迟、显存和 2B/8B 成本差。基础设施写得很细，最终资源账单却缺失。

### 7.3 Full attention 的代价未被正面对比

作者主张 full attention 更能建模时空关系，但没有与分解式 attention 做同算力对照。超长序列靠系统工程托住，不等于算法复杂度被解决。

### 7.4 I2V 和长视频证据不足

I2V 只有定性样例；主要视频 4 秒、24 FPS、720p。没有分钟级一致性、叙事多镜头或长程身份保持评测。

### 7.5 评测协议不完全统一

GenEval 使用 prompt rewriting，UCF-101 使用 Tarsier-34B 丰富 caption；商业模型版本和采样参数也不完全透明。应把 84.85/0.76 视为论文协议下结果，而非无条件排名。

### 7.6 未单列 limitations/future work

合理后续包括稀疏/层级 attention、分钟级生成、标准化 I2V 评测、完整速度和成本披露，以及版权、隐私和深伪安全治理；这些是本文缺口推导，不是作者明确承诺。

---

## 8. 复现与实践建议

最小复现可先做 1B ImageNet RF vs DDPM，验证 flow 收敛；然后用公开视频构建 3D VAE+full attention 小模型，重点实现 Patch n' Pack、3D RoPE、QK norm。完整 2B/8B 训练需要大规模视频治理与分布式系统，不现实。

工程上应优先监控：不同分辨率 token packing 的 padding 利用率；full attention 长序列显存；QK norm 对 loss spike；图像/视频 batch 比例；以及阶段切换时 RoPE 外推和 VAE 分布漂移。

---

## 9. 个人判断

**★★★★☆。** Goku 是一篇强系统报告，价值在于把“工业级联合图像视频生成到底需要哪些环节”讲完整，并通过 RF 先导实验和主 benchmark 证明配方有效。缺点也典型：真正决定上限的数据和算力不可复现，架构模块消融不足，很多结论依赖定性图。它更适合作为搭建视频基础模型的工程蓝图，而不是单点算法创新范本。

---

## 10. 材料来源

- [arXiv](https://arxiv.org/abs/2502.04896)
- [项目页](https://saiyan-world.github.io/goku/)
- [本地来源清单](assets/Goku%20-%20联合图像视频整流流基础模型/sources.md)
