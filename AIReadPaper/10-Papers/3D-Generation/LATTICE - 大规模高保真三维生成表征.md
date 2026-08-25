---
type: paper-note
status: done
domain: 3D-Generation
paper: LATTICE: Democratize High-Fidelity 3D Generation at Scale
year: 2025
arxiv: 2512.03052
doi: null
source: https://arxiv.org/abs/2512.03052
project: https://lattice3d.github.io
code: null
tags: [3D-Generation, VoxSet, Voxel-Queries, Rectified-Flow, DiT, Test-Time-Scaling, Image-to-3D]
rating: 4.5
created: 2026-08-25
updated: 2026-08-25
---

# LATTICE：用 VoxSet 表征与可定位位置编码把三维生成扩到 4.5B

## 基本信息

- **论文全称**：LATTICE: Democratize High-Fidelity 3D Generation at Scale
- **作者机构**：Zeqiang Lai, Yunfei Zhao, Zibo Zhao, Haolin Liu, Qingxiang Lin, Jingwei Huang, Chunchao Guo, Xiangyu Yue（CUHK MMLab + Tencent Hunyuan）
- **会议/年份**：arXiv 预印本，2025-11-24 提交
- **arXiv**：[2512.03052](https://arxiv.org/abs/2512.03052)
- **项目页**：[https://lattice3d.github.io](https://lattice3d.github.io)
- **核心贡献**：提出 **VoxSet** 半结构化潜空间表征与 **Voxel Queries** 位置编码，让 3D 生成模型既能像 VecSet 那样高效训练，又能像 Sparse Voxel 那样获得强结构先验，支持从 0.6B 到 4.5B 的稳定 scaling，并具备「训练 6k token、推理 30k token」的可控 test-time 扩缩能力
- **领域**：3D-Generation（latent diffusion / 3D 形状表征 / 模型扩规模 / test-time scaling）

## 一句话结论

LATTICE 的核心判断是：**3D 生成的「表征-位置编码-scaling」三件套里，被低估的是「位置」**。VecSet 缺位置、Sparse Voxel 的位置又「太重」——VoxSet 通过「用粗体素中心替换表面点作为 query」+「RoPE 注入位置」，让潜空间既保留 VecSet 的全局压缩力，又获得稀疏体素的结构性，从而**首次让 3D 扩散模型在 0.6B→4.5B 范围内表现出单调可预测的 scaling 曲线**，并解锁了 token 级别的 test-time scaling。

![LATTICE 单图 3D 生成 teaser](assets/LATTICE%20-%20大规模高保真三维生成表征/figures/01-teaser.png)

*图 1（LATTICE 的 teaser：给定单张输入图，输出高保真三角网格；右下角放大展示细节区域的几何锐利度）。来源：LATTICE 论文 Fig.1 (p1)。*

---

## 1. 研究背景与问题定义

### 1.1 研究问题

3D 表征领域的两大流派各有问题：
- **VecSet（3DShape2VecSet / CLAY / Hunyuan3D-2 / TripoSG）**：靠 cross-attention 把 3D 形状压成 1D latent 集合，训练快、token 短，但**「latent 之间没有显式空间关系」**，扩规模时容易「记忆 > 抽象」，参数翻倍收益甚微。
- **Sparse Voxel（TRELLIS / XCube / SparseFlex / Sparc3D）**：把 latent 锚定在 active voxel 上，结构化强、细节好，但 **active voxel 数量在 64³ 分辨率上能到 20k+ token**，需要稀疏卷积 + 复杂 attention 系统，scalability 仍是开放问题。

作者从一个**生成视角**出发，抛出一个尖锐问题：**「什么才是真正适合 3D 扩散生成器的表征？」** 答案不是「local」也不是「structural」，而是 **「localizable」——位置信息在测试时能不能拿到」**。这是 LATTICE 全部方法设计的出发点。

### 1.2 现有方法的瓶颈

| 表征 | 优点 | 关键问题 |
| --- | --- | --- |
| **Global latent**（OccNet, DeepSDF） | 简单 | 容量小，无细节 |
| **Triplane**（Direct3D） | 2D 风格，硬件友好 | 三正交面有视角偏差 |
| **Sparse Voxel**（XCube, TRELLIS, SparseFlex） | 结构强、编辑好 | active voxel 数量大，需稀疏算子，scalability 不确定 |
| **VecSet**（3DShape2VecSet, CLAY, Hunyuan3D-2, TripoSG） | token 短、纯 Transformer | 缺乏显式位置，扩参数收益小 |

### 1.3 本文核心贡献

- **VoxSet 半结构化表征**：用一组锚定在「表面相交的体素中心」的 latent 向量，**继承 VecSet 的紧凑 + 引入 Sparse Voxel 的结构**。
- **Voxel Queries + RoPE**：把 3DShape2VecSet 的「表面点 query」换成「体素中心 query」，并对 DiT 注入 RoPE，让 latent 位置在测试时已知。
- **两阶段 coarse-to-fine 流程**：第一阶段用 off-the-shelf 3D 生成器（Hunyuan3D-2 / TRELLIS）出粗网格并 voxelize，第二阶段在已知 voxel 中心上做整流流精修。
- **稳定 scaling + test-time token scaling**：模型从 0.6B 扩到 4.5B，Normal-FID 单调下降；训练 6144 token，推理时可拉到 30720 token 持续提升。
- **低训练成本**：2B base 模型 64 GPU 24 小时可训练出显著超越前 SOTA。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：单张 RGB 图像（推理）；训练时为 Objaverse 系数据 + 过滤清洗。
- **输出**：高保真三角网格（无纹理，几何）。
- **潜空间**：VoxSet latent tokens，$T \in \mathbb{R}^{T \times C}$，$T$ 为 token 数（训练 1024-6144，推理可扩到 30720），每个 token 锚定在「表面相交的体素中心」上。
- **核心假设**：① 位置信息是生成 scaling 的关键先验；② coarse 网格可由任意 off-the-shelf 模型提供，voxelize 后即得 Voxel Queries；③ 训练 / 推理位置分布对齐是性能来源。

### 2.2 关键符号

| 符号 | 含义 |
| --- | --- |
| $P \in \mathbb{R}^{N \times 7}$ | 输入点云（3D 坐标 + 法向 + sharp-edge 标志） |
| $T$ | VoxSet token 数（可训练 / 推理调整） |
| $\epsilon \sim U(-\frac{1}{2R}, \frac{1}{2R})$ | Query Jitter 偏移（$R$ 为最小支持分辨率） |
| $\text{pos}_i$ | 第 $i$ 个 token 的体素中心位置 |
| $\text{RoPE}(\text{pos}_i)$ | 旋转位置编码 |
| $\mathcal{L}_{\text{flow}}$ | 整流流损失（linear coupling，来自 SiT） |
| $\mathcal{L}_{\text{vae}}$ | VAE 重建损失（CD + F-score） |

---

## 3. 方法总览

### 3.1 Localizable Code 原则

这是 LATTICE 全文的方法论基石。论文挑战两个常见认知：
1. **「Global vs Local」**：VecSet 全局、Sparse Voxel 局部——作者认为这只是对**压缩**有影响，对**生成**不重要。
2. **「Structural Sparse Grid vs Unstructural Set」**：Sparse Voxel 天然结构化、适合编辑——但 VecSet 也有结构，**只是测试时拿不到**。

结论：**「Localizable Code is All You Need」**——只要位置信息在测试时可用，生成就能 scaling。**Locality 是压缩 / 重建的事情，Localizability 是生成的事情。** 这一原则指导了后面所有设计。

### 3.2 VoxSet 表征

![VecSet / Sparse Voxel / VoxSet 三种 latent 表征 + 三种 query 类型](assets/LATTICE%20-%20大规模高保真三维生成表征/figures/04-representations.png)

*图 4（上：VecSet = 无序点集；Sparse Voxel = 锚定在 active voxel；VoxSet = 锚定在「表面相交的体素中心」，介于两者之间；下：三种 query 类型 — Learnable Embedding 全局 / Surface Point 局部但位置未知 / Voxel Center 局部且位置可获取）。来源：LATTICE 论文 Fig.4 (p4)。*

**VoxSet 怎么定义**：给一个 3D 资产，先 voxelize 到粗体素网格（如 64³），取**所有与表面相交的 active voxel 中心**作为 query 位置，再用 VecSet 风格的 cross-attention 编码器把这些中心 + 表面点云压成 latent tokens。每个 latent token 都有**显式位置**（体素中心坐标），且位置**在测试时已知**。

**为什么是「体素中心」而不是「表面点」**：
- 表面点在测试时是「被生成出来」的目标，**位置不可获取**——VecSet 永远没法在测试时用上位置信息。
- 体素中心是**先验固定**的网格点，位置始终可访问（只要粗网格在第一阶段预测出来）。
- 体素中心**与具体表面解耦**，减少了训练-测试分布 gap。

### 3.3 训练时的 Query Jitter

论文 §3.2 + 附录 B 引入一个关键技术——**Query Jitter**：在训练时给 query 加一个 $\epsilon \sim U(-\frac{1}{2R}, \frac{1}{2R})$ 的随机偏移，其中 $R$ 是最小支持分辨率。这样做的好处：

- **让 VAE 学会编码任意分辨率的 query**——本质上是一种随机 dropout。
- **让 DiT 也具备 test-time token scaling 的能力**——训练时见过的 query 数量是「模糊化」的，推理时给多少都不会跑出分布外。
- **跨分辨率泛化**——同一套权重可处理 64³ / 128³ / 256³ 任意网格。

**消融数据**（Tab.3）：在 baseline（固定 64 训练）上测 64/128 重建，CD 10.7/7.72；加 Fixed Train 后跨分辨率改善到 6.42/5.73/5.69；**加 Query Jitter 后进一步改善到 6.03/5.32/5.36**——Jitter 在 256 分辨率上比 Fixed Train 略差（5.36 vs 5.36 持平），但在其他分辨率上一致更优。

### 3.4 两阶段 Coarse-to-Fine 管线

![LATTICE 整体管线 + VoxSet VAE + VoxSet DiT 架构](assets/LATTICE%20-%20大规模高保真三维生成表征/figures/05-architecture.png)

*图 5（左：两阶段管线，Structure Generation Model 出粗网格 → Voxelize → Detailed Geometry Generation Model 出精细网格；中：VoxSet VAE，8 层 Cross Attention + 16 层 Self Attention，用 Voxel Queries 编码；右：VoxSet DiT，叠加 RoPE + DINOv2 图像条件，做 N 步去噪）。来源：LATTICE 论文 Fig.5 (p5)。*

**第一阶段 Structure Generation**：直接调用 off-the-shelf 3D 生成器（Hunyuan3D-2 / TRELLIS），出粗网格 → voxelize 到 $N^3$ 网格 → 取所有与表面相交的 active voxel 中心。**这一阶段的输出不是重点，重点是位置信息。**

**第二阶段 Detailed Geometry Generation**（LATTICE 主干）：
- **VoxSet VAE**：编码端 cross-attn 接收点云 + voxel queries，输出 VoxSet tokens；解码端 cross-attn 用 SDF 网格 queries 读回 latent tokens。**8 层 cross-attn + 16 层 self-attn 编码 + 对称解码**。
- **VoxSet DiT**：标准 Transformer Block，**对每个 noisy latent token 加 RoPE 位置编码**（基于体素中心位置），再做 DINOv2-Giant（1022 分辨率）图像 cross-attn。N 步去噪。

**关键简化**：两阶段只共享**体素中心位置**，第一阶段和第二阶段的模型**完全独立**——这意味着可以用任意 SOTA 3D 生成器当 backbone，论文验证了 Hunyuan3D-2 和 TRELLIS 都能用。

### 3.5 模型 + Token Scaling 行为

论文 §A.1 给出了**「VecSet 几乎不 scaling，VoxSet 持续 scaling」**的关键观察：

| 模型族 | 参数量 | 行为 |
| --- | --- | --- |
| **Hunyuan3D-2-mini / Hunyuan3D-2 / Hunyuan3D-2.1**（VecSet） | 0.6B / 1.1B / 3B | 三个结果「largely similar」，mini 反而略好——**VecSet 扩参数不受益** |
| **LATTICE 0.6B / 1.9B / 4.5B**（VoxSet） | 0.6B / 1.9B / 4.5B | 随参数增加，细节、平滑度、规律性**单调提升** |

**为什么？** 论文给出一个精彩解释：模型 scaling 的前提是**「condition 和 output 之间有清晰对应」**——否则大模型只是更会记忆而不是更会推理。VecSet 的 latent 没有显式位置，condition（图像）到 output（latent 集合）的对应是模糊的，所以扩参数收益有限；VoxSet 有位置锚点，condition 到空间位置的对应是明确的，所以扩参数直接换质量。**这是个很本质的观察，也解释了为什么同期 DIT/SD3/Flux 在 2D 上能 scale 起来——图像网格本身就给位置。**

**Test-time token scaling**：训练 6144 token 的模型，推理时 12288 / 24576 token 持续提升（Fig.6 / Fig.13）。VecSet 也有弱 test-time scaling（FlashVDM 报过），但 2N 之后基本饱和；VoxSet 4N 还在提升。**这是 VoxSet 因位置显式而能"学位置-内容对应"的结果**。

---

## 4. 实验与消融

### 4.1 训练配置

- **三种规模**：0.6B / 1.9B / 4.5B 参数。
- **多阶段 token scaling 训练**：先训 1024 token，逐步扩到 6144 token；不采用 CLAY 的 progressive model scaling（CLAY 是先训大模型 backbone 再训小 head）。
- **学习率**：每阶段内 constant + linear warm-up；阶段间从 1e-4 渐降到 1e-6。
- **批大小**：最大化 GPU 显存，单批最大 2048。
- **优化器**：DeepSpeed ZeRO（optimizer / gradient / parameter partitioning）。
- **目标**：SiT 风格的整流流 + linear coupling。
- **CFG**：10% 概率把 conditioning 替换为零嵌入（支持 classifier-free guidance）。
- **训练成本**：2B base 模型 64 GPU × 24 小时（论文 §3.2 强调的「低训练成本」卖点）。
- **图像条件**：DINOv2-Giant 1022 分辨率 patch，去掉 class token。
- **后训练**：① 高质量数据子集（15K）微调提升细节；② FlashVDM 加速 VAE 解码；③ Guidance / step distillation 减少采样步数。

### 4.2 评估协议

- **重建（Tab.1）**：CD（×10⁴）+ F-score@0.001（×10²），point-to-surface 距离，mesh 归一化到 [-1,1]。
- **生成（Tab.2）**：ULIP-T / ULIP-I（文本-网格、图像-网格余弦相似度）+ Uni3D-T / Uni3D-I。
- **基准集**：LATTICE-Bench(R)（自构，含 detailed assets，比 Dora 难）。
- **对比方法**：Michelangelo、Craftsman 1.5、TRELLIS、Hunyuan3D-2、Hi3DGen、Direct3D-s2、4 个商业模型（Model 1-4）。
- **用户研究**：500 题，每题 3 人独立排序；分类为 Overall / Subject / Scene。

### 4.3 主要实验结果

**Tab.1 重建性能**（×10⁴ CD ↓，×10² F1 ↑）

| 方法 | 网格分辨率 | Latent 规模 | CD | F1 |
| --- | --- | --- | --- | --- |
| Hunyuan3D-2 | N/A | 64×4096 | 12.35 | 82.78 |
| Hunyuan3D-2 | N/A | 64×8192 | 9.157 | 91.57 |
| SparseFlex | 512 | 8×48557 | 8.020 | 90.94 |
| SparseFlex | 1024 | 8×196028 | 2.972 | 97.76 |
| Direct3D-s2 | 1024 | 64×46592 | 4.987 | 97.46 |
| **LATTICE** | N/A | 64×4096 | **5.321** | **95.31** |
| **LATTICE** | N/A | 64×8192 | **2.909** | **98.53** |
| **LATTICE** | N/A | 64×20480 | **1.893** | **99.59** |

**LATTICE 64×4096 比 Hunyuan3D-2 64×4096 CD 降一半（5.32 vs 12.35），F1 提升 12.5 个百分点**；LATTICE 64×20480 接近 SparseFlex 1024 大 latent 的水平，但 latent 规模只有它的 1/10。

**Tab.2 生成性能**（LATTICE-1.9B，与同规模对比）

| 方法 | ULIP-T | ULIP-I | Uni-T | Uni-I |
| --- | --- | --- | --- | --- |
| Michelangelo | 0.075 | 0.115 | 0.213 | 0.261 |
| Craftsman 1.5 | 0.074 | 0.129 | 0.237 | 0.298 |
| TRELLIS | 0.076 | 0.126 | 0.249 | 0.311 |
| Hunyuan3D 2.0 | 0.077 | 0.130 | 0.251 | 0.315 |
| Hi3DGen | 0.066 | 0.112 | 0.246 | 0.299 |
| Direct3D-s2 | 0.074 | 0.122 | 0.247 | 0.314 |
| **LATTICE-1.9B** | **0.078** | **0.130** | **0.254** | **0.315** |

**LATTICE-1.9B 在 4 个指标上全部第一**（与 Hunyuan3D 2.0 在 ULIP-T、Uni-T、Uni-I 上持平，ULIP-I 上略胜）。

![LATTICE 与 5 个 SOTA 开源方法视觉对比](assets/LATTICE%20-%20大规模高保真三维生成表征/figures/08-comparison.png)

*图 8（LATTICE 与 Michelangelo / Craftsman 1.5 / TRELLIS / Hunyuan3D-2 / Direct3D-s2 的视觉对比：LATTICE 在细节锐利度、几何一致性与复杂结构上领先）。来源：LATTICE 论文 Fig.8 (p7)。*

**User Study（Fig.11）**：

| 对比 | Overall | Subject | Scene |
| --- | --- | --- | --- |
| LATTICE vs Model 1 | +23.1% | +26.1% | +58.6% |
| LATTICE vs Model 2 | +59.3% | +26.5% | +59.3% |
| LATTICE vs Model 3 | +29.0% | +19.6% | +57.8% |
| LATTICE vs Model 4 | +51.3% | +23.2% | +67.3% |

LATTICE 在所有 12 个对比组中均显著胜出，**Scene 类胜率最高（+57-67%）**，说明对复杂结构生成帮助最大。

![LATTICE 用户研究胜率（vs 4 个商业模型）](assets/LATTICE%20-%20大规模高保真三维生成表征/figures/11-user-study.png)

*图 11（用户研究：LATTICE 对比 4 个匿名商业模型，在 Overall / Subject / Scene 三类上的胜率，Scene 类优势最明显）。来源：LATTICE 论文 Fig.11 (p8)。*

### 4.4 消融

**Tab.3：VAE 训练策略对重建的影响**（4096 token + voxel query 测试）

| 训练 | 测试 64 | 128 | 256 |
| --- | --- | --- | --- |
| Baseline (point query train) | CD 10.7 / F1 85.3 | 7.72 / 91.4 | — |
| + Fixed Train | 6.42 / 92.9 | 5.73 / 94.5 | 5.69 / 94.7 |
| + Query Jitter | 6.03 / 93.7 | **5.32 / 95.3** | 5.36 / 95.3 |

**Jitter 在大多数分辨率下最优**；只有 256 上略逊 Fixed Train（5.36 vs 5.36 基本持平）。

**Fig.10：Voxel Query + VoxSet VAE 组合消融**（3 个 DiT，每个 200k 步）

| VAE | Query | 效果 |
| --- | --- | --- |
| Vecset VAE | Point Query | baseline（最多 artifact） |
| Vecset VAE | Voxel Query | 减少 artifact（domain gap 小） |
| **VoxSet VAE** | **Voxel Query** | **细节最丰富、伪影最少** |

**结论**：Voxel Query 和 VoxSet VAE **缺一不可**——Voxel Query 解决 train-test 位置 gap，VoxSet VAE 提供足够重建能力，两者乘积效应明显。

---

## 5. 局限与评价

### 5.1 方法论层面

- **两阶段管线引入上游误差**：LATTICE 第二阶段依赖第一阶段（Hunyuan3D-2 / TRELLIS）的输出，**如果上游粗网格质量差，VoxSet 也救不回来**。论文没详细分析「上游错误传播到下游的衰减率」。
- **位置先验的双刃剑**：VoxSet 的核心是「位置在测试时已知」，但**这种位置信息是粗网格的近似**——如果粗网格漏掉了薄结构或尖锐边，Voxel Queries 本身就没有这些位置，**细结构无法生成**。这是 LATTICE 适用于「粗结构 + 细节精修」、不适用于「极细几何创新」的根因。
- **DINOv2 图像条件没做细粒度优化**：论文用 1022 分辨率 DINOv2-Giant 取代 Hunyuan3D-2 的 518，对细节更友好；但**没用 CLIP 也没有跨模型融合**——图像语义一致性可能不如 TripoSG 的 CLIP+DINOv2 双条件。
- **未涉及纹理 / 材质生成**：LATTICE 只出几何，与 TRELLIS.2 / Hunyuan3D-2.1 的「PBR 材质 + 几何」联合建模还有差距。**需要额外接纹理管线**。
- **数据集细节不公开**：附录 B 只说「filter AI-generated, scanned, extreme complex scenes, plane-based assets」，**具体规则 / 阈值 / 人工校验环节未给出**。
- **训练细节有限**：没有公开 4.5B 模型的具体训练步数、wall-clock time、checkpoint 选择策略。

### 5.2 与 TripoSG / TRELLIS.2 / Hunyuan3D-2 的关系

- **vs TripoSG（2502.06608）**：TripoSG 走纯 VecSet 路线，扩规模受位置缺失限制；LATTICE 的 VoxSet 是**对 TripoSG 表征层面的改进**。但 LATTICE 整体系统（2B base + 24h）比 TripoSG（4B + 3 weeks）更轻量。
- **vs TRELLIS.2（2512.14692）**：TRELLIS.2 走 Sparse Voxel + O-Voxel 路线，结构强但 active voxel 数量大；LATTICE 用 VoxSet 走「轻量结构」路线，**两者在 SOTA 榜单上各有胜负**。
- **vs Hunyuan3D-2（2501.12202）**：直接对比的 baseline，也是 LATTICE 实验中的核心对照。LATTICE 同样规模下重建 CD 降一半以上，**可以说是 Hunyuan3D-2 的「VoxSet 升级版」**。
- **核心位置**：LATTICE 是 **2025 年末对「3D 生成到底要不要结构」这个问题给出最明确答案** 的工作——**要结构，但要用轻结构（Voxel Queries）而不是重结构（active voxel 全集）**。

### 5.3 复现建议

- **算力门槛**：2B base 24h × 64 GPU，4.5B 可能 ~1 周 × 64-128 GPU，相对 TripoSG 友好。
- **小规模复现路径**：用 LATTICE 公开的预训练 VAE（如果开放）+ 0.6B base DiT，64 GPU 24h 训练，对照 Tab.2 的 LATTICE-1.9B 应该能复现 70-80% 效果。
- **数据复现**：依赖上游 3D 生成器（Hunyuan3D-2 / TRELLIS）的输出，**需要先有这两个模型之一的访问权**。
- **代码**：项目页 [https://lattice3d.github.io](https://lattice3d.github.io) 提到 demo + 视频，**训练代码 / 权重是否公开需关注**。

### 5.4 个人判断（综合）

- **方法论贡献 ★★★★★**：把「localizable > locality」的判断升华为系统设计原则，并配套 VoxSet 表征 + Voxel Queries + Query Jitter + RoPE 一整套工程化方案，**对 3D 生成领域未来 1-2 年的表征设计影响很大**。这是 2025 年末最值得读的 3D 生成表征论文之一。
- **学术创新性 ★★★★**：VoxSet 是「已知组件的新组合」（VecSet + 体素位置 + RoPE），但**对组合的论证非常扎实**——附录 A.1 给出的「VecSet 不 scaling vs VoxSet scaling」观察极具说服力。
- **落地价值 ★★★★★**：2B base 24h 训练、test-time token scaling、上下游解耦，**对实际 3D 资产生成 pipeline 非常友好**。
- **与同期 SOTA 的差距**：在重建上 LATTICE 64×20480 接近 SparseFlex 1024（CD 1.89 vs 2.97）；在生成上 LATTICE-1.9B 在 ULIP/Uni3D 四个指标上全部第一。**但用户研究的对比方法（Model 1-4）未命名，胜率数据需要谨慎对待**。
- **核心 takeaway**：**3D 生成的「位置」和 2D 的「像素网格」是同等级别的设计先验**——VecSet 的 1D latent 集合是把 3D 强行压成 2D 图像的形状，丢掉了位置；VoxSet 找回了位置，但**只找回「测试时可获取的位置」**，不是全量结构。这是个优雅的折中。

---

## 6. 参考资源

- **论文 PDF**：[assets/sources/paper.pdf](assets/LATTICE%20-%20大规模高保真三维生成表征/sources/paper.pdf)（14 页，22.7 MB）
- **溯源表**：[assets/sources/sources.md](assets/LATTICE%20-%20大规模高保真三维生成表征/sources/sources.md)
- **项目页**：[https://lattice3d.github.io](https://lattice3d.github.io)
- **关键参考**（论文中引用的核心工作）：
  - 3DShape2VecSet（[2301.11445](https://arxiv.org/abs/2301.11445)）—— Latent Set 范式源头
  - XCube（[2404.15275](https://arxiv.org/abs/2404.15275)）—— Sparse Voxel hierarchy
  - TRELLIS（[2412.01506](https://arxiv.org/abs/2412.01506)）—— Structured 3D Latents (SLAT)
  - Hunyuan3D-2（[2501.12202](https://arxiv.org/abs/2501.12202)）—— 同体系 VecSet 路线
  - TripoSG（[2502.06608](https://arxiv.org/abs/2502.06608)）—— 同期 Rectified Flow 工作
  - CLAY（[2404.12297](https://arxiv.org/abs/2404.12297)）—— progressive model scaling
  - FlashVDM（[2503.02182](https://arxiv.org/abs/2503.02182)）—— VAE 解码加速 + test-time scaling 早期工作
  - DiT（[2212.09748](https://arxiv.org/abs/2212.09748)）—— DiT backbone
  - SiT（[2401.08740](https://arxiv.org/abs/2401.08740)）—— Scalable Interpolant Transformer
  - DINOv2（[2304.07193](https://arxiv.org/abs/2304.07193)）—— 图像条件

## 7. 关键图源

| 图编号 | 描述 | 文件 | PDF 页 |
| --- | --- | --- | --- |
| Fig.1 | 单图 3D 生成 teaser | `figures/01-teaser.png` | p1 |
| Fig.4 | VecSet/Sparse Voxel/VoxSet 三种表征 + 三种 query 类型 | `figures/04-representations.png` | p4 |
| Fig.5 | LATTICE 整体管线 + VoxSet VAE + VoxSet DiT | `figures/05-architecture.png` | p5 |
| Fig.8 | 与 5 个 SOTA 开源方法视觉对比 | `figures/08-comparison.png` | p7 |
| Fig.11 | 用户研究胜率（vs 4 个商业模型） | `figures/11-user-study.png` | p8 |

## 8. 元信息

- **初版时间**：2026-08-25
- **精读耗时**：约 2 小时
- **精读重点**：VoxSet 表征定义、Voxel Queries + RoPE 设计、Query Jitter、附录 A.1 的 scaling 行为分析
- **可改进项**：① 笔记未深入展开 §C post-training 的 high-quality finetuning 与 step distillation 细节（依赖 FlashVDM）；② 4.5B 模型的具体训练配置未完全公开
