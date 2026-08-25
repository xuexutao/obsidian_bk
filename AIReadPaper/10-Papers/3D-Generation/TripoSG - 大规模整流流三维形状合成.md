---
type: paper-note
status: done
domain: 3D-Generation
paper: TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models
year: 2025
arxiv: 2502.06608
doi: null
source: https://arxiv.org/abs/2502.06608
project: https://www.tripo3d.ai/work/triposg
code: null
tags: [3D-Generation, Rectified-Flow, MoE, SDF, VAE, VecSet, Data-Building, Image-to-3D]
rating: 4.5
created: 2026-08-25
updated: 2026-08-25
---

# TripoSG：用大规模整流流 Transformer 重建高保真三维形状

## 基本信息

- **论文全称**：TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models
- **作者机构**：Yangguang Li, Zhuoer Cai, Jiahao Zhu, Qihan Ren, Donglin Di, Mingyu Zhang, Feng Liu, Xiaofeng Wang, Chunhua Shen, Liang Dong, Lu Qi, Tao Yu, Xin Yang, Yifan Peng, Marcelo H. Ang Jr, Shi-Min Hu, Dahua Lin（VAST AI Research、CUHK、UT Austin、Shanghai AI Lab、Tsinghua、NTU 等）
- **会议/年份**：ICLR 2025
- **arXiv**：[2502.06608](https://arxiv.org/abs/2502.06608)（2025-02-10 提交）
- **项目页**：[https://www.tripo3d.ai/work/triposg](https://www.tripo3d.ai/work/triposg)
- **核心贡献**：把 3DShape2VecSet 的 latent set 范式 + 整流流（Rectified Flow）+ 数据清洗管线 + MoE 扩规模拼成一套端到端的 image-to-3D SOTA 系统，VAE 监督创新地融合 SDF + 表面法向 + Eikonal
- **领域**：3D-Generation（latent diffusion / 整流流 / MoE / 神经 SDF / 数据工程）

## 一句话结论

TripoSG 的核心判断是：**3D 生成的「数据-表征-采样」三轴都已齐备，关键缺口是数据质量和生成模型扩规模**。为此它在 3DShape2VecSet 的 latent set 骨架上，把 ① VAE 表征从 occupancy 换成 SDF+法向+Eikonal 的混合监督，② 生成器从 DDPM/EDM 换成 Rectified Flow + 跳连 + MoE 的 1.5B→4B 扩规模方案，③ 用一条四阶段数据管线把 Objaverse-XL 10M 清洗到 2M 高质量样本——三件事一起做，让单图 3D 在 Normal-FID 上从 ~10 一路压到 3.36，刷新 2025 年初的 image-to-3D SOTA。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

在 2025 年 2 月之前，image-to-3D 主要走两条路：①「2D 提升」—— DreamFusion 系（SDS）或多视角扩散 + LRM 重建，依赖 2D 先验，单次优化几分钟到几小时；② 「3D 重建」—— LRM/InstantMesh/CRM/MeshLRM 等，把多视图直接回归到 Triplane / NeRF / 网格，属于「重建」而非「生成」，质量天花板受多视图一致性限制。第三条路——直接在 3D 潜空间训练扩散/流模型——长期受困于两点：**没有大规模高质量 3D 数据**，以及**没有验证「2D 时代的扩规模经验能直接迁移到 3D」**。TripoSG 的切入点是把这第三条路真正跑通。

### 1.2 现有方法的瓶颈

- **SDS 系**（DreamFusion、ProlificDreamer、LucidDreamer、Fantasia3D）：单次优化慢、几何精度差、对 prompt 敏感；难以端到端 scale。
- **LRM 系**（LRM、TripoSR、InstantMesh、MeshLRM、CRM、Unique3D、Era3D）：本质是重建，多视图不一致时未观察区域会糊；上限被多视图扩散锁死。
- **早期 3D 扩散**（Point-Voxel Diffusion、LION、3DShape2VecSet、Michelangelo、CLAY、CraftsMan、Direct3D）：多在 ShapeNet 训练，几何差；2024 年开始尝试大规模数据，但几何细节与图像一致性仍不够。
- **关键空白**：① 大规模整流流尚未在 3D 上跑通；② occupancy 表征的离散化带来 aliasing 和细节损失；③ CLIP 全局特征 + 拼接式注入缺乏细粒度控制。

### 1.3 本文核心贡献

- **大规模整流流 Transformer**：把 Rectified Flow 接入 3D latent set 训练，1.5B base + MoE 扩到 4B，在 latent 长度上从 512 → 2048 → 4096 三阶段渐进训练。
- **混合监督的神经 SDF VAE**：抛弃 occupancy 的 BCE 损失，改用 SDF L1+L2 + 表面法向余弦损失 + Eikonal 正则 + KL 的四件套；几何细节与抗 aliasing 显著提升。
- **2M 量级数据构建系统**：四阶段管线（scoring → filtering → fixing → field production），把 Objaverse-XL 10M 数据清洗到 2M 高质量 Image-SDF 配对样本，并做了消融证明「质量先于数量，但二者缺一不可」。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：单张 RGB 图像（推理）；训练时为 Objaverse-XL 中渲染的多视角 RGB（8 个前方位姿，焦距 / 俯仰 / 方位角随机）+ 对应的 watertight 网格 + 体素 UDF 场。
- **输出**：无纹理三角网格（推理），后续用 Meta 3D TextureGen 风格的多视图纹理 pipeline 上色。
- **潜空间**：3DShape2VecSet 风格的 latent set $X \in \mathbb{R}^{L \times C}$，$L \in \{512, 2048, 4096\}$，$C = 64$，由 VAE 编码表面点云得到。
- **核心假设**：① 神经 SDF 隐式场为统一几何监督；② VAE 训练不加位置编码，可外推更高 latent 长度；③ 4B 模型仍跑得起 Rectified Flow 的少量采样步（logit-normal + resolution-dependent shift）。

### 2.2 关键符号

| 符号 | 含义 |
| --- | --- |
| $X \in \mathbb{R}^{L \times C}$ | VAE 编码得到的 latent set |
| $t \in [0,1]$ | 整流流时间步 |
| $I_{\text{global}}, I_{\text{local}}$ | CLIP-ViT-L/14 全局特征、DINOv2-Large 局部特征 |
| $s, \hat{s} \in \mathbb{R}$ | 预测 / 真值（截断）SDF 值 |
| $\hat{n} \in \mathbb{R}^3$ | 表面法向真值 |
| $\mathcal{L}_{\text{vae}}$ | VAE 联合监督损失 |
| $\mathcal{L}_{\text{sdf}}, \mathcal{L}_{\text{sn}}, \mathcal{L}_{\text{eik}}, \mathcal{L}_{\text{kl}}$ | SDF、法向、Eikonal、KL 四件套 |
| $m, s$ | logit-normal 采样的偏置 / 宽度参数 |
| $p_{\text{router}}$ | MoE router 的 top-K 概率 |

---

## 3. 方法总览

### 3.1 整流流 Transformer 架构

TripoSG 整流流 backbone 借鉴 DiT（Peebles & Xie, 2023）和 3DShape2VecSet，**关键创新是把 U-Net 风格的长跳连搬进 Transformer**——把 backbone 拆成 N 个 encoder block + 1 个 middle block + N 个 decoder block，encoder 第 $i$ 个 block 的输出跳连到 decoder 第 $N-i$ 个 block。$N=10$，共 $2N+1=21$ 个 block，$W=2048$，16 头注意力，base 约 1.5B 参数。decoder block 的跳连公式：

$$
Z_{DB}^{(N-i)} = DB_{N-i}\!\left(Z_{DB}^{(N-i-1)}\right) + EB_i\!\left(Z_{EB}^{(i-1)}\right),\quad i \in \{0, 1, \dots, N\}
$$

**为什么不沿用 DiT 的逐层堆叠？** 论文做了消融（Tab.1 第 2 vs 5 行，Normal-FID 10.61→9.47），证明浅层与深层特征直接融合对 Normal-FID 有明显收益；直觉上，浅层学的是局部几何先验，深层学的是语义组合，跳连让信息流通更顺畅。

**时序与图像条件注入**：与 Michelangelo（把 timestep 和 latent 拼接）和 CLAY（cross-attention 注入图像）不同，TripoSG 在**每一个 block**都做双 cross-attention：
- $I_{\text{global}}$：CLIP-ViT-L/14 提取的全局语义特征（控制整体身份）。
- $I_{\text{local}}$：DINOv2-Large 提取的 patch-level 局部特征（控制细节对齐）。

每个 block 的计算流程（公式 2-6）：

$$
Z = \text{Concat}(X, t)
$$
$$
Z = Z + \text{SelfAttn}(\text{Norm}(Z))
$$
$$
Z = Z + \text{CrossAttn}(\text{Norm}(Z), I_{\text{local}}) + \text{CrossAttn}(\text{Norm}(Z), I_{\text{global}})
$$
$$
Z = Z + \text{FFN}(\text{Norm}(Z))
$$

**为什么不用文本 caption？** 论文明确反对：渲染图像的 caption 受光照、阴影、纹理干扰；用 GPT-4V 给 3D 写 caption 又会引入语义噪声。CLIP+DINOv2 双特征直接来自图像本身，**收敛更快、细粒度对齐更好**。

![TripoSG 整流流架构 + 块内结构 + MoE 细节](assets/figures/03-architecture.png)

*图 3（左：整流 backbone + MoE 块；中：每个块的内部组件，含 self-attn + 双 cross-attn；右：MoE 内部，含 1 shared expert + N 普通 expert + Router）。来源：TripoSG 论文 Fig.3 (p5)。*

### 3.2 整流流采样与训练策略

**为什么选 Rectified Flow 而不是 DDPM / EDM？** 论文 Tab.1 第 3-5 行给出明确消融（同一 backbone，分别训练 DDPM、EDM、R-Flow，Normal-FID 9.63 → 9.50 → 9.47）：

| 采样方式 | 轨迹 | Normal-FID |
| --- | --- | --- |
| DDPM（公式 7）$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ | 复杂曲线 | 9.63 |
| EDM（公式 8）$x_t = x_0 + \sigma(t)\epsilon$ | 平滑曲线 | 9.50 |
| **Rectified Flow（公式 9）$x_t = t x_0 + (1-t)\epsilon$** | **直线** | **9.47** |

整流流最简形式为 $x_t = t x_0 + (1-t)\epsilon$，轨迹是线性 ODE，训练目标最简单、数值最稳定、推理步数少。配套策略有两个：

1. **logit-normal 采样**（公式 10），把 $t$ 的采样权重偏向 0.5 附近，因为 Rectified Flow 在中段最难学：
$$
\pi_{\ln}(t; m, s) = \frac{1}{s\sqrt{2\pi}\,t(1-t)} \exp\!\left(-\frac{(\log(t/(1-t)) - m)^2}{2s^2}\right)
$$
2. **Resolution-Dependent Timestep Shifting**（公式 11），把训练 / 推理的 $t_n$ 按 latent 长度的平方根比重新映射，让高分辨率下噪声水平与小分辨率对齐，保持同样的「信号-噪声比」。

### 3.3 3D VAE：SDF + 法向 + Eikonal 的混合监督

**为什么放弃 occupancy 改用 SDF？** occupancy 的本质是二分类（BCE loss），训练稳定但几何表达力弱：重建网格会有明显 aliasing、需要后处理平滑。SDF 是连续距离场，天然适合 Marching Cubes，细节更锐利。但**单独 SDF 损失不够**——论文发现加上**表面法向余弦损失**和 **Eikonal 正则**，可以让细节从 SDF 值域迁移到「梯度域」学习（公式 17-20）：

$$
\mathcal{L}_{\text{vae}} = \mathcal{L}_{\text{sdf}} + \lambda_{\text{sn}}\mathcal{L}_{\text{sn}} + \lambda_{\text{eik}}\mathcal{L}_{\text{eik}} + \lambda_{\text{kl}}\mathcal{L}_{\text{kl}}
$$

$$
\mathcal{L}_{\text{sdf}} = |s - \hat{s}| + \|s - \hat{s}\|_2^2
$$

$$
\mathcal{L}_{\text{sn}} = \left(1 - \left\langle \frac{\nabla D(x, f)}{\|\nabla D(x, f)\|}, \hat{n} \right\rangle\right)
$$

$$
\mathcal{L}_{\text{eik}} = \|\nabla D(x, f) - 1\|_2^2
$$

注意 $\mathcal{L}_{\text{sn}}$ 只在**表面点**采样（不是近表面，也不是体素点），梯度效率最高；Eikonal 项保证 $\nabla D$ 范数接近 1，避免隐式场退化。VAE 实验消融（Tab.3）显示加上法向和 Eikonal 后，F-Score 0.999、Normal Consistency 0.957、Chamfer 4.57，**比 occupancy baseline 提升明显**。

**VAE 网络结构**沿用 3DShape2VecSet 的 latent set 范式（公式 21-24），但有几个增强：

- **输入更密**：每模型 20,480 个表面点（远超 3DShape2VecSet 的 2048）；VAE 的目的不是稀疏重建，而是尽可能多地编码几何信息。
- **多分辨率共享权重**：训练时用 $M \in \{512, 2048\}$ token 两种长度，**权重共享**；推理时直接外推到 4096 token（论文验证过无损），避免重训。
- **8 层 encoder + 16 层 decoder**：decoder 更深，把几何解码能力做厚，**不增加 flow 阶段的推理成本**（decoder 只训一次）。

![TripoSG VAE 编码器-解码器 + 表面法向引导](assets/figures/04-vae-architecture.png)

*图 4（上：encoder 用 downsample 后的表面点做 cross-attn；下：decoder 用 query point 跨注意力读 latent，再用 Marching Cubes 抽面；中间红框：表面法向 + Eikonal 的梯度域监督）。来源：TripoSG 论文 Fig.4 (p7)。*

### 3.4 Occupancy vs SDF 的可视化对比

![Occupancy 与 SDF 重建质量对比](assets/figures/05-occupancy-vs-sdf.png)

*图 5（同一组输入，occupancy 重建 vs SDF 重建：occupancy 在边缘、平面、纹理合成上都有可见 aliasing 与 floaters；SDF 重建无 aliasing，且为后续多视图纹理生成提供更稳的几何）。来源：TripoSG 论文 Fig.5 (p7)。*

### 3.5 MoE 扩规模

**为什么是 MoE 而不是直接 dense 扩到 4B？** dense 扩参数会把计算量与推理延迟都拉满，MoE 走「**参数增长、激活量几乎不变**」的路子，部署友好。

设计细节：
- **8 个 expert + 1 shared expert**（每个 MoE 块），Router 取 top-2 激活，外加一个跨所有 token 共享的 expert。公式 12-13 描述了 MoE 路径。
- **只对最后 6 层 decoder 加 MoE**：浅层学通用几何，扩参收益低；深层学特定细节，扩参最有效。
- **从 base 初始化**：每个 expert 的 FFN 权重从 base 的单一 FFN 复制而来，而不是从零训；这样避免训 MoE 的不稳定窗口。
- **加权激活 + 辅助损失**：top-2 expert 用 router 给的概率加权求和，加 auxiliary loss 平衡 expert 负载，**避免个别 expert 啥也不学**。

最终模型从 1.5B 涨到 ~4B，但**推理 FLOPs 几乎和 1.5B 一样**（只激活 2 expert + 1 shared）。

### 3.6 数据构建系统（DBS）

Objaverse(-XL) 的 10M 模型大多是「互联网野生数据」，质量参差不齐。TripoSG 设计了四阶段清洗管线：

![TripoSG 数据构建系统四阶段管线](assets/figures/06-data-building-system.png)

*图 6（I 评分：从渲染 normal map 预测 1-5 分；II 过滤：剔除大平面底座 / 多物体 / 动画错误；III 修复 + 增强：角色朝前归位、无纹理模型用 ControlNet++ 上色；IV 场数据生成：UDF 512³ grid + Marching Cubes 抽 watertight + PySDF 采样 SDF 监督）。来源：TripoSG 论文 Fig.6 (p9)。*

| 阶段 | 关键做法 | 输入 → 输出 |
| --- | --- | --- |
| I 数据评分 | 10K 人工标注 normal map → 线性回归（CLIP+DINOv2 特征）训打分器 | 10M → 按分排序 |
| II 数据过滤 | 大平面底座 / 多物体 / 动画错误三类规则 | 剔除约 78%（800K → 180K） |
| III 修复 + 增强 | 24 朝向 DINOv2 特征训朝前预测器；无纹理模型用 ControlNet++ 上色 | 统一朝向 + 准备 image condition |
| IV 场数据生成 | 非水密网格 → 512³ UDF + Marching Cubes 抽水密面 → PySDF 采样 surface / near-surface / volume 点 | mesh → SDF 监督对 |

**Tab.4 给出的关键消融（Normal-FID 越低越好）**：

| 数据集 | 规模 | DBS | Normal-FID |
| --- | --- | --- | --- |
| Objaverse 原始 | 800K | ✗ | 11.61 |
| Objaverse 过滤 | 180K | ✓ | 9.47 |
| TripoSG | 2M | ✓ | 5.81 |

**结论**：① 800K 原始 → 180K 过滤反而降了 2.14，**「质量先于数量」**；② 180K → 2M 又降了 3.66，**「高质量数据也要扩规模」**；③ DBS + 2M 的组合比单独提质量或扩规模都强得多。

---

## 4. 实验与消融

### 4.1 训练配置

- **3 阶段渐进训练**（按 latent 长度 + 模型规模逐步放大）：
  1. Stage 1：1.5B 模型，latent 长度 512，lr 1e-4，700k 步。
  2. Stage 2：1.5B 模型，latent 长度 2048，lr 5e-5，300k 步。
  3. Stage 3：4B MoE 模型，latent 长度 4096，lr 1e-5，100k 步（用 1M 高质量子集）。
- **总训练成本**：~3 周 × 160 张 A100 GPU。
- **VAE 训练**：8 层 encoder + 16 层 decoder，$\lambda_{\text{sn}}=10$, $\lambda_{\text{eik}}=0.1$, $\lambda_{\text{kl}}=0.001$，per-GPU batch 6，2.5M 步，约 12 天 × 32 A100。
- **推理时图像预处理**：检测前景 → 缩放到 90% 占比 → 中心小幅旋转（与训练增强一致）。

### 4.2 评估协议

- **Normal-FID**：与 FID 同源（Heusel et al., 2017），但比较的是「预测 normal map」 vs 「GT normal map」，更适合评估无纹理 3D 几何质量。计算方式：GT 网格渲染 RGB 输入 → 模型生成网格 → 同视点渲染 normal map → 与 GT normal map 算 FID。
- **GPTEval3D**：用 Claude 3.5 替代 GPT-4V 当 LMM 评估器（论文 §5.3.2），从 5 维打分（3D plausibility / text-asset 对齐 / 几何细节 / 纹理细节 / 纹理-几何一致性）。
- **VAE 评估**：Chamfer、F-Score@0.02、Normal Consistency。

### 4.3 主要消融结果

**Tab.1：Flow 改进项（975M 模型，latent 512，300k 步，180K Objaverse）**

| 条件 | 跳连 | 采样 | Normal-FID |
| --- | --- | --- | --- |
| DINOv2 only | ✗ | R-Flow | 10.69 |
| CLIP + DINOv2 | ✗ | R-Flow | 10.61 |
| CLIP + DINOv2 | ✓ | DDPM | 9.63 |
| CLIP + DINOv2 | ✓ | EDM | 9.50 |
| **CLIP + DINOv2** | **✓** | **R-Flow** | **9.47** |

**跳连贡献最大**（-1.14），CLIP 双特征次之（-0.08），R-Flow 优于 EDM/DDPM。

**Tab.2：Flow 扩规模（同样条件，逐步扩）**

| 数据 | Token | MoE | Normal-FID |
| --- | --- | --- | --- |
| Objaverse | 512 | ✗ | 9.47 |
| Objaverse | 2048 | ✗ | 8.38 |
| Objaverse | 4096 | ✗ | 8.12 |
| Objaverse | 4096 | ✓ | 7.94 |
| **TripoSG (2M + 4B + 4096)** | **4096** | **✓** | **3.36** |

**关键观察**：① latent 长度 512→2048 提升最大（-1.09），2048→4096 边际递减（-0.26）；② MoE 提升小但稳定（-0.18）；③ **真正让 Normal-FID 从 7.94 跳到 3.36 的是「1.5B → 4B + 1M 高质量子集 → 2M 高质量全集」**——质量 + 数量的协同收益远超单项。

**Tab.3：VAE 监督项消融（180K Objaverse 训练）**

| 数据 | 表征 | $\mathcal{L}_{\text{sn}}$ | $\mathcal{L}_{\text{eik}}$ | Chamfer↓ | F-Score↑ | N.C.↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Objaverse | Occ | ✗ | ✗ | 4.59 | 0.999 | 0.952 |
| Objaverse | SDF | ✗ | ✗ | 4.60 | 0.999 | 0.955 |
| Objaverse | SDF | ✓ | ✗ | 4.56 | 0.999 | 0.956 |
| Objaverse | SDF | ✓ | ✓ | 4.57 | 0.999 | 0.957 |
| **TripoSG** | **SDF** | **✓** | **✓** | **4.51** | **0.999** | **0.958** |

SDF + 法向 + Eikonal 组合小幅但稳定提升；Occ baseline 在细节处明显 aliasing（Fig.9）。

### 4.4 定性对比

![TripoSG 与同期 SOTA 对比（normal map 渲染）](assets/figures/07-comparison-sota.png)

*图 7（输入图 vs 6 个 baseline：MeshFormer / MeshLRM / CRM / TripoSR / InstantMesh / CraftsMan-1.5 vs TripoSG；TripoSG 在细节、几何一致性、复杂结构生成上全面领先）。来源：TripoSG 论文 Fig.7 (p11)。*

5 维度评估（GPTEval3D/Claude 3.5）：① 语义一致性（图 1&2）、② 细节（衣服 / 配饰，图 3&4）、③ 泛化（漫画 / 卡通风格，图 5&6）、④ 复杂空间结构（图 7&8）、⑤ 整体表现。TripoSG 在 5 项上同时领先。

### 4.5 纹理生成

论文 §6 用了**Meta 3D TextureGen** 风格的多视图纹理 pipeline：用 TripoSG 生成的网格渲 normal map 作为多视图扩散的条件，生成一致的多视图 RGB，再投影到 mesh 上得 texture map（Fig.12）。这部分不属本文主要贡献，依赖外部纹理模型。

---

## 5. 局限与评价

### 5.1 方法论层面的限制

- **VAE 训练成本极高**：8 层 encoder + 16 层 decoder 在 32 张 A100 跑 12 天只是 1 次实验；如果想换数据集重训、换 latent 长度、换监督形式，每次都接近这个量级。
- **4B MoE 模型的「推理便宜」是有条件的**：每个 token 走 top-2 expert + 1 shared expert，**单 batch 推理时 expert 路由不均衡会拖累速度**；4096 token × 21 blocks 的 attention 在 4B 模型上对显存的压力很大。
- **数据管线的人工成本不公开**：10K 标注、10 位 3D 建模师打 1-5 分、CLIP+DINOv2 线性回归打 1-5 分……这套流程的「人工 → 模型」转换没有详细说明，难复现。
- **DBS 的失败模式**：论文承认「无纹理模型用 ControlNet++ 上色」会引入生成式伪影；CLIP+DINOv2 打分器对「风格化 / 抽象」3D 模型可能判错分。
- **未涉及多物体 / 场景生成**：仍然是单物体生成，不直接支持场景级 3D。

### 5.2 与 3DShape2VecSet / LATTICE / CLAY 的关系

TripoSG 是 3DShape2VecSet 范式（latent set + DiT）走向 **「大数据 + 大模型 + Rectified Flow」** 路线上的关键工程化样本。LATTICE（2512.03052）在 TripoSG 之后做稀疏体素路线的对应版本；CLAY（2404.12297, Zhang et al.）是 TripoSG 直接对标的同期工作，但 CLATE 用 1.5B 文本 base + 352M 图像 head，**分离训练**，而 TripoSG 把 1.5B 直接扩到 4B + 联合训练，是更激进的端到端路线。CraftsMan-1.5（2405.14979）也是同期 VecSet 系工作，但模型更小、Normal-FID 显著落后。

### 5.3 复现建议

- **算力门槛**：完整复现需要 160 张 A100 跑 3 周（flow）+ 32 张 A100 跑 12 天（VAE），中小团队难以承担。
- **可行的小型复现路径**：用 TripoSG 公开的预训练 VAE（如果在 HF 上）+ 1.5B R-Flow 模型（如果有），只复现 Stage 1 训练（latent 512、180K 数据、lr 1e-4、300k 步），32 张 A100 跑 3 天即可对照 Tab.1 的 9.47。
- **数据复现**：DBS 管线中的「10K 人工打分」是最大瓶颈；如果替换为「GPT-4V 打分 + 人工校验 1K」，可降低人力成本但需重新校准打分器。
- **代码**：项目页 [https://www.tripo3d.ai/work/triposg](https://www.tripo3d.ai/work/triposg) 提供 demo + 权重，但训练脚本未完全公开。

### 5.4 个人判断（综合）

- **方法论贡献 ★★★★**：把 3DShape2VecSet 的 latent set 骨架 + Rectified Flow + 数据工程 + MoE 拼成一套可复现的端到端系统，且每一项都有明确消融支撑，**「R-Flow + 跳连 + 4096 token + 4B MoE + 2M 高质量数据」的乘数效应**是 2025 年初 image-to-3D 最重要的工程经验之一。
- **学术创新性 ★★★**：本质是「已知组件的系统整合 + 工程化创新」，**没有提出全新表征或全新学习范式**；贡献主要在「数据 + 训练范式 + 监督形式」上做对的事。
- **落地价值 ★★★★★**：Tripo3D 团队本身是做 3D 资产生成的商业公司，论文即产品原型，**Normal-FID 3.36 + 无纹理 mesh 直接可用 + 纹理 pipeline 解耦**的设计非常实用。
- **与同期的差距**：相对 LATTICE 的「O-Voxel + 1.5B → 4B」路线，TripoSG 走的是「Latent Set + 1.5B → 4B」路线，两者在 SOTA 榜单上互有胜负；相对 TRELLIS.2 的「O-Voxel + 几何材质解耦」路线，TripoSG 没有同时建模 PBR 材质，这是其后续工作（TripoSR v2、Tripo 2.x）的演进方向。
- **核心 takeaway**：3D 生成在 2025 年初的 SOTA 公式 ≈ **「VecSet 骨架 + 神经 SDF VAE + 整流流 + MoE 扩规模 + 大数据工程」**；TripoSG 把这个公式跑通并稳定发布，对后续 3D 生成研究的「方法选型」影响很大。

---

## 6. 参考资源

- **论文 PDF**：[assets/sources/paper.pdf](assets/sources/paper.pdf)（22 页，11.5 MB）
- **项目页**：[https://www.tripo3d.ai/work/triposg](https://www.tripo3d.ai/work/triposg)
- **关键参考**（论文中引用的核心工作）：
  - 3DShape2VecSet（[2301.11445](https://arxiv.org/abs/2301.11445)）—— Latent Set 范式源头
  - DiT（[2212.09748](https://arxiv.org/abs/2212.09748)）—— Transformer 扩散基础
  - Rectified Flow（[2209.03003](https://arxiv.org/abs/2209.03003)）—— 线性轨迹流
  - CLAY（[2404.12297](https://arxiv.org/abs/2404.12297)）—— 大规模 VecSet 同期工作
  - Michelangelo（[2312.08741](https://arxiv.org/abs/2312.08741)）—— 同期 shape-image-text 潜空间工作
  - DiT-MoE（[2407.11633](https://arxiv.org/abs/2407.11633)）—— MoE Transformer 范式
  - SD3（[2403.03206](https://arxiv.org/abs/2403.03206)）—— logit-normal + resolution-dependent timestep shift
  - Objaverse / Objaverse-XL（[2212.08051](https://arxiv.org/abs/2212.08051) / [2407.10608](https://arxiv.org/abs/2407.10608)）—— 训练数据源

## 7. 关键图源

| 图编号 | 描述 | 文件 | PDF 页 |
| --- | --- | --- | --- |
| Fig.1 | 输入图 + 4 个生成结果 | `figures/01-teaser.png` | p1 |
| Fig.3 | 整流流架构 + 块内结构 + MoE | `figures/03-architecture.png` | p5 |
| Fig.4 | VAE 编码器-解码器 + 法向监督 | `figures/04-vae-architecture.png` | p7 |
| Fig.5 | Occupancy vs SDF 重建对比 | `figures/05-occupancy-vs-sdf.png` | p7 |
| Fig.6 | 数据构建系统 4 阶段管线 | `figures/06-data-building-system.png` | p9 |
| Fig.7 | 与 6 个 SOTA baseline 对比 | `figures/07-comparison-sota.png` | p11 |

## 8. 元信息

- **初版时间**：2026-08-25
- **精读耗时**：约 3 小时（pymupdf 全文本提取 + 关键图裁切 + 论文全文精读）
- **精读重点**：整流流 backbone 设计、VAE 混合监督、数据清洗管线、4 维度消融对照
- **可改进项**：① 笔记未详细覆盖 §6 纹理生成（依赖 Meta 3D TextureGen，非本文主要贡献）；② 推理时显存 / 延迟数据未在论文中给出，落地时需自行 benchmark
