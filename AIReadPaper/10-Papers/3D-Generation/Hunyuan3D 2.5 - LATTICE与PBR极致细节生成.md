---
title: Hunyuan3D 2.5：LATTICE 与 PBR 极致细节生成精读
tags: [论文精读, 3D生成, LATTICE, PBR, Scaling]
paper: https://arxiv.org/abs/2506.16504
code: 截至2026-10-30未开源2.5专属代码或权重；报告所链Hunyuan3D-2仓库仅登记技术报告
rating: 4/5
created: 2026-10-30
---

# Hunyuan3D 2.5：LATTICE 与 PBR 极致细节生成精读

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details* |
| 日期 | 2025-06-19，arXiv:2506.16504 |
| 核心升级 | LATTICE Shape Foundation Model（最大 10B）+ 高分辨率多视图 PBR Paint |
| 输入 | 单图或四张多视图图像 |
| 输出 | 高细节裸网格 + albedo/metallic/roughness PBR 材质 |
| 开源 | **2.5 专属代码和权重未开源**；报告只链接旧的 Hunyuan3D-2 仓库 |
| 离线材料 | [论文](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/sources/paper.pdf) · [来源](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/sources.md) |
| 评级 | **★★★★☆**：结果强、方向关键，但技术报告过于简略且不可本地复现 |

> [!summary] 一句话结论
> Hunyuan3D 2.5 保留 2.0/2.1 的 Shape→Texture 两阶段接口，却把几何主干换成可稳定 scaling 到 10B 的 LATTICE，并把 PBR Paint 改为共享 Albedo attention mask、分离 Albedo/MR values 的多通道生成，再用“6 视图 $512^2$ 基础训练 + zoom-in 高分辨率训练”提高纹理—几何对齐；它代表系列从开放研究模型走向闭源满血生产模型的转折点。

---

## 1. 版本位置：2.5 不是 2.1 的小修补

### 1.1 系列主线

- [[Hunyuan3D 1.0 - 统一两阶段三维生成]]：多视图扩散→稀疏视图重建；
- [[Hunyuan3D 2.0 - 大规模几何与纹理扩散]]：VecSet ShapeVAE/DiT→RGB Paint；
- [[Hunyuan3D 2.1 - 生产级PBR三维资产生成]]：3.3B Shape→Albedo/MR PBR Paint；
- **2.5**：LATTICE 最大 10B→更高分辨率 PBR Paint。

2.5 的核心不是增加一个后处理模块，而是同时替换/放大几何基础模型与材质生成策略。报告宣称最大模型 10B，目标从“正确生成对象”升级为“手工资产级细节、锐边与光滑表面同时成立”。

### 1.2 两阶段接口保持不变

![](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/figures/01-pipeline.png)

*Figure 3 / PDF p.3。输入先去背景和缩放，Shape 模型生成裸网格，网格后处理得到 normal/UV 等条件，最后 PBR Paint 输出材质。*

总体仍是：

$$I\xrightarrow{\text{Preprocess}}\hat I\xrightarrow{\text{LATTICE Shape}}\mathcal M\xrightarrow{\text{Mesh Processing}}\{\mathcal M,N,\mathrm{CCM},UV\}\xrightarrow{\text{PBR Paint}}\mathcal A$$

这种接口稳定性说明 2.x 的真正设计资产是“几何与材质解耦”，而不是某一版具体 DiT。

---

## 2. LATTICE Shape：报告告诉了什么，又没告诉什么

### 2.1 报告明确披露

LATTICE 是大规模 3D diffusion foundation model：

- 支持单图或 4-view 输入；
- 在大规模高质量复杂 3D 数据上训练；
- 随数据、模型参数和计算增长表现稳定提升；
- 最大模型 10B；
- 使用 guidance distillation 与 step distillation 降低推理时间；
- 目标能力包括正确手指数、复杂轮辐、小尺度对象、薄结构、锐边、光滑表面和背面细节。

![](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/figures/02-shape-features.png)

*Figure 4 / PDF p.3。论文用实例展示“极细节、锐边、光滑面、薄结构和背面”能力，但没有结构图、训练公式或消融。*

### 2.2 报告没有披露

Hunyuan3D 2.5 报告没有解释 LATTICE 的 latent 表示、VAE、token 数、DiT 层数、宽度、RoPE、损失、数据量、训练 GPU 或采样步数。也就是说，它只说“新基础模型叫 LATTICE、可 scaling 到 10B”，却没有给出能复现的模型定义。

这些细节后来在独立笔记 [[LATTICE - 大规模高保真三维生成表征]] 中得到更系统说明：VoxSet 以表面相交体素中心作为可定位 query，Voxel Queries 与 3D RoPE 解决 VecSet token 不可定位问题，并展示 0.6B→4.5B 的 scaling 与 test-time token scaling。但必须区分：

- 独立 LATTICE 论文解释**表示与 scaling 原理**；
- Hunyuan3D 2.5 报告描述**10B 完整产品系统**；
- 当前没有证据证明独立开源配置等同于 2.5 的 10B 权重与训练配方。

### 2.3 为什么 LATTICE 比 VecSet 更适合放大

2.0/2.1 的 VecSet token 不对应固定空间位置，DiT 甚至刻意不加 positional embedding。这样压缩紧凑，但模型扩大时仍要从 token 内容反推几何位置。

LATTICE 的核心思想可以概括为：**localizable 比强行局部化更重要**。把 queries 锚定到表面相交 voxel 后，Transformer 可以明确知道 token 的 3D 位置，并通过 3D RoPE 建模相对空间关系；模型容量增加更容易转化为高频结构与复杂拓扑能力。这正解释了 2.5 为什么把最大规模推到 10B，而不继续单纯扩大 2.1 VecSet。

---

## 3. PBR Paint：共享空间对应、分离材质值

![](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/figures/03-pbr-framework.png)

*Figure 5 / PDF p.4。2.5 延续 Reference/Generation 双流、normal/CCM 和 3D-Aware RoPE；关键升级是 Albedo/MR 双分支共享 QK attention mask，却使用各自 V。*

### 3.1 多通道材质嵌入

论文称为三种材质 maps，但实现语义仍是：

- Albedo；
- Metallic；
- Roughness；
- 其中 Metallic/Roughness 合并为 MR channel。

模型初始化三个可学习 embedding：

$$E_{\mathrm{albedo}},E_{\mathrm{mr}},E_{\mathrm{normal}}\in\mathbb R^{16\times1024}$$

并通过 cross-attention 注入对应通道。这里 normal 在报告文字中被列为 material embedding，但总体架构仍把 normal/CCM 作为 mesh geometry conditions；报告对此表述不够严谨。

### 3.2 Shared QK Mask + Separate Values

多材质错位的主要原因被归因于 reference attention masks 不一致。2.5 用 Albedo 分支计算共享空间对应：

$$M_{\mathrm{attn}}=\operatorname{Softmax}\left(\frac{Q_{\mathrm{albedo}}K_{\mathrm{ref}}^T}{\sqrt d}\right)$$

然后使用不同 value 更新两类材质：

$$z_{\mathrm{albedo}}^{\mathrm{new}}=z_{\mathrm{albedo}}+\operatorname{MLP}_{\mathrm{albedo}}\left[M_{\mathrm{attn}}V_{\mathrm{albedo}}\right]$$

$$z_{\mathrm{MR}}^{\mathrm{new}}=z_{\mathrm{MR}}+\operatorname{MLP}_{\mathrm{MR}}\left[M_{\mathrm{attn}}V_{\mathrm{MR}}\right]$$

直觉上，QK 决定“参考图的这个位置对应 mesh 的哪里”，因此各材质应共享；V 决定“在那里写入什么物理属性”，因此 Albedo 与 MR 必须分开。这比 2.1 笼统的跨分支特征传播更明确。

### 3.3 3D-Aware RoPE 与照明解耦

2.5 继承 3D-Aware RoPE，用 normal/CCM 建立跨视图三维对应，减少 seam 和 ghosting；同时沿用 illumination-invariant consistency loss，让不同光照参考图得到一致 intrinsic materials。报告将具体方法指向 RomanTex 与 MaterialMVP，并没有重新披露完整损失。

### 3.4 双阶段分辨率增强

直接训练高分辨率多视图会爆显存，只能减少 views，进而破坏 dense-view consistency。2.5 采用：

1. **Phase 1**：6-view、$512\times512$ 常规多视图训练，先学全局一致性；
2. **Phase 2**：随机同步 zoom-in reference 与 generated views，在不全量高分辨率训练的情况下学习局部高频细节；
3. **Inference**：最高 $768\times768$ 多视图，使用 UniPC 加速。

这本质是局部 crop curriculum：保留 6-view 全局学习，再把有限显存集中到局部纹理—几何对齐。

---

## 4. 实验结果

### 4.1 Shape 定量

| 方法 | ULIP-T ↑ | ULIP-I ↑ | Uni3D-T ↑ | Uni3D-I ↑ |
|---|---:|---:|---:|---:|
| Michelangelo | 0.0752 | 0.1152 | 0.2133 | 0.2611 |
| Craftsman 1.5 | 0.0745 | 0.1296 | 0.2375 | 0.2987 |
| Trellis | 0.0769 | 0.1267 | 0.2496 | 0.3116 |
| Commercial Model 1 | 0.0741 | **0.1308** | 0.2464 | 0.3106 |
| Commercial Model 2 | 0.0746 | 0.1284 | 0.2516 | 0.3131 |
| Hunyuan3D 2.0 | 0.0771 | 0.1303 | 0.2519 | **0.3151** |
| **Hunyuan3D 2.5** | **0.07853** | 0.1306 | **0.2542** | **0.3151** |

2.5 在 ULIP-T 与 Uni3D-T 第一，Uni3D-I 与 2.0 持平，ULIP-I 仍略低于 Commercial Model 1。指标提升远小于视觉差异，作者也承认 embedding metrics 不能完整反映表面质量。

![](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/figures/04-shape-comparison.png)

*Figure 6 / PDF p.6。定性对比强调细小部件、锐边和背面完整度；但报告没有给 test set 大小、随机种子或失败样例。*

### 4.2 Texture 定量

| 方法 | CLIP-FID ↓ | FID ↓ | CMMD ↓ | CLIP-I ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Text2Tex | 31.83 | 187.7 | 2.738 | - | 0.1448 |
| SyncMVD | 29.93 | 189.2 | 2.584 | - | 0.1411 |
| Paint-it | 33.54 | 179.1 | 2.629 | - | 0.1538 |
| Paint3D | 26.86 | 176.9 | 2.400 | 0.8871 | 0.1261 |
| TexGen | 28.23 | 178.6 | 2.447 | 0.8818 | 0.1331 |
| **Hunyuan3D 2.5** | **23.97** | **165.8** | **2.064** | **0.9281** | **0.1231** |

2.5 在 CLIP-FID、FID、CMMD 与 CLIP-I 第一；LPIPS 则略差于 Paint3D 的 0.1261？实际上 0.1231 更低，仍为表中最好。报告没有单列 2.1，因此无法从该表直接量化 2.5 相比 2.1 的升级。

![](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/figures/05-texture-comparison.png)

*Figure 7 / PDF p.7。论文同时展示正背面、完整材质与 albedo，强调 MR 估计和去光照能力。匿名商业模型版本与参数未披露。*

### 4.3 用户研究

每名参与者对测试集每个样本的各方法排序。报告只给比例图，没有参与者人数、样本数、盲测和显著性检验。

![](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/figures/06-user-study.png)

*Figure 8 / PDF p.9。Image-to-3D 对 Commercial Model 1 的 Better 比例为 72.25%，约为对方优势比例的 9 倍；但协议披露不足。*

图中可读的 Better 比例包括 49.36%、60.87%、72.25% 等；作者重点声称 image-to-3D 达到 72% win rate。没有原始票数，不能重建置信区间。

### 4.4 缺少消融

报告没有对以下组件做控制变量：LATTICE vs VecSet、模型从 3.3B 到 10B 的 scaling 曲线、distillation、shared QK mask、3D RoPE、zoom-in phase、$512^2$ vs $768^2$。因此“2.5 整体强”成立，但“哪项设计贡献多少”没有答案。

---

## 5. 开源与复现核查

### 5.1 没有 2.5 专属代码和权重

截至 2026-10-30：

- 未找到 `Hunyuan3D-2.5` 官方仓库；
- 未找到腾讯 Hugging Face 的 2.5/10B 权重；
- 论文链接的 `Tencent/Hunyuan3D-2` 仓库只在 README 中发布技术报告链接；
- 仓库没有 2.5 配置、模型类、权重或专属推理命令。

因此本篇**不能写代码级张量表**。把 2.0 的代码复制过来，或把独立 LATTICE 仓库当成 10B 2.5 产品代码，都会制造错误事实。

### 5.2 可借助但不可替代的关联笔记

- [[LATTICE - 大规模高保真三维生成表征]]：解释 VoxSet、Voxel Queries、3D RoPE 与 scaling；
- [[Hunyuan3D 2.1 - 生产级PBR三维资产生成]]：解释 Albedo/MR 双分支、3D-Aware RoPE、illumination consistency 与 UV baking；
- 本篇 2.5：解释这些技术如何在最大 10B 系统中组合，以及新的 shared attention mask、zoom-in resolution enhancement 和端到端结果。

### 5.3 复现成本无法估计

报告没有公布数据量、模型层数、token 数、训练 GPU、GPU-days、推理步数、显存、网格分辨率或纹理烘焙实现。唯一明确的规模数字是最大 10B，纹理训练/推理分辨率是 $512^2/768^2$。这不足以做可信计算账。

---

## 6. 局限与评价

1. **技术报告过于高层**：10B LATTICE 的结构和训练几乎完全未披露。
2. **无代码无权重**：无法验证 scaling、蒸馏、几何质量和运行成本。
3. **没有 scaling 曲线**：宣称“stable improvement”，却没有不同参数规模的定量表。
4. **没有消融**：无法归因 shared mask、zoom-in 或更大数据的收益。
5. **PBR 仍只有基础通道**：没有 normal/displacement/opacity/emission/clearcoat 的系统评估。
6. **几何不是生产拓扑**：报告强调视觉表面，却没评估 quad topology、edge flow、UV seam 与动画可用性。
7. **用户研究不透明**：参与者、样本数、盲测和显著性均缺失。
8. **匿名商业基线不可复核**：版本与运行设置未知。

我的判断：2.5 是系列中**视觉能力跃迁最大、科研可审计性下降也最大**的一版。它证明 LATTICE + PBR 可以把单图资产推到更接近手工表面的观感，但不能仅凭这份报告得出“10B scaling 规律已被充分验证”或“已达到完整 production-ready”的结论。

---

## 7. 系列总览

| 版本 | 几何表示 | 材质 | 公开程度 | 核心价值 |
|---|---|---|---|---|
| 1.0 | triplane/SDF 重建 | Baking | 推理开放 | 多视图生成与重建统一 |
| 2.0 | VecSet ShapeVAE + 1.1B DiT | RGB texture | 推理/权重开放 | 建立 Shape→Texture 框架 |
| 2.1 | 4096-token VecSet + 3.3B MoE DiT | Albedo/MR PBR | 推理与部分训练开放 | 可重光照材质 |
| **2.5** | **LATTICE，最大 10B** | **共享空间对应的高分辨率 PBR** | **报告/在线服务，权重未开放** | 极细节、锐边与光滑表面兼顾 |

---

## 8. 图表与材料索引

| 图 | 内容 | 来源 |
|---|---|---|
| 图 1 | 2.5 两阶段管线 | Figure 3 / p.3 |
| 图 2 | Shape 能力目标 | Figure 4 / p.3 |
| 图 3 | PBR 材质框架 | Figure 5 / p.4 |
| 图 4 | Shape 定性对比 | Figure 6 / p.6 |
| 图 5 | Texture/PBR 定性对比 | Figure 7 / p.7 |
| 图 6 | 用户研究 | Figure 8 / p.9 |

- [论文 PDF](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/sources/paper.pdf)
- [来源与开源核查](assets/Hunyuan3D%202.5%20-%20LATTICE与PBR极致细节生成/sources.md)
