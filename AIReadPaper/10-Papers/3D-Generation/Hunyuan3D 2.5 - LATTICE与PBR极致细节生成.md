---
title: Hunyuan3D 2.5：LATTICE 与 PBR 极致细节生成精读
tags: [论文精读, 3D生成, LATTICE, PBR, Scaling]
paper: https://arxiv.org/abs/2506.16504
code: 截至2026-10-30未开源2.5专属代码或权重；报告所链Hunyuan3D-2仓库仅登记技术报告
rating: 4/5
created: 2026-10-30
updated: 2026-10-30
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

### 3.2 Shared Attention Map + Separate Values

多材质错位的主要原因被归因于 reference attention masks 不一致。2.5 用 Albedo 分支计算共享空间对应：

$$M_{\mathrm{attn}}=\operatorname{Softmax}\left(\frac{Q_{\mathrm{albedo}}K_{\mathrm{ref}}^T}{\sqrt d}\right)$$

然后使用不同 value 更新两类材质：

$$z_{\mathrm{albedo}}^{\mathrm{new}}=z_{\mathrm{albedo}}+\operatorname{MLP}_{\mathrm{albedo}}\left[M_{\mathrm{attn}}V_{\mathrm{albedo}}\right]$$

$$z_{\mathrm{MR}}^{\mathrm{new}}=z_{\mathrm{MR}}+\operatorname{MLP}_{\mathrm{MR}}\left[M_{\mathrm{attn}}V_{\mathrm{MR}}\right]$$

直觉上，QK 决定“参考图的这个位置对应 mesh 的哪里”，因此各材质应共享；V 决定“在那里写入什么物理属性”，因此 Albedo 与 MR 必须分开。这比 2.1 笼统的跨分支特征传播更明确。

### 3.3 3D-Aware RoPE 与照明解耦

2.5 继承 RomanTex 的 3D-Aware RoPE，用于 other-view latents 的跨视图 attention，以减少 seam 和 ghosting；报告没有说明 RoPE 具体编码 CCM、表面点还是相机射线，因此“由 normal/CCM 建立对应”只能作为解释性推断。同时模型沿用 illumination-invariant consistency loss，让不同光照参考图得到一致 intrinsic materials；报告将细节指向 RomanTex 与 MaterialMVP，没有重新披露完整损失。

### 3.4 双阶段分辨率增强

直接训练高分辨率多视图会爆显存，只能减少 views，进而破坏 dense-view consistency。2.5 采用：

1. **Phase 1**：6-view、$512\times512$ 常规多视图训练，先学全局一致性；
2. **Phase 2**：随机同步 zoom-in reference 与 generated views，在不全量高分辨率训练的情况下学习局部高频细节；
3. **Inference**：扩散生成的多视图图像最高 $768\times768$，使用 UniPC 加速；该数字不是最终 UV atlas 分辨率。

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

## 5. 算法级实现：按完整执行流程写清输入输出

2.5 专属代码和权重没有开放，所以本节不能伪造类名与固定配置。为了既把算法讲清又不混淆证据，下面统一使用三种标记：**【2.5 报告明确】**、**【LATTICE 后续论文补充】**、**【无法确认】**。后续 LATTICE 论文的 VoxSet 细节能解释 2.5 的技术谱系，但其最高 4.5B 研究配置不等于 2.5 的 10B 产品模型。

### 5.1 Shape 总流程与张量总表

| 步骤 | 证据 | 输入 | 输出 |
|---:|---|---|---|
| 1 | 【2.5 报告明确】 | 单图或四视图 $I\in\mathbb R^{B\times V_I\times3\times H_0\times W_0}$ | LATTICE shape model 的高细节 mesh；2.5 内部中间量未披露 |
| 2 | 【LATTICE 后续论文补充】 | 单图 $I$ | coarse generator 输出 $V_0\in\mathbb R^{N_v\times3},F_0\in\mathbb N^{N_f\times3}$ |
| 3 | 【LATTICE 后续论文补充】 | coarse mesh | 与表面相交的 active voxel centers $Q_{pos}\in\mathbb R^{B\times T\times3}$ |
| 4 | 【LATTICE 后续论文补充】 | noisy VoxSet latent、voxel positions、DINO tokens | Rectified-flow DiT 输出 $Z\in\mathbb R^{B\times T\times64}$ |
| 5 | 【LATTICE 后续论文补充】 | $Z$ 与 grid queries $X\in\mathbb R^{B\times M\times3}$ | $SDF\in\mathbb R^{B\times M\times1}$ |
| 6 | 【LATTICE 后续论文补充】 | 离散 SDF field | 精细 mesh $V\in\mathbb R^{N'_v\times3},F\in\mathbb N^{N'_f\times3}$ |
| 7 | 【2.5 报告明确】 | mesh、reference image、normal/CCM | 多视图 Albedo/MR，再融合为 PBR 材质 |

$N_v,N_f,T,M$ 都随对象或配置变化。2.5 报告没有公开它们的实际值；后续 LATTICE 论文中的 4096/8192/20480 tokens 分别属于重建实验，训练主线又使用 1024→6144，推理测试到 12288/24576、最高声称 30720，不能把这些数字拼成一个固定 10B 配置。

### 5.2 Shape 第一步：coarse generator 只负责预测“哪里有表面”

【LATTICE 后续论文补充】给定单图：

$$I\in\mathbb R^{B\times3\times H_0\times W_0}\xrightarrow{G_{coarse}}(V_0,F_0)$$

$G_{coarse}$ 可以是 Hunyuan3D-2、TRELLIS 等现成 image-to-3D 模型。以 Hunyuan3D-2 为例，其公开接口是 `image → trimesh`；但【无法确认】2.5/LATTICE 内部使用哪个 checkpoint、如何归一化尺度和朝向。

粗 mesh 的目的不是提供最终细节，而是把开放的三维搜索空间变成“沿哪些区域生成内容”。它随后被 voxelize 到某个 $R^3$ 网格，取所有与表面相交的 active voxel centers：

$$Q_{pos}=\{q_i\}_{i=1}^{T},\qquad Q_{pos}\in\mathbb R^{B\times T\times3}$$

【无法确认】$R$、surface-intersection 判定、是否做 dilation 和实际 $T$；因此不能把 $64^3$ 写成 2.5 固定配置。这样设计的关键是解耦“where”和“what”：coarse stage 先告诉精细生成器哪里可能有表面，第二阶段只在这些已知位置生成高频几何内容。

### 5.3 Shape 第二步：VoxSet VAE 如何定义几何 latent

【LATTICE 后续论文补充】训练输入点云为：

$$P\in\mathbb R^{B\times N\times7}$$

每点 7 维由 xyz、normal 和 sharp-edge flag 组成。Voxel Queries $Q_{pos}\in\mathbb R^{B\times T\times3}$ 对点特征做一次 cross-attention，随后经过 8 层 self-attention，形成可定位的 VoxSet tokens；表 1 的 `64×T` 最合理地解释为：

$$Z\in\mathbb R^{B\times T\times64}$$

但论文没有声明内存布局是 `[B,T,64]` 还是 `[B,64,T]`。需要纠正一个容易误读的点：Figure 5 的顺序是：

```text
Point features + Voxel Queries
→ Cross Attention
→ Self Attention × 8
→ VoxSet latent
→ Self Attention × 16
→ Cross Attention(Grid Queries)
→ SDF
```

并不是“8 层 Cross Attention + 16 层 Self Attention 的对称编解码器”。为什么 voxel center 比 surface point query 更适合生成？因为 surface point 是要被生成的未知目标，测试时拿不到其位置；coarse mesh 的 active voxel centers 在去噪前就已知，可以作为每个 latent token 的显式空间锚点。

### 5.4 Shape 第三步：DINOv2-Giant 条件下的 Rectified-Flow DiT

【LATTICE 后续论文补充】DINOv2-Giant 接收经 binary mask crop、保持宽高比的高分辨率输入，最长边/目标分辨率写作 1022。ViT-g/14 的 patch width 为 1536，因此图像条件可写成：

$$E_I\in\mathbb R^{B\times L_I\times1536}$$

若额外假设固定方形 $1022\times1022$、普通 ViT-g/14 且删除 CLS，则 $L_I=73^2=5329$；但论文同时说保持 aspect ratio 以减少 token length，所以实际更严谨是：

$$L_I=\left\lfloor\frac{H_I}{14}\right\rfloor\left\lfloor\frac{W_I}{14}\right\rfloor$$

【无法确认】是否使用 register-token 版本、实际 $H_I,W_I$ 和预处理 padding，因此 5329 不能当作无条件固定值。

Rectified-flow DiT 的输入输出为：

$$Z_t\in\mathbb R^{B\times T\times64},\quad Q_{pos}\in\mathbb R^{B\times T\times3},\quad E_I\in\mathbb R^{B\times L_I\times1536}$$

$$v_\theta(Z_t,t,Q_{pos},E_I)\in\mathbb R^{B\times T\times64}$$

每个 noisy latent 根据对应 voxel center 注入 3D RoPE，再以 DINO tokens 为图像条件预测同 shape 的 velocity。训练使用 rectified flow/linear coupling，并以 10% 概率把图像条件替换为零以支持 CFG。显式位置把“第 $i$ 个 latent 应该生成哪里”提前告诉模型，参数增大时不必重复从内容中猜位置，正是 VoxSet 比 VecSet 更易 scaling 的原因。

【无法确认】2.5 10B 的 DiT depth、hidden width、heads、64→hidden projection、RoPE xyz 维度分配、solver、sampling steps 与 CFG scale；后续 4.5B 论文也没有公开到代码配置级别。

### 5.5 Shape 第四步：SDF 查询与 Marching Cubes

【LATTICE 后续论文补充】VoxSet latent 先通过 16 层 self-attention，再让规则或分块 grid queries 读取：

$$X\in\mathbb R^{B\times M\times3}$$

$$D_{vae}(X\mid Z)\rightarrow S\in\mathbb R^{B\times M\times1}$$

$M$ 是隐式场查询点数。Cross-attention 的直觉是：每个 xyz query 主动从 $T$ 个 VoxSet tokens 中读取与该空间位置相关的几何内容，最终线性头输出一个 SDF 标量。离散到网格后，Marching Cubes 对零等值面提取：

$$S\rightarrow V\in\mathbb R^{N'_v\times3},\quad F\in\mathbb N^{N'_f\times3}$$

【无法确认】2.5 的 SDF 分辨率、$M$、chunk size、iso-level 和是否采用 FlashVDM；报告只说使用 guidance/step distillation 加速 shape model，未给具体推理账。

### 5.6 PBR Paint 第一步：三条条件流的 symbolic shape

【2.5 报告明确】Paint 同时接收参考图、mesh geometry 与两类 noisy material latents：

| 流 | 输入 | 中间表示 |
|---|---|---|
| Reference | $I_{ref}\in\mathbb R^{B\times3\times H_r\times W_r}$ | $K_{ref}\in\mathbb R^{B\times L_r\times d}$ |
| Geometry | mesh + $V$ 个相机 | normal/CCM：$N,C\in\mathbb R^{B\times V\times C_g\times H\times W}$ |
| Albedo | noisy albedo latent | $z_A\in\mathbb R^{B\times L_g\times C_z}$ |
| MR | noisy metallic/roughness latent | $z_{MR}\in\mathbb R^{B\times L_g\times C_z}$ |

【无法确认】2.5 的 VAE 下采样倍率、$L_r,d,C_g,L_g,C_z$、normal/CCM channel packing 以及最终 UV atlas resolution。报告明确的三个 learnable embeddings 为：

$$E_{albedo},E_{mr},E_{normal}\in\mathbb R^{16\times1024}$$

最自然的解释是 16 个材质 condition tokens、每 token 1024 维，但报告没有解释这两个维度，不能再进一步补造。

### 5.7 PBR Paint 第二步：共享空间对应，分离材质 Value

【2.5 报告明确】共享的不是 Q/K projection 参数，而是由 Albedo Query 和 Reference Key 算出的 attention map：

$$Q_A\in\mathbb R^{B\times L_g\times d},\quad K_{ref}\in\mathbb R^{B\times L_r\times d}$$

$$M_{attn}=\operatorname{Softmax}(Q_AK_{ref}^{T}/\sqrt d)\in\mathbb R^{B\times L_g\times L_r}$$

然后两类材质分别读取自己的 Value：

$$z_A^{new}=z_A+\operatorname{MLP}_A(M_{attn}V_A)$$

$$z_{MR}^{new}=z_{MR}+\operatorname{MLP}_{MR}(M_{attn}V_{MR})$$

这一步的设计逻辑非常具体：QK 决定“参考图的哪个位置对应 mesh 的哪块表面”，因此 Albedo 与 MR 应共享；V 决定“该表面写入什么材质值”，因此颜色与 metallic/roughness 必须分开。原先称作“Shared QK Mask”容易误解为共享 Q/K 参数，准确叫法应是“共享 Albedo-Reference attention map”。

### 5.8 PBR Paint 第三步：跨视图位置与 zoom-in 训练

【2.5 报告明确】模型继承 RomanTex 的 3D-Aware RoPE，用于 other-view latents 的 multi-view attention，以减少 seam 和 ghosting。但报告没有说明 RoPE 到底编码 CCM、表面点、相机射线还是其他坐标，也没有给频率、xyz 分配或应用到 Q/K 的细节。因此“用 normal/CCM 建立三维对应”只能作为合理解释，不能写成 2.5 源码事实。

分辨率训练分两阶段：

1. Phase 1：六视图 $512\times512$，即 $X\in\mathbb R^{B\times6\times3\times512\times512}$，先学习全局一致性；
2. Phase 2：reference 与 generated multiview images 同步随机 zoom-in，把显存集中到局部高频纹理—几何对应；zoom ratio 与 crop size 未披露；
3. Inference：多视图图像最高 $768\times768$，并用 UniPC 加速。

$768^2$ 指扩散生成的多视图图像分辨率，不等于最终 UV atlas 或所谓“4K texture”的具体代码实现。

### 5.9 10B 2.5 与 4.5B LATTICE 不能互相替代

| 项目 | Hunyuan3D 2.5 | 后续 LATTICE 论文 |
|---|---|---|
| 最大模型 | 10B | 4.5B |
| 规模序列 | 未披露 | 0.6B / 1.9B / 4.5B |
| VoxSet/VAE/DiT 细节 | 未披露 | 论文级披露 |
| 代码与权重 | 未开放 | 截至官方仓库 2025-12-04 快照仍 TODO |
| 是否同一 checkpoint | 无法确认 | 无法确认 |

正确结论是：2.5 报告明确“最大 LATTICE shape model 为 10B”；后续论文解释了 VoxSet/voxel queries/rectified-flow 的方法原理，并验证到 4.5B。没有证据证明 10B 只是 4.5B 配置的等比例放大，也不能假设两者使用相同 VAE、token 数、数据与蒸馏配方。

---

## 6. 开源与复现核查

### 6.1 没有 2.5 专属代码和权重

截至 2026-10-30：

- 未找到 `Hunyuan3D-2.5` 官方仓库；
- 未找到腾讯 Hugging Face 的 2.5/10B 权重；
- 论文链接的 `Tencent-Hunyuan/Hunyuan3D-2` 仓库只在 README 中登记技术报告；
- 官方 `wasahaiah/LATTICE` 仓库截至 2025-12-04 commit `8355d55` 只有 `README.md` 与 `assets/`，TODO 仍明确列出 `Code and Model`；
- 因而不存在可供核对 2.5 模型类、配置、权重与实际推理命令的官方实现。

所以本篇可以写**算法级 symbolic 张量表**，也可以引用后续 LATTICE 论文解释技术来源，但不能把 2.0/2.1 类名复制过来，也不能把 4.5B LATTICE 研究配置当成 10B 2.5 产品代码。

### 6.2 可借助但不可替代的关联笔记

- [[LATTICE - 大规模高保真三维生成表征]]：解释 VoxSet、Voxel Queries、3D RoPE 与 scaling；
- [[Hunyuan3D 2.1 - 生产级PBR三维资产生成]]：解释 Albedo/MR 双分支、3D-Aware RoPE、illumination consistency 与 UV baking；
- 本篇 2.5：解释这些技术如何在最大 10B 系统中组合，以及新的 shared attention mask、zoom-in resolution enhancement 和端到端结果。

### 6.3 复现成本无法估计

报告没有公布数据量、模型层数、token 数、训练 GPU、GPU-days、推理步数、显存、网格分辨率或纹理烘焙实现。唯一明确的规模数字是最大 10B，纹理训练/推理分辨率是 $512^2/768^2$。这不足以做可信计算账。

---

## 7. 局限与评价

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

## 8. 系列总览

| 版本 | 几何表示 | 材质 | 公开程度 | 核心价值 |
|---|---|---|---|---|
| 1.0 | triplane/SDF 重建 | Baking | 推理开放 | 多视图生成与重建统一 |
| 2.0 | VecSet ShapeVAE + 1.1B DiT | RGB texture | 推理/权重开放 | 建立 Shape→Texture 框架 |
| 2.1 | 4096-token VecSet + 3.3B MoE DiT | Albedo/MR PBR | 推理与部分训练开放 | 可重光照材质 |
| **2.5** | **LATTICE，最大 10B** | **共享空间对应的高分辨率 PBR** | **报告/在线服务，权重未开放** | 极细节、锐边与光滑表面兼顾 |

---

## 9. 图表与材料索引

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
