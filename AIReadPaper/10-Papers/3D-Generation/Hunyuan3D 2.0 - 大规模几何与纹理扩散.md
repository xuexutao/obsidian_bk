---
title: Hunyuan3D 2.0：大规模几何与纹理扩散模型精读
aliases:
  - Hunyuan3D 2.0
  - Hunyuan3D-DiT
  - Hunyuan3D-Paint
tags:
  - 论文精读
  - 3D生成
  - Image-to-3D
  - Flow-Matching
  - Texture-Generation
  - Diffusion-Transformer
paper: https://arxiv.org/abs/2501.12202
code: 官方仅开源推理代码与预训练权重，未发现完整训练代码和训练数据
code_url: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
code_commit: f8db63096c8282cb27354314d896feba5ba6ff8a
rating: 4.5/5
created: 2026-10-30
updated: 2026-10-30
---

# Hunyuan3D 2.0：大规模几何与纹理扩散模型精读

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation* |
| 团队 | Tencent Hunyuan3D Team |
| 时间 | arXiv:2501.12202，2025 |
| 任务 | 单图条件下生成高质量裸网格，再为生成或手工网格合成高分辨率纹理 |
| 核心组件 | Hunyuan3D-ShapeVAE、Hunyuan3D-DiT、Hunyuan3D-Paint、Hunyuan3D-Studio |
| 论文 | [arXiv](https://arxiv.org/abs/2501.12202) |
| 代码 | [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)，检查 commit `f8db63096c8282cb27354314d896feba5ba6ff8a` |
| 开源状态 | **仅推理代码、权重和应用示例**；仓库没有完整训练循环、优化器配置和训练数据 |
| 离线材料 | [论文 PDF](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/sources/paper.pdf) · [来源记录](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/sources.md) |
| 个人评级 | **★★★★☆（4.5/5）**：完整定义了 2.x 的 Shape→Texture 主干，但论文对“Scaling”最关键的数据、参数与算力披露明显不足 |

> [!summary] 一句话结论
> Hunyuan3D 2.0 把 3D 资产生成明确拆成“**Flow-Matching DiT 生成 ShapeVAE latent，再由多视图扩散生成纹理并烘焙到 UV**”两个基础模型：前者用 3072 个无固定空间索引的 latent tokens 表示几何，后者用去光照参考图、normal/position 几何条件、reference attention 与 multi-view attention 合成一致视图。它建立了混元 3D 2.x 的工程骨架，但公开实现与论文在视角选择、超分辨率和训练开放程度上存在重要落差。

---

## 1. 从 1.0 到 2.0：问题定义发生了什么变化

### 1.1 1.0 的核心仍是“先造视图，再重建”

[[Hunyuan3D 1.0 - 统一两阶段三维生成]] 的主体逻辑是：**<span style="color: red">先用多视图扩散模型把一张条件图扩展为六张环轨视图，再用稀疏视图重建模型把这些视图压成 triplane，最后以 SDF 与 Marching Cubes 输出 mesh</span>**。它解决的是单图条件不充分的问题，核心中间表示是**多视图图像与三平面神经场**。

2.0 不再把“多视图生成→重建”作为几何主干，而是转向 3DShape2VecSet / Michelangelo / CLAY 一脉的**3D latent set 生成**：先学习能够压缩 mesh 的 ShapeVAE，再直接在 VAE latent 上训练 image-conditioned Flow-Matching DiT。多视图扩散被保留，但职责移到第二阶段——生成纹理视图，而不是生成几何重建视图。

### 1.2 2.0 真正解决的是“可扩展的完整资产生成”

系统目标从“由文本或图像得到一个可见的 3D 结果”收紧为两个可组合的基础能力：

1. **Image-to-Shape**：参考图像必须生成结构合理、细节丰富的裸三角网格；
2. **Image/Geometry-to-Texture**：参考图像与给定网格共同决定纹理，多视图之间要一致，最终输出 UV texture map。

这种解耦有三个现实收益。第一，几何和纹理的错误不再被迫由同一 latent 同时解释；第二，Paint 可给手工 mesh 重贴图；第三，Shape 与 Paint 可以独立升级，后续 2.1、2.5 正是沿这条接口继续扩展 PBR 与 LATTICE。

### 1.3 核心贡献

1. **高容量 ShapeVAE**：以均匀采样与重要性采样共同构建 point queries，用最长 3072 个 latent tokens 保存边缘、角点与高频几何。
2. **Flow-Matching Hunyuan3D-DiT**：16 个 double-stream blocks 负责图像与形状深度交互，32 个 single-stream blocks 负责联合建模，直接从高斯噪声生成 3D latent set。
3. **双流 Hunyuan3D-Paint**：冻结的 reference branch 保留参考图细节，generation branch 同时接收 normal、canonical coordinate 与 camera ID，并用 reference/multi-view attention 保证对齐和一致性。
4. **Dense-view UV baking**：论文提出从 44 个预设相机中按 UV 新增覆盖率贪心选择 8–12 个视角，再超分辨率和烘焙；公开实现则退化为固定六视角，形成论文与工程版本的重要差异。
5. **生产工具链**：Studio 将生成结果继续接入低模化、草图生成、绑定与动画，但技术报告对这些工具的算法细节披露很少。

---

## 2. 整体架构：Shape 与 Texture 两个基础模型

![](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/figures/01-overall-pipeline.png)

*Figure 2 / PDF p.4。左侧不再依赖多视图重建：DiT 直接生成 ShapeVAE latent；右侧 Paint 通过参考图、normal 与 position maps 生成多视图纹理，再烘焙到 mesh。*

### 2.1 第一阶段：Image → Bare Mesh

完整信息流是：

$$I\xrightarrow{\text{DINOv2-Giant}}C\xrightarrow{\text{Flow-Matching DiT}}Z_s\xrightarrow{\text{ShapeVAE Decoder}}F_{\mathrm{SDF}}\xrightarrow{\text{Marching Cubes}}\mathcal M$$

其中：

- $I$ 是去背景、居中并缩放后的条件图；
- $C$ 是 DINOv2 的图像 tokens；
- $Z_s\in\mathbb R^{L\times d_0}$ 是长度可变的 ShapeVAE latent set；
- $F_{\mathrm{SDF}}$ 是规则三维查询网格上的隐式 SDF；
- $\mathcal M$ 是三角网格。

### 2.2 第二阶段：Image + Mesh → UV Texture

$$\{I,\mathcal M\}\rightarrow I_{\mathrm{delight}},\{N_v,P_v,c_v\}_{v=1}^{V}\xrightarrow{\text{Multi-view Diffusion}}\{T_v\}_{v=1}^{V}\xrightarrow{\text{Baking/Inpainting}}U$$

- $I_{\mathrm{delight}}$ 是去除环境光和阴影后的参考图；
- $N_v$、$P_v$ 是第 $v$ 个相机下的 canonical normal/position map；
- $c_v$ 是离散 camera embedding；
- $T_v$ 是生成的纹理视图；
- $U$ 是最终 UV texture map。

几何先验并非辅助提示，而是第二阶段的坐标锚点：参考图告诉模型“长什么样”，normal 与 position maps 告诉模型“当前像素属于表面哪一处、朝向哪里”。

---

## 3. Hunyuan3D-ShapeVAE：把 mesh 压成可扩散的 latent set

### 3.1 为什么不用规则 voxel 或 triplane

规则 3D 网格的计算量随分辨率立方增长；triplane 虽更紧凑，但以轴对齐平面承载几何，薄结构与局部尖锐细节常被有限分辨率平滑。ShapeVAE 继承 [[3DShape2VecSet - 神经场与扩散生成的三维形状表征]] 的思路：用一组无显式固定位置的 tokens 对输入点云做 cross-attention，把 mesh 压成固定或可变长度的集合表示。

这种表示适配 Transformer 与扩散模型，但新的瓶颈是：如果 point queries 只来自均匀表面采样，大面积平坦区域会占据大量 query，边缘、尖角和小零件反而被稀释。

### 3.2 均匀采样 + 重要性采样

![](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/figures/02-shape-vae.png)

*Figure 3 / PDF p.5。均匀点与边缘/角点重要性点分别执行 FPS，再合并为 encoder queries；decoder 以规则 grid queries 查询 latent，并输出 SDF。*

给定 mesh，论文构造两组带法向的表面点：

$$P_u\in\mathbb R^{M\times3},\qquad P_i\in\mathbb R^{N\times3}$$

- $P_u$：均匀表面采样；
- $P_i$：对边缘、角点等高复杂区域的重要性采样。

然后分别执行 Farthest Point Sampling：

$$Q_u\in\mathbb R^{M'\times3},\qquad Q_i\in\mathbb R^{N'\times3}$$

再拼接输入点和 query：

$$P=[P_u;P_i],\qquad Q=[Q_u;Q_i]$$

**<span style="color: red">“分别 FPS、最后合并”很关键。如果先混合再 FPS，高密度重要性点仍可能被均匀覆盖目标压掉</span>**；分别 FPS 等于显式给高频区域保留 token 预算。发布配置将 `num_latents` 设为 3072，而 `ShapeVAE` 类默认的 uniform/sharp-edge `pc_size` 均为 5120；但 encoder 中 `downsample_ratio=20` 与这组输入点数静态不自洽，且公开推理不调用 encoder。因此“5120+5120 点经过 FPS 得到 3072 queries”只能作为论文训练流程描述，不能说已由发布代码验证。

### 3.3 Encoder：Point Cross-Attention + 8 层 Self-Attention

点坐标经过 Fourier positional encoding 与线性映射：

$$X_p\in\mathbb R^{(M+N)\times d},\qquad X_q\in\mathbb R^{(M'+N')\times d}$$

query 对点云特征做一次 cross-attention，随后经过 8 层 self-attention，得到：

$$H_s\in\mathbb R^{(M'+N')\times d}$$

VAE 的 `pre_kl` 线性层预测均值和对数方差；发布配置下：

$$[B,3072,1024]\rightarrow[B,3072,128]\rightarrow\{\mu,\log\sigma^2\}\in\mathbb R^{B\times3072\times64}$$

即每个几何 token 的 VAE latent width 是 64，而 Transformer 内部 width 是 1024。

### 3.4 Decoder：latent set → SDF → mesh

采样出的 $Z_s$ 先从 64 维投影回 1024 维，经 16 层 self-attention，然后使用规则 3D 网格点作为 queries：

$$Q_g\in\mathbb R^{(HWD)\times3}$$

这些 3D queries 通过 cross-attention 从 latent set 读取局部隐式场特征，线性层输出每个查询点的 SDF，最后 Marching Cubes 提取零等值面。

论文把 grid feature 的部分符号写成 $F_n\times W\times D$、$F_o\times W\times D$，其中 $F_n,F_o$ 未定义，且与此前 $H\times W\times D$ 不一致，应视为排版错误而不是新的维度设计。

### 3.5 训练目标与可变 token length

ShapeVAE 的损失是 SDF 重建与 KL 正则：

$$\mathcal L_r=\mathbb E_{x\in\mathbb R^3}\left[\operatorname{MSE}\left(D_s(x\mid Z_s),\operatorname{SDF}(x)\right)\right]+\gamma\mathcal L_{\mathrm{KL}}$$

完整三维 SDF 代价过高，因此期望通过随机空间点与近表面点估计。论文还在训练时随机抽取预定义 token lengths：短序列降低计算，长序列提升重建，发布版最长 3072。

遗憾的是，论文没有公布 $\gamma$、SDF 截断范围、空间点与近表面点数量、token length 集合、optimizer、学习率和训练步数。这使 ShapeVAE 无法仅凭论文复现。

---

## 4. Hunyuan3D-DiT：在无固定位置的几何 tokens 上做 Flow Matching

### 4.1 图像条件

输入图像先去背景、统一前景尺度、居中并填白，再送入 DINOv2-Giant。论文使用 $518\times518$ 图像以及最后一层全部 patch tokens，并保留 CLS token。

公开实现中：

- patch size：14；
- patch grid：$37\times37=1369$；
- 加 1 个 CLS token；
- 条件序列：$[1,1370,1536]$。

这种全 patch 条件与单个全局 CLIP embedding 相比，更能保留局部形状对应关系，但它仍是注意力学习出的隐式 2D→3D 对齐，不具备 Pixal3D 那种显式射线锚定。

### 4.2 Double-stream 与 Single-stream

![](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/figures/03-shape-dit.png)

*Figure 4 / PDF p.6。前 16 层让 image/shape 保留独立归一化、QKV 与 MLP，后 32 层把二者拼接到同一序列，完成更深联合建模。*

真实发布配置为：

| 项目 | 数值 |
|---|---:|
| Shape latent | $3072\times64$ |
| Transformer width | 1024 |
| Attention heads | 16 |
| Head dimension | 64 |
| Double-stream blocks | 16 |
| Single-stream blocks | 32 |
| MLP ratio | 4 |
| Image condition width | 1536 |

Double-stream block 中 image 与 shape 各自做 LayerNorm、timestep scale/shift、QKV 和 MLP，但两者的 Q/K/V 会沿 token 维拼接，在同一个 attention 中交互。这样既保留不同模态的统计尺度，又允许图像条件直接影响几何 token。

随后 image 与 shape tokens 拼成长度 $1370+3072=4442$ 的序列，进入 32 个 single-stream blocks。最终裁掉前 1370 个图像 tokens，仅保留形状部分并线性映射回 64 维速度预测。

### 4.3 为什么不加 3D positional embedding

论文明确指出，第 $k$ 个 ShapeVAE token 并不对应固定体素位置，token 顺序没有“左上、右下”之类的含义；几何位置/占据信息由 token 内容本身编码。因此 DiT 对 shape sequence 不加 positional embedding。

这不等于模型被严格证明为 permutation invariant，但它避免人为向 latent set 注入不存在的规则网格顺序。它与 [[LATTICE - 大规模高保真三维生成表征]] 的分歧也很清楚：Hunyuan3D 2.0 接受“位置由内容隐式携带”，而 LATTICE 认为这种不可定位性正是 VecSet 无法稳定 scaling 的根源，于是用 voxel queries 与 3D RoPE 显式锚定位置。

### 4.4 Flow Matching 目标

定义噪声 $x_0\sim\mathcal N(0,I)$ 与数据 latent $x_1$ 之间的线性路径：

$$x_t=(1-t)x_0+tx_1,\qquad t\sim U(0,1)$$

真实速度恒为：

$$u_t=\frac{dx_t}{dt}=x_1-x_0$$

DiT 学习条件速度场：

$$\mathcal L=\mathbb E_{t,x_0,x_1}\left[\left\|u_\theta(x_t,c,t)-u_t\right\|_2^2\right]$$

推理时从 $x_0$ 出发，以一阶 Euler ODE solver 积分到数据端。公开标准路径使用 50 步、CFG 5.0；Turbo 形状模型使用 5 步。论文只写“Steps × N”，没有披露 $N$ 和 guidance，因此真实数字必须由代码补齐。

---

## 5. Hunyuan3D-Paint：双流参考注入与多视图一致纹理

![](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/figures/04-paint-architecture.png)

*Figure 5 / PDF p.7。reference branch 在 $t=0$ 读取无噪参考 latent；generation branch 同时接收 normal/position，并并行执行 self、reference 与 multi-view attention。*

### 5.1 Image Delight：先把“光照”从“材质”中尽量分开

用户图像中的阴影、高光和环境照明如果直接被当作纹理，会在新视角与新光照下形成“双重光照”。论文用同一 3D 资产在随机 HDRI 与均匀白光下渲染成对数据，训练 image-to-image Delight 模块，把参考图变为更接近 unlit 的外观。

但这里要谨慎：Delight 并不是严格的 inverse rendering，也没有显式估计光源、BRDF 或 intrinsic albedo。论文未给架构、数据规模、损失和量化指标，因此它更像一种学习式外观归一化。金属、高光、透明材质仍可能被错误抹除或烘焙。

### 5.2 论文中的几何感知视角选择

论文先放入 4 个正交视角，再从候选集里选择能带来最大新增 UV 覆盖的视角，最多 12 个。候选视角 $v_i$ 的得分为：

$$\mathcal F(v_i,V_s,M)=\mathcal A_{\mathrm{area}}\left(\mathcal{UV}_{\mathrm{cover}}(v_i,M)\setminus\bigcup_{v_s\in V_s}\mathcal{UV}_{\mathrm{cover}}(v_s,M)\right)$$

直觉上，这不是追求“相机均匀分布”，而是追求“每增加一个视角，尽可能多覆盖尚未上色的 UV texels”。它对凹结构和非对称对象尤其合理。

不过公开默认实现没有这套贪心算法，而是固定六视角：

- azimuth：$[0,90,180,270,0,180]$；
- elevation：$[0,0,0,0,90,-90]$。

这意味着论文所宣称的 8–12 dense views 与默认代码并不一致。

### 5.3 Double-stream Reference-Net

参考图经过 VAE 得到无噪 latent，reference branch 的 timestep 固定为 0。与常见共享权重 ReferenceNet 不同，作者保留并冻结原始 SD2.1 权重作为参考分支，在各 self-attention 层前缓存 reference features；generation branch 再通过 reference attention 读取这些特征。

冻结的意义不是省参数，而是把 reference branch 当作 soft regularizer：它保持在自然图像分布上学到的特征，不随合成 3D render 训练数据一起漂移。公开 checkpoint 中也确实同时保存 generation UNet 与 reference UNet，但训练代码未开源，无法验证论文所说的冻结策略是否完整对应发布模型。

### 5.4 Multi-task Attention

生成分支将三类 attention 并行相加：

$$Z_{\mathrm{MVA}}=Z_{\mathrm{SA}}+\lambda_{\mathrm{ref}}\operatorname{Softmax}\left(\frac{Q_{\mathrm{ref}}K_{\mathrm{ref}}^T}{\sqrt d}\right)V_{\mathrm{ref}}+\lambda_{\mathrm{mv}}\operatorname{Softmax}\left(\frac{Q_{\mathrm{mv}}K_{\mathrm{mv}}^T}{\sqrt d}\right)V_{\mathrm{mv}}$$

- self-attention：维持单视图内部结构；
- reference attention：复制参考图的身份、图案和局部细节；
- multi-view attention：把多个视图的空间 tokens 拼在一起，减少 Janus、色彩漂移和纹理断裂。

论文没有披露 $\lambda_{\mathrm{ref}}$ 和 $\lambda_{\mathrm{mv}}$，也没有定量拆分三种 attention 的独立贡献。

### 5.5 Geometry 与 Camera Conditioning

每个视角渲染：

1. canonical normal map；
2. canonical coordinate/position map。

二者分别经过 VAE，与 noisy RGB latent 在通道维拼接。公开实现中：

$$[1,6,4,64,64]_{\mathrm{noise}}\oplus[1,6,4,64,64]_{\mathrm{normal}}\oplus[1,6,4,64,64]_{\mathrm{position}}$$

得到：

$$[1,6,12,64,64]\rightarrow[6,12,64,64]$$

normal 提供局部朝向，position map 在 canonical object coordinates 中给不同视图建立像素身份对应。camera 则不是连续外参矩阵，而是 44 个预设视角对应的 learned integer embedding。

### 5.6 UV baking 与补洞

论文路径是：多视图 RGB → 单图超分辨率 → 投影到 UV → 顶点颜色传播补洞。公开实现的真实路径为：

1. `xatlas` 展开 UV；
2. 六张 $512^2$ 生成图直接 resize 到 $2048^2$，单图超分代码被注释；
3. 按视角权重和 $\cos^4\theta$ 融合回 $2048^2$ UV texture；
4. UV texture 转 vertex color，在网格邻接图上传播；
5. 再执行 OpenCV Navier–Stokes inpaint，半径为 3。

论文说未覆盖 texel 使用几何距离倒数加权，代码实际是 $1/d^2$。这些差异说明：论文给的是完整研究管线，仓库默认路径更偏向可部署的固定流程。

---

## 6. 训练数据与训练设置

### 6.1 论文明确披露的部分

| 模块 | 披露信息 |
|---|---|
| ShapeVAE | 均匀+重要性点采样；SDF MSE + KL 正则；可变 token length；最长 3072 |
| Shape DiT | DINOv2-Giant；$518^2$ 输入；16 double + 32 single；Flow Matching；Euler solver |
| Delight | 同一资产的随机 HDRI 与均匀白光成对渲染 |
| Paint 数据 | self-collected large-scale 3D dataset；白光渲染；reference azimuth 随机；elevation 为 $[-20^\circ,20^\circ]$ |
| Paint 训练 | SD2.1 v-model ZSNR checkpoint；$512^2$；80,000 steps；batch 48；LR $5\times10^{-5}$；1000-step warm-up；trailing scheduler |

### 6.2 “Scaling”论文最严重的披露缺口

标题强调 Scaling，但论文没有给出：

- Shape 训练集数量、来源占比、清洗和去重；
- ShapeVAE/DiT optimizer、batch size、学习率、训练步数；
- GPU 型号、数量、训练时长与总 FLOPs；
- ShapeVAE KL 权重、SDF 采样数量和网格分辨率；
- Paint 数据规模、类别分布、许可与测试划分；
- 完整模型参数量和不同规模模型的 scaling 曲线。

因此，这篇论文证明了“大模型版本效果强”，但没有真正提供“规模如何推动性能”的可审计证据。后续 [[LATTICE - 大规模高保真三维生成表征]] 才把 0.6B→4.5B 的 scaling 行为作为核心实验系统展示。

---

## 7. 实验结果

### 7.1 ShapeVAE 重建

所有 VecSet 模型使用 1024 tokens，Direct3D 使用 3072 tokens：

| 方法 | V-IoU ↑ | S-IoU ↑ |
|---|---:|---:|
| 3DShape2VecSet | 87.88% | 80.66% |
| Michelangelo | 84.93% | 76.27% |
| Direct3D | 88.43% | 81.55% |
| **Hunyuan3D-ShapeVAE** | **93.60%** | **89.16%** |

相对最佳 baseline Direct3D，V-IoU 提升 5.17 个百分点，近表面 S-IoU 提升 7.61 个百分点。更大的 S-IoU 增益与“重要性采样保留边缘/角点”的设计吻合，但论文没有给出“去掉 importance sampling”的控制变量，因而不能把全部增益单独归因于该模块。

### 7.2 Image-to-Shape 条件对齐

| 方法 | ULIP-T ↑ | ULIP-I ↑ | Uni3D-T ↑ | Uni3D-I ↑ |
|---|---:|---:|---:|---:|
| Michelangelo | 0.0752 | 0.1152 | 0.2133 | 0.2611 |
| Craftsman 1.5 | 0.0745 | 0.1296 | 0.2375 | 0.2987 |
| Trellis | 0.0769 | 0.1267 | 0.2496 | 0.3116 |
| Shape Model 1 | **0.0799** | 0.1181 | 0.2469 | 0.3064 |
| Shape Model 2 | 0.0741 | **0.1308** | 0.2464 | 0.3106 |
| Shape Model 3 | 0.0746 | 0.1284 | 0.2516 | 0.3131 |
| **Hunyuan3D-DiT** | 0.0771 | 0.1303 | **0.2519** | **0.3151** |

Hunyuan3D-DiT 并不是四项全第一：它在 Uni3D-T/I 第一，ULIP-T/I 分别略低于匿名闭源模型。更严谨的表述应是“综合对齐能力领先”，而不是“所有指标全面超过”。

![](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/figures/05-shape-comparison.png)

*Figure 7 / PDF p.12。高频结构、人脸与复杂标志的视觉对比体现了 ShapeVAE 容量和 DiT 条件对齐，但成功样例不能代替随机测试集和失败案例。*

### 7.3 纹理生成

| 方法 | CMMD ↓ | FID$_{\mathrm{CLIP}}$ ↓ | CLIP-score ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| TEXTure | 3.047 | 35.75 | 0.8499 | 0.0076 |
| Text2Tex | 2.811 | 31.72 | 0.8680 | 0.0071 |
| SyncMVD | 2.584 | 29.93 | 0.8751 | 0.0063 |
| Paint3D | 2.810 | 30.29 | 0.8724 | 0.0063 |
| TexPainter | 2.483 | 28.83 | 0.8789 | 0.0062 |
| **Hunyuan3D-Paint** | **2.318** | **26.44** | **0.8893** | **0.0059** |

相对 TexPainter，CMMD 降低约 6.65%，FID$_{\mathrm{CLIP}}$ 降低约 8.29%。这组结果支持 Paint 在语义与局部纹理质量上的优势，但测试对象数、prompt、视图数和各 baseline 参数均未披露。

![](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/figures/06-texture-comparison.png)

*Figure 8 / PDF p.14。Paint 在鱼鳞、兔子毛色、复杂服饰和整体一致性上更稳，但论文没有展示强高光、透明、镜面和严重遮挡的系统失败案例。*

### 7.4 端到端 textured asset

| 方法 | CMMD ↓ | FID$_{\mathrm{CLIP}}$ ↓ | FID$_{\mathrm{Incept}}$ ↓ | CLIP-score ↑ |
|---|---:|---:|---:|---:|
| Trellis | 3.591 | 54.639 | 289.287 | 0.787 |
| Model 1 | 3.600 | 55.866 | 305.922 | 0.779 |
| Model 2 | 3.368 | 49.744 | 294.628 | 0.806 |
| Model 3 | 3.218 | 51.574 | 295.691 | 0.799 |
| **Hunyuan3D 2.0** | **3.193** | **49.165** | **282.429** | **0.809** |

这张表的优势存在但幅度并不都大，例如 CMMD 相比 Model 3 只低 0.025，CLIP-score 只高 0.003。正文声称使用 LPIPS，表格却没有 LPIPS，而多出 FID$_{\mathrm{Incept}}$；caption 还将端到端结果写成 Paint 结果，是明确的报告不一致。

### 7.5 用户研究与消融缺失

50 名志愿者比较 300 个未筛选结果，Hunyuan3D 2.0 在整体满意度、资产质量与图像跟随上柱图最高，但论文没有公开每个样本评分数、盲测协议、精确票数、置信区间或显著性检验。

更重要的是，全文**没有正式消融表**。以下关键问题都没有量化回答：

- 重要性采样单独提升多少；
- 3072 tokens 相比 1024 tokens 提升多少；
- double-stream 和 single-stream 各自贡献；
- Delight、normal、position、camera embedding 各自贡献；
- reference attention 与 multi-view attention 各自贡献；
- 8–12 视角选择、超分辨率和 UV 补洞分别贡献多少。

因此，主结果证明“系统有效”，却没有充分证明“每个设计为什么有效”。

---

## 8. 代码级实现：按真实推理执行顺序拆解

本节不再按“模块名列表”泛讲，而是按一次 `image → bare mesh → textured mesh` 的真实调用顺序展开。张量默认取发布配置、batch size $B=1$；开启 CFG 后的临时 batch 会显式写出。证据分三类：**代码明确**表示固定 commit 源码直接定义，**配置推导**表示由官方配置与 reshape 确定，**论文描述**表示训练代码未公开、只能由论文确认。

### 8.1 仓库概览与开源边界

| 项目 | 状态 |
|---|---|
| 推理代码 | 已开源 |
| 预训练权重 | 已开源 |
| Shape/Paint/Delight pipeline | 已开源 |
| Gradio、API、Blender 示例 | 已开源 |
| 完整训练循环 | **未发现** |
| optimizer / dataloader / backward | **未发现** |
| 训练数据 | **未提供** |
| README 所称 fine-tuning | Open-Source Plan 已勾选，但当前树中未找到完整可执行链路 |

`NOTICE` 将 Hunyuan 3D 2.0 明确定义为 inference-enabling code、parameters 与 weights。许可证还包含地域和规模限制，并非 Apache/MIT：适用地域排除欧盟、英国和韩国；月活超过 100 万的产品需另行申请；还限制使用模型或输出训练其他非 Hunyuan3D 模型。工程使用前必须审阅原始 LICENSE。

### 8.2 论文模块 → 代码入口

| 论文模块 | 代码文件 / 类 | 产物 |
|---|---|---|
| 总体 Shape pipeline | `hy3dgen/shapegen/pipelines.py` / `Hunyuan3DDiTFlowMatchingPipeline` | `trimesh` 裸网格 |
| 图像预处理 | `shapegen/preprocessors.py` / `ImageProcessorV2` | $[1,3,512,512]$ 图像与 mask |
| DINO 条件编码 | `shapegen/models/conditioner.py` / `DinoImageEncoder` | $[1,1370,1536]$ |
| Shape DiT | `models/denoisers/hunyuan3ddit.py` / `Hunyuan3DDiT` | $[1,3072,64]$ velocity |
| Flow scheduler | `shapegen/schedulers.py` / `FlowMatchEulerDiscreteScheduler` | Euler 更新后的 latent |
| ShapeVAE | `models/autoencoders/model.py` / `ShapeVAE` | SDF query decoder |
| Volume decoder | `models/autoencoders/volume_decoders.py` | $385^3$ 默认稠密 grid |
| Surface extraction | `surface_extractors.py` / `MCSurfaceExtractor` | vertices/faces |
| Paint pipeline | `texgen/pipelines.py` / `Hunyuan3DPaintPipeline` | 带 UV 纹理网格 |
| Delight | `texgen/utils/dehighlight_utils.py` | 去光照参考图 |
| 2.5D UNet | `texgen/hunyuanpaint/unet/modules.py` | 六视图噪声预测 |
| UV 展开 | `texgen/utils/uv_warp_utils.py` / `xatlas.parametrize` | UV mesh |
| Baking | `differentiable_renderer/mesh_render.py` | $2048^2$ texture map |

### 8.3 全链路张量总表

| # | 阶段 | 输入 | 输出 | 真实默认 |
|---:|---|---|---|---|
| 1 | Shape 预处理 | RGBA/RGB 图像 | image $[1,3,512,512]$，mask $[1,1,512,512]$ | 前景占画布 85%，白底，归一化到 $[-1,1]$ |
| 2 | DINOv2-Giant | $[1,3,518,518]$ | $[1,1370,1536]$ | $37^2$ patches + CLS |
| 3 | CFG 扩展 | condition | $[2,1370,1536]$ | cond + zero uncond |
| 4 | Shape 初始噪声 | 高斯噪声 | $[1,3072,64]$ | 标准 50 步，CFG 5.0 |
| 5 | DiT double stream | shape $[2,3072,1024]$ + image $[2,1370,1024]$ | 两条更新序列 | 16 blocks |
| 6 | DiT single stream | $[2,4442,1024]$ | shape velocity $[2,3072,64]$ | 32 blocks |
| 7 | VAE latent | $[1,3072,64]$ | $[1,3072,1024]$ | 16 层 decoder self-attention |
| 8 | SDF queries | $57,066,625\times3$ | $[1,385,385,385]$ | `octree_resolution=384` 实为稠密网格 |
| 9 | Marching Cubes | SDF grid | vertices $[N_v,3]$、faces $[N_f,3]$ | 数量随对象变化 |
| 10 | Paint 几何渲染 | 六相机 + mesh | normal/position 各 $[1,6,3,512,512]$ | 原始渲染 $2048^2$ 后缩放 |
| 11 | Paint VAE | 六视图条件 | noise/normal/position 各 4 通道 latent | 拼成 $[6,12,64,64]$ |
| 12 | Paint UNet | $[6,12,64,64]$ + reference | RGB latent $[6,4,64,64]$ | Turbo 实际 10 次循环 |
| 13 | VAE decode | $[6,4,64,64]$ | 六张 $512^2$ RGB | 直接 resize 为 $2048^2$ |
| 14 | UV baking | 六张图 + cosine maps | texture $[2048,2048,3]$ | $\cos^4$ 加权 + 补洞 |

### 8.4 Shape 第一步：图像预处理与 DINOv2-Giant 条件

#### 8.4.1 RGBA 裁剪、居中与归一化

入口是 `Hunyuan3DDiTFlowMatchingPipeline.prepare_image()` 调用 `ImageProcessorV2`。输入是任意尺寸的 PIL RGB/RGBA 图像 $[H,W,3/4]$；代码先统一转 RGBA，根据 alpha 非零区域取 bbox，把物体最长边缩放到方形画布的 85%，与白底合成，再 resize 到 $512\times512$，最后执行 HWC→CHW 与 $x/255\times2-1$：

$$[H,W,3/4]\rightarrow I\in\mathbb R^{1\times3\times512\times512},\quad M\in\mathbb R^{1\times1\times512\times512}$$

这么做的目的不是“增强图片”，而是把前景尺度和背景统计固定下来，避免 DINO patch 被大片背景浪费。需要注意两点：普通 RGB 会被补成全不透明 alpha，因此本类不会自动语义抠图；`mask` 虽传给 conditioner，DINO encoder 实际没有使用它。Gradio/API 会显式调用 rembg，而 README 示例在 `.convert("RGBA")` 后再判断 RGB，分支不可达。

#### 8.4.2 DINOv2-Giant：二维 patch tokens 是什么 shape

`DinoImageEncoder` 把 $512^2$ 图像 resize/crop 为 $518^2$，使用 DINOv2-Giant：patch size 14、40 层、hidden 1536、24 heads。$518/14=37$，保留一个 CLS token，因此：

$$[1,3,512,512]\rightarrow[1,3,518,518]\rightarrow[1,1+37^2,1536]=[1,1370,1536]$$

外部模型 DINOv2 在这里不是生成器，而是冻结的视觉条件编码器：1370 个 token 保留局部 patch 信息，1536 是每个 token 的语义宽度。代码开启 CFG 时构造同 shape 的全零 unconditional tokens，并按 `[conditional, unconditional]` 拼接：

$$C_{\mathrm{img}}:[1,1370,1536]\rightarrow[2,1370,1536]$$

这与常见 Stable Diffusion 的 `[uncond, cond]` 顺序相反，后续切 batch 时也按对应顺序处理。

### 8.5 Shape 第二步：从高斯噪声到 $3072\times64$ 几何 latent

#### 8.5.1 初始噪声与统一 hidden width

`prepare_latents()` 直接采样：

$$x_0\sim\mathcal N(0,I),\qquad x_0\in\mathbb R^{1\times3072\times64}$$

这里 3072 是 ShapeVAE token 数，64 是每个 token 的 KL latent 维度。每个 denoise step 为 CFG 临时把 latent 复制为 $[2,3072,64]$。`Hunyuan3DDiT.forward()` 分别投影：

$$[2,3072,64]\xrightarrow{W_x}[2,3072,1024]$$

$$[2,1370,1536]\xrightarrow{W_c}[2,1370,1024]$$

标量 timestep 则先做 256 维 sinusoidal embedding，再经 MLP 得到 $[2,1024]$ 的调制向量。把三类输入都投到 width 1024，是为了让 attention 可以共享 head 配置；但图像与几何在前 16 层仍保留独立 LayerNorm、QKV 与 MLP。

#### 8.5.2 16 个 double-stream blocks：先保留模态边界再交互

对 shape stream 和 image stream 分别计算：

$$Q,K,V:[2,L,1024]\rightarrow[2,16,L,64]$$

再沿 token 维拼接：

$$[2,16,3072,64]\oplus[2,16,1370,64]\rightarrow[2,16,4442,64]$$

共享 attention 后再切回 $3072$ 个 shape tokens 与 $1370$ 个 image tokens，两个序列的 shape 均不变。这样设计的原因是：DINO feature 与 ShapeVAE latent 的统计分布不同，不能一开始就共用全部归一化和 MLP；但 QKV 联合 attention 又允许每个几何 token 读取所有图像 patch。

#### 8.5.3 32 个 single-stream blocks：把两个模态变成一个联合序列

前 16 层结束后，代码沿 token 维拼成：

$$[2,1370,1024]\oplus[2,3072,1024]\rightarrow[2,4442,1024]$$

每个 single-stream block 用一次线性层同时产生 QKV 和 MLP 分支：QKV 为 $3\times1024$，MLP hidden 为 $4\times1024=4096$；attention 输出 1024 维，与 GELU 后的 4096 维 MLP 特征拼接，再由 $5120\rightarrow1024$ 投影回主干。32 层结束后裁掉前 1370 个 image tokens，只保留 shape 部分并执行 $1024\rightarrow64$：

$$[2,4442,1024]\rightarrow[2,3072,1024]\rightarrow\hat u_\theta\in\mathbb R^{2\times3072\times64}$$

#### 8.5.4 CFG 与 Euler Flow 更新

按代码的 `[cond, uncond]` 顺序切开速度后：

$$\hat u=\hat u_{\mathrm{uncond}}+5.0(\hat u_{\mathrm{cond}}-\hat u_{\mathrm{uncond}})$$

Scheduler 以：

$$x_{\mathrm{next}}=x+(\sigma_{\mathrm{next}}-\sigma)\hat u_\theta(x,\sigma,c)$$

执行 Euler 更新。标准路径调用 DiT 50 次，但 sigma 序列包含端点，最后一次有 $\sigma_{\mathrm{next}}-\sigma=0$，因此是 50 次模型评估、49 次非零状态更新。最终输出仍是：

$$Z_s\in\mathbb R^{1\times3072\times64}$$

Shape DiT 本体约 1.113B；完整 checkpoint 还包含 DINOv2-Giant 与 ShapeVAE。Turbo shape 示例将模型评估降到 5 次，并可配合 FlashVDM。

### 8.6 Shape 第三步：$3072\times64$ latent 如何变成 mesh

#### 8.6.1 ShapeVAE decoder 只走解码侧

公开推理不会调用 ShapeVAE encoder，而是把 DiT 生成的 latent 除以 `scale_factor=0.9990943042622529`，再经 `post_kl` 从 64 维投影回 1024 维，并通过 16 层 latent self-attention：

$$[1,3072,64]\rightarrow[1,3072,1024]$$

64 维用于降低扩散成本，1024 维用于恢复隐式场查询所需的容量。发布配置中 encoder 的默认 `pc_size/downsample_ratio` 组合静态上并不自洽，而且推理根本不走 encoder，因此本笔记不再把“5120+5120 点如何编码成 3072 tokens”写成已由发布代码验证的训练链；那一部分只能依据论文描述。

#### 8.6.2 默认所谓 `octree_resolution=384` 实际是完整稠密查询

`VanillaVolumeDecoder` 创建：

$$(384+1)^3=385^3=57,066,625$$

个三维坐标，展平为 $[57,066,625,3]$，再按最多 8000 点分块。每个 query 的 xyz 经过 8 频带 Fourier embedding：

$$3\rightarrow3(2\times8+1)=51\rightarrow1024$$

随后 query 对 $[1,3072,1024]$ latent 做 cross-attention，线性层输出每点一个无激活标量：

$$[1,P,3]\rightarrow[1,P,1024]\rightarrow[1,P,1],\qquad P\le8000$$

拼接后得到 $[1,385,385,385]$。论文将该标量定义为 SDF，推理代码变量则写作 `occ/logits` 且没有距离标定；因此严谨说法是“论文语义下的 SDF/代码层的隐式标量场”，不能仅凭推理代码证明它是 metric SDF。

#### 8.6.3 Marching Cubes

`MCSurfaceExtractor` 对零等值面执行 Marching Cubes：

$$[1,385,385,385]\rightarrow V\in\mathbb R^{N_v\times3},\quad F\in\mathbb N^{N_f\times3}$$

$N_v,N_f$ 随对象变化。参数名虽然叫 `octree_resolution`，标准路径并不是 sparse octree；只有显式开启 hierarchical decoding 或 FlashVDM 才会进行近表面筛选。

### 8.7 Paint 第一步：参考图、网格与六视图几何条件

#### 8.7.1 默认加载的是 Turbo Paint

`Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")` 默认子目录是 `hunyuan3d-paint-v2-0-turbo`，因此 README 最短调用实际组合为“标准 50-eval Shape + Turbo Paint”，不是论文中的标准 Paint。

#### 8.7.2 Delight 是外部图像到图像扩散模型

输入参考图先根据 alpha bbox 居中到 $512^2$，mask 腐蚀后白底合成，再进入 `StableDiffusionInstructPix2PixPipeline`：空 prompt、50 steps、`image_guidance_scale=1.5`、`guidance_scale=1.0`。输出仍为一张 $512^2$ RGB 图，再按输入前景的 RGB 均值/方差做颜色校正。

因此 Delight 的接口可写为：

$$I_{\mathrm{ref}}\in\mathbb R^{1\times3\times512\times512}\rightarrow I_{\mathrm{delight}}\in\mathbb R^{1\times3\times512\times512}$$

它的作用是学习式去阴影/高光，不是显式 BRDF 分解。当前保存材料没有 Delight 内部 UNet/VAE config，不能可靠补写其 latent shape。

#### 8.7.3 UV、坐标归一化与几何渲染

`xatlas.parametrize` 展开 UV，seam 会复制顶点，所以 $N'_v$ 可能大于原网格顶点数。mesh 经坐标轴调整、bbox 居中，并按最大顶点半径归一化到 `scale_factor=1.15`。默认相机为：

$$\mathrm{azimuth}=[0,90,180,270,0,180]$$

$$\mathrm{elevation}=[0,0,0,0,90,-90]$$

先以 $2048^2$ 渲染 canonical normal/position，再缩放为扩散输入：

$$N,P\in\mathbb R^{1\times6\times3\times512\times512}$$

normal 表示局部朝向；position 是归一化 object-space 坐标，用于告诉不同视图中的像素是否来自同一表面区域。

### 8.8 Paint 第二步：SD VAE、双 UNet 与多视图 attention

#### 8.8.1 外部 SD VAE 的输入输出

参考图、normal 和 position 都先从 $[0,1]$ 映射到 $[-1,1]$，展平 view 维后进入 `AutoencoderKL.encode()`，从 posterior 采样并乘 scaling factor：

$$[1,1,3,512,512]\rightarrow Z_{\mathrm{ref}}\in\mathbb R^{1\times1\times4\times64\times64}$$

$$[1,6,3,512,512]\rightarrow Z_N,Z_P\in\mathbb R^{1\times6\times4\times64\times64}$$

VAE 的 8 倍空间下采样把 $512^2$ 变成 $64^2$，每个 latent 有 4 通道。三类输入落到相同 latent grid，才能直接沿通道维拼接。

#### 8.8.2 Camera embedding 的精确含义

代码使用 `nn.Embedding(49,1280)`，不是“49 个生成视角”：其中 5 个是 reference 保留位，44 个是 generation camera classes。默认六视图的原始 ID 为 $[21,12,15,18,43,37]$，generation UNet 查表前统一加 5，实际索引是 $[26,17,20,23,48,42]$。相机向量与 timestep embedding 同为 1280 维，用于调制 generation UNet；reference UNet 没有安装同一 camera embedding。

#### 8.8.3 Generation UNet 的 12 通道输入

Turbo 初始 RGB noise 为：

$$Z_t\in\mathbb R^{1\times6\times4\times64\times64}$$

与 normal/position latent 拼接：

$$[1,6,4,64,64]^{\oplus3}\rightarrow[1,6,12,64,64]\rightarrow[6,12,64,64]$$

`conv_in` 再把 12 通道映射为 $[6,320,64,64]$。公开 UNet 是 SD2.x 风格，四级 channels 为 $[320,640,1280,1280]$，cross-attention width 1024。Turbo 已蒸馏 guidance，不复制 CFG batch；标准非 Turbo 才把 batch 复制为 2。

#### 8.8.4 Reference UNet 每个 denoise step 都会重新执行

`Z_{\mathrm{ref}}:[1,1,4,64,64]\rightarrow[1,4,64,64]`，reference timestep 固定为 0。独立 `unet_dual` 在各 attention 层保存 reference hidden states，generation UNet 随后以 reference attention 读取。缓存只在**一次 UNet forward 内**有效，不会跨十个 denoise step 持久化，因此 Turbo 中 reference UNet 同样执行十次。

论文声称训练时冻结 SD2.1 reference branch；公开推理仓库只能确认双 UNet 和推理时 `no_grad`，不能验证训练期间如何冻结。

#### 8.8.5 三类 attention 的真实 reshape

设某层每个视图有 $L$ 个空间 tokens、channel 为 $C$，generation hidden 为 $[6,L,C]$：

1. self-attention 在每个视图内部执行，shape 不变；
2. reference feature 从 $[1,L_r,C]$ 复制为 $[6,L_r,C]$，每个生成视图单独读取同一参考；
3. multi-view attention 把六视图拼为 $[1,6L,C]$，交换跨视图信息，再 reshape 回 $[6,L,C]$。

三项 residual contribution 都加回主干：self 维持单视图结构，reference 传递身份与局部图案，multi-view 减少视图间颜色漂移。Turbo pipeline 虽计算 position attention mask/voxel indices，但固定 commit 中 `UNet2p5DConditionModel.is_turbo` 没有同步为 `True`，这些 mask 确定没有进入 transformer block；原笔记中“待运行复核”的表述应改为静态可确认的失效路径。

### 8.9 Paint 第三步：10 次采样、VAE 解码与 UV 烘焙

#### 8.9.1 Turbo 的真实循环次数

外层传 `num_inference_steps=30`，实际只索引：

$$[29,26,23,20,17,14,11,8,5,2]$$

因此 generation/reference UNet 各执行 10 次，latent 始终为 $[6,4,64,64]$；进度条仍显示 30。learned text embedding 为 $[1,77,1024]$，不是用户输入文本 prompt。

#### 8.9.2 SD VAE decode

最终 latent 除以 scaling factor，经 `AutoencoderKL.decode()`：

$$[6,4,64,64]\rightarrow[6,3,512,512]$$

得到六张 RGB 纹理视图。默认代码没有运行单图超分模型，只用 PIL 把它们 resize 为 $2048^2$。

#### 8.9.3 反投影、加权融合与补洞

每个视图反投影到 $2048^2$ UV atlas，过滤夹角大于 $75^\circ$、深度边界和 silhouette 附近像素；权重为：

$$w_v=w_v^{\mathrm{preset}}\cos^4\theta,\
\qquad w^{\mathrm{preset}}=[1,0.1,0.5,0.1,0.05,0.05]$$

六张 projected textures 加权平均得到：

$$U\in\mathbb R^{2048\times2048\times3},\quad M_{\mathrm{trust}}\in\mathbb R^{2048\times2048\times1}$$

未覆盖 texels 先转成 vertex colors，在 mesh 邻接图上以 $1/d^2$ 权重传播，再写回 UV，最后用 OpenCV Navier–Stokes、半径 3 补剩余空洞。几何传播优先保持跨 UV island 的表面连续性，二维 inpaint 只负责最后的无信息区域。

### 8.10 论文与公开代码差异表

| 论文描述 | 默认公开实现 |
|---|---|
| 从 44 个相机中按 UV 覆盖率选择 8–12 视角 | 固定六视角，无贪心选择 |
| 每个生成视图经过单图超分辨率 | SR 模型加载与调用被注释，只做 PIL resize |
| 未覆盖 texel 按距离倒数加权 | 实际使用 $1/d^2$ |
| 顶点颜色传播完成 UV inpainting | 顶点传播后又执行 OpenCV Navier–Stokes inpaint |
| Shape decoder 支持完整/多分辨率查询 | 默认标准路径是完整 $385^3$ 稠密查询 |
| 冻结 SD2.1 reference branch | checkpoint 有双 UNet，但无训练代码验证冻结过程 |
| 输入前去背景 | Gradio/API 会做；README 最短示例对 RGB 的判断不可达 |

此外，Turbo pipeline 会计算 position attention mask 和 voxel indices，但 `set_turbo(True)` 只修改外层 pipeline，`UNet2p5DConditionModel.is_turbo` 仍固定为 `False`，构造 transformer blocks 时也继承该值。因此在固定 commit 中，这些 position mask **确定没有进入 multi-view attention**；这不是需要运行时猜测的版本差异，而是静态可确认的失效路径。

### 8.11 计算与部署成本

| 阶段 | 主要成本 | 公开信息 |
|---|---|---|
| Shape DiT | 50 次、长度 4442、width 1024、48 blocks | Shape 本体约 1.113B |
| Shape SDF decode | 5706 万个 3D queries | 分 8000 点查询，通常是显存/耗时瓶颈 |
| Delight | 50 步 diffusion | 总参数约 1.29B |
| Turbo Paint | 10 次六视图 2.5D UNet | checkpoint 含 generation + reference UNet，约 1.861B |
| UV baking | 六张 $2048^2$ 投影与融合 | 依赖自定义 CUDA rasterizer |

官方文档对显存有冲突：README 写 Shape 6 GB、Shape+Texture 16 GB；`docs/source/modelzoo.md` 写 Shape+Texture 24.5 GB。应按 **6 GB shape-only，完整纹理建议至少 24.5 GB** 做保守规划。

完整 Paint 路径硬依赖 CUDA：custom rasterizer 使用 PyTorch `CUDAExtension` 并需要 `nvcc`；代码中还有硬编码 `.to("cuda")`。README 虽写支持 macOS、Windows、Linux，但 macOS 无法按默认实现完整运行纹理管线。

### 8.12 最小复现建议

1. 优先跑 shape-only，显式指定 `hunyuan3d-dit-v2-0`、fp16、50 steps、CFG 5.0；
2. 输入先用 rembg 得到正确 alpha，不能照抄示例中的不可达判断；
3. 若只验证结构，降低 `octree_resolution`；标准 384 实际需要查询 5706 万点；
4. Paint 前编译 custom rasterizer 与 differentiable renderer；
5. 对 texture pipeline 显式写出 Turbo 子目录，避免默认值随仓库更新漂移；
6. 不建议直接 `snapshot_download` 整个 Shape 子目录：当前目录包含多个重复 checkpoint，约 22.95 GiB，而默认所需 fp16 权重约 4.93 GB；
7. 记录 Git commit 与 Hugging Face revision。当前 HF `main` 没有绑定代码 commit，未来可能产生组件不兼容。

---

## 9. 局限与个人评价

### 9.1 方法局限

1. **单图不可见区域仍是生成而非重建**：背面几何和纹理没有观测约束，条件指标无法证明 3D 真值正确。
2. **SDF + Marching Cubes 偏向水密三角网格**：薄片、开放曲面、透明壳体和 artist-friendly topology 仍不自然。
3. **Paint 不能修复 Shape 的拓扑错误**：第一阶段缺腿、粘连或穿插会直接传递到 normal/position maps。
4. **Delight 不是 PBR 分解**：最终 2.0 仍主要输出 RGB texture，不能在新光照下正确控制 metallic/roughness；这正是 2.1 要补的缺口。
5. **离散视角与 UV 烘焙仍有工程脆弱性**：凹面、自遮挡、缝隙、背面和 UV seams 依赖视角覆盖与后处理。
6. **输出 mesh 不等于生产拓扑**：Marching Cubes 的密集三角面需要 remesh/retopology，不能直接视为游戏或动画资产。

### 9.2 实验与可复现性局限

- 没有真正支撑“Scaling”标题的数据/参数/算力曲线；
- 没有正式消融；
- 匿名闭源基线无法核验版本和公平性；
- 用户研究缺少原始计数与统计检验；
- 没有系统失败案例；
- 只开源推理和权重，论文最关键的训练数据与训练策略不可复现；
- 论文与代码在视角数、SR 和 inpainting 上不完全一致。

### 9.3 我的判断

Hunyuan3D 2.0 的历史价值不在于某一项指标绝对领先，而在于它把腾讯混元 3D 的技术栈稳定成了四层接口：

1. **可生成的 3D latent representation**；
2. **image-conditioned 3D Flow/DiT**；
3. **geometry-conditioned multi-view texture diffusion**；
4. **UV baking 与生产后处理**。

后续 2.1 将第 3 层从 RGB texture 推向 production-ready PBR，2.5 则用 LATTICE 重构第 1、2 层的几何 scaling。换句话说，2.0 是整个 2.x 系列真正的“操作系统版本”，后续版本更多是在替换其核心模块，而不是重新发明端到端接口。

---

## 10. 与后续系列的关系

| 版本 | 几何主干 | 纹理/材质 | 最值得关注的变化 |
|---|---|---|---|
| 1.0 | 多视图扩散 + SVRM triplane/SDF | Baking | 从单图构造多视图再重建 |
| **2.0** | ShapeVAE latent set + Flow-Matching DiT | RGB multi-view diffusion + UV baking | 建立 Shape→Texture 两阶段基础架构 |
| 2.1 | 更大 Shape 模型 | production-ready PBR material | 从 RGB 外观进入可重光照材质 |
| 2.5 | LATTICE，最大 10B | PBR multi-view architecture | 显式追求几何 scaling、锐利细节与高保真资产 |

这里尤其要注意：2.5 报告所说的 LATTICE 与目前 Obsidian 中独立的 [[LATTICE - 大规模高保真三维生成表征]] 有直接方法血缘，但 2.5 是完整 Hunyuan3D 系统报告，不能被 LATTICE 单篇笔记替代；它还需要独立讨论 10B shape foundation model、PBR texture 与端到端评测。

---

## 11. 复习提纲

### 11.1 三句话记忆

1. ShapeVAE 用“均匀点 + 边缘重要性点”分别 FPS，最长压成 3072 个 $64$ 维 latent tokens。
2. Hunyuan3D-DiT 用 16 个 double-stream + 32 个 single-stream Flow-Matching Transformer，从 DINO 图像条件直接生成这些 3D tokens。
3. Paint 用 Delight、normal/position、reference attention 和 multi-view attention 生成六/多视图纹理，再用 $\cos^4$ 权重烘焙到 $2048^2$ UV。

### 11.2 最容易混淆的点

- 2.0 的“two-stage”是 **Shape→Texture**，不是 1.0 的 **Multi-view Diffusion→Sparse-view Reconstruction**。
- Shape latent 没有固定 3D 位置，所以 DiT 不加 shape positional embedding；这后来正被 LATTICE 批评为不利于 scaling。
- 论文 Paint 是 8–12 个贪心视角，默认代码是固定 6 视角。
- `octree_resolution=384` 不代表默认走 octree，标准 decoder 实际查询完整 $385^3$ 网格。
- README 的 Paint 最短调用默认载入 Turbo；外层传 30 steps，真实循环只有 10 次。
- 官方“开源”主要指推理代码和权重，不等于训练可复现。

---

## 12. 图表索引与材料

| 图 | 内容 | 原始位置 |
|---|---|---|
| 图 1 | Hunyuan3D 2.0 Shape→Texture 总架构 | Figure 2 / PDF p.4 |
| 图 2 | ShapeVAE 重要性采样与编解码 | Figure 3 / PDF p.5 |
| 图 3 | 16+32 层 Hunyuan3D-DiT | Figure 4 / PDF p.6 |
| 图 4 | Hunyuan3D-Paint 双流与多任务注意力 | Figure 5 / PDF p.7 |
| 图 5 | 几何生成定性对比 | Figure 7 / PDF p.12 |
| 图 6 | 纹理生成定性对比 | Figure 8 / PDF p.14 |

- [论文 PDF](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/sources/paper.pdf)
- [论文抽取文本](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/sources/paper-text.md)
- [完整来源、代码与配置记录](assets/Hunyuan3D%202.0%20-%20大规模几何与纹理扩散/sources.md)
