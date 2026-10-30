---
title: Hunyuan3D 2.1：生产级 PBR 三维资产生成精读
tags: [论文精读, 3D生成, PBR, Image-to-3D, Flow-Matching]
paper: https://arxiv.org/abs/2506.15442
code: 官方公开推理、Shape DiT 微调与 Paint 训练主体；缺 ShapeVAE 从头训练、正式数据，Paint 默认训练配置不可原样运行
code_url: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
code_commit: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
rating: 4/5
created: 2026-10-30
updated: 2026-10-30
---

# Hunyuan3D 2.1：生产级 PBR 三维资产生成精读

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material* |
| 核心升级 | 3.3B Shape + 双分支 PBR Paint，显式输出 albedo、metallic、roughness |
| 数据 | Shape 100K+；PBR 70K+ 精选资产 |
| 代码 | [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)，commit `82920d6` |
| 开源边界 | 推理基本完整；Shape 只开放 DiT 微调，Paint 训练主体存在但默认 YAML target 错误；正式数据未公开 |
| 离线材料 | [论文](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/sources/paper.pdf) · [来源](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/sources.md) |
| 评级 | **★★★★☆**：PBR 方向关键，但“production-ready”证据和训练复现完整度不够 |

> [!summary] 一句话结论
> Hunyuan3D 2.1 沿用 [[Hunyuan3D 2.0 - 大规模几何与纹理扩散]] 的 Shape→Texture 接口，把几何扩展为 21 层、末 6 层 MoE 的 3.3B Flow-Matching DiT，并把 RGB Paint 升级为 albedo/MR 双分支 PBR diffusion：Spatial-Aligned Multi-Attention 对齐材质通道，3D-Aware RoPE 约束跨视图对应，跨光照一致性监督减少阴影污染；但代码与论文在 3072/4096 tokens、DINO Giant/Large 等处明显不一致。

---

## 1. 2.1 相比 2.0 到底升级了什么

### 1.1 几何侧：更大模型，而非更换表示

2.1 仍使用 VecSet ShapeVAE、SDF decoder、Marching Cubes 和 Flow Matching。真正变化是模型规模和训练数据：Shape 数据超过 100K，DiT 为 21 层，发布模型约 3.3B，并在末 6 层加入 8 experts、top-2 的 MoE FFN。

论文仍称最大 latent 长度 3072、图像编码器为 DINOv2-Giant；公开权重却使用固定 4096×64 latent 与 DINOv2-Large。故 2.1 的论文模型和发布模型不能视作逐项完全一致。

### 1.2 材质侧：从“看起来有纹理”到“可重光照”

2.0 主要生成 RGB texture，它把参考图中的反射、阴影与固有颜色混在一起。2.1 明确采用 Disney Principled BRDF 语义，生成：

- **Albedo**：不含光照的基色 RGB；
- **Metallic**：金属度；
- **Roughness**：粗糙度。

Normal 与 canonical coordinate map 仍是几何条件，不是生成材质。公开代码也没有 normal texture、height、opacity、emission、clearcoat 或 subsurface 独立输出，因此“production-ready PBR”在本论文中实际等于 **Base Color + Metallic + Roughness**。

---

## 2. 数据处理：2.1 披露最详细的部分

![](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/figures/01-data-pipeline.png)

*Algorithm 1 / PDF p.4。论文集中给出了 watertight、SDF/surface sampling 与相机采样，但伪代码中的变量名存在自相矛盾。*

### 2.1 Shape 数据

数据超过 100K，来源包括 ShapeNet、ModelNet40、Thingi10K、Objaverse 与自建数据。mesh 先中心化并缩放到 unit cube，再用 generalized winding number 判断内外：

$$\operatorname{SDF}(q)=\operatorname{distance\_to\_mesh}(q,V,F)\cdot\operatorname{sign}(\omega(q))$$

以 $\omega(q)>0.5$ 判内部，并在 SDF 零等值面 Marching Cubes 得到 watertight mesh。

SDF queries 同时包含近表面点与 $[-1,1]^3$ 中均匀点；图中分别标注 $249,856$，但伪代码把 `P_surface/P_near/P_uniform` 混写，无法确认这两个数字是各自数量还是命名错误。

Surface point set 约 50% uniform、50% high-curvature/sharp sampling；图中 `N=124,928`，最自然的解释是两组各 124,928，但论文没有明确写合计关系。

### 2.2 条件图渲染

每个对象渲染 150 个相机，球面位置用 Hammersley sequence；FoV 服从 $U(10^\circ,70^\circ)$，相机半径在 $[1.51,9.94]$ 内随 FoV 调整，以保持前景 framing。

### 2.3 PBR 数据

70K+ 人工标注、严格筛选的 Objaverse/Objaverse-XL 资产。每个对象采用 4 个 elevation（$-20^\circ,0^\circ,20^\circ$ 与一个随机角），每个 elevation 24 个 azimuth，即约 96 views/object。每个 view 生成 albedo、metallic、roughness 及 HDR/point-light render，分辨率 $512^2$。

光照采样为 point light 0.3、HDR 0.7；reference elevation 在 $[-30^\circ,70^\circ]$。这些成对照明样本是后续 illumination-invariant training 的基础。

---

## 3. ShapeVAE 与 Hunyuan3D-DiT

![](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/figures/02-shape-pipeline.png)

*Figure 2 / PDF p.5。DINO 图像条件驱动 Flow-Matching DiT 生成 latent set，VAE decoder 查询 SDF 并提取网格。*

### 3.1 ShapeVAE

均匀点 $P_u$ 与重要性点 $P_i$ 分别 FPS 得到 $Q_u,Q_i$，拼接后经 Fourier encoding、cross-attention 与 self-attention 得到隐表示：

$$H_s\in\mathbb R^{(M'+N')\times d}$$

VAE 预测：

$$\mathbb E(Z_s),\operatorname{Var}(Z_s)\in\mathbb R^{(M'+N')\times d_0}$$

Decoder 把 latent 投影回宽度 $d$，经 16 层 self-attention，以规则 3D grid queries 读取隐式场并输出 SDF。损失为：

$$\mathcal L_r=\mathbb E_x\left[\operatorname{MSE}(D_s(x\mid Z_s),\operatorname{SDF}(x))\right]+\gamma\mathcal L_{\mathrm{KL}}$$

论文没有披露 $\gamma$、ShapeVAE optimizer、学习率、batch、训练步数和 SDF 截断。

### 3.2 21 层 DiT 与 MoE

![](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/figures/03-dit-block.png)

*Figure 3 / PDF p.5。每层包含 self-attention、图像 cross-attention 与 MoE/FFN；U 形 latent skip 通过通道拼接进入后半段。*

发布配置的具体形状：

| 项目 | 数值 |
|---|---:|
| latent | $4096\times64$ |
| hidden | 2048 |
| depth | 21 |
| heads | 16 |
| head dim | 128 |
| MoE layers | 最后 6 层 |
| experts / top-k | 8 / 2 |
| image condition | DINOv2-Large，$1370\times1024$ |

前 10 层输出被保存，后半段逆序拼接形成 U 型 skip。Flow Matching 仍使用：

$$x_t=(1-t)x_0+tx_1,\qquad u_t=x_1-x_0$$

$$\mathcal L=\mathbb E\left[\|u_\theta(x_t,c,t)-u_t\|_2^2\right]$$

公开标准推理为 50-step Euler、CFG 5.0。

---

## 4. PBR Paint：双材质流与三类约束

![](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/figures/04-pbr-paint.png)

*Figure 4 / PDF p.6。Albedo 与 MR 两条生成流并行；normal/position 提供几何坐标，reference/multi-view/material attention 共同约束材质。*

### 4.1 双分支表示

模型不是直接生成五六种独立通道，而是生成两个 RGB diffusion streams：

1. `albedo`：RGB；
2. `mr`：RGB 容器，最终 **R=metallic、G=roughness、B 丢弃**。

六视角默认 latent 为：

$$[1,2,6,4,64,64]$$

其中 2 是 albedo/MR；normal 与 position 各自复制到两种材质流，与噪声沿通道拼成：

$$[1,2,6,12,64,64]\rightarrow[12,12,64,64]$$

### 4.2 Spatial-Aligned Multi-Attention

Albedo 与 MR 不应各自随意生成：例如金属区域必须与基色纹理空间对齐。2.1 先计算 Albedo branch 的 reference-attention 输出，再直接传播给 MR branch，让后者学习参考图、albedo 与 metallic/roughness 的联合对应。

论文未给显式公式或单独 alignment loss，因此这是结构性耦合，而非额外像素监督。

### 4.3 3D-Aware RoPE

普通 multi-view attention 只知道不同视图中有哪些 tokens，不知道它们是否对应同一 3D 点。2.1 从 position maps 构造多分辨率 voxel indices，在 $64,32,16,8$ 四级 latent grid 中分别产生跨视图 3D 坐标索引，使 attention 更容易把相同表面位置关联起来，减少 seam 与 ghosting。

论文称其为 RoPE，但没有公开标准旋转频率/角度公式，只描述 3D coordinate encoding 对 hidden states 的注入；精确机制必须以代码为准。

### 4.4 跨光照一致性

同一物体的两张不同光照参考图应预测相同 intrinsic materials。训练损失在公开实现中为：

$$\mathcal L=0.85(\mathcal L_{\mathrm{albedo}}+\mathcal L_{\mathrm{MR}})+0.15\mathcal L_{\mathrm{consistency}}$$

这是一种近似 intrinsic-material disentanglement，但论文没有完整 intrinsic decomposition 方程，也没有显式估计光源或 shading。

### 4.5 推理与烘焙

公开默认 Paint：6 views、$512^2$、UniPC 15 steps、CFG 3.0。VAE 输出六张 albedo 与六张 MR，回投到 $4096^2$ UV atlas，再导出 $2048^2$ 贴图。GLB 按规范把 roughness 写入 G、metallic 写入 B。

---

## 5. 实验

### 5.1 Shape 指标

| 模型 | ULIP-T ↑ | ULIP-I ↑ | Uni3D-T ↑ | Uni3D-I ↑ |
|---|---:|---:|---:|---:|
| Michelangelo | 0.0752 | 0.1152 | 0.2133 | 0.2611 |
| Craftsman 1.5 | 0.0745 | 0.1296 | 0.2375 | 0.2987 |
| TripoSG | 0.0767 | 0.1225 | 0.2506 | 0.3129 |
| Step1X-3D | 0.0735 | 0.1183 | 0.2554 | 0.3195 |
| Trellis | 0.0769 | 0.1267 | 0.2496 | 0.3116 |
| Direct3D-S2 | 0.0706 | 0.1134 | 0.2346 | 0.2930 |
| **Hunyuan3D-DiT** | **0.0774** | **0.1395** | **0.2556** | **0.3213** |

每个 mesh 采样 8192 surface points。Uni3D-T 相比 Step1X-3D 只高 0.0002，论文没有误差条，不能把微小差异解读为稳定显著优势。

![](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/figures/05-shape-results.png)

*Figure 5 / PDF p.8。2.1 在尖锐结构与局部组件数量上更完整，但仍缺随机失败样例。*

### 5.2 PBR Paint 指标

| 方法 | CLIP-FID ↓ | CMMD ↓ | CLIP-I ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| SyncMVD-IPA | 28.39 | 2.397 | 0.8823 | 0.1423 |
| TexGen | 28.24 | 2.448 | 0.8818 | 0.1331 |
| Hunyuan3D 2.0 | 26.44 | 2.318 | 0.8893 | 0.1261 |
| **Hunyuan3D-Paint** | **24.78** | **2.191** | **0.9207** | **0.1211** |

相对 2.0，CLIP-FID 改善 6.28%、CMMD 5.48%、CLIP-I 3.53%、LPIPS 3.97%。

![](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/figures/06-pbr-results.png)

*Figure 6 / PDF p.9。PBR 材质能在几何相同的情况下提供更可信的金属/粗糙表现，但论文没有对各 PBR 通道单独评估物理准确性。*

### 5.3 实验证据的缺口

论文没有任何正式消融，也没有用户研究。以下关键设计均无独立量化：MoE、importance sampling、4096/3072 tokens、Spatial-Aligned Attention、3D-Aware RoPE、illumination consistency、Albedo/MR 双分支。因此指标只证明整套系统强，不能证明各模块的因果贡献。

---

## 6. 代码级实现：按 Shape 与 PBR Paint 的真实数据流拆解

下面默认 batch size $B=1$。Shape 以独立 demo 的发布配置为准；Paint 以六视图、$512^2$ 的独立 demo/worker 为主，同时指出 Gradio 的八视图、$768^2$ 差异。每一步同时回答五件事：入口、输入 shape、内部 reshape、输出 shape，以及为什么这么做。

### 6.1 开源覆盖

| 模块 | 覆盖情况 |
|---|---|
| Shape 推理 | 完整 |
| Shape DiT 微调/过拟合 | 较完整 |
| ShapeVAE 从头训练 | 缺失 |
| Paint 推理 | 完整但有实现缺陷 |
| Paint UNet 训练主体 | 已公开 |
| Paint 默认训练命令 | YAML target 指向不存在模块；即使修正 target，构造函数位置参数数量仍不匹配，不能原样运行 |
| 正式 100K+/70K+ 数据 | 未公开 |

许可证不是标准 Apache/MIT；存在地域、100 万 MAU 和模型用途限制，商用必须单独审阅。

### 6.2 Shape 全链路张量

| 阶段 | 输入 | 输出 |
|---|---|---|
| 预处理 | RGBA/RGB | image $[1,3,512,512]$，mask $[1,1,512,512]$ |
| DINOv2-Large | $[1,3,518,518]$ | $[1,1370,1024]$ |
| 初始噪声 | Gaussian | $[1,4096,64]$ |
| DiT 投影 | latent | $[2,4096,2048]$（CFG） |
| 加 time token | shape sequence | $[2,4097,2048]$ |
| 21 层 DiT | image cross-attention | velocity $[2,4096,64]$ |
| Euler×50 | CFG 5.0 | latent $[1,4096,64]$ |
| VAE decoder | $4096\times64$ | $4096\times1024$ |
| SDF query | $385^3$ 点 | grid $[1,385,385,385]$ |
| MC | zero level set | vertices/faces |

默认参数 `octree_resolution=384` 仍走完整 $385^3=57,066,625$ 查询，不是稀疏 octree。

### 6.3 Paint 全链路张量

| 阶段 | 形状 |
|---|---|
| 参考图 VAE | $[1,1,4,64,64]$ |
| DINOv2-Giant | $[1,1370,1536]\rightarrow[1,5480,1024]$ |
| noise | $[1,2,6,4,64,64]$ |
| normal/position latent | 各 $[1,2,6,4,64,64]$ |
| UNet 输入 | $[1,2,6,12,64,64]\rightarrow[12,12,64,64]$ |
| CFG 三分支 | $[36,12,64,64]$ |
| UniPC | 15 steps，CFG 3.0 |
| VAE decode | 12 张 $512^2$，前 6 albedo、后 6 MR |
| UV bake | 内部 $4096^2$，导出 $2048^2$ |

### 6.4 Shape 执行链：图像如何变成 $4096\times64$ latent

#### 6.4.1 图像预处理与 DINOv2-Large

`ImageProcessorV2` 根据 alpha bbox 裁剪前景、居中到物体占画布约 85%、白底合成、resize 为 $512^2$，输出：

$$I\in\mathbb R^{1\times3\times512\times512},\quad M\in\mathbb R^{1\times1\times512\times512}$$

像素值映射到 $[-1,1]$。普通 RGB 会被补成全不透明 alpha，所以它本身不负责抠图；mask 虽传入 conditioner，DINO encoder 并没有使用它。

发布 Shape 权重使用 DINOv2-Large，不是论文所写的 Giant。外部模型先把图像 resize/crop 为 $518^2$，patch size 为 14，因此：

$$[1,3,512,512]\rightarrow[1,3,518,518]\rightarrow[1,1+37^2,1024]=[1,1370,1024]$$

1370 是一个 CLS 加 1369 个 patch tokens，1024 是 DINOv2-Large hidden width。CFG 将条件与同 shape 的全零 tokens 拼成 $[2,1370,1024]$。保留所有 patch tokens 是为了让三维 latent 能通过 cross-attention读取局部部件，而不是只读取一个全局类别向量。

#### 6.4.2 初始噪声、投影与 time token

`prepare_latents()` 由 `vae.latent_shape` 读取发布配置，采样：

$$x_0\sim\mathcal N(0,I),\qquad x_0\in\mathbb R^{1\times4096\times64}$$

每一步 CFG 临时复制为 $[2,4096,64]$，再用线性层 $64\rightarrow2048$：

$$[2,4096,64]\rightarrow[2,4096,2048]$$

标量 timestep 经 sinusoidal embedding 与 MLP：

$$[2]\rightarrow[2,2048]\rightarrow[2,1,2048]$$

然后作为第一个 token 拼入 shape sequence：

$$[2,1,2048]\oplus[2,4096,2048]\rightarrow[2,4097,2048]$$

time token 进入全部 self-attention、cross-attention 和 MoE 层，使噪声强度不只是一次全局加法，而能在每层参与 token 交互。发布配置 `use_attention_pooling=false`，所以这里没有再叠加 DINO pooled token。

#### 6.4.3 21 层 DiT 的 U 形 skip

每层顺序是：可选 skip 融合 → self-attention → image cross-attention → FFN/MoE。cross-attention 的 Query 长度为 4097，Key/Value 长度为 1370：

$$Q\in\mathbb R^{2\times16\times4097\times128},\quad K,V\in\mathbb R^{2\times16\times1370\times128}$$

hidden 为 2048、16 heads，所以 head dim 为 128。U 形 skip 的实际顺序为：

- 第 0–9 层保存 10 个 $[2,4097,2048]$ hidden states；
- 第 10 层作为中间层；
- 第 11–20 层逆序弹出 skip，与当前 hidden 在 channel 维拼成 $[2,4097,4096]$，再线性压回 2048。

这里的 U 形不是图像 UNet 的空间下采样/上采样，而是 Transformer depth 上的长跳连。它让后半段直接取回浅层条件对齐特征，避免 21 层深网中细节被过度改写。

#### 6.4.4 最后 6 层 MoE

第 15–20 层把普通 FFN 换成 8 个 routed experts、top-2，并额外保留一个始终执行的 shared expert。每个 $2048$ 维 token 经 gate 得到 8 路概率，选择两个 expert 做加权和；默认 `norm_topk_prob=false`，top-2 权重不会重新归一化。MoE 处理全部 4097 tokens，包括 time token。

这么做是把新增参数主要放到条件计算的 FFN 中：每个 token 只激活两个 routed experts，理论计算量不会按 8 倍增长；shared expert 则提供所有 token 都能访问的通用变换。公开训练代码还有系数 0.01 的 routing auxiliary loss，用于避免 experts 严重负载失衡。

#### 6.4.5 速度预测、CFG 与 Euler

21 层结束后删除第一个 time token，并执行 $2048\rightarrow64$：

$$[2,4097,2048]\rightarrow[2,4096,2048]\rightarrow\hat u_\theta\in\mathbb R^{2\times4096\times64}$$

CFG 与 2.0 相同：

$$\hat u=\hat u_{\mathrm{uncond}}+5.0(\hat u_{\mathrm{cond}}-\hat u_{\mathrm{uncond}})$$

Flow Matching 目标为 $x_t=(1-t)x_0+tx_1$、$u_t=x_1-x_0$；Euler 以 $x_{next}=x+(\sigma_{next}-\sigma)\hat u$ 更新。标准配置调用模型 50 次，最后一次步长为零，因此是 49 次非零状态更新。

### 6.5 ShapeVAE 解码：latent 如何变成 mesh

DiT 输出先除以 `scale_factor=1.0039506158752403`，再由 $64\rightarrow1024$ 的 `post_kl` 与 16 层 latent self-attention得到：

$$[1,4096,64]\rightarrow[1,4096,1024]$$

`VanillaVolumeDecoder` 建立完整 $385^3=57,066,625$ 个 xyz queries，按最多 8000 点分块；每个 query 先做 Fourier embedding 和 $\rightarrow1024$ 投影，再以 cross-attention 从 4096 个 latent tokens 读取局部场值：

$$[1,P,3]\rightarrow[1,P,1024]\xrightarrow{\mathrm{cross\ attn}\ Z}[1,P,1],\quad P\le8000$$

拼接为 $[1,385,385,385]$ 后对零等值面执行 Marching Cubes，输出 $V\in\mathbb R^{N_v\times3}$ 与 $F\in\mathbb N^{N_f\times3}$。论文称其为 SDF，代码默认 `label_type="binary"`、变量名为 `occ`，且没有距离标定；因此代码级准确叫法应是“隐式 occupancy/logit 标量场”，不能把 metric SDF 当成已由推理代码证明的事实。`octree_resolution=384` 仍是稠密网格，不是 sparse octree。

### 6.6 PBR Paint 第一步：视图、参考图与几何条件

#### 6.6.1 不同入口的视图数不同

独立 demo/worker 和训练 YAML 使用六视图、$512^2$；Gradio 默认八视图、$768^2$。前六个固定为 front/right/back/left/top/bottom；超过六个时，代码按“新增加的可见三角形面积”从候选相机中贪心补视图。因此后文以 $N=6$ 写 shape，但不能把“六视图”当成所有入口的唯一默认值。

#### 6.6.2 normal、position 与参考图

mesh 经 UV 展开和坐标归一化后，从每个相机渲染：

$$N,P\in\mathbb R^{1\times6\times3\times512\times512}$$

normal 使用 object-space 朝向，position 使用归一化 object coordinates，背景为 1。参考图 resize 为 $[1,1,3,512,512]$。normal 告诉网络表面朝向，position 提供跨视图三维位置对应；两者不是最终输出的 PBR maps。

### 6.7 PBR Paint 第二步：外部 VAE/DINO 与双材质 latent

#### 6.7.1 SD VAE

参考图、normal、position 都通过 `AutoencoderKL.encode()` 做 8 倍空间下采样：

$$I_{ref}:[1,1,3,512,512]\rightarrow Z_{ref}:[1,1,4,64,64]$$

$$N,P:[1,6,3,512,512]\rightarrow Z_N,Z_P:[1,6,4,64,64]$$

初始材质噪声在 pipeline 中先展平成 $[12,4,64,64]$，其中 $12=2$ 个材质流 $\times6$ 个视图；进入 UNet 前恢复为：

$$Z_t\in\mathbb R^{1\times2\times6\times4\times64\times64}$$

材质轴 0 是 albedo，轴 1 是 MR。normal/position 复制到两个材质流，沿 channel 拼接：

$$[1,2,6,4,64,64]^{\oplus3}\rightarrow[1,2,6,12,64,64]$$

三分支 CFG 后变为 $[3,2,6,12,64,64]$，再展平成 generation UNet 输入：

$$[36,12,64,64]$$

#### 6.7.2 DINOv2-Giant 不是 Shape 侧的 DINOv2-Large

Paint 参考图同时进入 DINOv2-Giant。其输出是：

$$[1,3,512,512]\rightarrow[1,1370,1536]$$

每个 1536 维 token 再由线性层展开为 $4\times1024$，并把该维拆到 token 轴：

$$[1,1370,1536]\rightarrow[1,1370,4096]\rightarrow[1,5480,1024]$$

DINO tokens 提供语义与局部外观条件；SD VAE reference latent 则进入逐层 reference attention，提供与 UNet 空间尺度更接近的图像特征。两条外部编码路径并不重复。

#### 6.7.3 Reference UNet

`unet_dual` 是 generation UNet 的副本，不是 MR 专属网络。reference latent 的 CFG batch 为 $[3,1,4,64,64]\rightarrow[3,4,64,64]$，timestep 固定为 0；它在每个 transformer block 保存归一化 hidden states $[3,L,C]$，generation UNet 随后读取。论文训练时冻结 reference branch，推理代码能确认双 UNet 与特征缓存，但不能单独证明训练冻结过程。

### 6.8 PBR Paint 第三步：UNet 内部 attention 与张量变换

#### 6.8.1 多尺度主干

UNet 接收 $[36,12,64,64]$，`conv_in` 后为 $[36,320,64,64]$。发布配置的尺度为：

| 阶段 | hidden shape | 增强 attention |
|---|---|---|
| down-0 | $[36,320,64,64]$ | material/reference/multi-view/text/DINO |
| down-1 | $[36,640,32,32]$ | 同上 |
| down-2 | $[36,1280,16,16]$ | 同上 |
| down-3 | $[36,1280,8,8]$ | 无 cross-attention |
| mid | $[36,1280,8,8]$ | 增强 attention |
| up-0 | $[36,1280,8,8]$ | 无 cross-attention |
| up-1 | $[36,1280,16,16]$ | 增强 attention |
| up-2 | $[36,640,32,32]$ | 增强 attention |
| up-3 | $[36,320,64,64]$ | 增强 attention |
| output | $[36,4,64,64]$ | — |

这不是两套完整 diffusion UNet，而是一套共享卷积/ResNet 主干中的显式 $[B,2,N,L,C]$ 材质轴，加上部分材质专用 attention 参数。

#### 6.8.2 Spatial-Aligned Multi-Attention 的真实机制

代码先恢复：

$$H\in\mathbb R^{B\times2\times N\times L\times C}$$

reference attention 的 Query **只取 albedo hidden**，Key 来自 reference UNet；由两者得到的 attention 权重同时给 albedo 与 MR 使用：

$$M=\operatorname{Softmax}(Q_AK_{ref}^{T}/\sqrt d)$$

但两类材质拥有独立的 Value 与输出投影：

$$H_A'=H_A+W_A^O(MV_A),\qquad H_{MR}'=H_{MR}+W_{MR}^O(MV_{MR})$$

因此不是“先计算 albedo attention 输出，再把输出复制给 MR”，而是**共享空间对应关系，分离要写入的材质值**。这正符合物理语义：同一表面位置应同时对应 base color 与 metallic/roughness，但两者的属性值不同。

#### 6.8.3 3D-Aware RoPE 的公式与 shape

公开代码已经给出标准 RoPE，而不是“公式未公开”：频率为 $1/10000^{2i/d}$，x/y/z 分配 head dim 的 $3/8,3/8,2/8$，并对 Q/K 同时执行复数对旋转。position map 在四个 latent 尺度平均池化并量化：

| 空间尺度 | voxel resolution | 六视图 token 数 |
|---:|---:|---:|
| $64\times64$ | 512 | $6\times4096=24576$ |
| $32\times32$ | 256 | $6\times1024=6144$ |
| $16\times16$ | 128 | $6\times256=1536$ |
| $8\times8$ | 64 | $6\times64=384$ |

在 multi-view attention 中：

$$[3,2,6,L,C]\rightarrow[6,6L,C]$$

首维 $6=3$ 个 CFG branches $\times2$ 个材质流；每组把六个视图拼起来后，Q/K 根据同一表面的三维坐标做 RoPE，再执行跨视图 attention。这样模型不只知道“其他视图有哪些 token”，还知道 token 在 object coordinates 中的相对位置。

#### 6.8.4 三分支 CFG 的真实语义与代码退化

三支分别是：

| 分支 | Reference attention | DINO condition |
|---|---|---|
| uncond | 关闭，`ref_scale=0` | 全零 |
| ref | 开启 | 全零 |
| full | 开启 | 真实 DINO |

UNet 输出从 $[36,4,64,64]$ 切成三份，每份 $[12,4,64,64]$。但当前代码使用同一系数 $s\alpha$ 组合：

$$u+s\alpha(r-u)+s\alpha(f-r)=u+s\alpha(f-u)$$

所以中间 `ref` 分支在代数上完全抵消，只可能留下浮点误差。也就是说，三分支结构表达了“reference guidance 与 DINO guidance 分开控制”的意图，但固定 commit 的相同系数组合实际退化为普通的 uncond→full guidance。

### 6.9 PBR Paint 第四步：UniPC、解码与 UV 材质

独立 demo 使用 UniPC 15 steps、基础 CFG 3.0；代码还根据 azimuth 乘 view scale，因此不同视图的有效 guidance 并不完全相同。采样完成后：

$$[12,4,64,64]\xrightarrow{\mathrm{VAE\ decode}}[12,3,512,512]$$

前六张是 albedo，后六张是 MR；MR 图的 R 通道取 metallic，G 通道取 roughness，B 丢弃。RealESRGAN 对全部视图执行 $4\times$ 超分，默认 $512\rightarrow2048$。注意代码 bug 发生在**后续 resize 循环**，它因 `len(dict)==2` 只 resize 前两张，而不是“超分只处理两张”；在 $512$ 模式输出本来就是 $2048$，通常不暴露，在 $768\rightarrow3072$ 的 Gradio 路径会造成分辨率不一致。

每个视图反投影到内部 $4096^2$ UV atlas，过滤深度不一致和夹角大于 $75^\circ$ 的区域，以 preset view weight $\times\cos^4\theta$ 融合，mesh vertex propagation 与 OpenCV Navier–Stokes 补洞，最终 OBJ 独立输出 $2048^2$ albedo、metallic、roughness 贴图。

OBJ 路径能明确确认独立 metallic/roughness；GLB 是 Blender 导入 OBJ/MTL 后再导出，虽然 glTF 规范通常要求 roughness=G、metallic=B，但固定 commit 的实际 Paint 调用链没有显式调用 `combine_metallic_roughness()`，因此不能把最终 GLB 通道打包写成源码已严格保证。

### 6.10 论文与代码差异

| 论文 | 发布代码 |
|---|---|
| 最大 3072 tokens | 固定 4096 tokens |
| Shape 使用 DINOv2-Giant | Shape 使用 DINOv2-Large；Paint 才用 Giant |
| 约 50/50 uniform/sharp points | 公开 fine-tune 配置为 81,920 uniform、0 sharp |
| Paint warm-up 2000 | 配置为 1000 |
| 动态 token length | 发布配置固定 4096 |
| 完整训练叙述 | ShapeVAE 训练缺失，Paint 默认 target 错误 |

### 6.11 工程缺陷

- Root README 使用未定义 `mesh_path`；
- Shape minimal demo 引用不存在输入并覆盖预处理结果；
- API schema 的 seed/steps/resolution 等参数没有传给 worker pipeline；
- Paint 的 RealESRGAN 超分会处理全部视图，但后续 resize 循环误用 `len(dict)==2`，只 resize 前两个视图；$512\rightarrow2048$ 默认路径通常不暴露，$768\rightarrow3072$ 路径会造成视图分辨率不一致；
- `low_vram_mode` 没有真正模型 offload；
- Paint 返回 OBJ 路径，即使同时创建 GLB；
- Paint 数据 augmentation 对 albedo/MR/normal/position 独立随机，可能破坏空间对齐；
- geometry dropout 检查错误 key，分支不会生效。

这些问题不否定模型权重质量，但说明“fully open-source production-ready training framework”强于该 commit 的实际可执行程度。

### 6.12 计算成本

| 模块 | 官方参数量/显存 |
|---|---|
| Shape | 3.3B，约 10 GB 推理 |
| Paint | 2B，6 views/$512^2$ 约 21 GB |
| Shape+Paint 同载 | 约 29 GB |
| Paint 训练 | 论文称约 180 GPU-days，未说明 GPU 型号 |

完整 Paint 强依赖 CUDA 12.4、NVCC custom rasterizer、C++ mesh inpaint 与 RealESRGAN；README 的 macOS 支持不能理解为默认 PBR 全链路原生可用。

---

## 7. 局限与评价

1. **PBR 通道仍不完整**：只有 albedo/metallic/roughness，缺 normal、height、opacity、emission 等生产常见材质。
2. **物理正确性未验证**：指标仍主要基于 CLIP/FID/LPIPS，没有在受控 relighting 下评估 BRDF 参数误差。
3. **“Production-ready”证据不足**：没有 DCC/game-engine 导入测试、UV/seam 指标、材质编辑实验或专业美术师用户研究。
4. **训练难复现**：ShapeVAE 缺失，Paint 默认训练入口失效，正式数据未开源。
5. **没有消融与失败案例**：无法判断 3D RoPE、通道对齐和照明一致性各自贡献。
6. **单图歧义仍在**：背面几何与材质主要依赖生成先验。

我的判断是：2.1 的真正价值是把混元 3D 从“生成一张贴图”推进到“生成可被渲染器解释的材质参数”，它是明确的资产化进步；但离“工业生产完成态”仍差 topology、UV 质量、全套 PBR 通道、可靠训练复现和 DCC 验证。

---

## 8. 与系列版本的技术位置

- 1.0：多视图扩散 + SVRM 重建；
- 2.0：VecSet Shape DiT + RGB Paint，建立 Shape→Texture 接口；
- **2.1：放大 Shape，并把 RGB Paint 升为 Albedo/MR PBR Paint**；
- 2.5：用 LATTICE 重构几何表示与 scaling，最大到 10B，并继续加强 PBR。

因此，2.1 不是简单小版本：它首次把“可重光照材质”提升为系统主目标；而 2.5 的核心变化又主要回到几何表示和 scaling。

---

## 9. 图表与材料索引

| 图 | 内容 | 来源 |
|---|---|---|
| 图 1 | 数据处理与采样 | Algorithm 1 / p.4 |
| 图 2 | Shape 总管线 | Figure 2 / p.5 |
| 图 3 | 21 层 DiT block | Figure 3 / p.5 |
| 图 4 | 双分支 PBR Paint | Figure 4 / p.6 |
| 图 5 | Shape 定性对比 | Figure 5 / p.8 |
| 图 6 | PBR texture 对比 | Figure 6 / p.9 |

- [论文 PDF](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/sources/paper.pdf)
- [来源与代码核验](assets/Hunyuan3D%202.1%20-%20生产级PBR三维资产生成/sources.md)
