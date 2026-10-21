---
type: paper-note
status: complete
domain: World-Models
paper: "HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds"
year: 2026
arxiv: "2604.14268"
doi: "10.48550/arXiv.2604.14268"
source: https://arxiv.org/abs/2604.14268
project: https://3d-models.hunyuan.tencent.com/world/
code: https://github.com/Tencent-Hunyuan/HY-World-2.0
tags:
  - World-Models
  - 3D-Generation
  - 3D-Reconstruction
  - 3DGS
  - Multi-Modal
  - WorldStereo
  - WorldMirror
created: 2026-10-16
updated: 2026-10-16
---

# HY-World 2.0：多模态三维世界生成重建与模拟

## 基本信息

- **正式标题**：*HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds*
- **作者/团队**：Team HY-World；项目负责人 Chunchao Guo、Tengfei Wang
- **发布时间**：2026-04-15（arXiv v1）
- **输入**：文本、单图、多视图图像或视频；重建分支可额外接收相机内外参和深度先验
- **输出**：3DGS 世界；中间/辅助输出包括全景图、轨迹、关键帧、点云、深度、法线、相机参数和 coarse mesh
- **范式**：离线显式 3D 世界模型，不是在线逐动作视频模型
- **本地论文**：[paper.pdf](assets/HY-World%202.0%20-%20多模态三维世界生成重建与模拟/sources/paper.pdf)

## 一句话结论

HY-World 2.0 用一个系统级闭环把“稀疏条件下需要想象未见区域的生成”和“密集观测下需要恢复几何的重建”统一到显式 3DGS 输出：文本/单图走 **HY-Pano 2.0 → WorldNav → WorldStereo 2.0 → WorldMirror 2.0 + 3DGS 优化**，多图/视频可直接走 WorldMirror 2.0；其真正创新是让关键帧视频生成、几何记忆和前馈重建互相服务，而不是单个万能网络，但全链路仍需约 **712 秒/世界（H20 系统）**、依赖多模型与逐场景优化，所谓“交互/实时”只发生在生成完成后的 WorldLens 渲染和碰撞阶段。

---

## 1. 问题演化与研究定位

### 1.1 为什么 1.0 和 1.5 都不够

团队把此前路线明确分成两类：

- **HY-World 1.0**：离线 3D-based world generation。全景代理经语义分层与深度对齐变成 mesh，资产兼容图形管线，但自由平移覆盖、几何完整度和重建通用性受限。
- **HY-World 1.5 / WorldPlay**：在线 video-based world generation。根据用户动作以 16 帧 chunk 流式生成视频，用记忆保持回访一致，并在 8×H800 系统上报告 720p/24 FPS；但主状态仍是视频/历史图像，不直接交付可编辑、可测量的 3D 世界。

2.0 不是把 1.5 的实时视频升级成更快版本，而是**回到离线显式 3D**，吸收视频模型的生成先验来补足未见区域，再用重建模型把生成关键帧固化成 3DGS。它试图同时覆盖两种输入密度：稀疏条件负责“生成”，密集观测负责“重建”。

![](assets/HY-World%202.0%20-%20多模态三维世界生成重建与模拟/figures/01-teaser.png)

> Figure 1：同一框架覆盖文本/单图生成、真实多视图重建、带几何的 3D 资产、机器人仿真、游戏和环境建图。图中“simulation”依赖输出资产与外部渲染/物理平台，不代表模型学习了通用物理动力学。来源：论文 Figure 1，PDF p.3。

### 1.2 核心贡献

1. **多模态任务路由**：文本/单图走生成链，多图/视频走重建链，最终都落到显式 3D 表示。
2. **HY-Pano 2.0**：以 MMDiT 隐式学习透视图到 ERP 的映射，不再要求 1.0 那样依赖准确内参/FoV。
3. **WorldNav**：先解析点云、mesh、语义和 NavMesh，再规划 5 类互补轨迹，按信息增益和避障补全视角。
4. **WorldStereo 2.0**：从 Video-VAE 改为 Keyframe-VAE，以 GGM + SSM++ 维持跨轨迹几何与细节一致，并蒸馏为 4-step DiT。
5. **WorldMirror 2.0**：统一 token 化图像与几何先验，一次前向预测点图、深度、法线、相机和 3DGS 属性；通过 normalized RoPE、depth-to-normal、mask head 和分布式推理解决 1.0 版重建器的分辨率与规模问题。
6. **生成式 3DGS composition**：深度对齐、非天空 densification、MaskGaussian 和几何监督把生成视图转成高保真 3DGS，并可提取 coarse mesh。

---

## 2. 任务定义、输入输出与统一边界

### 2.1 两种任务模式

#### 2.1.1 World generation

输入 $x_s$ 是文本或单张图像，输出 3DGS 世界 $\mathcal{G}$：

$$x_s\xrightarrow{\text{HY-Pano 2.0}}I^{pan}\xrightarrow{\text{WorldNav}}\{C_i\}\xrightarrow{\text{WorldStereo 2.0}}\{V_i\}\xrightarrow{\text{WorldMirror 2.0 + GS opt.}}\mathcal{G}.$$

这里“统一”是**共享组件和共同输出空间**，不是一个端到端参数统一的大模型。世界生成需要 4 个阶段和逐场景 3DGS 优化。

#### 2.1.2 World reconstruction

输入多视图图像/视频 $\{I_i\}$，以及可选 pose、intrinsics、depth priors。WorldMirror 2.0 一次前向给出：

$$\Phi(\{I_i\},\text{optional priors})\rightarrow\{P_i,D_i,N_i,C_i,G_i\},$$

分别对应稠密点图、深度、法线、相机和像素级 3D Gaussian 属性。它既可独立重建，也在生成链里负责从 WorldStereo 关键帧恢复几何。

### 2.2 最终表示

最终世界以 3DGS 为主。每个 Gaussian 包含中心 $\mu_k$、协方差 $\Sigma_k=R_kS_kS_k^TR_k^T$、opacity $\sigma_k$ 和颜色 $c_k$。论文认为生成世界的 view-dependent effect 较弱，因此使用 view-independent RGB 而非 spherical harmonics，以减少冗余和颜色伪影。另由 RGB-D 渲染融合到 TSDF，再 marching cubes 提取简化 mesh，主要服务碰撞和物理代理。

### 2.3 “生成、重建、模拟”的边界

- **生成**：补出输入未观察区域，结果不要求等同某个真实世界。
- **重建**：从密集观测恢复几何，能用 GT/基准评测。
- **模拟/交互**：WorldLens 加载生成后的 3DGS/mesh，提供角色、IBL、碰撞和渲染；论文没有训练动力学模型，也未评测因果物理预测。

---

## 3. 总体架构

![](assets/HY-World%202.0%20-%20多模态三维世界生成重建与模拟/figures/02-overview.png)

> Figure 2：四阶段不是简单串联图像模型。Stage 2 用几何/语义决定“还缺哪些观察”，Stage 3 用生成模型补出这些视图并维护记忆，Stage 4 再把视图约束回显式 3D。来源：论文 Figure 2，PDF p.5。

| 阶段 | 核心组件 | 输入 | 输出 | 解决的瓶颈 |
|---|---|---|---|---|
| 1. Panorama | HY-Pano 2.0 | 文本/单图 | 360° ERP | 稀疏输入缺全局上下文 |
| 2. Planning | WorldNav | 全景、点云、mesh、语义、NavMesh | 最多 35 条互补轨迹 | 盲目轨迹覆盖不足、会穿墙 |
| 3. Expansion | WorldStereo 2.0 | 全景、轨迹、点云、记忆 | 多轨迹一致关键帧 | 单视图未见区域和跨轨迹漂移 |
| 4. Composition | WorldMirror 2.0 + 3DGS | 关键帧、相机、全景点云 | 3DGS、点云、mesh | 生成帧不是持久几何资产 |

---

## 4. Stage I：HY-Pano 2.0

### 4.1 从显式投影转向隐式映射

1.0 先估计输入图内参/FoV，再显式投影到 ERP；元数据不准时会直接造成畸变。2.0 把输入图 latent 与 panorama noisy latent 拼成统一 token 序列，用 MMDiT self-attention 学习透视—ERP 对应关系。优点是不依赖显式标定且可以补全任意视角，代价是几何关系更依赖训练分布，缺少确定性投影约束。

左右接缝同时在两层处理：latent circular padding 强制周期边界，像素空间做线性 blending。数据来自大规模真实全景和 UE 合成环境，但论文未披露数量与许可构成。

### 4.2 全景定量

| 任务/指标 | HY-World 1.0 | HY-Pano 2.0 | 观察 |
|---|---:|---:|---|
| T2P CLIP-T↑ | 0.250 | **0.258** | 文本对齐小幅提升 |
| T2P Q-Align Qual（Perspective）↑ | 3.992 | **4.103** | 透视投影视觉质量提升 |
| T2P Q-Align Qual（ERP）↑ | **4.493** | 4.403 | 该项反而略降 |
| T2P Q-Align Aes（Perspective）↑ | **3.404** | 3.376 | 该项也未超过 1.0 |
| I2P CLIP-I↑ | 0.831 | **0.844** | 输入保真提升 |
| I2P Q-Align Qual（Perspective）↑ | 3.317 | **4.026** | 质量提升明显 |

因此“多数指标最佳”准确，“所有指标全面超越 1.0”不准确。2.0 的优势在图生全景更稳定，文生全景的 ERP quality / perspective aesthetics 仍有权衡。

---

## 5. Stage II：WorldNav

### 5.1 场景解析

系统先把 ERP 拆成 42 个透视视图，用 MoGe2 + GPU LSMR 对齐单目深度并合成全景点云 $P^{pan}$；再过滤天空和 depth discontinuity floaters。相比默认 12 views，42 views 增加覆盖。Qwen3-VL 识别地标/障碍，SAM3 给 mask 并定位到 3D。低分辨率 panoramic mesh 服务严格碰撞，Recast Navigation 生成 NavMesh；系统还做地面对齐、边界腐蚀和孤立区域桥接。

### 5.2 五类轨迹

| 轨迹 | 最大数 | 目标 |
|---|---:|---|
| Regular | 9 | 从中心向一般视角扩展，先俯仰再方位旋转 |
| Surrounding | 5 | 围绕显著对象补前后侧；72 个候选节点 + ray casting |
| Reconstruction-aware | 10 | 找高长宽比退化 mesh 面，迭代瞄准缺失区域 |
| Wandering | 3 | 走向 8 个扇区的最远可达点，覆盖走廊/街道边界 |
| Aerial | 8 | 给 surrounding/wandering 加俯视，补 BEV 盲区 |

WorldNav 是强启发式规划器，并非学习出的通用 navigation policy。它依赖初始单目几何；初始 mesh 错误会让 NavMesh、碰撞和“缺失区域”判断一起偏移。

![](assets/HY-World%202.0%20-%20多模态三维世界生成重建与模拟/figures/05-trajectory-ablation.png)

> Figure 16：只用全景视图时存在大面积空洞；regular 去掉大块伪影；surrounding/reconstruction-aware 补对象背面；wandering 改善远墙/地面；aerial 再补俯视。该消融是定性递增案例，没有跨场景几何均值。来源：论文 Figure 16，PDF p.27。

---

## 6. Stage III：WorldStereo 2.0

### 6.1 为什么是“关键帧世界扩展”而不是普通视频生成

普通 Video-VAE 做时空压缩，慢运动产生大量冗余帧，快相机又会造成 motion blur 和几何变形，恰好不利于重建。WorldStereo 2.0 对每个稀疏关键帧独立做空间压缩：

$$V_i\in\mathbb{R}^{1\times H\times W\times3}\xrightarrow{\text{Keyframe-VAE}}F_i\in\mathbb{R}^{1\times H/8\times W/8\times C}.$$

它以更大的采样间隔保持相同视角跨度，牺牲播放级连续帧，换取高频细节和并行编码。这说明 2.0 的 Stage 3 是**为 3D 重建服务的视图生成器**，不是 1.5 那种面向人机交互的视频流。

### 6.2 显式相机控制

参考图深度 $D(x)$ 被按目标相机外参和内参 warp：

$$P_i^{tar}(x)\simeq R_i^{c\rightarrow w}D(x)K_i^{-1}\hat{x}.$$

渲染后的 point-cloud views 与 Plücker rays 一起进入轻量 camera adapter。为了让 DiT 适应 keyframe latent，训练会解冻部分主干；最终设置冻结 cross-attention 和 FFN，缓解全量训练导致的风格漂移。

### 6.3 GGM + SSM++ 双记忆

![](assets/HY-World%202.0%20-%20多模态三维世界生成重建与模拟/figures/03-worldstereo-pipeline.png)

> Figure 7：右侧 GGM 把全景点云沿目标轨迹渲染为粗几何引导；左侧 SSM++ 从 memory bank 检索视图，与目标关键帧横向拼接并直接进入主 DiT。前者守全局结构，后者补局部纹理。来源：论文 Figure 7，PDF p.10。

**GGM** 把参考点云与额外视角点云拼成

$$P^{glo}=[P^{ref},\hat P]\in\mathbb{R}^{(N+\hat N)\times3},$$

训练模型真正利用 360° 点云，而不是只把它当“软相机提示”。推理时用全景点云 $P^{pan}$。

**SSM++** 做四件事：

1. 去掉独立 memory branch，检索关键帧直接进主 DiT；
2. target 与 retrieved view 沿宽度拼成 $2W$，并共享同一 temporal RoPE index；
3. 只检索最相关关键帧，降低冗余；
4. pose 归一化为 quaternion + translation 的 7D 向量，经 3 层 MLP、zero-init 加到 keyframe token。

真实多视图训练以 30%–90% 时间重叠做 temporally misaligned retrieval；UE 多轨迹数据按 3D-FoV 检索。GGM 的深度被下采样、轻微 blur 或保留原始 floaters；SSM++ 参考帧做 motion blur、color jitter 和随机 crop，以模拟推理噪声。论文发现过强 point-cloud distortion 反而破坏跨视频几何。

### 6.4 4-step DMD

$$\nabla\mathcal{L}_{DMD}=-\mathbb{E}_t\left[\int(s_{real}(x_t,t)-s_{fake}(x_t,t))\frac{dx_t}{d\theta}\,dz\right].$$

Student、real score 和 fake score 都从 middle-training VDM 初始化；real 冻结，另外两者训练，fake 每次 generator update 更新 5 次。采用 stochastic gradient truncation，省略收益不显著但拖慢训练的 GAN loss。最终 WorldStereo 2.0 是 4-step DiT。

这与 WorldPlay 的 Context Forcing 不同：2.0 的目标是多轨迹关键帧与后续重建，报告没有构造 WorldPlay 那种长期 AR student/teacher 记忆上下文对齐。

### 6.5 WorldStereo 主结果与消融

在更难闭环/可探索轨迹上，单视图生成重建：

| 方法 | T&T F1↑ / AUC↑ | MipNeRF360 F1↑ / AUC↑ |
|---|---:|---:|
| Gen3C | 31.24 / 42.44 | 35.26 / 52.10 |
| FlashWorld | 22.29 / 30.45 | 42.60 / 53.86 |
| WorldStereo 2.0 | 41.43 / 58.19 | **51.27 / 65.79** |
| WorldStereo 2.0 DMD | **43.16 / 60.09** | 50.52 / 65.64 |

DMD 在 T&T 提升 recall/F1，在 MipNeRF360 略低于未蒸馏模型，说明 4-step 没有“无损统治”。

Camera benchmark 中，WorldStereo 2.0 的 RotErr/TransErr/ATE 为 **0.492/0.968/1.768**，优于旧 WorldStereo 的 0.762/1.245/2.141；但 CLIP-IQA+ 0.544 略低于旧版 0.547 和 WorldPlay 0.552。

Memory 消融最有信息量：

- camera-only → GGM+SSM++：PSNR **16.13→20.94**，SSIM 0.474→0.640，LPIPS 0.349→0.170；
- 空间拼接改成 temporal concatenation：PSNR **21.06→19.83**，ATE 0.048→0.114，证明 SSM 的 stereo-like 对齐不是装饰；
- batch 64 最终 memory model：21.63 / 0.669 / 0.156；
- DMD 后：PSNR 21.84、masked consistency 30.93/0.656，但 LPIPS 0.165、ATE 0.072，部分指标回退。

Camera adaptation 也揭示反直觉结论：全量解冻给最高部分视觉指标，却过拟合并漂移风格；冻结 cross-attention + FFN 才得到最低三项相机误差和最高 64.39% 画质偏好。

---

## 7. WorldMirror 2.0：统一多模态前馈重建

![](assets/HY-World%202.0%20-%20多模态三维世界生成重建与模拟/figures/04-worldmirror-architecture.png)

> Figure 12：图像、pose、intrinsics、depth 都被 token 化并合入共享序列，Transformer 后接 point map、camera、depth、normal、3DGS 五类 DPT heads。输出不仅服务独立重建，也为生成链的 3DGS 初始化和监督提供几何。来源：论文 Figure 12，PDF p.15。

### 7.1 Any-Modal Tokenization 与输出

每种几何 prior 在训练时独立以 0.5 概率 dropout，因此推理可以给任意组合。共享 backbone 同时预测点图、深度、法线、相机和 3DGS 属性。相比 1.0，2.0 增加：

- normalized RoPE；
- depth-to-normal coupling；
- 显式 depth validity mask；
- UE 合成几何与 normal pseudo-label；
- token-budget dynamic batching；
- 3-stage curriculum；
- token/frame sequence parallel、BF16、FSDP。

### 7.2 Normalized RoPE

将 patch 绝对整数位置改成与分辨率无关的坐标：

$$\hat x_i=\frac{2i+1}{H_p}-1,\qquad \hat y_j=\frac{2j+1}{W_p}-1.$$

所有分辨率都落在 $[-1,1]$，高分辨率只是更密采样而不是位置外推。论文分析中跨分辨率 encoding cosine similarity 保持 $>0.95$；这解释了 WorldMirror 1.0 高分辨率崩溃而 2.0 稳定。

### 7.3 深度—法线耦合与无效像素

预测深度先反投影为点，再由局部导数叉积得到法线：

$$\tilde N_i(x)=\operatorname{normalize}\left(\frac{\partial P_i}{\partial u}\times\frac{\partial P_i}{\partial v}\right),\quad P_i=K^{-1}\hat D_i[u,v,1]^T.$$

以角度损失监督：

$$\mathcal{L}_{d2n}=\frac{1}{|\mathcal V|}\sum_{x\in\mathcal V}\arccos\frac{\tilde N_i(x)\cdot\hat N_i(x)}{\|\tilde N_i(x)\|\|\hat N_i(x)\|}.$$

合成数据的 normal 来自 GT depth，真实数据用单目 normal teacher。作者刻意不采用逐视图 pseudo-depth，因为它会制造多视图分层伪影。另一个 mask head 用 BCE 预测有效深度，替代只在训练损失里学 confidence、推理还需手工阈值的旧做法。

### 7.4 多分辨率与先验结果

- RealEstate10K camera AUC@30：WorldMirror 1.0 高分辨率 66.29，2.0 为 **86.89**；
- Depth AbsRel（高分辨率）：0.195→**0.162**；
- NVS 高分辨率 PSNR：17.78→**19.98**，SSIM 0.659→**0.726**；
- 7-Scenes point map Acc.（M）：0.043→**0.033**；带全部 priors 的 H 配置为 **0.012**；
- DTU 带全部 priors H：Acc. **0.554**、Comp. **0.771**。

需要保留两个限定：WorldMirror 2.0 无先验低分辨率并非每项都优于 1.0；最强结果经常依赖 all priors，因此不能把 prior-guided SOTA 当成纯图像输入能力。

### 7.5 效率

$518\times378$、128 views：单卡 FP32 为 59.26 GB / 18.00 s；4×H20 的 SP+BF16+FSDP 为 42.71 GB/卡、5.60 s。256 views 时完整配置为 78.78 GB/卡、17.52 s；单卡 FP32 OOM。这里的“高效”是相对旧重建器的分布式可扩展性，不是消费级单卡。

---

## 8. Stage IV：点云对齐与 3DGS composition

### 8.1 深度对齐

WorldMirror 对生成关键帧估计 $D_i^m,N_i^m$，但深度有 scale ambiguity。系统从全景点云渲染 guidance depth $D_i^g$，在可靠 mask 上做 RANSAC 线性对齐：

$$D_i^a=\gamma_iD_i^m+\beta_i.$$

有效 mask 是 WorldMirror confidence、guidance validity、normal consistency、percentile depth filter 与 non-sky mask 的交集。全序列用 9 个 anchor depths 检查 $(\gamma_i,\beta_i)$ 的全局离群；超过 90th percentile 的系数替换为最近 inlier，整段全异常则丢弃。

该机制快，但失败条件也很清楚：重叠过少、轨迹极端或全景点云错误时，线性标定失去锚点。论文明确承认 WorldMirror 在挑战性户外场景仍会失败。

### 8.2 3DGS 目标

颜色损失结合 $L_1$、SSIM 与 LPIPS；几何损失结合 sparse aligned depth 和 dense normal：

$$\mathcal{L}_{geo}=\lambda_d\|\hat D_i-D_i^a\|_1+\lambda_n(1-\cos(\hat N_i,N_i)).$$

总目标为：

$$\mathcal{L}_{GS}=\mathcal{L}_{color}+\mathcal{L}_{geo}+\mathcal{L}_{reg}+\mathcal{L}_{mask}.$$

MaskGaussian 以 Gumbel-Softmax 学习每个 Gaussian 的存在变量 $M_k$，并用平均激活平方促稀疏；densification 只用于 non-sky，避免无深度天空产生 floaters。

### 8.3 3DGS 消融

| 配置 | Gaussian 数 | PSNR↑ | LPIPS↓ |
|---|---:|---:|---:|
| 6M baseline | 6.000M | **25.176** | **0.209** |
| voxel only | 1.000M | 24.504 | 0.276 |
| voxel + densification | 5.254M | 25.158 | 0.210 |
| + MaskGaussian | 1.383M | 25.017 | 0.216 |
| + non-sky densification | **1.381M** | 25.023 | 0.215 |

完整配置相对 6M baseline 减少约 **77%** Gaussians，仅损失 0.153 dB PSNR；相对 5.254M densification 减少 73.7%，质量接近。这是论文最清晰的质量—效率消融。

### 8.4 端到端时延

在 NVIDIA H20 系统上单世界：Panorama 15 s、Trajectory 182 s、World Expansion 286 s、Recon & Align 102 s、3DGS 127 s，总计 **712 s（约 11.9 分钟）**。正文称“only 10 minutes”是近似表述；表格总和应作为精确核验值。

WorldLens 的实时碰撞和渲染发生在资产完成后，不能把 712 秒离线生成写成实时世界生成。

---

## 9. 与 1.0、1.5 / WorldPlay 的范式差别

| 维度 | HunyuanWorld 1.0 | HY-World 1.5 / WorldPlay | HY-World 2.0 |
|---|---|---|---|
| 根本问题 | 从文本/单图制作可导出世界资产 | 用户动作后立刻看到下一段世界 | 统一生成与重建并得到高质量显式 3D |
| 主表示 | 语义分层 mesh，可选分层 3DGS | RGB 视频 chunk + pose-aware 历史图像记忆 | 3DGS + 点云/深度/法线/相机，附 coarse mesh |
| 时序方式 | 离线级联 | 在线 chunk-wise AR diffusion | 离线多轨迹关键帧生成 + 前馈重建 + GS 优化 |
| 输入 | 文本/单图 | 初始文本/图 + 键鼠/相机动作 | 文本/单图/多图/视频 + 可选几何 priors |
| 生成未见区域 | 全景 inpainting、可选 Voyager | 自回归视频预测 | WorldStereo 2.0 多轨迹关键帧 + 双记忆 |
| 几何 | 单目深度 + 跨层对齐 + sheet warp | 隐式视频一致性；3D 是可选后处理 | WorldMirror 2.0 + depth alignment + optimized 3DGS |
| 实时声明 | 未给端到端生成数字 | 24 FPS，依赖 8×H800 与工程优化 | 生成 712 s；生成后 WorldLens 可实时交互 |

**问题演化**：1.0 解决“有资产”；1.5 解决“有即时反馈”；2.0 解决“资产既能由稀疏条件想象，也能由密集观测重建，并以同一 3D 表示交付”。2.0 没有替代 1.5 的实时交互，它替代的是 1.0 较薄弱的显式世界构建链。

---

## 10. 局限、失败案例与 Future Work

### 10.1 论文明确证据

论文没有单列 limitations / future work 章节。正文明确暴露的故障包括：

- aggressive point-cloud augmentation 会削弱 GGM，导致跨视频几何不一致；
- 全量 DiT 微调会过拟合并产生全局风格漂移；
- WorldMirror 在高挑战户外场景仍有困难；
- depth alignment 在重叠 mask 稀疏或轨迹困难时会产生错误系数；
- 标准 densification 在天空区域产生严重 floaters；
- WorldMirror 2.0 在部分低分辨率无先验配置未必优于 1.0。

### 10.2 结构性局限

1. **不是端到端模型**：四阶段和多种第三方几何工具使部署、版本和误差诊断复杂。
2. **生成后再重建的双重不确定性**：WorldStereo 幻觉先成为“观测”，WorldMirror/3DGS 再把它固化成几何。
3. **初始全景仍是全链锚点**：接缝、语义或深度错误会影响 WorldNav、GGM、alignment 与最终资产。
4. **动态世界能力弱**：最终 3DGS 主要表示静态场景；角色和碰撞由 WorldLens/外部引擎提供。
5. **闭源对照不充分**：与 Marble 只有精选定性图，没有可复现实验接口或盲测统计。
6. **成本仍高**：80B HY-Pano、17B WorldStereo、1.2B WorldMirror、H20 多卡、3DGS 优化与原生扩展均提高复现门槛。
7. **数据披露不足**：全景规模、UE 内部数据、总 GPU-hours 和完整训练资源未给出。

### 10.3 个人判断

论文最重要的范式贡献是承认“世界生成”不应只靠一个长视频，也不应只靠一次单图 3D 回归：先让规划器决定应观察哪里，再让生成模型产生**为重建优化的关键帧**，最后让重建器把生成内容约束成显式 3D。Keyframe-VAE、GGM/SSM++ 和 WorldMirror 的接口设计比“多模态世界模型”这一宏大标签更有技术价值。

但 2.0 仍是**生成式静态世界资产工厂**，不是能持续接受动作、保持对象状态并预测物理后果的在线 simulator。将 WorldPlay 的流式观测与 2.0 的持久 3DGS/对象状态结合，才可能形成同时实时且可验证的下一代路线。

---

## 11. 复现与开源状态

### 11.1 当前仓库可确认内容

- **WorldMirror 2.0**：约 1.2B，Python API、CLI、FSDP、多分辨率与模型权重入口；
- **HY-Pano 2.0**：约 80B，代码文档和权重入口；
- **WorldStereo 2.0**：约 17B，world-generation 流程与权重入口；
- **3DGS composition**：数据准备、场景优化与定制 `gsplat`；
- 推荐 CUDA 12.8、Python 3.11+，需要 FlashAttention、自定义 gsplat 和 Recast Navigation 扩展。

### 11.2 “全部开源”的核验边界

论文明确宣称发布全部模型、代码和技术细节；仓库新闻显示它经历 partial code → WorldMirror → HY-Pano → 完整 generation 代码的分阶段开放。当前主要入口已存在，但仍需注意：

- WorldNav 的具体 VLM 依赖和独立权重没有像其他模型那样清晰列入 Model Zoo；
- `HY-Pano-2-Qwen` 链接表现为 LoRA 文件，基础模型和独立运行方式需查子文档；
- WorldStereo 权重位于外部账号目录时应核对 commit/版本；
- mesh extraction 在论文中明确，但仓库顶层 README 对导出操作链披露不如 3DGS 流程清晰；
- 代码、权重和第三方模型可能受不同许可证约束。

因此最准确的表述是：**核心组件和主要推理链已公开，但不等于单仓、无外部资产、消费级硬件的开箱复现。**

### 11.3 推荐复现层级

1. **WorldMirror 单组件**：用公开多视图基准复现 M/H 分辨率与 priors ablation；
2. **WorldStereo**：在固定 panorama/trajectory 上测 GGM、SSM++ spatial vs temporal concatenation、DMD；
3. **3DGS**：复现 6M baseline、voxel、densification、MaskGaussian 四组质量—点数曲线；
4. **端到端小场景**：记录每阶段时延、峰值显存、失败类型和最终碰撞 mesh；
5. **完整训练**：需要内部/UE 数据、大规模视频扩散训练与 H20 级多卡，普通实验室难以严格复刻。

---

## 12. 后续问题

1. 如何把 3DGS 从静态辐射表示升级为持久对象状态和可编辑 scene graph？
2. WorldNav 是否可以用 reconstruction uncertainty 学习规划，而不是固定五类启发式轨迹？
3. 生成关键帧作为“伪观测”时，能否把不确定性传到 WorldMirror/3DGS，避免幻觉被硬固化？
4. 如何把 WorldPlay 的低延迟动作接口接到 2.0 的显式地图，并保持生成与渲染状态同步？
5. 评测是否应加入 agent task success、碰撞正确率、闭环状态误差和资产导入成功率，而不只看图像/点云指标？

---

## 13. 材料来源

完整来源、图片映射与核验记录见 [sources.md](assets/HY-World%202.0%20-%20多模态三维世界生成重建与模拟/sources.md)。
