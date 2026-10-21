---
title: "3DGS - 实时辐射场渲染的三维高斯表示"
aliases:
  - "3DGS"
  - "3D Gaussian Splatting"
  - "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
authors:
  - Bernhard Kerbl
  - Georgios Kopanas
  - Thomas Leimkühler
  - George Drettakis
year: 2023
venue: "SIGGRAPH 2023 / ACM Transactions on Graphics 42(4)"
arxiv: "2308.04079"
doi: "10.1145/3592433"
tags:
  - paper
  - 3d-reconstruction
  - novel-view-synthesis
  - radiance-field
  - 3dgs
  - gaussian-splatting
  - real-time-rendering
status: detailed-read
created: 2026-10-15
updated: 2026-10-15
source: "https://arxiv.org/abs/2308.04079"
project: "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
code: "https://github.com/graphdeco-inria/gaussian-splatting"
---

# 3D Gaussian Splatting：实时辐射场渲染的三维高斯表示

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *3D Gaussian Splatting for Real-Time Radiance Field Rendering* |
| 作者 | Bernhard Kerbl, Georgios Kopanas（共同一作）, Thomas Leimkühler, George Drettakis |
| 机构 | Inria、Université Côte d’Azur、MPI Informatik |
| 发表 | SIGGRAPH 2023；ACM TOG 42(4)，Article 139 |
| 任务 | 从带相机标定的多视图照片建立静态场景辐射场，用于高质量新视角合成 |
| 输入 | RGB 图像、相机内外参、SfM 顺带产生的稀疏点云 |
| 输出 | 每场景优化得到的 1–5M 个各向异性 3D Gaussians |
| 本地材料 | `assets/3DGS - 实时辐射场渲染的三维高斯表示/sources/paper.pdf`、`arxiv-source.tar` |

## 一句话结论

3DGS 观察到 NeRF 的连续隐式场虽然质量高，却因逐射线采样和网络查询难以实时，而传统点/mesh rasterization 又难以从稀疏 SfM 观测直接优化出高质量辐射场，于是以可学习的**各向异性 3D Gaussians**显式存储位置、协方差、不透明度和球谐颜色，并通过 gradient-driven clone/split/prune 自适应分配百万级 primitives，再用 visibility-aware tile rasterizer、全局 radix sort 和前向 alpha compositing 实现可微高速渲染。它在约 **35–45 分钟** per-scene 优化后达到接近或超过当时 NeRF SOTA 的质量，并在 1080p 实现约 **134–154 FPS**；代价是数百 MB 存储、弱观测区域与 depth-order popping，以及缺少 mesh 拓扑和可直接用于物理仿真的表面语义。

![3DGS teaser](assets/3DGS%20-%20实时辐射场渲染的三维高斯表示/figures/fig1_teaser.png)

*图 1 解读：同一 Bicycle 测试视图上，3DGS 在 6 分钟即可达到 23.6 dB/135 FPS，充分训练 51 分钟达到 25.2 dB/93 FPS；图中速度来自特定场景与硬件，论文的通用“实时”主张仍是 1080p ≥30 FPS。来源：论文 Figure 1，PDF Page 1；官方源码 `figures/teaser/teaser_02.pdf`。*

## 2. 问题背景、核心洞察与贡献

### 2.1 NeRF 质量高，但图像形成过程不适合实时显示

NeRF 把场景写成连续函数：给定位置与方向，网络输出密度和颜色；渲染时沿每条相机射线采样并做体渲染。连续表示利于梯度优化，却有两个工程代价：一是必须在大量空空间中寻找有效样本，二是每帧都要重复网络查询与 ray marching。Mip-NeRF 360 在无界真实场景上质量很高，但论文采用的对照训练口径约 48 GPU-hours、渲染约 10 s/frame；Instant-NGP、Plenoxels 虽把训练压到分钟级，1080p 质量和帧率仍未同时满足实时要求。

传统 mesh/point rasterization 正相反：primitive 显式、GPU 管线高效，但从稀疏 SfM 点直接优化出高质量辐射场并不容易。3DGS 的关键洞察是：**NeRF 的体渲染与有序点的 alpha compositing 可以写成同一图像形成公式；差别主要在如何得到并遍历样本。** 因而可以把样本显式化为可增删、可移动、可投影的 3D Gaussians，避免每帧重新 ray-march。

### 2.2 三项核心贡献

1. **各向异性 3D Gaussian 辐射场**：每个 primitive 同时携带几何范围、不透明度与方向相关颜色；它是连续可微的体 primitive，又能直接投影为 2D 椭圆 splat。
2. **优化与密度控制交替进行**：从 SfM 稀疏点出发，通过 gradient-based clone、split、prune 和 opacity reset 自动生成适合当前场景的百万级 primitive 集合。
3. **可见性感知的可微 tile rasterizer**：以 tile–Gaussian 实例化、GPU radix sort、前向 alpha compositing 和反向遍历实现快速训练与实时渲染，并允许所有实际参与混合的 Gaussian 接收梯度。

这三项缺一不可：只有 Gaussian 表示而没有增密，稀疏 SfM 点无法覆盖细节；只有表示与增密而没有专用 rasterizer，仍无法达到实时；只有快速 rasterizer 而限制反向传播深度，又会让优化不稳定。

## 3. Gaussian 参数化：一个 primitive 究竟存什么

### 3.1 每个 Gaussian 的参数

第 $i$ 个 Gaussian 可写为

$$
\mathcal G_i=\{\mu_i,s_i,q_i,\alpha_i,\mathbf h_i\},
$$

其中：

- $\mu_i\in\mathbb R^3$：世界坐标中的均值/中心；
- $s_i\in\mathbb R^3$：三个主轴尺度；
- $q_i\in\mathbb R^4$：旋转四元数；
- $\alpha_i\in[0,1)$：primitive opacity；
- $\mathbf h_i$：RGB 球谐（Spherical Harmonics, SH）系数，用于视角相关颜色。

以 $\mu$ 为中心的未归一化 Gaussian 为

$$
G(x)=\exp\left[-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right].
$$

论文公式把 $x$ 理解为相对均值的局部坐标，因此排版中省略了 $x-\mu$。渲染某像素时，屏幕空间 Gaussian 值再乘学习到的 $\alpha$，才得到该 primitive 对该像素的有效 alpha。

### 3.2 为什么不直接学习协方差矩阵

完整 $3\times3$ 协方差 $\Sigma$ 必须半正定；直接对九个矩阵元素做无约束梯度下降容易得到非法矩阵。论文将其分解为

$$
\Sigma=RSS^\top R^\top,
$$

其中 $S=\operatorname{diag}(s_x,s_y,s_z)$，$R=R(q)$。实现中归一化四元数以保证合法旋转，对 scale 使用 exponential activation、对 opacity 使用 sigmoid：

$$
s=\exp(\tilde s),\qquad \alpha=\sigma(\tilde\alpha).
$$

于是任意优化变量都能映射到合法椭球。各向异性的重要性不只是“多三个自由度”：一个扁平、旋转后的椭球能贴住墙面、叶片或细杆，在相同 primitive 数下比球形点更高效地覆盖局部表面。

### 3.3 颜色为何使用 SH

颜色不是固定 RGB，而是视线方向 $d$ 的低阶球谐函数 $c_i(d)$，用来表达非朗伯反射和视角相关高光。训练开始只启用零阶 diffuse/base color，之后每 1000 iterations 增加一阶，最终启用 4 个 SH bands。这样可避免角度覆盖不足时高频系数过早主导、连基础颜色都学错。

### 3.4 初始化

真实场景以 COLMAP/SfM 点初始化 $\mu$ 和基础颜色；每个点先设为各向同性 Gaussian，三个轴的初始尺度取最近三个点距离的均值。SfM 的价值不是提供精密表面，而是把 primitive 放到“有证据的占据区域”。Synthetic NeRF 场景因相机覆盖完整、背景受限，随机初始化也能成功；无界真实场景随机初始化则容易在弱覆盖背景留下 floaters。

## 4. 从 3D 椭球到像素：投影、排序与 alpha compositing

### 4.1 投影到屏幕空间

给定世界到相机的 viewing transform $W$，透视投影在 Gaussian 均值附近的一阶 Jacobian 为 $J$。3D 协方差经局部仿射近似变换为

$$
\Sigma'=JW\Sigma W^\top J^\top.
$$

取 $\Sigma'$ 的左上 $2\times2$ 子矩阵，得到屏幕空间椭圆协方差 $\Sigma_{2D}$。直觉上，$W$ 先把椭球变到相机坐标，$J$ 再把透视除法在均值附近线性化；因此一个倾斜、细长的世界空间椭球会变成带旋转和长短轴的 2D splat。该方法源自 EWA splatting，也解释了为什么 3DGS 不需要显式表面法线。

对像素坐标 $u$，其屏幕空间权重近似为

$$
g_i(u)=\exp\left[-\frac12(u-\mu_{i,2D})^\top\Sigma_{i,2D}^{-1}(u-\mu_{i,2D})\right],
\qquad \hat\alpha_i(u)=\alpha_i g_i(u).
$$

### 4.2 NeRF 与 splatting 的同构关系

NeRF 体渲染离散形式为

$$
C=\sum_i T_i\alpha_i c_i,\qquad
T_i=\prod_{j<i}(1-\alpha_j).
$$

按深度排序的 Gaussian splatting 使用完全相同的 front-to-back over operator：

$$
C(u)=\sum_{i\in\mathcal N(u)} c_i(d)\,\hat\alpha_i(u)
\prod_{j<i}\left(1-\hat\alpha_j(u)\right).
$$

$\mathcal N(u)$ 是覆盖像素 $u$ 的有序 Gaussian 集合。两者的主要区别不在合成公式，而在**样本如何产生**：NeRF 运行时沿每条 ray 找样本；3DGS 已把样本显式存好，只需投影和 rasterize。

### 4.3 可见性排序是近似而非严格几何求交

3DGS 不为每个像素精确排序椭球，而是按 Gaussian 的 view-space mean depth 为 tile 排序。一个大椭球覆盖多个像素时，各像素上的真实前后关系可能不同，因此该排序并非数学上逐像素精确。论文经验是收敛后多数 splat 足够小，近似不会产生明显伪影；但大 Gaussian 或弱观测区域会出现 depth-order switching 和 popping，这正是方法的已知边界。

## 5. Visibility-aware tile-based rasterizer

![3DGS 优化与渲染总览](assets/3DGS%20-%20实时辐射场渲染的三维高斯表示/figures/fig2_overview.png)

*图 2 解读：黑线是正向操作流，蓝线是梯度流；同一可微 rasterizer 同时服务训练与推理，优化参数和 density control 都围绕渲染误差闭环更新。来源：论文 Figure 2，PDF Page 5；官方源码 `figures/overview/overview_01.pdf`。*

### 5.1 前向流程

1. **Frustum/guard-band culling**：只保留 99% confidence interval 与视锥相交的 Gaussians；均值过近 near plane 或远离视锥者被 guard band 拒绝，避免投影协方差数值爆炸。
2. **投影与 footprint 计算**：得到 $\mu_{2D}$、$\Sigma_{2D}$ 和覆盖的 tile 范围。
3. **按 tile 复制实例**：屏幕划为 $16\times16$ pixel tiles；一个 Gaussian 覆盖几个 tile，就生成几个 Gaussian–tile 实例。
4. **构造排序 key**：高位编码 tile ID，低 32 bit 编码 view-space depth。
5. **一次 GPU radix sort**：使用 CUB 对所有实例全局排序；排序后同 tile 连续、tile 内近似 front-to-back。
6. **识别 tile range**：比较相邻 key 即可得到各 tile 在排序数组中的起止区间。
7. **协作式 rasterization**：每个 tile 由一个 thread block 处理，分批把 Gaussian 数据载入 shared memory；每个 pixel thread 顺序求椭圆权重并做 alpha compositing。

关键优化是**不做 per-pixel sort**。排序单位从“像素×所有候选”降为“Gaussian–tile 实例的一次全局 radix sort”，并通过 tile shared memory 复用 Gaussian 数据。

### 5.2 Early termination 与数值规则

实现跳过 $\hat\alpha<1/255$ 的微小贡献，将单个 alpha clamp 到 0.99；若加入下一个 Gaussian 后累计 opacity 将超过 0.9999，则停止该像素的 front-to-back 遍历。tile 内所有 pixel 都饱和后，整个 block 提前结束。

### 5.3 为什么 backward 也是贡献

若只让最前面的固定 $N$ 个 splats 得到梯度，后方 primitive 永远没有机会移动、变透明或补充颜色，优化会被当前错误遮挡锁死。论文反向复用同一排序数组与 tile ranges，从最后一个有贡献的 Gaussian 向前遍历，并利用 forward 保存的最终累计 opacity/transmittance 逐步恢复中间 compositing 系数；这样无需保存每像素任意长的 contributor list，却能让所有实际参与混合的 Gaussian 接收梯度。

消融把梯度限制为前 10 个 splats（已是 Pulsar 默认值的两倍），Truck 仍下降约 11 dB PSNR，说明“unlimited depth complexity of splats with gradients”不是实现细节，而是高质量训练的必要条件。

## 6. Per-scene 优化、训练目标与自适应密度控制

### 6.1 训练目标与 schedule

每次 iteration 随机取一个训练相机，渲染当前 Gaussian 集合，与真值图像比较：

$$
\mathcal L=(1-\lambda)\mathcal L_1+\lambda\mathcal L_{\mathrm{D-SSIM}},
\qquad \lambda=0.2.
$$

也就是 0.8 L1 + 0.2 D-SSIM。Adam 更新位置、尺度/旋转、opacity 和 SH；position 单独使用指数衰减学习率。训练初始图像分辨率缩小 4 倍，在 iteration 250、500 两次上采样，以减少早期不稳定和计算量。完整模型通常训练 30K iterations；7K 是快速版本。

### 6.2 为什么优化过程中必须改变 primitive 数量

固定 SfM 点数无法同时处理两种失败：

- **under-reconstruction**：细杆、叶片、纹理边界附近 primitive 太少；
- **over-reconstruction**：一个大 Gaussian 粗略盖住高频结构，能降低一部分损失，却无法表达细节。

两者都会产生较大的 view-space position gradient，因为优化在强烈要求 Gaussian 移动。论文每 100 iterations 检查平均 view-space position gradient，阈值为 $\tau_{pos}=0.0002$，再按 Gaussian 尺度区分 clone 与 split。

![自适应密度控制](assets/3DGS%20-%20实时辐射场渲染的三维高斯表示/figures/fig3_density_control.png)

*图 3 解读：上排对小 Gaussian 做 clone 并沿位置梯度移动，增加覆盖总体积；下排把过大的 Gaussian split 为两个更小 primitive，近似保持总体积但提高局部自由度。来源：论文 Figure 4，PDF Page 6；官方源码 `figures/density_control/density_control_01.pdf`。*

### 6.3 Clone、split、prune 与 opacity reset

- **Clone**：高梯度且尺度小，复制同尺寸 Gaussian，并把副本沿 position gradient 方向移动；适合补齐薄结构和空洞。
- **Split**：高梯度且尺度大，用两个 Gaussian 替换原 primitive；新位置从原 3D Gaussian 的 PDF 采样，尺度除以经验因子 $\phi=1.6$；适合把模糊大块拆成细节。
- **Prune**：删除 opacity 低于 $\epsilon_\alpha$ 的 Gaussian，也删除 world-space 过大或 view-space footprint 过大的 Gaussian。论文未在正文给出所有 size/opacity 阈值的具体数字。
- **Opacity reset**：每 3000 iterations 把 opacity 设到接近 0。真正必要的 primitive 会重新长出 opacity，无贡献点则在下一轮 pruning 被删掉；该机制也抑制靠近输入相机的 floaters 和 Gaussian 数量无界增长。

![密度控制消融](assets/3DGS%20-%20实时辐射场渲染的三维高斯表示/figures/fig4_densification_ablation.png)

*图 4 解读：无 split 时背景由大 splat 模糊覆盖；无 clone 时自行车细杆/辐条附近欠覆盖；完整策略同时改善背景频率和细结构。来源：论文 Figure 8，PDF Page 10；官方源码 `figures/ablations/densification.pdf`。*

### 6.4 方法的本质：拓扑可变的显式优化

虽然 Gaussian 没有 mesh connectivity，但 clone/split/prune 等价于在训练中改变表示“拓扑”：模型容量被按误差和梯度动态分配。它比固定 voxel/hash grid 更稀疏，又比固定 point cloud 更能适应不同频率区域。代价是训练状态动态、显存峰值与最终 primitive 数依赖场景和阈值，复现时不能只记录学习率，还必须记录整个 densification schedule。

## 7. 实验、主要结果与消融证据

### 7.1 数据集与评价

真实场景共 13 个：Mip-NeRF360 全部 9 个场景、Tanks and Temples 的 Truck/Train、Deep Blending 的 Playroom/DrJohnson；每第 8 张图作为测试视图，其余训练，并保持原始分辨率评测。另在 8 个 Synthetic NeRF/Blender 场景测试随机初始化。指标为 PSNR、SSIM、LPIPS、训练时间、FPS 和模型存储。

### 7.2 主结果

| 数据集 | 方法 | SSIM↑ | PSNR↑ | LPIPS↓ | 训练 | FPS↑ | 模型存储 |
|---|---|---:|---:|---:|---:|---:|---:|
| Mip-NeRF360 | Mip-NeRF360 | 0.792 | **27.69** | 0.237 | 48h | 0.06 | 8.6MB |
| Mip-NeRF360 | **3DGS-30K** | **0.815** | 27.21 | **0.214** | 41m33s | **134** | 734MB |
| T&T | Mip-NeRF360 | 0.759 | 22.22 | 0.257 | 48h | 0.14 | 8.6MB |
| T&T | **3DGS-30K** | **0.841** | **23.14** | **0.183** | 26m54s | **154** | 411MB |
| Deep Blending | Mip-NeRF360 | 0.901 | 29.40 | 0.245 | 48h | 0.09 | 8.6MB |
| Deep Blending | **3DGS-30K** | **0.903** | **29.41** | **0.243** | 36m02s | **137** | 676MB |

最重要的结论不是“所有 PSNR 都第一”：Mip-NeRF360 上 3DGS 的 PSNR 略低 0.48 dB，但 SSIM/LPIPS 更好，而且渲染快约三数量级。显式表示的交换条件也很清楚：质量与速度换来数百 MB 模型，而 Mip-NeRF360 只有 8.6 MB。

7K 版本通常训练 5–10 分钟，质量接近 Instant-NGP/Plenoxels，却已达到 160–197 FPS；30K 再消除背景伪影。Synthetic NeRF 从 100K 随机 Gaussians 开始，早期 prune 到约 6–10K 有效点，最终增长至 200–500K，平均 PSNR 33.32、渲染 180–300 FPS。

### 7.3 消融的因果含义

在 Truck/Garden/Bicycle 三场景上，30K 平均 PSNR：

| 变体 | 平均 PSNR↑ | 相对 Full | 说明 |
|---|---:|---:|---|
| Limited-BW（仅前 10 个点有梯度） | 19.19 | -6.86 | 错误遮挡无法被后方 primitive 修正 |
| Random Init | 20.42 | -5.63 | 真实无界背景缺少几何锚点，floaters 难清理 |
| No-Split | 23.90 | -2.15 | 大 splat 无法细分，背景/高频结构退化最明显 |
| Isotropic | 25.23 | -0.82 | 球形 primitive 不能紧贴表面 |
| No-SH | 25.35 | -0.70 | 不能表达视角相关外观 |
| No-Clone | 25.91 | -0.14 | 最终数值差距较小，但薄结构与早期收敛变差 |
| **Full** | **26.05** | — | 完整方法 |

![各向异性协方差消融](assets/3DGS%20-%20实时辐射场渲染的三维高斯表示/figures/fig5_anisotropy_ablation.png)

*图 5 解读：相同 Gaussian 数预算下，Full 的扁平椭球能贴合叶片、车身与轮胎边缘；Isotropic 只能用球形模糊覆盖，细结构和轮廓均明显退化。来源：论文 Figure 10，PDF Page 11；官方源码 `figures/ablations/isotropy_vs_anisotropy2.pdf`。*

实验总体充分支撑三项主张：表示选择、动态密度控制和 rasterizer 反向设计都不是可互换的小技巧。但质量指标仍主要是图像空间 NVS；论文没有证明 Gaussians 的中心或椭球轴就是度量准确的表面几何。

## 8. 失败案例、局限与复现成本

### 8.1 论文明确展示的失败

1. **弱覆盖区域**：训练视图几乎看不到的背景会形成粗大、拉长或斑块状 Gaussians；其他 NeRF 方法也困难，但错误形态不同。
2. **Popping**：大 Gaussian 穿过 guard band，或 center-depth 排序突然交换 blending order，会在相机移动时跳变。
3. **无正则化**：原始方法没有几何、尺度或时序正则，未观察区和大 splat 只能靠图像损失间接约束。
4. **超大场景超参数不通用**：城市级场景可能需要降低 position learning rate；“所有场景同一参数”不能外推到任意尺度。
5. **静态、已标定场景假设**：动态物体、滚动快门、曝光变化和错误相机位姿都会被优化吸收到颜色、opacity 或伪几何中。

### 8.2 更深层的表示局限

- **可渲染不等于可编辑表面**：Gaussian 是体 primitive，没有拓扑、语义、watertight surface 或碰撞边界；直接用于 CAD/物理仿真并不可靠。
- **排序近似不可交换**：alpha compositing 非交换，中心深度排序错误会改变颜色；该误差在大 splat、交叉几何与透明层中最明显。
- **视角外推仍由训练覆盖决定**：显式表示提升了速度，不会自动解决未观测区域；它优化的是对训练图像成立的辐射解释。
- **内存随场景复杂度增长**：真实场景通常 1–5M Gaussians，论文模型约数百 MB；原始 prototype 大场景训练峰值可超过 20GB，rasterizer 另需约 30–500MB。

### 8.3 原论文复现成本

官方实现采用 PyTorch + 自定义 CUDA extension + CUB + COLMAP/SfM + SIBR viewer。接近论文设置的环境是 CUDA 11（作者使用 11.8）、Compute Capability 7.0+；官方 README 建议 24GB VRAM 以达到论文评测质量。默认 30K iterations，论文场景约 35–45 分钟；7K 快速配置约 5–10 分钟。官方完整评测脚本在作者 A6000 上约 7 小时，不含 COLMAP、数据下载和编译排障。

严格复现应固定：代码 commit、CUDA/PyTorch/编译器、输入图像缩放、COLMAP 结果、随机种子、30K/7K、densification 起止与间隔、gradient/size/opacity 阈值、SH schedule、是否使用 `--eval`。当前官方仓库后来加入训练加速、depth regularization、exposure compensation、antialiasing 和 sparse Adam；这些是后续更新，不能混作 SIGGRAPH 2023 原始 baseline。

## 9. 作为 Lyra 1.0/2.0、DA3 与 FantasyWorld 输出表示的基础

3DGS 的历史影响不只是一种更快的 per-scene NeRF 替代物，而是提供了一套**神经网络可以直接预测、GPU 可以立即渲染、图像损失可以端到端反传的显式 3D 输出协议**。但必须区分两种范式。

### 9.1 原始 3DGS：per-scene optimization

原论文没有一个把图像“一次前向”变成 Gaussian 集合的网络。每个场景单独运行 7K/30K iterations；Gaussian 的位置、尺度、旋转、opacity、SH 都是该场景的自由变量，集合大小还会通过 clone/split/prune 改变。SfM 点云负责初始化，rasterizer 负责把图像误差传回这些自由变量。

其优势是可以围绕单一场景反复拟合到高质量；代价是需要相机标定、逐场景优化和动态数据结构，不能直接作为大规模在线系统的低延迟预测器。

### 9.2 后续范式：feed-forward Gaussian prediction

后续工作把 $G=\{\mu,s,q,\alpha,c\}$ 从“待优化变量”改成网络输出：

$$
G=f_\theta(I_{1:N},K_{1:N},T_{1:N},\text{latent/geometry features}),
$$

随后仍调用 3DGS rasterizer 渲染并以 RGB/感知/深度损失训练。表示与渲染方程被保留，但原论文的 SfM 初始化、每场景 Adam、clone/split/prune 往往被取消或只作为可选后处理。此时 Gaussian 数量通常由输入像素/token 网格固定，模型训练成本被前置为跨场景学习，单场景推理则降为一次前向。

### 9.3 Lyra 1.0：视频扩散 latent → 3DGS decoder

[[Lyra 1.0 - 视频扩散自蒸馏三维重建|Lyra 1.0]] 让冻结视频扩散教师生成多轨迹 latent/RGB，学生 decoder 一次前向输出位置、尺度、四元数、opacity 和 RGB，再用 3DGS/gsplat 渲染监督。它继承的是 3DGS 的**参数协议与可微 splatting**，不是原始 per-scene densification。由此才可能把 726 个生成视图压成一个可实时浏览的静态 3DGS，并以目标时间条件扩展为动态 4DGS。

### 9.4 Lyra 2.0：生成长视频后用 DA3 式 Gaussian head 落成资产

[[Lyra 2.0 - 可探索的生成式三维世界|Lyra 2.0]] 的主贡献在长程视频生成与空间记忆；最终 3D 输出不再沿用 Lyra 1.0 自家 decoder，而是用经过生成视频域适配的 Depth Anything v3 前馈模型预测 per-pixel Gaussian 属性，再导出紧凑 3DGS/mesh。这里 3DGS 是**可探索世界的显式资产层**：视频模型负责想象与长期一致性，前馈重建器负责把结果投到可实时渲染表示。

### 9.5 DA3：几何基础模型 → GS-DPT head

[[DA3 - 任意视角恢复视觉空间|Depth Anything 3]] 先用 depth–ray 最小目标恢复几何和相机，再通过 pose-conditioned / pose-adaptive GS-DPT head 逐像素预测 opacity、quaternion、scale、color 等 Gaussian 属性。DA3 的关键变化是：Gaussian center 可由深度与 ray map 的几何关系锚定，其他属性由共享视觉骨干直接回归。因此 3DGS 从“场景优化器的内部状态”变成**几何 foundation model 的下游输出头**。

### 9.6 FantasyWorld：核心输出是隐式几何，3DGS 是下游评测/资产层

[[FantasyWorld - 统一视频与三维预测|FantasyWorld]] 冻结 Wan2.1 视频主干并增加 geometry branch，直接输出 depth、point map、uncertainty 和 camera pose；它的核心网络并不直接输出完整 Gaussian 参数。论文的 3DGS reconstruction 指标来自把生成视频与预测几何送入后处理重建流程。因此对 FantasyWorld 更准确的说法是：3DGS 提供了**统一的 NVS 评测和显式场景落地接口**，而不是它的原生生成 head。

### 9.7 这条技术链的共同点与断点

| 工作 | Gaussian 如何得到 | 是否 per-scene 优化 | 3DGS 扮演的角色 |
|---|---|---|---|
| 3DGS 2023 | SfM 初始化后直接优化自由变量 | **是，核心** | 场景表示 + 优化对象 + 实时 renderer |
| Lyra 1.0 | video latent 的学生 decoder 直接预测 | 否 | 生成式 3D/4D 的显式输出 |
| DA3 | 几何骨干 + GS-DPT head 逐像素预测 | 否 | feed-forward NVS 输出头 |
| Lyra 2.0 | 域适配 DA3 从生成长视频预测 | 否 | 长世界的视频→可浏览资产桥梁 |
| FantasyWorld | 深度/点图/位姿后接 3DGS 重建 | **有后处理优化** | 评测与资产化层，而非核心 head |

真正被继承的是三件事：可微的 Gaussian 参数化、投影后的 alpha compositing、GPU 友好的显式渲染。真正被后续工作替换的是：SfM 稀疏初始化、逐场景 30K Adam 和动态 clone/split/prune。理解这一区分，才能避免把所有“输出 Gaussians”的模型都误称为原始 3DGS 优化。

## 10. 复现清单、术语、图表索引与最终评价

### 10.1 最小复现路线

1. 用 COLMAP 得到相机内外参和 sparse points，确认相机模型/坐标约定。
2. 先只实现固定 Gaussian 集合：$\mu+s+q+\alpha+$零阶颜色，验证投影与 alpha compositing 梯度。
3. 加入 SH progressive schedule 和 L1 + D-SSIM；用小场景检查训练/测试 split。
4. 加入每 100 iter 的 gradient-statistics、clone/split/prune，以及每 3000 iter opacity reset。
5. 对照 Full / No-Split / Isotropic / Limited-BW，确认结果方向与论文一致，再追求主表数字。
6. 最后编译 SIBR viewer、测试 1080p FPS；不要用训练 FPS 或低分辨率 FPS 替代论文实时口径。

### 10.2 关键术语

- **Splatting**：把 3D primitive 投影成屏幕 footprint，再将其贡献散布到覆盖像素。
- **Anisotropic Gaussian**：三个主轴尺度可不同且可旋转的椭球 Gaussian。
- **SH**：以观察方向为输入的球谐颜色基函数，用于表达 view-dependent appearance。
- **Densification**：依据 view-space position gradient 动态增加 Gaussian 数量。
- **Clone / Split**：分别解决小 Gaussian 的欠覆盖与大 Gaussian 的过度覆盖。
- **Visibility-aware**：按近似深度顺序 compositing，而非 order-independent blending。
- **Tile rasterizer**：把屏幕分块，共享 Gaussian 数据与排序结果的 GPU 光栅化设计。
- **Per-scene optimization**：每个场景独立优化表示参数；与跨场景训练、一次前向预测相对。

### 10.3 图表索引

1. `fig1_teaser.png`：速度–质量对比，Figure 1 / Page 1。
2. `fig2_overview.png`：SfM 初始化、Gaussian 优化、投影、density control 与 rasterizer 总览，Figure 2 / Page 5。
3. `fig3_density_control.png`：under-/over-reconstruction 对应 clone/split，Figure 4 / Page 6。
4. `fig4_densification_ablation.png`：No-Split、No-Clone 与 Full 对比，Figure 8 / Page 10。
5. `fig5_anisotropy_ablation.png`：各向异性与球形 Gaussian 对比，Figure 10 / Page 11。

### 10.4 材料与缺失信息

官方 PDF、arXiv 源码包和图源记录保存在 `assets/3DGS - 实时辐射场渲染的三维高斯表示/sources/`，逐项 URL、Figure/Page 与获取日期见 `sources.md`。论文正文没有给出所有 pruning/size 阈值和每组 optimizer learning rate；严格数值复现还需要官方代码配置。官方仓库当前主分支包含论文发表后的功能，必须用 commit 与开关区分原始结果。

**最终评价**：3DGS 的突破不是单独发明 Gaussian、alpha blending 或 tile rasterization，而是把三者组织成一条可优化、可动态分配容量、又能直接进入 GPU 图形管线的完整闭环。它牺牲了隐式场的紧凑性和部分几何语义，换来此前无法同时获得的训练速度、视觉质量和实时渲染；更重要的是，它把“3D 输出”变成适合神经网络直接预测的参数化对象，为后来的 feed-forward reconstruction、生成式 3D/4D 与世界模型提供了共同的显式表示层。
