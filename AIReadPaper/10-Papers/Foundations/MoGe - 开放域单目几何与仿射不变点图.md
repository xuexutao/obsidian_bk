---
type: paper-note
status: complete
domain: Foundations
series: MoGe
series_order: 1
series_hub: "[[MoGe系列 - Survey]]"
paper: "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision"
year: 2024
venue: CVPR 2025 Oral
arxiv: "2410.19115"
source: https://arxiv.org/abs/2410.19115
project: https://wangrc.site/MoGePage/
code: https://github.com/microsoft/MoGe
tags:
  - monocular-geometry
  - pointmap
  - affine-invariant
  - camera-estimation
created: 2026-10-15
updated: 2026-10-21
---

# MoGe：开放域单目几何与仿射不变点图

## 基本信息

| 项目 | 内容 |
|---|---|
| 作者/机构 | Ruicheng Wang 等，USTC / Microsoft Research / Harvard / Tsinghua |
| 输入输出 | 单张 RGB → affine-invariant pointmap + infinity mask |
| 数据 | 21 个数据集，约 9M 帧 |
| 核心 | 仿射不变点图、ROE 最优对齐、多尺度局部几何监督、FOV 恢复 |
| 重要性 | ★★★★☆：把单目深度提升为直接单目 3D 几何，并把性能焦点从架构转到监督表示 |

## 一句话结论

MoGe 发现单目几何的主要瓶颈不只是网络容量，而是监督中混入了单图无法辨识的**焦距—距离、全局尺度和对象间相对距离歧义**；它因此直接预测逐像素 3D point map，并把目标定义为允许统一尺度与光轴平移的 affine-invariant 几何，再用 ROE 对齐器求训练时的鲁棒最优 scale/shift、用三尺度局部球监督与法线损失保住局部表面，同时从投影一致性反求焦距/FOV。当前 v3 在 8 个未见数据集上把 affine point-map 平均 Rel 从此前 UniDepth 的 9.43% 降到 **6.07%**、local point Rel 从 DUSt3R 的 7.97% 降到 **5.50%**，但输出仍无米制尺度、依赖中心主点针孔假设，超细结构与天空/透明区域仍可能失败；它为后续 [[MoGe-2 - 米制尺度与锐利单目几何]] 的尺度解耦和 [[MoGe-3 - 自引导稀疏体积几何细化]] 的三维 refinement 奠定了表示与监督基础。

---

## 1. 为什么从 depth 转向 pointmap

单目深度必须和相机内参结合才能恢复 3D；即使深度很准，焦距错误也会把点云横向拉伸。DUSt3R 已证明直接回归 pointmap 可绕过显式内参，但其目标主要为多视图且只做尺度不变。MoGe 面向单图开放域，直接预测可见表面的 3D pointmap，并从结果反求焦距、FOV 和 camera-space depth。

![](assets/MoGe%20-%20开放域单目几何与仿射不变点图/figures/01-teaser.png)

> 图 1：开放域图像的 disparity、可见表面几何与新视角展示，覆盖室内、室外、物体和非真实图像。来源：论文 Figure 1，第 1 页。

---

## 2. 仿射不变 Pointmap

模型输出 $P\in\mathbb R^{H\times W\times3}$，允许：

$$P\cong sP+t$$

这里“affine-invariant”特指统一尺度 $s$ 与全局平移 $t$，不是任意仿射矩阵。主点居中、方形像素假设下，平移简化为 $t=(0,0,t_z)$。

为什么要加入 shift？长焦远拍与广角近拍可形成相似图像。若只允许尺度对齐，同一形状会因相机距离不同得到冲突监督；加入光轴平移后可把二者对齐到一致形状目标。消融中最优仿射表示全面优于“depth + camera”和尺度不变 pointmap。

---

## 3. ROE 全局对齐

全局损失：

$$L_G=\sum_{i\in M}\frac1{z_i}\|s\hat p_i+t^*-p_i\|_1$$

其中 $(s^*,t^*)$ 必须是当前实际 L1 目标的最优解。过去用 median/分位数估 scale/shift 只是近似，会把对齐误差误当成模型误差。

ROE（Robust, Optimal, Efficient）利用一维 shift 情况下最优解至少有一个点的 Z 残差为零：

$$s\hat z_k+t_z-z_k=0\Rightarrow t_z=z_k-s\hat z_k$$

枚举候选 $k$ 后把二维优化降为关于 $s$ 的一维分段线性问题，并并行搜索不可微点。复杂度 $O(N^2\log N)$；训练时先把 pointmap 缩到 64²。为抑制错误边界与离群点，残差进一步截断：

$$\min(\tau,w_i|r_i|)$$

---

## 4. 多尺度局部几何监督

单图中对象之间的相对距离不可辨识，但对象内部形状仍应准确。MoGe 以 GT 点 $p_j$ 为锚定义球形邻域：

$$S_j=\{i\mid\|p_i-p_j\|\le r_j\},\qquad r_j=\alpha z_j\frac{\sqrt{W^2+H^2}}{2f}$$

对每个邻域独立用 ROE 对齐，再计算局部点误差。三个尺度：

$$\alpha\in\left\{\frac14,\frac1{16},\frac1{64}\right\}$$

再加 pointmap 邻域叉积得到的法线角度损失。这样全局层允许对象间距离不确定，局部层仍强制表面几何精细。

![](assets/MoGe%20-%20开放域单目几何与仿射不变点图/figures/02-method-overview.png)

> 图 3：ViT + 卷积 decoder 输出 affine pointmap 与 infinity mask；训练端使用全局、三尺度局部、法线和 mask 监督；推理端恢复 shift、焦距与深度。来源：论文 Figure 3，第 4 页。

---

## 5. Infinity Mask 与相机恢复

天空和纯背景没有有限 3D 坐标，强行给 1000× 大深度会污染前景。模型另预测 mask，0.5 阈值排除无定义区域。

对有效点，联合求焦距 $f$ 与归一化 Z-shift $t'_z$：

$$\min_{f,t'_z}\sum_i\left(\frac{fx_i}{z_i+t'_z}-u_i\right)^2+\left(\frac{fy_i}{z_i+t'_z}-v_i\right)^2$$

固定 $t'_z$ 时，$f$ 有闭式解：

$$f=\frac{\sum_i a_iu_i+\sum_i b_iv_i}{\sum_i a_i^2+\sum_i b_i^2},\quad a_i=\frac{x_i}{z_i+t'_z},\ b_i=\frac{y_i}{z_i+t'_z}$$

因此只需用 Levenberg–Marquardt 优化一个变量，64² pointmap 通常 10 次内收敛，约 3 ms。

---

## 6. 数据与训练

约 9M 帧、21 个来源，含 A2D2、Argoverse2、ARKitScenes、Taskonomy、Waymo、HyperSim、TartanAir、Objaverse 等。Synthetic 使用完整六项监督；真实数据按标签来源降低局部监督强度，并丢弃逐像素损失最高的 5%。

- encoder LR $5\times10^{-6}$；decoder LR $5\times10^{-5}$；
- batch 256；
- 每 100K iterations 学习率除以 5；
- 像素数 250K–500K，宽高比 1:2 到 2:1；
- DINOv2 ViT encoder + 轻量 CNN decoder。

---

## 7. 主要结果

八个未见数据集平均：

| 任务 | MoGe Rel↓ | $\delta_1$↑ | 前一最佳 Rel |
|---|---:|---:|---:|
| Scale-invariant pointmap | **7.91** | **93.3** | UniDepth 12.3 |
| Affine-invariant pointmap | **6.07** | **94.7** | UniDepth 9.43 |
| Local affine pointmap | **5.50** | **95.6** | DUSt3R 7.97 |
| Scale-invariant depth | **6.17** | 93.8 | Metric3D v2 6.33 |
| Affine-invariant depth | **4.72** | **95.8** | Marigold 8.41 |
| Affine-invariant disparity | **5.76** | **95.2** | DA V1 7.31 |

FOV 三数据集平均 Mean/Median 误差为 2.91°/2.21°，综合排名第一。

![](assets/MoGe%20-%20开放域单目几何与仿射不变点图/figures/03-qualitative-comparison.png)

> 图 7：不同方法的 disparity、pointmap 与三维形状。Depth Anything V2 的深度图看起来锐利，但错误/缺失内参仍会造成三维扭曲；MoGe 直接监督 3D 形状。来源：论文 Figure 7。

---

## 8. 消融

ViT-B 消融中：

- Depth+camera → optimal scale-invariant → optimal affine-invariant，Affine Point Rel 从 9.20 → 8.50 → 8.07；
- optimal affine alignment 明显优于 median alignment：8.07 vs 9.36；
- 去掉局部损失，Local Point Rel 7.07 → 7.51；
- 去掉残差 clip，多数全局指标略退化；
- 去法线损失，表面明显更噪；
- 不预测 mask、而给天空超大深度，会伤害前景。

它验证了论文主张：性能提升来自**监督表示和最优对齐**，不是更复杂 backbone。

---

## 9. 局限与评价

- 输出没有绝对米制尺度，只在 scale+shift 等价类中有效；
- 相机恢复假设针孔、主点居中、方形像素、$f_x=f_y$；
- 单图不能恢复遮挡背面和图外区域；
- ROE 仍为 $O(N^2\log N)$，完整 3D shift 需要近似；
- 天空 mask 依赖伪标签，对透明/反射和超细结构仍可能失败；
- 论文明确展示栅栏薄结构被压平的 failure case。

**个人判断**：MoGe 代表一个重要分叉：与 Depth Anything 把深度做得更稳相比，它直接优化“反投影后的三维形状是否对”。这更接近 3D 生成、编辑、相机恢复和 VLA 数据预处理的实际需要，也是单目深度线与 DUSt3R/DA3 pointmap 线的连接点。

---

## 10. 复现建议

若只做推理直接使用官方模型。训练复现应先验证 optimal affine alignment 与 median alignment 的差距，再加入局部球监督；ROE 是主要实现难点。用于下游时要明确输出不含绝对尺度，并对非针孔相机单独标定。

---

## 11. 材料来源

见 [sources.md](assets/MoGe%20-%20开放域单目几何与仿射不变点图/sources.md)。
