---
type: survey
status: complete
domain: Foundations
topic: MoGe Series
year: 2026
papers:
  - "[[MoGe - 开放域单目几何与仿射不变点图]]"
  - "[[MoGe-2 - 米制尺度与锐利单目几何]]"
  - "[[MoGe-3 - 自引导稀疏体积几何细化]]"
tags:
  - Foundations
  - monocular-geometry
  - pointmap
  - metric-depth
  - sparse-3D
created: 2026-10-21
updated: 2026-10-21
---

# MoGe 系列：从仿射不变点图到稀疏三维体积细化

## 基本信息

| 版本 | 论文 | 时间与出处 | 核心问题 |
|---|---|---|---|
| MoGe-1 | *MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision* | 2024-10 arXiv；CVPR 2025 Oral | 单图有哪些不可辨识变量，应该监督什么几何？ |
| MoGe-2 | *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details* | 2025-07 arXiv；NeurIPS 2025 | 如何补回米制尺度，同时避免真实标签把细节训练模糊？ |
| MoGe-3 | *MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement* | 2026-07 arXiv v2 | 即使监督锐利，二维 decoder 为何仍扭曲三维细结构？ |

## 一句话结论

MoGe 三代不是简单扩大 backbone，而是连续修正单目几何链路中的三个不同瓶颈：MoGe-1 发现 focal-distance、全局尺度和平移等不可辨识量会污染监督，于是用 affine-invariant point map、ROE 最优对齐和多尺度局部几何监督先把“**学什么**”定义正确；MoGe-2 再把“相对形状”与“全局米制尺度”解耦，用 CLS-MLP 预测 scale，并以 synthetic teacher 对真实 LiDAR/SfM 标签做局部过滤与 Poisson 补全，解决“**尺度怎么补、数据怎么变锐利**”；MoGe-3 最后指出二维 decoder 的邻域本身与三维表面不一致，构建由当前点图自引导的 sparse $(i,j,D\log Z)$ voxel shell，在真实 3D 邻域迭代修正 log-depth，解决“**几何在哪里计算**”。系列能力由 affine relative geometry 推进到 metric scale，再推进到 fine-detail 3D fidelity；代价也依次从无米制尺度，转为依赖尺度先验，再转为更高推理成本与边界 fly-points。

---

## 1. 系列要解决的根本问题

### 1.1 单目图像到底能恢复什么

单张图像到 3D 不是一个唯一逆问题。至少存在：

1. **全局尺度歧义**：缩小物体并靠近相机，可形成相似投影；
2. **焦距—相机距离歧义**：长焦远拍与广角近拍可能相似；
3. **对象间相对距离歧义**：局部形状明确，不代表对象前后关系唯一；
4. **遮挡与图外不可见性**：单图只能恢复可见表面；
5. **边界归属歧义**：混合像素可能属于前景或背景；
6. **真实尺寸先验依赖**：米制尺度需要对象常识或外部参照物。

MoGe 系列的路线不是假装这些歧义不存在，而是把它们分层处理：不可观测的变量从 dense geometry supervision 中剔除，可预测但不稳定的全局 scale 用独立 head 学，局部结构则用三维邻域增强。

### 1.2 三个瓶颈位于不同层

| 层次 | 瓶颈 | 对应版本 |
|---|---|---|
| 目标与监督 | 错把不可辨识 scale/shift/focal 当成唯一 GT | MoGe-1 |
| 数据与全局尺度 | 真实标签边界脏；尺度误差干扰 shape | MoGe-2 |
| 网络邻域 | 3D 输出仍由 2D proximity 决定交互 | MoGe-3 |

后代出现不是因为前代失效，而是前代解决一层后，下一层成为主瓶颈。

---

## 2. 第一次转移：从 depth prediction 到 affine point-map supervision

### 2.1 MoGe-1 的反题

传统单目 depth 模型输出 $D(u,v)$，只有结合内参 $K$ 才能得到：

$$\mathbf p=D(u,v)K^{-1}[u,v,1]^\top$$

深度图看起来正确，不代表 3D point cloud 正确；焦距错误会把形状横向拉伸。DUSt3R 已展示 pointmap 的价值，但 scale-only invariant 对单图仍不够。

### 2.2 Affine-invariant point map

MoGe-1 直接输出 $\mathbf P\in\mathbb R^{H\times W\times3}$，并只要求它在统一尺度和光轴平移后与 GT 一致：

$$\mathbf P\cong s\mathbf P+(0,0,t_z)$$

这一步把网络不可能从单图稳定决定的 global scale 和 focal-distance ambiguity 从 dense target 中拿走。

### 2.3 ROE：监督前先精确对齐

训练损失：

$$\mathcal L_G=\sum_i\frac1{z_i}\|s^*\hat{\mathbf p}_i+\mathbf t^*-\mathbf p_i\|_1$$

关键不是公式里的 L1，而是 $s^*,\mathbf t^*$ 必须对当前目标最优。ROE 利用最优解至少令一个 Z 残差为零，将二维 scale/shift 优化拆成候选锚点下的一维分段线性问题，以 $O(N^2\log N)$ 求解，并对残差截断以抵抗边界异常点。

### 2.4 局部几何监督

全局对齐允许对象间距离不确定，局部球形区域再独立 scale/translation 对齐：

$$\mathcal S_j=\{i\mid\|\mathbf p_i-\mathbf p_j\|\le r_j\},\quad\alpha\in\{1/4,1/16,1/64\}$$

于是全局 loss 学场景形状，local loss 学对象内部曲面，normal loss 抑制噪声，infinity mask 排除天空。

![](../../10-Papers/Foundations/assets/MoGe%20-%20开放域单目几何与仿射不变点图/figures/02-method-overview.png)

> **MoGe-1 代表图（Figure 3）**：DINOv2+CNN 输出 affine point map 与 mask，训练端同时使用全局、三尺度局部、法线和 mask 监督。来源：<https://arxiv.org/abs/2410.19115>。

### 2.5 结果与新瓶颈

当前 v3 的 8 数据集平均：

- SI point Rel：7.91%；
- affine point Rel：6.07%；
- local point Rel：5.50%；
- affine depth Rel：4.72%；
- FOV mean error：2.91°。

这证明监督表示比堆架构更重要，但留下最明显问题：**输出没有米制尺度。**

---

## 3. 第二次转移：从相对形状到米制尺度与锐利标签

### 3.1 为什么不能把 scale 直接塞回 point map

若令 dense point map 直接承担 metric scale，场景尺度从桌面到街景跨越巨大动态范围。尺度猜错造成的每像素大梯度会反过来伤害相对几何。MoGe-2 因此保留 affine point branch，并单独预测：

$$\hat{\mathbf P}_{metric}=\hat s\hat{\mathbf P}_{affine}$$

### 3.2 CLS-MLP scale

$$\hat s=\exp(\operatorname{MLP}(\mathrm{CLS}))$$

$$\mathcal L_s=\|\log\hat s-\operatorname{stopgrad}(\log s^*)\|_2^2$$

CLS token 聚合场景类型、对象尺寸和整体透视，适合做全局尺度；stop-gradient 防止 scale supervision 扭曲 dense shape。

### 3.3 真实数据与合成数据的矛盾

- Synthetic：边界完美，metric 泛化弱；
- Raw real：尺度准确，边界错位和孔洞严重；
- Refined real：用 synthetic teacher 比较局部结构，保留真实标签中可靠部分，并填补错误区域。

局部 ROE filtering 消除 teacher 的绝对深度 bias，log-depth Poisson completion 用 synthetic gradient 补细节、真实边界值守尺度。

![](../../10-Papers/Foundations/assets/MoGe-2%20-%20米制尺度与锐利单目几何/figures/03-data-refinement.png)

> **MoGe-2 代表图（Figure 4）**：synthetic teacher 指出 LiDAR/SfM 的错位和缺失；过滤删除错误深度，Poisson completion 在保留真实尺度的同时恢复薄结构。来源：<https://arxiv.org/abs/2507.02546>。

### 3.4 结果与反直觉事实

MoGe-2 在 metric benchmark 上明显领先：metric point Rel 8.19%，无 GT camera metric depth Rel 15.7%。但 MoGe-1 在全局 relative Rel 上仍更好：SI point 7.46 vs 10.8，affine point 5.69 vs 7.98。MoGe-2 领先的是 local geometry 和综合排名。

这说明“加 metric capability 不损失 relative geometry”应理解为整体能力没有崩，并非每项都反超前代。

### 3.5 新瓶颈暴露

即使训练标签更锐利，卷积 decoder 仍按图像邻近关系聚合。监督变好了，但模型结构仍会跨前景/背景混特征。于是瓶颈从 data quality 转移到 **parameterization mismatch**。

---

## 4. 第三次转移：从二维锐边到真实三维细节

### 4.1 二维锐利不等于三维正确

一个深度边缘可以在图像中很锐利，但栏杆各点深度如果略有系统偏差，换视角后仍会弯曲、分叉或与背景混合。Boundary F1 因此不能代替 point-map fidelity。

### 4.2 Self-Guided Sparse 3D Refinement

MoGe-3 将点因子化：

$$\mathbf q=(u=X/Z,v=Y/Z,\zeta=\log Z)$$

由当前预测构建 sparse voxel coordinate：

$$\mathbf c_{ij}^{(k)}=(i,j,\lfloor D\zeta_{ij}^{(k)}\rceil)$$

Sparse 3D U-Net 只在 occupied shell 上运行，预测 $\Delta\zeta$ 并迭代更新。图像相邻但深度相远的前景/背景不会成为邻居。

![](../../10-Papers/Foundations/assets/MoGe-3%20-%20自引导稀疏体积几何细化/figures/02-method-pipeline.png)

> **MoGe-3 代表图（Figure 2）**：base 先输出 coarse point map，SSR 根据当前几何体素化、用 sparse 3D U-Net 与 DINO feature 修正 log-depth，重复多轮。来源：<https://arxiv.org/abs/2607.17967>。

### 4.3 为什么是稀疏壳而非完整体积

完整 $H\times W\times D$ volume 大量为空。壳层最高分辨率只有 $HW$ occupied voxels，计算随可见点数量增长。它既保留 per-pixel correspondence，又让邻域依赖 log-depth。

### 4.4 结果与代价

MoGe-3 ViT-L 相对 MoGe-2：

- local point Rel：3.19→2.79；
- local point $\delta_{0.01}$：46.6→55.9；
- global point Rel：8.73→7.91；
- metric depth $\delta$：77.3→82.7；
- latency：39 ms→121 ms。

Boundary F1 只从 15.6→16.0，仍低于 InfiniDepth 19.3。这再次说明其主要收益是三维结构而非像素边缘。

---

## 5. 三代方法横向对比

| 维度 | MoGe-1 | MoGe-2 | MoGe-3 |
|---|---|---|---|
| 核心问题 | 如何移除不可辨识监督 | 如何补尺度并修复脏标签 | 如何避免 2D 邻域扭曲 3D 细节 |
| 输出尺度 | affine-invariant | metric | metric |
| 主要表示 | point map | affine point + global scale | factorized point + sparse shell |
| 关键机制 | ROE + local spheres | CLS scale + data refinement | iterative sparse 3D U-Net |
| 视觉 backbone | DINOv2 | DINOv2 | DINOv2 |
| 几何 decoder | 2D CNN | 2D Conv neck/heads | 2D base + 3D refiner |
| 数据治理 | 21 datasets，按标签质量选择 losses | synthetic teacher 清理 real labels | SSR 只用 synthetic 精确梯度，base 用混合数据 |
| 主要强项 | 全局 relative geometry、FOV | metric scale、真实边界、速度 | strict fine-detail 3D geometry |
| 主要局限 | 无米制尺度 | OOD scale、极细结构 | 3× latency、fly-points、边缘多解 |

---

## 6. 技术时间线与问题转移

| 时间 | 工作 | 解决了什么 | 新瓶颈 |
|---|---|---|---|
| 2024-10 | MoGe v1 | affine point target、ROE、local geometry | 没有 metric scale |
| 2025-04 | MoGe v3 / CVPR 2025 Oral | 完成当前评测与论文版本 | 绝对尺度和真实标签细节 |
| 2025-07 | MoGe-2 arXiv | scale decoupling、real-data refinement | 2D decoder 跨深度混合 |
| 2025 | MoGe-2，NeurIPS 2025 | metric/detail 系统验证 | 三维局部结构仍平滑 |
| 2026-07 | MoGe-3 | sparse volumetric refinement、fine-detail benchmark | 边界多解、fly-points、速度 |

作者重叠非常高：Ruicheng Wang、Sicheng Xu、Jianfeng Xiang、Jiaolong Yang 贯穿三代，说明它是一条明确的连续研究主线，而非名字相似的独立工作。

---

## 7. 评测体系如何变化

### 7.1 第一阶段：全局 relative geometry

MoGe-1 强调 SI/affine/local point map、depth、disparity 与 FOV，核心是证明 point map 和监督表示正确。

### 7.2 第二阶段：metric + sharpness

MoGe-2 增加：

- metric point map；
- metric depth（有/无 GT camera）；
- boundary F1。

评测开始同时考察 shape、scale 和 edge。

### 7.3 第三阶段：严格三维 fine-detail

MoGe-3 指出 Boundary F1 是 2D 指标，增加 local segment 内的 point/depth Rel 与 $\delta_{0.01}$。评测从“边界是否锐利”推进到“细结构在三维中是否正确”。

这是领域成熟的重要标志：指标逐渐针对方法真正声称解决的失败模式，而不再只看全图平均 AbsRel。

---

## 8. 与 Depth Anything / DA3 路线的关系

### 8.1 共同点

- 都以 DINOv2 为基础视觉骨干；
- 都追求开放域 zero-shot geometry；
- 都从纯 depth 推向相机/point map/3D；
- 都高度重视数据与监督目标。

### 8.2 差异

| 路线 | 核心立场 |
|---|---|
| Depth Anything | 用超大无标签真实数据与 teacher pseudo labels扩泛化 |
| DA3 | depth-ray 是统一单图/多视图几何的最小充分目标 |
| MoGe | 单图几何需先剔除不可辨识变量，再逐步恢复尺度和 3D 细节 |

DA3 面向任意数量视图，MoGe 聚焦单图。DA3 用 ray map 内生相机；MoGe 用 affine point map 后处理恢复相机，再独立学习 global metric scale。MoGe-3 的 sparse 3D refinement 也可视为对“仅靠 2D DPT head 是否足够”的反问。

---

## 9. 当前正在攻坚的问题

### 9.1 边界多解与生成式几何

MoGe-3 明确承认回归范式不能表达一个像素属于前景/背景的多模态分布。未来结合 Pixel-Perfect Depth 式生成模型，是减少 fly-points 和获得更锐利边缘的直接方向。

### 9.2 真正可解释的 metric scale

MoGe-2 的 scale head 依赖图像语义和训练分布。OOD 尺度仍可能错误。需要：

- 已知相机或 EXIF；
- 参照物；
- 地面/重力；
- 物体尺寸知识；
- 多帧运动；
- 不确定性输出。

### 9.3 非针孔和复杂相机

MoGe-1/2 的 camera recovery 假设中心主点、方形像素和针孔模型。鱼眼、全景、rolling shutter 和未知畸变仍未统一。

### 9.4 遮挡后和图外几何

三代都主要恢复可见 surface point map，不做完整 scene completion。对 3D 资产、机器人和世界模型，仍需把 visible geometry 与 generative completion 区分。

### 9.5 速度—细节 Pareto

MoGe-2 39 ms 很适合在线模块；MoGe-3 121 ms 更适合高质量处理。未来需要一步式蒸馏、可变 $K$、局部触发 SSR 或更高效 sparse kernel。

---

## 10. 工程选择建议

| 需求 | 推荐 |
|---|---|
| 快速开放域 point/depth，相对几何优先 | MoGe-1 或 MoGe-2 低 token |
| 必须米制尺度 | MoGe-2 |
| 还需要 normal 和官方统一接口 | MoGe-2-normal / MoGe-3 |
| 栏杆、细杆、小物体、资产前处理 | MoGe-3 |
| macOS 本机运行 | MoGe-1/2；MoGe-3 的 Triton/FlexGEMM 当前不适合 |
| 机器人安全距离 | 不应只信单目 scale；加入传感器或参照物 |
| 多视图统一几何 | 更适合 DA3/VGGT，而非强行用单图 MoGe |

---

## 11. 已确立事实与我的判断

### 11.1 已确立事实

1. Affine-invariant point supervision 显著优于不匹配的 depth/camera 或 scale-only target。
2. Metric scale 与 dense relative geometry 解耦，整体 Pareto 优于完全纠缠。
3. Synthetic-only 最锐、raw-real 最准；refined-real 在二者之间取得更好平衡。
4. 参数匹配 2D refiner 无法复制 sparse 3D refinement 的 local geometry 增益。
5. MoGe-3 的局部三维指标提升明显，但二维 Boundary F1 并非第一。

### 11.2 我的判断

MoGe 系列最重要的研究方法是“**每一代只修一个层次的错误归因**”：第一代不把不可观测变量怪给网络，第二代不把尺度错误怪给 dense geometry、也不把传感器噪声当真值，第三代不把二维锐边当成三维正确。这种逐层归因比单纯扩大模型更可复用。

对 3D 生成和世界模型而言，MoGe 的价值不只是提供 depth：它提供 point map、camera/FOV、normal 和 metric scale 的统一可见表面接口。MoGe-3 进一步提示，任何消费单目几何的下游系统都应在三维邻域评估细结构，不能只看彩色深度图。

---

## 12. 文献索引

### 12.1 系列主论文

- [[MoGe - 开放域单目几何与仿射不变点图]]
- [[MoGe-2 - 米制尺度与锐利单目几何]]
- [[MoGe-3 - 自引导稀疏体积几何细化]]

### 12.2 关键前置与对照

- [[DINOv2 - 无监督鲁棒视觉特征]]：三代共同 backbone。
- [[Depth Pro - 高分辨率锐利度量深度]]：MoGe-2/3 的 sharp metric baseline。
- [[UniDepth - 自提示相机的通用度量深度]]：metric point/depth 直接对照。
- [[DA3 - 任意视角恢复视觉空间]]：depth-ray 多视图几何路线。
- [[DUSt3R - 前馈式几何重建奠基]]：pointmap 前馈几何范式。
- [[从 Depth Anything 到 DA3：前馈深度与三维几何 - Survey]]：更大领域脉络。

---

## 13. 最终评价

**系列评级：★★★★★**

MoGe 系列的完整性来自三代之间清楚的因果链：MoGe-1 先建立正确 target，MoGe-2 再补 metric scale 和数据细节，MoGe-3 最后修网络的 3D inductive bias。它已经从一篇 point-map 监督论文发展成可输出 metric point/depth/normal/camera 的开放域单目几何工具链。

它仍不是完整 3D 世界模型：没有遮挡后结构，不保证 OOD 米制尺度，不处理所有相机模型，边界处仍有多解。但作为 3D 生成、重建、VLA 和世界模型的可见几何基础模块，MoGe 是当前最值得保留完整谱系的一条路线。
