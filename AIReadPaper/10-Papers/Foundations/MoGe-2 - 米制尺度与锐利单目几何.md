---
type: paper-note
status: complete
domain: Foundations
series: MoGe
series_order: 2
series_hub: "[[MoGe系列 - Survey]]"
paper: "MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details"
year: 2025
venue: NeurIPS 2025
arxiv: "2507.02546"
doi: "10.48550/arXiv.2507.02546"
source: https://arxiv.org/abs/2507.02546
project: https://wangrc.site/MoGe2Page/
code: https://github.com/microsoft/MoGe
tags:
  - Foundations
  - monocular-geometry
  - metric-depth
  - pointmap
  - data-refinement
  - DINOv2
created: 2026-10-21
updated: 2026-10-21
---

# MoGe-2：米制尺度与锐利单目几何

## 基本信息

- **作者**：Ruicheng Wang、Sicheng Xu、Yue Dong、Yu Deng、Jianfeng Xiang、Zelong Lv、Guangzhong Sun、Xin Tong、Jiaolong Yang
- **发表**：NeurIPS 2025；arXiv v1，2025-07-03
- **输入输出**：单张 RGB → metric point map、metric depth、valid mask、相机内参/FOV；官方后续权重另提供 normal head
- **基础模型**：DINOv2 ViT-L + convolutional neck；继承 MoGe 的 affine-invariant point-map supervision
- **数据**：24 个数据集、约 8.9M 有标签帧；16 synthetic、3 LiDAR、5 SfM
- **资源**：[[assets/MoGe-2 - 米制尺度与锐利单目几何/sources|来源记录]]；[官方 PDF](assets/MoGe-2%20-%20米制尺度与锐利单目几何/sources/paper.pdf)

## 一句话结论

MoGe-2 看到 [[MoGe - 开放域单目几何与仿射不变点图]] 虽然把单目不可辨识的焦距—距离歧义从 point-map 监督中剔除了，却仍没有米制尺度；而直接把绝对尺度塞回稠密点图又会让尺度误差的大梯度破坏相对形状，同时 LiDAR/SfM 真实标签的错位、孔洞和重影会把薄结构训练得模糊。它的答案是把任务拆成“**affine-invariant dense geometry + CLS-token 全局 metric scale**”：几何分支继续学稳定的相对三维形状，独立两层 MLP 在 log-space 预测尺度并用 stop-gradient 隔离两者；数据侧再用 synthetic-only teacher 的锐利几何对真实深度做局部 ROE 过滤和 log-depth Poisson 补全。最终 7 个 metric benchmark 上 metric point Rel 达 **8.19%**（vs UniDepth V2 10.1%），700²/2500 tokens 仅 **39 ms**，并在真实 iBims-1/HAMMER 边界 F1 上领先；但 MoGe-1 在若干全局 relative 指标仍更强，极细毛发、巨大前后景尺度差和 OOD 绝对尺度依旧是边界。

---

## 1. 问题演化：MoGe-1 之后缺什么

### 1.1 Affine-invariant 几何解决了错误监督，却没有真实尺度

MoGe-1 输出只在统一尺度与光轴平移的等价类中有效：

$$\mathbf P\cong s\mathbf P+\mathbf t$$

这非常适合学习“形状是否正确”，因为单图无法唯一确定焦距、相机距离和对象绝对尺寸。但机器人、AR、测量和资产尺度估计需要米制结果。MoGe-2 的第一个问题因此是：**如何补回 metric scale，而不破坏 MoGe 已经学好的 relative geometry？**

### 1.2 真实标签准确但不锐利，合成标签锐利但存在域差

真实数据并不等于完美监督：

- LiDAR 与 RGB 不同步，运动车辆边缘出现错位和拖影；
- SfM/MVS 在金属、反射、薄结构处产生孔洞和 ghost surfaces；
- 稀疏深度投影的边界与 RGB 轮廓不一致。

合成数据恰好相反：深度边界准确、薄结构完整，但对真实图像的绝对几何泛化较弱。MoGe-2 的第二个问题是：**如何保留真实标签的尺度与场景分布，同时借用合成标签的锐利局部结构？**

![](assets/MoGe-2%20-%20米制尺度与锐利单目几何/figures/01-teaser.png)

> **图 1（论文 Figure 1/6 相关对比）**：MoGe-2 同时给出接近 GT 的米制距离和更完整的局部结构；MoGe 没有 metric scale，UniDepth V2 和 Depth Pro 在尺度或开放域几何上出现偏差。来源：<https://arxiv.org/abs/2507.02546>。

### 1.3 核心贡献

1. 将相对几何与米制尺度解耦，保护 MoGe 的 affine-invariant shape prior。
2. 用 DINOv2 CLS token 而非局部 convolutional feature 预测场景级尺度。
3. 以 synthetic-only geometry model 为老师，统一清理 LiDAR 与 SfM 真实标签。
4. 通过 local ROE filtering + log-depth Poisson completion 在真实标签上恢复薄结构和边界。
5. 在一个模型中兼顾 relative geometry、metric geometry、细节和高吞吐。

---

## 2. 输入、输出与总体架构

### 2.1 模型输出

给定 RGB 图像 $I$，模型输出：

- affine-invariant point map $\hat{\mathbf P}_{\mathrm{aff}}$；
- 全局 metric scale $\hat s$；
- valid mask $\hat M$；
- 从 point map 投影关系恢复的焦距/FOV。

最终 metric point map 为：

$$\hat{\mathbf P}_{\mathrm{metric}}=\hat s\,\hat{\mathbf P}_{\mathrm{aff}}$$

需要注意，论文的 metric point evaluation 仍允许一个最优全局 translation：它要求**尺度正确**，但相机坐标原点的整体平移可单独求解。

![](assets/MoGe-2%20-%20米制尺度与锐利单目几何/figures/02-method-pipeline.png)

> **图 2（Figure 2）**：DINOv2 image tokens 经共享卷积 neck 和 point head 输出 affine point map；CLS token 不进入 dense neck，而由独立 MLP 预测 metric scale；二者相乘得到米制点云，Z 通道给出 metric depth。来源：<https://arxiv.org/abs/2507.02546>。

### 2.2 DINOv2 与卷积解码器

主模型使用 DINOv2 ViT-L，从第 6、12、18 和最后层抽取 image tokens，投影到统一通道后求和，再送入逐级上采样的 convolutional neck。输入会按目标 token 数调整到 $14h\times14w$，因此可在测试时改变分辨率。

卷积 ResBlock 移除 normalization，以降低 GPU latency；不同 dense heads 共享 feature pyramid。UV positional encoding 映射到单位圆并保留宽高比，帮助 decoder 感知图像位置和相机几何。

---

## 3. 尺度解耦：为什么不能直接预测绝对点图

### 3.1 三种失败或次优设计

论文比较：

1. **SI-Log entangled**：在稠密几何中直接预测绝对尺度；
2. **Shift-invariant point map**：点图已有 metric scale，只对齐 translation；
3. **Decoupled Conv head**：尺度独立，但从局部卷积特征汇总；
4. **Decoupled CLS-MLP**：最终方案。

绝对尺度跨室内、街景和远景具有巨大动态范围。若尺度错误直接作用于每个像素，产生的大梯度会同时扭曲相对几何。卷积尺度头虽然解耦，但不擅长读取场景类型、对象常识和整体透视。

### 3.2 CLS-token scale head

尺度分支直接读取 DINOv2 CLS token：

$$\hat s=\exp(\operatorname{MLP}(\mathrm{CLS}))$$

两层 MLP 的隐藏维与 CLS 维度相同，指数映射保证 $\hat s>0$。

尺度监督由当前 point map 与 GT 的 ROE 最优对齐尺度 $s^*$ 动态生成：

$$\mathcal L_s=\left\|\log\hat s-\operatorname{stopgrad}(\log s^*)\right\|_2^2$$

`stopgrad` 是关键：scale head 学习追随几何分支当前所需的 metric conversion，但尺度损失不会反向推动 point-map branch 改变相对形状。

### 3.3 消融结果

| 尺度设计 | Metric point Rel↓ | Metric depth Rel↓ | Local point Rel↓ | Sharpness F1↑ |
|---|---:|---:|---:|---:|
| Entangled SI-Log | 10.0 | 17.9 | 8.21 | 10.7 |
| Entangled shift-invariant | **8.99** | 16.9 | 6.69 | 11.8 |
| Decoupled Conv head | 9.62 | 17.7 | 6.34 | **12.7** |
| **Decoupled CLS-MLP** | 9.20 | **16.5** | **6.26** | 12.5 |

Shift-invariant 的 metric point Rel 略低，但 CLS-MLP 在 metric depth 和全部 relative 指标上更稳，因此总体最优。这里的结论不是“某一列绝对第一”，而是**解耦设计给出了更好的多目标 Pareto**。

---

## 4. 数据精炼：用合成标签修真实标签

### 4.1 Synthetic-only teacher

先只用合成数据训练 $G_{\mathrm{syn}}$。它在真实域的绝对深度未必准，但边界、薄结构和 RGB-depth 对齐通常更干净。因此不能直接替换真实深度，只能把它作为局部结构参考。

### 4.2 Local ROE mismatch filtering

在 $G_{\mathrm{syn}}$ 预测几何中，以预测点为中心建立多尺度球邻域：

$$\hat{\mathcal S}_j=\{i\mid\|\hat{\mathbf p}_i-\hat{\mathbf p}_j\|\le\hat r_j,\ i\in\mathcal M\}$$

在每个邻域内，对真实采集点和 synthetic prediction 求局部 ROE scale/translation。若对齐后的偏差超过邻域半径，则判为 outlier：

$$\mathcal O_j=\left\{i\mid\|s_j^*\mathbf p_i+\mathbf t_j^*-\hat{\mathbf p}_i\|>\hat r_j\right\}$$

最终：

$$\mathcal M_{\mathrm{filtered}}=\mathcal M\setminus\bigcup_j\mathcal O_j$$

局部对齐而非全局对齐，是为了避免 synthetic teacher 的整体深度 bias 误删正确前景。

### 4.3 Log-depth Poisson completion

过滤后留下孔洞。补全目标不直接复制 teacher 深度，而是在缺失区域匹配 teacher 的 log-depth gradient，同时将区域边界钉在真实深度上：

$$\min\sum_{i\in\mathcal M_{\mathrm{filtered}}^c}\|\nabla\log d_i^c-\nabla\log\hat d_i\|^2,\quad\text{s.t. }d_i^c=d_i\text{ on boundary}$$

因此内部结构继承 synthetic 的锐利变化，绝对值则由真实数据锚定。

![](assets/MoGe-2%20-%20米制尺度与锐利单目几何/figures/03-data-refinement.png)

> **图 3（Figure 4）**：真实 LiDAR/SfM 深度在物体边界、金属和薄结构处有错位与空洞；synthetic model 提供局部形状，过滤清除错误值，Poisson completion 恢复连续且锐利的标签。来源：<https://arxiv.org/abs/2507.02546>。

### 4.4 数据消融的真实权衡

| 训练数据 | Metric point Rel↓ | Metric depth Rel↓ | Local point Rel↓ | F1↑ |
|---|---:|---:|---:|---:|
| Synthetic only | 12.4 | 21.7 | 6.42 | **13.3** |
| Raw real | **9.01** | **15.8** | 6.37 | 10.3 |
| **Refined real** | 9.20 | 16.5 | **6.26** | 12.5 |

Refined data 并没有在 metric average 上超过 raw real；它以极小的 metric 损失换回显著锐度，并改善 local/relative geometry。论文最强的证据是**折中被推向更优 Pareto**，而不是所有列同时最优。

---

## 5. 数据与训练

### 5.1 数据组成

24 个数据集约 8.9M 帧：

- 16 个 synthetic：Hypersim、TartanAir、Objaverse、Structured3D 等；
- 3 个 LiDAR：A2D2、Argoverse2、Waymo；
- 5 个 SfM/MVS：ARKitScenes、BlendedMVS、MegaDepth、ScanNet++、Taskonomy。

具备可信 metric scale 的数据才监督 scale head；无 metric 数据仍可监督 affine geometry。

### 5.2 训练配置

- DINOv2 ViT-L，约 326M 参数；
- 120K iterations；
- 32×A100，约 120 小时；
- backbone LR $10^{-5}$；neck/heads LR $10^{-4}$；
- 每 25K steps 学习率减半；
- 消融用 ViT-B，训练 100K steps；
- 训练覆盖不同宽高比与 token 数。

---

## 6. 实验结果

### 6.1 Relative geometry：不是全面击败 MoGe-1

10 个数据集平均：

| 方法 | SI point Rel↓ | Affine point Rel↓ | Local point Rel↓ | SI depth Rel↓ | Affine depth Rel↓ | 综合排名↓ |
|---|---:|---:|---:|---:|---:|---:|
| MoGe-1 | **7.46** | **5.69** | 5.50 | **5.77** | **4.51** | 2.53 |
| **MoGe-2** | 10.8 | 7.98 | **5.33** | 7.35 | 5.62 | **2.05** |

MoGe-1 的全局误差更低，MoGe-2 的 local point 最好、跨六类任务平均排名更优。准确表述应是：MoGe-2 在加入 metric capability 后基本保住 relative geometry，并改善局部结构；不能写成“所有相对指标超过 MoGe”。

### 6.2 Metric geometry

七个 metric 数据集平均：

| 方法 | Metric point Rel↓ / $\delta_1$↑ | Metric depth Rel↓ / $\delta_1$↑ | GT camera depth Rel↓ / $\delta_1$↑ |
|---|---:|---:|---:|
| UniDepth V2 | 10.1 / 91.9 | 21.3 / 75.3 | 18.5 / 82.6 |
| Depth Pro | 13.7 / 81.9 | 27.6 / 54.4 | — |
| **MoGe-2** | **8.19 / 93.6** | **15.7 / 76.8** | **13.6 / 87.4** |

GT camera 对 KITTI 帮助尤其明显，depth Rel 从 18.1 降到 8.64；HAMMER 即使有 GT camera 仍困难，显示问题不只是内参误差。

### 6.3 Boundary sharpness

| 方法 | iBims-1 | HAMMER | Sintel | Spring | 平均排名↓ |
|---|---:|---:|---:|---:|---:|
| Depth Pro | 14.3 | 5.36 | **41.6** | **11.0** | **1.50** |
| MoGe-1 | 11.4 | 3.89 | 26.3 | 8.36 | 4.67 |
| **MoGe-2** | **17.9** | **5.42** | 35.2 | 8.63 | **1.75** |

MoGe-2 在两个真实数据集第一，Depth Pro 在两个合成集第一。结合 metric/relative 准确性，MoGe-2 的综合折中更强。

### 6.4 速度与分辨率

单 A100 FP16：

| 配置 | 分辨率 / tokens | 延迟 | 显存 |
|---|---:|---:|---:|
| MoGe-1 | 700² / 2500 | 70 ms | 0.88 GB |
| **MoGe-2** | 700² / 2500 | **39 ms** | 1.1 GB |
| MoGe-2 | 840² / 3600 | 55 ms | 1.3 GB |
| MoGe-2 | 1188² / 7200 | 108 ms | 1.9 GB |
| Depth Pro | 1536² / 20160 | 139 ms | 3.7 GB |

测试时增加 token 可提升锐度；延迟大致随 token 线性增长。移除卷积 normalization 是 MoGe-2 快于 MoGe-1 的重要工程因素。

---

## 7. 局限与批判性评价

### 7.1 论文明确局限

1. hair、thin lines 等极细结构仍会消失；
2. 巨大前景/背景尺度差时，直线、平行和对齐关系可能弯曲；
3. 单图绝对尺度依赖训练先验，OOD 对象尺寸会出错；
4. 后续应引入更强网络和更多真实世界尺度先验。

### 7.2 个人判断

MoGe-2 的贡献有一半在模型，有一半在数据治理。它揭示：**深度标签越“真实”不一定越适合学习细节**；传感器误差会被神经网络平均成平滑几何。synthetic-guided refinement 的可迁移价值很高，可推广到 normal、flow、point map、LiDAR densification 等像素级监督。

同时，CLS scale head 预测的不是从图像严格可辨识的物理量，而是数据先验下的最可能尺度。metric SOTA 不意味着解决了单目尺度歧义；它只是把 relative geometry 与尺度猜测分开，使猜错尺度时不再连带毁掉形状。

---

## 8. 复现与实践建议

1. 只做下游推理优先使用官方 ViT-L-normal 权重；
2. 需要速度时可降低 token 数，细节任务则提高分辨率；
3. 对 OOD 物体尺寸必须引入已知相机、参照物或外部尺度；
4. 重训时先复现 scale-head 消融，再实施昂贵的数据 refinement；
5. 注意论文与后续仓库能力边界：normal 权重是官方系列工程扩展，但论文未提供独立 normal benchmark。

---

## 9. 系列位置

MoGe-1 解决“监督什么”：用 affine-invariant point map 去除不可辨识变量；MoGe-2 解决“如何补回尺度、如何把训练标签变锐利”；下一代 [[MoGe-3 - 自引导稀疏体积几何细化]] 则指出，即使标签变锐利，二维 decoder 的邻域仍会跨深度边界混特征，因此把 refinement 真正搬到 sparse 3D space。

---

## 10. 材料来源

- [arXiv](https://arxiv.org/abs/2507.02546)
- [项目页](https://wangrc.site/MoGe2Page/)
- [官方代码](https://github.com/microsoft/MoGe)
- [[assets/MoGe-2 - 米制尺度与锐利单目几何/sources|本地来源记录]]
