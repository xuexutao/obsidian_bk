---
type: paper-note
status: complete
domain: Foundations
series: MoGe
series_order: 3
series_hub: "[[MoGe系列 - Survey]]"
paper: "MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement"
year: 2026
arxiv: "2607.17967"
doi: "10.48550/arXiv.2607.17967"
source: https://arxiv.org/abs/2607.17967
project: https://qft-333.github.io/moge3page/
code: https://github.com/microsoft/MoGe
tags:
  - Foundations
  - monocular-geometry
  - sparse-convolution
  - pointmap
  - 3D-refinement
  - fine-detail
created: 2026-10-21
updated: 2026-10-21
---

# MoGe-3：自引导稀疏体积几何细化

## 基本信息

- **作者**：Lingyu Kong、Ruicheng Li、Ruicheng Wang、Sicheng Xu、Chengtang Yao、Jianfeng Xiang、Jiaolong Yang
- **版本**：arXiv v2，2026-07-21；截至 2026-10-21 未标会议录用
- **输入输出**：单张 RGB → metric point map、metric depth、surface normal、valid mask、camera/FOV
- **核心模块**：MoGe-2 base + Self-Guided Sparse 3D Refinement（SSR）
- **默认配置**：DINOv2 ViT-L，log-depth voxel resolution $D=200$，3 次 refinement
- **资源**：[[assets/MoGe-3 - 自引导稀疏体积几何细化/sources|来源记录]]；[官方 PDF](assets/MoGe-3%20-%20自引导稀疏体积几何细化/sources/paper.pdf)

## 一句话结论

MoGe-3 发现 [[MoGe-2 - 米制尺度与锐利单目几何]] 即使通过数据精炼获得锐利标签，最终几何仍由 2D decoder 按图像邻域混合特征：遮挡边缘两侧的像素在图上相邻、在三维中却相隔很远，因而细杆、栏杆和小物体会被跨表面平滑。它保留 MoGe-2 作为 coarse foundation model，再把当前点图因子化为 $(u=X/Z,v=Y/Z,\zeta=\log Z)$，由预测自身构建 $(i,j,D\zeta)$ 稀疏体素壳，用 sparse 3D U-Net 在真实三维邻域中预测 log-depth residual，并迭代重建邻接关系；DINO 特征在 bottleneck 提供语义，全局 metric scale 继续由 base 预测。ViT-L 把严格 local point $\delta_{0.01}$ 从 **46.6 提到 55.9**、全局 point Rel 从 **8.73% 降到 7.91%**，三步推理 121 ms；但 Boundary F1 仍低于 InfiniDepth，回归式边界多解会产生 fly-points，SSR 还把 MoGe-2 的 39 ms 延迟提高约 3.1×。

---

## 1. 研究问题：几何输出为何仍被二维参数化束缚

### 1.1 2D parameterization mismatch

当前单目几何模型无论输出 depth 还是 point map，通常仍用：

- 2D convolutional decoder；
- patch/UV query；
- image-space diffusion/VAE decoder；
- 多尺度 2D features。

这些模块的邻域由像素坐标决定。前景栏杆和背景墙只差一个像素时，二维卷积会直接混合它们；同一三维表面的投影若在图像中不相邻，又无法直接交互。这是“预测目标是 3D、信息传播却是 2D”的结构错配。

结果是：

- depth discontinuity 处 feature bleeding；
- 细长结构被抹平或扭曲；
- disparity 图看似锐利，换视角看 point cloud 却错误；
- 增加 2D refiner 参数也无法解决邻域定义错误。

![](assets/MoGe-3%20-%20自引导稀疏体积几何细化/figures/01-teaser.png)

> **图 1（Figure 1）**：MoGe-3 在桌椅细腿、铁门栏杆和轨道设施上保持更完整的三维结构；MoGe-2、DA3 和 InfiniDepth 即使二维深度边界清晰，点云仍出现弯折、涂抹或缺失。来源：<https://arxiv.org/abs/2607.17967>。

### 1.2 核心贡献

1. 提出 Self-Guided Sparse 3D Refinement，让当前几何预测自己定义下一轮三维邻域。
2. 用 projection-aligned sparse voxel shell 避免填满完整 3D 体积，最高层只含 $HW$ 个 occupied voxels。
3. 固定射线方向 $(u,v)$、只修正 log-depth，稳定处理跨尺度场景。
4. 将 DINOv2 高层语义注入 sparse U-Net bottleneck，结合 2D 视觉先验与 3D locality。
5. 建立严格 fine-detail benchmark，以 $\delta_{0.01}$ 和自动细节区域衡量真实三维局部结构，而不只看二维边界。

---

## 2. 总体架构

![](assets/MoGe-3%20-%20自引导稀疏体积几何细化/figures/02-method-pipeline.png)

> **图 2（Figure 2）**：左侧 MoGe-2-style base 输出 coarse point map 与 scale/normal/mask；右侧 SSR 按当前几何体素化，Sparse 3D U-Net 融合 DINO 2D feature，预测 log-depth residual，重复 $K$ 轮。下方展示动态稀疏 U-Net 的下采样、skip 与上采样。来源：<https://arxiv.org/abs/2607.17967>。

### 2.1 Base model

Base 初始化自 MoGe-2，DINOv2 encoder 与 dense heads 输出：

- 初始点图 $\mathbf P^{(0)}$；
- dense feature map $\mathbf F$；
- valid mask；
- surface normal；
- global metric scale。

点图继续采用沿光轴的 affine-invariant supervision：

$$(X,Y,Z)\mapsto(sX,sY,sZ+t)$$

SSR 只负责修正相对几何，最终 metric scale 沿用 base 的全局尺度预测。

### 2.2 为什么不从 3D volume 直接生成

完整体积在高分辨率下大部分为空，成本按体积增长。MoGe-3 已有一个相当准确的 coarse surface，因此只需在当前表面附近建立一层**稀疏壳**，把 3D refinement 变成 occupied-surface computation。

---

## 3. 点图因子化：固定射线，只改深度

对每像素点 $(X,Y,Z)$ 定义：

$$\mathbf q_{ij}=(u_{ij},v_{ij},\zeta_{ij})=\left(\frac XZ,\frac YZ,\log Z\right)$$

- $(u,v)$ 表示投影射线方向；
- $\zeta$ 是 log-depth。

细化时固定 $(u,v)$，只更新：

$$\zeta_{ij}^{(k+1)}=\zeta_{ij}^{(k)}+\Delta\zeta_{ij}^{(k)}$$

恢复欧氏点：

$$\mathbf p_{ij}^{(K)}=e^{\zeta_{ij}^{(K)}}(u_{ij},v_{ij},1)$$

这种设计把 residual 限制为沿相机射线移动，避免三维残差任意改变投影对应；log-depth 则把乘性尺度变成加性平移，适合开放域尺度变化。

---

## 4. Self-Guided Sparse Voxel Shell

### 4.1 体素坐标

第 $k$ 轮将每个像素映射到：

$$\mathbf c_{ij}^{(k)}=\left(i,j,\left\lfloor D\zeta_{ij}^{(k)}\right\rceil\right)$$

默认：

$$D=200$$

输入层每个像素对应一个 occupied voxel，因此体素数为 $HW$，而不是 $HWD$。坐标不是欧氏 $(X,Y,Z)$ 规则网格，而是 image-aligned $(i,j,D\log Z)$。

### 4.2 三维邻域为什么更合理

图像上相邻的两个像素只有当 log-depth 也接近时，才会落在 sparse convolution 邻域。前景/背景跨深度边界会在第三轴上分离，从而避免 feature bleeding。

这里的“3D”需要准确理解：它不是世界坐标等距体素，而是**投影对齐的图像坐标 + log-depth**。优势是每像素天然对应、尺度稳定、稀疏高效；代价是邻域仍受相机参数化影响，不具备旋转/视角不变性。

### 4.3 Self-Guided 的含义

当前点图决定体素坐标，体素邻接决定 U-Net 信息传播，残差更新点图后下一轮重新体素化。随着 geometry 改善，邻接关系也改善，形成自举式正反馈。

---

## 5. Sparse 3D U-Net

### 5.1 动态稀疏层级

第 $l$ 层体素集合：

$$\mathcal V^{(k),l}=\left\{\left\lfloor\frac{\mathbf c_{ij}}{2^l}\right\rfloor\right\}$$

下采样后的体素数和分布由场景几何动态决定。即使图像坐标在低分辨率合并，前景和背景仍可能因 depth coordinate 不同而保持分离。

### 5.2 输入与视觉特征

最高分辨率 voxel feature 是 $(u,v,\zeta)$。在 bottleneck，DINOv2 dense feature $\mathbf F$ 按图像坐标注入；同一图像位置若对应多个 depth voxels，则复制视觉 feature，再由 3D sparse convolution 区分其空间关系。

这体现了分工：

- DINO feature：语义、材质与全局上下文；
- sparse shell：几何邻域；
- U-Net：根据两者预测 $\Delta\zeta$。

### 5.3 Zero initialization

最后一层零初始化，使训练开始时：

$$\Delta\zeta\approx0$$

SSR 初始为 identity mapping，不会破坏已有 coarse geometry。

### 5.4 2D refiner 为什么失败

参数匹配的 2D U-Net 也预测相同 residual，但局部深度 Rel 从 3.81 变差到 4.09，Boundary F1 从 15.6 降到 14.8。它证明收益不是“多一个网络”，而是**信息交互的邻域从 2D 改为 3D**。

---

## 6. 损失与训练

### 6.1 多阶段监督

每阶段：

$$\mathcal L_{\mathrm{geo}}^{(k)}=\lambda_g\mathcal L_{\mathrm{global}}^{(k)}+\lambda_\ell\mathcal L_{\mathrm{local}}^{(k)}+\lambda_e\mathcal L_{\mathrm{edge}}^{(k)}$$

总损失：

$$\mathcal L=\sum_{k=0}^K\mathcal L_{\mathrm{geo}}^{(k)}+\lambda_m\mathcal L_{\mathrm{mask}}+\lambda_n\mathcal L_{\mathrm{nml}}+\lambda_s\mathcal L_{\mathrm{scale}}$$

权重：

$$(\lambda_g,\lambda_\ell,\lambda_e,\lambda_m,\lambda_n,\lambda_s)=(1,1,1,0.1,0.1,0.1)$$

### 6.2 新的 radial-partition local loss

MoGe-1/2 用球形局部区域；MoGe-3 为高效多阶段训练改为多尺度 3D radial partition：

$$\pi_\alpha(i,j)=\left\lfloor\alpha\left(\frac{\mathbf r_{ij}}{\|\mathbf r_{ij}\|_\infty},\log\|\mathbf r_{ij}\|_2\right)\right\rceil,\quad\alpha\in\{4,16,64\}$$

分区参考以 0.5 概率取 GT 或当前全局对齐预测，避免分区只适配一侧；每个分区共享全局 scale，只拟合独立 3D translation，防止局部 scale 退化。

### 6.3 Edge-angle loss

通过预测/GT 在水平、垂直方向的 3D 差分夹角约束边缘：

$$\mathcal L_{\mathrm{edge}}=\sum\angle(\partial_x\mathbf p,\partial_x\hat{\mathbf p})+\angle(\partial_y\mathbf p,\partial_y\hat{\mathbf p})$$

### 6.4 两阶段优化

- 前 5K steps：detach refiner 与 backbone feature，SSR 从 identity 学 residual；
- 之后：base + SSR 联合微调；
- base 用 synthetic + real；SSR 只接收 synthetic pixel-perfect supervision；
- 训练默认 $K=3$；global batch 96，16×A100；
- SSR LR $2\times10^{-4}$，2D heads $10^{-4}$，DINO $5\times10^{-6}$。

SSR 只用 synthetic 的理由是 residual 细节监督需要精确标签；它也意味着真实域细节泛化仍依赖 base feature 与合成到真实的迁移。

---

## 7. Fine-detail benchmark

### 7.1 为什么普通指标不够

全图平均 AbsRel 会被大平面主导，细栏杆即使完全消失，对整体误差影响也小。二维 Boundary F1 只看像素边缘，不能判断升到 3D 后是否扭曲。

论文因此在 Spring 与 Synth4K 自动构建 fine-detail segments，并在每个区域共享全局 scale、只拟合局部 translation，然后用严格的 1% 阈值：

- local depth Rel / $\delta_{0.01}$；
- local point Rel / $\delta_{0.01}$。

细节区域由多尺度 Laplacian residual、形态学 top-/bottom-hat 与 SAM2 segments 联合筛选。

### 7.2 主结果

| 方法 | Local depth Rel↓ | Local depth $\delta_{0.01}$↑ | Local point Rel↓ | Local point $\delta_{0.01}$↑ | Boundary F1↑ | Global point Rel↓ | Metric depth $\delta$↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Depth Pro | 4.15 | 46.9 | 3.01 | 45.2 | 16.3 | 11.1 | 54.0 |
| DA3 | 10.5 | 41.8 | 7.39 | 44.7 | 6.61 | 10.6 | 70.1 |
| InfiniDepth | 5.82 | 43.2 | 4.29 | 43.7 | **19.3** | 11.6 | — |
| MoGe-2 | 3.81 | 47.1 | 3.19 | 46.6 | 15.6 | 8.73 | 77.3 |
| **MoGe-3 ViT-L** | 3.76 | **52.5** | **2.79** | **55.9** | 16.0 | **7.91** | **82.7** |
| **MoGe-3 ViT-G** | **3.66** | **54.1** | 2.80 | **56.8** | 17.3 | **7.43** | **82.7** |

最大提升来自严格 local point $\delta_{0.01}$：46.6→55.9。默认 ViT-L 的 Boundary F1 仅 16.0，不及 InfiniDepth 19.3；这正说明 MoGe-3 更关注三维结构而非单纯二维锐边。

![](assets/MoGe-3%20-%20自引导稀疏体积几何细化/figures/03-qualitative-comparison.png)

> **图 3（Figure 3）**：多组 disparity 看起来都较锐利，但 3D point cloud 暴露出桌腿、栏杆、路杆和街景结构的弯折与混面；MoGe-3 的优势主要体现在换视角后的三维保真。来源：<https://arxiv.org/abs/2607.17967>。

---

## 8. 消融

### 8.1 体素分辨率

| 配置 | Local depth Rel↓ | Local depth $\delta$↑ | Local point Rel↓ | Local point $\delta$↑ | F1↑ |
|---|---:|---:|---:|---:|---:|
| Base | 3.81 | 47.1 | 3.19 | 46.6 | 15.6 |
| 2D Conv | 4.09 | 45.0 | 3.19 | 47.0 | 14.8 |
| RGB Input | 3.90 | 48.1 | 3.13 | 49.7 | 14.9 |
| $D=100$ | 3.86 | 49.8 | 2.86 | 53.2 | 15.7 |
| **$D=200$** | 3.70 | **51.4** | **2.76** | **55.1** | 15.6 |
| $D=300$ | **3.64** | 48.8 | 2.92 | 52.0 | 15.1 |
| $D=400$ | 4.26 | 48.8 | 3.11 | 50.4 | 14.3 |

$D$ 太小分不开细结构，太大则 occupancy 过稀、卷积没有足够邻居。$D=200$ 是准确率、密度、显存和效率折中。

### 8.2 DINO feature vs RGB

RGB + 3D structure 已有少量收益，但 DINO injection 将 local point $\delta$ 从 49.7 提到 55.1。几何邻域不能替代视觉语义，两者必须协作。

### 8.3 迭代次数

训练 $K=3$，测试可运行到 $K=7$：

- 第一步收益最大；
- local metrics 到 $K\approx5$ 饱和；
- global metrics 基本稳定；
- 超过训练步数没有发散。

这说明 residual update 近似收敛过程，但也意味着实时系统可只运行 1 步换速度，离线高质量可运行更多步。

---

## 9. 速度与工程代价

单 A100 FP16：

| 方法 | 分辨率 | 延迟 |
|---|---:|---:|
| MoGe-2 ViT-L | 700² | 39 ms |
| **MoGe-3 ViT-L，3 steps** | 700² | **121 ms** |
| MoGe-3 ViT-G，3 steps | 700² | 177 ms |
| Depth Pro | 1536² | 142 ms |
| DA3 | 630×840 | 213 ms |

SSR 使 ViT-L 延迟约 3.1×，但仍快于表中若干高分辨率 baseline。不同分辨率和 backend 不同，绝对秒数不能视为严格公平。

官方仓库的 MoGe-3 使用 FlexGEMM/Triton 稀疏算子，当前 macOS 没有 Triton wheel，因此本机 macOS 不能直接跑 v3；MoGe-1/2 仍可走普通 PyTorch 环境。

---

## 10. 局限与批判性评价

### 10.1 作者明确局限

1. 回归模型无法表达物体边界像素的前景/背景多解性；
2. 深度边缘仍不完全锐利；
3. 可能出现 fly-points；
4. 未来考虑结合 Pixel-Perfect Depth 式生成建模。

### 10.2 表示边界

$(i,j,D\log Z)$ 壳层非常高效，但不是欧氏世界坐标体素；相机视角、分辨率和投影参数仍决定邻域。单图也没有真正恢复遮挡后表面。因此 MoGe-3 是**可见表面细化**，不是完整 3D scene completion。

### 10.3 评测边界

fine-detail local metric 只在 Spring、Synth4K 上构建，Boundary F1 在四个数据集上评测。真实开放域细结构的系统统计仍有限；论文也没有给出参数量和精确显存，部署成本不够透明。

### 10.4 个人判断

MoGe-3 是系列中第一次改动“几何在哪里计算”。MoGe-1 改监督表示，MoGe-2 改尺度与数据，MoGe-3 改网络邻域。它的关键贡献不是简单加 sparse U-Net，而是指出**二维锐度不等于三维细节**，并建立能揭露这种差异的 benchmark。

---

## 11. 复现建议

1. 先用 $K=0$ 对齐 MoGe-2 base，再加入 zero-init SSR；
2. 验证 2D parameter-matched refiner，确认收益来自 3D locality；
3. 逐步扫描 $D=100/200/300/400$，同时记录 occupancy、显存和延迟；
4. 检查 object boundaries 的 fly-points，而不只看 depth colormap；
5. Linux + NVIDIA + Triton 环境优先，macOS 暂不适合 MoGe-3 backend；
6. 下游若重实时，可运行 1-step；资产或重建预处理可运行 3–5 steps。

---

## 12. 系列位置

- [[MoGe - 开放域单目几何与仿射不变点图]]：定义可学的 affine point-map target；
- [[MoGe-2 - 米制尺度与锐利单目几何]]：解耦 metric scale，并清理真实标签；
- **MoGe-3**：把局部 refinement 从 2D 图像平面搬到自引导 sparse 3D shell。

三代形成了“**监督表示 → 数据与尺度 → 网络几何邻域**”的完整演化。

---

## 13. 材料来源

- [arXiv](https://arxiv.org/abs/2607.17967)
- [项目页](https://qft-333.github.io/moge3page/)
- [官方代码](https://github.com/microsoft/MoGe)
- [[assets/MoGe-3 - 自引导稀疏体积几何细化/sources|本地来源记录]]
