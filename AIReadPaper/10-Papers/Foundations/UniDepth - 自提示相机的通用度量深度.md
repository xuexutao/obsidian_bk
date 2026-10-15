---
type: paper-note
status: complete
domain: Foundations
paper: "UniDepth: Universal Monocular Metric Depth Estimation"
year: 2024
arxiv: "2403.18913"
source: https://arxiv.org/abs/2403.18913
code: https://github.com/lpiccinelli-eth/unidepth
tags:
  - metric-depth
  - camera-prediction
  - monocular-geometry
created: 2026-10-15
updated: 2026-10-15
---

# UniDepth：自提示相机的通用度量深度

## 基本信息

| 项目 | 内容 |
|---|---|
| 任务 | 单张 RGB、无需外部内参 → metric depth + per-pixel 3D points |
| 作者 | Luigi Piccinelli 等，ETH Zürich / INSAIT |
| 模型 | ConvNeXt-L 或 DINO ViT-L encoder + camera module + depth module |
| 数据 | 9 个真实数据集，约 3M 图像 |
| 重要性 | ★★★★☆：把“未知相机时的 metric depth”显式拆成相机自提示与深度预测 |

> **一句话结论：** UniDepth 的核心不是直接回归 Cartesian 3D，而是先预测稠密相机射线，用球谐编码把它作为 prompt 注入深度模块，并在 $(\theta,\phi,z_{log})$ pseudo-spherical 空间中把相机方向与深度轴解耦。这样模型只凭单图即可给出米制 3D，且在相机内参缺失或有噪时比依赖外部标定的路线更稳。

---

## 1. 问题定义

传统 universal metric depth 常要求测试时提供内参；但网络图片、用户视频和历史素材经常没有 EXIF，或者焦距本身有误。UniDepth 定义更严格的任务：只给 RGB，不给相机参数，直接恢复 metric 3D；若可靠内参存在，也允许用外部 camera prompt 替换自预测结果。

![](assets/UniDepth%20-%20自提示相机的通用度量深度/figures/01-teaser.png)

> 图 1：pseudo-spherical 表示将相机方向 $(\theta,\phi)$ 与深度 $z$ 正交化；Cartesian $(x,y,z)$ 中，深度误差会通过 $x=r_xz,y=r_yz$ 反向污染相机预测。来源：论文 Figure 1，第 1 页。

---

## 2. 输入与输出

输入 $I\in\mathbb R^{H\times W\times3}$。输出为：

$$O=C\Vert Z_{log}\in\mathbb R^{H\times W\times3}$$

其中 $C=(\theta,\phi)$ 是逐像素方位角/仰角，$Z=\exp(Z_{log})$ 是米制深度。二者组合成逐像素三维点。

---

## 3. 核心方法

### 3.1 Camera self-prompting

Camera module 从 encoder 特征预测四个针孔参数残差 $(\Delta f_x,\Delta f_y,\Delta c_x,\Delta c_y)$，按图像尺寸还原 $f_x,f_y,c_x,c_y$，再将每个像素反投影为单位球面射线与角度场 $C$。

稠密角度场经最高阶 $l=8$ 的 Laplace spherical harmonic encoding，形成 81 维相机嵌入 $E$。深度特征 $D$ 通过 cross-attention 接受相机提示：

$$D|E=\operatorname{MLP}(\operatorname{CA}(D,E))$$

相机分支与 encoder 之间 stop-gradient，避免有限相机类型主导 backbone，造成训练域内相机过拟合。

### 3.2 Pseudo-spherical 表示

Cartesian 3D 中 $x,y$ 同时受射线与深度影响，导致相机和深度梯度纠缠。UniDepth 用 $(\theta,\phi,z_{log})$ 使方向和径向深度在输出空间正交。消融中把它改回 Cartesian 后，OOD $\delta_1$ 从 57.06 降到 37.09，camera ARel 从 13.54 恶化到 87.51，是最剧烈的退化之一。

### 3.3 几何不变性损失

同一图像采样两个 resize/crop 视图 $T_1,T_2$，要求**经过 camera conditioning 后**的深度特征互相可变换：

$$\mathcal L_{con}=\left\|T_2\circ T_1^{-1}(D_1|E_1)-\operatorname{sg}(D_2|E_2)\right\|_1$$

它模拟同一场景由不同“表观相机”拍摄，强化跨相机泛化。若在 camera conditioning 前对齐，则会错误抹除本应存在的相机差异。

![](assets/UniDepth%20-%20自提示相机的通用度量深度/figures/02-architecture.png)

> 图 2：Camera module、SHE 相机提示、Depth module 与两视图几何一致性。来源：论文 Figure 2，第 4 页。

### 3.4 主监督目标

定义三维误差 $\epsilon=(\epsilon_\theta,\epsilon_\phi,\epsilon_z)$：

$$\mathcal L_{\lambda MSE}=\|\mathbb V[\epsilon]\|_1+\lambda^T(\mathbb E[\epsilon]\odot\mathbb E[\epsilon])$$

参数 $\lambda_\theta=\lambda_\phi=1,\lambda_z=0.15$。深度维较小的 $\lambda_z$ 引入 SIlog 风格尺度不变性，同时仍保留米制监督。总损失：

$$\mathcal L=\mathcal L_{\lambda MSE}+0.1\mathcal L_{con}$$

---

## 4. 数据与训练

训练数据约 3M 张真实图，来自 Argoverse2、Waymo、DrivingStereo、Cityscapes、BDD100K、Mapillary-PSD、A2D2、ScanNet、Taskonomy。训练 1M iterations，batch 128，8×A100 约 12 天。每张原图生成两个几何增强视图，数据集在 batch 内均衡采样。

---

## 5. 主要结果

### 5.1 NYUv2 / KITTI zero-shot

| 方法 | NYU $\delta_1$/AbsRel | KITTI $\delta_1$/AbsRel |
|---|---|---|
| Metric3D | 0.926 / 0.0938 | 0.975 / 0.0533 |
| ZoeDepth（域内训练） | 0.952 / 0.077 | 0.965 / 0.0576 |
| **UniDepth-V** | **0.984 / 0.0578** | **0.986 / 0.0421** |

UniDepth 在这两项上甚至超过对应域内训练方法，但在 ETH3D、iBims-1 等极端 OOD 相机上可能出现“SIlog 很好、绝对尺度错”的情况，不能把局部最好结果外推为全面优势。

### 5.2 受控对比

在相同 ConvNeXt-L、Metric3D 数据子集、500K iterations 下，UniDepth 在 KITTI/NYU 上分别达 $\delta_1=0.979/0.971$，均高于 iDisc 与 Metric3D，说明收益不只来自数据规模。

### 5.3 相机噪声

![](assets/UniDepth%20-%20自提示相机的通用度量深度/figures/03-camera-noise.png)

> 图 3：外部相机参数被加入噪声后，依赖外部 camera 的模式明显退化；自提示模式曲线几乎保持水平。可靠内参可提高峰值，但未知或噪声内参时自预测更稳。来源：论文 Figure 3，第 5 页。

---

## 6. 消融结论

| 变体 | OOD $\delta_1$ | OOD SIlog | OOD $F_A$ |
|---|---:|---:|---:|
| Full | **57.06** | **14.83** | **49.71** |
| 去 Camera | 48.38 | 15.55 | 45.21 |
| 去 Spherical | 37.09 | 22.49 | 21.78 |
| 去 $\mathcal L_{con}$ | 52.89 | 14.85 | 45.17 |
| 稀疏相机参数替代 dense prompt | 55.65 | 15.04 | 43.19 |
| 不 detach camera gradient | 46.60 | 15.26 | 43.85 |
| Cartesian baseline | 48.19 | 23.05 | 14.29 |

最有价值的结论是：相机分支不是“多一个 head 就好”，而是依赖稠密表示、输出空间解耦、stop-gradient 和跨增强一致性形成整体。

---

## 7. 局限与评价

- 极端 OOD 相机或尺度分布下，模型可能恢复相对结构却估错绝对尺度；
- 自预测相机内部仍是针孔四参数，对鱼眼和强畸变缺乏系统验证；
- 3M 图、1M iterations 的训练成本高；
- metric depth 的强结果依赖训练数据中尺度与相机分布，单图绝对尺度的病态性没有被理论消除。

**个人判断**：UniDepth 的长期价值在“相机作为 prompt”这个接口。它把相机参数从外部硬条件变成模型可预测、可替换、可注入的条件，为后续 MapAnything、WorldMirror、DA3 等“任意先验”几何模型提供了设计模板。

---

## 8. 复现建议

优先使用官方权重。若自训，首先复现 pseudo-spherical 与 stop-gradient 两项，因为消融最强；部署时若 EXIF 可靠，比较 external camera prompt 与 self-prompt 两个模式；对鱼眼、裁剪、缩放图片必须单独做尺度稳定性测试。

---

## 9. 材料来源

见 [sources.md](assets/UniDepth%20-%20自提示相机的通用度量深度/sources.md)。
