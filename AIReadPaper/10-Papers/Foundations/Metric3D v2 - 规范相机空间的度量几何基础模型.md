---
type: paper-note
status: complete
domain: Foundations
paper: "Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation"
year: 2024
arxiv: "2404.15506"
source: https://arxiv.org/abs/2404.15506
project: https://JUGGHM.github.io/Metric3Dv2/
tags:
  - metric-depth
  - surface-normal
  - canonical-camera
  - geometric-foundation-model
created: 2026-10-15
updated: 2026-10-15
---

# Metric3D v2：规范相机空间的度量几何基础模型

## 基本信息

| 项目 | 内容 |
|---|---|
| 任务 | 单图 + 相机内参 → metric depth + surface normal |
| 数据 | 超过 16M 图像，数千种相机模型 |
| 模型 | ConvNeXt-L 或 DINOv2-reg ViT-S/L/G + DPT/UNet |
| 训练 | 48×A100，800K iterations，batch 192 |
| 重要性 | ★★★★☆：把跨相机 metric ambiguity 显式转化为 canonical camera 归一化问题 |

> **一句话结论：** Metric3D v2 的关键是把所有训练样本变换到统一焦距的 canonical camera space，使不同相机下外观相近的图像不再对应冲突的米制监督；再以 Random Proposal Normalization 强化局部几何，并用 ConvGRU 联合优化深度与法线，使法线分支能从大量只有深度标签的数据中吸收几何知识。

---

## 1. 问题定义

相同物体在焦距 $f$、距离 $d$ 同比例变化时可形成近似相同图像，因此只凭外观不能区分绝对尺度。Metric3D v2 认为焦距是 metric depth 的关键外部变量；传感器尺寸与像素尺寸在投影中会相互抵消。

输入为单图 RGB 与已知/估计内参，输出米制深度 $D$ 和单位表面法线 $N$。与 UniDepth、Depth Pro 不同，严格推理仍需要焦距；没有 EXIF 的网络图示例中，论文会人工选取合理焦距。

![](assets/Metric3D%20v2%20-%20规范相机空间的度量几何基础模型/figures/01-depth-normal-results.png)

> 图 2：开放域图像上的深度与法线。模型直接输出法线，而不是从深度差分得到，因此在细节和噪声上更干净。来源：论文 Figure 2，第 2 页。

---

## 2. Canonical Camera Space Transformation

### 2.1 标签变换 CSTM-label

设 canonical 焦距为 $f^c$，原始焦距为 $f$：

$$\omega_d=\frac{f^c}{f},\qquad D_c^*=\omega_dD^*$$

图像不变，网络学习 canonical space 深度；推理时：

$$D=\frac{f}{f^c}D_c$$

### 2.2 图像变换 CSTM-image

$$\omega_r=\frac{f^c}{f},\qquad I_c=T(I,\omega_r)$$

图像按焦距比例缩放，深度值不缩放，只做空间 resize；推理后再 resize 回原分辨率。

两种实现都把“相机变化”从学习问题变成确定的几何预处理/后处理。消融中无 CSTM 的模型在 NuScenes AbsRel 达 1.000，而 CSTM-image 为 0.197、CSTM-label 为 0.213，说明直接混合多相机 metric supervision 几乎不可行。

![](assets/Metric3D%20v2%20-%20规范相机空间的度量几何基础模型/figures/02-pipeline.png)

> 图 9：输入与深度 GT 先转到 canonical camera space，网络联合预测深度与法线；深度在输出端 de-canonicalize 回真实尺度。来源：论文 Pipeline 图。

---

## 3. Random Proposal Normalization Loss

全图 scale-shift normalization 容易压缩局部差异。RPNL 每图随机采样 $M=32$ 个区域，尺寸占原图 12.5%–50%，在每个 proposal 内独立做 median/MAD 归一化：

$$\tilde d_{p_i,j}=\frac{d_{p_i,j}-\mu(d_{p_i})}{\frac1N\sum_j|d_{p_i,j}-\mu(d_{p_i})|}$$

$$L_{RPNL}=\frac1{MN}\sum_{i,j}|\tilde d_{p_i,j}^*-\tilde d_{p_i,j}|$$

它牺牲局部 absolute scale，只强化区域内部形状与深度层次，补足全图 metric loss 对细节不敏感的问题。

---

## 4. 深度—法线联合优化

### 4.1 为什么联合

论文收集约 9.49M 室外深度图，但真实室外法线标签少于 20K。直接训练法线会被室内平面主导。Metric3D v2 用深度作为大规模几何教师，让法线分支从深度数据学习。

### 4.2 ConvGRU 迭代更新

初始低分辨率深度 $\hat D_c^0$、法线 $\hat N_u^0$ 与 hidden state $H^0$ 输入循环块：

$$H^{t+1}=\operatorname{ConvGRU}(\hat D^t,\hat N^t,H^0,H^t)$$

$$\hat D_c^{t+1}=\hat D_c^t+G_d(H^{t+1}),\qquad \hat N_u^{t+1}=\hat N_u^t+G_n(H^{t+1})$$

最终深度经 ReLU，法线单位化。没有法线 GT 时，从预测 metric depth 计算 pseudo-normal，施加 depth-normal consistency。

![](assets/Metric3D%20v2%20-%20规范相机空间的度量几何基础模型/figures/03-joint-optimization.png)

> 图 10：深度和法线在共享 ConvGRU hidden state 中迭代交换信息，两个 projection head 分别输出更新量。来源：论文 Figure 10。

总损失：

$$L=0.5L_d+L_n+0.01L_{d-n}$$

深度损失包含 PWN、VNL、SIlog 与 RPNL。

---

## 5. 数据与训练

论文称使用超过 16M 图像，数据表列出 DDAD、Argoverse2、Waymo、Taskonomy、ScanNet、HM3D、HyperSim 等 18 个来源；正文“16 个数据集”与表格数量不一致，应以具体列表为准。所有训练数据都需要内参。

- ViT-S/L/G 使用 DINOv2-reg 初始化，DPT 解码；
- 训练 800K iterations，batch 192，48×A100；
- ConvNet 输入 $512\times960$，ViT 输入 $616\times1064$；
- 数据集在 batch 内平衡采样。

---

## 6. 主要结果

### 6.1 Zero-shot metric depth

| 数据集 | 模型 | $\delta_1$ | AbsRel |
|---|---|---:|---:|
| NYUv2 | ViT-L CSTM-label | 0.975 | **0.063** |
| NYUv2 | ViT-G CSTM-label | **0.980** | 0.067 |
| KITTI | ViT-L CSTM-label | 0.974 | 0.052 |
| KITTI | ViT-G CSTM-label | **0.977** | **0.051** |

微调后，ViT-G 在 KITTI 达 $\delta_1=0.989$、AbsRel=0.039；ViT-L 在 NYU 达 $\delta_1=0.989$、RMSE=0.183。

### 6.2 未见数据集

同一 ViT-G CSTM-image 模型在 DIODE Indoor/Outdoor、iBims-1、ETH3D、NuScenes 统一测试。NuScenes AbsRel=0.129，明显优于 ZoeDepth 0.504，体现跨相机归一化的价值；但 ETH3D AbsRel=0.363，说明极端域仍有明显尺度问题。

### 6.3 法线

NYUv2 zero-shot，ViT-L 的 $11.25^\circ/22.5^\circ/30^\circ$ 准确率为 0.662/0.831/0.881，平均角误差 $13.1^\circ$；ScanNet 上 ViT-G 达 0.778/0.901/0.935。

### 6.4 SLAM 下游

将 metric depth 直接输入 Droid-SLAM，KITTI 7 个序列平均平移漂移从约 25.38% 降至 1.81%。这比 benchmark 上小幅提升更能说明 metric scale 的实际价值：它直接约束单目 SLAM 的尺度漂移。

---

## 7. 消融

CSTM AbsRel：

| 方法 | DDAD | NuScenes | KITTI | NYU |
|---|---:|---:|---:|---:|
| 无 CSTM | 0.530 | 1.000 | 0.568 | 0.584 |
| CamConvs | 0.295 | 0.423 | 0.178 | 0.333 |
| CSTM-image | 0.190 | **0.197** | 0.097 | **0.210** |
| CSTM-label | **0.183** | 0.213 | **0.081** | 0.212 |

显式 canonical transform 远胜于让网络隐式理解相机模型。作者还发现 canonical focal length $f^c=1000$ 略优。

联合优化定性消融显示：只用法线标签会产生平坦表面；去掉 depth-normal consistency 后法线明显变粗；说明循环块与显式一致性都是知识迁移的关键。

---

## 8. 局限与评价

- 依赖已知或可靠焦距；无 EXIF 的开放域图片不能完全自动 metric reconstruction；
- 错误焦距会直接转化为错误尺度；
- 室外法线标签极少，pseudo-normal 质量受深度误差约束；
- 逐帧模型没有视频/多视图一致性；
- ViT-G 约 1B 参数，训练成本极高；
- ScanNet 对部分模型并非严格 zero-shot，阅读结果时必须区分。

**个人判断**：Metric3D v2 解决的是“已知相机时，如何统一跨相机 metric supervision”。UniDepth/Depth Pro 则解决“相机未知时如何预测它”。它们不是简单替代关系：可靠内参存在时 CSTM 更直接；网络图片/历史视频中，self-calibration 更实用。

---

## 9. 复现建议

先复现 CSTM-label，这是收益最大、实现最简单的部分；再加 RPNL 验证局部细节；联合法线优化工程复杂度更高，可作为第二阶段。部署时必须记录焦距来源和置信度，避免把人工/估计焦距误当成真实标定。

---

## 10. 材料来源

见 [sources.md](assets/Metric3D%20v2%20-%20规范相机空间的度量几何基础模型/sources.md)。
