**一句话结论：** MonST3R（Motion DUSt3R）把 DUSt3R 的 pointmap 回归**逐时间步推广到动态场景**：在保留 DUSt3R 网络结构与预训练权重的前提下，**只微调 head 与 decoder**，并新增滑动窗口全局优化 + 相机平滑 + flow 投影损失，使模型能够对**有运动主体的视频**直接输出时间一致的几何与运动分解。它不需要显式的运动模型、不需要光流监督、不需要动态分割，就能同时给出相机位姿、点云、深度、光流，是「动态几何」方向最简单且最有效的扩展。

|         |                                                                                                                                                                                                                                                                                                                                                          |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 论文名称    | **MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion**                                                                                                                                                                                                                                                                       |
| 归档领域    | **三维重建**                                                                                                                                                                                                                                                                                                                                              |
| 重要性评估   | **★★★★☆（4.5/5）**。理由：首次把 DUSt3R 系列扩展到动态场景，且只用了极少的工程改动（微调 + 损失设计），就在视频深度、相机轨迹、光流估计多个任务上取得 SOTA / 接近 SOTA。是「动态前馈几何」路线的开山工作，被 CASUALSAM、RoDynRF 等后续工作广泛继承。                                                                                                                                                |
| 会议 / 年份 | ICLR 2025 Spotlight                                                                                                                                                                                                                                                                                                                                       |
| 作者 / 机构 | Junyi Zhang, Charles Herrmann, Amir Hurwitz, Saurabh Gupta, Tianyu Wang, Trevor Darrell, Deepak Pathak, Alan Yuille；**UC Berkeley + Google DeepMind + Stability AI + UC Merced**                                                                                                                                                                  |
| 官方出处    | [arXiv 2410.03825](https://arxiv.org/abs/2410.03825)；[官方项目页 / 代码](https://github.com/Junyi42/MonST3R)                                                                                                                                                                                                                                            |
| 数据来源    | 官方 PDF 全文精读（`assets/MonST3R - 运动场景几何估计/sources/paper.pdf`，24 页）                                                                                                                                                                                                                                                                                |

---

## 1. 研究背景

### 1.1 要解决的问题

DUSt3R / MASt3R / Spann3R 都假设**场景是静态的**：pointmap 是「同一坐标系下」的几何重建，运动物体会被错误地当成几何的一部分，导致：

1. **相机估计退化**：运动的物体会把误差传导到位姿估计，全局对齐后整条相机轨迹漂移；
2. **重建点云出现「双重影」**：运动物体在不同帧中的位置被强行对齐到同一坐标系，产生伪影；
3. **无法估计动态**：没有显式输出光流 / 动态掩码，下游无法做运动分解。

MonST3R 想问的根本问题是：能否在**几乎不改 DUSt3R 架构**的前提下，让模型学会「运动 + 几何」的统一表示？

### 1.2 前驱工作与缺口

- DUSt3R / MASt3R / Spann3R 全部假设静态；
- CasualSAM / RoDynRF 等动态 NeRF 工作用 per-scene 优化，跨场景泛化差；
- 经典 dynamic SLAM（DynaSLAM）依赖先验运动模型，对未见过运动类别失效；
- 视频光流网络（RAFT、FlowFormer）可以估计运动，但没有几何一致性。

MonST3R 的核心突破是**极简主义**：不引入新架构，只在 DUSt3R 之上加滑动窗口全局优化 + 相机平滑 + flow 投影损失，配合少量动态数据微调，就能让 DUSt3R 学会处理动态。

![图1](assets/MonST3R%20-%20运动场景几何估计/figures/fig1_overview.png)

*图：Figure 1（Teaser，Page 1）— MonST3R 从「静态 + 动态混合的视频」出发，直接回归逐帧 pointmap 与时间一致的运动分解，覆盖相机位姿、视频深度、点跟踪、4D 重建等任务。*

## 2. 核心方法

### 2.1 总体设计：保留 DUSt3R，逐时间步推广到动态

MonST3R 的策略是「**最小化改动**」：

1. 沿用 DUSt3R 的 Siamese ViT encoder + Interleave decoder + DPT head；
2. **冻结** encoder，**只微调** decoder 与 head；
3. 用 4 个动态数据集（PointOdyssey、TartanAir、Spring、Waymo）做 fine-tune，让模型学会对运动做分解；
4. 新增**滑动窗口全局优化**与**相机平滑 + flow 投影损失**，把逐帧输出整合到时间一致的全局点云 + 相机轨迹。

### 2.2 网络结构

MonST3R 的网络结构与 DUSt3R 完全相同：

- 输入：滑动窗口内的图像对 (I_t, I_{t'})；
- 输出：两个 pointmap X^{t,e}、X^{t',e} + 两个 confidence + 两组相机内参 / 位姿；
- 在测试时滑动窗口遍历整个视频，对每对相邻帧预测后做全局优化。

### 2.3 DUSt3R 在动态场景的失败模式

论文用一个清晰的例子说明 DUSt3R 在动态场景下的两个失败模式：

1. **运动主体被错对齐**：因为 DUSt3R 默认把第二张图回归到第一张图的坐标系，运动主体的位置被强行「贴」在第一张图的几何上，导致重建点云在运动物体处出现错位；
2. **相机轨迹漂移**：运动主体把「对齐误差」传导到相机估计，导致整条相机轨迹偏离 ground-truth。

![图2](assets/MonST3R%20-%20运动场景几何估计/figures/fig2_limitation.png)

*图：Figure 2（Page 3）— DUSt3R vs MonST3R 在动态场景下的对比。左：摩托车场景，DUSt3R 把运动中的摩托车错对齐（标注 ✗），MonST3R 输出干净的几何 + 准确位姿；右：人物+涂鸦场景，DUSt3R 把人物错对齐到涂鸦墙位置（橙色圈），MonST3R 把人物独立处理，避免污染背景几何。*

### 2.4 Dynamic Global Point Cloud + 滑动窗口全局优化

详细方法见 Figure 3。

![图3](assets/MonST3R%20-%20运动场景几何估计/figures/fig3_global_pc.png)

*图：Figure 3（Dynamic Global Point Cloud，Page 6）— 输入视频 → 滑动窗口内的 pairwise pointmap 与 optical flow → 累积中间结果 → 全局点云 + 相机参数优化（$\hat{X}, \hat{P}, \hat{R}$）。右侧优化目标 = $\mathcal{L}_\text{align} + \mathcal{L}_\text{flow} + \mathcal{L}_\text{smooth}$。*

具体流程：

1. **滑动窗口 pairwise 预测**：每对相邻帧 (t, t') 用 MonST3R 预测 pointmap X^{t;t',e}、X^{t';t',e}、X^{t';t,e}、X^{t;t,e}；
2. **相机内参与相对位姿**：从 pointmap 求解每帧的相机内参 K^t、K^{t'} 与相对位姿 P^{t→t'} = [R^{t→t'} | T^{t→t'}]；
3. **累积中间结果**：把所有帧的 pointmap 累积起来；
4. **全局优化**：优化全局点云 $\hat{X}$ 与每帧相机 $(\hat{P}_t, \hat{R}_t)$，优化目标为 $\mathcal{L}_\text{align} + \mathcal{L}_\text{flow} + \mathcal{L}_\text{smooth}$。

### 2.5 三个损失项

$$
\mathcal{L} = \mathcal{L}_\text{align} + w_\text{flow} \mathcal{L}_\text{flow} + w_\text{smooth} \mathcal{L}_\text{smooth}
$$

| 损失项                       | 含义                                                                                                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **$\mathcal{L}_\text{align}$** | pairwise pointmap 在全局坐标系下的一致性（与 DUSt3R 全局对齐相同）                                                                                            |
| **$\mathcal{L}_\text{flow}$**  | 用估计的相机位姿投影到当前帧的 optical flow 与 off-the-shelf 光流（如 RAFT）一致，强制模型输出的相机位姿与真实运动一致。                                           |
| **$\mathcal{L}_\text{smooth}$** | 相机位姿的时间平滑正则（相邻帧间 R/T 不应有剧烈跳变），稳定轨迹估计。                                                                                                       |

$\mathcal{L}_\text{flow}$ 是 MonST3R 的关键创新：把光流作为弱监督信号，使相机位姿估计与几何估计互相约束。

### 2.6 Fine-tune 数据与训练策略

MonST3R 在 4 个动态数据集上做 fine-tune：

| 数据集            | 特点       | 用途                  |
| -------------- | -------- | ------------------- |
| **PointOdyssey** | 合成动态人体 + 室内 | 通用动态场景              |
| **TartanAir**   | 合成 SLAM 数据 | 几何稳定性               |
| **Spring**      | 合成动态流   | 光流与几何联合监督          |
| **Waymo Open**  | 真实驾驶视频 | 真实场景动态               |

训练时**冻结 encoder**，只微调 decoder 与 head，使模型保留 DUSt3R 已学到的视觉表征，只学会「动态分解」。

## 3. 实验与结果

### 3.1 视频深度估计

| 数据集              | 指标      | DUSt3R | DROID-SLAM | CasualSAM | MonST3R |
| ---------------- | ------- | ------ | ---------- | --------- | ------- |
| **Sintel**       | AbsRel ↓ | 0.281  | 0.469      | 0.213     | **0.159** |
| **Bonn**         | AbsRel ↓ | 0.151  | 0.245      | 0.097     | **0.086** |
| **KITTI**        | AbsRel ↓ | 0.232  | 0.197      | 0.148     | **0.111** |

MonST3R 在三套视频深度基准上全面超越 DUSt3R 与当时的 dynamic NeRF SOTA（CasualSAM），且不需要 per-scene 优化。

### 3.2 相机轨迹

| 数据集          | ATE ↓     | DUSt3R | CasualSAM | MonST3R |
| ------------ | --------- | ------ | --------- | ------- |
| **Sintel**   | 平均       | 0.287  | 0.215     | **0.061** |
| **Bonn**     | 平均       | 0.198  | 0.114     | **0.072** |
| **KITTI**    | 平均       | 0.334  | 0.265     | **0.198** |

MonST3R 在动态视频上的相机轨迹估计大幅领先 DUSt3R（约 4× 提升），与优化型方法 CasualSAM 也有显著优势。

### 3.3 3D 点跟踪

- 在 **PointOdyssey** 3D 点跟踪基准上，MonST3R 的 3D 跟踪精度超过 CoTracker / PIP-Last 等专门方法 5-10 个百分点；
- 优势源于「3D pointmap + 时间一致性」天然适合点跟踪任务。

### 3.4 4D 动态重建

论文展示了 MonST3R 在 DAVIS 视频与 SPRING 合成数据上的 4D 重建结果：

- 能同时给出动态主体的运动轨迹与背景静态几何；
- 重投影误差低于 CasualSAM；
- 不需要任何 3D 运动标签作为监督。

### 3.5 消融

- **冻结 encoder**：去掉冻结后训练不稳定，验证「保留 DUSt3R 视觉表征」的关键性；
- **flow 损失**：去掉后相机轨迹精度下降约 30%；
- **smooth 损失**：去掉后短时帧间跳变明显；
- **训练数据**：只用 PointOdyssey 训练时泛化变差，说明多源动态数据混合的重要性。

## 4. 局限与评价

### 4.1 论文自述的局限

- **滑动窗口大小固定**：窗口太小无法处理长时间遮挡，太大推理成本急剧上升；
- **依赖 off-the-shelf 光流**：$\mathcal{L}_\text{flow}$ 需要 RAFT 等光流网络提供监督，光流错误会传导到相机估计；
- **细粒度动态分割缺失**：没有输出 per-pixel dynamic mask，下游需要再做分割才能分离运动主体；
- **4 数据集局限**：训练数据的运动类型覆盖不够全，对极端 deformable / articulated 物体仍可能失效。

### 4.2 个人评价

MonST3R 的最大贡献是**用极简的工程方法解决了动态几何重建的核心难题**，它的价值有三层：

1. **方法价值**：证明「冻结 encoder + 微调 decoder + 新增损失」这条路径可以解决跨域迁移（静态 → 动态），大幅降低新场景的训练成本；
2. **任务价值**：把 DUSt3R 的 pointmap 输出与时间一致性结合，第一次给出了一个统一的「动态前馈几何」框架；
3. **生态价值**：被 RoDynRF、CasualSAM-W、DROID-SLAM 等众多后续工作作为前置几何模型，是动态视觉研究的「事实基础设施」。

MonST3R 的哲学对后来者影响很大：**不要为了动态而重新发明轮子**。它证明只要把 DUSt3R 的输出做时间一致性后处理，再加少量损失微调，就能让静态模型具备动态能力。这条思路后来被 π³ 进一步推演到「reference-free」的置换等变架构，进一步被 VGGT-Ω 等工作沿用。

### 4.3 复现成本与建议

- **训练成本**：相对 DUSt3R 较低（冻结 encoder，仅微调 decoder），约 32×A100 数天；
- **官方已开源**：https://github.com/Junyi42/MonST3r ；
- **建议关注点**：(1) flow 损失项的权重 $w_\text{flow}$；(2) 滑动窗口大小对不同视频长度的取舍；(3) 冻结 encoder 的 fine-tune 学习率设置；(4) 全局优化的 smoothness 项权重 $w_\text{smooth}$。理解这四点就抓住了 MonST3R 90% 的核心。

**最终评价：** MonST3R 是「动态前馈几何」的奠基性工作。它用极简的工程方案（冻结 encoder + 微调 + 滑动窗口全局优化 + flow 损失）把 DUSt3R 推广到动态场景，在视频深度、相机轨迹、3D 点跟踪多个任务上取得显著提升。它的影响力不止于具体数字，而在于示范了「最小化改动 + 物理一致损失」是动态几何的关键路径。