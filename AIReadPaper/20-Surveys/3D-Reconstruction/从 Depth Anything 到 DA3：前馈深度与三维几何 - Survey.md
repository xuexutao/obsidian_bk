---
type: survey
status: complete
domain: 3D-Reconstruction
title: "从 Depth Anything 到 DA3：前馈深度与三维几何的范式演化"
period: "2023-2025"
created: 2026-10-15
updated: 2026-10-15
tags:
  - monocular-depth
  - feed-forward-3D
  - geometric-foundation-model
  - survey
---

# 从 Depth Anything 到 DA3：前馈深度与三维几何的范式演化

## 基本信息

**范围。** 本综述以 2023–2025 年为主，追踪两条原本分离、后来逐渐汇合的路线：

1. **单目深度 / 单目几何基础模型**：ZoeDepth、Metric3D、Depth Anything 1/2、UniDepth、Metric3D v2、Depth Pro、MoGe；
2. **前馈式多视图几何重建**：DUSt3R、MASt3R、Spann3R、MonST3R、Fast3R、VGGT、π³、MapAnything、DA3。

**核心判断。** 这段历史不是“深度模型越来越大”，而是视觉几何的预测接口发生了三次升级：

$$\text{单图相对深度}\rightarrow\text{单图度量/点图}\rightarrow\text{任意视图的相机+几何}$$

Depth Anything 三代恰好落在三个节点：V1 解决开放域数据覆盖，V2 解决标签精度与边界，DA3 则把单目深度扩展为统一的视觉空间恢复。

---

## 1. 根本问题：前馈模型究竟要预测什么

### 1.1 同一张图存在多个几何等价解

单目图像不能唯一决定真实尺度。把相机焦距扩大 $k$ 倍、拍摄距离也扩大 $k$ 倍，投影可近似不变。因此“像素 → 米制距离”必须依赖训练分布、相机先验或场景语义。不同方法并不是在解完全相同的任务，而是在选择不同的等价类：

| 输出 | 允许的不确定性 | 能直接做什么 | 代表方法 |
|---|---|---|---|
| Relative / affine-invariant depth | 全局 scale + shift | 遮挡关系、ControlNet、相对布局 | MiDaS、DA1/DA2 |
| Metric depth | 绝对米制尺度 | 测量、SLAM、metric fusion | ZoeDepth、Metric3D v2、UniDepth、Depth Pro |
| Affine-invariant pointmap | 3D scale + translation | 单图可见表面几何、FOV 恢复 | MoGe |
| Shared-frame pointmap | 多视图共同参考系中的 3D 点 | 匹配、位姿、稠密重建 | DUSt3R、MASt3R |
| Depth + camera | 每视图深度与相机参数 | 反投影全局点云 | VGGT、MapAnything |
| Depth + ray | 像素深度与世界射线 | 统一相机、点云和多视图几何 | DA3 |

### 1.2 “Feed-forward”不是单一标签

这里用 feed-forward 表示：主要几何信号由训练好的网络一次或有限次前向产生，而不是把每个新场景作为优化变量，运行完整 SfM+BA+MVS。仍需进一步区分：

- **严格单次前馈**：VGGT、Fast3R、DA3；
- **前馈预测 + 轻量闭式求解**：MoGe 的 FOV solver、DA3 的 DLT/RQ；
- **前馈 pairwise + 全局对齐**：DUSt3R、MASt3R；
- **状态式/递归前馈**：Spann3R、CUT3R；
- **前馈初始化 + 可选 BA**：VGGT、Fast3R 的增强模式。

把这些统称“前馈”而不注明后处理，会掩盖实际延迟、可扩展性和误差来源。

---

## 2. 第一阶段：从域内深度到开放域相对深度

### 2.1 MiDaS / DPT 奠定结构

MiDaS 用多数据集混合与 scale-shift-invariant loss，把训练目标从“某个相机的米制深度”改成跨域相对深度。DPT 则解决 ViT token 到逐像素预测的分辨率落差。此后 Depth Anything、DUSt3R、VGGT、DA3 都继续使用 DPT 或其变体，说明 2023–2025 的主要进步更多来自 backbone、数据与监督，而非解码器革命。

### 2.2 Depth Anything V1：数据覆盖成为第一矛盾

[[Depth Anything - 大规模无标注数据驱动的单目深度]] 用约 1.5M 有标签图像训练 teacher，再为 62M 无标签图像生成伪深度。真正有效的不是朴素伪标签，而是：

- student 接收 Color Jitter、Blur、CutMix 强扰动；
- teacher 标签来自 clean image；
- student 保持与冻结 DINOv2 的语义特征相似，但设 $\alpha=0.85$ 容忍阈值。

![](assets/从%20Depth%20Anything%20到%20DA3：前馈深度与三维几何%20-%20Survey/figures/01-depth-anything-data-engine.png)

> 图 1：V1 把深度估计变成“海量无标签真实图 + 伪标签难化学习”的数据工程。来源：Depth Anything Figure 2。

V1 的消融揭示了一个可迁移结论：**<span style="color: red">伪标签数据只有在 student 被迫解决 teacher 没有直接给出的难题时，才会变成新知识</span>。** 这条原则后来贯穿 V2 和 DA3 的 teacher-student 路线。

### 2.3 瓶颈转移：不是数据不够，而是标签不准

V1 已在传统相对深度 benchmark 上显著超过 MiDaS，但边界、玻璃、镜面、屏幕和薄结构仍粗糙。原因不是判别式架构缺乏表达力，而是真实深度“GT”本身由传感器、双目或 SfM 产生，包含系统性错误。

---

## 3. 第二阶段：从真实粗标签到合成精确监督

### 3.1 Depth Anything V2：精度与分布分工

[[Depth Anything V2 - 精确合成监督与真实伪标签桥接]] 把数据角色拆开：

- **合成图**提供透明面、反射面、薄结构的精确深度；
- **真实无标签图**提供开放域外观与场景多样性；
- **DINOv2-G teacher**把两者连接起来；
- **小 student**只在真实伪标签上训练，实现 1.3B → 25M 的预测级蒸馏。

V2 最重要的消融是：同一 DIML 真实图，teacher 伪标签训练的 DA-2K 准确率 89.7%，原始人工/传感器标签只有 80.2%。这推翻“真实标注天然比伪标签可靠”的朴素假设。

### 3.2 Depth Pro：边界成为独立评价维度

[[Depth Pro - 高分辨率锐利度量深度]] 进一步把锐度正式化：固定 1536²，多尺度共享 ViT patch encoder + 全局 image encoder，第一阶段真实/合成混合学习泛化，第二阶段只用合成深度和 Scharr/Laplace 多尺度导数损失锐化。

它还提出 scale-invariant 遮挡边界 F1/Recall。Depth Pro 在六个边界数据集全部第一，并在 V100 上约 0.3 秒输出 2.25MP 深度。这说明现代深度模型的评价已从平均 AbsRel 扩展到**边界定位 × 绝对尺度 × 延迟**。

### 3.3 评测体系为什么必须变化

旧 benchmark 的三类问题：

1. GT 噪声：玻璃、镜子、动态物体和细边界错误；
2. 场景窄：NYU 主要室内，KITTI 主要街景；
3. 分辨率低：难以测量高频结构。

V2 的 DA-2K 用稀疏人工相对深度对补充开放域鲁棒性；Depth Pro 用边界指标补充几何锐度；MoGe 用 pointmap error 衡量反投影后的 3D 形状。评测变化本身就是领域成熟的标志。

---

## 4. 第三阶段：度量尺度与相机进入模型

### 4.1 ZoeDepth：相对预训练 + metric head

ZoeDepth 代表早期折中：先学开放域 relative depth，再在 NYU/KITTI 等域内数据上接 metric bins head。优点是复用相对深度泛化；缺点是室内/室外需要不同模型，且相机分布变化仍困难。

### 4.2 Metric3D v2：已知焦距时规范化相机

[[Metric3D v2 - 规范相机空间的度量几何基础模型]] 把多相机 metric ambiguity 显式变换到 canonical camera space：

$$D_c=\frac{f^c}{f}D$$

只要焦距可信，所有样本可在统一 $f^c$ 下训练 metric depth。它还联合输出 surface normal，并通过 ConvGRU 让法线从大量 depth-only 数据学习。将其深度输入 Droid-SLAM 后，KITTI 平均平移漂移从约 25.38% 降到 1.81%，说明 metric depth 对下游不是“多一个指标”，而是直接改变系统稳定性。

### 4.3 UniDepth：相机从输入变成自提示

[[UniDepth - 自提示相机的通用度量深度]] 不要求外部内参。Camera module 预测稠密射线角度场，经球谐编码提示 depth module；输出用 $(\theta,\phi,z_{log})$ pseudo-spherical 表示解耦方向与深度。相机有可靠值时可以外部注入，未知或有噪时则用 self-prompt。

关键消融中，去 pseudo-spherical 表示使 OOD $\delta_1$ 从 57.06 降到 37.09；不做 camera gradient detach 则训练域相机更准、OOD 反而更差。问题从“预测焦距”上升为“如何让相机监督不污染视觉表征”。

### 4.4 Depth Pro：把焦距估计独立出来

Depth Pro 选择另一种工程答案：深度网络预测 canonical inverse depth，独立 FOV head 在不同数据配方上训练。六个焦距测试集 $\delta_{25\%}$ 领先巨大，说明“深度与焦距分开训练、推理时组合”在可部署性上很强。

### 4.5 三种 metric 路线的适用边界

| 路线 | 相机要求 | 优点 | 主要风险 |
|---|---|---|---|
| Metric3D v2 | 已知/可信焦距 | 几何变换明确、metric 稳定、可联合 normal | 内参错即尺度错，网络图需人工/外部焦距 |
| UniDepth | 自预测或外部稠密 camera prompt | 无 EXIF 可用，接口灵活 | 极端 OOD 相机下绝对尺度波动 |
| Depth Pro | 独立焦距头 | 高分辨率、快速、焦距表现强 | 焦距误差线性传到深度，缺 uncertainty |

---

## 5. 第四阶段：从深度图到单目三维点图

### 5.1 MoGe：直接监督 3D 形状

[[MoGe - 开放域单目几何与仿射不变点图]] 指出：深度图视觉上很锐利，不代表反投影点云正确；错误焦距会造成三维横向扭曲。它直接预测 affine-invariant pointmap，并用 ROE 求训练时的最优 scale/shift。

![](assets/从%20Depth%20Anything%20到%20DA3：前馈深度与三维几何%20-%20Survey/figures/02-moge-pointmap.png)

> 图 2：MoGe 把单图输出从 depth 提升为 pointmap，推理后再恢复 depth、shift 与 FOV；训练同时监督全局形状、三尺度局部几何、法线与 infinity mask。来源：MoGe Figure 3。

MoGe 的价值在于改变评价对象：不再只问每个像素的 $z$ 是否接近，而是问整个可见表面在 3D 中是否变形。它在 affine pointmap Rel 上把 UniDepth 的 9.86 降到 6.43。

### 5.2 单目 pointmap 的边界

MoGe 仍无法从单图恢复绝对尺度和遮挡背面；它假设针孔、主点居中、方形像素。它解决的是“可见表面形状监督一致性”，不是完整 3D 重建。

---

## 6. 第五阶段：DUSt3R 把双目几何改写成回归

### 6.1 DUSt3R：pointmap 成为多视图接口

[[DUSt3R - 前馈式几何重建奠基]] 输入两张无标定、无位姿图像，输出统一参考相机坐标系下的两个 pointmap 与 confidence。相机内参、相对位姿、匹配和深度都变成输出 pointmap 的下游几何读取。

它仍需 pairwise 推理和多视图全局对齐，但完成了最根本的范式切换：

$$\text{特征匹配→对极几何→BA→MVS}\quad\Rightarrow\quad\text{图像对→pointmap}$$

### 6.2 MASt3R：匹配与重建统一

[[MASt3R - 匹配与重建统一]] 在 DUSt3R 上增加 dense local descriptor head，用 3D 真值对应定义 InfoNCE 正样本。Map-free 视觉定位 AUC@0.5m/10° 达 94.4%，比此前 VCRE 64.1% 高约 30 个百分点。它说明 3D 几何本身就是强匹配监督。

### 6.3 Pairwise 范式留下的新瓶颈

- $N$ 张图需要大量图像对；
- 每对尺度和参考系不同；
- 多视图必须全局对齐；
- 无法自然在线或处理上千视角；
- 静态场景假设会把动态对象污染到相机与全局点云。

后续工作几乎都在回答这几个问题。

---

## 7. 第六阶段：多视图、流式、动态与规模化

### 7.1 Spann3R：空间记忆替代全局对齐

[[Spann3R - 空间记忆增量式重建]] 把“历史图像”变成 working + long-term spatial memory，每来一帧直接在全局坐标系回归 pointmap，速度超过 50 FPS。它解决在线性，但引入顺序依赖、固定 memory 阈值和长时漂移。

### 7.2 MonST3R：动态对象进入几何

[[MonST3R - 运动场景几何估计]] 在 DUSt3R 基础上用动态数据微调 decoder/head，并加入 optical-flow 投影和相机平滑。它显著改善 Sintel/Bonn/KITTI 视频深度与相机轨迹，但仍依赖滑动窗口全局优化和 off-the-shelf flow，严格来说是前馈预测与优化的混合系统。

### 7.3 Fast3R：all-to-all 扩到 1000+ 图像

[[Fast3R - 千图单次前馈重建]] 用 Fusion Transformer 同时处理所有视图，预测 local/global 双 pointmap。通过 FlashAttention、位置插值、ZeRO-2 和 tensor parallelism，把训练时 20 视图外推到推理 1500 视图。代价是大规模 all-to-all 的显存/算力门槛极高。

### 7.4 CUT3R：状态式持续更新

CUT3R 用持续更新 Transformer 维护场景 state，新帧既检索 state 又写回 state，兼容视频和无序图集。它比 pairwise 更在线，也比纯 memory bank 更端到端，但 recurrent state 的误差会因果累积。它在世界模型 SPMem、VMem 等系统中被用作在线 pointmap/相机估计器，说明前馈几何正在成为生成系统的基础模块。

---

## 8. 第七阶段：VGGT 把 SfM 输出一次性网络化

[[VGGT - 前馈式通用三维重建]] 用 1.2B Transformer、交替 frame-wise/global attention，一次输出 camera、depth、pointmap、track 与 uncertainty。无需 BA 的前馈结果已超过 DUSt3R/MASt3R 的全局对齐，并获得 CVPR 2025 Best Paper。

![](assets/从%20Depth%20Anything%20到%20DA3：前馈深度与三维几何%20-%20Survey/figures/03-vggt-architecture.png)

> 图 3：VGGT 用相机 token、register token 与交替注意力统一多视图信息，再由 camera head 和 DPT dense head 输出相机、深度、点图与跟踪特征。来源：VGGT Figure 2。

### 8.1 VGGT 的真正创新

不是某个新几何算子，而是证明“最小 3D 归纳偏置 + 大模型 + 海量 3D 标注”足以直接学习完整视觉几何。ETH3D 中 depth+camera 反投影甚至比直接 pointmap 更准（0.677 vs 0.709），显示**过完备输出并非浪费，分解复杂任务有时更容易学习。**

### 8.2 新问题：固定第一参考帧

VGGT 把第一帧设为世界参考，并使用特殊 token 区分它；输入顺序因此不是完全置换等变。[[π³ - 置换等变视觉几何学习]] 去掉固定参考帧与 frame 位置编码，预测尺度不变 local pointmap 和相对位姿，使输入重排的结果方差接近零。它提醒我们：前馈模型虽然摆脱 BA，也可能把传统 SfM 的坐标系偏置偷偷写进网络。

### 8.3 MapAnything：先验可插拔与 metric scale

MapAnything 将图像与可选内参、位姿、深度、部分重建统一 token 化，输出 factored metric representation（深度、局部射线、位姿、单一 metric scale 因子）。它代表前馈几何从“无先验模型”走向“任意先验都可利用”的工程阶段。

---

## 9. 第八阶段：DA3 用最小 depth-ray 统一三代问题

### 9.1 为什么 V3 不再只是“更好的深度”

[[DA3 - 任意视角恢复视觉空间]] 把任务定义成：输入单图、多视图或视频，可选已知相机，输出空间一致的几何。它用 vanilla DINOv2，前 $L_s$ 层单图注意力、后 $L_g$ 层交替跨视图注意力，默认 $L_s:L_g=2:1$。

### 9.2 Depth-ray 是什么

逐像素 ray 为 $r=(t,d)$，其中 $t$ 是相机中心，$d=RK^{-1}p$ 是未归一化射线方向：

$$P=t+D(u,v)d$$

Depth 表示沿射线的场景结构，ray 同时编码相机位置、旋转与内参。相机可从 ray map 经 DLT+RQ 恢复，也可由轻量 camera head 直接回归。

![](assets/从%20Depth%20Anything%20到%20DA3：前馈深度与三维几何%20-%20Survey/figures/04-da3-depth-ray.png)

> 图 4：DA3 用单个预训练 DINOv2 + input-adaptive attention + Dual-DPT 输出 depth/ray；已知相机作为 camera token 注入。来源：DA3 Figure 2。

### 9.3 DA3 如何吸收 V1/V2 经验

- 延续 V1/V2 的 DINOv2 + DPT 极简骨架；
- 继续用 synthetic-only monocular teacher 获得细边界；
- 将 teacher 相对深度与真实稀疏 metric depth 通过 RANSAC scale-shift 对齐；
- 用任意视图训练把“单图泛化”升级为“跨视图一致”。

### 9.4 结果与意义

论文当前版本报告相对 VGGT 平均位姿精度 +35.7%、几何 +23.6%，20 个 setting 拿 18 个 SOTA。DA3-Large 约 0.36B 总参数（0.30B backbone），显著小于 VGGT 1.2B；FF-NVS 也随几何 backbone 提升。

但消融并非所有指标单调：teacher 在 DTU 可能略降数值、pose conditioning 在 7Scenes 下降、DA3 主模型在 NYU/Sintel 单目深度不总胜 DA2。更准确的结论是：**最小目标提升了整体跨任务效率与泛化，不是每个数据集、每个指标都严格支配。**

---

## 10. 技术时间线：谁解决了什么

| 时间 | 工作 | 关键解决 | 暴露的新瓶颈 |
|---|---|---|---|
| 2023.02 | ZoeDepth | relative 预训练迁移到 metric bins | 域内模型、相机变化 |
| 2023.07 | Metric3D | canonical camera metric depth | 依赖内参 |
| 2023.12 | DUSt3R | 图像对直接回归 shared-frame pointmap | pairwise 与全局对齐 |
| 2024.01 | Depth Anything V1 | 62M 无标签数据扩展开放域相对深度 | 真实标签粗糙、边界不足 |
| 2024.03 | UniDepth | 相机自提示 + pseudo-spherical metric 3D | 极端 OOD 相机尺度 |
| 2024.04 | Metric3D v2 | 16M 数据、metric depth + normal | 焦距与算力依赖 |
| 2024.06 | Depth Anything V2 | 精确合成 teacher + 真实伪标签 student | 仍是单目 relative depth |
| 2024.06 | MASt3R | 3D matching + pointmap | 仍 pairwise |
| 2024.08 | Spann3R | spatial memory 在线重建 | 顺序依赖与漂移 |
| 2024.10 | Depth Pro | 高分辨率锐利 metric depth + 焦距 | 单值深度表示边界 |
| 2024.10 | MoGe | affine pointmap + optimal supervision | 无绝对尺度、单图可见面 |
| 2024.10 | MonST3R | 动态视频几何 | 仍需 flow/窗口优化 |
| 2025.01 | Fast3R | 1000+ 图 all-to-all 单次前馈 | 显存与分布式门槛 |
| 2025.03 | VGGT | camera/depth/point/track 统一前馈 | 参考帧偏置、模型大 |
| 2025.07 | π³ | 置换等变、reference-free | metric scale 与长序列 |
| 2025.09 | MapAnything | 任意几何先验 + metric factorization | 先验质量与复杂输入 |
| 2025.11 | DA3 | plain ViT + depth-ray 统一任意视图 | 动态、非针孔、训练成本 |

---

## 11. 横向选择表

| 方法 | 输入 | 相机输入 | 输出尺度 | 多视图一致 | 主要后处理 | 最适合 |
|---|---|---|---|---|---|---|
| DA V2 | 单图 | 不需要 | Relative | 否 | 可选 metric 微调 | 快速开放域深度、控制条件 |
| Metric3D v2 | 单图 | **需要焦距** | Metric | 否 | de-canonical | 测量、SLAM、法线 |
| UniDepth | 单图 | 可选 | Metric | 否 | 反投影 | EXIF 缺失的通用 metric depth |
| Depth Pro | 单图 | 不需要 | Metric | 否 | 焦距头组合 | 高分辨率边界、NVS 预处理 |
| MoGe | 单图 | 不需要 | Scale+shift 不定 | 否 | FOV/shift solver | 单图可见表面 3D |
| DUSt3R | 图像对/集合 | 不需要 | Relative 3D | pairwise | 全局对齐 | 两视图几何、研究基线 |
| MASt3R | 图像对/集合 | 不需要 | Relative/metric 3D | pairwise | matching + alignment | 匹配、定位 |
| Spann3R | 有序序列 | 不需要 | 全局 pointmap | 状态式 | memory 管理 | 在线静态重建 |
| Fast3R | 大规模图集 | 不需要 | global/local pointmap | 是 | 可选 BA | 上千视图离线重建 |
| VGGT | 1–数百视图 | 不需要 | 相对/归一化 | 是 | 可选 BA | 通用多视图几何 backbone |
| π³ | 多视图/视频 | 不需要 | 相对几何 | 是且置换等变 | 少量对齐 | 输入顺序不稳定的图集 |
| MapAnything | 任意视图 + 可选几何先验 | 可选 | Metric | 是 | factored fusion | 有部分标定/深度的工业数据 |
| DA3 | 1–数千视图 | 可选 | 相对/metric 变体 | 是 | DLT/RQ 或 camera head | 通用几何、下游 FF-NVS |

---

## 12. 当前正在攻坚的问题

### 12.1 动态与非刚体

VGGT、DA3 的主要强项仍是静态几何。MonST3R、CUT3R、D4RT、Dyna3 等分别从光流、状态、查询或 VLM mask 切入，但“相机运动 vs 物体运动”的可辨识性仍没有统一表示。DA3 结论明确把动态场景列为未来方向。

### 12.2 非针孔相机

Metric3D、MoGe、VGGT、DA3 大多围绕针孔模型。鱼眼、全景、滚动快门和复杂畸变会让 ray 参数化、FOV 恢复和 DLT 假设失效。工业部署中，先做去畸变仍是常见前置步骤。

### 12.3 长序列扩展

- all-to-all：全局一致但显存随视图数快速增长；
- recurrent/memory：可在线但误差因果累积；
- pairwise：局部稳但全局对齐重。

目前没有同时满足“无限长度、随机访问、无漂移、低显存”的方案。更可能的方向是层级记忆 + 局部全注意力 + 回环检索的混合系统。

### 12.4 Metric scale 的来源

单图绝对尺度仍依赖：

- 相机焦距；
- 训练数据的物体/场景尺度先验；
- 多视图运动；
- 已知深度或部分重建。

未来模型会更明确地输出尺度置信度，并利用可选传感器/先验，而不是宣称纯 RGB 可以无条件恢复真实尺度。

### 12.5 标签与评测闭环

前馈模型的训练标签大量来自经典 SfM/MVS、LiDAR 或合成引擎；旧优化管线没有消失，而是变成 teacher 和数据生产器。模型再用自身伪标签扩大数据，可能形成系统偏差闭环。独立、真实、高分辨率的几何评测仍稀缺。

### 12.6 训练成本与开放性

V1/V2 的 62M 伪标签、VGGT 的 64×A100×9 天、DA3-Giant 的 128×H100×10 天，都意味着“极简架构”不等于低训练成本。公开数据、开源权重和可审计数据管线将决定复现可信度。

---

## 13. 已确立事实与我的判断

### 13.1 已确立事实

1. DINOv2 + DPT 已成为单目与多视图密集几何的稳定骨架；
2. 数据覆盖、标签质量和监督表示的影响已超过多数局部架构技巧；
3. 直接 pointmap/depth-ray 回归能替代传统管线中的大量显式匹配与优化；
4. 多视图前馈模型已能在若干 benchmark 上超过含 BA 的旧方法，并显著更快；
5. 传统平均深度指标无法充分评价透明面、细边界和 3D 形状。

### 13.2 我的判断

1. **单目深度与前馈 3D 会继续合流。** Depth 只是沿相机射线的 pointmap；ray/camera 是把它变成空间模型所缺的另一半。
2. **下一阶段的核心不是更多输出头，而是条件接口。** 相机、稀疏深度、部分点云、语言和交互信息应以可选 prompt 注入，同一模型在无先验与有先验之间连续退化。
3. **严格前馈与优化不会二选一。** 网络负责秒级初始化与困难域鲁棒性，几何优化负责极高精度和全局闭环；最实用系统会保留可选 refinement。
4. **不确定性会成为一等输出。** 当模型被用于机器人、测量或数据引擎，仅给几何值而不给可信度已经不够。
5. **动态 4D 是真正的下一主战场。** 静态几何基础模型已较成熟，但动态对象身份、跨帧对应、长期记忆和生成内容的一致性仍分散在不同方法中。

---

## 14. 阅读路径

### 14.1 先读 Depth Anything 三代

1. [[Depth Anything - 大规模无标注数据驱动的单目深度]]：数据规模与伪标签学习；
2. [[Depth Anything V2 - 精确合成监督与真实伪标签桥接]]：标签质量与合成—真实分工；
3. [[DA3 - 任意视角恢复视觉空间]]：depth-ray 与任意视图统一。

### 14.2 再读单目 metric / geometry 分支

- [[UniDepth - 自提示相机的通用度量深度]]
- [[Metric3D v2 - 规范相机空间的度量几何基础模型]]
- [[Depth Pro - 高分辨率锐利度量深度]]
- [[MoGe - 开放域单目几何与仿射不变点图]]

### 14.3 最后读多视图前馈主线

- [[DUSt3R - 前馈式几何重建奠基]]
- [[MASt3R - 匹配与重建统一]]
- [[Spann3R - 空间记忆增量式重建]]
- [[MonST3R - 运动场景几何估计]]
- [[Fast3R - 千图单次前馈重建]]
- [[VGGT - 前馈式通用三维重建]]
- [[π³ - 置换等变视觉几何学习]]

如果只想快速建立全局地图，再配合已有的 [[前馈式三维重建 - 分支谱系总览]] 阅读：本文按“问题演化”组织，后者按“输出表征/分支谱系”组织，两篇互补。

---

## 15. 材料来源

见 [sources.md](assets/从%20Depth%20Anything%20到%20DA3：前馈深度与三维几何%20-%20Survey/sources.md)。
