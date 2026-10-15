---
type: paper-note
status: complete
domain: 3D-Reconstruction
paper: "Depth Anything 3: Recovering the Visual Space from Any Views"
year: 2025
arxiv: "2511.10647"
source: https://arxiv.org/abs/2511.10647
project: https://depth-anything-3.github.io/
code: https://github.com/ByteDance-Seed/depth-anything-3
tags:
  - feed-forward-3D
  - multi-view-geometry
  - depth-ray
  - geometric-foundation-model
created: 2026-08-26
updated: 2026-10-15
---

# 3. Depth Anything 3：从任意视图恢复视觉空间

**一句话结论：** Depth Anything 3（DA3）用"一个 vanilla DINOv2 骨干 + 一个 depth-ray 预测目标"这套**最小化建模**，把从单图、多视图到视频的任意数量视觉输入统一到"恢复视觉空间"这一个目标下，且在相机位姿、几何重建、视觉渲染三个维度全面超越前 SOTA VGGT（位姿精度平均 +35.7%、几何精度平均 +23.6%），同时单目深度还反超 Depth Anything 2。最有冲击力的一点是：DA3-Large 仅约 **0.30B** 参数，比 VGGT 的 1.19B 小 3 倍，却在 20 个评测 setting 中拿下 18 个 SOTA、其中一半以小打大反超 VGGT——它用"做减法"证明前馈重建不需要复杂架构和冗余多任务。

|         |                                                                                                                                                                                                                                                      |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 论文名称    | **Depth Anything 3: Recovering the Visual Space from Any Views**                                                                                                                                                                                      |
| 归档领域    | **三维重建**                                                                                                                                                                                                                                             |
| 重要性评估   | **★★★★★（5/5）**。理由：以最小化建模在前馈重建上全面反超 VGGT（含以小打大），并首次给出系统的视觉几何评测基准与 FF-NVS 应用验证，是"前馈重建"路线承上启下的关键节点。                                                                                                                                          |
| 作者 / 机构 | Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, Bingyi Kang（Project Lead）；**ByteDance Seed**                                                                                                               |
| 官方出处    | [arXiv 2511.10647](https://arxiv.org/abs/2511.10647)；[官方项目页](https://depth-anything-3.github.io/)                                                                                                                                                 |
| 数据来源    | 官方 PDF 全文精读（`assets/DA3 - 任意视角恢复视觉空间/sources/paper.pdf`）                                                                                                                                                                                  |

---

## 1. 研究背景

### 1.1 从"任务专用"到"最小建模"

单目深度估计、SfM、多视图立体（MVS）、SLAM 这些 3D 视觉任务，概念上高度重叠——常常只差"输入视角数量"这一个因素，但业界长期为每个任务训练专用模型。DUSt3R 是转折点：它用 Transformer 在两视图间直接回归点图，纯前馈地同时算出深度和相对位姿。后续方法（含 VGGT）沿此扩展，但普遍依赖复杂定制的架构 + 多任务联合训练 + 从头训练，**难以利用大规模预训练模型**。

DA3 的立场是"往后退一步"：回到人类空间智能的根本目标——**从任意视觉输入恢复 3D 结构**，并用"最小化建模"回答两个问题：

1. 是否真的需要一堆预测目标，还是存在一个**最小充分的目标集**？
2. 一个**普通（plain）Transformer** 是否就够，而不需要专门架构？

论文对两者都给出肯定回答。

### 1.2 与 VGGT 的差异

VGGT 用了 DINO 骨干 + 多个任务头（相机 / 点图 / 深度 / 跟踪）联合训练，参数 1.19B。DA3 反其道而行：**只保留一个深度-射线（depth-ray）预测目标**，靠它同时支撑结构重建和相机运动恢复，并用更小的模型反超 VGGT。

![图1](assets/DA3%20-%20任意视角恢复视觉空间/figures/fig1_teaser.png)

*图：Figure 1（teaser）— DA3 的核心主张：无论输入是单张图、多视图还是视频（有无相机位姿），都用同一个最小化模型"恢复视觉空间"，并统一支撑深度估计、相机位姿、几何重建与视觉渲染。*

## 2. 核心方法

### 2.1 任务形式化与 Depth-Ray 表示

输入 $I = \{I_i\}_{i=1}^{N_v}$（$N_v=1$ 即单目，$N_v>1$ 即多视图/视频），每张图有深度 $D_i$、相机外参 $[R_i | t_i]$、内参 $K_i$。像素 $p=(u,v,1)^\top$ 通过

$$
P = R_i \big( D_i(u,v)\,K_i^{-1}p \big) + t_i
$$

投影到世界坐标 3D 点 $P$。

关键设计：不直接回归旋转矩阵（正交约束难学），而是用一张与输入逐像素对齐的 ray map $M \in \mathbb{R}^{H\times W\times 6}$ 隐式编码相机位姿：

- 前 3 通道存 ray origin $t$；
- 后 3 通道存 ray direction $d = RK^{-1}p$（不归一化，保留投影尺度）。

于是世界坐标 3D 点可简单地写成

$$
P = t + D(u,v)\cdot d
$$

把预测的 depth map 和 ray map 做逐元素运算，就能生成全局一致的点云。这就是"最小目标集"的核心。

### 2.2 从 Ray Map 反解相机参数

有了 ray map 后，相机参数可用闭式方法恢复：

1. **相机中心**：对所有像素的 ray origin 求平均 $t_c = \frac{1}{HW}\sum M(h,w,:3)$。
2. **旋转 + 内参**：把问题转成求解 homography $H = KR$——在"单位内参相机"下 $d_I = p$，目标相机下 $d_{cam} = KR\,d_I$，用 **DLT 算法**最小化几何误差得到 $H^*$，再经 **RQ 分解**唯一分解出 $K$ 与 $R$。

但推理时从 ray map 反解相机成本较高，所以额外加一个**轻量 camera head** $D_C$：它只处理每视图 1 个 camera token，直接预测视场角 $f\in\mathbb{R}^2$、四元数旋转 $q\in\mathbb{R}^4$、平移 $t\in\mathbb{R}^3$，几乎零额外开销。

### 2.3 架构：单 Transformer + Dual-DPT head

![图2](assets/DA3%20-%20任意视角恢复视觉空间/figures/fig2_pipeline.png)

*图：Figure 2 — DA3 Pipeline。DA3 采用一个未做任何架构改动的 vanilla DINOv2 模型；通过 input-adaptive 的跨视图自注意力实现跨视图推理；Dual-DPT head 从视觉 token 预测 depth 与 ray map；若相机参数可用则编码为 camera token 与 patch token 拼接，参与全部注意力运算。*

三个组件：

| 组件 | 设计 |
| --- | --- |
| **单 Transformer 骨干** | vanilla DINOv2（L 层），不改结构；跨视图推理靠"重排 token"实现 |
| **input-adaptive 跨视图注意力** | 前 $L_s$ 层做单图内自注意力，后 $L_g$ 层在跨视图 / 单图内注意力之间交替（tensor reordering）；实践中 $L_s : L_g = 2:1$ |
| **Dual-DPT head** | 共享 reassembly 模块 + 两套独立 fusion 层（depth 分支 / ray 分支）+ 两个输出层 |

**相机条件注入**：每个视图前置一个 camera token $c_i$。有相机参数时 $c_i = E_c(f_i,q_i,t_i)$（轻量 MLP），否则用共享可学习 token $c_l$。这样同一个模型能无缝处理"有 / 无位姿"两种输入——单图输入时自然退化为单目深度估计，零额外成本。

### 2.4 训练：Teacher-Student 范式

真实数据的深度标签常带噪声、不完整，直接监督有害。DA3 采用伪标签策略：

1. **训练 Teacher**：一个只在**纯合成数据**上训练的单目相对深度模型（DINOv2 + DPT，无特殊结构），在室内 / 室外 / 物体级 / in-the-wild 大规模合成语料上训出精细几何。
2. **生成伪深度**：Teacher 为所有真实数据生成稠密高质量相对深度。
3. **RANSAC 对齐**：用 RANSAC least squares 估计 scale $s$ 和 shift $t$，把稠密伪深度对齐到原始稀疏 / 噪声 GT 上，兼顾"细节完整"与"几何准确"。

**训练目标**（所有 GT 先除以"有效重投影点图的平均 $\ell_2$ 范数"归一化到一致尺度）：

$$
L = L_D(\hat D, D) + L_M(\hat R, M) + L_P(\hat D \odot d + t, P) + \beta L_C(\hat c, v) + \alpha L_{grad}(\hat D, D)
$$

- $L_D$：深度损失（尺度不变，带 aleatoric uncertainty）；
- $L_M$：ray map 损失；
- $L_P$：点图损失；
- $L_C$：相机损失（权重 $\beta$）；
- $L_{grad}$：深度梯度损失（权重 $\alpha$）。

## 3. 实验与结果

### 3.1 视觉几何基准（自建 benchmark）

DA3 同时贡献了一套**视觉几何评测基准**：覆盖 5 个数据集、89+ 场景（物体级 → 室内 → 室外），直接评测位姿精度 + 重建精度 + 视觉渲染质量，忠实度量一个几何估计器的真实能力。

**核心结果：**

| 对比 | 结果 |
| --- | --- |
| 位姿精度（vs VGGT） | 平均 **+35.7%** |
| 几何精度（vs VGGT） | 平均 **+23.6%** |
| 总 SOTA 数 | **18 / 20** 个 setting |

![图6](assets/DA3%20-%20任意视角恢复视觉空间/figures/fig6_pointcloud.png)

*图：Figure 6 — 点云重建定性对比。DA3 融合 pose + depth 得到的点云，在几何畸变、结构真实感上明显优于 VGGT / π³ 等前 SOTA，体现了 depth-ray 表示带来的更强全局一致性。*

### 3.2 模型规模：以小打大

这是 DA3 最反直觉的结论——**DA3-Large 仅约 0.30B 参数**，比 VGGT 的 1.19B 小约 3 倍，却：

- 在 10 个 setting 中 **5 个直接反超** VGGT；
- 加上单目深度超 DA2 的能力，说明"最小建模"不仅没损失精度，反而更高效、更可 scale。

### 3.3 单目深度估计

在标准单目基准（KITTI / NYU / Sintel / ETH3D / DIODE 等）上，DA3 的**单目变体（Depth-Anything-3-Monocular）** 超过 Depth Anything 2，同时保持其细节与鲁棒性。关键是：DA3 的主干和 Teacher 完全对齐（都是 DINOv2 + DPT），没有为单目单独设计架构。

### 3.4 下游应用：前馈 3DGS 新视角合成

论文用 FF-NVS 验证 DA3 作为几何基座的能力，提出两种方式：

- **Pose-Conditioned FF-3DGS**：在 DA3 骨干上微调一个 GS-DPT head，预测逐像素的 3D Gaussian 参数（不透明度、四元数、尺度、颜色）。
- **Pose-Adaptive FF-3DGS**：同一套预训练权重，同时支持"有 / 无位姿、任意分辨率、任意视图数"的 in-the-wild 新视角合成。

两个关键发现：① 微调几何基座做 NVS，显著优于高度专用的任务模型；② **几何重建能力越强，FF-NVS 越好**——直接印证"DA3 是最优 NVS 骨干"。

### 3.5 在 Lyra 2.0 中的真实角色：复用 FF-3DGS 支线，而非照搬完整 DA3

这里必须区分**DA3 原论文能力**与**Lyra 2.0 的二次适配**。

- **DA3 原论文主线**是通用视觉几何：任意数量、已知或未知相机的图像输入，经 DINOv2 + Dual-DPT 输出 depth、ray/camera 与全局一致 3D；其 FF-3DGS 只是验证几何 backbone 可迁移性的下游支线，通过 GS-DPT head 让每个输出位置预测 Gaussian 的 opacity、rotation、scale、color 等属性。
- **Lyra 2.0 的生成主线**仍是 Wan2.1-14B 长视频模型。DA3 在生成过程中只作为逐帧深度估计器，服务空间检索、对应和最近帧 depth warp；它不负责扩散生成，也没有把 DA3 的完整 depth-ray/camera-recovery pipeline 直接嵌进 Wan。
- **Lyra 2.0 的资产化阶段**才真正复用 DA3 的前馈 per-pixel 3DGS pipeline / GS-DPT head：把已生成的长视频与已知相机轨迹一次前馈转换为 Gaussians，再进一步提取 mesh。这里需要的是“从视频像素解码 Gaussian”的能力，而不是再次求解一套未知相机下的任意视图几何。

Lyra 2.0 又做了两项原论文之外的定向改造：

1. **$k=2$ 空间降采样**：GS-DPT 不再按全分辨率每像素产生一个 Gaussian，而在每个空间维度下采样 2 倍，使 Gaussian 数量降为原来的 $1/4$，直接控制长视频资产的显存、存储和渲染成本。
2. **生成域微调**：真实多视图训练出来的 DA3 对生成视频中的轻微跨视图不一致、纹理漂移和局部伪影并不天然鲁棒。Lyra 2.0 从 DL3DV 图像/轨迹出发，自回归生成 **3,000 段约一分钟视频**，再对该 3DGS pipeline 微调 10K iterations（lr $5\times10^{-5}$，batch size 8），让重建器适应它实际要消费的“模型生成域”。

因此，准确口径不是“Lyra 2.0 使用完整 DA3 重建世界”，而是：**Lyra 2.0 把 DA3 当作几何工具箱——在线阶段取其单帧深度，离线/末端资产化阶段取其 FF-3DGS + GS-DPT，并针对长生成视频重新设计输出密度与训练分布。**

### 3.6 与 VGGT、FantasyWorld 的关系

| 系统 | 几何在系统中的位置 | 与 DA3 的关系 |
|---|---|---|
| **VGGT** | 独立前馈几何基础模型；多任务头联合预测 camera、depth、point map、track | DA3 的直接基线与反题。VGGT 依赖过完备多任务输出和约 1.19B 参数；DA3 用 depth-ray 最小目标与约 0.30B Large 证明更小、更简的几何 backbone 也能更强。|
| **DA3 原论文** | 先训练通用几何 backbone，再把它微调成 FF-NVS / per-pixel 3DGS | 证明“几何 backbone 排名可迁移为 NVS 排名”，并提供 Lyra 2.0 后续改造的 GS-DPT 起点。|
| **FantasyWorld** | 在冻结 Wan 视频主干旁加入 **VGGT-style geometry branch**，通过双向 cross-attention 联合生成视频与 depth/point/camera | 它不是 DA3 的下游：目标是把几何监督内生到固定长度视频生成；3DGS 评测仍需后处理，且以 VGGT 初始化最好。路线是 **generation–geometry joint modeling**。|
| **Lyra 2.0** | 长视频先生成，几何用于在线路由；最后用生成域适配的 DA3 FF-3DGS 资产化 | 路线是 **generation–reconstruction decoupling**。它没有复制 DA3 全部任务，而是选择 depth 与 GS-DPT 两个接口，并为长视频做 $k=2$ 与 3,000 段生成数据微调。|

这四者形成两条不同主线：VGGT → DA3 讨论“怎样做更好的通用前馈几何”；FantasyWorld 与 Lyra 2.0 讨论“几何怎样进入生成系统”。前者把 VGGT-style 几何分支放进视频模型共同训练，后者让几何退出像素生成主循环、只做路由，并在末端用 DA3-derived GS-DPT 落成可渲染资产。

### 3.7 消融（关键结论）

- **depth-ray 是最小充分目标**（Tab 6）：优于 point map 或更复杂的多目标组合，既捕捉场景结构又捕捉相机运动，且避免冗余目标引入的"纠缠"。
- **Ls:Lg = 2:1 最优**（Tab 7）：单图内注意力与跨视图注意力的配比存在最优平衡。
- **数据 scaling 有效**：扩大训练语料带来稳定的深度估计提升。

## 4. 局限与评价

### 4.1 局限

- 论文自述"最小建模"仍有边界：深度-ray 表示对**鱼眼 / 全景**等非针孔相机未覆盖（与 VGGT 同）。
- 相机反解依赖 DLT + RQ 分解，虽然闭式高效，但极端视角 / 退化配置下的稳定性仍需观察。
- 所有模型**只用公共学术数据集训练**，与用大规模私有数据的模型（如 VGGT）相比，数据体量仍有上限。

### 4.2 个人评价

DA3 的贡献在"做减法"：它证明前馈重建**不需要** VGGT 式的多任务冗余和专用架构，一个 vanilla 骨干 + 一个 depth-ray 目标就够，且更小、更准、更易 scale。这与 VGGT-Ω 的"做加法（scaling up）"形成有趣的对照——两条路线都在推进同一条主线，但 DA3 暗示"效率"与"简洁"可能是更优的 scaling 维度。

它留下的关键方法论遗产：

1. **depth-ray 表示**：把相机位姿隐式编码进逐像素射线，绕开旋转矩阵的正交约束，是可迁移的核心技巧。
2. **最小充分目标集**：回答了"到底要预测什么"这个被多数工作跳过的根本问题。
3. **几何基座 → 下游**：用 FF-NVS 定量证明"几何重建强"直接转化为"下游任务强"，把"几何基础模型"的定位做实。

### 4.3 复现成本与建议

- 官方已开源，模型相对小（Large 约 0.30B），复现 / 推理门槛显著低于 VGGT。
- **建议关注点**：ray map 的不归一化方向 $d$ 的数值稳定性、DLT 反解相机在极端位姿下的退化、Dual-DPT 的共享 reassembly 设计，是理解这套"最小建模"的关键。

**最终评价：** DA3 是"前馈重建"路线上一篇承上启下的高价值工作。它用最小化建模反超 VGGT，重新校准了这条路线的发展方向——从"堆架构、堆多任务"转向"找最小充分表示 + 用好预训练"。读它，等于读到这条主线从"能不能前馈"走向"怎么更优雅地前馈"的关键一步。

---

## 5. 更完整的定量结果

### 5.1 相机位姿

DA3-Giant 在 HiRoom、ETH3D、DTU、7Scenes、ScanNet++ 的 AUC3/AUC30 分别达到 80.3/95.9、48.4/91.2、94.1/99.4、28.5/86.8、85.0/98.1。除 DTU AUC30 以 0.4 个百分点低于 VGGT 外，其余主要 setting 领先。摘要报告相对 VGGT 平均位姿精度提升 35.7%；项目早期材料曾出现 44.3% 的另一统计口径，引用时应以论文当前版本为准。

### 5.2 Pose-free 几何

| 数据集 | VGGT | DA3-Giant |
|---|---:|---:|
| HiRoom F1↑ | 56.7 | **85.1** |
| ETH3D F1↑ | 57.2 | **79.0** |
| DTU CD/mm↓ | 2.05 | **1.85** |
| 7Scenes F1↑ | 47.9 | **53.5** |
| ScanNet++ F1↑ | 66.4 | **77.0** |

五项全部胜出。已知姿态时 DA3-Giant 在四项领先，但 DTU 上 DA3-Large 的 1.23 mm 反而优于 Giant 1.85 mm，说明参数规模与几何精度并非所有域都单调。

### 5.3 单目与 metric depth

DA3 主模型的单目相对深度并非逐数据集都超过 DA2：NYU 97.4 vs 97.9、Sintel 75.5 vs 77.2；但专门的 mono-student 在 KITTI/NYU/Sintel/ETH3D/DIODE 达 97.1/98.0/82.3/98.8/96.5，全面超过 DA2。Metric student 在 ETH3D $\delta_1=0.917$、AbsRel=0.104，明显强于 UniDepth v2 的 0.863；但 NYUv2/KITTI 仍不如最强专用 metric 方法。

### 5.4 FF-NVS

用同样 12 个 context views、270×480 分辨率，DA3 backbone + GS-DPT 在 DL3DV 达 PSNR 21.33/SSIM 0.711/LPIPS 0.241；Tanks and Temples 为 18.10/0.578/0.311；MegaDepth 为 17.89/0.561/0.351，三组均优于 DepthSplat 和 VGGT backbone。这里最重要的不是又一个 NVS SOTA，而是**几何 backbone 排名与 NVS 排名一致**，支持“更强几何基础模型可直接抬高下游上限”。

---

## 6. 关键消融的反直觉细节

### 6.1 Depth-ray 不是每项都绝对最好

`depth + ray` 在 HiRoom/ETH3D 等总体最稳；加入轻量 camera head 后 DTU 指标更好，但 HiRoom/ETH3D 可能下降。这说明 camera head 的首要作用是工程便利，而不是核心几何能力。论文“最小目标”应理解为 depth-ray 承担主要监督，相机直接回归可保留为低成本辅助输出。

### 6.2 Teacher 监督存在指标—视觉质量权衡

去掉 teacher 后，DTU 数值从 CD 3.82 改善到 3.29，但 HiRoom、7Scenes、ScanNet++ 明显变差；metric depth 中也有“无 teacher 标准指标略好，但边界更糊”的现象。Teacher 的收益更多落在细结构、边缘和跨域泛化，不保证每个传统平均误差都改善。

### 6.3 Pose conditioning 不是全域单调

已知姿态输入在 HiRoom、ETH3D、DTU、ScanNet++ 改善，但 7Scenes F1 从 58.4 降到 46.0。更准确的结论是“多数数据集有益”，不能写成“所有指标一致提升”。

---

## 7. 训练与复现成本

- DA3-Giant：128×H100，200K steps，约 10 天；
- 基础分辨率 504²，并混合 504×378、896×504 等多种宽高比；
- 每步 2–18 视图，动态调 batch 保持 token 量；
- 120K steps 后从原始 GT depth 切换到 teacher 标签；
- pose conditioning 以 0.2 概率启用；
- 推理时 A100 80GB、32 图、504×336：Giant 约 37.6 FPS/图，Large 78.37，Base 126.5，Small 160.5；最大图像数约 1K/1.6K/2.2K/4.1K。

训练复现成本极高，但官方 Small/Base/Large 更适合做下游几何 backbone。实际使用时，应优先验证相机模型、视图数、分辨率和显存上限，而不是只看模型参数量。

---

## 8. 材料来源

见 [sources.md](assets/DA3%20-%20任意视角恢复视觉空间/sources.md)。

