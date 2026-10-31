---
type: paper-note
status: complete
domain: 3D-Reconstruction
series: Depth Anything
series_order: 3
series_hub: "[[从 Depth Anything 到 DA3：前馈深度与三维几何 - Survey]]"
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
updated: 2026-10-31
---

# Depth Anything 3：从任意视图恢复视觉空间

> 系列导航：[[Depth Anything - 大规模无标注数据驱动的单目深度]] → [[Depth Anything V2 - 精确合成监督与真实伪标签桥接]] → **Depth Anything 3** · [[从 Depth Anything 到 DA3：前馈深度与三维几何 - Survey|系列综述]]

## 一句话结论

Depth Anything 3（DA3）观察到单目深度、SfM、多视图重建和视频几何本质相通，既有 VGGT 类方法却用复杂专用架构和相机/点图/深度/跟踪多任务联合训练造成冗余，于是采用“**外部 DINOv2 家族初始化的 plain ViT 骨干 + depth-ray 最小充分目标**”：逐像素预测 depth 与包含 ray direction/origin 的 ray map，用简单运算恢复全局点云并可闭式反解相机，配合后段 input-adaptive 跨视图 attention、Dual-DPT 和合成 teacher 伪标签统一单图、多视图与视频输入。DA3 在 20 个 setting 中拿下 **18 个 SOTA**，相对 VGGT 位姿/几何平均提升 **35.7%/23.6%**；论文按骨干口径称 DA3-Large 约 **0.30B**，当前 `DA3-LARGE-1.1` safetensors 则包含 **410,941,767** 个参数张量（骨干 + heads），两种口径不可混写。它证明前馈重建的关键不是堆任务，而是找对最小表示并复用预训练，但鱼眼/全景相机、极端退化位姿和训练算力仍是边界。

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

关键设计：不直接回归旋转矩阵（正交约束难学），而是用一张与输入逐像素对齐的 ray map $M \in \mathbb{R}^{H\times W\times 6}$ 隐式编码相机位姿。按锁定发布代码 `camray_to_caminfo` 的通道约定：

- 前 3 通道存 ray direction $d = RK^{-1}p$（不归一化，保留投影尺度）；
- 后 3 通道存 ray origin / translation $t$。

论文只需把 $d,t$ 代入几何式即可，但工程复现必须遵守上述 `[d_x,d_y,d_z,t_x,t_y,t_z]` 顺序；反过来会让相机反解完全错误。

于是世界坐标 3D 点可简单地写成

$$
P = t + D(u,v)\cdot d
$$

把预测的 depth map 和 ray map 做逐元素运算，就能生成全局一致的点云。这就是"最小目标集"的核心。

### 2.2 从 Ray Map 反解相机参数

有了 ray map 后，相机参数可用闭式方法恢复：

1. **相机中心**：对所有像素的 ray origin 求平均；按发布代码 `[d,t]` 顺序是 $t_c = \frac{1}{HW}\sum M(h,w,3{:}6)$。
2. **旋转 + 内参**：把问题转成求解 homography $H = KR$——在“单位内参相机”下 $d_I=p$，目标相机下 $d_{cam}=KR\,d_I$。论文以 DLT / 矩阵分解描述；锁定发布实现具体采用 confidence-weighted RANSAC homography，再用 `ql_decomposition` 恢复旋转与下三角内参因子。

但推理时从 ray map 反解相机成本较高，所以额外加一个**轻量 camera head** $D_C$：它只处理每视图 1 个 camera token，直接预测视场角 $f\in\mathbb{R}^2$、四元数旋转 $q\in\mathbb{R}^4$、平移 $t\in\mathbb{R}^3$，几乎零额外开销。

### 2.3 架构：单 Transformer + Dual-DPT head

![图2](assets/DA3%20-%20任意视角恢复视觉空间/figures/fig2_pipeline.png)

*图：Figure 2 — DA3 Pipeline。论文以“plain / vanilla DINO encoder”强调不再堆叠 VGGT 式专用 aggregator；锁定发布代码则更具体：它以 DINOv2 家族 ViT 为外部视觉骨干，在后段 blocks 中通过 token 重排交替执行单视图与跨视图注意力，并启用 QK norm / 2D RoPE。Dual-DPT 从融合 token 预测 depth 与 ray map；已知相机被编码后覆写后段的 CLS/camera slot，而不是额外追加一个 token。*

三个组件：

| 组件 | 设计 |
| --- | --- |
| **外部视觉骨干** | 复用 DINOv2 S/B/L/g 的 patch-14 ViT 架构与预训练起点；官方仓库内 vendored Meta DINOv2 实现，DA3 checkpoint 已打包 backbone 权重，无需推理时再下载独立 DINOv2 |
| **input-adaptive 跨视图注意力** | 保持单一 Transformer stack，但发布实现从 `alt_start` 起交替做单图内 / 跨视图 attention，并同步启用 QK norm 与 2D RoPE；跨视图靠 tensor reordering，不另设 VGGT aggregator |
| **Dual-DPT head** | 四层 token 各自重组为 feature map；depth 与 ray 使用独立 fusion chain，最终输出 depth/conf 与 6D ray/conf |

**相机条件注入**：每个视图复用原 CLS 位置作为 camera slot，不增加序列长度。有相机参数时 $c_i=E_c(f_i,q_i,t_i)$；否则参考视图与其他视图分别使用 learned `ref_token/src_token`。到 `alt_start` 时它们覆写 CLS，随后参与全部后段 local/global attention。单图输入时只剩一个参考 slot，模型自然退化为单目深度估计。

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

这是 DA3 最反直觉的结论——论文按视觉骨干口径将 **DA3-Large 记为约 0.30B 参数**，比 VGGT 的 1.19B 小约 3 倍；当前官方 `DA3-LARGE-1.1` artifact 若把 Dual-DPT 与 camera heads 一并统计，则为 410,941,767 个参数张量。无论采用哪种口径，都必须注明统计边界。它却：

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

- 官方已开源；论文按骨干口径称 Large 约 0.30B，当前 `DA3-LARGE-1.1` 完整 safetensors 为 410,941,767 个参数张量。即便计入 heads，推理门槛仍显著低于 VGGT，但复现记录必须注明参数统计边界。
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

## 8. 源码锁定、模块入口与完整张量链路

### 8.1 固定代码、config 与权重 revision

本节把“论文方法”“官方仓库当前推理实现”“模型 artifact”分开锁定，避免把浮动 `main`、旧 checkpoint 与 1.1 修正版混在一起。

| 对象 | 固定值 |
|---|---|
| 官方仓库 | `ByteDance-Seed/Depth-Anything-3` |
| 源码 commit | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`（2026-07-27） |
| 初始公开架构 commit | `6c58ec93436c903ec72d37ae123951c732e0ca00`（2025-11-14）；四个 main-series YAML 自该提交加入 |
| Large config | `src/depth_anything_3/configs/da3-large.yaml` |
| Giant config | `src/depth_anything_3/configs/da3-giant.yaml` |
| 推荐推理 artifact | `depth-anything/DA3-LARGE-1.1`；HF revision `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c` |
| 最大 main-series artifact | `depth-anything/DA3-GIANT-1.1`；HF revision `72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19` |

官方 README 说明 `-1.1` 是修复训练 bug 后重训的版本；原始 `DA3-LARGE` / `DA3-GIANT` 仍可下载但已不应作为新复现实验默认项。四个可直接锁定的 main-series artifact 为：

| 权重 | HF revision | safetensors 参数数 | 文件大小 | LFS SHA-256 |
|---|---|---:|---:|---|
| `DA3-SMALL` | `e08cab65ca0ec38e7826075418411ab90cab4da3` | 34,299,463 | 137,248,940 B | `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf` |
| `DA3-BASE` | `f4a6c9b3c95e41c82048423d3493a81ec3fa810e` | 135,366,599 | 541,518,028 B | `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5` |
| `DA3-LARGE-1.1` | `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c` | 410,941,767 | 1,643,843,860 B | `739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64` |
| `DA3-GIANT-1.1` | `72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19` | 1,355,674,125 | 5,422,814,644 B | `1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001` |

这里的 safetensors 参数数是**整个发布 artifact**的张量统计，不等于论文只报视觉骨干时的约 0.3B / 1.1B 口径。

### 8.2 模块 → 真实代码入口

| 模块 | 代码入口 | 关键职责 |
|---|---|---|
| 高层加载 / inference | `src/depth_anything_3/api.py::DepthAnything3` | 从 HF revision 加载完整 checkpoint；预处理、forward、对齐与 `Prediction` 转换 |
| 图像预处理 | `utils/io/input_processor.py::InputProcessor` | 保长宽比 resize/crop、统一多图尺寸、14 整除、ImageNet normalize |
| 总控网络 | `model/da3.py::DepthAnything3Net.forward` | camera condition→backbone→Dual-DPT→camera/ray pose→可选 3DGS |
| 外部视觉 backbone 包装 | `model/dinov2/dinov2.py::DinoV2` | 构造 DINOv2 S/B/L/g patch-14 ViT，读取四个中间层 |
| token fusion / Transformer | `model/dinov2/vision_transformer.py::DinoVisionTransformer` | patch/CLS/absolute pos；后段 local/global attention、QK norm、RoPE、local/global feature concat |
| depth / ray geometry heads | `model/dualdpt.py::DualDPT` | 四层 token reassembly；独立 depth 与 ray fusion chains；输出值与 confidence |
| 已知相机编码 | `model/cam_enc.py::CameraEnc` | `[t,q,fov]∈R^9` 经 MLP+4 blocks 变成每视图 camera token |
| 快速相机解码 | `model/cam_dec.py::CameraDec` | 从每视图 camera token 回归 `t(3)+q(4)+fov(2)` |
| ray→camera | `utils/ray_utils.py::get_extrinsic_from_camray` | confidence-weighted RANSAC homography、QL 分解、方向/原点恢复相机 |
| depth→points | `utils/geometry.py::unproject_depth`、`utils/export/glb.py::_depths_to_world_points_with_colors` | 用 $K^{-1}$ 与 $T_{c2w}$ 把深度反投影为世界点 |
| 输出字段 | `utils/io/output_processor.py::OutputProcessor` | 去掉 batch=1 维并转为 `Prediction`；默认公开 depth/conf/camera，不直接公开 point map |

### 8.3 外部视觉 backbone 到底是什么

DA3 的“plain DINO”不能理解成“直接调用一个完全冻结、逐行不改的 `facebookresearch/dinov2` Hub 模型”。准确分三层：

1. **来源**：patch embed、CLS、learned absolute position、ViT blocks 和 S/B/L/g 规模来自外部 DINOv2 家族；vendored 文件保留 Meta copyright。DA3 训练以视觉预训练为起点，而非从随机视觉 tokenizer 开始。
2. **DA3 适配**：发布实现仍是一条 ViT stack，但从 `alt_start` 起交替 local/global attention，并同时启用 QK norm 与 2D RoPE；这正是多视图 token fusion 的位置。
3. **发布部署**：HF 的 `model.safetensors` 已包含 DA3 训练后的 backbone + heads；加载 DA3 checkpoint 时不需要再单独下载 DINOv2 权重，也不能把独立 DINOv2 checkpoint 直接覆盖回去。

四个 main-series config 的 backbone 形状如下。令 backbone 宽度为 $D$，attention heads 为 $A$；由于 `cat_token=true`，几何 heads 实际接收 $C=2D$。

| 模型 | blocks / $D$ / $A$ | `out_layers` | `alt_start=qknorm_start=rope_start` | 几何 head 输入 $C$ |
|---|---|---|---:|---:|
| Small | 12 / 384 / 6 | `[5,7,9,11]` | 4 | 768 |
| Base | 12 / 768 / 12 | `[5,7,9,11]` | 4 | 1536 |
| Large | 24 / 1024 / 16 | `[11,15,19,23]` | 8 | 2048 |
| Giant | 40 / 1536 / 24 | `[19,27,33,39]` | 13 | 3072 |

所有版本都是 patch size 14、一个 CLS slot、`num_register_tokens=0`。Giant 使用 SwiGLU；S/B/L 使用 MLP FFN。它们仍保留 DINOv2 的 learned absolute position grid（以 `img_size=518` 构造为 $37\times37$），再按实际输入 grid 做 bicubic interpolation；后段 RoPE 是额外位置机制，不是替代该 absolute embedding。

### 8.4 动态尺寸公式与预处理

记输入视图数为 $V$，模型 batch 为 $B$，预处理后尺寸为 $H\times W$。官方 `InputProcessor` 会先保持长宽比 resize，再把每一维 resize/crop 到 14 的倍数；若同批视图尺寸不同，还会 center-crop 到共同最小尺寸并同步修正主点。因此：

$$
h=H/14,\qquad w=W/14,\qquad P=hw,\qquad T=1+P.
$$

输入与相机可选字段为：

- normalized RGB：$x\in\mathbb{R}^{B\times V\times3\times H\times W}$；
- 可选外参：$E\in\mathbb{R}^{B\times V\times4\times4}$ 或兼容的 $3\times4$；
- 可选内参：$K\in\mathbb{R}^{B\times V\times3\times3}$；
- ImageNet normalize：mean `(0.485,0.456,0.406)`、std `(0.229,0.224,0.225)`；
- 高层 API 默认 `process_res=504`、`upper_bound_resize`，不是固定方形输入。

### 8.5 backbone：单图 / 多图的完整 token 流

1. **展平视图**：$[B,V,3,H,W]\to[BV,3,H,W]$。
2. **PatchEmbed**：`Conv2d(k=14,s=14)` 得 $[BV,D,h,w]$，flatten 为 $[BV,P,D]$，再按视图恢复为 $[B,V,P,D]$。
3. **CLS 与 absolute position**：每视图前置一个 CLS，得到 $[B,V,T,D]$；训练于 518 构造的 $37\times37$ patch position 被插值到 $h\times w$。main-series 不使用 register tokens，也不在 DA3 inference 中使用 mask token。
4. **前段 local blocks**：每个视图独立 self-attention。计算时重排为 $[BV,T,D]$；Q/K/V 为 $[BV,A,T,D/A]$。这一段保留单图外观与局部几何，不让不同视图过早混合。
5. **camera slot**：到 `alt_start`，代码**覆写原 CLS slot**，而不是 append token。未知相机时，第一个参考视图使用 learned `ref_token`，其余使用 learned `src_token`；已知相机时，`CameraEnc` 把每视图 9D pose 编码为 $[B,V,D]$ 后写入同一 slot。
6. **后段交替 fusion**：local block 仍处理 $[BV,T,D]$；global block 将全部视图重排为 $[B,VT,D]$，Q/K/V 为 $[B,A,VT,D/A]$，因此任意视图的 patch/camera slot 可直接交换信息。block 后再恢复 $[B,V,T,D]$。
7. **QK norm + RoPE**：从 config 指定层起启用。local attention 使用每视图实际二维坐标；global attention 让不同视图共享同一 patch-grid 坐标模板，以避免把“第几个视图”误当作二维图像位移。
8. **四层读取与双流拼接**：在 `out_layers` 处，把最近一次 local 表示与当前融合表示沿 channel 拼接；去掉 camera/CLS slot 后，每层 patch feature 为 $[B,V,P,2D]$，对应 camera feature 为 $[B,V,2D]$。

当 $V=1$ 时，global 序列长度退化为 $T$，所以同一模型自然变成单图编码器；它仍会跑同一组 blocks，但不存在跨视图证据。当 $V>1$ 时，global 序列长度为 $VT$，attention 主矩阵按 $O((VT)^2)$ 增长，这是视图数显存开销的主要来源。

### 8.6 Dual-DPT：四层 token 到 depth / ray

对四个 $[B,V,P,C]$ 特征（$C=2D$），`DualDPT` 先合并 $B,V$：

$$
[B,V,P,C]\to[BV,P,C]\to[BV,C,h,w].
$$

每层经 `LayerNorm + 1×1 projection` 后，四级 resize 分别相对 patch grid 做 $\times4,\times2,\times1,\div2$，再进入两条**参数独立**的 top-down fusion chain：

- depth 主分支：最后两通道 logits 分成 1 个 depth value 与 1 个 confidence；depth 经过 `exp`，confidence 经过 `exp(x)+1`；
- ray 辅分支：最后 7 通道分成 6D ray 与 1 个 confidence；ray 为 linear 输出，confidence 同样用 `exp(x)+1`。

融合结果最终双线性插值到 $H\times W$，因此不是停留在 patch grid：

| raw 字段 | shape | 含义 |
|---|---|---|
| `depth` | $[B,V,H,W]$ | 每像素正深度 |
| `depth_conf` | $[B,V,H,W]$ | depth 置信度，严格大于 1 |
| `ray` | $[B,V,H,W,6]$ | `[direction(3), origin(3)]` |
| `ray_conf` | $[B,V,H,W]$ | ray 置信度，严格大于 1 |

`DualDPT` 的“Dual”不是共享同一 fusion 后只分两个最后卷积；代码拥有独立的 `refinenet*_aux`，只有输入的四层 backbone features 共享。

### 8.7 具体全链路张量表：DA3-Large，4×378×504

取 $B=1,V=4,H=378,W=504$，则 $h=27,w=36,P=972,T=973$；Large 有 $D=1024,A=16,D/A=64,C=2048$。

| 阶段 | 动态 shape | 具体 shape |
|---|---|---|
| normalized images | $[B,V,3,H,W]$ | `[1,4,3,378,504]` |
| view-flattened images | $[BV,3,H,W]$ | `[4,3,378,504]` |
| patch tokens | $[B,V,P,D]$ | `[1,4,972,1024]` |
| CLS/camera + patches | $[B,V,T,D]$ | `[1,4,973,1024]` |
| local-attention Q/K/V | $[BV,A,T,D/A]$ | `[4,16,973,64]` |
| global-attention Q/K/V | $[B,A,VT,D/A]$ | `[1,16,3892,64]` |
| 每个 selected layer patch feature | $[B,V,P,2D]$ | `[1,4,972,2048]` |
| 每个 selected layer camera feature | $[B,V,2D]$ | `[1,4,2048]` |
| DPT reassembly | $[BV,2D,h,w]$ | `[4,2048,27,36]` |
| depth / depth conf | $[B,V,H,W]$ | `[1,4,378,504]` |
| raw ray / ray conf | $[B,V,H,W,6]$ / $[B,V,H,W]$ | `[1,4,378,504,6]` / `[1,4,378,504]` |
| camera pose encoding | $[B,V,9]$ | `[1,4,9]` |
| `extrinsics` / `intrinsics` | $[B,V,3,4]$ / $[B,V,3,3]$ | `[1,4,3,4]` / `[1,4,3,3]` |
| derived world point map | $[B,V,H,W,3]$ | `[1,4,378,504,3]` |
| 高层 `Prediction.depth/conf` | 去掉 $B=1$ | `[4,378,504]` |

### 8.8 camera 分支、ray 分支与 point 输出的边界

#### 默认快速路径：camera head

最后一层 camera feature $[B,V,2D]$ 经两层 MLP，输出：

$$
[t_x,t_y,t_z,q_x,q_y,q_z,q_w,\mathrm{fov}_h,\mathrm{fov}_w]
\in\mathbb{R}^{9}.
$$

再转成 `extrinsics:[B,V,3,4]`（公开 API / README 口径为 OpenCV/COLMAP world-to-camera）与 `intrinsics:[B,V,3,3]`。这条路径会从返回字典删除 raw `ray` / `ray_conf`，因为 camera head 已提供更快的相机输出。

#### `use_ray_pose=true`：由 ray 闭式恢复相机

代码用 ray 前 3 通道与单位相机平面做 confidence-weighted homography RANSAC，再经 QL 分解得到旋转、焦距和主点；后 3 通道加权平均得到 translation。随后同样返回 $[B,V,3,4]$ 与 $[B,V,3,3]$，并删除 raw ray 字段。官方 README 将其描述为“略慢但通常更准”。

#### point map：几何可导出，但不是默认网络字段

发布 `Prediction` 默认包含 depth/conf/camera，不含名为 `points` 的 tensor。点图是确定性派生量：

- 若保留 raw depth-ray：
  $$
  P_w(u,v)=t(u,v)+D(u,v)d(u,v),
  $$
  shape 为 $[B,V,H,W,3]$；
- 公共 API 默认先得到相机，再按 exporter 的实际实现：
  $$
  X_c=D(u,v)K^{-1}[u,v,1]^\top,\qquad
  X_w=T_{c2w}[X_c^\top,1]^\top,
  $$
  同样得到 $[B,V,H,W,3]$，再按 confidence 过滤并 flatten 为点云。

因此“DA3 输出 point”应准确写成：**depth + ray/camera 足以无学习地恢复 point map；默认 `forward/inference` 不直接返回独立 point-map head。** 这正是它相对 VGGT 多任务 heads 做减法的核心。

### 8.9 单图、多图、已知相机为何共用同一模型

| 模式 | token fusion | camera slot | 能恢复什么 / 原因 |
|---|---|---|---|
| 单图 $V=1$ | global attention 退化为单序列 attention | learned ref token | 依赖预训练与深度监督恢复相对单目几何；没有多视图基线，绝对尺度/位姿可观测性有限 |
| 多图未知相机 $V>1$ | 后段 global blocks 直接处理 $VT$ tokens | learned ref/src tokens；必要时先选参考视图并重排 | 跨视图 patch 对应与 camera slot 共同建立共享坐标系，输出相对一致 depth 与相机 |
| 多图已知相机 | 同上 | `CameraEnc(E,K)→[B,V,D]` 覆写 CLS slot | 相机条件在后段每个 local/global block 中参与注意力，使已知 pose 直接约束 feature fusion |
| 视频 | 把抽帧视作 $V$ 个有序视图 | 与未知/已知相机相同 | 架构本身没有专用时序卷积；一致性来自跨帧 global attention，长视频需 streaming/sliding-window 工程 |

### 8.10 与论文叙述、DINOv2 和 VGGT 的实现差异

1. **“vanilla DINO”是架构复杂度主张，不是逐行等同上游**：DA3 没有独立大型 aggregator，但后段 blocks 已加入 global token 重排、QK norm 与 RoPE。
2. **没有 register tokens**：DA3 vendored backbone 默认 `num_register_tokens=0`，序列前缀只有一个被后续 camera slot 复用的 CLS。
3. **absolute position 与 RoPE 共存**：patch 在进入第一层前已加 DINOv2 learned pos；RoPE 只从 `rope_start` 后进入 attention。
4. **camera token 是覆写，不是拼接**：序列长度始终 $1+P$，不会因为相机条件变成 $2+P$。
5. **几何 head 输入会翻倍**：`cat_token=true` 把最新 local stream 与当前融合 stream 拼成 $2D$；所以 Large Dual-DPT 是 2048 输入，不是 1024。
6. **ray 通道顺序以代码为准**：`camray[:,:,:3]` 用于方向/homography，`camray[:,:,3:]` 用于 translation 平均，即 `[d,t]`。
7. **default inference 不返回 raw ray**：camera head 路径会删除 ray；`use_ray_pose` 也在反解后删除。若研究 ray 本身，需要在 `_process_camera_estimation/_process_ray_pose_estimation` 前截取 raw head 输出。
8. **没有 point-map head**：点云来自 depth + camera/ray 的解析几何，而不是额外监督输出。
9. **单图不是多图模型的另一个 checkpoint**：$V=1$ 只让 global attention 自然退化；专门 `DA3MONO-LARGE` 是另一个针对单目质量优化的 artifact，不能与 main-series 单图模式混称。
10. **Giant 的 3DGS 是可选支线**：只有 config 配置 `gs_head/gs_adapter` 且 `infer_gs=true` 才输出 Gaussians；它不属于 depth-ray 最小主输出。

### 8.11 复现与工程建议

- **锁三类 revision**：代码 commit、HF repo revision、`model.safetensors` LFS SHA-256 缺一不可；不要只写 `main` 或模型简称。
- **优先 1.1**：Large/Giant 新实验使用 `-1.1`，旧 artifact 只用于复刻早期结果；同时记录 checkpoint 许可，不能从代码仓库 Apache-2.0 推断所有大权重都同许可。
- **shape 单测**：先用 $B=1,V=1/2,H=W=56$ 验证 patch grid 4×4、$P=16$；再断言 Large 四层 feature 最后一维为 2048、raw ray 最后一维为 6。
- **显式保留变换**：预处理会 resize/crop 并在多图尺寸不一时再次 center-crop；若需把 depth、主点和 3D 点映回原图，必须保存每一步 scale/crop offset。
- **分清 raw head 与公共 API**：调试 depth-ray 表示时直接调用/插桩 `DualDPT`；做常规重建时使用 `Prediction.depth/conf/extrinsics/intrinsics`。两者字段集合不同不是模型缺失。
- **视图数显存不是线性**：后段 global attention 的 score 规模近似 $B A[V(1+P)]^2$。提高 $V$ 与提高分辨率会相乘，长视频应使用官方 streaming/sliding-window，而不是一次堆入全部帧。
- **point map 自己断言坐标系**：输出 extrinsics 是 w2c；反投影必须先取 inverse 得 c2w。把 w2c 直接左乘 camera point 会产生看似成形但跨视图错位的点云。
- **ray pose 数值核验**：先检查 `[d,t]` 通道、`ray_conf` 权重和针孔相机假设，再比较 camera-head 与 ray-head AUC；鱼眼/全景不能只靠换投影公式宣称已复现。
- **外部 backbone 消融**：若要验证 DINOv2 贡献，应比较相同 DA3 heads/训练数据下的随机初始化与 DINOv2 初始化；不能在推理阶段把已训练 DA3 backbone 换回原始 DINOv2 后直接比较。

---

## 9. 材料来源

见 [sources.md](assets/DA3%20-%20任意视角恢复视觉空间/sources.md)。

