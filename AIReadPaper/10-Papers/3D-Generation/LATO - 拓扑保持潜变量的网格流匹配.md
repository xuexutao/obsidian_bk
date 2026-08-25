---
type: paper-note
status: done
domain: 3D-Generation
paper: "LATO: 3D Mesh Flow Matching with Structured TOpology Preserving LAtents"
year: 2026
arxiv: 2603.06357
doi: null
source: https://arxiv.org/abs/2603.06357
project: null
code: null
tags: [3D-Generation, Mesh, Flow-Matching, Sparse-Voxel, VDF, Topology-Preserving, Image-to-Mesh]
rating: 4.5
created: 2026-08-25
updated: 2026-08-25
---

# LATO：结构化拓扑保持潜变量的网格流匹配

## 基本信息

- **论文全称**：LATO: 3D Mesh Flow Matching with Structured TOpology Preserving LAtents
- **作者机构**：Tianhao Zhao*、Youjia Zhang*、Hang Long、Jinshen Zhang、Wenbing Li、Yang Yang、Wei Yang†（华中科技大学 HUST）、Gongbo Zhang（北京大学）、Jozef Hladký（独立研究者）、Matthias Nießner（TUM）
- **会议/年份**：Preprint（v1，2026-03-06 提交，2026-03-09 发布）
- **arXiv**：[2603.06357](https://arxiv.org/abs/2603.06357)
- **核心贡献**：用「顶点位移场 VDF + 稀疏体素 VAE（T-Voxels）+ 两阶段流匹配」把**显式网格拓扑**（顶点分布 + 边连通性）直接纳入可扩展的扩散生成管线，绕过 isosurface 抽取，产出艺术家级边流
- **领域**：3D-Generation（拓扑保持 / 流匹配 / 稀疏体素潜变量 / 显式网格生成）

## 一句话结论

LATO 要解决的是两条主流路线各自的老大难：隐式场系（VecSet / 稀疏体素 + Marching Cubes）**不暴露显式拓扑**，抽取出的网格过密、不规则，且被迫假设 watertight、丢掉了海量开表面数据；显式自回归系（MeshGPT 系）**算力天花板**——网格变复杂序列就变长，只能截断训练导致破面、碎片。LATO 的答案是设计一个「拓扑保持」的潜变量：用顶点位移场（VDF）把「顶点在哪 + 谁和谁连」的信息连续地编码进稀疏体素潜变量（T-Voxels），解码时分层细分 + 剪枝定位顶点、connection head 直接预测边，再用两阶段流匹配并行采样——既保留显式拓扑，又享受扩散的可扩展性与 3~10 秒的推理速度。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

大规模 3D 生成的主流范式是「VAE 压缩 + latent diffusion」。绝大多数方法把潜变量解码成隐式场（SDF / occupancy），再用 Marching Cubes 抽取网格。它们的空间归纳偏置分两派：**VecSet 潜变量**（3DShape2VecSet、CLAY、Michelangelo、TripoSG、Hunyuan3D）与**稀疏体素潜变量**（TRELLIS、Direct3D-S2、Sparc3D、LATTICE）。这两派都能学好整体几何，却共享一个关键缺陷——**不暴露显式网格拓扑**，导致后抽取的网格过密、不规则、偏离艺术家手工拓扑，无法直接用于 rigging、形变、游戏引擎；且隐式场强加的「watertight」假设迫使丢弃大量开表面、non-manifold 资产，加剧 3D 数据稀缺。

### 1.2 现有方法的瓶颈

- **隐式场系**：TRELLIS.2 用 dual-grid 允许开表面/non-manifold 训练，但拓扑仍由网格规则导出，而非学习到的艺术家意图。
- **显式扩散系**（PolyDiff、MeshCraft）：把面级特征转成连续表示，但表示随网格复杂度**二次增长**，内存受限只能截断训练 → 破面、碎片。
- **显式自回归系**（MeshGPT、MeshAnythingV2、EdgeRunner、Mesh Silksong、BPT、MeshTron、MeshXL、DeepMesh、MeshRFT）：序列化网格学 next-token，几何细节一多 token 流就极长，同样被内存逼到截断训练。
- **共同空白**：没有一种表征能同时做到「显式拓扑可学习」+「可扩展到高保真」+「快速并行推理」。LATO 的定位正是弥合这个 scalability 与显式拓扑学习之间的鸿沟。

### 1.3 本文核心贡献

1. **VDF（顶点位移场）**：采样表面点并嵌入「指向其所在面顶点的相对位移向量」，把顶点位置与连通性连续地编码进表面信号，避开离散标签的类别不平衡与不连续问题。
2. **T-Voxels 潜变量**：稀疏体素 VAE 把 VDF 压成结构化、拓扑感知的体素潜变量——不像 vecset/体素潜变量只暗示「表面存在」，而是显式编码「顶点的空间分布与连通性」。
3. **分层细分 + 剪枝的解码协议**：解码器逐级细分体素、可学习剪枝精确定位顶点；并行 connection head 直接预测顶点对的边连通性（假设三角面是图内闭合 3-环）。
4. **两阶段流匹配**：先生成粗结构体素，再细化体素级拓扑特征，支持 image-to-mesh 且推理 3~10 秒。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **任务**：图像（或几何）条件到显式三角网格的生成，以及无条件/条件重建。
- **输入**：重建时是表面采样点云（每点带位移向量 + 法向）；生成时是图像（经 TRELLIS 结构模型得到结构体素）。
- **输出**：显式网格 $M=(V,F)$，顶点集 $\{\hat{v}\}$ + 边连通性 $\{\hat{e}_{ij}\in\{0,1\}\}$。
- **关键假设**：① 三角面是顶点图内闭合 3-环，因此「顶点位置 + 边连通性」足以完整恢复网格；② 顶点数可用标量 $c_v=\log(N_v)$ 作生成条件控制。

---

## 3. 核心方法

### 3.1 顶点位移场 VDF

既有潜变量（点云 / occupancy 体素）本质是**拓扑无关**的——只编码连续表面边界，丢掉离散顶点分布与相互连通性。LATO 提出 VDF $F$：对网格 $M=(V,F)$ 表面任一点 $p$（落在面 $f=[f_0,f_1,f_2]$ 上），定义

$$F(p) = \left\{ v - p \;\middle|\; v \in \{v_{f_0}, v_{f_1}, v_{f_2}\} \right\}, \quad p \in f$$

直觉上：**零模长的位移**（$\|v-p\|\to0$）精确标记顶点位置，**位移组成的跳变**（跨越边时顶点集突变）勾勒出边。相比「把点分类成 vertex/edge/face」的朴素标签方案，VDF 有两个优势：① 顶点和边相对连续表面是**零测度集**，随机采样会极端类别不平衡；② 离散类别标签在潜空间制造突变，阻碍扩散学习所需的平滑梯度。VDF 提供**稠密、连续**的信号——面上每个点都携带重建顶点与连通性的坐标信息（Fig.9 有消融对比）。

### 3.2 稀疏体素 VAE：T-Voxels

把 VDF 压成紧凑结构化潜变量的 VAE，其编码器沿用 TRELLIS / TripoSF 的稀疏卷积/transformer 骨干，但解码协议专为拓扑保真设计。

对网格表面均匀采样 $K=819{,}200$ 个点，每个点 $p_k$ 构造拓扑注入的特征向量 $x_k = [p_k, F(p_k), n(p_k)]$（坐标 + 位移向量 + 法向）：

$$z = \mathcal{E}(X), \qquad \{\hat{v}, \hat{e}_{ij}\} = \mathcal{D}(z)$$

**编码器**：先把 $X$ 离散化到稀疏体素网格，每个被占体素内用 PointNet（shared MLP + 对称 mean-pooling）聚合局部特征，再用稀疏 transformer 捕获全局形状上下文，最后对每个 active 体素预测 mean/log-var，经重参数化采样得到 T-Voxel 潜变量 $z$（$128^3$ 分辨率 × 16 通道）。

**顶点集预测（分层细分 + 剪枝）**：直接回归顶点坐标难在顶点数随形状可变。解法是「细分 + 剪枝」级联：coarse T-Voxels 先过 sparse transformer + 初始 Pruning Head（线性分类器预测 occupancy 概率，<0.5 丢弃保证稀疏），然后级联 $L=3$ 个 Upsample-and-Refine 模块——每个模块把 active 父体素细分为 $2^3=8$ 个八分体，复制父特征 + 相对位置编码 + 3D 卷积消除子体素歧义，再剪枝去空体素、sparse self-attention 聚合局部、cross-attention（子体素为 query、全局 T-Voxels 为 K/V）。最终剩余体素的**质心**构成顶点集 $\{\hat{v}\}$。

**连接预测头（Connection Head）**：先让顶点体素特征与 T-Voxels 做 cross-attention 聚合全局上下文，再拼接两顶点特征用 MLP 预测边：

$$\hat{h}_v = \text{CrossAttn}(h_v, z), \qquad \hat{e}_{ij} = \text{MLP}(h_{\hat{v}_i} \oplus h_{\hat{v}_j})$$

训练时全枚举顶点对是二次复杂度、不可行，改用**采样策略**：ground-truth 边全作正样本，每顶点再采样 $N_n$ 个邻居 + 全局随机 $N_r$ 个顶点作候选；边无向，故用两种顺序拼接各算一次再平均，保证 permutation invariance；推理时全枚举所有顶点对确保不遗漏。

**损失函数**：端到端联合优化几何保真 + 拓扑精度：

$$L_{total} = L_{prune} + L_{vtx} + L_{conn} + \beta L_{KL}$$

- $L_{prune}$：每级分层 occupancy 的 BCE；$L_{vtx}$：最细层用 Asymmetric Loss（$\gamma_-=4, \gamma_+=0$，降权海量负样本、聚焦正顶点位置）；$L_{conn}$：候选顶点对的 BCE 边预测；$\beta L_{KL}$：变分正则。

![LATO 管线总览：VDF → 稀疏体素 VAE（T-Voxels）→ 分层细分 + connection head](assets/LATO%20-%20拓扑保持潜变量的网格流匹配/figures/03-pipeline.png)

### 3.3 显式网格流匹配

生成沿用 TRELLIS 的级联扩散，因子化两阶段：

1. **结构合成**：微调预训练 TRELLIS 结构模型（binary occupancy 监督），从图像生成 $128^3$ 稀疏体素 occupancy，显式纳入开表面/non-manifold 资产。
2. **拓扑特征生成**：以第一阶段的空间 anchor 为条件，flow matching transformer 合成 T-Voxel 特征，条件为顶点数标量 $c_v=\log(N_v)$：

$$L_{TF}(\theta) = \mathbb{E}_{z_0,t,\epsilon}\left\| v_\theta(z_t, t, c_v) - (\epsilon - z_0) \right\|_2^2$$

推理时先采样结构体素，再为每个体素生成拓扑特征，最后填充的 T-Voxels 交给解码器 $\mathcal{D}$ 直接实例化顶点 + 预测连通性，得到显式网格。

![LATO vs 既有范式：隐式 vs 显式 vs T-Voxels](assets/LATO%20-%20拓扑保持潜变量的网格流匹配/figures/02-paradigm.png)

---

## 4. 实验与结果

### 4.1 实验设置

- **数据集**：~400K 资产（TRELLIS-500K、Objaverse、Toys4K、ABO），**保留 non-watertight 网格**保证拓扑多样性。评测两组：artist mesh（G-Objaverse/Toys4K/ShapeNet 200 个 hold-out）+ dense mesh（TRELLIS 生成的 200 个高保真样本）。
- **指标**：Chamfer Distance（CD L1/L2）、Hausdorff Distance（HD）、Normal Consistency（NC，越大越好）。
- **训练**：AdamW，cosine lr $10^{-4}\to10^{-5}$；VAE 8×H100 4 天（batch 8/GPU）；两个流匹配模型在冻结潜变量上共 7 天（effective batch 128）。

### 4.2 VAE 重建（Tab.1）

| 方法 | CD(L2)↓ dense/artistic | CD(L1)↓ | HD↓ | NC↑ |
| --- | --- | --- | --- | --- |
| MeshGPT | 0.042 / 0.050 | 0.061 / 0.069 | 0.070 / 0.096 | 0.824 / 0.831 |
| PivotMesh | − / 0.129 | − / 0.175 | − / 0.263 | − / 0.614 |
| MeshCraft | 0.150 / 0.243 | 0.212 / 0.327 | 0.282 / 0.299 | 0.531 / 0.625 |
| **LATO** | **0.042 / 0.040** | **0.061 / 0.056** | **0.065 / 0.092** | **0.847 / 0.834** |

PivotMesh 在 dense 网格上直接 OOM（"−"），凸显显式逐面表示的内存瓶颈；LATO 在 dense/artistic 两类上 CD/HD/NC 全面最优，且 NC 最高说明拓扑最规整。

### 4.3 几何条件生成（Tab.2）

| 方法 | CD(L2)↓ dense/artistic | HD↓ | NC↑ |
| --- | --- | --- | --- |
| MeshAnythingv2 | 0.108 / 0.066 | 0.227 / 0.137 | 0.696 / 0.766 |
| MeshSilkSong | 0.062 / 0.052 | 0.145 / 0.101 | 0.784 / 0.818 |
| FastMesh | 0.064 / 0.048 | 0.110 / 0.102 | 0.757 / 0.813 |
| BPT | 0.059 / 0.052 | 0.107 / 0.088 | 0.811 / 0.824 |
| DeepMesh | 0.051 / 0.046 | 0.092 / 0.089 | 0.828 / 0.827 |
| **LATO** | **0.043 / 0.044** | **0.084 / 0.081** | **0.832 / 0.835** |

LATO 在 dense/artistic 上都拿 SOTA 对齐分数。定性上（Fig.4），自回归 baseline 因截断训练普遍产生孔洞，LATO 输出无孔、拓扑规整。

![LATO 几何条件生成对比](assets/LATO%20-%20拓扑保持潜变量的网格流匹配/figures/04-generation.png)

### 4.4 推理效率（Fig.7 / Tab.4）

关键差异：自回归方法受顺序预测约束，推理时间随面数**线性增长**（几分钟级）；LATO 的流匹配是**并行求解**，延迟与网格复杂度解耦。Tab.4 实测（单 H100，200 网格）：

| 方法 | 时间(s) vs 面数（关键点） |
| --- | --- |
| MeshAnythingv2 | 13.65s/460面 → 70.86s/2001面 |
| MeshSilkSong | 42.23s/638面 → 299.9s/4283面 |
| BPT | 36.85s/623面 → 251.8s/4029面 |
| DeepMesh | 147.5s/3526面 → 565.7s/13737面 |
| **LATO** | **5.47s/265面 → 9.87s/14809面** |

LATO 面数涨 55 倍（265→14809），时间只从 5.5s 涨到 9.9s——这是「扩散并行 vs 自回归串行」最直观的碾压。

### 4.5 消融（Tab.3）

**VAE 输入消融**（CD L2/L1 / HD / NC）：

| 变体 | CD(L2)↓ | CD(L1)↓ | HD↓ | NC↑ |
| --- | --- | --- | --- | --- |
| w/o $n(p_k)$（去法向） | 0.040 | 0.057 | 0.076 | 0.832 |
| w/o $F(p)$（去位移） | 0.043 | 0.061 | 0.092 | 0.822 |
| $K=81{,}920$（减采样点） | 0.040 | 0.056 | 0.078 | 0.832 |
| w/ norm$(F(p))$ | 0.042 | 0.059 | 0.082 | 0.827 |
| w/ norm$_\tau(F(p))$ | 0.041 | 0.058 | 0.081 | 0.832 |
| **Full** | **0.039** | **0.056** | **0.075** | **0.837** |

去 $F(p)$ 掉点最明显（HD 0.092 vs 0.075），证明 VDF 是核心信号；法向是辅助；对位移做归一化反而略差（保留原始尺度更重要）。此外 Fig.9 显示「显式 edge voxel 标签」输入会导致结构断裂，而 VDF 连续信号更稳；去掉解码器的 T-Voxel cross-attention 会产生孔洞与失配。

### 4.6 应用与可扩展性

- **拓扑对比隐式模型**（Fig.5）：面对 Tripo3D、TRELLIS、Hunyuan3D、CLAY 这些工业级隐式基础模型，LATO 在规模劣势下仍产出艺术家友好的边流，而非过密不规则三角化。
- **顶点数条件生成**（Fig.10）：调 $c_v$ 可控制面数/顶点数，提供生成粒度控制。
- **城市级生成**（Fig.8）：稀疏体素天然可组合，逐块生成建筑再拼装成高保真城市场景。

![LATO 与隐式基础模型的拓扑对比](assets/LATO%20-%20拓扑保持潜变量的网格流匹配/figures/05-topology.png)

---

## 5. 局限与评价

### 5.1 局限性（论文自述 + 个人观察）

1. **VDF 分辨率受限**：受底层稀疏体素网格约束，无法表示极小三角形 / 超精细几何细节；作者计划引入 octree 缓解。
2. **两阶段解耦的误差累积**：结构合成 → 拓扑特征 → 解码三级级联，若结构体素失准，后续拓扑特征会继承偏差。
3. **connection head 推理全枚举**：顶点多时边预测的二次枚举仍是一个潜在瓶颈（虽训练采样缓解）。
4. **代码未开源**（截至笔记撰写），复现门槛高。

### 5.2 个人判断与启示

LATO 是这份综述里**理论完成度最高的一篇**：它没有在「序列更长怎么办」上打补丁，而是换了一个表征——把拓扑信息**连续地**（VDF）编码进**结构化的**（稀疏体素）潜变量，从而同时拿到了显式网格系梦寐以求的「拓扑可控」和隐式场系赖以 scale 的「并行扩散」。VDF 对「离散标签 vs 连续场」的洞察尤其关键：它点出拓扑信息（顶点、边）本质是零测度、稀疏的，只有用位移场这种**稠密连续**形式才能喂进扩散模型。

它和本综述另一半的 LATTICE 形成有趣的镜像：LATTICE 用「表面相交体素中心」作 query 强化**几何定位**（Localizable Code），LATO 用「顶点位移场」强化**拓扑定位**（Topology-preserving）——两者都借 TRELLIS 系稀疏体素骨架，一个朝几何精度、一个朝拓扑规整各自再进一步。放在 MeshGPT → FACE → LATO 这条「直接网格」支线上，LATO 代表了从「自回归串行」到「扩散并行」的范式跃迁，是当前显式拓扑生成的较优解。

### 5.3 复现成本与建议

- **成本**：VAE 8×H100 4 天 + 流匹配 7 天，规模中等偏上，但比自回归系（DeepMesh 单次推理 565 秒）的边际成本友好得多。
- **建议**：核心难点在① VDF 特征构造（位移向量 + 法向的拼接）与② connection head 的采样训练 + 推理全枚举。若复现，可先从 TRELLIS 预训练结构模型微调入手，拓扑特征模型是独立可替换模块。

---

## 相关链接

- 论文 PDF：[assets/sources/paper.pdf](assets/LATO%20-%20拓扑保持潜变量的网格流匹配/sources/paper.pdf)（16 页，28.6 MB）
- 溯源表：[assets/sources/sources.md](assets/LATO%20-%20拓扑保持潜变量的网格流匹配/sources/sources.md)
- arXiv：https://arxiv.org/abs/2603.06357
