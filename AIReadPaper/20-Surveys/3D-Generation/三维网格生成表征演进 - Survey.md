---
type: survey
status: done
domain: 3D-Generation
paper: null
year: 2026
arxiv: null
doi: null
source: null
project: null
code: null
tags: [3D-Generation, Survey, Mesh, Neural-Field, VecSet, Sparse-Voxel, Autoregressive, Flow-Matching]
created: 2026-08-25
updated: 2026-08-25
---

# 三维网格生成的神经表征演进：从隐式场到显式几何 Token 化

## 基本信息

**报告主题**：生成式三维网格的「表征」之争——从连续隐式场，到向量集合，再到直接 token 化显式几何的演进

**覆盖范围**：2023.01（3DShape2VecSet）— 2026.03（FACE / LATO），重点为 2025 年之后

**核心论文**：3DShape2VecSet（arXiv 2301.11445）、TripoSG（arXiv 2502.06608）、LATTICE（arXiv 2512.03052）、TRELLIS.2（arXiv 2512.14692）、MeshGPT（arXiv 2311.15475）、FACE（arXiv 2603.01515）、LATO（arXiv 2603.06357）

**撰写日期**：2026-08-25

**报告定位**：以「问题演化」为主轴，回答「为什么后面的表征会出现」，梳理 VecSet 系、稀疏体素系、直接 Mesh 生成三条路线的来龙去脉与合流趋势

---

## 1. 研究背景

### 1.1 问题的原始形态：网格生成的三重约束

生成式三维网格（3D Mesh Generation）要回答的问题看似简单：给定一个文本、一张图或一段点云，生成一个可直接进入渲染与仿真管线的三角网格。但「网格」这个载体本身给生成模型设下了三道几乎互相矛盾的约束。

第一，不规则性与变长性。一张三角网格由数量不定的顶点与面构成，没有固定的空间排布，也无法像图像那样铺成规则的二维格点。生成模型——尤其是 Transformer 与扩散模型——恰恰最擅长处理规则、定长的 token 序列。

第二，拓扑与几何的耦合。网格的质量不仅取决于顶点位置（几何），还取决于顶点之间如何连成面（拓扑）。一个顶点放错只影响局部，一个连接写错就会让整个网格穿洞、翻转、非流形。这两件事在网格里是纠缠在一起的，但在生成模型里天然想拆开。

第三，下游管线的强约束。与点云、体素、隐式场不同，网格是要直接进渲染器、物理引擎、游戏资产的。它必须是流形的、水密的、面数可控的。这意味着「视觉上像」远远不够，还得「结构上对」。

这三重约束共同指向一个核心问题：生成式网格到底该用什么样的神经表征？ 本报告梳理的七篇论文，本质上都是在回答这同一个问题，只是答案随时代推进而不断演化。

### 1.2 表征是隐藏的主战场

在很长一段时间里，网格生成的主导路线是「绕开网格」：**先在隐式神经场（SDF / occupancy）里表示形状，最后再用 Marching Cubes 提取网格**。这条路线优雅地回避了不规则性问题——隐式场是连续的、处处可查的，扩散模型可以在场值上做文章。

但它把问题推迟到了最后一步。Marching Cubes 的拓扑被体素分辨率锁死，提取出的网格面数不可控、细节被抹平、还可能产生退化三角形。更关键的是，隐式场「逐点查询」的性质让它难以被大规模生成模型直接消费——你没法把一张网格自然地切成 token 喂给 Transformer。

于是「表征」从幕后走到台前，成为决定一个方法能走多远的核心变量。三条路线由此分化：

- 路线 A（VecSet 系）：把隐式场压缩成固定长度的「向量集合」，让它像 token 一样被 Transformer 处理；
- 路线 B（稀疏体素系）：用稀疏体素直接做结构化潜变量，绕开「稠密均匀」的浪费，保留原生三维结构；
- 路线 C（直接 Mesh 生成）：干脆跳过隐式中间层，把网格的顶点与面直接 token 化，自回归地生成显式几何。

### 1.3 本文脉络

本报告按「表征的三次进化」组织（第 2 章），再以时间线梳理三条路线各自的推进（第 3 章），随后对七篇论文逐一细读（第 4 章），最后横向对比并讨论尚未解决的问题（第 5-7 章）。七篇论文均已单独精读，本报告聚焦它们在共同问题下的相对位置，而非重复每篇的全部细节。

---

## 2. 表征的三次进化

### 2.1 第一次：从连续隐式场到向量集合

隐式场的根本不便在于它「不 token 化」。3DShape2VecSet（2023.01）给出了第一次系统性的答案：**Latent Set**。

它的想法是把形状编码成一组固定数量（M=512）的向量，这些向量没有显式空间坐标，也不按空间顺序排列，形状的全部信息都编码在向量之间的相互作用里。解码时，一个 cross-attention 层负责「查表」——任意空间位置用注意力从这组向量里检索出该处的场值。

这一步的关键贡献，是把「形状」从「连续函数」变成了「离散的、可被 Transformer 消费的 token 集合」。它用可学习的注意力相似度替换了此前表征（RBF、3DILG 等）里手工设计的位置核函数，让表征与 Transformer 架构天然对齐。此后 diffusion 可以直接在 Latent Set 上做，条件生成（类别 / 文本 / 图像 / 部分点云）也能统一接入。

下图对比了三种表征：隐式场、3DILG（上一代「带坐标的向量集合」）与本文的 Latent Set。核心区别在于 Latent Set 的向量不再绑定显式坐标，位置信息完全由 cross-attention 隐式编码。

![](assets/三维网格生成表征演进%20-%20Survey/figures/01-3dshape2vecset-representation.png)

> 图 1：3DShape2VecSet 的表征对比。从带显式坐标的 3DILG 向量集合，演化为完全无坐标、靠 cross-attention 编解码的 Latent Set。  
> 来源：论文 Figure 2，p.3，https://arxiv.org/abs/2301.11445

但 Latent Set 也埋下了伏笔：向量「无坐标」虽然换来了灵活，却也丢掉了「可定位性」——这正是路线 A 后面 LATTICE 要回头解决的东西。

### 2.2 第二次：从「能生成」到「能规模化」

VecSet 路线在 2025 年连过两关，把「能生成」推进到「能规模化」。

TripoSG（2025.02） 是第一关：它证明了 VecSet + 整流流（Rectified Flow）可以在生产级规模上工作。2M 高质量数据、四阶段清洗管线、1.5B→4B 的 MoE 整流流 DiT、加上 SDF / 法向 / Eikonal 混合监督的 3D VAE，把这条路线推到了 GPTEval3D 五维评估的全面领先。它最有价值的发现是「质量先于数量」——从 800K 数据里精筛到 180K，FID 反而下降，说明数据质量比堆数量更重要。

LATTICE（2025.12） 是第二关，也是更本质的一关。TripoSG 虽然 scale 上去了，但它没有回答「为什么 VecSet 很难 scale」。LATTICE 的答案是：VecSet 的表面点 query 缺乏可定位性（localizability）。表面采样点没有稳定的空间锚点，跨样本难以对齐，导致模型只能靠「局部性」这种模糊的归纳偏置去猜，而「局部性」在三维里根本不够。

LATTICE 的解法是 <span style="color: red">VoxSet + Voxel Queries</span>：用「与表面相交的体素中心」替换表面采样点，并为这些体素 query 注入 RoPE 位置编码。这句话的含义被它提炼为一条设计原则——localizable > locality：**表征首先得能定位，其次才谈得上局部性**。结果是 3D 扩散模型第一次在 0.6B→4.5B 范围内表现出单调、可预测的 scaling 曲线，并解锁了 token 级 test-time scaling（训练 6k、推理 30k）。

下图展示了 LATTICE 的 <span style="color: red">VoxSet</span> 与既有表征的对比：体素中心 query 提供了稳定、可定位的空间锚点。

![475](assets/三维网格生成表征演进%20-%20Survey/figures/03-lattice-voxset.png)

> 图 2：LATTICE 的 VoxSet 表征对比。用表面相交的体素中心替换 VecSet 的表面点 query，并注入 RoPE，把「可定位性」做成了 scaling 的前提。  
> 来源：论文表征对比图，https://arxiv.org/abs/2512.03052

### 2.3 第三次：从中间表征回归显式几何

VecSet 系和稀疏体素系都在「隐式或半隐式」的中间表征里做生成，网格仍是最后一步「解码」出来的产物。这条路径有个挥之不去的隐忧：**中间表征学得再好，网格的拓扑与顶点分布仍不受直接控制**。

MeshGPT（2023.11）率先换了个思路：直接把网格 token 化。它用图卷积编码器把网格编码成「逐面的 token 序列」，再让 GPT-2 自回归地生成下一个 token，从零「写」出一张网格。这是「显式几何 token 化」的开山之作。

这条路线随后被两次关键改进推向成熟。FACE（2026.03）点破了 MeshGPT 式逐顶点 token 化的症结——语义层级错误：把不同语义粒度的几何（一张面 vs 一个顶点）压进同一个 token，序列冗长且学习困难。它改为「一面一 token」，序列缩短约 9 倍、压缩比压到 0.11。LATO（2026.03）则另辟蹊径，**用顶点位移场（VDF）把拓扑信息嵌入稀疏体素**，再用两阶段流匹配分别生成结构与拓扑，直接控制顶点的位置与连接。

这一进化的本质是：从「隐式连续场」回归「显式结构化几何」，从「均匀稠密网格」走向「稀疏、自适应、可定位的结构」。三条路线到 2026 年已经开始互相缝合——FACE 用了 VecSet 编码器（路线 A）加面自回归（路线 C），LATO 用了稀疏体素（路线 B）加流匹配（路线 A/C）。路线之间的边界正在消失。

---

## 3. 技术时间线

### 3.1 路线 A：VecSet 系（隐式神经场派生）

| 时间      | 工作             | 解决了什么                                                                                      | 留下什么问题                                             |
| ------- | -------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| 2023.01 | 3DShape2VecSet | 首次把形状压缩为无坐标、固定长度的 Latent Set，让 3D 可被 Transformer 处理，五条件生成全面刷新 ShapeNet SOTA                | 表征容量受限（M=512）、ShapeNet-only 评测、Marching Cubes 限制拓扑 |
| 2025.02 | TripoSG        | 把 VecSet 推到生产级：2M 数据 + 4B MoE 整流流 DiT + 混合监督 VAE，发现「质量先于数量」与「整流流最稳」                        | 训练成本 3 周×160 A100；未解释 VecSet 为何难 scale             |
| 2025.12 | LATTICE        | 用 VoxSet + RoPE 解决表面点 query 的可定位性问题，首次实现 0.6B→4.5B 单调 scaling，解锁 token 级 test-time scaling | 两阶段管线依赖上游粗网格、薄结构难生成、无材质                            |

这条路线贯穿始终的是「隐式场」的血统：无论 Latent Set 还是 VoxSet，最终都还是要解码回场值再提网格。它的核心命题是**「先把形状压成好 token，再谈生成」**。

### 3.2 路线 B：稀疏体素系（结构化潜变量）

| 时间 | 工作 | 解决了什么 | 留下什么问题 |
|---|---|---|---|
| 2024 | TRELLIS | 用统一的 SLAT（稀疏结构潜变量）把图/文生成 3D 资产、局部编辑、多格式解码整合进一套框架 | 几何与外观分离不足，材质需要额外处理 |
| 2025.12 | TRELLIS.2 | 提出 O-Voxel 统一编码几何与 PBR 材质，SC-VAE + 4B flow-matching 在紧凑潜空间实现高质量资产生成 | 原生三角网格的拓扑仍由解码器决定，不是直接生成 |

稀疏体素系的价值主张是**「保留原生三维结构」**：体素天然是规则的三维格点，稀疏化后既省算力又保留空间结构，天然适合做 3D 基础模型的潜空间。它与 VecSet 系的区别在于，它不追求「压成无坐标向量」，而是「保留坐标但只保留有用的格点」。

下图是 TRELLIS.2 的方法总览：O-Voxel 表示 + SC-VAE + flow-matching 生成。

![](assets/三维网格生成表征演进%20-%20Survey/figures/04-trellis2-overview.png)

> 图 3：TRELLIS.2 总体框架。O-Voxel 统一编码几何与 PBR 材质，SC-VAE 压缩为紧凑潜变量，4B flow-matching 负责生成。  
> 来源：论文 Figure 2，https://arxiv.org/abs/2512.14692

### 3.3 路线 C：直接 Mesh 生成（显式几何 token 化）

| 时间 | 工作 | 解决了什么 | 留下什么问题 |
|---|---|---|---|
| 2023.11 | MeshGPT（CVPR 2024） | 开创三角网格自回归生成：图卷积 + RQ 量化把网格编码为逐面 token，GPT-2 medium 从零生成锐利紧凑的网格 | 采样 30-90 秒、GPT2-medium 偏小、不支持场景级 |
| 2026.03 | FACE | 改为「一面一 token」，序列缩短 9 倍、压缩比 0.11，VecSet 编码器 + 面自回归 + CausalMLP，三数据集全面 SOTA | 离散表征精度上限、依赖点云、两级自回归采样仍慢 |
| 2026.03 | LATO | 用 VDF 把拓扑嵌入稀疏体素，两阶段流匹配分别生成结构与拓扑，重建与生成双 SOTA，推理比 DeepMesh 快 15-57 倍 | VDF 分辨率受限、两阶段误差累积、代码未开源 |

这条路线的核心命题是**「跳过中间层，直接生成网格本身」**。它最激进，也最直接地面对 1.1 节的三重约束——不规则性、拓扑几何耦合、下游强约束。它的演进主线是「token 的粒度」：从逐顶点（MeshGPT）到逐面（FACE）再到「结构 + 拓扑分层」（LATO）。

---

## 4. 核心方法：七篇细读

### 4.1 3DShape2VecSet（2023.01）

**要解决的问题**：隐式神经场难以被生成模型直接消费，需要一种「可 token 化」的 3D 表征。

**核心命题**：把形状压成固定数量、无显式坐标的向量集合，位置信息由 cross-attention 隐式编码。

方法上分三步：一个编码器把形状（点云或场）编码为 Latent Set；一个 cross-attention 解码器在任意位置「查表」出该处的场值（SDF / occupancy / unsigned distance）；一个 latent diffusion 在 Latent Set 上做无条件与条件生成。关键设计是用**可学习的注意力相似度**替换 RBF / 3DILG 的手工位置核，让表征与 Transformer 天然对齐。

它确立了 VecSet 系的基本范式：**编码为集合 → 扩散在集合上 → 解码回场**。这条范式后来被 TripoSG、LATTICE、乃至 FACE 的编码器继承。它的局限也一并被继承下来：Latent Set 的容量（M=512）限制了细节，无坐标带来的「可定位性缺失」直到 LATTICE 才被系统解决。

### 4.2 TripoSG（2025.02）

**要解决的问题**：VecSet 路线能不能 scale 到生产级？scale 的瓶颈在哪？

**核心命题**：能，但前提是「数据质量先于数量」+「整流流最稳」+「MoE 扩模型不扩推理」。

三个层面的贡献：数据侧，四阶段 DBS 清洗管线从 2M 候选里筛出高质量子集，实验证明 800K→180K 精筛后 FID 反而下降；模型侧，1.5B→4B 的 MoE 整流流 DiT，用稀疏激活在几乎不增推理 FLOPs 的前提下扩容量；表征侧，3D VAE 用 SDF / 法向 / Eikonal 混合监督，让解码出的形状更规整。

下图是 TripoSG 的整体架构：VAE 编码到 Latent Set，整流流 DiT 在 Latent Set 上做 text/image 条件生成。

![](assets/三维网格生成表征演进%20-%20Survey/figures/02-triposg-architecture.png)

> 图 4：TripoSG 架构。3D VAE 把形状压缩为 Latent Set，整流流 DiT 在集合上做条件生成，MoE 支持 1.5B→4B 扩展。  
> 来源：论文架构图，https://arxiv.org/abs/2502.06608

它在 GPTEval3D 五维评估上全面领先 MeshFormer / MeshLRM / CraftMan 等 SOTA。但它本质上是「工程把 VecSet 推到底」，没有回答「VecSet 为什么难 scale」这个更深的问题——这个问题的答案要等 LATTICE。

### 4.3 LATTICE（2025.12）

**要解决的问题**：为什么 VecSet 系很难 scale？TripoSG 的 4B 是终点还是瓶颈？

**核心命题**：VecSet 的表面点 query 缺乏**可定位性**，这是 scaling 的隐藏瓶颈；用体素中心 + RoPE 可以修好。

这是七篇里「问题意识最锐利」的一篇。它把原则提炼成一句话：**localizable > locality**。表面采样点作为 query 时，没有稳定的空间锚点，跨样本难以对齐，模型只能靠「局部性」这种模糊先验去猜——而「局部性」在三维里天然不够用（两个空间上很近的点，可能分属不同的几何部件）。

VoxSet 的解法：用「与表面相交的体素中心」做 query，这些体素中心是规则格点、天然可定位，再注入 RoPE 编码空间位置。结果是 3D 扩散模型第一次表现出**单调可预测的 scaling 曲线**（0.6B→4.5B），并解锁 token 级 test-time scaling——训练时 6k token、推理时 30k token，质量持续提升，这在 VecSet 系里前所未有。2B base 模型用 64 GPU 24 小时就超越了前 SOTA。

它的局限也清楚：仍是两阶段管线（依赖上游粗网格），薄结构难生成，无材质。但它的方法论贡献——「表征首先得能定位」——已经溢出到整个领域。

### 4.4 TRELLIS.2（2025.12）

**要解决的问题**：稀疏体素潜变量能否同时承载几何与材质，成为「可直接使用资产」的紧凑表征？

**核心命题**：O-Voxel 统一编码几何与 PBR 材质，SC-VAE 压缩，flow-matching 生成。

TRELLIS.2 把 TRELLIS 的 SLAT 升级为 O-Voxel：每个体素不仅记录几何占用，还记录 PBR 材质属性，从而让生成结果从「几何模型」升级为「可直接渲染的资产」。SC-VAE 负责把 O-Voxel 压缩为紧凑潜变量，4B flow-matching 在潜空间生成。它代表稀疏体素系的成熟形态——**「结构化 + 紧凑 + 原生材质」**。

它在这份报告里的位置，是路线 B 的代表：与 VecSet 系不同，它不把形状压成「无坐标向量」，而是保留体素的规则三维结构，只做稀疏化。这条路线与路线 C 的张力在于——体素仍是「中间表征」，网格拓扑最终仍由解码器决定。

### 4.5 MeshGPT（2023.11）

**要解决的问题**：能不能跳过隐式场，直接生成三角网格？

**核心命题**：把网格 token 化成「面序列」，用自回归 Transformer 从零写出一张网格。

这是「直接 Mesh 生成」的开山之作，确立了「几何 token 化 + 自回归」的完整范式。方法分三段：**图卷积编码器**（SAGEConv）把网格编码为逐顶点的几何特征；**残差向量量化（RQ）** 把这些特征离散化为 codebook index（depth 2，每面 D=6 codes，per-vertex 量化 D/3）；**GPT-2 medium**（24 层 16 头 768 宽，context 4608）自回归地预测下一个 codebook index，逐面「长」出网格。

下图是它的编码器-解码器结构：图卷积编码 + RQ 量化 + transformer 自回归。

![](assets/三维网格生成表征演进%20-%20Survey/figures/05-meshgpt-encoder-decoder.png)

> 图 5：MeshGPT 的编码-解码流程。图卷积编码器把网格编码为逐面 token，RQ 量化离散化，Transformer 自回归生成。  
> 来源：论文架构图，https://arxiv.org/abs/2311.15475

在 ShapeNet 上，它以远少于 Polygen 的面数（228 vs 603）取得更低 FID（18.46 vs 61.10），COV 43.28。它证明「直接生成网格」可行且锐利。局限是采样 30-90 秒、模型偏小、不支持场景级——这些都为后来的 FACE、LATO 留下了改进空间。

### 4.6 FACE（2026.03）

**要解决的问题**：MeshGPT 式逐顶点 token 化为什么低效？能不能更紧凑地 token 化一张面？

**核心命题**：逐顶点 token 化犯了「语义层级错误」；「一面一 token」才是正确的粒度。

FACE 的诊断最犀利：把每个顶点都单独 token 化，等于把「面」这个语义上更自然的基本单元打散了，导致序列冗长、学习困难、且顶点坐标之间无法共享上下文。它的解法是 **one-face-one-token**——用单个 token 编码整张三角面，顶点坐标在面内由 CausalMLP 自回归地解码。序列长度缩短约 9 倍，压缩比压到 0.11（此前最优 0.22）。

架构上，FACE 缝合了两条路线：**VecSet 编码器**（承袭 3DShape2VecSet 的 FPS + cross-attention + Transformer Encoder）负责把点云压成集合；**自回归 Face Decoder**（causal self-attention + cross-attention 注入 VecSet）负责逐面生成。500M 的非对称 ARAE 做网格重建，350M 的 DiT 做 image-to-mesh。

下图是 FACE 的端到端管线：VecSet 编码器压缩点云 → 自回归 Face Decoder 逐面生成。

![](assets/三维网格生成表征演进%20-%20Survey/figures/06-face-pipeline.png)

> 图 6：FACE 端到端管线。VecSet 编码器压缩点云为集合，自回归 Face Decoder 用「一面一 token」逐面生成网格。  
> 来源：论文 pipeline 图，https://arxiv.org/abs/2603.01515

它在 Objaverse HD（0.090）、Toys4K（0.067）、Famous（0.077）三个数据集全面刷新重建 SOTA。局限是离散表征有精度上限、依赖点云输入、两级自回归采样仍偏慢。但它点破的「语义层级」问题，是自回归网格生成从「能用」走向「好用」的关键转折。

### 4.7 LATO（2026.03）

**要解决的问题**：能不能在生成网格时**显式控制拓扑**，而不只是生成顶点位置？

**核心命题**：用顶点位移场（VDF）把拓扑信息嵌入稀疏体素，两阶段流匹配分别生成「结构」与「拓扑」。

LATO 的视角又不一样。它不 token 化面，而是把网格拆成「顶点在哪里」与「顶点怎么连」两件事。顶点位移场（VDF）在表面采样点处嵌入「指向其所在面顶点的相对位移向量」，把顶点信息压进稀疏体素（T-Voxels，128³×16 通道）；分层细分 / 剪枝负责精确定位顶点；connection head 直接预测边连通性（三角面 = 闭合 3-环假设）。

生成用**两阶段流匹配**：第一阶段学习 TRELLIS 的结构合成，第二阶段生成拓扑特征（条件 $c_v = \log(N_v)$ 控制顶点数）。Asymmetric Loss（$\gamma^-=4, \gamma^+=0$）处理顶点类别的极度不平衡。

下图是 LATO 的管线总览：VDF → 稀疏体素 VAE（T-Voxels）→ 分层细分 + connection head。

![](assets/三维网格生成表征演进%20-%20Survey/figures/07-lato-pipeline.png)

> 图 7：LATO 管线。VDF 把顶点位移嵌入稀疏体素，T-Voxels VAE 编码，分层细分定位顶点，connection head 预测连通性。  
> 来源：论文 pipeline 图，https://arxiv.org/abs/2603.06357

它在重建（CD 0.042、NC 0.847）与生成（CD 0.043）上双 SOTA，推理比 DeepMesh 快 15-57 倍（5.47s→9.87s vs 147.5s→565.7s）。它是「结构合成 + 拓扑特征」分层范式中理论完成度最高之作，与 LATTICE 互为镜像——一个在隐式场里修「可定位性」，一个在显式网格里做「拓扑保持」。

---

## 5. 三条路线横向对比

| 维度 | 路线 A：VecSet 系 | 路线 B：稀疏体素系 | 路线 C：直接 Mesh 生成 |
|---|---|---|---|
| 代表工作 | 3DShape2VecSet → TripoSG → LATTICE | TRELLIS → TRELLIS.2 | MeshGPT → FACE / LATO |
| 表征形态 | 无坐标向量集合（Latent Set / VoxSet） | 稀疏体素（SLAT / O-Voxel） | 顶点 / 面 token 序列 |
| 生成范式 | 扩散 / 整流流 | flow-matching | 自回归 / 流匹配 |
| 网格怎么来 | 解码回场，Marching Cubes 提取 | 解码器从体素提取 | 直接生成顶点与连接 |
| 拓扑可控性 | 间接（受 MC 分辨率限制） | 间接（受体素分辨率限制） | 直接（显式预测连接） |
| 核心瓶颈 | 可定位性 → 已被 LATTICE 修好 | 仍是中间表征，拓扑非直接 | token 粒度与序列长度 |
| 是否产材质 | 否（纯几何） | 是（TRELLIS.2 的 PBR） | 否 |

一个值得注意的现象是**路线边界正在消失**。FACE 用了路线 A 的 VecSet 编码器，配路线 C 的面自回归；LATO 用了路线 B 的稀疏体素，配路线 A 的流匹配。这说明三条路线回答的其实是同一个问题的不同侧面，而 2026 年的工作开始把它们拼回一个整体。

---

## 6. 当前正在攻坚的问题

### 6.1 拓扑与几何的分层生成

LATO 已经证明「结构 + 拓扑」两阶段流匹配可行，但这是否是唯一解？MeshGPT / FACE 把拓扑和几何揉进同一个 token 序列，LATO 把它们拆成两个阶段。前者序列冗长但统一，后者干净但误差会跨阶段累积。**「拆还是合」是自回归网格生成尚未定论的根本问题**。

### 6.2 可定位性成为跨路线的共识

LATTICE 提出的「localizable > locality」并非只对隐式场成立。FACE 的「语义层级」诊断、LATO 的「VDF 相对位移嵌入」，本质上都在解决同一个问题——**让表征里的每一个单元都知道自己在空间中的位置和角色**。这已经从路线 A 的一个工程技巧，上升为整个网格生成领域的共同课题。

### 6.3 从「生成几何」到「生成可用资产」

TRELLIS.2 的 O-Voxel 把材质纳入表征，是朝「可用资产」迈出的重要一步。但三条路线里，只有路线 B 明确处理了材质；路线 A 和 C 仍是「纯几何」。未来的网格生成，大概率要在「直接生成拓扑」与「同时携带材质」之间找到结合点——这是路线 B 与路线 C 最自然的合流方向。

### 6.4 面数可控与自适应分辨率

网格的价值之一在于「面数可控」——低面数给实时渲染，高面数给精修。但自回归生成天然是「生成到停止」，面数不可精确控制；LATO 用 $\log(N_v)$ 条件控制顶点数，是朝这个方向的一个尝试。**「按需生成指定精度的网格」仍是一个开放问题**。

---

## 7. 局限与评价

### 7.1 已经确立的事实

**隐式场的「不 token 化」是根本缺陷**，向量集合与稀疏体素都是对这一缺陷的系统性修补，且都取得了显著成功。

**「可定位性」是 3D 表征 scaling 的前提**。LATTICE 用一条单调的 scaling 曲线证明了这一点，这是本领域少有的「有说服力的负结果→正结果」的转变。

**直接生成显式网格是可行的，且正快速逼近中间表征路线的质量**。MeshGPT 开山、FACE 点破语义层级、LATO 显式控制拓扑，三代之间每一代都解决了一个实质问题。

### 7.2 尚未解决的结构性问题

**拓扑与几何的最优分层方式仍未定论**（见 6.1）。

**材质与几何的统一表征**只在路线 B 里成熟，跨路线整合尚无成熟方案。

**评测的粒度仍偏粗**。CD / FID / NC 能区分「整体像不像」，但难以区分「拓扑对不对、面数合不合理、是否流形」——而这恰恰是网格区别于其他 3D 表征的核心价值。

### 7.3 接下来可能发生什么

短期可预期：拓扑-几何分层生成成为自回归网格的主流；「可定位性」成为审稿人默认会问的设计点；材质几何统一表征向路线 C 渗透。

中期值得盯：直接生成 + 材质结合（路线 B × 路线 C 合流）；面数精确可控的生成；评测指标从「像不像」升级到「结构对不对」。

---

## 8. 关键文献索引

### 8.1 细读的七篇

| 工作 | 出处 | 链接 | 精读笔记 |
|---|---|---|---|
| 3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models | arXiv 2301.11445 | https://arxiv.org/abs/2301.11445 | [[3DShape2VecSet - 神经场与扩散生成的三维形状表征]] |
| TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models | arXiv 2502.06608 | https://arxiv.org/abs/2502.06608 | [[TripoSG - 大规模整流流三维形状合成]] |
| LATTICE: Democratize High-Fidelity 3D Generation at Scale | arXiv 2512.03052 | https://arxiv.org/abs/2512.03052 | [[LATTICE - 大规模高保真三维生成表征]] |
| Native and Compact Structured Latents for 3D Generation（TRELLIS.2） | arXiv 2512.14692 | https://arxiv.org/abs/2512.14692 | [[TRELLIS.2 - 原生几何与材质的紧凑三维潜变量生成]] |
| MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers | CVPR 2024，arXiv 2311.15475 | https://arxiv.org/abs/2311.15475 | [[MeshGPT - 三角网格自回归生成]] |
| FACE: A Face-based Autoregressive Representation for High-Fidelity and Efficient Mesh Generation | arXiv 2603.01515 | https://arxiv.org/abs/2603.01515 | [[FACE - 基于面的自回归网格生成]] |
| LATO: 3D Mesh Flow Matching with Structured Topology Preserving Latents | arXiv 2603.06357 | https://arxiv.org/abs/2603.06357 | [[LATO - 拓扑保持潜变量的网格流匹配]] |

### 8.2 直接前置工作

| 工作 | 时间 / 出处 | 在本脉络中的角色 |
|---|---|---|
| 3DILG | 2022 | 3DShape2VecSet 的直接前身（带坐标的向量集合） |
| TRELLIS | 2024 | 路线 B 起点（SLAT 结构化潜变量） |
| Polygen / GET3D | 2022 | MeshGPT 的主要对比基线（上一代网格生成） |
| DeepMesh | 2024 | LATO 的主要对比基线（拓扑优化网格生成） |

---

*本报告基于七篇论文全文（含附录）的细读整理，定量数字均取自原论文表格。文中插图裁剪自各论文原文，并已在本库对应单篇笔记的 `sources.md` 中留存溯源。*

*姊妹篇：[[三维多模态大语言模型架构 - Survey]] 覆盖「网格/点云如何被 LLM 消费」——mesh-as-text（LLaMA-Mesh / MeshLLM / MeshCoder）与 tokenizer 效率竞赛（MeshAnything V2 / BPT / EdgeRunner / TreeMeshGPT / Mesh-Silksong / MeshWeaver，压缩比 1.00 → 0.18）两条线，与本文的生成侧表征之争互补。*
