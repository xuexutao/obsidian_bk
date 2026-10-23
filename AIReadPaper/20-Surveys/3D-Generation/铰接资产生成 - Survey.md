---
type: survey
status: final
domain: 3D-Generation
topic: 铰接资产生成（Articulated Asset Generation）
created: 2026-10-23
updated: 2026-10-23
related-papers:
  - "[[Particulate - 前馈式三维物体铰接]]"
  - "[[FAMOS - 稀疏多状态观测的铰接建模]]"
  - "[[Articulate AnyMesh - 开放词汇铰接物体建模]]"
  - "[[Articulate-Anything - VLM驱动的URDF铰接生成]]"
  - "[[SIMART - 静态网格到仿真就绪铰链资产]]"
  - "[[PhysX-Omni - 单图生成仿真就绪三维资产]]"
  - "[[CubePart - 开放词汇部件可控三维生成]]"
tags:
  - 3D-Generation
  - Articulated-Object
  - Survey
---

# 铰接资产生成 - Survey

## 0. 这篇综述覆盖什么

**铰接资产（articulated asset）**指带可动部件与关节约束的 3D 资产：柜门绕铰链旋转、抽屉沿滑轨平移、笔记本绕轴开合。它是「3D 生成」与「具身仿真」的接口——静态 mesh 再好看，机器人仿真里打不开的柜子就是一块石头。本综述梳理 2024-2026 年这条线的技术版图，精读笔记覆盖四篇代表作：

- [[Particulate - 前馈式三维物体铰接]]（CVPR 2026）——前馈学习路线的当前标杆
- [[FAMOS - 稀疏多状态观测的铰接建模]]（arXiv 2026）——多状态观测路线
- [[Articulate AnyMesh - 开放词汇铰接物体建模]]（CoRL 2025）——VLM 零训练流水线
- [[Articulate-Anything - VLM驱动的URDF铰接生成]]（ICLR 2025）——agentic actor-critic 路线

另与已有笔记 [[SIMART - 静态网格到仿真就绪铰链资产]]（MLLM 统一路线）、[[PhysX-Omni - 单图生成仿真就绪三维资产]]（单图到带物理属性资产）、[[CubePart - 开放词汇部件可控三维生成]]（部件可控生成）互相衔接。

## 1. 任务的正式定义

给定静态几何观测（mesh / 点云 / 图像 / 视频），输出铰接结构四元组：

$$A = (P,\ S,\ K,\ \mathcal{M})$$

- $P$：部件数；$S$：面/点到部件的分割；
- $K$：运动学树（部件间的父子连接，通常假设单一基座部件）；
- $\mathcal{M}$：每个关节的类型（revolute / prismatic）、轴（方向 + 位置）、限位范围。

三个隐含的常见假设构成这条线的表达边界：**部件刚性**（部件内不形变）、**单自由度关节**（排除球铰、齿轮、柔索）、**树状拓扑**（排除闭环机构）。最终交付形态通常是 URDF（直接进 Isaac/Sapien/PyBullet）。

与相邻任务的关系：**部件分割**（PartField、P3SAM、[[CubePart - 开放词汇部件可控三维生成]]）只回答"分成几块、各是什么"，不回答"怎么动"——Particulate 论文实证了「语义部件」与「铰接部件」是两种不对齐的监督目标；**部件级生成**（CubePart、PartCrafter 等）给部件不给关节；**骨骼绑定 rigging**（MagicArticulate，自回归生成骨架 + functional diffusion 蒙皮）面向动画而非物理仿真，关节语义不同。

## 2. 技术路线版图

按「先验从哪里来」划分，四条路线泾渭分明：

### 2.1 前馈学习路线（Particulate、FAMOS）

把任务当监督学习问题：在大规模铰接资产上训练 query-based Transformer，一次前馈输出全部属性。

- [[Particulate - 前馈式三维物体铰接]]：单静态网格输入。DETR 式部件匹配 + 多头解码；两个关键设计——**旋转轴过参数化**（方向直回归、位置由点级投票取中位数）与 **PartField 语义特征注入**；推理秒级，能处理 VLM 看不见的内部部件。
- [[FAMOS - 稀疏多状态观测的铰接建模]]：多状态稀疏局部点云输入。核心洞察是**单状态铰接估计本质歧义**，用「状态内+全局」交替注意力（点 token 直接跨观测交互）+ **观测运动跨度损失**（集合级目标，防止单观测作弊）强制模型利用运动证据；配程序化数据生成器（构造即标注、在线无限流）。

### 2.2 VLM agentic 路线（Articulate-Anything、Articulate AnyMesh）

拒绝在稀缺数据上训练，把赌注押在基础模型的常识上。

- [[Articulate-Anything - VLM驱动的URDF铰接生成]]：agent 写 Python 代码拼 URDF，actor-critic 渲染回喂自纠错，mesh retrieval 拼装部件。胜在输入模态（视频消歧）、可自纠错、开源；输出是「库里部件的重组物」。
- [[Articulate AnyMesh - 开放词汇铰接物体建模]]：三段流水线——PartSlip++ 开放词汇分割 → **几何感知视觉提示**（连接区域候选点投影 2D，GPT-4o 指认铰链）→ HoloPart 补全 + Meshy 纹理。胜在保留用户原网格、完全零训练；败在内部部件盲区与级联误差。

### 2.3 MLLM 统一路线（SIMART、PhysX-Omni）

- [[SIMART - 静态网格到仿真就绪铰链资产]]：Sparse 3D VQ-VAE 压缩几何 token（4138→516）后，Qwen3-VL 一次多模态推理同时输出部件 grounding、运动学与 URDF 物理属性（尺度/密度/摩擦）。比 2.2 路线多覆盖了物理属性，比 2.1 路线语义更强，代价是 MLLM 推理开销。
- [[PhysX-Omni - 单图生成仿真就绪三维资产]]：Template-based RLE 几何表示，单图统一生成刚体/柔体/关节体 + 物理属性，配套 PhysXVerse 数据与 benchmark。

### 2.4 生成式与程序化路线

- **分布学习生成**：NAP、CAGE（图上扩散，部件在顶点、关节在边）、ArtFormer、ArtiLatent（SIGGRAPH Asia 2025）、MeshArt（层次自回归：先 bbox+铰接模式再部件几何）——从零「创作」新铰接物体，受训练类别限制；
- **程序化合成**：Infinite Mobility（布局约束求解 + 物理规则）、FAMOS 的程序化生成器——产出有限习语但标注免费；
- **感知重建**：PARIS、Ditto 等从多状态扫描重建，是本领域的前身。

## 3. 横向对比

| | Particulate | FAMOS | Articulate AnyMesh | Articulate-Anything | SIMART |
|---|---|---|---|---|---|
| **输入** | 单静态 mesh | 多状态稀疏点云（含单状态） | 任意静态 mesh | 文本/图像/视频 | mesh + 渲染图 + 指令 |
| **先验来源** | 监督学习（3.8K 资产） | 监督学习 + 程序化数据 | VLM 常识 + 几何启发式 | VLM 常识 + 检索库 | MLLM |
| **是否保留原几何** | ✅ | ✅（提升为点云） | ✅ | ❌（检索拼装） | ✅ |
| **内部部件** | ✅（数据学出） | ✅ | ❌ | ❌ | ✅ |
| **运动限位** | ✅ 显式预测 | ❌（仅训练辅助目标） | ❌ | ✅（代码生成） | ✅ |
| **物理属性** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **推理速度** | 秒级 | 前馈，秒级 | 分钟级（多轮 VLM） | 分钟级（迭代闭环） | MLLM 推理 |
| **开放词汇泛化** | 受训练分布限制 | 受训练分布限制（多状态缓解） | ✅ 强 | ✅ 强 | 较强 |
| **自纠错** | ❌ | ❌ | ❌ | ✅ critic 闭环 | ❌ |
| **代码开源** | ❌（截至 2026-10） | ❌ | 部分 | ✅ | ✅（ByteDance-Seed） |
| **一剑封喉的证据** | Lightwheel gIoU 0.332 vs AnyMesh 0.172 | ACD MAO F1 64.5 vs 多状态 Particulate 36.7 | OOD 角度误差 6.58° vs URDFormer 33.86° | 成功率 75% vs ~10% | 30% token + OOD 关节类型 0.831 |

读表的两个要点：**「保留原几何」与「开放词汇」目前不可兼得**（学习路线保几何但受分布限制，VLM 路线开放但不保留/不精确）；**自纠错回路只在 agentic 路线存在**，前馈模型错了就是错了——test-time scaling（如 Particulate 多采样投票）是明显的未开垦地。

## 4. 数据与基准：这条线的真正瓶颈

所有路线都卡在同一件事上：**铰接标注数据极度稀缺**。Particulate 的训练集只有 3,800 物体；PartNet-Mobility 约 2.3K 物体；对比 Objaverse 的 10M+ 静态资产差三个数量级。当前的数据策略有四种：

1. **拼现有数据集**：PartNet-Mobility + GRScenes + ACD（Particulate/FAMOS）；
2. **程序化合成**：FAMOS 生成器（15 族、构造即标注、无限流）、ArtiCraft-10K（LLM agent 生成）——消融显示这是 OOD 泛化的最大功臣；
3. **自建精细基准**：Lightwheel 243 物体（CC-BY-NC，Particulate）；
4. **自动打标既有资产**：AnyMesh 给 Objaverse 打标——流水线自身成为数据引擎，形成飞轮。

评测协议也刚经历一次修正：旧协议只算匹配上的部件对，导致「整物当一个部件」的 naive baseline 全面最优；Particulate 引入未匹配惩罚 + 全展开状态几何对比 + 新指标 OC。**读这条线的论文时，先确认它用哪套协议**，否则数字不可比。

## 5. 演进脉络（2023 → 2026）

1. **2023 及以前：交互与重建**。需要人去开合物体、多状态扫描（Ditto、PARIS、Structure from Action），速度慢、场景受限；
2. **2024：closed-set 学习与 agentic 起点**。CAGE/NAP 学类别内分布；URDFormer/Real2Code 从图像重建 URDF；Articulate-Anything（ICLR 2025）证明 VLM agent + 代码生成可用，成功率 10%→75%；
3. **2025：开放词汇与零训练**。AnyMesh（CoRL 2025）把范围放开到任意 mesh；同期 FreeArt3D、ArtiLatent 把 3D 生成扩散模型直接建在铰接表征上；
4. **2026：前馈化与统一化**。Particulate 把「理解已有网格」做成秒级前馈；FAMOS 引入多状态与程序化数据；SIMART 用 MLLM 统一部件/关节/物理属性——三条 2026 年的线共同把「铰接估计」从 demo 推向基础设施。

贯穿的宏观趋势与 3D 生成主干一致：**逐物体优化 → 前馈大模型**，同时 VLM/MLLM 语义先验从「外挂」走向「骨干」。

## 6. 趋势判断与开放问题

**我当前的判断**（个人推断，随证据更新）：

1. **单状态歧义是信息论层面的，多状态/视频证据终将标配**。FAMOS（几何多状态）与 Articulate-Anything（视频 critic）从两端证明了同一件事。下一步显然是**视频条件化的前馈铰接预测**——把 FAMOS 的聚合架构与视频输入、Particulate 的输出头合并；
2. **「前馈小模型 vs MLLM」正在合流**。Particulate 的 PartField 特征注入已经是「把语义先验喂给小模型」；SIMART 是「把小模型表示喂给 MLLM」。终局大概率是：MLLM 级语义骨干 + 结构化几何输出头 + 前馈速度；
3. **数据飞轮是战略资产**。AnyMesh 式自动打标 → 训练更强前馈模型 → 打标更准更多，这条飞轮里谁先跑起来谁有数据护城河；程序化生成（构造即标注）则是不依赖真实数据的平行供给；
4. **交付物正在从「会动」升级到「sim-ready」**：限位（Particulate）→ 物理属性（SIMART、PhysX-Omni）→ 部件穿透修复/接触一致性（均未解决）。评估一个铰接资产能否进机器人训练，穿透与非穿透部件的碰撞几何是下一个硬指标；
5. **与本人在研方向的接口**：铰接部件的连接区域（connecting area，AnyMesh 的核心几何量）本质是 mesh 拓扑的「功能连接先验」，与 Asset-JEPA 构想中 topology/部件级 latent 的学习高度相关——部件间如何连接、可动性如何，是 mesh 表征里目前缺失的一层语义。

**开放问题清单**：多自由度关节（球铰、螺旋、齿轮）与闭环机构的表示；软性/线缆连接；场景级多物体；test-time compute 用于铰接（多假设生成 + 物理仿真验证择优）；跨数据集限位标注规范。

## 7. 精读笔记导航

| 笔记 | 路线 | 一句话记忆点 |
|---|---|---|
| [[Particulate - 前馈式三维物体铰接]] | 前馈学习 | 轴位置点级投票 + PartField 语义注入，秒级前馈 |
| [[FAMOS - 稀疏多状态观测的铰接建模]] | 前馈 + 多状态 | 观测跨度损失强制跨状态聚合，程序化数据养 OOD |
| [[Articulate AnyMesh - 开放词汇铰接物体建模]] | VLM 流水线 | 连接区域候选点投影 2D 让 GPT-4o 指认铰链 |
| [[Articulate-Anything - VLM驱动的URDF铰接生成]] | VLM agent | actor-critic 渲染闭环写 URDF 代码，75% 成功率 |
| [[SIMART - 静态网格到仿真就绪铰链资产]] | MLLM 统一 | Sparse VQ-VAE 压 token，一次推理出部件+关节+物理 |
| [[PhysX-Omni - 单图生成仿真就绪三维资产]] | 单图生成 | 单图直出刚/柔/关节体 + 物理属性 |
| [[CubePart - 开放词汇部件可控三维生成]] | 部件可控生成 | schema 驱动的多部件生成接口 |
