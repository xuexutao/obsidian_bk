---
type: paper-note
status: final
domain: 3D-Generation
paper: "Particulate: Feed-Forward 3D Object Articulation"
year: 2026
arxiv: "2512.11798"
doi: null
source: https://arxiv.org/abs/2512.11798
project: https://ruiningli.com/particulate
code: null
venue: CVPR 2026
tags:
  - 3D-Generation
  - Articulated-Object
  - Feed-Forward
created: 2026-10-23
updated: 2026-10-23
---

# Particulate：前馈式三维物体铰接

## 基本信息

- **论文全称**：Particulate: Feed-Forward 3D Object Articulation（CVPR 2026，pp. 27708-27718）
- **作者与机构**：Ruining Li、Yuxin Yao（共同一作）、Chuanxia Zheng、Christian Rupprecht、Joan Lasenby、Shangzhe Wu、Andrea Vedaldi；Oxford / Cambridge / NTU
- **arXiv**：[2512.11798](https://arxiv.org/abs/2512.11798)
- **项目页**：[ruiningli.com/particulate](https://ruiningli.com/particulate)（代码未开源，截至笔记撰写时）
- **一句话标签**：给定一个静态 3D 网格，用 Transformer 一次前馈推理预测完整的铰接结构——部件分割、运动学树、关节类型/轴/限位——数秒级推理，并可直接导出 URDF 进物理仿真。

## 一句话结论

Particulate 把「给静态网格加铰链」这件事从逐物体优化（VLM 循环、多状态拟合，动辄十几分钟）搬进了**单次前馈 Transformer**：以网格表面点云 + 法向 + 冻结 PartField 语义特征为输入，用可学习 part query 做部件级的 DETR 式匹配，多头解码分别输出分割、运动学树（软邻接矩阵 + Edmonds 最大生成树）与运动参数；其中**旋转轴采用「方向直接回归 + 点级投票轴位置」的过参数化设计**，是它轴定位精度高于直接回归的关键。在自建的 Lightwheel 基准上，它对 Articulate AnyMesh、SINGAPO 等方法取得了明显优势，并且能直接吃 AI 生成的网格，构成「文本/图像 → 静态网格 → 可动资产」的一站式流水线。局限也很明确：训练数据只有几千个物体，运动学结构超出训练分布就失效，且生成资产存在部件穿透。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

现实中的大多数物体不只由形状定义，还由「能怎么动」定义：柜门绕铰链旋转、抽屉沿滑轨平移、笔记本绕转轴开合。机器人操作、游戏交互、数字孪生都需要机器自动理解这种**铰接结构（articulated structure）**。论文考虑的具体任务是：**给定单个静态 3D 网格 $M=(V,F)$，推断其完整铰接结构**——哪些面属于哪个可动部件、部件之间的运动学树、以及每个关节的类型/轴/限位。

这个任务定位非常聪明：它不与 3D 生成器竞争，而是**互补**。静态 mesh 生成（TRELLIS、Hunyuan3D 等）已经非常成熟，缺的正是「让生成的物体动起来」这最后一公里。Particulate 补的就是这一公里。

### 1.2 现有方法的瓶颈

作者把已有路线分成三类，各自的瓶颈都很清楚：

1. **程序化/规则生成**：无法扩展到真实世界长尾物体；
2. **3D 部件分割方法**（PartField、P3SAM 等）：只预测语义部件分割，不建模部件间的铰接关系，而且「语义部件」与「铰接部件」的定义并不一致；
3. **铰接物体生成/估计方法**（CAGE、ArtFormer、SINGAPO 等）：通常只在少数类别上训练、假设运动学结构已知、或依赖部件检索拼装；基于 VLM 的方法（Articulate AnyMesh）虽然开放词汇，但需要逐物体多轮优化，十几分钟才能出一个结果，且**无法处理静止状态下被遮挡的内部部件**（例如关着的微波炉内部结构，VLM 从外面看不见）。

### 1.3 本文核心贡献

1. **Particulate 框架**：前馈式预测完整铰接结构（分割 + 运动学树 + 运动参数），可直接导出 URDF；
2. **对未见物体与 AI 生成网格的泛化**：与图像到 3D 生成器组合后，实现从单图到可动资产的端到端生成；
3. **Lightwheel 基准**：243 个高质量手工精标的铰接物体（14 类，CC-BY-NC），并重新设计了更符合人类偏好的评测协议（对未匹配部件施加惩罚的 gIoU/mIoU/PC）。

![](assets/Particulate%20-%20前馈式三维物体铰接/figures/01-teaser.png)

> 图 1：Particulate 的 teaser。左列为输入静态网格（多由 Hunyuan3D 生成），中列为预测的部件分割与关节轴，右列为按预测运动约束展开后的铰接状态。覆盖家电、卫浴、乐器等类别，强调「单次前馈、秒级推理」。
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2512.11798

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

输入是网格 $M=(V,F)$；输出是铰接结构的四元组 $A=(P, S, K, \mathcal{M})$：

- $P \in \mathbb{N}$：部件数量；
- $S: [|F|] \to [P]$：面到部件的分割；
- $K \subseteq [P]\times[P]$：运动学树。假设每个物体有一个**基础部件** $b$（如微波炉外壳），边从基础部件向外定向，$K$ 构成以 $b$ 为根的树形图（arborescence）；
- $\mathcal{M} = (\mathsf{M_{tp}}, \mathsf{M_{pd}}, \mathsf{M_{ra}}, \mathsf{M_{pr}}, \mathsf{M_{rr}})$：运动约束，分别为每部件的运动类型（fixed / prismatic / revolute / both）、平移方向、旋转轴（方向 + 轴上一点）、平移范围与旋转范围。

关键假设与边界：物体是**部分刚性**的（部件内不形变）；运动学树有唯一基础部件；旋转/平移都是单自由度（或两者的组合）。这套参数化可以无损转换成 URDF。

### 2.2 与 FAMOS/SIMART 的符号差异

值得注意（个人备注）：Particulate 的 $\mathcal{M}$ 把**运动范围（range）**作为一等公民显式预测（$[-l_{\min}, l_{\max}]$、$[-\theta_{\min}, \theta_{\max}]$），而同期的 FAMOS 只监督「观测到的运动跨度」、SIMART 输出 URDF 属性。限位预测是 Particulate 相对完整的一点——机器人仿真里关节限位错了，策略学出来就是错的。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/Particulate%20-%20前馈式三维物体铰接/figures/02-method-pipeline.png)

> 图 2：Part Articulation Transformer（PAT）总览。输入网格采样为点云，每个点由坐标 XYZ、法向 Normal、冻结 PartField 特征经 MLP 求和得到 token；一组可学习 part queries 与点 token 交替做 query 自注意力与 query→point 交叉注意力；随后多个解码头分别输出部件分割、运动学树、关节类型/轴/范围。
> 来源：论文 Figure 2，第 4 页，https://arxiv.org/abs/2512.11798

整体是一个 DETR 风格的 query-based Transformer，但把「检测目标」换成了「铰接部件」：

**输入 token 化**。对网格表面采样点云 $P=\{p_i\}$，每个点附带法向 $n_i$ 与 PartField 特征 $f_i$。PartField 是在 2D 渲染上训练的语义部件特征提取器，给网络提供「这个局部区域属于哪个功能部件」的全局语义先验。三个输入各自过 MLP 映射到维度 $D$ 后相加：$\tilde{p}_i = \text{MLP}_p(p_i) + \text{MLP}_n(n_i) + \text{MLP}_f(f_i)$。

**Part queries**。初始化 $P_{\max}=16$ 个可学习 query $Q=\{q_j\}$，远大于典型物体的真实部件数 $P$，多余的 query 在推理时自然空出（没有任何点分配给它）。

**注意力结构**。$B=8$ 个注意力块，交替两种注意力：(a) query 之间的自注意力——协调各 query 分别代表哪个部件；(b) query→point 交叉注意力——让 query 从点云中收集证据。**点 token 之间不做自注意力**：因为 $N \gg P_{\max}$，点间自注意力的 $O(N^2)$ 开销不可接受。这是一个非常务实的复杂度取舍，也是它能吃 10 万级点云（推理时 102,400 点）的原因。

### 3.2 旋转轴的过参数化设计

这是论文里我认为最精巧的工程细节。旋转轴 $\tilde{M}_{ra}^i = (\tilde{d}_{ra}^i, \tilde{x}_{ra}^i) \in S^2 \times \mathbb{R}^3$，方向好学（大多与坐标轴对齐），**位置难学**——直接用一个 MLP 回归 6 维 Plücker 坐标会得到系统性偏移的轴（作者推测是数据量不大导致的过拟合）。Particulate 的解法是**分解 + 投票**：

1. **方向**由 part query 直接回归：$\tilde{d}_{ra}^i = h_{rd}(\tilde{q}_i)/\|h_{rd}(\tilde{q}_i)\|_2$；
2. **位置**由每个属于该部件的 3D 点「投票」：MLP $h_{cp}$ 接收点 token 与部件 token，预测该点到轴的**正交投影点**

$$\tilde{x}_j^i := \underset{x \text{ on the revolute axis}}{\arg\min}\ \|x - p_j\|_2 = h_{cp}(\tilde{p}_j, \tilde{q}_i)$$

推理时取该部件所有投票点的逐坐标**中位数**作为轴位置：

$$M_{ra}(i) = \left(\tilde{d}_{ra}^i,\ \text{median}(\{h_{cp}(\tilde{p}_j, \tilde{q}_i) : \tilde{s}_j = i\})\right)$$

直觉：单点预测轴位置是高风险回归，但「每个点都预测自己到轴的垂足」把问题变成了几十万次冗余预测的鲁棒聚合，误差在统计上相互抵消。消融（表 4 A vs C）显示这一设计在 Lightwheel gIoU 上贡献了 0.236→0.259。

### 3.3 训练目标与损失函数

**DETRE 式二分匹配**。由于 query 无固有身份，训练时用匈牙利算法将 $P$ 个 GT 部件注入式匹配到 $P_{\max}$ 个 query。匹配准则是**点级分割分布的对齐**（对分割 logits 做列置换后的最大似然）：

$$\hat{\pi} = \arg\max_{\pi} \sum_{i=1}^{P}\sum_{j=1}^{N} S_{j,i} \log \frac{\exp(\tilde{S}_{j,\pi(i)})}{\sum_k \exp(\tilde{S}_{j,k})}$$

**多任务损失** $L = L_S + L_K + L_M$：分割交叉熵、运动学树 BCE、运动约束五项（类型 CE + 平移/旋转范围 L1 + 平移方向 L1 + 轴方向 L1 + 轴位置 L1，其中旋转范围权重 0.1）。全部端到端联合训练，没有阶段化。

### 3.4 推理流程

1. 采样点云（保证每个面至少一个点），前馈得到每点的部件归属；
2. 面按多数点投票归属部件，活跃 query 集合即部件集合；
3. 运动学树：对软邻接矩阵 $\tilde{K} \in \mathbb{R}^{P_{\max}\times P_{\max}}$ 跑 **Edmonds 算法取最大生成树形图（MSA）**——把「树结构推断」变成图上的组合优化，而不是让网络直接输出合法树；
4. 运动参数直接从对应 query 读出，轴位置按 3.2 的中位数聚合；
5. 对手工资产可选做**连通域精修**：同一连通域的面强制归入同一部件（覆盖率最大的那个），因为手工建模的部件通常是连通的。

整个流程**数秒级**，对比 Articulate AnyMesh 的逐物体 VLM 多轮循环（分钟级到十分钟级）是量级差异。

---

## 4. 数据集与实验设置

### 4.1 数据与实现

- **训练集**：PartNet-Mobility（沿用 SINGAPO 划分）+ GRScenes 全部铰接资产，过滤掉超过 16 部件的物体，共 **3,800 个物体、50 个类别**——注意这个量级比图像/静态 3D 数据集小好几个数量级，是全文所有泛化局限的根源；
- **训练配置**：N=2048 点（50% 均匀采样 + 50% 二面角 >30° 的锐边采样，后者保护铰链、棱边等细结构），150M 参数，8×H100 训练 100K 步，batch 128；
- **评测集**：PartNet-Mobility 测试集（7 类）+ 自建 Lightwheel 基准（243 物体、14 类，标注更细、类别更多样）。

### 4.2 评测协议的一个重要修正

论文指出先前工作（NAP、CAGE、SINGAPO）的分割指标只算**匹配上的部件对**、忽略未匹配部件——在这个协议下，「把整物当一个部件」的 Naive Baseline 竟然全面优于所有方法，说明旧协议对漏检小部件几乎没有惩罚。Particulate 的修正是：对任何未匹配部件施加惩罚项（gIoU 记 -1、mIoU 记 0、PC 记包围盒对角线一半），即式 (10) 的双向平均。这个协议修正是对整个子领域评测方式的有益纠偏，也是 Lightwheel 基准的组成部分。

---

## 5. 实验结果

### 5.1 主要定量结果

**部件分割**（表 2，含连通域精修 †）：

| 方法 | Lightwheel gIoU↑ | Lightwheel mIoU↑ | PartNet-Mobility gIoU↑ | PM mIoU↑ |
|---|---|---|---|---|
| Naive Baseline | 0.018 | 0.413 | 0.296 | 0.612 |
| PartField†（给定 GT 部件数） | 0.079 | 0.264 | 0.183 | 0.361 |
| P3SAM†（给定 GT 部件数+点提示） | 0.122 | 0.411 | -0.116 | 0.267 |
| SINGAPO†（1@10） | -0.050 | 0.297 | 0.271 | 0.471 |
| Articulate AnyMesh† | 0.172 | 0.452 | 0.383 | 0.542 |
| **Particulate†** | **0.332** | **0.576** | **0.880** | **0.884** |

两点值得展开：其一，PartField/P3SAM 拿到了 GT 部件数等特权信息仍大幅落后，说明「语义部件」与「铰接部件」确实是两种不同的监督目标，拿分割模型做铰接任务天然不对口；其二，PartNet-Mobility 上 gIoU 0.880 对 Lightwheel 上 0.332 的巨大落差，反映的是训练分布内 vs 分布外的真实差距——Lightwheel 的部件标注更细、物体更多样。

**运动预测**（表 3，把所有部件移到最大行程后与 GT 全展开状态比几何，Particulate 甚至比拿了 GT 运动范围的 Articulate AnyMesh 更准）：Lightwheel OC 0.008 vs AnyMesh 0.010；PartNet-Mobility gIoU 0.843 vs 0.378。注意 PartNet-Mobility 上 AnyMesh 只有 0.378，与分割表里的 0.383 一致——它的 VLM 流水线在「看不见内部结构」时系统性失效。

![](assets/Particulate%20-%20前馈式三维物体铰接/figures/04-comparison.png)

> 图 4：Lightwheel 基准上的定性对比（右列起：PartField、P3SAM、SINGAPO、Articulate AnyMesh、Ours、GT）。语义分割方法（PartField/P3SAM）的分割边界与铰接部件不对齐；SINGAPO 类别外泛化差；Articulate AnyMesh 在内部结构不可见时失败（如 b、d 行的抽屉内部）；Particulate 恢复了包括内部部件在内的大部分铰接结构。
> 来源：论文 Figure 4，第 7 页，https://arxiv.org/abs/2512.11798

### 5.2 定性结果

![](assets/Particulate%20-%20前馈式三维物体铰接/figures/03-qualitative.png)

> 图 3：对 Hunyuan3D 生成网格的预测结果。每组三列：输入网格、预测分割、按预测约束展开的铰接态。训练只用美术资产，但对 AI 生成网格泛化良好——这是「文本/图像 → 生成网格 → 自动加铰链」流水线的可行性证明。
> 来源：论文 Figure 3，第 7 页，https://arxiv.org/abs/2512.11798

### 5.3 消融实验

表 4 在 Lightwheel 子集上消融了三个设计（铰接态指标）：

| 配置 | gIoU↑ | OC↓ |
|---|---|---|
| A：全量（PM+GRS + PartField + 轴过参数化） | **0.259** | **0.008** |
| B：只用 PartNet-Mobility | 0.231 | 0.013 |
| C：去掉轴过参数化（直接回归 6D） | 0.236 | 0.008 |
| D：去掉 PartField 特征 | 0.174 | 0.009 |

三个结论：数据越多越好（但 B 仍大幅超越基线，说明增益主要来自架构与训练方式本身）；轴过参数化有效；**PartField 特征是最大单项贡献**（去掉后 gIoU 掉 0.085）——语义先验弥补了几千个训练样本远不足以学出的功能理解。

### 5.4 失败案例与效率

![](assets/Particulate%20-%20前馈式三维物体铰接/figures/05-failure-cases.png)

> 图 5：失败案例。对运动学结构明显超出训练分布的物体（如多头同轴、非常规连接方式），模型给出错误或不完整的结构。
> 来源：论文 Figure 5，第 13 页，https://arxiv.org/abs/2512.11798

推理耗时数秒；训练成本 8×H100 × 100K 步，按当前价格属于普通实验室可负担的范围，但**代码未开源**，复现只能等。

---

## 6. 与相关工作的关系

- **vs Articulate AnyMesh**（见 [[Articulate AnyMesh - 开放词汇铰接物体建模]]）：AnyMesh 靠 VLM 的开放词汇能力换取覆盖面，代价是逐物体优化慢、看不见内部部件；Particulate 靠学习换取速度与内部部件感知，代价是受训练分布限制。两者的失效模式几乎完全互补——AnyMesh OOD 强 ID 内部弱，Particulate 反之。
- **vs SINGAPO**：SINGAPO 从单图生成 bbox 再检索部件几何，同样受类别限制；Particulate 直接作用于输入网格，保留原几何与外观，这是本质优势——生成的资产不需要「长得像库里某个部件」。
- **vs FAMOS**（见 [[FAMOS - 稀疏多状态观测的铰接建模]]）：FAMOS 明确把 Particulate 当作直接前驱（其多状态版本即以 Particulate 架构为 baseline 重训），核心增量是「多状态输入 + 全局注意力 + 观测跨度监督」。
- **vs SIMART**（见 [[SIMART - 静态网格到仿真就绪铰链资产]]）：SIMART 用 MLLM 一次性输出部件 + 关节 + 物理属性（密度/摩擦/尺度），覆盖面比 Particulate 的纯运动学更宽，但依赖 MLLM 推理，速度与可扩展性不同路。
- **vs MagicArticulate / MeshArt**：那是自回归「生成」骨架与部件，属于另一个子方向（从零生成 vs 理解已有网格），Particulate 刻意不做生成。

---

## 7. 局限与批判性评价

**作者自述**：(1) 训练数据比其他领域小几个数量级，运动学结构偏离训练分布时无法恢复（图 5）；(2) 与生成器组合产生的资产常有**部件穿透**——生成网格本身的缺陷 + 运动预测误差叠加，物理合理性不足以支撑大规模 sim-to-real。

**我的补充判断**（个人推断，已标注）：

1. $P_{\max}=16$ 的硬上限把复杂机械（多级传动、并联结构）直接排除在任务定义之外，而真实机器人环境里超过 16 个可动部件的资产并不罕见；
2. 旋转/平移单自由度的参数化覆盖了绝大多数家具家电，但齿轮、球铰、柔索这类多自由度/非刚性连接不在表达范围内——这是整个「URDF 范式」的共同边界，不独此篇；
3. Lightwheel 基准由 Lightwheel 公司出品且 CC-BY-NC 授权，学术可用性没问题，但「自建基准 + 自定义协议 + 自家最优」的组合需要后续第三方工作在协议上跟进才能坐实；
4. 方向上它与 [[SIMART - 静态网格到仿真就绪铰链资产]] 代表的 MLLM 路线是「学习小模型 vs 大模型零样本」的经典对赌。以 2026 年中的进展看，前馈小模型赢在速度与内部部件，MLLM 赢在语义与泛化；真正的赢家可能是把 PartField 式语义特征、MLLM 式常识与 Particulate 式结构化输出头拼起来的混合体。

---

## 8. 复现与实践建议

- **代码**：截至 2026-10 未开源，项目页仅展示结果与 benchmark 申请入口。想复现需要自己实现：DETR 式分割匹配、Edmonds MSA、点级轴投票均为成熟组件，工程量不大；
- **数据**：PartNet-Mobility 公开；GRScenes 需向苏州图书馆/相关方申请；Lightwheel 基准 CC-BY-NC 可申请；
- **依赖**：PartField 特征是关键输入，其开源实现可直接使用；
- **算力**：8×H100（或等量 A100 更长时间）×数天，普通实验室可负担；
- **实践建议**：若目标是给自有资产加铰链且能接受分钟级延迟，AnyMesh 更省事；若要批量处理（例如给整个 Objaverse 子集打标、或做 sim 数据流水线），Particulate 的前馈架构是唯一现实选择。

---

## 9. 个人启发与后续问题

1. **「分析而非生成」的正交性策略**值得记住：当生成侧竞争白热化时，抓住生成结果的「可用性缺口」（这里是不动）做分析侧补全，反而能与所有生成器形成上下游绑定。这与我关注的 mesh 拓扑质量方向同构——生成器出 mesh，wiring quality 是「能不能用」的缺口；
2. **轴位置的点级投票**是「难回归问题 → 易分类/易投影问题 + 鲁棒聚合」的漂亮案例，同类技巧或许能用在关节限位、接触点估计上；
3. **过参数化缓解小数据过拟合**的观察（直接回归 Plücker 过拟合、投票聚合不过拟合）与我最近关注的训练动力学里「低秩退化/谱结构」话题有暗合——冗余参数化 + 平均化本身就是一种隐式正则；
4. 遗留问题：多状态观测到底能换来多少（FAMOS 的答案见 [[FAMOS - 稀疏多状态观测的铰接建模]]）？语义与运动学两种部件定义能否在表征层统一（PartField 特征只做输入，未做联合训练）？

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | https://arxiv.org/pdf/2512.11798 | 全文 | 2026-10-23 | 精读主来源 |
| `figures/01-teaser.png` | teaser 图 | 同上 | Figure 1, p.1 | 2026-10-23 | 背景与贡献章节 |
| `figures/02-method-pipeline.png` | 方法总览图 | 同上 | Figure 2, p.4 | 2026-10-23 | 核心方法章节 |
| `figures/03-qualitative.png` | 定性结果图 | 同上 | Figure 3, p.7 | 2026-10-23 | 实验定性结果 |
| `figures/04-comparison.png` | 对比图 | 同上 | Figure 4, p.7 | 2026-10-23 | 基线对比分析 |
| `figures/05-failure-cases.png` | 失败案例图 | 同上 | Figure 5, p.13 | 2026-10-23 | 局限分析 |

其他来源：[项目页](https://ruiningli.com/particulate)（CVF Open Access 版本核对页码：[openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Particulate_Feed-Forward_3D_Object_Articulation_CVPR_2026_paper.html)）。
