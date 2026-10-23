---
type: paper-note
status: final
domain: 3D-Generation
paper: "Articulate AnyMesh: Open-Vocabulary 3D Articulated Objects Modeling"
year: 2025
arxiv: "2502.02590"
doi: null
source: https://arxiv.org/abs/2502.02590
project: https://articulate-anymesh.github.io
code: null
venue: CoRL 2025
tags:
  - 3D-Generation
  - Articulated-Object
  - VLM
created: 2026-10-23
updated: 2026-10-23
---

# Articulate AnyMesh：开放词汇铰接物体建模

## 基本信息

- **论文全称**：Articulate AnyMesh: Open-Vocabulary 3D Articulated Objects Modeling（CoRL 2025）
- **作者与机构**：Xiaowen Qiu*、Jincheng Yang*（共同一作）、Yian Wang、Zhehuan Chen、Yufei Wang、Tsun-Hsuan Wang、Zhou Xian、Chuang Gan（MIT-IBM / CMU 系）
- **arXiv**：[2502.02590](https://arxiv.org/abs/2502.02590)（2025-02-04 提交）
- **项目页**：[articulate-anymesh.github.io](https://articulate-anymesh.github.io)
- **一句话标签**：不训练任何铰接模型，用「VLM 语义 + 几何视觉提示」搭一条三段式流水线，把任意来源的静态网格（手工资产、生成网格、真实扫描）开放词汇地变成可动 URDF 资产。

## 一句话结论

Articulate AnyMesh 的立场与 Particulate 相反：**拒绝在稀缺的铰接数据上训练**，把整个任务拆成「开放词汇分割 → 几何线索 + GPT-4o 推理关节 → 生成模型补全」的模块化流水线。它最核心的观察是：**铰链必然位于两个部件的连接区域，且旋转轴与铰链的几何走向一致**——把候选点投到 2D 图上让 GPT-4o「指认」铰链，就能用基础模型的常识替代训练数据。代价同样鲜明：逐物体多轮 VLM 调用（分钟级）、看不见静止状态下被遮挡的内部部件、关节精度受分割质量与 GPT-4o 偏见的双重制约。它是四篇里「泛化性 vs 精度/速度」光谱上最偏泛化的一端。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

具身 AI 与机器人仿真极度缺数据，而**铰接物体数据比静态 3D 资产稀缺得多**。已有的三类路都走不通：整体 3D 生成只能出「看起来像但打不开」的外壳；部件感知生成有部件无运动参数；铰接物体创建方法要么需要多状态的稠密观测，要么受限于训练类别。作者总结铰接资产收集的三个额外门槛：需要几何+语义+关节参数三者齐备、标注成本高、现有数据集规模不足以训练泛化模型。

于是问题被重述为：**能否完全不依赖铰接训练数据，把任意 mesh 变成铰接资产？**答案是把赌注押在 VLM 的常识与可泛化的几何先验上。

### 1.2 本文核心贡献

1. 开放词汇的 mesh→铰接资产自动化流水线（分割/关节/补全三段）；
2. **Geometry-aware Visual Prompting**：利用连接区域几何线索 + 定制视觉提示，让 GPT-4o 做开放词汇关节参数估计；
3. 大规模实验：in-domain 与 OOD 关节估计均超过 URDFormer、Real2Code，并用生成资产增强机器人策略学习（bucket lifting 成功率 33.9%→47.9%）。

![](assets/Articulate%20AnyMesh%20-%20开放词汇铰接物体建模/figures/01-teaser.png)

> 图 1：流水线 teaser。输入覆盖日用品、家具、载具乃至虚构物体（左起烤箱、老虎钳、宝可梦球、冰箱等），输出带可动部件与关节的 URDF 资产。
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2502.02590

---

## 2. 任务定义与输入输出

输入是任意来源的 3D mesh（手工资产 / text-to-3D 或 image-to-3D 生成表面网格 / 真实世界扫描重建）；输出是 URDF 铰接资产，包括：可动部件的 3D 分割及语义标签、每部件的关节类型（限于 revolute / prismatic 两大类）、关节轴与位置。任务链上还有一个**可选的后处理段**处理表面网格的先天缺陷（无内部几何 → 有洞）。

与 Particulate 的任务差异需要注意：AnyMesh 的输入输出对与 Particulate 几乎一致（静态网格 → 铰接结构），但方法族不同（零训练流水线 vs 端到端学习）。Particulate 的对比实验里 AnyMesh 在分割上 gIoU 0.172 vs 0.332（Lightwheel），但没有被拉开数量级——说明 VLM 路线的下限并不差，差在内部部件与速度。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/Articulate%20AnyMesh%20-%20开放词汇铰接物体建模/figures/02-method-pipeline.png)

> 图 2：三段式流水线。A. 可动部件分割（PartSlip++ + VLM 生成部件标签）；B. 铰接估计（连接区域候选点投影到 2D，GPT-4o 视觉提示选点定轴）；C. 后处理（HoloPart 形状补全 + Meshy 纹理生成）。
> 来源：论文 Figure 2，第 4 页，https://arxiv.org/abs/2502.02590

### 3.2 模块 A：VLM 辅助的可动部件分割

采用 PartSlip++ 范式：先用 VLM 看输入 mesh 的图像，**生成该物体的可动部件标签列表**（开放词汇的关键——标签不来自固定词表）；再对多视角渲染图跑开放词汇 grounding（用 set-of-mark 视觉提示改进）+ SAM 得到 2D mask；多视角 2D 检测融合成 3D 部件分割。整段没有任何 3D 训练，语义完全由 VLM 提供。

### 3.3 模块 B：几何感知视觉提示做铰接估计

这是论文的方法论核心。先定义**连接区域（connecting area）**：两个相邻部件 $P_a, P_b$ 的点云中，彼此距离低于阈值的点集（找不到就倍增阈值重试）。铰接估计按关节类型分两路：

**Revolute（旋转）关节**。物理观察：旋转轴与铰链的长度方向一致，而铰链位于连接区域。流程：

1. 对连接区域多次跑不同簇数的 K-means，选能最大化候选点数且 2D 标注不重叠的簇数，簇中心即候选点；
2. 把候选点按编号标注投影到「该部件像素占比最大」的渲染视角上；
3. **GPT-4o 看图选出定义铰链的两个以上点**，两点定轴。

三个兜底设计覆盖 GPT-4o 看不见的情况：铰链只露一端（如旋钮）时，假设轴垂直于连接区域拟合平面，由平面法向 + 单点定轴；铰链完全藏在表面下时，用 $l_0$-cut pursuit 从部件点云（法向+颜色特征）生成超点、补充候选点，让标注覆盖不同几何/纹理区域。

**Prismatic（平移）关节**。滑轨通常藏在内部，铰链式策略失效。先让 GPT-4 把平移关节分成两类再分别处理：

1. **内外滑动**（抽屉、按钮）：滑动方向 ≈ 连接区域拟合平面的法向——「抽出来」的方向就是法向；
2. **表面滑动**（推拉窗、烤面包机拨杆）：方向被约束在表面二维内，在渲染图上标注候选箭头让 GPT-4o 选。

**我的评价**：这一段的本质是**把每个关节参数的估计问题先约化成一个 2D 指认问题**（选点/选箭头/选类别），从而完全避开数值回归——VLM 擅长指认、不擅长出数，这个任务分解是对模型能力边界的清醒认识。代价是每一步都有手工设计的分支逻辑，新关节类型（球铰、螺旋）需要新的提示方案，系统不「生长」。

### 3.4 模块 C：几何与纹理后处理

表面网格（生成/扫描来源）分割出的部件缺内部几何、有洞。用 HoloPart 做部件级形状补全（输入带 3D 分割标签的 mesh，输出补全后的各部件），Meshy 生成纹理。这一段让流水线对「生成网格」自洽——补上了生成式来源最常见的缺陷。

---

## 4. 数据集与实验设置

- **评测**：PartNet-Mobility，分 in-domain（URDFormer 训练用过的 oven/fridge/cabinet/washer/dishwasher 五类）与 out-of-domain（其余类）。指标：角度误差（轴方向）与线距离（轴位置）；
- **基线对齐**：URDFormer 拿前视图 + GT 部件 bbox；Real2Code 拿 GT 部件 mesh 做代码生成；AnyMesh 拿 GT mesh 全流程。即基线拿了「更接近答案」的输入，AnyMesh 从头做；
- **VLM**：GPT-4o（附录有 GPT/Gemini/Claude 的消融）。

---

## 5. 实验结果

### 5.1 关节估计精度

表 1（ID = 上述五类，OOD = 其余）：

| 方法 | ID angle↓ | ID pos↓ | OOD angle↓ | OOD pos↓ |
|---|---|---|---|---|
| URDFormer | 13.674 | 0.144 | 33.864 | 0.299 |
| Real2Code | 7.950 | 0.167 | 24.035 | 0.194 |
| **AnyMesh** | **6.257** | **0.076** | **6.582** | **0.083** |

最值得注意的不是绝对值而是 **ID 与 OOD 的落差**：URDFormer 13.7°→33.9°、Real2Code 8.0°→24.0°，而 AnyMesh 6.3°→6.6° 几乎无衰减。这就是「不训练、靠常识」路线的招牌优势——没有训练分布就没有分布外。

### 5.2 三个应用实验

![](assets/Articulate%20AnyMesh%20-%20开放词汇铰接物体建模/figures/03-generation-results.png)

> 图 3：生成式铰接资产结果。每组从左到右：Meshy 生成的原始表面网格 → HoloPart 补全几何 → 纹理化成品，两个铰接姿态渲染。补全段把「有洞的表面网格」修成可用的部件几何。
> 来源：论文 Figure 3，第 7 页，https://arxiv.org/abs/2502.02590

![](assets/Articulate%20AnyMesh%20-%20开放词汇铰接物体建模/figures/04-real2sim2real.png)

> 图 4：Real-to-Sim-to-Real。真实物体（电钻、微波炉、方向盘）经 2DGS 重建 → AnyMesh 铰接化 → 仿真中运动规划采样轨迹 → 直接迁移到真实机械臂执行。
> 来源：论文 Figure 5，第 7 页，https://arxiv.org/abs/2502.02590

**策略学习增强**（表 2）：在 laptop 与 bucket 两个操作任务上，用 AnyMesh 生成的同类资产增强训练集，bucket 提升成功率 33.9%→47.9%、laptop 40.8%→51.3%——证明流水线产出的资产**对下游机器人学习有实际数据价值**，不只是好看。这是四篇里唯一把「生成的资产真的能帮机器人学策略」做成对照实验的。

### 5.3 与其他方法在 PartNet-Mobility 上的横向数据

Particulate 论文表 2/3 里 AnyMesh（†连通域精修，且运动评估时**被发给 GT 运动范围**）在 Lightwheel 分割 gIoU 0.172、PM 0.383，运动预测 PM gIoU 0.378 vs Particulate 0.843。两边论文放在一起读的结论：AnyMesh 赢在类别覆盖与 OOD，输在内部部件、限位（它根本不预测范围）与速度。

---

## 6. 与相关工作的关系

- **vs Articulate-Anything**（见 [[Articulate-Anything - VLM驱动的URDF铰接生成]]）：同为 VLM 流水线，但 Any-Anything 检索现有部件 mesh 拼装（输出受资产库限制），AnyMesh 分割**用户自己的网格**（保留原几何外观）。前者的 actor-critic 循环可自纠错，后者没有渲染-评估回路、一次定终身；
- **vs URDFormer / Real2Code**：两者都是「训练/微调一个小模型或代码生成器」的闭集路线，AnyMesh 用零训练换 OOD 鲁棒；
- **vs Particulate / FAMOS**：学习路线的镜像。Particulate 论文把 AnyMesh 当主要对比对象并指出其内部部件盲区；AnyMesh 论文则强调学习路线的类别天花板。两边数据合起来恰好构成这个子问题 2025-2026 年「先验来源之争」的完整光谱。

---

## 7. 局限与批判性评价

**作者自述**：(1) 现有与生成的 mesh 普遍缺正确的物理内部结构；(2) 3D 分割偶尔出错并级联到关节估计；(3) GPT-4o 在部分场景有系统性偏见（如「抽屉朝上拔」这类不常见设计不被支持）。

**我的补充判断**（个人推断）：

1. **速度是结构性的**：每物体需要多次渲染 + 多轮 GPT-4o 调用，且无法 batch 化——「开放词汇」的账是用推理成本付的。对万级资产库打标，成本模型完全不同于前馈方法；
2. **误差是级联的**：分割错 → 连接区域错 → GPT-4o 指认错，全程无自纠错回路（对比 Articulate-Anything 的 critic）。流水线越长、单点故障越多；
3. **几何启发式有隐含假设**：连接区域存在、铰链可见或至少一端可见、平移方向可用平面法向近似——对薄壁、软性连接（线缆、布）会失效；
4. 论文没有报告运行时间的系统数字（只说优于逐物体优化方法），这在一篇以「自动化流水线」为卖点的论文里是个遗憾。

---

## 8. 复现与实践建议

- **代码**：项目页提供了数据与部分结果展示，官方实现见 GitHub（articulate-anymesh.github.io 链接的仓库）；核心依赖全是现成组件：PartSlip++、SAM、GPT-4o、HoloPart、Meshy——**这是四篇里最容易拼出可用 pipeline 的一篇**，因为每个模块都有公开 API；
- **成本**：无训练成本，但每物体消耗 GPT-4o 多模态调用若干次（论文未披露具体次数与费用，需实测）；
- **实践建议**：适合「少量、多样、要求不高」的场景：给 Objaverse 物体打标、做 demo、生成机器人训练数据增强。若目标是高精度 sim-ready（限位、质量属性），需接 SIMART 类方法或人工校验。

---

## 9. 个人启发与后续问题

1. **「把回归问题约化成指认问题」**是整篇论文最可迁移的思想：凡是可以用「候选点 + 选择」表达的空间量（轴、方向、位置），都可以投影到 2D 让 VLM 指认。这与 Particulate 的「点级投票」是同一哲学的两个实现——都是避免让模型直接吐数值；
2. **连接区域（connecting area）**是个被低估的几何量：两部件距离最近的区域编码了关节的几乎全部信息。这个量在我关注的 Asset-JEPA / mesh topology 方向也许可以作为部件间「连接性先验」的表征；
3. 遗留问题：VLM 的 3D 推理能力随基座升级自动提升（作者也如此展望），那么这条流水线的精度上限会随时间「免费」上涨——相比之下前馈模型需要重训。长期看谁的曲线更陡，是这个方向最值得跟踪的问题。

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | https://arxiv.org/pdf/2502.02590 | 全文 | 2026-10-23 | 精读主来源 |
| `figures/01-teaser.png` | teaser 图 | 同上 | Figure 1, p.1 | 2026-10-23 | 背景章节 |
| `figures/02-method-pipeline.png` | 方法总览图 | 同上 | Figure 2, p.4 | 2026-10-23 | 核心方法章节 |
| `figures/03-generation-results.png` | 生成结果图 | 同上 | Figure 3, p.7 | 2026-10-23 | 实验章节 |
| `figures/04-real2sim2real.png` | Real2Sim2Real 图 | 同上 | Figure 5, p.7 | 2026-10-23 | 应用实验章节 |

其他来源：[项目页](https://articulate-anymesh.github.io)、[CoRL 2025 OpenReview](https://openreview.net)（ Lacuna 摘要页核对 DOI）。
