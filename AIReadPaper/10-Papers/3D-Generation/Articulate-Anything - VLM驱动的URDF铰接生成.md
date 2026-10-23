---
type: paper-note
status: final
domain: 3D-Generation
paper: "Articulate-Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model"
year: 2025
arxiv: "2410.13882"
doi: null
source: https://arxiv.org/abs/2410.13882
project: https://articulate-anything.github.io
code: https://github.com/vlongle/articulate-anything
venue: ICLR 2025
tags:
  - 3D-Generation
  - Articulated-Object
  - VLM
  - Actor-Critic
created: 2026-10-23
updated: 2026-10-23
---

# Articulate-Anything：VLM 驱动的 URDF 铰接生成

## 基本信息

- **论文全称**：Articulate-Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model（ICLR 2025）
- **作者与机构**：Long Le、Jason Xie、William Liang、Hung-Ju Wang、Yue Yang、Yecheng Jason Ma、Kyle Vedder、Arjun Krishna、Dinesh Jayaraman、Eric Eaton（University of Pennsylvania）
- **arXiv**：[2410.13882](https://arxiv.org/abs/2410.13882)
- **项目页 / 代码**：[articulate-anything.github.io](https://articulate-anything.github.io) / [github.com/vlongle/articulate-anything](https://github.com/vlongle/articulate-anything)（**四篇中唯一开源的**）
- **一句话标签**：文本/图像/视频输入 → VLM agent 写 Python 代码 → 编译成 URDF → 仿真渲染回喂 critic 自纠错，把 PartNet-Mobility 自动铰接成功率从 ~10% 拉到 75%。

## 一句话结论

Articulate-Anything 是「agentic 流水线」路线的代表作：它不分割、不回归、不检索几何，而是**让一群 Gemini Flash-1.5 agent 写代码拼装 URDF**——mesh retrieval 找部件、link placement 摆位置、joint prediction 定运动，后两段各配一个 actor-critic 闭环，critic 看渲染结果与输入的差异打「真实感分」，低于 5 分就让 actor 改代码，直到收敛。它最重要的设计判断有两个：**直接生成 XML URDF 太脆（冗长 + 需要现场算运动学），改生成高层 Python API 调用**；**视频输入能消解静态输入的铰接歧义**（窗子是推拉还是外翻，动了才知道）。成功率 75% 对比此前 8.7-12.2% 是数量级的提升，但它拼出来的是「检索部件的重组物」，几何与外观受限于资产库，与 AnyMesh/Particulate「保留原网格」的路线是不同的产品形态。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

可交互铰接 3D 资产对 AR/VR、动画、机器人仿真都是刚需，但人工建模 URDF 需要大量专业知识与工时。作者要的是一个**从人类最容易提供的输入（文本/图像/视频）到仿真可用的 URDF** 的端到端自动流水线，且能处理开放多样的物体。

### 1.2 问题形式化

给定输入 $x$（文本/图像/视频），假设存在人类认可的 GT URDF $u^*$，目标是最小化：

$$L_{total} = L_{mesh} + L_{link} + L_{joint}$$

- $L_{mesh}$：各部件点云的 Chamfer 距离（外观/几何差异）；
- $L_{link}$：部件 3D 位姿差（位置 L2 + 四元数朝向差）；
- $L_{joint}$：关节四要素之差（类型交叉熵 + 轴角度差 + 原点位姿差 + 限位差）。

关键的现实困难：$u^*$ 通常拿不到，传统数值优化无法运行。作者的替代方案是**用 VLM agent 从视觉输入直接近似这些损失**（打分），并用 VLM 提议解（代码）+ 迭代精化——即把「优化」整体替换成「语言模型反馈回路」。这是一个范式层面的决定：不是用 VLM 做感知前端，而是用 VLM 做整个优化器。

### 1.3 本文核心贡献

1. 多模态输入（文本/图像/视频）的自动铰接 agent 系统；
2. actor-critic VLM 闭环 + 高层 Python→URDF 编译层；
3. PartNet-Mobility 上成功率 8.7-12.2% → 75%，并用生成的数字孪生训练机器人策略零样本迁移到真实机械臂（关马桶盖、合笔记本、开微波炉、关抽屉）。

---

## 2. 任务定义与输出表示

输出强制为 **URDF**：link-joint 层级树，link 是部件（门、抽屉），joint 主要是 prismatic（平移）与 revolute（旋转）两类。选择 URDF 而非自定义格式的理由很实际：标准 3D 仿真器（Isaac、Sapien、PyBullet）直接可用，生成的资产立即能进机器人训练循环。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/Articulate-Anything%20-%20VLM驱动的URDF铰接生成/figures/02-method-pipeline.png)

> 图 2：系统总览。输入（文本/图像/视频）经三阶段：(1) Mesh Retrieval 从资产库为每个部件找 mesh；(2) Link Placement 的 actor 写摆放代码、critic 对照输入图打分反馈；(3) Joint Prediction 的 actor 写关节代码、critic 对照输入视频与仿真动画评估。全流程产出可编译为 URDF 的 Python 代码。
> 来源：论文 Figure 2，第 4 页，https://arxiv.org/abs/2410.13882

系统由多个专用 VLM agent 组成，每个 agent 是一个带不超过 20 个 in-context 示例的 Gemini Flash-1.5 提示。所有 agent 的产物都是**调用自研 API 的 Python 代码**（`add_link`、`place_relative_to`、`make_revolute_joint`、`make_prismatic_joint` 等），由系统编译成 URDF 并在 Sapien 中渲染、PyBullet 中做物理检查。

### 3.2 为什么是 Python 代码而不是直接 URDF/XML

这是全文我认为最有洞察力的工程判断，理由有二：

1. **URDF/XML 冗长**，长上下文里 LLM 幻觉概率高；
2. **运动学计算无法回避**：比如让一扇门绕全局 z 轴旋转，需要沿 URDF 树做正向运动学把全局轴换算到相对坐标系——让 LLM「在 XML 里现场算数学」是灾难；而「把马桶盖放在马桶上方」这种看似简单的摆放，因为 mesh 几何的不规则，实际上需要物理引擎里的迭代碰撞检查才能做对。

把这些交给**代码层之下的确定性系统**（API 内部处理坐标变换、碰撞求解），LLM 只负责高层决策（选哪个 mesh、放在哪个方位、轴穿哪两个点）。这个「LLM 出意图、确定性系统出精度」的分工是整篇论文成功的基础。

### 3.3 Mesh Retrieval

**视觉输入**：VLM 先检测目标物体（视频利用运动线索辅助），CLIP 相似度缩小到 top-k 候选类别，再由 Hierarchical ObjectSelector agent 用**锦标赛式分组选优**（每批 max_num_images 张递归收敛到一个）从 PartNet-Mobility 资产库里挑出视觉最像的模板物体。

**文本输入**：TaskSpecifier 先把稀疏 prompt 细化，LayoutPlanner 描述每个部件的语义与尺寸，然后按部件级 CLIP 文本-描述相似度检索 mesh 并缩放到指定尺寸。注意他们**自己用 VLM 重新标注了资产库每个 mesh 的描述**（材料、形状、功能），因为原始数据集描述太粗。

### 3.4 Link Placement 与 Joint Prediction 的 actor-critic 闭环

![](assets/Articulate-Anything%20-%20VLM驱动的URDF铰接生成/figures/03-actor-critic.png)

> 图 4：两个闭环的工作实例（门柜示例）。上：link placement 中 critic 发现「门被放在了侧面而不是正面」，评分 4/10，actor 修正后得 10/10；下：joint prediction 中 critic 指出「按本系统约定，向外开的门轴必须取 [0,0,-1] 而非 [0,0,1]」，actor 改轴后收敛。
> 来源：论文 Figure 4，第 6 页，https://arxiv.org/abs/2410.13882

- **Link actor**：按父-子部件轴线对齐 + 碰撞检查收紧摆放；**Link critic** 对照输入图像与仿真渲染，输出差异描述、错误定位（精确到 actor 代码的哪一行）、0-10 真实感分；低于 5 分则回炉；
- **Joint actor**：在摆放代码上追加关节代码；**Joint critic** 对照输入视频与预测关节运动的仿真动画，按「类型错（最严重）→ 原点错 → 轴错 → 限位错（最轻）」分级定位错误，同样打分循环；
- **Targeted affordance extraction**（可选）：不想生成全部关节时，从视频里指认关心的那个部件，省 token 也好调试。

这套闭环本质上是把传统优化的「loss → 梯度 → 更新」替换成「渲染 → 语言描述的残差 → 代码修改」。没有梯度，但**反馈的带宽远高于标量 reward**——critic 能说出「门在侧面而不是正面」这种可执行的修正方向，这是标量优化信号做不到的。

### 3.5 视频输入消解铰接歧义

静态输入下「椅腿是升降滑轨还是固定」「窗是推拉还是外翻」都有多种合理解释（AnyMesh/Particulate 的单状态歧义，见 [[Particulate - 前馈式三维物体铰接]]、[[FAMOS - 稀疏多状态观测的铰接建模]]）。Articulate-Anything 的处理最直接：把演示视频给 critic，让它比对仿真动画与真实运动。论文的 in-the-wild 实验显示，text/image 模态会给出「合理但错误」的预测（椅子升降、窗户内翻），视频模态全部纠正——**歧义在信息层面就不存在于视频里**。

---

## 4. 数据集与实验设置

- **数据**：PartNet-Mobility（约 2.3K 物体、1.9K revolute + 7.6K prismatic 关节的人工标注）；
- **任务**：把 GT URDF 的 link/joint 信息全部抹掉，考察重建能力——(1) link placement（位置阈值 50mm）；(2) joint prediction（类型/轴/原点/限位全部达标才算成功；角度阈值 0.25 rad ≈ 14.3°）。若 link placement 错，该物体全部关节记错——**严格口径**；
- **基线**：URDFormer（图像直出离散化坐标）与 Real2Code（OBB 输入 + 代码生成），均在五个类别上训练/微调过，评测分 ID（这五类）与 OOD（其余 41 类）；
- **实现**：Gemini Flash-1.5、PyBullet 物理、Sapien 渲染、CoTracker 做视频运动标注。

---

## 5. 实验结果

### 5.1 主要结果

joint prediction 成功率：**Articulate-Anything 75.0%（全部类别，不区分 ID/OOD）**，对比 URDFormer 与 Real2Code 的 8.7-12.2%（官方口径：从 8.7-12.2% 提升至 75%，ID 类别上基线约 8-14%、OOD 类别普遍个位数）。且 VLM 方法根本不做 ID/OOD 区分——没有训练分布就没有分布外，与 AnyMesh 的观察一致（见 [[Articulate AnyMesh - 开放词汇铰接物体建模]]）。

### 5.2 输入模态消融

link placement 成功率：text 68.6% → image 77.7%；joint prediction：text 48.9% → image 54.4% → **video 88.6%**。两个结论：视觉锚定对摆放关键（纯文本连「水龙头把手在顶还是在侧」都无法决定——这不是模型弱，是问题本身欠定）；视频对关节预测是决定性增量（+34 个点）。

### 5.3 In-the-wild 定性结果

![](assets/Articulate-Anything%20-%20VLM驱动的URDF铰接生成/figures/04-in-the-wild.png)

> 图 7：wild 场景重建对比。行：输入文本/图像/视频，以及 Ours(text/image/video/video+generation)、URDFormer、Real2Code 的输出；绿框正确、红框错误。视频模态在杂乱随手拍输入下全胜，基线对轻微错位与分割噪声极敏感（URDFormer 几乎全部输出抽屉状结构）。
> 来源：论文 Figure 7，第 8 页，https://arxiv.org/abs/2410.13882

### 5.4 机器人应用

用生成的数字孪生在仿真中训练 PPO 策略（Robosuite，2M episodes × 3 seeds），零样本迁移到 Franka：关马桶盖、合笔记本、开微波炉、关抽屉——超越 pick-and-place 的细粒度交互任务。这是「自动铰接 → 仿真训练 → 真机部署」完整闭环的早期示范。

---

## 6. 与相关工作的关系

- **vs Articulate AnyMesh**（见 [[Articulate AnyMesh - 开放词汇铰接物体建模]]）：同为 VLM 流水线。AnyMesh 分割用户网格、保留原几何；Articulate-Anything **检索拼装**，输出受限于资产库但自带 critic 自纠错回路。前者适合「给已有资产加铰链」，后者适合「从描述/视频造数字孪生」；
- **vs URDFormer / Real2Code**：它们是「训练一个感知/代码模型」的路线，输入简化（裁剪图、OBB 文本），开环无自纠错，OOD 崩溃；
- **vs Particulate / FAMOS**：学习路线与 agentic 路线的正面对比。Particulate 论文实测其分割/运动精度显著更高、速度快几个量级；Articulate-Anything 赢在输入模态灵活（视频）、可自纠错、以及语义理解（知道「关马桶盖」任务需要什么）。

---

## 7. 局限与批判性评价

论文对失败的讨论散在附录与图 7，我的批判性评价：

1. **输出是「拼装的近似物」**（个人判断）：mesh retrieval 决定了资产几何上限——库里没有的形状就做不出来，$L_{mesh}$ 只能近似。这与「保留用户网格」的三篇形成产品形态分歧，适合数字孪生原型、不适合精确重建；
2. **成本与不确定性**：每个物体多次 Gemini 调用 + 每轮 critic 一次渲染，时延与费用都高于前馈方法（论文未系统报告时延/成本，与 AnyMesh 同样的披露缺失）；75% 成功率意味着每四个物体仍有一个需要人工介入；
3. **critic 的天花板就是 VLM 的感知天花板**：分不清左开右开、看不清细小轴位置时，自纠错回路会把 actor 带向错误的「共识」。附录的混淆矩阵分析显示 critic 对某些关节类型系统性混淆；
4. 阈值口径（50mm / 0.25 rad）较宽，跨论文对比数字时要小心（Particulate 的 5° 轴误差标准严得多）。

---

## 8. 复现与实践建议

- **代码**：**已开源**（GitHub），agent prompt、API、评测全部可查——四篇中唯一可直接上手复现的，工程参考价值最高；
- **依赖**：Gemini API（可换其他 VLM 但需重调 prompt）、PyBullet、Sapien、CoTracker；
- **成本**：无训练，推理期每物体若干轮 VLM + 渲染循环，论文未披露量化成本；
- **实践建议**：想给机器人仿真造数字孪生、输入是自然语言或手机视频，这是当前最完整的开源方案；想给已有高模资产加铰链，用 AnyMesh；要批量、要精度，等 Particulate 开源或用其思想自训。

---

## 9. 个人启发与后续问题

1. **「LLM 出意图 + 确定性系统出精度」的分层**是我认为最值得带走的设计：凡是「语义决策 + 精确执行」的 3D 任务（布线、拓扑修复、UV 展开）都可以套这个结构——模型不碰数值，数值交给几何算法；
2. **critic 反馈的带宽优势**：语言化的残差（哪里错、为什么错、怎么改）远比标量 reward 可执行。这提示 3D 资产质检（如 mesh wiring quality 评审）也可以做成「渲染 + 语言诊断 + 定向修复」的闭环；
3. 与 FAMOS 的对照很有意味：**消解铰接歧义的两种路径**——几何上「动给模型看」（多状态点云）vs 语义上「动给 VLM 看」（视频）。两个信号能否在单一模型里融合（视频条件化的前馈铰接预测）是明显的一步棋；
4. 遗留问题：闭环的收敛性没有任何保证，何时该放弃并上报人工（active learning 式的置信度估计）论文未涉及；URDF 的限位/物理属性质量也未系统评测。

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | https://arxiv.org/pdf/2410.13882 | 全文（v4, ICLR 2025） | 2026-10-23 | 精读主来源 |
| `figures/01-teaser.png` | teaser 图 | 同上 | Figure 1, p.1 | 2026-10-23 | 背景章节 |
| `figures/02-method-pipeline.png` | 方法总览图 | 同上 | Figure 2, p.4 | 2026-10-23 | 核心方法章节 |
| `figures/03-actor-critic.png` | 闭环实例图 | 同上 | Figure 4, p.6 | 2026-10-23 | 方法章节 |
| `figures/04-in-the-wild.png` | 定性对比图 | 同上 | Figure 7, p.8 | 2026-10-23 | 实验章节 |

其他来源：[项目页](https://articulate-anything.github.io)（含在线 demo、机器人迁移视频）、[GitHub 仓库](https://github.com/vlongle/articulate-anything)。
