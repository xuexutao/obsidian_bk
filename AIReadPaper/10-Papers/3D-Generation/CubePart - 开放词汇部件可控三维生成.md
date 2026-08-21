---
type: paper-note
status: done
domain: 3D-Generation
paper: CubePart: An Open-Vocabulary Part-Controllable 3D Generator
year: 2026
arxiv: 2605.28763
source: https://arxiv.org/abs/2605.28763
project: null
code: null
tags:
  - 3D-Generation
created: 2026-08-21
updated: 2026-08-21
---

# CubePart：开放词汇部件可控三维生成

## 基本信息

| 字段 | 内容 |
|---|---|
| 论文名称 | CubePart: An Open-Vocabulary Part-Controllable 3D Generator |
| arXiv | 2605.28763v1 [cs.AI]，2026-05-27 |
| 会议/期刊 | SIGGRAPH 2026 / ACM TOG（DOI: 10.1145/3799902.3811117，ISBN: 979-8-4007-2554-8） |
| 作者 | Yiheng Zhu、Kangle Deng、Jean-Philippe Fauconnier、Inaki Navarro、Daiqing Li、Ava Pun、Yinan Zhang、Peiye Zhuang、Xiaoxia Sun、Maneesh Agrawala、Kiran Bhat、Tinghui Zhou（前四位 equal contribution） |
| 单位 | Roblox Foundation AI Team + Carnegie Mellon University + Stanford University |
| 项目页 | https://cubepart.github.io/ |
| 代码 | https://github.com/Roblox/cube/tree/main/cubepart |
| 模型/演示 | https://huggingface.co/Roblox/cubepart，https://huggingface.co/spaces/Roblox/cubepart-demo |
| 许可 | CC-BY 4.0 |

## 一句话结论

CubePart 把"开放词汇文本 schema"作为推理时显式控制信号，通过两阶段 vecset 扩散（整体形状 → 多个独立部件 latent）并辅以零初始化的 Cross-Part Attention 残差块，生成与用户给定部件清单严格对齐的、结构完整且可被游戏引擎脚本直接驱动的多部件 3D 网格；其能力由一个约 462K 资产、2.02M 部件的自动化 VLM 标注数据引擎支撑。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

游戏与交互式 3D 应用中，"可用"的 3D 资产从来都不是一整块 mesh。车辆需要可旋转的轮子、机器人需要可独立控制的手臂、容器需要能开合的盖子——这些行为由游戏引擎中的物理模拟、动画绑定和 Lua/Blueprint 脚本驱动，而脚本操作的是**预定义的语义部件集**。换句话说，资产必须按应用期望的"schema"分解，否则再好看的几何也无法被引擎消费。

随着 3D 生成建模的快速进展（DreamFusion、3DShape2VecSet、TRELLIS、CraftsMan 等），从文本或图像合成复杂几何已经可行。但这些方法要么输出**单一整体网格（monolithic mesh）**，要么输出**任意 / 不可控的部件集合**。对一位需要"四个轮子 + 一个底盘"的游戏开发者而言，输出随机切分的网格和单一整体网格同样无用。

### 1.2 现有方法的瓶颈

论文把现有方案的不足归纳为三条线：

第一类**单 mesh 生成**（CraftsMan、TripoSG、CLAY、Hunyuan3D 等）：质量高但不暴露任何部件控制信号。  
第二类**基于 2D 分割的多阶段管线**（Part123、PartGen、ComboVerse、OmniPart 等）：用 SAM/多视图扩散先得到 2D 掩码再升到 3D。问题是 2D 掩码**无法表示被遮挡的部件**（动物背面、车辆底盘），并且**视角依赖**——同一语义在不同视角下是不同掩码，升到 3D 后天然存在歧义。  
第三类**3D 原生部件生成**（HoloPart、PartCrafter、PartPacker、OmniPart、X-Part 等）：  
- HoloPart 需要先有 3D 分割结果再补全，依赖上游分割质量；  
- PartCrafter / PartPacker / BANG 只能控制部件数量，无法控制部件**身份**；  
- OmniPart / FullPart / PartVerse 等使用**固定词表或学到的隐式部件词汇**，用户无法在推理时改写。

三类方法的共同缺口：**缺一个 3D 原生、开放词汇、用户可直接编辑的 schema 控制接口**。

### 1.3 本文核心贡献

作者明确给出三点贡献：

1. **可扩展的数据引擎**：在非结构化 3D 网格上利用 VLM 的 3D 感知聚类与命名能力，自动构建出**开放词汇、带部件标签**的大规模 3D 数据集；
2. **schema 驱动的两阶段生成架构**：将整体形状合成与部件级解码解耦，配套一个**Cross-Part Attention 残差块**，让部件之间能够通信而保持预训练单 mesh 先验不被破坏；
3. **端到端可用性演示**：生成的多部件 mesh 能在 Roblox 引擎里直接绑定行为脚本（驾驶、开火、机械臂伸展、激光、无人机起飞等），无需人工后处理。

![](assets/CubePart%20-%20开放词汇部件可控三维生成/teaser.png)

> 图 1：CubePart 的能力示意（论文 Figure 1）。左侧海龟搭载石质城堡，给定 `turtle head / turtle shell / flippers / castle towers / main keep structure / cannons` 的 schema，模型输出一组按颜色区分的、彼此独立的多部件网格。右侧三组样例（水母赛车、宝箱、机器人）说明 CubePart 既能从文本 + schema 生成，也能以**已有 mesh 作为输入**按 schema 重新切分；最底行则演示这些部件被脚本驱动后可以产生交互行为。  
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/html/2605.28763

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

任务形式化定义如下：

- **输入 I — 全局文本描述** $p_{\text{global}}$：描述物体整体外观，例如 `"A jellyfish-themed race car."`。
- **输入 II — 开放词汇部件 schema** $\mathcal{S} = \{s_1, s_2, \dots, s_N\}$：用户希望拆分出的语义部件名称列表，可以是 `"front left wheel"`，也可以是 `"魔法杖"`、`"main keep structure"` 等自由词。$N$ 在 2–32 之间（数据集预处理过滤范围）。
- **可选输入 III — 已有的单 mesh 输入** $M_{\text{in}}$：CubePart 也支持把一个**已有整体网格**按 schema 重新切分为多个独立部件。
- **输出**：$N$ 个独立、**结构完整**的 mesh $\{M_1, M_2, \dots, M_N\}$，每个 $M_i$ 与 schema 中 $s_i$ 一一对应，共同组装成一个连贯物体。

任务有三个关键假设：

- **刚性分解假设**：当前方法只输出**刚体部件**，不输出蒙皮权重或骨架，因此不能直接服务需要顶点变形的角色动画。
- **可拼装假设**：生成部件应能在原坐标空间正确拼回整体，但允许接缝处出现少量 interpenetration。
- **schema 自洽假设**：用户提供的 schema 应大致对应物体的真实语义分解（"四个轮子"应当出现在"赛车"上）。如果 schema 与全局描述严重矛盾，质量会下降。

### 2.2 关键符号和目标函数

- **形状 VAE 潜变量集合** $Z_0 = \{z_j\}_{j=1}^{K}$：从 3DShape2VecSet 风格 VAE 编码得到的无序向量集合，$K$ 为 token 数。
- **Stage 1 整体潜变量** $Z_{\text{full}}$：由 MM-DiT 在文本条件 $c$ 下生成的整体形状潜变量。
- **Stage 2 部件潜变量** $Z_i = \{z_{i,j}\}_{j=1}^{K}$：第 $i$ 个部件的潜变量集合。
- **Flow matching 训练目标**（论文公式 1）：

$$
Z_t = t \cdot Z_0 + (1 - t) \cdot Z_1, \quad v_t = Z_1 - Z_0
$$

$$
\mathcal{L} = \mathbb{E}_{(Z_0,c) \sim \mathcal{D},\, Z_1,\, t} \left\| f_\theta(Z_t, t, c) - v_t \right\|^2
$$

其中 $Z_1 \sim \mathcal{N}(0, I)$ 是标准正态噪声，$t$ 从 logit-normal 分布采样并以 4.0 的因子偏移；$f_\theta$ 是带可学习参数 $\theta$ 的扩散网络（MM-DiT），$c$ 由 Qwen-VL 文本编码给出。

- **部件感知提示模板**：
  - 整体生成 prompt：`"{global caption}. This object contains the following parts: {part list}."`
  - 部件分割 prompt：`"This object has the following parts: {all parts}. Target to segment: {target part}."`

> 说明：论文正文没有给出显式的"多部件损失"，多部件能力来自**共享的同一网络 + 部件感知 prompt + 跨部件注意力残差块**这三者的组合。`f_θ` 在 Stage 1 与 Stage 2 是**同一套 MM-DiT 架构**，Stage 2 通过插入残差块（而非修改原层）扩展其行为。

### 2.3 任务的关键评估维度

- **schema 遵从性**：生成的部件集合是否与用户 schema 一一对应。
- **几何保真度**：单个部件 mesh 是否结构完整、无孔洞、无塌陷。
- **整体一致性**：所有部件拼回后是否还原"全局文本描述"对应的整体形状。
- **方向/空间正确性**：像 `front-left` / `rear-right` 这类带空间标识的部件是否落在正确一侧。
- **下游可用性**：是否能直接挂到引擎脚本上驱动行为。

---

## 3. 核心方法

### 3.1 总体框架

CubePart 是一个**两阶段 vecset 扩散框架**，关键思想是**把"零件结构"前置到生成过程本身**，而不是先生成完整网格再切。

- **Stage 1（Single-Part Mesh Generation）**：把"全局文本 + schema 列表"作为 prompt，训练一个文本条件 vecset 扩散模型，输出**一个完整的整体形状潜变量** $Z_{\text{full}}$，再由 Shape VAE 解码为整体 mesh。
- **Stage 2（Multi-Part Mesh Generation）**：以 $Z_{\text{full}}$ 为全局条件，针对 schema 中每个部件名分别采样一个**部件潜变量** $Z_i$。这一阶段在 Stage 1 预训练权重上额外插入 4 个**零初始化的 Cross-Part Attention 残差块**，使不同部件之间能彼此看见上下文，从而学会切分边界。

**为什么一定要拆成两阶段**（论文正文与附录的隐含论据，个人判断整理）：

- **显式接口的代价**：开放词汇 schema 的部件数 $N$ 是用户实时决定的，与之对应的部件潜变量集合 $\{Z_i\}$ 也必须动态变化。把"部件数"显式塞进网络会让模型与特定 $N$ 强绑定。拆成两阶段后，**Stage 1 只负责"整体长什么样"，不绑定部件数；Stage 2 才把整体拆成 $N$ 份**，等于把"动态 schema 长度"这一最难处理的变量推迟到第二阶段。
- **几何先验的复用**：3D 几何本身有大量跨对象的共有规律（人体对称性、车辆前后轴分布、动物四肢拓扑），这些规律只需 Stage 1 学一次；Stage 2 只需在"已知整体"基础上做部件级解耦，避免从零学习几何。
- **预训练 + 微调的天然契合**：Stage 1 实质是一个**通用文本 → vecset 网格**扩散模型，与 CraftsMan、CLAY 等的预训练目标完全同构，未来可直接受益于更大规模的 3D 基础模型；Stage 2 是一个**部件级解耦头**，规模小、训练快。

下图给出整体 pipeline：

![](assets/CubePart%20-%20开放词汇部件可控三维生成/method.png)

> 图 2：CubePart 总体框架（论文 Figure 2）。(a) Stage 1 使用单 mesh DiT（MM-DiT 块堆叠）从 Qwen-VL 文本条件直接生成整体形状潜变量。(b) Stage 2 在相同 MM-DiT 主干上**插入** 4 个 Cross-Part Attention Residual Block（蓝色高亮），所有部件共享同一个 Qwen-VL 文本条件，最后由共享的 Shape VAE 解码器并行输出多个独立 mesh。  
> 来源：论文 Figure 2，第 3 页，https://arxiv.org/html/2605.28763

（**个人判断**）这种"主干 + 旁路残差块"的设计有清晰动机：Stage 1 已经学好了"如何生成一个结构合理的 3D 物体"，如果直接修改其注意力层来加入部件通信，会破坏这一先验；用零初始化残差块插入既有层之间，相当于"在不打扰主模型的前提下，让网络逐步学会"看其他部件"——这与 ControlNet 思路一脉相承，但作用对象从图像潜变量换成了多个部件潜变量。

### 3.2 Stage 1：单部件网格生成（Schema-aware Finetuning）

**预训练 backbone**：

- **形状 VAE**：使用 3DShape2VecSet 风格的无序 token 集合 VAE，解码器采用 SDF 表征以获得锐利几何。
- **扩散 backbone**：**Multi-Modal Diffusion Transformer（MM-DiT）**，文本条件走 cross-attention 注入。
- **文本编码器**：**精简版 Qwen-VL**——层数 21、隐藏维 1536，共 1.9B 可训练参数（论文未披露原始 Qwen-VL 的层数与宽度，仅说明缩小比例与最终参数规模）。
- **预训练语料**：约 4.7M 网格-文本对，由 745K 专有资产 + 约 4M 合成资产组成，用以扩充文本多样性。

**为什么预训练规模能做到 4.7M**（论文 3.2 节与 4.1 节综合，**个人判断**）：

- 745K 专有资产是 Roblox 内部积累的真实游戏资产，每一件都有真实玩家场景下的视觉描述；
- 4M 合成资产由"用 captioner 反向生成文本 + 用现有图像到 3D 模型生成对应几何"组成，目的是**让文本侧覆盖更多长尾描述**（颜色组合、风格词、跨类别类比），解决真实资产文本多样性问题；
- 这种"专有保真 + 合成补量"的组合是当前大规模 3D 预训练的常见做法，论文没有公开 captioner 与生成模型的具体选择。

**Schema-aware 微调**：

预训练好的单 mesh 模型虽然能生成高质量 3D 形状，但即使把 schema 拼进 prompt，也**不保证所有部件都出现**——某些部件可能被忽略、某些被过度强调。论文在 462K 资产、2.02M 部件的策划数据集上做 schema-aware finetuning，prompt 模板为：

```text
{a global caption}. This object contains the following parts: {list of part labels}.
```

- **作用 1**：让 Stage 1 在采样阶段就把"必须包含哪些部件"作为显式约束，从而在 $Z_{\text{full}}$ 内部为各部件预留几何位置与比例。
- **作用 2**：为 Stage 2 提供**预训练起点**——Stage 2 整体沿用这套权重再插入 Cross-Part Attention，避免从零学习。
- **（个人判断）作用 3 的猜测**：把 schema 文本前置到 Stage 1 等于让"哪些部件存在"成为 $Z_{\text{full}}$ 的**可线性探测属性**。这给 Stage 2 的 Cross-Part Attention 提供了一个隐式锚点：$Z_{\text{full}}$ 中已经"知道"有哪些部件、各部件大致在哪，Stage 2 只需要把它们"画"出来。

**训练实现细节**（论文 3.2 节）：

- 训练目标为上述 flow matching loss。
- 优化器：AdamW，$\beta_1 = 0.9$，$\beta_2 = 0.99$，**关闭 weight decay**。
- batch size = 768，学习率 $= 10^{-4}$，前 2000 步线性 warm-up。
- 24 块 H200 GPU 上训练约 3 天，约 1500 GPU-hours。

### 3.3 Stage 2：多部件网格生成与 Cross-Part Attention

Stage 2 的目标是把"一个 mesh 的潜变量"拆成"每个部件一个潜变量"。表示上，物体被编码为 $N$ 个部件的潜变量集合：

$$
\mathcal{O} = \{p_i\}_{i=1}^{N}, \quad z_i = \{z_{i,j}\}_{j=1}^{K} \in \mathbb{R}^{K \times C}
$$

**朴素的部件感知基线**：直接让 $f_\theta$ 预测特定部件的潜变量，文本条件为

```text
"This object has the following parts: {all parts}. Target to segment: {target part}."
```

——即把所有部件名都列出来、提示当前要切的是哪一块。这一步"目标 prompt"的设计让模型能在切 $p_i$ 时知道其他部件是谁，从而识别切分边界。

**关键创新：Cross-Part Attention Residual Block**

朴素方案只让文本上下文提供"其他部件是谁"，但部件间缺乏直接的潜变量级通信。论文观察到一个反直觉的结论：**在原 MM-DiT 层上做部件间全局注意力（PartCrafter 风格）反而损害性能**——因为这会扰动 Stage 1 学到的几何先验。

替代方案是**新增专用 Transformer 块**：

- **零初始化**：在插入手写零的输出投影层，使新块在训练初期等价于恒等映射，**不破坏**预训练权重。
- **输入**：所有部件的 $z_i$（按部件维度拼接）+ 来自 Stage 1 的整体 $Z_{\text{full}}$ 作为全局条件。
- **作用**：让每个 $z_i$ 都能"看见"其他部件潜变量，从而推断边界、邻接与互斥区域。
- **插入位置**：第 1、5、9、17 层，共 4 个块。

**设计上的几个细节选择**（论文图 3 与正文综合，**个人判断**整理）：

- **为什么是"残差块"而不是"并行支路"**：残差形式让 $Z_{\text{full}}$ 既可以通过 Cross-Part Attention 块反向影响各部件 $z_i$，又保证即使新块学得不好，主干 $z_i$ 仍能保持 Stage 1 风格。这是"软插入"思路的常见做法。
- **为什么 $Z_{\text{full}}$ 也作为输入**：论文明确把"来自 Stage 1 的整体 shape latent"送进残差块——这相当于**把 Stage 1 的全局几何信息作为 Stage 2 各部件的强共享锚点**。如果不加这条信号，每个 $z_i$ 只能依赖文本条件间接知道"整体长什么样"，会出现部件之间各自跑偏。
- **为什么用 Transformer block 而不是更轻的 MLP/Conv**：跨部件通信需要建模"任意部件对"的关系（即 O(N²)），Transformer 自注意力的结构最直接；MLP 只能看到拼接后的全局，不能区分部件对。
- **（个人判断）零初始化的妙处**：把"先验 = 单 mesh 生成"和"新能力 = 跨部件通信"显式分离——训练初期模型表现和 Stage 1 完全一样；随着训练推进，残差项开始"接管"那些需要跨部件语义的子空间，主干部分不必重新学。这种"主干不动、旁路学新东西"的范式与 LoRA、ControlNet、Adapter 同源。

![](assets/CubePart%20-%20开放词汇部件可控三维生成/cross_part_attention.png)

> 图 3：Cross-Part Attention Block 内部结构（论文 Figure 3，原始为 SVG 矢量图，已转 PNG 便于 Obsidian 渲染）。所有 part latent 与 Stage 1 的整体 shape latent 一同送入一个**零初始化的 Transformer 块**做全局注意力，再以残差方式加回主网络。该块被"插入"而非"替换"原层，因此对预训练单 mesh 能力是零侵入的。  
> 来源：论文 Figure 3，第 4 页，https://arxiv.org/html/2605.28763（S3.F3 容器对应 `2605.28763v1/part_attn.svg`）

**Stage 2 训练实现细节**：

- 初始化：直接继承 Stage 1 训练好的 MM-DiT 权重。
- 优化器与学习率同 Stage 1，**batch size 降至 72**（每个样本含多个部件）。
- 24 块 H200 上训练约 18 小时，约 450 GPU-hours。
- **推理时延**（H200，含 VAE 解码）：Stage 1 约 2–3 秒，Stage 2 约 3–4 秒。

**关于"4 个 Cross-Part Attention 块插入到第 1/5/9/17 层"这一选择**（论文未直接给出消融曲线，**个人判断**）：MM-DiT 主干共 21 层（沿用 Stage 1 文本编码器层数），作者把跨部件块分别插在浅层（第 1 层，处理局部几何对齐）、中浅层（第 5 层，部件邻接感知）、中层（第 9 层，全局一致性）和深层（第 17 层，最终接缝协调）。这一分布兼顾"早对齐、晚修正"的直觉；论文没给"换成 1 个、8 个、均匀插"的消融，是个值得后续工作补全的实验维度。

### 3.4 训练目标与损失函数

两阶段共享同一个 flow matching 目标 $\mathcal{L}$（见 2.2 节公式 1），区别仅在条件 $c$ 与输入潜变量集合：

- **Stage 1** 输入是单 mesh 的 $Z_0$、$Z_1$，条件是"全局 + schema" 文本；
- **Stage 2** 输入是每个部件独立的 $Z_0^{(i)}$ 与 $Z_1^{(i)}$，条件包括两部分：
  - **文本条件**：`This object has the following parts: {all parts}. Target to segment: {target part}.`
  - **形状条件**：Stage 1 输出的 $Z_{\text{full}}$ 通过 Cross-Part Attention 块进入各部件的扩散过程。

论文没有给出"部件数量"或"schema 长度"相关的额外监督；schema 遵从性主要靠两部分保障：（1）Stage 1 schema-aware finetuning 使整体形状对齐 schema；（2）Stage 2 的"全列表 + target"提示与 Cross-Part Attention 隐式约束。

**（个人判断）** 这是一处论文刻意保持克制的设计：与其为 schema 长度加显式损失，不如把它作为**控制信号**而不是**优化目标**。优势是模型不会因为 schema 长度变化而出现训练-推理不一致；代价是没有"硬约束"——一旦 Cross-Part Attention 失效，模型就缺乏兜底（这正是 §5.3 中 `Ours w/o Cross-Part Attention` F-score 暴跌到 0.398 的根因）。

### 3.5 推理流程与复杂度

推理时按以下顺序执行：

1. **解析输入**：全局文本 $p_{\text{global}}$ 与 schema $\mathcal{S} = \{s_1, \dots, s_N\}$ 拼成 Stage 1 prompt。
2. **Stage 1 整体生成**：用 flow matching 求解器从 $\mathcal{N}(0, I)$ 采样 $Z_{\text{full}}$，经 Shape VAE 解码得到整体 mesh $M_{\text{full}}$（可选，用于可视化与行为编辑）。
3. **逐部件采样**：对每个 $s_i$，构造 `Target to segment: {s_i}` prompt，从 $\mathcal{N}(0, I)$ 采样 $Z_i$，经 VAE 解码得到 $M_i$。
4. **拼装与输出**：所有 $M_i$ 沿用 Stage 1 的同一坐标空间，共同表达完整物体。

复杂度上，Stage 1 与 Stage 2 都做了固定步数（例如 50 步）的 ODE 求解；Stage 2 因为要循环 $N$ 次，**推理时间与部件数近似线性**。论文报告的端到端时延（H200，含 VAE 解码）约 5–7 秒（2–3 + 3–4），部件数 $N$ 在 2–32 之间时延差异论文未披露（个人判断：当 $N$ 较大时 VAE 解码可能成为主导）。

**一个被论文隐含但值得展开的工程点**：Stage 2 实际上是**对每个部件名独立做一次完整扩散采样**。这意味着同一个物体跑 $N$ 次前向，时间复杂度与 $N$ 线性相关。如果生产场景中"同一物体需要不同 schema 切分"（比如先要"车身+四个轮子"再要"车身+底盘+引擎"），可以**只重跑 Stage 2，Stage 1 结果缓存复用**——这与 §5.4 的 varying schema 实验天然契合，也是该架构相对"全自回归"路线（如 AutoPartGen）的一个隐性优势。

**训练动力学层面的"先验保护"直觉**（**个人判断**）：

零初始化的 Cross-Part Attention 块在训练初期的行为完全等价于恒等映射——**Stage 2 一开始就具备 Stage 1 100% 的能力**。这给优化过程提供了一个非常重要的"安全网"：即使 Cross-Part Attention 块在新任务上**完全没学成**，Stage 2 也会退化成"Stage 1 + 部件感知 prompt"的朴素版本，而不是崩塌成随机输出。随着训练推进，残差块开始贡献"跨部件去歧义"的能力，主干权重微调以适应新任务（但幅度很小，因为残差项已经在消化大部分新梯度）。这种**"先可退化、再学增强"**的训练轨迹，解释了为什么 Stage 2 仅用 450 GPU-hours（约 Stage 1 的 30%）就达到 0.743 部件级 F-score——Stage 1 的 1500 GPU-hours 没有白花，它们以"可退化为先验"的形式被 Stage 2 继承。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

**规模与来源**：约 462K 资产、2.02M 部件，组合如下（论文 Table 1）：

| 来源 | 子集 | 内容 | 资产 | 部件 |
|---|---|---|---|---|
| Sketchfab | Objaverse、Texverse、PartVerse、PartVerse-XL | 角色、动物、建筑等 | 270K | 1.14M |
| 商业授权库 | 授权素材库 | 家具、CAD 等 | 64K | 201K |
| Roblox 内部 | 游戏资产 | Avatar、车辆等 | 129K | 679K |
| **合计** | | | **462K** | **2.02M** |

去重策略：按资产名去重，PartVerse 系列优先（人工修正过的分割更干净），限制为宽松许可的资产，**排除 PartObjaverse-Tiny 以避免测试集污染**。

**与既有部件数据集的比较**（论文 Table 2）：

| 数据集 | 资产 | 部件 | 开放词汇 | 部件文本 |
|---|---|---|---|---|
| ShapeNetPart | 16K | 93K | ✗ | 分类法标签 |
| PartNet | 26K | 573K | ✗ | 分类法标签 |
| PartVerse | 12K | 91K | ✓ | 描述性 caption |
| PartVerse-XL | 40K | 320K | ✓ | 描述性 caption |
| **CubePart** | **462K** | **2.02M** | **✓** | **简洁部件名** |

CubePart 的资产规模比 PartVerse-XL 大 11 倍以上、部件数 6 倍以上，且生成的是"**name**"（如 `front left wheel`）而非"caption"（如 `A long, slender component extracted from the top section of a futuristic tank-like vehicle`），更接近用户实际查询时会输入的形态。

**四阶段数据流水线**（论文 4.1 节）：

1. **预处理**：过滤退化几何，保留 2–32 个部件的资产，排除过简或过复杂的样本以最大化下游 VLM 成功率。
2. **VLM 质量过滤**：从 8 个视角渲染，由 VLM 按几何缺陷（撕裂、碎片化）、扫描伪影（噪声表面）、场景级内容（房间截面、剖面）、问题几何（零体积、过薄）打分，输出 0–1 复杂度分与简短描述；仅 **moderate / excellent** 资产进入聚类。
3. **VLM 部件聚类与命名**（核心步骤，详见 4.2）。
4. **后处理**：在 512³ SDF 上做 Dual Marching Cubes 得到水密 mesh；从每个部件采样 128K 可见表面点（带法线），整体再采 128K；用 Farthest Point Sampling（FPS）保证均匀覆盖。

### 4.2 核心环节：Set-of-Mark + VLM 聚类命名

很多 3D 资产带有"过度分割"和"不一致命名"。CubePart 的关键洞察是**不重新发明 3D 分割**，而是在已有部件边界之上做"语义聚类"——把过度切分的 rim + tire + hub 合并为 `front left wheel`，并赋予简洁的语义名。

**Set-of-Mark（SoM）渲染**：受 Set-of-Mark 提示技术启发，但做了 3D 适配：

- **每个视角生成两张配对图**：
  - 左侧：纹理渲染 + 部件轮廓 + 编号标记；
  - 右侧：每个部件纯色着色的渲染（无纹理干扰）。
- **标记从 3D 几何直接生成**：部件掩码保证不重叠，每个标记放在到部件边界**最远**的位置以避免遮挡。
- **配色一致性**：纹理视图里的编号色与部件视图里的纯色匹配，便于 VLM 在两张图之间建立对应。
- **视角数**：每个资产渲染 **14 个轨道视角**，给 VLM 充分的 3D 上下文。

下图为一张示例 SoM 渲染：

![](assets/CubePart%20-%20开放词汇部件可控三维生成/som_rendering.png)

> 图 5：Set-of-Mark 配对渲染示例（论文 Figure 5 的纹理视图，编号视图原图为配对的 notexture 版本）。雪人身上的 10 个部件被赋予不同编号和对应纯色，VLM 可同时读"语义（纹理）+ 身份（编号 + 纯色）"两个证据源。  
> 来源：论文 Figure 5，第 5 页，https://arxiv.org/html/2605.28763v1/figures/data/render_front_tilt_texture.png

**VLM 标注细节**：

- 模型：**GPT-5**（论文做了 GPT-4o vs. GPT-5 对比，报告 GPT-5 在聚类准确性、粒度与命名一致性上均更优）。
- 输入：14 个视角对，跨多图像联合推理。
- 提示要点：（1）按"功能/逻辑关系"而非"视觉相似度或空间邻近"分组；（2）为每组分配简洁名称；（3）允许**单例聚类**（identity clustering）——若某部件已经代表完整语义单元则不必合并。
- 输出：结构化 JSON，包含聚类名与组成部件的编号列表。
- **有意省略**：不把源资产已有的部件名/层次结构喂给 VLM，因为它们常含噪声且跨来源不一致。
- **后处理规则**：重复分配保留第一次出现；VLM 退化为单聚类（实质丢失部件结构）的样本被过滤。

**命名 vs. caption 的差别**（论文 4.2 节末尾强调）：

PartVerse 这类基于 caption 的标注会出现 `"A close-up of …"`、无法区分左右件（同一 caption 复用）等 VLM 伪影；CubePart 改用**简洁部件名**，只在需要时附加位置形容词（`left arm`、`rear right wheel`）。这对下游**文本条件 schema 检索**至关重要——用户大概率会写"front left wheel"而非一段长描述。

下图对比三种来源在同一辆坦克上的差异：

![](assets/CubePart%20-%20开放词汇部件可控三维生成/qualitative_comparison_data_processing.png)

> 图 4：同一 Objaverse 坦克的部件标注对比（论文 Figure 4）。最左：原始艺术家分解 7 个部件；中：PartVerse 给出 17 个部件，caption 出现 `"A close-up of …"` 等 VLM 伪影，且无法区分左右件；最右：CubePart 流水线给出 **4 个**简洁、空间可辨的部件名（`hull`、`tracks`、`turret and cannon`、`side arms`）。这说明 VLM 聚类既在做"语义合并"也在做"语义命名"，而不是简单继承原始切分。  
> 来源：论文 Figure 4，第 4 页，https://arxiv.org/html/2605.28763

### 4.3 Baseline 与评价指标

**评估数据集**：**PartObjaverse-Tiny**，从 Objaverse 均匀采样 200 个 mesh，由人工策展的开放词汇部件 schema 标注，**论文明确说明该集主要用作评测**而非训练。

**Baseline**（论文 5.2 节）：

- **PartCrafter**：图像 → 多个 3D 部件，只控制数量、不控制身份。
- **PartPacker**：图像 → 多个 3D 部件，SDF 双体打包解决接触伪影。
- **PatchAlign3D + HoloPart**：先用 PatchAlign3D 给出 3D 分割、再用 HoloPart 补全；这是与 CubePart 最接近的对照，因为同样接受 mesh + schema，但依赖上游分割质量。
- **SAM3 + OmniPart**：图像 + 2D 分割 → 3D 部件（额外用真值体素初始化），属于"不可控"对照。

**评价指标**：

- **Chamfer Distance (CD)**：越低越好。
- **F-score**：阈值 0.1，越高越好。
- 报告两个层级：
  - **整体级（Object-level）**：对完整网格计算。
  - **部件级（Part-level）**：对每个独立部件分别计算 CD/F-score，再对所有部件取平均。
- **所有形状先归一化**到 $[-1, 1]$ 单位盒。

**实现细节**：

- 形状 VAE 沿用 3DShape2VecSet。
- Stage 1 用 AdamW，关闭 weight decay，batch=768，lr=$10^{-4}$，24 块 H200 训 3 天。
- Stage 2 继承 Stage 1 权重，插入 4 个 Cross-Part Attention 块到第 1/5/9/17 层，batch 降至 72，同样 24 块 H200 训约 18 小时。
- 推理时延：H200，Stage 1 约 2–3 s，Stage 2 约 3–4 s（含 VAE 解码）。
- 评测用脚本与原模型权重均已公开（见 §10）。

---

## 5. 实验结果

### 5.1 主要定量结果

在 PartObjaverse-Tiny 上的统一对比（论文 Table 3）：

| 方法 | 部件级 CD ↓ | 部件级 F-score ↑ | 整体级 CD ↓ | 整体级 F-score ↑ |
|---|---|---|---|---|
| PartCrafter | 0.493 | 0.290 | 0.272 | 0.552 |
| PartPacker | 0.374 | 0.475 | 0.164 | 0.792 |
| PatchAlign3D + HoloPart | 0.309 | 0.549 | 0.050 | 0.970 |
| SAM3 + OmniPart | 0.309 | 0.630 | 0.053 | 0.970 |
| Ours w/o pre-training | 0.287 | 0.625 | 0.051 | 0.970 |
| Ours w/o Cross-Part Attention | 0.433 | 0.398 | 0.148 | 0.792 |
| Ours w/ PartCrafter-style attention | 0.386 | 0.529 | 0.089 | 0.864 |
| **Ours** | **0.251** | **0.743** | **0.048** | **0.974** |

**结果说明了什么**：

- **整体级指标上，CubePart 与最强的"双阶段"对照（PatchAlign3D + HoloPart、SAM3 + OmniPart）几乎持平**（F-score 0.974 vs. 0.970，CD 0.048 vs. 0.050/0.053）。这说明 Stage 1 的整体形状生成能力已经追平现有 SOTA。
- **部件级指标上，CubePart 显著优于所有对照**——F-score 0.743 比第二名（SAM3 + OmniPart 0.630）高出 11 个百分点，CD 0.251 也明显更低。这是 schema 控制接口的直接收益：可控部件生成能更准确地切出"用户要的那几块"。
- 单纯看 PartCrafter / PartPacker 只能控制数量不控制身份，F-score 0.290 / 0.475 与 CubePart 差距巨大，**验证了"开放词汇 schema"这一控制信号的必要性**。

**逐行解读**（**个人判断** 整理）：

- **PartCrafter (0.290 / 0.475 / 0.272 / 0.552)**：F-score 最低、CD 最高。原因是它只控数量不控身份，输入图像的语义部件与生成结果无显式对应，**部件身份完全靠网络"猜"**。
- **PartPacker (0.475 / 0.374 / 0.164 / 0.792)**：相对 PartCrafter 部件级 CD 下降约 0.12，说明双体 SDF 打包策略对解决接触伪影确实有效，但仍未解决"控身份"问题。
- **PatchAlign3D + HoloPart (0.549 / 0.309 / 0.970 / 0.050)**：整体级已经接近 CubePart，但部件级仍低 0.20 个 F-score 点。差距来自 PatchAlign3D 分割本身的不完美（上游错了下游再补也回不来），以及 HoloPart 补全未做部件间通信。
- **SAM3 + OmniPart (0.630 / 0.309 / 0.970 / 0.053)**：借助 SAM3 强 2D 分割 + 体素初始化把部件级 F-score 拉高到 0.630，但仍受限于 2D 视角依赖与遮挡问题。
- **Ours w/o Cross-Part Attention (0.398 / 0.433)**：去掉跨部件块后 F-score 直接掉到 0.398，**几乎比去掉 Stage 1 预训练 (0.625) 还惨**。说明对"切得对"这件事，**部件间通信是更关键的能力**。
- **Ours w/ PartCrafter-style attention (0.529 / 0.386)**：用"修改原层"代替"插入新块"，F-score 0.529 显著差于 0.743。验证了 §3.3 的论点：**粗暴修改预训练层会损害单 mesh 几何先验**。
- **Ours w/o pre-training (0.625 / 0.287)**：整体级几乎不变（0.970 / 0.051），但部件级 F-score 从 0.743 降到 0.625。说明 Stage 1 的预训练对 Stage 2 的"切分"很关键，但 Stage 1 本身的整体生成不依赖这部分预训练——即 Stage 1 是一个**通用文本 → 整体网格**模型。

### 5.2 定性结果

下图是与多个基线在 PartObjaverse-Tiny 五个案例上的定性比较：

![](assets/CubePart%20-%20开放词汇部件可控三维生成/qualitative_comparison.png)

> 图 6：多部件网格生成定性对比（论文 Figure 6）。列从左到右：输入 schema、Ground-Truth、Ours、PatchAlign3D + HoloPart（同样接受 mesh+schema 的对照）、SAM3 + OmniPart（不可控，仅图+2D 分割）、PartCrafter、PartPacker。CubePart 在 schema 遵从性和几何细节上都明显占优；最右边三列的图条件方法既不能"指名道姓"切出 schema 中的部件，也更容易出现部件错位。  
> 来源：论文 Figure 6，第 6 页，https://arxiv.org/html/2605.28763v1/result_multiMesh_v4.png

**从图中可以读出的细节**（**个人判断**）：

- **房屋行**（Door/Earth/Roof Frames/Steps/Wall/Window）：CubePart 把"earth"和"steps"切得很干净；HoloPart 整体形状像但墙体接缝处有穿插；OmniPart 屋顶几何出现破洞；PartCrafter 几乎完全崩坏。
- **怪物行**（Body/Hair/Head/Leg/Tail）：CubePart 把 Hair 与 Body 干净分离，HoloPart 几乎把 Hair 整个吞进 Body。
- **盆栽行**（Body/Flowerpot/Hair/Head/Leaf）：CubePart 唯一稳定切出"flowerpot 容器"边界，其他方法都把容器与花叶混在一起。
- **水壶行**（Bottom of Pot/Kettle Body/Kettle Handle/Pot Lid）：CubePart 能把 Handle（手柄）作为独立件切出——这一类**带可活动铰接关系**的部件对游戏逻辑最有价值，也是 HoloPart 范式最难做的。

**对定性结果的二次解读**（**个人判断**）：

- 图 6 的对照设计本身揭示了一个**范式分水岭**。前两列（Ours、PatchAlign3D + HoloPart）属于 **mesh + schema 条件**——能直接控制部件身份；后三列（SAM3 + OmniPart、PartCrafter、PartPacker）属于 **image-only 条件**——只能给一个参考图。CubePart 的领先并非仅靠"模型更大"，而是把"控制接口"从图升级到文本 schema。**这一定性差异比 0.11 个 F-score 点更结构性地说明了 CubePart 的优势**。
- 注意 OmniPart 列虽然用了真值体素初始化，**仍然出现部件身份错位**（如盆栽行把 body 和 leaf 颜色互换）。这说明 2D 视角的语义对齐即使叠加 3D 先验也无法完全弥补——CubePart 的"全程 3D 原生 + 文本 schema"才是结构性优势。
- PartPacker 列的几何细节最差（墙面穿孔、腿部断裂），这暴露了**只控数量不控身份**的范式固有的"边界质量"问题：网络不知道哪些像素归哪个部件，只能凭"看起来像个独立物体"去划分，必然在高曲率区域失败。

下图是 Stage 1 + Stage 2 联合生成的多样化示例（drilling machine、charging tank、fantasy chicken hut）：

![](assets/CubePart%20-%20开放词汇部件可控三维生成/qualitative_twostage.png)

> 图 8：两阶段生成的定性结果（论文 Figure 8）。每个示例都给出了完整文本 prompt 与显式 schema 列表（彩色），输出 mesh 整体可识别、部件分割干净。可以看到 schema 文本直接对应到生成 mesh 的颜色块，证明"用户写什么，模型就切什么"。  
> 来源：论文 Figure 8，第 7 页，https://arxiv.org/html/2605.28763v1/result_full_mesh.png

### 5.3 消融实验

Table 3 的下半段就是 CubePart 的内部消融：

- **去掉 Cross-Part Attention**（`Ours w/o Cross-Part Attention`）：部件级 F-score 从 0.743 暴跌到 0.398（−34.5 个百分点），CD 从 0.251 上升到 0.433。整体级也明显退化（F-score 0.792，CD 0.148）。这说明**没有部件间通信，模型根本无法学到"切边界"**。
- **改用 PartCrafter 风格注意力**（`Ours w/ PartCrafter-style attention`）：在原 MM-DiT 层上做全局注意力而不是插入新块，结果 F-score 只有 0.529、CD 0.386，**比零初始化残差块差很多**。这印证了"修改预训练层会破坏先验"的论点。
- **去掉 Stage 1 预训练**（`Ours w/o pre-training`）：部件级 F-score 0.625、CD 0.287，整体级几乎不变。说明 **Stage 1 主要是为 Stage 2 提供几何先验**——它本身在整体指标上几乎没有信息损失，但 Stage 2 显著受益。
- **Schema-aware Finetuning 的视觉消融**（论文 Figure 7）：不做 schema-aware finetuning 时，模型会漏掉 schema 中的部件（如 steering wheel）或错误夸大某些部件（如 exhaust pipe）；做了之后所有请求部件都正确出现并合理成比例。**这一消融解释了为什么 Stage 1 不能直接用预训练版本**——没有 schema 注入，整体形状 latent 里就没有"为部件留位置"的隐式结构，Stage 2 即便做部件通信也缺乏全局几何依据。

（**个人判断**）消融实验设计有一个**非常聪明的对照**：选 `Ours w/ PartCrafter-style attention` 而不是任意的"非零初始化残差块"。它直接证明"问题不在于残差连接的形式，而在于是否动预训练层"——这是一个**架构层面的、不可替代的**结论，而非超参调优问题。

### 5.4 泛化、效率与失败案例

**Schema 灵活性（part 数量可变）**：

![](assets/CubePart%20-%20开放词汇部件可控三维生成/varying_schema.png)

> 图 9：同一输入 mesh 在不同 schema 长度下的切分（论文 Figure 9）。摩托车上：2 parts 时只能分出 body + wheels；4 parts 时多出 mirrors + stands；8 parts 时进一步切出 controls、seat、engine、frame、headlights。赛车行：2 parts → 4 parts → 8 parts 体现"fenders 是否独立"的可控性。这说明 CubePart **真正把 schema 数量当连续可调控制信号**，而不是训练死的 N-way 分类。  
> 来源：论文 Figure 9，第 7 页，https://arxiv.org/html/2605.28763v1/different_parts_new.png

**对图 9 的进一步解读**（**个人判断**）：

- **2 parts → 4 parts → 8 parts** 不是"切得更细"那么简单。从图中看，**部件数增加时整体几何本身也在变化**（摩托车挡泥板在 2 parts 时合并到 wheels，4 parts 后才独立成 fenders；赛车 8 parts 时车顶、悬挂、窗户都浮出）。这说明 Stage 2 不是在 Stage 1 结果上"再切"，而是**根据 schema 重新调整每个部件的几何**——Cross-Part Attention 在部件数变化时会被重新激活，整组潜变量联合重采样。
- 这种"schema 长度改变 ⇒ 几何整体重塑"的能力意味着 CubePart 的"控制粒度"对生产链路是**真正可用**的：游戏开发者可以先在低粒度看"长什么样"，再切到高粒度做精细化绑定，不用从零换模型。
- 论文没有给出"切到 32 parts"（即数据集上限）时的稳定性数据，**个人判断**：在边缘情况（如 schema 中出现奇怪组合、部件名拼写错误）下模型会按"最像的语义"做近似切分，这本质是 VLM 标注噪声的延伸。

**推理效率**：H200 上端到端约 5–7 秒（Stage 1 + Stage 2 + VAE 解码），对游戏生产链路中的"一次生成、多次使用"足够快。论文没有给出在更低规格 GPU（如 A100 / 4090）上的时延数据（个人判断：随 batch 大幅下降，单 sample 时延会因 step 数固定而接近线性放大）。

**与"自回归逐部件"路线的隐性对比**（论文 2.2 节简要提到 AutoPartGen，**个人判断**）：AutoPartGen 这类自回归方案在每一步都要基于已生成部件预测下一个，几百个部件的对象（如复杂建筑、装配体）会出现**误差累积 + O(N) 串行时延**。CubePart 走的是"一次性并行采样所有部件潜变量 + Cross-Part Attention 联合去歧义"，理论上**并行度更高、误差不累积**，但代价是单次前向计算量随部件数平方增长（Transformer 自注意力 O(N²)）。二者各有最优区间：对极复杂部件数，AutoPartGen 风格可能反而更经济；对 2–32 个部件的常见游戏资产，CubePart 的并行设计是更合适的选择。

**失败案例**（论文 Figure 11，§7 也专门讨论）：

![](assets/CubePart%20-%20开放词汇部件可控三维生成/failure_cases.png)

> 图 11：典型失败模式（论文 Figure 11）。挖掘机与龙的示例标出三类失败：**(1) 部件互相穿插**（Parts Overlapping），尤其是接缝处；**(2) 细节丢失**（Missing Details / Missing Lower Torso Part），复杂输入几何在切分时可能丢失局部结构；**(3) 左右镜像错误**（"Left" and "Right" Flip），空间标识被反转。  
> 来源：论文 Figure 11，第 8 页，https://arxiv.org/html/2605.28763v1/failure_example.png

**应用展示**：将生成的多部件 mesh 导入 Roblox 引擎并用 Lua 脚本驱动行为。

![](assets/CubePart%20-%20开放词汇部件可控三维生成/fig_behavior_v6.png)

> 图 10：游戏内行为脚本驱动示例（论文 Figure 10）。从左到右、从上到下依次为：水母赛车（Gun Shooting / Driving & Steering Wheels）、机器人（Arm Extension / Powered Takeoff）、青蛙角色（Laser Emission / Single-Leg Spin 360°）、无人机（Drone Takeoff / Drone Hover Right & Back）、巫师（Swing Staff / Illuminate Staff）。这一组图直接验证了 CubePart 的"下游可用性"主张——部件不是装饰性切分，而是 Lua 脚本里能 `MeshPart:Rotate()` 的真对象。  
> 来源：论文 Figure 10，第 8 页，https://arxiv.org/html/2605.28763v1/fig_behavior_v6.png

**应用场景的内在逻辑**（论文 §6 展开，**个人判断**整理）：

- **驾驶场景**（水母赛车）：基础 5 件（body + 4 wheels）已能驱动；想加灯光/排气时再细化 schema，把 jellyfish body、headlights、exhaust pipe 与轮子分离开；想加射击时再分出 gun 部件并配合粒子脚本。**这说明 schema 粒度直接对应行为粒度**——CubePart 让"功能需求 → 部件结构"的工程链路显式化。
- **角色场景**（机器人、青蛙、巫师）：头/手/武器/外置道具各自独立，支持"手臂伸展"、"激光发射"、"单腿旋转"、"挥杖发光"等独立绑定。**对具身智能/角色动画特别有价值**：可单独采样某个部件 latent 而不重生成整体。
- **飞行场景**（无人机）：把两个螺旋桨分立后能实现"非对称驱动"——这是单 mesh 资产几乎做不到的，因为单 mesh 没有"左桨 / 右桨"语义轴。CubePart 让物理仿真需要的部件级力矩控制有了真正可用的几何对象。
- **（个人判断）** 这一组应用案例的"工程含金量"比论文的定量表格更难替代——它直接展示了**"schema 即脚本接口"**这一新范式如何在生产链路里落地。比起 paper benchmark 上的 +11 个 F-score 百分点，**让游戏开发者少写一版切模脚本**是更实在的收益。

**关于"应用集为何不挑几何完美样本"**（**个人判断**）：

仔细看 Figure 10 的样例，**几何细节其实并不"满"**——赛车轮子有简化、巫师身体相对素模、青蛙有低面数痕迹。这恰恰是 Roblox 平台美术的典型特征：几何需要支持实时渲染、跨设备（手机/PC/主机）、快速迭代，**"足够好 + 立刻可用"**比"完美但需要后期"重要得多。CubePart 选择这样的目标函数与展示策略，说明它**把"生产可用"作为一等约束**，不是 demo-style 论文——这对评估其工程价值很关键。

---

## 6. 与相关工作的关系

论文把相关工作分三条线：

- **3D 生成模型（2.1 节）**：CubePart 与 CraftsMan、TripoSG、CLAY、Hunyuan3D 2.1 等共享 vecset 扩散骨架，但**将图像条件换成文本条件**（Qwen-VL），并在此基础上做 schema-aware 增强与多部件解耦。区别于 XCube / TRELLIS / SparseFlex / Sparc3D / Direct3D-S2 等稀疏体素路线——后者生成单 mesh 而不暴露部件控制。
- **部件感知 3D 生成（2.2 节）**：
  - 类别特定、监督式（SPAGHETTI、SALAD、DiffFacto、Neural Template）受限于固定词表，**不属于开放词汇**；
  - 2D-grounded 多阶段管线（Part123、PartGen、ComboVerse）受限于 2D 掩码的视角依赖与遮挡问题；
  - 3D 原生部件生成（HoloPart、PartCrafter、PartPacker、OmniPart、FullPart、UniPart、PartVerse、PatchAlign3D、X-Part、BANG、AutoPartGen）中，HoloPart / PatchAlign3D 仍依赖上游 3D 分割，PartCrafter / PartPacker 只能控制数量，BANG 缺少逐部件监督导致细粒度差，AutoPartGen 自回归带来误差累积。  
  **CubePart 的相对位置**：是第一个把"用户定义的开放词汇 schema"作为推理时显式控制信号、且完全 3D 原生、不依赖 2D 分割的方法。其 schema-aware finetuning + Cross-Part Attention 残差块是相对同类工作最明确的两个新设计。
- **3D 部件数据集（2.3 节）**：与 PartVerse / PartVerse-XL / PartObjaverse-Tiny 相比，CubePart 数据流水线在**规模（462K vs. 40K vs. 12K）、自动化（无需人工修正）、命名形式（name vs. caption）** 三个维度上都不同。ShapeNetPart、PartNet 这类闭集词表数据集不在 CubePart 的对标范围。

（**个人判断**）从工程落地上看，CubePart 的核心相对优势不在"部件质量"或"几何质量"任何单一指标，而在 **"控制接口的语义清晰度"**——它让"先有 schema、再生成 mesh"成为一等公民，这是 HoloPart / PartCrafter / OmniPart 都还做不到的事。

---

## 7. 局限与批判性评价

论文 §7 与附录明确给出三个局限：

1. **只支持刚体分解**。当前模型聚焦刚性部件，对需要顶点变形（蒙皮/skinning）的角色动画无能为力。未来方向之一是**在部件几何之外预测骨骼绑定权重**。
2. **几何穿透未完全消除**。尽管 Cross-Part Attention 显著减少了 interpenetration，相接部件的接缝处仍可能出现交叉。这一问题在接缝复杂、几何细节密的资产上更明显（见 Figure 11 挖掘机示例）。
3. **空间与位置标识噪声**。`front-left` / `rear-right` 这类方向标识仍可能被镜像反转。根因是训练数据中 VLM 偶尔会混淆"物体局部坐标系"和"相机视角空间"——这个问题从数据层就继承了。

**（个人判断）** 还有几个值得补充的局限：

4. **schema 长度上限**。数据流水线只保留 2–32 个部件的资产，**超过 32 的 schema 没有训练覆盖**。这意味着像复杂机械、整栋建筑这样的高度部件化对象不在模型能力范围内。
5. **数据引擎对 GPT-5 的强依赖**。聚类与命名完全依赖 GPT-5 的多视角推理能力，且作者明确说 GPT-5 比 GPT-4o 更准。这意味着**复现成本与上游 VLM 能力绑定**，一旦 GPT-5 不可用或行为变化，pipeline 就会受影响。
6. **推理时延随部件数线性增长**。论文没有给出大 $N$（如 30+）时的时延与显存数据，但 Stage 2 是逐部件循环，**长 schema 的端到端时延可能进入 10 秒级**。
7. **schema 遵从性的隐式监督**。没有显式的"schema 损失"，所有部件都靠文本提示与 Cross-Part Attention 隐式约束——schema 命中率的边界条件还没被充分研究。
8. **评测集规模偏小**。PartObjaverse-Tiny 仅 200 个样本，且论文未公布 inter-annotator agreement，对评测稳健性的把握有限。
9. **文本条件 vs 视觉条件的边界**。CubePart 选纯文本条件以避免 2D 视角依赖，但**实际游戏美术团队更可能"看到一张概念图想生成 3D 资产"**。论文没有探索把文本 schema + 图像参考联合作为条件的可能性，**这是一个清晰的工程缺口**（个人判断：技术上把 Stage 1 的 Qwen-VL 替换为带图像输入的 VL 编码器即可，论文没做可能是因为几何精度会下降）。

---

## 8. 复现与实践建议

**可获取资源**：

- 论文 PDF：https://arxiv.org/pdf/2605.28763
- 项目页：https://cubepart.github.io/
- 代码：https://github.com/Roblox/cube/tree/main/cubepart
- 模型权重：https://huggingface.co/Roblox/cubepart
- 在线 Demo：https://huggingface.co/spaces/Roblox/cubepart-demo

**复现成本估计**（论文 §3.2 / §3.3 与代码 README 综合，**个人判断**）：

- **数据重建**：完全重做 462K 资产的数据流水线需要 GPT-5 的多视角推理（OpenAI API 成本 + 速率限制），加上数千 GPU-hours 的 SDF / Marching Cubes 后处理。**复现门槛最高的就是这一步**。
- **训练资源**：24 块 H200 × 约 3.5 天（Stage 1 约 3 天 + Stage 2 约 18 小时），约 **2000 GPU-hours**。H100 / A100 集群预计更慢。
- **推理资源**：H200 单卡约 5–7 秒/样本，**A100 / 4090 级别没有公开数据**，需自行 benchmark。
- **代码依赖**：Qwen-VL（精简化版本，未直接公开下载链接——个人判断：要么作者把改造后的权重与原版权重分开发布，要么需自行做 distill）、3DShape2VecSet VAE、Shape VAE 解码器等。

**工程实践建议**：

- **先验证下游应用对刚性的需求**：若项目一定要蒙皮角色，CubePart 现阶段不适用。
- **schema 长度控制在 8–16 之间**最稳，超长 schema 容易出现细节丢失。
- **对左右/前后敏感的部件**（`front left wheel` / `rear right wheel`）要预留后处理校验，必要时人工对调。
- **若复现数据流水线**，建议先用 GPT-4o 跑 100 个样本做冒烟测试，再切到 GPT-5 跑全集，避免一次性花完 API 预算。
- **复用 Stage 1 单独使用**：Stage 1 本身就是一个高质量文本 → 整体网格 vecset 扩散模型，可以脱离 Stage 2 单独用。

---

## 9. 个人启发与后续问题

**与已有论文笔记的对照**：

- 与 [[PhysX-Omni - 单图生成仿真就绪三维资产]] 关注"刚/柔/关节体自动拆分 + 仿真就绪"不同，CubePart 走的是"开放词汇 schema 显式控制 + 刚体部件 + 游戏引擎直接挂脚本"路线，二者在"3D 资产生成"这条主线上互补。
- 与 [[TRELLIS - 结构化潜变量三维生成]]、[[TRELLIS.2 - 原生几何与材质的紧凑三维潜变量生成]] 共享 vecset 扩散与结构化潜变量思想，但 CubePart 把"潜变量粒度"从整体级别拆到部件级别，是对 SLAT 类表示的自然延伸。
- 与 [[SIMART - 静态网格到仿真就绪铰链资产]] 同样关心"3D 资产如何被引擎消费"，但 SIMART 是把已有 mesh 转换为铰链资产，CubePart 则是**直接生成可被消费的资产**——前者是后处理，后者是生成端介入。
- 与 [[DoSs - 超二次曲面基元扩散三维形状生成]]、[[QuadGPT - 原生四边形网格自回归生成]]、[[LATO.2 - 顶点与拓扑因子化网格生成]] 等侧重几何表示/拓扑的论文相比，CubePart 不追求表示创新，而是**把表示层（vecset）作为前提、把控制接口作为创新点**。

**值得继续跟踪的问题**：

1. **蒙皮/骨骼扩展**：能否把"刚性 → 可变形"做到不破坏现有训练范式？例如在 Stage 2 之上加一层 LBS 预测头，或者从数据流水线侧引入带骨骼标注的资产。
2. **更长的 schema**：32 个部件的硬上限是否能放宽到 100+？若能，将打开"完整建筑 / 复杂工业装配"的应用场景。
3. **空间语义的去噪**：能否用显式的"局部坐标系监督"减轻 `left/right flip` 现象？一种思路是让 schema 文本在标注时带上相机坐标说明。
4. **schema 与其他控制信号联合**：把 schema 与图像 / 草图 / 参考 mesh 联合作为条件，进一步降低长尾 query 的歧义。
5. **数据流水线的开源**：目前依赖 GPT-5 是最大不确定性。开源版若能用 Qwen-VL-Plus / InternVL3 等替代将大幅降低复现门槛。

**总体判断**（**个人判断**）：

- **重要性**：★★★★☆（4/5）。它解决的不是"几何质量再高一点"的边际改进，而是 3D 资产生成的"接口问题"——把开放词汇 schema 变成了一等公民，对游戏 / 仿真 / 具身智能资产生态有直接价值。
- **成熟度**：约 4/5。代码、模型、Demo 均已发布，Hugging Face 上可即时试玩；距离生产可用还差蒙皮、方向一致性、schema 长度三个工程缺口。
- **跟踪价值**：高。若 Roblox 沿 Cube 系列继续推进，下一步大概率是 rig-ready / behavior-ready / editable 3D assets，值得与 PartCrafter / HoloPart / OmniPart / PartPacker 的演化路径并列观察。

---

## 10. 材料来源

### 10.1 论文与官方材料

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| 论文 PDF（未本地化） | arXiv PDF | https://arxiv.org/pdf/2605.28763 | 全文 | 2026-08-21 | 精读主来源 |
| arXiv HTML | 在线 HTML 版 | https://arxiv.org/html/2605.28763 | 全文 + 图 | 2026-08-21 | 精读主来源、图源 |
| 项目页 | Roblox 官方项目页 | https://cubepart.github.io/ | 摘要 + 视频 | 2026-08-21 | 验证与补充 |
| GitHub 代码仓 | Roblox 官方代码 | https://github.com/Roblox/cube/tree/main/cubepart | — | 2026-08-21 | 复现参考 |
| Hugging Face 模型 | 模型权重 | https://huggingface.co/Roblox/cubepart | — | 2026-08-21 | 复现参考 |
| HF Demo | 在线 Demo | https://huggingface.co/spaces/Roblox/cubepart-demo | — | 2026-08-21 | 能力验证 |

### 10.2 关键图片溯源

所有图片位于 `assets/CubePart - 开放词汇部件可控三维生成/`。

| 论文图号 | 本地文件名 | 类型 | 原始 URL | 用途章节 |
|---|---|---|---|---|
| Figure 1（teaser） | `teaser.png` | PNG | arXiv HTML `2605.28763v1/teaser.png` | §1.3 |
| Figure 2（method） | `method.png` | PNG | arXiv HTML `2605.28763v1/method.png` | §3.1 |
| Figure 3（Cross-Part Attention） | `cross_part_attention.png` | PNG（由 SVG 转换） | arXiv HTML `2605.28763v1/part_attn.svg` | §3.3 |
| Figure 4（数据标注对比） | `qualitative_comparison_data_processing.png` | PNG | arXiv HTML `2605.28763v1/qualitative_comparison_data_processing.png` | §4.2 |
| Figure 5（Set-of-Mark 纹理视图） | `som_rendering.png` | PNG | arXiv HTML `2605.28763v1/figures/data/render_front_tilt_texture.png` | §4.2 |
| Figure 6（多部件定性对比） | `qualitative_comparison.png` | PNG | arXiv HTML `2605.28763v1/result_multiMesh_v4.png` | §5.2 |
| Figure 8（两阶段定性） | `qualitative_twostage.png` | PNG | arXiv HTML `2605.28763v1/result_full_mesh.png` | §5.2 |
| Figure 9（可变 schema） | `varying_schema.png` | PNG | arXiv HTML `2605.28763v1/different_parts_new.png` | §5.4 |
| Figure 10（行为驱动） | `fig_behavior_v6.png` | PNG | arXiv HTML `2605.28763v1/fig_behavior_v6.png` | §5.4 |
| Figure 11（失败案例） | `failure_cases.png` | PNG | arXiv HTML `2605.28763v1/failure_example.png` | §5.4 / §7 |

### 10.3 信息缺失说明

- **数据流水线的 GPT-5 提示词全文**：论文未公开完整 prompt（只在附录给出要点），故"指令 (1)(2)(3)" 在 §4.2 中以"按功能或逻辑关系分组 / 分配简洁名称 / 允许单例聚类"概括，不照搬原文。
- **Stage 1 / Stage 2 的推理步数（ODE 求解步数）**：论文未披露具体 ODE 步数，本笔记的"约 50 步"为个人判断、未在文中出现。
- **H100 / A100 上的推理时延**：论文只给 H200 数据，**§8 复现部分关于 A100 / 4090 的讨论为个人判断**。
- **PartObjaverse-Tiny 的人工一致性指标**：论文未公开 inter-annotator agreement，**§7 第 8 条为个人判断**。
