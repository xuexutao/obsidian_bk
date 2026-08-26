---
type: survey
status: done
domain: VLM
paper: null
year: 2026
arxiv: null
doi: null
source: null
project: null
code: null
tags: [VLM, Survey, 3D-LLM, Point-Cloud, Mesh, Spatial-Reasoning, Tokenization]
created: 2026-08-25
updated: 2026-08-26
---

# 三维多模态大语言模型架构：3D 如何进入 LLM 的 token 空间

## 基本信息

- **领域**：VLM / 3D 多模态
- **类型**：领域综述（Survey）
- **主题**：3D 表征如何接入 LLM——2D 提升路线、点云编码器路线、Mesh 序列化路线三条主线，及其 token 预算、语言对齐、评测可信度三重约束
- **覆盖范围**：2022.03（PointCLIP）— 2026.08（MeshWeaver / SpatialStack / ReVSI），重点为 2025 年之后
- **写作时间**：2026-08-25（2026-08-26 重构扩写）
- **报告定位**：不做模型清单，而是回答一个问题——**把三维世界塞进一个为一维 token 序列设计的模型，代价是什么，各家怎么付这个代价**。本文与 [[三维网格生成表征演进 - Survey]] 分工：那篇讲「怎么生成网格」，本篇讲「网格/点云怎么被 LLM 消费」。

---

## 1. 核心问题：一个维度错配

LLM 的全部机制建立在一个假设上：输入是**一维、有序、离散**的 token 序列。而三维资产恰好三条全不满足——它是三维的、无固有顺序的、连续的。所有 3D-LLM 的架构差异，归根结底都是在回答同一个问题的不同侧面：

> **如何把一个三维结构压成一段 token，且压缩过程中不丢掉 LLM 需要的东西？**

这里有三重相互撕扯的约束，构成了本文的分析主轴：

1. **Token 预算**。一张 5000 面的网格（还只是低模）朴素序列化要 4.5 万 token；一个场景的点云特征动辄上万。而 LLM 的上下文既贵又有限，注意力还是平方复杂度。
2. **语言对齐**。LLM 的词嵌入空间是被语言塑造的。一个纯几何自监督训练出的特征，即使几何精度很高，也和语言空间毫无关系——连接器（通常只是几层 MLP）没有能力从零学出这个映射。
3. **几何保真**。压缩必然丢信息。丢掉的是无关冗余还是恰好是空间推理需要的那部分，决定了模型是真懂 3D 还是在背语言先验。

第 6 节会给出一个不舒服的结论：**目前主流评测基本测不出第 3 条**，因此很多架构结论的证据强度比表面看起来弱得多。

### 1.1 一个已经失效的二分法

早期（2023–24）讨论 3D-LLM 习惯按「有没有 3D 编码器」二分：

| 架构类型 | 说明 | 优点 | 缺点 |
|---|---|---|---|
| **Wo 3D encoder** | 基于 2D VLM：多视角图像输入，辅以深度/位姿等 3D 线索 | 摆脱高质量 3D 资产稀缺的限制；直接继承 2D VLM 的语义能力与数据规模 | 几何感知受制于 2D，尺度与遮挡歧义难解 |
| **Wi 3D encoder** | 基于点云 / Mesh 的原生 3D 编码器 | 直接编码 3D 资产，几何感知更强 | 受 3D 资产质量与规模限制；无法利用已有的海量多视角数据 |

这张表**仍然是理解这个领域的正确入口**，但作为 2026 年的分类框架它已经失效了，有三重证据：

**其一，绝大多数「Wi」方法其实是混合的。** Chat-Scene 用 Uni3D（3D）+ **DINOv2**（2D）；LEO 用 PointNet++ + **OpenCLIP ConvNeXt**；Pts3D-LLM 是 **SigLIP 为主、3D 特征为增强**；3D-LLaVA 的语言对齐来自 **CLIP-ViT-L 的 2D→3D 蒸馏**。在本文覆盖的方法里，推理时真正不碰 2D 的只有 **LL3DA** 和 **Grounded 3D-LLM** 两家。

**其二，2D 分支往往比 3D 分支更强。** Chat-Scene 在 Multi3DRefer F1@0.5 上的消融：3D-only **41.2** → 2D-only **45.8** → 两者兼用 **50.2**，论文原话是「*semantic information from 2D visual contexts is more crucial than spatial information from 3D point clouds*」。Inst3D-LMM 同向：纯几何 39.2 → 加多视图 2D **46.7**（+10.7）。PQ3D 把它拆得更细：voxel 67.8 → +point 74.6 → +image 80.3（Scan2Cap）。

**其三，「3D 编码器」被重新定义了。** 2025–26 的共识做法是：保留 2D CLIP/SigLIP 编码器，再并上一个**冻结的前馈多视角几何 Transformer**（VGGT / CUT3R）——它只吃 RGB 视频，不需要 GT 深度、GT 位姿、点云或 Mask3D。这是一个**吃 2D 输入的原生 3D 编码器**，它并没有反驳「3D 数据稀缺」这个论点，而是绕开了它（VGGT 本身在大规模多视角数据上训练并被冻结）。

所以本文改用**「3D 信息以什么形式进入 token 序列」**作为分类轴，分三条路线：

| 路线 | 3D 怎么变成 token | 代表 | 本文章节 |
|---|---|---|---|
| **A. 2D 提升** | 2D patch token + 3D 位置信息（位置编码 / 深度 / BEV 图） | 3D-LLM、LLaVA-3D、Video-3D LLM、GPT4Scene | §2 |
| **B. 点云编码器** | 点/patch/superpoint/物体 token | PointLLM、LL3DA、3D-LLaVA、Chat-Scene、Pts3D-LLM | §3–4 |
| **C. Mesh 序列化** | 顶点/面文本、学习词表、或**生成程序** | LLaMA-Mesh、MeshLLM、MeshCoder / MeshGPT 系 | §5 |

---

## 2. 路线 A：2D 提升——不要原生 3D 编码器

这条路线的赌注是：**3D 能力的瓶颈是数据与表述方式，不是架构**。既然 2D VLM 已经在数十亿图文对上训练过，不如把 3D 信息「翻译」成它已经会处理的形式。

### 2.1 3D-LLM：把多视角特征反投影成 3D 特征场

**3D-LLM: Injecting the 3D World into Large Language Models**（NeurIPS 2023 Spotlight，[arXiv 2307.12981](https://arxiv.org/abs/2307.12981)）是这条路线的奠基工作。它不直接吃点云，而是**渲染多视角图像 → 提取 2D 稠密特征 → 反投影回 3D**。

![3D-LLM 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/3dllm-1.png)

*图 1：3D-LLM 的整体流程。左侧是三种 3D 特征构建策略（由多视角 2D 特征提升而来），中间是加入位置编码与 location token 的 3D 特征序列，右侧是复用的 2D VLM（BLIP-2 / Flamingo）。来源：论文原图。*

三种特征构建策略对应三种数据条件：

1. **直接重建（direct reconstruction）**：用 GT 相机内外参从渲染的 RGBD 重建点云，特征直接赋给点。要求位姿完美。
2. **特征融合（feature fusion）**：用 **gradslam** 把 2D 特征融进 3D，容忍带噪的深度与位姿。
3. **神经场（neural field）**：每个体素除密度/颜色外再带一个特征，用 MSE 对齐 3D 与 2D。适用于只有 RGB 的数据。

2D 特征提取「following **ConceptFusion**」（论文未直接点名 CLIP/SAM；官方实现是 Mask2Former 或 SAM 出 mask → CLIP/BLIP 出每 mask 特征）。3D 位置以 x/y/z 各 $D_v/3$ 维的 sin/cos 位置编码**拼接**（不是相加）到特征上，另加离散化的 **location token** `<x_min,y_min,z_min,x_max,y_max,z_max>`。

数据是 **30 万+**（不是常被误引的 100 万，那是 LLaVA-3D-Instruct-1M）3D-语言对，用三种 prompt 机制生成（boxes-demonstration-instruction / ChatCaptioner / revision-based）。

**结果里有一个值得注意的裂缝**：它在生成类指标上赢，在 EM 上**输给不含 LLM 的 ScanQA 基线**——

| 模型 | B-1 | B-4 | CIDEr | EM |
|---|---|---|---|---|
| ScanQA 基线（非 LLM） | 30.2 | 10.1 | 64.9 | **21.0** |
| flamingo-MultiView | 25.6 | 8.4 | 55.0 | 18.8 |
| 3D-LLM (BLIP2-flanT5) | **39.3** | **12.0** | **69.4** | 20.5 |

ScanRefer 定位只有 **30.3** Acc@0.25（平均 IoU 24.9，平均距离 1.03 m）——这是纯 2D 提升在**定位**任务上的短板，后续工作正是从这里切入。

论文自陈的局限很诚实：「the 3D feature extractor relies on 2D multi-view images, and thus all 3D scenes need to be rendered... which introduces an additional rendering process」。

### 2.2 LLaVA-3D：3D Patch——最小改动路线

**LLaVA-3D**（ICCV 2025，[arXiv 2409.18125](https://arxiv.org/abs/2409.18125)）把改动压到极限：不要 3D 编码器、不要渲染、不要反投影特征场，只在 CLIP 的 2D patch 特征上**逐元素相加一个 3D 位置编码**。

![LLaVA-3D 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/llava3d-1.png)

*图 2：LLaVA-3D 的 3D Patch 机制。2D patch 特征加上由深度与相机内外参算出的 3D 位置编码，经体素池化压缩后与文本 token 拼接进 LLM。它保留了 2D VLM 的原有能力，因此可以联合 2D+3D 指令微调。来源：论文原图。*

核心式子极简：$X_{3D} = X_p' + P'$。世界坐标 $P \in \mathbb{R}^{V\times 3\times w\times h}$ 由**最近邻深度 + 已知相机内参与外参**得到，再经一个**可学习的两层 MLP** 编码（对比 3D-LLM 用的是拼接式正弦编码）。

**体素池化是必需的，不是优化**：20 个视角在池化前是 **11,520 个 token**，而 LLaVA-1.5 的上限约 3,096。体素尺寸的消融很干净（ScanQA EM / SQA3D EM）：0.4 m → 24.1/53.2、0.3 m → 25.9/54.8、**0.2 m → 27.0/55.6**。FPS 池化更差。

它保留 2D 能力的方式是**联合 2D+3D 指令微调**：v3 基于 LLaVA-Video-7B，MVBench 58.1（基线 58.6）、VideoMME 62.8（基线 63.3）——几乎无损。收敛也快，论文称比现有 3D LMM **快 3.5 倍**，附录 B 显示 300 步内超过 LEO。

**3D 位置编码的消融最能说明问题**：+2.9 ScanQA、+3.3 SQA3D、+8.8 MMScan，但 **Scan2Cap 是 +50.1%**。几何注入对**描述与定位**的作用远大于对 QA——这个不对称会在 §6 反复出现。

定位靠一个 **location token**，其末层嵌入喂给一个 grounding decoder 出框，**推理时不需要 Mask3D**——这是它相对 Video-3D LLM / Chat-Scene 的结构优势。

> ⚠️ 版本陷阱：v1 的数据集叫 **LLaVA-3D-Instruct-1M**（2D+3D 混合），v3 与官方仓库叫 **LLaVA-3D-Instruct-86K**（ScanNet / 3RScan / Matterport3D）+ LLaVA-Video 的 2D 数据。引用时要说明版本，两版分数差别很大（ScanQA 91.7 → 103.1）。

### 2.3 Video-3D LLM 与 GPT4Scene：把场景当视频 / 当图

**Video-3D LLM**（CVPR 2025，[arXiv 2412.00493](https://arxiv.org/abs/2412.00493)）的立场是：场景本来就是**带位姿的 RGB-D 视频**产生的，那就按视频处理。3D 正弦位置编码逐元素加到 ViT patch token 上，每个 patch 的坐标取其反投影像素坐标的**平均**（优于只取 patch 中心）。

它的亮点是 **Maximum Coverage Sampling（MCS）**：把选帧建模成场景体素上的最大覆盖问题（NP-hard），贪心求解到预算 K 或约 95% 覆盖率。消融显示 8 帧时均匀采样 48.93 vs 最大覆盖 **53.47（+4.54）**，但 32 帧时两者趋同——**MCS 的价值主要是省成本**（约 18 帧自适应 vs 32 帧；309 ms vs 1050 ms）。

位置编码的消融同样呈现那个不对称（Scan2Cap C@0.5 / ScanRefer / ScanQA EM）：无 PE 31.03/57.50/30.03、MLP 76.23/59.63/29.62、正弦 **83.76**/58.11/30.09。**位置编码在描述任务上值约 53 个 CIDEr，在 QA 上约等于 0。**

**GPT4Scene**（ICCV 2025，[arXiv 2501.01428](https://arxiv.org/abs/2501.01428)）走得更极端：**完全没有 3D 编码器，也没有任何 3D 位置编码**。它从重建的点云渲一张 **BEV 俯视图**，配上视频帧，再用 **STO（Spatial-Temporal Object）标记**给物体在 BEV 与各帧间保持一致的 ID。骨干是 Qwen2-VL-7B。

这套纯视觉方案目前是 **2D 路线里最强的定位成绩**：ScanRefer **62.6/57.0**、Multi3DRefer **64.5/59.8**、Scan2Cap 86.3 CIDEr@0.5。它还罕见地做了鲁棒性验证——用 **SLAM3R** 替代 GT 重建，并报告 50/100/200 帧间隔下的表现；去掉 30% 标记只从 43.6 掉到 42.7 ROUGE。代价是需要昂贵的重建 + 实例分割预处理。

### 2.4 这条路线的隐藏依赖：GT 深度与 GT 位姿

一个容易被忽略的问题：LLaVA-3D 与 Video-3D LLM 都依赖 ScanNet 的 **GT 深度与 GT 外参**，而**两者都没有做「换成估计值会怎样」的消融**。最干净的数字来自 **VG-LLM**（NeurIPS 2025，[arXiv 2505.24625](https://arxiv.org/abs/2505.24625)）：

| 空间信号 | ScanRefer @0.25 | Scan2Cap CIDEr |
|---|---|---|
| 无 | 47.4 | 59.8 |
| 预测的相机信息 | 49.1 | 62.5 |
| 预测的深度信息 | 49.2 | 62.0 |
| **VGGT 几何特征** | **51.0** | **74.1** |

也就是说：依赖**估计**的位姿/深度相比隐式几何特征要付约 **12 个 CIDEr** 的代价，而显式喂预测出的空间信息**反而在注入噪声**。VLM-3R 的消融更尖锐：完整 60.90、纯 2D 58.12、**显式点云融合 57.87——比纯 2D 还差**。

这解释了为什么 2025 年之后主流做法转向「冻结的前馈几何编码器」而非「显式几何量」，见 §7.2。

---

## 3. 路线 B 之一：点云 3D-LLM 的架构设计

这条路线直接消费点云。所有方法都是「点云编码器 → 连接器 → LLM」三段式，差异集中在两处：**token 预算怎么花**，以及**定位结果怎么输出**。

### 3.1 PointLLM：物体级基线，且证明了「不压缩更好」

**PointLLM: Empowering Large Language Models to Understand Point Clouds**（ECCV 2024 **Oral**，[arXiv 2308.16911](https://arxiv.org/abs/2308.16911)）理解**带颜色的物体点云**，按人类指令生成回应。

![PointLLM 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointllm-1.png)

*图 3：PointLLM 的三段式架构——点云编码器 $f_{pe}$、投影器 $f_{proj}$、LLM 主干 $f_{llm}$。点云与文本 token 拼接后交给 LLM 自回归生成。来源：论文原图。*

架构参数：

- 输入点云 $P \in \mathbb{R}^{n\times d}$，$n=8192$ 点、$d=6$（xyz + RGB）；输出点特征序列 $X \in \mathbb{R}^{m\times c}$，**$m=513$、$c=384$**。
- 编码器是用 **ULIP-2** 在 Objaverse 上多模态预训练过的 **Point-BERT**（注意：ULIP-2 是编码器的预训练配方，不是一个 3D-LLM，见 §4.4）。
- 投影器是**三层线性层 + GeLU**（消融显示三层最优，中间维 1024、2048）。
- LLM 主干是 LLaMA/Vicuna 的 **7B 与 13B** 检查点，词表 $V=32003$（新增 `<p_start>`、`<p_end>`）。

**两阶段训练**：第一阶段特征对齐，冻结点云编码器与 LLM，只训投影器，用 660K 简短描述；第二阶段指令微调，冻结点云编码器，联合训练 LLM 与投影器，用 70K 复杂指令。总数据 730K 样本 / 66 万个 Objaverse 物体。

> ⚠️ **维度的引用陷阱**：论文正文写的是「c′=5120 (7B model) or c′=5120 (13B model)」——**两处都是 5120，这是论文自己的笔误**。发布的 HuggingFace 配置证明架构上应为 `PointLLM_7B_v1.2 → hidden_size 4096`（32 层）、`PointLLM_13B_v1.2 → hidden_size 5120`（40 层）。引用时应写「按发布配置为 4096 / 5120；论文正文两处均误写为 5120」，**不能把 4096 说成论文的说法**。
>
> 另：ECCV 2024 的确凿身份是 **Oral**；「Best Paper Candidate」只出现在项目 GitHub 的 badge 上，官方 ECCV 材料无法核实。期刊版是**另一篇独立标题**的论文——*PointLLM-V2: Empowering LLMs to **Better** Understand Point Clouds*，TPAMI 2025。

**它最有价值的贡献之一是一个负结果**：不做 max-pooling、把全部 513 个 token 都喂给 LLM，比池化更好——7B 上 **52.82 vs 48.72**，代价是 A100 小时从 34 涨到 126（约 4 倍）。也就是说，在物体级别上，**token 压缩是有实际精度代价的**，这为 §3.2 那些激进压缩方案提供了对照。

结果：生成式分类平均 **52.82**（7B）/ **53.39**（13B），对比 LLaVA-13B 44.17、3D-LLM 45.25。局限是**只做物体级**（场景级「仅能给出合理输出」，留作未来工作），且 **13B 比 7B 幻觉更多**（0.84 vs 0.66）。

### 3.2 LL3DA：把整个场景压进 32 个 token

**LL3DA**（CVPR 2024，[arXiv 2311.18651](https://arxiv.org/abs/2311.18651)）解决的是场景级的 token 爆炸，同时引入**视觉 prompt**（点击、3D 框）来消解杂乱场景中的指代歧义。

它的核心动作是本文最激进的压缩：场景编码器输出 **1024 × 256** 的场景嵌入，但**一个都不直接进 LLM**——全部经 cross-attention 压进 **32 个可学习 query token**，作为前缀喂给 LLM。**LLM 的上下文开销与场景大小完全无关**。

- 编码器：ScanNet 检测预训练的 masked transformer（Vote2Cap-DETR 血统，PointNet++ set-abstraction 分词），**全程冻结**（论文明说为省显存）。
- 连接器 **Interactor3D**：32 个 query、768 维、6 层、12 头，对 [query ; 文本 ; 视觉 prompt] 做**共享自注意力**，再 cross-attention 进场景嵌入——是**早融合**而非末端拼接（消融：早融合比直接拼接 prompt 好 **+3.45** C@0.5）。
- 视觉 prompt：点击坐标归一化到 [0,1] 后用 **3D Fourier 位置编码** → FFN → **8 个 token**；3D 框取检测器的 **ROI 特征** → FFN → 8 个 token。
- LLM：**OPT-1.3B，冻结，fp16**。可训练参数约 **111M**（不到 LLM 的 10%）。

训练极其便宜：**8×RTX 3090，一天内**跑完 ~100k 步。这是本文覆盖的方法里成本最低的一档，也印证了 §6.3 那个「LLM 规模不重要」的规律。

结果：Scan2Cap CIDEr@0.5 **65.19**（Vote2Cap-DETR 61.81）、Nr3D CIDEr@0.5 **51.18**（43.84）、ScanQA CIDEr **76.8**。视觉 prompt 值 **+6.12** CIDEr。

**一个诚实的负结果值得写进来**：generalist 模型在 Nr3D 上只有 **23.94** C@0.5，finetune 后才 51.18。论文归因于 ScanRefer 式与 Nr3D 式描述混训时缺少任务区分信号——**「一个模型打通所有任务」在 3D 上还没真正成立**。

### 3.3 3D-LLaVA：用 OST 把选特征、编码 prompt、解码 mask 合成一个模块

**3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer**（CVPR 2025，[arXiv 2501.01163](https://arxiv.org/abs/2501.01163)）的目标是去掉外部依赖：既不要多视角特征提取，也不要外部检测器。

![3D-LLaVA 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/3dllava-1.png)

*图 4：3D-LLaVA 架构。点云经 Sparse 3D U-Net 与 Omni Superpoint Transformer（OST）得到 superpoint 特征；视觉 prompt（点/框/mask）经同一个 OST 的对齐头进入相同嵌入空间；LLM 输出 `[SEG]` 时，其隐状态经投影回到冻结的 OST 解出 mask。LLM 主干冻结，只训 LoRA 与两个投影器。来源：论文原图。*

**「prompt 与点云对齐」的确切含义**——这是你笔记原来只写了一句话的地方。答案是：**没有独立的对齐模块**。OST 一个模块干三件事：

1. **视觉特征筛选器**：按 objectness 分数（前景类的最大分）保留 **top-K=100** 个 superpoint；
2. **视觉 prompt 编码器**：**无参数**的视觉采样器处理点击（3-NN 插值）与框/mask（平均池化），并用 **masked attention 阻断 prompt → superpoint 的信息流**；
3. **mask 解码器**：LLM 吐出 `[SEG]` 时，其前一个隐状态经投影器 $W_S$ 与 superpoint query 拼接，送回**冻结的** OST 出 mask。

prompt 与点云落在同一空间，是因为二者都经过 OST 的**对齐头**——视觉 prompt 嵌入 $Z_P$ 与视觉特征嵌入 $Z_V$ 都是对齐头输出，再由共享投影器 $W_V$ 映射进 LLM token 空间。而**与语言的**对齐来自第一阶段的 **2D→3D 蒸馏**：以 **CLIP-ViT-L**（LLaVA-1.5 的编码器）为教师，靠像素-点对应关系提升到 3D 并按 superpoint 池化——所以 3D 特征继承了一个已经与语言对齐的 2D 空间。

编码器是 **Sparse 3D U-Net**（submanifold 稀疏卷积）+ 自底向上聚类 + superpoint 平均池化，并用**距离自适应自注意力**替代标准自注意力。LLM 是 Vicuna-1.5-7B + LoRA。

训练两阶段：S1 在 **ScanNet200 上 512 epoch** 联合训练 U-Net 与 OST（实例分割 + 2D→3D 蒸馏混合损失）；S2 指令微调，**冻结 U-Net、OST、LLM 主体**，只训视觉投影器、SEG 投影器与 LoRA。

结果：ScanRefer 指代分割 mIoU **43.3**、Multi3DRefer mIoU **42.7**、ScanQA CIDEr **92.6**、SQA3D EM **54.5**、Scan2Cap CIDEr@0.5 **78.8**。消融显示 superpoint token 超过 200 后收益趋平——与 LL3DA 的 32 token 一起，勾出了场景级 token 预算的合理区间。

### 3.4 Pts3D-LLM：Apple 的「token 结构」对照实验

这是你笔记里标为「Point3D LLM 2025 Apple」的那篇。**真实身份需要更正**：

![Pts3D-LLM 首页](assets/三维多模态大语言模型架构%20-%20Survey/figures/point3dllm-1.png)

*图 5：Pts3D-LLM 的方法总览（论文首页图）。(a) 用统一空间 token 解决 3D VG / 3D captioning / 3D QA；(b) token 由 SigLIP 图像特征、Sonata 点云特征与位置编码构成，点特征按最近邻插值汇到图像 patch 位置；(c) 提出 FPS6D 做视点敏感的子采样；(d) 雷达图显示其在多个 3D 场景理解基准上达到 SOTA。来源：论文首页。*

- **arXiv 标题是 Pts3D-LLM**：*Studying the Impact of Token Structure for 3D Scene Understanding With Large Language Models*（[arXiv 2506.05689](https://arxiv.org/abs/2506.05689)），作者 Hugues Thomas、Chen Chen、Jian Zhang（Apple）。「Point-3D LLM」只是 Apple ML 网站的页面标题写法，引用应用 arXiv 形式。
- **目前仍是 preprint**，comments 字段只写「Main paper and appendix」，无发表信息。

它的定位不是新架构，而是一个**控制变量的对照研究**：固定骨干与参数量，比较**视频式 token 结构 vs 点式 token 结构**。

- 图像侧 **SigLIP** ViT，3D 侧 **Sonata 预训练的 Point Transformer V3**；点特征按**最近邻插值**投到图像 patch 位置；图像语义 + 3D 特征 + 正弦位置编码**逐元素相加**成统一空间 token。
- 连接器是**线性层**（优于 MLP 与全量微调）；LLM 是 **LLaVA-Video 7B**。
- **FPS6D**：在 6 维 `[点 xyz ; 相机 xyz]` 上做最远点采样，得到视点敏感的点序，兼顾空间覆盖与视角多样性。
- 最优配置**冻结 Sonata**，只训线性投影器。单阶段多任务，1 epoch，8×H100。

结果（多种子平均）：ScanRefer Acc@0.25 **60.2**、Multi3DRefer F1@0.25 **53.5**、Scan2Cap CIDEr@0.5 **83.0**、ScanQA EM **53.4**、SQA3D EM **40.7**，归一化总分 **101.1±0.6** vs Video-3D LLM 99.5、LLaVA-3D 91.9、Chat-Scene 91.5。

它自陈了一个很关键的自我怀疑：**LLM 骨干是视频预训练的，因此可能天然偏向视频式 token 结构，从而低估了点式结构的潜力**。这种诚实在这个领域不常见，值得记下。

### 3.5 GPT4Point、Chat-Scene、LEO、ShapeLLM、Grounded 3D-LLM

余下五家各自补上一个维度：

**GPT4Point**（CVPR 2024 Highlight，[arXiv 2312.02980](https://arxiv.org/abs/2312.02980)）是唯一**同时做理解与生成**的：BLIP-2 式 **Point-Q-Former** 对齐点文特征后，一路接 LLM 出文本，另一路经一个 FC 投进 CLIP token 空间去条件化 **Point-E** 扩散模型，从而「输入低质点云 + 文本 → 输出保持几何与颜色的高质 3D 资产」。只训 **~110M** 参数（对比 PointLLM 的 13.3B），却在 Objaverse-XL-LVIS 上 CIDEr **32.2** vs PointLLM 26.2。数据是 **Pyramid-XL** 引擎在 Objaverse-XL 上产出的 **>100 万** 点文对（Level 1 用 BLIP-2 出 ~100 万单视图短描述、Level 2 用 GPT-4 合成 66 万多视图描述、Level 3 用 Qwen-VL 出 7 万稠密描述+QA）。

**Chat-Scene**（NeurIPS 2024，[arXiv 2312.08168](https://arxiv.org/abs/2312.08168)）的贡献是**物体标识符**：约 100 个可学习 token `<OBJ_i>` 加进词表，每个绑定到该场景的一个 Mask3D 提议。定位于是变成**生成式地预测标识符字符串**再查 mask——不需要坐标回归、不需要框头，多物体指代就是连续吐几个标识符（这正是它在 Multi3DRefer 上暴涨 +14.0 的原因）。它冻结 Mask3D + Uni3D（3D）+ **DINOv2**（2D）三个基础模型，只训两个 3 层 MLP、标识符嵌入与 LoRA-16，单阶段 4×A100 约 8 小时。

> ⚠️ **两个引用陷阱**（这类错误最难自查）：
> 1. **Chat-3D 的发表标题与 arXiv 标题不同**。arXiv 是 *Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes*（[arXiv 2308.08769](https://arxiv.org/abs/2308.08769)），但 **NAACL 2025 Findings** 发表时题为 *Data-Efficiently Learn Large Language Model for Universal 3D Scene Perception*。两个标题都要写，否则读者搜不到；发表版数字也变了（相对 GPT-4 从 75.6 → 82.2）。
> 2. **Chat-3D v2 与 Chat-Scene 是同一个 arXiv 条目改名**。标「Chat-3D v2」的数字（ScanRefer 35.9/30.4）是 2023-12 版（用 PointGroup、无 2D 分支）；标「Chat-Scene」的（55.5/50.2）是 2024-09 版。Grounded 3D-LLM 对比的是前者，因此它**并没有**超过已发表的 Chat-Scene。后继工作 **Chat-Scene++** 已在 TPAMI 2026。

**LEO**（ICML 2024，[arXiv 2311.12871](https://arxiv.org/abs/2311.12871)）把 3D 输入组织成 **Mask3D 提议上的物体中心 token**，每提议经 PointNet++ 再过一个**空间 Transformer**（用相对位置与尺寸给注意力打偏置，捕捉物体间 3D 关系）。最漂亮的设计是把**离散化动作映射到 SentencePiece 里最少用的 token**，于是文本与具身动作共享一套离散词表，一切都是同一个序列预测目标。ScanQA CIDEr **101.4**、SQA3D EM **50.0**，并在 CLIPort 操作与 Habitat 导航上给出零样本结果。注意其 ScanQA 设定会访问与问题相关的 GT 物体，3D-LLaVA 因此把它标灰为不可直接比较。

**ShapeLLM**（ECCV 2024，[arXiv 2402.17766](https://arxiv.org/abs/2402.17766)）的编码器 **ReCon++** 是**生成式掩码建模 + 跨模态对比**双分支，多视图 query token 用**匈牙利算法**做自适应视图匹配。它额外把原始 xyz 的**绝对位置编码 $E_{APE}$** 与**前缀可学习 prompt** 一并喂进投影器，以支持 6-DoF 位姿任务。ReCon++ 微调后 ScanObjectNN **95.25%**。它对本文最重要的一个数据点在 §4.5：把 PointLLM 的 ULIP-2 编码器换成 ReCon++，3D MM-Vet **+4.20%**——说明纯对比式语言对齐把几何留在了桌上。

**Grounded 3D-LLM**（preprint，[arXiv 2405.10370](https://arxiv.org/abs/2405.10370)）用 **referent token** `<ref>` 做短语级定位：`<ref>` 的隐状态经 2-MLP 出 mask 嵌入，与 CLASP 物体 query 点积匹配后取其点级 mask，用 sigmoid focal loss 训练。关键是**一个 `<ref>` 可解出多个物体**（「三把棕色椅子」），一对多（**45.2** F1@0.25）优于匈牙利一对一（44.2）。数据集 **G-SceneCap** 密度惊人：1.5K 个 ScanNet 场景上 **107K** 场景描述、**102 万**短语-区域对应、**11.1%** 对应密度（ScanRefer 4.4%、Multi3DRefer 7.2%）——「每九个 token 就有一个物体」。

它自陈了一个极其重要的局限：**它自己的 CLASP（判别式）在定位上打败了它的 LLM（生成式）**——ScanRefer 53.2 vs 48.6，检测 27.2 vs 12.1 AP。而且**LLM 规模超过 1B 收益极小**，「当前任务范围可能没有充分挑战更大规模的 LLM」。

---

## 4. 路线 B 之二：点云编码器调研

上一节的 3D-LLM 都要挑一个点云编码器。这一节梳理可选项。关键结论前置：**分类的有效轴不是「用了什么自监督信号」，而是「输出特征是否已经落在语言空间附近」**——因为连接器通常只是几层 MLP，它没有能力从零学出几何空间到词嵌入空间的映射。

### 4.1 基于点：PointNet → PointNet++ → PointMLP

**PointNet**（CVPR 2017）要解决的是**置换不变性**：点集没有固定顺序，$n$ 个点有 $n!$ 种排列，网络输出必须不变。它的构造是

$$f(\{x_1,\dots,x_n\}) \approx \gamma \circ \operatorname*{MAX}_{i=1..n}\{h(x_i)\}$$

$h$ 是共享的逐点 MLP，MAX 是沿点维的逐元素最大池化（那个对称函数），$\gamma$ 是后接 MLP。**对称性由 MAX 保证，而 $h$ 权重共享保证它不依赖下标。**

![PointNet 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointnet-1.png)![PointNet 细节](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointnet-2.png)

*图 6：PointNet 的分类与分割网络。逐点 MLP 升维后经 max-pooling 得到全局特征；分割分支把全局特征拼回每点特征。来源：论文原图。*

两条理论结果值得知道：**定理 1（万能逼近）**说任何 Hausdorff 距离下连续的集合函数都能被这个形式逼近——max-pooling 不是启发式，而是充分的；**定理 2（临界点集）**说存在 $\mathcal{C}_S \subseteq S \subseteq \mathcal{N}_S$ 使得区间内任意 $T$ 都有 $f(T)=f(S)$，其中 $\mathcal{C}_S$ 是**临界点集**（赢得 max 的那些点），$|\mathcal{C}_S| \le K$。这解释了它对离群点与缺失点的鲁棒性，也说明瓶颈维 $K$（=1024）**限制了可表达的形状复杂度**。

![T-Net](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointnet-3.png)![PointNet 结果](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointnet-4.png)

*图 7：T-Net（输入/特征变换）与实验结果。T-Net 是一个 mini-PointNet，回归一个仿射矩阵作用于输入。来源：论文原图。*

**T-Net** 是一个 mini-PointNet，回归仿射矩阵；特征侧的 $64\times64$ 变换需要正则项才稳：$L_{reg} = \|I - AA^{T}\|_F^2$，权重 0.001。消融：基线 87.1 → 加输入 T-Net 87.9 → 两个 T-Net 加正则 **89.2**（ModelNet40）。

**根本局限是完全没有局部结构**：每个点独立嵌入，然后一次全局 max-pooling。无层级、无局部坐标系、无密度自适应。

**PointNet++**（NeurIPS 2017）补上层级。**Set Abstraction = 采样 → 分组 → PointNet**：

![PointNet++ 层级结构](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointnetpp-1.png)

*图 8：PointNet++ 的层级式 set abstraction。每层用 FPS 选中心点、ball query 分组、mini-PointNet 提特征，逐层扩大感受野；分割分支用插值上采样并加 skip 连接。来源：论文原图。*

1. **采样**：**最远点采样（FPS）**选 $n'$ 个中心点，固定预算下覆盖优于随机；
2. **分组**：**ball query**（半径 $r$，上限 $K$ 个邻居）。选它而不选 kNN 的理由很关键——**ball query 保证固定的尺度**，使学到的局部特征在密度变化下可比，而 kNN 的区域大小会随密度浮动；
3. **PointNet 层**：坐标先局部化（$x_i \leftarrow x_i - \hat{x}$）再过 mini-PointNet + max-pool。

密度处理才是它相对 PointNet 的真贡献：**MSG**（同一层多个半径并行再拼接）、**MRG**（把上一层更细尺度的特征与当前层直接算的特征拼接，比 MSG 便宜）、以及**随机输入 dropout**（每 batch 丢弃比例 $\theta \sim U[0,p]$，$p=0.95$）。论文的招牌图是：密度扰动下 SSG 崩溃而 MSG+DP 基本持平——**让它鲁棒的是 DP 这个训练技巧，不只是架构**。

ModelNet40 **90.7%**（SSG）/ **91.9%**（MSG+DP + 法向 + 5000 点）。局限：FPS 是 $O(n\cdot n')$ 串行、对 GPU 不友好；ball query 的半径要按数据集调；感受野仍只靠局部聚合，无长程推理。

**PointMLP**（ICLR 2022）的论点很反潮流：**2018–2021 那些精巧的局部几何提取器（卷积式、图式、注意力式）贡献远小于人们以为的**，一个深残差 MLP 加一个归一化技巧就能追平甚至超过它们，且快得多。

![PointMLP 残差块](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointmlp-1.png)

*图 9：PointMLP 的残差点块。$\Phi_{pre}$ 对每个邻居做共享残差 MLP，$\mathcal{A}$ 是 max-pooling 聚合，$\Phi_{pos}$ 对聚合结果再做残差 MLP——全程没有任何学习式的局部几何算子。来源：论文原图。*

$$g_{i} = \Phi_{pos}\!\left(\mathcal{A}\big(\Phi_{pre}(f_{i,j})\ \big|\ j=1..K\big)\right)$$

唯一不可省的部件是**几何仿射模块（Geometric Affine Module）**：

$$\{f_{i,j}\} = \alpha \odot \frac{f_{i,j} - f_i}{\sigma + \epsilon} + \beta$$

关键在 $\sigma$ 是一个**标量**，在**所有组、所有邻居、所有通道**上联合统计：$\sigma = \sqrt{\frac{1}{k n d}\sum_{i,j}(f_{i,j}-f_i)^2}$。它把差异极大的局部邻域尺度/稀疏度归一到同一分布上。**它是个归一化，不是几何提取器**——这正是全文论点所在。

![PointMLP 邻域](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointmlp-2.png)

*图 10：局部邻域的密度不均匀问题。PointMLP 用 kNN（$K=24$）选邻居，并靠几何仿射模块把尺度差异归一化。注意这里是 kNN 而非 PointNet++ 的 ball query。来源：论文原图。*

**$K=24$ 已核实**（论文与官方配置一致），但要注意是 **kNN，不是 ball query**。ModelNet40 **94.5%**、ScanObjectNN PB_T50_RS **85.4%**，推理约 **112 ms** vs CurveNet 约 230 ms。局限：仍需 kNN（它声称在算子层面省掉的成本，在采样层面又回来了）；ModelNet40 已饱和，那里的增益在投票噪声范围内；不能像稀疏卷积/Transformer 那样扩展到场景级分割。

### 4.2 基于 Transformer 与自监督：Point-BERT → Point-MAE → PointGPT

**Point-BERT**（CVPR 2022，[arXiv 2111.14819](https://arxiv.org/abs/2111.14819)）把 BERT 的掩码建模搬到点云。

![Point-BERT 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointbert-1.png)

*图 11：Point-BERT 的流程。点云经 FPS+kNN 切成局部 patch，mini-PointNet 得到 patch 嵌入；dVAE tokenizer 把连续特征量化成离散 token ID；掩码后的 patch 由 Transformer 预测其 token ID。来源：论文原图。*

流程：给定 $p \in \mathbb{R}^{N\times3}$，先 **FPS** 采 $g$ 个中心点，**kNN** 为每个中心选 $n$ 个近邻构成局部 patch $\{p_i\}_{i=1}^g$，**减去中心坐标**使 patch 无偏（这样才能类比 NLP/ViT 的 patch），再用 **mini-PointNet** 得到**点嵌入序列** $\{f_i\}_{i=1}^g$；中心点 $\{c_i\}$ 另经 MLP 得位置嵌入 $\{pos_i\}$。ShapeNet 预训练用 1024 点 → **g=64、k=32**。

**Tokenizer** 把每个 patch 的连续特征**量化**成「点词表」里的离散 token ID：$\{f_i\}_{i=1}^g \to \{z_i\}_{i=1}^g$。具体是先单独训一个 **dVAE**（DGCNN 编码器 + **8192 词表** + Gumbel-softmax 松弛 + DGCNN 解码器，Chamfer 距离重建损失）。

> ⚠️ **术语更正**：预训练任务的缩写是 **MPM = Masked Point Modeling（掩码点建模）**，论文原文为「we devise a Masked Point Modeling (MPM) task」。**不是 MLP**（那是多层感知机，完全无关），也不是 MLM（那是 BERT 的 Masked Language Modeling）。

掩码 25%–45%，用**块状掩码**（连续区域，比随机更难），预测被掩 patch 的 dVAE token ID（8192 类交叉熵）。另有辅助的 **Point Patch Mixing + MoCo 式对比损失**注入实例级语义。骨干 12 层、384 维、6 头、约 22.1M 参数。ModelNet40 **93.2%**、ScanObjectNN PB_T50_RS **83.1%**。

局限的核心是**两阶段训练**（dVAE 必须先收敛）以及**离散词表是有损的容量瓶颈**。

**Point-MAE**（ECCV 2022，[arXiv 2203.06604](https://arxiv.org/abs/2203.06604)）正是把这个瓶颈拆掉：**不要 dVAE、不要词表、单阶段**，直接用 Chamfer 距离**回归被掩 patch 的原始坐标**。另外两点差异很关键：**非对称编解码**——mask token 完全不进编码器，只在轻量解码器里引入（Point-BERT 让 mask token 走完整个骨干，既浪费算力又造成训练/测试输入分布不一致）；**掩码率 60% 且随机掩码优于块状掩码**（与 Point-BERT 相反，因为坐标回归本身已经够难）。ScanObjectNN PB_T50_RS **85.18%** 超过 Point-BERT 的 83.1%——用更简单的配方赢，这是「tokenizer 是负担而非资产」的关键证据。

**Point-M2AE**（NeurIPS 2022）补上层级：多阶段金字塔 + 跳连，且**掩码在最粗尺度生成后由粗到细传播**（否则跨尺度会泄漏答案）。ModelNet40 **线性 SVM 92.9%**（冻结特征，很能说明表征质量）。

**PointGPT**（NeurIPS 2023，[arXiv 2305.11487](https://arxiv.org/abs/2305.11487)）改成自回归，要额外解决三个文本没有的问题：**无固有顺序**（用 **Morton / Z-order 空间填充曲线**给 patch 中心排序）、**位置信息泄漏**（把「下一个 patch 的位置嵌入」只给 generator，让模型在已知位置的前提下预测几何）、以及在因果掩码之外再加一层**双重掩码**。PointGPT-L 在 ScanObjectNN PB_T50_RS 上 **93.4%**，是这一代物体级自监督里最强的。局限是序列化顺序**任意且不唯一**（旋转就会改变 Morton 序），且三阶段训练很重。

### 4.3 场景级骨干与 Sonata：可靠表征的转折点

**Point Transformer V3**（CVPR 2024 Oral，[arXiv 2312.10035](https://arxiv.org/abs/2312.10035)）做了一个明确的取舍：**牺牲邻域精确性换规模**。它放弃精确 kNN，改为沿空间填充曲线**序列化**后取**一维序列上的连续 patch** 作为注意力邻域，并在层间轮换/打乱四种序列化模式（Z-order、Trans Z-order、Hilbert、Trans Hilbert）以补回近似带来的损失。感受野从 16 点扩到 **1024 点**，比 PTv2 **快 3 倍、省 10 倍显存**。

**Sonata**（CVPR 2025 Highlight，[arXiv 2503.16429](https://arxiv.org/abs/2503.16429)）——这是你笔记里那个「挂着但没有对应章节」的名字，它其实值得单独一节，因为**它是点云自监督真正变得可用的转折点**，也是 Pts3D-LLM 的编码器。

它的问题意识很锐利：以往 3D 自监督在**全量微调**下看着不错，但在**线性探测**下极差——说明特征本身没有语义，是微调在干活。病根是**几何捷径（geometric shortcut）**：

> 表征坍缩到易得的低层几何线索，例如**法向方向或点的高度**。

**为什么这是 3D 独有的**：图像里信息都在像素值里，掩掉一个 patch 就真的移除了信息；而点云的信息**就在坐标本身**，且任何点算子都必须消费坐标来定义其核——所以几何在结构上**无法被掩掉**。网络永远可以靠读法向或高度来解掉预训练任务。证据是相似度热图：CSC 坍缩到表面法向、MSC 坍缩到点高度，而 Sonata 能把沙发所有扶手归到一起。

对策是两组策略、五个具体设计。**遮蔽空间信息**：(1) **去掉解码器**，只在编码器输出的粗分辨率上做自监督——层级池化本身就摧毁了精细位置信息，这一条是主导项，**线性探测 20.7 → 60.4**；(2) 被掩点加更强的高斯扰动；(3) 在池化后的粗尺度算损失。**增强对输入特征的依赖**：(4) 无参数的多尺度特征上采（hypercolumn 式渐进拼接）；(5) 前 5% 训练做**渐进参数调度**（掩码尺寸 10cm→40cm、掩码率 30%→70%、教师温度 0.04→0.07、权重衰减 0.04→0.2）——先迫使模型依赖特征，再把任务变难。

骨干是 **PTv3 仅编码器、108M 参数**，自蒸馏（DINOv2 血统：EMA 教师、Sinkhorn-Knopp 中心化、KoLeo 正则），数据 **139,769 个点云**（约 14 万，是 PointContrast 的 86.7 倍）。

结果非常硬：

| 设定 | 成绩 |
|---|---|
| ScanNet **线性探测** | **72.5** mIoU（PointContrast 5.6 / CSC 12.6 / MSC 21.8）——**3.3 倍** |
| 消融路径 | 23.0 → 60.4 → 62.5 → 65.5 → **72.5** |
| 1% ScanNet 数据 | 线性 43.6 vs 从零训 25.8 |
| ScanNet 全量微调 | **79.4** mIoU |
| 与 DINOv2 对比 | Sonata 72.5 · DINOv2 63.1 · **两者合并 76.4** |

线性探测头只占参数的 **<0.2%**。最后一行是本文最重要的证据之一：**Sonata 与 DINOv2 是互补而非冗余的**（72.5 + 63.1 → 76.4）——这是「原生 3D 编码器仍有不可替代价值」的最强论据，也正是 Pts3D-LLM 的设计依据。

局限也说得清楚：**只在室内预训练**；在稀疏 SLAM 点云上线性探测 32.0 **反而低于从零训的 34.9**（全量微调能补回）。

这条线在 Sonata 之后还在走：**Concerto**（NeurIPS 2025）加入与冻结 DINOv2 的跨模态分支，ScanNet 线性探测 **77.3**；**Utonia**（ICML 2026）解决 Sonata 的室内限制，用一个编码器覆盖室内/室外/物体/视频提升/遥感。但 Concerto 自己的实验也点出了瓶颈：它接了一个到 CLIP/LSeg 语言空间的线性翻译器，零样本只有 **44.56%** mIoU 对比监督线性探测的 77.3%，作者自称语言对齐仍然「shallow」。

### 4.4 多视图投影：PointCLIP 与 V2

**PointCLIP**（CVPR 2022，[arXiv 2112.02413](https://arxiv.org/abs/2112.02413)）的思路是把点云投影成深度图，直接借用冻结的 CLIP 做零样本识别——**需要两个编码器**。

![PointCLIP 流程](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclip-1.png)

*图 12：PointCLIP 整体流程。点云投影为多视角深度图经 CLIP 视觉编码器，类别名填入模板经文本编码器，两者做相似度匹配。来源：论文原图。*

1. **点云侧**：投影到 $M$ 个视角的深度图。投影是**在线计算、无需渲染**（沿用 SimpleView）：$(x,y,z) \to (\lceil x/z \rceil, \lceil y/z \rceil)$，像素值**直接等于 $z$ 并原样复制到 3 个通道**。
2. **文本侧**：类别名填入模板的 class token 位置。

![PointCLIP 模板](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclip-2.png)

*图 13：文本模板与多视角特征的通道维拼接 $\operatorname{Concat}(f_{1\sim M}) \in \mathbb{R}^{1\times MC}$。来源：论文原图。*

> **模板已逐字核实**：零样本用 `"point cloud depth map of a [CLASS]."`（你笔记里的写法与标点都对）。但**少样本换成了** `"point cloud of a big [CLASS]."`，因为在那里更好（16-shot 87.20 vs 85.82）。
>
> **视角数 6 与 10 都对，但属于不同设定**：**零样本用 6 个**、**少样本用 10 个**。这个交叉点本身是个有意思的发现——零样本下 4→18.68、**6→20.18**、8→16.98、10→14.91、12→13.65，**视角多于 6 反而变差**（论文归因于冗余）；而 16-shot 下 6→84.24、**10→87.20**、12→86.35。

![PointCLIP 适配器](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclip-3.png)![PointCLIP 残差](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclip-4.png)

*图 14：inter-view adapter。拼接后的多视角特征经三层 MLP 得到全局特征，再生成逐视角的自适应特征并残差连接回原输入，得到最终 Adapted Features。消融显示全局聚合是承重部件（去掉它 87.20→83.87）。来源：论文原图。*

$f_{global} = \text{ReLU}(\text{Concat}(f_{1..M})W_1^T)W_2^T$，逐视角残差 $f_i^a = f_i + \text{ReLU}(f_{global}W_{3i}^T)$。只训适配器，两个 CLIP 编码器全冻结。

> **数字要点**：ModelNet40 零样本 **20.18%（RN50）**、**23.78%（RN50×16，最佳骨干）**——两个都是真的，PointCLIP V2 对比的是 23.78。集成 CurveNet 后 **94.08%**（不是 94.2）。一个很好的**反面对照**：集成两个已充分训练的 3D 网络反而**变差**（PointNet+++RSCNN 92.14 < 92.22），说明增益来自 CLIP 的**互补 2D 知识**，而非集成本身。
>
> 隐含局限值得点出：投影距离、分辨率、视角权重、模板、骨干**每个数据集都不一样**，这实质上削弱了「零样本」的说法。

**PointCLIP V2**（ICCV 2023，[arXiv 2211.11682](https://arxiv.org/abs/2211.11682)）明确指出 V1 的两个缺点，且论文给了小节标题名——都归因于 **2D-3D 域差距**：

![V2 缺点一](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclipv2-1.png)![V2 缺点二](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclipv2-2.png)

*图 15：V1 的两个缺点。左：**Sparse Projection**——投影出的是稀疏散点，与 CLIP 预训练所见的真实图像在外观与语义上都差别巨大。右：**Naive Text**——沿用 2D 文本输入只追加简单 3D 词汇，相似度图显示文本特征「几乎无法聚焦到目标物体」。来源：论文原图。*

**改进一：真实感投影，四步——量化 → 稠密化 → 平滑 → 压缩。**

![V2 投影](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclipv2-3.png)![V2 投影结果](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclipv2-4.png)

*图 16：真实感投影。把点云转为体素后经**局部最小值池化**稠密化、**非参数高斯核**平滑，最后沿深度维取最小值压成图——渲染图不再是散点图，而是渐变的深度图。来源：论文原图。*

1. **量化**：零初始化网格 $G$，坐标归一化到 [0,1]，$G(\lceil sHx\rceil, \lceil sWy\rceil, \lceil Dz\rceil) = z$，同体素多点**取最小深度**（近处遮挡远处）；
2. **稠密化**：**局部最小值池化（local mini-value pooling）**，填补稀疏点之间的空体素而背景保持为空。最小值优于均值与最大值，因为它「符合被遮挡的视觉外观」；
3. **平滑**：**非参数高斯核**，去掉池化伪影同时「保留边缘与角的锐利度」；
4. **压缩**：沿深度维取最小值作为像素值，**复制 3 次**成 RGB。

投影消融（ModelNet40 零样本）很有说服力：仅量化+压缩 44.50 → 加最小值稠密化 59.64 → 加平滑 **64.22**（稠密化 +15.14、平滑 +5.7）。

> ⚠️ 「shape-invariant（形状不变）」这个说法在原文中**查不到**，不要这样描述；论文声称的性质是**真实感/CLIP 偏好、非参数、时间高效**。

**改进二：LLM 生成的 3D 语义 prompt。**

![V2 GPT prompt](assets/三维多模态大语言模型架构%20-%20Survey/figures/pointclipv2-5.png)

*图 17：用 GPT-3 生成富含 3D 语义的文本，使图文有更好的相似度。来源：论文原图。*

用 **GPT-3（`text-davinci-002`**，温度 0.7）生成，**四类**命令（不是三类）：**Caption Generation、Question Answering、Paraphrase Generation、Words to Sentence**。规模是 **50 条命令（13+13+12+12）× 20 条描述 ⇒ 每类约 1000 条描述**。消融：无 GPT 文本 39.11 → 仅 caption 61.67 → 四类齐全 **64.22**（**+25.11**）。

零样本结果：

| | MN10 | MN40 | S-OBJ_ONLY | S-OBJ_BG | S-PB_T50_RS |
|---|---|---|---|---|---|
| PointCLIP | 30.23 | 23.78 | 21.34 | 19.28 | 15.38 |
| **V2** | **73.13** | **64.22** | **50.09** | **41.22** | **35.36** |
| Δ | +42.90 | **+40.44** | +28.75 | +21.94 | +19.98 |

> ⚠️ 摘要里那个 **+42.90% 是 ModelNet10**，ModelNet40 是 **+40.44%**。一个有价值的归因实验：把 V1 的投影放进 V2 的 GPT prompt 流程里能到 **42.53**（单独用是 23.78），所以 GPT prompt 约值 +19 分、真实感投影约值 +22 分。

### 4.5 跨模态对齐预训练：ULIP 与 ULIP-2

> ⚠️ **分类更正**：你原来的表把 **ULIP** 和 PointLLM、3D-LLaVA 并列在「关于点云的 3DLLM 总体设计」下，这是**放错了类别**。ULIP 是**语言/图像/点云三模态对齐的预训练方法，全程不涉及 LLM**。你自己在正文里写的是「学习语言、图像和点云的统一表示」，而 PointLLM 那节更是明确写了它**用 ULIP-2 去训练自己的 Point-BERT**——也就是说 ULIP 是**编码器侧的预训练配方**，应归到本节。

**ULIP**（CVPR 2023，[arXiv 2212.05171](https://arxiv.org/abs/2212.05171)）的机制是：用（图像、文本、点云）三元组，把 3D 编码器**对齐进一个已经对齐好的图文空间**。

![ULIP 框架](assets/三维多模态大语言模型架构%20-%20Survey/figures/ulip-1.png)

*图 18：ULIP 的三模态对齐。图文编码器来自预训练模型并**冻结**，只更新点云编码器 $f_P$，用逐对的对称 InfoNCE 把点云拉进图文共享空间。来源：论文原图。*

已核实的细节：

- 数据用 **ShapeNet55** 构建三元组 $T_i:(I_i,S_i,P_i)$，**约 52.5K** 个 CAD 模型、55 类，每个关联有文本描述的元数据；
- 从原始点云均匀采样 $N_p$ 个点，编码器试了 **PointNet++（ssg 与 msg）、PointMLP、Point-BERT**，附录还有 PointNeXt——所以「三个编码器」需要限定说法；
- 单个资产用虚拟相机渲染多视角图，**30 个 RGB + 30 个深度图**（每 12 度一张），每次迭代随机取其一；
- **64 种文字模板**经编码器后取平均 $h^S_i = \text{Avg}(f_S(S_i))$（63 个标准 prompt + 1 个专用的 "a point cloud model of [WORD]"）。

![ULIP 消融](assets/三维多模态大语言模型架构%20-%20Survey/figures/ulip-2.png)

*图 19：模态消融——三模态对齐优于两模态。Point-BERT 上 P+I+T 60.4 > P+T 44.7 > P+I 35.5，三个模态都是承重的。来源：论文原图。*

> ⚠️ 两处需要修正：**图文塔用的是 SLIP ViT-B 而非原味 CLIP**（论文原话「we utilize an advanced version of CLIP, namely SLIP」），写「CLIP/SLIP」可以，只写「CLIP」不准确。另外损失 $L_{final} = \alpha L_{(I,S)} + \beta L_{(I,P)} + \theta L_{(P,S)}$ 里 **$\alpha = 0$ 是默认值**——图文项写在式子里但被关掉了，因为在有限 3D 数据上更新图文编码器会灾难性遗忘。所以实际只优化 P↔I 与 P↔T。

零样本 ModelNet40：PointCLIP 20.2 → PointNet++(ssg) 55.7 → PointNet++(msg) 58.4 → Point-BERT 60.4 → **PointMLP 61.5**。微调提升：Point-BERT ScanObjectNN 83.1 → **86.4（+3.3）**、PointMLP 85.7 → **88.8（+3.1）**。

**ULIP-2**（CVPR 2024，[arXiv 2305.08275](https://arxiv.org/abs/2305.08275)）的改动是把「需要人工元数据」这个瓶颈拆掉：用 **BLIP-2-opt6.7B** 给渲染图**自动生成描述**（每图生成 10 条，用 CLIP-ViT-Large 按图文相似度重排，**只留 top-1**——k=1 的 69.7 明显优于 k=3 的 66.7 与 k=10 的 66.3，激进过滤是必需的），于是**只需要 3D 数据本身**，从而能吃下 Objaverse。图文塔升级为**冻结的 OpenCLIP ViT-G/14**，损失简化为恰好两项 $\min_{E_P} L_{P2I} + L_{P2T}$。

发布的数据集：ULIP-Objaverse 约 **80 万**点云 / 约 1000 万图像 / 约 1 亿语言；ULIP-ShapeNet 约 5.25 万 / 300 万 / 3000 万。零样本（Point-BERT）：ShapeNet 预训练 O-LVIS 16.4、MN40 **75.2**；Objaverse+ShapeNet 预训练 O-LVIS **50.6**、MN40 **84.7**。

**这就是 PointLLM 的编码器来源**，PointLLM 原文逐字写道：「Point-BERT, pre-trained with ULIP-2 on the Objaverse dataset, serves as our point encoder.」

ULIP-2 自陈的局限，恰好是本文 §7 的关键前提：「ULIP-2's pre-training primarily utilizes **object-level** 3D shape datasets, which inherently differ from **scene-level** 3D data in their distribution and complexity.」

### 4.6 小结：给 LLM 喂什么编码器

| 编码器 | 输入处理 | 监督信号 | 给到的下游性质 | 适用范围 |
|---|---|---|---|---|
| PointNet | 原始点、全局 max-pool | 有监督 | 弱全局语义 | 物体 |
| PointNet++ | 原始点、层级局部 | 有监督 | 局部+全局几何 | 物体 / 小场景 |
| PointMLP | 原始点、kNN + 残差 MLP | 有监督 | 几何，快 | 物体 |
| PTv3 | 序列化 patch token | 有监督 | 可扩展的场景语义 | 场景，室内外 |
| Point-BERT | FPS+kNN patch token | 掩码自监督（离散 dVAE）+ 对比 | 可迁移的物体特征 | 物体 |
| Point-MAE | patch token | 掩码自监督（坐标回归） | 几何，配方更简单 | 物体 |
| PointGPT | Morton 序 token 序列 | 自回归自监督 | 最强物体级精度 | 物体 |
| **Sonata** | 序列化 token（PTv3 仅编码器） | **自蒸馏（DINOv2 式）** | **线性可分的可靠语义 + 几何** | 场景（室内） |
| PointCLIP / V2 | **投影深度图** | 无（冻结 CLIP） | **语言对齐**、零样本；几何弱 | 物体 |
| ULIP / ULIP-2 | 任意 3D 编码器 | **跨模态对比** | **语言对齐**（ULIP-2 达到规模） | 物体 |
| ReCon++ | patch token、双分支 | 掩码自监督 **+** 对比 | 语言对齐 **+ 几何** | 物体 |

三条正交的轴，可以归结成一句原则：

- **投影式**编码器免费继承语言对齐，但丢掉 3D 结构（V1 那 20% 的零样本就是代价），且上限被光栅化能保留的信息卡住；
- **掩码/自蒸馏自监督**给出最好的几何与 3D 内部语义，但特征空间与语言**毫无关系**；
- **跨模态对比**是唯一产出「文本可直接寻址」特征的路线，但代价是：对齐到一个冻结的 CLIP 空间意味着丢掉 CLIP 表达不了的几何，而且目标是实例级的，精细空间细节不被保留。

**给 LLM 用该选哪个？** 物体级的答案是**跨模态对比，即 ULIP-2 预训练的 Point-BERT**——理由是机械的而非审美的：LLM 的连接器只是薄薄几层 MLP（PointLLM 是 3 层 GeLU，384→4096），在小规模指令数据上训练，它**没有能力凭空发明**一个从任意几何特征空间到 token 嵌入空间的映射。若编码器特征已在 CLIP 对齐空间里，连接器只需做两个**语义组织方式相同**的空间之间的基变换；否则它得从指令数据里从零学出语义，容量与数据都不够。ULIP-2 的 O-LVIS 零样本 2.6 → 50.6 与它作为 LLM 前端的可用性是同步的。

**但纯对比对齐可测量地不够。** 把 PointLLM 的 ULIP-2 编码器换成 **ReCon++**（对比 **+** 生成式掩码建模），3D MM-Vet **+4.20%**。纯对比优化实例级判别，于是保住了「这是什么物体」而丢掉了「它的部件如何排布」——恰是 LLM 做定位与部件级推理需要的空间细节。**正确选择是双目标：生成式/掩码管几何 + 对比管语言可寻址性。**

**场景级答案不同，这也是 Sonata 的位置。** ULIP-2 自己声明了不适用（物体级数据与场景级「本质不同」）。场景级没有 ULIP-2，只有两个选项：一是**提升 2D 特征**（3D-LLM、LLaVA-3D，见 §2），语言对齐天然具备但几何只能透过 2D 推断；二是**把几何自监督编码器当作补充而非替代**——证据很干净：Sonata 72.5 vs DINOv2 63.1 vs **两者合并 76.4**。Pts3D-LLM 正是这么做的：Sonata/PTv3 特征线性投影后按最近邻汇到图像 patch 位置，然后**相加**到图像与位置 token 上。

诚实的结论：**不要单独用 Sonata 当唯一前端。** Concerto 那个到 CLIP/LSeg 的线性翻译器只有 44.56% 零样本 mIoU 对比 77.3% 监督线性探测，作者自称语言对齐「shallow」。几何自监督给你 2D 路线给不了的空间推理，但它不给你语言可寻址性——**要并联使用**。

---

## 5. 路线 C：Mesh 如何进入 LLM

> 本节聚焦「mesh 怎么被 LLM 消费」。Mesh **生成**本身的表征之争（VecSet 系、稀疏体素系、MeshGPT→FACE→LATO 的显式几何 token 化）已在 [[三维网格生成表征演进 - Survey]] 详述，此处不重复，只在需要对照时引用。

### 5.1 根本约束：面数决定一切

网格的 token 成本是 **$O(F)$ 且常数很大**。朴素序列化是 3 顶点 × 3 坐标 = **9 token/面**，且**没有顶点共享**——每个内部顶点按其价数被重复发射约 6 次。这个冗余就是整条 tokenizer 文献的靶子。更糟的是自回归解码本身是串行的，token 数同时决定上下文需求**与**墙钟延迟，而误差累积意味着长序列**成功率更低**（EdgeRunner 明确报告了这一点）。

**具体算一个 5000 面网格**（5000 面大致是简单游戏资产；游戏角色是 1 万–10 万面）：

| 方案 | tok/面 | 5000 面 | 能装进 8k 上下文吗 |
|---|---|---|---|
| LLaMA-Mesh（OBJ 文本） | ~7.5（推算） | **~37,500** | 否（超约 4.6 倍） |
| MeshXL / 朴素 9n | 9.0 | 45,000 | 否 |
| MeshGPT / PivotMesh（RQ-VAE） | 6.0 | 30,000 | 否 |
| MeshAnything V2（AMT） | ~4.5 | 22,500 | 否 |
| EdgeRunner（EdgeBreaker） | 4–5 | ~22,500 | 否 |
| BPT | ~2.34 | 11,700 | 否 |
| TreeMeshGPT / Mesh-Silksong | ~2.0 | 10,000 | 否 |
| **MeshWeaver（CVPR 2026 SOTA）** | **~1.62** | **8,100** | 勉强 |
| **MeshCoder（程序）** | **与面数无关** | ~10²–10³ | 轻松 |

锚点：MeshXL 里一个 800 面网格 = **7,200 token**；LLaMA-Mesh 的 500 面上限约占其 8k 预算的 **3.7k**。**一句话的重点：在 LLaMA-Mesh 的方案里塞一个 5000 面网格，比塞一整章小说还贵——而 5000 面还只是低模。**

两年间的压缩轨迹是 9 → 6 → 4.5 → 2.34 → 2.0 → **1.62** tok/面（约 5.5×），最大面数 800 → 1,600 → 4,000 → 8,000 → **16,000**。三个已被依次挖掘的思路：(1) 相邻面共享顶点（AMT、EdgeRunner）；(2) 分层/分块坐标索引（BPT、DeepMesh、MeshWeaver）；(3) 重排**序列顺序**以贴合网格拓扑（TreeMeshGPT 的树、Silksong 的层、Nautilus 的局部性）。

### 5.2 Mesh 3DLLM：把网格当文本

**LLaMA-Mesh**（NVIDIA + 清华，[arXiv 2411.09595](https://arxiv.org/abs/2411.09595)）通过把网格数据转为纯文本表示，暴力地让 LLM 直接处理和生成 3D mesh——不扩词表、不加模态编码器、不加适配器。

![LLaMA-Mesh 概念](assets/三维多模态大语言模型架构%20-%20Survey/figures/llamamesh-1.png)![LLaMA-Mesh 流程](assets/三维多模态大语言模型架构%20-%20Survey/figures/llamamesh-2.png)

*图 20：LLaMA-Mesh 把 mesh 表示为 OBJ 纯文本，与语言共处同一 token 流。顶点坐标缩放到 $[0,64]$ 并取整（**64 格/轴、6 bit**），顶点按 z-y-x 升序、面按最小顶点索引排序（PolyGen 式规范化）。来源：论文原图。*

![LLaMA-Mesh 结果](assets/三维多模态大语言模型架构%20-%20Survey/figures/llamamesh-3.png)![LLaMA-Mesh 对话](assets/三维多模态大语言模型架构%20-%20Survey/figures/llamamesh-4.png)

*图 21：生成结果与对话示例。因为几何与语言在同一序列里，文生网格、网格描述与多轮对话式细化都是天然支持的。来源：论文原图。*

量化的动机是直接的：「Using floating-point numbers for vertex coordinates directly leads to long token sequences, exceeding most LLM's context length limitations.」于是**零位小数、只用整数**。基座是 LLaMA-3.1-8B-Instruct，上下文 **8k**，**最大 500 面**；数据是 Objaverse 里筛出的 31k 个 ≤500 面网格，旋转增广到约 125k 样本，配比 40% 生成 / 20% 理解 / 40% 通用对话（UltraChat）。

**语言能力的对齐税是可量化的**：

| 基准 | LLaMA-3.1-8B-Instruct | LLaMA-Mesh |
|---|---|---|
| MMLU (5-shot) | 66.07 | 61.74 |
| PIQA (0-shot) | 81.01 | 79.16 |
| HellaSwag (0-shot) | 79.19 | 77.35 |
| **GSM8K (8-shot)** | **77.18** | **62.09** |

数学/推理掉得最狠，论文归因于文本回放只用了 UltraChat。**「复用 LLM 权重」不是免费的**。

**MeshLLM**（[arXiv 2508.01242](https://arxiv.org/abs/2508.01242)）与 LLaMA-Mesh 一样把 Mesh 当作纯文本序列（**序列化方案完全相同**——OBJ 文本、[0,64] 整数量化、同样的排序），它的贡献不在 tokenizer，而在**数据管线**：

![MeshLLM 部件分割](assets/三维多模态大语言模型架构%20-%20Survey/figures/meshllm-1.png)

*图 22：MeshLLM 利用 3D 部件级语义分割（SAMPart3D）把整体网格拆成 Primitive-Mesh，从而把训练样本量放大约 50 倍。来源：论文原图。*

![MeshLLM 训练](assets/三维多模态大语言模型架构%20-%20Survey/figures/meshllm-2.png)

*图 23：渐进式训练与两个前置任务。(1) 先用 KNN 聚类的 Primitive-Mesh 预训练，(2) 再用语义分割后的 Primitive-Mesh，(3) 最后在具体任务上微调；前置任务是「从顶点预测面」与「从部件组装完整网格」。来源：论文原图。*

两条分解路径：**KNN 式 Primitive-Mesh**（最远点采样 + KNN 聚类，约 0.2 s/网格，产出 **150 万+** 样本）与**语义式 Primitive-Mesh**（**SAMPart3D** 零样本 3D 部件分割 + 文本标注，10 万+ 样本）。训练分三阶段（KNN 预训练 → 语义 → 任务微调），呼应 LLM 的「大规模粗数据后接高质量数据」配方。

两个前置任务的存在本身就是证据：**顶点→面预测**（给顶点推面的连接关系 = 拓扑推理）与**网格组装**（从部件重建整体），它们存在的唯一目的就是**把一维序列化摧毁掉的结构重新注入**。数据量约 **50 倍于** LLaMA-Mesh 的 31k，面数上限 800。结果：Chair COV **47.33** vs LLaMA-Mesh 19.53；理解侧 CIDEr **1.753** vs 0.397。

**MeshLLM 自陈的局限最有指向性**：它建议改用 **MeshAnything-V2 式的紧凑序列化**加更长上下文——等于**承认纯文本 OBJ 在 token 效率上不可持续**。

**MeshCoder**（NeurIPS 2025，[arXiv 2508.14879](https://arxiv.org/abs/2508.14879)）走的是第三条路：点云 → **可执行的 Blender Python 脚本**，代码按语义部件分解。**网格从未被 token 化，被 token 化的是构造它的程序。**

![MeshCoder 概览](assets/三维多模态大语言模型架构%20-%20Survey/figures/meshcoder-1.png)

*图 24：MeshCoder 总览。点云经三平面 tokenizer 编码为 128 个条件 token，LLM（LLaMA-3.2-1B + LoRA）输出结构化的 Blender Python 代码，执行后得到带语义部件命名的网格。来源：论文原图。*

![MeshCoder DSL](assets/三维多模态大语言模型架构%20-%20Survey/figures/meshcoder-2.png)![MeshCoder 结果](assets/三维多模态大语言模型架构%20-%20Survey/figures/meshcoder-3.png)

*图 25：DSL 算子族与重建/编辑结果。五类算子——基本体、扫掠（2D 截面沿 3D 轨迹）、bridge loop（放样）、布尔、阵列——覆盖了人造物体的主要构造方式。来源：论文原图。*

DSL 是专门设计的 Blender Python API，五类算子：**基本体**（立方体/圆柱/球/锥/环）、**扫掠**（5 种截面 × 6 种轨迹，覆盖旋转体）、**Bridge Loop**（在两个截面间放样，处理扫掠表达不了的形状）、**布尔**（并/交/差，做复合体与孔洞）、**阵列**（1D 曲线与 2D 平面重复，廉价表达周期结构）。

数据是**先合成程序、再学逆映射**的两级构造：部件级约 **1000 万** 点云↔代码对（按概率采样 API 参数），物体级用改造的 **Infinigen Indoor** 程序化生成器产出 **100 万** 物体-代码对、41 类。CD **0.063e-2** vs PLAD 1.87e-2；体素 IoU **86.75%** vs 67.62%。

**代码作为表征买到了什么**：**可编辑性**（改一个参数得到语义上合法的编辑，如方桌面→圆桌面，而不是挪顶点）；**拓扑与分辨率解耦**（一个分辨率参数独立于形状调节网格密度——本节其他方法都做不到）；**长度压缩随规整度而非面数变化**（一个 `Array` 调用就编码了 N 个重复部件）；**语义免费获得**（部件有名字，论文展示 GPT-4 直接从推断出的代码回答结构问题）。局限是**只处理人造物体**——有机形态（动物、人）无法用扫掠/放样/布尔表达，且代码里没有颜色与材质。

### 5.3 Mesh Encoder：学习式 tokenizer 与效率竞赛

**MeshGPT**（CVPR 2024 Highlight，[arXiv 2311.15475](https://arxiv.org/abs/2311.15475)）是学习式 mesh tokenizer 的奠基工作：**基于图网络对 mesh 编码**。

![MeshGPT 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/meshgpt-1.png)

*图 26：MeshGPT 两阶段。阶段一用面邻接图上的 SAGEConv 图卷积 + 残差向量量化（RQ-VAE）学出一套「三角形词表」；阶段二在该词表上训练一个 decoder-only Transformer 自回归生成面序列。来源：论文原图。*

tokenizer 细节：编码器是**面邻接图**上的 **SAGEConv** 图卷积，每面输入特征含 9 个坐标的位置编码 + 面积 + 边角 + 法向——**拓扑通过图结构进入，而非通过序列**。残差 VQ **深度 D=6 每面**（即每顶点 2 个残差码）⇒ **6 token/面**，码本 16,384、嵌入维 192，坐标离散到 $128^3$。解码器是序列上的 1D ResNet-34，输出离散坐标的 logits（分类而非回归），因为它看到整个序列，所以单面嵌入可以局部歧义仍能解码。Transformer 是 GPT2-medium，上下文 4,608 = 768 面 × 6。**最大 800 面**。被低估的部分是增广：**带角度容差 $\alpha\in[1°,60°]$ 的平面简化**，为同一形状生成多种合法的三角化。Chair 上 COV 43.28% vs PolyGen 31.22%、FID **18.46** vs 61.10。缺点是采样要 **30–90 秒**。

**MeshAnything**（ICLR 2025，[arXiv 2406.10163](https://arxiv.org/abs/2406.10163)）**基于自回归方式对 mesh 编码**，核心是重新定义问题：网格提取不是等值面抽取问题，而是**生成**问题。它把「拿到形状」（任何现有 3D 方法）与「拿到美术级拓扑」（本模型）解耦，从而定位成**任何 3D 管线的后处理器**。

![MeshAnything 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/meshanything-1.png)

*图 27：MeshAnything 的 VQ-VAE + 形状条件自回归 Transformer。输入任意 3D 表征 → SDF → Marching Cubes 粗网格 → 采带法向的点云（这一步是**域对齐**：训练时的条件也是 MC 派生的点云）→ 自回归生成 token → 抗噪解码器输出美术级网格。来源：论文原图。*

关键技巧是**抗噪解码器**：解码器用**在码本采样 logits 上加 Gumbel 噪声**的方式微调，以模拟 Transformer 推理时真正会产生的不完美 token 序列——这是对 VQ 经典训练/测试错配的显式修补。码本 8,192、RQ 深度 3；Transformer 是 OPT-350M，条件是 4,096 个带法向的点经预训练点编码器得到的 **257 个 token** 作前缀。COV **53.1**（MeshGPT 41.7）、FID **14.5**（25.1），面数比 MC 输出少「几百倍」。它诚实地指出一个重要警告：**生成式因而不如重建式稳定**——输出不确定性对生产管线是缺点。

**V2 的 AMT（Adjacent Mesh Tokenization）** 是效率竞赛的第一次大跃进：若序列中**最后两个顶点**与某个未访问面构成边，则只发射**第三个顶点**（1 个 token 三元组/面）；否则发 `&` 断开并重新以一个新面的 3 顶点开始，另有 `$` 顶点交换算子延长连续段。**9×F → ~4.5×F，序列比 0.497（正好约一半），最大面数 800 → 1,600**。而且 AMT 既更短又略更准（CD 0.874 vs 0.895），因为短序列更好建模。

**InstantMesh**（[arXiv 2404.07191](https://arxiv.org/abs/2404.07191)）——【图生 3D Mesh】。

![InstantMesh 架构](assets/三维多模态大语言模型架构%20-%20Survey/figures/instantmesh-1.png)

*图 28：InstantMesh 流程。**Zero123++** 多视图扩散生成 6 个固定视角 → 稀疏视图大重建模型（三平面骨干）→ **FlexiCubes** 可微等值面抽取。FlexiCubes 使得深度与法向监督可以直接作用在网格上，而非只在图像空间。来源：论文原图。*

> ⚠️ **分类更正**：InstantMesh 在你原来的表里被列进「MeshEncoder」，但它**不是 mesh 编码器**——它没有 token 序列、没有词表、面数不可控，输出是**稠密/程序化抽取**的网格，三角化均匀、无锐边结构。它恰好是 MeshAnything 存在的意义所指的那类输出（待后处理的对象）。前馈因此确定且快（**<10 秒**），与自回归方法的取舍完全相反。它应单列为「前馈重建路线」做对照。GSO 上 SSIM 0.897、CD 0.180。局限是 $64^3$ 三平面限制细节，且 FlexiCubes 在细小/薄结构上不如 NeRF。

**效率竞赛的当前前沿**（2024→2026 的主要进展都在这里，本文只列结论，细节可参 [[三维网格生成表征演进 - Survey]]）：

| 工作 | 出处 | 机制要点 | 压缩比 | 最大面数 |
|---|---|---|---|---|
| MeshXL | NeurIPS 2024 | NeurCF，无 VQ-VAE，纯量化+排序（**9n 基准线**） | 1.00 | 800 |
| MeshGPT / PivotMesh | CVPR 2024 / ICLR 2025 | RQ-VAE 6 token/面；PivotMesh 先生成 10% 枢轴顶点 | 0.67 | 800 / <500 |
| MeshAnything V2 | — | AMT 相邻面共享顶点 | 0.497 | 1,600 |
| EdgeRunner | ICLR 2025 | EdgeBreaker + 半边遍历，词表仅 518，保证面朝向一致 | 0.47 | ~4,000 |
| BPT | CVPR 2025 | 分块索引（顶点≤2 token）+ patch 聚合（中心顶点出现 6→1 次） | 0.26 | 8,000 |
| Nautilus | ICCV 2025 | 局部性保持 tokenization + 双流点条件器 | 0.27 | 5,000 |
| DeepMesh | ICCV 2025 | 改进分块索引，词表 40,960→4,736；**首个把 DPO 引入网格生成** | ~0.28 | **30,000** |
| TreeMeshGPT | CVPR 2025 | 三角邻接**动态树**，从最后一个面局部延伸 | 0.22 | — |
| Mesh Silksong | ICLR 2026 | 顶点按图距分层 (L,i)，显式**保证流形/水密/法向一致** | 0.22 | ~8,000 |
| **MeshWeaver** | **CVPR 2026** | **顶点级（非坐标级）多层体素索引，粗到细单步解码** | **0.18** | **16,000** |

MeshWeaver 值得点出的一处细节：它的骨干是 **LLaMA3 式 Transformer + RoPE**。也就是说**两个阵营的架构已经一样了，只差词表**——这正是 §7.3 要讲的收敛方向。

### 5.4 两种哲学的取舍

| 维度 | (i) 序列化成文本/代码，复用 LLM 权重 | (ii) 专用 tokenizer/VQ 词表，从零训练 |
|---|---|---|
| 代表 | LLaMA-Mesh、MeshLLM、MeshCoder | MeshGPT、MeshAnything、BPT、EdgeRunner、MeshWeaver |
| **token 效率** | ~7.5 tok/面（实际比朴素 9n 更差，因还要付 `v`/`f`/换行/多位索引） | **1.62–6 tok/面，赢 4–5 倍** |
| **最大面数** | 500 / 800 | **8,000–30,000（差 16–60 倍）** |
| 几何精度 | 64 格/轴（6 bit） | $128^3$ 至 $512^3$（**每轴细 8 倍**） |
| **语言先验** | **免费获得**：文生网格、网格描述、多轮对话式细化 | **完全没有**，需外挂编码器；无对话、无指令跟随 |
| 拓扑保证 | 无。1D 展平摧毁邻接信息（MeshLLM 的两个前置任务就是为补这个洞） | 编码在表征里：面图卷积 / 半边 / 邻接树 / 顶点层；Silksong 甚至**保证**流形与水密 |
| 数据需求 | 低（31k 网格即可工作，先验替代数据） | 极高（MeshXL 250 万网格 / 1500 亿 token / 23k A100 时） |
| 训练稳定性 | 单阶段 | 两阶段脆弱（AR 发出解码器没见过的序列，故需 Gumbel 抗噪等补丁） |
| 副作用 | **语言退化**（GSM8K 77.2→62.1） | 慢、随机、长序列成功率下降；词表复杂度膨胀 |
| 可编辑性 | **MeshCoder 独有**：改参数即语义合法编辑 + 分辨率解耦 | 无。VQ token 序列人不可读改 |

### 5.5 Mesh 3DLLM 范式小结

| 工作 | 范式 | 关键代价 |
|---|---|---|
| **LLaMA-Mesh** | 将 Mesh 解析为**纯文本**（OBJ）嵌入 LLM | 500 面上限、64 量化格、GSM8K −15.1 |
| **MeshLLM** | 同为纯文本，但用部件级分解把数据放大 50 倍 + 渐进训练 + 拓扑前置任务 | 800 面上限；自陈应转向紧凑序列化 |
| **MeshCoder** | 将 mesh 结构为代码类的**格式化语言**（Blender Python）嵌入 LLM | 仅人造物体、无材质、依赖程序化生成器造数据 |

---

## 6. 一个不舒服的问题：这些分数测到了 3D 吗

前面所有架构结论都建立在 ScanQA / SQA3D / 3D MM-Vet 这类基准上。2025–26 有两条独立的工作说明：**这些基准很大程度上测不出 3D 能力**。这一节必须放在架构讨论之后、结论之前，因为它决定了前文证据的强度。

### 6.1 Real-3DQA：滤掉语言先验后，分数崩塌

**Do 3D LLMs Really Understand 3D Spatial Relationships?**（ICLR 2026，[arXiv 2603.23523](https://arxiv.org/abs/2603.23523)）把 SQA3D 里「不看 3D、靠语言先验就能答」的题滤掉，构造 **Real-3DQA**：

| 模型 | SQA3D | Real-3DQA | 掉幅 |
|---|---|---|---|
| 3D-LLM | 47.8 | **7.5** | −40.3 |
| Chat-3D v2 | 45.0 | **3.4** | −41.6 |
| LEO | 49.4 | **14.3** | −35.1 |
| Chat-Scene | 54.4 | **17.0** | −37.4 |
| GPT4Scene | 60.6 | **33.1** | −27.5 |

纯文本盲基线在 SQA3D 上就能追平甚至超过 3D-LLM。更刺眼的是**视点旋转一致性**（0/90/180/270° 全对的比例）：3D-LLM **0.1%**、Chat-Scene **0.1%**——**几乎没有任何模型是视点不变的**。而且他们的修正方案虽提升 Real-3DQA，却**降低**原 SQA3D（Chat-Scene 57.2→48.9）：**基准分数与真实 3D 能力是反相关的**。

### 6.2 2D-Cheating：渲染出来喂 GPT-4o 就能赢

**Revisiting 3D LLM Benchmarks**（ACL 2025 Findings，[arXiv 2502.08503](https://arxiv.org/abs/2502.08503)）把点云渲染成图喂 GPT-4o：

- **物体级基准基本是坏的**：3D MM-Vet **58.1（2D）vs 43.2（3D SOTA），+14.9**。他们测出的「真正需要 3D」的题占比：3D MM-Vet **16.0%**、ObjaverseXL-LVIS **7.3%**。
- **场景级基准站得住**：ScanQA METEOR 12.8 vs 20.0、SQA3D EM 42.2 vs 52.6（需 3D 比例 62.9% / 81.9%）。
- **但给到最优视角就翻转**：ScanQA METEOR **28.2（best-of-N）vs 20.0**——**选对视角，纯 2D 超过 3D SOTA**。这是视角选择敏感性的直接证据。
- 指标本身也有问题：与人类判断的 Spearman 相关，LLM-eval 77.7–80.2% 而 **CIDEr 仅 25.5%**。

### 6.3 两个反复出现的规律

**规律一：LLM 规模几乎不重要。** Grounded 3D-LLM 说超过 1B「收益极小」；PointLLM-13B 幻觉**比 7B 更多**；LL3DA 冻结的 OPT-1.3B 就打过专用模型。三条都指向同一结论：**瓶颈在数据与感知，不在语言容量**——这与每篇论文的局限一节都首先提「3D-语言数据稀缺」是一致的。

**规律二：阶段数在下降，且不是偶然。** Chat-3D 3 阶段 → Chat-3D v2 3 阶段 → Chat-Scene **1 阶段**；Pts3D-LLM 与 LL3DA 也是单阶段。当每个感知模块都是冻结的基础模型时，分阶段对齐就没有必要了。反向趋势也成立：**训练**感知模块的方法（3D-LLaVA 512 epoch、Grounded 3D-LLM 600 epoch）仍需两阶段。

### 6.4 该看哪些基准

| 基准 | 出处 | 测什么 | 指标 | 领先（路线） |
|---|---|---|---|---|
| ScanQA | CVPR'22 | 3D QA（ScanNet） | EM@1 / CIDEr | LLaVA-3D 103.1 CIDEr（2D 提升） |
| SQA3D | ICLR'23 | **情境化** QA（具身视点） | EM | LLaVA-3D 60.1 / VLM-3R 60.7 |
| ScanRefer | ECCV'20 | 单目标定位 | Acc@0.25/0.5 | **GPT4Scene 62.6/57.0** |
| Multi3DRefer | ICCV'23 | 零/单/多目标 | F1@0.25/0.5 | **GPT4Scene 64.5/59.8** |
| Nr3D / Sr3D | ECCV'20 | GT 框指代、细粒度 | Accuracy | 3D-VisTA 64.2 / 76.4（原生 3D） |
| Scan2Cap | CVPR'21 | 稠密描述 | CIDEr@0.5 | GPT4Scene 86.3 / LLaVA-3D 84.1 |
| **VSI-Bench** | CVPR'25 | 视频空间智能，8 类任务 | MCA + MRA | SpatialStack-5B / Cambrian-S-7B **67.5**（人类 79） |

**承重的指标是 VSI-Bench、MMSI-Bench、Real-3DQA、VSI-SUPER**，而不是 ScanQA CIDEr。

**VSI-Bench**（CVPR 2025，[arXiv 2412.14171](https://arxiv.org/abs/2412.14171)）本身有两个必须引用的发现：(1) **CoT −4%、self-consistency −1.1%、ToT −4%**，而同样的 CoT 在 VideoMME 上是 +1.6%——**这种损害是空间任务特有的**；(2) **认知地图有用**：46% → 56%（自生成）→ 66%（GT 地图）。邻近物体定位准确率 64%，但相距 8.9–10 格时降到 **0%**——局部强、无全局地图。

> ⚠️ **VSI-Bench 的效力现已被正式质疑**：**ReVSI**（ICML 2026，[arXiv 2604.24300](https://arxiv.org/abs/2604.24300)）重标了 381 个场景，指出 3D 标注漂移与帧采样不一致（GT 假设可见全场景，模型只看 16–64 帧）。闭源模型基本持平（Gemini 3 Pro 60.9 vs 60.5），但**开源/微调模型掉约 8 分**（Qwen3-VL-32B 53.3 vs 61.8）——说明空间微调过的模型部分是在过拟合标注瑕疵。**任何 2026 年 4 月前的 VSI-Bench 数字都需要加这个注解。**

---

## 7. 三条路线的合流

### 7.1 「3D 编码器」换了身份

2023–24 的二分法已经溶解。2025–26 的共识做法是：**保留 2D CLIP/SigLIP 编码器，并上一个冻结的前馈多视角几何 Transformer（VGGT / CUT3R），后者只吃 RGB 视频**——不要 GT 深度、不要 GT 位姿、不要点云、不要 Mask3D。

三代演进，带数字：

1. **Spatial-MLLM**（NeurIPS 2025 Spotlight，[arXiv 2505.23747](https://arxiv.org/abs/2505.23747)）：Qwen2.5-VL-3B + **VGGT 骨干**，两个 MLP 后**逐元素相加**，约 4B。VSI-Bench **48.4**，以 4B 排在 72B 竞争者之前，16 帧 vs Gemini 约 85 帧。消融：纯 2D SFT 40.0 → 加 VGGT 分支 46.1 → 加空间感知采样 48.4。
2. **VG-LLM**（NeurIPS 2025）：VGGT-1B，4B 总量，VSI-Bench **46.1，以 4B 超过 Gemini-1.5-Pro**。
3. **SpatialStack**（CVPR 2026，[arXiv 2603.27437](https://arxiv.org/abs/2603.27437)）：指出先前工作「只融合深层」，改为把 VGGT 的 11/17/23 层经残差适配器注入 LLM 的**浅层**。**VSI-Bench 67.5，开源第一**（基座 Qwen3.5-4B 53.6，+13.9）。
4. **G²VLM**（CVPR 2026，[arXiv 2511.21688](https://arxiv.org/abs/2511.21688)）：下一步是**不要独立的 3D 编码器**。用 Mixture-of-Transformer-Experts，几何专家（「where」/背侧流）与语义专家（「what」/腹侧流，冻结 Qwen2-VL-2B）**共享自注意力**，几何由 VLM 内部的重建损失学得。2B 规模上 SPAR-Bench **54.87 vs GPT-4o 38.81**，Sintel AbsRel **0.297 vs VGGT 0.335**。消融显示**几何越好、空间推理越好**。

VSI-Bench 轨迹：Gemini-1.5-Pro 45.4 → VG-LLM-4B 46.1 → Spatial-MLLM-4B 48.4 → VLM-3R-7B 60.9 → Cambrian-S-7B 67.5 → SpatialStack-5B 67.5。**两条路线用不同方式到达约 67**——海量空间 SFT 数据（纯 2D）vs 分层几何编码。几何混合路线在**参数效率**上赢，纯 2D + 数据规模在 7B 量级仍有竞争力。**两者都没接近人类**（VSI-Bench 79%、MMSI-Bench 97%）。

### 7.2 长时程是下一个断点

**Cambrian-S**（[arXiv 2511.04670](https://arxiv.org/abs/2511.04670)）提出「空间超感知」四阶段，直到**预测式世界建模**——用**预测误差（surprise）驱动记忆与事件切分**。VSI-Bench **67.5**（比基座 +约 30）。但 **VSI-SUPER**（拼接的 10→240 分钟长巡览）是诚实的负结果：Gemini-2.5-Flash 60 分钟 41.5 VSR，**120 分钟时即便有 1M+ 窗口也出上下文**；VSC 任务所有模型约 **0%**。surprise 记忆是**用短视频精度换长视频存活**（10 分钟 38.3→26.7，但 120 分钟 0.0→30.0）。作者自称「proof-of-concept, not a final solution」。

### 7.3 Mesh 两阵营的合流点

MeshLLM 的局限一节已经指明方向：采用 **MeshAnything-V2 式紧凑序列化** + 更长上下文。而 MeshWeaver 的骨干本来就是 **LLaMA3 式 + RoPE**。**架构已经共享，只差词表。** 可能的综合是：一个 LLM，其 tokenizer 扩展了 BPT/Silksong 级别的网格词表——保住语言先验，把 ~7.5 tok/面降到 ~2。

**MeshCoder 是正交的第三个答案**：如果物体是人造且结构化的，就根本不要压缩几何，直接发射程序——token 成本从此不再追踪面数，还附带可编辑性、语义部件名与分辨率控制。

值得一并跟踪的是**统一理解与生成**的路线：[[ShapeLLM-Omni - 原生三维多模态理解生成与编辑]] 把 mesh 经体素化 + 3D VQVAE 压成 **1024 个离散 token** 并加进 LLM 词表，用纯自回归 next-token 同时做 3D 理解、生成与编辑。这是本文路线 C 与 GPT4Point 的生成支路在更高层次上的汇合。

---

## 8. 局限与评价

### 8.1 已经确立的事实

**Token 预算是这个领域的第一约束，而且有三种成熟解法**：硬上限 query（LL3DA 32 个，与场景大小无关）、objectness 排序筛选（3D-LLaVA top-100 superpoint，超 200 饱和）、一物体一 token（Chat-Scene、LEO，受检测器 recall 限制）。PointLLM 是有价值的反例：**在物体级，不压缩确实更好**（513 token 不池化 52.82 vs 池化 48.72），代价 4 倍算力。

**语言对齐必须由编码器提供，连接器补不上**。这解释了为什么物体级 3D-LLM 几乎全用 ULIP-2 式对比预训练的编码器，也解释了为什么场景级（没有 ULIP-2）只能靠 2D 提升或 2D 蒸馏。

**几何自监督与 2D 语义是互补的，不是竞争的**。Sonata 72.5 + DINOv2 63.1 → **合并 76.4** 是本文最干净的一个证据。

### 8.2 尚未解决的结构性问题

**评测危机是最紧迫的**。物体级 3D 基准只有 7–16% 的题真正需要 3D；SQA3D 滤掉语言先验后模型掉 27–42 分；几乎没有模型视点不变（全对率 0.1%）。**这意味着本文第 3–5 节引用的大量架构结论，证据强度都弱于其表面数字。**

**「一个模型打通所有任务」尚未成立**。LL3DA 的 generalist 在 Nr3D 上 23.94 vs finetune 51.18；Grounded 3D-LLM 自己的判别式 CLASP 在定位上打败自己的生成式 LLM（53.2 vs 48.6，检测 27.2 vs 12.1 AP）。

**长时程空间记忆是断裂的**。VSI-SUPER 上 120 分钟即失效，VSC 全员约 0%。堆上下文窗口无效。

**Mesh 的 token 效率仍不够用**。SOTA 的 1.62 tok/面、16K 面上限，距离游戏角色的 1 万–10 万面仍有距离；而 LLM 路线的 500–800 面上限差了 16–60 倍。

**3D 数据稀缺是所有局限的共同根因**，每篇论文的局限一节都首先提它。这也是为什么「吃 2D 输入的原生 3D 编码器」（VGGT 系）是当前最有效的绕行方案。

### 8.3 接下来可能发生什么

短期可预期：VGGT 类冻结几何编码器成为空间 VLM 的标配，且注入位置从深层移向浅层（SpatialStack）；评测重心从 ScanQA/SQA3D 转向 VSI-Bench 系与 Real-3DQA 式的抗捷径基准；Mesh 侧出现「LLM + 紧凑网格词表」的混合体。

中期值得盯：几何专家与语义专家在同一个 Transformer 内共享注意力（G²VLM 路线）能否稳定 scale；长时程空间记忆是否需要显式的事件切分与世界模型（Cambrian-S 路线）；以及双目标编码器（对比 + 生成式，ReCon++ 路线）是否会取代纯对比成为 3D-LLM 前端的默认选择。

**总体评价：★★★★☆（4/5）。** 这条技术线在 2023–2026 三年里完成了一次完整的范式轮换——从「二选一：用不用 3D 编码器」，到「几乎都是混合的」，再到「3D 编码器可以只吃 2D 输入」。扣掉的一星不在方法，而在**评测**：当一个领域的主流基准有七成以上的题不需要它声称要解决的模态时，其上建立的架构结论都需要打折。本文覆盖的方法中，做了视点鲁棒性或估计位姿消融的（GPT4Scene、VG-LLM、Pts3D-LLM 的自我怀疑）反而是最可信的少数。

---

## 9. 关键文献索引

### 9.1 路线 A：2D 提升

| 工作 | 出处 | 链接 | 3D 怎么注入 |
|---|---|---|---|
| 3D-LLM | NeurIPS 2023 Spotlight | [arXiv 2307.12981](https://arxiv.org/abs/2307.12981) | 多视角 2D 特征反投影成 3D 特征场（三种策略）+ 拼接式正弦 PE + location token |
| LLaVA-3D | ICCV 2025 | [arXiv 2409.18125](https://arxiv.org/abs/2409.18125) | 2D patch + 可学习 MLP 编码的 3D 位置编码**相加**；0.2 m 体素池化 |
| Video-3D LLM | CVPR 2025 | [arXiv 2412.00493](https://arxiv.org/abs/2412.00493) | 场景当带位姿 RGB-D 视频；3D 正弦 PE 相加；最大覆盖选帧 |
| GPT4Scene | ICCV 2025 | [arXiv 2501.01428](https://arxiv.org/abs/2501.01428) | **无 3D 编码器无 3D PE**：BEV 图 + STO 跨视图一致标记 |
| Scene-LLM | WACV 2025 | [arXiv 2403.11401](https://arxiv.org/abs/2403.11401) | 逐像素 CLIP 特征反投影；0.18 m 体素；增量场景记忆 |
| VG-LLM | NeurIPS 2025 | [arXiv 2505.24625](https://arxiv.org/abs/2505.24625) | **VGGT 几何特征**（给出了估计位姿/深度代价的关键消融） |

### 9.2 路线 B：点云 3D-LLM 与编码器

| 工作 | 出处 | 链接 | 关键设计 |
|---|---|---|---|
| PointLLM | **ECCV 2024 Oral**（期刊版为独立论文 PointLLM-V2, TPAMI 2025） | [arXiv 2308.16911](https://arxiv.org/abs/2308.16911) | 513 token 不池化；证明压缩有代价 |
| Pts3D-LLM | preprint（Apple） | [arXiv 2506.05689](https://arxiv.org/abs/2506.05689) | SigLIP + Sonata/PTv3；FPS6D；token 结构对照实验 |
| 3D-LLaVA | CVPR 2025 | [arXiv 2501.01163](https://arxiv.org/abs/2501.01163) | OST 一模块三职；`[SEG]` 出 mask；无需外部检测器 |
| GPT4Point | CVPR 2024 Highlight | [arXiv 2312.02980](https://arxiv.org/abs/2312.02980) | Point-Q-Former；唯一带 3D 生成支路；仅 110M 可训 |
| LL3DA | CVPR 2024 | [arXiv 2311.18651](https://arxiv.org/abs/2311.18651) | 1024×256 → **32 query token**；视觉 prompt |
| Chat-3D | arXiv 题名与 **NAACL 2025 Findings** 发表题名**不同** | [arXiv 2308.08769](https://arxiv.org/abs/2308.08769) | 三阶段；零初始化关系模块；需预分割物体 |
| Chat-Scene（= Chat-3D v2 改名） | NeurIPS 2024（后继 Chat-Scene++, TPAMI 2026） | [arXiv 2312.08168](https://arxiv.org/abs/2312.08168) | `<OBJ_i>` 标识符；Uni3D + **DINOv2**；2D>3D 的关键消融 |
| Grounded 3D-LLM | preprint | [arXiv 2405.10370](https://arxiv.org/abs/2405.10370) | `<ref>` 一对多出 mask；G-SceneCap 102 万短语-区域对 |
| LEO | ICML 2024 | [arXiv 2311.12871](https://arxiv.org/abs/2311.12871) | 物体中心 token；动作映射到最少用 SentencePiece token |
| ShapeLLM / ReCon++ | ECCV 2024 | [arXiv 2402.17766](https://arxiv.org/abs/2402.17766) | 对比 **+** 生成式双目标；换掉 ULIP-2 得 3D MM-Vet **+4.20%** |
| PointNet / PointNet++ | CVPR 2017 / NeurIPS 2017 | [1612.00593](https://arxiv.org/abs/1612.00593) / [1706.02413](https://arxiv.org/abs/1706.02413) | 对称函数保证置换不变；SA = FPS + ball query + PointNet |
| PointMLP | ICLR 2022 | [arXiv 2202.07123](https://arxiv.org/abs/2202.07123) | 残差 MLP + 几何仿射模块；kNN **K=24** |
| Point-BERT | CVPR 2022 | [arXiv 2111.14819](https://arxiv.org/abs/2111.14819) | **MPM = Masked Point Modeling**；dVAE 8192 词表 |
| Point-MAE / Point-M2AE | ECCV 2022 / NeurIPS 2022 | [2203.06604](https://arxiv.org/abs/2203.06604) / [2205.14401](https://arxiv.org/abs/2205.14401) | 去掉 tokenizer 直接回归坐标，反而更好 |
| PointGPT | NeurIPS 2023 | [arXiv 2305.11487](https://arxiv.org/abs/2305.11487) | Morton 序 + 位置泄漏防护 |
| PTv3 | CVPR 2024 Oral | [arXiv 2312.10035](https://arxiv.org/abs/2312.10035) | 序列化注意力换规模；感受野 16→1024 |
| **Sonata** | **CVPR 2025 Highlight** | [arXiv 2503.16429](https://arxiv.org/abs/2503.16429) | **几何捷径**诊断；线性探测 21.8→**72.5**；与 DINOv2 互补 |
| PointCLIP / V2 | CVPR 2022 / ICCV 2023 | [2112.02413](https://arxiv.org/abs/2112.02413) / [2211.11682](https://arxiv.org/abs/2211.11682) | 投影深度图借 CLIP；V2 真实感投影四步 + GPT-3 四类 prompt |
| ULIP / ULIP-2 | CVPR 2023 / CVPR 2024 | [2212.05171](https://arxiv.org/abs/2212.05171) / [2305.08275](https://arxiv.org/abs/2305.08275) | **编码器预训练配方，非 3DLLM**；ULIP-2 是 PointLLM 的编码器来源 |

### 9.3 路线 C：Mesh

| 工作 | 出处 | 链接 | 关键设计 | 本库相关笔记 |
|---|---|---|---|---|
| LLaMA-Mesh | NVIDIA + 清华 | [arXiv 2411.09595](https://arxiv.org/abs/2411.09595) | OBJ 纯文本、64 格量化、**500 面上限**、GSM8K −15.1 | — |
| MeshLLM | preprint | [arXiv 2508.01242](https://arxiv.org/abs/2508.01242) | SAMPart3D 部件分解放大 50 倍数据 + 拓扑前置任务 | — |
| MeshCoder | NeurIPS 2025 | [arXiv 2508.14879](https://arxiv.org/abs/2508.14879) | **mesh → Blender Python 程序**，token 与面数解耦 | — |
| MeshGPT | CVPR 2024 Highlight | [arXiv 2311.15475](https://arxiv.org/abs/2311.15475) | 图卷积 RQ-VAE 学三角形词表，6 tok/面 | [[MeshGPT - 三角网格自回归生成]] |
| MeshAnything / V2 | ICLR 2025 | [2406.10163](https://arxiv.org/abs/2406.10163) / [2408.02555](https://arxiv.org/abs/2408.02555) | 抗噪解码器；**AMT 压缩到 0.497** | — |
| InstantMesh | — | [arXiv 2404.07191](https://arxiv.org/abs/2404.07191) | **前馈重建路线**（非 tokenizer）：Zero123++ + LRM + FlexiCubes | — |
| BPT / MeshWeaver | CVPR 2025 / CVPR 2026 | [2411.07025](https://arxiv.org/abs/2411.07025) / [2606.04688](https://arxiv.org/abs/2606.04688) | 分块索引 0.26；顶点级多层体素 **0.18、16K 面** | — |
| ShapeLLM-Omni | — | [arXiv 2506.01853](https://arxiv.org/abs/2506.01853) | 3D VQVAE 压成 1024 离散 token 进 LLM 词表，统一理解/生成/编辑 | [[ShapeLLM-Omni - 原生三维多模态理解生成与编辑]] |

### 9.4 评测与空间智能前沿

| 工作 | 出处 | 链接 | 为什么重要 |
|---|---|---|---|
| **Real-3DQA** | ICLR 2026 | [arXiv 2603.23523](https://arxiv.org/abs/2603.23523) | 滤掉语言先验后模型掉 27–42 分；视点全对率 0.1% |
| **2D-Cheating** | ACL 2025 Findings | [arXiv 2502.08503](https://arxiv.org/abs/2502.08503) | 物体级 3D 基准只有 7–16% 真需要 3D |
| VSI-Bench | CVPR 2025 | [arXiv 2412.14171](https://arxiv.org/abs/2412.14171) | CoT 在空间任务上**有害**；认知地图有用 |
| **ReVSI** | ICML 2026 | [arXiv 2604.24300](https://arxiv.org/abs/2604.24300) | 质疑 VSI-Bench 标注；开源模型掉约 8 分 |
| Spatial-MLLM / SpatialStack / G²VLM | NeurIPS 2025 / CVPR 2026 ×2 | [2505.23747](https://arxiv.org/abs/2505.23747) / [2603.27437](https://arxiv.org/abs/2603.27437) / [2511.21688](https://arxiv.org/abs/2511.21688) | 冻结 VGGT 几何编码器 → 浅层注入 → 共享注意力的双专家 |
| Cambrian-S / VSI-SUPER | — | [arXiv 2511.04670](https://arxiv.org/abs/2511.04670) | 长时程空间记忆的断点；surprise 驱动记忆 |
| SpatialLM | NeurIPS 2025 | [arXiv 2506.07491](https://arxiv.org/abs/2506.07491) | 点云 → **Python dataclass 代码**表达布局；Sonata + Qwen2.5-0.5B |

### 9.5 本库相关综述

- [[三维网格生成表征演进 - Survey]]——Mesh **生成**侧的表征之争（VecSet / 稀疏体素 / 显式几何 token 化），与本文 §5 互补
- [[多模态大模型架构与训练全流程 - Survey]]——Transformer 原理、视觉编码器四条路线、训练全流程，本文的前置知识
- [[VLM³ - 原生三维学习的视觉语言模型]]——「标准 VLM 本身就是天生的 3D 学习者」，路线 A 的极端形态
- [[CoSMo3D - 开放世界可提示三维语义部件分割]]——3D 部件级分割，与 MeshLLM 的 SAMPart3D 分解、ShapeLLM 的部件理解相关

---

*本综述以「3D 如何进入 LLM 的 token 空间」为主轴重构，定量数字均取自原论文或官方发布配置，命名与版本陷阱已在正文以 ⚠️ 标注。图片为原论文截图，正文与图注为本库整理。未能核实的说法已在正文说明，不作断言。*
