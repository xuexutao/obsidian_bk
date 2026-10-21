---
type: paper-note
status: complete
domain: World-Models
paper: Scaling Spatial Intelligence with Multimodal Foundation Models
year: 2026
venue: CVPR 2026
arxiv: 2511.13719
doi: 10.48550/arXiv.2511.13719
source: https://arxiv.org/abs/2511.13719
project: null
code: https://github.com/OpenSenseNova/SenseNova-SI
models: https://huggingface.co/collections/sensenova/sensenova-si
tags:
  - Spatial-Intelligence
  - Multimodal-Foundation-Models
  - Data-Scaling
  - Benchmark-Robustness
created: 2026-10-16
updated: 2026-10-16
---

# SenseNova-SI：空间智能多模态基础模型

## 基本信息

- **作者/机构**：Zhongang Cai、Ruisi Wang、Chenyang Gu 等；SenseTime/OpenSenseNova 联合学术团队
- **版本**：arXiv v4，2026-03-28；报告基于 SenseNova-SI v1.1；CVPR 2026
- **底座**：Qwen3-VL、InternVL3 和 Bagel 三类多模态基础模型；主分析以 InternVL3-8B 为主。
- **核心资源**：SenseNova-SI-8M 空间指令数据、开源模型权重与训练/评测代码。

## 一句话结论

SenseNova-SI 的最强结论不是“提出了新空间架构”，而是证明了按空间能力 taxonomy 定向组织约 8M 训练样本，能在同一训练 recipe 下跨 Qwen3-VL、InternVL3、Bagel 稳定提升多项空间 benchmark；不过 scaling 曲线已显著趋于饱和，主表仍混有潜在训练/测试域重叠，真实空间泛化证据主要来自去偏、跨数据集迁移、长帧外推与仿真机器人任务，距离真实机器人闭环和严格无泄漏评测仍有明显距离。

---

## 1. 研究问题与定位

### 1.1 问题不是“模型不够大”，而是空间监督不成体系

通用 MLLM 在图文知识、OCR 和语义识别上快速进步，但常在距离估计、跨视角对应、相机运动、物体朝向和路线规划上失败。论文把原因主要归为高质量、任务平衡的空间数据稀缺，研究问题因而是：在不改底座架构的前提下，系统化扩大空间监督能否形成可迁移的空间能力？

### 1.2 方法本质：同一监督目标，不同底座的 continued tuning

SenseNova-SI 没有引入新几何模块、显式 3D memory 或新的损失函数。训练样本可抽象为视觉/视频输入 $x$、问题 $q$ 和答案 $y$，沿用底座的自回归监督：

$$\mathcal L_{\mathrm{SFT}}=-\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x,q,y_{<t}).$$

论文没有单独写出这条公式或新的空间目标；真正的变量是数据量、能力覆盖、混合比例、底座规模和帧数。因此，这是一篇“数据与评测驱动”的基础模型研究，而不是结构创新论文。

---

## 2. 空间能力 taxonomy

### 2.1 五类能力

论文沿用 EASI 思路，但训练 taxonomy 聚焦五类：

1. **Metric Measurement（MM）**：相机—物体、物体—物体距离，以及物体到场景尺度的尺寸估计。
2. **Spatial Relations（SR）**：在 3D 坐标概念下判断前后、左右、上下、远近和相对尺度。
3. **Mental Reconstruction（MR）**：从有限 2D 观察恢复物体 3D 结构，诊断任务包括判断可见面。
4. **Perspective-taking（PT）**：分为跨视角点/物体对应、相机运动推理、allocentric 视角变换。
5. **Comprehensive Reasoning（CR）**：组合多种空间能力、长期记忆与多步推理；论文仅把已有数据作为轻量补充。

与 EASI 的六类 taxonomy 相比，SenseNova-SI 未把 Deformation & Assembly 作为主训练轴，并明确没有重点扩大 CR。这意味着 8M 规模主要覆盖“静态/多视角几何问答”，不等于完整具身空间智能。

![](assets/SenseNova-SI%20-%20空间智能多模态基础模型/figures/02-data-taxonomy.png)

> **图 2（论文 Figure 2）**：上方 Sankey 图把 General QA、Open3D-VQA、CLEVR、SAT、MultiSpa、ViCA、VSI-590K 等来源映射到非空间、2D grounding、MM、SR、PT 对应/相机运动/allocentric 变换、MR 和 CR；下方给出各能力样例。来源：官方 arXiv LaTeX 图源，论文数据章节。

### 2.2 PT 是真正的能力瓶颈

PT 不只是识别两张图中同一个物体，还要求推断相机变换并在 ego/object/allocentric 坐标间转换。论文观察到 InternVL3 2B 和 8B 在 MM、SR、CR 上的 scaling 趋势相近，但在 PT 上明显分叉，推测 2B 容量不足以稳定学习视角变换。这是“数据规模 × 模型容量”存在交互的证据，而不是所有能力都可由小模型同样吸收。

---

## 3. SenseNova-SI-8M 数据构建

### 3.1 三层来源

- **General QA：约 0.6M**。VSR、SPEC、GQA、VQA、IconQA，用于保留通用图像理解。
- **社区空间数据：约 3.3M**。Open3D-VQA、CLEVR 系列、REL3D、SAT、GRiD-3D、MultiSpa、MindCube、ViCA、VLM-3R、VSI-590K。
- **进一步空间扩展：约 4.5M**。从 MessyTable、ScanNet/ScanNet++、SUN RGB-D、CA-1M、Ego-Exo4D、Matterport3D 等有 3D/多视角标注的数据生成 QA，补足 MR 与 PT 的不足。

正文相加得到 **8.5M QA**，而模型和 scaling table 使用“8M”。论文没有清楚说明 0.5M 差异来自去重、过滤、采样还是命名取整；复现时应以最终发布数据清单为准，不能把“8M”当精确原始总量。

### 3.2 数据扩展的关键不是均匀加量

原社区数据被 MM/SR 主导，PT 与 MR 稀缺。进一步扩展集中于跨视角 correspondence、camera motion、allocentric transformation 和 object reconstruction。作者的核心主张是“coverage scaling”而非只增加样本；这与后续 PT 和 MindCube 的大幅提升一致。

### 3.3 潜在污染与域重叠

论文在训练中明确纳入 MindCube、VSI-590K 等数据，并在 MindCube-Tiny、VSI-Bench 等相关 benchmark 上报告主结果。作者对 MindCube 使用 non-overlapping MindCube-10K 训练/测试设置，并做单数据集跨域实验，但并未给出所有训练样本与八个 benchmark 测试图像/场景的全局哈希去重或场景级 contamination audit。ScanNet、Matterport3D、Ego-Exo4D 等基础资产也可能被多个 benchmark 复用。

因此，主表证明“按当前 protocol 的 benchmark 性能”，不能单独证明对完全未见真实空间的泛化。论文自己也承认，大规模 mixed-domain training 使能力迁移与偶然 overlap 难以区分。

---

## 4. 训练设置

三个底座都在同一数据上训练 1 epoch，使用 128 张 GPU、总 batch size 2048、AdamW、学习率 $5\times10^{-6}$，每次约 3 天。视频训练最多采样 16 帧。

论文未披露 GPU 型号、分辨率/token budget、weight decay、warmup、数据采样温度、各能力 mixture 比例、训练框架吞吐与 checkpoint 策略。虽然“128 GPU × 3 天”提供了量级，但仍不足以精确复现。

![](assets/SenseNova-SI%20-%20空间智能多模态基础模型/figures/01-teaser.png)

> **图 1（论文 Figure 1）**：中心雷达图给出 InternVL3-8B 版本的八项结果；四周 scaling 曲线显示 0→8M 在 PT、MM、SR、CR 上总体提升，但高数据区间逐渐平台化。来源：官方 arXiv LaTeX 图源，论文首页。

---

## 5. 主结果与数据缩放

### 5.1 八项空间 benchmark

SenseNova-SI-InternVL3-8B 在 EASI-8 protocol 下得到：VSI-Bench 68.8、MMSI 43.3、MindCube 85.7、ViewSpatial 54.7、SITE 47.7、BLINK 63.9、3DSR 55.5、EmbSpatial 72.0，平均 61.5。

相对 InternVL3-8B，VSI +26.7、MMSI +15.3、MindCube +44.2、ViewSpatial +16.0、SITE +6.6、BLINK +10.4、3DSR +11.2；但 EmbSpatial **下降 4.3**（76.3→72.0）。这反驳了“所有空间任务无损提升”的过强表述，也说明数据 taxonomy 与某些 egocentric embodied relation 数据并非完全对齐。

### 5.2 0M→8M scaling

| 数据量 | VSI | MMSI | MindCube-Tiny | ViewSpatial | SITE |
|---:|---:|---:|---:|---:|---:|
| 0M | 42.1 | 28.0 | 41.5 | 52.0 | 42.1 |
| 1M | 56.3 | 36.0 | 58.8 | 55.2 | 44.5 |
| 2M | 60.3 | 39.6 | 76.2 | 56.7 | 47.1 |
| 4M | 62.7 | 41.9 | 83.9 | 55.6 | 45.2 |
| 6M | 66.3 | 41.8 | 85.0 | 55.2 | 47.0 |
| 8M | 68.7 | 43.3 | 85.6 | 54.6 | 47.7 |

最显著增益发生在前 1–2M；MindCube 在 4M 后接近饱和，ViewSpatial 甚至从 2M 的 56.7 回落到 8M 的 54.6，SITE 也非单调。这表明“总体 scaling 有效”成立，但“每项能力随数据稳定单调提升”不成立。作者也明确认为 data scaling alone 不太可能达到人类水平。

### 5.3 通用能力是否保持

InternVL3-8B 版本在 MMBench-EN 从 81.7 升到 84.9，但 MMStar 68.2→65.4、AI2D 85.2→79.0、OCRBench 880→815、DocVQA 92.1→84.9、MMMU 55.6→49.4、Vid-MME 76.1→72.2。论文用“数据多样性缓解灾难遗忘”概括，但逐项看是**部分保留而非无损保持**。对实际部署，应把空间增益与 OCR、文档、学科推理退化一起评估。

---

## 6. 消融、shortcut 与泛化证据

### 6.1 帧数外推

模型训练最多 16 帧，推理时 VSI 在 16/32/64/128 帧为 64.6/68.7/68.8/66.3；VSI-Debiased 为 58.9/62.8/62.4/59.7。32–64 帧优于训练窗口，128 帧下降，说明有有限长上下文外推，但不能支持“任意长视频空间记忆”。

### 6.2 语言捷径诊断

MindCube-SFT-RawQA 在有图 51.7、无图 50.7，几乎不依赖视觉；SenseNova-SI 为 85.6→52.5，证明其主要增益确实来自视觉输入。答案选项循环测试中，SenseNova-SI Standard/Soft/Hard 为 85.6/84.0/75.6，而 MindCube baseline 为 51.7/45.8/23.1；后者显著依赖选项模式。

VSI-Debiased 上 SenseNova-SI 的下降也小于 Cambrian-S-7B。需要谨慎：无图下降证明“使用了视觉”，并不自动证明形成正确 3D 表征；模型仍可能利用纹理、场景类别和 benchmark 特定的视觉 shortcut。

### 6.3 跨任务 spill-over

作者用**单数据集训练、异域任务测试**减弱 mixed-data overlap：Ego-Exo4D view transformation 训练迁移到 SITE Maze Pathfinding 和 MMSI Pos-Cam-Cam；MessyTable 跨视角 correspondence 迁移到 MMSI Pos-Cam-Cam 与 Attr-Appr。能力转移矩阵中，PT-only 训练使 MR +46.1%、PT +83.8%、CR +10.9%，而 SR-only 训练却让 MR -61.5%、PT -19.2%、CR -6.0%。这说明数据并非天然协同，窄 SR 监督甚至会伤害更复杂能力。

![](assets/SenseNova-SI%20-%20空间智能多模态基础模型/figures/03-capability-transfer.png)

> **图 3（论文 Figure 3）**：上方为 Ego-Exo4D view transformation 对 maze/camera-camera reasoning 的迁移，下方为 MessyTable correspondence 对相机旋转、跨视角位置和属性任务的迁移；绿色箭头标出相对提升。来源：官方 arXiv LaTeX 图源，论文 capability emergence 章节。

### 6.4 仿真具身泛化

在 EmbodiedBench 空间子集上不做额外 fine-tuning，InternVL3-8B 的 OP/SIP 成功率为 10.4/20.8，SenseNova-SI 为 16.6/33.3，提升约 60%；但 GPT-4o 仍为 37.5/45.8。该实验提供“空间训练能转移到动作规划”的证据，却仍是虚拟 Franka Panda、带 bounding box/object grounding prompt 的设置，不等于真实机器人感知—控制闭环。

### 6.5 Spatial CoT 反直觉结果

在 VSI 上，单独针对 relative-direction 子集训练的长 CoT 反而低于 base；在完整 QA+CoT 上，作者 CogMap 达 49.2，GRPO 后降到 43.1。长文本空间推理容易自相矛盾，且 token 成本远高于直接回答。论文据此判断，空间智能可能需要超越纯文本 CoT 的中间表示。

---

## 7. 局限与批判性评价

### 7.1 论文明确或数据直接显示的局限

- scaling 高段收益递减，部分 benchmark 非单调。
- 128 帧外推下降，长时空间记忆仍有限。
- CoT/RL 增益不稳定，长文本会累积空间错误。
- 训练数据清单总量与“8M”命名有 0.5M 口径差异。
- EmbSpatial 和多项通用 benchmark 出现退化。

### 7.2 benchmark 泄漏与语言捷径

论文对语言 shortcut 的诊断较扎实：VSI-Debiased、无视觉 MindCube、soft/hard circular 都直接测了非视觉与选项位置偏置。但对**数据泄漏**的处理较弱：没有跨所有基础场景、图像和 QA 的统一去重报告；训练来源与评测来源存在同名数据族或共享 3D 资产的可能。单数据集跨域实验缓解但没有完全消除该问题。

### 7.3 我的判断

SenseNova-SI 最有价值的是把“空间数据应该补什么”说清楚：真正缺的不是更多左右关系，而是跨视角 correspondence、camera motion 和 allocentric transformation。它也用反例说明，错误的窄数据扩张会造成负迁移。

但主结果仍是 benchmark-centric SFT。模型没有显式深度、pose、3D token 或世界状态，无法从结果表确认它内部形成了可用于闭环预测的几何表示。应把它视为强空间问答 foundation，而不是完整 world model。

---

## 8. 复现与实践建议

### 8.1 数据复现

1. 先实现 EASI 五类训练 taxonomy，并记录每条样本的源场景、相机、任务和答案生成器。
2. 对所有评测集做 image hash、scene ID、3D asset ID 和问题模板四级去重。
3. 按能力而不是只按数据源报告 mixture，尤其公开 PT/MR 的实际样本数。
4. 保留 General QA 对照，监控 OCR/DocVQA/MMMU 等退化。

### 8.2 实验复现

以 InternVL3-8B 为主线，固定 1 epoch、batch 2048、$5\times10^{-6}$，至少训练 0M/1M/2M/4M/8M checkpoint。除主表外，必须复现：无图测试、circular test、VSI-Debiased、16→128 帧、单能力 transfer matrix 和 EmbodiedBench 无微调评测。

### 8.3 资源

官方报告每个底座用 128 GPU 训练约 3 天。若资源有限，优先复现 0M/1M/2M 早期斜率，因为大部分增益发生在这里；再以单数据集 transfer 验证“能力迁移”而不是盲目追求 8M。

---

## 9. 后续问题

1. 若加入显式 pose/depth/3D latent，能否把 ViewSpatial 的非单调 scaling 变成更稳定的结构学习？
2. PT-only 数据为何能大幅提升 MR，而 SR-only 会严重伤害 MR/PT？是数据域、答案格式还是能力本身导致？
3. 如何把仿真 EmbodiedBench 迁移到真实机器人，并避免 bounding box prompt 替模型完成 grounding？
4. 是否能用图结构、坐标场或可微 3D memory 取代冗长文本 CoT？

---

## 10. 材料来源

- 官方论文：<https://arxiv.org/abs/2511.13719>
- 代码：<https://github.com/OpenSenseNova/SenseNova-SI>
- 模型集合：<https://huggingface.co/collections/sensenova/sensenova-si>
- 本地来源与图号映射：[[assets/SenseNova-SI - 空间智能多模态基础模型/sources]]
