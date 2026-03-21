
## 1. 引言

  

3D Gaussian Splatting（3DGS）最初主要用于高质量、新视角实时渲染，但很快研究者意识到：如果3DGS不仅表示颜色、透明度与几何，还能承载语言与语义特征，那么它就可以成为一个可交互的3D语义场，支持开放词汇检索、文本驱动分割、3D grounding，甚至进一步支持关系理解与机器人交互。沿着这条路线，`LangSplat` 是一个非常关键的起点。它把语言特征直接蒸馏到3D Gaussians中，在保持3DGS高效率渲染优势的同时，建立了可查询的3D language field。

  

从时间线上看，LangSplat发表于2023年底、CVPR 2024接收，属于“给3DGS赋予语义理解能力”这一方向的代表性早期工作。此后，相关研究快速演化，大致沿着三条主线推进：第一，继续提升LangSplat式语言蒸馏的多视角一致性、边界质量和检索精度；第二，从开放词汇分割进一步扩展到3D grounding、指代表达理解和关系推理；第三，从每个场景单独优化的方案，逐步走向可泛化、前馈式、可部署的统一3D语义高斯表示。

  

本文首先总结LangSplat的核心思想、技术细节、优点与局限，然后综述近两年若干与其方向高度相关的工作，并对该研究方向的发展趋势做一个简要归纳。

  

## 2. LangSplat论文总结

  

### 2.1 论文信息

  

- 题目：`LangSplat: 3D Language Gaussian Splatting`

- 作者：Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, Hanspeter Pfister

- 时间：arXiv 2023.12，CVPR 2024

- 链接：`https://arxiv.org/abs/2312.16084`

  

### 2.2 研究背景与动机

  

在LangSplat之前，典型的3D语言场工作以LERF为代表，通常将CLIP特征嵌入NeRF中，从而支持开放词汇的3D检索与定位。但NeRF类方法存在两个突出问题：

  

第一，渲染与查询成本高。NeRF依赖体渲染，训练与推理都比较慢，不适合交互式3D语义查询。

  

第二，语义边界往往模糊。已有方法虽然能够大致定位与文本相关的区域，但在对象边界、尺度适配和实例区分方面表现不够理想，常常出现语义响应弥散的问题。

  

LangSplat的核心动机，就是利用3DGS的高效显式表示，替代NeRF式隐式语言场；同时，针对开放词汇查询中常见的边界模糊问题，引入分层语义监督，使3D语言场不仅快，而且更准。

  

### 2.3 核心方法

  

LangSplat的核心思想可以概括为：把每个3D Gaussian都变成一个带语言特征的基本语义单元，从而构建一个可实时渲染、可文本查询的3D语言高斯场。

  

#### 2.3.1 语言特征蒸馏到3D Gaussians

  

该方法不是像LERF那样把CLIP特征隐式编码进NeRF，而是直接将语言特征赋予一组3D Gaussians。这样做的直接好处是：语言特征的渲染可以沿用3DGS的tile-based splatting流程，而不需要昂贵的体积分层采样，因此推理效率显著提升。

  

#### 2.3.2 场景级语言自编码器

  

CLIP特征维度较高，如果直接让每个Gaussian都携带完整CLIP嵌入，会带来较大的存储与优化负担。为此，LangSplat先训练一个场景级语言自编码器，将语言特征压缩到场景特定的潜空间，再在该潜空间内学习每个Gaussian的语言表示。这样既减少了内存占用，也减轻了直接拟合高维CLIP特征的难度。

  

#### 2.3.3 基于SAM的层次语义学习

  

LangSplat指出，已有3D语言场方法效果不佳的一个重要原因，在于目标边界不清晰、尺度不匹配，导致语言查询结果经常“泛化过头”。为了解决这个问题，作者引入SAM产生层次化mask，把对象不同粒度的区域结构用于训练监督。这样模型不必依赖对不同尺度进行大量查询和额外的DINO正则，即可获得更清晰的语义边界和更稳定的开放词汇定位效果。

  

### 2.4 主要贡献

  

LangSplat的贡献可以概括为以下几点：

  

1. 它首次系统地把3DGS作为3D language field的承载结构之一，证明了3DGS不仅能用于重建与渲染，也能有效承载开放词汇语义。

2. 它提出场景级语言自编码器，将高维CLIP特征压缩到场景潜空间中，降低了显式语言建模的存储成本。

3. 它通过SAM层次语义监督改善了对象边界模糊的问题，提升了开放词汇查询和定位精度。

4. 它在效率上显著优于LERF。根据论文摘要，在1440x1080分辨率下可达到约199倍速度提升。

  

### 2.5 效果与意义

  

LangSplat最大的意义在于，它重新定义了“3D语义场”的实现路径。此前大家更自然地会想到NeRF，因为NeRF是连续场表示；而LangSplat说明，高效的显式高斯表示同样可以承载语言语义，而且在实时性与交互性上更有优势。这一点对机器人、AR/VR、3D内容编辑和场景交互都非常重要。

  

### 2.6 局限性分析

  

虽然LangSplat很有代表性，但从后续研究来看，它也暴露出一些典型局限：

  

1. 多视角2D监督不一致。LangSplat依赖2D foundation model特征进行蒸馏，而这些特征在不同视角下可能并不一致，导致投影回3D时存在语义漂移。

2. 前景与背景语义混叠。由于3DGS渲染是alpha blending式聚合，一些实际贡献很小的背景Gaussians可能也获得与前景相近的语言特征。

3. 仍偏向场景级优化。LangSplat本质上仍是per-scene方式，对每个场景单独训练，泛化能力和部署效率有限。

4. 更偏重开放词汇分割与检索，对关系建模、复杂推理、指代表达理解的支持不足。

  

从这个意义上说，LangSplat像是这个方向的“奠基工作”，而2024年以来的研究，基本都在补它的这些短板。

  

## 3. LangSplat之后的相关工作综述

  

下面按照方法脉络，将与LangSplat方向接近的近期工作分成四类进行梳理。

  

### 3.1 一类：继续改进LangSplat式语言蒸馏

  

这类方法与LangSplat最接近，核心目标仍然是“把2D语言/语义特征稳定地蒸馏进3DGS”，但更关注多视角一致性、粒度控制和可见性建模等问题。

  

#### 3.1.1 GAGS: Granularity-Aware Feature Distillation for Language Gaussian Splatting（2024）

  

- 链接：`https://arxiv.org/abs/2412.13654`

  

GAGS直接针对LangSplat类方法中最核心的难点之一：2D特征在多视角下不一致，导致3D蒸馏监督不稳定。其改进点主要有两个。

  

第一，它将SAM的prompt point density与相机距离关联起来，使不同视距下得到的分割粒度更加一致。第二，它引入可学习的granularity factor，在蒸馏过程中自动筛选跨视角更一致的2D特征，而不是把所有特征都无差别地灌输到3D Gaussians中。

  

与LangSplat相比，GAGS不是重构整体范式，而是在“特征蒸馏”这一关键环节做了更细致的稳健化设计。因此它可以看作是LangSplat方法论上的直接延续和增强版。它说明该方向的研究重点已经从“能否做3D语言场”转向“如何让蒸馏监督更可信”。

  

#### 3.1.2 Semantic Consistent Language Gaussian Splatting for Point-Level Open-vocabulary Querying（2025）

  

- 链接：`https://arxiv.org/abs/2503.21767`

  

这篇工作进一步指出，LangSplat类方法不仅有多视角监督不一致问题，而且在查询阶段也常常只能做到区域级、视图级响应，难以稳定落到3D点或Gaussian级别。

  

因此，该方法提出两个关键设计：一是通过跨视角mask tracking构造语义一致的ground truth，从源头上减少监督噪声；二是设计GT-anchored querying机制，先检索蒸馏后的GT表示，再以此为锚点查询具体Gaussians，实现point-level open-vocabulary querying。

  

相较LangSplat，这篇工作强调“点级”精细查询能力，更贴近机器人操作和空间交互场景。它反映出该方向的精度需求正在从“找到物体大概在哪”过渡到“精确找到3D中哪些点属于目标”。

  

#### 3.1.3 Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting（2025）

  

- 链接：`https://arxiv.org/abs/2509.05515`

  

这篇工作非常有代表性，因为它明确指出了LangSplat类聚合方法中的一个渲染层面问题：某个像素的语言特征在聚合时，前景高斯和几乎不可见的背景高斯可能被赋予相近语义，这会带来明显的语义污染。

  

作者提出VALA（Visibility-Aware Language Aggregation），通过计算每条光线上各Gaussian的边际贡献，利用visibility-aware gate只保留真正可见、贡献显著的Gaussians；同时在多视角特征融合时，采用cosine空间中的streaming weighted geometric median抑制噪声。

  

与LangSplat相比，VALA把重点从“怎么蒸馏”推进到“怎么聚合”。这一点很重要，因为它说明语言高斯场的质量，不仅受监督信号影响，也强烈依赖3DGS渲染中的可见性机制。

  

### 3.2 二类：从开放词汇分割走向grounding与复杂语言理解

  

LangSplat主要支持开放词汇查询和分割，但现实场景中的人机交互通常不是单个名词检索，而是更复杂的自然语言表达，比如“桌子旁边的红色杯子”或者“被椅子挡住的箱子”。这推动研究从分割走向grounding、指代和推理。

  

#### 3.2.1 ReasonGrounder: LVLM-Guided Hierarchical Feature Splatting for Open-Vocabulary 3D Visual Grounding and Reasoning（2025）

  

- 链接：`https://arxiv.org/abs/2503.23297`

  

ReasonGrounder将大型视觉语言模型（LVLM）引入3DGS语义理解流程中，目标是处理隐式描述、遮挡目标以及需要常识推理的开放词汇3D grounding问题。其核心思想是构建层级3D feature Gaussian fields，并依据物体物理尺度做自适应分组，再结合SAM mask和多视角CLIP特征进行定位。

  

相较LangSplat，ReasonGrounder的变化有两点。第一，它的任务从单纯分割扩展为grounding与reasoning。第二，它开始借助更强的LVLM解释复杂文本，而不是只依赖CLIP式对齐。这代表该领域开始从“语义匹配”迈向“多模态推理”。

  

#### 3.2.2 ReferSplat: Referring Segmentation in 3D Gaussian Splatting（2025）

  

- 链接：`https://arxiv.org/abs/2508.08252`

  

ReferSplat提出一个新的任务R3DGS，即在3D Gaussian场景中根据自然语言指代表达进行分割。这类表达通常包含属性、空间关系，甚至目标可能当前视角不可见或被遮挡，因此难度明显高于LangSplat中的开放词汇查询。

  

该工作的重要性在于，它不再把文本看成简单类别提示词，而是将其视为含空间关系和对象属性的组合表达，并显式建模3D Gaussian点与自然语言之间的空间感知关系。它还构建了Ref-LERF数据集，推动这一任务标准化。

  

如果说LangSplat解决的是“文本标签驱动的3D响应”，那么ReferSplat解决的是“自然语言描述驱动的3D目标解析”。这一步非常关键，因为它更接近真实用户交互方式。

  

### 3.3 三类：从对象级语义走向关系建模与场景图理解

  

LangSplat类方法本质上更擅长“物体在哪里”，但对于“物体之间是什么关系”并不擅长。随着研究深入，越来越多工作开始把3DGS中的语义表示提升到结构化场景理解层面。

  

#### 3.3.1 GaussianGraph: 3D Gaussian-based Scene Graph Generation for Open-world Scene Understanding（2025）

  

- 链接：`https://arxiv.org/abs/2503.04034`

  

GaussianGraph认为，仅仅把压缩后的CLIP特征嵌入3D Gaussians，虽然能支持基本定位，但对象分割精度不高，而且缺乏空间推理能力。为此，它提出自适应语义聚类与scene graph generation，将3DGS中的对象表示组织成“对象-属性-关系”的图结构。

  

其方法中有两个值得注意的点：一是通过“Control-Follow”聚类策略根据场景尺度与特征分布适应性聚类；二是通过3D correction module对2D模型提取的空间关系进行三维一致性修正，过滤不合理关系。

  

相较LangSplat，这篇工作把3D高斯语义场从“连续特征场”升级为“结构化语义图”，是3DGS语义理解走向高层场景理解的重要标志。

  

#### 3.3.2 COS3D: Collaborative Open-Vocabulary 3D Segmentation（2025）

  

- 链接：`https://arxiv.org/abs/2510.20238`

  

COS3D指出，现有高斯语义分割方法要么依赖单一language field，导致分割质量有限；要么依赖外部类无关分割结果，容易产生误差累积。为解决这个问题，它提出collaborative field，由instance field和language field共同组成，并通过instance-to-language映射与两阶段训练实现二者协同。

  

它的核心思想是：3D开放词汇分割不应只靠语言特征场，也应同时显式保留实例层面的结构信息。推理阶段再用adaptive language-to-instance prompt refinement，实现从文本提示到高质量实例分割的过渡。

  

相对于LangSplat，COS3D不再满足于单一语言场，而是引入“语言+实例”双场协作，这说明该领域开始重视纯language embedding在分割任务中的表示瓶颈。

  

#### 3.3.3 ReLaGS: Relational Language Gaussian Splatting（2026）

  

- 链接：`https://arxiv.org/abs/2603.17605`

  

ReLaGS可以看作LangSplat路线在关系推理层面的进一步延伸。它构建分层language-distilled Gaussian scene，并进一步建立3D semantic scene graph，在此基础上用图神经网络进行对象间关系推理。它支持的任务不只是开放词汇分割，还包括scene graph generation和relation-guided retrieval。

  

这类方法的意义在于，它们开始把3DGS当成“3D世界模型”的底座，而不只是语义检索载体。相比LangSplat的对象级语义，ReLaGS已经走向“对象内语义 + 对象间关系”的统一建模。

  

### 3.4 四类：从场景专属优化走向可泛化与基准化

  

LangSplat的另一大限制，是每个场景都需要单独优化。随着应用需求增强，研究者开始转向更具可扩展性的前馈式方案，并建立统一基准来衡量不同方法。

  

#### 3.4.1 Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images（2025）

  

- 链接：`https://arxiv.org/abs/2508.03643`

  

Uni3R提出一个前馈式框架，直接从无位姿多视图图像回归带开放词汇语义特征的3D Gaussians。它通过Cross-View Transformer融合任意多视图输入，联合完成3D重建、开放词汇分割和深度预测。

  

相较LangSplat，Uni3R的根本变化在于：它不再是per-scene optimization，而是generalizable feed-forward inference。这意味着模型可以在新场景上快速推理，而不必像LangSplat那样重新训练。对大规模应用和机器人在线感知来说，这种变化可能比单点性能提升更重要。

  

#### 3.4.2 SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting（2025）

  

- 链接：`https://arxiv.org/abs/2506.08710`

  

SceneSplat++虽然不是一个新的语义场模型，但对该领域非常关键。它系统性地将Language Gaussian Splatting方法分成三类：per-scene optimization-based、per-scene optimization-free、generalizable approach，并在1060个场景上直接从3D空间而不是仅从渲染2D视图上进行评测。

  

此外，它还提供GaussianWorld-49K这一大规模3DGS数据资源，用于支撑可泛化方法学习更强的数据先验。其结论非常值得注意：generalizable范式在大规模评测中展现出明显优势。

  

对于理解LangSplat之后的发展，这篇工作有“坐标系”的意义。它帮助我们把LangSplat放回整个方法谱系中，并清楚看到该方向正在从“单场景优化”迈向“数据驱动泛化”。

  

## 4. 综合分析：LangSplat之后的研究趋势

  

结合上述工作，可以把LangSplat之后的发展趋势概括为以下几个方面。

  

### 4.1 从“能嵌入语言”到“如何高质量嵌入语言”

  

LangSplat解决了“3DGS能否承载语言特征”这一问题，但后续研究表明，真正困难的不是嵌入本身，而是多视角特征的不一致、边界模糊、可见性误差和粒度错配。因此像GAGS、Semantic Consistent LGS、VALA这样的工作，都在更细粒度地修正LangSplat中的蒸馏与聚合机制。

  

### 4.2 从开放词汇分割到复杂语言理解

  

LangSplat本质上更适合处理单对象关键词查询，而ReasonGrounder、ReferSplat等工作开始处理更长、更复杂、更具关系性的自然语言表达。这说明3DGS语义理解正在从“分类/分割型语义”向“grounding/指代/推理型语义”升级。

  

### 4.3 从对象语义到关系语义

  

GaussianGraph、ReLaGS等方法表明，只知道“这个物体是什么”已经不够，还需要知道“这个物体和其他物体之间的关系是什么”。这意味着未来3DGS不只是一个带语言标签的点云替代物，而可能演化成显式、可推理的3D语义结构表示。

  

### 4.4 从场景专属训练到可泛化前馈模型

  

这是一个非常明显的趋势。LangSplat代表的是scene-specific optimization时代，而Uni3R以及SceneSplat++反映的，是generalizable language 3DGS正在成为新主流。对于真实部署，尤其是机器人与在线AR场景，这种范式转变几乎是必然的。

  

## 5. 对LangSplat的总体评价

  

如果把LangSplat放在整个研究脉络中看，它的价值主要体现在三个方面。

  

第一，它把3DGS正式带入了3D语言场研究，使高效显式表示第一次在这一任务上展现出压倒性的速度优势。

  

第二，它提出了场景级语言自编码器与SAM层次语义监督，为后续方法提供了两个重要启发：高维语言特征需要压缩建模，语义监督需要多粒度结构先验。

  

第三，它定义了“Language Gaussian Splatting”这条技术路线。后续无论是做多视角一致性、点级检索、可见性感知、协同场建模，还是做scene graph、grounding、泛化前馈，本质上都可以视为对LangSplat路线的补全与拓展。

  

当然，LangSplat也存在典型的早期方法局限：依赖场景级优化，语义对齐质量受2D监督噪声制约，缺乏关系建模与复杂推理能力。但也正因如此，它成为了后续大量工作的出发点。

  

## 6. 小结

  

总体来看，LangSplat是3DGS语义理解方向上的里程碑工作。它证明了3D Gaussian Splatting不仅能高效表示外观和几何，也能承载开放词汇语言语义，从而把3D重建、语言理解与空间交互连接起来。

  

而在LangSplat之后，相关工作正沿着三个更深入的方向发展：

  

1. 提升语言蒸馏与聚合的稳定性，使3D语义场更精确、更鲁棒；

2. 扩展到grounding、指代、关系推理等更高层次的语义任务；

3. 从单场景优化走向可泛化、可扩展、可部署的统一3D语义高斯表示。

  

因此，如果把LangSplat视为第一阶段，那么当前该领域已经进入第二阶段：研究者不再满足于“让3DGS有语义”，而是在追求“让3DGS具备可泛化、可推理、可交互的真实世界理解能力”。

  

## 7. 参考文献列表

  

1. Qin, M., Li, W., Zhou, J., Wang, H., Pfister, H. `LangSplat: 3D Language Gaussian Splatting`. arXiv:2312.16084, CVPR 2024.

2. Peng, Y., Wang, H., Liu, Y., Wen, C., Dong, Z., Yang, B. `GAGS: Granularity-Aware Feature Distillation for Language Gaussian Splatting`. arXiv:2412.13654, 2024.

3. Yin, H., Zhan, H., Xu, Y., Yeh, R. A. `Semantic Consistent Language Gaussian Splatting for Point-Level Open-vocabulary Querying`. arXiv:2503.21767, 2025.

4. Wang, S., Li, K., Liang, S., Alegret, E., Ma, J., Navab, N., Gasperini, S. `Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting`. arXiv:2509.05515, 2025.

5. Liu, Z., Wang, Y., Zheng, S., Pan, T., Liang, L., Fu, Y., Xue, X. `ReasonGrounder: LVLM-Guided Hierarchical Feature Splatting for Open-Vocabulary 3D Visual Grounding and Reasoning`. arXiv:2503.23297, 2025.

6. He, S., Jie, G., Wang, C., Zhou, Y., Hu, S., Li, G., Ding, H. `ReferSplat: Referring Segmentation in 3D Gaussian Splatting`. arXiv:2508.08252, ICML 2025 Oral.

7. Wang, X., Yang, D., Gao, Y., Yue, Y., Yang, Y., Fu, M. `GaussianGraph: 3D Gaussian-based Scene Graph Generation for Open-world Scene Understanding`. arXiv:2503.04034, 2025.

8. Zhu, R., Hui, K.-H., Liu, Z., Wu, Q., Tang, W., Qiu, S., Heng, P.-A., Fu, C.-W. `COS3D: Collaborative Open-Vocabulary 3D Segmentation`. arXiv:2510.20238, NeurIPS 2025.

9. Sun, X., Jiang, H., Liu, L., Nam, S., Kang, G., Wang, X., Sui, W., Su, Z., Liu, W., Wang, X., Park, E. `Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images`. arXiv:2508.03643, 2025.

10. Ma, M., Ma, Q., Li, Y., Cheng, J., Yang, R., Ren, B., Popovic, N., Wei, M., Sebe, N., Van Gool, L., Gevers, T., Oswald, M. R., Paudel, D. P. `SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting`. arXiv:2506.08710, 2025.

11. Xie, Y., Arafa, A., Javanmardi, A., Millerdurai, C., Hu, J. C., Wang, S., Pagani, A., Stricker, D. `ReLaGS: Relational Language Gaussian Splatting`. arXiv:2603.17605, CVPR 2026.