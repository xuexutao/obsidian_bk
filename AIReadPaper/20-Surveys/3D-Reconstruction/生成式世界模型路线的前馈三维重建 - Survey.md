# 生成式世界模型路线的前馈三维重建：Lyra、WorldMirror 与同路代表作巡礼

## 基本信息

**报告主题**：回应「Lyra 这一系列之外还有哪些 feed-forward 工作、WorldMirror 之类的混元系工作是怎么回事」——把 2025–2026 年生成式三维场景重建 / 世界生成这条线上的主要前馈工作做一次统一梳理

**覆盖范围**：2024.12（Wonderland）— 2026.09（HY-World 2.0 / Mirage / MosaicMem），重点为 2025.03 之后

**与既有笔记的关系**：本馆已有三篇互补笔记——[[前馈式三维重建与生成式补全 - Survey]]（问题演化主轴）、[[前馈式三维重建 - 分支谱系总览]]（输出表征主轴）、[[Lyra系列 - 生成式三维场景重建]] 与 [[WorldMirror - 多模态先验驱动的通用前馈三维重建]]（两篇单点精读）。本篇换一个切面：以「**3D 信息在哪里、记忆怎么维护**」为主轴，把 Lyra / WorldMirror 所在的这条生成式路线上的散兵游勇收拢成阵。

**撰写日期**：2026-10-10

---

## 1. 先厘清：「前馈」在这批工作里有三种含义

「前馈 = 推理时不做逐场景优化」这个定义过于宽泛，落到这批工作里其实叠加了三层不同的含义，分清它们才不会把 WorldMirror 和 GEN3C 混为一谈：

1. **几何前馈**：不做优化，直接从图像回归几何（深度、点图、位姿）。代表是 DUSt3R→VGGT→DA3→π³ 这条 pointmap 主线，WorldMirror 1.0 是它在 2026 年的「条件注入」完成态。
2. **重建前馈**：输入可能是**生成的**多视图内容，但从内容到 3D 表示（通常是 3DGS）这一步不做优化、一次前向回归。代表是 Wonderland 的 LaLRM、Matrix-3D 的 Panorama LRM、Lyra 的自蒸馏解码器。
3. **世界生成前馈**：整个系统端到端地**生成**一个可探索的三维世界，用某种显式 3D 缓存/记忆维持长程一致性。代表是 GEN3C、Voyager、Spatia、Mirage、HY-World 系列。

2026 年最能打的工作几乎都是三层叠加：**生成模型管「编内容」，几何基础模型管「落几何」，记忆机制管「不忘记」**。

---

## 2. 总表：一条线上的全部主要工作

| 工作 | 时间 / 机构 | 输入 → 输出 | 核心机制 | 3D 记忆形态 | 开源 |
|---|---|---|---|---|---|
| **Wonderland** | 2024.12 / Snap·多伦多·UCLA（CVPR 2025） | 单图 + 轨迹 → 3DGS | 相机引导视频扩散（CogVideoX5B-I2V）+ LaLRM 从 video latent 直接回归高斯 | 无（一次前向，广度受生成窗口限制） | 权重未放出 |
| **GEN3C** | 2025.03 / NVIDIA·多伦多（CVPR 2025） | 单图/多图/视频 + 轨迹 → 3D 一致视频 | 时空 3D 缓存（深度反投影点云）渲染 + 遮挡掩码作条件，max-pooling in-layer 融合 | RGB 点云缓存（自回归更新） | 部分开源 |
| **Voyager** | 2025.06 / 哈工大·东南·**腾讯混元** | 单图 + 轨迹 → RGB-D 视频 → 点云序列 | 联合生成对齐 RGB+Depth（高度维拼接），Control Blocks 注入每层 DiT，world cache 点剔除 | RGB-D 点云缓存 | ✅ 全开源 |
| **HunyuanWorld 1.0** | 2025.07 / 腾讯混元 | 文本/单图 → 语义分层 3D mesh | 全景世界代理 + agentic 分层 + 分层重建 | 全景分层 mesh | ✅ 全开源 |
| **Spatia** | 2025 / — | 视频 → 可更新 3D 场景 + 视频 | 3D 场景点云作持久空间记忆，SLAM 持续更新，动静态解耦 | RGB 点云（SLAM 维护） | — |
| **Matrix-3D** | 2025.08 / Skywork | 单图/文本 → 全景视频 → 3D 世界 | 轨迹引导全景视频扩散（场景网格渲染条件）+ 前馈 Panorama LRM 与优化 3DGS 双路 | 无（分段生成） | ✅ 开源 |
| **Lyra 1.0** | 2025.09 / NVIDIA（ICLR 2026） | 单图/视频 → 3DGS / 4D | 冻结 GEN3C 教师 latent 自蒸馏出 3DGS 解码器；Transformer+Mamba-2 混合块 | 无（见 2.0） | ✅ Apache 2.0 |
| **WorldMirror 1.0** | 2025.10 / 浙大·CUHK·腾讯混元（ICML 2026） | 多视图 ± 位姿/内参/深度 → 点图、相机、深度、法线、3DGS | 多模态先验 prompting（token 随机丢弃 0.5 训练）+ token merging，五头同出 | 无（单次前向） | ✅ 全开源 |
| **Lyra 2.0** | 2026.04 / NVIDIA | 单图 + 长轨迹 → 可探索 3DGS 世界 | per-frame 几何缓存只做路由（贪心检索 ≤5 帧、warp 规范坐标）+ self-augmentation 抗漂移；重建端用 DA3 | per-frame 3DGS 几何缓存 | ✅ Apache 2.0 |
| **HY-World 2.0** | 2026.04 / 腾讯混元 | 文本/单图/多视图/视频 → 3DGS 世界 | 四阶段：HY-Pano 2.0 → WorldNav 轨迹规划 → WorldStereo 2.0（关键帧生成+一致性记忆）→ WorldMirror 2.0 重建；WorldLens 渲染平台 | 全局几何记忆 + 立体记忆（中训练注入） | ✅ 全开源 |
| **VMem** | 2025 / — | 交互式视频世界模型 | surfel（面元）索引的视图记忆 | Surfels | — |
| **WorldPlay** | 2025 / — | 交互式世界模型，720p@24FPS | Reconstituted Context Memory + Context Forcing 蒸馏 | 重构上下文记忆 | — |
| **Mirage** | 2026.06 / 微软研究院·浙大·阿德莱德·Monash | 单图 + 轨迹 → 长程一致视频 | **Latent Spatial Memory**：3D latent token 直接投影到目标相机，免渲染-重编码 | 潜空间 3D 记忆（Initialize/Read/Denoise/Update） | 未放出 |
| **MosaicMem** | 2026.03 / — | 交互式视频世界模型 | Warped RoPE + Warped Latent 混合显隐记忆，patch 级世界坐标 | 混合式（显式坐标+隐式特征） | — |
| **Infinite-World** | 2026 / — | 交互式世界模型，1000 帧级 | Pose-free Hierarchical Memory Compressor 递归压缩历史 | 分层压缩记忆 | ✅ |
| **FLARE** | 2025.02 / 浙大·蚂蚁·Oxford·Stanford（CVPR 2025） | 2–8 张无位姿图 → 位姿 + 点图 + 3DGS | 级联学习：先回归粗位姿 → 相机系局部点图 → 全局投影 → 高斯外观头 | 无（0.5 秒一次前向） | ✅ |
| **AnySplat** | 2025.05 / CUHK·上海 AI Lab 等（ACM TOG） | 无约束视图集（稀疏或密集）→ 3DGS + 相机内外参 | 交替注意力（帧内 + 全局）+ 双 DPT 头；**可微体素化**把逐像素高斯聚成体素，撑住 32+ 密集视角 | 无 | ✅ |
| **Splatt3R** | 2024 / Oxford | 未标定图像对 → 3DGS | MASt3R 编码器 + 高斯参数头，双图一次前向直接出 .ply | 无（双视图） | ✅ |
| **MapAnything** | 2025.09 / Meta Reality Labs·CMU（3DV 2026） | 1–上千张图 ± 内参/位姿/深度/部分重建 → **度量尺度**几何 + 相机 | factored 表示（深度图 + 局部射线图 + 位姿 + 单一度量尺度因子），跨 13 个数据集统一监督 | 无（单次前向） | ✅ Apache 2.0 |
| **交互式世界模型族** | 2025–2026 / 多家 | 动作/轨迹 → 流式视频世界 | 记忆重构（WorldPlay）、分层压缩（Infinite-World）、轨迹打包（WorldPack）、循环状态（RELIC） | 压缩 / 循环记忆 | 部分 |
| **FastPano3D** | 2026.06 / — | 单张全景图 → 3DGS | ERP 感知的前馈高斯回归（自适应高斯采样 + 点云引导精修），秒级出场景 | 无 | — |
| Marble（参照） | 2025.11 / World Labs | 单图/全景 → 可探索 3D 世界 | 闭源，未公开技术细节 | 未知 | ❌ 闭源 |

---

## 3. 几何基座：WorldMirror 1.0 → 2.0

### 3.1 WorldMirror 1.0：any-prior prompting

详细精读见 [[WorldMirror - 多模态先验驱动的通用前馈三维重建]]，此处只放它在谱系里的位置：它是「条件注入」子方向里工程完成度最高的一个——内参、位姿、深度全部编码成 token（全局量走 MLP 出单个 token，稠密深度走 patch embedding 后**逐元素加**到 image token 上保持效率），训练时每个先验 token 以 0.5 概率随机丢弃，于是推理时「有多少用多少，没有也能跑」。共享 Transformer 后挂五个头：点图、相机、深度、法线、3DGS。无先验已领先 VGGT/π³，全先验下 7-Scenes 点图误差 0.043→0.018。

![WorldMirror 总体流水线：多模态先验 token 化 + 五头统一预测](assets/生成式世界模型路线的前馈三维重建%20-%20Survey/figures/wm_pipeline.png)

*WorldMirror pipeline（Figure 2）：左侧任意组合的可选先验 → 多模态 token merging → 共享 Transformer → 五个预测头（点图 / 相机 / 深度 / 法线 / 3DGS）。来源：[arXiv 2510.10726](https://arxiv.org/abs/2510.10726)*

它的一个很有说服力的消融是**先验数量的单调收益**：

![先验数量增加带来全任务一致提升](assets/生成式世界模型路线的前馈三维重建%20-%20Survey/figures/wm_num_prior.png)

*WorldMirror 注入先验数量的收益曲线（Figure 6）：先验越多，位姿、点图、深度、NVS 全部单调变好——「输入灵活性」与「多任务预测」互相成就而非互相拖累。来源同上*

### 3.2 WorldMirror 2.0：混元全家桶的重建底座

HY-World 2.0 技术报告（arXiv 2604.14268）里，WorldMirror 升级为整个系统的第四阶段「World Composition」的重建引擎，三点架构改进值得注意：归一化位置编码（Normalized Position Encoding）、深度估计的显式法线监督、深度掩码预测（Depth Mask Prediction）；另配合数据与推理效率改进，支持从多视图图像或**直接从视频**做世界重建。更重要的是定位变化：1.0 是独立论文，2.0 变成了「生成侧出内容、WorldMirror 出几何」这条 2026 年事实标准分工里的几何端（Lyra 2.0 用 DA3，混元用自己的 WorldMirror，选择不同、分工相同）。

---

## 4. 视频扩散 + 3D 缓存：记忆机制的演化主线

这一支的共同骨架是：**维护一个显式 3D 缓存 → 沿新轨迹渲染成 2D 条件 → 视频扩散模型只负责填洞和推进**。分歧全在「缓存里存什么」。

### 4.1 GEN3C：RGB 点云缓存的定义者

GEN3C（NVIDIA / 多伦多，arXiv 2503.03751）把这件事的形式定下来了：对每张输入/已生成帧估计深度并反投影成点云，组成时空 3D 缓存；沿用户轨迹渲染成 partial RGB + 遮挡掩码，通道维拼进加噪 latent 里微调视频扩散模型。三个设计选择后来被反复引用：

- **融合不引入新参数**：多视角渲染结果分别过扩散模型第一层后 max-pooling，避免「新参数泛化不如原模型」；
- **掩码用逐元素相乘**而非拼接，保持模型架构不变；
- **自回归更新缓存**：新生成帧估计深度、对齐后拼回缓存，长视频分段生成。

A100 上 14 帧约 30 秒。局限也很典型：动态内容依赖输入视频提供运动，缓存对深度误差敏感。

### 4.2 Voyager：把「深度」搬进生成端

Voyager（哈工大/东南/腾讯混元，arXiv 2506.04225）的判断是：GEN3C 式的 partial RGB 条件在复杂遮挡下伪影严重，而 **partial depth 恰好能准确表达遮挡**。于是它让模型**联合生成对齐的 RGB 与 Depth 序列**（两者沿高度维拼接、中间加分隔行），是首个按给定轨迹同时出 RGB+Depth 的视频模型；再配 Control Blocks（从 DiT 早期层复制、逐层注入输出）强化几何约束。缓存端做 normal-check 点剔除，采样端用重叠平滑采样缝合片段。配套的数据引擎自动产出位姿+度量深度标注（十万级视频对），免人工 3D 标注。WorldScore 静态基准上超过 WonderJourney、WonderWorld 与一众纯视频模型。

![Voyager 方法图：world cache 渲染 RGB+Depth 条件，Control Blocks 注入 DiT](assets/生成式世界模型路线的前馈三维重建%20-%20Survey/figures/voyager_method.png)

*Voyager pipeline（Figure 3）：单图估深反投影成 world cache → 沿轨迹渲染 partial RGB/Depth → 联合 RGB-D 视频扩散（Control Blocks 逐层注入）→ 生成帧更新缓存。来源：[arXiv 2506.04225](https://arxiv.org/abs/2506.04225)*

### 4.3 Mirage：把记忆搬进潜空间

GEN3C/Voyager/Spatia 这类「RGB 点云缓存」方案有一个共同的隐性税：每生成一段都要把点云渲染成像素、再重新编码进模型——Mirage（微软研究院等，2026.06）给这个循环算了账：**比潜空间记忆慢 10.57×、多吃 55× 显存**。它的替代方案是 Latent Spatial Memory：直接保存扩散模型内部特征并给每个特征配 3D 坐标，生成时把记忆投影到目标相机、免像素往返；Initialize→Read→Denoise→Update 的记忆生命周期跨 chunk 共享，写入前过滤动态物体与天空。效果：WorldScore 平均 70.36（Voyager 66.08、Spatia 69.73、WonderJourney 54.19），闭环一致性测试（相机绕一圈回到起点）SSIM 0.825 大幅领先。代价：动态内容在段边界被直接丢弃。

### 4.4 记忆机制横评

| 记忆形态 | 代表 | 优点 | 缺点 |
|---|---|---|---|
| RGB 点云缓存（渲染-重编码） | GEN3C、Voyager、Spatia、WonderWorld | 几何直观、可用真深度校准 | 深度误差传导；每步渲染+重编码开销（慢 10×、显存 55×） |
| 潜空间 3D 记忆 | Mirage | 免像素往返，长 rollout 显存近乎平坦 | 动态物体难保留；绑定特定模型内部特征 |
| 混合显隐（Warped RoPE/Latent） | MosaicMem | 一致性最好（RotErr 0.51 vs CaM 4.65） | 依赖预训练深度/位姿估计器 |
| 压缩/分层记忆 | Infinite-World、WorldPack、RELIC | 千帧级 horizon | 有损压缩，细节丢失 |
| Surfels / 重构上下文 | VMem、WorldPlay | 检索效率高 | 面元表示粗糙 |

一条值得记住的判据：**闭环一致性（走一圈回来像不像）是区分这些记忆机制的最狠指标**——插值质量大家都能做，回得去才是记忆。

---

## 5. Latent → 3DGS 直接回归：Wonderland、Matrix-3D 与 Lyra

这一支回答的问题是：**生成内容能不能不经过「多视图 → SfM/MVS → 优化 3DGS」这条老路，直接从扩散模型的 latent 一次前向变成 3DGS？**

### 5.1 Wonderland：第一个证明

Wonderland（Snap / 多伦多 / UCLA，arXiv 2412.12091，CVPR 2025）是这条路的定义者，两段式：

1. **相机引导视频扩散**：CogVideoX5B-I2V 基座 + 双分支相机控制（ControlNet 吃 Plücker 射线、LoRA 吃相机 token），把单图沿轨迹铺成 3D 一致的视频 latent；
2. **LaLRM（Latent Large Reconstruction Model）**：3D patchify + 轻量 Transformer + 3D 反卷积，从 video latent 逐 latent token 回归像素对齐高斯（位置/协方差/颜色/不透明度/射线距离），渐进式训练（低分辨率基准数据预训练 + 高分辨率自生成数据微调）。

单张 A100 端到端约 5 分钟（CAT3D 16 分钟、ViewCrafter 6+ 分钟、ZeroNVS 3 小时），RE10K 上 FID 16.16、重建 LPIPS 0.292 全面领先。它第一次证明「重建模型可以有效地建在扩散模型的 latent 空间上」——这句话后来被 Lyra 和 Matrix-3D 反复兑现。

![Wonderland 双分支相机控制 + LaLRM 从视频 latent 直接回归 3DGS](assets/生成式世界模型路线的前馈三维重建%20-%20Survey/figures/wonderland_pipeline.png)

*Wonderland pipeline（Figure 2）：单图 + 噪声过相机引导 DiT（Cam-LoRA + VisCam token）生成视频 latent → LaLRM 3D patchify 后一次前向出 3DGS。来源：[arXiv 2412.12091](https://arxiv.org/abs/2412.12091)*

### 5.2 Matrix-3D：全景表示 + 双路重建

Matrix-3D（Skywork，arXiv 2508.08086）把中间表示从透视图像换成**全景图**，范围问题（生成场景只有一小块）被 360°×180° 覆盖直接绕开：轨迹引导的全景视频扩散以**场景网格渲染**为条件（比点云条件少摩尔纹、几何更一致）；把全景视频抬到 3D 时给了两条独立路径——**前馈 Panorama LRM**（10 秒出粗场景）与**优化 3DGS 流水线**（571–745 秒，PSNR 27.62 vs 前馈的快速路）。配套 Matrix-Pano：首个 116K 带深度与轨迹标注的大规模合成全景视频数据集（UE5 渲染）。这个「快前馈 + 慢优化」双路设计对工程落地很友好。

![Matrix-3D：场景网格渲染条件 + 全景 LRM 前馈重建](assets/生成式世界模型路线的前馈三维重建%20-%20Survey/figures/matrix3d_overview.png)

*Matrix-3D pipeline（Figure 3）：场景网格渲染与掩码经 3D causal encoder / 下采样后与噪声 latent 通道拼接 → 视频扩散 Transformer → Panorama LRM 前馈出全向可探索 3D 场景。来源：[arXiv 2508.08086](https://arxiv.org/abs/2508.08086)*

### 5.3 Lyra：自蒸馏的集大成者

Lyra 1.0/2.0 的精读见 [[Lyra系列 - 生成式三维场景重建]]。放在这条线的坐标里：Wonderland 证明了「latent → 3DGS」可行，Lyra 用**自蒸馏**把「教师」从外部多视图数据换成了冻结的视频扩散模型自身（教师生成 → 学生在 latent 空间模仿教师的 RGB 解码），训练数据全合成；2.0 再补上长程记忆（几何只做路由 + self-augmentation），成为这条线 2026 年的长程天花板。

### 5.4 前优化时代的对照

理解这一支的分量，需要知道它取代了什么：CAT3D（多视图扩散 → NiER 优化）、ViewCrafter（点云条件视频扩散 → 优化）、ReconX——都是「生成视频/视图，再做 per-scene 优化」的两段式。Wonderland/Lyra/Matrix-3D 的共同贡献是把第二段也前馈化了，端到端时间从小时级压到分钟级、甚至秒级。

---

## 6. 工程路线：全景 / 分层世界生成（可导出资产）

### 6.1 HunyuanWorld 1.0：语义分层 mesh

混元的第一个世界生成系统（arXiv 2507.21809）走的是与视频扩散派不同的工程路线：以**全景图作 360° 世界代理**，agentic 流程做实例识别→层分解→层补全，再分层重建出**语义分层的 3D mesh**（前景物体独立、背景层、天空层）。三个卖点都冲着工业界：可导出 mesh（进图形学管线）、物体解耦（可交互）、360° 沉浸。代价是全景代理的几何精度有限、探索范围受全景分辨率约束。

### 6.2 HY-World 2.0：四阶段全家桶

2.0（arXiv 2604.14268）把 1.0 的「全景→mesh」升级为四阶段流水线，并直接对标闭源 Marble：

![HY-World 2.0 四阶段总览](assets/生成式世界模型路线的前馈三维重建%20-%20Survey/figures/hy2_overview.png)

*HY-World 2.0 overview（Figure 2）：Stage 1 全景生成（HY-Pano 2.0）→ Stage 2 轨迹规划（WorldNav，含场景几何/语义解析）→ Stage 3 世界扩展（WorldStereo 2.0 关键帧生成 + 更新记忆）→ Stage 4 世界组装（WorldMirror 2.0 重建 + Gaussian Splatting）。来源：[arXiv 2604.14268](https://arxiv.org/abs/2604.14268)*

几个关键设计：

- **WorldStereo 2.0 的一致性记忆**：Global-Geometric Memory（全局几何）+ Improved Spatial-Stereo Memory（立体记忆）在中训练阶段注入，配合记忆增强，解决关键帧生成阶段的漂移——这是混元版的「空间遗忘」对策；
- **WorldMirror 2.0** 承担重建（见第 3.2 节），点云扩展后深度对齐、再做 3DGS；
- **WorldLens**：自研 3DGS 渲染平台（引擎无关、自动 IBL 光照、碰撞检测、训练-渲染协同设计），负责「能玩」这最后一公里。

开源程度上混元这次相当彻底：1.0、Voyager、WorldMirror、2.0 的代码与权重全放出——在 DUSt3R/MASt3R 系非商用授权占主流的背景下，这是目前**授权上最能直接进产品管线的一整套世界生成方案**。

### 6.3 同路其他

WonderJourney / WonderWorld（斯坦福，全景链式扩展，WorldScore 上已被 Voyager/Mirage 明显超越）、WorldExplorer、PanoDreamer、ScenePainter 等属于同一全景家族，此处不展开。

---

## 7. 三条路线怎么拼：一张图看懂 2026 年的分工

- **生成侧**（编内容）：视频扩散模型——GEN3C / WorldStereo / Voyager / Lyra 的教师；
- **几何侧**（落几何）：前馈几何基础模型——DA3（Lyra 2.0 用）、WorldMirror（混元用）、VGGT（Voyager 的对比基线与初始化）、MapAnything（要度量尺度时的首选）；
- **记忆侧**（不忘记）：点云缓存（GEN3C/Voyager）→ 潜空间记忆（Mirage）→ 几何路由（Lyra 2.0）→ 全局几何记忆（WorldStereo 2.0）。

$$
\text{单图/文本} \xrightarrow{\text{视频扩散}} \text{多视图内容} \xrightarrow{\text{前馈几何}} \text{3DGS / 点云} \xrightarrow{\text{记忆机制}} \text{可探索世界}
$$

---

## 8. 选型指南

| 需求 | 首选 | 备选 | 提醒 |
|---|---|---|---|
| 有部分先验（内参/位姿/深度）的高精度重建 | MapAnything（要度量尺度）/ WorldMirror 1.0（要多任务输出） | Pi3X、Pow3R | 先验越全收益越大；商用选 MapAnything 的 Apache 2.0 版本 |
| 稀疏视角（2–8 张无标定）快速出位姿+几何+外观 | FLARE（0.5s） | MV-DUSt3R+、AnySplat | 这一档是 VGGT/WorldMirror 的弱项 |
| 密集/随便拍视角集 → 可实时渲染的 3DGS | AnySplat | MVSplat、DepthSplat、Splatt3R | 32+ 视角一定要有体素化之类的基元压缩 |
| 单图 → 可探索世界（要开源可商用） | Voyager / HY-World 2.0 | Matrix-3D、Lyra 1.0 | 混元系与 Lyra 均开放权重；注意各仓具体 license |
| 长程闭环一致性（走一圈回得来） | Mirage（未开源）/ Lyra 2.0 | Spatia | 看 closed-loop SSIM，别只看单段 PSNR |
| 要导出 mesh 进引擎/仿真 | HunyuanWorld 1.0 / 2.0 | Matrix-3D（优化路） | 视频派只出 3DGS，需再转 mesh |
| 速度优先的粗重建 | Matrix-3D 前馈路（10s） | Wonderland（5min） | 前馈快但质量低于优化路 |
| 交互式实时世界（动作条件、流式） | WorldPlay / Infinite-World | RELIC、WorldPack、Context as Memory | 千帧级只有 Infinite-World 开源 |
| 单张全景图 → 3DGS | FastPano3D（秒级） | Pano2Room（优化式，156× 慢） | ERP 畸变要专门处理 |
| 闭源能力参照 | Marble（World Labs） | Genie 3（DeepMind） | HY-World 2.0 报告称已可比 Marble |

**授权提醒**：本篇涉及的开源工作（混元全系、Lyra、Matrix-3D、MapAnything Apache 版、Infinite-World）授权相对友好，但逐仓核实 license 再商用仍是硬规矩；GEN3C、Wonderland、Mirage 目前不能作为完整开源方案依赖。

---

## 9. 局限与判断

### 9.1 共同局限

**动态内容仍是软肋。** Mirage 直接在写入记忆前丢弃动态物体；GEN3C 的动态能力依赖外部视频提供运动。生成式世界目前本质上是「静态世界 + 表面动态」。

**记忆的可扩展性没人真正解决。** 点云缓存随场景无限膨胀（靠剔除缓解）、潜空间记忆绑定模型内部特征、压缩记忆有损。1000 帧以上目前只有 Infinite-World 这类压缩派在硬扛。

**评测仍然分裂。** WorldScore 正在成为世界生成的事实基准（可控性/一致性/质量/运动四维），但几何精度与视觉质量在多数论文里仍然各报各的，横向选型困难依旧。

### 9.2 个人判断

其一，**「渲染-重编码」这条路正在被淘汰**。Mirage 的 10×/55× 数据基本宣判了「每步把 3D 缓存渲成像素再喂回模型」的死刑，记忆的归宿在潜空间或路由机制，不在像素。

其二，**生成侧与几何侧的分工已经定型**，且两侧都在用别家的最好的：NVIDIA 的 Lyra 2.0 用字节 DA3 落几何，腾讯混元用自己的 WorldMirror——几何基础模型成了所有生成式世界系统的公共底座，这与 [[前馈式三维重建 - 分支谱系总览]] 里「专用架构贬值」的判断互为印证。

其三，**闭环一致性是下一个主战场**。WorldScore 上的静态分大家都涨得快，closed-loop 指标的差距（Mirage 0.825 vs Spatia 0.579）说明「回得去」仍是稀缺能力，谁先把闭环做成默认能力，谁拿下可玩性。

其四，**前馈几何的 2026 年共识是「度量尺度 + 可插拔先验」**。MapAnything 与 WorldMirror 是两支独立团队、两种不同解法（factored 尺度因子 vs 先验 token），却给出了同一个单调性结论：先验注入越全、几何越准。这意味着「无先验也能跑」只是及格线，「有多少先验就用多少」才是竞争点；同时 MapAnything 用 factored 表示直接输出度量尺度，把「尺度歧义」这个问题从后处理挪进了网络里——这是 2026 年最值得跟进的一个接口变化。

---

## 10. 补遗：容易漏掉但很关键的前馈工作

前面四支讲的全是「有记忆、能生成世界」的系统。这一节补齐三类平时容易被归错类、但在工程上极常遇到的前馈工作。

### 10.1 几何基座补遗：MapAnything、FLARE

**MapAnything**（Meta Reality Labs / CMU，arXiv 2509.13414，**3DV 2026**）是 2025 下半年最该被记住的前馈几何工作，也是 WorldMirror 的直接竞品。它的核心是一个 **factored（因式分解）的多视几何表示**：不做全局点图，而是输出「每张图的深度图 + 局部射线图 + 相机位姿 + **一个全局度量尺度因子**」。这个尺度因子是把局部重建升级成全局一致度量坐标系的关键——比 VGGT/π³ 的「点图 + 事后对齐」干净，也比 WorldMirror 的「先验 token 注入」更结构化。配合跨 13 个数据集的标准化监督与灵活的输入增强，它在 2–100 视角、甚至单图上都是 SOTA，同时覆盖免内参 SfM、标定 MVS、单目深度、相机定位、深度补全等一整排任务。两视图配置下相对误差随先验注入从 0.18（纯图像）一路降到 0.01（图像+内参+位姿+深度），这个单调性与 WorldMirror 的结论完全一致——**两个独立团队在 2025 年下半年得出了同一个结论：先验可插拔是前馈几何的必答题**。授权上它给出 Apache 2.0（6 个数据集训练）与 CC BY-NC 4.0（+7 个数据集）两个版本，商用要选前者。

![MapAnything 方法图](assets/生成式世界模型路线的前馈三维重建%20-%20Survey/figures/mapanything_method.png)

*MapAnything（Figure 2/方法图）：图像与可选几何输入统一 token 化 → 共享 Transformer → factored 输出（深度图、局部射线、位姿、度量尺度因子），一次前向覆盖 SfM / MVS / 单目深度 / 定位 / 补全。来源：[arXiv 2509.13414](https://arxiv.org/abs/2509.13414)*

**FLARE**（浙大 / 蚂蚁 / Oxford / Stanford，arXiv 2502.12138，CVPR 2025）代表了另一条思路：**级联而不是并行**。它认为位姿是把 3D 结构映射到 2D 平面的桥梁，于是把难题拆成四步——神经位姿预测器先出粗位姿 → 在各相机坐标系下预测局部点图 → 可学习几何投影器统一到全局 → 3DGS 头出外观与渲染。2–8 张无标定图、0.5 秒内出位姿+几何+高斯。它的工程价值在于「稀疏视角」（2–8 张）这个真实最常见的输入档位，正是 VGGT/WorldMirror 这类动辄 8+ 视角训练的模型容易掉链子的地方。

同支还有 **MV-DUSt3R+**（稀疏视角 2 秒单阶段重建）、**Reloc3R / MASt3R-SfM**（定位与非商用 COLMAP 替代，已见 [[前馈式三维重建 - 分支谱系总览]]）。

### 10.2 前馈 3DGS 补遗：AnySplat、Splatt3R

**AnySplat**（CUHK / 上海 AI Lab 等，arXiv 2505.23716，ACM TOG）解决了前馈 3DGS 一个很实际的问题：**密集视角下逐像素出高斯会爆**。它的做法是交替注意力（帧内注意力 + 全局注意力交替）配相机 token 与图像 token，深度头出深度再反投影成高斯中心、高斯头出不透明度/朝向/尺度/球谐；关键是引入**可微体素化**——像素级高斯按预测的置信度 softmax 加权聚合成体素，把基元数量压下来，从而从稀疏视角（2–16 张）扩展到 32+ 视角的随便拍场景，一次前向同时出 3DGS 与内外参。零样本评测下匹配「有位姿」基线的质量、明显超过免位姿方法，并把渲染延迟压到实时量级。

![AnySplat 流水线](assets/生成式世界模型路线的前馈三维重建%20-%20Survey/figures/anysplat_pipeline.png)

*AnySplat pipeline：交替注意力 Transformer → 深度头 + 高斯头（DPT + 浅层 CNN 外观特征）→ 可微体素化聚合 → 一次前向输出全场景高斯与相机。来源：[arXiv 2505.23716](https://arxiv.org/abs/2505.23716)*

**Splatt3R**（Oxford，2024）是这条线更早也更极简的一支：MASt3R 编码器 + 高斯参数头，未标定图像对一次前向直接出 .ply，可以理解为「把 MASt3R 的点图换成高斯」。它证明了 **pointmap 基座 + 高斯头**这条组合路在 2024 年就成立——后来 DA3 的「冻结骨干 + DPT 头」结论与之同源。同支的 Flash3D、LatentSplat、MonoSplat、SparSplat 各有侧重（单图 / 潜空间自编码 / 单目深度监督 / 2DGS 加速），选型时按输入档位挑即可。

### 10.3 前馈世界模型补遗：交互式实时派与单图全景派

这一小节补的是严格意义上「前馈世界模型」里最贴近交互与落地的两族——它们的目标不是出一张漂亮的图，而是**能一直走下去**。

**交互式 / 实时派**（2025–2026 集中爆发）

| 工作 | 记忆方案 | 关键主张 |
|---|---|---|
| **WorldPlay** | Reconstituted Context Memory + Context Forcing 蒸馏 | 720p@24FPS 流式生成，重建可用上下文同时保留关键几何帧 |
| **Infinite-World** | Pose-free 分层记忆压缩器（HPMC）递归压缩 + 不确定性感知动作标注 + 回访密集微调 | 把交互世界模型推到 **1000 帧 horizon**，激活 loop-closure；开源 |
| **WorldPack** | 轨迹打包 + 记忆检索的压缩记忆 | LoopNav 基准上显著超过基线 |
| **RELIC** | 高压缩历史 latent token（带相对动作 + 绝对位姿）进 KV cache | 循环式长程记忆 |
| **SEVA / CaM** | 显式（全历史帧 + 时间感知 PE warp）vs 隐式（隐状态） | MosaicMem 报告里作为两类上限基线：SEVA RotErr 1.42、CaM 4.65 |
| **Context as Memory** | 直接存历史帧 + 按相机 FOV 重叠检索 | 极简但有效，靠检索过滤控制计算爆炸 |
| **VMem / WorldMem** | surfel 索引 / 长程一致世界模拟 | 面向 revisit（回访）任务 |

与第 4 节的生成派相比，这一族的共同点是**动作条件 + 固定预算记忆 + 实时约束**——它们不追求「从单图长出一个世界」，而是「用户持续输入时世界不崩」。

**单图全景派**：FastPano3D（arXiv 2606.30352，2026.06）从单张全景图前馈回归 3DGS，专门处理 ERP 等距柱状投影的畸变与特征分布不均（自适应高斯采样 + 点云引导精修），秒级出场景、比 Pano2Room 快 **156×**、参数减半。同族的 Pano2Room、DreamScene360、LayerPano3D 属于优化式前辈，如今基本被前馈方案取代。

**闭源参照系**：World Labs 的 **Marble**（2025.11）与 DeepMind 的 **Genie 3**——前者是可探索 3D 世界生成的商业标杆（HY-World 2.0 报告称已可比），后者是实时交互世界模型的标杆。两者都不公开技术细节，但对标意义在于：**它们定义了「够好」的下限**。

---

## 11. 关键文献索引

| 工作 | 链接 |
|---|---|
| WorldMirror 1.0 | [arXiv 2510.10726](https://arxiv.org/abs/2510.10726) · [GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) |
| HY-World 2.0 | [arXiv 2604.14268](https://arxiv.org/abs/2604.14268) · [GitHub](https://github.com/Tencent-Hunyuan/HY-World-2.0) |
| Voyager | [arXiv 2506.04225](https://arxiv.org/abs/2506.04225) · [GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager) |
| HunyuanWorld 1.0 | [arXiv 2507.21809](https://arxiv.org/abs/2507.21809) · [GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) |
| GEN3C | [arXiv 2503.03751](https://arxiv.org/abs/2503.03751) · [项目页](https://research.nvidia.com/labs/toronto-ai/GEN3C/) |
| Wonderland | [arXiv 2412.12091](https://arxiv.org/abs/2412.12091) · [项目页](https://snap-research.github.io/wonderland/) |
| Matrix-3D | [arXiv 2508.08086](https://arxiv.org/abs/2508.08086) · [GitHub](https://github.com/SkyworkAI/Matrix-3D) |
| Mirage（Latent Spatial Memory） | arXiv 2026.06（微软研究院，代码未放出） |
| MapAnything | [arXiv 2509.13414](https://arxiv.org/abs/2509.13414) · [项目页](https://map-anything.github.io/) |
| AnySplat | [arXiv 2505.23716](https://arxiv.org/abs/2505.23716) · [项目页](https://city-super.github.io/anysplat/) |
| FLARE | [arXiv 2502.12138](https://arxiv.org/abs/2502.12138) · [项目页](https://zhanghe3z.github.io/FLARE/) |
| Splatt3R | [GitHub](https://github.com/btsmart/splatt3r) |
| FastPano3D | arXiv 2606.30352（2026.06） |
| WorldScore 记忆家族综述 | [Awesome World Models with Memory](https://github.com/SiriYep/Awesome-World-Models-with-Memory) |

本馆内链：[[Lyra系列 - 生成式三维场景重建]] · [[WorldMirror - 多模态先验驱动的通用前馈三维重建]] · [[VGGT - 前馈式通用三维重建]] · [[DA3 - 任意视角恢复视觉空间]] · [[前馈式三维重建 - 分支谱系总览]] · [[前馈式三维重建与生成式补全 - Survey]]
