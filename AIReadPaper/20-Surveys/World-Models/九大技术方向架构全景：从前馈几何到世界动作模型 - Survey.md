---
type: survey
status: complete
domain: World-Models
title: 九大技术方向架构全景：从前馈几何到世界动作模型
created: 2026-10-26
updated: 2026-10-26
reviewed_until: 2026-10-26
tags:
  - 3D-Reconstruction
  - Depth-Estimation
  - Video-Generation
  - World-Models
  - JEPA
  - VLA
  - WAM
---

# 九大技术方向架构全景：从前馈几何到世界动作模型

## 基本信息

- **覆盖方向**：前馈 3D 重建、单目深度估计、广义深度估计、视频生成模型、视频世界模型、3D/4D 世界模型、JEPA、具身智能 VLA、世界动作模型 WAM。
- **资料截止**：2026-10-26。
- **组织原则**：不按公司或模型名堆清单，而按「输入证据—内部状态—信息流—训练目标—输出接口—是否进入动作闭环」重排。
- **证据优先级**：正式论文 / arXiv / 官方项目页 / 官方代码仓库 > 官方技术博客 > 二手报道。闭源系统只写官方明确披露内容，未知项不反推。
- **关联精读**：[[前馈式三维重建 - 分支谱系总览]]、[[从 Depth Anything 到 DA3：前馈深度与三维几何 - Survey]]、[[视频生成模型技术谱系与2026现状 - Survey]]、[[世界模型演化 - Survey]]、[[JEPA研究进展 - Survey]]、[[VLA 模型架构与数据演化 - Survey]]、[[Genie 1-3 架构演进 - Survey]]、[[Lyra与持久可探索三维世界 - Survey]]。

> **一句话总览**：九个方向看似分散，实际围绕同一个问题逐层展开——模型如何从视觉观测中形成可复用状态，如何预测状态变化，以及预测是否真正改变动作选择。

---

## 1. 全局地图：九个方向不是九条平行赛道

![](assets/九大技术方向架构全景：从前馈几何到世界动作模型%20-%20Survey/figures/01-global-landscape.svg)

> 图 1：九个方向可压缩为三层：几何感知层负责恢复空间状态，生成与世界模拟层负责预测未来，预测与行动闭环层负责把未来用于规划或控制。图为本综述自绘。

### 1.1 三层关系

第一层是**几何感知**。单目深度解决最欠约束的单图几何；Stereo、MVS、视频深度和 depth completion 逐步加入双目基线、多视角、时间和稀疏传感器证据；前馈 3D 重建则把深度、相机、pointmap、track、3DGS 等输出统一进一个模型。它们共同回答：**当前世界是什么形状？相机在哪里？尺度可靠吗？**

第二层是**生成与世界模拟**。视频生成模型学习视觉序列分布；视频世界模型进一步要求历史和动作能够控制未来；3D/4D 世界模型则要求模型维护可回访、可查询、可碰撞或至少空间一致的持久状态。它们回答：**世界接下来可能发生什么？**

第三层是**预测与行动闭环**。JEPA 主张在 embedding 空间预测可预测部分；VLA 直接把视觉和语言映射到动作；WAM 则要求未来预测真正进入动作生成、动作评分或规划。它们回答：**为了完成任务，应该做什么？**

### 1.2 一个模型可同时属于多个家族

架构家族不是互斥标签。例如：

- DA3 既是单目深度模型，也是任意视图前馈几何模型；
- CUT3R 既是前馈重建，也是有状态的视频几何模型；
- Genie 既属于动作条件视频世界模型，也属于交互式生成模型；
- V-JEPA 2 既是 JEPA，也能在动作后训练后成为潜空间世界模型；
- DreamZero 同时落入视频世界模型、VLA 和联合式 WAM；
- Lyra 2.0 同时使用视频生成、显式几何记忆和前馈 3DGS 编译。

因此，下文统计“多少种结构”时采用**设计轴上的架构家族数**，而不是假装每篇论文只能落到一个盒子里。

### 1.3 九方向的结构数量概览

| 方向 | 可稳定辨认的主架构家族 | 2026 年主线 |
|---|---:|---|
| 前馈 3D 重建 | 6 类输出表示 × 5 类信息流 | pointmap/depth-ray 几何基础模型 + 可选几何条件 + 流式状态 |
| 单目深度估计 | 6 类 | relative foundation model → camera-aware metric pointmap |
| 广义深度估计 | 5 个任务分支、约 7 类机制 | 单目基础先验 + 显式对应 / 投影 / 传感器尺度 |
| 视频生成模型 | 8 类 | 3D causal VAE + 时空 DiT + diffusion/flow + 稀疏化/蒸馏 |
| 视频世界模型 | 7 类 | 动作条件 causal video model + 记忆 + 实时流式 |
| 3D/4D 世界模型 | 8 类 | 生成先验 + 显式几何寻址 + 持久 3D/4D 状态 |
| JEPA | 7 类 | 表征预测 → 动作条件 latent dynamics → 联合世界—动作 |
| VLA | 6 类动作解码 × 5 类融合方式 | 低频语义系统 + 高频连续 action expert + action chunk |
| WAM | 3 类耦合 × 3 类中间状态 | 训练期预测性表征 + 测试时策略/评分/规划按需组合 |

---

## 2. 前馈 3D 重建：从任务管线到几何基础模型

### 2.1 任务边界

前馈 3D 重建指模型对一张或多张图像做一次或有限次共享网络传播，直接输出深度、相机、pointmap、3DGS、NeRF、mesh、occupancy 等三维状态；它不再为每个新场景从零长时间优化大量自由变量。

需要区分四个口径：

1. **严格单次前馈**：VGGT、Fast3R、DA3；
2. **pairwise 前馈 + 全局对齐**：DUSt3R、MASt3R；
3. **共享网络反复更新状态**：CUT3R、UniT；
4. **前馈初始化 + 可选 BA / PnP / 融合**：工程上最稳健，也最可能落地。

### 2.2 六类输出表示

| 家族 | 核心表示 | 代表作 | 代表性贡献 | 主要问题 |
|---|---|---|---|---|
| Pointmap 几何 | 每像素三维点 | [DUSt3R](https://arxiv.org/abs/2312.14132)、[MASt3R](https://arxiv.org/abs/2406.09756)、[VGGT](https://vgg-t.github.io/)、[π³](https://yyfz.github.io/pi3/) | 将匹配、深度、相机恢复压成统一稠密几何回归；VGGT 推进到全视图联合预测；π³ 去掉固定参考帧 | 坐标规范、尺度、参考帧偏置与跨头一致性 |
| Depth-ray 因子化 | 深度 + 射线 / 相机 | [DA3](https://depth-anything-3.github.io/)、MapAnything | 把几何和相机解耦，以更少目标统一单图到多视图 | 射线、深度与 metric scale 如何稳定联合标定 |
| 前馈 3DGS | Gaussian 位置、尺度、颜色、透明度 | [pixelSplat](https://davidcharatan.com/pixelsplat)、[MVSplat](https://donydchen.github.io/mvsplat/)、[NoPoSplat](https://noposplat.github.io/)、[DepthSplat](https://haofeixu.github.io/depthsplat/) | 预测后立即可渲染；将单目先验、cost volume 与可微渲染融合 | 渲染好不等于表面准；透明度可掩盖几何错误 |
| NeRF / triplane / LRM | 连续场或三平面 | pixelNeRF、[LRM](https://arxiv.org/abs/2311.04400)、TripoSR | 从跨场景 NeRF 发展到单图对象级大重建模型 | 不可见面来自生成先验；更像资产补全而非测量 |
| Surface / occupancy | mesh、SDF、voxel | InstantMesh 等 | 直接输出下游可编辑、可碰撞表面 | 拓扑、薄结构、非流形和离散化误差 |
| 3D-free 视图合成 | target-view latent / light field | 部分 NVS Transformer | 不显式构建完整 3D，直接预测目标视角 | 难导出可测量、可规划的持久几何 |

### 2.3 五类信息流

| 信息流 | 代表作 | 优势 | 局限 |
|---|---|---|---|
| Pairwise cross-attention + 图优化 | DUSt3R、MASt3R、MASt3R-SfM | 两视图训练稳定，容易接入 SfM | 图像对可达 $O(N^2)$，全局对齐慢 |
| 全视图 all-to-all | Fast3R、VGGT、π³、DA3 | 在网络内部直接建立全局一致性 | 视频很长时 token 和全局注意力成本高 |
| Recurrent / persistent state | Spann3R、CUT3R、UniT | 在线、流式、内存可控 | 状态会遗忘，错误会被持续写回，回环困难 |
| Cost volume / plane sweep | MVSplat、DepthSplat、MVSNet 系 | 显式使用相机与对应，局部几何可靠 | 依赖相机、深度候选和较大显存 |
| Optional-condition model | MapAnything、OmniVGGT、WorldMirror、DA3 | 相机、深度、点图可作为任意组合输入 | 训练组合空间大，模型可能忽视辅助模态 |

### 2.4 目前在争什么

**第一，前馈能否取代 SfM / BA？** 目前更合理的答案是“替代初始化与粗重建，但尚未普遍取代高精度优化”。前馈模型擅长弱纹理、低重叠和未知相机；BA 在亚像素对应、闭环和严格测量中仍有优势。

**第二，pointmap 还是 depth-ray？** Pointmap 接口直接，但几何、相机和尺度纠缠；depth-ray 更可解释、更容易注入相机和深度先验。DA3 的强结果说明“少而充分的因子化目标”可能优于多头堆叠，但尚不能证明 pointmap 已过时。

**第三，是否需要固定参考帧？** 固定首帧简单，但输入顺序影响预测。π³ 推动置换等变；流式视频又天然存在时间方向。可能的终局是“组内无序或双向，组间因果”。

**第四，重建与生成如何划界？** 单图完整 3D 必然补造不可见区域。评价必须拆成观测区几何精度、未观测区合理性、渲染质量和可碰撞表面质量，不能用一个 PSNR 统治所有目标。

### 2.5 判断

前馈 3D 的主线已经从“神经网络替换某一步 SfM”转向“统一维护几何状态”。2026 年最值得盯的是：

- 相机、深度、ray、pointmap 的最小充分表示；
- 全视图 Transformer 与流式记忆如何合并；
- metric scale 与 invariant geometry 解耦；
- 网络置信度驱动的可选 BA；
- 重建 latent 是否可直接服务 VLA、空间检索和世界模型。

---

## 3. 单目深度估计：从一张标量图到相机感知三维几何

### 3.1 三种输出口径

1. **Relative / affine-invariant depth**：只保证相对前后关系或允许尺度—平移对齐，MiDaS、Depth Anything 属于这一主线；
2. **Metric depth**：输出米制深度，但绝对尺度来自训练统计、相机/FOV、物体尺度先验或外部锚点，并非单图投影几何天然可辨识；
3. **Monocular geometry / pointmap**：直接预测每像素三维点，同时隐式或显式恢复相机。

### 3.2 六类架构家族

| 家族 | 代表作 | 核心贡献 | 当前地位 |
|---|---|---|---|
| 监督式多尺度回归 | Eigen、DORN、AdaBins、DPT | 从连续回归发展到 ordinal / adaptive bins / ViT dense decoder | 奠基组件，已不是通用前沿主战场 |
| 光度自监督 | Monodepth、SfMLearner、Monodepth2、PackNet-SfM | 用双目或视频重投影替代密集深度标注 | 适合领域适配；受静态、Lambertian、遮挡和相机假设约束 |
| Relative depth foundation model | MiDaS、[Depth Anything](https://depth-anything.github.io/)、[DA V2](https://depth-anything-v2.github.io/) | 多数据集混训、海量伪标签和强 ViT backbone | 通用部署与下游特征主线 |
| Zero-shot metric depth | ZoeDepth、Metric3D、[UniDepth](https://lpiccinelli-eth.github.io/pub/unidepth/)、[Depth Pro](https://github.com/apple/ml-depth-pro) | canonical camera、FOV head、pseudo-spherical 表示 | 走向 camera-aware metric geometry |
| 单目 pointmap / geometry | UniDepth、[MoGe](https://github.com/microsoft/MoGe)、DA3 | 从标量深度升级为可反投影的三维状态 | 2026 年最强主线之一 |
| 生成式深度 | [Marigold](https://marigoldmonodepth.github.io/)、GeoWizard、[DepthFM](https://depthfm.github.io/) | diffusion / flow 提供边界、细节和后验多样性 | 高保真与不确定性支线，尚未取代判别式部署主线 |

### 3.3 目前在争什么

**Metric depth 是否真的“可恢复”？** 单图尺度在几何上不可辨识。模型输出米制深度，本质上是在训练分布下估计 $p(Z,K,\text{scale}\mid I)$。面对微缩模型、非常规镜头、新型机器人视角或医学图像时，语义尺度先验可能失效。

**Depth 还是 pointmap？** Depth 兼容传统系统、轻量；pointmap 更适合表达局部三维形状和相机耦合。depth-ray 因子化是折中接口。

**生成式还是判别式？** 判别式更快、更稳定、更容易 metric 化；生成式更适合不确定性、补全和 refinement。更可能的组合是判别式主干 + 生成式局部修复，而不是完全二选一。

**2D 指标是否够用？** 不够。薄杆、栅栏、透明边界在 AbsRel 上可能不错，反投影后却弯曲或粘连。未来应联合评估 depth boundary、pointmap、normal、novel-view reprojection 和真实测量误差。

### 3.4 判断

单目深度正在被重新定义为：

> **单图条件下的相机感知三维几何估计。**

其架构主线是强 ViT/DINO 表征、精确合成标签 + 真实伪标签、相机与尺度解耦、metric pointmap/depth-ray 以及可校准不确定性。

---

## 4. 广义深度估计：显式几何证据不会消失

![](assets/九大技术方向架构全景：从前馈几何到世界动作模型%20-%20Survey/figures/02-geometry-continuum.svg)

> 图 2：深度任务不是彼此割裂的 benchmark，而是沿观测证据逐渐增加的连续谱。图为本综述自绘。

### 4.1 五个任务分支

#### 4.1.1 Stereo

Stereo 用已知焦距 $f$、基线 $B$ 和视差 $d$ 得到 $Z=fB/d$。主结构从 3D cost volume 演化到 correlation pyramid + recurrent refinement，再叠加单目基础先验。代表作包括 RAFT-Stereo、CREStereo、IGEV-Stereo、[FoundationStereo](https://nvlabs.github.io/FoundationStereo/)。

当前共识不是“单目基础模型取代 stereo”，而是：

$$\text{monocular prior}+\text{epipolar correlation}+\text{known calibration}$$

单目先验解决无纹理、遮挡和跨域；极线几何提供可验证的物理尺度。

#### 4.1.2 MVS

MVS 的主结构是 plane-sweep / homography warping + cost volume regularization。演化路径为：

- MVSNet：3D cost volume + 3D U-Net；
- CasMVSNet：粗到细级联；
- PatchmatchNet：学习式传播与采样；
- MVSFormer / MVSFormer++：预训练 ViT 特征与 Transformer 正则；
- MVSAnywhere：自适应深度范围 + 单目/多视 cue 融合。

MVS 与未知相机前馈重建的关键区别是：MVS 把相机当已知条件，DUSt3R/VGGT/DA3 同时估计相机与结构。

#### 4.1.3 Temporal / Video Depth

逐帧跑单目模型不等于视频深度。视频深度还要求 scale/shift 稳定、动态物体正确、长视频不漂移、在线模式不访问未来。

两条路线并行：

- **生成式一致深度**：ChronoDepth、DepthCrafter、RollingDepth，借视频扩散先验获得时间一致性；
- **几何状态 / 记忆模型**：Video Depth Anything、CUT3R、UniT，维护可在线更新的几何状态。

前者擅长开放域和视觉一致性，后者更适合 metric 感知与机器人在线应用。

#### 4.1.4 Depth Completion

旧范式是“稀疏深度补洞 + RGB 引导”；新范式是“强单目稠密先验 + 稀疏 metric anchor”。代表作从 CSPN、NLSPN、CompletionFormer 发展到 Prompt Depth Anything、Marigold-DC、MapAnything。

#### 4.1.5 Geometry Foundation Model

真正统一的几何模型把任务差异变成输入配置差异：

$$\text{views}+\text{optional cameras}+\text{optional depths}\rightarrow\text{rays/depths/points/cameras/confidence}$$

DUSt3R、VGGT、CUT3R、DA3、MapAnything、OmniVGGT、UniT 是这一方向的代表。

### 4.2 目前在争什么

- **专家模型 vs 统一模型**：固定传感器和实时指标上专家模型仍强；未知域和缺失模态下统一模型更强。终局可能是统一 backbone + 专家 adapter/head。
- **离线 all-to-all vs 在线 memory**：离线模型全局一致但昂贵；在线模型高效但遗忘和漂移。组自回归是中间方案。
- **metric scale 来自哪里**：必须明确是基线、相机平移、速度/IMU、稀疏深度还是训练统计。只报告 scale-aligned 指标不能宣称 metric 能力。
- **显式几何是否过时**：没有。尤其工业测量、亚像素 stereo、MVS 和安全机器人仍需要可验证投影约束。

---

## 5. 视频生成模型：从运动模块到时空基础模型

### 5.1 八类生成架构

| 家族 | 代表作 | 核心贡献 | 关键局限 |
|---|---|---|---|
| 视频 GAN | TGAN、MoCoGAN、DVD-GAN | 内容/运动分解、3D 判别器、单次生成 | 训练不稳、模式坍缩、长时和文本条件扩展差 |
| 离散 token AR / masked | VideoGPT、Phenaki、MAGVIT-v2、VideoPoet、Emu3 | 将视频转成 token，复用语言模型或 MaskGIT 基础设施 | token 长、量化误差、串行延迟 |
| 像素扩散 | Video Diffusion Models、Imagen Video | 建立 3D U-Net 和级联视频扩散 | 计算成本高，主要是历史奠基 |
| 潜扩散 U-Net | Make-A-Video、Video LDM、AnimateDiff、SVD | 复用图像模型，仅增加时间层/运动模块 | 空间与时间人为分离，复杂运动受限 |
| 时空 DiT | Sora 报告、CogVideoX、HunyuanVideo、Wan、Movie Gen | spacetime latent patch + 大规模 Transformer | full attention 的 token 成本成为核心瓶颈 |
| Flow matching | Movie Gen、Wan、HunyuanVideo、LTX | 学习连续速度场，便于少步采样和蒸馏 | 效果仍强依赖 path、采样器、时间分布和 VAE |
| AR diffusion / Diffusion Forcing | Diffusion Forcing、SkyReels-V2、MAGI-1 | 块内并行去噪、块间因果，连接离线扩散与流式生成 | exposure bias、逐块漂移、缓存增长 |
| 因果蒸馏 / 实时流式 | CausVid、Self-Forcing、LongLive、Matrix-Game 2.0 | 块因果注意力、KV cache、少步蒸馏、自 rollout 训练 | 教师上限、模式收缩、硬件与系统强耦合 |

### 5.2 骨干的事实收敛

离线高质量开源模型逐渐收敛到：

```text
文本 / VLM 编码器
  → 3D causal VAE
  → spacetime latent patches
  → DiT（full / sparse 3D attention）
  → diffusion / flow matching
  → VAE decoder / super-resolution
```

实时交互模型则收敛到：

```text
双向多步教师
  → 少步蒸馏学生
  → block / frame causal attention
  → self-rollout / self-forcing
  → rolling KV cache
  → streaming VAE decoder
```

### 5.3 三个关键结构变量

**第一，视频 tokenizer / VAE。** 从逐帧 VAE 到 3D causal VAE，再到 LTX 式高压缩表示。压缩越强，DiT 越省，但小文字、接触、微表情和快速运动越容易丢失。VAE 的重建指标并不必然预测生成质量。

**第二，时空注意力。** 轴向/分解注意力便宜且易复用图像模型；full 3D attention 表达强但成本为 $O(n^2)$；稀疏、滑窗、径向和 tile attention 正从推理外挂变成原生骨干设计。

**第三，信息流。** 使用 causal VAE 不代表生成器因果。真正的流式生成必须让 DiT 的未来 token 不读取尚未生成内容，并处理训练—推理历史分布不一致。

### 5.4 目前在争什么

- **连续 latent diffusion/flow vs 离散 next-token**：前者画质和运动强，后者统一文本/图像/音频/动作接口更自然；短期不会一统。
- **full attention vs sparse attention**：质量基准仍常依赖 full attention，但长视频不可能无限支付二次成本。
- **高压缩 VAE 是解法还是转移成本**：它把 DiT 成本转给 decoder，也可能丢失最关键的物理短时事件。
- **长视频靠长上下文还是结构化记忆**：分钟级一致性本质是状态和误差管理，不只是窗口长度。
- **视频逼真是否等于物理理解**：目前没有。碰撞、接触、数量守恒、遮挡后身份和反事实动作仍普遍失败。
- **评测**：FVD、CLIP、VBench、VLM-as-judge、人评和 Elo 各自只覆盖局部，且可能奖励静止、审美或提示重写。

### 5.5 判断

视频生成的下一阶段竞争不再是“能否生成短视频”，而是 VAE、稀疏注意力、流式信息流、音画联合、动作控制、蒸馏和系统工程的共同优化。

---

## 6. 视频世界模型：从“生成未来”到“可干预的未来”

### 6.1 定义

视频世界模型建模：

$$p(o_{t+1:t+H}\mid o_{\le t},a_{t:t+H-1},c)$$

严格意义上，它至少应具备部分能力：动作干预、闭环逐步交互、对象持久性、因果/物理一致性，以及对规划或策略的实际效用。普通 T2V 只有视觉分布生成，不自动是世界模型。

### 6.2 七类架构

| 家族 | 状态与动力学 | 代表作 | 优势 / 局限 |
|---|---|---|---|
| RSSM / latent dynamics | 随机潜状态 + RNN/SSM | PlaNet、DreamerV3 | 快、适合规划；跨域视觉知识有限 |
| 离散 token AR | VQ token + Transformer | IRIS、iVideoGPT | 与 LLM 技术栈兼容；序列长、误差积累 |
| Masked token world model | 离散 token 并行填充 | Genie 1 | 可从无动作视频发现 latent action；因果闭环需额外设计 |
| 视频 diffusion / DiT simulator | 视频 latent 迭代去噪 | GameNGen、Cosmos | 画质和多模态条件强；延迟和控制频率受限 |
| Causal / streaming diffusion | 块或帧级因果去噪 | Matrix-Game、Yume、WorldPlay | 在线交互；长期漂移与缓存压力 |
| 显式记忆增强 | RGB、几何、surfel、检索 cache | CaM、SPMem、VMem | 改善重访；错误记忆会污染未来 |
| 世界基础模型平台 | 大规模视频基座 + 领域后训练 | Cosmos、Genie 2/3 | 可服务驾驶、机器人和游戏；通用物理能力尚未严格证明 |

### 6.3 历史主线

- **模型式 RL**：PlaNet / Dreamer 在紧凑 latent 中做 imagined rollout；
- **Transformer 世界模型**：IRIS 将世界建模重写为离散序列预测；
- **大型生成骨干**：GameNGen、Cosmos、Genie 将环境模拟扩到开放视觉世界；
- **实时交互**：因果扩散、Self-Forcing、KV cache 和少步蒸馏把输出推进到可玩；
- **持久状态**：记忆检索、几何 cache 和回环一致性成为新主战场。

### 6.4 目前在争什么

- 没有动作干预的未来视频是否够资格称世界模型？
- 生成完整像素是否是控制不需要支付的“pixel tax”？
- 互联网视频缺少动作标签，学到的是因果动力学还是相关性？
- 长时一致性需要隐式状态、外部记忆，还是原生 3D/4D 状态？
- 视频预测 benchmark 的提升是否能稳定转化为控制成功率？

### 6.5 判断

视频世界模型的真正分水岭不是画质，而是：**动作可控、状态持久、反事实一致、闭环可用。** 2026 年实时性进展明显，但物理可靠性和长期状态仍远未解决。

---

## 7. 3D / 4D 世界模型：从“有 3D 观感”到“维护同一个世界”

### 7.1 三种容易混淆的 3D

1. **3D 观感视频世界**：输出仍是 RGB，但可导航、有空间感，例如 Genie 2/3；
2. **显式 3D/4D 状态**：状态是 occupancy、point cloud、NeRF、3DGS、mesh、scene graph 等；
3. **视频后提升为 3D 资产**：先生成视频，再用前馈重建器编译为 3DGS / mesh，例如 Lyra 路线。

能移动相机不等于内部拥有可查询 3D；视频后重建出 3D 也不等于世界模型内部原生维护 3D 状态。

### 7.2 八类架构

| 家族 | 世界状态 | 代表作 | 主要价值 |
|---|---|---|---|
| Occupancy world model | 3D voxel / semantic occupancy | OccWorld | 驾驶未来预测与碰撞规划 |
| LiDAR / point world model | 点云、range image、point flow | PointWorld | 度量几何、动态点预测 |
| Gaussian world model | 3D/4D Gaussian | GaussianWorld、AEON | 实时渲染与持续场景更新 |
| NeRF / 3DGS dynamics | 神经场或高斯随时间变化 | 多类动态 NeRF/4DGS | 自由视角动态重建 |
| 几何 cache 引导视频 | 点云、深度、warped context | GEN3C、Voyager | 精确相机控制和新区域补全 |
| 检索式空间记忆 | RGB、latent、pointmap、surfel | SPMem、VMem、WorldCrafter | 长程回访与空间寻址 |
| Video-to-3D lifting | 生成视频 → 前馈 3DGS/mesh | Lyra 1.0/2.0 | 复用视频先验并导出显式资产 |
| Asset / object-centric world | 物体、mesh、材质、关系 | HunyuanWorld、Marble 类系统 | 可编辑、可组合、可接物理引擎 |

### 7.3 代表性路线

**GaussianWorld** 把自车运动、动态对象和新区域补全放进 Gaussian 状态。**GEN3C** 用三维缓存渲染目标视角条件，解决相机控制。**Lyra 1.0** 把视频扩散先验蒸馏成前馈 3DGS；**Lyra 2.0** 增加几何检索记忆和 self-augmentation，使可探索时长和回访一致性提高。**AEON** 尝试用持久 4DGS 统一视频与 3D 世界。

### 7.4 目前在争什么

- **显式 3D 是否一定优于隐式视频 latent？** 显式几何有可测量和回环优势，但错误深度回灌会放大伪影；隐式 latent 灵活但难审计。
- **3DGS 是否适合物理世界状态？** 它擅长渲染，不天然提供碰撞体、对象关系、质量、摩擦和关节。
- **几何该负责什么？** 当前最稳的组合不是让几何直接决定所有像素，而是让几何负责寻址、可见性和约束，生成模型负责外观补全。
- **静态持久世界是否等于可交互世界？** 不是。动态对象被移动后，新状态能否长期保存才是更严格标准。
- **评价**：PSNR/FID 不足以评估回环、拓扑、动态身份、可碰撞性和规划收益。

### 7.5 判断

3D 世界模型正在从“每一帧有三维感”转向“始终是同一个世界”。真正前沿是持久状态、动态对象、原生 4D、对象化资产、物理属性与不确定几何。

---

## 8. JEPA：把预测目标从像素换成可预测结构

### 8.1 基本形式

JEPA 预测目标 embedding，而非重建目标像素：

$$s_x=E_x(x),\quad s_y=E_y(y),\quad \hat s_y=P(s_x,z),\quad \mathcal L=D(\hat s_y,s_y)$$

核心假设是：像素中存在大量不可预测且与任务无关的细节，世界模型应优先学习语义、几何、对象和动力学等可预测结构。

### 8.2 七类 JEPA

| 家族 | 代表作 | 预测对象 | 用途 |
|---|---|---|---|
| Canonical image JEPA | [I-JEPA](https://arxiv.org/abs/2301.08243) | masked image embedding | 自监督表征 |
| Video JEPA | [V-JEPA](https://arxiv.org/abs/2404.08471) | 时空 target embedding | 视频理解与动作预判 |
| Action-conditioned JEPA | [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 动作条件未来 embedding | latent MPC、机器人规划 |
| Frozen-feature world model | DINO-WM | 预训练 dense feature 的未来 | 零样本目标规划 |
| End-to-end JEPA world model | LeWorldModel | 从像素端到端学习 latent dynamics | 轻量规划 |
| Flow / stochastic JEPA | Flow-JEPA | 多未来 latent trajectory | 处理多模态未来与鲁棒性 |
| JEPA-WAM | LeWAM 等 | future latent + action flow | 直接策略与规划统一 |

### 8.3 从表征学习到世界模型

- **I-JEPA**：大块上下文预测大块目标表征，避免像素重建偏向纹理；
- **V-JEPA**：扩展到时空 mask，学习运动与对象表征；
- **V-JEPA 2**：先用大规模无动作视频预训练，再用少量机器人数据训练动作条件 predictor，用 MPC 比较候选动作；
- **LeJEPA / SIGReg**：尝试用显式分布正则简化 anti-collapse；
- **LeWorldModel / Flow-JEPA / LeWAM**：把 latent prediction 推进到端到端动力学和动作闭环。

### 8.4 目前在争什么

**Anti-collapse 不等于学到正确状态。** 表征可以不坍缩，却主要编码慢背景、相机身份、数据集来源等“可预测但任务无关”的捷径。需要区分非坍缩、可预测、控制相关和因果充分。

**如何处理多未来？** 简单 embedding MSE 可能平均多个未来。随机 latent、energy objective、latent diffusion/flow 和多候选 embedding 都在竞争。

**Latent 距离是否等于任务距离？** MPC 会直接利用距离函数；如果 latent 几何没有对齐任务进展，规划器可能 exploit 模型误差。

**是否需要 decoder？** 纯 latent 高效，但难审计。更现实的结构可能是 latent 高频规划 + 稀疏像素/几何 decoder 用于校准和可视化。

**与生成模型谁更优？** 目前缺少同数据、同算力、同策略、同真实机器人任务的干净对照，不能宣布胜负。

### 8.5 判断

JEPA 的意义不只是“另一种 SSL loss”，而是提出世界模型的另一种接口：不必每次渲染未来，只需维护对控制充分的可预测状态。其真正考验是能否在开放域、接触丰富、长时和多未来任务上持续改善真实决策。

---

## 9. 具身智能 VLA：动作如何进入多模态基础模型

### 9.1 六类动作表示与解码

| 家族 | 代表作 | 特点 | 主要取舍 |
|---|---|---|---|
| 离散动作 token AR | RT-1、RT-2、OpenVLA | 动作语言化，复用 LLM 交叉熵和推理栈 | 量化误差、串行延迟、高频冗余 |
| 压缩动作 token | FAST / π0-FAST | DCT + 量化 + BPE 压缩动作块 | 保留 AR 优势，但仍是离散接口 |
| 连续直接回归 | OpenVLA-OFT、StarVLA-α、Helix S1 | 极低延迟，结构简单 | 多峰动作可能被均值化 |
| Diffusion policy | Octo、RDT-1B | 连续、多峰、平滑 action chunk | 多步采样与部署复杂度 |
| Flow action expert | π0、GR-3、GR00T | VLM 提供语义，action expert 生成连续块 | 专家梯度、时延和 chunk 开环风险 |
| 离散 diffusion | MMaDA-VLA | 并行去噪图像、语言和动作 token | 统一性强，尚需更充分真机验证 |

### 9.2 五类融合结构

1. **动作进入 VLM 词表**：RT-2、OpenVLA；
2. **强 VLM + 轻量 action head**：OpenVLA-OFT、StarVLA-α；
3. **VLM + policy adapter**：小型机器人模块读取多层 VLM 特征；
4. **VLM + action expert / MoT**：π0、GR-3、GR00T；
5. **快慢双系统**：低频 VLM 处理语义，高频小策略输出控制，Helix 和 GR00T 是典型。

### 9.3 策略时间结构

- **单步闭环**：反馈快，但大模型调用频繁；
- **Action chunk**：一次输出 $H$ 步，摊薄 VLM 成本，已经成为主流；
- **异步 / 流式 chunk**：RTC 类方法在旧动作块执行时后台生成新块，显式处理推理延迟；
- **分层规划**：RT-H、π0.5 等先生成语言子任务或视觉子目标，再由低层策略执行；
- **记忆增强**：显式维护已完成子任务、被移动物体和失败原因，而不是只扩大上下文。

### 9.4 数据与训练范式

VLA 数据分为四层：

1. **真实机器人轨迹**：动力学准确但昂贵；
2. **跨本体数据**：Open X-Embodiment 等提供规模，但动作语义不一致会负迁移；
3. **仿真 / 合成数据**：提供状态和失败标签，但接触、摩擦和柔性物体存在 sim-to-real gap；
4. **人类 / 互联网视频**：语义和交互多样，但缺机器人动作、力觉和精确状态，需要 inverse dynamics、latent action、轨迹提取或少量机器人后训练。

主训练仍是行为克隆、连续回归或 flow matching；前沿开始加入 offline/online RL、成功失败价值学习、world-model-generated data 和模型式筛选。

### 9.5 目前在争什么

- 离散 token、连续回归、diffusion/flow 哪个更好？必须固定 VLM、数据、动作空间和推理预算才能公平比较。
- 强 VLM + 简单 MLP 是否已经足够？标准短程 benchmark 上很强，但柔性物体、灵巧手、复杂接触和多峰动作仍未被充分裁决。
- Web 语义能否变成物理技能？认识工具不等于知道力、角度和接触。
- action chunk 多长？越长越省算力，但越开环；实时性越来越是模型—推理引擎—控制系统协同问题。
- 跨本体统一动作空间是否可行？更可能是物理语义槽位 + embodiment 描述 + mask + learned embedding 的组合，而非简单 padding。
- 长程瓶颈是规划、记忆还是恢复？三者缺一不可。
- 高 benchmark 分是否能转化为真实部署？需要无人值守成功率、平均无故障时间、恢复率、接管率和控制延迟。

### 9.6 判断

VLA 的工程主线正在收敛为：

```text
多模态 VLM（低频）
  → 语义理解 / 子目标 / 记忆
  → 紧凑空间目标或 hidden state
  → 连续 Action Expert（高频）
  → 真实反馈与恢复
```

但这仍主要是策略模型；是否成为 WAM，取决于未来预测是否进入动作路径。

---

## 10. 世界动作模型 WAM：不是单一模型，而是预测—动作架构家族

![](assets/九大技术方向架构全景：从前馈几何到世界动作模型%20-%20Survey/figures/03-world-action-relations.svg)

> 图 3：视频世界模型、3D 世界模型、JEPA、VLA 与 WAM 的集合关系。WAM 的判据不是“是否生成视频”，而是预测未来是否真正作用于动作。图为本综述自绘。

### 10.1 定义

WAM 的实用定义是：**使未来预测对动作生成、动作评分、规划或动作学习真正可用的 embodied predictive-action model。** 它不是一个官方唯一架构，也不是“视频模型 + 动作头”的同义词。

### 10.2 三种耦合方式

#### 10.2.1 Predict-then-Act

$$p(O'\mid O,l)\,p(A\mid O,O',l)$$

先生成目标或未来，再由 inverse dynamics、动作解码器或几何跟踪得到动作。代表：UniPi、早期视频规划策略。优点是中间未来可解释；缺点是串行、慢，并且视频好看不保证动作可执行。

#### 10.2.2 Action Scoring / MPC

$$A_i\rightarrow \hat O'_i\rightarrow \operatorname{score}(\hat O'_i,g)$$

策略先提出候选动作，世界模型预测后果，价值函数或目标距离选动作。优点是测试时可适应新目标；缺点是多次 rollout 昂贵且会 exploit model error。

#### 10.2.3 Joint World–Action Modeling

$$p(O',A\mid O,l)$$

未来和动作共享 Transformer、Mixture-of-Transformers、联合 DiT 或 flow matching。代表：GR-1、DreamZero、OpenWAM、LeWAM 等。优点是动作与后果互相约束；缺点是训练目标耦合复杂、部署成本高、相关性不等于因果性。

### 10.3 三类中间状态

| 中间状态 | 代表路线 | 优点 | 局限 |
|---|---|---|---|
| 显式 RGB / RGB-D / 多视未来 | Render-and-Decode | 可审计、可预览、能复用视频基座 | 支付完整 pixel tax，延迟高 |
| Latent / feature / flow / geometry | Fast-WAM、JEPA-WAM、光流/点流路线 | 快、任务充分、适合闭环 | 难解释，latent 距离可能不可靠 |
| Video-generation-free predictive state | JEPA、affordance、object-centric state | 不依赖视频扩散，可极致高效 | 需要证明状态保留了接触和精细动力学 |

### 10.4 与相邻概念的边界

| 概念 | 接口 | 是否自动属于 WAM |
|---|---|---|
| 普通 VLA | $(O,l)\rightarrow A$ | 否；可以完全没有未来预测 |
| 世界模型 | $(O,A)\rightarrow O'$ | 否；只是 simulator，未生成或选择动作 |
| 动作条件视频生成 | action in, video out | 否；除非输出进入动作路径 |
| Future-prediction auxiliary VLA | 训练时多一个未来预测 loss | 有争议；若部署完全丢弃，严格口径下不是完整 WAM |
| World-model-powered VLA | 世界模型生成数据/子目标 | 广义相关，但策略本体未必是 WAM |
| 联合 world–action model | 同时建模未来与动作 | 是最明确的 WAM |

### 10.5 代表作与贡献

| 工作 | 类型 | 贡献 |
|---|---|---|
| [UniPi](https://arxiv.org/abs/2302.00111) | 级联 Predict-then-Act | 文本生成视频计划，再由 inverse dynamics / planner 解出动作 |
| GR-1 | 联合视频—动作 | 共享 Transformer 同时预测未来图像和动作，证明视频预训练可进入策略 |
| JOWA | 联合 world–action | 在 Atari 离线 RL 中联合世界预测与动作价值，较早明确使用 World-Action 名称 |
| [DreamZero](https://dreamzero0.github.io/) | 视频扩散联合 WAM | 在视频生成骨干中联合预测未来视频与动作，推进零样本和闭环机器人控制 |
| Fast-WAM | 训练期预测、测试期跳过显式未来 | 质疑部署时完整视频想象是否必要，强调预测式协同训练价值 |
| Motus / FlowWAM | 多专家或统一动作表示 | 探索视频、动作、逆动力学和世界模型共享主干 |
| [OpenWAM](https://openwam.stanford.edu/) | 模块化系统研究 | 控制变量比较视频骨干、latent、动作容量、信息流和数据配方 |
| LeWAM | JEPA-WAM | 联合 next-latent prediction 与 action flow，兼容直接策略和 latent planning |

### 10.6 目前在争什么

**测试时是否必须显式想象未来？** Fast-WAM 类结果提示收益可能主要来自训练期视频/未来监督；但关键节点规划、可解释性和失败诊断仍可能需要显式 rollout。

**联合生成是否等于因果理解？** 未来与动作共同出现，只能证明相关建模。真正的因果性需要反事实动作、失败轨迹和多候选动作评估。

**规划还是 amortized policy？** MPC 灵活但慢；直接策略快但测试时纠错弱。更现实的是策略提案 + 轻量 latent reranking，关键节点才做重规划。

**像素还是 latent？** 像素可审计，latent 高效。可能的终局是 latent 高频控制 + 按需渲染。

**动作标签从哪里来？** 真实控制精确但稀缺；latent action、inverse dynamics、光流、人手轨迹和高层 waypoint 可扩展但存在语义与本体偏差。

**WAM 是否会取代 VLA？** 更准确的关系是融合：VLM/VLA 提供开放世界语义与目标，WAM 提供动作后果、候选评估和恢复。所谓“VLA 已死”更像路线宣言，不是已被同条件实验验证的结论。

### 10.7 判断

WAM 最有价值的变化不是增加一个视频输出头，而是把问题从：

> “看到什么、听到什么后直接做什么？”

改成：

> “不同动作会怎样改变世界，哪些未来可达，哪个动作最值得执行？”

它能否成为下一代具身基础模型，取决于是否在同数据、同算力、同控制频率下稳定提升真实闭环，而不是只展示更逼真的“想象”。

---

## 11. 九方向的统一坐标系

以后读一篇新论文，可以不用先问“它属于哪个名词”，而直接沿以下十个轴定位：

| 轴 | 可能取值 | 关键问题 |
|---|---|---|
| 输入证据 | 单图 / stereo / 多视 / 视频 / 稀疏深度 / 多传感器 | 问题可观测到什么程度？ |
| 相机状态 | 已知 / 部分已知 / 网络估计 / 隐式 ray | 能否直接使用投影几何？ |
| 尺度规范 | ordinal / affine / scale-invariant / metric | 能否用于真实测量？ |
| 状态表示 | depth / pointmap / token / latent / occupancy / 3DGS / mesh | 世界以什么形式存在？ |
| 信息流 | 单帧 / pairwise / all-to-all / recurrent / causal / group-AR | 如何交换空间和时间信息？ |
| 预测目标 | 像素 / 视频 latent / embedding / 几何 / reward / action | 模型被迫保留什么信息？ |
| 时间结构 | 离线整段 / chunk / frame causal / streaming / hierarchical | 能否实时，如何处理长期状态？ |
| 动作接口 | 无动作 / 离散 token / 连续向量 / latent action / flow | 动作如何影响未来？ |
| 决策耦合 | 无 / 辅助 loss / 生成子目标 / 候选评分 / 联合策略 | 预测是否真的改变动作？ |
| 可信度 | 点估计 / confidence / uncertainty / 多未来后验 | 系统何时应该不相信自己？ |

---

## 12. 跨方向仍未解决的八个核心争议

### 12.1 像素预测 vs 潜变量预测

视频生成和显式未来可视化信息丰富、可审计，但昂贵且可能浪费容量在纹理；JEPA/latent dynamics 高效，却难证明 latent 距离与物理、任务进展一致。当前没有同条件决定性实验。

### 12.2 2D 视频 vs 显式 3D/4D 持久状态

视频模型容易利用海量互联网数据，显式 3D 更适合度量、回环、碰撞和规划。最可能的融合是：视频先验负责外观与开放域补全，几何状态负责寻址、可见性、尺度和持久性。

### 12.3 专用模块 vs 统一基础模型

DA3、VGGT、Cosmos、统一 VLA 都在减少任务专用模块。但传感器固定、实时严格或安全关键场景仍需要专家 head、显式约束和后端优化。“统一 backbone + 专家接口”比“一个 head 解决全部任务”更现实。

### 12.4 纯前馈 vs 优化 / 搜索

前馈速度快、泛化强；BA、MPC、CEM 和价值搜索能在测试时纠错。真正问题不是谁消灭谁，而是何时触发优化、优化什么变量、如何避免 exploit model error。

### 12.5 大数据缩放 vs 结构归纳偏置

越来越多结果显示强 backbone + 更多数据可压过复杂模块；但物理、拓扑、接触、尺度和因果仍可能需要结构化表示。需要受控比较，而不是把所有增益都归因于“架构创新”。

### 12.6 逼真度 vs 决策效用

渲染质量、FVD 或人类偏好不能代替 action adherence、反事实一致性、物理合理性、持久性和真实任务成功率。世界模型最终必须用决策效用评价。

### 12.7 长上下文 vs 真正记忆

无限扩大 token 窗口无法自动维护对象状态。长期世界需要区分近期高分辨率视觉、长期语义/几何记忆、事件状态和可检索外部存储，并处理错误记忆的写入和修复。

### 12.8 开源可复现 vs 闭源系统性能

视频生成、世界模型和机器人领域大量领先系统只披露产品能力。架构比较必须标注：正式论文、开源权重、训练数据、独立复现、官方博客还是营销演示。闭源第一名不能自动成为科研主线证据。

---

## 13. 未来 12–24 个月最值得跟踪的十条线

1. **Depth-ray / pointmap 的统一几何接口**能否稳定覆盖单图、多视、视频、depth completion 和 metric scale；
2. **前馈几何 latent 是否成为 VLA 的空间预训练底座**，而不只是重建输出；
3. **流式几何模型的回环与重定位**，即神经状态能否吸收传统 SLAM 的全局一致性；
4. **视频 VAE 的生成友好表示**，而不只追求重建指标和高压缩率；
5. **原生稀疏时空注意力**能否在不牺牲快速运动和接触细节下替代 full attention；
6. **视频世界模型的持久状态 benchmark**：回访、对象移动后状态、反事实动作、物理约束；
7. **原生 4D 世界状态**：不再只是视频生成后重建，而是状态本体可查询、可编辑、可规划；
8. **JEPA 与生成模型的同条件对照**：同数据、同算力、同策略、同机器人任务；
9. **VLA 的强表征 + 简单 head 基线**在灵巧手、柔性物体、多峰接触和长程恢复上是否仍成立；
10. **WAM 是否真正提升闭环决策**，尤其是失败轨迹、反事实数据和模型误差利用问题。

---

## 14. 最终判断：现在到底“收敛”到哪里了

### 14.1 几何方向

已经从大量任务专用网络，收敛到“强视觉 backbone + 少量统一几何变量 + 可选条件输入”。但显式对应、相机尺度和后端优化不会消失。

### 14.2 视频方向

已经从 U-Net 运动模块，收敛到“3D causal VAE + 时空 DiT + diffusion/flow”；下一步差异主要在 tokenizer、稀疏注意力、流式因果、蒸馏、音画联合与系统工程。

### 14.3 世界模型方向

尚未收敛。至少三条主线并存：生成式视频模拟、显式 3D/4D 状态、JEPA 潜空间预测。最可能的终局是混合式，而不是单一路线彻底胜出。

### 14.4 具身智能方向

VLA 正收敛到“低频语义理解 + 高频 action expert + action chunk + 异步执行”；WAM 则试图把动作后果重新放回预训练和推理核心。两者更可能融合，而不是替代。

### 14.5 一句话结论

> **下一代空间智能系统不会只是一台重建器、视频生成器、VLA 或世界模型；它会维护一个可度量、可预测、可记忆、可查询的世界状态，并在行动前后持续用真实反馈修正它。**

---

## 15. 材料来源与阅读路径

- 详细官方来源清单：`assets/九大技术方向架构全景：从前馈几何到世界动作模型 - Survey/sources.md`
- 图 1–3 均为本综述基于上述论文自行绘制的架构关系图，不是论文原图。
- 对单篇模型的张量级、实验级和代码级细节，应进入对应精读笔记；本综述的目标是建立全局坐标系，而非替代单篇复现。
