---
type: survey
status: complete
domain: World-Models
topic: Persistent explorable 3D worlds
core_papers: 11
foundation_papers: 10
created: 2026-10-15
updated: 2026-10-15
tags:
  - World-Models
  - Video-Generation
  - 3D-Reconstruction
  - Long-Term-Memory
  - Lyra
---

# Lyra 与持久可探索三维世界：从相机控制视频到可仿真三维资产

## 基本信息

- **覆盖论文**：GEN3C、Lyra 1.0、Lyra 2.0、Context as Memory、SPMem、VMem、FramePack、Self-Forcing、Yume 1.5、WorldPlay、FantasyWorld
- **时间跨度**：2025-03 至 2026-04
- **核心问题**：怎样让视频生成模型沿用户控制的相机轨迹持续扩展场景，重访旧区域时保持一致，同时把生成结果转成可实时渲染、可进入仿真器的显式三维资产？
- **一句话判断**：这条路线已经从“短片段相机控制”进入“长程记忆 + 自回归纠错 + 前馈几何编译”的系统阶段，但距离真正的物理世界模型仍缺动态对象、拓扑可靠性、因果交互和统一评测。

---

## 1. 领域要解决的根本问题

### 1.1 从新视角生成到世界生成

单图新视角生成的传统目标，是在输入视图附近合成若干合理视角；交互式世界模型的目标则更强：用户可以连续移动、转身、回到旧位置、探索未观测区域，并期待场景保持身份、布局与外观稳定。这里至少同时包含四个不同问题：

1. **相机控制**：模型输出是否真的遵守指定内外参，而不是只生成“看起来像移动”的视频；
2. **空间记忆**：数百帧前看到的房间、门窗和物体，在重访时是否仍然存在于同一位置；
3. **时间稳定**：自回归条件来自模型自身输出时，误差是否不断累积；
4. **显式落地**：生成视频能否被转换成 3DGS、mesh 或其他可渲染、可交互表示。

这四者并非自然兼容。硬几何约束能够提高相机准确性，却会把深度错误和 warp 伪影传播给后续生成；纯视频上下文视觉质量更高，却容易在历史帧退出上下文窗口后遗忘；更强的记忆能够保留正确细节，也会同时记住错误并加速漂移。

### 1.2 三组核心矛盾

| 核心矛盾 | 一端 | 另一端 | 代表解决思路 |
|---|---|---|---|
| 几何约束 vs 生成自由度 | 直接渲染点云，控制强但伪影硬 | 纯生成先验，画质高但控制弱 | Lyra 2.0：几何只做路由，不做 RGB 合成 |
| 长期记忆 vs 计算预算 | 保存全部历史，成本持续增长 | 固定窗口，旧场景被遗忘 | FramePack 压缩；CaM/VMem/SPMem 检索 |
| 长程依赖 vs 错误积累 | 依赖历史保证连续 | 历史自身含误差 | Self-Forcing / self-augmentation / Context Forcing |

---

## 2. 问题演化：为什么后一篇论文会出现

### 2.1 第一次转移：从“输入相机参数”到“显式几何控制”

早期相机可控视频模型主要把相机位姿编码成网络条件，让模型自行推断像素如何随相机运动。这种做法对小幅平移有效，但很难保证复杂轨迹、内参变化和回看时的一致性。GEN3C 的关键转移是引入 **3D cache**：先估计输入或历史帧深度，把像素反投影成彩色点云，再从目标相机渲染结构条件。于是相机控制不再完全依赖 DiT 从位姿向量中“猜”几何变化。

![](../../10-Papers/Video-Generation/assets/GEN3C%20-%20%E4%B8%89%E7%BB%B4%E6%84%9F%E7%9F%A5%E7%9A%84%E4%B8%96%E7%95%8C%E4%B8%80%E8%87%B4%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90/figures/fig3_method.png)

> GEN3C 把几何缓存渲染成目标视角的 RGB、深度与空洞掩码，引导视频模型只对未观测区域发挥生成能力。它解决了精确控制，但缓存仍可能积累生成误差。来源：GEN3C Figure 3。

GEN3C 暴露的新瓶颈是：它最终仍输出二维视频，而且几何缓存越长，深度与生成误差越可能被重新渲染并放大。

### 2.2 第二次转移：从“生成视频”到“蒸馏显式 3D”

Lyra 1.0 不再把视频扩散只看成最终生成器，而把它视作隐含多视图知识的教师。冻结的 GEN3C 生成不同相机轨迹的视频 latent，RGB 解码器提供合成多视图监督，新增 3DGS decoder 学习把这些不完全一致的视图压缩成统一显式场景。

![](../../10-Papers/3D-Reconstruction/assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig3_method.png)

> Lyra 1.0 的本质是 video latent → explicit 3DGS 的自蒸馏：教师负责想象不可见内容，学生负责把多轨迹观测统一成可实时渲染表示。来源：Lyra Figure 3。

这一阶段解决了“视频怎样成为 3D”，却仍受教师生成窗口约束：一次只覆盖有限空间，没有长期重访记忆，无法不断扩张为大型世界。

### 2.3 第三次转移：从“有限片段”到“长期空间记忆”

2025 年中，多篇工作同时指出固定时间窗口并不等于世界记忆，但给出了三种不同答案：

- **Context as Memory**：直接保存历史 RGB 帧，用相机视场重叠检索；
- **SPMem**：维护长期静态点图、近期工作记忆与稀疏情景关键帧；
- **VMem**：用 surfel 记录哪些历史视图看过当前表面，再检索这些视图。

![](../../10-Papers/World-Models/assets/Context%20as%20Memory%20-%20%E5%8E%86%E5%8F%B2%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A9%B1%E5%8A%A8%E7%9A%84%E5%9C%BA%E6%99%AF%E8%AE%B0%E5%BF%86/figures/02-method.png)

> CaM 不建立持久三维外观，而是把历史帧本身当记忆，以相机视锥重叠筛选当前需要的上下文。来源：CaM 方法图。

![](../../10-Papers/World-Models/assets/SPMem%20-%20%E9%95%BF%E6%9C%9F%E7%A9%BA%E9%97%B4%E8%AE%B0%E5%BF%86%E8%A7%86%E9%A2%91%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/figures/02-method.png)

> SPMem 的认知式三层记忆把近期像素、长期静态几何和稀疏历史关键帧分开存储；优点是表示完整，缺点是融合点图会累积误差。来源：SPMem 方法图。

![](../../10-Papers/World-Models/assets/VMem%20-%20Surfel%E7%B4%A2%E5%BC%95%E8%A7%86%E5%9B%BE%E8%AE%B0%E5%BF%86/figures/02-method.png)

> VMem 用几何做“视图索引”，并不直接用 surfel 颜色作为最终渲染结果。这一点与 Lyra 2.0 的“几何只做路由”非常接近。来源：VMem Figure 2。

这些方法共同证明：真正重要的不是保存最近帧，而是能根据目标位置取回**空间相关**的旧观察。但它们仍没有完整解决自回归误差积累。

### 2.4 第四次转移：从“增加记忆”到“让模型会纠错”

FramePack 指出遗忘与漂移存在内在张力：历史越强，正确信息保留越久，但错误也传播得越久。它用按重要性压缩的时间上下文把 token 数限制在固定上限，再通过端点先行、倒序生成等采样策略缩短误差传播链。

![](../../10-Papers/Video-Generation/assets/FramePack%20-%20%E5%9B%BA%E5%AE%9A%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A2%84%E7%AE%97%E7%9A%84%E9%95%BF%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90/figures/01-packing-method.png)

> FramePack 不是空间记忆，而是固定预算的时间记忆：越近、越重要的帧获得更多 token，远处历史逐级压缩。来源：FramePack 方法图。

Self-Forcing 则把问题追溯到训练—推理分布错配：训练时用真值历史，推理时依赖自身不完美输出。其核心是训练阶段直接执行可微 self-rollout，并用视频级分布匹配损失监督整条生成轨迹。

![](../../10-Papers/Video-Generation/assets/Self-Forcing%20-%20弥合自回归视频扩散的训练推理鸿沟/figures/01-training-paradigms.png)

> Self-Forcing 不再用真值或人工加噪历史近似推理状态，而是在训练中真实滚出模型自身输出；这是后续交互世界模型进行因果蒸馏和误差纠正的重要基础。来源：Self-Forcing Figure 1。

Lyra 2.0 的 self-augmentation 是更便宜的近似：不做完整多步 rollout，而是对历史 latent 加噪并进行一次模型去噪，把得到的退化历史作为条件，同时仍监督干净当前片段。消融显示这会略微损失单帧主观质量，却显著改善长期风格与控制稳定性。

### 2.5 第五次转移：从“记住视频”到“持久、可重建的三维世界”

Lyra 2.0 把以上分支组合成 retrieve–generate–update 循环：

1. 用逐帧深度和相机建立空间缓存；
2. 根据目标视角检索覆盖最多的历史帧；
3. warp 规范坐标和深度，建立历史 token 与目标 token 的稠密对应；
4. 让 Wan 2.1 DiT 负责真正的 RGB 合成；
5. 通过 self-augmentation 抑制自回归漂移；
6. 把长视频交给针对生成数据微调的 DAv3，输出 3DGS 与 mesh。

![](../../10-Papers/3D-Reconstruction/assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig2_method.png)

> Lyra 2.0 的关键并不是更大的视频模型，而是把时间历史、空间历史和几何对应分别编码，再在注意力中汇合。来源：Lyra 2.0 Figure 2。

### 2.6 平行分支：交互实时化与联合几何预测

Yume 1.5、WorldPlay 与 FantasyWorld 并非 Lyra 的直接前后代，却分别补足了三块关键能力。Yume 1.5 关注键鼠控制、文本事件和长视频上下文压缩；WorldPlay 关注记忆感知模型如何蒸馏到实时流式生成；FantasyWorld 则把几何分支直接接到冻结视频基础模型内部，让视频外观与三维预测在一次生成中双向交换信息。

![](../../10-Papers/World-Models/assets/Yume%201.5%20-%20文本控制的交互式世界生成/figures/02-tscm-architecture.png)

> Yume 1.5 的 TSCM 同时压缩时间、空间与通道，使历史增长不再线性推高上下文开销，并加入文本事件和键盘控制。来源：Yume 1.5 Figure 3。

![](../../10-Papers/World-Models/assets/WorldPlay%20-%20长期一致的实时交互世界模型/figures/02-method-overview.png)

> WorldPlay 把双重动作表示、重组上下文记忆和 Context Forcing 蒸馏组合起来，目标不是离线最高画质，而是 720p、24 FPS 的实时交互。来源：WorldPlay 方法总览。

![](../../10-Papers/World-Models/assets/FantasyWorld%20-%20统一视频与三维预测/figures/02-architecture.png)

> FantasyWorld 使用视频想象分支与几何分支的双向交叉注意力，把“先生成、后重建”改成联合视频—几何预测；但当前仍以固定长度片段为主，缺少 Lyra 2.0 式持久记忆。来源：FantasyWorld 架构图。

---

## 3. 技术时间线

| 时间 | 工作 | 解决的问题 | 新暴露的瓶颈 |
|---|---|---|---|
| 2025-03 | GEN3C | 3D cache 提升相机控制和局部一致性 | 仍是视频；硬渲染条件会传播误差 |
| 2025-04 | FramePack | 固定 token 预算容纳长历史 | 时间压缩不能恢复空间上相关但时间很远的细节 |
| 2025-06 | Context as Memory | 基于视场重叠检索历史 RGB | 无稠密几何对应，大视角变化下匹配有限 |
| 2025-06 | SPMem | 空间/工作/情景三层记忆 | 融合点图可能积累深度和配准误差 |
| 2025-06 | Self-Forcing | 训练时使用自身 rollout，缓解 exposure bias | 完整 rollout 训练代价高，训练时域外仍会退化 |
| 2025-06 | VMem | surfel 索引最相关历史视图 | 几何估计和 surfel 合并仍会失真 |
| 2025-09 | Lyra 1.0 | 视频扩散知识自蒸馏为 3D/4D GS | 场景范围受教师窗口限制，无长期探索 |
| 2025-09 | FantasyWorld | 视频与几何双分支相互增强 | 固定长度生成，缺持续记忆与流式缓存 |
| 2025-12 | Yume 1.5 | 长上下文压缩、文本事件和键盘交互 | 几何持久性仍以视频一致性为主 |
| 2025-12 | WorldPlay | 记忆感知蒸馏与实时 720p 流式生成 | 显式三维质量和物理可用性仍不足 |
| 2026-04 | Lyra 2.0 | 空间检索、稠密路由、自纠错和 3D 导出闭环 | 静态场景、曝光漂移、生成几何仍不精确 |

---

## 4. 主要路线横向对比

### 4.1 记忆放在哪里

| 方法 | 长期记忆形式 | 检索依据 | 是否直接生成 3D | 主要风险 |
|---|---|---|---|---|
| FramePack | 压缩历史 latent/token | 时间距离或特征重要性 | 否 | 远历史细节不可逆丢失 |
| CaM | 原始历史 RGB 帧 | 相机 FOV 重叠 | 否 | 大基线时无稠密对应 |
| SPMem | 融合点图 + 近期帧 + 关键帧 | 目标视图几何可见性 | 否 | 全局点图误差累积 |
| VMem | surfel 到历史视图 ID 的倒排索引 | 当前表面对应的历史观察 | 否 | surfel 建图与合并误差 |
| WorldPlay | 动态重组的历史上下文 | 几何重要性与 temporal reframing | 可后续重建 | 蒸馏需保留记忆使用能力 |
| Lyra 2.0 | 每帧独立深度、位姿、点云和 RGB | 目标视角覆盖 + 稠密 canonical correspondence | 是，后接 DAv3 | 深度误差与静态场景假设 |

### 4.2 几何在系统中扮演什么角色

这 11 篇工作可以按几何作用分为四层：

1. **无显式几何**：FramePack、Self-Forcing，主要处理序列建模和训练对齐；
2. **几何只做粗检索**：CaM 使用相机视锥，不恢复稠密表面；
3. **几何做记忆索引或路由**：VMem、Lyra 2.0，保存历史图像作为真正外观来源；
4. **几何参与生成或成为最终输出**：GEN3C 用点云渲染作为生成条件，SPMem 融合长期点图，FantasyWorld 联合预测视频与隐式几何，Lyra 1.0/2.0 最终输出显式 3DGS 或 mesh。

我的判断是，第 3 层是当前最稳健的折中。生成视频里的深度并不可靠，直接把其当作持久世界本体容易形成错误闭环；完全不用几何则无法跨大视角找到正确历史。让几何只承担“寻址”职责，可以获得空间结构的好处，同时把外观修复留给生成先验。

### 4.3 Lyra 1.0 与 Lyra 2.0 不是简单版本升级

| 维度 | Lyra 1.0 | Lyra 2.0 |
|---|---|---|
| 主要任务 | 从有限多轨迹视频 latent 解码 3D/4D | 先生成持久长视频，再前馈重建为 3D |
| 视频模型 | 冻结 GEN3C 教师 | 微调 Wan 2.1-14B 自回归生成器 |
| 显式 3D | 专用 latent-to-3DGS decoder | 生成数据微调的 DAv3 3DGS head |
| 训练监督 | 视频教师 RGB、估计深度 | DL3DV 长视频、深度/位姿和自增强历史 |
| 记忆 | 多轨迹输入一次性融合 | 持续增长的逐帧空间缓存 |
| 动态场景 | 支持单目视频到时间条件化 4DGS | 当前版本主要面向静态世界 |

2.0 在“长期探索”上明显前进，但并不在所有维度支配 1.0：Lyra 1.0 的动态 4D 表示反而是 2.0 当前未覆盖的能力。因此更准确的理解是，两者分别研究**显式解码**与**持久生成**，而非严格替代关系。

---

## 5. 实验与评测体系的变化

### 5.1 从短片画质到回环一致性

传统视频生成指标偏向单段 FID、FVD 或主观画质，无法判断模型离开房间后再回来时是否保留原布局。CaM、SPMem、VMem 与 Lyra 2.0 因此引入或强调：

- 历史位置重访；
- 长轨迹 SSIM、LPIPS 与 FID；
- 相机可控性和重投影误差；
- 风格一致性；
- 生成视频经 3D 重建后的新视角误差。

这反映出领域正在从“视频像不像”转向“视频能否被解释为同一个世界”。

### 5.2 单帧质量与长期稳定不是同一目标

Lyra 2.0 最有价值的消融之一是：去掉 self-augmentation 后，单帧主观质量上升，但长期风格一致性和相机控制显著下降。FramePack 与 Self-Forcing 也揭示相同现象：局部更锐利不意味着长程更稳定。未来 benchmark 应至少分别报告：

- 短段视觉保真；
- 重访身份一致；
- 几何/相机一致；
- 长程退化曲线；
- 显式重建可用性。

### 5.3 目前证据仍存在的缺口

不同论文使用不同数据、轨迹长度、分辨率、用户研究和主观评分器，跨论文数字无法直接比较。尤其是“可无限生成”“长期持久”等表述，很多时候只由有限的数百帧演示支持。WorldPlay 的实时帧率与 Lyra 2.0 的高质量离线生成也属于不同目标，不应以单一 FPS 或 FID 排序。

---

## 6. 当前正在攻坚的问题

### 6.1 动态世界的持久记忆

Lyra 2.0 明确以静态环境为主。若场景中有人、车辆或可操作物体，记忆系统必须区分：

- 静态背景中可长期复用的几何；
- 动态对象的身份、轨迹和遮挡关系；
- 用户动作造成的持久状态变化；
- 可逆运动与不可逆事件。

SPMem 会过滤动态部分，实际上绕开了“动态世界怎样记忆”的核心难题；Lyra 1.0 虽支持 4DGS，却没有无限探索和状态更新机制。把两者统一仍是开放问题。

### 6.2 从视觉一致走向物理一致

现有系统输出的 mesh 可进入 Isaac Sim，不代表几何已经适合物理仿真。主要缺口包括：

- 薄结构、遮挡面和大平面拓扑不稳定；
- 没有可靠材质、摩擦、质量和碰撞语义；
- 未观测区域是视觉上合理的生成，不是真实恢复；
- 尺度和可通行空间可能不满足机器人规划。

因此，当前更适合概念验证、视觉域随机化和环境原型，而不是高精度数字孪生。

### 6.3 记忆容量、写入策略与遗忘策略

“内存持续增长”本身不是可扩展答案。真正长期的系统需要决定：

- 哪些帧应当保存原始分辨率；
- 哪些历史可被压缩、合并或丢弃；
- 何时相信新观察，何时保留旧观察；
- 如何检测记忆冲突与错误写入；
- 如何让检索复杂度随世界规模次线性增长。

Lyra 2.0 当前的逐帧缓存避免融合误差，却把存储和检索成本留给未来。

### 6.4 训练—推理一致仍未完全解决

Self-Forcing 在训练时执行真实 self-rollout，最贴近推理分布但成本高；Lyra 2.0 的一步 self-augmentation 便宜，但只近似模型在长轨迹中的复杂错误分布；WorldPlay 的 Context Forcing 则进一步要求蒸馏学生与教师使用一致的记忆上下文。三者说明未来训练至少需要同时对齐：

1. 自身生成历史；
2. 实际记忆检索结果；
3. 蒸馏教师可见的信息结构；
4. 推理时的长程动作分布。

---

## 7. 判断与展望

### 7.1 已经较为确立的事实

1. 只保留最近几帧不足以构成世界记忆；
2. 只扩大上下文也不能自动解决空间重访与自回归漂移；
3. 显式几何能显著改善相机控制，但硬 RGB/点云回灌容易放大错误；
4. 训练时让模型接触自己的输出，是稳定长程生成的关键；
5. 视频基础模型可以为前馈 3D 重建提供比传统 3D 数据更开放域的生成监督；
6. 生成视频要成为可用 3D 资产，重建模型必须针对生成分布进行适配。

### 7.2 我的推测

下一代系统很可能不会选择“纯视频记忆”或“单一全局三维地图”中的任一极端，而会采用分层混合状态：

- 近期高保真视频 latent，负责运动和细节连续；
- 稀疏关键帧/对象记忆，负责身份和纹理；
- 可更新的度量几何或场景图，负责空间寻址与规划；
- 对自身不确定性的显式估计，决定哪些区域可以复用、哪些需要重新生成或重建。

Lyra 2.0 已经呈现这一结构的雏形，但其状态仍以静态逐帧几何为主。真正的“3D world model”需要把对象、动作后果和物理属性也写进可持续更新的世界状态。

---

## 8. 阅读顺序建议

### 8.1 想理解 Lyra 主线

1. [[GEN3C - 三维感知的世界一致视频生成]]
2. [[Lyra 1.0 - 视频扩散自蒸馏三维重建]]
3. [[Lyra 2.0 - 可探索的生成式三维世界]]

### 8.2 想理解空间记忆

1. [[Context as Memory - 历史上下文驱动的场景记忆]]
2. [[SPMem - 长期空间记忆视频世界模型]]
3. [[VMem - Surfel索引视图记忆]]
4. [[Lyra 2.0 - 可探索的生成式三维世界]]

### 8.3 想理解长视频与抗漂移

1. [[FramePack - 固定上下文预算的长视频生成]]
2. [[Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟]]
3. [[Yume 1.5 - 文本控制的交互式世界生成]]
4. [[WorldPlay - 长期一致的实时交互世界模型]]

### 8.4 想理解视频与 3D 联合建模

1. [[FantasyWorld - 统一视频与三维预测]]
2. [[Lyra 1.0 - 视频扩散自蒸馏三维重建]]
3. [[Lyra 2.0 - 可探索的生成式三维世界]]

---

## 9. 文献索引

| 短名 | 正式论文名 | 原始链接 |
|---|---|---|
| GEN3C | GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control | https://arxiv.org/abs/2503.03751 |
| FramePack | Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models | https://arxiv.org/abs/2504.12626 |
| CaM | Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval | https://arxiv.org/abs/2506.03141 |
| SPMem | Video World Models with Long-term Spatial Memory | https://arxiv.org/abs/2506.05284 |
| Self-Forcing | Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion | https://arxiv.org/abs/2506.08009 |
| VMem | VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory | https://arxiv.org/abs/2506.18903 |
| Lyra 1.0 | Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation | https://arxiv.org/abs/2509.19296 |
| FantasyWorld | FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction | https://arxiv.org/abs/2509.21657 |
| WorldPlay | WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling | https://arxiv.org/abs/2512.14614 |
| Yume 1.5 | Yume-1.5: A Text-Controlled Interactive World Generation Model | https://arxiv.org/abs/2512.22096 |
| Lyra 2.0 | Lyra 2.0: Explorable Generative 3D Worlds | https://arxiv.org/abs/2604.13036 |

---

## 10. 基础模型、几何引擎与表示依赖图谱

### 10.1 依赖并不是同一种依赖

上述 11 篇论文下方至少有四层不同底座。第一层是**视频生成骨干**，负责外观、运动与未观测区域补全；第二层是**几何感知与标注引擎**，负责深度、相机、点图或训练伪标签；第三层是**训练和加速范式**，负责把多步双向生成器变成可自回归、少步或实时的模型；第四层是**显式三维表示与高效序列算子**，决定最终资产形态和重建网络的计算方式。

把它们统称为“基础模型”会掩盖重要差别。例如 ViPE 是学习前端与稠密 BA 组成的混合标注引擎，3DGS 是可微三维表示，Mamba-2 是序列算子；它们都很基础，却不是同一种 foundation model。

### 10.2 核心依赖清单与当前精读状态

| 层级 | 基础工作 | 在 Lyra 相关路线中的作用 | 独立精读与处理结果 |
|---|---|---|---|
| 视频骨干 | [[Stable Video Diffusion - 大规模潜视频扩散模型]] | GEN3C 原始 I2V backbone；生成先验负责修复 3D cache 渲染空洞 | 已有，新增 GEN3C 精确接口与 Cosmos 迁移对照 |
| 视频骨干 | [[Wan2.1 - 开源大规模视频生成基础模型]] | Lyra 2.0、Self-Forcing、FantasyWorld 的主要生成底座 | 已有，新增 Wan2.2 MoE/TI2V/VAE 与四个下游改造对照 |
| 几何深度 | [[Depth Anything V2 - 精确合成监督与真实伪标签桥接]] | GEN3C 单目深度前端，把 seed/history frame 反投影为 3D cache | 已有，新增 depth 错误闭环和 Lyra 2.0 解耦对照 |
| 几何基础模型 | [[DA3 - 任意视角恢复视觉空间]] | Lyra 2.0 3DGS 重建后端的基础；生成域微调后容忍轻微多视不一致 | 已有，新增 GS-DPT、$k=2$ 降采样、3,000 段生成视频微调边界 |
| 通用几何 | [[VGGT - 前馈式通用三维重建]] | FantasyWorld 的架构灵感和独立重建评测；不是 Lyra 的直接 backbone | 已有，新增“VGGT-style 不等于使用 VGGT 权重”的边界说明 |
| 在线几何 | [[CUT3R - 带持久状态的连续三维感知]] | SPMem 在线 pointmap/pose 引擎；FantasyWorld 几何伪标签来源 | 本轮新增完整精读 |
| 位姿标注 | [[ViPE - 视频位姿与稠密几何标注引擎]] | Lyra 与 WorldPlay 等训练数据的位姿、内参与近 metric 深度标注 | 本轮新增完整精读 |
| 少步蒸馏 | [[DMD - 分布匹配蒸馏一步图像生成]] | Lyra 2.0 35→4 步、Self-Forcing video-level DMD、WorldPlay Context Forcing 的上游方法论 | 已有，新增三类视频迁移的条件分布差异 |
| 显式表示 | [[3DGS - 实时辐射场渲染的三维高斯表示]] | Lyra 1.0/2.0 的可实时渲染输出，也是 DA3 Gaussian head 的表示基础 | 本轮新增完整精读 |
| 序列算子 | [[Mamba-2 - 状态空间对偶与高效序列建模]] | Lyra 1.0 的 3DGS decoder 以 Transformer + Mamba-2 融合 726 视图 latent | 本轮新增完整精读 |

### 10.3 四个本轮补齐的基础工作

#### 10.3.1 3DGS：为什么 Lyra 最终选择高斯而不是只输出视频

原始 3DGS 用位置、各向异性协方差、不透明度和球谐颜色描述显式高斯，再通过 visibility-aware tile rasterizer 做可微 alpha compositing。它使场景表示同时具备高质量、快速训练和 1080p 实时渲染。Lyra 1.0 的 3DGS decoder 与 Lyra 2.0 的 Gaussian DPT head 都跳过了原论文的逐场景优化，直接前馈预测高斯参数；但渲染公式、协方差参数化和最终资产语义仍来自 3DGS。

关键边界是：**原始 3DGS 解决“给定充分照片如何优化场景”，Lyra 解决“如何生成缺失观察并一次预测高斯”**。后者继承表示与渲染器，不等于继承原始优化流程。

#### 10.3.2 CUT3R：生成世界为什么需要持续几何状态

CUT3R 用 recurrent state 在每个新观测到来时同时执行 state update 与 state readout，输出统一坐标系中的 metric pointmap 和相机参数，还能用虚拟 raymap 查询未观测视角。SPMem 用它在线追踪点图/位姿，FantasyWorld 用它生成部分几何标签。

它与生成式记忆的结构性关系在于：CUT3R 的 state 是紧凑、可持续更新的几何状态，但没有传统 SLAM 式强回环全局校正；一旦生成帧或位姿出错，误差会沿 `pose → pointmap → state → readout` 写回。这也是 Lyra 2.0 宁愿保存逐帧独立几何、避免融合为单一全局点云的原因。

#### 10.3.3 ViPE：真正容易被忽略的数据基础设施

ViPE 并非单个端到端神经网络，而是 learned front-end、动态处理、稠密 BA 与深度尺度对齐组成的混合引擎。它从普通视频估计相机内参、运动和近 metric 稠密深度，支持透视、广角和 360° 相机，并以约 3–5 FPS 处理标准输入。更重要的是，它被用于构建约 96M 帧几何标注。

在 Lyra 路线中，ViPE 的价值主要发生在训练数据侧：没有可靠轨迹和深度，就无法训练相机控制、空间检索或前馈重建。因此“视频世界模型的数据规模”背后还依赖一个常被忽略的几何标注系统。

#### 10.3.4 Mamba-2：Lyra 1.0 的效率来自哪里

Mamba-2 通过 Structured State Space Duality 把 selective SSM 与一类 structured masked attention 统一为 semiseparable matrix 乘法，并用 SSD 分块算法在序列递推与 GPU 矩阵乘之间取得折中。Lyra 1.0 要融合 6 条轨迹、每条 121 帧的 latent；如果全程使用全局 Transformer，像素空间版本直接 OOM、纯 Transformer latent 版本也明显更慢。混合 block 让少量 Transformer 负责跨轨迹全局交换，Mamba-2 负责大部分线性序列混合。

需要明确：Mamba-2 不懂深度、相机或高斯。它只是让大量视频 latent 的融合更高效，三维归纳偏置来自 Plücker 相机编码、训练监督和 3DGS 输出头。

### 10.4 没有单独扩写为精读的次级依赖

- **DL3DV、RealEstate10K、Tanks and Temples**：它们是数据集或 benchmark，不是基础模型；已在各论文实验章节中说明。后续若做“世界模型数据引擎”专项综述，再单独建立数据集笔记更合适。
- **Mega-SAM、TSDF fusion、OpenVDB/FVDB**：分别是离线相机/几何估计和三维工程组件，对单篇系统重要，但不是 11 篇共同底座。
- **CLIP、T5/umT5、Qwen 标注模型**：属于通用语义编码与数据标注组件；在本路线中通常冻结使用，不决定空间记忆或三维一致性的核心机制。
- **Cosmos**：GEN3C 用它验证方法可迁移到更强视频 backbone，但 GEN3C 主方法并不依赖 Cosmos；因此仍作为扩展实验处理，而不是本轮必补核心精读。
- **Video Depth Anything**：FantasyWorld 借鉴其时间深度损失思想，但未直接将其作为主干。

### 10.5 基础依赖的最短阅读顺序

若只想理解 Lyra 2.0，建议先读：

1. [[Wan2.1 - 开源大规模视频生成基础模型]]：理解视频 DiT、causal VAE 与 flow matching；
2. [[Depth Anything V2 - 精确合成监督与真实伪标签桥接]] 与 [[DA3 - 任意视角恢复视觉空间]]：理解几何缓存和最终重建；
3. [[DMD - 分布匹配蒸馏一步图像生成]]：理解 35 步教师为何能压成 4 步；
4. [[3DGS - 实时辐射场渲染的三维高斯表示]]：理解最终输出究竟是什么；
5. 最后回到 [[Lyra 2.0 - 可探索的生成式三维世界]]。

若想理解完整系列，则在 Lyra 1.0 前补读 [[Stable Video Diffusion - 大规模潜视频扩散模型]]、[[ViPE - 视频位姿与稠密几何标注引擎]] 和 [[Mamba-2 - 状态空间对偶与高效序列建模]]；想理解 SPMem/FantasyWorld 的几何前端，再读 [[CUT3R - 带持久状态的连续三维感知]] 与 [[VGGT - 前馈式通用三维重建]]。
