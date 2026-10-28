---
type: paper-note
status: complete
domain: World-Models
paper: "DreamX-World 1.0: A General-Purpose Interactive World Model"
year: 2026
arxiv: "2606.16993"
doi: null
source: https://arxiv.org/abs/2606.16993
project: https://amap-ml.github.io/DreamX_World/
code: https://github.com/AMAP-ML/DreamX-World
code_commit: a1f4c6e5e45600718e5236955f2e0702e53fc275
tags:
  - World-Models
  - Video-Generation
  - Interactive-World
  - Long-Horizon
  - Scene-Memory
  - Camera-Control
  - Diffusion-Distillation
created: 2026-10-28
updated: 2026-10-28
---

# DreamX-World 1.0：长程记忆与实时交互世界模型

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | **DreamX-World 1.0: A General-Purpose Interactive World Model** |
| 团队 | DreamX Team（高德 AMAP-ML） |
| 时间 | arXiv v1，2026-06-15 |
| 基座 | Wan2.2-TI2V-5B |
| 任务 | 文本/图像条件的长程交互式视频世界生成；支持 6-DoF 相机导航、场景重访与可提示事件 |
| 模型 | DreamX-World-1.0-5B；另发布双向短视频版 DreamX-World-5B-Cam |
| 论文 | [arXiv](https://arxiv.org/abs/2606.16993) · [PDF](https://arxiv.org/pdf/2606.16993) |
| 项目 | [官方展示页](https://amap-ml.github.io/DreamX_World/) · [在线体验](https://www.dreamx-world.com/) |
| 代码 | [GitHub](https://github.com/AMAP-ML/DreamX-World)，审阅 commit `a1f4c6e` |
| 权重 | [DreamX-World-5B](https://huggingface.co/GD-ML/DreamX-World-5B) · [DreamX-World-5B-Cam](https://huggingface.co/GD-ML/DreamX-World-5B-Cam) |
| 开源判断 | **官方权重与推理代码已开源，但只覆盖双向相机控制和 AR/KV-cache 推理；完整训练与实时部署栈未开源** |
| 评分 | **★★★★☆（4.5/5）**：全栈系统设计与代码可读性强，但核心训练消融、数据规模和完整复现链路不足 |

## 一句话结论

DreamX-World 1.0 的核心价值不是提出一个单点网络模块，而是把**多源相机数据、E-PRoPE 几何控制、基于相机重叠的长期记忆、DMD-forcing 少步自回归、RL 对齐和系统级部署加速**串成完整交互式视频世界模型：5B 模型在 5 秒评测达到 84.76 overall、在约 30 秒评测达到 70.41，并在 8×RTX 5090 的未完整开源部署栈上宣称最高 16 FPS。真正的短板也同样明显：论文最关键的 memory、E-PRoPE 压缩、DMD/RL 训练和实时 serving 并不在公开代码中。

![](assets/DreamX-World%201.0%20-%20长程记忆与实时交互世界模型/figures/01-teaser.png)

> 图 1：DreamX-World 1.0 同时覆盖写实、游戏和风格化世界，并将相机动作作为持续交互接口。图中展示的是“能生成什么”，但不能单独证明其几何或物理一致性。  
> 来源：论文 Figure 1，第 1 页，[arXiv PDF](https://arxiv.org/pdf/2606.16993)。

---

## 1. 研究背景与问题定义

### 1.1 从“视频生成”到“可进入的世界”

普通视频生成模型解决的是 $p(V\mid c)$：给定文本或首帧条件 $c$，一次性采样一段视觉上合理的视频 $V$。交互式世界模型则要求条件持续进入生成循环：用户在时刻 $t$ 给出相机动作、事件指令或智能体行为 $a_t$，模型必须根据既有历史 $V_{\le t}$ 生成未来观测 $V_{>t}$。更接近的目标是

$$p(V_{t+1:t+k}\mid V_{\le t},a_{t:t+k},c,m_t),$$

其中 $m_t$ 是超出局部上下文的长期场景记忆。这个条件分布不仅要“看起来像”，还要满足三项额外约束：

1. **控制可执行**：规定的相机轨迹必须变成正确的视点变化；
2. **世界可持续**：离开视野后再回来，布局、物体身份和外观不能凭空改写；
3. **延迟可交互**：生成必须以流式 chunk 前进，而不是等待整段扩散完成。

### 1.2 三个耦合瓶颈

第一，相机几何条件若在全分辨率视频 token 上额外做一套 attention，会近似把 DiT attention 成本翻倍。第二，自回归视频把模型自己的输出再次喂回模型，微小误差会变成 exposure bias，并在 chunk 之间积累为颜色、风格、背景和身份漂移。第三，几步蒸馏能降延迟，却通常牺牲画质、运动多样性和相机控制；因此“少步”与“稳定”不能只靠一次蒸馏同时解决。

### 1.3 核心贡献

1. **全栈数据引擎**：统一 UE5 精确相机数据、游戏录屏和真实视频的恢复位姿，用同一相机坐标表示训练跨域世界生成。
2. **E-PRoPE**：把相机投影几何 attention 放到空间降采样 token 上，论文中将 18,480 token 压到 4,096，保持相机控制的同时把 5 秒 720P 延迟从 80 秒降到 59 秒。
3. **Memory-Conditioned Scene Persistence**：按相机几何重叠检索远期历史帧，将 memory、recent history 与 target 一起放入 self-attention，并用 residual recycling 适应有误差的生成记忆。
4. **少步长程生成**：用 causal forcing、DMD-style distillation、长 rollout 和局部时间窗监督，把双向模型转成几步自回归世界模型。
5. **对齐与部署**：以相机和画质 reward 做 DiffusionNFT 式后训练，再结合 INT8/FP8、残差复用、75% 剪枝 VAE 和异步流水线，报告最高 16 FPS。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与控制接口

模型输入由四类信号构成：

| 输入 | 形式 | 作用 |
|---|---|---|
| 文本 | 场景描述或结构化事件转写的自然语言 | 规定世界外观、实体与事件 |
| 首帧图像 | 单张 RGB 图像 | 锚定 I2V 第一 chunk 的世界外观 |
| 相机动作 | `WASD` 平移、`IJKL` 旋转及组合动作 | 生成 6-DoF 相机轨迹与相对 pose |
| 生成历史/记忆 | 最近 latent、检索到的远期 latent、rolling KV cache | 维持短期连续性与长期重访一致性 |

输出是按 chunk 自回归延长的 RGB 视频。论文宣称稳定生成可达约 1 分钟；官方开源 AR shell 的实际默认值为 252 个 latent frames，经 4 倍时间解压得到 1005 个像素帧，在 16 FPS 下约 62.8 秒。

### 2.2 世界状态并非显式三维表示

DreamX-World 仍是**视频 latent 世界模型**。相机矩阵为生成提供几何条件，记忆检索也依赖几何线索，但系统没有维护 mesh、point cloud、3DGS 或可查询的显式对象状态。世界状态隐含在视频 latent、KV cache 和检索帧中。因此它更准确地属于“**几何条件化的视频世界模型**”，而不是“显式三维模拟器”。

### 2.3 训练目标

记忆阶段把三类 latent 沿 token 维拼接：

$$z_{\mathrm{pack}}=[z_{\mathcal M}\mid z_{\mathcal H}\mid z_{\mathcal C}^{\tau}],$$

其中 $z_{\mathcal M}$ 是远期记忆，$z_{\mathcal H}$ 是近期历史，$z_{\mathcal C}^{\tau}$ 是噪声等级为 $\tau$ 的目标窗口。三者进入同一 DiT self-attention，但 rectified-flow 损失只作用于目标 token。这个不对称监督很关键：memory/recent 是条件，不应被当作待重建目标。

相机控制误差定义为

$$e_{\mathrm{camera}}=\sqrt{e_\theta\cdot e_t},$$

其中 $e_\theta$ 与 $e_t$ 分别是尺度不变旋转与平移误差。几何均值避免某一项极差时被另一项的好结果简单抵消。

---

## 3. 核心方法

### 3.1 总体框架：数据—训练—部署的递进链路

![](assets/DreamX-World%201.0%20-%20长程记忆与实时交互世界模型/figures/02-system-overview.png)

> 图 2：左侧数据系统把 UE、游戏和真实视频统一为训练数据；中间按 camera → memory → event/RL → AR/distillation 递进适配；右侧把相机/提示控制接入流式推理。DreamX-World 的创新主要来自模块协同，而非单个 end-to-end loss。  
> 来源：论文 Figure 2，第 2 页，[arXiv PDF](https://arxiv.org/pdf/2606.16993)。

论文的训练顺序不是随意拼接。先学相机控制，模型才知道“动作意味着什么视点变化”；再加记忆，模型才能在几何可控的轨迹上定义“离开—重访”；事件 tuning 放在基础世界先验之后，避免对象级指令破坏画质与运动；最后再做少步 AR 蒸馏和 RL，解决部署效率和蒸馏退化。

### 3.2 多源数据引擎

#### 3.2.1 UE5：精确控制但域有限

UE 子集包含第一人称、第三人称和事件三种观察模式。每帧有键盘式动作向量、相机位置和 Euler 角；第三人称额外保存角色世界坐标与朝向。轨迹采集和高质量渲染被拆成在线探索与离线 Movie Render Queue 两阶段：前者用 NavMesh、碰撞检测、最短时长/路径约束和 stuck detection 丢弃无效轨迹，后者只对通过验证的轨迹做昂贵渲染。

这类数据提供准确几何，但视觉域容易受 UE 资产与渲染器偏置限制。论文因此不把 UE 当作唯一数据源，而是让它承担“控制可监督性”的角色。

#### 3.2.2 真实视频与游戏数据：补覆盖但需恢复几何

真实视频来自 SpatialVID、RealEstate10K、Sekai 和 DL3DV。系统先用 MegaSaM 在关键帧恢复稀疏 pose，再用 SLERP 插值旋转、线性插值平移。游戏数据来自 Sekai-Game 与 OmniWorld-Game，并把引擎导出的 pose 统一到同一相机坐标系。

数据清洗分三层：

1. 基础质量：时长、帧率、OCR 文字、黑边与首尾 CLIP 相似度；
2. 几何质量：内参不一致、平移尖峰、急转、垂直抖动与非法朝向；
3. 属性标注：美学、运动强度、场景、风格、主体，以及“只有相机动”的 3D 与“相机+物体都动”的 4D 分类。

**关键缺口**：论文没有披露数据规模、各源比例、采样策略和每阶段训练时长，这是复现性最弱的一环。

### 3.3 E-PRoPE：以低成本保留相机投影几何

PRoPE 为每个 token 构造由投影几何和标准 RoPE 组成的块对角矩阵：

$$D_s^{\mathrm{PRoPE}}=\begin{bmatrix}D_s^{\mathrm{Proj}}&0\\0&D_s^{\mathrm{RoPE}}\end{bmatrix}.$$

$D_s^{\mathrm{Proj}}$ 编码相机内外参与世界到图像投影关系，$D_s^{\mathrm{RoPE}}$ 提供相机无关的 token 位置。DreamX-World 的判断是：原 DiT self-attention 已有足够细粒度时空偏置，额外相机分支主要需要视点相关的高层语义。因此 E-PRoPE 做三件事：

1. 沿空间维下采样相机分支 token；
2. 把 Q/K/V 投影到更低维 $d'<d$；
3. 只保留 $D_s^{\mathrm{Proj}}$，省掉重复的 RoPE 子块，输出再上采样并残差加回原 attention。

论文举例：5 秒 720P 的 Wan2.2 5B latent 有 $S=18{,}480$ 个 token，E-PRoPE 用 $N=4{,}096$ 个 token，空间 token 数降超过 4.5 倍；训练时间约减 50%，推理约减 30%。初始 camera stage 冻结 DiT 主干，只训练 E-PRoPE 参数，这等价于给基座加一个几何 adapter。

**需要特别警惕**：官方开源 5B-Cam 代码没有实现论文所述 $18{,}480\rightarrow4{,}096$ 的空间 token 下采样/上采样；它仍对全 token 运行平行 PRoPE attention。AR 配置中的 `attn_compress=4` 压缩的是 Q/K/V 通道，不是 token 数。因此论文 E-PRoPE 的核心效率机制目前无法从开源代码复现。

### 3.4 Memory-Conditioned Scene Persistence

![](assets/DreamX-World%201.0%20-%20长程记忆与实时交互世界模型/figures/03-memory-persistence.png)

> 图 6：系统依据目标相机与历史视锥重叠检索远期 memory，将其与 recent history、noisy target 一起送入 self-attention；Error Bank 把历史预测误差施加到条件 token，却保持监督目标干净。  
> 来源：论文 Figure 6，第 7 页，[arXiv PDF](https://arxiv.org/pdf/2606.16993)。

#### 3.4.1 为什么不是“取最近帧”

时间距离近不等于空间相关。相机可能直线前进很久后掉头，真正相关的是很早之前看过的区域。DreamX-World 按相机 pose 与视野重叠检索，而非固定时间间隔。被检索记忆仍使用其**原始视频时间位置**对应的 RoPE，避免模型误把远期视图当成目标的紧邻帧；大时间跨度参考 NTK-aware scaling、YaRN 和 randomized positional encoding 做轻量扩展。

#### 3.4.2 为什么采用 self-attention packing

memory、recent、target 放到同一 self-attention 流，使目标 token 可以直接读取非局部视觉证据。作者试过 cross-attention 和 VACE-style 条件注入，但只定性称生成质量更差，没有给出数值消融。这个选择符合一个直觉：记忆不是外部语义，而是与当前视频 latent 同构的视觉 token，让它们共享 attention 空间比另建 cross-attention 接口更自然。

#### 3.4.3 residual recycling 解决 exposure bias

训练条件来自真实视频，推理条件却是模型自己生成的 latent。若只在干净条件上训练，模型会过度信任记忆，一旦记忆有错就放大错误。Residual recycling 保存历史预测误差并注入 memory/recent 条件，但目标仍保持干净监督。模型由此学会一种软门控：记忆可靠时利用它，误差明显时退回生成先验。

这里值得强调：它提高的是**对错误条件的鲁棒性**，并不保证记忆内容本身正确；长程布局仍可能漂移，论文也在限制中明确承认这一点。

### 3.5 Event Instruction Tuning

事件标注采用两层结构：全局描述总结静态背景和整体演化；实体级记录包含 entity reference、event predicate、spatial anchor 和 temporal interval。多对象事件为每个实体分配独立记录，并在全局描述中明确碰撞、交接等对象间关系。

模型架构不变，结构化记录先转成自然语言，只通过 text-conditioning 进入 DiT；随后微调整个 DiT，并把事件样本与普通视频混合。保守更新和严格 gradient clipping 的目的，是避免小规模事件数据把原有画质和运动先验冲坏。

论文表 2 显示 DreamX-World 是对比系统中唯一宣称同时完整支持 promptable、object-level、region-guided、multi-entity composition 和 inter-object interaction 的模型。但该表是能力勾选，不是受控定量基准；事件成功率、指令跟随率和组合泛化没有给出数字，所以它更多证明“接口覆盖”，而非严格证明“事件能力领先多少”。

### 3.6 DMD-forcing：从双向视频生成器到少步 AR 世界模型

![](assets/DreamX-World%201.0%20-%20长程记忆与实时交互世界模型/figures/04-dmd-forcing.png)

> 图 7：AR student 在长 rollout 中产生 clean latent，再从中抽局部 DMD window；双向 E-PRoPE teacher 对真实与生成分布打分并形成 DMD loss。I2V 时还把窗口首 latent 解码为图像，作为 teacher 的首帧条件。  
> 来源：论文 Figure 7，第 8 页，[arXiv PDF](https://arxiv.org/pdf/2606.16993)。

直接把双向扩散模型改成 causal attention 并少步采样会产生两重错配：teacher 看得到窗口两侧，student 只能看历史；teacher 训练在真实上下文，student 推理在自己生成的上下文。论文用 causal forcing 和长 student rollout 让训练上下文来自模型自身，再在长序列中抽局部时间窗做 DMD，使 student 分布向双向 teacher 靠拢。

长程适配还引入 Infinity-RoPE，缓解位置超过训练窗口后的外推问题。相机控制版把 E-PRoPE 分支加入 AR student，并重复长视频 DMD 训练，以修复 chunk-wise 生成造成的运动不平滑与控制衰减。

对 I2V，每个 DMD window 的首 latent 先经 VAE 解码成图像，再作为 bidirectional teacher 的图像条件。这样 teacher 和 student 在局部窗口上共享可观察首帧锚点，而不是让 teacher 凭空评估一段截断 latent。

### 3.7 RL 对齐：修复少步蒸馏后的能力回退

每个 text-image-camera 条件先生成多个长 rollout，再从完整 rollout 中抽短 clip 计算两类 reward：水平平移/旋转准确度，以及视觉质量。DiffusionNFT 对短窗口做软更新，KL 正则让新模型靠近 DMD 模型。rollout horizon 与反向传播窗口解耦，既让 reward 看到长期 AR 状态，又避免整分钟视频反传爆显存。

作者强调逐步、保守更新，因为几步模型的自由度比多步扩散小，强 reward 容易导致模式坍缩或训练崩溃。遗憾的是论文没有提供 RL 前后、DMD 前后的定量对照，无法拆出 RL 对 camera/quality 的独立收益。

### 3.8 流式推理与系统加速

流式模型每个 chunk 从噪声开始，在文本、chunk-relative pose 和 rolling KV cache 条件下做少步去噪；完成后把生成 token 写回 cache。I2V 只在第一 chunk 特殊：首帧 latent 被替换成输入图像 latent。第一 chunk 的 pose 相对首帧，后续 chunk 相对上一 chunk 最后一帧，防止绝对 pose 数值随时间增长削弱条件。

论文的实时部署包含：

- attention：INT8 SageAttention；
- FFN：AngelSlim FP8；
- DiT：sequence parallel、Triton 融合 kernel、TeaCache 风格 residual reuse；
- VAE：Matrix-Game 3.0 decoder，75% pruning，单 chunk 约 0.25 秒，`torch.compile` 与 ParaVAE 高度切分；
- serving：chunk $k$ 的 VAE decode 与 chunk $k+1$ 的控制接收、KV 更新和 DiT 去噪异步重叠。

8×RTX 5090 上报告最高 16 FPS，但论文没有同时披露该数字对应的分辨率、chunk 大小、去噪步数与端到端延迟分解；公开仓库也没有完整部署实现。因此它应被理解为**定制系统峰值吞吐**，不能等同于公开代码“开箱即 16 FPS”。

---

## 4. 数据集与实验设置

### 4.1 Baseline 与评测维度

主要对比 HY-WorldPlay 1.5（8B）和 LingBot-World（14B），均为公开交互式世界模型。评测分四部分：

1. 5 秒基础评测：相机控制、画质、transition、flicker、smoothness、dynamic、artifact；
2. 约 30 秒长程评测：同一组指标观察 AR 退化；
3. 10 秒重访一致性：像素、感知、语义、地点识别、局部几何与时序平滑；
4. 双盲人类偏好：overall、camera、visual quality、artifact。

### 4.2 重访一致性协议

重访对 $(i,j)$ 满足

$$|\theta_i-\theta_j|\le\tau_\theta,\qquad\|\mathbf t_i-\mathbf t_j\|_2\le\tau_t,$$

其中 $\tau_\theta=2^\circ$、$\tau_t=0.1$，并要求 $|j-i|\ge\lfloor0.2T\rfloor$。多个候选中选最小加权距离

$$|\theta_i-\theta_j|+10\|\mathbf t_i-\mathbf t_j\|_2.$$

轨迹包含 out-and-back、translation-rotation 和 closed-loop。为避免慢速相机天然得到高相似度，除 CLIP-V 外均报告重访对相对“相同时间间隔、但非重访”的 baseline gain：相似度指标用 $S_{\mathrm{revisit}}-S_{\mathrm{baseline}}$，LPIPS 反向。

这套协议是论文较扎实的贡献，因为它明确把“画面平滑”与“回到同一地点后还是同一世界”分开。

### 4.3 Artifact 指标与潜在偏差

作者用 Gemini-3.1-Pro 进行 artifact detection：2 FPS 采样，每帧输出 pass/fail，每个样本评两次取平均。它能覆盖重复肢体、物体瞬间消失和几何穿透，但也引入闭源 judge 的版本、提示词和随机性依赖。论文没有公开完整 prompt、置信区间或 judge 校准，因此 artifact 数字更适合作为相对信号，而非可长期复现的绝对标尺。

### 4.4 未披露的关键训练配置

论文没有给出总数据量、各源配比、分辨率/帧率混合策略、batch size、optimizer、learning rate、各阶段 step、DMD 具体步数、RL reward 权重、KL 系数、gradient clipping 阈值和人评规模。即使权重与推理代码已开源，训练 recipe 仍无法复现。

---

## 5. 实验结果

### 5.1 E-PRoPE 效率消融

在 8×H20、5 秒、1280×720 的表 1 设置下：

| 模型 | Camera ↑ | Quality ↑ | Dynamic ↑ | Transition ↑ | Flicker ↑ | Smooth ↑ | Latency ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRoPE | 73.89 | 66.15 | 87.50 | 96.67 | 96.02 | 98.65 | 80 s |
| E-PRoPE | 73.75 | 66.75 | 85.83 | 98.33 | 96.17 | 98.79 | 59 s |

相机控制只下降 0.14，而延迟下降 21 秒，即按表值约 26.25%；论文概括为约 30%。Dynamic 从 87.50 降到 85.83，说明压缩不是完全无损，但整体 trade-off 很好。问题在于只有一个压缩点，没有 token 数/通道宽度的系统曲线。

### 5.2 5 秒基础评测

| 模型 | 参数 | Camera | Quality | Trans. | Flicker | Smooth. | Dynamic | Artifact | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HY-WorldPlay 1.5 | 8B | 65.12 | **68.23** | **98.33** | **96.45** | **99.05** | 66.67 | 71.66 | 80.79 |
| LingBot-World | 14B | 71.73 | 67.76 | 85.00 | 94.94 | 97.06 | **88.33** | 58.33 | 80.45 |
| **DreamX-World-1.0-5B** | **5B** | **73.75** | 66.75 | **98.33** | 96.17 | 98.79 | 85.83 | **73.75** | **84.76** |

DreamX 的优势来自均衡性：Camera、Artifact 和 Overall 第一，但纯 Quality 不及 HY-WorldPlay，Dynamic 不及 LingBot。换言之，论文支持“综合最好”，不支持“每一维都最好”。

### 5.3 约 30 秒长程评测

| 模型 | Camera | Quality | Trans. | Flicker | Smooth. | Dynamic | Artifact | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HY-WorldPlay 1.5 | **65.86** | 63.02 | **91.00** | **97.00** | **99.11** | 52.00 | 14.00 | 68.85 |
| LingBot-World | 63.76 | 60.81 | 54.00 | 96.59 | 97.86 | **87.00** | 12.00 | 67.43 |
| **DreamX-World-1.0-5B** | 62.03 | **64.11** | 80.00 | 96.35 | 98.41 | 75.00 | **17.00** | **70.41** |

长程 Overall 仍第一，但 Camera 从 5 秒的 73.75 降到 62.03，并落后于两个 baseline；这直接说明 chunk-relative pose、long rollout 和 RL 并没有消除长期控制衰减。Artifact 绝对值也从 73.75 降到 17.00，表明 30 秒仍是明显困难区间。

### 5.4 定性结果

![](assets/DreamX-World%201.0%20-%20长程记忆与实时交互世界模型/figures/05-qualitative-results.png)

> 图 10：五组不同风格场景的均匀关键帧显示较稳定的外观、运动方向与视角变化。它证明模型覆盖域很广，但挑选样例不能替代失败率统计，尤其无法观察视频中间的瞬时 artifact。  
> 来源：论文 Figure 10，第 11 页，[arXiv PDF](https://arxiv.org/pdf/2606.16993)。

### 5.5 重访一致性

![](assets/DreamX-World%201.0%20-%20长程记忆与实时交互世界模型/figures/06-memory-results.png)

> 表 5：DreamX 在 $\Delta$PSNR、$\Delta$SSIM、$\Delta$LPIPS、$\Delta$DINO-Sim 和 $\Delta$VPR-Sim 上领先，说明从像素到地点身份的重访收益更高；HY-WorldPlay 在局部 SuperPoint 匹配与 CLIP 时序平滑上更高。  
> 来源：论文 Table 5，第 14 页，[arXiv PDF](https://arxiv.org/pdf/2606.16993)。

具体看，DreamX 的 $\Delta$PSNR=3.92、$\Delta$LPIPS=0.232、$\Delta$DINO-Sim=0.246、$\Delta$VPR-Sim=0.142 均第一；但 $\Delta$SP-Match=0.216 低于 HY-WorldPlay 的 0.251，CLIP-V=0.991 也略低于 0.992。这意味着 DreamX 对“整体场景是什么、看起来是否像原地点”更强，局部几何特征对应却未全面领先，不能据此声称已建立精确三维地图。

### 5.6 人类偏好

DreamX 对 HY-WorldPlay 的 overall win/tie/lose 为 57.5/14.4/28.1，对 LingBot 为 61.9/10.6/27.5；画质胜率分别 57.5% 与 61.3%，artifact 胜率 59.4% 与 56.2%。相机控制对 HY-WorldPlay 为 37.5/31.9/30.6，对 LingBot 为 31.9/35.6/32.5，基本接近。

论文没有报告评审人数、样本数、每样本评审数和置信区间。偏好方向与自动指标一致，但统计强度不可判断。

### 5.7 消融是否充分

不充分。唯一完整量化的模块消融是 PRoPE vs E-PRoPE。论文没有给出 memory on/off、residual recycling on/off、event tuning、long-rollout、DMD 前后、RL 前后，或 INT8/FP8/TeaCache/VAE pruning/ParaVAE 各自的独立贡献。对于一个强调“全栈协同”的系统，这使读者无法判断 84.76/70.41 到底由哪些模块主导。

---

## 6. 与相关工作的关系

DreamX-World 可以放在四条技术线的交汇处理解：

1. **相机可控视频**：从 MotionCtrl/CameraCtrl 的显式控制，走到 AC3D/PRoPE 的投影几何 conditioning；E-PRoPE 是效率版几何 attention。
2. **自回归长视频**：延续 Self-Forcing、LongLive、Infinity-RoPE、Causal Forcing 的“在模型自身历史上训练 + 少步 AR”路线。
3. **长期记忆**：与 RELIC、Context as Memory、WorldPlay 等工作一样，解决超出局部窗口后的重访；DreamX 的特色是相机几何检索 + 同流 token packing + error recycling。
4. **生成模型 RL**：借 DiffusionNFT/WorldCompass/Astrolabe 的短窗口 reward、长 rollout 上下文和 forward-process 更新修复蒸馏退化。

相较显式 3D/4D 世界模型，DreamX 的优势是能直接继承强视频生成基座、视觉域宽、渲染成本低；劣势是没有可查询、可编辑、可验证的显式场景状态，几何与物理只能通过观测一致性间接评估。

---

## 7. 局限与批判性评价

### 7.1 作者明确承认的局限

1. 长时间交互后，物体外观和场景布局仍可能显著漂移；
2. caption、camera 与 event 可能冲突，模型缺少显式冲突解析；
3. 自动指标不能完整覆盖开放式交互，仍需人类和任务评测。

### 7.2 论文证据链的主要缺口

- **数据不可审计**：无总量、配比、许可与过滤存活率；
- **训练不可复现**：关键超参和训练代码缺失；
- **核心模块缺消融**：memory、residual recycling、DMD 与 RL 的因果贡献无法拆分；
- **实时数字口径不完整**：16 FPS 缺分辨率、chunk 和延迟分解，且部署代码未开源；
- **评测自建成分高**：artifact 依赖 Gemini-3.1-Pro，事件能力主要是勾选表；
- **世界状态仍是像素 latent**：重访一致不等于真正掌握可组合的三维/物理状态。

### 7.3 我的判断

这篇工作最强的地方是**工程系统观**：它诚实地把交互式世界模型视为数据、几何控制、记忆、蒸馏、对齐和 serving 的联合问题，而不是把单个模块包装成万能解法。最需要降温的地方是“实时”和“记忆”两个词：实时依赖未开源的 8 卡定制栈；记忆评测显示高层重访一致性改善，但局部几何匹配并非第一，且作者自己承认长程布局仍会漂移。

因此它更像一个“**高完成度的通用交互视频世界系统**”，还不是具备稳定因果状态、精确几何和可验证物理规则的世界模拟器。

---

## 8. 复现与实践建议

### 8.1 最小可跑路径

若只想验证相机控制，优先跑 DreamX-World-5B-Cam：官方 shell 默认 704×1280、81 帧、16 FPS、50 steps、CFG 3、单 GPU，README 报告 1×H20 约 509 秒/38 GB。它慢但逻辑简单，适合先检查动作到 pose、PRoPE 与 VAE 输出。

若验证长程 AR，使用 DreamX-World-5B：1×H20 5 秒约 26 秒/40 GB，60 秒约 342 秒/72 GB。显存低于约 40 GB 时，代码会启用 low-memory/offload 路径。依赖包括 PyTorch 2.5.1、diffusers、transformers、xfuser、FlashAttention 2.8.3 与 Triton 3.1.0。

### 8.2 复现时必须显式记录的参数

- 分辨率、像素帧数、latent 帧数与 FPS；
- `num_output_frames`、每 block latent 帧数；
- `action_speed_list` 的入口语义；
- 是否使用 chunk-relative pose；
- 去噪 timestep、context noise 与 color correction；
- GPU 数与实际是否初始化 distributed；
- base Wan2.2、T5、VAE 与 DreamX 权重版本。

尤其要注意三组不一致：AR shell 默认 252 latent frames，README 示例 123，HF 1 分钟示例 243；5B-Cam shell 默认 81 帧/16 FPS/1 GPU，而 README 参数块是 121 帧/24 FPS/8 GPU；同一 `action_speed_list` 在双向入口表示运动幅度，在 AR 入口表示动作段时长比例。

### 8.3 研究复现的优先补充实验

1. memory/recycling 的 2×2 消融；
2. E-PRoPE token 数与通道压缩的效率—控制曲线；
3. RL 前后 camera、quality、diversity 与 mode collapse；
4. 30 秒、1 分钟、多次闭环下的分段退化曲线；
5. 用几何重建误差或跨视图 correspondence 补足高层语义指标；
6. 在同一公开硬件/分辨率/步数下对比端到端交互延迟。

---

## 9. 个人启发与后续问题

### 9.1 对世界模型研究的启发

DreamX 提醒我们，长程世界一致性至少有三种不同状态：

- **短期动力学状态**：rolling KV cache 最擅长；
- **长期视觉证据**：几何检索 memory 负责；
- **显式可组合世界状态**：DreamX 尚未建立。

后续可以把检索到的历史帧先转成显式 object/geometry state，再以可微渲染或 state token 条件化视频 DiT。这样 memory 不只回答“以前长什么样”，还能回答“对象是谁、在哪、状态如何变化”。

### 9.2 与 3D/4D 表示结合

相机几何只被用于 token attention 与检索，没有形成闭环地图。一个自然方向是维护轻量 3DGS/feature field 作为外部记忆：视频模型负责生成和补全，几何记忆负责重访约束、遮挡推理与可查询状态。关键问题是如何避免把生成错误永久写回地图；DreamX 的 residual recycling 思路可以迁移成带不确定性的 memory write gate。

### 9.3 可检验的后续问题

1. E-PRoPE 的有效性究竟来自投影几何，还是来自额外 attention 参数量？
2. 几何检索若 pose 本身由生成视频估计并逐渐漂移，会不会形成错误闭环？
3. Event instruction 只走文本接口，如何处理持续状态、事件互斥和因果顺序？
4. RL 的相机 reward 会不会鼓励低动态、容易估姿的保守视频？
5. 若用 Orbia 等三维一致性 benchmark，DreamX 的高层重访优势能否转化为可重建几何？

---

## 10. 代码级实现

### 10.1 仓库概览与开源边界

官方仓库 commit 为 `a1f4c6e5e45600718e5236955f2e0702e53fc275`。GitHub 代码为 Apache-2.0；HF 模型卡标 MIT，且运行还依赖 Wan2.2 组件，需分别核对条款。

公开内容主要是两个推理快照：

- `DreamX-World-5B-Cam`：双向 50 步 flow matching，生成短视频；
- `DreamX-World-5B`：3 latent-frame/chunk 的 causal AR、4 个去噪等级、rolling KV cache，支持约 1 分钟。

训练数据、训练入口、memory/event/DMD/RL 训练和论文完整实时栈均未开源。

| 论文模块 | 代码入口 | 开源范围 |
|---|---|---|
| 动作到相机轨迹 | `utils/inference_utils.py:242-372`；`utils/trajectory_processor.py:827-958` | 已开源推理 |
| 双向相机模型 | `inference_dreamx5b.py`；`pipeline/pipeline_dreamxworld.py` | 已开源 |
| PRoPE | `models/prope_utils.py:61-195`；`models/wan_transformer3d.py:222-267` | 全 token PRoPE 已开源 |
| 论文版 E-PRoPE 空间压缩 | 无 $18{,}480\rightarrow4{,}096$ 实现 | 未开源 |
| AR chunk 生成 | `inference_ar_forcing.py`；`pipeline/pipeline_causal_camera.py:146-233` | 已开源 |
| Infinity/block-relative RoPE | `wan/modules/causal_camera_model_2_2_prope_infinity.py:18-56` | 已开源推理 |
| rolling KV cache | 同文件 `:59-231,664-707` | 已开源 |
| Memory-Conditioned Persistence | 无 | 未开源 |
| Event Instruction Tuning | 无 | 未开源 |
| DMD/causal-forcing 训练 | 无 | 未开源 |
| RL/DiffusionNFT | 无 | 未开源 |
| INT8/FP8、剪枝 VAE、异步 serving | 零散 helper 或无对应路径 | 完整栈未开源 |

### 10.2 全链路张量总表

以下以 5B-Cam shell 实际默认 `704×1280、81 frames、16 FPS` 和 AR 配置为准。

| # | 模块 | 关键输入 | 关键输出 | 依据 |
|---|---|---|---|---|
| 1 | 图像/视频容器 | `[1,3,81,704,1280]` | 首帧有效，其余为占位 | `pipeline_dreamxworld.py` |
| 2 | Wan VAE encode | `[1,3,81,704,1280]` | `[1,48,21,44,80]` | 时间 4×、空间 16× |
| 3 | DiT patchify | `[2,48,21,44,80]`（CFG×2） | `[2,18480,3072]` | patch `[1,2,2]`，$21×22×40$ |
| 4 | self-attention | `[2,18480,3072]` | Q/K/V `[2,18480,24,128]` | 24 heads，head dim 128 |
| 5 | camera PRoPE | 21 个 pose、每帧 880 token | 同序列长度 camera-attention 输出 | 开源代码未做 4096 token 压缩 |
| 6 | 50-step flow | latent `[1,48,21,44,80]` | 同形状 clean latent | CFG=3，FlowMatch Euler |
| 7 | VAE decode | `[1,48,21,44,80]` | `[1,3,81,704,1280]` | 输出短视频 |
| 8 | AR 输入 | `[1,N,48,44,80]` | 每 chunk `[1,48,3,44,80]` | 3 latent frames/chunk |
| 9 | AR patchify | `[1,48,3,44,80]` | `[1,2640,3072]` | $3×22×40=2640$ token |
| 10 | AR 主 KV | 当前 2640 token | 每层 K/V `[1,10560,24,128]` | 12 latent-frame local window |
| 11 | Text KV | UMT5 最长 512 token | 每层 K/V `[1,512,24,128]` | 首次计算后复用 |
| 12 | AR decode | `[1,48,N,44,80]` | `[1,3,(N-1)×4+1,704,1280]` | 252 latent → 1005 pixel frames |

### 10.3 双向 5B-Cam

#### 10.3.1 输入输出与内部张量流

81 像素帧经 VAE 时间压缩：

$$T_{\mathrm{latent}}=1+\frac{81-1}{4}=21.$$

空间 latent 为 $704/16=44$、$1280/16=80$。经 patch `[1,2,2]` 后，token 网格是 $21×22×40$，共 18,480 token。CFG 把正负 prompt 和 latent 复制成 batch 2，UMT5 context 从 4096 维投影到 DiT 3072 维。

相机 pose 从 81 帧插值到 21 帧，形成 `viewmats [2,21,4,4]` 与 `K [2,21,3,3]`。每帧对应 $22×40=880$ 个空间 token。

#### 10.3.2 为什么这样设计

704 而非 720 是为了被 16×VAE 与 2×DiT patch 整除；首帧直接写入 latent 第 0 帧，比额外图像 cross-attention 更贴合 Wan TI2V 接口。PRoPE 与普通 RoPE attention 并行残差相加，使 camera adapter 可在冻结 backbone 时单独训练。

#### 10.3.3 关键超参

- 81 frames / 16 FPS；
- 704×1280；
- 50 steps；
- CFG 3；
- BF16；
- scheduler train steps 1000、shift 5；
- VAE latent channels 48；
- DiT hidden 3072、30 layers、24 heads、FFN 14336。

#### 10.3.4 代码与论文差异

论文 E-PRoPE 先将 18,480 token 压到 4,096，再上采样回原序列；源码 `models/wan_transformer3d.py:222-267` 仍对全序列做 camera attention，HF 5B-Cam config 也写 `prope_compress_mode: none`。因此公开 5B-Cam 更准确地是 PRoPE 推理版，而不是论文完整 E-PRoPE。

### 10.4 长视频 AR-forcing

#### 10.4.1 输入输出与内部张量流

AR shell 默认 `N=252` latent frames：

$$T_{\mathrm{pixel}}=(252-1)×4+1=1005,$$

即 16 FPS 下约 62.8 秒。每次取 3 个 latent frame，输入 `[1,48,3,44,80]`，patch 后 2640 token。相机矩阵在进入 pipeline 前已按 token 展开为 `[1,N×880,4,4]` 与 `[1,N×880,3,3]`，每 chunk 切出 2640 个 pose。

去噪配置写 `[1000,750,500,250]`，启用 warp/shift=5 后实际约映射为 `[1000,937.5,833.3,625]`。每 chunk 做 4 次 denoising forward，再以 `context_noise=0.1` 做 1 次 forward 写 cache，因此 252/3=84 个 chunk 共约 420 次 Transformer forward。

#### 10.4.2 为什么这样设计

3 latent frames 约对应 12 像素帧，是延迟、cache 更新频率与 GPU 利用率之间的折中。每阶段直接预测 clean latent

$$x_0=x_t-\sigma_t v,$$

再把它投回下一噪声等级

$$x_{t'}=(1-\sigma_{t'})x_0+\sigma_{t'}\epsilon,$$

比传统 Euler 多步积分更符合 few-step forcing 的接口。

#### 10.4.3 关键超参

- block 3 latent frames；
- local attention 12 latent frames；
- sink 3 latent frames；
- 4 个 denoising levels；
- context noise 0.1；
- hidden 3072、30 层、24 heads；
- 官方 shell 默认 252 latent frames，README 示例 123，HF 一分钟示例 243。

#### 10.4.4 代码与论文差异

公开 AR 模型能复现 chunk、few-step 与 KV cache 推理，但没有 causal forcing/DMD 的训练 loss、teacher/student rollout 或 RL。README 称多 GPU 会自动启用 sequence parallel，但 AR shell 直接调用普通 Python，入口未初始化 distributed；多张卡不会像双向 5B-Cam 一样自动 `torchrun`。

### 10.5 Rolling KV cache 与 Block-Relative RoPE

#### 10.5.1 输入输出与内部张量流

每 latent frame 880 token，12-frame cache 共 10,560 token；其中前 3 帧是固定 sink（2,640 token），其余 9 帧是滚动 local（7,920 token）。每层 K/V 为 `[1,10560,24,128]`，30 层 BF16 主 cache 理论约 3.625 GiB；text K/V 约 0.176 GiB。

#### 10.5.2 为什么这样设计

固定 sink 保存开头身份/场景锚点，滚动窗口保存近期运动连续性；block-relative RoPE 在 cache 滚动后重新按局部窗口位置编号，避免绝对时间索引无限增长超出训练范围。

#### 10.5.3 关键超参

- sink 3 latent frames；
- local window 12 latent frames；
- text context 512；
- 每层 head 24×128。

#### 10.5.4 代码与论文差异

普通 self-attention 使用 rolling KV，但官方 pipeline 没有初始化 `prope_k/prope_v`，所以 camera PRoPE 只在当前 3-frame chunk 内 attention，并不跨 chunk 保留相机分支 cache。论文的 geometry-guided non-local memory 也完全不在该 cache 逻辑中；不要把 KV cache 误当成论文长期场景记忆。

### 10.6 计算账

| 路径 | 序列/窗口 | 前向次数 | 开源性能口径 | 论文性能口径 |
|---|---|---:|---|---|
| 5B-Cam | 18,480 token，全视频 | 50 | 1×H20，5 秒约 509 s / 38 GB | E-PRoPE 表：8×H20，59 s |
| AR 5B，5 秒 | 21 latent frames，7 chunks | 约 35 | 1×H20，26 s / 40 GB | 未单列 |
| AR 5B，约 60 秒 | 243–252 latent frames，81–84 chunks | 405–420 | 1×H20，342 s / 72 GB | 未单列 |
| 定制实时栈 | chunk-wise | 未披露 | 未开源 | 8×RTX 5090，最高 16 FPS |

这些数字不能横向直接比较：GPU 数、分辨率、实现版本、量化、VAE、残差复用和异步流水线都不同。

### 10.7 复现建议

1. 先跑 21 latent-frame AR 路径，确认 VAE、UMT5、动作轨迹、PRoPE 与 cache 接口；
2. 显式传 `--num_output_frames`，不要依赖三套不同默认值；
3. 记录 `action_speed_list`：AR 中是时长权重，5B-Cam 中是运动幅度；
4. 用 `--chunk_relative` 并可视化每 chunk 的相对 pose，优先排查控制漂移；
5. 显存不足先缩长度，不要先改分辨率，因为 704×1280 在 AR 脚本中被固定；
6. 不要期待公开代码复现 16 FPS；若研究 E-PRoPE，应自行补齐空间 token downsample/upsample，并用表 1 协议重新评测。

---

## 11. 材料来源

完整来源与本地文件映射见：

- [sources.md](assets/DreamX-World%201.0%20-%20长程记忆与实时交互世界模型/sources.md)
- 官方论文：[arXiv 2606.16993](https://arxiv.org/abs/2606.16993)
- 官方代码：[AMAP-ML/DreamX-World](https://github.com/AMAP-ML/DreamX-World)
- 官方项目页：[DreamX-World](https://amap-ml.github.io/DreamX_World/)
- 官方权重：[5B AR](https://huggingface.co/GD-ML/DreamX-World-5B) / [5B-Cam](https://huggingface.co/GD-ML/DreamX-World-5B-Cam)
