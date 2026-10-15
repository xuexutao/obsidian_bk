---
type: survey
status: complete
domain: World-Models
topic: Google DeepMind Genie 1-3
created: 2026-10-15
updated: 2026-10-15
sources:
  - https://arxiv.org/abs/2402.15391
  - https://proceedings.mlr.press/v235/bruce24a.html
  - https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
  - https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
  - https://deepmind.google/models/genie/
tags:
  - World-Models
  - Genie
  - Interactive-Video
  - Latent-Action
  - Autoregressive-Generation
---

# Genie 1–3 架构演进：从无标签视频到实时交互世界

## 基本信息

| 代际 | 发布时间 | 公开材料 | 直接输入 | 官方定位 | 架构披露程度 |
|---|---|---|---|---|---|
| Genie 1 | 2024-02 预印本；ICML 2024 | 20 页正式论文、附录、完整超参数 | 初始图像/若干帧 + 每步离散潜动作；文本通常先经外部文生图模型 | 2D 平台游戏为主的生成式交互环境 | **高**：三大模块、参数、损失、数据和消融均公开 |
| Genie 2 | 2024-12-04 | DeepMind 官方技术博客，无正式论文 | 单张提示图 + 键盘/鼠标动作 | 可生成可玩的多样 3D 世界的基础世界模型 | **中低**：只公开自动编码器、潜扩散、因果 Transformer、CFG 和蒸馏 |
| Genie 3 | 2025-08-05；后续 Project Genie 持续更新 | 官方公告与模型页，无可复现级论文 | 文本世界描述 + 导航动作 + 运行时文本事件 | 通用、实时、720p 的交互世界模型 | **低**：只明确逐帧自回归、长历史条件和无显式 3D；骨干与训练细节未公开 |

> **一句话结论**：三代的真正演进不是简单的“2D→3D→更高清”，而是三个瓶颈依次转移：Genie 1 解决“没有动作标签怎么学控制”，Genie 2 解决“怎样把可控生成扩展到开放视觉域和 3D 观感”，Genie 3 解决“怎样把高质量世界模型做成 720p、20–24 FPS、分钟级一致的在线系统”。但公开透明度反向下降：只有 Genie 1 能做严格的逐层架构复盘。

---

## 1. 先建立正确的阅读框架

### 1.1 Genie 不是传统意义上的游戏引擎

传统引擎保存显式状态：网格、碰撞体、材质、骨骼、物理量和脚本。Genie 的外部可见状态主要是视觉观察；用户给动作，模型生成下一帧或下一观察。到 Genie 3，DeepMind 仍明确强调世界是根据描述、历史轨迹和用户动作**逐帧生成**的，而不是先建立一个 NeRF、Gaussian Splatting、mesh 或可导出的场景图再渲染。

因此，“生成 3D 世界”应读作：模型生成具有第一/第三人称导航、视角变化、遮挡恢复和物体持久性的**3D 观感交互视频世界**。官方没有证明 Genie 2/3 输出显式几何、碰撞信息或可供传统引擎直接编辑的 3D 资产。

### 1.2 三种控制信号不能混为一谈

1. **Genie 1 latent action**：从无动作标签视频中学出的 8 个离散动作码。它是论文最核心的架构创新。
2. **Genie 2 keyboard/mouse action**：官方只说模型逐帧接收动作，没有披露动作标签从哪里来、是否还使用 LAM、动作编码是离散还是连续。
3. **Genie 3 navigation + promptable world events**：导航动作控制主体运动；文本事件改变天气、对象或角色。事件是环境干预，不等于主体能执行任意动作。

后两代没有论文，所以不能把 Genie 1 的“8 个 VQ 动作码 + LAM”自动套到 Genie 2/3。

### 1.3 本文的确定性标记

- **已公开**：可在论文或 DeepMind 官方页面逐字核对。
- **合理判断**：由输入输出和官方描述推出，但不是模块级披露。
- **未公开**：没有可靠官方材料；不使用社区猜测填空。

---

## 2. Genie 1：从无动作视频中发现可玩的控制接口

### 2.1 任务定义与总体架构

Genie 1 的训练数据只有视频帧，没有动作、奖励、状态或文本标注。目标是把视频变成一个可迭代环境：给定历史帧与用户动作，生成下一帧。

![](assets/Genie%201-3%20架构演进%20-%20Survey/figures/01-genie1-overview.png)

> 图 1：Genie 1 的三模块总览。视频同时进入时空视频 tokenizer 和潜动作模型；离散视频 token 与动作 token 再进入动力学模型。来源：Genie 论文 Figure 3，arXiv:2402.15391。

其核心分解是：

$$z_{1:T}=E_{\text{video}}(x_{1:T})$$

$$\tilde a_{1:T-1}=E_{\text{LAM}}(x_{1:T})$$

$$\hat z_t=F_{\text{dyn}}(z_{1:t-1},\operatorname{stopgrad}(\tilde a_{1:t-1}))$$

$$\hat x_t=D_{\text{video}}(\hat z_t)$$

三个组件合计约 10.7B 参数，论文以 **11B Genie** 称呼：视频 tokenizer 约 200M、LAM 约 300M、动力学模型 10.1B。主要容量明显集中在下一帧动力学预测，而不是编码器或动作发现模块。

### 2.2 共同骨干：ST-Transformer

三个组件都使用 Spatial-Temporal Transformer（ST-Transformer）。一个 block 依次做：

1. 单帧内部的 spatial self-attention；
2. 同一空间位置跨时间的 causal temporal self-attention；
3. 一个 feed-forward layer。

![](assets/Genie%201-3%20架构演进%20-%20Survey/figures/03-genie1-st-transformer.png)

> 图 2：ST-Transformer 将全时空注意力拆成空间与时间两轴。空间注意力处理每帧的 $H\times W$ token，时间注意力处理同一位置的 $T$ 个 token。来源：论文 Figure 4。

与对所有 $T\times H\times W$ token 做全注意力相比，这种分解使主要的空间注意力开销随帧数近似线性增长。论文还省略 spatial attention 后单独的一次 FFW，只在空间与时间注意力之后放一个 FFW，把算力留给更大的模型宽度和深度。

### 2.3 时空视频 tokenizer：因果 ST-ViViT VQ-VAE

Tokenizer 是一个带 ST-Transformer 编码器和解码器的 VQ-VAE，论文称为 ST-ViViT。它把视频 $x_{1:T}$ 压缩为离散 token $z_{1:T}$。由于 temporal attention 使用因果掩码，第 $t$ 帧的 token 可以聚合 $x_{1:t}$ 的历史，但不会偷看未来。

主模型配置：

| 配置 | Encoder | Decoder / codebook |
|---|---:|---:|
| 层数 | 12 | 20 |
| hidden size | 512 | 1024 |
| attention heads | 8 | 16 |
| patch size | 4 | — |
| codebook | — | 1024 codes，latent dim 32 |
| 参数量 | — | 合计约 200M |

论文的关键消融是：在相近参数下，ST-ViViT 的 FVD 为 81.4，优于 spatial-only ViT 的 114.5 和 C-ViViT 的 272.7；动作敏感指标 $\Delta_t\mathrm{PSNR}$ 也最高。这说明 tokenizer 不能只追求单帧重建，必须保留与运动有关的时间信息。

### 2.4 潜动作模型 LAM：用信息瓶颈逼出“左、右、跳”

LAM 是一个 VQ 风格的 encoder–decoder。Encoder 同时看历史 $x_{1:t}$ 和真实未来帧 $x_{t+1}$，把两者间的变化压缩成离散动作 $a_t$；Decoder 只能看历史与这个动作码，并尝试重建未来帧。

![](assets/Genie%201-3%20架构演进%20-%20Survey/figures/02-genie1-lam.png)

> 图 3：LAM 的训练信息瓶颈。未来帧只通过一个很小的离散动作码传给 decoder，因此动作码倾向于捕获可控变化，而不是完整复制未来画面。来源：论文 Figure 5。

形式化地：

$$a_t=Q(E_{\text{LAM}}(x_{1:t+1}))$$

$$\hat x_{t+1}=D_{\text{LAM}}(x_{1:t},a_t)$$

主模型动作 codebook 只有 **8 个码**，每个码 32 维；LAM encoder 与 decoder 均为 20 层、hidden size 1024、16 heads，patch size 16，总计约 300M 参数。小 codebook 是刻意的信息瓶颈：码太多更容易重建视频，却会让动作语义碎片化、难以当作人类控制器。

训练完成后，LAM encoder/decoder 在交互推理中都不需要；保留的是动作 codebook。用户直接选动作 ID，查表得到 action embedding，送入动力学模型。不同世界里的动作语义并非人工指定，但论文观察到它们通常能稳定对应 left、right、jump、no-op 等行为。

一个容易忽略的消融：LAM 直接读**像素**比读视频 tokenizer 的 token 更可控。Platformers 上 pixel-input 的 FVD 略差（40.1 vs 38.8），但 $\Delta_t\mathrm{PSNR}$ 明显更高（1.91 vs 1.33）；Robotics 上两项都更好。这说明用于视觉重建的 token 可能丢掉动作发现所需的细粒度运动线索。

### 2.5 动力学模型：10.1B 的 decoder-only MaskGIT

Dynamics model 接收历史离散视频 token 和动作 embedding，预测下一帧 token。它并不是逐 token 左到右生成一帧，而是采用 MaskGIT 式并行 masked-token refinement。

训练时，对未来帧 token 施加 50%–100% 的随机 mask，模型用交叉熵恢复被遮挡的 token：

$$\mathcal L_{\text{dyn}}=\operatorname{CE}(F_\theta(z_{1:t-1}^{\text{masked}},a_{1:t-1}),z_{2:t})$$

动作 embedding 以**加法**而非拼接方式注入视频表示；论文报告这能提高可控性。Dynamics loss 对 LAM 使用 stop-gradient，避免大型生成模型反过来扭曲动作空间。

最终 Dynamics Transformer 的公开配置为：

| 参数 | 数值 |
|---|---:|
| 参数量 | 10.1B |
| 层数 | 48 |
| hidden size | 5120 |
| heads | 36 |
| Q/K size | 128 |
| batch size | 512 |
| 训练步数 | 125k |
| 训练 token | 942B |
| 计算量 | $6.6\times10^{22}$ FLOPs |
| 硬件 | 256 TPU v5p |

稳定性措施包括 bfloat16、QK normalization、batch parallelism、ZeRO stage 3 sharding 与 tensor parallelism。

### 2.6 数据、训练规格与推理

Platformers 原始抓取约 55M 个 16 秒 clip，约 244k 小时；经 11M 参数 ResNet-18 质量过滤后保留 6.8M clip，即约 30k 小时。视频被统一到 160×90、10 FPS。论文也在 RT-1 等机器人视频上训练 2.5B 版本，忽略原有动作标签，以验证 latent action 发现并不限于平台游戏。

主模型的上下文只有 **16 帧**，约 1.6 秒。每生成一帧要做 25 次 MaskGIT refinement，temperature 2，速度约 1 FPS。网页演示使用额外的大 decoder 把 token 解码到 360p，但论文没有把它当作基础 11B 配置的一部分详细公开。

完整推理循环是：

$$x_1\rightarrow z_1,\quad a_1\rightarrow e(a_1),\quad (z_1,e(a_1))\rightarrow \hat z_2\rightarrow \hat x_2$$

随后把生成结果放回历史，接收下一个动作继续滚动。

### 2.7 Genie 1 真正解决了什么

Genie 1 的历史意义不是视觉质量，而是把“互联网视频”转成了“弱交互数据”。LAM 利用极小离散瓶颈，从没有按键记录的视频中发现可重复的动作接口；Dynamics 再学习动作条件下的下一帧分布。它第一次把“看视频”与“可玩环境”系统地连接起来。

但它的边界也很清楚：160×90 训练分辨率、约 1 FPS、1.6 秒上下文、2D 平台游戏数据占主导、动作语义需要试探、长程误差容易累积。Genie 2 的出现，正是因为瓶颈已经从“动作从哪里来”转移到“世界能否足够通用、逼真和长程一致”。

---

## 3. Genie 2：从离散 MaskGIT 转向潜扩散 3D 世界

### 3.1 官方明确披露的架构

Genie 2 的直接输入是一张 prompt image。官方演示通常先让 Imagen 3 把文字变成图，再由 Genie 2 把图变成可交互世界；因此不能说 Genie 2 自身已经公开了文本编码器。

官方给出的高层流程是：

$$x_t\xrightarrow{\text{autoencoder}}h_t$$

$$(h_{\le t},a_t)\xrightarrow{\text{causal Transformer + latent diffusion}}\hat h_{t+1}$$

$$\hat h_{t+1}\xrightarrow{\text{decoder}}\hat x_{t+1}$$

已确认的模块只有：

- 视频先经过 **autoencoder** 得到 latent frames；
- 核心是一个 **large Transformer dynamics model**；
- 采用类似 LLM 的 **causal mask**；
- 整体被官方定义为 **autoregressive latent diffusion model**；
- 推理时逐帧接收单个动作与过去 latent frames；
- 使用 **classifier-free guidance** 增强动作可控性；
- 官方博客样例来自未蒸馏 base model；另有可实时玩的 distilled model，但质量下降。

![Genie 2 官方页面能力与架构说明](assets/Genie%201-3%20架构演进%20-%20Survey/figures/04-genie2-official-page.png)

> 图 4：Genie 2 官方页面截图，展示单图初始化、动作可控 3D 世界及官方“Diffusion world model”说明。来源：Google DeepMind 官方博客，2024-12-04。

### 3.2 与 Genie 1 可以确认的架构变化

**第一，生成头从离散 MaskGIT 变成潜扩散。** Genie 1 在 VQ 离散 token 上做交叉熵 masked prediction；Genie 2 明确是 latent diffusion。它更接近现代高质量视频生成路线，目的很可能是提升复杂场景的视觉分布建模，但官方没有给出噪声参数化、采样器或扩散步数。

**第二，输入表示从公开的 ST-ViViT VQ-VAE 变成未说明类型的 autoencoder latent。** 官方没有说是连续 VAE、VQ、时空压缩还是逐帧编码，因此不能写成“沿用 Genie 1 tokenizer”。

**第三，控制信号从 8 个 latent action code 扩展到键盘/鼠标输入。** 这表明系统面向更复杂的第一/第三人称控制，但动作向量格式、训练标签与注入位置都未公开。

**第四，目标从 2D 平台器升级为多视角、开放域 3D 观感世界。** Genie 2 展示第一人称、等距、第三人称驾驶、复杂角色、NPC、开门、爆炸、重力、水、烟、光照等现象。这里的“3D”来自可导航视图与动态一致性，不代表内部显式 3D。

### 3.3 推理、时长与实时版本

基础模型逐帧自回归采样。世界最长可保持约 1 分钟的一致性，但官方说明大多数演示只有 10–20 秒。Genie 2 能从相同起始帧在不同动作下分叉出反事实轨迹，也能在目标离开视野后再次渲染出相对一致的场景内容。

“实时”只属于蒸馏版；官方没有公布帧率、分辨率、硬件、蒸馏算法和质量下降幅度。博客中的高质量样例来自**未蒸馏** base model，不能把演示质量与实时速度同时归到同一模型配置上。

### 3.4 Genie 2 未公开的关键项

| 类别 | 未公开信息 |
|---|---|
| 规模 | 参数量、层数、宽度、heads、context length |
| 表示 | autoencoder 类型、latent shape、压缩率、是否离散、codebook |
| 扩散 | 训练目标、noise schedule、sampler、步数、CFG scale |
| 控制 | 动作空间、鼠标编码、动作标签来源、是否使用 LAM |
| 数据 | 视频小时数、来源、标注方式、清洗策略、分辨率分布 |
| 训练 | 优化器、batch、算力、训练时长、课程与数据配比 |
| 推理 | 输出分辨率、base FPS、distilled FPS、延迟和硬件 |
| 评测 | 动作准确率、长期一致性指标、与基线的定量比较 |

因此，任何“Genie 2 是 X B 参数”“沿用 8 个 latent actions”“使用 Genie 1 的 ST-ViViT”之类说法，都不是官方已确认事实。

### 3.5 Genie 2 真正解决了什么

Genie 2 把问题从“无标签视频里能否发现动作”推进到“能否用一个高容量视频先验即时长出一个可探索世界”。它的关键贡献是**开放域视觉泛化 + 3D 交互观感 + 短期世界记忆**。同时，它暴露的新瓶颈是：高质量 base model 不实时，实时 distilled model 牺牲质量；世界一致性通常只有几十秒；架构和评测几乎不可复现。

---

## 4. Genie 3：分钟级长上下文的实时自回归世界

### 4.1 官方能确认的“架构”只有三层

Genie 3 没有公开 technical report。官方材料能严格确认的是：

1. **逐帧自回归**：每一帧根据世界描述、用户动作和此前生成轨迹产生；
2. **长历史条件**：用户一分钟后回到旧地点，模型需要引用一分钟前的信息；
3. **无显式 3D 表示**：一致性不是由 NeRF、Gaussian Splatting 或预先构建的显式场景提供，而是生成过程中的涌现能力。

可写成抽象接口：

$$x_{t+1}=G(x_{\le t},a_t,c_{\text{world}},e_t)$$

其中 $c_{\text{world}}$ 是初始文本世界描述，$a_t$ 是导航动作，$e_t$ 是可选的 promptable world event。这个式子描述输入输出依赖，不代表官方披露了具体网络层。

![Genie 3 官方页面实时规格与长时一致性](assets/Genie%201-3%20架构演进%20-%20Survey/figures/05-genie3-official-page.png)

> 图 5：Genie 3 当前官方模型页的能力卡片，明确给出 20–24 FPS、文本生成可控世界与 720p 输出；长时一致性与逐帧自回归信息来自同页后续章节及 2025-08-05 公告。

### 4.2 文本世界描述与可提示世界事件

Genie 3 首次在官方接口层面把文本直接放到世界模型入口：用户给出世界描述，模型生成可实时导航的环境。交互过程中还可以发送 promptable world events，例如改变天气、加入对象或角色。

这形成两层控制：

- **低层导航控制**：主体移动、视角与路径；
- **高层世界干预**：改变环境分布或引入事件。

高层事件扩大了反事实训练的覆盖面，但官方特别说明这些事件不一定是 agent 自己完成的动作。换言之，“让场景出现一场暴风雨”和“让 agent 学会操作设备制造暴风雨”是两回事。

### 4.3 实时规格、记忆与生成时长

原始公告给出的规格是 **720p、24 FPS、连续几分钟**；当前官方模型页表述为 **20–24 FPS**。视觉记忆可回溯约 1 分钟，环境整体可维持数分钟的基本一致性。

这里包含一个很强的系统约束：在每个 41–50 ms 左右的帧预算里，模型不仅要生成 720p 下一帧，还要把不断增长的轨迹中与当前视角相关的旧信息带回来。官方称为“significant technical breakthroughs”，但没有公布它靠滑动窗口、压缩记忆、检索、循环状态、KV cache、蒸馏、少步扩散还是其他组合实现。

所以可以确认**功能上存在长历史读取**，不能确认**结构上使用某种特定记忆模块**。

### 4.4 SIMA 接入方式

SIMA 接收自然语言目标并输出导航动作，Genie 3 并不知道任务目标，只根据动作模拟未来。系统分工是：

$$\text{goal}\rightarrow\text{SIMA}\rightarrow a_t\rightarrow\text{Genie 3}\rightarrow x_{t+1}$$

相比 Genie 2，分钟级一致性让 SIMA 能执行更长的动作序列和更复杂的目标。Genie 3 在这里扮演观察级 simulator，而不是 policy 或 planner。

### 4.5 后续官方页面更新

当前 Genie 模型页还披露：

- Project Genie 是可创建和探索世界的实验研究原型；
- 运行规格写为 20–24 FPS、720p；
- Genie 已加入 Google Maps Street View grounding，使生成世界可以锚定现实视觉数据。

Street View grounding 是当前产品页的新信息，但官方仍没有披露它如何进入训练或推理、是否是检索条件、数据配比和地理精度。页面同时承认：真实地点仍不能完美复刻。

### 4.6 绝对不能当成已知的内容

截至本文核对时间，官方没有确认 Genie 3 使用：

- Transformer、DiT 或某一具体 backbone；
- diffusion、flow matching 或 MaskGIT；
- Genie 1 的 ST-Transformer / VQ tokenizer / LAM；
- 任何具体参数量；
- 分层 tokenizer、3D convolution、KV cache、显式 memory bank 或 retrieval；
- 某个已知 Veo 版本作为底座；
- 特定训练数据小时数、动作标签来源、损失函数或蒸馏方法。

官方提到 Veo 2/3 是同一条世界模拟能力脉络，不等于公开宣布 Genie 3 “就是 Veo 3 加动作头”。社区常见的 DiT、长 KV cache、分层 tokenizer 等解释目前都只能算猜测。

### 4.7 Genie 3 的局限

官方列出的局限包括：

1. agent 可直接执行的动作空间仍有限；
2. 多个独立 agent 的复杂交互仍难模拟；
3. 真实地点不能达到精确地理复现；
4. 可读文字通常需要在输入描述中明确要求；
5. 连续交互只有几分钟，而不是数小时。

这说明 Genie 3 已跨过“能否实时玩”的门槛，但离“稳定、可验证、长时、可作为科学模拟器或完整游戏引擎”仍有明显距离。

---

## 5. 三代架构横向对比

| 维度 | Genie 1 | Genie 2 | Genie 3 |
|---|---|---|---|
| 主要场景 | 2D platformer；另有机器人验证 | 开放域 3D 观感世界 | 通用照片级/想象世界 |
| 初始条件 | 图像/草图/照片；文本常先转图 | 单张 prompt image，常由 Imagen 3 生成 | 直接文本世界描述 |
| 运行时控制 | 8 个无监督离散 latent actions | 键盘/鼠标单步动作 | 导航动作 + 文本 world events |
| 视觉表示 | 公开：ST-ViViT VQ-VAE，1024 codes | 仅知 autoencoder latent frames | 未公开 |
| 动作学习 | 公开：约 300M LAM，VQ codebook 8 | 未公开；不能确认沿用 LAM | 未公开；不能确认沿用 LAM |
| 动力学骨干 | 10.1B decoder-only ST-Transformer | large causal Transformer | 仅确认 frame-by-frame autoregressive；具体骨干未公开 |
| 生成目标 | 离散 token 的 MaskGIT 交叉熵 | latent diffusion | 未公开 |
| 动作引导 | additive action embedding | CFG 提升动作可控性 | 未公开 |
| 上下文/记忆 | 16 帧，约 1.6 秒 | 最长约 1 分钟；多数例子 10–20 秒 | 视觉记忆约 1 分钟，世界数分钟 |
| 分辨率 | 训练 160×90；网站额外 decoder 到 360p | 未公开 | 720p |
| 速度 | 约 1 FPS | base 未说明；蒸馏版实时但降质 | 20–24 FPS |
| 参数量 | 10.7B / 统称 11B | 未公开 | 未公开 |
| 数据 | 过滤后约 30k 小时平台游戏视频 | 仅称大规模视频数据 | 未公开；当前页增加 Street View grounding |
| 显式 3D | 无 | 未说明输出显式 3D；外部表现为 3D 世界 | 官方明确对比：不依赖显式 3D 表示 |
| 正式论文 | 有，ICML 2024 | 无 | 无 |
| 复现可能性 | 可理解、小规模配置可复现；大模型权重未开放 | 很低 | 很低 |

---

## 6. 架构演进的真正主线

### 6.1 第一阶段：动作标签瓶颈

如果世界模型必须依赖 $(x_t,a_t,x_{t+1})$，互联网视频几乎不可直接使用。Genie 1 的 LAM 用离散信息瓶颈把“帧间变化”压成动作，证明无动作标签视频也能转化为可控模型训练数据。这是系列里最清晰、最可复用的算法贡献。

### 6.2 第二阶段：视觉先验与开放域瓶颈

Genie 1 的 MaskGIT + 离散低分辨率 token 能验证范式，却难覆盖复杂 3D 场景。Genie 2 转向 latent diffusion，并用大规模视频先验生成更丰富的场景、角色、物体和物理现象。动作控制从少量潜码走向键鼠接口，但公开材料没有解释训练侧如何得到这些控制信号。

### 6.3 第三阶段：在线算力与长程状态瓶颈

Genie 2 的高质量 base model 与实时 distilled model 分裂，且多数演示只有 10–20 秒。Genie 3 的核心不是再展示“会生成水和树林”，而是把 720p 下一帧生成、动作响应和分钟级历史引用塞进实时循环。架构研究的焦点从“生成器选什么”转为“长轨迹如何保存、检索、压缩并在几十毫秒内利用”。

### 6.4 下一瓶颈：可验证状态，而不只是视觉一致

三代一直主要输出像素观察。像素看起来一致，不等价于：

- 物体拥有可查询、持久、精确的位置和属性；
- 碰撞、质量、摩擦和因果规律数值正确；
- 多 agent 共享同一个不冲突的世界状态；
- 奖励与任务状态能被可靠读取；
- 世界可连续运行数小时而不漂移。

因此下一阶段的关键问题，很可能不是单纯提高画质，而是怎样把观察级生成模型升级为**可验证、可持久、可供 agent 训练闭环依赖的状态模拟器**。

---

## 7. 常见误读纠正

### 7.1 “Genie 1–3 都是 11B”——错误

只有 Genie 1 明确公开 10.7B/11B。Genie 2 和 Genie 3 参数量均未公开。

### 7.2 “Genie 3 延续 Genie 1 的 LAM”——没有证据

Genie 3 有动作控制，不等于它仍通过无监督 VQ-LAM 学动作。官方没有披露动作训练管线。

### 7.3 “Genie 2/3 生成了真正的 3D 资产”——不成立

它们展示的是可导航、具有 3D 一致性的视觉世界；没有公开 mesh、NeRF、3DGS、scene graph 或物理状态输出。Genie 3 还明确表示不依赖显式 3D 表示。

### 7.4 “Genie 3 的架构是 DiT + KV cache”——目前只是推测

逐帧自回归和一分钟记忆可以由很多结构实现。没有技术报告前，不能把常见工程做法写成官方架构。

### 7.5 “Genie 2 已经 1 分钟稳定实时”——混合了两个版本

高质量博客样例来自未蒸馏 base model；官方只说蒸馏版本可实时，但质量下降。最长约一分钟是一致性展示，不等于高质量 base model 在该时长上实时运行。

---

## 8. 我的判断

### 8.1 已确立的事实

- Genie 1 的关键发明是 LAM，而不是 11B 本身；它让无动作视频成为可控世界模型数据。
- Genie 2 明确从离散 MaskGIT 路线切换到 autoregressive latent diffusion，并以因果 Transformer 建模世界动力学。
- Genie 3 明确是逐帧自回归、无显式 3D 的实时系统，达到 720p、20–24 FPS、分钟级一致性。
- 公开透明度逐代降低，导致 Genie 2/3 更像能力里程碑而非可独立检验的算法论文。

### 8.2 有依据但仍需验证的判断

- Genie 2 的 latent diffusion 很可能承担了比 Genie 1 MaskGIT 更强的开放域视觉先验，但缺定量对照，不能判断提升来自目标函数、数据、规模还是系统工程。
- Genie 3 最大的研究价值可能在长上下文压缩与低延迟推理，而不是基本自回归范式；遗憾的是这正是官方尚未披露的部分。
- 纯观察级 world model 会越来越接近“神经渲染器 + 隐式动力学”，但要成为可靠 agent simulator，仍需要更强的状态可读性、动作覆盖、失败检测和闭环评测。

### 8.3 研究上最值得追的四个问题

1. Genie 3 的一分钟视觉记忆究竟是 attention context、检索记忆、循环状态还是混合系统？
2. Genie 2/3 的动作标签如何获得，是否仍能从无标签视频扩展控制空间？
3. 实时版本通过什么蒸馏或少步生成技术在 720p 下达到 20–24 FPS？
4. 视觉一致性怎样转化为对 agent 有用的状态一致性与物理可验证性？

---

## 9. 材料来源

1. Bruce et al., **Genie: Generative Interactive Environments**, ICML 2024 / PMLR 235, arXiv:2402.15391：Genie 1 架构、训练、消融和限制的主来源。
2. Google DeepMind, **Genie 2: A large-scale foundation world model**, 2024-12-04：Genie 2 唯一官方架构说明。
3. Google DeepMind, **Genie 3: A new frontier for world models**, 2025-08-05：Genie 3 原始公告、规格与局限。
4. Google DeepMind, **Genie 3 model page / Project Genie**：后续 20–24 FPS、Street View grounding 和产品形态更新。
5. 本地材料清单见 [sources.md](assets/Genie%201-3%20架构演进%20-%20Survey/sources.md)。
