**结论先看：** 这篇《From Seeing to Knowing the World: A Survey of Vision World Models》可以看作视觉世界模型方向的一篇“总纲式综述”。它最重要的价值，不是提出某个新模型，而是把分散在视频生成、表征学习、机器人、自动驾驶里的 world model 研究，统一收束到一个 **Vision Encoding → Knowledge Learning → Controllable Simulation** 的分析框架里。

**一个必须带着读的补充判断：** 这篇 survey 的框架是 vision-centric 的，但它的方法语料库是 **video-generation-heavy** 的——配套论文库 294 条方法条目里，两类 diffusion 合计 142 条（48%），且 2026 年的增量几乎全在 diffusion 侧。因此如果只读框架图和 taxonomy，很容易得到“视觉世界模型 ≈ 可控视频生成”的错误印象。要纠正它，必须主动去看 JEPA、state transition、object-centric、显式几何这四条体量小但性质不同的主线（本笔记第 5 节专门展开）。

**重要性评估：★★★★☆（4/5）**

- **为什么值得刷入视野：** 它适合作为“世界模型方向的地图册”和后续追踪论文的坐标系。
- **为什么不是 5 星：** 它本质上是综述，不是提出新 SOTA 模型；对训练 recipe、数据配比、统一实验复现并没有给出可直接落地的一套新范式；且它的分类学与语料分布之间存在张力（见 7.5）。

## 1. 背景

这篇内容对应的微信公众号文章标题是《图文速览|VWM综述解读：视觉世界模型》，主线指向官方论文《From Seeing to Knowing the World: A Survey of Vision World Models》。原论文项目页为[AIWorldLab Survey](https://aiworldlab.github.io/survey/)，官方 PDF 为[preprint.pdf](https://aiworldlab.github.io/survey/preprint.pdf)，预印本落在[Preprints 页面](https://www.preprints.org/manuscript/202604.2072)。

论文要解决的问题不是“再造一个世界模型”，而是先回答一个更基础的问题：**什么才算 Vision World Model（VWM），以及我们该如何系统比较不同 world model 路线。**

作者认为，现有相关工作虽然都在讨论“世界建模”，但社区来源非常分散：

- 一部分来自视频生成，强调未来视觉内容的逼真合成；
- 一部分来自表征学习，强调在 latent / embedding 空间里做预测；
- 一部分来自强化学习与机器人，强调 planning、control 和 imagined rollout；
- 还有一部分来自自动驾驶、游戏环境、物理推理等应用领域。

因此，同样叫“world model”，不同论文常常在以下维度上互不兼容：

- **问题定义不统一**：到底是在做未来帧生成、latent rollout，还是行动条件下的可控仿真；
- **方法分类不统一**：autoregressive、diffusion、JEPA、RSSM、object-centric 常被分开讨论；
- **评测协议不统一**：有的看 FVD/FID，有的看 planning score，有的看 physics benchmark；
- **研究目标不统一**：有的偏视觉 realism，有的偏物理一致性，有的偏下游控制成功率。

所以这篇综述的核心贡献，是试图给出一个**视觉中心（vision-centric）**的统一视角：不是把 vision 当作输入模态之一，而是把“视觉如何揭示世界变化”当作 world model 的核心出发点。

## 2. 文章主线 / 论文线索

### 2.1 核心论文

- **论文名称：** From Seeing to Knowing the World: A Survey of Vision World Models
- **论文类型：** 长篇综述 / survey
- **项目页：** [AIWorldLab Survey](https://aiworldlab.github.io/survey/)
- **官方 PDF：** [preprint.pdf](https://aiworldlab.github.io/survey/preprint.pdf)
- **预印本页面：** [Preprints](https://www.preprints.org/manuscript/202604.2072)
- **配套论文库：** [Awesome-Vision-World-Model](https://github.com/AIWorldLab/Awesome-Vision-World-Model)（持续更新，7 类方法设计 + 5 类数据集/benchmark，本笔记的条目统计以 2026-07-07 版为准）
- **作者单位：** 北京交通大学等（靳潇杰等）

### 2.2 论文主线

这篇综述围绕三个核心问题展开：

1. **VWM 的定义是什么？**
    1. 作者把 VWM 定义为：**从视觉数据中学习世界知识，并在交互条件下生成未来世界状态的 AI 模型。**
2. **VWM 的方法版图如何统一整理？**
    1. 作者提出一个统一框架，把方法拆成三个部件：
        
        - **Vision Encoding**：如何把视觉输入编码成适合建模世界变化的表示；
        - **Knowledge Learning**：如何从视觉中学到时空一致性、物理动力学、因果机制；
        - **Controllable Simulation**：如何在动作 / 指令等交互条件下模拟未来。
            
3. **我们应该如何评估 VWM？**
    1. 不能只看视频生成好不好看，还要评估：
        
        - 视觉质量；
        - 物理合理性；
        - 下游任务性能；
        - 长时程一致性与因果响应。

### 2.3 与用户方向的关系

这篇综述与“世界模型 / 具身 / VLA / 视频世界建模”都强相关，但如果主标签只选一个，最合适的是世界模型。原因是它讨论的中心不是机器人 policy，也不是单纯视频生成，而是“如何从视觉中学世界演化规律，并支持交互条件下的未来仿真”。

## 3. Pipeline / Architecture + I/O 数据流

### 3.1 统一定义下的 I/O

论文给出的形式化表达非常关键。它把 VWM 写成一个条件概率模型：给定历史视觉上下文与交互条件，预测未来世界状态的分布：

$$p_\theta\big(S_{t+1:T} \mid E(v_{0:t}),\, c_t\big)$$

这个式子看起来平淡，但它其实是整篇 survey 的判据来源。把三个符号分别追问一次，就能得到区分"世界模型"与"视频生成器"的三个问题：

| 符号 | 含义 | 追问 |
|---|---|---|
| $E(v_{0:t})$ | 视觉编码器作用于历史观测 | 编码保留了什么、丢掉了什么？ |
| $c_t$ | 交互条件 | 条件是真的在控制演化，还是只是风格提示？ |
| $S_{t+1:T}$ | 未来世界状态 | 输出是像素、latent，还是结构化状态？ |

**关键观察：** 一个纯视频生成模型也能写成这个形式（$c_t$ 取文本、$S$ 取像素）。所以这个式子本身**不能**区分 VWM 与 video generator——真正的分野在于 $c_t$ 对 $S_{t+1:T}$ 的**因果有效性**，以及 $S$ 是否可被下游 planner 消费。这也是为什么综述必须额外引入 Controllable Simulation 与 Task Performance 两个维度。

可以把它读成如下 I/O 数据流：

1. 输入（Input）
    1. 视觉上下文 $v_{0:t}$：从时刻 0 到 t 的视觉观测
        
        - 可能是 RGB 图像 / 视频
        - 也可能包含 depth、optical flow、point cloud、BEV、多视角视频等
            
    2. 交互条件 $c_t$：控制未来演化的条件
        
        - 动作信号（action）
        - 语言指令（text instruction）
        - 轨迹、控制命令、任务目标等
            
2. 中间表示（Intermediate Representation）
    1. 先经过视觉编码器 $E(v_{0:t})$，把原始观测变成更适合世界建模的表示；
    2. 再通过不同范式的世界模型学习机制，建模世界的时空规律、物理规律和因果结构。
3. 输出（Output）
    1. 未来世界状态 $S_{t+1:T}$，可能有三种形式：
        
        - Latent states：latent rollout，用于 planning / reasoning；
        - Visual states：图像、视频或 3D/4D 视觉结果；
        - Structured outputs：轨迹、对象属性、空间结构、occupancy、action trajectory 等。

### 3.2 四大家族其实是这个式子的四种分解方式

把上面那个条件分布按不同方式分解，就直接得到四大家族。这是我认为理解这篇 survey 最快的一条路径：

**Sequential Generation** —— 把 $S$ 离散成 token 序列，按 chain rule 逐 token 分解：

$$p_\theta(S_{t+1:T}\mid \cdot)=\prod_{i} p_\theta\big(s_i \mid s_{<i},\, E(v_{0:t}),\, c_t\big)$$

**Diffusion-based** —— 不做序列分解，而是在连续 latent 上定义一条从噪声到数据的反向过程，条件注入每一步去噪：

$$\mathcal{L}=\mathbb{E}_{z,\epsilon,\tau}\big\|\epsilon-\epsilon_\theta(z_\tau,\tau,E(v_{0:t}),c_t)\big\|_2^2$$

**Embedding Prediction (JEPA)** —— 干脆放弃对 $S$ 建模分布，只预测未来的**表征**，用点估计代替生成：

$$\mathcal{L}=\big\|\,P_\theta(E(v_{0:t}),c_t)-\mathrm{sg}\big[E_{\text{target}}(v_{t+1})\big]\,\big\|$$

**State Transition** —— 引入一个紧凑隐状态 $h_t$ 承载全部历史，把长时程分解成一步步 Markov 转移：

$$h_{t+1}=f_\theta(h_t, z_t, a_t),\qquad z_{t+1}\sim p_\theta(z_{t+1}\mid h_{t+1})$$

四者的根本差别可以压缩成一句话：**对 $S$ 的处理方式不同**——序列生成把它离散化，扩散把它当连续分布，JEPA 放弃它只留表征，状态转移把它压进一个递归状态。后面 §4 给每一族补上具体 pipeline 与损失。

### 3.3 三段式框架

#### A. Vision Encoding

目标是把原始视觉数据变成适合建模世界变化的表示。

主要分两层：

- 输入模态层：RGB、depth、flow、point cloud、BEV、多视角等；
- 表示形式层：
    - continuous latent representations
    - discrete tokenized representations
    - object-/entity-centric representations
    - hybrid / hierarchical representations

作者强调：视觉编码并不是“压缩输入”这么简单，而是在决定哪些世界信息会被保留、以什么抽象粒度保留。这直接决定后续能学到什么 world knowledge。

#### B. Knowledge Learning

这是论文中最值得关注的一层。作者把“世界知识”拆成三类：

1. Spatio-temporal coherence
    1. 对象身份跨时间与跨视角保持一致；
    2. 遮挡之后仍能维持 object permanence；
    3. 状态演化平滑，不出现无故跳变。
2. Physical dynamics
    1. 满足重力、接触、碰撞、材质、形变、流体等物理约束；
    2. 不只是视频“看起来像”，而是演化过程物理上说得通。
3. Causal mechanisms
    1. 学到 action → outcome 的因果关系；
    2. 支持 counterfactual reasoning，即“如果动作换一个，结果会怎样”。

#### C. Controllable Simulation

这一层关注的是：模型如何在给定交互条件下产生未来。

作者把它拆成：

- Simulation：生成 latent / visual / structured future states；
- Interaction：通过 action、语言、轨迹、控制信号等引导未来走向。

一个很重要的判断是：

- 如果模型只能 unconditional 或 weakly conditioned 地生成未来视频，它更像 generative model；
- 如果模型能稳定响应 action / instruction 并产出可用于 planning 的未来，它才更接近真正的 VWM。

### 3.4 论文中的总图

下面两张图基本就是这篇综述最值得保存的视觉材料：

先看统一框架图，它说明了从视觉输入到知识学习再到交互仿真的整体闭环：

![](assets/视觉世界模型%20-%20Survey/framework.png)

再看 taxonomy 图，它把不同 architectural family 放到同一张版图里，便于后续把新论文定位进去：

![](assets/视觉世界模型%20-%20Survey/taxonomy.png)

## 4. Methodology：作者如何组织整个 VWM 版图

### 4.1 四大架构家族

论文把现有方法归成四个 major architectural families。下面给每一族补上代表工作，方便把新论文往里挂：

1. Sequential Generation
    1. 代表路线：visual autoregressive model、MLLM-guided multimodal autoregressive model
    2. 核心思想：把世界建模转成 token sequence generation
    3. 代表工作：`GAIA-1`（2309.17080，最早一批驾驶生成式世界模型）、`Genie`（2402.15391）→ `Genie 2` → `Genie 3`（可交互环境生成三代）、`iVideoGPT`（2405.15223）、`VideoWorld`（2501.09781，从无标注视频学知识）、`OccWorld`（2311.16038，3D occupancy 而非 RGB）、`WorldVLA`（2506.21539）、`RLVR-World`（2505.13934，用 RL 训世界模型）
    4. 优点：易于利用 LLM / Transformer 的扩展性
    5. 问题：长时 rollout 容易误差累积，物理一致性主要靠数据归纳
2. Diffusion-based Generation
    1. 代表路线：latent diffusion、autoregressive diffusion
    2. 核心思想：在连续 latent 里用 denoising 生成未来
    3. 代表工作（latent）：`UniPi`（2302.00111）、`DriveDreamer`（2309.09777）、`Cosmos`（2501.03575）、`Cosmos transfer1`（2503.14492）、`GAIA-2`（2503.20523）、`Aether`（2503.18945）、`DreamGen`（2505.12705）
    4. 代表工作（autoregressive diffusion）：`UniSim`（2310.06114）、`GameNGen`（2408.14837）、`diamond`（2405.12399）、`Vista`（2405.17398）、`Matrix-Game` / `2.0` / `3.0`、`Genie Envisioner`（2508.05635）、`Dreamer V4`（2509.24527）、`WORLDMEM`（2504.12369）与 `VMem`（2506.18903）/ `spmem`（2506.05284）这条显式记忆线、`Geometry Forcing`（2507.07982）、`PAN`（2511.09057）、`Cosmos 3`（2606.02800）
    5. 优点：视觉质量高，短窗口内时空一致性更强
    6. 问题：采样重、成本高，长时程可扩展性仍困难
3. Embedding Prediction
    1. 代表路线：JEPA / DINO feature prediction
    2. 核心思想：不重建像素，而是在 representation space 预测未来 embedding
    3. 代表工作：`V-JEPA`（2404.08471）→ `V-JEPA 2`（2506.09985）→ `V-JEPA 2.1`（2603.14482）、`IWM`（2403.00504）、`DINO-WM`（2411.04983）、`DINO-world`（2507.19468）、`Flare`（2505.15659）、[[LeWorldModel - 像素端到端稳定JEPA世界模型|LeWorldModel]]（2603.19312）、`VLA-JEPA`（2602.10098）
    4. 优点：rollout 高效，非常适合 planning
    5. 问题：可解释性弱，且上限受预训练 encoder 能力约束
4. State Transition
    1. 代表路线：RSSM / Dreamer family / object-centric state transition
    2. 核心思想：维护 compact latent state，并通过 transition model 做长时更新
    3. 代表工作（state-space）：`PlaNet`（1811.04551）→ `DreamerV1`（1912.01603）→ `V2`（2010.02193）→ `V3`（2301.04104）、`TD-MPC2`（2310.16828）、`IRIS`（2209.00588）、`STORM`（2310.09615）、`DayDreamer`（2206.14176）、`Dynalang`（2308.01399）、`Think2Drive`（2402.16720）
    4. 代表工作（object-centric）：`G-SWM`（2010.02054）、`SlotFormer`（2210.05861）、`RoboDreamer`（2404.12377）、`Dreamweaver`（2501.14174）、`SlotPi`（2506.10778）、`Dyn-o`（2507.03298）
    5. 优点：对 planning 和 long-horizon imagination 很友好
    6. 问题：细粒度几何与视觉细节表达能力常弱于直接视觉生成路线

另有一类 Other Architectures，收纳不好归入上面四族的混合路线，其中值得注意的是显式 3D 表示这一支：`GaussianWorld`（2412.10373）、`ManiGaussian++`（2506.19842）、`GAF`（2506.14135，Gaussian Action Field 做 4D 表示）、`LiDARWM`（2506.23434，LiDAR latent flow matching）、`NeMo`（neural volumetric world model）。这类工作说明“世界状态”不一定要以视频或 latent 向量为载体，也可以直接是几何。

### 4.2 七类代表性子设计（含配套论文库的实际规模）

论文进一步细分成 7 个子设计，另设一类 Other Architectures 收纳混合路线。配套的开源论文库[Awesome-Vision-World-Model](https://github.com/AIWorldLab/Awesome-Vision-World-Model)把每类的代表工作都列了出来，截至 2026-07 共收录 294 条方法条目。把条目数统计出来，比读框架图更能看清这个领域的真实重心：

| 子设计 | 条目数 | 占比 | 最早 | 最新 | 年度分布（2023/24/25/26） |
|---|---|---|---|---|---|
| Latent Diffusion | 76 | 26% | UniPi (2302) | RynnWorld-4D (2607) | 6 / 11 / 37 / 21 |
| Autoregressive Diffusion | 66 | 22% | UniSim (2310) | Cosmos 3 (2606) | 2 / 12 / 35 / 14 |
| Latent State-Space Modeling | 44 | 15% | PlaNet (1811) | NE-Dreamer (2603) | 12 / 7 / 17 / 1 |
| Visual Autoregressive | 32 | 11% | GAIA-1 (2309) | VideoWorld 2 (2602) | 2 / 11 / 16 / 1 |
| MLLM-guided Multimodal AR | 23 | 8% | ADriver-I (2311) | WLA (2606) | 1 / 7 / 12 / 3 |
| JEPA / Embedding Prediction | 23 | 8% | IWM (2403) | FR3D (2606) | 0 / 5 / 8 / 10 |
| Object-Centric Modeling | 16 | 5% | G-SWM (2010) | LPWM (2603) | 2 / 4 / 6 / 1 |
| Other Architectures | 14 | 5% | FleetWM (2410) | PhysWorld (2510) | 0 / 2 / 11 / 0 |

这个拆法的价值在于它不再按应用领域来分，而是按“视觉表示如何编码、世界知识如何学习、未来如何模拟”三个统一部件来比较各路线。但上面这张表同时暴露了一件更重要的事，见下一节。

### 4.3 为什么这篇 survey 读起来像“视频世界模型综述”

这是读这篇 survey 时最值得警惕的一点，而且它不是综述作者的表述问题，而是领域现状的真实投影：

1. 两类 diffusion 合计 142 条，占全部方法条目的 48%。 也就是说，接近一半的“视觉世界模型”工作，本体是视频扩散模型加上动作条件。综述在正文里反复强调“world model ≠ video generator”，但它的方法版图本身就被视频生成占了半壁江山。
2. 2026 年的新增几乎全在 diffusion 侧。 2026 年的新条目里 Latent Diffusion 有 21 条、Autoregressive Diffusion 有 14 条，而 Visual AR、State-Space、Object-Centric 三类 2026 年各只有 1 条。领域的增量投入高度集中。
3. 非视频路线的“时间形状”完全不同。 State-Space 的高峰在 2023（12 条）与 2025（17 条），Object-Centric 一直维持在每年 4-6 条的小体量，只有 JEPA 是唯一逐年递增的一类（0 → 5 → 8 → 10），2026 年条目数已经超过它此前任何一年。

所以更准确的读法是：这篇 survey 的框架是 vision-centric 的，但它的语料库是 video-generation-heavy 的。 如果只读三段式框架和 taxonomy 图，很容易得到“视觉世界模型 ≈ 可控视频生成”的印象；要纠正这个印象，必须主动去看下面几条被条目数稀释掉的主线。

### 4.4 一个容易被忽略的信号：Dreamer V4 被归到了 diffusion 家族

论文库把 `PlaNet`、`DreamerV1/V2/V3` 都放在 Latent State-Space Modeling 下，但把 `Dreamer V4`（2509.24527）放进了 Autoregressive Diffusion。

这个分类不是随手为之，它记录了一次真实的路线迁移：state-space 血脉里最有代表性的一支，在最新一代把 transition model 换成了 diffusion-based 的生成式 rollout。同类信号还有 `StateSpaceDiffuser`（2505.22246，被归到 Latent Diffusion）与 `S5WM`、`DMWM` 这类在 SSM 里塞进扩散/世界建模目标的工作。

判断：四大家族的边界正在从“互斥路线”变成“可组合模块”。 后续读新论文时，与其问“它属于哪一家族”，不如问“它在表示层用了什么、在 transition 层用了什么、在解码层用了什么”这三个独立问题。

### 4.5 这篇综述的真正方法论价值

我认为这篇 survey 真正的价值，在于它把原本混在一起的几个讨论层级分开了：

- 表示层：token、latent、slot、state 到底哪种更适合 world knowledge；
- 学习目标层：next-token、denoising、mask-and-predict、state transition 分别在学什么；
- 仿真层：模型输出的是视频、latent 还是 structured state；
- 交互层：action、language、trajectory 哪种条件能真正控制未来；
- 评估层：视觉逼真不等于世界理解。

这会让后续阅读新论文时，判断更快：

- 它是在补表示，还是补 physics，还是补 controllability？
- 它主要提升的是画面质量，还是 planning utility？
- 它算不算真正的 VWM，还是只是“可控视频生成器”？

## 5. 被条目数稀释的四条非视频主线

这一节是我自己补的，不是原文的章节划分。理由是：如果按条目数读这篇 survey，注意力会全部被 diffusion 吸走；但真正决定“world model 是不是只是视频生成器”这个问题答案的，恰恰是下面这四条体量不大的线。

### 5.1 JEPA / Embedding Prediction：唯一逐年递增的一族

这一族的关键立场是拒绝像素重建。它认为要预测的不是“未来看起来什么样”，而是“未来的表征是什么”。

#### 5.1.1 V-JEPA 的目标函数：注意是 $\ell_1$ 而不是 L2

`V-JEPA`（2404.08471）的朴素形式会立刻坍塌：

$$\min_{\theta,\phi}\ \big\lVert P_\phi(E_\theta(x),\Delta_y)-E_\theta(y)\big\rVert_1$$

实际使用的是带 EMA target 与 stop-gradient 的版本（原文 Eq. 1）：

$$\min_{\theta,\phi}\ \big\lVert P_\phi(E_\theta(x),\Delta_y)-\mathrm{sg}\big(\overline{E}_\theta(y)\big)\big\rVert_1$$

其中 $x$ 是可见上下文、$y$ 是被 mask 掉的目标区域、$\overline{E}_\theta$ 是 $E_\theta$ 的 EMA 副本（momentum 0.998 → 1.0 线性上升），$P_\phi$ 是一个窄预测器（12 层、宽 384）。

**一个容易被写错的细节：这里用的是 $\ell_1$ 而不是 L2**，原文明确说改成 $\ell_1$ 回归"更稳定"。这不只是工程调参——$\ell_1$ 的最优预测器是条件**中位数**，于是梯度对应的是 median absolute deviation 而不是方差。对"未来本质多模态"的视频预测来说，预测中位数比预测均值更不容易糊成一团。

**Mask 策略的具体数字**（这是它和图像 JEPA 的关键差别）：short-range 取 8 个各覆盖单帧 15% 的 block 的并集，long-range 取 2 个各覆盖 70% 的 block 的并集，且**block 在时间维上整段重复**——所以 mask 是空间连续的"管道"，不是逐帧随机。平均遮蔽率约 90%。之所以必须沿时间整段遮，是因为如果逐帧随机遮，模型可以直接从相邻帧抄答案，任务退化成插值而不是学动力学。

**为什么不重建像素**——原文给的是消融证据而非理论：ViT-L/16 下，feature target 对比 normalized-pixel MSE，K400 冻结评测 **73.7 vs 68.6**，而 fine-tune 后几乎持平（85.6 vs 85.4）。所以准确的说法是"在冻结表征评测上明显更好"，把它讲成"重建像素一定浪费容量"属于合理解读但不是原文证明的结论。

#### 5.1.2 V-JEPA 2-AC：世界模型其实只是一个 300M 的 predictor

`V-JEPA 2`（2506.09985）扩到 ViT-g（1B）、22M 视频、100 万小时以上。真正值得记住的是它的动作条件版 **V-JEPA 2-AC**，因为它把"什么是世界模型"这件事压到了极简：

- **encoder 完全冻结**，逐帧编码 $z_k = E(x_k)\in\mathbb{R}^{16\times16\times1408}$；
- 只训练一个 ~300M 的 block-causal predictor（24 层）；
- 动作 $a_k\in\mathbb{R}^7$ 定义为末端执行器状态的**差分**（3 位置 + 3 欧拉角 + 1 夹爪）；
- 训练数据是 **不到 62 小时** 的 Droid 视频（23k 条轨迹，含失败样本），无奖励、无任务标签、无成功标志。

两项损失。teacher forcing（$T=15$）：

$$\mathcal{L}_{\text{tf}}(\phi)=\frac{1}{T}\sum_{k=1}^{T}\Big\lVert P_\phi\big((a_t,s_t,E(x_t))_{t\le k}\big)-E(x_{k+1})\Big\rVert_1$$

以及 rollout（实际只取 $T=2$，即梯度只穿过一步递归）：

$$\mathcal{L}_{\text{rollout}}(\phi)=\big\lVert P_\phi(a_{1:T},s_1,z_1)-z_{T+1}\big\rVert_1$$

总损失 $L(\phi)=\mathcal{L}_{\text{tf}}+\mathcal{L}_{\text{rollout}}$，**只对 $\phi$ 求导**。

**推理时的 planning 是能量最小化**，这是整条 JEPA 路线最干净的部分：

$$\mathcal{E}(\hat a_{1:T};z_k,s_k,z_g):=\big\lVert P(\hat a_{1:T};s_k,z_k)-z_g\big\rVert_1,\qquad a^\star_{1:T}=\arg\min_{\hat a_{1:T}}\mathcal{E}$$

代价函数就是**在冻结 encoder 的 latent 空间里，到目标图像编码 $z_g$ 的 $\ell_1$ 距离**——没有奖励、没有 decoder、不碰像素。用 CEM 求解（800 采样、10 轮精修、top-10 elite、horizon $T=1$），执行第一步后重规划。

这里有一个数字最能说明"不解码像素"的价值：**每个动作 16 秒，而基于 Cosmos 的对照是 4 分钟**——约 15× 的差距几乎全部来自 planning 循环里没有像素解码。

> **两处必须一起记住的局限**（否则容易高估这条路线）：
> 1. pick-and-place 任务需要人工提供 2 张 sub-goal 图像，且切换步数是**硬编码**的（4 步 / 10 步 / 4 步），这不是自主长程规划；
> 2. 能量地形虽然平滑，但最小值落在 $(\Delta x,\Delta y)\approx(0,-0.05)$，真值是 $(0,-0.1)$——**方向对、幅值偏小**。

#### 5.1.3 DINO-WM：把 collapse 问题结构性地消掉

`DINO-WM`（2411.04983）走得更极端：encoder 用**冻结的 DINOv2**，只训练 transition：

$$\mathcal{L}_{pred}=\Big\lVert p_\theta\big(\mathrm{enc}_\theta(o_{t-H:t}),\phi(a_{t-H:t})\big)-\mathrm{enc}_\theta(o_{t+1})\Big\rVert^2$$

注意三件事：

1. **用的是平方 L2，而不是 V-JEPA 的 $\ell_1$**；
2. **不需要 stop-gradient，也不需要 EMA**——因为 target encoder 是冻结的预训练网络，坍塌在结构上就不可能发生。这是它与 V-JEPA 最本质的差别；
3. 保留 **patch token** $z_t\in\mathbb{R}^{N\times E}$ 而不是池化成 CLS，动作经 MLP 编成 10 维后**拼到每一个 patch 向量上**。

planning 的代价函数只在终点算：

$$\mathcal{C}=\lVert \hat z_T-z_g\rVert^2$$

同样用 CEM。论文里一个有意思的观察：模型是可微的，所以可以直接对动作做梯度下降，但**CEM 实际打不过梯度下降**——作者推测是因为学到的动力学没有平滑性约束，梯度方向不可靠。

它还顺带解释了 `TD-MPC2` 在这个 setting 下为什么差：**"缺少奖励信号使得学好 latent 表征很困难"**——TD-MPC2 的表征目标在 reward-free 离线设定下根本没有定义。

> 这里有一个值得单独拎出来的**张力**：TD-MPC2 去掉重建是为了让 latent **更任务化**（reward 与 value 塑造表征），DINO-WM 去掉 reward 是为了让 latent **更任务无关**（任何目标图都能当 $z_g$）。两者都声称"去掉某一项让模型更好"，但去掉的东西正好相反。这说明"该保留什么监督"取决于你要的是专用 planner 还是通用世界模型，不存在统一答案。

#### 5.1.4 三种防坍塌方案

把这一族连起来看，"预测自己的特征为什么不会坍塌"其实只有三个答案：

| 方案 | 代表 | 机制 | 代价 |
|---|---|---|---|
| EMA + stop-grad + 非对称 predictor | V-JEPA / V-JEPA 2 | target 是自己的慢速副本 | 需要调 momentum schedule，训练脆 |
| 冻结预训练 encoder | V-JEPA 2-AC、DINO-WM、SlotFormer | 没有可坍塌的东西 | 上限被 encoder 锁死 |
| 分布正则 | [[LeWorldModel - 像素端到端稳定JEPA世界模型\|LeWorldModel]] 的 SIGReg | 强制 latent 边缘分布逼近各向同性高斯 | 低 intrinsic-dim 数据上反而不占优 |

第三条是本 vault 已有的深读：[[LeWorldModel - 像素端到端稳定JEPA世界模型|LeWorldModel]]（2603.19312）只用 next-embedding prediction + SIGReg 两项损失，在 15M 参数下从像素端到端稳定训练，planning 提速最高 48×。它的关键论点是 VICReg 式的 variance+covariance 只约束二阶矩，挡不住 cluster collapse；而 SIGReg 用 Epps–Pulley 检验比较经验特征函数与高斯特征函数，**约束的是全部阶的矩**。

理论侧则是 [[LeJEPA - 世界模型线性可识别性理论|LeJEPA]]：在高斯潜变量 + 平稳加噪转移的假设下，表征可线性恢复真实 latent（只差一个正交变换），于是 latent-space planning 与真实状态规划等价。而且它是双向结论——想要这种线性可识别性，latent **必须**是高斯。

为什么这一族值得单独盯：它是唯一在结构上就不可能被误认为视频生成器的路线（它压根不输出像素），同时也是 2026 年唯一还在加速的路线。

### 5.2 State Transition：最老、也最被 planning 验证过的一族

这一族从 `PlaNet`（2018.11）算起已经八年，是唯一贯穿 RL 时代的血脉：`PlaNet` → `DreamerV1/V2/V3` → `TD-MPC2`，加上 token 化变体 `IRIS`（2209.00588）与 `STORM`（2310.09615）。

#### 5.2.1 DreamerV3 的 RSSM：六个组件

$$\begin{aligned}
\text{序列模型：} &\quad h_t = f_\phi(h_{t-1},z_{t-1},a_{t-1})\\
\text{编码器（后验）：} &\quad z_t \sim q_\phi(z_t\mid h_t,x_t)\\
\text{动力学预测器（先验）：} &\quad \hat z_t \sim p_\phi(\hat z_t\mid h_t)\\
\text{奖励预测器：} &\quad \hat r_t \sim p_\phi(\hat r_t\mid h_t,z_t)\\
\text{continue 预测器：} &\quad \hat c_t \sim p_\phi(\hat c_t\mid h_t,z_t)\\
\text{解码器：} &\quad \hat x_t \sim p_\phi(\hat x_t\mid h_t,z_t)
\end{aligned}$$

$z_t$ 取自 **32 个 categorical × 64 类**的 softmax 向量，用 straight-through 梯度；并且 categorical 参数化成 **1% 均匀 + 99% 网络输出**（unimix）以避免确定性与 KL 尖峰。

#### 5.2.2 KL balancing 与 free bits：这是最值得抄走的一段

$$\begin{aligned}
\mathcal{L}_{\mathrm{dyn}}(\phi)&=\max\Big(1,\ \mathrm{KL}\big[\ \mathrm{sg}(q_\phi(z_t\mid h_t,x_t))\ \big\|\ p_\phi(z_t\mid h_t)\ \big]\Big)\\
\mathcal{L}_{\mathrm{rep}}(\phi)&=\max\Big(1,\ \mathrm{KL}\big[\ q_\phi(z_t\mid h_t,x_t)\ \big\|\ \mathrm{sg}(p_\phi(z_t\mid h_t))\ \big]\Big)
\end{aligned}$$

$$\mathcal{L}(\phi)=\mathbb{E}_{q_\phi}\Big[\sum_{t=1}^{T}\big(\beta_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}+\beta_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}+\beta_{\mathrm{rep}}\mathcal{L}_{\mathrm{rep}}\big)\Big],\quad \beta_{\mathrm{pred}}{=}1,\ \beta_{\mathrm{dyn}}{=}1,\ \beta_{\mathrm{rep}}{=}0.1$$

**这两项是同一个 KL，只是 stop-gradient 放在相反的一侧，权重差 10 倍。** $\mathcal{L}_{\mathrm{dyn}}$ 把先验拉向后验（学动力学），$\mathcal{L}_{\mathrm{rep}}$ 把后验拉向先验（让表征可预测）。free bits 用 $\max(1,\cdot)$ 在 1 nat 以下**直接关闭该项**，把学习重心交回 $\mathcal{L}_{\mathrm{pred}}$，这样就不会走向"极易预测但毫无信息"的退化解。消融里 KL 目标被列为最重要的稳健性技术。

> 需要说清楚的一点：因为有 stop-gradient、不对称权重和截断，**这已经不是一个严格的 ELBO 了**，而是一个放松版本。把它当成"VAE 的变分下界"讲会误导。

#### 5.2.3 imagination 里的 actor-critic

$$R^\lambda_t=r_t+\gamma c_t\big((1-\lambda)v_t+\lambda R^\lambda_{t+1}\big),\qquad R^\lambda_T=v_T$$

注意 **continue 标志 $c_t$ 是乘进递推式里的**，所以 episode 终止在 return 内部就处理掉了。$\gamma=0.997$、imagination horizon 16、$\lambda=0.95$。

actor 用 Reinforce，且**归一化的是 return 而不是 advantage**：

$$\mathcal{L}(\theta)=-\sum_{t}\mathrm{sg}\Big(\big(R^\lambda_t-v_\psi(s_t)\big)/\max(1,S)\Big)\log\pi_\theta(a_t\mid s_t)+\eta\,\mathrm{H}\big[\pi_\theta\big]$$

其中 $S$ 是 return 的 95–5 百分位差的 EMA。$\max(1,S)$ 只会把大 return **缩小**、不会放大小 return——这是刻意的，否则稀疏奖励任务的噪声会被放大到盖过 entropy bonus。

还有一个被广泛借用的技巧，symlog 压缩与 two-hot 离散回归：

$$\mathrm{symlog}(x)=\mathrm{sign}(x)\ln(|x|+1),\qquad \mathrm{symexp}(x)=\mathrm{sign}(x)(\exp(|x|)-1)$$

$$\mathcal{L}(\theta)=-\mathrm{twohot}(y)^{\!\top}\log\mathrm{softmax}(f(x,\theta))$$

关键性质是原文那句话：**损失只取决于分配给各 bin 的概率，与 bin 对应的连续值无关，从而把梯度大小与目标大小解耦。** 这就是 DreamerV3 能用同一套超参跨 150+ 任务的核心原因。

#### 5.2.4 TD-MPC2：彻底没有 decoder

$$\mathcal{L}(\theta)=\mathbb{E}\Big[\sum_{t=0}^{H}\lambda^t\Big(\underbrace{\lVert z'_t-\mathrm{sg}(h(s'_t))\rVert^2_2}_{\text{latent 一致性}}+\underbrace{\mathrm{CE}(\hat r_t,r_t)}_{\text{奖励}}+\underbrace{\mathrm{CE}(\hat q_t,q_t)}_{\text{价值}}\Big)\Big]$$

三项，没有任何重建。两个细节：latent 一致性的 target 是**在线 encoder 的 stop-grad**，不是 EMA；奖励与价值用**对数空间的离散回归**（与 DreamerV3 的 two-hot 同族），而一致性项保持连续回归。稳定性靠 SimNorm 把 $z$ 投到若干单纯形上。

planning 用 MPPI，而**终端价值项是它成立的关键**：

$$\max_{(\mu,\sigma)}\ \mathbb{E}\Big[\gamma^H Q(z_{t+H},a_{t+H})+\sum_{h=t}^{H-1}\gamma^h R(z_h,a_h)\Big]$$

有限步的模型奖励之和 + 一个 $\gamma^H$ 折扣的**学习到的终端价值**去 bootstrap $H$ 之后的一切。没有这个 $Q$ 项，MPC 只能做到"时间局部最优"，不解决完整的 RL 问题。

它与视频生成路线最本质的区别在目标函数的归属：Dreamer 系的重建只是辅助项、TD-MPC2 连重建都没有，真正的目标是 policy return；而视频扩散世界模型的目标函数就是生成质量本身，planning 效果是外挂评估出来的。

值得注意的是这一族在 2026 年只有 1 条新增（`NE-Dreamer`，2603.02765），而它最新的旗舰 `Dreamer V4` 已经被归进 diffusion 家族。这不代表路线死了，而是它的思想被吸收进了别的架构——compact latent state + 长时 imagination 这套东西现在以模块形式出现在各家里。

### 5.3 Object-Centric：体量最小，评测意义大于方法意义

`G-SWM`（2010.02054）、`SlotFormer`（2210.05861）、`RoboDreamer`（2404.12377）、`Dreamweaver`（2501.14174）、`SlotPi`（2506.10778）、`Dyn-o`（2507.03298）——十六条，八年，是全表最小的一族。

以 `SlotFormer` 为例看它怎么做 rollout。先用 Slot Attention 把每帧绑成 $N$ 个 slot $\mathcal{S}_t\in\mathbb{R}^{N\times D}$（这一步的 object-centric 模型是**先用重建损失预训练好再冻结**的），然后：

$$V=[G_1,\dots,G_T]+[P_1,\dots,P_T],\qquad U=\mathcal{T}(V),\qquad \hat{\mathcal{S}}_{T+1}=\mathrm{Linear}(U_T)$$

$$\mathcal{L}=\underbrace{\frac{1}{KN}\sum_{k}\sum_{n}\lVert \hat{\bm s}^n_{T+k}-\bm s^n_{T+k}\rVert^2}_{\mathcal{L}_S\ \text{slot 空间}}+\lambda\underbrace{\frac{1}{K}\sum_k\lVert f_{dec}(\hat{\mathcal{S}}_{T+k})-\bm x_{T+k}\rVert^2}_{\mathcal{L}_I\ \text{像素辅助}}$$

两个设计决定比公式更重要：

1. **位置编码只加在时间维上**（$P_t$ 复制 $N$ 份，同一时刻所有 slot 共享），因此保持了 slot 之间的**置换等变性**——没有哪个 slot 获得独立身份。这正是"能对单个物体做推理"的来源，而且参数更少、效果还更好。
2. **训练时喂的是自己预测的 slot 而不是 teacher forcing**，消融显示 teacher forcing"显著变差"。这与 V-JEPA 2-AC、DINO-WM 的选择正好相反。

> **一处必须纠正的常见说法（也包括我这篇笔记的上一版）：** SlotFormer **没有做 counterfactual 实验**。CLEVRER 有 descriptive / explanatory / predictive / counterfactual 四类问题，而原文明确只聚焦 **predictive**（+3.4% per-option、+6.0% per-question）。"counterfactual"一词在全文只出现两次，都只是在列举 CLEVRER 的问题类型。
>
> 所以准确的表述是：slot 表示**在架构上支持**替换/删除/扰动单个 slot 再 rollout，因此是做对象级干预的天然载体——但这是**合理推断，不是 SlotFormer 的实验结论**。原文真正支撑的是：object-factored、置换等变、长程 rollout 中属性一致。
>
> 另外 PHYRE 上的"planning"其实是**动作排序**（用 SlotFormer 模拟的未来训一个成功分类器，对 100 个候选动作排序），不是 CEM/MPC 那种轨迹优化，和上面几族的 planning 不是一回事。

判断：这一族大概不会自己长成主流，但它的评测意义大于方法意义——它定义了"真正理解世界"的一个可检验下界。同时它也是这四条线里**最不"非生成"**的一支：它的 slot 模型靠重建预训练，损失里还留着一项像素辅助。

### 5.4 显式几何 / 4D：世界状态不一定是视频

散落在 Other Architectures 与 latent diffusion 里的一批工作，把世界状态直接表示成几何：`OccWorld`（2311.16038）与 `OccTENS`（2509.03887）用 3D occupancy，`GaussianWorld`（2412.10373）与 `ManiGaussian++`（2506.19842）用 3D Gaussian，`GAF`（2506.14135）提出 Gaussian Action Field 作为 4D 表示，`LiDARWM`（2506.23434）在 LiDAR latent 上做 flow matching，`RynnWorld-4D`（2607.06559）是论文库里最新的一条。

这条线与本 vault 已有的 [[GaussianDream - 机器人操控的前馈三维高斯世界模型|GaussianDream]] 直接同源，也与 [[前馈式三维重建与生成式补全 - Survey|前馈式三维重建与生成式补全]] 那篇 survey 的路线 A/B 之分呼应：几何表示天然带来跨视角一致性与长期空间记忆，代价是丢掉外观细节与生成灵活性。

## 6. 应用域与数据集版图

原文把数据集与 benchmark 分成 5 类，配套论文库共收录 126 条。这部分适合直接当索引用：

| 类别 | 条目数 | 代表条目 |
|---|---|---|
| General World Prediction & Simulation | 31 | Ego4D、Ego-Exo4D、EgoVid-5M、InternVid、HowTo100M、Kinetics、Sekai、OmniWorld、WorldScore、WorldModelBench、4DWorldBench、World-in-World |
| Physics & Causality Benchmark | 24 | IntPhys / IntPhys 2、Physion / Physion++、CoPhy、VideoPhy / VideoPhy-2、PhyBench、PhyGenBench、Physics-IQ、VBench-2.0、T2VPhysBench、PisaBench |
| Embodied AI & Robotics | 28 | OXE、DROID、BridgeData V2、LIBERO、CALVIN、RLBench、Meta-World / Meta-World+、RoboCasa、BEHAVIOR-1K、AgiBot-World、RH20T、[[DreamDojo - 基于大规模人类视频的通用机器人世界模型\|DreamDojo]]、WorldArena / 2.0、GigaWorld-0 / 1 |
| Autonomous Driving | 26 | KITTI、nuScenes、Waymo、Argoverse / 2、CARLA、NAVSIM、NuPlan、Occ3D、SemanticKITTI、OpenDV-2K、DrivingDojo、ACT-Bench、DriveArena |
| Interactive Environments & Gaming | 17 | ALE、Procgen、DMC、Crafter、MineRL、NLE、GameGen-X、The Matrix、Matrix-Game-MC、JARVIS-VLA |

两个值得记住的观察：

1. 物理与因果类 benchmark 有 24 条，且大部分是 2024 年之后的新构造。 这本身就说明社区已经意识到 FVD/FID 不够用了。`IntPhys`（2018 年那批 violation-of-expectation 范式）到 `IntPhys 2`、`Physion` 到 `Physion++` 的迭代，是这条线成熟的标志。
2. 通用视频数据集与具身数据集之间存在明显断层。 前者（Ego4D、HowTo100M）量大但无动作标签，后者（OXE、DROID）有动作但量小。这正是 [[Dyna-2 - 世界动作模型百万小时缩放定律|Dyna-2]] 与 [[DreamDojo - 基于大规模人类视频的通用机器人世界模型|DreamDojo]] 这类“用大规模人类视频 + 隐式动作”工作的存在理由——它们在填这个断层。

## 7. 实验与关键信息

### 7.1 先说结论：这篇 survey **没有**传统意义上的统一新实验

这是一个必须明确说明的点：

原文不是提出单个新模型的实验论文，因此没有“自家模型 vs baseline”的统一实验表，也没有统一训练配置、超参数表、消融实验或 benchmark leaderboard。

它的“实验/结果”部分，本质上是对现有研究的评测体系、数据集、基准和指标的系统梳理。因此阅读时不能用“新模型论文”的模板去期待它。

### 7.2 原文真正给出的“评测结构”

作者把 VWM 评估组织成三大类：

1. Visual Quality
    1. 评估生成图像/视频是否清晰、自然、逼真；
    2. 常见指标：PSNR、SSIM、FID、FVD、LPIPS、DreamSim 等。
2. Physical Plausibility
    1. 评估预测是否满足运动学、几何和长时一致性约束；
    2. 典型指标包括：ADE/FDE、Chamfer Distance、AbsRel、Revisit Error、Scene Revisit Consistency 等。
3. Task Performance
    1. 评估 world model 是否真的支持下游决策；
    2. 典型指标包括：Raw Return、Driving Score、Success Rate、MTLC Success Rate、Top-K Accuracy 等。

下面这张图是论文里对 evaluation ecosystem 的总结，很适合后续当评测索引来用：

![](assets/视觉世界模型%20-%20Survey/evaluation.png)

### 7.3 数据集与 benchmark 的系统整理

原文把数据集分成 Foundational（general prediction / physics & causality）与 Domain-specific（robotics / driving / gaming）两大层，具体条目已经在上面的 [[#6. 应用域与数据集版图]] 展开成表，这里不重复。

### 7.4 这篇综述给出的关键观察

作者的关键判断基本可以概括为：

- 只看画面质量会高估模型的世界理解能力；
- 真正可靠的 VWM 必须同时满足视觉合理、物理合理、任务可用；
- 长时程一致性、几何稳定性、因果干预响应，才是下一阶段 evaluation 的重点。

### 7.5 限制与警惕点

这篇 survey 有几个需要警惕的点，其中第 4 条是我读完条目分布后补的：

1. 方法谱系很全，但很多条目是 2025-2026 的新工作
    1. 适合做方向地图；
    2. 但不能把文中所有方法都等同看作“已经验证成熟”——294 条里有相当比例是 preprint，未经独立复现。
2. 对不同 family 的比较仍然偏框架化
    1. 综述擅长“分类”；
    2. 但对于“哪条路线在什么 compute/data regime 下更优”，原文并没有统一定量结论。
3. survey 擅长定坐标，不擅长给 recipe
    1. 它告诉你问题版图；
    2. 但不会直接给你一套训练 generalist VWM 的最优 recipe。
4. 分类学与语料库存在张力
    1. 框架宣称 vision-centric、反对“world model = video generator”；
    2. 但方法条目里 48% 是 diffusion 视频路线，且 2026 年增量几乎全在这一侧；
    3. 结果是：读框架会高估领域的多样性，读条目会低估非视频路线的重要性。 两边都要看。
5. 家族边界正在失效
    1. `Dreamer V4` 被归入 diffusion、`StateSpaceDiffuser` 被归入 latent diffusion，说明四族划分已经开始不足以描述混合架构；
    2. 这不是分类错误，而是提示这套 taxonomy 有保鲜期。

## 8. 未来方向：这篇综述最值得带走的研究判断

作者把未来方向总结成三个“Re-”主题，我觉得非常值得单独记住。

### 8.1 Re-grounding

目标是把 world knowledge 重新锚定到更真实的物理与因果基础上。

重点包括：

- 更复杂的自然物理交互：接触、摩擦、形变、材料性质；
- 人类规则与社会约束：不是只有 physics，还有 social conventions；
- geometry-aware modeling；
- neuro-symbolic hybrid modeling。

### 8.2 Re-evaluation

目标是重新定义“好 world model”的评测方式。

重点包括：

- judge model / VLM-based judge；
- execution-based evaluation，把模型放进真实 planning loop；
- 设计更强调 counterfactual 和复杂动力学的 benchmark。

### 8.3 Re-scaling

目标是研究 scaling law 在 VWM 上到底如何起作用。

重点包括：

- pretraining scaling：走向 generalist VWM；
- inference-time scaling：reasoning before generation；
- 不只是把视频做得更真，而是把物理与因果做得更可靠。

## 9. 个人评注 / 下一步

### 9.1 我对这篇工作的判断

如果把最近一批 VWM / world foundation model 工作放在一起看，这篇 survey 的价值主要体现在三点：

1. 它是方向地图，不是单点模型
    1. 适合用来建立“世界模型全景认知”；
    2. 也适合作为后续读新论文时的归类工具。
2. 它把“世界模型 ≠ 视频生成器”这件事讲清楚了
    1. 这是当前讨论中最容易混淆的点；
    2. 有些工作画面很强，但 world knowledge 很弱。
3. 它对你后续跟踪 VLA / 机器人 / 视频世界建模都有共用价值
    1. 尤其是当你需要判断一篇工作到底是补 physics、补 geometry、补 action conditioning，还是只是补 rendering quality 时，这篇 survey 很有用。

### 9.2 建议后续重点跟踪的分叉

基于这篇 survey，后续我建议把世界模型继续拆成几条主线跟踪，每条都给出“该盯什么、以及本 vault 已有的锚点”：

- Generalist VWM / World Foundation Model
    - 关注统一接口、预训练规模、交互条件表达；
    - 血脉：`Cosmos` → `Cosmos transfer1` → `Cosmos 3`、`Genie` → `2` → `3`、`PAN`；
    - 判据：是否公开了 action space 的表达方式，还是只有 text/image 条件。
- Non-reconstructive / JEPA 路线
    - 关注 embedding 空间的 planning 等价性与训练稳定性；
    - 锚点：[[LeWorldModel - 像素端到端稳定JEPA世界模型|LeWorldModel]]、[[LeJEPA - 世界模型线性可识别性理论|LeJEPA]]、`V-JEPA 2.1`、`VLA-JEPA`；
    - 判据：collapse 怎么防、planning 是否真的比像素 rollout 快一个量级。
- Geometry-aware / 4D World Model
    - 关注 3D/4D 表示、跨视角一致性、长期空间记忆；
    - 锚点：[[GaussianDream - 机器人操控的前馈三维高斯世界模型|GaussianDream]]、`GAF`、`GaussianWorld`、`RynnWorld-4D`；
    - 判据：几何是训练期辅助还是推理期真的在用。
- 显式记忆 / 长时程一致性
    - 这是我认为这篇 survey 里最被低估的一支：`WORLDMEM`（2504.12369）、`VMem`（2506.18903）、`spmem`（2506.05284）、`Echo-Memory`（2606.09803）；
    - 判据：能否在离开再返回同一场景后保持一致（Scene Revisit Consistency 这类指标就是为它设计的）。
- Physics-grounded World Model
    - 关注接触、碰撞、摩擦、材料与 counterfactual evaluation；
    - 锚点：`RoboScape`、`PhysWorld`、`PIN-WM`，评测侧看 `IntPhys 2`、`Physion++`、`VideoPhy-2`。
- World Model for VLA / Robotics
    - 关注 latent rollout 是否真正提升 planning、policy learning 和 task completion；
    - 锚点：[[DreamDojo - 基于大规模人类视频的通用机器人世界模型|DreamDojo]]、[[Dyna-2 - 世界动作模型百万小时缩放定律|Dyna-2]]、[[Fast-WAM - 无测试时未来想象的世界动作模型|Fast-WAM]]、[[DreamZero - 视频世界动作模型的零样本机器人策略|DreamZero]]、[[GigaBrain-0 - 世界模型驱动的VLA数据引擎|GigaBrain-0]]；
    - 判据：[[Fast-WAM - 无测试时未来想象的世界动作模型|Fast-WAM]] 提出的那个尖锐问题——世界模型的价值到底在训练期还是推理期？这个问题对上面每一条主线都成立。

### 9.3 一句话总结

这篇《From Seeing to Knowing the World》非常适合作为“视觉世界模型方向的总目录”。它最大的贡献不是提出新系统，而是把 VWM 的定义、方法分型、I/O 框架、评测维度和未来路线图 一次性讲清楚。

但要用好它，需要一个补充读法：它的框架是 vision-centric 的，语料库却是 video-generation-heavy 的（两类 diffusion 占 48%）。 光读框架会以为领域百花齐放，光读条目会以为世界模型就是可控视频生成。真正的判断力来自把 JEPA、state transition、object-centric、显式几何这四条小体量主线单独拎出来看——它们体量不大，但决定了“世界模型”这个词是否还有超出视频生成的含义。
