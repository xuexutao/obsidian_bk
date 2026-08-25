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

论文给出的形式化表达非常关键。它把 VWM 写成一个条件概率模型：给定历史视觉上下文和交互条件，预测未来世界状态分布。

可以把它读成如下 I/O 数据流：

1. 输入（Input）
    1. 视觉上下文 `v0:t`：从时刻 0 到 t 的视觉观测
        
        - 可能是 RGB 图像 / 视频
        - 也可能包含 depth、optical flow、point cloud、BEV、多视角视频等
            
    2. 交互条件 `ct`：控制未来演化的条件
        
        - 动作信号（action）
        - 语言指令（text instruction）
        - 轨迹、控制命令、任务目标等
            
2. 中间表示（Intermediate Representation）
    1. 先经过 视觉编码器 `E(v0:t)`，把原始观测变成更适合世界建模的表示；
    2. 再通过不同范式的世界模型学习机制，建模世界的时空规律、物理规律和因果结构。
3. 输出（Output）
    1. 未来世界状态 `S_{t+1:T}`，可能有三种形式：
        
        - Latent states：latent rollout，用于 planning / reasoning；
        - Visual states：图像、视频或 3D/4D 视觉结果；
        - Structured outputs：轨迹、对象属性、空间结构、occupancy、action trajectory 等。

### 3.2 三段式框架

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

### 3.3 论文中的总图

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

这一族的关键立场是拒绝像素重建。它认为要预测的不是“未来看起来什么样”，而是“未来的表征是什么”，因为像素里绝大部分信息（纹理细节、光照抖动）对决策毫无用处，强迫模型重建它们等于浪费容量。

- 表征侧血脉：`V-JEPA`（2404.08471）→ `V-JEPA 2`（2506.09985）→ `V-JEPA 2.1`（2603.14482），后两代已经明确把 understanding / prediction / planning 三件事一起做。
- 用现成 encoder 做世界模型：`DINO-WM`（2411.04983）与 `DINO-world`（2507.19468）证明冻结的 DINO 特征空间已经足够支撑 latent rollout 与 planning。
- 训练稳定性问题：JEPA 的老问题是 representation collapse 与训练不稳，[[LeWorldModel - 像素端到端稳定JEPA世界模型|LeWorldModel]]（2603.19312）用 next-embedding prediction + SIGReg 两项损失在 15M 参数下从像素端到端稳定训练，并把 planning 提速最高 48×。
- 理论侧：[[LeJEPA - 世界模型线性可识别性理论|LeJEPA]] 给出了“在高斯潜变量与平稳加噪转移条件下，latent-space planning 与真实状态规划等价”的可证明结论，这是这一族目前唯一有硬理论支撑的部分。

为什么这一族值得单独盯：它是唯一在结构上就不可能被误认为视频生成器的路线（它压根不输出像素），同时也是 2026 年唯一还在加速的路线。

### 5.2 State Transition：最老、也最被 planning 验证过的一族

这一族从 `PlaNet`（2018.11）算起已经八年，是唯一贯穿 RL 时代的血脉：`PlaNet` → `DreamerV1/V2/V3` → `TD-MPC2`，加上 token 化变体 `IRIS`（2209.00588）与 `STORM`（2310.09615）。

它与视频生成路线最本质的区别在目标函数的归属：Dreamer 系的重建损失只是辅助，真正的目标是 policy return；而视频扩散世界模型的目标函数就是生成质量本身，planning 效果是外挂评估出来的。

值得注意的是这一族在 2026 年只有 1 条新增（`NE-Dreamer`，2603.02765），而它最新的旗舰 `Dreamer V4` 已经被归进 diffusion 家族。这不代表路线死了，而是它的思想被吸收进了别的架构——compact latent state + 长时 imagination 这套东西现在以模块形式出现在各家里。

### 5.3 Object-Centric：体量最小，但对 counterfactual 最关键

`G-SWM`（2010.02054）、`SlotFormer`（2210.05861）、`RoboDreamer`（2404.12377）、`Dreamweaver`（2501.14174）、`SlotPi`（2506.10778）、`Dyn-o`（2507.03298）——十六条，八年，是全表最小的一族。

但它承担了一件别人做不了的事：把世界显式分解成 object slots，从而支持组合泛化与对象级 counterfactual。 综述在 Knowledge Learning 里把 causal mechanisms 列为最高一层，而“如果把这个物体换成那个物体、结果会怎样”这类干预，在一个纠缠的 latent 或一段像素序列里是无法干净表达的，在 slot 表示里却是天然操作。

判断：这一族大概不会自己长成主流，但它的评测意义大于方法意义——它定义了“真正理解世界”的一个可检验下界。

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
