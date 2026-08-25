结论先看： 这篇《From Seeing to Knowing the World: A Survey of Vision World Models》可以看作视觉世界模型方向的一篇“总纲式综述”。它最重要的价值，不是提出某个新模型，而是把分散在视频生成、表征学习、机器人、自动驾驶里的 world model 研究，统一收束到一个 Vision Encoding → Knowledge Learning → Controllable Simulation 的分析框架里。

一个必须带着读的补充判断： 这篇 survey 的框架是 vision-centric 的，但它的方法语料库是 video-generation-heavy 的——配套论文库 294 条方法条目里，两类 diffusion 合计 142 条（48%），且 2026 年的增量几乎全在 diffusion 侧。因此如果只读框架图和 taxonomy，很容易得到“视觉世界模型 ≈ 可控视频生成”的错误印象。要纠正它，必须主动去看 JEPA、state transition、object-centric、显式几何这四条体量小但性质不同的主线（本笔记第 5 节专门展开）。

重要性评估：★★★★☆（4/5）

- 为什么值得刷入视野： 它适合作为“世界模型方向的地图册”和后续追踪论文的坐标系。
- 为什么不是 5 星： 它本质上是综述，不是提出新 SOTA 模型；对训练 recipe、数据配比、统一实验复现并没有给出可直接落地的一套新范式；且它的分类学与语料分布之间存在张力（见 7.6）。

## 1. 背景

这篇内容对应的微信公众号文章标题是《图文速览|VWM综述解读：视觉世界模型》，主线指向官方论文《From Seeing to Knowing the World: A Survey of Vision World Models》。原论文项目页为[AIWorldLab Survey](https://aiworldlab.github.io/survey/)，官方 PDF 为[preprint.pdf](https://aiworldlab.github.io/survey/preprint.pdf)，预印本落在[Preprints 页面](https://www.preprints.org/manuscript/202604.2072)。

论文要解决的问题不是“再造一个世界模型”，而是先回答一个更基础的问题：什么才算 Vision World Model（VWM），以及我们该如何系统比较不同 world model 路线。

作者认为，现有相关工作虽然都在讨论“世界建模”，但社区来源非常分散：

- 一部分来自视频生成，强调未来视觉内容的逼真合成；
- 一部分来自表征学习，强调在 latent / embedding 空间里做预测；
- 一部分来自强化学习与机器人，强调 planning、control 和 imagined rollout；
- 还有一部分来自自动驾驶、游戏环境、物理推理等应用领域。

因此，同样叫“world model”，不同论文常常在以下维度上互不兼容：

- 问题定义不统一：到底是在做未来帧生成、latent rollout，还是行动条件下的可控仿真；
- 方法分类不统一：autoregressive、diffusion、JEPA、RSSM、object-centric 常被分开讨论；
- 评测协议不统一：有的看 FVD/FID，有的看 planning score，有的看 physics benchmark；
- 研究目标不统一：有的偏视觉 realism，有的偏物理一致性，有的偏下游控制成功率。

所以这篇综述的核心贡献，是试图给出一个视觉中心（vision-centric）的统一视角：不是把 vision 当作输入模态之一，而是把“视觉如何揭示世界变化”当作 world model 的核心出发点。

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

### 4.2 Sequential Generation 内部：四个代表工作的具体 pipeline

上面的列表只能当索引用。这一节挑四篇最有代表性的，把 pipeline 和目标函数写出来——因为**同属"自回归"家族，它们的 token 化对象完全不同**，而这恰恰是理解这一族的关键。

#### 4.2.1 GAIA-1：把"推理"和"渲染"彻底拆开

`GAIA-1`（2309.17080，Wayve）是这一族最早的旗舰，它的结构值得单独记住，因为后来很多工作都在重复这个分工：

$$\mathcal{L}_{\text{world model}}=-\sum_{t=1}^{T}\sum_{i=1}^{n}\log p\big(z_{t,i}\mid z_{<t},\, z_{t,j<i},\, c_{\le t},\, a_{<t}\big)$$

三个组件与具体数字：

| 组件 | 参数量 | 作用 |
|---|---|---|
| 图像 tokenizer | 0.3B | 288×512 → $n=18\times32=576$ token，词表 $K=8192$，压缩率约 470:1 |
| 世界模型 | 6.5B | 在 token 序列上做 next-token prediction |
| 视频扩散解码器 | 2.6B | 把 token 翻译回高清视频（3D U-Net，DDIM 50 步） |

序列布局是每个时刻交错 **文本 32 + 图像 576 + 动作 2 = 610** 个 token，取 $T=26$ 帧（6.25 Hz，约 4 秒），总上下文 $26\times610=15{,}860$。

几个容易记错的点：

- **损失只加在图像 token 上**（内层求和到 $n=576$）。文本与动作是条件输入，不是预测目标。
- **动作不做离散化**：2 个标量（speed、curvature）各经一个线性层映到 $d$ 维，绕开码本。
- tokenizer 不是普通 VQ-VAE，而是 VQ-GAN 家族，且额外加了 **DINO 蒸馏**——把量化后的特征用余弦相似度拉向冻结 DINO 特征（$\lambda=0.1$），目的是让 token 空间偏向**语义**而不是高频细节。这一步是"让 token 空间值得被 transformer 推理"的关键。
- 训练时条件 dropout 为 **20% 无条件 / 40% 动作条件 / 40% 文本条件**，推理用 CFG 作用在 **logits** 上：$l_{\text{final}}=(1+t)\,l_{\text{cond}}-t\,l_{\text{uncond}}$。
- 一个不常见的细节：guidance 在**帧内随 token 线性递减**，并**随未来帧数按 cosine 衰减**（越远的未来给越少引导）。
- 采样用 top-k，动机很实用：argmax 会退化成重复循环，全分布采样会漂出分布。

**为什么不直接用一个纯扩散视频模型？** 原文的分工是：世界模型负责推理场景的高层组成与动力学，扩散模型**只负责把 latent 翻译成高质量视频**。这样做的收益是可以把 LLM 那一整套机器搬过来——因果 transformer、logits 上的 CFG、top-k 采样，以及可预测的 scaling。它确实拟合了幂律 $f(x)=c+(x/a)^b$，并声称用**少于 20× 的算力**就能预测出 6.5B 模型的最终交叉熵。

> ⚠️ 但要注意这是**单模型的算力外推曲线，不是 Chinchilla 式的 compute-optimal 参数/数据配比研究**，$a,b,c$ 的数值也没有公布。把它说成"给出了世界模型的 scaling law 最优前沿"是过度解读。另外论文没有说明 token 具体如何注入 U-Net（拼通道还是 cross-attention），也没有明确说扩散是在像素空间还是 latent 空间——前者是我的强推断（全文没有引入第二个 VAE），但**原文没有这句话**。

#### 4.2.2 Genie：从无动作标注的视频里"发明"动作空间

`Genie`（2402.15391）是这一族里唯一**自己造出动作空间**的工作，也是我认为整个 survey 里机制最巧妙的一篇。它的核心是 Latent Action Model（LAM）：

$$\tilde{a}_{1:t} = \mathrm{Enc}_{\text{LAM}}(x_{1:t}, x_{t+1}),\qquad \hat{x}_{t+1} = \mathrm{Dec}_{\text{LAM}}(x_{1:t}, \tilde{a}_{1:t})$$

$$\mathcal{L}_{\text{LAM}} = \lVert x_{t+1} - \hat{x}_{t+1}\rVert^2 + \mathcal{L}_{\text{VQ}},\qquad |\mathcal{A}| = 8$$

**关键在于一个非对称信息瓶颈：** encoder 能看到未来帧 $x_{t+1}$，但 decoder 看不到——它只拿到历史 $x_{1:t}$ 和那个极小的 latent action $\tilde a_t$。因为历史已经把外观解释掉了，这个 3 bit 的通道能携带的唯一**非冗余**信息就是"从过去到未来的变化"，也就是动作。码本压到 8 个则是为了**人类可玩**。

动力学模型用 MaskGIT 式的目标：

$$\mathcal{L}_{\text{dyn}} = -\sum_{t=2}^{T}\sum_{i} \log p\big(z_t^{(i)} \mid \tilde{z}_{1:t-1},\, \tilde{a}_{1:t-1}\big)$$

三个组件加起来 200M（tokenizer）+ 300M（LAM）+ 10.1B（dynamics）= **10.7B**（摘要里的"11B"是取整）。数据是 55M 筛到 **6.8M 段 16 秒视频、约 3 万小时**。推理时 **LAM 的 encoder/decoder 都被丢掉**，只留 8 个码本项，由用户每帧给一个离散动作。

架构上用 **ST-transformer**：帧内空间注意力 → 跨时间的因果注意力 → **只有一个 FFW**（省掉空间注意力后那个）。代价随帧数线性而非二次增长。消融（同量级 ~200M）：ST-ViViT 的 FVD **81.4** / ΔPSNR **1.66**，而纯空间 ViT 是 114.5 / 1.39，C-ViViT 是 272.7 / 1.37。

可控性指标定义得很干净：$\Delta_t\text{PSNR} = \text{PSNR}(x_t,\hat x_t) - \text{PSNR}(x_t,\hat x'_t)$，其中 $\hat x_t$ 用推断出的动作、$\hat x'_t$ 用随机动作。越大说明动作越真的在控制画面。另一个消融很说明问题：**LAM 建在像素上比建在 token 上可控性更好**（Platformers ΔPSNR 1.91 vs 1.33），因为 tokenization 会破坏运动信息。

> ⚠️ 一个必须带着的批判：后续工作 [What Do Latent Action Models Actually Learn?](https://arxiv.org/abs/2506.15691) 指出这个目标与 PCA 有关，且**并不能内在地把"可控动作"与"外生噪声"分开**——能否恢复真实动作取决于数据生成策略的性质。所以"latent action 就是动作"这个说法需要打折。另外原文**从未提到 classifier-free guidance**，也没说明 latent 网格尺寸（160/4×90/4 不是整数，说明有 padding/crop），这些不要替它补。

#### 4.2.3 iVideoGPT：用非对称 token 预算换交互速度

`iVideoGPT`（2405.15223）解决的是一个很具体的工程问题：**交互式 rollout 里每步都要自回归解码，token 太多就没法用。**

它的答案是条件 VQGAN + 非对称 token 预算：上下文帧走一条通路给足 token，未来帧走另一条通路只给很少：

$$z_t^{(1:N)}=E_c(o_t)\ \ \text{（上下文，}N=16\times16=256\text{）},\qquad z_t^{(1:n)}=E_p(o_t \mid o_{1:T_0})\ \ \text{（未来，}n=4\times4=16\text{）}$$

**16× 的压缩**。之所以成立，是因为只有第一帧需要完整空间细节，后续帧在给定上下文后大部分是冗余的。条件注入用**多尺度 cross-attention**：query 来自预测流，key/value 来自上下文特征。

$$\mathcal{L}_{\text{pre-train}} = -\sum_{i=(N+1)T_0+1}^{L} \log p(x_i \mid x_{<i})$$

注意求和下界——**损失只加在未来帧的 token 上**，跳过上下文帧。动作不是单独的 token，而是**线性投影后加到 slot token 的 embedding 上**。预训练用 Open X-Embodiment 等 35 个数据集共 **140 万条轨迹**（无动作标注），微调时**连 tokenizer 一起更新**（参数高效的变体反而更差）。

这个设计的意义：每个环境步只需约 16 次自回归解码而不是 256 次，MBRL/planning 循环里要查询模型上千次，这个差别决定了能不能用。

#### 4.2.4 OccWorld：世界状态是 3D occupancy，不是像素

`OccWorld`（2311.16038）把"世界状态"换成 3D occupancy，于是**场景预测变成了对码本索引的分类，规划变成了对位移的回归**，两件事进一个损失：

$$\mathcal{J}_{w,d_{\text{ego}}} = \sum_t \Big(\sum_j \mathcal{L}_{\text{soft}}\big(\hat z^t_{j,0},\, \mathcal{C}(z^t_{j,0})\big) + \lambda_2 \mathcal{L}_{L2}\big(d_{\text{ego}}(\hat z^t_0),\, p^t\big)\Big)$$

数据流：occupancy $200\times200\times16$（17 类）→ 每个 voxel 的类别查 embedding → **沿高度维拼接**成 BEV 张量 $200\times200\times128$ → 2D CNN 降采样 4× → token 网格 $50\times50$，码本 512 个 128 维码 → 空间时序 transformer → 解码回 occupancy。

两个设计值得记：

1. **ego token** $z_0$ 与场景 token 拼在一起，预测的是**位移**而不是绝对位姿。消融显示 ego token 上的时序注意力至关重要——ego 预测差会"误导"场景预测，反之亦然。这是"联合建模自车与场景"的实证支持。
2. 用 **2×2 window merging（stride 2）** 做多尺度（200/100/50/25），每个尺度一个子世界模型，再用 U-Net 式聚合保证空间一致性。

它的价值在于证明了：**不需要实例级或地图级监督**，只靠 occupancy 的自回归就能做规划——这是与 UniAD / VAD 那条重监督路线的根本差别。

> 说明：17 类与 $200\times200\times16$ 来自官方 config 而非论文正文；论文也没有明确写帧率（按 nuScenes 关键帧惯例是 2 Hz）。

#### 4.2.5 一张表看清"自回归"内部的差异

| | 离散化的对象 | 条件来自 | 关键压缩 |
|---|---|---|---|
| GAIA-1 | 8192 词表的图像 token | 文本 + speed/curvature | 470:1，扩散负责渲染 |
| Genie | VQ token + **8 个自己学出来的动作** | 推断出的 latent action | 3 bit 动作瓶颈 |
| iVideoGPT | 双码本（上下文/未来分离） | 真实动作 | 256 → 16 token/帧 |
| OccWorld | 512 码的 BEV 场景 token | ego token（与场景联合预测） | 200×200 → 50×50 |

**这张表是我认为本节最值得带走的东西：**四篇都叫"自回归世界模型"，但离散化的对象分别是像素、动作、冗余度、几何。所以读一篇新的自回归世界模型时，第一个要问的问题不是"它是不是 AR"，而是**"它把什么东西离散化了"**。

### 4.3 Diffusion-based Generation 内部：为什么"自回归扩散"是这一族的真正拐点

这一族占了全部条目的 48%，但它内部的 latent diffusion 与 autoregressive diffusion 之分，比家族之间的差别更值得搞清楚。

#### 4.3.1 目标函数：先把三种写法的关系理清

三种常见写法（$\alpha_t,\sigma_t$ 为噪声调度）：

$$\text{DDPM }\epsilon\text{-预测：}\quad \mathcal{L}=\mathbb{E}\big\lVert\epsilon-\epsilon_\theta(z_t,t,c)\big\rVert_2^2,\qquad z_t=\alpha_t z_0+\sigma_t\epsilon$$

$$\text{v-预测：}\quad v\equiv\alpha_t\epsilon-\sigma_t x$$

$$\text{Flow matching / rectified flow：}\quad \mathcal{L}=\mathbb{E}\big\lVert v_\theta(z_t,t,c)-(\epsilon-z_0)\big\rVert^2$$

各家的**实际选择**（这里最容易出错，所以逐一核对过）：

| 模型 | 目标函数 |
|---|---|
| `Cosmos`（2501.03575） | **EDM，$x_0$-预测** + 不确定性重加权 |
| `Cosmos-Predict2.5`（2511.00062） | **rectified flow 速度场** $v=\epsilon-x$ |
| `GAIA-2`（2503.20523） | **flow matching 速度场** $v=x-\epsilon$ |
| `Vista`（2405.17398） | EDM 框架 + v-预测 |
| `DIAMOND`（2405.12399） | **EDM，$x_0$-预测** |

> ⚠️ **两个陷阱。**（1）**没有任何一版 Cosmos 用 Salimans–Ho 意义上的 v-预测**——Predict1 是 EDM 预处理下的 $x_0$-预测，Predict2.5 是 flow matching 速度场，两者不是一回事。（2）**符号约定互相冲突**：SD3 与 Cosmos-Predict2.5 取 $t{=}0$ 为数据，故 $v=\epsilon-x$；GAIA-2、Lipman、Liu 取 $t{=}0$ 为噪声，故 $v=x-\epsilon$。写速度场时必须同时说明方向，否则符号是错的。
>
> 至于"扩散与 flow matching 等价"——Cosmos 原文自己就说了两者"理论上等价，主要区别在预处理设计与超参数选择"。精确关系是 $\hat u=\hat\epsilon-\hat x$，且 DDIM ≡ flow matching 的 Euler 步。但注意这说的是**框架可互换，不是任意两个训好的模型等价**。

#### 4.3.2 Diffusion Forcing：一个改动，把 teacher forcing 与全序列扩散统一了

`Diffusion Forcing`（2407.01392）是理解 autoregressive diffusion 这一支的钥匙。核心改动只有一句话：**给每一帧独立采一个噪声等级**。

$$\mathbb{E}\sum_{t=1}^{T}\Big[\big\lVert\epsilon_t-\epsilon_\theta(z_{t-1},x_t^{k_t},k_t)\big\rVert^2\Big],\qquad k_{1:T}\sim\text{Uniform}([K]^T)$$

关键在 $k_{1:T}$ 是**逐帧独立**采的。原文的 Theorem 3.1 说这优化的是一个重加权 ELBO，并且在适当条件下是对**所有噪声等级序列同时**成立的似然下界。

**"加噪即遮蔽"这个视角最值得记：** $k_t=0$ 表示未遮蔽，$k_t=K$ 表示完全遮蔽。于是——

- teacher forcing = 沿**时间轴**遮蔽；
- 全序列扩散 = 沿**噪声轴**遮蔽；
- Diffusion Forcing 因为可以自由采整个噪声向量 $k_{1:T}$，**同时涵盖两者**。

还有一个更漂亮的解读（原文附录 B.5）：$p(z_t\mid z_{t-1},x_t^k)$ 在 $k=0$ 时是**后验**、在 $k=K$ 时是**先验**，一个网络参数化了两者之间的一切。

**长 rollout 稳定化的机制比通常转述的更微妙。** 它并**不是**把生成的干净帧重新喂回去，而是把上一步的 latent 维持在一个**小但非零**的噪声等级 $0<k_{\text{small}}\ll K$ 上，用 $z_t^{k_{\text{small}}}$ 继续。理由是：带噪的 $x_t^{k_{\text{small}}}$ 对训练分布来说是**分布内**的，因为 Diffusion Forcing 本来就是从带噪的过去观测训练的。结果是能稳定 rollout 到 **1000 帧**，远超训练时的视界，而 teacher forcing 与全序列扩散在此都已发散。

> ⚠️ 两个必须标注的点：（1）**"history guidance" 不在这篇论文里**，它属于后续工作 `DFoT`（2502.06764，History-Guided Video Diffusion）；（2）这篇论文的**视频结果只有定性展示，没有 FVD/LPIPS 表格**，定量表格只覆盖 D4RL maze 规划与时间序列。模型也很小（24–36M 的 GRU+U-Net，像素空间），作者自陈没做 scaling 研究。

`DFoT` 补上的那个洞察很关键：**无条件 score 就是把 history 用噪声完全遮蔽后的条件 score**（$\mathcal{H}=\varnothing$），于是"对变长历史做 CFG"不需要改架构：

$$\nabla\log p_k(x_\mathcal{G}^k)+\sum_i \omega_i\big[\nabla\log p_k(x_\mathcal{G}^k\mid x_{\mathcal{H}_i}^{k_{\mathcal{H}_i}})-\nabla\log p_k(x_\mathcal{G}^k)\big]$$

它由此做到**单图 rollout 862 帧**。另一个有用的负面结论：**二值 history dropout 的表现不如逐帧噪声**，因为 token 利用率低。

#### 4.3.3 DIAMOND：扩散在 RL 世界模型里赢在哪

`DIAMOND`（2405.12399）常被简化成"扩散比离散 token 好"，但原文给的理由具体得多：**离散 tokenizer 会让奖励相关的小物体在时间上不一致**。在 Asterix 里"一个敌人（橙色）在第二帧变成了奖励（红色），随后又变回来"；Breakout 的砖块与分数会闪烁；Road Runner 的奖励点渲染不稳。DIAMOND 没有这些问题——Breakout 里打到红砖时分数还能正确 **+7**。

$$\mathcal{L}(\theta)=\mathbb{E}\Big[\big\lVert\mathbf{F}_\theta(c_{\text{in}}^\tau\mathbf{x}_{t+1}^\tau,y_t^\tau)-\tfrac{1}{c_{\text{out}}^\tau}(\mathbf{x}_{t+1}^0-c_{\text{skip}}^\tau\mathbf{x}_{t+1}^\tau)\big\rVert^2\Big]$$

**为什么选 EDM 而不是 DDPM 才是真正的关键：** 当 $\sigma\gg\sigma_{\text{data}}$ 时 $c_{\text{skip}}\to0$，训练目标退化成**干净图像**本身；而 DDPM 的 $\epsilon$-预测在噪声主导时会让模型学成**恒等函数**，在采样起点给出很差的 score 估计。量化对比：**1 步 EDM 比 10 步 DDPM 还稳**。最终选 $n=3$ 步（Euler），因为 Boxing 存在多模态转移，1 步去噪会在多个结果之间插值而糊掉。

Atari-100k：mean HNS **1.46**、IQM 0.64、11 个游戏超越人类。$n{=}3$ vs $n{=}1$ 的消融在 top-10 游戏上是 **3.052 → 1.962**，Boxing 最惨（86.9 → 41.9）。

#### 4.3.4 动作条件到底怎么注入：三篇论文给出了互相矛盾的结论

这是我认为本节最有价值的一张表，因为它**不是设计偏好，而是三组各自做过消融的实证结论**：

| 论文 | 机制 | 原文理由 |
|---|---|---|
| `DIAMOND` | U-Net res-block 里的 **Adaptive GroupNorm**（动作 + 扩散时间）；观测走**通道拼接** | 消融：帧堆叠 CS:GO FVD **34.8** vs cross-attention **81.4** |
| `GAIA-2` | ego 动作走 **AdaLN**；3D 框/元数据走 cross-attention | "动作影响每一个空间 token，AdaLN 提供显式信息通道，无需依赖可学习的注意力机制" |
| `Vista` | **cross-attention** + 零初始化投影 + LoRA | "相比加性 embedding 等方案，收敛更快、可控性更强" |
| `Cosmos` 扩散版 | 动作 MLP → **加到 timestep embedding 上** | — |
| `Cosmos` 自回归版 | 动作 MLP → **cross-attention** | 同一个动作，两个家族注入方式不同 |

**三家做了同一个消融，得出了方向相反的结论。** 但都站得住脚，因为决定因素是**条件信号本身的结构**：

- GAIA-2 的动作是一对全局标量（speed、curvature），影响每个 token → AdaLN 合适；
- Vista 要复用五种异质动作格式（命令/轨迹/速度/角度/目标），需要 attention 的灵活性；
- DIAMOND 的历史是稠密的逐像素信息 → 直接拼通道的归纳偏置最好。

所以正确的结论不是"哪种注入方式更好"，而是**注入方式应当匹配条件信号的结构**。顺带一提：**这些论文里没有一篇用 ControlNet 式方案做动作条件**（`Cosmos-Transfer` 是 ControlNet 式，但用于模态迁移而非动作）。

#### 4.3.5 Cosmos 的 tokenizer 与条件注入

`Cosmos`（2501.03575）作为这一族的基础设施代表，有几个数字值得记：

$$\mathcal{L}(D_{\theta})=\mathbb{E}_{\sigma}\left[\frac{\lambda(\sigma)}{e^{u(\sigma)}}\mathcal{L}(D_{\theta},\sigma)+u(\sigma)\right],\qquad \lambda(\sigma)=\frac{\sigma^{2}+\sigma_{\text{data}}^{2}}{(\sigma\cdot\sigma_{\text{data}})^{2}}$$

其中 $u(\sigma)$ 是一个 **MLP 参数化的逐噪声等级不确定性**——模型不确定的噪声等级上损失被自动降权。

tokenizer 侧：**时间上因果**（因果时间卷积 + 因果时间注意力，因子化 3D 卷积），前端接**小波变换**，用 LayerNorm 而非 GroupNorm。latent 形状 $(1+T')\times H'\times W'\times C$，前导的 "+1" 让第一帧成为独立 token，从而**图像（$T=0$）与视频统一在一个 latent 空间**。连续版是**纯 AE、latent 维 16、没有 KL**；离散版用 **FSQ，levels $(8,8,8,5,5,5)$ → 词表 64,000**。

> ⚠️ 一个很容易写错的区分：**因子化的是 tokenizer，不是 DiT。** Cosmos 的 DiT 主干用的是**完整时空自注意力**（block 结构为 FA→CA→MLP，其中 FA 是 flattened token 上的 full attention），只有 tokenizer 用了因子化 3D 卷积与因子化时空注意力。把 Cosmos 的 DiT 写成"factorized spatio-temporal attention"是错的。相比之下 GAIA-2 的 denoiser 才是真正的空间-时间因子化 block。

Video2World 的条件注入是三件事一起：条件帧沿**时间维拼接**、一个**二值 mask 沿通道维拼接**、以及对条件帧**加增广噪声**（$P_{\text{mean}}=-3.0$、$P_{\text{std}}=2.0$），损失排除条件帧位置。

AdaLN-LoRA 是个实用技巧：把 AdaLN 的密集投影做低秩分解，**参数量从 11B 降到 7B（-36%）而性能持平**——因为 AdaLN 占大量参数却几乎不贡献 FLOPs。

#### 4.3.6 GAIA-1 → GAIA-2：扩散从"解码器"迁移进"动力学模型"

这一对是整篇 survey 里最能说明"家族边界正在失效"的案例。同一个团队（Wayve）两年内做了一次**架构反转**，而不是简单放大：

| | **GAIA-1**（2023，被归入 Sequential） | **GAIA-2**（2025，被归入 Diffusion） |
|---|---|---|
| 范式 | 自回归 next-token + 独立扩散**解码器** | 单一 **latent flow matching** 模型 |
| latent | **离散**，VQ 最近邻，词表 8192 | **连续**，64 通道，KL 权重 1e-6 |
| 每帧 token | 576（$18\times32$，$D{=}16$） | $14\times30$ 连续，空间 $32\times$ / 时间 $8\times$ |
| 目标函数 | 图像 token 的**交叉熵** | **速度回归** $v = x - \epsilon$ |
| 参数 | 0.3B tokenizer + 6.5B 世界模型 + 2.6B 视频解码器 | 0.085B enc / 0.2B dec + **8.4B** denoiser |
| 相机 | **1** 路（288×512） | **5** 路（448×960），空间与视角联合注意力 |
| 动作条件 | 序列里的 2 个**动作 token** | **AdaLN** 调制 |
| 序列长度 | **15,860** | 12,600 |
| 帧率处理 | 25 Hz → **降到 6.25 Hz**，靠解码器做时间超分 | 原生，不降采样 |

三个值得单独记住的点：

1. **GAIA-1 其实已经在用扩散了——但只用在解码器上。** 而且它的 2.6B 视频解码器选的是 **v-parameterization**，理由与 DIAMOND 的稳定性论证同源：原文说这样"**避免了不自然的色偏**并保持长时一致性"。所以 GAIA-1 → GAIA-2 的真实轨迹是：**扩散从解码器迁移进了动力学模型本身**。

2. **离散 tokenizer 的解码器被直接扔掉了。** 原文说得很直白：解码器"仅用于训练图像自编码器，最终 GAIA-1 只包含离散编码器 $E_\theta$。由于解码器是在单图上训练的，解码成视频时缺乏时间一致性。" —— 这**正是 DIAMOND 在 Atari 上实测到的那个病症**（小物体闪烁）。两个互不相关的团队在两个完全不同的领域（驾驶 / Atari）撞上了同一个问题：GAIA-1 的补丁是外挂一个时序扩散解码器，GAIA-2 的做法是**干脆去掉离散瓶颈**。这个跨论文的收敛是这批材料里最强的一条证据。

3. **GAIA-1 有拟合好的 scaling law，GAIA-2 没有。** 这是换成连续 latent 时**丢掉的一项能力**——回归损失不像 next-token 交叉熵那样能跨 tokenizer 比较。如果要谈世界模型的 scaling，这个代价值得写一句。

> 顺带解释了一个我先前标为"可疑"的数字：GAIA-2 那个 80% 的独立条件 dropout 看起来异常高，但 GAIA-1 就已经用 **20% 无条件 / 40% 动作 / 40% 文本** 的激进 dropout。Wayve 一贯如此，所以它仍是"原文陈述但不寻常"，至少不再是孤例。

### 4.4 七类代表性子设计（含配套论文库的实际规模）

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

### 4.5 为什么这篇 survey 读起来像“视频世界模型综述”

这是读这篇 survey 时最值得警惕的一点，而且它不是综述作者的表述问题，而是领域现状的真实投影：

1. 两类 diffusion 合计 142 条，占全部方法条目的 48%。 也就是说，接近一半的“视觉世界模型”工作，本体是视频扩散模型加上动作条件。综述在正文里反复强调“world model ≠ video generator”，但它的方法版图本身就被视频生成占了半壁江山。
2. 2026 年的新增几乎全在 diffusion 侧。 2026 年的新条目里 Latent Diffusion 有 21 条、Autoregressive Diffusion 有 14 条，而 Visual AR、State-Space、Object-Centric 三类 2026 年各只有 1 条。领域的增量投入高度集中。
3. 非视频路线的“时间形状”完全不同。 State-Space 的高峰在 2023（12 条）与 2025（17 条），Object-Centric 一直维持在每年 4-6 条的小体量，只有 JEPA 是唯一逐年递增的一类（0 → 5 → 8 → 10），2026 年条目数已经超过它此前任何一年。

所以更准确的读法是：这篇 survey 的框架是 vision-centric 的，但它的语料库是 video-generation-heavy 的。 如果只读三段式框架和 taxonomy 图，很容易得到“视觉世界模型 ≈ 可控视频生成”的印象；要纠正这个印象，必须主动去看下面几条被条目数稀释掉的主线。

### 4.6 一个容易被忽略的信号：Dreamer V4 被归到了 diffusion 家族

论文库把 `PlaNet`、`DreamerV1/V2/V3` 都放在 Latent State-Space Modeling 下，但把 `Dreamer V4`（2509.24527）放进了 Autoregressive Diffusion。

这个分类不是随手为之，它记录了一次真实的路线迁移：state-space 血脉里最有代表性的一支，在最新一代把 transition model 换成了 diffusion-based 的生成式 rollout。同类信号还有 `StateSpaceDiffuser`（2505.22246，被归到 Latent Diffusion）与 `S5WM`、`DMWM` 这类在 SSM 里塞进扩散/世界建模目标的工作。

**GAIA-1（Sequential）→ GAIA-2（Diffusion）是同一现象的另一个版本**，而且更彻底：同一团队、同一应用领域，两年内跨了家族（见 4.3.6）。所以论文库里 `GAIA-1` 与 `GAIA-2` 被分到两个不同的一级家族，这**不是分类错误，而是这套 taxonomy 有保鲜期的直接证据**。

判断：四大家族的边界正在从“互斥路线”变成“可组合模块”。 后续读新论文时，与其问“它属于哪一家族”，不如问“它在表示层用了什么、在 transition 层用了什么、在解码层用了什么”这三个独立问题。

#### 4.6.1 一条贯穿三篇论文的因果链：离散瓶颈的代价

把前面几节的证据串起来，会得到一条我认为比任何单篇结论都更有价值的线索——**三个互不相关的团队，在三个完全不同的领域，撞上了同一个问题**：

| 论文 | 领域 | 观察到的病症 | 采取的对策 |
|---|---|---|---|
| `GAIA-1`（2023） | 驾驶 | 离散 tokenizer 的解码器"在单图上训练，解码成视频时缺乏时间一致性" | 外挂一个 2.6B 时序扩散解码器（v-prediction，理由是避免色偏） |
| `DIAMOND`（2024） | Atari | 离散 token 让奖励相关小物体闪烁：敌人变成奖励再变回来、砖块与分数抖动 | 整个世界模型换成 EDM 扩散 |
| `GAIA-2`（2025） | 驾驶 | —— | **直接去掉离散瓶颈**，改连续 latent + flow matching |

这条链说明：**离散化在"能用 LLM 那套机器"和"时间一致性"之间存在一个真实的权衡**，而不是一个纯粹的实现选择。GAIA-1 用外挂解码器打补丁，DIAMOND 换掉整个生成器，GAIA-2 则回到源头拆掉瓶颈——三种不同层次的应对，指向同一个诊断。

代价也要一起记：GAIA-2 换成连续 latent 后**丢掉了 scaling law 拟合能力**（回归损失不像 next-token 交叉熵那样能跨 tokenizer 比较）。所以这不是"连续一定更好"，而是**你想要可预测的 scaling，还是想要时间一致性**。

### 4.7 这篇综述的真正方法论价值

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

#### 5.4.1 GaussianWorld：把场景演化拆成三个先验

`GaussianWorld` 的状态是 25,600 个高斯 $\mathbf{g}=\{\mathbf{p},\mathbf{s},\mathbf{r},\mathbf{c},\mathbf{f}\}$（位置、尺度、旋转、语义概率、时序特征）。它把 occupancy 预测重写成 4D 预测，并按三个先验分解：

1. **自车对齐**：$\mathbf{g}^T_A=\text{Align}(\mathbf{g}^{T-1},\mathbf{M}_{\text{ego}})$——**只对位置做仿射变换，零参数**；
2. **动态运动**：只精修**位置**，且由动态类语义权重 $I(\cdot)$ 门控；
3. **场景补全**：出界的高斯丢弃、等量随机重初始化，**这一支才更新全部五个属性**。

这个设计的巧处在于**每个因子只更新它物理上能改变的属性**。自车运动是闭式变换所以不花参数，动力学只碰位置。代价几乎为零：228ms / 7030M，而时序融合类基线是 379–382ms / ~10GB；mIoU 22.13 vs GaussianFormer-B 的 19.73。

消融值得看：去掉自车对齐掉到 **18.47（比单帧的 19.73 还差）**，去掉动力学 21.17，而**去掉补全直接训练崩溃**——说明补全不是可选项。

> ⚠️ 原文**没有给高斯到 voxel 的 splatting 公式**，也没给损失权重；这些要回到 GaussianFormer（2405.17429）：$\Sigma=RSS^\top R^\top$、$g(p)=\exp(-\tfrac12(p-m)^\top\Sigma^{-1}(p-m))\,c$、occupancy $=\sum_{i\in\mathcal{N}(p)}g_i$。另外 Table 1（22.13）与 Table 3（21.50）的差异原文未解释。

#### 5.4.2 GAF：动作是"读"出来的，不是回归出来的

`GAF` 的映射是 $\mathcal{F}_\Theta:\{g(x),t\}\mapsto\{\mu,\Delta\mu,f\}$，即 3DGS 加一个**可学习位移** $\Delta\mu\in\mathbb{R}^3$。同一个场从三个 query 取三种输出：当前帧渲染、未来帧渲染（用 $\mu+\Delta\mu$）、以及动作。

动作那一路是整篇最漂亮的地方——**用 ICP 在夹爪高斯上拟合刚体变换**：

$$T^{t\to t+\Delta t}=\arg\min_{T\in SE(3)}\sum_{k\in\text{gripper}}\lVert T(\mu_k^t)-\mu_k^{t+\Delta t}\rVert^2$$

因为夹爪是刚体，对它的高斯位移拟合一个 $SE(3)$ 就直接得到动作，**不需要额外的动作回归头**。RLBench 9 任务成功率 54.11% vs ManiGaussian 39.22%，NVS PSNR 27.92 vs 16.38。

> ⚠️ 一个真实的方法空缺：**原文从未说明夹爪高斯是如何被识别出来的**——没有 mask、没有分类器、没有机器人模型。而整个 ICP 动作读出都依赖这一步分割，这是个实质性的缺口。

#### 5.4.3 显式记忆：WorldMem 与 VMem 的两种检索

这一支我认为被严重低估，它解决的是"离开再回来，场景还一致吗"。

`WorldMem`（2504.12369）的记忆单元是 $(\mathbf{x}^m_i, \mathbf{p}_i, t_i)$——**latent token**（不是像素）、位姿 $\mathbf{p}\in\mathbb{R}^5$、时间戳。检索用置信度贪心：

$$\boldsymbol{\alpha} = \mathbf{o}\cdot w_o - \mathbf{d}\cdot w_t$$

$\mathbf{o}$ 是 FOV 重叠率（蒙特卡洛估计：在半径 30m 球内采 10,000 点，算两个相机同时可见的比例），$\mathbf{d}$ 是时间距离。取 8 帧记忆。位姿用 **Plücker 射线** $\mathbf{l}_{uv}=(\mathbf{c}\times\mathbf{d}_{uv}, \mathbf{d}_{uv})\in\mathbb{R}^6$ 编码，且**状态只加到 Q/K 上、不加到 V**。

它的机制建立在 Diffusion Forcing 之上：**记忆帧被赋予 $k_{\min}$（近乎干净），上下文帧取 $[k_{\min},k_{\max}]$，待生成帧取 $k_{\max}$**。所以"3D 一致性"来自**条件在正确的过去帧上，而不是拉长窗口**。600 帧记忆 + 100 帧生成下 PSNR 19.32 / LPIPS 0.1429 vs DF 基线 18.04 / 0.4376。检索只占推理 10–20%。原文自陈局限：**显存仍随记忆线性增长**。

`VMem`（2506.18903）更进一步，用 **surfel 作为可见性索引**：$\mathbf{s}_k=(\mathbf{p}_k, \mathbf{n}_k, r_k, \mathcal{I}_k)$，其中 $\mathcal{I}_k$ 是**观测过该 surfel 的历史视角集合**。检索方式是"渲染投票"：把目标位姿平均后光栅化 surfel，取**被渲染次数最多的 top-$K$ 视角索引**。

**关键洞察：surfel 是可见性索引，不是重建。** 所以 CUT3R 的深度误差不会传导到像素上——原文说"只要成功检索到最相关的过去视角，方法就保持稳健"。这也解释了它为什么同时打败相机距离检索与 FOV 重叠检索：**两者都忽略遮挡**，而光栅化天然处理遮挡。环形轨迹 $K{=}4$：LPIPS 0.381 vs FOV 的 0.424 vs 时间最近的 0.794，旋转误差差距超过 2 倍。

#### 5.4.4 Geometry Forcing：把几何"蒸馏"进扩散特征

`Geometry Forcing`（2507.07982）的做法是把扩散中间特征对齐到 **VGGT** 的几何特征，用两项损失：

$$\mathcal{L}_{\text{Angular}} = -\frac{1}{LNP}\sum_{\ell,n,p}\cos\big(y_{\ell,n,p},\, f_\phi(h_{n,p})\big)$$

$$\hat{h}=\frac{f_\phi(h)}{\lVert f_\phi(h)\rVert_2},\qquad \mathcal{L}_{\text{Scale}}=\frac{1}{LNP}\sum\lVert g_\varphi(\hat h)-y\rVert_2^2$$

$$\mathcal{L}=\mathcal{L}_{\text{FM}}+0.5\,\mathcal{L}_{\text{Angular}}+0.05\,\mathcal{L}_{\text{Scale}}$$

对齐在 7 层 U-ViT 的**第 3 层**。动机是个很有说服力的诊断：拿冻结的 DFoT 特征接 DPT 深度头做线性探测，发现原始视频特征"产生不出有意义的几何表征"。

**关键在于把方向与幅值解耦**：单用 cosine 对尺度盲，单用 MSE 会坍塌（FVD 1648）；先用 cosine 管方向、再从**归一化后的特征**回归幅值才有效。256 帧 FVD 消融：基线 364 → 只加 angular 253 → 两项都加 **243**。

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
    2. 典型指标包括：ADE/FDE、Chamfer Distance、AbsRel、Revisit Error、Scene Revisit Consistency 等——但**后两个的地位远不如前三个稳固，见 7.3.4**。
3. Task Performance
    1. 评估 world model 是否真的支持下游决策；
    2. 典型指标包括：Raw Return、Driving Score、Success Rate、MTLC Success Rate、Top-K Accuracy 等。

下面这张图是论文里对 evaluation ecosystem 的总结，很适合后续当评测索引来用：

![](assets/视觉世界模型%20-%20Survey/evaluation.png)

### 7.3 关键指标的精确定义（含一处必须纠正的说法）

上面那份指标清单如果只当名词记，很容易在复现或比较时踩坑。这里把最常用的几个写清楚。

#### 7.3.1 FVD：三个让数字不可比的陷阱

$$\text{FVD}=\lVert\mu_R-\mu_G\rVert^2+\mathrm{Tr}\big(\Sigma_R+\Sigma_G-2(\Sigma_R\Sigma_G)^{1/2}\big)$$

特征提取器是 **Kinetics-400 上预训练的 I3D**，取的是 **logits 层（400 维）**，输入 224×224 归一到 $[-1,1]$。注意是 logits 而非 pre-logits pooling，这是官方实现里明确选择的。

三个陷阱：

1. **样本量偏差。** 原文明确说"跨模型比较 FVD 时，必须使用相同的样本量"。所以看到没写样本数的 FVD 就该警惕。
2. **没有正典的片段长度。** 原论文里**并没有**规定用多少帧，FVD$_{16}$ / FVD$_{128}$ 与 2048 样本这些都是 StyleGAN-V 时代形成的惯例。
3. **两份"官方"权重不一致。** StyleGAN-V 发现 tf.hub 上的 I3D 权重与官方仓库的不同，于是两套"官方"数字互相不可比。此外 FVD **对运动坍塌不敏感、偏向图像质量**——这正是它不足以评价世界模型的原因。

CD-FVD（2404.12391）的修法是换成 **VideoMAE-v2 ViT-g**（1408 维）以消除 Kinetics 的内容偏置，结论是自监督预训练才是关键。

#### 7.3.2 AbsRel 与 Chamfer Distance：对齐协议决定数字

$$\text{AbsRel}=\frac{1}{|T|}\sum|y-y^*|/y^*$$

**真正的复现陷阱在对齐协议**：median scaling（SfMLearner 式，只对尺度、在深度空间）与 MiDaS 式的**在视差空间做最小二乘 scale+shift** 是两套不同标准，后者严格更宽松。所以不同时说明"对齐方式、所在空间、深度截断、裁剪范围"的 AbsRel 之间不可比。

Chamfer Distance 至少有四种互不兼容的约定（求和/求均值、平方/不平方、有无 ½ 因子、是否归一化）。原始 PSGN 版本是**平方距离之和**：

$$d_{CD}=\sum_{x\in S_1}\min_{y\in S_2}\lVert x-y\rVert_2^2+\sum_{y\in S_2}\min_{x\in S_1}\lVert x-y\rVert_2^2$$

而现在常用的 CD-$L_1$（Occupancy Networks 式）是 $\tfrac12(\text{Acc}+\text{Comp})$、不平方。**所以孤立的一个"CD = 0.05"是无法解释的**，必须连同约定一起写。

#### 7.3.3 ADE/FDE 与 IntPhys / Physion

$$\text{ADE}=\frac1T\sum_t\lVert\hat s_t-s_t\rVert_2,\qquad \text{FDE}=\lVert\hat s_T-s_T\rVert_2,\qquad \text{minADE}_K=\min_k \frac1T\sum_t\lVert\hat s^{(k)}_t-s_t\rVert_2$$

一个 benchmark 层面的坑：**Argoverse v1 的 minADE 实际是"min-FDE 那个模态的 ADE"**，不是对所有 ADE 取最小；nuScenes 按**概率**而非 oracle 排序。跨 benchmark 比这些数字要格外小心。

**IntPhys** 用的是 violation-of-expectation 范式，给出可能/不可能配对后计算两个分数：

$$L_R=\frac1N\sum_i \mathbb{1}\Big[\sum_j P(Pos_i^j)<\sum_j P(Imp_i^j)\Big],\qquad L_A=1-\text{AUC}\big(\{P(Pos)\},\{P(Imp)\}\big)$$

$L_R$（相对）比较的是**整个 $k$-uplet 上的求和**，不是逐对比较；$L_A$（绝对）是 AUC。三个 block 分别是 O1 物体永存 / O2 形状恒常 / O3 时空连续——**solidity 只在 IntPhys 2 里**。因为不知道违规发生在哪一帧，视频级分数取**逐帧最小值**。IntPhys 2（2506.09849）报的是普通准确率：人类约 96.4%，而所有模型接近随机（V-JEPA 2 最好，57.5%）。

**Physion** 是二分类 OCP（红色 agent 会不会在所有物体静止前接触黄色 patient），除准确率外还报**与人类平均反应的 Pearson $r$** 和**与单个人类的 Cohen $\kappa$**。人类准确率 0.71，模型约 0.53–0.69。Physion++ 进一步要求推断质量/摩擦/弹性等潜在属性，人类掉到 60%，而最好的模型-人类相关只有 $r=0.12$（人类 split-half 是 0.37），且**多数模型根本没有用上属性推断阶段**。

#### 7.3.4 ⚠️ "Scene Revisit Consistency" 并不是一个已定义的指标

这一条是我核查后必须写进来的纠正，也包括对本笔记上一版的纠正：

- **Revisit Error** 确实存在，但它**只是一个协议 + 借用的统计量，没有公式**：绕 Y 轴环绕 256 帧，用 **rFID** 比较首帧与末帧。出自 `Geometry Forcing` 一篇，一个实现。
- **Scene Revisit Consistency**：综述把它引到 `SlowFast-VGen`（2410.23277），但**那篇论文没有定义这个名字的指标**——它报的是 FVD 与 "scene cuts"（0.37 vs 0.89）。这个名称看起来是综述自己起的。**所以任何把 SRC 当成有公式的既成指标来引用的表格，都属于过度陈述。**
- `LoopNav`（2505.22976）的摘要宣称有 Scene Graph Consistency Score，但两个 HTML 版本的正文里都找不到定义，只报 FVD/LPIPS/SSIM。**存疑，不要引用。**

现实是：revisit 一致性目前**没有公认指标**，各家都是"协议 + 现成统计量"。`WorldMem` 用 PSNR/LPIPS/rFID 对比自己早先的帧；`VMem` 用环形轨迹在回程上评分 + DUSt3R 位姿误差；WorldScore 的"3D consistency"甚至只是**相邻帧**重投影误差，根本没测闭环。`VMem` 自己也承认 RealEstate10K"的轨迹很少重访已观测区域"。

**这正好印证了综述 Re-evaluation 那一节的判断：长时程一致性还缺可靠度量。** 换句话说，这是一个**空位**，而不是一个已解决的评测维度。

### 7.4 数据集与 benchmark 的系统整理

原文把数据集分成 Foundational（general prediction / physics & causality）与 Domain-specific（robotics / driving / gaming）两大层，具体条目已经在上面的 [[#6. 应用域与数据集版图]] 展开成表，这里不重复。

### 7.5 这篇综述给出的关键观察

作者的关键判断基本可以概括为：

- 只看画面质量会高估模型的世界理解能力；
- 真正可靠的 VWM 必须同时满足视觉合理、物理合理、任务可用；
- 长时程一致性、几何稳定性、因果干预响应，才是下一阶段 evaluation 的重点。

### 7.6 限制与警惕点

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
