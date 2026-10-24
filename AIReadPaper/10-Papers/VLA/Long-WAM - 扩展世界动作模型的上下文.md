# Long-WAM：扩展世界动作模型的视觉上下文

## 基本信息

- **论文**：Long-WAM: Scaling the Context of World-Action Models（arXiv:2610.10528，v1，2026-10-07 提交，cs.RO）
- **机构 / 作者**：NVIDIA（Wei Huang、Shuai Yang、Weian Mao、Luozhou Wang、Weifeng Lin、Bryan Chu、Sifei Liu、Linxi "Jim" Fan、Song Han、Yukang Chen）、MIT（Bohan Zhang）、HKU（Chenzhi Liu、Yicheng Xiao、Xiaojuan Qi）、UCSD（Isabella Liu）；Wei Huang 与 Bohan Zhang 共同一作
- **归属**：世界动作模型（World-Action Model, WAM）/ VLA / 具身基础模型
- **链接**：[arXiv](https://arxiv.org/abs/2610.10528) ｜ [项目页](https://nvlabs.github.io/)（论文首页给出 Code 与 Project Page 链接）｜ © 2026 NVIDIA
- **一句话总结**：机器人控制需要历史，但处理历史会拖慢响应——Long-WAM 证明「能读到历史」≠「会用历史」：只有视频底座是**自回归预训练**的，加长上下文才真的兑换成成功率，再用流式编码 + NVFP4 + 设备级 kernel 调优把 19.2 秒历史压进 107 ms 的实时预算。

---

## 1. 研究背景

### 1.1 问题：上下文既是资源也是负担

快速机器人控制需要理解场景「正在怎么变」。单张图能给出物体位置，却给不出运动状态与交互进度。世界动作模型（WAM）把视频预测带进闭环控制，视觉历史于是成为天然资源——但**处理更多历史会延迟它本该改善的响应**。论文的问题因此很具体：WAM 控制如何随视觉上下文伸缩？这些收益在实时约束下如何保住？

### 1.2 定位：与 DreamZero / LingBot-VA 的分歧点

现有 WAM 靠因果 cache、持久或检索式记忆来保留历史。但作者指出「接触到历史」与「学会从历史预测」是两件事：DreamZero、LingBot-VA 是把**双向预训练**的视频生成器适配成因果视频-动作预测；LingBot-VA 2.0 则联合预训练因果视频与潜动作。Long-WAM 走的是互补路线——**先学自回归（AR）视频预测，再在保持 history-to-future 结构的前提下适配动作**。AR 预训练已经学会了因果时序分解，等于给模型一个「使用历史」的基座，而不只是「访问历史」的接口。

### 1.3 核心结论预览

| 设置 | 0 s 历史 | 最优历史 | 增益 |
| --- | --- | --- | --- |
| RoboCasa GR-1（AR 机器人域） | 63.3% | 78.7% @ 19.2 s | +15.4 pts |
| LIBERO-Long（AR 机器人域） | 94.5% | 99.5% @ 2.4 s | +5.0 pts |

而双向预训练（Wan2.2）初始化在同样的因果适配下**净收益为零**（61.7% @ 2.4 s → 64.1% @ 9.6 s → 61.6% @ 19.2 s）。

---

## 2. 核心方法

![](assets/Long-WAM%20-%20扩展世界动作模型的上下文/figures/fig1_overview.png)

**图 1（Page 1）｜Long-WAM 总览**：(1) AR 视频预训练——LongLive2.0-Robot 从约 10,000 window-equivalent 小时的机器人 + 第一人称视频学习预测动力学；(2) Causal-to-causal 世界动作适配——保留因果视频依赖，同时让 action-chunk 去噪条件于观测历史；(3) 上下文伸缩——GR-1 63.3%→78.7%（峰值 19.2 s），LIBERO-Long 94.5%→99.5%（峰值 2.4 s）；(4) 高效部署——异步执行、流式 VAE、决策期 prefill、通话内 KV 复用与硬件加速，覆盖 RTX 5090 / DGX Spark / Jetson Thor。

### 2.1 LongLive2.0-Robot：机器人域 AR 视频预训练

**数据与初始化**：从 LongLive-2.0 的 16 秒 AR checkpoint 继续训练，语料约 10,000 window-equivalent 小时，来自 RoVid-X（82.7%）、EgoDex（9.0%）、AgiBot World（4.8%）、EgoVerse（2.6%）、VITRA（0.8%），共 2,294,889 个「文本+图像→视频」样本（2.25M clips / 5 个数据集）。**监督只有视频、没有动作标签**，因此可以跨本体混合学习，不需要共享动作空间——这是这一阶段的关键性质。注意"10,000 小时"是窗口等价估计（每样本按 16 秒折算），不是去重原始素材时长。

**Teacher-forcing 训练**：序列最长 30 秒，沿用 LongLive-2.0 的序列并行 AR 训练（长序列跨 GPU 分片），block-causal attention 让每个带噪 chunk 并行地从真值前缀获得监督，条件图像保持干净且不进 loss；沿用 SVI 的 error recycling。对干净目标 chunk $z_i$：

$$x_i^{\sigma_i}=(1-\sigma_i)\bar{z}_i+\sigma_i\bar{\epsilon}_i,\quad \sigma_i\in[0,1]$$

$$\mathcal{L}_{\text{TF-AR}}=\mathbb{E}\left[w(\sigma_i)\left\|v_\theta(x_i^{\sigma_i},\sigma_i\,|\,h_{<i},c)-(\bar{\epsilon}_i-z_i)\right\|_2^2\right]$$

恢复目标保留原始 $z_i$（而非被扰动版本），即教模型纠偏 rollout 式误差。给定一张图 + 语言 prompt，模型能预测连贯的机器人运动与物体交互，构成动作适配的预测先验。

### 2.2 Causal-to-Causal 世界动作适配

**因果耦合**：沿用 DreamZero / LingBot-VA 的**非对称接口**——video query 只读自己及更早的视觉块，从不看 action token；action query 读观测历史、部分去噪的未来，以及整个带噪 action chunk。这样既保住预训练的因果视觉依赖，又把动作同时锚定在过去与预期交互上。关键在于：history-to-future 的因果序是**在视频预训练阶段就学到的**，不是动作适配时才引入的。

在控制步 $t$ 决策，记观测视频潜变量 $Z_t^-$、机器人状态 $q_t$、语言指令 $c$、$H$ 步动作块 $A_t$：

$$\tilde{Z}_t^+=\text{Rollout}_{\theta,\sigma^\star}(\epsilon_v\,|\,Z_t^-,c,q_t),\quad \mathcal{K}_t=\text{Prefill}_\theta(Z_t^-,\tilde{Z}_t^+;c,q_t,\sigma^\star),\quad \hat{A}_t=\text{Denoise}_\psi(\epsilon_a\,|\,\mathcal{K}_t,c,q_t)$$

这就是论文所谓的 **IDM（inverse dynamics modeling，predict-then-act）**：video expert 先 rollout $K_v$ 步未来潜变量到噪声水平 $\sigma^\star\in(0,1)$，prefill 出逐层 KV cache $\mathcal{K}_t$，action expert 再条件去噪；观测潜变量全程保持干净。

![](assets/Long-WAM%20-%20扩展世界动作模型的上下文/figures/fig2_attention.png)

**图 2（Page 3）｜注意力模式与视频-动作推理策略**：(a) 只用当前观测生成动作；(b) 历史条件但无未来预测；(c) 视频-动作联合去噪 CoD（未来视频在各步间变化）；(d) 先视频预测后动作去噪 IDM——动作条件于观测历史 + 预测未来潜变量，并在动作去噪各步间复用视觉 KV。图例区分「看观测 / 看视频 / 看动作 / 被 mask」。

**训练与推理**：两个 pass——视频 flow matching（条件于干净历史），然后动作 flow matching（条件于历史 + 前向加噪的真值未来，$\sigma^\star=0.9$）。第二遍 **detach 视觉 cache**，所以动作 loss 只更新 action expert 与本体感受适配器。两个分支都预测 noise-minus-data velocity，总目标 $\mathcal{L}=\lambda_v\mathcal{L}_{\text{video}}+\lambda_a\mathcal{L}_{\text{action}}$。推理时视频只跑 4 步（V4）并在 $\sigma^\star=0.9$ 就停，**未来潜变量从不解码成像素**，其 cache 在整个动作去噪过程中复用。

### 2.3 上下文伸缩的实验设计

只缩放因果前缀中**真实观测**的时长，每个 benchmark 内固定视觉预测范围与动作 horizon——这样检验的是「更多过去证据是否改善同一个近期决策」，而不是改变预测/动作的长度。每个窗口是**单独训练的模型**，并在其训练上下文长度上评测；零历史保留当前观测；早期缺失的历史用首帧重复填充。对比三种视频初始化：Wan2.2（双向）、LongLive-2.0（通用 AR）、LongLive2.0-Robot（机器人域 AR），在 GR-1 上扫 0–38.4 秒，LIBERO-Long 做补充研究。

---

## 3. 实时部署基础设施

### 3.1 异步执行与流式 VAE

**纯异步就够**：推理与机器人执行重叠，不需要 blending 或 prefix guidance。决策 $k$ 预测动作块 $\hat{A}_{t_k}$，只有前 $R$ 步可执行，名义 stride $S$（$R/2\le S<R\le H$），重叠 $O=R-S$ 步；交接时丢弃已执行前缀、执行新块对齐的后缀，迟到则等待。交接deadline：

$$T_{\text{ready}}\le O\Delta t=(R-S)\Delta t$$

其中 $\Delta t$ 是控制周期，$T_{\text{ready}}$ 包含从触发观测可用起的传输、排队与计算。重叠 $O$ 越小，动作连续性越好（12→8 步让重叠 RMSE 降约 4 倍、jerk 降近 3 倍），但要求更晚的推理触发和更紧的 deadline。**流式因果 VAE** 在观测到达时就逐组编码，触发后只编码剩余帧再拼接投影，从而避免整窗编码的反复等待与累积空闲，让更晚触发 + 更短重叠成为可能。

为什么 Long-WAM 敢用纯异步？因为条件于 LongLive2.0-Robot 时序连续的预测，相邻块在重叠区自然一致：RoboTwin 2.0 上异步几乎不掉点（94.4%→94.2%），而 Fast-WAM 掉 15.4 分、LingBot-VA 掉 45.5 分。

### 3.2 边缘加速

共享优化：video expert 线性层用 **W4A4 NVFP4** 做生成与 KV prefill，动作计算与 KV 存储保持 BF16（动作量化省不了多少延迟还可能损控制精度）；NVFP4 + CUDA Graph replay + torch.compile 抵消小 kernel 的 launch 开销；**去噪不变量复用**（text/state KV、观测视频 KV、FP32 RoPE 表）在单次推理调用内复用；**共享输入量化**把 Q/K/V 投影的激活量化一次供三个 GEMM 复用；**video-action online softmax** 用共享归一化合并两段注意力而不拼接 KV buffer：

$$m=\max(m_v,m_a),\qquad \text{Attn}=\frac{e^{m_v-m}u_v+e^{m_a-m}u_a}{e^{m_v-m}\ell_v+e^{m_a-m}\ell_a}$$

设备特调：Quant/GEMM 调 tile、warp、流水级（RTX 5090 保留 base 实现 + shape dispatch；Spark 用紧凑 buffer 降低量化时间，提高每 SM 常驻 block 使 grid 一波跑完；Thor 靠更大的 shared memory 预算换 GEMM 配置，小 epilogue tile 避免寄存器 spill），后端选择含 VAE layout / 卷积调优与注意力后端。

---

## 4. 实验与消融

### 4.1 仿真主结果

| Benchmark | 关键数字 |
| --- | --- |
| LIBERO（avg / Long） | **99.5% / 99.5%**（IDM）；Long 上超 LingBot-VA 1.0 分、超 Fast-WAM 4.3 分 |
| RoboTwin 2.0（Clean / Rand / Avg） | 94.7 / 94.2 / **94.4%**，超 ABot-M0.5 0.3、超 LingBot-VA 2.0 0.8 |
| DOMINO（动态数据微调后） | **34.9% SR / 45.1 MS**，SR 超 Fast-WAM 15.0 分，MS 超 PUMA 10.1 分 |
| RoboCasa GR-1（19.2 s） | **78.7%**，超最强基线（Cosmos Policy 67.1）11.6 分 |

IDM 相对「无未来视频去噪（w/o V）」和「联合去噪（CoD）」的增益在 LIBERO-Long 最大：99.5% vs 94.5% / 97.8%——先估计交互如何演化，为跨子步协调动作提供了条件。且增益来自**部分去噪的潜变量**，不做像素级合成：预测是控制的中间表示。

### 4.2 上下文伸缩：这是全文最硬的证据

![](assets/Long-WAM%20-%20扩展世界动作模型的上下文/figures/fig6_context.png)

**图 6（Page 8）｜三种视频初始化的上下文伸缩曲线**：(a) LIBERO-Long 在 2.4 s 就饱和到 99.5%（LongLive2.0-Robot）；(b) RoboCasa GR-1 一路涨到 19.2 s 的 78.7%，再往上到 38.4 s 反而回落到 75.2%。三条曲线中两条 AR 变体（机器人域 / 通用域）持续上升，双向初始化（Wan2.2）先微升后回落、净收益为零。

要点三层：

1. **AR vs 双向**：在 GR-1 上，机器人域 AR 变体对双向初始化的领先从「无历史时 3.3 分」扩大到「19.2 秒时 17.1 分」。所有变体都用同样的因果 WAM 适配，所以差别不在掩码，而在预训练是否学到了被适配阶段保留下来的 history-to-future 依赖。
2. **任务依赖的记忆需求**：LIBERO-Long 2.4 秒就吃满收益，GR-1 需要 19.2 秒——短历史足以捕捉前者，后者是多次物体传递的链式任务。
3. **38.4 秒的回落**：该窗口是平均训练轨迹（12.1 秒）的三倍，其采样历史帧有 80.4% 是 padding。作者判断这是**历史覆盖不足**而非内在记忆上限——更长更密的机器人录像是延伸有效窗口的直接路径。

### 4.3 真机结果

![](assets/Long-WAM%20-%20扩展世界动作模型的上下文/figures/fig7_dynamic.png)

**图 7（Page 10）｜Unitree G1 动态操作**：上传送带速度从 3.0 加到 7.5 cm/s，Fast-WAM 抓取成功率从 75% 掉到 0%、$\pi_{0.5}$ 从 15% 掉到 0%，Long-WAM 稳定在 90–100%（100 / 100 / 95 / 90）；3 cm/s 下把移动的绿杯抓入蓝杯的动态叠杯，Long-WAM 20 次成功 19 次，两个基线一次都没成功（人类遥操作 50/54）。下图是 Long-WAM 与 Fast-WAM 的 rollout 对比与速度消融柱状图。

YAM 双臂上的长程任务（平均 40 秒以上）：按颜色分积木 80%、放饺子入锅 80%、叠碗 85%（平均 81.7%），把评测从「快速拦截」延伸到「持续执行」。

### 4.4 复合规划：作为记忆知情执行器

RoboCasa365 上用 Human300 训练的 checkpoint（2.4 秒上下文）+ GPT-6 Astra 高层规划器，**策略权重与视觉上下文完全不变**，只加规划器：Composite-Seen 15.8%→38.8%，Composite-Unseen 6.1%→35.0%，Overall 31.4%→54.4%。对比之下 GPT-6 Astra 单独 25.2%、$\pi_{0.5}$+Astra 30.5%。规划器带来的增益对 Long-WAM 是 +23.0 分，对 $\pi_{0.5}$ 只有 +13.6 分——**执行越强，规划越值钱**。

### 4.5 延迟与效率

| 配置 | RTX 5090 | DGX Spark | Jetson Thor |
| --- | --- | --- | --- |
| BF16 eager | 356.0 ms (1.0×) | 1342.8 ms (1.0×) | 1215.2 ms (1.0×) |
| 优化后 V4/A4 | **107.4 ms (3.3×)** | 328.2 ms (4.1×) | 378.7 ms (3.2×) |
| 优化后 V2/A2 | 81.8 ms | — | — |

优化拆分（5090）：NVFP4+CUDA Graph+compile → 172.4 ms；+去噪不变量复用 → 144.9；+共享输入量化 → 130.3；+online softmax → 126.9；+Quant/GEMM 调优 → 119.1；+后端选择 → 107.4。相比基线方法：LingBot-VA 3618.4 ms、Motus 1201.1 ms、Cosmos Policy 470.6 ms、Fast-WAM 244.1 ms——Long-WAM 比 Fast-WAM 还快 2.3×，而后者并不预测未来视频。

代价也要看到：**上下文 8 倍（2.4→19.2 s）让延迟从 107.4 涨到 341.0 ms（约 3.2×）**；无历史时仅 74.6 ms。CoD 更快（90.9 / 65.7 ms）但成功率略低。异步重叠 $S$ 从 12 增至 16（$O$ 12→8）时成功率反而升到 95.0%，再增到 20 回落至 94.3%——平滑性与决策频率之间存在权衡，最平滑不等于最成功。

---

## 5. 局限与评价

**论文没有独立的 Limitations 章节，只在 Appendix A（Discussion and Conclusion）里交底**，主要四条：

1. **每个上下文窗口都是单独训练的模型**——这换来了「每个长度上历史有多大用」的干净视图，但下一步自然是单个策略在测试时自适应窗口。
2. **38.4 秒的回落与数据稀疏耦合**（80.4% padding），不是内在记忆上限；需要更长更密的机器人录像。
3. **延迟随上下文增长**（74.6 ms → 341.0 ms），自适应分配上下文才有工程意义。
4. **不在底层控制器里追求无限历史**：小时级目标、推理与任务分解应由高层规划器负责，窗口长度应取决于任务与可用算力，而非通用常数。

**我的补充评价**：

1. **核心发现的价值高于方法本身**。「access ≠ use，AR 预训练才让长上下文兑现」这句话对整条 WAM/VLA 路线都有指导性——它意味着「加大 cache」「加记忆模块」这类系统侧努力，如果底座是双向预训练的，收益可能接近零。这个对照实验（同掩码、不同初始化、扫 0–38.4 s）做得干净，是全文最值得记住的部分。
2. **IDM 的选择有代价也有说服力**。视频先 rollout 再条件动作，是串行推理，天然更慢；论文不回避这一点，用 CoD 对照（更快但成功率更低）+ 整章工程优化把 107 ms 做出来，属于「承认代价再解决代价」的诚实写法。而且未来潜变量停在 $\sigma^\star=0.9$、从不解码成像素，把「想象」明确定位为中间表示而非生成任务——这个判断很关键。
3. **工程部分（第 4 章）信息密度高但可迁移性有限**。NVFP4 + CUDA Graph + 共享输入量化 + online softmax 合并双 KV，这套组合对任何 video-DiT + action-expert 架构都有参考价值；但 tile/warp/后端的选择是 Blackwell/Thor 专属，换平台基本要重调。
4. **评测口径需注意**：每个 context 长度单独训练模型，意味着「上下文伸缩曲线」背后是不同的训练成本，不能理解成同一模型的可调参数；baseline 延迟是在各自原生部署配置与去噪预算下测的，不是严格同条件对比（论文也说明了 timing runs 与任务评测分离）。真机只有 20 trials/条件，统计噪声不小。
5. **与我的关注点的关系**：我关心 3D/4D 表示与世界模型的接口，这篇从另一侧给了印证——**显式预测未来（哪怕是潜变量层的部分去噪结果）作为中间表示，比隐式记忆更适合闭环控制**；而它对「历史要多少」给出了任务依赖的实证答案，这对设计具身系统的记忆预算是直接可用的结论。

---

## 6. 复现成本

- **预训练**：LongLive2.0-Robot 约 30,720 GPU-hours（64× H100），从 LongLive-2.0 的 16 秒 AR checkpoint 续训；语料 2.29M 样本 / ~10,000 window-equivalent 小时。
- **下游 WAM 训练**：16× GB200，AdamW，峰值 lr 1e-4，cosine 衰减，5% warmup，BF16，梯度裁剪 1.0。
- **推理**：RTX 5090 / DGX Spark / Jetson AGX Thor 三档实测，107.4 / 328.2 / 378.7 ms per action chunk（含完整观测 VAE 计算）；V2/A2 可降到 81.8 ms（9.3→12.2 Hz）。
- **真机平台**：Unitree G1 人形 + YAM 双臂。
- **门槛判断**：极高。光 AR 视频预训练就是 3 万 GPU 小时量级；能复现的部分只有下游适配与部署优化，而这两块又强依赖 LongLive-2.0 底座与 NVIDIA 硬件。论文给了 Code / Project Page 链接，值得看是否有开源权重。

---

## 附件溯源

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `figures/fig1_overview.png` | 官方 PDF Page 1 裁剪（Figure 1） | 总览：AR 预训练 → causal-to-causal 适配 → 上下文伸缩 → 高效部署 |
| `figures/fig2_attention.png` | 官方 PDF Page 3 裁剪（Figure 2） | 四种注意力模式与 IDM / CoD 推理策略对比 |
| `figures/fig6_context.png` | 官方 PDF Page 8 裁剪（Figure 6） | 三种视频初始化的上下文伸缩曲线 |
| `figures/fig7_dynamic.png` | 官方 PDF Page 10 裁剪（Figure 7） | Unitree G1 传送带动态抓取与动态叠杯真机结果 |
| `sources/Long-WAM-arXiv2610.10528v1.pdf` | [arXiv:2610.10528v1](https://arxiv.org/pdf/2610.10528v1)，2026-10-24 获取 | 原始论文（28 页，含附录 A–H） |
