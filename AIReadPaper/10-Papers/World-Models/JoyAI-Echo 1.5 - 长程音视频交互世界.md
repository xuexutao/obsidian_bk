---
type: paper-note
status: complete
domain: World-Models
paper: Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds
year: 2026
arxiv: 2608.23383
doi: 10.48550/arXiv.2608.23383
source: https://arxiv.org/abs/2608.23383
project: https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/
code: https://github.com/jd-opensource/JoyAI-Echo
tags:
  - World-Models
  - Audio-Visual-Generation
  - Long-Video
  - Self-Gradient-Forcing
created: 2026-10-16
updated: 2026-10-16
---

# JoyAI-Echo 1.5：长程音视频交互世界

## 基本信息

- **作者/机构**：Nan Duan、Haoyang Huang、Weiyang Jin 等；Joy Future Academy, JD
- **版本**：arXiv v2，2026-08-25；46 页
- **任务**：同一套音视频生成底座分化为长视频与交互世界模型两个版本，同时解决跨 shot 身份/声纹保持、连续 6DoF 控制和长时自回归稳定性。
- **核心判断**：这不是靠单一“更长上下文”获得长程能力，而是把问题拆成三层：显式跨 shot 记忆负责身份，度量 6DoF 条件负责动作，Self-Gradient Forcing（SGF）负责让模型适应自己生成的历史。三者互补，不能互相替代。

## 一句话结论

JoyAI-Echo 1.5 的关键贡献是把长程音视频生成中的三种分布偏移分别处理：历史身份用可组合音视频 memory 固化，异构控制用统一度量相机轨迹对齐，推理时累积误差则用在自生成 rollout 上恢复上下文梯度的 SGF 训练；结果在长视频一致性与 WBench/SANA-WM-Bench 上都领先，但困难轨迹的旋转漂移和缺少显式持久 3D 世界状态仍是根本瓶颈。

---

## 1. 研究背景与问题定义

### 1.1 从独立片段到“持续存在”的故事与世界

传统视频扩散模型优化的是固定窗口内的单个片段。长故事要求角色外观、服装线索、声纹和对白在多次切镜后仍可识别；交互世界则要求模型把用户持续输入变成可预测的视点变化，并在反复生成自己的输出时不逐步崩坏。论文指出，单纯扩大模型容量无法直接解决三类失配：训练看到干净历史而推理看到模型历史、不同数据源的动作词表和尺度不一致、相邻帧条件不足以保存跨场景身份。

### 1.2 两个版本、一个技术主轴

长视频版本把故事分解为 shot 序列，以跨 shot 音视频 memory 维持角色与说话人；世界模型版本把键盘或连续位姿统一为度量相对 6DoF 相机轨迹。二者最后都被转换为因果少步生成器，并通过短程与长程 SGF 学习如何在自生成历史上继续写入可用的因果记忆。

### 1.3 贡献拆解

1. **可组合跨 shot memory**：多段历史各抽一帧视觉证据，同时保留经语音分离后的整段 speech audio，以独立 RoPE 区间隔开目标、历史和初始 memory。
2. **统一 6DoF 控制**：键盘增量和连续位姿都转成相对 $SE(3)$ 轨迹，用全数据集固定尺度校准平移，再由 UCPE 几何注意力注入视频分支。
3. **因果少步长程训练**：teacher forcing 完成双向到因果的初始化；SGF 在无梯度 rollout 后用可微并行重放恢复“历史如何写入 KV”的梯度，并结合 DMD 蒸馏为四步/少步生成。

---

## 2. 任务定义与输入输出

### 2.1 长视频概率分解

给定 shot 文本 $C=\{c_t\}_{t=1}^{T}$、可选初始 memory $M_0$ 和逐 shot 图像条件 $I_t$，输出同步音视频 shot $S_t=(V_t,A_t)$：

$$p_\theta(S\mid C,M_0,I)=\prod_{t=1}^{T}p_\theta(S_t\mid c_t,M_{t-1},I_t),\qquad M_t=\mathcal{U}(M_{t-1},S_t).$$

这里 $I_t$ 是当前 shot 的首帧，不占 memory slot；$M_{t-1}$ 聚合更早 shot 的外观与声纹证据。这个区分很重要：首帧解决“从哪里开始”，memory 解决“这个人是谁”。

### 2.2 世界模型输入输出

世界模型接收可选媒体上下文 $\mathcal M$、结构化文本 $\mathcal C$ 和控制序列 $U_{1:T}$，先将控制序列序列化为校准后的相对 6DoF 事件流 $\mathcal E_P$，联合输出视频与音频：

$$p_{\theta,\phi}(V_{1:T},A_{1:T}\mid\mathcal M,\mathcal C,\mathcal E_P).$$

$\theta$ 是音视频主干，$\phi$ 是并行轨迹条件路径。轨迹只直接进入视频分支，音频依靠 A2V/V2A 跨模态耦合随画面变化。

---

## 3. 核心方法

### 3.1 总体框架：memory、几何控制与少步蒸馏

![](assets/JoyAI-Echo%201.5%20-%20长程音视频交互世界/figures/01-method-overview.png)

> **图 1（论文 Figure 2）**：上半部分展示多源音视频 memory、分离 RoPE 坐标与四种统一条件模式；下半部分展示 8-step DMD 中视频全参、音频 LoRA 的非对称优化，以及音频判别器与 RMS 能量约束。图中能看出，跨 shot 一致性与少步加速是同一生成器上的两层设计，而不是两个串联系统。来源：官方 arXiv LaTeX 图源，论文第 6 页附近。

### 3.2 跨 shot memory：外观抽样，声音保留完整 speech 区间

每个历史条目 $m_i=(m_i^v,m_i^a)$ 从最多 $K$ 个按时间排序的历史 shot 构建。视觉只随机抽一帧并经 video VAE 编码：

$$m_i^v=E_v(V_i[f_i]),\qquad f_i\sim\mathcal U\{1,\ldots,|V_i|\}.$$

音频不再像 1.0 那样选择能量最高的短窗，而是先做 speech filtering，再编码完整 shot 的语音区间：

$$\widetilde A_i=\mathcal F_{\mathrm{speech}}(A_i),\qquad m_i^a=E_a(\widetilde A_i).$$

论文实现使用 MSST 工具链中的语音提取模型。设计动机很具体：能量峰值常被音乐或环境声占据，且错误背景声会在自回归 reuse 时不断被强化；完整 speech-bearing interval 更稳定地保留音色和说话方式。视觉与音频按同一 slot 顺序排列，但 1.5 删除了 1.0 的 slot-paired 跨模态硬 mask，让所有有效 token 通过原生 self-attention 与双向跨模态注意力交互。

### 3.3 用 RoPE 分区防止 memory 与目标时间线混淆

目标音视频 token 保持在低于 500 的原始对齐坐标；第 $i$ 个历史 memory 中心为

$$\mu_i=500+50(i-1).$$

视觉 memory 的所有空间 token 共用 $\mu_i$；音频 token 整体平移，使其时间中点对齐同一中心：

$$r_{i,\ell}^{a}=\mu_i+\tau_{i,\ell}^{a}-\bar\tau_i^a,\qquad \bar\tau_i^a=\frac{1}{L_i^a}\sum_{\ell=1}^{L_i^a}\tau_{i,\ell}^a.$$

预先给定的 $M_0$ 放到 $\rho_j\ge5000$ 的更高区间，并保证不同音频 span 不重叠。这样，模型仍可全局注意，但能从位置上区分当前目标、滚动历史与外部 memory；图像首帧仍处在目标坐标 0，因此可与 memory 同时使用。

> **校读更正**：上式中的视觉中心应为 $\mu_i$；论文公式与图均如此。memory token 始终保持 clean、噪声级为 0，并从预测 loss 中排除。

### 3.4 四种条件模式与联合 flow matching

用 $b_M,b_I\in\{0,1\}$ 控制是否启用 memory/首帧，统一覆盖 T2AV、I2AV、MT2AV、MTI2AV。对视频与音频 latent 使用 rectified-flow 插值

$$z_\sigma^q=(1-\sigma)z_0^q+\sigma\epsilon^q,\qquad u^q=\epsilon^q-z_0^q,$$

并仅对 noised target token 计算联合目标：

$$\mathcal L=\mathbb E\left[\|W_v\odot(v_\theta^v-u^v)\|_2^2+\lambda_a\|W_a\odot(v_\theta^a-u^a)\|_2^2\right].$$

这个接口避免为“有无首帧/有无记忆”训练四套模型，也使 memory 与图像条件真正可组合。

### 3.5 6DoF 控制：控制器无关，但不是任意动作模型

相机到世界位姿为 $T_t\in SE(3)$，相对运动写作

$$\Delta T_{ij}=T_i^{-1}T_j,\qquad \boldsymbol\xi_{ij}=\operatorname{Log}(\Delta T_{ij})^\vee\in\mathbb R^6.$$

离散键盘状态经固定映射 $\mathcal F_{\mathrm{ctrl}}$ 变成局部 twist，再通过指数映射积分；连续位姿则直接进入相对变换。因此不同控制器在模型内部都成为同一连续轨迹。其覆盖的是导航与视点相关动作，并不编码攻击、跳跃、操控物体等 actor intent。

不同数据源的平移尺度用全训练集每条 clip 最大位移的 90% 分位数统一：

$$s_{\mathrm{global}}=Q_{0.9}\left(\left\{\max_k\|\Delta\mathbf t_{i,k}\|_2\right\}_{i\in\mathcal D_{\mathrm{train}}}\right),\qquad \widehat{\mathbf t}_{i,k}=\Delta\mathbf t_{i,k}/s_{\mathrm{global}}.$$

固定全局尺度保留慢/快和短/长位移差异，也避免按 chunk 单独归一化在 turn 边界制造速度跳变。UCPE 将 ray-local 变换注入并行注意力分支，zero-init 输出残差加回视频 self-attention；第一/第三人称共用参数，初始 viewpoint 文本决定“相机相对角色在哪里”，轨迹决定“之后相机怎么动”。

### 3.6 音视频同步：物理时间对齐与五条因果路径

同步并非只靠末端音频后处理。模型把视频和音频 latent 分成覆盖同一物理时间的 macro-chunk；同一对音视频 chunk 共享噪声级。因果 mask 同时施加在 video self-attention、audio self-attention、A2V、V2A 和 UCPE 五条路径，因而未来视频不能泄漏给当前音频，未来音频也不能泄漏给当前视频。

人评提供了同步证据：与 HappyOyster 相比，JoyAI-Echo 1.5 在 audio-visual synchronization 上被偏好 48.4%，对方为 20.3%，31.4% 判为相似。该结果比单纯声纹相似度更直接，但仍是感知判断，论文没有给出毫秒级 onset alignment 指标。

### 3.7 Self-Gradient Forcing：训练“如何写历史”，而非穿过完整采样链反传

Teacher forcing 阶段让 noisy chunk 只能读取先前 clean chunk 与自身 noisy token：

$$\mathcal S(\mathcal B_{i,\mathrm{noisy}})=\{\mathcal B_{j,\mathrm{clean}}:j<i\}\cup\{\mathcal B_{i,\mathrm{noisy}}\}.$$

它解决双向主干到因果主干的初始化，却保留了 exposure bias。SGF 用两次 forward 处理：

1. **Forward 1**：以部署时四步 sampler 无梯度串行 rollout，记录每个 chunk 的生成 clean latent 与随机 exit-step noisy state。
2. **Forward 2**：把已生成 clean latent 作为 stop-gradient 输入，在一个可微、可并行的因果图中重建 K/V 投影与上下文注意力；后续 chunk 的 loss 因而能监督前面 chunk“如何被编码成可读历史”，但不穿过 Forward 1 的逐步采样决策。

SGF 再叠加 DMD。冻结 real-score，训练 fake-score 跟踪学生分布，对恢复预测 $\bar x_i$ 使用

$$\mathcal L_{\mathrm{SGF-DMD}}=\mathbb E\left[w(s)\left\langle\operatorname{sg}(s_{\mathrm{fake}}-s_{\mathrm{real}}),\bar x_i\right\rangle\right].$$

这比普通 self-forcing 多恢复了“生成历史写入未来可读 causal memory”的梯度，是论文最值得复现的部分。

![](assets/JoyAI-Echo%201.5%20-%20长程音视频交互世界/figures/03-causal-mask.png)

> **图 3（论文 Figure 4）**：左为 teacher forcing，当前 noisy query 读取之前的 clean K/V；右为 long-video SGF Forward 2，历史进一步受 sink + FIFO 稀疏窗口约束。完整坐标轴和四个象限均已从官方源图保留。来源：官方 arXiv LaTeX 图源，论文因果训练章节。

### 3.8 长程 SGF：bounded cache 不等于 bounded gradient horizon

长程阶段串联多个短程 segment，并复用 segment 末帧作为下一段首帧。每个模态只保留 $S_m$ 个 sink chunk 与 $R_m$ 个最近 FIFO chunk：

$$\mathcal H_i^m=(\widehat x_1^m,\ldots,\widehat x_{S_m}^m)\Vert(\widehat x_{\max(S_m+1,i-R_m)}^m,\ldots,\widehat x_{i-1}^m).$$

单层直接 fan-in 被限制为 $S_m+R_m$，但上层最近 chunk 已聚合更早层的历史，所以梯度可通过多层、多 chunk 和 A2V/V2A 路径多跳传播。论文强调，这仍不是 full rollout BPTT：Forward 1 的 latent、采样决策与串行 KV cache 全部 detached。

每个 rollout segment 都计算 DMD，而非只监督终点：

$$\mathcal L_{\mathrm{Long-SGF}}=\sum_{k=1}^{K}\mathcal L_{\mathrm{DMD}}^{(k)}.$$

这让后段 loss 在已积累误差的历史上训练。论文还用稀疏背景时间正则抑制四步蒸馏后稳定区域偶发的灰色 speckle，但只处罚显著超过历史/局部正常变化的 token，以免把真实运动抹平。

### 3.9 少步音视频 DMD 的非对称优化

视频分支全参训练，音频与跨模态模块用 LoRA：视频需要大容量重建空间结构与运动，音频则容易被强 DMD 更新破坏声学先验。训练先用高噪声/大学习率建立视频全局结构，再降噪声偏置和学习率恢复音色、瞬态等细节。任务权重也不同：T2AV/I2AV 的音频权重 0.25，已有声纹 memory 的 MT2AV 为 0.10。

完整学生目标为

$$\mathcal L_G=\mathcal L_{\mathrm{DMD}}^v+\lambda_a(\kappa)\mathcal L_{\mathrm{DMD}}^a+\lambda_{\mathrm{GAN}}\mathcal L_{\mathrm{GAN}}^a+\lambda_{\mathrm{energy}}\mathcal L_{\mathrm{energy}}^a.$$

其中对数 RMS matching 防止 latent 能量持续抬升导致 clipping，冻结 audio teacher 特征上的轻量判别头负责补回局部声学纹理。

---

## 4. 数据与训练设置

### 4.1 两阶段长视频数据

![](assets/JoyAI-Echo%201.5%20-%20长程音视频交互世界/figures/02-data-pipeline.png)

> **图 2（论文 Figure 3）**：Stage I 从影视/网络长视频中构建同身份、跨场景的 memory-target 组；Stage II 强化高分辨率、无字幕、中英双语和多 shot 转场样本。来源：官方 arXiv LaTeX 图源，论文数据章节。

Stage I 的基本单位是身份组，而不是独立 clip。memory 与 target 来自同角色但 scene-disjoint 的 shot，迫使模型学习稳定身份而非复制相邻内容。Stage II 使用 $720\times720$、$1280\times720$、$720\times1280$ 三种格式，增加无烧录字幕与中文对白样本，并为多 shot 视频显式标注 shot 边界、角色进入、动作、相机变化、音效和 BGM 连续性。

论文没有披露训练视频小时数、身份组数量、GPU 数、学习率、完整 batch size 或各 mixture 精确比例，这是复现的主要信息缺口。

### 4.2 世界模型几何数据

UE 提供一人称与四个角色相对第三人称相机的真值位姿；内部 gameplay 有动作日志和原生立体声但无位姿；人类 gameplay/网页视频只有音视频。非真值数据先在约一分钟长窗上用 metric-depth-assisted ViPE 恢复 6DoF，再以 SLERP/线性插值回到逐帧轨迹，并过滤抖动、突变、零位移和不稳定内参。

### 4.3 渐进课程

- **AV-CPT**：关闭轨迹条件，全参适配互动场景的图像、脚步、环境声与对白。
- **Action-SFT**：冻结主干，只训练 UCPE 分支；删除动态 motion narrative 和音频字段，避免文本泄漏相机运动，并偏向首帧条件。
- **Joint-FT**：恢复 speech/sound 字段但继续删除 motion narrative，以较小学习率联合优化主干和轨迹路径。
- **AR post-training**：teacher forcing → short SGF + DMD → long SGF。

---

## 5. 实验结果

### 5.1 长视频定量结果

在 100 个故事、3,000 个 shot 上，JoyAI-Echo 1.5 获得 ViCLIP 0.8264、Self-CIDS 0.7937、Voice 0.8524、Imaging 0.7467、CLIP 0.2868、speech Recall 0.9674，七项中六项第一；相对 JoyAI-Echo 1.0，前三项分别提升 0.0238、0.0144、0.0395。它的 Aesthetic 0.5705 仅第二，低于 LTX-2.5 的 0.6019，说明身份/声纹一致性优势不等于单帧美学全面领先。

与级联 ShotStream/StoryMem + MMAudio 相比，joint audio-video 方案的 speech recall 优势极大（级联系统约 0.03），支持“先视频后配音难以保留脚本对白”的论点。需要注意，默认结果启用了 Director Agent 进行 shot planning 和 memory selection，因此这里测的是完整系统，不是纯生成器隔离比较。

### 5.2 世界模型定量结果

WBench Navigation 158 例上，未蒸馏模型平均 81.7 排第一，四步因果版为 81.0；后者 Interaction 87.9 甚至略高于原版 87.2，但 Consistency 从 89.8 降为 77.5，说明少步因果化主要损失在状态保持而不是即时响应。

SANA-WM-Bench 961 帧（60 秒、16 FPS）中，四步因果版在 Simple/Hard split 的 VBench 为 80.13/81.06，旋转误差为 5.009°/14.221°，回访 PSNR 14.41/14.04。它在对比的因果模型中整体领先，但 Hard split 旋转误差显著上升，直接对应作者在结论中承认的 accumulated rotational drift。

### 5.3 消融与证据强度

论文**没有提供完整的逐模块定量消融表**来单独移除 speech filtering、RoPE 分区、全局尺度、SGF 或音频约束。现有证据主要是：

- 1.5 对 1.0 的系统级提升，混合了 memory、数据、标注、DMD 与 Agent 变化；
- 未蒸馏模型对四步因果版，能量化少步/因果化代价；
- 因果版与其他 causal baseline 的长程对比，支持 SGF 整体有效，但无法隔离 short SGF 与 long SGF；
- 人评的同步与身份偏好支持最终体验，但不是机制因果证明。

因此，论文的系统效果证据强，模块归因证据中等。复现时最应补做的消融是 TF only、short SGF、long SGF 三阶段递进，以及 memory 中短能量窗与完整 speech-filtered audio 的对照。

### 5.4 失败模式

论文与结果共同暴露四类边界：困难长轨迹上旋转误差累积；FIFO 淘汰后旧地点缺少可靠显式 3D 状态；6DoF 接口不代表任意语义动作；网络视频恢复的 pose 仍可能与模糊、场景内容相关。A/V 同步则有强人评证据，但没有细粒度事件时延测量。

---

## 6. 与 EchoWM 的关系

JoyAI-Echo 1.5 是系统总论文，长视频线与世界模型线共用底座、6DoF/UCPE、teacher forcing 和 SGF；EchoWM 则把世界模型分支的数据引擎、视角语义、评测与限制展开得更完整。两篇作者高度重叠，WBench/SANA-WM 数字也基本一致，应视为同一京东路线的总览与专项报告，而非独立平行方法。

---

## 7. 局限与批判性评价

### 7.1 已确立的事实

- 长视频自动指标和人评均显示跨 shot 外观、声纹、对白覆盖与同步改善。
- 统一度量轨迹在多 benchmark 上提供强控制结果。
- 四步因果版保留大部分整体能力，但一致性明显低于未蒸馏版。
- 困难 60 秒轨迹仍有明显旋转漂移。

### 7.2 我的判断

论文最扎实的思想不是“音视频世界模型”这个标签，而是把部署分布纳入训练：SGF 的 Forward 2 明确优化历史表示的可读性，避免 teacher forcing 只教会模型消费完美历史。另一方面，跨 shot memory 与 FIFO/sink causal history 是两种不同 memory：前者是身份级检索条件，后者是局部连续生成状态。论文把它们放在同一系统中很有价值，但真正持久的 world state 仍未建立。

最大的证据缺口是机制消融与训练成本披露。没有逐项 ablation，无法判断增益中多少来自更好的数据/Agent，多少来自结构本身；没有数据规模与算力，也无法评估 1.0→1.5 的可迁移成本。

---

## 8. 复现与实践建议

### 8.1 最小可验证复现

1. 先在短音视频 backbone 上实现物理时间对齐的 macro-chunk causal mask，确认四条 A/V attention 无未来泄漏。
2. 构造同身份、跨场景 memory-target 小数据；视觉抽帧，音频用 speech separator 保留完整 speech span。
3. 实现三个 RoPE 区间：target、rolling history、external memory，并验证 memory 与首帧可同时工作。
4. 对 6DoF 数据统一坐标约定，按训练集 90% 分位固定平移尺度；避免 per-clip normalization。
5. 依次训练 TF only、short SGF、long SGF，固定 sampler 和数据做递进消融。
6. 评估必须包含回访、长 rollout、hard rotation 与 A/V 事件同步，不能只看短片 VBench。

### 8.2 成本与可用性

官方已给出代码仓库，但论文未披露本工作的完整训练配置和数据规模。复现最昂贵的部分不是 UCPE 本身，而是大规模带原生音频的视频清洗、同身份跨 shot 分组、长窗 metric pose recovery，以及 SGF Forward 1/2 + real/fake score 的多模型训练。公开实现若不含数据引擎与训练 recipe，只能复现推理或局部机制。

---

## 9. 个人启发与后续问题

1. 能否把跨 shot identity memory 与 causal world memory 合并为带对象/地点索引的持久状态，而不是一个处理角色、一个处理最近帧？
2. SGF 的上下文梯度是否可用于 VLA，让后续动作损失监督先前观测如何写入 policy memory？
3. 全局 90% 分位尺度对跨域速度有效，但对真实米制控制仍是相对标定；需要怎样的绝对尺度锚点？
4. A/V 同步应补充 onset error、lip-sync 与动作-声效延迟，而不仅是主观偏好。

---

## 10. 材料来源

- 官方论文：<https://arxiv.org/abs/2608.23383>
- 项目页：<https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/>
- 代码：<https://github.com/jd-opensource/JoyAI-Echo>
- 本地来源与图号映射：[[assets/JoyAI-Echo 1.5 - 长程音视频交互世界/sources]]
