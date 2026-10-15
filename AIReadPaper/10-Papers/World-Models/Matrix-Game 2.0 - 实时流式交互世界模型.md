---
type: paper-note
status: complete
domain: World-Models
paper: "Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model"
year: 2025
arxiv: "2508.13009"
doi: null
source: https://arxiv.org/abs/2508.13009
project: https://matrix-game-v2.github.io/
code: https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2
tags:
  - World-Models
  - Interactive-Video
  - Self-Forcing
  - Real-Time-Generation
created: 2026-09-04
updated: 2026-09-04
---

# Matrix-Game 2.0：实时流式交互世界模型

## 基本信息

- **作者**：Xianglong He、Chunli Peng、Zexiang Liu、Boyang Wang 等，Skywork AI
- **版本**：arXiv v4，Technical Report
- **模型**：1.8B 参数，基于 SkyReels-V2-I2V-1.3B / Wan 2.1 架构
- **核心资源**：[论文](https://arxiv.org/abs/2508.13009) · [项目页](https://matrix-game-v2.github.io/) · [代码](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) · [权重](https://huggingface.co/Skywork/Matrix-Game-2.0)

## 一句话结论

Matrix-Game 2.0 把双向图生视频模型改造成带滚动 KV cache 的 3 步因果扩散模型，并用 ODE 初始化与 Self-Forcing/DMD 让学生在自己的历史上学习，从而在单张 H100 上达到约 **25 FPS**、生成分钟级键鼠可控视频；其真正突破是“实时 + 流式 + 开源推理”同时成立，但 OOD 退化、352×640 分辨率和缺乏显式长期记忆仍限制它成为可靠模拟器。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

交互世界模型的输出必须随用户动作立即更新。传统双向视频扩散在每次生成时处理整段序列，计算和显存随长度增长，又需几十次去噪；自回归模型虽然天然流式，却会把自己的错误反复喂回，几十帧后常出现模糊、饱和或场景坍塌。Matrix-Game 2.0 试图同时解决三个问题：缺少大规模精确动作数据、双向模型不能流式、因果模型长期误差累积。

![](assets/Matrix-Game%202.0%20-%20实时流式交互世界模型/figures/01-teaser.png)

> **Figure 1**：模型从单张图像出发，在 Minecraft、写实户外、室内、GTA 与 TempleRun 等场景中接受逐帧键鼠动作并持续生成。图中绿色按键代表当时控制信号。  
> 来源：论文 Figure 1，第 2 页，https://arxiv.org/pdf/2508.13009

### 1.2 核心贡献

1. 建立 Unreal Engine 与 GTA5 数据生产管线，形成约 1.2M clips、约 1200 小时口径的帧级动作数据。
2. 从 Wan/SkyReels I2V 移除文本支路，保留图像条件并加入键盘 cross-attention、鼠标 concat 动作模块。
3. 通过 ODE trajectory 初始化因果学生，再以 Self-Forcing 和 distribution matching distillation 训练，使模型适应自己的生成历史。
4. 结合 3 步去噪、VAE cache、半数 DiT block 动作模块和固定长度 KV cache，在单张 H100 上达到 **25.15 FPS**。
5. 开源 1.8B 权重、普通推理和流式推理代码。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与因果约束

输入为单张条件图像 $I_0$ 与逐帧动作流 $a_{1:t}$。动作分两类：离散键盘移动和连续鼠标视角变化。模型不接收文本，因此输出内容主要受初始图像、历史生成 latent 与动作决定。目标是在线预测

$$
p(x_t\mid x_{<t},a_{\le t},I_0),
$$

并以固定窗口缓存最近历史。输出分辨率为 **352×640**，论文称可生成 597 帧评测序列、1350 帧定性序列和分钟级视频。所谓“无限长度”是滚动缓存不会随总时长增长，并不意味着语义与地图状态可以无限保持。

### 2.2 评价问题

一个合格输出至少要同时满足：单帧质量、时间连贯、键鼠跟随、对象/场景一致性和实时吞吐。论文采用 GameWorld Score 的八项指标，而没有奖励、可规划状态或精确几何误差，因此它评测的是“可交互视频质量”，并未证明模型能替代游戏引擎做确定性仿真。

---

## 3. 核心方法

### 3.1 视觉基础模型

![](assets/Matrix-Game%202.0%20-%20实时流式交互世界模型/figures/02-architecture.png)

> **Figure 8**：条件图像同时进入 3D Causal VAE 与 CLIP image encoder，DiT 根据图像 latent、噪声 latent、动作和时间步预测视频；文本分支被移除。右侧展示鼠标与键盘的不同注入方式。  
> 来源：论文 Figure 8，第 9 页，https://arxiv.org/pdf/2508.13009

基础权重来自 SkyReels-V2-I2V-1.3B。3D Causal VAE 在空间压缩 $8\times8$、时间压缩 4 倍，初始图像还由 CLIP image encoder 提供条件。作者认为移除语言可迫使模型更多从视觉动力学而非文本语义学习世界变化；但这是设计动机，不是由消融严格证明的因果结论。

先对移除文本支路后的模型微调 5K steps，再在每个 DiT block 加动作模块，总参数量变为 **1.8B**，基础模型训练 120K steps，learning rate $2\times10^{-5}$，batch size 256。

### 3.2 动作注入

Matrix-Game 2.0 延续 GameFactory 的异构设计：连续鼠标量与视频 latent 拼接，经 MLP 与 temporal self-attention；离散键盘 embedding 作为 key/value，由视觉特征 query 做 cross-attention。不同点是键盘位置编码改为 RoPE，以便长度外推。最终加速版本只在 DiT 前半层保留动作模块，几乎不损失控制分数，却显著提升吞吐。

### 3.3 从双向教师到因果学生

双向 foundation model 能在完整片段内相互注意，质量好但不能流式。因果学生 $G_\phi$ 使用 block-wise causal mask 与固定窗口。作者先从教师的 probability-flow ODE 轨迹构造训练对：从三步离散时间集合中抽取噪声状态，把序列分成 $L$ 个具有独立时间步的 chunk，以 clean trajectory 为监督：

$$
\mathcal{L}_{student}=\mathbb{E}_{x,t_i}\left\|G_\phi\left(\{x^i_{t_i}\}_{i=1}^{L},\{c_i\}_{i=1}^{L},\{t_i\}_{i=1}^{L}\right)-\{x^i_0\}_{i=1}^{L}\right\|_2^2.
$$

$c_i$ 包含对应图像/动作条件。这个阶段的作用不是最终对齐分布，而是把双向教师稳定地“折叠”为几步因果学生，避免直接做 DMD 时学生输出过差、判别信号失稳。论文收集 **40K ODE pairs**，训练学生 6K steps。

### 3.4 Self-Forcing 与分布匹配蒸馏

![](assets/Matrix-Game%202.0%20-%20实时流式交互世界模型/figures/03-self-forcing.png)

> **Figure 10**：因果 generator 在自己的历史上滚动；对生成序列重新加噪后，冻结的真实 teacher 与持续训练的 fake teacher/critic 构成 DMD，分布匹配梯度回传给 generator。  
> 来源：论文 Figure 10，第 10 页，https://arxiv.org/pdf/2508.13009

Teacher Forcing 用真实历史训练，推理却看到模型历史，形成 exposure bias。Self-Forcing 的关键是训练时也令学生条件于自己生成的前帧。论文随后用 DMD 把学生分布 $p_{\theta,t}(x^{1:N}_t)$ 对齐教师目标分布 $p_{real,t}(x^{1:N}_t)$。论文没有在正文完整展开 DMD 梯度公式，因此不能凭空补写；可以确定的是：real foundation model 冻结，fake 模型以学生样本上的 diffusion loss 更新，二者 score 差形成 distribution matching gradient。

该阶段训练 4K steps，learning rate $6\times10^{-6}$。latent chunk size 为 3，attention local size 为 6。Self-Forcing 是 data-free 的：rollout 动作序列可以人为设计，更接近真实用户输入分布，而不是继续依赖自动脚本常见动作。

### 3.5 滚动 KV cache

推理仅保留固定数量的最近 latent 与动作 embedding；超过容量就移除最旧 token，因此显存不随视频总长度线性增长。窗口过大并不一定更好：早期伪影会被长缓存反复当作真历史，导致错误固化。实验最终选 **6 个 latent frames**，在上下文与自纠错之间折中。

### 3.6 速度优化

速度来自多项叠加，而非单靠少步扩散：

1. Wan2.1 VAE cache；
2. 动作模块只保留在前半 DiT blocks；
3. 去噪步数由 4 降到 3；
4. 因果注意力和固定 KV cache；
5. 单张 H100 上专门优化的推理实现。

因此 25 FPS 不能简单外推到 24 GB 消费卡。官方 README 的 24 GB 是“可运行”门槛，不是实时保证。

### 3.7 训练与流式推理的数据流

基础训练阶段，57 帧 clip 经 3D VAE 压缩为短 latent 序列，条件图像 latent、CLIP image token、键鼠动作与噪声时间步共同进入 DiT。蒸馏后，流式推理不再重复处理全部历史：每次对最新 noisy chunk 做 3 步去噪，读取最近 6 个 latent 的 K/V 和动作 embedding，生成后把新 clean latent 写回 cache，最老条目被淘汰，VAE cache 再增量解码到像素。

这个设计把每步成本从“随已生成总长度增长”改为“近似由固定窗口决定”。但它也切断了窗口外信息，解释了为什么系统可以分钟级运行，却不能可靠记住早期对象。Self-Forcing 解决的是短窗口内自生成历史的分布偏移，KV cache 解决重复计算，二者都不是长期记忆模块。

---

## 4. 数据集与实验设置

### 4.1 Unreal Engine 管线

输入 3D scene 与 navigation mesh，系统通过 NavMesh 路径规划、自动移动和相机控制导出 MP4 与 CSV。NavMesh query 平均低于 **2 ms**。为消除相机旋转约 **0.2%** 的计算错误，作者在中间计算中用双精度四元数。增强输入系统以毫秒级同步多键状态：

$$
Input_{frame_i}=(\{k_1,k_2,\ldots,k_n\},timestamp_i).
$$

RL agent 的奖励为

$$
R_t=\alpha R_{collision}+\beta R_{exploration}+\gamma R_{diversity},
$$

将避碰、探索与轨迹多样性结合。OpenCV 删除冗余帧，并以 $\|\vec v\|>\epsilon$ 过滤静止或物理异常样本。单张 RTX 3090 可并行两路数据生产。

### 4.2 GTA5 管线

通过 Script Hook V 与 C# mod 记录 RGB、键鼠、NPC/车辆交互，OBS 编码视频，CSV/JSON 保存行为。车辆相机按

$$
Camera_{position}=Vehicle_{position}+offset\times rotation
$$

逐 tick 更新。车辆密度范围 0.1–2.0，NPC 密度 0.2–1.5。

### 4.3 数据规模与统计口径

论文数据口径容易混淆：摘要说约 **1200 小时**；实验中 foundation model 训练集写作约 **800 小时**，包含 153h Minecraft、615h Unreal，加上清洗后 85h Sekai；另外为域微调收集 574h GTA-driver 和 560h TempleRun。所有视频 resize 为 352×640，主训练 clip 为 57 帧。数据生产共超过 **1.2M clips**、准确率超过 99%，相机旋转精度提升 50 倍。这里“1200h”更接近总体生产规模，不应与主训练混为一谈。

### 4.4 Baseline 与评测

Minecraft 对比 **Oasis**，wild scene 对比 **YUME**。统一使用 597 帧复合动作序列，评测 32 个 Minecraft 场景和 16 张开放域图像。指标包括 Image Quality、Aesthetic、Temporal Consistency、Motion Smoothness、Keyboard Accuracy、Mouse Accuracy、Object Consistency、Scenario Consistency。

---

## 5. 实验结果

### 5.1 Minecraft：对 Oasis

![](assets/Matrix-Game%202.0%20-%20实时流式交互世界模型/figures/04-minecraft-results.png)

> **Figure 11 / Table 1**：Oasis 在数十至数百帧后模糊或饱和，Matrix-Game 2.0 保持可辨场景；定量上后者在画质、时间一致、动作准确和对象一致性明显领先。  
> 来源：论文 Figure 11、Table 1，第 11 页，https://arxiv.org/pdf/2508.13009

| 指标 | Oasis | Matrix-Game 2.0 |
|---|---:|---:|
| Image Quality↑ | 0.27 | **0.61** |
| Aesthetic↑ | 0.27 | **0.50** |
| Temporal Consistency↑ | 0.82 | **0.94** |
| Motion Smoothness↑ | **0.99** | 0.98 |
| Keyboard Accuracy↑ | 0.73 | **0.91** |
| Mouse Accuracy↑ | 0.56 | **0.95** |
| Object Consistency↑ | 0.18 | **0.64** |
| Scenario Consistency↑ | **0.84** | 0.80 |

Oasis 在 Motion/Scenario 上略高，作者指出其坍塌后趋于静止，反而会抬高“平滑”和“场景不变”指标。这提示世界模型评测不能只看稳定性：完全不动的错误视频也可能“很一致”。

### 5.2 Wild scenes：对 YUME

Matrix-Game 2.0 相比 YUME：Image **0.67 vs 0.65**、Aesthetic **0.51 vs 0.48**、Temporal **0.86 vs 0.85**；但 Motion **0.98 vs 0.99**、Object **0.71 vs 0.77**、Scenario **0.76 vs 0.80** 较低。论文解释 YUME 后段静态化会虚高一致性。由于 wild scene benchmark 没有适用的键鼠准确率，开放域动作控制主要靠定性演示，证据弱于 Minecraft。

### 5.3 长视频与实时性

![](assets/Matrix-Game%202.0%20-%20实时流式交互世界模型/figures/05-long-video-speed-ablation.png)

> **Figure 13 / Table 3**：展示到 1350 帧的多域结果；VAE cache、减半动作模块、3 步去噪逐步把吞吐从 15.49 提升到 25.15 FPS，质量指标基本保持。  
> 来源：论文 Figure 13、Table 3，第 13 页，https://arxiv.org/pdf/2508.13009

加速消融：仅 VAE cache 为 **15.49 FPS**；动作模块减半后 **21.03 FPS**；去噪 4→3 步后 **25.15 FPS**。最终 Keyboard/Mouse 为 **0.91/0.95**，与中间版本近似。这里最有价值的工程结论是动作调制无需铺满全部 DiT blocks，早层注入已足够驱动控制。

### 5.4 KV cache 消融与失败案例

![](assets/Matrix-Game%202.0%20-%20实时流式交互世界模型/figures/06-kv-cache-and-failure-cases.png)

> **Figure 16–17**：9-latent 缓存比 6-latent 缓存更早固化伪影；OOD 场景中持续向上转或长时间前进会出现过饱和、黑屏或整体退化。  
> 来源：论文 Figure 16–17，第 15 页，https://arxiv.org/pdf/2508.13009

窗口从 6 增至 9 本想增加上下文，结果反而更差。说明纯 KV 记忆是一把双刃剑：它保存场景，也保存错误；若没有显式状态校正，增加历史不等于增加记忆。官方代码还提示向上移动相机可能短暂黑屏，改变方向后可恢复。

### 5.5 实验充分性判断

论文较好地覆盖了速度、长序列、动作跟随、跨域和失败案例；对“实时交互”主张的支撑较强。但训练数据未公开、25 FPS 只在 H100 上报告、开放域动作准确缺乏定量评价、长期记忆仅有视觉展示，因此对“通用世界模型”的支撑仍有限。

速度消融尤其有说服力，因为每一步优化同时报告质量和 FPS，而不是只给最终数字；相反，Self-Forcing 本身缺少“相同 3 步学生、只移除 Self-Forcing”的完整长期定量曲线。论文通过 Oasis/YUME 的坍塌对比间接支持它，但仍难区分收益来自数据、初始化、因果 mask 还是分布匹配训练。

---

## 6. 与相关工作的关系

GameFactory 首先确立了键盘 cross-attention + 鼠标 concat 与开放域控制迁移；Matrix-Game 2.0 复用这一思想，把重点转向因果化和实时化。相较 Oasis，它保持更长视觉稳定性；相较普通 Diffusion Forcing，它用 Self-Forcing 直接训练在自身历史上；相较 CausVid，额外引入动作流和固定 KV cache，面向持续人机交互。

后续 Solaris 以其 distilled checkpoint 为初始化，将单玩家 DiT 扩展为多人共享注意力，并进一步发现 ODE 初始化和预先 few-step DMD 可以简化为 causal finetuning + Self-Forcing。这也说明 Matrix-Game 2.0 的训练链条并非唯一方案。

---

## 7. 局限与批判性评价

### 7.1 作者承认的限制

- OOD 中持续向上转或长时间前进会过饱和、黑屏或退化；
- 352×640 低于主流高分辨率视频模型；
- 没有显式历史记忆，长视频内容一致性仍会下降。

### 7.2 我的判断

第一，25 FPS 是重要但硬件特定的吞吐数字；用户动作到可见反馈的延迟、首帧延迟和输入采样周期未完整披露。第二，模型只在滚动像素 latent 中“记忆”，没有地图、对象或库存状态，因此回访地点和不可见对象持续存在仍不可靠。第三，动作数据来自脚本或游戏钩子，跨真实视频或新控制器的迁移仍未验证。第四，GameWorld Score 中一些指标会奖励静态坍塌，需引入动作反事实、回环和状态变化验证。

---

## 8. 复现与实践建议

### 8.1 推理复现

官方仓库提供 `inference.py` 与 `inference_streaming.py`、三个预训练权重和 MIT 代码许可。环境为 Linux、Python 3.10、Apex、FlashAttention，建议至少 24 GB NVIDIA GPU 与 64 GB RAM。24 GB 是运行门槛；若目标是论文的 25 FPS，应准备 H100 并使用相同优化路径。

### 8.2 训练复现

官方未公开完整训练代码与 UE/GTA 数据管线。论文给出 120K foundation steps、batch 256、40K ODE pairs、6K 初始化 steps、4K Self-Forcing steps，但缺少完整数据和基础训练基础设施。因此“加载权重跑推理”可复现，“从数据生产到 25 FPS 模型”不可严格复现。

### 8.3 工程建议

先复现 150 帧离线推理，再测试 streaming；记录首帧延迟、平均/尾部 FPS、显存和不同动作频率。对每个场景增加 no-op、相反动作、回环动作三类反事实测试，避免仅凭好看的视频判断控制。对 KV cache 同时测 6/9 latent，并观察错误一旦出现是否被固化。

---

## 9. 个人启发与后续问题

Matrix-Game 2.0 说明，实时世界模型不是单一算法突破，而是**因果架构、训练分布、蒸馏、缓存和 VAE 工程共同优化**。更大的上下文甚至可能更差，意味着下一阶段不能只扩窗口，而要引入可校正的状态记忆：显式 3D、对象槽、检索记忆或周期性重建。

值得继续追问：如何把 25 FPS 提升为低尾延迟交互；如何在不牺牲实时性的情况下加入长期地图；如何将 Self-Forcing 的分布匹配扩展到动作反事实；以及如何构建不会奖励静态坍塌的世界模型指标。

---

## 10. 材料来源

完整来源、图表追溯与资料缺口见：[sources.md](assets/Matrix-Game%202.0%20-%20实时流式交互世界模型/sources.md)。本地保存官方 PDF：`assets/Matrix-Game 2.0 - 实时流式交互世界模型/sources/paper.pdf`。
