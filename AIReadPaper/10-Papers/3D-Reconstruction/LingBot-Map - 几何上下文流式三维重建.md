---
type: paper-note
status: complete
domain: 3D-Reconstruction
paper: Geometric Context Transformer for Streaming 3D Reconstruction
year: 2026
arxiv: 2604.14141
doi: 10.48550/arXiv.2604.14141
source: https://arxiv.org/abs/2604.14141
project: https://technology.robbyant.com/lingbot-map
code: https://github.com/Robbyant/lingbot-map
code_commit: b9dd1b1de535e6d36e777fb42b8433c781bfd783
tags:
  - 3D-Reconstruction
  - Streaming-3D
  - SLAM
  - VGGT
  - Geometric-Context-Attention
  - Paged-KV-Cache
created: 2026-10-26
updated: 2026-10-26
---

# LingBot-Map：几何上下文流式三维重建

## 基本信息

- **论文正式名**：*Geometric Context Transformer for Streaming 3D Reconstruction*
- **作者**：Lin-Zhuo Chen、Jian Gao、Shangzhan Zhang、Yihang Chen、Ka Leong Cheng、Yipengjing Sun、Liangxiao Hu、Nan Xue、Xing Zhu、Yujun Shen、Yao Yao、Yinghao Xu
- **版本**：arXiv:2604.14141 v3，2026-09-08；ECCV 2026
- **项目页**：<https://technology.robbyant.com/lingbot-map>
- **官方代码**：<https://github.com/Robbyant/lingbot-map>，Apache-2.0；本文代码核查 commit 为 `b9dd1b1de535e6d36e777fb42b8433c781bfd783`
- **官方权重**：<https://huggingface.co/robbyant/lingbot-map>，含 `lingbot-map`、偏长序列的 `lingbot-map-long` 与可载入 VGGT 双向推理的 `lingbot-map-stage1`
- **本地论文**：[paper.pdf](assets/LingBot-Map%20-%20几何上下文流式三维重建/sources/paper.pdf)
- **定位**：单目 RGB 视频的因果流式三维重建；逐帧输出相机位姿、稠密深度/世界点及置信度，不使用未来帧，也不在默认 Direct 模式中做测试时优化。

## 一句话结论

LingBot-Map 的核心不是再造一个更大的 VGGT，而是把经典 SLAM 的“初始坐标锚点—局部关键帧窗口—全局轨迹摘要”翻译成可端到端学习的 **Geometric Context Attention（GCA）**：锚帧和最近窗口保留完整图像 token，远历史每帧仅保留 6 个相机/锚点/寄存器 token，再以 Video RoPE 注入顺序；它因此把因果注意力每帧新增约 500 个图像 token 的代价压到 6 个摘要 token，在 518×378 上实现 H800 约 20 FPS、Oxford Spires 3840 帧 ATE 5.60，并能借助窗口式 VO 跑到万帧以上，但超长程仍会因窗口重置与 Sim(3) 拼接产生漂移，也没有显式回环和 BA。

---

## 1. 研究背景与问题定义

### 1.1 为什么“离线 3D 基础模型”不能直接变成在线地图

VGGT、Depth Anything 3 等模型已能一次前向从一组图像估计相机、深度和点图，但它们默认所有视图同时到齐，允许跨帧双向全注意力。机器人、AR 眼镜和自动驾驶面对的是不断到来的视觉流：第 $t$ 帧只能使用 $I_{1:t}$，还必须在视频越来越长时控制显存和延迟。

已有流式路线各自牺牲一项能力：CUT3R 把历史压进固定 recurrent state，状态紧凑但容易遗忘；StreamVGGT/Stream3R 保留近完整历史 KV，几何信息丰富但成本随序列增长；VGGT-SLAM/MASt3R-SLAM 用传统关键帧、图优化与 BA 管理上下文，精度高却重新引入手工模块和迭代开销。LingBot-Map 把问题重新表述为：**哪些历史信息值得永久保留，应以完整观测还是摘要 token 保留？**

![](assets/LingBot-Map%20-%20几何上下文流式三维重建/figures/01-overview-results.png)

> **图 1（论文 Figure 1）**：Oxford Spires 上 LingBot-Map 的 ATE 为 5.4 m，显著低于 TTT3R、Wint3R、Stream3R 和 InfiniteVGGT；雷达图显示其优势同时覆盖位姿和重建，而非只优化单一任务。需要注意，该图是跨数据集归一化概览，严谨数字仍应读正文表格。  
> 来源：论文 Figure 1，第 1 页，<https://arxiv.org/abs/2604.14141>

### 1.2 任务定义

给定无界图像流：

$$\mathcal I=\{I_1,I_2,\ldots\},$$

在第 $t$ 帧到来时，模型仅使用当前与过去图像估计相机位姿和深度：

$$\left(\hat P_t,\hat D_t\right)=f_\theta\left(I_t;\mathcal C_{t-1}\right),$$

其中 $\mathcal C_{t-1}$ 是结构化流式状态，不能包含未来帧。相机头预测 camera-to-world 位姿编码，深度头输出稠密深度，并可结合内参反投影为世界坐标点云。论文真正优化的是三个同时冲突的目标：几何精度、长时一致性、每帧计算/显存可控。

### 1.3 核心贡献

1. 提出 GCA，把上下文拆成 Anchor Context、Local Pose-Reference Window 和 Trajectory Memory。
2. 用“完整近期观测 + 每帧 6-token 远程摘要”代替全历史图像 token，显著降低长序列成本。
3. 以两阶段训练、24→320 视图课程、相对位姿损失和 Ulysses context parallel 稳定优化。
4. 通过 FlashInfer paged KV cache 落地实时推理，并提供 Direct 与窗口式 VO 两种模式。
5. 在 Oxford Spires、ETH3D、7-Scenes、Tanks and Temples、NRGBD、Sintel 和 KITTI 上系统评估。

---

## 2. 表示与总体架构

### 2.1 每帧 token 组成

输入图像经 DINOv2-Large ViT（patch size 14）变为 $M$ 个图像 token，并添加：

- 1 个 camera token $c\in\mathbb R^C$；
- 4 个 register tokens $r_j\in\mathbb R^C$；
- 1 个 anchor token $a\in\mathbb R^C$。

因此每帧共有 $M+6$ 个 token。网络沿用 VGGT 风格，堆叠 24 组交替的 Frame Attention 与跨帧注意力；不同在于第二阶段把全局跨帧注意力替换为 GCA。Frame Attention 在单帧内部整理视觉特征，GCA 决定当前帧能从哪些历史 token 读取几何上下文。

### 2.2 完整 pipeline

![](assets/LingBot-Map%20-%20几何上下文流式三维重建/figures/03-method-pipeline.png)

> **图 4（论文 Figure 4）**：前 $n$ 帧形成锚点，超出局部窗口的旧帧只留下紫色 camera token 与橙色 context tokens，当前帧则同时读取锚点、远历史摘要和最近完整观测；最终 camera head 从 camera token 解码位姿，DPT depth head 从 image tokens 解码深度。  
> 来源：论文 Figure 4，第 7 页，<https://arxiv.org/abs/2604.14141>

### 2.3 输出与张量形状

官方 `demo.py` 的高层输入为 `[S,3,H,W]`；内部先以初始尺度帧 `[1,F_s,3,H,W]` 建立坐标与尺度，再逐帧输入 `[1,1,3,H,W]`。典型单序列输出为：

| 输出 | 形状 | 语义 |
|---|---|---|
| `pose_enc` | `[S,P]` | 相机编码，后处理为内外参 |
| `extrinsic` | `[S,3,4]` | Demo 最终转换后的 W2C；camera head 原始语义为 C2W |
| `intrinsic` | `[S,3,3]` | 相机内参 |
| `depth` | `[S,H,W,1]` | 稠密深度 |
| `depth_conf` | `[S,H,W]` | 深度置信度 |
| `world_points` | `[S,H,W,3]` | 世界坐标点 |
| `world_points_conf` | `[S,H,W]` | 世界点置信度 |

---

## 3. Geometric Context Attention

### 3.1 Anchor Context：确定坐标和尺度

单目重建有尺度不确定性。模型取最前面的 $n$ 帧为锚帧，锚帧内部全注意力，并永久保留其 image/camera/register/anchor tokens；后续帧都能把它们当作固定坐标参考。锚帧和流式帧使用不同可学习 anchor token，使网络能区分初始化阶段与常规帧。

训练时以锚帧真值点云的平均原点距离归一化全序列：

$$s=\frac{1}{|\bar X_{\mathrm{anchor}}|}\sum_{x\in\bar X_{\mathrm{anchor}}}\|x\|_2,$$

再用 $s$ 除所有深度和相机平移。它不是恢复现实世界绝对米制尺度，而是让同一视频以锚点定义的统一规范尺度展开。

### 3.2 Local Pose-Reference Window：保留稠密局部重叠

最近 $k$ 帧保留全部 $M+6$ 个 token。局部窗口承担“跟踪”职责：相邻帧通常有大量视觉重叠，完整 image tokens 比压缩状态更适合做细粒度匹配和相对位姿。训练中 $k$ 在 16–64 随机采样，默认推理取 64。

### 3.3 Trajectory Memory：远历史每帧仅留 6 个 token

一旦某帧既不属于 anchor，也被挤出局部窗口，模型丢弃其 $M$ 个图像 token，只保留 camera、anchor 和 4 个 register tokens，共 6 个。它们是网络已经加工过的几何摘要，不是手工 pose graph 节点。Video RoPE 给这些摘要加入时间顺序，让模型知道“这帧何时出现、相隔多远、运动方向如何”。

### 3.4 注意力模式与复杂度

![](assets/LingBot-Map%20-%20几何上下文流式三维重建/figures/02-gca-attention-mask.png)

> **图 3（论文 Figure 3）**：全注意力无法流式；标准 causal 保留全部历史，成本持续增长；纯滑窗丢掉远程上下文；GCA 则对 anchor 和最近窗口保留完整块，对远历史只保留窄条 context tokens。  
> 来源：论文 Figure 3，第 6 页，<https://arxiv.org/abs/2604.14141>

$T$ 帧时，GCA 的上下文 token 数为：

$$n(M+6)+k(M+6)+6(T-n-k)=(n+k)M+6T.$$

标准 causal attention 为：

$$T(M+6)=MT+6T.$$

每增加一帧，标准 causal 追加 $M+6$，GCA 对被驱逐的远历史只追加 6 个摘要 token。论文以 $M\approx500,n=3,k=16,T=10,000$ 举例：前者约 $5\times10^6$ tokens，后者约 $7\times10^4$。论文摘要称约 70×，结论处写约 80×，应理解为随实际 $M/n/k$ 变化的数量级而非固定常数。

---

## 4. 损失与训练

### 4.1 总目标

$$\mathcal L=\lambda_{\mathrm{depth}}\mathcal L_{\mathrm{depth}}+\lambda_{\mathrm{abs}}\mathcal L_{\mathrm{abs-pose}}+\lambda_{\mathrm{rel}}\mathcal L_{\mathrm{rel-pose}}.$$

深度损失沿用 VGGT，联合像素深度、梯度与预测不确定性；绝对位姿同样沿用 VGGT 形式，但监督 camera-to-world 而非 world-to-camera，以减轻长序列中旋转误差对平移的耦合放大。

局部窗口内所有帧对使用相对位姿约束：

$$\mathcal L_{\mathrm{rel-pose}}=\frac{1}{k(k-1)}\sum_{\substack{i\ne j\\i,j\in\{1,\ldots,k\}}}\left(\mathcal L_{\mathrm{rot}}(i,j)+\lambda_{\mathrm{trans}}\mathcal L_{\mathrm{trans}}(i,j)\right),$$

其中旋转用 geodesic error，平移用 $\ell_1$。这个损失只比较已观察帧，严格保持因果性；其作用类似把传统 SLAM 的局部几何一致性约束蒸进网络训练。

### 4.2 Stage 1：离线几何基座

- DINOv2-Large 初始化，24 组 Frame/Cross-frame blocks；跨帧使用全注意力。
- 每样本随机 2–24 视图；混合 unordered multi-view、video 与 mesh 数据。
- AdamW，学习率 $2\times10^{-4}$，weight decay 0.05，5% warmup 后 cosine 衰减，160K iterations。
- 最大边 518；强 color jitter、blur、grayscale、尺度/宽高比变化，30% 概率整段 co-jitter。
- 64 GPUs 约两周，论文估计约 21,500 GPU hours；FSDP、gradient checkpointing、bf16。

### 4.3 Stage 2：流式长轨迹

- 从 Stage 1 迁移 Q/K/V，直接以 GCA 替换全局注意力。
- 160K iterations，学习率 $5\times10^{-5}$。
- 视图数从 24 线性增加到 320；窗口 $k\sim U[16,64]$。
- Ulysses context parallel dimension 16，构建于 TorchTitan/Magi Attention；64 GPUs 约一周。
- 数据向长轨迹倾斜，并引入 foldback video sampler：到视频边界后反向并重采 stride，避免只学习单向时间推进。

### 4.4 数据覆盖

论文列出 31 个数据入口，覆盖 BlendedMVS、HyperSim、MegaDepth、CO3D、Objaverse、TartanAir/V2/Ground、DL3DV、ScanNet/++、MatrixCity、MidAir、Waymo、KITTI-360、Gibson、Matterport3D、HM3D 与内部游戏视频等。Stage 1 强调多样短程几何，Stage 2 提高长视频/多场景轨迹权重。数据配比公开到表 1，但内部游戏数据和全部清洗细节没有开放，完整预训练仍无法按论文原样复现。

---

## 5. 推理系统

### 5.1 Paged KV cache

GCA 仍需频繁追加新帧、把旧帧从完整窗口驱逐为摘要。如果采用连续 KV buffer，每次更新可能引发搬移和重分配；官方用 FlashInfer 的 paged KV cache，只更新新页和映射。论文在 518×378、最长 1000 帧、窗口 64 下报告约 20 FPS，而等价 PyTorch 连续缓存约 10.5 FPS。

### 5.2 Direct Output 与 VO 两种模式

| 模式 | 工作方式 | 适用范围 | 主要误差 |
|---|---|---|---|
| Direct | 三层状态持续累积、不重置；每帧直接输出绝对 pose/depth | 论文经验上约 3000 帧以内最佳 | 纯模型逐帧漂移 |
| VO / windowed | 长视频切重叠窗口；窗口内重新定尺度，窗口间做 Sim(3) 对齐 | 数万帧与城市尺度 | 每次窗口拼接额外漂移 |

超过训练的 320 视图时，缓存每隔 $m$ 帧保留一个 keyframe；非关键帧仍产生预测，只是不写入长期缓存。工程 README 建议万帧级使用窗口模式，并提供约 25,000 帧、13 分钟视频示例。

### 5.3 默认配置与速度

论文默认 518×378、bf16、$k=64,m=1$。Table 9 给出 $n=8,k=64$ 时 Oxford ATE 5.3，H800 约 20 FPS、RTX 4090 约 12 FPS。窗口从 64 改为 16 可到 H800 26 FPS，但 ATE 从约 5.3 恶化到 11–12，说明主要精度—速度旋钮是局部稠密窗口，而非 anchor 数量。

---

## 6. 实验结果

### 6.1 Oxford Spires：长轨迹是最强证据

![](assets/LingBot-Map%20-%20几何上下文流式三维重建/figures/04-oxford-results.png)

> **表 2–3（论文 Table 2–3）**：320 帧稀疏设置中，LingBot-Map AUC@15=63.22、ATE=5.37，超过离线 DA3（49.84/12.87）和优化式 VIPE（45.35/10.52）；扩到 3840 帧时 ATE 仅升到 5.60（+0.23），而 CUT3R 从 18.16 升到 32.47。  
> 来源：论文 Table 2–3，第 13 页，<https://arxiv.org/abs/2604.14141>

这个结果支持的最强结论是：**有选择的结构化历史优于“全量缓存”与“单固定隐状态”**。但不能简单表述为“神经网络全面击败 SLAM”：VIPE 等基线的工程目标、硬件和输入预处理不同，LingBot-Map 也在 31 类数据上做了大规模监督预训练。

### 6.2 其他位姿基准

| 数据集 | LingBot-Map | 最强对照（论文表内） | 结论 |
|---|---:|---:|---|
| ETH3D ATE | **0.43** | Wint3R 0.86 | 约 2× 更低 |
| 7-Scenes ATE | **0.08** | 多方法 0.11 | 室内短轨迹仍领先 |
| Tanks & Temples ATE | **0.21** | TTT3R 0.67 | 约 3.2× 更低 |
| Sintel ATE | **0.10** | VGGT-Long 0.17 | 动态视频上有优势，但模型没有显式动态分解 |
| KITTI ATE | **24.12** | VGGT-Long 31.89 | 城市驾驶序列更稳 |

### 6.3 稠密点云

Table 6 中 LingBot-Map 在 ETH3D/7-Scenes/NRGBD 的 F1 分别为 **86.80/82.38/65.10**；对应最强流式对照 Wint3R 为 77.28/78.81/56.96。位姿漂移减少后，同一表面不再被投到多个位置，因此 Accuracy 与 Completeness 同时改善。

![](assets/LingBot-Map%20-%20几何上下文流式三维重建/figures/05-pointcloud-comparison.png)

> **图 6（论文 Figure 6）**：TTT3R 与 Wint3R 在多建筑和长回访序列中出现墙面重影、结构重复和点云断裂；LingBot-Map 的轮廓与连续表面更完整。它说明 pose 稳定性会直接转化为稠密几何质量，但图中没有展示动态物体或低纹理极端失败例。  
> 来源：论文 Figure 6，第 16 页，<https://arxiv.org/abs/2604.14141>

### 6.4 消融实验

![](assets/LingBot-Map%20-%20几何上下文流式三维重建/figures/06-ablation.png)

> **表 7（论文 Table 7）**：Anchor Init 把 AUC@3 从 9.80 提到 13.63；加入 trajectory context tokens 后到 15.75；Video RoPE 带来最大单项 ATE 改善 7.46→5.98。没有 relative pose loss 时 RPE-rot 从 2.26 恶化到 5.35，说明局部旋转约束特别关键。  
> 来源：论文 Table 7，第 18 页，<https://arxiv.org/abs/2604.14141>

另一个关键消融是窗口 64 对比全历史 causal attention：

| 注意力 | ATE ↓ | RPE-trans ↓ | RPE-rot ↓ | FPS ↑ | 显存 GB ↓ |
|---|---:|---:|---:|---:|---:|
| GCA 窗口 64 | **5.98** | **1.33** | 1.93 | **20.29** | **13.28** |
| Full causal | 6.60 | 1.50 | **1.71** | 11.87 | 36.06 |

GCA 不只是更省：过滤掉遥远帧的冗余 image tokens 后，全局 ATE 反而更好；只有局部旋转略输，说明完整历史确实仍含细粒度旋转线索，但收益不足以抵偿噪声与成本。

---

## 7. 与相关方法的关系

| 方法 | 长时状态 | 是否优化 | 关键差异 |
|---|---|---:|---|
| VGGT / DA3 | 离线全局 token | 否 | 看未来帧，不能直接在线 |
| CUT3R | 固定 recurrent state | 否 | 常量状态，但压缩导致遗忘 |
| Stream3R / StreamVGGT | 近全历史 KV | 否 | 因果，但成本与冗余持续增长 |
| Wint3R | 窗口 + camera token pool | 否 | 同样重视窗口，GCA 更明确分成三层上下文 |
| VGGT-SLAM / MASt3R-SLAM | 关键帧/pose graph | 是 | 精度依赖手工系统与迭代后端 |
| LingBot-Map | anchor + 完整局部 + 6-token/帧全局摘要 | Direct 否；VO 做 Sim(3) 拼接 | 把 SLAM 式状态分层写成 attention mask 与学习式 token |

LingBot-Map 与 [[CUT3R - 带持久状态的连续三维感知]] 的真正差异在于“记忆容量是否固定”：CUT3R 把所有过去不断压入同一个状态；LingBot-Map 允许轨迹摘要随 $T$ 以每帧 6 tokens 缓慢增长，所以并非严格常量内存，而是把增长率从约 $M+6$ 降到 6。它与 [[VGGT - 前馈式通用三维重建]] 则是继承关系：DINO/VGGT 的几何骨架负责“会重建”，GCA 负责“能长期在线重建”。

---

## 8. 局限与批判性评价

### 8.1 作者明确承认的边界

1. 无显式 loop closure；重访只能通过注意力里的隐式轨迹记忆减漂移。
2. 无 test-time optimization；困难帧不会通过 BA 或 reprojection refinement 纠正。
3. 超过 Direct 有效范围后必须切窗，Sim(3) 窗口拼接误差会累计。
4. 当前不支持 distortion-aware camera、LiDAR/IMU 融合和显式动态场景建模。

### 8.2 额外判断

- “万帧稳定”主要依赖 VO/windowed 模式，不能与 Direct 模式 3840 帧的近恒定 ATE 混为一谈。
- 论文把每帧计算称为“nearly constant”，但 trajectory memory 仍以 $6T$ 增长；只是相比 image tokens 极慢。
- Sintel 成绩好不代表模型解决了动态 4D：它仍假设主相机运动与静态背景几何，移动物体缺少独立轨迹/分层表示。
- 大规模训练成本约为 Stage 1 的 21,500 GPU hours 加 Stage 2 的 64 GPU×一周，且内部游戏数据未开源；代码可跑不等于预训练可复现。
- Anchor 固定首段坐标有工程脆弱性：若开头纹理差、模糊或运动过快，错误尺度会成为全局参考。论文没有专门给“坏开头”鲁棒性实验。

### 8.3 我的核心评价

**LingBot-Map 最有价值的贡献是把“空间记忆的粒度”变成架构设计变量。** 过去讨论流式 3D 模型时常在“固定隐状态”和“全量 KV”之间二选一；GCA 说明可以按几何角色混合粒度：全局坐标需要少量完整锚帧，当前跟踪需要完整近期视觉，远历史只需紧凑但有时间顺序的 pose-aware tokens。这种思路不仅适用于重建，也很适合作为 VLA/世界模型的可查询空间状态。

---

## 9. 复现与实践建议

### 9.1 权重选择

- `lingbot-map`：短/长序列平衡，论文 benchmark 的主选择。
- `lingbot-map-long`：仓库当前推荐长序列与大场景。
- `lingbot-map-stage1`：离线 Stage-1 几何权重，可载入 VGGT 做双向 C2W 推理，不是最终流式模型。

### 9.2 最低可行复现

1. Python 3.10，安装与 CUDA/PyTorch 对应的 FlashInfer；无 FlashInfer 可用 SDPA，但速度更低。
2. 先用 518 分辨率、短序列与官方 example 验证位姿/深度输出。
3. 320 帧以内用 `keyframe_interval=1`；更长先稀疏写入 KV，不要直接把 `stride` 当同一概念。
4. 约 3000 帧以内优先 Direct；更长才用 windowed/VO，并同时报告窗口边界漂移。
5. Outdoor 点云可用 sky mask 清理显示，但天空分割是工程后处理，不是模型重建能力。
6. 评测必须固定 commit；仓库历史曾修复 FlashInfer/SDPA KV-cache bug，旧代码会静默影响长序列质量。

### 9.3 复现成本

主权重约 4.6 GB；推理端论文给出 H800/RTX 4090 的 FPS，但未给完整最低显存表。离线渲染另依赖 Kaolin、Open3D、ffmpeg 与本地 CUDA 扩展，比 `demo.py` 更难配置。完整预训练需要 64 GPU 级资源、31 个数据入口及内部数据，不适合普通实验室从零复现。

---

## 10. 代码级实现

### 10.1 仓库概览与论文模块映射

| 论文模块 | 代码入口 | 作用 |
|---|---|---|
| 流式主模型 | `lingbot_map/models/gct_stream.py::GCTStream` | KV cache 初始化、尺度帧与逐帧推理 |
| 超长序列 | `lingbot_map/models/gct_stream_window.py::GCTStream` | 窗口重置、重叠区域对齐 |
| GCA 聚合 | `lingbot_map/aggregator/stream.py` | Anchor/Window/Trajectory 的 mask 与缓存逻辑 |
| 离线基座 | `lingbot_map/aggregator/base.py`、`gct_base.py` | DINOv2 patch tokens 与 VGGT 式交替 block |
| 相机头 | `lingbot_map/heads/camera_head.py` | 多次迭代预测 pose encoding |
| 稠密头 | `lingbot_map/heads/dpt_head.py` | DPT 多尺度融合，输出 depth/world points/confidence |
| 注意力后端 | `lingbot_map/layers/block.py` | FlashInfer 与 SDPA 路径 |
| 位置编码 | `lingbot_map/layers/rope.py` | 图像空间/视频时序 RoPE |
| Demo | `demo.py` | 输入预处理→模型加载→流式推理→位姿后处理→Viser |

仓库当时较大，本文以官方页面和源码入口核对为主；具体文件路径可在当前 `main` 继续验证。关键行为已由 `demo.py` 直接确认：默认 `img_size=518`、`patch_size=14`、`kv_cache_sliding_window=64`、`num_scale_frames=8`、`camera_num_iterations=4`、3D RoPE 开启。

### 10.2 全链路张量总表

以输入预处理后 $H\times W=378\times518$、patch 14 为例，patch 网格约 $27\times37$，每帧约 $M=999$ 个 image tokens（不同 crop/实现可能变化，因此代码与论文复杂度示例的 $M\approx500$ 不应混用）：

| 阶段 | 张量 | 典型形状 |
|---|---|---|
| 高层输入 | images | `[S,3,378,518]` |
| 初始尺度批 | warm scale | `[1,8,3,378,518]` |
| 单帧流式输入 | current frame | `[1,1,3,378,518]` |
| DINO patch tokens | image tokens | `[1,F,M,C]` |
| 特殊 token 拼接 | frame tokens | `[1,F,M+6,C]` |
| 锚点缓存 | anchor KV | `[n,M+6,C]` 等价 token 集 |
| 局部窗口 | local KV | `[k,M+6,C]` |
| 轨迹摘要 | trajectory KV | `[T-n-k,6,C]` |
| 位姿编码 | pose_enc | `[1,S,P]` |
| 深度 | depth | `[1,S,H,W,1]` |
| 世界点 | world_points | `[1,S,H,W,3]` |
| 后处理外参 | extrinsic | `[S,3,4]` |
| 后处理内参 | intrinsic | `[S,3,3]` |

### 10.3 逐模块拆解

#### 10.3.1 DINO + Frame Attention

- **输入输出**：RGB `[B,F,3,H,W]` → patch features `[B,F,M,C]`。
- **为什么**：DINOv2-Large 提供强视觉局部表征；Frame Attention 先在单帧内整理纹理与结构，跨帧 GCA 不必同时承担图像编码。
- **关键超参**：patch 14、24 组交替 blocks、最大尺寸 518。
- **代码与论文差异**：Demo 的实际 crop 可能让 $H/W$ 非正方形；论文复杂度中的 $M\approx500$ 是说明性取值。

#### 10.3.2 GCA 与缓存

- **输入输出**：当前帧 query 读取锚点完整 KV、最近 $k$ 帧完整 KV、远历史 6-token KV。
- **为什么**：用不同信息粒度匹配“定尺度、局部注册、长时漂移”三类几何需求。
- **关键超参**：论文默认 $k=64$；Demo 还有 `num_scale_frames=8`、自动 `keyframe_interval`。
- **代码与论文差异**：论文强调三个逻辑上下文；工程又加入关键帧间隔、窗口模式和 paged cache slot 管理，以跨越训练 320 视图范围。

#### 10.3.3 Camera Head

- **输入输出**：camera token → `pose_enc [B,F,P]` → C2W pose 与 intrinsics；Demo 最终反演为 W2C `[S,3,4]`。
- **为什么**：相机位姿是将逐帧深度放入统一世界坐标的关键；迭代 refinement 改善精度。
- **关键超参**：默认 4 次迭代；可降为 1 次省约 4× camera-head cache，牺牲少量位姿精度。
- **代码与论文差异**：论文主要描述一个 camera head，工程暴露了迭代次数和独立缓存优化。

#### 10.3.4 DPT Dense Head

- **输入输出**：多层 image tokens → depth/world points/confidence，空间形状恢复到 `[B,F,H,W,*]`。
- **为什么**：DPT 融合多尺度 Transformer features，兼顾局部边缘与全局结构。
- **关键超参**：输出分辨率随 canonical crop；置信度阈值主要用于点云显示与过滤。
- **代码与论文差异**：项目页强调 camera+depth；Demo 还直接提供 world points 及置信度，便于可视化。

#### 10.3.5 Windowed VO

- **输入输出**：长视频 → 重叠局部窗口预测 → Sim(3) 对齐后的全局轨迹/点云。
- **为什么**：RoPE 与模型状态超出训练范围后直接模式逐渐退化，重置可以把每段拉回训练分布。
- **关键超参**：`window_size`、`overlap_keyframes`、`keyframe_interval`。
- **代码与论文差异**：它是必要的工程扩展，但引入传统几何对齐，因此“纯自回归单模型万帧”需要加上这个限定。

### 10.4 计算账与复现建议

GCA 并未改变单帧内 DINO/DPT 的主成本，节省主要发生在跨帧 attention：完整视觉 token 只覆盖 $n+k$ 帧，远历史以 6 token/帧线性增长。Table 8 显示窗口化将显存从 36.06 GB 降至 13.28 GB、FPS 从 11.87 升至 20.29。实际部署时优先做三组 Pareto：$k\in\{32,48,64\}$、camera iterations $\in\{1,4\}$、keyframe interval；同时分别报告 pose、重建 F1、峰值显存与端到端 FPS，避免只看可视化。

---

## 11. 材料来源

完整附件与溯源见 [sources.md](assets/LingBot-Map%20-%20几何上下文流式三维重建/sources.md)。
