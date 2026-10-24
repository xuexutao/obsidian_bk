---
type: paper-note
status: done
domain: World-Models
paper: World Simulation with Video Foundation Models for Physical AI
year: 2025
arxiv: 2511.00062
doi: 10.48550/arXiv.2511.00062
source: https://arxiv.org/abs/2511.00062
project: https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/
code: https://github.com/nvidia-cosmos/cosmos-predict2.5
tags:
  - World-Models
  - Physical-AI
  - Video-Generation
  - Flow-Matching
  - Reinforcement-Learning
  - Model-Merging
created: 2026-10-24
updated: 2026-10-24
---

# Cosmos-Predict 2.5：视频基础模型驱动的物理世界模拟

## 基本信息

- **正式名称**：World Simulation with Video Foundation Models for Physical AI
- **机构 / 作者**：NVIDIA，署名 88 人（第一作者 Arslan Ali，通讯提交人 Yin Cui）
- **版本**：arXiv 2511.00062，v1 提交于 2025-10-28，v2 修订于 2026-02-24（正文 44 页）
- **发布模型**：Predict2.5 基座 2B / 14B（pre-trained、post-trained、distilled），领域特化 `auto/multiview`、`robot/multiview-agibot`、`robot/action-cond`、`robot/policy`、`robot/gr00tdream-gr1`，以及兄弟模型 Cosmos-Transfer2.5 系列
- **开源边界**：代码 Apache 2.0，权重 NVIDIA Open Model License；推理、后训练、蒸馏脚本与 Cosmos Cookbook 一并开放
- **当前状态**：官方仓库已挂出醒目告示，Cosmos 3 发布后本仓库 **不再积极开发，仅有限维护**，官方建议新项目迁移 Cosmos 3
- **重要性评估**：★★★★☆。它是 Cosmos 家族里工程完成度最高的一代"预测型"世界模型，把流匹配、物理 AI 推理型文本编码器、分域后训练 + 模型合并 + RL + 蒸馏这条工业流水线跑通并全部开源；但论文对 Predict1 的直接同基准对比偏少、14B 相对 2B 的自动指标几乎不涨，且很快被 Cosmos 3 收编，属于"重要但已被取代"的一代。

## 一句话结论

Cosmos-Predict2.5 用**一个**流匹配 DiT 把 Text2World / Image2World / Video2World 三种任务合并进同一套权重，以 Cosmos-Reason1 取代 T5 作为文本编码器，用"更狠的数据筛选（存活率 30% → 4%）+ 删绝对位置编码 + 分域 SFT 后模型合并 + GRPO 式 RL + 4 步蒸馏"五件事，把 2B 模型推到与 Wan2.2-27B-A14B 相当、人评更优的水平，并把这条底座扩展到 Transfer2.5（控制生成）、多视角驾驶、相机可控多视角机器人、动作条件预测与 VLA 合成数据五类下游。

---

## 1. 研究背景与问题定义

### 1.1 Physical AI 需要"环境的数字孪生"

论文延续 Cosmos 1.0 的问题设定：Physical AI 系统（机器人、自动驾驶车）直接在真实世界试错成本高、风险大、长尾事件难以覆盖，因此需要一个能根据智能体行为生成未来视觉状态的世界模拟器，作为安全代理。世界状态被具体化为 **RGB 视频**，控制条件 $c_t$ 则可以是文本、参考帧、相机轨迹、机器人动作或结构化场景地图。

### 1.2 Predict1 暴露的三个瓶颈

作者明确列出相对 Cosmos-Predict1 的三条改进，这三条也正好对应前代的三处短板：

1. **数据**：Predict1 的过滤管线偏松（约 30% 片段存活），预训练语料的物理合理性与多样性不足；
2. **架构**：Predict1 把 Text2World、Image2World、Video2World 拆成多个模型，部署与后训练都要维护多套权重；
3. **训练范式**：Predict1 用 EDM 参数化 + T5 文本编码器，缺少偏好对齐环节，指令跟随与世界演化的物理合理性都受限。

### 1.3 核心贡献

- 数据侧：处理 **3500 万小时**原始视频（Predict1 为 2000 万小时），经七阶段管线产出 60 亿片段，最终保留 **2 亿高质量 clip**，存活率压到 4%；另建机器人、自动驾驶、智能空间、人体动力学、物理五类域数据；
- 架构侧：单模型统一 T2W / I2W / V2W；从 EDM 换 **流匹配**；移除绝对位置编码只留 3D RoPE；视觉 tokenizer 换成 WAN2.1 VAE；文本编码器换成 **Cosmos-Reason1** 的多层激活拼接；
- 训练侧：分域 SFT → cooldown → **模型合并**（soup / TIES / DARE）→ **RL 后训练**（VideoAlign 奖励 + GRPO + 扩散损失正则）→ **rCM 4 步蒸馏**；
- 应用侧：Transfer2.5 控制生成、多视角驾驶世界模型、相机可控多视角机器人生成、动作条件世界模型、VLA 合成数据，共五条落地路径。

---

## 2. 任务定义与输入输出

### 2.1 三种模式，一套权重

| 模式 | 输入 | 条件帧数 | 用途 |
|---|---|---|---|
| Text2World | 仅文本 | 0 | 从零构想世界 |
| Image2World | 文本 + 单张参考图 | 1 | 把静态观测续演成动态世界 |
| Video2World | 文本 + 历史视频 | 1 或 5（训练时随机） | 世界状态的滚动外推 |

训练时把条件帧与噪声帧在 token 序列里拼接，并用 **mask token 二值标记**指明哪些输入是条件；去噪损失只施加在被指定的帧上，因此三种模式可以在同一目标下联合训练。推理时对 I2W / V2W 采用 **frame replacement**：生成序列的开头若干帧直接替换为条件帧，条件帧数量可按任务调节，既保证首帧忠实，也让视觉线索向后平滑传播。

### 2.2 输出规格

单次生成 **93 个像素帧**，对应 WAN2.1 VAE（时/高/宽压缩 $4\times8\times8$）下的 **24 个 latent 帧**，16 fps 下约 **5.8 秒**。这是理解全篇实验的基准单位：论文里所有"长视频"都是把 93 帧作为一个 chunk 自回归滚动得到的。

---

## 3. 核心方法

### 3.1 流匹配目标

给定数据样本 $x$、噪声 $\epsilon\sim\mathcal{N}(0,I)$ 与时间步 $t\in[0,1]$，插值 latent 与真值速度为

$$x_t=(1-t)x+t\epsilon$$

$$v_t=\epsilon-x$$

训练目标即预测速度场的 MSE：

$$\mathcal{L}(\theta)=\mathbb{E}_{x,\epsilon,c,t}\lVert u(x_t,t,c;\theta)-v_t\rVert^2$$

其中 $c$ 为条件信息（文本嵌入、参考帧等）。作者特别强调，FM 与 Predict1 用的 EDM 在前后向过程上数学等价，差别只在**去噪网络的参数化**：EDM 通过预条件系数让输入输出近似标准高斯，FM 则让网络直接预测轨迹速度。后者训练目标更直接、优化更平滑。

### 3.2 时间步采样的两处工程补丁

高分辨率内容相邻像素高度相关，噪声太小则模型学不到结构，因此对时间步做偏移 logit-normal 变换：

$$t_s=\frac{\beta t}{1+(\beta-1)t}$$

$\beta$ 随训练分辨率渐进增大（256p 时 $\beta=1$，720p 时 $\beta=5$）。即便如此仍出现帧间突变伪影，作者归因于高噪声区样本不足，于是**强制 5% 训练样本采样自噪声分布最高的 2% 区间**——这是一个很小但很关键的 trick，直接消掉了跳变伪影。

### 3.3 网络架构

![](assets/Cosmos-Predict%202.5%20-%20视频基础模型驱动的物理世界模拟/figures/02-unified-architecture.png)

> **图 2：Cosmos-Predict2.5 总体架构。** 左侧是 Cosmos-Reason1 文本编码器（同时保留图像/视频输入通道，本文暂未使用）；右侧是潜空间 DiT，每个 block 为 self-attention → cross-attention → FFN，均由 AdaLN 的 scale / shift / gate 按时间步调制。图的核心信息是"一个 DiT 同时吃三种条件形态"，而不是三个模型。  
> 来源：论文 Figure 2，p.10，https://arxiv.org/pdf/2511.00062

骨干沿用 Predict1 的 DiT，唯一结构性改动是**删掉绝对位置编码、只保留相对位置编码（3D RoPE）**。动机直接借自长上下文 LLM：绝对编码把分辨率与序列长度钉死在训练配置上，移除后才能在后训练阶段自由扩展到更高分辨率与更长序列。

视觉侧使用 **WAN2.1 VAE**（因果 VAE，$4\times8\times8$ 压缩），再叠 $1\times2\times2$ patchify。文本侧用 **Cosmos-Reason1**：不使用单层输出，而是把每个 token 在多个 block 上的激活**拼接后投影到 1024 维**，再经 cross-attention 注入去噪过程——这是"更丰富的文本 grounding"的具体实现。

| 配置 | 2B | 14B |
|---|---|---|
| 层数 | 32 | 36 |
| 模型维度 | 2,048 | 5,120 |
| FFN 隐层 | 8,192 | 20,480 |
| 注意力头数 / 头维度 | 16 / 128 | 40 / 128 |
| AdaLN-LoRA 维度 | 256 | 256 |
| 激活 / 位置编码 | GELU / 3D RoPE | GELU / 3D RoPE |

### 3.4 渐进式预训练

| 阶段 | 任务 | 分辨率 | 帧数 |
|---|---|---|---|
| 1 | Text2Image | 256p | 1 |
| 2 | T2I + Video2World | 256p | 1 / 93 |
| 3 | T2I + V2W | 480p | 1 / 93 |
| 4 | T2I + V2W | 720p | 1 / 93 |
| 5 | T2I + V2W + Text2World | 720p | 1 / 93 / 93 |

先让模型学会生成高质量单帧，再引入时序；分辨率按"收敛且画质平台化"逐档提升；最后才加入零条件帧的 T2W，并以 0.5 / 0.25 / 0.25 的概率采样 0 / 1 / 2 个条件帧。优化器 AdamW（$\beta_1=0.9,\beta_2=0.999$，weight decay 0.001），学习率 2B 为 $3\times10^{-5}$、14B 为 $1.3\times10^{-5}$，含 2000 步 warmup 的线性衰减。

### 3.5 后训练：分域 SFT → cooldown → 模型合并

SFT 阶段先用 InternVideo2 嵌入训练多头分类器，把样本划入**物体永存、高动态、复杂场景、驾驶、机器人操作**五域（对应 10.4M / 1.0M / 1.6M / 3.1M / 730K 条）。作者**不为五域训一个联合模型**，而是每域各训一个专精模型（30k iterations，batch 256），理由是免去混合比例调参、又能把每域数据吃干榨净；代价是通用域轻微退化，交给模型合并修复。

合并前还加了一道 **cooldown**：用精选 4K 高质视频把学习率线性衰减到零，提升细节与运动平滑度。随后比较 model soup / TIES / DARE-Linear / DARE-TIES 四种合并方式，扫参产出 20 多个候选，最终选 **model soup**。一个反直觉的经验是：**对超参做朴素网格搜索，稳定优于按各 SFT 模型胜率做启发式选择**。

### 3.6 RL 后训练

把条件视作状态、整条去噪轨迹视作动作，用 RL 做偏好对齐：

- 奖励模型 **VideoAlign**（VLM -based，评文本对齐 / 运动质量 / 视觉质量）；
- 每个条件采样 8 个输出（20 步），按 GRPO 在 rollout 组内归一化奖励得到 advantage；
- 显存受限，轨迹概率被拆解为逐步条件概率之和，每两个条件概率累积一次梯度，共十步构成一次参数更新；
- **用微调数据集上的扩散损失做正则**，抑制 reward hacking（详见 Ye et al. 2025）；
- 规模很小：**256 步、batch 32，且仅在 2B（pre-trained 与 merged）上做过**。

![](assets/Cosmos-Predict%202.5%20-%20视频基础模型驱动的物理世界模拟/figures/03-rl-human-vote.png)

> **图 5：RL 前后的人评胜率。** 无论起点是 pre-trained 还是 merged，RL 版本的胜率都高于 RL 前（40.0% / 46.7% vs 18.9% / 16.3%），说明偏好对齐带来的增益与模型合并并不重叠，是独立可叠加的一层。  
> 来源：论文 Figure 5，p.13，https://arxiv.org/pdf/2511.00062

### 3.7 蒸馏与基础设施

蒸馏采用 **rCM**（连续时间一致性蒸馏 + 分布匹配蒸馏的混合框架），配套实现带 JVP 的 fused flash attention、FSDP2 与上下文并行适配，最终 **4 步**采样即可接近 teacher（T2W 总分 0.764 vs 0.768，I2W 0.816 vs 0.810，蒸馏版在 I2W 上甚至反超）。

训练栈：**FSDP2 混合分片**（逐参数分片，比 FSDP1 的 bucket 策略更省显存）、**Ulysses 风格上下文并行**（取代 Predict1 的 ring attention，通信更高效、便于接入稀疏注意力与 JVP）、**选择性激活检查点**（优先重计算逐元素算子与归一化层）。4096 张 H100 下 MFU：2B 为 **36.49%**，14B 因上下文并行度更大降至 **33.08%**。

---

## 4. 数据

![](assets/Cosmos-Predict%202.5%20-%20视频基础模型驱动的物理世界模拟/figures/01-curation-pipeline.png)

> **图 1：七阶段视频策展流水线。** 镜头感知切分 → GPU 转码 → 裁剪 → 六重过滤 → 多粒度标注 → 语义去重 → 分片。它把 3500 万小时原始视频收敛为 2 亿可训练 clip，是全文最"工业"的一张图，也是 Predict2.5 相对前代提升的主要来源。  
> 来源：论文 Figure 1，p.5，https://arxiv.org/pdf/2511.00062

### 4.1 通用预训练数据

- **切分**：镜头边界检测去掉转场，短于 5 秒丢弃，得到 **60 亿**个 5–60 秒片段；
- **六重过滤**：审美分 → 运动 → OCR（去字幕）→ 感知质量（DOVER 类）→ 语义伪影（VTSS 类，如画中画、劣质转场）→ **VLM 高精度终筛**（放最后因为最贵）。存活率仅 **4%**（Predict1 为 30%），得 **2 亿 clip**；再剔除游戏、合成图案、动画、卡通等不符合物理世界分布的内容；
- **标注**：每 clip 切成 5 秒窗口，用 **Qwen2.5-VL-7B** 生成事实性 caption，提示词引导模型突出主体、运动与关键语义，并输出短/中/长三种长度；
- **去重**：嵌入聚类 + 簇内两两比对，同语义保留最高分辨率版本；采用在线增量策略支持数据持续增长；
- **分片**：自建 **26 类**视频类型分类器，按内容类型、分辨率、长宽比、长度分 shard，支持课程式训练与细粒度域平衡；底层接 Delta Lake lakehouse 与 Milvus 向量库。

### 4.2 五个专业域

| 域 | 规模与要点 |
|---|---|
| Robotics | AgiBot-Beta、Bridge、DROID、GR00T、1X、OpenX、RoboMIND 等；滤低分辨率与近静态片段，过慢动作加速播放；caption 强制按时间顺序描述动作，并注入各数据集自带元数据以降幻觉 |
| Autonomous Driving | 自建专有数据集约 **310 万** 条 20 秒环视 clip，7 路同步相机；按地域、车流密度、车速、加速度、操纵、道路类型、能见度、天气、光照做属性均衡采样；caption 覆盖交通参与者、环境因素、纵横向元动作、速度、状态转移、交互 |
| Smart Spaces | 仓库 / 工厂 / 工地等，关键词召回后由 VLM 判相关性，约 4 万条 |
| Human Dynamics | YOLOX 检测 + RTMPose 关键点；要求人物出现帧数 >40%、单帧不超过 8 人、单人占画面 ≥3% |
| Physics | 先定义经典力学/流体力学等可观测物理现象的 taxonomy，再据此定向采集，caption 强调物理过程与物体交互 |

---

## 5. 实验结果

### 5.1 PAI-Bench 主结果

| 模型 | T2W 域分 | T2W 质量 | T2W 总分 | I2W 域分 | I2W 质量 | I2W 总分 |
|---|---|---|---|---|---|---|
| Predict2.5-2B [pre-train] | 0.782 | 0.720 | 0.751 | 0.824 | 0.775 | 0.799 |
| Predict2.5-2B [post-train] | 0.804 | 0.732 | 0.768 | 0.840 | 0.779 | **0.810** |
| Predict2.5-14B [pre-train] | 0.791 | 0.722 | 0.757 | 0.835 | 0.777 | 0.806 |
| Predict2.5-14B [post-train] | 0.803 | 0.732 | 0.768 | 0.838 | 0.781 | **0.810** |
| Wan2.1-14B | 0.794 | 0.727 | 0.761 | 0.827 | 0.768 | 0.797 |
| Wan2.2-27B-A14B | 0.810 | 0.728 | 0.769 | 0.841 | 0.772 | 0.806 |

后训练把 2B 从 0.751 / 0.799 提到 0.768 / 0.810，收益明确；但 **14B 与 2B 的总分几乎完全相同**（T2W 0.768 vs 0.768，I2W 0.810 vs 0.810）。这说明 PAI-Bench 的自动指标在 0.76–0.81 这个区间已经饱和，区分度不足，论文也因此在人评上补了一刀。

### 5.2 人评才是尺寸收益的证据

- 2B post-train：对 Wan2.2-5B 胜率 30.0% vs 26.2%（模型小 60%）；对 Wan2.1-14B 33.0% vs 34.8%（基本持平，模型小 85.7%）；
- 14B post-train：对 Wan2.1-14B **48.6% vs 31.8%**；对 Wan2.2-27B-A14B 38.1% vs 35.9%（参数量只有一半）。

同一对照下，模型从 2B 放大到 14B，人评胜率从 33.0% 跳到 48.6%——**尺寸的收益只在人的主观判断里显现，自动指标看不出来**。这是本文最值得记住的一条实验观察。

### 5.3 Transfer2.5：3.5× 更小却全面更强

Transfer2.5-2B 相对 Transfer1-7B 的关键结构改动是：控制分支不再集中插在主干开头，而是**每 7 个 block 插 1 个、共 4 个**，让条件信息渐进注入。训练数据侧，深度条件用 Video Depth Anything 处理 1000 万视频（SAMv2 做 300 万分割，边缘/模糊沿用 Transfer1 管线共 1400 万），每个控制分支独立训 100k iterations。在 PAIBench-Transfer（600 视频）上：

| 配置 | Blur SSIM↑ | Edge F1↑ | Depth si-RMSE↓ | Mask mIoU↑ | Quality↑ |
|---|---|---|---|---|---|
| Transfer1-7B（各单分支） | 0.62–0.89 | 0.11–0.38 | 0.66–1.13 | 0.70–0.73 | 6.02–6.89 |
| Transfer2.5-2B（各单分支） | 0.68–0.90 | 0.14–0.49 | 0.59–1.02 | 0.71–0.75 | 8.73–9.75 |
| Transfer1-7B 均匀权重 | 0.82 | 0.26 | 0.70 | 0.74 | 9.24 |
| Transfer2.5-2B 均匀权重 | **0.87** | **0.41** | **0.67** | **0.76** | **9.31** |

画质分从 6.x 抬到 8.x–9.x 是最大幅度的提升，控制对齐（Edge F1 0.26→0.41）也明显改善。作者归因于两点：底座 Predict2.5 更强、训练数据更偏 Physical AI 域。

![](assets/Cosmos-Predict%202.5%20-%20视频基础模型驱动的物理世界模拟/figures/04-long-video-error.png)

> **图 10：长视频误差累积。** 纵轴为归一化的 RNDS（相对 DOVER 分数，起点固定为 1），横轴为自回归 chunk 序号。四种控制模态下 Transfer2.5-2B（绿）都比 Transfer1-7B（蓝）衰减更慢。这张图同时暴露了问题的存在：即便改进后，18 个 chunk 后 RNDS 仍普遍掉到 0.8 以下。  
> 来源：论文 Figure 10，p.21，https://arxiv.org/pdf/2511.00062

作者为此专门提出 **RNDS** 指标：把每个 chunk 的 DOVER 分数除以真值 DOVER，再用首个 chunk 归一化，从而把不同视频的退化曲线拉到同一起点比较。这个指标本身就说明：**分块自回归的误差累积是这个范式绕不开的结构性缺陷**。

### 5.4 机器人：合成数据增强策略鲁棒性

真机平台为双臂 7-DoF Kinova Gen3 + Robotiq 夹爪，头戴 RealSense D455，任务为"抓苹果入碗再放回"。基线 Diffusion Policy 用 100 条遥操作演示训练；增强组用 Transfer2.5-2B 对每条演示生成 5 个视觉变体（全局 edge 控制 + 仅机器人像素 blur 控制，CFG=3），只改图像不动动作标签。

| 策略 | 基础场景 | 9 个新场景合计 | 总计 |
|---|---|---|---|
| Base（仅 100 条原始演示） | 1/3 | 0/27 | 1/30 |
| Baseline（标准图像增强） | 3/3 | 2/27 | 5/30 |
| Proposed（Transfer2.5 增强） | 3/3 | 21/27 | **24/30** |

结论很有说服力：颜色/光照类变化（在增强分布内）几乎全通，而纯 OOD 变化（换深色柜子、开抽屉、三者叠加）也能拿到部分成功。但必须注意规模——**每场景仅 3 次试验、无统计检验**，且失败案例（抓取精度不足）只被一句带过。

### 5.5 驾驶多视角与相机可控多视角

多视角实现上有个巧思：**把 7 个视角沿 latent 时间维拼接**，等效为序列帧；latent 时间维压到 8 以塞进显存；每视角在 latent 通道维拼一个长度 7 的可学习嵌入，RoPE **逐视角独立构造**。在 RDS-HQ-HL 上，Predict2.5-2B/auto/mv 相对 Predict1-7B-Sample-AV：FVD-StyleGAN 63.685 → 23.060、FVD-I3D 69.613 → 25.308、FID 25.341 → 12.095（约 2.3× 提升），跨视角 Sampson 误差基本持平（TSE 0.948 vs 0.930）。用 BEVFormer / LATR 在生成视频上做检测，Transfer2.5 相对 Transfer1 的 3D cuboid LET-AP 从 0.243 提到 0.394（真值参考 0.476），车道 F1 0.604 → 0.637（真值 0.637）。

机器人相机可控多视角用 **Plücker raymap** 表达目标相机：像素映射为 6D ray 表示后 patchify，经 camera projection layer 对齐维度，再**加到 video token 上**送进 self-attention；训练时冻结除 self-attention 与 camera projection 之外的所有层。多视角联合生成的跨视角 Sampson 误差 26.61 px → 19.73 px，代价是相机轨迹精度基本不变（TransErr 0.08，RotErr 0.19 → 0.20，略升）。

### 5.6 VLA 合成数据与动作条件

14B 在 GR1 人形数据上后训练得到 `robot/gr00tdream-gr1`，DreamGen 指令跟随（GPT-4o 评分）在物体 / 行为 / 环境三维上为 91.8 / 70.2 / 69.0，较未使用 Reason1 的 Predict2 版本（90.0 / 59.6 / 69.0）与 WAN2.1（72.0 / 72.3 / 48.3）领先，尤其物体与环境泛化。

动作条件模型把 7 维动作（$\Delta x,\Delta y,\Delta z,\Delta\theta_r,\Delta\theta_p,\Delta\theta_y$, 夹爪宽度）经 MLP 映射后**加到 DiT 的时间步嵌入上**。Bridge 数据集上 PSNR 24.95（Predict1 动作条件版 21.14）、Latent L2 0.28（0.32）、FVD 146（190）。消融显示注入方式排序为 **TimeEmbedding > CrossAtten(24.41) > ChannelConcat(23.11)**——把动作当"时间条件"而非"空间条件"是更优选择。

---

## 6. 与 Cosmos 的联系和区别

这一节回答"Predict 2.5 到底和 Cosmos 是什么关系"。

### 6.1 Cosmos 不是模型，是平台 + 三族模型

Cosmos 是 NVIDIA 面向 Physical AI 的**世界基础模型平台**（Cosmos World Foundation Model Platform，arXiv 2501.03575），其官方定义包含三类可后训练的 WFM：

| 族 | 职责 | 代表模型 |
|---|---|---|
| **Predict** | 预测未来世界状态（生成式） | Cosmos-Predict1 → **Predict2**（2B/14B）→ **Predict2.5** |
| **Transfer** | 在结构控制下做世界到世界的转换（Sim2Real / Real2Real） | Cosmos-Transfer1-7B → **Transfer2.5-2B** |
| **Reason** | 物理常识与具身推理（VLM，做评判与决策） | Cosmos-Reason1 → Reason2 |

所以 **Predict2.5 不是 Cosmos 的竞品或替代，而是 Cosmos 家族里负责"预测"的那一条产品线的 2.5 代**；它和 Transfer2.5 共享底座，又反过来把 Reason1 当文本编码器——三者构成"Reason 提供语义与评判 → Predict 生成未来 → Transfer 施加结构控制"的闭环。

### 6.2 相对 Cosmos 1.0 平台的六处实质变化

| 维度 | Cosmos 1.0 / Predict1 | Predict2.5 |
|---|---|---|
| 生成范式 | 扩散（EDM 参数化）与自回归两条腿走路 | **只保留流匹配扩散**，不再探索 AR 路线 |
| 任务组织 | T2W / I2W / V2W 多套权重 | **单模型统一三种模式**（mask token + frame replacement） |
| Tokenizer | 自研 Cosmos Tokenizer（连续 CV / 离散 DV） | **WAN2.1 VAE**（$4\times8\times8$）+ $1\times2\times2$ patchify |
| 文本编码 | T5 | **Cosmos-Reason1 多层激活拼接 → 1024 维**（decoder-only VLM） |
| 位置编码 | 绝对 + 相对 | **仅 3D RoPE**（移除绝对编码以利分辨率/长度外推） |
| 后训练 | 领域 SFT 为主 | **分域 SFT + cooldown + 模型合并 + GRPO 式 RL + rCM 蒸馏** |
| 数据 | 2000 万小时，存活率 30% | 3500 万小时，存活率 **4%**，2 亿 clip + 五类域数据 |

一句话概括区别：**Cosmos 1.0 是"定义平台"的论文（我要做什么、有哪些件），Predict2.5 是"把平台里最好用的那件打磨成产品"的论文（怎么把预测这一件做到 SOTA 并交付）**。前者强调覆盖面（两种生成范式、两个 tokenizer、guardrail），后者专注把一条路线推到极致并砍掉分叉。

### 6.3 与 Transfer2.5 的具体关系

Transfer2.5-2B 就是在 Predict2.5-2B 上加控制分支得到的兄弟模型：主干权重直接继承，控制块从"集中插在开头"改为"每 7 块插一个"，参数量比 Transfer1-7B 小 3.5×，却在画质与对齐上全面反超。论文的机器人策略增强、驾驶多视角两个应用，实际上跑的都是 Transfer2.5 而非 Predict2.5——**Predict 负责"世界会怎么变"，Transfer 负责"在指定结构下把变化画出来"**。

### 6.4 与 Cosmos 3：已被收编

2026 年 Cosmos 3 发布后，官方在 Predict2.5 仓库顶部挂出告示：Cosmos 3 用**单一统一架构**同时具备推理、未来状态预测、跨域跨模态转换与动作/策略生成能力——也就是把 Predict / Transfer / Reason 三条线合并。本仓库不再积极开发、仅有限维护。**因此 Predict2.5 的推荐定位是"可读、可复现的一代完整工程范本"，而不是新项目的默认起点。**

---

## 7. 局限与批判性评价

- **世界模型仍是像素级的，没有显式状态**。论文在 Related Work 里承认自己属于"像素空间视频预测"范式，与潜空间预测（JEPA 系）和原生 3D/4D 表示并列；它不做物理求解，物理合理性完全来自数据分布，长尾交互（接触、形变、遮挡后重现）依然不可靠。
- **长程一致性靠自回归硬扛**。"30 秒视频"是 93 帧 chunk 滚动出来的，作者自己提出的 RNDS 曲线显示 18 个 chunk 后质量仍掉到 0.8 以下；没有持久的 3D 记忆或显式几何缓存——这一点明显弱于同期做显式 3D/4D 记忆的世界模型路线。
- **14B 的自动指标收益近乎为零**。T2W / I2W 总分与 2B 完全相同，只有人评拉开差距。这既说明收益真实存在，也说明 **PAI-Bench 已经接近饱和**，论文的量化证据强度弱于其 qualitative 结论。
- **RL 规模极小且只覆盖 2B**：256 步、batch 32、仅 2B 模型，14B 的最终 checkpoint 并未经过 RL。RL 的增益能否随规模保持，论文没有回答。
- **缺少与 Predict1 的同表全面对比**。全文大量"相对 Predict1 显著提升"的表述，但缺一张同基准、同尺寸的系统对照表；可查的直接对比只有 Tab.16（相机控制）与 Tab.18（DreamGen，Predict2 vs 2.5）。
- **真机实验统计力度不足**：10 个场景 × 3 次试验，无显著性检验；失败模式（抓取精度）只有定性描述。
- **数据不可完全复现**：自动驾驶 310 万条环视 clip、Smart Spaces、Physics 子集均为 NVIDIA 专有或未公开，200M clip 的筛选模型细节也只给"某类 VLM"，外部复现只能拿到结果而非配方。
- **多视角相机精度未改善**：多视角联合生成换来跨视角一致性（Sampson 19.73 px），相机轨迹精度却略降（RotErr 0.19 → 0.20），说明一致性收益部分来自"相互妥协"而非各自更准。

---

## 8. 复现与实践建议

- **代码与权重**：Apache 2.0 + NVIDIA Open Model License，推理 / LoRA 后训练 / 蒸馏 / Diffusers 支持齐备，另有 Cosmos Cookbook 提供分领域 recipe；2B 提供 distilled（4 步）与 robot/policy（Libero、RoboCasa）等即用检查点，工程可用性在同类工作里属于第一梯队。
- **算力**：论文只披露 4096×H100 下的 MFU（36.49% / 33.08%），**未给完整 GPU-hours**，无法估算复现成本；可确定的是预训练为数千卡级别，个人研究者不应尝试从零复现，合理路径是在 2B 基座上做 LoRA 后训练。
- **选型建议**（2026 年的现实判断）：
  - 需要**最强预测能力** → 直接用 Cosmos 3；
  - 需要**可控、可解释、能改的工程底座**（比如要接自己的机器人数据做 action-cond 后训练）→ Predict2.5-2B 仍是很好的起点，其 action 注入用 TimeEmbedding 的消融结论可直接复用；
  - 需要**结构控制生成**（深度/分割/边缘条件）→ Transfer2.5-2B，3.5× 更小却有更高画质；
  - 需要**长程 3D 一致的世界** → 这条路本身不对，应转看显式 3D/4D 表示的世界模型。
- **值得直接抄的三个工程 trick**：5% 样本强制采自最高 2% 噪声区（消跳变伪影）；分域 SFT 后做 model soup 而非联合训练；模型合并超参用朴素网格搜索胜过按胜率启发式挑。

---

## 9. 个人启发与后续问题

- **"三模型合一"与"极简胜复杂"在同一年出现两次**。Predict2.5 把 T2W/I2W/V2W 合并进一套权重，DA3 把多任务砍成单目标 depth-ray——表面相反（一个合并、一个削减），实质是同一个判断：**能被统一参数化表达的东西，不应拆成独立模块**。Predict2.5 的"合并"是合并任务，DA3 的"削减"是削减冗余目标，二者都在压缩接口数量。
- **Plücker raymap 与单 token 内参的对照很有意思**。Predict2.5 在可控多视角里用稠密 6D Plücker raymap 注入相机位姿（加到 video token 上），而 WorldMirror 2.0 的消融却证明"内参用 4 维向量压成 1 个 token"优于稠密 raymap。二者结论看似冲突，实则条件不同：前者要表达**逐像素变化的相机轨迹**（同一时刻不同像素光线不同，本质是稠密量），后者表达的是**全局解析量**。判断标准应当是"这个几何量的本质维度是多少"——这跟我关注各向异性/椭球表达的直觉是一致的：表示维度应当匹配被表达对象的内在自由度，过参数化只会引入对齐负担。
- **一个待验证的问题**：前馈式生成的视频世界模型（Predict2.5 这类）能否当作 3D 重建的数据源？它有相机可控多视角输出，理论上可以用 WorldMirror / DA3 反过来重建它的输出，检验"生成的世界是否三维自洽"。已有 Orbia 这类 benchmark 在做类似评测，但把"生成 → 重建 → 再比较"做成闭环评测的工作似乎还不多。
- **RL 与世界模型的结合点**：Predict2.5 把整条去噪轨迹当 action 做 GRPO，这一思路对"可微分渲染 + 3D 表示"的世界模型同样成立——如果世界状态是 3DGS，去噪轨迹就变成高斯参数的演化轨迹，奖励可以用渲染后的物理合理性来定义。这条线值得单独追。

---

## 10. 材料来源

本地材料见同目录 `assets/Cosmos-Predict 2.5 - 视频基础模型驱动的物理世界模拟/`：

- `sources/paper.pdf`：官方论文 PDF（arXiv 2511.00062v2，44 页），全文精读主来源；
- `figures/01-curation-pipeline.png`：Figure 1（p.5）视频策展流水线；
- `figures/02-unified-architecture.png`：Figure 2（p.10）统一架构；
- `figures/03-rl-human-vote.png`：Figure 5（p.13）RL 前后人评；
- `figures/04-long-video-error.png`：Figure 10（p.21）长视频 RNDS 误差累积；
- `sources.md`：逐项来源、获取日期与核验说明。

在线来源：arXiv https://arxiv.org/abs/2511.00062 ｜ 代码 https://github.com/nvidia-cosmos/cosmos-predict2.5 ｜ Transfer2.5 https://github.com/nvidia-cosmos/cosmos-transfer2.5 ｜ 项目页 https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/
