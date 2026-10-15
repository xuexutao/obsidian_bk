---
type: paper-note
status: complete
domain: World-Models
paper: Advancing Open-source World Models
year: 2026
arxiv: 2601.20540
doi: 10.48550/arXiv.2601.20540
source: https://arxiv.org/abs/2601.20540
project: https://technology.robbyant.com/lingbot-world
code: https://github.com/robbyant/lingbot-world
tags: [World-Models, LingBot-World, Video-Diffusion, Action-Control, Long-Term-Memory, Diffusion-Forcing, Self-Forcing, DMD, Real-Time-Generation]
created: 2026-09-04
updated: 2026-09-04
---

# LingbotWorld：动作可控交互世界模型

## 基本信息

- **论文正式名**：*Advancing Open-source World Models*
- **模型**：LingBot-World（Robbyant Team）
- **作者**：Zelin Gao、Qiuyu Wang、Yanhong Zeng、Jiapeng Zhu、Ka Leong Cheng、Yixuan Li、Hanlin Wang、Yinghao Xu、Shuailei Ma、Yihang Chen、Jie Liu、Yansong Cheng、Yao Yao、Jiayi Zhu、Yihao Meng、Kecheng Zheng、Qingyan Bai、Jingye Chen、Zehong Shen、Yue Yu、Xing Zhu、Yujun Shen、Hao Ouyang
- **提交时间**：2026-01-28，arXiv v1
- **论文**：<https://arxiv.org/abs/2601.20540>
- **项目页**：<https://technology.robbyant.com/lingbot-world>
- **官方代码**：<https://github.com/robbyant/lingbot-world>（Apache-2.0；当前 README 已提示 v1 停止积极维护，后续转向 LingBot-World-Infinity/v2）
- **本地 PDF**：[paper.pdf](assets/LingbotWorld%20-%20动作可控交互世界模型/sources/paper.pdf)

## 一句话结论

LingBot-World 以 Wan2.2-14B 视频扩散模型为起点，通过**三阶段训练**把开放域视频生成器演化为动作可控、分钟级记忆、可实时流式运行的世界模型：Middle-Training 用 28B 双专家 MoE、5→60 秒课程与 Plücker/AdaLN 动作注入学会长程动态，Post-Training 再用 Diffusion Forcing、block-causal attention、Self-Forcing、自 rollout DMD 与对抗优化蒸馏为 Fast 模型；其 480p 实现达到 **16 FPS / 首段延迟低于 1 秒**，但仍依赖企业级多 GPU，且显式消融和训练数据规模披露不足。

---

## 1. 研究背景与问题定义

### 1.1 从 video “dreamer” 到交互 simulator

高质量视频生成模型善于合成短片，却通常只学到像素相关性：用户不能决定下一步动作，物体离开视野后可能消失，长时间 rollout 会累积漂移，多步 diffusion 也难以在线响应。LingBot-World 把差距归结为三项瓶颈：

1. **动作—环境响应数据稀缺**：互联网视频很少同时带精确按键、camera pose 和结果画面。
2. **长时记忆困难**：标准 diffusion 只训练秒级片段，分钟级生成会“灾难性遗忘”。
3. **采样速度不足**：双向 attention 和多步去噪适合离线生成，不适合闭环控制。

其路线不是一步到位训练一个 simulator，而是先继承大视频模型的视觉先验，再用专用数据注入世界知识，最后改变 attention 与训练分布来获得因果实时性。

![](assets/LingbotWorld%20-%20动作可控交互世界模型/figures/01-teaser.png)

> **图 1（论文 Figure 1）**：模型覆盖写实景观、科学/太空内容、卡通和艺术风格，并以 W/A/S/D 展示导航可控性。Teaser 证明的是开放域视觉与控制覆盖面，不等于每个场景都满足严格物理规律。  
> 来源：论文 Figure 1，第 1 页，<https://arxiv.org/abs/2601.20540>

### 1.2 主要贡献

1. **可扩展数据引擎**：统一一般视频、游戏动作数据和 Unreal Engine 精确渲染数据，并用三级 caption 解耦静态场景、相机运动与时序事件。
2. **三阶段演化训练**：Pre-Training 建视觉先验，Middle-Training 注入长期动态/动作，Post-Training 改为因果 few-step 模型。
3. **28B MoE 长视频基础模型**：高噪声 expert 负责全局结构与动态，低噪声 expert 负责细节；每步只激活一个 14B expert。
4. **动作控制**：连续 camera rotation 的 Plücker embedding 与 W/A/S/D multi-hot 联合，经 AdaLN 注入 DiT。
5. **实时后训练**：Diffusion Forcing + block causal attention + KV cache，再以 Self-Forcing、DMD 与 GAN 监督缩小长 rollout 的 train-test gap。
6. **开源与应用**：发布代码/权重，展示 promptable events、Action Agent 和生成视频的 3D reconstruction。

---

## 2. 任务定义与输入输出

### 2.1 条件世界状态转移

令视频序列与动作序列分别为：

$$
\mathcal V=\{x_1,x_2,\ldots,x_T\},\qquad
x_t\in\mathbb R^{H\times W\times C},
$$

$$
\mathcal A=\{a_1,a_2,\ldots,a_T\}.
$$

模型学习未来视觉状态的条件分布：

$$
\max_\theta\;\mathbb E\left[
\log p_\theta\left(x_{t:t+L}\mid x_{<t},a_{t:t+L}\right)
\right],
\tag{1}
$$

其中 $x_{<t}$ 是历史帧，$a_{t:t+L}$ 是预测区间对应动作，$L\ge1$ 是 horizon。把 $t=0$ 视为双向视频生成，可先学习整段全局关系；把 $t\ge0$ 且仅允许看 $x_{<t}$，则转为在线因果 world transition。

### 2.2 实际输入与输出

- **输入状态**：单张图像或已有视频前缀；文本描述；noisy video latents。
- **动作条件**：camera pose/rotation 与离散键鼠控制，可选无动作生成。
- **输出**：480p 或 720p 视频未来片段；Fast 版本按 chunk 流式输出。
- **记忆表示**：不是独立数据库，而是长上下文 video token 与 KV cache 中**涌现的隐式记忆**。
- **假设**：历史视觉上下文足以表示静态几何与动态对象状态；论文没有显式 object state、scene graph 或物理引擎状态。

---

## 3. 数据引擎

### 3.1 三类数据来源

1. **General Video Curator**：内部与开源视频，重点覆盖第一/第三人称，人、动物、车辆，以及 walking、cycling、transit 等运动方式。
2. **Game Data Acquisition**：同步采 RGB、W/A/S/D、camera movement 和高精度 timestamp，去除 UI overlay。采集策略包括 free navigation、闭环/多点往返、室内外 transition、绕 landmark sightseeing、固定位置 360°、后退，以及拾取/开门/战斗/破坏等交互。
3. **Synthetic Rendering（UE）**：随机语义位置与朝向，程序化矩形/360°/多点回看轨迹，或导入真实设备轨迹；逐条 collision detection，并输出精确 camera intrinsics/extrinsics。

合成数据的核心作用不是替代真实外观，而是补足现实视频中稀少的**回看、闭环、反向移动和精确 pose**，这些正是训练长期空间记忆所需分布。

### 3.2 Profiling 与三级 Caption

- 基础过滤：时长、分辨率、文件大小；Koala/TransNet v2 做语义连续切片。
- 语义分析：内部 VLM 评 brightness、sharpness、motion magnitude、scene type、第一/第三人称。
- 几何伪标注：缺 pose 的视频用 MegaSAM 生成 camera pose。
- **Narrative caption**：场景、相机轨迹、时间演化与故事整体描述。
- **Scene-static caption**：只写环境与美学，刻意删除相机/人物动作，使运动控制与静态生成解耦。
- **Dense temporal caption**：以 start/end time + event + caption 描述局部事件，为时间对齐监督服务。

重要缺口是：论文没有公布总视频数、总小时数、来源比例、去重规则和各阶段数据配方，因此无法严格重建训练分布。

---

## 4. 三阶段训练总览

![](assets/LingbotWorld%20-%20动作可控交互世界模型/figures/02-training-pipeline.png)

> **图 4（论文 Figure 4）**：Stage I 建立开放域高保真视频先验；Stage II 注入动作、长程一致性和领域规则，得到双向 physical world model；Stage III 通过 causal attention 与 latent few-step distillation 得到实时模型。  
> 来源：论文 Figure 4，第 8 页，<https://arxiv.org/abs/2601.20540>

### 4.1 Stage I：Pre-Training

以 **Wan2.2 image-to-video 14B diffusion model** 初始化。该阶段主要继承 visual realism、object permanence、temporal dynamics 和开放域语义，并非从 LingBot 专用交互数据重新训练一个视频 foundation model。论文未披露本阶段是否额外预训练、训练轮数或算力。

### 4.2 Stage II：Middle-Training

把基础视频生成器训练成双向世界模型，分三步：

1. Fundamental World Model：MoE + 5→60 秒 progressive curriculum + I2V/V2V multi-task，学习长期一致性与空间记忆。
2. Action-Conditioned Fine-tuning：冻结主 DiT，只训练动作 adapter / projection / AdaLN。
3. Parallelism：FSDP2 切参数/梯度/优化器状态，Ulysses 沿长序列切 context。

### 4.3 Stage III：Post-Training

先把双向模型变为 block-causal AR，再用 few-step distillation 和长程 self-rollout 恢复质量与稳定性。论文把这一阶段再分为：

- Causal Architecture Adaptation：Diffusion Forcing、混合 timestep、clean-frame supervision、KV cache。
- Few-Step Distillation：Self-Forcing/self-rollout、DMD、fake score two-time-scale updates、对抗监督。

---

## 5. 核心方法详解

### 5.1 28B 双专家 MoE 基础世界模型

模型继承 Wan2.2 的两 expert 设计：

| Expert | 负责的 diffusion 区间 | 功能 |
|---|---|---|
| High-noise expert | 去噪早期，高噪声 | 全局结构、粗布局、大尺度动态 |
| Low-noise expert | 去噪后期，低噪声 | 高频纹理、细粒度空间与时间细节 |

每个 expert 约 **14B**，总参数 **28B**，但一个 denoising timestep 只激活一个 expert，因此论文称单步计算和显存接近 dense 14B。这里“近似 14B 成本”只指单次激活，不包含存放/切换全部权重、长视频 activation 或多步 sampling 的系统成本。

#### 5.1.1 Progressive curriculum

训练视频长度从 **5 秒逐步扩到 60 秒**：短片先扩大基础模型原本较窄的生成分布，长片再迫使模型保持环境与对象。随着长度增加，训练提高 **flow shift**，让采样更重视负责全局结构的 high-noise timesteps。论文未给 flow shift 数值。

#### 5.1.2 I2V + V2V 多任务

- Image-to-video：从单帧推断未来动态。
- Video continuation：从历史视频外推运动。

两者联合使模型学习可从不同初始状态启动的统一转移函数，而不是只会“首图动画化”。

### 5.2 动作表示与注入

![](assets/LingbotWorld%20-%20动作可控交互世界模型/figures/03-method-pipeline.png)

> **图 5（论文 Figure 5）**：图像/视频前缀、噪声 latent 和动作进入 MoE DiT；视频 token 先 self-attention，动作经 Plücker Encoder 变成 AdaLN scale/shift，文本通过 cross-attention 注入。  
> 来源：论文 Figure 5，第 9 页，<https://arxiv.org/abs/2601.20540>

动作由两部分拼接：

- **连续 camera rotation / pose**：编码为 Plücker embedding，以统一的射线几何表示视角变化。
- **离散动作**：W/A/S/D 等编码为 multi-hot vector，表达逻辑移动。

融合 embedding 投影后生成 AdaLN 的 scale/shift，调制 DiT normalized features。与直接把动作当文本 token 相比，AdaLN 在每个 block 中持续影响去噪轨迹，更容易形成强控制；与全量微调相比，只训练 action adapter、projection 和 AdaLN 能降低 action 数据稀缺造成的 catastrophic forgetting。

但论文的表述存在一个边界：主文主要描述 continuous camera **rotation** + 离散键盘动作，代码侧 Base (Cam) 则要求每帧 intrinsics `[fx,fy,cx,cy]` 与 OpenCV `4×4 poses`。工程实现的控制信号比正文一句“rotation”更完整，复现时应以 checkpoint 和脚本要求为准。

### 5.3 隐式长期记忆

LingBot-World 没有 HYWorld 式显式检索/重构模块。它通过三项设计让记忆从 context 中涌现：

1. **分钟级长视频训练**迫使模型跨长区间维持 landmark、布局和对象。
2. **回看/闭环/反向轨迹数据**让“离开视野后再出现”成为训练常态，而非 out-of-distribution。
3. **长上下文 attention + rolling KV cache**保存历史 token；Post-Training 的 self-rollout 又让模型适应由自身历史形成的 context。

优点是无需维护显式 3D map，能自然覆盖流水、车辆、行人等动态场景；缺点是 memory 不可查询、不可纠错，稳定性依赖 context window 与生成质量。论文在局限中明确承认：它是 emergent memory，不是 explicit storage。

### 5.4 Causal adaptation：Diffusion Forcing

#### 5.4.1 Block-causal attention

Post-Training 的 student 从 **high-noise expert** 初始化，因为作者观察它的动作动态优于 low-noise expert；该比较没有数值表。attention 改为：

- chunk 内双向，保留局部运动与细节；
- chunk 间严格 causal，只看当前/过去；
- 推理复用过去 chunk 的 K/V，仅计算新 token。

训练序列含 $N$ 帧、分成 $L$ 个 chunks，每个 chunk 独立分配 noise timestep，这就是 Diffusion Forcing。为降低训练开销，只选少量目标 timesteps $\{t_1,\ldots,t_m\}$，它们也作为后续 distillation steps。由于 student 来自只见过高噪声的 expert，训练额外加入 $t=0$ 的 clean supervision。

其目标为：

$$
\mathcal L
=
\mathbb E_{x^i\sim p(x),\;t\in\{t_1,\ldots,t_m\}}
\left\|G_\theta(x_t^i,t,a)-x_0^i\right\|_2^2,
\tag{2}
$$

$G_\theta$ 是 causal student，$x_t^i$ 是加噪 latent，$x_0^i$ 是 clean target，$a$ 是动作。论文未披露 $N,L,m$ 及最终 few-step 的具体步数。

![](assets/LingbotWorld%20-%20动作可控交互世界模型/figures/04-causal-post-training.png)

> **图 6（论文 Figure 6）**：左侧 high-noise expert 经 block-causal attention 适配为 streaming generator；右侧在 fake score network feature 上附加 cross-attention discriminator head，给长程 DMD 加真实数据监督。  
> 来源：论文 Figure 6，第 12 页，<https://arxiv.org/abs/2601.20540>

### 5.5 Self-Forcing：让模型在自己的错误历史上训练

Teacher forcing 用真实历史训练，但推理时 student 条件于自己生成的帧；早期小误差进入 history 后会不断放大。LingBot-World 采用 Self-Forcing / self-rollout：

- student 生成的 chunk 写入 rolling KV cache，后续直接条件于这些输出；
- 模型必须学会从自己的 artifact 和漂移中恢复；
- forward 保留完整长上下文，但只对最近 $K$ 个生成步骤回传梯度，用 stochastic gradient truncation 控制成本。

论文未给 $K$。该策略缩小 train-test gap，却不能从根本上保证状态正确：若 rollout 进入训练未覆盖的错误模式，模型仍可能自我强化。

### 5.6 DMD：真实/伪 score 差驱动 few-step student

使用三套网络角色：

- **real score network**：冻结的 middle-trained MoE teacher；
- **fake score network**：同一 teacher 初始化，持续拟合 student 当前输出分布；
- **student generator**：few-step causal model。

Student 分布到数据分布的 KL 梯度为：

$$
\nabla_\theta\mathbb E_t
\left[D_{\mathrm{KL}}(p_{\theta,t}\|p_{\mathrm{data},t})\right]
=-\mathbb E\left[
\left(s_{\mathrm{real}}(\hat x_t,t,a)-s_{\mathrm{fake}}(\hat x_t,t,a)\right)
\frac{\partial\hat x}{\partial\theta}
\right],
\tag{3}
$$

其中 $\tilde x\sim p_\theta(\tilde x\mid a)$ 是 student clean sample，$\hat x_t$ 是对它做 forward diffusion 后的 noisy sample。可训练目标写成：

$$
\mathcal L_{\mathrm{DMD}}(\theta)=
\mathbb E\left[
\frac12\left\|
\hat x-
\operatorname{sg}\left[
\hat x-\big(\mu_{\mathrm{real}}(\hat x_t,t,a)-\mu_{\mathrm{fake}}^\phi(\hat x_t,t,a)\big)
\right]
\right\|_2^2
\right].
\tag{4}
$$

$\operatorname{sg}$ 是 stop-gradient。Fake score 在 student 视频上用标准 diffusion loss 更新，real score 固定；每更新一次 student，fake score 更新多次，使其及时追踪 student 分布，即 two-time-scale rule。

### 5.7 为什么还要对抗优化

DMD 后 student 仍落后 teacher，原因包括：只继承 high-noise expert、丢失 low-noise 高频细节；causal mask 限制建模；few-step 减少优化机会；teacher/student 都没有直接由真实数据纠错。于是作者在 fake score feature 上加 discriminator $D$：

$$
\mathcal L_G=
\mathbb E_{p(\tilde x)}
\left[f\left(1-D(\mu_{\mathrm{fake}}(\tilde x_t,t,a))\right)\right],
\tag{5}
$$

$$
\mathcal L_D=
\mathbb E_{p(x)}\left[f\left(D(\mu_{\mathrm{fake}}(x_t,t,a))\right)\right]
-
\mathbb E_{p(\tilde x)}
\left[f\left(1-D(\mu_{\mathrm{fake}}(\tilde x_t,t,a))\right)\right],
\tag{6}
$$

$f$ 为 softplus。对抗 loss **只更新 discriminator head**，fake score 仍只由 DMD loss 更新；作者未用 R1/R2，称当前 DMD 足够稳定。论文声称视觉质量显著改善，但没有给去掉 GAN head 的数值消融。

### 5.8 Diffusion Forcing、Self-Forcing 与 Context Forcing 的区别

- 本文明确使用 **Diffusion Forcing**：不同 chunk 独立 noise timestep，用于因果适配。
- 本文明确使用 **Self-Forcing**：训练时让 student 条件于自己的 rollout，减少 exposure bias。
- 本文**不使用 HYWorld 1.5 命名的 Context Forcing**，因为它没有显式检索后需对齐的 teacher/student memory context。其“memory”主要在长 context/KV 中涌现。两篇工作都做 self-rollout + DMD，但 HYWorld 额外解决检索上下文错配。

---

## 6. 实验设置与结果

### 6.1 定性能力

论文分别展示 LingBot-World-Base（middle-trained 双向模型）与 LingBot-World-Fast（post-trained causal 模型）。Base 覆盖真实、风格化、科学和游戏环境；Fast 在 480p 下由 **one GPU node** 达到 **16 FPS**。这里的 node 未说明 GPU 型号和卡数，官方推理脚本使用 `torchrun --nproc_per_node=8`、FSDP 与 Ulysses，因此不能表述为“单 GPU 16 FPS”。摘要另称生成 16 FPS 时 latency **<1 second**。

作者称 Fast 的视觉退化“perceptually marginal”、无明显 mode collapse，但未提供 Base vs Fast 同指标对比或用户研究。

### 6.2 长期记忆与生成边界

![](assets/LingbotWorld%20-%20动作可控交互世界模型/figures/05-emergent-memory.png)

> **图 12（论文 Figure 12）**：前三行中 landmark 离开视野最多 60 秒后仍保持结构；后两行展示未观察状态继续演化——相机前进后桥更近，车辆离开视野后继续行驶并在合理位置重现。它比单纯首尾图相似更强，但仍是少量定性样例。  
> 来源：论文 Figure 12，第 19 页，<https://arxiv.org/abs/2601.20540>

Figure 13 进一步展示一个最长 **10 分钟**的稳定场景。需要谨慎解释：作者在正文中称该样例没有显著画质/叙事退化，但局限章节同时承认长视频仍会 drift，连贯长度不足以支持 extended gameplay。因此“能生成 10 分钟样例”不是“任意场景稳定 10 分钟”的统计保证。

### 6.3 定量协议

- Benchmark：VBench。
- 测试集：**100 个生成视频**。
- 每个视频：**超过 30 秒**。
- Baseline：Yume-1.5、HY-World 1.5。
- 指标：Imaging Quality、Aesthetic Quality、Dynamic Degree、Motion Smooth、Temporal Flickering、Overall Consistency。

![](assets/LingbotWorld%20-%20动作可控交互世界模型/figures/06-quantitative-results.png)

> **表 2（论文 Table 2）**：LingBot-World 在画质、美学、动态程度和总体一致性上最高；HY-World 1.5 在 Motion Smooth 与 Temporal Flickering 略优。  
> 来源：论文 Table 2，第 21 页，<https://arxiv.org/abs/2601.20540>

| Model | Imaging Quality | Aesthetic Quality | Dynamic Degree | Motion Smooth | Temporal Flickering | Overall Consistency |
|---|---:|---:|---:|---:|---:|---:|
| Yume-1.5 | 0.5838 | 0.5185 | 0.7612 | 0.9709 | 0.9545 | 0.1994 |
| HY-World 1.5 | 0.6512 | 0.5487 | 0.7217 | **0.9897** | **0.9773** | 0.2016 |
| **LingBot-World** | **0.6683** | **0.5660** | **0.8857** | 0.9895 | 0.9648 | **0.2178** |

LingBot 的最明显优势是 Dynamic Degree：比 Yume-1.5 高 **0.1245**，比 HY-World 1.5 高 **0.1640**；说明它减少了长视频模型常见的“为了稳定而静止”。但 Temporal Flickering 低于 HY-World 1.5（0.9648 vs 0.9773），表明更强动态并非无代价。Overall Consistency 的绝对值虽最高，提升只有 0.0162（相对 HYWorld），不能仅凭这一项证明分钟级几何严格正确。

### 6.4 Baseline 公平性

Table 1 还比较 Matrix-Game 2.0、Yume-1.5、HY-World 1.5、Mirage 2、Genie 3，在 domain、horizon、dynamic、resolution、realtime、open-source 六个标签上把 LingBot 标为 general/long/high/720p/realtime/open-source。该表主要是能力声明，不是统一 benchmark：闭源模型无法严格复现，Short/Medium/Long 和 Low/Medium/High 也没有量化阈值。

Table 2 只有两个开源 baseline，且未披露统一 prompt、action trajectory、采样预算、模型版本与方差。VBench 本质仍是视频质量评测，不覆盖动作跟随准确率、回环 pose error、碰撞或 agent task success。因此它支持“视频动态与观感有竞争力”，不足以完全证明“更好的世界模拟”。

### 6.5 消融实验

论文**没有独立 ablation section、表格或数字**。只给出四类非定量陈述：

1. high-noise expert 初始化优于 low-noise expert 的 action dynamics，但无数值。
2. Fast 相对 Base 的感知退化较小，但无同指标比较。
3. adversarial objective 改善视觉质量，但无 `w/o GAN` 对照。
4. progressive curriculum、flow shift、multi-task、Self-Forcing 均有动机解释，却没有逐项移除实验。

这是报告最大的实验缺口。方法链很长，若没有组件消融，很难判断 5→60 秒 curriculum、数据回环、MoE、Self-Forcing、DMD 和 GAN 各自贡献。

### 6.6 应用结果

- **Promptable World Events（Figure 14）**：文本可改变 winter/night/pixel-art/steampunk 等全局状态，或加入 fireworks/birds/fish 等局部事件。它展示分支未来，但没有 object grounding 成功率。
- **Action Agent（Figure 15）**：用同一动作数据微调 Qwen3-VL-2B，输入 image observation，输出未来 **10 秒** action chunks；W/A/S/D 移动，I/J/K/L 离散鼠标方向，再转 camera trajectory 驱动世界模型。
- **3D Reconstruction（Figure 16）**：用外部大型 3D foundation model 把生成视频转点云，覆盖 indoor/sci-fi/outdoor；没有 Chamfer、pose error 或 completeness 数字。

---

## 7. 失败案例、局限与 Future Work

### 7.1 作者明确列出的失败模式

1. **Memory stability**：记忆来自 context window 涌现，而非显式存储，长时模拟会不一致。
2. **Computational cost**：需要 enterprise-grade GPU，消费级硬件不可及。
3. **Limited action space**：目前主要是 navigation 和基础移动，复杂交互不足。
4. **Interaction precision**：缺 object-level grounding，例如在杂乱桌面精准拿指定杯子仍困难。
5. **Generation length & drifting**：视频越长，环境越可能丢失初始结构，不足以支撑 extended gameplay。
6. **Single-agent**：只支持单 agent 视角，未建模多 agent 交互。

论文没有提供专门 failure-case Figure、失败率或按类别统计。Figure 13 的成功样例与上述 drift 限制应同时阅读。

### 7.2 Future Work

作者计划：扩大 action space、增强 physics engine、设计显式 memory module、解决 generation drift，最终走向 infinite-time gameplay。论文没有给显式记忆架构、多 agent 技术路线或消费级部署方案。

### 7.3 个人批判性判断

**我认为 LingBot-World 的核心价值是开源了一条“从强视频模型到实时世界模型”的完整训练路线，而不是证明了模型真正掌握通用物理。** 最有技术含量的是把三种 distribution shift 分开处理：短→长视频用 curriculum；双向→因果用 Diffusion Forcing / block causal；真实历史→自生成历史用 Self-Forcing / DMD。再用 real-data GAN 监督补回少步 student 的高频细节。

最薄弱处是证据与方法复杂度不匹配。28B MoE、长视频课程、动作 adapter、Self-Forcing、DMD、GAN 共同作用，却没有数值消融；100 个 VBench 视频也不能衡量动作与物理。更可信的后续工作应报告：动作分类/轨迹误差、回环重访误差、未观察动态预测准确率、交互成功率、按 rollout 长度的 drift 曲线，以及统一硬件的 Base/Fast 质量—速度 Pareto。

---

## 8. 与 HY-World 1.5 的对照

| 维度 | LingBot-World | HY-World 1.5 / WorldPlay |
|---|---|---|
| 基础模型 | Wan2.2 I2V 14B；Middle 为 28B MoE | HunyuanVideo-8B 主路线，也提供 WAN-5B |
| 训练分期 | Pre / Middle / Post 三阶段 | Pre / Middle / RL Post / Distillation 四阶段 |
| 动作表示 | Plücker camera + discrete multi-hot，经 AdaLN | discrete key + continuous camera pose/PRoPE |
| 记忆 | 长 context 中涌现，无显式检索 | temporal + spatial retrieval，Temporal Reframing |
| 因果化 | Diffusion Forcing + block causal | Diffusion Forcing + chunk-wise block causal |
| 长程后训练 | Self-Forcing + DMD + adversarial | Context Forcing + memory-aligned DMD；另有 WorldCompass RL |
| 实时指标 | 480p 16 FPS，one GPU node，<1s latency | 24 FPS，8-GPU 流式系统 |
| 定量重点 | VBench：动态与观感 | 短/长期重建、pose、VBench、人类偏好 |

两者共同证明“视频 diffusion + 自回归 chunk + 少步蒸馏”是 2025–2026 实时世界模型的主路线；差别在于 LingBot 把 memory 视作长上下文涌现能力，HYWorld 则把 memory retrieval 和 teacher/student context alignment 显式化。

---

## 9. 复现成本与实践建议

### 9.1 官方开源状态

官方 v1 仓库提供：

- `LingBot-World-Base (Cam)`：camera pose 控制，480P / 720P。
- `LingBot-World-Base (Act)`：离散动作控制，480P / 720P。
- `LingBot-World-Fast`：加速 causal 推理，480P / 720P。
- 代码基于 Wan2.2，许可证 Apache-2.0。

截至 2026-09-04，v1 README 已标记“不再积极维护”，后续更新转向 LingBot-World-Infinity/v2；复现本文仍应固定 v1 commit/checkpoint，避免新版本行为改变。

### 9.2 运行要求

- Python 环境需 `torch >= 2.4.0`、`flash_attn`，以及 Wan2.2 依赖。
- 官方 480p/720p 示例使用 `torchrun --nproc_per_node=8`、DiT/T5 FSDP、DeepSpeed Ulysses。
- Base 示例常用 161 frames；可把 `frame_num` 提到 **961** 生成约 1 分钟（16 FPS），显存不足可把 T5 放 CPU。
- Fast 使用 `generate_fast.py`，chunk-wise causal inference + KV cache。官方仓库后续加入 `prewarm()`；其 README 记录在 8×H100、480×832、81 帧下，首 chunk 去噪从约 7284 ms 降至 1027 ms，总生成从 22146 ms 降至 15564 ms，输出 bit-identical。该数字属于后续工程更新，不是论文主实验。

论文和 README 都没有给 v1 模型的明确单卡显存下限；“one GPU node”不等于 one GPU。模型权重大、8-GPU 脚本和 enterprise-grade GPU 限制意味着本地完整复现成本高。

### 9.3 复现层级建议

1. **最低成本：Base (Cam) 推理**。用官方 examples 的 image/prompt/`intrinsics.npy`/`poses.npy`，先验证 480p 161 帧。
2. **动作对齐测试**。固定初图和 prompt，构造 W/A/S/D 与相机旋转序列，测 optical-flow direction、pose estimator error 和人工成功率。
3. **长期记忆测试**。使用闭环 camera trajectory，按 5/15/30/60 秒离屏时间测回访 LPIPS、特征距离和重建 pose。
4. **Fast 质量—速度测试**。同硬件比较 Base 与 Fast 的 FPS、首帧延迟、VBench、回环误差；论文缺少这一关键 Pareto。
5. **完整训练**。需要 28B MoE、60 秒 token 序列、FSDP2/Ulysses、real/fake teacher 与 student 并行更新、长 self-rollout；再加未公开的数据规模，基本不适合普通实验室从零复现。

---

## 10. 关键术语

| 术语 | 含义 |
|---|---|
| Fundamental World Model | Middle-Training 中先学长视频一致性、尚未注入动作的双向模型 |
| Plücker embedding | 用相机射线几何编码连续视角/pose 的表示 |
| Diffusion Forcing | 不同 chunk 使用独立噪声时间，使 diffusion 适配因果序列生成 |
| Block causal attention | chunk 内双向、chunk 间因果的混合 attention |
| Self-Forcing | 训练时以 student 自己生成的历史为条件，缩小 train-test gap |
| DMD | 用 real/fake score 差匹配 student 与数据分布的少步蒸馏 |
| Emergent memory | 从长上下文 attention 中自然出现、但没有显式可写存储的记忆 |

---

## 11. 图表索引与材料来源

- 图 1：开放域 teaser 与动作可控性。
- 图 4：Pre/Middle/Post 三阶段训练。
- 图 5：LingBot-World 生成与动作注入 pipeline。
- 图 6：block-causal generator 与 discriminator。
- 图 12：60 秒离屏重访与未观察动态。
- 表 2：VBench 定量对比。

完整来源清单见 [sources.md](assets/LingbotWorld%20-%20动作可控交互世界模型/sources.md)。
