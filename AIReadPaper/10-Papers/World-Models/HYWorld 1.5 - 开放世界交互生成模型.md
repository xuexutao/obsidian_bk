---
type: paper-note
status: complete
domain: World-Models
paper: "HY-World 1.5: A Systematic Framework for Interactive World Modeling with Real-Time Latency and Geometric Consistency"
year: 2025
arxiv: null
doi: null
source: https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf
project: https://3d-models.hunyuan.tencent.com/world/
code: https://github.com/Tencent-Hunyuan/HY-WorldPlay
tags: [World-Models, Interactive-World-Model, Video-Diffusion, Action-Control, Long-Term-Memory, Diffusion-Forcing, Context-Forcing, Reinforcement-Learning]
created: 2026-09-04
updated: 2026-09-04
---

# HYWorld 1.5：开放世界交互生成模型

## 基本信息

- **正式标题**：*HY-World 1.5: A Systematic Framework for Interactive World Modeling with Real-Time Latency and Geometric Consistency*
- **作者/团队**：Tencent Hunyuan / HunyuanWorld Team；项目负责人 Chunchao Guo、Tengfei Wang；核心贡献者包括 Haiyu Zhang、Wenqiang Sun、Haoyuan Wang、Zehan Wang、Junta Wu、Zhenwei Wang、Xuhui Zuo、Chenjie Cao
- **发布时间**：2025 年 12 月（官方仓库记录 2025-12-17 发布报告与模型）
- **官方报告**：<https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>
- **官方代码**：<https://github.com/Tencent-Hunyuan/HY-WorldPlay>
- **官方模型**：<https://huggingface.co/tencent/HY-WorldPlay>
- **本地 PDF**：[paper.pdf](assets/HYWorld%201.5%20-%20开放世界交互生成模型/sources/paper.pdf)

## 一句话结论

HY-World 1.5 / WorldPlay 把开放场景交互建模写成“每次预测 16 帧”的动作条件自回归视频扩散问题，并用**离散按键 + 连续相机位姿、重构上下文记忆、WorldCompass 强化学习和 Context Forcing 蒸馏**同时解决控制、长程几何一致性与速度问题；完整系统在 8-GPU 流式部署下达到 **24 FPS**，且在 ≥250 帧回环测试上显著优于无记忆或非实时 baseline。

---

## 1. 研究背景与问题定义

### 1.1 从“生成一个 3D 场景”到“持续可玩的世界”

HY-World 1.0 的目标是从文字或图像离线生成可漫游 3D 世界，但生成过程较长，也不能在用户每次键鼠输入后立即反馈。HY-World 1.5 改走视频世界模型路线：不先构造完整显式 3D 地图，而是根据历史画面、当前动作与场景条件持续生成下一段视频。这样更容易覆盖真实、风格化、第一/第三人称及动态事件，却引入两个互相牵制的要求：

1. **实时性**：扩散模型通常需要多步去噪；每次动作若等待数秒便不能交互。
2. **长程几何一致性**：只保留最近帧虽然快，但转身离开后再返回，建筑、物体与相机位置容易漂移；显式 3D cache 或复杂检索又会增加延迟，并让蒸馏更困难。

此前方法往往只占一端：Oasis、Matrix-Game 2.0 重实时而轻记忆；WorldMem、VMem 能保存记忆却不实时；Gen3C 的显式 3D cache 依赖深度与中间生成质量。WorldPlay 的核心研究问题因此是：**如何让记忆感知的动作条件视频扩散模型，在长序列中仍能被蒸馏成实时模型？**

![](assets/HYWorld%201.5%20-%20开放世界交互生成模型/figures/01-teaser.png)

> **图 1（论文 Figure 1）**：WorldPlay 在真实、风格化和第三人称场景中按键漫游；红框强调离开后重访的外观/几何一致性，同时展示点云重建与文本触发烟雾事件。它直观呈现了论文要同时满足的四个维度：泛化、动作响应、长期记忆和可扩展应用。  
> 来源：技术报告 Figure 1，第 4 页，<https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>

### 1.2 核心贡献

1. **完整四阶段训练框架**：Pre-Training → Middle-Training → RL Post-Training → Distillation，而非只展示最终模型。
2. **Dual Action Representation**：离散键鼠动作提供跨场景尺度鲁棒性，连续相机内外参提供精确空间定位，两者互补。
3. **Reconstituted Context Memory**：每个新 chunk 动态组合近期时序记忆与远期空间记忆，再用 Temporal Reframing 重写时间位置，抵抗 RoPE 长距离衰减。
4. **WorldCompass**：用 clip-level rollout、动作跟随奖励与视觉质量奖励，对长程自回归世界模型做 RL 后训练。
5. **Context Forcing**：在蒸馏时对齐 teacher 与 student 实际看到的记忆上下文，保留长期一致性，并把推理压到 4 个去噪步。
6. **系统级实时化**：DiT/VAE 8-GPU 混合并行、Triton 流式服务、渐进 VAE 解码、量化、SageAttention 与 KV cache 合力达到 24 FPS。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与状态转移

模型把世界表示为持续增长的视频 chunk 序列。第 $t$ 次交互时：

- **场景条件 $c$**：一张初始图像或文本 prompt；推理中还可以更新文本以触发事件。
- **历史观测**：$O_{t-1}=\{x_{t-1},\ldots,x_0\}$，其中每个 $x_i$ 是 16 帧视频 chunk。
- **历史动作**：$A_{t-1}=\{a_{t-1},\ldots,a_0\}$。
- **当前动作 $a_t$**：键盘移动、鼠标/相机旋转及连续 camera pose。
- **输出 $x_t$**：下一段 16 帧视频（VAE latent 中为 4 个 temporal latents）。

条件转移可写成：

$$
x_t \sim N_\theta\left(x_t\mid O_{t-1},A_{t-1},a_t,c\right).
$$

这里模型不是一次性生成固定时长视频，而是在每个 chunk 后重建上下文并继续生成，因此理论上可流式延长。关键假设是：RGB 历史与相机轨迹中包含足够信息，让 Transformer 隐式恢复局部世界状态；论文没有维护一个可查询、可编辑的显式 3D scene graph。

### 2.2 基础 Flow Matching 目标

双向视频扩散 backbone 先用 3D VAE 把视频压到 latent $z_0$，采样高斯噪声 $z_1\sim\mathcal N(0,I)$，在扩散时间 $k$ 得到 $z_k$，并训练 DiT 预测目标速度 $v_k$：

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{k,z_0,z_1}
\left[\left\|N_\theta(z_k,k)-v_k\right\|_2^2\right].
$$

$N_\theta$ 是扩散 Transformer。直觉上，它学习从不同噪声程度沿概率流把 latent 拉回真实视频；动作、文本、图像与记忆随后都作为条件进入这个去噪过程。

---

## 3. 数据系统与四阶段训练总览

![](assets/HYWorld%201.5%20-%20开放世界交互生成模型/figures/02-system-overview.png)

> **图 2（论文 Figure 2）**：系统把数据库经筛选、重平衡和结构化标注后送入四阶段训练；在线端接收图像/文本与用户动作，以流式 DiT 去噪和流式 VAE 解码返回视频，必要时再接 3D/4D reconstruction。  
> 来源：技术报告 Figure 2，第 5 页，<https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>

### 3.1 320K 视频训练集

报告披露总计 **320K curated clips**，四类数据互补：

| 数据类别 | 数量 | 比例 | 作用 |
|---|---:|---:|---|
| AAA 游戏录屏 | 170K | 53.125% | 第一/第三人称、复杂 agent 行为、物理与环境交互；按键可直接记录 |
| Real-world 3D（DL3DV） | 60K | 18.75% | 先做 3D reconstruction，再设计漫游轨迹；提供真实外观与可回访几何 |
| Synthetic 4D（Unreal Engine） | 50K | 15.625% | 动态对象、可控光照、精确相机与动作真值 |
| Real-world video（Sekai） | 40K | 12.5% | 自然动态、真实物体行为和运动先验 |

过滤包括审美质量、watermark/UI、压缩伪影、光流运动强度与异常相机运动。随后以动态帧率增强不同速度分布；HunyuanVideo 1.5 captioner 生成结构化文本；相机 pose 来自 VIPE、UE 或 3D 渲染管线，再从 pose 派生移动/视角旋转离散动作。数据设计值得肯定之处是把“画质/语义多样性”“动作监督”“回环与相机几何”分给不同来源，而不是要求单一互联网视频同时满足所有条件。

### 3.2 四阶段的职责分工

| 阶段 | 主要目标 | 关键技术 | 产物 |
|---|---|---|---|
| 1. Pre-Training | 从双向视频模型得到可无限延长的基础 AR 生成器 | Diffusion Forcing、block causal attention、16 帧 chunk | 无动作/弱记忆的 chunk-wise AR diffusion |
| 2. Middle-Training | 注入控制与长期几何记忆 | Dual Action、PRoPE、Reconstituted Context Memory、Temporal Reframing | 可控、可回访的 memory-aware AR model |
| 3. RL Post-Training | 显式优化复杂动作跟随与视觉质量 | WorldCompass、clip-level rollout、互补奖励、DiffusionNFT | 更强动作对齐与更少视觉伪影 |
| 4. Distillation | 把多步模型压到实时，同时保留记忆 | Context Forcing、memory-aligned teacher/student、self-rollout、DMD | 4-step realtime model |

这个拆分很重要：动作控制、记忆、策略对齐和速度不是一次 end-to-end 训练自然涌现，而是逐阶段改变训练分布与目标。

---

## 4. 核心方法

### 4.1 Stage 1：Diffusion Forcing 与 chunk-wise 自回归

双向视频 diffusion 可同时看整段视频，不能直接做无限流式生成。WorldPlay 把视频切为 **4 个 VAE latents / 16 帧**一组的 chunk，并做两处改变：

1. **不同 chunk 独立加不同噪声级别**，即 Diffusion Forcing。模型在同一训练样本中同时面对干净历史、部分去噪状态和高噪声未来，从而把连续扩散去噪与 next-chunk prediction 结合起来。
2. **双向 self-attention 改为 block-causal attention**：chunk 内仍可双向建模局部运动；chunk 间只看过去，满足在线生成因果性。

训练仍沿用 Flow Matching 形式，但噪声时间按 chunk 独立采样。其价值不是单纯“让视频更长”，而是让模型在训练时见到推理时会遇到的混合状态，为后续 self-rollout 和蒸馏建立接口。

![](assets/HYWorld%201.5%20-%20开放世界交互生成模型/figures/03-worldplay-architecture.png)

> **图 4（论文 Figure 4）**：每轮根据最近片段与空间检索帧重构 memory context，再把双动作注入 AR Diffusion Transformer，生成并解码下一 16 帧；新结果写回 memory cache。  
> 来源：技术报告 Figure 4，第 7 页，<https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>

### 4.2 Stage 2A：Dual Action Representation

只使用 W/A/S/D 等离散动作有两个优点：动作含义跨不同场景尺度较稳定，且训练不容易因单位尺度混乱而发散；缺点是“向前”没有绝对位姿，难以精确回到旧位置。连续相机旋转/平移矩阵则反过来：可定位，但不同数据源的尺度差异会导致训练不稳定。因此模型同时编码两种信号。

#### 4.2.1 离散动作注入

键盘和鼠标动作先转成 embedding，经**零初始化 MLP**后加入 diffusion timestep embedding，再通过 AdaLN 调制每个 DiT block。零初始化使新分支开始时近似不改变预训练模型，随后逐步学会动作控制，降低对原视频先验的破坏。

#### 4.2.2 连续相机 pose 注入

连续内外参通过 Projective Positional Encoding（PRoPE）进入 self-attention。普通 3D RoPE attention 为：

$$
\operatorname{Attn}_1=
\operatorname{Attn}\left(R^\top\odot Q,\;R^{-1}\odot K,\;V\right),
$$

其中 $R$ 是视频 latent 的 3D RoPE，$\odot$ 表示矩阵-向量乘。为了把相机 frustum 的相对关系编码进去，再计算：

$$
\operatorname{Attn}_2=D_{\mathrm{proj}}\odot
\operatorname{Attn}\left(
D_{\mathrm{proj}}^\top\odot Q,
D_{\mathrm{proj}}^{-1}\odot K,
D_{\mathrm{proj}}^{-1}\odot V
\right),
$$

$D_{\mathrm{proj}}$ 由 camera intrinsics/extrinsics 构造。最终输出：

$$
\operatorname{Attn}_{\mathrm{out}}=
\operatorname{Attn}_1+\text{zero\_init}(\operatorname{Attn}_2).
$$

第一项保留视频自身时空建模，第二项以几何相对位置修正 attention；零初始化同样避免刚接入 pose 时破坏基础模型。

![](assets/HYWorld%201.5%20-%20开放世界交互生成模型/figures/04-dual-action-dit.png)

> **图 5（论文 Figure 5）**：左侧离散 key token 与时间 embedding 调制 AdaLN；右侧连续 camera token 通过 PRoPE 形成第二条 causal attention，再经零初始化残差合入。该图解释了“动作控制”与“空间定位”如何在同一 DiT 中分工。  
> 来源：技术报告 Figure 5，第 7 页，<https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>

### 4.3 Stage 2B：Reconstituted Context Memory

若把所有历史帧放进 context，注意力与显存随时间增长，不可持续；若只留最近帧，回访旧区域就会丢失几何。WorldPlay 在预测每个 $x_t$ 前，从 $O_{t-1}$ 重构固定规模上下文：

$$
C_t=C_t^T\cup C_t^S.
$$

- **Temporal memory $C_t^T$**：最近 $L$ 个 chunks $\{x_{t-L},\ldots,x_{t-1}\}$，负责局部运动连续、速度与外观平滑。
- **Spatial memory $C_t^S$**：从非邻接历史帧 $O_{t-1}\setminus C_t^T$ 中采样，以 **FOV overlap + camera distance** 的几何相关性排序，负责回访时恢复旧区域。

仅检索还不够。标准 RoPE 中，旧帧和当前 chunk 的相对时间距离会不断增大，最终超出训练插值范围，也让 attention 倾向忽略“很久以前但空间上很近”的帧。**Temporal Reframing** 丢弃这些帧的绝对时间索引，按当前重构上下文重新分配位置，使空间相关旧帧在 RoPE 中始终离当前 chunk 很近。它相当于把“视频时间”暂时改写成“当前决策所需的记忆顺序”。

这里的个人判断是：该机制是一个有效但不完备的隐式地图。它把检索成本控制住，并以 pose 提升召回准确性，却没有显式处理遮挡、动态物体身份和拓扑冲突；记忆仍以图像 token 形式存在，错误画面一旦写入 cache 也可能被后续反复强化。

### 4.4 Stage 3：WorldCompass 强化学习后训练

像素重建/扩散损失只能间接学到“动作执行得对不对”，在组合按键、长 rollout 和复杂场景中容易达到平台。WorldCompass 用显式反馈继续优化：

1. **Clip-Level Rollout**：以视频 clip 为粒度进行 AR rollout，而不是只评一个独立 frame。模型必须条件于自己的不完美历史，缓解 exposure bias；同时 clip 级奖励比整条超长轨迹更密集。
2. **Complementary Rewards**：至少包含 action-following score 与 visual-quality score。只优化前者可能通过扭曲画面来夸大运动，只优化后者可能输出漂亮但几乎静止的视频；互补奖励用于压制 reward hacking。
3. **DiffusionNFT**：采用面向 diffusion 的在线 RL 算法，并配合效率策略减少长轨迹采样成本。

技术报告没有给出完整 reward 公式、权重与训练超参数，而是指向独立 WorldCompass 论文。报告内的 Figure 11 仅提供定性消融：无 RL 时复杂动作下出现动作错配和画质退化；加入 RL 后二者均改善。因此可确认“有效”，但不能从本报告量化 RL 单独贡献。

### 4.5 Stage 4：Context Forcing 蒸馏

#### 4.5.1 为什么普通 DMD 会失败

常规 Distribution Matching Distillation（DMD）让 AR student 的联合分布匹配数据/teacher：

$$
\nabla_\theta\mathcal L_{\mathrm{DMD}}
=
\mathbb E_k\left[
\nabla_\theta\operatorname{KL}
\left(p_\theta(x_{0:t})\;\|\;p_{\mathrm{data}}(x_{0:t})\right)
\right].
$$

但双向 teacher 看得到一段内的过去与未来，AR student 只能看过去；对 memory-aware 模型，二者还可能检索到不同的 memory context。即便输出视频相似，条件分布 $p(x\mid C)$ 也因 $C$ 不同而不在同一问题上，score 差会把 student 往错误方向推，表现为记忆丢失、模式坍缩或长程漂移。

#### 4.5.2 记忆对齐的自 rollout

Context Forcing 让 student 基于真实前缀和重构记忆，自 rollout **4 个 chunks**：

$$
p_\theta(x_{j:j+3}\mid x_{0:j-1})
=
\prod_{i=j}^{j+3}p_\theta(x_i\mid C_i).
$$

随后构造 memory-augmented bidirectional teacher $V_\beta$，但从 teacher 的上下文中屏蔽当前 4 个 student 生成 chunks：

$$
p_{\mathrm{data}}(x_{j:j+3}\mid x_{0:j-1})
=
p_\beta\left(x_{j:j+3}\mid C_{j:j+3}\setminus x_{j:j+3}\right).
$$

这样 teacher 与 student 都依赖同一批历史记忆，不再让 teacher 偷看目标区间。训练中 rollout 起点/长度逐步增加；每个 chunk 都重新检索 $C_i$，随机使用 $1\ldots d$ 个去噪步；real score 网络冻结，fake score 网络用 flow matching 跟踪 student 分布，student 用 DMD 更新。最终推理使用 **4 denoising steps**。

![](assets/HYWorld%201.5%20-%20开放世界交互生成模型/figures/05-context-forcing.png)

> **图 7（论文 Figure 7）**：student 在 memory cache 条件下自 rollout；同一对齐后的记忆被送入冻结的 real teacher 与可训练 fake score 模型，二者 score 差形成 DMD 信号。Context Forcing 的关键并非再加一个记忆模块，而是让 teacher/student 的条件信息一致。  
> 来源：技术报告 Figure 7，第 10 页，<https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>

#### 4.5.3 与 Diffusion Forcing / Self Forcing 的关系

- **Diffusion Forcing**解决“怎样把扩散与 AR chunk 预测统一”：训练时给不同 chunk 不同噪声级别。
- **Self Forcing / self-rollout**解决“怎样让模型见到自己的错误历史”：后续条件由 student 自己生成，而非永远用 ground truth。
- **Context Forcing**进一步解决“有记忆检索时 teacher/student 条件不一致”：不仅 self-rollout，还显式对齐重构 memory context。

三者分别对应建模形式、train-test gap 和 memory-aware distillation，不是同义词。

### 4.6 实时推理与系统优化

仅把去噪变成 4 步还不足以达到在线体验，报告还组合了：

- **DiT + VAE 混合并行**：在 **8 GPUs** 上同时使用 sequence parallelism 与 attention parallelism，把一个 chunk 的 token 分散到设备。
- **NVIDIA Triton 流式部署**：DiT 生成 latent 后，不等待整个 chunk 完成，VAE 分小批逐步解码并立刻传输。
- **异步缓冲**：生成、解码、网络传输解耦，降低 time-to-first-frame。
- **量化与高效 attention**：SageAttention、权重/激活浮点量化、矩阵乘量化。
- **KV cache**：每个新 chunk 只计算新增 token；当上下文重构后重建对应 KV。

官方仓库给出的 HunyuanVideo-8B 路线显存需求为：AR distilled、125 帧输入时，sequence parallel size `sp=8/4/1` 分别约 **28/34/72 GB 每卡**；训练同长度、`sp=8` 约 **60 GB 每卡**。另有 WAN-5B 轻量路线，但官方明确说明动作控制和长期记忆会折损。因此“24 FPS”是系统吞吐结论，不应解读为消费级单卡速度。

---

## 5. 实验设置与结果

### 5.1 评测协议

报告使用 **600 个**测试案例，覆盖真实视频、游戏录屏和 AI 生成图。分两组：

- **短期**：61 帧，用测试视频 pose 驱动生成，与 ground truth 比较 PSNR、SSIM、LPIPS，以及相机旋转/平移误差 $R_{dist}$、$T_{dist}$。
- **长期**：≥250 帧，设计出去再返回的 cycle trajectory，把回程帧与首次经过同一路径的帧比较。该协议比普通视频指标更直接检验“回访是否仍是同一地点”。

Baseline 分两类：

1. **无专门记忆的动作控制 diffusion**：CameraCtrl、SEVA、ViewCrafter、Matrix-Game 2.0、GameCraft。
2. **有记忆机制**：Gen3C（显式 3D cache）与 VMem（surfel-indexed view memory）。

### 5.2 主定量结果

| Model | 实时 | 短期 PSNR↑ | SSIM↑ | LPIPS↓ | $R_{dist}$↓ | $T_{dist}$↓ | 长期 PSNR↑ | SSIM↑ | LPIPS↓ | $R_{dist}$↓ | $T_{dist}$↓ |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CameraCtrl | ✗ | 17.93 | 0.569 | 0.298 | 0.037 | 0.341 | 10.09 | 0.241 | 0.549 | 0.733 | 1.117 |
| SEVA | ✗ | 19.84 | 0.598 | 0.313 | 0.047 | 0.223 | 10.51 | 0.301 | 0.517 | 0.721 | 1.893 |
| ViewCrafter | ✗ | 19.91 | 0.617 | 0.327 | 0.029 | 0.543 | 9.32 | 0.277 | 0.661 | 1.573 | 3.051 |
| Gen3C | ✗ | 21.68 | 0.635 | 0.278 | **0.024** | 0.477 | 15.37 | 0.431 | 0.483 | 0.357 | 0.979 |
| VMem | ✗ | 19.97 | 0.587 | 0.316 | 0.048 | 0.219 | 12.77 | 0.335 | 0.542 | 0.748 | 1.547 |
| Matrix-Game 2.0 | ✓ | 17.26 | 0.505 | 0.383 | 0.287 | 0.843 | 9.57 | 0.205 | 0.631 | 2.125 | 2.742 |
| GameCraft | ✗ | 21.05 | 0.639 | 0.341 | 0.151 | 0.617 | 10.09 | 0.287 | 0.614 | 2.497 | 3.291 |
| Ours w/o Context Forcing | ✗ | 21.27 | 0.669 | 0.261 | 0.033 | 0.157 | 16.27 | 0.425 | 0.495 | 0.611 | 0.991 |
| **Ours full** | **✓** | **21.92** | **0.702** | **0.247** | 0.031 | **0.121** | **18.94** | **0.585** | **0.371** | **0.332** | **0.797** |

最关键的不是短期 PSNR 比 Gen3C 高 0.24，而是**长短期差距**：完整模型从短期 21.92/0.702 降到长期 18.94/0.585；Gen3C 降到 15.37/0.431，VMem 降到 12.77/0.335，无记忆模型更差。也就是说，WorldPlay 的优势随轨迹拉长而扩大。它的长期 $R_{dist}=0.332$ 虽略好于 Gen3C 0.357，但短期旋转误差不是最佳，说明优势来自综合稳定性，而非每个几何指标都统治。

### 5.3 Context Forcing 消融

`w/o Context Forcing → full` 同时改变了速度与质量：

- 短期：PSNR **21.27→21.92**，SSIM **0.669→0.702**，LPIPS **0.261→0.247**。
- 长期：PSNR **16.27→18.94（+2.67 dB）**，SSIM **0.425→0.585（+0.160）**，LPIPS **0.495→0.371**。
- 长期 pose：$R_{dist}$ **0.611→0.332**，$T_{dist}$ **0.991→0.797**。
- 系统从非实时变为实时。

这个消融非常支持论文主张：若蒸馏只是换成少步 AR，而没有对齐记忆条件，最先崩的正是长期指标；Context Forcing 不仅提速，反而减轻了误差累积。但表格没有与“普通 DMD 但同样 4 步”的严格等算力对照，因此无法把全部收益唯一归因于记忆对齐。

### 5.4 VBench 与人类偏好

![](assets/HYWorld%201.5%20-%20开放世界交互生成模型/figures/06-key-results.png)

> **表 2 + 图 9（论文 Table 2 / Figure 9）**：上方是短/长期重建与 pose 指标；左下 VBench 雷达图显示 WorldPlay 在多数维度更均衡；右下人类偏好中，WorldPlay 对 GameCraft、Matrix-Game 2.0、ViewCrafter、Gen3C 的偏好率分别为 **88.5%、78.4%、92.1%、72.9%**。与双向 teacher 比较为 **48.1% vs 51.9%**，说明实时 student 已接近但尚未超过教师画质。  
> 来源：技术报告 Table 2、Figure 9，第 13 页，<https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>

### 5.5 WorldCompass、定性结果与应用

- **RL 定性消融**：Figure 11 显示不做 RL 时复杂组合动作存在响应偏差和视觉退化；RL 后动作方向与画质改善。但报告没有数值，不能估算 reward 各分量贡献。
- **长期回访**：Figure 10 中 Matrix-Game 2.0 / GameCraft 缺专门记忆，不能自由回访；Gen3C 对深度误差敏感；WorldPlay 在红框区域保持布局与对象。
- **3D reconstruction**：把一致多视角视频交给 WorldMirror 得到点云。它证明视频可作为重建输入，但报告没有 Chamfer、重投影误差或完整率，不能等同于模型本身输出 3D。
- **Promptable events**：流式中更新文本，可添加/移除对象、改变天气/光照、触发爆炸或 NPC 行为。Figure 8 的气球和彩虹、Figure 13 的天暗/火焰属定性演示。
- **Video continuation**：从已有视频继续生成，在运动、外观和光照上延续。

### 5.6 失败案例与证据缺口

技术报告**没有给出专门 failure-case figure 或失败率**。从方法与实验可确定的风险包括：

1. 回环评测主要由 camera trajectory 驱动，不能代表物体级操作、角色碰撞与因果交互都正确。
2. 记忆依赖历史生成帧；一旦早期 hallucination 被检索回来，Temporal Reframing 可能提高错误记忆权重。
3. 复杂动态对象的身份与遮挡没有显式状态；点云结果主要覆盖几何稳定场景。
4. 人类评测只给偏好率，未披露受试者数量、置信区间和随机化细节。
5. 报告把详细 action/memory/RL ablation 分流到 WorldPlay 与 WorldCompass 独立论文，本报告自身无法审计各组件的独立增益。

---

## 6. 与相关工作的关系

- **对比 Matrix-Game 2.0 / Oasis**：同为实时流式视频世界模型，但 WorldPlay 重点加入 pose-aware memory；代价是 8-GPU 系统和更复杂的训练链。
- **对比 Gen3C / VMem**：它不把显式 3D cache 或 surfel memory 作为唯一世界状态，而以相机几何引导的历史图像 token 做隐式记忆，动态场景适应性更强，但缺乏可验证、可编辑的显式几何。
- **对比普通 Self-Forcing / DMD**：Context Forcing 识别出 memory-aware student 的条件分布错配，把“对齐上下文”提升为蒸馏的一等约束。
- **对比 HunyuanWorld 1.0**：1.0 先生成持久 3D 资产；1.5 先追求实时可交互视频，再可选地做 3D/4D reconstruction。两者输出表示和应用边界不同。

---

## 7. 局限、Future Work 与个人判断

### 7.1 作者明确提出的后续方向

结论把未来工作集中在：**更长视频、多 agent 交互、复杂物理动态与更丰富交互**。当前系统主要仍是 navigation control；文本事件虽扩展了语义操控，但不是精确的物体级动作接口。

### 7.2 关键局限

1. **“长期一致”不是持久世界状态**：记忆由筛选后的历史视频 token 构成；没有显式对象 ID、可查询地图或确定性物理状态。
2. **实时性依赖重工程**：24 FPS 来自 4-step student、8-GPU 混合并行、量化、缓存、流式解码的系统组合，不能只归功于算法。
3. **训练成本与细节披露不足**：给出了 320K 数据构成和部分显存，却未披露总 GPU-hours、优化器、学习率、各阶段训练时长及 RL reward 权重。
4. **评测仍以视频/相机指标为主**：缺少交互成功率、碰撞正确率、闭环 agent performance、世界状态可重复性等 simulator 指标。
5. **安全与数据治理未展开**：AAA 游戏录屏和真实视频的授权范围、过滤偏差与风格记忆没有在报告中系统讨论。

### 7.3 个人判断

**我认为这篇报告最有价值的不是“24 FPS”本身，而是把 memory-aware world model 的工程矛盾拆成可训练的接口。** Dual Action 让检索有坐标，Temporal Reframing 让远期记忆重新获得注意力，Context Forcing 则让蒸馏 teacher 在同一记忆条件上评分。这三者形成闭环，解释了为何长期指标提升明显。

但它更准确的名称仍是**可交互生成式视觉模拟器**，还不是具备可验证状态转移的通用世界模型。要用于机器人训练或严肃游戏，下一步应把显式对象/几何状态与隐式视频先验结合，并用 agent task success、物理约束违反率和跨回环状态误差评测，而不只看 PSNR/VBench 与主观偏好。

---

## 8. 复现成本与实践建议

### 8.1 开源状态

官方仓库已提供 HunyuanVideo-8B 路线的 bidirectional、AR、AR+RL、4-step distilled checkpoints，以及训练/推理文档；另有 WAN-5B 轻量版本。仓库新闻显示训练代码、量化推理和 WorldCompass 代码已分阶段开放。`AR + RL + distilled` 权重在所读 README 的 model zoo 中仍标注待发布，具体以仓库当前状态为准。

### 8.2 最低可行复现层级

1. **推理验证**：优先跑官方 4-step AR distilled 480P-I2V，验证单段生成、动作方向与短回环。
2. **记忆验证**：自行构造“前进—转向—返回”轨迹，固定 seed，对比关闭 spatial memory、关闭 temporal reframing 和 full model；报告没有直接给这些开关的现成数值。
3. **训练 action adapter / 小规模微调**：需要 pose/action 对齐视频；可先用 UE 合成回环轨迹，避免重建完整 320K 数据。
4. **完整四阶段训练**：成本极高，除 8-GPU 长序列训练外，还需要视频 curation、VIPE/3D 重建、RL rollout 与 teacher/fake/student 多模型 DMD，普通实验室很难严格复现。

### 8.3 资源估计

- 官方 HunyuanVideo 路线推理：`sp=8` 约 28 GB/卡，`sp=4` 约 34 GB/卡，`sp=1` 约 72 GB；均为 125-frame 配置。
- 同配置训练：`sp=8` 约 60 GB/卡。
- 依赖 CUDA、SageAttention（WAN 路线必需）、可选 FlashAttention；实时服务还涉及 Triton、量化与网络链路。
- 报告未披露完整训练 GPU 数量、总时长与总成本，不能给出可信 GPU-hour 估计。

### 8.4 建议的可复现实验

最值得先复现的不是全模型，而是一个受控的 **memory ablation benchmark**：统一 50–100 个 UE 场景与闭环轨迹，比较 recent-only、retrieval-only、retrieval+reframing、Context-Forcing distilled 四组；同时记录长期 LPIPS/pose error、动作延迟和显存。它能直接验证论文最核心、也最可迁移的贡献。

---

## 9. 关键术语

| 术语 | 含义 |
|---|---|
| Chunk-wise AR diffusion | 每次生成一个 16 帧 chunk，chunk 内联合去噪、chunk 间因果递推 |
| Diffusion Forcing | 对不同时间 chunk 施加独立噪声时间，把 full-sequence diffusion 与 AR prediction 统一 |
| Dual Action Representation | 离散键鼠动作 + 连续相机 pose 的联合控制表示 |
| Reconstituted Context Memory | 每步动态组合近期 temporal memory 与远期 spatial memory |
| Temporal Reframing | 重写被检索旧帧的 RoPE 时间索引，使其相对当前保持近距离 |
| WorldCompass | 用 clip rollout 和动作/视觉互补奖励做世界模型 RL 后训练 |
| Context Forcing | 蒸馏时对齐 teacher/student 的记忆条件，并在自 rollout 上做分布匹配 |

---

## 10. 图表索引与材料来源

- 图 1：多场景 teaser 与能力概览。
- 图 2：数据—四阶段训练—流式推理全系统。
- 图 4：WorldPlay chunk-wise AR 与记忆重构。
- 图 5：Dual Action / PRoPE 注入结构。
- 图 7：Context Forcing 的 memory-aligned DMD。
- 表 2 + 图 9：短长期指标、VBench 和人类偏好。

完整来源清单见 [sources.md](assets/HYWorld%201.5%20-%20开放世界交互生成模型/sources.md)。
