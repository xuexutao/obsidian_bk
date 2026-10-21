---
type: paper-note
status: complete
domain: World-Models
paper: "AnchorWorld: Embodied Egocentric World Simulation with View-based Evolution Customization"
year: 2026
arxiv: "2606.07326"
source: https://arxiv.org/abs/2606.07326
project: https://yuli0103.github.io/AnchorWorld/
code: null
tags:
  - World-Models
  - Egocentric-Video
  - Embodied-AI
  - Anchor-Views
  - Kling
created: 2026-10-16
updated: 2026-10-16
---

# AnchorWorld：可定制具身第一人称世界模拟

## 基本信息

- **正式题名**：AnchorWorld: Embodied Egocentric World Simulation with View-based Evolution Customization
- **作者**：Yu Li、Menghan Xia、Gongye Liu、Xintao Wang、Conglang Zhang、Lei Ke、Yuxuan Lin、Ruihang Chu、Pengfei Wan、Kun Gai、Yujiu Yang
- **机构**：清华大学、华中科技大学、快手 Kling 团队、香港科技大学、武汉大学
- **版本**：arXiv v1，2026-06-05
- **基础模型**：Wan2.2 TI2V 5B
- **核心关键词**：3D human motion、hybrid-view supervision、Spatial Pose Attention、anchor view、view-based evolution customization

## 一句话结论

AnchorWorld 把“世界可控”拆成两层：用完整 3D 人体动作控制第一人称主体，用带 RGB、6-DoF pose 与 evolution prompt 的 anchor views 定义局部世界；其关键是借第三人称完整身体可见性补足第一人称监督盲区，再通过统一坐标中的锚点视图让初始视野外区域也能被定位、保持并随时间演化。

---

## 1. 研究背景与问题定义

### 1.1 第一人称交互的双重缺口

现有交互世界模型多用键鼠、相机轨迹或文本控制，适合导航，却不能完整表达人体根轨迹、肢体动作、头部运动与环境交互。即使输入完整 3D 人体动作，第一人称视频里大部分身体不可见，模型只能间接观察动作后果，监督稀疏且错位。

第二个缺口是世界本身通常只由初始帧、全局 prompt 或历史隐式决定。用户无法指定“某个三维位置应该是什么样、之后如何变化”，尤其难约束最初不可见的区域。AnchorWorld 因而同时提出 embodied action control 与 localized world-state customization。

![](assets/AnchorWorld%20-%20可定制具身第一人称世界模拟/figures/01-teaser.png)

> 图 1：上半部分由完整人体动作生成第一人称视频；下半部分用 anchor view 的图像、pose 和演化文本约束局部世界及其动态。来源：论文 Figure 1。

### 1.2 输入与输出

人体动作来自 SMPL-X：

$$M\in\mathbb R^{f\times k\times6}$$

每个关节的 6 维为 3D 位置加 3D axis-angle 旋转；实际使用 22 个主要关节，不含手部。逐帧相机姿态为：

$$C\in\mathbb R^{f\times3\times4}$$

世界由初始第一人称帧 $I_0$ 与锚点集合定义：

$$\mathcal S=\{(\mathbf I_i,\mathbf c_i,\mathbf t_i)\}_{i=1}^{n}$$

其中 $\mathbf I_i$ 指定局部外观，$\mathbf c_i=[\mathbf R_i\mid\mathbf p_i]$ 是 6-DoF 视点，$\mathbf t_i$ 描述局部状态随时间的演化。输出为 77 帧、480p 的第一人称视频。

### 1.3 贡献

1. 混合第一/第三人称监督：第三人称学习完整动作与场景交互，第一人称阶段适配头部视点。
2. Spatial Pose Attention：按帧联合视频、人体与相机 token，显式建立动作—视角—图像对应。
3. pose-associated anchor views：用图像、三维位姿和文本定义可编辑局部世界状态。
4. 四阶段渐进训练，依次学习外部视角动作、第一人称动作、静态锚点和动态演化。

---

## 2. 混合视角人体动作监督

### 2.1 为什么第三人称是必要辅助

第一人称中身体常在视野外，但第三人称能看到完整人体和人与环境的接触。AnchorWorld 把全身动作与相机轨迹组合成统一 action representation：当相机是外部观察位姿时生成 TPV，当相机对齐头部时生成 FPV。论文称其为 projection-based action conditioning；它不是显式可微渲染，而是让模型学习 3D 动作在不同相机下对应的 2D 观察。

### 2.2 Spatial Pose Attention

动作编码器输出：

$$\mathbf z_m\in\mathbb R^{f'\times k\times d}$$

相机编码器输出：

$$\mathbf z_c\in\mathbb R^{f'\times1\times d}$$

与视频 token 按每一 latent 帧拼接：

$$\mathbf T=[\mathbf z_v^{(t)};\mathbf z_m;\mathbf z_c]\in\mathbb R^{f'\times(h\cdot w+k+1)\times d}$$

再执行帧级空间注意力：

$$\mathbf z_v^{(t)}\leftarrow\mathbf z_v^{(t)}+\operatorname{Truncate}(\operatorname{Attn}(W_Q\mathbf T,W_K\mathbf T,W_V\mathbf T))$$

`Truncate` 在注意力后丢弃姿态和相机 token，只保留更新的视频特征，避免与 VAE latent 分布不同的辅助 token 长期滞留主序列。

---

## 3. Evolvable Anchor-View Customization

![](assets/AnchorWorld%20-%20可定制具身第一人称世界模拟/figures/02-method-pipeline.png)

> 图 2：初始 ego frame 与多个 anchor images 经 VAE 编码后按帧拼接；完整人体动作和相机 pose 进入 Spatial Pose Attention；anchor pose 注入视觉 token，evolution prompt 经 cross-attention 控制局部演化。来源：论文 Figure 2。

### 3.1 In-context anchor image

anchor 图像编码为：

$$\mathbf z_s\in\mathbb R^{f_s\times h\cdot w\times d}$$

与视频 latent 沿帧维拼接：

$$\mathcal T_{\mathrm{total}}=[\mathbf z_v^{(t)};\mathbf z_s]$$

不同 anchor view 通过 3D RoPE 的 frame-axis 位置区分。该策略沿用预训练视频模型的上下文接口，而不是为锚点另建大型 cross-attention 分支。

### 3.2 View pose injection

生成帧与 anchor views 的 pose 统一编码并广播到空间网格：

$$\mathbf z_{\mathrm{pose}}\in\mathbb R^{(f'+f_s)\times h\cdot w\times d}$$

在 self-attention 前相加：

$$\mathcal T_{\mathrm{total}}\leftarrow\mathcal T_{\mathrm{total}}+\mathbf z_{\mathrm{pose}}$$

因此模型可依据统一世界坐标把第一人称轨迹与正确的 anchor 对齐，即使初始帧与 anchor image 没有像素重叠。

### 3.3 Evolution prompt 与局部 mask

第 $j$ 个 prompt 的 attention mask 为：

$$\mathcal M(q,k_j)=\begin{cases}0,&q\in\mathbf z_v\ \text{或}\ q\in\mathbf z_s^{(j)}\\-\infty,&\text{其他}\end{cases}$$

这样 prompt 可作用于全部生成视频 token，但只和对应 anchor image 交互，避免多个锚点文本互相污染。形式上每个 anchor 有独立 $\mathbf t_i$；但受训练数据限制，当前实现实际令 $t_1=\cdots=t_n$，即同一样本的所有锚点共享全局演化描述。这是能力声明与当前实现之间必须明确区分的地方。

---

## 4. 四阶段渐进训练

![](assets/AnchorWorld%20-%20可定制具身第一人称世界模拟/figures/03-progressive-training.png)

> 图 3：Stage I 学第三人称完整动作，Stage II 切换到第一人称头部视点，Stage III 加静态 anchor image/pose，Stage IV 再加入 evolution prompt。火焰表示对应阶段训练的模块。来源：论文 Figure 3。

| 阶段 | 目标 | 数据规模 | Steps | GPU hours |
|---|---|---:|---:|---:|
| I | Exocentric motion | 内部 200K + MultiCamVideo 101K | 30K | 600 |
| II | Egocentric motion | Ego-Exo4D + LEMMA 100K | 15K | 300 |
| III | Static anchor customization | 25K | 10K | 253 |
| IV | Dynamic anchor evolution | 25K + 10K | 10K | 253 |

统一设置为 16 张 80GB GPU、batch 16、AdamW、学习率 $10^{-4}$。视频为 480p，保留原始宽高比，timestep shift 15；pose 与 anchor 条件各以 5% 概率独立 dropout。推理使用 50 denoising steps、CFG scale 5。

动作与 anchor pose 通过 GVHMR 估计并统一到全局坐标；动作初始姿态原点化、水平朝向 canonicalize。Ego-Exo4D 鱼眼视频先去畸变和增亮；evolution prompt 由 Qwen3-VL-32B-Instruct 自动标注。anchor 图中若含玩家，训练时不 inpaint，以避免低分辨率修补伪影；推理可使用干净 anchor image。

---

## 5. 实验设置

### 5.1 Baseline

- **PlayerOne**：主要动作控制 baseline；因官方代码不可用，作者在 Wan2.2 TI2V 5B 和同一数据上重实现。
- **PlayerOne-Scene**：加入本文 anchor-view 注入。
- **CaM-UE / CaM-Ego**：Context-as-Memory，分别在官方 UE 与本文第一人称数据上训练。
- 动态场景没有同设定先例，因此把 evolution prompt 追加到静态 baseline 的全局文本。

### 5.2 测试集与指标

Ego Static、UE/CineScene、Ego Dynamic 各 100 个序列，另有真实世界定性集。动作准确性用 MegaSaM 估计相机轨迹并报告 ATE/RTE/RRE；第三人称动作报告 WA-MPJPE/PA-MPJPE；场景一致性用 GIM Matched Pixels、CLIP-V、PSNR、SSIM、LPIPS；动态文本用 VideoAlign Text Alignment；总体质量用 VBench。

---

## 6. 主要结果

### 6.1 静态与动态场景

| 设置 | 方法 | Mat. Pix.(K) ↑ | CLIP-V ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ATE ↓ | RTE ↓ | RRE ↓ | TA ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ego Static | CaM-Ego | 4379.4 | 0.872 | 15.16 | 0.554 | 0.515 | 0.125 | 0.032 | 3.207 | – |
| Ego Static | **Ours** | **4493.4** | **0.885** | **16.06** | **0.578** | **0.470** | **0.112** | **0.029** | **3.145** | – |
| Ego Dynamic | CaM-Ego | 4459.0 | 0.871 | 14.57 | 0.501 | 0.574 | 0.083 | 0.016 | 1.636 | 0.385 |
| Ego Dynamic | **Ours** | **4634.6** | **0.899** | **16.37** | **0.555** | **0.486** | **0.048** | **0.013** | 1.346 | **0.717** |

动态集上，AnchorWorld 除 RRE 外均最佳；RRE 最佳是 CaM-UE 的 1.230，AnchorWorld 为 1.346。UE/CineScene 上，AnchorWorld 的 Mat. Pix. 4555.1 和 RRE 1.656 最佳，但 CLIP-V 0.851 略低于 CaM-UE 0.852，VBench 0.769 略低于 CaM-Ego 0.770。论文总体优势成立，但不是所有表格单元都第一。

### 6.2 混合视角监督消融

| 变体 | ATE ↓ | RTE ↓ | RRE ↓ |
|---|---:|---:|---:|
| w/o Stage I | 0.125 | 0.033 | 3.532 |
| w/o Head Pose | 0.123 | 0.032 | 3.806 |
| Joint Stage I & II | 0.123 | 0.030 | 3.372 |
| **完整模型** | **0.112** | **0.030** | **3.187** |

第三人称预训练和头部 pose 都必要。去掉 Head Pose 时旋转误差最差，说明完整人体动作并不能替代相机视点。分阶段训练优于直接混合两种视角，表明先学可见的完整身体关系、再适配 egocentric sensorium 更稳定。

### 6.3 动作表示与 Spatial Pose Attention

第三人称动作控制中，本文 WA-MPJPE/PA-MPJPE 为 **74.57/28.01**；仅关节位置为 90.47/38.71，3D Pose Attention、Cross-Attention Fusion、In-Context Frame Concat 都超过 160/74。加入 axis-angle 朝向和帧级 Spatial Pose Attention 是显著收益来源。

### 6.4 Anchor 设计与数量

移除 anchor pose 或 anchor-view RoPE 会降低 Matched Pixels、CLIP-V、PSNR 和 LPIPS；完整模型在 Ego Static 的 Mat. Pix. 为 4493.4，w/o pose 为 4401.7，w/o RoPE 为 4395.2。anchor 数从 1/2/3 增加时，Mat. Pix. 从 4074.94→4152.91→4233.59，说明多视角约束总体有效，但 SSIM/LPIPS 从 1 到 2 个时略有非单调。

### 6.5 视野外演化

![](assets/AnchorWorld%20-%20可定制具身第一人称世界模拟/figures/04-out-of-sight-evolution.png)

> 图 6：通过删除或复制人体动作片段改变转向时刻。同一 anchor 中的人物在被看见之前仍按 evolution prompt 推进；较晚转向时，首次看到对象已处于更晚状态。来源：论文 Figure 6。

该实验是本文最有辨识度的证据：动态不是在对象进入画面后才开始，而是锚点局部状态在视野外按时间演化。不过当前所有 anchor 共享同一 evolution description，因此“多个独立局部事件并行演化”仍未被证明。

---

## 7. 与 Astra、DecMem、GIM-World 的关系

Astra 的动作接口更通用，覆盖相机、机器人和键鼠，但人体具身结构较弱；AnchorWorld 以 22 关节 6D 动作为核心，并用第三人称监督弥补第一人称身体不可见。DecMem/GIM-World 解决长期自动记忆，而 AnchorWorld 的 anchor views 是用户主动指定的外部世界状态：RGB 定义外观，pose 定义位置，文本定义演化。它目前只生成 77 帧，尚未具备 DecMem 的分钟级持续探索；未来最自然的组合是用 anchor 定义“必须存在的世界事实”，用 DecMem/GIM 维护访问历史和几何一致性。

---

## 8. 局限与失败案例

### 8.1 作者明确局限

1. **Long-Term Exploration**：当前主要是短视频，未解决实时自回归交互、大范围状态维护和持续环境更新。
2. **Open World**：训练场景有限，开放世界泛化不足，需要更大规模、多环境数据。
3. **Diverse Dynamic Scenarios**：动态数据主要记录同一人类活动，所以当前实现所有 anchor 共用 evolution prompt；天气、物体状态、复杂多主体和独立局部事件尚未覆盖。

### 8.2 明确失败案例

附录 Figure 14 展示两类失败：复杂局部纹理无法保持精细一致；快速第一人称运动导致模糊和手部质量下降。作者将前者部分归因于 Wan2.2 TI2V 5B VAE 的 16×空间下采样，后者还受到训练数据原生 motion blur、手部出视野和遮挡影响。

### 8.3 个人判断

AnchorWorld 把“世界模型”从被动延续画面推进到可指定局部状态，但其锚点仍是视图而非对象/场景图：当多个 anchor 内容互相矛盾，或视图覆盖之间存在几何不一致时，模型没有显式冲突解决机制。论文的 scene coherence 实验也显示，不兼容风格或镜像 pose 会造成墙面等区域不连续。这说明模型内部有世界一致性偏好，但尚没有可验证、可编辑的显式世界状态。

---

## 9. 复现与实践建议

1. 使用 Wan2.2 TI2V 5B，严格按 Stage I→II→III→IV 训练；联合 Stage I/II 或 III/IV 都被消融证明较差。
2. 动作使用 22 个关节的 3D 位置+axis-angle，不含手；先完成 canonicalization 和头部相机对齐，再进入编码器。
3. 验收混合视角时至少复现 w/o Stage I、w/o Head Pose 与 Joint I&II 三组相机误差。
4. anchor image token 沿帧维拼接，anchor pose 在 self-attention 前相加，evolution prompt 走带局部 mask 的 cross-attention；三者不要混成单一 embedding。
5. 先在 1 个 anchor 上验证无重叠视角定位，再扩到 3 个 anchor；同时测试不兼容 anchor，避免只评估容易的重叠视图。
6. 推理为 77 帧、480p、50 steps、CFG 5。训练总计约 1406 GPU-hours（600+300+253+253），不含数据标注与 GVHMR 预处理。
7. 官方数据集已发布，但截至记录日未提供明确代码/权重仓库；GVHMR 估计质量、多人主体筛选和 Qwen prompt 标注是主要复现变量。

---

## 10. 个人启发与后续问题

- 能否把 anchor 从视图升级为对象级状态，并显式处理锚点冲突？
- 多个独立 evolution prompts 如何在全局时间轴上同步，而不互相污染？
- DecMem/GIM 的自动记忆与用户锚点冲突时，如何设计可解释的优先级？
- 手部动作是否可由专门的局部高分辨率分支补足，而不是继续依赖 16× VAE latent？
- 长程 rollout 中，锚点状态是否需要被动作反向更新，例如玩家移动、拿走或破坏物体？

---

## 11. 材料来源

- 官方论文：https://arxiv.org/abs/2606.07326
- 项目页：https://yuli0103.github.io/AnchorWorld/
- 官方数据：https://huggingface.co/datasets/lyabc/anchorworld-dataset-full
- 代码：论文未提供明确仓库链接
- 本地 PDF 与逐图溯源：`assets/AnchorWorld - 可定制具身第一人称世界模拟/sources.md`
