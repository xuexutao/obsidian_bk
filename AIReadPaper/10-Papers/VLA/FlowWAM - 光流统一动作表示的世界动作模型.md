---
type: paper-note
status: complete
domain: VLA
paper: "FlowWAM: Optical Flow as a Unified Action Representation for World Action Models"
year: 2026
arxiv: "2607.13017"
doi: null
source: "https://arxiv.org/abs/2607.13017"
project: "https://flow-wam.github.io/"
code: "https://github.com/YixiangChen515/FlowWAM"
model: "https://huggingface.co/YixiangChen/FlowWAM"
tags:
  - VLA
  - WAM
  - World-Action-Model
  - Optical-Flow
  - Video-Generation
  - Flow-Matching
  - Robot-Learning
created: 2026-10-25
updated: 2026-10-25
---

# FlowWAM：以光流作为世界动作模型的统一动作表示

## 基本信息

- **论文标题**：FlowWAM: Optical Flow as a Unified Action Representation for World Action Models
- **作者**：Yixiang Chen*、Peiyan Li*、Yuan Xu、Qisen Ma、Jiabing Yang、Kai Wang、Jianhua Yang、Dong An、He Guan、Gaoteng Liu、Jianlou Si、Jun Huang、Jing Liu、Nianfeng Liu、Yan Huang†、Liang Wang†（*共同一作，†通讯）
- **机构**：中国科学院自动化研究所 NLPR、中国科学院大学人工智能学院、FiveAges（五季）、MBZUAI、阿里巴巴
- **arXiv**：[2607.13017](https://arxiv.org/abs/2607.13017)，v1，2026-07-14，cs.RO / cs.CV
- **项目页**：https://flow-wam.github.io/
- **代码**：RoboTwin 策略版 https://github.com/YixiangChen515/FlowWAM ；WorldArena 世界模型版 https://github.com/YixiangChen515/FlowWAM_WorldArena
- **权重与数据**：https://huggingface.co/YixiangChen/FlowWAM ；https://huggingface.co/datasets/YixiangChen/FlowWAM_RoboTwin
- **骨干与规模**：视频生成器为 Wan2.2-TI2V-5B（VAE 与 UMT5-XXL 冻结），外接约 780M 参数的 AdaLN Action Expert；全部训练在 32 张 H100（4 节点）上完成
- **核心结果**：RoboTwin 2.0 Clean 92.94% / Random 92.14%；WorldArena EWMScore 63.71，Trajectory Accuracy 64.26（相对次优 +18.4%）；真机 7 任务平均 75.7%（π0.5 为 61.4%，Motus 为 57.1%）

## 一句话结论

FlowWAM 的核心主张是：**WAM 里的“动作”不必是数值 token，也不必是抽象的 latent action，而可以直接是一段 HSV 编码的光流视频**。光流与 RGB 同格式，所以能共用同一个 VAE 和同一套 DiT block，构成「RGB 流 + Flow 流」双流联合扩散；去噪 flow 就是策略模式（再由 Action Expert 把 DiT 中间特征解码成关节动作），给定 flow 只去噪 RGB 就是世界模型模式，而 flow 又能从无动作标签的人类第一视角视频里直接提取，于是预训练、控制、世界建模三件事被一个表示统一起来。

> **我的总体评价：★★★★☆。** 思路干净、接口优雅，世界模型侧的证据（同一生成器只换条件：text 49.31 → 数值动作 54.18 → raw (u,v) 56.72 → mask 57.84 → HSV flow 65.23）是全文最有说服力的部分。但策略侧对最强 WAM 基线 Fast-WAM 的领先只有 +1.06 / +0.36 个点，而且去掉 EgoDex 预训练后 FlowWAM（82.40 / 80.80）低于所有 WAM 基线，正文中“flow 本身贡献了大部分收益”的说法与表格并不一致；RoboTwin 上的 flow 监督来自仿真器回放渲染的 robot-only 视频，属于较“干净”的特权信号，真机侧如何获得 flow 目标、推理延迟多少，论文都没有交代。

---

## 1. 研究背景与问题定义

### 1.1 研究问题：WAM 需要什么样的动作表示

World Action Model（WAM）是 2025—2026 年与 VLA 并行发展的一条路线：VLA 继承的是图文预训练的**语义先验**，WAM 继承的是视频生成器的**运动与时序动态先验**。对机械臂操控来说，任务成败不仅取决于“找对物体”，还取决于机器人和环境逐帧连续的运动，所以视频先验在理论上很契合。

然而一旦要把视频生成器拿来做控制，就绕不开一个接口问题：**动作应该以什么形式进入视频生成器？** 作者给出了理想动作表示的两条标准：

1. **格式对齐**：要符合预训练视频生成器的视觉输入格式，才能最大程度继承其先验；
2. **运动充分**：要保留跨帧的稠密运动线索，才能支撑精确控制。

### 1.2 现有方法的瓶颈

作者把已有 WAM 的动作表示分成三类，并逐一指出它们只满足其中一条标准：

- **数值动作 token**（DreamZero、Cosmos Policy、UWM、Fast-WAM 等）：精确，但与像素空间存在 *pixel-action gap*，而且不同机器人动作空间不一样，跨本体迁移困难。
- **学习得到的 latent action**（LAPA、Motus 等）：从帧间变化中学习，不依赖具体机器人，但它是抽象编码，未必保留控制所需的、空间上落地的稠密运动信息。
- **图像空间动作**（action ray map、embodiment mask、多视角 action image 等）：把控制信号渲染成图像，缩小了模态差距，但这些条件**大多是静态的空间提示**，只告诉生成器“动作发生在哪里”，没有告诉它“每个可见部件跨帧怎样移动”，因此仍然是帧级注入的条件，而不是与未来视频共同演化的时间稠密动作表示。

由此作者提出的开放问题是：**能否找到一种视频原生（video-native）、保留稠密跨帧运动、同时又能解码成可执行控制的动作表示？**

![](assets/FlowWAM%20-%20光流统一动作表示的世界动作模型/figures/01-teaser.png)

> 图 1：FlowWAM 的动机总览。左上展示 Flow Image 的构造：相邻两帧经光流估计得到逐像素位移，再经 HSV 色轮编码（色相 = 方向，饱和度 = 幅值）变成一张“光流图像动作”；右上对比三种动作表示——数值动作存在像素—动作鸿沟，图像空间动作静态且稀疏，而 Flow Image 是稠密的视觉运动；左下表示可以在无动作标签的第一视角数据上只用 $\mathcal{L}_{video}$ 学习运动先验；右下是双模式：噪声 flow 用于动作预测（策略模型），干净 flow 作为动作条件生成未来视频（世界模型）。
> 来源：论文 Figure 1，第 2 页，https://arxiv.org/abs/2607.13017

### 1.3 本文核心贡献

1. 首次（作者自称）把**光流作为 WAM 的统一动作表示**：它可以被预测、可以作为条件、也可以从无动作标签的第一视角视频中学习；
2. 提出 **FlowWAM 双流扩散框架**：RGB 与 flow 两路视频共享一个预训练视频生成器，同时支持操控的动作预测与世界建模的视频生成；
3. 在两种模式下验证：RoboTwin 2.0 上 Clean 92.94% / Random 92.14%，超过 VLA 与 WAM 基线；WorldArena 上 EWMScore 63.71 达到 SOTA，动作条件轨迹建模显著提升。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

FlowWAM 的输入是一张参考观测图像 $I_0$、一条语言指令 $\tau$，策略模式下还有当前本体状态（14 维关节位置 qpos）。模型联合建模两段视频：RGB 视频 $\mathcal{V}$ 和光流视频 $\mathcal{F}$。

- **策略模式**：输入 $I_0$、$\tau$、qpos；输出未来 RGB 视频、未来 flow 视频，以及由 Action Expert 解码出的 $N$ 步动作块 $\hat{a}_0\in\mathbb{R}^{N\times d_a}$（RoboTwin 上 $d_a=14$，$N=32$）。
- **世界模型模式**：输入 $I_0$、$\tau$ 和一段目标 flow 序列（代表“想要执行的动作”）；输出与该运动一致的未来 RGB 视频。

一个容易被忽略但很关键的工程假设：**flow 并不是全场景光流，而是“只有机器人在动”的光流**。RoboTwin 数据中，作者用 URDF 在 SAPIEN 里回放记录的关节动作，把机器人单独渲染在静态背景前，再对相邻 robot-only 帧跑 RAFT，从而去掉背景、光照和物体运动的干扰。世界模型模式下，给定关节动作轨迹时也需要走这条“回放渲染 → RAFT → HSV”的路径才能得到条件 flow；论文正文没有明说这一点，但 WorldArena 代码仓库中包含 `robot_only_renderer.py`（SAPIEN robot-only renderer）和 `video_flow_codec_pipeline.py`，可以印证这一流程（个人根据代码结构推断）。

另外，flow 只在 **头部相机** 区域提供监督。RoboTwin 输入把头部相机（320×256）放在上方、左右腕部相机降采样后拼在下方，形成 320×384 的 T 形拼图；腕部区域的 flow 由常数占位填充，因为预处理流程拿不到腕部视角的 robot-only flow。

### 2.2 关键符号和目标函数

**Flow 的 RGB 编码。** 光流场 $f_t\in\mathbb{R}^{H\times W\times 2}$ 记录逐像素位移 $(u,v)$，用 HSV 色轮编码 $\phi$ 转换为 RGB 图像：

$$F_t=\phi(f_t):\quad H=\frac{\operatorname{atan2}(v,u)+\pi}{2\pi},\quad S=\frac{\|f_t\|}{m},\quad V=1$$

色相表示方向，饱和度表示幅值，亮度恒为 1——所以“不动”的像素是纯白，这就是论文图里 flow 帧大面积白底、只有机器人区域有颜色的原因。$m$ 是幅值归一化常数，实现上先把幅值截断在 25 px，再把小于 0.5 px 的位移置零以压制近静止区域的噪声。作者强调该编码在给定归一化下**可逆**，$\phi^{-1}(F_t)$ 能恢复数值光流；严格来说，截断与阈值会损失超出 25 px 和低于 0.5 px 的信息，可逆性是“在有效区间内”成立的。

这一编码的价值在于：$F_t$ 与场景帧 $I_t$ **格式完全相同**，可以直接走同一个 VAE 编码器和视频生成器，不需要单独的动作 tokenizer。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/FlowWAM%20-%20光流统一动作表示的世界动作模型/figures/02-method-pipeline.png)

> 图 2：FlowWAM 总体架构。上半部分：参考图像和语言指令进入双流 DiT，RGB 分支与 Flow 分支各有独立的 patch embed 和 RoPE，在共享的 Joint Self-Attn + FFN 中交互；输出的 RGB 视频与 Flow 视频（实际是 DiT 各层隐状态）被 Action Expert 通过 cross-attention 读取，以 flow matching 生成动作。左下是两阶段训练：Stage 1 在无动作标签视频上只训练 DiT（$\mathcal{L}_{video}$），Stage 2 接上 Action Expert 在机器人数据上联合优化（$\mathcal{L}_{video}+\mathcal{L}_{action}$）。右下是两种推理模式：策略模式两路都去噪，世界模型模式 flow 给定、只去噪 RGB。
> 来源：论文 Figure 2，第 4 页，https://arxiv.org/abs/2607.13017

整个 pipeline 可以概括为四步：

1. 把参考帧 $I_0$ 分别作为 RGB 流和 flow 流的首帧条件（flow 流首帧是什么，论文只说“参考观测的干净 VAE 编码”，个人理解即零运动的白帧或参考帧对应的 flow 编码，论文未细化）；
2. 双流 DiT 在共享 Transformer block 中联合去噪 RGB latent 和 flow latent；
3. 策略模式下，Action Expert 逐层 cross-attend 到 DiT 的 RGB+flow 隐状态，以 flow matching 生成 32 步动作块，执行 25 步后重新规划；
4. 世界模型模式下，flow latent 固定为目标运动的干净编码，只生成 RGB 未来。

### 3.2 统一的视频—光流双流生成

**共享 VAE 与 Transformer。** RGB 帧和 flow 帧分别用同一个冻结的 VAE 编码器 $E$ 编码，得到 $z=E(\mathcal{V})$ 和 $z^f=E(\mathcal{F})$。两条流唯一的专属参数是**各自的 patch embedding 层和输出 head**，所有 Transformer block 共享。Flow 分支的 patch embed 与 head 从 RGB 分支深拷贝初始化，并在 patchify 之后加入一个可学习的 flow-token embedding 作为流身份标识。作者的解释是：既然 flow 帧与 RGB 帧走同一个 VAE，用 RGB 权重初始化 flow 分支能提供兼容的图像 latent 先验，比随机初始化收敛更快。

**联合注意力与独立 RoPE。** 在每个 self-attention 层，RGB token 和 flow token 拼接后做联合注意力，再拆回各自的流；RoPE 对两条流**分别独立施加**。这意味着同一时空位置的 RGB token 与 flow token 拥有相同的位置编码，注意力可以天然地在“同一像素位置的外观与运动”之间建立对应。作者强调共享 block 保证模型完整保留预训练视频生成器的生成能力，这本质上是 MM-DiT 式的“双流共享权重 + 联合注意力”设计，只不过两条流是同构的视频模态，因此连 block 参数也可以共享。

**两种推理模式只差一个开关。** 两种模式中，每条流的首个 latent 帧都固定为参考观测的干净 VAE 编码（image-to-video 条件），后续帧通过迭代去噪产生：

- **策略模式**：两条流剩余帧都从高斯噪声初始化并联合去噪，DiT 同时合成未来 RGB rollout 与对应 flow 视频，Action Expert 读取中间特征解码动作；
- **世界模型模式**：flow latent $z^f$ 被设成目标运动轨迹的干净 VAE 编码，并在采样全程保持不变，只有 RGB latent 从噪声初始化，模型据此渲染与指定运动一致的未来视频。

这一设计的优雅之处在于：**策略与世界模型的差异被还原为“flow 是未知量还是观测量”**，同一套权重、同一个前向图，只是条件掩码不同。这和 Cosmos Policy、Motus 等用条件掩码切换模式的思路一脉相承，但 FlowWAM 把被切换的那一路做成了与 RGB 同构的视频。

### 3.3 从 flow latent 预测动作：Action Expert

Flow 是图像空间的运动计划，要落到机器人关节空间还需要一个“翻译器”。FlowWAM 的 Action Expert 是一个约 780M 参数的 AdaLN 扩散 Transformer：30 层（与 Wan2.2 视频 DiT 深度一致）、hidden 1024、16 头、FFN 4096。每个 block 包含三种注意力：

1. 对 action token 的 self-attention；
2. 对**配对视频 DiT 层**输出的 RGB+flow 拼接隐状态的 cross-attention——第 $l$ 层 Action Expert 读第 $l$ 层视频 DiT 的特征，而不是对 VAE latent 重新 patchify；
3. 对 T5 指令 token 的 cross-attention；本体状态（归一化 14 维 qpos）被投影到指令嵌入维度，作为一个 proprio token 追加到 T5 上下文里。

预测头直接输出动作块的 flow-matching 速度，推理时由 flow-matching 采样器积分。HuggingFace 模型卡给出的解码配置还包括 `action_pred_target=velocity`、`action_pos_mode=rope`、`proprio_mode=text`、`action_snr_shift=5.0`，与论文描述一致。

**随机 latent 条件化（stochastic AE conditioning）。** 这里有一个训练—推理分布错配：训练时 Action Expert 可以读取由**干净**视觉 latent（真实帧的 VAE 编码）算出的隐状态，推理时读取的却是双流模型**生成**的 latent，难免带有残余去噪误差。为对齐两者，作者在比例 $p=0.5$ 的训练步上，给送入视频生成器的 latent 混入噪声，并把噪声水平作为辅助嵌入传给 Action Expert：

$$\tilde{z}^f=(1-\sigma)z^f+\sigma\epsilon_f,\quad\tilde{z}=(1-\sigma)z+\sigma\epsilon_r,\quad\sigma\sim\mathcal{U}[0,1]$$

其中 $\epsilon_f,\epsilon_r\sim\mathcal{N}(0,I)$ 相互独立，其余 $1-p$ 的训练步看纯净 latent。直觉上，这是让 Action Expert 学会“在视频计划不完美时依然能读出动作”，类似于 Diffusion Forcing 或 noise augmentation 的思想。消融显示去掉它会让成功率从 89.8% 掉到 82.1%（见 5.3 节）。

### 3.4 训练目标与损失函数

**视频生成损失。** 双流 DiT 用 flow matching（rectified flow）目标训练。每步对两条流采样**同一个**时间步 $t\sim\mathcal{U}[0,1]$，构造 $z_t=(1-t)z_0+t\epsilon$、$z^f_t=(1-t)z^f_0+t\epsilon'$，模型预测速度场 $v_\theta(z_t,t)$ 与 $v^f_\theta(z^f_t,t)$（按 rectified flow 的常规约定目标速度为 $\epsilon-z_0$，论文只写“target velocities”）。视频损失是两条流 MSE 的加权和：

$$\mathcal{L}_{video}=(1-\lambda_f)\mathcal{L}_{RGB}+\lambda_f\mathcal{L}_{flow}$$

实现中 $\lambda_f=0.1$。值得注意的是：全文把 flow 称为“主动作模态”，但它在视频损失里只占 10% 的权重——这说明作者更想保护 RGB 生成先验不被破坏，flow 的学习更多依赖共享 block 与联合注意力的“搭便车”，以及下面的运动重加权。为匹配 image-to-video 推理分布，每条噪声流的首个 latent 帧被替换为干净条件 latent，并在该帧屏蔽损失；此外对条件帧施加小幅高斯扰动（ref-aug strength 0.1），模拟自回归 rollout 中最新观测的残余误差。

**运动感知重加权。** 操控场景的 flow 在空间上非常稀疏，运动集中在机器人本体、被操作物体和接触区域，大面积静态背景会主导 $\mathcal{L}_{flow}$。作者用 flow latent 相对参考帧的通道平均偏差来计算逐位置权重：

$$w_{motion}=1+\alpha\cdot\frac{\langle|z^f-z^f_{ref}|\rangle_c}{\max\langle|z^f-z^f_{ref}|\rangle_c}$$

$\langle\cdot\rangle_c$ 表示在 latent 通道上取平均，$\alpha$ 控制增强强度（Stage 2 中 $\alpha=2.0$），即运动最强的位置权重为 3，静止位置权重为 1。这是一个很朴素但有效的前景加权，消融中去掉它成功率从 89.8% 降到 83.9%。

**动作损失与总目标。** 有动作标签时，Action Expert 在独立的 flow-matching 调度下从 DiT 各层隐状态预测真实动作块，得到 $\mathcal{L}_{action}$，总目标为：

$$\mathcal{L}=\mathcal{L}_{video}+\lambda_a\mathcal{L}_{action}$$

其中 $\lambda_a=1.0$。

**两阶段训练。**

- **Stage 1：无动作运动预训练。** 在 EgoDex（人类第一视角操作视频，无机器人动作标签）上只用 $\mathcal{L}_{video}$ 训练双流 DiT。视频从 30 fps 下采样到 15 fps，缩放到 320×256，clip 长度从 {17, 33, 49, 65, 81} 帧中分桶采样，学习率 5×10⁻⁵，每卡 batch 1，不做任务级过滤。EgoDex 的 flow 直接在相邻 RGB 帧之间用 RAFT 计算（即全场景 flow，包含人手和物体运动）。
- **Stage 2：联合具身训练。** 从 Stage 1 检查点初始化视频生成器，接上 Action Expert，在 RoboTwin 2.0 上优化完整目标。每个 chunk 生成 9 帧像素帧，并解码出 32 步动作（像素帧之间时间跨度为 4 步）；学习率 1×10⁻⁴，每卡 batch 16；可训练参数包括完整 DiT、flow 分支和 Action Expert。

作者对两阶段的解释是：**视觉运动建模与本体相关的动作解码被解耦**，人类视频中学到的运动先验可以在不改变 flow 表示的情况下迁移到机器人控制。这里有一个潜在的分布错位值得指出（个人判断）：Stage 1 的 flow 是全场景 flow（人手 + 物体），Stage 2 的 flow 是仿真回放得到的 robot-only flow（只有机械臂），两者的“flow 语义”并不完全相同，论文没有讨论这种错位如何影响迁移。

### 3.5 推理流程与复杂度

RoboTwin 评测时，视频去噪 25 步、动作去噪 50 步，每次预测 32 步动作块，采用 execute-and-replan 窗口 25 步。推理时系统以 WebSocket server 形式部署，RoboTwin 环境作为客户端调用。

复杂度方面，每次重规划都需要：5B 视频 DiT 在双流（token 数约为单流两倍）上去噪 25 步，再加上 780M Action Expert 去噪 50 步（每步都要 cross-attend 30 层视频特征）。**论文没有报告任何推理延迟、控制频率或显存占用**。对比同期 Fast-WAM 明确主张“测试时跳过未来想象”以换取实时性，FlowWAM 恰恰走了相反方向——显式生成完整的未来 RGB+flow 计划。在真机部署上这很可能是一个实际瓶颈，但论文没有给出数据。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

| 用途 | 数据 | 规模/协议 | flow 来源 |
|---|---|---|---|
| Stage 1 预训练 | EgoDex（人类第一视角操作视频） | 15 fps，320×256，帧分桶 {17…81} | 相邻 RGB 帧 RAFT，全场景 |
| Stage 2 策略训练 | RoboTwin 2.0（aloha-agilex 双臂，50 任务） | 每任务 Clean 50 条 + Random 500 条，合并训练 | SAPIEN URDF 回放 robot-only 渲染 + RAFT，仅头部相机 |
| 世界模型评测 | WorldArena（RoboTwin 2.0 子集） | 121 帧、24 fps rollout，16 项指标 | 代码中通过 robot-only renderer 获得（论文未明说） |
| 真机 | Franka Research 3（单臂，ZED 2i）+ ARX 5（双臂，3×RealSense D405） | 7 任务，每任务 100 条遥操作示范，10 次随机位姿试验 | **论文未披露** |

### 4.2 Baseline 与评价指标

**策略基线**分两类：直接动作 VLA（π0.5、X-VLA）与统一 WAM（Motus、GigaWorld-Policy、X-WAM、Fast-WAM）。作者说明所有 WAM 基线都按各自的大规模视频/无标签数据预训练协议评测。附录中对它们的定位也很清楚：Motus 是与 FlowWAM 最接近的策略侧对照（同样利用无标签视频和光流衍生的运动线索，但动作在学习得到的 latent action 空间里）；Fast-WAM 用来隔离视频先验的作用（训练时联合视频，测试时跳过未来想象，动作是数值 token）。

**世界模型基线**分两类：通用视频/基础世界模型（CogVideoX、Veo 3.1、Wan 2.6、Cosmos-Predict 2.5、ABot-PhysWorld）检验“仅有大规模视觉先验是否够用”；动作条件机器人世界模型又分数值动作条件（Cosmos-Predict 2.5 action、IRASim、Ctrl-World）和图像空间动作条件（Vidar、Genie Envisioner、GigaWorld-1）。

**指标**：RoboTwin 报告每任务 100 次 rollout 的成功率；WorldArena 报告 16 项指标（归一化到 [0,100]）的算术平均 EWMScore，正文挑出 6 项：JEPA 相似度、Subject Consistency、Background Consistency、Trajectory Accuracy（用 SAM3 提取机械臂框后算 NDTW 对齐分数）、Depth Accuracy、Semantic Alignment。

### 4.3 实现细节

| 类别 | 设置 |
|---|---|
| 视频生成器 | Wan2.2-TI2V-5B；Wan2.2 VAE 与 UMT5-XXL 冻结（bf16） |
| Flow 分支初始化 | 深拷贝 RGB 的 patch embedding 与 head |
| Action Expert | 30 层 / 1024 / 16 头 / FFN 4096，约 780M，动作维度 14 |
| 输入（RoboTwin） | 单相机 320×256，T 形拼图 320×384；9 像素帧 / 32 动作步 |
| Flow | RAFT；幅值上限 25 px；噪声阈值 0.5 px |
| 优化 | AdamW（weight decay 10⁻²）；LR Stage 1 / 2 = 5×10⁻⁵ / 1×10⁻⁴；每卡 batch 1 / 16 |
| 损失权重 | $\lambda_f=0.1$，$\lambda_a=1.0$，$\alpha=2.0$，ref-aug 0.1 |
| 推理 | 视频 / 动作去噪步数 25 / 50；执行后重规划窗口 25 |
| 硬件 | 32×H100，4 节点（训练时长论文未披露） |

---

## 5. 实验结果

### 5.1 主要定量结果

#### 5.1.1 RoboTwin 2.0 策略

| 方法 | 类型 | Clean | Random |
|---|---|---|---|
| π0.5 | VLA | 42.98 | 43.84 |
| X-VLA | VLA | 72.88 | 72.84 |
| Motus | WAM | 88.66 | 87.02 |
| GigaWorld-Policy | WAM | 86.36 | 85.04 |
| X-WAM | WAM | 89.76 | 90.68 |
| Fast-WAM | WAM | 91.88 | 91.78 |
| FlowWAM w/o PT | WAM | 82.40 | 80.80 |
| **FlowWAM w/ PT** | WAM | **92.94** | **92.14** |

（50 任务平均，单位 %；论文 Table 1，第 6 页，逐任务结果见附录 Table 4。）

这张表需要细读，而不是只看最后一行：

- **相对 VLA 的优势非常明显**：即使不做预训练，FlowWAM 也比 X-VLA 高约 9.5 / 8 个点；π0.5 在 RoboTwin 2.0 这种 50 条小样本 Clean 设置下表现很差，部分任务接近 0。
- **相对最强 WAM 的优势很小**：带预训练的 FlowWAM 只比 Fast-WAM 高 1.06（Clean）和 0.36（Random）个点，比 X-WAM 高 3.18 / 1.46 个点。在每任务 100 次 rollout、单次运行、无置信区间的条件下，这个差距不足以支撑“明显领先”。
- **预训练才是关键**：EgoDex 预训练带来 +10.54（Clean）/ +11.34（Random）个点。没有它，FlowWAM 低于表中**所有** WAM 基线（比 GigaWorld-Policy 还低约 4 个点）。论文正文写“即使没有 ego 视频预训练，FlowWAM 已超过 VLA 基线，说明 flow 流本身贡献了大部分收益、预训练只是放大器”——前半句成立，后半句与表格不符：相对 WAM 同类，没有预训练的 flow 表示并不占优。
- **Random 增益“明显更大”的说法也偏强**：+11.34 对 +10.54，差距不足 1 个点。

逐任务看，FlowWAM 的优势集中在 Move Can Pot（96/97，Fast-WAM 为 90/88）、Place Mouse Pad（96/94，Fast-WAM 为 83/89）、Hanging Mug（65/68，全场最高但仍是难点）这类需要精确放置或复杂接触的任务；在 Pick Diverse Bottles 上反而不如 X-WAM（90/91 对 91/92）。

#### 5.1.2 WorldArena 动作条件世界建模

| 方法 | 条件 | JEPA | SC | BC | TA | DA | SA | EWMScore |
|---|---|---|---|---|---|---|---|---|
| CogVideoX | Text | 94.84 | 80.97 | 88.38 | 34.79 | 91.09 | 89.70 | 58.79 |
| Veo 3.1 | Text | 57.97 | 75.82 | 91.67 | 11.36 | 74.27 | 83.79 | 57.77 |
| ABot-PhysWorld | Text | 90.36 | 80.56 | 89.02 | 31.50 | 71.99 | 89.58 | 62.63 |
| IRASim | Action | 93.72 | 72.82 | 78.69 | 35.92 | 93.50 | 89.43 | 56.15 |
| Ctrl-World | Action | 92.77 | 83.56 | 90.30 | 48.20 | 93.25 | 88.68 | 59.98 |
| GigaWorld-1 | Img-Act | 96.77 | 80.97 | 86.43 | 54.27 | 98.44 | 89.42 | 62.34 |
| **FlowWAM** | Flow | **97.14** | 82.46 | 89.97 | **64.26** | **98.97** | **89.93** | **63.71** |

（节选自论文 Table 2，第 7 页。）

最有诊断意义的是 **Trajectory Accuracy（TA）**：它衡量生成视频中的机械臂轨迹是否忠实地跟随给定动作。文本条件模型普遍很低（Veo 3.1 仅 11.36），数值动作条件有所改善（Ctrl-World 48.20），图像空间动作再进一步（GigaWorld-1 54.27），FlowWAM 达到 64.26，领先次优近 10 个点（相对 +18.4%）。Depth Accuracy 也是全场最高。作者的解释是：稠密 flow 给了生成器一个逐像素的运动场，生成器只需“对着它去噪”。

但 EWMScore 总分只比 ABot-PhysWorld（62.63）高 1.08、比 GigaWorld-1（62.34）高 1.37。看附录 Table 5 的 16 项完整指标能发现原因：FlowWAM 在画质指标 IQ / AQ 上只有 48.87 / 40.83，明显低于 Veo 3.1（65.57 / 48.79）、Wan 2.6（67.36 / 44.40）；Action Following（AcF，对不同动作指令是否产生不同 rollout）只有 3.50，所有方法在这一项上都很低。也就是说，FlowWAM 的总分优势几乎全部来自“动作跟随与几何”类指标，而非视觉质量。

### 5.2 定性结果

![](assets/FlowWAM%20-%20光流统一动作表示的世界动作模型/figures/05-qualitative-two-modes.png)

> 图 5：两种模式的定性可视化。上方黄色区域为策略模式：每帧主图是执行结果，左上小窗是预测的 RGB 与 flow，可以看到 flow 只集中在运动的机械臂上，沿着连贯的、指向目标的轨迹变化；下方绿色区域为世界模型模式：给定条件 flow（小窗），生成的 RGB rollout 跟随指定运动，物体身份与静态背景跨帧保持稳定。
> 来源：论文 Figure 5，第 9 页，https://arxiv.org/abs/2607.13017

从图中能直观看出 FlowWAM 中 flow 的实际形态：大部分像素是白色（零运动），只有机械臂所在的一小块区域着色。这其实也提示了 flow 的信息结构——在 robot-only 渲染的设定下，flow 视频接近于“带位移方向与幅值的机械臂运动剪影序列”。

### 5.3 消融实验

![](assets/FlowWAM%20-%20光流统一动作表示的世界动作模型/figures/04-ablation-flow-decodability.png)

> 图 4：消融与 flow 可解码性。(a) RoboTwin 上的策略侧设计消融；(b) WorldArena 上固定视频生成器、只替换动作条件；(c) 50 个 RoboTwin 任务中预测 flow 误差与策略成功率的散点图，Pearson $r=-0.81$。
> 来源：论文 Figure 4，第 9 页，https://arxiv.org/abs/2607.13017

**(a) 策略侧。** 完整 FlowWAM 为 89.8%。把 flow 换成数值动作向量降到 69.8%；把 HSV RGB flow 换成原始 $(u,v)$ 张量降到 72.3%；去掉运动感知重加权降到 83.9%；去掉随机 AE 条件化降到 82.1%。作者据此认为：真正弥合动作—视频鸿沟的是“RGB 格式的 flow 通道”本身，而非泛泛的视觉条件。

这组数字有两个值得思考的地方。第一，raw $(u,v)$ 只比数值动作高 2.5 个点，而 HSV 编码高 20 个点——**收益主要来自“格式对齐”而不是“稠密运动信息”本身**，这对理解 FlowWAM 很关键：同样的位移信息，如果不能被预训练 VAE 和 DiT 当作图像来处理，几乎发挥不出作用。第二，89.8% 与 Table 1 的 92.94 / 92.14 不一致，论文没有说明消融用的是哪个设置（是否带预训练、是否为任务子集），这降低了消融结果与主表的可比性。

**(b) 世界模型侧。** 固定视频生成器只改条件：纯文本 49.31，数值动作 54.18，raw $(u,v)$ 56.72，图像空间 mask 57.84，完整 RGB flow 65.23。这条单调上升的链条完整对应了作者在引言中的论证：文本对运动欠约束；数值动作在视觉空间之外；raw $(u,v)$ 稠密但格式不对；mask 是静态空间提示，不编码跨帧运动方向和幅值；只有 RGB flow 同时具备逐像素运动与视频原生格式。需要注意脚注说明：WorldArena 官方评测需要远程提交且没有本地 GT 视频，所以消融使用了自定义验证集划分，65.23 与 Table 2 的 63.71 不直接可比。

**(c) Flow 可解码性。** 作者在每个任务上新生成 50 条示范，用 RAFT 提取伪真值 flow，计算 FlowWAM 预测 flow 的误差，并与该任务的成功率做相关分析：flow 误差越低的任务成功率越高，Pearson $r=-0.81$。作者把这作为“策略收益确实来自 flow 预测质量，而非解码器的无关捷径”的直接证据。个人认为这个证据是相关性而非因果：难任务（如 Hanging Mug）本身的运动更复杂，flow 更难预测，成功率也更低，两者可能同时被“任务难度”驱动。更有力的验证应当是干预实验，例如用 GT flow 替换预测 flow 后看成功率上限，或人为向 flow 注入不同程度扰动看成功率的变化曲线。

### 5.4 泛化、效率与失败案例

![](assets/FlowWAM%20-%20光流统一动作表示的世界动作模型/figures/03-real-world-success.png)

> 图 3：真机成功率。左侧为 Franka 单臂四个任务（Stack Bowls、Place in Drawer、Put in Plate、Place Two Cups），右侧为 ARX 双臂三个任务（Fold Towel、Stack Bowls II、Clean Plate）；粉色 π0.5、绿色 Motus、黄色 FlowWAM。
> 来源：论文 Figure 3，第 8 页，https://arxiv.org/abs/2607.13017

真机各任务成功率（%，每任务 10 次）：

| 任务 | π0.5 | Motus | FlowWAM |
|---|---|---|---|
| Stack Bowls（Franka） | 80 | 80 | 90 |
| Place in Drawer（Franka） | 70 | 80 | 90 |
| Put in Plate（Franka） | 80 | 80 | 100 |
| Place Two Cups（Franka） | 40 | 30 | 60 |
| Fold Towel（ARX） | 40 | 30 | 40 |
| Stack Bowls II（ARX） | 70 | 70 | 90 |
| Clean Plate（ARX） | 50 | 30 | 60 |
| **平均** | 61.4 | 57.1 | 75.7 |

作者的结论是 FlowWAM 在每个任务上都最高，并且双臂任务上与基线的差距大于单臂任务，因为 flow 直接在图像空间表达两臂的耦合运动，而不是只通过低维关节状态。核对数据：Fold Towel 上 FlowWAM 与 π0.5 持平（都是 40%），并非“每个任务最高”；单臂平均领先 π0.5 约 17.5 个点，双臂平均领先约 10 个点，按与 π0.5 的差距算，双臂差距反而**更小**，只有对 Motus 时双臂差距（+20.0）才略大于单臂（+17.5）。另外每任务仅 10 次试验，10 个百分点的差距就是 1 次成功，统计上非常脆弱。

**失败与弱项**（论文没有专门的失败案例章节，以下为根据表格整理）：

- Hanging Mug 仍只有 65% / 68%，是 RoboTwin 上最难的接触任务之一；
- Fold Towel 这类可变形物体任务只有 40%，flow 在可变形物体上的表达（尤其是 robot-only flow 根本不包含物体运动）可能存在先天不足；
- WorldArena 的画质指标（IQ / AQ）偏弱；
- 效率：论文完全没有报告推理延迟。

---

## 6. 与相关工作的关系

把 FlowWAM 放到 WAM 演化脉络中看，它回答的是“**动作接口应该长什么样**”这个问题：

- **数值动作 token 路线**：[[DreamZero - 视频世界动作模型的零样本机器人策略]]、[[Cosmos Policy - 视频模型微调为机器人控制与规划策略]] 把动作 token 与视频 latent 共同生成；[[Fast-WAM - 无测试时未来想象的世界动作模型]] 在此基础上进一步提出测试时可以不想象未来。FlowWAM 与 Fast-WAM 是正相反的观点：前者认为显式生成稠密运动计划有价值，后者认为视频共训练的收益主要在表征层面、测试时可省略。两者在 RoboTwin 上只差约 1 个点，说明“是否显式想象未来”在这个 benchmark 上可能并不是决定性因素。
- **generate-then-infer 路线**：[[mimic-video - 超越VLA的视频动作模型]]、Vidar 先生成视频再用逆动力学模型（IDM）恢复动作。FlowWAM 的 Action Expert 本质上也是一个 IDM（HuggingFace 模型卡直接称其为 “flow-action IDM”），区别是它读取的是 DiT 逐层隐状态而不是解码后的像素，并且中间多了一路显式的 flow。
- **latent action 路线**：Motus 用光流衍生的 latent action 连接视觉动态与控制，是与 FlowWAM 最接近的对照。FlowWAM 相当于把 Motus 的“光流 → 隐变量”这一步拿掉，直接让光流本身成为被生成的模态。
- **图像空间动作路线**：action ray map、embodiment mask、GigaWorld-1 等。FlowWAM 属于这一类，但把静态空间提示换成了时间稠密的位移场。消融 (b) 中 mask 57.84 对 flow 65.23 是对这一改进最直接的支持。
- **模块化 WAM 研究**：[[OpenWAM - 模块化世界动作模型预训练]] 系统比较了视频骨干、视觉 latent、动作容量和信息流，同样选择 Wan2.2-TI2V-5B 作为世界流，并发现独立动作容量和训练期 world-to-action 信息流最关键。FlowWAM 的设计（独立 780M Action Expert + 逐层读取 DiT 特征）与 OpenWAM 的结论吻合，可以看作在“动作表示”这一维上补充了 OpenWAM 没有覆盖的变量。[[Long-WAM - 扩展世界动作模型的上下文]] 则从上下文长度切入，与 FlowWAM 的短 chunk（9 帧）设定形成对照。
- **光流用于操控**：ATM（稀疏点轨迹）、LangToMo、DAWN 等把 flow 或 pixel motion 当作辅助监督、规划变量或模块间桥梁。FlowWAM 的区别是把稠密 flow 放进视频生成器内部，作为原生的一路流。

世界模型侧的对照可参考 [[Ctrl-World - 可控多视角世界模型]]、[[Cosmos-Predict 2.5 - 视频基础模型驱动的物理世界模拟]]、[[ABot-PhysWorld - 物理对齐机器人操控世界模型]]；策略侧 VLA 对照见 [[π0.5 - 开放世界泛化视觉语言动作模型]]、[[X-VLA - 软提示跨具身视觉语言动作模型]]。

---

## 7. 局限与批判性评价

**方法层面的价值。** FlowWAM 最有价值的洞察是“**格式对齐比信息量更重要**”：raw $(u,v)$ 与 HSV flow 承载的信息几乎相同，但前者在策略和世界模型两侧都只带来很小提升，后者带来大幅提升。这说明 WAM 继承视频先验的关键，是让动作“看起来像视频”，让预训练的 VAE 和 DiT 能够直接处理它。这一结论对任何想把新模态接入视频生成器的工作都有借鉴意义。

**论文主张与证据之间的落差。**

1. **策略侧领先幅度被夸大**：对 Fast-WAM 只领先 +1.06 / +0.36，对 X-WAM 领先 +3.18 / +1.46；没有多随机种子或置信区间。去掉预训练后 FlowWAM 低于所有 WAM 基线，与“flow 本身贡献了大部分收益”的说法矛盾。
2. **消融与主表不可比**：策略消融的完整模型是 89.8%，与主表 92.94 / 92.14 都不一致，且未说明设置。
3. **真机描述不准确**：Fold Towel 与 π0.5 持平；双臂相对 π0.5 的差距并不比单臂大；每任务只有 10 次试验。
4. **可解码性证据是相关性**：$r=-0.81$ 可能被任务难度这一共同因素驱动。

**表示本身的适用边界（个人判断）。**

1. **robot-only flow 依赖特权信息**：RoboTwin 上的 flow 目标需要 URDF、关节动作回放和与相机外参一致的渲染，在仿真里免费，在真机上则需要精确的相机标定和机器人模型。论文没有说明真机的 flow 监督如何获得（是 robot-only 渲染还是全场景 RAFT）。代码中提供的 `FLOW_MODE=full_scene` 选项表明作者也试过全场景 flow，但论文没有报告对比。
2. **flow 只覆盖头部相机**：腕部视角用常数占位，意味着对精细接触最有用的近距离视角没有运动计划；Action Expert 要从 RGB 腕部特征和头部 flow 中自行推断。
3. **robot-only flow 不编码物体运动**：在可变形物体（叠毛巾 40%）或需要推理物体动态的任务上，flow 无法表达“物体应该怎样动”，这部分只能依赖 RGB 流。
4. **2D 投影的歧义**：光流是 3D 运动在图像平面上的投影，沿视线方向的运动、遮挡和自遮挡都会造成歧义；14 维关节动作要从 2D flow 中恢复，必然依赖 RGB 上下文与 qpos 共同消歧。
5. **Stage 1 与 Stage 2 的 flow 语义错位**：EgoDex 上是全场景 flow（人手 + 物体），RoboTwin 上是 robot-only flow，迁移机制没有分析。

**评测公平性（个人判断）。** WorldArena 代码仓库中包含 SeedVR2-3B 作为 stage-2 refiner（`inference/refiner/`，视频超分/修复模型），训练时还会把低分辨率参考帧做 BICUBIC 上采样以匹配推理时的退化。论文正文完全没有提到这个后处理模块。WorldArena 的画质与一致性类指标可能受它影响，在与其他世界模型比较时，是否所有基线都使用了类似后处理是一个需要澄清的问题。另外，世界模型模式下条件 flow 来自 URDF 渲染，等于把“机械臂在每一帧应该出现在哪里、怎样移动”几乎完整地交给了生成器，这让 Trajectory Accuracy（用 SAM3 检测机械臂框后做 NDTW）很难不高；它更像是“给定机械臂渲染的视频补全”，而不是从低维动作想象未来。这并不否定方法价值，但解读 TA 的大幅领先时需要考虑信号本身的信息优势。

**缺失信息**：推理延迟与控制频率、训练总时长与 GPU 小时、多随机种子统计、真机 flow 获取方式、消融实验设置、Stage 1 预训练规模的缩放曲线。

---

## 8. 复现与实践建议

**代码与权重可用性：较好。** 两个官方仓库分别覆盖 RoboTwin 策略和 WorldArena 世界模型，均包含训练与推理代码；HuggingFace 发布了 `flowwam_robotwin.safetensors`（DiT + flow_stream + action expert）、动作归一化统计 `flowwam_robotwin_action_norm_stats.npz` 和 `flowwam_worldarena_stage1.safetensors`；RoboTwin 训练数据（含 robot-only 视角）也已发布，许可证为 Apache-2.0。RoboTwin 评测以 additive policy package 的方式接入，不改动 RoboTwin 源码，并支持每 GPU 一个 server、多卡分片评测 50 个任务。

**复现成本评估：**

- **只做 RoboTwin 推理评测**：需要一张能跑 5B DiT 双流 + 780M Action Expert 的 GPU（显存需求未披露，按 Wan2.2-5B 推理经验，单卡 40GB 以上较稳妥），外加独立的 RoboTwin conda 环境。50 任务 × 2 设置 × 100 次 rollout，每步重规划都要 25 + 50 步去噪，评测本身会相当耗时，建议多卡并行。
- **复现 Stage 2 训练**：论文配置为 32×H100，每卡 batch 16，没有报告训练时长。若只有少量 GPU，可以利用官方的 latent cache 预计算（`precompute.sh`）和从官方检查点 warm-start。
- **复现 Stage 1 预训练**：需要 EgoDex 数据并对其全量跑 RAFT，算力和存储开销都不小；这一阶段贡献了约 10 个点的提升，是复现完整结果的关键。
- **真机迁移**：最大的不确定性在 flow 监督的获取方式。若采用 robot-only 渲染，需要准确的 URDF、相机内外参标定和时间同步；若采用全场景 RAFT（`FLOW_MODE=full_scene`），则需要自行验证效果，论文没有给出对比。

**实践建议：**

1. 先用官方检查点在少量 RoboTwin 任务上复现，再测量单次重规划的端到端延迟，判断是否适合真机控制频率；
2. 如果想验证“格式对齐”结论，最便宜的实验是在自己的视频生成器上复刻消融 (b)：固定生成器，比较 raw $(u,v)$ 与 HSV flow 条件；
3. 想改进的话，最值得尝试的是把腕部相机的 flow 补上，以及比较 robot-only flow 与全场景 flow 的差异。

---

## 9. 个人启发与后续问题

1. **“让新模态看起来像视频”是一个可推广的原则。** FlowWAM 证明，只要把信号编码成预训练 VAE 能处理的图像格式，就能以极小的结构改动（只多一个 patch embed、一个 head）接入视频生成器并继承其先验。深度、法线、语义分割、甚至力觉热图都可以用同样的方式接入。这与 3D 方向“把几何属性编码成图像/视频以复用 2D 生成先验”的思路完全同构。
2. **Flow 作为动作表示与跨本体的关系还没被真正检验。** 作者在引言中把“数值动作难以跨本体”作为动机之一，但实验中每个平台都是单独训练的，没有任何跨本体迁移实验（例如在 Franka 上训练的 flow 计划能否被 ARX 的 Action Expert 解码）。理论上 flow 是本体无关的，但 Action Expert 仍是本体专属的，这个承诺还需验证。
3. **显式未来想象到底值多少？** FlowWAM 与 Fast-WAM 在 RoboTwin 上只差约 1 个点，一个显式生成 RGB+flow，一个测试时跳过想象。在同一代码库里做“生成 flow 后再解码”与“只用训练时 flow 共训练、测试时直接解码”的对照，才能回答 flow 的收益究竟来自测试时规划还是训练时表征。
4. **Flow 质量与成功率的因果关系**：可以用 GT flow 替换预测 flow 测出成功率上限，并逐步加噪画出“flow 误差—成功率”的干预曲线，这比相关系数更有说服力。
5. **与长时域的结合**：当前 9 帧 / 32 步的短 chunk 加上 25 步重规划，本质上是短视距 MPC。作者把扩展到更长时域列为未来工作，这与 [[Long-WAM - 扩展世界动作模型的上下文]] 的问题自然衔接：flow 计划能否在更长上下文下保持一致？

---

## 10. 材料来源

- 官方论文 PDF：`assets/FlowWAM - 光流统一动作表示的世界动作模型/sources/paper.pdf`（arXiv 2607.13017v1，27 页，附录在同一 PDF 中，无单独补充材料）
- 来源清单：`assets/FlowWAM - 光流统一动作表示的世界动作模型/sources.md`
- 项目页：https://flow-wam.github.io/
- 代码：https://github.com/YixiangChen515/FlowWAM ；https://github.com/YixiangChen515/FlowWAM_WorldArena
- 权重与数据：https://huggingface.co/YixiangChen/FlowWAM ；https://huggingface.co/datasets/YixiangChen/FlowWAM_RoboTwin
