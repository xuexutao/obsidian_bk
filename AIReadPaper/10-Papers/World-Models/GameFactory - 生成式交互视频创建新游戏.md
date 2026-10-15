---
type: paper-note
status: complete
domain: World-Models
paper: "GameFactory: Creating New Games with Generative Interactive Videos"
year: 2025
arxiv: "2501.08325"
doi: null
source: https://arxiv.org/abs/2501.08325
project: https://yujiwen.github.io/gamefactory/
code: https://github.com/KwaiVGI/GameFactory
tags:
  - World-Models
  - Interactive-Video
  - Video-Diffusion
  - Game-Generation
created: 2026-09-04
updated: 2026-09-04
---

# GameFactory：生成式交互视频创建新游戏

## 基本信息

- **作者**：Jiwen Yu、Yiran Qin、Xintao Wang、Pengfei Wan、Di Zhang、Xihui Liu
- **机构**：The University of Hong Kong、Kuaishou Technology
- **版本**：arXiv v4；ICCV 2025 Highlight
- **核心资源**：[论文](https://arxiv.org/abs/2501.08325) · [项目页](https://yujiwen.github.io/gamefactory/) · [代码入口](https://github.com/KwaiVGI/GameFactory) · [GF-Minecraft](https://huggingface.co/datasets/KwaiVGI/GameFactory-Dataset)

## 一句话结论

GameFactory 的关键不是把 Minecraft 模拟得更像，而是把从 **70 小时 Minecraft 动作数据**中学到的键鼠控制能力，与预训练视频模型的开放域视觉先验解耦，再把控制能力迁移到樱花林、雪山、室内、动物甚至赛车等未见场景；它证明了“少量动作标注 + 大规模无动作视频先验”可以产生场景可泛化的交互视频，但离实时、长记忆、可编辑状态的真正游戏引擎仍很远。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

传统游戏引擎显式维护地图、对象、碰撞和状态，而生成式游戏引擎直接预测用户动作之后的画面。已有 DIAMOND、GameNGen、Oasis 等方法通常绑定某一款游戏或固定视觉域：它们能“继续生成 Minecraft/DOOM”，却很难把学到的控制规律迁移到一个由文本临时创造的新世界。GameFactory 因此把问题定义为：给定开放域文本提示、初始视觉内容和逐帧键盘/鼠标动作，生成既符合场景语义、又响应动作、还能持续滚动的第一人称视频。

困难来自数据错配。互联网视频覆盖开放世界，但没有精确键鼠标签；游戏日志有动作标签，却只覆盖少数视觉风格。若直接在 Minecraft 上微调开放域视频模型，模型会把“动作控制”和“像素方块风格”一起学进去，最终发生 domain collapse：有控制，但所有新游戏都像 Minecraft。

![](assets/GameFactory%20-%20生成式交互视频创建新游戏/figures/01-teaser.png)

> **Figure 1**：同一套从 Minecraft 学到的 W/A/S/D、Space 与鼠标转向，被迁移到樱花林、室内椅子、雪山圣伯纳犬和文艺复兴宫殿。关键观察不是单帧好看，而是按键和视角变化仍能驱动画面。  
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/pdf/2501.08325

### 1.2 现有方法的瓶颈

一是**动作标签昂贵**。开放域视频无法像游戏引擎那样直接导出逐帧键鼠事件。二是**人类行为偏差**。VPT 等人类游戏录像中 W 占 50.11%，S 只有 0.32%，模型容易把“向前”当作默认运动，忽略后退、原地跳跃等低频动作。三是**扩散视频固定长度**。全序列双向扩散不天然适合持续交互。四是**风格与控制纠缠**。小域微调会覆盖开放域先验。

### 1.3 本文核心贡献

1. 提出 **GF-Minecraft**：70 小时、动作分布刻意均衡、带文本描述的第一人称 Minecraft 数据。
2. 针对连续鼠标和离散键盘设计不同注入路径，并用滑动动作窗口处理 VAE 时间压缩与动作延迟。
3. 以类似 Diffusion Forcing 的逐帧噪声/条件设计实现多帧块自回归，而不是低效的逐帧生成。
4. 用 LoRA domain adapter 与多阶段冻结策略，把游戏风格学习和动作控制学习拆开，从而在推理时移除 Minecraft 风格、保留动作能力。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

设视频为 $X=[x_0,x_1,\ldots,x_{rn}]$，VAE 时间压缩率为 $r$，编码后得到 $Z=[z_0,z_1,\ldots,z_n]=E(X)$。动作序列 $A=[a_1,\ldots,a_{rn}]$ 中，$a_i$ 表示从 $x_{i-1}$ 转移到 $x_i$ 的动作。输入包括：

- 文本提示 $p$，负责场景和语义；
- 第一段历史视频或初始帧；
- 每个渲染帧对应的键盘离散动作 $K$；
- 鼠标 yaw/pitch 连续变化 $M$。

输出是 $360\times640$ 的动作条件视频。动作空间含 W/A/S/D、Space、Shift、Ctrl 与鼠标两轴；论文没有建模背包、对象 ID、显式三维状态或奖励，因此它本质上是像素预测器，而非可查询的符号化游戏状态机。

### 2.2 目标函数

基础训练是动作条件噪声预测：

$$
\mathcal{L}_{a}(\phi)=\mathbb{E}\left[\left\|\epsilon_{\phi}(Z_t,p,A,t)-\epsilon\right\|_2^2\right].
$$

其中 $Z_t$ 是加噪潜变量，$t$ 是扩散时刻，$\epsilon$ 是注入噪声，$\epsilon_\phi$ 是带文本和动作条件的预测器。直觉上，若相同视觉历史配不同动作，网络必须利用 $A$ 才能正确还原下一段潜变量；否则损失无法下降。

---

## 3. 核心方法

### 3.1 总体框架

骨干是内部的 **1B Transformer 视频扩散模型**，由更大模型蒸馏而来，预先拥有开放域生成先验。GameFactory 不重训骨干，而是在 Transformer block 中插入动作控制模块，再用 LoRA 暂存 Minecraft 风格。推理时移除 LoRA，只保留骨干和动作模块。

这是一种参数角色分工：骨干回答“世界长什么样”，LoRA 在训练中回答“Minecraft 长什么样”，动作模块回答“按键之后如何变化”。真正的创新不在某个单独层，而在于训练阶段让三类参数避免共同吸收同一域偏差。

### 3.2 动作注入：键盘与鼠标不能一视同仁

![](assets/GameFactory%20-%20生成式交互视频创建新游戏/figures/02-action-module.png)

> **Figure 3**：连续鼠标量经 group、repeat、拼接、MLP 和 temporal self-attention 注入；离散键盘类别先嵌入，再作为 K/V 与视频特征做 cross-attention。  
> 来源：论文 Figure 3，第 4 页，https://arxiv.org/pdf/2501.08325

设中间视频特征为 $F\in\mathbb{R}^{(n+1)\times l\times c}$，鼠标输入 $M\in\mathbb{R}^{rn\times d_1}$，键盘输入 $K\in\mathbb{R}^{rn\times d_2}$。由于 VAE 把 4 个视频帧压成一个 latent frame（$r=4$），动作数量和潜变量数量不一致。作者为第 $i$ 个 latent 汇集窗口

$$
[a_{r(i-w+1)},\ldots,a_{ri}],\qquad w=3,
$$

并用边界动作补齐越界位置。窗口不仅对齐时间粒度，也显式覆盖“按下跳跃后效果延续几帧”这样的动作延迟。

连续鼠标幅度具有几何含义。作者将 $M_{group}$ reshape 后沿 token 维重复，与 $F$ 在通道维拼接，再经 MLP 与时间注意力；这样鼠标位移大小不会被相似度归一化抹平。键盘则是类别组合，先 embedding + positional encoding，再令视频特征作 query、键盘组作 key/value 做 cross-attention。消融印证了这个异构设计：离散键盘更适合 cross-attention，连续鼠标更适合 concat。

### 3.3 长视频：Diffusion Forcing 式条件化，而非蒸馏

GameFactory **没有使用 Self-Forcing，也没有将模型蒸馏成少步实时生成器**。它借鉴 Diffusion Forcing 的思想，允许同一序列不同帧处于不同噪声水平：较早帧保持干净并作为历史，较晚帧加噪并被预测。

训练时，从 $N+1$ 个 latent 中随机选前 $k+1$ 个作为无噪条件，只对后 $N-k$ 个预测帧计算损失。推理先生成首段，再反复取最新 $k+1$ 帧作为条件，一次生成 $N-k$ 个新帧。与 next-frame 方法相比，它每次生成一块，吞吐更高；但因为训练历史来自真实数据、推理历史来自模型自身，仍存在 exposure bias。论文用“无限长度”描述算法可重复执行，不应理解为质量可无限维持。

### 3.4 风格—动作解耦与多阶段训练

![](assets/GameFactory%20-%20生成式交互视频创建新游戏/figures/03-multiphase-training.png)

> **Figure 6**：Phase 0 开放域预训练；Phase 1 仅用 LoRA 适配 Minecraft 风格；Phase 2 冻结骨干和 LoRA，只训练动作模块；Phase 3 移除 LoRA，以原始开放域骨干和动作模块生成新场景。  
> 来源：论文 Figure 6，第 6 页，https://arxiv.org/pdf/2501.08325

- **Phase 0**：在开放域无动作视频上预训练骨干，论文直接采用内部模型。
- **Phase 1**：在 GF-Minecraft 上只训练 rank 128 的 LoRA，使“解释像素风格”的压力先被 adapter 吸收。
- **Phase 2**：冻结骨干和 LoRA，只训练动作模块。此时视觉域已拟合，动作模块主要负责降低动作条件误差，而不是顺便记住方块纹理。
- **Phase 3**：推理时拔掉 LoRA，开放域先验恢复，动作模块仍可工作。

这并非严格统计意义上的因果解耦，模块之间仍共享扩散损失，也没有互信息约束；它更像利用优化顺序和冻结边界实现的工程性解耦。其优点是简单，缺点是依赖 Phase 1 足够好地“吃掉”风格差异。

### 3.5 推理流程与复杂度

论文使用 DDIM 50 步，且每个生成块仍需运行完整扩散网络。一次动作响应不是单帧即时反馈，而是一段视频的批量去噪；项目也没有报告 FPS 或端到端交互延迟。与后续 Matrix-Game 2.0 的 3 步因果模型相比，GameFactory 更适合证明“控制可迁移”，不适合证明“可实时玩”。

### 3.6 从输入到输出的完整数据流

一次训练样本先由 VAE 把 77 帧左右的视频压为 latent 序列，文本经文本编码器成为场景条件，逐帧动作则按 $r=4,w=3$ 聚合。每个 latent 位置因此看到当前时间段和前两个时间段的动作。鼠标组被复制到所有空间 token，保留旋转幅度；键盘组保留多个离散 token，通过 cross-attention 选择相关动作。两路控制在各 Transformer block 中与视觉特征融合，最后由噪声预测损失共同监督。

在 Phase 1，动作虽然随数据存在，但动作模块尚未承担优化，LoRA 先拟合 Minecraft 的纹理、HUD 与视角统计；Phase 2 冻结 LoRA 后，视觉重建误差已较低，剩余可下降的部分更多来自动作—运动对应。推理移除 LoRA，相当于把“Minecraft 外观残差”置零，但保留动作控制残差。这个过程解释了为什么 one-phase 容易风格泄漏，也说明失败条件：若 LoRA 容量不足，动作模块仍会被迫编码视觉域；若 LoRA 过强，又可能替骨干记住太多训练片段。

长视频阶段把历史 clean latent 与未来 noisy latent 放在同一序列中。随机 $k$ 让模型见到不同长度的历史条件，且只在未来区域回传损失。每轮推理将新生成块追加到 history，再截取最近 $k+1$ 帧。它减少了逐帧调用次数，但每块内部仍是 50 步 DDIM，延迟由块大小、DiT 规模和去噪步数共同决定。

---

## 4. 数据集与实验设置

### 4.1 GF-Minecraft

作者借助 MineDojo 采集 **70 小时**游戏视频：3 个 biome（forest、plains、desert）× 3 种天气（clear、rain、thunder）× 6 个时段；形成 2,000 个带动作视频，每个 2,000 帧。预处理把长视频随机切成 81 帧片段，并令抽样总帧数达到原视频的 3 倍；MiniCPM-V 生成场景描述。

动作采样不是模仿人类，而是把原子动作频率与持续时间刻意均衡并随机组合。因此它牺牲“像人类操作”的先验，换取“模型见过所有控制组合”的覆盖。对交互生成器而言，这比只追求自然轨迹更重要。

### 4.2 Baseline 与指标

- **数据 baseline**：VPT Find Cave 子集，检验人类动作偏差。
- **训练 baseline**：one-phase 直接联合微调，对比 multi-phase。
- **动作模块消融**：键盘/鼠标分别采用 cross-attention 或 concatenation。
- **长视频消融**：所有帧计算 loss，对比只在预测帧计算 loss。
- **指标**：Cam↓（GLOMAP 相机姿态距离）、Flow↓（相对参考视频光流误差）、CLIP↑、FID↓、FVD↓，另有 Dom↑ 衡量与原始开放域模型的域相似度。

这些指标大多是代理量。尤其 Cam/Flow 依赖参考视频，开放域文本场景中并不存在唯一正确未来；而 FID/FVD 不能证明碰撞、对象持久性或游戏规则正确。

### 4.3 实现细节与资源

每个阶段需 **8×A100，2–4 天**，batch size 64。LoRA 学习率 $10^{-4}$、rank 128；动作模块学习率 $10^{-5}$；VAE 时间压缩率 4；50 步 DDIM。粗略看，仅两个微调阶段就可能占用 768–1,536 A100-GPU 小时，尚不含内部基础模型预训练。

---

## 5. 实验结果

### 5.1 动作注入消融

![](assets/GameFactory%20-%20生成式交互视频创建新游戏/figures/04-control-and-generalization-ablation.png)

> **Table 2–3 / Figure 7–8**：左侧比较键盘与鼠标的不同注入组合，右侧比较多阶段与单阶段开放域结果。最佳控制结构是键盘 cross-attention + 鼠标 concat；多阶段训练减少 Minecraft 风格泄漏与伪影。  
> 来源：论文 Table 2–3、Figure 7–8，第 7 页，https://arxiv.org/pdf/2501.08325

在 only-key 集上，键盘 cross-attention + 鼠标 concat 的 Cam/Flow 为 **0.0439/7.79**，优于双 concat 的 **0.0853/22.37** 和双 cross-attention 的 **0.0527/8.67**。在 mouse-small 上该组合为 **0.0685/18.64**，在 mouse-large 上为 **0.1021/249.54**。鼠标幅度越大，误差显著上升，说明大视角旋转仍更难。

### 5.2 场景泛化

开放域上，multi-phase 相比 one-phase：Cam **0.0997 vs 0.1134**，Flow **54.13 vs 76.02**，Dom **0.7565 vs 0.7345**，CLIP **0.3181 vs 0.3111**，FID **121.18 vs 167.79**，FVD **1256.94 vs 1323.58**。方向一致：解耦既改善动作跟随，也更接近原模型视觉域。

但比较需谨慎：开放域没有与其它公开世界模型在统一输入、统一动作、统一分辨率下的 head-to-head。论文 Table 1 主要比较能力表，不是同一 benchmark 的性能表。因此“优于所有 baseline”并不是本文能支持的结论；更准确的说法是，它在自身消融中证明多阶段优于直接微调。

### 5.3 GF-Minecraft 对人类偏差的作用

![](assets/GameFactory%20-%20生成式交互视频创建新游戏/figures/05-dataset-and-long-video-results.png)

> **Table 4–6 / Figure 9–10**：均衡数据使模型能执行 VPT 中罕见的后退和原地跳跃；只在待预测帧计算损失，也显著改善长视频训练。  
> 来源：论文 Table 4–6、Figure 9–10，第 8 页，https://arxiv.org/pdf/2501.08325

VPT 中 W 为 **50.11%**、S 为 **0.32%**，相差超过百倍；GF-Minecraft 的 W/A/S/D 均为 **13.56%**，Space/Shift/Ctrl 均为 **15.25%**。在同域评价中，GF-Minecraft 相比 VPT 将 Cam 从 **0.1324** 降到 **0.0839**，Flow 从 **107.67** 降到 **43.48**，FID 从 **156.69** 降到 **125.85**，FVD 从 **1233.15** 降到 **1047.59**。图中 VPT 模型面对“持续后退”会停住，面对“原地跳”会擅自向前走；这是很具体的控制失败，而非审美差异。

### 5.4 长视频消融

仅对预测帧计算损失，相比所有帧计算损失：Cam **0.0924 vs 0.1547**，Flow **85.45 vs 148.73**，CLIP **0.3190 vs 0.2965**，FID **136.95 vs 176.07**，FVD **1154.45 vs 1592.43**。论文展示超过 **300 帧**的结果。改善说明把“已知干净历史帧”也纳入噪声回归会引入无关梯度；但 300 帧不等于长期世界状态稳定，论文没有分钟级或回环一致性测试。

### 5.5 失败案例与证据边界

论文把 VPT 的后退/原地跳失败展示得很清楚，却较少展示完整 GameFactory 在开放域中的失败。可从数据与实验推断的边界包括：大幅鼠标移动误差显著更高；第一人称控制迁移到赛车时，yaw 意外成为转向，而 backward/sideways 等动作会自动减弱，说明动作语义并未被显式绑定；碰撞能力来自训练数据共现而非可验证几何。所有这些能力都可能在长滚动中漂移。

### 5.6 实验结果之间如何互相支撑

三组消融形成了一条较完整的证据链。Table 2 先证明“异构动作需要异构注入”；Table 4–5 再证明控制上限受到动作数据覆盖影响；Table 3 则证明即使控制模块有效，若训练阶段不解耦，开放域能力仍会被 Minecraft 风格污染。最后 Table 6 说明从短片段扩展到滚动生成时，损失区域本身也会影响长期质量。

但这条证据链仍缺两项关键实验。其一是冻结相同基础模型、只更换公开/内部 backbone 的跨骨干复现，以判断泛化来自方法还是强视频先验；其二是动作反事实：保持初始帧和随机种子一致，仅翻转单个动作，再测几何位移。现有 Cam/Flow、CLIP/FID/FVD 混合了动作、画质和采样随机性，不能完全隔离动作因果效应。

从数值幅度看，方法对“动作是否执行”的改善大于对 CLIP 语义的改善。例如 GF-Minecraft 把 Flow 从 107.67 降到 43.48，而 CLIP 从 0.3174 略降到 0.3135；这表明均衡动作数据主要修正动力学，不必然提高文本对齐。多阶段训练则同时改善 FID/FVD 与 Dom，说明它更直接作用于域保持。把这些指标分组解读，比寻找一个总分更符合论文的实际贡献。

长视频 Table 6 还揭示一个训练细节：条件帧并非需要“重建得更好”，因为推理时它们已经存在；对它们计算噪声损失会让优化预算偏离未来预测。这个结论可迁移到其它带干净历史的 diffusion forcing 系统：损失掩码应与推理时真正未知的变量一致。

---

## 6. 与相关工作的关系

GameFactory 位于两条路线的交叉点：一条是把视频扩散用于特定游戏的 Oasis、GameNGen、DIAMOND；另一条是开放域视频先验的控制迁移。它比前者更强调场景泛化，比后来的 Matrix-Game 2.0 更强调“风格—动作解耦”。后续实时世界模型沿用了它的异构动作注入：键盘 cross-attention、鼠标 concatenation；但通过 Self-Forcing、少步蒸馏与 KV cache 补上了实时性。

---

## 7. 局限与批判性评价

### 7.1 作者明确指出的局限

论文列出：关卡与玩法多样性、玩家反馈、对象操控、长上下文记忆和实时生成仍未解决。换言之，GameFactory 展示的是“可控视频原型”，不是具备持久状态、规则系统和闭环延迟保证的游戏引擎。

### 7.2 我的判断

第一，所谓 scene-generalizable 主要是**视觉域泛化**，不是动力学泛化。模型可能让“椅子场景”响应 WASD，但这不等于理解椅子、角色和墙的碰撞规则。第二，解耦证据来自 one-phase 对照，没有测量 LoRA 和动作模块分别携带多少风格信息。第三，基础模型是内部 1B 模型，决定开放域质量的最大变量不可复现。第四，50 步 DDIM 与 8×A100 训练使系统离低延迟交互较远。

---

## 8. 复现与实践建议

### 8.1 可复现部分

GF-Minecraft 数据入口公开，论文把 MineDojo 采集、动作空间、切片和标注流程写得较清楚。可以在公开视频 DiT 上复现以下局部问题：

1. 先做游戏域 LoRA，再冻结 LoRA 训练动作模块；
2. 对比 one-phase 联合训练；
3. 复现键盘 cross-attention / 鼠标 concat；
4. 构造均衡动作与人类动作分布对照；
5. 只对预测帧计算损失。

### 8.2 成本与缺口

官方仓库目前没有完整训练/推理代码和权重，基础视频模型也未公开。严格复现至少需要 8×A100、每阶段 2–4 天、70 小时动作视频和类似的开放域基础模型。若只做机制验证，可把分辨率、DiT 参数量和动作空间缩小，在公开 I2V 模型上复现消融；但不能把这种结果直接等同于论文的开放域生成质量。

---

## 9. 个人启发与后续问题

GameFactory 最值得保留的思想是：**稀缺监督不应承担全部世界建模，只负责教会模型“干预如何改变未来”**。这对机器人、自动驾驶也成立：开放视频提供外观与常识，小规模可控数据提供 action-to-consequence 映射。

后续更关键的问题是：如何把动作从“按键 token”提升为可组合、可跨 embodiment 的干预表示；如何加入可查询的对象与地图记忆；如何用反事实测试验证模型真的利用动作，而不是依赖动作—场景相关性；以及如何在少步实时生成时保持 GameFactory 的开放域泛化。

---

## 10. 材料来源

完整来源、图表追溯与资料缺口见：[sources.md](assets/GameFactory%20-%20生成式交互视频创建新游戏/sources.md)。本地保存官方 PDF：`assets/GameFactory - 生成式交互视频创建新游戏/sources/paper.pdf`。
