# 扩散模型面试题（文生图 / 图生图 / 文生视频）标准答案

> 覆盖：DDPM/LDM/CFG、采样与加速、T2I 条件注入、I2I 与 Inpainting、ControlNet、T2V 时序一致性、工程诊断。

## 目录

- [题1：为什么训练常用预测噪声 ε，而不是直接预测 x0？等价吗？](kuosan.md#题1为什么训练常用预测噪声-ε而不是直接预测-x0等价吗)
- [题2：写出 q(xt|x0)，解释为何训练变成“去噪回归”](kuosan.md#题2写出-qxtx0解释为何训练变成去噪回归)
- [题3：扩散与 score matching 的深层联系？如何用 SDE/ODE 统一？](kuosan.md#题3扩散与-score-matching-的深层联系如何用-sdeode-统一)
- [题4：噪声日程（schedule）为什么关键？怎么选型与诊断？](kuosan.md#题4噪声日程schedule为什么关键怎么选型与诊断)
- [题5：DDPM / DDIM / DPM-Solver / Heun / ODE 采样差别？DDIM 为何少步？](kuosan.md#题5ddpm--ddim--dpm-solver--heun--ode-采样差别ddim-为何少步)
- [题6：Classifier Guidance vs CFG？为什么 CFG 会过饱和/过锐/崩？怎么缓解？](kuosan.md#题6classifier-guidance-vs-cfg为什么-cfg-会过饱和过锐崩怎么缓解)
- [题7：文生图文本条件怎么注入？Cross-Attention 的 Q/K/V 来自哪里？](kuosan.md#题7文生图文本条件怎么注入cross-attention-的-qkv-来自哪里)
- [题8：为什么能在 VAE latent 上做扩散？代价是什么？](kuosan.md#题8为什么能在-vae-latent-上做扩散代价是什么)
- [题9：除了噪声回归，还能怎么提升对齐/美学/可控性？](kuosan.md#题9除了噪声回归还能怎么提升对齐美学可控性)
- [题10：图生图为何等价于“从某个噪声等级开始反推”？strength/t_start 是什么？](kuosan.md#题10图生图为何等价于从某个噪声等级开始反推strengtht_start-是什么)
- [题11：Inpainting/Outpainting：mask 怎么进入模型/采样过程？](kuosan.md#题11inpaintingoutpaintingmask-怎么进入模型采样过程)
- [题12：ControlNet / T2I-Adapter 为什么有效？不破坏原模型能力的关键是什么？](kuosan.md#题12controlnet--t2i-adapter-为什么有效不破坏原模型能力的关键是什么)
- [题13：把 T2I 直接扩到视频会出什么问题？如何解释“时间一致性”？](kuosan.md#题13把-t2i-直接扩到视频会出什么问题如何解释时间一致性)
- [题14：视频扩散架构：3D U-Net、时序注意力、2D+Temporal Module 的权衡？](kuosan.md#题14视频扩散架构3d-u-net时序注意力2dtemporal-module-的权衡)
- [题15：视频采样如何做条件约束与稳定控制（首帧/关键帧/光流/轨迹/相机运动）？](kuosan.md#题15视频采样如何做条件约束与稳定控制首帧关键帧光流轨迹相机运动)
- [题16：训练/采样问题（塌缩、黑图、糊、字差、脸崩）排查路径？](kuosan.md#题16训练采样问题塌缩黑图糊字差脸崩排查路径)

---

## 题1：为什么训练常用预测噪声 ε，而不是直接预测 x0？等价吗？

### 核心结论

- 在标准 DDPM 里，最常用训练目标是预测注入噪声 `ε` 的 MSE：

$$

\mathbb{E}_{t, x_0, \epsilon}\big[\lVert \epsilon - \epsilon_\theta(x_t, t, c) \rVert_2^2\big]

$$

- 预测 `ε`、预测 `x0`、预测 `v`（velocity）在理想条件下是“等价参数化”（可线性互转），但在不同噪声步 `t` 上的数值尺度与有效学习信号不同，导致训练稳定性与采样表现差异。

### 关键公式（会写就能扛住追问）

- 前向闭式：

$$

x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)

$$

- 从 `ε` 反推 `x0`：

$$

\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}

$$

- 从 `x0` 反推 `ε`：

$$

\hat{\epsilon} = \frac{x_t - \sqrt{\bar\alpha_t}\,\hat{x}_0}{\sqrt{1-\bar\alpha_t}}

$$

### 工程直觉 / 常见坑

- 直接预测 `x0` 在高噪声步（`\bar\alpha_t` 很小）会出现尺度放大（除以 `\sqrt{\bar\alpha_t}`），训练更敏感。
- `v`-prediction 往往在 CFG 场景更稳定，能缓解不同 `t` 上目标尺度差异。
- 进一步追问常考：为什么要做 SNR weighting / min-SNR / P2 loss？答：为了重平衡不同噪声段的梯度贡献，避免模型只擅长某个 `t` 区间。

---

## 题2：写出 q(xt|x0)，解释为何训练变成“去噪回归”

### 核心结论

- 前向马尔可夫链：

$$

q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)

$$

- 合成闭式：

$$

q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)

$$

- 因为 `x_t` 是“干净信号 + 高斯噪声”的线性组合，训练时随机采样 `t` 用闭式构造 `(x_t, t, ε)`，就把最大似然/变分目标转为“去噪回归”（估计噪声/score/干净图）。

### 关键点

- `\bar\alpha_t = \prod_{s=1}^t \alpha_s`。
- 随机采样 `t` 的意义：覆盖从“几乎干净”到“几乎纯噪声”的所有尺度，让模型学到全程去噪动力学。

---

## 题3：扩散与 score matching 的深层联系？如何用 SDE/ODE 统一？

### 核心结论

- 扩散模型在学习每个噪声层的 score：

$$

s_t(x) = \nabla_x \log p_t(x)

$$

- 有了 score，就能写出反向 SDE（带随机项）进行采样；也能写出概率流 ODE（无随机项）做确定性采样。

### 必会叙述（口头讲清就很加分）

- Denoising Score Matching：在加噪分布下，最优去噪器与 score 存在等价关系，因此“预测噪声/预测 `x0`”间接等价于在估计 `\nabla_x\log p_t(x)`。
- 反向 SDE 直觉：正向是把数据推向高斯噪声；反向则用 score 引导，把噪声逐步拉回数据流形。
- 概率流 ODE：把随机扩散过程等价改写成一个确定性流，便于用高阶 ODE solver 减步。

### 常见追问

- SDE vs ODE：SDE 多样性更强，ODE 更一致/更快但多样性可能下降。

---

## 题4：噪声日程（schedule）为什么关键？怎么选型与诊断？

### 核心结论

- schedule 决定每个 `t` 的信噪比（SNR）分布，直接影响模型把容量用在全局语义还是局部纹理。
- schedule 不匹配常导致：训练不稳、细节糊、采样步数敏感、CFG 一大就崩等。

### 关键概念

- 离散 VP 常用：

$$

\mathrm{SNR}_t = \frac{\bar\alpha_t}{1-\bar\alpha_t}

$$

- 高噪声段（低 SNR）：决定语义/布局；低噪声段（高 SNR）：决定细节/锐度。

### 诊断与修正

- 语义弱/不听 prompt：通常高噪声段学得不够（或条件注入弱）。
- 细节糊/锐度不足：低噪声段有效学习信号不足，或采样步数/solver 不够。
- 常用修正：SNR weighting、min-SNR、P2 loss、或切换到更适配的 `σ` 参数化（例如 EDM 思路）。

---

## 题5：DDPM / DDIM / DPM-Solver / Heun / ODE 采样差别？DDIM 为何少步？

### 核心结论

- DDPM：随机反向过程，每步有噪声项；稳但步数多。
- DDIM：构造确定性（或可控随机性）的反向路径（常用 `η` 控制随机性），可用更少步数近似生成。
- DPM-Solver / Heun：把采样视为对连续动力学（ODE/SDE）的数值积分，用高阶方法减少步数。

### 你要能说出的“少步原因”

- 采样就是积分：步数越少误差越大；高阶 solver 能在相同步数下降低误差。
- DDIM 的 `η=0` 近似确定路径，减少随机扰动，从而更易用少步达到可接受质量（代价是多样性下降）。

### 常见坑

- 少步不必然更好：可能丢高频或引入偏差；不同模型对 sampler 匹配程度不同。

---

## 题6：Classifier Guidance vs CFG？为什么 CFG 会过饱和/过锐/崩？怎么缓解？

### 核心结论

- Classifier Guidance：额外训练分类器，生成时用 `\nabla_x\log p(y|x)` 修正 score。
- CFG（Classifier-Free Guidance）：同一扩散模型同时学有条件/无条件，通过差分放大条件效果：

$$

\epsilon_{\text{guided}} = \epsilon_{\text{uncond}} + w(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})

$$

### 为什么会“过饱和/过锐/崩”

- `w` 放大了条件方向的更新，尤其在低噪声末期容易把状态推到训练分布外，出现过锐、过饱和、伪影甚至发散。

### 常见缓解

- 动态 guidance（后期减小 `w`）
- CFG-rescale（对 guided 输出做幅度/方差校准）
- 增加 steps 或换更稳的 solver
- 使用 `v`-prediction 或更适配参数化

---

## 题7：文生图文本条件怎么注入？Cross-Attention 的 Q/K/V 来自哪里？

### 核心结论

- 主流 U-Net/LDM：在多个分辨率层插入 cross-attention。
- `Q` 来自图像 latent 特征（空间位置 token），`K/V` 来自文本编码器 token。

### 关键细节

- 多尺度注入：低分辨率层影响布局/语义，高分辨率层影响局部细节。
- `t` embedding 通常通过 AdaGN/FiLM 注入到 block 中，控制不同噪声级别的去噪行为。

### 常见坑

- token 绑定错位（属性绑错对象）：注意力竞争 + 数据噪声 + 组合分布稀疏导致。

---

## 题8：为什么能在 VAE latent 上做扩散？代价是什么？

### 核心结论

- 扩散不要求在像素空间；只要定义合理的加噪过程并能表达数据分布，latent 空间同样可做扩散。
- 好处：计算/显存大幅下降；代价：生成上限受 VAE 压缩与 decoder 表达力限制。

### 常见追问

- 为什么文字/细线容易糊：VAE 压缩丢高频细节 + 文本排版在数据中本就难学。

---

## 题9：除了噪声回归，还能怎么提升对齐/美学/可控性？

### 核心结论

- 质量与对齐来自“数据 + 条件编码器 + loss/权重 + 采样 + 后训练”。

### 常见路径（分清阶段）

- 训练阶段：更干净 caption、更强文本编码器、SNR/min-SNR/P2 等重权、引入结构条件（depth/edge/seg）。
- 后训练/对齐：偏好/奖励信号蒸馏到模型（可在采样输出层面做 ranking/偏好学习，再蒸馏回去）。
- 推理阶段：CFG/负提示词、refiner（二阶段扩散）、prompt 重写等。

### 常见坑

- 只说“加 RLHF”但讲不清如何把 reward 作用到扩散过程：应能说明 reward 可对最终样本打分，再通过蒸馏/引导近似影响每步更新。

---

## 题10：图生图为何等价于“从某个噪声等级开始反推”？strength/t_start 是什么？

### 核心结论

- 图生图：先把输入图 `x0` 加噪到 `t_start` 得到 `x_{t_start}`，再从该噪声级别反向去噪。
- `strength`（或 `t_start`）越大，改动越大、越听 prompt；越小越保真。

### 关键公式

$$

x_{t\_start} = \sqrt{\bar\alpha_{t\_start}}x_0 + \sqrt{1-\bar\alpha_{t\_start}}\,\epsilon

$$

---

## 题11：Inpainting/Outpainting：mask 怎么进入模型/采样过程？

### 核心结论

- 目标是“已知区域严格满足约束，未知区域自由生成”。
- 典型做法：输入拼接 mask 相关通道；并在每一步采样后对已知区域做重注入（把已知区域替换回对应噪声状态）。

### 常见坑

- 边界缝合差：mask 硬边导致不连续；可用 feather、边缘扩张、边界融合。
- 已知区域被改写：说明没有做每步重注入，或注入方式不正确。

---

## 题12：ControlNet / T2I-Adapter 为什么有效？不破坏原模型能力的关键是什么？

### 核心结论

- 它们通过“额外条件分支 + 残差注入”提供可控结构先验。
- 关键：冻结主干、残差初始为 0（zero-conv），保证初始行为等价于原模型，只学习“在何时加多少结构残差”。

### 常见坑

- 控制过强导致贴图感/僵硬：需要降低控制权重或做阶段性调度（高噪声阶段更重结构，低噪声阶段更重纹理）。

---

## 题13：把 T2I 直接扩到视频会出什么问题？如何解释“时间一致性”？

### 核心结论

- 逐帧独立：`p(x_{1:T}|c) \approx \prod_t p(x_t|c)` 会产生闪烁、身份漂移、纹理随机变化。
- 视频需要建模联合分布的时序依赖（共享运动/外观隐变量），至少要显式引入时间维建模或跨帧约束。

---

## 题14：视频扩散架构：3D U-Net、时序注意力、2D+Temporal Module 的权衡？

### 核心结论

- 3D U-Net：时空耦合强、一致性好；但计算/显存开销大，长度扩展难。
- 分解注意力（空间/时间）：显著降复杂度，是常见折中。
- 2D backbone + temporal module：复用 T2I 权重，训练成本低、迁移快，工业常用。

### 工程策略

- 长视频：滑窗/分段 + overlap 融合；必要时加跨段 reference attention 保一致。

---

## 题15：视频采样如何做条件约束与稳定控制（首帧/关键帧/光流/轨迹/相机运动）？

### 核心结论

- 首帧/关键帧是强条件（硬约束或强引导），用来固定身份/风格/布局锚点。
- 光流/轨迹/相机运动是运动先验，帮助共享时序隐变量，减少漂移。

### 常见做法

- Reference attention：参考帧特征做 K/V，当前帧做 Q，对齐身份/风格。
- Latent warping：按光流变形上一帧 latent 作为初始化/条件。
- 分阶段：先定运动与布局（低分辨率/低帧率），再超分与细化。

---

## 题16：训练/采样问题（塌缩、黑图、糊、字差、脸崩）排查路径？

### 核心结论

- 排查要有顺序：先排推理设置（sampler/steps/CFG）→ 再看 VAE 上限 → 再看 loss/SNR 与条件注入 → 最后看数据质量。

### 典型症状与原因

- 黑图/发散：混合精度溢出、学习率过大、schedule/solver 不匹配、CFG 过大、VAE 异常。
- 普遍发糊：VAE 瓶颈、步数太少、solver 不匹配、低噪声段学习不足。
- prompt 不听：高噪声段没学好、文本编码器弱、caption 噪声、uncond 分支质量差。
- 文字差：VAE 压缩 + 数据中文字复杂 + cross-attn 对精确排版弱；可用更高分辨率训练、文字专用数据、refiner/编辑模块。
- 脸崩/身份不稳：采样末期 CFG 过强、分辨率外推、数据分布；可用 ID 条件（reference/embedding）、动态 guidance、专项 finetune。

### 推荐排查顺序（可直接背）

1. 固定权重，先调 sampler/steps/CFG 看是否立刻改善
2. 检查 VAE 重建质量（决定细节上限）
3. 看不同噪声段（t/σ）的 loss 与梯度是否失衡（SNR weighting 是否需要）
4. 可视化 cross-attention/条件分支输出（条件是否真的进模型）
5. 回到数据：过滤、caption 质量与分布覆盖