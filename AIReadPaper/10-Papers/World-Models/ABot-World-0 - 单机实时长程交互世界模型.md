---
type: paper-note
status: complete
domain: World-Models
paper: "Infinite Interactive World Rollout on a Single Desktop GPU"
year: 2026
arxiv: 2607.19191
doi: null
source: https://arxiv.org/abs/2607.19191
project: https://amap-cvlab.github.io/ABot-World/
code: https://github.com/amap-cvlab/ABot-World
tags:
  - World-Model
  - Interactive-Video
  - LongForcing
  - Real-Time-Inference
  - Action-Control
  - Video-Diffusion
created: 2026-10-16
updated: 2026-10-16
---

# ABot-World-0：单机实时长程交互世界模型

## 基本信息

- **正式标题**：Infinite Interactive World Rollout on a Single Desktop GPU
- **作者署名**：ABot-World Team；论文末尾按 Foundation Model、Data、AI Infra、Benchmark 与 Engineering 列贡献者
- **来源**：arXiv:2607.19191，2026 年 7 月公开预印本；截至本笔记写作时未确认同行评审 venue
- **基础骨干**：Wan2.2 系列视频生成模型，因果学生约 5B 参数
- **目标输出**：根据初始视觉条件、逐帧键盘动作与可选角色参考图，持续生成 1280×704 动作条件视频流
- **核心部署点**：单张 NVIDIA RTX 5090；最高 15.831 FPS（论文概括为 16 FPS）；action-to-first-frame 1.2 s；低比特配置峰值显存约 17–19 GiB

## 一句话结论

ABot-World-0 的贡献不是“无限长度”这个无法严格证明的口号，而是把**双向视频教师、因果少步学生、学生自回归分布上的 LongForcing，以及 LightVAE/低比特 DiT/局部 KV cache 等系统优化**连成一条能在单张桌面 GPU 上闭环运行的链路；论文确实给出 60 秒 LongForcing 消融和单卡吞吐实测，但小时级、天级稳定性仍主要是作者选择的定性关键帧，不能等同于统一 benchmark 下的长期状态一致性证明。

---

## 1. 研究背景与问题定义

### 1.1 从短视频生成转向可进入的世界

短视频模型可以生成数秒视觉上合理的运动，却没有被要求在用户连续操作下维持世界状态。交互世界模型同时面对四个相互耦合的问题：动作监督数据稀缺；双向视频模型不能在线因果生成；自回归误差会不断回灌；即便采样步数足够少，高分辨率 VAE、注意力、RoPE 和显存驻留仍可能阻止实时运行。

论文将任务写成条件分布：给定历史帧 $oldsymbol v_{0:t-1}$、未来动作 $oldsymbol a_{t:t+L-1}$ 与多模态条件 $oldsymbol c$，生成下一段长度为 $L$ 的视频：

$$p_\theta(\boldsymbol v_{t:t+L-1}\mid\boldsymbol v_{0:t-1},\boldsymbol a_{t:t+L-1},\boldsymbol c).$$

生成 chunk 被追加到历史后继续下一轮，因而时长没有架构上的固定终点。但“没有硬编码终点”与“任意长时间保持正确状态”是两件事：后者必须由长期评测支持。

![](assets/ABot-World-0%20-%20单机实时长程交互世界模型/figures/01-teaser.png)

> **图 1：ABot-World-0 的动作控制场景。** 图中统一用 `W/A/S/D` 与 `I/J/K/L` 控制第一人称漫游、第三人称角色与车辆，体现统一离散动作接口覆盖多种场景，而不是证明每个场景都具有可靠物理。图片已从 arXiv 源码原图提取并视觉核验。  
> 来源：论文 Figure 1，<https://arxiv.org/abs/2607.19191>

### 1.2 论文的系统性贡献

1. 以 AAA 游戏、仿真引擎与互联网视频构建统一 action-video 数据基础设施；
2. 在预训练视频 DiT 上训练动作条件双向教师，再渐进转换为因果学生；
3. 提出 LongForcing，在学生自己的长程 rollout 上用扩展时域教师做分布匹配；
4. 用角色参考记忆减轻第三人称身份漂移；
5. 以 chunk streaming、LightVAE、低比特 DiT、SageAttention2、Fast-RoPE、局部 KV cache 和模块换入换出实现单卡部署。

---

## 2. 数据基础设施

### 2.1 三类数据源与动作标签

**AAA 游戏数据**是主要来源，运行时 API 同步记录 1920×1080 RGB、逐帧控制、相机位姿与环境元数据，动作标签质量最高。**仿真数据**来自 Unreal Engine 与 ABot-3DGS 场景，轨迹经相邻位姿变化投影到相机基向量后离散化。**互联网视频**补充真实光照与运动分布，但只能先估计 6-DoF 轨迹，再阈值化为伪动作，因此监督噪声最高。

三路数据最终映射到统一的 8 键动作空间。论文还设计 WorldExplorer 闭环采集 Agent，根据训练弱项调整场景与任务采样，并以 14 项确定性规则及 VLM 语义评分过滤 UI、加载页、穿模、死亡序列、动作错位和元数据异常。

### 2.2 证据与复现边界

作者没有披露总视频时长、三类数据精确占比、完整游戏清单和训练总算力；部分 3DGS 资产是非公开专有数据。因此论文能支持“建立了多源数据系统”，不能支持外部研究者按同等数据条件完整复现。

---

## 3. 动作与身份条件

### 3.1 统一动作表示

每帧动作是 8 维 multi-hot 向量：`W/A/S/D` 表示平移，`I/J/K/L` 表示相机旋转。由于 VAE 时间压缩率为 4，连续四帧动作按通道拼接为 32 维 token：

$$\tilde a_\tau=\operatorname{Concat}(a_{4\tau-3},a_{4\tau-2},a_{4\tau-1},a_{4\tau}),\qquad\tilde a_\tau\in\{0,1\}^{32}.$$

Action Control Adapter 用 PixelUnshuffle、卷积与残差块编码动作，然后在 patch embedding 后逐 token 相加：

$$\hat{\boldsymbol z}=\operatorname{PatchEmbed}(\boldsymbol z)+\mathcal F_\psi(\tilde{\boldsymbol a}).$$

这个设计的优点是训练标签与用户按键一致、计算便宜；局限是动作语义简单，不能直接表达连续控制量、复杂事件或高层意图。

### 3.2 角色参考记忆

多视角角色参考图经同一 VAE 编码为 identity-memory tokens，并放在固定负时间 RoPE 位置；视频 token 使用非负时间索引。非对称 attention 允许视频读取身份记忆，但不让身份 token 被当前视频反向污染。它提供持续外观锚点，却不是通用场景记忆：模型仍可能忘记远期空间布局与物体状态。

![](assets/ABot-World-0%20-%20单机实时长程交互世界模型/figures/03-model-architecture.png)

> **图 4：动作与身份记忆注入。** 动作 adapter 在 patch token 入口做加法条件，角色参考图作为负时间索引的持久 memory；两者分别解决“怎么动”和“谁在动”。图像边界、文字与图例已视觉核验完整。  
> 来源：论文 Figure 4，§4.2，<https://arxiv.org/abs/2607.19191>

---

## 4. 三阶段因果学生训练

### 4.1 双向教师

双向教师可在完整时间轴交换信息，因此更容易学习高质量外观、动作响应和运动先验，但不满足在线推理的因果约束。论文先全参数适配预训练视频生成器，再把知识逐级转移到部署结构相同的因果学生。

### 4.2 Stage 1：Teacher Forcing

学生从教师初始化，只能看干净真值历史与当前 noisy chunk，并用 causal mask 禁止访问未来视觉 token。这一步完成信息结构转换，但存在典型 exposure bias：训练看真值历史，部署看自己的生成历史。

### 4.3 Stage 2：Causal ODE Distillation

固定 Stage 1 因果模型作为多步参考。从其 probability-flow ODE 的中间状态 $oldsymbol z_s^c$ 积分到干净端点 $oldsymbol z_0^c$，少步学生直接回归该端点：

$$\mathcal L_{\mathrm{ODE}}=\mathbb E\left[\left\|f_\theta(\boldsymbol z_s^c,s,\mathcal C_t)-\operatorname{sg}(\boldsymbol z_0^c)\right\|_2^2\right].$$

它解决的是**单个 chunk 的采样步数与延迟**，并不自动解决多个 chunk 串联后的分布漂移。

### 4.4 Stage 3：LongForcing

LongForcing 让学生在训练中连续使用自身生成结果作为上下文，主动进入带误差的后期状态，再由扩展时域双向教师通过 Distribution Matching Distillation 纠正分布。它与 ODE 蒸馏的分工非常清楚：前者减少长期 rollout drift，后者压缩局部去噪过程。

![](assets/ABot-World-0%20-%20单机实时长程交互世界模型/figures/02-training-pipeline.png)

> **图 3：双向教师到因果学生的渐进训练。** Teacher Forcing 先建立因果能力，ODE Distillation 压缩去噪步数，LongForcing 最后处理学生自生成历史上的长程分布偏移。该顺序说明 LongForcing 不是替代教师强制，而是专门补训练—部署失配。  
> 来源：论文 Figure 3，§4.3，<https://arxiv.org/abs/2607.19191>

论文未给出一个可独立复现的 LongForcing 完整损失公式，也未公开教师跨度、rollout 长度分布和 DMD 权重等全部配置。因此其机制可理解，训练配方仍不完全透明。

---

## 5. 单卡实时推理栈

### 5.1 Chunk-wise streaming 与响应延迟

系统每个 chunk 生成 3 个 latent frames，经 VAE 解码为 12 帧视频。整块完成解码后第一张响应帧才可展示，所以吞吐和交互延迟不是同一个指标：最高配置接近 16 FPS，但按键到首帧仍为 1.2 s。增大 chunk 能提高吞吐，却会恶化控制响应，这是系统固有权衡。

### 5.2 每一层优化解决什么

- **LightVAE**：剪枝并简化解码器，使原本 OOM 的 720p 端到端系统首次可运行；
- **SageAttention2**：降低高分辨率长序列 attention 成本，但单独启用仍 OOM；
- **FP8 / MXFP6 / MXFP4 DiT**：量化计算密集线性层，数值敏感模块保持高精度；
- **Fast-RoPE**：在局部窗口重锚定时间位置，用 Triton kernel 避免反复处理完整历史；
- **有界 KV cache**：固定局部上下文与滚动淘汰，使显存不随总时长线性增长；
- **模块调度**：按执行顺序换入/换出 DiT、VAE 与文本编码器，降低同时驻留峰值。

### 5.3 Table 2 的真实系统数据

| 配置 | DiT ms/chunk | VAE ms/chunk | FPS | 峰值显存 GiB |
|---|---:|---:|---:|---:|
| Base | — | — | OOM | OOM |
| + SageAttention2 | — | — | OOM | OOM |
| + LightVAE | 1191.081 | 78.276 | 9.117 | 20.491 |
| + FP8 | 845.180 | 75.980 | 12.405 | 15.925 |
| + FP8 + Fast-RoPE | 786.871 | 71.730 | 13.269 | 19.281 |
| + MXFP6 + Fast-RoPE | 718.281 | 85.994 | 14.098 | 18.287 |
| + MXFP4 + Fast-RoPE | 638.843 | 72.957 | **15.831** | **17.148** |

这里最重要的判断是：**“16 FPS”对应更激进的 MXFP4 性能点，不是默认高质量 FP8 点。**论文没有在同一表中给出各量化配置的感知质量退化，因此不能把最快点直接理解为无损实时。

---

## 6. 实验结果与消融

### 6.1 WorldRoamBench

| 模型 | Strict Acc. | Partial Acc. | Traj. | Aesthetic | Imaging | Mechanics | Memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Genie 3 | 0.4700 | 0.6608 | 0.6719 | 0.4711 | **0.4757** | **0.5454** | 0.6073 |
| HappyOyster | **0.5317** | **0.7631** | **0.7737** | **0.5235** | 0.4377 | 0.5395 | **0.6309** |
| LingBot-World 14B | 0.3235 | 0.4198 | 0.4094 | 0.2898 | 0.2875 | 0.2777 | 0.3006 |
| HY-World 1.5 8.3B | 0.1640 | 0.2088 | 0.2015 | 0.1400 | 0.1236 | 0.1115 | 0.1562 |
| **ABot-World-0 5B** | 0.5266 | 0.7290 | 0.6752 | 0.5039 | 0.4651 | 0.5223 | 0.5041 |

ABot-World-0 在动作、轨迹和视觉指标上具有竞争力，并明显超过表中的两个开源较大模型；但 Mechanics 和 Memory 均不是最优，尤其 Memory 低于 Genie 3 与 HappyOyster。因此“总体有竞争力”有表格支持，“长期记忆领先”没有。

### 6.2 LongForcing 消融

Figure 10 在 60 秒 rollout 上比较 adapted Causal-Forcing-style baseline 与 LongForcing。两者都使用学生自生成历史和最终 DMD，后者额外使用扩展时域教师。后半程差异明显扩大：LongForcing 的 HPSv3 更高，高饱和像素率、模糊分数与 patch 重复率更低。

![](assets/ABot-World-0%20-%20单机实时长程交互世界模型/figures/04-longforcing-ablation.png)

> **图 10：LongForcing 的逐帧消融。** 四条曲线支持“扩展时域教师降低 60 秒内视觉误差累积”；它没有度量世界几何、对象持久性或小时级状态正确性，因此证据范围不能外推为“无限一致”。曲线、图例和坐标轴已视觉核验完整。  
> 来源：论文 Figure 10，§5.2，<https://arxiv.org/abs/2607.19191>

---

## 7. 作者声称、论文实测与未证明部分

| 层级 | 可以成立的表述 | 不能越界的解释 |
|---|---|---|
| **系统实测** | RTX 5090、1280×704、最高 15.831 FPS、1.2 s 首帧延迟、对应显存数据 | 不能推断其他消费卡也有同等性能 |
| **统一 benchmark** | WorldRoamBench 所列维度有明确横向数值 | ABot 并非所有列第一，Memory/Mechanics 也未领先 |
| **受控消融** | LongForcing 在 60 秒四个视觉指标上优于适配 baseline | 不能证明小时/天级空间状态严格正确 |
| **作者定性展示** | Figure 5–7 展示小时级和天级采样关键帧 | 不是连续全序列自动评测，也可能有样例选择偏差 |
| **案例级物理现象** | Figure 9 展示阻挡、足迹、水面扰动等案例 | 模型没有显式物理保证，不能称为通用物理模拟器 |

---

## 8. 局限与未来工作

### 8.1 论文明确局限

动作仍局限于离散键盘控制；局部 KV cache 会遗忘远程信息；LongForcing 只能缓解、不能消除长期漂移；高分辨率实时性仍受 VAE、attention 和内存传输限制；量化感知训练尚未实施。

### 8.2 评测与开放性局限

数据总规模和来源比例不透明；教师权重与完整训练数据并未随论文全部开放；最快量化点缺少质量表；小时级与天级证据主要是定性展示；1.2 s 控制延迟距离高响应游戏体验仍有明显差距。

### 8.3 作者提出的方向

作者明确提到多尺度 LongForcing、持久场景记忆、连续或语义控制、更高效视频解码、上下文压缩和 quantization-aware training。它们恰好对应当前系统的三条主瓶颈：控制表达力、远期状态保存和单卡延迟。

---

## 9. 复现与实践建议

公开仓库与 5B 因果学生权重足以研究推理、量化和交互前端，但不等于完整复现训练。实践时应分别报告：量化格式、分辨率、chunk 大小、DiT/VAE 时间、吞吐、action-to-first-frame、峰值显存和持续运行时长；只报 FPS 会掩盖响应延迟和质量配置差异。长程评测也应加入重访一致性、对象状态、动作成功率与完整视频自动指标，而非只截取关键帧。

---

## 10. 个人判断与材料来源

### 10.1 个人判断

LongForcing 最有价值的思想是把问题从“给干净历史预测得准不准”改写为“模型进入自己的错误分布后还能否被教师拉回”。但 ABot-World-0 真正让该思想落地的，是整套推理栈而不是单一训练算法。其研究意义在于证明 5B 级系统可以进入单桌面 GPU 的可交互区间；其尚未解决的问题，则是如何让长期世界状态具有可验证的持久性，而不只是视觉上没有立即崩坏。

### 10.2 材料来源

本地官方 PDF、四张源码原图及链接清单见：

- `assets/ABot-World-0 - 单机实时长程交互世界模型/sources/paper.pdf`
- `assets/ABot-World-0 - 单机实时长程交互世界模型/sources.md`
