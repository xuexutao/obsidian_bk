---
type: paper-note
status: complete
domain: VLA
paper: "GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation"
year: 2025
arxiv: 2512.01801
doi: null
source: https://arxiv.org/abs/2512.01801
project: https://seed.bytedance.com/gr_rl
code: null
tags:
  - VLA
  - Reinforcement-Learning
  - Offline-RL
  - Real-World-RL
  - Distributional-Critic
  - Flow-Matching
  - Dexterous-Manipulation
  - Bimanual-Manipulation
  - ByteDance-Seed
  - GR-Series
created: 2026-10-21
updated: 2026-10-21
---

# GR-RL：长时程灵巧真机强化学习

## 基本信息

- **核心贡献者**：Yunfei Li、Xiao Ma、Jiafeng Xu；团队：ByteDance Seed；监督：Hang Li、Yonghui Wu
- **版本**：arXiv 2512.01801v3，2025-12-23
- **任务**：ByteMini-v2 双臂系鞋带，涵盖柔性体交互、毫米级穿孔、双手交接、长时程恢复
- **起点模型**：GR-3；Qwen2.5-VL-3B-Instruct + Action Diffusion Transformer
- **完整模型**：VLA policy + 多任务分布式 critic，总参数 5B；在线只新增/训练 51.5M latent noise predictor 与 critic heads
- **训练阶段**：离线 RL 学任务进度并过滤示范 → 形态对称增强 → latent-noise 在线 off-policy RL
- **主要结果**：GR-3 45.7% → Filtered BC 61.6% → +Aug 72.7% → GR-RL 83.3%
- **官方材料**：<https://arxiv.org/abs/2512.01801>；<https://seed.bytedance.com/gr_rl>

## 一句话结论

GR-RL 不是把 GR-3 再做一次更大规模预训练，而是回答“通用 VLA 如何变成可靠真机专家”：它把稀疏终局奖励下的分布式 Q 值解释为任务进度，用该进度删除人类示范中犹豫、失败尝试和倒退片段，再利用双臂形态对称做镜像增强，最后冻结大部分基础策略、只学习 51.5M 参数的 latent-noise predictor 去 steering flow policy；在系鞋带任务上四阶段把成功率从 45.7% 提到 83.3%，直接兑现 GR-3 论文“用 RL 解决 OOD 恢复与模仿上限”的 future work。

---

## 1. 研究背景与问题定义

### 1.1 通用不等于可靠

[[GR-3 - 通用机器人基础模型]] 已能处理抽象指令、未见对象、table bussing 与布料操控，但仍是 imitation learning。模仿学习默认人类示范近似最优；在精细长程操作中，这一假设不成立。遥操作员会减速、犹豫、抓错后重试，动作 chunk 还会因滑动窗口产生多模态冲突。策略若无差别克隆所有片段，就会把示范中的不确定性与错误一并学进去。

第二个问题是训练—部署错位。离线训练拟合原始 teleoperation chunk；部署时为平滑，系统使用 temporal ensembling、异步 receding horizon、trajectory optimization 等后处理，机器人真正执行的轨迹不再等于训练标签。毫米级穿鞋眼任务中，这种小偏差足以导致失败。

### 1.2 为什么选系鞋带

鞋带是柔性细长物体，鞋本身也会形变；鞋眼尺度要求毫米级定位；完整任务包括拿起鞋带、穿过正确鞋眼、另一夹爪接手、拉出并放回桌面，任何阶段错误都会累积。传统方法通常依赖预定义 primitive 和固定穿线 pattern，难以对未见位姿、掉落、交叉鞋带或失败重试泛化。

GR-RL 用该任务作为“通用 base → 高可靠 specialist”的压力测试。它不声称一个 RL checkpoint 仍保留 GR-3 的全部通用能力，而是展示如何在真实世界闭环中对高价值任务做专家化。

### 1.3 两个根本瓶颈

1. **示范次优**：需要自动识别并过滤局部倒退，而不是人工逐帧标注；
2. **部署错位**：需要让策略在自己实际执行的闭环分布上学习，而不是只看 teleoperation。

对应的解决路线是：离线分布式 critic 做进度评估和过滤，在线 latent-space RL 做部署对齐；形态对称增强则在两者之间提升基础策略覆盖。

![](assets/GR-RL%20-%20长时程灵巧真机强化学习/figures/01-pipeline.png)

> **图 1：GR-RL 三阶段管线（论文 Figure 1，p.2）。** 左到右分别是离线过滤行为克隆、形态对称增强、在线强化学习；下方展示拿鞋带、穿鞋眼、拉鞋带。图的重点是每一阶段解决不同误差来源，而非直接在 5B VLA 上端到端暴力在线 RL。来源：<https://arxiv.org/abs/2512.01801>。

---

## 2. 输入输出与基础模型

### 2.1 Policy

策略 $\pi_\theta$ 输入语言 $l$、三视角 RGB 观测 $o_t$（头部、左腕、右腕）和 proprioception $s_t$，输出长度为 $k$ 的双臂 action chunk：

$$a_t=a_{t:t+k}=\pi_\theta(l,o_t,s_t)$$

架构沿用 GR-3：Qwen2.5-VL-3B-Instruct 提供 VLM backbone，Action Diffusion Transformer 以 flow matching 生成连续动作，并只读取 VLM 后半层 KV-cache。推理还叠加 trajectory optimization，约束 jerk 与时间连续性。

### 2.2 Critic

多任务 critic $Q_\phi(o_t,l,s_t,a_t)$ 是因果 Transformer，采用 Q-chunking：对 action chunk 中每个动作预测一段 Q 值。它不是标量无限回归，而是在 $[0,1]$ 上预测离散 value distribution。训练用 cross-entropy 拟合 target distribution。

有界分布的两个作用：一是表达不确定性；二是在长程稀疏奖励下避免早期状态 Q 值严重过估。由于成功终点价值接近 1、失败/远离目标接近 0，分布均值可自然解释为进度。

### 2.3 总参数与模块分工

Policy + critic 总计 5B。在线阶段不是更新全部 5B，而是在共享 VLM 后增加 51.5M noise predictor $\pi_{\theta'}$，另训练原动作空间 critic $Q_\phi$ 和 noise-space critic $Q_{\phi'}$。这一参数隔离既提高样本效率，也降低大模型在线更新的不稳定性；代价是轻量 predictor 容量可能不足，论文将其列为 behavior drift 的潜在原因。

![](assets/GR-RL%20-%20长时程灵巧真机强化学习/figures/02-model-architecture.png)

> **图 2：GR-RL 模型（论文 Figure 2，p.3）。** VLM 的 KV-cache 同时条件化 Action DiT 与 critic；右侧分布式 critic 用 cross-entropy 学 target value distribution。与 GR-3 相比新增的是价值建模与 policy steering，而不是更换 VLM。来源：<https://arxiv.org/abs/2512.01801>。

---

## 3. 阶段一：离线 RL 任务进度与过滤行为克隆

### 3.1 稀疏奖励构造

论文使用 TD3+BC 训练离线 critic。对于长度为 $T$ 的轨迹 $\tau$，只有最后一个 action chunk 覆盖的尾部时间获得成功奖励：

$$r(o_t,l,s_t,a_t)=\begin{cases}\gamma^{T-t}\mathbb{I}(\tau),&t>T-k\\0,&t\leq T-k\end{cases}$$

$\mathbb{I}(\tau)$ 表示轨迹最终是否成功。折扣项让靠近成功的时刻价值更高。数据里多数轨迹最终成功，但内部可能有多次失败与重试；若只给整条轨迹 success label，critic 可能把错误尝试也当成正样本。

### 3.2 Hindsight 失败轨迹

人工只需标记 retry keyframe，而非逐帧标优劣。若成功轨迹中标记 $M$ 个重试点 $m_i$，则把前缀 $\tau_{0:m_i}$ 作为 $M$ 条失败轨迹。这样同一示范被分解为“到失败尝试为止的负前缀”和“最终完整成功轨迹”，TD 学习能识别导致倒退的局部动作。

这一设计的标注成本低于全量 preference 或 dense reward，但仍依赖 retry keyframe。论文未披露 keyframe 数量、标注一致性和自动检测可行性。

### 3.3 用 Q 分布均值定义进度

对数据集中每个 transition，定义：

$$\rho_t:=\operatorname{mean}(Q_\phi(o_t,l,s_t,a_t))$$

若 action chunk 对应的未来进度序列 $\rho_{t:t+k}$ 中出现大于阈值 $\delta$ 的下降，则该样本视为次优并从行为克隆数据中删除。过滤后的 policy 仍用普通 BC/flow matching 学习，不需要把 Q 直接放进离线 policy loss。

这是一种“RL 做数据清洗，监督学习做主策略”的保守方案。它避免离线 actor 在有限数据上做激进 Q 最大化，也把 critic 错误限制在样本筛选层。

### 3.4 为什么不用时间回归

时间回归 baseline 直接让状态预测 $t/T$。它会把进度看成单调时间，无法识别重试、暂时退让和长期有益的倒退。例如机器人放下鞋带以调整抓点，短期动作远离目标，但长期价值提高；GR-RL critic 会产生明显正跳，时间回归几乎平坦。穿孔差几毫米或未完全拉出时，时间回归也过度平滑。

分布式 critic 与非分布式 TD3+BC 的比较进一步显示：后者在长程稀疏奖励的早期部分严重过估，输出还无界；有界 categorical distribution 与真实时序更一致。

---

## 4. 阶段二：形态对称增强

### 4.1 观测、状态、动作与语言同步镜像

双臂任务有天然左右对称。增强步骤为：

1. 所有图像水平翻转；
2. 左右腕相机图像互换；
3. proprioception 与动作先按世界坐标镜像，再转换回各腕局部坐标；
4. 语言中的空间描述同步翻转，例如“左边的孔”改为“右边的孔”。

如果只翻图像而不变动作/语言，会制造错误监督；GR-RL 的价值在于完成了跨模态一致变换。

### 4.2 作用机制

系鞋带中左右手角色与鞋眼方向高度相关，原始数据可能偏向固定手或固定鞋姿。镜像等价于补充对称构型，扩大状态—动作覆盖，尤其能改善双手交接和穿孔阶段。成功率从 Filtered BC 的 61.6% 提到 72.7%，说明简单几何先验在高成本真机数据中非常有效。

### 4.3 边界

该增强依赖本体、相机和任务近似左右对称。若工具、环境或关节限制不对称，镜像动作可能不可行。论文没有报告单独按鞋颜色、左右鞋、鞋眼方向分层的收益，也未说明是否对所有语言自动规则替换还是人工模板。

---

## 5. 阶段三：latent-noise 在线强化学习

### 5.1 为什么不直接在动作上加噪声

毫米级穿孔下，直接给腕位姿或关节加噪声几乎只会失败，难以探索到更好动作。Flow policy 本来从初始高斯噪声出发去噪；改变初始 latent noise 可以产生结构化、仍接近 BC 分布的整段动作变化。因此 GR-RL 学一个 noise predictor $\pi_{\theta'}$，让它根据观测输出 Action DiT 的初始噪声 $\epsilon_t$。

### 5.2 Noise-space actor objective

为了不让 predictor 远离原高斯先验，目标包含 Q 最大化与 norm 约束：

$$\mathcal{L}(\pi_{\theta'})=\mathbb{E}_{(o_t,l,s_t)\sim\mathcal{D}}\left[-Q_{\phi'}(o_t,l,s_t,\epsilon_t)+c\max\left(\frac{1}{2}\|\epsilon_t\|_2^2-\beta,0\right)\right]$$

其中 $\epsilon_t\sim\pi_{\theta'}(o_t,l,s_t)$。只有当噪声能量超过阈值 $\beta$ 才处罚，允许在有限区域内 steering。

### 5.3 把动作价值蒸馏到噪声空间

如果直接通过 flow model 从动作 critic 反传到噪声 predictor，成本高且不稳定。论文训练 noise-space critic $Q_{\phi'}$ 拟合原动作空间 critic 对去噪动作的 value distribution：

$$\mathcal{L}(Q_{\phi'})=\operatorname{CE}\left(Q_{\phi'}(o_t,l,s_t,\epsilon_t),Q_\phi(o_t,l,s_t,\pi_\theta(o_t,l,s_t\mid\epsilon_t))\right)$$

训练 $Q_{\phi'}$ 时，$\epsilon_t$ 以 0.5 概率来自 $\mathcal{N}(0,I)$，否则来自当前 predictor。混合采样扩大 noise space 覆盖，避免 critic 只在狭窄当前策略分布上拟合。

### 5.4 Offline-to-online buffer

系统维护 off-policy 与 on-policy 两个 buffer，并等量采样。训练前，用离线 checkpoint 的 673 条真机 rollout 预热 off-policy buffer，并继续训练 $Q_\phi,Q_{\phi'}$，使 critic 对齐模型自身分布。论文明确不混入 teleoperation 轨迹，避免再引入人与策略动力学不匹配。on-policy buffer 只保留最近两个 checkpoint 的轨迹，旧数据转入 off-policy buffer。

每收集 12 个新 episode，就对 noise predictor 和两个 critic head 更新 50 步。前几轮因 offline→online 分布移动，成功率先下降，随后超过 90% 的移动平均；最终使用 500 online optimization steps 的 checkpoint 评测，成功率 83.3%。报告未披露完整真实交互 episode 总数和墙钟时间，Figure 5 横轴可见约 160 个 episode，但不应把图上终点当作严格公布总量。

---

## 6. 机器人系统与任务

### 6.1 ByteMini-v2

ByteMini-v2 有双 7-DoF arm、双 1-DoF gripper、2-DoF head、1-DoF lift、3-DoF omni base，配头部 RGB-D、双腕 RGB-D、3D LiDAR。相较 GR-3 的 v1：肘部峰值扭矩从 17 Nm 提到 35 Nm，单臂峰值负载从 1.4 kg 提到 3.15 kg；底盘投影从 500×720 mm 缩到 450×650 mm；线缆封装和屏幕位置也改善。

这些硬件升级并非系鞋带成功率的直接消融变量，但说明 GR-RL 结果依赖更新后的本体。跨论文比较 GR-3→GR-RL 时，应避免把所有变化都归因于算法。

### 6.2 任务与奖励

观测是三路 RGB、proprioception 和语言。完整成功要求鞋带穿过正确鞋眼并完全放回桌面；在线 RL 只有成功时奖励 1，否则 0。稀疏奖励使 credit assignment 极难，因此离线 progress critic、强基础策略和 noise-space 结构化探索缺一不可。

### 6.3 评测阶段

论文按四个关键阶段统计：拿到正确鞋带、穿入正确鞋眼、双手交接成功、把鞋带拉出/拉紧并完成。阶段成功率有助于区分“识别/抓取”“毫米级穿孔”“双手协调”“最终执行”各自瓶颈。

---

## 7. 实验结果与消融

### 7.1 主结果：三阶段逐步增益

| 模型 | 完整成功率 | 相对前一阶段增益 |
|---|---:|---:|
| GR-3（全量示范 BC） | 45.7% | — |
| Filtered BC | 61.6% | +15.9 pp |
| Filtered BC + Aug. | 72.7% | +11.1 pp |
| GR-RL（+ online steering） | **83.3%** | +10.6 pp |

从 45.7% 到 83.3% 总增益 37.6 个百分点，错误率从 54.3% 降到 16.7%，相对错误减少约 69.2%。三个阶段都贡献两位数百分点，表明不存在单一“魔法模块”：先清洗示范得到可用起点，再通过对称先验扩大覆盖，最后闭环纠正部署错位。

![](assets/GR-RL%20-%20长时程灵巧真机强化学习/figures/03-training-results.png)

> **图 3：主结果与在线曲线（论文 Figure 5，p.8）。** 左侧显示 0.457→0.616→0.727→0.833；右侧在线初期因分布偏移下降，随后快速恢复并超过离线水平。点是二值 episode reward，曲线是窗口 24 的移动平均。来源：<https://arxiv.org/abs/2512.01801>。

### 7.2 分阶段失败归因

Figure 6 的阶段累计成功率：

| 模型 | 拿起鞋带 | 穿入鞋眼 | 交接成功 | 最终拉出/完成 |
|---|---:|---:|---:|---:|
| GR-3 | 90.6% | 58.7% | 46.4% | 45.7% |
| Filtered BC | 92.8% | 71.0% | 63.8% | 61.6% |
| Filtered BC + Aug. | 96.2% | 76.5% | 74.2% | 72.7% |
| GR-RL | **97.9%** | **89.6%** | **83.3%** | **83.3%** |

过滤与在线 RL 对“穿入鞋眼”提升最大，符合它们分别删除次优精细动作、在实际部署分布中纠正毫米级误差。增强对所有阶段均有提升，尤其交接从 63.8% 到 74.2%，符合左右对称覆盖的作用。

![](assets/GR-RL%20-%20长时程灵巧真机强化学习/figures/04-stage-results.png)

> **图 4：阶段成功率（论文 Figure 6，p.9）。** 实色区域是到当前阶段仍成功的比例，斜线表示相对前一阶段损失。GR-3 最大断点在穿鞋眼；GR-RL 显著缩小该断点。来源：<https://arxiv.org/abs/2512.01801>。

### 7.3 Progress evaluator 消融

时间回归 $t/T$ 会平滑穿孔失败、鞋带滑落和长期有益重抓；非分布式 TD critic 在轨迹早段过估。分布式 critic 受 $[0,1]$ 支撑约束，顺序更合理，并能在微妙失败时下降。

![](assets/GR-RL%20-%20长时程灵巧真机强化学习/figures/05-progress-ablation.png)

> **图 5：分布式进度与回归 critic（论文 Figure 7，p.9）。** 灰色回归曲线在早期维持异常高值，蓝色 GR-RL 进度从低到高并与成功时序一致。该图验证了“有界分布”而非一般 Q 回归的重要性。来源：<https://arxiv.org/abs/2512.01801>。

### 7.4 可视化恢复行为

GR-RL 展示多种闭环行为：不同颜色/尺寸鞋；鞋带掉落后重抓；未穿准时重试；主动把鞋带放到鞋面再调整抓点；先旋正鞋；把远处鞋拉近并调整鞋带；鞋带交叉时找出压在下方的正确端。这些行为说明策略不只是重复固定轨迹。

但论文没有与 BC 在相同异常初始状态上的大规模量化，也没有区分这些恢复是在线 RL 新学、还是基础 GR-3 已有而被成功筛选。定性案例支持能力存在，不足以单独证明机制来源。

---

## 8. 与 GR 主线及同类 RL 的关系

### 8.1 GR-3 的直接续篇

GR-3 局限明确写道：模仿策略进入 OOD 后无法恢复，未来要加入 RL。GR-RL 直接复用 Qwen2.5-VL-3B、Action DiT、后半层 KV-cache 与 ByteMini 系统，论文也称 starting point is GR-3。因此 GR-RL 是主线中“从 foundation generalist 到 reliable specialist”的直接分支。

它不叫 GR-4，原因在于目标不是更新通用基础模型，而是建立专家化训练框架。最终 checkpoint 是否保留 GR-3 的开放语义与跨任务能力没有评测。

### 8.2 与 GR-1/GR-2 的更长关系

GR-1/2 解决“外部 Web 视频如何减少机器人数据依赖”；GR-3 解决“VLM 知识、人类轨迹和双臂移动如何形成通用策略”；GR-RL 解决“高质量示范仍不足时，如何用真实闭环经验超过模仿”。问题依次从表示/动态、规模、语义/本体，转移到可靠性/精度。

### 8.3 与 π*0.6 等同期工作

论文讨论同期 π*0.6：两者都使用分布式 critic 学任务进度。区别是 GR-RL 先用 Q 进度做 filtered BC，获得更强离线 base，再做 latent-noise steering；π*0.6 使用 advantage-conditioned denoising。GR-RL 还专门处理双臂形态镜像和部署后处理 mismatch。

### 8.4 BagelVLA 的边界

[[BagelVLA - 交错式视觉语言动作生成]] 是 Seed 参与的合作线，关注语言规划、关键帧预测与动作流；它不是 GR 编号主线，也不是 GR-RL 的 base。GR-RL 的 base 明确是 GR-3。BagelVLA 可作为“预测/规划提升长程”的并行证据，但不能被写成 GR-RL 的前身或 GR-4。

---

## 9. 局限、失败与批判性评价

### 9.1 论文明确局限

作者指出在线 RL 存在 **behavior drifting**：稀疏、噪声奖励下行为不稳定。可能原因是 51.5M 轻量 noise predictor 容量有限，或大 latent action space 中 credit assignment 过难。作者还提出把改进后的专家策略蒸馏回 base VLA，以同时保留通用与专长，这是 future work。

### 9.2 实验范围

1. 只有一个任务族——系鞋带；虽高难，但无法证明框架对插销、装配、布料折叠等普适；
2. 主要 baseline 是 GR-3 的阶段版本，没有与 π*0.6、SERL、HIL-SERL 等在同平台直接比较；
3. 未披露人类示范总数、过滤比例、$\delta$、$\gamma$、distribution atoms、$\beta$、$c$、chunk 长度、训练硬件；
4. 真机评测次数可由成功率分数猜测，但论文未在正文明确，不能反推样本量；
5. v1→v2 硬件升级与算法更新同时发生，跨 GR-3 的 45.7% base 虽在 v2 上重新训练，但系统变化仍影响系列外比较；
6. 奖励依赖“穿对鞋眼并放下”的自动/人工判定，论文未说明 reward sensor 与误标率。

### 9.3 机制风险

- progress critic 若误判“暂时倒退但长期有益”，过滤会删除恢复行为；作者给出放下重抓的正例，但没有全量 precision/recall；
- 只保留最近两 checkpoint 的 on-policy 数据减少 stale bias，也可能忘记旧失败模式；
- 不混 teleoperation 到 online buffer 避免 dynamics mismatch，但会降低多样性；
- noise norm 约束只限制能量，不保证动作语义安全；
- 专家化可能造成通用语言/对象能力遗忘，论文未测。

### 9.4 个人判断

GR-RL 最可靠的证据是阶段增益与失败归因一致：过滤主要修复示范噪声，镜像主要改善双臂覆盖，在线 RL 主要修复穿孔和部署错位。相比只报告最终 83.3%，这种分解更有说服力。

最值得推广的组件可能不是在线 RL，而是“分布式 Q 作为进度 → 数据过滤”。它把困难 RL 转为稳健监督学习，能复用失败前缀并减少 actor extrapolation。在线阶段仍昂贵且不稳定，适合在强 base 之后做最后 10 个百分点，而不是从弱策略开始探索。

---

## 10. 复现成本与实践建议

### 10.1 必需资源

- 4B GR-3 base 或等价 Qwen2.5-VL + flow action model；
- 5B policy+critic 训练能力，多视角 RGB 与双臂状态；
- ByteMini-v2 或近似双臂移动机器人、轨迹优化、可靠自动复位；
- 大量高精度 teleoperation 示范及 retry keyframe 标注；
- 在线真机交互、成功奖励判定和安全监控。

### 10.2 论文未披露导致的复现不确定性

初始示范总数、成功/失败比例、过滤掉多少 transition、critic atom 数、阈值 $\delta$、折扣 $\gamma$、actor 正则 $c$ 与 $\beta$、在线学习率、batch、策略频率均缺失。项目没有公开代码链接。因而可以复现思想，但难以严格复现实验曲线。

### 10.3 建议的分阶段复现

1. **先验证 progress**：在离线轨迹上标少量 retry，比较 $t/T$、标量 TD、categorical TD 的失败检测 AUROC；
2. **再验证过滤**：报告过滤比例、保留轨迹覆盖和 BC 成功率，防止只保留短/简单轨迹；
3. **镜像单测**：逐维检查世界坐标镜像、局部 wrist 转换、相机交换和语言替换；
4. **仿真 noise steering**：先在可复位精密插孔任务中验证 latent noise 的探索半径；
5. **真机 warm start**：用当前 policy rollout 重新拟合 critic，再在线 actor 更新；
6. **安全措施**：动作/速度/力矩边界、碰撞检测、人工急停、异常状态自动终止；
7. **通用性回归**：专家化前后在 GR-3 原 pick-and-place/VL 指令集上测试遗忘。

### 10.4 计算与真机成本判断

在线只训练 51.5M predictor 和 critic heads，比全量 5B 更新便宜；但每次更新仍需 VLM 前向和真机 12-episode 批次。真正昂贵的是机器人占用、人工复位和安全监督，而不是参数量。论文没有墙钟时间，因此不应给出具体小时估计。

---

## 11. 启发与后续问题

### 11.1 已确立事实

- 过滤示范、镜像增强和在线 steering 在同一任务上依次带来 +15.9、+11.1、+10.6 pp；
- 分布式 critic 比时间回归和非分布式 critic 更符合长程稀疏奖励的进度；
- 在线训练先退化后恢复，说明 offline-to-online shift 真实存在；
- GR-RL 能处理多种扰动和主动重试案例。

### 11.2 尚未确立

- 对其他长程任务是否同样有效；
- 恢复行为有多少由 RL 新产生；
- 83.3% 是否在更多鞋型、鞋眼数量、光照和背景下保持；
- 专家策略是否保留 GR-3 的通用能力；
- noise predictor 容量和 drift 的最佳折中。

### 11.3 后续研究问题

- 能否用视觉语言模型自动标 retry keyframe，减少人工标注？
- progress critic 能否跨任务共享，而不是每个专家任务单训？
- 如何把 Q uncertainty 用于安全拒绝和人工接管？
- 能否把专家 rollout 蒸馏回 base，形成“通用模型—专家—回流数据”的闭环？
- 与 GR-2 式视频 world model 结合后，能否在虚拟 rollout 中预筛选 latent noise，减少真机探索？
- 触觉与力觉能否改善穿孔、滑移和柔性体 credit assignment？
- 怎样设计不会导致 behavior drift 的保守 latent-space actor update？

---

## 12. 材料来源

| 本地材料 | 位置 | 用途 |
|---|---|---|
| 官方论文 PDF | `assets/GR-RL - 长时程灵巧真机强化学习/sources/paper.pdf` | 全文、公式、曲线和局限核对 |
| 来源记录 | `assets/GR-RL - 长时程灵巧真机强化学习/sources.md` | PDF、原图与项目页溯源 |
| Figure 1 | `figures/01-pipeline.png` | 三阶段流程 |
| Figure 2 | `figures/02-model-architecture.png` | VLA + critic 架构 |
| Figure 5 | `figures/03-training-results.png` | 主结果与在线曲线 |
| Figure 6 | `figures/04-stage-results.png` | 分阶段失败归因 |
| Figure 7 | `figures/05-progress-ablation.png` | critic 消融 |

> 本笔记基于本地官方 PDF（16 页）逐页阅读，公式与数字按 v3 核对。论文未披露的复现细节未作推断。五张图来自 arXiv 官方源码并已逐张检查。关联综述：[[ByteDance GR 系列 - Survey]]。
