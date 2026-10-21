---
type: paper-note
status: complete
domain: World-Models
paper: "SparseWorld: Enhancing End-to-End Autonomous Driving via World Models with Sparse Scene Representation"
year: 2026
arxiv: "2605.24354"
doi: "10.48550/arXiv.2605.24354"
source: "https://arxiv.org/abs/2605.24354"
project: "https://wryzju.github.io/SparseWorld/"
code: null
tags:
  - World-Models
  - Autonomous-Driving
  - Sparse-Scene-Representation
  - Planning
  - Instance-Forecasting
  - Safety
created: 2026-10-16
updated: 2026-10-16
---

# SparseWorld：稀疏场景驾驶世界模型

## 基本信息

- **作者与机构：** Ruoyu Wang、Jingke Wang、Yukai Ma、Yuehao Huang、Shuangming Lei、Guanglin Xu、Aixue Ye、Yong Liu；浙江大学与华为 2012 实验室。
- **版本与会议：** arXiv v2，2026-06-22；项目页标注 IROS 2026。主文 8 页，另有 7 页官方 Supplementary。
- **任务：** 从历史六视角观测得到 agent/map instance，按 ego trajectory、velocity、command 自回归预测未来 0.5–2.0 秒稀疏实例，再用 future feature 改善 motion prediction 和 ego planning，并从 base/future/refined 三条候选中选安全轨迹。
- **Baseline：** 插件式接入 SparseDrive-S 与 VAD-Tiny；open-loop 用 nuScenes，closed-loop 用 Bench2Drive 220 routes。
- **重要性评估：** ★★★★★（5/5）。它把显式空间 latent 从稠密 occupancy 进一步压缩为任务相关 instance，并通过 future-feature interaction、Safety-Critical Loss 和 Adaptive Trajectory Selection 直接进入 planning；闭环指标提供了比纯 FID 更强的证据。

## 一句话结论

SparseWorld 认为驾驶 world model 不必重建每个像素或 voxel，只需滚动 900 个 agent anchor 与 100 个 map anchor 的几何+feature；这种稀疏未来以 70 ms、约 4.4 GB 生成四帧，并通过 motion/planning refiner 与碰撞几何约束把 SparseDrive-S 的 nuScenes 平均 collision rate 从 0.08% 降到 0.05%、Bench2Drive driving score 从 44.54 提到 48.95，但 L2 从 0.61 略变差到 0.65，说明安全收益来自明确风险优化而非单纯轨迹拟合。

---

## 1. 研究背景与表示选择

### 1.1 稠密世界模型为什么可能“预测了太多”

驾驶世界模型常预测 RGB 视频、稠密 BEV feature 或 3D occupancy。它们表达力强，却把大量容量用于天空、建筑纹理、空 voxel 等与下一步 trajectory 决策弱相关的内容。人类驾驶更关注关键交通参与者、道路边界和拓扑；SparseWorld 因此提出实例级 world state：只预测 agent 与 map layout，同时保留每个 instance 的高维 feature。

![](assets/SparseWorld%20-%20稀疏场景驾驶世界模型/figures/01-representation-comparison.png)

> **图 1：三类驾驶世界模型表示。** (a) 像素 future；(b) 稠密 BEV/occupancy；(c) agent/map instance future。SparseWorld 选择最接近 planning state 的稀疏表示。来源：论文 Figure 1，第 1 页，官方 TeX 原图；已用 Read 验证。

### 1.2 论文要回答的两个问题

1. 稀疏 instance rollout 是否仍能学习 2 秒 scene evolution，而不是简单复制/投影当前检测？
2. predicted future 是否通过明确机制改善 motion/planning，尤其是 collision 与闭环 driving score？

这一区分很重要。世界模型可以生成准确 future，却不一定被 planner 使用；SparseWorld 将 future instance 直接作为 attention context，另用 safety loss 和 trajectory selection 把预测转成行为改变。

### 1.3 核心贡献

- 提出 Sparse Dreamer：instance memory queue + global instance alignment + action-conditioned sparse decoder。
- 用 future agent/map feature 重新执行 motion predictor 与 trajectory planner。
- 训练时用 Safety-Critical Loss，推理时用 Adaptive Trajectory Selection。
- 在 nuScenes 同时评估 instance forecasting、motion、open-loop planning 与效率；在 Bench2Drive 做 220 route closed-loop。

---

## 2. 输入、输出与核心表示

### 2.1 Instance state

时刻 $t$ 的稀疏 state 为

$$I_t=\{(A_t,F_t)\},$$

其中 $A_t=\{a_{ego},a_{agent},a_{map}\}$ 是几何 anchor，$F_t$ 是对应高维 feature。Agent anchor 含 11 维：中心 $(x,y,z)$、尺寸 $(w,l,h)$、yaw 的 $(\sin\psi,\cos\psi)$ 与速度 $(v_x,v_y,v_z)$。Map element 由在线 mapping query/anchor 表示。

动作条件为

$$C_t=(v_t,\tau_t,s_t),$$

包含 ego velocity、planned trajectory 与 steering command。给历史 $I_{-h:0},C_{-h:0}$，world model 输出

$$\mathcal W(\{I_{-h},\ldots,I_0\},\{C_{-h},\ldots,C_0\})=\{I_1,\ldots,I_f\}.$$

补充材料明确使用两个历史帧+当前帧，预测四个 future timestamp（0.5、1.0、1.5、2.0 秒）。

### 2.2 为什么 feature 不能只留几何框

最终 refiner 不只读取 box/line 坐标，也读取 predicted $F_t$。这些 feature 保留类别、外观、上下文和不确定性，可比纯几何 projection 提供更丰富的 motion cue。其代价是 feature 的可解释性低于 anchor，且未来 feature 的训练 target 依赖 baseline perception query matching。

---

## 3. 核心方法

### 3.1 三阶段系统

![](assets/SparseWorld%20-%20稀疏场景驾驶世界模型/figures/02-framework.png)

> **图 2：SparseWorld 总体流程。** 左上 baseline 从观测得到 instance 与初始 motion/ego trajectory；左下 Sparse Dreamer 自回归未来 instance；右侧 future instance 进入 motion/planning refiner，并由 ATS 在三类候选中选最终轨迹。来源：论文 Figure 2，第 3 页，官方 TeX 原图；已用 Read 验证。

完整数据流是：

1. SparseDrive/VAD 感知当前与历史 agent/map，输出 baseline motion 与 ego trajectory；
2. baseline trajectory、velocity、command 组成 action condition；
3. Sparse Dreamer 预测下一帧 instance，Future Motion Planner 根据预测 state 更新下一步 action condition，再自回归四帧；
4. future instance 替换/补充历史 context，重新预测 agent motion 与 ego candidate；
5. ATS 比较 base、future、refined trajectory 的 collision risk，必要时应用 adjustment vector。

这形成了 planning-conditioned world rollout 与 future-conditioned replanning 的循环，但训练采用分阶段冻结，并非端到端通过多步闭环共同优化。

### 3.2 Global Instance Alignment

先根据 agent 自身速度和 ego motion 做 coarse preprojection。动态 agent 中心先补偿：

$$c_e^{comp}=c_e-V\Delta t.$$

再用 ego angular velocity $\omega_t$ 和规划位移 $(\Delta x_t,\Delta y_t)$ 转换：

$$c'_e=R_z(\omega_t)c_e^{comp}+(\Delta x_t,\Delta y_t,0)^\top,$$

$$\begin{pmatrix}\cos\psi'\\\sin\psi'\end{pmatrix}=R_z^{2\times2}(\omega_t)\begin{pmatrix}\cos\psi\\\sin\psi\end{pmatrix}.$$

静态 map 只做 ego-motion compensation。该模块把确定的刚体运动先验显式写入模型，decoder 只需学习 residual dynamics，既提高精度也减轻坐标系切换误差。

### 3.3 SparseWorld Decoder

![](assets/SparseWorld%20-%20稀疏场景驾驶世界模型/figures/03-sparse-dreamer.png)

> **图 3：Sparse Dreamer。** Instance Memory Queue 提供历史与已预测 future；GIA 提供预投影 anchor；decoder 交替使用 self-attention、temporal cross-attention、action cross-attention 与 cross-last-attention。来源：论文 Figure 3，第 4 页，官方 TeX 原图；已用 Read 验证。

首层 decoder 含 self-attention、cross-temporal、cross-action 与 FFN；后续五层还读取 preprojection output。历史 instance 来自不同 ego frame，模型注入时间 embedding 与由 planner trajectory proposal 计算的 relative-position embedding。Action condition 用 Fourier embedding 注入 cross-attention。

自回归虽可能累积误差，但每步都有明确 anchor preprojection 与 action condition；而且 horizon 仅四帧、2 秒，避免长视频式无界 rollout。

### 3.4 Motion prediction 与 planning refinement

Motion refiner 复用 baseline motion predictor，但把历史 instance context 替换为未来 $I_{1:f}$，让 agent trajectory 直接看预期未来布局。Planning refiner 使 ego feature attend future instance，再用同结构 planning head 输出 refined ego trajectory。

这一步是“world model 影响 planning”的核心路径：不是把 future 转成一张图给人看，而是 future token 与 planner query 直接交互。

### 3.5 Safety-Critical Loss

给 ego trajectory $\tau\in\mathbb R^{T\times2}$、agent anchor 与 agent future trajectory，计算满足 safety distance $\theta$ 的最小调整：

$$v_{adj}=SAV(\tau,a_{agent},\tau^{agent},\theta).$$

补充材料说明先求 ego/agent oriented bounding box 的 minimum distance vector $v_{i,j}$，安全向量为 $s_{i,j}=\theta\hat v_{i,j}$；若 $\|v_{i,j}\|_2<\theta$，则

$$v_{i,j}^{adj}=s_{i,j}-v_{i,j},$$

否则为零。对 agent 平均、跨时间拼接；$\theta=0.5$ m。主文的安全损失为非零 adjustment norm 的均值：

$$\mathcal L_{scl}=\frac{\sum_{t=1}^{T}\|v_t^{adj}\|_2}{\sum_{t=1}^{T}\mathbf1[\|v_t^{adj}\|_2\neq0]}.$$

Refinement 目标为

$$\mathcal L_{refine}=\mathcal L_{mp}+0.1\mathcal L_{scl}.$$

与 VAD 只看 centroid 不同，SCL 使用 future agent 位置、尺寸与朝向；这解释了 collision 下降比 L2 更明显。

### 3.6 Adaptive Trajectory Selection

推理时有三条来源：baseline trajectory、Future Motion Planner trajectory、SCL-refined trajectory。分别与 baseline/refined agent trajectory 做碰撞检测并计算风险，选择无碰撞且 safety loss 最低者；若同样安全，偏好 refined trajectory；最后应用对应 adjustment vector。

![](assets/SparseWorld%20-%20稀疏场景驾驶世界模型/figures/04-trajectory-selection.png)

> **图 4：ATS 定性例。** 原始青色 trajectory 存在碰撞；world-model future 与 refiner 提供额外候选，ATS 选中蓝色安全轨迹。来源：论文 Figure 5，第 6 页，官方 TeX 原图；已用 Read 验证。

---

## 4. 数据、训练与复现成本

### 4.1 数据与评测

- **nuScenes：** 700 train、150 validation scene，2 Hz keyframe；评估 future online mapping mAP、3D detection NDS、motion EPA/MR/mADE/mFDE、planning L2/collision。
- **Bench2Drive：** 1,000 clip base set 训练，220 route 测试，44 类 safety-critical interaction；报告 open-loop L2/efficiency/comfort 与 closed-loop success rate/driving score。

为了保证历史与未来连续帧，作者裁剪了训练/评测序列。nuScenes 官方 test 中裁剪后缺失的 sample 用 baseline result 填充。这一细节会影响 aggregate planning metric，复现时必须一致。

### 4.2 两阶段训练

以 SparseDrive-S 为例：

1. Stage 1 训练 future forecasting，监督四帧；
2. Stage 2 冻结 Stage 1，仅训练 motion/planning refinement，rollout 两帧。

nuScenes：100 epochs + 20 epochs，batch 4，学习率 $2\times10^{-4}$；Bench2Drive：10 + 4 epochs。AdamW + cosine annealing，ResNet50，nuScenes image 256×704、Bench2Drive 384×704。Agent 感知半径 55 m，map 纵向 60 m/横向 30 m；motion 12 steps、planning 6 steps，均 6 modes。

训练使用 8×RTX 4090D 24GB，closed-loop 评测用 8×RTX 3090 24GB。代码/权重截至 2026-10-16 未公开；复现还依赖 SparseDrive/VAD、nuScenes 与 CARLA/Bench2Drive stack。

### 4.3 效率

| 方法 | 表示 | 推理显存 | 感知时间 | 生成时间 |
|---|---|---:|---:|---:|
| UniScene | Image | 23,524 MB | — | 28.29 s |
| Drive-OccWorld | Occupancy | 20,581 MB | 401 ms | 398 ms |
| **SparseWorld-S** | Instance | **4,397 MB** | **174 ms** | **70 ms** |

SparseWorld 预测 900 agent + 100 map anchor 的四帧；比较对象的输出尺寸与实现不同，因此不是严格同 FLOP 条件，但部署差距足够大，支持稀疏表示的效率动机。

---

## 5. 实验结果

### 5.1 Future instance forecasting

SparseWorld-S 在 2 秒时仍有 map mAP 48.96（当前 56.66 的 86.4%）和 NDS 0.433（当前 0.523 的 82.8%）。简单 copy/paste 的 2 秒 mAP/NDS 仅 9.71/0.212，基于 GT projection 的 21.84/0.390。说明模型不只是 ego-motion warp，尤其 map topology 保持显著更好。

SparseWorld-V 也能接 VAD：2 秒保留当前 NDS 的 87.9%、mAP 的 78.7%，证明插件对 sparse 与 BEV dense perception baseline 都适用。

补充材料给 anchor noise：map noise $\sigma=0.9$ 时当前 perception 降至 83.8%，future 平均降至 82.8%；3D detection $\sigma=0.3$ 时为 91.6%/93.4%。future degradation 大体跟随输入 degradation，没有额外灾难性放大，但只覆盖高斯 anchor noise。

### 5.2 Motion prediction

| 方法 | EPA↑ | MR↓ | mADE↓ | mFDE↓ |
|---|---:|---:|---:|---:|
| SparseDrive-S | 0.482 | 0.136 | 0.62 | 0.99 |
| SparseWorld-S | **0.488** | **0.128** | 0.62 | 0.99 |
| VAD-Tiny | 0.598 | 0.121 | 0.79 | 1.08 |
| SparseWorld-V | **0.619** | **0.113** | **0.78** | **1.03** |

SparseDrive 本身较强，主要改善 coverage/miss rate；VAD 上四项都有改善。Agent 与 map future 各自有效，二者同时用时 EPA 0.619、MR 0.113、mADE 0.78、mFDE 1.03 最佳。

### 5.3 nuScenes planning

| 方法 | Avg L2↓ | Avg Collision↓ |
|---|---:|---:|
| VAD-Tiny | 0.78 m | 0.38% |
| SparseWorld-V | **0.59 m** | **0.24%** |
| SparseDrive-S | **0.61 m** | 0.08% |
| SparseWorld-S | 0.65 m | **0.05%** |

在中等 baseline VAD 上 accuracy 与 safety 都提升；在强 baseline SparseDrive 上 collision 下降 37.5%，但 L2 变差 0.04 m。补充材料显示 lane keeping 96.40→96.73、comfort 98.31→98.54，说明 safety gain 并非简单急刹/过度保守，但 L2 trade-off 仍应诚实保留。

### 5.4 Bench2Drive 闭环

SparseDrive-S 的 closed-loop success rate/driving score 为 16.71%/44.54；SparseWorld-S 为 18.23%/48.95，分别相对提高约 9.1% 与 9.9%。MomAD 为 18.11%/47.91，SparseWorld 略高。Open-loop L2 却是 0.84 vs baseline 0.87，提升较小，再次显示闭环收益主要来自对 interaction risk 的处理，而非更贴 expert trajectory。

---

## 6. 消融：各模块怎样改变 planning

### 6.1 Sparse Dreamer

在 SparseWorld-S 上，只有 positional embedding 时 map/NDS future average 为 51.82/45.54；只有 preprojection 为 52.08/45.83；二者都有为 52.87/46.66。GIA 的物理 preprojection 与 learned temporal relation 互补，preprojection略更重要。

### 6.2 Planning refinement

| FIF | SCL | ATS | Avg L2↓ | Avg Collision↓ |
|---|---|---|---:|---:|
| 否 | 否 | 否 | 0.78 | 0.38% |
| 是 | 否 | 否 | 0.62 | 0.34% |
| 是 | 是 | 否 | 0.59 | 0.28% |
| 是 | 是 | 是 | **0.59** | **0.24%** |

Future Instance Feature（FIF）主要改善 L2，说明预测 future 为 planner 提供更好动态 context；SCL 主要降 collision；ATS 在保持 L2 的同时进一步降 collision。这个拆分清楚回答了“world model 怎样影响 planning”，而不是把全部提升归于一个黑箱。

### 6.3 与 dense occupancy world model 的对比

在作者统一表中，Drive-OccWorld 把 occupancy baseline 的平均 collision 0.24% 降到 0.09%；SparseWorld-V 把 VAD 0.22% 降到 0.08%，SparseWorld-S 从 0.02% 降到表中四舍五入的 0.00%。不同 baseline 不宜直接宣称表示优劣，但至少表明 instance-level future 足以支持 collision mitigation，无需 dense occupancy 才能规划。

---

## 7. 两条主轴中的位置

### 7.1 显式空间 latent 的极简化

SparseWorld 比 BEVWorld 更任务导向：它不保留可渲染的整场 BEV，而只保留可用于检测、地图、运动和规划的 query/anchor。几何坐标、尺寸、yaw 和速度明确，适合 collision geometry；feature 负责难以手工表示的语义 context。

这种稀疏性也是风险：未被 detector 实例化的对象在 world model 中不存在，纹理/交通灯/可行驶细节可能丢失。Dense occupancy 对未知障碍与自由空间更完整，instance latent 则在已检测关键对象上更高效。

### 7.2 与隐式生成先验的关系

SparseWorld 不使用大视频生成模型，也不生成像素；它把“世界”定义为 planner 所需 state。VEGA-3D/VLA-World 的隐式先验可用于增强 instance feature 或补全漏检，但最终最好投影到显式 anchor/uncertainty，使 safety constraint 可计算。二者结合比互相替代更合理。

---

## 8. 局限与 Future Work

### 8.1 作者没有独立 limitations 段

主文和 Supplementary 未显式列 future-work 清单。以下根据论文设置区分作者实测与个人判断。

### 8.2 关键局限

1. **依赖 instance perception。** 漏检的 pedestrian/obstacle 不会进入 future rollout；高斯 anchor noise 实验没有覆盖 missing/false-positive/class-switch。
2. **自回归 action loop 可能自洽但错误。** Future Motion Planner 由预测 instance 更新下一步 condition，scene 与 action error 可能互相强化。
3. **规划安全依赖预测几何。** ATS 若低估 agent box/trajectory，可能选择错误“安全”候选；论文未给不确定性校准。
4. **两阶段冻结限制联合优化。** Stage 2 冻结 forecasting，planner 的 safety loss 不能反向塑造 world representation。
5. **对比表示不完全公平。** Image/occupancy/instance 系统的分辨率、backbone、GPU 与任务不同。
6. **closed-loop 仍是 CARLA benchmark。** 真车域差距、sensor failure 与交互式罕见行为未验证。
7. **代码未发布。** 数据裁剪、matching、trajectory selection threshold 与 collision implementation 难以核验。

### 8.3 建议的后续方向

- 为每个 future instance 预测存在概率、轨迹分布与 box uncertainty，并做 risk-aware ATS。
- 引入“未知占用/自由空间”的轻量 dense safety channel，弥补 instance漏检。
- 用 planner loss 端到端微调 forecasting，同时保留独立 future supervision 防止只学 reward shortcut。
- 做 counterfactual action rollout：同一 scene 对多条 ego trajectory 预测 agent response，而不是主要由 baseline action condition驱动单一路径。
- 在闭环中统计 world-model error→planner error 的因果链，而不只报告最终 DS。

---

## 9. 复现建议

1. 先严格复现 SparseDrive-S/VAD-Tiny baseline 与官方 metric，尤其 nuScenes ego-status 去除设置。
2. 只加 Sparse Dreamer，验证 Table I 与 GIA/PE/PP 消融；再加 FIF、SCL、ATS，逐步追踪 L2/collision。
3. 核对 agent/map query 的跨帧 matching 与坐标系；错误匹配会被自回归放大。
4. 复现 supplement 的两阶段训练、四帧监督/两帧 Stage-2 rollout 和 trimmed-sequence 填充规则。
5. 对 ATS 做 failure injection：漏检、速度偏差、yaw 偏差、agent trajectory 多模态冲突。
6. 闭环评测记录每次最终轨迹来自 base/future/refined 哪一支，才能知道 world model 是否真正改变行为。

---

## 10. 个人启发

SparseWorld 的最大价值是把世界模型表示与规划接口共同设计：预测什么，不由“能否生成逼真画面”决定，而由“planner 为何需要未来”决定。FIF 提供未来 context，SCL 把安全几何写入优化，ATS 把不同模型输出变成可选择的行为。它也揭示一个重要事实：更安全的 trajectory 不一定更接 expert L2；闭环 world-model planning 需要把 imitation accuracy 与 risk objective 分开评价。

---

## 11. 材料来源

详见 [sources.md](assets/SparseWorld%20-%20稀疏场景驾驶世界模型/sources.md)。核心依据为 arXiv v2 官方 PDF、官方 Supplementary、项目页与 TeX 源包原图；截至 2026-10-16 未发现官方代码。所有插图均已逐张 Read 验证。