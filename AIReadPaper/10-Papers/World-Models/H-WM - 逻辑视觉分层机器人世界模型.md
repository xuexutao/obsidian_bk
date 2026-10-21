---
type: paper-note
status: complete
domain: World-Models
paper: "H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model"
year: 2026
arxiv: "2602.11291"
doi: "10.48550/arXiv.2602.11291"
source: "https://arxiv.org/abs/2602.11291"
project: null
code: null
tags:
  - World-Models
  - Robotics
  - VLA
  - Task-and-Motion-Planning
  - Neuro-Symbolic
  - Flow-Matching
created: 2026-10-16
updated: 2026-10-16
---

# H-WM：逻辑视觉分层机器人世界模型

## 基本信息

- **作者与机构：** Jinbang Huang、Wenyuan Chen、Zhiyuan Li、Oscar Pang、Xiao Hu、Lingfeng Zhang、Yuanzhao Hu、Mark Coates、Tongtong Cao、Xingyue Quan、Zhanguang Zhang、Yingxue Zhang；华为诺亚方舟实验室、多伦多大学、英属哥伦比亚大学、麦吉尔大学。
- **版本：** arXiv v3，2026-09-29；8 页、3 张主图。
- **任务：** 为长时机器人操作构建双层世界模型：低频逻辑层预测 PDDL 风格 action/state transition，高频执行前由视觉层预测每个子任务的终止 visual latent，再共同引导 π0.5 VLA 输出连续 action chunk。
- **数据与评测：** LIBERO-Logic、RoboCerebra；LIBERO-10、RoboCerebra、作者构建的 LIBERO-LoHo，以及 UR5e 八步真机清桌任务。
- **重要性评估：** ★★★★★（5/5）。五篇中它最直接回答“世界模型如何真正影响 planning”：world model 输出不是旁路生成结果，而是进入 symbolic search、子目标构造、完成检测与 action expert；代价是符号标注、系统复杂度和任务必须可符号化。

## 一句话结论

H-WM 不滚动长视频，而是在两个时间尺度上规划：Qwen3-4B 逻辑世界模型用 beam search 选择可达的 action–predicate state trajectory，300M flow-matching 视觉世界模型只预测每个子任务终点的 PaliGemma latent，guided π0.5 持续以逻辑 action 和固定 visual subgoal 生成动作；在 5–7 步 LIBERO-LoHo 上，它把 π0.5 成功率从 6.4% 提至 64.8%，其中 logic-only 已达 48.4%，visual subgoal 再贡献 16.4 个百分点。

---

## 1. 研究背景与核心矛盾

### 1.1 端到端 VLA 为什么在长时任务崩溃

VLA 直接把图像、语言和 robot state 映射为动作，短任务上表现强，但长时任务会累积三类错误：漏掉中间步骤、顺序错误、提前终止。自然语言子目标可以分解任务，却不显式跟踪世界状态；纯视觉 world model 能给出未来画面，却难保证几十步的任务逻辑；经典 TAMP 具备长时一致性，却依赖手工 predicate/action domain，且与视觉观测对齐弱。

H-WM 的核心判断是：长时稳定性与低层可执行性需要不同表示。符号状态压缩任务进度、适合搜索；视觉 latent 描述“这个子任务完成时应该看起来怎样”、适合给 policy 做感知目标。它们必须在同一个 transition 上对齐，而不是先让 LLM 写计划、再让 VLA 独立执行。

### 1.2 为什么不预测完整视频

视觉 world model 只预测每个子任务的终止 latent，而不是子任务内每一帧。这一设计避开长视频自回归误差和像素生成成本，也使目标在子任务执行期间保持固定。其 world modeling 对象不是完整物理轨迹，而是**逻辑边界间的视觉终态**。

### 1.3 核心贡献

1. 以学习式 logical transition model + trajectory evaluator 替代固定 PDDL transition rules。
2. 用逻辑 action/state 条件预测 terminal visual latent，避免长时视觉 rollout。
3. 为 π0.5 增加 goal expert，把逻辑与视觉两级 guidance 注入 action expert。
4. 构建帧级对齐 RGB、robot state、logical action 与 predicate state 的 LIBERO-Logic。
5. 用长时成功率、Q-Score 与真机逐步成功率直接评估 world model 对执行的影响。

![](assets/H-WM%20-%20逻辑视觉分层机器人世界模型/figures/01-framework.png)

> **图 1：H-WM 与 guided VLA。** 顶部逻辑世界模型搜索 action/state transition；中部视觉世界模型用理解与预测 expert 生成终点 visual latent；底部 guided VLA 的 action expert 同时 attend 当前理解与 goal expert。来源：论文 Figure 1，第 2 页，官方 TeX 原图；已用 Read 验证。

---

## 2. 任务定义与输入输出

### 2.1 符号状态与长时任务

符号 domain 为 $D=(P,A)$：$P$ 是 predicate schema 集合，$A$ 是参数化 action schema。给定对象集合 $O$，ground atoms 构成集合 $G$；某时刻逻辑状态是当前为真的子集 $X\subseteq G$。任务从初态 $X_0$ 到 goal $X_G$，被分为 $M$ 个子任务 transition：

$$X_0\xrightarrow{a_1}X_1\xrightarrow{a_2}\cdots\xrightarrow{a_M}X_M,$$

当 $X_G\subseteq X_M$ 时完成。与经典 PDDL 按手工 precondition/add/delete effect 计算 successor 不同，H-WM 从数据预测 candidate pair $(a_{m+1}^{(i)},\hat X_{m+1}^{(i)})$。

### 2.2 visual subgoal

在子任务边界 $m$，输入为 RGB $obs_m$、robot configuration $q_m$、选中的 logical action $a_{m+1}$ 与 successor state $X_{m+1}$：

$$c_m=(obs_m,q_m,a_{m+1},X_{m+1}).$$

冻结视觉 encoder $\phi$ 把成功执行后的 terminal keyframe 编码为监督目标：

$$f_{goal}^{m+1}=\phi(obs_{goal}^{m+1}).$$

视觉世界模型预测

$$f_{pred}^{m+1}=M_V(obs_m,q_m,a_{m+1},X_{m+1}).$$

该 latent 在整个子任务内固定；VLA 在更高频控制步 $t$ 读取当前 $obs_t,q_t$，并以 $(a_{m+1},f_{pred}^{m+1})$ 为中间 guidance 输出 action chunk $\hat\alpha_{t:t+k}^m$。这是明确的双时间尺度系统：world model 每个 symbolic transition 调一次，policy 每个 control step 调一次。

---

## 3. 核心方法

### 3.1 逻辑世界模型与 beam-style search

逻辑模型从 planning-enhanced Qwen3-4B 初始化，再在 LIBERO-Logic 上全参数微调。推理时同一模型扮演 generator $M_L^{search}$ 与 evaluator $M_L^{eval}$。generator 从当前 state、goal 和历史 trajectory 产生 $K$ 个 action–successor 候选；evaluator 对 partial trajectory $\tau$ 给出一致性分数 $s_{eval}(\tau)$。

显式 goal progress 为

$$s_{goal}(X)=\frac{|X\cap X_G|}{|X_G|},$$

总排名分数为

$$S(\tau)=\lambda_{eval}s_{eval}(\tau)+\lambda_{goal}s_{goal}(X).$$

默认 beam width $B=3$，每次扩展 $K=2$，两项权重均为 0.5，最大深度 20。搜索在 predicted state 满足 goal 时停止。这里 world model 直接承担 planning：它不只是预测下一状态，还评估整条 symbolic trajectory 并选择下一 transition。

### 3.2 逻辑条件视觉世界模型

理解 expert 采用 π0.5 的 PaliGemma backbone，冻结 visual encoder 与 multimodal projector；Gemma 处理视觉 token、$a_{m+1}$ 与 $X_{m+1}$。随机初始化约 300M 参数 prediction expert，接收 $q_m$、noisy visual-subgoal feature 与 flow time，并通过单向 fused self-attention 读取理解表示。可训练 head $R_\theta$ 在 PaliGemma projected visual feature space 中输出 velocity field。

训练采用 conditional flow matching：从高斯噪声向 $f_{goal}^{m+1}$ 积分；推理时积分得到 $f_{pred}^{m+1}$。论文没有给出展开公式，但机制与 π0.5 action flow 相同，只是 target 从 action token 换成 terminal visual feature。

与像素视频 world model 相比，它主动舍弃颜色/纹理重建，把容量聚焦在 policy 已能消费的 feature space。实验中 Stable Diffusion 像素终点再编码的版本成功率低 10.4 个百分点，支持这一选择。

### 3.3 Guided VLA

guided π0.5 保留 understanding expert 和 action expert，新增从 Gemma 初始化的 goal expert。控制步 $t$：

- understanding expert 编码当前 $obs_t$ 与 logical action，得到 $h_t^u$；
- goal expert 编码固定 $f_{pred}^{m+1}$，得到 $h_m^g$；
- action expert 输入 $q_t$ 与 noisy action tokens，通过非对称 cross-attention 读取 $h_t^u,h_m^g$，输出连续 action chunk。

信息只从 understanding/goal 流向 action，不允许反向 attention，从架构上防止低层 action noise 污染高层目标。VLA 用与 π0.5 相同的 conditional flow-matching action objective 端到端训练。

### 3.4 子任务完成检测

独立 completion predictor 从 trained guided VLA 的 understanding expert 初始化，加入 `[CLS]` 与二分类 head：

$$p_t^{done}=\sigma(Wh_t^{cls}).$$

每个控制步检测；判定完成后，逻辑世界模型推进到下一 transition，视觉世界模型生成新的 visual subgoal。这一模块看似辅助，实际上是层级系统的时钟；误触发会跳过任务，漏触发会反复执行。论文给出 BCE 与 10k steps，但未单独报告 detection precision/recall，是复现和安全评估中的缺口。

---

## 4. 数据、Baseline 与训练成本

### 4.1 LIBERO-Logic

作者重放 LIBERO episode，用预定义 predicate classifier 推断每帧 logical state，并由 state change 识别 logical action，随后人工检查与修正。最终抽取 transition tuple

$$\langle X_m,a_{m+1},X_{m+1},obs_m,q_m\rangle.$$

RoboCerebra 则采用其 task decomposition，并统一 logical action/predicate vocabulary。该数据构建使逻辑与视觉严格对齐，但需要 domain-specific predicate classifier 与人工修订，规模化到开放世界并不免费。

### 4.2 Benchmark 与 Baseline

- **LIBERO-10：** 1–3 步，较短。
- **LIBERO-LoHo：** 五个 5–7 步任务，报告 success rate 与完成子目标比例 Q-Score。
- **RoboCerebra：** 最长 20 步。
- **真机：** UR5e + Intel RealSense，八步清桌，40 条人工 demonstration，各方法 20 trials。

端到端 baseline 包括 π0、π0.5、OpenVLA、OpenVLA-OFT、X-VLA、GR00T-N1.5。层级 baseline 包括自然语言分解的 LLM-guided π0.5 与只有 PDDL action/state、没有 visual subgoal 的 Logic-guided π0.5。

### 4.3 训练成本与可复现性

- guided VLA：π0.5 初始化，60k steps，global batch 256，学习率 $5\times10^{-5}$；
- completion predictor：10k steps；
- logical model：Qwen3-4B，全参数微调 10k steps，batch 16，学习率 $10^{-5}$；
- visual world model：约 300M prediction expert，projection head 1024→4096→2048，100k steps，global batch 256，学习率 $5\times10^{-5}$。

论文未披露 GPU 型号/数量、wall-clock、显存、LIBERO-Logic 下载和代码。完整复现需要 π0.5/PaliGemma/SigLIP、Qwen3-4B planning checkpoint、LIBERO/RoboCerebra、predicate 标注管线和真机控制栈，属于高成本系统工程。

---

## 5. 实验结果与消融

### 5.1 LIBERO-LoHo：world model 是否影响 planning

| 方法 | Q-Score↑ | Success Rate↑ |
|---|---:|---:|
| π0.5 | 55.3 | 6.4% |
| LLM-guided π0.5 | 66.8 | 26.8% |
| Logic-guided π0.5 | 73.2 | 48.4% |
| H-WM + Stable Diffusion | 81.1 | 54.4% |
| **H-WM** | **84.9** | **64.8%** |

相对 π0.5，完整 H-WM 成功率提高 58.4 个百分点、Q-Score 提高 29.6。更关键的是 controlled ablation：logic-only 已比 π0.5 高 42.0 个百分点；加入 latent visual guidance 再提高 16.4 个百分点。由此可区分两种作用：逻辑层主要解决漏步/乱序与 goal tracking，视觉层主要解决每个 subtask 的可执行落地。

逐任务看，H-WM 在 Task 4 达 42% 成功率，而 logic-only 和 LLM-guided 都只有 4%；Task 3 为 46% vs 16%/24%。复杂依赖越长，双层 guidance 越重要。

### 5.2 短任务与 RoboCerebra

![](assets/H-WM%20-%20逻辑视觉分层机器人世界模型/figures/02-benchmark-results.png)

> **图 2：LIBERO-10 与 RoboCerebra。** 短任务上 π0.5 已 92.4%，H-WM 为 93.5%，收益很小；RoboCerebra 上 H-WM 达 67.1 Q-Score/11.0% success，logic-only 为 63.8/6.0，π0.5 为 54.4/1.88。来源：论文 Figure 2，第 6 页，官方 TeX 原图；已用 Read 验证。

这一结果说明层级 world model 不是普遍替换端到端 policy：任务越短，额外系统开销越难换来收益；长时依赖才是适用区间。

### 5.3 latent visual guidance vs 像素生成

H-WM-Stable-Diffusion 用同一 subtask-aligned data 微调 SD1.5 生成 terminal image，再经同一 frozen visual encoder 编码给 VLA。其平均成功率 54.4%，比 latent H-WM 低 10.4 个百分点。作者解释像素生成引入与动作无关的纹理细节与 reconstruction noise，而 latent target 更贴近 VLA 消费空间。

这个结果支持“世界模型不是画得越真越有用”：对 planning 而言，可行动的中间表示比高保真像素更重要。它也是五篇主轴中的关键证据：世界模型价值应由 action success 衡量，而不是仅由 generation metric 衡量。

### 5.4 真机

![](assets/H-WM%20-%20逻辑视觉分层机器人世界模型/figures/03-real-robot.png)

> **图 3：UR5e 八步任务。** π0.5 在第三步后成功率迅速归零，logic-only 能维持到后期，H-WM 在第 8 步仍约 30%。视觉 subgoal 对后半程物体配置的 grounding 是完整任务成功的关键。来源：论文 Figure 3，第 7 页，官方 TeX 原图；已用 Read 验证。

每种方法仅 20 trials，且逻辑 state 由 Qwen3-VL-2B one-shot 估计；因此曲线提供了强烈趋势，但统计置信区间与感知误差敏感性仍需补充。

---

## 6. 表示路线与两条主轴

### 6.1 H-WM 不是 BEV，也不是视频隐式三维先验

它使用两种任务导向 latent：predicate set 是高层显式 symbolic latent；PaliGemma projected visual feature 是低层隐式视觉 latent。后者没有米制三维坐标，也没有依赖大视频生成模型的隐式 3D prior。它的创新维度不是空间表示本身，而是**抽象层级与时间尺度**。

### 6.2 与两条主轴的可组合性

- 可用 BEV/SparseWorld 一类显式 3D latent 替换当前 PaliGemma visual subgoal，使碰撞与几何约束更可验证。
- 可用 VEGA-3D 一类生成先验增强 current/goal visual token，减少遮挡或视角变化带来的 grounding error。
- 但无论 latent 从哪里来，都必须通过 action expert 和 task success 验证；H-WM 提供了“表示怎样进入 planning”的系统模板。

---

## 7. 局限与 Future Work

### 7.1 作者明确局限

- 额外 logical model、visual model、goal expert 与 completion detector 增加训练成本和系统复杂度。
- 假设任务可以被有意义的符号 predicate/action 表示。
- Future work：用 self-supervised learning 降低 logical supervision；加入更多传感器以增强 spatial reasoning。

### 7.2 进一步批判

1. **符号 state 估计是单点故障。** 论文主要关注 transition planning，却没有系统评估 predicate noise、state omission 与 completion false positive。
2. **视觉 world model 只预测成功终态。** 它不能展示 subtask 中途碰撞或不可逆失败；低层 policy 可能朝“正确终点”走一条危险路径。
3. **搜索评价器同源偏差。** generator 与 evaluator 由同一逻辑模型实例化，可能共享错误先验；显式 PDDL precondition checker 或执行反馈能提供独立约束。
4. **长时 benchmark 规模有限。** LIBERO-LoHo 只有五个定制任务；20-trial 真机结果不足以覆盖开放世界扰动。
5. **无代码与数据。** 当前无法验证标注规则、beam search prompt、flow integration step 和 completion threshold。

### 7.3 最关键的 future work

真正下一步应把 H-WM 从 open-loop subtask plan 推向反馈式 belief update：每次观测更新 predicate belief 与 visual goal uncertainty；在中途检测失败或环境变化时重规划；让 visual world model 同时预测成功终态、失败模式和可达性。这样 world model 不仅给 policy 一个目标，还能决定何时否定原计划。

---

## 8. 复现建议

1. 先复现 logic-only π0.5，确认 symbolic decomposition 与 completion detector 的收益，再加入视觉世界模型。
2. 逐项测 state estimation accuracy、action transition accuracy、goal completion precision/recall，而非只看最终 success。
3. visual latent 先用 GT terminal feature 上界，再换 predicted feature，分离 policy 与 world model 误差。
4. 做 pixel image、PaliGemma latent、显式 3D/BEV latent 三种 visual subgoal 对比，保持 policy 和数据一致。
5. 对 beam width、candidate $K$、最大深度与 evaluator/goal weight 做成本—成功率曲线。
6. 真机增加环境扰动、失败恢复和中途重规划，否则“长时”仍主要是顺序执行 benchmark。

---

## 9. 个人启发

H-WM 给出一个比“预测未来视频”更接近机器人 planning 本质的定义：世界模型的输出必须成为可搜索的状态、可执行的子目标或可评分的候选未来。逻辑层控制 horizon，视觉层控制 grounding，VLA 负责局部闭环；这三者的接口比任何单个生成指标都更重要。它也说明层级 abstraction 不是退回传统模块化，而是把不同预测误差限制在合适的时间尺度上。

---

## 10. 材料来源

详见 [sources.md](assets/H-WM%20-%20逻辑视觉分层机器人世界模型/sources.md)。核心依据为 arXiv v3 官方 PDF与 TeX 源包原图；截至 2026-10-16 未发现官方代码/数据发布。所有插图均已逐张 Read 验证。