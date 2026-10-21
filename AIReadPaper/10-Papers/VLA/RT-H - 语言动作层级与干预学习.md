---
type: paper-note
status: complete
domain: VLA
paper: "RT-H: Action Hierarchies Using Language"
year: 2024
arxiv: 2403.01823
doi: null
source: https://arxiv.org/abs/2403.01823
project: https://rt-hierarchy.github.io/
code: null
tags:
  - VLA
  - RT-H
  - Action-Hierarchy
  - Language-Motions
  - Interactive-Imitation-Learning
created: 2026-10-21
updated: 2026-10-21
---

# RT-H：语言动作层级与干预学习

## 基本信息

- **作者**：Suneel Belkhale、Tianli Ding、Ted Xiao、Pierre Sermanet、Quan Vuong、Jonathan Tompson、Yevgen Chebotar、Debidatta Dwibedi、Dorsa Sadigh
- **机构**：Google DeepMind、Stanford University
- **论文原题**：*RT-H: Action Hierarchies Using Language*
- **arXiv**：2403.01823
- **项目页**：<https://rt-hierarchy.github.io/>
- **骨干**：PaLI-X 55B / RT-2
- **本地 PDF**：`assets/RT-H - 语言动作层级与干预学习/sources/paper.pdf`

## 一句话结论

RT-H 在 RT-2 的“任务语言 → 动作 token”之间加入可读的 language motion，如“move arm forward”“close gripper”，用同一 PaLI-X 55B 分两次查询先预测中间动作语义、再结合图像和任务解码精确动作；9 维本体感知自动组合出 2500+ 语言动作，无人工离线标注，在 100k 多任务演示上比 RT-2 平均高 15%、动作 MSE 低约 20%，30 条纠正/任务即可把八任务平均成功率从 40% 提到 63%，而同量遥操作 IWR 从 25% 降到 13%，证明可解释中间层既是跨任务共享瓶颈，也是比低层动作更样本高效的人机干预接口。

---

## 1. 研究背景与问题定义

### 1.1 扁平 VLA 的共享瓶颈

RT-2 把高层任务语言直接映射到低层动作。对“pick coke can”和“pick apple”，语言结构相近，数据容易共享；对“pick coke can”和“pour cup”，高层语义差异很大，但前半段都可能是伸手、对准、闭夹爪和抬起。扁平模型需要从图像和任务文本直接学习高维动作映射，难以显式利用这些跨任务公共子轨迹。

传统分层策略可学习离散 skill/latent，但通常不可解释、对聚类和超参数敏感；固定运动原语又过于僵硬，同一句“向左”在不同物体、速度、腕姿和障碍上下文中应对应不同数值动作。RT-H 的核心判断是：**自然语言可以处在任务与动作之间，既压缩低层动作，又保留组合结构和人类可编辑性。**

### 1.2 两个目标

1. 通过 language motion hierarchy 改善多任务数据共享和策略泛化；
2. 将人类纠正从 VR 遥操作转成键盘/语音提供“向左、抬起、闭夹爪”等中间指令，并只学习修正高层 motion query。

### 1.3 核心贡献

- 定义高层策略 $\pi_h:O\times G\to Z$ 和低层策略 $\pi_l:O\times G\times Z\to A$；
- 用同一 PaLI-X 55B、两个 prompt 实现两层策略；
- 从 9 个动作维自动生成 2500+ 组合语言动作；
- 异步预测下一步 motion，使双查询延迟接近 RT-2；
- 比较 RT-H、Joint、Cluster、OneHot、RT-2 和多种纠正训练；
- 评估多任务性能、上下文性、在线纠正、场景/对象/任务泛化。

---

## 2. 任务定义与输入输出

### 2.1 分层概率模型

数据集：

$$D=\{(\tau_i,g_i)\}_{i=1}^{N},\qquad \tau_i=\{(o_t,z_t,a_t)\}_{t=1}^{L_i}$$

$g$ 是高层任务语言，$o$ 是图像观察，$z$ 是中间 language motion，$a$ 是低层机器人动作。策略分解为：

$$\pi(a,z\mid o,g)=\pi_h(z\mid o,g)\pi_l(a\mid o,g,z)$$

先问“当前应该做什么 motion”，再问“在当前场景和任务下，这个 motion 对应什么精确动作”。$z$ 比任务细、比动作粗，例如 `move arm forward and close gripper`。

### 2.2 动作空间

沿用 RT-2：末端三维位置增量、三维 axis-angle 旋转、夹爪、终止标记，并在相关数据中包含底座运动；每个连续维度量化到 256 bin，编码为整数 token。Language motion 是普通自然语言 token，因此可直接利用 PaLI-X 预训练语义结构。

### 2.3 两个查询

**Motion query**：

> `Q: What skill should the robot do to [task]? A:`

编码器看图像和任务，解码器输出 $z$。

**Action query**：

> `Q: What action should the robot do to [task], with current skill: [motion]? A:`

编码器同时看图像、任务和 motion，解码动作 token。把 motion 放进 encoder 而非和动作串在 decoder 中，是 RT-H 与 Joint 变体的重要差异。

---

## 3. 核心方法

### 3.1 总体架构与纠正闭环

![](assets/RT-H%20-%20语言动作层级与干预学习/figures/02-method.png)

> **图 2：RT-H 的两层查询与干预训练。** 左侧先预测 language motion，再条件生成动作；右上人类可替换 motion，右下只用纠正标签更新 motion query，然后部署回完整层级。来源：论文 Figure 2，p.4，arXiv:2403.01823。

同一 VLM 同时训练两种 query，避免维护两个模型，并让两层都继承 Web 预训练。PaLI-X 的 22B ViT 在机器人共同训练时冻结，编码解码骨干更新。机器人样本中 motion query 和 action query 等概率采样，各占整体训练 25%；原 VLM 预训练混合占 50%。

### 3.2 自动提取 language motion

作者从 9 个动作维构造标签：末端位置 3 维、旋转 3 维、底座 2 维、夹爪 1 维。每维映射到自然语言方向，如 up/down、forward/backward、left/right、clockwise/counterclockwise、base turn、open/close gripper。低于 small-action threshold 的维度过滤，剩余按幅值排序后组合。

例如位置前向最大、夹爪开始闭合，标签为 `move arm forward and close gripper`。组合性使少量原子动作产生 2500+ motion，无需人类逐帧标注。标签直接由动作生成，故对 action query 有很高预测力；代价是语义偏底层坐标，缺少“抓住把手”“绕开杯子”等对象相对概念。

抽象层存在权衡：越细，action query 获得信息越多，但 motion query 越难预测；越粗，容易预测，却对精确动作帮助有限。论文选择的方向/旋转/夹爪组合在性能、纠正和标注成本之间取得可用平衡。

### 3.3 训练目标

两个查询都是自回归交叉熵：

$$\mathcal L_h=-\log p_\theta(z\mid o,g),\qquad \mathcal L_l=-\log p_\theta(a\mid o,g,z)$$

共同训练可写成：

$$\mathcal L=\lambda_{web}\mathcal L_{VLM}+\lambda_h\mathcal L_h+\lambda_l\mathcal L_l$$

基础训练采样比例为 50% Web、25% motion、25% action。学习率 $10^{-3}$，线性 warmup 后平方根衰减，batch 1024。两个 query 共用参数，因此即便纠正阶段“只采样 motion query”，参数更新仍可能影响 action query；持续混入原动作数据是防止退化的关键。

### 3.4 异步双查询

若每个控制步先推 motion 再推 action，55B 模型延迟近乎翻倍。RT-H 训练 motion query 预测**下一步** motion；推理时用上一步预测的 motion 生成当前动作，同时批处理预测下一步 motion：

$$a_t\sim\pi_l(o_t,g,\hat z_{t-1}),\qquad \hat z_t\sim\pi_h(o_t,g)$$

两次查询并行后延迟接近 RT-2。替代方案是每 $H$ 步更新一次 motion，但 motion 切换可能恰好发生在抓放瞬间，固定频率更易错，故实验采用异步方案。这一方案也引入一拍条件滞后，是高速任务中的潜在风险。

### 3.5 人类在线纠正

![](assets/RT-H%20-%20语言动作层级与干预学习/figures/01-overview.png)

> **图 1：语言动作层级与人类干预。** 模型在“关开心果罐”中显示预测 motion；人类可键入 `move arm left` 覆盖一步或一段时间，退出后模型恢复自主预测。下方展示多样任务共享同类 motion。来源：论文 Figure 1，p.1。

操作者按键进入纠正模式，通过键盘、热键或麦克风提供新 motion。系统可显示当前预测以增强透明度，并周期询问“保持、更新或退出”。纠正后的 motion 仍交给同一个 action query，所以具体方向、速度、腕姿由场景和任务上下文决定，而不是执行固定速度原语。

### 3.6 从纠正数据学习

只保留成功 episode 中人类提供的 `(o,g,z^{corr})`。因为 action query 已经能把纠正 motion 变成成功动作，训练时主要更新 motion query，使模型以后自己预测该纠正。Diverse+Kitchen 下采样比例为：

- Web 预训练 query：50%；
- 原示范 motion query：23%；
- 原示范 action query：23%；
- 纠正 motion query：4%。

纠正数据约比原示范少 300 倍，4% 权重相当于每条纠正样本被过采样约 50 倍。RT-2-IWR 使用同样配方，但纠正样本是 VR 遥操作动作。

---

## 4. 数据集、Baseline 与实验协议

### 4.1 100k Diverse+Kitchen

- **Kitchen**：RT-1/RT-2 数据的 70k 演示、6 类语义任务、542 个指令；
- **Diverse**：新增 30k 演示、24+ 类复杂任务、165 个指令；
- **合计**：100k，场景中物体位置和背景随机化。

Diverse 包括开/关开心果罐、拿勺、从分配器抽纸、放碗到麦片机下、移开碗、把燕麦包放碗里、倒/舀零食、挤蜂蜜等。新增任务数量多但每任务样本少，正适合测试层级共享。

评测挑 8 个困难任务：翻正碗、开罐、关罐、把碗移离出料口、把碗放到出料口下、把燕麦包放碗、拿勺、抽纸。每方法每任务 10 次，共 80 次；checkpoint 按验证集 action MSE 选择。

### 4.2 Baseline 和消融

- **RT-2**：无层级的平坦 55B VLA；
- **RT-H-Joint**：一次 query 自回归输出 motion 再输出 action，motion 在 decoder 侧；
- **RT-H-Cluster**：对归一化动作做 k-means，256 类替代语言 motion；
- **RT-H-OneHot**：保留自动 motion 划分，但把每条短语替换成独立整数，破坏语言结构；
- **RT-H + Human Intervention**：测试时在线人工纠正，视作上界；
- **RT-H-Intervene**：只从语言 motion 纠正学习；
- **RT-H-InterveneAction**：同时学习纠正时产生的 motion 与动作；
- **RT-2-IWR**：从同量 VR 遥操作纠正学习。

### 4.3 纠正数据

八个任务各收集 30 条成功纠正 episode。RT-H 由 motion 干预收集，RT-2 由 VR teleop 干预收集，均过滤失败。比较重点是训练前后变化，而非只看绝对值，因为 RT-H 初始性能高于 RT-2。

---

## 5. 实验结果与消融

### 5.1 多任务主结果

![](assets/RT-H%20-%20语言动作层级与干预学习/figures/03-multitask-results.png)

> **图 3：Diverse+Kitchen 八个困难任务结果。** RT-H 平均 40%，RT-2 为 25%；RT-H 在 6/8 任务非零，Joint 在 8/8 非零，OneHot 明显退化，说明层级和语言结构都重要。来源：论文 Figure 3，p.6。

RT-H 平均比 RT-2 高 15 个百分点，并在 6/8 任务上更强。RT-2 只有 4/8 非零，RT-H 为 6/8，RT-H-Joint 为 8/8。图中平均约为：RT-2 25、OneHot 23、Cluster 35、Joint 41、RT-H 40。具体任务上 RT-H-Cluster 在开/关罐等精确任务可能更强，作者解释为数据驱动 cluster 提供更细动作上下文；但在广泛任务上，cluster 不如语言容易预测。

OneHot 使用与 RT-H 相同标签划分却显著下降，说明收益不只是“多一个离散 latent”。`move arm forward` 与 `move arm forward and left` 在语言 embedding 中共享结构，独立整数则等距，丢失预训练语言先验和组合泛化。

### 5.2 离线动作 MSE

| 训练/评测 | RT-2 | RT-H-Joint | RT-H | RT-H 使用真值 motion |
|---|---:|---:|---:|---:|
| Kitchen → Kitchen | 30.2 | 28.22 | **24.9** | **17.9** |
| D+K → Diverse | 27.7 | 25.44 | **23.6** | **17.8** |

RT-H 比 RT-2 低约 17–20%，Joint 低 5–10%。真值 motion 又比端到端 RT-H 低约 28–40%，说明 action query 已能很好利用 motion，当前瓶颈更多在 motion query。在线人工干预接近满成功率也支持这一诊断。

### 5.3 上下文性与灵活性

同一句 `move arm forward` 会根据任务稍向目标偏移、倾斜腕部或避开麦片机；`move arm left` 在燕麦任务中指向碗，在关罐中使盖子扣住罐口。Appendix 按 motion 统计动作均值/方差，主轴最大，但非主轴也有显著方差，量化了上下文解码。

模型还可响应任务分布外 motion：抽纸可“右下”或“上后”两种方式；关罐时让机械臂暂时远离也能执行。说明 motion 不是固定 macro，而是条件策略的中间提示。Beam search 还能给出多个合理 motion，显示同一状态存在多模态行为。

### 5.4 纠正学习主结果

![](assets/RT-H%20-%20语言动作层级与干预学习/figures/04-intervention-results.png)

> **图 4：八任务纠正结果。** RT-H-Intervene 平均从 40% 升到 63%；在线人工干预约 94%；RT-2-IWR 从 25% 降到 13%。来源：论文 Figure 6，p.9。

每任务仅 30 条纠正：

- RT-H：40% → **63%**；
- RT-2-IWR：25% → **13%**，出现退化；
- RT-H-InterveneAction：比 RT-H 平均高约 9%，但低于只学 motion；
- RT-H + Human Intervention：约 94%，说明 action query 对正确 motion 的可控性很强。

开/关开心果罐等精确任务，RT-H-Intervene 提升 60–70 个点。RT-2-IWR 退化可能因为数据太少且遥操作动作分布与原示范不同。Motion 纠正仍由基础 action query 产生动作，只轻微改变中间条件，因此与原策略分布更一致。

同时训练纠正动作会出现 policy degeneration：干预时模型生成的动作可能亚优，学习后在抓到盖子等状态输出近零动作。结果反而支持“只修正高层 motion，不把在线模型动作当专家”的设计。

### 5.5 失败模式

- motion query 可能持续预测 `close gripper`，导致振荡或卡住；
- 错误 motion 会误导 action query，平坦模型未必有同类错误；
- 纠正质量受 action query 上限限制，若“向左”仍过冲，操作者会比遥操作更挫败；
- `place oatmeal packet` 中纠正数据偏向 `close gripper`，RT-H-Intervene 反而退步；
- 精密任务绝对成功率即使纠正后仍有提升空间。

好处是失败可读：系统显示 motion 后，开发者能定位“motion 预测错”还是“motion 到动作解码错”，比不可解释动作 token 更易调试。

### 5.6 泛化

新对象测试 50 个场景：

| 模型 | Pick | Move | 平均 |
|---|---:|---:|---:|
| RT-2 | 60 | 50 | 55 |
| **RT-H** | **70** | **60** | **65** |

新建筑含不同背景、光照和地面，RT-H/Joint 在 upright 与 drawer 等困难任务上明显优于 RT-2。对全新任务，零样本仍困难，但 RT-H 往往能自主完成共享的“先抓取”阶段，只需少量后续 motion 纠正即可成功。

### 5.7 消融归纳

1. RT-H 与 Joint 都优于平坦 RT-2，说明层级本身有效；
2. RT-H 比 Joint 的 MSE 更低，说明专门 prompt 与把 motion 放入 encoder 有益；
3. Cluster 在精确任务强、总体略弱，体现细粒度动作信息与可预测性的权衡；
4. OneHot 明显退化，证明自然语言内部结构和预训练先验有贡献；
5. 真值 motion 大幅降低 MSE，定位瓶颈在高层 motion 预测；
6. 只学 motion 纠正优于同时学在线动作，说明压缩空间更抗小数据分布漂移。

---

## 6. 与 Google RT 系列的关系

### 6.1 从 RT-2 的单层生成到语言动作层级

[[RT-2 - Web知识迁移视觉语言动作模型]] 统一文字和动作 token，RT-H 进一步让自然语言进入动作层内部。它没有更换 55B 骨干，而是改变监督结构，证明相同观察、任务和动作可以因中间表示而更易学习。

### 6.2 对 RT-Trajectory 的互补

[[RT-Trajectory - 轨迹草图条件任务泛化]] 用连续二维几何表达整段运动，适合用户画路径；RT-H 用离散、局部、可组合短语表达当前动作，适合在线纠正。前者解决“语言说不清路径”，后者解决“高层语言离动作太远”。一个自然组合是高层任务 → 轨迹草图/对象相对路径 → language motion → 动作。

### 6.3 对 RT-X 与 AutoRT 的意义

论文明确提出 language motion 可桥接 [[RT-X - 跨本体机器人数据与模型]] 的 OXE，因为“向前、抬起、闭夹爪”比原始关节动作更接近本体共享语义。对 [[AutoRT - 基础模型编排机器人数据收集]]，语言纠正可降低一人多机监督成本：人无需接管完整遥操作，只在关键错误点给 motion。

---

## 7. 局限与批判性评价

### 7.1 论文明确局限

1. 绝对成功率仍不够高，需扩大离线数据和纠正管线；
2. 中间层最佳抽象未知，对象相对语言可能优于坐标方向；
3. 目前只有一层动作推理，长时任务可能需要 task→subtask→motion→action 多层；
4. 未来可把 motion 作为 RL 的压缩探索空间，但论文未验证。

### 7.2 模型与部署局限

PaLI-X 55B、每步双 query 的成本极高；异步方案使用上一时刻 motion，快速接触转换可能产生一拍延迟。自动标签依赖机器人坐标轴，跨本体时“forward/left”未必语义一致。2500+ 组合中长尾 motion 样本少，模型可能受频率偏置。

### 7.3 评测边界

八任务每方法 80 次，纠正每任务 30 条，足以显示大趋势但仍是单一机器人平台。IWR 只做一轮、小数据，不能得出“语言纠正普遍优于所有遥操作纠正”的广泛结论。最可信的是：在该 55B 模型和数据分布下，修正压缩 motion query 比直接拟合少量动作稳定。

---

## 8. 复现成本与实践建议

### 8.1 成本

完整复现很高：PaLI-X 55B、100k 真机演示、云推理、八任务真机纠正。标签生成本身便宜，不需人工离线标注。可用开源 3B–7B VLA 在 OXE 子集上复现核心层级消融。

### 8.2 实现要点

1. 先离线检查自动 motion 标签的阈值、坐标方向和组合顺序；
2. 训练时同时保留 Web、原 motion、原 action，纠正数据只高权重采样 motion；
3. motion 与 action 查询使用不同 prompt，并测试 encoder 条件与 decoder 串联差异；
4. 在线界面显示预测 motion、支持热键和一键退出；
5. 对异步查询测 motion 切换延迟，抓/放关键帧可强制同步；
6. 分开记录 motion accuracy、action MSE、任务成功和人工纠正次数；
7. 不把干预时模型动作自动当专家，先验证其质量再训练 action query。

---

## 9. 个人启发与后续问题

RT-H 的深层价值是把“可解释性”变成训练和数据效率工具，而不只是展示文本。中间 motion 让不同任务共享子轨迹，让人能纠正，让错误可定位，还可能成为跨本体与强化学习探索空间。它说明 VLA 的中间语言不必只是 CoT 解释，也可以是与动作强绑定、可监督、可执行的结构变量。

后续值得研究：motion 应是坐标方向、对象相对动作还是接触事件？多层层级如何自动决定在哪一层请求人类？能否从人类视频或 OXE 不同本体自动对齐 language motion？异步上一拍条件会否在高速任务中失效？若用小模型，语言层级的收益是否仍存在，还是依赖 55B 的强语言先验？

---

## 10. 材料来源

详见 `assets/RT-H - 语言动作层级与干预学习/sources.md`。本文形式化、自动标签、50/25/25 训练混合、100k 数据、MSE、真机成功率、纠正结果、泛化与局限均逐页核对本地官方 PDF；四张图由 arXiv 官方源码提取并逐张检查。
