---
type: paper-note
status: complete
domain: World-Models
paper: "Programmable World Model"
year: 2026
arxiv: "2609.10540"
doi: "10.48550/arXiv.2609.10540"
source: https://arxiv.org/abs/2609.10540
project: https://alaya-lab.github.io/pwm/
code: https://github.com/AlayaLab/pwm
code_commit: "0286dc7a193a99f84c29f8d25fc67976311d0833"
tags:
  - World-Models
  - Interactive-Video
  - Programmable-World
  - Explicit-State
  - Generative-Rendering
  - 3D-OBB
created: 2026-10-29
updated: 2026-10-29
---

# PWM：显式状态驱动的可编程世界模型

## 基本信息

- **完整标题**：Programmable World Model
- **作者**：Zheng-Hui Huang、Guixu Lin、Jiacheng Lin、Yi-Chuan Huang、Ruihan Yu、Muyao Niu、Siqi Yang、Yu-Lun Liu、Yung-Yu Chuang、Kaipeng Zhang、Zhixiang Wang
- **版本**：arXiv 2609.10540；v1 于 2026-09-09 提交，v2 于 2026-10-13 修订
- **研究定位**：把交互式视频世界从“根据历史画面猜下一帧”改造成“程序维护事实、视频模型渲染事实”的混合系统
- **核心表示**：state-augmented 3D oriented bounding boxes（状态增强 3D OBB）
- **生成骨干**：预训练 LingBot-World-v1，相机控制路径冻结，新增 Structured Spatial ControlNet
- **评测**：CombatStateBench，50 个战斗片段、400 个计数评测帧、50 个死亡事件
- **核心结果**：Count Accuracy 94.00%，State Accuracy 98.00%
- **项目页**：<https://alaya-lab.github.io/pwm/>
- **官方仓库**：<https://github.com/AlayaLab/pwm>
- **代码状态**：截至 2026-10-29，仓库 commit `0286dc7a` 仅含 README 与 teaser；推理代码和权重仍在 roadmap 中，尚不能代码级复现
- **个人评级**：★★★★☆（4.5/5）——架构切分非常重要，状态指标也足够醒目；扣分主要因为 benchmark 较窄、消融缺失、状态指标偏宽松、训练与系统细节未披露、代码尚未发布

## 一句话结论

PWM 最重要的贡献不是“给视频模型再加一种框条件”，而是明确划分了**事实层与外观层**：可执行程序和轻量引擎负责维护世界中什么是真的，生成式视频模型只负责这些事实看起来是什么样子；状态增强 3D OBB 则是两者之间兼顾可编程性、跨视角投影和生成自由度的接口。

---

## 1. 研究背景与问题定义

### 1.1 “看起来合理”不等于“世界事实正确”

现有交互式视频世界模型通常把最近的视频帧、相机动作和玩家输入作为条件，继续生成下一段画面。它们越来越擅长生成“视觉上合理的后续”，却没有一个独立于当前视野的权威状态源。角色离开画面后是否仍然存活、门是否已解锁、背包里还剩多少物品、某个任务是否完成，这些事实往往只能隐式寄存在视频上下文或记忆特征中。

这会带来一个根本冲突：视频模型的优化目标是生成高概率、视觉连贯的观察，但游戏引擎需要的是精确、离散、可查询、可修改且长期不漂移的事实。前者允许“看上去差不多”，后者要求“一次状态转移之后永远成立，直到规则再次改变它”。

### 1.2 论文识别的三个缺口

论文把问题拆成三个互相关联的缺口：

1. **实体级控制不足**：相机轨迹、键鼠动作或高层文本只能粗略影响画面，难以直接指定某个实体的身份、位置、生命值、阵营与交互结果。
2. **缺少显式持久状态**：视频上下文不适合保存离屏实体和不可见属性，也难以支持多人共享世界所需的统一事实。
3. **规则不可编程**：提示词可以要求“这次把门打开”，却不能自然地建立“持有钥匙且靠近门时才打开”的可重复执行规则。

![](assets/PWM%20-%20显式状态驱动的可编程世界模型/figures/01-overview.png)

> **图 1：PWM 总览。** coding agent 把自然语言规格写成世界程序，轻量引擎读写状态并执行规则；实体状态以 3D OBB 为几何锚点，再被编译成像素对齐控制，最后由视频模型渲染。图中 `hp: 20→0` 与 `active→down` 表明“死亡”首先是状态事实，画面只是该事实的视觉实现。  
> 来源：论文 Figure 1，PDF p.1，<https://arxiv.org/abs/2609.10540>

### 1.3 核心贡献

论文的三项正式贡献是：

- 提出 Programmable World Model，把可执行状态演化与生成式渲染解耦；
- 构建自动数据管线，从普通游戏视频恢复相机、深度、实例、语义、3D OBB 与运动方向监督；
- 提出 CombatStateBench，评估生成视频是否忠实实现引擎维护的角色数量与死亡状态。

更高层地看，这篇论文把“world model”从单一神经网络重新定义成一个系统：**程序化状态机 + 确定性编译器 + 概率生成渲染器**。

---

## 2. 任务定义与表示选择

### 2.1 交互主循环

给定初始观察 $I_0$，时刻 $t$ 的玩家动作是 $a_t$，系统维护 canonical world state $s_t$。完整循环写成：

$$s_t\xrightarrow[a_t]{F}s_{t+1}\xrightarrow[C_{t+1}]{P}M_{t+1}^{\mathrm{ctrl}}\xrightarrow{G}I_{t+1}.$$

其中：

- $F$ 是轻量引擎执行的规则化状态转移；
- $C_{t+1}$ 是目标相机；
- $P$ 是确定性 state compiler；
- $M_{t+1}^{\mathrm{ctrl}}$ 是相机对齐的结构化控制图；
- $G$ 是生成式视频渲染器。

关键在于 $F$ 与 $G$ 的职责完全不同：$F$ 决定事实，$G$ 不能推翻事实，只能把它实现成观察。

### 2.2 Canonical world state

论文把世界状态写为：

$$s_t=(\mathcal E_t,\mathcal A_t,\mathcal Q_t;\mathcal R_t).$$

- $\mathcal E_t$：持久实体及其 3D pose/OBB；
- $\mathcal A_t$：语义与功能属性，如类别、生命值、阵营、库存、任务进度；
- $\mathcal Q_t$：实体关系，如敌对、所有权、可交互关系；
- $\mathcal R_t$：可执行世界规则与事件触发器。

状态转移是：

$$s_{t+1}=F(s_t,a_t).$$

因为这个转移由显式规则执行，而不是由视频模型或 learned logic model 猜测，所以离屏角色的生命值、所有权和任务进度可以被准确保留。

### 2.3 为什么选择状态增强 3D OBB

论文把候选接口放在一条“控制力—成本”谱线上：

| 表示 | 控制能力 | 主要优点 | 主要代价 |
|---|---|---|---|
| 文本 | 弱 | 最容易编写，可表达类别和高层状态 | 无法确定世界坐标中的位置、尺度和朝向 |
| 2D box / mask | 中等 | 直接约束图像平面 | 依赖当前视角，不能自然地跨视角重投影 |
| **状态增强 3D OBB** | 中强 | 世界坐标、实例级、紧凑、可投影、易被程序修改 | 只能给出粗粒度占用和运动，细节交给生成模型 |
| coarse 3D proxy / G-buffer | 强 | 更精细的几何与表面约束 | 标注和推理构造成本高，开放域难扩展 |
| 完整 3D 场景/骨骼/动态几何 | 最强 | 可精确控制姿态、碰撞和动画 | 等价于重建完整图形/动画/物理栈 |

![](assets/PWM%20-%20显式状态驱动的可编程世界模型/figures/02-representation-tradeoffs.png)

> **图 2：状态表示的权衡。** 从 prompt 到完整 3D 场景，控制变强，但数据获取难度、推理时维护成本和训练—推理不匹配也同步增大。3D OBB 被选作中间点：显式约束“实体在哪里、占多大、朝哪边”，把精细形状、纹理、关节和次级运动留给生成先验。  
> 来源：论文 Figure 2，PDF p.3，<https://arxiv.org/abs/2609.10540>

### 2.4 训练—推理不对称

论文指出，结构表示在训练与推理时的产生方向相反：

$$\text{Training: realized dynamics}\longrightarrow\text{structural representation}.$$

训练视频中动作已经发生，系统只需从结果中抽取结构。而推理时，程序给出的只是“站立→倒地”这种高层状态变化：

$$\text{Inference: state transition}\longrightarrow\text{structural representation}\longrightarrow\text{generated dynamics}.$$

如果接口是骨骼级运动或动态网格，推理阶段就必须先求出完整倒地动画，再交给渲染器，生成模型的价值被大幅削弱；如果接口只有文本，实体位置、身份和跨视角一致性又无法可靠约束。OBB 的作用正是把显式系统的责任限制在“必须严格正确的结构”，让生成模型补全“可以概率生成的细节”。

---

## 3. 核心方法

### 3.1 总体架构

PWM 由三段组成：

1. **World Programming**：coding agent 从自然语言生成状态 schema、实体属性、关系、动作、规则和目标；
2. **Control Compilation**：确定性编译器把 3D OBB 状态投影成 identity、semantic、direction 三类像素控制；
3. **Generative Rendering**：相机可控视频模型结合结构控制、时间历史和空间记忆生成下一段视频。

![](assets/PWM%20-%20显式状态驱动的可编程世界模型/figures/03-architecture.png)

> **图 3：PWM 的三阶段架构。** 左侧执行规则，中间把世界坐标状态压缩并栅格化，右侧由复制的部分 DiT block 构成 Structured Spatial ControlNet；主干同时读取当前 chunk、时间历史和几何对齐空间记忆。  
> 来源：论文 Figure 3，PDF p.6，<https://arxiv.org/abs/2609.10540>

### 3.2 Agent-Orchestrated Box World

#### 3.2.1 初始化

系统输入一张参考图像 $I_0$ 和自然语言世界描述 $q$。off-the-shelf 3D detector 从首帧恢复可见实体、初始 3D OBB 和持久 ID；coding agent 再补充非视觉内容：

- 初始 health、faction、ownership、inventory；
- 支持的玩家动作；
- 合法攻击/交互关系；
- 伤害、死亡、目标切换与胜负规则；
- 事件触发器和任务目标。

这使“创建世界”近似于生成一个可执行的轻量游戏脚本，而不是只生成一段描述世界的视频提示词。

#### 3.2.2 引擎执行

每个交互步中，引擎依次：

1. 验证动作是否合法；
2. 查询实体属性、关系和规则；
3. 应用动作效果；
4. 解析连锁事件与触发器；
5. 更新实体、属性和关系；
6. 输出新的 canonical state。

例如 `on_hit: hp -= damage` 与 `on_hp_zero: status = down` 是两条可重复执行的规则。角色即使离开视野，其 $hp$ 仍存于 canonical state 中；重新进入视野时，compiler 会根据当前状态再次生成正确控制。

#### 3.2.3 这种设计与 learned state transition 的区别

StatePlay 会联合预测视觉与游戏状态，MASS 虽然有 authoritative typed state，但仍由 learned Logic Engine 推进。PWM 则把状态转移交给显式程序，因此避免了“状态本身预测错，再把错误状态当真”的累积风险。代价是：系统只会执行已经写进规则的机制，无法自动覆盖开放世界中所有未建模物理规律。

### 3.3 Deterministic State Compiler

#### 3.3.1 从世界状态到像素控制

编译器计算：

$$M_{t+1}^{\mathrm{ctrl}}=P(s_{t+1},C_{t+1}).$$

它不修改 canonical state，只选择与视觉相关的实体字段，把每个 3D OBB 投影到目标相机，并通过深度感知栅格化/z-buffer 处理重叠和可见性。这样，同一个世界状态可以被任意相机查询；相机变化只影响投影，不改变世界事实。

#### 3.3.2 Identity map：哪个实例

系统维护 $K$ 个可学习身份槽：

$$\mathcal E^{\mathrm{id}}=\{q_j^{\mathrm{id}}\}_{j=1}^{K}.$$

训练时，一个样本内的不同实体被分配到互异 slot，同一序列跨帧保持不变，但跨样本随机重排。这可防止某个 slot 永久绑定“士兵”“汽车”或某种外观。推理时，每个持久实体始终使用同一 slot，因此遮挡、离屏和重新出现后仍有实例级锚点。

Identity map 不是实体外观本身，而是“这个像素区域属于同一个对象”的对应关系。外观仍需由初始图像和视觉记忆恢复。

#### 3.3.3 Semantic map：是什么类别

对于实体类别 $y_i$，预训练文本编码器给出：

$$q_i^{\mathrm{sem}}=E_{\mathrm{text}}(y_i).$$

该 embedding 被栅格化到 OBB 的可见投影区域。相同类别共享语义表示，不同实例由 identity map 区分。语义分支尤其适合首帧中不存在、没有可靠外观参考的新实体，因为它至少告诉生成器“这里应该生成什么类别”。

#### 3.3.4 Direction map：实体自身向哪运动

系统把世界坐标速度旋转到目标相机坐标系，再量化为七类：`forward / backward / left / right / static / up / down`。对应 embedding bank 为：

$$\mathcal E^{\mathrm{dir}}=\{q_\ell^{\mathrm{dir}}\}_{\ell=1}^{7}.$$

最关键的细节是：方向来自 **world-space velocity**，不是 image-space optical flow。一个静止角色会因为相机平移而在画面中移动，但仍应标成 `static`；否则模型会混淆相机运动与实体自主运动。

#### 3.3.5 控制拼接

三类控制最终按通道拼接：

$$M_t^{\mathrm{ctrl}}=\operatorname{Concat}(M_t^{\mathrm{id}},M_t^{\mathrm{sem}},M_t^{\mathrm{dir}}).$$

它们分别回答：

- `which instance`：哪个持续存在的实体；
- `what semantic category`：它是什么；
- `in which direction`：它自身如何运动。

论文未披露 $K$、embedding 维度、控制图分辨率和通道数，因此无法给出真实实现中的具体张量形状。

### 3.4 Generative Renderer

#### 3.4.1 双控制通路

生成器以 LingBot-World-v1 为预训练 backbone：

- 原有 camera-conditioning pathway 控制全局视点和相机轨迹；
- 新增 Structured Spatial ControlNet 控制实体位置、身份、语义和运动方向。

这种拆分避免让方向图同时承担相机控制：相机怎么走与实体怎么动是两个独立变量。

#### 3.4.2 ControlNet 注入

结构控制先编码到视频 latent 的空间分辨率，再与 noisy video latent 一起输入 ControlNet。第 $\ell$ 层控制特征 $r^{(\ell)}$ 以残差方式加到主干：

$$h_{\mathrm{main}}^{(\ell)}\leftarrow h_{\mathrm{main}}^{(\ell)}+r^{(\ell)}.$$

训练时冻结预训练主干参数 $\theta$ 和原相机条件路径，只优化空间控制分支 $\phi$。固定窗口渲染可写为：

$$I_{1:T}=G_{\theta,\phi}(I_0,\mathcal C,M_{1:T}^{\mathrm{ctrl}}).$$

其中 $\mathcal C=\{C_t\}_{t=1}^{T}$ 是相机轨迹。显式控制决定实体布局与粗运动，生成先验补全精确形状、纹理、材质、关节运动、局部形变、光照和次级动态。

### 3.5 Chunk-Autoregressive 长时生成

#### 3.5.1 分块生成

系统按 chunk 自回归：chunk 内仍可双向去噪，chunk 之间严格因果。第一块由 $I_0$ 做 image-to-video 初始化；从第二块开始：

$$I^{(n)}=G_{\theta,\phi,\psi}(\mathcal C^{(n)},M^{\mathrm{ctrl},(n)},\mathcal H_{\mathrm{temp}}^{(n)},\mathcal H_{\mathrm{spa}}^{(n)}),\quad n\ge2.$$

$\mathcal H_{\mathrm{temp}}$ 是时间历史，$\mathcal H_{\mathrm{spa}}$ 是几何对齐空间记忆。已完成的过去 chunk 才能写入记忆，避免未来信息泄漏。

#### 3.5.2 多尺度时间历史

时间历史分为 recent、mid-range、long-range：近期 latent 保留更高时空分辨率，越远历史压缩得越粗；初始帧 $I_0$ 的 latent 始终作为 persistent anchor。训练时还会对历史做噪声扰动、特征破坏和部分丢弃，使模型适应推理中逐步污染的自生成上下文。

#### 3.5.3 几何对齐空间记忆

空间记忆沿用 AlayaWorld 的思路：

1. 用估计深度和相机参数把已完成 RGB 帧提升到世界空间；
2. 查询与下一目标视角相关的历史观察；
3. 把历史观察重投影到目标相机；
4. 编码为空间 latent tokens，供下一 chunk 使用。

因此 PWM 有两套“记忆”：

- canonical state 保存**离散且必须精确的世界事实**；
- temporal/spatial memory 保存**用于恢复外观和几何细节的视觉证据**。

这两套记忆不能互相替代：显式状态记得“这个角色已死亡”，视觉记忆才记得“这个角色具体长什么样”。

### 3.6 自动训练数据管线

每个训练视频经过以下链路：

| 阶段 | 工具/模型 | 输出与作用 |
|---|---|---|
| 相机与几何 | ViPE | 相机内参、相机位姿、metric depth；建立共享世界坐标 |
| 类别发现 | Qwen3-VL | 全局 caption 与离散可数类别词表 |
| 实例分割与跟踪 | SAM3 | mask、2D box、类别、persistent track ID |
| 3D OBB 恢复 | WildDet3D | 每帧 3D box center、尺寸、朝向 |
| 世界轨迹 | 相机位姿变换 | 把逐帧 OBB 统一到世界坐标，消除相机运动 |
| 方向标签 | 轨迹差分 + 量化 | 七类相机坐标系实体运动方向 |
| 控制生成 | 投影 + z-buffer | identity / semantic / direction maps |

每帧实体表示写成：

$$b_i^t=(\mathbf p_i^t,\mathbf d_i^t,\mathbf R_i^t,\ell_i,k_i),$$

其中 $\mathbf p$ 是中心，$\mathbf d$ 是尺寸，$\mathbf R$ 是朝向，$\ell_i$ 是语义类别，$k_i$ 是持久 track ID。

这条管线的优势是不用从游戏引擎直接导出完整状态；风险是 ViPE、SAM3、WildDet3D 的误差会层层传播。论文只说明过滤小 mask 和做一致性验证，没有报告各阶段标注精度。

### 3.7 训练与推理的完整分工

| 环节 | 训练 | 推理 |
|---|---|---|
| 世界状态 | 从视频抽取 OBB/身份/类别/方向监督 | agent 生成程序，引擎执行规则 |
| 结构控制 | 从已发生的视频动态反推 | 从目标状态和相机确定性编译 |
| 视频模型 | 冻结 backbone，训练空间控制分支与跨 chunk 条件模块 | 依据控制和记忆生成下一 chunk |
| 视觉历史 | 真实历史并随机破坏 | 模型自己的历史生成结果 |
| 状态真值 | 来自恢复/构造的标注 | canonical state 是权威事实 |

论文没有披露 optimizer、learning rate、batch size、训练步数、数据时长、视频分辨率、chunk 长度、identity slot 数、参数量、硬件或推理速度；因此目前只能复现概念流程，不能复现论文结果。

---

## 4. 数据集与实验设置

### 4.1 训练数据

训练视频来自三款无 HUD 游戏：

- Cyberpunk 2077：第一人称；
- Forza Horizon 6：第三人称、多样相机；
- Grand Theft Auto V：第三人称、多样相机。

论文没有给出每款游戏的视频数量、总时长、clip 数、分辨率、帧率和授权细节。这意味着“自动管线可扩展”是方法设计上的优势，但真正的数据规模仍不可判断。

### 4.2 CombatStateBench

CombatStateBench 包含 50 个短战斗片段。每个场景从首帧重建 3D layout，由 agent 按游戏规则推进短战斗序列，覆盖：

- 不同相机运动；
- 不同实体运动；
- 初始离屏实体；
- 死亡事件；
- 死亡后实体继续保留在世界中。

每条序列包含同步 3D boxes、entity states、camera parameters、projected controls 和 instance masks。自动验证器检查首帧重投影、metric depth、一致 box geometry、地面接触、时间连续性、规定的相机/实体运动和状态转移持久性。

### 4.3 Baseline 与提示协议

对比方法是 LingBot-World-V2 和 YUME。两者没有外部实例级状态接口，论文只能沿用其原生控制通路，并在死亡事件发生时切换 prompt：

$$\texttt{\{environment description\}. \{action description\}.}$$

这是一种现实但不完全同质的比较：PWM 获得 3D box、身份、语义、方向和外部状态引擎，baseline 只有文本结果提示。实验可以证明“完整显式状态系统优于纯 prompt 接口”，却无法分离 OBB、identity、direction、规则引擎和视觉记忆各自的贡献。

### 4.4 评价指标

#### 4.4.1 Count Accuracy

每个视频随机采 8 帧，共 $50\times8=400$ 帧。Qwen3.6-27B 只看 RGB，统计可见且存活的角色数：

$$\mathrm{CountAcc}=\frac{1}{N}\sum_{i=1}^{N}\mathbf1[\hat n_i=n_i^{\mathrm{eng}}].$$

这里要求严格相等，能直接测出角色凭空消失或新增。

#### 4.4.2 State Accuracy

共 50 个死亡事件，每个事件后均匀采 3 帧。若至少一帧出现死亡角色，就判定该事件被视觉实现：

$$\mathrm{StateAcc}=\frac{\text{visually realized death events}}{\text{total death events}}.$$

这个指标明显偏宽松：不检查是不是正确角色、不检查倒地位置、不要求死亡持续，也不惩罚其他角色被误杀。

#### 4.4.3 视频质量

使用 VBench 的 Imaging Quality、Subject Consistency、Background Consistency 和 Temporal Stability。评测器只看 RGB，不获得 box、身份、预期数量和死亡位置等特权标注。

---

## 5. 实验结果

### 5.1 主要定量结果

| Method | Imaging Quality | Subject Cons. | Background Cons. | Temporal Stability | Count Acc. | State Acc. |
|---|---:|---:|---:|---:|---:|---:|
| LingBot-World-V2 | 67.46 | 81.87 | 91.89 | 96.85 | 40.75 | 8.00 |
| YUME | 64.10 | 92.35 | 93.63 | 98.76 | 32.00 | 58.00 |
| **PWM** | **67.62** | **94.74** | **96.98** | **99.00** | **94.00** | **98.00** |

最有价值的不是 Imaging Quality 仅提升 0.16，而是两个状态指标出现数量级差异：

- Count Accuracy 相对 LingBot-World-V2 提升 53.25 个百分点，相对 YUME 提升 62.00；
- State Accuracy 相对 LingBot-World-V2 提升 90.00 个百分点，相对 YUME 提升 40.00。

这说明显式状态接口确实改变了模型“能否忠实执行世界事实”，而不是只美化画面。与此同时，视频质量四项也没有因加入粗 OBB 控制而下降，说明生成先验仍能把粗结构实现为稳定视觉。

![](assets/PWM%20-%20显式状态驱动的可编程世界模型/figures/04-benchmark-results.png)

> **表 1 + 图 4：CombatStateBench 结果与死亡交互。** 静态和动态相机场景中，baseline 会出现应死未死、数量漂移或场景演化不一致；PWM 的输入行给出结构控制，输出更稳定地执行指定死亡事件并保留其余实体。  
> 来源：论文 Table 1 与 Figure 4，PDF p.12，<https://arxiv.org/abs/2609.10540>

### 5.2 定性泛化与长时结果

Figure 5 展示五类能力：

1. 未见过的 minotaur 场景；
2. 大角度相机转动后看到初始镜头背后的角色；
3. 从战斗角色泛化到赛车；
4. 同时渲染人和车辆；
5. 897 帧自回归序列中，多名 NPC 逐步进入场景。

![](assets/PWM%20-%20显式状态驱动的可编程世界模型/figures/05-generalization-results.png)

> **图 5：开放域、离屏持久性与长时生成。** 输入 OBB 只提供粗空间和状态约束，模型仍能生成森林、机场、赛车和人车混合场景；其中大角度相机旋转与 897 帧序列最能体现“显式状态不随视野消失”的价值。  
> 来源：论文 Figure 5，PDF p.13，<https://arxiv.org/abs/2609.10540>

需要谨慎的是，897 帧只属于定性展示。论文没有给出随时间增长的身份准确率、状态漂移率、重访一致性曲线或生成延迟，因此不能从一条样例推出普遍的分钟级稳定性。

### 5.3 消融实验缺失

论文没有正式 ablation section，也没有回答以下问题：

- 去掉 identity map 后，离屏重现和身份一致性下降多少？
- 去掉 semantic map 后，新类别和无参考实体下降多少？
- 去掉 direction map 后，相机运动与实体运动是否混淆？
- 去掉 temporal history 或 spatial memory 后，897 帧结果如何变化？
- 3D OBB 是否显著优于 2D box、mask 或 coarse proxy？
- 显式规则引擎与 learned state transition 的受控差异是多少？

因此，主表证明的是“完整系统有效”，而不是逐模块因果贡献。

### 5.4 失败案例与证据边界

论文没有集中展示失败案例。根据方法和评测协议，可以预期以下薄弱点：

- OBB 无法严格指定肢体姿态、接触点、关节轨迹和局部形变；
- 自动标注的相机、深度、分割、跟踪和 3D box 错误会污染训练；
- State Accuracy 可能把错误角色倒地也算成功；
- baseline 接口弱于 PWM，比较存在条件不对称；
- benchmark 仅 50 个 combat clips，任务规则和状态类型都较单一；
- 未报告速度，尚不清楚“playable”是否达到实时交互。

---

## 6. 与相关工作的关系

### 6.1 相对于隐式视频世界模型

LingBot-World、YUME、WorldPlay、AlayaWorld 等系统主要依靠视频上下文、KV cache、时间压缩或空间记忆维持视觉一致。它们的记忆回答的是“过去看起来怎样”，而 PWM 的 canonical state 回答的是“现在事实是什么”。

PWM 并没有否定视觉记忆，反而把它保留为 renderer 的一部分。真正的变化是：视觉记忆不再承担生命值、身份、关系和任务进度等精确事实的唯一责任。

### 6.2 相对于 StatePlay 与 MASS

| 方法 | 是否显式状态 | 状态如何更新 | 主要风险 |
|---|---|---|---|
| StatePlay | 是 | 模型联合预测 | 状态预测误差累积 |
| MASS | authoritative typed state | learned Logic Engine | 转移仍可能预测错误 |
| **PWM** | canonical executable state | 程序规则执行 | 需要事先写出规则，开放物理覆盖有限 |

PWM 的设计更接近传统 ECS/game engine 与神经渲染器的结合，而不是让一个神经网络同时学世界逻辑和视觉外观。

### 6.3 相对于生成式渲染

DiffusionRenderer、AlayaRenderer 和 Coarse-to-Real 已证明结构条件可以驱动生成式渲染。PWM 的新增点是把结构条件放入一个**持续演化、可执行、可查询的世界状态循环**中，并显式加入身份、语义和运动方向。

### 6.4 与“把 Agent 记忆写入数据库”的同构性

从系统设计角度看，PWM 与 Agent 系统把长期记忆外置到数据库是同一个原则：不要让概率模型独占必须精确、可审计的状态。模型擅长模糊生成，数据库/程序擅长精确持久；两者之间需要一个结构化编译接口。

---

## 7. 局限与批判性评价

### 7.1 真正的优点

1. **责任边界干净**：世界事实与视觉外观分离，是比“扩大上下文窗口”更根本的架构选择。
2. **OBB 抽象合理**：既保留世界坐标和实例身份，又避免重建完整 mesh/骨骼/物理。
3. **状态可查询与可验证**：适合任务进度、库存、多人共享状态和可重复规则。
4. **训练数据可自动构造**：不强依赖游戏引擎导出的内部状态，理论上更易扩展到普通视频。
5. **状态指标提升巨大**：Count 94%、State 98%，与 baseline 的差距远大于一般画质改进。

### 7.2 方法层面的局限

1. **OBB 不是物理状态**：它不能表达接触、质量、力、关节、软体形变和精确碰撞几何。
2. **程序只保证符号逻辑**：`hp=0` 可以绝对正确，但倒地是否自然、尸体是否穿模仍由生成器决定。
3. **规则覆盖问题**：用户必须显式写出或由 agent 生成规则；未建模事件没有可靠动力学来源。
4. **感知—状态绑定仍可能错**：若首帧 3D detector 或后续身份对应错误，程序会精确维护一个错误实体。
5. **视觉实现非确定**：renderer 仍可能不遵循 box、语义或方向控制，canonical state 正确不等于画面必然正确。

### 7.3 实验层面的局限

1. CombatStateBench 仅 50 个片段，聚焦角色数量与死亡，无法代表库存、任务、所有权、门锁、多人同步等复杂状态；
2. State Accuracy 不做实例匹配，也不检查死亡持续性；
3. baseline 无同等结构接口，不能做公平的组件级归因；
4. 没有消融、效率、参数量、延迟、训练成本和多随机种子；
5. 长时结果只有 897 帧定性样例；
6. 训练数据规模与授权边界不透明；
7. 官方推理代码和权重尚未发布。

### 7.4 我的判断

这篇论文的价值更像“架构方向被说清楚了”，而不是“完整可复现系统已经成熟”。它击中了交互式视频世界模型的核心软肋：**视频历史不是数据库，像素生成也不是逻辑执行**。但当前验证仍接近一个 combat demo；要证明它是通用 world engine，需要扩展到复合任务、多人共享、物理交互和长时规则链，并给出更严格的实例级评测。

---

## 8. 复现与实践建议

### 8.1 当前可复现程度

| 部分 | 可复现性 | 原因 |
|---|---|---|
| 世界状态 schema 与规则执行 | 中 | 论文定义清楚，可用 Python/ECS 自行实现 |
| 3D OBB 投影与栅格化 | 中高 | 标准相机几何和 z-buffer 可实现 |
| 自动标注管线 | 中低 | 依赖 ViPE、Qwen3-VL、SAM3、WildDet3D，多阶段误差与配置未披露 |
| Structured Spatial ControlNet | 低 | 网络层数、通道、embedding、训练配置均未披露 |
| 长时 renderer | 低 | 基座、记忆实现和跨 chunk 训练细节不完整 |
| 论文结果复现 | 很低 | 无代码、权重、数据规模、超参和硬件信息 |

### 8.2 最小可行复现路线

如果目标是验证思想而不是复刻画质，可以按四步实现：

1. 用简单 JSON/ECS 保存实体 ID、3D OBB、health、status 和关系；
2. 用规则函数执行 `move / attack / die / spawn`，确保 canonical state 可单元测试；
3. 用已知相机把 OBB 投影成实例色图、类别色图与方向色图；
4. 选择支持视频 ControlNet/adapter 的开源视频模型，先只训练短片段结构控制，再逐步加入历史帧和空间记忆。

最先应该验证的不是画质，而是四项可测指标：

- 实例级 box IoU/center error；
- 正确实体的状态实现率；
- 离屏后重新出现的身份一致率；
- 随 rollout 长度变化的状态与身份漂移曲线。

### 8.3 计算与工程成本

论文没有报告算力。根据系统组成，主要成本会集中在视频 DiT 的 ControlNet 训练与长时 chunk 推理；状态引擎和 OBB compiler 本身很轻。自动数据管线则会产生显著离线成本：每段视频需依次运行相机/深度、VLM、视频分割跟踪和 3D 检测。真正决定部署可用性的仍是视频 renderer 的延迟，而不是规则引擎。

---

## 9. 个人启发与后续问题

### 9.1 对世界模型定义的启发

PWM 迫使我们区分三件常被混在一起的东西：

- **状态**：世界中什么是真的；
- **记忆**：过去观察提供了哪些证据；
- **渲染**：从某个视角看起来是什么样子。

很多“长记忆世界模型”只改进了第二项，却把它描述成第一项。PWM 的贡献是让状态成为独立的一等公民。

### 9.2 对 3D/资产世界模型的启发

状态增强 OBB 可以进一步扩展为层级资产状态：

- asset-level：类别、实例、变换、生命周期；
- part-level：部件层级、关节与可交互区域；
- topology-level：mesh 连通性、破损、开闭状态；
- material-level：材质槽、湿润/燃烧/损坏状态；
- relation-level：接触、支撑、包含、所有权。

这与 Asset-JEPA/3D JEPA 的潜在结合点是：显式程序维护离散、可验证的资产事实，JEPA 或生成模型预测难以手写的连续动力学和外观变化。

### 9.3 最值得继续追问的问题

1. coding agent 生成错误规则时，如何做静态检查、测试和回滚？
2. 如何把 learned physics 与 executable rules 组合，而不让两套动力学互相冲突？
3. OBB 何时不足，应该升级到 part graph、articulated proxy 或 3DGS？
4. 如何让 renderer 对 canonical state 提供可验证的执行回执，而不是只生成 RGB？
5. 多人世界中，状态同步、冲突解决和相机私有观察如何设计？
6. 能否用 PROWBench 的 Logic-Render Alignment、Interaction Success 和实例几何指标替代当前较宽松的 State Accuracy？

---

## 10. 开源状态与论文—代码映射

### 10.1 仓库检查

截至 2026-10-29，官方仓库最新提交为：

- **commit**：`0286dc7a193a99f84c29f8d25fc67976311d0833`
- **日期**：2026-10-13
- **文件**：`README.md`、`assets/teaser.png`
- **roadmap**：项目页、technical report 已完成；inference code、pretrained weights 尚未提供

因此，虽然论文和 frontmatter 有官方 GitHub 链接，但它目前是发布占位仓库，不能把它视为“已开源完整代码”。

### 10.2 只能基于论文建立的概念模块映射

| 论文模块 | 预期软件职责 | 当前能否核验代码 |
|---|---|---|
| Coding agent | 自然语言→状态 schema、实体、规则与目标 | 否 |
| State executor | 动作验证、规则执行、事件解析、canonical state 更新 | 否 |
| State compiler | OBB 投影、z-buffer、identity/semantic/direction 栅格化 | 否 |
| Structured Spatial ControlNet | 控制编码、复制 DiT blocks、残差注入主干 | 否 |
| Temporal memory | recent/mid/long 历史 latent 管理与退化训练 | 否 |
| Spatial memory | RGB-D 提升、世界点云、目标视角重投影 | 否 |
| Data engine | ViPE→Qwen3-VL→SAM3→WildDet3D→控制图 | 否 |

由于没有配置文件、权重和源码，无法按默认配置给出具体张量形状、参数量、FLOPs、显存与运行时间。任何更具体的“代码级实现”都将是猜测，故本笔记只保留论文层面的符号张量和职责映射。

### 10.3 PROWBench 与本文 benchmark 的关系

官方仓库在 2026-10-11 宣布后续发布 PROWBench。它评估程序指定的实体轨迹、相机运动与交互是否被视频模型忠实渲染，指标覆盖 Entity、Logic/State、Camera、Quality、Temporal 和 Text，并引入 Interaction Success、Logic-Render Alignment 等更细指标。

需要区分：

- **CombatStateBench**：本文 Table 1 使用的 50-clip 战斗状态评测；
- **PROWBench**：论文发布后的更通用 renderer benchmark，不应把其 leaderboard 数字回填为本文实验结果。

---

## 11. 材料来源

- 论文：<https://arxiv.org/abs/2609.10540>
- 项目页：<https://alaya-lab.github.io/pwm/>
- 官方仓库：<https://github.com/AlayaLab/pwm>
- 后续 benchmark：<https://alaya-lab.github.io/PROWBench/>
- 本地材料索引：[sources.md](assets/PWM%20-%20显式状态驱动的可编程世界模型/sources.md)
