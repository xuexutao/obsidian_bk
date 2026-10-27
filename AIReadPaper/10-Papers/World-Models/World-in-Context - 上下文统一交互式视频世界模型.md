---
type: paper-note
status: complete
domain: World-Models
paper: "World-in-Context: Interactive Video World Modeling with In-Context Conditioning"
year: 2026
arxiv: "2610.33169"
doi: null
source: "https://arxiv.org/abs/2610.33169"
project: "https://haroldchen19.github.io/WiC-site/"
code: "https://github.com/aigc3d/World-in-Context"
code_commit: "a985fd003458f07de5c8a537d750efbadd989aa9"
code_status: "截至 2026-10-27，官方 GitHub 仅为占位展示仓库；推理代码、训练代码、配置、WiC 权重、数据和 examples 均未发布，README 下半部分是 planned release。"
created: 2026-10-27
updated: 2026-10-27
tags:
  - paper
  - World-Models
  - Video-Generation
  - Interactive-World-Model
  - In-Context-Conditioning
  - IT-RoPE
  - Camera-Control
  - Action-Control
rating: 9/10
---
# World-in-Context：用上下文统一视角、动作与相机控制的交互式视频世界模型——论文详细精读
## 基本信息
| 项目 | 内容 |
|---|---|
| 论文 | World-in-Context: Interactive Video World Modeling with In-Context Conditioning |
| 作者 | Harold Haodong Chen、Rui Peng、Rongjin Guo、Lingteng Qiu、Kejie Qiu、Chao Xu、Zilong Dong |
| 机构 | Alibaba Token Hub, Alibaba Group |
| 版本 | arXiv:2610.33169v1，2026-10-23 |
| 基座 | Wan2.2-TI2V-5B |
| 核心任务 | 在视频生成中联合控制 FREE、FPP、TPP 视角模式、角色动作与 6-DoF 相机轨迹 |
| 核心方法 | In-Context Conditioning（ICC）与 Interaction-Transport RoPE（IT-RoPE） |
| 主评测 | WBench：289 cases、1,058 interaction turns |
| 主结果 | Navigation 81.5，FPP↔TPP switching 93.8% |
| 训练数据 | 100K 单视角模式导航 clips；Stage 2 为混合构造的 8K curriculum，并对 500 条切换序列过采样 |
| 代码状态 | **截至 2026-10-27 未开源 / coming soon**；仓库只有 README、LICENSE、`.gitignore` 与 6 个展示资产，共 9 个 tracked files |
| 仓库固定版本 | `a985fd003458f07de5c8a537d750efbadd989aa9` |
| 复现状态 | 本文只做论文、图片、README、API 快照与仓库树静态审计；没有运行模型、推理或训练 |
**一句话结论：** World-in-Context 最重要的贡献不是简单增加三种控制信号，而是把“当前处于哪种观察制度、角色执行什么动作、相机怎样移动”都改写为共享自注意力中的显式上下文，并用有序、非交换的 transport 修正跨时间注意力。
**核心判断：** 论文对“控制关系显式化”给出了很强的受控证据，尤其是 6.25%→37.5%→93.8% 的视角切换结果；但它仍不是已证明具有显式 3D 状态、真实物理状态或实时在线闭环能力的通用世界模拟器。
**阅读边界：** 论文披露的实现细节不足以恢复真实张量尺寸、训练预算与推理成本，README 中的环境、分辨率和采样参数又明确属于计划接口，二者必须分开。
## 1. 一页结论与证据分级
### 1.1 论文真正回答的问题
交互式视频世界模型不再只接收文本并一次性生成视频，而要随时间接收控制命令并预测视觉观察。
这类系统至少面对两个相互耦合的问题。
第一，世界从什么视角被观看：自由相机 FREE、第一人称 FPP、第三人称 TPP 的动力学语义并不相同。
第二，用户如何改变世界演化：角色动作与相机视口运动既相关，又不能总被视为同一自由度。
许多先前方法把视角写进自然语言，把动作和相机压成单一控制，或放进彼此隔离的分支。
这样做在单一控制下可能够用，但控制越丰富，分支间隐式协调的负担越大。
World-in-Context 的问题意识是：能否把多维交互重新表述为统一上下文建模，而不是不断增加专用侧路？
它的答案由两部分构成。
ICC 解决“把哪些控制送进模型、在哪里交互”。
IT-RoPE 解决“跨越多个时刻时，控制序列如何进入时间注意力”。
前者是表示与接口设计，后者是有序动力学关系编码。
### 1.2 四级证据结论
#### 1.2.1 强结论
**强结论 1：显式 view token 显著改善视角切换。**
文本表达视角的基线只有 6.25% switching success。
仅用单一视角 clips 训练、但显式改变 view-token schedule，成功率已升到 37.5%。
再用包含 500 条切换序列的 Stage 2 混合 curriculum 后，成功率达到 93.8%。
这组受控结果直接支持“显式视角接口更容易被组合与微调”。
**强结论 2：在论文统一协议下，ICC 的收益随条件丰富度扩大。**
只有 camera 时，ICC 与 Additive/AdaLN 基本同档。
加入 action 后，ICC 的 Navigation 与 Interaction 优势开始出现。
再加入 view 后，ICC 达到 81.2/86.8，并扩大相对差距。
这支持共享注意力空间更适合协调复合控制。
**强结论 3：有序 transport 比只看端点更有效。**
Full IT-RoPE 为 81.2/86.8。
Endpoint-Only 为 80.5/86.6。
两条轨迹即使端点相同，只要动作执行顺序不同，非交换乘积就能给出不同 transport。
**强结论 4：WiC-5B 在 WBench 主表中达到 81.5 Navigation 与 93.8 switching。**
Navigation 与 22B EchoWM 的 81.6 接近。
这是一条清楚的结果事实，但不是 5B 架构普遍优于 22B 的证明。
#### 1.2.2 中等结论
**中等结论 1：action 更偏局部命令响应，camera 更偏全局稳定。**
action-only 相比 camera-only 的 Translation/Rotation Accuracy 高 6.5/3.9 分。
camera-only 相比 action-only 的 Geometric/Photometric Consistency 高 2.1/1.9 分。
joint training 在旋转和几何一致性上最好，同时保留较强平移准确率。
证据来自排除 TPP 的 WBench Navigation 子集，外推到更广任务仍需谨慎。
**中等结论 2：ICC 在给定数据预算下表现出更高监督利用率。**
ICC 用 50K+250 得到 75.0% switching。
Additive 用 100K+500 只有 71.9%。
但这只是同一论文实现中的架构对照，不代表总训练 FLOPs 或总数据成本更低。
**中等结论 3：默认双向模型可以迁移到 causal 形式。**
WiC-Causal 达到 81.2/90.6，接近双向版 81.5/93.8。
这说明 ICC 接口与历史 latent token 在形式上兼容。
论文没有给出可持续在线多长时间、延迟多少、显存如何增长的定量答案。
#### 1.2.3 弱结论
**弱结论 1：模型形成了可泛化的长期空间记忆。**
Figure 15 展示回环与视点重现时能复现地标和布局。
但这里只有定性帧，没有 loop-closure error、重定位准确率或长时漂移曲线。
因此它最多说明“出现了令人鼓舞的视觉一致性案例”。
**弱结论 2：5B 匹配 22B 主要来自 ICC。**
主表中的不同模型使用不同数据、训练尺度、控制协议与实现。
不能从 81.5 对 81.6 直接把参数效率归因于 ICC。
#### 1.2.4 尚未证明
尚未证明模型内部学习了显式 3D scene graph、地图、物体状态或物理参数。
尚未证明 WBench 高分可转化为真实机器人或游戏引擎中的闭环可控性。
尚未证明默认双向 clip 模型具备实时在线生成能力。
尚未证明 causal rollout 可在任意长时域保持稳定。
尚未证明训练语料与 WBench 没有近重复、风格污染或教师模型泄漏。
尚未证明 action 与 camera 是统计独立监督，因为 action 主要由 ViPE 相机轨迹反演得到。
### 1.3 论文全景图
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/01-teaser.png)
**图解：** Figure 1 把 FREE、FPP、TPP、TPP→FPP 与 FPP→TPP 放在同一模型下展示；键盘覆盖层强调控制是时序输入，而不是只靠起始文本决定整段视频。
**关键阅读：** 图中同一模型覆盖无角色的自由漫游、随身体摆动的第一人称、角色与镜头分离的第三人称，并支持视频内部切换观察制度。
**图号与页码：** 论文 Figure 1，PDF Page 1。
**官方来源：** 官方 GitHub 高分辨率 `assets/teaser.png`，https://github.com/aigc3d/World-in-Context/blob/a985fd003458f07de5c8a537d750efbadd989aa9/assets/teaser.png 。
## 2. 任务定义：从条件视频生成到交互式世界演化
### 2.1 条件分布
令视频为$x\in\mathbb{R}^{T\times H\times W\times3}$。
$T$是帧数，$H$与$W$是像素高度和宽度，3 是 RGB 通道。
令$y$为初始上下文，可以是文本提示、起始帧或两者组合。
令$u=\{u_t\}_{t=1}^{T}$为与时间对齐的用户交互序列。
模型目标是参数化：
$$p_\theta(x\mid y,u)$$
$\theta$代表生成模型参数。
$x$不是预先存在的未来，而是模型对未来视觉观察的条件分布。
$y$负责指定初始世界与静态语义。
$u$负责指定随时间变化的干预。
这一分工是全文后续设计的逻辑起点。
### 2.2 潜空间 flow-matching 表述
现代视频扩散或 flow-matching 主干通常不在 RGB 空间直接建模。
令$z_\tau$为扩散时间$\tau\in[0,1]$上的视频 latent。
传统做法写成：
$$\hat v_\tau=F_\theta(z_\tau,f_{\mathrm{cond}}(u);y,\tau)$$
$F_\theta$是速度场预测器。
$\hat v_\tau$是预测的 flow velocity。
$f_{\mathrm{cond}}$是控制专用编码或注入通路。
它可能是 camera AdaLN、action additive injection、text cross-attention，或几何位置注意力。
问题不是这些模块“不能工作”，而是各模态被放进不同参数空间。
跨条件依赖只能通过更深层网络间接形成。
ICC 将其改写为：
$$\hat v_\tau=F_\theta([c;z_\tau];y,\tau)$$
$c$是所有交互条件的统一 token 序列。
分号表示沿序列维拼接。
控制 token 与视觉 latent 共同进入 shared self-attention。
因此“action 的含义是否随 view 改变”可以在同一注意力图中直接计算。
### 2.3 FREE、FPP、TPP 的动力学差异
#### 2.3.1 FREE：相机本身是主要运动主体
FREE 对应无人称、自由相机或航拍式漫游。
相机可以平滑地平移、转向和环绕，而不必模拟人体步态。
同样一条几何轨迹在 FREE 中通常应显得稳定、平滑、少周期摆动。
它的控制语义更接近 viewport navigation。
#### 2.3.2 FPP：观察者与角色身体耦合
FPP 表示第一人称具身视角。
向前移动不仅造成相机位移，还会出现由步态带来的横向 sway 与纵向 bobbing。
如果只给相机终点，模型可能错过“这是角色行走而非浮空摄像机”的运动风格。
FPP 中 action 与 camera 高度相关，但二者仍提供不同监督：命令意图与实际轨迹。
#### 2.3.3 TPP：角色与镜头是两个自由度
TPP 中角色可以向前走，而相机可以跟随、侧移、旋转或保持相对位置。
把角色动作完全等价成相机轨迹会丢掉角色—镜头解耦关系。
因此 TPP 是联合 action-camera 建模最直接的动机。
它也解释了为什么论文默认 TPP 推理使用 joint action and camera。
### 2.4 为什么文本只描述静态世界状态
论文明确限制$y$只描述 world state。
caption 总结天气、空间几何、布局实体和环境构成。
caption 不写详细时间叙事，不负责动作进程，也不描述相机如何走。
所有 temporal progression、dynamic steering 与 viewport navigation 都交给控制 token。
这样做避免文本与 action/camera 同时描述运动却互相冲突。
也避免模型把“向前移动”记成某个 caption 的语言模式，而不是可组合控制。
代价是静态/动态边界由标注流程人为规定。
例如树叶摇动、行人运动与角色动作是否属于 world state，论文未给细粒度规则。
## 3. In-Context Conditioning：统一交互接口
### 3.1 总体结构
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/02-icc-it-rope.png)
**图解：** Figure 2 左侧概括旧范式：视角藏在文本中，view/camera/action 分别走 cross-attention、modulation 或 injection；右侧把三类信号都转成 Interactive In-Context Tokens，与视频 latent 共享 self-attention。
**IT-RoPE 部分：** action 与 camera 分别累积有序 transport，再由 view-conditioned gate 调节两条修正分支，最终加到 3D-RoPE base score 上。
**图号与页码：** 论文 Figure 2，PDF Page 3。
**官方来源：** 官方 GitHub 高分辨率 `assets/method.png`，https://github.com/aigc3d/World-in-Context/blob/a985fd003458f07de5c8a537d750efbadd989aa9/assets/method.png 。
统一条件序列定义为：
$$c=[c^{\mathrm{view}};c^{\mathrm{action}};c^{\mathrm{camera}}]\in\mathbb{R}^{L_c\times D}$$
$L_c$是所有条件 token 的总长度。
$D$必须与视频主干 hidden dimension 匹配。
$c^{\mathrm{view}}$编码观察制度。
$c^{\mathrm{action}}$编码角色动作及局部增量。
$c^{\mathrm{camera}}$编码相机几何与运动。
三部分沿 sequence dimension 拼接，不是沿 channel dimension 简单相加。
随后构造共享序列$[c;z_\tau]$。
这一设计让控制之间、控制与视觉之间都可形成直接 attention edge。
### 3.2 ICC 与 Additive/AdaLN 的本质差异
Additive injection 把控制 feature 广播到视觉 token 后相加。
它要求不同控制先在外部组合成同形状表示。
信息进入后，token 身份与来源可能被加法混合。
AdaLN 用控制预测 affine scale/shift 调节主干激活。
它适合全局调制，但多控制间关系仍先在调制器中压缩。
ICC 保留每个控制 token 的独立身份和序列位置。
模型可以让某个视觉 patch 选择性关注某个时间、某个 slot、某类控制。
代价是条件 token 增加序列长度，从而增加 self-attention 计算与显存。
论文没有报告这种额外成本。
### 3.3 显式 view token
#### 3.3.1 公式与符号
每个 latent 时间位置$t$都有一个 view token：
$$c_t^{\mathrm{view}}=e_{\mathrm{view}}(v_t)+e_{\mathrm{type}}(\mathrm{view}),\quad v_t\in\{\mathrm{FREE},\mathrm{FPP},\mathrm{TPP}\}$$
$v_t$是第$t$个时刻的离散视角类别。
$e_{\mathrm{view}}$是类别 embedding lookup。
$e_{\mathrm{type}}(\mathrm{view})$是模态类型 embedding。
前者回答“是哪种 view”。
后者回答“这个 token 属于 view 模态”。
二者相加后直接与视觉 latent 交互。
论文将时间范围写到 latent temporal grid，而不是原始 RGB 每一帧。
#### 3.3.2 time-varying schedule
view token 不是整段视频只有一个常量标签。
它是随时间变化的 schedule：$\{v_t\}$。
若前半段为 TPP、后半段为 FPP，模型会在对应时间位置收到不同类别 token。
这正是 in-video switching 的接口基础。
单视角训练中的每条样本只出现常量 schedule。
推理时拼出分段 schedule，就构成训练组合之外的新控制序列。
37.5% 表明模型在没有切换样本时已部分学会组合。
93.8% 则说明 500 条切换监督显著修正过渡伪影和终态视角。
#### 3.3.3 动力学诊断
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/03-emergent-view-switching.png)
**图解：** Figure 3 左侧显示文本视角基线不产生目标角色/视点变化；显式 view token 已能切换但残留角色或构图伪影；加入 Stage 2 切换监督后过渡更稳定。
**数值：** text 6.25%，显式 view token 37.5%，加入 500 switching sequences 后 93.8%。
**图号与页码：** 论文 Figure 3，PDF Page 4。
**来源：** PDF 裁图，https://arxiv.org/pdf/2610.33169 。
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/04-view-motion-dynamics.png)
**图解：** Figure 4 在 100 个场景中固定 world-state caption 与连续前进 W，只改变 FREE/FPP view；显式建模后 FPP 保留较高周期振荡，FREE 更平滑，而文本基线两者接近。
**谨慎解读：** 阴影/曲线展示的是聚合轨迹诊断，不是对每个场景都正确的证明；图中虚线基线标为 illustrative。
**图号与页码：** 论文 Figure 4，PDF Page 5。
**来源：** PDF 裁图，https://arxiv.org/pdf/2610.33169 。
### 3.4 action token：四个语义 slot
#### 3.4.1 slot 分解
每个动作$a_t$被拆成集合：
$$S=\{\mathrm{move},\mathrm{turn},\Delta\mathrm{trans},\Delta\mathrm{rot}\}$$
`move`表达前后左右等离散移动键状态。
`turn`表达离散转向或视角按键状态。
$\Delta\mathrm{trans}$表达连续平移增量。
$\Delta\mathrm{rot}$表达连续旋转增量。
离散 slot 强调命令语义。
连续 slot 保留幅度与局部运动细节。
这种 hybrid action 避免只用键位丢失实际运动尺度，也避免只用轨迹丢失命令意图。
#### 3.4.2 token 公式
动作 token 定义为：
$$c_{t,p,s}^{\mathrm{action}}=\operatorname{LayerNorm}\!\left(h_s(a_t)+r_s(a_t)+e_t(t)+e_p(p)+e_s(s)\right)+e_{\mathrm{type}}(\mathrm{act})$$
$t$索引动作发生的时间步。
$p\in P_{\mathrm{act}}$索引同一时间步内部的 token 位置。
$s\in S$索引四种 slot 类型。
$h_s$是 slot-specific encoder，例如各 slot 独立 MLP。
$r_s$是连续幅度线索的 residual branch。
$e_t(t)$是时间 embedding。
$e_p(p)$是 within-step token-position embedding。
$e_s(s)$是 slot embedding。
$e_{\mathrm{type}}(\mathrm{act})$是 action 模态 embedding。
LayerNorm 在加 type embedding 之前应用。
#### 3.4.3 为什么需要五类身份信息
时间 embedding 说明“何时执行”。
within-step embedding 说明“同一步里的第几个 token”。
slot embedding 说明“移动、转向、平移增量还是旋转增量”。
type embedding 说明“这是 action，不是 view 或 camera”。
slot encoder 说明“不同字段有不同数值语义”。
这些身份被显式编码后，self-attention 不必从值分布中猜 token 类别。
#### 3.4.4 zero-init residual
$r_s$被零初始化。
训练开始时，新增连续 magnitude 分支对主干输出没有扰动。
模型先继承基座的视频生成行为，再逐步学习增量幅度。
这与 IT-RoPE 的$\lambda_b=0$初始化遵循相同稳定化原则。
论文没有给出$r_s$的层数、激活、输入维度与参数量。
也没有披露每个时间步究竟生成多少 action token。
因此不能虚构固定 action-token count。
### 3.5 camera token：相对位姿、Plücker ray 与局部 twist
#### 3.5.1 三种互补几何量
相机输入首先被标准化为相对位姿$\hat E_t$。
相对位姿避免绝对世界坐标系任意选择带来的漂移。
对采样像素/ray$(u,v)$，相机中心为$\hat o_t$。
世界坐标中的单位射线方向为$d_{t,u,v}$。
其 Plücker coordinate 为：
$$\pi_{t,u,v}=[\hat o_t\times d_{t,u,v};d_{t,u,v}]$$
前半部分$\hat o_t\times d$是 moment。
后半部分$d$是 direction。
二者共同表示三维有向直线，且不依赖在线上选哪个点。
局部刚体增量用$\xi_t\in\mathbb{R}^6$表示。
$\xi_t$可理解为$\mathfrak{se}(3)$中的 local twist，包含三维平移与三维旋转增量。
#### 3.5.2 camera token 公式
$$c_{t,u,v}^{\mathrm{camera}}=\operatorname{LayerNorm}\!\left(\phi_{\mathrm{ray}}(\pi_{t,u,v})+\phi_{\mathrm{pose}}(\operatorname{vec}(\hat E_t))+\phi_\Delta(\xi_t)\right)+e_{\mathrm{type}}(\mathrm{cam})$$
$\phi_{\mathrm{ray}}$投影每条 Plücker ray 的几何。
$\phi_{\mathrm{pose}}$投影展平后的全局相对 pose。
$\phi_\Delta$投影局部刚体运动。
第一项强调每个视线的空间方向与位置。
第二项提供相对于参考帧的全局姿态锚点。
第三项提供相邻时间之间的局部运动。
$e_{\mathrm{type}}(\mathrm{cam})$让模型区分 camera token 与其他 token。
#### 3.5.3 为什么三项不能简单互换
只有 pose 时，模型知道相机整体位置，但对每个视觉位置的射线几何表达较弱。
只有 ray 时，局部成像几何充分，却未必方便恢复全局轨迹锚点。
只有 local twist 时，短期运动明确，但累积误差可能随时间增长。
三者相加把 local、global 与 per-ray geometry 放进同一 token。
这是一种冗余但互补的编码。
论文没有披露 ray sampling 密度、camera token 数、内参处理与畸变模型。
### 3.6 action-camera 的互补性证据
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/05-action-camera-control.png)
**图解：** Figure 5 上方显示 action-only 更容易出现结构伪影，camera-only 又可能移动不足；下方量化四项指标，joint training 在局部响应和全局稳定之间折中。
| 训练条件 | Translation Accuracy | Rotation Accuracy | Geometric Consistency | Photometric Consistency |
|---|---:|---:|---:|---:|
| action-only | 89.6 | 89.5 | 88.1 | 79.4 |
| camera-only | 83.1 | 85.6 | 90.2 | 81.3 |
| joint training | 89.3 | 90.6 | 90.8 | 80.8 |
**逐项解释：** action-only 的平移最高，说明离散命令对短窗方向响应最直接。
**逐项解释：** joint 的旋转最高，说明 action 意图与 camera 几何共同约束转向更稳。
**逐项解释：** joint 的几何一致性最高，但 photometric 低于 camera-only 0.5 分。
**图号与页码：** 论文 Figure 5，PDF Page 5。
**来源：** PDF 裁图，https://arxiv.org/pdf/2610.33169 。
### 3.7 实际 tensor shape 的信息边界
论文只明确给出$c\in\mathbb{R}^{L_c\times D}$。
论文还给出 IT-RoPE 控制子通道$q_i^b,k_j^b\in\mathbb{R}^r$。
动作局部增量是$\Delta\mathrm{trans}$与$\Delta\mathrm{rot}$，但未给原始维度如何序列化。
相机 local twist 明确为$\xi_t\in\mathbb{R}^6$。
除此之外，论文未披露 VAE latent channel。
未披露 latent temporal downsample ratio。
未披露空间 patch/token 网格。
未披露 Transformer hidden size$D$。
未披露 attention head 数、层数与 head dimension。
未披露$r$的具体值。
未披露 camera ray 数。
未披露每时刻 action token 数。
未披露$c$与$z_\tau$拼接后的真实 sequence length。
因此不能把 Wan 社区常见尺寸自动写成 WiC 实际实现。
README 计划值 832×480、81 帧也不能倒推出论文训练张量。
## 4. IT-RoPE：把有序控制过程写进时间注意力
### 4.1 为什么普通 3D-RoPE 不够
3D-RoPE 编码 token 之间的相对时间与空间位置。
它能回答“两个 token 相隔多久、位于哪里”。
它不能直接回答“从时刻$i$到$j$之间执行了哪些动作”。
若两条轨迹起点和终点相同，但一条先左转再前进，另一条先前进再左转，端点编码相同。
交互世界模型却应区分两种过程，因为中间观察、遮挡与角色姿态不同。
IT-RoPE 的关键是把 interval 内的 ordered control progression 变成 transport operator。
### 4.2 双分支与一步变换
IT-RoPE 保持两条独立分支：
$$b\in\{a,c\}$$
$a$代表 action transport。
$c$代表 camera transport。
每个时间步与每条分支产生一个正交变换：
$$U_t^b\in O(r),\qquad (U_t^b)^\top U_t^b=I$$
$r$是控制专用 query/key 子通道维度。
正交性意味着变换保持向量内积尺度与范数。
因此 transport 更像旋转/反射式关系编码，而不是任意放大缩小 feature。
论文没有披露$U_t^b$由控制值生成的具体网络。
也没有披露采用 Cayley、Householder、指数映射还是其他正交参数化。
### 4.3 有序、非交换的 interval transport
当$i<j$时，定义：
$$T_{i\to j}^b=\prod_{k=i+1}^{j}U_k^b$$
乘积按时间顺序排列。
它不是把控制向量求和，也不是只取$u_j-u_i$。
如果$U_p^bU_q^b\neq U_q^bU_p^b$，执行顺序会改变结果。
这就是 non-commutative composition。
例如“先转向再前进”与“先前进再转向”可得到不同$T$。
同端点轨迹因此不再被压成同一关系。
### 4.4 逆、组合与一致性
正交性给出反向关系：
$$T_{j\to i}^b=(T_{i\to j}^b)^\top=(T_{i\to j}^b)^{-1}$$
这使同一套累计量同时支持 forward 与 backward temporal relation。
区间 transport 还满足组合：
$$T_{i\to j}^bT_{j\to k}^b=T_{i\to k}^b$$
从$i$到$j$再到$k$等于直接累计$i$到$k$。
这一代数一致性是 transport 比任意 pairwise MLP bias 更有结构的地方。
它也为 prefix factorization 提供基础。
### 4.5 base attention 与控制子通道
普通 3D-RoPE score 为：
$$s_{ij}^{\mathrm{base}}=\frac{\bar q_i^\top\bar k_j}{\sqrt d}$$
$\bar q_i$与$\bar k_j$是应用基础 3D-RoPE 后的 query/key。
$d$是基础注意力通道维度。
IT-RoPE 不替换这项，而是在其上添加控制修正。
对每个分支$b$，另取$q_i^b,k_j^b\in\mathbb{R}^r$。
这些是 action/camera 专用子通道。
缩放因子使用$1/\sqrt r$，与标准 scaled dot-product 一致。
### 4.6 centered correction
完整 score 为：
$$s_{ij}=s_{ij}^{\mathrm{base}}+\sum_{b\in\{a,c\}}\frac{\lambda_b}{\sqrt r}m_{ij}^bg_i^bg_j^b(q_i^b)^\top(T_{t_i\to t_j}^b-I)k_j^b$$
$t_i$与$t_j$是 token$i,j$对应的 latent 时间索引。
$T_{t_i\to t_j}^b$累计两时刻之间的控制变换。
减去$I$称为 centered correction。
如果$t_i=t_j$，则$T=I$，修正严格为零。
如果区间静止且每一步$U_t^b=I$，修正也为零。
因此 IT-RoPE 不会给静态关系平白增加偏置。
它只编码“相对 base attention 多出来的控制 transport”。
### 4.7 view/type gate
$g_i^b\in(0,1)$是分支 gate。
它由 view mode 与 token type 条件化。
同一个 action 在 FPP、FREE 与 TPP 中可以有不同影响。
同一 transport 对 view token、action token、camera token 与 visual token 也可被不同门控。
乘积$g_i^bg_j^b$要求 query 与 key 两端共同决定分支强度。
这是论文把“当前观察制度”与“控制关系”耦合起来的关键位置。
论文没有给 gate 网络结构、输入拼接方式或参数量。
### 4.8 interval mask
$m_{ij}^b\in\{0,1\}$是有效区间 mask。
它只允许同一 contiguous valid execution segment 内的 token pair 使用修正。
若控制缺失、片段断开或跨越无效边界，mask 置零。
这样可避免把一个片段的 action transport 错连到另一个片段。
在 mixed action-only/camera-only 数据中，分支 mask 尤其重要。
它也说明“不具备某控制”并不等价于“该控制恒为零运动”。
### 4.9 zero-init scale
$\lambda_a$与$\lambda_c$是可学习分支尺度。
两者初始化为零。
训练第一个 step 时：
$$s_{ij}=s_{ij}^{\mathrm{base}}$$
所以新模块初始行为严格退化回 Wan 主干的 base attention。
这降低了在有限 100K 交互数据上破坏预训练生成能力的风险。
随着训练进行，模型才逐步打开 action/camera correction。
### 4.10 prefix product factorization
在从起始时间$s$开始的连续有效片段内定义 prefix：
$$P_s^b=I,\qquad P_t^b=P_{t-1}^bU_t^b$$
于是：
$$T_{i\to j}^b=(P_i^b)^\top P_j^b$$
推导如下。
因为$P_i^b=U_{s+1}^b\cdots U_i^b$。
而$P_j^b=P_i^bU_{i+1}^b\cdots U_j^b$。
左乘$(P_i^b)^\top=(P_i^b)^{-1}$后，前缀部分抵消。
剩余正是$U_{i+1}^b\cdots U_j^b$。
### 4.11 query-key 因子化
centered inner product 可写成：
$$(q_i^b)^\top(T_{t_i\to t_j}^b-I)k_j^b=(P_{t_i}^bq_i^b)^\top(P_{t_j}^bk_j^b)-(q_i^b)^\top k_j^b$$
第一项先分别用各自时间的 prefix 变换 query/key，再做普通点积。
第二项减去未 transport 的点积。
这样无需为每个$(i,j)$显式构造$r\times r$矩阵$T_{i\to j}^b$。
prefix 累积随时间长度线性增长。
对每个 token 应用 prefix 变换也随 token 数线性增长。
### 4.12 “线性”的准确含义
论文所说 linear scaling 仅指 transport preprocessing 与 query/key transform。
它消除了为所有 pair 显式存储 interval transformation matrix 的额外二次开销。
但后续仍要形成所有 query-key pair 的 attention score。
若主干使用 full attention，总体潜在复杂度仍为：
$$O(N^2d)$$
$N$是包含视觉与条件 token 的总序列长度。
IT-RoPE 没有把 full self-attention 变成线性注意力。
“linear transport preprocessing”绝不能写成“整个模型线性复杂度”。
### 4.13 IT-RoPE 与状态空间的关系
transport 提供的是关系算子，不是可读出的显式物理状态。
它保留了控制序列的顺序信息。
但它不直接估计物体位置、速度、质量、接触或拓扑地图。
因此 IT-RoPE 更接近“控制条件化的各向异性亲和性”。
它改变哪些历史 token 与当前 token 更应相似或相关。
这与显式 3D simulator 或 structured state-space model 仍有本质差异。
## 5. 数据、标注与训练课程
### 5.1 数据规模与两阶段事实
论文的基础集合是 100K single-regime navigation clips。
它覆盖 FREE、FPP、TPP，但每条 clip 内通常只有一种 view regime。
另有 500 条 FPP↔TPP switching sequences。
不能把 Stage 2 写成“只在 500 条上微调”。
真实 Stage 2 构造了 8K mixed curriculum。
它混合普通导航 clips 与 500 条 switching sequences。
switching sequences 被过采样，以强化切换稳定性。
同时保留 navigation clips，避免灾难性遗忘基础导航。
### 5.2 数据来源比例
| 数据源 | 比例 |
|---|---:|
| DL3DV | 29.22% |
| SpatialViD | 27.50% |
| ABot-World-Explorer | 25.24% |
| EgoExo | 8.76% |
| Sekai | 5.45% |
| RealEstate | 3.83% |
六项合计为 100%。
前三项合计 81.96%，数据分布明显集中。
500 条 switching sequences 来自 EgoExo 与内部数据。
论文没有披露两者的具体比例。
也没有披露 internal data 的许可、地域、场景与去重策略。
### 5.3 三阶段自动过滤
第一阶段用 TransNetV2 检测 shot boundary。
含明显切镜或多镜头拼接的视频被移除。
这使训练片段更接近连续相机运动，而非剪辑叙事。
第二阶段用 UniMatch optical flow 与 VMAF 评估运动。
静止停滞、过度运动模糊或运动不连续片段被过滤。
第三阶段用 Dover 评估美学和感知质量。
严重压缩、过曝或低视觉质量样本被剔除。
论文没有给各阶段保留率与阈值。
没有给人工抽检误差率。
### 5.4 view 标注
Qwen3-VL-32B 把 clips 分类为 FREE、FPP、TPP。
FREE 由平滑、无约束的航拍或相机漫游定义。
FPP 由具身运动或驾驶舱视角定义。
TPP 由外部跟随角色的相机定义。
边界模糊样本会结合重建 camera trajectory statistics 复核。
论文没有说明复核是自动规则还是人工过程。
也没有报告三分类准确率与混淆矩阵。
### 5.5 camera 与 action 标注
camera 使用 ViPE 重建连续 6-DoF trajectory 与 Plücker ray 参数。
action 并非主要来自游戏引擎 telemetry。
大部分离散角色动作由 ViPE 估计的 camera trajectory 反演得到。
FPP 中会先分离并过滤 body-induced oscillation 与 head bobbing，再离散化 action。
这会产生一个重要统计事实：action supervision 与 camera supervision 并不独立。
二者共享 ViPE 误差源和相机运动先验。
因此“action 与 camera 互补”不等于它们来自独立真值通道。
### 5.6 static world-state caption
caption 只总结 atmosphere、spatial geometry、layout entities 等静态构成。
动作、视点变化与时序演化不写入文本。
这一设计把 scene constitution 与 action-driven exploration 解耦。
它降低语言动态描述与控制 token 冲突的可能。
但也把动态背景留给视频先验，缺少显式控制。
### 5.7 mixed-control schema
论文所说 joint training 不是每条样本都有 action+camera。
真实混合包含四类：action-only、camera-only、joint、unconditioned。
原因是现实视频的双模态标注经常不可靠。
TPP loose camera following 会让 action 反演困难。
合成场景也可能有 camera pose estimation artifacts。
混合训练让模型适应控制缺失。
所有架构对照使用相同混合 schema，从而减少数据条件混淆。
论文图示了分布，但正文抽取未给每类精确百分比。
### 5.8 训练目标与默认形态
论文称训练遵循 conditional flow-matching objective。
默认模型使用 bidirectional finetuning。
视频 token 与控制 token 在 shared attention 中双向交互。
论文没有给出完整 flow-matching loss 公式与噪声采样分布。
也没有说明是否联合更新 Wan 全部权重、LoRA 或 adapter。
README 计划接口称只优化 LoRA、condition encoder 与 IT-RoPE。
但当前仓库没有源码/config，不能验证该说法是否等于论文实际训练。
### 5.9 论文未披露的训练信息
论文未披露训练分辨率。
未披露每条 clip 的帧数与 FPS。
未披露 GPU 型号与数量。
未披露训练 step、epoch 与 wall-clock。
未披露 learning rate、scheduler、optimizer 与 weight decay。
未披露 global batch、gradient accumulation 与随机种子。
未披露 LoRA rank、target modules 与 adapter 参数量。
未披露各 view/action/camera mode 的精确采样概率。
因此无法对训练成本与统计稳定性做完整复现判断。
## 6. WBench 评测协议
### 6.1 规模与输入
WBench 包含 289 cases 与 1,058 interaction turns。
每个 case 提供 initial image、world description 与 multi-turn control sequence。
论文使用 Navigation split 与 FPP↔TPP perspective-switching subset。
主表同时报告五维 Navigation 与单独 Switching。
### 6.2 五个 Navigation 维度
Quality 衡量美学、视觉保真、运动动态、闪烁与帧间平滑。
Setting 衡量场景外观、环境风格与主体持续性对 prompt 的遵循。
Interaction 衡量平移、旋转和对称转向命令的响应与轨迹符合度。
Consistency 衡量角色身份、无切镜、光度与几何一致性。
Physical 衡量真实物理、常识动力学与因果连贯性。
Average 是五维聚合后的 Navigation headline score。
论文没有在本报告中展开每个维度的 judge 模型、prompt 与置信度。
### 6.3 Perspective Switching
每次请求 FPP↔TPP transition 后，评测生成 rollout。
评估关注过渡平滑度、终态 perspective 与 framing geometry。
每个 turn 得到 binary success decision。
最后聚合成独立 Switching 百分比。
这能隔离切换能力，但也把连续质量压成二元判断。
VLM judge 的具体模型、提示、阈值敏感性与人类一致率并不透明。
### 6.4 Figure 4 的振荡指标
固定连续前进 W，并分别用 FREE/FPP view 生成。
ViPE 重建$T_t=[R_t,p_t;0,1]$。
相对初始位置为：
$$q_t=R_0^\top(p_t-p_0)$$
横向/竖向坐标为$u_t=(q_{t,x},-q_{t,y})$。
使用二阶、33 帧 Savitzky–Golay 滤波提取平滑趋势。
残差为：
$$r_t=u_t-\operatorname{SG}_{33,2}(u)_t$$
归一化振荡幅度为：
$$\operatorname{RMS}_{\mathrm{osc}}=\frac{100}{L}\sqrt{\frac{1}{|\mathcal I|}\sum_{t\in\mathcal I}(r_{t,x}^2+r_{t,y}^2)}$$
$L=|q_{T-1,z}|$是净前进位移。
$\mathcal I$排除序列首尾各 0.5 秒。
归一化降低了不同前进距离对振幅的影响。
### 6.5 action-camera 四项诊断
Translation Accuracy 在不重叠 0.5 秒窗口中比较局部位移方向。
方向误差需不超过 30°，且位移超过深度归一化阈值。
Rotation Accuracy 在$\mathfrak{so}(3)$中比较相对旋转向量。
方向误差同样需不超过 30°，旋转幅度还需超过时间相关阈值。
Geometric Consistency 使用 Depth Anything 3 深度、内参和 pose 做跨帧重投影。
其分数为$100/(1+\operatorname{mean}(e_j))$。
Photometric Consistency 通过 depth-ordered splatting 与 PSNR 聚合。
该分析排除 TPP，只研究 FPP/FREE。
因此它不能直接验证最需要 action-camera 解耦的第三人称场景。
## 7. 主结果：性能、规模与可比性
### 7.1 Table 1 完整结果
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/06-main-results.png)
**图解：** 官方仓库高分辨率 Table 1 同时给出模型规模、支持控制、注入策略、五维 Navigation、平均与 Switching；WiC 的突出点是 Switching 93.8，而不是所有五维都第一。
| 模型 | Size | Strategy | Quality | Setting | Interaction | Consistency | Physical | Avg. | Switch |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| EchoWM | 22B | Positional Attn. | 81.5 | 79.4 | 86.6 | 89.8 | 70.6 | 81.6 | 32.0 |
| LingBot-World v2 | 14B | AdaLN | 81.8 | 76.8 | 82.8 | 86.5 | 69.1 | 79.4 | 21.9 |
| Alaya-EVOKE | 14B | Warp Condition | 82.8 | 83.8 | 78.6 | 86.9 | 72.1 | 80.8 | 18.8 |
| HY-World 1.5 | 8.3B | Modulation + PRoPE | 78.1 | 72.2 | 86.8 | 86.9 | 66.3 | 78.1 | 12.5 |
| Wan 2.2 | 5B | Text | 72.2 | 69.3 | 65.4 | 86.4 | 52.3 | 69.1 | 6.3 |
| YUME 1.5 | 5B | Text + Cross-Attn. | 77.6 | 72.4 | 71.4 | 80.1 | 65.2 | 73.3 | 12.5 |
| ABot-World | 5B | Additive Inject. | 76.8 | 71.4 | 84.0 | 79.5 | 61.7 | 74.7 | 9.4 |
| DreamX-World | 5B | Positional Attn. | 77.5 | 80.8 | 78.6 | 74.9 | 63.3 | 75.0 | 18.8 |
| Zing-0.5 | 5B | Additive Inject. | 80.6 | 77.8 | 84.2 | 88.5 | 73.8 | 81.0 | 12.5 |
| World-in-Context | 5B | In-Context | 80.9 | 81.0 | 86.0 | 87.2 | 72.3 | 81.5 | 93.8 |
**图号与页码：** 论文 Table 1，PDF Page 6。
**官方来源：** 官方 GitHub `assets/table.png`，https://github.com/aigc3d/World-in-Context/blob/a985fd003458f07de5c8a537d750efbadd989aa9/assets/table.png 。
### 7.2 WiC 的强项与非强项
WiC 的 Setting 81.0，高于其他 5B 模型。
Interaction 86.0，也高于其他 5B 模型。
Average 81.5，比 Zing-0.5 高 0.5。
Switch 93.8，比最佳其他 5B 的 18.8 高 75.0 个百分点。
但 Quality 80.9 低于 Alaya-EVOKE 82.8。
Consistency 87.2 低于 EchoWM 89.8 和 Zing-0.5 88.5。
Physical 72.3 低于 Zing-0.5 73.8。
Interaction 86.0 也略低于 HY-World 1.5 的 86.8 与 EchoWM 的 86.6。
因此“全面第一”是不准确的。
### 7.3 5B 与 22B 的正确比较
WiC-5B Average 81.5，EchoWM-22B 为 81.6。
二者 headline 平均只差 0.1。
WiC Switching 93.8，EchoWM 32.0。
但训练数据规模跨工作不可得。
控制接口、输入条件、生成设置与评测输出也可能不同。
所以可说“在 WBench 表中分数相当”。
不能说“ICC 让 5B 普遍等价于 22B”。
更不能把参数量差异直接当作训练或推理 FLOPs 比例。
### 7.4 定性主结果
论文 Figure 6 将 WiC 与 Zing-0.5 对比。
Zing-0.5 容易产生夸张位移尺度。
WiC 的轨迹更克制，更符合 Interaction 分数优势。
在 FPP↔TPP 例子中，WiC 完成视角转换，baseline 未完成。
Figures 7–9 展示 FREE/FPP/TPP 的长序列结果。
Figure 10 展示 TPP→FPP→TPP 多阶段切换。
这些图支持视觉可行性，但没有展示随机失败率或 cherry-picking 规则。
### 7.5 主表协议不完全公平
不同工作可能使用不同训练数据、分辨率、时长、采样步数与后处理。
论文自己也承认跨方法数据规模经常不披露。
只有论文内部重实现的 Additive/AdaLN 对照能较好控制数据 schema。
主表中的外部模型更适合作为 reference，而不是严格因果比较。
93.8 的视角切换还依赖专门 Stage 2，而很多 baseline 未必拥有等价 switching supervision。
## 8. 消融与受控分析
### 8.1 ICC vs Additive/AdaLN
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/07-icc-conditioning-ablation.png)
**图解：** Figure 11 从 C 到 A+C 再到 V+A+C 逐步增加条件；ICC 的优势不是 camera-only 时突然出现，而是随组合复杂度扩大。
| 方法 | 条件 | Navigation | Interaction |
|---|---|---:|---:|
| Additive Injection | C | 76.3 | 79.4 |
| Additive Injection | A+C | 78.5 | 83.7 |
| Additive Injection | V+A+C | 80.2 | 85.4 |
| AdaLN Modulation | C | 77.0 | 78.5 |
| AdaLN Modulation | A+C | 78.2 | 83.3 |
| AdaLN Modulation | V+A+C | 79.6 | 84.8 |
| ICC | C | 76.3 | 79.1 |
| ICC | A+C | 79.1 | 84.8 |
| ICC | V+A+C | 81.2 | 86.8 |
**图号与页码：** 论文 Figure 11，PDF Page 12。
**来源：** PDF 裁图，https://arxiv.org/pdf/2610.33169 。
### 8.2 对 Figure 11 的逐层解释
在 C-only 下，AdaLN Navigation 77.0 最高。
ICC 的 76.3 与 Additive 相同，Interaction 79.1 还低于 Additive 79.4。
这说明 ICC 不是对单一控制天然占优。
在 A+C 下，ICC Navigation 79.1。
相对 Additive 78.5 提升 0.6。
相对 AdaLN 78.2 提升 0.9。
ICC Interaction 84.8。
相对 Additive 83.7 提升 1.1。
相对 AdaLN 83.3 提升 1.5。
在 V+A+C 下，ICC 81.2/86.8。
相对最佳非 ICC 的 Additive 80.2/85.4 提升 1.0/1.4。
“compounding advantage”有数值依据，但幅度仍在 1–1.5 分量级。
没有多 seed 方差，无法判断部分差距是否稳定显著。
### 8.3 IT-RoPE 消融
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/08-it-rope-ablation.png)
**图解：** Figure 12 比较标准 3D-RoPE、action-only transport、camera-only transport、只看端点与 full 双分支；action branch 对 Interaction 更敏感，camera branch在图示精度下主要改善 Navigation。
| 位置/transport 变体 | Navigation | Interaction |
|---|---:|---:|
| Standard 3D-RoPE | 80.5 | 86.2 |
| Action-Transport-Only | 80.9 | 86.9 |
| Camera-Transport-Only | 80.9 | 86.2 |
| Endpoint-Only | 80.5 | 86.6 |
| Full IT-RoPE | 81.2 | 86.8 |
**图号与页码：** 论文 Figure 12，PDF Page 12。
**来源：** PDF 裁图，https://arxiv.org/pdf/2610.33169 。
### 8.4 对 Figure 12 的严格解读
Full 相对 Standard 的 Navigation 提升 0.7。
Interaction 提升 0.6。
Action-only 的 Interaction 86.9，甚至比 Full 86.8 高 0.1。
因此 Full 不是每个单项都绝对最高。
它的优势是 Navigation 最高且 Interaction 接近最高，即更均衡。
Camera-only 的 Interaction 与 Standard 同为 86.2。
Endpoint-only Interaction 86.6，高于 Standard 0.4，但 Navigation 不变。
Full 相对 Endpoint 的 Navigation 提升 0.7、Interaction 提升 0.2。
这支持“过程顺序有用”，但提升较小且无误差条。
### 8.5 data efficiency
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/09-data-efficiency.png)
**图解：** Figure 13 在三个预算上比较 ICC 与 Additive；ICC 在 Navigation 与 Switch 全部领先，Switch 的差距最明显。
| 预算 | Additive Navigation | ICC Navigation | Additive Switch | ICC Switch |
|---|---:|---:|---:|---:|
| 50K+250 | 77.3 | 79.1 | 59.4 | 75.0 |
| 100K+500 | 80.0 | 81.5 | 71.9 | 93.8 |
| 125K+750 | 80.3 | 81.9 | 78.1 | 96.9 |
**图号与页码：** 论文 Figure 13，PDF Page 12。
**来源：** PDF 裁图，https://arxiv.org/pdf/2610.33169 。
### 8.6 data scaling 的含义
ICC 50K+250 的 Switch 75.0 已高于 Additive 100K+500 的 71.9。
ICC 100K+500 的 Navigation 81.5 高于 Additive 125K+750 的 80.3。
ICC 100K+500 的 Switch 93.8 高于 Additive 125K+750 的 78.1。
这证明同一训练框架内 ICC 的样本利用率更高。
它没有证明 ICC 的总计算更低，因为条件 token 可能增加每 step 成本。
125K+750 相比 100K+500 的 ICC Navigation 只增 0.4。
Switch 从 93.8 增到 96.9，仍有 3.1 分增益。
曲线只有三个点，不能拟合普适 scaling law。
### 8.7 view switching 不是“纯涌现”
37.5% 是最接近 emergent composition 的结果。
因为它来自 single-regime clips 与时变 token schedule 的组合。
93.8% 则明确依赖 Stage 2 switching consolidation。
Stage 2 不是只看 500 条，而是构造 8K mixed curriculum。
所以 93.8 不能宣传为“完全未见切换数据的纯涌现”。
更准确表述是“显式 token 先产生可组合接口，再用少量切换序列高效巩固”。
## 9. Causal 与 richer controls 的适用性
### 9.1 从双向 clip 到 causal history
默认 WiC 是双向 clip finetuning。
它在一个片段内允许视频与控制 token 共同双向交互。
这种设置适合比较 conditioning 方法，却不是严格 streaming online generation。
WiC-Causal 改为预测未来 latent，并把历史放进 context。
条件序列扩展为：
$$c^{\mathrm{causal}}=[c^{\mathrm{view}};c^{\mathrm{action}};c^{\mathrm{camera}};c^{\mathrm{sink}};c^{\mathrm{recent}}]$$
$c^{\mathrm{sink}}$保留最初上下文，作为全局语义锚点。
$c^{\mathrm{recent}}$保存近期视觉 token 的滑动窗口。
这种设计避免单纯依赖外部 KV retrieval。
但论文未给 sink 数、recent window 长度与更新策略。
### 9.2 WiC-Causal 完整 Table 2
| 模型 | Quality | Setting | Interaction | Consistency | Physical | Avg. | Switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wan 2.2 | 72.2 | 69.3 | 65.4 | 86.4 | 52.3 | 69.1 | 6.3 |
| YUME 1.5 | 77.6 | 72.4 | 71.4 | 80.1 | 65.2 | 73.3 | 12.5 |
| ABot-World | 76.8 | 71.4 | 84.0 | 79.5 | 61.7 | 74.7 | 9.4 |
| DreamX-World | 77.5 | 80.8 | 78.6 | 74.9 | 63.3 | 75.0 | 18.8 |
| Zing-0.5 | 80.6 | 77.8 | 84.2 | 88.5 | 73.8 | 81.0 | 12.5 |
| World-in-Context | 80.9 | 81.0 | 86.0 | 87.2 | 72.3 | 81.5 | 93.8 |
| WiC-Causal | 80.5 | 80.4 | 85.7 | 87.6 | 72.0 | 81.2 | 90.6 |
Causal 相比双向 Avg 下降 0.3。
Switch 下降 3.2 个百分点。
Consistency 反而从 87.2 升到 87.6。
这支持 causal adaptation 的可行性。
但没有 rollout 长度—性能曲线，无法判断误差何时累积失稳。
### 9.3 WiC-Causal 定性结果
![](assets/World-in-Context%20-%20上下文统一交互式视频世界模型/figures/10-wic-causal.png)
**图解：** Figure 15 展示企鹅雪地、教堂、自行车雪景、黑白庭院、霓虹城市与洞穴等长序列；多行帧序列包含转身、回望、场景重现和视角变化。
**积极信号：** 场景主色、建筑结构与主体身份在若干回环中得以保持。
**边界：** 图像没有标注真实时长、帧数、回环误差或失败样例，不能据此量化“长期记忆”。
**图号与页码：** 论文 Figure 15，PDF Page 15。
**来源：** PDF 裁图，https://arxiv.org/pdf/2610.33169 。
### 9.4 WiC-Game
WiC-Game 在 action context 中加入：
$$[c^{\mathrm{fire}};c^{\mathrm{duck}};c^{\mathrm{walk}}]$$
三者分别代表开火、下蹲与慢走。
训练只使用 Counter-Strike 2 的 DUST II 紧凑子集。
论文定性展示训练地图内新控制轨迹。
也展示 MIRAGE、NUKE 等未见地图。
还用合成起始图测试真实风格视觉迁移。
结果说明 token 接口容易扩展新离散 control type。
但没有 quantitative score，也没有与专用 game world model 比较。
## 10. 官方仓库审计与工程重建边界
### 10.1 结论：未开源 / coming soon
截至 2026-10-27，官方仓库不是可运行代码 release。
固定 commit 为`a985fd003458f07de5c8a537d750efbadd989aa9`。
GitHub tree 只有 9 个 tracked files。
根目录是`.gitignore`、`LICENSE`、`README.md`。
`assets/`下有两个 badge、logo、method、table、teaser，共 6 个文件。
仓库没有任何`.py`源码。
没有`requirements.txt`。
没有`pyproject.toml`或`setup.py`。
没有`wic/`package。
没有`scripts/`。
没有`configs/`。
没有`examples/`。
没有模型权重或下载 manifest。
没有数据与数据格式文档。
因此不能写完整“代码级实现”章节，也不能引用不存在的`wic.*`类或函数。
### 10.2 README planned release 的准确记录
README 明确写明 code 与 WiC weights 正在 review，尚不可用。
其下半部分 usage instructions 描述 planned release。
计划环境是 Linux、Python 3.10、NVIDIA BF16 GPU。
计划测试栈是 PyTorch 2.4、CUDA 12.4、FlashAttention 2.7.4。
计划推理默认是 832×480。
计划生成 81 frames、16 FPS。
计划 CFG 为 5。
计划 sampling steps 为 50。
计划提供 11 examples：3 FREE、3 FPP、3 TPP、2 switching。
计划训练只优化 LoRA、condition encoder 与 IT-RoPE。
计划每 GPU batch 为 1 clip。
这些值全部是**计划接口**。
当前仓库不存在对应源码、config、requirements、example 或 checkpoint。
所以无法验证命令、默认值、显存、输出与论文训练的一致性。
### 10.3 计划接口不能冒充论文训练配置
832×480 不能写成论文训练分辨率。
81 帧不能写成训练 clip length。
16 FPS 不能写成训练采样率。
CFG 5 与 50 steps 只能视为计划推理默认值。
per-GPU batch 1 只能视为计划 launcher 说明。
LoRA+condition encoder+IT-RoPE 只能视为 README 预告的优化范围。
论文正文没有给这些对应事实。
### 10.4 仓库与权重覆盖
GitHub tags 快照为空数组。
GitHub releases 快照为空数组。
未找到官方 Hugging Face 的 WiC adapter 或 WiC model weight。
未找到官方 ModelScope 的 WiC 权重。
README 只链接 Wan2.2-TI2V-5B backbone。
backbone 可得不等于 WiC 可得。
没有 WiC condition encoder 与 IT-RoPE 参数，就不能复现论文模型。
### 10.5 许可证记录
仓库根 LICENSE 是 Apache License 2.0 全文。
README 称计划代码按 Apache-2.0 发布。
但当前没有实际代码可判断覆盖范围。
LICENSE 模板末尾仍写`Copyright [2023] [Zhongjie Duan]`。
这与论文作者/机构信息不一致，可能是模板残留。
这里只记录 provenance 异常，不作法律判断。
### 10.6 可做与不可做的工程重建
可做：依据论文公式独立实现 ICC token 拼接。
可做：独立实现 orthogonal prefix transport 与 attention correction。
可做：在公开 Wan2.2 主干上设计近似 condition encoder。
不可做：声称独立实现就是官方实现。
不可做：猜测官方类名、配置键、tensor shape 或模块插入层。
不可做：把 README 计划命令当作已运行接口。
不可做：声称已复现 81.5/93.8。
## 11. 与相关工作的关系
### 11.1 相机中心范式
许多交互视频模型把 camera trajectory 作为统一控制。
这种做法在 FREE 导航中自然。
在 FPP 中也能近似角色移动。
但在 TPP 中角色与镜头明显分离。
WiC 的价值是把“camera 是所有运动的代理”改成“camera 与 action 是并列上下文”。
### 11.2 模块化 conditioning
AdaLN、cross-attention 与 additive injection 都是成熟接口。
它们的优势是计算边界清楚、容易复用现有主干。
WiC 并没有证明这些接口本身错误。
Figure 11 甚至显示 C-only 时 AdaLN 更好。
WiC 证明的是条件丰富后，共享 token 交互更有优势。
### 11.3 与 in-context generation 的关系
语言模型中的 in-context learning 依赖同一序列内示例与任务共同推断。
视觉生成中的 in-context 方法也把多任务条件放进共享 attention。
WiC 进一步把 interactive control 视为 context token。
它不是 few-shot demonstration 的狭义 in-context learning。
这里的“in-context”主要指条件作为序列成员参与 self-attention。
### 11.4 与相对相机位置编码的关系
普通 camera-aware RoPE 把相机位姿变成 token pair 的几何关系。
IT-RoPE 更进一步：关系由一串 action/camera step 乘积产生。
它强调 path，而非只强调 endpoint pose。
这种从状态差到 ordered operator product 的转变，是本文最有数学辨识度的贡献。
## 12. 局限与批判性评价
### 12.1 主表不是严格同协议竞技
模型规模、训练数据、训练预算与采样设置不一致。
外部结果与内部实现混在同一表中。
训练数据规模又普遍不披露。
因此主表适合展示位置，不足以隔离架构因果。
### 12.2 5B vs 22B 不能归因 ICC
81.5 对 81.6 很亮眼。
但模型规模之外还改变了数据、主干、损失、控制与推理设置。
只有同一 Wan 基座、同一数据和预算下的 ICC/Additive/AdaLN 才接近因果证据。
### 12.3 93.8 不是纯涌现
37.5 才是 single-regime training 后的零切换监督结果。
93.8 使用 500 unique switching sequences。
而且 Stage 2 实际是过采样这些序列的 8K mixed curriculum。
把 93.8 写成“无切换数据涌现”会误导。
### 12.4 WBench VLM 二元 judge 不透明
Switching 用 VLM-based binary decision 聚合。
论文没有充分展示 judge prompt、模型版本、阈值敏感性与人工一致率。
二元指标也掩盖轻微伪影与过渡质量差异。
五维 Navigation 中若也依赖 learned evaluator，同样需要 calibration 报告。
### 12.5 默认双向模型不是实时在线模型
默认配置在 clip 内双向建模。
它不能自然等价为因果 streaming。
WiC-Causal 是另一个训练/适配设置。
论文没有报告实时帧率、首帧延迟或在线控制回路延迟。
### 12.6 causal rollout 长度缺少定量
Figure 15 看起来较长且能回环。
但没有明确生成秒数、累计帧数与 memory window。
没有 score-versus-horizon 曲线。
没有展示不同历史压缩策略的漂移。
“long-horizon”目前主要是定性主张。
### 12.7 无显式 3D/物理状态
Plücker ray 与 camera pose 是显式几何条件。
这不等于模型内部维护显式 3D scene representation。
论文没有 depth/map/object pose/velocity 状态读出。
Physical 分数也不能证明学到真实物理定律。
模型本质仍是视觉 latent dynamics generator。
### 12.8 数据污染、许可与标注误差不透明
100K 数据来自多个公开视频集与 ABot 数据。
没有样本级 provenance、许可汇总与去重统计。
没有 WBench contamination audit。
Qwen3-VL view label 与 ViPE pose 的误差没有量化。
内部 500 switching data 的来源边界也不清楚。
### 12.9 action 与 camera 监督相关
动作主要从 camera trajectory 反演。
因此 action-only 与 camera-only 不是两套独立传感器真值。
ViPE 偏差可能同时进入两者。
互补结果可能部分来自离散化与连续表述的差别，而非独立因果因素。
更强实验应使用游戏 telemetry action 与真实 camera pose 对照。
### 12.10 ICC token 的额外计算未报告
将$c$前缀拼到$z_\tau$会增加序列长度。
full attention 的开销随总 token 数潜在二次增长。
论文没有报告$L_c$、显存、吞吐与每步 latency。
因此“统一且可扩展”仍缺计算尺度证据。
### 12.11 没有延迟、显存与 FLOPs
没有训练 FLOPs。
没有推理 FLOPs。
没有峰值显存。
没有 clips/s 或 frames/s。
没有 IT-RoPE 相对 3D-RoPE 的 kernel 开销。
README 的计划硬件栈不能替代实际 benchmark。
### 12.12 没有多 seed 与置信区间
Figure 11–13 都没有 error bar。
主表也没有 seed 数与标准差。
0.1–1.0 分差距可能受采样随机性影响。
Switch 93.8 的样本数与二项置信区间也未展开。
### 12.13 TPP 动机与消融范围错位
论文强调 TPP 最需要 action-camera 分离。
但 Figure 5 的 action-camera 消融明确排除 TPP。
这使实验只验证 FPP/FREE 中的互补性。
最核心的 TPP 解耦主张反而缺少 action-only/camera-only/joint 定量表。
### 12.14 总体评价
**问题定义：高价值。** 它把视角制度与动作—相机自由度放回交互世界模型中心。
**方法创新：强。** ICC 简洁，IT-RoPE 的 ordered transport 有明确数学结构。
**受控实验：较强。** Figure 11–13 提供同数据架构比较。
**外部比较：中等。** 主表异质，不能进行强因果归因。
**复现性：低。** 官方仓库目前仅占位，核心代码、权重、配置与数据均不可得。
**工程部署证据：低。** 没有速度、显存、实时性或长期稳定定量。
## 13. 观点分级与旭涛研究方向的联系
### 13.1 强结论
显式 view token 是有效的可组合控制接口。
共享 token context 在丰富条件下优于分离式注入。
有序 control transport 能补充普通时空 RoPE。
这些结论都有同数据受控实验支撑。
### 13.2 中等结论
动作主要提供局部 responsiveness，相机主要提供 global stability。
ICC 在同等样本预算下提高切换监督效率。
因 action 来自 camera inversion、无独立 telemetry 与 seed 方差，证据强度只能定为中等。
### 13.3 弱结论
WiC 具备长期空间记忆。
WiC-Game 对真实世界有广泛控制迁移。
5B 规模已等价于更大世界模型。
这些说法主要依赖定性图或异质主表。
### 13.4 尚未证明
模型拥有显式物理状态。
模型可实时在线运行。
模型在任意长时域不漂移。
ICC 在相同 FLOPs 下仍然更优。
训练数据不存在污染与许可问题。
### 13.5 与 3D JEPA / Asset-JEPA 的联系
3D JEPA 与 Asset-JEPA 关注从观察中形成可预测、可复用的结构表示。
WiC 关注把控制关系写进 latent token 的亲和性。
两者共性是减少让网络“自行猜关系”的负担。
JEPA 可显式定义跨视角、跨部件或跨状态的预测目标。
WiC 可显式定义 view/action/camera token 与 ordered transport。
可研究用 IT-RoPE 式 operator 约束 3D asset view transition。
也可用 JEPA target 检验 transport 后的 latent 是否保持对象结构。
### 13.6 显式状态与潜空间世界模型
WiC 是典型潜空间世界模型。
它的状态主要存在于视频 latent 与 attention relation 中。
相机几何是显式输入，但物体状态仍隐式。
旭涛若研究显式状态，可把 object pose、contact、visibility、topology 也变成 token。
然后比较“只加 token”“加结构化 transport”“显式 simulator state”三种层级。
这能回答关系显式化到什么程度才真正提升长期一致性。
### 13.7 控制 token 与各向异性亲和性
旭涛的各向异性亲和性研究关心关系如何随方向、类型与条件变化。
IT-RoPE 正是一个控制条件化的非对称、有序关系例子。
$T_{i\to j}$与$T_{j\to i}$互为逆，但一般不相同。
view/type gate 又让相同 transport 对不同 token 类型产生不同权重。
这与“关系显式化”的共性非常直接。
可进一步研究把 gate 从标量扩展为低秩、head-specific 或 object-specific affinity。
### 13.8 可迁移的研究假设
假设一：显式关系 token 的价值随关系组合复杂度增加。
假设二：ordered operator 比 endpoint difference 更适合非交换动作。
假设三：zero-init relation branch 能在小数据下保护预训练先验。
假设四：显式状态与隐式 transport 可能互补，而非互斥。
这些假设都可在 3D asset manipulation、camera planning 与 VLA latent dynamics 中测试。
## 14. 复现与后续实验建议
### 14.1 最小独立复现
选择公开 Wan2.2-TI2V-5B 或更小视频 DiT。
先只实现 view token，不实现 action/camera。
构造 FREE/FPP 两类可控合成轨迹。
用常量 view schedule 训练，用分段 schedule 测 zero-shot switching。
复现 6.25→37.5 的核心现象，而不是先追求 93.8。
### 14.2 ICC 架构对照
保持数据、优化步数、参数更新范围相同。
对比 Additive、AdaLN、cross-attention 与 prefix token ICC。
严格匹配新增参数量。
同时报告训练 FLOPs、显存与 tokens/s。
将性能画成 score-versus-compute，而不仅是 score-versus-data。
### 14.3 IT-RoPE 单元测试
验证$U_t^\top U_t\approx I$。
验证$T_{j\to i}=T_{i\to j}^\top$。
验证$T_{i\to j}T_{j\to k}=T_{i\to k}$。
验证同时间和静态区间 correction 为零。
验证$\lambda=0$时与 base attention 数值一致。
验证 prefix factorization 与显式 pairwise product 误差。
### 14.4 非交换合成任务
构造“先转后移”和“先移后转”的 SE(2)/SE(3) 序列。
让两条序列共享起点与终点。
Endpoint-only 理论上难区分。
Ordered transport 应产生不同 relation。
这会比 WBench 0.2–0.7 分差更直接地检验机制。
### 14.5 action-camera 独立真值
优先使用游戏引擎 telemetry。
action 读取键鼠或角色 controller。
camera 读取渲染引擎真实 extrinsics。
避免由 camera 反演 action。
再比较 action-only/camera-only/joint，尤其保留 TPP。
这样才能检验两个自由度的真正互补。
### 14.6 长时 causal 评测
报告 5、10、30、60 秒上的质量与几何漂移。
加入 loop-closure pose error。
加入 landmark re-identification。
加入 identity persistence 与 state recovery。
报告 sink/recent window 长度对显存和性能的曲线。
### 14.7 WBench judge 审计
公开 judge model、版本、prompt 与 threshold。
随机抽样做多人类标注。
报告 Cohen’s kappa 或 Krippendorff’s alpha。
对 binary switching 同时报告 transition quality 的连续分数。
给出 bootstrap confidence interval。
### 14.8 数据透明度
为每个 clip 记录 source dataset、license、hash 与近重复簇。
对 WBench 做 perceptual hash 与 embedding nearest-neighbor audit。
对 Qwen3-VL view label 抽样人工复核。
对 ViPE pose 与 engine truth 做误差分析。
对 internal switching data 单独说明来源与许可范围。
### 14.9 当前可执行性判断
现在不能按官方命令运行 WiC。
现在不能下载 WiC adapter。
现在不能复现论文 Table 1。
可做的只是独立工程重建与机制验证。
任何复现报告都应写明“非官方实现”。
## 15. 关键术语
### 15.1 Interactive video world model
随时间接收用户控制并生成未来视觉观察的视频生成模型。
### 15.2 FREE
无具身角色约束的自由相机漫游制度，通常更平滑。
### 15.3 FPP
第一人称视角，镜头运动与角色身体步态耦合。
### 15.4 TPP
第三人称视角，角色动作与外部跟随镜头是独立自由度。
### 15.5 ICC
将 view、action、camera 都表示为 token，并与视觉 latent 共享 self-attention 的条件范式。
### 15.6 view schedule
随 latent 时间变化的 FREE/FPP/TPP 类别序列，可在视频内部触发视角切换。
### 15.7 Plücker ray
用 moment 与 direction 表示三维有向直线的六维几何坐标。
### 15.8 local twist
$\mathfrak{se}(3)$中的局部六维刚体增量，包含三平移与三旋转。
### 15.9 IT-RoPE
在基础 3D-RoPE score 上加入 action/camera 有序 transport correction 的位置关系编码。
### 15.10 orthogonal step transform
每一步控制诱导的范数保持矩阵$U_t^b$。
### 15.11 ordered transport
按时间顺序乘积得到的$T_{i\to j}^b$，能保留非交换控制过程。
### 15.12 centered correction
用$T-I$而非$T$修正 attention，使静态或同时间关系修正为零。
### 15.13 prefix factorization
用$(P_i)^\top P_j$恢复 interval transport，避免显式构造全部 pairwise 矩阵。
### 15.14 sink token
causal 模型中长期保留的初始上下文 token，用作全局语义锚点。
### 15.15 recent token
causal 模型中滑动维护的近期视觉 token，用于局部连续性。
### 15.16 WBench
包含 289 cases、1,058 turns 的交互式视频世界模型评测。
## 16. 图表索引
### 16.1 本笔记十张图
| 本地文件 | 论文位置 | PDF 页 | 内容 |
|---|---|---:|---|
| `01-teaser.png` | Figure 1 / GitHub teaser | 1 | FREE、FPP、TPP 与双向切换全景 |
| `02-icc-it-rope.png` | Figure 2 / GitHub method | 3 | ICC 共享上下文与 IT-RoPE 双 transport |
| `03-emergent-view-switching.png` | Figure 3 | 4 | 6.25→37.5→93.8 视角切换 |
| `04-view-motion-dynamics.png` | Figure 4 | 5 | FPP 周期振荡与 FREE 平滑轨迹 |
| `05-action-camera-control.png` | Figure 5 | 5 | action/camera/joint 四指标 |
| `06-main-results.png` | Table 1 / GitHub table | 6 | WBench 主结果 |
| `07-icc-conditioning-ablation.png` | Figure 11 | 12 | ICC、Additive、AdaLN 与条件丰富度 |
| `08-it-rope-ablation.png` | Figure 12 | 12 | IT-RoPE 分支与 endpoint 消融 |
| `09-data-efficiency.png` | Figure 13 | 12 | 三档数据预算效率 |
| `10-wic-causal.png` | Figure 15 | 15 | causal 长序列定性结果 |
### 16.2 关键表与公式
Table 1：十个模型的五维 Navigation、Average 与 Switching。
Table 2：WiC-Causal 与双向 WiC、5B baselines 的完整对比。
Table 3：六个导航数据源比例。
Equation 1：$p_\theta(x\mid y,u)$任务分布。
Equations 4–5：统一$c$与$[c;z_\tau]$。
Equation 6：时变 view token。
Equation 7：四 slot action token。
Equation 8：camera ray/pose/twist token。
Equations 9–10：ordered transport 与 centered attention correction。
Equations 13–14：prefix factorization 与 query-key 分解。
## 17. 材料来源
### 17.1 官方入口
arXiv 页面：https://arxiv.org/abs/2610.33169 。
官方 PDF：https://arxiv.org/pdf/2610.33169 。
项目页：https://haroldchen19.github.io/WiC-site/ 。
官方 GitHub：https://github.com/aigc3d/World-in-Context 。
固定 commit：https://github.com/aigc3d/World-in-Context/commit/a985fd003458f07de5c8a537d750efbadd989aa9 。
### 17.2 本地一手材料
全文抽取：`world-in-context-staging/source-clean.md`。
官方 PDF：`world-in-context-staging/source.pdf`。
仓库快照：`/tmp/world-in-context-code-20261027`。
API、README、LICENSE 与 tree 快照位于本笔记 assets 的`sources/`。
详细文件级 provenance、10 张图的 URL 与开源审计见同目录`sources.md`。
### 17.3 最终可信度说明
论文方法、公式、数据比例与实验数值以官方 PDF 为主。
仓库状态以 commit、GitHub tree/API、README 与本地快照交叉核对。
README 下半部分只按 planned release 记录。
本文未运行任何模型、推理、训练或 benchmark。
本文未声称官方代码、权重或数据已发布。
