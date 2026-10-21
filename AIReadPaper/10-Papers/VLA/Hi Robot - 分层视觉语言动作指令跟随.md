---
type: paper-note
status: complete
domain: VLA
paper: "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models"
year: 2025
arxiv: "2502.19417"
doi: null
source: "https://arxiv.org/abs/2502.19417"
project: "https://www.pi.website/research/hirobot"
code: null
tags:
  - VLA
  - hierarchical-policy
  - instruction-following
  - human-robot-interaction
created: 2026-10-20
updated: 2026-10-20
---

# Hi Robot：分层视觉语言动作指令跟随

## 基本信息

- **论文**：Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models
- **作者**：Lucy Xiaoyang Shi、Brian Ichter、Michael Equi、Liyiming Ke、Karl Pertsch、Quan Vuong 等
- **机构**：Physical Intelligence、Stanford、UC Berkeley
- **发表**：ICML 2025，PMLR 267；arXiv [2502.19417](https://arxiv.org/abs/2502.19417)，v2，2025-07-15
- **项目页**：[Physical Intelligence — Hi Robot](https://www.pi.website/research/hirobot)
- **关键词**：hierarchical VLA、开放式指令、在线纠正、合成交互数据、π0

## 一句话结论

Hi Robot 用高层 PaliGemma VLM 每秒把复杂用户意图和现场反馈翻译为原子语言技能，再由低层 π0 VLA 以 flow matching 执行动作；在清桌、做三明治和移动购物三域中，它相对 GPT-4o 高层平均提升 **40 个百分点以上的指令准确率**，合成数据分别贡献约 **46/39 个百分点**的指令准确率/任务进度，而分层结构相对同数据 flat VLA 贡献约 **34/19 个百分点**。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

传统语言条件策略擅长“拿起杯子”这类原子指令，却难以处理“做一个不要番茄的素食三明治”“只收垃圾，不要碰餐具”，更难在动作进行中响应“那不是垃圾”“不要那个”。这类任务同时要求：理解开放式语义、根据当前图像落地指代、维护多步任务结构、把用户反馈实时改写为可执行动作，并保持精细真机控制。

论文把它类比为 System 2 与 System 1：高层慢速推理负责解释意图与分解任务，低层快速策略负责已学会技能的动作执行。研究问题是：能否用两个共享 VLM 先验的模型，通过自然语言中间接口，把复杂交互能力与高频控制解耦？

![](assets/Hi%20Robot%20-%20分层视觉语言动作指令跟随/figures/01-teaser.png)

> **图 1（论文原图）**：系统覆盖五类交互：多阶段指令、未见任务组合、现场纠正、用户约束和开放式需求。黄色块是高层生成的原子技能序列，绿色块是确认或纠错话语；图中重点不是单个抓取，而是任务执行中持续重规划。  
> **来源**：官方论文 Figure 1，第 2 页，[arXiv:2502.19417](https://arxiv.org/abs/2502.19417)

### 1.2 现有方法的瓶颈

Flat VLA 把复杂 prompt 直接映射到动作，语义规划与接触控制共享同一输出压力，长任务中容易退回训练数据的默认行为。另一类 SayCan/VLM planner 以商用大模型选择预定义技能，语义强但缺乏机器人现场分布的视觉 grounding，也不了解低层策略真实 affordance。YAY Robot、RACER 等可接受纠正，但分别受人工纠正覆盖、模拟器恢复行为或任务范围限制。

### 1.3 本文核心贡献

1. 构造 VLM 高层 + VLA 低层的层级闭环，以语言作为可解释、可人工干预的接口。
2. 从普通遥操作 demonstration 反向合成“可能导致该技能的用户 prompt、插话与机器人回复”，无需真实采集大量交互对话。
3. 在单臂、双臂和移动双臂三种平台验证复杂指令、约束与在线反馈。
4. 对比 Flat VLA、GPT-4o high-level 与 expert human oracle，并分别消融合成数据和层级结构。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

标准低层策略在时刻 $t$ 接收多相机图像、机器人状态和语言：

$$
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t],
$$

输出长度 $H$ 的动作块

$$
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}],\qquad p(A_t\mid o_t).
$$

Hi Robot 将其分解为高层与低层：

$$
\hat\ell_t\sim\phi(\hat\ell_t\mid I_t^1,\ldots,I_t^n,\ell_t),
$$

$$
A_t\sim p_{\mathrm{lo}}(A_t\mid I_t^1,\ldots,I_t^n,\hat\ell_t,q_t).
$$

$\ell_t$ 是开放式用户输入，$\hat\ell_t$ 是低层已知的原子命令，如“pick up one piece of lettuce”。高层还可产生机器人话语 $u_t$；系统先用 TTS 播放，再从传给低层的命令中移除话语段。

### 2.2 运行假设

方法假设复杂任务能被分解为低层数据已有或可组合的短技能，且当前图像足以决定下一步。系统没有持久记忆，也没有显式技能完成检测：高层固定每 1 秒重算一次，或收到用户新话语时立即触发。用户可在临时插话完成后示意恢复上一任务，但任务栈的长期管理仍较弱。

---

## 3. 核心方法

### 3.1 总体分层架构

![](assets/Hi%20Robot%20-%20分层视觉语言动作指令跟随/figures/02-architecture.png)

> **图 2（论文原图）**：高层 VLM 读取用户 prompt/插话与 base、wrist 图像，输出机器人话语和低层语言命令；低层 VLA 再结合相同图像与关节状态输出连续动作。语言中间层既是任务抽象，也是调试与安全审计接口。  
> **来源**：官方论文 Figure 2，第 4 页，[arXiv:2502.19417](https://arxiv.org/abs/2502.19417)

两层均从 PaliGemma-3B 初始化。高层是标准自回归 VLM，输出文本；低层是 π0，在 PaliGemma 上增加 flow-matching action expert，输出连续动作块。架构模块化意味着低层可替换为其他语言条件策略，但实验全部使用 π0。

高层不是一次性生成完整计划，而是反复看最新图像并生成下一原子动作。这使“that’s not trash”有视觉语境：模型可以看到当前夹持的碗，决定张开夹爪、放回或道歉，而非仅在文本层解释纠正。

### 3.2 数据收集与技能切分

先通过遥操作收集 $D_{\mathrm{demo}}$，每条轨迹只有粗粒度总目标，如“make a sandwich”。人工把长轨迹切成 1–3 秒技能并标注 $\hat\ell_t$，同时从原始动作启发式提取“左移一点”“张开夹爪”等基本修正原语，得到

$$
D_{\mathrm{labeled}}=\{(\hat\ell_t,I_t^1,\ldots,I_t^n)\}.
$$

这一切分决定低层可执行词表，也是高层学习机器人 affordance 的主要来源。

### 3.3 反向合成互动数据

论文不要求采集真实用户在每个动作前说过什么，而是给生成 VLM $p_{\mathrm{gen}}$ 当前图像、目标技能、历史技能和任务说明，让它想象合理的前置用户话语 $\ell_t$ 与机器人回复 $u_t$：

$$
(\ell_t,u_t)\sim p_{\mathrm{gen}}(\cdot\mid I_t^{1:n},\hat\ell_{0:t},P).
$$

合成 prompt 按场景类型组织，包括 negative task、situated correction、specific constraint；机器人回复分为确认、澄清和错误处理。把历史技能 $\hat\ell_{0:t-1}$ 放入条件可防止生成孤立、与任务进程冲突的请求。例如已放生菜和番茄后，合成指令应请求后续配料，而不是重复第一步。

这种反向标注的关键价值是：低层动作仍来自真实 demonstration，只有“为何此时做这个动作”的语言解释是合成的，因此不会让生成模型伪造机器人动力学。

### 3.4 训练目标

高层在 $D_{\mathrm{syn}}\cup D_{\mathrm{labeled}}$ 上做 next-token cross entropy：

$$
\mathcal L_{\mathrm{high}}=-\sum_k\log \phi_\theta(x_k\mid I^{1:n},\ell,x_{<k}).
$$

低层在 $D_{\mathrm{labeled}}\cup D_{\mathrm{demo}}$ 上沿用 π0 flow matching。对噪声动作与真实动作插值得到 $A^\tau$，动作专家学习速度场，使采样时从噪声积分到动作块。论文没有重新给出完整 flow 公式，但训练机制与 π0 一致。

两层全量解冻。高层优化使用 AdamW，$\beta_1=0.9,\beta_2=0.95$、无 weight decay，梯度裁剪 1，EMA 0.999；前 1,000 steps warmup，随后学习率 $10^{-5}$，batch size 512。高层训练约需 **8×H100、2 小时**；低层时间依数据规模而变，论文未披露统一总 GPU 小时。

### 3.5 实时推理与硬件

语音输入由 Whisper large-v2 本地转写，Cartesia API 负责 TTS。系统使用 1–2 张 RTX 4090。低层延迟分解为图像编码 14 ms、观测处理 32 ms、10 次动作预测 27 ms，总 on-board 73 ms、经 Wi-Fi off-board 86 ms；约 10 Hz 重规划，配合 action chunking 控制器可在 50 Hz 执行动作。

高层单步在 RTX 4090 上 prefill 47 ms、decode 13.2 ms；H100 为 17.3/5.7 ms。高层每秒运行一次，因此不是主要实时瓶颈。

### 3.6 三种机器人平台

- **UR5e**：单 6-DoF 臂、平行夹爪、腕部+肩部相机，7 维状态/动作。
- **Bimanual ARX**：两只 6-DoF 臂、两腕部+一 base 相机，14 维状态/动作。
- **Mobile ARX**：双臂加非完整移动底盘，14 维状态、16 维动作。

这三者分别承担 table bussing、sandwich making、grocery shopping，验证层级接口跨 embodiment，而非只在一台平台上成立。

---

## 4. 数据集与实验设置

### 4.1 三个任务域

**Table Bussing** 要把餐具/碗盘放入 bussing bin、垃圾放入垃圾桶，并处理“只收垃圾”“只收餐具”“收黄色物体”等改变默认目标的指令。难点包括遮挡、物体分类、边缘抓盘、借助盘子倾倒垃圾。

**Sandwich Making** 涉及面包与最多六种配料，要求处理“素食”“对腌黄瓜过敏”“不要番茄”及执行中“够了”等约束；同时需要轻柔抓取与精确堆叠可变形食材。

**Grocery Shopping** 由移动双臂机器人从货架取指定组合、放入篮子并送到桌面；prompt 可是具体商品，也可以是“电影之夜”“来点甜的”“拿些饮料”等开放语义，过程中还能追加 KitKat。

### 4.2 Baseline

1. **Expert human high level**：人手输入最可能成功的低层技能，估计低层上限。
2. **GPT-4o high-level**：同一层级接口，prompt 提供任务说明与高频技能候选，GPT-4o 每 2 秒从列表选择；低层与 Hi Robot 相同。
3. **Flat VLA**：π0 直接执行复杂 prompt，无高层、无合成数据。
4. **Flat VLA + synthetic data**：用于隔离“数据更多”与“层级结构”的贡献。
5. **Hi Robot without synthetic data**：只用人工技能标注训练高层，隔离合成语言覆盖的贡献。

### 4.3 指标与协议

每种方法、每个任务 **20 次 trial**，由不知道方法身份的人类评测员打分。

- **Instruction Accuracy（IA）**：每次高层输出是否同时符合用户意图与当前图像，trial 内取正确预测比例。Flat VLA 没有文本中间量，由评测员根据行为推断意图。
- **Task Progress（TP）**：正确放置或达到目标配置的对象比例，适合长任务的部分完成度。

IA 测“想做对没有”，TP 同时混入低层执行能力。两者并列可以把高层错误与动作失败区分开。

---

## 5. 实验结果

### 5.1 与 Flat VLA、GPT-4o 的定量比较

![](assets/Hi%20Robot%20-%20分层视觉语言动作指令跟随/figures/03-results.png)

> **图 5（论文原图）**：Hi Robot 在三个任务域的 IA 与 TP 都高于 Flat VLA 和 GPT-4o high-level，并接近 expert human high-level 的虚线上限。论文只提供柱状图而非精确数值表；按柱高读取，平均 IA 约为 **34% / 28% / 73%**，平均 TP 约为 **42% / 61% / 78%**（依次为 Flat、GPT-4o、Hi Robot），应视为图读近似值。  
> **来源**：官方论文 Figure 5，第 7 页，[arXiv:2502.19417](https://arxiv.org/abs/2502.19417)

论文明确报告 Hi Robot 的平均 IA 比 GPT-4o 高 **40 个百分点以上**。分域按图读近似值：

| 任务 | Flat IA | GPT-4o IA | Hi Robot IA | Flat TP | GPT-4o TP | Hi Robot TP |
|---|---:|---:|---:|---:|---:|---:|
| Table Bussing | 约 33% | 约 31% | 约 71% | 约 59% | 约 61% | 约 74% |
| Sandwich Making | 约 33% | 约 12% | 约 80% | 约 40% | 约 53% | 约 78% |
| Grocery Shopping | 约 36% | 约 40% | 约 69% | 约 27% | 约 69% | 约 82% |

GPT-4o 失败并非单纯语言能力不足，而是缺少机器人分布 grounding：会把所有东西反复叫成 plate/spoon、输出不存在的“Bermuda triangle”、夹爪仍占用时切到新物体。Flat VLA 则常回到数据中“清走所有物品、加入所有配料”的默认轨迹，不响应中途反馈。

### 5.2 定性行为与在线纠正

Table Bussing 中用户说“that’s not trash”后，Hi Robot 会先道歉并张开夹爪，而不是继续扔掉碗；做三明治时能把“我乳糖不耐受”落实为跳过 cheese；购物中可在已有清单上追加 KitKat。expert human high-level 下低层几乎无误，说明主要瓶颈集中在高层意图解释而不是基本抓放。

### 5.3 合成数据消融

Figure 7 给出明确 gap：完整 Hi Robot 相对无合成数据版本，平均 **IA +46 个百分点、TP +39 个百分点**。无合成数据模型能识别图像中的对象，却容易忽略“不要”“过敏”“只收某类”这类偏离默认演示分布的语言约束。合成数据真正增加的是反事实和组合语言覆盖，而不是动作样本量。

### 5.4 层级结构消融

Figure 8 比较相同合成数据下的 Flat VLA 与 Hi Robot：层级模型平均 **IA +34 个百分点、TP +19 个百分点**。这排除了“Hi Robot 只是因为看了更多文本”的解释。高层每一步重新读取 prompt 与图像，使局部动作分布只需学习短技能；Flat VLA 必须在连续动作隐状态中同时维护长期约束，容易遗忘中途澄清。

### 5.5 泛化与失败案例

附录明确列出三类失败：

1. **高层长上下文失败**：当前系统无记忆，长任务中不能稳定维护很早的约束。
2. **低层暂时忽略指令**：例如明知乳糖不耐受，机器人靠近 cheese 时仍可能因“近物体偏置”去抓 cheese。
3. **误差累积/OOD 恢复不足**：掉落物体后缺少恢复 demonstration，后续任务状态偏离训练分布。

作者建议使用更强 instruction-following/long-context 模型、针对边缘案例的 adversarial synthetic data，以及专门收集失败恢复数据。

### 5.6 实验是否支撑主张

实验较有说服力地支撑“层级 + 合成 situated language 优于 flat 与通用 API VLM”这一主张：有三类硬件、三种任务、盲评、oracle 和正交消融。边界也明显：只有三个任务域，每域 20 trials；高层是按任务分别生成数据和训练，而不是一个统一跨域高层；开放世界主要指 prompt 组合和反馈，不等于任意新技能。

---

## 6. 与相关工作的关系

Hi Robot 与 SayCan 类系统同样把高层语义与低层执行拆开，但高层接收实时多视角图像，并通过机器人轨迹反向生成的数据做 in-domain 微调。与 RT-H 相比，它强调开放 prompt、语境纠正和 verbal response；与 YAY Robot 相比，合成数据突破了人工纠正集合；与 π0.5 的统一模型不同，Hi Robot 保留两个独立 PaliGemma 模型，接口更清晰但能力耦合较弱。

---

## 7. 局限与批判性评价

1. **两层能力未闭环对齐**：高层只从标注示例间接知道低层会什么，不观察技能成功概率，也不基于执行结果学习。
2. **固定 1 秒高层节拍粗糙**：技能可能未完成就换令，也可能完成后等待；真正的 option termination 或异步事件机制会更合理。
3. **任务专用高层削弱“开放式”含义**：附录说明为清楚评测，每个任务分别生成 $D_{\mathrm{syn}}$、训练高层；跨域统一能力尚未验证。
4. **合成数据依赖 prompt engineering**：生成类别与提示模板决定可见交互边界，遗漏的约束类型不会凭空出现。
5. **安全机制未展开**：语言接口可解释，但论文没有拒绝危险请求、碰撞检测、动作 veto 或 uncertainty gating。
6. **个人判断**：论文最重要的贡献不是两个模型串联，而是“从已发生的技能反推可能的用户意图”这一数据构造。它把昂贵的在线人机对话采集转化为可扩展离线重标注，但前提是技能库本身已足够可靠。

---

## 8. 复现与实践建议

### 8.1 最小复现路径

先选一个含 10–20 个原子技能的桌面任务：人工切分真实 demonstration；用 VLM 生成正向请求、否定约束、现场纠正三类 prompt；训练 PaliGemma 高层输出技能名，固定低层策略。必须同时评估 IA 与 TP，并加入同合成数据的 Flat VLA，才能证明层级贡献。

### 8.2 复现成本

- **高层训练**：论文给出约 8×H100、2 小时，属于中等成本。
- **低层训练**：需要 π0 级 VLA 和真实 demonstration，论文未给统一时长，是主要训练门槛。
- **硬件**：三种平台、最多三相机，部署用 1–2×RTX 4090；语音还依赖麦克风、Whisper 与外部 TTS API。
- **标注**：长轨迹切分成 1–3 秒技能需要人工定义一致词表；这比合成 prompt 更影响系统上限。
- **代码/数据**：PDF 未给出完整公开代码与数据链接，难以一比一复刻。

### 8.3 工程建议

实际部署应为每个技能加入完成检测、超时和失败返回码，让高层接收结构化执行反馈；保存用户约束到显式短期记忆；在把 $\hat\ell_t$ 交给低层前做允许技能校验；将 TTS 回复与动作计划分字段输出，避免文本解析歧义。

---

## 9. 个人启发与后续问题

自然语言中间层既能提供语义抽象，也能成为 human-in-the-loop 控制面。下一步值得研究：单模型能否通过不同 decode mode 同时承担两层；高层能否学习低层的成功率模型；合成 prompt 能否自动发现低层盲区；以及把用户纠正变成在线偏好学习，而不是仅触发一次重规划。

---

## 10. 材料来源

- 官方论文页：[https://arxiv.org/abs/2502.19417](https://arxiv.org/abs/2502.19417)
- 官方 PDF：`assets/Hi Robot - 分层视觉语言动作指令跟随/sources/paper.pdf`
- 项目页：[https://www.pi.website/research/hirobot](https://www.pi.website/research/hirobot)
- 本地来源清单：`assets/Hi Robot - 分层视觉语言动作指令跟随/sources.md`
- 柱状图数值若原文未给表格，本文均标成“按图读取的近似值”；明确 gap、trial 数、延迟与训练配置来自官方 PDF 正文或附录。
