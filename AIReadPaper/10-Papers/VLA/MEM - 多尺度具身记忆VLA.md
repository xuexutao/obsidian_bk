---
type: paper-note
status: complete
domain: VLA
paper: "MEM: Multi-Scale Embodied Memory for Vision Language Action Models"
year: 2026
arxiv: 2603.03596
doi: 10.48550/arXiv.2603.03596
source: https://arxiv.org/abs/2603.03596
project: https://pi.website/research/memory
code: null
tags:
  - VLA
  - Memory
  - Video-Encoder
  - Long-Horizon
  - π0.6
  - Physical-Intelligence
created: 2026-10-20
updated: 2026-10-20
---

# MEM：多尺度具身记忆 VLA

## 基本信息

- **正式标题**：MEM: Multi-Scale Embodied Memory for Vision Language Action Models
- **作者**：Marcel Torne*、Karl Pertsch*、Homer Walke、Kyle Vedder、Suraj Nair、Brian Ichter、Allen Z. Ren、Haohuan Wang、Jiaming Tang、Kyle Stachowicz、Karan Dhabalia、Michael Equi、Quan Vuong、Jost Tobias Springenberg、Sergey Levine、Chelsea Finn、Danny Driess（* 共同一作）
- **机构**：Physical Intelligence、Stanford University、UC Berkeley、MIT
- **版本**：arXiv:2603.03596v2，2026-03-08
- **项目页**：<https://pi.website/research/memory>
- **本地原文**：`assets/MEM - 多尺度具身记忆VLA/sources/paper.pdf`

## 一句话结论

**MEM 认为机器人记忆没有“一种模态通吃”的方案：它把最近数秒到 54 秒的稠密视觉/本体状态压进短期 video memory，把最长约 15 分钟的任务进展压成模型自行更新的长期 language memory，并集成到 π0.6 的高低层 VLA；这种多尺度分工不仅显著优于无记忆、平均池化视觉记忆和纯 proprioception memory，还能在失败后在线换抓法/换开门方向，同时不牺牲普通灵巧操作性能。**

---

## 1. 研究背景与问题定义

### 1.1 为什么单一记忆表示不够

长程机器人任务同时需要两种性质相反的信息。最近几秒的图像要保留精确几何：手臂遮住目标前它在哪里、筷子刚才从哪个高度抓失败、冰箱门刚才向哪边拉不动。这类信息密、连续、难以用文字无损概括。十几分钟前的历史却往往只剩少数语义位：配方里哪些材料已取出、哪块台面已擦、盘子是否已加洗洁精、柜门是否还没关；继续保留所有像素既昂贵又会淹没有效信息。

以往方案通常选择一种折中：拼接若干历史帧，计算随上下文迅速增长；平均池化或只留关键帧，丢失时序和细节；只保留 proprioception，无法记住环境；只保留语言，难以处理精细几何与动态。MEM 的问题定义因此是：**在真实机器人几百毫秒延迟预算内，如何同时保存高带宽短期记忆和低带宽长期记忆？**

![](assets/MEM%20-%20多尺度具身记忆VLA/figures/01-teaser.png)

> **图 1：MEM 的能力总览。** 输入任务 prompt 后，系统以短期 video memory 支撑遮挡处理、计数、计时和失败后策略调整，以长期 text memory 记录配方、清洁步骤等语义状态；实验覆盖最长约 15 分钟的 recipe setup、grilled cheese、clean kitchen，以及多种 partial-observability/in-context adaptation 任务。  
> **来源**：论文 Figure 1，第 1 页；官方 PDF：<https://arxiv.org/pdf/2603.03596>

### 1.2 核心贡献

1. 提出 **Multi-Scale Embodied Memory**：短期用稠密视频、长期用压缩语言摘要，两种记忆分别服务低层动作与高层任务状态。
2. 设计无新增可学习参数的 **space-time separable video encoder**：每四层加入一次因果时间注意力，只把当前帧的时序融合 token 交给 VLA backbone，使 token 数仍等同单帧。
3. 让高层策略递归生成更新后的语言记忆 $m_{t+1}$，而不是无限拼接历史 subtask，主动压缩已完成、失败或不再相关的信息。
4. 将其集成到 Gemma 3 4B + SigLIP 400M 的 π0.6，并证明预训练阶段就学习“如何使用记忆”比只在下游 post-training 临时加历史更有效。
5. 展示记忆带来的新行为：15 分钟长程任务、遮挡/计数/计时、失败后的 in-context strategy adaptation；同时在不需要记忆的操作任务上匹配原 π0.6，避免常见 causal confusion 退化。

---

## 2. 任务定义与输入输出

### 2.1 原始目标

给定自然语言目标 $g$、当前及历史观测 $o_{t-T:t}$，策略输出长度为 $H$ 的连续动作 chunk：

$$
\pi(a_{t:t+H}\mid o_{t-T:t},g).
$$

若把十几分钟内的多相机图像全部输入，计算和 KV context 均不可接受。MEM 将预测因子化为高层与低层：

$$
\begin{aligned}
&\pi(a_{t:t+H},\ell_{t+1},m_{t+1}\mid o_{t-T:t},m_t,g)\\
&\quad\approx
\pi_{\mathrm{LL}}(a_{t:t+H}\mid o_{t-K:t},\ell_{t+1},g)\,
\pi_{\mathrm{HL}}(\ell_{t+1},m_{t+1}\mid o_t,m_t,g),
\end{aligned}
$$

其中 $K\ll T$。高层策略 $\pi_{\mathrm{HL}}$ 读取当前观测、目标和上一条语言记忆 $m_t$，同时输出下一子任务 $\ell_{t+1}$ 与更新后的记忆 $m_{t+1}$；低层策略 $\pi_{\mathrm{LL}}$ 读取较短但稠密的视觉/本体历史与子任务，输出动作。长期信息走低带宽文字，短期信息走高带宽视频。

### 2.2 输入与输出接口

- **高层输入**：全局任务 $g$、当前视觉观测 $o_t$、上一条 memory summary $m_t$；
- **高层输出**：下一条原子语言指令 $\ell_{t+1}$、更新摘要 $m_{t+1}$；
- **低层输入**：$\ell_{t+1}$、任务 $g$、最近 $K$ 个多相机帧及 $K$ 个连续本体状态 token；
- **低层输出**：50 步连续动作 chunk（沿用 π0.6 action expert），并同时保留 FAST 离散动作训练目标；
- **执行假设**：机器人在动作 chunk 异步执行时使用 inference-time RTC 或 training-time RTC，避免记忆计算延迟造成轨迹不连续。

### 2.3 两个时间尺度并非硬边界

video memory 在预训练中使用 6 个观测（5 个历史帧 + 当前帧），相邻观测相隔 1 秒，覆盖约 5 秒；post-training 可扩展到 18 帧、最长 54 秒。language memory 则跨 subtask 递归携带，任务实验最长约 15 分钟。短期与长期不是按固定秒数截断，而是按“需要几何细节还是只需语义状态”分工。

---

## 3. 核心方法

### 3.1 总体架构：高层语言记忆 + 低层视频记忆

MEM 被实例化为 **π0.6-MEM**。高层 VLM 负责从 prompt、当前画面和旧摘要生成新摘要及下一 subtask；低层 VLA 接收 subtask、稠密短视频和本体状态，经 860M action expert 生成连续动作。两者均以 SigLIP 400M + Gemma 3 4B 为预训练骨架。

![](assets/MEM%20-%20多尺度具身记忆VLA/figures/02-architecture.png)

> **图 2：π0.6-MEM 架构。** 左侧高层策略递归更新 $m_t\rightarrow m_{t+1}$ 并生成 subtask；右侧低层策略用 video memory encoder 把多帧观测压成与单帧等长的视觉 token，再由 860M flow-matching action expert 输出连续动作。  
> **来源**：论文 Figure 2，第 4 页；官方 PDF：<https://arxiv.org/pdf/2603.03596>

### 3.2 长期语言记忆：让模型学会“记什么、忘什么”

语言记忆 $m_t$ 是此前语义事件的最小充分摘要。训练标签由离线 LLM 根据整段 subtask 序列及每个 subtask 成功/失败标记生成：LLM 被要求仅保留未来仍相关的信息，并在可能时合并细节。例如，不必记住三个碗的每种颜色，只需记住“三个碗已放入右上柜”。

高层模型学习：

$$
\mathcal{L}_{\mathrm{HL}}=-\log p_\theta(\ell_{t+1},m_{t+1}\mid o_t,m_t,g).
$$

重要设计是**递归更新**而非 append-only history。若模型连续三次输出 “pick up bowl” 才成功，naive history 会形成训练中罕见的重复序列，导致 train–test distribution shift；MEM 在失败期间可以保持 $m_{t+1}=m_t$，成功后才写入“已拿起碗”。这既压缩 token，也过滤失败尝试和无关细节。

风险也很明确：摘要是有损状态。若高层模型错误地删掉一个以后才重要的事实，低层视频窗口又已滑过，就无法恢复；论文没有引入外部可检索日志或事实一致性校验。

### 3.3 短期视频编码器：时空可分注意力

设第 $l-1$ 层、空间 patch $p$、时间 $t\in[-K,0]$ 的表示为 $z^{l-1}_{p,t}$。先加固定正弦时间编码：

$$
\hat z^{l-1}_{p,t}=z^{l-1}_{p,t}+e(t),\qquad e(0)=0.
$$

边界条件 $e(0)=0$ 保证 $K=1$ 时与原始单图 ViT 完全一致，便于无损继承预训练视觉能力。查询、键、值继续复用原 ViT 参数：

$$
q^{l,a}_{p,t}=W^{l,a}_{Q}\operatorname{LN}(\hat z^{l-1}_{p,t}),\quad
k^{l,a}_{p,t}=W^{l,a}_{K}\operatorname{LN}(\hat z^{l-1}_{p,t}),\quad
v^{l,a}_{p,t}=W^{l,a}_{V}\operatorname{LN}(\hat z^{l-1}_{p,t}).
$$

标准 ViT 每层在同一帧所有 patch 间做双向空间注意力。MEM 每第 4 层额外加入**同一 patch 跨时间的因果注意力**，再做空间注意力；过去可影响当前，当前不能泄漏给过去。若每帧有 $n$ 个 patch、$K$ 个时间步，朴素联合注意力复杂度为

$$
O(n^2K^2),
$$

时空可分后变为

$$
O(Kn^2+nK^2).
$$

上层只保留当前时刻 $t=0$ 的 patch token，丢弃历史帧 token；但当前 token 已经通过多层时间注意力吸收历史。因此送入 Gemma backbone 的视觉 token 数与单帧 VLA 相同，主要额外开销被限制在 SigLIP 内。

### 3.4 为什么不是“每帧编码后平均池化”

Pool Memory 先独立编码每帧，再把历史 frame embedding 做平均；它便宜，却无法在压缩前让具体 patch 跨时关联，时间顺序也被平均操作抹去。MEM 在 ViT 中间层反复交织空间和时间注意力，能先识别“同一位置/对象怎样变化”，再把信息写入当前 token。Figure 9 显示，即使两者都只在 post-training 引入，MEM encoder 仍优于 Pool Memory，说明收益不只来自更多预训练数据。

### 3.5 连续本体状态压缩

π0.6 原本把机器人状态离散成文本 token；若每个历史时刻都文本化，会迅速占满上下文。MEM 对每个 proprioceptive state 做线性投影：

$$
s_t=W_s q_t+b_s,
$$

长度为 $K$ 的历史只产生 $K$ 个 state token。视觉历史与状态历史同步；这种表示也被后续 π0.7 采用。

### 3.6 动作专家与训练隔离

π0.6-MEM 同时训练 FAST token prediction 与 860M flow-matching action expert。对真实动作 $a_1$、噪声 $a_0\sim\mathcal N(0,I)$、流时间 $\tau$，连续动作目标可写为

$$
a_\tau=(1-\tau)a_0+\tau a_1,
$$

$$
\mathcal L_{\mathrm{FM}}=
\mathbb E\left\|v_\theta(a_\tau,\tau,c)-(a_1-a_0)\right\|_2^2.
$$

按照 Knowledge Insulation recipe，action expert 的梯度不回传到 VLM backbone；这样高速动作学习不会破坏语言记忆和视觉语义能力。输入分辨率为每相机 $448\times448$，最多四路相机。

### 3.7 预训练：先学会使用记忆，再拉长窗口

预训练混合包括遥操作机器人示范、policy rollout、人类纠正、vision-language 任务和 video-language/video captioning。预训练窗口只有 6 帧、约 5 秒，但 post-training 能扩展到 18 帧/54 秒。作者的解释是，多样视频预训练让编码器先学会“从历史提取动态信息”，下游只需扩大时间跨度；只把 memory module 接到已预训练的无记忆 π0.6 上，虽然机器人数据相同，却明显更差。

### 3.8 推理与延迟

Figure 3 在单张 NVIDIA H100、四相机输入上比较帧数与延迟：逐帧编码并把所有 token 送入 backbone 很快越过约 300 ms 的 real-time barrier；MEM 在最多 16 帧的曲线上仍保持在该阈值下。论文未在正文给出每个点的精确毫秒数，因此不从图中伪造具体延迟；可确认的结论是 token 压缩使多帧上下文保持在作者采用的 RTC 实时预算内。

---

## 4. 数据集与实验设置

### 4.1 长程任务

**Recipe setup**：prompt 指定 6–7 个食材/器具、初始柜体位置与目标位置；训练覆盖 42 个（附录称 40+）recipes 和多厨房，测试 5 个 recipes，在未见厨房和未见物体上评估。每正确取出并放到指定位置得 1 分，还需记得关闭抽屉、柜门和冰箱。

**Clean kitchen**：擦台面、纸巾擦干并丢弃、食物放冰箱、餐具入柜、盘子加清洁剂后清洗正反面并放置沥水架。平均约 8 个 subtask，每完成一个得 1 分。两项任务每个 policy/task 或 recipe 评测 10 次，图中报告均值和标准误。

**Grilled cheese** 也用于分析记忆：组装三明治、每面烹饪 30 秒到 3 分钟、翻面并装盘；每个阶段计分。它同时要求计时、状态跟踪和精细操作。

### 4.2 In-context adaptation

- **Pick Up Chopstick**：训练时桌高位于范围上半段；测试放在最低桌高。机器人若按旧高度抓会 miss。发生误抓后由人介入示范正确高度，训练时保留失败历史。成功要求拿起并放入盒中（两步都完成）。
- **Open Refrigerator**：门铰链没有明显视觉线索。数据包含错误方向尝试及随后纠正；评测要求不超过 4 次 grasp 打开，以排除无限随机尝试。

两项都用同一纠正数据比较有/无 memory 的 π0.6，隔离“记住刚才失败策略”的作用。

### 4.3 核心记忆能力任务

| 任务 | 必须记住的信息 | 评分要点 |
|---|---|---|
| Swap 3 Mugs | 哪些杯子已放过咖啡机 | 每个未重复、未倾倒的杯子计分 |
| Find Object | 人把物体放进四抽屉中的哪个 | 不开错抽屉并取出 |
| Unpack Groceries | 袋中是否仍有被遮挡物体 | 全部取出并回 home |
| Scoop Coffee | 已放几勺 | 恰好两勺且盖回盖子 |
| Grilled Cheese | 阶段与烹饪时间 | 组装、两面计时、翻面、装盘 |
| Window Cleaning | 哪些步骤/区域已完成 | 全流程且无漏擦区域 |

另用 Table Bussing、Shirt Folding、Clean Up Counter、Make Bed、Dishes to Sink、Batch Folding、Box Building 测试加入记忆后是否损害本来不依赖记忆的灵巧任务。

### 4.4 Baseline

- **π0.6 / No Memory**：只看当前观测；
- **Pool Memory**：每帧由单帧 ViT 独立编码，历史 frame embedding 平均池化为一个 memory token；当前帧单独输入；
- **Proprio Memory**：只输入低维机器人状态历史；
- **Only Video Memory / Only Text Memory**：长程任务的组件消融；
- **Naive Text + Video Memory**：不生成压缩摘要，直接拼接所有历史 subtask，直到上下文上限；
- **MEM-Posttrain-Only**：从同一 π0.6 checkpoint 出发，但只在目标任务后训练时引入 video memory；
- **π0.6-MEM / Ours**：完整多尺度记忆并在多样数据上预训练。

---

## 5. 实验结果

### 5.1 长程任务：两种记忆缺一不可

Figure 6 显示，无记忆 π0.6 在 Recipe Setup 和 Clean Kitchen 上进展很低；Only Video Memory 能处理局部遮挡/动作持续时间，却无法长期记录配方步骤和未关门；Only Text Memory 能记录高层进度，却缺少清洗、擦拭和局部纠错所需的视觉动态。完整 π0.6-MEM 的平均 task progress 最高。

Naive Text + Video Memory 也明显落后于压缩摘要。其失败不是 token 长度这一单一因素：训练示范通常一次完成 subtask，而部署时会重复失败，直接拼接会产生训练中罕见的“pick up bowl”重复序列；MEM 只在状态真正改变后更新摘要，减小分布偏移。

论文的柱状图没有标注每项精确数字，因此这里不把视觉读数伪装成表格真值；可复核的严格结论是完整模型在两个长程任务和平均分上均最佳，且 video-only、text-only、naive-history 都不能替代双记忆。

### 5.2 失败后的在线策略调整

Figure 7 给出两项明确增量：

- **Open Fridge**：加入记忆后成功率提升 **11 个百分点**；模型能记住刚才尝试的开门方向并换方向。
- **Pick Up Chopstick**：加入记忆后成功率提升 **62 个百分点**；模型看到误抓历史后调整抓取高度。

这不是传统 test-time gradient update，而是上下文条件行为：参数不变，策略从近期失败视频中推断“当前策略已无效”。无记忆模型即便在同一纠正数据上训练，也无法判断本 episode 中哪个策略已经试过。

### 5.3 与记忆 baseline 的对比

![](assets/MEM%20-%20多尺度具身记忆VLA/figures/03-ablation.png)

> **图 3：六项核心记忆任务的 baseline 对比。** No Memory 在 Find Object 接近四选一随机水平（25%），在 Scoop Coffee 接近二选一/计数不足水平（50%）；Pool Memory 在多杯顺序和袋内剩余物任务上受平均压缩限制；Proprio Memory 在需要环境状态的 Find Object 上失效。只有 MEM 在六项任务上都保持较高 task progress，平均约为图中 70%+。  
> **来源**：论文 Figure 8，第 8 页；官方 PDF：<https://arxiv.org/pdf/2603.03596>

从图中可读的趋势（仅作图示近似，不冒充作者表格数字）是：MEM 在 Swap 3 Mugs、Find Object、Unpack Groceries、Scoop Coffee、Grilled Cheese、Window Cleaning 上分别约为 77%、90%、74%、80%、56%、65%，平均约 74%；其余 baseline 没有一个能跨任务稳定。最强差异出现在 Find Object 和 Window Cleaning，说明像素级环境记忆不能被关节状态替代。

### 5.4 预训练记忆能力的消融

Figure 9 比较 Ours 与 Ours (post-train only)。二者都从具有同样多样机器人知识的 π0.6 路线出发，差别在于 video encoder 是否在预训练阶段见过机器人和非机器人视频。完整 MEM 在六项任务上总体更强，说明“有历史输入接口”不等于“知道怎样从历史提取有用信息”。尤其值得注意的是，预训练只有约 5 秒窗口，后训练却能拓展到最长 54 秒，体现了 temporal position encoding 和视频预训练的外推能力。

### 5.5 不需要记忆的任务没有退化

Figure 10 在 Table Bussing、Shirt Folding、Clean Up Counter、Make Bed、Dishes to Sink、Batch Folding、Box Building 上对比 π0.6 和 MEM，包括 out-of-the-box 与 fine-tuning 设置。MEM 总体匹配 state-of-the-art π0.6，没有出现“看历史就复制旧动作”的明显 causal confusion。作者将其归因于预训练混合包含不同质量、速度、控制频率的机器人 episode 以及互联网视频，迫使模型学习真实动态而非简单复读过去动作。

### 5.6 失败模式

1. **语言摘要遗忘**：一旦高层错误压缩或删除关键信息，视觉窗口过后无法恢复；论文没有 external memory audit。
2. **naive history 分布偏移**：失败造成重复 subtask，训练示范却近乎最优，导致长任务滚雪球式偏离。
3. **过度压缩视觉**：只留平均 token 的 Pool Memory 在多对象身份与顺序上失败；这类任务需要结构化时序关联。
4. **只记机器人自己不够**：Proprio Memory 适合运动状态，却不知道物体在哪个抽屉、袋中还剩什么。
5. **短期窗口仍有限**：54 秒之外的细粒度视觉事件若无法准确语言化就会丢失。
6. **评测次数较少**：长程任务每条件 10 次，置信区间较宽；许多结果仅图示、未提供精确表格。

---

## 6. 与 π 系列及相关工作的关系

### 6.1 从 π0 到 π0.6-MEM

π0 的核心是 PaliGemma VLM + flow-matching action expert；π0.5 增加开放世界泛化和高层子任务；Knowledge Insulation 把 FAST/语言学习与连续动作专家隔离；π0.6 进一步吸收部署数据与纠正。MEM 在 π0.6 上补上时间轴：以前的策略主要把当前观测映射到动作，MEM 则显式携带近期视觉动态和长期任务状态。

### 6.2 对 π0.7 的直接影响

π0.7 明确复用 MEM 的 history vision encoder：最多四路相机，每路最多 6 个 history frames，1 秒 stride，并把连续 proprioception 投影为 backbone token；同时沿用 860M action expert 和 RTC。区别在于 π0.7 把 memory 变成统一 generalist 的默认能力，并进一步加入 subgoal images、质量/速度/错误 metadata 和多源数据条件化。MEM 证明“怎样记”，π0.7 则把“记忆 + 可操控提示 + 异构数据蒸馏”组合为开箱通用模型。

### 6.3 与其他记忆路线

- **关键帧/检索记忆**适合长时间跨度，但检索或抽帧可能错过连续动态；MEM 的 language memory 更压缩，video memory 更稠密。
- **point track / proprio history**计算便宜，却只能保留预先指定的几何变量；MEM 的视觉 token 更通用。
- **单一长上下文 VLM**概念上简单，但多相机高频像素使计算无法满足真实控制延迟；MEM 用模态分工规避这一点。
- **显式状态机**可解释、可靠，却需要任务工程；MEM 的自然语言状态可跨任务复用，但一致性和可验证性较弱。

---

## 7. 局限与批判性评价

### 7.1 方法边界

- 语言记忆是一条模型自生成的自由文本，缺乏结构化 schema、置信度、事实校验与回滚机制；长程错误可能永久累积。
- 长期 memory 仍限于单 episode 的约 15 分钟。作者也把跨周、月、年部署记忆和持续学习列为未来方向。
- 论文依赖 Physical Intelligence 私有的大规模数据、π0.6 checkpoint 与机器人基础设施；只公开论文不足以复现。
- 4B VLM + 860M action expert、最多四路 448² 视频和 H100 推理仍是高成本方案，“高效”是相对朴素多帧 VLA，而非边缘设备高效。
- 许多定量结果只绘图而无数据表，长程任务仅 10 rollouts/条件，难以精确比较小差异。
- 高层摘要标签由 off-the-shelf LLM 离线生成，标签质量、prompt、模型版本和审核流程未充分披露。

### 7.2 个人判断

MEM 最有价值的洞见是**信息应按时间尺度改变表示，而不是只扩大 context window**。短期像素回答“刚刚怎样变化”，长期文字回答“哪些事实仍重要”，两者对应不同信息率。这比单纯堆帧更接近分层状态估计。

但自然语言摘要并不天然等于可靠 belief state。对于“洗洁精已加/未加”这样的离散事实，可用结构化键值或事件图增强可验证性；对于对象身份、数量和位置，可保留带置信度的 object-centric memory。未来系统很可能不是 video + free-form text 的二选一，而是视频、结构化事件、可检索轨迹和语言解释的多层记忆。

---

## 8. 复现成本与实践建议

### 8.1 复现成本

| 项目 | 评估 | 依据 |
|---|---|---|
| 模型 | 极高 | Gemma 3 4B + SigLIP 400M + 860M action expert |
| 预训练 | 极高 | 多机器人示范、rollout、人类纠正、VL/视频语言数据；算力未披露 |
| 推理 | 高 | 最多四路 448×448 视频；延迟图使用单张 H100 |
| 标注 | 高 | subtask、成功/失败、LLM memory summary、长程评分 rubric |
| 真机 | 极高 | 单臂、双臂、移动机器人，多厨房/卧室/灵巧任务 |
| 代码/权重 | 低可得性 | 论文未给出独立公开代码或 π0.6-MEM 权重 |

### 8.2 最小可行复现

可在开源 VLA 上实现缩小版：保留原 ViT 权重，每四层对同 patch 加一次 causal temporal attention；用 $e(0)=0$ 的固定正弦时间编码；最后只输出当前帧 token。先在公开视频 caption/机器人轨迹上训练 4–6 帧历史，再在一个遮挡任务和一个重复失败纠正任务上拉长窗口。长期部分可先把 subtask 历史压成结构化 checklist，再与自由文本摘要比较。

必须做的 baseline 包括：single-frame、frame-token concatenation、mean pool、proprio-only、post-train-only，以及固定 token 预算下的公平比较。评测应单独报告记忆任务收益和普通任务退化，防止模型通过记忆提升一种任务却普遍变慢或出现 causal confusion。

---

## 9. 个人启发与后续问题

1. **Memory pretraining 应成为 VLA 预训练的一等目标。** 先训无记忆模型、下游再拼历史，效果明显不足。
2. **失败数据的价值依赖可见历史。** 如果策略看不到失败前因，人类纠正只是另一个动作样本；有记忆后才成为“如果刚才失败，就换策略”的条件示范。
3. **压缩策略应可学习但也应可审计。** 可让模型输出 memory、置信度和删除理由，并保留只读事件日志用于恢复。
4. **应测试跨 episode 和记忆污染。** 长期部署会遇到过期事实、环境变化、用户冲突指令与错误摘要，这比 15 分钟 episode 更难。
5. **π0.7 已验证模块可迁移，但仍需开源基准。** 一个包含遮挡、计数、计时、纠错、长期 checklist 且能自动评分的公共套件，将比单纯平均成功率更能推动记忆 VLA。

---

## 10. 材料来源

- 官方论文页：<https://arxiv.org/abs/2603.03596>
- 官方 PDF：<https://arxiv.org/pdf/2603.03596>
- 官方项目页：<https://pi.website/research/memory>
- 本地来源清单：`assets/MEM - 多尺度具身记忆VLA/sources.md`
- 文中明确数字来自官方 PDF；Figure 8 未标注的柱高只作为“约值/图示读数”呈现，未冒充作者提供的精确表格。
