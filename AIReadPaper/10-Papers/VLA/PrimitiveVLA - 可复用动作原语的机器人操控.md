---
type: paper-note
status: done
domain: VLA
paper: PrimitiveVLA: Learning Reusable Motion Primitives for Efficient and Generalizable Robotic Manipulation
year: 2026
arxiv: 2605.28634
doi: null
source: https://arxiv.org/abs/2605.28634
project: null
code: null
tags:
  - VLA
created: 2026-05-29
updated: 2026-08-21
---

# PrimitiveVLA：可复用动作原语的机器人操控

## 基本信息

| 字段 | 内容 |
|---|---|
| 论文名称 | PrimitiveVLA: Learning Reusable Motion Primitives for Efficient and Generalizable Robotic Manipulation |
| arXiv ID | 2605.28634v2 (cs.RO) |
| 发表日期 | 2026-07-18（v2） |
| 机构 | 中科院计算所、中科院软件所、寒武纪、中科院大学 |
| 一作 | Yutai Li（计算所） |
| 通讯作者 | Yunji Chen |
| 许可证 | CC BY 4.0 |
| 原始链接 | https://arxiv.org/abs/2605.28634 |
| HTML 版 | https://arxiv.org/html/2605.28634v2 |
| PDF | assets/PrimitiveVLA - 可复用动作原语的机器人操控/paper.pdf |
| 项目页 | 论文未披露 |
| 代码仓库 | 论文未披露 |

## 一句话结论

PrimitiveVLA 把 VLA 的训练监督从"任务指令 → 整段轨迹"重构为"任务指令 → primitive 序列 → 单步动作"，通过 VLM 推理 + LLM 代码生成完成 primitive 自动化拆解、用 MCR（多模态规范化表示）抹平任务文本与物体外观差异、再在推理时用 VLM 规划器 + LLM 切换器进行双线程闭环组装，使同一套机制同时提升数据效率、未见任务泛化和长程任务稳定性，在 OpenVLA / OpenVLA-OFT / π₀.₅ 三种底座上全面取得 SOTA 级增益。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

VLA（Vision-Language-Action）模型被视为通用机器人策略的有前景范式，但在真实部署时普遍遭遇两个瓶颈：**数据效率低**（同样的任务量很难榨出足够性能）与**泛化能力差**（在新任务、新物体、新组合上掉点严重）。论文把这两个表象背后归因到同一个根上——当前主流微调范式把任务指令 $l$ 直接对齐到整条低层控制轨迹 $\tau$，模型被迫去"背轨迹"而不是去"理解动作模式"。

### 1.2 现有方法的瓶颈

作者将主流做法概括为 **Direct Instruction-to-Control Mapping**，其训练目标是把整段示范 $\tau$ 在 $l$ 条件下做最大似然：

$$\mathcal{L}_{task} = -\sum_{t=1}^{T} \log \pi_\theta(a_t \mid o_t, s_t, l)$$

这种"端到端黑箱"监督会带来三类问题：

1. **轨迹级记忆化**：模型容易把"开抽屉 → 抓碗 → 放进去"这种特定组合当作不可拆的整体记下来，遇到相似但不完全一样的 OOD 任务就直接掉链子；
2. **跨任务表征纠缠**：抓黑色碗、抓红色杯子、抓绿色积木在物理上其实是同一种运动模式（grasp），但在文本监督下却互不共享，效率低；
3. **长程规划难**：推理时没有任何显式中间表示，规划、控制、错误恢复全部压进一个 Transformer，前几步一旦跑偏后面很难回正。

### 1.3 本文核心贡献

论文的核心不是换 backbone，而是**重构 VLA 的监督组织方式**，主要贡献有四项：

1. **Primitive-Centric Disassemble & Assemble 范式**：把任务拆成有限、可解释、可复用的运动原语（primitive），训练阶段把 $\tau$ 自动切成 primitive 片段，推理阶段把 primitive 重新组装成完整行为；
2. **自动化 primitive 拆解流水线**：用 VLM（Qwen3-VL）做高层 primitive 序列推理 + LLM（DeepSeek-V3）自动生成 Python 规则代码做时间边界定位，避免任何人工逐帧标注；
3. **MCR（Multimodal Canonical Representation）统一学习接口**：语言侧把所有抓取指令统一成"抓绿色 mask 标出的物体"这种 canonical form，视觉侧用 SAM + Cutie 跟踪得到 object-centric mask，从语义和外观两端去噪；
4. **Planner + Switch 双线程推理**：高层 VLM planner 生成 primitive 序列，低层 fine-tuned VLA 输出连续动作，同时 LLM 生成的 switch 代码在滑动窗口上判断是否切换到下一个 primitive，闭环执行。

![](assets/PrimitiveVLA%20-%20可复用动作原语的机器人操控/fig1-teaser.png)

> 图 1：PrimitiveVLA 的动机对比。a) 传统 Direct Mapping 范式把"Pick sponge and place in the plate"等任务直接喂给 VLA，ID 任务能学会、但 OOD 任务（"Pick sponge and place in the caddy"）会失败。b) PrimitiveVLA 范式先在训练时把任务拆成 grasp / move / place 等 primitive 片段并应用 MCR canonical 化，推理时按 primitive 重组；ID 和 OOD 任务均成功，因为底层组合的是已学到的物理运动模式。  
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/html/2605.28634v2

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

论文把机器人操作形式化为序列决策过程。在第 $t$ 步，策略观测到状态元组 $(o_t, s_t)$：

- 视觉观测 $o_t = \{I_t^{(\text{global})}, I_t^{(\text{wrist})}\}$：第三人称 RGB 图像 + 腕部 RGB 图像；
- 本体感觉状态 $s_t \in \mathbb{R}^7$；
- 动作 $a_t \in \mathbb{R}^7$：6-DoF 末端执行器增量位姿（$\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}$）+ 夹爪开合；
- 完整任务示范轨迹 $\tau = \{(o_t, s_t, a_t)\}_{t=1}^{T}$。

任务指令 $l$ 是自然语言高层描述（如 "Pick the sponge and place on the plate"）。PrimitiveVLA 不改动作空间和观测空间，**模型无关地**插入到任意 pre-trained VLA（OpenVLA、OpenVLA-OFT、π₀.₅）之上。

### 2.2 关键符号和目标函数

**原始任务级训练目标**（传统 VLA 微调）：

$$\mathcal{L}_{task} = -\sum_{t=1}^{T} \log \pi_\theta(a_t \mid o_t, s_t, l)$$

直觉：把整段 $\tau$ 在 $l$ 条件下做似然最大化；缺点是粒度太粗，模型只能学到"任务 ↔ 轨迹"映射，primitive 这一层完全被淹没。

**Primitive-Centric 训练目标**（本文）：

$$\mathcal{L}_{prim} = -\sum_{i=1}^{N} \log \pi_\theta(a_i \mid \tilde{o}_i, s_i, c_i)$$

其中 $c_i$ 是 canonical primitive 指令（与任务解耦，如 "Grasp the masked object with color_1"），$\tilde{o}_i = o_i \otimes M_i$ 是 object-centric mask 观测，$N$ 是单条任务被拆出的 primitive 数。**损失函数形式没变，变化的是监督粒度**——把"一条样本 = 一条任务轨迹"拆成"一条样本 = 一个 primitive 片段"，从而把跨任务共享的物理模式暴露给模型。

论文的拆解算子形式化为：

$$\tau \rightarrow \{(\tilde{o}_i, s_i, a_i, c_i)\}_{i=1}^{N}$$

---

## 3. 核心方法

### 3.1 总体框架

PrimitiveVLA 把"如何让 VLA 学到可复用动作"这件事拆成两个互补阶段，并通过共享的 MCR 把它们缝起来：

- **Fine-tuning 阶段（离线 Primitive Disassembly）**：原始任务轨迹 → VLM 推理 + LLM 代码切分 → primitive 片段 → MCR canonical 化 → 喂给任意 VLA 做监督学习；
- **Inference 阶段（在线 Primitive Assembly）**：测试指令 → VLM planner（带检索）→ primitive 序列 → VLA policy 闭环执行 + LLM switch 在滑动窗口上判断切换时机。

两个阶段通过中间的 MCR 共享同一套表征规范，从而保证训练时学到的 primitive 在推理时能直接复用。

![](assets/PrimitiveVLA%20-%20可复用动作原语的机器人操控/fig2-method.png)

> 图 2：PrimitiveVLA 总体框架。左侧绿色框为离线 Fine-Tuning 流水线：原始任务数据 → VLM (Qwen3-VL) 推理 + LLM (DeepSeek-V3) 生成边界代码 → 自动化 primitive 切分 → 规范化指令与 masked 图像 → 任意预训练 VLA 训练。右侧红色框为在线 Inference 流水线：测试指令 → VLM Planner（带检索的 plan examples）→ Primitive Switch（LLM 代码 + 滑动窗口）→ 双线程执行 VLA policy + 环境监控。中间蓝色 MCR 模块把训练和推理两端的视觉与语言表征都做 canonical 化，使两阶段共享同一套动作语义。  
> 来源：论文 Figure 2，第 5 页，https://arxiv.org/html/2605.28634v2

### 3.2 关键模块一：自动化 Primitive Disassembly

把一条任务轨迹 $\tau$ 拆成 primitive 片段需要同时解决两个子问题：(a) 拆出哪几个 primitive？(b) 每个 primitive 何时开始、何时结束？论文刻意用 VLM 解决 (a)、用 LLM 生成的代码解决 (b)，避免直接用 VLM 做时间边界（黑箱 + 不稳定）。

**(a) Primitive Sequence Reasoning（VLM 推理高层序列）**

$$\mathcal{S} = f_{\text{VLM}}(l, \mathcal{V}_\tau, \mathcal{C})$$

- $l$：任务指令；
- $\mathcal{V}_\tau$：示范的 RGB 序列；
- $\mathcal{C}$：primitive 库（11 个预定义 primitive 的定义）；
- 实际使用 $f_{\text{VLM}}$ = **Qwen3-VL**。

输出 $\mathcal{S} = [p_1, p_2, \dots, p_k]$ 是该任务在 primitive 维度上的高层结构。生成的 $(l, \mathcal{S})$ 对被索引到 **Disassembly Library** $\mathcal{D}$，在推理时作为检索先验使用。

**(b) State-Based Boundary Segmentation（LLM 生成代码做时间切分）**

每个 primitive 都有自己的物理终止条件。论文把这个条件显式代码化：把 primitive 集合 $\mathcal{P} = \bigcup_i \{p \mid p \in \mathcal{S}_i\}$ 和它们的物理定义丢给 LLM，让 LLM 自动生成 Python 函数 $\phi_i$。对于从 $t_{\text{start}}$ 开始的 primitive，终点 $t_{\text{end}}$ 判定为：

$$t_{\text{end}} = \min\{t \mid t > t_{\text{start}} + \delta, \ \phi_i(s_{t-k:t+k}) = \text{True}\}$$

其中 $\delta$ 是时间偏移（默认 10 步），$\phi_i$ 在长度为 $2k+1$ 的局部窗口上判断，避免单步传感器噪声造成误切。LLM 用 **DeepSeek-V3**。论文给了一个典型例子：Grasp primitive 的终止条件是"夹爪宽度保持恒定（$w < \epsilon$）且末端高度开始增加（$\Delta z > \delta$）"。

论文附录里特别强调为什么不直接用 VLM 做时间切分——黑箱性 + 数值敏感性会导致边界在相邻帧之间漂移，而 LLM 生成的代码可以反复执行、可读、可校准，也方便针对复杂场景提供少量样本做阈值校准。

**(c) Primitive 库 $\mathcal{C}$（11 个预定义 primitive）**

| 类别 | Primitive | 运动学定义 |
|---|---|---|
| 空间运输 | Grasp | 接近、抓取目标并进行初步提升 |
| | Place | 下降、释放物体并垂直后退 |
| | Lift | 持物时沿 +z 方向垂直平移 |
| | Move | 保持抓取状态下的平面 xy 平移 |
| 接触与交互 | Push | 在表面上将物体滑动到目标位置（不抓取） |
| | Pull | 将物体拖拽或拉向目标位置 |
| | Insert | 对齐并插入受约束的槽口 |
| | Press | 对物体或表面施加向下的力 |
| 姿态调整 | Twist | 旋转夹爪（roll）以操作旋转机构 |
| | Tilt | 调整末端执行器姿态（pitch/yaw）改变位姿 |
| | Rotate | 沿铰接轨迹运动（如盖子） |

11 个 primitive 是基于标准夹爪 manipulation 任务归纳出来的，强调覆盖空间运输、接触交互、姿态调整三大物理动作族，但**论文未披露**是否做过任务-primitive 覆盖度分析，也没有给出"哪些操作必然被排除在外"的边界说明。

![](assets/PrimitiveVLA%20-%20可复用动作原语的机器人操控/fig4-segcompare.png)

> 图 4：不同 primitive 拆解方法的定性对比。第一行 Universal Visual Decomposer（UVD，纯视觉）对任务 a）会把"开抽屉"和"抓碗"两类动作合并、对任务 b）会出现冗余分割 seg3/seg4。第二行 Qwen3.5-Omni-Plus（直接 VLM）会出现边界不精确、任务中途终止、冗余"move"片段。第三行 PrimitiveVLA（VLM 语义推理 + LLM 确定性代码）给出最稳定、最符合语义、且时间边界最干净的切分结果。  
> 来源：论文 Figure 4，第 7 页，https://arxiv.org/html/2605.28634v2

### 3.3 关键模块二：MCR（Multimodal Canonical Representation）

MCR 的目标是把所有"同一种 primitive"的不同任务表达都对齐到统一接口，从而让 VLA 真正学 primitive 而不是学任务。它由两步 canonical 化组成：

**Step 1：Instruction Canonicalization**

把所有同一 primitive $p_i$ 的样本统一到一条 canonical 指令 $c_i$。论文给出了一张完整的 canonical 指令表：

| Primitive | MCR 规范化指令 |
|---|---|
| grasp | "Grasp the masked object with color_1" |
| lift | "Lift the object with color_1" |
| move | "Move to above the object with color_2" |
| place | "Place in the object with color_2" |
| push | "Push the object with color_1" |
| press | "Press the object with color_2" |
| pull | "Pull the object with color_1" |
| insert | "Insert into the area with color_1" |
| twist | "Twist the grasped object" |
| rotate | "Rotate the object with color_1" |
| tilt | "Tilt the object with color_1" |

`color_1` 表示被操作物体的语义掩码，`color_2` 表示目标位置/物体的语义掩码。**论文未披露**具体的 prompt 模板和如何在 primitive 类型间系统化生成这套句子。

**Step 2：Object-centric Masking（视觉侧 canonical 化）**

$$\tilde{o}_i = o_i \otimes M_i$$

- 用 **Qwen2.5-VL-72B-Instruct** 做零样本物体检测，得到 JSON 格式的边界框；
- 把边界框送进 **SAM** 生成高保真初始掩码；
- 在整条轨迹中用 **Cutie**（视频物体分割）跟踪掩码，提供实时更新。

这套链路只在预执行初始化阶段调用 Qwen2.5-VL-72B 和 SAM 各一次（重模型），运动开始后由轻量级 Cutie 持续跟踪，因此**在线 MCR 变换的延迟可忽略**——这是 PrimitiveVLA 能在闭环控制里同时维持 primitive 语义清晰度和执行实时性的关键工程设计。

### 3.4 训练目标与损失函数

PrimitiveVLA **不引入新的损失函数**。其监督方式的变化体现在数据组织层：

$$\mathcal{L}_{prim} = -\sum_{i=1}^{N} \log \pi_\theta(a_i \mid \tilde{o}_i, s_i, c_i)$$

对比 $\mathcal{L}_{task}$，变量层面的差别是：

| 项 | $\mathcal{L}_{task}$ | $\mathcal{L}_{prim}$ |
|---|---|---|
| 监督粒度 | 一条样本 = 一条任务 | 一条样本 = 一个 primitive 片段 |
| 条件 $l$ / $c$ | 任务级自然语言 | canonical primitive 指令 |
| 视觉 $o$ / $\tilde{o}$ | 原始 RGB | object-centric masked RGB |
| 轨迹长度 $T$ | 整段任务长度 | 单个 primitive 长度（通常更短） |

这意味着 $\mathcal{L}_{prim}$ 在**函数形式**上与标准 VLA 微调完全一致，所以可以直接套用 OpenVLA、OpenVLA-OFT、π₀.₅ 的训练 pipeline，无需修改 loss 也不需要新增模块（个人判断：这也是论文能横跨三种不同底座都取得稳定增益的根本原因——它改的是数据组织而不是 loss landscape）。

### 3.5 推理流程与复杂度

推理流程是 **Planner + Switch + VLA** 三件套的双线程执行：

**（1）Primitive Planner**

对测试任务 $l_{\text{test}}$：

- 从 Disassembly Library $\mathcal{D}$ 中按语义余弦相似度检索 top-3 相似的 $(l, \mathcal{S})$ 样本；
- 把这些先验、当前指令、初始观测 $o_0$、primitive 库 $\mathcal{C}$ 一起喂给 VLM：

$$\mathcal{S}_{\text{plan}} = f_{\text{VLM}}(l_{\text{test}}, o_0, \mathcal{C}, \text{Retrieve}(l_{\text{test}}, \mathcal{D}))$$

通过 in-context example 约束 VLM 输出符合训练分布的序列，抑制幻觉（论文未披露 VLM 的具体 prompt 模板）。

**（2）Primitive Switch**

仍然用 LLM 生成 Python 代码 $f_{\text{switch}}$，但**镜像的是分割标准** $\phi_i$、优化方向变成"从滑动窗口统计量判断是否完成"：

$$\text{SwitchTrigger} \iff \forall s \in s_{\text{stat}}, \ f_{\text{switch}}(s, p_i) = \text{True}$$

论文给出的 Switch 算法（Algorithm 2）核心是：在长度为 $W$ 的历史窗口 $\mathcal{H}_t$ 上算统计量 $s_{\text{stat}}$（如变化率、相对位移），再用 LLM 生成的 `GetCondition(P)` 函数判断当前 primitive 是否完成。

**（3）双线程 Closed-loop Execution**

- 执行线程：把当前 primitive、实时观测、VLA 串起来生成连续动作；
- 监控线程：并行持续计算 $s_{\text{stat}}$ 并判断是否触发 Switch。

执行线程不需要等监控线程，每个控制周期独立输出动作；监控线程只在统计量越过阈值时通知主线程切换到下一个 primitive，从而实现反应式闭环。

**复杂度讨论（个人判断）**：

- 训练时增加的开销主要是 VLM（Qwen3-VL）推理一次 + LLM（DeepSeek-V3）一次生成 Python 代码 + 物体检测与 SAM 各一次——这些都是离线一次性成本，不进反向传播；
- 推理时增加的开销是 VLM Planner 一次（在线但低频）+ LLM Switch 周期性评估（用代码执行，常数级）——**论文未披露**具体的端到端延迟数字，但定性描述"在线 MCR 变换延迟可忽略"，加上 Switch 用代码而非 LLM 推理，推断控制频率应能维持标准 VLA 水平（10-30 Hz）。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

| 数据集 | 用途 | 任务数 | 说明 |
|---|---|---|---|
| Libero-Spatial / Object / Goal | 仿真 ID 微调 | 各 10 | 经典 Libero 短程任务 |
| Libero-90（子集） | 大规模微调 | 84 | 从 90 个任务中剔除 6 个低质量任务 |
| Libero-90-Novel | OOD 评估 | 8 | 自定义的未见逻辑任务（开抽屉-放碗-关抽屉、不同厨房场景组合） |
| Libero-Long | 长程评估 | - | 长时序多阶段任务 |
| RLBench | 高多样性压力测试 | 10 | Meat Off Grill、Lamp Off、Push Button、Open Jar 等 |
| UR5e 真实环境 | 真机评估 | 11 | ID 6 + 任务泛化 3 + 组合泛化 2 |

**真实世界 11 任务清单**：

- ID（6）：Pick sponge on plate; Pick cup on cabinet; Pick cube in top drawer; Pick cube in caddy; Push top drawer; Push bottom drawer
- 任务泛化（3）：Pick sponge in caddy（未见配对）; Pick cube on cabinet（未见配对）; Pick blue cup on cabinet（新物体）
- 组合泛化（2）：Pick cup then push drawer; Pick sponge then cube

**RLBench 10 任务**：Meat Off Grill、Lamp Off、Push Button、Open Jar、Lift Numbered Block、Open Wine Bottle、Take Off Scales、Take Lid Off Saucepan、Put Rubbish In Bin、Take Umbrella。

数据预处理：每条示范都经过 Disassembly 流水线（VLM 推理 + LLM 代码切分）拆成 primitive 片段，再经 MCR canonical 化形成 $\mathcal{L}_{prim}$ 监督样本。

### 4.2 Baseline 与评价指标

**底座 VLA 模型**（论文验证了模型无关性）：

| 底座 | 特点 |
|---|---|
| OpenVLA | 7B 参数的标准 VLA Transformer |
| OpenVLA-OFT | OpenVLA 的优化微调变体，加入并行 decoding 等 trick |
| π₀.₅ | 通用机器人基础模型，使用流匹配输出动作 |

**外部对比**（Libero 子集）：TraceVLA、SpatialVLA、Diffusion Policy、Octo、π₀。

**评价指标**：成功率（Success Rate）。仿真任务 50 次试验取平均，真实世界和 RLBench 任务 20 次试验取平均。**论文未披露**每条试验的随机种子数量与方差。

### 4.3 实现细节

- **Qwen2.5-VL-72B-Instruct** 做物体检测；**SAM** 生成初始 mask；**Cutie** 视频对象分割做在线跟踪；
- **Qwen3-VL** 做 primitive 序列推理（VLM planner 也用同一类模型）；
- **DeepSeek-V3** 生成 primitive 分割和 switch 条件的 Python 代码；
- 真实世界硬件：UR5e 机械臂 + Robotiq 2F-85 夹爪 + 两台 Intel RealSense L515 相机（一台第三人称、一台腕部安装）。

![](assets/PrimitiveVLA%20-%20可复用动作原语的机器人操控/fig5-real.png)

> 图 5：真实世界硬件与任务示例。左侧为实验台：UR5e 6-DoF 机械臂 + Robotiq 2F-85 夹爪 + 一台第三人称 Intel RealSense L515 相机 + 一台腕部 RealSense L515 相机。右侧三行为三组真实任务示例，分别为 ID 任务（"Pick the sponge and place on the plate"）、任务泛化（"Pick the sponge and place in the caddy"，物体-容器配对是训练时未见的）、组合泛化（"Pick the cup and place on the cabinet and push in the top drawer"）。  
> 来源：论文 Figure 5，第 8 页，https://arxiv.org/html/2605.28634v2

---

## 5. 实验结果

### 5.1 主要定量结果

**Libero 综合评估（Table 3）**——这是论文最核心的成绩表，覆盖 ID、90、90-Novel、Long 四个子基准，三档不同底座：

| 模型 | Object | Spatial | Goal | ID 均值 | Libero-90 | 90-Novel (OOD) | Long |
|---|---|---|---|---|---|---|---|
| OpenVLA | 87.40 | 82.80 | 74.00 | 82.40 | 70.60 | 7.38 | 4.50 |
| **OpenVLA + Ours** | **90.60** | **91.20** | **82.20** | **88.00** | **79.80** | **45.50** | **38.50** |
| OpenVLA-OFT | 97.40 | 98.60 | 96.80 | 97.60 | 89.70 | 13.50 | 3.75 |
| **OpenVLA-OFT + Ours** | **98.40** | **99.40** | **97.80** | **98.53** | **94.70** | **71.00** | **66.50** |
| π₀.₅ | 98.20 | 98.40 | 96.00 | 97.53 | 96.80 | 56.00 | 30.50 |
| **π₀.₅ + Ours** | 98.00 | 98.20 | 95.80 | 97.33 | 96.20 | **75.50** | **80.25** |

**这张表说明了什么**：

- 在 ID 子集上，OpenVLA 提升 +9.2pp（70.60 → 79.80）、OpenVLA-OFT 提升 +5.0pp（89.70 → 94.70），但本身已经接近饱和的 π₀.₅ 基本不动；
- 在 90-Novel（OOD）上，**OpenVLA 直接从 7.38% 拉到 45.50%（约 6× 提升）**，OpenVLA-OFT 从 13.50% 拉到 71.00%，π₀.₅ 从 56.00% 拉到 75.50%——这是论文最戏剧化的结果；
- 在 Long（长程）上，OpenVLA-OFT 从 3.75% 拉到 66.50%，π₀.₅ 从 30.50% 拉到 80.25%——是 VLA 长程执行最头疼的痛点上的明显改善；
- **底座越弱、绝对提升越显著**。π₀.₅ 这种本身就有大量预训练的底座在 ID 上几乎饱和，但在 OOD / Long 上仍能拿到 +19.5pp 和 +49.75pp 的强增益，说明 primitive 重组带来的不只是"学得更好"，更是"泛化结构对得上"。

**RLBench 高多样性压力测试（Table 5）**：

| 模型 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | 均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenVLA | 25 | 90 | 95 | 20 | 40 | 20 | 15 | 95 | 40 | 55 | 49.5 |
| **+Ours** | **35** | **100** | 95 | 15 | 35 | **30** | **40** | **100** | **50** | **65** | **56.5** |

**平均 +7.0pp**，10 个任务里有 8 个持平或提升、2 个略掉。RLBench 任务更复杂（铰链物体、旋转机构、按钮、盖子），仍能稳赚不亏，说明 primitive 库（11 个含 Push/Twist/Rotate 等）覆盖了关键操作族。

### 5.2 定性结果

**OOD 任务可视化**——以 Libero-90-Novel 中的"Pick up the bowl on the right and put it in the tray"为例：

![](assets/PrimitiveVLA%20-%20可复用动作原语的机器人操控/fig7-gen.png)

> 图 7：OOD 任务上的执行对比。给定指令"pick up the bowl on the right and put it in the tray"：上排 baseline 反复重复训练集中的相似任务（Pick sponge and place near plate）但完全不动碗，任务失败；下排 PrimitiveVLA 先 grasp 碗（红框 mask 高亮），再将其平移至托盘，任务成功。Baseline 表现说明它学到的是"轨迹 ↔ 表面文本"映射、遇到没见过的物体-容器组合就退化为最像的训练轨迹；PrimitiveVLA 表现说明它在底层用 primitive 重组的方式真正完成了"抓右边的碗 → 放进托盘"的物理动作。  
> 来源：论文 Figure 7，第 10 页，https://arxiv.org/html/2605.28634v2

**真实世界定性示例**（图 5 右侧三组）：ID 任务、任务泛化、组合泛化三类场景下 PrimitiveVLA 均能完成对应动作序列（**论文未披露**对真实任务的视频失败切片，仅有成功示例）。

**Disassembly 拆解质量对比**（图 4）——通过 UVD / Qwen3.5-Omni-Plus / PrimitiveVLA 三种方法对比证明：UVD 出现合并相邻动作、冗余分割；Qwen3.5-Omni-Plus 出现边界不精确、冗余 move、任务中途终止；PrimitiveVLA 既符合语义又边界稳定。

### 5.3 消融实验

Table 6 给出两个底座上三种设置的消融：

| 底座 | 设置 | Libero-90 | 90-Novel | Long |
|---|---|---|---|---|
| OpenVLA | Baseline | 70.60 | 7.38 | 4.50 |
| OpenVLA | w/o MCR（仅拆解） | 73.90 | 21.50 | 28.20 |
| OpenVLA | w/o 拆解（仅 MCR） | 79.60 | 33.75 | 4.25 |
| OpenVLA | **Ours（完整）** | **79.80** | **45.50** | **38.50** |
| OpenVLA-OFT | Baseline | 89.70 | 13.50 | 3.75 |
| OpenVLA-OFT | w/o MCR（仅拆解） | 89.60 | 15.00 | 52.30 |
| OpenVLA-OFT | w/o 拆解（仅 MCR） | 94.30 | 60.00 | 39.75 |
| OpenVLA-OFT | **Ours（完整）** | **94.70** | **71.00** | **66.50** |

**这张表清晰说明了三件事**：

1. **MCR 是 OOD 泛化的主驱动力**——以 OpenVLA-OFT 为例，去掉 MCR 仅保留 Disassembly，Novel 从 13.50 → 15.00，几乎没动；但仅保留 MCR 去 Disassembly，Novel 直接拉到 60.00。说明"用 canonical 指令 + masked 观测去对齐跨任务语义"才是 OOD 泛化的关键；
2. **Primitive Disassembly 是长程稳定性的主驱动力**——同样的 OpenVLA-OFT，仅 Disassembly 不 MCR 让 Long 从 3.75 → 52.30（+48.55pp），但 MCR 不 Disassembly 只到 39.75。说明把任务显式拆成 primitive 段落、在推理时用 Switch 判定进度，是长程能跑完的根本原因；
3. **两者是互补的**——MCR 解决"看不懂任务"问题、Disassembly 解决"做不完任务"问题；合起来在 Novel / Long 上都比单模块更好（OFT 上 71.00 / 66.50 vs 60.00 / 52.30）。

### 5.4 泛化、效率与失败案例

**数据效率（Table 4）**——用一半数据即可打过全量基线：

| 底座 | 数据规模 | Libero-90 | Libero 全集均值 |
|---|---|---|---|
| OpenVLA | 50% | 65.89 | 74.07 |
| OpenVLA | 100% | 70.60 | 78.70 |
| **OpenVLA + Ours** | **50%** | **73.40** | **80.30** |
| **OpenVLA + Ours** | **100%** | **79.80** | **85.95** |
| OpenVLA-OFT | 50% | 87.00 | 92.80 |
| OpenVLA-OFT | 100% | 89.70 | 95.63 |
| **OFT + Ours** | **50%** | **92.12** | **95.18** |
| **OFT + Ours** | **100%** | **94.70** | **97.58** |

**关键点**：OpenVLA + Ours 50% 数据（80.30）超过 OpenVLA 100% 数据（78.70）；OFT + Ours 50% 数据（92.12）已经超过 OFT 100% 数据（89.70）。这意味着同样标注预算下，PrimitiveVLA recipe 能直接砍掉 50% 数据采集成本（个人判断：这才是工业界最关心的数字，单点性能涨 5-9pp 远不如"少采一半数据"来得实在）。

**真实世界评估（Table 7，UR5e）**——π₀.₅ + Ours vs π₀.₅：

| 类型 | T1 | T2 | T3 | T4 | T5 | T6 | 均值 | T7 | T8 | T9 | OOD 均值 | T10 | T11 | 组合均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| π₀.₅ | 60 | 75 | 60 | 45 | 95 | 85 | 70 | 15 | 35 | 10 | 20 | 0 | 20 | 10 |
| **π₀.₅ + Ours** | **85** | **95** | **85** | **85** | **100** | **90** | **90** | **40** | **50** | **80** | **57** | **60** | **70** | **65** |

- ID 平均：70 → 90（**+20pp**）；
- 任务泛化平均：20 → 57（**接近 3×**）；
- 任务 T9 "Pick blue cup on cabinet"（新物体）：10 → 80（**8× 提升**）；
- 组合任务 T10 "Pick cup then push drawer"（baseline 几乎 0%）：→ 60。

**失败案例**：论文未单独列出失败样本集。从数据反推，RLBench 中 T4（Open Jar）和 T5（Lift Numbered Block）出现轻微掉点，说明对**铰链开启+小物体精准操作**类任务，PrimitiveVLA 的 11 个 primitive 可能不够细粒度（个人判断：这是限制 1 的实证痕迹）。

---

## 6. 与相关工作的关系

| 相关工作 | 与 PrimitiveVLA 的关系 |
|---|---|
| **OpenVLA / OpenVLA-OFT** | 底座 VLA；PrimitiveVLA 兼容并以之做底座 |
| **π₀ / π₀.₅** | 流匹配通用基础模型；本文用它做最强底座，验证模型无关性 |
| **TraceVLA / SpatialVLA** | 同样做 trace-level 增强；本文证明 primitive-level 才是更优监督粒度 |
| **Octo / Diffusion Policy** | 通用策略基线；本文在 Libero 子集上明显超过 |
| **UVD（Universal Visual Decomposer）** | 纯视觉动作分割方法；本文图 4 证明纯视觉无法处理"开抽屉+抓碗"等语义相邻动作的边界 |
| **VLM-as-Planner 系列（HuggingGPT、ReAct 等）** | 同样用 VLM 做高层规划；本文把 VLM 限制在 primitive 序列层面、并用 LLM 代码做切换以保证时间稳定性 |
| **LLM 自动化代码生成（Code-as-Policies、VOIDC 等）** | 本文把 LLM 代码用于 primitive 边界判定和切换判定，代替易漂移的 VLM 时间切分 |

PrimitiveVLA 在定位上与 ReconVLA、DreamZero、GaussianDream、Qwen-VLA 都属于"VLA 主线新工作"，但差异明显：ReconVLA 改视觉对齐（重建任务区域）、DreamZero 用视频世界模型零样本、GaussianDream 引入 3D 表征、Qwen-VLA 做统一跨任务跨 embodiment；而 **PrimitiveVLA 改的是监督组织粒度**——这是它最独到的方法论信号（个人判断）。

---

## 7. 局限与批判性评价

1. **primitive 集合依赖人工定义**：当前 11 个 primitive 偏向标准夹爪 manipulation，对灵巧手（dexterous hand）、双臂协作、接触丰富任务、铰链体 + 旋转机构等是否仍然完备，论文未给出系统覆盖度分析。RLBench 中 T4（Open Jar）掉点可能是该限制的早期信号。
2. **自动拆解的鲁棒性依赖 LLM 代码质量**：DeepSeek-V3 生成的 Python 代码质量在跨 embodiment、跨传感器、跨场景切换时是否仍然可靠，**论文未披露**相关压力测试；论文提到可"提供少量样本做阈值校准"，但这本身又增加了工程负担。
3. **MCR 对 mask 质量的依赖**：SAM + Cutie 链路在遮挡严重、目标切换频繁、多物体同时交互时是否仍然稳定，**论文未做专门实验**。Mask 抖动会直接污染 primitive 学习与切换判定。
4. **系统复杂度高**：VLM 推理 + LLM 代码生成 + 检索增强规划 + 双线程监控，远比端到端 VLA policy 复杂，对部署提出了更高要求。推理延迟、显存峰值、组件版本稳定性都是工程隐患。
5. **VLM/LLM 选型与训练数据未完全披露**：哪些 prompt、用哪些 in-context example、VLM planner 的检索评分函数等，**论文未披露**细节；reproducibility 受到影响。
6. **未扩展到非夹爪 embodiment**：6-DoF 末端 + 1-DoF 夹爪的范式很强，但论文没有验证 mobile manipulation、软体机器人、双臂等场景，primitive 库在这些场景下需要重新设计。
7. **未给出方差与显著性检验**：每条成功率是 50 次（仿真）/ 20 次（真机）取平均，但**论文未披露**标准差、置信区间或显著性 p 值，部分提升（如 RLBench T4 -5pp、OpenVLA+ Spatial +8.4pp）需要谨慎看待。

---

## 8. 复现与实践建议

**复现成本评估**：

- **算力门槛**：底座 VLA（OpenVLA 7B / OpenVLA-OFT / π₀.₅）的微调需要 8×H100 量级，PrimitiveVLA 没有改 loss，只改数据组织，所以训练算力基本与原版 VLA 一致；
- **数据门槛**：论文中 Libero-90 子集 + Libero-90-Novel 8 + Libero-Long + RLBench 10 都在公开仿真器内可生成（Libero 是公开 benchmark），真机数据则需要 UR5e + Robotiq 硬件；
- **API 门槛**：Qwen3-VL / DeepSeek-V3 / Qwen2.5-VL-72B / SAM / Cutie 全部可调开源或 API，但**论文未披露**完整 prompt 模板和阈值参数；
- **代码可用性**：**论文未披露**官方代码仓库（project: null、code: null），**复现的最大瓶颈**其实是 VLM planner 与 switch 的 prompt + 阈值参数未公开。

**实操建议**（个人判断）：

1. **从仿真起步**：用 Libero-90 + Libero-90-Novel + Libero-Long 三件套验证 pipeline，把 VLM/LLM 调用都换成自部署版本（如 Qwen2.5-VL-7B-Instruct 替代 72B 试探成本），观察 primitive 拆解质量；
2. **用 OpenVLA-OFT 做底座性价比最高**：Table 4 显示 OFT 底座在 50% 数据下已能 95.18%，训练算力需求比 π₀.₅ 低很多；
3. **MCR 链路先做轻量验证**：可以先用 SAM 静态 + GrabCut 替代 Cutie 视频跟踪，看性能下降幅度以判断在线跟踪的边际收益；
4. **在自家真机上做小规模域内验证**：直接套到现有 6-DoF + 1-DoF 夹爪平台，先看 ID 任务是否能涨点，再逐步做 OOD 与组合；
5. **重点关注 primitive 库是否覆盖自家任务**：如果任务里有显著超出 11 个 primitive 的操作（如铰链开关、旋转拧紧），需要先扩充 primitive 库并补 MCR 规范化指令。

---

## 9. 个人启发与后续问题

**三个方法论信号**：

1. **从 task-level supervision 转向 primitive-level supervision**：VLA 的泛化瓶颈不是 backbone 能力，而是监督粒度。这条信号对所有具身方向的人都重要——它把"如何让 VLA 学到通用动作"从模型问题降维成数据组织问题。
2. **从单次任务拟合转向 compositional skill reuse**：把"任务 = 不可拆整体"换成"任务 = primitive 序列组合"，自然导出长程、组合泛化等"新能力"——这是 compositional generalization 思想在 VLA 时代的具象化。
3. **从 purely end-to-end policy 转向"高层规划 + 低层动作 + 显式切换"混合范式**：与"端到端大一统"的趋势相反，PrimitiveVLA 重新引入了显式中间层（planner、switch），但用 VLM/LLM 做软实现而不是硬规则——是 hybrid 范式的典型代表。

**后续想跟进的问题**：

- 与 π₀.₇、DreamZero、ReconVLA 在"结构化中间表示"上的差异对比；
- primitive 库能否通过 self-supervision 自动发现（即论文 limitation 中也提到的方向）；
- 拆解的 LLM 代码能否用更小的 specialist 模型替代 DeepSeek-V3，从而降低部署成本；
- 在双臂、灵巧手、mobile manipulation 上 primitive 库需要如何重新设计；
- 与 Video Prediction 世界模型结合：能否用未来帧预测自动生成 primitive 切换信号，进一步去掉 LLM switch。

**联系上下文**：

- 与 [[ReconVLA - 目标区域重建驱动的机器人视觉对齐]] 对比：两边都在"显式中间层"上做文章，ReconVLA 走视觉重建路线、PrimitiveVLA 走 primitive 路线；
- 与 [[DreamZero - 视频世界动作模型的零样本机器人策略]] 对比：DreamZero 把世界模型作为可执行 policy，PrimitiveVLA 把 VLM+LLM 显式模块作为可解释策略，是 hybrid 范式与 end-to-end 范式的两个方向；
- 与 [[GaussianDream - 机器人操控的前馈三维高斯世界模型]] 对比：都尝试给 VLA 加显式几何/语义结构，GaussianDream 用 3D Gaussian、PrimitiveVLA 用 primitive；
- 与 [[Qwen-VLA - 跨任务环境与机器人本体的统一动作建模]] 对比：都强调跨任务，但 Qwen-VLA 走"统一 action 空间 + 大底座"路线、PrimitiveVLA 走"显式 primitive 重组"路线。

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `paper.pdf` | 论文 PDF | https://arxiv.org/pdf/2605.28634v2 | 全文 | 2026-08-21 | 精读主来源 |
| `fig1-teaser.png` | 动机图 | https://arxiv.org/html/2605.28634v2/intro.png | Figure 1, p.1 | 已有 | 研究背景章节 |
| `fig2-method.png` | 方法框架图 | 论文 PDF 第 5 页裁剪 | Figure 2, p.5 | 2026-08-21 | 核心方法章节 |
| `fig4-segcompare.png` | 拆解对比图 | 已有 | Figure 4, p.7 | 已有 | Primitive Disassembly 章节 |
| `fig5-real.png` | 真实环境与任务 | 已有 | Figure 5, p.8 | 已有 | 数据集/实验设置章节 |
| `fig7-gen.png` | OOD 执行对比 | 已有 | Figure 7, p.10 | 已有 | 定性结果章节 |

- 论文 HTML：https://arxiv.org/html/2605.28634v2
- 论文摘要页：https://arxiv.org/abs/2605.28634
- 项目页：论文未披露
- 代码仓库：论文未披露
