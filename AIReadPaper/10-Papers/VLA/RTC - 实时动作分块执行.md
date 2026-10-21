---
type: paper-note
status: complete
domain: VLA
paper: Real-Time Execution of Action Chunking Flow Policies
year: 2025
arxiv: 2506.07339
doi: 10.48550/arXiv.2506.07339
source: https://arxiv.org/abs/2506.07339
project: https://pi.website/research/real_time_chunking
code: null
tags:
  - VLA
  - Physical-Intelligence
  - Real-Time-Control
  - Action-Chunking
  - Flow-Matching
  - Inpainting
created: 2026-10-20
updated: 2026-10-20
---

# RTC：实时动作分块执行

## 基本信息

- **作者**：Kevin Black、Manuel Y. Galliker、Sergey Levine
- **机构**：Physical Intelligence、UC Berkeley
- **发表**：NeurIPS 2025；arXiv 2506.07339v2
- **项目页**：<https://pi.website/research/real_time_chunking>
- **原文 PDF**：<https://arxiv.org/pdf/2506.07339>
- **核心对象**：任意已经训练好的 diffusion / flow-based action-chunking policy；真机实验以 π0.5 为基础策略

## 一句话结论

RTC（Real-Time Chunking）把“模型生成下一段动作时，机器人仍在执行上一段动作”形式化为**动作序列 inpainting**：冻结因推理延迟而必然执行的前缀，对可修改的重叠区施加指数衰减软约束，再补全未来动作；它无需重训即可让 flow/diffusion VLA 异步、平滑地运行，在 12 个动态 Kinetix 环境和 6 个真机双臂任务上对延迟最鲁棒，但代价是每个去噪步都要反向传播，π0.5 推理从 76 ms 增至 97 ms。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

大模型控制机器人时，推理耗时不是“用户多等一会儿”，而是控制回路在现实世界继续演化期间暂时失去新决策。Action chunking 让策略一次预测未来 $H$ 个动作，只执行前 $s$ 个动作再重规划，缓解了高频控制与大模型低频推理的矛盾；但它仍有两个耦合问题：较大的 $s$ 使策略不够闭环、难以及时响应外界变化，较小的 $s$ 又会频繁切换由多模态动作分布采出的 chunk，造成跨 chunk 的策略跳变和抖动。

论文研究的是：**在模型单次推理慢于控制周期时，怎样一边执行旧 chunk，一边生成与已承诺动作兼容、又能吸收最新观测的新 chunk？**

### 1.2 现有执行方式的瓶颈

- **同步执行**：执行 $s$ 步后停住，等新 chunk 生成。位置控制机器人可以“保持”，但任务时长被推理暂停直接拉长；力/力矩控制或动态环境中甚至不可行。
- **Naive asynchronous**：后台生成，完成后立即切换。速度快，却可能从“绕障碍上方”的旧模式切到“绕下方”的新模式，产生离分布的瞬时跳变。
- **Temporal Ensembling（TE）**：平均多个重叠预测。平均不保证仍位于有效动作模态；两个可行但不同的策略取均值，可能恰好撞上障碍。
- **Bidirectional Decoding（BID）**：以 rejection sampling 选出与旧 chunk 一致的样本，但计算量大；在 π0.5 上批量采样的延迟高于 RTC。
- **仅加速模型**：蒸馏、并行解码或更少去噪步可以减少成本，却无法突破“至少一次前向传播”的下限；只要一次前向仍慢于控制周期，就仍需异步执行协议。

### 1.3 核心贡献

1. 提出纯推理期 RTC，不改模型、不改训练数据，直接适配 flow-based policy，并可把 diffusion policy 转为 flow 形式后使用。
2. 将异步 chunk 衔接写成 guided inpainting：硬冻结不可更改的动作前缀，并以 soft mask 约束其余重叠动作。
3. 构造 12 个动态 Kinetix 任务，系统考察执行 horizon、推理 delay 与跨 chunk 连续性的关系。
4. 在 6 个复杂真机任务、480 次试验、约 28 小时纯执行中验证吞吐与延迟鲁棒性，包括点燃火柴、插网线、叠衣和移动操作。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与时序变量

基础策略为

$$\pi(A_t\mid o_t),\qquad A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}],$$

其中 $o_t$ 是控制时刻观测，$H$ 是预测 horizon，$A_t$ 是未来动作 chunk。每次只执行其中 $s\le H$ 步，$s$ 为 execution horizon。模型推理耗时换算成控制步数记作 $d$：若在 $t$ 开始推理，新 chunk 到 $t+d$ 才可用，因此其前 $d$ 个时间位置已经由旧 chunk 执行，不能再修改。

RTC 的输入是最新观测 $o_t$、旧 chunk 尚未执行完的重叠部分 $A_{\mathrm{prev}}$、预计延迟 $d$ 和本轮执行 horizon $s$；输出是一个长度为 $H$ 的新 chunk。约束为 $d\le s\le H-d$：前 $d$ 步应与已承诺动作一致，中间 $H-d-s$ 步允许修正，最后 $s$ 步必须新生成。

### 2.2 基础 flow policy

条件 flow matching 策略从高斯噪声 $A_t^0\sim\mathcal N(0,I)$ 出发，用 $n$ 步 Euler 积分：

$$A_t^{\tau+1/n}=A_t^\tau+\frac{1}{n}v_\pi(A_t^\tau,o_t,\tau),\qquad \tau\in[0,1).$$

$v_\pi$ 是已训练的速度场。RTC 不改变它的参数，而是在每个积分步给速度场叠加一个保证跨 chunk 一致性的 guidance。

---

## 3. 核心方法

### 3.1 总体框架：把 chunk 衔接改写成 inpainting

RTC 的关键不是预测“完全新的未来”，而是求一个满足部分已知约束的条件样本：已知旧 chunk 中会被执行的动作，相当于图像修复中不可改的像素；未知的未来动作相当于待补区域。这样既能保留旧计划已经承诺的动力学连续性，又能让剩余动作根据新观测重算。

![](assets/RTC%20-%20实时动作分块执行/figures/01-overview.png)

> **图 1（原论文 Figure 3，第 4 页）：RTC 的时间轴与 soft mask。** 前 $d$ 步权重为 1，保证已经承诺的动作不被新采样破坏；重叠但尚未执行的中间区权重指数衰减；最后 $s$ 步没有旧动作可参照，权重为 0、完全新生成。来源：<https://arxiv.org/abs/2506.07339>

### 3.2 伪逆引导的推理期 inpainting

RTC 基于 pseudoinverse-guided diffusion/flow。先用当前中间状态估计最终去噪 chunk：

$$\widehat A_t^1=A_t^\tau+(1-\tau)v_\pi(A_t^\tau,o_t,\tau).$$

令 $Y$ 为从旧 chunk 取得并右侧补齐的目标序列，$W$ 为逐时间位置 mask。引导后的速度场为

$$v_{\Pi\mathrm{GDM}}=v_\pi+\min\left(\beta,\frac{1-\tau}{\tau r_\tau^2}\right)
(Y-\widehat A_t^1)^\top\operatorname{diag}(W)
\frac{\partial \widehat A_t^1}{\partial A_t^\tau},$$

$$r_\tau^2=\frac{(1-\tau)^2}{\tau^2+(1-\tau)^2}.$$

最后一项是 vector-Jacobian product，需要在每个去噪步反向传播。直觉上，$(Y-\widehat A^1)$ 衡量最终动作预测对旧计划的偏离，Jacobian 将这个终点误差拉回当前噪声空间，$W$ 决定各时间位置约束强度。作者额外加入上限 $\beta$；当 $\tau\to0$ 时理论 guidance 会发散，而控制策略常只做 5 步去噪，若不裁剪会数值不稳定。仿真消融后使用 $\beta=5$。

### 3.3 Soft masking：约束“策略”，而非只钉死前缀

只硬约束前 $d$ 步并不够。$d$ 小时，约束信号很短，新 chunk 仍可能在前缀之后迅速切换到另一种动作模态。RTC 因而对全部 $H-s$ 个重叠位置施加不同强度：

$$W_i=\begin{cases}
1,&i<d,\\
 c_i\dfrac{e^{c_i}-1}{e-1},&d\le i<H-s,\\
0,&i\ge H-s,
\end{cases}
\qquad
c_i=\frac{H-s-i}{H-s-d+1}.$$

越靠近当前时刻，越应该延续旧计划；越远的未来不确定性越大，越应允许最新观测改变动作。附录比较指数、线性、无衰减和立即衰减（hard mask），指数总体最好，线性很接近；Diffuser 式“每步直接覆盖已知位置”虽更便宜，也有收益，但弱于 gradient guidance。

### 3.4 异步运行时

系统由控制线程 `GETACTION` 与后台 `INFERENCELOOP` 组成。控制线程每 $\Delta t$ 更新观测并取一个动作；后台线程在达到最小执行 horizon 后读取旧 chunk 剩余部分，以近期延迟队列最大值保守估计下一次 $d$，释放锁后做 guided inference，完成即原子替换当前 chunk。实际 $s$ 随本次推理时间变化，取 $\max(d,s_{\min})$。

这套设计的重要工程属性是：即使延迟波动，也总有旧动作可供控制线程消费；新 chunk 一旦完成就切换，但它已被旧 chunk 约束，不需要额外插值或停顿。

### 3.5 平滑性与额外计算

![](assets/RTC%20-%20实时动作分块执行/figures/02-method.png)

> **图 2（原论文 Figure 1 下半部分，第 1 页）：点火任务前 10 秒一个肩关节的位置、速度与加速度。** RTC（绿）与 temporal ensembling（棕）都明显比同步执行（红）的边界尖峰更平滑；论文同时报告 RTC 完成同一动作约快 20%。来源：<https://arxiv.org/abs/2506.07339>

代价集中在反向 guidance：RTX 4090、bfloat16、5 步去噪时，vanilla π0.5 为 76 ms，RTC 为 97 ms；其中图像编码均为 18 ms、Gemma 2B prefill 均为 44 ms，而 5 个去噪步从 14 ms 增为 35 ms，即去噪部分约 2.5×。尽管单次推理更慢，异步执行消除了停顿，端到端任务仍更快。

---

## 4. 数据集与实验设置

### 4.1 动态 Kinetix 仿真

作者从 Kinetix 选择 10 个动态环境并新增 2 个，涵盖抛掷、接取、平衡和运动控制；使用 force control，因此不存在同步方法的“保持当前位置”。每个环境先以 RPO 训练 6 个不同随机种子的专家，每 episode 随机选专家，生成 **100 万 transition**；再训练 $H=8$、4 层 MLP-Mixer flow policy 32 epochs。动作加入高斯噪声以迫使闭环纠错。每个图点为 **2048 次 rollout**，测试 $d=0\ldots4$。

基线包括 naive async、TE、BID（batch 32、mode size 3、弱策略 8 epochs）、hard-mask RTC 与 Diffuser inpainting。评价是二元 solve rate。

### 4.2 真机任务

基础模型为 π0.5，双 6-DoF 机械臂 + 平行夹爪，位置控制，$H=50$、$\Delta t=20$ ms（50 Hz）、5 步去噪。测试 6 项任务：点蜡烛（5 子步）、插两端网线（6）、移动铺床（3）、叠衬衫（1）、批量叠衣（4）、移动收碗（8）。每种方法、任务、延迟各 10 次，共 **480 episodes、约 28 小时纯机器人执行**。

比较同步执行、稀疏 TE、稠密 TE 和 RTC；注入 0/100/200 ms 额外延迟。RTC 基础推理约 97 ms，加 LAN 后起始 $d\approx6$；注入后约为 $d\approx11/16$。指标同时看离散任务进度、成功率、用时，以及“完成比例/episode 时长”的平均吞吐（tasks/min）。

### 4.3 复现资源

仿真专家训练为 12 环境 × 6 seeds，约需 **4×H100、4 小时**；数据生成约 **6×H100、20 分钟**；每环境 flow imitation policy 约 **2×H100、1.5 小时**；2048 次评测约 **6×H100、5 分钟**。每个 π0.5 真机微调约 **8×H100、24 小时**，部署推理用单张 RTX 4090。

---

## 5. 实验结果

### 5.1 仿真主结果与 baseline

RTC 在固定 $s=\max(d,1)$ 下跨全部延迟都优于 naive async、TE、BID 和 hard masking。Figure 5 的曲线还显示：当 $d=1$、逐步缩短执行 horizon 时，只有 RTC 与 BID 能持续从更频繁的闭环更新中获益；naive async/TE 因 chunk 边界不连续，更新更频繁不等于更好。论文未给 Figure 5 的逐点数值表，因此不把图读数伪装成精确数字；可精确复核的是每点 2048 trials 与 95% Wilson 区间。

Soft masking 在低延迟、短 horizon 时贡献最大，因为此时硬前缀太短，最容易发生动作模态切换。BID 也能维持连续性，但算力更贵：π0.5 上 batch 16 时，BID 无 forward model 为 115 ms、共享 backbone 为 169 ms、完整版本 223 ms，均高于 RTC 的 97 ms。

### 5.2 真机吞吐与延迟鲁棒性

![](assets/RTC%20-%20实时动作分块执行/figures/03-results.png)

> **图 3（原论文 Figure 6 摘要，第 8 页）：真机累计进度与平均吞吐。** RTC 在 +0/+100/+200 ms 均保持约 0.8–0.9 tasks/min 的量级，曲线没有随注入延迟退化；同步执行随延迟从约 0.7 降至约 0.5；稀疏/稠密 TE 在 +100/+200 ms 因振荡触发机器人保护停机，吞吐为 0。图中数值仅按坐标读取，论文的精确结论是 RTC 在全部延迟下最高，并在 +100/+200 ms 达统计显著。来源：<https://arxiv.org/abs/2506.07339>

RTC 的收益不仅来自消除等待。把同步推理的停顿从时间轴扣除后，RTC 在多数任务仍更早完成更多子步骤，说明连续的 chunk 衔接减少了错误与重试。点火任务最精密且失败后不能重试，RTC 的最终分数优势最大；铺床也有明显提升，主要瓶颈是枕头操作。

### 5.3 消融结论

- **Soft vs. hard mask**：soft mask 在小 $d$ 时显著更好，证明“只匹配不可更改前缀”不足以保持动作策略连续。
- **权重日程**：指数衰减整体最好，线性接近；无衰减会过度锁死未来，立即衰减退化为 hard mask。
- **Guidance clipping**：较大的、未裁剪的 guidance 会导致 chunk 发散；$\beta=5$ 是仿真选出的保守值。
- **Guided vs. overwrite inpainting**：Diffuser 式直接覆盖更便宜但效果较差，说明利用网络 Jacobian 把终点约束传回去噪状态确有价值。
- **延迟开销**：RTC 的 21 ms 模型增量全部来自去噪反向传播；图像与 VLM 编码不变，是后续 Training-Time RTC 重点消除的成本。

### 5.4 结果是否支撑主张

实验较强之处是同时覆盖直接力控制的动态仿真和大模型真机位置控制，并人为扩大推理延迟验证趋势。局限是主真机结论来自单一 π0.5 系列、6 个双臂任务；吞吐曲线多于数值表，难以独立复算效应量。尽管如此，TE 在高延迟下直接触发保护停机、RTC 不退化，是比小幅均值提升更有说服力的系统级证据。

---

## 6. 与相关工作及系列关系

### 6.1 与 π0/π0.5 的关系

π0/π0.5 解决“如何用 flow matching VLA 生成高质量连续动作 chunk”，RTC 解决“如何在推理比控制周期慢时消费这些 chunk”。它不提升模型知识或任务分布，而是改变 runtime sampling 和调度，因此可以叠加在已训练 π0.5 上。

### 6.2 与 Training-Time RTC 的关系

后续 Training-Time RTC 把 RTC 的前缀条件移进训练：前缀输入干净动作、flow timestep 置 1，postfix 正常去噪且只在 postfix 算 loss。它保留相同异步 runtime 接口，却取消每步 VJP。RTC 更灵活、无需重训并支持 soft mask；Training-Time RTC 更便宜且高延迟更稳，但需要训练时匹配延迟分布。

### 6.3 与 π星0.6、RL Token 的关系

RTC 是**执行层**优化，不从在线经验学习；π星0.6/RECAP 和 RL Token 是**策略改进层**。三者可正交组合：先用 RECAP 或 RLT 提升任务策略，再用 RTC 让大模型策略无停顿执行。但在线 RL 数据分布会受到执行协议影响，若训练时用同步策略、部署时改 RTC，最好重新验证 critic/value 的状态分布。

---

## 7. 局限与批判性评价

1. **仅适用于 diffusion/flow policy**：离散自回归动作模型没有可直接利用的连续去噪轨迹与 Jacobian guidance。
2. **推理计算反而增加**：RTC 解决的是 wall-clock 执行与连续性，不是模型 FLOPs；边缘硬件上反向传播可能比论文 RTX 4090 环境更难承担。
3. **依赖延迟可被保守预测**：算法用近期 delay 最大值估计 $d$。突发网络抖动超过预算时，仍可能耗尽旧 chunk。
4. **约束可能抑制必要的急停/急转**：soft mask 假设近期旧计划仍值得延续；环境突然发生危险变化时，连续性与反应性可能冲突。
5. **真机本体与控制方式有限**：未验证腿式 locomotion、力控接触任务和安全约束密集的工业系统；作者也明确指出真实动态场景覆盖不足。
6. **对多模态策略的改善带有经验性**：指数 mask、$\beta$ 和 horizon 都是关键超参数，缺少针对稳定性或约束误差的理论界。

---

## 8. 复现与实践建议

### 8.1 最小复现路径

优先在 Kinetix 复现：训练 $H=8$ 的小型 flow policy，实现同步、naive async、hard mask 与 soft-mask RTC，对 $d=0\ldots4$、$s=\max(d,1)$ 评测。单元测试应检查：前 $d$ 步约束误差、chunk 边界速度/加速度、延迟突增时是否越界、控制线程是否始终有动作。

### 8.2 工程实现重点

- 控制线程与推理线程只通过短临界区交换观测、时间索引和 chunk，绝不能让反向传播持锁。
- delay 队列应按控制步而非毫秒记录，并用保守分位数或窗口最大值。
- 记录 position/velocity/acceleration 以及保护停机，而不只记录成功率。
- guidance 用 VJP，不显式构造 Jacobian；低精度部署需特别检查 $\tau\to0$ 附近的数值稳定性。
- 真机上线先限制 $d+s\le H$，若预算不足则降级到安全保持或停止，不应读取 chunk 之外的动作。

### 8.3 成本判断

若已有 flow policy，算法代码与仿真复现成本中等；高质量真机复现成本高，主要瓶颈是 π0.5 权重/训练基础设施、双臂平台、数百次安全评测和人工打分。论文未提供 RTC 官方代码链接，完整复现仍需按公式与伪代码自行实现。

---

## 9. 个人启发与后续问题

RTC 的重要启发是：大模型机器人控制不应只讨论“网络输出什么”，还要把**推理调度本身视为策略的一部分**。在物理系统中，动作已经发出后便成为条件，而不是可撤销的预测。值得继续研究的问题包括：能否把安全约束与急停信号加入 mask；能否学习延迟条件化的 soft mask；能否用 forward-mode/JVP、蒸馏或训练期条件化消除反向成本；以及在线 RL 是否可以直接优化“策略 + 异步执行器”的联合闭环。

---

## 10. 材料来源

本笔记基于官方 PDF 全文、论文内 Figure 1/3/6 与项目页信息撰写；本地材料逐项溯源见 [sources.md](assets/RTC%20-%20实时动作分块执行/sources.md)。论文未披露 RTC 的官方代码仓库，因此 frontmatter 中 `code: null`。
