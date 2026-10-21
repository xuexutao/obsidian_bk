---
type: paper-note
status: complete
domain: VLA
paper: "SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust Attention"
year: 2023
arxiv: 2312.01990
doi: null
source: https://arxiv.org/abs/2312.01990
project: null
code: null
tags:
  - VLA
  - Robotics-Transformer
  - Linear-Attention
  - Efficient-Inference
  - SARA
created: 2026-10-21
updated: 2026-10-21
---

# SARA-RT：机器人 Transformer 线性注意力加速

## 基本信息

- **作者**：Isabel Leal、Krzysztof Choromanski、Deepali Jain、Avinava Dubey、Jake Varley、Michael Ryoo、Yao Lu、Frederick Liu、Vikas Sindhwani、Quan Vuong、Tamas Sarlos、Ken Oslund、Karol Hausman、Kanishka Rao
- **机构**：Google
- **论文原题**：*SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust Attention*
- **arXiv**：2312.01990
- **核心对象**：RT-2 PaLI-X 5B 与 Point Cloud Transformer（PCT）
- **本地 PDF**：`assets/SARA-RT - 机器人Transformer线性注意力加速/sources/paper.pdf`

## 一句话结论

SARA-RT 用“up-training”把已训练好的 softmax 注意力替换成带可学习查询/键预处理的核化线性注意力，以少量下游微调逼近原注意力分布：在 196 个视觉 token 的 PaLI-X 5B RT-2 上，单帧版本将 TPU forward 从 53.2 ms 降到 45.7 ms（14%），平均成功率基本不变（65.8% 对 65.1%），配合三帧历史与向量动作表示后平均成功率升至 76.4%；在 800–4000 点的 PCT 上推理约稳定在 100 ms，真机抓取成功率还从 0.64 提到 0.75，说明线性注意力可作为既有机器人 Transformer 的后训练部署改造，而不是从头重训的替代架构。

---

## 1. 研究背景与问题定义

### 1.1 RT 规模化后的部署瓶颈

RT-1 只有 35M 参数，控制频率仍至多约 3 Hz；RT-2 扩展到 5B/55B 后依赖云 TPU 才能达到 1–5 Hz。注意力对序列长度 $L$ 的时间和空间复杂度为 $O(L^2)$，对高分辨率图像、多帧历史、长语言、点云和多传感器融合尤其不利。机器人与纯生成任务不同：控制延迟会直接降低闭环稳定性，且算力必须在机载、边缘或可靠网络条件下满足硬实时预算。

常规线性注意力通过随机特征近似 softmax，但高质量近似常需很大随机特征维度，只有 $L>4000$ 时才可能抵消投影开销；简单 ReLU/exp 特征在 $L\approx128$ 已快，却通常损失精度。SARA-RT 要解决的是：**能否从已有高质量 softmax 策略出发，把注意力后训练成适合较短机器人序列的线性形式，并尽量保持行为？**

### 1.2 贡献边界

SARA-RT 不是新的 VLA 语义模型，也不增加训练数据。它提供一种部署层改造，并分别在两种尺度验证：

- RT-2：序列约 196 token，模型 5B，证明短序列也可获益；
- PCT：800–4000 点，证明长点云序列可获得近常数推理时间。

论文还给出随机 exp 特征与 softmax 核的联系，以及 SARA 存在性/逼近误差分析。理论解释“为什么可逼近”，实验回答“是否能在真机保持质量”。

![](assets/SARA-RT%20-%20机器人Transformer线性注意力加速/figures/01-overview.png)

> **图 1：SARA-RT 的两类真机验证场景。** 左侧是 RT-2 的语言条件移动操作任务，右侧是点云抓取任务；两者分别代表约 196 个视觉 token 的大 VLA 与 800–4000 点的长序列 PCT。来源：论文 Figure 1，<https://arxiv.org/abs/2312.01990>。

### 1.3 核心贡献

1. 提出 Self-Adaptive Robust Attention，以可学习查询/键特征映射替代固定随机近似；
2. 提出 up-training，从已收敛 softmax checkpoint 出发转换注意力，无需从头训练策略；
3. 给出随机 exp 特征逼近 softmax 核的无偏性、集中界和矩阵一致逼近分析；
4. 在 RT-2 5B 与 PCT 两种序列长度、两种机器人任务上验证质量—效率权衡；
5. 证明线性注意力释放的预算可用于更多历史帧和更精细动作表示。

---

## 2. 注意力问题与输入输出

### 2.1 标准注意力

给定查询 $Q\in\mathbb R^{M\times d}$、键 $K\in\mathbb R^{N\times d}$、值 $V\in\mathbb R^{N\times d_v}$，标准 softmax 注意力为：

$$\mathrm{Attn}(Q,K,V)=D^{-1}\exp(QK^\top)V$$

其中 $D$ 是每行核和的对角归一化。显式计算 $QK^\top$ 需要 $O(MN)$ 时间与空间；自注意力 $M=N=L$ 时为 $O(L^2)$。当图像分辨率、帧数或点云点数增加时，注意力矩阵快速成为延迟与内存瓶颈。

### 2.2 核化线性注意力

若相似度核可写成：

$$\mathcal K(x,y)=\mathbb E[\phi(x)^\top\phi(y)]$$

则单个查询的注意力可重排：

$$\tilde a_i=\frac{\phi(q_i)^\top\left(\sum_{j=1}^{N}\phi(k_j)v_j^\top\right)}{\phi(q_i)^\top\left(\sum_{j=1}^{N}\phi(k_j)\right)}$$

先聚合键和值，再处理所有查询，复杂度从 $O(MN)$ 变为关于 $M+N$ 的线性形式（忽略特征维度常数）。问题在于 $\phi$ 若太小会近似差，太大又失去速度优势。对于机器人常见的 128–1000 token，随机特征法的常数开销尤其敏感。

### 2.3 机器人任务中的输入输出

PCT 输入单物体点云 $N\times3$、物体中心和主轴，输出相对抓取姿态：指尖位置、接近方向和腕部 roll；在 Kuka IIWA + Weiss 夹爪上开环执行。RT-2 输入图像/图像历史与语言，输出 6-DoF 位姿增量、夹爪和终止信号。SARA 不改变这些任务接口，只替换内部注意力，因此可直接比较替换前后的策略成功率。

---

## 3. 核心方法：Self-Adaptive Robust Attention

### 3.1 从简单线性特征到随机预处理

简单线性特征写作 $\phi_f(z)=(f(z_1),\dots,f(z_d))$，$f$ 可取 ReLU 或 exp。论文先用 CLIP 零样本视觉语言导航建立直觉：直接 ReLU 注意力分布过平，机器人撞墙；exp 虽能到达目标，却会被无关物体干扰。将输入先乘高维高斯矩阵 $G$：

$$\phi_f^{\mathrm{rand}}(z)=f(Gz),\qquad G_{ij}\sim\mathcal N(0,1)$$

注意力会更尖锐，但示例使用 $m=2048$，对不到 200 个 token 的 RT-2 不划算。随机映射没有利用下游任务数据，只能用较多特征降低方差。

### 3.2 可学习预处理

SARA 的核心是把 $G$ 学出来，并允许查询、键采用不同映射：

$$\phi^{\mathrm{SARA}}_{f,1}(z)=v\odot f(G_Qz),\qquad \phi^{\mathrm{SARA}}_{f,2}(z)=v\odot f(G_Kz)$$

其中 $v\in\mathbb R^m$、$G_Q,G_K\in\mathbb R^{m\times d}$ 可学习，$\odot$ 为 Hadamard 积。实践中可取 $m=d_{QK}$，无需 2048 级随机特征，因此在 $L<200$ 也可能加速。

![](assets/SARA-RT%20-%20机器人Transformer线性注意力加速/figures/02-attention-navigation.png)

> **图 2：零样本导航中的注意力形态。** 普通 ReLU 过平、exp 会短暂受干扰；随机高斯预处理改善聚焦；训练后的 SARA-ReLU 用 $m=d$ 即可产生与 softmax 近似的尖锐注意力并走相同路线。来源：论文 Figure 2，p.2，arXiv:2312.01990。

图中 SARA 每个新视图都能模仿 softmax 的高峰 patch。这个导航例子不是主要机器人实验，而是解释为何“简单 ReLU + 学习旋转/缩放”可能比固定 ReLU 强。

### 3.3 up-training

“up-training”从已有 softmax checkpoint 开始：

1. 将目标层 softmax 注意力替换成核化线性注意力；
2. 复用原网络其余参数；
3. 在原下游机器人数据上短程微调，使查询/键投影适应新特征映射；
4. 验证成功率和延迟后部署。

在 Transformer 中，学习 $G_Q,G_K$ 可与重新微调 $W_Q,W_K$ 联系起来理解。论文 PCT 测试 $f\in\{\mathrm{ReLU},\exp,x^2\}$，RT-2 采用最简单的 ReLU 与全 1 向量 $v$。SARA 因从已收敛策略开始，不必承担从头学习视觉、语言和动作表示的风险；这对真机数据昂贵的场景尤为重要。

### 3.4 理论联系

对于等长、归一化输入 $\lVert x\rVert=\lVert y\rVert=r$，随机 exp 特征给出 softmax 核的无偏估计（差固定常数）：

$$m\exp(r^2)\mathcal K(x,y)=\mathbb E[(\phi_{\exp}^{\mathrm{rand}}(x))^\top\phi_{\exp}^{\mathrm{rand}}(y)]$$

论文进一步用方差与 Chebyshev 不等式给出集中界。主定理考虑注意力矩阵 $A=[\mathcal K(q_i,k_j)]$，令 $\tau=\min_{ij}\mathcal K(q_i,k_j)$、$\rho=\max_{ij}\mathcal K(q_i,k_j)$，证明存在 $v,G_1,G_2,f$ 使近似矩阵满足：

$$\lVert A-\hat A\rVert_\infty\le\delta$$

所需随机特征数含 $\log(2MN)$ 项。其意义是训练式预处理有能力用相对少的参数逼近完整注意力；但这是存在性结论，不保证实际非凸 up-training 总能找到最优映射，也不直接给出闭环策略误差或安全界。

### 3.5 SARA-PCT

PCT 有 2 层 Transformer、16 维潜表示，点数常超过 1000。先训练普通 PCT 1000 iterations，再以不同 $f$ up-train。三种 SARA 曲线几乎立即保持高奖励，说明替换后的适应阶段短。

![](assets/SARA-RT%20-%20机器人Transformer线性注意力加速/figures/03-pct-uptraining.png)

> **图 3：PCT 与 SARA-PCT 的训练曲线。** 左上是从头训练普通 PCT；其他三图放大高奖励区间，显示 ReLU、exp、平方特征从普通 checkpoint 出发后快速适应。来源：论文 Figure 4，p.4。

### 3.6 SARA-RT-2 注入位置

PaLI-X 中至少有两个候选位置：每帧 ViT 编码器，以及把所有帧和文本 token 拼接后的 fuser。论文只替换 **ViT 自注意力**，fuser 保持 softmax，因而速度收益是局部而非整个 RT-2 的理论线性化。

![](assets/SARA-RT%20-%20机器人Transformer线性注意力加速/figures/04-sara-rt2-architecture.png)

> **图 4：三帧 SARA-ViT 与 fuser。** 每帧由 sViT 编码，文本由 TT 编码，再在 fuser 进行完整交互；论文只在线性化 ViT，fuser 留作未来工作。来源：论文 Figure 7，p.6。

---

## 4. 数据与实验设置

### 4.1 PCT 抓取

仿真中训练对象只有可乐罐、水瓶、黑板擦、香蕉和章鱼软玩具；测试包含训练未见形状。点云由 RealSense 获取、滤除桌面外点并按对象聚类。优化采用黑盒 BGS：50 个扰动方向、平滑参数 0.02、步长 0.02、保留前 30% 方向。仿真训练后直接上 Kuka IIWA，无需额外真机微调。

A/B 真机测试共 200 个随机物体配置，每次随机选择普通 PCT 或 SARA-PCT，抓取成功为二元奖励。速度测试对 0–5000 点、多随机种子测 encoder latency。由于动作开环，感知编码质量对抓取姿态影响直接。

### 4.2 RT-2 机器人任务

使用 PaLI-X 5B RT-2，在 RT-1 真实专家数据上微调。任务为 pick、knock、open/close drawer、drawer place、upright、move，以及测试泛化的 diverse pick。普通与 SARA 单帧版都使用 196 个视觉 token；第三个版本额外使用 3 帧历史和新的向量动作文本表示。

传统动作表示是每维 256-bin 整数 token。作者提出向量表示：连续动作四舍五入到小数点后四位，再作为普通文本 token 序列。论文观察向量表示质量更高，但它与三帧历史一起变更，未提供独立正交消融，所以不能判断 12%+ 提升各自贡献。

### 4.3 评价指标与 trial

RT-2 报告每类任务成功率与 TPU forward latency；PCT 报告 A/B 抓取成功率和随点数变化的 encoder 时间。RT-2 各任务 trial 数为 pick 36、knock 35、drawer open/close 18、drawer place 36、upright 35、move 48、diverse pick 21。单类样本仍不大，因此应重点看整体平均和大幅差异。

---

## 5. 实验结果与消融

### 5.1 PCT：质量不降、长序列延迟近常数

普通 PCT 真机平均奖励为 0.64，SARA-ReLU 为 0.75。由于是 200 次随机 A/B 测试，结果支持 SARA 至少没有因近似造成明显质量损失，甚至可能因 up-training 正则化或训练波动而提升。不过论文未给置信区间和完全相同配置的配对统计，0.11 增益不应直接归因于线性注意力本身。

速度上，SARA-PCT encoder 在点云从数百增长到 4000+ 时约保持 100 ms；普通 PCT 随序列增长显著变慢，注意力持续是瓶颈。这里是 SARA 最符合理论优势的场景，因为 $L$ 足够大，线性重排可压倒映射常数开销。

### 5.2 RT-2 单帧公平比较

| 模型 | Pick | Knock | Drawer 开关 | Drawer place | Upright | Move | Diverse pick | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-2 5B，softmax，单帧，动作 token | 81 | 86 | 67 | 39 | 57 | 98 | 33 | 65.8 |
| SARA-RT-2 5B，单帧，动作 token | 83 | 91 | 78 | 31 | 46 | 79 | 48 | 65.1 |
| SARA-RT-2 5B，三帧，向量动作 | **100** | **91** | **89** | **56** | 51 | 81 | **67** | **76.4** |

前两行是最干净的注意力消融：平均 65.8 与 65.1 基本相等。SARA 在 pick、knock、drawer 开关、diverse pick 更好，在 drawer place、upright、move 更差。它不是逐任务无损，但总体保持质量。

延迟从 53.2 ms 降至 45.7 ms：

$$\frac{53.2-45.7}{53.2}=14.1\%$$

这只是模型 forward 的局部加速，不等于端到端控制频率提高 14%，因为网络、预处理、解码和机器人执行仍占时间。196 token 已能获益，说明学习特征避免了随机线性注意力只适用于超长序列的常见困境。

### 5.3 三帧 + 向量动作组合

第三行平均 76.4%，比原 RT-2 的 65.8 高 10.6 点，相对提升约 16%；论文概括为 12%+ mean accuracy improvement。Pick 为 100%，diverse pick 从 33 提到 67。三帧提供运动信息，向量文本可能减少离散动作表示限制，SARA 抵消新增帧带来的视觉注意力成本。

但这不是严格单变量消融：历史、动作 tokenizer 和 SARA 同时存在。最稳妥结论是“线性注意力释放了使用更多帧和新表示的计算预算，组合系统更强”，而不是“SARA 单独提高 12%”。

### 5.4 高分辨率趋势

CPU 速度图显示，随着图像分辨率、即 patch 数增长，普通 PaLI ViT 越来越受注意力瓶颈限制，SARA 增长更慢。作者计划利用这一点提高 RT-2 输入分辨率。论文未报告高分辨率下的真机成功率，因此这里只能视为性能潜力，不是已证明的视觉精度收益。

### 5.5 消融总结

1. 简单 ReLU 线性注意力直接替换会失败，但学习预处理后可恢复尖锐注意力；
2. PCT 的 ReLU/exp/平方三种 up-training 都迅速适应，说明方法不依赖唯一激活；
3. RT-2 公平单帧比较平均基本无损，且 forward 加速 14%；
4. SARA 只替换 ViT，尚未覆盖 fuser，实际加速上限未被探索；
5. 三帧历史和向量动作的质量增益与 SARA 耦合，缺少因子化消融。

---

## 6. 与 Google RT 系列的关系

### 6.1 对 RT-2 部署局限的直接回应

[[RT-2 - Web知识迁移视觉语言动作模型]] 把 55B VLM 用于低层控制，但承认计算成本会限制高频控制。SARA-RT 是系列中少数直接处理注意力复杂度的工作：不改变 RT-2 的 Web 知识和机器人数据，只在 checkpoint 后做部署优化。

### 6.2 与 RT-1、RT-X、RT-H 的潜在连接

RT-1 已用 TokenLearner 压缩视觉 token，是“先压长度”；SARA 是“改变注意力计算”，两者可组合。[[RT-X - 跨本体机器人数据与模型]] 和 [[RT-H - 语言动作层级与干预学习]] 都扩大了模型负担：前者需要更大容量和历史帧，后者每步有语言动作与动作两次查询。SARA 类后训练线性化可能降低这些系统的部署成本，但论文没有直接实验。

### 6.3 与高频动作生成的关系

Google RT 这一阶段多为 1–5 Hz 末端控制；后续 VLA 使用动作块、连续生成或更高相机频率，序列和实时要求更高。SARA 的价值会随历史长度、分辨率和传感器 token 增长，但若瓶颈转向巨大 MLP/解码器或网络服务，单独线性化 ViT 收益会受限。

---

## 7. 局限与批判性评价

### 7.1 实验局限

1. RT-2 只测 5B，不含 55B；
2. 只替换 ViT，不替换 fuser/语言解码器；
3. RT-2 序列 196，14% 加速尚属中等；
4. 第三行同时改变历史和动作表示，无法拆分贡献；
5. 没有模型总吞吐、云网络尾延迟、内存峰值和能耗；
6. 理论注意力矩阵误差不直接保证闭环策略稳定性。

### 7.2 适用边界

若模型序列短、MLP 或解码占主导，SARA 收益可能小于工程复杂度；若任务对少数精确 token 关系极敏感，近似误差可能造成稀有但危险的动作。对机器人部署，平均成功率相同并不足够，还需测最坏延迟、罕见失败、安全约束和长时漂移。

### 7.3 方法价值判断

SARA-RT 最有价值的不是声称“线性注意力更好”，而是提供**从已训练机器人策略迁移到高效注意力**的路径。机器人数据昂贵，从头训练高效骨干风险高；checkpoint up-training 能复用既有资产，并让效率优化和能力训练解耦。其工程思路比单个 14% 数字更可迁移。

---

## 8. 复现成本与实践建议

### 8.1 复现成本

PCT 部分相对可复现：网络小、仿真训练、点云和 Kuka 真机；RT-2 部分较难，需要 PaLI-X 5B 权重、机器人数据和 TPU。论文未披露开源代码项目，完整复现受权重与内部基础设施限制。

### 8.2 最小实现步骤

1. 保存原 softmax checkpoint 和每层注意力输出基准；
2. 只替换一个候选模块，保持动作 tokenizer、历史和训练 batch 不变；
3. 初始化 SARA 后以短程 up-training 对齐下游损失，可额外监控注意力矩阵差异；
4. 用同一真机场景做 A/B，而非只比较离线 action MSE；
5. 同时记录端到端 P50/P95/P99 latency、显存和功耗；
6. 单独消融 $f$、特征维度 $m$、替换层数、ViT/fuser 注入点；
7. 增加历史或分辨率时，用成功率—延迟 Pareto 曲线报告，而不是只给最佳组合。

---

## 9. 个人启发与后续问题

SARA-RT 提醒我们：VLA 的“规模化”不仅是参数和数据规模，也包括部署复杂度。最实用的研究方向可能不是重新训练一个更小模型，而是把已经昂贵训练好的大策略转换成更高效的等价近似。类似思想可延伸到蒸馏、量化、KV 缓存、token pruning 和异步层级查询。

后续问题包括：能否把 fuser 和语言解码器一并线性化？怎样根据每层注意力敏感度选择替换层？注意力矩阵逼近误差能否转成动作分布或闭环稳定性界？对 RT-H 的双查询、RT-X 的多帧和高分辨率、AutoRT 的多机器人服务，SARA 能否带来端到端吞吐提升而非局部 kernel 加速？

---

## 10. 材料来源

详见 `assets/SARA-RT - 机器人Transformer线性注意力加速/sources.md`。本文公式、PCT 设置、RT-2 表格、53.2/45.7 ms、0.64/0.75 抓取结果及方法边界均逐页核对本地官方 PDF；四张图由 arXiv 官方源码提取并逐张检查。
