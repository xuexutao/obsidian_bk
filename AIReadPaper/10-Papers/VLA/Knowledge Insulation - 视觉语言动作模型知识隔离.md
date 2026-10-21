---
type: paper-note
status: complete
domain: VLA
paper: "Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better"
year: 2025
arxiv: "2505.23705"
doi: null
source: "https://arxiv.org/abs/2505.23705"
project: "https://pi.website/research/knowledge_insulation"
code: null
tags:
  - VLA
  - knowledge-insulation
  - flow-matching
  - co-training
created: 2026-10-20
updated: 2026-10-20
---

# Knowledge Insulation：视觉语言动作模型知识隔离

## 基本信息

- **论文**：Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better
- **作者**：Danny Driess、Jost Tobias Springenberg、Brian Ichter、Lili Yu、Adrian Li-Bell、Karl Pertsch、Allen Z. Ren、Homer Walke、Quan Vuong、Lucy Xiaoyang Shi、Sergey Levine
- **机构**：Physical Intelligence
- **arXiv**：[2505.23705](https://arxiv.org/abs/2505.23705)，v1，2025-05-29
- **项目页**：[Physical Intelligence — Knowledge Insulation](https://pi.website/research/knowledge_insulation)
- **关键词**：knowledge insulation、stop-gradient、action expert、FAST、flow matching、VLM co-training

## 一句话结论

连续 action expert 虽让 VLA 推理快，却会把随机初始化模块的梯度灌入预训练 VLM，导致训练慢、语言遵循与知识迁移变差；Knowledge Insulation 用 FAST 离散动作继续训练 backbone，同时以 stop-gradient 隔离 300M flow action expert，使模型在 generalist table bussing 上约 **160k steps 达到 0.93**、而 π0 需约 **1.2M steps 才到 0.85**，即约 **7.5× 步数差**，并兼得快速推理与更强泛化。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

VLA 希望继承网页 VLM 的语言与视觉知识，又要输出高频、连续、精确动作。纯自回归动作 token 训练自然、知识迁移好，但 π0-FAST 在 RTX 4090 预测一秒动作块约需 750 ms，仅约 1.3 Hz；flow/diffusion action expert 可到约 10 Hz，却是随机初始化的新模块。其梯度在微调早期穿过注意力回传到 VLM backbone，可能破坏预训练表征。

论文提出的核心问题不是简单比较离散和连续动作谁更好，而是：**如何让 backbone 用稳定目标学习机器人表征，同时让连续 expert 学会快速动作，而不让后者污染前者？**

![](assets/Knowledge%20Insulation%20-%20视觉语言动作模型知识隔离/figures/01-problem.jpg)

> **图 2（论文原图）**：同一“把勺子放入 bin”任务中，π0 推理快但抓错物体，π0-FAST 遵循语言但完成用时 53 秒；知识隔离模型抓对目标且约 22 秒完成。下方表格将目标概括为 simultaneously follows language、fast training、fast inference。  
> **来源**：官方论文 Figure 2，第 5 页，[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)

### 1.2 为什么冻结 backbone 也不行

最直接的保护方法是冻结预训练 VLM，只训练动作 expert。但 VLM 从未在机器人本体状态、多视角接触和动作后果上预训练，固定特征不足以支持高性能策略；Items in Drawer 与 T-shirt Folding 中冻结版本接近 0。问题因此是“需要适应 backbone，但不能让不稳定 expert 梯度主导适应”。

### 1.3 本文核心贡献

1. 实证指出随机初始化连续动作模块会降低 VLA 收敛速度、语言遵循与 web knowledge transfer。
2. 提出单阶段联合训练：FAST token 负责 backbone 表示学习，flow expert 负责部署时连续动作。
3. 在跨 expert attention 中对 backbone key/value 使用 stop-gradient，实现单向信息流但双向参数隔离。
4. 系统比较 π0、π0-FAST、OpenVLA-OFT、Transfusion、HybridVLA、冻结、joint-training 等设计。
5. 在 7 类真机/模拟设置、12 种机器人配置与 VLM 共训练数据上验证性能、收敛和 OOD 迁移。

---

## 2. 任务定义与输入输出

### 2.1 VLA 任务

策略接收多视角图像 $I^{1:V}$、本体状态 $q\in\mathbb R^s$ 和语言指令 $\ell$，输出长度 $H$ 的动作块：

$$
a_{1:H}\sim\pi(\cdot\mid I^{1:V},q,\ell),\qquad a_i\in\mathbb R^d.
$$

模型同时维护两种动作表示：FAST 离散 token $y^{\ell,a}$ 只用于训练 backbone，连续动作 $a_{1:H}$ 由 action expert 生成并用于部署。

### 2.2 标准目标

自回归 VLA 的负对数似然为

$$
\mathcal L_{\mathrm{AR-VLA}}(\theta)
=\mathbb E_{(x,y)\sim D}\left[-\sum_{j=1}^{n-1}M_j\log p_\theta(y_{j+1}\mid x_{1:j})\right].
$$

flow VLA 对

$$
a_{1:H}^{\tau,\omega}=\tau a_{1:H}+(1-\tau)\omega,
\qquad \omega\sim\mathcal N(0,I)
$$

学习向量场：

$$
\mathcal L_{\mathrm{FLOW-VLA}}(\theta)
=\mathbb E_{D,\tau,\omega}
\left[\left\|\omega-a_{1:H}-f^a(a_{1:H}^{\tau,\omega})\right\|_2^2\right].
$$

---

## 3. 核心方法

### 3.1 总体架构

![](assets/Knowledge%20Insulation%20-%20视觉语言动作模型知识隔离/figures/02-architecture.png)

> **图 1（论文原图）**：3B VLM backbone 同时预测语言与 FAST 离散动作，获得稳定的自回归监督；300M action expert 读取 backbone 表征并预测连续动作，但黄色连接处 stop-gradient。训练保留两种动作表示，部署只运行较小的连续 expert。  
> **来源**：官方论文 Figure 1，第 2 页，[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)

完整配方包含三点：

1. **joint discrete/continuous action prediction**：同一 batch 同时学习 FAST 和 flow；
2. **VLM data co-training**：混入 caption、VQA、检测和 robot planning，持续保护语义能力；
3. **knowledge insulation**：切断 expert 对 backbone 的梯度，但保留 expert 对 backbone feature 的前向读取。

### 3.2 联合训练目标

$$
\mathcal L_{\mathrm{CO-VLA}}(\theta)=
\mathbb E_{D,\tau,\omega}\left[
-\sum_{j=1}^{n-1}M_j^\ell\log p_\theta(\hat\ell_{j+1}\mid x_{1:j})
+\alpha M^{\mathrm{act}}
\left\|\omega-a_{1:H}-f_\theta^a(a_{1:H}^{\tau,\omega})\right\|_2^2
\right].
$$

$\hat\ell$ 同时包含普通文本和 FAST 动作 token；$M^\ell$、$M^{\mathrm{act}}$ 允许一套模型混合纯 VLM、纯动作、语言+动作样本。FAST 与连续动作互相不可见，避免模型把答案从另一表示复制过来。

当 stop-gradient 使两组权重近似独立后，作者直接设 $\alpha=1$，不再需要精细平衡 expert 梯度对 backbone 的影响。

### 3.3 注意力中的知识隔离

设 backbone token 为 $X_b$，action expert token 为 $X_a$。注意力概率分块为

$$
P=
\begin{pmatrix}
P_{bb} & 0\\
P_{ab} & P_{aa}
\end{pmatrix}.
$$

backbone 不看 expert；expert 可看 backbone。关键改动是在 expert 查询 backbone 的 key/value 时停止梯度：

$$
P=\operatorname{softmax}\left(
\begin{pmatrix}
Q_b(X_b)K_b(X_b)^\top & 0\\
Q_a(X_a)\operatorname{sg}(K_b(X_b)^\top) & Q_a(X_a)K_a(X_a)^\top
\end{pmatrix}+A\right),
$$

$$
E_b=P_{bb}V_b(X_b),
$$

$$
E_a=P_{ab}\operatorname{sg}(V_b(X_b))+P_{aa}V_a(X_a).
$$

因此前向信息仍从 VLM 流向动作专家，expert 能利用图像、语言与 FAST 学到的控制表征；反向时 flow loss 只能更新 expert 自身。backbone 仍由 FAST 和 VLM next-token loss 适应机器人任务，不会被冻结。

### 3.4 FAST 作为表示学习目标

FAST 不在部署时解码，只提供高信息密度的训练信号。朴素逐时刻 token 也比只有 flow 更好，但因序列冗余，表示学习较弱；stride 5 子采样优于 dense naive token，FAST 最好。这个消融说明知识隔离不只是 stop-gradient，还需要 backbone 有足够强的替代监督，否则切断 flow 后 backbone 学不到动作相关信息。

### 3.5 VLM co-training 的作用

共训练数据包括 CapsFusion/COCO caption，Cambrian-7M/PixMo/VQAv2，室内与家居物体定位，以及 robot planning。它有两种作用：直接维持预训练任务能力；在 OOD 物体上把语义知识转入 action expert。实验显示 no VLM data 对普通任务平均只略有影响，却显著降低 OOD follow rate；对不做 stop-gradient 的 joint-training，VLM 数据还是防止灾难性干扰的重要缓冲。

### 3.6 模型细节

PaliGemma backbone 中语言部分约 2B，配置 width 2048、depth 18、MLP 16384、18 heads、1 KV head、head dim 256；action expert 300M，width 1024、MLP 4096。动作 horizon $H=50$。flow 时间采样为

$$
p(\tau)=\operatorname{Beta}\left(\frac{s-\tau}{s};1.5,1\right),\qquad s=0.999.
$$

图像、语言和 text state 使用 full prefix mask；FAST token 自回归注意 prefix 和先前 FAST token；action expert 注意 prefix 与自身，但不看 FAST，防止信息泄漏。

### 3.7 状态表示

论文比较三种 $q$：普通数字文本、每个 bin 的 special token、连续线性投影。方法对 text state 与 continuous state 都有效，special token 较差；π0 在两种主要状态表示下都仍弱，因此优势不能归因于状态编码不同。

---

## 4. 数据集与实验设置

### 4.1 任务与场景

真机任务包括静态单臂 Table Bussing、Items in Drawer，静态双臂 T-shirt Folding，移动双臂 Make Bed、Dish in Sink、Mobile Items in Drawer、Laundry in Basket；另有真实 DROID 与模拟 LIBERO。Items in Drawer、DROID 与移动任务在完全未见环境评测。

每个真机任务、每个策略 **10 episodes**，双侧 t-test 检验显著性。具体评分：

- Items in Drawer：开抽屉 1 分、每个物体入抽屉各 1 分、关抽屉 1 分，最高 5。
- T-shirt Folding：按方正和褶皱程度，最高 5。
- Table Bussing：12 个对象正确分类放置，最高 12。
- Make Bed：被子与两枕分阶段，最高 5。
- Dish in Sink：4 件餐具，每件拿起+放入，最高 8。
- Mobile Items in Drawer：拿起、开抽屉、放入、关抽屉，最高 4。
- Laundry in Basket：导航抓取、放到篮上/内、完全入篮，最高 3。

### 4.2 Generalist 数据

通用模型覆盖 **12 种机器人配置**：ARX、UR5、Franka 单臂；ARX、AgileX、Trossen、UR5 双臂；mobile Trossen、ARX slate、Galaxea G1、Hexmove H1、Fibocom 等移动双臂，并加入 OXE。任务远超评测集，如磨咖啡、挂毛巾等，环境包含办公室与真实家庭。

### 4.3 Baseline

- **π0**：continuous action expert，只用 robot data。
- **π0-FAST**：FAST 自回归 VLA，只用 robot data。
- **OpenVLA-OFT**：并行解码+双向注意力，本实验保留 text state、不用 FiLM。
- **Transfusion**：连续动作在同一 backbone 内去噪，只有动作投影新初始化。
- **HybridVLA**：同时 naive token 与 diffusion，离散 token 可看连续输入；实验版加 action expert。
- **joint-training**：与本文相同但不 stop-gradient。
- **joint-training w/o VLM data**：同时去掉隔离和 VLM 共训练。
- **frozen backbone**：仅训练 robotics-specific 参数。
- **naive token / stride-5 token**：替换 FAST 的表示学习消融。

---

## 5. 实验结果

### 5.1 训练速度

![](assets/Knowledge%20Insulation%20-%20视觉语言动作模型知识隔离/figures/03-training-curve.png)

> **图 6b（论文原图）**：Table Bussing generalist 的训练曲线。本文方法从 20k/40k/100k/160k steps 约为 0.21/0.48/0.82/0.93；π0-FAST 约为 0.27/0.52/0.79/0.90；π0 在 160k 仍约 0.29，直到约 1.2M steps 才到约 0.85。论文据此报告 π0 需要约 **7.5×** 更多训练步数达到相近表现。  
> **来源**：官方论文 Figure 6b，第 9 页，[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)

这张曲线验证了“训练时用 FAST、推理时用 flow”的互补性：本文几乎保留 π0-FAST 的收敛速度，却不承担其部署延迟。双目标使单步训练成本增加约 20%，但总 wall-clock 仍因步数减少而显著低于纯 flow π0。

### 5.2 真机性能与语言遵循

Items in Drawer 中本文方法同时在完成度与语言遵循上领先。π0 与无 stop-gradient joint-training 常抓错指定对象；π0-FAST 知道目标但动作慢、开抽屉精度不足；HybridVLA 因离散 token 可读取连续输入，表现明显受损；冻结 backbone 基本不可用。

Table Bussing 中本文最高、语言遵循好、推理快。π0-FAST 任务表现也好，但完成同任务 wall-clock 约为本文 **2 倍**（示例为 53 秒 vs. 22 秒）；OpenVLA-OFT 快且遵循语言，但总任务完成度最低；Transfusion 可用但慢于本文。

### 5.3 DROID 与 LIBERO 数字

DROID 未见环境平均 task-progress：

| 方法 | 得分 |
|---|---:|
| 本文 | **0.55 ± 0.09** |
| π0 | 0.49 ± 0.09 |
| π0-FAST | 0.45 ± 0.09 |

LIBERO success rate（%）：

| 方法 | Spatial | Object | Goal | Long/10 | LIBERO-90 |
|---|---:|---:|---:|---:|---:|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 未报告 |
| π0 | 96.8 | **98.8** | 95.8 | 85.2 | 未报告 |
| π0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 未报告 |
| 本文（from scratch） | 96.6 | 97.2 | 94.6 | 84.8 | 92.7 |
| 本文（from generalist） | **98.0** | 97.8 | 95.6 | 85.8 | **96.0** |

论文称 Spatial 与 LIBERO-90 达到新 SOTA，但在 LIBERO-10/Long 不如 OpenVLA-OFT，说明知识隔离并非所有仿真套件全面领先。

### 5.4 Stop-gradient 消融

Items in Drawer 与 language following 中，完整 stop-gradient 明显优于 joint-training。若大量 VLM data 同时共训练，无隔离 joint-training 也能部分恢复语言遵循，说明 VLM loss 能抵消一部分干扰；但 stop-gradient 更直接、稳定，并让 flow loss 权重无需与 backbone loss 精细平衡。

关键因果链为：action expert 随机初始化 → flow 梯度在注意力中进入 backbone → 语言/视觉表征偏移 → 策略更多依赖场景统计而忽略语言。冻结失败则说明解决方案不能只是“不要更新 backbone”。

### 5.5 VLM 数据与 OOD 泛化

移动机器人需把训练未见物体放入已打开抽屉。含 VLM data 的模型 OOD Follow Rate 最好；去掉 VLM 数据下降最大，说明 caption/VQA/定位知识能通过被保护的 backbone 前向传给动作 expert。普通 ID 动作完成对 VLM data 较不敏感，符合“网页数据提供物体语义，不提供接触动力学”的分工。

### 5.6 动作与状态表示消融

FAST 作为表示目标优于 dense naive token；stride 5 naive 比 dense naive 好，验证冗余 token 会稀释学习信号。text state 与 continuous state 都能在本文方法下工作，special-token state 更差；所以主要收益来自梯度路径与动作目标，而不是某个状态技巧。

### 5.7 失败案例与局限数字

语言遵循虽改善但不完美，训练数据中图像—动作相关性仍会诱导模型忽略文本。双动作输出令单步训练算力增加约 **20%**。纯自回归 π0-FAST 的一秒 chunk 约 **750 ms**，而 action expert 支持约 **10 Hz**；论文未给所有硬件上的统一毫秒表，不能把示例延迟泛化为绝对常数。

### 5.8 实验充分性判断

这是四篇中对 architecture choice 因果分析最完整的一篇：有强 baseline、梯度隔离、数据共训练、动作 token、状态表示、收敛曲线、OOD 与多任务。仍有两个证据缺口：没有直接测量 backbone 的通用 VLM benchmark 遗忘程度；“随机 expert 梯度破坏知识”主要由下游语言遵循间接支持，缺少梯度冲突角度或表征相似度分析。

---

## 6. 与相关工作的关系

π0 提供连续 action expert，但训练慢、语言遵循弱；π0-FAST 提供高信息离散目标，但推理慢；π0.5 先 FAST 预训练再 joint post-training，是两阶段方案。Knowledge Insulation 把这条路线改成单阶段，并用 stop-gradient 明确规定参数更新边界。与 Transfusion 在同一 backbone 去噪不同，它保留小 expert 的推理效率；与 HybridVLA 不同，它禁止两种动作表示互看，避免捷径学习。

---

## 7. 局限与批判性评价

1. **训练开销增加 20%**：尽管总收敛更快，显存与每 step 计算仍更高。
2. **语言偏置未消失**：隔离保护了知识，却没有从根本上打破机器人数据中“看图就猜默认动作”的 shortcut。
3. **backbone→expert 单向瓶颈**：expert 的连续控制误差不能反向改善 backbone；如果 FAST 目标遗漏细微动力学，backbone 可能学不到对 expert 最有用的表示。
4. **依赖双重动作标注一致**：同一动作要同时经过 FAST 和 flow，预处理、mask 或 padding 错误会引入隐蔽泄漏。
5. **私有 generalist 数据门槛**：12 种配置的大规模数据不可公开复刻，公开基准只验证部分结论。
6. **个人判断**：知识隔离本质上是多模态系统的“梯度 API 设计”。它提醒 VLA 架构不仅要设计前向信息流，还要显式设计哪些目标有权修改哪些预训练参数；这一原则可推广到触觉、力觉和世界模型模块。

---

## 8. 复现与实践建议

### 8.1 最小关键实验

在 LIBERO 或 DROID 训练四组：π0 flow-only、FAST-only、joint no-stop、joint stop-gradient。保持 backbone、batch、数据与总 step 一致，记录 task success、language counterfactual accuracy、每 step 时间、达到固定性能的 wall-clock、单 chunk latency。再加 no VLM data 测 OOD 物体，才能完整验证三项主张。

### 8.2 成本评估

- **模型**：2B/3B backbone + 300M expert，双目标提高约 20% 单步训练成本。
- **训练**：本文约 160k steps 达高性能；flow-only 对照可到 1.2M，完整复现算力仍高。
- **数据**：公开 LIBERO/DROID 可验证基础结论；generalist 12 embodiment 与真实家庭任务难复现。
- **真机**：至少单臂、双臂、移动双臂三类硬件与 10 episodes/task。
- **代码可用性**：PDF 未提供明确完整代码链接；stop-gradient attention、双动作 mask 和数据混合需自行实现，工程复杂度中高。

### 8.3 实现检查清单

1. FAST token 与 flow token 在 attention 中互不可见；
2. expert 对 backbone 的 K/V 前向可见，但两处都 `stop_gradient`；
3. VLM token 不得看 expert；
4. FAST loss 必须覆盖 robot action 样本，否则 backbone 没有动作适应信号；
5. 同时监控 AR loss、flow loss、language following 与 rollout，不能只看训练 loss；
6. 计算速度时分开报告单步训练成本、收敛 steps、总 wall-clock 和部署 latency。

---

## 9. 个人启发与后续问题

后续可研究软隔离：按层或按训练阶段允许小比例 expert 梯度；用 gradient surgery 而非完全 stop；测量 VLM benchmark 遗忘与 CKA 表征漂移；为触觉/力觉 expert 分别定义梯度权限；以及让 backbone 通过预测世界状态、接触事件等离散辅助目标获得比 FAST 更完整的控制表征。

---

## 10. 材料来源

- 官方论文页：[https://arxiv.org/abs/2505.23705](https://arxiv.org/abs/2505.23705)
- 官方 PDF：`assets/Knowledge Insulation - 视觉语言动作模型知识隔离/sources/paper.pdf`
- 项目页：[https://pi.website/research/knowledge_insulation](https://pi.website/research/knowledge_insulation)
- 本地来源清单：`assets/Knowledge Insulation - 视觉语言动作模型知识隔离/sources.md`
- 训练曲线数值按官方 Figure 6b 坐标读取并明确标注；DROID 与 LIBERO 数字来自正文和 Table 1。
