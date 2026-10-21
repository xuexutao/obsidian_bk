---
type: paper-note
status: complete
domain: VLA
paper: "FAST: Efficient Action Tokenization for Vision-Language-Action Models"
year: 2025
arxiv: "2501.09747"
doi: null
source: "https://arxiv.org/abs/2501.09747"
project: "https://pi.website/research/fast"
code: null
tags:
  - VLA
  - action-tokenization
  - autoregressive-policy
  - robot-learning
created: 2026-10-20
updated: 2026-10-20
---

# FAST：高频机器人动作压缩与 π0-FAST

## 基本信息

- **论文**：FAST: Efficient Action Tokenization for Vision-Language-Action Models
- **作者**：Karl Pertsch、Kyle Stachowicz、Brian Ichter、Danny Driess、Suraj Nair、Quan Vuong、Oier Mees、Chelsea Finn、Sergey Levine
- **机构**：Physical Intelligence、UC Berkeley、Stanford
- **arXiv**：[2501.09747](https://arxiv.org/abs/2501.09747)，v1，2025-01-16
- **项目页**：[Physical Intelligence — FAST](https://pi.website/research/fast)
- **关键词**：动作 tokenization、DCT、BPE、自回归 VLA、高频控制、π0-FAST

## 一句话结论

FAST 先用离散余弦变换把高频动作块压缩到频域，再以 BPE 消除稀疏系数中的冗余，使自回归 VLA 不再靠“复制上一个动作 token”获得虚假的低损失；它把 50 Hz 双臂一秒动作块从 700 个 token 压到约 53 个，并让 π0-FAST 在 10k 小时、903M 时间步的通用机器人数据上以约 **1/5 训练算力**达到扩散 π0 的总体水平，但推理仍约慢 **7.5 倍**。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

论文关注的不是“Transformer 能不能预测机器人动作”，而是一个更基础的表示问题：连续动作序列 $a_{1:H}$ 应如何变成适合 next-token prediction 的离散符号。目标策略为

$$
\pi(a_{1:H}\mid o),
$$

其中 $o$ 包含图像、语言指令和机器人状态，$a_{1:H}$ 是未来 $H$ 步动作块。动作 tokenizer 定义映射

$$
T_a:a_{1:H}\rightarrow[T_1,\ldots,T_n],\qquad T_i\in\{1,\ldots,|V|\}.
$$

关键不在于重建单个动作，而在于 token 序列是否为自回归学习提供足够的**边际信息量**。高频控制下相邻动作极其相似，逐维逐时刻量化会产生很长、强相关的序列；给定 $T_{1:i-1}$ 后，$T_i$ 的新增信息接近零，模型只复制最近 token 也能得到较低训练损失，却学不到由视觉与语言决定的轨迹形状。

### 1.2 现有方法的瓶颈

RT-2、OpenVLA 等方法通常把每个时间步、每个动作维度独立分入 256 个区间。对 $D$ 维、长度 $H$ 的动作块，它产生

$$
T_a(a_{1:H})=[T_{1,1},\ldots,T_{1,D},\ldots,T_{H,1},\ldots,T_{H,D}],
$$

总长度为 $H\cdot D$。这一表示在 BridgeV2 等 5 Hz 数据上尚可，在 20–50 Hz 的精细操控中则同时造成：目标序列过长、token 高度冗余、推理串行步数过多、训练容易落入复制局部最优。

论文用三次样条插值玩具实验隔离了这一因素：底层曲线分布不变，只把采样点从 25 增到 800。朴素 tokenization 的预测 MSE 随频率急剧恶化，最后退化为复制第一个动作；FAST 在各采样率下保持低误差。这个实验说明问题来自表示与学习目标，而不是高频数据本身更难。

![](assets/FAST%20-%20高频机器人动作压缩与π0-FAST/figures/01-teaser.png)

> **图 2（论文原图）**：左侧展示 FAST 将高频动作经 DCT 压缩后交给 VLA；右侧显示控制频率从 2 Hz 升至 20 Hz 时，OpenVLA-style 逐步量化得分快速跌至接近 0，而 FAST 随频率提高仍保持较高得分。它直观验证了“动作频率越高，先压缩越必要”。  
> **来源**：官方论文 Figure 2，第 2 页，[arXiv:2501.09747](https://arxiv.org/abs/2501.09747)

### 1.3 本文核心贡献

1. 从 next-token prediction 的信息量角度解释高频动作 tokenization 的失败，而不是把失败归因于模型容量或数据难度。
2. 提出无需神经网络编码器的 FAST：DCT、量化、低频优先展平和 BPE 四步完成高保真动作压缩。
3. 训练 FAST+ 通用 tokenizer，使用约 100 万个一秒真实机器人动作块，覆盖单臂、双臂、移动平台、关节/末端动作空间与多种频率。
4. 在 π0 与 OpenVLA 两种 backbone、7 个评测环境上验证方法，并首次让 DROID 上训练的语言条件策略直接零样本部署到未见场景。
5. 将 π0-FAST 扩展到 10k 小时通用数据，性能匹配扩散 π0，训练 GPU 小时减少约 5 倍。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

FAST 的输入是一秒动作块 $A\in\mathbb{R}^{H\times D}$。论文假设真实机器人动作在时间上大体平滑，因此能量集中于低频 DCT 系数；突变仍可由高频系数表达，只是需要更多 token。输出是长度可变的离散 token 序列，加入已有 VLM 词表后可直接用标准自回归交叉熵训练。

策略输入包括 2–3 路 $224\times224$ 图像、自然语言任务指令和本体状态；输出为下一秒动作块。π0/OpenVLA backbone 全量微调，不冻结视觉或语言权重。本体状态本身只出现一次，不存在未来时间冗余，因此仍可用 256-bin 文本化，而动作输出需要 FAST。

### 2.2 关键符号和目标函数

对第 $i$ 个动作维度，DCT 产生频率系数

$$
C_j^i=\operatorname{DCT}(a_{1:H}^i),
$$

再以尺度 $\gamma$ 量化：

$$
\bar C_j^i=\operatorname{round}(\gamma C_j^i).
$$

$\gamma$ 控制压缩—保真权衡：越大，保留的小系数越多、重建误差越低、token 越长。量化矩阵按频率优先展平后交给 BPE，最终模型最小化动作 token 的自回归负对数似然：

$$
\mathcal L_{\mathrm{AR}}=-\sum_{k=1}^{n}\log p_\theta(T_k\mid o,T_{<k}).
$$

与 VQ/FSQ 相比，DCT 部分是解析可逆变换；唯一需要拟合的是 BPE 词表。

---

## 3. 核心方法

### 3.1 总体框架

FAST 的关键是把“逐时刻预测动作”改成“从低频到高频预测轨迹形状”。完整 pipeline 为：

1. 以训练集每维第 1/99 百分位把动作归一化到 $[-1,1]$；
2. 对每个动作维度沿时间轴独立执行 DCT；
3. 乘 $\gamma$ 后四舍五入，形成大量零元素的稀疏频率矩阵；
4. 按频率列优先展平，使所有维度的低频系数先出现；
5. 用 BPE 合并重复零串和常见跨维系数组合，输出固定词表中的可变长 token；
6. 推理时逆 BPE、反量化、IDCT，并反归一化为连续动作块。

![](assets/FAST%20-%20高频机器人动作压缩与π0-FAST/figures/02-method.png)

> **图 4（论文原图）**：两维归一化轨迹先被 DCT 分解为频率系数，量化后得到稀疏矩阵；低频优先展平保证模型先决定整体轨迹，BPE 再把零串与常见系数组合压成少量 token。这里 BPE 是无损压缩，信息损失只来自 DCT 系数量化。  
> **来源**：官方论文 Figure 4，第 5 页，[arXiv:2501.09747](https://arxiv.org/abs/2501.09747)

### 3.2 DCT 为什么适合机器人动作

DCT 把平滑时间序列表示为余弦基的线性组合。低频项描述整体位移和弯曲趋势，高频项描述快速修正或突变。机器人伺服数据虽然采样频率高，但有效自由度并不会随采样率线性增长，因此“每秒约固定数量的显著频率项”比“每个采样点一个 token”更接近信号固有复杂度。

这一判断得到 Table I 支持。相同的一秒动作块、相近重建误差下：

| 数据 | 动作维度 | 频率 | 朴素 token | FAST token | 压缩比 |
|---|---:|---:|---:|---:|---:|
| BridgeV2 | 7 | 5 Hz | 35 | 20 | 1.75× |
| DROID | 7 | 15 Hz | 105 | 29 | 3.6× |
| Table Bussing | 7 | 20 Hz | 140 | 28 | 5.0× |
| T-Shirt Folding | 14 | 50 Hz | 700 | 53 | 13.2× |

单臂任务基本稳定在约 30 token/秒，双臂约 60 token/秒；这正是“表示信号复杂度而不是采样点数”的证据。

### 3.3 低频优先展平

DCT 矩阵可按动作维度优先或频率优先展平。FAST 选择后者：先输出所有动作维度的最低频系数，再逐渐输出高频系数。自回归模型因此先确定动作块的全局方向与幅度，后续 token 只补充细节。若按维度优先，模型会在完整知道某一维高频细节后才看到另一维的整体趋势，跨维协调更差，rollout 稳定性也更弱。

### 3.4 BPE 的角色

DCT+量化已经把能量集中到少数非零系数，但展平后仍有大量显式 0。BPE 将高频出现的整数子序列合并：一方面压缩零串，另一方面学习常见的跨维动作模式。默认超参数为 $\gamma=10$、BPE 词表 1024；论文称两者不敏感，单数据集实验统一使用同一设置。

消融表明，去掉 BPE 后仍优于朴素量化，因为 DCT 已提升每个前部 token 的信息量；但 Table Bussing 与 T-Shirt Folding 的 rollout 都下降，而且必须串行预测数百个零 token，推理明显变慢。因此 BPE 不是原理成立的唯一条件，却是把频域表示变成实用动作语言的关键步骤。

### 3.5 FAST+ 通用 tokenizer

FAST+ 的 BPE 词表来自约 100 万个一秒动作块。训练混合覆盖 ARX、AgileX、Franka、UR5、Trossen、ALOHA、DROID、BridgeV2、Open X-Embodiment 等，包含单臂、双臂、移动机器人，以及 joint、end-effector 和 camera-frame 控制；动作统一 padding 到 32 维。

在 tokenizer 从未见过的机器人数据上，FAST+ 对所有测试集都至少实现 2× 压缩，在部分高频平台上超过 10×；用于策略训练时又几乎匹配每个数据集单独拟合的 FAST。它因此把“为每个数据集训练动作 tokenizer”降为一次性的黑盒预处理。

### 3.6 π0-FAST 的训练与推理

论文把 FAST token 覆盖到 VLM 词表中最少使用的 token，直接全量微调 PaliGemma-3B 驱动的 π0，或 Prismatic-7B 驱动的 OpenVLA。优化设置为：1k steps 线性 warmup，之后固定学习率 $5\times10^{-5}$；AdamW，$\beta_1=0.9,\beta_2=0.95$，无 weight decay；梯度范数裁剪 1；EMA 0.999。推理采用 greedy autoregressive decoding。

这种设计训练快，却把成本移到推理：π0-FAST 通常需自回归生成 30–60 个 token，并调用完整约 2B 语言 backbone；扩散 π0 只需 10 个 denoising steps，且动作专家约 300M 参数。因此 RTX 4090 上一秒动作块约为 **750 ms vs. 100 ms**。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

评测含 7 个环境：LIBERO 四套模拟基准的平均结果，以及 6 个真实机器人场景中的 Table Bussing（20 Hz）、T-Shirt Folding（50 Hz）、Grocery Bagging（20 Hz）、Toast out of Toaster（50 Hz）、Laundry Folding（50 Hz）、DROID tabletop manipulation（15 Hz）。后三个复杂任务主要用于通用策略比较。

DROID 使用 75k 个成功 episode、21M 样本，训练 240k iterations，约等于遍历 3 个 epoch；量化测试含 16 个任务、每模型共 44 次 trial，并在 Berkeley、Stanford、University of Washington 三地做未计分的定性零样本部署。

通用 π0-FAST 使用与 π0 相同的跨具身混合：903M 自有时间步，另有 9.1% 来自 BridgeV2、DROID 与 OXE，总量约 10k 小时。

### 4.2 Baseline 与评价指标

- **Naive/OpenVLA-style**：每维每时刻独立分箱。
- **FSQ**：学习式有限标量量化，代表 VQ 类压缩。
- **FAST**：每个数据集单独训练 BPE。
- **FAST+**：固定通用 tokenizer。
- **OpenVLA / OpenVLA+FAST**：验证 backbone 无关性。
- **扩散 π0**：比较训练速度、语言遵循和通用策略性能。

指标按任务使用 success rate 或 task progress，图中报告均值与 95% 置信区间；压缩实验同时测 token 数与重建误差。

### 4.3 实现细节与复现边界

论文公开 FAST+ 的 HuggingFace `AutoProcessor` 用法，tokenizer 本身可在几分钟内拟合。动作归一化必须复刻 1/99 分位数，chunk 时长推荐固定一秒；如果机器人控制频率变化而 chunk 秒数不变，FAST 仍能把冗余吸收在频域中。真正昂贵的是策略数据和真实机器人评测，而不是 tokenizer。

---

## 5. 实验结果

### 5.1 Tokenizer 与策略主结果

Figure 6 的核心不是小幅平均提升，而是“能否训练起来”的分界：20 Hz Table Bussing 和 50 Hz T-Shirt Folding 上，朴素 tokenization 基本无法推进任务；FAST、FSQ 均显著改善，其中 FAST 在精细真机任务更强。FAST+ 与数据集专用 FAST 接近，说明通用 BPE 词表没有明显损害控制。

在 OpenVLA 上替换 tokenizer 后，高频 T-shirt folding 从朴素版本的低成功表现提升到可有效执行，证明收益不依赖 π0 backbone。DCT 压缩—重建曲线还显示：FSQ 在允许较大重建误差时可能更省 token，但 FAST 在高保真区域扩展更好，更符合细粒度控制需求。

### 5.2 与扩散 π0 的比较

小数据任务（LIBERO、T-Shirt Folding，少于 50 小时）中 π0-FAST 与扩散 π0 大体相当；大数据 Table Bussing 上，FAST 约用 **1/3 训练步数**达到高水平。在 DROID 上，扩散 π0 更容易忽略语言，FAST 自回归目标的语言遵循更好。

通用模型结果如下图。按原图柱高读取，π0-FAST 与 π0 的五任务平均约 **68% vs. 70%**，误差区间明显重叠；分任务各有胜负：π0-FAST 在 Table Bussing、Grocery Bagging 更高，π0 在 T-Shirt Folding、Toast 更高，Laundry 基本接近。重要的是 π0-FAST 用约 **5× 更少 GPU 小时**达到这一总体水平；compute-matched 比较中也明显胜过同算力扩散 π0。

![](assets/FAST%20-%20高频机器人动作压缩与π0-FAST/figures/03-results.png)

> **图 11（论文原图）**：π0-FAST 与扩散 π0 在 T-shirt folding、table bussing、grocery bagging、toast 与 laundry folding 五项通用策略评测上总体持平；这张图需与训练成本一起读，论文主张的是相似最终性能下训练算力约降低 5 倍，而非每个任务都胜出。  
> **来源**：官方论文 Figure 11，第 10 页，[arXiv:2501.09747](https://arxiv.org/abs/2501.09747)

### 5.3 消融实验

1. **Backbone 消融**：OpenVLA+FAST 显著优于 OpenVLA 朴素分箱，说明 FAST 是表示层改进，而非 π0 特定技巧。
2. **BPE 消融**：DCT 无 BPE 仍强于 naive，但任务表现下降、推理更慢，证明稀疏频率矩阵还必须做符号压缩。
3. **通用/专用 tokenizer**：FAST+ 几乎匹配专用 FAST，支持跨具身共享。
4. **压缩—误差扫描**：FAST 在高重建保真区优于 FSQ，且同一 $\gamma$ 能覆盖多数据集。
5. **compute-matched π0**：相同训练算力下 π0-FAST 明显更强，排除了“只是训练更多”的解释。

### 5.4 泛化、效率与失败案例

DROID 策略可在三个未见校园场景中按语言完成抓放、开关柜门、擦拭、开水龙头等。失败 rollout 往往仍有语义合理性，例如接近微波炉或洗碗机把手但最终打不开，表明高层目标理解与接触动力学精度之间仍有缺口。

效率呈现鲜明的训练—推理交换：训练收敛最多快 5×，但一秒动作块推理从约 100 ms 增到约 750 ms。静态操控可接受，动态任务、快速反应或低时延闭环中可能直接改变系统稳定性。

### 5.5 实验充分性判断

证据链较完整：有可控玩具实验、压缩率、重建误差、两种 backbone、学习式 tokenizer baseline、单任务与 10k 小时通用训练、未见场景真机评测。主要不足是多数真机柱状图没有逐项精确数值表，且 FAST+ 的跨形态结论多数来自压缩率而非在灵巧手、人形或导航平台上的闭环策略部署。

---

## 6. 与相关工作的关系

FAST 与 RT-2/OpenVLA 的区别不是把 bin 数调得更细，而是显式利用动作的时间结构；与 VQ-VAE/FSQ 的区别是不用训练神经压缩器，避免 codebook 和重建网络超参；与扩散/flow matching VLA 的关系则是互补竞争：FAST 保留预训练 Transformer 的 next-token 接口和快速训练，扩散动作专家保留并行连续输出与快速推理。后续 π0.5 与 Knowledge Insulation 正是沿着“训练时离散、部署时连续”进一步组合两条路线。

---

## 7. 局限与批判性评价

1. **推理速度是实质性缺陷**：750 ms/chunk 对动态操控并非小问题，动作 chunking 只能掩盖平均吞吐，无法消除观测到动作之间的延迟。
2. **平滑性先验有边界**：冲击、接触切换、灵巧手快速离散事件会把能量推向高频，压缩率可能下降；论文尚未在这些闭环平台验证。
3. **通用 tokenizer 不等于通用策略**：FAST+ 跨人形、手、导航主要验证 token 压缩，不证明策略性能同样稳定。
4. **BPE 依赖数据统计**：固定词表虽能迁移，但极端新动作空间或单位规范仍可能产生较长序列；归一化协议也是部署接口的一部分。
5. **结果可复现性不均**：tokenizer 代码与算法简单，策略训练所需 10k 小时私有数据、903M 时间步和多套机器人则难以独立复刻。
6. **个人判断**：FAST 最大价值不是“DCT 很新”，而是识别出 VLA 的动作语言存在错误的 token 粒度。它把性能瓶颈从模型规模转回表示的信息率，是后续混合离散/连续 VLA 的关键起点。

---

## 8. 复现与实践建议

### 8.1 最小复现

可先在 LIBERO 或单一真实机器人数据上做四组：naive、DCT without BPE、FAST、FSQ。固定一秒 chunk 与相近重建 MSE，记录 token 数、teacher-forcing loss、rollout 成功率和单 chunk 延迟。这样能区分“压缩提高训练信息量”和“重建误差改变控制”两种因素。

### 8.2 成本评估

- **Tokenizer**：低成本；DCT 为解析操作，BPE 拟合通常数分钟，CPU 即可。
- **策略训练**：高成本；PaliGemma-3B/OpenVLA-7B 全量微调，需要多 GPU，通用模型达到数千 GPU 小时级别。
- **数据**：最大门槛；复刻通用结论需多具身、不同频率和动作空间的大规模真实机器人数据。
- **真机评测**：至少 6 类硬件场景，人工打分与维护成本高。
- **代码可用性**：FAST+ tokenizer 已公开；论文未在 PDF 中给出完整通用训练数据与全部策略检查点的可复现路径。

### 8.3 工程注意事项

必须保存每个数据集的 1/99 分位归一化统计；训练和推理的 chunk 秒数、动作维度顺序、单位和 control mode 必须一致。低频优先展平不可随意改为按维度展平。若部署要求低于 100 ms 反馈，优先考虑让 FAST 只承担训练期表示学习，再蒸馏或接入并行动作专家。

---

## 9. 个人启发与后续问题

FAST 揭示了一个普遍原则：序列模型的 token 应接近“可预测信息单元”，而不是传感器或控制器的原始采样单元。值得继续研究的问题包括：能否用因果多尺度变换替代固定 DCT；能否根据动作不确定性自适应决定保留频率；能否通过 speculative decoding、并行解码或离散到连续蒸馏消除 750 ms 延迟；以及 FAST 表示是否能直接支持跨机器人动作检索、技能分段和世界模型预测。

---

## 10. 材料来源

- 官方论文页：[https://arxiv.org/abs/2501.09747](https://arxiv.org/abs/2501.09747)
- 官方 PDF：`assets/FAST - 高频机器人动作压缩与π0-FAST/sources/paper.pdf`
- 项目页：[https://pi.website/research/fast](https://pi.website/research/fast)
- 本地来源清单：`assets/FAST - 高频机器人动作压缩与π0-FAST/sources.md`
- 本文全部公式、数据、baseline、实验数字与局限均以官方 PDF 为主；柱状图近似值已明确标注为按原图读取。
