---
type: paper-note
status: complete
domain: VLM
paper: "InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models"
year: 2025
arxiv: "2504.10479"
doi: null
source: "https://arxiv.org/abs/2504.10479"
project: null
code: "https://github.com/OpenGVLab/InternVL"
tags:
  - VLM
  - native-multimodal-pretraining
  - preference-optimization
  - test-time-scaling
  - InternVL
created: 2026-10-15
updated: 2026-10-15
---

# InternVL3：原生多模态预训练与测试时扩展

## 基本信息

- **模型族**：1B、2B、8B、9B、14B、38B、78B；小型号用 InternViT-300M，大型号用 InternViT-6B，语言侧主要来自 Qwen2.5 base，9B 使用 InternLM3-8B。
- **主线**：ViT–MLP–LLM + 动态高分辨率 + V2PE；约 200B token 的 native multimodal pre-training；21.7M SFT；约 300K MPO；VisualPRM Best-of-N 测试时扩展。
- **代表结果**：InternVL3-78B 在 MMMU 为 72.2；MathVista 79.0；OCRBench 906。VisualPRM Best-of-8 后，七项推理平均由 54.6 升至 56.5。
- **重要性判断**：★★★★★。它把 InternVL 从“给文本 LLM 事后接视觉”的多阶段适配，推进到预训练期共同更新视觉、投影和语言组件，并把后训练与测试时计算形成完整闭环。

**一句话结论：**

InternVL3 最重要的不是再刷新一组榜单，而是将视觉从 LLM 的后装插件提升为预训练期的一等数据源：**文本和多模态数据共同训练全模型，MPO 修正长推理的分布偏移，VisualPRM 再把额外测试时计算转化为可选择的正确轨迹**；这三层分别作用于表征、策略和搜索。

---

## 1. 研究背景与问题定义

### 1.1 从 post-hoc 适配到原生多模态预训练

传统 MLLM 流程通常先得到文本 LLM，再冻结或部分冻结 LLM/ViT，通过 projector warmup 和若干对齐阶段“补装”视觉能力。问题在于：文本预训练从未为视觉 token 预留结构，后续必须依赖复杂的冻结计划、OCR 等专项补课和文本数据回灌，且视觉适配可能损伤语言能力。

InternVL3 的问题设定是：既然最终模型需要同时处理文本、图像和视频，能否在预训练阶段就把纯文本与多模态样本交错输入，并联合更新 ViT、MLP 和 LLM？论文称之为 **native multimodal pre-training**。这里的“原生”不是从随机权重起训：为节省成本，ViT 和 LLM 仍由预训练 base 权重初始化；“原生”指的是后续基础预训练过程不再把视觉视作事后 adapter 任务。

![](assets/InternVL3%20-%20原生多模态预训练与测试时扩展/figures/01-overall-performance.png)

> 图 1：InternVL3-8B/78B 与 InternVL2.5、Qwen2.5-VL 和若干闭源模型的代表性结果。图中 78B 相对前代在 MathVista、OCR、长视频等任务有稳定增益，但 HallusionBench 仍落后 Gemini-2.5-Pro。来源：论文 Figure 1，第 2 页。

### 1.2 三个层级的问题

1. **预训练层**：如何用同一个自回归目标兼容纯文本、图文、视频文本和交错多模态数据，并保留语言能力？
2. **后训练层**：SFT 的 teacher forcing 与自回归推理分布不同，长 CoT 容易逐步偏离，如何同时学习相对偏好、绝对质量与首选答案生成？
3. **推理层**：同一模型可采样多个解答时，如何用视觉过程奖励模型选择更可靠的轨迹，而非仅看最终答案或依赖长度启发式？

---

## 2. 任务定义与输入输出

### 2.1 统一接口

输入消息可以包含文本、单图、多图或视频。每张图按宽高比匹配预定义网格，切为若干 $448\times448$ tile，并附加 thumbnail；InternViT 编码后通过 pixel unshuffle 将 token 数降为原来的四分之一，每个 tile 得到 256 个视觉 token。两层随机初始化 MLP 将视觉通道映射到 LLM embedding 维度，随后视觉 token 与文本 token 组成一条序列，由 LLM 自回归输出文本。

![](assets/InternVL3%20-%20原生多模态预训练与测试时扩展/figures/02-architecture.png)

> 架构图：动态高分辨率先按纵横比切块，InternViT 编码，pixel unshuffle 压缩，再由两层 MLP 接入 Qwen2.5/InternLM3。它与 2023 年原始 InternVL 的 QLLaMA 重型中间件不同，是后续系列已经确立的 ViT–MLP–LLM 路线。来源：arXiv 官方源码未编号完整架构资产。

### 2.2 预训练目标

序列 $\mathbf{x}=(x_1,\ldots,x_L)$ 可包含文本、图像和视频表示。标准全 token 自回归损失为：

$$\mathcal{L}_{\mathrm{full}}(\theta)=-\sum_{i=2}^{L}w_i\log p_\theta(x_i\mid x_{<i})$$

实际只对文本 token 计算损失：

$$\mathcal{L}_{\mathrm{text}}(\theta)=-\sum_{\substack{i=2\\x_i\in\mathrm{Text}}}^{L}w_i\log p_\theta(x_i\mid x_{<i})$$

视觉 token 不作为离散目标直接预测，而是条件上下文；梯度仍通过文本预测回传到 MLP 与 ViT。为避免 token averaging 偏向长样本、sample averaging 偏向短样本，论文沿用 square averaging：样本有效文本长度为 $l$ 时，$w_i=l^{-0.5}$。

### 2.3 输出与评测

输出统一为文本：开放问答、OCR、文档答案、边界框坐标、GUI action、数学 CoT 等都由 LLM token 表达。测试时 scaling 在推理任务上采样 $N$ 条完整解答，再由 VisualPRM 对每步打分、选择总分最高的响应。因此最终输出仍是一条文本轨迹，但计算预算从一次生成扩展为 $N$ 次生成加 $N$ 次过程评分。

---

## 3. 核心方法

### 3.1 ViT–MLP–LLM 与动态高分辨率

InternVL3 沿用 InternVL2.5 的主架构，而不是回到原始 InternVL 的 QLLaMA。1B–14B 使用 300M InternViT，38B/78B 使用 6B InternViT；LLM 从 base 而非 instruct 权重初始化。pixel unshuffle 将 $448\times448$ tile 的视觉 token 压到 256，直接影响高分辨率、多图和视频的上下文成本。

这一选择表明 InternVL 路线已经完成转型：原始 InternVL 用 8B QLLaMA 作为可独立参与对比与生成的“大 glue”；InternVL3 则把连接器简化为两层 MLP，把跨模态能力主要交给联合预训练后的 ViT 和 LLM 本体。

### 3.2 V2PE：让视觉 token 慢速消耗位置窗口

普通位置索引对每个 token 加 1。V2PE 对文本仍加 1，对同一图像内的视觉 token 只加 $\delta<1$：

$$p_i=p_{i-1}+\begin{cases}1,&x_i\text{ 为文本 token}\\\delta,&x_i\text{ 为视觉 token}\end{cases}$$

训练时每张图从以下集合随机选择固定 $\delta$：

$$\Delta=\left\{1,\frac12,\frac14,\frac18,\frac1{16},\frac1{32},\frac1{64},\frac1{128},\frac1{256}\right\}$$

视觉 token 因而占据更紧凑的位置区间，长视频和多图不容易提前耗尽 RoPE 窗口；同一图内部 $\delta$ 固定，仍保持相对次序。推理时可按序列长度选择 $\delta$。但论文除 V2PE 专项消融外，其余主结果固定 $\delta=1$，所以不能把全部榜单提升归因于位置压缩。

### 3.3 Native multimodal pre-training

其关键不是多加一阶段，而是替换旧的 MLP warmup：纯文本与多模态样本交错进入同一训练流，ViT、MLP、LLM 所有层联合更新。数据约 200B token，其中 50B 纯文本、150B 多模态，即语言:多模态为 1:3。作者先在各模态内部搜索配比，再固定总预算搜索跨模态比例。

多模态数据继承 InternVL2.5 的 caption、通用 QA、数学、图表、OCR、知识、文档、多轮对话和医疗数据，并补充 GUI、工具使用、3D 场景和视频；纯文本基于 InternLM2.5 语料并加入开源数学、代码和知识数据。论文特别强调总多模态规模并未简单增加，性能提升部分来自让 ViT/LLM 本体真正参与训练。

### 3.4 SFT：扩大能力覆盖

SFT 继承随机 JPEG 压缩、square loss re-weighting 和 multimodal data packing，样本从 InternVL2.5 的 16.3M 增至 21.7M，新增工具、3D、GUI、长上下文、视频、科学图、创意写作和多模态推理。此阶段的作用不是基础对齐，而是把预训练能力组织成可交互格式。

### 3.5 MPO：相对偏好、绝对质量与生成三合一

MPO 针对 teacher forcing 与自由生成的分布偏移。总目标为：

$$\mathcal{L}=w_p\mathcal{L}_p+w_q\mathcal{L}_q+w_g\mathcal{L}_g$$

偏好项 $\mathcal{L}_p$ 使用 DPO，比较 chosen $y_c$ 与 rejected $y_r$ 相对参考模型 $\pi_0$ 的 log-ratio：

$$\mathcal{L}_p=-\log\sigma\left(\beta\log\frac{\pi_\theta(y_c\mid x)}{\pi_0(y_c\mid x)}-\beta\log\frac{\pi_\theta(y_r\mid x)}{\pi_0(y_r\mid x)}\right)$$

质量项 $\mathcal{L}_q$ 使用 BCO，分别判断 chosen 的绝对质量应高、rejected 应低，并以历史 reward 的移动均值 $\delta$ 做 shift；生成项 $\mathcal{L}_g$ 则继续对 chosen response 做语言模型损失。三者分工是：DPO 学“哪个更好”，BCO 学“各自是否够好”，LM loss 防止只学判别边界而丢失首选答案的生成过程。

MPO 数据约 300K，覆盖 VQA、科学、图表、数学、OCR 和文档。8B、38B、78B 的 SFT 模型产生 rollout，所有尺度使用同一 MPO 数据。由于 MPO 数据是 SFT 数据的子集，消融增益更能指向算法而非额外知识。

### 3.6 VisualPRM 与 Best-of-N 测试时扩展

对候选解答 $s=(s_0,\ldots,s_n)$，VisualPRM 把图像 $I$、问题 $q$ 和截至当前的步骤 $s_{\le i}$ 组织成多轮对话，预测该步正确标签：

$$c_i\sim M(y_i\mid I,q,s_{\le i}),\quad c_i\in\{+,-\}$$

推理时以生成“+”的概率作为 step score，再对各步平均得到整条解答分数，最后从 $N$ 条 rollout 里选最高者。它不同于 outcome reward：能在中间推理步骤发现错误；也不同于多数投票：选择依据是视觉条件下的过程质量，而非答案频次。

VisualPRM 以 VisualPRM400K 为基础，并用 InternVL3-8B/38B rollout 扩充。主表使用 Best-of-8。需要注意，Bo8 的成本近似为 8 次策略生成加 8 条过程评分，不能与单次解码只按准确率比较。

---

## 4. 数据、训练阶段与工程设置

### 4.1 阶段汇总

| 阶段 | 规模 | 可训练参数 | 核心目的 |
|---|---:|---|---|
| Native pre-training | 约 200B token：50B text + 150B multimodal | ViT、MLP、LLM 全部 | 同时建立语言和视觉条件生成能力 |
| SFT | 21.7M 样本 | 全模型，论文未逐型号列统一超参 | 指令遵循与专项任务适配 |
| MPO | 约 300K preference 样本 | policy 相对 reference 优化 | 提升 CoT 与多模态推理 |
| VisualPRM | VisualPRM400K + InternVL3 rollouts | 独立 8B critic | 测试时选择最佳轨迹 |

### 4.2 基础设施与成本

InternEVO 对 ViT、MLP、LLM 分别做可解耦 sharding，并支持 data/tensor/sequence/pipeline parallel 组合；针对视觉与文本 token 比例变化造成的负载不均衡做动态平衡。论文称在相同预算、相近模型规模下，相比 InternVL2.5 获得 50%–200% 训练速度提升，并支持 32K sequence 的 head/sequence parallel。

论文未披露完整 GPU 数、总训练时长和能耗；“可扩展到数千 GPU”是基础设施能力描述，不等于每个型号的实际用量。复现成本因此只能定性评估。

---

## 5. 实验结果与 baseline

### 5.1 总体结果

InternVL3-78B 在 MMMU 72.2、MathVista 79.0、AI2D 89.7、ChartQA 89.7、DocVQA 95.4、InfoVQA 85.2、HallusionBench 59.1、OCRBench 906、LongVideoBench 65.7。对比 InternVL2.5-78B，分别在 MMMU +2.1、MathVista +6.7、AI2D +0.6、ChartQA +1.4、DocVQA +0.3、InfoVQA +1.1、HallusionBench +1.7、OCRBench +52、LongVideoBench +2.1。

相对 Qwen2.5-VL-72B，InternVL3-78B 在 MMMU、MathVista、AI2D、OCRBench 和 LongVideoBench 更高，但 ChartQA、DocVQA、InfoVQA 并非全面领先。相对 Gemini-2.5-Pro，InternVL3 在 MathVista 接近、AI2D略高，但 MMMU 与 HallusionBench 仍落后。因此“接近闭源”成立于部分维度，而非统一领先。

### 5.2 推理与数学

单次生成时，InternVL3-8B 七项平均 44.3，高于 InternVL2.5-8B 的 32.8 和 Qwen2.5-VL-8B 的 41.4；38B 为 52.8；78B 为 54.6。Best-of-8 + VisualPRM 后，8B/38B/78B 分别提升到 50.2/56.6/56.5。78B 在 MathVision 从 43.1 降到 40.8，却在 MathVista、MathVerse、DynaMath、WeMath、LogicVista 上提升，说明 critic 并非对每个分布都可靠。

### 5.3 OCR、视频、GUI 与空间能力

78B 的 OCRBench 906、VCR 96.0/98.6；Video-MME 72.7/75.7，但 Qwen2.5-VL-72B 的 73.3/79.1 更高。GUI grounding 中，论文正文将大模型写作 InternVL3-72B（型号表为 78B，报告存在命名不一致），ScreenSpot 88.7、ScreenSpot-V2 90.9。VSI-Bench 上 8B/38B/78B 为 42.1/48.9/48.4，38B 略高于 78B，表明规模收益并不单调。

### 5.4 语言能力是否被牺牲

与同源 Qwen2.5 Chat 对比，InternVL3 多数知识、阅读和推理项更高，综合分数从 0.5B/1.5B/7B/14B/32B/72B 的 33.5/51.6/69.4/73.4/77.4/78.9，提高到对应 InternVL3 的 42.4/59.2/72.9/76.6/78.9/80.5。但代码并非全面提升：例如 78B 的 HumanEval/MBPP 为 82.3/76.7，低于 Qwen2.5-72B Chat 的 87.2/86.8。原生多模态训练保住了总体语言能力，不等于每类文本技能无损。

---

## 6. 消融实验

### 6.1 Native multimodal pre-training

![](assets/InternVL3%20-%20原生多模态预训练与测试时扩展/figures/03-native-pretraining-ablation.png)

> 图 3：在架构、初始化和训练数据不变时，用 native multimodal pre-training 替换旧 MLP warmup。仅预训练模型已接近旧多阶段 SFT baseline，再叠加高质量 SFT 后多数任务继续提升。来源：论文 Figure 3，第 18 页。

该消融控制了 InternVL2-8B 的架构、初始化和数据，因而是全文最有说服力的因果证据。不过图中比较仍把“训练范式”与具体 schedule 绑定，未披露等算力下的 token/step 完整对账，效率结论仍需谨慎。

### 6.2 V2PE

InternVL3-8B 预训练模型不用 V2PE 时总体 75.2；$\delta=1/4$ 时为 75.9，是表中最佳；$1/16$ 为 75.6，$1/256$ 反而降到 75.0。说明位置压缩不是越强越好，适度压缩能改善短中上下文，但过小增量可能降低视觉 token 的位置可分辨性。主结果固定 $\delta=1$，这也意味着论文对长上下文收益的系统量化仍不充分。

### 6.3 MPO

MPO 对 1B/2B/8B/9B/14B/38B/78B 的七项平均增益分别为 +0.5/+1.7/+2.9/+1.5/+3.7/+4.5/+4.1。较大模型收益更明显，但单项并非全升：1B 的 MathVista、WeMath、LogicVista下降；9B 的 MMMU 与 MathVision下降。MPO 提高总体推理分布，不是无条件 Pareto improvement。

---

## 7. 局限、失败案例与未来工作

### 7.1 论文明确暴露的局限

1. **多图差距**：MuirBench 等基准仍落后 Qwen2.5-VL，作者提出需要更好的数据筛选和模型改进。
2. **幻觉指标不一致**：HallusionBench总体提升，但 MMHal 有小幅下降，说明优化尚不稳定。
3. **grounding 平台期**：78B 的平均 grounding 91.4，低于 InternVL2.5-78B 的 92.3；作者推测是新增数据中 grounding 占比相对下降。
4. **规模非单调**：VSI-Bench 38B 高于 78B；VisualPRM-Bo8 使 78B MathVision下降，显示更大 policy/更多采样不自动解决 critic 分布偏差。
5. **数据透明度不完整**：论文脚注明确说开放数据清单仍在整理，训练数据虽承诺发布，但当前报告不足以按比例复原。

### 7.2 论文给出的未来方向

正文明确提到：扩展到动态、交互式 GUI 环境；把 3D 场景理解集成到更多下游应用；继续改善多图数据筛选、幻觉一致性和 grounding 数据比例。更一般的研究问题是如何改进视觉 token 位置编码，以及如何让过程奖励模型跨数学、科学图表、文档等分布稳定泛化。

### 7.3 我的批判性判断

“Native”容易被误读为从零训练多模态模型。实际上 InternVL3 仍初始化自预训练 InternViT 与 Qwen2.5/InternLM3 base；贡献是 **联合继续预训练的时机和全参数优化方式**。这是现实、有效的工程折中，但不等于证明从随机初始化开始的原生多模态训练更优。

此外，总体进步由 native pre-training、额外纯文本、专项数据、21.7M SFT、MPO 和 test-time scaling 共同产生。除 native、V2PE、MPO 外，数据新增的独立贡献没有被充分拆开。VisualPRM-Bo8 结果也应单列计算预算，不能直接与单样本 baseline 当作同成本排名。

---

## 8. 复现成本与实践建议

### 8.1 可复现层级

- **推理复现**：1B/2B/8B 权重最现实。需实现动态切图、pixel unshuffle、模板与 V2PE 配置；高分辨率多图/视频的显存主要由视觉 token 与 KV cache 决定。
- **MPO 复现**：需要 SFT policy、reference model、chosen/rejected 对和 DPO+BCO+LM 混合损失；约 300K 数据规模可行，但 rollout 生成与多尺度清洗昂贵。
- **VisualPRM-Bo8**：除 policy 外还需 8B critic，推理成本约为八次候选生成和过程评分；适合离线高价值任务，不适合低延迟服务默认开启。
- **全量预训练**：200B token、全参数训练和复杂并行基础设施超出普通团队能力，且论文缺少完整 GPU-hour 与数据清单。

### 8.2 推荐的最小实验

以 8B 为例，固定架构和 token budget，对比旧 projector warmup 与 text:multimodal=1:3 的联合训练；评测至少覆盖一个语言 benchmark、一个 OCR、一个数学和一个多图任务。后训练再分别加入 MPO 与 Bo8，报告准确率、平均输出 token、总 FLOPs/延迟，才能区分“模型变强”和“推理花得更多”。

---

## 9. 个人判断与后续问题

### 9.1 三层能力栈

我更倾向把 InternVL3 理解为三层系统：native pre-training 改善**表征底座**，MPO 改善**生成策略**，VisualPRM 改善**候选搜索与选择**。这三层可分别替换，因此其价值不只在 InternVL：任何 ViT–projector–LLM 模型都可用类似结构组织研究。

### 9.2 最重要的未决问题

1. 同 token、同 FLOPs、同数据下，联合预训练相对多阶段训练究竟节省多少总计算？
2. 1:3 配比是否随模型规模、数据质量和任务族变化？
3. VisualPRM 的平均 step score 是否偏好短解答，如何校准不同长度和领域？
4. V2PE 在真正 32K 多图/长视频上的收益是否大于短基准上的小幅增益？
5. 代码能力下降是否来自多模态数据挤占、SFT 配比，还是 MPO 对输出风格的改变？

### 9.3 最终判断

InternVL3 的核心贡献是训练范式而非新网络层。它说明在已有强 base 权重上，**把视觉和语言数据置于同一继续预训练阶段、全参数共同演化，比反复冻结—解冻的事后对齐更自然**。MPO 和 VisualPRM 则证明仅靠预训练不足以得到可靠长推理：还需要偏好学习与测试时 verifier。其最大不足是全系统变量过多、成本披露不足，尚未完全回答等算力因果问题。

---

## 10. 材料来源

- 官方论文：<https://arxiv.org/abs/2504.10479>
- 本地 PDF：`assets/InternVL3 - 原生多模态预训练与测试时扩展/sources/paper.pdf`
- 来源与 Figure/Page 记录：`assets/InternVL3 - 原生多模态预训练与测试时扩展/sources.md`
- 官方代码：<https://github.com/OpenGVLab/InternVL>
