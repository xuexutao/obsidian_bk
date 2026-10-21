---
type: paper-note
status: complete
domain: World-Models
paper: Holistic Evaluation of Multimodal LLMs on Spatial Intelligence
year: 2025
arxiv: 2508.13142
doi: 10.48550/arXiv.2508.13142
source: https://arxiv.org/abs/2508.13142
project: https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard
code: https://github.com/EvolvingLMMs-Lab/EASI/
tags:
  - Spatial-Intelligence
  - Benchmark
  - Multimodal-LLM
  - Evaluation
created: 2026-10-16
updated: 2026-10-16
---

# EASI：多模态模型空间智能统一评测

## 基本信息

- **作者**：Zhongang Cai、Yubo Wang、Qingping Sun、Ruisi Wang 等
- **版本**：arXiv v5，2025-12-30；39 页
- **定位**：EASI（Evaluation of multimodAl LLMs on Spatial Intelligence）不是新模型，而是把八个近期空间 benchmark、两套评测 protocol、统一接口、模型适配和排行榜组织成一套评测系统。
- **评测规模**：约 31K 图像、4.5K 视频、24K QA；作者称完整研究消耗超过 100 亿 token。

## 一句话结论

EASI 的价值在于把零散空间任务重构成六类能力，并显式量化 prompt、metric、答案抽取、选项位置和推理 token 对排名的影响；统一 protocol 下 GPT-5 平均 CAA 43.06，显著领先但与人类 86.41 仍差 43.35 点，弱项集中在 mental reconstruction、perspective-taking、deformation/assembly 和 comprehensive reasoning。其不足是没有系统做图像/场景级 contamination audit 或 text-only/no-image 诊断，因此“benchmark 很新”和 circular robustness 不能替代对泄漏与语言捷径的严格排除。

---

## 1. 为什么需要统一空间评测

### 1.1 现有问题

空间 benchmark 的 taxonomy、system prompt、输出格式与 metric 各不相同：同一模型在 Direct QA、zero-shot CoT、manual CoT 下可能差异很大；有的报告普通 Accuracy，有的用 chance-adjusted accuracy（CAA），数值答案又用 MRA；答案抽取失败和推理被 token limit 截断也会被计为能力失败。直接把原论文数字拼表无法公平比较。

### 1.2 EASI 的两个目标

1. **能力层统一**：把不同 benchmark 的 subtask 映射到六类基础空间能力。
2. **执行层统一**：提供模型接口、prompt、答案抽取、metric、circular evaluation 与 token 配置，同时保留各 benchmark 的官方设置。

EASI 因此同时报告两类结果：Official Protocol 用各 benchmark 原生 prompt/metric，便于与原论文比较；EASI Protocol 用统一 zero-shot CoT prompt 与统一 CAA/MRA，便于跨 benchmark 求平均。

---

## 2. 六类空间能力 taxonomy

![](assets/EASI%20-%20多模态模型空间智能统一评测/figures/02-taxonomy.png)

> **图 2（论文 Figure 2）**：六类能力依次为 Metric Measurement、Mental Reconstruction、Spatial Relations、Perspective-taking、Deformation & Assembly、Comprehensive Reasoning。来源：官方 arXiv LaTeX 图源，论文第 4 页附近。

### 2.1 Metric Measurement（MM）

从没有内参的 2D 观察估计深度、长度与物理尺度，本质存在歧义，依赖对象常识和场景先验。EASI 发现强模型在典型物体/房间尺度上已接近甚至超过所用 human baseline，但复杂属性测量仍弱。

### 2.2 Mental Reconstruction（MR）

从有限视角构造对象的完整 3D mental model，并做旋转、新视图或三视图推理。它区别于 PT：MR 更强调对象内部精细几何，而 PT 更强调观察坐标变化。

### 2.3 Spatial Relations（SR）

判断多个对象的位置、方向和朝向。简单单视角关系较强，但透视造成的视觉错觉、复杂多对象关系仍会击穿模型。

### 2.4 Perspective-taking（PT）

在 egocentric、exocentric、object-centric 或 hypothetical viewpoint 间转换。它往往需要先重建场景，再做坐标变换，是八个 benchmark 中最普遍的核心瓶颈。

### 2.5 Deformation & Assembly（DA）

处理形状变化和结构组合，如绳结、纸盒展开、零件组装。该类不能假设刚体不变，对具身操作尤其重要。

### 2.6 Comprehensive Reasoning（CR）

长程导航、遮挡计数、复杂物理模拟等任务需要多类能力、记忆与多步推理联合。EASI 中 route planning、appearance order 等 CR 子任务与人类差距可达数十点。

---

## 3. 八个 benchmark 与 protocol

### 3.1 Benchmark 组成

- **VSI-Bench**：288 视频、5K QA；覆盖 MM/SR/PT/CR。
- **SITE**：13.2K 图像、3.8K 视频、8.1K QA；覆盖面较广。
- **MMSI**：2K 图像、1K 专家标注 QA；包含 MM/MR/PT/CR。
- **OmniSpatial**：1.3K 图像、1.5K QA；偏 MM/PT/CR。
- **MindCube**：原版 21.1K QA，EASI 使用平衡的 MindCube-Tiny（1,050 QA、428 unique images）。
- **STARE**：10.3K 图像、4K QA；覆盖 PT/DA/CR。
- **CoreCognition**：1.5K 图像、217 视频、1.5K QA；同时包含 SI 与非 SI 认知任务。
- **SpatialViz**：1.2K 图像、1.2K QA；覆盖 MR/SR/DA/CR。

### 3.2 Official 与 EASI Protocol 的张力

Official Protocol 保持原 prompt/metric，结果可与各论文直接对照，但不同列不能直接平均：例如 SITE 用 CAA，VSI 数值题用 MRA，其余多用 Acc/F1；prompt 也从 Direct QA 到 manual CoT 不等。

EASI Protocol 强制统一 prompt 和 metric，支持跨 benchmark 平均，但改变了原生交互方式。附录显示，对五个官方 Direct-QA benchmark，加入 CoT 并没有一致收益；SITE 和 MindCube 的开源模型常因官方 prompt 更贴近答案匹配格式而表现更好。因此，统一 protocol 提高可比性，也可能降低“每个 benchmark 上的最优设置”性能。

### 3.3 指标公式

对第 $i$ 题有 $n_i$ 个选项，预测正确指示量为 $X_i$，CAA 为

$$\mathrm{CAA}=\frac{\sum_{i=1}^{N}X_i-\sum_{i=1}^{N}1/n_i}{N-\sum_{i=1}^{N}1/n_i}=\frac{\mathrm{Acc}-r}{1-r},\qquad r=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{n_i}.$$

CAA=0 对应随机猜测，负值表示差于随机。它消除了不同选项数的随机基线，但同一数据上会把 Acc 差异放大 $1/(1-r)$ 倍。

VSI 数值题使用十个阈值的 MRA：

$$\mathrm{MRA}=\frac{1}{10}\sum_{\vartheta\in\{0.5,0.55,\ldots,0.95\}}\mathbb 1\left(\frac{|\hat y-y|}{y}<1-\vartheta\right).$$

答案抽取先读 `<answer>...</answer>`，失败后匹配 Answer/Final answer 等规则；两步都失败即判错，不用 LLM judge，降低额外评审模型偏差。

### 3.4 Circular evaluation

每道 $k$ 选项题循环旋转 $k$ 次。Soft-circular 把所有旋转当等权新题；Hard-circular 要全部旋转都正确才算对。GPT-5 在 SITE 的 Non/Soft/Hard 为 80.09/78.30/72.43，在 MMSI 为 41.86/41.37/26.14，后者显示普通分数中存在明显选项位置/偶然猜中成分。

EASI 最终主要采用 non-circular，因为排名大体一致且 circular 成本增加 $k$ 倍；这是工程折中，不等于 option bias 已消失。

---

## 4. 主要结果

### 4.1 统一 EASI Protocol

| 模型 | 平均 CAA | VSI | SITE | MMSI | OmniSpatial | MindCube | STARE | CoreCognition | SpatialViz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5 | **43.06** | 36.33 | 61.52 | 20.13 | 44.41 | 35.37 | 33.46 | 78.45 | 34.80 |
| Gemini-2.5-Pro | 40.25 | 35.79 | 56.05 | 19.33 | 41.35 | 39.53 | 26.73 | 73.21 | 30.05 |
| InternVL3-78B | 26.11 | 25.48 | 49.08 | 6.27 | 35.18 | 13.54 | 11.79 | 58.01 | 9.49 |
| Human | **86.41** | 95.08 | 67.50 | 96.27 | 90.18 | 91.94 | 94.63 | 79.10 | 76.59 |

GPT-5 是统一设置下最强模型，但与 human 平均差 43.35 点；MMSI 差 76.14，STARE 差 61.17，VSI 差 58.75。CoreCognition 差距仅 0.65，说明总体平均不能掩盖“非空间/基础认知已强、三维空间仍弱”的结构性差异。

### 4.2 Official Protocol

GPT-5 在八项中的分数为 VSI 55.03、SITE 61.88、MMSI 41.80、OmniSpatial 59.90、MindCube 56.30、STARE 54.59、CoreCognition 84.37、SpatialViz 51.27。MindCube 上 Grok-4 63.56 领先，说明 GPT-5 并非所有 benchmark 第一。

Official 表的列使用不同 metric，禁止直接平均。EASI 同时给出两套表是正确做法，也说明任何“总体榜首”都依赖 protocol 定义。

### 4.3 能力层结论

- **较强**：MM 和简单 SR。GPT-5 可借助常见物体尺寸先验做合理估计。
- **最弱**：PT、DA、MR、CR。MMSI 的多个 PT 子任务与人类差距最高约 80 点；SpatialViz 的 3D mental rotation 比 2D counterpart 低约 46 点；GPT-5 在部分 MR 子任务的 CAA 低于 0。
- **开闭源差距缩小**：在人类差距超过 60 点的困难任务上，闭源对开源常仅领先约 15 点以内，说明大家共同缺少基础空间表示，而非只差模型规模。

![](assets/EASI%20-%20多模态模型空间智能统一评测/figures/01-teaser.png)

> **图 1（论文 Figure 1）**：GPT-5 能解复杂非空间图形规律，却在简单 3D top-down view 上选错；人类儿童可答对。它说明语言/抽象推理强并不自动产生三维视角转换能力。来源：官方 arXiv LaTeX 图源，论文首页。

---

## 5. 推理 token 与消融

### 5.1 Token 配置本身会改变“能力”

默认 2,048 token 会让 GPT-5 在 SpatialViz 超过 50% 样本因推理截断失败，因此作者把闭源模型上限设为 16,384，开源模型为 2,048。GPT-5 平均内部 token 1,359、外部 104；Grok-4 内部平均 7,666、$p_{95}=16,384$，更容易耗尽预算。

### 5.2 Thinking mode 消融

SpatialViz-Tiny 118 题上，GPT-5 Minimal/Low/Medium/High 准确率为 48.31/54.24/56.78/52.54，平均时延 11.69/53.89/140.3/305.2 秒。High 模式有 28 题超 15 分钟或触及 token limit，被判错；排除这些题后为 68.89%，但剩余题本身偏易。结论不是“thinking 越多越差”，而是固定预算下存在准确率—成本—截断三方权衡。

### 5.3 消融边界

EASI 的消融集中在评测 protocol：Official vs unified prompt、Non/Soft/Hard circular、thinking effort、token cap。它不做模型结构消融，因为论文没有训练新模型。这些实验说明 benchmark 分数不是纯能力常数，而是模型、prompt、metric、抽取规则和预算共同决定的观测值。

---

## 6. Benchmark 泄漏与语言捷径

### 6.1 EASI 实际控制了什么

- circular evaluation 控制选项位置与随机猜测；
- CAA 校正不同选项数的 chance level；
- 统一/官方双 protocol 揭示 prompt sensitivity；
- 规则化答案抽取减少 judge 偏差；
- 较新的 benchmark 降低但不能消除预训练污染概率。

### 6.2 没有控制什么

EASI 没有报告：模型预训练数据与测试图像/视频的哈希去重、基础 3D 场景/资产 ID 去重、问题模板重合、text-only/no-image baseline、caption-only baseline，或对网页可搜索答案的审计。MindCube 等模板题即使 hard-circular 稳定，也仍可能通过题干统计、场景类别或模板先验获得分数。

因此 circular robustness **不是**“没有语言捷径”的充分证据。后续 SenseNova-SI 用 no-image MindCube 和 VSI-Debiased 才更直接地测了这类 shortcut，但仍不等同于完整 contamination audit。

### 6.3 Human baseline 也不是绝对标尺

各 benchmark 的 human 分数来自不同原始设置、样本和标注群体；EASI 把部分结果转换为 CAA 后汇总。SITE human 仅约 67.5，明显低于多数数据集，作者也提醒不要据此过度宣称模型“超越人类”。跨 benchmark 的 human average 是有用参照，但不是统一受控的人类实验。

---

## 7. 真实空间泛化证据

### 7.1 定量证据

八个 benchmark 覆盖室内视频、多图、多视角、导航、3D mental rotation、纸盒折叠等，任务跨度比单一 benchmark 大。统一排名在 official 与 EASI protocol 下大体一致，也说明核心结论不是单一 prompt 偶然造成。

### 7.2 定性证据

![](assets/EASI%20-%20多模态模型空间智能统一评测/figures/03-failure-cases.png)

> **图 3（论文 Figure 3）**：从 MM、MR、SR、PT、DA 到 CR 的九类案例并列比较 GPT-5-thinking、GPT-5、o3 和 Doubao-Seed。模型在度量题较强，却在三视图重建、低重叠相机运动、纸盒折叠、遮挡计数上系统失败。来源：官方 arXiv LaTeX 图源，论文 case study。

case study 同时包含 benchmark 与 in-the-wild 图例，揭示模型即使 reasoning strategy 看似合理，也会在 camera rotation、透视缩放、隐藏物体和结构变换处犯基础错误。

### 7.3 证据边界

EASI 仍是离线 QA/生成评测，没有真实机器人闭环、主动视点选择、持续 3D memory 或行动后果验证。它能判断“模型会不会回答空间题”，不能直接判断“模型能否在真实空间可靠行动”。对真实泛化的最强结论应是：多 benchmark 和野外案例显示弱点跨任务复现；尚未证明具身控制泛化。

---

## 8. 局限与批判性评价

### 8.1 论文未单列完整 limitation section

结论强调 taxonomy、protocol 和工具链，但没有系统列出 contamination、human baseline 异质性、API 版本漂移、闭源模型不可复验、网页 case study 与 API 定量设置不一致等限制。上述问题需要从正文和附录自行归纳。

### 8.2 我的判断

EASI 最重要的贡献是把“空间智能差”从一句印象变成可定位的能力缺口：模型不是所有空间任务都弱，而是 MM/简单 SR 已有进展，PT/MR/DA/CR 仍缺基础表征。它也揭示评测工程对结论的巨大影响，尤其是 token 截断和 prompt/metric 不统一。

但 EASI 的“holistic”主要指 benchmark breadth，不代表生态有效性。八个 benchmark 都是被动观察后的回答任务，缺少行动、探索、状态更新和错误恢复。后续研究不应只在 EASI 平均分上优化，还应建立无污染、跨场景、可交互的空间泛化测试。

---

## 9. 复现与实践建议

### 9.1 复现顺序

1. 固定模型版本、API 日期、thinking effort、temperature、max token 和图像/视频采样策略。
2. 先跑每个 benchmark 的 Official Protocol，确认能复现原生数字。
3. 再跑 EASI Protocol，统一 CAA/MRA 与 zero-shot CoT prompt。
4. 保存原始输出、解析后的答案、截断/超时标记和 token 统计，避免只保留最终分数。
5. 对 MCQ 至少抽样做 hard-circular；新增 text-only/no-image baseline 和可搜索性检查。
6. 为测试图像、视频、场景资产与问题模板建立 contamination manifest。

### 9.2 资源与成本

完整评测超过 100 亿 token，circular 会再乘以平均选项数，闭源模型还存在 API 成本和时延。实践中可先用分层小样本做 protocol regression，再对主模型跑全量；但最终排名必须全量执行，并把失败/截断算入结果。

### 9.3 代码可用性

官方代码和 leaderboard 已公开，适合作为统一 runner；闭源模型结果仍受 API 后端更新影响。复现报告应记录精确 model snapshot（如 `GPT-5-2025-08-07`），不能只写家族名。

---

## 10. 后续问题

1. 如何设计与公开训练语料物理隔离的空间 benchmark，并提供 scene/asset 级 contamination 证明？
2. 能否把 no-image、caption-only、shuffled-view 和 counterfactual camera tests 纳入默认 protocol？
3. 六类能力能否扩展到 active perception、action consequence、persistent memory 和 uncertainty calibration？
4. 是否应以真实闭环成功率替代平均 QA 分数，检验 PT/MR 提升是否真正转移到具身任务？

---

## 11. 材料来源

- 官方论文：<https://arxiv.org/abs/2508.13142>
- 代码：<https://github.com/EvolvingLMMs-Lab/EASI/>
- Leaderboard：<https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard>
- 本地来源与图号映射：[[assets/EASI - 多模态模型空间智能统一评测/sources]]
