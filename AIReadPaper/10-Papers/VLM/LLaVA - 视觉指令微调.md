---
type: paper-note
status: done
domain: VLM
paper: Visual Instruction Tuning
year: 2023
arxiv: 2304.08485
doi: null
source: https://arxiv.org/abs/2304.08485
project: https://llava-vl.github.io/
code: https://github.com/haotian-liu/LLaVA
tags:
  - VLM
  - instruction-tuning
  - visual-assistant
  - synthetic-data
created: 2026-10-15
updated: 2026-10-15
---

# LLaVA：视觉指令微调

## 基本信息

| 字段 | 内容 |
|---|---|
| 正式标题 | Visual Instruction Tuning |
| 作者 | Haotian Liu、Chunyuan Li、Qingyang Wu、Yong Jae Lee |
| 发表 | NeurIPS 2023 |
| arXiv | 2304.08485 |
| 官方项目 | https://llava-vl.github.io/ |
| 官方代码 | https://github.com/haotian-liu/LLaVA |
| 一句话结论 | LLaVA 证明了多模态能力不一定先依赖复杂连接器：用 GPT-4 把现有视觉标注改写成高质量指令数据，再以 CLIP 线性投影接入 Vicuna，就能把视觉识别底座转化为能遵循用户意图的开放视觉助手。 |

---

## 1. 研究背景

### 1.1 从“描述图像”到“按视觉指令行动”

CLIP 提供开放词汇识别，Flamingo 与 BLIP-2 提供图像条件文本生成，但早期 VLM 常把任务固定为字幕、VQA 或检索，语言更多是输出格式，而不是显式任务接口。ChatGPT、Vicuna 等文本模型则显示：指令微调可以把预训练知识组织成可交互助手。LLaVA 提出“visual instruction tuning”，研究如何把这种对齐迁移到图像—语言空间。

核心瓶颈首先是数据，而非结构：公开图文对很多，却缺少“用户提出多样视觉问题—助手按意图回答”的对话、详细描述和复杂推理样本。直接人工收集昂贵；多模态 GPT-4 当时也不可直接批量访问。作者因此把 COCO 的字幕与检测框转成文本上下文，让 text-only GPT-4 扮演教师生成视觉指令数据。

### 1.2 核心贡献

论文贡献包括：一套 GPT 辅助的视觉指令数据生成流水线；一个 CLIP ViT-L/14 + 线性 projector + Vicuna 的简洁 LMM；包含 COCO 与 In-the-Wild 两部分的 LLaVA-Bench；以及公开代码、数据、权重和 demo。其方法论重点是“数据重构”：不是重新采集 15.8 万幅图，而是把已有字幕与框标注重写为助手可学习的交互格式。

![](assets/LLaVA%20-%20视觉指令微调/figures/01-data-generation.png)

> **Table 1，PDF 第 3 页。** 上部用五条字幕与对象框坐标把图像符号化给 text-only GPT-4；下部生成对话、详细描述、复杂推理三类响应。图像仅供论文读者参考，教师实际没有看图。该流程低成本扩展了指令多样性，但生成质量受原始字幕/框信息上限约束。来源：https://arxiv.org/abs/2304.08485

---

## 2. 任务定义与方法

### 2.1 输入、输出与架构

输入是一幅图像 $X_v$、系统提示、当前及历史用户指令；输出是自回归助手文本。视觉编码器 $g$ 使用冻结 CLIP ViT-L/14，取最后一层前或最后一层 patch grid feature $Z_v$。一个可训练线性矩阵 $W$ 把视觉维度投影到 Vicuna 词嵌入空间：

$$H_v=W\,Z_v,\qquad Z_v=g(X_v)$$

视觉 token $H_v$ 与文本 instruction token $H_q$ 直接拼接送入 Vicuna。与 Flamingo 的层间 cross-attention、BLIP-2 的 Q-Former 相比，这个连接器没有 query 压缩，也不在语言层内新增视觉专家；其优点是结构极简、迭代快，缺点是视觉—语言对齐与复杂推理主要由数据和 Vicuna 微调承担。

![](assets/LLaVA%20-%20视觉指令微调/figures/02-architecture.png)

> **Figure 1，PDF 第 4 页。** CLIP 输出 patch 特征，经单个 Projection $W$ 变成与语言 embedding 同维的视觉 token，再与用户指令一起输入语言模型生成响应。图的简洁正是论文主张：先以最小结构隔离数据因素，验证视觉指令微调本身是否有效。来源：https://arxiv.org/abs/2304.08485

### 2.2 GPT 辅助指令数据生成

作者从 COCO 图像出发，把多条人工字幕与对象类别、归一化框坐标作为 text-only GPT-4 可读的视觉代理。少量人工编写的 in-context 示例约束教师生成三类数据：

- **Conversation**：58K，多轮询问对象、数量、动作、位置与关系；
- **Detailed description**：23K，要求完整描述场景；
- **Complex reasoning**：77K，在可见内容上进一步提出逻辑与常识问题。

合计 158K 独立语言—图像指令样本。论文早期比较 ChatGPT 与 GPT-4，认为 GPT-4 在空间推理等方面质量更高。需要强调：教师并不直接读取图像，所谓视觉推理监督是从字幕和框派生；若上下文漏掉对象或关系，教师无法补回可靠视觉证据。

### 2.3 多轮训练目标

多轮对话中，第一轮随机排列图像与问题，之后只追加文本问题。损失仅计算 assistant answer 与停止符，不在 system/user token 上训练。若目标回答 token 序列长度为 $L$：

$$p(X_a\mid X_v,X_{instruct})=\prod_{i=1}^{L}p_\theta(x_i\mid X_v,X_{instruct,<i},X_{a,<i})$$

这个标准 next-token objective 没有额外视觉损失；视觉 grounding 完全由图像 token 对回答 token 的条件预测产生。因此数据是否要求精确引用图像，是决定模型会“看图回答”还是只学回答风格的关键。

---

## 3. 训练数据与训练流程

### 3.1 Stage 1：特征对齐

作者从 CC3M 过滤出 CC-595K：先提取 caption noun phrase，忽略频次低于 3 的短语，对频次高于 100 的短语最多采 100 条，以较少样本维持概念覆盖。每个样本采用简单“请简短描述图像”的单轮格式。冻结 CLIP 与 Vicuna，只训练 $W$，相当于为冻结 LLM 学一个兼容视觉 tokenizer。

训练 1 epoch，batch size 128，学习率 $2\times10^{-3}$，8×A100 约 4 小时。这个阶段若省略，ScienceQA 从 90.92% 降至 85.81%，说明直接用 158K 指令数据同时学习模态对齐与任务跟随并不充分。

### 3.2 Stage 2：视觉指令微调

视觉编码器继续冻结，更新 projector 与 Vicuna。多模态 chatbot 使用 LLaVA-Instruct-158K，三类响应均匀采样；训练 3 epoch，batch size 32，学习率 $2\times10^{-5}$，8×A100 约 10 小时。ScienceQA 则把问题与上下文作为 instruction，把推理与答案作为 assistant 输出，训练约 4 小时。

优化采用 Adam、无 weight decay、cosine schedule、3% warmup、FSDP、gradient checkpointing、BF16/TF32。相较 Flamingo/BLIP-2 的大规模从头桥接预训练，LLaVA 的复现成本低得多；但原始 13B Vicuna 权重获取受其许可证与 LLaMA 基座条件约束，GPT-4 数据生成还涉及服务成本与版本不可控性。

---

## 4. 实验与消融

### 4.1 LLaVA-Bench 与 GPT-4 裁判

LLaVA-Bench (COCO) 取 30 张 COCO-Val-2014 图像，每图三类问题，共 90 条；In-the-Wild 取 24 张跨场景图像与 60 条人工问题。候选模型看真实图像作答，text-only GPT-4 基于人工描述/框生成参考答案并作为裁判，从 helpfulness、relevance、accuracy、detail 四方面给 1–10 分，报告相对参考分数。

| 模型 | Conversation | Detail | Complex reasoning | All |
|---|---:|---:|---:|---:|
| OpenFlamingo | 19.3 ± 0.5 | 19.0 ± 0.5 | 19.1 ± 0.7 | 19.1 ± 0.4 |
| BLIP-2 | 54.6 ± 1.4 | 29.1 ± 1.2 | 32.9 ± 0.7 | 38.1 ± 1.0 |
| LLaVA | **57.3 ± 1.9** | **52.5 ± 6.3** | **81.7 ± 1.8** | **67.3 ± 2.0** |

LLaVA 在复杂推理上优势最大，支持“指令数据改变回答行为”的结论。但 GPT-4 同时参与训练数据生成、参考答案生成和评分，存在教师偏好闭环；90/60 问题的规模也很小。因此这些数字适合比较同一协议下的 instruction-following，不应等价于客观视觉事实准确率。

### 4.2 数据消融

![](assets/LLaVA%20-%20视觉指令微调/figures/03-ablation-results.png)

> **Table 4–5，PDF 第 7 页。** 上表显示完整三类训练数据 Overall 85.1；仅 conversation 为 73.8；完全无 instruction tuning 仅 21.5。下表显示 LLaVA 在 In-the-Wild 的 Overall 67.3，高于 BLIP-2 38.1 与 OpenFlamingo 19.1。图表直接证明增益不只是 projector 连接，而主要来自多样指令监督，尤其详细描述和复杂推理样本。来源：https://arxiv.org/abs/2304.08485

在 COCO benchmark 上，完整数据为 85.1；去掉 conversation、只保留 detail+complex 为 81.9；以 conversation 为主、只加 5% detail 与 10% complex 为 80.5；仅 conversation 为 73.8；无指令微调为 21.5。完整复杂推理分数 96.5，而仅 conversation 为 84.9。不同数据类型并非只提升对应子任务：加入少量推理数据也提升对话，说明“回答深度”可以跨格式迁移。

### 4.3 ScienceQA 与设计消融

LLaVA-13B 在 ScienceQA 平均准确率 90.92%，接近当时 MM-CoT-Large 的 91.68%；与 text-only GPT-4 通过“有分歧时由 GPT-4 再判断”的 ensemble 达 92.53%。这项 SOTA 依赖外部 GPT-4 裁判，不应归功于单体 LLaVA。

设计消融中，取 CLIP 倒数第二层特征为 90.92%，最后层为 89.96%；从头跳过 Stage 1 对齐为 85.81%；7B 模型为 89.84%。reason-first 能更快收敛，但最终精度贡献有限。结果支持局部视觉细节在倒数第二层更充分、预对齐不可省、13B 相对 7B 有约 1.08 点收益。

### 4.4 定性能力

![](assets/LLaVA%20-%20视觉指令微调/figures/04-qualitative-result.png)

> **Figure 2，PDF 第 16 页。** 用户给出手绘网页草图并要求生成交互式 HTML/JS；LLaVA 输出可运行页面，论文同时标出一个需修正的小错误，并承认布局与交互仍未完全遵循意图。这是有解释价值的结果：视觉输入不只触发描述，还可作为开放任务规格，但代码正确性和细节忠实性仍不稳定。来源：https://arxiv.org/abs/2304.08485

---

## 5. 局限与失败案例

### 5.1 视觉粒度、幻觉与评测偏差

论文的典型失败是把“冰箱里同时有草莓和原味/蓝莓酸奶”组合误答成“有草莓味酸奶”。作者称模型有时把图像当作 bag of patches，能识别局部概念却不能稳健建模复杂组合关系。高分辨率品牌文字、细小对象、精确计数和多语 OCR 也明显受 CLIP 224/336 尺度和线性投影限制。

合成指令数据由文本描述和框派生，教师可能生成看似合理但图像无法验证的细节；模型又以自回归似然学习这些答案，容易形成“风格对齐强、视觉忠实度弱”。GPT-4-as-judge 对长而完整的 GPT 风格回答可能有偏好，且裁判看到的是文本化视觉信息而非原图。

### 5.2 Future work

论文明确把更强连接器——如 Flamingo gated cross-attention、BLIP-2 Q-Former——留作未来工作，并把本文定位为 visual instruction tuning 的第一步。后续还需要更全面的学术 benchmark、更高分辨率与细粒度视觉能力、更可靠的多模态事实评测，以及超越日常图像场景的任务覆盖。论文没有给出系统安全对齐、拒答或事实校准方案。

---

## 6. 与后续工作的关系

LLaVA 与 CLIP 的关系最直接：它把 CLIP 从相似度模型改造成视觉 token 生成器；与 Flamingo/BLIP-2 的差别则在训练对象——LLaVA Stage 2 更新 Vicuna，让语言模型本身适应视觉指令，而前两者强调冻结 LM、训练桥梁。它证明数据与开放实现可以比复杂架构更快形成社区生态。

后续 LLaVA-1.5 使用 MLP projector、更多学术 VQA 数据与更高分辨率，缓解原版 benchmark 覆盖和线性投影不足；LLaVA-NeXT、OneVision 继续扩展动态分辨率、多图和视频。CogVLM 则代表另一种回应：如果视觉 token 与文本 token 共用全部语言参数，深层处理中会竞争容量，因此为视觉 token 增加独立专家 QKV/FFN，同时冻结语言路径。

LLaVA 最深远的影响是确立了“视觉编码器 + connector + 开源 LLM + 指令数据”的模块化配方，以及用强语言模型合成多模态监督的 data engine。今天许多 LMM 的差异都可以被理解为分别升级这四个部件。

---

## 7. 个人判断

LLaVA 的突破主要是产品接口与数据范式，而不是网络新颖性。它把用户意图、视觉证据和长文本回答放进统一对话序列，证明 15.8 万条合成指令足以显著改变一个视觉语言模型的行为。结构越简单，论文越能清楚显示：没有 instruction tuning 时 Overall 只有 21.5，完整数据达到 85.1，真正的变量是监督格式和多样性。

复现成本是五篇中最低的一档：官方代码、数据和权重完整，8×A100 数小时级训练；小规模版本还能进一步降低门槛。但训练与评测都高度依赖 GPT-4，视觉细节又受冻结 CLIP 和单层投影限制。综合判断：LLaVA 不是最强视觉表征方法，却是现代开源 LMM 工程配方的奠基工作；它把研究重心从“如何做一个视觉任务”转向“如何让通用语言模型按视觉指令切换任务”，同时也开启了合成数据闭环、视觉幻觉与裁判偏差的新问题。