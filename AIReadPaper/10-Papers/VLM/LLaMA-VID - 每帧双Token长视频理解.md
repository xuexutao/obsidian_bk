---
type: paper-note
status: complete
domain: VLM
paper: "LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models"
year: 2023
arxiv: "2311.17043"
doi: null
source: https://arxiv.org/abs/2311.17043
project: null
code: https://github.com/dvlab-research/LLaMA-VID
tags:
  - VLM
  - Video-LMM
  - Long-Video
  - Token-Compression
  - Visual-Grounding
created: 2026-10-16
updated: 2026-10-16
---

# LLaMA-VID：每帧双 Token 的长视频理解

## 基本信息

| 项目 | 内容 |
|---|---|
| 正式标题 | LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models |
| 作者 | Yanwei Li, Chengyao Wang, Jiaya Jia |
| 论文版本 | arXiv:2311.17043v1，2023-11-28 |
| 核心任务 | 用视觉语言模型理解单图、短视频与小时级长视频 |
| 关键组件 | EVA-G 视觉编码器、Q-Former/BERT 文本解码器、context attention、双线性投影器、Vicuna LLM |
| 核心压缩率 | 长视频默认每帧 1 个 context token + 1 个 content token |
| 训练资源 | 每阶段 1 epoch；8×A100；论文称整体可在单机约 2 天内完成 |
| 本地原文 | [paper.pdf](assets/LLaMA-VID%20-%20每帧双Token长视频理解/sources/paper.pdf) |
| 来源清单 | [sources.md](assets/LLaMA-VID%20-%20每帧双Token长视频理解/sources.md) |

## 一句话结论

LLaMA-VID 不再把一帧图像的几百个 patch token 全部送入 LLM，而是用一个**随用户问题变化的 context token**和一个**保留全局视觉内容的 content token**表示每帧；这种“问题相关检索 + 问题无关保底”的双通道压缩把长视频视觉长度降到每帧 2 token，并在视频 QA 上优于当时的 7B 级 baseline。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

图像 VLM 常把 ViT 的 patch 序列经 connector 投影后直接送入 LLM。以 224×224、patch size 14 为例，一帧有 256 个 patch；LLaVA 一类方法在更高分辨率下可超过 256 token/图。若视频以 1 FPS 采样 10,000 帧，视觉前缀就会达到几十万乃至数百万 token，标准 2K/4K 上下文完全无法容纳，注意力与 KV cache 成本也会随长度快速增长。

问题因此不是“能否把图像模型用于视频”，而是：**能否在不彻底丢失问题相关细节的前提下，把每帧压缩到常数个 token，并让现有 LLM 处理小时级视频？**

### 1.2 现有方法的瓶颈

当时的代表方案有两类：一类用 Q-Former/learned queries 把图像压成约 32 个 token；另一类在时空维度做平均池化。前者对 10K 帧仍会产生 320K token，后者若直接沿时间强压缩，会把短暂事件和长程顺序抹掉。LLaMA-VID 的判断是：长视频必须先把**单帧成本**压到极低，同时保留一个可由问题驱动的选择机制。

### 1.3 核心贡献

1. **双 Token 表示。** 每帧拆成一个指令条件化的 context token 与一个视觉内容 content token，视频默认总计仅 2 token/帧。
2. **Context attention。** 用户指令经文本解码器形成 query，用视觉 patch 同时充当 key/value，直接检索与问题最相关的区域。
3. **弹性内容预算。** content token 数量 $n$ 可在 1 到原始 patch 数 $N$ 之间调整：视频取 $n=1$，单图可保留全部 patch，因此同一框架覆盖长视频和精细图像任务。
4. **三阶段训练与 Long-VideoQA。** 在模态对齐、通用指令微调后，以电影脚本/梗概构造 9K 电影问答，并加入 6K 长文本数据把 Vicuna 的上下文扩到 64K。
5. **统一验证。** 同时报告视频 QA、生成式视频评测、图像 VQA 与 token/decoder 消融，证明压缩效率不是单纯依赖牺牲全部性能获得的。

![](assets/LLaMA-VID%20-%20每帧双Token长视频理解/figures/01-benchmark-overview.png)

> 图 1：LLaMA-VID 在视频指标（蓝色）和图像指标（紫色）上的总体表现。雷达图说明方法并非只优化长视频吞吐，也试图保持通用图像能力；但不同轴量纲不一致，应以正文表格的逐项数字为准。来源：论文 Figure 1，第 1 页。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与基本假设

输入可以是单图 $V$，也可以是按时间顺序采样的视频帧 $\{V_t\}_{t=1}^{T}$，另有用户指令文本 $U$；长视频阶段还可把第 $t$ 帧的字幕 $S_t$ 与其视觉 token 相邻插入。输出是 LLM 自回归生成的自然语言回答 $Y=(y_1,\ldots,y_L)$，任务包括开放式视频问答、描述、摘要、情节与细节推理。

模型依赖三项假设：第一，许多视频问题只需要每帧少量与指令相关的信息；第二，即便问题检索失准，一个全局 content token 仍可提供视觉保底；第三，帧按时间排列后，LLM 的一维位置编码足以承载基本时间顺序，不必额外引入视频专用时空 Transformer。

### 2.2 Token shape 总览

| 阶段 | 张量 | Shape | 含义 |
|---|---|---|---|
| 原始帧 | $V_t$ | $H\times W\times3$ | 第 $t$ 帧 RGB 图像 |
| 视觉编码 | $X_t$ | $N\times C$ | ViT patch 序列，$N=(H/p)(W/p)$，$p=14$ |
| 文本引导查询 | $Q_t$ | $M\times C$ | 文本解码器结合指令与视觉生成的 $M$ 个 query |
| 相关性矩阵 | $A_t$ | $M\times N$ | 每个 query 对每个视觉 patch 的注意力 |
| context embedding | $E_t$ | $1\times C$ | 对 $M$ 个问题相关视觉摘要取均值 |
| context token | $E_t^T$ | $1\times D$ | 投影到 LLM hidden size $D$ |
| content token | $E_t^V$ | $n\times D$ | 池化后投影，$1\le n\le N$ |
| 单帧最终表示 | $Z_t$ | $(n+1)\times D$ | $[E_t^T;E_t^V]$ |
| 视频视觉序列 | $Z$ | $T(n+1)\times D$ | 长视频默认 $n=1$，故长度为 $2T$ |

视觉编码公式为：$ X_t=f_v(V_t)\in\mathbb{R}^{N\times C},\quad N=(H/p)(W/p) $

当单图分辨率为 336×336、$p=14$ 时，$N=24^2=576$。论文的单图设置使用 1 个 context token 加 576 个 content token，共 577 token；长视频把 $n$ 设为 1，因此每帧仅 2 token。

### 2.3 训练目标

论文沿用 LLaVA/Vicuna 式监督微调：视觉 token 与用户文本作为条件，只对 assistant 回答位置计算 next-token cross entropy。可写为：$ \mathcal{L}_{\mathrm{SFT}}=-\sum_{i=1}^{L}\log p_\theta(y_i\mid Z,U,y_{<i}) $

论文没有单独定义视觉重建、对比学习或 temporal loss；所有视觉选择能力最终都受语言生成损失监督。实现上通常把视觉前缀、用户指令和 padding 的 label 置为 ignore，仅保留回答 token 的监督；这是其所依赖的 LLaVA 训练范式，论文未逐项披露 label mask 代码。

---

## 3. 核心方法

### 3.1 总体框架与完整信息流

![](assets/LLaMA-VID%20-%20每帧双Token长视频理解/figures/02-method-pipeline.png)

> 图 2：每帧先经冻结视觉编码器得到 patch 序列；用户问题和视觉特征进入文本解码器形成 query；上支路生成问题相关 context token，下支路池化出 content token，二者按时间拼接后输入 LLM。来源：论文 Figure 2，第 3 页。

完整 pipeline 是：采样视频帧 → EVA-G/ViT 编码 patch → Q-Former 或两层 BERT 生成文本引导 query → context attention 从 patch 中读取相关区域 → 独立线性 projector 对齐到 LLM hidden space → content 分支按预算做二维平均池化并投影 → 每帧拼成 $[context,content]$ → 按时间与指令/字幕组成一维 token 序列 → Vicuna 自回归回答。

这种设计的关键不只是“把 patch 平均成两个 token”。两个 token 的角色是互补的：context token 具有高选择性但受问题约束，content token 与问题无关但更稳健。前者像可学习的检索结果，后者像全局缩略图。

### 3.2 Encoder 与 connector

视觉编码器默认是预训练 EVA-G，训练全程冻结。对每帧独立编码意味着模型本身没有视觉域内的跨帧 self-attention；跨帧关系完全交给 LLM 在压缩后的 token 序列上建模。优点是显存与实现复杂度低，缺点是高速运动、局部轨迹等需要帧间细粒度对应的现象会在进入 LLM 前丢失。

文本解码器默认采用预训练 Q-Former，也实验了随机初始化的两层 BERT。它接收用户文本并与 $X_t$ 进行跨模态交互，输出 $Q_t\in\mathbb{R}^{M\times C}$。随后有两个彼此独立的线性 projector：`ctxproj` 负责 context embedding，`visproj` 负责池化后的 content embedding。二者输出维度均匹配 LLM 的 $D$，但参数不共享，从而允许“检索摘要”和“全局内容”学习不同的对齐变换。

### 3.3 Context attention：Q/K/V 如何构造

LLaMA-VID 最有辨识度的操作是一个轻量 dot-product attention。这里 query 是文本解码器输出 $Q_t$，key 与 value 都来自视觉 patch $X_t$：$ Q=Q_t,\quad K=X_t,\quad V=X_t $

按伪代码，相关性与归一化为：$ A_t=\operatorname{Softmax}(Q_tX_t^\top/\sqrt{C})\in\mathbb{R}^{M\times N} $

每个 query 对视觉 value 加权后得到 $M$ 个摘要，再沿 query 维平均成单个向量：$ E_t=\operatorname{Mean}_{M}(A_tX_t)\in\mathbb{R}^{1\times C} $

最后投影为：$ E_t^T=W_{ctx}E_t+b_{ctx}\in\mathbb{R}^{1\times D} $

论文正文公式省略了 $1/\sqrt{C}$，但 Algorithm 1 的伪代码明确执行了缩放。该 attention 没有再学习一套显式的 $W_Q,W_K,W_V$；表示变换主要已发生在视觉编码器、文本解码器和末端 projector 中。与 Q-Former 直接把 32 个 query 全部作为 LLM token 不同，这里把它们进一步聚合成 1 个 context token。

![](assets/LLaMA-VID%20-%20每帧双Token长视频理解/figures/05-context-attention.png)

> 图 6：不同问题使 query 高响应区域发生变化，例如“是否打斗”集中到象鼻接触区域，“渔村”关注河岸建筑与海鸟。这验证 context token 确实是 instruction-conditioned，而非固定全局池化。来源：论文 Figure 6，第 8 页。

### 3.4 Content token：可伸缩的视觉保底

content 分支不读取文本。它把 $X_t$ 还原为空间网格，按目标边长做二维平均池化，再展平为 $n$ 个 token 并用 `visproj` 投影。其抽象形式为：$ E_t^V=W_{vis}\operatorname{Pool}_n(X_t)+b_{vis}\in\mathbb{R}^{n\times D} $

$n=N$ 时不压缩，适合单图细节；$n=64,16,4$ 是中间预算；$n=1$ 是全局平均，适合长视频。最终每帧表示为：$ Z_t=[E_t^T;E_t^V]\in\mathbb{R}^{(n+1)\times D} $

如果仅做 content 全局平均，模型不知道当前问题应关注何处；如果仅做 context token，则视觉内容完全受 query 选择影响，容易遗漏。二者拼接是一个极小但有效的 mixture：一个 token 做条件化稀疏读取，一个 token 保留无条件低频信息。

### 3.5 Mask、位置编码与时间顺序

LLaMA-VID 没有提出新的 attention mask。LLM 使用标准 decoder-only causal mask：回答 token 可关注全部先前的指令、视觉与字幕 token，但不能看未来回答。视觉 token 作为输入 embedding 参与前缀注意力，其位置不承担语言监督。视觉编码器和 Q-Former 内部沿用各自预训练 attention/mask，论文未修改。

短上下文阶段沿用 Vicuna/LLaMA 的 RoPE。长视频阶段采用 position interpolation，把原 4K RoPE 扩展到 64K，再用 Long-VideoQA 做监督微调。帧序列按 $t=1,2,\ldots,T$ 排列；若有字幕，则采用 `<image-0><subtitle-0>,...,<image-i><subtitle-i>` 的交错格式。因此时间信息主要来自**一维绝对顺序和 RoPE 相对相位**，没有显式时间戳 token、时间坐标或跨帧位置编码。

这也划定了能力边界：LLM 可以对低频采样下的事件顺序做语言推理，但精确持续时间、快速运动与密集轨迹并非该结构的强项。

### 3.6 复杂度与 KV cache

将每帧从 $N$ 个视觉 token 压到 2 个，视觉前缀长度由 $TN$ 降为 $2T$。仅比较 LLM self-attention 主项，视觉部分理论计算量比例约为：$ \frac{(2T)^2}{(NT)^2}=\frac{4}{N^2} $

论文以 256 patch 的未压缩设置比较，称每帧 2 token 相当于 token 数降至 $1/128$；实际端到端成本还包括逐帧 ViT 与 Q-Former，因此不能把该比值直接当作整模型加速比。另一个工程代价是 context token 依赖用户问题：问题改变时需重新运行文本解码器/context attention，无法像纯 content embedding 那样对视频做完全问题无关的永久缓存。

---

## 4. 数据集与训练设置

### 4.1 三阶段训练

![](assets/LLaMA-VID%20-%20每帧双Token长视频理解/figures/03-training-stages.png)

> 图 3：训练先做图像/视频 caption 对齐，再混合文本、图像和视频指令，最后可选地加入电影级长视频与长文本。长视频阶段把帧与字幕交错，直接训练 64K 序列。来源：论文 Figure 3，第 4 页。

| 阶段 | 数据 | 可训练模块 | 最大长度 | Batch / LR |
|---|---|---|---:|---|
| 1. Modality Alignment | 558K CC3M 图像 caption + 232K WebVid 视频 caption，共 790K | context attention 与两个 projector；视觉编码器、文本解码器、LLM 冻结 | 2,048 | 256 / $10^{-3}$ |
| 2. Instruction Tuning | 40K ShareGPT 文本对话 + 625K 图像 QA + 98K 视频 QA | 除视觉编码器外均训练 | 2,048 | 128 / $2\times10^{-5}$ |
| 3. Long Video Tuning | 9K 电影级 QA + 6K LongAlpaca/LongLoRA 长文本，共 15K | 附录说明文本解码器再次冻结；其余按长上下文 SFT | 65,536 | 8 / $2\times10^{-5}$ |

三个阶段均训练 1 epoch，使用 AdamW、cosine decay、warmup ratio 0.03、weight decay 0、DeepSpeed ZeRO-2，硬件为 8×A100。视频以 1 FPS 抽帧。

### 4.2 Long-VideoQA 的数据工程

作者从 MovieNet 选取 400 多部长电影，以 synopsis 和 script 作为教师模型的文本代理：GPT-4 根据 synopsis 生成简/详摘要、情节和角色问答；Claude-2 根据完整 script 生成情节因果推理与细节描述。最终约 1K summary、4K plot/character QA、4K reasoning/detail QA，共 9K 电影数据，再加入 6K 长文本指令用于上下文扩展。

这套数据工程把昂贵的视频逐帧人工标注替换为“已有电影结构化文本 → 教师 LLM 生成监督”。它能教会模型长叙事回答格式，却存在明显偏差：细节问题声称需要视觉信息，但教师主要依据 script；模型可能学到字幕/电影先验，而非真正从两枚视觉 token 中恢复细粒度证据。

### 4.3 Baseline、数据集与指标

视频主基准包括 MSVD-QA、MSRVTT-QA、ActivityNet-QA，以及 Video-ChatGPT 的生成式评测（Correctness、Detail、Context、Temporal、Consistency）。baseline 包括 FrozenBiLM、VideoLLaMA、LLaMA-Adapter、VideoChat、Video-ChatGPT、BT-Adapter。图像侧使用 GQA、MMBench、MME、POPE、SEED、ScienceQA-Image、VizWiz、VQAv2，并与 InstructBLIP、IDEFICS、Qwen-VL、LLaVA-1.5 等比较。

公平性要分开看：视频表 1/2 使用阶段 1+2、每帧 2 token，明确不含长视频阶段；图像表 3 与 LLaVA-1.5 对齐训练数据，并使用 1 context + 576 content token。因此“视频高效”和“图像领先”不是同一个 token 预算下的结论。

---

## 5. 实验结果

### 5.1 视频主结果

| 模型 | LLM | MSVD Acc / Score | MSRVTT Acc / Score | ActivityNet Acc / Score |
|---|---|---:|---:|---:|
| Video-ChatGPT | Vicuna-7B | 64.9 / 3.3 | 49.3 / 2.8 | 35.2 / 2.7 |
| BT-Adapter | Vicuna-7B | 67.5 / 3.7 | 57.0 / 3.2 | 45.7 / 3.2 |
| LLaMA-VID | Vicuna-7B | **69.7 / 3.7** | **57.7 / 3.2** | **47.4 / 3.3** |
| LLaMA-VID | Vicuna-13B | **70.0 / 3.7** | **58.9 / 3.3** | **47.5 / 3.3** |

7B 模型相对 BT-Adapter 的准确率增益分别为 +2.2、+0.7、+1.7 个百分点；13B 进一步提高，但幅度并不均匀，说明瓶颈不只在语言模型规模。在生成式评测上，13B 的 Correctness/Detail/Context/Temporal/Consistency 为 3.07/3.05/3.60/2.58/2.63，五项均高于表中 baseline。

### 5.2 图像结果及其正确解读

图像设置用 577 token/图，而非 2 token/图。Vicuna-7B 版本相对 LLaVA-1.5-7B 在 GQA、MME、VizWiz 上由 62.0/1510.7/50.0 提升到 64.3/1521.4/54.2；MMBench、POPE、SEED、ScienceQA、VQAv2 也有小幅提升。这里证明的是“额外 context token 可提升问题相关视觉读取”，不能据此断言两 token 图像表示全面超越 LLaVA。

### 5.3 双 Token 与压缩比例消融

论文 Table 4 在 1-token/分支设置下给出：仅 content 为 53.3/80.9/66.1/46.5，仅 context 为 54.3/82.4/67.7/48.3，两者合用达到 55.5/83.1/68.8/49.0（GQA/POPE/SQAI/TextVQA）。context 单独就比 content 单独更强，表明问题条件化选择确实有价值；两者继续互补，则说明全局视觉保底没有冗余。

![](assets/LLaMA-VID%20-%20每帧双Token长视频理解/figures/04-token-ablation.png)

> 表 5–6：content 从 256 压到 1 时，多数细节任务逐步下降，但 ScienceQA 在极压缩时仍保持；加入 context token 比无 context 的 256-token 基线更好。预训练 Q-Former 又明显优于随机两层 BERT和无文本解码器。来源：论文 Tables 5–6，第 8 页。

压缩消融体现了清晰的 rate–accuracy trade-off：content=256 且无 context 时 GQA 为 61.9；加 1 个 context 后升至 63.0；随后 content=64/16/4/1 时 GQA 依次为 60.8/58.2/56.2/55.5。极限压缩不是无损的，论文自己报告相对未压缩有约 2%–6% 的性能下降。双 Token 的价值在于把下降控制在可用范围，同时让小时级视频首次进入上下文窗口。

### 5.4 长视频证据是否充分

模型可在 64K 上下文中输入 3 小时以上视频，论文给出 Avatar、Titanic、Interstellar 等定性对话，并与只看片名的 LLaMA 2、只看字幕的 LongLoRA 比较。案例显示 LLaMA-VID 在部分情节问题上更准确，但没有专门的小时级、视觉证据可定位的标准化基准；而电影训练数据又直接使用 script/synopsis。因此“能装入 3 小时视频”证据充分，“真正理解 3 小时内细粒度视觉事件”证据仍偏弱。

---

## 6. 与相关工作的关系

相对 BLIP-2/InstructBLIP，LLaMA-VID 不把全部 32 个 Q-Former query 送入 LLM，而是再聚合为 1 个问题相关 token；相对 LLaVA，它在 patch projector 前加入文本条件化读取，并允许 content 动态池化；相对 Video-ChatGPT 的固定时空池化，它避免直接把所有问题都压成同一个视频向量；相对后续 LongVA 一类“扩 LLM 上下文、保留更多视觉 token”的路线，它站在相反端点：优先把单帧压到 2 token，再扩位置长度。

因此 LLaMA-VID 更适合**极长、低帧率、问题驱动检索**，而不适合要求像素级时空对应、密集动作识别或无问题条件的通用视频表征学习。

---

## 7. 局限与批判性评价

1. **极端压缩必然丢细节。** content=1 是全局平均；小目标、OCR、短暂出现物体与相似实例区分很容易被抹掉，消融中的 2%–6% 下降已显示该代价。
2. **问题相关 token 需要重新计算。** context token 随用户指令变化，多轮对话的新问题不能完全复用旧视觉 token，视频预计算与在线响应之间存在工程张力。
3. **缺少显式时序建模。** 帧独立编码，LLM 只看到每帧两个向量及序列位置；运动方向、速度和跨帧身份保持没有专用机制。
4. **长视频训练监督有文本捷径。** 电影 QA 由 synopsis/script 和教师 LLM 生成，可能奖励剧情知识、字幕与语言先验，而不是视觉证据。
5. **小时级评测偏定性。** 主定量基准仍以短视频为主；长电影案例没有统一的时间定位、证据引用和人工盲评协议。
6. **“计算降至 1/128”只针对 token 数。** ViT/Q-Former 仍需逐帧运行，长序列 LLM 的 KV cache 也仍随 $2T$ 增长，不能等价为端到端 128 倍加速。
7. **Context attention 聚合过早。** $M$ 个 query 的结果最终求均值，query 间的多样性可能互相抵消；复杂问题涉及多个对象或事件时，一个 context token 的容量有限。

---

## 8. 复现与实践建议

### 8.1 最小复现路径

1. 使用官方仓库与论文默认 EVA-G、Q-Former、Vicuna-7B 配置，先复现 224 分辨率、每帧 2 token 的视频 QA。
2. 检查张量断言：视觉输出 `[B,N,C]`，文本 query `[B,M,C]`，attention `[B,M,N]`，context `[B,1,D]`，content `[B,n,D]`，最终 `[B,n+1,D]`。
3. 单独单元测试 context attention，确认 softmax 沿 patch 维 $N$，mean 沿 query 维 $M$；若维度弄反，仍能运行但语义完全不同。
4. 先做阶段 1，只训练 `ctxproj`、`visproj` 与 context attention；验证 image/video caption loss 收敛后再解冻阶段 2 模块。
5. 用 Table 4 的三组输入（content-only、context-only、双 token）验证实现，而不是直接投入长视频训练。
6. 最后做 RoPE position interpolation 与 64K SFT；先用合成长序列测试显存、mask、position id 与截断行为，再加载电影数据。

### 8.2 工程检查点

- **帧采样：** 论文为 1 FPS；复现对比必须固定采样率，否则 token 数和信息密度不可比。
- **位置：** 确认 64K position interpolation 与 checkpoint 一致，不能只修改 tokenizer 的 `model_max_length`。
- **Mask：** visual/user token 不计语言 loss；padding mask 与 causal mask 必须同时正确。
- **缓存：** 可缓存 $X_t$ 或 content token，但 context token 与问题相关；服务化时应把两者区分。
- **数值稳定：** context dot product 应除以 $\sqrt{C}$；长序列建议使用 bf16、FlashAttention 与 ZeRO。
- **评测：** 视频 QA 用论文指定 GPT 评判协议时要固定 judge 版本与 prompt；否则分数不可直接横比。

### 8.3 资源与可复现性评估

官方开放代码，论文披露了阶段数据量、冻结策略、主要优化超参数与 8×A100 配置，可复现性在同代视频 LMM 中较好。难点主要是 EVA-G/Q-Former/Vicuna 权重组合、多个混合数据集的许可与预处理，以及 64K 训练显存。若资源有限，最有研究价值的精简实验是：冻结视觉/LLM，只训练双 projector 与 context attention，在短视频或图像 VQA 上验证 token 类型消融。

---

## 9. Future Work 与个人启发

### 9.1 值得继续的方向

1. **从固定 2 token 变为按帧自适应预算。** 镜头切换、文本密集帧或问题相关帧分配更多 content token，静态背景帧保留 1–2 个。
2. **两阶段检索。** 先用廉价双 token 扫完整视频，再对高相关时间段回取原始 patch，实现 coarse-to-fine 视觉检索。
3. **问题无关缓存 + 问题相关轻量读取。** 离线保存视觉 key/value，在线只计算 query 与 context attention，降低多轮问答成本。
4. **加入时间坐标与证据输出。** 回答同时预测帧区间，使“理解”能被定位验证，而不只由语言流畅度判断。
5. **保持多个 context slots。** 用可学习门控从 $M$ 个 query 中保留 $k>1$ 个互补摘要，处理多对象、多事件问题。

### 9.2 个人判断

这篇论文最重要的贡献不是宣称“两枚 token 足以表达一帧”，而是把视觉压缩拆成了**条件化选择**与**无条件保底**两个不同信息通道。它提示后续长视频系统不必在“保留全部 patch”和“固定全局池化”之间二选一；更合理的方向是让 token 预算由任务、问题与时间段共同决定。

---

## 10. 材料来源

- 官方论文：[arXiv:2311.17043](https://arxiv.org/abs/2311.17043)
- 官方代码：[dvlab-research/LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
- 本地 PDF：[assets/LLaMA-VID - 每帧双Token长视频理解/sources/paper.pdf](assets/LLaMA-VID%20-%20每帧双Token长视频理解/sources/paper.pdf)
- 完整来源与图片溯源：[sources.md](assets/LLaMA-VID%20-%20每帧双Token长视频理解/sources.md)
