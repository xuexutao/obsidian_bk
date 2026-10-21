---
type: paper-note
status: complete
domain: VLM
paper: "Long Context Transfer from Language to Vision"
year: 2024
arxiv: "2406.16852"
doi: null
source: https://arxiv.org/abs/2406.16852
project: null
code: https://github.com/EvolvingLMMs-Lab/LongVA
tags:
  - VLM
  - Video-LMM
  - Long-Context
  - Long-Video
  - RoPE
  - Needle-In-A-Haystack
created: 2026-10-16
updated: 2026-10-16
---

# LongVA：从语言长上下文迁移到长视频

## 基本信息

| 项目 | 内容 |
|---|---|
| 正式标题 | Long Context Transfer from Language to Vision |
| 模型名 | Long Video Assistant（LongVA） |
| 作者 | Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, Ziwei Liu |
| 论文版本 | arXiv:2406.16852v2，2024-07-01 |
| 核心任务 | 将语言模型的长上下文能力零样本迁移到长视频视觉 token 序列 |
| 模型骨架 | Qwen2-7B-Instruct（扩展到 224K）+ CLIP-ViT-L-336px + 2-layer MLP + UniRes |
| 标志性规模 | 144 token/帧；约 2000 帧即 288K 视觉 token；V-NIAH 测到 3000 帧 |
| 训练资源 | 8×A100-80G；文本长上下文训练约 2 天，视觉对齐约 1.5 天 |
| 本地原文 | [paper.pdf](assets/LongVA%20-%20语言长上下文迁移到长视频/sources/paper.pdf) |
| 来源清单 | [sources.md](assets/LongVA%20-%20语言长上下文迁移到长视频/sources.md) |

## 一句话结论

LongVA 证明了一个反直觉现象：**先只用长文本把 LLM 的上下文扩到 224K，再用普通短图像数据做视觉对齐，即使完全不看视频训练数据，也能在测试时读取 2000 帧以上的视觉序列。**它没有把每帧压到极少 token，而是把长视频瓶颈从 connector 转移到语言骨干的上下文长度与系统内存。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

长视频 VLM 通常沿“少生成视觉 token”的路线演进：Q-Former/Perceiver resampler、平均池化、token pruning、时空卷积都试图压缩每帧表示。但只要语言模型仍受 2K/4K/32K 窗口限制，connector 再优化也很难容纳成百上千帧。与此同时，公开视频指令数据大多短于 1 分钟，且文字监督常只描述少量采样帧，专门训练“2000 帧理解”既昂贵又缺数据。

LongVA 提出的问题是：**如果视觉 embedding 已与语言 embedding 对齐，那么语言模型通过长文本训练学到的长距离读取能力，是否会自然泛化到视觉 token？**若答案成立，长视频能力不一定需要长视频监督，关键可能只是先把 LM 的有效上下文做长。

### 1.2 与压缩路线的差异

![](assets/LongVA%20-%20语言长上下文迁移到长视频/figures/01-long-context-transfer.png)

> 图 1：左侧的传统路线在视觉侧压缩帧；右侧 LongVA 先扩展语言模型，再用短图像对齐，测试时直接把大量视频帧作为长视觉序列输入。核心主张是“train short vision, test long video”。来源：论文 Figure 1，第 4 页。

LLaMA-VID 用 2 token/帧减小序列，LongVA 则保留 144 token/帧，让语言模型处理更长序列。两者并不互斥：前者降低每帧成本，后者提高总容量。LongVA 的实验刻意避免复杂视觉压缩，以隔离“上下文来自 LM”这一变量。

### 1.3 核心贡献

1. **Long context transfer。** 发现语言模型经长文本 continued pretraining 后，其有效长度可直接迁移到视觉序列，不需要长视频训练。
2. **UniRes。** 把高分辨率图像视为多网格，把视频视为横向延长的多网格图像，统一训练时图像与测试时视频的展平顺序。
3. **V-NIAH。** 构造 Visual Needle-In-A-Haystack，在任意深度插入一帧反常识视觉“针”，测量模型真正可检索的视觉上下文长度。
4. **LongVA。** 7B 模型可近乎完美读取 2000 帧以内的单帧证据，并在 Video-MME/MLVU 上达到当时 7B 开源模型领先水平。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与训练/测试协议

训练分两部分：先向 Qwen2-7B-Instruct 输入纯文本长序列，扩展其上下文；再输入普通图像—文本/图像指令数据，做 LLaVA-1.6 式视觉对齐与 SFT。视觉训练阶段不包含视频。

测试时输入视频帧序列 $\{V_t\}_{t=1}^{T}$ 与问题 $U$，输出自然语言回答 $Y$。UniRes 把每帧当作一个 336×336 网格，独立编码为 144 个视觉 token，然后按帧序拼接。V-NIAH 的输入是 haystack 视频、随机位置插入的一张 needle 图与定位式问题，输出通常是一个短答案。

### 2.2 Token shape

| 位置 | 张量 | Shape | 说明 |
|---|---|---|---|
| 单个 grid/帧 | $V_t$ | $336\times336\times3$ | 图像网格或视频帧 |
| CLIP patch | $X_t$ | $24\times24\times C_v=576\times C_v$ | ViT-L/14 在 336 分辨率下的 patch 网格 |
| MLP 投影 | $P_t$ | $576\times D$ | 2-layer MLP 对齐到 Qwen2 hidden size |
| 2×2 pool | $Z_t$ | $12\times12\times D=144\times D$ | 每个 grid/帧 144 token |
| $T$ 帧视频 | $Z$ | $144T\times D$ | frame-major 顺序拼接 |
| 语言输入 | $H_0$ | $(144T+L_U)\times D$ | 视觉 token 与问题 token 组成的序列 |

对 $a\times b$ 图像，grid 数约为 $(a//336)(b//336)$，最多 49 个；对 $T$ 帧视频，直接视为尺寸 $336\times(336T)$ 的“长图像”，每帧对应一个 grid。视觉 token 数为：$ L_v=144T $

因此 1555 帧约对应 $1555\times144=223{,}920$ token，接近 224K 训练窗口；2000 帧是 288K token，3000 帧是 432K token，后两者已属于长度外推而非训练窗口内插值。

### 2.3 输出目标与 loss mask

多模态阶段沿用 decoder-only next-token prediction，只对 assistant 回答位置计算交叉熵：$ \mathcal{L}_{\mathrm{SFT}}=-\sum_i\log p_\theta(y_i\mid Z,U,y_{<i}) $

图像/视频 embedding、用户问题与 padding 位置不计语言 loss。论文没有引入视频重建、帧顺序分类或专门时序 loss；长视频能力来自 LM 长上下文训练与短视觉对齐的组合。

---

## 3. 核心方法

### 3.1 先训练长语言模型

作者以 Qwen2-7B-Instruct 为骨干，在 224K context length 上继续预训练 900M 文本 token。数据来自 SlimPajama，提升长度超过 4096 的文档采样权重，同时保持领域混合比例；多个文档以 BOS 分隔后 pack 成长序列。

关键训练设置如下：RoPE base frequency 调到 $10^9$；constant learning rate $10^{-5}$；全局 batch 约 100 万 token；1000 steps；8×A100-80G，约 2 天。论文没有用 LoRA 或近似稀疏注意力，而是完整训练 dense attention。

RoPE 对第 $m$ 个二维通道对的角频率可概括为：$ \omega_m=\theta^{-2m/d},\quad \theta=10^9 $

增大 $\theta$ 会降低高维旋转速度，减小长距离位置相位过快变化。这里不是只在推理时修改配置，而是带着新 RoPE base 在 224K 长文本上训练；实验表明仅对 32K baseline 做 training-free RoPE 外推，改善很有限。

### 3.2 长序列训练系统

224K dense attention 的主要难点是显存与通信。LongVA 组合使用 FlashAttention-2、Ring Attention、activation checkpointing、parameter offload，并在 ring 上用 zigzag sequence sharding 平衡因果 attention 的负载。Qwen2 的 152K 大词表使 embedding/LM head 占用显著；作者指出同样系统下，32K 词表的 LLaMA2 可训练到约 700K context，而 Qwen2 受限于 224K。

这提示“上下文长度”不是单一算法超参，还由 vocabulary、embedding、并行策略和硬件拓扑共同决定。

### 3.3 UniRes：图像与视频统一成 grid 序列

![](assets/LongVA%20-%20语言长上下文迁移到长视频/figures/02-unires-encoding.png)

> 图 2：图像在训练时被切成多个 336×336 grid，视频在推理时把每帧当作一个 grid；CLIP 每 grid 输出 576 patch，经 2×2 average pooling 变成 144 token，再展平送入 LLM。来源：论文 Figure 2，第 5 页。

UniRes 基于 LLaVA-NeXT 的 AnyRes，但做了三处有针对性的修改：去掉低分辨率 base image；在每个 grid 内按 raster order 独立展平，而不是先拼成大特征图后跨 grid 展平；所有 grid 均执行 2×2 average pooling。这样训练图像的多 grid 序列与测试视频的多帧序列具有相同局部结构。

单帧变换可写为：$ Z_t=\operatorname{Flatten}(\operatorname{AvgPool}_{2\times2}(g_\phi(f_v(V_t))))\in\mathbb{R}^{144\times D} $

其中 $f_v$ 是 CLIP-ViT-L-336px，$g_\phi$ 是两层 MLP connector。connector 不做 cross-attention、动态 pruning 或 learnable resampling；这使实验更清楚地把长程能力归因于 LM。

### 3.4 Q/K/V、attention mask 与位置

视觉编码器内部使用标准 ViT self-attention，Q/K/V 只在单个 336 grid 内计算；不同帧在视觉侧互不交互。两层 MLP 也不产生 Q/K/V。真正的跨帧建模发生在 Qwen2：视觉 embedding 与文本 embedding 进入同一个 decoder 序列，沿用 Qwen2 的 self-attention/GQA 参数生成 Q/K/V，LongVA 没有增加视频专用 attention head。

LLM 使用标准 causal mask。问题和回答 token 可以读取其前方全部视觉 token；回答不能读取未来 token。视觉 token 自身由 CLIP 已完成双向编码，因此即使 LLM 侧为因果 mask，每帧内部 patch 的视觉上下文仍完整。论文没有使用帧间局部窗口、block-sparse mask 或视觉全连接 special mask。

位置方面，每个视觉 token 直接占用一个 Qwen2 RoPE position。UniRes 的 frame-major 展平保证第 $t$ 帧 144 个位置连续，第 $t+1$ 帧紧随其后；没有显式 frame id、时间戳、2D RoPE 或独立 temporal position embedding。也就是说，长文本 RoPE 被直接当作视频时间轴，这正是“language-to-vision transfer”的结构基础，也是精确时间建模的限制。

### 3.5 V-NIAH：测量“有效视觉长度”

![](assets/LongVA%20-%20语言长上下文迁移到长视频/figures/03-v-niah-design.png)

> 图 3：V-NIAH 把一张包含反常识答案的 needle 图插入长视频不同位置，并用定位式问题询问针中细节。反常识内容减少模型靠语言常识猜对的可能。来源：论文 Figure 3，第 6 页。

V-NIAH 只设计 5 个 needle QA，人类标注成本很低；haystack 按 1 FPS 采样，needle 放在不同深度与总长度组合中。题目包含“找到某帧”的 locating prompt，内容包括婚礼气球颜色、药片剂量、熊猫科学家等，以降低训练污染和先验猜测。

超过 200K token 时，7B 模型生成所需 KV cache 可接近 100GB。作者因此采用 perplexity-based evaluation：预先保存视觉 embedding，把视觉、问题和正确答案拼接后做一次 ring-attention forward；只有答案 span 的每个位置 argmax 都匹配正确 token 才算正确。这个协议适合测检索上限，却绕过了实际自回归解码时的 KV cache 与采样成本，不能等同于“可交互生成 3000 帧视频答案”。

---

## 4. 训练数据与实现设置

### 4.1 两段能力获取

| 阶段 | 数据模态 | 目标 | 关键设置 |
|---|---|---|---|
| Long-context LM continued pretraining | 纯文本 | 学会在 224K 中读取长距离信息 | 900M token，1000 steps，RoPE base $10^9$，LR $10^{-5}$ |
| Visual alignment + instruction tuning | 仅图像—文本/图像指令 | 把 CLIP/MLP 输出对齐到长上下文 Qwen2 | 沿用 LLaVA-1.6 两阶段 recipe；不含视频 |
| Inference | 图像或视频 | 图像直接输入；视频按多 grid 长图输入 | 144 token/帧，可远超训练时视觉长度 |

论文强调的控制变量是：LongVA 与 LLaVA-NeXT-Qwen2 使用相同多模态数据，差别在 Qwen2 是否经过 224K 文本训练；LongVA (AnyRes) 与 LongVA 仅在视觉展平/池化方案上不同。由此分别隔离 long LM 与 UniRes 的贡献。

### 4.2 训练成本

长文本训练约 2 天，图像对齐约 1.5 天，总计约 3.5 天，硬件均为 8×A100-80G。这里的“学术预算可承受”是相对从头预训练 7B LM 而言；224K dense attention、Ring Attention 与 640GB 总显存仍明显超出单卡/普通工作站。

### 4.3 Benchmark 与 baseline

长上下文诊断使用 V-NIAH；真实长视频主评测是 Video-MME（平均视频约 1017 秒，分 short/medium/long），附录报告 MLVU；短视频还测 NeXTQA、ActivityNetQA、VideoChatGPT、Video Detailed Description；图像侧测 AI2D、ChartQA、DocVQA、InfoVQA、RealWorldQA、MMMU。

baseline 包括 LLaVA-NeXT-Video-32K、其 training-free RoPE 外推版、LLaVA-NeXT-Qwen2、LongVA (AnyRes)，以及 InternVL-Chat-V1.5、VILA-1.5、VideoLLaVA、ST-LLM、VideoChat2、Chat-UniVi、VideoLLaMA2 等公开模型。

---

## 5. 实验结果

### 5.1 V-NIAH：2000 帧来自哪里

![](assets/LongVA%20-%20语言长上下文迁移到长视频/figures/04-v-niah-results.png)

> 图 4：LongVA 在 2000 帧以内的大部分 needle 深度保持接近满分，3000 帧仍有一定能力；AnyRes 在超过训练长度后退化更早。32K baseline 约 200 帧后崩溃，单纯把其 RoPE base 改到 100M 也只带来有限改善。虚线对应 LM 训练长度约 1555 帧。来源：论文 Figure 4，第 6 页。

这张图支持两层结论。第一，视觉有效长度与语言骨干长度强相关：32K LM 对应约 200 帧，224K LM 对应约 1555 帧；长文本训练确实迁移到了视觉。第二，LongVA 在训练长度之外还能外推：2000 帧仍很强，3000 帧部分位置仍可检索。但越靠后、越长的组合出现明显退化，因此“可处理 3000 帧”应理解为有非零/可用检索能力，不是所有深度稳定满分。

UniRes 与 AnyRes 的对比说明展平结构也重要。相同 224K LM 下，UniRes 在 1.8K–3K 帧区域更稳，支持作者关于“训练时图像 grid 与测试时视频帧结构一致”的解释。

### 5.2 Video-MME 主结果与帧数消融

![](assets/LongVA%20-%20语言长上下文迁移到长视频/figures/05-video-mme-results.png)

> 表 4：LongVA 的 overall 随 8→128 帧由 47.9 升到 52.6，之后 384 帧回落到 51.8；long 子集在 128 帧达到 46.2。LongVA 超过 7B baseline，但仍低于 34B VILA-1.5 的 59.0。来源：论文 Table 4，第 7 页。

| 模型/帧数 | Short | Medium | Long | Overall |
|---|---:|---:|---:|---:|
| LLaVA-NeXT-Qwen2，32 帧 | 58.0 | 47.0 | 43.4 | 49.5 |
| LongVA，32 帧 | 61.1 | 48.8 | 45.4 | 51.8 |
| LongVA，64 帧 | **61.4** | **50.9** | 45.0 | 52.4 |
| LongVA，128 帧 | 61.1 | 50.4 | **46.2** | **52.6** |
| LongVA，384 帧 | 60.3 | 48.9 | 46.1 | 51.8 |

长上下文使“多看帧”成为可能，但收益不是单调的。8→128 帧改善明显，128→384 帧反而下降，说明额外帧可能引入冗余/干扰，且视觉编码、prompt 长度和推理资源共同限制有效利用率。论文以算力和显存为由没有在 Video-MME 真正生成式评测 2000 帧；2000 帧数字主要来自 V-NIAH 检索测试。

### 5.3 MLVU 与短视频

附录中 LongVA 以 256 帧在 MLVU 获得多选平均 56.3、生成平均 4.33，在公开模型中领先，仅低于 GPT-4o。短视频 NeXTQA 随 8/16/32/64 帧从 50.78 提高到 68.27，但 ActivityNetQA 很快平台；VideoChatGPT/Video-DD 不随帧数稳定提升。轻量 DPO 后生成式 judge 分数显著改善，说明部分差距来自回答风格，而非纯视觉理解。

### 5.4 UniRes 图像消融

相对 AnyRes，UniRes 在 InfoVQA 从 43.3 提升到 49.4、MMMU 从 42.1 到 42.6，但 AI2D 从 73.1 降到 70.7、ChartQA 从 74.4 降到 70.4。原因是 UniRes 每 grid 做 2×2 pooling，却允许最多 49 grids；高分辨率 InfoVQA 得益于更多 grid，低分辨率图表则主要承受池化损失。这证明 UniRes 不是无条件更好的图像编码器，而是为长视觉结构做的 trade-off。

---

## 6. 与相关工作的关系

LongVA 与视觉 token 压缩正交：LLaMA-VID、Q-Former、token pruning 解决单位帧成本，LongVA 解决 LM 可接受的总长度；二者组合理论上可把视频扩得更长。相对专门的视频时空 encoder，LongVA 几乎没有运动归纳偏置，优势是直接继承强图像模型与长文本系统，弱点是动作和时间关系只能由 LLM 在一维 token 中间接学习。

它也与一般长上下文 LLM 工作直接相连：RoPE scaling、长文数据工程、Ring Attention、sequence parallel 并非视频专属，但 LongVA 说明这些语言侧进步会外溢到多模态模型。后续多模态长上下文系统因此可以把“视觉压缩、LM 长度、系统并行”作为三个可独立扩展的轴。

---

## 7. 局限与批判性评价

1. **V-NIAH 只测单针检索。** 找一帧并读一个短答案，不等于理解跨镜头因果、人物状态演化或多事件组合。
2. **2000/3000 帧并非实用生成吞吐。** Perplexity-based 单次 forward 避开 KV cache；真实自回归服务仍会面临约 100GB 级 cache、极低 batch 与高延迟。
3. **没有视频训练是优点也是上限。** 零样本泛化很干净，但模型没有学习光流、动作边界、镜头切换与时间定位监督。
4. **144 token/帧仍很昂贵。** 2000 帧是 288K 视觉 token；CLIP 还需逐帧运行。若问题只涉及少数片段，大量计算浪费在无关帧。
5. **密采样收益会饱和或下降。** Video-MME 128 帧优于 384 帧，说明“窗口更长”不自动等于“利用得更好”。
6. **位置表示把视频当文本。** 只有一维 RoPE，没有时间戳和二维/三维位置；不等帧率、缺帧和长静态段都可能改变 token 距离语义。
7. **高分辨率图像存在 trade-off。** UniRes 对 InfoVQA 有利，却损害 AI2D/ChartQA，统一编码不是免费午餐。
8. **多模态数据细节披露有限。** 论文主要写“沿用 LLaVA-1.6 recipe”，精确数据版本、清洗和 prompt 模板需依赖官方代码复核。

---

## 8. 复现与实践建议

### 8.1 最小复现顺序

1. 从 Qwen2-7B-Instruct 出发，先复现 224K 纯文本 continued pretraining；用文本 NIAH 验证长度，而不是先接视觉。
2. 检查 RoPE base、position ids、packing/BOS 分隔、最大序列长度和 checkpoint config 是否一致；只改 `max_position_embeddings` 不构成长上下文训练。
3. 接入 CLIP-ViT-L-336px 与两层 MLP，复现 576→144 的 2×2 pooling，确保按 grid 内 raster order 展平。
4. 仅使用图像数据完成 LLaVA-1.6 两阶段对齐，严禁误混入视频，否则无法验证 long context transfer。
5. 依次跑 LLaVA-NeXT-Qwen2、LongVA (AnyRes)、LongVA (UniRes) 三个控制组；V-NIAH 应先复现 200/1000/1555/2000 帧几个关键长度。
6. 最后再测 Video-MME 的 8/16/32/64/128/384 帧曲线，避免只报最佳帧数。

### 8.2 QKV、mask 与 shape 检查

- CLIP 输出应为每 grid 24×24=576 patch；pool 后必须是 12×12=144，不要跨相邻帧做 2×2 pooling。
- 2-layer MLP 输出维度必须等于 Qwen2 hidden size，视觉 token 直接替换图像占位符位置。
- Qwen2 使用原生 self-attention/GQA 的 QKV 权重，不新增视频 QKV；加载长上下文 checkpoint 时不能漏掉 RoPE 配置。
- Causal mask、padding mask 与 packed document boundary 要分别检查；多文档文本训练以 BOS 分隔，但仍是同一 causal 序列。
- 视频 position id 应连续递增，frame-major 顺序每 144 个 token 换帧；若预处理跨帧重排，UniRes 的关键假设会被破坏。

### 8.3 资源与部署评估

训练复现需要 Ring Attention/sequence parallel、8×80GB GPU 与稳定高速互联；单卡很难忠实复现 224K dense attention。推理时可预计算并落盘视觉 embedding，但问答生成仍需要把长视觉前缀送入 LLM。工程上更实用的方案是先用检索器筛选时间段，或结合 token pruning/低 token connector，把 LongVA 的“长 LM”作为容量上限而非每次都填满。

---

## 9. Future Work 与个人启发

### 9.1 后续研究方向

1. **长上下文 + 自适应视觉压缩。** 不必在 2 token/帧与 144 token/帧之间固定选择，可根据镜头变化和问题相关性动态分配。
2. **层级时间结构。** 加入 frame/clip/chapter 三级位置或时间戳，让 token 距离对应真实时间，而不是只对应采样后序号。
3. **从单针到多跳视觉 NIAH。** 要求从两个远距离帧组合答案，才能测试跨时段推理而非单点检索。
4. **Streaming/KV 压缩。** 对已读帧做可验证的 KV 合并或分层缓存，使 2000 帧能力能用于真实交互生成。
5. **长视频监督的增量价值。** 在相同长 LM 下逐步加入短视频、长视频、时间定位数据，量化“语言迁移”与“视频专门训练”的边际贡献。

### 9.2 个人判断

LongVA 最有影响力的观点是：多模态模型的上下文能力未必需要在每种模态上分别从头学习；只要 connector 真正完成表示对齐，语言骨干的算法和系统进步可以跨模态复用。但论文也提醒，**可寻址长度、可生成长度、可推理长度是三件不同的事**：V-NIAH 证明前者，Video-MME 部分证明后两者，却还没有把三者统一到 2000 帧规模。

---

## 10. 材料来源

- 官方论文：[arXiv:2406.16852](https://arxiv.org/abs/2406.16852)
- 官方代码：[EvolvingLMMs-Lab/LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)
- 本地 PDF：[assets/LongVA - 语言长上下文迁移到长视频/sources/paper.pdf](assets/LongVA%20-%20语言长上下文迁移到长视频/sources/paper.pdf)
- 完整来源与图片溯源：[sources.md](assets/LongVA%20-%20语言长上下文迁移到长视频/sources.md)
