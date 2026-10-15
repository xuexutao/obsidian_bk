---
type: paper-note
status: done
domain: VLM
paper: "DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding"
year: 2024
arxiv: "2412.10302"
doi: "10.48550/arXiv.2412.10302"
source: "https://arxiv.org/abs/2412.10302"
project: "https://github.com/deepseek-ai/DeepSeek-VL2"
code: "https://github.com/deepseek-ai/DeepSeek-VL2"
tags:
  - VLM
  - MoE
  - Dynamic-Resolution
  - MLA
  - OCR
created: 2026-10-15
updated: 2026-10-15
---

# DeepSeek-VL2：MoE 动态高分辨率视觉语言模型

## 基本信息

- **作者：** Zhiyu Wu、Xiaokang Chen、Zizheng Pan 等
- **版本：** arXiv v1，2024-12-13
- **模型系列：** Tiny / Small / DeepSeek-VL2，激活参数约 1.0B / 2.8B / 4.5B
- **本地原文：** [paper.pdf](assets/DeepSeek-VL2%20-%20MoE动态高分辨率视觉语言模型/sources/paper.pdf)
- **完整溯源：** [sources.md](assets/DeepSeek-VL2%20-%20MoE动态高分辨率视觉语言模型/sources.md)

**一句话结论：**

DeepSeek-VL2 用一个清晰的效率组合升级 DeepSeek-VL：视觉侧以 **动态切片 + 全局缩略图 + 结构 token** 适配高分辨率和极端宽高比，语言侧以 **DeepSeekMoE** 降低每 token 激活参数，并在中/大模型中用 **MLA** 压缩 KV cache；约 800B token 的全参数视觉语言预训练带来强 OCR、文档、图表和 grounding，但论文几乎没有正式模块消融与端到端推理效率数据，因此“高效”的证据主要来自稀疏激活、显存可部署性和参数—性能曲线。

---

## 1. 研究背景与问题定义

### 1.1 DeepSeek-VL 的两个结构瓶颈

前代 DeepSeek-VL 同时使用 384×384 的 SigLIP 与 1024×1024 的 SAM-B，能够兼顾语义和细节，却受固定分辨率约束。信息图、长文档、密集 OCR 和极端宽高比图像需要更灵活的视觉预算。与此同时，dense LLM 的全部 FFN 参数和完整 KV cache 会抬高推理成本。

VL2 将问题拆成两部分：让视觉 token 数随图像布局动态变化，让语言计算只激活少量专家并压缩 KV cache。数据侧则扩展 OCR、文档、图表、grounding、grounded conversation 与视觉代码任务。

![](assets/DeepSeek-VL2%20-%20MoE动态高分辨率视觉语言模型/figures/01-teaser.png)

> **图 1：平均性能与激活参数量。** DeepSeek-VL2 家族在约 1.0B、2.8B、4.5B 激活参数处形成较高的性能—计算前沿；纵轴平均了 MMBench v1.1、MMStar、MMMU、MathVista、AI2D 与缩放后的 OCRBench。来源：论文 Figure 1。

### 1.2 核心贡献

1. 用单个 SigLIP-SO400M-384 和最多 9 个局部 tile 处理不同宽高比与高分辨率图像。
2. 用 2×2 pixel shuffle、`<tile_newline>` 和 `<view_separator>` 把二维 tile 布局显式编码为一维视觉序列。
3. 用 DeepSeekMoE 的共享专家+路由专家降低激活参数；Small/Large 加入 rank 512 的 MLA。
4. 建立三阶段训练：视觉语言对齐、约 800B token 联合预训练、约 20B token SFT。
5. 强化视觉 grounding、grounded conversation 和 in-context grounding，三种规模均给出系统评测与训练配置。

---

## 2. 输入、输出与总体架构

### 2.1 输入

模型接受单图/多图、不同分辨率和宽高比图像、纯文本、视觉提示，以及 grounding 特殊 token。训练序列长度固定为 4,096。输入图片超过两张时关闭动态切片，以控制计算和上下文；关闭后的具体分辨率处理方式没有披露。

### 2.2 输出

输出是文本 token，可表示普通问答、密集描述、OCR、表格/图表/文档答案、HTML 或 Python 绘图代码、视觉推理、故事与定位框。grounding 坐标归一化到 `[0,999]`：

```text
<|ref|><query><|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>
```

`<|grounding|>` 可触发 grounded conversation，使自然语言实体与相应位置同时输出。

### 2.3 三组件结构

![](assets/DeepSeek-VL2%20-%20MoE动态高分辨率视觉语言模型/figures/02-architecture.png)

> **图 2：DeepSeek-VL2 总体架构。** 动态切片后的图像由共享 Vision Encoder 编码，经 VL Adaptor 变成视觉 token，与文本 token 一同输入 DeepSeekMoE decoder。来源：论文 Figure 2。

架构延续 LLaVA-style decoder-only：SigLIP-SO400M-384 视觉 encoder、包含 pixel shuffle 与两层 MLP 的 VL adaptor、DeepSeekMoE LLM。视觉 encoder 约 0.4B 参数。论文未披露 adaptor 的 hidden size、激活函数和初始化。

---

## 3. 动态高分辨率切片

### 3.1 候选分辨率与最小 padding

基础 tile 是 384×384，普通任务的候选集合为：

$$C_R=\{(m\cdot384,n\cdot384)\mid m,n\in\mathbb N,\ 1\le m,n,\ mn\le9\}$$

对尺寸 $(H,W)$ 的输入，先保持原宽高比缩放，使长边匹配候选分辨率，再对另一维 padding；选择 padding 面积最小的 $(m_i\cdot384,n_i\cdot384)$。随后生成 $m_i\times n_i$ 个局部 tile，并额外加入一张全局缩略图，总 tile 数为 $1+m_in_i$。

InfoVQA 含极端宽高比和超大图，评测时把限制扩为 $mn\le18$。论文没有给出 padding 面积闭式公式或平局处理规则。

### 3.2 每个 tile 的编码与压缩

SigLIP 对每个 tile 输出 $27\times27=729$ 个 1152 维 embedding。2×2 pixel shuffle 把网格压缩为 $14\times14=196$ token。由于 27 是奇数，shuffle 前如何 padding 或对齐没有披露。

全局缩略图的每一行后插入一个 `<tile_newline>`，因此 token 数为：

$$14\times(14+1)=210$$

局部 tile 先还原成 $(m_i\cdot14,n_i\cdot14)$ 的大网格，每行末尾加入 newline，并在全局与局部视图之间插入 `<view_separator>`。最终视觉序列长度为：

$$N_{\mathrm{vision}}=210+1+m_i\cdot14\times(n_i\cdot14+1)$$

### 3.3 为什么需要结构 token

![](assets/DeepSeek-VL2%20-%20MoE动态高分辨率视觉语言模型/figures/03-dynamic-tiling.png)

> **图 3：动态切片与一维序列布局。** 全局缩略图保留整体语义，局部 tile 提供细节；newline token 保留二维行边界，view separator 区分全局/局部视图。来源：论文 Figure 3。

如果只把所有 tile 特征连续拼接，decoder 很难知道某个 token 属于哪一行、哪一视图。newline 和 separator 相当于最低成本的布局语法。它不像额外 2D attention 那样增加网络模块，却把二维结构编码进自回归序列。

### 3.4 方法边界

动态切片会让 token 数依图像布局变化，训练和推理负载更不均。论文通过跨 data-parallel rank 平衡 tile 数、细粒度切分视觉 encoder 层、为纯文本和图文 batch 切换两套 pipeline 来缓解。超过两张图关闭动态切片意味着高分辨率多图场景仍受限。

---

## 4. DeepSeekMoE 与 MLA

### 4.1 MoE 配置

| 配置 | Tiny | Small | DeepSeek-VL2 |
|---|---:|---:|---:|
| LLM 总参数 | 3B | 16B | 27B |
| LLM 激活参数 | 0.57B | 2.4B | 4.1B |
| 加视觉塔后的激活参数 | 约 1.0B | 约 2.8B | 约 4.5B |
| 层数 | 12 | 27 | 30 |
| Attention | MHA | MLA, rank 512 | MLA, rank 512 |
| 路由专家 | 64 | 64 | 72 |
| 共享专家 | 2 | 2 | 2 |
| 每 token Top-K | 6 | 6 | 6 |
| 路由函数 | Softmax | Softmax | Sigmoid |
| correction bias | 无 | 无 | 有 |

每个 token 固定经过共享专家，再选择得分最高的 6 个路由专家。论文没有给出 gating 与专家输出融合公式、单专家 FFN 宽度、capacity factor、token drop 或通信成本。Large 使用 global expert bias 改善负载均衡，但更新公式没有展开。

### 4.2 MLA 的作用与披露边界

Small 和 Large 使用 latent rank 512 的 Multi-head Latent Attention。核心作用是把完整多头 Key/Value cache 压缩为 latent vector，从而降低长序列解码缓存并提高吞吐。VL2 报告没有重述下投影/上投影公式、RoPE 组合方式、每层 cache 字节数或实际压缩比；完整机制需参考 DeepSeek-V2。因此不能从本论文单独量化 MLA 的速度收益。

### 4.3 单卡部署口径

论文称 Tiny/Small/Large 可分别部署在约 10GB/40GB/80GB 单 GPU 上，但没有披露精度、量化、batch、上下文长度和 KV cache 条件。这个数字可以说明模型权重级可部署性，不能直接等价为真实服务显存。

---

## 5. 数据构建

### 5.1 Stage 1 对齐数据

使用约 1.2M ShareGPT4V caption 和 conversation，将视觉表示对齐到预训练 LLM embedding。具体两类比例和语言分布未披露。

### 5.2 Stage 2 联合预训练数据

总混合约 70% 视觉语言、30% 纯文本。视觉部分包括 WIT、WikiHow、OBELICS 的 30% 子集、Wanjuan 中文内容、内部知识数据、DenseFusion-1M、PixelProse、MMSci、Multimodal ArXiv、YFCC100M、LAION-Aesthetics、MegaLith-10M、JourneyDB、Pick-a-Pic、Wukong 等。caption 会结合 OCR hints、元数据和原 caption 重写，再由 DeepSeek Chat 评分过滤。

OCR 包括 LaTeX OCR、12M RenderedText 和中英文内部 OCR；VQA 包括通用 VQA、PubTabNet、FinTabNet、Docmatix、Websight 和 plot-to-Python；grounding 使用 Kosmos-2、Objects365 等并加入“目标不存在”的负样本；grounded conversation 把实体文本与框一起监督。各子类样本量、权重和内部数据规模未披露。

### 5.3 Stage 3 SFT 数据

SFT 清洗并重写通用 VQA、文档 QA、表格/图表、推理、学术问题、web-to-code、grounding 和 grounded conversation。纯文本部分包括 Magpie、GSM8K、Orca-Math、Magicoder、OpenMathInstruct-2、COIG-CQIA、Dolly 等，以保留语言能力。

---

## 6. 三阶段训练与超参数

### 6.1 训练阶段

1. **Vision-Language Alignment：** 解冻视觉 encoder 与 adaptor，冻结 LLM；约 2.0B token。与常见只训 connector 不同，此阶段让 SigLIP 本身适应动态高分辨率。
2. **Vision-Language Pre-training：** 全参数解冻，Tiny/Small/Large 分别训练 798.5B/808.9B/796.5B token，序列长度 4096。
3. **Supervised Fine-Tuning：** 全参数训练，只监督 assistant answer 与特殊 token，mask system/user；约 19.5B/20.0B/19.5B token。

所有阶段主要使用文本 token 的 next-token prediction loss，视觉 token 不直接计算重建或对比损失。

### 6.2 关键超参数

| 项目 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| LR scheduler | cosine | step | constant |
| LLM | 冻结 | 解冻 | 解冻 |
| 视觉 LR multiplier | 0.1 | 0.1 | 0.1 |
| optimizer | AdamW，$\beta_1=0.9,\beta_2=0.95$ | 同左 | 同左 |
| weight decay | 0.1 | 0.1 | 0.1 |
| gradient clip | 1.0 | 1.0 | 1.0 |
| sequence length | 4096 | 4096 | 4096 |

Large 的学习率为 $4.5\times10^{-4}$、$4.5\times10^{-4}$、$2.0\times10^{-5}$；Stage 2 在 50% 和 75% 总步数处把学习率除以 $\sqrt{10}$。Large 的 batch 为 256/3360/64。

### 6.3 训练资源

| 模型 | 节点 | 推算 GPU 数 | 时间 |
|---|---:|---:|---:|
| Tiny | 16 | 128×A100 | 7 天 |
| Small | 33 | 264×A100 | 10 天 |
| Large | 42 | 336×A100 | 14 天 |

每节点 8 张 NVIDIA A100。论文没有说明 A100 40GB/80GB、利用率、总 FLOPs、能耗和费用。

---

## 7. 定量结果与 baseline

### 7.1 OCR、文档与图表

| 模型 | 激活参数 | DocVQA | ChartQA | InfoVQA | TextVQA | OCRBench |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | 1.0B | 88.9 | 81.0 | 66.1 | 80.7 | 809 |
| Small | 2.8B | 92.3 | 84.5 | 75.8 | 83.4 | **834** |
| Large | 4.5B | **93.3** | **86.0** | **78.1** | **84.2** | 811 |

Small 的 OCRBench 高于 Large，说明规模并不保证所有 OCR 指标单调提升。对照中 Qwen2-VL-7B 的 DocVQA 94.5、OCRBench 845，MiniCPM-V2.6 的 OCRBench 852，Aria-MoE 的 ChartQA 86.4；因此 VL2 的优势是“以更少激活参数达到竞争水平”，不是每项 SOTA。

### 7.2 通用理解与数学

| 模型 | MMStar | AI2D | MMMU | MME | MMBench-EN | RealWorldQA | MathVista |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 45.9 | 71.6 | 40.7 | 1915 | 73.3 | 64.2 | 53.6 |
| Small | 57.0 | 80.0 | 48.0 | 2123 | 82.3 | 65.4 | 60.7 |
| Large | **61.3** | **81.4** | **51.1** | **2253** | **83.1** | **68.4** | **62.8** |

Large 对 Qwen2-VL-7B：MMStar 61.3 vs 60.7、MathVista 62.8 vs 58.2，但 AI2D 81.4 vs 83.0、MMMU 51.1 vs 54.1、MME 2253 vs 2327。结果再次支持参数效率，而非全面领先。

### 7.3 Visual grounding

Large 在 RefCOCO、RefCOCO+、RefCOCOg 的全部列上超过 Florence-2-L，例如 RefCOCO val/testA/testB 为 95.1/96.7/92.7，RefCOCO+ 为 91.2/94.9/87.4。grounding 是论文中最一致的强项，也是新数据和结构 token 最有说服力的结果之一。

---

## 8. 消融、失败案例与证据缺口

### 8.1 正式消融缺失

论文没有提供动态切片 vs 固定分辨率、9 tile vs 其他数量、去全局缩略图、去 newline/separator、MoE vs dense、MLA vs MHA、冻结 vs 解冻视觉塔、70:30 数据比例、三阶段贡献或 expert bias 的正式消融。这是报告最大的证据缺口。

### 8.2 只有经验性观察

作者提到 30% OBELICS 比例来自 Tiny 的初步实验；冗长推理对 Tiny 不一定有利；增加内部中文 QA 可减少中英混杂；文档多轮 QA 早期有帮助；grounding 出现向 meme、动漫、名人和抽象概念的泛化。这些观察均缺少完整定量对照。

### 8.3 明确局限

论文列出的边界包括：4096 上下文较短；一次只能处理少量图像；超过两图关闭动态切片；模糊图像和未见对象鲁棒性不足；感知强于推理；内部 OCR 主要覆盖中英文；故事多样性与安全约束存在冲突。内部“未观察到显著有害输出”没有公开测试集和数字，不能视为系统安全结论。

---

## 9. 与前作及相关路线的关系

相对 DeepSeek-VL，VL2 去掉 SigLIP+SAM-B 的固定双 encoder，改为单 SigLIP 动态切片；语言侧从 dense/旧基座升级为 DeepSeekMoE，并把 DeepSeek-V2 的 MLA 引入中/大版本；数据侧增加 grounding、grounded conversation、GUI 与更强 OCR/文档数据。

相对 InternVL/Qwen2-VL 等动态分辨率模型，VL2 的特色不是“首次切片”，而是把切片、结构 token、稀疏 MoE 和 MLA 组合成低激活参数路线。视觉 tile 数仍可能很大，真正的解码效率还取决于视觉 token 序列、MLA cache 和 expert serving；论文没有给出统一吞吐对比。

---

## 10. 复现成本与实践建议

### 10.1 成本

官方训练 Large 需要 336 张 A100 约 14 天，且约 800B token 全参数预训练。即使不算数据获取和清洗，独立从零复现成本也很高。Tiny 仍需 128 张 A100 约 7 天。

### 10.2 可行复现路径

1. 先用官方 Tiny/Small checkpoint 验证动态切片布局和 grounding 格式，而不是从零训练。
2. 动态切片实现必须核对官方代码中的 27→14 pixel shuffle 对齐、候选分辨率 tie-breaking 和多图关闭逻辑。
3. 做缺失消融：固定 384、全局缩略图、1/4/9/18 tile、去 newline/separator，并按 OCR/文档/自然图分别报告。
4. 推理评测应同时报告权重显存、KV cache、视觉 token 数、首 token 延迟、decode tok/s 和 expert 并行通信。
5. MLA 复现应引用 DeepSeek-V2 公式和官方实现，不应从本报告猜测。

---

## 11. 个人判断

DeepSeek-VL2 是一篇工程边界很清楚的模型报告。动态切片公式、token 布局和训练超参数披露得相对完整，三种规模也给出了真实训练资源；这使得“如何做”比很多同类报告更可操作。结构 token 尤其值得借鉴：用非常便宜的 newline/separator 把二维布局注入一维 decoder。

但论文把最重要的因果问题留空了：强结果究竟来自动态高分辨率、MoE/MLA、800B token，还是数据重写与 grounding 监督，无法通过消融分离。性能—激活参数曲线也不能等价为真实系统效率，因为总参数加载、专家通信和动态视觉 token 都影响服务成本。

我的结论是：VL2 很适合作为低激活参数、高分辨率 VLM 的实现参考，尤其适合 OCR、文档和 grounding；若目标是证明 MLA/MoE 或动态切片各自的独立贡献，这篇论文证据不足，需要依靠官方代码和补充实验。

---

## 12. 材料来源

官方 PDF、arXiv source、代码仓库与三张原文关键图的完整溯源见 [sources.md](assets/DeepSeek-VL2%20-%20MoE动态高分辨率视觉语言模型/sources.md)。本文对 MLA、MoE routing、数据比例和未给出的消融均明确保留“论文未披露”，未自行补全。