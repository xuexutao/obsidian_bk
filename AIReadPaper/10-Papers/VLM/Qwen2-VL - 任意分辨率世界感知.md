---
type: paper-note
status: complete
domain: VLM
paper: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
year: 2024
arxiv: "2409.12191"
doi: null
source: "https://arxiv.org/abs/2409.12191"
project: null
code: "https://github.com/QwenLM/Qwen2-VL"
tags:
  - VLM
  - Qwen-VL
  - native-resolution
  - video-understanding
  - multimodal-agent
  - OCR
created: 2026-10-15
updated: 2026-10-15
---

# Qwen2-VL：任意分辨率世界感知

## 基本信息

- **作者**：Peng Wang、Shuai Bai、Sinan Tan、Shijie Wang、Zhihao Fan、Jinze Bai 等，Qwen Team / Alibaba Group
- **论文版本**：arXiv:2409.12191v2，2024-10-03
- **模型家族**：Qwen2-VL-2B、Qwen2-VL-7B、Qwen2-VL-72B；三者共享约 675M 参数视觉编码器，语言模型分别约 1.5B、7.6B、72B
- **本地原文**：[paper.pdf](assets/Qwen2-VL%20-%20任意分辨率世界感知/sources/paper.pdf)
- **来源清单**：[sources.md](assets/Qwen2-VL%20-%20任意分辨率世界感知/sources.md)

**一句话结论：**

Qwen2-VL 相对初代 Qwen-VL 的根本变化，是把“固定 448×448 图像经 256 个 query 压成固定视觉序列”改为**原宽高比、变长视觉 token 的统一图像—视频序列**：ViT 用 2D-RoPE 接受动态分辨率，LLM 用 M-RoPE 分解时间/高度/宽度位置，视频再以深度为 2 的 3D 卷积形成时空 tube；这一组合显著改善文档、OCR、长视频和视觉 Agent，但计算量随像素与上下文增长，训练数据、算力规模和部分内部评测又未完全公开。

![](assets/Qwen2-VL%20-%20任意分辨率世界感知/figures/01-any-resolution.jpg)

> 图 2：Qwen2-VL 将原始宽高比不同的图片和视频编码成不同长度的视觉 token，再与文本统一送入 QwenLM Decoder。示例中不同媒体分别占用 11427、8、1125、2208 个 token，直观说明“任意分辨率”的实际含义是动态 token 预算，而非无成本地保留所有像素。来源：论文 Figure 2，第 4 页；官方 arXiv 源码原图。

---

## 1. 研究背景与前代瓶颈

### 1.1 初代 Qwen-VL 做到了什么

初代 Qwen-VL 已经建立“ViT → 跨模态适配器 → Qwen-7B”的生成式接口，支持中英文 VQA、OCR、文档读取和文本化 bounding box。它用固定 256 个 learnable queries 压缩视觉特征，并在第二阶段把图像分辨率提高到 448×448。这个设计的优点是视觉长度固定、LLM 成本可控；缺点也来自同一约束：超宽网页、长文档、小字密集图都必须 resize 到固定画布，任意复杂图像最终都挤进 256 个视觉 token，视频更没有原生的时空位置表示。

### 1.2 Qwen2-VL 要解决的三个问题

第一，固定方形分辨率既破坏宽高比，也会在高分辨率图像上丢失细节；简单切 tile 又引入局部—全局拼接规则。第二，传统 LLM 的 1D-RoPE 只能表示序列先后，不能显式区分视频的时间轴与图像的二维空间。第三，若每帧都按独立 2D 图像编码，长视频视觉序列会迅速膨胀，无法兼顾帧数、分辨率和上下文长度。

### 1.3 本文相对 Qwen-VL 的可核验增量

| 维度 | Qwen-VL | Qwen2-VL | 直接效果 |
|---|---|---|---|
| 图像尺寸 | 固定 224/448 方形输入 | Naive Dynamic Resolution，按原宽高比生成变长 token | 高分辨率文档、图表、小字与极端宽高比更少失真 |
| ViT 位置 | 二维绝对位置参与 adapter | 移除原绝对位置 embedding，改用 2D-RoPE | ViT 可处理训练时未固定的二维网格 |
| 视觉压缩 | 256 个 learnable queries | 相邻 $2\times2$ token 经 MLP 合并为 1 个 | token 数约降为四分之一，同时保留变长特性 |
| LLM 位置 | 常规 1D 序列位置 | M-RoPE：时间、高度、宽度三组位置 | 文本、图像、视频共享 decoder 且保留模态结构 |
| 视频编码 | 非原生重点 | 2 fps 采样 + 深度为 2 的 3D convolution | 两帧形成时空 tube，在不增序列长度时覆盖更多帧 |
| 底座与规模 | Qwen-7B | Qwen2 + 2B/7B/72B 三档；共享 675M ViT | 检验模型与数据扩展规律，覆盖端侧到高性能档 |

这里需区分 **2D-RoPE** 与 **M-RoPE**：前者位于 ViT 内部，解决变尺寸 patch 网格的位置编码；后者位于 LLM 侧，把旋转位置维度拆给 temporal/height/width，解决多模态序列如何进入同一个 decoder。二者不是同一个模块的两个名字。

---

## 2. 任务定义、输入输出与统一接口

### 2.1 输入、输出与假设

输入可以是纯文本、单图、多图、视频及其交错对话。每个视觉段由 `<|vision_start|>` 与 `<|vision_end|>` 包围，视觉特征与文本 embedding 拼成一条序列。输出仍是自回归文本，可表现为自然语言回答、OCR/公式/表格转写、边界框坐标、函数调用参数，或视觉 Agent 的“观察—推理—动作—新观察”轨迹。

模型不额外增加检测头。grounding 的框坐标归一化到 $[0,1000)$，以 `(x_top_left,y_top_left),(x_bottom_right,y_bottom_right)` 文本输出，并用 `<|object_ref_start|>`、`<|object_ref_end|>`、`<|box_start|>`、`<|box_end|>` 建立指称与框的对应关系。统一目标仍是只监督文本 token 的 next-token prediction：

$$\mathcal{L}_{\mathrm{NTP}}=-\sum_{t\in\mathcal{T}_{\mathrm{text}}}\log p_\theta(y_t\mid y_{<t},V)$$

其中 $V$ 是图像或视频产生的变长视觉 token；视觉 token 作为条件进入上下文，但论文明确说明预训练时 loss 只施加在文本 token 上。

### 2.2 动态预算而非真正“无限分辨率”

“Any Resolution”表示架构不要求固定高宽，而非无限像素。推理时不同图像被打包成一个变长序列，并设置长度上限以控制显存。训练消融中，动态分辨率使用 `min_pixels=100×28×28` 与 `max_pixels=16384×28×28`；视频则动态调整每帧分辨率，使单视频总视觉 token 不超过 16384。因而模型的输入边界由像素上下限、视觉 token 合并和 LLM 上下文共同决定。

---

## 3. 核心方法

### 3.1 Naive Dynamic Resolution 与 2D-RoPE

视觉编码器是约 675M 参数的 ViT，初始化自 DFN ViT。为了让二维 patch 网格可变，模型移除 DFN 中固定尺寸相关的绝对位置 embedding，并在视觉 self-attention 中使用沿高度和宽度的 2D-RoPE。设 patch 位于 $(h,w)$，其 query/key 的不同通道子空间分别应用 $R(h)$ 与 $R(w)$，抽象写成 $q'=[R(h)q_h;R(w)q_w]$、$k'=[R(h)k_h;R(w)k_w]$。旋转位置编码以相对相位影响注意力，因此不需要为每个新尺寸插值一张固定位置表。

ViT 输出后不再用初代 256-query adapter，而是用简单 MLP 将相邻 $2\times2$ patch token 合并为一个 token。以 224×224、patch size 14 为例，ViT 先产生 $16\times16=256$ 个 patch token，合并后为 $8\times8=64$ 个，再加起止标记共 66 个 token。视觉长度因此近似满足 $N_v\approx(H/28)(W/28)$，与图像面积成正比。

这一路线的价值在于保持宽高比和信息密度；代价是高像素图会直接增加 LLM 序列长度。与 tile-based 方法相比，它减少手工切图与重复边界，但没有消除二次注意力的长度成本。

### 3.2 M-RoPE：在 decoder 中统一文本、图像与视频

![](assets/Qwen2-VL%20-%20任意分辨率世界感知/figures/02-mrope.png)

> 图 3：M-RoPE 将旋转位置编码拆成 temporal、height、width 三部分。文本在三轴上使用相同 position ID，图像固定 temporal 并展开二维坐标，视频逐帧增加 temporal。它既保留各模态结构，也避免把二维/三维坐标强行压成很大的单轴编号。来源：论文 Figure 3，第 5 页；官方 arXiv 源码 `images/mrope.png`。

对文本 token，三组 position ID 相同，因此 M-RoPE 退化为普通 1D-RoPE。对图像，同一图像所有 token 的 temporal ID 相同，height/width ID 由二维网格位置决定。对视频，height/width 规则与图像一致，而 temporal ID 随帧递增。多模态片段串联时，新片段从上一片段最大 position ID 加一开始编号。

可将第 $i$ 个 token 的旋转写成 $R_i=R_t(t_i)\oplus R_h(h_i)\oplus R_w(w_i)$；注意这表示**通道维拆分**，不是将三个位置简单相加。由于一个 $H\times W$ 图像不再被展平成最大位置约 $HW$ 的单轴编号，position ID 的最大值更小，这也是论文声称其更容易做长度外推的原因。

### 3.3 统一图像—视频建模与深度为 2 的 3D 卷积

视频以 2 fps 采样。视觉 patch embedding 使用时间深度为 2 的 3D convolution，将相邻两帧对应空间区域编码成一个 3D tube；这样相同序列长度可覆盖约两倍帧数。为统一接口，静态图像复制成两帧再经过同一视觉入口。这里 3D conv 的作用是**早期时间压缩与局部时序融合**，真正跨长时间范围的关系仍由 LLM 中的 M-RoPE 和 self-attention 建模。

论文没有给出单独移除 3D convolution 的消融，因此不能把视频增益完全归因于该模块。视频能力同时受混合图像/视频训练、2 fps 采样、动态帧分辨率、M-RoPE 和更大训练数据影响。

### 3.4 视觉 Agent 的序列决策接口

Qwen2-VL 把 UI 操作、机器人控制、游戏和导航统一为多步决策：模型接收截图与指令，输出带函数名和参数的动作；环境执行动作后把新观察追加到上下文，再循环到成功或终止。AITZ 中动作包括 tap、input、swipe；tap 坐标同样归一化。模型支持 32K 上下文，用于保存多轮 `(observation, action)` 历史。

这是一种“VLM 作为 policy”的文本接口，不等于模型内部拥有显式世界模型。论文在导航实验中也指出，多图生成的地图信息不完整且非结构化，导致 R2R、REVERIE 明显落后专用 VLN 模型。

---

## 4. 数据、训练与系统实现

### 4.1 三阶段训练

Qwen2-VL 延续初代三阶段课程，但规模与数据类型显著增加。

| 阶段 | 参数状态 | 数据与 token | 学习重点 |
|---|---|---|---|
| Stage 1 | 主要训练 ViT | 约 600B token；图文、OCR、分类 | 建立视觉语义与文字识别，DFN ViT 的固定位置改成 2D-RoPE |
| Stage 2 | 全参数解冻 | 额外约 800B image-related token；混合图文、VQA、多任务与纯文本 | 深化多模态理解并保持语言能力 |
| Stage 3 | 冻结 ViT、微调 LLM | ChatML 指令数据；图像 QA、文档解析、多图比较、视频、流式对话、Agent | 指令遵循与交互输出 |

两段预训练累计约 **1.4T token**，统计同时包含文本 token 与图像 token；知识截止时间为 2023 年 6 月。原始数据来自清洗网页、开源数据和合成数据，但论文没有公开逐数据集规模、去重规则、合成比例、许可证清单或 Stage 3 总样本量。这使训练配方只能部分复现。

### 4.2 训练基础设施

训练使用 Alibaba Cloud PAI-Lingjun、CPFS 与 OSS。长视频解码被识别为主要瓶颈，团队采用缓存解码。并行策略包括 DP、TP、PP、SP、ZeRO-1 和 selective activation checkpointing；72B 版本使用 1F1B pipeline parallelism。软件栈为 PyTorch 2.1.2、CUDA 11.8、FlashAttention 和 fused LayerNorm/RMSNorm/Adam。

论文还记录了一个复现细节：TP 下 convolution 的非确定性会导致 shared weights 不一致，团队通过离线 reduction 处理，而不是额外 all-reduce。这说明视觉卷积与大模型并行组合可能产生数值一致性问题。

### 4.3 复现成本

- **可复现部分**：2B/7B/72B 权重与官方代码公开，可直接验证推理、动态像素预算、M-RoPE 和下游评测。
- **难复现部分**：1.4T 多模态 token、网页/合成数据治理、内部多语 OCR 和 function-calling 数据、72B 训练集群均未完整公开。
- **推理成本**：动态分辨率使成本与原图面积强相关；长视频还要承担帧采样、解码、视觉编码和长上下文 KV cache。2B 可面向受限资源，72B 则需要多卡或量化部署。
- **工程风险**：生产环境必须显式设置 `min_pixels`/`max_pixels`，否则超高分辨率输入可能造成不可预测的延迟与显存占用。

---

## 5. 实验、Baseline 与定量结果

### 5.1 综合视觉能力

论文把 Qwen2-VL 与 GPT-4o、Claude 3.5 Sonnet、InternVL2 等通用模型，以及各 benchmark 的 previous SoTA 对比。72B 在文档、OCR、真实世界感知和综合 VQA 上尤其突出：DocVQA 96.5，高于 GPT-4o 的 92.8 与表中 previous SoTA 94.1；InfoVQA 84.5；OCRBench 877；RealWorldQA 77.8，高于 GPT-4o 75.4；MMVet 74.0，高于 GPT-4o 69.1；MMT-Bench 71.7，高于 GPT-4o 65.5。

但“全面超过闭源模型”并不成立。MMMU 上 Qwen2-VL-72B 为 64.5，低于 GPT-4o 69.1 与 Claude 3.5 Sonnet 68.3；MathVision 为 25.9，也低于 GPT-4o 30.4。更准确的结论是：它在高分辨率感知、文档/OCR 和多项综合 VQA 上达到或超过同期强基线，而高难学科推理仍受语言与推理能力限制。

### 5.2 视频理解

| Benchmark | Previous SoTA | Gemini 1.5-Pro | GPT-4o | Qwen2-VL-72B | Qwen2-VL-7B |
|---|---:|---:|---:|---:|---:|
| MVBench | 69.6 | — | — | **73.6** | 67.0 |
| PerceptionTest | 66.9 | — | — | **68.0** | 62.3 |
| EgoSchema | 62.0 | 63.2 | 72.2 | **77.9** | 66.7 |
| Video-MME（无/有字幕） | 66.3/69.6 | **75.0/81.3** | 71.9/77.2 | 71.2/77.8 | 63.3/69.0 |

Qwen2-VL-72B 在 MVBench、PerceptionTest、EgoSchema 最好，但 Video-MME 无字幕低于 Gemini 1.5-Pro 和 GPT-4o。评测最多抽取 768 帧，长达一小时的视频可能因采样上限损失关键信息，因此结果不等于连续看完所有帧。

### 5.3 Grounding 与视觉 Agent

RefCOCO/+/g 上，Qwen2-VL-72B 在多数 split 达到通用模型顶级水平，例如 RefCOCO test-A 95.3、RefCOCO+ val 90.1，但个别项仍与专用模型或 InternVL2-76B 接近。相比初代 Qwen-VL，其高分辨率细节感知带来明显提升。

Agent 表中，Qwen2-VL-72B 的 function calling Type Match/Exact Match 为 93.1/53.2，高于 GPT-4o 的 90.2/50.0；AITZ 为 89.6/72.1，高于 GPT-4o 的 70.0/35.3。Card game 的 Number Line、EZPoint 达到 100.0。不过导航并不强：R2R success rate 51.7，低于专用 SoTA 79.0；REVERIE 为 31.0，低于专用 SoTA 61.0，也略低于 GPT-4o 31.6。这组结果揭示“屏幕局部定位”与“跨视角三维地图构建”是不同难度层级。

---

## 6. 消融实验：哪些设计真正起作用

### 6.1 动态分辨率的收益与反例

![](assets/Qwen2-VL%20-%20任意分辨率世界感知/figures/03-dynamic-resolution-ablation.png)

> 图 4：提高 `min_pixels` 对 InfoVQA、HallusionBench 通常有益，但 OCRBench 在中等分辨率达到峰值后明显下降，MMMU 也没有单调增益。它直接否定“像素越多一定越好”，说明训练分布与任务瓶颈同样重要。来源：论文 Figure 4，第 14 页；官方 arXiv 源码 `images/peng_fig.pdf`。

固定 token 消融比较 64、576、1600、3136 个平均视觉 token 与动态策略。动态策略平均使用 1924 token，InfoVQA 75.89、RealWorldQA 70.07、OCRBench 866、MMMU 53.44。它不是每项绝对最高：3136-token 固定策略的 InfoVQA 77.27、RealWorldQA 70.59；但动态策略以更低平均 token 达到接近最优，并在 OCRBench 显著领先固定策略。结论应表述为**更好的跨任务精度—成本折中**，而不是所有任务精度都最高。

OCRBench 在过高 `min_pixels` 下退化，作者解释为其中许多原图极小，过度放大使输入偏离训练分布。MMMU 对像素变化不敏感，说明其瓶颈更偏向复杂推理而非视觉分辨率。

### 6.2 M-RoPE 与长度外推

表 8 使用 Qwen2-1.5B + ViT-L 受控比较 1D-RoPE 与 M-RoPE。M-RoPE 在 MathVista 39.2→43.4、MMBench 58.6→60.6、PerceptionTest 46.6→47.4、NextQA 43.9→46.0、STAR 55.5→57.9；但 RealWorldQA 54.5→53.7、InfoVQA 50.8→50.3 略降，MMStar 不变。由此能支持“总体更好，视频增益更稳定”，不能声称每个图像任务都提升。

![](assets/Qwen2-VL%20-%20任意分辨率世界感知/figures/04-length-extrapolation.png)

> 图 5：Qwen2-VL-72B 在 Video-MME medium video 上从训练上限 16K 外推到 80K；32K、48K、64K、80K 的准确率仍在 70.4、71.3、70.4、71.1。M-RoPE 支持长度外推，但曲线非单调，且该图没有与普通 1D-RoPE 的同图长序列对照。来源：论文 Figure 5，第 15 页；官方 arXiv 源码 `images/peng_fig2.pdf`。

### 6.3 模型与数据规模

Figure 6 显示 2B→7B→72B 在多类能力上整体提升，数学能力与模型参数规模的正相关尤其明显；OCR 相关任务的小模型也相对强。对 Qwen2-VL-7B 的 Stage 2 token 进度曲线则显示，数据增加总体提升性能，AI2D、InfoVQA 更稳定，通用 VQA 有波动。论文给出趋势图但没有拟合明确的 power law，因此标题中的“explores scaling laws”更接近规模趋势探索，而非给出可预测的标度定律公式。

### 6.4 消融缺口

论文没有分别消融 2D-RoPE、$2\times2$ token merge、3D convolution、2 fps、视频分辨率预算，也没有给出动态分辨率与固定策略在统一 FLOPs/延迟下的系统对账。因此，M-RoPE 和动态分辨率的证据较直接，视频架构各组件的独立贡献则仍不明确。

---

## 7. 局限、失败模式与 Future Work

### 7.1 论文明确披露的边界

1. **长视频仍受帧数与上下文限制。** Video-MME 最多取 768 帧，作者明确把“支持更长序列、容纳更长视频”列为 future work。
2. **复杂推理仍落后强闭源模型。** MMMU 和 MathVision 暴露出单纯增加视觉分辨率无法解决的推理瓶颈。
3. **三维导航弱。** 论文指出多图产生的地图信息不完整、非结构化，准确建模三维环境中的地图与位置仍是重大挑战。
4. **分辨率存在分布外风险。** 对极小图过度放大可显著降低 OCRBench；动态输入不是无条件鲁棒。

### 7.2 由原文证据支持的批判性评价

动态分辨率把视觉信息损失从“固定 resize”转成“可配置 token 成本”，没有消除信息—计算权衡。M-RoPE 为统一图像/视频位置提供了简洁接口，但长序列实验主要集中在一个视频 benchmark；80K 外推不能自动推出任意长文档或多图 Agent 都同样稳定。Agent 结果中一部分来自内部 function-calling 集，数据构建又使用了高级 LVLM/LLM 生成问题与参数，外部可比性弱于公开 benchmark。

### 7.3 后续研究方向

论文明确提出扩展更长序列与更长视频。结合其失败分析，合理但需标为**本文启发而非作者明确承诺**的方向包括：用层次化视频压缩替代固定 2 fps；为导航引入显式地图/状态记忆；根据图像内容而非仅像素数分配视觉 token；在固定 FLOPs 下联合学习分辨率路由；为 Agent 建立公开、可复现的闭环评测。

---

## 8. 复现与实践建议

### 8.1 最小验证路线

1. 使用官方 2B 或 7B 权重，固定同一组文档、极端宽高比图片和视频。
2. 扫描 `min_pixels`/`max_pixels`，同时记录视觉 token 数、峰值显存、首 token 延迟和任务准确率。
3. 对视频固定抽帧集合，避免解码器差异污染结果；分别测试短视频、20 分钟视频与关键事件稀疏视频。
4. 对 GUI Agent 区分单步 grounding 与多步成功率，不能用 tap 坐标准确率替代完整任务完成率。
5. 若研究 M-RoPE，至少在同 backbone、同训练 token 下对比 1D-RoPE，并分别报告图像、视频和长上下文结果。

### 8.2 使用时的成本控制

生产系统应先按任务设像素预算：普通 VQA 不必把所有图像放到最高分辨率；OCR/文档可提高 `min_pixels`，但应防止极小图过度放大；视频应联合限制帧数、每帧像素和总视觉 token。对多轮 Agent，还需定期压缩历史截图，否则 32K 上下文会被重复观察快速占满。

---

## 9. 总结

Qwen2-VL 完成了 Qwen-VL 系列第一次架构级转向：从固定画布、固定视觉瓶颈，转到动态二维网格、变长视觉 token 和显式时空位置。2D-RoPE 让 ViT 接受任意宽高，M-RoPE 让 decoder 在同一序列里理解文本、图像与视频，3D convolution 则以两帧 tube 控制视频长度。其文档/OCR、真实场景、视频和 UI 操作结果证明这条路线有效；消融也说明动态分辨率和 M-RoPE 的收益并非“所有任务单调上涨”。它留下的核心问题是：如何在更长视频、更复杂三维环境与更高像素下，以可控算力保留真正有用的信息。

---

## 10. 材料来源

- 官方论文页：https://arxiv.org/abs/2409.12191
- 官方 PDF：https://arxiv.org/pdf/2409.12191
- 官方代码：https://github.com/QwenLM/Qwen2-VL
- 本地完整溯源：[sources.md](assets/Qwen2-VL%20-%20任意分辨率世界感知/sources.md)
