---
type: paper-note
status: complete
domain: VLM
series: BLIP
series_order: 5
paper: "BLIP3-o: A Family of Fully Open Unified Multimodal Models—Architecture, Training and Dataset"
year: 2025
arxiv: 2505.09568
doi: 10.48550/arXiv.2505.09568
source: https://arxiv.org/abs/2505.09568
project: https://github.com/JiuhaiChen/BLIP3o
code: https://github.com/JiuhaiChen/BLIP3o
tags:
  - VLM
  - Unified-Multimodal
  - Image-Generation
  - Flow-Matching
  - Diffusion-Transformer
  - CLIP
  - BLIP
created: 2026-10-20
updated: 2026-10-20
---

# BLIP3-o：全开放统一多模态理解与生成

## 基本信息

- **论文**：*BLIP3-o: A Family of Fully Open Unified Multimodal Models—Architecture, Training and Dataset*
- **机构**：Salesforce Research、University of Maryland、UC Davis、New York University、Virginia Tech、University of Washington 等
- **提交时间**：2025-05-14，arXiv:2505.09568
- **官方代码**：<https://github.com/JiuhaiChen/BLIP3o>
- **模型与数据**：发布 4B/8B 模型、预训练 caption 和 BLIP3o-60k 指令数据；但 8B 额外使用了 30M 专有图像，严格意义上只有公开数据训练的 4B 版本满足“训练数据全开放”
- **系列位置**：[[BLIP - 生成式字幕过滤的统一视觉语言预训练]] → [[BLIP-2 - 冻结底座的高效视觉语言桥接]] → [[InstructBLIP - 指令感知视觉特征抽取]] → [[BLIP-3 - 开放大规模多模态模型家族]] → **BLIP3-o**
- **本地 PDF**：[paper.pdf](assets/BLIP3-o%20-%20全开放统一多模态理解与生成/sources/paper.pdf)

## 一句话结论

BLIP3-o 不是“BLIP-4”，而是 BLIP-3/xGen-MM 向**统一理解—生成模型**扩展的分支：它冻结已有 Qwen2.5-VL 理解主干，用自回归模型产生中间视觉条件，再以 Flow Matching DiT 生成固定长度的 CLIP 语义特征，最后经扩散解码器恢复像素。论文最有价值的结论不是又做了一个文生图模型，而是用受控实验说明：在统一框架中，**CLIP 语义特征比 VAE latent 更易学，Flow Matching 比 MSE 更能覆盖多模态图像分布，先理解后生成的顺序训练比联合训练更稳**。

---

## 1. 研究背景与问题定义

### 1.1 统一理解与生成究竟统一什么

LLaVA、Qwen-VL、InternVL 等模型已经证明，图像编码器、connector 和语言模型可以高效完成理解任务；Emu3、Chameleon 则把图像离散成 token，与文本一起自回归预测。BLIP3-o 关注另一种统一方式：**语言与视觉语义由自回归模型组织，连续视觉分布由扩散模型生成**。

这条路线面临三个尚未统一回答的问题：

1. 生成目标应使用偏低层外观的 VAE latent，还是偏高层语义的 CLIP feature？
2. 连续视觉特征应用简单 MSE 回归，还是用扩散/Flow Matching 建模完整条件分布？
3. 理解和生成应联合训练，还是先获得理解能力、再冻结主干增加生成能力？

BLIP3-o 的研究贡献，就是把这三个轴放在相近训练设置下比较，再据此构建完整模型。

### 1.2 与 BLIP 主线的连续性和断裂

BLIP 初代用 CapFilt 改善图文语料，BLIP-2 用 Q-Former 把冻结视觉编码器连接到冻结 LLM，InstructBLIP 让视觉查询感知指令，BLIP-3/xGen-MM 则强调开放的大规模 VLM 配方。BLIP3-o 延续了“开放资源 + 高效桥接”的组织思想，但技术重心已经变化：

- 不再以 Q-Former 为核心；理解主干直接使用 Qwen2.5-VL。
- 任务从理解扩展到文生图，因此需要连续生成头和视觉解码器。
- “o”更接近 omni/unified 分支标记，而不是正式的第四代编号。

---

## 2. 任务定义与输入输出

### 2.1 图像理解

输入是图像 $I$ 与文本指令 $C$。CLIP/视觉编码器将图像映射为视觉特征，自回归多模态模型根据图文上下文预测答案 token：

$$\mathcal L_{\mathrm{understand}}=-\sum_i\log p_\theta(y_i\mid y_{<i},I,C).$$

实际模型直接复用 Qwen2.5-VL 已有的理解能力，生成训练阶段冻结这部分参数，因此 BLIP3-o 没有重新证明“共同训练会让理解更强”，而是证明附加生成能力可以不明显破坏已有理解能力。

### 2.2 图像生成

给定文本提示 $C$，模型在文本后附加一组学习查询 $Q$。自回归模型输出的查询状态形成中间视觉条件 $h_Q$，DiT 以 $h_Q$ 为条件，从噪声采样 CLIP 图像特征 $z_{\mathrm{clip}}$；一个单独训练的扩散解码器再把 CLIP feature 还原为像素图像。

因此实际生成链是：

$$C\rightarrow h_Q\rightarrow z_{\mathrm{clip}}\rightarrow I.$$

这不是单阶段 latent diffusion。第一阶段在语义特征空间采样“要画什么”，第二阶段把语义特征解码成“具体怎样画”。代价是推理链更长，也要求 CLIP feature decoder 足够强。

![](assets/BLIP3-o%20-%20全开放统一多模态理解与生成/figures/01-architecture.png)

> **图 1（论文 Figure 1）**：理解侧把 CLIP 图像特征送入自回归模型并预测文本；生成侧把文本产生的 Visual Feature 作为 DiT 条件，以 Flow Matching 对齐真实 CLIP feature。两侧“统一”的核心是共享语义空间，而不是把所有模态强行离散成同一词表。  
> 来源：arXiv TeX `fig/fig1.pdf`，<https://arxiv.org/abs/2505.09568>

---

## 3. 核心方法

### 3.1 CLIP feature 与 VAE latent 的差别

VAE latent 为高保真像素重建服务，通常保留颜色、纹理和局部结构；空间分辨率随输入图像增长，序列较长。CLIP feature 则把图像压成固定 **64 个连续向量**，更加语义化、长度更短，对自回归模型更友好。

作者的判断是：统一模型中的自回归主干本来擅长语义和组合关系，因此不必让它直接承担低层像素 latent 的复杂分布；先预测紧凑 CLIP feature，再交给扩散解码器补外观，是更自然的职责分工。

这个设计并非无损。CLIP 对颜色细节、文字和精确空间布局未必敏感；生成质量上限受 CLIP encoder 丢失信息和 decoder 重建能力共同约束。

### 3.2 MSE 与 Flow Matching

最简单的方式是把自回归查询状态线性映射到真实视觉特征：

$$\mathcal L_{\mathrm{MSE}}=\lVert X-WQ\rVert_2^2.$$

MSE 易训练，但对同一提示倾向预测条件均值，采样随机性弱，同一 prompt 多次推理的结果相似。

Flow Matching 从高斯噪声 $X_0$ 与真实视觉特征 $X_1$ 之间构造路径：

$$X_t=(1-t)X_0+tX_1,\qquad t\sim\mathcal U(0,1),$$

训练条件速度场：

$$\mathcal L_{\mathrm{FM}}=\mathbb E\left[\left\|v_\phi(X_t,h_Q,t)-(X_1-X_0)\right\|_2^2\right].$$

推理时对该速度场积分得到 CLIP feature。它学习的是条件分布而非单点回归，因此能从不同噪声得到多样图像。

![](assets/BLIP3-o%20-%20全开放统一多模态理解与生成/figures/02-design-choices.png)

> **图 2（论文 Figure 3）**：三种受控设计分别是 CLIP+MSE、CLIP+Flow Matching 和 VAE+Flow Matching。实验没有把 VAE+MSE 作为主要方案，因为论文关注的是“自回归条件 + 连续扩散生成”路线。  
> 来源：arXiv TeX `fig/figure3_arxiv.pdf`。

### 3.3 DiT 结构

生成头基于 Lumina-Next / improved Next-DiT，使用：

- 3D RoPE 编码时间、高度与宽度，而非学习绝对位置表；
- attention/MLP 前后的 sandwich RMSNorm；
- Grouped-Query Attention 降低 KV 成本并改善稳定性。

8B 版本以 Qwen2.5-VL-7B-Instruct 为冻结主干，额外训练约 **1.4B** 生成参数；4B 版本以 Qwen2.5-VL-3B-Instruct 为主干。这里的“8B/4B”是整套系统口径，不应误读为全部参数都在生成训练中更新。

### 3.4 联合训练与顺序训练

联合训练将理解与生成样本混合，同时更新 LLM/MLLM 与生成头；顺序训练则先获得理解模型，再冻结主干，只训练生成模块。

![](assets/BLIP3-o%20-%20全开放统一多模态理解与生成/figures/04-training-strategies.png)

> **图 4（论文 Figure 5）**：左侧联合训练需要同时平衡理解和生成梯度；右侧顺序训练冻结已有 MLLM，只让 Visual Feature 与 DiT 学生成。论文最终采用后者，优势是避免生成任务扰动语言与理解能力，缺点是两侧不能端到端共同适应。  
> 来源：arXiv TeX `fig/figure5_arxiv.pdf`。

---

## 4. 数据与训练配方

### 4.1 预训练数据

4B 开放版本使用约 **25M** 公开图像，来自 CC12M、SA-1B 和 JourneyDB；每张图用 Qwen2.5-VL-7B-Instruct生成平均约 120 token 的详细 caption，并混入约 **3M** 条短 caption。

8B 版本在约 25M 公开图像外加入 **30M 专有图像**，并混入约 **6M** 短 caption。论文因此同时存在两种“开放”口径：资源接口全部开放，但 8B 训练语料并非全部公开；若要求从数据到权重完全可复核，应以 4B 为主。

### 4.2 BLIP3o-60k 指令微调

预训练模型对复杂人体姿势、常见物体类别、地标和文字生成仍不稳定。作者让 GPT-4o 针对这些类别生成 prompt–image 对，再混入 JourneyDB/DALL·E 3 风格提示，构成约 **60k** 的 BLIP3o-60k。

这揭示一个重要事实：大规模 caption 预训练并没有自然覆盖高难组合与人类偏好。小而有针对性的 SFT 数据能快速改善 aesthetic 与 prompt alignment，但也使模型能力依赖外部闭源生成器产生的合成数据。

---

## 5. 实验结果

### 5.1 设计消融

![](assets/BLIP3-o%20-%20全开放统一多模态理解与生成/figures/03-design-ablation.png)

> **图 3（论文 Figure 4）**：CLIP+Flow 在 GenEval 与 DPG 上整体最好，VAE+Flow 的 FID 最低，CLIP+MSE 早期收敛快但后期上限较低。它说明“最好的感知分布指标”与“最好的提示对齐”并不是同一个设计目标。  
> 来源：arXiv TeX `fig/ablation.pdf`。

三个结论应分开读：

1. **CLIP 比 VAE 更快学到语义对齐**，不是说 CLIP 重建外观必然更强。
2. **Flow 比 MSE 更有多样性**，但增加 DiT 参数与迭代采样开销。
3. **VAE+Flow 的 FID 更好**，表明 FID 偏好局部外观分布；作者最终选择 CLIP+Flow，是因为统一模型更重视语义和指令对齐。

### 5.2 理解 benchmark

| 模型 | VQAv2 | MMBench | SEED | MM-Vet | MME-P | MMMU | TextVQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLIP3-o 4B | 75.9 | 78.6 | 73.8 | 60.1 | 1527.7 | 46.6 | 78.0 |
| BLIP3-o 8B | **83.1** | **83.5** | **77.5** | **66.6** | **1682.6** | **50.6** | **83.1** |

这些成绩主要继承 Qwen2.5-VL，而不是生成头反向增强理解。正确结论是“加生成能力后仍保住强理解”，而不是“BLIP3-o 的生成训练创造了这些理解能力”。

### 5.3 生成 benchmark

| 模型 | GenEval ↑ | DPG-Bench ↑ | WISE ↑ |
|---|---:|---:|---:|
| Janus-Pro 7B | 0.80 | **84.19** | 0.35 |
| BLIP3-o 4B | 0.81 | 79.36 | 0.50 |
| BLIP3-o 8B | **0.84** | 81.60 | **0.62** |

BLIP3-o 在 GenEval 与 WISE 上更强，但 DPG 自动评分低于 Janus-Pro。作者进一步用约 1000 个 DPG prompt 做人工比较。

![](assets/BLIP3-o%20-%20全开放统一多模态理解与生成/figures/05-human-study.png)

> **图 5（论文 Figure 6）**：视觉质量上 BLIP3-o 胜 50.4%、Janus-Pro 胜 44.9%；提示对齐上 BLIP3-o 胜 51.5%、Janus-Pro 胜 46.1%。绝对优势不大，但统计显著，且方向与自动 DPG 分数相反，说明单一 VLM judge 不足以决定统一生成模型排名。  
> 来源：arXiv TeX `fig/figure6_arxiv.pdf`。

---

## 6. 与相关工作的关系

### 6.1 对比离散统一模型

Chameleon、Emu3、Show-o 将图像变成离散 token，使训练目标统一为 next-token prediction。优点是接口简洁，缺点是视觉词表量化损失和长序列成本。BLIP3-o 则保留连续图像特征，通过 DiT 专门建模视觉分布；它牺牲“单一损失函数”的形式统一，换来更适合连续生成的模块分工。

### 6.2 对比 Transfusion/MetaMorph

这些工作也把自回归与扩散结合起来。BLIP3-o 的区别是系统比较 CLIP/VAE 表征、MSE/Flow 目标和 joint/sequential 训练，并把开放模型、数据与训练脚本作为主要贡献之一。

### 6.3 对比 BLIP-2/BLIP-3

BLIP-2 的 Q-Former 是固定数量查询的视觉—语言桥，重点是冻结底座的理解效率；BLIP-3 构建开放理解模型家族；BLIP3-o 直接从强 Qwen2.5-VL 出发，将查询状态变成扩散生成条件。三者共同保留“视觉信息先压成小接口，再接语言/生成主干”的思想，但具体桥接模块已不相同。

---

## 7. 局限与批判性评价

1. **统一仍是模块化统一。** 理解 encoder、自回归主干、CLIP feature DiT、扩散 decoder 是多段系统，并非一个共享参数的端到端世界模型。
2. **两级扩散推理成本高。** 先采样 CLIP feature、再扩散解码像素，速度与部署成本都高于单一生成头；论文没有把端到端 latency、显存和 NFE 作为核心比较。
3. **顺序训练避免冲突，也限制互相促进。** 冻结理解主干让能力稳定，却阻止生成监督改善空间、文字与外观表征。
4. **复杂人体、手部和文字仍会失败。** BLIP3o-60k 能缓解但不能根治，说明 CLIP 语义空间对精确姿态、拓扑和字形仍不充分。
5. **8B 并非数据完全开放。** 30M 专有图像无法由外部严格复现；开放性判断应区分“代码权重开放”和“训练数据可重建”。
6. **理解提升不可归因于统一训练。** 主要 benchmark 来自冻结 Qwen2.5-VL 的既有能力，论文更可靠地证明的是“兼容”，而不是“生成促进理解”。
7. **评测仍偏静态图像。** 尚未展示多轮交错生成、图像编辑、视频或 agent 行动；离“全模态通用模型”仍有距离。

---

## 8. 复现与实践建议

- **最低成本验证**：冻结公开 BLIP3-o 4B/Qwen2.5-VL-3B，只复现 CLIP+MSE 与 CLIP+Flow 小规模对照，观察同一 prompt 的多样性和收敛速度。
- **完整预训练**：需要约 25M 公开图像及重 caption、训练 1B 级 DiT、训练 CLIP feature decoder；数据清洗和 caption 成本远高于普通 LoRA。
- **指标建议**：除 GenEval、DPG、WISE 外，应报告人工偏好、文本渲染、手部/人体、推理 NFE、延迟与显存，并在相同 decoder 下比较 representation。
- **工程风险**：两级扩散使调试链较长。应分别验证 CLIP feature generator 与 decoder 的 oracle reconstruction，避免把 decoder 上限误归因于文本条件模型。

---

## 9. 个人判断与后续问题

BLIP3-o 最值得保留的不是某个榜单第一，而是一条清楚的架构经验：**在统一多模态系统中，自回归模型负责语义决策，扩散模型负责连续视觉分布；共享的 CLIP 空间是两者的接口。** 这比把视觉强行离散成语言 token 更符合模块能力，但它也说明“统一”并不意味着“所有模态都用同一预测头”。

下一步真正关键的问题有三个：

1. CLIP feature 是否足以承载精确几何、文字和物理关系？
2. 能否让生成监督反哺理解，而不造成灾难性干扰？
3. 能否把双扩散链蒸馏成少步或单步，同时保持理解—生成一致性？

---

## 10. 材料来源

完整材料、图号与原始 URL 见 [sources.md](assets/BLIP3-o%20-%20全开放统一多模态理解与生成/sources.md)。
