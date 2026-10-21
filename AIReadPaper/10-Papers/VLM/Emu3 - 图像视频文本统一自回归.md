---
type: paper-note
status: done
domain: VLM
paper: "Emu3: Next-Token Prediction is All You Need"
year: 2024
arxiv: "2409.18869"
doi: "10.48550/arXiv.2409.18869"
source: "https://arxiv.org/abs/2409.18869"
project: "https://emu.baai.ac.cn"
code: null
tags:
  - VLM
  - unified-multimodal
  - autoregressive-generation
  - video-generation
  - MoVQGAN
  - next-token-prediction
created: 2026-10-16
updated: 2026-10-16
---

# Emu3：图像视频文本统一自回归

## 0. 基本信息

- **作者：** Emu3 Team，BAAI；完整作者名单见原论文
- **模型：** 8B decoder-only Transformer，从零训练
- **模态：** 文本、图像、视频
- **核心主张：** 不依赖预训练 CLIP、预训练 LLM 或扩散模型，统一做 next-token prediction
- **本地原文：** [paper.pdf](assets/Emu3%20-%20图像视频文本统一自回归/sources/paper.pdf)
- **完整溯源：** [sources.md](assets/Emu3%20-%20图像视频文本统一自回归/sources.md)

## 1. 一句话结论

Emu3 把图像和视频先用时空 MoVQGAN 离散化，再与 Qwen tokenizer 的文本 token、分辨率/帧率元信息和结构 token 拼成“文档式”统一序列，用一个 8B decoder-only Transformer 从零学习 next-token prediction。它证明单一离散视觉表示可以同时支撑理解、文生图、文生视频和未来帧预测；代价是超长视觉序列、昂贵串行采样，以及视觉 tokenizer 的语义—重建折中被隐藏进数据规模和模型容量，而没有被结构性消除。

![](assets/Emu3%20-%20图像视频文本统一自回归/figures/01-next-token-overview.png)

> **图 1：Emu3 的统一范式。** 图像、视频、文本都先 token 化，随后由单一 Transformer decoder 预测下一个 token；输出再按模态 detokenize。来源：论文 Figure 1。

---

## 2. 研究问题与核心贡献

### 2.1 为什么挑战“扩散 + 组合式 VLM”

2024 年的主流系统通常分成两类：

- 理解：CLIP/SigLIP vision encoder + projector + 预训练 LLM；
- 生成：文本 encoder + diffusion/DiT，视频再使用独立时空生成器。

这种组合方式效果强，但每种模态与任务都有不同模块、目标和训练基础设施。Emu3 问的是：语言模型的 next-token prediction 能否像处理文本一样直接处理视觉，并且在同一模型里覆盖理解和生成？

### 2.2 核心贡献

1. **统一离散视觉 tokenizer。** 同一个基于 SBER-MoVQGAN 的 tokenizer 编码任意尺寸图像和视频，空间压缩 $8\times8$、时间压缩 $4\times$。
2. **统一词表与文档式序列。** 文本 token、视觉 code、结构 token 和纯文本 meta 信息进入同一序列。
3. **单一训练目标。** 预训练只使用加权 next-token cross-entropy，不引入扩散 loss 或单独对比学习目标。
4. **跨任务能力。** 同一预训练模型经不同 post-training 支持文生图、文生视频、视频未来预测和视觉语言理解。
5. **把 DPO 用于视觉 token。** 对生成图像 token 序列做偏好优化，提升人工评估中的美学与提示词遵循。

---

## 3. MoVQGAN：把图像与视频变成同一种离散 token

### 3.1 编码与量化

设图像或视频为 $x$，encoder 输出连续 latent：

$$
z_e=E_\phi(x).
$$

对每个 latent 位置，选择 codebook 中最近的向量：

$$
k^*=\arg\min_{k\in\{1,\ldots,K\}}\|z_e-e_k\|_2,
\qquad z_q=e_{k^*},
$$

再由 decoder 重建：

$$
\hat{x}=D_\psi(z_q).
$$

Emu3 的 codebook 大小为：

$$K=32{,}768.$$

视觉 tokenizer 基于 SBER-MoVQGAN-270M，并在 encoder/decoder 中各加入两个带 3D convolution 的 temporal residual layer，使同一模型既能处理单张图，又能建模连续帧。

### 3.2 压缩率与 token 数

空间压缩为 $8\times8$，时间压缩为 $4\times$。因此 $T\times H\times W$ 视频的视觉 token 数近似为：

$$
N_v=\left\lceil\frac{T}{4}\right\rceil
\left\lceil\frac{H}{8}\right\rceil
\left\lceil\frac{W}{8}\right\rceil.
$$

对 $4\times512\times512$ 视频：

$$N_v=1\times64\times64=4096.$$

对 $512\times512$ 单图同样是 $64\times64=4096$ 个 token。这个数字解释了为什么视频训练必须扩到 131,072 context：即使时间压缩 4 倍，较长视频仍迅速耗尽上下文。

### 3.3 tokenizer 训练目标

视觉 tokenizer 在 LAION-High-Resolution 图像和 InternVid 视频上端到端训练，组合目标包括：

$$
\mathcal{L}_{\text{tok}}=
\lambda_2\mathcal{L}_{2}+
\lambda_p\mathcal{L}_{\text{LPIPS}}+
\lambda_g\mathcal{L}_{\text{GAN}}+
\lambda_c\mathcal{L}_{\text{commit}}.
$$

论文列出目标组成但未披露全部权重。L2 保像素，LPIPS 保感知结构，GAN 强化纹理，commitment loss 约束 encoder 输出靠近 codebook。

![](assets/Emu3%20-%20图像视频文本统一自回归/figures/02-tokenizer-reconstruction.png)

> **图 2：视觉 tokenizer 重建。** 左侧是 540×960 视频的 8 帧原图/重建对比，右侧是 512×512 图像；整体内容稳定，但微小文字、细线和高频纹理仍受离散压缩影响。来源：论文 Figure 3。

在 3,172 个 Pexels 视频上，分辨率从 $128^2$ 到 $720^2$ 的 LPIPS 约 0.099–0.112；$720^2$ 时 PSNR 24.30、SSIM 0.771。指标并非随分辨率单调变化，且这是 tokenizer 重建质量，不等于最终 Transformer 生成质量。

---

## 4. 统一词表、统一序列与模型结构

### 4.1 模型配置

| 配置 | Emu3 |
|---|---:|
| 参数量 | 8B |
| 层数 | 32 |
| hidden size | 4096 |
| intermediate size | 14,336 |
| attention heads | 32 |
| KV heads | 8 |
| vocabulary size | 184,622 |
| RoPE base | 1,000,000 |
| context length | 131,072 |

架构沿用 Llama 类 decoder-only Transformer：RMSNorm、GQA、SwiGLU、RoPE，qkv 与线性投影不使用 bias，dropout 0.1。与“CLIP + LLM”不同，Emu3 没有独立语义 vision encoder；与 Janus 不同，也没有理解/生成两套 adaptor。视觉 code 直接扩展 embedding 和输出词表。

### 4.2 文档式序列

预训练样本统一成：

```text
[BOS] {caption text} [SOV] {meta text} [SOT]
{vision tokens with [EOL]/[EOF]} [EOV] [EOS]
```

- `[SOV]`：视觉输入开始；
- `[SOT]`：真正视觉 token 开始；
- `[EOL]`：二维网格换行；
- `[EOF]`：视频帧边界；
- `[EOV]`：视觉输入结束；
- `meta text`：以普通文本描述分辨率；视频还包含帧率和时长。

将 caption 放在视觉序列前面得到生成样本；把部分 caption 移到 `[EOV]` 之后，就得到理解样本。统一性来自**改变序列方向而非改变网络**：

$$
\text{生成}: \quad \text{text}\rightarrow\text{vision},
$$

$$
\text{理解}: \quad \text{vision}\rightarrow\text{text}.
$$

### 4.3 统一词表的含义

词表大小 184,622，覆盖 QwenTokenizer 的多语言文本 token、32,768 个视觉 code 和结构 token。论文没有逐项给出精确词表分解，因此不应把“184,622−32,768”直接当作纯文本词表大小；其中还包含新增特殊 token。

统一 embedding/head 的优势是所有模态共享参数和尺度化路径；风险是视觉 token 数远多于文本 token，会主导梯度和 softmax 类别竞争。

---

## 5. 纯 next-token prediction 与 attention mask

### 5.1 训练目标

对统一序列 $x_{1:L}$：

$$
\mathcal{L}=-\sum_{i=1}^{L}w(x_i)
\log p_\theta(x_i\mid x_{<i}),
$$

权重为：

$$
w(x_i)=\begin{cases}
0.5,&x_i\text{ 是视觉 token},\\
1,&\text{其他受监督 token}.
\end{cases}
$$

视觉 loss 降为 0.5，是为了防止每张图数千个 token 淹没文本学习。这个处理说明“只有一个 objective”不等于“所有 token 完全同权”。

### 5.2 attention mask

Emu3 使用 decoder-only 标准 causal mask：

$$
M_{ij}=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

论文没有报告为图像使用双向 attention、理解/生成切换 mask 或额外 cross-attention。

- 文生图/视频：视觉 token 看得到 caption、meta 和此前视觉 token。
- 图像理解：回答文本位于 `[EOV]` 之后，因此能看见完整图像 token。
- 视频：视觉 token 先按帧内二维栅格排序，再用 `[EOF]` 串起时间；后续帧能看过去帧，不能看未来帧。

`[EOL]` 与 `[EOF]` 是关键，因为纯一维 causal Transformer 本身不知道二维行边界和帧边界。它们把空间/时间结构写进序列语法，而不是写进 attention 算子。

### 5.3 与扩散模型的差异

扩散模型通常在噪声时间步 $t$ 上并行预测噪声/速度：

$$
\mathcal{L}_{\text{diff}}=\mathbb{E}_{x,\epsilon,t}
\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2.
$$

Emu3 则显式建模离散 token 的有序联合概率：

$$
p(\mathbf{q}\mid c)=\prod_{i=1}^{N_v}p(q_i\mid c,q_{<i}).
$$

优点是训练接口统一、天然支持任意前缀续写和视频 future prediction；缺点是 $N_v$ 次串行解码、顺序偏置和误差累积。

---

## 6. 数据与预训练

### 6.1 数据构建

**语言数据：** 使用 Aquila 的中英文高质量语料。

**图像数据：** 开源网页、AI 生成和内部高质量数据混合；过滤低于 $512\times512$ 的图，LAION aesthetic < 5.5 的样本通常删除，同时用 OCR 与颜色过滤保留部分低审美但有理解价值的图。额外构建图表、表格、文本密集等理解数据。约 1M 图文对由 GPT-4V 标注，用其微调 Emu2-17B captioner，再进行大规模合成 caption。

**视频数据：**

1. PySceneDetect 按内容变化和淡入淡出切 scene；
2. 2 FPS、短边 256 做 OCR，删除文字覆盖过多片段；
3. 计算 optical flow，过滤几乎静止或运动过强片段；
4. 每段取 3 帧做 aesthetic 评分，最低分 < 5 则删除；
5. 用 GPT-4V 标注小规模数据，再训练 video captioner。短于 20 秒取 12 帧 caption，长视频切成 10–20 秒子片段。

论文没有披露三类数据的总样本数、token 数和采样比例，这是复现的主要缺口。

### 6.2 两阶段预训练

| 阶段 | 模态 | context | 学习率 | 说明 |
|---|---|---:|---:|---|
| Stage 1 | 文本+图像 | 5,120 | $5\times10^{-5}$ | 从零训练，不用视频 |
| Stage 2 | 文本+图像+视频 | 131,072 | $5\times10^{-5}$ | 引入视频与超长上下文 |

两阶段都使用 cosine annealing 到 0。训练结合 tensor parallel、context parallel 和 data parallel；文本—图像样本 packing 到最大长度，但保证一张完整图像不被切断。论文没有报告 GPU 数、训练天数、总 FLOPs 和各并行度配置。

---

## 7. Post-training

### 7.1 视觉生成：QFT

Quality Fine-Tuning 只对视觉 token 计算 cross-entropy：

$$
\mathcal{L}_{\text{QFT}}=-\sum_{i\in\mathcal{T}_{v}}
\log p_\theta(q_i\mid x_{<i}).
$$

图像按 HPSv2.1、MPS 和 LAION Aesthetics 三个偏好分数筛选，训练分辨率从约 512 提到约 720；视频使用更严格的分辨率和 optical-flow 过滤。QFT 仍是 next-token prediction，没有换成 diffusion loss。

### 7.2 视觉生成：DPO

每个用户 prompt 采样 8–10 个候选，三位标注者按视觉吸引力和提示词一致性评分，最高分作 chosen、最低分作 rejected。DPO 目标可写为：

$$
\mathcal{L}_{\text{DPO}}=-\log\sigma\left[
\beta\left(
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)}-
\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}
\right)\right].
$$

训练把 DPO loss 与 next-token cross-entropy 组合。论文直接保存候选的视觉 token，避免重新 tokenization 引入 reconstruction 差异。这一点很重要：偏好应针对同一离散序列，而不是受第二次量化误差污染。

### 7.3 视觉理解：两阶段后训练

1. image-to-text：混合图像理解与纯语言数据，忽略视觉 token loss，只做文本预测；
2. instruction tuning：加入 QA 数据，提高视觉指令遵循。

理解阶段图像保持宽高比；第一阶段约 $512^2$ 面积，第二阶段把分辨率限制在 512–1024 范围。这说明同一个视觉 tokenizer 能服务理解，但训练时仍通过 loss mask 选择文本目标。

---

## 8. 生成采样与视频处理

### 8.1 图像采样

Emu3 使用 classifier-free guidance：

$$
\ell_{\text{cfg}}=\ell_u+s(\ell_c-\ell_u).
$$

附录披露：

- MSCOCO：CFG scale 5.0；
- GenEval：每 prompt 生成 4 张，scale 5.5；
- T2I-CompBench：每 prompt 生成 10 张，scale 5.0；
- DPG-Bench：每 prompt 生成 4 张，scale 5.0。

论文未披露常规采样的 temperature、top-k、top-p 等参数，因此这些必须从官方配置补齐，不能自行猜测。

### 8.2 视频生成

模型原生生成 5 秒、24 FPS 视频，并可自回归续写。由于 tokenizer 时间压缩 4 倍，24 FPS 的 5 秒视频在 latent 时间轴上约 30 个时间位置；空间 token 数再乘每帧 latent 网格，因此序列仍很长。

生成步骤是：

1. 文本 caption + 视频 meta（分辨率、FPS、时长）形成前缀；
2. 逐 token 生成帧内视觉 code；
3. `[EOL]` 标记行结束，`[EOF]` 标记帧结束；
4. `[EOV]` 后由 MoVQGAN decoder 还原视频；
5. 附录还训练了独立 spatial-temporal U-Net 做 4× super-resolution。

最后一点限制了“单一 Transformer”的字面解释：**核心世界/内容建模确实只有 next-token Transformer，但像素重建依赖视觉 tokenizer decoder，高分辨率输出还可依赖额外超分网络。**“无需 diffusion”不等于“端到端只有一个神经网络组件”。

### 8.3 视频未来预测

![](assets/Emu3%20-%20图像视频文本统一自回归/figures/03-video-extension.png)

> **图 3：视频续写。** 上下文帧先转成视觉 token，模型继续预测后续 token，再统一 detokenize；同一个因果目标自然支持 future prediction。来源：论文 Figure 8。

实验以 2 秒、24 FPS 视频作上下文，再预测后续 2 秒；作者还观察到可用 2 秒上下文迭代扩展约 8 秒。论文主要给定性案例，没有报告 future prediction 的 FVD、动作准确率或长期一致性曲线，因此不能从图例断言模型已学到可靠物理规律。

---

## 9. 实验结果

### 9.1 文生图

| 模型 | COCO CLIP-I↑ | CLIP-T↑ | FID↓ | GenEval↑ | DPG-Bench↑ |
|---|---:|---:|---:|---:|---:|
| SDXL | 0.674 | 0.310 | — | 0.55 | 74.65 |
| DALL-E 3 | — | 0.320 | — | 0.67* | 83.50 |
| Chameleon | — | — | 26.74 | 0.39 | — |
| Emu3 | **0.689** | **0.313** | 12.8 | **0.66*** | 80.60 |
| Emu3-DPO | 0.680 | 0.312 | 19.3 | 0.64* | **81.60** |

`*` 表示使用 prompt rewriting。Emu3 的 GenEval 接近 DALL-E 3、高于 SDXL；DPG-Bench 高于 SDXL/PixArt-alpha，但低于 DALL-E 3。DPO 后人工评估改善，自动 FID/GenEval 反而下降，说明偏好优化倾向美学与整体对齐，不一定符合自动指标分布。

100 个中英文 prompt、每个三位评审的人评显示，Emu3-DPO 整体分数高于 SDXL，并接近 DALL-E 3 与 Midjourney v5.2。样本量仍较小，且 prompt 与评审细节不足以支撑广泛产品级结论。

### 9.2 文生视频

VBench 总分为 **80.96**，高于表中多数开源视频扩散模型，但低于 Kling（81.85）与 Gen-3（82.32）。Dynamic Degree 为 **79.27**，在表中较高；Subject/Background Consistency 分别为 95.32/97.69，而 Scene、Appearance Style 分别为 37.11/20.92，并非各维度都领先。更准确的结论是：Emu3 在运动程度和一致性上很强，但场景与风格表现仍有短板。

### 9.3 视觉语言理解

| 指标 | Emu3 | LLaVA-1.6-7B |
|---|---:|---:|
| SEEDBench-Img | **68.2** | 64.7 |
| OCRBench | **687** | 532 |
| MM-Vet | 37.2 | **43.9** |
| POPE | 85.2 | **86.5** |
| VQAv2 | 75.1 | **81.8** |
| GQA | 60.3 | **64.2** |
| ScienceQA-Img | **89.2** | 70.2 |
| TextVQA | 64.7 | **64.9** |
| ChartQA | **68.6** | 54.8 |
| DocVQA | **76.3** | 74.4 |
| MMMU | 31.6 | **35.1** |
| MMBench | 58.5 | **67.4** |

Emu3 在 SEED、OCR、ScienceQA、ChartQA、DocVQA 等有竞争力，但并未全面超过 encoder-based VLM。论文的更准确贡献是：**一个不依赖预训练 CLIP/LLM、只用离散视觉 token 的 decoder-only 模型，第一次在多项理解基准上达到可竞争水平。**

### 9.4 消融证据与缺口

论文没有系统提供以下控制变量消融：

- MoVQGAN 单视觉表示 vs CLIP/SigLIP+VQ 双表示；
- 去掉 `[EOL]`/`[EOF]`；
- 视觉 loss 权重 0.5 vs 1.0；
- 无视频 Stage 1 vs 直接三模态训练；
- 5,120 vs 131,072 context；
- 不同 temporal compression；
- 预训练数据模态比例。

最清楚的后训练证据是 DPO 人评：视觉质量 60.6→61.6，prompt alignment 52.3→57.3；但自动指标变差。tokenizer 则有重建指标和图例，却没有与其他视频 tokenizer 在统一设置下的完整对比。因此论文的系统结果很强，机制归因弱于 Janus 的表示冲突消融。

---

## 10. 与 Janus 的统一范式对比

参见：[[Janus - 解耦视觉编码的理解生成统一模型]]。

| 维度 | Emu3 | Janus |
|---|---|---|
| 视觉表示 | 单一 MoVQGAN 离散 code | SigLIP 连续语义 + VQ 离散重建双表示 |
| 统一位置 | 输入词表、embedding、Transformer、输出 head 全部统一 | 只统一 Transformer；视觉入口与输出 head 分开 |
| 是否预训练视觉语义 encoder | 否 | 是，SigLIP |
| 是否预训练 LLM | 否，从零训练 8B | 是，DeepSeek-LLM 1.3B |
| attention mask | 标准 causal | 标准 causal |
| 视觉结构 | `[EOL]`、`[EOF]`、meta text | 视觉网格展平；原版未强调显式行/帧 token |
| 预训练视觉 loss | 视觉 token 权重 0.5 | 理解不预测视觉；生成预测 VQ token |
| 冲突处理 | 用 tokenizer 质量、规模与数据吸收冲突 | 用双 encoder 从结构上隔离冲突 |
| 视频 | 原生支持 | 原版不支持 |
| 最大 context | 131K | 4K |
| 生成后训练 | QFT + DPO | 三阶段统一 SFT；Janus-Pro 主要改数据/日程 |

### 10.1 单表示 vs 双表示

Emu3 的单表示具有真正的词表统一和视频扩展优势：任何视觉输入都能成为可预测 token，理解与生成只需调换序列方向。Janus 的双表示更务实：理解直接继承 SigLIP 的语义先验，生成保留 VQ 可逆性，避免让 32K codebook 同时承担判别和重建。

### 10.2 attention mask

两者都使用 causal mask，没有为理解图像设置双向视觉 attention。区别在于：

- Janus 的 SigLIP 在进入 LLM 前已用双向 ViT 完成全局视觉编码，所以 LLM 中视觉前缀即使 causal，单个视觉 token 的输入特征已经包含全图上下文。
- Emu3 的视觉 token 是局部离散 code，跨位置全局语义主要由 causal Transformer 在序列中逐步建立；早期视觉位置看不到后续位置，但回答文本位于图像之后，最终能汇总全图。

这可能是 Janus 在小模型理解上更高效的重要原因，也解释 Emu3 为什么需要更大模型和更多数据。

### 10.3 语义—重建冲突

Janus 的 Table 5 直接表明：普通 VQ 联合理解+生成时理解很差；语义 tokenizer 在联合训练中也比仅理解训练明显下降。Emu3 没有否认这种冲突，而是展示 8B 规模和大规模混合数据下，单 tokenizer 仍能达到竞争结果。两篇论文并不矛盾：

- Janus 回答“在有限规模下如何高效避免冲突”；
- Emu3 回答“是否可能通过规模化让一个离散接口覆盖三模态”。

---

## 11. 局限与批判性思考

1. **序列成本极高。** $512^2$ 单图即 4096 视觉 token；视频需要 131K context。标准 attention 的训练成本随长度近似 $O(L^2)$，即使 CP 能分摊显存，也不能消除总计算。
2. **生成串行延迟。** 图像/视频每个 code 逐步采样；高分辨率和长视频的 wall-clock latency 是扩散并行去噪之外的另一种高成本。
3. **单 tokenizer 存在不可见上限。** 模型不能恢复量化器已丢失的细小文字、面部和纹理；超分网络只能补外观，不能保证恢复真实语义。
4. **统一主张有工程边界。** 系统仍需要 MoVQGAN encoder/decoder，附录还有 4× 时空 U-Net 超分；统一的是生成建模目标，不是所有视觉处理都由 Transformer 完成。
5. **训练数据与成本不透明。** 总规模、采样比、内部数据、GPU 数和训练时长均未完整披露，难以复现 scaling 结论。
6. **结果比较存在协议差异。** 部分 GenEval 使用 GPT-4V rewriting；VBench 对不同模型的视频长度/分辨率和公开可比性也需谨慎。
7. **未来预测缺少定量评估。** 定性续写不能充分证明物理一致性、长期因果或 world model 能力。
8. **自动指标与人偏好冲突。** DPO 改善人评但使 FID/GenEval 下降，说明对齐目标会改变输出分布，不能用单一榜单概括质量。

---

## 12. 复现路线与检查点

### 12.1 分阶段复现

1. **先复现 tokenizer。** 在公开图像+视频上训练 MoVQGAN，验证 $4\times8\times8$ 压缩和 32,768 codebook；先报告重建 LPIPS/PSNR/SSIM。
2. **构建统一词表。** 为视觉 code 预留不与文本冲突的 ID，加入 `[SOV]/[SOT]/[EOL]/[EOF]/[EOV]`。
3. **验证序列方向。** 同一图文对同时构造 text→vision 与 vision→text；检查 meta text、宽高比和 frame boundary 可逆。
4. **小规模 Stage 1。** 用 5K context 训练文本+图像，先验证单图生成与 captioning。
5. **再引入视频。** 扩 context，使用 context parallel；监控每 batch 各模态有效 token 数。
6. **分别 post-train。** 生成做 QFT/DPO，理解做 image-to-text + instruction tuning，避免混淆各阶段效果。

### 12.2 必做 sanity checks

- codebook utilization/perplexity，防止 codebook collapse；
- 图像和视频重建误差按文字、脸、小目标、快速运动分桶；
- `[EOL]/[EOF]` 去除消融，验证结构 token 是否必要；
- 视觉 loss 权重 $0.5$ 的网格搜索，观察语言 perplexity 与生成质量 Pareto 曲线；
- 固定 prompt rewrite、CFG、样本数与分辨率再比较 GenEval/FID；
- 报告生成吞吐（tokens/s、单图/5 秒视频延迟）与 KV cache 显存；
- 视频续写使用 FVD、VBench temporal 指标和动作一致性，而不只展示样例。

### 12.3 可复现性判断

模型结构和 tokenizer 压缩参数披露较清楚，但完整训练数据规模、采样比例、算力、常规采样参数未披露；完整复现困难。最现实的研究复现是缩小模型，验证“统一离散序列 + causal LM”是否在同一预算下优于双表示或组合式 baseline。

---

## 13. 关键术语

- **MoVQGAN：** 可同时处理图像和视频的向量量化生成式 tokenizer。
- **Unified vocabulary：** 文本 token、视觉 code 和结构 token 共用一个模型词表。
- **Next-token prediction：** 根据所有历史 token 预测下一个离散符号的因果建模目标。
- **Context parallelism：** 沿超长序列维度切分 attention 计算和状态。
- **QFT：** 用高质量样本继续做视觉 token 交叉熵训练。
- **DPO：** 直接利用 chosen/rejected 序列优化偏好，无需显式奖励模型。
- **Future prediction：** 把已有视频 token 作为前缀，继续自回归生成未来帧。

## 14. 图表索引

1. `01-next-token-overview.png`：统一图像/视频/文本 next-token 框架，论文 Figure 1。
2. `02-tokenizer-reconstruction.png`：MoVQGAN 图像与视频重建，论文 Figure 3。
3. `03-video-extension.png`：视频上下文与未来预测，论文 Figure 8。