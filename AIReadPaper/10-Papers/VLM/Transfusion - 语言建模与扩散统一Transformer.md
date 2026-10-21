---
type: paper-note
status: complete
domain: VLM
paper: "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model"
year: 2024
arxiv: "2408.11039"
doi: null
source: "https://arxiv.org/abs/2408.11039"
project: null
code: null
tags:
  - VLM
  - unified-multimodal
  - diffusion
  - continuous-latent
  - decoder-only
  - multimodal-generation
created: 2026-10-16
updated: 2026-10-16
---

# Transfusion：语言建模与扩散统一 Transformer

## 0. 基本信息与阅读边界

- **作者**：Chunting Zhou、Lili Yu、Arun Babu、Kushal Tirumala、Michihiro Yasunaga、Leonid Shamis、Jacob Kahn、Xuezhe Ma、Luke Zettlemoyer、Omer Levy
- **模型规模**：0.16B / 0.37B / 0.76B / 1.4B / 7B；大模型另有约 0.27B U-Net encoder/decoder 参数
- **核心定位**：同一个 decoder-only Transformer 对文本使用 next-token prediction，对连续图像 latent 使用 DDPM noise prediction。
- **本地原文**：[paper.pdf](assets/Transfusion%20-%20语言建模与扩散统一Transformer/sources/paper.pdf)
- **来源清单**：[sources.md](assets/Transfusion%20-%20语言建模与扩散统一Transformer/sources.md)

> [!important] 与普通 image-to-text VLM 的区别
> Transfusion 的视觉输入不是只用来条件化文本输出。图像 VAE latent 可以出现在混合序列中，Transformer 既能从图像生成 caption，也能在 `<BOI>...<EOI>` 区间直接预测连续噪声并通过 DDPM 反演生成图像。它补的是**连续视觉输出 head、图像块 attention 与扩散解码过程**。

**一句话结论：** Transfusion 用“共享 Transformer、模态专用输入输出层、双损失”避开了 Chameleon 的图像量化与共享 softmax 竞争；受控实验显示连续 diffusion 比离散图像 NTP 有更好的 scaling，但统一模型仍依赖 VAE/U-Net、250 步采样和私有大规模图片数据。

![](assets/Transfusion%20-%20语言建模与扩散统一Transformer/figures/01-unified-transformer.png)

> **Figure 1，PDF p.2。** 文本位置输出离散 token 分布；BOI/EOI 内的连续 patch 同时处理并输出噪声估计。来源：官方 PDF 完整图，已 Read 验证。

---

## 1. 核心问题：为什么不把图像也做成语言 token

离散 AR 路线架构简单，却把 VQ 量化误差、长图像序列和文本/图像词表竞争一起带入 LM。专业图像生成器采用连续 latent diffusion，说明图像和文本可能不应共享同一种概率分布与损失。

Transfusion 的原则是：

- **共享高容量推理主干**；
- **不强迫共享数据类型、output head 或 loss**；
- 文本保持离散分类，图像保持连续回归。

这是一种比“单词表统一”更弱、但更符合模态统计的统一。

---

## 2. 数据表示与主干

### 2.1 混合序列

文本经 Llama 2 tokenizer 得到整数 token。每张图先进入预训练 VAE，得到连续 latent，再按 raster order 展成 patch vector，并用 `<BOI>` / `<EOI>` 包围：

```text
[text tokens] <BOI> [continuous image patches] <EOI> [text tokens] ...
```

同一序列可表示 text→image、image→text 与多轮交错。BOI/EOI 是离散控制 token；中间图像元素不是 vocabulary ID。

### 2.2 decoder-only Transformer

主干采用 Llama 风格 decoder block：SwiGLU、RoPE、causal 基础 mask。绝大多数参数共享；模态差异只在轻量输入/输出层与 attention mask。

### 2.3 文本 embedding 与 output head

- input：标准 token embedding；
- output：线性投影到固定文本 vocabulary，softmax 后计算 CE；
- BOI 输入位置不计算 LM loss；
- 图像连续 patch 不占用文本词表，因此没有“图像 codebook + 文本 BPE”联合 softmax。

论文未给出完整文本 vocabulary 数字，沿用 Llama 2 tokenizer；也未强调 embedding/head weight tying。

---

## 3. VAE、patch encoder/decoder 与视觉 output head

![](assets/Transfusion%20-%20语言建模与扩散统一Transformer/figures/02-vae-patch-architecture.png)

> **Figure 3，PDF p.5。** 图像先经 VAE 进入连续 latent，训练时加噪；线性层或 U-Net down 把局部 latent patch 映射到 Transformer hidden，输出侧对称还原并由 VAE decoder 解码。来源：官方 PDF 完整图，已 Read 验证。

### 3.1 VAE

86M CNN VAE 将 $256\times256$ 图像压成 $32\times32\times8$ latent：每个 latent pixel 对应约 $8\times8$ 原图区域。VAE 训练目标为

$$
\mathcal L_{\mathrm{VAE}}
=\mathcal L_1+\mathcal L_{\mathrm{LPIPS}}+0.5\mathcal L_{\mathrm{GAN}}
+0.2\mathcal L_{\mathrm{ID}}+10^{-6}\mathcal L_{\mathrm{KL}}.
$$

GAN loss 在 50k step 后启用。与 Chameleon 的受控对比使用相同数据、compute、encoder/decoder；Chameleon 只把 KL 换成 codebook commitment loss，尽量隔离“连续 vs 离散”。

### 3.2 patch 化

| latent patch | 原图感受野 | 每图序列元素 | 线性层 | U-Net down/up |
|---|---:|---:|---|---|
| $1\times1$ | $8\times8$ | 1,024 | 无额外压缩 | — |
| $2\times2$ | $16\times16$ | 256 | 默认受控设置 | 默认大模型设置 |
| $4\times4$ | $32\times32$ | 64 | 可用 | 更稳健 |
| $8\times8$ | $64\times64$ | 16 | 性能明显掉 | 仍保持较强图像结果 |

线性版只做局部向量投影；U-Net down/up 带二维卷积与归纳偏置，可在压到 16 patch 时仍保留图像结构。7B 时 U-Net 额外 0.27B，仅占约 3.8%。

### 3.3 图像 output head

Transformer 在图像位置输出 hidden vector，经 linear 或 U-Net up 还原到 VAE latent 维度，预测的是连续 Gaussian noise $\epsilon$，不是 code ID。VAE decoder 只在 diffusion 完成后将干净 latent 解成像素。

---

## 4. Transfusion attention：因果序列 + 图像块内双向

![](assets/Transfusion%20-%20语言建模与扩散统一Transformer/figures/03-intra-image-attention.png)

> **Figure 4，PDF p.5。** 全局仍遵循序列因果顺序，但同一图像的所有 patch 互相可见；后续文本能看整张图，图像不能看后续文本。来源：官方 PDF 完整图，已 Read 验证。

设 $b(i)$ 表示 token $i$ 所属图像块，mask 为：

$$
M_{ij}=\begin{cases}
0,&j\le i,\\
0,&i,j\text{ 都是图像 patch 且 }b(i)=b(j),\\
-\infty,&\text{otherwise}.
\end{cases}
$$

因此：

- 文本仍严格 causal；
- 图像内部 full attention，符合 DDPM 同时预测整张 latent 的需求；
- 一个图像块能看之前文本/图像，不能看之后内容；
- 后续文本能把整张图作为条件。

0.76B 线性版把 causal 改为块内双向后，FID 61.3→20.3、CIDEr 12.7→16.0；U-Net 本身已有局部双向信息，因此增益较小但仍在。

---

## 5. 双目标训练

### 5.1 文本 NTP

$$
\mathcal L_{\mathrm{LM}}
=\mathbb E_{y_i}\left[-\log P_\theta(y_i\mid y_{<i})\right].
$$

只在文本 token 上逐 token 计算；图像位置不进入文本 softmax。

### 5.2 图像 DDPM

前向加噪：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
$$

cosine noise schedule，训练预测噪声：

$$
\mathcal L_{\mathrm{DDPM}}
=\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2\right].
$$

噪声在 patchification 前加到整张 VAE latent；loss 按图像聚合，而不是每个离散 token 分类。

### 5.3 联合损失

$$
\mathcal L_{\mathrm{Transfusion}}
=\mathcal L_{\mathrm{LM}}+\lambda\mathcal L_{\mathrm{DDPM}},
\qquad \lambda=5.
$$

同一 batch、同一 Transformer 参数同时收到两类梯度。论文只做了初步 $\lambda$ 选择，没有系统扫描，这也是复现敏感项。

### 5.4 理解方向的噪声限制

训练中 80% pair 为 caption→image，20% 为 image→caption。后者若也把图像加到接近纯噪声，captioning 条件会被破坏。作者将 image-first 样本的最大 timestep 限为 $t\le500$；7B CIDEr 33.7→35.2，其他指标变化小。这说明同一图像既作 diffusion target 又作 understanding condition 时，noise policy 必须随方向调整。

---

## 6. 推理与解码：LM/扩散状态机

### 6.1 文本模式

按标准 LM 自回归采样；论文实验主要用 greedy decoding，分类用候选排序。

### 6.2 图像模式

1. LM 采到 `<BOI>` 后切换 diffusion mode；
2. 按目标尺寸插入 $n$ 个纯高斯噪声 patch $x_T$；
3. 在每个 timestep 让图像块 full-attend，并预测 $\epsilon_\theta(x_t,t,c)$；
4. scheduler 计算 $x_{t-1}$，**原位覆盖**序列中的 $x_t$；模型看不到历史噪声轨迹；
5. 重复到 $x_0$，追加 `<EOI>`；
6. 切回 LM mode，必要时继续生成文本；
7. patch decoder + VAE decoder 输出图像。

模型训练 1,000 diffusion timesteps，推理标准用 250 steps。受控实验 CFG=5；消融常用 CFG=3；大模型按 benchmark 调参。CFG 需要 conditional/unconditional 两次 forward，约加倍图像阶段计算。

---

## 7. 数据与训练

### 7.1 受控 scaling 实验

- 0.5T token/patch，文本与图像按 1:1 token ratio；
- 文本：Llama 2 tokenizer + 2T-token corpus；
- 图像：380M licensed Shutterstock pairs，$256^2$ center crop；
- caption-first 80%，image-first 20%；
- sequence 4096、batch 2M token、250k steps；
- AdamW，$\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-8}$，LR $3\times10^{-4}$，warm-up 4k，cosine decay 到 $1.5\times10^{-5}$，weight decay 0.1，clip 1.0。

### 7.2 7B / 2T 实验

- 1T text token；
- 约 3.5B image-caption pair exposure / 1T patch-token equivalent；
- 每 epoch 混合 692M pairs：Shutterstock、220M 公开无人物图片、上采样 80M 含人物图、CC12M；
- 最后 1% upweight 高审美图片；
- batch 4M token、500k steps。

数据大量为 licensed/private collection，完整训练集不可复现；但论文的受控 VAE/VQ-VAE 设计为连续/离散比较提供了较可信的内部证据。

---

## 8. 结果与消融

![](assets/Transfusion%20-%20语言建模与扩散统一Transformer/figures/04-scaling-comparison.png)

> **Figure 5，PDF p.9。** 控制参数、数据与 compute 后，Transfusion 在文本 PPL、理解 CIDEr、T2I FID/CLIP 上的 scaling 曲线整体优于 Chameleon。来源：官方 PDF 完整图，已 Read 验证。

### 8.1 与离散 AR 的受控比较

7B、0.5T：

| 模型 | C4 PPL↓ | Wiki PPL↓ | Llama Acc↑ | CIDEr↑ | FID↓ | CLIP↑ |
|---|---:|---:|---:|---:|---:|---:|
| Transfusion | 7.72 | 4.28 | 61.5 | 27.2 | 16.8 | 25.5 |
| Chameleon | 8.41 | 4.69 | 59.1 | 18.0 | 29.6 | 24.3 |

达到 Chameleon 同等性能时，Transfusion 的 estimated parity FLOP ratio 分别约 0.489、0.526、0.600、0.218、0.029、0.319；FID 上约 34 倍 compute efficiency 是最强主张，但来自拟合 scaling 曲线而非同一次直接运行。

### 8.2 patch/U-Net 消融

- 线性 patch 从 256 压到 16：FID 20.3→43.5，明显损伤。
- U-Net patch 256/64/16：FID 16.7/16.0/16.1，压缩稳定；CIDEr 25.4/29.9/29.5。
- 7B 上 U-Net vs linear：CIDEr 33.7 vs 27.2，FID 16.0 vs 18.6。
- U-Net 并非只靠参数量：随着主干变大，额外参数占比降到 3.8%，收益仍在。

### 8.3 大模型对比

7.3B Transfusion：COCO FID 6.78、GenEval 0.63、Llama suite 66.1；超过 DALL-E 2、SDXL 的表中 GenEval，接近 DeepFloyd，低于 SD3 0.68。它同时有 Llama-1 级文本结果，这是专业 diffusion model 不具备的。

但跨论文比较的数据、caption 增强、采样与模型组件不同；例如 SD3 使用 synthetic recaption，论文自己也承认会显著提高 GenEval。

---

## 9. 局限与批判性判断

1. **并非完全单体**：仍依赖 86M VAE 与可选 0.27B U-Net 模态层；“single model”主要指共享 Transformer。
2. **图像推理仍慢**：250 DDPM steps，CFG 时每步双 forward；比 Show-o 的 25–50 轮离散去 mask 更慢。
3. **分辨率有限**：主实验只有 $256^2$，没有原生多尺度、任意宽高比或高分辨率验证。
4. **理解较弱**：image→text 主要用 COCO captioning，没有广泛 VQA/OCR/grounding；不能等同于成熟通用 VLM。
5. **数据不可公开复现**：380M Shutterstock 与最终 692M mixture 的关键部分受许可限制。
6. **loss balance 未充分消融**：$\lambda=5$ 来自 preliminary experiments，没有说明规模变化时是否仍最优。
7. **比较边界**：对 Chameleon 的实现是作者重建的受控 baseline，不是原始 34B 完整训练；scaling 结论可靠性高于绝对产品质量结论。
8. **条件图像被加噪**：必须为 image-first 样本限制噪声，说明统一 diffusion/understanding 存在目标冲突。

**判断**：Transfusion 最重要的不是“又一个文生图模型”，而是证明视觉输出不必被迫进入语言词表。它保留语言 NTP 与图像 DDPM 各自的统计优势，再通过共享主干和 block attention 统一上下文，这是四篇中最清晰的“连续输出 head”方案。

---

## 10. 复现建议

1. 先复现同数据 VAE/VQ-VAE，报告 reconstruction FID/LPIPS，保证连续/离散输入公平。
2. 从 0.76B、$2\times2$ latent patch 开始，对比 causal 与 intra-image bidirectional mask。
3. 固定序列 token ratio、image exposure 与 theoretical FLOPs，避免 patch 压缩导致“看过更多图片”的混杂。
4. 扫描 $\lambda$、noise-limit 与 CFG；同时记录文本 PPL、CIDEr、FID、CLIP 和实际 wall-clock。
5. 验证原位覆盖 $x_t$：序列不能累计所有 diffusion timestep，否则条件长度和训练定义不一致。
6. 若扩展高分辨率，先评估 2D position、U-Net receptive field 与 block attention 显存，而不是只增 patch 数。

---

## 11. 材料路径与自检

| 检查项 | 路径/结论 | 状态 |
|---|---|---|
| 笔记 | `10-Papers/VLM/Transfusion - 语言建模与扩散统一Transformer.md` | 已完成 |
| 官方 PDF | `assets/Transfusion - 语言建模与扩散统一Transformer/sources/paper.pdf` | 已保存 |
| 来源记录 | `assets/Transfusion - 语言建模与扩散统一Transformer/sources.md` | 已完成 |
| 完整原图 | `figures/01-*.png` 至 `04-*.png` | 4 张，已逐张 Read 验证 |
| 编号标题 | 0–11 连续 | 通过 |
| 公式 | VAE、attention mask、NTP、DDPM、联合损失 | 通过 |
| 视觉输出架构 | 连续 latent、U-Net/head、block mask、DDPM 采样均覆盖 | 通过 |
| Paper-Index | 未写入、未修改 | 通过 |
