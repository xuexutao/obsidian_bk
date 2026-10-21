---
type: paper-note
status: complete
domain: VLM
paper: "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation"
year: 2024
arxiv: "2408.12528"
doi: null
source: "https://arxiv.org/abs/2408.12528"
project: null
code: "https://github.com/showlab/Show-o"
tags:
  - VLM
  - unified-multimodal
  - discrete-diffusion
  - MaskGIT
  - Omni-Attention
  - decoder-only
created: 2026-10-16
updated: 2026-10-16
---

# Show-o：统一理解与离散扩散生成

## 0. 基本信息与阅读边界

- **作者**：Jinheng Xie、Weijia Mao、Zechen Bai、David Junhao Zhang、Weihao Wang、Kevin Qinghong Lin、Yuchao Gu、Zhijie Chen、Zhenheng Yang、Mike Zheng Shou
- **发表**：ICLR 2025；本地 PDF 为 arXiv v7
- **规模**：1.3B，基于 Phi-1.5；attention 层增加 QK-Norm
- **核心定位**：同一个 decoder-only Transformer 对文本做 AR next-token prediction，对离散图像 token 做 MaskGIT 风格 masked token prediction。
- **本地原文**：[paper.pdf](assets/Show-o%20-%20统一理解与离散扩散生成/sources/paper.pdf)
- **来源清单**：[sources.md](assets/Show-o%20-%20统一理解与离散扩散生成/sources.md)

> [!important] 与普通 image-to-text VLM 的区别
> Show-o 不只把 MAGVIT/CLIP feature 喂给 LLM 回答问题，还扩展 8,192 个 image token，并让同一 Transformer 在图像块上执行离散 diffusion；图像经 25–50 轮并行去 mask 后由 MAGVIT-v2 解码。它补足的是**视觉输出词表、Omni-Attention、MTP loss 与图像迭代解码**。

**一句话结论：** Show-o 在 Chameleon 的全离散统一接口上，把图像从 1,024 次串行 AR 改为 25–50 轮并行 masked denoising；Omni-Attention 让文本保持 causal、每个图像块保持 full attention，代价是离散 tokenizer 对理解较弱、需要大规模对齐，且视觉质量仍受 codebook 与重建上界限制。

![](assets/Show-o%20-%20统一理解与离散扩散生成/figures/01-unified-model-taxonomy.png)

> **Figure 1，PDF p.2。** Show-o 位于“一个 LLM 同时包含 AR + diffusion”这一象限，区别于只理解的 LLaVA、只生成的 diffusion/AR 模型和 LLM 外挂生成器。来源：官方 PDF 完整图，已 Read 验证。

---

## 1. 问题定义：AR 文本与 diffusion 图像能否共享一个 decoder

Chameleon 证明图文都离散化后可用单 AR Transformer，但图像二维 token 的 raster-order 没有自然因果性，$512^2$、downsample 16 时需要 1,024 次串行采样。连续 diffusion 质量更强，却通常需要独立 text encoder 与 denoiser。

Show-o 的折中是：

- 文本仍是 LLM 擅长的 AR NTP；
- 图像仍是离散 token，但使用 absorbing-state discrete diffusion / MaskGIT 式 MTP；
- 共享 embedding space、Transformer 与任务 prompt；
- 用 attention mask 和 loss 选择让不同模态保留不同生成范式。

![](assets/Show-o%20-%20统一理解与离散扩散生成/figures/02-show-o-overview.png)

> **Figure 2，PDF p.4。** 统一主干覆盖 caption/VQA、T2I、inpainting/extrapolation 与交错视频关键帧生成；文字用 causal attention，图像 masked block 用 full attention。来源：官方 PDF 完整图，已 Read 验证。

---

## 2. Tokenizer、主干、词表与 output head

### 2.1 文本 tokenizer

直接继承 Phi-1.5 的 tokenizer 与预训练权重，不修改文本 tokenization。这样语言知识来自已有 LLM，而不是像 Unified-IO 2/Chameleon 从零重建。

### 2.2 默认视觉 tokenizer：MAGVIT-v2 lookup-free quantizer

- codebook $K=8,192$；
- $256\times256\rightarrow16\times16=256$ 个离散 image token；
- $512\times512$ 阶段相应为 $32\times32=1,024$ token；
- image de-tokenizer 将最终 code grid 解码为像素。

默认 Show-o 在**理解和生成两侧都使用同一离散 token**。论文另做两种理解入口消融：MAGVIT-v2 quantization 前连续 feature、CLIP-ViT continuous feature；生成侧仍使用 MAGVIT token。

### 2.3 decoder-only 主干

除每层 attention 前增加 QK-Norm 外，主干不改 Phi-1.5。模型扩展 embedding table，加入 8,192 个 learnable image embeddings，并加入 `[MMU]`、`[T2I]`、`[SOT/EOT]`、`[SOI/EOI]`、`[MASK]` 等控制 token。

### 2.4 output head

- 文本位置：预测文本 vocabulary，用 AR NTP。
- 图像 masked 位置：论文明确 logits $\ell_i^t\in\mathbb R^{K+1}$，覆盖 8,192 image codes + `[MASK]`。
- 论文没有明确说明文本/图像 projection 是否物理上为一个完全共享矩阵，也没有披露 weight tying；可确认的是共享 hidden/embedding space，并按任务与位置限制有效输出域。

---

## 3. Unified Prompting：任务语义进入序列

![](assets/Show-o%20-%20统一理解与离散扩散生成/figures/03-unified-prompting.png)

> **Figure 4，PDF p.5。** `[MMU]` / `[T2I]` 决定目标，SOT/EOT 与 SOI/EOI 标记文本/图像块；交错生成只是在同一序列重复这些块。来源：官方 PDF 完整图，已 Read 验证。

典型格式：

```text
MMU: [MMU] [SOI] image_tokens [EOI] [SOT] question/answer [EOT]
T2I: [T2I] [SOT] prompt [EOT] [SOI] masked_image_tokens [EOI]
Mixed: [T2I] text image text image ...
```

任务 token 不只是自然语言 instruction；它还控制 attention pattern、有效 output token 域和使用 NTP/MTP 的位置。

---

## 4. Omni-Attention：因果跨块，图像块内 full attention

![](assets/Show-o%20-%20统一理解与离散扩散生成/figures/04-omni-attention.png)

> **Figure 5，PDF p.5。** 灰色为可见。纯文本退化为 causal；MMU/T2I 中图像块内部双向，跨模态按序列方向传递；交错文档中每个图像块 full-attend 自身并只读取前序块。来源：官方 PDF 完整图，已 Read 验证。

设图像块集合为 $\mathcal B$，mask 可概括为：

$$
M_{ij}=\begin{cases}
0,&j\le i,\\
0,&\exists B\in\mathcal B:\ i,j\in B,\\
-\infty,&\text{otherwise}.
\end{cases}
$$

具体含义：

- **MMU**：输入 image tokens 在块内全连接；后续 question/answer text 能看完整图像，文本内部 causal。
- **T2I**：image tokens 都能看前序 prompt，并在图像块内互看，适合并行预测 masked positions。
- **Language modeling**：无图像块时就是标准 triangular causal mask。
- **Mixed generation**：后面的文本/图像块能看所有前序块；每个图像块内部 full attention，但不能看未来块。

它与 Transfusion 的 block mask几乎同构，差别在图像数据与 loss：Show-o 是离散 code classification，Transfusion 是连续 noise regression。

---

## 5. 双目标：NTP + MTP

### 5.1 多模态理解与文本生成

给定图像 token $u_{1:M}$ 和文本 $v_{1:N}$，最大化：

$$
\mathcal L_{\mathrm{NTP}}
=\sum_i\log p_\theta(v_i\mid v_{<i},u_{1:M}).
$$

实现为负对数似然最小化。理解侧只监督文本答案，图像是条件。

### 5.2 图像离散 diffusion / Mask Token Prediction

随机 timestep 决定 mask ratio，将部分图像 token 替换成 `[MASK]` 得到 $u^*$。只在 masked positions 计算：

$$
\mathcal L_{\mathrm{MTP}}
=\sum_{j\in\mathcal M}\log p_\theta(u_j\mid u^*,v_{1:N}).
$$

所有 masked position 一次 forward 并行预测；未 mask 的图像 token 提供全局上下文。训练时以一定概率把条件文本替换为空字符串，为 classifier-free guidance 准备 unconditional branch。

### 5.3 联合目标

$$
\mathcal L=\mathcal L_{\mathrm{MTP}}+\alpha\mathcal L_{\mathrm{NTP}}.
$$

论文给出形式但未公开主实验 $\alpha$ 的明确数值，因此复现应把它视为关键超参数而非固定常数。

### 5.4 与三种替代目标的区别

- **Chameleon AR**：图像 token 逐个 NTP，严格 1,024 步。
- **Unified-IO 2 AR denoising**：encoder 提供 corrupted input，decoder 仍因果地产生离散目标。
- **Transfusion DDPM**：连续 VAE latent 上回归 Gaussian noise。
- **Show-o MTP**：离散 code 上分类，整块并行去 mask。

---

## 6. 图像采样与解码

### 6.1 MaskGIT 式迭代

初始图像块全部为 `[MASK]`。第 $t$ 轮并行得到 $M$ 个 logits。CFG 为：

$$
\ell^t=(1+w)\ell_c^t-w\ell_u^t.
$$

对每个 masked 位置从 $\operatorname{softmax}(\ell_i^t)$ 采样 code，并把该 code 的概率作为 confidence $s_i$。根据 mask schedule $\gamma$ 计算下一轮需保留的 mask 数：

$$
m=\left\lceil\gamma\left(\frac{t}{T}\right)M\right\rceil.
$$

confidence 最低的 $m$ 个位置重新 mask，其余锁定；循环到 $T$ 后由 MAGVIT-v2 decoder 解码。

### 6.2 采样步数与 CFG

- 5 steps：只能生成大致相关内容；
- 25 steps：prompt adherence 明显改善；
- 50 steps：细节与真实感更强；
- $w=3$ 或 5 时颜色、内容与 prompt 一致性增强。

同分辨率 AR 需要 1,024 次 token step，Show-o 约 20 倍更少轮数；但每轮会并行处理整个图像块，不能把“50 vs 1024”直接等同于 20 倍 wall-clock。

### 6.3 Inpainting、extrapolation 与 editing

已知区域 token 固定，只 mask 待编辑位置，就自然得到 inpainting/extrapolation；无需额外微调。mask-free editing 则对原图 token 迭代随机 masking，再由文本条件重采样，可做局部对象替换或全局风格变化。

---

## 7. 三阶段训练与数据

### 7.1 训练阶段

1. **Image embedding / pixel dependency**：RefinedWeb 保持语言；ImageNet-1K class→image 学图像依赖；image-text pair 做 captioning。
2. **双向图文对齐**：用 image-text pair 同时训练 captioning 与 text-to-image，替换 class-conditional generation。
3. **高质量微调**：约 1M 内部高质量 T2I pair + LLaVA instruction data + GenHowTo mixed generation。

### 7.2 数据

- RefinedWeb：约 1B instances / 968M web pages / 2.8TB；
- ImageNet-1K：1.28M images；
- 基础图文：约 35M，来自 CC12M、SA1B、LAION-aesthetics-12M；
- 扩展图文：约 2.0B，加入 DataComp、COYO700M，并用 ShareGPT4V recaption；
- instruction：LLaVA-Pretrain-558K、LLaVA-v1.5-mix-665K；
- mixed video keyframe/text：GenHowTo；
- final T2I：约 1M internal high-quality pairs。

### 7.3 优化

- Stage 1：500k steps；Stage 2：1,000k steps；
- 48×A100 80GB，总 batch 1,152；
- AdamW，weight decay 0.01，warm-up 5k，初始 LR $10^{-4}$ + cosine；
- 2B-data 版本继续 500k steps，再升到 $512^2$ 训练 500k steps，最后高质量/指令微调。

---

## 8. 结果与消融

### 8.1 理解

默认离散 Show-o 1.3B：POPE 80.0、MME 1097.2、Flickr30k 62.5、VQAv2 69.4、GQA 58.0、MMMU 26.7。CLIP continuous 版本提升到 84.5、1232.9、67.6、74.7、61.0、27.4。

它接近同规模 LLaVA-v1.5-Phi-1.5，但理解指标并非全部更好；这支持“统一没有完全毁掉理解”，不支持“小模型全面胜过专用 VLM”。

### 8.2 图像生成

- COCO-30K zero-shot FID：9.24（1.3B、35M pairs）；
- GenEval：0.68，CLIP continuous 理解版 0.69；
- 表中优于 Chameleon 0.39、Transfusion 0.63、SDXL 0.55，与/高于 SD3 0.62；但数据、caption、分辨率与评测设置跨论文不完全一致。

### 8.3 数据、分辨率与表示消融

| 图文规模 | 分辨率 | POPE | MME | Flickr30k | VQAv2 | GQA |
|---:|---:|---:|---:|---:|---:|---:|
| 35M | 256 | 73.8 | 948.4 | 36.2 | 59.3 | 48.7 |
| 2.0B | 256 | 76.2 | 1014.9 | 48.9 | 64.7 | 54.2 |
| 2.0B | 512 | 80.0 | 1097.2 | 62.5 | 69.4 | 58.0 |

离散 image embedding 必须从头与语言对齐，因此数据规模和 token 数都显著影响理解。35M/256 条件下，CLIP continuous 特征明显优于 MAGVIT continuous，更明显优于 MAGVIT discrete；原因是 CLIP 的 400M 对比学习已塑造语义空间，而 MAGVIT 优先重建像素。

Unified pretraining 对 MAGVIT discrete 理解提升最大；对 CLIP continuous 略有负迁移，说明共享 backbone 仍存在任务干扰。

---

## 9. 视频与混合模态输出

Show-o 将 MAGVIT-v2 膨胀为视频 tokenizer：8 FPS、$3\times17\times256\times256$ 输入压为 $5\times16\times16$ token。它能：

- 视频 token→AR 文本描述；
- text→离散 diffusion 视频 token；
- 交错生成“步骤文字→关键帧→步骤文字→关键帧”。

单帧内部是 diffusion/full attention，帧/块之间按序列 causal，因此时间建模是“块内并行、块间自回归”。论文主要给定性例子，没有强视频 benchmark 与时序一致性量化，不能视为成熟视频生成器。

---

## 10. 局限与失败模式

1. **离散 tokenizer 的双重瓶颈**：生成受重建质量限制，理解受 code embedding 语义不足限制。
2. **理解仍偏弱**：默认离散版低于 CLIP continuous；统一表示最干净，但不是最优感知接口。
3. **文字与计数失败**：论文例子中 OCR 把 “closing down” 识别错误，人数计数错误，生成也难写出 “mardefly”。
4. **对象绑定失败**：多人物场景不能为每个人生成正确的 skis，暴露关系/计数问题。
5. **数据依赖重**：2B pairs + automatic recaption + 1M internal high-quality data，完整复现困难。
6. **采样仍是迭代式**：25–50 full-image rounds；CFG 需要 conditional/unconditional logits。
7. **loss balance 未公开**：$\alpha$、mask schedule 细节与 null-text 概率对复现重要，但论文主文没有完整数值。
8. **对比存在版本/设置差异**：表中跨论文 GenEval/FID 不能替代统一评测。
9. **视频证据不足**：主要为定性演示，分辨率、长度与 temporal consistency 都有限。

**判断**：Show-o 的核心贡献不是单一 benchmark，而是给出一套可执行的“AR 文本 + 离散 diffusion 图像 + block-wise Omni-Attention”协议。它比 Chameleon 更适合高效视觉输出，又比 Transfusion 更容易保持全离散统一接口，但 tokenizer 的语义/重建矛盾仍未解决。

---

## 11. 复现建议

1. 固定 MAGVIT-v2 checkpoint，先测 $256^2$ reconstruction、code usage、OCR 与小物体保真。
2. 在 1.3B backbone 上分别验证 MMU/T2I/mixed mask，写单元测试确保图像块内双向、块间不看未来。
3. 分开记录 NTP 与 MTP loss，扫描 $\alpha$，避免图像训练吞噬语言能力。
4. 扫描 5/25/50 steps 与 CFG $w=1/3/5$，同时测 FID、GenEval、实际延迟和显存。
5. 对离散/continuous understanding 比较时统一 vision encoder 数据规模，避免 CLIP 400M 预训练成为主要混杂。
6. OCR、counting、attribute binding 必须列为独立失败集；不能只展示审美样图。
7. 复现 mixed video 时报告每帧重建、跨帧一致性与长序列 drift，而非仅抽帧展示。

---

## 12. 四路线对照中的位置

| 维度 | Show-o 选择 | 直接后果 |
|---|---|---|
| 主干 | Phi-1.5 decoder-only + QK-Norm | 继承语言能力，改动小 |
| 图像 tokenizer | MAGVIT-v2 LFQ，8,192 codes | 输入输出统一离散，但语义需重学 |
| output | 文本 vocab / image $K+1$ logits | 同主干按位置切换有效输出域 |
| attention | Omni：文本 causal、图像块 full | 支持并行 MTP 与交错因果 |
| loss | MTP + $\alpha$NTP | 各模态保留合适目标 |
| 采样 | confidence remask，25–50 轮 | 比 1,024-step AR 少轮，但仍需迭代 |

---

## 13. 材料路径与自检

| 检查项 | 路径/结论 | 状态 |
|---|---|---|
| 笔记 | `10-Papers/VLM/Show-o - 统一理解与离散扩散生成.md` | 已完成 |
| 官方 PDF | `assets/Show-o - 统一理解与离散扩散生成/sources/paper.pdf` | 已保存 |
| 来源记录 | `assets/Show-o - 统一理解与离散扩散生成/sources.md` | 已完成 |
| 完整原图 | `figures/01-*.png` 至 `04-*.png` | 4 张，已逐张 Read 验证 |
| 编号标题 | 0–13 连续 | 通过 |
| 公式 | Omni mask、NTP、MTP、联合 loss、CFG、remask | 通过 |
| 视觉输出架构 | MAGVIT、词表/head、Omni-Attention、MTP、采样均覆盖 | 通过 |
| Paper-Index | 未写入、未修改 | 通过 |
