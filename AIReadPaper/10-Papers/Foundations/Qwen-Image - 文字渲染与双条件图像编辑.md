---
type: paper-note
status: complete
domain: Foundations
paper: Qwen-Image Technical Report
year: 2025
arxiv: 2508.02324
doi: null
source: https://arxiv.org/abs/2508.02324
project: https://github.com/QwenLM/Qwen-Image
code: https://github.com/QwenLM/Qwen-Image
code_commit: 6b5e1f5cec987d404be5ac6657db3b9aacb56a89
tags:
  - Foundations
  - Image-Generation
  - Image-Editing
  - MMDiT
  - Flow-Matching
  - Text-Rendering
  - Qwen
created: 2026-11-06
updated: 2026-11-06
---

# Qwen-Image：复杂文字渲染与语义—外观双条件图像编辑

## 基本信息

- **作者 / 团队**：Qwen Team；核心贡献者包括 Chenfei Wu、Jiahao Li、Jingren Zhou、Junyang Lin、Kaiyuan Gao、Kun Yan、Shengming Yin、Shuai Bai、Xiao Xu、Yilei Chen、Yuxiang Chen、Zecheng Tang、Zekai Zhang、Zhengyi Wang 等。
- **论文**：[Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)，2025-08-04。
- **模型定位**：以 20B 双流 MMDiT 为生成主干、冻结 Qwen2.5-VL-7B 为条件编码器、Qwen-Image VAE 为视觉 tokenizer 的图像生成基础模型，统一支持文生图（T2I）和文本+图像到图像（TI2I）等任务。
- **代码状态**：官方仓库 [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) 已开源推理示例和版本说明，本文核对 commit `6b5e1f5cec987d404be5ac6657db3b9aacb56a89`；模型核心推理实现位于 Hugging Face Diffusers，本文核对 HEAD `d3d8af89e5b08a352cf47bba3a451c5fed9a49b1`。**训练代码与完整数据配方未公开。**
- **权重**：[Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)、[Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)。
- **与 2509 的关系**：论文详细解释的是 Qwen-Image 与初代统一编辑机制；Qwen-Image-Edit-2509 没有独立论文，只在相同 20B MMDiT 基础上进一步加入多图拼接训练、人物/商品/文字一致性增强及深度图、边缘图、关键点图等控制条件。

## 一句话结论

Qwen-Image 的真正贡献不是单纯把 MMDiT 放大到 20B，而是把「**文字可生成性**」拆成数据、VAE 重建与课程训练三个协同问题，并把图像编辑拆成「**Qwen2.5-VL 提供高层语义控制 + VAE latent 提供低层外观控制**」两条互补条件流；最终在 GenEval-RL 达 0.91、OneIG 中英文总体 0.539/0.548、ChineseWord 58.30、GEdit 中英文总体 7.56/7.52、ImgEdit 4.27，证明它既是强文生图模型，也是较完整的通用图像编辑底座。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

截至 2025 年，扩散 Transformer 已经可以生成观感很好的自然图像，但模型在两个更接近生产场景的维度仍不稳定：其一是复杂文字，尤其是中文、长段落、多个文本区域与带透视/材质的排版；其二是图像编辑的一致性，即只改用户指定内容，同时保留人物身份、商品外观、背景结构和局部纹理。

Qwen-Image 把这两个问题看作同一个基础模型问题：模型既要理解语言、场景和空间关系，也要在连续像素分布中生成可读字形与保持参考图细节。仅扩大常规图文对规模并不足够，因为自然数据中的长尾汉字、整段文字、PPT/海报布局极少；仅把参考图编码成高层语义也不够，因为会丢掉纹理和未编辑区域；仅把像素 latent 当条件则容易复制造型，却不一定理解“把杯子换成珐琅材质”之类的语义指令。

### 1.2 现有方法的瓶颈

文字渲染失败通常来自三个层面。第一，真实图文数据里的文字频率高度长尾，很多汉字几乎不出现；第二，通用 VAE 对小字和细线的压缩重建本身就会模糊，扩散主干不可能生成超过 tokenizer 上限的信息；第三，训练早期让模型同时学习自然图、排版、字形和高分辨率细节，会使优化目标过于混杂。

编辑一致性则存在「语义—像素」张力。Qwen2.5-VL 的视觉 token 能表达“图中是谁、有什么、用户要改什么”，但语义表征不保留每根发丝和局部纹理；VAE latent 保留外观，却不天然具备语言级指令理解。Qwen-Image 的方法不是二选一，而是让同一参考图同时进入这两条路径。

### 1.3 本文核心贡献

1. 构建 20B、60 层双流 MMDiT，以冻结的 Qwen2.5-VL-7B 最后一层 hidden state 作为条件，支持复杂语言和多模态输入。
2. 基于 Wan2.1-VAE 建立单编码器、图像/视频双解码器的 Qwen-Image VAE；冻结编码器，仅用文字密集数据微调图像解码器，显著提升小字重建。
3. 提出 MSRoPE，把图像按二维中心坐标编码、把文本位置放在二维对角线上，同时保留文本侧与 1D RoPE 的功能等价性。
4. 建立数十亿图文对的数据治理与文字合成体系：自然图约 55%、设计图约 27%、人物约 13%、程序化文字合成约 5%，并采用七阶段过滤和 256→640→1328 的训练课程。
5. 将 T2I、TI2I 与 I2I reconstruction 纳入统一训练；编辑时同时使用 Qwen2.5-VL 语义条件和 VAE 外观条件，并以 frame 轴扩展 MSRoPE 区分参考图与目标图。

---

## 2. 任务定义与输入输出

### 2.1 文生图与图像编辑

对文生图任务，输入是文本提示 $oldsymbol{S}=y$，Qwen2.5-VL 将模板化文本编码为条件序列 $h$；生成目标是图像 $x_0$ 的 VAE latent。对图像编辑任务，输入变为 $oldsymbol{S}=(x_{	ext{ref}},y)$：参考图既随文本一起进入 Qwen2.5-VL，得到语义条件，又经 VAE 得到干净外观 latent；目标仍是编辑后图像的加噪 latent。

可以把编辑表示成：

$$h=\phi(y,x_{\text{ref}}),\qquad z_{\text{ref}}=\mathcal{E}(x_{\text{ref}}),\qquad z_{\text{target}}=\mathcal{E}(x_{\text{target}})$$

其中 $\phi$ 是冻结的 Qwen2.5-VL，$\mathcal{E}$ 是 VAE encoder。MMDiT 同时接收 $h$、干净的 $z_{\text{ref}}$ 和被加噪的 $z_{\text{target},t}$，但损失只监督目标图对应的速度预测。

### 2.2 Flow Matching 目标

论文使用 Rectified Flow / Flow Matching。设干净图像 latent 为 $x_0$，标准高斯噪声为 $x_1\sim\mathcal{N}(0,\mathbf I)$，时间 $t\in[0,1]$ 从 logit-normal 分布采样：

$$x_t=tx_0+(1-t)x_1$$

这条路径从 $t=0$ 的噪声走向 $t=1$ 的数据，因此目标速度恒为：

$$v_t=\frac{dx_t}{dt}=x_0-x_1$$

模型以均方误差回归条件速度场：

$$\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{(x_0,h)\sim\mathcal D,x_1,t}\left\|v_\theta(x_t,t,h)-(x_0-x_1)\right\|_2^2$$

直觉上，MMDiT 不是逐步预测像素，而是在任意时间点回答“当前位置应该沿哪个方向移动，才能从噪声抵达条件图像分布”。

---

## 3. 核心方法

### 3.1 总体框架：Qwen2.5-VL + Qwen-Image VAE + 20B MMDiT

![](assets/Qwen-Image%20-%20文字渲染与双条件图像编辑/figures/02-architecture.png)

> **图 6：Qwen-Image 总体架构。** 左侧冻结的 Qwen2.5-VL 把系统提示与用户提示编码成文本/语义 token；VAE encoder 把图像压到 16 通道 latent，再以 $2\times2$ patch 打包成生成 token；右侧每个 MMDiT block 保留图像流和文本流各自的归一化、调制、QKV、输出投影与 MLP，但在 self-attention 中拼接为联合序列，使两种模态双向交互。来源：论文 Figure 6，PDF 第 7 页，[arXiv](https://arxiv.org/abs/2508.02324)。

Qwen2.5-VL-7B 的 ViT 为 32 层、patch size 14，语言部分 28 层、hidden size 3584。论文选择它而不是普通文本 LLM，有三个原因：视觉—语言空间已经对齐；语言能力没有明显退化；它原生接受图片，因而同一条件编码器可从 T2I 平滑扩展到 TI2I。Qwen-Image 取其 LLM 最后一层 hidden state，而不是只取 pooled embedding，这为长提示、文字内容和多图语义保留逐 token 条件。

MMDiT 为 60 层、24 头、每头 128 维，总 hidden size $24\times128=3072$，FFN 中间维 12288。图像 token 和文本 token 分别拥有独立的投影与 FFN，但 Q/K/V 在序列维拼接后共同做非因果 attention，因此它不是经典“图像查询文本”的单向 cross-attention，而是每层都进行完整文本—图像联合建模。

### 3.2 文字生成的第一道上限：专门微调图像 VAE decoder

![](assets/Qwen-Image%20-%20文字渲染与双条件图像编辑/figures/01-vae-text-reconstruction.png)

> **图 17：小字重建对比。** 从 FLUX、Hunyuan、SD3.5、Cosmos-CI 到 Wan2.1，多数 VAE 在放大后都把 “double-aspect” 的笔画粘连或重排；Qwen-Image-VAE 与原图最接近。论文用这张图强调：文字生成能力不能只归因于大 MMDiT，VAE 是否保存字形本身就是硬上限。来源：论文 Figure 17，PDF 第 26 页，[arXiv](https://arxiv.org/abs/2508.02324)。

Qwen-Image VAE 沿用 Wan2.1-VAE 的单编码器、双解码器设计：图像和视频共用 encoder，图像与视频分别使用专门 decoder。作者冻结 encoder，只对图像 decoder 训练，并加入 PDF、PPT、海报和程序化段落等文字密集图像。损失只保留 reconstruction 与 perceptual loss，并动态调整比例；当重建已经足够强时，判别器无法提供有效方向，因此去掉 adversarial loss。

量化上，在所有 VAE 均采用 $8\times8$ 空间压缩、16 通道 latent 的协议下，Qwen-Image-VAE 在 ImageNet-256 达 PSNR 33.42 / SSIM 0.9159，在文字集达 36.63 / 0.9839。其总 encoder/decoder 参数为 54M/73M，但处理单图时只激活约 19M/25M 的等效二维参数。

### 3.3 MSRoPE：让文本与二维图像共享同一套旋转位置编码

![](assets/Qwen-Image%20-%20文字渲染与双条件图像编辑/figures/03-msrope.png)

> **图 8：三种图文联合位置编码。** A 直接把文本接在展平图像之后，丢掉文本—图像的二维几何关系；B 把文本放到图像某一行，虽然支持分辨率缩放，却会与该行图像位置发生同构；C 的 MSRoPE 让图像坐标以中心为原点，并令文本位置沿二维对角线递增，使文本两个坐标相等，因而仍等价于 1D RoPE。来源：论文 Figure 8，PDF 第 9 页，[arXiv](https://arxiv.org/abs/2508.02324)。

Diffusers 配置中的 `axes_dims_rope=(16,56,56)` 把每个 128 维 attention head 分给 frame、height、width 三轴：16+56+56=128。实数通道两两组成复数旋转对，对应 8、28、28 组复频率。对图像，空间坐标围绕中心生成；对文本，同一个一维位置同时写入 h/w 坐标，所以旋转结果保持 1D RoPE 等价性，又避免把文本硬塞到图像某一行。

对于编辑，多出来的 frame 轴负责区分目标图和一张或多张参考图；同一个 $(h,w)$ 在不同 frame 上拥有不同相位，因此模型能知道“这是要生成的画布”还是“这是第几张条件图”。

### 3.4 双流 MMDiT block

每层先把 64 维 latent patch 投影到 3072 维，把 3584 维 Qwen 条件经 RMSNorm 和线性层投到 3072 维。时间步嵌入产生两套模态独立的 shift、scale、gate；图像流和文本流分别做无仿射 LayerNorm 和调制：

$$\operatorname{Mod}(x;t)=\operatorname{LN}(x)\odot(1+s_t)+b_t$$

两路各自产生 Q/K/V，并对 Q/K 做 head-wise RMSNorm 和 MSRoPE，然后按 `[text, image]` 拼接成长度 $T+I$ 的联合序列：

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{128}}\right)V$$

联合输出再拆回文本段和图像段，经各自的输出投影、gate 与残差；随后各走独立的 GELU-approximate FFN（3072→12288→3072）和第二次门控残差。60 层之后只保留图像流，经 AdaLN 和线性层投影回每 token 64 维，即一个 $2\times2\times16$ latent patch。

这套设计的关键不是“两个 Transformer”，而是**参数分流、注意力合流**：语言和图像可以使用不同的特征变换，但每层 attention 都允许它们互相读取。

### 3.5 图像编辑：语义条件与外观条件分工

![](assets/Qwen-Image%20-%20文字渲染与双条件图像编辑/figures/04-editing-dual-condition.png)

> **图 14：TI2I 编辑机制。** 左侧参考图和用户指令进入 Qwen2.5-VL，形成语义条件；参考图另经 VAE 编码形成干净外观 token，与目标图的加噪 latent token 沿序列维拼接。右侧在 h/w 之外加入 frame 轴，目标图和输入图被赋予不同 frame 位置。来源：论文 Figure 14，PDF 第 18 页，[arXiv](https://arxiv.org/abs/2508.02324)。

两条条件路径各自解决不同失效模式：

- **Qwen2.5-VL 语义路径**：负责识别人、物、文字、空间关系以及指令中的操作对象，提升 instruction following 和大幅结构变化时的身份语义保持。
- **VAE 外观路径**：参考图以干净 latent token 直接加入 MMDiT 图像序列，保存颜色、材质、纹理、布局和未编辑区域。

论文摘要还指出训练同时加入 I2I reconstruction，以对齐 Qwen2.5-VL 与 MMDiT 表征；但正文没有披露 I2I/TI2I/T2I 的比例、编辑数据规模、身份损失或重建损失权重，因此无法仅凭论文复现完整编辑 checkpoint。

### 3.6 Qwen-Image-Edit-2509 的增量

2509 并没有新论文。官方只明确披露三项变化：多图输入通过 image concatenation 继续训练，最佳范围为 1–3 张；单图人物、商品和文字一致性增强；原生接受深度、边缘、关键点和草图等控制图。模型配置仍是同一个 60 层、20B、3072 hidden 的 `QwenImageTransformer2DModel`，推理类由单图 `QwenImageEditPipeline` 改为多图 `QwenImageEditPlusPipeline`。

代码显示，多图不是先在像素空间拼成一张大图。每张图分别走 Qwen2.5-VL 语义路径和 VAE 外观路径，再按序列拼接为：

```text
[target noisy latent | source 1 clean latent | source 2 clean latent | ...]
```

MMDiT 对整个序列前向，但 scheduler 只保留并更新 target 段，所有 source latent 始终是固定条件。官方没有公开 2509 的训练数据规模、损失改动和消融；“支持 ControlNet 条件”也不能直接推导为经典的复制 U-Net/零卷积 ControlNet 架构，更稳妥的表述是“支持常见空间控制图作为多图条件”。

---

## 4. 数据与训练策略

### 4.1 数十亿图文对与四类分布

训练数据划分为 Nature 约 55%、Design 约 27%、People 约 13%、Synthetic Data 约 5%。Design 包含海报、UI、PPT、绘画和工艺品，直接为布局与文字服务；Synthetic 并不是由其他生成模型产生的图，而是程序化文字渲染，避免把生成模型的字形错误和伪影重新蒸馏回训练集。

七阶段过滤从损坏文件、低分辨率、去重和安全过滤开始，逐步加入清晰度、亮度、饱和度、熵、纹理、图文匹配、文字密度、小字、审美、水印、类别重平衡和高分辨率筛选。特别值得注意的是：高质量 caption 由 Qwen2.5-VL 生成，但原始 caption 不会全部丢弃，因为它包含细粒度 IP 名称和真实世界短文本知识。

### 4.2 三类文字合成

- **Pure Rendering**：把高质量语料段落在简单背景上排版，任何字符无法渲染则整段丢弃，直接补足长尾字形曝光。
- **Compositional Rendering**：把文字写到纸张、木牌等物理介质并合成进场景，再由 Qwen-VL 生成包含文字和上下文关系的 caption。
- **Complex Rendering**：程序化修改 PPT、UI 模板中的占位文字，同时保留布局、对齐、字体和颜色，用于多区域、多行、复杂排版。

这套数据设计说明，复杂文字不是靠提示工程“逼”出来的，而是从字形、场景和结构三层建立训练分布。

### 4.3 预训练课程与系统工程

分辨率按 256→640→1328 逐步提升，训练内容从无文字到有文字、从简单文字到段落，从海量数据到精炼数据，从类别不平衡到重采样平衡，并在后期加入程序化合成。

数据处理与训练采用 Producer–Consumer 解耦：Producer 完成过滤、Qwen2.5-VL 编码、VAE 编码和按分辨率分桶；GPU 密集 Consumer 只训练 MMDiT。MMDiT 使用 Megatron-LM、4-way tensor parallel 和 data parallel。论文给出一个重要负结果：activation checkpointing 只把单卡显存从 71GB 降到 63GB（11.3%），却把每步耗时从 2s 拉到 7.5s（3.75×），因此最终关闭 checkpointing，依赖 distributed optimizer。

### 4.4 后训练：SFT、Flow-DPO 与 Flow-GRPO

SFT 使用分层语义类别和人工精标图，针对真实性、细节和已知短板精修。随后先做大规模 DPO，再用小规模 GRPO。

Flow-DPO 比较 winning / losing 图像在 policy 与 reference 下的速度回归误差差：

$$\Delta_\theta=\|v_\theta(x_t^{win})-v_t^{win}\|_2^2-\|v_\theta(x_t^{lose})-v_t^{lose}\|_2^2$$

$$\mathcal L_{DPO}=-\mathbb E\log\sigma\left[-\beta(\Delta_\theta-\Delta_{ref})\right]$$

GRPO 则在线采样一组完整去噪轨迹，用组内标准化 reward 作为 advantage。因为确定性 ODE 不利于探索，论文把采样改写为带噪 SDE，再用 Euler–Maruyama 离散。这一后训练在 GenEval 上把 0.87 提升到 0.91，说明 RL 对计数、位置和属性绑定等可验证约束确有收益。

---

## 5. 实验结果

### 5.1 文生图与复杂提示遵循

| 基准 | Qwen-Image | 关键对比 |
|---|---:|---|
| DPG Overall | **88.32** | 略高于 Seedream 3.0 的 88.27；属性 92.02、关系 94.31 |
| GenEval SFT / RL | 0.87 / **0.91** | GPT Image 1 与 Seedream 3.0 均为 0.84；RL 首个超过 0.9 |
| OneIG-EN Overall | **0.539** | GPT Image 1 0.533；Alignment 0.882、Text 0.891 第一 |
| OneIG-ZH Overall | **0.548** | Seedream 3.0 0.528；Text 0.963 第一 |
| TIIF short / long | 86.14 / 86.83 | 低于 GPT Image 1 的 89.15 / 88.29，说明复杂推理指令仍非全面领先 |

这些结果说明 Qwen-Image 的优势主要集中在 prompt alignment 与文字，而不是所有“推理型生成”维度都第一。OneIG 的 diversity 只有 0.197，明显低于若干模型；这也是高引导、强对齐模型常见的保真度—多样性权衡。

### 5.2 中英文文字渲染

ChineseWord 将 8105 个常用规范汉字分三档。Qwen-Image 的 Level-1/2/3 为 97.29/40.53/6.48，整体 58.30；GPT Image 1 整体 36.14，Seedream 3.0 为 33.05。差距巨大，但 Level-3 仍只有 6.48，说明极低频汉字远未解决。

LongText-Bench 上，Qwen-Image 英文 0.943，略低于 GPT Image 1 的 0.956；中文 0.946，明显高于 Seedream 3.0 的 0.878。这里最有价值的不是“中文第一”本身，而是证明文字合成、VAE decoder 与高分辨率课程确实形成系统性增益。

### 5.3 图像编辑

GEdit-Bench 以 GPT-4.1 评估语义一致性、感知质量和总体分数。Qwen-Image 英文为 8.00/7.86/7.56，中文为 7.82/7.79/7.52，均略高于 GPT Image 1；FLUX.1 Kontext 中文总体仅 1.23，反映语言能力而非纯视觉能力也是编辑模型的重要组成。

ImgEdit 的九类任务总体 4.27，高于 GPT Image 1 的 4.20 和 FLUX Kontext 的 4.00。Qwen-Image 在 Extract 3.43、Replace 4.66、Remove 4.14 上第一；但 Add、Adjust、Background、Style、Hybrid、Action 多数仍由 GPT Image 1 领先。更准确的结论是：Qwen-Image 的总体平均最强，优势来自多类任务稳定，而不是所有类别都领先。

### 5.4 生成式理解、三维与失败边界

GSO 新视角合成上，Qwen-Image PSNR 15.11 / SSIM 0.884 / LPIPS 0.153，超过所有通用生成/编辑基线，接近专用 CRM 的 15.93 / 0.891 / 0.152，但并未超过 CRM。深度估计上，它在扩散/多任务模型中领先：KITTI AbsRel 0.078、NYUv2 0.055、ScanNet 0.047、DIODE 0.197；但仍落后 Metric3D v2 等专用判别模型。

论文没有提供标准化的失败案例表，不过从指标可以确认四类边界：极低频汉字仍失败；TIIF 的复杂推理不全面领先；高分辨率推理成本极高；生成式深度和新视角虽具迁移性，仍难替代强专用模型。

---

## 6. 与相关工作的关系

Qwen-Image 的 MMDiT 直接继承 SD3 / FLUX 的双流路线，但用冻结的 Qwen2.5-VL 替代纯文本编码器，把生成条件提升为可接受图像输入的 MLLM。与 BAGEL 的统一 decoder-only MoT 不同，Qwen-Image 仍是“条件编码器 + 独立生成主干”：理解模型被冻结，生成模型不输出文本回答，统一性弱于原生 omni 模型，但更容易保留 20B 生成容量。

编辑路线与 FLUX.1 Kontext 相似之处是把干净参考图 latent 作为 MMDiT 的上下文，而非传统 img2img 对参考图直接加噪再去噪。区别在于 Qwen-Image 还显式通过 Qwen2.5-VL 提取参考图语义；这种双条件结构解释了它在大动作、文字编辑与场景一致性上的优势。

MSRoPE 与 Qwen-VL 系列的 MRoPE 都属于多轴旋转位置编码，但目标不同：VLM 的 MRoPE 解决文本序列中图像/视频 token 的 t/h/w 编号；Qwen-Image 的 MSRoPE 服务 MMDiT 的图文联合生成与分辨率扩展，并用 `(frame,height,width)` 区分多张 latent 图。

---

## 7. 局限与批判性评价

### 7.1 论文透明度不足

论文只称训练数据包含数十亿图文对，没有给出精确样本总数、训练 token 数、训练 GPU 数、总算力、学习率、batch size、各阶段步数、T2I/TI2I/I2I 配比、编辑数据构造和 RL reward 细节。训练代码未开源，官方仓库主要是推理调用，因此“完整复现”实际上不可行。

### 7.2 20B MMDiT 成本极高

Qwen-Image 不是轻量模型。BF16 仅 20B MMDiT 权重理论上就约 40GB，还要加载 7B Qwen2.5-VL、VAE、激活和 CFG 双前向。官方基础 T2I 示例常显式使用 1328×1328、50 steps 和 true CFG 4.0；Diffusers 管线未显式传入宽高时通常采用 1024×1024。即便单次结果很强，延迟、显存和吞吐都限制了本地与在线规模化部署。

### 7.3 Benchmark 仍依赖模型裁判

GEdit、ImgEdit 大量使用 GPT-4.1 评价。虽然覆盖真实请求，但模型裁判可能偏好文字清晰、风格和 instruction-following，难以完全衡量逐像素保真、身份相似度和未编辑区域误差。论文也没有对“参考图输入 Qwen2.5-VL”“加入 VAE latent”“加入 I2I reconstruction”分别做量化消融，导致核心双条件设计主要靠经验陈述和最终性能支撑。

### 7.4 2509 的增量无法从论文验证

2509 所谓多图拼接训练、一致性增强和“原生 ControlNet”没有独立报告。代码能证明多张 source latent 被拼入图像序列，却不能证明人物一致性是靠哪类损失或数据获得；控制图被当作多图条件，与经典 ControlNet 残差旁路是否同构也没有证据。实践中应避免把营销命名当成已公开的架构事实。

---

## 8. 复现与实践建议

### 8.1 推理配置

基础 T2I 官方常用 1328²、50 steps、true CFG 4.0。编辑 2509 示例用 40 steps、true CFG 4.0、`guidance_scale=1.0`，输入为 1–3 张 PIL 图。若不用负面提示，true CFG 不会真正启用；官方通常传一个空格作为 negative prompt。编辑结果对 prompt rewrite 较敏感，官方明确建议重写编辑指令。

### 8.2 显存与速度

原始 BF16 整体权重超过 50GB，实际运行通常至少需要 80GB 级 GPU，或采用 CPU offload、模型量化、分组卸载。CFG 每步做条件/无条件两次 MMDiT 前向，成本近似翻倍。高分辨率 token 数按面积增长，MMDiT 联合注意力又近似二次增长，因此不要把 1328² 当作所有编辑任务的默认起点。

### 8.3 数据与微调

若目标是文字编辑，应优先保证 VAE/latent 能重建小字，并准备“文字内容 + 字体 + 颜色 + 材质 + 布局”的结构化训练对；若目标是身份保持，单靠 caption 不够，应混入 reconstruction、局部编辑和大姿态变化数据。LoRA 应优先加在 MMDiT 的 q/k/v/out 与调制 MLP，而不要先动冻结的 Qwen2.5-VL，除非任务语言或视觉域偏差很大。

---

## 9. 个人判断与后续问题

Qwen-Image 最值得复用的方法论是“先找生成上限所在层级”。中文文字差，不一定首先加大 DiT；可能是训练分布没覆盖字形，也可能是 VAE 先把字压糊。编辑不一致，也不一定靠更强 CFG；高层语义和低层外观本来就是两类条件，需要两条路径分别保存。

后续最值得追踪的三个问题是：第一，能否用更小的视觉语言编码器保留同等编辑理解，以降低每次采样的条件编码成本；第二，多图条件数量增大后，frame 轴和联合 attention 如何避免不同人物/商品特征串扰；第三，能否为编辑建立可解释、可量化的“未编辑区域误差 + 身份一致性 + 文字准确率”联合指标，减少对 VLM 裁判的依赖。

---

## 10. 代码级实现

### 10.1 仓库概览与论文模块映射

| 论文模块 | 开源实现 |
|---|---|
| Qwen2.5-VL 条件编码器 | Transformers `Qwen2_5_VLForConditionalGeneration` |
| Qwen-Image VAE | Diffusers `AutoencoderKLQwenImage` |
| 20B 双流 MMDiT | Diffusers `QwenImageTransformer2DModel` |
| Flow Matching 采样 | `FlowMatchEulerDiscreteScheduler` |
| 基础 T2I | `QwenImagePipeline` |
| 初代单图编辑 | `QwenImageEditPipeline` |
| 2509/2511 多图编辑 | `QwenImageEditPlusPipeline` |

官方 QwenLM 仓库没有 MMDiT 训练实现，主要保存模型介绍、prompt rewrite 和推理 demo；真正的张量组织、MSRoPE、VAE 和 Transformer block 位于 Diffusers。因此 `code_commit` 记录官方仓库，正文同时记录 Diffusers commit。

### 10.2 1328×1328 文生图全链路张量

设 batch=1，输出 1328×1328，文本有效长度为 $L\le512$：

| 阶段 | 张量形状 | 说明 |
|---|---:|---|
| Qwen2.5-VL 最后一层 | `[1,T,3584]` | 模板化 prompt 的全部 hidden states |
| 去模板并截断 | `[1,L,3584]` | 作为 MMDiT 文本流条件 |
| 初始目标噪声 | `[1,1,16,166,166]` | VAE 8×压缩；第二维为单帧 |
| $2\times2$ latent pack | `[1,6889,64]` | $83\times83$ token，每 token $2\times2\times16$ |
| `img_in` 后 | `[1,6889,3072]` | 64→3072 |
| `txt_in` 后 | `[1,L,3072]` | 3584→3072 |
| 24 头 Q/K/V | 图像 `[1,6889,24,128]`；文本 `[1,L,24,128]` | 分别投影、QK RMSNorm + MSRoPE |
| 联合 attention | `[1,L+6889,24,128]` | `[text,image]` 非因果双向 attention |
| 60 层输出 | 图像 `[1,6889,3072]` | 文本流不输出像素 |
| `proj_out` | `[1,6889,64]` | 回归速度/噪声 patch |
| unpack | `[1,16,1,166,166]` | 恢复 VAE latent |
| VAE decode | `[1,3,1,1328,1328]` | 取第 0 帧得到 RGB 图 |

这里 1328/8=166，再由 MMDiT 的 2×2 latent patch 得到 83×83=6889 个生成 token。它与 VLM 的“图像 patch token”概念不同：前者位于生成 VAE latent 空间，后者位于 Qwen2.5-VL 的视觉语义空间。

### 10.3 2509 单参考图编辑张量

代码默认将参考图分别缩放为约 384²（Qwen2.5-VL 条件）和约 1024²（VAE 条件）。若显式要求目标 1328²，单张正方形参考图形成：

| 路径 | 张量形状 |
|---|---:|
| Qwen2.5-VL 条件图 | 约 384×384；标准 processor 对齐后通常为 392×392 |
| 语义 hidden state | `[1,L,3584]`，前 64 个模板 token 被去掉 |
| 参考图 VAE latent | `[1,16,1,128,128]` |
| 参考图 packed latent | `[1,4096,64]` |
| 目标 noisy latent | `[1,6889,64]` |
| MMDiT 图像输入 | `[1,10985,64]` |
| `img_shapes` | `[[(1,83,83),(1,64,64)]]` |
| 保留的预测 | `[1,6889,64]`，只更新 target |

多图时继续在末尾追加每张 source packed latent。需要注意，默认输出分辨率并不等于参考图原始分辨率，而是按约 1024² 面积和最后一张参考图宽高比计算；要得到 1328² 必须显式传入 width/height。

### 10.4 逐模块四段拆解

#### 10.4.1 Qwen2.5-VL 条件编码器

- **输入输出**：T2I 输入系统模板+文本，Edit 输入系统模板+`Picture n` 视觉占位+多图+指令；输出最后一层 `[B,L,3584]` hidden state。
- **为什么这样设计**：比 T5/CLIP 文本编码器更能理解中文、复杂关系和参考图语义，且图文条件天然同空间。
- **关键超参**：T2I 默认最多截到 512 token；Edit 管线模板前缀固定去掉 64 token。
- **代码与论文差异**：论文只说明“取最后层 hidden state”；代码还暴露固定模板偏移、编辑图约 384² 的语义输入分辨率和多图 `Picture 1/2/...` 顺序。

#### 10.4.2 Qwen-Image VAE

- **输入输出**：RGB `[B,3,1,H,W]` → posterior 参数 `[B,32,1,H/8,W/8]` → 16 通道 latent；反向解码恢复 RGB。
- **为什么这样设计**：8×压缩控制生成 token 数；16 通道保留更多局部细节；3D causal VAE 为后续视频扩展保留接口。
- **关键超参**：`base_dim=96`、`dim_mult=[1,2,4,4]`、`z_dim=16`、空间压缩 8×；单图时间维保持1。
- **代码与论文差异**：论文强调单 encoder/双 decoder 与 decoder 微调；公开 Diffusers checkpoint 只呈现最终 image pipeline VAE 类，外部 pipeline 负责使用逐通道 latent mean/std 做规范化。

#### 10.4.3 双流 MMDiT

- **输入输出**：图像 `[B,I,64]`、文本 `[B,T,3584]` → 共同投影到 3072 → 60 层 → 图像 `[B,I,64]`。
- **为什么这样设计**：独立模态参数避免强行共享表示，联合 attention 保证文字内容、布局和图像结构每层都可互相读取。
- **关键超参**：60层、24头、head dim 128、FFN 12288、MSRoPE `(16,56,56)`、QK RMSNorm、其余 LayerNorm。
- **代码与论文差异**：论文架构图没有给出每层精确门控；代码显示 image/text 各有两组 shift/scale/gate、独立 QKV/输出/FFN，联合 attention 顺序为 `[text,image]`。

#### 10.4.4 Flow scheduler 与 CFG

- **输入输出**：每步输入 packed latent 和时间 $t$，MMDiT 输出同形状速度，Euler scheduler 更新 latent。
- **为什么这样设计**：Rectified Flow 路径简单、稳定；true CFG 用正/负条件预测差强化约束。
- **关键超参**：基础 T2I 通常 50 步、true CFG 4.0；2509 编辑通常 40 步。
- **代码与论文差异**：代码中的 true CFG 并非把正负样本合成双 batch，而是每步独立执行 cond/uncond 两次前向，再做 $v_u+s(v_c-v_u)$ 并按 token 范数重标定；没有 negative prompt 时不会启用 true CFG。

### 10.5 计算账与复现建议

1328² 对应 6889 个图像 token。联合 attention 的序列长度是 $L+6889$；2509 加一张 1024² 参考图后变为 $L+10985$，加三张则约 $L+19177$。联合注意力的计算量随序列长度近似平方增长；显存增长取决于 attention backend，朴素实现可能接近平方，而 Flash Attention / SDPA 可显著降低注意力激活显存。因此多图编辑仍远比单图 T2I 贵。MMDiT 20B BF16 约 40GB，Qwen2.5-VL-7B 另需约 14GB，再加 VAE、KV/激活、CFG 双前向，80GB 单卡也只算合理起点。

如果目标是“理解原理”，无需下载约 58GB 完整权重，阅读 config、Diffusers pipeline 和 transformer/VAE 实现即可；如果目标是 LoRA，建议 1024² 以下、小 batch、gradient checkpointing/CPU offload，并接受论文已证明 checkpointing 会显著拖慢全量训练。若做生产推理，应优先量化 MMDiT、缓存 Qwen 条件、减少 steps，并根据内容复杂度动态选择分辨率。

---

## 11. 材料来源

- 论文：[Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- 官方代码：[QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image)，commit `6b5e1f5cec987d404be5ac6657db3b9aacb56a89`
- 模型权重：[Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)、[Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- 核心推理实现：[Hugging Face Diffusers](https://github.com/huggingface/diffusers)，核对 commit `d3d8af89e5b08a352cf47bba3a451c5fed9a49b1`
- 初代编辑说明：[Qwen-Image-Edit 官方博客](https://qwenlm.github.io/blog/qwen-image-edit)
- 2509 增量说明：[Qwen-Image-Edit-2509 Introduction](https://github.com/QwenLM/Qwen-Image/blob/main/Qwen-Image-Edit-2509.md)
- 本地 PDF、关键图、图号、页码与获取日期：见 [sources.md](assets/Qwen-Image%20-%20文字渲染与双条件图像编辑/sources.md)。
