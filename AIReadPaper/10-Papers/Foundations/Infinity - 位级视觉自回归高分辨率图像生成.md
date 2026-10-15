---
type: paper-note
status: complete
domain: Foundations
paper: "Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis"
year: 2024
arxiv: "2412.04431"
doi: "10.48550/arXiv.2412.04431"
source: "https://arxiv.org/abs/2412.04431"
project: null
code: "https://github.com/FoundationVision/Infinity"
tags:
  - Image-Generation
  - Visual-Autoregressive
  - Bitwise-Tokenizer
  - Next-Scale-Prediction
  - Self-Correction
created: 2026-10-15
updated: 2026-10-15
---

# Infinity：位级视觉自回归高分辨率图像生成

## 基本信息

- **论文**：Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis
- **作者**：Jian Han、Jinlai Liu、Yi Jiang、Bin Yan、Yuqi Zhang、Zehuan Yuan、Bingyue Peng、Xiaobing Liu
- **机构**：字节跳动
- **会议**：CVPR 2025
- **代码**：https://github.com/FoundationVision/Infinity

## 一句话结论

Infinity 沿用 VAR 的 coarse-to-fine next-scale prediction，却把视觉 token 从“有限码本索引”改成“$d$位符号向量”，用 IVC 将$2^d$类 softmax 降为$d$个二分类，再用 BSC 模拟历史 bit 错误并重算后续残差标签；这三件事共同解决高分辨率视觉 AR 的表示容量、分类头爆炸和 teacher-forcing 误差传播，使 2B 模型以 13 个尺度在论文设置下 0.8 秒生成 1024²图像，但“无限词表”是理论容量表述，主 T2I 并未直接在$2^{64}$显式类别空间上训练。

---

## 1. 研究背景与问题定义

### 1.1 从 VAR 的成功到新瓶颈

[[VAR - 下一尺度自回归图像生成]]把 raster scan 改为 next-scale prediction，串行步数从像素/token 数降到尺度数。Infinity 认为下一瓶颈已经不在 AR 顺序，而在离散表示：有限 VQ 词表限制细节；整段 bit 映射成单一整数索引后，一个符号位轻微变化会让类别编号完全改变；若直接增大词表，分类头参数随$2^d$指数爆炸；teacher forcing 又让早期尺度错误在推理时逐级放大。

![](assets/Infinity%20-%20位级视觉自回归高分辨率图像生成/figures/01-teaser.png)

> 图 1：Infinity-2B 的 1024 级多宽高比生成，展示文字、空间关系、复杂主体和风格。来源：论文 Figure 1，p.1，https://arxiv.org/abs/2412.04431

### 1.2 三个核心问题

1. 如何把离散 tokenizer 词表扩到极大而不枚举码本？
2. 如何预测极大词表而不建立$2^d$类 softmax？
3. 如何让 next-scale 模型在早期尺度预测错时学会后续修复？

Infinity 对应给出 BSQ/LFQ、Infinite-Vocabulary Classifier 和 Bitwise Self-Correction。

---

## 2. 任务定义与总体分解

图像编码为$K$个多尺度残差 bit maps$\mathbf R_1,\ldots,\mathbf R_K$，文本表示为$\Psi(t)$。概率分解保持 VAR 形式：

$$p(\mathbf R_1,\ldots,\mathbf R_K)=\prod_{k=1}^{K}p(\mathbf R_k\mid\mathbf R_{<k},\Psi(t))$$

尺度间自回归、尺度内并行。生成 1024×1024 时使用$K=13$；1:1 的 latent 尺度从$(1,1)$递增到$(64,64)$，还预定义 1:3 到 3:1 共 15 种宽高比 schedule。

---

## 3. 核心方法

### 3.1 Bitwise Multi-Scale Tokenizer

编码器输出连续特征$\mathbf F\in\mathbb R^{h\times w\times d}$。多尺度残差逐级量化，累计特征为：

$$\mathbf F_k=\sum_{i=1}^{k}\operatorname{up}(\mathbf R_i,(h,w))$$

第$k$尺度输入由累计粗特征下采样得到：

$$\widetilde{\mathbf F}_{k-1}=\operatorname{down}(\mathbf F_{k-1},(h_k,w_k))$$

对连续向量$z_k\in\mathbb R^d$，LFQ/BSQ 只取各维符号：

$$q_k=\mathcal Q(z_k)=\begin{cases}\operatorname{sign}(z_k),&\mathrm{LFQ}\\\frac{1}{\sqrt d}\operatorname{sign}(z_k/\|z_k\|),&\mathrm{BSQ}\end{cases}$$

每维只有$\pm1$，理论组合数为：

$$V_d=2^d$$

$d=32$即$2^{32}$，$d=64$即$2^{64}$。它不是存储一个如此巨大的显式码本，而是用$d$个符号维隐式表示组合空间。

BSQ 用$O(d)$熵近似替代 LFQ 的$O(2^d)$码本相似度。训练显存实验中，LFQ 在$d=20$已 OOM，而 BSQ 从$d=16$到 64 始终约 32.4GB，因此主方法采用 BSQ。

### 3.2 Infinite-Vocabulary Classifier

传统分类器把 bit 向量映射成整数：

$$y_k(m,n)=\sum_{p=0}^{d-1}\mathbbm1[\mathbf R_k(m,n,p)>0]2^p$$

然后输出$V_d=2^d$类 logits。若$d=32,h=2048$，输出层约需 8.8T 参数。IVC 改为并行预测每个 bit 的正负，用$d$个二分类器把复杂度从$O(2^d)$降到$O(d)$。

![](assets/Infinity%20-%20位级视觉自回归高分辨率图像生成/figures/02-ivc.png)

> 图 2：连续特征轻微越过零点时，整数 index 可能从 9 跳到 1，监督看起来“完全不同”；按位监督只改变对应 bit，并把指数级分类矩阵改成线性维度输出。来源：论文 Figure 2，p.2。

在$V_d=2^{16}$实验中，传统头 124M 参数/2GB，IVC 仅 0.65M/10MB；同时 FID 4.49→3.83，ImageReward 0.79→0.91。它不只省资源，还提供更平滑的局部监督。

### 3.3 Bitwise Self-Correction

训练时随机翻转第$k$尺度的一部分 bit：

$$\mathbf R_k^{\mathrm{flip}}=\operatorname{RandomFlip}(\mathbf R_k,p)$$

关键不是单纯加噪，而是用污染后的历史重新构造累计特征，并重新量化下一尺度真实残差：

$$\mathbf R_{k+1}=\mathcal Q\left(\operatorname{down}(\mathbf F-\mathbf F_k^{\mathrm{flip}},(h_{k+1},w_{k+1}))\right)$$

这让标签本身变成“在此前预测已经出错时，下一尺度应该补什么”。随机翻转而不重算后续标签几乎无效；完整 BSC 才是真正的自纠错训练。

![](assets/Infinity%20-%20位级视觉自回归高分辨率图像生成/figures/03-framework.png)

> 图 3：Infinity 用 Causal VAR Transformer 做 next-scale prediction，IVC 输出 bit labels，BSC 在历史尺度注入错误并重新构造后续监督。来源：论文 Figure 3，p.4。

### 3.4 Transformer 与位置编码

每个 block 含 RoPE2d self-attention、文本 cross-attention 与 FFN；尺度间用 block-wise causal mask，推理用 KV cache。Infinity 用 RoPE2d+learnable scale embedding 替换 VAR 的可学习绝对位置编码，适配变长尺度序列与多宽高比。

### 3.5 采样

论文对 greedy、normal sampling、pyramid CFG、feature CFG 和 logits CFG 做对比。最终选择温度$\tau=1$、logits CFG=4，FID 2.82、ImageReward 0.962；HPSv2.1 的最高值 32.48 实际来自 pyramid CFG 1→3，而非最终配置。

BSC 让模型更能承受早期尺度的大 CFG，不必像 VAR 那样从 1 缓慢增加。

---

## 4. 数据与训练

主模型约 2B（架构表精确为 2.2B），使用 LAION、COYO、OpenImages 与内部高质量数据；总图文对数和内部占比未披露。训练分三阶段：

| 阶段 | 分辨率 | 迭代 | Batch | LR |
|---|---:|---:|---:|---:|
| 1 | 256 | 150K | 4096 | $6\times10^{-5}$ |
| 2 | 512 | 110K | 4096 | $6\times10^{-5}$ |
| 3 | 1024 | 60K | 2048 | $2\times10^{-5}$ |

词表/BSC 消融使用 5M 高质量数据，模型 scaling 使用 10M 数据，因此消融数字不能直接等同于主模型。

---

## 5. 实验结果

### 5.1 主文生图指标

| 模型 | 参数 | GenEval↑ | DPG↑ | ImageReward↑ | HPSv2.1↑ | 延迟↓ |
|---|---:|---:|---:|---:|---:|---:|
| SDXL | 2.6B | 0.55 | 74.65 | 0.600 | 30.06 | 2.7s |
| SD3-Medium | 2B | 0.62 | 84.08 | 0.871 | 30.91 | 2.1s |
| PixArt-Sigma | 0.63B | 0.55 | 80.54 | 0.872 | 31.47 | 1.1s |
| **Infinity** | **2B** | **0.73†** | 83.46 | **0.962** | **32.25** | **0.8s** |

†表示 prompt rewriting。Infinity 的 DPG 83.46 实际略低于 DALL-E 3 的 83.50，也低于 SD3-Medium 84.08；正文“超过 DALL-E 3”的说法与表格差 0.04。0.8 秒缺少 GPU、batch、精度、编译和端到端范围说明，因此只应视为论文内部口径。

### 5.2 Tokenizer scaling

| 词表容量 | ImageNet-256 rFID↓ | ImageNet-512 rFID↓ |
|---|---:|---:|
| $2^{16}$ | 1.22 | 0.31 |
| $2^{24}$ | 0.75 | 0.30 |
| $2^{32}$ | 0.61 | 0.23 |
| $2^{64}$ | **0.33** | **0.15** |
| SD VAE | 0.87 | — |

$2^{64}$结果是 tokenizer 重建实验，不等于主 T2I Transformer 显式预测$2^{64}$类别。生成侧主要验证到$2^{32}$。图 9 还显示：125M/361M 小模型更适合$2^{16}$，2.2B 训练约 40K steps 后$2^{32}$才反超，说明词表和模型必须共同扩展。

### 5.3 BSC 消融

| 方法 | FID↓ | ImageReward↑ | HPSv2.1↑ |
|---|---:|---:|---:|
| Baseline | 9.76 | 0.52 | 29.53 |
| + Random Flip | 9.69 | 0.52 | 29.20 |
| + BSC | **3.48** | **0.76** | **30.71** |

只加随机 bit 噪声无效，证明收益不来自普通正则化，而来自“污染历史后重算后续目标”。测试的翻转上限 10/20/30% 中，30% 最优（FID 3.33），但不能据此断言更高比例一定更差。

![](assets/Infinity%20-%20位级视觉自回归高分辨率图像生成/figures/04-self-correction.png)

> 图 12：普通 teacher forcing 在早期尺度出错后逐级放大；BSC 训练过“错误历史→修复残差”后能保持结构。来源：论文 Figure 12，p.15。

### 5.4 Scaling law

模型从 125M 扩到 4.7B，validation loss 随 FLOPs 持续下降；loss 与 GenEval、ImageReward、HPSv2 的 Pearson 相关系数分别为 -0.983、-0.981、-0.979。证据说明训练 loss 是下游质量的良好代理，但论文没有给出如 VAR 那样明确的幂律指数公式。

### 5.5 人类偏好

360 条 prompt、双盲两图比较中，Infinity 相对 SD3-Medium 的总体胜率 66%、审美胜率 71.1%；相对 HART 的总体/提示/审美胜率为 90.0/83.9/93.2%。内部人评流程有第三人裁决，但样本规模和模型采样配置仍限制可泛化性。

---

## 6. 与 VAR、LlamaGen 的关系

[[LlamaGen - 可扩展自回归图像生成]]坚持 raster next-token，用规模证明普通 Llama 能生成图像；[[VAR - 下一尺度自回归图像生成]]把因果轴改成分辨率尺度；Infinity 则进一步回答“尺度 token 应如何表示与纠错”。

技术演进可以概括为：

> LlamaGen：模型扩大 → VAR：AR 顺序重定义 → Infinity：离散词表与历史误差重定义。

Infinity 的长期价值在于把极大离散空间从“枚举类别”改成“结构化 bit 预测”，这对统一视觉 tokenizer、视频和多模态序列也有潜在启发。

---

## 7. 局限与批判性评价

### 7.1 “无限词表”是隐式组合空间

模型没有维护无限码本，只是$d$位符号允许$2^d$组合。其有效使用率、语义结构和生成模型可学习性仍受数据与模型容量限制。$2^{64}$主要是重建结果，不能宣传成 2B T2I 已完整驾驭$2^{64}$视觉词汇。

### 7.2 数据与速度不透明

主训练集规模、内部数据比例、延迟硬件和推理配置均缺失。0.8 秒和“最快”主张难以跨系统复核。

### 7.3 预定义宽高比 schedule

支持 15 种比例并不等于任意连续分辨率外推；schedule 外的尺寸、超高分辨率和极端比例没有系统验证。

### 7.4 评测口径有不一致

GenEval 0.73 依赖 prompt rewriting；DPG 的文字结论与表格冲突；主模型 2B/2.2B 两种口径并存。阅读时应优先引用表格并注明条件。

### 7.5 任务仍限于 T2I

论文讨论统一 tokenizer 潜力，却没有视频生成、图像编辑、多模态理解或统一理解—生成实验。安全、版权、偏差和人物生成风险也未系统讨论。

### 7.6 论文未单列 limitations

合理 future work 包括把 bitwise next-scale 扩到视频/统一多模态、学习动态尺度 schedule、减少固定 13 步与 full attention 成本、公开数据和端到端 latency 配置；以上为基于缺口的推断。

---

## 8. 复现与实践建议

优先从官方代码的 tokenizer 和 125M/361M 模型开始。复现顺序应是：

1. 验证 BSQ 在$d=16/32$下的 rFID 与显存；
2. 实现 IVC bit-wise cross entropy，核对输出维度$d$而非$2^d$；
3. 对照 random flip 与 BSC，确保翻转后重新量化后续残差；
4. 再实现 13-scale schedule、RoPE2d、scale embedding 和 logits CFG。

最容易误实现的是 BSC：如果只污染输入却仍用原始后续 token 作标签，得到的是论文中几乎无效的 Random Flip baseline，而不是自纠错。

---

## 9. 个人判断

**★★★★★。** VAR 改变了视觉 AR 的自回归轴，Infinity 则把表示、输出头和训练—推理错配一起补上，方法链完整、消融强，尤其 BSC 的对照很有说服力。需要降温的地方是“无限词表”和速度宣传：实际主模型的证据边界比标题窄。但从方法论看，它是高分辨率离散视觉生成中非常关键的一步。

---

## 10. 材料来源

- [arXiv](https://arxiv.org/abs/2412.04431)
- [CVPR 2025 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Han_Infinity_Scaling_Bitwise_AutoRegressive_Modeling_for_High-Resolution_Image_Synthesis_CVPR_2025_paper.html)
- [代码与模型](https://github.com/FoundationVision/Infinity)
- [本地来源清单](assets/Infinity%20-%20位级视觉自回归高分辨率图像生成/sources.md)
