# DDPM 精读来源与图表核验

## 1. 原论文一手来源

- arXiv 摘要页：https://arxiv.org/abs/2006.11239
- arXiv 官方 PDF：https://arxiv.org/pdf/2006.11239
- arXiv 官方 TeX 源包：https://arxiv.org/e-print/2006.11239
- NeurIPS 2020 论文页：https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html
- 作者官方代码：https://github.com/hojonathanho/diffusion
- 本地官方 PDF：`DDPM - Denoising Diffusion Probabilistic Models.pdf`
- 本地 TeX 源：`tex-source/main.tex`
- 本地抓取正文：`source.md`

## 2. 后续边界核对来源

这些资料只用于界定 DDPM 原论文之后的发展，不将后续结论倒灌为原论文贡献。

- DDIM（非马尔可夫前向过程与快速/确定性采样）：https://arxiv.org/abs/2010.02502
- Score SDE（连续时间统一视角、reverse-time SDE、probability-flow ODE）：https://arxiv.org/abs/2011.13456
- Improved DDPM（可学习方差、混合目标、重要性采样与更少采样步）：https://arxiv.org/abs/2102.09672
- Latent Diffusion（在压缩潜空间训练扩散模型）：https://arxiv.org/abs/2112.10752
- Classifier-Free Guidance（联合条件/无条件分数进行引导）：https://arxiv.org/abs/2207.12598
- EDM（把噪声参数化、预条件、采样器和时间离散化拆成设计空间）：https://arxiv.org/abs/2206.00364

## 3. 原文关键图表与逐张核验记录

| 本地文件 | 原始出处 | 提取方式 | 核验结果 |
|---|---|---|---|
| `figure-2-directed-graphical-model.png` | 原文 Figure 2；TeX 源 `images/pgm_diagram_xarrow_small.pdf` | 直接将 TeX 原始矢量 PDF 以 4 倍比例、白底渲染为 PNG | 已逐张目视核验：前向虚线箭头与反向实线箭头完整，`x_T`、`x_t`、`x_{t-1}`、`x_0` 标签清晰，无裁边 |
| `table-2-ablation.png` | 原文 Table 2；arXiv TeX `main.tex` 生成表格，官方 PDF 第 5 页 | 2026-10-16 由官方 arXiv 源码编译版 PDF 精确裁剪并以 1600 px 高清渲染；已替换原先混入相邻 Table 1 和正文的错误裁图 | 已逐张目视核验：表题、Objective/IS/FID、两组参数化及全部数值完整，无相邻栏内容、无截边 |
| `figure-6-progressive-generation.jpg` | 原文 Figure 6；TeX 源 `images/cifar10_eps-fixedlarge-mse_20_progressive.jpg` | 直接复制 TeX 原图 | 已逐张目视核验：20 行从噪声到 CIFAR-10 图像的逐步生成序列完整，横向阶段无缺列 |
| `figure-7-stochastic-decoding.jpg` | 原文 Figure 7；TeX 源 `images/stochastic_decoding.jpg` | 直接复制 TeX 原图 | 已逐张目视核验：共享 `x_1000/x_750/x_500/x_250` 的四组分叉采样均完整，底部标签可读 |
| `figure-8-interpolation.jpg` | 原文 Figure 8；TeX 源 `images/interp_with_diagram.jpg` | 直接复制 TeX 原图 | 已逐张目视核验：左侧插值示意、右侧三组 CelebA-HQ 插值和 `λ=0.1…0.9` 标注完整 |

## 4. 数字核对锚点

- 训练与采样算法：`tex-source/main.tex` 第 233–268 行。
- 简化噪声预测目标：第 287–299 行。
- `T=1000`、线性 `β` 日程与网络设计：第 301–306 行。
- CIFAR-10 主结果与消融：第 310–392 行。
- 渐进生成、随机解码与插值：第 410–539 行。
- 模型规模、TPU、训练速度、训练步数和超参数：第 646–664 行。

核验日期：2026-10-16。
