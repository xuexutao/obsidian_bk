# DDIM 来源与图表核验记录

## 1. 论文主来源

- **标题**：Denoising Diffusion Implicit Models
- **作者**：Jiaming Song, Chenlin Meng, Stefano Ermon
- **会议标记**：Published as a conference paper at ICLR 2021
- **arXiv ID**：2010.02502
- **核验版本**：arXiv:2010.02502v4，PDF 首页标记日期为 2022-10-05
- **摘要页**：https://arxiv.org/abs/2010.02502
- **官方 PDF**：https://arxiv.org/pdf/2010.02502
- **本地文件**：`DDIM_arXiv_2010.02502.pdf`
- **页数**：22 页
- **SHA-256**：`07cb8796d7c100bf7c02bc575f16aab72c32630a8f85abd54ad45dc4f63e80fb`
- **下载与核验日期**：2026-10-16

精读正文中的公式编号、实验数字、限制条件与图注均以该本地官方 PDF 为准；没有用后续博客或第三方实现替代原论文证据。

## 2. 关键原文定位

| 主题 | 原文位置 | 核验内容 |
|---|---|---|
| DDPM 条件边际与训练目标 | §2，Eq. (3)–(5)，第 2–3 页 | 论文记号 $\alpha_t$ 对应 DDPM 常见的累计 $\bar\alpha_t$；训练 loss 只采样单个 $(x_0,x_t)$ |
| non-Markovian joint | §3.1，Eq. (6)–(8)，第 3–4 页 | $q_\sigma(x_{t-1}\mid x_t,x_0)$ 的构造及相同 $q_\sigma(x_t\mid x_0)$ 边际 |
| 统一变分目标 | §3.2，Theorem 1；Appendix B，Eq. (29)–(35) | $J_\sigma=L_\gamma+C$；严格条件为 $\sigma>0$；$L_1$ 同最优解论证涉及各 $t$ 参数不共享 |
| 通用采样更新 | §4.1，Eq. (12)，第 5 页 | predicted $x_0$、direction、random noise 三项 |
| DDPM / DDIM 端点 | §4.1，第 5 页 | 特定 $\sigma_t$ 恢复 DDPM；$\sigma_t=0$ 给出确定性 DDIM，且不在 Theorem 1 的严格覆盖内 |
| 跳步生成 | §4.2；Appendix C.1，第 5、16–17 页 | 用时间子序列 $\tau$ 定义短生成轨迹，无需重新训练 |
| ODE 关系 | §4.3，Eq. (13)–(15)；Appendix B，Eq. (36)–(51) | DDIM 连续形式、与 VE probability flow ODE 的关系、有限步离散差异 |
| $\eta$ 参数化 | §5，Eq. (16)，第 6 页 | $\eta=0$ 为 DDIM，$\eta=1$ 为 DDPM 型方差 |
| FID 与速度 | §5.1，Table 1、Figure 3–4，第 7–8 页 | 步数、随机性、FID 和 wall-clock 权衡 |
| 一致性与插值 | §5.2–5.3，Figure 5–6，第 7–9 页 | 固定 $x_T$ 的跨步数语义一致与 slerp 插值 |
| 重建 | §5.4，Table 2，第 9 页 | 反向 ODE encoding / decoding 的 CIFAR-10 重建误差 |
| 数据集与 schedule | Appendix D.1–D.2，第 18 页 | CIFAR-10、CelebA、LSUN Bedroom/Church；linear / quadratic 子序列 |
| 原文局限与展望 | §7，第 9–10 页 | 非高斯过程、高阶 ODE 多步法和 implicit model 性质是未来方向 |

## 3. 原文图表裁剪清单

全部图片直接从 `DDIM_arXiv_2010.02502.pdf` 以 PyMuPDF 300 DPI 渲染裁剪，保留原图注，并逐张人工检查边缘、文字与图像内容完整性。

| 本地文件 | PDF 页码 | 原文对象 | 像素尺寸 | 用途与核验结论 |
|---|---:|---|---:|---|
| `fig1_graphical_models.png` | 2 | Figure 1 | 1760×459 | Markov diffusion 与 non-Markovian inference joint 对比；图与图注完整 |
| `fig2_accelerated_generation.png` | 5 | Figure 2 | 1510×496 | $\tau=[1,3]$ 跳过 $x_2$ 的 accelerated generation 图模型；图与图注完整 |
| `table1_fig3_fid_samples.png` | 7 | Table 1 + Figure 3 | 1718×1097 | CIFAR-10 / CelebA 的 FID、$\eta$、步数及定性样本；表格数字与图注完整 |
| `fig4_sampling_efficiency.png` | 8 | Figure 4 | 1676×497 | 50k 样本耗时随步数变化；双数据集曲线与图注完整 |
| `fig5_trajectory_consistency.png` | 8 | Figure 5 | 1676×1063 | 固定 $x_T$、不同采样步数的一致性；三组样本与图注完整 |
| `fig6_latent_interpolation.png` | 8 | Figure 6 | 1676×789 | 50 步 DDIM latent interpolation；三行插值与图注完整 |
| `table2_reconstruction_error.png` | 9 | Table 2 | 1510×313 | CIFAR-10 重建 MSE；$S=10$ 到 $1000$ 的全部数字与图注完整 |

## 4. 数字交叉核验

- Table 1：CIFAR-10 的 DDIM FID 在 $S=10,20,50,100,1000$ 时分别为 `13.36, 6.84, 4.67, 4.16, 4.04`。
- Table 1：CelebA 的 DDIM FID 在相同步数下分别为 `17.33, 13.73, 9.17, 6.53, 3.51`。
- Table 2：CIFAR-10 重建误差在 $S=10,20,50,100,200,500,1000$ 时分别为 `0.014, 0.0065, 0.0023, 0.0009, 0.0004, 0.0001, 0.0001`。
- Table 3：Bedroom 20 步 DDIM / DDPM FID 为 `8.89 / 22.77`；Church 20 步为 `12.47 / 23.37`。
- 正文 §5.1：与 1000 步模型相近的质量可在 20–100 步获得，对应约 10×–50× 加速。

## 5. 关于 inversion 表述的来源边界

本笔记对 inversion 的描述只把以下内容归于 DDIM 原论文：

1. §4.3 说明可反向模拟 Eq. (14) 的 ODE，从 $x_0$ 得到 $x_T$ encoding；
2. §5.4 在无条件 CIFAR-10 模型上进行相同步数的 encoding / decoding 重建；
3. Table 2 报告逐维 MSE，且脚注说明 $x_T$ 与 $x_0$ 同维，不讨论压缩。

Stable Diffusion、VAE latent、文本 prompt、classifier-free guidance、Null-text Inversion、Prompt-to-Prompt 等均不属于该论文内容；笔记中仅将它们作为后世用法边界提及，未倒推为原论文贡献。
