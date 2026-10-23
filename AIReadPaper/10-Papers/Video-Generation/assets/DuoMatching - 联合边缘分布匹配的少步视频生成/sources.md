# DuoMatching 精读素材溯源

论文：DuoMatching: Joint-Marginal Distribution Matching for Few-Step Video Generation
arXiv: https://arxiv.org/abs/2610.03543 （v1，2026-10-02）
项目页： https://johnzhan2023.github.io/DuoMatching/
获取日期：2026-10-23

## 官方 PDF

| 文件 | 来源 | 说明 |
|---|---|---|
| `sources/duomatching-2610.03543v1.pdf` | https://arxiv.org/pdf/2610.03543v1 | 官方 v1 全文，19 页，2026-10-23 下载 |

## Figures（均从官方 PDF 裁剪，extract_figure.py，ZOOM=2.4）

| 文件 | Figure | 页码 | 裁剪框 (x0,y0,x1,y1) | 用途 |
|---|---|---|---|---|
| `figures/01-teaser-motivation.png` | Figure 1 | p.2 | 0.105,0.085,0.895,0.505 | §1.1 动机：三方案对比 + Q-Align 曲线 |
| `figures/02-score-diff-heatmaps.png` | Figure 2 | p.2 | 0.474,0.678,0.930,0.842 | §1.2 joint vs marginal 信号空间结构 |
| `figures/03-overview-three-components.png` | Figure 3 | p.4 | 0.090,0.085,0.910,0.372 | §2.1 三组件总览 |
| `figures/04-qualitative-vs-cfpp.png` | Figure 4 | p.7 | 0.100,0.086,0.855,0.725 | §3.1 与 Causal Forcing++ 定性对比 |
| `figures/05-qualitative-vs-baselines.png` | Figure 5 | p.15 | 0.095,0.105,0.905,0.718 | §3.1 附录定性对比（Reward/One-Forcing/CausVid） |
| `figures/06-latentbridge-reconstruction.png` | Figure 6 | p.18 | 0.090,0.085,0.910,0.350 | §2.2 LatentBridge 重建质量 |
| `figures/07-latentbridge-l1-error.png` | Figure 7 | p.18 | 0.100,0.355,0.900,0.552 | §2.2 映射误差稳定性 |

裁剪逐张经视觉抽查：01/03/04/05/06/07 一次通过；02 因左侧混入正文与底部残片重裁两次后通过。

## 补充信息源

- mlllm.io 报道（2026-10，代码 Apache-2.0、两步 checkpoint 在 HuggingFace 的佐证）：https://mlllm.io/news/3484-bytedance-releases-duomatching-video-model-distillation-to-two-steps-with-an-image-teacher
- 论文正文与附录 A–G 均直接阅读自官方 PDF；VBench/人评/消融数字摘自正文表 1–6，理论证明摘自附录 C/D。
