# Self-Forcing 来源清单

获取日期：2026-10-15

| 本地文件 | 官方 URL | Figure / Page | 用途 |
|---|---|---|---|
| `paper.pdf` | https://arxiv.org/pdf/2506.08009 | 全文，19 页 | 公式、训练细节、实验与局限主来源 |
| `../figures/01-training-paradigms.png` | https://arxiv.org/e-print/2506.08009 | Figure 1 / PDF p.2；源码 `figures/overview.pdf` | 对比 Teacher/Diffusion/Self Forcing |
| `../figures/02-attention-masks.png` | https://arxiv.org/e-print/2506.08009 | Figure 2 / PDF p.4；源码 `figures/attention.pdf` | 解释 self-rollout 与 KV cache |
| `../figures/03-rolling-kv-cache.png` | https://arxiv.org/e-print/2506.08009 | Figure 3 / PDF p.7；源码 `figures/rolling_kv.pdf` | 解释长视频外推复杂度 |
| `../figures/04-qualitative-comparison.png` | https://arxiv.org/e-print/2506.08009 | Figure 5 / PDF p.8；源码 `figures/Self-Forcing-Baseline-2-compressed.pdf` | 展示误差累积与定性对比 |
| `../figures/05-training-efficiency.png` | https://arxiv.org/e-print/2506.08009 | Figure 6 / PDF p.9；源码 `figures/training_efficiency_combined.pdf` | 训练耗时与收敛效率消融 |

补充官方材料：项目页 https://self-forcing.github.io/ ；代码 https://github.com/guandeh17/Self-Forcing 。图片均由 arXiv 官方 TeX 源包中的矢量原图转为 PNG，并已逐图验图。
