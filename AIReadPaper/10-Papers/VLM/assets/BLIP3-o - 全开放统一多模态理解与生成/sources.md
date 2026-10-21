# BLIP3-o 来源清单

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2505.09568 | 全文 | 2026-10-20 | 精读主来源 |
| `figures/01-architecture.png` | 总体架构图 | https://arxiv.org/e-print/2505.09568，源码 `fig/fig1.pdf` | Figure 1 | 2026-10-20 | 解释理解与生成如何共享 CLIP 语义空间 |
| `figures/02-design-choices.png` | 生成设计对照图 | https://arxiv.org/e-print/2505.09568，源码 `fig/figure3_arxiv.pdf` | Figure 3 | 2026-10-20 | 比较 CLIP+MSE、CLIP+Flow、VAE+Flow |
| `figures/03-design-ablation.png` | 设计消融曲线 | https://arxiv.org/e-print/2505.09568，源码 `fig/ablation.pdf` | Figure 4 | 2026-10-20 | 比较 GenEval、DPG 与 FID 收敛行为 |
| `figures/04-training-strategies.png` | 训练策略对照图 | https://arxiv.org/e-print/2505.09568，源码 `fig/figure5_arxiv.pdf` | Figure 5 | 2026-10-20 | 联合训练与顺序训练对比 |
| `figures/05-human-study.png` | 人类偏好评测图 | https://arxiv.org/e-print/2505.09568，源码 `fig/figure6_arxiv.pdf` | Figure 6 | 2026-10-20 | 核验 BLIP3-o 与 Janus-Pro 人类偏好差异 |

## 官方资源

- arXiv：<https://arxiv.org/abs/2505.09568>
- 官方代码：<https://github.com/JiuhaiChen/BLIP3o>
- 模型：<https://huggingface.co/BLIP3o/BLIP3o-Model>
- 预训练数据：<https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain>
- 指令微调数据：<https://huggingface.co/datasets/BLIP3o/BLIP3o-60k>

五张图均直接由 arXiv TeX 源码内的完整矢量 PDF 渲染，不是从论文页面截图；已逐张视觉检查，无截边、缺图例或相邻正文。
