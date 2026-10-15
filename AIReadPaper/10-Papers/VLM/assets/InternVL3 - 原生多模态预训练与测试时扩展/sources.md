# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2504.10479 | 全文 | 2026-10-15 | 精读主来源 |
| `figures/01-overall-performance.png` | 核心结果总览 | https://arxiv.org/e-print/2504.10479 | Figure 1, p.2；源码 `figure/InternVL3_f1.pdf` | 2026-10-15 | 对比 InternVL2.5、开源与闭源 MLLM |
| `figures/02-architecture.png` | 架构与动态高分辨率示意 | https://arxiv.org/e-print/2504.10479 | arXiv 源码中的未编号完整架构资产 `figure/overall_architecture.pdf`；渲染版未单列页码 | 2026-10-15 | 解释 ViT–MLP–LLM、pixel unshuffle 与动态切图 |
| `figures/03-native-pretraining-ablation.png` | 原生多模态预训练消融 | https://arxiv.org/e-print/2504.10479 | Figure 3, p.18；源码 `figure/performance_comparison_bar1.pdf` | 2026-10-15 | 比较有无 native multimodal pre-training 及 SFT |

## 补充链接

- arXiv 摘要页：https://arxiv.org/abs/2504.10479
- 官方代码仓库：https://github.com/OpenGVLab/InternVL
- 三张图片均从 arXiv 官方 LaTeX 源码的完整矢量资产导出；架构资产在该版本源码中存在，但未在渲染版正文中以编号 Figure 单列，故如实标注为未编号。
