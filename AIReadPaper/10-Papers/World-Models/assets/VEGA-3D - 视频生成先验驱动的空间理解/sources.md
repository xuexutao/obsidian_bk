# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v3，含 Supplementary） | https://arxiv.org/pdf/2603.19235 | 全文，32 页 | 2026-10-16 | 精读主来源 |
| `figures/01-paradigm.png` | 范式对比原图 | arXiv TeX 源包 `figs/intro.pdf` | Figure 1，p.1 | 2026-10-16 | 对比显式 3D、几何监督与生成先验 |
| `figures/02-method-pipeline.png` | 方法总览原图 | arXiv TeX 源包 `figs/pipeline.pdf` | Figure 4，p.7 | 2026-10-16 | 解释 Latent World Simulator 与门控融合 |
| `figures/03-noise-layer-ablation.png` | 消融原图 | arXiv TeX 源包 `figs/ablationv2.png` | Figure 6，p.14 | 2026-10-16 | 分析噪声步与 DiT 层选择 |
| `figures/04-failure-case.png` | 失败案例原图 | arXiv TeX 源包 `figs/ScanRefer_failure_case.pdf` | Figure A2，p.23 | 2026-10-16 | 讨论近邻同类实例消歧失败 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2603.19235
- 官方代码：https://github.com/H-EmbodVis/VEGA-3D
- 官方项目页：https://h-embodvis.github.io/VEGA-3D/
- 官方空间推理模型：https://huggingface.co/H-EmbodVis/VEGA-3D-Spatial-Reasoning

## 可用性说明

官方仓库已提供 3D 场景理解训练/评测代码、空间推理分支与 checkpoint，许可证为 Apache-2.0。默认环境为 Python 3.10、PyTorch 2.4.0+cu121、FlashAttention 2.7.4.post1；论文主实验使用 8×H100。Wan2.1、LLaVA-Video、SigLIP、ScanNet/Video-3D-LLM 数据结构等仍是大体量外部依赖。