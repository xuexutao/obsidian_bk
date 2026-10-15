# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | <https://arxiv.org/pdf/2404.08471> | 全文（含附录） | 2026-09-03 | 精读主来源 |
| `figures/01-teaser.png` | Figure 1：V-JEPA 在 Kinetics-400 与 Something-Something-v2 上的 frozen-evaluation 性能散点图 | arXiv:2404.08471 | 第 1 页 | 2026-09-03 | 概述 / 主要定量结果 |
| `figures/02-jepa-architecture.png` | Figure 2：Joint-Embedding Predictive Architecture 通用框架（z → predictor + x-encoder → ŝ_y；y-encoder → s_y → D(ŝ_y, s_y)） | arXiv:2404.08471 | 第 3 页 | 2026-09-03 | 核心方法（JEPA 原理） |
| `figures/03-method-pipeline.png` | Figure 3：V-JEPA 训练流程（mask token → x-encoder → predictor → L₁ → y-encoder；EMA target encoder） | arXiv:2404.08471 | 第 4 页 | 2026-09-03 | 核心方法（视频版 pipeline） |
| <https://github.com/facebookresearch/jepa> | 官方代码仓库 | — | — | 2026-09-03 | 复现参考（与 I-JEPA/V-JEPA 同源） |
| <https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/> | 官方项目页 | — | — | 2026-09-03 | 论文/博客补充 |