# Sources

## 官方入口

- arXiv 页面：https://arxiv.org/abs/2412.10302
- 官方 PDF（v1，2024-12-13）：https://arxiv.org/pdf/2412.10302
- arXiv TeX source：https://arxiv.org/e-print/2412.10302
- 官方代码与模型：https://github.com/deepseek-ai/DeepSeek-VL2
- arXiv DOI：https://doi.org/10.48550/arXiv.2412.10302
- 获取日期：2026-10-15

## 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2412.10302 | v1 全文，28 页 | 2026-10-15 | 精读主来源、架构/训练表和结果复核 |
| `figures/01-teaser.png` | 原文结果概览图 | arXiv TeX source 中 `figures/vl2_teaser.pdf`，300 DPI 渲染 | Figure 1 | 2026-10-15 | 激活参数量与平均性能关系 |
| `figures/02-architecture.png` | 原文架构图 | arXiv TeX source 中 `figures/main_framework_v2.pdf`，300 DPI 渲染 | Figure 2 | 2026-10-15 | Vision Encoder、VL Adaptor、DeepSeekMoE 数据流 |
| `figures/03-dynamic-tiling.png` | 原文方法图 | arXiv TeX source 中 `figures/cropping_strategy_v2.pdf`，300 DPI 渲染 | Figure 3 | 2026-10-15 | 全局缩略图、局部 tile、换行和分隔 token |

所有图片均直接来自官方 arXiv source，并已逐张本地预览检查；未把 PDF 内嵌碎片误作完整方法图。TeX source 仅作为取材来源，未在最终附件目录重复保留。

## 说明

论文没有正式的模块消融表，因此本笔记明确区分“主结果”“经验观察”和“未披露的消融”。MLA 的完整公式应参考 DeepSeek-V2；本笔记只记录 DeepSeek-VL2 论文明确披露的 latent rank、使用范围和 KV-cache 压缩作用。