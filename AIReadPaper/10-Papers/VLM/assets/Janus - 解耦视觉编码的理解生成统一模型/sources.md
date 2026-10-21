# Sources

## 1. 官方入口

- Janus arXiv 页面：https://arxiv.org/abs/2410.13848
- Janus 官方 PDF：https://arxiv.org/pdf/2410.13848
- Janus arXiv DOI：https://doi.org/10.48550/arXiv.2410.13848
- Janus-Pro arXiv 页面：https://arxiv.org/abs/2501.17811
- Janus-Pro 官方 PDF：https://arxiv.org/pdf/2501.17811
- 官方项目与代码：https://github.com/deepseek-ai/Janus
- 获取日期：2026-10-16

## 2. 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | Janus 官方论文 PDF | arXiv:2410.13848 | 全文，24 页 | 主精读来源、公式、训练配置、结果与消融复核 |
| `sources/janus-pro.pdf` | Janus-Pro 官方论文 PDF | arXiv:2501.17811 | 全文，13 页 | 核对 Janus-Pro 的训练、数据和模型扩展 |
| `figures/01-overview.png` | 原文完整结果概览图 | Janus PDF 300 DPI 裁剪 | Figure 1，第 2 页 | 理解与生成总体能力 |
| `figures/02-architecture.png` | 原文完整架构图 | Janus PDF 300 DPI 裁剪 | Figure 2，第 4 页 | 双视觉入口、共享自回归 Transformer |
| `figures/03-training-stages.png` | 原文完整训练流程图 | Janus PDF 300 DPI 裁剪 | Figure 3，第 5 页 | 三阶段冻结/解冻策略 |

## 3. 核验说明

- 两份 PDF 均由 arXiv 官方 PDF 入口下载。
- 三张图片均从官方 Janus PDF 的整页渲染中按图注坐标裁剪，不使用 PDF 内嵌碎片。
- 三张图已于 2026-10-16 逐张用 `Read` 预览核验：图像主体、边缘与图注完整，文字清晰。
- 主笔记中的 Janus-Pro 数字以本地 `sources/janus-pro.pdf` 为准；Janus-Pro 并未改变 Janus 的核心架构，其提升主要来自训练策略、数据和模型规模。