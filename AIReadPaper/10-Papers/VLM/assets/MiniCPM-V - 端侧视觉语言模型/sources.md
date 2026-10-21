# Sources

## 官方入口

- arXiv 页面：https://arxiv.org/abs/2408.01800
- 官方 PDF：https://arxiv.org/pdf/2408.01800
- arXiv TeX source：https://arxiv.org/e-print/2408.01800
- 官方项目与代码：https://github.com/OpenBMB/MiniCPM-V
- arXiv DOI：https://doi.org/10.48550/arXiv.2408.01800
- 获取日期：2026-10-16

## 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | arXiv PDF | 全文 | 架构、训练、端侧部署、结果与消融复核 |
| `figures/01-architecture.png` | 完整原图，300 DPI 渲染 | arXiv TeX `figs/method.pdf` | Figure 3 | 自适应切片、视觉编码与共享压缩层 |
| `figures/02-rlaif-v.png` | 完整原图，300 DPI 渲染 | arXiv TeX `figs/rlaif.pdf` | Figure 4 | RLAIF-V 与 DPO 流程 |
| `figures/03-deployment-framework.png` | 完整原图，300 DPI 渲染 | arXiv TeX `figs/deployment_framework.pdf` | Figure 5 | CPU/GPU/NPU 与框架覆盖 |
| `figures/04-deployment-ablation.png` | 完整原图，300 DPI 渲染 | arXiv TeX `figs/end-side_abl.pdf` | Figure 6 | Xiaomi 14 Pro 部署优化消融 |

四张图均来自官方 arXiv source，未使用 PDF 内嵌碎片；已逐张本地读取，确认边界、文字、图例与子图完整。

## 版本与口径说明

- 论文是系列技术报告，重点实验对象为 MiniCPM-Llama3-V 2.5；笔记同时区分 V1.0、V2.0 与 2.5。
- `Object HalBench` 主表给的是 hallucination rate（越低越好），RLAIF-V 消融表使用 86.9/93.6 与 89.7/95.0 的正确率式口径；笔记没有把两种尺度混用。
- 与 MiniCPM-V 4.5 的关系依据本 Vault 已完成的 4.5 官方论文精读，并以内部 wikilink 连接。
