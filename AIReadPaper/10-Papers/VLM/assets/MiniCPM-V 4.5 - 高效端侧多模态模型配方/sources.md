# Sources

## 官方入口

- arXiv 页面：https://arxiv.org/abs/2509.18154
- 官方 PDF（v1，2025-09-16）：https://arxiv.org/pdf/2509.18154
- arXiv TeX source：https://arxiv.org/e-print/2509.18154
- 官方项目与代码：https://github.com/OpenBMB/MiniCPM-V
- arXiv DOI：https://doi.org/10.48550/arXiv.2509.18154
- 获取日期：2026-10-15

## 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2509.18154 | v1 全文，25 页 | 2026-10-15 | 精读主来源、公式/结果/消融复核 |
| `figures/01-architecture.png` | 原文架构图 | arXiv TeX source 中 `tex/figs/4dot5-model-v4.pdf`，300 DPI 渲染 | Figure 1 | 2026-10-15 | 3D-Resampler、图像切片、视频打包及双推理模式 |
| `figures/02-document-ocr-unified.png` | 原文方法图 | arXiv TeX source 中 `tex/figs/ocr.png` | Figure 2 | 2026-10-15 | 低/中/高破坏统一 OCR 与文档知识学习 |
| `figures/03-probability-reward-ablation.png` | 原文消融子图 | arXiv TeX source 中 `tex/figs/vr.pdf`，300 DPI 渲染 | Figure 4(a) | 2026-10-15 | VR 与 VR+PR 的 OpenCompass 曲线 |
| `figures/04-response-length-ablation.png` | 原文消融子图 | arXiv TeX source 中 `tex/figs/vr_length.pdf`，300 DPI 渲染 | Figure 4(b) | 2026-10-15 | 概率奖励下的响应长度稳定性 |

所有图片均来自官方 arXiv source，并已逐张本地预览验证；Figure 4 的子图按官方 source 原样分别保存，未重绘或拼接。TeX source 仅作为取材来源，未在最终附件目录重复保留。

## 版本与口径说明

- 摘要称模型为 8B，效率表按 8.7B 统计；笔记保留两种口径并明确差异。
- 论文宣称端侧友好，但公开定量效率实验是在 8×A100 上完成；没有手机 SoC、CPU/NPU、量化、功耗或设备端延迟结果，故不得把“端侧”理解为已被真实移动设备 benchmark 证明。