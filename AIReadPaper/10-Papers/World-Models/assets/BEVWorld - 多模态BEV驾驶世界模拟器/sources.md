# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v3） | https://arxiv.org/pdf/2407.05679 | 全文，18 页 | 2026-10-16 | 精读主来源 |
| `figures/01-overview.png` | 方法总览原图 | arXiv TeX 源包 `figs/overview-5.pdf` | Figure 1，p.4 | 2026-10-16 | 解释多模态 tokenizer 与 BEV 序列扩散 |
| `figures/02-ray-rendering.png` | 渲染机制原图 | arXiv TeX 源包 `figs/volume-render-2.pdf` | Figure 3，p.6 | 2026-10-16 | 解释相机/LiDAR 射线渲染 |
| `figures/03-future-predictions.png` | 定性结果原图 | arXiv TeX 源包 `figs/visual_results.pdf` | Figure 5，p.10 | 2026-10-16 | 检查 1 s/3 s 多模态预测 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2407.05679
- arXiv HTML（v3）：https://arxiv.org/html/2407.05679v3
- 作者仓库：https://github.com/zympsyche/BevWorld

## 可用性说明

截至 2026-10-16，作者仓库仅包含 README、示意图和演示材料，未提供训练/推理源码、配置或模型权重；论文正文也仍写明“Code will be released soon”。因此本文的复现成本只能依据论文披露的结构、超参数与硬件估算。