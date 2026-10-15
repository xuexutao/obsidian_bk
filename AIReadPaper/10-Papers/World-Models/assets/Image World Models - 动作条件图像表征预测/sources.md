# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2403.00504 | 全文，共 23 页 | 2026-10-15 | 精读主来源 |
| `sources/arxiv-source.tar.gz` | arXiv 官方源码包 | https://arxiv.org/e-print/2403.00504 | 全部 TeX 与原始图 | 2026-10-15 | 公式、表格、原图溯源 |
| `sources/arxiv-source/` | 官方源码展开目录 | https://arxiv.org/e-print/2403.00504 | `paper.tex`、`images/` | 2026-10-15 | 核对正文、附录与实验细节 |
| `figures/01-method-pipeline.png` | 官方原图渲染 | https://arxiv.org/e-print/2403.00504 | Figure S1，论文第 13 页；源码 `images/figure_1_color.pdf` | 2026-10-15 | 解释 IWM 完整 pipeline |
| `figures/02-world-model-taxonomy.png` | 官方原图渲染 | https://arxiv.org/e-print/2403.00504 | Figure 2，论文第 2 页；源码 `images/all_methods.pdf` | 2026-10-15 | 比较生成式、联合嵌入与 JEPA 世界模型 |
| `figures/03-finetuning-efficiency.png` | 官方原图渲染 | https://arxiv.org/e-print/2403.00504 | Figure 3，论文第 6 页；源码 `images/efficiency.pdf` | 2026-10-15 | 预测器微调效率结果 |
| `figures/04-equivariance-abstraction.png` | 官方原图渲染 | https://arxiv.org/e-print/2403.00504 | Figure 4，论文第 9 页；源码 `images/modularity.pdf` | 2026-10-15 | 等变性与表示抽象度分析 |

## 在线核验

- arXiv 论文页：https://arxiv.org/abs/2403.00504
- arXiv HTML（用于交叉核对附录结构）：https://arxiv.org/html/2403.00504v1
- 可核验出版信息：arXiv v1，2024-03-01；本笔记不将其写成已正式会议录用论文。
- 官方代码：论文、arXiv 页面及公开检索未给出可核验的官方 IWM 代码仓库；`code` 保留为 `null`。

> 图片均直接取自 arXiv 官方源码中的完整 PDF 图件，并以 300 DPI 渲染；已逐张目视核验，不是从论文 PDF 抽出的零散位图。