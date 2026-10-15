# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v1） | [arXiv 2211.10831](https://arxiv.org/abs/2211.10831)；[PDF](https://arxiv.org/pdf/2211.10831) | 全文，含附录 | 2026-10-15 | 精读主来源 |
| `sources/paper.txt` | PDF 文本抽取 | 由本地 `sources/paper.pdf` 使用 PyMuPDF 抽取 | 全文 | 2026-10-15 | 全文检索与公式、配置核对 |
| `figures/01-jepa-training-diagrams.png` | JEPA 训练架构图 | [arXiv 2211.10831](https://arxiv.org/abs/2211.10831) | Figure 1, p.2 | 2026-10-15 | 对比 VICReg 与 SimCLR 训练路径 |
| `figures/02-dataset-examples.png` | 数据集与噪声示例 | [arXiv 2211.10831](https://arxiv.org/abs/2211.10831) | Figure 2, p.3 | 2026-10-15 | 解释 changing/fixed、uniform/structured 干扰 |
| `figures/03-noise-performance.png` | 主实验结果图 | [arXiv 2211.10831](https://arxiv.org/abs/2211.10831) | Figure 3, p.4 | 2026-10-15 | 比较不同方法在四类噪声下的 RMSE |

## 其他官方材料

- 作者公开代码仓库：https://github.com/vladisai/JEPA_SSL_NeurIPS_2022
- arXiv 页面注明：`4 pages (3 figures) short paper for SSL Theory and Practice workshop at NeurIPS 2022`。本地 PDF 还包含参考文献、检查表和附录，共 13 个 PDF 页面。
- arXiv 页面给出的 DOI：https://doi.org/10.48550/arXiv.2211.10831

三张图片均直接从 arXiv 官方 PDF 的主文 Figure 1–3 所在页以 300 DPI 裁剪，保留完整图形与原始图注，并已逐张目视核验。
