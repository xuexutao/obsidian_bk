# 来源与图表核验记录

## 1. 原论文

- **标题**：*Auto-Encoding Variational Bayes*
- **作者**：Diederik P. Kingma, Max Welling
- **arXiv ID**：1312.6114
- **官方摘要页**：https://arxiv.org/abs/1312.6114
- **官方 PDF**：https://arxiv.org/pdf/1312.6114
- **核对版本**：arXiv:1312.6114v11，PDF 标注 2022-12-10；初稿年份为 2013
- **本地 PDF**：[[sources/paper.pdf]]
- **页数**：14
- **SHA-256**：`bdcb63b79ed88f64f8ee2d43a721930d01dde5441a4dba20ecaa2342b34e017c`

本地 `sources/paper.pdf` 直接取自 arXiv 官方 PDF。落盘后与抓取原件重新计算 SHA-256，二者哈希完全一致。

## 2. 关键图表来源

以下图片均直接从上述官方 PDF 以 PyMuPDF、300 DPI 裁取，没有使用网页转载图、重绘图或其他二手材料。

| 文件 | 原文对象 | PDF 页码 | 裁取区域（pt，左/上/右/下） | 完整性核验 |
|---|---|---:|---|---|
| `figures/figure-1-graphical-model.png` | Figure 1 | 2 | `100, 72, 512, 232` | 图模型、plate、参数标记及完整图注均保留，文字清晰 |
| `figures/algorithm-1-aevb.png` | Algorithm 1 | 4 | `100, 74, 512, 212` | 算法标题、初始化、repeat/until、参数更新和 return 均完整 |
| `figures/figure-2-lower-bound-comparison.png` | Figure 2 | 7 | `100, 72, 512, 334` | 9 个子图、坐标轴、图例及完整图注均保留，曲线清晰 |
| `figures/figure-3-marginal-likelihood.png` | Figure 3 | 8 | `100, 72, 512, 292` | 2 个子图、坐标轴、6 项图例及完整图注均保留，曲线清晰 |

## 3. 逐张视觉复核结果

1. **Figure 1**：无边缘截断；实线生成路径、虚线推断路径以及 $N$ plate 可辨认。
2. **Algorithm 1**：上下边框完整；$M=100$、$L=1$、梯度式与更新步骤可直接阅读。
3. **Figure 2**：MNIST/Frey Face 全部 latent-dimension 面板均在图内；红绿、实虚线含义可由图例确认；图注未截断。
4. **Figure 3**：$N_{train}=1000/50000$ 两个面板和右侧完整图例均保留；图注未截断。

## 4. 笔记引用关系

主笔记：[[../../VAE - 变分推断与重参数化生成模型.md]]

主笔记只引用上述 4 张从官方 PDF 裁取并核验的图表。Paper-Index 未在本次任务中修改。
