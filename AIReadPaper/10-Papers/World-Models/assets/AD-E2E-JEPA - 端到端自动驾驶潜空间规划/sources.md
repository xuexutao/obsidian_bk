# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v1） | https://arxiv.org/pdf/2609.34085 | 全文（含附录） | 2026-10-15 | 精读主来源、公式与实验数据核对 |
| `sources/arxiv-source.tar.gz` | 官方 arXiv TeX 源包 | https://arxiv.org/e-print/2609.34085 | 论文源文件与 `images/*.pdf` | 2026-10-15 | 获取完整矢量原图并核对图号 |
| `figures/01-overview.png` | arXiv TeX 矢量源图渲染 | https://arxiv.org/e-print/2609.34085 | Figure 1，第 1 页；源文件 `images/overview.pdf` | 2026-10-15 | 方法动机、速度、误差、命中率与迁移收益总览 |
| `figures/02-method-pipeline.png` | arXiv TeX 矢量源图渲染 | https://arxiv.org/e-print/2609.34085 | Figure 2，第 4 页；源文件 `images/architecture.pdf` | 2026-10-15 | 世界模型、patch projector、轨迹库规划和迁移架构 |
| `figures/03-planning-comparison.png` | arXiv TeX 矢量源图渲染 | https://arxiv.org/e-print/2609.34085 | Figure 3，第 8 页；源文件 `images/visualization.pdf` | 2026-10-15 | LeWM、DINO-WM、JEPA-WM 与本文的规划定性比较 |
| `figures/04-imitation-transfer.png` | arXiv TeX 矢量源图渲染 | https://arxiv.org/e-print/2609.34085 | Figure 4，第 16 页；源文件 `images/imitation_learning_architecture.pdf` | 2026-10-15 | 自监督 projector 迁移到模仿学习的结构 |

## 官方页面与实现

| 材料 | 原始来源 | 获取日期 | 说明 |
|---|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2609.34085 | 2026-10-15 | 标题、作者、摘要、版本与提交日期 |
| 官方代码仓库 | https://github.com/HaoranZhuExplorer/AD-E2E-JEPA | 2026-10-15 | 仓库已建立，但 README 明确写明源代码仍在准备中，预计 2026-10-19 前公开；当前未见训练脚本、配置或 checkpoint |

## 版本与材料说明

- 本地 `paper.pdf` 保存 arXiv v1（2026-09-28）官方版本，正文与附录共 17 页；arXiv 页面未提供独立 supplementary PDF，实验细节已收入附录。
- 四张本地图均直接由官方 TeX 源包中的矢量 PDF 以 300 DPI 完整渲染，未使用 PDF 内嵌碎片；已逐张预览核对完整性、图例与文字清晰度。
- 复现状态以 2026-10-15 官方仓库页面为准：论文声明代码可用，但仓库当前只有 README 与图片，完整实现尚未发布，因此暂时无法核验训练命令、环境锁定文件和 checkpoint。
