# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2610.42146 | 全文，arXiv v1 | 2026-11-04 | 精读主来源 |
| `sources/arxiv-source.tar` | 官方 arXiv TeX 源码包 | https://arxiv.org/src/2610.42146 | 全文源码与原始图表 | 2026-11-04 | 核对公式、表格、图注与原始图片 |
| `figures/01-overview.png` | 总览图 | https://arxiv.org/abs/2610.42146 | Figure 1，PDF 第 2 页 | 2026-11-04 | 研究动机与两级规划总览 |
| `figures/02-architecture.png` | 方法架构图 | https://arxiv.org/abs/2610.42146 | Figure 2，PDF 第 4 页 | 2026-11-04 | 高层扩散世界模型与低层确定性世界模型 |
| `figures/03-neighboring-states.png` | 可达邻域可视化 | https://arxiv.org/abs/2610.42146 | Figure 4，PDF 第 7 页 | 2026-11-04 | 分析不同时间半径下的邻域生成 |
| `figures/04-scaling-results.png` | 复杂度扩展实验图 | https://arxiv.org/abs/2610.42146 | Figure 7，PDF 第 9 页 | 2026-11-04 | 迷宫规模与目标时程扩展分析 |
| `figures/05-search-ablation.png` | 规划预算与正则消融图 | https://arxiv.org/abs/2610.42146 | Figure 10，PDF 第 20 页 | 2026-11-04 | CEM 样本数与噪声正则分析 |
| `figures/06-state-mapping.png` | 隐式动作映射对比图 | https://arxiv.org/abs/2610.42146 | Figure 12，PDF 第 26 页 | 2026-11-04 | ID-WM 与等 frameskip 朴素世界模型对比 |

## 在线资料

| 标题 | URL | 获取日期 | 用途 |
|---|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2610.42146 | 2026-11-04 | 论文元信息、作者、arXiv ID、许可 |
| arXiv HTML | https://arxiv.org/html/2610.42146v1 | 2026-11-04 | 章节、图表锚点与可访问文本核对 |
| 官方项目页 | https://id-wm.github.io | 2026-11-04 | 演示视频、NeurIPS 2026 标注与代码发布状态 |

## 代码状态

截至 2026-11-04，官方项目页将代码标为 `Code (coming soon)`，未发现可核验的官方代码仓库、权重或配置文件。因此本笔记不记录虚构 commit，也不把同名第三方仓库视为官方实现。论文 Appendix 仅提供算法伪代码、结构超参数和计算资源信息；其中 HWM 基线的 DINO-WM 变体也因官方实现未开源而由作者自行复现。

## 补充材料状态

论文 Appendix 已合并在 26 页官方 PDF 中；arXiv 页面未提供独立 Supplementary Material 文件。