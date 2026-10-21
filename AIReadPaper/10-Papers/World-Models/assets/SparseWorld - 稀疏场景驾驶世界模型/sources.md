# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v2） | https://arxiv.org/pdf/2605.24354 | 主文，8 页 | 2026-10-16 | 精读主来源 |
| `sources/supplementary.pdf` | 官方补充材料 | https://wryzju.github.io/SparseWorld/static/pdfs/SparseWorld_IROS2026_appendix_20260316_1110_fixed.pdf | Supplementary，7 页 | 2026-10-16 | 损失、训练配置、鲁棒性与额外可视化 |
| `figures/01-representation-comparison.png` | 表示对比原图 | arXiv TeX 源包 `figures/png/Sparseworld-teaser-0802-1223.pdf` | Figure 1，p.1 | 2026-10-16 | 对比像素、占用与实例预测 |
| `figures/02-framework.png` | 总体框架原图 | arXiv TeX 源包 `figures/png/Sparseworld-framework-0802-1925.pdf` | Figure 2，p.3 | 2026-10-16 | 解释预测—规划闭环 |
| `figures/03-sparse-dreamer.png` | 模块结构原图 | arXiv TeX 源包 `figures/png/Sparseworld-sparsedreamer.pdf` | Figure 3，p.4 | 2026-10-16 | 解释稀疏实例自回归与动作条件 |
| `figures/04-trajectory-selection.png` | 规划结果原图 | arXiv TeX 源包 `figures/png/Sparseworld-framework-planning_vis.pdf` | Figure 5，p.6 | 2026-10-16 | 解释候选轨迹安全选择 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2605.24354
- 官方项目页：https://wryzju.github.io/SparseWorld/

## 可用性说明

项目页提供论文、补充材料和可视化，但截至 2026-10-16 未列出代码仓库或模型权重。补充材料披露两阶段训练：nuScenes 上 100+20 epochs、Bench2Drive 上 10+4 epochs，主实验使用 8×RTX 4090D 24GB 训练、8×RTX 3090 24GB 闭环评测。