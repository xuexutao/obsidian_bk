# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2403.01823> | 全文，23 页 | 2026-10-21 | 精读主来源 |
| `figures/01-overview.png` | 官方源码图 | <https://arxiv.org/abs/2403.01823> | Figure 1，p.1；源码 `figures/teaser_v3.png` | 2026-10-21 | 语言动作层级、在线干预与多任务 |
| `figures/02-method.png` | 官方源码图 | <https://arxiv.org/abs/2403.01823> | Figure 2，p.4；源码 `figures/method_v3.png` | 2026-10-21 | motion/action 双查询与纠正训练 |
| `figures/03-multitask-results.png` | 官方源码图 | <https://arxiv.org/abs/2403.01823> | Figure 3，p.6；源码 `figures/phase4_results_v7_err.png` | 2026-10-21 | Diverse+Kitchen 八任务主结果 |
| `figures/04-intervention-results.png` | 官方源码图 | <https://arxiv.org/abs/2403.01823> | Figure 6，p.9；源码 `figures/phase4_intervention_results_v9_err.png` | 2026-10-21 | motion 纠正、IWR 与人工干预对比 |

## 提取与核验记录

- 使用 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv 官方源码提取，最大宽度 2400 px。
- 四张图逐张预览，任务标签、柱状数值、误差棒、动作箭头和边界完整。
- MSE、训练采样比例、纠正数据权重与新对象结果均从 PDF 正文/附录表格核对，不依赖柱状图估读。
