# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2512.01801> | 全文，16 页 | 2026-10-21 | 精读主来源 |
| `figures/01-pipeline.png` | 官方源码原图 | <https://arxiv.org/e-print/2512.01801> | Figure 1，p.2，`images/gr_rl_main.pdf` | 2026-10-21 | 三阶段训练与系鞋带任务 |
| `figures/02-model-architecture.png` | 官方源码原图 | <https://arxiv.org/e-print/2512.01801> | Figure 2，p.3，`images/model_arch.pdf` | 2026-10-21 | VLA、动作 DiT 与分布式 critic |
| `figures/03-training-results.png` | 官方源码原图 | <https://arxiv.org/e-print/2512.01801> | Figure 5，p.8，`images/main_metric.pdf` | 2026-10-21 | 各训练阶段成功率与在线学习曲线 |
| `figures/04-stage-results.png` | 官方源码原图 | <https://arxiv.org/e-print/2512.01801> | Figure 6，p.9，`images/bar_stage_label.pdf` | 2026-10-21 | 分阶段失败归因 |
| `figures/05-progress-ablation.png` | 官方源码原图 | <https://arxiv.org/e-print/2512.01801> | Figure 7，p.9，`images/progress_ablation.pdf` | 2026-10-21 | 分布式与回归 critic 消融 |
| 项目页 | 官方项目页 | <https://seed.bytedance.com/gr_rl> | 论文首页 | 2026-10-21 | 视频与系统展示 |

## 提取与核验说明

- 五张矢量图均使用 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv 官方源码包提取并转为限宽 2400 px 的 PNG。
- 所有图片均已逐张目视检查：流程标签、数值、图例、阶段名称和边缘完整，无切边；正文全部引用。
- 论文文本由本地官方 PDF 逐页解析。论文未披露初始人类示范总量、完整在线交互总量、计算硬件与墙钟训练成本，笔记不作推测性补数。
