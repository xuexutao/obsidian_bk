# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2604.05014> | 全文，25 页 | 2026-10-21 | 精读主来源；框架、接口、训练、benchmark 与效率数据核对 |
| `figures/01-unified-formulation.png` | 官方源码原图 | <https://arxiv.org/e-print/2604.05014> | Figure 1，p.3，`Figures/vla-form.pdf` | 2026-10-21 | 解释统一策略视角、输入输出与辅助目标 |
| `figures/02-action-head-frameworks.png` | 官方源码原图 | <https://arxiv.org/e-print/2604.05014> | Figure 2，p.6，`Figures/A2-frameworks.pdf` | 2026-10-21 | 对比 FAST、OFT、GR00T 与 π 四类动作头 |
| `figures/03-training-dataflow.png` | 官方源码原图 | <https://arxiv.org/e-print/2604.05014> | Figure 3，p.7，`Figures/starvla_dataflow.pdf` | 2026-10-21 | 说明 dataloader、统一字典、模型接口、训练和部署数据流 |
| `figures/04-multinode-scaling.png` | 官方源码原图 | <https://arxiv.org/e-print/2604.05014> | Figure 6，p.18，`Figures/efficiency/multi_node_scaling.pdf` | 2026-10-21 | 分析 8–256 GPU 弱扩展效率与系统成本 |

## Additional official sources

- arXiv 页面：<https://arxiv.org/abs/2604.05014>
- 项目页：<https://starvla.github.io>
- 代码仓库：<https://github.com/starVLA/starVLA>

## Notes

- 四张图片均优先从官方 arXiv LaTeX 源码包提取，使用 `research-field-survey/scripts/extract_source_figures.py` 转为限宽 2400 px 的 PNG；已逐张人工预览，未发现切边、缺图例或不可读问题。
- 本文是持续维护的技术报告/开源框架报告，而不是单一固定模型的封闭算法论文；报告明确说明 RL 微调仍在集成中。
- 正文对“已集成 benchmark 数量”存在版本性表述差异：摘要/引言列出五个主 benchmark，而能力比较表标为七个。笔记据此把它视作活跃演化代码库，而不将数字解释成稳定 API 承诺。
