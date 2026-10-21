# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2604.11757> | 全文与补充材料，27 页 | 2026-10-21 | 精读主来源；受控基线、设计轴、benchmark、消融与真机结果核对 |
| `figures/01-teaser.png` | 官方源码原图 | <https://arxiv.org/e-print/2604.11757> | Figure 1，p.2，`figues/teaser.pdf` | 2026-10-21 | 解释“去复杂化”的研究动机与三条受控设计轴 |
| `figures/02-method.png` | 官方源码原图 | <https://arxiv.org/e-print/2604.11757> | Figure 2，p.4，`figues/method.pdf` | 2026-10-21 | 说明 Qwen3-VL + MLP、最小处理和跨 benchmark 统一评测 |
| `figures/03-action-head-ablation.png` | 官方源码原图 | <https://arxiv.org/e-print/2604.11757> | Figure 3，p.6，`figues/model-compare.pdf` | 2026-10-21 | 对比 FAST、MLP、GR00T 与 π 动作头设计 |
| `figures/04-scaling-trends.png` | 官方源码原图 | <https://arxiv.org/e-print/2604.11757> | Figure 5，p.9，`figues/combined_side_by_side.pdf` | 2026-10-21 | 分析模型规模与总 batch size 的 generalist 训练趋势 |

## Additional official sources

- arXiv 页面：<https://arxiv.org/abs/2604.11757>
- 项目页：<https://starvla.github.io>
- 代码仓库：<https://github.com/starVLA/starVLA>

## Notes

- 四张图片均从官方 arXiv LaTeX 源码包提取，使用 `research-field-survey/scripts/extract_source_figures.py` 转为限宽 2400 px 的 PNG；已逐张人工预览，图中文字、坐标轴与边界完整。
- 论文 PDF 已包含 supplementary material，因此未另造 `supplementary.pdf`。
- 论文没有独立的 limitations/future-work 章节；笔记把作者没有验证的边界与从实验设置可直接推出的限制单独标明，并避免将其伪装成作者原话。
