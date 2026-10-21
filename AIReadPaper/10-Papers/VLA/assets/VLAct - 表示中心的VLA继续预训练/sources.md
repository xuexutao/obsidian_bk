# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2608.27550> | 全文与附录，37 页 | 2026-10-21 | 精读主来源；继续预训练机制、公式、消融、benchmark、失败分析与未来工作核对 |
| `figures/01-teaser.png` | 官方源码原图 | <https://arxiv.org/e-print/2608.27550> | Figure 1，p.3，`RAW/figures/teaser_v6.pdf` | 2026-10-21 | 概括 naive VLA pre-training 与表示中心方案的差异 |
| `figures/02-method.png` | 官方源码原图 | <https://arxiv.org/e-print/2608.27550> | Figure 3，p.6，`RAW/figures/vla_pt_main_v11.pdf` | 2026-10-21 | 说明浅层保护、caption 混训、多动作头与统一动作空间 |
| `figures/03-head-transfer.png` | 官方源码原图 | <https://arxiv.org/e-print/2608.27550> | Figure 2，p.4，`RAW/figures/action_head_transfer_bar_v4.pdf` | 2026-10-21 | 展示离散损失与连续单头 decoder lock-in 的 pilot study |
| `figures/04-realworld-transfer.png` | 官方源码原图 | <https://arxiv.org/e-print/2608.27550> | Figure 5，p.12，`RAW/figures/real_world_robocasa_vx_v3.pdf` | 2026-10-21 | 展示真机、双臂和未见本体 RoboCasa-GR1 迁移结果 |

## Additional official sources

- arXiv 页面：<https://arxiv.org/abs/2608.27550>
- 项目页：<https://starvla.github.io/VLAct>

## Notes

- 四张图片均优先从官方 arXiv LaTeX 源码包提取，使用 `research-field-survey/scripts/extract_source_figures.py` 转为限宽 2400 px 的 PNG；已逐张人工预览。Figure 5 的源 PDF 触发了 MuPDF xref 警告，但导出的整图内容、坐标轴、图例和边界均完整。
- 论文声明模型与训练 pipeline 已开源，但 PDF 正文未给出独立代码仓库 URL；因此笔记 frontmatter 的 `code` 保留为 `null`，项目页单独记录。
- 论文明确披露的主要限制是只研究 4B 规模 VLM；完整继续预训练步数、学习率、总 batch size 和各机器人数据的精确轨迹配比未在 PDF 中集中披露，笔记将其列为复现缺口而不补造。
