# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2312.13139> | 全文，22 页 | 2026-10-21 | 精读主来源 |
| `figures/01-overview.png` | 官方源码原图 | <https://arxiv.org/e-print/2312.13139> | Figure 1，p.2，`images/overview.pdf` | 2026-10-21 | 两阶段训练总览 |
| `figures/02-architecture.png` | 官方源码原图 | <https://arxiv.org/e-print/2312.13139> | Figure 2，p.4，`images/encoder_decoder.pdf` | 2026-10-21 | 编解码器与输入输出机制 |
| `figures/03-ablation.png` | 官方源码原图 | <https://arxiv.org/e-print/2312.13139> | Figure 7，p.15，`images/ablation.pdf` | 2026-10-21 | 视频预测与预训练消融 |
| `figures/04-real-robot.png` | 官方源码原图 | <https://arxiv.org/e-print/2312.13139> | Figure 5，p.8，`images/real_robot.pdf` | 2026-10-21 | 真机任务与 OOD 设置 |
| 项目页 | 官方项目页 | <https://gr1-manipulation.github.io> | 论文首页 | 2026-10-21 | 视频与补充展示 |

## 提取与核验说明

- 图片优先从 arXiv 官方 LaTeX 源码包提取，使用 `research-field-survey/scripts/extract_source_figures.py` 转为限宽 2400 px 的 PNG；未从 PDF 抽取零散内嵌图片。
- 四张图均已逐张目视检查：标题、坐标轴、图例、边缘与子图完整，无切边；正文全部引用。
- 论文文本由本地 `sources/paper.pdf` 逐页解析，并与 PDF 页码、表格和附录交叉核对。
