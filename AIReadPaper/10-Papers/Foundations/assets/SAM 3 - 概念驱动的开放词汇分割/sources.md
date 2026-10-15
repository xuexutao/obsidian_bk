# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2511.16719 | 全文，arXiv:2511.16719 | 2026-10-15 | 精读主来源、公式与实验数据核对 |
| `figures/01-promptable-concept-segmentation.png` | 论文关键图 | https://arxiv.org/pdf/2511.16719 | Figure 1, p.1 | 2026-10-15 | 区分 PVS 与新提出的 PCS 任务 |
| `figures/02-detector-tracker-architecture.png` | 论文关键图 | https://arxiv.org/pdf/2511.16719 | Figure 4, p.4 | 2026-10-15 | 说明概念检测、分割、视频跟踪与记忆库的统一流程 |
| `figures/03-data-engine.png` | 论文关键图 | https://arxiv.org/pdf/2511.16719 | Figure 5, p.5 | 2026-10-15 | 说明人类与 AI verifier 协同的数据引擎闭环 |
| `figures/04-open-vocabulary-results.png` | 论文关键图 | https://arxiv.org/pdf/2511.16719 | Figure 2, p.2 | 2026-10-15 | 对比 SAM 3 与 OWLv2 的开放词汇分割结果 |

## 官方链接

- 论文主页：https://arxiv.org/abs/2511.16719
- 官方演示：https://segment-anything.com
- 官方项目页：https://ai.meta.com/sam3
- 官方代码：https://github.com/facebookresearch/sam3

## 来源说明

- 本文研究对象是 **SAM 3: Segment Anything with Concepts**，不是 SAM 3D。
- 所有本地图片均从官方 PDF 对应页面以完整图表区域裁剪，保留原始 Figure 编号和图注，不使用 PDF 内嵌位图碎片。
- 本笔记中的数据规模、模型结构、结果和消融数字均以本地 `sources/paper.pdf` 为准。
