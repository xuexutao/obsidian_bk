# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v5） | https://arxiv.org/pdf/2602.02214 | 全文，20 页 | 2026-09-04 | 精读主来源、理论与实验核对 |
| `figures/01-existing-gap.png` | 问题动机完整图 | https://arxiv.org/pdf/2602.02214 | Figure 1, p.2 | 2026-09-04 | 标准 DMD 与 Self Forcing 差距 |
| `figures/02-injectivity-principle.png` | 核心理论完整图 | https://arxiv.org/pdf/2602.02214 | Figure 3, p.4 | 2026-09-04 | video/frame-level injectivity |
| `figures/03-qualitative-comparison.png` | 定性对比完整图 | https://arxiv.org/pdf/2602.02214 | Figure 6, p.7 | 2026-09-04 | 动态与视觉质量比较 |
| `figures/04-main-results.png` | 主结果完整表 | https://arxiv.org/pdf/2602.02214 | Table 1, p.8 | 2026-09-04 | 速度、VBench 与偏好指标 |
| — | 官方项目页 | https://thu-ml.github.io/CausalForcing.github.io/ | 全站 | 2026-09-04 | 方法摘要与视频演示 |
| — | 官方代码仓库 | https://github.com/thu-ml/Causal-Forcing | README / configs | 2026-09-04 | checkpoint、I2V 与长视频扩展说明 |

## 获取与处理说明

- PDF 直接取自 arXiv 官方地址，版本为 v5（2026-06-01）。
- 四张图片均从 `sources/paper.pdf` 对应页面按完整图表区域以 300 DPI 裁剪并保留图注；已逐张预览检查，未使用碎片图。
- 未发现独立 Supplementary Material PDF；论文说明 prompts 与更多视频位于补充材料/项目页，但公开入口未提供单独可下载的 supplement 文件。
- 官方代码明确提醒原始 81 帧 checkpoint 不原生支持长于 5 秒的视频；Rolling Forcing、LongLive 等属于正交扩展。
