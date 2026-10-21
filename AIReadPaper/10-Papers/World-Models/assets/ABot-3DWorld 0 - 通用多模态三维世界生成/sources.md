# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2607.11673 | 全文 | 2026-10-16 | 精读主来源 |
| `figures/01-teaser.png` | arXiv 源码原图 | https://arxiv.org/abs/2607.11673 | Figure 1 | 2026-10-16 | 多模态输入与场景覆盖 |
| `figures/02-sgp-pipeline.png` | arXiv 源码原图 | https://arxiv.org/abs/2607.11673 | Figure 2 | 2026-10-16 | SGP、探索与 3DGS 总流程 |
| `figures/03-panoramic-video.png` | arXiv 源码原图 | https://arxiv.org/abs/2607.11673 | Figure 6 | 2026-10-16 | 全景视频几何条件与 scene memory |
| `figures/04-repair-refit.png` | arXiv 源码原图 | https://arxiv.org/abs/2607.11673 | Figure 13 | 2026-10-16 | 两轮 FLUX repair-and-refit |

## 官方链接

- arXiv：https://arxiv.org/abs/2607.11673
- arXiv HTML：https://arxiv.org/html/2607.11673
- 官方 Plaza / 项目展示：https://abot-world.amap.com/plaza

## 核验说明

- 四张图片均从 arXiv 官方源码包提取，并已使用 `Read` 逐张视觉核验；图像、文字、列标签和边缘完整。
- `paper.pdf` 可正常解析，共 31 页并具有文本层。
- 论文是 2026 年 arXiv 预印本；截至获取日未从官方材料确认同行评审 venue。
- “优于 Marble”主要来自作者定性比较；与 HY-World 2.0 有数值表，但并非所有指标领先。
- 全景视频生成约 12 分钟/场景，3DGS 修复重建阶段约 10 分钟/场景，二者均基于 4×RTX 4090，应分开理解。
