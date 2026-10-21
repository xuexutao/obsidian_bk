# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | arXiv 官方预印本 PDF | https://arxiv.org/pdf/2603.23376 | 扩展全文，26 页 | 2026-10-16 | 精读主来源与附录 |
| `sources/github-tech-report.pdf` | GitHub 官方技术报告 | https://github.com/amap-cvlab/ABot-PhysWorld/blob/main/tech_report/ABot-PhysWorld-tech-report.pdf | 技术报告，16 页 | 2026-10-16 | 核对官方仓库版本 |
| `figures/01-data-pipeline.png` | arXiv 源码原图 | https://arxiv.org/abs/2603.23376 | Figure 1 | 2026-10-16 | 数据过滤、平衡与 caption |
| `figures/02-physics-alignment.png` | arXiv 源码原图 | https://arxiv.org/abs/2603.23376 | Figure 2 | 2026-10-16 | SFT、VLM 偏好构造与 DPO |
| `figures/03-action-injection.png` | arXiv 源码原图 | https://arxiv.org/abs/2603.23376 | Figure 4 | 2026-10-16 | action map 并行 context blocks |
| `figures/04-pbench-results.png` | arXiv 源码原图 | https://arxiv.org/abs/2603.23376 | Figure 5 | 2026-10-16 | PBench 定性物理错误比较 |

## 官方链接

- GitHub：https://github.com/amap-cvlab/ABot-PhysWorld
- GitHub 技术报告目录：https://github.com/amap-cvlab/ABot-PhysWorld/tree/main/tech_report
- arXiv：https://arxiv.org/abs/2603.23376
- arXiv HTML：https://arxiv.org/html/2603.23376

## 核验说明

- 用户指定的“官方 GitHub 技术报告”真实存在，文件名为 `ABot-PhysWorld-tech-report.pdf`；另有更长的 arXiv 预印本，因此本笔记按 `paper-note` 而非 `project-note` 处理。
- 四张图片来自 arXiv 官方源码包，均已使用 `Read` 逐张视觉核验；图像、标签和边界完整。
- arXiv PDF 共 26 页，GitHub 技术报告共 16 页，两者均可正常解析并具有文本层。
- 论文尚未在官方材料中确认同行评审 venue；“trajectory consistency 超过 Veo 3.1/Sora v2 Pro”未被 A2V 数值表直接覆盖。
