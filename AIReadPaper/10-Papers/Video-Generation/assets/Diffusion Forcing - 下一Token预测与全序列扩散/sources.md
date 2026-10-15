# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v4） | https://arxiv.org/pdf/2407.01392 | 全文，35 页 | 2026-09-04 | 精读主来源、公式与实验核对 |
| `figures/01-capabilities.png` | 能力对比完整图 | https://arxiv.org/pdf/2407.01392 | Figure 1, p.2 | 2026-09-04 | 研究动机与范式对比 |
| `figures/02-method-overview.png` | 方法总览完整图 | https://arxiv.org/pdf/2407.01392 | Figure 2, p.3 | 逐 token 独立噪声机制 |
| `figures/03-video-rollout.png` | 视频长期生成完整图 | https://arxiv.org/pdf/2407.01392 | Figure 3, p.7 | 视频 baseline 与长 rollout |
| `figures/04-planning-results.png` | 规划结果与表格完整图 | https://arxiv.org/pdf/2407.01392 | Table 1, p.8 | Maze2D 定量结果与 MCG 消融 |
| — | 官方项目页 | https://boyuan.space/diffusion-forcing | 全站 | 2026-09-04 | 演示、失败案例、后续方向 |
| — | 官方代码仓库 | https://github.com/buoyancy99/diffusion-forcing | README / `paper` branch | 2026-09-04 | 环境、checkpoint 与复现口径 |

## 获取与处理说明

- PDF 直接取自 arXiv 官方地址。
- 四张图片均从 `sources/paper.pdf` 对应页面按完整图表区域以 300 DPI 裁剪，保留原始图注；已逐张预览检查，无标题、图例、坐标轴或图像边缘被截断。
- 未发现独立 Supplementary Material PDF；附录已包含在 35 页 arXiv PDF 中。
- 项目页的 2025 年 Wan2.1 / Diffusion Forcing 2 更新属于后续工作，正文已与原论文实验分开说明。
