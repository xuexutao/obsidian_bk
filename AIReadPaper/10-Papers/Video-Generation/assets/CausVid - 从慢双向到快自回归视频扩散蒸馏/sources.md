# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2412.07772 | 全文，18 页 | 2026-09-04 | 精读主来源、方法与实验核对 |
| `figures/01-teaser.png` | Teaser 完整图 | https://arxiv.org/pdf/2412.07772 | Figure 1, p.1 | 2026-09-04 | 延迟与流式生成动机 |
| `figures/02-architecture.png` | AR-DiT 架构完整图 | https://arxiv.org/pdf/2412.07772 | Figure 5, p.7 | 2026-09-04 | 块内双向、块间因果结构 |
| `figures/03-distillation-method.png` | 两阶段蒸馏完整图 | https://arxiv.org/pdf/2412.07772 | Figure 6, p.8 | 2026-09-04 | ODE 初始化与 asymmetric DMD |
| `figures/04-vbench-results.png` | VBench 雷达完整图 | https://arxiv.org/pdf/2412.07772 | Figure 10, p.17 | 2026-09-04 | 16 维结果对比 |
| — | 官方项目页 | https://causvid.github.io/ | 全站 | 2026-09-04 | 视频演示、VBench-Long 与发布说明 |
| — | 官方代码仓库 | https://github.com/tianweiy/CausVid | README / configs | 2026-09-04 | Wan2.1 开源实现、checkpoint 与推理命令 |

## 获取与处理说明

- PDF 直接取自 arXiv 官方地址。
- 四张图片均从 `sources/paper.pdf` 对应页面按完整图表区域以 300 DPI 裁剪并保留图注；已逐张预览检查，未使用 PDF 内嵌碎片。
- 未发现独立 Supplementary Material PDF；补充实验位于 arXiv PDF 后半部分，项目页另提供视频演示。
- 论文原始实验与后续 Wan2.1 开源代码存在模型/数据口径差异，正文应明确区分，不能把代码版配置当作原论文训练配置。
