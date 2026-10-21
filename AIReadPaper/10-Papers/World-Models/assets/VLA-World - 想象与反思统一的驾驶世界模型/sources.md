# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v1，内含 Supplementary） | https://arxiv.org/pdf/2604.09059 | 全文，18 页 | 2026-10-16 | 精读主来源 |
| `figures/01-overview.png` | 训练与结果总览原图 | arXiv TeX 源包 `figures/intro.pdf` | Figure 1，p.1 | 2026-10-16 | 解释三阶段训练与双指标结果 |
| `figures/02-training-pipeline.png` | 方法流程原图 | arXiv TeX 源包 `figures/method.pdf` | Figure 3，p.4 | 2026-10-16 | 解释生成—反思—规划 token 流 |
| `figures/03-generation-planning.png` | 定性对比原图 | arXiv TeX 源包 `figures/vis.pdf` | Figure 4，p.8 | 2026-10-16 | 对比 FSDrive 的未来帧与轨迹 |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2604.09059
- 官方项目页：https://vlaworld.github.io

## 可用性说明

项目页提供论文、结果与方法展示，但截至 2026-10-16 未列出代码仓库、nuScenes-GR-20K 下载、checkpoint 或可执行训练配置。项目页的 “Supplementary” 实际仍指向同一 arXiv PDF；补充材料已包含在本地 `paper.pdf` 后半部分。