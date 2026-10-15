# Sources

| 本地文件 | 材料类型 | 原始来源 | 位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/genie1-paper.pdf` | 正式论文 PDF | https://arxiv.org/pdf/2402.15391 | 全文 | 2026-10-15 | Genie 1 架构、训练、数据、实验与限制 |
| `figures/01-genie1-overview.png` | 论文原图 | https://arxiv.org/abs/2402.15391 | Figure 3 | 2026-10-15 | Genie 1 三模块总览 |
| `figures/02-genie1-lam.png` | 论文原图 | https://arxiv.org/abs/2402.15391 | Figure 5 | 2026-10-15 | 潜动作模型信息瓶颈 |
| `figures/03-genie1-st-transformer.png` | 论文原图 | https://arxiv.org/abs/2402.15391 | Figure 4 | 2026-10-15 | ST-Transformer 空间/时间分解 |
| `figures/04-genie2-official-page.png` | 官方页面截图 | https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/ | Diffusion world model 区域 | 2026-10-15 | Genie 2 高层架构说明 |
| `figures/05-genie3-official-page.png` | 官方页面截图 | https://deepmind.google/models/genie/ | Capabilities/long-horizon consistency 区域 | 2026-10-15 | Genie 3 720p、20–24 FPS、长时一致性 |

## 官方网页来源

- Genie 1（Google DeepMind publication）：https://deepmind.google/research/publications/60474/
- Genie 1（PMLR / ICML 2024）：https://proceedings.mlr.press/v235/bruce24a.html
- Genie 1（arXiv HTML）：https://arxiv.org/html/2402.15391
- Genie 2 官方博客：https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
- Genie 3 原始公告：https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- Genie 3 当前模型页 / Project Genie：https://deepmind.google/models/genie/

## 公开性说明

- Genie 1 有正式论文与完整附录，但大模型权重和主训练数据未公开。
- Genie 2、Genie 3 截至 2026-10-15 未找到 Google DeepMind 发布的可复现级正式论文或 technical report；相关架构叙述只采用官方博客和模型页明确内容。
- 未把社区推测的 DiT、KV cache、分层 tokenizer、LAM 沿用关系或参数量当作事实。
