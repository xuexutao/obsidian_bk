# Sources — DreamZero

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `fig1-overview.png` | 总览图（teaser） | https://arxiv.org/html/2602.15922v1/dreamzero-header-v2.png | Figure 1, p.1 | 2026-08-20 | §5.2 定性结果章节 |
| `fig4-model.png` | 方法架构图 | https://arxiv.org/html/2602.15922v1/dreamzero_model.png | Figure 4, p.4 | 2026-08-20 | §3.1 总体框架 |
| `fig5-main-results.png` | 评估任务图谱 | https://arxiv.org/html/2602.15922v1/main_eval_setting.png | Figure 5, p.5 | 2026-08-21 | §5.4 评测协议 |
| `fig11-embodiment-transfer.png` | 跨 embodiment 迁移 | https://arxiv.org/html/2602.15922v1/embodiment_transfer.png | Figure 11, p.10 | 2026-08-20 | §5.4 跨 embodiment 迁移 |

## 在线资源（未本地化保存）

| 类别 | 名称 | 链接 | 用途 |
|---|---|---|---|
| 论文 PDF | arXiv:2602.15922v1 | https://arxiv.org/abs/2602.15922 | 精读主来源 |
| 论文 HTML | arXiv HTML 版 | https://arxiv.org/html/2602.15922 | 提取方法图与定量结果 |
| 项目页 | DreamZero Project | https://dreamzero0.github.io/ | 视频与定性结果 |
| 代码仓库 | dreamzero/dreamzero | https://github.com/dreamzero0/dreamzero | 复现与推理代码 |
| OpenReview | cd33uUB609 | https://openreview.net/forum?id=cd33uUB609 | 同行评议记录 |

## 备注

- 论文 PDF、Supplementary Material 与官方仿真器（PolaRiS / Genie Sim 3.0）暂未本地化保存。复现 AgiBot 主实验时建议从 GitHub 仓库下载 `paper.pdf` 与官方 supplementary。
- 论文中部分实现细节（如多视角拼接分辨率、$w(t_k)$ 权重函数具体形式、DreamZero-Flash 微调步数）作者未公开，已在笔记中以"论文未披露"标注。
