# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v2） | https://arxiv.org/abs/2609.10540 | 全文，17 页 | 2026-10-29 | 精读主来源 |
| `figures/01-overview.png` | 论文总览图 | https://arxiv.org/abs/2609.10540 | Figure 1，PDF p.1 | 2026-10-29 | 动机与总体框架 |
| `figures/02-representation-tradeoffs.png` | 表示权衡图 | https://arxiv.org/abs/2609.10540 | Figure 2，PDF p.3 | 2026-10-29 | 解释为何选择 state-augmented 3D OBB |
| `figures/03-architecture.png` | 方法架构图 | https://arxiv.org/abs/2609.10540 | Figure 3，PDF p.6 | 2026-10-29 | 核心方法与长时记忆 |
| `figures/04-benchmark-results.png` | 关键定量与定性结果 | https://arxiv.org/abs/2609.10540 | Table 1 + Figure 4，PDF p.12 | 2026-10-29 | CombatStateBench 结果与死亡事件对比 |
| `figures/05-generalization-results.png` | 泛化与长时结果 | https://arxiv.org/abs/2609.10540 | Figure 5，PDF p.13 | 2026-10-29 | 开放域、离屏实体与 897 帧生成 |

## 官方网页与仓库

| 名称 | URL | 获取日期 | 说明 |
|---|---|---|---|
| 项目页 | https://alaya-lab.github.io/pwm/ | 2026-10-29 | 展示显式状态、离屏持久性、事件触发和 897 帧长时生成 |
| 官方仓库 | https://github.com/AlayaLab/pwm | 2026-10-29 | 检查 commit `0286dc7a193a99f84c29f8d25fc67976311d0833`；仓库仅含 `README.md` 与 `assets/teaser.png`，尚未发布推理代码和权重 |
| PROWBench | https://alaya-lab.github.io/PROWBench/ | 2026-10-29 | 2026-10-11 后续发布的程序条件视频渲染 benchmark；不是论文 Table 1 的 CombatStateBench |

## 材料缺失说明

- 论文与项目页未提供独立 Supplementary Material。
- 截至 2026-10-29，官方仓库路线图列出 inference code 与 pretrained weights，但对应文件尚未发布，因此无法进行代码入口、默认配置和精确张量形状核验。
- 论文未披露优化器、学习率、batch size、训练步数、训练视频规模、控制 embedding 维度、identity slot 数量、chunk 长度、推理延迟和硬件配置。
