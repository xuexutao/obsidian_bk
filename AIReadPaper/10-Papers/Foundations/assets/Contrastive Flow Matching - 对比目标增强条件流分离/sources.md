# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2506.05350 | 全文，v1，13 页（含附录 A–B） | 2026-10-25 | 精读主来源、公式推导与实验表格 |
| `figures/01-teaser.png` | Teaser 对比图 | https://arxiv.org/pdf/2506.05350 | Figure 1, p.1 | 2026-10-25 | 定性结果：FM 与 ΔFM 同噪声成对比较 |
| `figures/02-flow-overlap-motivation.png` | 动机示意图 | https://arxiv.org/pdf/2506.05350 | Figure 2, p.2 | 2026-10-25 | 研究背景：条件流重叠的直觉 |
| `figures/03-toy-2d-flows.png` | 二维 toy 实验 | https://arxiv.org/pdf/2506.05350 | Figure 3, p.5 | 2026-10-25 | 方法：流分离示意及其偏离真实条件分布的讨论 |
| `figures/04-denoising-trajectories.png` | 去噪轨迹可视化 | https://arxiv.org/pdf/2506.05350 | Figure 4, p.7 | 2026-10-25 | 定性结果：中间步骤结构形成速度 |
| `figures/05-training-inference-efficiency.png` | 训练/推理效率曲线 | https://arxiv.org/pdf/2506.05350 | Figure 5, p.9 | 2026-10-25 | 实验：9× 训练、5× 采样步数主张 |
| `figures/06-main-tables.png` | 主结果表 | https://arxiv.org/pdf/2506.05350 | Table 1–3, p.6（左右两栏拼接，去除正文） | 2026-10-25 | 实验：SiT / REPA SiT / CC3M 主结果 |
| `figures/07-cfg-lambda-batch-tables.png` | CFG、λ 与 batch 消融表 | https://arxiv.org/pdf/2506.05350 | Table 4–6, p.8（左右两栏拼接，去除正文） | 2026-10-25 | 实验：CFG 叠加、λ 敏感性、batch size |

## 其它官方来源

| 材料 | URL | 获取日期 | 用途 |
|---|---|---|---|
| 官方代码（ICCV 2025） | https://github.com/gstoica27/DeltaFM | 2026-10-25 | 训练/评测命令、interference vector 用法 |
| 对比损失实现 | https://github.com/gstoica27/DeltaFM/blob/main/triplet_loss.py | 2026-10-25 | 核对负样本采样（默认任意非自身样本，不区分类别）与损失形式 |
| 官方权重 | https://huggingface.co/gstoica3/DeltaFM | 2026-10-25 | REPA SiT-XL/2 256 / 512 检查点 |
| arXiv 摘要页 | https://arxiv.org/abs/2506.05350 | 2026-10-25 | 版本历史（仅 v1）与 DOI |

## 说明

- 论文无单独 Supplementary Material，附录已包含在 arXiv PDF 中。
- 所有图表均从官方 PDF 按图注坐标裁剪（170–300 DPI），非内嵌碎片；Table 图为同页左右两栏拼接，未改动内容。
