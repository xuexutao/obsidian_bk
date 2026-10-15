# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2609.07398 | 全文，v1 | 2026-10-15 | 精读主来源、公式、实验表格与附录 |
| `figures/01-overview.png` | 论文总览图 | https://arxiv.org/pdf/2609.07398 | Figure 1, p.1 | 2026-10-15 | 解释 OpenWAM-Infra / Study / α 三层结构 |
| `figures/02-infrastructure.png` | 模块化基础设施图 | https://arxiv.org/pdf/2609.07398 | Figure 2, p.5 | 2026-10-15 | 解释视觉编码器、流骨干、可见性掩码与六类架构 |
| `figures/03-visual-representation.png` | 视觉表征对比图 | https://arxiv.org/pdf/2609.07398 | Figure 7, p.12 | 2026-10-15 | 分析 reconstructive / representation encoder 与 S-VAE |
| `figures/04-information-flow-and-denoising.png` | 信息流与去噪消融图 | https://arxiv.org/pdf/2609.07398 | Figure 8、Table 2、Figure 9, p.14 | 2026-10-15 | 分析训练期可见性与推理期同步/异步去噪 |
| `figures/05-pretraining-ood.png` | 预训练迁移结果图 | https://arxiv.org/pdf/2609.07398 | Figure 10, p.15 | 2026-10-15 | 分析 robot / ego 数据对 ID 与 OOD 的不同贡献 |
| `figures/06-pretraining-visibility.png` | 预训练前后信息流偏好图 | https://arxiv.org/pdf/2609.07398 | Figure 11, p.16 | 2026-10-15 | 分析 Mutual visibility 的小幅反转 |
| `figures/07-openwam-alpha-architecture.png` | OpenWAM-α 总体架构图 | https://arxiv.org/pdf/2609.07398 | Figure 12, p.17 | 2026-10-15 | 解释双流联合注意力、数据混合、噪声平面和 80-D 动作空间 |
| `figures/08-simulation-benchmarks.png` | 仿真基准总览图 | https://arxiv.org/pdf/2609.07398 | Figure 13, p.21 | 2026-10-15 | 汇总八个仿真 benchmark 的代表性对比 |
| `figures/09-libero-plus-and-data.png` | LIBERO-Plus 分项表与数据覆盖图 | https://arxiv.org/pdf/2609.07398 | Table 5、Figure 15, p.22 | 2026-10-15 | 深入分析相机/噪声失败与单臂预训练数据缺口 |
| `figures/10-vla-vs-wam.png` | VLA 与 WAM 的 ID/OOD 对比图 | https://arxiv.org/pdf/2609.07398 | Figure 16, p.24 | 2026-10-15 | 分析两类范式的拟合与泛化差异 |

## 其它官方来源

| 材料 | URL | 获取日期 | 用途 |
|---|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2609.07398 | 2026-10-15 | 核对标题、作者、版本、摘要与外部资源 |
| 项目页 | https://openwam-official.github.io/ | 2026-10-15 | 核对发布资源、演示、模型概览和 46 个 checkpoint |
| 官方代码仓库 | https://github.com/OpenWAM-Official/OpenWAM | 2026-10-15 | 核对配置、训练/部署方式、公开 backbone 与硬件建议 |
| 模型与数据 | https://huggingface.co/OpenWAM | 2026-10-15 | 公开 checkpoint 与数据资源入口 |
| 完整 benchmark 页面 | https://openwam-official.github.io/results/ | 2026-10-15 | 在线逐 benchmark 结果入口 |

## 获取说明

- arXiv v1 已把 Appendix A–D 合并在同一份 PDF 中，未发现独立 supplementary PDF；因此 `sources/paper.pdf` 同时覆盖主文与补充实验细节。
- 所有本地图片均从官方 PDF 以高清页面区域裁剪，保留图号、图例或图注；未使用 PDF 内嵌的零散位图碎片。
