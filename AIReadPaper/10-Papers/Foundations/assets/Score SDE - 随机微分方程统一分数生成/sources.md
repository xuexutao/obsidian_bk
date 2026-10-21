# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `2011.13456.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2011.13456 | 全文，arXiv v2 / ICLR 2021 | 2026-10-16 | 精读、公式与数字核验主来源 |
| `fig1_forward_reverse_sde.png` | 前向/反向 SDE 示意图 | 官方 PDF 300 DPI 裁剪 | Figure 1，PDF p.2 | 2026-10-16 | 解释 forward / reverse SDE 与 score |
| `fig2_framework_overview.png` | 框架总览图 | 官方 PDF 300 DPI 裁剪 | Figure 2，PDF p.4 | 2026-10-16 | 解释 SDE 与 probability-flow ODE 的共同边际 |
| `table1_sampler_ablation.png` | 采样器对比表 | 官方 PDF 300 DPI 裁剪 | Table 1，PDF p.6 | 2026-10-16 | Predictor / Corrector / PC 消融 |
| `table2_3_likelihood_fid.png` | likelihood 与样本质量结果表 | 官方 PDF 300 DPI 裁剪 | Tables 2–3，PDF p.7 | 2026-10-16 | CIFAR-10 NLL、FID、IS 核验 |
| `fig3_probability_flow_ode.png` | Probability-flow ODE 图 | 官方 PDF 300 DPI 裁剪 | Figure 3，PDF p.8 | 2026-10-16 | NFE、插值与确定性编码说明 |
| `fig4_inverse_problems.png` | 条件生成与逆问题结果 | 官方 PDF 300 DPI 裁剪 | Figure 4，PDF p.9 | 2026-10-16 | 类条件生成、修复、着色 |
| `fig9_pc_allocation.png` | PC 计算分配消融图 | 官方 PDF 300 DPI 裁剪 | Figure 9，PDF p.23 | 2026-10-16 | predictor / corrector 计算预算分析 |
| `fig10_architecture_ablation.png` | 架构组件消融图 | 官方 PDF 300 DPI 裁剪 | Figure 10，PDF p.25 | 2026-10-16 | NCSN++ 架构改进分析 |

## 原始来源

- arXiv 摘要页：https://arxiv.org/abs/2011.13456
- arXiv 官方 PDF：https://arxiv.org/pdf/2011.13456
- 会议版本：ICLR 2021 conference paper（论文 PDF 首页标注）
- 官方代码（论文附录 H 给出）：https://github.com/yang-song/score_sde

## 历史边界核验条目

- DDPM：Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020，arXiv:2006.11239。
- DDIM：Song, Meng, Ermon, *Denoising Diffusion Implicit Models*, arXiv:2010.02502；其 arXiv 初稿早于 Score SDE 的 arXiv:2011.13456，二者是 2020 年同期的确定性扩散采样路线，不能把 DDIM 归为本文后续贡献。
- Flow Matching：Lipman et al., *Flow Matching for Generative Modeling*, arXiv:2210.02747，初稿发表于 2022 年，晚于本文；FM 直接回归概率路径的速度场，diffusion probability-flow ODE 可作为其路径特例，但 FM / CFM / 条件 OT 路径并非 Score SDE 原论文贡献。

## 核验说明

- 上述图表均从保存的官方 PDF 按原页坐标以 300 DPI 裁剪，并逐张人工检查完整性、清晰度及图注；未把 PDF 内嵌碎片图误当作完整原图。
- `source.md`、`meta.json`、`images/pages/` 与 `images/img_*` 是抓取与逐页核验材料，保留在本附件目录内以支持离线溯源；未清理个人目录中的材料。
- 论文数字以官方 PDF 为准。Table 3 报告的是训练过程中最低 FID 的 checkpoint 与 PC 采样结果；Table 2 使用最后 checkpoint 和 black-box ODE solver，二者不能直接混为同一评估协议。
- “exact likelihood”指在学得的 probability-flow ODE 所定义连续密度模型下，通过瞬时变量替换公式计算；实际数值仍包含 score 估计、ODE 离散和 Hutchinson 散度估计误差。
