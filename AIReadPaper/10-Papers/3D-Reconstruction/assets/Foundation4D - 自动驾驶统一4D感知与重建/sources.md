# Sources

## 论文与本地材料

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2610.30340 | 全文，v1 | 2026-10-26 | 精读主来源 |
| `figures/01-teaser.png` | 任务概览图 | https://arxiv.org/pdf/2610.30340 | Figure 1, p.1 | 2026-10-26 | 说明五种互补输出 |
| `figures/02-method-pipeline.png` | 方法架构图 | https://arxiv.org/pdf/2610.30340 | Figure 2, p.7 | 2026-10-26 | 核心方法与张量流 |
| `figures/03-gaussian-results.png` | 前馈 Gaussian 结果 | https://arxiv.org/pdf/2610.30340 | Figure 15, p.36 | 2026-10-26 | 六相机 Gaussian 重建 |
| `figures/04-unified-scene-output.png` | 统一场景输出 | https://arxiv.org/pdf/2610.30340 | Figure 17, p.39 | 2026-10-26 | 几何、占据、轨迹的场景级对齐 |
| `figures/05-lidarless-nurec.png` | NuRec 应用流程 | https://arxiv.org/pdf/2610.30340 | Figure 18, p.41 | 2026-10-26 | 说明 Foundation4D 作为优化式重建先验 |

以上图片均从官方 PDF 以 300 DPI 完整裁剪，保留原始图例与图注；未使用 PDF 内嵌碎片图。

## 官方数据与产品资料

| 资源 | URL | 获取日期 | 用途与结论 |
|---|---|---|---|
| PAI–Foundation4D 数据集 | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-Foundation4D | 2026-10-26 | 1,253 个 clips 的 metric depth 与 voxel occupancy/semantic/velocity 标签；gated access，受 NVIDIA 非商业用途许可约束；不含 Foundation4D 代码或权重 |
| PhysicalAI Autonomous Vehicles | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles | 2026-10-26 | 上游多传感器驾驶数据集 |
| NuRec 工作原理 | https://docs.nvidia.com/nurec/basics/how-nurec-works.html | 2026-10-26 | 核实 NuRec 是逐场景优化式神经重建系统，不是 Foundation4D 模型代码 |
| NCore 传感器模型 | https://nvidia.github.io/ncore/data/sensor_models.html | 2026-10-26 | 数据格式/相机模型背景；NCore 开源不等于 Foundation4D 开源 |

## 开源状态核查

结论：**截至 2026-10-26，论文和官方 PAI–Foundation4D 数据标注已发布；Foundation4D 模型代码、训练/推理代码与模型权重未公开。**

核查入口：

- arXiv 论文页：https://arxiv.org/abs/2610.30340
- Hugging Face 论文页：https://huggingface.co/papers/2610.30340
- 官方数据页：https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-Foundation4D
- GitHub 官方关键词检索：未发现 NVIDIA/Foundation4D 仓库
- Hugging Face 模型关键词检索：未发现 Foundation4D 模型仓库或 checkpoint

论文提到的 “publicly released checkpoints and codebases” 指 DVGT、LiAuto-GeoX、GeoUP 等 baselines；Instant NuRec 1.1.0 checkpoint 也属于独立项目，均不能计为 Foundation4D 官方代码。

## 领域发展的一手来源

| 工作 | 官方来源 | 获取日期 | 与 Foundation4D 的关系 |
|---|---|---|---|
| VGGT | https://vgg-t.github.io/ | 2026-10-26 | 通用前馈视觉几何基础模型，Foundation4D 的架构起点 |
| GeoUP | https://arxiv.org/abs/2608.13147 | 2026-10-26 | geometry + detection + occupancy 的最直接感知侧前身 |
| DynamicVGGT | https://openaccess.thecvf.com/content/CVPR2026/html/He_DynamicVGGT_Learning_Dynamic_Point_Maps_for_4D_Scene_Reconstruction_in_CVPR_2026_paper.html | 2026-10-26 | 运动感知 attention 与动态 Gaussian head |
| DGGT | https://arxiv.org/abs/2512.03004 | 2026-10-26 | 无位姿输入的驾驶场景前馈 4D Gaussian 重建 |
| Any4D | https://arxiv.org/abs/2512.10935 | 2026-10-26 | 度量 4D 因子化表示与多模态输入 |
| QuadricFormer | https://arxiv.org/abs/2506.10977 | 2026-10-26 | 用 superquadric 替代单一椭球 Gaussian 的 occupancy primitive 路线 |
| HUGSIM | https://www.computer.org/csdl/journal/tp/2026/04/11314796/2cHSBuLRGeI | 2026-10-26 | 3DGS 驾驶闭环仿真与评测 |

## 阅读与复核记录

- 论文方法、训练、实验、Tables 2–18 与 limitations 均基于官方 50 页 PDF 逐节核对。
- 关键张量数字中，$37\times37=1369$、每图序列长度 1375、occupancy 网格 $600\times300\times40$ 为依据论文明确尺寸推导，正文已按“推导”语气表述。
- 公开 benchmark 成绩均有 benchmark-specific fine-tuning，不按 zero-shot 结果表述。
- Gaussian 直接前馈分支与 lidarless NuRec 逐场景优化应用已分开描述，避免混淆。
