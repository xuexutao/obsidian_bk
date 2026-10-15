# CUT3R 资料来源与图片核验

- 获取日期：2026-10-15
- 论文：Continuous 3D Perception Model with Persistent State
- arXiv：2501.12387
- 会议：CVPR 2025 Oral

## 1. CUT3R 官方来源

1. arXiv 摘要页：https://arxiv.org/abs/2501.12387
2. arXiv 官方 PDF：https://arxiv.org/pdf/2501.12387
3. arXiv 官方源码：https://arxiv.org/e-print/2501.12387
4. 官方项目页：https://cut3r.github.io/
5. 官方代码：https://github.com/CUT3R/CUT3R

## 2. 本地归档

- `paper.pdf`：arXiv 官方 PDF，17 页。
- `cut3r-arxiv-source.tar`：arXiv 官方 LaTeX 源码包。
- `latex/`：源码包展开目录，保留论文正文、表格与原始矢量图。
- `fetch/source.md`：由官方 PDF 抽取的全文文本。
- `fetch/meta.json`：PDF 元信息与页面渲染清单。
- `fetch/images/pages/`：17 页 PDF 的逐页高清渲染，用于交叉核对版面与图注。

## 3. 采用图片与原始文件映射

| 本地图片 | 论文位置 | 官方源码原图 | 内容 | Read 核验 |
|---|---|---|---|---|
| `figures/01-teaser.png` | Figure 1 | `latex/figs/v6.pdf` | 静态场景、动态场景、稀疏照片集合三类输入与重建结果 | 已核验：画面完整，三组标题和相机轨迹清晰 |
| `figures/02-method.png` | Figure 3 | `latex/figs/state_overview.pdf` | recurrent state、state update/readout、pointmap/camera 输出与 raymap readout | 已核验：左右流程完整，未裁掉模块或文字 |
| `figures/03-virtual-query.png` | Figure 2 | `latex/figs/fig2_4.pdf` | 图像观测重建与虚拟视角补全对比 | 已核验：蓝色虚拟相机及补全地面/凳子区域完整 |
| `figures/04-state-analysis.png` | Figure 5 | `latex/figs/revisiting5.pdf` | Online 与 Revisiting 的状态分析 | 已核验：输入顺序、两种结果及红圈区域完整 |
| `figures/05-unseen-results.png` | Figure 6 | `latex/figs/raymap3.pdf` | 未见视图查询的 RGB、GT、深度、查询前后点图 | 已核验：四列五行与标签完整；白底可读 |
| `figures/06-dynamic-results.png` | Figure 4 | `latex/figs/qualitative/qual_res_3.pdf` | 互联网动态视频上 Spann3R、MonST3R、CUT3R 定性比较 | 已核验：三组场景、方法名与输入帧完整 |

所有 PNG 均由 arXiv 源码中的官方矢量 PDF 以白底转换；常规图片为 240 DPI，较小的 state-analysis 原图提升到 480 DPI；没有使用论文 PDF 内嵌碎片图。

## 4. 下游工作来源

### SPMem

- 论文：Video World Models with Long-term Spatial Memory
- arXiv：https://arxiv.org/abs/2506.05284
- NeurIPS 2025 论文：https://proceedings.neurips.cc/paper_files/paper/2025/file/467655d26fcc207bca08915dc91964c6-Paper-Conference.pdf
- 项目页：https://spmem.github.io/
- 关键核验：数据构建阶段使用 Mega-SaM；自回归推理阶段改用 CUT3R 在线重建 metric pointmap 与 camera parameters，并在相邻生成步骤间保存 CUT3R state dict 和 pose retriever 参数，使点图保持同一坐标系；之后用 TSDF-Fusion 过滤动态部分并写入长期静态空间记忆。

### FantasyWorld

- 论文：FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction
- arXiv：https://arxiv.org/abs/2509.21657
- 关键核验：约 180K 视频片段中，DL3DV、WildRGB、ScanNet、TartanAir 使用 CUT3R 提取 geometric labels；这些标签监督几何分支的 depth、point map 与 camera prediction。RealEstate10K、ACID 则使用另一套 reconstruction-based pipeline。

## 5. 原文披露与歧义

- 正文称训练使用 **32 datasets**；附录 Table 6 有 **33 行数据条目**。可将 ARKitScenes 与 ARKitScenes-HighRes 视为同一数据集家族的两个版本，从而得到 32 个家族，但论文没有显式解释该计数口径。
- 正文总述称估计 camera intrinsics and extrinsics；Method 明确定义的独立 camera head 只输出 current-to-world 的 6-DoF pose。内参没有独立 head/损失，更多由 pointmap 或 raymap 参数化隐式承载。
- 论文没有标准模块移除消融；Table 5 是 Online 与 Revisit 的状态分析，不应误写成完整架构消融。
- unseen-region 能力仅提供定性 Figure 6，没有几何误差的定量表。
