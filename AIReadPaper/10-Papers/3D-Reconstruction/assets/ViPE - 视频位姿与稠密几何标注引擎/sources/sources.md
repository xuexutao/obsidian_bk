# ViPE 来源清单

获取与核验日期：2026-10-15

## 官方材料

| 本地文件 | 官方来源 | SHA-256 | 用途 |
|---|---|---|---|
| `paper.pdf` | https://arxiv.org/pdf/2508.10934 | `f5401b8c2ca5be4d11c7301268eaa79a247f3710dfd6693f549a24aed12a40a9` | 21 页论文全文；方法、公式、实验与数据规模主来源 |
| `arxiv-source.tar` | https://arxiv.org/e-print/2508.10934 | `a5cca57b4e6b1dfc13143858c38b49bfc02f8824fbec1ae41102383e2257e871` | 官方 TeX 源码与矢量原图 |

补充官方页面：

- arXiv 摘要页：https://arxiv.org/abs/2508.10934
- NVIDIA 项目页：https://research.nvidia.com/labs/toronto-ai/vipe/
- 官方代码仓库：https://github.com/nv-tlabs/vipe
- 代码许可证：https://github.com/nv-tlabs/vipe/blob/main/LICENSE

## 原图与逐图验证

下列 PNG 均由 `arxiv-source.tar` 中的官方矢量 PDF 直接以 300 DPI 渲染，不使用 PDF 内嵌碎片图，也未改变图中内容。

| 本地图片 | 官方源码文件 | Figure / PDF page | 分辨率 | Read 验证结果 |
|---|---|---|---:|---|
| `../figures/01-teaser.png` | `figures/teaser.pdf` | Figure 1 / p.1 | 12404×4488 | 完整；可辨认输入视频、相机/3D 场景、metric depth，覆盖透视/室内动态/360° |
| `../figures/02-pipeline.png` | `figures/method.pdf` | Figure 2 / p.4 | 6025×2250 | 完整；动态掩码、关键帧图、dense/sparse/depth BA、smooth alignment 均清晰 |
| `../figures/03-camera-models.png` | `figures/qual-mei.pdf` | Figure 3 / p.7 | 3897×1300 | 完整；针孔错误结果、统一广角模型、原帧和矫正帧四组标记可辨认 |
| `../figures/04-kitti-seq02.png` | `figures/kitti-02.pdf` | Figure 4 左半 / p.10 | 1772×1195 | 完整；ViPE scaled/metric、MegaSAM scaled/raw 与 GT 图例和轨迹均清晰 |
| `../figures/05-kitti-seq07.png` | `figures/kitti-07.pdf` | Figure 4 右半 / p.10 | 1772×1195 | 完整；Sequence 07 两组轨迹和尺度差异清晰 |
| `../figures/06-wild-sdg-dataset.png` | `figures/qual-sdg.pdf` | Figure 7 / p.14 | 2363×3072 | 完整；6 组生成视频的 3D scene、input、depth 均完整可见 |

## 内容核验说明

- 论文原始配置的速度为单张 NVIDIA RTX 5090、640×480 输入时 3–5 FPS；TUM 表中 runtime 在可能情况下排除了逐帧稠密深度估计。
- “TUM/KITTI 提升 18%/50%”来自摘要的总体概括；正文笔记同时保留表格原始指标，避免误写成所有序列和指标均提升同一比例。
- 数据规模按论文正文记录：DynPose-100K++ 为 99,501 段、15.7M 帧、约 150 小时；Wild-SDG-1M 约 1M 段、78M 帧；Web360 约 2K 段；总计约 96M 帧。
- Lyra 1.0、Lyra 2.0 与 WorldPlay 的具体使用方式交叉核对了 Vault 中对应精读：Lyra 1.0 消费 ViPE 深度监督；Lyra 2.0 使用 ViPE pose、DAv3 depth；WorldPlay 使用 ViPE pose 和 pose 跳变过滤。
- 许可证按 2026-10-15 官方仓库页面核验：ViPE 主体代码为 Apache-2.0；README 对 Unik3D 部分另标 BY-NC-SA 4.0；第三方模型和发布数据不能自动继承主体代码许可证，须分别核对上游/下载页条款。
