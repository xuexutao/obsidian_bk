# Sources

## 论文、项目与模型

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v2） | https://arxiv.org/pdf/2608.02980 | 全文，21 页 | 2026-10-28 | 精读主来源 |
| `sources/qwen2.5-vl-3b-config.json` | Qwen2.5-VL-3B 官方配置 | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/config.json | 基础模型配置 | 2026-10-28 | 核对 hidden size、层数、注意力头数和 ViT 配置 |
| — | arXiv 摘要页 | https://arxiv.org/abs/2608.02980 | 元信息、版本历史 | 2026-10-28 | 标题、作者、DOI、版本信息 |
| — | 官方项目页 | https://qwen-3d.github.io/ | 方法概览与结果展示 | 2026-10-28 | 交叉核对资源链接与主要结论 |
| — | 官方代码仓库 | https://github.com/ll220/qwen3d | commit `bf91573212c3538fc314d2add05519d7e9fd6b7e` | 2026-10-28 | 代码级实现与张量流核查 |
| — | 官方模型权重 | https://huggingface.co/katefgroup/Qwen-3D | 3B / 7B checkpoints | 2026-10-28 | 权重可用性与许可证核查 |
| — | 官方 ScanNet ViT 特征缓存 | https://huggingface.co/datasets/katefgroup/qwen3d_cached_scannet_vit_features | 3B / 7B cache | 2026-10-28 | 训练存储成本与缓存格式核查 |
| — | ECCV 2026 论文页 | https://eccv.ecva.net/virtual/2026/poster/3901 | 会议接收信息 | 2026-10-28 | 会议状态核查 |

## 关键图表

| 本地文件 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|
| `figures/01-overview.png` | https://arxiv.org/pdf/2608.02980 | Figure 1, p.2 | 2026-10-28 | 任务范围与总体动机 |
| `figures/02-decoding-comparison.png` | https://arxiv.org/pdf/2608.02980 | Figure 2, p.3 | 2026-10-28 | 四种 3D LMM 解码接口对比 |
| `figures/03-method-architecture.png` | https://arxiv.org/pdf/2608.02980 | Figure 3, p.5 | 2026-10-28 | 完整方法 pipeline |
| `figures/04-ablation-table.png` | https://arxiv.org/pdf/2608.02980 | Table 5, p.9 | 2026-10-28 | 关键设计消融 |
| `figures/05-failure-cases.png` | https://arxiv.org/pdf/2608.02980 | Figure 6, p.17 | 2026-10-28 | 失败模式分析 |
| `figures/06-grounding-results.png` | https://arxiv.org/pdf/2608.02980 | Figure 7, p.18 | 2026-10-28 | 3D 指称定位定性结果 |

所有图片均从官方 arXiv v2 PDF 以 300 DPI 完整裁剪，保留原始图注；没有使用 PDF 内嵌碎片图。

## 代码阅读记录

- 仓库：`https://github.com/ll220/qwen3d`
- 阅读 commit：`bf91573212c3538fc314d2add05519d7e9fd6b7e`（2026-10-12T15:26:48-04:00）
- 阅读日期：2026-10-28
- 仓库根目录未提供 `LICENSE`；第三方 `libs/` 算子注明来自 Pointcept（MIT），Hugging Face 模型权重和 ViT 缓存标注为 CC BY-NC 4.0。

重点核查文件：

- `README.md`
- `docs/RUN.md`
- `docs/DATA.md`
- `scripts/main_qwen.sh`
- `qwen3d/config.py`
- `qwen3d/configs/base.yaml`
- `qwen3d/configs/qwen_base.yaml`
- `qwen3d/configs/qwen_3d.yaml`
- `qwen3d/model.py`
- `qwen3d/model_data.py`
- `qwen3d/data_video/data_utils.py`
- `qwen3d/data_video/dataset_mapper_scannet.py`
- `qwen3d/modeling/backproject/backproject.py`
- `qwen3d/modeling_qwen2_5_vl_modified.py`
- `qwen3d/utils/rope.py`
- `qwen3d/modeling/transformer_decoder/simple_mask_decoder.py`
- `qwen3d/modeling/matcher.py`
- `qwen3d/modeling/criterion.py`

张量形状与代码—论文差异先由只读代码分析定位，再人工回读上述源文件、官方推荐命令与 Qwen 官方配置交叉核查。默认 3B 张量形状属于基于公开配置和代码控制流的静态推导，未通过完整 ScanNet 环境实际执行 forward 验证；笔记中已将其标注为“代码事实”或“推断”。
