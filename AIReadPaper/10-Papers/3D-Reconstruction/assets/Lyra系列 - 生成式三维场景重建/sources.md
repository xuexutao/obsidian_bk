# sources — Lyra 系列：生成式三维场景重建

核验日期：**2026-10-31**。本文件记录系列总览所用的不可变 revision、版本边界和既有图片来源；未新增资产。

## 1. 官方论文与项目

| 对象 | 官方 URL | 固定版本/状态 | 用途 |
|---|---|---|---|
| Lyra 1.0 论文 | https://arxiv.org/abs/2509.19296 | arXiv 2509.19296；本地既有 PDF `sources/Lyra1_2509.19296.pdf` | 自蒸馏、3DGS decoder、动态 4D、实验 |
| Lyra 2.0 论文 | https://arxiv.org/abs/2604.13036 | arXiv 2604.13036；本地既有 PDF 条目曾标下载中断，正文以 arXiv/官方源码与独立 2.0 本地 PDF 交叉核验 | persistent world、spatial memory、self-augmentation、DA3 |
| 1.0 项目页 | https://research.nvidia.com/labs/toronto-ai/lyra/ | 官方 NVIDIA Research | 论文、展示、代码入口 |
| 2.0 项目页 | https://research.nvidia.com/labs/sil/projects/lyra2/ | 官方 NVIDIA Research | 论文、展示、代码入口 |
| 系列仓库 | https://github.com/nv-tlabs/lyra | `9fffc9adc37004091ecf26ef03abfb3abdf4d59a`，提交时间 2026-07-20 | 两代代码基线、README、配置、许可 |

远端核验命令：`git ls-remote https://github.com/nv-tlabs/lyra.git`。总览明确区分 `Lyra-1/` 与 `Lyra-2/`，不把 2.0 当作 1.0 decoder 的增量版本。

## 2. 权重、数据 revision 与许可

| 仓库 | revision | 已核验内容 | 许可/边界 |
|---|---|---|---|
| https://huggingface.co/nvidia/Lyra | `240ecc75fedadd3d0056666c679d82348903806e` | `.gitattributes`、`README.md`、`config.json`、`lyra_static.pt`、`lyra_dynamic.pt` | NVIDIA Open Model License；模型卡称可商用，仍须遵守其条款 |
| https://huggingface.co/nvidia/Lyra-2.0 | `c178c3fcf12b63cf98f6749999e6ecb63901669f` | DCP 主模型、VAE、image/text encoder、recon、3 个 LoRA、sample training data | NVIDIA Internal Scientific Research and Development Model License；限制部署、分发和生产使用 |
| https://huggingface.co/datasets/nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG | `45bc128fdf3a98053aeb4b6ee1113d4ec429552b` | 1.0 静态/动态 SDG 数据仓库 | 数据条款需独立核验 |

源码根目录及 `Lyra-2/LICENSE` 为 Apache-2.0；**源码许可不覆盖模型权重**。

## 3. 系列代码基线证据

| 结论 | 代码/文档位置（均相对 commit `9fffc9a…`） |
|---|---|
| 1.0 基于 Cosmos/GEN3C，直接解码 3DGS | `Lyra-1/README.md`；`src/models/recon/model_latent_recon.py` |
| 1.0 static/dynamic 独立 checkpoint | `Lyra-1/configs/demo/lyra_static.yaml`、`lyra_dynamic.yaml` |
| 1.0 权重文件与 MD5 | `Lyra-1/scripts/download_lyra_checkpoints.py` |
| 2.0 基于 Wan2.1、FramePack、DA3 | `Lyra-2/README.md`、`TRAINING.md` |
| 2.0 persistent state / retrieval | `Lyra-2/lyra_2/_src/models/lyra2_model.py::Sparse3DCache` |
| 2.0 有效上下文 config | `Lyra-2/lyra_2/_src/configs/experiment.py::register_lyra2` |
| 2.0 3D 重建不沿用 1.0 decoder | `Lyra-2/lyra_2/_src/inference/vipe_da3_gs_recon.py` |
| 2.0 长视频建议 chunked DA3 | `Lyra-2/README.md` §3D Gaussian Splatting Reconstruction |

## 4. 既有图片映射

| 本地文件 | 原始 URL / 来源 | 论文位置 | 用途 |
|---|---|---|---|
| `figures/lyra1_teaser.png` | https://arxiv.org/html/2509.19296v1/teaser.png | Lyra 1.0 Fig.1 | 单图→3DGS / 视频→4D |
| `figures/lyra1_method.png` | https://arxiv.org/html/2509.19296v1/method.png | 1.0 method figure | RGB teacher / 3DGS student |
| `figures/lyra1_dynamic_data_augmentation.png` | https://arxiv.org/html/2509.19296v1/dynamic_data_augmentation.png | 1.0 dynamic extension | 倒放监督 |
| `figures/lyra1_sota_static_comparison.png` | https://arxiv.org/html/2509.19296v1/sota_static_comparison.png | 1.0 附录 | 静态结果 |
| `figures/lyra1_ablation_depth.png` | https://arxiv.org/html/2509.19296v1/ablation_depth.png | 1.0 附录 | depth loss 消融 |
| `figures/lyra2_fig1_teaser.png` | https://arxiv.org/html/2604.13036v1 | Lyra 2.0 Fig.1；既有资产复用 | 长程探索与回访 |
| `figures/lyra2_fig2_pipeline.png` | https://arxiv.org/html/2604.13036v1 | Lyra 2.0 pipeline；既有资产复用 | geometry routing |

原始获取日期沿用旧记录：1.0 图片 2026-10-10；2.0 图片 2026-08-20 首次获取、2026-10-10 复制。2026-10-31 仅校验引用和文件存在性，未替换图片。

## 5. 总览的证据边界

- 1.0 关键训练/推理/config/权重/数据入口均公开，但依赖 Cosmos、GEN3C、ViPE、FLUX 等外部组件；这些组件的精确权重 revision 不由 Lyra 单一仓库锁定。
- 2.0 在 2026-07 revision 已公开 GUI 和训练代码，不再适合描述为“仅推理代码”；但论文展示的完整分层 mesh/仿真后处理没有与 3DGS PLY 同等清晰的一键公开主路径。
- “arbitrarily long”表示 rollout/cache 接口可继续扩展；公开实验、GPU 内存、检索歧义与 DA3 长序列能力并未证明无限质量稳定。
- 2.0 官方仓库/模型卡在核验版本只给 arXiv citation；总览不写未经官方入口确认的正式 venue。
