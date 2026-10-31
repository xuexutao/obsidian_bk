# Lyra 1.0 来源与源码核验

核验日期：**2026-10-31**。图片和论文文件为既有资产；本次只修改清单与笔记，不新增文件。

## 1. 论文、代码、权重与数据

| 对象 | 官方 URL | 固定版本 | 用途 |
|---|---|---|---|
| 论文 | https://arxiv.org/abs/2509.19296 | arXiv 2509.19296；本地 `sources/paper.pdf`（29 页） | 方法、公式、实验、附录 |
| arXiv 源码 | https://arxiv.org/e-print/2509.19296 | 本地 `sources/arxiv-source.tar` | TeX、高清图片、表格 |
| 代码 | https://github.com/nv-tlabs/lyra/tree/main/Lyra-1 | commit `9fffc9adc37004091ecf26ef03abfb3abdf4d59a` | 静态/动态训练与推理 |
| 权重 | https://huggingface.co/nvidia/Lyra | revision `240ecc75fedadd3d0056666c679d82348903806e` | static/dynamic 3DGS decoder |
| 合成数据 | https://huggingface.co/datasets/nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG | revision `45bc128fdf3a98053aeb4b6ee1113d4ec429552b` | static/dynamic SDG 训练集 |
| 项目页 | https://research.nvidia.com/labs/toronto-ai/lyra/ | 官方 NVIDIA Research | 展示与官方入口 |

代码 revision 由 `git ls-remote` 与浅克隆 `git rev-parse HEAD` 双重核验；最新审阅提交时间 2026-07-20。

## 2. 权重与 config 核验

HF revision `240ecc…` 文件树：

```text
.gitattributes
README.md
config.json
lyra_dynamic.pt
lyra_static.pt
```

官方 `scripts/download_lyra_checkpoints.py`：

| 文件 | 本地目标 | MD5 |
|---|---|---|
| `lyra_static.pt` | `checkpoints/Lyra/lyra_static.pt` | `28ddd05011b33350c70edab02c23c602` |
| `lyra_dynamic.pt` | `checkpoints/Lyra/lyra_dynamic.pt` | `91c1b3920332287ed161bc47a1fd7077` |

模型卡：V1.0、基于 Cosmos Predict 1、32.75M 参数；权重许可为 NVIDIA Open Model License。源码为 Apache-2.0。

## 3. 源码入口与结论映射

以下路径均相对 `Lyra-1/`、commit `9fffc9a…`：

| 结论 | 源码/配置 |
|---|---|
| 单图多轨迹 GEN3C | `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py` |
| 动态 GEN3C 与 flip supervision | `cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py` |
| 静态 demo 权重/config | `configs/demo/lyra_static.yaml` |
| 动态 demo 权重/目标时间 | `configs/demo/lyra_dynamic.yaml` |
| VAE 16 通道、时空压缩 8× | `configs/training/default.yaml`：`num_latent_c:16`、`latent_time_compression:8`、`latent_spat_compression:8` |
| 16 层、512 hidden、2×2 patch | 同 config：`enc_depth:16`、`enc_embed_dim:512`、`patch_size:2` |
| 1 Transformer + 7 Mamba-2 | `src/models/utils/model.py::get_model_blocks`、`src/models/utils/mamba2.py` |
| `B(VT)↔(BV)T` reshape | `src/models/recon/model_latent_recon.py::reshape_mv_*` |
| latent/camera/time patch | `LatentRecon.__init__`、`forward_gaussians` |
| raw 12D → final 14D | `configs/training/default.yaml::output_dims:12`；`LatentRecon.gaussian_processing` |
| deconv 16→121 | `LatentRecon.__init__` 的 kernel/stride/padding 计算 |
| opacity prune 80% | dynamic/static prune configs：`gaussians_prune_ratio:0.8`；`gaussian_pruning` |
| 动态 6 输入轨迹、12 target views | `3dgs_res_704_1280_views_121_multi_6_dynamic_prune.yaml`：`num_views:133`、`num_input_views:121`、`num_input_multi_views:6` |
| render RGB/depth/alpha | `src/rendering/gs.py`、`gs_deferred.py` |

## 4. Shape 核验口径

full config `V=6,L=121,H=704,W=1280,C=16`：

- VAE：`L'=(121-1)/8+1=16`，`h=88,w=160`；
- 每视角 patch tokens：`16×44×80=56,320`；
- 联合 tokens：`6×56,320=337,920`；
- deconv 输出网格每视角：`121×88×160=1,703,680`；
- 6 视角候选：`10,222,080`；
- prune 80%：约 `2,044,416`。

最终 14D 顺序按代码为 `xyz(3), opacity(1), scale(3), rotation(4), rgb(3)`。论文图的“14D”与 config `output_dims:12` 的差别来自 xyz 由 ray depth 计算，而非 head 直接回归。

## 5. 既有图片

| 本地文件 | 官方来源 | Figure / Page | 用途 |
|---|---|---|---|
| `figures/fig1_teaser.png` | https://arxiv.org/e-print/2509.19296；`imgs/teaser.pdf` | Figure 1 / Page 1 | 静态 3D / 动态 4D |
| `figures/fig3_method.png` | 同上；`imgs/method.pdf` | Figure 4 / Page 5 | teacher-student 与 decoder |
| `figures/fig4_main_results.png` | 同上；`imgs/main_results.png` | Figure 6 / Page 8 | 新视角结果 |
| `figures/fig8_depth_ablation.png` | 同上；`imgs/ablation_depth.pdf` | Figure 11 / Page 26 | depth loss 消融 |

图片原获取日期 2026-10-15；2026-10-31 校验文件存在、Markdown 引用保留，无切边修改。

## 6. 证据边界

- 公开仓库包含 decoder 训练/推理、GEN3C 分支、progressive configs、权重下载和训练数据入口，足以审计核心 shape 与 reshape。
- bitwise 复现仍缺 FLUX prompt/seed 全表、合成数据完整过滤规则、所有外部基础模型固定 revision、ViPE checkpoint hash。
- 论文主表部分 baseline 使用对方报告值；不能视为统一代码、统一硬件的完全公平重跑。
- 最终 3D 是教师视频的可渲染一致解释，不是有真实 3D GT 保证的度量重建。
