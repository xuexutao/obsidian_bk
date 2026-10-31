# GEN3C 来源清单

论文与图片获取于 **2026-10-15**；官方代码、权重 revision 与实际推理 shape 复核于 **2026-10-31**。图片均直接取自 arXiv 官方源码包并完整渲染，未从 PDF 页面截图、未裁切边缘。

## 论文与本地材料

| 本地文件 | 官方 URL | 源码内文件 | Figure / Page | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | https://arxiv.org/pdf/2503.03751 | — | 全文，16 页 | 论文 SVD 版方法、训练配置、实验数字与公式 |
| `sources/arxiv-source.tar` | https://arxiv.org/e-print/2503.03751 | — | 官方 TeX 源码包 | 高清原图、表格与公式溯源 |
| `figures/fig1_teaser.png` | https://arxiv.org/e-print/2503.03751 | `fig/teaser/teaser_fig_1114_3.pdf` | Figure 1 / Page 1 | 输入形态、长视频、Dolly Zoom 与驾驶编辑能力 |
| `figures/fig3_method.png` | https://arxiv.org/e-print/2503.03751 | `fig/method_compatible.pdf` | Figure 3 / Page 3 | 时空 3D cache、渲染、掩码与论文版扩散条件注入 |
| `figures/fig5_single_view_results.png` | https://arxiv.org/e-print/2503.03751 | `fig/single_view_img.pdf` | Figure 5 / Page 5 | 单图 NVS 与 CameraCtrl、NVS-Solver 定性比较 |
| `figures/fig9_fusion_ablation.png` | https://arxiv.org/e-print/2503.03751 | `fig/buffer_vs_point_crop.pdf` | Figure 10 / Page 8 | 论文版 feature pooling 对显式点云融合的消融 |

## 固定代码与权重版本

| 资源 | 固定版本 | 关键文件 | 用途 |
|---|---|---|---|
| [nv-tlabs/GEN3C](https://github.com/nv-tlabs/GEN3C/tree/db2ffe12ced12ddafcec5e0422ee46ce8520746b) | commit `db2ffe12ced12ddafcec5e0422ee46ce8520746b` | `gen3c_single_image.py`、`gen3c_dynamic.py`、`gen3c_multiview.py` | 单图/动态视频/稀疏多视图入口与真实调用顺序 |
| 同上 | 同一 commit | `inference/cache_3d.py`、`forward_warp_utils_pytorch.py` | depth 反投影、3D cache、top-2 buffer、目标相机 forward warp、cache 更新 |
| 同上 | 同一 commit | `inference/gen3c_pipeline.py`、`model/model_gen3c.py` | T5→seed latent→warp RGB/mask VAE→条件对象 |
| 同上 | 同一 commit | `config/inference/cosmos-1-diffusion-gen3c.py` | `latent_shape=[16,16,88,160]`、`frame_buffer_max=2`、81 通道条件配置 |
| 同上 | 同一 commit | `config/base/net.py`、`networks/general_dit_video_conditioned.py`、`networks/general_dit.py` | 7B DiT：4096 hidden、28 blocks、32 heads、`(1,2,2)` patch、条件 concat 与 T5 cross-attention |
| 同上 | 同一 commit | `config/base/tokenizer.py`、`module/pretrained_vae.py` | Cosmos tokenizer 的 16 channels、$8×8×8$ 压缩、121→16 时间公式与 decode |
| 同上 | 同一 commit | `model/model_v2w.py`、`model/model_t2w.py` | EDMEuler、35 steps、seed frame 写回、代码实际 CFG 公式 |
| [nvidia/GEN3C-Cosmos-7B](https://huggingface.co/nvidia/GEN3C-Cosmos-7B/tree/9bcfdb4f3924f41376daeadf6200826c12a3bf8e) | revision `9bcfdb4f3924f41376daeadf6200826c12a3bf8e` | `model.pt`、`config.json` | 官方 7B 权重；下载脚本校验 `model.pt` MD5=`38644bf823aa5272acef60cfad8bc0f7`；config 标记 `Cosmos_GEN3C` / `7b` |
| [Cosmos Predict1](https://github.com/nvidia-cosmos/cosmos-predict1) | 由上述固定 GEN3C 仓库内 vendor 代码为准 | Cosmos Tokenize1、T5-11B、EDM | 公开 checkpoint 的基础框架语义 |

## 外部几何组件与论文/代码边界

| 组件 | 出现场景 | 已核实结论 |
|---|---|---|
| MoGe v1 `Ruicheng/moge-vitl` | 公开单图推理 | 输出 depth、mask、intrinsics；代码固定 resize 720×1280，再适配到 704×1280 |
| ViPE | 公开动态视频推理 | README 推荐的 depth+intrinsics+w2c 预处理；与 GEN3C 使用不同环境 |
| VGGT | 公开多视图 README 建议 | 只是可选上游；GEN3C 实际读取预计算 NPZ，不在 DiT 内运行 VGGT |
| DROID-SLAM + Depth Anything V2 | 论文训练/评测 | 用于 RE10K/DL3DV/Waymo 论文配方，不是当前单图官方推理的实际调用 |
| SVD | 论文主实验骨干 | 论文的 feature max-pooling、25 steps、CFG 3 属于 SVD 版；公开 checkpoint 是 Cosmos 7B |

> **开放边界**：官方仓库开放完整 Cosmos 推理路径与 7B 权重，但没有 GEN3C 专用训练入口、论文 SVD checkpoint、驾驶专用 checkpoint、完整数据清洗/标注脚本。仓库通用 `training/` 目录不能当作论文训练配方已开源。

> **编号注**：源码文件名 `buffer_vs_point_crop.pdf` 在论文定稿中对应 Figure 10；笔记沿用语义化本地名 `fig9_fusion_ablation.png`，不把文件名中的数字当作论文 Figure 编号。
