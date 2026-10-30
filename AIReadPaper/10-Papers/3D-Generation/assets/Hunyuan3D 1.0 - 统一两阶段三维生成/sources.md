# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 / 版本 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2411.02293 | arXiv:2411.02293v5，2025-01-23 | 2026-10-30 | 精读主来源 |
| `figures/01-teaser.png` | 结果总览 | https://arxiv.org/pdf/2411.02293 | Figure 1，PDF p.1 | 2026-10-30 | 说明文生/图生 3D 的统一输出能力 |
| `figures/02-method-pipeline.png` | 方法总览 | https://arxiv.org/pdf/2411.02293 | Figure 2，PDF p.3 | 2026-10-30 | 两阶段 pipeline、Reference Attention、Adaptive CFG、SVRM 与 triplane |
| `figures/03-triplane-super-resolution.png` | 模块消融 | https://arxiv.org/pdf/2411.02293 | Figure 3，PDF p.3 | 2026-10-30 | 解释线性 unpatchify 对几何细节的作用 |
| `figures/04-qualitative-comparison.png` | 定性对比 | https://arxiv.org/pdf/2411.02293 | Figure 4，PDF p.6 | 2026-10-30 | 比较 OpenLRM、TripoSR、InstantMesh、Lite 与 Standard |
| `figures/05-adaptive-cfg-ablation.png` | Adaptive CFG 消融 | https://arxiv.org/pdf/2411.02293 | Figure 7，PDF p.7 | 2026-10-30 | 比较固定 CFG、时间自适应 CFG 与视角自适应 CFG |
| `figures/06-hybrid-input-ablation.png` | Hybrid Input 消融 | https://arxiv.org/pdf/2411.02293 | Figure 8，PDF p.7 | 2026-10-30 | 说明未标定条件图对顶部不可见区域的补偿 |
| `sources/svrm.yaml` | 官方 SVRM 配置 | https://github.com/Tencent-Hunyuan/Hunyuan3D-1/blob/454284503670312d4e06f6251c9be2f9f6d0fae7/svrm/configs/svrm.yaml | commit `454284503670312d4e06f6251c9be2f9f6d0fae7` | 2026-10-30 | 核验 DINO、Transformer、triplane 与渲染超参 |
| `sources/mvd-lite-unet-config.json` | 官方 Lite UNet 配置 | https://huggingface.co/tencent/Hunyuan3D-1/resolve/main/mvd_lite/unet/config.json | Hugging Face main | 2026-10-30 | 核验 SD2.1 路线通道、cross-attention 维度 |
| `sources/mvd-std-unet-config.json` | 官方 Standard UNet 配置 | https://huggingface.co/tencent/Hunyuan3D-1/resolve/main/mvd_std/unet/config.json | Hugging Face main | 2026-10-30 | 核验 SDXL 路线通道、cross-attention 维度 |
| `sources/mvd-lite-vae-config.json` | 官方 Lite VAE 配置 | https://huggingface.co/tencent/Hunyuan3D-1/resolve/main/mvd_lite/vae/config.json | Hugging Face main | 2026-10-30 | 核验 VAE 4 通道 latent 与缩放系数 |
| `sources/mvd-std-vae-config.json` | 官方 Standard VAE 配置 | https://huggingface.co/tencent/Hunyuan3D-1/resolve/main/mvd_std/vae/config.json | Hugging Face main | 2026-10-30 | 核验 VAE 4 通道 latent 与缩放系数 |

## 官方代码

- 仓库：https://github.com/Tencent-Hunyuan/Hunyuan3D-1
- 精读 commit：`454284503670312d4e06f6251c9be2f9f6d0fae7`（2025-11-19，`Update requirements.txt`）
- 开源范围：推理代码、权重下载说明、纹理映射与 Baking；截至 2026-10-30 未开源训练代码。
- 主要复核文件：
  - `main.py`
  - `infer/text_to_image.py`
  - `infer/removebg.py`
  - `infer/image_to_views.py`
  - `infer/views_to_mesh.py`
  - `mvd/hunyuan3d_mvd_lite_pipeline.py`
  - `mvd/hunyuan3d_mvd_std_pipeline.py`
  - `svrm/predictor.py`
  - `svrm/ldm/models/svrm.py`
  - `svrm/ldm/modules/encoders/dinov2_mod.py`
  - `svrm/ldm/modules/translator/img_to_triplane.py`
  - `svrm/ldm/modules/attention.py`
  - `svrm/ldm/modules/rendering_neus/synthesizer.py`
  - `third_party/mesh_baker.py`
  - `third_party/gen_baking.py`

## 补充材料说明

截至 2026-10-30，arXiv 页面与官方仓库未提供独立 Supplementary PDF。论文正文未披露的训练损失、优化器、batch size、训练步数、训练硬件与用户研究协议均在笔记中明确标为“未披露”，没有用代码常见做法补造。
