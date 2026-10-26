# Sources

## 图片

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `figures/01-overview.png` | arXiv 源码原图 `pics/overview.pdf` | https://arxiv.org/e-print/2604.14268 | Figure 2，PDF p.5 | 2026-10-26 | 1.1 四阶段总览 |
| `figures/02-hypano-arch.png` | 源码原图 `pics/pano_arch.pdf` | 同上 | Figure 3，PDF p.7 | 2026-10-26 | 2.2 HY-Pano 架构与接缝处理 |
| `figures/03-scene-parsing.png` | 源码原图 `pics/scene_parsing.pdf` | 同上 | Figure 4，PDF p.7 | 2026-10-26 | 3.1 场景解析 |
| `figures/04-worldnav-trajectories.png` | 源码原图 `pics/traj_plan.pdf` | 同上 | Figure 5，PDF p.8 | 2026-10-26 | 3.2 五类轨迹 |
| `figures/05-worldstereo-training-stages.png` | 源码原图 `pics/worldexpand_training_stage.pdf` | 同上 | Figure 6，PDF p.9 | 2026-10-26 | 4 三段训练 |
| `figures/06-worldstereo-pipeline.png` | 源码原图 `pics/worldexpand_overview.pdf` | 同上 | Figure 7，PDF p.10 | 2026-10-26 | 4 WorldStereo 总体流程 |
| `figures/07-keyframe-vae.png` | 源码原图 `pics/keyframe-vae.pdf` | 同上 | Figure 9，PDF p.11 | 2026-10-26 | 4.2 Keyframe-VAE 形状 |
| `figures/08-memory-training-data.png` | 源码原图 `pics/training_memory.pdf` | 同上 | Figure 10，PDF p.12 | 2026-10-26 | 4.6 记忆训练数据 |
| `figures/09-ssm-rope.png` | 源码原图 `pics/ssm.pdf` | 同上 | Figure 11，PDF p.13 | 2026-10-26 | 4.5 SSM++ RoPE |
| `figures/10-worldmirror-arch.png` | 源码原图 `pics/worldmirror_overview.pdf` | 同上 | Figure 12，PDF p.15 | 2026-10-26 | 5.1 WorldMirror 架构 |
| `figures/11-normalized-rope.png` | 源码原图 `pics/rope_combined.pdf` | 同上 | Figure 13，PDF p.17 | 2026-10-26 | 5.3 归一化 RoPE |
| `figures/12-depth-alignment.png` | 源码原图 `pics/depth_alignment.pdf` | 同上 | Figure 14，PDF p.19 | 2026-10-26 | 6.1 深度对齐 |
| `figures/13-recon-compare.png` | 源码原图 `pics/gen_pcd_compare.pdf` | 同上 | Figure 15，PDF p.20 | 2026-10-26 | 5.2 生成帧重建对比 |

全部图片由 `research-field-survey/scripts/extract_source_figures.py` 从 arXiv LaTeX 源码中的矢量 PDF 转为 PNG（限宽 2400），并抽查 02/06/07/09/12 确认无裁边。

## 官方配置文件（`sources/`）

| 本地文件 | 原始来源 | 获取日期 | 用途 |
|---|---|---|---|
| `sources/hy-pano-2.0_config.json` | https://huggingface.co/tencent/HY-World-2.0/blob/main/HY-Pano-2.0/config.json | 2026-10-26 | HunyuanImage-3 层数、MoE、VAE、SigLIP2 配置 |
| `sources/hy-pano-2.0_generation_config.json` | https://huggingface.co/tencent/HY-World-2.0/blob/main/HY-Pano-2.0/generation_config.json | 2026-10-26 | 50 步、CFG 4.0、flow_shift 6.5、think_recaption |
| `sources/worldstereo-memory-dmd_config.json` | https://huggingface.co/hanshanxue/WorldStereo/blob/main/worldstereo-memory-dmd/config.json | 2026-10-26 | Wan2.1-I2V-14B 基座、21 帧、ControlNet、DMD 超参 |
| `sources/worldstereo-memory_config.json` | https://huggingface.co/hanshanxue/WorldStereo/blob/main/worldstereo-memory/config.json | 2026-10-26 | 记忆模型训练超参 |
| `sources/worldmirror-2.0_config.json` | https://huggingface.co/tencent/HY-World-2.0/blob/main/HY-WorldMirror-2.0/config.json | 2026-10-26 | DINOv2-L、24 层、归一化 RoPE、先验策略 |

## 论文与代码

- 论文 PDF：未在本综述重复保存，见单篇精读附件 `10-Papers/World-Models/assets/HY-World 2.0 - 多模态三维世界生成重建与模拟/sources/paper.pdf`（arXiv 2604.14268v1）。
- 官方代码：https://github.com/Tencent-Hunyuan/HY-World-2.0 ，阅读版本 commit `df9988efb87bfc0f4947eb3889411cf957478b06`（2026-08-13）。主要阅读文件：
  - `hyworld2/panogen/pipeline.py`、`pipeline_with_qwen_image.py`、`qwen_image/pipeline_qwen_pano.py`、`hunyuan_image_3/{configuration,image_processor,hunyuan_image_3_pipeline}.py`
  - `hyworld2/worldgen/traj_generate.py`、`traj_render.py`、`video_gen.py`、`gen_gs_data.py`、`world_gs_trainer.py`、`gs/extract_mesh.py`
  - `hyworld2/worldgen/src/{panorama_utils,retrieval_wm,data_utils,vae_utils,pointcloud,depth_alignment}.py`
  - `hyworld2/worldgen/models/{worldstereo,worldstereo_wrapper,controlnet,attention,camera,dmd_scheduler}.py`、`models/pipelines/*.py`
  - `hyworld2/worldrecon/pipeline.py`、`hyworldmirror/models/models/{worldmirror,visual_transformer,rasterization}.py`、`heads/*`、`layers/norm_rope.py`、`utils/{priors,inference_utils}.py`
- 权重目录体积（HF API，2026-10-26）：`HY-Pano-2.0/` 合计 169.4 GB；`HY-WorldMirror-2.0/model.safetensors` 5.05 GB；`hanshanxue/WorldStereo` 的 memory-dmd / memory / camera 检查点分别 34.9 / 22.3 / 10.9 GB。

## 核验说明

- 文中“代码”类陈述均对照上述 commit 的源码；“推断”类陈述（如 token 数、参数量换算、上极区采样密度的动机）已在正文中标注。
- 论文与代码的差异（潜空间 circular padding 未见于开源推理代码、对齐锚点数量与分布、SSM++ 为 RoPE 虚拟拼接、检索上限为 8）均基于该 commit；后续版本可能变化。
- 部分细节由代码阅读子代理辅助检索（3DGS 训练超参、WorldMirror 输出形状），关键数字已人工回读源码复核（3DGS 损失调度、WorldMirror 先验编码、生成链调用参数）。
