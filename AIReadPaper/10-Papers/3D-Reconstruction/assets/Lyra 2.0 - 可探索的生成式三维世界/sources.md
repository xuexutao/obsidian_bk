# Lyra 2.0 来源与源码核验

核验日期：**2026-10-31**。图片和论文文件为既有资产；本次仅原地更新本清单与笔记。

## 1. 论文、项目、代码与权重

| 对象 | 官方 URL | 固定版本 | 用途 |
|---|---|---|---|
| 论文 | https://arxiv.org/abs/2604.13036 | arXiv 2604.13036；本地 `sources/paper.pdf`（25 页） | persistent world、memory、self-aug、实验 |
| arXiv 源码 | https://arxiv.org/e-print/2604.13036 | 本地 `sources/arxiv-source.tar` | TeX、图、表与附录 |
| 项目页 | https://research.nvidia.com/labs/sil/projects/lyra2/ | 官方 NVIDIA Research | 展示与官方入口 |
| 代码 | https://github.com/nv-tlabs/lyra/tree/main/Lyra-2 | commit `9fffc9adc37004091ecf26ef03abfb3abdf4d59a` | inference、training、GUI、recon |
| 权重 | https://huggingface.co/nvidia/Lyra-2.0 | revision `c178c3fcf12b63cf98f6749999e6ecb63901669f`，2026-07-15 | 主模型、VAE、编码器、recon、LoRA、sample data |

官方仓库该 revision 对论文仍使用 arXiv citation；本清单不为其补写未经官方入口确认的会议 venue。

## 2. HF revision 文件树摘要

通过 blobless git tree 在 revision `c178c3…` 核验：

```text
checkpoints/model/model/.metadata
checkpoints/model/model/__0_0.distcp ... __63_0.distcp
checkpoints/vae/vae.pth
checkpoints/vae/images_mean_std.pt
checkpoints/vae/video_mean_std.pt
checkpoints/image_encoder/model.pth
checkpoints/text_encoder/encoder.pth
checkpoints/text_encoder/negative_prompt.pt
checkpoints/recon/model.pt
checkpoints/lora/dmd_distillation.safetensors
checkpoints/lora/detail_enhancer.safetensors
checkpoints/lora/realism_boost.safetensors
assets/training_data/{rgb,pose,intrinsics,depth,framepack-caption-*}/...
config.json
README.md
```

`config.json` 仅声明 `input_types:["Lyra-2.0"]`、`model_size:"14B"`；真正可执行 config 在代码仓库 `lyra_2/_src/configs/`。模型卡版本 V1.0，基础模型 WAN-14B/Wan2.1。

许可证：源码 Apache-2.0；权重为 NVIDIA Internal Scientific Research and Development Model License，限制分发、部署、生产与公开使用等行为，详见模型卡与官方协议。

## 3. 关键源码入口与结论映射

以下路径均相对 `Lyra-2/`、commit `9fffc9a…`：

| 结论 | 源码/配置 |
|---|---|
| 有效 FramePack config | `lyra_2/_src/configs/experiment.py::register_lyra2`：`f1k1f4s2f1s1f16k4f2k2f1k1_g20` |
| 80 帧/20 latent | `lyra_2/_src/models/lyra2_model.py::_init_lyra2_metadata`，`num_frames_per_latent=4` |
| temporal 20 + spatial 5 + new 20 | 同 config 与 `_init_lyra2_metadata` 的 k/s 计数 |
| per-frame world points | `Sparse3DCache.add` |
| 当前 cache downsample=4 | `_tokenizing_video_to_latents` 与 AR inference 的 `Sparse3DCache` 实例 |
| 几何可见性检索 | `Sparse3DCache.retrieve`：投影、`scatter_reduce(amin)`、`depth_threshold=0.1` |
| 贪心最大覆盖 | `retrieve(max_coverage=True)` |
| canonical coords | `_build_canonical_spatial_coords`：`[N,3,H,W]`、xy/slot 均归一化到 `[-1,1]` |
| 4 通道对应转 latent | `_coord_pixels_to_latents`、`_pixelshuffle_hw_to_latent` |
| 每 slot 256 通道 | 常量 `LYRA2_CORRESPONDENCE_CHANNELS_PER_SLOT=4*8*8` |
| 5 slots 共 1280 通道 | effective config 的 4+1 spatial slots；`torch.cat(spatial_latents,dim=1)` |
| 最近帧 RGB depth warp | `_apply_camera_controls::_warp_multisrc` |
| Plücker 384 通道 | `_apply_camera_controls`：`6*8*8` pixel shuffle |
| spatial RGB image encoding | `_prepare_lyra2_inputs` 中 `self.encode(video[:,:,t:t+1])` |
| K/Q-only correspondence | config `inject_kq_only=True`；`networks/wan2pt1_lyra2.py::Lyra2AttentionBlock` |
| FramePack reshape | `WanModel._patchify_lyra2` |
| self-augmentation | effective config 与 `_tokenizing_video_to_latents` stage A/B |
| AR 流式 VAE | `lyra_2/_src/inference/lyra2_ar_inference.py` |
| 自定义轨迹格式 | `README.md` §Custom Trajectory |
| 3DGS 重建 | `vipe_da3_gs_recon.py`；默认 `checkpoints/recon/model.pt` |
| >500 帧分块重建 | `README.md` 与 `vipe_da3_chunked_gs_recon.py` |

## 4. Shape 核验口径

公开主分辨率 `480×832`：

- Wan VAE 空间 latent：`60×104`，channel 16；
- 80 个新增像素帧对应 20 video latent slices；完整轨迹长度须 `1+80k`；
- canonical + depth：每 slot `[B,4,80,480,832]`；时间取 `3,7,…,79` 后为 20；
- `8×8` pixel shuffle：每 slot `[B,256,20,60,104]`；5 slots concat `[B,1280,20,60,104]`；
- Plücker：`[B,T,H,W,6] → [B,384,T,60,104]`；
- patch 前 context slices：20 temporal + 5 spatial + 20 generation = 45；
- Wan base patch `(1,2,2)` 下 generation tokens：`20×30×52=31,200`，history 再按 k/s kernel 压缩。

## 5. 论文—代码差异记录

| 项目 | 论文口径 | 当前固定代码 |
|---|---|---|
| retrieval 点云下采样 | 8× | `Sparse3DCache(downsample=4)` |
| self-aug | 论文概率化描述、一次近似去噪 | `self_aug_prob=1.0`、`self_aug_every_k=2`、`self_aug_steps=1` |
| FramePack | 概念性 anchor/16/2/1/5 | 精确 `f1k1f4s2f1s1f16k4f2k2f1k1_g20` |
| mesh | 论文展示分层 mesh 与仿真 | README 的公开一键主路径清楚到 VIPE+DA3→3DGS PLY；未提供同等清晰的完整 mesh 一键流程 |
| 长度 | arbitrarily long | rollout/cache 可持续追加；实验有限，DA3 >500 帧建议 chunked |

## 6. 既有图片

| 本地文件 | 官方来源 | Figure / Page | 用途 |
|---|---|---|---|
| `figures/fig1_teaser.png` | https://arxiv.org/e-print/2604.13036；`figures/teaser_v3.pdf` | Figure 1 / Page 1 | 长程探索、回望、约 90m、仿真 |
| `figures/fig2_method.png` | 同上；`figures/method_v3.pdf` | Figure 2 / Page 6 | retrieve–generate–update 与对应注入 |
| `figures/fig5_video_results.png` | 同上；`figures/comparison_grid_all.png` | Figure 3 / Page 10 | 约第 800 帧定性比较 |
| `figures/fig7_ablation.png` | 同上；`figures/comparison_grid_ablations.png` | Figure 6 / Page 14 | memory/FramePack/self-aug 消融 |

图片原获取日期 2026-10-15；2026-10-31 校验文件存在并保留全部 Markdown 引用，未替换或裁剪。

## 7. 开放边界与复现注意

- 2026-07 revision 已含训练代码、GUI、sample training data 和推理；不应再写“training planned”。
- 公开 checkpoint 是 DCP 64 shards，训练还需单独下载 Wan2.1-I2V-14B diffusion shards并转换为约 33GB bf16 `.pth`；该基础权重有自己的许可。
- 训练文档要求 DL3DV-10K、VIPE metric depth/pose、80-frame captions 和 T5 embeddings严格帧对齐，但未公开完整训练片段清单、全部 captions/轨迹和 WorldScore 原始评分记录。
- persistent state 的数据结构可增长，不表示世界质量、GPU 内存、检索歧义可无限稳定。
- 2.0 针对静态场景相机运动；动态对象、物理尺度、碰撞和 mesh watertightness 无系统保证。
