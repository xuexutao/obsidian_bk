# 来源记录 — VGGT

## 官方论文

| 材料 | 说明 |
| --- | --- |
| `sources/paper.pdf` | VGGT 官方论文 PDF（arXiv 2503.11651） |
| 原始 URL | https://arxiv.org/abs/2503.11651 |
| 获取日期 | 2026-08-26 |
| 用途 | 全文精读、关键数据与图源 |

## 延伸精读：DPT

为展开正文 §2.4.1 / 第 5 节（dense head 与 DPT 谱系）而补充精读的论文。

| 材料 | 说明 |
| --- | --- |
| `sources/dpt_2103.13413.pdf` | DPT 官方论文 PDF《Vision Transformers for Dense Prediction》（Ranftl, Bochkovskiy, Koltun；ICCV 2021） |
| 原始 URL | https://arxiv.org/abs/2103.13413 |
| 获取日期 | 2026-08-26 |
| 用途 | Reassemble / Read 三变体 / Fusion 结构与消融数据（Table 1–9）的一手依据 |

## 关键图来源

| 图片 | 对应论文图 | 页码 | 用途 |
| --- | --- | --- | --- |
| `figures/fig1_teaser.png` | Figure 1（teaser） | Page 1 | 展示 VGGT 单次前馈同时输出相机/点图/深度/点轨迹的能力 |
| `figures/fig2_architecture.png` | Figure 2（Architecture Overview） | Page 3 | 展示 DINO patchify → 特殊 token → Alternating-Attention → 多头输出架构 |
| `figures/fig3_qualitative.png` | Figure 3（定性对比） | Page 4 | 多视图深度与点云重建的定性对比 |
| `figures/dpt_fig1_architecture.png` | DPT 论文 Figure 1（三联图） | Page 3（DPT PDF） | 展示 DPT 整体架构 + Reassemble 展开 + Fusion block 内部，支撑正文 §2.4.1 |

> VGGT 三张图均由官方 PDF 按图区坐标完整裁剪，非内嵌碎片。DPT 图同样由 DPT 官方 PDF 第 3 页按图区坐标（含 Left/Center/Right 三联）整体裁剪，400 dpi。

## 官方代码与配置

| 本地文件 | 材料类型 | 原始来源 | 版本 / 位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/code-version.txt` | 代码版本记录 | https://github.com/facebookresearch/vggt | commit `a288dd0f14786c93483e45524328726ab7b1b4ce` | 2026-10-31 | 固定代码阅读版本 |
| `sources/training-default.yaml` | 官方训练重实现默认配置 | https://github.com/facebookresearch/vggt/blob/main/training/config/default.yaml | 同上 commit | 2026-10-31 | 微调开关、优化器与 loss 口径 |
| `sources/training-default-dataset.yaml` | 官方数据模板 | https://github.com/facebookresearch/vggt/blob/main/training/config/default_dataset.yaml | 同上 commit | 2026-10-31 | 训练帧数、宽高比与 tracking query 配置 |

- 仓库：https://github.com/facebookresearch/vggt
- Commit：`a288dd0f14786c93483e45524328726ab7b1b4ce`
- Commit date：2026-05-18
- 阅读日期：2026-10-31

人工回读并核对的主要文件：

- `README.md`
- `vggt/models/vggt.py`
- `vggt/models/aggregator.py`
- `vggt/layers/vision_transformer.py`
- `vggt/layers/patch_embed.py`
- `vggt/layers/block.py`
- `vggt/layers/attention.py`
- `vggt/heads/camera_head.py`
- `vggt/heads/dpt_head.py`
- `vggt/heads/track_head.py`
- `vggt/heads/track_modules/base_track_predictor.py`
- `vggt/heads/track_modules/blocks.py`
- `vggt/utils/load_fn.py`
- `vggt/utils/pose_enc.py`
- `training/README.md`
- `training/loss.py`
- `training/trainer.py`

## 关键结论的人工复核记录

| 结论 | 一手代码位置 | 人工复核 |
|---|---|---|
| 518/14 得 37×37=1369 patch；每帧再加 1 camera + 4 register | `aggregator.py`、`patch_embed.py` | 已复核 |
| 24 轮中每轮都执行 frame + global 两个 block | `aggregator.py:241-261` | 已复核 |
| head latent 是 `cat(frame,global)` 后的 2048 维 | `aggregator.py:254-259` | 已复核 |
| camera 代码顺序为 `[T,q,FoV]`，4 次 refinement | `camera_head.py`、`pose_enc.py` | 已复核 |
| DPT 读取官方索引 `[4,11,17,23]` | `dpt_head.py` | 已复核 |
| depth/point 最终为 channels-last | `vggt.py`、`dpt_head.py`、`head_act.py` | 已复核 |
| tracking query frame 固定为第 0 帧 | `base_track_predictor.py` | 已复核 |
| 公开训练目录是 re-implementation，tracking loss 未接通 | `training/README.md`、`training/loss.py` | 已复核 |

未加载 1B checkpoint 执行完整 forward；笔记中的具体 tensor 数字以 `[B,S,H,W]=[1,3,518,518]` 为标准示例，由源码逐行静态推导。论文训练成本、精度与数据混合口径仍以官方 PDF 为准。
