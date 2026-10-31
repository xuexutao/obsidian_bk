---
title: "Lyra 2.0 - 可探索的生成式三维世界"
aliases:
  - "Lyra 2.0"
  - "Explorable Generative 3D Worlds"
authors:
  - Tianchang Shen
  - Sherwin Bahmani
  - Kai He
  - Sangeetha Grama Srinivasan
  - Tianshi Cao
  - Jiawei Ren
  - Ruilong Li
  - Zian Wang
  - Nicholas Sharp
  - Zan Gojcic
  - Sanja Fidler
  - Jiahui Huang
  - Huan Ling
  - Jun Gao
  - Xuanchi Ren
year: 2026
venue: arXiv preprint
arxiv: "2604.13036"
tags:
  - paper
  - 3d-reconstruction
  - world-generation
  - long-video
  - spatial-memory
  - 3dgs
status: source-verified
created: 2026-10-15
updated: 2026-10-31
source: "https://arxiv.org/abs/2604.13036"
---

# Lyra 2.0：可探索的生成式三维世界

## 0. 一句话结论、版本与证据边界

> Lyra 2.0 的 persistent world state 不是融合后的全局 3DGS，而是持续增长的历史 RGB、逐帧独立 depth/pose/point cache、FramePack 时间 slots 与最多 5 个几何检索空间 slots；几何只决定“找哪帧、目标像素该看哪个历史位置”，Wan2.1 扩散先验决定外观，最后再用 VIPE + DA3 把长视频重建成 3DGS。

本文固定审阅：

- 官方代码 `nv-tlabs/lyra` commit **`9fffc9adc37004091ecf26ef03abfb3abdf4d59a`**；
- 官方权重 `nvidia/Lyra-2.0` revision **`c178c3fcf12b63cf98f6749999e6ecb63901669f`**；
- 该 revision 的权重树含 DCP 主模型、Wan VAE、image/text encoder、`checkpoints/recon/model.pt`、`dmd_distillation.safetensors`、`detail_enhancer.safetensors`、`realism_boost.safetensors` 与两份 sample training data。

源码 Apache-2.0；权重是 **NVIDIA Internal Scientific Research and Development Model License**，明确不等同于开源生产许可。官方仓库当前只把论文列为 arXiv；因此本笔记不保留未经官方入口确认的会议录用信息。

![Lyra 2.0 总览](assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig1_teaser.png)

## 1. 模块入口表

| 模块 | 公开入口 | 作用 |
|---|---|---|
| 有效实验 config | `Lyra-2/lyra_2/_src/configs/experiment.py::register_lyra2` | FramePack 布局、memory、self-aug、可训练模块 |
| 总模型 | `_src/models/lyra2_model.py::Lyra2Model` | VAE、history 选择、camera condition、self-aug、AR 输入组织 |
| 持久状态 | 同文件 `Sparse3DCache` | per-frame point cache、可见性评分、贪心最大覆盖检索 |
| canonical warp | `Lyra2Model._apply_camera_controls` | `(u,v,slot,depth)` 前向 warp 到 80 个目标帧 |
| 上下文组装 | `_prepare_lyra2_inputs` | temporal slots + spatial RGB slots + 20 new latent slots |
| DiT | `_src/networks/wan2pt1_lyra2.py::WanModel` | 14B Wan 主干、FramePack patchify |
| 对应注入 | 同文件 `Lyra2AttentionBlock` | correspondence encoder，只注入 K/Q |
| AR 推理 | `_src/inference/lyra2_ar_inference.py` | streaming VAE cache、chunk rollout、memory update |
| preset/custom | `lyra2_zoomgs_inference.py`、`lyra2_custom_traj_inference.py` | 轨迹入口；长度须 `1+80k` |
| depth | `_src/inference/depth_utils.py` | 生成帧深度与 DA3 加载 |
| 3DGS 重建 | `vipe_da3_gs_recon.py` | 单次 VIPE pose + DA3 Gaussian recon |
| 长视频重建 | `vipe_da3_chunked_gs_recon.py` | overlapping chunks、对齐与合并 |
| GUI | `Lyra-2/gui/` | 交互规划相机并持续扩展 |

## 2. 全链路表：seed/camera → persistent rollout → 3DGS

以公开默认分辨率 `H×W=480×832`、单次新增 `F=80` 帧为例；batch 为 `B`。

| 顺序 | 操作 | 关键 shape | 原因 |
|---:|---|---|---|
| 1 | seed image | `[B,3,1,480,832]` | 世界初始观测与 anchor |
| 2 | 用户轨迹 | `w2c [B,N,4,4]`、`K [B,N,3,3]`；`N=1+80k` | 每次消费下一段 80 帧；1 是 chunk 边界共享帧 |
| 3 | 最近帧 depth warp | RGB/depth 从 buffer frame warp 到 80 targets：`[B,3,80,H,W]` | 提供局部精确像素/相机控制 |
| 4 | Plücker | `[B,T,H,W,6] → [B,384,T,H/8,W/8]` | `8×8` pixel shuffle：`6·64=384` 通道；无覆盖处仍有射线条件 |
| 5 | Wan causal VAE | 新 80 帧 ↔ 20 video latents；空间 `[H,W]→[60,104]`，通道 16 | 时间压缩 4×、空间 8×；流式 cache 避免重编码全部历史 |
| 6 | 时间 history | 20 个 temporal history latent slices | FramePack 保留远粗近细的时间连续性 |
| 7 | 几何检索 | 所有候选 per-frame points → 最多 5 帧 | 回访按 3D overlap，而非时间距离 |
| 8 | spatial RGB slots | 5 张检索图分别 image encode：每帧 `[B,16,1,60,104]` | 保留未 warp 的真实/生成外观 value |
| 9 | canonical correspondence | 每 slot `[B,3,80,H,W]` + depth → `[B,256,20,60,104]`；5 slots concat → `[B,1280,20,60,104]` | 每 4 帧采样，`4 channels·8·8=256`；只传几何对应，不传 warped RGB |
| 10 | FramePack context | 20 temporal + 5 spatial + 20 new latent slices | 固定 attention 预算承载可变长历史 |
| 11 | Wan DiT flow matching | patch tokens → 20 个新 latent | 外观由 14B 生成先验合成；correspondence 只改 K/Q |
| 12 | VAE decode | 20 latent → 80 新 RGB frames | causal feature cache 保持 chunk 边界连续 |
| 13 | update state | 新 RGB/depth/pose/points 加入 cache | persistent/explorable 的真正状态更新 |
| 14 | world rollout | 重复 retrieve → generate → update | 数据结构可持续增长，质量不保证无限不退化 |
| 15 | 3D recon | 长视频 `[T,H,W,3]` + VIPE pose → DA3 Gaussians → `.ply` | 生成与最终显式重建分离 |
| 16 | render/mesh | Gaussian render；论文另有分层 mesh | 3DGS 实时渲染；mesh 开放程度见证据边界 |

## 3. persistent/explorable world state 到底存什么

### 3.1 四类状态

1. **历史 RGB/视频 latent**：FramePack 的时间上下文；保持局部时间连续和风格。
2. **每帧独立 depth + `w2c/K`**：用于反投影、检索和 correspondence；不融合成全局点云。
3. **`Sparse3DCache._world_points`**：每条 entry `[B,H/d,W/d,3]`；当前代码实例 `downsample=4`。
4. **VAE encoder/decoder causal cache**：让下一 chunk 只编码/解码新增部分。

这与“维护一个越来越大的 3DGS 世界并从中渲染条件”不同。Lyra 2.0 故意不让错误深度成为永久融合几何；每帧错误被限制在自己的 entry，检索时由覆盖和多帧选择缓冲。

### 3.2 `Sparse3DCache.add`

```text
depth [B,1,H,W]
→ stride d subsample [B,1,H/d,W/d]
→ K 同比例缩放
→ unproject(depth,w2c,K)
→ world_points [B,H/d,W/d,3]
→ append；可选保留 full-res depth/w2c/K
```

论文描述检索点云下采样 8×，当前审阅代码在训练/推理路径实例化为 `downsample=4`。这是明确的论文—代码差异，复现报告必须写 revision，而不能只抄论文数字。

## 4. geometry-aware retrieval：从候选帧到 5 个空间 slots

### 4.1 投影与遮挡

对每个候选帧 `i` 的 world points 和一个或多个目标相机：

1. stack 为 `[C,B,H',W',3]`；
2. 齐次化并广播到 `V_target`，得到 camera points `[V,C,B,H',W',3]`；
3. 用缩放后的 `K` 投影到目标低分辨率像素；
4. 每个 `(target-view,batch,y,x)` 用 `scatter_reduce(amin)` 得最小深度；
5. 若 `z≤z_min+δ`，视为未被遮挡，默认 `δ=0.1`。

训练时可按可见分数随机采样；推理 `max_coverage=True` 时先把最近 buffer frame 的覆盖视为已满足，再逐次选能覆盖最多“尚未覆盖 pixels”的候选，最多 `num_latents=5`。这避免 top-5 全是相邻冗余视角。

### 4.2 为什么回访不依赖时间距离

只要新目标视图与 500 帧前某 entry 有 3D overlap，它就能获得高覆盖并被召回；FramePack 的时间窗口可以忘掉它，但空间 cache 不会按时间窗口删除它。因此“持久”来自 cache 地址空间，“可探索”来自用户可继续提供轨迹。

## 5. canonical correspondence：几何只路由，不渲染外观

### 5.1 canonical map

当前代码为每个空间 slot 构造：

`C_j(u,v)=(x_norm,y_norm,z_j)∈[-1,1]^3`，shape `[N_s,3,H,W]`。

`z_j` 不是真实深度，而是 slot id 在 `[-1,1]` 的编码。随后使用该历史帧 full-res depth/pose 将 canonical map forward warp 到 80 个目标相机，并附上归一化 warped depth：

`[coords(3);depth(1)] = [B,4,80,480,832]`。

每 4 帧取索引 `3,7,…,79`，得到 20 个时间位置；`8×8` pixel shuffle：

`[B,4,20,480,832] → [B,256,20,60,104]`。

5 个 slots 沿 channel concat：`[B,1280,20,60,104]`。padding slot 以 `coords=-1, depth=+1` 标记。

### 5.2 为什么不 warp RGB

warped RGB 会直接携带洞、拉伸、边界渗色；模型容易把这些伪影复制到输出。canonical map 只告诉 attention：“目标 token 应从第几张历史图的哪个位置取信息”。检索 RGB 本身以 spatial image token 进入 value 路径；correspondence encoder 只注入 self-attention 的 **K/Q**，不替换 V。

因此真正原则是：

> **Geometry as routing, not appearance rendering.**

![Lyra 2.0 方法总览](assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig2_method.png)

## 6. FramePack 上下文窗口：精确 config 与动态长度

公开有效配置：

`f1k1f4s2f1s1f16k4f2k2f1k1_g20`

解析规则：`f<count><type><kernel>`；`k` 是时空压缩，`s` 只做空间压缩，`g20` 是 20 个待生成 latent slices。

| 段 | 数量 | 类型 | kernel | 语义 |
|---|---:|---|---:|---|
| `f1k1` | 1 | temporal | 1 | anchor/远端稳定参考 |
| `f4s2` | 4 | spatial | 2 | 4 个检索帧，空间压缩 2× |
| `f1s1` | 1 | spatial | 1 | 1 个高分辨率检索帧 |
| `f16k4` | 16 | temporal | 4 | 远历史，时空更强压缩 |
| `f2k2` | 2 | temporal | 2 | 中距历史 |
| `f1k1` | 1 | temporal | 1 | 最近历史 |
| `g20` | 20 | generation | base | 20 latent = 80 新视频帧 |

因此 context tensor 在 patch 前含 **45 个 latent slices = 20 temporal + 5 spatial + 20 new**。这不等于 45×4 个独立原视频帧：spatial slots 是检索图片单帧编码，temporal slots 经过不同 kernel packing；FramePack 的目的正是让 attention token 数不随全部历史线性增长。

以 latent 空间 `60×104`、Wan base patch `(1,2,2)` 为例，base 新段 token 数为 `20×30×52=31,200`；历史段根据 k/s kernel 进一步压缩。代码通过 `_patchify_lyra2` 把每段 `B,C,T,H,W` rearrange 为 `B,(f·h·w),(C·pt·ph·pw)` 后线性映射到 DiT hidden，最后 concat 全段 tokens。

## 7. 相机条件：轨迹不是一个 embedding

Lyra 2.0 同时使用两路：

### 7.1 最近帧 depth warp

buffer RGB/depth 被 forward warp 到每个目标相机，得到 `[B,3,80,H,W]`，再经 VAE 形成 20 个 camera conditioning latents，覆盖 `cond_latent` 的 generation tail。优点是局部控制精确；缺点是大视角/反遮挡区域没有投影内容。

### 7.2 全序列 Plücker

每个目标像素生成 `(d,o×d)` 六通道：

`[B,T,H,W,6] → permute [B,6,T,H,W] → pixelshuffle8 [B,384,T,H/8,W/8]`。

它与 FramePack 选中的 latent 时间位置对齐，再 concat 到 condition channel。即使 warp 全空，Plücker 仍说明每个像素对应哪条世界射线。因此二者分别回答：

- depth warp：已知内容怎样移动；
- Plücker：相机究竟在看哪里。

自定义 trajectory 文件必须含 `w2c: [N,4,4]`、`intrinsics: [N,3,3]`、`image_height/image_width`；`N` 应是 `1+80k`。首帧单目深度决定轨迹尺度，`--pose_scale` 只是相对缩放，不保证用户轨迹的绝对米制语义。

## 8. anti-drifting：self-augmentation 的代码路径

### 8.1 训练—推理偏差

训练条件通常是干净历史，推理条件是模型自己的输出；chunk 越多，颜色、纹理和结构误差越会复利。Lyra 2.0 对上一生成 chunk 做短程自去噪，把退化结果作为下一 chunk 历史，但当前 chunk GT 仍由干净 VAE cache 编码。

代码阶段 A/B 的关键拼接：

```text
previous generated chunk → one-step self-aug → lat1
zero tail for next chunk                    → lat2
clean current chunk via clean VAE cache     → lat3
training latent target = prefix + lat1 + lat3
conditioning latent   = prefix + lat1 + lat2
```

这样网络必须从不完美 `lat1` 继续生成干净 `lat3`，而不是复制漂移。

### 8.2 论文与当前 config

论文用 flow-matching 公式解释一次近似去噪；公开有效 config 明确为：

- `self_aug_enabled=True`
- `self_aug_steps=1`
- `self_aug_every_k=2`
- `self_aug_prob=1.0`
- `self_aug_max_T=500`
- `self_aug_encode_gt_with_clean_history=True`

论文中概率化描述与此 revision 的“每两次训练触发、触发时概率 1”不能直接互换，应按固定 config 复现。

## 9. 视频模型、权重与 rollout

### 9.1 主干

- 基础：Wan2.1 I2V 14B 480P；
- latent channel：16；
- DiT hidden：5120，40 layers，40 heads；
- flow matching；完整推理 35 steps；
- DMD LoRA：4 steps，蒸馏掉 CFG，README 报告约 15× 加速，但 prompt following/重复纹理可能变差。

权重目录的角色：

| 文件/目录 | 角色 |
|---|---|
| `checkpoints/model/model/*.distcp` | Lyra 2.0 主模型 DCP shards |
| `checkpoints/vae/vae.pth` + mean/std | Wan causal VAE |
| `checkpoints/image_encoder/model.pth` | 图像条件编码器 |
| `checkpoints/text_encoder/*` | T5 text condition/negative prompt |
| `checkpoints/lora/dmd_distillation.safetensors` | 4-step DMD |
| `detail_enhancer.safetensors`、`realism_boost.safetensors` | 推理增强 LoRA |
| `checkpoints/recon/model.pt` | 生成域适配的 DA3 Gaussian recon |

### 9.2 retrieve–generate–update

每轮：

1. 从下一段 80 帧轨迹取一个或多个目标视图；
2. cache 检索最多 5 帧；
3. 组装 temporal/spatial/Plücker/warp/correspondence 条件；
4. DiT 生成 20 latent；
5. causal VAE 流式 decode 80 帧；
6. 对新帧估 depth，逐帧加到 cache；
7. 用户规划下一段。

“arbitrarily long”应理解为循环接口和 cache 没有论文式固定终点，不应理解为质量已证明无限稳定。

## 10. 3DGS/场景表示：生成状态与最终资产分开

### 10.1 当前公开主路径

README 的最终重建不是 Lyra 1.0 decoder，而是：

```text
generated video [T,H,W,3]
→ VIPE：pose/intrinsics/depth
→ DA3（checkpoints/recon/model.pt）
→ Gaussians: means/scales/rotations/opacities/SH
→ reconstructed_scene.ply
→ Gaussian render
```

`vipe_da3_gs_recon.py` 接受 `[T,H,W,3]` 或 `[T,3,H,W]`。PLY 读取后显式恢复：

- means `[1,N,3]`
- scales `[1,N,3]`
- rotations `[1,N,4]`
- opacities `[1,N]`
- harmonics `[1,N,3,D_sh]`

论文的 Gaussian head 使用 `k=2` 空间降采样，候选数量约从 `T·H·W` 降为 `T·H/2·W/2`；准确属性通道布局由内置 DA3 子模块和 `Gaussians` spec 管理，不应套用 1.0 的 14D RGB Gaussian 格式。

### 10.2 长视频为什么要 chunked recon

官方 README 明确：超过约 500 帧时，DA3 Gaussian head 未针对如此长序列微调，单次 pass 可能模糊。推荐 `chunk_size=128`、`chunk_overlap=10`、`num_chunks≥2`，再做 chunk 对齐与合并。

这说明 rollout 的动态长度与 recon 的有效上下文不是同一限制：视频可继续生成，但最终 3D 重建器仍需分块。

## 11. 实验与消融该怎样读

![约第 800 帧长视频比较](assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig5_video_results.png)

Tanks-and-Temples 长视频：

| 方法 | SSIM↑ | LPIPS↓ | FID↓ | 主观质量↑ | 风格一致↑ | 相机控制↑ |
|---|---:|---:|---:|---:|---:|---:|
| GEN3C | 0.350 | 0.589 | 79.07 | 21.75 | 75.54 | **70.91** |
| SPMem | 0.383 | 0.571 | 60.11 | 34.41 | 79.68 | 45.07 |
| **Lyra 2.0** | **0.384** | 0.552 | 51.33 | **43.35** | **85.07** | 63.87 |
| Lyra 2.0 DMD | 0.362 | **0.545** | **49.71** | 43.02 | 78.91 | 58.12 |

T&T 3D 重建的 `LPIPS-P / LPIPS-G / FID / subjective`：GEN3C+DA3 `0.511/0.694/125.19/5.38`，SPMem+DA3 `0.412/0.666/94.11/9.95`，Lyra video+DA3 `0.409/0.648/79.36/14.42`，完整生成域微调 recon `0.372/0.629/72.47/18.80`。

关键不是单帧最锐：去掉 self-augmentation 时主观质量 47.88 高于 full 43.35，但风格一致 77.98 低于 85.07、相机控制 53.92 低于 63.87。长程目标和单帧美观存在冲突。

![关键组件消融](assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig7_ablation.png)

## 12. 代码—论文差异与开放边界

| 项目 | 论文 | revision `9fffc9a` 代码 | 复现含义 |
|---|---|---|---|
| cache 下采样 | 描述 8× | `Sparse3DCache(downsample=4)` | 以代码 revision 为准，记录差异 |
| self-aug 频率 | 概念上概率采样 | `prob=1.0,every_k=2,steps=1` | 不要只抄论文 `p` |
| FramePack | 概念性 1+16/2/1+5 | 明确 `f1k1f4s2f1s1f16k4f2k2f1k1_g20` | 可还原 slot 顺序与 kernel |
| correspondence | canonical+depth 注入 attention | 每 slot 256 通道，5 slot 1280；`inject_kq_only=True` | shape 与注入点已开放 |
| training | 初版曾标 planned | 2026-07 revision 已有 `TRAINING.md`、config、sample data | 当前不是“仅推理代码” |
| mesh | 论文展示分层 mesh/仿真 | 公共 README 一键主路径清楚到 3DGS PLY；未见与论文同等完整的 mesh 一键说明 | 不能把展示资产等同完整开放 pipeline |
| 无限长度 | 可任意续写 | cache 可增长；实验有限，DA3 >500 帧建议分块 | 接口无限 ≠ 质量无限 |
| 许可 | 研究发布 | 权重限制生产/部署/分发 | 不可因 Apache 源码推导商用 |

## 13. 复现建议

1. 固定 code/HF revision；先只下载 `checkpoints/*`，记录 DCP shard 数和本地哈希。
2. 首次只跑 81 帧（1 seed + 80），断言新 latent 是 20、空间是 `60×104`。
3. 每 chunk 落盘：selected frame ids、每候选 coverage、canonical warp mask、cache entry 数、相机轨迹。
4. 做三个最小消融：无 spatial slots、RGB slots 但无 correspondence、完整 K/Q correspondence。
5. 轨迹必须 `1+80k`；先做小平移和回头，再做跨房间，避免轨迹穿物导致把规划错误归因于模型。
6. self-augmentation 复现按当前 config，不将论文概率直接代入。
7. rollout 与 recon 分开验收；>500 帧使用 chunked DA3，否则重建模糊会污染对 world persistence 的判断。
8. 2.0 只适合静态世界；输入中的人车运动可能被当作世界纹理固化。
9. 权重仅按内部科研许可使用；任何部署/公开演示/分发前重新审查条款。

## 14. 局限

- spatial memory 依赖与目标视图有 3D overlap；长距离跳跃无法检索不存在的重叠。
- depth 错误虽不直接作为 RGB 外观，却会通过错检索、错 correspondence 影响生成。
- cache 随时间增长；公开实现没有证明无限规模下的检索、内存和歧义稳定性。
- 静态世界假设没有继承 1.0 的动态 4D 能力。
- 首帧单目 depth 决定轨迹尺度，几何视觉可信不等于 metric/碰撞/物理正确。
- DMD 提速会损失 prompt following、风格一致性或相机控制。

## 15. 官方资源

- [论文](https://arxiv.org/abs/2604.13036)
- [项目页](https://research.nvidia.com/labs/sil/projects/lyra2/)
- [官方代码](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2)
- [官方权重](https://huggingface.co/nvidia/Lyra-2.0)
- 精确 revision、权重树、配置、图片和证据映射见 `assets/Lyra 2.0 - 可探索的生成式三维世界/sources.md`。
