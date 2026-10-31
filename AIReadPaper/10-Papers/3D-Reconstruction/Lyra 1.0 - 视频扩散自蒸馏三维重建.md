---
title: "Lyra 1.0 - 视频扩散自蒸馏三维重建"
aliases:
  - "Lyra"
  - "Lyra 1.0"
  - "Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation"
authors:
  - Sherwin Bahmani
  - Tianchang Shen
  - Jiawei Ren
  - Jiahui Huang
  - Yifeng Jiang
  - Haithem Turki
  - Andrea Tagliasacchi
  - David B. Lindell
  - Zan Gojcic
  - Sanja Fidler
  - Huan Ling
  - Jun Gao
  - Xuanchi Ren
year: 2025
venue: ICLR 2026
arxiv: "2509.19296"
tags:
  - paper
  - 3d-reconstruction
  - 3dgs
  - self-distillation
  - video-diffusion
  - 4d-generation
status: source-verified
created: 2026-10-15
updated: 2026-10-31
source: "https://arxiv.org/abs/2509.19296"
---

# Lyra 1.0：视频扩散自蒸馏三维重建

## 0. 一句话结论与代码版本

> Lyra 1.0 冻结 Cosmos/GEN3C 视频教师，让 RGB decoder 与新加的 3DGS decoder 读取同一 denoised video latent；RGB 输出提供合成监督，学生把 6 条相机轨迹、每条 121 帧的 latent 一次融合成 `[B,N,14]` 显式高斯，动态版再以 source/target time 查询同一段视频的不同运动状态。

本文源码结论固定在官方仓库 `nv-tlabs/lyra` commit **`9fffc9adc37004091ecf26ef03abfb3abdf4d59a`**；权重固定在 `nvidia/Lyra` revision **`240ecc75fedadd3d0056666c679d82348903806e`**。权重仓库实际含 `lyra_static.pt`、`lyra_dynamic.pt`、`config.json`；下载脚本还给出 MD5：

- `lyra_static.pt`：`28ddd05011b33350c70edab02c23c602`
- `lyra_dynamic.pt`：`91c1b3920332287ed161bc47a1fd7077`

源码 Apache-2.0；权重是 NVIDIA Open Model License，不应把两者写成同一种许可。

![Lyra 1.0 能力总览](assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig1_teaser.png)

*图 1：单图→静态 3DGS；单目视频→可按目标时间与视角查询的动态 4DGS。来源：论文 Figure 1。*

## 1. 模块入口表

| 模块 | 公开入口 | 读代码时的职责 |
|---|---|---|
| 静态 SDG | `Lyra-1/cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py` | 单图 + 6 条轨迹 → GEN3C 视频/denoised latent |
| 动态 SDG | `gen3c_dynamic_sdg.py` | 单目视频 + ViPE geometry → 多轨迹 latent；`--flip_supervision` 生成反向监督 |
| demo config | `configs/demo/lyra_static.yaml`、`lyra_dynamic.yaml` | checkpoint、轨迹顺序、目标时间 |
| 有效基础 config | `configs/training/default.yaml` | VAE 压缩率、16 层/512 维、patch、loss、Gaussian 参数化 |
| 静态末阶段 | `3dgs_res_704_1280_views_121_multi_6_prune.yaml` | 6 轨迹、121 帧、0.8 opacity prune |
| 动态阶段 | `3dgs_res_704_1280_views_121_multi_6_dynamic_prune.yaml` | 6 输入轨迹、12 目标帧、source/target time |
| 3DGS decoder | `src/models/recon/model_latent_recon.py::LatentRecon` | latent/camera/time → Gaussian attributes |
| 视角 reshape | 同文件 `reshape_mv_*` | `B(VT)↔(BV)T↔B(VT)` |
| 主干 | `src/models/utils/model.py::get_model_blocks` | 1 Transformer + 7 Mamba-2，重复两组 |
| Gaussian 参数化 | `LatentRecon.gaussian_processing` | ray depth → xyz；激活 scale/rotation/opacity/RGB |
| renderer | `src/rendering/gs.py`、`gs_deferred.py` | 训练 render、novel-view render、depth/alpha |
| 权重下载 | `scripts/download_lyra_checkpoints.py` | 固定文件名和 MD5 |

## 2. 全链路表：image/video/camera → latent → 3DGS → render

记 batch 为 `B`，轨迹数 `V`，每轨迹像素帧数 `L`，latent 通道 `C=16`，输入分辨率 `H×W=704×1280`。

| 顺序 | 张量/操作 | full config shape | reshape 的原因 |
|---:|---|---|---|
| 1 | 单图/单目视频 | image `[B,3,H,W]`；video `[B,L,3,H,W]` | 生成教师的 seed observation |
| 2 | 相机轨迹 | `V=6`；每条 `L=121`；`c2w/w2c [B,V·L,4,4]`、`K [B,V·L,3,3]` | 让教师按已知相机拍摄合成多视图 |
| 3 | GEN3C | RGB `[B,V·L,3,H,W]` | 冻结教师负责补全遮挡区和未见内容 |
| 4 | Cosmos VAE encode | `[B,V·121,3,704,1280] → [B,V·16,16,88,160]` | 时间 8×、空间 8×压缩；`L'=(L-1)/8+1=16` |
| 5 | 分视角 | `[B,V·16,C,88,160] → [B·V,16,C,88,160]` | VAE/patch embedding 先逐轨迹共享参数处理 |
| 6 | `PatchEmbed3D(1,2,2)` | 每轨迹 `[B·V,16,16,88,160] → [B·V,16·44·80,512]` | 空间 token 数降 4×；每轨迹 56,320 tokens |
| 7 | 联合多视图 | `[B·V,56320,512] → [B,V·56320,512]` | 6 条轨迹必须在同一序列里交换信息，不能独立重建再合并 |
| 8 | 16 层重建块 | `[B,337920,512] → 同 shape` | Transformer 建全局交互，Mamba-2 降低长序列成本 |
| 9 | 拆回视角 | `[B,V·56320,512] → [B·V,56320,512] → [B·V,512,16,44,80]` | deconv 按每条轨迹恢复规则网格 |
| 10 | transposed Conv3D | `[B·V,512,16,44,80] → [B·V,12,121,88,160]` | temporal kernel/stride/padding 把 16 latent steps 精确映回 121；空间恢复到 latent-grid 分辨率 |
| 11 | flatten + ray 参数化 | raw `[B·V,121·88·160,12] → Gaussian [B·V,N_v,14]` | raw 只预测 1D ray depth，不直接回归 xyz；最终 `N_v=1,703,680` |
| 12 | 融合/剪枝 | 6 视图拼成 `N=10,222,080`；剪 80% 后约 `2,044,416` | 保留联合覆盖，移除低 opacity 高斯 |
| 13 | render | Gaussians + target camera → RGB/depth/alpha `[B,S,C,H,W]` | 训练对齐教师 RGB/depth；推理支持 novel view |

> [!note] “12 维 head”与“14 维高斯”不是矛盾
> `output_dims: 12` 对应 `distance(1)+rgb(3)+scale(3)+quaternion(4)+opacity(1)`；代码用 `pos=ray_o+ray_d·depth` 构造 xyz(3)，最终顺序为 `xyz(3)+opacity(1)+scale(3)+rotation(4)+rgb(3)=14`。

## 3. 教师：外部视频扩散与 RGB/VAE 分支

### 3.1 生成式监督的真实闭环

给 seed image `I` 和 6 条相机轨迹 `C^{1:L}`：

1. GEN3C 生成 denoised latent `z`；
2. 冻结 RGB decoder 得教师视频 `X_rgb=D_rgb(z)`；
3. 学生 `G=D_gs(z,E_cam[,E_t])` 得显式高斯；
4. `X_gs=Render(G,C)`；
5. 只更新 3DGS decoder，使 `X_gs` 对齐 `X_rgb`、教师深度和正则项。

这是真正的 self-distillation：监督目标来自教师自己的生成结果，不需要真实多视图或真实 3D GT。推理时如果目标是 3DGS，RGB decoder 不在主输出路径中。

### 3.2 为什么学生读 latent，而不是教师 RGB

像素输入有 `6×121×704×1280≈6.54×10^8` 像素；若直接把每像素变 token 或 Gaussian，注意力和显存不可接受。VAE 后仅 `6×16×88×160` 网格，再做 `2×2` patch 得 337,920 tokens。它仍很长，所以主干混用 Mamba-2；论文报告纯 Transformer 20,922 ms，混合主干 3,213 ms。

### 3.3 Plücker 相机编码

每个像素射线由 origin `o` 和 direction `d` 表示，Plücker 为 `(d,o×d)∈R^6`。代码把两组 3 通道分别送入 Cosmos VAE，得到两个 `[B,V·16,16,88,160]` latent 后 concat 成 32 通道，再由零初始化 `PatchEmbed3D` 映到 512 维，与 video token 相加。

这样做的原因：

- 相机条件与视频 latent 具有同一时空网格；
- 零初始化保证加载静态视频特征时，新条件分支初始不扰动主干；
- ray origin/direction 也直接用于最终 Gaussian xyz，条件与输出参数化一致。

![Lyra 自蒸馏与解码器](assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig3_method.png)

*图 2：冻结 RGB/VAE/视频扩散教师，训练 3DGS decoder。图中 14 维指最终 Gaussian；公开代码 raw head 为 12 维。*

## 4. 3DGS decoder：逐模块与 shape

### 4.1 Patch 与多视图融合

`forward_gaussians` 的顺序非常明确：

```text
[B,V·T',C,h,w]
  → reshape_mv_temp_to_batch
[B·V,T',C,h,w]
  → patch(video) + patch(plücker) [+ patch(t_src)+patch(t_tgt)]
[B·V,T'·h/2·w/2,512]
  → reshape_mv_batch_to_temp
[B,V·T'·h/2·w/2,512]
  → 16 blocks
  → reshape_mv_temp_to_batch
[B·V,T'·h/2·w/2,512]
  → [B·V,512,T',h/2,w/2] → deconv
```

先“视角入 batch”是为了复用同一 patch encoder；再“视角入 sequence”是为了联合注意力；最后拆开是因为卷积上采样要求规则 `T×H×W` 网格。

### 4.2 Transformer + Mamba-2

有效 config：

- `enc_depth=16`
- `enc_embed_dim=512`
- `enc_num_heads=8`
- `llrm_7m1t=true`
- `llrm_7m1t_index=8`

`get_model_blocks` 将每 8 层组织成 1 个 Transformer + 7 个 Mamba-2，重复两次。Transformer 提供跨轨迹全局混合；Mamba-2 以线性序列扫描承担大部分深层传播。

### 4.3 deconv 为什么能从 16 恢复到 121

full config 下时间输入 `T'=16`，deconv temporal stride 为 8、kernel 为 9、padding 为 4：

`(16-1)×8 - 2×4 + 9 = 121`。

空间从 `44×80` 以 kernel/stride 2 恢复为 `88×160`。注意这仍是 VAE 空间网格，不是 `704×1280`；每个 Gaussian 对应原图约 `8×8` 区域，因此 6 轨迹候选总数是 `6×121×88×160=10,222,080`。

### 4.4 Gaussian 激活与物理含义

| raw 分量 | 代码变换 | 最终 shape/意义 |
|---|---|---|
| `distance[...,1]` | sigmoid 后映射到 `[dnear,dfar]` | 射线深度 |
| ray + depth | `o+d·depth` | `pos[...,3]`，强制像素对齐射线 |
| `rgb[...,3]` | `0.5*tanh(x)+0.5` | `[0,1]` 颜色 |
| `scale[...,3]` | capped exp | 椭球尺度，cap 0.3 |
| `rotation[...,4]` | L2 normalize | 单位四元数 |
| `opacity[...,1]` | `sigmoid(x-2)` | 初始偏透明，便于正则/剪枝 |

## 5. 损失与监督

`L = L_mse + 0.5L_lpips + 0.05L_depth + 0.1L_opacity`（最后一项在 prune 阶段打开）。

- **MSE / LPIPS**：学生 render 对齐教师 RGB；LPIPS 防止只追平均颜色。
- **depth**：ViPE 对教师生成视频估深度，使用尺度不敏感损失；它修复仅 RGB 监督时的浅浮雕几何。
- **opacity L1 + prune**：动态/最终阶段 `lambda_opacity=0.1`，按 opacity 排序剪 80%。

![深度损失消融](assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig8_depth_ablation.png)

*图 3：无 depth loss 时 RGB 尚可，但几何被压成浅浮雕。*

## 6. 动态 4D：source time、target time 与 12 视角监督

### 6.1 表示不是“高斯轨迹”，而是条件查询

动态模型仍输出一个目标时刻的 `[B,N,14]` Gaussian 集：

`G(t_tgt)=D(z_video,E_cam,E(t_src),E(t_tgt))`。

- `t_src`：每个输入帧自己的时间；
- `t_tgt`：希望重建的运动状态，复制到全部输入 token；
- 改 `t_tgt` 得不同状态，改 render camera 得同一状态的新视图。

时间标量与 sinusoidal 分量组成 3 通道，再经冻结 VAE 编码到 `[B,V·16,16,88,160]`，由两个零初始化 patch 层分别注入 source/target time。

### 6.2 动态训练的真实长度

有效动态 config：

- `num_input_views=121`
- `num_input_multi_views=6`
- `num_views=133`
- 因此每个样本是 **121 输入帧/轨迹 × 6 条轨迹 + 12 个目标监督视角**；`num_views=133` 是单轨迹时间口径的 121+12，不应误读成 133 条轨迹。
- demo 的 `target_index_manual=[0,60,120]` 只是导出三个目标时间，模型可遍历 0–120。

### 6.3 为什么要倒放增强

原始 6 条轨迹通常从近到远；某时间 `t` 只有同一运动状态的少量监督，模型可走捷径，只依赖当前位置附近输入。作者将源视频倒放后再由教师生成 6 条反方向轨迹，并映回原时间，使每个 `t` 同时得到近/远成对的 12 个监督视角，避免早期时刻在远视角 opacity collapse。

这不是推理时双向生成：`--flip_supervision` 只用于构造训练监督，推理仍用 6 条输入轨迹。

## 7. 数据、训练阶段与动态长度

静态数据：59,031 张生成 seed image × 6 轨迹 = 354,186 段视频。动态数据：7,378 个生成视频 × 6 轨迹 = 44,268 段视频。

| 阶段 | 分辨率 | 输入 L | V | 每步监督 S | Batch | Steps |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 176×320 | 17 | 1 | 17 | 4 | 10K |
| 2 | 176×320 | 49 | 1 | 49 | 4 | 2.5K |
| 3 | 352×640 | 49 | 1 | 49 | 2 | 2.5K |
| 4 | 704×1280 | 49 | 1 | 49 | 1 | 2.5K |
| 5 | 704×1280 | 121 | 1 | 9 | 1 | 57.5K |
| 6 | 704×1280 | 121 | 1–6 | 9 | 1 | 7K |
| 动态 | 704×1280 | 121 | 6 | 12 | 1 | 10K |

渐进策略先扩时间、再扩空间、最后扩多轨迹；否则一开始就处理 337,920 tokens 和千万候选高斯，训练成本不可控。

## 8. 推理路径与最终表示

### 静态

```text
image
→ gen3c_single_image_sdg.py（6×121 video latents）
→ sample.py + configs/demo/lyra_static.yaml
→ lyra_static.pt
→ [1,N,14] Gaussians
→ novel-view render / PLY
```

### 动态

```text
monocular video
→ ViPE: depth + K + pose
→ gen3c_dynamic_sdg.py（6 条相机轨迹）
→ sample.py + lyra_dynamic.yaml + t_tgt
→ lyra_dynamic.pt
→ [1,N(t_tgt),14]
→ 改 camera 或 t_tgt 渲染
```

全 offload 的官方 README 记录峰值约 43GB，测试平台为 A100/H100。

![Lyra 静态主结果](assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig4_main_results.png)

## 9. 代码—论文差异与证据边界

| 项目 | 论文/图示 | 公开代码证据 | 结论 |
|---|---|---|---|
| Gaussian 输出维度 | 常写 per-pixel 14D | `output_dims:12`，`gaussian_processing` 再构造 xyz | 论文描述最终表示，代码 config 描述 raw head |
| “每像素 Gaussian” | 视觉上可理解为输出像素 | 实际网格是 `121×88×160`，空间对应原图 8×8 区域 | 不是 704×1280 每像素 |
| 训练开放 | 论文承诺发布 | 当前仓库有 progressive configs、`train.sh`、数据仓库 | 核心 decoder 可训练复现 |
| 教师冻结 | 方法图明确 | 仓库含 Cosmos/GEN3C 分支，但依赖外部 checkpoint | 可审代码，不等于所有基础权重由 Lyra 仓库自行许可 |
| 动态 4D | 论文强调时间/视角控制 | 公开 `lyra_dynamic.pt`、time patch、flip supervision | 关键推理链路已开放 |
| 严格复现缺口 | 论文级数据引擎 | 未固定 FLUX seed/prompt 全表、GEN3C 所有依赖 revision、ViPE checkpoint hash | 结果级 bitwise 复现仍有边界 |

## 10. 复现建议

1. checkout `9fffc9a…`，不要直接跟随 `main`；HF 权重下载固定 `240ecc…` 并核 MD5。
2. 先用仓库自带 pre-generated latents 跑 `sample.py`，把视频生成与 3DGS decoder 两类问题隔离。
3. 保存并断言关键 shape：`[1,96,16,88,160]`（6 轨迹折叠时间前）、每轨迹 56,320 tokens、联合 337,920 tokens、剪枝前 10,222,080 Gaussians。
4. 先关动态/time，只验证单轨迹→六轨迹；再从静态权重微调动态。
5. ViPE 放单独环境，确保 depth、pose、RGB 的 frame index 完全对齐；动态错一帧会被误判为模型运动失败。
6. 训练时分别记录 RGB 指标和几何 depth；只看 PSNR/LPIPS 会漏掉浅浮雕。
7. 导出 PLY 前记录 opacity prune 前后数量，不把更少高斯带来的速度提升误认成网络加速。

## 11. 局限

- 学生只会固化教师能生成出的 3D 一致解释，不保证真实度量几何。
- 121 帧/6 轨迹仍是固定局部窗口，没有持久记忆和跨房间回访。
- 动态表示以目标时间条件查询，不显式维护对象身份、拓扑和运动轨迹。
- 约 200 万剪枝后高斯仍是重资产。
- GEN3C 幻觉、ViPE depth/pose 误差和第三方许可都会传到最终资产。

## 12. 官方资源

- [论文](https://arxiv.org/abs/2509.19296)
- [项目页](https://research.nvidia.com/labs/toronto-ai/lyra/)
- [官方仓库](https://github.com/nv-tlabs/lyra/tree/main/Lyra-1)
- [模型权重](https://huggingface.co/nvidia/Lyra)
- [合成训练数据](https://huggingface.co/datasets/nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG)
- 精确 revision、文件、图片与证据映射见 `assets/Lyra 1.0 - 视频扩散自蒸馏三维重建/sources.md`。
