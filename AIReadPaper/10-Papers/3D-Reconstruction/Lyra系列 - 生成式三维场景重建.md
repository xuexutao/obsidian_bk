# Lyra 系列：生成式三维场景重建

## 0. 阅读结论与证据版本

Lyra 不是同一网络的两次小升级，而是两条不同代码基线共享一个“先生成可控视频，再落成显式 3D”的产品接口：

- **Lyra 1.0**：基于 **Cosmos Predict 1 + GEN3C**，把同一 denoised video latent 同时送给冻结 RGB decoder 与可训练 3DGS decoder，用视频扩散自蒸馏得到一次前馈的静态 3DGS / 动态 4DGS。
- **Lyra 2.0**：基于 **Wan2.1-I2V-14B + FramePack**，先用持久 per-frame 几何记忆生成长程、可回访视频，再用 **VIPE + 经过生成域适配的 Depth Anything 3 Gaussian head** 重建 3DGS；它没有沿用 1.0 的 `LatentRecon`。

本笔记按以下不可变版本读代码：

| 对象 | 官方位置 | 固定 revision | 本文用途 |
|---|---|---|---|
| 系列代码 | `nv-tlabs/lyra` | `9fffc9adc37004091ecf26ef03abfb3abdf4d59a`（2026-07-20） | 1.0/2.0 源码、配置、README、许可 |
| Lyra 1.0 权重 | `nvidia/Lyra` | `240ecc75fedadd3d0056666c679d82348903806e` | `lyra_static.pt`、`lyra_dynamic.pt`、`config.json` |
| Lyra 2.0 权重 | `nvidia/Lyra-2.0` | `c178c3fcf12b63cf98f6749999e6ecb63901669f`（2026-07-15） | DCP 主模型、VAE、图像/文本编码器、DA3 recon、DMD/增强 LoRA |
| Lyra 1.0 合成数据 | `nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG` | `45bc128fdf3a98053aeb4b6ee1113d4ec429552b` | 静态/动态 SDG 训练集 |

> [!warning] 许可不能按“系列统一 Apache 2.0”理解
> 源码是 Apache-2.0；1.0 权重是 NVIDIA Open Model License，模型卡称可商用；2.0 权重是 NVIDIA Internal Scientific Research and Development Model License，明确限制部署、分发、生产使用和生成供销售/分发的作品。代码许可不覆盖权重。

## 1. 系列统一接口，而非统一实现

### 1.1 统一外部接口表

| 阶段 | 统一语义 | Lyra 1.0 实现 | Lyra 2.0 实现 |
|---|---|---|---|
| 输入 image/video/camera | 给定观测与相机运动意图 | 单图或单目视频；GEN3C 生成 6 条轨迹 | 单图 + `w2c[N,4,4]` + `K[N,3,3]` + chunk captions |
| 预处理 | 规范像素、深度、位姿、相机射线 | ViPE 为动态输入给 pose/depth；Plücker rays | 首帧深度定轨迹尺度；VIPE/DA3 深度；Plücker rays |
| 外部视频模型/VAE | 把相机轨迹变成视频 latent | Cosmos/GEN3C；VAE 压缩 `(τ,σ)=(8,8)` | Wan2.1 causal VAE；压缩 `(τ,σ)=(4,8)` |
| latent 与上下文 | 承载生成先验 | 一次性联合 6×121 帧 latent | 80 帧一 chunk；FramePack 时间 slots + 最多 5 个空间 slots |
| 场景表示 | 产生可渲染 3D | 自家 Mamba-2/Transformer 3DGS decoder | per-frame 几何仅作生成记忆；最终由 DA3 Gaussian head 生成 3DGS |
| render / novel view | 显式资产渲染 | gsplat/3DGUT 渲染任意相机、动态目标时间 | 先 world rollout，再把长视频重建为 `.ply`，可渲染或提 mesh |
| 长程 rollout | 持续扩展世界 | 不支持；受一次 GEN3C 窗口限制 | 支持；retrieve → generate 80 frames → update cache 循环 |

### 1.2 系列模块入口表

| 版本 | 模块 | 公开代码入口 | 说明 |
|---|---|---|---|
| 1.0 | 单图/视频扩散 | `Lyra-1/cosmos_predict1/diffusion/inference/gen3c_*_sdg.py` | GEN3C 多轨迹 latent 生成 |
| 1.0 | demo config | `Lyra-1/configs/demo/lyra_static.yaml`、`lyra_dynamic.yaml` | 固定 checkpoint 与训练 config 继承链 |
| 1.0 | 3DGS decoder | `Lyra-1/src/models/recon/model_latent_recon.py::LatentRecon` | latent、Plücker、time → Gaussians |
| 1.0 | 主干 | `Lyra-1/src/models/utils/model.py::get_model_blocks` | 16 层，Transformer/Mamba-2 混合 |
| 1.0 | 渲染 | `Lyra-1/src/rendering/gs*.py` | 训练重投影和导出渲染 |
| 2.0 | 训练配置 | `Lyra-2/lyra_2/_src/configs/experiment.py::register_lyra2` | 有效 FramePack、self-aug、memory 配置 |
| 2.0 | 持久世界模型 | `Lyra-2/lyra_2/_src/models/lyra2_model.py::Lyra2Model` | VAE、上下文选择、相机条件、AR rollout |
| 2.0 | per-frame memory | 同文件 `Sparse3DCache` | 独立点云、可见性检索、最大覆盖贪心选择 |
| 2.0 | DiT | `Lyra-2/lyra_2/_src/networks/wan2pt1_lyra2.py` | FramePack patchify、相机/对应注入 |
| 2.0 | 长视频推理 | `Lyra-2/lyra_2/_src/inference/lyra2_*_inference.py` | preset/custom/交互轨迹 |
| 2.0 | 3D 重建 | `vipe_da3_gs_recon.py`、`vipe_da3_chunked_gs_recon.py` | VIPE pose + DA3 GS，长视频建议分块 |

## 2. 统一全链路：同一抽象，不同数据流

| 顺序 | Lyra 1.0 | Lyra 2.0 | 为什么不同 |
|---:|---|---|---|
| 1 | image/video + 6 条相机轨迹 | seed image + 用户相机轨迹 + chunk text | 1.0 求一次前馈覆盖；2.0 求持续探索 |
| 2 | GEN3C 生成每轨迹 121 帧 | 最近帧 depth warp + 全轨迹 Plücker | 2.0 必须在未覆盖区仍保留相机控制 |
| 3 | Cosmos VAE：`[B,V·121,3,704,1280] → [B,V·16,16,88,160]` | Wan VAE：80 新帧 → 20 个新 latent，空间 `480×832 → 60×104` | VAE 时间压缩率分别 8 与 4 |
| 4 | reshape `B(VT)CHW ↔ (BV)TCHW`，再把 6 视角 token 合并 | FramePack 把时间 history、检索 RGB slots、新 latent 拼成固定预算 | 1.0 联合融合固定窗口；2.0 压缩可变历史 |
| 5 | `2×2` patch → 16 层重建块 → deconv | Wan DiT flow matching；canonical/depth correspondence 只注入 Q/K | 1.0 latent 直接解 3D；2.0 latent 先解长视频 |
| 6 | 每个候选沿 ray 预测 depth + RGB/scale/rotation/opacity，得到 `[B,N,14]` | 每轮解码 80 帧并向 `Sparse3DCache` 追加逐帧 depth/pose | 2.0 的 persistent state 是生成状态，不是最终融合 3D |
| 7 | 3DGS render 任意 novel view；动态版额外选 `t_tgt` | rollout 可无限追加数据结构；完成后 VIPE + DA3 → `.ply`/mesh | 2.0 把 world rollout 与资产重建解耦 |

## 3. 版本对照：真正改变了什么

![Lyra 1.0 teaser：单图前馈出 3DGS、单视频出 4DGS](assets/Lyra系列%20-%20生成式三维场景重建/figures/lyra1_teaser.png)

![Lyra 1.0 训练框架与 3DGS 解码器结构](assets/Lyra系列%20-%20生成式三维场景重建/figures/lyra1_method.png)

| 维度 | Lyra 1.0 | Lyra 2.0 |
|---|---|---|
| 论文问题 | 如何把视频扩散隐式 3D 蒸馏成显式 3DGS | 如何让视频生成跨长轨迹、回访和大视角变化仍保持世界 |
| 视频基线 | Cosmos Predict 1 / GEN3C | Wan2.1-I2V-14B / FramePack |
| 核心状态 | 一次前向的多轨迹 latent | 时间上下文 + 持久 per-frame RGB/depth/pose/point cache |
| 几何角色 | 最终表示；3DGS 直接从 latent 解码 | 生成期只做检索/对应路由；最终 3D 另由 DA3 重建 |
| 上下文长度 | 固定 6×121 像素帧，对应 6×16 latent 帧 | 每 chunk 80 新帧/20 新 latent；有效配置含 20 temporal + 5 spatial history slices |
| 新视图条件 | Plücker + 视频 latent；动态再加 source/target time | 最近帧 RGB depth warp + 全序列 Plücker + 检索帧 RGB slots + canonical/depth correspondence |
| 输出 | 静态 `[B,N,14]` 3DGS 或按目标时间变化的 4DGS | 长视频，再生成 3DGS `.ply` / mesh |
| 动态内容 | 支持动态 4D | 只建模静态世界中的相机运动 |
| 权重许可 | NVIDIA Open Model License | NVIDIA Internal Scientific R&D Model License |

一句话总结：**1.0 把 video latent 直接变成场景表示；2.0 把 video latent 变成可持续更新的世界观测，再离线重建场景表示。**

## 4. 两代不能混用的代码基线

### 4.1 Lyra 1.0：`LatentRecon` 是主角

1. `gen3c_*_sdg.py` 输出多轨迹 denoised latent；
2. `LatentRecon.forward_gaussians` 把视角维临时折进 batch，分别 patchify；
3. 再把 `(B·V)` 还原为 `B`，把视角 token 拼入序列联合建模；
4. deconv 恢复到 `121×88×160` 候选网格；
5. 网络输出 12 个 raw 属性，其中位置不是直接 xyz，而是 `ray_o + ray_d·depth`，最终拼成 14 维高斯；
6. 训练由冻结 RGB decoder 的合成帧监督，推理不需 RGB decoder。

![Lyra 1.0 动态扩展与数据增强](assets/Lyra系列%20-%20生成式三维场景重建/figures/lyra1_dynamic_data_augmentation.png)

### 4.2 Lyra 2.0：`Lyra2Model + Sparse3DCache` 是主角

1. `Sparse3DCache` 为每个历史帧独立保存下采样 world points；不做全局融合；
2. `retrieve` 将所有候选点投到目标视图，做 z-buffer 式遮挡过滤，再按新增覆盖像素数贪心取最多 5 帧；
3. 检索 RGB 经 image VAE 形成 spatial slots；
4. 每个 slot 的 `(u,v,slot-id)` canonical map 与 depth 被 forward warp 到 80 个目标帧；
5. 每 4 帧取一个对应图，经 `8×8` pixel shuffle 变成每 slot 256 通道的 latent-grid 条件，5 slots 共 1280 通道；
6. correspondence encoder 仅修改 self-attention 的 K/Q，V 仍来自 RGB/history token；
7. 新生成 80 帧加入 persistent state，下一轮重复。

![Lyra 2.0 teaser：单图出发的可探索世界](assets/Lyra系列%20-%20生成式三维场景重建/figures/lyra2_fig1_teaser.png)

![Lyra 2.0 pipeline：几何路由与外观合成的解耦](assets/Lyra系列%20-%20生成式三维场景重建/figures/lyra2_fig2_pipeline.png)

## 5. 论文—代码差异与证据边界

| 项目 | 论文表述 | revision `9fffc9a` 公开代码 | 解读 |
|---|---|---|---|
| 1.0 开放范围 | 承诺训练/推理/数据 | 3DGS 训练、静态/动态推理、GEN3C 分支和数据下载均在仓库 | 关键链路可审计，但仍依赖外部 Cosmos/GEN3C/ViPE 权重 |
| 1.0 位置维度 | 图中常概括为每高斯 14 维 | config `output_dims: 12`，代码先预测 ray depth 等 12 维，再构造 xyz 得 14 维 | 不是矛盾，是 raw head 与最终表示口径不同 |
| 2.0 memory 下采样 | 论文描述 retrieval 点云下采样因子 8 | 当前 `Sparse3DCache` 实例使用 `downsample=4` | 复现以 revision/config 为准，并记录该偏差 |
| 2.0 FramePack | 论文给概念性 1 anchor + 16/2/1 temporal + 5 spatial | 有效 config 为 `f1k1f4s2f1s1f16k4f2k2f1k1_g20` | 代码明确了 spatial slots 插入位置与 kernel 类型 |
| 2.0 self-augmentation | 论文常写概率 0.7、一次前向 | 当前公开训练 config 为 `self_aug_prob=1.0`、`self_aug_every_k=2`、`self_aug_steps=1` | “每两次触发一次”与论文抽象概率不能直接等同 |
| 2.0 3D 输出 | 论文含 3DGS 与分层 mesh | 公共 README 主路径完整提供 VIPE+DA3→PLY；mesh 的论文级完整资产管线没有作为同等清晰的一键入口 | 不把论文展示等同于所有后处理代码都完整开放 |
| 2.0 无限探索 | “arbitrarily long” | 状态/cache 可继续增长，公开实验与 README 只覆盖有限轨迹；长视频 DA3 还建议分块 | 表示数据结构可续写，不代表质量无限稳定 |

## 6. 复现建议：按版本拆开，不做“统一环境”

1. **固定 revision**：代码用 `9fffc9a…`；HF 下载时显式指定上表 revision，保存 `git rev-parse` 与文件哈希。
2. **先复现 1.0 静态最小闭环**：使用预生成 latent + `lyra_static.pt`，避免先调 GEN3C；确认 `[B,N,14]`、PLY 和 novel-view render 后再生成自己的多轨迹 latent。
3. **再复现 1.0 动态**：ViPE 单独环境；检查 121 帧、6 输入轨迹、12 个目标监督视角以及 `t_tgt∈[0,120]`。
4. **2.0 先只做 81 帧**：一张 seed + 80 新帧，逐项保存 temporal slots、5 个 retrieved frame ids、warp coverage、cache size。
5. **回访单元测试**：设计“前进 2 chunks → 转身 → 回原位”，对比关闭 spatial memory、关闭 correspondence、关闭 self-augmentation。
6. **重建最后接**：先验证 rollout 视频，再运行 `vipe_da3_gs_recon.py`；超过约 500 帧按 README 用 `vipe_da3_chunked_gs_recon.py`，不要把 DA3 长序列模糊误判为世界模型遗忘。
7. **许可隔离**：2.0 权重不得因源码 Apache-2.0 就进入生产或对外服务。

## 7. 图像与结果索引

![Lyra 1.0 静态场景 SOTA 对比（附录大图）](assets/Lyra系列%20-%20生成式三维场景重建/figures/lyra1_sota_static_comparison.png)

![Lyra 1.0 深度监督消融（附录大图）](assets/Lyra系列%20-%20生成式三维场景重建/figures/lyra1_ablation_depth.png)

- 1.0 的完整数据流、reshape、损失与动态监督见 [[Lyra 1.0 - 视频扩散自蒸馏三维重建]]。
- 2.0 的 persistent state、上下文窗口、轨迹条件与 correspondence shape 见 [[Lyra 2.0 - 可探索的生成式三维世界]]。
- 本文只保留统一接口、版本对照和代码基线边界，避免复制两篇逐模块正文。

## 8. 官方入口

- [系列官方仓库](https://github.com/nv-tlabs/lyra)
- [Lyra 1.0 论文](https://arxiv.org/abs/2509.19296) · [项目页](https://research.nvidia.com/labs/toronto-ai/lyra/) · [权重](https://huggingface.co/nvidia/Lyra)
- [Lyra 2.0 论文](https://arxiv.org/abs/2604.13036) · [项目页](https://research.nvidia.com/labs/sil/projects/lyra2/) · [权重](https://huggingface.co/nvidia/Lyra-2.0)
- 精确来源、revision、文件树与本地图片映射见 `assets/Lyra系列 - 生成式三维场景重建/sources.md`。
