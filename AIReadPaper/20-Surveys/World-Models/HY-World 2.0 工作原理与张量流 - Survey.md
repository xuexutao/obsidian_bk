---
type: survey
status: complete
domain: World-Models
paper: "HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds"
year: 2026
arxiv: "2604.14268"
doi: "10.48550/arXiv.2604.14268"
source: https://arxiv.org/abs/2604.14268
project: https://3d-models.hunyuan.tencent.com/world/
code: https://github.com/Tencent-Hunyuan/HY-World-2.0
code_commit: df9988efb87bfc0f4947eb3889411cf957478b06
tags:
  - World-Models
  - 3D-Generation
  - 3D-Reconstruction
  - 3DGS
  - Video-Diffusion
  - Code-Reading
created: 2026-10-26
updated: 2026-10-26
---

# HY-World 2.0 工作原理与张量流：从一张图到可漫游 3DGS 世界

## 基本信息

- **研究对象**：HY-World 2.0 技术报告（arXiv 2604.14268v1，44 页）+ 官方开源代码 `Tencent-Hunyuan/HY-World-2.0`（commit `df9988e`，2026-08-13）+ HuggingFace 上的官方权重配置（`tencent/HY-World-2.0`、`hanshanxue/WorldStereo`）
- **定位**：这是一篇“代码级工作原理”综述，和已有的论文精读 [[HY-World 2.0 - 多模态三维世界生成重建与模拟]] 互补。精读侧重“论文说了什么、实验证明了什么”；本文侧重“**数据在每个模块里长什么样、形状为什么是这样、代码和论文哪里不一样**”，并对途中遇到的关键技术做拓展。
- **阅读方式**：第 1 章给出全链路张量总表，第 2–7 章逐阶段拆解（每章都按“输入输出 → 内部张量流 → 设计动机 → 代码与论文差异 → 相关技术拓展”组织），第 8 章做计算量与复现分析，第 9 章是个人判断。
- **形状记号**：`B` 为 batch，`S` 为视角数，`F` 为帧数，`C` 为通道，`H×W` 为像素分辨率；没有特殊说明时以代码默认配置举例（全景 960×1920、关键帧 480×832、21 帧、WorldMirror 长边 952）。
- **陈述分级**：「论文」= 技术报告原文；「代码」= 开源仓库/权重配置可直接核对；「推断」= 我根据代码或公开资料推出的结论，会显式标注。

## 一句话结论

HY-World 2.0 的世界生成链在代码层面是一条“同一 token 网格贯穿始终”的流水线：HY-Pano 2.0 在 60×122 的潜变量网格上生成 960×1920 的 ERP 全景；WorldNav 把全景切成 27 张 480×832 透视图作为记忆库，并规划约 35 条、每条 21 帧的相机轨迹；WorldStereo 2.0 以 Wan2.1-I2V-14B 为骨干，在“逐帧独立编码”的关键帧潜空间 `[1,16,21,60,104]` 中，用点云渲染 ControlNet、7 维位姿嵌入和“RoPE 虚拟横向拼接”的检索记忆，4 步生成每条轨迹的关键帧；WorldMirror 2.0 以 826×476 输入、相机先验条件，对数百张视图一次前向回归深度；随后在视差空间做 RANSAC 线性对齐与锚点异常剔除，最后用约 5M 初始点、RGB 颜色、MaskGaussian 稀疏化的 3DGS 训练 1500 步得到世界。理解它的关键，不在单个模型，而在每一级都把上一级的不确定性压成下一级能用的“先验”：全景 → 点云先验 → 渲染条件 → 生成视图 → 相机先验重建 → 对齐深度 → 3DGS 监督。

---

## 1. 全链路总览

### 1.1 四阶段与代码脚本的对应

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/01-overview.png)

> 图 1：HY-World 2.0 四阶段架构。论文把流程描述为 Panorama Generation → Trajectory Planning → World Expansion → World Composition，每阶段由一个核心模型/算法支撑（HY-Pano 2.0、WorldNav、WorldStereo 2.0、WorldMirror 2.0）。代码里第 4 阶段实际又拆成“前馈重建 + 深度对齐”（在 `video_gen.py` 末尾执行）和“GS 数据准备 + 3DGS 训练”两段。来源：论文 Figure 2，PDF p.5，https://arxiv.org/abs/2604.14268

论文的四阶段在仓库里落到 6 个脚本上，这一层映射关系是读代码的第一把钥匙：

| 论文阶段 | 代码入口 | 运行方式（README 推荐） | 主要产物 |
|---|---|---|---|
| Stage I 全景生成 | `panogen/pipeline.py`（HunyuanImage-3 后端）或 `pipeline_with_qwen_image.py`（Qwen-Image-Edit 后端） | 单机多卡 `device_map=auto` | `panorama.png`（960×1920） |
| Stage II 场景解析 + 轨迹规划 | `worldgen/traj_generate.py` + `src/navi_utils.py` + `src/panorama_utils.py` | 单 GPU + 外部 vLLM 服务（Qwen3-VL-8B） | 全景深度/点云/网格、`pano_bank`、`polar_bank`、各轨迹 `camera.json` |
| Stage II 轨迹渲染 | `worldgen/traj_render.py` | `torchrun` 8 卡 | 每条轨迹 `render.mp4`、`render_mask.mp4`、`traj_caption.json` |
| Stage III 世界扩展 | `worldgen/video_gen.py` + `models/` | `torchrun` 8 卡 + FSDP + 序列并行 | 每条轨迹 21 帧生成结果 mp4；记忆库更新 |
| Stage IV-a 重建 + 对齐 | `video_gen.py` 末尾调用 `PanoramaMemoryBank.apply_worldmirror()` / `.alignment()`，内部 `torchrun -m worldrecon.pipeline` | 8 卡 SP + BF16 + FSDP | 对齐深度、`aligned_pcd.ply` |
| Stage IV-b 3DGS | `gen_gs_data.py` → `world_gs_trainer.py`（自带 `gsplat_maskgaussian`） | 8 卡 | `.ply/.spz` 高斯、TSDF mesh |

### 1.2 全链路张量总表

下表是全文的“地图”，后面各章会逐行展开。所有数字都是代码默认值推出来的。

| #   | 模块                    | 关键输入                                   | 关键输出                                                                        | 主干/算法                                                | 参数量（约）                  |
| --- | --------------------- | -------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------- |
| 1   | HY-Pano 2.0           | 透视图（任意尺寸）+ 文本                          | ERP `[960,1952,3]` → 接缝融合裁成 `[960,1920,3]`                                  | HunyuanImage-3（80B MoE）或 Qwen-Image-Edit-2509 + LoRA | ~84B（权重目录 169 GB，bf16）  |
| 2   | 全景深度                  | ERP `[960,1920,3]`                     | 距离图 `[960,1920]`、射线 `[960,1920,3]`                                          | 42 视角 MoGe-2 + 梯度域 LSMR 融合                           | MoGe-2 ViT-L            |
| 3   | 全景点云/网格               | 距离 + 射线 + 天空/边缘 mask                   | `global_pcd`：`N≤1.84M ×(xyz+rgb)`；`global_mesh`：960×1920 网格                 | 反投影                                                  | —                       |
| 4   | 记忆库初始化                | ERP                                    | `pano_bank`：27 张 `[480,832,3]` + 16bit 深度；`polar_bank`：16 张                 | 透视重采样（FoV 120°×90°）                                  | —                       |
| 5   | WorldNav              | 网格、NavMesh、语义 3D 地标                    | 每条轨迹 `w2c [21,4,4]`、`K [21,3,3]`，最多 35 条                                    | 启发式 + Dijkstra + 光线投射                                | Qwen3-VL-8B、SAM3        |
| 6   | 轨迹渲染                  | `global_pcd` + 相机                      | `render [21,480,832,3]`、`mask [21,480,832,1]`、caption                       | PyTorch3D 点溅射                                        | —                       |
| 7   | WorldStereo 2.0       | 起始帧、渲染视频、Plücker、检索参考帧、文本              | 关键帧 `[1,3,21,480,832]`                                                      | Wan2.1-I2V-14B + 20 层 ControlNet + 记忆                | 检查点 34.9 GB（bf16 约 17B） |
| 8   | WorldMirror 2.0（生成链内） | `S≈27+8×轨迹数` 张 `[476,826,3]` + 位姿 + 内参 | 深度 `[S,476,826,1]`、深度 mask                                                  | DINOv2-L 交替注意力 24+24 层 + DPT                         | ~1.26B（权重 5.05 GB，fp32） |
| 9   | 深度对齐                  | WorldMirror 深度、MoGe 法线/mask、点云渲染引导深度   | 每帧 `(k,b)`，对齐后点云 ≤2M                                                        | 视差空间 RANSAC + 锚点异常检测                                 | —                       |
| 10  | 3DGS                  | ~800 张训练视图 + ≤5.3M 初始点                 | `N` 个高斯：`means [N,3]`、`scales [N,3]`、`quats [N,4]`、`opac [N]`、`rgb [N,1,3]` | gsplat + MaskGaussian                                | —                       |
| 11  | Mesh                  | 3DGS 渲染 RGB-D                          | TSDF → Marching Cubes 三角网格                                                  | Open3D                                               | —                       |

一个值得注意的巧合（其实是刻意设计）：HY-Pano 两种后端在 960×1952 输出时的图像 token 网格都是 **60×122=7320**（HunyuanImage-3 是 16× VAE、patch 1；Qwen-Image 是 8× VAE、2×2 pack），而 WorldStereo 的 480×832 关键帧每帧 token 网格是 **30×52=1560**。整个系统的显存和时延基本由这几个网格大小决定。

### 1.3 端到端耗时（论文 Table 10，H20）

| 阶段 | 全景 | 轨迹规划 | 世界扩展 | 重建与对齐 | 3DGS | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 时间 | 15 s | 182 s | 286 s | 102 s | 127 s | 712 s |

“轨迹规划”比全景生成还慢 12 倍，主要因为它串行调用了 Qwen3-VL、SAM3、MoGe-2、42 视角深度融合、NavMesh 构建和大量光线投射。这在第 3 章会看到。

---

## 2. Stage I：HY-Pano 2.0 全景生成

### 2.1 任务与输入输出

| 项 | 形状/取值 | 说明 |
|---|---|---|
| 输入图像 | 任意 `[h,w,3]` | 不需要相机内参、FoV |
| 输入文本 | 字符串 | 代码强制前缀 `"Expand this image to a 360-degree equirectangular panorama."` |
| 生成尺寸 | `960×1952` | 比 2:1 多出 32 列 |
| 最终输出 | `[960,1920,3]` uint8 | 左右接缝融合后裁掉多余 32 列，严格 2:1 ERP |

### 2.2 两种后端的内部张量流

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/02-hypano-arch.png)

> 图 2：HY-Pano 2.0 架构。左：文本 token、带 circle padding 的全景噪声潜变量 token、条件图潜变量 token 拼成一个序列送入 MMDiT；右上：潜空间环形填充（第 0 列复制到最右）；右下：像素空间左右边缘线性融合。来源：论文 Figure 3，PDF p.7。

#### 2.2.1 后端 A：HunyuanImage-3（默认，论文主线）

这不是传统意义上的独立 DiT，而是 **HunyuanImage 3.0 统一多模态自回归模型**（`HunyuanImage3ForCausalMM`）：同一个 Transformer 既做文本自回归，又对图像 token 做扩散去噪。从 HF 上的 `config.json` 可读出：

| 组件 | 配置（代码/权重配置） |
|---|---|
| Transformer | 32 层，hidden 4096，32 个 query head、8 个 KV head（GQA），head dim 128，QK-Norm，RMSNorm |
| FFN | MoE：64 个专家，top-8 路由 + 1 个共享专家，专家 FFN 3072 |
| 位置编码 | 文本 1D RoPE，图像 `rope_type="2d"`；最大位置 22,800 |
| VAE | `AutoencoderKLConv3D`，32 通道潜变量，空间 16× 下采样，scaling 0.5627 |
| 图像 token 化 | `patch_size=1` 的 `UNetDown`（带时间步嵌入的卷积残差块）把每个潜变量位置变成一个 4096 维 token |
| 条件图 | `cond_image_type="vae_vit"`：同时送 VAE token 和 SigLIP2-so400m（patch16、最多 1024 个 patch、1152 维，经 2 层 MLP 对齐到 4096）token；条件 token 采用 `joint_full` 全注意力 |
| 采样 | `FlowMatchDiscreteScheduler`，50 步，SD3 式时移 `flow_shift=6.5`，CFG 4.0 |
| 文本侧 | `bot_task="think_recaption"`：先自回归生成“思考 + 重写 caption”（最多 2048 token，temperature 0.6，top-p 0.95），再生成图像 |

960×1952 的数据流如下（推断，按配置计算）：

$$x_0\in\mathbb{R}^{1\times32\times60\times122}\xrightarrow{\text{UNetDown},p=1}\mathbb{R}^{7320\times4096}$$

条件图按分辨率组（base 1024）缩放，例如 1024×1024 的输入会得到 64×64=4096 个 VAE token；再加至多 1024 个 SigLIP2 token、system prompt、用户 prompt 和 CoT 文本。全部 token 拼成一条序列，生成区 7320 个 token 在 50 步中反复去噪。时移公式为：

$$\sigma'=\frac{s\,\sigma}{1+(s-1)\,\sigma},\quad s=6.5$$

$s$ 越大，越多的步数落在高噪声区，这对“需要先定全局布局”的全景很重要：ERP 的难点是 360° 构图一致，而不是局部纹理。

#### 2.2.2 后端 B：Qwen-Image-Edit-2509 + LoRA（轻量替代）

| 张量 | 形状 | 说明 |
|---|---|---|
| 噪声潜变量 | `[1,16,120,244]` | Qwen-Image VAE：16 通道、8× 下采样 |
| 打包后 token | `[1,7320,64]` | 2×2 patch pack：60×122 个 token，每个 16×4=64 维 |
| 条件 VAE 潜变量 | 约 `[1,4096,64]` | 条件图缩放到约 1024² 像素后编码 |
| 条件 VL 输入 | 约 384² 缩略图 | 喂给 Qwen2.5-VL 文本编码器 |
| `img_shapes` | `[(1,60,122),(1,h_c/16,w_c/16)]` | 告诉 MMDiT 每段序列的 2D 网格尺寸，用于 RoPE |
| 采样 | 40 步，true CFG 7.5 | 负提示词里写了“车、近景压迫、比例失调、画面上方的树叶”等全景常见失败模式 |

去噪时 `latent_model_input=cat([latents, image_latents], dim=1)`，即**沿序列维拼接**，输出只取前 7320 个 token 的预测。这就是论文所说“把条件潜变量和全景噪声潜变量拼成统一 token 序列，靠 self-attention 隐式学习透视→ERP 映射”。

### 2.3 为什么这么设计

**为什么用“序列拼接 + 自注意力”代替显式投影？** HunyuanWorld 1.0 先估计输入图的内参/FoV，再把它投影到 ERP 的对应位置做 outpainting。只要 FoV 估错几度，投影区就会出现拉伸和接缝。2.0 把“输入图在全景里放在哪、放多大”交给注意力学习，好处是对无标定、裁剪过、风格化的输入都鲁棒。代价是失去确定性几何约束：模型可能把输入图“重画”而不是严格保留（论文和 Marble 对比时强调了 2.0 更忠实于输入，说明训练数据让模型学会了保真，但这不是结构保证）。

**为什么输出 960×1920 而不是 4K？** 下游一律以 1920 宽为上限：`traj_generate.py` 在做深度时把全景缩到长边 ≤1920，切 480×832 透视图时 1920 宽 ERP 对应 120° FoV 约 640 像素，已经足够 832 宽的透视图不过度插值；生成更大全景只会增加 Stage I 的 token 数（与像素数线性、注意力平方）。

**为什么多生成 32 列再融合？** ERP 的第 0 列和第 1919 列在物理上相邻。代码的做法是让模型多生成 32 列“越过 360°”的内容，然后对左边 32 列做线性渐变：

$$I'[:,x]=\left(1-\tfrac{x}{32}\right)I[:,W'-32+x]+\tfrac{x}{32}I[:,x],\quad x\in[0,32)$$

再丢掉最后 32 列。这样最左边的像素完全来自“360° 之后”的那段内容，与最右边自然连续。

### 2.4 代码与论文的差异

- **潜空间 circular padding 在开源推理代码里找不到。** 论文图 3 和正文都写了“在去噪过程中对潜变量做环形填充”。我在 `panogen/` 全目录（两个 pipeline、`hunyuan_image_3/*`、`qwen_image/*`）检索 `circular / wrap / pad`，只找到 VAE 分块解码里的普通 padding 和像素级 `circular_blend_edges`。推断：要么环形填充只在训练数据构造时使用（让模型学会生成“越界 32 列”的连续内容），要么尚未开源。实践上，开源版的接缝质量完全依赖模型本身学到的连续性加 32 像素融合。
- **两个后端不等价。** README 把 HunyuanImage-3 称为“完整推理链（带 CoT recaption）”，Qwen 后端是 0.85 GB 的 LoRA。论文的 Table 4 指标对应哪个后端，报告没有说明。

### 2.5 相关技术拓展

#### 2.5.1 ERP 表示与接缝/极点问题

ERP（等距柱状投影）把球面 $(\theta,\phi)$ 均匀映射到 2:1 矩形。它的两个固有缺陷是：左右边界拓扑相连但卷积/注意力不知道；两极被横向拉伸，同一行像素在极点附近对应极小的立体角。业界常见三种对策：

| 方法        | 做法                               | 代表                   |
| --------- | -------------------------------- | -------------------- |
| 环形填充/旋转增强 | 在卷积或潜变量上做 wrap padding，训练时随机水平滚动 | PanoDiffusion、DiT360 |
| 立方体贴图     | 把球面拆成 6 个 90° 面，用跨面注意力保证一致       | CubeDiff             |
| 球面位置编码    | 在 RoPE 中使用球面坐标而非平面坐标             | 若干 360° 视频模型         |

HY-Pano 2.0 走的是第一类，并把极点问题留给下游：Stage II 的 `smooth_south_pole_depth`、`polar_bank` 都是在补极点区域。

#### 2.5.2 统一自回归 + 扩散模型

HunyuanImage 3.0 属于 Transfusion 一类“一个 Transformer 同时做 next-token 和 diffusion”的统一模型：文本 token 用因果注意力和交叉熵，图像 token 在序列内部用双向注意力和流匹配损失。对全景任务的意义是**可以先“想”再“画”**：`think_recaption` 让模型先用语言描述“这张图向四周扩展应该是什么场景”，再把这段文字作为条件生成。代价是推理慢、显存大（80B MoE 需要多卡），这也是为什么还提供 Qwen 轻量后端。

---

## 3. Stage II：场景解析与 WorldNav 轨迹规划

### 3.1 场景解析：从一张 ERP 到四种几何/语义资产

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/03-scene-parsing.png)

> 图 3：轨迹规划前的场景解析。从全景得到 MoGe-2 全景点云、SAM3 语义 mask、HY-World 1.0 风格的全景网格和 Recast NavMesh。来源：论文 Figure 4，PDF p.7。

代码顺序（`traj_generate.py` 第 228–420 行）如下，形状以 960×1920 输入为例：

| 步骤 | 输入 | 输出 | 关键参数（代码） |
|---|---|---|---|
| 场景分类 | ERP（base64 发给 vLLM） | `meta_info.json: {"scene_type": "indoor/outdoor"}` | Qwen3-VL-8B，temperature 0 |
| 天空 mask | ERP | `sky_mask [960,1920]` bool | 室外：GroundingDINO 检测 `"sky."` + ZIM 抠图（阈值 0.3/0.3）；天空占比 >90% 时整幅视为非天空；室内全 False |
| 42 视角深度 | ERP 缩到长边 ≤1920 | 42 张 `[512,512]` 距离图 + mask | 二十面体细分 1 次得 42 个方向，每视角 FoV 90°，MoGe-2 ViT-L |
| 全景深度融合 | 42 张距离图 | `distance [960,1920]`、`rays [960,1920,3]` | 对数距离梯度 + 拉普拉斯约束的超定线性系统，cupy GPU LSMR，金字塔由粗到细 |
| 深度后处理 | distance | 截断/压缩后的 distance | 99% 分位截断；室外超过 $c=8\times$中位数的部分做 $d'=2c-c^2/d$ 压缩 |
| 全景点云 | distance、rays、RGB、mask | `global_pcd.ply`：最多 1.84M 点 | 去掉天空与深度边缘（`depth_edge` rtol 0.1） |
| 全景网格 | 960×1920 网格化 | `global_mesh.ply` | 用于严格碰撞检测 |
| 全景法线 | distance + rays | `global_normal.npy` `[N,3]` | 用于后续对齐的法线一致性检查 |
| 天空点云 | 天空区域深度置为最大值，边界 100 像素平滑 | `sky_pcd.ply` | 3DGS 阶段单独处理 |
| polar_bank | ERP | 16 张 `[480,832,3]` + 深度 | 4 个近极点方向 × 4 个方位角 |
| pano_bank | ERP | 27 张 `[480,832,3]` + 深度 | 3 个仰角（方向向量 z=0、±0.5，约 ±26.6°）× 9 个方位角（40° 步长） |

透视图尺寸来自一个小公式：高固定 480，宽按 FoV 比例换算，再对齐到 16 的倍数：

$$w=\mathrm{round}\left(\frac{\tan(60^\circ)}{\tan(45^\circ)}\times480\right)=831\rightarrow832$$

480×832 恰好是 Wan2.1-480P 的标准分辨率，**这不是巧合，而是整个系统的“分辨率锚点”**：WorldStereo 的关键帧、记忆库、轨迹渲染都用这个尺寸，避免任何重采样。

#### 3.1.1 为什么要 42 视角 + 梯度域融合

MoGe-2 是透视单目模型，不能直接吃 ERP。最朴素的做法是切 12 个视角分别预测，再拼回球面；问题是每个视角的深度都有独立的尺度，硬拼会在交界处出现台阶。代码的做法是**只信任每个视角预测的“相对变化”**：把每个视角的 $\log{d}$ 重投影到 ERP，计算水平/垂直梯度和拉普拉斯，再求解

$$\min_{\mathbf{x}}\left\|\begin{bmatrix}G\\L\end{bmatrix}\mathbf{x}-\begin{bmatrix}\bar g\\\bar l\end{bmatrix}\right\|_2^2,\quad \mathbf{x}=\log d_{\text{pano}}$$

其中 $G$ 是带水平环绕（`wrap_x=True`）的差分算子，$\bar{g}$ 是多视角重叠处的梯度均值。在对数域里，各视角的尺度差变成常数偏移，求梯度时自然消失，这就是它能无缝拼接的原因。视角从 12 增加到 42，相当于每个 ERP 像素平均被更多视角覆盖，梯度均值更稳；代价是线性系统变大，所以用 GPU LSMR 并以低分辨率解作为初值（递归减半，直到 ≤256）。

### 3.2 WorldNav：五类轨迹

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/04-worldnav-trajectories.png)

> 图 4：WorldNav 的五类轨迹：(a) Regular、(b) Surrounding、(c) Reconstruction-aware、(d) Wandering、(e) Aerial，(f) 为全部叠加。来源：论文 Figure 5，PDF p.8。

| 轨迹 | 论文上限 | 代码中的关键参数 | 目的 |
|---|---:|---|---|
| Regular（`view*/traj0..2`） | 9 | 3 个分割视角（120° 间隔）；每个视角：左/右旋转 ±120°、“右上航拍”（方位 −60°、俯仰 −45°） | 走出全景中心的固定视点 |
| Surrounding（`target_*`） | 5 | VLM 识别地标 → SAM3 → 3D 质心；72 个候选节点；半径随物体尺寸自适应 | 补物体侧面和背面 |
| Reconstruction-aware（`reconstruct_*`） | 10 | 检测高长宽比退化面片 → NMS → `recon_topk=5`；`traj1` 为迭代环绕 | 补几何空洞 |
| Wandering（`wonder_*`） | 3 | NavMesh 分 8 个扇区，Dijkstra 找最远可达点；`wonder_topk=3`；`move_dist=8.0` | 走廊、街道边界 |
| Aerial（各类 `traj>0`） | 8 | 在 surrounding/wandering 上加 +45° 俯视，碰撞时动态减小 | 补 BEV 盲区 |

NavMesh 的 Recast 参数为 `cellSize=0.1, cellHeight=0.1, agentHeight=0.2, agentRadius=0.1, agentMaxClimb=0.1, maxSlope=30°`。注意这些是在 MoGe 的相对尺度下定义的，场景尺度由全景深度决定，因此“0.1”的物理含义随场景变化。

每条轨迹输出：

| 文件 | 形状 | 说明 |
|---|---|---|
| `camera.json["extrinsic"]` | `[21,4,4]` | OpenCV 约定的 w2c |
| `camera.json["intrinsic"]` | `[21,3,3]` | 像素单位，对应 480×832 |
| `start_frame.png` | `[480,832,3]` | 轨迹第 0 帧，从全景切出 |

**为什么每条轨迹是 21 帧？** 它直接对应 Wan2.1 的 81 帧视频在 4× 时间压缩后的 21 个潜变量帧。WorldStereo 2.0 把“81 帧视频”换成“21 个关键帧”，潜变量帧数不变，所以骨干的时间 RoPE、显存都和原模型对齐。

### 3.3 轨迹渲染与轨迹描述

`traj_render.py` 用 PyTorch3D 的点云渲染器把 `global_pcd` 沿每条轨迹渲染出来（半径 0.008，每像素 20 个点，8 卡切片并行），并把第 0 帧替换成高质量的 `start_frame`：

| 张量 | 形状 | 取值 |
|---|---|---|
| `render.mp4` | `[21,480,832,3]` | 点云渲染 RGB，空洞为黑 |
| `render_mask.mp4` | `[21,480,832,1]` | 有点覆盖为 1 |
| `traj_caption.json` | 字符串 | Qwen3-VL 看渲染视频写出的描述，作为 WorldStereo 的文本条件 |

这里的设计逻辑是：**渲染视频既是相机控制信号，也是“已知区域”的强约束**。被 mask 覆盖的像素告诉扩散模型“这里已经有内容，照着画”；mask 为 0 的空洞才是需要想象的区域。

### 3.4 相关技术拓展

#### 3.4.1 MoGe / MoGe-2

MoGe 系列直接回归仿射不变点图（而不是深度），配合最优尺度-平移对齐的损失，在开放场景几何上很强；MoGe-2 进一步给出度量尺度和法线。它在 HY-World 2.0 里出现了三次：全景深度（42 视角）、对齐阶段的有效 mask 与法线、3DGS 阶段的稠密法线监督。详见 [[MoGe系列 - Survey]]。

#### 3.4.2 NavMesh 与 Recast

NavMesh 是游戏引擎里的标准可行走区域表示：把场景体素化，按代理高度/半径/爬坡能力筛出可站立体素，再合成凸多边形网格，路径搜索在多边形图上做 A*/Dijkstra。WorldNav 借它保证“相机不穿墙、不钻地”，并补了三件事：把顶点通过光线投射贴到真实地面；用 KD-Tree 做边界腐蚀，防止相机贴墙；检测孤立可行走区域并合成桥接多边形。

---

## 4. Stage III：WorldStereo 2.0 世界扩展

这一章是全系统最复杂的部分，我按“骨干 → 条件 → 记忆 → 采样 → 循环”五层拆。

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/06-worldstereo-pipeline.png)

> 图 5：WorldStereo 2.0 总体流程。(a) 主 DiT 分支接收目标视图噪声、参考视图、检索视图及各自的相机嵌入，输出写回记忆库；(b) 相机控制分支把全景点云沿轨迹渲染、叠加 Plücker 射线，通过 Camera Adapter 注入主 DiT。来源：论文 Figure 7，PDF p.10。

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/05-worldstereo-training-stages.png)

> 图 6：WorldStereo 2.0 的三段训练：Domain-Adaption（相机控制）→ Middle-Training（一致性记忆）→ Post-Distillation（快速推理）。来源：论文 Figure 6，PDF p.9。

### 4.1 骨干：Wan2.1-I2V-14B-480P

HF 上 `hanshanxue/WorldStereo` 的 `config.json` 写明 `base_model = Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`。代码 `WorldStereoRefSModel` 继承 diffusers 的 `WanTransformer3DModel`：

| 组件 | 配置 |
|---|---|
| DiT | 40 层，40 head × 128 = 5120 维，FFN 13824，patch `(1,2,2)`，QK RMSNorm，adaLN（6 组 shift/scale/gate） |
| 输入通道 | 36 = 噪声 16 + 首帧 mask 4 + 条件潜变量 16 |
| 文本编码器 | umT5-XXL，4096 维，512 token（不足补零） |
| 图像编码器 | CLIP ViT-H/14，取倒数第二层 `[1,257,1280]` |
| VAE | Wan-VAE，z_dim 16，空间 8×、时间 4×（因果 3D 卷积） |
| 检查点 | `worldstereo-memory-dmd/model.safetensors` 34.9 GB；`worldstereo-memory` 22.3 GB；`worldstereo-camera` 10.9 GB（后两者只存非骨干/差异权重，缺失键由 Wan 基座补齐） |

### 4.2 关键帧潜空间：Keyframe-VAE

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/07-keyframe-vae.png)

> 图 7：(a) 常规 Video-VAE 把 `[1+T_vid,H,W,3]` 压到 `[1+T_vid/4,H/8,W/8,C]`；(b) Keyframe-VAE 对每个关键帧单独做图像编码，得到 `[1,H/8,W/8,C]×(1+T_kf)`。来源：论文 Figure 9，PDF p.11。

代码 `src/vae_utils.py::keyframe_vae_encode` 的实现非常直接：对 `[B,3,F,H,W]` 的每一帧切片 `images[:,:,i:i+1]` 单独调用 Wan-VAE 编码，再沿时间维拼接。Wan-VAE 是因果卷积，**单帧输入时第一帧本来就按“图像模式”编码**（时间维不压缩），所以无需改动任何权重。

| 张量 | 形状（480×832，21 帧） |
|---|---|
| 关键帧像素 | `[1,3,21,480,832]`，取值 [-1,1] |
| 关键帧潜变量 | `[1,16,21,60,104]`（逐帧编码后拼接，再按 latents_mean/std 归一化） |
| 对比：常规视频潜变量 | 81 帧 → `[1,16,21,60,104]` |

两者形状完全相同。**区别在于每个潜变量帧代表什么**：常规视频里，第 $k$ 个潜变量帧压缩了第 $4k-3$ 到 $4k$ 四帧的时空信息；关键帧模式里，它只代表一个独立视角。因此 21 个关键帧可以覆盖与 81 帧视频相同的视角跨度（相邻关键帧之间的相机运动约是原来的 4 倍），而每帧都保留完整的高频细节。

**为什么这么做？** 论文 Figure 8 和 Table 7 说明了两点：快速相机运动时，时空压缩会把运动模糊和几何扭曲“烘焙”进潜变量，下游重建直接吃到这些伪影；而 3D 重建真正需要的是**视角多样性**，不是播放流畅度，慢运动产生的冗余帧对重建几乎没用。代价是失去时间平滑先验，相邻关键帧之间的一致性只能靠注意力和条件来保证，所以后面要加强相机控制和记忆。

### 4.3 主分支的输入拼装

`pipelines/_pipeline_common.py::prepare_latents` 在 `latent_cond_mode="first_frame_only"` 下构造条件：

| 张量 | 形状 | 构造方式 |
|---|---|---|
| `latents`（噪声） | `[1,16,21,60,104]` | 标准高斯 |
| `first_frame_latent` | `[1,16,1,60,104]` | VAE(起始帧) |
| `zero_latents` | `[1,16,20,60,104]` | VAE(全零图) 复制 20 份 |
| `mask_lat` | `[1,4,21,60,104]` | 第 0 帧全 1，其余 0（Wan I2V 约定：4 通道对应 4× 时间压缩的 4 个子帧） |
| `condition` | `[1,20,21,60,104]` | `cat([mask_lat, first_frame_latent ‖ zero_latents])` |
| `latent_model_input` | `[1,36,21,60,104]` | `cat([latents, condition], dim=1)` |

“其余帧用 VAE(零图)”而不是“把零视频整体编码”，这正是关键帧潜空间带来的改动：每帧独立，所以条件也要逐帧独立构造。

Patch 化：

$$[1,36,21,60,104]\xrightarrow{\text{Conv3d}(36\to5120,\,k=(1,2,2))}[1,5120,21,30,52]\xrightarrow{\text{flatten}}[1,32760,5120]$$

再加上**主分支相机嵌入**（SSM++ 的“隐式相机嵌入”）：

| 张量 | 形状 | 说明 |
|---|---|---|
| `camera_qt` | `[1,21,7]` | 统一归一化后的 w2c：四元数 4 + 平移 3 |
| MLP | 7→2560→5120→5120 | SiLU 激活，最后一层零初始化 |
| 广播 | `[1,21,30,52,5120]`→`[1,32760,5120]` | 同一帧的所有 token 加同一个向量 |

“统一归一化”（`models/camera.py::unified_camera_normalization`）把目标帧和参考帧的相机**放在同一个坐标系**里：以第一个目标帧为单位阵，用目标相机中心的均值去中心化，再缩放使第一帧到中心的距离为 2。参考帧使用同一套变换，所以参考帧与目标帧的相对位姿是被保留的。

### 4.4 相机控制分支：点云渲染 ControlNet

这是论文所说的“Camera Adapter”，代码里是 `WanXControlNet` + `MaskCamEmbed`：

| 步骤 | 张量与形状 | 说明 |
|---|---|---|
| ① 渲染视频编码 | `render_video [1,3,21,480,832]` → Keyframe-VAE → `render_latent [1,16,21,60,104]` | 首帧已替换为起始帧 |
| ② 拼接 | `cat([x_t[:, :20], render_latent])` → `[1,36,21,60,104]` | 前 20 通道是当前噪声 16 + mask 4 |
| ③ patch 嵌入 | Conv3d(36→5120, k=(1,2,2))，**权重从主 DiT 的 patch_embedding 复制初始化** → `[1,32760,5120]` | 复制初始化让 ControlNet 一开始就“看得懂”潜变量 |
| ④ 附加通道 | `cat([render_mask(1), Plücker(6)])` → `[1,7,21,480,832]` | 全像素分辨率 |
| ⑤ MaskCamEmbed | Conv3d(7→256, k=(1,8,8)) → GroupNorm → SiLU → 零初始化 Conv3d(256→5120, k=(1,2,2)) → `[1,32760,5120]` | 8×8 再 2×2，合计 16× 下采样对齐 token 网格 |
| ⑥ 相加 | ③ + ⑤ → `[1,32760,5120]` | |
| ⑦ 降维 | Linear 5120→1024 | |
| ⑧ 20 层控制块 | `SingleAttentionBlock(dim=1024, heads=16, ffn=4096)` | 独立的 3D RoPE（head dim 64） |
| ⑨ 输出 | 每层零初始化 Linear 1024→5120 | 第 $i$ 层输出加到主 DiT 第 $i$ 层（$i<20$）的 hidden states 上 |

Plücker 射线（`get_camera_embedding`）对每个像素计算世界系下的单位方向 $\mathbf{d}$ 和相机中心 $\mathbf{o}$，嵌入为

$$\mathbf{p}=[\mathbf{d},\ \mathbf{o}\times\mathbf{d}]\in\mathbb{R}^{6}$$

这 6 维唯一确定一条射线（与沿射线的位置无关），因此是“相机内外参 → 像素级稠密信号”的标准编码。

**为什么同时用点云渲染和 Plücker？** 点云渲染只在“看得到已有点”的地方提供信息，空洞处没有信号；Plücker 在每个像素都有定义，能告诉模型“这个空洞像素对应的视线朝哪”。两者互补：前者约束已知内容的外观与位置，后者约束未知区域的几何朝向。

**为什么 ControlNet 只有 1024 维、20 层？** 控制信号的信息量远小于主分支的生成内容，用 1/5 宽度、1/2 深度的旁路足以；零初始化输出保证训练初期不扰动预训练 Wan。只注入前 20 层，是延续 Uni3C 的做法：浅层决定布局与几何，深层负责纹理，控制几何只需作用于前半段。

### 4.5 SSM++ 检索记忆：“虚拟横向拼接”

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/09-ssm-rope.png)

> 图 8：SSM++ 的 RoPE 修改。每个目标帧沿宽度与其检索参考视图拼接成 2W，参考视图继承所配对目标帧的时间索引。来源：论文 Figure 11，PDF p.13。

论文的描述是“把检索视图水平拼到目标帧右边”。**代码并没有真的把像素或潜变量拼成 2W**，而是用位置编码实现了等价效果（`models/worldstereo.py` 第 659–677 行）：

1. 先用一个虚拟张量 `cat([hidden_states, hidden_states], dim=-1)` 计算宽度为 $2\times52=104$ 的 3D RoPE，得到网格 `[F=21,H=30,W=104]` 上的旋转频率。
2. 目标 token 取左半边（列 0–51）的 RoPE。
3. 参考 token 取右半边（列 52–103）、时间索引为 `ref_index+1` 的 RoPE。也就是说，参考帧在位置编码上“坐在”它所服务的那个目标帧的右侧。
4. 参考 token 是**独立的一条 token 流**，在每层自注意力中与目标 token 拼成 `[ref; target]` 做全双向注意力（`WanAttnProcessorSparseSpatialSP`），之后再拆开。

| 张量 | 形状（R=8 个参考帧） | 说明 |
|---|---|---|
| 检索参考帧 | `[1,3,8,480,832]` | 来自记忆库 |
| 参考潜变量 | `[1,16,8,60,104]` | Keyframe-VAE |
| 参考输入 | `[1,36,8,60,104]` | `cat([lat, ones(4), lat])`：把参考帧伪装成“干净、被条件化的帧”，以复用 36 通道 patch embedding |
| 参考 token | `[1,12480,5120]` | 8×1560 |
| 参考相机嵌入 | `camera_qt_ref [1,8,7]` → 同一个 MLP | 加到参考 token |
| 自注意力长度 | 12480 + 32760 = 45,240 | 每层、每个 head |
| 交叉注意力 | 只有目标 token 做；参考 token 跳过 | 参考帧不需要文本条件 |
| FFN、adaLN | 两条流共享权重 | `update_ref=True`：参考流逐层更新 |

**为什么“虚拟拼接”与“真实拼接”等价？** 注意力本身对 token 顺序不敏感，token 之间的关系只由内容和 RoPE 决定。只要参考 token 拿到“目标帧右侧、同一时刻”的位置编码，它和目标 token 的相对位置关系就与把两张图横向拼成 2W 完全一样。这样做的好处是：无需每个目标帧都配一个参考（论文说 SSM++ 只检索最相关的关键帧），8 个参考帧可以只对应 21 个目标帧中的部分帧，计算量从“21 对拼接”降到“21+8 帧”。

**为什么要横向拼而不是时间维拼？** 论文 Table 8 的 A* 配置把参考帧改成时间维拼接，PSNR 从 21.06 掉到 19.83，ATE 从 0.048 涨到 0.114。原因可以从视频模型的先验理解：Wan 在预训练中学到的是“同一时间索引、相邻列的 token 属于同一画面的相邻区域”，于是“同一时刻的右侧画面”会被当作同一场景的另一部分来对齐，类似立体匹配；而“另一个时刻的帧”会被当作运动后的画面，模型倾向于生成运动而非对应。

### 4.6 检索策略与记忆库

`src/retrieval_wm.py::PanoramaMemoryBank` 管理记忆库：

| 项 | 代码行为 |
|---|---|
| 初始化 | 读入 `pano_bank` 的 27 张透视图、相机和 16bit 深度 |
| 检索 | 对目标轨迹第 1–20 帧，各自在记忆库中找**视锥体积重叠**最大的相机（near 0.1，far 为 8 倍深度中位数，带角度惩罚）；记录每个检索帧首次服务的目标帧位置作为 `ref_index` |
| 捷径检索 | regular 轨迹的 traj0/traj1 固定从同视角 traj2（向上航拍）的结果里取若干帧；target/wonder 的航拍轨迹从各自 traj0 取 |
| 去冗余 | 若不同的检索帧超过 `max_reference=8`，用 DINOv2 图像特征（权重 0.7）+ 相机特征（0.3）+ 质量偏置 0.1 做质量感知最远点采样，选出 8 张 |
| 更新 | 每条轨迹生成后，从 21 帧中均匀采样 `align_nframe=8` 帧（含相机）加入记忆库，深度先占位 |
| 生成顺序 | view*（traj2 向上 → traj0 右 → traj1 左）→ target* → reconstruct*/traj0 → wonder* → reconstruct*/traj1（迭代环绕最后） |

生成顺序是精心安排的：先生成向上航拍，让后续左右旋转能检索到“高处俯视”的内容；先做物体环绕，再做漫游，最后才做依赖前序结果的迭代重建轨迹。这是一种**课程式的世界填充**。

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/08-memory-training-data.png)

> 图 9：记忆训练的数据构造。(a) GGM 训练时把参考视图点云与 2 个随机新视角的点云合并为全局点云；(b) SSM++ 训练时，真实多视图数据用 30%–90% 时间重叠的“错位检索”，UE 合成数据用多轨迹 3D-FoV 检索。来源：论文 Figure 10，PDF p.12。

### 4.7 采样：4 步 DMD

`models/dmd_scheduler.py::FlowGeneratorScheduler` 的时间步由 `start=250, end=1000, steps=4, shift=5` 决定：

$$t=\mathrm{linspace}(1000,250,4)/1000=[1,0.75,0.5,0.25]\xrightarrow{\sigma'=\frac{5t}{1+4t}}[1.0,0.9375,0.8333,0.625]$$

每步是流匹配的欧拉更新：

$$x\leftarrow x+(\sigma_{i+1}-\sigma_i)\,v_\theta(x,\sigma_i),\quad \sigma_4=0$$

最后一步从 0.625 直接跳到 0，相当于由学生模型一次性预测干净样本。DMD 模式不做 CFG（`guidance_scale=1`），因此每步只需一次前向；未蒸馏的 `worldstereo-memory` 用 UniPC + CFG 5.0。

单条轨迹单步前向的 token 规模：目标 32,760 + 参考 ≤12,480 + ControlNet 32,760（宽 1024），交叉注意力键值 257+512=769 个。8 卡序列并行（`all_to_all_4D` 在 head 维切分）+ FSDP + bf16。

输出解码同样逐帧：`keyframe_vae_decode` → `[1,3,21,480,832]` → 写 16 fps mp4。

### 4.8 训练配置（来自 HF config）

| 项 | memory 模型 | DMD 模型 |
|---|---|---|
| 学习率 | 1e-5（主块 4e-6） | 生成器 2e-6，fake score 4e-7 |
| 步数 | 90,001 | 3,001 |
| 时间步采样 | logit-normal，shift 5 | 同左，t∈[0.02,0.98] |
| 条件 dropout | 渲染 0.05，相机 0.08 | 同左 |
| 冻结 | `freeze_attn2=True`（交叉注意力冻结），`freeze_ffn=False` | 同左 |
| 点云增强 | 裁剪 0.1、边缘 mask 0.5、飞点 0.1；全局/局部扭曲比例为 0 | 同左 |
| 参考帧增强 | 运动模糊 0.1（核 7），亮度/对比度/饱和度抖动 0.15 | 同左 |
| DMD 专有 | — | real score CFG 5.0，fake 每次生成器更新训 5 次，EMA 0.999 |

“全局/局部点云扭曲比例为 0”与论文的消融结论一致：强扭曲增强会削弱 GGM 的几何约束，导致多条轨迹之间几何不一致。

### 4.9 代码与论文的差异

- **Domain-Adaption 冻结策略**：论文最终方案是“冻结 cross-attention 和 FFN”（Table 7 蓝行）；发布的 memory/DMD 配置是 `freeze_attn2=True, freeze_ffn=False`，对应 Table 8 的 Config B“解冻 FFN”。两者分别描述第一阶段和第二阶段，不矛盾，但读论文容易混淆。
- **SSM++ 的“拼接”是位置编码层面的虚拟拼接**，参考帧是独立 token 流、跳过交叉注意力。论文图 11 容易让人误以为输入张量宽度变成了 2W。
- **检索上限**：论文写“最多 $T_r$ 帧（$T_r<T_{kf}$）”，代码默认 8。
- **一个小细节**（推断为实现瑕疵，影响很小）：`batch_sample_rays` 里像素网格 x 用 `arange-0.5`、y 用 `arange+0.5`，两轴的像素中心约定不一致，Plücker 方向在 x 方向有 1 像素偏移。由于训练和推理用同一函数，模型会自然适应。

### 4.10 相关技术拓展

#### 4.10.1 视频扩散的相机控制谱系

| 路线 | 条件形式 | 代表 | 特点 |
|---|---|---|---|
| 位姿向量 | 每帧 $[R\vert{t}]$ 的 MLP 嵌入 | MotionCtrl | 简单，但像素级对应弱 |
| Plücker 射线图 | 每像素 6 维 | CameraCtrl、CamCo | 稠密、与内参解耦 |
| 点云渲染/warp | 把参考帧深度点云渲染到目标视角 | ViewCrafter、Gen3C、Uni3C | 已知区域强约束，依赖深度质量 |
| 3D 记忆缓存 | 持久点云/高斯随生成更新 | Voyager、Lyra | 长程一致，管理成本高 |

WorldStereo 2.0 是第 2+3 类的组合，再叠加检索记忆（类似 Context-as-Memory 的检索思想），并用 GGM 把“参考帧点云”升级为“全景点云”。相关笔记：[[Lyra 2.0 - 可探索的生成式三维世界]]、[[Context as Memory - 历史上下文驱动的场景记忆]]、[[世界模型记忆机制架构演进 - Survey]]。

#### 4.10.2 3D RoPE 与“位置即关系”

Wan 的 3D RoPE 把 128 维 head 拆成时间、高、宽三段分别旋转。SSM++ 的核心洞察是：**在注意力模型里，“把两张图拼在一起”只是一种位置编码约定**。同类技巧在图像编辑（参考图放在右侧的 in-context LoRA）、多视图生成（多视图拼图）里都出现过。它比 ReferenceNet（额外一整套 UNet）便宜得多，代价是参考 token 占用主干的注意力预算。

#### 4.10.3 DMD 蒸馏

DMD 把“学生分布与教师分布的 KL 散度”的梯度写成两个 score 的差：

$$\nabla_\theta\mathcal{L}_{\mathrm{DMD}}=-\mathbb{E}_t\left[\left(s_{\mathrm{real}}(x_t,t)-s_{\mathrm{fake}}(x_t,t)\right)\frac{\partial x_t}{\partial\theta}\right]$$

$s_{\mathrm{real}}$ 是冻结的教师，$s_{\mathrm{fake}}$ 是在线拟合学生分布的模型。WorldStereo 2.0 的特点是**在带记忆的完整任务上蒸馏**（旧版只在相机控制任务上蒸馏、冻结记忆分支），这得益于 UE 合成的多轨迹数据。相关笔记：[[少步视频扩散蒸馏：提升成像质量的六条路线 - Survey]]、[[因果视频蒸馏路线 - Survey]]。

---

## 5. Stage IV-a：WorldMirror 2.0 前馈重建

### 5.1 模型结构与张量流

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/10-worldmirror-arch.png)

> 图 10：WorldMirror 2.0 架构。图像 token 与可选的位姿、内参、深度 token 统一进入共享 Transformer，后接点图、相机、深度、法线、3DGS 五类头。来源：论文 Figure 12，PDF p.15。

HF 配置 `HY-WorldMirror-2.0/config.json`：`patch_embed=dinov2_vitl14_reg, embed_dim=1024, depth=24, num_heads=16, num_register_tokens=4, normalized_rope=true, rope_base=100, rope_normalize_coords="separate", condition_strategy=["token","pow3r","token"], gs_dim=256`。

以独立重建的默认设置举例（`--target_size 952`，输入 4:3 图像 `S` 张）：

| 阶段 | 张量 | 形状 | 说明 |
|---|---|---|---|
| 预处理 | `images` | `[1,S,3,714,952]`，[0,1] | 长边缩到 952，短边按比例并取整到 14 的倍数，超出部分中心裁剪 |
| Patch 嵌入 | `patch_tokens` | `[S,3468,1024]` | DINOv2 ViT-L/14：51×68 个 patch |
| 特殊 token | `cam_token`、`reg_tokens` | `[S,1,1024]`、`[S,4,1024]` | 第一帧与其余帧使用不同的可学习 token（形状 `[1,2,·,1024]`） |
| 位姿先验 | `poses` | `[1,S,7]` → MLP(7→1024→1024) → `[S,1,1024]` | 平移按 5–95% 分位归一化到 [0,1] 立方体，加四元数 |
| 内参先验 | `rays` | `[1,S,4]` → MLP → `[S,1,1024]` | $(f_x/W,f_y/H,c_x/W,c_y/H)$ |
| 深度先验 | `depth` | `[1,S,714,952]` → conv+MLP patch embed → `[S,3468,1024]` | 每张图按 1–99% 分位归一化到 [0,1]，**加到**图像 patch token 上（Pow3R 风格） |
| 序列 | `all_tokens` | `[S,3475,1024]` | `[cam, reg×4, pose, ray, patches]`，缺失先验用零 token 占位 |
| 交替注意力 | 24×(frame block + global block) | frame：`[S,3475,1024]`；global：`[1,S×3475,1024]` | VGGT 式：帧内注意力 + 全局跨帧注意力交替 |
| 中间特征 | 第 4/11/17/23 层 | 4×`[1,S,3475,2048]` | 帧内与全局输出沿通道拼接 |
| 相机头 | 4 层 trunk、迭代细化 | `[1,S,9]` → `c2w [1,S,4,4]`、`K [1,S,3,3]` | 9 = 平移 3 + 四元数 4 + FoV 2 |
| 点图头 | DPT | `pts3d [1,S,714,952,3]` + conf | `inv_log` 激活 |
| 深度头 | DPT | `depth [1,S,714,952,1]` + conf + `depth_mask` | exp / 1+exp / 线性 logit → sigmoid |
| 法线头 | DPT | `normals [1,S,714,952,3]` + conf | L2 归一化 |
| 3DGS 头 | DPT（features 256） | `gs_feat [1,S,128,714,952]` + gs 深度/conf/mask | 再经 Conv(128→256)→ReLU→Conv(256→12) |
| 高斯参数 | 12 通道拆分 | quat 4、scale 3、opacity 1、残差 SH 3、weight 1 | SH 阶数 0；颜色 = RGB2SH(像素) + 残差；中心 = gs 深度沿相机射线反投影；0.002 体素合并 |

DPT 头把 4 个尺度的 token 特征重组为特征图，逐级上采样回全分辨率，所以所有稠密输出都与输入像素一一对应。每个像素一个高斯，`S` 张 714×952 的图就是 `S×68万` 个初始高斯，靠置信度过滤和体素合并控制数量。

### 5.2 在世界生成链中的调用方式

这是读代码才能看到的关键信息。`PanoramaMemoryBank.apply_worldmirror()` 实际执行的命令是：

```bash
torchrun --nproc_per_node=8 -m worldrecon.pipeline \
    --input_path   <bank>/world_mirror_data/images \
    --prior_cam_path <bank>/world_mirror_data/cameras.json \
    --target_size 832 \
    --no_save_gs --no_save_normal --no_save_points \
    --no_sky_mask --no_edge_mask \
    --use_fsdp --enable_bf16 \
    --disable_heads normal points gs
```

| 项 | 取值 | 含义 |
|---|---|---|
| 输入视图 | 记忆库全部帧：27 张 pano_bank + 每条轨迹 8 帧 | 35 条轨迹时约 307 张 |
| 分辨率 | 832 → 向下取整到 826，高 476 | 59×34=2006 个 patch |
| 每帧 token | 2006 + 7 = 2013 | |
| 全局注意力序列 | 约 307×2013≈61.8 万 token | 只能靠 8 卡序列并行 |
| 先验 | 位姿 + 内参（来自轨迹规划，完全已知） | 不给深度先验 |
| 启用的头 | 相机 + 深度（含深度 mask） | 法线、点图、高斯头全部关闭 |
| 输出 | 每帧 `depth_XXXX.npy` `[476,826]` | 之后最近邻缩放回 480×832 |

**为什么关掉其他头？** 生成链只需要“在已知相机下的深度”。相机是规划出来的、精确已知；法线由 MoGe-2 提供；点图可由深度+相机直接得到；高斯在后面单独优化。关掉三个 DPT 头可以省下大量显存（每个 DPT 头都要在全分辨率上展开 `S` 张特征图）。

**为什么必须带相机先验？** 生成的关键帧来自扩散模型，帧间存在细微不一致；如果让 WorldMirror 自己估计相机，它会把这些不一致“解释”成相机误差，导致位姿漂移。给定相机后，模型只需回答“在这个相机下每个像素多远”，不一致只会体现在深度残差上，后续对齐和 3DGS 还能消化。论文 Figure 15 显示，在相同相机条件下，WorldMirror 2.0 的点云比 MapAnything、DA3 更一致。

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/13-recon-compare.png)

> 图 11：生成关键帧上的重建对比：(a) MoGe-2 全景点云参考，(b) MapAnything，(c) DA3，(d) WorldMirror 2.0，(e) WorldMirror 2.0 + 深度对齐。前馈方法都去掉了 10% 最低置信度点。来源：论文 Figure 15，PDF p.20。

### 5.3 Normalized RoPE：为什么能从 518 训练分辨率跑到 1036

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/11-normalized-rope.png)

> 图 12：(a) 中心点 RoPE 编码与其他分辨率的平均余弦相似度：归一化 RoPE 始终 >0.995，标准 RoPE 在 2048 时跌到 0.008；(b)(c) 编码值的均值和标准差。来源：论文 Figure 13，PDF p.17。

标准 2D RoPE 用整数网格坐标 $(i,j)$；归一化 RoPE 改为

$$\hat x_i=\frac{2i+1}{H_p}-1,\quad \hat y_j=\frac{2j+1}{W_p}-1$$

`separate` 表示两个轴各自归一化，保留长宽比信息。这样训练分辨率 37×37 patch 与推理 74×74 patch 都落在 $[-1,1]$，高分辨率只是更密的采样，**外推变内插**。这也解释了生成链为什么敢用 826×476 这种非训练分辨率，而 WorldMirror 1.0 在高分辨率时 RealEstate10K 相机 AUC@30 从 86.13 掉到 66.29。

`rope_base=100` 比 LLM 常用的 10000 小得多，因为图像网格最多几十到上百个位置，低基频让相邻 patch 的相位差更大、位置更可分。

### 5.4 训练侧的形状管理：Token 预算动态 batch

论文 Eq.(8)–(9)：固定每卡 token 预算 $T_{\max}$（如 25,000），先采样分辨率（5 万–50 万像素）和长宽比，算出单图 token 数 $t$，再决定最多视图数：

$$N_{\max}=\min\left(N_{\mathrm{cap}},\left\lfloor\frac{T_{\max}}{t}\right\rfloor\right),\quad N\cdot\frac{H}{p}\cdot\frac{W}{p}\le T_{\max}$$

这与 LLM 训练里的 sequence packing 是同一个思想。它让模型在训练中见过“高分辨率少视图”和“低分辨率多视图”的各种组合，这对推理时 307 视图×2006 patch 的极端配置是必要的。

### 5.5 相关技术拓展

#### 5.5.1 VGGT 式交替注意力

帧内注意力负责单视图的 2D 语义与几何，全局注意力负责跨视图对应；交替堆叠使每层都能在两种尺度间交换信息。相机 token 放在每帧序列首位，最后由相机头读出。WorldMirror 在此之上加了先验 token，并在图像 token 上直接叠加深度嵌入。详见 [[VGGT - 前馈式通用三维重建]]、[[VGGT 系列：几何状态的强化与复用 - Survey]]、[[π³ - 置换等变视觉几何学习]]。

#### 5.5.2 先验注入的三种方式

| 方式 | 适用先验 | WorldMirror 的选择 |
|---|---|---|
| 额外 token | 全局量（位姿、内参） | 位姿、内参各 1 个 token |
| 逐 patch 相加 | 稠密量（深度、射线图） | 深度：conv+MLP 后加到图像 token（Pow3R 风格） |
| 输入通道拼接 | 稠密量 | 未使用（会破坏 DINOv2 预训练的输入分布） |

训练时每种先验以 0.5 概率独立丢弃，缺失时用零 token 占位，所以推理时可以任意组合。和 MapAnything 的“所有输入统一因子化”相比，WorldMirror 更轻量，但先验的表达能力受限于单个 token。详见 [[WorldMirror - 多模态先验驱动的通用前馈三维重建]]、[[LDM多条件控制的三种注入方式 - Survey]]。

#### 5.5.3 深度—法线耦合

论文 Eq.(5)–(6)：由预测深度反投影得到点图，用有限差分叉积得到法线，与目标法线算角度损失。真实数据的法线来自单目法线教师（因为单目伪深度在多视图间不一致，而法线是局部量，天然不需要全局尺度一致）。这是一种**用局部几何量弥补全局几何标签缺失**的通用技巧。

---

## 6. Stage IV-b：深度对齐

### 6.1 对齐流程

![](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/figures/12-depth-alignment.png)

> 图 13：深度对齐流程。全景点云按相机渲染得到引导深度 $D^g_i$（红框为遮挡错误），与可靠性 mask $M_i$ 一起，把 WorldMirror 深度 $D^m_i$ 线性对齐为 $D^a_i$，再做异常检测。来源：论文 Figure 14，PDF p.19。

`PanoramaMemoryBank.alignment()` 的五个阶段（代码第 1320 行起）：

| 阶段 | 做什么 | 关键参数 |
|---|---|---|
| ① 视频分组 | pano_bank 27 帧按 8 帧一组成“虚拟视频”；每条轨迹的 8 帧为一个视频；按 rank 分配 | `align_nframe=8` |
| ② 单目先验 | 每帧跑 MoGe-2 得到 mask 与法线；**深度替换为 WorldMirror 深度**；室外场景用 SAM3 视频模式（文本 `"sky"`）去天空 | — |
| ③ 逐帧拟合 | 渲染全景点云得引导深度/法线；构造有效 mask；在**逆深度空间**用 RANSAC 拟合 $1/D^g=k\cdot(1/D^m)+b$ | 见下表 |
| ④ 锚点异常检测 | 把每帧的 $(k,b)$ 作用到一组锚点逆深度上，与全局中位数比较 | P90 相对偏差 |
| ⑤ 修复与导出 | 异常帧借用同视频最近的正常帧的 $(k,b)$；全异常视频丢弃；反投影、统计滤波、自适应体素下采样到 ≤2M 点 | `pcd_nb_neighbors=10, std_ratio=2.0` |

第 ③ 步的有效 mask 是五个条件的交集：

| mask | 代码实现 | 作用 |
|---|---|---|
| 引导深度有效 | 点云渲染有覆盖，且覆盖率 ≥15%（`valid_threshold`） | 只在有锚点的地方对齐 |
| 分位一致 | 引导深度和单目深度各自转成分位图，差值 >20 个百分点的像素剔除 | 消除图 13 中的遮挡错误 |
| 单目有效 | MoGe mask（含天空剔除） | |
| 非深度边缘 | `depth_edge(rtol=0.05)` 取反 | 去飞点 |
| 法线一致 | MoGe 法线转世界系后与全景法线夹角 ≤90° | 去背面/穿透 |

再加两个约束：有效像素占比 <5% 直接判失败；只用深度 ≤8 倍中位数的“近处”像素拟合。RANSAC 使用带约束的线性回归（$k\ge10^{-4}$，$b\in[-10,10]$），最少 100 个样本，停止概率 0.995。

### 6.2 为什么在逆深度（视差）空间对齐

论文脚注 2 说“为了更好的前景对齐，在视差空间做对齐”。直观理解：深度空间的线性拟合由远处大数值主导，一个远处 50 米、误差 5 米的点，在最小二乘里的权重远超近处 1 米、误差 0.1 米的点；而 3DGS 和漫游最在意的恰恰是近处几何。视差空间把远处压缩、近处拉开，和人眼/相机的感知精度分布一致。另外，单目/前馈深度在远处本来就不可靠，`8× 中位数` 截断和视差空间是同一思路的两层保险。

### 6.3 锚点异常检测：把耦合的 $(k,b)$ 变成一维问题

直接对 $k$ 和 $b$ 分别做离群检测是错的：两个参数高度耦合，$k$ 大一点、$b$ 小一点可能在常用深度范围内效果几乎一样。代码的做法是看“效果”而不是“参数”：

$$V_{i,q}=k_i\cdot\frac{1}{A_q}+b_i,\quad \hat V_q=\mathrm{median}_j V_{j,q},\quad \Delta_i=\max_q\frac{\left\lvert V_{i,q}-\hat V_q\right\rvert}{\left\lvert\hat V_q\right\rvert}$$

$\Delta_i$ 超过全体帧的 90% 分位即判为异常。

**代码与论文的差异**：论文说用 $Q=9$ 个在深度范围内均匀分布的锚点；代码的锚点是自适应的：最小深度到中位数之间 5 个（0、1/4、1/2、3/4、中位数），再加 1.5/2/3/4 倍中位数（不超过最大深度时），再加中远过渡点和最远点（不超过 8 倍中位数），最多 11 个，**近处密、远处疏**，与视差空间对齐的偏好一致。

### 6.4 相关技术拓展：尺度—平移对齐的家族

| 方法 | 对齐空间 | 鲁棒性手段 | 代表 |
|---|---|---|---|
| 最小二乘尺度+平移 | 深度或视差 | 无 | MiDaS 评测协议 |
| 中位数尺度 | 深度 | 中位数 | 自监督单目深度评测 |
| RANSAC 线性 | 视差 | 随机一致 | HY-World 2.0 |
| 非线性/分段 | 深度 | 样条或网络 | Gen3C、video2world |
| 联合 BA | 点 + 相机 | 鲁棒核 | COLMAP、DROID-SLAM |

论文报告这条轻量方案在 300 张生成图上不到 2 分钟，而基于特征匹配 ICP 的 video2world 约 5 小时（论文 8.1.4）。它能“偷懒”的前提是：相机完全已知、WorldMirror 深度已经足够好，残差主要是每帧一个尺度和偏移。

---

## 7. Stage IV-c：3DGS 优化与 Mesh 提取

### 7.1 训练数据（`gen_gs_data.py`）

| 来源 | 视图数（默认） | 图像 | 深度 | 法线 |
|---|---:|---|---|---|
| 全景透视视图 | 27×2=54（`pano_density_mult=2`） | 从 ERP 重切 480×832 | 全景深度 | MoGe-2 |
| 上半极区视图 | 8×4=32（`polar_up_density_mult=4`） | 同上 | 同上 | 同上 |
| 下半极区视图 | 8 | 同上 | 同上 | 同上 |
| 生成关键帧 | 每条轨迹 21 帧，约 35 条 → 约 735 | WorldStereo 输出 | 只有对齐过的 8 帧有深度（稀疏监督） | MoGe-2（全部帧） |

初始点云：全景点云（≤3M，超出随机下采样）+ 对齐扩展点云（≤2M）+ 天空点云（≤300K，`--split_sky` 单独保存）。相机写成自定义 `cameras.json`（每图一个 extrinsic/intrinsic），不是 COLMAP 格式。

上极区采样密度是下极区的 4 倍，原因推断为：室外场景的天空和室内天花板在漫游时经常抬头可见，而地面区域已经被大量轨迹视角覆盖。

### 7.2 高斯参数化

| 参数 | 形状 | 存储空间 | 初始化 | 学习率 |
|---|---|---|---|---:|
| `means` | `[N,3]` | 世界坐标 | 初始点云 | 1.6e-4 |
| `scales` | `[N,3]` | log | KNN 距离 | 5e-3 |
| `quats` | `[N,4]` | 未归一化四元数 | 随机 | 1e-3 |
| `opacities` | `[N]` | logit | logit(0.1) | 5e-2 |
| `sh0` | `[N,1,3]` | SH 0 阶（即 RGB） | RGB2SH(点颜色) | 2.5e-3 |
| `mask_score` | `[N,2]` | MaskGaussian 的二类 logit | `[4.0, 1.0]`（偏向保留） | 1e-2 |

协方差由 $\Sigma_k=R_kS_kS_k^TR_k^T$ 构造。**SH 阶数为 0**：论文观察到生成场景的视角相关效应很弱，而高阶 SH 在生成视图上会过拟合不同帧之间的颜色漂移，在新视角产生色斑（论文 Figure 20(d)）。用纯 RGB 等价于告诉优化器“颜色不随视角变”，强行把帧间颜色不一致平均掉。训练器里还实现了逐图像双边网格（16×16×8）做曝光/色调补偿，但默认关闭，README 推荐命令也没有打开，说明作者认为 RGB 颜色本身已足以吸收帧间外观差异。

### 7.3 损失与调度

README 推荐命令下的完整目标（`world_gs_trainer.py`）：

$$\mathcal{L}=0.8\,L_1+0.2\,(1-\mathrm{SSIM})+w_{\mathrm{lpips}}\,\mathrm{LPIPS}+w_d\,\mathcal{L}_{\mathrm{disp}}+w_n\,\mathcal{L}_{\mathrm{normal}}+\mathcal{L}_{\mathrm{scale}}+0.003\left(\tfrac{1}{N}\textstyle\sum_kM_k\right)^2$$

训练器还实现了天空 TV 平滑、深度畸变损失（distortion）、不透明度正则和双边网格 TV，但在推荐命令下都处于关闭状态。天空的处理改用 `--sky_depth_from_pcd`：先把天空点云渲染成深度，在“没有 GT 深度、法线判为天空、点云有覆盖”三个条件同时满足的像素上补进深度监督。

带调度的权重按 $w=\lambda_2+\mathrm{decay}\cdot(\lambda_1-\lambda_2)$ 变化，`decay` 从 1 线性降到 0：

| 项 | 公式 | 起始→结束权重 | 含义 |
|---|---|---|---|
| LPIPS | 感知损失 | 0.2 → 0.1 | 早期重结构，后期让位给像素精度 |
| 视差 L1 | $\lVert1/\hat{D}-1/D^a\rVert_1\cdot{s_{\text{scene}}}$ | 0.2 → 0.01 | **早期用深度定几何，后期放手** |
| 法线 | $1-\cos(\hat{N},N_{\text{MoGe}})$，$\hat{N}$ 由渲染深度求导 | 0.1 → 0.5 | **后期用法线修表面** |
| 尺度正则 | $0.1\cdot\mathrm{mean}(\max(r,5)-5)$，$r$ 为最大/最小轴比，每 10 步 | 常数 | 抑制针状高斯 |

深度权重递减、法线权重递增，这个设计很值得注意：对齐深度只有尺度/偏移意义上的正确，细节可能有误；法线是局部量、来自 MoGe-2，细节更可靠。所以先让深度把高斯“摆到大致正确的位置”，再让法线把表面“抹平、对齐方向”。

### 7.4 致密化与 MaskGaussian

| 机制 | 参数 |
|---|---|
| 训练步数 | 1500（8 卡；4 卡 2000，2 卡 4000，单卡 8000） |
| 致密化 | 第 150–750 步，每 100 步；`grow_grad2d=1e-4`；`prune_scale3d=0.1`；不重置不透明度（`reset_every=99990`） |
| 锚点保护 | 天空点等受保护点不参与致密化 |
| MaskGaussian | 第 300–1500 步启用；Gumbel-Softmax 温度 0.7，直通估计；每 1000 步剪枝“采样 4 次从未存活”的高斯；导出时按保留概率 0.2 随机采样 |

MaskGaussian 修改了光栅化器的合成公式（`third_party/gsplat_maskgaussian`）：

$$c(x)=\sum_{k=1}^{N}M_k\,c_k\,\sigma_k\,T_k,\quad T_{k+1}=T_k\,(1-M_k\sigma_k)$$

$M_k=0$ 的高斯既不贡献颜色、也不消耗透射率，等价于不存在；但通过 Gumbel-Softmax 的直通梯度，它仍能收到“如果存在会怎样”的梯度，因此可以“复活”。论文 Table 9：完整配置相对 6M 基线减少 77% 高斯，PSNR 仅降 0.153 dB。

### 7.5 Mesh 提取

`gs/extract_mesh.py`：从所有训练视角渲染 RGB-D，融合进 Open3D TSDF（代码默认体素 0.004、截断 0.02、深度截断 3.0，支持 GPU `VoxelBlockGrid`），Marching Cubes 提取三角网格，保留最大的 50 个连通分量。这个 mesh 主要用于 WorldLens 里的碰撞代理，不追求拓扑质量。

### 7.6 相关技术拓展

#### 7.6.1 为什么生成场景的 3DGS 和重建场景不一样

| 维度 | 真实重建 | 生成场景（HY-World 2.0） |
|---|---|---|
| 视图一致性 | 物理一致 | 帧间存在细微几何/颜色不一致 |
| 视角相关效应 | 镜面、高光真实存在 | 基本是扩散模型的噪声 |
| 深度监督 | 可有可无 | 必须（防止为解释不一致而产生漂浮物） |
| 天空 | 通常远景无穷远 | 无深度监督，致密化会产生大量飞点 |
| 颜色模型 | SH 3 阶 | RGB（SH 0 阶） |

这张表解释了 Stage IV-c 几乎每一个“魔改”的动机。相关笔记：[[3DGS - 实时辐射场渲染的三维高斯表示]]。

#### 7.6.2 Gumbel-Softmax

对离散变量 $M\in\{0,1\}$，从 logit $\pi$ 采样时加 Gumbel 噪声 $g$，用 $\mathrm{softmax}((\pi+g)/\tau)$ 作为可微近似；`hard=True` 时前向取 one-hot、反向用软概率的梯度（直通估计）。温度 $\tau=0.7$ 是“够离散又有梯度”的折中。

---

## 8. 横向分析：计算量、瓶颈与复现

### 8.1 各阶段的“token 账”

| 阶段 | 单次前向序列长度 | 宽度 | 层数 | 前向次数 |
|---|---:|---:|---:|---|
| HY-Pano（HunyuanImage-3） | 约 7320 图像 + ≤5120 条件 + CoT 文本 | 4096（MoE 8/64） | 32 | 50 步 × 2（CFG） |
| WorldStereo（DMD） | 32,760 目标 + ≤12,480 参考 | 5120 | 40（+20 层 1024 宽 ControlNet） | 每条轨迹 4 步 × 约 35 条 |
| WorldMirror（生成链） | 约 307 帧 × 2013 ≈ 61.8 万（全局层） | 1024 | 24+24 | 1 次 |
| 3DGS | — | — | — | 1500 步 × 8 卡 |

WorldStereo 部分的总前向约 140 次 14B 模型调用，每次 4.5 万 token，这就是 286 秒的来源。WorldMirror 只有一次前向，但全局注意力的序列长度达到 60 万级，必须依赖 token 级序列并行（每层 all-to-all）和帧级 DPT 并行。

### 8.2 一张表看懂“先验的接力”

| 上游产物 | 下游如何使用 | 作为什么 |
|---|---|---|
| 全景 ERP | 切 27/16 张透视图 | 记忆库初值、3DGS 监督 |
| 全景深度/点云 | 轨迹渲染、GGM、引导深度 | 几何先验、对齐锚点 |
| 全景网格/NavMesh | 碰撞、可行走区域 | 轨迹约束 |
| VLM 地标 + SAM3 | 环绕/重建轨迹的目标 | 语义先验 |
| 轨迹相机 | Plücker、WorldMirror 相机先验、3DGS 相机 | **全链路唯一精确已知的量** |
| 生成关键帧 | 记忆库、WorldMirror 输入、3DGS 监督 | 伪观测 |
| WorldMirror 深度 | 对齐后反投影 | 扩展点云 |
| MoGe-2 法线 | 对齐 mask、3DGS 法线损失 | 局部几何先验 |

“相机完全已知”是整个系统能以前馈 + 线性对齐这种轻量方式闭环的根本原因。这与真实视频重建（相机未知、需要 BA）是完全不同的问题设定。

### 8.3 复现成本

| 组件 | 显存/硬件门槛（推断） | 外部依赖 |
|---|---|---|
| HY-Pano（HunyuanImage-3） | 约 170 GB bf16 权重，需多卡 | 或改用 Qwen-Image-Edit + 0.85 GB LoRA |
| 轨迹规划 | 1 卡 + vLLM（Qwen3-VL-8B，README 示例用 8 卡 TP） | GroundingDINO、ZIM、SAM3、MoGe-2、cupy、Recast（需编译） |
| WorldStereo | 8 卡 FSDP（检查点 34.9 GB） | Wan2.1-I2V-14B 基座、umT5、CLIP |
| WorldMirror | 4–8 卡（数百视图时） | DINOv2 |
| 3DGS | 8 卡；自定义 gsplat 需编译 | fused-ssim、LPIPS |

### 8.4 复现建议

1. **先单独跑 WorldMirror**：最容易验证，也最能体会归一化 RoPE 与先验注入的效果。
2. **用自带 `examples/worldgen/case000/panorama.png` 跳过 Stage I**，从 `traj_generate.py` 开始，先看 `render_results/` 下的 `pano_bank`、各轨迹 `render.mp4` 是否合理。
3. **调试记忆时打开 `alignment(debug_mode=True)`**：它会把引导深度、分位差、有效 mask 逐帧存到 `tmp_debug/`，是理解对齐失败模式最直接的材料。
4. **想替换视频模型时**，接口约束是：输入 `[1,36,21,60,104]`、输出 `[1,16,21,60,104]`、支持外加 ControlNet 残差和参考 token 流；换成 Wan2.2 的 5B TI2V 时，代码里已经为 16× VAE（`"5B" in base_model`）留了分支。

---

## 9. 判断与展望

### 9.1 已确立的事实

- 世界生成链的“世界”由 WorldStereo 2.0 生成，WorldMirror 2.0 在链中只做“已知相机下的深度回归”，不负责补全内容。
- 关键帧潜空间、虚拟横向拼接的检索记忆、视差空间对齐、RGB 颜色 + MaskGaussian 的 3DGS，都是代码里可以逐行核对的设计。
- 开源推理代码中，HY-Pano 的接缝处理是“多生成 32 列 + 线性融合”，未见论文所述的潜空间 circular padding；对齐锚点是自适应的约 11 个，而非论文所说的均匀 9 个。

### 9.2 我的判断

1. **这套系统的真正创新是接口，不是模型。** 每个模型都是在已有基座上微调（HunyuanImage-3、Wan2.1、DINOv2/VGGT 范式、gsplat），但“21 帧关键帧 ↔ 21 个潜变量帧”“480×832 贯穿始终”“相机规划后全链路已知”“参考帧伪装成 36 通道干净帧”这些接口设计，让异构模型几乎没有阻抗失配。读代码比读论文更能看出这一点。
2. **最脆弱的一环是全景深度。** 它同时决定 NavMesh、轨迹碰撞、轨迹渲染条件、GGM、对齐锚点和天空分割。一旦 MoGe-2 在某个方向估错，错误会以“先验”的身份传遍全链路，且没有任何环节会回头修正它。
3. **WorldStereo 的记忆在本质上仍是 2D 检索。** 记忆库存的是 RGB + 相机，检索靠视锥重叠，一致性靠注意力“看图对齐”。显式 3D 只以“渲染条件”的形式进入，生成结果不会反过来更新全景点云（更新发生在全部生成结束后的对齐阶段）。这意味着前后生成的轨迹之间只能通过记忆库里的图像保持一致，而不是通过一个持续演化的 3D 状态。把对齐提前到每条轨迹之后、让 GGM 使用“滚动更新的点云”，是一个很自然的改进方向。
4. **对研究的启发**：如果要做“生成式重建”的下游，WorldMirror 的“相机已知 → 只回归深度”是很好的模板；如果要做 3D 资产或 mesh 质量，HY-World 2.0 的 TSDF mesh 只是碰撞代理，拓扑质量完全没有被优化，这里有很大空间。

### 9.3 后续值得追的问题

- 潜空间环形填充是否只用于训练？可以对比开源推理在不同随机种子下的接缝统计。
- 把检索打分从“视锥重叠”换成“在当前点云下的可见性重叠”，能否减少错误检索？
- DMD 学生在 MipNeRF360 上略差于教师（50.52 vs 51.27 F1），差距是否集中在大视角变化的帧上？
- WorldMirror 对生成帧的深度残差能否被显式估计，并作为 3DGS 的逐像素置信度权重，而不是只用二值 mask？

---

## 10. 文献与代码索引

### 10.1 本系统直接相关

| 工作 | 角色 | 笔记 |
|---|---|---|
| HY-World 2.0（2026-04） | 本文对象 | [[HY-World 2.0 - 多模态三维世界生成重建与模拟]] |
| HunyuanWorld 1.0（2025-07） | 全景 + 分层 mesh 前身 | [[HunyuanWorld 1.0 - 分层可探索三维世界生成]] |
| WorldMirror 1.0（2025-10） | 前馈重建前身 | [[WorldMirror - 多模态先验驱动的通用前馈三维重建]] |
| HY-World 1.5 / WorldPlay（2025-12） | 在线视频世界模型路线 | [[WorldPlay - 长期一致的实时交互世界模型]] |
| Voyager（2025-09） | 世界缓存 + RGB-D 视频扩散 | [[Voyager - 长程世界一致的RGB-D视频扩散]] |
| WorldStereo 1.0、FlashWorld、Uni3C | 相机控制与关键帧思想来源 | 暂无独立笔记 |

### 10.2 关键技术组件溯源

| 组件 | 出处 | 本文位置 |
|---|---|---|
| HunyuanImage 3.0 | 腾讯混元统一多模态生成模型 | 2.2.1 |
| Qwen-Image-Edit-2509 | 阿里通义 MMDiT 图像编辑模型 | 2.2.2 |
| MoGe / MoGe-2 | 微软，单目几何 | 3.1、6.1、7.1，[[MoGe系列 - Survey]] |
| Recast Navigation | 开源 NavMesh 库 | 3.2 |
| SAM3、GroundingDINO、ZIM、Qwen3-VL | 分割、检测、抠图、VLM | 3.1、6.1 |
| Wan2.1-I2V-14B | 阿里通义视频扩散 | 4.1 |
| CameraCtrl / Plücker | 相机射线编码 | 4.4 |
| DMD / DMD2 | 分布匹配蒸馏 | 4.7、4.10.3 |
| DINOv2、DPT、VGGT | 重建骨干与头 | 5.1、5.5.1，[[VGGT - 前馈式通用三维重建]] |
| DINOv3 归一化 RoPE | 分辨率泛化 | 5.3 |
| Pow3R、MapAnything | 先验注入 | 5.5.2 |
| 3DGS、gsplat、MaskGaussian | 显式表示与稀疏化 | 7.2–7.4，[[3DGS - 实时辐射场渲染的三维高斯表示]] |
| TSDF + Marching Cubes | Mesh 提取 | 7.5 |

### 10.3 相关综述

- [[前馈式三维重建与生成式补全 - Survey]]
- [[生成式世界模型路线的前馈三维重建 - Survey]]
- [[世界模型记忆机制架构演进 - Survey]]
- [[VAE、Attention、DiT 与 Flow Matching 基础模型路线 - Survey]]
- [[少步视频扩散蒸馏：提升成像质量的六条路线 - Survey]]

### 10.4 材料来源

完整来源、图片映射、代码版本与核验记录见 [sources.md](assets/HY-World%202.0%20工作原理与张量流%20-%20Survey/sources.md)。
