---
type: paper-note
status: done
domain: World-Models
paper: "FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction"
year: 2025
venue: ICLR 2026
arxiv: "2509.21657"
source: https://arxiv.org/abs/2509.21657
project: https://fantasy-amap.github.io/fantasy-world/
code: https://github.com/Fantasy-AMAP/fantasy-world
tags:
  - World-Models
  - Video-Diffusion
  - 3D-Reconstruction
  - Geometry
created: 2026-10-15
updated: 2026-10-15
---

# FantasyWorld：统一视频与三维预测

## 基本信息

- **作者**：Yixiang Dai, Fan Jiang, Chiyu Wang, Mu Xu, Yonggang Qi
- **机构**：AMAP, Alibaba Group；Beijing University of Posts and Telecommunications
- **任务**：给定参考图、可选文本和目标相机轨迹，一次前向过程联合生成相机条件视频与隐式 3D 表征
- **基础模型**：Wan2.1 video diffusion（核心 backbone 冻结）+ 可训练 VGGT-style geometry branch
- **官方披露核验（2026-10-15）**：论文、项目页、官方代码与模型权重均已公开；官方提供论文配置和增强配置。论文未报告推理速度/显存、完整模型总参数量、公开训练数据预处理脚本的逐项成本。

## 一句话结论

FantasyWorld 不在视频生成后再做独立重建，而是从冻结 WanDiT 的中间 latent 分叉出几何分支，并用双向 cross-attention 让视频先验与深度/点图/相机监督互相约束；其强项是固定长度、多视角几何一致性，不是实时或长期交互世界。

## 1. 问题、动机与贡献

视频基础模型有强大的外观和遮挡区域想象能力，却缺乏显式 3D grounding；3D foundation model 几何可靠，却没有视频生成先验。以往串联两者会产生 feature/domain mismatch，或必须每场景运行 NeRF/3DGS 优化。

![](assets/FantasyWorld%20-%20统一视频与三维预测/figures/01-teaser.png)

> **图解**：输入图像和相机轨迹后，模型沿目标视角生成视频，同时输出相机/几何特征，可形成更完整的场景结构。来源：论文 **Figure 1, PDF p.1**，https://arxiv.org/pdf/2509.21657（获取：2026-10-15）。

贡献：
1. 冻结视频主干并增加几何分支，实现统一的 video latent + implicit 3D field 预测。
2. PCB 先把纯噪声变成有结构 latent，再交给几何分支，降低梯度方差。
3. IRG 的非对称双分支和 MM-BiCrossAttention 实现 geometry→video 与 video→geometry 双向约束。
4. 3D DPT heads 同时预测 depth、point maps、camera poses，提供可复用 3D 表征。

## 2. 输入、输出与任务边界

- **输入**：第一帧图像、可选文本、目标 camera trajectory；CLIP 编图、umT5 编文、Plücker ray camera encoder 编相机。
- **视频输出**：336×592、81 帧（论文配置）的 camera-conditioned video。
- **几何输出**：每帧 depth $D$、3D point map $P$、uncertainty $\Sigma_P$、9D camera 参数（旋转、平移、FOV）。
- **边界**：模型生成固定长度 clip；没有在线 memory、KV streaming、动作闭环或长期状态；论文的 3DGS 指标仍包含后处理重建。

## 3. 总体架构：PCB + IRG

![](assets/FantasyWorld%20-%20统一视频与三维预测/figures/02-architecture.png)

> **图解**：前 16 个冻结 Wan blocks 作为 PCB；后续 24 个 block 与几何分支组成 IRG。想象分支输出视频，几何分支由 3D DPT/camera head 输出 depth、point map 与 pose。来源：论文 **Figure 2, PDF p.4**，https://arxiv.org/pdf/2509.21657（获取：2026-10-15）。

### 3.1 PCB（Preconditioning Blocks）

扩散 latent 在早期接近纯噪声，直接做 3D 监督会产生高方差。PCB 复用 Wan2.1 前 16 层冻结 denoiser，把 noisy latent 变成初步结构化特征。相机 adapter 只预测 shift $\beta_i$：

$$f_i=f_{i-1}+\beta_i,$$

并只施加于 40 个 transformer blocks 的前 24 个。

![](assets/FantasyWorld%20-%20统一视频与三维预测/figures/03-pcb-analysis.png)

> **图解**：PCA 可视化显示，随去噪 timestep 和网络深度推进，建筑轮廓逐渐清晰；红框 block 16 是 IRG 输入，说明几何分支不是从纯噪声学习。来源：论文 **Figure 3, PDF p.5**，https://arxiv.org/pdf/2509.21657（获取：2026-10-15）。

### 3.2 IRG（Integrated Reconstruction and Generation Blocks）

- **Imagination Prior Branch**：复用冻结 Wan2.1 的 appearance-rich 时空特征。
- **Geometry-Consistent Branch**：将 Wan latent 映射到 geometry-aligned latent；全局/帧注意力建模跨视角结构。
- **MM-BiCrossAttention**：几何约束视频的多视角一致性；视频先验帮助几何补全遮挡区。

## 4. 双向跨分支公式与 3D Heads

对 video tokens $X_v$ 和 geometry tokens $X_g$：

$$A=\mathrm{softmax}\left(\frac{Q_vK_g^\top}{\sqrt{d_k}}\right),$$
$$X_v^+=X_v+\gamma_vAV_g,\qquad X_g^+=X_g+\gamma_gA^\top V_v.$$

可学习 gate $\gamma_v,\gamma_g$ 控制双向信息流，避免一开始破坏冻结视频先验。

3D DPT 从 geometry branch 的 layers $\{8,12,18,24\}$ 取特征，并反转传统 DPT 的 reassemble 逻辑：深层 diffusion feature 更干净，故被更强上采样。两个 temporal blocks 各把时间分辨率翻倍，最终

$$T=4(t-1)+1.$$

输出 depth、point map/uncertainty 和 camera pose。

## 5. 两阶段训练与损失

### 5.1 Stage 1：Latent Bridging

冻结 Wan2.1，仅训练 geometry branch，使其消费 block-16 hidden feature：

$$L_{geo}=L_{depth}+L_{pmap}+3L_{camera}.$$

- camera：Huber loss；
- depth：Video Depth Anything 风格的 temporal gradient matching + per-frame scale-sensitive loss；
- point map：位置/局部梯度误差并由 uncertainty 加权。

### 5.2 Stage 2：Unified Co-Optimization

在后 24 个 blocks 后加入双向 adapter，只微调交互模块与 camera adapter，主干仍冻结。总目标：

$$L_{total}=\mathbb E\|\epsilon_\theta(z_t,t,c)-\epsilon\|_2^2+\lambda L_{geo}.$$

论文未披露 $\lambda$ 以及 Stage 1 中 depth/point/camera 子损失的全部标量权重（正文符号化、附录仅给最终 camera 系数），这是精确复现缺口。

## 6. 数据、训练与评测

### 6.1 数据

约 180K clips：RealEstate10K、ACID 使用重建管线生成一致深度；DL3DV、WildRGB、ScanNet、TartanAir 用 CUT3R 生成几何标签。真实与仿真混合，但各子集数量、采样比例和过滤阈值 **未披露**。

### 6.2 训练

- AdamW，lr $10^{-5}$。
- Stage 1：20K steps，global batch 64，64×H20，36 小时。
- Stage 2：10K steps，81 帧、592×336/336×592，global batch 112，112×H20，144 小时。
- 仅计算已披露主阶段约为 **18,432 H20-hours**（2304 + 16128），不含数据标注、基础模型与失败实验。

### 6.3 评测

WorldScore static photorealistic 1,000 cases；另设 100 个最大 90° 轨迹的 Large split。几何 fidelity 在 RealEstate10K 100 cases 上用 3DGS 后处理评估 PSNR/SSIM/LPIPS。

## 7. 主要定量、定性结果与消融

WorldScore 关键一致性指标：

| 设置 | 模型 | 3D Consist. ↑ | Photo Consist. ↑ | Style Consist. ↑ |
|---|---|---:|---:|---:|
| Small | Ours w/o 3D | 79.77 | 83.86 | 92.54 |
| Small | **Ours w/ 3D** | **83.31** | **86.11** | **94.22** |
| Large | Ours w/o 3D | 72.06 | 56.98 | 81.59 |
| Large | **Ours w/ 3D** | **74.83** | **60.61** | **82.02** |

完整模型在一致性指标领先，但 Camera Ctrl/Object Ctrl/Content Align 并非最佳。例如 Small setting 的 camera ctrl 57.05，低于 WonderWorld 92.32；说明论文优化重点是几何稳定，而非指令控制。

![](assets/FantasyWorld%20-%20统一视频与三维预测/figures/04-world-generation-results.png)

> **图解**：WonderWorld 有缺洞，Voyager 多视角错位，AETHER 细节低，Uni3C 有风格突变；FantasyWorld 的店面与室内结构更稳定。来源：论文 **Figure 4, PDF p.8**，https://arxiv.org/pdf/2509.21657（获取：2026-10-15）。

3DGS reconstruction：

| 方法 | 初始化 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| Ours w/o 3D | VGGT | 26.89 | 0.84 | 0.17 |
| **Ours w/ 3D** | **VGGT** | **28.24** | **0.86** | **0.14** |
| Ours w/ 3D | 自身 point cloud | 26.54 | 0.85 | 0.19 |

同一 VGGT 初始化下，3D branch 带来 +1.35 dB PSNR、+0.02 SSIM、-0.03 LPIPS。自身点云无需外部 VGGT 仍可用，但 LPIPS 较差，说明“完全替代外部几何模型”尚未达到最佳。

![](assets/FantasyWorld%20-%20统一视频与三维预测/figures/05-geometry-results.png)

> **图解**：统一重建流程下，FantasyWorld 的墙角闭合、室外招牌与立面更清楚；Uni3C/Voyager/AETHER 出现分层或重复结构。来源：论文 **Figure 5, PDF p.9**，https://arxiv.org/pdf/2509.21657（获取：2026-10-15）。

## 8. 失败案例、局限与 Future Work

1. 作者明确承认只支持固定长度 clip；连续长程生成需要对 implicit 3D representation 设计 cache/streaming state。
2. 大视角下虽优于 baseline，但 Large split 的 Photo Consist. 仅 60.61，说明显著遮挡与新区域仍困难。
3. camera/object control 指标不是强项；几何约束不会自动带来精确交互控制。
4. 3D fidelity 指标依赖后接 3DGS，尚不能完全隔离视频质量、VGGT/自身点云初始化和优化器的贡献。
5. 训练标签大量来自重建/CUT3R 伪标签，会继承尺度、动态物体和弱纹理误差。
6. Future work：把隐式 3D 特征变为可持久更新的 memory，加入自回归/流式生成，并在导航等下游任务直接验证 task-agnostic 表征。

## 9. 复现成本与个人判断

- 已披露训练本身约 18,432 H20-hours；此外需 180K 视频几何标注和 Wan2.1 权重。
- 主干虽冻结，但 Stage 2 的 81-frame 双分支、多层 cross-attention、3D heads 仍需 112 张 H20；复现不是“轻量 adapter 微调”。
- 官方公开论文配置权重使推理验证可行，但论文未给显存、步数与 wall-clock 推理速度，部署预算不能仅凭论文估算。
- **个人判断**：论文最有价值之处是把几何监督放进视频 latent 的生成过程，而不是事后约束 RGB。结果证明双向互馈有效，但目前更像“几何增强的 camera-controlled video model”，离持续可交互、可操作物体的世界模型仍有明显距离。

## 10. 材料来源与披露状态

- 本地 PDF、5 张官方源码图与来源清单：`../../assets/FantasyWorld - 统一视频与三维预测/`。
- 详细清单：`../../assets/FantasyWorld - 统一视频与三维预测/sources/sources.md`。
- **已披露**：PCB/IRG 结构、训练步数/batch/lr、GPU 数/时长、数据来源、评测 split、代码与权重。
- **未披露**：推理速度/显存、总参数量、数据子集配比、关键 loss 权重 $\lambda$、长期/交互评测。
