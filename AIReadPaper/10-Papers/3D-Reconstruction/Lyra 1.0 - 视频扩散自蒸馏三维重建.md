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
status: detailed-read
created: 2026-10-15
source: "https://arxiv.org/abs/2509.19296"
---

# Lyra 1.0：视频扩散自蒸馏三维重建

## 1. 基本信息与一句话结论

**论文全名**：*Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation*。论文作者来自 NVIDIA、University of Toronto、Vector Institute 与 Simon Fraser University，录用 ICLR 2026。官方全文见 [arXiv:2509.19296](https://arxiv.org/abs/2509.19296)，本地 PDF 为 `assets/Lyra 1.0 - 视频扩散自蒸馏三维重建/sources/paper.pdf`。

> **一句话结论**：Lyra 把相机可控视频扩散模型当成“虚拟多相机摄影棚”，用其 RGB 解码结果监督一个直接读取视频 latent 的 3DGS 学生解码器，从而无需任何真实多视图训练数据，就能由单图一次前向生成静态 3DGS、由单目视频生成可换时间与视角的动态 4DGS。

![Lyra 1.0 能力总览](assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig1_teaser.png)

*图 1 解读：上半部分从单图一次前向输出 3DGS 并支持任意新视角；下半部分从单目视频输出随目标时间变化的动态 3DGS。来源：论文 Figure 1，Page 1；官方源码 `imgs/teaser.pdf`。*

## 2. 研究背景、核心问题与贡献

学习式 3D 重建长期受制于数据：静态场景需要有位姿的多视图，动态场景还需要同步多相机；前馈重建虽免去 per-scene 优化，却仍依赖有限的 3D 数据，分布外泛化差。与此同时，互联网规模训练的视频扩散模型在手持、航拍、电影镜头中隐式学到了物体和场景如何随视角变化，但其输出只是 2D 帧，不能直接实时渲染、交互或进入物理仿真。

Lyra 将二者拼成“generative 3D scene reconstruction”：视频模型负责想象未观测区域并生成带相机位姿的多视图，3DGS 解码器负责把这些视图压成显式、可渲染的世界。关键问题由“怎样采集更多真实多视图”变成“怎样把视频模型的隐式 3D 知识蒸馏出来”。

论文贡献有三层：第一，建立冻结视频教师、训练 3DGS 学生的自蒸馏框架，监督完全来自合成视频；第二，提出在压缩 video latent 上处理 726 个高分辨率视图的前馈解码器；第三，以源时间/目标时间条件和倒放增强，把静态框架最小改动扩展到动态 4D。

## 3. 输入、输出与总体框架

Lyra 的静态输入可以是文本或单张图像。文本先经图像扩散模型生成 seed image；单图则直接进入相机可控 GEN3C 教师。每个 seed image 采样 6 条轨迹，每条 121 帧，形成 726 个$704×1280$合成视图。输出是一组显式 3D Gaussians，每个 Gaussian 含位置、尺度、旋转、不透明度和 RGB，可实时 splat 渲染。

动态输入是一段带相机估计的单目视频；输出不是一个固定 3DGS，而是目标时间$t_{tgt}$条件下的 3DGS。改变$t_{tgt}$得到不同运动状态，改变相机得到同一状态的新视角，因此形成 4D 表示。

![Lyra 自蒸馏与解码器](assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig3_method.png)

*图 2 解读：左侧冻结 RGB encoder、Video Diffusion Transformer 与 RGB decoder，只训练 3DGS decoder；学生渲染结果对齐教师 RGB。右侧 video latent、Plücker 相机、源时间与目标时间分别 patchify 后相加，再经多视图重建块和转置 3D 卷积输出 Gaussians。来源：论文 Figure 4，Page 5；官方源码 `imgs/method.pdf`。*

## 4. 教师—学生机制与完整公式

### 4.1 GEN3C 教师提供什么

GEN3C 从输入帧深度建立时空点云 cache，再沿目标轨迹渲染 cache 作为视频扩散条件。Lyra 使用升级后的 GEN3C，其 VAE latent 通道$C=16$，空间压缩因子$σ=8$、时间压缩因子$τ=8$：

$$z=E(X)∈R^{L'×C×h×w},\quad L'=(L-1)/τ+1,\quad h=H/σ,\quad w=W/σ.$$

Lyra 并不让学生读取 RGB 教师输出后再重建，而是让学生和 RGB decoder 共享同一 denoised latent。这减少了像素级输入数量，也避免再次运行视觉编码器。

### 4.2 自蒸馏闭环

给定输入图$I$和轨迹$C^{1:L}$，教师生成 denoised video latent：

$$z=V(I,C^{1:L}).$$

冻结的 RGB decoder 得到教师视频$X_{rgb}=D_{rgb}(z)$；学生输出 Gaussians$G=D_s(z,E_{cam})$，并在同一组相机下渲染$X_{gs}=Render(G,C^{1:L})$。训练让$X_{gs}$逼近$X_{rgb}$。这里没有真实 3D、真实深度或真实多视图标签；ViPE 深度也来自教师生成视频，仍属于合成监督。

每个 seed image 使用 6 条轨迹、每条 121 帧。学生不是分别输出 6 个场景再硬合并，而是联合读取所有轨迹 latent，让 token 之间在重建块中交换信息，直接输出一个一致 3DGS。

### 4.3 为什么必须在 latent 空间重建

像素空间共有$6×121×704×1280$个观测，若每像素生成一个 Gaussian，将产生 654,213,120 个候选且注意力不可承受。视频 latent 已做时空压缩，Lyra 再以$2×2$ patchify 转 token。附录还说明最终每个$8×8$空间邻域只生成一个 Gaussian，把数量降到 10,222,080；按 opacity 剪掉 80% 后约 2,044,416 个。

### 4.4 3DGS 解码器

video latent 与相机 Plücker embedding 都具有$V×L'×C×h×w$形状。Plücker 的 6 维由 ray direction 和$origin×direction$组成；两组 3 通道各自经冻结 RGB encoder 编码，拼成$2C$通道，再由轻量 MLP 降回$C$。两路 patch token 相加后进入两个重建 block；每个 block 是 1 层 Transformer 加 7 层 Mamba-2，共 16 层、hidden size 512。

转置 3D 卷积输出每像素 14 维：位置$(x,y,z)$、尺度$(s_x,s_y,s_z)$、四元数$(q_w,q_x,q_y,q_z)$、opacity$α$和颜色$(r,g,b)$。Mamba-2 的线性序列建模把一次前向从纯 Transformer 的 20,922 ms 降到 3,213 ms，约 6.5 倍。

### 4.5 损失

总损失为

$$L=λ_{mse}L_{mse}+λ_{lpips}L_{lpips}+λ_{depth}L_{depth}+λ_{opacity}L_{opacity},$$

权重依次是$1.0,0.5,0.05,0.1$。$L_{mse}$和$L_{lpips}$约束教师/学生 RGB；仅 RGB 会产生“浅浮雕”几何，所以用 ViPE 对教师视频估计一致深度，再采用 Long-LRM 的 scale-invariant depth loss；opacity 的 L1 正则让大部分无贡献 Gaussian 可被剪枝。

## 5. 动态 4D 扩展

动态学生$D_d$从静态$D_s$初始化，新增源时间$T^{src}$与目标时间$T^{tgt}$。原始时间是 1 通道标量，与 2 维 sinusoidal embedding 拼成 3 通道，再由冻结 RGB encoder 压到 latent 尺度。新增时间 patchify 层零初始化，避免微调开始即破坏静态能力：

$$G=D_d(Z,E_{cam},T^{src},T^{tgt}).$$

动态监督的陷阱是：目标时间$t$只能由同一时刻的帧监督。原始 6 条相机轨迹从近视角向远视角运动，早期时间只有近视角、晚期时间只有远视角，网络会学会忽略其他时间帧，并在极端视角下出现 opacity collapse。Lyra 把输入视频倒放后再送入教师，得到 6 条由远到近的轨迹，再倒回原运动时间，与原 6 条由近到远轨迹合并。于是每个时间步都有 12 个监督视角，且近/远成对。该增强只用于训练，推理无需额外轨迹。

这个设计的实质不是普通时间反转，而是重新平衡“运动状态—相机距离”的联合分布，使学生不能靠时间位置猜视角难度。

## 6. 数据引擎、训练与推理

### 6.1 全合成数据

静态数据含 59,031 张由多样文本 prompt 生成的图像，每张 6 条视频，共 354,186 个视频。动态数据含 7,378 个生成视频，每个扩展 6 条轨迹，共 44,268 个视频。prompt 覆盖室内、室外、人、动物、现实与幻想内容。论文明确不使用任何现成真实多视图数据训练 3DGS decoder。

### 6.2 渐进训练

完整$V=6,L=121,H=704,W=1280$成本过高，因此静态模型分六阶段：

| 阶段 | 分辨率 | $L$ | $V$ | 每步监督帧$S$ | Batch | Steps |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 176×320 | 17 | 1 | 17 | 4 | 10K |
| 2 | 176×320 | 49 | 1 | 49 | 4 | 2.5K |
| 3 | 352×640 | 49 | 1 | 49 | 2 | 2.5K |
| 4 | 704×1280 | 49 | 1 | 49 | 1 | 2.5K |
| 5 | 704×1280 | 121 | 1 | 9 | 1 | 57.5K |
| 6 | 704×1280 | 121 | 1～6 | 9 | 1 | 7K |
| 动态 7 | 704×1280 | 121 | 6 | 12 | 1 | 10K |

总训练约 6 天、8 张 A100 80GB，3DGS 实现使用 gsplat。推理只需视频 diffusion 得到 latents，再运行一次 3DGS decoder；RGB decoder 在最终 3D 输出路径中可丢弃，也不需要 per-scene 优化或后处理。

## 7. 实验设置与主要定量结果

静态单图到 3D 采用 RealEstate10K、DL3DV、Tanks-and-Temples，指标为 PSNR/SSIM/LPIPS。论文指出部分 baseline 未开源，因此主表使用其论文报告数字；这一点限制了严格统一测试的程度。

| 方法 | RE10K PSNR / SSIM / LPIPS | DL3DV PSNR / SSIM / LPIPS | T&T PSNR / SSIM / LPIPS |
|---|---|---|---|
| ZeroNVS | 13.01 / 0.378 / 0.448 | 13.35 / 0.339 / 0.465 | 12.94 / 0.325 / 0.470 |
| Wonderland | 17.15 / 0.550 / 0.292 | 16.64 / 0.574 / 0.325 | 15.90 / 0.510 / 0.344 |
| Bolt3D | 21.54 / 0.747 / 0.234 | — | — |
| **Lyra** | **21.79 / 0.752 / 0.219** | **20.09 / 0.583 / 0.313** | **19.24 / 0.570 / 0.336** |

在自建 Lyra 数据集上，与“GEN3C 生成 12 帧 + BTimer 像素空间重建”的统一 baseline 比较：静态由 16.32/0.580/0.427 提升到 24.92/0.834/0.183；动态由 20.29/0.687/0.315 提升到 23.07/0.779/0.231。提升不仅来自教师，因为 baseline 同样使用 GEN3C；核心差异是 latent 多视图聚合可读取 726 帧，而 BTimer 受内存限制只能采样 12 帧。

![Lyra 静态主结果](assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig4_main_results.png)

*图 3 解读：每个场景展示多个 3DGS 新视角，覆盖自然景观、室内、机器人与城市；它主要证明合成训练数据的语义覆盖，而非单个场景的严格几何误差。来源：论文 Figure 6，Page 8；官方源码 `imgs/main_results.png`。*

## 8. 消融、失败案例与证据解读

完整模型在 Lyra 数据集上为 PSNR 24.77、SSIM 0.837、LPIPS 0.224。关键消融如下：

| 变体 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|---|---:|---:|---:|---|
| Real data only | 19.08 | 0.659 | 0.413 | 真实 RE10K+DL3DV 多样性不足 |
| Self-distill + real data | 24.74 | 0.823 | 0.236 | 加真实数据未超过纯自蒸馏 |
| w/o depth loss | 24.31 | 0.811 | 0.247 | 图像指标小降，但几何显著压扁 |
| w/o opacity pruning | 24.55 | 0.820 | 0.237 | 表示更大，渲染 30ms 而非 18ms |
| w/o LPIPS | 23.74 | 0.766 | 0.370 | 高频与输入不一致鲁棒性下降 |
| w/o multi-view fusion | 17.73 | 0.632 | 0.446 | 独立重建再合并严重失败 |
| w/o Mamba-2 | 24.58 | 0.818 | 0.241 | 质量略降，速度慢 6.5 倍 |
| w/o latent 3DGS | OOM | OOM | OOM | 像素空间无法承载全部视图 |

![深度损失消融](assets/Lyra%201.0%20-%20%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E8%87%AA%E8%92%B8%E9%A6%8F%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA/figures/fig8_depth_ablation.png)

*图 4 解读：去掉 depth loss 后 RGB 外观看似仍合理，但汽车和飞船的深度轮廓收缩成浅层；这说明仅以新视角 RGB 指标判断 3D 质量可能漏掉几何退化。来源：论文 Figure 11，Page 26；官方源码 `imgs/ablation_depth.pdf`。*

失败案例和适用边界包括：教师 GEN3C 的视角范围与一致性直接限定学生上限；视频教师对遮挡区的错误想象会被学生固化为几何；高斯仍约 200 万个，虽可实时渲染但不是轻量资产；动态解码器依赖 ViPE 相机/深度与教师生成质量，快速非刚体运动、遮挡交换和拓扑变化并未单独量化；主表部分 baseline 使用已报告结果而非同一代码重跑。

实验总体能支撑“视频教师可取代真实多视图监督”和“latent 空间可扩展到大量视图”。但“显式 3D 保证几何一致”应理解为表示层面的强约束，不等于恢复真实几何；学生学习的是教师可渲染的一致解释。

## 9. 局限、Future Work 与个人判断

作者承认场景尺度与一致性受相机可控视频模型能力限制，并把自回归大场景生成列为 future work；动态方向则建议在重建网络中显式建模 motion 与 tracking。论文没有解决长程回访：一次前向的 726 个视图仍来自有限生成窗口，无法持续穿过多房间再回到原处。

我的判断是，Lyra 最重要的不是 3DGS decoder 的某个局部模块，而是**监督来源的改写**：生成模型不再只负责最终媒体，而成为可按需采样、带相机标签的数据引擎。它把视频模型升级直接转化为 3D 数据质量提升；同时也把 teacher bias、幻觉和授权问题一并传给下游 3D。对物理 AI 而言，Lyra 适合生成大量视觉上可信的训练环境，但若任务依赖度量几何、碰撞边界或因果动力学，仍需额外校验与物理化。

后续最自然的工作是：让学生增量更新并维护长程记忆；把静态/动态统一而非两个解码器分支；用对象轨迹与 scene graph 约束非刚体运动；减少 Gaussian 数量并直接输出可碰撞 mesh；给生成式 3D 引入不确定性，区分观测、教师补全和学生推断区域。

## 10. 复现资源、图表索引与缺失信息

论文 Reproducibility Statement 承诺发布 3D/4D 训练与推理代码、权重和数据。官方项目页为 <https://research.nvidia.com/labs/toronto-ai/lyra/>，代码仓库为 <https://github.com/nv-tlabs/lyra>。核心依赖包括 GEN3C/Cosmos 视频教师、ViPE、Flux 类图像生成器、gsplat、Transformer/Mamba-2 重建块与 3DGUT 导出工具。官方 PDF 和源码包已保存为 `sources/paper.pdf`、`sources/arxiv-source.tar`，逐项来源见 `sources.md`。

复现建议按静态最小闭环推进：固定 GEN3C，先生成少量多轨迹 latent/RGB；实现 Plücker encoder 与 3DGS decoder；以 MSE+LPIPS 对齐教师，再加入 ViPE 深度与 opacity pruning；确认单轨迹稳定后再扩展 6 轨迹和渐进分辨率；最后从静态权重微调动态时间条件与倒放增强。

**图表索引**：

1. `fig1_teaser.png`：静态 3D / 动态 4D 能力，Figure 1 / Page 1。
2. `fig3_method.png`：自蒸馏与解码器结构，Figure 4 / Page 5。
3. `fig4_main_results.png`：单图到 3DGS 结果，Figure 6 / Page 8。
4. `fig8_depth_ablation.png`：深度监督消融，Figure 11 / Page 26。

**仍缺信息**：官方论文没有给出 prompt 全量清单、合成数据去重/过滤规则、GEN3C 教师 checkpoint 哈希、完整动态失败集和最终发布数据的精确许可拆分；这些会显著影响严格复现与商业使用判断，应以仓库实际文件和各模型权重许可证为准。
