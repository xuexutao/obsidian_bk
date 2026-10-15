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
venue: SIGGRAPH Asia 2026
arxiv: "2604.13036"
tags:
  - paper
  - 3d-reconstruction
  - world-generation
  - long-video
  - spatial-memory
  - 3dgs
status: detailed-read
created: 2026-10-15
source: "https://arxiv.org/abs/2604.13036"
---

# Lyra 2.0：可探索的生成式三维世界

## 1. 基本信息与一句话结论

**论文全名**：*Lyra 2.0: Explorable Generative 3D Worlds*。作者均来自 NVIDIA，论文已被 SIGGRAPH Asia 2026 接收。官方全文见 [arXiv:2604.13036](https://arxiv.org/abs/2604.13036)，本地 PDF 为 `assets/Lyra 2.0 - 可探索的生成式三维世界/sources/paper.pdf`。

> **一句话结论**：Lyra 2.0 通过“逐帧几何只负责检索和建立对应、扩散模型负责外观生成”的空间记忆解决回访遗忘，再用一次去噪构造自劣化历史解决自回归漂移，使单图能够沿用户轨迹持续扩展为数百帧、可回望、可重建成 3DGS/mesh 的大尺度世界。

![Lyra 2.0 总览](assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig1_teaser.png)

*图 1 解读：用户从单图开始迭代指定相机轨迹，模型能穿行、回望并扩展到约 90 米；生成视频再被重建为 3DGS/mesh，可进入交互查看器和机器人仿真。来源：论文 Figure 1，Page 1；官方源码 `figures/teaser_v3.pdf`。*

## 2. 从 Lyra 1.0 到 2.0：问题重新定义

Lyra 1.0 已证明视频扩散先验可被蒸馏成显式 3D，但一次生成只覆盖有限相机窗口。要从“生成一段局部场景”变成“用户可持续探索的世界”，视频模型必须跨越房间、街道、大视角旋转和数百帧后的位置回访。论文把失败归纳为两个不同问题：

- **Spatial forgetting**：早期观察离开有限 context window 后，回到旧区域时模型只能重新编造布局。
- **Temporal drifting**：每个自回归 chunk 的轻微色偏、模糊和结构误差被下一段当作条件，逐段复利。

已有 3D memory 路线把历史帧融合成全局点云并渲染为后续条件。它的几何约束强，但生成伪影会污染深度，坏几何又污染条件，形成错误放大。另一类方法把历史 RGB 和相机 embedding 塞进上下文，不产生损坏的全局点云，却要求 self-attention 在大视角差下自己寻找远程几何对应。

Lyra 2.0 的核心贡献是把二者拆开：**显式几何只做信息路由，不直接提供外观。** 每帧几何独立保存，用于选对历史帧、告诉 attention 哪些位置对应；真正像素仍由扩散先验合成。第二个贡献是 self-augmentation，让训练条件接近推理时的自生成历史。第三个贡献是把长视频域适配后的 Depth Anything v3 重建器和分层 mesh 提取接到生成端，形成可部署资产。

## 3. 输入、输出与 retrieve–generate–update 总体循环

输入包括单张 seed image$I_0$、用户分段指定的相机内外参轨迹${(T_i,K_i)}$和可选文本 prompt。每次生成一个 80 帧 chunk，系统维护：历史 RGB、每帧深度、相机参数、下采样点云、FramePack 时间上下文和检索出的空间 slots。输出首先是长视频，随后是紧凑 3DGS 与 surface mesh。

![Lyra 2.0 方法总览](assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig2_method.png)

*图 2 解读：左侧是交互式 retrieve–generate–update 循环；右侧 spatial memory 先按目标视角召回历史帧，再把 canonical coordinate warp 成稠密 3D 对应，连同历史图像 slots 和时间 slots 注入 DiT。来源：论文 Figure 2，Page 6；官方源码 `figures/method_v3.pdf`。*

一次循环分三步：

1. **Retrieve**：预测目标轨迹的可见区域，从所有历史帧中选最多 5 个、联合覆盖最大的视图。
2. **Generate**：将空间 slots、warped correspondence、FramePack 时间 slots、相机条件和文本送入 Wan2.1-14B DiT，生成下一段。
3. **Update**：对新帧估深度，逐帧保存几何，不进行全局点云融合，然后允许用户继续规划轨迹。

## 4. 视频骨干、相机控制与上下文压缩

### 4.1 Flow matching 骨干

Wan 2.1 causal VAE 对空间下采样 8 倍、时间下采样 4 倍，latent 通道$C=16$：

$$z=E(x)∈R^{F'×C×h×w},\quad F'=floor((F-1)/4)+1,\quad h=H/8,\quad w=W/8.$$

对干净 latent$z_0$和噪声$ε∼N(0,I)$，线性插值得到

$$z_t=(1-t)z_0+tε.$$

DiT$v_θ$回归速度$ε-z_0$：

$$L_{flow}=E[||v_θ(z_t,t,c)-(ε-z_0)||^2].$$

### 4.2 双重相机条件

第一路是 GEN3C 式 depth-based warping：把最近帧按估计深度前向 warp 到每个目标相机，经 VAE 编码后与 denoising latent 拼接，提供精确局部相机控制。大视角下若没有像素落到目标视图，这一路会完全失效，因此另加 6D Plücker ray$(d,o×d)$，经 pixel shuffle 与线性层映射到 token hidden dimension，并加到每层 attention 的 query/key 输入。前者给可见内容，后者在空洞时仍提示射线几何。

### 4.3 FramePack 时间记忆

FramePack 在固定 token budget 中按时间距离改变空间压缩：近帧细粒度、远帧粗粒度。原始布局可记为：1 个全分辨率 anchor；16 个$k=4$远历史帧、2 个$k=2$中距帧、1 个$k=1$近帧；最后是待生成 20 帧 token。anchor 始终是$I_0$，用于抑制整体风格漂移。但相机探索时早期帧常与新区域无视觉重叠，所以仅拉长时间记忆不足以解决空间回访。

## 5. Anti-forgetting：逐帧几何检索与稠密对应

### 5.1 几何缓存为何绝不融合

每个历史帧$I_i$保存全分辨率深度$D_i$、相机$(T_i,K_i)$，以及按$d=8$下采样再反投影的点云$P_i$。全分辨率深度用于准确 correspondence，下采样点云只用于快速检索。关键是$P_i$彼此独立：生成视频越长，单帧深度越可能有误；若融合为一个全局点云，跨视图错位会永久写入世界模型。独立保存把错误限制在单帧，检索时让多个观察相互补充。

### 5.2 Geometry-aware retrieval

对目标相机$(T^*,K^*)$，将每个$P_i$投影到目标平面。每个目标 pixel 先在所有历史点中求最小投影深度$d_{min}$处理遮挡；若某点深度与$d_{min}$之差小于阈值$δ=0.1$，则视为可见。帧$i$的 visibility score$φ(i)$就是可见点数。

训练时按$φ(i)$比例采样历史帧，使网络适应非最优检索；推理时贪心选择：每轮取覆盖“尚未覆盖目标 pixels”最多的帧，最多$N_s=5$。这比独立选 top-5 更好，因为后者容易选出五张相邻、彼此冗余的图。只要数百帧后的回访与某个旧观察有 3D overlap，召回就不依赖时间距离。

### 5.3 空间 slots 与 canonical coordinates

被召回的 5 帧单独经 VAE image encoding，不做时间压缩。空间 slots 采用 4 帧$k=2$、1 帧$k=1$；与 anchor、时间 slots、generation tokens 一起进入完整 DiT self-attention。

仅放 RGB 还不够，大视角下 attention 不知道哪个历史 pixel 对应哪个目标 pixel。Lyra 2.0 给第$j$个历史帧定义三通道 canonical map：

$$C_j(u,v)=(u,v,2j/N_s-1),$$

再利用该帧全分辨率深度前向 warp：

$$C'_j=FwdWarp(C_j,D_{s_j},T_{s_j},T^*,K_{s_j},K^*).$$

warped depth 作为第 4 通道，得到$[C'_j;D'_j]$。它经 sinusoidal positional encoding、pixel shuffle 和线性层聚合后，注入每个 Transformer block 的 query/key，而不改 value。直觉是：对应信号决定“生成 token 应看哪个历史 token”，外观内容仍来自原历史 value 与预训练先验。

论文刻意不 warp RGB，因为 RGB warp 会带来反遮挡洞、拉伸和深度边界渗色；网络很容易把这些伪影复制到输出。canonical coordinates 携带同等几何对应，却不携带坏外观。这是全文最关键的设计原则：**几何只回答去哪找，不能替扩散模型回答画什么。**

## 6. Anti-drifting：self-augmentation 与加速蒸馏

### 6.1 Observation bias

训练时模型看真实历史，推理时看自己的不完美输出，这种 train–test gap 导致误差累积。Self-Forcing 若对双向视频模型模拟完整历史，需要约 35 步去噪和全 attention，代价过高。

Lyra 2.0 以概率$p_{aug}=0.7$对干净历史 latent$z_0^{hist}$采样$t∼U(0,0.5)$并加噪：

$$z_t^{hist}=(1-t)z_0^{hist}+tε.$$

只做一次 DiT forward，得到近似干净历史：

$$z'^{hist}_0=z_t^{hist}-t·v_θ(z_t^{hist},t,c).$$

训练当前 chunk 时，用$z'^{hist}_0$替代完美历史作为条件；但目标$z_0^{cur}$始终由干净历史的 causal VAE cache 编码。于是网络必须在坏历史条件下恢复干净当前帧，而不是把历史误差复制下去。额外成本只有一次 forward。

### 6.2 DMD 4 步模型

完整模型使用 35 步 FlowUniPC 和 CFG。论文再用 Distribution Matching Distillation 把教师蒸馏成 4 步学生，并把 CFG 合并进单次条件前向；蒸馏时保留 self-augmentation。每 80 帧从单张 GB200 的约 194 秒降到 15 秒，约 13 倍，空间检索本身低于 1 秒，真正瓶颈仍是 DiT 去噪。

## 7. 3DGS/mesh 重建、训练数据与设置

### 7.1 生成视频如何落成 3D

Lyra 2.0 不沿用 Lyra 1.0 的自家 3DGS decoder，而采用 Depth Anything v3 前馈模型预测 per-pixel Gaussian 属性。预训练 DAv3 有两个问题：高分辨率每像素一个 Gaussian 数量过大；真实图像训练不足以容忍生成视频的轻微多视不一致。作者将 Gaussian DPT head 下采样$k=2$，Gaussian 数量降 4 倍，并在自生成视频上微调。

微调集由 DL3DV 图像/轨迹出发，自回归生成 3,000 段一分钟视频；DAv3 训练 10K iterations，学习率$5×10^{-5}$、batch size 8。mesh 提取用 OpenVDB/FVDB 分层稀疏网格：靠近生成相机用细 voxel，远背景用粗 voxel；由 Gaussian median depth 和法线构建 SDF，各层 marching cubes 后拼接并 decimate。

### 7.2 视频模型训练

训练数据为 DL3DV 10K 长视频；ViPE 估相机，DAv3 估逐帧深度，Qwen3-VL-8B-Instruct 生成 caption。每个视频采样 1,000 帧。30% 概率做 I2V：单帧条件生成最初 80 帧；70% 做 chunk 自回归训练：随机取历史前缀，下一段 80 帧作为目标。

骨干是 Wan2.1-14B，分辨率$832×480$。AdamW 学习率$3×10^{-5}$、weight decay 0.1、batch size 64，使用 64 张 NVIDIA GB200 训练 7,000 iterations，bf16；新模块零初始化，使训练初始行为等于预训练 Wan。推理文本 CFG scale 5.0，完整模型 35 steps。

## 8. 主要定量结果、消融与失败案例

### 8.1 长视频生成

论文在 DL3DV-Evaluation（域内）和 Tanks-and-Temples（域外）同时报告 SSIM/LPIPS/FID、WorldScore 主观质量/风格一致性/相机可控性，以及基于 ViPE 的 reprojection error。T&T 关键结果如下：

| 方法 | SSIM↑ | LPIPS↓ | FID↓ | 主观质量↑ | 风格一致↑ | 相机控制↑ | 重投影误差↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| GEN3C | 0.350 | 0.589 | 79.07 | 21.75 | 75.54 | **70.91** | **0.054** |
| SPMem | 0.383 | 0.571 | 60.11 | 34.41 | 79.68 | 45.07 | 0.059 |
| **Lyra 2.0** | **0.384** | 0.552 | 51.33 | **43.35** | **85.07** | 63.87 | 0.069 |
| Lyra 2.0 DMD | 0.362 | **0.545** | **49.71** | 43.02 | 78.91 | 58.12 | 0.077 |

GEN3C 的硬 depth warp 获得最高相机控制与最低重投影误差，却牺牲画质；SPMem 画质较强但全局点云误差拖累长程风格；Lyra 2.0 在画质、一致性和控制之间取得更均衡结果。DMD 的单帧 LPIPS/FID 甚至略好，但长程 style 和 camera 指标下降，说明少步蒸馏不是无损替代。

![约第 800 帧长视频比较](assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig5_video_results.png)

*图 3 解读：约第 800 帧时，VMem 已结构崩塌，GEN3C/Yume 出现扭曲，CaM/SPMem 有明显漂移；Lyra 2.0 在回访区域仍保留较真实的几何和外观。来源：论文 Figure 3，Page 10；官方源码 `figures/comparison_grid_all.png`。*

### 8.2 3D 场景生成

Tanks-and-Temples 上，GEN3C+DAv3 的 LPIPS-P/LPIPS-G/FID/主观质量为 0.511/0.694/125.19/5.38；SPMem+DAv3 为 0.412/0.666/94.11/9.95；Lyra 2.0+DAv3 为 0.409/0.648/79.36/14.42；**完整 Lyra 2.0** 为 **0.372/0.629/72.47/18.80**。两层结论应分开：视频生成机制降低 LPIPS-P，说明底层视频更 3D 一致；DAv3 的生成域微调进一步改善 LPIPS-G/FID，说明重建器域适配不可省略。

### 8.3 消融

| 变体 | 主观质量↑ | 风格一致↑ | 相机控制↑ | 重投影误差↓ | 结论 |
|---|---:|---:|---:|---:|---|
| Full | 43.35 | **85.07** | **63.87** | 0.069 | 完整平衡 |
| Global Point Cloud | 44.58 | 82.42 | 49.86 | 0.067 | 融合深度误差损害相机执行 |
| Explicit Corr. Fusion | 45.71 | 83.28 | 57.29 | 0.071 | 硬融合不如 learned aggregation 容错 |
| w/o FramePack | 45.27 | 80.61 | 62.62 | 0.079 | 时间锚点缺失导致漂移 |
| w/o Self-Augmentation | **47.88** | 77.98 | 53.92 | **0.066** | 单帧更漂亮，长程目标反而崩坏 |

![关键组件消融](assets/Lyra%202.0%20-%20%E5%8F%AF%E6%8E%A2%E7%B4%A2%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E4%B8%89%E7%BB%B4%E4%B8%96%E7%95%8C/figures/fig7_ablation.png)

*图 4 解读：全局点云和显式对应融合会受深度错位影响；无 FramePack 出现明显外观漂移；无 self-augmentation 的单帧可能更锐利，却无法维持长程风格与相机轨迹。来源：论文 Figure 6，Page 14；官方源码 `figures/comparison_grid_ablations.png`。*

失败与边界包括：仅支持静态环境，没有继承 Lyra 1.0 的动态 4D；依赖目标视角与历史帧有足够 3D overlap，激进长距离跳跃时检索无从召回；深度错误虽然不直接作为 RGB 条件，仍会通过“选错历史帧/错对应”间接影响生成；DL3DV 的曝光变化会被模型继承，并在 3DGS 重建中产生光度伪影；生成世界视觉可信不代表碰撞、尺度和物理语义真实。

## 9. 局限、Future Work 与个人判断

作者明确提出两个未来方向：把当前静态系统扩展到动态世界；通过网络内 photometric stability 或光度一致的游戏引擎合成数据，减少曝光变化。还可进一步研究检索失败时的 global localization、跨 chunk 的对象身份、动态遮挡与几何不确定性。

我的判断是，Lyra 2.0 最值得保留的原则是“**geometry as routing, not rendering**”。GEN3C 的经验是强几何渲染能提高控制，Lyra 2.0 则发现长期运行时硬几何会把局部错误写成永久事实，因此退一步只让几何决定 attention 路由。这个解耦比具体采用 Wan、DAv3 或 FramePack 更有迁移价值，可用于机器人记忆、多视图 agent、长视频编辑等系统。

但“arbitrarily long”更准确地说是数据结构可持续增长，而不是质量已被证明无限稳定。检索成本虽低于 1 秒，memory 大小、错误深度数量和世界分支歧义仍随时间增加；实验展示约 800+ 帧和特定探索轨迹，不能外推到无限城市或反复闭环。对于仿真用途，应额外验证 metric scale、mesh watertightness、可通行区域和碰撞体。

## 10. 复现资源、图表索引与缺失信息

官方项目页为 <https://research.nvidia.com/labs/sil/lyra2/>，代码与系列资源仓库为 <https://github.com/nv-tlabs/lyra>，模型资源页为 <https://huggingface.co/nvidia/Lyra-2.0>。复现关键依赖包括 Wan2.1-14B、FramePack、ViPE、Depth Anything v3、Qwen3-VL-8B-Instruct、OpenVDB/FVDB 与 Isaac Sim。官方 PDF 和源码包已保存为 `sources/paper.pdf`、`sources/arxiv-source.tar`，逐项来源见 `sources.md`。

复现顺序建议：先实现 Wan+depth warp+Plücker 的 80 帧 I2V；再实现 per-frame cache 和 visibility greedy retrieval；加入 canonical coordinate 的 Q/K 注入并做回望测试；加入 FramePack；最后加入$p_{aug}=0.7$的一步 self-augmentation。3D 输出可先直接使用预训练 DAv3，再做生成域微调，避免同时调试生成与重建两套系统。

**图表索引**：

1. `fig1_teaser.png`：可探索大世界与仿真应用，Figure 1 / Page 1。
2. `fig2_method.png`：完整循环与空间记忆注入，Figure 2 / Page 6。
3. `fig5_video_results.png`：约第 800 帧定性比较，Figure 3 / Page 10。
4. `fig7_ablation.png`：记忆、对应、FramePack、自增强消融，Figure 6 / Page 14。

**仍缺信息**：官方论文未给出 DL3DV 训练片段的精确划分清单、所有用户轨迹、WorldScore 评测 prompt/人工评分原始记录、DMD 全部超参数、最终 checkpoint 哈希和 mesh 质量的系统定量结果；此外，CaM/SPMem 因未开源而由作者复现，公平性依赖复现质量。这些缺口应在严格复现报告中单独标记。
