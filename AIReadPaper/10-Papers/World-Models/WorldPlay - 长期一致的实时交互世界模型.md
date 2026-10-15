---
type: paper-note
status: done
domain: World-Models
paper: "WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling"
year: 2026
venue: ICML 2026
arxiv: "2512.14614"
source: https://arxiv.org/abs/2512.14614
project: https://3d-models.hunyuan.tencent.com/world/
code: https://github.com/Tencent-Hunyuan/HY-WorldPlay
tags:
  - World-Models
  - Memory
  - Interactive-Video
  - Geometry-Consistency
created: 2026-10-15
updated: 2026-10-15
---

# WorldPlay：长期一致的实时交互世界模型

## 基本信息

- **作者**：Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, Chunchao Guo
- **机构**：HKUST；Beihang University；Tencent Hunyuan
- **任务**：从文本或单图初始化世界，按用户动作实时生成下一视频块，并在回访时保持几何一致
- **输出规格**：每块 16 帧；4 步推理；720p、24 FPS（8×H800，含系统优化）
- **官方披露核验（2026-10-15）**：论文、项目页、在线演示、官方 HY-WorldPlay 代码、训练/推理框架和权重已公开。论文未给出全部训练 GPU 数/总 wall-clock、精确 backbone 规模与在线 24 FPS 的端到端网络延迟分解。

## 一句话结论

WorldPlay 用“短期时间记忆 + 按视场/距离检索的长期空间记忆”，再把远帧的 RoPE 时间索引重写为近邻，并在蒸馏时严格对齐师生记忆上下文；这使 4-step 学生兼顾回访一致性与实时性，但 24 FPS 依赖 8×H800 和大量部署优化，论文验证时长仍约 30 秒。

## 1. 问题、动机与贡献

实时世界模型通常在速度与记忆之间二选一：少步自回归快，却在回访时改变场景；显式 3D 或复杂检索记忆能维持一致性，却难以蒸馏和实时运行。WorldPlay 把问题定义为带动作的 next-chunk prediction，并联合解决控制、记忆和 few-step 蒸馏。

![](assets/WorldPlay%20-%20长期一致的实时交互世界模型/figures/01-teaser.png)

> **图解**：红框显示绕行后回访的结构保持；模型覆盖真实/风格化、第一/第三人称，还可把视频交给外部重建器或触发文本事件。来源：论文 **Figure 1, PDF p.1**，https://arxiv.org/pdf/2512.14614（获取：2026-10-15）。

贡献：Dual Action Representation、Reconstituted Context Memory、Temporal Reframing、Context Forcing，以及面向 720p/24 FPS 的并行、量化和流式解码系统。

## 2. 输入、输出与任务定义

模型为

$$N_\theta(x_t\mid O_{t-1},A_{t-1},a_t,c),$$

其中 $O_{t-1}$ 是历史 chunks，$A_{t-1}$ 和 $a_t$ 是历史/当前动作，$c$ 是图像或文本世界描述。3D VAE 每 4 个 latent frames 解码为 16 个视频帧。

- **离散动作**：键盘/鼠标，提供尺度自适应、易学习的控制。
- **连续动作**：相机内外参，提供精确空间位置和检索坐标。
- **输出**：自回归视频流；显式 point cloud 是后接 feed-forward 重建器的应用，不是主模型直接状态。

## 3. 主干与双动作表示

Flow Matching 训练目标：

$$z_k=(1-k)z_1+kz_0,\quad v_k=z_0-z_1,$$
$$L_{FM}=\mathbb E\|N_\theta(z_k,k)-v_k\|_2^2.$$

离散键由 PE + zero-init MLP 编码并加入 timestep modulation；连续相机由 PRoPE 注入 causal self-attention。普通 3D RoPE attention 与相机 frustum attention 相加：

$$\mathrm{Attn}=\mathrm{Attn}(R^\top\odot Q,R^{-1}\odot K,V)+\mathrm{zeroInit}(\mathrm{Attn}_{PRoPE}).$$

离散控制保证跨场景训练稳定，连续 pose 则为精确移动量和空间检索服务。

## 4. Reconstituted Context Memory

![](assets/WorldPlay%20-%20长期一致的实时交互世界模型/figures/02-method-overview.png)

> **图解**：每个新 chunk 前，从全局 memory cache 重构空间记忆 $C_t^S$ 与最近时间记忆 $C_t^T$，再与 dual-action 一起送入自回归扩散模型。来源：论文 **Figure 2, PDF p.4**，https://arxiv.org/pdf/2512.14614（获取：2026-10-15）。

记忆由两部分构成：

- $C_t^T$：最近 $L=3$ 个 chunks，维持局部运动和时间连续性。
- $C_t^S$：从非相邻历史按 FOV overlap 与相机距离检索 1 个 chunk，负责回访几何。
- 首 chunk 作为 attention sink 保留，以缓解长序列漂移。

### 4.1 Temporal Reframing

标准 RoPE 的绝对时间索引让远历史与当前块距离不断增大，超出训练插值范围并弱化注意力。Temporal Reframing 丢弃原始绝对索引，为检索帧重新分配固定、接近当前块的相对索引，相当于把“空间上重要但时间久远”的帧拉近。

这是一种隐式图像记忆，不依赖在线深度重建；优势是鲁棒，风险是检索仍依赖 pose/FOV，遮挡和错误姿态会取错记忆。

## 5. Context Forcing：记忆感知蒸馏

![](assets/WorldPlay%20-%20长期一致的实时交互世界模型/figures/03-context-forcing.png)

> **图解**：学生做 memory-augmented self-rollout；双向 real/fake score 网络看到与学生对齐、且移除当前预测块的同一记忆上下文，避免“有记忆学生 vs 无记忆教师”的条件分布错位。来源：论文 **Figure 5, PDF p.5**，https://arxiv.org/pdf/2512.14614（获取：2026-10-15）。

学生 rollout 4 个 chunks：

$$p_\theta(x_{j:j+3}\mid x_{0:j-1})=\prod_{i=j}^{j+3}p_\theta(x_i\mid C_i).$$

教师条件设为

$$p_\beta(x_{j:j+3}\mid C_{j:j+3}-x_{j:j+3}),$$

即复用对应学生记忆，但 mask 掉当前预测序列。再以 DMD reverse-KL score difference 蒸馏。训练逐步增加 self-rollout 最大长度，最终保留长期记忆并把 NFE 从 100 降至 4。

## 6. 数据、训练与系统加速

### 6.1 数据（320K clips）

| 类别 | 来源 | 数量 | 比例 |
|---|---|---:|---:|
| Real-World Dynamics | Sekai | 40K | 12.5% |
| Real-World 3D Scene | DL3DV 重建/重渲染 | 60K | 18.75% |
| Synthetic 3D Scene | UE rendering | 50K | 15.625% |
| Simulation Dynamics | 游戏录制 | 170K | 53.125% |

长视频切为 30–40 秒；VLM 生成描述；VIPE 补相机 pose；以相邻 pose 运动的 Peak-to-Median Ratio > 5.0 过滤跳变。

### 6.2 三阶段训练

1. **Action control**：双向 action model 30K iter，再换 block-causal attention 训练 AR model 30K；61 帧，Adam，lr $10^{-5}$，batch 64。
2. **Memory**：师生加入 memory；最长 160 latents/637 frames；memory 条件加 $[0,0.2]$ 噪声。
3. **Context Forcing**：student lr $10^{-6}$，fake-score bidirectional model lr $2\times10^{-7}$，2K iter，batch 64。

### 6.3 24 FPS 的系统前提

8×H800；token 内 sequence+attention mixed parallelism；Triton streaming；progressive VAE decoding；SageAttention、浮点与 GEMM 量化；KV cache。加速表从 2.8 FPS，经 DiT P&Q 到 3.8、再加 VAE P&Q 到 16，最终加流式解码到 24 FPS。

## 7. 主要定量、消融与定性结果

长程（≥250 帧）主结果：

| 方法 | 实时 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Rdist ↓ | Tdist ↓ |
|---|---|---:|---:|---:|---:|---:|
| Gen3C | 否 | 15.37 | 0.431 | 0.483 | 0.357 | 0.979 |
| VMem | 否 | 12.77 | 0.335 | 0.542 | 0.748 | 1.547 |
| Ours w/o Context Forcing | 否 | 16.27 | 0.425 | 0.495 | 0.611 | 0.991 |
| **WorldPlay full** | **是** | **18.94** | **0.585** | **0.371** | **0.332** | **0.797** |

MEt3R（越低越好）：WorldPlay 0.133，Gen3C 0.187，GameCraft 0.305，Matrix-Game-2.0 0.367。

![](assets/WorldPlay%20-%20长期一致的实时交互世界模型/figures/04-qualitative-comparison.png)

> **图解**：在返回路径与第三人称场景中，WorldPlay 比无记忆模型更能恢复红框目标；但该 montage 仍是精选轨迹。来源：论文 **Figure 6, PDF p.7**，https://arxiv.org/pdf/2512.14614（获取：2026-10-15）。

### 7.1 消融

Dual action 的完整配置达到 PSNR 22.09、SSIM 0.687、LPIPS 0.219、Rdist 0.028、Tdist 0.113，优于仅离散或仅连续动作。

![](assets/WorldPlay%20-%20长期一致的实时交互世界模型/figures/05-reframed-rope-ablation.png)

> **图解**：普通 RoPE 随轨迹增长出现云雾/结构漂移；reframed RoPE 保持远处记忆的影响。对应长程量化从 PSNR 14.03/SSIM 0.358 提升到 16.27/0.425。来源：论文 **Figure 7, PDF p.8**，https://arxiv.org/pdf/2512.14614（获取：2026-10-15）。

![](assets/WorldPlay%20-%20长期一致的实时交互世界模型/figures/06-context-forcing-ablation.png)

> **图解**：师生上下文错位直接崩溃；无记忆教师导致返回不一致；用学生 rollout 历史给教师会产生伪影；对齐且来自真实视频的历史上下文最稳定。来源：论文 **Figure 8, PDF p.9**，https://arxiv.org/pdf/2512.14614（获取：2026-10-15）。

Context Forcing 模型对比：AR student 100 NFE 为 16.27/0.425/0.495，bidirectional teacher 100 NFE 为 19.31/0.599/0.383，最终 4 NFE student 为 18.94/0.585/0.371，说明蒸馏保留了大部分教师质量。

## 8. 失败案例、局限与 Future Work

1. 作者明确说明稳定生成约 30 秒；分钟/小时级扩展仍未解决。
2. Context Forcing 缓解但没有从训练目标上根除自回归误差累积。
3. 基于 FOV 的检索遇到大遮挡会选不到正确历史；pose 估计错误也会污染索引。
4. 主要动作仍是导航和文本事件；复杂物理、多主体交互与可编辑对象状态不足。
5. 24 FPS 使用 8×H800，不能与单卡“实时”直接比较；端到端交互还包括网络、调度和编码延迟。
6. 数据超过一半来自游戏录制，真实复杂动力学与授权/可再分发性未在论文充分展开。

## 9. 复现成本与个人判断

- **训练**：至少需要两套大视频扩散网络、长达 637 帧的 memory training 和 320K 视频数据；论文没有披露 GPU 数与 wall-clock，无法可信估算总 GPU-hour。
- **论文级实时推理**：8×H800 加定制并行/量化/Triton；普通实验室复现“算法正确性”可用开源小模型，但难复现 720p/24 FPS 服务指标。
- **数据工程**：DL3DV 重建/轨迹重渲染、UE 场景、游戏录制、VIPE pose 和质量过滤本身就是主要成本。
- **个人判断**：最有价值的创新是把“记忆条件”纳入蒸馏分布定义。Temporal Reframing 简单但有效；然而系统仍把场景状态存为历史图像，尚未形成可组合、可编辑、物理可检验的世界状态。

## 10. 材料来源与披露状态

- 本地 PDF、6 张官方源码图与来源清单：`../../assets/WorldPlay - 长期一致的实时交互世界模型/`。
- 详细清单：`../../assets/WorldPlay - 长期一致的实时交互世界模型/sources/sources.md`。
- **已披露**：记忆大小、三阶段训练迭代/lr/batch、数据构成、4-step 蒸馏、系统加速分项、代码和权重。
- **未披露**：全训练 GPU 数/时长、精确 backbone 参数规模、24 FPS 的端到端延迟分解、分钟级评测与数据完整许可说明。
