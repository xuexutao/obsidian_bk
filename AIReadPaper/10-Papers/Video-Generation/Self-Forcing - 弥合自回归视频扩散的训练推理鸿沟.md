---
type: paper-note
status: done
domain: Video-Generation
paper: "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"
year: 2025
venue: NeurIPS 2025 Spotlight
arxiv: "2506.08009"
source: https://arxiv.org/abs/2506.08009
project: https://self-forcing.github.io/
code: https://github.com/guandeh17/Self-Forcing
tags:
  - Video-Generation
  - Autoregressive-Diffusion
  - Distribution-Matching
  - Exposure-Bias
created: 2026-08-29
updated: 2026-10-15
---

# Self-Forcing：弥合自回归视频扩散的训练-推理鸿沟

## 基本信息

- **作者**：Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman
- **机构**：Adobe Research；The University of Texas at Austin
- **任务**：将预训练视频扩散模型后训练为低延迟、可流式、自回归生成的视频模型
- **基础模型**：Wan2.1-T2V-1.3B；4-step flow matching；帧级或 3 latent-frame chunk 级自回归
- **官方披露核验（2026-10-15）**：论文、项目页、官方代码、训练/推理脚本与模型权重均已公开；论文未给出从 Wan2.1 原始预训练开始的总成本，也未披露 70K GAN 视频的生成成本。

## 一句话结论

Self-Forcing 的关键不是再设计一个去噪器，而是让学生在训练时真正按推理方式吃自己的输出，并在整段 rollout 上做 DMD/SiD/GAN 分布匹配；这一“顺序后训练”显著抑制暴露偏差，在单张 H100 上实现 17 FPS、0.69 秒首 chunk 延迟，同时达到 VBench 84.31。

## 1. 问题、动机与贡献

Teacher Forcing 用干净真值作历史，Diffusion Forcing 用加噪真值作历史；两者在推理时却必须依赖模型自己生成且带误差的历史，因此训练条件分布与推理分布不一致。CausVid 虽使用 DMD，但其训练样本仍来自 Diffusion Forcing 路径，匹配的并非真实推理分布。

![](assets/Self-Forcing%20-%20弥合自回归视频扩散的训练推理鸿沟/figures/01-training-paradigms.png)

> **图解**：前两种范式的历史条件来自数据分布；Self-Forcing 的历史条件来自学生自身，最终完整视频再接受分布匹配监督。来源：论文 **Figure 1, PDF p.2**，https://arxiv.org/pdf/2506.08009（获取：2026-10-15）。

核心贡献：
1. 把自回归视频扩散的训练改为 inference-like self-rollout，直接处理 exposure bias。
2. 以随机梯度截断使“时间自回归 × 空间多步去噪”的展开可训练。
3. 将 DMD、SiD、GAN 统一为完整视频级分布匹配目标。
4. 引入 rolling KV cache，使滑窗外推由需要重算历史变为固定窗口缓存。

## 2. 输入、输出与任务边界

- **输入**：文本提示词 $c$；模型也可在已有首帧/视频条件下做 I2V 或外推。
- **输出**：832×480、16 FPS 视频流；基础评测为 5 秒视频，自回归机制允许继续生成。
- **生成单位**：frame-wise 变体每次一帧；chunk-wise 变体每次 3 个 latent frames。
- **任务边界**：论文解决视觉生成与流式延迟，不提供显式 3D 状态、物理引擎或可验证的因果动力学。

## 3. 架构与 Self-Rollout

视频联合分布按链式法则分解：

$$p_\theta(x_{1:N})=\prod_{i=1}^{N}p_\theta(x_i\mid x_{<i}).$$

每个条件分布不是一次 token 分类，而是由 $T=4$ 步扩散近似。训练时，第 $i$ 个 frame/chunk 从高斯噪声开始，在已生成历史 $\hat{x}_{<i}$ 的 KV cache 条件下去噪；因此样本确实来自推理时的 $p_\theta$。

![](assets/Self-Forcing%20-%20弥合自回归视频扩散的训练推理鸿沟/figures/02-attention-masks.png)

> **图解**：TF/DF 依靠特殊 mask 并行训练整段视频；Self-Forcing 像真实推理一样沿时间 rollout，只缓存已经生成的 clean KV。来源：论文 **Figure 2, PDF p.4**，https://arxiv.org/pdf/2506.08009（获取：2026-10-15）。

### 3.1 随机梯度截断

完整反传会跨越 $N\times T$ 个模型调用。作者每次随机采样 $s\in\{1,\dots,T\}$：前面的去噪步骤关闭梯度，只对每个 frame 的第 $s$ 步输出保留梯度；历史 frame 对当前 frame 的 KV 也 detach。这样不同迭代覆盖不同中间步，但代价是无法直接沿长历史分配信用。

Wan2.1 的 shift flow schedule 为

$$t'(k,t)=\frac{kt/1000}{1+(k-1)t/1000}\cdot1000,\quad k=5,$$

4 步时间点为 $[1000,750,500,250]$。

## 4. 视频级分布匹配与蒸馏

对完整 self-rollout 视频 $\hat{x}\sim p_\theta$ 加噪后，DMD 用真实分数与伪分数之差近似 reverse-KL 梯度：

$$\nabla_\theta\mathbb E_t D_{KL}(p_{\theta,t}\|p_{data,t})
=-\mathbb E[(s_{real}(\hat{x}_t,t)-s_{fake}(\hat{x}_t,t))\,\partial\hat{x}/\partial\theta].$$

- **DMD**：real score 为 Wan2.1-T2V-14B，fake critic 从 1.3B 初始化；生成器 lr $2\times10^{-6}$，critic lr $4\times10^{-7}$。
- **SiD**：以 Fisher divergence 的 score identity 形式训练，论文采用稳定性更好的 $\alpha=1$。
- **GAN**：R3GAN relativistic loss + 近似 R1/R2 正则，$\lambda=30,\sigma=0.05$，批量 768。

DMD/SiD 可不读取真实视频，只依赖教师分数；GAN 与 many-step TF/DF baseline 使用 Wan2.1-14B 生成的 70K 视频。

## 5. Rolling KV Cache 与长程机制

![](assets/Self-Forcing%20-%20弥合自回归视频扩散的训练推理鸿沟/figures/03-rolling-kv-cache.png)

> **图解**：双向滑窗需 $O(TL^2)$；旧式因果滑窗仍重算重叠 KV，约 $O(L^2+TL)$；rolling KV 逐出最旧条目并复用其余缓存，降为 $O(TL)$。来源：论文 **Figure 3, PDF p.7**，https://arxiv.org/pdf/2506.08009（获取：2026-10-15）。

因 causal 3D VAE 的首 latent frame 统计特殊，直接逐出首块会产生闪烁。作者在训练时限制最后 chunk 不看首 chunk，模拟滑窗越界后的条件分布。该机制保持固定计算量，但它只是有限最近历史，不是可检索的场景级长期记忆。

## 6. 数据、训练与复现设置

- **提示词**：VidProM 的 VidProS 约 1M 去重提示；过滤短于 20 字符、命令行参数和 NSFW 后约 250K，再由 Qwen2.5-7B-Instruct 扩写。
- **ODE 初始化**：沿用 CausVid，使用 16K 个由基础模型采样的 ODE solution pairs。
- **训练硬件**：多数实验为 64×H100 80GB，per-GPU batch 1；DMD 约 1.5 小时收敛，SiD/GAN 约 2–3 小时。
- **优化**：生成器均以 lr $2\times10^{-6}$ 训练；EMA 0.99；完整细节见论文 Table 3。
- **评测**：VBench 16 维、MovieGenBench 1003 prompts 单人单票 A/B 偏好、吞吐与首帧延迟；速度在单张 H100 上测试。

## 7. 主要结果、定性比较与消融

| 模型 | 参数 | 吞吐 FPS ↑ | 延迟 s ↓ | VBench Total ↑ |
|---|---:|---:|---:|---:|
| Wan2.1 | 1.3B | 0.78 | 103 | 84.26 |
| CausVid | 1.3B | 17.0 | 0.69 | 81.20 |
| **Self-Forcing chunk-wise** | **1.3B** | **17.0** | **0.69** | **84.31** |
| **Self-Forcing frame-wise** | **1.3B** | **8.9** | **0.45** | **84.26** |

用户偏好中，本方法相对 MAGI-1、SkyReels-V2、Wan2.1、CausVid 分别为 54.2%、57.9%、62.7%、66.1%。

![](assets/Self-Forcing%20-%20弥合自回归视频扩散的训练推理鸿沟/figures/04-qualitative-comparison.png)

> **图解**：CausVid 随时间出现过饱和和锐化；Self-Forcing 在 0/2.5/5 秒的外观更稳定。该图只能说明代表性案例，不能代替盲评。来源：论文 **Figure 5, PDF p.8**，https://arxiv.org/pdf/2506.08009（获取：2026-10-15）。

Table 2 的受控消融：chunk-wise Self-Forcing-DMD 为 84.31，SiD 84.07，GAN 83.88；DF+DMD 82.76、TF+DMD 82.32。frame-wise 时差距更大：Self-Forcing-DMD 84.26，而 DF+DMD 80.56、TF+DMD 78.12。这说明关键变量是 rollout 分布对齐，而非某一种特定 divergence。

![](assets/Self-Forcing%20-%20弥合自回归视频扩散的训练推理鸿沟/figures/05-training-efficiency.png)

> **图解**：Self-Forcing 单步迭代并未显著慢于 TF/DF，且同等 wall-clock 下 VBench 上升更快；原因是 chunk 内并行且可使用 FlashAttention-3。来源：论文 **Figure 6, PDF p.9**，https://arxiv.org/pdf/2506.08009（获取：2026-10-15）。

## 8. 失败案例、局限与 Future Work

1. 论文明确承认：生成长度显著超过训练上下文后仍会退化，rolling KV 不能消除长期语义漂移。
2. 梯度截断与 KV detach 牺牲长程 credit assignment；模型主要学会局部纠错，而非真正记住远处事件。
3. 首 latent 的分布特殊，naive rolling cache 会闪烁，说明系统仍依赖 VAE/训练窗口的工程假设。
4. 评测集中在 5 秒视频和通用 VBench；实时交互控制、物理一致性、分钟级剧情未得到系统量化。
5. 未来方向：更强外推、循环/状态空间架构、可检索长期记忆，以及不依赖双向教师的稳定后训练。

## 9. 复现成本与个人判断

- **最低推理门槛**：官方仓库称测试过至少 24GB 显存的 RTX 4090/A100/H100；论文中的 17 FPS 数字来自 H100，不能直接外推到 4090。
- **论文级后训练**：仅 DMD 主阶段约 96 H100-hours；SiD/GAN 约 128–192 H100-hours，尚不含 ODE 初始化、教师/基础权重准备和 GAN 70K 视频生成。
- **个人判断**：这是训练范式上的关键工作。它把“训练时见到自己犯的错”从 RNN 时代的口号转化为可执行的视频扩散后训练。但其成功依赖 few-step 模型、强教师与截断梯度，尚不能等同于长期世界状态建模。

## 10. 材料来源与披露状态

- 本地 PDF、5 张官方源码图及逐项来源记录见：`../../assets/Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟/`。
- 详细清单：`../../assets/Self-Forcing - 弥合自回归视频扩散的训练推理鸿沟/sources/sources.md`。
- **已披露**：算法、核心公式、完整超参数、GPU 数量、训练时长、代码和权重。
- **未披露/不足**：Wan2.1 基础预训练成本、70K 视频生成成本、端到端能耗、分钟级稳定性定量结果。
