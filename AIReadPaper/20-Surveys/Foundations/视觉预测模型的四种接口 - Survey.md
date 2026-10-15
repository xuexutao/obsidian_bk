---
type: survey
status: complete
domain: Foundations
title: "视觉预测模型的四种接口：从离散自回归、连续流到几何与潜动力学"
created: 2026-10-15
updated: 2026-10-15
tags:
  - Visual-Generation
  - Autoregressive
  - Flow-Matching
  - 4D-Reconstruction
  - World-Models
---

# 视觉预测模型的四种接口：从离散自回归、连续流到几何与潜动力学

## 基本信息

本文对照六篇论文：

- [[LlamaGen - 可扩展自回归图像生成]]
- [[VAR - 下一尺度自回归图像生成]]
- [[Infinity - 位级视觉自回归高分辨率图像生成]]
- [[Goku - 联合图像视频整流流基础模型]]
- [[Dyna3 - VLM引导的免训练动态四维重建]]
- [[Flow-JEPA - 流匹配的鲁棒JEPA潜动力学]]

它们不是同一条窄技术线：前三篇是离散视觉 AR，Goku 是连续 latent flow 生成，Dyna3 是动态几何重建，Flow-JEPA 是用于规划的潜动力学。把它们放在一起真正有价值的问题是：**模型究竟在预测什么对象，预测顺序如何组织，错误如何传播，输出最终服务于“看起来像”还是“可用于决策”。**

---

## 1. 总体地图：四种预测接口

| 路线 | 代表工作 | 预测对象 | 自回归/积分轴 | 输出用途 | 主要瓶颈 |
|---|---|---|---|---|---|
| 离散视觉 AR | LlamaGen | 单个 VQ token | raster scan | 图像生成 | 串行长度、tokenizer 上限 |
| 多尺度离散 AR | VAR、Infinity | 尺度 token map / bit map | coarse-to-fine scale | 高效图像生成 | 高分辨率注意力、误差逐尺度传播 |
| 连续视觉流 | Goku | 图像/视频 VAE latent 的速度场 | ODE time | 图像+视频生成 | 长序列 full attention、数据/算力 |
| 几何/潜状态预测 | Dyna3、Flow-JEPA | 4D 点云或未来 latent trajectory | 视频时间/flow time | 重建、规划 | 对应与鲁棒性、物理可用性 |

这张表揭示一个容易混淆的事实：自回归、flow matching、JEPA 不是简单替代关系。它们通常作用在不同的表示对象上：离散 token、连续生成 latent、几何点、抽象状态。真正决定系统性质的是“表示×预测轴×下游目标”的组合。

---

## 2. 第一次转移：先证明普通视觉 AR 还能扩展

### 2.1 LlamaGen 的问题意识

LlamaGen 保留最传统的概率分解：逐 raster token 预测下一个 token。它的贡献是消除两个历史借口：离散 tokenizer 不一定比连续 VAE 差；Llama 架构只要规模够大也能达到 ImageNet 强结果。3.1B 模型 FID 2.18 略超 DiT-XL/2 2.27，说明视觉 AR 不是天然落后。

但它也暴露了原生 raster AR 的极限：576 token 只有大模型才受益；1024²会带来更长串行链；文生图依然依赖外置 FLAN-T5，且缺标准定量结果。

### 2.2 为什么 VAR 必然出现

当模型容量不再是唯一瓶颈，下一步自然会质疑“图像为什么必须按左上到右下生成”。VAR 把因果单位从位置改成尺度：尺度间自回归、尺度内并行。其消融显示仅 AR→VAR 就把 FID 18.65 降到 5.22、相对推理成本降到 0.013，证明主要收益来自概率分解方式，而不是更大模型或 CFG。

这一步的本质是：**视觉 AR 的设计自由度不在是否 autoregressive，而在如何选择条件分解轴。**

---

## 3. 第二次转移：VAR 之后，瓶颈转向表示与误差纠正

### 3.1 Infinity 为什么不是“更大的 VAR”

VAR 解决了串行尺度，却仍使用有限索引词表和 teacher forcing。Infinity 的三件套分别补三个洞：

- BSQ/LFQ：用$d$位符号隐式表示$2^d$组合，扩大离散容量；
- IVC：把$2^d$类分类改成$d$个 bit 二分类，避免输出头爆炸；
- BSC：污染历史 bit 并重算后续残差标签，让模型学会纠错。

BSC 的关键消融很干净：普通 baseline FID 9.76；只 random flip 为 9.69；重算标签的 BSC 直接到 3.48。说明“看过噪声”不够，必须让训练目标告诉模型如何从错误历史中补偿。

### 3.2 三篇离散 AR 的技术谱系

| 论文 | 主要问题 | 关键答案 | 仍未解决 |
|---|---|---|---|
| LlamaGen | 普通 Llama 能否生成图像 | 强 tokenizer+规模化 | raster 串行链 |
| VAR | 图像 AR 应沿什么轴分解 | next-scale prediction | 有限索引词表、teacher forcing |
| Infinity | 如何扩大词表并修复早期错误 | bitwise+IVC+BSC | 数据透明度、任意分辨率、视频扩展 |

这是一条非常清晰的“瓶颈迁移”：模型能力→因果顺序→表示与纠错。

---

## 4. 第三次分叉：离散 token 与连续 flow 各自擅长什么

### 4.1 Goku 选择连续 latent flow

Goku 没有沿 bitwise AR 前进，而是用 3D VAE latent+Rectified Flow 联合建模图像和视频。原因在于视频 token 极长、运动连续、离散逐尺度方案在 2025 年尚缺成熟视频扩展。Flow matching 可以对完整时空 latent 做连续输运，再通过 full attention 统一空间和时间。

Goku 的 RF 400K steps FID 2.1572，优于 DDPM 1000K 的 2.2568；更重要的是，它把 160M 图像和 36M 视频混入同一模型，形成“图像教细节、视频教运动”的数据互补。

### 4.2 两条路线的结构差别

- 离散 AR 的训练目标是条件分类，优势是 likelihood 风格目标、KV cache 和 LLM 生态；难点是 tokenizer、离散误差和串行因果链。
- 连续 flow 的训练目标是速度回归，优势是连续表示和联合生成；难点是多次向量场求值、长序列注意力和数值求解。

二者并未决出永久胜负。Infinity 证明离散高分辨率生成可很快；Goku 证明连续 flow 更容易统一图像与视频。未来很可能继续混合：尺度级 AR 负责全局结构，尺度内 flow/diffusion 负责连续细节。

---

## 5. 第四次转移：从“生成像素”到“恢复世界与支持规划”

### 5.1 Dyna3：几何输出要求可对齐，而不只是可生成

Dyna3 不生成新视觉内容，而是从视频恢复静态点云、动态点云和相机。它遇到的核心问题是 correspondence：DA3 attention 没有 tracking supervision，不能直接当运动信号；于是改用 feature best-match，再借 VLM/SAM 3 获得实例语义和边界。

这说明生成模型常见的“看起来合理”对重建不够。几何任务需要跨帧可对齐、相机一致和动态边界正确。Dyna3 用效率换取逐帧动态点云，DyCheck 质量仍落后 VGGT4D，正体现“外观合理”和“时间对应正确”的差别。

### 5.2 Flow-JEPA：规划不需要像素，但需要动力学鲁棒

Flow-JEPA 更进一步：输出甚至不可见，只预测可规划的未来 latent trajectory。它把 flow matching 从图像/视频生成工具改造成动力学模型。背景高斯噪声不改变物理状态，却让 LeWM 平均成功率从 86%掉到 67%；轨迹级 flow 将 noisy 成功率提高到 86%。

这里的目标不再是 FID，而是 action-conditioned rollout 是否支持 CEM 找到正确动作。相同数学工具在 Goku 中追求视觉质量，在 Flow-JEPA 中追求规划稳定性。

---

## 6. 主要结果横向对比

| 工作 | 代表结果 | 最有说服力的消融 | 证据边界 |
|---|---|---|---|
| LlamaGen | ImageNet FID 2.18 | tokenizer dim 256→8：usage 0.29%→97% | 3.1B、类别条件、T2I无定量 |
| VAR | 常规 FID 1.92，10 steps | AR→VAR：FID 18.65→5.22 | 主要为 ImageNet 256/512 |
| Infinity | GenEval 0.73†，0.8s/1024² | BSC：FID 9.76→3.48 | prompt rewrite、速度配置不透明 |
| Goku | VBench 84.85 | RF 400K 优于 DDPM 1000K | 内部数据、主消融偏定性 |
| Dyna3 | DAVIS-16 J 65.04 | DA3 8.16→motion 29.65→VLM 53.35→SAM 65.04 | 动态点无跨帧对应 |
| Flow-JEPA | noisy success 67%→86% | NFE/CEM/噪声强度压力测试 | 未拆 joint trajectory 与 flow |

†使用 prompt rewriting。

---

## 7. 共通问题：错误在哪里累积

六篇论文可以用“误差传播位置”统一理解：

1. **LlamaGen**：逐 token 错误沿 raster 序列传播。
2. **VAR**：串行链缩成尺度数，但粗尺度错误会污染细尺度。
3. **Infinity**：BSC 专门学习如何根据错误历史修复后续残差。
4. **Goku**：连续流通过多步积分积累向量场误差；视频还叠加长序列时间漂移。
5. **Dyna3**：运动先验、VLM、SAM、深度和位姿形成级联系统误差。
6. **Flow-JEPA**：一步 latent 递归误差被轨迹级联合预测替代，但 ODE 求解和模型偏差仍存在。

因此 2024–2026 的重要趋势不是单纯“更大模型”，而是**让训练时看到推理时的错误形态**：Infinity 的 BSC、Flow-JEPA 的随机路径、视频模型中的 self-forcing 都属于这一大方向。

---

## 8. 当前攻坚问题

### 8.1 表示是否真正可扩展

离散路线要证明极大 bit vocabulary 在视频和统一理解—生成中仍可学；连续路线要降低 full attention 与多次 ODE 求解成本；几何/JEPA 路线要证明 latent 保留了下游需要的对应和动力学。

### 8.2 长时域与一致性

Goku 主要 4 秒视频，Dyna3 动态点逐帧独立，Flow-JEPA horizon=5。三条路线都还没有解决分钟级、长规划链和多次重访下的一致性。

### 8.3 评测口径

图像生成依赖 FID、GenEval 和自动 reward；视频依赖 VBench；重建依赖几何误差；规划依赖成功率。不同指标无法互相替代。尤其 prompt rewriting、caption enrichment、rejection sampling、内部人评都应单独标注，不能只传播最好数字。

### 8.4 数据和系统透明度

Goku/Infinity 的内部数据与速度配置、Dyna3 的 VLM 2B/7B 冲突、LlamaGen T2I 内部数据，都说明现代视觉基础模型的上限越来越依赖不可见工程。方法创新必须与可复现证据分开评价。

---

## 9. 判断与展望

### 9.1 已较确定的事实

- 视觉 AR 的因果轴不是固定的，next-scale 明显优于传统 raster baseline。
- 离散 tokenizer 不再必然是图像质量瓶颈，但生成模型容量必须与 token 精度共同扩展。
- 训练—推理历史错配是多尺度 AR 和世界模型的共同问题。
- Flow matching 已从图像生成扩展到视频与潜动力学，成为通用条件分布建模工具。
- 几何重建与规划对“可对齐、可预测”的要求高于视觉上合理。

### 9.2 我的推测

未来最有潜力的系统不一定在离散 AR 与连续 flow 中二选一，而可能采用层级混合：语义/布局用少量离散尺度或计划 token，自然图像细节与运动用连续 flow，几何与动作再由专用 latent head 读取。训练上会越来越多地显式模拟自生成历史，而不是只做 teacher forcing。

### 9.3 阅读优先级

若目的是理解视觉生成范式，建议顺序：VAR→Infinity→LlamaGen→Goku；若目的是世界模型和机器人，建议 Flow-JEPA→Dyna3，并把它们分别与 LeWorldModel、DA3/VGGT4D 对照阅读。

---

## 10. 文献入口

- [LlamaGen](https://arxiv.org/abs/2406.06525)
- [VAR](https://arxiv.org/abs/2404.02905)
- [Infinity](https://arxiv.org/abs/2412.04431)
- [Goku](https://arxiv.org/abs/2502.04896)
- [Dyna3](https://arxiv.org/abs/2610.01286)
- [Flow-JEPA](https://arxiv.org/abs/2608.29029v3)
