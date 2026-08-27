---
type: paper-note
status: draft
domain: World-Models
paper: Large Video Planner
year: 2025
arxiv: 2512.15840
doi: null
source: https://arxiv.org/abs/2512.15840
project: null
code: null
tags:
  - World-Models
  - Video-Planning
  - Foundation-Model
  - Open-Source
  - Video-Diffusion
created: 2026-08-27
updated: 2026-08-27
---

# Large Video Planner：大规模开源视频规划模型

## 基本信息

- **作者**：Boyuan Chen 等
- **机构**：MIT / UC Berkeley / Harvard 等
- **发表**：arXiv 2512.15840（2025-12）
- **原文 PDF**：<https://arxiv.org/pdf/2512.15840>

## 一句话结论

Large Video Planner 是一个 foundation-model 规模的**开源**视频规划模型：给定当前观测与任务目标，直接生成「未来视频轨迹」作为规划结果，动作再从预测的视频中解出。它把视频生成从「内容创作」搬到「机器人规划」，并强调开源与规模，是石麻笔记 EP06 中「视频规划」路线在开源侧的规模化尝试。

---

## 1. 研究背景与问题定义

### 1.1 问题定位

视频规划（video planning）的基本思想是：与其在动作空间里搜索，不如先在视频空间里「想象」一条可行轨迹，再反推动作。这条路线（UniPi、SuSIE、mimic-video 等）长期受限于两点：一是视频基础模型不够强，二是缺乏一个开源、规模化、能复现的规划模型。Large Video Planner 试图同时解决这两点。

### 1.2 与 VPP / mimic-video 的关系

- **VPP**：用视频模型的内部表征做策略的视觉编码器，不做完整规划；
- **mimic-video**：基于 Cosmos 的 VAM，把视频 latent plan 当策略动态先验；
- **Large Video Planner**：站在「开源 + 规模」立场，把视频规划做成可复现的 foundation model，强调从大规模视频数据里学到的物理先验能直接服务规划。

### 1.3 核心贡献

1. 训练一个 foundation-model 规模的开源视频规划模型；
2. 验证「视频轨迹生成 → 动作解出」的规划范式在大规模下的有效性；
3. 强调可复现与开源，为社区提供视频规划的基础设施。

---

## 2. 核心方法

### 2.1 视频即规划

规划过程被建模为条件视频生成：给定当前观测 $\mathbf{o}$ 与目标 $\mathbf{g}$，生成满足目标的未来视频 $\hat{\mathbf{v}}$，再通过逆动力学或闭环控制器从 $\hat{\mathbf{v}}$ 解出动作。

$$\hat{\mathbf{v}} = \text{VideoModel}(\mathbf{o}, \mathbf{g}), \quad \mathbf{a} = \text{InvDyn}(\mathbf{o}, \hat{\mathbf{v}})$$

### 2.2 规模化的关键

论文强调三点支撑规模化：大规模视频预训练数据带来的物理先验、foundation-model 级别的模型容量、以及开源权重让下游能低成本微调到具体机器人。这与 Cosmos、SVD 的「视频基础模型」路线一致，区别在于更聚焦「规划」这一下游任务。

---

## 3. 实验与结果

- 在多项规划/操控基准上验证「视频生成 → 动作解出」的可行性；
- 强调开源模型在跨任务、跨场景泛化上的表现，尤其对比仅用动作监督的 VLA 基线。

---

## 4. 局限与评价

1. **推理成本**：生成完整视频轨迹比直接回归动作更重，实时闭环需要工程优化（或采用少步/一致性模型加速）；
2. **动作解出依赖逆动力学**：视频生成误差会传导到动作，长程累积漂移是固有风险；
3. **「规模」与「可复现」的张力**：foundation-model 规模意味着训练成本高，社区复现门槛仍不低；
4. **与 VPP 路线的互补性**：VPP 证明「不需要完整视频也能规划」，Large Video Planner 坚持完整视频规划，两条路线在数据量与推理延迟上的取舍尚未收敛。

---

## 5. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 获取日期 | 用途 |
|---|---|---|---|---|
| — | 论文 PDF（未下载） | <https://arxiv.org/pdf/2512.15840> | 2026-08-27 | 精读主来源 |
| — | 摘要页 | <https://arxiv.org/abs/2512.15840> | 2026-08-27 | 官方信息核对 |

> 注：本文为 VPP 专访文章中的「次要提及工作」，本篇笔记依据 arXiv 摘要与专访口径整理；官方 PDF 与关键图尚未单独下载，后续如需完整精读可按上方链接补齐 figures/ 与 sources/。
