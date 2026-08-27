---
type: paper-note
status: draft
domain: World-Models
paper: Neural Dynamics for Locomotion
year: 2025
arxiv: 2506.19827
doi: null
source: https://arxiv.org/abs/2506.19827
project: null
code: null
tags:
  - World-Models
  - Locomotion
  - Neural-Dynamics
  - Self-Forcing
  - Uncertainty-Aware
created: 2026-08-27
updated: 2026-08-27
---

# Robotic World Model：面向四足运动的神经动力学模型

## 基本信息

- **作者**：Chenhao Li 等
- **机构**：ETH Zurich（Marco Hutter / Andreas Krause 组）
- **发表**：arXiv 2506.19827（2025-06）
- **原文 PDF**：<https://arxiv.org/pdf/2506.19827>

## 一句话结论

本文构建了一个面向四足机器人 locomotion 的神经动力学世界模型，核心创新是「self-forcing」自回归训练：让模型一边预测未来、一边用自己上一步的输出作为下一步输入来逐步自激励，从而在长程推演中保持稳定；并进一步提出 uncertainty-aware 变体，让模型对不确定区域显式建模。它代表了石麻笔记 EP06 中「Dynamics/Graph/Particle」这一世界模型分支在足式机器人上的落地。

---

## 1. 研究背景与问题定义

### 1.1 问题定位

足式机器人（四足/人形）的 locomotion 控制高度依赖对自身动力学与地面接触的准确建模。传统 MPC 需要显式动力学模型，难以泛化到非结构化地形；而纯模型无关的强化学习又缺少对未来状态的可预测性。本文的目标是学一个**数据驱动的神经动力学世界模型**，既能预测机器人在给定动作下的状态演化，又能用于规划与控制。

### 1.2 与 Yunzhu Li 研究线的关系

ETH 的 Marco Hutter（四足/人形机器人）与 Andreas Krause（不确定性/主动学习）组合，加上本文的神经动力学世界模型，是「Dynamics / Graph / Particle」这条世界模型路线（Yunzhu Li 系列工作）在足式机器人场景的具体应用与扩展。石麻笔记 EP06 把这条路线与 2D 视频、3D 表征、隐式表征（JEPA）、统一模型并列。

### 1.3 核心贡献

1. 提出 self-forcing 自回归训练，缓解世界模型长程推演的误差累积；
2. 引入 uncertainty-aware 变体，对预测不确定性显式建模，服务下游规划的安全决策；
3. 在四足 locomotion 任务上验证世界模型的预测质量与规划收益。

---

## 2. 核心方法

### 2.1 Self-forcing 自回归训练

传统世界模型的训练常采用「teacher forcing」——每步都用真实状态作为输入，推理时却只能用模型自己的预测，导致训练/推理分布不匹配、长程误差累积。self-forcing 的思路是在训练过程中逐步用模型自己的输出替换真实状态作为下一步输入，让模型学会在自己的预测上继续推演：

$$\hat{\mathbf{s}}_{t+1} = f_\theta(\hat{\mathbf{s}}_t, a_t)$$

其中 $\hat{\mathbf{s}}_t$ 在训练后期由模型自身预测给出，而非真实 $\mathbf{s}_t$。这显著提升了长程自回归推演的稳定性。

### 2.2 Uncertainty-aware 变体

针对世界模型预测的不确定性（尤其在地形未知、接触不确定时），论文提出 uncertainty-aware 版本，让模型输出一个状态分布而非点估计，供下游规划器做风险感知的决策。

---

## 3. 实验与结果

- 在四足机器人的 locomotion 数据上验证 self-forcing 训练显著优于 teacher forcing，长程推演漂移更小；
- uncertainty-aware 变体在非结构化地形上展现出更稳健的规划与安全行为。

---

## 4. 局限与评价

1. **领域聚焦**：本文聚焦四足 locomotion，动态相对结构化（本体动力学主导），与桌面操控的「多物体交互 + 语义理解」场景差异较大，结论的可迁移性有限；
2. **self-forcing 的调度**：何时、以多大概率替换真实状态需要精细设计，是影响效果的关键超参；
3. **与 2D/3D 世界模型的分工**：locomotion 场景用状态空间（本体位姿、接触力）建模即可，不需要像素/点云级别的表征；这是「Dynamics 分支」与「视频/3D 分支」的根本差异所在。

---

## 5. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 获取日期 | 用途 |
|---|---|---|---|---|
| — | 论文 PDF（未下载） | <https://arxiv.org/pdf/2506.19827> | 2026-08-27 | 精读主来源 |
| — | 摘要页 | <https://arxiv.org/abs/2506.19827> | 2026-08-27 | 官方信息核对 |

> 注：本文为 VPP 专访文章中的「次要提及工作」，本篇笔记依据 arXiv 摘要与专访口径整理；官方 PDF 与关键图尚未单独下载，后续如需完整精读可按上方链接补齐 figures/ 与 sources/。
