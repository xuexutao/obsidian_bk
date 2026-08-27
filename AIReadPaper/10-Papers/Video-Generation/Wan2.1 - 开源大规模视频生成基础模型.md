---
type: paper-note
status: draft
domain: Video-Generation
paper: Wan: Open and Advanced Large-Scale Video Generative Models
year: 2025
arxiv: 2503.20314
doi: null
source: https://arxiv.org/abs/2503.20314
project: null
code: https://github.com/Wan-Video/Wan2.1
tags:
  - Video-Generation
  - Diffusion-Transformer
  - Open-Source
  - Foundation-Model
  - Wan
created: 2026-08-27
updated: 2026-08-27
---

# Wan2.1：开源大规模视频生成基础模型

## 基本信息

- **作者**：Wanx 团队（通义万相）
- **机构**：阿里巴巴
- **发表**：arXiv 2503.20314（2025-03）
- **代码**：<https://github.com/Wan-Video/Wan2.1>

## 一句话结论

Wan2.1 是阿里开源的大规模视频生成基础模型（Wan 系列），采用 3D 变分自编码器 + 视频 Diffusion Transformer（DiT）架构，支持文生视频、图生视频等任务。它把「高质量开源视频基础模型」带到一个新台阶，并被大量机器人世界模型工作复用为视频骨干——Fast-WAM 用 Wan2.2-5B 做一次前向世界表征编码，Dyna-2 等也以 Wan 系为视频底座。

---

## 1. 背景与定位

在 SVD（Stable Video Diffusion）之后，业界需要更大、更强、且开源可控的视频基础模型。Wan2.1 填补了这一空缺：它把视频生成规模推到数十亿参数级，同时保持开源与可复现，成为 2025 年「视频基础模型」的公共底座之一。对机器人领域而言，Wan 系列的关键价值在于**提供了可免费复用的视频 DiT 权重**，让 VPP 的「视频模型当世界表征」路线不再局限于 SVD 一个选项。

## 2. 架构要点

- **3D VAE（Wan-VAE）**：把视频压缩到时空潜空间，作为扩散的表示；
- **视频 Diffusion Transformer（DiT）**：在潜空间做去噪生成，支持文生视频/图生视频；
- **开源可复现**：提供完整权重与推理代码，社区可低成本微调。

## 3. 在机器人世界模型中的应用

Wan 系列在本库多篇笔记中被引用：

- **Fast-WAM**：用 Wan2.2-5B 视频 DiT 做一次前向编码，得到隐式世界表征后直接生成动作，推理仅 190ms；
- **Dyna-2**：以 Wan 系视频模型为底座做百万小时缩放；
- 与 SVD、Cosmos 并列，成为「视频基础模型 → 世界动作模型」这条路线的主要视频骨干来源。

---

## 材料来源

| 本地文件 | 材料类型 | 原始来源 | 获取日期 | 用途 |
|---|---|---|---|---|
| — | 论文 PDF（未下载） | <https://arxiv.org/pdf/2503.20314> | 2026-08-27 | 精读主来源 |
| — | 代码仓库 | <https://github.com/Wan-Video/Wan2.1> | 2026-08-27 | 架构核对 |

> 注：本文为 VPP 专访文章中的「基础模型」类提及，本篇笔记为视频基础模型条目；官方 PDF 与关键图尚未单独下载。
