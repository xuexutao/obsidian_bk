---
type: paper-note
status: draft
domain: VLA
paper: DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset
year: 2024
arxiv: 2403.12945
doi: null
source: https://arxiv.org/abs/2403.12945
project: https://droid-dataset.github.io/
code: null
tags:
  - VLA
  - Dataset
  - Robot-Manipulation
  - In-The-Wild
  - Franka
created: 2026-08-27
updated: 2026-08-27
---

# DROID：大规模野外机器人操控数据集

## 基本信息

- **作者**：Alexander Khazatsky、Karl Pertsch 等（跨 13 所高校/机构的联合团队）
- **机构**：Stanford / UC Berkeley / Google DeepMind 等多家机构联合采集
- **发表**：RSS 2024（arXiv 2403.12945）
- **项目页**：<https://droid-dataset.github.io/>

## 一句话结论

DROID 是首个真正「野外（in-the-wild）」的大规模机器人操控数据集：在 564 个真实场景、跨 18 个实验室的 50 台 Franka 机械臂上采集 7.6 万条轨迹（350 小时），覆盖家居、办公、工业等多样场景与任务。它把机器人学习的数据来源从「实验室固定桌面」推向「真实世界分布」，成为 OpenVLA、π0、Ctrl-World、RoboArena 等一大批后续工作的共同训练/评测底座。

---

## 1. 背景与定位

在 DROID 之前，机器人操控数据主要来自两类：一是实验室固定场景的小规模自采数据（如 RT-1 的 130k 内部演示），二是仿真数据。前者场景单一、泛化弱，后者存在 sim-to-real gap。DROID 的核心主张是：**只有在足够多样、足够野外的真实场景里采集数据，才能训练出真正通用的机器人策略**。

DROID 在石麻笔记 EP06 及本库多篇笔记中被反复引用——Ctrl-World 用 DROID（95k 轨迹、564 场景）训练世界模型，RoboArena 用 DROID 机器人平台做策略评测，VPP 的数据策略也把它列为对比/补充来源。

## 2. 数据规模与采集

- **规模**：约 7.6 万条轨迹、350 小时，覆盖 564 个场景；
- **硬件**：50 台 Franka Panda 机械臂，跨 18 个实验室；
- **采集方式**：统一遥操作协议（带 SLAM 的相机系统 + 统一的机器人控制接口），保证跨机构数据的格式一致；
- **多样性**：场景从厨房、办公桌到工业台面，任务从抓取、放置到复杂多步操作，强调「野外」的真实分布。

## 3. 影响与评价

DROID 的贡献在于**标准化了「野外操控数据」这一稀缺资源**，使 VLA 训练得以摆脱对单机构私有数据的依赖。它配套的机器人平台与评测协议，也直接催生了 RoboArena 这类分布式评测框架。局限在于：数据仍以单臂 Franka 桌面操作为主，双机械臂、灵巧手、移动操作等更复杂的本体与任务覆盖有限。

---

## 材料来源

| 本地文件 | 材料类型 | 原始来源 | 获取日期 | 用途 |
|---|---|---|---|---|
| — | 论文 PDF（未下载） | <https://arxiv.org/pdf/2403.12945> | 2026-08-27 | 精读主来源 |
| — | 项目页 | <https://droid-dataset.github.io/> | 2026-08-27 | 数据规模核对 |

> 注：本文为 VPP 专访文章中的「数据集」类提及，本篇笔记为数据集条目；官方 PDF 与关键图尚未单独下载。
