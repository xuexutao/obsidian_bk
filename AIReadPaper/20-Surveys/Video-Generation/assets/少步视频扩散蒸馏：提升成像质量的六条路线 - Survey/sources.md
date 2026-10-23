# 溯源：少步视频扩散蒸馏——提升成像质量的六条路线

获取日期：2026-10-23
撰写用途：回答「DuoMatching 属于哪一类、近邻有哪些、还有哪些路线能提升少步蒸馏后的成像质量」，归类到 `20-Surveys/Video-Generation/`。

---

## 一、配图溯源

| 文件 | 原始 Figure | 页码 | 来源 PDF | 获取方式 | 用途 |
|---|---|---|---|---|---|
| `figures/01-duomatching-teaser-quality-gap.png` | DuoMatching Figure 1 | p.2 | arXiv:2610.03543v1 | 从官方 PDF 按 caption bbox 精确裁剪，逐张视觉抽查 | 展示"少步学生的质量缺口"，含首帧对比与 Q-Align 曲线，作为全文立论证据 |
| `figures/02-duomatching-overview.png` | DuoMatching Figure 3 | p.4 | arXiv:2610.03543v1 | 同上 | 展示联合-边缘匹配目标 + LatentBridge + Latent Variation Sampling 三组件 |
| `figures/03-joint-vs-marginal-signal.png` | DuoMatching Figure 2 | p.2 | arXiv:2610.03543v1 | 同上 | 展示 Joint vs Marginal matching signal 的空间结构差异，说明"联合监督会压制单帧改进" |

三张图均复用自 `10-Papers/Video-Generation/assets/DuoMatching - 联合边缘分布匹配的少步视频生成/figures/`，为从 arXiv 官方 PDF 裁剪的**完整原图**（非 PDF 内嵌碎片），已通过视觉抽查确认无正文混入。

论文 PDF 原件：`10-Papers/Video-Generation/assets/DuoMatching - 联合边缘分布匹配的少步视频生成/sources/duomatching-2610.03543v1.pdf`

---

## 二、文献来源（按六条路线）

### E 类：跨模态与多教师（本文主角所在）

| 工作 | 来源 | 状态 |
|---|---|---|
| DuoMatching: Joint-Marginal Distribution Matching for Few-Step Video Generation (2026.10) | https://arxiv.org/abs/2610.03543 | **已核实**，已有精读笔记 |
| MILD: From Static to Dynamic — On-Policy Distillation from Image to Video Diffusion Models (HKU, 2026.09) | https://arxiv.org/abs/2609.34371 | **已核实**（摘要与 Fig.1 成本对比已读），尚无精读笔记 |

### D 类：奖励与偏好对齐

| 工作 | 来源 | 状态 |
|---|---|---|
| Reward Lightning: Fast Video Generation via Homologous Preference Distillation (ECCV 2026) | https://arxiv.org/abs/2607.03960 | **已核实**（LRM 优于像素级/潜空间奖励基线 11.0% / 14.7%；VBench 平均 +2.1%） |
| Reward Forcing: Efficient Streaming Video Generation with Rewarded DMD (CVPR 2026) | 项目页 https://reward-forcing.github.io | 摘要与方法描述已核实（EMA-Sink + Re-DMD，单 H100 23.1 FPS），arXiv ID 未核对 |
| Stream-R1 | 检索自 researchpapers.dev 解读页 | 核心数字已读（VBench 84.40 vs 教师 84.26；1000 步 / 8×A100 / ~56h），arXiv ID 未核对 |

### C 类：对抗与判别

| 工作 | 来源 | 状态 |
|---|---|---|
| DMAD: Distribution Matching as Adversarial Distillation (Texas A&M + ByteDance, 2026.10) | 检索自 lilys.ai / ai.miraheze.org 解读 | 核心数字已读（Wan2.1-14B 四步 VBench 85.15；ImageNet-64 单步 FID 1.04），arXiv ID 未核对 |
| ADD / SDXL-Turbo (Stability AI, 2023.11) | arXiv:2311.17042 | 常识性引用，未逐字核对 |
| ADM / DMDX | 见 DMAD 相关工作总结 | 二手引用 |

### B 类：分数分布匹配

| 工作 | 来源 | 状态 |
|---|---|---|
| DMD / DMD2 / SiD | 已有精读笔记，见 [[因果视频蒸馏路线 - Survey]] | 已核实 |
| f-distill、SGMD、SenseFlow、DP-DMD、PhasedDMD、Uni-Instruct、FSF-DMD | 检索自 emergentmind 与 DMAD 论文的 related work 总结 | **二手引用**，未逐篇核实，引用前需查原文 |

### A 类：轨迹与一致性

| 工作 | 来源 | 状态 |
|---|---|---|
| Progressive Distillation (Salimans & Ho, 2022) | 常识性引用 | 未逐字核对 |
| Consistency Models / sCM / rCM、MeanFlow、PDD、AnyFlow | 见 DMAD 与 Data-Forcing 的 related work | 二手引用 |
| TMD: Transition Matching Distillation (Nie et al., 2026.01) | 见 emergentmind 汇总 | 二手引用 |
| Data-Forcing Distillation (2026.06) | https://arxiv.org/abs/2606.18478 | **已核实**（其 Sec.2 对 trajectory / distribution-matching 两大类给了较完整梳理） |

### F 类：系统与工程

- TurboDiffusion（rCM 步蒸馏 + attention/权重量化）——二手引用
- LongLive-2.0（NVFP4）——已有精读笔记 [[LongLive-2.0 - NVFP4长视频生成训练与推理系统]]

---

## 三、方法可信度说明

本篇是**谱系梳理型综述**，不是精读。六条路线的**分类框架与 DuoMatching 的定位**来自对原始论文的阅读；但 A/B/C 三类中多数工作的具体数字属于**二手引用**（来自汇总站点与相关工作的 related work 段落），**引用前请回到原文核对**，尤其是：

- DMAD 的 VBench 85.15 / FID 1.04
- TMD、f-distill、SGMD 的具体机制描述
- Stream-R1 的训练成本（1000 步 / 8×A100 / 56h）

已在正文表格中注明"未核对 ID"者，均为检索所得但未经原文逐字验证的条目。
