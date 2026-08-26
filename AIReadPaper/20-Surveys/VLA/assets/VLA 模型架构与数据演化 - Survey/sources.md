# VLA 模型架构与数据演化 - Survey · 图片来源溯源

本综述正文引用的 7 张关键图，均从各论文精读笔记的 `assets/` 目录复制而来。原始出处（论文 Figure 编号 / 页）与获取方式记录如下。

| 文件名 | 来源论文 | 原始 Figure | 说明 |
|---|---|---|---|
| `01-reconvla-paradigm.png` | ReconVLA | Fig.2 Paradigm Comparison | 用目标区域重建替代显式 grounding 的范式对比 |
| `02-vitra-pipeline.png` | VITRA | Pipeline 图 | 三阶段人类视频 → VLA 预训练数据管线 |
| `03-dreamzero-overview.png` | DreamZero | Fig.1 Overview | WAM 联合预测未来视频 + 动作的总览 |
| `04-primitivevla-method.png` | PrimitiveVLA | Fig.2 Method | Primitive-Centric Disassemble & Assemble 方法 |
| `05-qwenvla-overview.png` | Qwen-VLA | Overview 图 | 统一动作建模（embodiment prompt + 统一张量） |
| `06-gigabrain-framework.png` | GigaBrain-0 | Framework 图 | 世界模型数据引擎五种生成方式 |
| `07-gaussiandream-paradigm.png` | GaussianDream | Fig.1 Paradigm Comparison | 训练期 3D Gaussian 重建 + 未来演化预测 |
| `08-mmadavla-architecture.png` | MMaDA-VLA | Fig.2 architecture | 统一离散 token 空间 + 掩码去噪 + 混合注意力 + 并行生成未来观测与动作块 |
| `09-motus-architecture.png` | Motus | Fig.1 Motus Architecture | 三专家 MoT 架构（VGM / Action / Und.）+ Tri-model Joint Attention H-Bridge |
| `10-atlasvla-method.png` | AtlasVLA | method 图 | 4D 持久世界状态记忆（体素哈希）+ ego 工作状态记忆驱动 DiT |
| `11-spatial-physical-solution.png` | VLA More Generalizable | Fig.3 solution | 五种 VLA 视觉空间适配方案对比：LoRA / 换骨干 / Prompt / FTM / FLA |

## 官方链接

- ReconVLA：arXiv 2508.10333（AAAI 2026 杰出论文）
- VITRA：arXiv 2510.21571
- DreamZero：arXiv 2602.15922
- PrimitiveVLA：arXiv 2605.28634
- Qwen-VLA：arXiv 2605.30280
- GigaBrain-0：arXiv 2510.19430
- GaussianDream：arXiv 2605.20752
- MMaDA-VLA：arXiv 2603.25406（西湖大学等，2026.3）
- Motus：arXiv 2512.13030（生数科技 + 清华，2025.12）；后续工业版 MotuBrain：arXiv 2604.27792（2026.4）
- AtlasVLA：arXiv 2608.06729（中科院自动化所，2026.8）
- VLA More Generalizable Than You Think：CVPR 2026 Compute Transparency Champion Award，arXiv 2512.02902（中山大学 + X-Era AI Lab）

## 图 08–11 的获取方式

四张 2026 新范式论文的 pipeline 图，均从 arXiv 官方 e-print 源码包（`https://arxiv.org/e-print/<id>`）解压后提取原始 PDF figure，再用 PyMuPDF（fitz）以 2.5× 缩放（约 180 DPI）渲染为 PNG，以保证完整性和清晰度。AtlasVLA 的方法 PDF 原稿为纵向排版，渲染后用 `page.set_rotation(90)` 摆正。来源对应关系：

- `08-mmadavla-architecture.png` ← `arxiv/e-print/2603.25406` → `assets/architecture.pdf`（fig:arch，Fig.2 Schematic of the framework and training pipeline）
- `09-motus-architecture.png` ← `arxiv/e-print/2512.13030` → `figs/block.pdf`（在 docs/1_introduction.tex:13 引用，width=0.7\linewidth，Motus Architecture）
- `10-atlasvla-method.png` ← `arxiv/e-print/2608.06729` → `method-AAAI-final-2.pdf`（AtlasVLA 主体方法图，旋转 90°）
- `11-spatial-physical-solution.png` ← `arxiv/e-print/2512.02902` → `figs/solution.pdf`（sec/3_method.tex:42 引用，fig:solution，对比 LoRA / 换骨干 / Prompt / FTM / FLA）

各论文的完整原文 PDF、补充材料与详细精读笔记（仅 7 篇精读论文），见 `10-Papers/VLA/<论文名>/` 及其 `assets/<论文名>/sources.md`。

## 第 8 章「系列脉络」的核实来源

第 8 章的系列代际链（RT / Gemini Robotics / π / GR00T / OpenVLA-OFT / Octo / WAM / 离散扩散 / 数据系列）均经 WebSearch 于 2026-08-26 二次核实，关键节点与数字来源如下：

- **RT 系列**：RT-1（2022.12）→ PaLM-E（2023.3，562B）→ RT-2（2023.7，首创 VLA）→ RT-X（2023.10，RT-1-X 较 RT-1 高 50%、RT-2-X 泛化约为 RT-2 的 3 倍）→ AutoRT/SARA-RT/RT-Trajectory（2024.1）→ RT-H（2024.3）。来源：亿欧、CSDN v_JULY_v《Google 视觉机器人超级汇总》、MakerPro RT-X 解读。
- **Gemini Robotics 系列**：Gemini Robotics/ER（2025.3）→ On-Device（2025.6）→ 1.5/ER 1.5（2025.9，Motion Transfer）→ ER 1.6（2026.4，仪表读取 23%→93%，落地 Boston Dynamics Spot）→ 2/ER 2（2026.7，ER 2 经 Gemini API 公开）。来源：aiwiki.ai Gemini Robotics 词条、百度百科 ER 1.6、DEV Community ER 1.6 Developer Guide。
- **π 系列**：π0（2024.10）→ π0-FAST（2025.1）→ Hi Robot（2025.2）→ π0.5（2025.4）→ π0.5-KI（2025.5）→ π0.6/π*0.6（2025.11，RECAP）→ π0.7（2026.4，组合泛化）。来源：搜狐《PI VLA 模型解读系列》、网易 π0.7 报道。
- **GR00T 系列**：N1（2025.3，Eagle-2 双系统）→ N1.5（2025.5，GR00T-Dreams 36 小时）→ N1.6（2025.9，Cosmos-Reason-2B，DiT 16→32 层）→ N1.7（2026.4，EgoScale 20,854 小时人类视频 + 首个 dexterity scaling law）。来源：aiwiki.ai GR00T N1 词条、NVIDIA Research GR00T N1.6 页。
- **OpenVLA / Octo**：OpenVLA（2024.6，Prismatic-7B，OXE 970K）→ OpenVLA-OFT（2025.2，arXiv:2502.19645，并行解码+连续动作+L1 回归，26× 加速，LIBERO 76.5%→97.1%）；Octo（2024.5，RSS 2024，OXE 800K，Octo-Small 27M/Octo-Base 93M）。来源：claru.ai OpenVLA 页、GitHub moojink/openvla-oft、百度百科 Octo。
- **人员流动暗线**：Chelsea Finn / Sergey Levine / Karol Hausman / Brian Ichter 由 Google 出走创办 Physical Intelligence，并主导 Octo / OpenVLA。来源：搜狐《机器人开源革命："免费大脑"背后的四派力量与博弈》。

> 重要更正：Octo 并无名为「Octo-Spatial」的官方续作；SpatialVLA（上海 AI Lab）是独立工作，不属 Octo 系列。综述第 8.5 节已注明。
