# Sources

本笔记的所有定量数据与架构描述均来自 Dyna Robotics 官方研究文章《Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models》。由于该文为网页版技术报告而非 PDF，且原文中的官方配图（Figure 1–Figure 18）无法在本轮抓取（dyna.co 在沙箱网络下不直达），本笔记的四张配图全部基于官方公开文字与数据**重绘**（忠实于原文描述与官方数值），并在每张图注里标明了官方对应 Figure 编号与原始链接。

## 本地材料

| 本地文件 | 材料类型 | 原始来源 | 报告位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `figures/01-scaling-laws.png` | 重绘示意图（官方拟合曲线+测量点） | https://www.dyna.co/dyna-2 | Figure 2 / Figure 5 | 2026-08-22 | 人类域内与跨具身缩放定律总览 |
| `figures/02-architecture.png` | 重绘架构示意图 | https://www.dyna.co/dyna-2 | Figure 3 | 2026-08-22 | MoT 双流 DiT 架构 |
| `figures/03-on-robot-posttraining.png` | 重绘实机缩放图 | https://www.dyna.co/dyna-2 | Figure 8 | 2026-08-22 | 14 任务实机后训练与涌现式能力 |
| `figures/04-video-co-training.png` | 重绘视频缩放轴与消融图 | https://www.dyna.co/dyna-2 | Figure 10 / Figure 11 | 2026-08-22 | 视频作为新缩放轴 |

## 原始来源链接

- 官方研究报告（精读主来源）：https://www.dyna.co/dyna-2
- 二级解读（用于交叉验证数字与公式）：
    - https://singularity.kiwi/dyna-2-robot-scaling-law-human-video-2026（提供确切幂律公式与数值，与官方 Figure 2 一致）
    - https://aiinsiders.net/article/dyna-claims-a-scaling-law-more-human-video-better-robot（独立批判性视角）
    - https://www.cet.com.cn/itpd/itxw/10500241.shtml（中文媒体转述）
    - https://www.toutiao.com/a7674910905645171250（中文媒体转述）
    - https://robot.m.ofweek.com/2026-08/ART-898890-11000-30698659.html（中文媒体转述）

## 官方报告的完整图表清单（未本地化）

> 出于完整性，下表列出官方报告 §1–§6 中提到的全部 18 张图及图注。由于沙箱网络无法直接访问 dyna.co（直连与代理 TLS 均失败；archive.org 也无法访问），本轮未能取得这些原始图片文件的本地副本。如后续需要，可在网络可达的环境下抓取 dyna.co 页面后用 `research-field-survey/scripts/extract_figure.py` 风格的整页渲染 + 裁剪流程补齐。

| 官方编号 | 图注摘录 | 主题 |
|---|---|---|
| Figure 1 | "Dyna-2 is a world-action model (WAM) pre-trained on over one million hours of human video data. It exhibits scaling laws on held-out human data, and for the first time, proves the existence of a human-to-robot transfer scaling law." | teaser，由 Dyna-2 自身生成的视频马赛克 |
| Figure 2 | "Scaling laws for Dyna-2 world-action models trained on a ladder of human video data, from 1,000 hours to 1,000,000 hours." | 两条缩放定律总览（人类域+零样本机器人） |
| Figure 3 | "The Dyna-2 architecture diagram." | MoT 双流 DiT 架构（详见笔记 §3.2） |
| Figure 4 | 预训练数据样本（36 个随机抽取的以自我为中心片段） | 数据样例展示 |
| Figure 5 | 1M 小时人类数据上的四项指标幂律 | 人类域内缩放详细曲线 |
| Figure 6 | 离线机器人评估任务的代表性图库 | 39 任务的视觉示例 |
| Figure 7 | 首次展示人类→机器人迁移缩放定律 | 跨具身迁移曲线 |
| Figure 8 | 预训练规模贯穿后训练延伸至机器人 | 14 任务实机后训练（详见笔记 §5.3） |
| Figure 9 | 后训练策略的自主运行示例（顶部摄像头，1× 速度） | 实机策略运行视频帧 |
| Figure 10 | 世界建模对人类→机器人迁移缩放定律的涌现至关重要 | Action-only vs Joint vs Video co-training 三路消融 |
| Figure 11 | 视频是新的缩放轴 | 固定动作量、扩展视频数据 |
| Figure 12 | 视频数据在大规模下的优势是跨具身泛化 | §3.5 实验 2（250k 动作小时固定） |
| Figure 13 | 实机聚合对比，归一化至 VLA | WAM vs VLA 柱状图（详见笔记 §5.6） |
| Figure 14 | 客户现场零样本部署通过率对比 | Dyna-1 vs Dyna-2 现场差距 |
| Figure 15 | 语言遵循分数 | 反事实基准任务结果 |
| Figure 16 | 为什么单步困难 | 各蒸馏方法对比示意 |
| Figure 17 | 一步蒸馏生成的指令条件未来视频（24 个生成视频） | §4.4 蒸馏管线的样本 |
| Figure 18 | 来自机器人数据后训练 Dyna-2 一步蒸馏管线的多视角一致未来视频（12 条指令） | §4.4 多视角一致性样本 |

## 写作说明

- 本轮抓取与写作于 2026-08-22 完成。Dyna Robotics 官方研究文章通过 WebFetch 服务端获取了完整正文（含 §1–§7 全部章节、Flow Matching 损失公式、四个评估指标的精确定义、1M 小时数据阶梯的全部数值、14 个实机任务的归一化分数、1 步蒸馏追逐博弈公式等）。
- 由于沙箱网络无法直接访问 dyna.co（包括 curl 直连、curl 经本地 https_proxy、Chrome / playwright 直连、jina.ai reader、archive.org Wayback Machine 等多种途径均失败），本轮未取得官方原图的本地副本。
- 为满足 README 的"图文并茂"要求，本笔记的 4 张配图均基于官方公开的**文字描述 + 数值数据**重绘：缩放定律曲线使用官方公开的精确幂律公式与四个数据点（1k/10k/100k/1M 小时）的真实测量值；架构图为官方 §2 描述的忠实结构示意；实机后训练与视频缩放轴两图使用官方公开的关键锚点数据（20%/28%/45%/53%、锁箱 0%→90%、开瓶盖 10 分钟演示、MSE 0.340→0.120 等）。
- 笔记末尾附有完整的批判性评价（§7）与个人启发（§9），其中明确指出"实验均由公司内部完成、缺乏外部复现、关键工程参数未披露"等已知局限——这是阅读时应当同步保留的怀疑。
- 如后续获得官方原图或模型权重，应优先把 4 张重绘图替换为官方版本，并通过 git / 笔记的 `updated` 字段保留时间戳。