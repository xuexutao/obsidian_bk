# Sources

获取日期：2026-09-04

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | arXiv 官方 PDF（v1） | <https://arxiv.org/pdf/2601.20540> | 全文，31 页 | 精读主来源、公式与实验核验 |
| `figures/01-teaser.png` | PDF 完整裁图 | 同上 | Figure 1, p.1 | 开放域生成与动作可控性概览 |
| `figures/02-training-pipeline.png` | PDF 完整裁图 | 同上 | Figure 4, p.8 | 三阶段训练总览 |
| `figures/03-method-pipeline.png` | PDF 完整裁图 | 同上 | Figure 5, p.9 | MoE DiT、Plücker 动作与 AdaLN 注入 |
| `figures/04-causal-post-training.png` | PDF 完整裁图 | 同上 | Figure 6, p.12 | Block-causal generator 与 adversarial discriminator |
| `figures/05-emergent-memory.png` | PDF 完整裁图 | 同上 | Figure 12, p.19 | 60 秒离屏重访及未观察状态演化 |
| `figures/06-quantitative-results.png` | PDF 完整裁图 | 同上 | Table 2, p.21 | VBench 定量对比 |

## 官方在线材料

| 材料 | URL | 获取日期 | 用途与核验结果 |
|---|---|---|---|
| arXiv 摘要页 | <https://arxiv.org/abs/2601.20540> | 2026-09-04 | 标题、作者、提交日期、DOI、摘要与项目链接 |
| 官方项目页 | <https://technology.robbyant.com/lingbot-world> | 2026-09-04 | 在线演示、能力说明、局限与 next steps |
| 官方 GitHub（v1） | <https://github.com/robbyant/lingbot-world> | 2026-09-04 | 代码、依赖、480P/720P 推理命令、Fast 脚本、模型发布状态；README 当前提示后续迁移至 v2 |
| Hugging Face 模型集合 | <https://huggingface.co/collections/robbyant/lingbot-world> | 2026-09-04 | Base (Cam)、Base (Act)、Fast checkpoint 入口 |
| ModelScope 模型页 | <https://www.modelscope.cn/models/Robbyant/lingbot-world-base-cam> | 2026-09-04 | Base (Cam) 镜像与推理说明交叉核验 |

## 资料缺口

- 未找到单独 Supplementary Material；arXiv v1 本身为 31 页技术报告。
- 论文没有披露总训练视频数/小时数、数据源比例、去重规则、训练 GPU 型号/数量、总 GPU-hours、学习率、batch size、flow shift 数值、Self-Forcing 截断长度 $K$、Diffusion Forcing 的 $N/L/m$ 或最终 few-step 步数。
- 没有独立消融表；high-noise 初始化、curriculum、MoE、Self-Forcing、DMD 与 adversarial head 的贡献均缺少数值拆解。
- `16 FPS` 仅说明 480p、one GPU node；论文未披露 node 内 GPU 数量。官方示例使用 8 个进程/GPU，因此不能解读为消费级单卡实时。
- 没有专门 failure-case 图或失败率；长期记忆与 10 分钟生成主要是定性样例，论文同时承认 drift、显式记忆缺失、动作空间有限、对象级控制困难和单 agent 限制。
