# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方技术报告 PDF | https://arxiv.org/pdf/2606.02800 | 全文，139 页 | 2026-10-16 | 精读主来源 |
| `figures/01-unified-backbone.png` | 全模态统一能力总览 | https://arxiv.org/pdf/2606.02800 | Figure 1, p.5 | 2026-10-16 | 解释一个模型覆盖 VLM、生成、策略与动力学 |
| `figures/02-unified-action.png` | 统一动作表示 | https://arxiv.org/pdf/2606.02800 | Figure 3, p.8 | 2026-10-16 | 解释 ego/effector/grasp 的跨具身动作 token 化 |
| `figures/03-action-modes.png` | FD/ID/Policy 三种动作模式 | https://arxiv.org/pdf/2606.02800 | Figure 4, p.10 | 2026-10-16 | 解释 clean/noisy token 组合如何统一动作任务 |
| `figures/04-mot-architecture.png` | MoT 双塔架构与注意力掩码 | https://arxiv.org/pdf/2606.02800 | Figure 5, p.11 | 2026-10-16 | 解释 AR Reasoner 与 diffusion Generator 的非对称耦合 |
| `figures/05-action-synergy.png` | 跨动作域协同消融 | https://arxiv.org/pdf/2606.02800 | Figure 28, p.71 | 2026-10-16 | 解释统一动作中训的正迁移与干扰 |

## 官方在线来源

- arXiv 摘要与版本页：https://arxiv.org/abs/2606.02800
- NVIDIA Cosmos 3 官方项目页：https://research.nvidia.com/labs/cosmos-lab/cosmos3/
- 官方技术报告：https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- 官方代码与推理入口：https://github.com/NVIDIA/Cosmos
- 训练/后训练框架：https://github.com/NVIDIA/cosmos-framework
- 数据策展框架：https://github.com/NVIDIA/cosmos-curator
- 自动评估框架：https://github.com/NVIDIA/cosmos-evaluator
- 官方模型集合：https://huggingface.co/collections/nvidia/cosmos3

## 核验说明

- 五张图片均从本地官方 PDF 按原始 Figure 页面完整裁剪并保留图注，已用 `Read` 逐张视觉核验；动作模式图和 MoT 图首次裁剪不完整，已调整边界并再次核验。
- 论文是架构、训练数据、计算规模、baseline 与消融的主来源；官方项目页和代码仓库用于核对模型发布档位、开放许可、推理硬件建议及官方已知限制。
- 技术报告披露 Nano/Super 预训练分别为 31.05T/17.86T token、使用 1024/2048 张 GB200，中训再用 2.4T/1.9T token；未给出完整 wall-clock，因此不能可靠换算总 GPU-hours 或美元成本。
- 2026-10-24 补写正文：逐页通读 v4 全文（139 页）后成稿，所有架构公式、数据量级、算力、评测数值、消融结论均直接引自该 PDF 的正文（Sec. 2–6）与附录（D/E/F），未采用二手解读。
- 版本与许可核对：arXiv abs 页显示 v1 提交于 2026-06-01、v4 于 2026-06-23，DOI 10.48550/arXiv.2606.02800；摘要声明代码/权重/SDG 数据/基准以 Linux Foundation OpenMDW-1.1 许可发布。
- 仓库现状（2026-10 核）：`NVIDIA/Cosmos` 已把 Cosmos 3 的 cookbook 与 evaluation 迁至 `NVIDIA/cosmos-framework`，主仓仅保留 README / COOKBOOK / 推理基准文档并以短链跳转；`RELEASE.md`、`CONTRIBUTING.md` 已在同步提交中删除，因此路线图类信息（如 Edge 发布）需以论文与 HuggingFace 集合页为准。
- 笔记中未引用 Fig. 8（Generator 数据课程）的逐格数字，因为 PDF 文本抽取会把三列数字打乱顺序，容易造成误读；正文改用正文段落中明确给出的量级（767M 图像、347.7M 视频、138.9M 带音轨、中训 15.6M 图像 / 74.7M 视频 / 18.8M 音频 / 8.4M 动作 episode）。
