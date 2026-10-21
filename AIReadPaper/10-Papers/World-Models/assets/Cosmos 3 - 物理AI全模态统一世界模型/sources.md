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
