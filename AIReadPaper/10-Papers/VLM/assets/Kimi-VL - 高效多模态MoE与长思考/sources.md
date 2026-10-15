# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2504.07491 | 全文 | 2026-10-15 | 精读主来源 |
| `figures/01-architecture.png` | 模型架构图 | https://arxiv.org/e-print/2504.07491 | Figure 3, p.3；源码 `figures/main-small.pdf` | 2026-10-15 | 解释 MoonViT、MLP、MoE decoder 与原生分辨率输入 |
| `figures/02-pretraining-stages.png` | 预训练阶段图 | https://arxiv.org/e-print/2504.07491 | Figure 4, p.4；源码 `figures/pretraining_stages.pdf` | 2026-10-15 | 解释 ViT、joint pretrain、cooldown、long-context 四阶段 |
| `figures/03-posttraining-stages.png` | 后训练阶段图 | https://arxiv.org/e-print/2504.07491 | Figure 5, p.6；源码 `figures/post-training-stages.pdf` | 2026-10-15 | 解释 32K/128K SFT、long-CoT SFT 与 RL |
| `figures/04-test-time-scaling.png` | 思考长度消融图 | https://arxiv.org/e-print/2504.07491 | Figure 13, p.18；源码 `figures/test_time_accuracy.pdf` | 2026-10-15 | 分析不同最大 thinking token 对准确率的影响 |

## 补充链接

- arXiv 摘要页：https://arxiv.org/abs/2504.07491
- 官方代码仓库：https://github.com/MoonshotAI/Kimi-VL
- 图片均由 arXiv 官方 LaTeX 源码中的完整矢量图导出，未使用 PDF 内嵌碎片。
