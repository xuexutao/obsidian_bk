# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v1） | https://arxiv.org/pdf/2609.01560 | 全文，15 页 | 2026-10-16 | 精读主来源 |
| `figures/01-teaser.png` | arXiv 源码原图 | https://arxiv.org/e-print/2609.01560 | Teaser / Figure 1，PDF p.1 | 2026-10-16 | 多视觉域角色/相机控制 |
| `figures/02-method.png` | arXiv 源码原图 | https://arxiv.org/e-print/2609.01560 | Figure 3，PDF p.4 | 2026-10-16 | 动作语言化、packed sequence、single-egress routing |
| `figures/03-temporal-control.png` | arXiv 源码原图 | https://arxiv.org/e-print/2609.01560 | Figure 5，PDF p.7 | 2026-10-16 | 预训练先验与时序适配对照 |
| `figures/04-interface-comparison.png` | arXiv 源码原图 | https://arxiv.org/e-print/2609.01560 | Figure 6，PDF p.8 | 2026-10-16 | 文本动作接口对比 additive bias / FiLM |

## 官方在线材料

- arXiv 摘要页：https://arxiv.org/abs/2609.01560
- 项目页：https://danzer1xxxxchan.github.io/H3-World/
- 代码：https://github.com/Danzer1xxxxChan/H3-World
- LoRA 模型：https://huggingface.co/DANNY621/H3-World

## 身份与实时性核验

- 论文作者机构为 Tencent、National University of Singapore、The Hong Kong Polytechnic University。Danze Chen 与 Zeqing Wang 的工作脚注明确为在 Tencent 研究实习期间、由 Yeying Jin 指导完成。
- H3-World 的基础生成器是 **MiniMax-H3 33B**；H3-World 只训练 rank-32 LoRA 和 token refiner 等轻量参数。因此它是腾讯联合高校研究者对 MiniMax-H3 的适配工作，不是 MiniMax 团队自研的 H3 系列新版本。
- 论文结论明确写明当前模型“does not yet provide persistent world state, real-time interaction, planning, or policy learning”。仓库中的“5.2 秒”表示输出视频内容时长，不能当成生成耗时或实时证据。

## 图像与开源核验

- 四张图片均从 arXiv 官方源码的完整 PDF 图转换，并于 2026-10-16 使用 Read 逐张检查，无裁边或缺失图例。
- 源代码采用 Apache-2.0（另有说明者除外）；H3-World LoRA 是 MiniMax-H3 的模型衍生物，受 MiniMax H3 Community License Agreement 约束，不能按代码许可证推断权重许可。
