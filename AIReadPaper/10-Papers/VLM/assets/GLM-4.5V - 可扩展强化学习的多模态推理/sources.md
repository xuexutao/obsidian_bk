# Sources

## 官方入口

- arXiv 页面：https://arxiv.org/abs/2507.01006
- 官方 PDF（v6，2026-01-01）：https://arxiv.org/pdf/2507.01006
- arXiv TeX source：https://arxiv.org/e-print/2507.01006
- 官方代码与模型：https://github.com/zai-org/GLM-V
- GLM-4.6V 官方概览：https://z.ai/blog/glm-4.6v
- arXiv DOI：https://doi.org/10.48550/arXiv.2507.01006
- 获取日期：2026-10-15

## 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2507.01006 | v6 全文，42 页 | 2026-10-15 | 精读主来源、版本关系与数字复核 |
| `figures/01-version-and-results.png` | 原文总览/结果图 | arXiv TeX source 中 `figs/intro_new.pdf`，300 DPI 渲染 | Figure 1 | 2026-10-15 | 4.1V→4.5V 的规模扩展与 RL 增益 |
| `figures/02-architecture.png` | 原文架构图 | arXiv TeX source 中 `figs/glm-4v-thinking.pdf`，300 DPI 渲染 | Figure 2 | 2026-10-15 | ViT、MLP Projector、LLM、原生分辨率与时间 token |
| `figures/03-reward-hacking.jpg` | 原文训练/失败图 | arXiv TeX source 中 `figs/rewardhacking-curve.jpg` | Figure 5 | 2026-10-15 | 低质量 verifier 导致跨领域 reward hacking 与崩溃 |
| `figures/04-cross-domain-rl.png` | 原文消融/迁移图 | arXiv TeX source 中 `figs/cross_domain_heatmap.pdf`，300 DPI 渲染 | Figure 6 | 2026-10-15 | 单领域与 Mix-all RL 的跨领域迁移 |

所有图片均来自官方 arXiv source，并已逐张本地预览验证完整性和清晰度；未使用 PDF 内嵌碎片。TeX source 仅用于取材，未在最终附件目录重复保留。

## 版本说明

- 本笔记以 arXiv v6 为准。论文标题仍保留 GLM-4.5V/4.1V-Thinking，但 v6 正文同时纳入 GLM-4.6V 与 4.6V-Flash。
- 4.1V-9B-Thinking 是 9B dense 的紧凑前作；4.5V 是 106B 总参数、12B 激活参数的 MoE；4.6V/4.6V-Flash 是更新版本，加入原生工具使用与 128K 级上下文。
- 论文没有披露从 4.1V checkpoint 到 4.5V 的直接扩参或蒸馏算法，笔记不作推断。