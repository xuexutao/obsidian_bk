# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | https://arxiv.org/abs/2404.04057 | 全文 | 2026-08-29 | 精读主来源 |
| `figures/01-distillation-progression.png` | 蒸馏进程图 | arXiv 源码 2404.04057 / sid_plots/imagenet_1024/（10 张样本拼接） | Figure 1 | 2026-08-29 | 研究背景章节 |
| `figures/02-fid-evolution-cifar.png` | FID 演化图 | arXiv 源码 2404.04057 / sid_fid/jsonl_cifar_uncond_run001.pdf | Figure 2 | 2026-08-29 | 实验结果章节 |
| `figures/03-imagenet-samples.png` | 定性样本图 | arXiv 源码 2404.04057 / sid_plots/imagenet_fakes_1.200000_939176.png | Figure 13（附录） | 2026-08-29 | 实验结果章节 |

## 原始来源链接

- 论文 HTML 全文（精读主来源）：https://arxiv.org/html/2404.04057
- 论文摘要页：https://arxiv.org/abs/2404.04057
- 官方代码：https://github.com/mingyuanzhou/SiD

## 说明

- 图片取自 arXiv HTML 版官方图文件，未使用 PDF 内嵌图片碎片。
- 论文为 ICML 2024（PMLR 235）收录，无独立 DOI，frontmatter `doi` 记为 `null`。
- 教师模型：VP-EDM（CIFAR-10 / ImageNet-64 / FFHQ / AFHQ-v2）。
