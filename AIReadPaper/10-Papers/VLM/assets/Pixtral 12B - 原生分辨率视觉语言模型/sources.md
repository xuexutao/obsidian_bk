# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2410.07073 | 全文，arXiv v2，24 页 | 2026-10-15 | 精读主来源 |
| `figures/01-main-results.png` | arXiv 源码完整双子图 | https://arxiv.org/e-print/2410.07073 | Figure 1，PDF p.2 | 2026-10-15 | MM-MT-Bench 与 LMSys-Vision 性能/规模对比 |
| `figures/02-vision-encoder.png` | arXiv 源码原图 | https://arxiv.org/e-print/2410.07073 | Figure 2，PDF p.3 | 2026-10-15 | Pixtral-ViT 的 RoPE-2D、packing 与 break token |
| `figures/03-full-architecture.png` | arXiv 源码原图 | https://arxiv.org/e-print/2410.07073 | Figure 3，PDF p.4 | 2026-10-15 | 多图交错输入与统一 decoder |
| `figures/04-vision-encoder-ablation.png` | arXiv 源码原图 | https://arxiv.org/e-print/2410.07073 | Figure 6，PDF p.8 | 2026-10-15 | Pixtral-ViT 与 CLIPA 消融 |

## 外部链接

- arXiv 摘要页：https://arxiv.org/abs/2410.07073
- 官方发布页：https://mistral.ai/news/pixtral-12b/
- 推理代码：https://github.com/mistralai/mistral-inference/
- 评测代码：https://github.com/mistralai/mistral-evals/
- MM-MT-Bench：https://huggingface.co/datasets/mistralai/MM-MT-Bench

> Figure 1 由源码中论文实际引用的两个子图按原顺序无裁切拼接；其余为源码完整原图。2026-10-15 已逐张预览，确认坐标轴、图例、文字和边缘完整。