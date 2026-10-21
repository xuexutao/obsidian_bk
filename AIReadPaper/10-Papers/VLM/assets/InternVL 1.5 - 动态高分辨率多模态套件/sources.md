# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2404.16821 | 全文 | 2026-10-16 | 精读、模型结构、数据与实验主来源 |
| `figures/01-commercial-comparison.png` | 与闭源模型代表结果 | https://arxiv.org/pdf/2404.16821 | Figure 1, p.1 | 2026-10-16 | 展示 OCR、图表与综合任务结果 |
| `figures/02-architecture.png` | ViT–MLP–LLM 总体架构 | https://arxiv.org/pdf/2404.16821 | Figure 3, p.3 | 2026-10-16 | 说明 dynamic high resolution、pixel shuffle 与 projector |
| `figures/03-dynamic-resolution.png` | 动态高分辨率切图流程 | https://arxiv.org/pdf/2404.16821 | Figure 4, p.4 | 2026-10-16 | 说明宽高比匹配、tile 与 thumbnail |
| `figures/04-resolution-ablation.png` | tile 数量消融 | https://arxiv.org/pdf/2404.16821 | Figure 6, p.8 | 2026-10-16 | 说明分辨率收益的任务依赖性 |

## 补充链接

- arXiv 摘要页：https://arxiv.org/abs/2404.16821
- 官方代码：https://github.com/OpenGVLab/InternVL
- 官方模型：https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5
- 图片均由官方 PDF 对应图表区域以 300 DPI 裁取，并逐张 Read 验证完整性与可读性。
