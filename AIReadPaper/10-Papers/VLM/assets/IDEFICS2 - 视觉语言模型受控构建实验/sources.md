# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2405.02246 | 全文 | 2026-10-16 | 精读、公式、实验与消融主来源 |
| `figures/01-architecture.png` | fully autoregressive 架构图 | https://arxiv.org/pdf/2405.02246 | Figure 2, p.3 | 2026-10-16 | 说明视觉 token 经投影与 Perceiver 后并入 LLM 序列 |
| `figures/02-ocr-base-result.png` | 基座模型结果与 OCR 示例 | https://arxiv.org/pdf/2405.02246 | Table 8 + Figure 3, p.8 | 2026-10-16 | 说明 64 token 效率及文字转录能力 |
| `figures/03-architecture-ablation.png` | 架构收敛曲线 | https://arxiv.org/pdf/2405.02246 | Figure 4, p.20 | 2026-10-16 | 对比 fully autoregressive 与 cross-attention 的 LoRA 训练 |

## 补充链接

- arXiv 摘要页：https://arxiv.org/abs/2405.02246
- 模型集合：https://huggingface.co/collections/HuggingFaceM4/idefics2-661d1971b7c50831dd3ce0fe
- 训练数据 The Cauldron：https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
- 图片均由官方 PDF 对应图表区域以 300 DPI 裁取，并逐张 Read 验证完整性与可读性。
