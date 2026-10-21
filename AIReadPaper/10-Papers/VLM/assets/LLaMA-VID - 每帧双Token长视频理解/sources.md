# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（含附录） | https://arxiv.org/pdf/2311.17043 | 全文，arXiv v1 | 2026-10-16 | 精读主来源 |
| `sources/source.md` | PDF 文本抽取 | https://arxiv.org/abs/2311.17043 | 全文 | 2026-10-16 | 全文检索与数字复核 |
| `figures/01-benchmark-overview.png` | 综合结果图 | https://arxiv.org/pdf/2311.17043 | Figure 1, p.1 | 2026-10-16 | 视频与图像基准总体对比 |
| `figures/02-method-pipeline.png` | 方法架构图 | https://arxiv.org/pdf/2311.17043 | Figure 2, p.3 | 2026-10-16 | 双 Token 生成与信息流 |
| `figures/03-training-stages.png` | 训练流程图 | https://arxiv.org/pdf/2311.17043 | Figure 3, p.4 | 2026-10-16 | 三阶段训练数据与格式 |
| `figures/04-token-ablation.png` | 消融表 | https://arxiv.org/pdf/2311.17043 | Tables 5–6, p.8 | 2026-10-16 | Token 数量与文本解码器消融 |
| `figures/05-context-attention.png` | 注意力可视化 | https://arxiv.org/pdf/2311.17043 | Figure 6, p.8 | 2026-10-16 | 验证指令条件化 context token |

## 公开实现

- 代码仓库：https://github.com/dvlab-research/LLaMA-VID
- arXiv 摘要页：https://arxiv.org/abs/2311.17043

## 说明

- `sources/source.pdf` 与 `sources/images/` 是既有取材缓存；本次复用原 PDF，并复制为规范路径 `sources/paper.pdf`。
- 正文引用的图片均从 `sources/paper.pdf` 对应页面以 300 DPI 完整裁剪，保留原始图号或表号及图注，且已逐图检查完整性和可读性。
