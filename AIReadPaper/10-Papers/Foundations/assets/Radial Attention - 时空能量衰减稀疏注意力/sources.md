# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2506.19852v2 | arXiv v2 / NeurIPS 2025 camera-ready 全文 | 2026-10-20 | 精读主来源 |
| `figures/01-teaser.png` | 论文关键图 | https://arxiv.org/pdf/2506.19852v2 | Figure 1，PDF p.1 | 2026-10-20 | 默认长度与 4× 长视频的总体效果 |
| `figures/02-energy-decay.png` | 观察与动机图 | https://arxiv.org/pdf/2506.19852v2 | Figure 4，PDF p.5 | 2026-10-20 | 时空能量衰减现象 |
| `figures/03-radial-mask.png` | 核心方法图 | https://arxiv.org/pdf/2506.19852v2 | Figure 5，PDF p.6 | 2026-10-20 | 计算密度、mask 与实际稀疏模式 |
| `figures/04-long-video-results.png` | 定性结果图 | https://arxiv.org/pdf/2506.19852v2 | Figures 6–7，PDF p.9 | 2026-10-20 | 默认长度与 4× 长视频视觉对比 |
| `figures/05-lora-regression.png` | 分析与消融图 | https://arxiv.org/pdf/2506.19852v2 | Figure 8，PDF p.10 | 2026-10-20 | LoRA 有效性与指数回归 |

## 官方在线材料

- arXiv：https://arxiv.org/abs/2506.19852
- arXiv HTML：https://arxiv.org/html/2506.19852v2
- 官方代码：https://github.com/mit-han-lab/radial-attention
- 许可证：https://github.com/mit-han-lab/radial-attention/blob/main/LICENCE.txt

## 取材说明

- 笔记基于 2025-12-06 修订的 arXiv v2，即 NeurIPS 2025 camera-ready 版本；v2 将主文压缩为 10 页并重排章节。
- 五张图均从官方 PDF 的完整页面区域裁剪，保留原始图注和必要标注，未使用 PDF 内嵌碎片图。
- 训练和消融细节同时核对了 Appendix B、D：LoRA rank 128；HunyuanVideo/Mochi/Wan2.1 分别训练 2400/1200、1500、2500 steps；默认长度使用 12 个 dense warm-up steps，4× 长度使用 2 个；训练保留前两个 DiT blocks 为 dense attention。
- 官方代码仓库采用 Apache-2.0；底层模型、数据和第三方组件仍需分别遵守各自许可证。
