# Sources — Vidu S1 精读笔记附件溯源

| 文件 | 内容 | 来源 URL | 位置 / Figure | 获取日期 | 用途 |
| --- | --- | --- | --- | --- | --- |
| `../figures/fig1_overview.png` | Vidu S1 总览：语音引导未来控制 / 无限时长高效推理 / 多样自定义角色 | arXiv:2607.03118v2 官方 PDF | Figure 1, p.1 | 2026-10-15 | 笔记「研究背景」章节配图 |
| `../figures/fig2_data_pipeline.png` | 数据过滤流水线：Prefiltering → Clipping → Subject/Other Filtering → Diarization → Caption+Embedding | arXiv:2607.03118v2 官方 PDF | Figure 2, p.3 | 2026-10-15 | 笔记「数据准备」章节配图 |
| `../figures/fig3_preference.png` | Vidu-StreamBench 人类偏好评测（vs HeyGen / LemonSlice / Kling Avatar 2.0） | arXiv:2607.03118v2 官方 PDF | Figure 3, p.9 | 2026-10-15 | 笔记「实验」章节配图 |
| `../figures/fig4_qualitative.png` | 指令跟随定性对比（抬头沉思 / 竖大拇指，vs Kling Avatar 2.0） | arXiv:2607.03118v2 官方 PDF | Figure 4, p.9 | 2026-10-15 | 笔记「实验」章节配图 |
| `2607.03118v2.pdf` | 论文官方 PDF 全文（v2, 2026-07-21） | https://arxiv.org/pdf/2607.03118v2 | 全文 | 2026-10-15 | 精读底稿 |

补充线上资料（未落盘）：项目页 / 在线 Demo https://vidu.com/vidu-stream；GitHub 文档仓 https://github.com/shengshu-ai/Vidu-S1。

裁剪方式：`pymupdf` 按页 + 相对坐标完整裁剪（ZOOM 2.4），非 PDF 内嵌碎片抽取。
