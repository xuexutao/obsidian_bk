# Sources — Vidu S2 精读笔记附件溯源

| 文件 | 内容 | 来源 URL | 位置 / Figure | 获取日期 | 用途 |
| --- | --- | --- | --- | --- | --- |
| `../figures/fig1_overview.png` | Vidu S2 总览：S2-Avatar（720p 实时数字人 + 动态参考）/ S2-Editing（风格/主体/服装/背景编辑）/ 实时 Spatial Video | arXiv:2609.11638v1 官方 PDF | Figure 1, p.1 | 2026-10-15 | 笔记「研究背景」章节配图 |
| `../figures/fig2_data_pipeline.png` | S2-Avatar 与 S2-Editing 双数据流水线（切点检测、高清晰度筛选、背景稳定、时间有序密集 caption、V2V 风格数据构造） | arXiv:2609.11638v1 官方 PDF | Figure 2, p.4 | 2026-10-15 | 笔记「数据准备」章节配图 |
| `../figures/fig3_agentic.png` | VLM Agentic System 流水线：读取指令/图像 → 生成 prompt → 检查视频帧反馈 | arXiv:2609.11638v1 官方 PDF | Figure 3, p.8 | 2026-10-15 | 笔记「Agentic System」章节配图 |
| `../figures/fig6_gsb_avatar.png` | GSB 人类偏好：S2-Avatar vs Runway Character GWM-1 / PixVerse Image Avatar / HeyGen | arXiv:2609.11638v1 官方 PDF | Figure 6, p.17 | 2026-10-15 | 笔记「实验」章节配图 |
| `../figures/fig8_qualitative.png` | 数字人生成定性对比：身份/发型/手部几何保持（vs PixVerse、Runway） | arXiv:2609.11638v1 官方 PDF | Figure 8, p.18 | 2026-10-15 | 笔记「实验」章节配图 |
| `../figures/fig13_spatial.png` | Spatial Video 示例：S2-Avatar 与 S2-Editing 生成的左右目同步视图 | arXiv:2609.11638v1 官方 PDF | Figure 13, p.21 | 2026-10-15 | 笔记「空间视频」章节配图 |
| `2609.11638v1.pdf` | 论文官方 PDF 全文（v1, 2026-09-10） | https://arxiv.org/pdf/2609.11638v1 | 全文 | 2026-10-15 | 精读底稿 |

补充线上资料（未落盘）：项目页 / 在线 Demo https://vidu.com/vidu-stream；GitHub 文档仓 https://github.com/shengshu-ai/Vidu-S；API 文档 https://platform.vidu.com/vidu-stream/doc。

裁剪方式：`pymupdf` 按页 + 相对坐标完整裁剪（ZOOM 2.4），非 PDF 内嵌碎片抽取。
