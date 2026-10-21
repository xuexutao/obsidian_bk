# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2507.17744 | 全文，arXiv v1，30 页 | 2026-10-16 | 精读主来源 |
| `figures/01-teaser.png` | arXiv LaTeX 源码原图 | https://export.arxiv.org/e-print/2507.17744 | Figure 1，PDF p.1，`figures/yume1.pdf` | 2026-10-16 | 任务与交互形式 |
| `figures/02-overview.png` | arXiv LaTeX 源码原图 | https://export.arxiv.org/e-print/2507.17744 | Figure 2，PDF p.9，`figures/yume2.pdf` | 2026-10-16 | 四组件总体框架 |
| `figures/03-long-video-memory.png` | arXiv LaTeX 源码原图 | https://export.arxiv.org/e-print/2507.17744 | Figure 3，PDF p.10，`figures/yume3.pdf` | 2026-10-16 | 历史压缩与自回归长视频 |
| `figures/04-qualitative-results.png` | arXiv LaTeX 源码原图 | https://export.arxiv.org/e-print/2507.17744 | Figure 6，PDF p.20，`figures/yume4.pdf` | 2026-10-16 | 开放域与外部图像编辑结果 |
| `figures/05-aam-results.png` | arXiv LaTeX 源码原图 | https://export.arxiv.org/e-print/2507.17744 | Figure 7，PDF p.20，`figures/vis2.pdf` | 2026-10-16 | AAM 结构细节对比 |

## 在线来源

- arXiv 摘要页：https://arxiv.org/abs/2507.17744
- arXiv HTML：https://arxiv.org/html/2507.17744
- 官方项目页：https://stdstu12.github.io/YUME-Project/
- 论文给出的原始仓库：https://github.com/stdstu12/YUME
- 当前可访问官方仓库：https://github.com/Trilec/YUME
- 模型权重：https://huggingface.co/stdstu123/Yume-I2V-540P
- Sekai 数据代码：https://github.com/Lixsp11/sekai-codebase

## 开源与缺口核验

- 当前仓库同时包含 Yume 1.0/1.5 相关实现；可定位 MVDT、FramePack-like 训练、长视频 DDP/FSDP 推理、TTS-SDE、对抗蒸馏和相机轨迹量化代码。
- 官方 README 声称开放数据、训练/推理代码和权重；训练 MVDT 至少需要 16 张 A100。论文的 14B/540p 配置与后续 5B/720p 版本必须区分。
- Sekai 提供数据与处理代码，但论文训练所用 139,019 clips 的精确快照、过滤版本和全部标注并未在论文中形成可校验 manifest。
- 论文未披露 A800 数量、训练总时长、TTS-SDE 噪声强度、完整缓存配置和蒸馏超参数，严格数值复现仍受限。
