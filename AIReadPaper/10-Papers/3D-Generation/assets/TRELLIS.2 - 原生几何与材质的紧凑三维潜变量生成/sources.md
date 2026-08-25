# Sources

本文件记录 `TRELLIS.2 - 原生几何与材质的紧凑三维潜变量生成.md` 全部本地材料的溯源信息。

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `overview_v5.png` | 方法总览图 | https://arxiv.org/html/2512.14692 | Figure 2 | 2026-08-20（原有保留） | §3.1 总体框架 |
| `representation.png` | O-Voxel 表示图 | https://arxiv.org/html/2512.14692 | Figure 3 | 2026-08-20（原有保留） | §3.2 O-Voxel |
| `03-scvae-architecture.png` | SC-VAE 架构图 | https://arxiv.org/html/2512.14692v1/ | Figure 4 | 2026-08-21 | §3.3 SC-VAE |
| `04-qualitative-comparison.png` | 定性对比图 | https://arxiv.org/html/2512.14692v1/ | Figure 6 | 2026-08-21 | §5.2 定性结果 |
| `05-test-time-scaling.png` | 测试时缩放图 | https://arxiv.org/html/2512.14692v1/ | Figure 8 | 2026-08-21 | §5.4 效率 / 测试时缩放 |

## 文字与文档来源

| 材料 | 类型 | 原始链接 | 获取日期 | 用途 |
|---|---|---|---|---|
| 论文 HTML 全文（v1，含附录 A–F） | 论文原文 | https://arxiv.org/html/2512.14692 | 2026-08-21 | 精读主来源 |
| arXiv 摘要页 | 元数据 | https://arxiv.org/abs/2512.14692 | 2026-08-21 | 基本信息 |
| 项目页 TRELLIS.2 | 项目宣传材料 | https://microsoft.github.io/TRELLIS.2/ | 2026-08-21 | 背景补充 |

> 说明：本次未单独下载 PDF 与 Supplementary 文件；论文附录（A 实现细节、B FlexGEMM、C 数据准备、D 评测协议与用户研究、E 更多结果、F 局限）已随 HTML 全文一并阅读，故本地不保留 `sources/paper.pdf` 与 `sources/supplementary.pdf`。图片均取自 arXiv HTML 版官方原图，未使用 PDF 内嵌碎片图。
