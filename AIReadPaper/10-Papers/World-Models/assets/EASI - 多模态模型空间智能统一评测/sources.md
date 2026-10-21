# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v5） | https://arxiv.org/pdf/2508.13142 | 全文，39 页 | 2026-10-16 | 精读主来源 |
| `figures/01-teaser.png` | 官方 LaTeX 原图转 PNG | https://arxiv.org/e-print/2508.13142 | Figure 1，`assets/figures/teaser_double.pdf` | 2026-10-16 | 非空间推理与 3D 视图失败对照 |
| `figures/02-taxonomy.png` | 官方 LaTeX 原图转 PNG | https://arxiv.org/e-print/2508.13142 | Figure 2，`assets/figures/concept_map.pdf` | 2026-10-16 | 六类空间能力 taxonomy |
| `figures/03-failure-cases.png` | 官方 LaTeX 原图转 PNG | https://arxiv.org/e-print/2508.13142 | Figure 3，`assets/case_studies/compare.pdf` | 2026-10-16 | MM/MR/SR/PT/DA/CR 失败案例 |

## 外部链接

- arXiv 页面：https://arxiv.org/abs/2508.13142
- 官方代码：https://github.com/EvolvingLMMs-Lab/EASI/
- 官方排行榜：https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard

## 完整性核验

- PDF 由 arXiv 官方 `pdf/2508.13142` 下载，版本对应 v5。
- 三张图片均来自官方 e-print 源码包，不是 PDF 内嵌碎片。
- 三张图片已用 Read 逐张预览：teaser 左右对照、六类 taxonomy 和纵向 case-study 表均完整，无裁切。
- 附录、详细分项结果、token 与 thinking-mode 分析均包含在主 PDF 中；无独立 supplementary PDF。
