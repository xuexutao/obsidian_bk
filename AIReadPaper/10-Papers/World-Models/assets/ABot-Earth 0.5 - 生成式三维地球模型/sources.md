# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | arXiv 官方预印本 PDF | https://arxiv.org/pdf/2606.09967 | 全文，17 页 | 2026-10-16 | 精读主来源 |
| `sources/github-tech-report.pdf` | GitHub 官方技术报告 | https://github.com/amap-cvlab/ABot-Earth-0.5/blob/main/tech-report.pdf | 全文，17 页 | 2026-10-16 | 核对官方仓库版本 |
| `figures/01-teaser.png` | arXiv 源码原图 | https://arxiv.org/abs/2606.09967 | Figure 1 | 2026-10-16 | 0.5 全球发布覆盖与城市样例 |
| `figures/02-data-pipeline.png` | arXiv 源码原图 | https://arxiv.org/abs/2606.09967 | Figure 2 | 2026-10-16 | 多源数据、3DGS 重建与质量筛选 |
| `figures/03-global-lod-pipeline.png` | arXiv 源码官方子图组合 | https://arxiv.org/abs/2606.09967 | Figure 4(a)(b) | 2026-10-16 | 全球切块与 zoom 14–19 LOD 层级 |
| `figures/04-lod-rendering.png` | arXiv 源码原图 | https://arxiv.org/abs/2606.09967 | Figure 5 | 2026-10-16 | 地图引擎 LOD 调度与渲染 |

## 官方链接

- arXiv：https://arxiv.org/abs/2606.09967
- arXiv HTML：https://arxiv.org/html/2606.09967
- GitHub：https://github.com/amap-cvlab/ABot-Earth-0.5
- 官方项目页：https://abot-earth.amap.com/

## 状态与核验说明

- ABot-Earth 0.5 确有 arXiv 预印本和官方技术报告，因此本条为 `type: paper-note`，没有伪造 `project-note` 或 `paper.pdf`。
- 截至 2026-10-16，官方仓库已宣布 ABot-Earth 0.7，项目页可能展示新版；本地材料与笔记只对应 0.5。
- 四张图片已使用 `Read` 逐张视觉核验；图像、文字、边缘和 LOD 标注完整。Figure 4 由源码中的两个官方子图并排整理，未修改数据含义。
- arXiv PDF 与 GitHub 技术报告均为 17 页并具有文本层。
- Table 2 的 FID/KID 基线来自不同 GT 与相机协议，不属于统一 benchmark；论文没有正式消融或闭环 UAV 实验。
