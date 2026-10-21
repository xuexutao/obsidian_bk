# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2507.02546v1 | 全文，arXiv v1 | 2026-10-21 | 精读主来源 |
| `figures/01-teaser.png` | 官方源码原图 | https://arxiv.org/abs/2507.02546 | Figure 1 / Figure 6 对比内容 | 2026-10-21 | 米制尺度、细节与开放域定性对比 |
| `figures/02-method-pipeline.png` | 官方源码原图 | https://arxiv.org/abs/2507.02546 | Figure 2 | 2026-10-21 | affine point map + CLS scale head 总体架构 |
| `figures/03-data-refinement.png` | 官方源码原图 | https://arxiv.org/abs/2507.02546 | Figure 4 | 2026-10-21 | synthetic-guided 真实深度过滤与补全 |

## 官方在线材料

- arXiv：https://arxiv.org/abs/2507.02546
- 项目页：https://wangrc.site/MoGe2Page/
- 系列代码：https://github.com/microsoft/MoGe
- 模型权重：https://huggingface.co/Ruicheng/moge-2-vitl-normal

## 说明

- 笔记以 arXiv v1 官方 PDF、HTML 和官方源码图为主。
- 论文主线输出是 metric point map / depth；当前官方仓库后来提供 normal 版本权重，但论文正文没有独立 normal benchmark，因此笔记不把“法线 SOTA”作为论文结论。
- 三张关键图均由 arXiv 官方 TeX source 中的矢量 PDF 渲染为 PNG，未使用 PDF 内嵌碎片。
