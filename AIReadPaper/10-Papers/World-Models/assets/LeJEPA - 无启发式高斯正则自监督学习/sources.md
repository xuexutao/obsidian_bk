# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v3） | https://arxiv.org/pdf/2511.08544 | 全文（50 页，含证明与附录） | 2026-10-15 | 精读主来源、公式、定理、实验与边界核对 |
| `sources/paper.txt` | 从官方 PDF 提取的本地全文文本 | https://arxiv.org/pdf/2511.08544 | 全文 | 2026-10-15 | 章节检索与数字交叉核对 |
| `figures/01-sigreg-pipeline.png` | arXiv TeX 源图组合 | https://arxiv.org/e-print/2511.08544v3 | Figure 2，p.3；源文件 `teaser_manifold_0.png` 与 `teaser_manifold_1.png` | 2026-10-15 | SIGReg 从任意数据流形到随机投影高斯匹配的总览 |
| `figures/02-directional-test.png` | arXiv TeX 矢量源图渲染 | https://arxiv.org/e-print/2511.08544v3 | Figure 5，p.8；源文件 `2d_slicing_example_2.pdf` | 2026-10-15 | 说明协方差正常仍可非高斯，以及方向检验如何发现退化 |
| `figures/03-lambda-stability.png` | arXiv TeX 矢量源图渲染 | https://arxiv.org/e-print/2511.08544v3 | Figure 8，p.13；源文件 `performance_vs_lambda.pdf` | 2026-10-15 | λ 与视图数量的消融和稳定性边界 |
| `figures/04-in-domain-results.png` | arXiv TeX 矢量源图渲染 | https://arxiv.org/e-print/2511.08544v3 | Figure 12，p.16；源文件 `exps/galaxy10.pdf` | 2026-10-15 | Galaxy10 域内预训练与 DINOv2/v3 迁移比较 |

## 官方页面与实现

| 材料 | 原始来源 | 获取日期 | 说明 |
|---|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2511.08544 | 2026-10-15 | 作者、摘要、版本历史；v3 更新于 2025-11-14 |
| 官方代码入口 | https://github.com/rbalestr-lab/lejepa | 2026-10-15 | arXiv 明确给出的代码链接；当前重定向到 `galilai-group/lejepa` |
| 当前代码仓库 | https://github.com/galilai-group/lejepa | 2026-10-15 | SIGReg 包、最小训练示例、推荐超参数与实验结果 |
| PyPI 安装方式 | https://github.com/galilai-group/lejepa | 2026-10-15 | README 给出 `pip install lejepa` |

## 版本、许可证与材料说明

- 本地 PDF 保存 arXiv v3 官方版本；论文暂未在 arXiv 页面声明会议或期刊正式发表信息，因此 frontmatter 不编造 venue。
- 官方仓库许可证为 **CC BY-NC 4.0**，允许署名条件下的非商业复制与修改；商业产品直接复用代码前需另行确认授权。
- 论文没有独立 supplementary PDF；证明、额外消融、计算耗时与实验表均包含在 50 页 arXiv PDF 的附录中。
- Figure 2 在 TeX 中由两张官方 PNG 与箭头排版组成；本地 `01-sigreg-pipeline.png` 按同一左右顺序组合，未改变数据内容。其余三张图均由 arXiv 源包中的完整矢量 PDF 白底高分辨率渲染。四张图均已本地预览复核，没有使用 PDF 内嵌碎片。
- 论文正文将 LeJEPA 展开为 **Latent-Euclidean JEPA**，当前仓库 README 使用 **Lean Joint-Embedding Predictive Architecture**；笔记保留这一命名差异，不擅自判定哪一版是唯一正式全称。
