# Sources — OmniForcing

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | https://arxiv.org/pdf/2603.11647v2 | 全文（18 页） | 2026-08-29 | 精读主来源 |
| `figures/01-ttfc-teaser.png` | Teaser（TTFC 对比） | arXiv 源码 2603.11647 / teaser_2.pdf | Figure 1, p.2 | 2026-08-29 | 第 1 章"研究背景" |
| `figures/02-distillation-pipeline.png` | 三阶段蒸馏 pipeline | arXiv 源码 2603.11647 / method.pdf | Figure 2, p.6 | 2026-08-29 | 第 3.1 节"总体框架" |
| `figures/03-block-causal-mask.png` | 块因果掩码示意 | arXiv 源码 2603.11647 / fig3.pdf | Figure 3, p.7 | 2026-08-29 | 第 3.2 节"块因果对齐" |
| `figures/04-qualitative.png` | 定性对比（波形+频谱） | arXiv 源码 2603.11647 / fig4_1.pdf | Figure 4, p.12 | 2026-08-29 | 第 5.3 节"定性结果" |

## 补充说明

- 论文未提供 Supplementary Material PDF；
- 截至 2026-08-29 无官方代码仓库，仅项目页 https://omniforcing.com；
- 作者来自京东探索研究院（JD Explore Academy）、复旦、北大、港大，是因果视频蒸馏路线（CausVid / Self-Forcing）向多模态联合生成的首个延伸工作；
- 教师模型为 LTX-2（arXiv:2601.03233，14B 视频流 + 5B 音频流的非对称双流架构）。
