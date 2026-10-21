# Sources

## 官方入口

- arXiv 页面：https://arxiv.org/abs/2603.03975
- 官方 PDF：https://arxiv.org/pdf/2603.03975
- arXiv TeX source：https://arxiv.org/e-print/2603.03975
- 官方项目页：https://aka.ms/Phi-4-reasoning-vision
- 官方 Hugging Face：https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B
- 官方代码：https://github.com/microsoft/Phi-4-reasoning-vision-15B
- arXiv DOI：https://doi.org/10.48550/arXiv.2603.03975
- 获取日期：2026-10-16

## 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | arXiv PDF | 全文 | 架构、数据、训练、评测与局限复核 |
| `figures/timing_and_tokens.png` | 官方完整原图 | arXiv TeX `blog/timing_and_tokens.png` | Figure 2 | accuracy–time/token Pareto |
| `figures/arch.png` | 官方完整原图 | arXiv TeX `blog/arch.png` | Figure 3 | mid-fusion 架构 |
| `figures/phi4mm_data.png` | 官方完整原图 | arXiv TeX `blog/phi4mm_data.png` | Figure 4 | 数据组成与 mixed reasoning 样例 |
| `figures/everyday.png` | 官方完整原图 | arXiv TeX `blog/everyday.png` | Figure 1 | 通用能力示例 |

四张图均直接复制自官方 arXiv source，未裁切、重绘或使用 PDF 内嵌碎片；已逐张本地读取，确认完整。

## 版本与口径说明

- 官方题名为 `Phi-4-reasoning-vision-15B Technical Report`；Vault 文件名按已注册中文题目保留。
- arXiv source 的 `architecture.tex` 含一整段被 `%` 注释的 bidirectional-attention 实验，最终 PDF 正文不包含该方法。精读笔记明确不把它当作发布模型配置。
- 分辨率与数据比例消融主要使用 5B proxy；笔记没有把这些数值直接写成最终 15B 的逐项结果。
