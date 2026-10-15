# 来源与图表核验：Attention Is All You Need

## 1. 官方来源

- 论文：Vaswani et al., *Attention Is All You Need*
- arXiv ID：`1706.03762`
- 官方摘要页：https://arxiv.org/abs/1706.03762
- 官方 PDF：https://arxiv.org/pdf/1706.03762
- 会议：NeurIPS 2017（原名 NIPS 2017）
- 本次取材时间：2026-10-15
- 取材版本：官方 PDF 标注 `arXiv:1706.03762v7 [cs.CL] 2 Aug 2023`
- PDF 页数：15
- 抓取状态：成功；正文约 40,796 字符；无抓取警告，无需网页降级取材。

## 2. 离线文件

| 文件 | 说明 |
|---|---|
| `sources/paper.pdf` | 官方 arXiv PDF，离线溯源主文件 |
| `sources/source.md` | 从官方 PDF 提取的文本流，含官方标题、作者与摘要 |
| `sources/meta.json` | 抓取元数据、作者、摘要、页面渲染清单 |
| `sources/images/pages/page_001.png`–`page_015.png` | 官方 PDF 各页 200 DPI 整页渲染，用于定位与复核 |
| `sources/images/img_000_p3.png`–`img_002_p4.png` | PDF 内嵌位图碎片，仅保留作取材记录，未当作完整原图引用 |

未清理或删除任何个人目录中的现有文件；原始页面渲染与抽取材料全部保留。

## 3. PDF 与关键图表 SHA-256

```text
bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697  sources/paper.pdf
87926e957bded5c89724170a001fe76f4423ea160db266f077a9152d882f95cc  figures/fig01_transformer_architecture.png
5249a3cf7f2b703227197945734f7d206becaf382678935eae5041c0d058d1e1  figures/fig02_attention_modules.png
446280f0134ac50598f6219755038ff15edac78f496ea228d2880b65df663d55  figures/table01_complexity.png
8cebd2beba08964e33390c76079d93cbbf3cf704b895d3a7dedf66bfc332dd3c  figures/table02_translation_results.png
e5f07752f289cac4fe06e7903ddad2782cd9102ef3954c5540b30f13a3a5d20e  figures/table03_ablation.png
1314236573d347c4bbdbce21e0eba6d4a162df23ae089a2af5ac50bbec383135  figures/fig03_attention_visualization.png
```

## 4. 图表提取记录

所有关键图表都从 `sources/paper.pdf` 的整页内容直接按 PDF point 坐标裁剪，输出 300 DPI PNG；坐标格式为 `(left, top, right, bottom)`，页码按 PDF 阅读器的 1-based 页码记录。

| 输出文件 | 原文对象 | 页码 | 裁剪坐标 | 像素尺寸 | 选取理由 |
|---|---|---:|---|---:|---|
| `figures/fig01_transformer_architecture.png` | Figure 1 | 3 | `(155, 45, 457, 425)` | 1260×1584 | 完整 encoder–decoder 主架构 |
| `figures/fig02_attention_modules.png` | Figure 2 | 4 | `(105, 55, 507, 305)` | 1676×1042 | scaled dot-product 与 multi-head 计算流程 |
| `figures/table01_complexity.png` | Table 1 | 6 | `(95, 55, 517, 205)` | 1760×626 | 复杂度、串行操作数、最大路径比较 |
| `figures/table02_translation_results.png` | Table 2 | 8 | `(95, 55, 517, 260)` | 1760×855 | WMT BLEU 与估算训练成本 |
| `figures/table03_ablation.png` | Table 3 | 9 | `(95, 55, 517, 410)` | 1760×1480 | 头数、维度、深宽度、正则化、位置编码变体 |
| `figures/fig03_attention_visualization.png` | Figure 3 | 13 | `(95, 50, 517, 365)` | 1760×1313 | 长距离依赖 attention 定性可视化 |

## 5. 逐张人工核验

1. **Figure 1**：encoder、decoder、三类 attention、FFN、Add & Norm、位置编码、shifted output、Linear/Softmax 和原始图注均完整；无边缘截断。
2. **Figure 2**：左侧 Scale/Mask/Softmax/MatMul 和右侧多头投影/Concat/Linear 均完整；$Q/K/V$ 标签和原始图注清晰。
3. **Table 1**：四种 layer type、三项比较指标、变量解释和全部复杂度均完整可读。
4. **Table 2**：所有基线、Transformer base/big、EN–DE/EN–FR BLEU 和 Training Cost 列均完整可读。
5. **Table 3**：base、(A)–(E)、big 行以及 PPL/BLEU/params 等表头完整，未截断任何实验组。
6. **Figure 3**：标题、两行 token、making 到远距离词的彩色连线、权重色块和原始图注完整；颜色可辨。

核验结论：六张图表均清晰、完整、可离线阅读，满足“至少三张关键原文图/表”的要求，并覆盖架构、注意力模块、复杂度、翻译结果、消融和注意力可视化。

## 6. 原文信息核对备注

- 摘要和 Table 2 报告 WMT 2014 EN–FR big 为 **41.8 BLEU**；Section 6.1 正文写 **41.0 BLEU**。精读笔记保留并明确标注此冲突，未擅自统一。
- 原始 Transformer 使用 **Post-LN**：`LayerNorm(x + Sublayer(x))`。Pre-LN、RMSNorm 等被放在后续变体部分，不作为原论文主张。
- 原文正弦位置编码的长长度外推是设计动机；论文未给出系统长上下文外推实验。
- 原文 attention 可视化属于定性案例，不能单独证明 attention weight 是完整因果解释。
- 后续 encoder-only/decoder-only、RoPE、relative bias、FlashAttention、MQA/GQA 等均在笔记中明确标为后世发展。

## 7. 主笔记

- [[../../Transformer - 自注意力与序列建模|Transformer - 自注意力与序列建模]]
- 固定路径：`/Users/eilonxtxue/Documents/obsidian/obsidian_bk/AIReadPaper/10-Papers/Foundations/Transformer - 自注意力与序列建模.md`
