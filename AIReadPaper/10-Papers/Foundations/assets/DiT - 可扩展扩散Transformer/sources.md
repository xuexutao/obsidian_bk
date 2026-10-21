# Sources

## 论文来源

- 论文：William Peebles, Saining Xie. *Scalable Diffusion Models with Transformers*.
- arXiv：2212.09748，当前归档版本为 v2（2023-03-02）。
- 摘要页：https://arxiv.org/abs/2212.09748
- 官方 PDF：https://arxiv.org/pdf/2212.09748
- 官方 TeX 源码包：https://arxiv.org/src/2212.09748
- 获取日期：2026-10-16

> arXiv 不为源码包内的单个图文件提供独立公开 URL。下表中的官方原图均来自同一 TeX 源码包，并以源码包内成员路径标识其原始来源。

## 原始附件

| 本地文件 | 原始 URL / 源码包成员 | 原论文 Figure / 用途 | SHA-256 |
|---|---|---|---|
| `sources/paper.pdf` | https://arxiv.org/pdf/2212.09748 | 论文全文；用于精读、正文与图注核对 | `5fbb0ec35e3ae76826240171ca63ed4c27de6cb43180367453019e9d28a8b076` |
| `sources/arxiv-source.tar` | https://arxiv.org/src/2212.09748 | 官方 v2 TeX 源码包；用于取得论文原图及核对图号 | `60c0743901bd007dd43eea7abb169acfc0b356b6cb37f5e82bab543084773c0f` |
| `figures/original/block.pdf` | https://arxiv.org/src/2212.09748 → `figures/block.pdf` | Figure 3：DiT 总体架构及 adaLN-Zero、cross-attention、in-context conditioning 等条件注入方式 | `4c045e3e779a709d4510c473cb0bbb7c7256e1a565c39239cc06a81132dde42b` |
| `figures/original/conditioning.pdf` | https://arxiv.org/src/2212.09748 → `figures/conditioning.pdf` | Figure 5：条件注入消融；比较 in-context、cross-attention、adaLN 与 adaLN-Zero | `a391056addf2553a3880e8006a580d633cee616b33ab2ff6d4bb8bd12151fb26` |
| `figures/original/teaser_new.pdf` | https://arxiv.org/src/2212.09748 → `figures/teaser_new.pdf` | Figure 2：Gflops–FID 总览；展示 DiT 扩展趋势及与 ADM、LDM 的比较 | `15ff6d5f0fa3e377e699a6e41c2e56cb697accfda46830be05ab0e00c6f01ba4` |
| `figures/original/scaling_new.pdf` | https://arxiv.org/src/2212.09748 → `figures/scaling_new.pdf` | Figure 6：12 个 DiT 变体的 FID-50K 训练曲线；比较模型规模与 patch size 的扩展效果 | `ef0641cc2aa18cf81e627050324aebe878ed07f05a5ebc7bd2858b9c35662c99` |
| `figures/original/visual_scaling_new.pdf` | https://arxiv.org/src/2212.09748 → `figures/visual_scaling_new.pdf` | Figure 7：固定输入噪声与类别标签下的视觉扩展对比，展示增加模型规模或 token 数带来的保真度提升 | `611670a185fdb4205cf7834b0256636ac3b1f1c111060d5a5752d2d16dd35f43` |

上述五个 `figures/original/*.pdf` 的 SHA-256 与 `arxiv-source.tar` 中对应成员逐项一致。

## 精读用 PNG

以下 PNG 由对应官方 PDF 原图栅格化，用于 Obsidian 精读展示；其原始 URL 均为官方 TeX 源码包 https://arxiv.org/src/2212.09748。获取日期均为 2026-10-16。

| 本地文件 | 官方源文件 | 原论文 Figure / 用途 | 图像尺寸 | SHA-256 |
|---|---|---|---:|---|
| `figures/01-architecture-conditioning.png` | `figures/block.pdf` | Figure 3：解释 DiT 数据流、Block 结构及条件注入设计 | 6624 × 2736 | `6441e2cd3fdff30eba5672bcf0b9b5ef8f2fbd4d86a4739fb7c4b220978ad4fb` |
| `figures/02-conditioning-ablation.png` | `figures/conditioning.pdf` | Figure 5：说明 adaLN-Zero 在各训练阶段优于其余条件注入策略 | 1844 × 1383 | `2043b600b23bbfdaa26e26d1dde14752a606ebff5b70320ac006ea85970ddb73` |
| `figures/03-gflops-fid.png` | `figures/teaser_new.pdf` | Figure 2：概括“更高 Transformer Gflops 对应更低 FID”的核心结论 | 3456 × 1352 | `a08876cea0d5989d4867194eb819d4a90ac4b4d63369f9b87d8d5aa9e6b17531` |
| `figures/04-scaling-curves.png` | `figures/scaling_new.pdf` | Figure 6：展示扩大深度/宽度或增加输入 token 均持续改善 FID | 3840 × 1440 | `9cd0913dfee623363333d114f0feac3a82d74c4e10ea5d7eb05b5e16e6b0e7fe` |
| `figures/05-visual-scaling.png` | `figures/visual_scaling_new.pdf` | Figure 7：直观比较 12 个 DiT 模型随规模和 token 数增加而提升的样本质量 | 1634 × 2038 | `df8397cafb65692971fdb601ab81de2e080d8c4eadbfe61da34f0becad48c6a0` |

## 核验记录

- 2026-10-16 已人工逐张打开核验上述五张 PNG。
- 五张 PNG 均内容完整、边界无异常裁切、文字与曲线清晰可辨，无空白图、损坏或明显压缩伪影。
- 图号与用途已依据官方 arXiv v2 正文及图注核对。
- 完整性提示：本地 `arxiv-source.tar` 可读取并提取本精读使用的五个官方原图成员，但全量归档列表检查报告 `Truncated input file`；因此此处仅确认已使用材料，不声明该源码包包含官方归档的全部成员。
