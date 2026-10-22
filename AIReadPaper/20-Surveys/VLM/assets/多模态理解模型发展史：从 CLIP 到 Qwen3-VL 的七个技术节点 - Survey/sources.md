# 溯源记录：多模态理解模型发展史：从 CLIP 到 Qwen3-VL 的七个技术节点

## 主来源

- 标题：多模态理解模型发展史：从 CLIP 到 Qwen3-VL 的七个技术节点
- 来源：微信公众号「深度学习之凡人修仙」，作者「飞星未来」
- 发布时间：2026-08-30 09:44（广东）
- URL：https://mp.weixin.qq.com/s/QL5OprQk5Cxx93BfPR_QHg
- 获取日期：2026-10-22
- 获取方式：WebFetch 提取正文全文 + curl 抓取原始 HTML（`--user-agent` 模拟浏览器，`-e` 带 referer 用于图片下载）

## 配图来源（官方论文 PDF 裁剪，非公众号截图）

所有论文图均从 arXiv 官方 PDF 按「页码 + 相对裁剪框」裁剪导出（2.4 倍缩放），提取工具为技能 `research-field-survey/scripts/extract_figure.py`（底层 pymupdf 页面渲染，保留矢量文字标注）。

| 文件 | 原始 Figure / 页码 | 源论文（arXiv ID） | 获取日期 | 用途 |
|---|---|---|---|---|
| `figures/01-qwen3vl-native-multimodal-arch.png` | 公众号原文开篇配图：千问原生多模态架构设计图 | 微信公众号原文 | 2026-10-22 | 「基本信息」章节，说明原生多模态整体形态 |
| `figures/02-clip-contrastive-pretraining.png` | Figure 1, p.2 | CLIP, arXiv:2103.00020 | 2026-10-22 | 节点 2.1，对比预训练 + zero-shot 推理 |
| `figures/03-flamingo-architecture.png` | Figure 3, p.4 | Flamingo, arXiv:2204.14198 | 2026-10-22 | 节点 2.2，冻结骨干 + 可训练桥接总览 |
| `figures/04-flamingo-gated-xattn-dense.png` | Figure 4, p.5 | Flamingo, arXiv:2204.14198 | 2026-10-22 | 节点 2.2，门控交叉注意力内部结构 |
| `figures/05-blip2-qformer-and-objectives.png` | Figure 2, p.3 | BLIP-2, arXiv:2301.12597 | 2026-10-22 | 节点 2.3，Q-Former 架构与三目标掩码 |
| `figures/06-llava-network-architecture.png` | Figure 1, p.4 | LLaVA, arXiv:2304.08485 | 2026-10-22 | 节点 2.3，最简线性投影架构 |
| `figures/07-qwen2vl-mrope.png` | Figure 3, p.5 | Qwen2-VL, arXiv:2409.12191 | 2026-10-22 | 节点 2.5，三维位置 ID 分配 |
| `figures/08-qwen2vl-dynamic-resolution.png` | Figure 2, p.4 | Qwen2-VL, arXiv:2409.12191 | 2026-10-22 | 节点 2.5，动态分辨率效果 |
| `figures/09-internvl15-architecture.png` | Figure 3, p.3 | InternVL 1.5, arXiv:2404.16821 | 2026-10-22 | 节点 2.5，ViT-MLP-LLM 架构 |
| `figures/10-internvl15-dynamic-high-resolution.png` | Figure 4, p.4 | InternVL 1.5, arXiv:2404.16821 | 2026-10-22 | 节点 2.5，动态 tiling + 缩略图 |
| `figures/11-qwen25vl-framework.png` | Figure 1, p.3 | Qwen2.5-VL, arXiv:2502.13923 | 2026-10-22 | 节点 2.6，窗口注意力 + 绝对时间 |
| `figures/12-internvl3-native-pretraining.png` | Figure 3, p.18 | InternVL3, arXiv:2504.10479 | 2026-10-22 | 节点 2.6，原生多模态预训练消融 |
| `figures/13-qwen3vl-framework.png` | Figure 1, p.3 | Qwen3-VL, arXiv:2511.21631 | 2026-10-22 | 节点 2.7，DeepStack 与时间戳 token |

裁剪参数存于会话记录；如需复现，脚本用法为 `extract_figure.py <pdf> <page> <out.png> <x0> <y0> <x1> <y1>`（比例为页面宽高）。

## 需要二次核对的口径

原文为科普长文，以下数字与型号引用前建议以官方技术报告 / 发布页为准：

1. Qwen2-VL 视觉 token 数：224×224 → 66 tokens，单图上限 16384（实际由 `min_pixels` / `max_pixels` 控制）
2. InternVL3 原生多模态预训练配比：约 50B 纯文本 token + 约 150B 图文/视频/交错数据（3:1）
3. Qwen3-VL-235B-A22B（MoE，激活 22B）、原生 256K 上下文可扩 1M、OCR 32 种语言
4. Gemini 3 发布时间（原文记为 2025.11）、GPT-5 发布时间（原文记为 2025.08）
