# 溯源记录：多模态理解模型发展史：从 CLIP 到 Qwen3-VL 的七个技术节点

## 主来源

- 标题：多模态理解模型发展史：从 CLIP 到 Qwen3-VL 的七个技术节点
- 来源：微信公众号「深度学习之凡人修仙」，作者「飞星未来」
- 发布时间：2026-08-30 09:44（广东）
- URL：https://mp.weixin.qq.com/s/QL5OprQk5Cxx93BfPR_QHg
- 获取日期：2026-10-22
- 获取方式：WebFetch 提取正文全文 + curl 抓取原始 HTML（`--user-agent` 模拟浏览器，`-e` 带 referer 用于图片下载）

## 附件清单

| 文件 | Figure / 位置 | 原始 URL | 获取日期 | 用途 |
|---|---|---|---|---|
| `figures/01-qwen3vl-native-multimodal-arch.png` | 原文开篇配图：千问原生多模态架构设计图 | https://mmbiz.qpic.cn/mmbiz_png/wFXibxtu4eL6Koug9YEibCoM53PyvZl5S9LSP1AdPsDUalRfRBiaeZMdY9ltuLpC67utV7cPuRr6zpNH3I1SH9cQx0qUNibGPFLYcibFgnFgdEGg/640?wx_fmt=png&from=appmsg | 2026-10-22 | 正文「基本信息」章节配图，说明原生多模态架构的整体形态 |

## 需要二次核对的口径

原文为科普长文，以下数字与型号引用前建议以官方技术报告 / 发布页为准：

1. Qwen2-VL 视觉 token 数：224×224 → 66 tokens，单图上限 16384（实际由 `min_pixels` / `max_pixels` 控制）
2. InternVL3 原生多模态预训练配比：约 50B 纯文本 token + 约 150B 图文/视频/交错数据（3:1）
3. Qwen3-VL-235B-A22B（MoE，激活 22B）、原生 256K 上下文可扩 1M、OCR 32 种语言
4. Gemini 3 发布时间（原文记为 2025.11）、GPT-5 发布时间（原文记为 2025.08）
