# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 / 版本 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2508.02324 | Qwen-Image Technical Report 全文 | 2026-11-06 | 精读主来源 |
| `figures/01-vae-text-reconstruction.png` | 论文关键图 | https://arxiv.org/abs/2508.02324 | Figure 17，PDF 第 26 页 | 2026-11-06 | 核对不同 VAE 对小字与文档纹理的重建能力 |
| `figures/02-architecture.png` | 论文方法图 | https://arxiv.org/abs/2508.02324 | Figure 6，PDF 第 7 页 | 2026-11-06 | Qwen2.5-VL + VAE + 双流 MMDiT 总体架构 |
| `figures/03-msrope.png` | 论文方法图 | https://arxiv.org/abs/2508.02324 | Figure 8，PDF 第 9 页 | 2026-11-06 | Naive / Scaling RoPE / MSRoPE 对比 |
| `figures/04-editing-dual-condition.png` | 论文方法图 | https://arxiv.org/abs/2508.02324 | Figure 14，PDF 第 18 页 | 2026-11-06 | 图像编辑的语义、VAE 双条件与 frame 位置编码 |
| — | arXiv 源文件 | https://arxiv.org/e-print/2508.02324 | TeX、表格和原始矢量 Figure | 2026-11-06 | 核对公式、表格、图号和图注；未复制整个源码包到 Vault |
| — | 官方项目仓库 | https://github.com/QwenLM/Qwen-Image | commit `6b5e1f5cec987d404be5ac6657db3b9aacb56a89` | 2026-11-06 | 版本演进、官方推理示例、Qwen-Image-Edit-2509 发布说明 |
| — | 基础模型卡 | https://huggingface.co/Qwen/Qwen-Image | `model_index.json`、`transformer/config.json`、`vae/config.json` | 2026-11-06 | 核对流水线组件、MMDiT 与 VAE 配置 |
| — | 编辑模型卡 | https://huggingface.co/Qwen/Qwen-Image-Edit-2509 | `QwenImageEditPlusPipeline`，Diffusers 0.36.0.dev0 配置 | 2026-11-06 | 补充 2509 多图输入和实际编辑流水线 |
| — | Diffusers 实现 | https://github.com/huggingface/diffusers | HEAD `d3d8af89e5b08a352cf47bba3a451c5fed9a49b1` | 2026-11-06 | 代码级张量流：pipeline、MMDiT、Qwen-Image VAE |
| — | Qwen-Image-Edit 官方博客 | https://qwenlm.github.io/blog/qwen-image-edit | 2025-08-19 | 2026-11-06 | 解释语义编辑与外观编辑双条件设计 |
| — | Qwen-Image-Edit-2509 官方说明 | https://github.com/QwenLM/Qwen-Image/blob/main/Qwen-Image-Edit-2509.md | 2025-09-22 | 2026-11-06 | 多图编辑、一致性增强和控制条件说明 |

## 取图说明

4 张本地图均由 arXiv 官方源码包中的完整 Figure PDF 渲染为 PNG，并经图像预览检查完整性；未使用 PDF 内嵌碎片。正文图片路径必须以本笔记所在目录为基准写成 `assets/Qwen-Image%20-%20文字渲染与双条件图像编辑/figures/...`。

## 范围说明

- 论文详细解释的是 2025 年 8 月 Qwen-Image / 初代统一编辑模型；`Qwen-Image-Edit-2509` 没有独立技术报告。
- 2509 的多图编辑、人物/商品/文字一致性增强与“原生支持 ControlNet 条件”仅有官方发布说明和开源推理流水线，没有公开训练数据规模、损失权重或受控消融。
- 官方 Qwen-Image 仓库主要提供调用示例而非训练代码；模型核心推理实现位于 Hugging Face Diffusers。