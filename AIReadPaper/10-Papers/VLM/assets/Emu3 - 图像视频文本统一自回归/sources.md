# Sources

## 1. 官方入口

- arXiv 页面：https://arxiv.org/abs/2409.18869
- 官方 PDF：https://arxiv.org/pdf/2409.18869
- arXiv DOI：https://doi.org/10.48550/arXiv.2409.18869
- 官方项目页：https://emu.baai.ac.cn
- 获取日期：2026-10-16

## 2. 本地材料溯源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | arXiv:2409.18869 | 全文，23 页 | 主精读来源、公式、训练配置和结果复核 |
| `figures/01-next-token-overview.png` | 原文完整总览图 | 官方 PDF 300 DPI 裁剪 | Figure 1，第 1 页 | 图像/视频/文本统一 next-token prediction |
| `figures/02-tokenizer-reconstruction.png` | 原文完整重建图 | 官方 PDF 300 DPI 裁剪 | Figure 3，第 3 页 | MoVQGAN 图像与视频 token 重建质量 |
| `figures/03-video-extension.png` | 原文完整视频延展图 | 官方 PDF 300 DPI 裁剪 | Figure 8，第 11 页 | 上下文视频与未来帧自回归预测 |

## 3. 核验说明

- PDF 由 arXiv 官方 PDF 入口下载。
- 三张图片均从官方 PDF 的整页渲染中按图注坐标裁剪，不使用零散内嵌位图。
- 三张图已于 2026-10-16 逐张用 `Read` 预览核验：图像主体、边缘与图注完整，文字清晰。
- 论文没有完整披露预训练数据总规模、各模态采样比例、总训练算力和常规采样的 temperature/top-k/top-p；主笔记对这些信息明确标为“未披露”，不作推测。