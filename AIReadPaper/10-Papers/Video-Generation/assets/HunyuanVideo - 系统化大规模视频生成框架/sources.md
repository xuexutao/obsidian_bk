# Sources

获取日期统一为 **2026-11-05**。主论文结论以 arXiv:2412.03603 v6 为准；HunyuanVideo 1.5 仅用于版本演进对照，不回写初代结论。

| 本地文件 / 资源 | 材料类型 | 原始来源 | 论文位置 / 固定版本 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方主论文 PDF | <https://arxiv.org/pdf/2412.03603v6> | arXiv:2412.03603 v6，全文 | 2026-11-05 | 初代 HunyuanVideo 精读主来源 |
| `sources/hunyuanvideo-1.5.pdf` | 官方 1.5 技术报告 PDF | <https://arxiv.org/pdf/2511.18870> | arXiv:2511.18870 v2，全文 | 2026-11-05 | 版本演进对照；不混入初代结果 |
| `figures/01-overall-architecture.png` | 官方 arXiv HTML 原图 | <https://arxiv.org/html/2412.03603v6/extracted/6270132/figures/hunyuanvideo_overview.png> | Figure 5，PDF 文件第 5 页（印刷页 5） | 2026-11-05 | 初代端到端总体架构 |
| `figures/02-causal-3d-vae.png` | 官方 arXiv HTML 原图 | <https://arxiv.org/html/2412.03603v6/x3.png> | Figure 6，PDF 文件第 6 页（印刷页 6） | 2026-11-05 | Causal 3D VAE 压缩路径 |
| `figures/03-diffusion-backbone.png` | 官方 arXiv HTML 原图 | <https://arxiv.org/html/2412.03603v6/x5.png> | Figure 8，PDF 文件第 7 页（印刷页 7） | 2026-11-05 | dual-stream / single-stream MMDiT 与文本条件 |
| `figures/04-scaling-laws.png` | 官方 PDF 300 DPI 整页渲染后完整裁剪 | <https://arxiv.org/pdf/2412.03603v6> | Figure 10，PDF 文件第 9 页（印刷页 9）；六个子图及完整图注 | 2026-11-05 | 图像→视频 scaling law 与 13B 选型 |
| HunyuanVideo 官方仓库 | 官方代码 | <https://github.com/Tencent-Hunyuan/HunyuanVideo> | commit [`4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e`](https://github.com/Tencent-Hunyuan/HunyuanVideo/tree/4eda9cb44c3be82a6b9c4261897cb6bd3cfb084e) | 2026-11-05 | 初代推理链、模型结构、默认参数、显存要求；旧链接 <https://github.com/Tencent/HunyuanVideo> 会指向同一项目 |
| HunyuanVideo 官方模型卡与权重 | 官方 Hugging Face 模型卡 | <https://huggingface.co/tencent/HunyuanVideo> | revision [`6204ad6aea1a77ff5aba337c88278bb9500eb37d`](https://huggingface.co/tencent/HunyuanVideo/tree/6204ad6aea1a77ff5aba337c88278bb9500eb37d) | 2026-11-05 | 权重目录、VAE config、FP8、默认推理配置 |
| `hunyuan-video-t2v-720p/vae/config.json` | 官方 VAE 配置 | <https://huggingface.co/tencent/HunyuanVideo/blob/6204ad6aea1a77ff5aba337c88278bb9500eb37d/hunyuan-video-t2v-720p/vae/config.json> | latent 16；block 128/256/512/512；time ratio 4；scaling 0.476986 | 2026-11-05 | 代码级张量与 VAE 超参核对 |
| HunyuanVideo 官方项目页 | 官方项目/演示页 | <https://hunyuan.tencent.com/video/en> | HunyuanVideo 系列项目页 | 2026-11-05 | 系列发布时间与官方定位 |
| HunyuanVideo-I2V | 官方下游仓库 | <https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V> | 2025-03-06 发布 | 2026-11-05 | 基础模型到 I2V 的 token replace 与语义图像 token 增量 |
| HunyuanCustom | 官方下游仓库 | <https://github.com/Tencent-Hunyuan/HunyuanCustom> | 2025-05-08 发布；arXiv:2505.04512 | 2026-11-05 | 图像/文本/音频/视频条件与身份保持扩展 |
| HunyuanVideo-Avatar | 官方下游仓库 | <https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar> | 2025-05-28 发布；arXiv:2505.20156 | 2026-11-05 | 音频驱动、多角色与情绪控制扩展 |
| HunyuanVideo 1.5 官方仓库 | 官方后继代码 | <https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5> | commit [`60783e704160023913bee78f0b47036d393d4dfa`](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/tree/60783e704160023913bee78f0b47036d393d4dfa) | 2026-11-05 | 1.5 独立版本边界、SSTA、8.3B、统一 T2V/I2V；不作为初代代码依据 |
| HunyuanVideo 1.5 模型卡 | 官方 Hugging Face 模型卡 | <https://huggingface.co/tencent/HunyuanVideo-1.5> | HunyuanVideo 1.5 | 2026-11-05 | 1.5 训练/推理能力与消费级部署信息 |

## 代码核对范围

初代仓库只读核对的主要入口：

- `sample_video.py`
- `hyvideo/config.py`
- `hyvideo/constants.py`
- `hyvideo/inference.py`
- `hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py`
- `hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py`
- `hyvideo/modules/models.py`
- `hyvideo/modules/token_refiner.py`
- `hyvideo/modules/posemb_layers.py`
- `hyvideo/vae/autoencoder_kl_causal_3d.py`
- `hyvideo/vae/vae.py`
- `hyvideo/text_encoder/__init__.py`
- `ckpts/README.md`

说明：初代官方仓库公开的是模型定义、权重与推理/采样代码，并未公开论文所述互联网规模基座预训练的数据加载器、完整训练循环、过滤阈值或 AngelPTM 训练实现。因此，本文把“论文训练系统”和“可运行公开代码”严格分开。