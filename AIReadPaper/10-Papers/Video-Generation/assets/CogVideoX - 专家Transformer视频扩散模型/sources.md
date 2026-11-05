# Sources

获取日期统一为 **2026-11-05**。主论文以 arXiv v3 为准；后续 CogVideoX1.5 信息只引用官方仓库发布说明与官方模型配置，不作为主论文消融证据。

| 本地文件 / 外部材料 | 材料类型 | 原始来源 | 论文位置 / 固定版本 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | <https://arxiv.org/pdf/2408.06072v3> | 全文；SHA-256 `699795d3cfdf7acef250e0e28043b001e31ae7ab101f35fd079370c43f6473f3` | 2026-11-05 | 精读主来源 |
| `figures/01-teaser.jpg` | 官方 arXiv HTML 原图 | <https://arxiv.org/html/2408.06072v3/images/front2.jpg> | Figure 1；论文首页附近 | 2026-11-05 | 长时事件与大运动示例 |
| `figures/02-overall-architecture.png` | 官方 arXiv HTML 原图 | <https://arxiv.org/html/2408.06072v3/images/transformer.png> | Figure 3；§2 The CogVideoX Architecture | 2026-11-05 | 总体架构、Expert AdaLN、3D full attention |
| `figures/03-3d-causal-vae.jpg` | 官方 arXiv HTML 原图 | <https://arxiv.org/html/2408.06072v3/images/3dvae_combined.jpg> | Figure 4；§2.1 3D Causal VAE | 2026-11-05 | VAE 与 temporal context parallel |
| `figures/04-ablation.png` | 官方 arXiv HTML 原图 | <https://arxiv.org/html/2408.06072v3/images/ab-fvd.png> | Figure 8；§4.1 Ablation Study | 2026-11-05 | Expert AdaLN、attention、uniform sampling 消融 |
| arXiv abstract / HTML | 官方论文页面 | <https://arxiv.org/abs/2408.06072v3> / <https://arxiv.org/html/2408.06072v3> | v3，2025-03-26；ICLR 2025 | 2026-11-05 | 元信息、正文交叉核对与官方完整原图 |
| CogVideo 官方仓库 | 官方代码 | <https://github.com/zai-org/CogVideo/tree/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2> | commit `7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2` | 2026-11-05 | 推理、微调、权重转换和发布演进 |
| `inference/cli_demo.py` | 官方推理入口 | <https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/cli_demo.py> | 同上 commit | 2026-11-05 | Pipeline、scheduler、CFG、分辨率、帧数、offload |
| `finetune/models/cogvideox_t2v/lora_trainer.py` | 官方 T2V 微调实现 | <https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/finetune/models/cogvideox_t2v/lora_trainer.py> | 同上 commit | 2026-11-05 | VAE/T5 编码、加噪、v-prediction、RoPE、loss |
| `finetune/models/cogvideox_i2v/lora_trainer.py` | 官方 I2V 微调实现 | <https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/finetune/models/cogvideox_i2v/lora_trainer.py> | 同上 commit | 2026-11-05 | 图像加噪、latent 补零、32 通道拼接、offset |
| `tools/convert_weight_sat2hf.py` | 官方 SAT→diffusers 转换 | <https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/tools/convert_weight_sat2hf.py> | 同上 commit | 2026-11-05 | 1.0/1.5 patch、输入通道、权重映射边界 |
| `finetune/README.md` | 官方微调说明 | <https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/finetune/README.md> | 同上 commit | 2026-11-05 | LoRA/SFT 显存测试、数据格式、尺寸约束 |
| CogVideoX-5B 模型卡/config | 官方 Hugging Face 权重 | <https://huggingface.co/zai-org/CogVideoX-5b/tree/8fc5b281006c82b82d34fd2543d2f0ebb4e7e321> | revision `8fc5b281006c82b82d34fd2543d2f0ebb4e7e321` | 2026-11-05 | 5B Transformer/VAE/T5/scheduler 配置与部署数据 |
| CogVideoX1.5-5B 模型卡/config | 官方 Hugging Face 权重 | <https://huggingface.co/zai-org/CogVideoX1.5-5B/tree/fdc5267c90b5c06492985b966e43aae984e189e0> | revision `fdc5267c90b5c06492985b966e43aae984e189e0` | 2026-11-05 | 1.5 temporal patch、默认 shape、VAE scale 与部署数据 |
| diffusers CogVideoX pipeline | 官方 diffusers v0.32.0 源码 | <https://github.com/huggingface/diffusers/blob/v0.32.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py> | tag `v0.32.0` | 2026-11-05 | 1.5 时间 padding、22 帧去噪 latent、decode 前裁回 21 帧 |
| diffusers CogVideoX Transformer | 官方 diffusers v0.32.0 源码 | <https://github.com/huggingface/diffusers/blob/v0.32.0/src/diffusers/models/transformers/cogvideox_transformer_3d.py> | tag `v0.32.0` | 2026-11-05 | temporal unpatchify 与输出 shape 核对 |
| CogVideo 原论文 | 官方 arXiv 页面 | <https://arxiv.org/abs/2205.15868> | CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers；ICLR 2023 | 2026-11-05 | CogVideo→CogVideoX 演进核对 |
| CogVLM2-Caption 模型卡 | 官方 Hugging Face 模型 | <https://huggingface.co/zai-org/cogvlm2-llama3-caption> | 仓库 README 指向的 caption 模型 | 2026-11-05 | 数据 caption 公开状态与模型来源 |

## 读取与图片说明

- 已按要求先加载 `workbuddy-pdf` skill，并用其 CLI 对 `sources/paper.pdf` 尝试 `md`、`txt`、`structure` 与 300 DPI `png` 渲染；当前本机组件对该 PDF 均以 exit code 139 崩溃，未产出可用文字或整页图。
- 为避免使用任何第三方 PDF 解析库，正文改用同一 v3 的官方 arXiv HTML 核对文本，并直接下载 arXiv HTML 提供的完整高分辨率原图。四张图均经视觉检查，未截边、模块/子图完整，尺寸分别为 2445×1133、2042×2278、1885×939、7446×1651。
- 因指定 PDF 工具未能渲染页面，物理 PDF 页码无法可靠自动核对；正文图片注释保留 Figure 编号、论文章节/页位与官方链接，不编造精确物理页号。
- 仓库当前主分支主要是 diffusers 调用、微调和权重转换代码；SAT 与 diffusers 实现未混写。正文明确区分论文描述、现行 diffusers 配置和 CogVideoX1.5 发布说明。
