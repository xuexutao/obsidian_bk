# Hunyuan3D 2.0 材料来源

## 1. 论文与项目

- 论文：Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation
- arXiv：https://arxiv.org/abs/2501.12202
- 官方项目：https://github.com/Tencent-Hunyuan/Hunyuan3D-2
- 官方模型：https://huggingface.co/tencent/Hunyuan3D-2
- 阅读日期：2026-10-30
- 本地论文：`sources/paper.pdf`
- 本地抽取文本：`sources/paper-text.md`

## 2. 官方代码

- 仓库：https://github.com/Tencent-Hunyuan/Hunyuan3D-2
- 检查 commit：`f8db63096c8282cb27354314d896feba5ba6ff8a`
- commit 时间：2025-10-28 17:58:12 +0800
- commit 标题：`Update LICENSE`
- 开源范围：推理代码、预训练权重、Gradio/API/Blender 示例；未发现完整训练循环与训练数据。
- 许可证副本：`sources/LICENSE`

重点人工核验文件：

- `README.md`
- `NOTICE`
- `LICENSE`
- `minimal_demo.py`
- `gradio_app.py`
- `api_server.py`
- `hy3dgen/shapegen/pipelines.py`
- `hy3dgen/shapegen/preprocessors.py`
- `hy3dgen/shapegen/schedulers.py`
- `hy3dgen/shapegen/models/conditioner.py`
- `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`
- `hy3dgen/shapegen/models/autoencoders/model.py`
- `hy3dgen/shapegen/models/autoencoders/attention_blocks.py`
- `hy3dgen/shapegen/models/autoencoders/volume_decoders.py`
- `hy3dgen/shapegen/models/autoencoders/surface_extractors.py`
- `hy3dgen/texgen/pipelines.py`
- `hy3dgen/texgen/utils/dehighlight_utils.py`
- `hy3dgen/texgen/utils/multiview_utils.py`
- `hy3dgen/texgen/hunyuanpaint/pipeline.py`
- `hy3dgen/texgen/hunyuanpaint/unet/modules.py`
- `hy3dgen/texgen/differentiable_renderer/mesh_render.py`
- `hy3dgen/texgen/utils/uv_warp_utils.py`

## 3. 官方配置

| 本地文件 | 原始 URL | 用途 |
|---|---|---|
| `sources/shape-config.yaml` | https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-dit-v2-0/config.yaml | Shape DiT、DINOv2 与 ShapeVAE 发布配置 |
| `sources/paint-model-index.json` | https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-paint-v2-0-turbo/model_index.json | Turbo Paint 组件索引 |
| `sources/paint-unet-config.json` | https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-paint-v2-0-turbo/unet/config.json | Paint UNet 配置 |
| `sources/paint-vae-config.json` | https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-paint-v2-0-turbo/vae/config.json | Paint VAE 配置 |
| `sources/delight-model-index.json` | https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-delight-v2-0/model_index.json | Delight 组件索引 |

说明：Hugging Face `main` 没有与上述 Git commit 绑定，代码与模型仓库应视为两个独立版本面。

## 4. 图片来源

所有图片均从官方 arXiv PDF 以 300 DPI 完整裁剪，不使用 PDF 内嵌碎片：

| 文件 | 来源 | 用途 |
|---|---|---|
| `figures/01-overall-pipeline.png` | Figure 2 / PDF p.4 | Shape→Texture 总体两阶段管线 |
| `figures/02-shape-vae.png` | Figure 3 / PDF p.5 | ShapeVAE 编解码结构 |
| `figures/03-shape-dit.png` | Figure 4 / PDF p.6 | Double/Single Stream DiT |
| `figures/04-paint-architecture.png` | Figure 5 / PDF p.7 | Paint 双流与多任务注意力 |
| `figures/05-shape-comparison.png` | Figure 7 / PDF p.12 | 几何生成定性对比 |
| `figures/06-texture-comparison.png` | Figure 8 / PDF p.14 | 纹理生成定性对比 |

## 5. 核验说明

- 论文方法、公式与实验数字来自 PDF 正文和表格。
- 代码张量形状由公开推理代码、官方配置和 checkpoint header 交叉核验。
- Shape DiT、DINOv2、ShapeVAE 与 Paint/Delight 参数量来自官方权重文件结构统计；笔记中区分“模型本体参数量”和“完整下载 checkpoint 参数量”。
- 未运行完整推理，仅做静态源码审查；Turbo Paint 几何 mask 是否实际进入 UNet 等版本敏感问题已标为“待运行复核”。
- 论文未披露的数据规模、Shape 训练超参、GPU 数量、训练时长等不作推测。
