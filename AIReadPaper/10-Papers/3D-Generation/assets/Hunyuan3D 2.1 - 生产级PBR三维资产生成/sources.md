# Hunyuan3D 2.1 材料来源

- 论文：https://arxiv.org/abs/2506.15442
- 官方仓库：https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- 代码 commit：`82920d643c0dc2f7bfd7255f45f62d386edfe60c`（2025-10-17）
- HF revision：`0b94677654c57bb9a6b6845cd7b704ccf551d327`
- 阅读日期：2026-10-30
- 本地论文：`sources/paper.pdf`
- 许可证：`sources/LICENSE`

## 配置

- `sources/shape-config.yaml`：官方 `hunyuan3d-dit-v2-1/config.yaml`
- `sources/paint-model-index.json`：官方 `hunyuan3d-paintpbr-v2-1/model_index.json`
- `sources/paint-unet-config.json`：官方 Paint UNet 配置

## 重点源码

- `hy3dshape/hy3dshape/pipelines.py`
- `hy3dshape/hy3dshape/models/denoisers/hunyuandit.py`
- `hy3dshape/hy3dshape/models/autoencoders/model.py`
- `hy3dshape/hy3dshape/models/diffusion/flow_matching_sit.py`
- `hy3dpaint/textureGenPipeline.py`
- `hy3dpaint/hunyuanpaintpbr/pipeline.py`
- `hy3dpaint/hunyuanpaintpbr/unet/modules.py`
- `hy3dpaint/hunyuanpaintpbr/unet/model.py`
- `hy3dpaint/DifferentiableRenderer/MeshRender.py`

## 图片

均从官方 PDF 以 300 DPI 裁剪：

- `figures/01-data-pipeline.png`：Algorithm 1 / p.4
- `figures/02-shape-pipeline.png`：Figure 2 / p.5
- `figures/03-dit-block.png`：Figure 3 / p.5
- `figures/04-pbr-paint.png`：Figure 4 / p.6
- `figures/05-shape-results.png`：Figure 5 / p.8
- `figures/06-pbr-results.png`：Figure 6 / p.9

## 核验边界

- 论文事实由 PDF 与抽取文本交叉核验。
- 张量和默认值由源码、HF 配置交叉核验，未实际运行完整推理。
- 官方仓库开放 Shape DiT 微调与 Paint 训练主体，但缺 ShapeVAE 从头训练、正式大规模数据和可原样执行的完整训练链；Paint 默认 YAML target 在该 commit 中指向不存在模块。
