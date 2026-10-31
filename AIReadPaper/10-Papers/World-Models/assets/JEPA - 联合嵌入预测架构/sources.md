# 来源记录 — I-JEPA

## 官方论文与项目

| 本地文件 | 材料类型 | 原始来源 | 位置 / 版本 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2301.08243 | 全文（含附录） | 2026-10-31 | 精读主来源 |
| `figures/01-architectures.png` | 论文关键图 | https://arxiv.org/abs/2301.08243 | Figure 2, p.2 | 2026-08-27 | JEA / Generative / JEPA 对比 |
| `figures/02-method.png` | 方法 pipeline | https://arxiv.org/abs/2301.08243 | Figure 3, p.3 | 2026-08-27 | context/target/predictor 数据流 |
| `figures/03-visualization.png` | 定性结果 | https://arxiv.org/abs/2301.08243 | Figure 6, p.8 | 2026-08-27 | predictor 表征可视化 |
| `sources/in1k_vith14_ep300.yaml` | 官方训练配置 | https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml | commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048` | 2026-10-31 | ViT-H/14、mask、优化器与 batch 超参 |
| `sources/code-version.txt` | 代码版本记录 | https://github.com/facebookresearch/ijepa | commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048` | 2026-10-31 | 固定代码阅读版本 |

## 代码审计范围

- 仓库：https://github.com/facebookresearch/ijepa
- Commit：`52c1ae95d05f743e000e8f10a1f3a79b10cff048`
- Commit date：2023-06-13
- 阅读日期：2026-10-31

人工回读并核对的主要文件：

- `main.py`、`main_distributed.py`
- `configs/in1k_vith14_ep300.yaml`
- `src/train.py`
- `src/helper.py`
- `src/transforms.py`
- `src/datasets/imagenet1k.py`
- `src/masks/multiblock.py`
- `src/masks/utils.py`
- `src/models/vision_transformer.py`
- `src/utils/tensors.py`
- `src/utils/schedulers.py`
- `src/utils/distributed.py`

## 关键 shape 与结论的溯源

| 结论 | 一手代码位置 | 人工复核 |
|---|---|---|
| ViT-H/14@224 为 256 patch、width 1280、depth 32 | `configs/in1k_vith14_ep300.yaml`; `src/models/vision_transformer.py` | 已复核 |
| target/context masks 为 4/1；target scale 0.15–0.2 | 配置文件；`src/masks/multiblock.py` | 已复核 |
| context 按 local-batch 最短长度截断 | `src/masks/multiblock.py:137-168` | 已复核 |
| predictor 将 target 数展开到 batch 维 | `src/models/vision_transformer.py:285-326` | 已复核 |
| predictor shape 为 `[4B,N_c+N_t,384]→[4B,N_t,1280]` | 同上 | 已复核 |
| 论文写 L2、代码用 Smooth L1 | `src/train.py:295-319` | 已复核 |
| optimizer step 后做 target EMA | `src/train.py:321-336` | 已复核 |
| mask 位置采样存在缺少 `+1` 的边界行为 | `src/masks/multiblock.py:88-92` | 已复核 |

## 开源覆盖范围

官方仓库包含 ImageNet 预训练主链路，但未提供论文完整的 linear probe、1% fine-tune、迁移评测与 RCDM 可视化 decoder 代码。笔记中的下游指标依据论文 PDF；代码级 shape 依据上述 commit。未运行完整 ViT-H/14 预训练，所有 tensor 数字由源码与官方配置静态推导，并对关键 reshape/cat/gather 逐行人工复核。