# Sources

## 论文来源

- 论文：Alexey Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*.
- 会议：ICLR 2021。
- arXiv：2010.11929v2，2021-06-03；v1 提交于 2020-10-22。
- 摘要页：https://arxiv.org/abs/2010.11929
- 官方 PDF：https://arxiv.org/pdf/2010.11929
- OpenReview：https://openreview.net/forum?id=YicbFdNTTy
- 获取日期：2026-10-30。

## 本地附件

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 | SHA-256 |
|---|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2010.11929 | 全文 | 2026-10-30 | 精读主来源 | `8ce7b83971a14508ca711a27c875c9b6914c4f6767cf3150fb1ca6c07aa056d6` |
| `figures/01-architecture.png` | 论文图裁剪 | 同上 | Figure 1，PDF p.3 | 2026-10-30 | 总体架构、patchify 与 pre-LN block | `825dfbf911f8777b5dd38bb8b65b6fbe6cb7700774451cc95d1914c27909b99c` |
| `figures/02-data-scaling.png` | 论文图裁剪 | 同上 | Figure 3，PDF p.7 | 2026-10-30 | 预训练数据规模与模型规模关系 | `526f986eb603a74e348399e2343fd89c4c2de686687d1c03d8c6cca8dc0c90cd` |
| `figures/03-compute-scaling.png` | 论文图裁剪 | 同上 | Figure 5，PDF p.7 | 2026-10-30 | 性能—预训练计算量关系 | `84bcfcb24f1b0a53ab044a838257f0405ecdde15e5ee8ca8e773379aa65d4036` |
| `figures/04-learned-representations.png` | 论文图裁剪 | 同上 | Figure 7，PDF p.9 | 2026-10-30 | patch 基、位置拓扑与注意力距离 | `b079ba68f8f4fb3072a0f088cbfb80ae6a431009b2c063215a19320ec65eded4` |

四张图片均从官方 PDF 以 300 DPI 完整裁剪，保留图注；已人工核验边界、图例、坐标轴和文字，没有使用 PDF 内嵌碎片图。

## 官方代码

- 仓库：https://github.com/google-research/vision_transformer
- 固定提交：[`64801f1b3b367b3611cc27a3d45cc22870a36fb3`](https://github.com/google-research/vision_transformer/commit/64801f1b3b367b3611cc27a3d45cc22870a36fb3)
- 提交时间：2026-08-01T21:36:09Z。
- 许可证：Apache-2.0。
- 核验日期：2026-10-30。

本机 `git`/`curl` 的 LibreSSL 与 GitHub 当前 TLS 协商失败，因此没有伪称成功 clone。代码通过 GitHub 官方 commit API、tree API 与 commit 固定的源码页逐文件核验，所有结论均绑定上述 commit。

### 阅读过的主要代码文件

| 文件 | 主要用途 | 固定版本链接 |
|---|---|---|
| `vit_jax/models_vit.py` | Patch embedding、class token、位置编码、Transformer block、分类头 | https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/vit_jax/models_vit.py |
| `vit_jax/configs/models.py` | ViT-B/L/H 结构配置 | https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/vit_jax/configs/models.py |
| `vit_jax/configs/vit.py` | 下游微调配置入口 | https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/vit_jax/configs/vit.py |
| `vit_jax/configs/common.py` | batch、steps、crop、warmup、优化器等通用配置 | https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/vit_jax/configs/common.py |
| `vit_jax/checkpoint.py` | 预训练权重加载、head 替换、二维位置编码插值 | https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/vit_jax/checkpoint.py |
| `vit_jax/train.py` | JAX/Flax 微调损失、SGD、梯度累积与数据并行 | https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/vit_jax/train.py |
| `README.md` | 官方定位、预训练仓库边界、checkpoint 和运行命令 | https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/README.md |

## 开源范围与边界

1. `google-research/vision_transformer` 是官方 JAX/Flax 实现，包含模型定义、公开 checkpoint、推理与下游微调代码。
2. 仓库 README 明确说明：这些模型最初在 `google-research/big_vision` 中训练；更完整的 multi-host 预训练和 transfer 配置在 https://github.com/google-research/big_vision 。
3. 论文最强结果依赖内部 JFT-300M，公众无法严格复现同一数据配方。公开可执行的合理口径是“加载官方 ImageNet-21k checkpoint 做迁移微调”，不能写成“完整复现 JFT 预训练”。
4. 本文代码级张量形状以 `224×224`、ViT-B/16、ImageNet-1K head 为默认实例；微调 preset 常使用 `384×384`，两者已在正文分开说明。
