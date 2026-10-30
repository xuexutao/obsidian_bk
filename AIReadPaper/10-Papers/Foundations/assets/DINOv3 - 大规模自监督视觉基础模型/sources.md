# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2508.10104 | 全文，arXiv:2508.10104 | 2026-10-15 | 精读主来源、公式与实验数据核对 |
| `figures/01-ssl-and-dense-features.png` | 论文关键图 | https://arxiv.org/pdf/2508.10104 | Figure 1, p.2 | 2026-10-15 | 说明 SSL 的定位与 DINOv3 稠密特征优势 |
| `figures/02-dense-feature-degradation.png` | 论文关键图 | https://arxiv.org/pdf/2508.10104 | Figure 6, p.11 | 2026-10-15 | 展示长训练导致 patch 局部一致性退化 |
| `figures/03-gram-anchoring-results.png` | 论文关键图 | https://arxiv.org/pdf/2508.10104 | Figure 8, p.13 | 2026-10-15 | 展示 Gram anchoring 对稠密与全局任务的恢复效果 |
| `figures/04-gram-anchoring-recovery.png` | 论文关键图 | https://arxiv.org/pdf/2508.10104 | Figure 10, p.14 | 2026-10-15 | 定性比较 Gram anchoring 前后的局部相似度图 |

## 源码、配置与权重锁定

- 官方仓库：https://github.com/facebookresearch/dinov3
- 固定 commit：`6876159a11b4df116f30f667f8c9888617df0751`
- commit 时间：2026-07-15T14:56:56+01:00
- commit 标题：`Remove unused LinearW24 (2:4 sparsity layer) (#352)`
- 取证日期：2026-10-31

### 发布配置

| 阶段 | config | 关键核验项 |
|---|---|---|
| 主预训练 | `dinov3/configs/train/dinov3_vit7b16_pretrain.yaml` | Gram off；256/112；4 storage；DINO 262144、iBOT 98304；RoPE；local CLS norm |
| Gram 修复 | `dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml` | 512 Gram crop；all-token image-level Gram；loss 0→2；每 10K 更新、最多 3 次 |
| 高分辨率适配 | `dinov3/configs/train/dinov3_vit7b16_high_res_adapt.yaml` | 五组绑定 crop 尺寸与采样比；Gram weight 1.5 |
| 蒸馏 ViT-L | `dinov3/configs/train/dinov3_vitl16_lvd1689m_distilled.yaml` | 固定 7B teacher 到 L student 的训练接口 |

### 逐模块源码

| 路径 | symbol / 用途 |
|---|---|
| `dinov3/models/vision_transformer.py` | `DinoVisionTransformer`、B/L/7B 架构、CLS/storage/patch 字段、训练专用 norm |
| `dinov3/layers/patch_embed.py` | patch stride 16；整除 assert 已注释，非整除输入 floor/drop 边缘 |
| `dinov3/layers/rope_position_encoding.py` | axial RoPE、坐标 normalize 与 log-uniform rescale |
| `dinov3/layers/attention.py` | RoPE 只作用 patch Q/K；CLS+storage prefix 不旋转 |
| `dinov3/layers/block.py` | SDPA、SwiGLU、LayerScale、sample-wise stochastic depth |
| `dinov3/data/augmentations.py` | 256 student 与 512 clean Gram teacher 从同一 global geometry crop 派生 |
| `dinov3/data/collate.py` | global/local/Gram crop-major stack 与 mask indices |
| `dinov3/train/ssl_meta_arch.py` | EMA teacher、Gram teacher、distillation teacher 的数据流；heads 与所有 shape |
| `dinov3/loss/gram_loss.py` | normalize→$XX^T$→`MSELoss`，是均值而非未归一化 Frobenius sum |
| `dinov3/train/train.py` | Gram teacher 载入与更新触发时机 |
| `dinov3/hub/backbones.py` | 发布架构、artifact 文件名、8 位 hash prefix |
| `dinov3/hub/utils.py` | 官方权重 base URL 与 `weights_only=True` 安全加载 |

### 权重 artifact / revision

官方 Hub 构造出的 LVD-1689M 文件：

- ViT-B/16：`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- ViT-L/16：`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- ViT-7B/16：`dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`

8 位后缀是代码内置 hash prefix，加载时可传 `check_hash=True`。本次没有下载大模型，因此完整 SHA-256、文件字节数未核验；笔记已明确标注。正式复现实验应记录完整哈希，不能只记录 Hugging Face `main` 或下载页的浮动引用。

### 代码事实的关键更正

- DINOv3 没有 DINOv2 式 learned absolute positional embedding / interpolation；它按动态 patch grid 在每层 attention 中应用 axial RoPE。
- 主模型固定 4 个 `storage_tokens`，即论文的 register tokens；RoPE 不作用这 4 个 token。
- Gram 主预训练默认关闭，只在修复/高分辨率阶段启用。
- 高分辨率 Gram teacher 特征先从 32×32（或更大）bicubic 到 student patch grid，再计算同 shape 的逐图 Gram。
- Gram config 为 `tokens_used=all,img_level=true,remove_neg=false`，不是只对 masked patches，也不是跨 batch Gram。
- `update_gram(m=0)` 是当前 EMA teacher 到 Gram teacher 的硬复制；修复 config 每 10K、最多 3 次。
- 高分辨率适配的 Gram weight 是 1.5；修复阶段 schedule 目标是 2.0。
- 发布 config 的 LR 数字与论文汇总口径不同，引用时必须标明来源。

## 来源说明

- 论文主页：https://arxiv.org/abs/2508.10104
- 所有本地图片均从官方 PDF 对应页面以完整图表区域裁剪，保留原始 Figure 编号和图注，不使用 PDF 内嵌位图碎片。
- 本笔记中的模型、训练、结果和复现成本数字均以本地 `sources/paper.pdf` 为准。
