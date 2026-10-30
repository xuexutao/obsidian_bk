# DINOv2 来源与图像溯源

- 整理日期：2026-10-15
- 对应笔记：`../../DINOv2 - 无监督鲁棒视觉特征.md`
- 原则：论文方法、数字和图表以官方 PDF 为最高优先级；发布历史与模型命名以 Meta 官方项目页和 `facebookresearch/dinov2` 为准；后续影响以对应工作的官方论文 / 项目页为准。

## 1. 核心一手来源

### 1.1 官方论文

- 标题：DINOv2: Learning Robust Visual Features without Supervision
- 作者：Maxime Oquab、Timothée Darcet、Théo Moutakanni、Huy V. Vo 等
- 期刊：Transactions on Machine Learning Research，2024-01
- arXiv 摘要页：https://arxiv.org/abs/2304.07193
- 官方 PDF：https://arxiv.org/pdf/2304.07193
- OpenReview：https://openreview.net/forum?id=a68SUt6zFt
- 本地文件：`sources/paper.pdf`
- 本地 PDF 页数：32
- 下载日期：2026-10-15
- SHA-256：`3add205547f3ece5fb1ac7a91720aa9fabf02d9cde29a096643ff022449f3e91`

论文内主要取证位置：

- pp.1–3：研究动机、通用特征定义、八类任务 scaling。
- pp.4–5：LVD-142M 数据来源、去重、retrieval 与 Faiss 实现。
- pp.5–6：DINO、iBOT、独立 heads、Sinkhorn-Knopp、KoLeo、高分辨率适配。
- pp.6–7：attention、sequence packing、stochastic depth、FSDP、蒸馏。
- pp.8–10：训练配方、数据、loss、蒸馏、分辨率消融。
- pp.11–16：分类、OOD、细粒度、视频、检索、分割、深度定量结果。
- pp.17–19：分割 / 深度定性结果、PCA 与 dense matching。
- pp.18–20：公平性与偏差分析。
- pp.20–21：训练能耗、项目总成本与讨论。
- pp.29–32：去重 / retrieval 细节、LVD-142M 精确构成、训练超参数、架构和数据集用途。

### 1.2 Meta 官方项目页 / 发布博客

- 标题：DINOv2: State-of-the-art computer vision models with self-supervised learning
- 发布日期：2023-04-17
- URL：https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning
- 抓取可用入口：https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning?_fb_noscript=1
- 访问日期：2026-10-15

用于核验：

- 142M 图像由约 1.2B 网页来源图像筛选而来；
- 约 25 个第三方种子数据集；
- DINOv2 不依赖标签、caption 或元数据；
- 分类、分割、检索与单目深度的 frozen-feature 定位；
- 相对 iBOT 约 2× 速度、约 1/3 显存；
- ViT-g/14 到 ViT-S/B/L 的蒸馏；
- 与 WRI 的树冠 / 森林制图应用；
- 官方 demo 与代码入口。

### 1.3 Meta 官方交互项目页

- URL：https://dinov2.metademolab.com/
- 访问日期：2026-10-15
- 用途：核验官方对 DINOv2 的任务范围描述，包括分类、实例检索、语义分割、单目深度、dense matching 与 sparse matching。

### 1.4 官方代码仓库

- 仓库：https://github.com/facebookresearch/dinov2
- 访问日期：2026-10-15
- 许可：DINOv2 主代码与主模型权重为 Apache License 2.0；仓库后续领域扩展模型可能使用不同许可，不能混同。

用于核验：

- 原始 ViT-S/B/L/g `/14` 模型与参数规模；
- ImageNet k-NN / linear 官方发布结果；
- PyTorch Hub 模型名；
- 训练、k-NN、logistic regression、linear、depth、segmentation 入口；
- README 更新日志中 registers 模型的加入日期；
- 原始模型与后续 `_reg` / `reg4` checkpoint 的命名边界。

## 2. 源码、配置与权重锁定（2026-10-31 补充）

### 2.1 仓库 revision

- 官方仓库：https://github.com/facebookresearch/dinov2
- 固定 commit：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- commit 时间：2026-06-03T17:28:49+02:00
- commit 标题：`Safely load weights from specified URLs (#598)`
- 复现原则：文中所有“代码实际行为”均以该 commit 为准；未来 `main` 的接口变化不回写为论文事实。

### 2.2 逐模块取证

| 源码路径 | 关键 symbol / 行为 | 笔记中用于核验的内容 |
|---|---|---|
| `dinov2/data/augmentations.py` | `DataAugmentationDINO` | 2 global、8 local、crop/颜色/模糊/solarize/normalize 顺序 |
| `dinov2/data/transforms.py` | `IMAGENET_DEFAULT_MEAN/STD` | mean `(0.485,0.456,0.406)`、std `(0.229,0.224,0.225)` |
| `dinov2/data/collate.py` | `collate_data_and_cast` | crop-major batch、global masks、`mask_indices_list` 与 $M$ |
| `dinov2/layers/patch_embed.py` | `PatchEmbed.forward` | `Conv2d(k=p,s=p)`，`BCHW→BN D`，输入必须整除 patch size |
| `dinov2/models/vision_transformer.py` | `prepare_tokens_with_masks`、`forward_features`、`get_intermediate_layers` | mask→CLS→pos interpolation→register→blocks→norm→输出字段的精确顺序 |
| `dinov2/models/vision_transformer.py` | `vit_base/vit_large/vit_giant2` | B/L/g 的 `D=768/1024/1536`、depth `12/24/40`、heads `12/16/24` |
| `dinov2/train/ssl_meta_arch.py` | `SSLMetaArch.forward_backward` | teacher global、student global+local、DINO/iBOT logits 与 KoLeo 的真实 shape |
| `dinov2/layers/dino_head.py` | `DINOHead` | MLP→L2 bottleneck→weight-normalized prototype layer |
| `dinov2/loss/dino_clstoken_loss.py` | `DINOLoss.sinkhorn_knopp_teacher` | teacher logits 的 3 次 SK 与跨 crop CE |
| `dinov2/loss/ibot_patch_loss.py` | `iBOTPatchLoss.forward_masked` | 只对 $M$ 个 masked patches 计算 loss |
| `dinov2/hub/backbones.py` | `_make_dinov2_model` | Hub 默认 `img_size=518`、权重 URL 组装、reg4 参数差异 |

### 2.3 固定配置

- 主配置：`dinov2/configs/train/vitg14.yaml`
  - `arch=vit_giant2`、`patch_size=14`、`drop_path_rate=0.4`、`ffn_layer=swiglufused`；
  - DINO / iBOT 独立 heads，均为 131072 prototypes、384 bottleneck；
  - 每卡 batch 12、teacher momentum 0.994、local crop size 98。
- 基础配置：`dinov2/configs/ssl_default_config.yaml`
  - 2 global + 8 local；global 224；mask sample probability 0.5；mask ratio 0.1–0.5；KoLeo 0.1。
- 重要覆盖关系：基础配置的 local size 是 96，主 `vitg14.yaml` 覆盖为 98；ViT-g/14 复现必须取合并后的 98。

### 2.4 固定权重 artifact

- ViT-B/14：https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
- ViT-L/14：https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
- ViT-g/14：https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
- 后续 register 版本对应文件名含 `reg4`，与原论文无-register artifact 分开记录。
- 本次未下载数百 MB/GB 权重，故未记录 SHA-256；这是明确的未核验项。严格复现实验应在首次下载后补记 SHA-256 和字节数。

### 2.5 代码—论文差异取证结论

- 当前代码类支持 register，不代表原论文 checkpoint 使用 register；原模型 `num_register_tokens=0`。
- `vit_base` 源码为 12 blocks，已修正旧笔记中的 18。
- `SSLMetaArch` 对两个 global CLS chunks 分别计算 KoLeo 后求和，不是只取第一个 crop。
- 当前 Hub 用 518 构造发布 backbone；训练 YAML 的 224 是主训练 crop，两者由论文 10k 高分辨率适配衔接。
- inference backbone 输出 $D$ 维 CLS/patch；131072 维 DINO/iBOT prototype logits 只属于训练 heads。

## 3. Register tokens 历史边界来源

### 2.1 后续 register 论文

- 标题：Vision Transformers Need Registers
- arXiv：https://arxiv.org/abs/2309.16588
- 访问日期：2026-10-15

### 2.2 官方仓库时间线

`facebookresearch/dinov2` README 记录：

- 2023-10-26：加入遵循 Vision Transformers Need Registers 的 DINOv2 backbones with registers；
- 每个后续模型使用 4 个 register tokens；
- Hub 名称以 `_reg` 区分；
- 权重文件常以 `reg4` 区分。

引用边界：

- arXiv:2304.07193 原始论文模型不使用 registers；
- 带 registers 的 DINOv2 是 2023-10 的后续版本；
- 本笔记中的原论文实验数字按无-register 模型解释。

## 4. 后续影响来源

### 3.1 Depth Anything

- 标题：Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data
- arXiv：https://arxiv.org/abs/2401.10891
- HTML：https://arxiv.org/html/2401.10891
- 访问日期：2026-10-15

用于核验：

- Depth Anything 使用 DINOv2 encoder 作为强初始化；
- 训练 / 推理尺寸需匹配 DINOv2 patch size 14；
- Table 11 对比原始 DINOv2 encoder 与深度专门化 encoder；
- Table 11：NYUv2 AbsRel 0.066→0.056、δ1 0.973→0.984；KITTI AbsRel 0.058→0.046、δ1 0.971→0.982；ADE20K mIoU 58.8→59.4；
- 论文明确指出大规模 MDE 训练在 DINOv2 初始化之上继续增强迁移能力。

### 3.2 Depth Anything V2

- 官方项目页：http://depth-anything-v2.github.io/
- 论文：https://arxiv.org/abs/2406.09414
- 访问日期：2026-10-15

用于核验：

- 最大 teacher 基于 DINOv2-Giant；
- 595K 合成标注图训练初始 teacher；
- 62M+ 真实无标签图由 teacher 生成伪深度，再训练 students。

### 3.3 DINO-WM

- 标题：DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
- arXiv：https://arxiv.org/abs/2411.04983
- HTML：https://arxiv.org/html/2411.04983v2
- 访问日期：2026-10-15

用于核验：

- DINO-WM 不重建 RGB，而预测未来 DINOv2 spatial patch features；
- 从离线行为轨迹训练；
- 测试时通过 action sequence optimization 达到目标图像特征；
- 在迷宫导航、桌面推物和粒子操作上评估；
- 不依赖 expert demonstration、reward model 或预训练 inverse model。

### 3.4 VGGT 与 3D 视觉

- 标题：VGGT: Visual Geometry Grounded Transformer
- arXiv：https://arxiv.org/abs/2503.11651
- PDF：https://arxiv.org/pdf/2503.11651
- 访问日期：2026-10-15

用于核验：

- VGGT 使用 DINOv2 做 image tokenization；
- patchifying 消融比较普通 14×14 convolution 与 pretrained DINOv2；
- 论文报告 DINOv2 具有更好性能、更稳定的早期训练，并降低对 learning rate / momentum 的敏感性；
- VGGT 最终把 DINOv2 作为默认 tokenizer，用于后续 camera、depth、point map 与 point track 预测。

## 5. 本地图像逐张溯源

所有图均直接从 `sources/paper.pdf` 以 PyMuPDF 300 DPI 裁取，保留完整图与原始英文 caption，并在 2026-10-15 逐张使用图像读取工具检查边界、清晰度和完整性。

### 4.1 `figures/01-general-features.png`

- 论文编号：Figure 1
- PDF 页码：p.2
- PDF 内部页索引：1（0-based）
- 裁剪区域：`Rect(60, 60, 552, 337)` points
- 来源 URL：https://arxiv.org/pdf/2304.07193
- 内容：四组跨类别 / 姿态 / 风格图像与 DINOv2 patch PCA 颜色图；用于说明对象部件对应和前景分离。
- 完整性检查：通过；四列 `(a)–(d)`、全部原图、PCA 图和 Figure 1 caption 均保留。

### 4.2 `figures/02-task-scaling.png`

- 论文编号：Figure 2
- PDF 页码：p.3
- PDF 内部页索引：2（0-based）
- 裁剪区域：`Rect(60, 60, 552, 395)` points
- 来源 URL：https://arxiv.org/pdf/2304.07193
- 内容：八类任务上 DINOv2 随模型 FLOPs 扩展的曲线，与此前 SSL / WSL 比较。
- 完整性检查：通过；8 个子图、图例、坐标轴、WSL 虚线与 Figure 2 caption 均保留。

### 4.3 `figures/03-data-curation-pipeline.png`

- 论文编号：Figure 3
- PDF 页码：p.4
- PDF 内部页索引：3（0-based）
- 裁剪区域：`Rect(60, 60, 552, 275)` points
- 来源 URL：https://arxiv.org/pdf/2304.07193
- 内容：Uncurated / Curated Data → Embedding → Deduplication → Retrieval → Augmented Curated Data。
- 完整性检查：通过；左右数据池、三阶段流程箭头与 Figure 3 caption 均保留。

### 4.4 `figures/04-distillation-ablation.png`

- 论文编号：Figure 5
- PDF 页码：p.10
- PDF 内部页索引：9（0-based）
- 裁剪区域：`Rect(60, 60, 552, 318)` points
- 来源 URL：https://arxiv.org/pdf/2304.07193
- 内容：ViT-L scratch、ViT-L distilled 与 ViT-g teacher 在 12 个指标及 8 类任务均值上的比较。
- 完整性检查：通过；雷达图、两张汇总表、图例与 Figure 5 caption 均保留。

### 4.5 `figures/05-high-resolution-ablation.png`

- 论文编号：Figure 6
- PDF 页码：p.10
- PDF 内部页索引：9（0-based）
- 裁剪区域：`Rect(60, 322, 552, 544)` points
- 来源 URL：https://arxiv.org/pdf/2304.07193
- 内容：ImageNet / ADE20K 上固定 224、固定 416 与短时 224→416 高分辨率适配的比较。
- 完整性检查：通过；两个子图、三组曲线、坐标轴、图例与 Figure 6 caption 均保留。

### 4.6 `figures/06-dense-results.png`

- 论文编号：Figure 7
- PDF 页码：p.17
- PDF 内部页索引：16（0-based）
- 裁剪区域：`Rect(60, 55, 552, 365)` points
- 来源 URL：https://arxiv.org/pdf/2304.07193
- 内容：ADE20K 分割与 NYUd / SUN RGB-D / KITTI 深度的 Input、OpenCLIP-G、DINOv2-g 定性比较。
- 完整性检查：通过；四行数据集、两组案例、全部方法列与 Figure 7 caption 均保留。

## 6. 关键数字的原文定位

### 5.1 数据

- LVD-142M 最终规模 142,109,386：Table 15，PDF p.30。
- self-dedup 1.3B→1.1B、relative dedup→744M：Appendix A.3，PDF p.29。
- 1.2B unique images 主文口径：Sec.3，PDF p.5。
- 20×8 V100、少于两天：Sec.3 Implementation Details，PDF p.5。

### 5.2 目标与训练

- DINO / iBOT：Sec.4，PDF pp.5–6。
- 独立 heads、SK 3 iterations、KoLeo：Sec.4，PDF p.6。
- KoLeo 权重 0.1：Appendix B.1，PDF p.29。
- 高分辨率 518×518：Sec.4，PDF p.6。
- 10k 高分辨率适配：Appendix B.2，PDF p.31。
- 625k iterations、优化器与 schedules：Table 16，PDF p.31。
- 模型维度 / 层数 / FFN：Table 17，PDF p.31。

### 5.3 消融

- iBOT→DINOv2 累积配方：Table 1，PDF p.8。
- 数据源消融：Table 2，PDF p.8。
- KoLeo / MIM 消融：Table 3，PDF p.9。
- 蒸馏：Figure 5，PDF p.10。
- 分辨率：Figure 6，PDF p.10。

### 5.4 结果

- ImageNet frozen linear：Table 4，PDF p.12。
- OOD domain generalization：Table 6，PDF p.13。
- 其他图像 / 视频分类：Table 7，PDF p.13。
- 12 个迁移分类：Table 8，PDF p.14。
- 实例检索：Table 9，PDF pp.14–15。
- 语义分割：Table 10，PDF p.15。
- 单目深度：Table 11，PDF p.16。
- dense qualitative：Figures 7–10，PDF pp.17–19。

### 5.5 成本与偏差

- Dollar Street：Table 12，PDF p.19。
- harmful association：Table 13，PDF p.20。
- 单次 DINOv2-g 能耗 / 碳排：Table 14，PDF p.20。
- 项目总计 200k GPU-days、0.5k–1k tCO2eq：Sec.9，PDF p.21。

## 7. 口径说明

- “无监督 / 自监督”指预训练不使用人工类别标签或 paired text；不表示下游 linear / DPT / Mask2Former 头不使用标注。
- “无需微调”在论文中主要指 backbone 冻结，不代表完全无需下游训练。
- “zero-shot depth transfer”是 NYUd 上训练深度头后直接迁移 SUN RGB-D，而不是不使用任何深度监督。
- LVD-142M 使用多个下游训练集做 retrieval seeds；论文同时做 test / validation relative dedup，并用未参与 curation 的数据集测试外域迁移。
- 原始 DINOv2 论文无 registers；仓库后来的 `_reg` 模型不可倒写为原论文组件。
- 本笔记所有 DINOv2 原论文数字以 2024-02-02 的 arXiv v2 / TMLR PDF 为准。
